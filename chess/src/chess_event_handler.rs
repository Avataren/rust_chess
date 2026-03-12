use bevy::prelude::*;
use bevy::ecs::message::{MessageReader, MessageWriter};
use bevy_async_task::TaskRunner;
use bevy_tweening::{lens::TransformPositionLens, Delay, *};
use std::task::Poll;
use chess_board::ChessBoard;
#[cfg(target_arch = "wasm32")]
use chess_evaluation::{search_root, OpeningBook, SearchContext, TranspositionTable, ASPIRATION_DELTA, TT_SIZE};
#[cfg(not(target_arch = "wasm32"))]
use chess_evaluation::{iterative_deepening_root, OpeningBook};
use chess_foundation::ChessMove;
use move_generator::{
    move_generator::get_all_legal_moves_for_color, piece_conductor::PieceConductor,
};
use std::time::Duration;

use crate::{
    board::BoardDimensions,
    game_events::{ChessAction, ChessEvent, RefreshPiecesFromBoardEvent},
    game_resources::{CurrentOpening, Difficulty, GameOverState, GamePhase, IsAiThinking, LastMove, OpeningBookRes, PendingGameOver, PendingMoveSound, PlayerColor},
    pieces::ChessPieceComponent,
    sound::{spawn_sound, SoundEffects},
    ChessBoardRes, PieceConductorRes,
};

// Function to get local position from board coordinates (col, row)
fn get_local_position_from_board_coords(
    col: u16,
    row: u16,
    board_dimensions: &BoardDimensions,
) -> Vec3 {
    Vec3::new(
        col as f32 * board_dimensions.square_size - board_dimensions.board_size.x / 2.0
            + board_dimensions.square_size / 2.0,
        row as f32 * board_dimensions.square_size - board_dimensions.board_size.y / 2.0
            + board_dimensions.square_size / 2.0,
        1.5,
    )
}

pub fn on_tween_completed(
    mut tween_completed_events: MessageReader<AnimCompletedEvent>,
    mut refresh_pieces_events: MessageWriter<RefreshPiecesFromBoardEvent>,
    mut pending_game_over: ResMut<PendingGameOver>,
    mut game_over_state: ResMut<GameOverState>,
    mut commands: Commands,
    sound_effects: Res<SoundEffects>,
    mut pending_move_sound: ResMut<PendingMoveSound>,
) {
    for _ in tween_completed_events.read() {
        if let Some(sound) = pending_move_sound.0.take() {
            spawn_sound(&mut commands, &sound_effects, sound);
        }
        refresh_pieces_events.write(RefreshPiecesFromBoardEvent);
        if let Some(state) = pending_game_over.0.take() {
            *game_over_state = state;
        }
        break;
    }
}

async fn alpha_beta_task(
    chess_board: &mut ChessBoard,
    conductor: &PieceConductor,
    book: &OpeningBook,
    max_depth: i32,
    is_white: bool,
    deadline: Option<web_time::Instant>,
) -> (i32, Option<ChessMove>) {
    // On WASM we reimplement the iterative-deepening loop here so we can yield
    // between depth iterations, letting the browser paint frames (animated
    // "Thinking..." indicator).  On native, the search runs on a background
    // thread so we just call iterative_deepening_root directly.
    #[cfg(not(target_arch = "wasm32"))]
    {
        let result = iterative_deepening_root(chess_board, conductor, Some(book), max_depth, is_white, deadline, None);
        return (result.score, result.best_move);
    }

    #[cfg(target_arch = "wasm32")]
    {
        use move_generator::move_generator::get_all_legal_moves_for_color;

        // Yield once so the browser paints the piece landing before we start.
        gloo_timers::future::TimeoutFuture::new(0).await;

        // Book probe
        if let Some((from, to)) = book.probe(chess_board) {
            let legal = get_all_legal_moves_for_color(chess_board, conductor, is_white);
            if let Some(book_move) = legal.into_iter().find(|m| m.start_square() == from && m.target_square() == to) {
                return (0, Some(book_move));
            }
        }

        let mut tt = TranspositionTable::new(TT_SIZE);
        let mut ctx = SearchContext::new();
        let mut best: (i32, Option<ChessMove>) = (if is_white { i32::MIN + 1 } else { i32::MAX }, None);

        for depth in 1..=max_depth {
            let (prev_score, prev_move) = best;

            best = if depth <= 2 {
                search_root(chess_board, conductor, &mut tt, &mut ctx, depth, i32::MIN + 1, i32::MAX, is_white, prev_move)
            } else {
                let mut lo = prev_score.saturating_sub(ASPIRATION_DELTA);
                let mut hi = prev_score.saturating_add(ASPIRATION_DELTA);
                loop {
                    let result = search_root(chess_board, conductor, &mut tt, &mut ctx, depth, lo, hi, is_white, prev_move);
                    if result.0 > lo && result.0 < hi {
                        break result;
                    } else if result.0 <= lo {
                        lo = i32::MIN + 1;
                    } else {
                        hi = i32::MAX;
                    }
                    if lo == i32::MIN + 1 && hi == i32::MAX {
                        break result;
                    }
                }
            };

            if let Some(dl) = deadline {
                if web_time::Instant::now() >= dl { break; }
            }

            // Yield between depths so the browser can paint a frame.
            gloo_timers::future::TimeoutFuture::new(0).await;
        }

        best
    }
}

pub fn handle_async_moves(
    mut task_executor: TaskRunner<(i32, Option<ChessMove>)>,
    mut commands: Commands,
    mut chess_board: ResMut<ChessBoardRes>,
    board_dimensions: Res<BoardDimensions>,
    move_generator: ResMut<PieceConductorRes>,
    mut piece_query: Query<(Entity, &mut Transform, &mut ChessPieceComponent)>,
    mut chess_ew: MessageReader<ChessEvent>,
    mut last_move: ResMut<LastMove>,
    mut game_over_state: ResMut<GameOverState>,
    mut pending_game_over: ResMut<PendingGameOver>,
    player_color: Res<PlayerColor>,
    game_phase: Res<GamePhase>,
    opening_book: Res<OpeningBookRes>,
    mut is_ai_thinking: ResMut<IsAiThinking>,
    difficulty: Res<Difficulty>,
    mut pending_move_sound: ResMut<PendingMoveSound>,
) {
    if task_executor.is_idle() {
        // Task is idle — check for new chess events to start a task
        for event in chess_ew.read() {
            if let ChessAction::MakeMove = event.action {
                if *game_over_state != GameOverState::Playing || *game_phase != GamePhase::Playing {
                    break;
                }
                let ai_is_white = *player_color == PlayerColor::Black;

                // If only one legal move, return it immediately (no search needed).
                let forced = {
                    let moves = get_all_legal_moves_for_color(
                        &mut chess_board.chess_board,
                        &move_generator.magic,
                        ai_is_white,
                    );
                    if moves.len() == 1 { Some(moves[0]) } else { None }
                };

                let mut chess_board_clone = chess_board.chess_board.clone();
                let move_generator_clone = move_generator.magic.clone();
                let book_clone = opening_book.book.clone();
                let depth = difficulty.search_depth();
                let deadline = difficulty.time_limit().map(|d| web_time::Instant::now() + d);
                task_executor.start(async move {
                    if let Some(m) = forced {
                        return (0, Some(m));
                    }
                    alpha_beta_task(
                        &mut chess_board_clone,
                        &move_generator_clone,
                        &book_clone,
                        depth,
                        ai_is_white,
                        deadline,
                    )
                    .await
                });
                is_ai_thinking.0 = true;
                break;
            }
        }
        return;
    }

    // Drain events while task is running to prevent buildup
    for _ in chess_ew.read() {}

    match task_executor.poll() {
        Poll::Pending => {
            // println!("Alpha-beta computation in progress...");
        }
        Poll::Ready((score, mut best_move)) => {
            is_ai_thinking.0 = false;
            println!("Received score {score}");
            let ai_is_white = *player_color == PlayerColor::Black;
            let player_is_white = !ai_is_white;

            if best_move.is_none() {
                let all_moves = get_all_legal_moves_for_color(
                    &mut chess_board.chess_board,
                    &move_generator.magic,
                    ai_is_white,
                );
                if all_moves.is_empty() {
                    // AI has no moves: player wins or stalemate (no piece animation needed)
                    if move_generator.magic.is_king_in_check(&chess_board.chess_board, ai_is_white) {
                        println!("Checkmate — player wins!");
                        *game_over_state = GameOverState::PlayerWins;
                    } else {
                        println!("Stalemate!");
                        *game_over_state = GameOverState::Stalemate;
                    }
                    return;
                } else {
                    best_move = Some(all_moves[0]);
                    println!("Random move");
                }
            }
            let mut engine_move = best_move.unwrap();

            if chess_board.chess_board.make_move(&mut engine_move) {
                last_move.start_square = Some(engine_move.start_square());
                last_move.target_square = Some(engine_move.target_square());

                let sound = if engine_move.has_flag(ChessMove::CASTLE_FLAG) {
                    "castle.ogg"
                } else if engine_move.capture.is_some() {
                    "capture.ogg"
                } else if move_generator.magic.is_king_in_check(&chess_board.chess_board, player_is_white) {
                    "move-check.ogg"
                } else {
                    "notify.ogg"
                };
                pending_move_sound.0 = Some(sound);

                // Check if the player has any legal moves after AI's move.
                // Store result as pending — the overlay will appear after the tween finishes.
                let player_moves = get_all_legal_moves_for_color(
                    &mut chess_board.chess_board,
                    &move_generator.magic,
                    player_is_white,
                );
                if chess_board.chess_board.is_repetition(3) {
                    println!("Draw by repetition!");
                    pending_game_over.0 = Some(GameOverState::Draw);
                } else if player_moves.is_empty() {
                    let outcome = if move_generator.magic.is_king_in_check(&chess_board.chess_board, player_is_white) {
                        println!("Checkmate — opponent wins!");
                        GameOverState::OpponentWins
                    } else {
                        println!("Stalemate!");
                        GameOverState::Stalemate
                    };
                    pending_game_over.0 = Some(outcome);
                }

                let start_local_position = get_local_position_from_board_coords(
                    engine_move.start_square() % 8,
                    engine_move.start_square() / 8,
                    &board_dimensions,
                );
                let end_local_position = get_local_position_from_board_coords(
                    engine_move.target_square() % 8,
                    engine_move.target_square() / 8,
                    &board_dimensions,
                );

                if let Some((entity, _, _)) = piece_query.iter_mut().find(|(_, _, chess_piece)| {
                    chess_piece.col == (engine_move.start_square() % 8) as usize
                        && chess_piece.row == 7 - (engine_move.start_square() / 8) as usize
                }) {
                    let tween = Tween::new(
                        EaseFunction::CubicOut,
                        Duration::from_millis(300),
                        TransformPositionLens {
                            start: start_local_position,
                            end: end_local_position,
                        },
                    );
                    let sequence = Delay::new(Duration::from_millis(150)).then(tween);

                    commands.entity(entity).insert(TweenAnim::new(sequence));
                } else {
                    println!("No piece found at start square");
                }
            }
        }
    }
}

pub fn handle_chess_events(
    mut chess_ew: MessageReader<ChessEvent>,
    mut chess_board: ResMut<ChessBoardRes>,
    mut refresh_pieces_events: MessageWriter<RefreshPiecesFromBoardEvent>,
    mut game_over_state: ResMut<GameOverState>,
    mut last_move: ResMut<LastMove>,
    mut pending_game_over: ResMut<PendingGameOver>,
    mut game_phase: ResMut<GamePhase>,
    mut current_opening: ResMut<CurrentOpening>,
) {
    for event in chess_ew.read() {
        match event.action {
            ChessAction::Undo => {
                println!("Undoing move");
                chess_board.chess_board.undo_move();
                refresh_pieces_events.write(RefreshPiecesFromBoardEvent);
            }
            ChessAction::Restart => {
                println!("Returning to start screen");
                chess_board.chess_board = chess_board::ChessBoard::new();
                *game_over_state = GameOverState::Playing;
                pending_game_over.0 = None;
                *last_move = LastMove::default();
                *game_phase = GamePhase::StartScreen;
                current_opening.0.clear();
                refresh_pieces_events.write(RefreshPiecesFromBoardEvent);
            }
            _ => {}
        }
    }
}

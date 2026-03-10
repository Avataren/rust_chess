use bevy::prelude::*;
use bevy::ecs::message::{MessageReader, MessageWriter};
use bevy_async_task::TaskRunner;
use bevy_tweening::{lens::TransformPositionLens, *};
use std::task::Poll;
use chess_board::ChessBoard;
use chess_evaluation::{iterative_deepening_root, OpeningBook};
use chess_foundation::ChessMove;
use move_generator::{
    move_generator::get_all_legal_moves_for_color, piece_conductor::PieceConductor,
};
use std::time::Duration;

use crate::{
    board::BoardDimensions,
    game_events::{ChessAction, ChessEvent, RefreshPiecesFromBoardEvent},
    game_resources::{CurrentOpening, Difficulty, GameOverState, GamePhase, IsAiThinking, LastMove, OpeningBookRes, PendingGameOver, PlayerColor},
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
) {
    for _ in tween_completed_events.read() {
        println!("Tween completed");
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
    depth: i32,
    is_white: bool,
) -> (i32, Option<ChessMove>) {
    iterative_deepening_root(chess_board, conductor, Some(book), depth, is_white)
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
    sound_effects: Res<SoundEffects>,
    difficulty: Res<Difficulty>,
) {
    if task_executor.is_idle() {
        // Task is idle — check for new chess events to start a task
        for event in chess_ew.read() {
            if let ChessAction::MakeMove = event.action {
                if *game_over_state != GameOverState::Playing || *game_phase != GamePhase::Playing {
                    break;
                }
                let ai_is_white = *player_color == PlayerColor::Black;
                let mut chess_board_clone = chess_board.chess_board.clone();
                let move_generator_clone = move_generator.magic.clone();
                let book_clone = opening_book.book.clone();
                let depth = difficulty.search_depth();
                task_executor.start(async move {
                    alpha_beta_task(
                        &mut chess_board_clone,
                        &move_generator_clone,
                        &book_clone,
                        depth,
                        ai_is_white,
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
                spawn_sound(&mut commands, &sound_effects, sound);

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
                        Duration::from_millis(250),
                        TransformPositionLens {
                            start: start_local_position,
                            end: end_local_position,
                        },
                    );

                    commands.entity(entity).insert(TweenAnim::new(tween));
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

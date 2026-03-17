use bevy::prelude::*;
use bevy::ecs::message::{MessageReader, MessageWriter};
use bevy_async_task::TaskPool;
use bevy_tweening::{lens::TransformPositionLens, Delay, *};
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
#[cfg(target_arch = "wasm32")]
use std::sync::Mutex;
use std::task::Poll;
use chess_board::ChessBoard;
use chess_evaluation::{evaluate_board, iterative_deepening_root, OpeningBook};
use chess_foundation::ChessMove;
use move_generator::{
    move_generator::get_all_legal_moves_for_color, piece_conductor::PieceConductor,
};
use rand::Rng;
use std::time::Duration;

use crate::{
    board::BoardDimensions,
    game_events::{AiMoveAnimEvent, ChessAction, ChessEvent, RefreshPiecesFromBoardEvent},
    game_resources::{
        CurrentOpening, GameClocks, GameOverState, GamePhase, GameSettings, IsAiThinking, LastMove,
        OpeningBookRes, PendingGameOver, PendingMoveSound, PlayerColor, PonderState,
    },
    pieces::ChessPieceComponent,
    sound::{spawn_sound, SoundEffects},
    ChessBoardRes, PieceConductorRes,
};

// ── Constants ─────────────────────────────────────────────────────────────────

/// How many opponent candidate positions to ponder simultaneously.
#[cfg(not(target_arch = "wasm32"))]
const PONDER_CANDIDATES: usize = 4;
#[cfg(target_arch = "wasm32")]
const PONDER_CANDIDATES: usize = 2;

// ── Position helpers ──────────────────────────────────────────────────────────

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

// ── Tween completion ──────────────────────────────────────────────────────────

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

// ── AI piece tween — separate system to stay within Bevy's param limit ───────

pub fn apply_ai_move_animation(
    mut anim_events: MessageReader<AiMoveAnimEvent>,
    mut piece_query: Query<(Entity, &mut Transform, &mut ChessPieceComponent)>,
    mut commands: Commands,
    board_dimensions: Res<BoardDimensions>,
) {
    for ev in anim_events.read() {
        let engine_move = ev.engine_move;
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

        if let Some((entity, _, _)) = piece_query.iter_mut().find(|(_, _, cp)| {
            cp.col == (engine_move.start_square() % 8) as usize
                && cp.row == 7 - (engine_move.start_square() / 8) as usize
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
            eprintln!("No piece found at start square");
        }
        break;
    }
}

// ── Search task output type ───────────────────────────────────────────────────

pub type SearchTaskOutput = (i32, Option<ChessMove>, Option<ChessMove>, bool, u64);
//                          score  best_move  ponder_move  is_ponder  board_hash

// ── Eval noise helper ─────────────────────────────────────────────────────────

/// After the main search, re-score all legal root moves with a fast 1-ply
/// static evaluation and add uniform random noise.  The move with the best
/// noisy score is returned.  When `noise_cp == 0` the original `best_move` is
/// returned unchanged.
fn apply_noise_move(
    chess_board: &ChessBoard,
    conductor: &PieceConductor,
    best_move: Option<ChessMove>,
    is_white: bool,
    noise_cp: i32,
) -> Option<ChessMove> {
    if noise_cp == 0 {
        return best_move;
    }
    let legal = get_all_legal_moves_for_color(&mut chess_board.clone(), conductor, is_white);
    if legal.len() <= 1 {
        return best_move.or_else(|| legal.into_iter().next());
    }
    let mut rng = rand::thread_rng();
    let scored = legal.into_iter().map(|mut m| {
        let mut b = chess_board.clone();
        b.make_move(&mut m);
        let eval = evaluate_board(&b, conductor);
        let noisy = eval + rng.gen_range(-noise_cp..=noise_cp);
        (m, noisy)
    });
    if is_white {
        scored.max_by_key(|(_, s)| *s).map(|(m, _)| m)
    } else {
        scored.min_by_key(|(_, s)| *s).map(|(m, _)| m)
    }
}

// ── Main search task ──────────────────────────────────────────────────────────

async fn alpha_beta_task(
    chess_board: &mut ChessBoard,
    conductor: &PieceConductor,
    book: &OpeningBook,
    max_depth: i32,
    is_white: bool,
    deadline: Option<web_time::Instant>,
) -> SearchTaskOutput {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let result = iterative_deepening_root(
            chess_board, conductor, Some(book), max_depth, is_white, deadline, None,
        );
        return (result.score, result.best_move, result.ponder_move, false, 0);
    }

    #[cfg(target_arch = "wasm32")]
    {
        // Opening book fast path (no worker needed).
        if let Some((from, to)) = book.probe(chess_board) {
            let legal = get_all_legal_moves_for_color(chess_board, conductor, is_white);
            if let Some(bm) = legal
                .into_iter()
                .find(|m| m.start_square() == from && m.target_square() == to)
            {
                return (0, Some(bm), None, false, 0);
            }
        }

        let stop = Arc::new(AtomicBool::new(false));
        let result: Arc<Mutex<Option<(i32, Option<ChessMove>, Option<ChessMove>)>>> =
            Arc::new(Mutex::new(None));
        {
            let stop_w = Arc::clone(&stop);
            let res_w = Arc::clone(&result);
            let mut board_w = chess_board.clone();
            let conductor_w = conductor.clone();
            let book_w = book.clone();
            rayon::spawn(move || {
                let r = iterative_deepening_root(
                    &mut board_w, &conductor_w, Some(&book_w),
                    max_depth, is_white, None, Some(stop_w),
                );
                *res_w.lock().unwrap() = Some((r.score, r.best_move, r.ponder_move));
            });
        }
        loop {
            gloo_timers::future::TimeoutFuture::new(5).await;
            if let Some(dl) = deadline {
                if web_time::Instant::now() >= dl {
                    stop.store(true, Ordering::Relaxed);
                }
            }
            if let Some((score, bm, pm)) = result.lock().unwrap().take() {
                return (score, bm, pm, false, 0);
            }
        }
    }
}

// ── Ponder search task ────────────────────────────────────────────────────────

async fn ponder_search_task(
    mut board: ChessBoard,
    conductor: PieceConductor,
    book: OpeningBook,
    depth: i32,
    is_white: bool,
    stop: Arc<AtomicBool>,
    board_hash: u64,
) -> SearchTaskOutput {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let result = iterative_deepening_root(
            &mut board, &conductor, Some(&book),
            depth, is_white, None, Some(stop),
        );
        return (result.score, result.best_move, result.ponder_move, true, board_hash);
    }

    #[cfg(target_arch = "wasm32")]
    {
        let result: Arc<Mutex<Option<(i32, Option<ChessMove>, Option<ChessMove>)>>> =
            Arc::new(Mutex::new(None));
        {
            let stop_w = Arc::clone(&stop);
            let res_w = Arc::clone(&result);
            let conductor_w = conductor.clone();
            let book_w = book.clone();
            rayon::spawn(move || {
                let r = iterative_deepening_root(
                    &mut board, &conductor_w, Some(&book_w),
                    depth, is_white, None, Some(stop_w),
                );
                *res_w.lock().unwrap() = Some((r.score, r.best_move, r.ponder_move));
            });
        }
        loop {
            gloo_timers::future::TimeoutFuture::new(5).await;
            if let Some((score, bm, pm)) = result.lock().unwrap().take() {
                return (score, bm, pm, true, board_hash);
            }
        }
    }
}

// ── Multi-ponder launch ───────────────────────────────────────────────────────

fn launch_multi_ponder(
    task_pool: &mut TaskPool<SearchTaskOutput>,
    ponder_state: &mut PonderState,
    board: &ChessBoard,
    conductor: &PieceConductor,
    book: &OpeningBook,
    depth: i32,
    ai_is_white: bool,
) {
    for s in &ponder_state.stops {
        s.store(true, Ordering::Relaxed);
    }
    task_pool.forget_all();
    ponder_state.stops.clear();
    ponder_state.results.clear();
    ponder_state.ponder_active = false;

    let opponent_is_white = !ai_is_white;
    let mut board_for_gen = board.clone();
    let legal = get_all_legal_moves_for_color(&mut board_for_gen, conductor, opponent_is_white);
    if legal.is_empty() {
        return;
    }

    let mut scored: Vec<(ChessMove, i32)> = legal
        .into_iter()
        .map(|mut m| {
            let mut b = board.clone();
            b.make_move(&mut m);
            let eval = evaluate_board(&b, conductor);
            let opp_score = if opponent_is_white { eval } else { -eval };
            (m, opp_score)
        })
        .collect();
    scored.sort_by(|a, b| b.1.cmp(&a.1));
    scored.truncate(PONDER_CANDIDATES);

    for (opp_mv, _) in scored {
        let mut ponder_board = board.clone();
        let mut mv_c = opp_mv;
        if !ponder_board.make_move(&mut mv_c) {
            continue;
        }
        let board_hash = ponder_board.current_hash();
        let stop = Arc::new(AtomicBool::new(false));
        ponder_state.stops.push(Arc::clone(&stop));

        let conductor_c = conductor.clone();
        let book_c = book.clone();
        task_pool.spawn(async move {
            ponder_search_task(
                ponder_board, conductor_c, book_c, depth, ai_is_white, stop, board_hash,
            )
            .await
        });
        eprintln!("Ponder candidate hash={board_hash:#x}");
    }

    if !ponder_state.stops.is_empty() {
        ponder_state.ponder_active = true;
    }
}

// ── Main async-move handler ───────────────────────────────────────────────────
// Exactly 16 system parameters (Bevy's limit).

pub fn handle_async_moves(
    mut task_pool: TaskPool<SearchTaskOutput>,         // 1
    mut chess_board: ResMut<ChessBoardRes>,            // 2
    move_generator: ResMut<PieceConductorRes>,         // 3
    mut chess_ew: MessageReader<ChessEvent>,           // 4
    mut anim_ew: MessageWriter<AiMoveAnimEvent>,       // 5
    mut last_move: ResMut<LastMove>,                   // 6
    mut game_over_state: ResMut<GameOverState>,        // 7
    mut pending_game_over: ResMut<PendingGameOver>,    // 8
    player_color: Res<PlayerColor>,                    // 9
    game_phase: Res<GamePhase>,                        // 10
    opening_book: Res<OpeningBookRes>,                 // 11
    mut is_ai_thinking: ResMut<IsAiThinking>,          // 12
    game_settings: Res<GameSettings>,                  // 13
    mut pending_move_sound: ResMut<PendingMoveSound>,  // 14
    mut ponder_state: ResMut<PonderState>,             // 15
    mut game_clocks: ResMut<GameClocks>,               // 16
) {
    let ai_is_white = *player_color == PlayerColor::Black;
    let player_is_white = !ai_is_white;

    // ── 1. Poll pool — route results ─────────────────────────────────────────
    let mut pending: Option<SearchTaskOutput> = None;
    for poll_result in task_pool.iter_poll() {
        if let Poll::Ready((score, bm, pm, is_ponder, board_hash)) = poll_result {
            if is_ponder {
                if ponder_state.ponder_active {
                    ponder_state.results.insert(board_hash, (score, bm, pm));
                    eprintln!("Ponder done hash={board_hash:#x} score={score}");
                }
            } else if ponder_state.main_search_active {
                ponder_state.main_search_active = false;
                is_ai_thinking.0 = false;
                pending = Some((score, bm, pm, false, 0));
            }
        }
    }

    // ── 2. If main search is in flight, drain events and wait ────────────────
    if ponder_state.main_search_active {
        for _ in chess_ew.read() {}
        return;
    }

    // ── 3. Process player move events ────────────────────────────────────────
    if pending.is_none() {
        for event in chess_ew.read() {
            if let ChessAction::MakeMove = event.action {
                if *game_over_state != GameOverState::Playing
                    || *game_phase != GamePhase::Playing
                {
                    break;
                }

                for s in &ponder_state.stops {
                    s.store(true, Ordering::Relaxed);
                }
                task_pool.forget_all();
                ponder_state.stops.clear();
                ponder_state.ponder_active = false;

                let current_hash = chess_board.chess_board.current_hash();
                let ponder_hit = ponder_state.results.remove(&current_hash);
                ponder_state.results.clear();

                if let Some((score, bm, pm)) = ponder_hit {
                    eprintln!("Ponder hit! hash={current_hash:#x} score={score}");
                    pending = Some((score, bm, pm, false, 0));
                } else {
                    let forced = {
                        let moves = get_all_legal_moves_for_color(
                            &mut chess_board.chess_board,
                            &move_generator.magic,
                            ai_is_white,
                        );
                        if moves.len() == 1 { Some(moves[0]) } else { None }
                    };

                    let mut board_c = chess_board.chess_board.clone();
                    let conductor_c = move_generator.magic.clone();
                    let book_c = opening_book.book.clone();
                    let depth = game_settings.strength.max_depth();
                    let budget = game_clocks.move_budget(ai_is_white);
                    let deadline = Some(web_time::Instant::now() + budget);

                    task_pool.spawn(async move {
                        if let Some(m) = forced {
                            return (0, Some(m), None, false, 0);
                        }
                        alpha_beta_task(
                            &mut board_c, &conductor_c, &book_c,
                            depth, ai_is_white, deadline,
                        )
                        .await
                    });
                    ponder_state.main_search_active = true;
                    is_ai_thinking.0 = true;
                }
                break;
            }
        }

        if pending.is_none() {
            return;
        }
    }

    // ── 4. Apply AI move ──────────────────────────────────────────────────────
    let (score, mut best_move, _ponder_move, _, _) = pending.unwrap();
    eprintln!("AI result: score={score}");

    // Apply eval noise — perturbs root-move selection without affecting search.
    let noise_cp = game_settings.strength.eval_noise_cp();
    if noise_cp > 0 {
        best_move = apply_noise_move(
            &chess_board.chess_board, &move_generator.magic,
            best_move, ai_is_white, noise_cp,
        );
    }

    if best_move.is_none() {
        let all_moves = get_all_legal_moves_for_color(
            &mut chess_board.chess_board, &move_generator.magic, ai_is_white,
        );
        if all_moves.is_empty() {
            *game_over_state =
                if move_generator.magic.is_king_in_check(&chess_board.chess_board, ai_is_white) {
                    eprintln!("Checkmate — player wins!");
                    GameOverState::PlayerWins
                } else {
                    eprintln!("Stalemate!");
                    GameOverState::Stalemate
                };
            return;
        } else {
            best_move = Some(all_moves[0]);
        }
    }
    let mut engine_move = best_move.unwrap();

    if chess_board.chess_board.make_move(&mut engine_move) {
        last_move.start_square = Some(engine_move.start_square());
        last_move.target_square = Some(engine_move.target_square());

        // Add increment after the AI's move.
        game_clocks.add_increment(ai_is_white);

        let sound = if engine_move.has_flag(ChessMove::CASTLE_FLAG) {
            "castle.ogg"
        } else if engine_move.capture.is_some() {
            "capture.ogg"
        } else if move_generator
            .magic
            .is_king_in_check(&chess_board.chess_board, player_is_white)
        {
            "move-check.ogg"
        } else {
            "notify.ogg"
        };
        pending_move_sound.0 = Some(sound);

        let player_moves = get_all_legal_moves_for_color(
            &mut chess_board.chess_board, &move_generator.magic, player_is_white,
        );
        if chess_board.chess_board.is_repetition(3) {
            eprintln!("Draw by repetition!");
            pending_game_over.0 = Some(GameOverState::Draw);
        } else if player_moves.is_empty() {
            let outcome =
                if move_generator.magic.is_king_in_check(&chess_board.chess_board, player_is_white) {
                    eprintln!("Checkmate — opponent wins!");
                    GameOverState::OpponentWins
                } else {
                    eprintln!("Stalemate!");
                    GameOverState::Stalemate
                };
            pending_game_over.0 = Some(outcome);
        }

        anim_ew.write(AiMoveAnimEvent { engine_move });

        // Launch multi-ponder for the next half-move.
        if game_settings.strength.ponders()
            && pending_game_over.0.is_none()
            && *game_over_state == GameOverState::Playing
        {
            launch_multi_ponder(
                &mut task_pool,
                &mut ponder_state,
                &chess_board.chess_board,
                &move_generator.magic,
                &opening_book.book,
                game_settings.strength.max_depth(),
                ai_is_white,
            );
        }
    }
}

// ── Non-search chess event handler ───────────────────────────────────────────

pub fn handle_chess_events(
    mut chess_ew: MessageReader<ChessEvent>,
    mut chess_board: ResMut<ChessBoardRes>,
    mut refresh_pieces_events: MessageWriter<RefreshPiecesFromBoardEvent>,
    mut game_over_state: ResMut<GameOverState>,
    mut last_move: ResMut<LastMove>,
    mut pending_game_over: ResMut<PendingGameOver>,
    mut game_phase: ResMut<GamePhase>,
    mut current_opening: ResMut<CurrentOpening>,
    mut ponder_state: ResMut<PonderState>,
    mut game_clocks: ResMut<GameClocks>,
    player_color: Res<PlayerColor>,
) {
    let player_is_white = *player_color == PlayerColor::White;
    for event in chess_ew.read() {
        match event.action {
            ChessAction::MakeMove => {
                // Add increment for the human player's move.
                game_clocks.add_increment(player_is_white);
            }
            ChessAction::Undo => {
                eprintln!("Undoing move");
                for s in &ponder_state.stops {
                    s.store(true, Ordering::Relaxed);
                }
                ponder_state.stops.clear();
                ponder_state.results.clear();
                ponder_state.ponder_active = false;
                ponder_state.main_search_active = false;
                chess_board.chess_board.undo_move();
                refresh_pieces_events.write(RefreshPiecesFromBoardEvent);
            }
            ChessAction::Restart => {
                eprintln!("Returning to start screen");
                for s in &ponder_state.stops {
                    s.store(true, Ordering::Relaxed);
                }
                ponder_state.stops.clear();
                ponder_state.results.clear();
                ponder_state.ponder_active = false;
                ponder_state.main_search_active = false;
                chess_board.chess_board = chess_board::ChessBoard::new();
                *game_over_state = GameOverState::Playing;
                pending_game_over.0 = None;
                *last_move = LastMove::default();
                *game_phase = GamePhase::StartScreen;
                current_opening.0.clear();
                game_clocks.reset();
                refresh_pieces_events.write(RefreshPiecesFromBoardEvent);
            }
        }
    }
}

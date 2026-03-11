use chess_board::ChessBoard;
use chess_foundation::ChessMove;
use move_generator::{
    move_generator::get_all_legal_moves_for_color, piece_conductor::PieceConductor,
};
use rand::seq::SliceRandom;
use rayon::prelude::*;
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use web_time::Instant;

use crate::{
    evaluate_board,
    opening_book::OpeningBook,
    transposition_table::{TranspositionTable, TtFlag},
};

/// Shared TT for the sequential path and ID iterations.
pub const TT_SIZE: usize = 1 << 20; // 1M entries ≈ 32 MB

/// Per-thread TT for parallel root search.  Sub-trees from a single root move
/// rarely exceed ~500K unique positions, so 256K entries suffices while keeping
/// peak memory at ≈ 8 MB × number of Rayon threads.
const PARALLEL_TT_SIZE: usize = 1 << 18; // 256K entries ≈ 8 MB

/// Initial aspiration window half-width in centipawns.  Searches at depth N
/// use [prev_score - DELTA, prev_score + DELTA]; on failure one side widens to
/// the full bound and we retry.
pub const ASPIRATION_DELTA: i32 = 50;

/// Maximum ply depth tracked by the search context.
pub const MAX_PLY: usize = 64;

// ── Search context (killers + history) ───────────────────────────────────────

/// Per-search state for move ordering heuristics.
///
/// * `killers` — up to 2 quiet moves per ply that caused a β-cutoff.  Tried
///   right after captures, before other quiet moves.
/// * `history` — `history[from][to]` accumulates `depth²` every time the
///   quiet move (from→to) causes a β-cutoff.  Used to rank the remaining
///   quiet moves.
pub struct SearchContext {
    killers: [[Option<ChessMove>; 2]; MAX_PLY],
    history: [[i32; 64]; 64],
    nodes: u64,
    stop: Option<Arc<AtomicBool>>,
    deadline: Option<Instant>,
}

impl SearchContext {
    pub fn new() -> Self {
        Self {
            killers: [[None; 2]; MAX_PLY],
            history: [[0; 64]; 64],
            nodes: 0,
            stop: None,
            deadline: None,
        }
    }

    pub fn with_stop_and_deadline(stop: Arc<AtomicBool>, deadline: Option<Instant>) -> Self {
        Self {
            killers: [[None; 2]; MAX_PLY],
            history: [[0; 64]; 64],
            nodes: 0,
            stop: Some(stop),
            deadline,
        }
    }

    /// Check every 4096 nodes whether the stop flag or deadline has been hit.
    #[inline]
    pub fn should_stop(&mut self) -> bool {
        self.nodes += 1;
        if self.nodes & 4095 == 0 {
            if let Some(ref s) = self.stop {
                if s.load(Ordering::Relaxed) { return true; }
            }
            if let Some(dl) = self.deadline {
                if Instant::now() >= dl { return true; }
            }
        }
        false
    }

    /// Halve all history scores between ID iterations so that shallower
    /// searches don't drown out discoveries from the current depth.
    pub fn age_history(&mut self) {
        for row in &mut self.history {
            for v in row {
                *v >>= 1;
            }
        }
    }

    /// Record a quiet move that caused a β-cutoff.
    fn record_cutoff(&mut self, ply: usize, depth: i32, mv: ChessMove) {
        let p = ply.min(MAX_PLY - 1);
        let k = &mut self.killers[p];
        // Only shift if this isn't already the first killer slot.
        if k[0].map_or(true, |k0| {
            k0.start_square() != mv.start_square() || k0.target_square() != mv.target_square()
        }) {
            k[1] = k[0];
            k[0] = Some(mv);
        }
        // Reward deeper cutoffs more.
        self.history[mv.start_square() as usize][mv.target_square() as usize] += depth * depth;
    }
}

// ── Quiescence search ─────────────────────────────────────────────────────────

/// Continues searching capture-only moves after the main search depth is
/// exhausted, so we never evaluate a position mid-capture-sequence.
/// This eliminates the horizon effect that causes higher depths to play worse.
fn quiescence(
    chess_board: &mut ChessBoard,
    conductor: &PieceConductor,
    mut alpha: i32,
    mut beta: i32,
    is_white: bool,
    qdepth: i32,
) -> i32 {
    if qdepth == 0 {
        return evaluate_board(chess_board, conductor);
    }
    let stand_pat = evaluate_board(chess_board, conductor);

    // Fail-soft quiescence: return the actual best score found, not alpha/beta.
    // This ensures the root can distinguish between a genuine best-score and a
    // fail-low that merely equals the accumulated alpha from a sibling search.
    if is_white {
        if stand_pat >= beta {
            return stand_pat; // fail-soft: return actual value
        }
        let mut best = stand_pat;
        if stand_pat > alpha {
            alpha = stand_pat;
        }

        let mut captures: Vec<ChessMove> = get_all_legal_moves_for_color(chess_board, conductor, is_white)
            .into_iter()
            .filter(|m| m.capture.is_some())
            .collect();
        captures.sort();

        for mut chess_move in captures {
            chess_board.make_move(&mut chess_move);
            let eval = quiescence(chess_board, conductor, alpha, beta, false, qdepth - 1);
            chess_board.undo_move();

            if eval >= beta {
                return eval; // fail-soft
            }
            if eval > best {
                best = eval;
            }
            if eval > alpha {
                alpha = eval;
            }
        }
        best
    } else {
        if stand_pat <= alpha {
            return stand_pat; // fail-soft
        }
        let mut best = stand_pat;
        if stand_pat < beta {
            beta = stand_pat;
        }

        let mut captures: Vec<ChessMove> = get_all_legal_moves_for_color(chess_board, conductor, is_white)
            .into_iter()
            .filter(|m| m.capture.is_some())
            .collect();
        captures.sort();

        for mut chess_move in captures {
            chess_board.make_move(&mut chess_move);
            let eval = quiescence(chess_board, conductor, alpha, beta, true, qdepth - 1);
            chess_board.undo_move();

            if eval <= alpha {
                return eval; // fail-soft
            }
            if eval < best {
                best = eval;
            }
            if eval < beta {
                beta = eval;
            }
        }
        best
    }
}

// ── Move ordering ─────────────────────────────────────────────────────────────

/// Order moves for best-first search:
///   1. TT / PV move (if present)
///   2. Captures + non-capture promotions (sorted by MVV-LVA via `Ord`)
///   3. Killer moves (quiet moves that caused β-cutoffs at this ply)
///   4. Remaining quiet moves, sorted by history score (descending)
fn order_moves(
    moves: &mut Vec<ChessMove>,
    tt_move: Option<ChessMove>,
    killers: &[Option<ChessMove>; 2],
    history: &[[i32; 64]; 64],
) {
    // Pull out the TT move first.
    let tt_entry = tt_move.and_then(|tt_m| {
        moves
            .iter()
            .position(|m| {
                m.start_square() == tt_m.start_square()
                    && m.target_square() == tt_m.target_square()
            })
            .map(|i| moves.remove(i))
    });

    // Partition remaining moves into captures/promotions vs quiet.
    let mut captures = Vec::with_capacity(moves.len());
    let mut quiets = Vec::with_capacity(moves.len());
    for m in moves.drain(..) {
        if m.capture.is_some() || m.is_promotion() {
            captures.push(m);
        } else {
            quiets.push(m);
        }
    }

    // Sort captures best-first (existing MVV-LVA Ord, `.reverse()` inside Ord
    // already ensures `sort()` gives highest-value captures first).
    captures.sort();

    // Extract killer moves from quiets, preserving killer priority order.
    let mut killer_entries = Vec::new();
    for killer in killers.iter().flatten() {
        if let Some(pos) = quiets.iter().position(|m| {
            m.start_square() == killer.start_square()
                && m.target_square() == killer.target_square()
        }) {
            killer_entries.push(quiets.remove(pos));
        }
    }

    // Sort remaining quiets by history score, highest first.
    quiets.sort_by(|a, b| {
        history[b.start_square() as usize][b.target_square() as usize]
            .cmp(&history[a.start_square() as usize][a.target_square() as usize])
    });

    // Reassemble: TT move → captures → killers → history-sorted quiets.
    if let Some(tt_m) = tt_entry {
        moves.push(tt_m);
    }
    moves.extend(captures);
    moves.extend(killer_entries);
    moves.extend(quiets);
}

// ── Zugzwang guard ────────────────────────────────────────────────────────────

/// Returns true when the position is likely a zugzwang situation:
/// fewer than 2 minor/major pieces on the board means the side to move
/// may have no good quiet moves to pass, making null-move pruning unsafe.
fn is_zugzwang_prone(chess_board: &ChessBoard) -> bool {
    let minor_and_major = chess_board.get_knights()
        | chess_board.get_bishops()
        | chess_board.get_rooks()
        | chess_board.get_queens();
    minor_and_major.count_ones() < 2
}

// ── Alpha-beta ────────────────────────────────────────────────────────────────

/// Internal recursive alpha-beta search with transposition table,
/// killer moves, and history heuristic.
///
/// `ply` is the distance from the root (0 = root node's children start at 1).
pub fn alpha_beta(
    chess_board: &mut ChessBoard,
    conductor: &PieceConductor,
    tt: &mut TranspositionTable,
    ctx: &mut SearchContext,
    depth: i32,
    ply: usize,
    mut alpha: i32,
    mut beta: i32,
    is_white: bool,
    null_move_allowed: bool,
) -> (i32, Option<ChessMove>) {
    if ctx.should_stop() {
        return (evaluate_board(chess_board, conductor), None);
    }
    if depth == 0 {
        return (quiescence(chess_board, conductor, alpha, beta, is_white, 4), None);
    }

    let hash = chess_board.current_hash();
    let original_alpha = alpha;
    let original_beta = beta;

    // --- Transposition table probe ---
    let tt_move: Option<ChessMove> = if let Some(entry) = tt.probe(hash) {
        if entry.depth >= depth {
            match entry.flag {
                TtFlag::Exact => return (entry.score, entry.best_move),
                TtFlag::LowerBound => {
                    if entry.score > alpha {
                        alpha = entry.score;
                    }
                }
                TtFlag::UpperBound => {
                    if entry.score < beta {
                        beta = entry.score;
                    }
                }
            }
            if alpha >= beta {
                return (entry.score, entry.best_move);
            }
        }
        entry.best_move
    } else {
        None
    };

    // Compute once; reused by both NMP and LMR.
    let in_check = conductor.is_king_in_check(chess_board, is_white);

    // --- Null Move Pruning ---
    if null_move_allowed
        && depth >= 3
        && !in_check
        && !is_zugzwang_prone(chess_board)
    {
        let r = if depth >= 6 { 3 } else { 2 };
        chess_board.make_null_move();
        let null_score = alpha_beta(
            chess_board, conductor, tt, ctx,
            depth - 1 - r, ply + 1, alpha, beta, !is_white,
            false,
        ).0;
        chess_board.undo_null_move();

        if is_white && null_score >= beta { return (beta, None); }
        if !is_white && null_score <= alpha { return (alpha, None); }
    }

    let mut legal_moves = get_all_legal_moves_for_color(chess_board, conductor, is_white);

    if legal_moves.is_empty() {
        if in_check {
            // Checkmate: worst possible for the side to move.
            return if is_white { (-1_000_000, None) } else { (1_000_000, None) };
        } else {
            // Stalemate: draw.
            return (0, None);
        }
    }

    let killers = &ctx.killers[ply.min(MAX_PLY - 1)];
    // Safety: we need the history borrow separate from ctx for the mutable borrow
    // later (record_cutoff).  Copy the pointer via a raw slice reference to avoid
    // a lifetime overlap.  We only read history here; the write happens after the
    // loop body, when we hold no other borrows on ctx.
    let history_ptr: *const [[i32; 64]; 64] = &ctx.history;
    // SAFETY: we do not mutate ctx.history between this point and the end of the
    // loop iteration; record_cutoff is only called via `ctx` after the borrow on
    // `killers` (which is a field borrow) has ended.
    let history: &[[i32; 64]; 64] = unsafe { &*history_ptr };

    order_moves(&mut legal_moves, tt_move, killers, history);
    // `killers` borrow of ctx ends here (it is not used again below).

    let mut best_move: Option<ChessMove> = None;

    if is_white {
        let mut max_eval = i32::MIN;
        for (move_index, mut chess_move) in legal_moves.into_iter().enumerate() {
            chess_board.make_move(&mut chess_move);

            let can_reduce = move_index >= 2
                && depth >= 3
                && chess_move.capture.is_none()
                && !chess_move.is_promotion()
                && !in_check;

            let eval = if chess_board.is_repetition(2) {
                0 // draw by repetition
            } else if move_index == 0 {
                // PV node: full window search for first move.
                alpha_beta(chess_board, conductor, tt, ctx,
                    depth - 1, ply + 1, alpha, beta, false, true).0
            } else if can_reduce {
                // LMR + PVS: reduced null-window search first.
                let reduced = alpha_beta(chess_board, conductor, tt, ctx,
                    depth - 2, ply + 1, alpha, alpha + 1, false, true).0;
                if reduced > alpha && reduced < beta {
                    // Re-search at full depth, full window.
                    alpha_beta(chess_board, conductor, tt, ctx,
                        depth - 1, ply + 1, alpha, beta, false, true).0
                } else if reduced > alpha {
                    // Re-search at full depth, null window — then widen if needed.
                    let score = alpha_beta(chess_board, conductor, tt, ctx,
                        depth - 1, ply + 1, alpha, alpha + 1, false, true).0;
                    if score > alpha && score < beta {
                        alpha_beta(chess_board, conductor, tt, ctx,
                            depth - 1, ply + 1, alpha, beta, false, true).0
                    } else {
                        score
                    }
                } else {
                    reduced
                }
            } else {
                // PVS: null-window search for non-PV moves.
                let score = alpha_beta(chess_board, conductor, tt, ctx,
                    depth - 1, ply + 1, alpha, alpha + 1, false, true).0;
                if score > alpha && score < beta {
                    // Fail high — re-search with full window.
                    alpha_beta(chess_board, conductor, tt, ctx,
                        depth - 1, ply + 1, alpha, beta, false, true).0
                } else {
                    score
                }
            };

            chess_board.undo_move();

            if eval > max_eval {
                max_eval = eval;
                best_move = Some(chess_move);
            }
            if eval > alpha {
                alpha = eval;
            }
            if beta <= alpha {
                if chess_move.capture.is_none() && !chess_move.is_promotion() {
                    ctx.record_cutoff(ply, depth, chess_move);
                }
                break;
            }
        }

        let flag = if max_eval >= original_beta {
            TtFlag::LowerBound
        } else if max_eval <= original_alpha {
            TtFlag::UpperBound
        } else {
            TtFlag::Exact
        };
        tt.store(hash, depth, max_eval, flag, best_move);
        (max_eval, best_move)
    } else {
        let mut min_eval = i32::MAX;
        for (move_index, mut chess_move) in legal_moves.into_iter().enumerate() {
            chess_board.make_move(&mut chess_move);

            let can_reduce = move_index >= 2
                && depth >= 3
                && chess_move.capture.is_none()
                && !chess_move.is_promotion()
                && !in_check;

            let eval = if chess_board.is_repetition(2) {
                0 // draw by repetition
            } else if move_index == 0 {
                // PV node: full window search for first move.
                alpha_beta(chess_board, conductor, tt, ctx,
                    depth - 1, ply + 1, alpha, beta, true, true).0
            } else if can_reduce {
                // LMR + PVS: reduced null-window search first.
                let reduced = alpha_beta(chess_board, conductor, tt, ctx,
                    depth - 2, ply + 1, beta - 1, beta, true, true).0;
                if reduced < beta && reduced > alpha {
                    // Re-search at full depth, full window.
                    alpha_beta(chess_board, conductor, tt, ctx,
                        depth - 1, ply + 1, alpha, beta, true, true).0
                } else if reduced < beta {
                    // Re-search at full depth, null window — then widen if needed.
                    let score = alpha_beta(chess_board, conductor, tt, ctx,
                        depth - 1, ply + 1, beta - 1, beta, true, true).0;
                    if score < beta && score > alpha {
                        alpha_beta(chess_board, conductor, tt, ctx,
                            depth - 1, ply + 1, alpha, beta, true, true).0
                    } else {
                        score
                    }
                } else {
                    reduced
                }
            } else {
                // PVS: null-window search for non-PV moves.
                let score = alpha_beta(chess_board, conductor, tt, ctx,
                    depth - 1, ply + 1, beta - 1, beta, true, true).0;
                if score < beta && score > alpha {
                    // Fail low — re-search with full window.
                    alpha_beta(chess_board, conductor, tt, ctx,
                        depth - 1, ply + 1, alpha, beta, true, true).0
                } else {
                    score
                }
            };

            chess_board.undo_move();

            if eval < min_eval {
                min_eval = eval;
                best_move = Some(chess_move);
            }
            if eval < beta {
                beta = eval;
            }
            if beta <= alpha {
                if chess_move.capture.is_none() && !chess_move.is_promotion() {
                    ctx.record_cutoff(ply, depth, chess_move);
                }
                break;
            }
        }

        let flag = if min_eval <= original_alpha {
            TtFlag::UpperBound
        } else if min_eval >= original_beta {
            TtFlag::LowerBound
        } else {
            TtFlag::Exact
        };
        tt.store(hash, depth, min_eval, flag, best_move);
        (min_eval, best_move)
    }
}

// ── Root search ───────────────────────────────────────────────────────────────

/// Root-level search at a single fixed depth.  The caller supplies the TT so
/// it can be reused across iterations.  `prev_best` (the best move from the
/// previous iteration) is ordered first to maximise alpha pruning.
///
/// Shallow depths (< 3) use a sequential growing-alpha window.  Deeper depths
/// evaluate all root moves in parallel — each on its own board clone with a
/// fresh per-thread TT and SearchContext — then pick the best result.
pub fn search_root(
    chess_board: &mut ChessBoard,
    conductor: &PieceConductor,
    tt: &mut TranspositionTable,
    ctx: &mut SearchContext,
    depth: i32,
    alpha: i32,
    beta: i32,
    is_white: bool,
    prev_best: Option<ChessMove>,
) -> (i32, Option<ChessMove>) {
    let mut legal_moves = get_all_legal_moves_for_color(chess_board, conductor, is_white);
    if legal_moves.is_empty() {
        return (evaluate_board(chess_board, conductor), None);
    }
    // Order: prev_best first (PV move from previous ID iteration), then
    // killers at ply 0 (usually empty), then history-scored quiets.
    order_moves(&mut legal_moves, prev_best, &ctx.killers[0], &ctx.history);

    // --- Parallel root search (depth >= 3, native only) ---
    // Each root move is evaluated independently on its own board clone with a
    // fresh per-thread TT and SearchContext.
    if depth >= 3 {
        let stop = ctx.stop.clone();
        let dl = ctx.deadline;
        let results: Vec<(i32, ChessMove)> = legal_moves
            .into_par_iter()
            .map(|mut chess_move| {
                let mut board = chess_board.clone();
                let mut local_tt = TranspositionTable::new(PARALLEL_TT_SIZE);
                let mut local_ctx = match &stop {
                    Some(s) => SearchContext::with_stop_and_deadline(Arc::clone(s), dl),
                    None => SearchContext::new(),
                };
                board.make_move(&mut chess_move);
                let eval = alpha_beta(
                    &mut board, conductor, &mut local_tt, &mut local_ctx,
                    depth - 1, 1, alpha, beta, !is_white, true,
                ).0;
                (eval, chess_move)
            })
            .collect();

        let best_score = if is_white {
            results.iter().map(|(s, _)| *s).max().unwrap_or(i32::MIN + 1)
        } else {
            results.iter().map(|(s, _)| *s).min().unwrap_or(i32::MAX)
        };
        let best_moves: Vec<ChessMove> = results.into_iter()
            .filter(|(s, _)| *s == best_score)
            .map(|(_, mv)| mv)
            .collect();
        return (best_score, best_moves.choose(&mut rand::thread_rng()).copied());
    }

    // --- Sequential root search ---
    let mut best_score = if is_white { i32::MIN + 1 } else { i32::MAX };
    let mut best_moves: Vec<ChessMove> = Vec::new();

    for mut chess_move in legal_moves {
        chess_board.make_move(&mut chess_move);
        let (eval, _) = alpha_beta(
            chess_board, conductor, tt, ctx, depth - 1, 1, alpha, beta, !is_white, true,
        );
        chess_board.undo_move();

        if is_white {
            if eval > best_score {
                best_score = eval;
                best_moves.clear();
                best_moves.push(chess_move);
            } else if eval == best_score {
                best_moves.push(chess_move);
            }
        } else {
            if eval < best_score {
                best_score = eval;
                best_moves.clear();
                best_moves.push(chess_move);
            } else if eval == best_score {
                best_moves.push(chess_move);
            }
        }
    }

    let best = best_moves.choose(&mut rand::thread_rng()).copied();
    (best_score, best)
}

// ── Public entry points ───────────────────────────────────────────────────────

/// Single-depth root search — thin wrapper kept for tests and direct callers.
pub fn alpha_beta_root(
    chess_board: &mut ChessBoard,
    conductor: &PieceConductor,
    book: Option<&OpeningBook>,
    depth: i32,
    is_white: bool,
) -> (i32, Option<ChessMove>) {
    if let Some(book) = book {
        if let Some((from, to)) = book.probe(chess_board) {
            let legal = get_all_legal_moves_for_color(chess_board, conductor, is_white);
            if let Some(book_move) = legal
                .into_iter()
                .find(|m| m.start_square() == from && m.target_square() == to)
            {
                eprintln!("Book move: {}", book_move.to_san_simple());
                return (0, Some(book_move));
            }
        }
    }
    let mut tt = TranspositionTable::new(TT_SIZE);
    let mut ctx = SearchContext::new();
    search_root(chess_board, conductor, &mut tt, &mut ctx, depth, i32::MIN + 1, i32::MAX, is_white, None)
}

/// Iterative-deepening root search.  Searches depth 1, 2, …, max_depth,
/// reusing the same TT and SearchContext across iterations so that shallower
/// results guide deeper ones.  History scores are halved between iterations
/// to age older information.  The best move from each completed iteration
/// seeds the move ordering for the next.
///
/// If `deadline` is `Some`, the loop stops after the first completed iteration
/// that exceeds the deadline.  The result of the last *fully completed*
/// iteration is always returned, so the move is never half-searched.
pub fn iterative_deepening_root(
    chess_board: &mut ChessBoard,
    conductor: &PieceConductor,
    book: Option<&OpeningBook>,
    max_depth: i32,
    is_white: bool,
    deadline: Option<Instant>,
    stop: Option<Arc<AtomicBool>>,
) -> (i32, Option<ChessMove>) {
    if let Some(book) = book {
        if let Some((from, to)) = book.probe(chess_board) {
            let legal = get_all_legal_moves_for_color(chess_board, conductor, is_white);
            if let Some(book_move) = legal
                .into_iter()
                .find(|m| m.start_square() == from && m.target_square() == to)
            {
                eprintln!("Book move: {}", book_move.to_san_simple());
                return (0, Some(book_move));
            }
        }
    }

    let mut tt = TranspositionTable::new(TT_SIZE);
    let mut ctx = match stop.as_ref() {
        Some(s) => SearchContext::with_stop_and_deadline(Arc::clone(s), deadline),
        None => SearchContext::new(),
    };
    let mut best: (i32, Option<ChessMove>) = (if is_white { i32::MIN + 1 } else { i32::MAX }, None);

    for depth in 1..=max_depth {
        let iter_start = Instant::now();
        let (prev_score, prev_move) = best;

        // Age history at the start of each new iteration (skip depth 1 — nothing
        // to age yet).
        if depth > 1 {
            ctx.age_history();
        }

        let result = if depth <= 2 {
            // Full window for shallow depths — aspirating an unknown score is useless.
            search_root(chess_board, conductor, &mut tt, &mut ctx, depth, i32::MIN + 1, i32::MAX, is_white, prev_move)
        } else {
            // Narrow aspiration window around previous score.  Widen one side on
            // failure and retry until the result lands inside the window.
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

        // If stop was triggered mid-iteration, the result is unreliable —
        // keep the last fully completed iteration's result.
        if let Some(ref s) = stop {
            if s.load(Ordering::Relaxed) { break; }
        }

        best = result;

        if let Some(dl) = deadline {
            let now = Instant::now();
            if now >= dl { break; }
            // Time estimation: each depth typically takes 3-5x longer than
            // the previous. Don't start the next depth if likely to overshoot.
            let iter_elapsed = now.duration_since(iter_start);
            let time_left = dl.duration_since(now);
            if iter_elapsed * 3 > time_left { break; }
        }
    }

    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use chess_board::ChessBoard;
    use move_generator::{
        move_generator::get_all_legal_moves_for_color, piece_conductor::PieceConductor,
    };

    fn conductor() -> PieceConductor {
        PieceConductor::new()
    }

    // -----------------------------------------------------------------------
    // Basic sanity
    // -----------------------------------------------------------------------

    #[test]
    fn returns_a_move_from_starting_position() {
        let mut board = ChessBoard::new();
        let c = conductor();
        let (_, mv) = alpha_beta_root(&mut board, &c, None, 2, true);
        assert!(mv.is_some(), "Engine must return a move from the starting position");
    }

    #[test]
    fn score_is_zero_for_perfectly_symmetric_position() {
        // Starting position is symmetric, so the score should be 0.
        let mut board = ChessBoard::new();
        let c = conductor();
        let (score, _) = alpha_beta_root(&mut board, &c, None, 2, true);
        // Allow a small margin for positional asymmetries, but rough equality.
        assert!(
            score.abs() < 50,
            "Starting position score should be near 0, got {score}"
        );
    }

    // -----------------------------------------------------------------------
    // Tactical: winning captures
    // -----------------------------------------------------------------------

    /// Position: white king e1 (4), white queen d4 (27), black king e8 (60),
    /// black queen d5 (35).  White should immediately capture the undefended queen.
    #[test]
    fn white_captures_hanging_queen() {
        let mut board = ChessBoard::new();
        board.set_from_fen("4k3/8/8/3q4/3Q4/8/8/4K3 w - - 0 1");
        let c = conductor();
        let (score, mv) = alpha_beta_root(&mut board, &c, None, 2, true);
        let m = mv.expect("Engine must find a move");
        assert_eq!(m.start_square(), 27, "Should move from d4 (square 27)");
        assert_eq!(m.target_square(), 35, "Should capture on d5 (square 35)");
        assert!(score > 800, "Score should reflect a queen-up advantage, got {score}");
    }

    /// Same position, but now it is black to move — black should capture white's queen.
    #[test]
    fn black_captures_hanging_queen() {
        let mut board = ChessBoard::new();
        board.set_from_fen("4k3/8/8/3q4/3Q4/8/8/4K3 b - - 0 1");
        let c = conductor();
        let (score, mv) = alpha_beta_root(&mut board, &c, None, 2, false);
        let m = mv.expect("Engine must find a move");
        assert_eq!(m.start_square(), 35, "Should move from d5 (square 35)");
        assert_eq!(m.target_square(), 27, "Should capture on d4 (square 27)");
        assert!(score < -800, "Score should reflect black's queen-up advantage, got {score}");
    }

    // -----------------------------------------------------------------------
    // Checkmate detection
    // -----------------------------------------------------------------------

    /// Position: white queen f7 (53), white king g6 (46), black king g8 (62).
    /// At depth 2 the engine sees that black has no legal reply — checkmate.
    #[test]
    fn white_finds_checkmate_in_one() {
        let mut board = ChessBoard::new();
        board.set_from_fen("6k1/5Q2/6K1/8/8/8/8/8 w - - 0 1");
        let c = conductor();
        let (_, mv) = alpha_beta_root(&mut board, &c, None, 2, true);
        let m = mv.expect("Engine must find a move");

        let mut board_after = board.clone();
        let mut mv_copy = m;
        board_after.make_move(&mut mv_copy);
        let replies = get_all_legal_moves_for_color(&mut board_after, &c, false);
        assert!(
            replies.is_empty(),
            "After white's best move ({}) black should have no legal replies",
            m.to_san_simple()
        );
    }

    /// Position: black queen b3 (17), white king a1 (0), black king c1 (2).
    /// At depth 2 the engine sees that white has no legal reply — checkmate.
    #[test]
    fn black_finds_checkmate_in_one() {
        let mut board = ChessBoard::new();
        board.set_from_fen("8/8/8/8/8/1q6/8/K1k5 b - - 0 1");
        let c = conductor();
        let (_, mv) = alpha_beta_root(&mut board, &c, None, 2, false);
        let m = mv.expect("Engine must find a move");

        let mut board_after = board.clone();
        let mut mv_copy = m;
        board_after.make_move(&mut mv_copy);
        let replies = get_all_legal_moves_for_color(&mut board_after, &c, true);
        assert!(
            replies.is_empty(),
            "After black's best move ({}) white should have no legal replies",
            m.to_san_simple()
        );
    }

    // -----------------------------------------------------------------------
    // TT correctness: results must be consistent
    // -----------------------------------------------------------------------

    /// Running the same search twice must produce the same score.
    #[test]
    fn search_score_is_deterministic() {
        let mut board = ChessBoard::new();
        board.set_from_fen("4k3/8/8/3q4/3Q4/8/8/4K3 w - - 0 1");
        let c = conductor();
        let (s1, _) = alpha_beta_root(&mut board, &c, None, 3, true);
        let (s2, _) = alpha_beta_root(&mut board, &c, None, 3, true);
        assert_eq!(s1, s2, "Same search must produce the same score");
    }

    /// The internal alpha_beta (with explicit TT) must agree with alpha_beta_root
    /// on the evaluation of the same position.
    #[test]
    fn internal_alpha_beta_agrees_with_root() {
        let mut board = ChessBoard::new();
        board.set_from_fen("4k3/8/8/3q4/3Q4/8/8/4K3 w - - 0 1");
        let c = conductor();

        let (root_score, _) = alpha_beta_root(&mut board, &c, None, 2, true);

        let mut tt = TranspositionTable::new(1 << 16);
        let mut ctx = SearchContext::new();
        let (ab_score, _) = alpha_beta(&mut board, &c, &mut tt, &mut ctx, 2, 0, i32::MIN + 1, i32::MAX, true, true);

        assert_eq!(
            root_score, ab_score,
            "alpha_beta_root and alpha_beta must agree on score"
        );
    }

    /// After applying the best move the board is back to its original state
    /// (i.e. undo_move works correctly and the TT didn't corrupt anything).
    #[test]
    fn board_state_unchanged_after_search() {
        let mut board = ChessBoard::new();
        let hash_before = board.current_hash();
        let c = conductor();
        alpha_beta_root(&mut board, &c, None, 3, true);
        assert_eq!(
            board.current_hash(),
            hash_before,
            "Search must not leave the board in a modified state"
        );
    }

    /// NMP and LMR must not corrupt the board state.
    #[test]
    fn nmp_lmr_do_not_corrupt_board_state() {
        let mut board = ChessBoard::new();
        let hash_before = board.current_hash();
        let c = conductor();
        // depth=4 ensures NMP and LMR are triggered
        alpha_beta_root(&mut board, &c, None, 4, true);
        assert_eq!(board.current_hash(), hash_before);
    }

    // -----------------------------------------------------------------------
    // Iterative deepening
    // -----------------------------------------------------------------------

    #[test]
    fn id_returns_a_move_from_starting_position() {
        let mut board = ChessBoard::new();
        let c = conductor();
        let (_, mv) = iterative_deepening_root(&mut board, &c, None, 3, true, None, None);
        assert!(mv.is_some(), "ID must return a move from the starting position");
    }

    /// ID at max_depth=N must produce the same score as a single-depth search
    /// at depth=N, because both use the same underlying alpha-beta.
    #[test]
    fn id_score_matches_fixed_depth() {
        let mut board = ChessBoard::new();
        board.set_from_fen("4k3/8/8/3q4/3Q4/8/8/4K3 w - - 0 1");
        let c = conductor();
        let (id_score, _) = iterative_deepening_root(&mut board, &c, None, 3, true, None, None);
        let (ab_score, _) = alpha_beta_root(&mut board, &c, None, 3, true);
        assert_eq!(id_score, ab_score,
            "ID and fixed-depth must agree: id={id_score}, ab={ab_score}");
    }

    #[test]
    fn id_board_state_unchanged_after_search() {
        let mut board = ChessBoard::new();
        let hash_before = board.current_hash();
        let c = conductor();
        iterative_deepening_root(&mut board, &c, None, 4, true, None, None);
        assert_eq!(board.current_hash(), hash_before,
            "ID must not leave the board in a modified state");
    }

    /// ID must find the forced capture even through multiple depth iterations.
    #[test]
    fn id_finds_hanging_queen() {
        let mut board = ChessBoard::new();
        board.set_from_fen("4k3/8/8/3q4/3Q4/8/8/4K3 w - - 0 1");
        let c = conductor();
        let (score, mv) = iterative_deepening_root(&mut board, &c, None, 2, true, None, None);
        let m = mv.expect("ID must find a move");
        assert_eq!(m.start_square(), 27);
        assert_eq!(m.target_square(), 35);
        assert!(score > 800, "got {score}");
    }

    /// Black side: ID must find the forced capture (exercises the minimising
    /// branch of search_root).
    #[test]
    fn id_black_finds_hanging_queen() {
        let mut board = ChessBoard::new();
        board.set_from_fen("4k3/8/8/3q4/3Q4/8/8/4K3 b - - 0 1");
        let c = conductor();
        let (score, mv) = iterative_deepening_root(&mut board, &c, None, 2, false, None, None);
        let m = mv.expect("ID must find a move for black");
        assert_eq!(m.start_square(), 35);
        assert_eq!(m.target_square(), 27);
        assert!(score < -800, "got {score}");
    }

    /// ID must find checkmate-in-one for white across multiple depth iterations.
    #[test]
    fn id_white_finds_checkmate() {
        let mut board = ChessBoard::new();
        board.set_from_fen("6k1/5Q2/6K1/8/8/8/8/8 w - - 0 1");
        let c = conductor();
        let (_, mv) = iterative_deepening_root(&mut board, &c, None, 2, true, None, None);
        let m = mv.expect("ID must find a move");
        let mut board_after = board.clone();
        let mut mv_copy = m;
        board_after.make_move(&mut mv_copy);
        let replies = get_all_legal_moves_for_color(&mut board_after, &c, false);
        assert!(replies.is_empty(), "After ID's best move black should have no legal replies");
    }

    /// ID must find checkmate-in-one for black across multiple depth iterations.
    #[test]
    fn id_black_finds_checkmate() {
        let mut board = ChessBoard::new();
        board.set_from_fen("8/8/8/8/8/1q6/8/K1k5 b - - 0 1");
        let c = conductor();
        let (_, mv) = iterative_deepening_root(&mut board, &c, None, 2, false, None, None);
        let m = mv.expect("ID must find a move");
        let mut board_after = board.clone();
        let mut mv_copy = m;
        board_after.make_move(&mut mv_copy);
        let replies = get_all_legal_moves_for_color(&mut board_after, &c, true);
        assert!(replies.is_empty(), "After ID's best move white should have no legal replies");
    }

    /// ID at depth=4 exercises NMP (depth>=3) and LMR across all iterations.
    /// The score must still reflect the correct material balance.
    #[test]
    fn id_score_correct_with_nmp_lmr_active() {
        let mut board = ChessBoard::new();
        board.set_from_fen("4k3/8/8/3q4/3Q4/8/8/4K3 w - - 0 1");
        let hash_before = board.current_hash();
        let c = conductor();
        let (score, mv) = iterative_deepening_root(&mut board, &c, None, 4, true, None, None);
        assert!(mv.is_some(), "ID must return a move at depth 4");
        assert!(score > 800, "ID with NMP/LMR must still reflect a queen-up advantage, got {score}");
        assert_eq!(board.current_hash(), hash_before, "Board must be clean after search");
    }

    // -----------------------------------------------------------------------
    // is_zugzwang_prone
    // -----------------------------------------------------------------------

    #[test]
    fn zugzwang_prone_kings_only() {
        let mut board = ChessBoard::new();
        board.set_from_fen("k7/8/8/8/8/8/8/7K w - - 0 1");
        assert!(is_zugzwang_prone(&board), "K vs K should be zugzwang-prone");
    }

    #[test]
    fn zugzwang_prone_one_rook() {
        let mut board = ChessBoard::new();
        board.set_from_fen("k7/8/8/8/8/8/8/6RK w - - 0 1");
        assert!(is_zugzwang_prone(&board), "K+R vs K should be zugzwang-prone");
    }

    #[test]
    fn not_zugzwang_prone_two_rooks() {
        let mut board = ChessBoard::new();
        board.set_from_fen("k7/8/8/8/8/8/8/5RRK w - - 0 1");
        assert!(!is_zugzwang_prone(&board), "K+2R vs K should not be zugzwang-prone");
    }

    #[test]
    fn not_zugzwang_prone_starting_position() {
        let board = ChessBoard::new();
        assert!(!is_zugzwang_prone(&board), "Starting position should not be zugzwang-prone");
    }

    // -----------------------------------------------------------------------
    // Tactical correctness at depth >= 3 (NMP + LMR active)
    // -----------------------------------------------------------------------

    /// Same queen-capture position at depth=4 — NMP and LMR both active.
    #[test]
    fn white_captures_hanging_queen_depth_4() {
        let mut board = ChessBoard::new();
        board.set_from_fen("4k3/8/8/3q4/3Q4/8/8/4K3 w - - 0 1");
        let hash_before = board.current_hash();
        let c = conductor();
        let (score, mv) = alpha_beta_root(&mut board, &c, None, 4, true);
        assert!(mv.is_some(), "Engine must return a move at depth 4");
        assert!(score > 800, "Score should reflect a queen-up advantage, got {score}");
        assert_eq!(board.current_hash(), hash_before, "Board must be clean after search");
    }

    #[test]
    fn black_captures_hanging_queen_depth_4() {
        let mut board = ChessBoard::new();
        board.set_from_fen("4k3/8/8/3q4/3Q4/8/8/4K3 b - - 0 1");
        let hash_before = board.current_hash();
        let c = conductor();
        let (score, mv) = alpha_beta_root(&mut board, &c, None, 4, false);
        assert!(mv.is_some(), "Engine must return a move at depth 4");
        assert!(score < -800, "Score should reflect black's queen-up advantage, got {score}");
        assert_eq!(board.current_hash(), hash_before, "Board must be clean after search");
    }

    // -----------------------------------------------------------------------
    // NMP in-check guard
    // -----------------------------------------------------------------------

    #[test]
    fn nmp_skipped_when_in_check() {
        let mut board = ChessBoard::new();
        board.set_from_fen("k3r3/8/8/8/8/8/8/4K3 w - - 0 1");
        let hash_before = board.current_hash();
        let c = conductor();
        let (_, mv) = alpha_beta_root(&mut board, &c, None, 3, true);
        assert!(mv.is_some(), "Engine must return an evasion move when in check");
        assert_eq!(board.current_hash(), hash_before, "Board must be clean after search");
    }

    // -----------------------------------------------------------------------
    // Stalemate detection
    // -----------------------------------------------------------------------

    #[test]
    fn stalemate_is_draw() {
        let mut board = ChessBoard::new();
        board.set_from_fen("k7/8/1QK5/8/8/8/8/8 b - - 0 1");
        let c = conductor();
        let moves = get_all_legal_moves_for_color(&mut board, &c, false);
        assert!(moves.is_empty(), "Black should have no legal moves (stalemate)");
        let mut tt = TranspositionTable::new(1 << 16);
        let mut ctx = SearchContext::new();
        let (score, _) = alpha_beta(&mut board, &c, &mut tt, &mut ctx, 2, 0, i32::MIN + 1, i32::MAX, false, true);
        assert_eq!(score, 0, "Stalemate must evaluate to 0 (draw), got {score}");
    }

    #[test]
    fn engine_avoids_stalemate_when_winning() {
        let mut board = ChessBoard::new();
        board.set_from_fen("k7/P7/2K5/3Q4/8/8/8/8 w - - 0 1");
        let c = conductor();
        let (score, mv) = alpha_beta_root(&mut board, &c, None, 4, true);
        let m = mv.expect("Engine must return a move");
        assert!(score > 500, "White is winning, score should be large, got {score}");
        assert!(
            !(m.target_square() == 40),
            "Engine must not play Qa6 (stalemate)"
        );
    }

    // -----------------------------------------------------------------------
    // Sequential vs parallel parity
    // -----------------------------------------------------------------------

    #[test]
    fn sequential_root_sees_free_piece_with_aspiration() {
        let mut board = ChessBoard::new();
        board.set_from_fen("3qk3/8/8/8/8/2N5/8/4K3 b - - 0 1");
        let c = conductor();
        let mut tt = TranspositionTable::new(TT_SIZE);
        let mut ctx = SearchContext::new();
        let (score, mv) = search_root(&mut board, &c, &mut tt, &mut ctx, 4, -50, 50, false, None);
        assert!(score < -200 || score <= -50,
            "Black should win material or fail-low, got {score}");
        if score > -50 && score < 50 {
            let m = mv.expect("Must return a move");
            assert_eq!(m.target_square(), 18, "Should capture on c3 (sq 18), got {}", m.target_square());
        }
    }

    #[test]
    fn id_sequential_defends_hanging_piece() {
        let mut board = ChessBoard::new();
        board.set_from_fen("3qk3/8/8/8/8/2N5/8/3QK3 w - - 0 1");
        let c = conductor();
        let (score, mv) = iterative_deepening_root(&mut board, &c, None, 4, true, None, None);
        let m = mv.expect("Must return a move");
        assert!(score > -200,
            "White should not blunder a piece, score {score}");
        let _ = m;
    }

    #[test]
    fn id_handles_pin_correctly() {
        let mut board = ChessBoard::new();
        board.set_from_fen("4k3/8/8/8/1b6/8/3N4/4K3 w - - 0 1");
        let c = conductor();
        let (_, mv) = iterative_deepening_root(&mut board, &c, None, 4, true, None, None);
        let m = mv.expect("Must return a move");
        let legal = get_all_legal_moves_for_color(&mut board, &c, true);
        assert!(legal.iter().any(|lm| lm.start_square() == m.start_square() && lm.target_square() == m.target_square()),
            "Engine must return a legal move");
    }

    #[test]
    fn aspiration_agrees_with_full_window() {
        let mut board = ChessBoard::new();
        board.set_from_fen("4k3/8/8/3n4/3Q4/8/8/4K3 w - - 0 1");
        let c = conductor();

        let mut tt1 = TranspositionTable::new(TT_SIZE);
        let mut ctx1 = SearchContext::new();
        let (score_full, mv_full) = search_root(&mut board, &c, &mut tt1, &mut ctx1, 4, i32::MIN + 1, i32::MAX, true, None);

        let mut tt2 = TranspositionTable::new(TT_SIZE);
        let mut ctx2 = SearchContext::new();
        let (score_asp, _) = search_root(&mut board, &c, &mut tt2, &mut ctx2, 4, -50, 50, true, None);

        let m = mv_full.expect("Full window must return a move");
        assert!(score_full > 200, "White should win material, got {score_full}");
        if score_asp > -50 && score_asp < 50 {
            panic!("Aspiration search returned {score_asp} — should have failed high");
        }
        assert!(score_asp >= 50 || score_asp <= -50,
            "Aspiration should fail, got {score_asp}");
        let _ = m;
    }
}

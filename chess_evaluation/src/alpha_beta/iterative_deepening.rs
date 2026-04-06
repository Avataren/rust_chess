use chess_board::ChessBoard;
use chess_foundation::ChessMove;
use move_generator::{
    move_generator::get_all_legal_moves_for_color,
    piece_conductor::PieceConductor,
};
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};
use web_time::Instant;

use crate::{
    opening_book::OpeningBook,
    transposition_table::TranspositionTable,
};
use super::{alpha_beta, search_root, RootNoiseConfig, SearchContext, SearchParams, TT_SIZE};

/// Result of an iterative-deepening search.
pub struct SearchResult {
    pub score: i32,
    pub best_move: Option<ChessMove>,
    /// The predicted opponent reply (PV[1]).  Used for pondering.
    pub ponder_move: Option<ChessMove>,
    /// Total nodes searched across all threads (main + SMP helpers).
    pub total_nodes: u64,
}

/// Extract the opponent's predicted reply from the TT by making the best move
/// and probing.  Falls back to a quick depth-1 search if the TT has no entry.
pub fn extract_ponder_move(
    chess_board: &mut ChessBoard,
    conductor: &PieceConductor,
    tt: &TranspositionTable,
    best_move: ChessMove,
    is_white: bool,
) -> Option<ChessMove> {
    let mut mv = best_move;
    chess_board.make_move(&mut mv);

    let opponent_white = !is_white;
    let hash = chess_board.current_hash();

    // Try TT first
    let ponder = if let Some(entry) = tt.probe(hash) {
        entry.best_move()
    } else {
        None
    };

    // Fall back to a quick depth-2 search if TT miss
    let ponder = if ponder.is_none() {
        let mut ctx = SearchContext::new();
        ctx.init_accumulators(chess_board);
        let (_, fallback_move) = alpha_beta(
            chess_board,
            conductor,
            tt,
            &mut ctx,
            2,
            1,
            i32::MIN + 1,
            i32::MAX,
            opponent_white,
            true,
            None,
        );
        fallback_move
    } else {
        ponder
    };

    // Validate: the ponder move must be legal
    let ponder = ponder.and_then(|pm| {
        let mut legal = Vec::new();
        get_all_legal_moves_for_color(
            chess_board,
            conductor,
            opponent_white,
            &mut legal,
            &mut Vec::new(),
        );
        if legal.iter().any(|m| {
            m.start_square() == pm.start_square() && m.target_square() == pm.target_square()
        }) {
            Some(pm)
        } else {
            None
        }
    });

    chess_board.undo_move();
    ponder
}

/// Number of available CPU threads.  Used by callers that want Lazy SMP.
pub fn available_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// Convenience wrapper: creates a fresh TT and runs a single-threaded search.
/// For multi-threaded search, use `iterative_deepening_root_with_tt` with an
/// explicit `num_threads` and a persistent TT.
pub fn iterative_deepening_root(
    chess_board: &mut ChessBoard,
    conductor: &PieceConductor,
    book: Option<&OpeningBook>,
    max_depth: i32,
    is_white: bool,
    deadline: Option<Instant>,
    stop: Option<Arc<AtomicBool>>,
    noise: RootNoiseConfig,
) -> SearchResult {
    let tt = TranspositionTable::new(TT_SIZE);
    iterative_deepening_root_with_tt(
        chess_board,
        conductor,
        book,
        &tt,
        max_depth,
        is_white,
        deadline,
        stop,
        1,
        None,
        noise,
    )
}

/// Like `iterative_deepening_root` but accepts an external `TranspositionTable`
/// so the caller can persist it across moves.  The caller should call
/// `tt.new_search()` before each invocation to age old entries.
///
/// When `num_threads > 1`, Lazy SMP is used: N-1 helper threads search the
/// same position with a shared TT via `rayon::scope`, while the main thread
/// runs the authoritative iterative deepening.  Helpers populate the TT;
/// their results are discarded.
///
/// `on_depth` is called on the main thread after each completed depth with
/// `(depth, score_cp, nodes, elapsed_ms)`.  Use this for UCI `info` output.
pub fn iterative_deepening_root_with_tt(
    chess_board: &mut ChessBoard,
    conductor: &PieceConductor,
    book: Option<&OpeningBook>,
    tt: &TranspositionTable,
    max_depth: i32,
    is_white: bool,
    deadline: Option<Instant>,
    stop: Option<Arc<AtomicBool>>,
    num_threads: usize,
    on_depth: Option<&(dyn Fn(i32, i32, u64, u128) + Sync)>,
    noise: RootNoiseConfig,
) -> SearchResult {
    // Book probe before spawning any threads.
    if let Some(book) = book {
        if let Some((from, to)) = book.probe(chess_board) {
            let mut legal = Vec::new();
            get_all_legal_moves_for_color(
                chess_board,
                conductor,
                is_white,
                &mut legal,
                &mut Vec::new(),
            );
            if let Some(book_move) = legal
                .into_iter()
                .find(|m| m.start_square() == from && m.target_square() == to)
            {
                eprintln!("Book move: {}", book_move.to_san_simple());
                return SearchResult {
                    score: 0,
                    best_move: Some(book_move),
                    ponder_move: None,
                    total_nodes: 0,
                };
            }
        }
    }

    if num_threads <= 1 {
        return id_search_single(
            chess_board,
            conductor,
            tt,
            max_depth,
            is_white,
            deadline,
            stop,
            on_depth,
            noise,
            SearchParams::default(),
        );
    }

    // ── Lazy SMP: spawn helpers, main thread runs authoritative search ───
    let helper_stop = Arc::new(AtomicBool::new(false));
    let helper_nodes = Arc::new(AtomicU64::new(0));

    let mut result = SearchResult {
        score: 0,
        best_move: None,
        ponder_move: None,
        total_nodes: 0,
    };

    // Wrap on_depth so UCI info lines report *all* threads' combined nodes/NPS.
    // Helpers flush their delta to `helper_nodes` after each depth, so by the
    // time the main thread fires this callback the counter reflects live helper
    // progress (not just end-of-search totals).
    let hn_cb = Arc::clone(&helper_nodes);
    let wrapped_cb = move |depth: i32, score: i32, nodes: u64, ms: u128| {
        let total = nodes + hn_cb.load(Ordering::Relaxed);
        if let Some(cb) = on_depth {
            cb(depth, score, total, ms);
        }
    };
    let wrapped_ref: &(dyn Fn(i32, i32, u64, u128) + Sync) = &wrapped_cb;
    let on_depth_smp = on_depth.map(|_| wrapped_ref);

    rayon::scope(|s| {
        // Spawn N-1 helper threads, each with its own board clone & context.
        for i in 0..num_threads - 1 {
            let mut board = chess_board.clone();
            let cond = conductor.clone();
            let hs = Arc::clone(&helper_stop);
            let ext = stop.clone();
            let hn = Arc::clone(&helper_nodes);
            s.spawn(move |_| {
                smp_helper(&mut board, &cond, tt, max_depth, is_white, hs, ext, i, hn);
            });
        }

        // Main thread: full iterative deepening with aspiration & deadline.
        result = id_search_single(
            chess_board,
            conductor,
            tt,
            max_depth,
            is_white,
            deadline,
            stop.clone(),
            on_depth_smp,
            noise,
            SearchParams::default(),
        );

        // Main thread done — signal helpers to stop.
        helper_stop.store(true, Ordering::Release);
    });

    result.total_nodes += helper_nodes.load(Ordering::Acquire);
    result
}

/// Like `iterative_deepening_root_with_tt` but uses custom `SearchParams`.
///
/// Intended for automated search-parameter tuning (e.g. Optuna): the caller
/// builds a `SearchParams` from trial values, runs this function against a
/// deterministic puzzle set, and returns the solve rate as the objective.
///
/// Only single-threaded search is supported (num_threads is always 1); this
/// guarantees full determinism when the puzzle bench fixes its seed.
pub fn iterative_deepening_root_with_params(
    params: SearchParams,
    chess_board: &mut ChessBoard,
    conductor: &PieceConductor,
    book: Option<&crate::opening_book::OpeningBook>,
    tt: &TranspositionTable,
    max_depth: i32,
    is_white: bool,
    deadline: Option<Instant>,
    stop: Option<Arc<AtomicBool>>,
    on_depth: Option<&(dyn Fn(i32, i32, u64, u128) + Sync)>,
    noise: RootNoiseConfig,
) -> SearchResult {
    // Book probe (same as iterative_deepening_root_with_tt).
    if let Some(book) = book {
        use move_generator::move_generator::get_all_legal_moves_for_color;
        if let Some((from, to)) = book.probe(chess_board) {
            let mut legal = Vec::new();
            get_all_legal_moves_for_color(chess_board, conductor, is_white, &mut legal, &mut Vec::new());
            if let Some(book_move) = legal
                .into_iter()
                .find(|m| m.start_square() == from && m.target_square() == to)
            {
                return SearchResult {
                    score: 0,
                    best_move: Some(book_move),
                    ponder_move: None,
                    total_nodes: 0,
                };
            }
        }
    }

    id_search_single(
        chess_board,
        conductor,
        tt,
        max_depth,
        is_white,
        deadline,
        stop,
        on_depth,
        noise,
        params,
    )
}

// ── Lazy SMP internals ───────────────────────────────────────────────────────

/// Single-threaded iterative deepening with aspiration windows.  Used by the
/// main thread (and as the sole path when num_threads == 1).
///
/// `on_depth` is called after each fully-completed depth with
/// `(depth, score_cp, nodes, elapsed_ms)` so callers can emit UCI `info` lines.
fn id_search_single(
    chess_board: &mut ChessBoard,
    conductor: &PieceConductor,
    tt: &TranspositionTable,
    max_depth: i32,
    is_white: bool,
    deadline: Option<Instant>,
    stop: Option<Arc<AtomicBool>>,
    on_depth: Option<&(dyn Fn(i32, i32, u64, u128) + Sync)>,
    noise: RootNoiseConfig,
    params: SearchParams,
) -> SearchResult {
    let t0 = Instant::now();
    let mut ctx = SearchContext::with_params(params);
    // Initialize incremental accumulators for the dual-perspective neural model.
    // If no dual model is loaded, this is a no-op (acc_valid stays false).
    ctx.init_accumulators(chess_board);
    let mut best: (i32, Option<ChessMove>) = (if is_white { i32::MIN + 1 } else { i32::MAX }, None);
    // Tracks the noise-selected move from the most recently *completed* depth.
    // Kept separate from `best.1` (the true best move) so that move ordering
    // across depth iterations always uses the true best — only the final played
    // move is the noisy selection.
    let mut final_noisy_move: Option<ChessMove> = None;

    for depth in 1..=max_depth {
        let (prev_score, prev_move) = best;

        if depth > 1 {
            ctx.age_history();
        }

        let result = if depth < ctx.params.aspiration_min_depth {
            search_root(
                chess_board,
                conductor,
                tt,
                &mut ctx,
                depth,
                i32::MIN + 1,
                i32::MAX,
                is_white,
                prev_move,
                stop.clone(),
                noise,
            )
        } else {
            // Progressive aspiration window: start narrow, multiply delta on failure
            // instead of opening directly to full window.  Saves re-searches.
            let mut delta = ctx.params.aspiration_delta;
            let mut lo = prev_score.saturating_sub(delta);
            let mut hi = prev_score.saturating_add(delta);
            loop {
                let result = search_root(
                    chess_board,
                    conductor,
                    tt,
                    &mut ctx,
                    depth,
                    lo,
                    hi,
                    is_white,
                    prev_move,
                    stop.clone(),
                    noise,
                );
                if stop.as_ref().map_or(false, |s| s.load(Ordering::Acquire)) {
                    break result;
                }
                if result.0 > lo && result.0 < hi {
                    break result;
                } else if result.0 <= lo {
                    delta = (delta * 4).min(2000);
                    lo = if delta >= 2000 {
                        i32::MIN + 1
                    } else {
                        prev_score.saturating_sub(delta)
                    };
                } else {
                    delta = (delta * 4).min(2000);
                    hi = if delta >= 2000 {
                        i32::MAX
                    } else {
                        prev_score.saturating_add(delta)
                    };
                }
                if lo == i32::MIN + 1 && hi == i32::MAX {
                    break search_root(
                        chess_board,
                        conductor,
                        tt,
                        &mut ctx,
                        depth,
                        lo,
                        hi,
                        is_white,
                        prev_move,
                        stop.clone(),
                        noise,
                    );
                }
            }
        };

        if let Some(ref s) = stop {
            if s.load(Ordering::Acquire) {
                if best.1.is_none() && result.1.is_some() {
                    best = result;
                    final_noisy_move = ctx.noisy_move;
                }
                break;
            }
        }

        best = result;
        final_noisy_move = ctx.noisy_move;

        if let Some(cb) = on_depth {
            cb(depth, best.0, ctx.nodes, t0.elapsed().as_millis());
        }

        if let Some(dl) = deadline {
            if Instant::now() >= dl {
                break;
            }
        }
    }

    let played_move = if noise.is_disabled() { best.1 } else { final_noisy_move };
    // Ponder must be extracted from the position after the *played* move, not
    // the true best move, otherwise the TT returns a continuation for the wrong
    // branch and we send an illegal ponder.
    // Skip entirely when UCI Ponder is disabled (skip_ponder=true) to avoid
    // the depth-2 fallback search overhead during data generation.
    let ponder_move = if noise.skip_ponder {
        None
    } else {
        played_move.and_then(|bm| extract_ponder_move(chess_board, conductor, tt, bm, is_white))
    };

    SearchResult {
        score: best.0,
        best_move: played_move,
        ponder_move,
        total_nodes: ctx.nodes,
    }
}

/// Helper thread for Lazy SMP: runs iterative deepening with aspiration windows
/// to populate the shared TT.  Like the main thread but without book probe or
/// ponder extraction.  Stops when either `helper_stop` or `ext_stop` fires.
///
/// Each helper uses its own aspiration window (centred on its previous
/// iteration score), matching Stockfish's approach.  This ensures helper TT
/// entries are consistent with the main thread's aspiration-window scores,
/// avoiding the "full-window helper floods TT with scores outside main
/// thread's window" regression.
///
/// Helpers loop continuously (restarting from their staggered start depth
/// each pass) so they keep populating the TT for the full duration of the
/// main thread's search.
fn smp_helper(
    chess_board: &mut ChessBoard,
    conductor: &PieceConductor,
    tt: &TranspositionTable,
    max_depth: i32,
    is_white: bool,
    helper_stop: Arc<AtomicBool>,
    ext_stop: Option<Arc<AtomicBool>>,
    thread_idx: usize,
    total_nodes: Arc<AtomicU64>,
) {
    // Stagger starting depth across helpers so they cover different layers.
    let start_depth = 1 + (thread_idx % 3) as i32;

    'outer: loop {
        let mut ctx = SearchContext::new();
        ctx.init_accumulators(chess_board);
        let mut prev_score: i32 = if is_white { i32::MIN + 1 } else { i32::MAX };
        let mut prev_move: Option<ChessMove> = None;
        let mut stopped = false;
        // Track node delta so we can flush to the shared counter after each depth.
        let mut prev_node_count = 0u64;

        for depth in start_depth..=max_depth {
            if helper_stop.load(Ordering::Relaxed) {
                stopped = true;
                break;
            }
            if ext_stop
                .as_ref()
                .map_or(false, |s| s.load(Ordering::Acquire))
            {
                stopped = true;
                break;
            }

            if depth > start_depth {
                ctx.age_history();
            }

            // Use aspiration windows (same as main thread) at depth >= aspiration_min_depth.
            let stop = Some(Arc::clone(&helper_stop));
            let result = if depth < ctx.params.aspiration_min_depth {
                search_root(
                    chess_board,
                    conductor,
                    tt,
                    &mut ctx,
                    depth,
                    i32::MIN + 1,
                    i32::MAX,
                    is_white,
                    prev_move,
                    stop,
                    RootNoiseConfig::NONE,
                )
            } else {
                let asp_delta = ctx.params.aspiration_delta;
                let mut lo = prev_score.saturating_sub(asp_delta);
                let mut hi = prev_score.saturating_add(asp_delta);
                loop {
                    let r = search_root(
                        chess_board,
                        conductor,
                        tt,
                        &mut ctx,
                        depth,
                        lo,
                        hi,
                        is_white,
                        prev_move,
                        Some(Arc::clone(&helper_stop)),
                        RootNoiseConfig::NONE,
                    );
                    if helper_stop.load(Ordering::Relaxed) {
                        break r;
                    }
                    if r.0 > lo && r.0 < hi {
                        break r;
                    } else if r.0 <= lo {
                        lo = i32::MIN + 1;
                    } else {
                        hi = i32::MAX;
                    }
                    if lo == i32::MIN + 1 && hi == i32::MAX {
                        break search_root(
                            chess_board,
                            conductor,
                            tt,
                            &mut ctx,
                            depth,
                            lo,
                            hi,
                            is_white,
                            prev_move,
                            Some(Arc::clone(&helper_stop)),
                            RootNoiseConfig::NONE,
                        );
                    }
                }
            };

            prev_score = result.0;
            prev_move = result.1;

            // Flush node delta after every depth so the main thread's on_depth
            // callback sees a live combined count rather than just its own nodes.
            let delta = ctx.nodes - prev_node_count;
            total_nodes.fetch_add(delta, Ordering::Relaxed);
            prev_node_count = ctx.nodes;
        }
        if stopped {
            break 'outer;
        }
        // Completed one full pass — loop back for the next pass.
    }
    // All nodes were flushed per-depth above; no final fetch_add needed.
}

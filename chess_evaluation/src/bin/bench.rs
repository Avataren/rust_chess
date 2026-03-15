//! bench — NPS / depth benchmark for the chess engine.
//!
//! Runs a fixed-depth search on a standard test suite and reports nodes
//! searched, wall time, and NPS per position.
//!
//! Usage:
//!   cargo run -p chess_evaluation --bin bench --release
//!   cargo run -p chess_evaluation --bin bench --release -- --depth 7
//!   cargo run -p chess_evaluation --bin bench --release -- --depth 7 --threads 4
//!   cargo run -p chess_evaluation --bin bench --release -- --threads 1,2,4,8

use chess_board::ChessBoard;
use chess_evaluation::{
    alpha_beta, iterative_deepening_root_with_tt,
    SearchContext, TranspositionTable, TT_SIZE,
};
use move_generator::{
    move_generator::get_all_legal_moves_for_color,
    piece_conductor::PieceConductor,
};
use std::sync::{Arc, Mutex};
use std::time::Instant;

// ── Position suite ─────────────────────────────────────────────────────────────
//
// 12 positions: openings, middlegame tactics, pawn endgames, rook endgames.
// (FEN, white_to_move, label)
const POSITIONS: &[(&str, bool, &str)] = &[
    // Opening / early middlegame
    ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
     false, "After 1.e4"),
    ("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
     true,  "Ruy Lopez setup"),
    ("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R b KQkq - 0 6",
     false, "Italian Game"),

    // Tactical middlegame (Bratko-Kopec style)
    ("r3r1k1/pp3pbp/1qp3p1/2B5/2BP2b1/Q1n2N2/P4PPP/3RR1K1 w - - 0 1",
     true,  "Tactical – Sacrifice"),
    ("2r3k1/p4rpp/1q2pp2/3p4/3P4/1P2P3/P3QRPP/5RK1 w - - 0 1",
     true,  "Open file tension"),
    ("r1b1k2r/1p3ppp/p1n1pn2/q7/1bPP4/2N1PN2/PP3PPP/R1BQK2R w KQkq - 0 9",
     true,  "Complex middle"),
    ("3r2k1/1p3ppp/2pq4/p1n1r3/P2NP3/1P1Q4/2P2PPP/R3R1K1 b - - 0 1",
     false, "Central tension"),

    // Pawn endgame
    ("8/4kp2/8/1p2P1KP/8/8/8/8 w - - 0 1",
     true,  "K+P endgame"),
    ("8/8/8/4k3/5p2/4K3/8/8 b - - 0 1",
     false, "Pawn race"),

    // Rook endgame
    ("8/8/4k3/2r5/4K3/8/8/1R6 w - - 0 1",
     true,  "KR vs Kr"),
    ("6k1/p6p/5Rp1/8/8/2r3KP/8/8 w - - 0 1",
     true,  "Active rook"),

    // Queen vs passed pawns
    ("8/8/1p6/1p6/1p6/1k6/1p6/1K1Q4 w - - 0 1",
     true,  "Q vs passers"),
];

/// Run a sequential fixed-depth search (deterministic node count, single-threaded).
fn bench_sequential(fen: &str, is_white: bool, depth: i32) -> (u64, u128, i32) {
    let mut board = ChessBoard::new();
    board.set_from_fen(fen);
    let conductor = PieceConductor::new();
    let tt = TranspositionTable::new(TT_SIZE);
    let mut ctx = SearchContext::new();

    let moves = get_all_legal_moves_for_color(&mut board, &conductor, is_white);
    if moves.is_empty() {
        return (0, 0, 0);
    }

    let t0 = Instant::now();

    let mut best_score = if is_white { i32::MIN + 1 } else { i32::MAX };
    let alpha = i32::MIN + 1;
    let beta  = i32::MAX;

    for mut mv in moves {
        board.make_move(&mut mv);
        let (score, _) = alpha_beta(
            &mut board,
            &conductor,
            &tt,
            &mut ctx,
            depth - 1,
            1,
            alpha,
            beta,
            !is_white,
            true,
            None,
        );
        board.undo_move();

        if is_white && score > best_score { best_score = score; }
        if !is_white && score < best_score { best_score = score; }
    }

    let elapsed_ms = t0.elapsed().as_millis();
    (ctx.nodes, elapsed_ms, best_score)
}

/// Run a multi-threaded iterative-deepening search up to `depth`.
/// Returns (main_thread_nodes_at_final_depth, elapsed_ms, score).
fn bench_threaded(fen: &str, is_white: bool, depth: i32, num_threads: usize) -> (u64, u128, i32) {
    let mut board = ChessBoard::new();
    board.set_from_fen(fen);
    let conductor = PieceConductor::new();
    let tt = TranspositionTable::new(TT_SIZE);

    // Capture nodes & ms from the last completed depth via on_depth callback.
    let last_nodes: Arc<Mutex<u64>> = Arc::new(Mutex::new(0));
    let last_nodes_cb = Arc::clone(&last_nodes);

    let t0 = Instant::now();

    let result = iterative_deepening_root_with_tt(
        &mut board,
        &conductor,
        None,
        &tt,
        depth,
        is_white,
        None, // no deadline — run to full depth
        None, // no external stop
        num_threads,
        Some(&move |_d, _score, nodes, _ms| {
            *last_nodes_cb.lock().unwrap() = nodes;
        }),
    );

    let elapsed_ms = t0.elapsed().as_millis();
    let nodes = *last_nodes.lock().unwrap();
    (nodes, elapsed_ms, result.score)
}

/// `force_id`: when true, always use iterative-deepening path (even for 1 thread)
///             so all thread counts are directly comparable.
fn run_suite(depth: i32, num_threads: usize, force_id: bool) {
    let mode = if !force_id && num_threads <= 1 {
        "sequential fixed-depth (1 thread, deterministic)".to_string()
    } else if num_threads <= 1 {
        "iterative deepening (1 thread)".to_string()
    } else {
        format!("Lazy SMP ({num_threads} threads)")
    };
    println!("\n=== depth={depth}  {mode} ===\n");
    println!("{:<28} {:>12} {:>10} {:>12}", "Position", "Nodes", "ms", "NPS");
    println!("{}", "-".repeat(66));

    let mut total_nodes = 0u64;
    let mut total_ms    = 0u128;

    for &(fen, is_white, label) in POSITIONS {
        let (nodes, ms, _score) = if !force_id && num_threads <= 1 {
            bench_sequential(fen, is_white, depth)
        } else {
            bench_threaded(fen, is_white, depth, num_threads)
        };
        let nps = if ms > 0 { nodes as u128 * 1000 / ms } else { 0 };
        total_nodes += nodes;
        total_ms    += ms;
        println!("{:<28} {:>12} {:>10} {:>12}", label, nodes, ms, nps);
    }

    println!("{}", "-".repeat(66));
    let avg_nps = if total_ms > 0 { total_nodes as u128 * 1000 / total_ms } else { 0 };
    println!("{:<28} {:>12} {:>10} {:>12}", "TOTAL / AVG NPS",
             total_nodes, total_ms, avg_nps);
    println!();
    println!("depth={depth}  threads={num_threads}  positions={}  total_nodes={total_nodes}  avg_nps={avg_nps}  total_ms={total_ms}",
             POSITIONS.len());
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let depth: i32 = args.windows(2)
        .find(|w| w[0] == "--depth")
        .and_then(|w| w[1].parse().ok())
        .unwrap_or(7);

    // --threads accepts a single value (e.g. 4) or a comma-separated list (e.g. 1,2,4,8).
    let threads_str = args.windows(2)
        .find(|w| w[0] == "--threads")
        .map(|w| w[1].as_str())
        .unwrap_or("1");

    let thread_counts: Vec<usize> = threads_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    let thread_counts = if thread_counts.is_empty() { vec![1] } else { thread_counts };

    // If --threads is explicitly given, use iterative-deepening for all counts
    // so results are directly comparable.  Without --threads, use the
    // deterministic sequential fixed-depth path as the canonical baseline.
    let threads_explicit = args.iter().any(|a| a == "--threads");

    for &tc in &thread_counts {
        run_suite(depth, tc, threads_explicit);
    }
}

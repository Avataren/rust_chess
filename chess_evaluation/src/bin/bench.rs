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
//!
//! Hash + thread sweep (always uses iterative deepening — TT hit rates matter):
//!   cargo run -p chess_evaluation --bin bench --release -- \
//!     --depth 12 --hash 64,128,256,512,1024 --threads 1,4,8,16
//!
//! Shortcut for a full hash/thread grid (depth 12 recommended for TT benchmarks):
//!   cargo run -p chess_evaluation --bin bench --release -- --hash-sweep

use chess_board::ChessBoard;
use chess_evaluation::{
    alpha_beta, iterative_deepening_root_with_tt,
    SearchContext, TranspositionTable, TT_SIZE,
};

/// Weights embedded at compile time — only included when a NN feature is active.
#[cfg(any(feature = "nn-full-forward", feature = "nn-incremental", feature = "runtime-switch"))]
static NNUE_WEIGHTS: &[u8] = include_bytes!("../eval.npz");

#[cfg(any(feature = "nn-full-forward", feature = "nn-incremental", feature = "runtime-switch"))]
fn init_nn() {
    match chess_evaluation::init_neural_eval_from_bytes(NNUE_WEIGHTS) {
        Ok(()) => {
            #[cfg(feature = "runtime-switch")]
            chess_evaluation::set_neural_eval_enabled(true);
            eprintln!("Neural eval loaded ({} KB).", NNUE_WEIGHTS.len() / 1024);
        }
        Err(e) => eprintln!("warn: neural eval not loaded: {e}"),
    }
}
use move_generator::{
    move_generator::get_all_legal_moves_for_color,
    piece_conductor::PieceConductor,
};
use std::sync::Arc;
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
/// Returns (total_nodes_across_all_threads, elapsed_ms, score).
/// `tt_entries` overrides TT_SIZE when non-zero.
fn bench_threaded(fen: &str, is_white: bool, depth: i32, num_threads: usize, tt_entries: usize) -> (u64, u128, i32) {
    let mut board = ChessBoard::new();
    board.set_from_fen(fen);
    let conductor = PieceConductor::new();
    let entries = if tt_entries > 0 { tt_entries } else { TT_SIZE };
    let tt = TranspositionTable::new(entries);

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
        None,
        0,
    );

    let elapsed_ms = t0.elapsed().as_millis();
    (result.total_nodes, elapsed_ms, result.score)
}

/// Result of one (hash, threads, depth) configuration.
#[derive(Clone)]
struct BenchResult {
    hash_mb:     usize,
    threads:     usize,
    depth:       i32,
    total_nodes: u64,
    total_ms:    u128,
    avg_nps:     u128,
}

/// `force_id`: when true, always use iterative-deepening path (even for 1 thread)
///             so all thread counts are directly comparable.
/// `hash_mb`: transposition table size in MiB (0 = use TT_SIZE default).
/// Returns a BenchResult for the whole suite.
fn run_suite(depth: i32, num_threads: usize, force_id: bool, hash_mb: usize) -> BenchResult {
    let entries_for_mb = |mb: usize| mb * 1024 * 1024 / 24;
    let tt_entries = if hash_mb > 0 { entries_for_mb(hash_mb) } else { 0 };

    let hash_str = if hash_mb > 0 { format!("{}MB", hash_mb) } else { "default".to_string() };
    let mode = if !force_id && num_threads <= 1 && hash_mb == 0 {
        "sequential fixed-depth (1 thread, deterministic)".to_string()
    } else if num_threads <= 1 {
        format!("iterative deepening (1 thread, hash={hash_str})")
    } else {
        format!("Lazy SMP ({num_threads} threads, hash={hash_str})")
    };
    println!("\n=== depth={depth}  {mode} ===\n");
    println!("{:<28} {:>12} {:>10} {:>12}", "Position", "Nodes", "ms", "NPS");
    println!("{}", "-".repeat(66));

    let mut total_nodes = 0u64;
    let mut total_ms    = 0u128;

    for &(fen, is_white, label) in POSITIONS {
        let (nodes, ms, _score) = if !force_id && num_threads <= 1 && hash_mb == 0 {
            bench_sequential(fen, is_white, depth)
        } else {
            bench_threaded(fen, is_white, depth, num_threads, tt_entries)
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
    println!("depth={depth}  threads={num_threads}  hash={hash_str}  positions={}  \
              total_nodes={total_nodes}  avg_nps={avg_nps}  total_ms={total_ms}",
             POSITIONS.len());

    BenchResult { hash_mb, threads: num_threads, depth, total_nodes, total_ms, avg_nps }
}

fn print_top_results(results: &[BenchResult], n: usize) {
    let mut sorted = results.to_vec();
    sorted.sort_by(|a, b| b.avg_nps.cmp(&a.avg_nps));

    println!("\n{}", "═".repeat(72));
    println!("  TOP {} CONFIGURATIONS  (sorted by avg NPS)", n.min(sorted.len()));
    println!("{}", "═".repeat(72));
    println!("{:>4}  {:>8}  {:>8}  {:>7}  {:>12}  {:>10}  {:>12}",
             "Rank", "Hash", "Threads", "Depth", "Nodes", "ms", "Avg NPS");
    println!("{}", "-".repeat(72));

    for (i, r) in sorted.iter().take(n).enumerate() {
        let hash_str = if r.hash_mb > 0 { format!("{}MB", r.hash_mb) } else { "default".to_string() };
        println!("{:>4}  {:>8}  {:>8}  {:>7}  {:>12}  {:>10}  {:>12}",
                 i + 1,
                 hash_str,
                 r.threads,
                 r.depth,
                 r.total_nodes,
                 r.total_ms,
                 r.avg_nps);
    }
    println!("{}", "═".repeat(72));

    if let Some(best) = sorted.first() {
        let hash_str = if best.hash_mb > 0 { format!("{}MB", best.hash_mb) } else { "default".to_string() };
        println!("\n  Best: hash={hash_str}  threads={}  avg_nps={}", best.threads, best.avg_nps);
        if let Some(baseline) = sorted.last() {
            if baseline.avg_nps > 0 {
                let gain = best.avg_nps * 100 / baseline.avg_nps;
                println!("  Best vs worst: {gain}% ({}x speedup)",
                         best.avg_nps / baseline.avg_nps.max(1));
            }
        }
    }
    println!();
}

fn main() {
    #[cfg(any(feature = "nn-full-forward", feature = "nn-incremental", feature = "runtime-switch"))]
    init_nn();

    let args: Vec<String> = std::env::args().collect();

    // --hash-sweep: predefined grid of hash sizes and thread counts at depth 12.
    // The right depth to show meaningful TT hit rate differences.
    let hash_sweep = args.iter().any(|a| a == "--hash-sweep");

    let depth: i32 = args.windows(2)
        .find(|w| w[0] == "--depth")
        .and_then(|w| w[1].parse().ok())
        .unwrap_or(if hash_sweep { 12 } else { 7 });

    // --threads: single value or comma-separated list.
    let threads_str = args.windows(2)
        .find(|w| w[0] == "--threads")
        .map(|w| w[1].as_str())
        .unwrap_or(if hash_sweep { "1,4,8,16" } else { "1" });

    let thread_counts: Vec<usize> = threads_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .filter(|&n| n >= 1)
        .collect();
    let thread_counts = if thread_counts.is_empty() { vec![1] } else { thread_counts };

    // --hash: single MB value or comma-separated list (e.g. 64,128,256,512).
    // 0 = use engine default (TT_SIZE).  Only used with iterative-deepening path.
    let hash_str = args.windows(2)
        .find(|w| w[0] == "--hash")
        .map(|w| w[1].as_str())
        .unwrap_or(if hash_sweep { "16,64,128,256,512,1024,2048" } else { "0" });

    let hash_sizes: Vec<usize> = hash_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let hash_sizes = if hash_sizes.is_empty() { vec![0] } else { hash_sizes };

    // When hash sizes or multiple thread counts are specified, always use ID
    // so comparisons are apples-to-apples across configurations.
    let force_id = args.iter().any(|a| a == "--threads")
        || args.iter().any(|a| a == "--hash")
        || hash_sweep
        || thread_counts.len() > 1
        || hash_sizes.iter().any(|&h| h > 0);

    let total_runs = hash_sizes.len() * thread_counts.len();
    if total_runs > 1 {
        println!("Hash × thread sweep: {} configurations  depth={}",
                 total_runs, depth);
        println!("Hash sizes (MB): {hash_sizes:?}");
        println!("Thread counts:   {thread_counts:?}");
    }

    let mut all_results: Vec<BenchResult> = Vec::with_capacity(total_runs);

    for &hash_mb in &hash_sizes {
        for &tc in &thread_counts {
            let result = run_suite(depth, tc, force_id, hash_mb);
            all_results.push(result);
        }
    }

    // Summary report when we ran multiple configurations.
    if all_results.len() > 1 {
        print_top_results(&all_results, 5);
    }
}

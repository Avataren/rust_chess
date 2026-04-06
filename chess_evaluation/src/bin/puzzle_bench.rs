//! puzzle_bench — solve Lichess puzzles and report engine accuracy.
//!
//! Downloads / reads the Lichess puzzle CSV database and measures what
//! fraction the engine solves at a given depth. Results are broken down by
//! rating band and tactical theme.
//!
//! Usage:
//!   cargo run -p chess_evaluation --bin puzzle_bench --release -- \
//!     --file lichess_db_puzzle.csv.zst \
//!     [--count 1000] [--min-rating 1200] [--max-rating 2200] \
//!     [--depth 7] [--seed 42]
//!
//! The Lichess puzzle CSV can be downloaded from:
//!   https://database.lichess.org/#puzzles  (lichess_db_puzzle.csv.zst, ~280 MB)
//!
//! CSV columns (no quoting, comma-separated):
//!   PuzzleId, FEN, Moves, Rating, RatingDeviation, Popularity,
//!   NbPlays, Themes, GameUrl, OpeningTags
//!
//! The FEN is one move *before* the puzzle starts.  `Moves[0]` is the
//! opponent's "mistake" move; the engine must find `Moves[1]`.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use serde_json;

use rayon::prelude::*;

use chess_board::ChessBoard;
use chess_evaluation::{
    iterative_deepening_root_with_params, iterative_deepening_root_with_tt,
    RootNoiseConfig, SearchParams, TranspositionTable,
};
use chess_foundation::{
    chess_move::ChessMove,
    piece::PieceType,
};
use move_generator::{
    move_generator::get_all_legal_moves_for_color,
    piece_conductor::PieceConductor,
};

// ── Embedded NNUE weights ─────────────────────────────────────────────────────

#[cfg(any(feature = "nn-full-forward", feature = "nn-incremental", feature = "runtime-switch"))]
static NNUE_WEIGHTS: &[u8] = include_bytes!("../eval.npz");

#[cfg(any(feature = "nn-full-forward", feature = "nn-incremental", feature = "runtime-switch"))]
fn init_nn(eval_file: &str) {
    let result = if eval_file.is_empty() {
        chess_evaluation::init_neural_eval_from_bytes(NNUE_WEIGHTS)
            .map(|_| NNUE_WEIGHTS.len())
    } else {
        std::fs::read(eval_file)
            .map_err(|e| format!("cannot read {eval_file}: {e}"))
            .and_then(|bytes| {
                let n = bytes.len();
                chess_evaluation::init_neural_eval_from_bytes(&bytes).map(|_| n)
            })
    };
    match result {
        Ok(n) => {
            #[cfg(feature = "runtime-switch")]
            chess_evaluation::set_neural_eval_enabled(true);
            let src = if eval_file.is_empty() { "embedded" } else { eval_file };
            eprintln!("NNUE loaded from {src} ({} KB).", n / 1024);
        }
        Err(e) => eprintln!("warn: NNUE not loaded: {e}"),
    }
}

// ── Puzzle data ───────────────────────────────────────────────────────────────

struct Puzzle {
    fen:    String,
    moves:  Vec<String>, // moves[0]=setup, moves[1]=solution, moves[2..]=continuation
    rating: u32,
    themes: Vec<String>,
}

fn parse_puzzle(line: &str) -> Option<Puzzle> {
    // Fields: PuzzleId(0), FEN(1), Moves(2), Rating(3), RatingDeviation(4),
    //         Popularity(5), NbPlays(6), Themes(7), GameUrl(8), OpeningTags(9)
    let mut fields = line.splitn(10, ',');
    let _id     = fields.next()?;
    let fen     = fields.next()?.to_string();
    let moves_s = fields.next()?;
    let rating  = fields.next()?.trim().parse::<u32>().ok()?;
    let _rd     = fields.next()?;
    let _pop    = fields.next()?;
    let _plays  = fields.next()?;
    let themes  = fields.next()?;

    let moves: Vec<String> = moves_s.split_whitespace().map(|s| s.to_string()).collect();
    if moves.len() < 2 { return None; }

    let themes: Vec<String> = themes.split_whitespace().map(|s| s.to_string()).collect();

    Some(Puzzle { fen, moves, rating, themes })
}

// ── UCI helpers ───────────────────────────────────────────────────────────────

fn sq_from_uci(b: &[u8], i: usize) -> Option<u16> {
    let f = b[i    ].wrapping_sub(b'a') as u16;
    let r = b[i + 1].wrapping_sub(b'1') as u16;
    if f > 7 || r > 7 { return None; }
    Some(r * 8 + f)
}

/// Apply a UCI move string to the board. Returns false if the move is illegal.
fn apply_uci(board: &mut ChessBoard, conductor: &PieceConductor, uci: &str) -> bool {
    let b = uci.as_bytes();
    if b.len() < 4 { return false; }
    let from  = match sq_from_uci(b, 0) { Some(s) => s, None => return false };
    let to    = match sq_from_uci(b, 2) { Some(s) => s, None => return false };
    let promo = b.get(4).copied().map(|c| c as char);

    let is_white = board.is_white_active();
    let mut legal = Vec::new();
    get_all_legal_moves_for_color(board, conductor, is_white, &mut legal, &mut Vec::new());

    let found = legal.iter().find(|m| {
        m.start_square() == from && m.target_square() == to
            && match promo {
                None      => !m.is_promotion(),
                Some('q') => m.has_flag(ChessMove::PROMOTE_TO_QUEEN_FLAG),
                Some('r') => m.has_flag(ChessMove::PROMOTE_TO_ROOK_FLAG),
                Some('b') => m.has_flag(ChessMove::PROMOTE_TO_BISHOP_FLAG),
                Some('n') => m.has_flag(ChessMove::PROMOTE_TO_KNIGHT_FLAG),
                _         => false,
            }
    }).copied();

    if let Some(mut mv) = found {
        board.make_move(&mut mv);
        true
    } else {
        false
    }
}

fn mv_to_uci(mv: ChessMove) -> String {
    let sq = |s: u16| {
        let f = (b'a' + (s % 8) as u8) as char;
        let r = (b'1' + (s / 8) as u8) as char;
        format!("{f}{r}")
    };
    let promo = mv.promotion_piece_type().map(|pt| match pt {
        PieceType::Queen  => "q",
        PieceType::Rook   => "r",
        PieceType::Bishop => "b",
        PieceType::Knight => "n",
        _                 => "q",
    }).unwrap_or("");
    format!("{}{}{}", sq(mv.start_square()), sq(mv.target_square()), promo)
}

// ── Solver ────────────────────────────────────────────────────────────────────

/// Returns true if the engine finds the correct first move of the puzzle.
fn solve(
    puzzle:    &Puzzle,
    conductor: &PieceConductor,
    tt:        &TranspositionTable,
    depth:     i32,
    fresh_tt:  bool,
    params:    Option<&SearchParams>,
) -> bool {
    let mut board = ChessBoard::new();
    board.set_from_fen(&puzzle.fen);

    // Apply the setup move (opponent's "mistake" that creates the tactic).
    if !apply_uci(&mut board, conductor, &puzzle.moves[0]) {
        return false;
    }

    let is_white = board.is_white_active();

    let local_tt;
    let tt_ref = if fresh_tt {
        local_tt = TranspositionTable::new(1 << 18);
        &local_tt
    } else {
        tt
    };

    let result = if let Some(p) = params {
        // Custom params: always single-threaded for full determinism.
        iterative_deepening_root_with_params(
            p.clone(),
            &mut board,
            conductor,
            None,
            tt_ref,
            depth,
            is_white,
            None,
            None,
            None,
            RootNoiseConfig::NONE,
        )
    } else {
        iterative_deepening_root_with_tt(
            &mut board,
            conductor,
            None,
            tt_ref,
            depth,
            is_white,
            None, // no time limit
            None, // no stop signal
            1,    // single-threaded for determinism
            None,
            RootNoiseConfig::NONE,
        )
    };

    match result.best_move {
        Some(mv) => mv_to_uci(mv) == puzzle.moves[1],
        None     => false,
    }
}

// ── Reservoir sampling (Algorithm R, no external RNG dep) ────────────────────

fn reservoir_sample(items: impl Iterator<Item = Puzzle>, k: usize, seed: u64) -> Vec<Puzzle> {
    // 64-bit LCG — enough quality for uniform sampling
    let mut rng = seed ^ 0xdeadbeef_cafebabe;
    let next = |s: u64| -> u64 {
        s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)
    };

    let mut res: Vec<Puzzle> = Vec::with_capacity(k);
    let mut i = 0usize;
    for item in items {
        i += 1;
        if res.len() < k {
            res.push(item);
        } else {
            rng = next(rng);
            let j = ((rng >> 11) as usize) % i;
            if j < k {
                res[j] = item;
            }
        }
    }
    res
}

// ── Rating bands ─────────────────────────────────────────────────────────────

const BANDS: &[(u32, u32, &str)] = &[
    (0,    999,  "< 1000"),
    (1000, 1249, "1000–1249"),
    (1250, 1499, "1250–1499"),
    (1500, 1749, "1500–1749"),
    (1750, 1999, "1750–1999"),
    (2000, 2249, "2000–2249"),
    (2250, 2499, "2250–2499"),
    (2500, 9999, "≥ 2500"),
];

fn band_index(rating: u32) -> usize {
    BANDS.iter().position(|&(lo, hi, _)| rating >= lo && rating <= hi).unwrap_or(0)
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();

    macro_rules! arg {
        ($flag:expr, $default:expr) => {
            args.windows(2)
                .find(|w| w[0] == $flag)
                .and_then(|w| w[1].parse().ok())
                .unwrap_or($default)
        };
    }
    macro_rules! flag_str {
        ($flag:expr) => {
            args.windows(2).find(|w| w[0] == $flag).map(|w| w[1].as_str()).unwrap_or("")
        };
    }

    let file             = flag_str!("--file");
    let eval_file        = flag_str!("--eval-file");
    let params_file      = flag_str!("--params");

    #[cfg(any(feature = "nn-full-forward", feature = "nn-incremental", feature = "runtime-switch"))]
    init_nn(eval_file);

    // Load custom search params if --params was given.
    let custom_params: Option<SearchParams> = if params_file.is_empty() {
        None
    } else {
        let json = std::fs::read_to_string(params_file)
            .unwrap_or_else(|e| { eprintln!("Cannot read {params_file}: {e}"); std::process::exit(1); });
        let p: SearchParams = serde_json::from_str(&json)
            .unwrap_or_else(|e| { eprintln!("Cannot parse {params_file}: {e}"); std::process::exit(1); });
        Some(p)
    };

    let count:  usize   = arg!("--count",      1000usize);
    // Default rating range is unrestricted for export use cases;
    // benchmark runs typically pass --min-rating / --max-rating explicitly.
    let min_r:  u32     = arg!("--min-rating", 0u32);
    let max_r:  u32     = arg!("--max-rating", u32::MAX);
    let depth:  i32     = arg!("--depth",      7i32);
    let seed:   u64     = arg!("--seed",       42u64);
    // 0 = use all available logical CPUs; 1 = single-threaded (deterministic).
    let num_threads: usize = arg!("--threads", 0usize);
    // When set, write one line per failed puzzle: "<FEN>\t<move0> <move1> ..."
    // A downstream Python script expands these into labeled training positions.
    let export_failures = flag_str!("--export-failures");
    // Like --export-failures but writes ALL puzzles (failed + solved).
    // Preferred for generating finetune data: more positions, same tactical richness.
    let export_all      = flag_str!("--export-all");
    // When set, skip the engine search entirely: just read, filter, and export.
    // Useful for dumping the entire puzzle database as finetune input without
    // waiting for a search run.  --count 0 means no limit (export everything).
    let export_only     = args.iter().any(|a| a == "--export-only");
    // When set, create a fresh TranspositionTable per puzzle instead of sharing
    // one across all puzzles.  Eliminates cross-puzzle TT pollution so that
    // move-ordering and heuristic changes get an unbiased signal.
    // Trade-off: slower (no cross-puzzle TT reuse) and non-deterministic between
    // runs with the shared-TT baseline, but more reliable for A/B comparisons.
    let fresh_tt        = args.iter().any(|a| a == "--fresh-tt");

    if file.is_empty() {
        eprintln!("puzzle_bench: solve Lichess puzzles and report engine accuracy");
        eprintln!();
        eprintln!("USAGE:");
        eprintln!("  cargo run -p chess_evaluation --bin puzzle_bench --release --");
        eprintln!("    --file lichess_db_puzzle.csv.zst");
        eprintln!("    [--count N]              puzzles to sample (default: 1000)");
        eprintln!("    [--min-rating N]         lower rating filter (default: 1200)");
        eprintln!("    [--max-rating N]         upper rating filter (default: 2200)");
        eprintln!("    [--depth N]              search depth (default: 7)");
        eprintln!("    [--seed N]               RNG seed for sampling (default: 42)");
        eprintln!("    [--eval-file FILE]       override embedded NNUE weights (any .npz)");
        eprintln!("    [--threads N]            parallel solvers (default 0=all CPUs; 1=single-threaded)");
        eprintln!("    [--params FILE]          JSON file with SearchParams for tuning (forces single-threaded)");
        eprintln!("    [--export-failures FILE] write failed puzzle lines for finetune");
        eprintln!("    [--export-all FILE]      write ALL puzzle lines for finetune (recommended)");
        eprintln!("    [--fresh-tt]             fresh TT per puzzle (unbiased A/B; implied by --threads>1)");
        eprintln!();
        eprintln!("Download: https://database.lichess.org/#puzzles");
        eprintln!("  (lichess_db_puzzle.csv.zst, ~280 MB compressed)");
        std::process::exit(1);
    }

    println!("Lichess Puzzle Benchmark");
    println!("========================");
    println!("File:         {file}");
    let max_r_display = if max_r == u32::MAX { "∞".to_string() } else { max_r.to_string() };
    println!("Sample:       {count}  (rating {min_r}–{max_r_display})");
    println!("Depth:        {depth}");
    println!("Seed:         {seed}");
    let effective_threads = if num_threads == 0 {
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
    } else { num_threads };
    println!("Threads:      {effective_threads}{}",
        if num_threads == 1 { " (single-threaded)" } else { " (parallel, fresh TT per puzzle)" });
    println!("TT mode:      {}", if num_threads != 1 || fresh_tt { "fresh per puzzle" } else { "shared" });
    if !params_file.is_empty()      { println!("Params:       {params_file}"); }
    if !export_failures.is_empty() { println!("Export (failures): {export_failures}"); }
    if !export_all.is_empty()      { println!("Export (all):      {export_all}"); }
    println!();

    // ── Load & sample ─────────────────────────────────────────────────────────

    print!("Loading puzzles... ");
    let _ = stdout().flush();
    let t0 = Instant::now();

    let reader: Box<dyn Read> = if file.ends_with(".zst") {
        let f = std::fs::File::open(file).unwrap_or_else(|e| { eprintln!("Cannot open {file}: {e}"); std::process::exit(1); });
        Box::new(zstd::Decoder::new(f).unwrap_or_else(|e| { eprintln!("zstd error: {e}"); std::process::exit(1); }))
    } else {
        Box::new(std::fs::File::open(file).unwrap_or_else(|e| { eprintln!("Cannot open {file}: {e}"); std::process::exit(1); }))
    };

    let mut lines = BufReader::new(reader).lines();
    let _ = lines.next(); // skip header

    let filtered = lines
        .filter_map(|l| l.ok())
        .filter_map(|l| parse_puzzle(&l))
        .filter(|p| p.rating >= min_r && p.rating <= max_r);

    // count == 0 means "all" — collect without sampling.
    let puzzles = if count == 0 {
        filtered.collect::<Vec<_>>()
    } else {
        reservoir_sample(filtered, count, seed)
    };
    println!("{} puzzles  ({:.1}s)", puzzles.len(), t0.elapsed().as_secs_f32());

    if puzzles.is_empty() {
        eprintln!("No puzzles matched the filter. Try widening --min-rating / --max-rating.");
        std::process::exit(1);
    }

    // ── Solve ─────────────────────────────────────────────────────────────────

    // ── Export-only fast path (no search) ────────────────────────────────────

    if export_only {
        let out = if !export_all.is_empty() { export_all } else { export_failures };
        if out.is_empty() {
            eprintln!("--export-only requires --export-all <file> or --export-failures <file>");
            std::process::exit(1);
        }
        let f = std::fs::File::create(out)
            .unwrap_or_else(|e| { eprintln!("Cannot create {out}: {e}"); std::process::exit(1); });
        let mut w = std::io::BufWriter::new(f);
        let mut n = 0usize;
        for puzzle in &puzzles {
            writeln!(w, "{}\t{}", puzzle.fen, puzzle.moves.join(" ")).ok();
            n += 1;
        }
        println!("Exported {n} puzzle lines to {out}");
        println!("Next: python3 nn_training/scripts/gen_puzzle_finetune_data.py --input {out} --output <dir> --stockfish <path>");
        return;
    }

    let conductor = PieceConductor::new();
    let total = puzzles.len();
    let skipped = 0usize;

    // Configure rayon thread pool.
    // num_threads == 0: use all logical CPUs (rayon default).
    // num_threads == 1: single-threaded; uses shared TT (or --fresh-tt if set).
    // num_threads >  1: parallel; always uses a fresh TT per puzzle (thread safety).
    let parallel = num_threads != 1;
    if num_threads > 1 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build_global()
            .ok();
    }

    let dot_every = (total / 40).max(1);
    let dot_counter = AtomicUsize::new(0);
    print!("Solving  [");
    let _ = stdout().flush();
    let t_solve = Instant::now();

    // Per-puzzle result: (solved, band, themes, export_line)
    // Collected in original order so export files are deterministically ordered.
    let need_export = !export_all.is_empty() || !export_failures.is_empty();
    // When custom params are given, force single-threaded so params take effect
    // (iterative_deepening_root_with_params is always single-threaded).
    let effective_parallel = parallel && custom_params.is_none();

    let results: Vec<(bool, usize, Vec<String>, Option<String>)> = if effective_parallel {
        puzzles.par_iter().map(|puzzle| {
            // Each parallel solver gets its own TT — no cross-puzzle pollution
            // and no data races.
            let fresh = TranspositionTable::new(1 << 18);
            let ok = solve(puzzle, &conductor, &fresh, depth, true, None);
            let c = dot_counter.fetch_add(1, Ordering::Relaxed);
            if (c + 1) % dot_every == 0 { print!("."); let _ = stdout().flush(); }
            let export_line = need_export
                .then(|| format!("{}\t{}", puzzle.fen, puzzle.moves.join(" ")));
            (ok, band_index(puzzle.rating), puzzle.themes.clone(), export_line)
        }).collect()
    } else {
        // Single-threaded: shared TT (or fresh per puzzle if --fresh-tt or custom params).
        let shared_tt = TranspositionTable::new(1 << 18);
        let use_fresh = fresh_tt || custom_params.is_some();
        puzzles.iter().enumerate().map(|(i, puzzle)| {
            let ok = solve(puzzle, &conductor, &shared_tt, depth, use_fresh, custom_params.as_ref());
            if (i + 1) % dot_every == 0 { print!("."); let _ = stdout().flush(); }
            let export_line = need_export
                .then(|| format!("{}\t{}", puzzle.fen, puzzle.moves.join(" ")));
            (ok, band_index(puzzle.rating), puzzle.themes.clone(), export_line)
        }).collect()
    };

    // ── Merge results ─────────────────────────────────────────────────────────

    let mut solved = 0usize;
    let mut band_total:  [usize; 8] = [0; 8];
    let mut band_solved: [usize; 8] = [0; 8];
    let mut theme_total:  HashMap<String, usize> = HashMap::new();
    let mut theme_solved: HashMap<String, usize> = HashMap::new();

    // Optional export writers (filled sequentially so file order is stable).
    let mut fail_writer: Option<std::io::BufWriter<std::fs::File>> = if !export_failures.is_empty() {
        let f = std::fs::File::create(export_failures)
            .unwrap_or_else(|e| { eprintln!("Cannot create {export_failures}: {e}"); std::process::exit(1); });
        Some(std::io::BufWriter::new(f))
    } else { None };

    let mut all_writer: Option<std::io::BufWriter<std::fs::File>> = if !export_all.is_empty() {
        let f = std::fs::File::create(export_all)
            .unwrap_or_else(|e| { eprintln!("Cannot create {export_all}: {e}"); std::process::exit(1); });
        Some(std::io::BufWriter::new(f))
    } else { None };

    for (ok, band, themes, export_line) in &results {
        if *ok { solved += 1; }
        band_total[*band]  += 1;
        if *ok { band_solved[*band] += 1; }
        for theme in themes {
            *theme_total.entry(theme.clone()).or_insert(0) += 1;
            if *ok { *theme_solved.entry(theme.clone()).or_insert(0) += 1; }
        }
        if let Some(line) = export_line {
            if let Some(ref mut w) = all_writer {
                writeln!(w, "{line}").ok();
            }
            if !ok {
                if let Some(ref mut w) = fail_writer {
                    writeln!(w, "{line}").ok();
                }
            }
        }
    }

    let elapsed = t_solve.elapsed();
    println!("]  {:.1}s  ({:.0} puzzles/s)", elapsed.as_secs_f32(), total as f32 / elapsed.as_secs_f32());
    println!();

    // ── Report ────────────────────────────────────────────────────────────────

    let pct = |n: usize, d: usize| -> f32 {
        if d == 0 { 0.0 } else { n as f32 * 100.0 / d as f32 }
    };

    println!("Results");
    println!("=======");
    println!("  Overall:  {solved}/{total}  ({:.1}%)", pct(solved, total));
    if skipped > 0 {
        println!("  Skipped:  {skipped}  (illegal setup move — corrupt puzzle data)");
    }
    println!();

    println!("  By rating:");
    for (bi, &(_, _, label)) in BANDS.iter().enumerate() {
        let n = band_total[bi];
        let s = band_solved[bi];
        if n == 0 { continue; }
        println!("    {:>10}   {:>4}/{:<4}  ({:.1}%)", label, s, n, pct(s, n));
    }
    println!();

    // Top 20 themes by frequency
    let mut themes: Vec<(&String, usize, usize)> = theme_total
        .iter()
        .map(|(t, &n)| (t, *theme_solved.get(t).unwrap_or(&0), n))
        .collect();
    themes.sort_by(|a, b| b.2.cmp(&a.2).then(a.0.cmp(b.0)));

    println!("  By theme (top 20 by frequency):");
    println!("    {:<30} {:>9}  {:>7}", "Theme", "Solved", "Rate");
    println!("    {}", "-".repeat(52));
    for (theme, s, n) in themes.iter().take(20) {
        println!("    {:<30} {:>4}/{:<4}  ({:.1}%)", theme, s, n, pct(*s, *n));
    }
    println!();
    println!("  depth={depth}  puzzles={total}  time={:.1}s  seed={seed}", elapsed.as_secs_f32());
}

fn stdout() -> std::io::Stdout {
    std::io::stdout()
}

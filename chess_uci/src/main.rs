use std::io::{self, BufRead, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use chess_board::ChessBoard;
use chess_evaluation::{
    init_neural_eval, is_neural_eval_enabled, is_neural_eval_initialized,
    iterative_deepening_root_with_tt, set_neural_confidence_threshold,
    set_neural_eval_enabled, OpeningBook, TranspositionTable,
};

/// Weights embedded at compile time for direct NN features (nn-full-forward / nn-incremental).
/// With runtime-switch the weights are loaded later via `setoption name EvalFile`.
#[cfg(any(feature = "nn-full-forward", feature = "nn-incremental"))]
static NNUE_WEIGHTS: &[u8] = include_bytes!("../../chess_evaluation/src/eval.npz");
use chess_foundation::{piece::PieceType, ChessMove};
use move_generator::{move_generator::get_all_legal_moves_for_color, piece_conductor::PieceConductor};

const NAME: &str = "XavChess";
const AUTHOR: &str = "XavChess";

// ── Mate detection & Borg taunts ─────────────────────────────────────────────

const MATE_SCORE_THRESHOLD: i32 = 999_000;

const BORG_TAUNTS: &[&str] = &[
    "Resistance is futile. You will be mated.",
    "We are the Borg. Your king will be assimilated.",
    "Your tactical distinctiveness has been added to our own. Mate is inevitable.",
    "You will adapt to service us. Checkmate approaches.",
    "Your biological and technological distinctiveness is irrelevant. Prepare to be mated.",
    "We are the Borg. Lower your defenses and surrender your king.",
    "Irrelevant. Mate in progress.",
    "Your pieces will be assimilated. Resistance is futile.",
];

fn borg_taunt(depth: i32) -> &'static str {
    BORG_TAUNTS[(depth.unsigned_abs() as usize) % BORG_TAUNTS.len()]
}

/// Format score for UCI `info` line. Returns `"mate N"` for forced mates,
/// `"cp N"` otherwise. Positive N = we deliver mate, negative = we are mated.
fn format_score(score: i32) -> String {
    if score.abs() >= MATE_SCORE_THRESHOLD {
        let half_moves = 1_000_000 - score.abs();
        let moves = (half_moves + 1) / 2;
        let signed = if score > 0 { moves } else { -moves };
        format!("mate {signed}")
    } else {
        format!("cp {score}")
    }
}

// ── Move format helpers ──────────────────────────────────────────────────────

fn sq_to_uci(sq: u16) -> String {
    let file = (b'a' + (sq % 8) as u8) as char;
    let rank = (b'1' + (sq / 8) as u8) as char;
    format!("{}{}", file, rank)
}

fn mv_to_uci(mv: ChessMove) -> String {
    let promo = mv.promotion_piece_type().map(|pt| match pt {
        PieceType::Queen  => "q",
        PieceType::Rook   => "r",
        PieceType::Bishop => "b",
        PieceType::Knight => "n",
        _                 => "q",
    }).unwrap_or("");
    format!("{}{}{}", sq_to_uci(mv.start_square()), sq_to_uci(mv.target_square()), promo)
}

/// Find the legal move matching a UCI string like "e2e4" or "e7e8q".
fn parse_uci_move(uci: &str, legal: &[ChessMove]) -> Option<ChessMove> {
    let b = uci.as_bytes();
    if b.len() < 4 { return None; }
    let ff = b[0].wrapping_sub(b'a') as u16;
    let fr = b[1].wrapping_sub(b'1') as u16;
    let tf = b[2].wrapping_sub(b'a') as u16;
    let tr = b[3].wrapping_sub(b'1') as u16;
    if ff > 7 || fr > 7 || tf > 7 || tr > 7 { return None; }
    let from = fr * 8 + ff;
    let to   = tr * 8 + tf;
    let promo = b.get(4).copied().map(|c| c as char);

    legal.iter().find(|m| {
        m.start_square() == from && m.target_square() == to && match promo {
            None      => !m.is_promotion(),
            Some('q') => m.has_flag(ChessMove::PROMOTE_TO_QUEEN_FLAG),
            Some('r') => m.has_flag(ChessMove::PROMOTE_TO_ROOK_FLAG),
            Some('b') => m.has_flag(ChessMove::PROMOTE_TO_BISHOP_FLAG),
            Some('n') => m.has_flag(ChessMove::PROMOTE_TO_KNIGHT_FLAG),
            _         => false,
        }
    }).copied()
}

// ── Position command ─────────────────────────────────────────────────────────

/// Apply a UCI position command and return the full-move number (1-based).
fn apply_position(board: &mut ChessBoard, conductor: &PieceConductor, tokens: &[&str]) -> usize {
    let mut idx = 0;
    let mut fen_fullmove: usize = 1;
    if tokens.get(idx) == Some(&"startpos") {
        *board = ChessBoard::new();
        idx += 1;
    } else if tokens.get(idx) == Some(&"fen") {
        idx += 1;
        let end = tokens[idx..].iter().position(|&t| t == "moves").unwrap_or(tokens.len() - idx);
        let fen_str = tokens[idx..idx + end].join(" ");
        // Parse fullmove number from FEN (6th field)
        let parts: Vec<&str> = fen_str.split_whitespace().collect();
        if parts.len() >= 6 {
            fen_fullmove = parts[5].parse().unwrap_or(1);
        }
        board.set_from_fen(&fen_str);
        idx += end;
    }

    let mut move_count = 0usize;
    if tokens.get(idx) == Some(&"moves") {
        for uci in &tokens[idx + 1..] {
            let is_white = board.is_white_active();
            let legal = get_all_legal_moves_for_color(board, conductor, is_white);
            if let Some(mut mv) = parse_uci_move(uci, &legal) {
                board.make_move(&mut mv);
                move_count += 1;
            }
        }
    }

    // Each pair of half-moves is one full move
    fen_fullmove + move_count / 2
}

// ── Go command parsing ───────────────────────────────────────────────────────

struct GoParams {
    max_depth: i32,
    /// Soft deadline: iterative deepening checks this to decide whether to
    /// start another iteration.
    soft_deadline: Option<Instant>,
    /// Hard deadline: a background timer fires the stop flag at this point.
    hard_deadline: Option<Instant>,
    /// Whether this is a ponder search (think on opponent's time).
    _is_ponder: bool,
    /// Raw movetime in ms — used by ponder to set deadline on ponderhit.
    movetime_ms: Option<u64>,
}

fn parse_go(tokens: &[&str], is_white: bool, move_number: usize) -> GoParams {
    let mut max_depth: i32 = 64;
    let mut movetime_ms: Option<u64> = None;
    let mut wtime: Option<u64> = None;
    let mut btime: Option<u64> = None;
    let mut winc: u64 = 0;
    let mut binc: u64 = 0;
    let mut movestogo: Option<u64> = None;
    let mut is_ponder = false;

    let mut i = 0;
    while i < tokens.len() {
        match tokens[i] {
            "depth"     => { max_depth  = tokens.get(i+1).and_then(|s| s.parse().ok()).unwrap_or(64); i += 2; }
            "movetime"  => { movetime_ms = tokens.get(i+1).and_then(|s| s.parse().ok()); i += 2; }
            "wtime"     => { wtime = tokens.get(i+1).and_then(|s| s.parse().ok()); i += 2; }
            "btime"     => { btime = tokens.get(i+1).and_then(|s| s.parse().ok()); i += 2; }
            "winc"      => { winc  = tokens.get(i+1).and_then(|s| s.parse().ok()).unwrap_or(0); i += 2; }
            "binc"      => { binc  = tokens.get(i+1).and_then(|s| s.parse().ok()).unwrap_or(0); i += 2; }
            "movestogo" => { movestogo = tokens.get(i+1).and_then(|s| s.parse().ok()); i += 2; }
            "ponder"    => { is_ponder = true; i += 1; }
            "infinite"  => { i += 1; } // max_depth=64, no deadline — stop signal controls it
            _           => { i += 1; }
        }
    }

    // Explicit movetime: use it directly (self-play, analysis, etc.)
    if let Some(ms) = movetime_ms {
        let now = Instant::now();
        return GoParams {
            max_depth,
            soft_deadline: Some(now + Duration::from_millis(ms)),
            hard_deadline: Some(now + Duration::from_millis(ms)),
            _is_ponder: is_ponder,
            movetime_ms,
        };
    }

    // Game-phase-aware time management from clock info
    let (remaining, inc) = if is_white { (wtime, winc) } else { (btime, binc) };
    if let Some(rem) = remaining {
        // Estimate moves remaining for our side
        let moves_remaining: u64 = if let Some(mtg) = movestogo {
            mtg.max(1)
        } else {
            let estimated_total: u64 = 42;
            estimated_total.saturating_sub(move_number as u64).max(8)
        };

        // Base: equitable share of clock + 80% of increment
        let base = rem / moves_remaining + inc * 8 / 10;

        // Time-control-aware soft/hard multipliers and clock-percentage caps.
        // (soft_num/soft_den, hard_num/hard_den, cap_pct of rem)
        let (soft_num, soft_den, hard_num, hard_den, cap_pct): (u64, u64, u64, u64, u64) =
            if rem < 10_000 && inc == 0 {
                // No increment, flagging danger
                (1, 20, 1, 10, 5)
            } else if rem < 10_000 {
                // Has increment: use most of one increment per move
                (1, 4, 1, 2, 8)
            } else if rem < 60_000 {
                // Bullet (10s–1min)
                (1, 2, 3, 4, 8)
            } else if rem < 300_000 {
                // Blitz (1–5 min)
                (3, 4, 5, 4, 10)
            } else {
                // Rapid/Classical (5+ min)
                (1, 1, 3, 2, 12)
            };

        let soft_ms = (base * soft_num / soft_den)
            .min(rem * cap_pct / 100)
            .max(50);
        let hard_ms = (base * hard_num / hard_den)
            .min(rem * cap_pct * 2 / 100)
            .max(soft_ms);

        let now = Instant::now();
        GoParams {
            max_depth,
            soft_deadline: Some(now + Duration::from_millis(soft_ms)),
            hard_deadline: Some(now + Duration::from_millis(hard_ms)),
            _is_ponder: is_ponder,
            // For ponder: use hard_ms as the duration after ponderhit
            movetime_ms: Some(hard_ms),
        }
    } else {
        GoParams { max_depth, soft_deadline: None, hard_deadline: None, _is_ponder: is_ponder, movetime_ms: None }
    }
}

// ── Search thread ────────────────────────────────────────────────────────────

/// Create an isolated stop flag for a search, with a hard-deadline timer
/// and a propagator that bridges an external stop signal.
fn make_search_stop(
    ext_stop: &Arc<AtomicBool>,
    hard_deadline: Option<Instant>,
) -> Arc<AtomicBool> {
    let search_stop = Arc::new(AtomicBool::new(false));

    // Hard-deadline timer
    if let Some(hard) = hard_deadline {
        let stop_c = Arc::clone(&search_stop);
        thread::spawn(move || {
            let remaining = hard.saturating_duration_since(Instant::now());
            if !remaining.is_zero() {
                thread::sleep(remaining);
            }
            stop_c.store(true, Ordering::Relaxed);
        });
    }

    // Propagate external stop into search_stop
    let ext = Arc::clone(ext_stop);
    let prop = Arc::clone(&search_stop);
    thread::spawn(move || {
        while !prop.load(Ordering::Relaxed) {
            if ext.load(Ordering::Relaxed) {
                prop.store(true, Ordering::Relaxed);
                break;
            }
            thread::sleep(Duration::from_millis(5));
        }
    });

    search_stop
}

fn search_and_respond(
    mut board: ChessBoard,
    conductor: PieceConductor,
    book: OpeningBook,
    params: GoParams,
    stop: Arc<AtomicBool>,
    is_white: bool,
    tt: Arc<TranspositionTable>,
    num_threads: usize,
) {
    let search_stop = make_search_stop(&stop, params.hard_deadline);

    let t0 = Instant::now();
    tt.new_search();
    let result = iterative_deepening_root_with_tt(
        &mut board,
        &conductor,
        Some(&book),
        &tt,
        params.max_depth,
        is_white,
        params.soft_deadline,
        Some(search_stop),
        num_threads,
        Some(&|depth, score, nodes, ms| {
            // score is from white's perspective; UCI expects engine's (side-to-move) perspective.
            let engine_score = if is_white { score } else { -score };
            let nps = if ms > 0 { nodes * 1000 / ms as u64 } else { nodes };
            let score_str = format_score(engine_score);
            println!("info depth {depth} score {score_str} nodes {nodes} nps {nps} time {ms}");
            if engine_score >= MATE_SCORE_THRESHOLD {
                println!("info string {}", borg_taunt(depth));
            }
            let _ = io::stdout().flush();
        }),
        0,
    );
    let _ms = t0.elapsed().as_millis();
    let mv_str = result.best_move.map(mv_to_uci).unwrap_or_else(|| "0000".to_string());
    let ponder_str = result.ponder_move.map(mv_to_uci);
    if let Some(ref p) = ponder_str {
        println!("bestmove {mv_str} ponder {p}");
    } else {
        println!("bestmove {mv_str}");
    }
    let _ = io::stdout().flush();
}

/// Ponder search: think indefinitely until ponderhit or stop.
///
/// On `ponderhit`: the search CONTINUES for movetime_ms more, preserving
/// all TT state built up during pondering. This effectively extends
/// thinking time by however long we pondered.
/// On `stop`: stop flag fires → we output whatever we have.
fn ponder_and_respond(
    mut board: ChessBoard,
    conductor: PieceConductor,
    book: OpeningBook,
    params: GoParams,
    stop: Arc<AtomicBool>,
    ponderhit: Arc<AtomicBool>,
    is_white: bool,
    tt: Arc<TranspositionTable>,
    num_threads: usize,
) {
    let search_stop = Arc::new(AtomicBool::new(false));

    // Watcher thread: monitors external stop and ponderhit.
    // Phase 1 (ponder): search runs freely, stopped only by external stop.
    // Phase 2 (ponderhit): start a hard-deadline timer, then stop.
    {
        let ext = Arc::clone(&stop);
        let hit = Arc::clone(&ponderhit);
        let ss = Arc::clone(&search_stop);
        let movetime = params.movetime_ms;
        thread::spawn(move || {
            // Phase 1: wait for ponderhit or external stop
            loop {
                if ss.load(Ordering::Acquire) { return; } // search finished naturally
                if ext.load(Ordering::Acquire) {
                    ss.store(true, Ordering::Release);
                    return;
                }
                if hit.load(Ordering::Acquire) { break; }
                thread::sleep(Duration::from_millis(2));
            }

            // Phase 2: ponderhit received — give the search movetime more ms
            if let Some(ms) = movetime {
                let deadline = Instant::now() + Duration::from_millis(ms);
                loop {
                    if ext.load(Ordering::Acquire) || ss.load(Ordering::Acquire) {
                        ss.store(true, Ordering::Release);
                        return;
                    }
                    if Instant::now() >= deadline {
                        ss.store(true, Ordering::Release);
                        return;
                    }
                    thread::sleep(Duration::from_millis(2));
                }
            }
            // No movetime — continue until external stop
            loop {
                if ext.load(Ordering::Acquire) || ss.load(Ordering::Acquire) {
                    ss.store(true, Ordering::Release);
                    return;
                }
                thread::sleep(Duration::from_millis(2));
            }
        });
    }

    let t0 = Instant::now();
    tt.new_search();
    let result = iterative_deepening_root_with_tt(
        &mut board,
        &conductor,
        Some(&book),
        &tt,
        params.max_depth,
        is_white,
        None, // no soft deadline — stop flag controls everything
        Some(search_stop),
        num_threads,
        Some(&|depth, score, nodes, ms| {
            // score is from white's perspective; UCI expects engine's (side-to-move) perspective.
            let engine_score = if is_white { score } else { -score };
            let nps = if ms > 0 { nodes * 1000 / ms as u64 } else { nodes };
            let score_str = format_score(engine_score);
            println!("info depth {depth} score {score_str} nodes {nodes} nps {nps} time {ms}");
            if engine_score >= MATE_SCORE_THRESHOLD {
                println!("info string {}", borg_taunt(depth));
            }
            let _ = io::stdout().flush();
        }),
        0,
    );

    let _ms = t0.elapsed().as_millis();
    let mv_str = result.best_move.map(mv_to_uci).unwrap_or_else(|| "0000".to_string());
    let ponder_str = result.ponder_move.map(mv_to_uci);
    if let Some(ref p) = ponder_str {
        println!("bestmove {mv_str} ponder {p}");
    } else {
        println!("bestmove {mv_str}");
    }
    let _ = io::stdout().flush();
}

// ── Main loop ────────────────────────────────────────────────────────────────

fn main() {
    // Weights are loaded lazily: EvalFile setoption takes priority.
    // Fallback to embedded bytes happens in the isready handler below.

    let conductor = PieceConductor::new();
    let book = OpeningBook::build(&conductor);
    let mut board = ChessBoard::new();
    let mut move_number: usize = 1;

    // TT size: default 96 MB (4M entries × 24 B).  Configurable via UCI Hash.
    let mut hash_mb: usize = 96;
    let entries_for_mb = |mb: usize| mb * 1024 * 1024 / 24;

    // Persistent TT: survives across moves so the engine reuses prior search
    // analysis.  `new_search()` is called before each search to age old entries.
    // Cleared on `ucinewgame` or when Hash size changes.
    let mut tt: Arc<TranspositionTable> = Arc::new(TranspositionTable::new(entries_for_mb(hash_mb)));

    // Lazy SMP thread count.  Default = min(6, available logical CPUs).
    // 6 threads is the empirical sweet spot on the benchmark suite (depth 7).
    // Override via "setoption name Threads value N"; the value is capped at
    // the number of logical CPUs to avoid over-subscription.
    let max_threads = chess_evaluation::available_threads();
    let default_threads = max_threads.min(6);
    let mut num_threads: usize = default_threads;

    let stop_flag = Arc::new(AtomicBool::new(false));
    let ponderhit_flag = Arc::new(AtomicBool::new(false));
    let mut search_handle: Option<thread::JoinHandle<()>> = None;

    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        let line = line.trim().to_string();
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.is_empty() { continue; }

        match tokens[0] {
            "uci" => {
                println!("id name {NAME}");
                println!("id author {AUTHOR}");
                println!("option name Threads type spin default {default_threads} min 1 max {max_threads}");
                println!("option name Hash type spin default 96 min 1 max 65536");
                println!("option name Ponder type check default true");
                println!("option name EvalFile type string default <empty>");
                println!("option name NeuralEval type check default false");
                println!("option name NeuralConfidence type string default 0.0");
                println!("uciok");
            }
            "setoption" => {
                // setoption name <name> value <value>
                if let (Some(name_pos), Some(val_pos)) = (
                    tokens.iter().position(|&t| t == "name"),
                    tokens.iter().position(|&t| t == "value"),
                ) {
                    let name: String = tokens[name_pos + 1..val_pos].join(" ");
                    let value = tokens.get(val_pos + 1).unwrap_or(&"");
                    match name.to_lowercase().as_str() {
                        "threads" => {
                            if let Ok(n) = value.parse::<usize>() {
                                num_threads = n.max(1).min(max_threads);
                            }
                        }
                        "hash" => {
                            if let Ok(mb) = value.parse::<usize>() {
                                let mb = mb.max(1).min(65536);
                                if mb != hash_mb {
                                    hash_mb = mb;
                                    tt = Arc::new(TranspositionTable::new(entries_for_mb(hash_mb)));
                                }
                            }
                        }
                        "evalfile" => {
                            if !value.is_empty() && *value != "<empty>" {
                                match init_neural_eval(value) {
                                    Ok(()) => eprintln!("info string Loaded neural weights from {value}"),
                                    Err(e) => eprintln!("info string Failed to load neural weights: {e}"),
                                }
                            }
                        }
                        "neuraleval" => {
                            let enable = value.eq_ignore_ascii_case("true");
                            set_neural_eval_enabled(enable);
                            eprintln!(
                                "info string Neural eval {}",
                                if is_neural_eval_enabled() { "enabled" } else { "disabled" }
                            );
                        }
                        "neuralconfidence" => {
                            if let Ok(t) = value.parse::<f32>() {
                                let t = t.clamp(0.0, 1.0);
                                set_neural_confidence_threshold(t);
                                eprintln!(
                                    "info string Neural confidence threshold set to {:.2} (fallback rate ~{}%)",
                                    t,
                                    ((1.0 - t) * 100.0) as u32,
                                );
                            }
                        }
                        _ => {}
                    }
                }
            }
            "isready" => {
                // Load embedded weights as fallback if EvalFile was not provided.
                #[cfg(any(feature = "nn-full-forward", feature = "nn-incremental"))]
                if !is_neural_eval_initialized() {
                    chess_evaluation::init_neural_eval_from_bytes(NNUE_WEIGHTS)
                        .expect("failed to load embedded neural eval weights");
                }
                println!("readyok");
            }
            "ucinewgame" => {
                // Wait for any ongoing search to finish first
                stop_flag.store(true, Ordering::Release);
                if let Some(h) = search_handle.take() { let _ = h.join(); }
                stop_flag.store(false, Ordering::Release);
                board = ChessBoard::new();
                move_number = 1;
                // Clear TT: new game → old analysis is irrelevant.
                tt = Arc::new(TranspositionTable::new(entries_for_mb(hash_mb)));
            }
            "position" => {
                move_number = apply_position(&mut board, &conductor, &tokens[1..]);
            }
            "go" => {
                // Stop any previous search
                if let Some(h) = search_handle.take() {
                    stop_flag.store(true, Ordering::Release);
                    let _ = h.join();
                }
                stop_flag.store(false, Ordering::Release);

                let is_ponder = tokens.contains(&"ponder");
                let is_white = board.is_white_active();
                let params = parse_go(&tokens[1..], is_white, move_number);

                let board_c     = board.clone();
                let conductor_c = conductor.clone();
                let book_c      = book.clone();
                let stop_c      = Arc::clone(&stop_flag);
                let tt_c        = Arc::clone(&tt);

                let threads = num_threads;
                if is_ponder {
                    ponderhit_flag.store(false, Ordering::Release);
                    let ponder_c = Arc::clone(&ponderhit_flag);
                    search_handle = Some(thread::spawn(move || {
                        ponder_and_respond(board_c, conductor_c, book_c, params, stop_c, ponder_c, is_white, tt_c, threads);
                    }));
                } else {
                    search_handle = Some(thread::spawn(move || {
                        search_and_respond(board_c, conductor_c, book_c, params, stop_c, is_white, tt_c, threads);
                    }));
                }
            }
            "ponderhit" => {
                // Opponent played the predicted move — signal the ponder
                // search to transition to a real timed search.
                ponderhit_flag.store(true, Ordering::Release);
            }
            "stop" => {
                stop_flag.store(true, Ordering::Release);
                if let Some(h) = search_handle.take() { let _ = h.join(); }
                stop_flag.store(false, Ordering::Release);
            }
            "quit" => break,
            _ => {}
        }

        let _ = io::stdout().flush();
    }

    // Clean shutdown
    stop_flag.store(true, Ordering::Relaxed);
    if let Some(h) = search_handle.take() { let _ = h.join(); }
}

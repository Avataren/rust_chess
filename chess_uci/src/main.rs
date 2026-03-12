use std::io::{self, BufRead, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use chess_board::ChessBoard;
use chess_evaluation::{iterative_deepening_root, OpeningBook};
use chess_foundation::{piece::PieceType, ChessMove};
use move_generator::{move_generator::get_all_legal_moves_for_color, piece_conductor::PieceConductor};

const NAME: &str = "XavChess";
const AUTHOR: &str = "XavChess";

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
    is_ponder: bool,
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
    let mut is_ponder = false;

    let mut i = 0;
    while i < tokens.len() {
        match tokens[i] {
            "depth"    => { max_depth  = tokens.get(i+1).and_then(|s| s.parse().ok()).unwrap_or(64); i += 2; }
            "movetime" => { movetime_ms = tokens.get(i+1).and_then(|s| s.parse().ok()); i += 2; }
            "wtime"    => { wtime = tokens.get(i+1).and_then(|s| s.parse().ok()); i += 2; }
            "btime"    => { btime = tokens.get(i+1).and_then(|s| s.parse().ok()); i += 2; }
            "winc"     => { winc  = tokens.get(i+1).and_then(|s| s.parse().ok()).unwrap_or(0); i += 2; }
            "binc"     => { binc  = tokens.get(i+1).and_then(|s| s.parse().ok()).unwrap_or(0); i += 2; }
            "ponder"   => { is_ponder = true; i += 1; }
            "infinite" => { i += 1; } // max_depth=64, no deadline — stop signal controls it
            _          => { i += 1; }
        }
    }

    // Explicit movetime: use it directly (self-play, analysis, etc.)
    if let Some(ms) = movetime_ms {
        let now = Instant::now();
        return GoParams {
            max_depth,
            soft_deadline: Some(now + Duration::from_millis(ms)),
            hard_deadline: Some(now + Duration::from_millis(ms)),
            is_ponder,
            movetime_ms,
        };
    }

    // Game-phase-aware time management from clock info
    let (remaining, inc) = if is_white { (wtime, winc) } else { (btime, binc) };
    if let Some(rem) = remaining {
        // Phase-dependent base allocation
        let base = if move_number <= 10 {
            // Opening: book moves are instant, save time
            rem / 40 + inc * 3 / 4
        } else if move_number <= 30 {
            // Midgame: critical decisions, spend more
            rem / 20 + inc * 9 / 10
        } else {
            // Endgame: positions simpler, moderate budget
            rem / 30 + inc * 4 / 5
        };

        // Safety cap: never use more than 1/5 of remaining time
        let soft_ms = base.min(rem / 5).max(50);
        // Hard limit: 3x soft but at most 1/3 of remaining
        let hard_ms = (soft_ms * 3).min(rem / 3).max(soft_ms);

        let now = Instant::now();
        GoParams {
            max_depth,
            soft_deadline: Some(now + Duration::from_millis(soft_ms)),
            hard_deadline: Some(now + Duration::from_millis(hard_ms)),
            is_ponder,
            // For ponder: use hard_ms as the duration after ponderhit
            movetime_ms: Some(hard_ms),
        }
    } else {
        GoParams { max_depth, soft_deadline: None, hard_deadline: None, is_ponder, movetime_ms: None }
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
) {
    let search_stop = make_search_stop(&stop, params.hard_deadline);

    let t0 = Instant::now();
    let result = iterative_deepening_root(
        &mut board,
        &conductor,
        Some(&book),
        params.max_depth,
        is_white,
        params.soft_deadline,
        Some(search_stop),
    );
    let ms = t0.elapsed().as_millis();
    let mv_str = result.best_move.map(mv_to_uci).unwrap_or_else(|| "0000".to_string());
    let ponder_str = result.ponder_move.map(mv_to_uci);
    println!("info score cp {} time {ms}", result.score);
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
    let result = iterative_deepening_root(
        &mut board,
        &conductor,
        Some(&book),
        params.max_depth,
        is_white,
        None, // no soft deadline — stop flag controls everything
        Some(search_stop),
    );

    let ms = t0.elapsed().as_millis();
    let mv_str = result.best_move.map(mv_to_uci).unwrap_or_else(|| "0000".to_string());
    let ponder_str = result.ponder_move.map(mv_to_uci);
    println!("info score cp {} time {ms}", result.score);
    if let Some(ref p) = ponder_str {
        println!("bestmove {mv_str} ponder {p}");
    } else {
        println!("bestmove {mv_str}");
    }
    let _ = io::stdout().flush();
}

// ── Main loop ────────────────────────────────────────────────────────────────

fn main() {
    let conductor = PieceConductor::new();
    let book = OpeningBook::build(&conductor);
    let mut board = ChessBoard::new();
    let mut move_number: usize = 1;

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
                println!("option name Ponder type check default true");
                println!("uciok");
            }
            "isready" => {
                println!("readyok");
            }
            "ucinewgame" => {
                // Wait for any ongoing search to finish first
                stop_flag.store(true, Ordering::Release);
                if let Some(h) = search_handle.take() { let _ = h.join(); }
                stop_flag.store(false, Ordering::Release);
                board = ChessBoard::new();
                move_number = 1;
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

                if is_ponder {
                    ponderhit_flag.store(false, Ordering::Release);
                    let ponder_c = Arc::clone(&ponderhit_flag);
                    search_handle = Some(thread::spawn(move || {
                        ponder_and_respond(board_c, conductor_c, book_c, params, stop_c, ponder_c, is_white);
                    }));
                } else {
                    search_handle = Some(thread::spawn(move || {
                        search_and_respond(board_c, conductor_c, book_c, params, stop_c, is_white);
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

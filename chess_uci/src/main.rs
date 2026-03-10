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

fn apply_position(board: &mut ChessBoard, conductor: &PieceConductor, tokens: &[&str]) {
    let mut idx = 0;
    if tokens.get(idx) == Some(&"startpos") {
        *board = ChessBoard::new();
        idx += 1;
    } else if tokens.get(idx) == Some(&"fen") {
        idx += 1;
        let end = tokens[idx..].iter().position(|&t| t == "moves").unwrap_or(tokens.len() - idx);
        board.set_from_fen(&tokens[idx..idx + end].join(" "));
        idx += end;
    }

    if tokens.get(idx) == Some(&"moves") {
        for uci in &tokens[idx + 1..] {
            let is_white = board.is_white_active();
            let legal = get_all_legal_moves_for_color(board, conductor, is_white);
            if let Some(mut mv) = parse_uci_move(uci, &legal) {
                board.make_move(&mut mv);
            }
        }
    }
}

// ── Go command parsing ───────────────────────────────────────────────────────

struct GoParams {
    max_depth: i32,
    deadline: Option<Instant>,
}

fn parse_go(tokens: &[&str], is_white: bool) -> GoParams {
    let mut max_depth: i32 = 64;
    let mut movetime_ms: Option<u64> = None;
    let mut wtime: Option<u64> = None;
    let mut btime: Option<u64> = None;
    let mut winc: u64 = 0;
    let mut binc: u64 = 0;

    let mut i = 0;
    while i < tokens.len() {
        match tokens[i] {
            "depth"    => { max_depth  = tokens.get(i+1).and_then(|s| s.parse().ok()).unwrap_or(64); i += 2; }
            "movetime" => { movetime_ms = tokens.get(i+1).and_then(|s| s.parse().ok()); i += 2; }
            "wtime"    => { wtime = tokens.get(i+1).and_then(|s| s.parse().ok()); i += 2; }
            "btime"    => { btime = tokens.get(i+1).and_then(|s| s.parse().ok()); i += 2; }
            "winc"     => { winc  = tokens.get(i+1).and_then(|s| s.parse().ok()).unwrap_or(0); i += 2; }
            "binc"     => { binc  = tokens.get(i+1).and_then(|s| s.parse().ok()).unwrap_or(0); i += 2; }
            "infinite" => { i += 1; } // max_depth=64, no deadline — stop signal controls it
            _          => { i += 1; }
        }
    }

    // Derive movetime from clock if not given explicitly
    if movetime_ms.is_none() {
        let (remaining, inc) = if is_white { (wtime, winc) } else { (btime, binc) };
        if let Some(rem) = remaining {
            // 1/30 of remaining + 80% of increment, minimum 50 ms
            movetime_ms = Some((rem / 30).saturating_add(inc * 4 / 5).max(50));
        }
    }

    GoParams {
        max_depth,
        deadline: movetime_ms.map(|ms| Instant::now() + Duration::from_millis(ms)),
    }
}

// ── Search thread ────────────────────────────────────────────────────────────

fn search_and_respond(
    mut board: ChessBoard,
    conductor: PieceConductor,
    book: OpeningBook,
    params: GoParams,
    stop: Arc<AtomicBool>,
    is_white: bool,
) {
    let t0 = Instant::now();
    let (score, best) = iterative_deepening_root(
        &mut board,
        &conductor,
        Some(&book),
        params.max_depth,
        is_white,
        params.deadline,
        Some(stop),
    );
    let ms = t0.elapsed().as_millis();
    let mv_str = best.map(mv_to_uci).unwrap_or_else(|| "0000".to_string());
    println!("info score cp {score} time {ms}");
    println!("bestmove {mv_str}");
    let _ = io::stdout().flush();
}

// ── Main loop ────────────────────────────────────────────────────────────────

fn main() {
    let conductor = PieceConductor::new();
    let book = OpeningBook::build(&conductor);
    let mut board = ChessBoard::new();

    let stop_flag = Arc::new(AtomicBool::new(false));
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
                println!("uciok");
            }
            "isready" => {
                println!("readyok");
            }
            "ucinewgame" => {
                // Wait for any ongoing search to finish first
                stop_flag.store(true, Ordering::Relaxed);
                if let Some(h) = search_handle.take() { let _ = h.join(); }
                stop_flag.store(false, Ordering::Relaxed);
                board = ChessBoard::new();
            }
            "position" => {
                apply_position(&mut board, &conductor, &tokens[1..]);
            }
            "go" => {
                // Stop any previous search
                if let Some(h) = search_handle.take() {
                    stop_flag.store(true, Ordering::Relaxed);
                    let _ = h.join();
                }
                stop_flag.store(false, Ordering::Relaxed);

                let is_white = board.is_white_active();
                let params = parse_go(&tokens[1..], is_white);

                let board_c     = board.clone();
                let conductor_c = conductor.clone();
                let book_c      = book.clone();
                let stop_c      = Arc::clone(&stop_flag);

                search_handle = Some(thread::spawn(move || {
                    search_and_respond(board_c, conductor_c, book_c, params, stop_c, is_white);
                }));
            }
            "stop" => {
                stop_flag.store(true, Ordering::Relaxed);
                if let Some(h) = search_handle.take() { let _ = h.join(); }
            }
            "quit" => break,
            // Silently ignore: debug, setoption, register, ponderhit
            _ => {}
        }

        let _ = io::stdout().flush();
    }

    // Clean shutdown
    stop_flag.store(true, Ordering::Relaxed);
    if let Some(h) = search_handle.take() { let _ = h.join(); }
}

//! self_play — pit two UCI engines against each other and report results.
//!
//! Usage: self_play <engine1> <engine2> [--games N] [--movetime MS]

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use chess_board::ChessBoard;
use chess_foundation::{piece::PieceType, ChessMove};
use move_generator::{
    move_generator::get_all_legal_moves_for_color, piece_conductor::PieceConductor,
};

// ── Move helpers (mirrors chess_uci) ─────────────────────────────────────────

fn sq_to_uci(sq: u16) -> String {
    let file = (b'a' + (sq % 8) as u8) as char;
    let rank = (b'1' + (sq / 8) as u8) as char;
    format!("{}{}", file, rank)
}

fn mv_to_uci(mv: ChessMove) -> String {
    let promo = mv
        .promotion_piece_type()
        .map(|pt| match pt {
            PieceType::Queen => "q",
            PieceType::Rook => "r",
            PieceType::Bishop => "b",
            PieceType::Knight => "n",
            _ => "q",
        })
        .unwrap_or("");
    format!(
        "{}{}{}",
        sq_to_uci(mv.start_square()),
        sq_to_uci(mv.target_square()),
        promo
    )
}

fn parse_uci_move(uci: &str, legal: &[ChessMove]) -> Option<ChessMove> {
    let b = uci.as_bytes();
    if b.len() < 4 {
        return None;
    }
    let ff = b[0].wrapping_sub(b'a') as u16;
    let fr = b[1].wrapping_sub(b'1') as u16;
    let tf = b[2].wrapping_sub(b'a') as u16;
    let tr = b[3].wrapping_sub(b'1') as u16;
    if ff > 7 || fr > 7 || tf > 7 || tr > 7 {
        return None;
    }
    let from = fr * 8 + ff;
    let to = tr * 8 + tf;
    let promo = b.get(4).copied().map(|c| c as char);

    legal
        .iter()
        .find(|m| {
            m.start_square() == from
                && m.target_square() == to
                && match promo {
                    None => !m.is_promotion(),
                    Some('q') => m.has_flag(ChessMove::PROMOTE_TO_QUEEN_FLAG),
                    Some('r') => m.has_flag(ChessMove::PROMOTE_TO_ROOK_FLAG),
                    Some('b') => m.has_flag(ChessMove::PROMOTE_TO_BISHOP_FLAG),
                    Some('n') => m.has_flag(ChessMove::PROMOTE_TO_KNIGHT_FLAG),
                    _ => false,
                }
        })
        .copied()
}

// ── Engine process wrapper ────────────────────────────────────────────────────

struct Engine {
    name: String,
    child: Child,
    stdin: ChildStdin,
    rx: mpsc::Receiver<String>,
}

impl Engine {
    fn start(path: &str) -> Self {
        let mut child = Command::new(path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap_or_else(|e| panic!("Failed to spawn '{}': {}", path, e));

        let stdin = child.stdin.take().expect("stdin");
        let stdout = child.stdout.take().expect("stdout");

        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            for line in BufReader::new(stdout).lines().flatten() {
                if tx.send(line).is_err() {
                    break;
                }
            }
        });

        let name = std::path::Path::new(path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(path)
            .to_string();

        Engine { name, child, stdin, rx }
    }

    fn send(&mut self, cmd: &str) {
        let _ = writeln!(self.stdin, "{}", cmd);
    }

    /// Block until a line starting with `token` arrives, or the timeout elapses.
    fn wait_for(&self, token: &str, timeout: Duration) -> Option<String> {
        let deadline = std::time::Instant::now() + timeout;
        loop {
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            if remaining.is_zero() {
                return None;
            }
            match self.rx.recv_timeout(remaining) {
                Ok(line) if line.starts_with(token) => return Some(line),
                Ok(_) => continue, // info lines, etc.
                Err(_) => return None,
            }
        }
    }

    fn init(&mut self) {
        self.send("uci");
        self.wait_for("uciok", Duration::from_secs(10))
            .unwrap_or_else(|| panic!("'{}' did not respond to uci", self.name));
        self.send("isready");
        self.wait_for("readyok", Duration::from_secs(10))
            .unwrap_or_else(|| panic!("'{}' did not respond to isready", self.name));
    }

    fn new_game(&mut self) {
        self.send("ucinewgame");
        self.send("isready");
        self.wait_for("readyok", Duration::from_secs(10))
            .unwrap_or_else(|| panic!("'{}' did not respond after ucinewgame", self.name));
    }

    fn best_move(&mut self, position_cmd: &str, movetime_ms: u64) -> Option<String> {
        self.send(position_cmd);
        self.send(&format!("go movetime {}", movetime_ms));
        let line = self.wait_for("bestmove", Duration::from_millis(movetime_ms + 10_000))?;
        let mut parts = line.split_whitespace();
        parts.next(); // "bestmove"
        let mv = parts.next()?;
        if mv == "0000" { None } else { Some(mv.to_string()) }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        let _ = writeln!(self.stdin, "quit");
        let _ = self.child.wait();
    }
}

// ── Game ──────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
enum GameResult {
    Engine1Wins,
    Engine2Wins,
    Draw,
}

/// Maximum half-moves before we adjudicate a draw.
const MAX_PLIES: usize = 400;

fn play_game(
    engine1: &mut Engine,
    engine2: &mut Engine,
    movetime_ms: u64,
    engine1_is_white: bool,
    conductor: &PieceConductor,
) -> (GameResult, String) {
    let mut board = ChessBoard::new();
    let mut move_list: Vec<String> = Vec::new();

    engine1.new_game();
    engine2.new_game();

    for ply in 0..MAX_PLIES {
        let is_white = ply % 2 == 0;

        // ── Termination checks ──
        let legal = get_all_legal_moves_for_color(&mut board, conductor, is_white);

        if legal.is_empty() {
            return if conductor.is_king_in_check(&board, is_white) {
                // The side to move is in checkmate — the other side wins.
                if is_white {
                    (if engine1_is_white { GameResult::Engine2Wins } else { GameResult::Engine1Wins },
                     "checkmate".to_string())
                } else {
                    (if engine1_is_white { GameResult::Engine1Wins } else { GameResult::Engine2Wins },
                     "checkmate".to_string())
                }
            } else {
                (GameResult::Draw, "stalemate".to_string())
            };
        }

        if board.is_repetition(3) {
            return (GameResult::Draw, "repetition".to_string());
        }

        // ── Build position command ──
        let position_cmd = if move_list.is_empty() {
            "position startpos".to_string()
        } else {
            format!("position startpos moves {}", move_list.join(" "))
        };

        // ── Ask the right engine ──
        let uci_str = {
            let ask_engine1 = is_white == engine1_is_white;
            if ask_engine1 {
                engine1.best_move(&position_cmd, movetime_ms)
            } else {
                engine2.best_move(&position_cmd, movetime_ms)
            }
        };

        let Some(uci_str) = uci_str else {
            return (GameResult::Draw, "engine returned no move".to_string());
        };

        // ── Apply move ──
        let Some(mut chess_move) = parse_uci_move(&uci_str, &legal) else {
            return (GameResult::Draw, format!("illegal move '{}'", uci_str));
        };
        board.make_move(&mut chess_move);
        move_list.push(mv_to_uci(chess_move));
    }

    (GameResult::Draw, format!("move limit ({})", MAX_PLIES))
}

// ── CLI ───────────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut engine1_path: Option<String> = None;
    let mut engine2_path: Option<String> = None;
    let mut num_games: usize = 20;
    let mut movetime_ms: u64 = 100;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--games" => {
                num_games = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(20);
                i += 2;
            }
            "--movetime" => {
                movetime_ms = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(100);
                i += 2;
            }
            s if !s.starts_with("--") && engine1_path.is_none() => {
                engine1_path = Some(s.to_string());
                i += 1;
            }
            s if !s.starts_with("--") && engine2_path.is_none() => {
                engine2_path = Some(s.to_string());
                i += 1;
            }
            _ => {
                i += 1;
            }
        }
    }

    let e1 = engine1_path
        .expect("Usage: self_play <engine1> <engine2> [--games N] [--movetime MS]");
    let e2 = engine2_path
        .expect("Usage: self_play <engine1> <engine2> [--games N] [--movetime MS]");

    let conductor = PieceConductor::new();

    // Start engines once and reuse across all games.
    let mut engine1 = Engine::start(&e1);
    let mut engine2 = Engine::start(&e2);
    engine1.init();
    engine2.init();

    let e1_name = engine1.name.clone();
    let e2_name = engine2.name.clone();

    println!("Engine 1 : {}", e1_name);
    println!("Engine 2 : {}", e2_name);
    println!("Games    : {}", num_games);
    println!("Movetime : {} ms", movetime_ms);
    println!("{}", "─".repeat(60));

    let mut e1_wins = 0u32;
    let mut e2_wins = 0u32;
    let mut draws = 0u32;

    for game_num in 0..num_games {
        let engine1_is_white = game_num % 2 == 0;
        let c1 = if engine1_is_white { "W" } else { "B" };
        let c2 = if engine1_is_white { "B" } else { "W" };

        print!(
            "Game {:>3}/{} │ {}({}) vs {}({}) │ ",
            game_num + 1,
            num_games,
            e1_name,
            c1,
            e2_name,
            c2
        );
        let _ = std::io::stdout().flush();

        let (result, reason) =
            play_game(&mut engine1, &mut engine2, movetime_ms, engine1_is_white, &conductor);

        match result {
            GameResult::Engine1Wins => {
                e1_wins += 1;
                println!("1-0  ({})", reason);
            }
            GameResult::Engine2Wins => {
                e2_wins += 1;
                println!("0-1  ({})", reason);
            }
            GameResult::Draw => {
                draws += 1;
                println!("½-½  ({})", reason);
            }
        }

        // Running score after each game
        let played = (e1_wins + e2_wins + draws) as f64;
        let score = (e1_wins as f64 + draws as f64 * 0.5) / played * 100.0;
        print!("       Score so far: {e1_wins}W / {draws}D / {e2_wins}L  ({score:.1}%)");
        println!();
    }

    let total = num_games as f64;
    let score = (e1_wins as f64 + draws as f64 * 0.5) / total * 100.0;

    println!("{}", "═".repeat(60));
    println!("Final results  ({} games, {}ms/move)", num_games, movetime_ms);
    println!(
        "  {} : {} wins  ({:.1}%)",
        e1_name,
        e1_wins,
        e1_wins as f64 / total * 100.0
    );
    println!("  Draws         : {}", draws);
    println!(
        "  {} : {} wins  ({:.1}%)",
        e2_name,
        e2_wins,
        e2_wins as f64 / total * 100.0
    );
    println!("  {} score: {:.1}%", e1_name, score);
    println!("{}", "═".repeat(60));
}

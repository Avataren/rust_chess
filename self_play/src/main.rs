//! self_play — pit two UCI engines against each other and report results.
//!
//! Usage: self_play <engine1> <engine2> [--games N] [--movetime MS] [--no-ponder]
//!        [--engine1-opt "Name=Value"] [--engine2-opt "Name=Value"]

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use chess_board::ChessBoard;
use chess_evaluation::{evaluate_board, iterative_deepening_root, SearchResult};

#[cfg(any(feature = "nn-full-forward", feature = "nn-incremental", feature = "runtime-switch"))]
static NNUE_WEIGHTS: &[u8] = include_bytes!("../../chess_evaluation/src/eval.npz");

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

    fn init(&mut self, opts: &[String]) {
        self.send("uci");
        self.wait_for("uciok", Duration::from_secs(10))
            .unwrap_or_else(|| panic!("'{}' did not respond to uci", self.name));
        for opt in opts {
            // Accept either "Name=Value" or "Name Value" formats
            let cmd = if let Some((name, value)) = opt.split_once('=') {
                format!("setoption name {} value {}", name.trim(), value.trim())
            } else {
                format!("setoption name {}", opt.trim())
            };
            self.send(&cmd);
        }
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

    /// Parse a "bestmove X [ponder Y]" line into (move, ponder_move).
    fn parse_bestmove(line: &str) -> (Option<String>, Option<String>) {
        let mut parts = line.split_whitespace();
        parts.next(); // "bestmove"
        let mv = parts.next().map(|s| s.to_string());
        let ponder = if parts.next() == Some("ponder") {
            parts.next().map(|s| s.to_string())
        } else {
            None
        };
        let mv = mv.filter(|s| s != "0000");
        (mv, ponder)
    }

    fn best_move(&mut self, position_cmd: &str, movetime_ms: u64) -> (Option<String>, Option<String>) {
        self.send(position_cmd);
        self.send(&format!("go movetime {}", movetime_ms));
        match self.wait_for("bestmove", Duration::from_millis(movetime_ms + 10_000)) {
            Some(line) => Self::parse_bestmove(&line),
            None => (None, None),
        }
    }

}

impl Drop for Engine {
    fn drop(&mut self) {
        let _ = writeln!(self.stdin, "quit");
        let _ = self.child.wait();
    }
}

// ── Multi-ponder ─────────────────────────────────────────────────────────────

/// How many opponent candidate moves to ponder in parallel.
const PONDER_CANDIDATES: usize = 4;

struct PonderThread {
    /// UCI string of the opponent move (for matching).
    opponent_uci: String,
    stop: Arc<AtomicBool>,
    handle: JoinHandle<SearchResult>,
}

struct MultiPonder {
    threads: Vec<PonderThread>,
}

impl MultiPonder {
    /// Start pondering the top N opponent candidate moves.
    /// `board` is the position AFTER our move was played.
    /// `is_white` is the color that will move next (our color, after opponent replies).
    fn start(
        board: &ChessBoard,
        conductor: &PieceConductor,
        is_white_next: bool,
        n: usize,
    ) -> Self {
        let opponent_white = !is_white_next;
        let mut board_for_gen = board.clone();
        let legal = get_all_legal_moves_for_color(&mut board_for_gen, conductor, opponent_white);

        if legal.is_empty() {
            return MultiPonder { threads: Vec::new() };
        }

        // Rank opponent moves: play each, evaluate from opponent's perspective,
        // pick the best N for the opponent.
        let mut scored: Vec<(ChessMove, i32)> = legal
            .into_iter()
            .map(|mut m| {
                let mut b = board.clone();
                b.make_move(&mut m);
                let eval = evaluate_board(&b, conductor);
                // Opponent wants low score if white_next, high if black_next
                let opponent_score = if opponent_white { eval } else { -eval };
                (m, opponent_score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.cmp(&a.1)); // best for opponent first
        scored.truncate(n);

        let threads: Vec<PonderThread> = scored
            .into_iter()
            .map(|(opp_mv, _score)| {
                let uci = mv_to_uci(opp_mv);
                let stop = Arc::new(AtomicBool::new(false));

                // Set up board: apply opponent's move, then search for our reply
                let mut board_c = board.clone();
                let mut mv_c = opp_mv;
                board_c.make_move(&mut mv_c);

                let cond_c = conductor.clone();
                let stop_c = Arc::clone(&stop);
                let handle = thread::spawn(move || {
                    iterative_deepening_root(
                        &mut board_c,
                        &cond_c,
                        None, // no opening book for ponder
                        64,
                        is_white_next,
                        None, // no deadline — stopped by flag
                        Some(stop_c),
                        0,
                    )
                });

                PonderThread {
                    opponent_uci: uci,
                    stop,
                    handle,
                }
            })
            .collect();

        MultiPonder { threads }
    }

    /// Check if any ponder thread matches the opponent's actual move.
    /// Stops all threads. Returns the SearchResult if there was a hit.
    fn resolve(self, actual_move: &str, movetime_ms: u64) -> Option<SearchResult> {
        // Find the matching thread (if any)
        let mut hit_idx = None;
        for (i, t) in self.threads.iter().enumerate() {
            if t.opponent_uci == actual_move {
                hit_idx = Some(i);
            } else {
                // Stop non-matching threads immediately
                t.stop.store(true, Ordering::Release);
            }
        }

        if let Some(idx) = hit_idx {
            // Let the matching thread continue for movetime_ms more
            let hit_stop = Arc::clone(&self.threads[idx].stop);
            let deadline = Instant::now() + Duration::from_millis(movetime_ms);

            // Wait for deadline or external conditions
            while Instant::now() < deadline {
                thread::sleep(Duration::from_millis(2));
            }
            hit_stop.store(true, Ordering::Release);
        }

        // Collect all results
        let mut hit_result = None;
        for (i, t) in self.threads.into_iter().enumerate() {
            let result = t.handle.join().unwrap_or(SearchResult {
                score: 0,
                best_move: None,
                ponder_move: None,
                total_nodes: 0,
            });
            if hit_idx == Some(i) {
                hit_result = Some(result);
            }
        }
        hit_result
    }

    /// Stop all threads without waiting for results.
    fn stop_all(self) {
        for t in &self.threads {
            t.stop.store(true, Ordering::Release);
        }
        for t in self.threads {
            let _ = t.handle.join();
        }
    }
}

// ── Game ──────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
enum GameResult {
    Engine1Wins,
    Engine2Wins,
    Draw,
}

/// Safety cap: if somehow 300 full moves pass the game is a draw.
const MAX_PLIES: usize = 600;

/// 50-move rule: 50 moves (100 half-moves) without a pawn push or capture.
const FIFTY_MOVE_PLIES: u32 = 100;

fn play_game(
    engine1: &mut Engine,
    engine2: &mut Engine,
    movetime_ms: u64,
    engine1_is_white: bool,
    conductor: &PieceConductor,
    start_fen: Option<&str>,
    ponder: bool,
) -> (GameResult, String, u32, u32) {  // result, reason, ponder_hits, ponder_attempts
    let mut board = if let Some(fen) = start_fen {
        let mut b = ChessBoard::new();
        b.set_from_fen(fen);
        b
    } else {
        ChessBoard::new()
    };
    let mut move_list: Vec<String> = Vec::new();
    let mut no_progress: u32 = 0; // half-moves since last pawn push or capture

    // Determine which color moves first based on the FEN active color.
    let first_is_white = board.is_white_active();

    engine1.new_game();
    engine2.new_game();

    // Multi-ponder state for engine1: pondering top N opponent candidate moves.
    let mut active_ponder: Option<MultiPonder> = None;
    let mut ponder_hits: u32 = 0;
    let mut ponder_attempts: u32 = 0;

    macro_rules! game_return {
        ($result:expr, $reason:expr) => {
            return ($result, $reason, ponder_hits, ponder_attempts)
        };
    }

    for ply in 0..MAX_PLIES {
        let is_white = if first_is_white { ply % 2 == 0 } else { ply % 2 != 0 };

        // ── Termination checks ──
        let legal = get_all_legal_moves_for_color(&mut board, conductor, is_white);

        if legal.is_empty() {
            if let Some(p) = active_ponder.take() { p.stop_all(); }
            if conductor.is_king_in_check(&board, is_white) {
                if is_white {
                    game_return!(if engine1_is_white { GameResult::Engine2Wins } else { GameResult::Engine1Wins },
                         "checkmate".to_string());
                } else {
                    game_return!(if engine1_is_white { GameResult::Engine1Wins } else { GameResult::Engine2Wins },
                         "checkmate".to_string());
                }
            } else {
                game_return!(GameResult::Draw, "stalemate".to_string());
            }
        }

        if board.is_repetition(3) {
            if let Some(p) = active_ponder.take() { p.stop_all(); }
            game_return!(GameResult::Draw, "repetition".to_string());
        }

        if no_progress >= FIFTY_MOVE_PLIES {
            if let Some(p) = active_ponder.take() { p.stop_all(); }
            game_return!(GameResult::Draw, "50-move rule".to_string());
        }

        // ── Build position command ──
        let position_base = if let Some(fen) = start_fen {
            format!("position fen {}", fen)
        } else {
            "position startpos".to_string()
        };
        let position_cmd = if move_list.is_empty() {
            position_base.clone()
        } else {
            format!("{} moves {}", position_base, move_list.join(" "))
        };

        // ── Ask the right engine ──
        let ask_engine1 = is_white == engine1_is_white;

        // Check multi-ponder hit for engine1
        let ponder_result = if ask_engine1 {
            if let Some(mp) = active_ponder.take() {
                ponder_attempts += 1;
                let last_move = move_list.last().map(|s| s.as_str()).unwrap_or("");
                let result = mp.resolve(last_move, movetime_ms);
                if result.is_some() { ponder_hits += 1; }
                result
            } else {
                None
            }
        } else {
            None
        };

        let uci_str = if let Some(ref result) = ponder_result {
            // Ponder hit! Use the result directly.
            result.best_move.map(|m| mv_to_uci(m))
        } else {
            // No ponder hit — ask the engine via UCI as usual.
            // If engine1 had a ponder running that missed, it was already
            // stopped by resolve() above (which stops all threads).
            let (mv, _) = if ask_engine1 {
                engine1.best_move(&position_cmd, movetime_ms)
            } else {
                engine2.best_move(&position_cmd, movetime_ms)
            };
            mv
        };

        let Some(uci_str) = uci_str else {
            if let Some(p) = active_ponder.take() { p.stop_all(); }
            game_return!(GameResult::Draw, "engine returned no move".to_string());
        };

        // ── Apply move ──
        let Some(mut chess_move) = parse_uci_move(&uci_str, &legal) else {
            if let Some(p) = active_ponder.take() { p.stop_all(); }
            game_return!(GameResult::Draw, format!("illegal move '{}'", uci_str));
        };
        let is_pawn = chess_move.chess_piece.map_or(false, |p| p.piece_type() == PieceType::Pawn);
        if is_pawn || chess_move.capture.is_some() {
            no_progress = 0;
        } else {
            no_progress += 1;
        }

        board.make_move(&mut chess_move);
        move_list.push(mv_to_uci(chess_move));

        // ── Start multi-ponder for engine1 after it moves ──
        if ponder && ask_engine1 {
            // board is now the position after engine1's move.
            // Ponder the opponent's top N candidate replies.
            active_ponder = Some(MultiPonder::start(
                &board,
                conductor,
                engine1_is_white, // engine1's color = the side that will move after opponent
                PONDER_CANDIDATES,
            ));
        }
    }

    if let Some(p) = active_ponder.take() { p.stop_all(); }
    (GameResult::Draw, "move limit".to_string(), ponder_hits, ponder_attempts)
}

// ── CLI ───────────────────────────────────────────────────────────────────────

fn main() {
    #[cfg(any(feature = "nn-full-forward", feature = "nn-incremental", feature = "runtime-switch"))]
    init_nn();

    let args: Vec<String> = std::env::args().collect();

    let mut engine1_path: Option<String> = None;
    let mut engine2_path: Option<String> = None;
    let mut num_games: usize = 20;
    let mut movetime_ms: u64 = 100;
    let mut start_fen: Option<String> = None;
    let mut ponder = true; // enabled by default
    let mut engine1_opts: Vec<String> = Vec::new();
    let mut engine2_opts: Vec<String> = Vec::new();

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
            "--fen" => {
                start_fen = args.get(i + 1).map(|s| s.to_string());
                i += 2;
            }
            "--no-ponder" => {
                ponder = false;
                i += 1;
            }
            "--ponder" => {
                ponder = true;
                i += 1;
            }
            "--engine1-opt" => {
                if let Some(v) = args.get(i + 1) {
                    engine1_opts.push(v.clone());
                }
                i += 2;
            }
            "--engine2-opt" => {
                if let Some(v) = args.get(i + 1) {
                    engine2_opts.push(v.clone());
                }
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
    engine1.init(&engine1_opts);
    engine2.init(&engine2_opts);

    let e1_name = engine1.name.clone();
    let e2_name = engine2.name.clone();

    println!("Engine 1 : {}", e1_name);
    println!("Engine 2 : {}", e2_name);
    println!("Games    : {}", num_games);
    println!("Movetime : {} ms", movetime_ms);
    println!("Ponder   : {}", if ponder { "on" } else { "off" });
    if let Some(ref fen) = start_fen {
        println!("Start FEN: {}", fen);
    }
    println!("{}", "─".repeat(60));

    let mut e1_wins = 0u32;
    let mut e2_wins = 0u32;
    let mut draws = 0u32;
    let mut total_hits = 0u32;
    let mut total_attempts = 0u32;

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

        let (result, reason, hits, attempts) =
            play_game(&mut engine1, &mut engine2, movetime_ms, engine1_is_white, &conductor, start_fen.as_deref(), ponder);

        total_hits += hits;
        total_attempts += attempts;

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
        let hit_rate = if total_attempts > 0 { total_hits as f64 / total_attempts as f64 * 100.0 } else { 0.0 };
        print!("       Score so far: {e1_wins}W / {draws}D / {e2_wins}L  ({score:.1}%)");
        if ponder && total_attempts > 0 {
            print!("  ponder: {total_hits}/{total_attempts} ({hit_rate:.0}%)");
        }
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

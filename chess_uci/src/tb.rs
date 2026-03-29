/// Syzygy endgame tablebase probing.
///
/// Call `init(path)` once (or on each `setoption name SyzygyPath`), then
/// call `probe_root` before starting a search when piece count is low.
use std::sync::{Mutex, OnceLock};

use chess_board::{ChessBoard, FENParser};
use chess_foundation::ChessMove;
use shakmaty::{Chess, CastlingMode};
use shakmaty::fen::Fen;
use shakmaty_syzygy::{Tablebase, Wdl};

static TB: OnceLock<Mutex<Option<Tablebase<Chess>>>> = OnceLock::new();

fn get_tb() -> &'static Mutex<Option<Tablebase<Chess>>> {
    TB.get_or_init(|| Mutex::new(None))
}

/// Load (or reload) Syzygy tablebase files from `path`.
/// Returns the number of tablebase files found, or an error string.
pub fn init(path: &str) -> Result<usize, String> {
    let mut tb: Tablebase<Chess> = Tablebase::new();
    let count = tb.add_directory(path).map_err(|e| e.to_string())?;
    if count == 0 {
        return Err(format!("no tablebase files found in '{path}'"));
    }
    *get_tb().lock().map_err(|e| e.to_string())? = Some(tb);
    Ok(count)
}

/// Maximum piece count the loaded tablebases can handle (0 if not loaded).
pub fn max_pieces() -> u32 {
    let guard = get_tb().lock().unwrap_or_else(|e| e.into_inner());
    guard.as_ref().map_or(0, |tb| tb.max_pieces() as u32)
}

/// Number of pieces currently on the board.
pub fn piece_count(board: &ChessBoard) -> u32 {
    board.get_all_pieces().count_ones()
}

fn board_to_chess(board: &ChessBoard) -> Option<Chess> {
    let fen_str = FENParser::board_to_fen(board);
    let fen: Fen = fen_str.parse().ok()?;
    fen.into_position(CastlingMode::Standard).ok()
}

fn wdl_rank(wdl: Wdl) -> i32 {
    match wdl {
        Wdl::Win         =>  2,
        Wdl::CursedWin   =>  1,
        Wdl::Draw        =>  0,
        Wdl::BlessedLoss => -1,
        Wdl::Loss        => -2,
    }
}

fn rank_to_score(rank: i32) -> i32 {
    match rank {
         2 => 20_000,
         1 =>    100,
         0 =>      0,
        -1 =>   -100,
         _ => -20_000,
    }
}

fn rank_to_label(rank: i32) -> &'static str {
    match rank {
         2 => "TB Win",
         1 => "TB CursedWin",
         0 => "TB Draw",
        -1 => "TB BlessedLoss",
         _ => "TB Loss",
    }
}

/// Probe all legal moves and return the best one according to the tablebase,
/// along with a `(score_cp, label)` pair for UCI `info` output.
///
/// Returns `None` if:
/// - no tablebases are loaded
/// - the position has too many pieces for the loaded tables
/// - probing fails for all legal moves
pub fn probe_root(
    board: &ChessBoard,
    legal_moves: &[ChessMove],
    probe_limit: u32,
) -> Option<(ChessMove, i32, &'static str)> {
    if piece_count(board) > probe_limit {
        return None;
    }

    let guard = get_tb().lock().ok()?;
    let tb = guard.as_ref()?;

    let mut best_mv: Option<ChessMove> = None;
    let mut best_rank: i32 = i32::MIN;
    let mut any_probed = false;

    for &mv in legal_moves {
        let mut board_clone = board.clone();
        let mut mv_mut = mv;
        if !board_clone.make_move(&mut mv_mut) {
            continue;
        }

        let next_chess = match board_to_chess(&board_clone) {
            Some(c) => c,
            None => continue,
        };

        // Probe WDL from the opponent's perspective after our move.
        // `probe_wdl_after_zeroing` gives precise WDL (unambiguous by 50-move rule).
        let opp_wdl = match tb.probe_wdl_after_zeroing(&next_chess) {
            Ok(w) => w,
            Err(_) => continue,
        };

        any_probed = true;
        // Negate: opponent's Win = our Loss, etc.
        let our_rank = -wdl_rank(opp_wdl);

        if our_rank > best_rank {
            best_rank = our_rank;
            best_mv = Some(mv);
        }
    }

    if !any_probed {
        return None;
    }

    best_mv.map(|mv| (mv, rank_to_score(best_rank), rank_to_label(best_rank)))
}

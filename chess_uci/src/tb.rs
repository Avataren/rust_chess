/// Syzygy endgame tablebase probing.
///
/// Call `init(path)` once (or on each `setoption name SyzygyPath`), then
/// call `probe_root` before starting a search when piece count is low.
use std::sync::{Mutex, OnceLock};

use chess_board::{ChessBoard, FENParser};
use chess_foundation::ChessMove;
use shakmaty::{Chess, CastlingMode, Move as SMove, Role};
use shakmaty::fen::Fen;
use shakmaty_syzygy::{Tablebase, Wdl, AmbiguousWdl};

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

/// Rank from `probe_wdl` output (respects halfmove clock).
/// MaybeWin/MaybeLoss are ambiguous; treat conservatively.
fn ambiguous_wdl_rank(wdl: AmbiguousWdl) -> i32 {
    match wdl {
        AmbiguousWdl::Win                              =>  2,
        AmbiguousWdl::MaybeWin | AmbiguousWdl::CursedWin =>  1,
        AmbiguousWdl::Draw                             =>  0,
        AmbiguousWdl::BlessedLoss | AmbiguousWdl::MaybeLoss => -1,
        AmbiguousWdl::Loss                             => -2,
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

/// Convert a shakmaty move to our ChessMove by matching from/to/promotion
/// against the provided legal moves.
///
/// Both encodings use rank*8 + file (a1=0, …, h8=63), so square indices match directly.
fn match_shakmaty_move(sm: &SMove, legal_moves: &[ChessMove]) -> Option<ChessMove> {
    let (from, to) = match sm {
        SMove::Normal { from, to, .. } | SMove::EnPassant { from, to } => {
            (u16::from(*from), u16::from(*to))
        }
        SMove::Castle { king, rook } => {
            let king_sq  = u16::from(*king);
            let rook_file = u16::from(*rook) % 8;
            let king_rank = king_sq / 8;
            // King goes to g-file (6) for kingside (rook on h=7), c-file (2) for queenside.
            let king_to = king_rank * 8 + if rook_file == 7 { 6 } else { 2 };
            (king_sq, king_to)
        }
        _ => return None,
    };

    let promo_role = if let SMove::Normal { promotion: Some(role), .. } = sm {
        Some(*role)
    } else {
        None
    };

    legal_moves.iter().find(|mv| {
        if mv.start_square() != from || mv.target_square() != to {
            return false;
        }
        match promo_role {
            Some(Role::Queen)  => mv.has_flag(ChessMove::PROMOTE_TO_QUEEN_FLAG),
            Some(Role::Knight) => mv.has_flag(ChessMove::PROMOTE_TO_KNIGHT_FLAG),
            Some(Role::Rook)   => mv.has_flag(ChessMove::PROMOTE_TO_ROOK_FLAG),
            Some(Role::Bishop) => mv.has_flag(ChessMove::PROMOTE_TO_BISHOP_FLAG),
            None               => !mv.is_promotion(),
            _                  => false,
        }
    }).copied()
}

/// Probe the tablebase at the root and return the best move.
///
/// Returns `None` if:
/// - no tablebases are loaded
/// - the position has too many pieces for the loaded tables
/// - probing fails entirely
///
/// Strategy (in order):
/// 1. **DTZ-based** (`best_move`): requires .rtbz files; selects the move that wins
///    fastest or delays a loss longest — prevents unnecessary sacrifices and
///    draw-by-repetition in winning endgames.
/// 2. **WDL fallback** (only .rtbw files loaded): among equal-outcome moves,
///    prefer those that retain more pieces to avoid accidental sacrifices.
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

    let chess_pos = board_to_chess(board)?;

    // WDL of the current position for the UCI info label.
    // probe_wdl respects the halfmove clock (CursedWin vs Win distinction).
    let rank = tb.probe_wdl(&chess_pos)
        .map(|w| ambiguous_wdl_rank(w))
        .unwrap_or(0);

    // --- Primary: DTZ-based move selection ---
    // best_move uses .rtbz tables to select the move that wins fastest
    // (minimum DTZ for winning positions) or delays longest (for losing ones).
    // This prevents both unnecessary piece sacrifices and repetition loops.
    if let Ok(Some((shakmaty_mv, _dtz))) = tb.best_move(&chess_pos) {
        if let Some(our_mv) = match_shakmaty_move(&shakmaty_mv, legal_moves) {
            return Some((our_mv, rank_to_score(rank), rank_to_label(rank)));
        }
    }

    // --- Fallback: WDL-only move selection (no .rtbz files) ---
    // Probe each resulting position; among equal-rank moves prefer those that
    // retain more pieces to avoid accidental queen sacrifices.
    let mut best_mv: Option<ChessMove> = None;
    let mut best_key: (i32, i32) = (i32::MIN, i32::MIN);
    let mut any_probed = false;

    for &mv in legal_moves {
        let mut board_clone = board.clone();
        let mut mv_mut = mv;
        if !board_clone.make_move(&mut mv_mut) {
            continue;
        }
        let next_chess = match board_to_chess(&board_clone) {
            Some(c) => c,
            None    => continue,
        };
        let opp_wdl = match tb.probe_wdl_after_zeroing(&next_chess) {
            Ok(w)  => w,
            Err(_) => continue,
        };
        any_probed = true;
        let our_rank = -wdl_rank(opp_wdl);
        let resulting_pieces = piece_count(&board_clone) as i32;
        // Tiebreak: prefer more pieces remaining to avoid unnecessary sacrifices.
        let key = (our_rank, resulting_pieces);
        if key > best_key {
            best_key = key;
            best_mv = Some(mv);
        }
    }

    if !any_probed {
        return None;
    }

    best_mv.map(|mv| (mv, rank_to_score(best_key.0), rank_to_label(best_key.0)))
}

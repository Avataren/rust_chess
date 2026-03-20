//! SAN (Standard Algebraic Notation) move parsing.
//!
//! Parses a SAN string into a [`ChessMove`] for the current board position.
//! Supports piece moves, pawn moves, captures (`x`), promotions (`=Q` etc.),
//! castling (`O-O` / `O-O-O`), and disambiguation by file or rank.
//! Check (`+`) and checkmate (`#`) suffixes are stripped automatically.

use chess_board::ChessBoard;
use chess_foundation::{piece::PieceType, ChessMove};

use crate::move_generator::get_pseudo_legal_move_list_from_square;
use crate::piece_conductor::PieceConductor;

/// Convert an algebraic square string (e.g. `"e4"`) to a board square index.
///
/// Square indexing: a1 = 0, h1 = 7, a2 = 8, …, h8 = 63.
pub fn san_to_square(sq: &str) -> Option<u16> {
    let mut chars = sq.chars();
    let file = chars.next()?;
    let rank = chars.next()?.to_digit(10)?;
    if !('a'..='h').contains(&file) || !(1..=8).contains(&rank) {
        return None;
    }
    let file_idx = file as u16 - 'a' as u16;
    let rank_idx = rank as u16 - 1;
    Some(rank_idx * 8 + file_idx)
}

/// Convert a board square index to its algebraic name (e.g. `28` → `"e4"`).
pub fn square_to_san(sq: u16) -> String {
    let file = (sq % 8) as u8 + b'a';
    let rank = (sq / 8) as u8 + b'1';
    format!("{}{}", file as char, rank as char)
}

fn char_to_piece_type(c: char) -> Option<PieceType> {
    match c {
        'N' => Some(PieceType::Knight),
        'B' => Some(PieceType::Bishop),
        'R' => Some(PieceType::Rook),
        'Q' => Some(PieceType::Queen),
        'K' => Some(PieceType::King),
        _ => None,
    }
}

fn promotion_flag(c: char) -> Option<u16> {
    match c.to_ascii_uppercase() {
        'Q' => Some(ChessMove::PROMOTE_TO_QUEEN_FLAG),
        'R' => Some(ChessMove::PROMOTE_TO_ROOK_FLAG),
        'B' => Some(ChessMove::PROMOTE_TO_BISHOP_FLAG),
        'N' => Some(ChessMove::PROMOTE_TO_KNIGHT_FLAG),
        _ => None,
    }
}

fn handle_castling(board: &ChessBoard, san: &str) -> Option<ChessMove> {
    let is_white = board.is_white_active();
    let (king_start, king_target) = match (san, is_white) {
        ("O-O", true) => (4u16, 6u16),
        ("O-O-O", true) => (4u16, 2u16),
        ("O-O", false) => (60u16, 62u16),
        ("O-O-O", false) => (60u16, 58u16),
        _ => return None,
    };
    Some(ChessMove::new_with_flag(king_start, king_target, ChessMove::CASTLE_FLAG))
}

/// Find the starting square of the piece matching the given criteria.
///
/// Iterates over pseudo-legal moves from all friendly pieces of the given type,
/// applying optional file/rank disambiguation hints.  Returns `None` if no
/// unique match is found.
fn find_start_square(
    board: &ChessBoard,
    piece_type: PieceType,
    target_square: u16,
    is_capture: bool,
    hint_file: Option<char>,
    hint_rank: Option<char>,
) -> Option<u16> {
    let is_white = board.is_white_active();
    let color_bb = if is_white { board.get_white() } else { board.get_black() };
    let piece_bb = match piece_type {
        PieceType::Pawn => board.get_pawns() & color_bb,
        PieceType::Knight => board.get_knights() & color_bb,
        PieceType::Bishop => board.get_bishops() & color_bb,
        PieceType::Rook => board.get_rooks() & color_bb,
        PieceType::Queen => board.get_queens() & color_bb,
        PieceType::King => board.get_kings() & color_bb,
        PieceType::None => return None,
    };

    let conductor = PieceConductor::new();

    // Whether there is an opponent piece on the target square (regular capture).
    let opponent_on_target = board
        .get_piece_at_square(target_square)
        .map_or(false, |p| p.is_white() != is_white);

    let mut found: Option<u16> = None;

    for sq in 0u16..64 {
        if !piece_bb.contains_square(sq as i32) {
            continue;
        }
        // Apply file disambiguation hint
        if let Some(f) = hint_file {
            if (sq % 8) as u8 + b'a' != f as u8 {
                continue;
            }
        }
        // Apply rank disambiguation hint
        if let Some(r) = hint_rank {
            if (sq / 8) as u8 + b'1' != r as u8 {
                continue;
            }
        }

        let pseudo_moves =
            get_pseudo_legal_move_list_from_square(sq, board, &conductor, is_white);

        for mv in pseudo_moves {
            if mv.target_square() != target_square {
                continue;
            }
            let is_ep = mv.has_flag(ChessMove::EN_PASSANT_CAPTURE_FLAG);
            // Reject if SAN marks a capture but neither opponent piece nor EP
            if is_capture && !opponent_on_target && !is_ep {
                continue;
            }
            // Reject quiet pawn move if there is an opponent on the target
            if !is_capture && piece_type == PieceType::Pawn && opponent_on_target {
                continue;
            }
            if found.is_some() {
                // Ambiguous — cannot resolve without legality filter
                return None;
            }
            found = Some(sq);
        }
    }
    found
}

/// Parse a SAN move string and return the corresponding [`ChessMove`] for the
/// current board position.
///
/// # Examples
/// ```ignore
/// let mv = get_move_from_san(&board, "e4");
/// let mv = get_move_from_san(&board, "Nxf3");
/// let mv = get_move_from_san(&board, "e8=Q");
/// let mv = get_move_from_san(&board, "O-O");
/// ```
pub fn get_move_from_san(board: &ChessBoard, san: &str) -> Option<ChessMove> {
    // Strip check/checkmate annotations
    let san = san.trim_end_matches(|c| c == '+' || c == '#');

    if san == "O-O" || san == "O-O-O" {
        return handle_castling(board, san);
    }

    let mut chars = san.chars().peekable();

    // Leading uppercase letter → named piece; otherwise pawn move
    let mut piece_type = PieceType::Pawn;
    if let Some(&first) = chars.peek() {
        if let Some(pt) = char_to_piece_type(first) {
            piece_type = pt;
            chars.next();
        }
    }

    let rest: String = chars.collect();

    // Extract promotion suffix "=X"
    let (rest, promo_flag) = if let Some(eq_pos) = rest.find('=') {
        let promo_char = rest.chars().nth(eq_pos + 1)?;
        let flag = promotion_flag(promo_char)?;
        (rest[..eq_pos].to_string(), Some(flag))
    } else {
        (rest, None)
    };

    // Split on capture marker 'x'
    let (pre_x, is_capture, post_x) = if let Some(x_pos) = rest.find('x') {
        (rest[..x_pos].to_string(), true, rest[x_pos + 1..].to_string())
    } else {
        (String::new(), false, rest)
    };

    // Target square is always the last two characters
    if post_x.len() < 2 {
        return None;
    }
    let target_str = &post_x[post_x.len() - 2..];
    let target_square = san_to_square(target_str)?;

    // Disambiguation characters: everything except the target square
    let disambiguation = format!("{}{}", pre_x, &post_x[..post_x.len() - 2]);
    let hint_file = disambiguation.chars().find(|c| ('a'..='h').contains(c));
    let hint_rank = disambiguation.chars().find(|c| c.is_ascii_digit());

    let start_square =
        find_start_square(board, piece_type, target_square, is_capture, hint_file, hint_rank)?;

    let flag = promo_flag.unwrap_or(ChessMove::NO_FLAG);
    Some(ChessMove::new_with_flag(start_square, target_square, flag))
}

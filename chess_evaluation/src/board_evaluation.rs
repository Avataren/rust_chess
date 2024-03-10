use chess_board::ChessBoard;
use chess_foundation::{piece::PieceType, Bitboard};

use crate::evaluate_pawn_position;

const PAWN_VALUE: i32 = 100;
const KNIGHT_VALUE: i32 = 320;
const BISHOP_VALUE: i32 = 330;
const ROOK_VALUE: i32 = 500;
const QUEEN_VALUE: i32 = 900;
const KING_VALUE: i32 = 20000; // Arbitrary high value since the game ends if the king is captured.

/// Evaluates the chess board state and returns a score from the perspective of the active player.
pub fn evaluate_board(chess_board: &ChessBoard) -> i32 {
    let mut score = 0;

    // Material value calculation
    score += chess_board.get_pawns().count_ones() as i32 * PAWN_VALUE;
    score += chess_board.get_knights().count_ones() as i32 * KNIGHT_VALUE;
    score += chess_board.get_bishops().count_ones() as i32 * BISHOP_VALUE;
    score += chess_board.get_rooks().count_ones() as i32 * ROOK_VALUE;
    score += chess_board.get_queens().count_ones() as i32 * QUEEN_VALUE;
    score += chess_board.get_kings().count_ones() as i32 * KING_VALUE;
    let is_white = chess_board.is_white_active();

    // Adjust score based on the active color
    if is_white {
        // Subtract the value of black pieces
        score -= evaluate_board_for_color(chess_board, false);
    } else {
        // Subtract the value of white pieces and invert the score
        score = -score + evaluate_board_for_color(chess_board, true);
    }

    let mut pawns = chess_board.get_pawns();
    if is_white {
        pawns = pawns & chess_board.get_white();
    } else {
        pawns = pawns & chess_board.get_black();
    }
    // Evaluate pawns
    while pawns != Bitboard::default() {
        let square = pawns.pop_lsb() as usize;
        score += evaluate_pawn_position(square, is_white);
    }

    score
}

/// Helper function to evaluate the board for a specific color.
pub fn evaluate_board_for_color(chess_board: &ChessBoard, is_white: bool) -> i32 {
    let mut score = 0;
    let color_board = if is_white {
        chess_board.get_white()
    } else {
        chess_board.get_black()
    };

    score += (color_board & chess_board.get_pawns()).count_ones() as i32 * PAWN_VALUE;
    score += (color_board & chess_board.get_knights()).count_ones() as i32 * KNIGHT_VALUE;
    score += (color_board & chess_board.get_bishops()).count_ones() as i32 * BISHOP_VALUE;
    score += (color_board & chess_board.get_rooks()).count_ones() as i32 * ROOK_VALUE;
    score += (color_board & chess_board.get_queens()).count_ones() as i32 * QUEEN_VALUE;
    score += (color_board & chess_board.get_kings()).count_ones() as i32 * KING_VALUE;

    score
}

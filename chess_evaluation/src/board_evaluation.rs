use chess_board::ChessBoard;
use chess_foundation::Bitboard;

use crate::{evaluate_knight_position, evaluate_pawn_position};

const PAWN_VALUE: i32 = 100;
const KNIGHT_VALUE: i32 = 320;
const BISHOP_VALUE: i32 = 330;
const ROOK_VALUE: i32 = 500;
const QUEEN_VALUE: i32 = 900;
const KING_VALUE: i32 = 20000; // Arbitrary high value since the game ends if the king is captured.

/// Evaluates the chess board state and returns a score from the perspective of the active player.
fn evaluate_material(pieces: Bitboard, value: i32) -> i32 {
    pieces.count_ones() as i32 * value
}

/// Evaluates the chess board and returns an absolute score:
/// positive = white is ahead, negative = black is ahead.
pub fn evaluate_board(chess_board: &ChessBoard) -> i32 {
    let white_board = chess_board.get_white();
    let black_board = chess_board.get_black();

    let mut score = 0;

    // White material (positive contribution)
    score += evaluate_material(white_board & chess_board.get_pawns(), PAWN_VALUE);
    score += evaluate_material(white_board & chess_board.get_knights(), KNIGHT_VALUE);
    score += evaluate_material(white_board & chess_board.get_bishops(), BISHOP_VALUE);
    score += evaluate_material(white_board & chess_board.get_rooks(), ROOK_VALUE);
    score += evaluate_material(white_board & chess_board.get_queens(), QUEEN_VALUE);
    score += evaluate_material(white_board & chess_board.get_kings(), KING_VALUE);

    // Black material (negative contribution)
    score -= evaluate_material(black_board & chess_board.get_pawns(), PAWN_VALUE);
    score -= evaluate_material(black_board & chess_board.get_knights(), KNIGHT_VALUE);
    score -= evaluate_material(black_board & chess_board.get_bishops(), BISHOP_VALUE);
    score -= evaluate_material(black_board & chess_board.get_rooks(), ROOK_VALUE);
    score -= evaluate_material(black_board & chess_board.get_queens(), QUEEN_VALUE);
    score -= evaluate_material(black_board & chess_board.get_kings(), KING_VALUE);

    // White piece-square tables (positive)
    let mut white_pawns = white_board & chess_board.get_pawns();
    while white_pawns != Bitboard::default() {
        let square = white_pawns.pop_lsb() as usize;
        score += evaluate_pawn_position(square, true);
    }
    let mut white_knights = white_board & chess_board.get_knights();
    while white_knights != Bitboard::default() {
        let square = white_knights.pop_lsb() as usize;
        score += evaluate_knight_position(square, true);
    }

    // Black piece-square tables (negative)
    let mut black_pawns = black_board & chess_board.get_pawns();
    while black_pawns != Bitboard::default() {
        let square = black_pawns.pop_lsb() as usize;
        score -= evaluate_pawn_position(square, false);
    }
    let mut black_knights = black_board & chess_board.get_knights();
    while black_knights != Bitboard::default() {
        let square = black_knights.pop_lsb() as usize;
        score -= evaluate_knight_position(square, false);
    }

    score
}

// Helper function to evaluate the board for a specific color.
// pub fn evaluate_board_for_color(chess_board: &ChessBoard, is_white: bool) -> i32 {
//     let mut score = 0;
//     let color_board = if is_white {
//         chess_board.get_white()
//     } else {
//         chess_board.get_black()
//     };

//     score += (color_board & chess_board.get_pawns()).count_ones() as i32 * PAWN_VALUE;
//     score += (color_board & chess_board.get_knights()).count_ones() as i32 * KNIGHT_VALUE;
//     score += (color_board & chess_board.get_bishops()).count_ones() as i32 * BISHOP_VALUE;
//     score += (color_board & chess_board.get_rooks()).count_ones() as i32 * ROOK_VALUE;
//     score += (color_board & chess_board.get_queens()).count_ones() as i32 * QUEEN_VALUE;
//     score += (color_board & chess_board.get_kings()).count_ones() as i32 * KING_VALUE;

//     score
// }

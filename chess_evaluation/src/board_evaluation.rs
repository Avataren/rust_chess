use chess_board::ChessBoard;
use chess_foundation::{piece::PieceType, Bitboard};

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

/// Evaluates the chess board state and returns a score from the perspective of the active player.
pub fn evaluate_board(chess_board: &ChessBoard) -> i32 {
    let active_color_board = if chess_board.is_white_active() {
        chess_board.get_white()
    } else {
        chess_board.get_black()
    };
    let opponent_color_board = if chess_board.is_white_active() {
        chess_board.get_black()
    } else {
        chess_board.get_white()
    };

    let mut score = 0;

    // Calculate material value for active color
    score += evaluate_material(active_color_board & chess_board.get_pawns(), PAWN_VALUE);
    score += evaluate_material(active_color_board & chess_board.get_knights(), KNIGHT_VALUE);
    score += evaluate_material(active_color_board & chess_board.get_bishops(), BISHOP_VALUE);
    score += evaluate_material(active_color_board & chess_board.get_rooks(), ROOK_VALUE);
    score += evaluate_material(active_color_board & chess_board.get_queens(), QUEEN_VALUE);
    score += evaluate_material(active_color_board & chess_board.get_kings(), KING_VALUE);

    // Calculate material value for the opponent and subtract from score
    score -= evaluate_material(opponent_color_board & chess_board.get_pawns(), PAWN_VALUE);
    score -= evaluate_material(
        opponent_color_board & chess_board.get_knights(),
        KNIGHT_VALUE,
    );
    score -= evaluate_material(
        opponent_color_board & chess_board.get_bishops(),
        BISHOP_VALUE,
    );
    score -= evaluate_material(opponent_color_board & chess_board.get_rooks(), ROOK_VALUE);
    score -= evaluate_material(opponent_color_board & chess_board.get_queens(), QUEEN_VALUE);
    score -= evaluate_material(opponent_color_board & chess_board.get_kings(), KING_VALUE);

    // Evaluate pawns
    let mut pawns = active_color_board & chess_board.get_pawns();
    while pawns != Bitboard::default() {
        let square = pawns.pop_lsb() as usize;
        score += evaluate_pawn_position(square, chess_board.is_white_active());
    }

    let mut knights = active_color_board & chess_board.get_knights();
    while knights != Bitboard::default() {
        let square = knights.pop_lsb() as usize;
        score += evaluate_knight_position(square, chess_board.is_white_active());
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

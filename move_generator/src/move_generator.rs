use chess_board::ChessBoard;
use chess_foundation::ChessMove;

use crate::magic::Magic;

pub fn get_move_list_from_square(
    square: u16,
    chess_board: &ChessBoard,
    is_white: bool,
    magic: &Magic,
) -> Vec<ChessMove> {
    let mut move_list = Vec::new();

    let friendly_pieces_bitboard = if is_white {
        chess_board.get_white()
    } else {
        chess_board.get_black()
    };

    let relevant_blockers = friendly_pieces_bitboard;

    if chess_board.get_rooks().contains_square(square as i32) {
        move_list = magic.get_rook_moves(square, relevant_blockers, chess_board);
    } else if chess_board.get_bishops().contains_square(square as i32) {
        move_list = magic.get_bishop_moves(square, relevant_blockers, chess_board);
    } else if chess_board.get_queens().contains_square(square as i32) {
        move_list = magic.get_rook_moves(square, relevant_blockers, chess_board);;
        move_list.extend(magic.get_bishop_moves(square, relevant_blockers, chess_board));
    } else if chess_board.get_kings().contains_square(square as i32) {
        move_list = magic.get_king_moves(square, relevant_blockers);
    } else if chess_board.get_knights().contains_square(square as i32) {
        move_list = magic.get_knight_moves(square, relevant_blockers);
    }

    move_list
}

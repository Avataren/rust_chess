use crate::magic::Magic;
use chess_board::{chessboard::GameState, ChessBoard};
use chess_foundation::{Bitboard, ChessMove};

pub fn get_legal_move_list_from_square(
    square: u16,
    chess_board: &mut ChessBoard,
    magic: &Magic,
) -> Vec<ChessMove> {
    let mut move_list = Vec::new();
    let is_white = chess_board.get_white().contains_square(square as i32);
    let pseudo_legal_moves =
        get_pseudo_legal_move_list_from_square(square, chess_board, magic, is_white);
    // check if move would leave king in check
    for mut chess_move in pseudo_legal_moves {
        chess_board.make_move(&mut chess_move);
        let in_check = magic.is_king_in_check(chess_board, is_white);
        if !in_check {
            move_list.push(chess_move);
        }
        chess_board.undo_move();
    }

    move_list
}

pub fn get_legal_move_list_from_square_perft(
    square: u16,
    chess_board: &mut ChessBoard,
    magic: &Magic,
) -> Vec<ChessMove> {
    let mut move_list = Vec::new();
    let is_white = chess_board.get_white().contains_square(square as i32);
    let pseudo_legal_moves =
        get_pseudo_legal_move_list_from_square(square, chess_board, magic, is_white);
    // check if move would leave king in check
    for mut chess_move in pseudo_legal_moves {
        chess_board.make_move(&mut chess_move);
        let in_check = magic.is_king_in_check(chess_board, is_white);
        if !in_check {
            if chess_move.promotion_piece_type().is_some() {
                chess_move.set_flag(ChessMove::PROMOTE_TO_QUEEN_FLAG);
                move_list.push(chess_move);
                chess_move.set_flag(ChessMove::PROMOTE_TO_ROOK_FLAG);
                move_list.push(chess_move);
                chess_move.set_flag(ChessMove::PROMOTE_TO_KNIGHT_FLAG);
                move_list.push(chess_move);
                chess_move.set_flag(ChessMove::PROMOTE_TO_BISHOP_FLAG);
                move_list.push(chess_move);
            } else {
                move_list.push(chess_move);
            }
        }
        chess_board.undo_move();
    }

    move_list
}

pub fn get_all_legal_moves_for_color(
    chess_board: &mut ChessBoard,
    magic: &Magic,
    is_white: bool,
) -> Vec<ChessMove> {
    let mut move_list = Vec::new();
    let mut friendly_pieces_bitboard = if is_white {
        chess_board.get_white()
    } else {
        chess_board.get_black()
    };

    while (friendly_pieces_bitboard != Bitboard::default()) {
        let square = friendly_pieces_bitboard.pop_lsb() as u16;
        let pseudo_legal_moves =
            get_pseudo_legal_move_list_from_square(square, chess_board, magic, is_white);
        // check if move would leave king in check
        for mut chess_move in pseudo_legal_moves {
            chess_board.make_move(&mut chess_move);
            let in_check = magic.is_king_in_check(chess_board, is_white);
            if !in_check {
                move_list.push(chess_move);
            }
            chess_board.undo_move();
        }
    }

    move_list
}

pub fn get_pseudo_legal_move_list_from_square(
    square: u16,
    chess_board: &ChessBoard,
    magic: &Magic,
    is_white: bool,
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
        move_list = magic.get_rook_moves(square, relevant_blockers, chess_board);
        move_list.extend(magic.get_bishop_moves(square, relevant_blockers, chess_board));
    } else if chess_board.get_kings().contains_square(square as i32) {
        move_list = magic.get_king_moves(square, relevant_blockers, chess_board, is_white);
    } else if chess_board.get_knights().contains_square(square as i32) {
        move_list = magic.get_knight_moves(square, relevant_blockers);
    } else if chess_board.get_pawns().contains_square(square as i32) {
        move_list = magic.get_pawn_moves(square, is_white, chess_board);
    }

    move_list
}

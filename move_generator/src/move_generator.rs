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

use rayon::prelude::*;
use std::sync::Arc;

pub fn get_all_legal_moves_for_color_threaded(
    chess_board: &ChessBoard, // Notice it's not mutable anymore
    magic: &Magic,
    is_white: bool,
) -> Vec<ChessMove> {
    let mut move_list = Vec::new();
    let friendly_pieces_bitboard = if is_white {
        chess_board.get_white()
    } else {
        chess_board.get_black()
    };

    // Collect all squares into a Vec because rayon works on iterators
    let mut squares = Vec::new();
    let mut temp_bitboard = friendly_pieces_bitboard;
    while temp_bitboard != Bitboard::default() {
        squares.push(temp_bitboard.pop_lsb() as u16);
    }

    // Use an atomic reference counter to safely share 'magic' across threads
    // let magic = Arc::new(*magic);

    // Parallelize the processing of each square
    let moves: Vec<ChessMove> = squares
        .into_par_iter()
        .flat_map(|square| {
            let mut local_board = chess_board.clone(); // Clone the board for thread safety

            let pseudo_legal_moves =
                get_pseudo_legal_move_list_from_square(square, &mut local_board, &magic, is_white);

            pseudo_legal_moves
                .into_iter()
                .filter_map(move |mut chess_move| {
                    local_board.make_move(&mut chess_move);
                    let in_check = magic.is_king_in_check(&local_board, is_white);
                    local_board.undo_move();
                    if !in_check {
                        Some(chess_move)
                    } else {
                        None
                    }
                })
                .collect::<Vec<ChessMove>>()
        })
        .collect();

    move_list.extend(moves);

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

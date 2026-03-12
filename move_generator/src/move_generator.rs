use crate::piece_conductor::PieceConductor;
use chess_board::ChessBoard;
use chess_foundation::{Bitboard, ChessMove};

pub fn get_legal_move_list_from_square(
    square: u16,
    chess_board: &mut ChessBoard,
    conductor: &PieceConductor,
) -> Vec<ChessMove> {
    if chess_board.is_white_active() && !chess_board.get_white().contains_square(square as i32) {
        return Vec::new();
    }

    if !chess_board.is_white_active() && !chess_board.get_black().contains_square(square as i32) {
        return Vec::new();
    }

    let mut move_list = Vec::new();
    let is_white = chess_board.get_white().contains_square(square as i32);
    let pseudo_legal_moves =
        get_pseudo_legal_move_list_from_square(square, chess_board, conductor, is_white);
    // check if move would leave king in check
    for mut chess_move in pseudo_legal_moves {
        chess_board.make_move(&mut chess_move);
        let in_check = conductor.is_king_in_check(chess_board, is_white);
        if !in_check {
            move_list.push(chess_move);
        }
        chess_board.undo_move();
    }

    move_list
}

pub fn get_all_legal_moves_for_color(
    chess_board: &mut ChessBoard,
    conductor: &PieceConductor,
    is_white: bool,
) -> Vec<ChessMove> {
    let mut move_list = Vec::new();
    let mut friendly_pieces_bitboard = if is_white {
        chess_board.get_white()
    } else {
        chess_board.get_black()
    };

    while friendly_pieces_bitboard != Bitboard::default() {
        let square = friendly_pieces_bitboard.pop_lsb() as u16;
        let pseudo_legal_moves =
            get_pseudo_legal_move_list_from_square(square, chess_board, conductor, is_white);
        // check if move would leave king in check
        for mut chess_move in pseudo_legal_moves {
            chess_board.make_move(&mut chess_move);
            let in_check = conductor.is_king_in_check(chess_board, is_white);
            if !in_check {
                move_list.push(chess_move);
            }
            chess_board.undo_move();
        }
    }

    move_list
}

/// Like `get_all_legal_moves_for_color` but only returns captures (and promotions).
/// Skips the make_move/undo_move legality check for quiet moves, which makes
/// quiescence search significantly cheaper.
pub fn get_all_legal_captures_for_color(
    chess_board: &mut ChessBoard,
    conductor: &PieceConductor,
    is_white: bool,
) -> Vec<ChessMove> {
    let mut move_list = Vec::new();
    let mut friendly_pieces_bitboard = if is_white {
        chess_board.get_white()
    } else {
        chess_board.get_black()
    };

    // The capture field is populated *by* make_move, not before it, so we detect
    // captures via the enemy-piece bitboard before paying the legality-check cost.
    let enemy = if is_white { chess_board.get_black() } else { chess_board.get_white() };

    while friendly_pieces_bitboard != Bitboard::default() {
        let square = friendly_pieces_bitboard.pop_lsb() as u16;
        let pseudo_legal_moves =
            get_pseudo_legal_move_list_from_square(square, chess_board, conductor, is_white);
        for mut chess_move in pseudo_legal_moves {
            // A move is a capture if the target square holds an enemy piece,
            // or if it carries the en-passant flag (captured pawn is off the target square).
            let is_capture = enemy.contains_square(chess_move.target_square() as i32)
                || chess_move.has_flag(ChessMove::EN_PASSANT_CAPTURE_FLAG);
            if !is_capture {
                continue;
            }
            chess_board.make_move(&mut chess_move);
            let in_check = conductor.is_king_in_check(chess_board, is_white);
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
    conductor: &PieceConductor,
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
        move_list = conductor.get_rook_moves(square, relevant_blockers, chess_board);
    } else if chess_board.get_bishops().contains_square(square as i32) {
        move_list = conductor.get_bishop_moves(square, relevant_blockers, chess_board);
    } else if chess_board.get_queens().contains_square(square as i32) {
        move_list = conductor.get_rook_moves(square, relevant_blockers, chess_board);
        move_list.extend(conductor.get_bishop_moves(square, relevant_blockers, chess_board));
    } else if chess_board.get_kings().contains_square(square as i32) {
        move_list = conductor.get_king_moves(square, relevant_blockers, chess_board, is_white);
    } else if chess_board.get_knights().contains_square(square as i32) {
        move_list = conductor.get_knight_moves(square, relevant_blockers);
    } else if chess_board.get_pawns().contains_square(square as i32) {
        move_list = conductor.get_pawn_moves(square, is_white, chess_board);
    }

    move_list
}

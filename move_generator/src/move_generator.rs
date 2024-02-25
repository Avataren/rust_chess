use chess_board::ChessBoard;
use chess_foundation::{Bitboard, ChessMove};

use crate::magic::Magic;

pub fn get_move_list_from_square(
    square: u16,
    chess_board: &ChessBoard,
    magic: &Magic,
) -> Vec<ChessMove> {
    let mut move_list = Vec::new();

    let is_white = chess_board.get_white().contains_square(square as i32);

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
        move_list = magic.get_king_moves(square, relevant_blockers);
    } else if chess_board.get_knights().contains_square(square as i32) {
        move_list = magic.get_knight_moves(square, relevant_blockers);
    } else if chess_board.get_pawns().contains_square(square as i32) {
        move_list = get_pawn_moves(square, relevant_blockers, is_white, chess_board);
    }

    move_list
}

pub fn get_pawn_moves(
    square: u16,
    relevant_blockers: Bitboard,
    is_white: bool,
    chess_board: &ChessBoard,
) -> Vec<ChessMove> {
    let mut move_list = Vec::new();

    // Calculate rank and file for the pawn
    let rank = square / 8;
    let file = square % 8;

    // Single move forward
    let single_step = if is_white { 8 } else { -8 };
    let single_move_pos = square as i32 + single_step;

    if chess_board.get_empty().contains_square(single_move_pos) {
        move_list.push(ChessMove::new(square, single_move_pos as u16));
    }

    // Double move forward
    let starting_rank = if is_white { 1 } else { 6 };
    if rank == starting_rank {
        let double_step = single_step * 2;
        let double_move_pos = square as i32 + double_step;
        // Check if both the single step and double step positions are unoccupied
        if chess_board.get_empty().contains_square(single_move_pos)
            && chess_board.get_empty().contains_square(double_move_pos)
        {
            move_list.push(ChessMove::new_with_flag(
                square,
                double_move_pos as u16,
                ChessMove::PAWN_TWO_UP_FLAG,
            ));
        }
    }

    // Captures
    let capture_offsets = if is_white { [7, 9] } else { [-9, -7] };
    for &offset in capture_offsets.iter() {
        let capture_pos = square as i32 + offset;

        if capture_pos >= 0
            && capture_pos < 64
            && ((file > 0 && offset < 0) || (file < 7 && offset > 0))
        {
            if !chess_board.get_empty().contains_square(capture_pos)
                && !relevant_blockers.contains_square(capture_pos)
            {
                move_list.push(ChessMove::new(square, capture_pos as u16));
            }
        }
    }

    // En Passant
    let last_move = chess_board.get_last_move();
    if let Some(last_move) = last_move {
        // Check if the last move was a pawn double step
        if last_move.has_flag(ChessMove::PAWN_TWO_UP_FLAG) {
            let last_move_to = last_move.target_square();

            // Calculate the file of the last move's destination
            let last_move_to_file = last_move_to % 8; // Assuming 0-indexed files

            // Check if your pawn is on the correct rank and adjacent to the last move's destination
            let pawn_rank = square / 8; // Assuming 0-indexed ranks
            let pawn_file = square % 8; // Assuming 0-indexed files

            let en_passant_rank = if is_white { 4 } else { 3 }; // 5th rank from each player's perspective, 0-indexed
            let is_adjacent_file = (pawn_file as i32 - last_move_to_file as i32).abs() == 1;

            if pawn_rank == en_passant_rank && is_adjacent_file {
                // Determine the en passant capture square (which is one rank behind the last move's to square)
                let en_passant_capture_square = if is_white { last_move_to + 8 } else { last_move_to - 8 };

                // Add the en passant move to the move list
                move_list.push(ChessMove::new_with_flag(
                    square,
                    en_passant_capture_square as u16,
                    ChessMove::EN_PASSANT_CAPTURE_FLAG,
                ));
            }
        }
    }


    move_list
}

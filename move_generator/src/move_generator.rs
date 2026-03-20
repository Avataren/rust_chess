use crate::piece_conductor::PieceConductor;
use chess_board::ChessBoard;
use chess_foundation::{Bitboard, ChessMove};

#[inline]
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
    let mut result = Vec::new();
    let mut pseudo_buf = Vec::new();
    let is_white = chess_board.get_white().contains_square(square as i32);
    get_pseudo_legal_move_list_from_square(square, chess_board, conductor, is_white, &mut pseudo_buf);
    for mut chess_move in pseudo_buf.drain(..) {
        chess_board.make_move(&mut chess_move);
        if !conductor.is_king_in_check(chess_board, is_white) {
            result.push(chess_move);
        }
        chess_board.undo_move();
    }
    result
}

pub fn get_all_legal_moves_for_color(
    chess_board: &mut ChessBoard,
    conductor: &PieceConductor,
    is_white: bool,
    result: &mut Vec<ChessMove>,
    pseudo_buf: &mut Vec<ChessMove>,
) {
    result.clear();
    let mut friendly_pieces_bitboard = if is_white {
        chess_board.get_white()
    } else {
        chess_board.get_black()
    };
    while friendly_pieces_bitboard != Bitboard::default() {
        let square = friendly_pieces_bitboard.pop_lsb() as u16;
        pseudo_buf.clear();
        get_pseudo_legal_move_list_from_square(square, chess_board, conductor, is_white, pseudo_buf);
        for mut chess_move in pseudo_buf.drain(..) {
            chess_board.make_move(&mut chess_move);
            if !conductor.is_king_in_check(chess_board, is_white) {
                result.push(chess_move);
            }
            chess_board.undo_move();
        }
    }
}

pub fn get_all_legal_captures_for_color(
    chess_board: &mut ChessBoard,
    conductor: &PieceConductor,
    is_white: bool,
    result: &mut Vec<ChessMove>,
    pseudo_buf: &mut Vec<ChessMove>,
) {
    result.clear();
    let mut friendly_pieces_bitboard = if is_white {
        chess_board.get_white()
    } else {
        chess_board.get_black()
    };
    let enemy = if is_white { chess_board.get_black() } else { chess_board.get_white() };
    while friendly_pieces_bitboard != Bitboard::default() {
        let square = friendly_pieces_bitboard.pop_lsb() as u16;
        pseudo_buf.clear();
        get_pseudo_legal_move_list_from_square(square, chess_board, conductor, is_white, pseudo_buf);
        for mut chess_move in pseudo_buf.drain(..) {
            let is_capture = enemy.contains_square(chess_move.target_square() as i32)
                || chess_move.has_flag(ChessMove::EN_PASSANT_CAPTURE_FLAG)
                || chess_move.has_flag(ChessMove::PROMOTE_TO_QUEEN_FLAG);
            if !is_capture { continue; }
            chess_board.make_move(&mut chess_move);
            if !conductor.is_king_in_check(chess_board, is_white) {
                result.push(chess_move);
            }
            chess_board.undo_move();
        }
    }
}

#[inline]
pub fn get_pseudo_legal_move_list_from_square(
    square: u16,
    chess_board: &ChessBoard,
    conductor: &PieceConductor,
    is_white: bool,
    move_list: &mut Vec<ChessMove>,
) {
    let friendly_pieces_bitboard = if is_white { chess_board.get_white() } else { chess_board.get_black() };
    let relevant_blockers = friendly_pieces_bitboard;
    if chess_board.get_rooks().contains_square(square as i32) {
        conductor.get_rook_moves(square, relevant_blockers, chess_board, move_list);
    } else if chess_board.get_bishops().contains_square(square as i32) {
        conductor.get_bishop_moves(square, relevant_blockers, chess_board, move_list);
    } else if chess_board.get_queens().contains_square(square as i32) {
        conductor.get_rook_moves(square, relevant_blockers, chess_board, move_list);
        conductor.get_bishop_moves(square, relevant_blockers, chess_board, move_list);
    } else if chess_board.get_kings().contains_square(square as i32) {
        conductor.get_king_moves(square, relevant_blockers, chess_board, is_white, move_list);
    } else if chess_board.get_knights().contains_square(square as i32) {
        conductor.get_knight_moves(square, relevant_blockers, move_list);
    } else if chess_board.get_pawns().contains_square(square as i32) {
        conductor.get_pawn_moves(square, is_white, chess_board, move_list);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::piece_conductor::PieceConductor;
    use chess_board::ChessBoard;

    #[test]
    fn test_legal_move_count_startpos() {
        let conductor = PieceConductor::new();
        let mut board = ChessBoard::new();
        let mut result = Vec::new();
        let mut pseudo_buf = Vec::new();
        get_all_legal_moves_for_color(&mut board, &conductor, true, &mut result, &mut pseudo_buf);
        assert_eq!(result.len(), 20, "Starting position should have 20 legal moves for white");
    }

    #[test]
    fn test_move_gen_buffer_reuse_consistency() {
        let conductor = PieceConductor::new();
        let mut board = ChessBoard::new();
        let mut result1 = Vec::new();
        let mut result2 = Vec::new();
        let mut pseudo_buf = Vec::new();

        get_all_legal_moves_for_color(&mut board, &conductor, true, &mut result1, &mut pseudo_buf);
        get_all_legal_moves_for_color(&mut board, &conductor, true, &mut result2, &mut pseudo_buf);

        let mut s1: Vec<_> = result1.iter().map(|m| (m.start_square(), m.target_square())).collect();
        let mut s2: Vec<_> = result2.iter().map(|m| (m.start_square(), m.target_square())).collect();
        s1.sort();
        s2.sort();
        assert_eq!(s1, s2, "Buffer reuse must not affect results");
    }
}

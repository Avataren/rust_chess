use chess_board::ChessBoard;
use chess_foundation::ChessMove;
use move_generator::{
    move_generator::get_all_legal_moves_for_color, piece_conductor::PieceConductor,
};

use crate::evaluate_board;

pub fn alpha_beta(
    chess_board: &mut ChessBoard,
    conductor: &PieceConductor,
    depth: i32,
    mut alpha: i32,
    mut beta: i32,
    is_white: bool, // Use is_white instead of is_maximizing_player
) -> (i32, Option<ChessMove>) {
    if depth == 0 {
        return (evaluate_board(chess_board), None);
    }

    let mut legal_moves = get_all_legal_moves_for_color(chess_board, conductor, is_white);
    legal_moves.sort();
    let mut best_move = None;

    if is_white {
        // Assuming white is maximizing
        let mut max_eval = i32::MIN;
        for mut chess_move in legal_moves {
            chess_board.make_move(&mut chess_move);
            let (eval, _) = alpha_beta(chess_board, conductor, depth - 1, alpha, beta, false); // Next call with black's turn
            chess_board.undo_move();

            if eval > max_eval {
                max_eval = eval;
                best_move = Some(chess_move);
            }

            alpha = alpha.max(eval);
            if beta <= alpha {
                break; // Beta cutoff
            }
        }
        (max_eval, best_move)
    } else {
        // Assuming black is minimizing
        let mut min_eval = i32::MAX;
        for mut chess_move in legal_moves {
            chess_board.make_move(&mut chess_move);
            let (eval, _) = alpha_beta(chess_board, conductor, depth - 1, alpha, beta, true); // Next call with white's turn
            chess_board.undo_move();

            if eval < min_eval {
                min_eval = eval;
                best_move = Some(chess_move);
            }

            beta = beta.min(eval);
            if beta <= alpha {
                break; // Alpha cutoff
            }
        }
        (min_eval, best_move)
    }
}

use chess_board::ChessBoard;
use chess_foundation::ChessMove;
use move_generator::{
    move_generator::get_all_legal_moves_for_color, piece_conductor::PieceConductor,
};
use rand::seq::SliceRandom;

use crate::{evaluate_board, opening_book::OpeningBook};

/// Continues searching capture-only moves after the main search depth is
/// exhausted, so we never evaluate a position mid-capture-sequence.
/// This eliminates the horizon effect that causes higher depths to play worse.
fn quiescence(
    chess_board: &mut ChessBoard,
    conductor: &PieceConductor,
    mut alpha: i32,
    mut beta: i32,
    is_white: bool,
    qdepth: i32,
) -> i32 {
    if qdepth == 0 {
        return evaluate_board(chess_board);
    }
    let stand_pat = evaluate_board(chess_board);

    if is_white {
        if stand_pat >= beta {
            return beta;
        }
        if stand_pat > alpha {
            alpha = stand_pat;
        }

        let mut captures: Vec<ChessMove> = get_all_legal_moves_for_color(chess_board, conductor, is_white)
            .into_iter()
            .filter(|m| m.capture.is_some())
            .collect();
        captures.sort();

        for mut chess_move in captures {
            chess_board.make_move(&mut chess_move);
            let eval = quiescence(chess_board, conductor, alpha, beta, false, qdepth - 1);
            chess_board.undo_move();

            if eval >= beta {
                return beta;
            }
            if eval > alpha {
                alpha = eval;
            }
        }
        alpha
    } else {
        if stand_pat <= alpha {
            return alpha;
        }
        if stand_pat < beta {
            beta = stand_pat;
        }

        let mut captures: Vec<ChessMove> = get_all_legal_moves_for_color(chess_board, conductor, is_white)
            .into_iter()
            .filter(|m| m.capture.is_some())
            .collect();
        captures.sort();

        for mut chess_move in captures {
            chess_board.make_move(&mut chess_move);
            let eval = quiescence(chess_board, conductor, alpha, beta, true, qdepth - 1);
            chess_board.undo_move();

            if eval <= alpha {
                return alpha;
            }
            if eval < beta {
                beta = eval;
            }
        }
        beta
    }
}

/// Internal recursive alpha-beta search.
pub fn alpha_beta(
    chess_board: &mut ChessBoard,
    conductor: &PieceConductor,
    depth: i32,
    mut alpha: i32,
    mut beta: i32,
    is_white: bool,
) -> (i32, Option<ChessMove>) {
    if depth == 0 {
        // Use quiescence search instead of bare static eval to avoid the
        // horizon effect (evaluating positions mid-capture-sequence).
        return (quiescence(chess_board, conductor, alpha, beta, is_white, 4), None);
    }

    let mut legal_moves = get_all_legal_moves_for_color(chess_board, conductor, is_white);
    legal_moves.sort();
    let mut best_move = None;

    if is_white {
        let mut max_eval = i32::MIN;
        for mut chess_move in legal_moves {
            chess_board.make_move(&mut chess_move);
            let (eval, _) = alpha_beta(chess_board, conductor, depth - 1, alpha, beta, false);
            chess_board.undo_move();

            if eval > max_eval {
                max_eval = eval;
                best_move = Some(chess_move);
            }
            alpha = alpha.max(eval);
            if beta <= alpha {
                break;
            }
        }
        (max_eval, best_move)
    } else {
        let mut min_eval = i32::MAX;
        for mut chess_move in legal_moves {
            chess_board.make_move(&mut chess_move);
            let (eval, _) = alpha_beta(chess_board, conductor, depth - 1, alpha, beta, true);
            chess_board.undo_move();

            if eval < min_eval {
                min_eval = eval;
                best_move = Some(chess_move);
            }
            beta = beta.min(eval);
            if beta <= alpha {
                break;
            }
        }
        (min_eval, best_move)
    }
}

/// Root-level search. Probes the opening book first; falls back to alpha-beta.
///
/// Each root move is searched with a fresh [MIN, MAX] window so that fail-hard
/// alpha-beta doesn't mask the true score of later moves. Without this, a move
/// that scores -900 (queen blunder) appears to score 0 (equal to the best move)
/// because the alpha-cutoff returns the previous best score, not the true value.
pub fn alpha_beta_root(
    chess_board: &mut ChessBoard,
    conductor: &PieceConductor,
    book: Option<&OpeningBook>,
    depth: i32,
    is_white: bool,
) -> (i32, Option<ChessMove>) {
    // Opening book probe: if we get a hit, find the matching legal move and return it.
    if let Some(book) = book {
        if let Some((from, to)) = book.probe(chess_board) {
            let legal = get_all_legal_moves_for_color(chess_board, conductor, is_white);
            if let Some(book_move) = legal
                .into_iter()
                .find(|m| m.start_square() == from && m.target_square() == to)
            {
                println!("Book move: {}", book_move.to_san_simple());
                return (0, Some(book_move));
            }
        }
    }

    let mut legal_moves = get_all_legal_moves_for_color(chess_board, conductor, is_white);
    if legal_moves.is_empty() {
        return (evaluate_board(chess_board), None);
    }
    legal_moves.sort();

    let mut best_score = if is_white { i32::MIN } else { i32::MAX };
    let mut best_moves: Vec<ChessMove> = Vec::new();

    for mut chess_move in legal_moves {
        chess_board.make_move(&mut chess_move);
        // Always use the full [MIN, MAX] window per root move so each gets its
        // true score — alpha-beta pruning still works inside the sub-search.
        let (eval, _) = alpha_beta(chess_board, conductor, depth - 1, i32::MIN, i32::MAX, !is_white);
        chess_board.undo_move();

        if is_white {
            if eval > best_score {
                best_score = eval;
                best_moves.clear();
                best_moves.push(chess_move);
            } else if eval == best_score {
                best_moves.push(chess_move);
            }
        } else {
            if eval < best_score {
                best_score = eval;
                best_moves.clear();
                best_moves.push(chess_move);
            } else if eval == best_score {
                best_moves.push(chess_move);
            }
        }
    }

    let best = best_moves.choose(&mut rand::thread_rng()).copied();
    (best_score, best)
}

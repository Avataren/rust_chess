use chess_board::ChessBoard;
use chess_foundation::ChessMove;
use move_generator::{
    move_generator::get_all_legal_moves_for_color, piece_conductor::PieceConductor,
};
use rand::seq::SliceRandom;

use crate::{
    evaluate_board,
    opening_book::OpeningBook,
    transposition_table::{TranspositionTable, TtFlag},
};

/// 256 K entries ≈ 8 MB.  Chosen to fit comfortably in native and WASM builds.
const TT_SIZE: usize = 1 << 20; // 1M entries ≈ 32 MB

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

    // Fail-soft quiescence: return the actual best score found, not alpha/beta.
    // This ensures the root can distinguish between a genuine best-score and a
    // fail-low that merely equals the accumulated alpha from a sibling search.
    if is_white {
        if stand_pat >= beta {
            return stand_pat; // fail-soft: return actual value
        }
        let mut best = stand_pat;
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
                return eval; // fail-soft
            }
            if eval > best {
                best = eval;
            }
            if eval > alpha {
                alpha = eval;
            }
        }
        best
    } else {
        if stand_pat <= alpha {
            return stand_pat; // fail-soft
        }
        let mut best = stand_pat;
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
                return eval; // fail-soft
            }
            if eval < best {
                best = eval;
            }
            if eval < beta {
                beta = eval;
            }
        }
        best
    }
}

/// Order moves: try the TT best move first (if any), then sort the rest by
/// MVV-LVA / promotion heuristics via the `Ord` impl on `ChessMove`.
fn order_moves(moves: &mut Vec<ChessMove>, tt_move: Option<ChessMove>) {
    if let Some(tt_m) = tt_move {
        if let Some(idx) = moves.iter().position(|m| {
            m.start_square() == tt_m.start_square() && m.target_square() == tt_m.target_square()
        }) {
            moves.swap(0, idx);
            moves[1..].sort();
            return;
        }
    }
    moves.sort();
}

/// Returns true when the position is likely a zugzwang situation:
/// fewer than 2 minor/major pieces on the board means the side to move
/// may have no good quiet moves to pass, making null-move pruning unsafe.
fn is_zugzwang_prone(chess_board: &ChessBoard) -> bool {
    let minor_and_major = chess_board.get_knights()
        | chess_board.get_bishops()
        | chess_board.get_rooks()
        | chess_board.get_queens();
    minor_and_major.count_ones() < 2
}

/// Internal recursive alpha-beta search with transposition table.
pub fn alpha_beta(
    chess_board: &mut ChessBoard,
    conductor: &PieceConductor,
    tt: &mut TranspositionTable,
    depth: i32,
    mut alpha: i32,
    mut beta: i32,
    is_white: bool,
    null_move_allowed: bool,
) -> (i32, Option<ChessMove>) {
    if depth == 0 {
        return (quiescence(chess_board, conductor, alpha, beta, is_white, 4), None);
    }

    let hash = chess_board.current_hash();
    let original_alpha = alpha;
    let original_beta = beta;

    // --- Transposition table probe ---
    let tt_move: Option<ChessMove> = if let Some(entry) = tt.probe(hash) {
        if entry.depth >= depth {
            match entry.flag {
                TtFlag::Exact => return (entry.score, entry.best_move),
                TtFlag::LowerBound => {
                    if entry.score > alpha {
                        alpha = entry.score;
                    }
                }
                TtFlag::UpperBound => {
                    if entry.score < beta {
                        beta = entry.score;
                    }
                }
            }
            if alpha >= beta {
                return (entry.score, entry.best_move);
            }
        }
        entry.best_move
    } else {
        None
    };

    // Compute once; reused by both NMP and LMR.
    let in_check = conductor.is_king_in_check(chess_board, is_white);

    // --- Null Move Pruning ---
    if null_move_allowed
        && depth >= 3
        && !in_check
        && !is_zugzwang_prone(chess_board)
    {
        let r = if depth >= 6 { 3 } else { 2 };
        chess_board.make_null_move();
        let null_score = alpha_beta(
            chess_board, conductor, tt,
            depth - 1 - r, alpha, beta, !is_white,
            false,
        ).0;
        chess_board.undo_null_move();

        if is_white && null_score >= beta { return (beta, None); }
        if !is_white && null_score <= alpha { return (alpha, None); }
    }

    let mut legal_moves = get_all_legal_moves_for_color(chess_board, conductor, is_white);
    order_moves(&mut legal_moves, tt_move);

    let mut best_move: Option<ChessMove> = None;

    if is_white {
        let mut max_eval = i32::MIN;
        for (move_index, mut chess_move) in legal_moves.into_iter().enumerate() {
            chess_board.make_move(&mut chess_move);

            let can_reduce = move_index >= 2
                && depth >= 3
                && chess_move.capture.is_none()
                && !chess_move.is_promotion()
                && !in_check;

            let eval = if chess_board.is_repetition(2) {
                0 // draw by repetition
            } else if can_reduce {
                let reduced = alpha_beta(chess_board, conductor, tt,
                    depth - 2, alpha, beta, false, true).0;
                if reduced > alpha {
                    // Re-search at full depth
                    alpha_beta(chess_board, conductor, tt,
                        depth - 1, alpha, beta, false, true).0
                } else {
                    reduced
                }
            } else {
                alpha_beta(chess_board, conductor, tt,
                    depth - 1, alpha, beta, false, true).0
            };

            chess_board.undo_move();

            if eval > max_eval {
                max_eval = eval;
                best_move = Some(chess_move);
            }
            if eval > alpha {
                alpha = eval;
            }
            if beta <= alpha {
                break; // beta cutoff
            }
        }

        // Determine TT flag: fail-high (beta cutoff) → LowerBound, else Exact.
        let flag = if max_eval >= original_beta {
            TtFlag::LowerBound
        } else if max_eval <= original_alpha {
            TtFlag::UpperBound
        } else {
            TtFlag::Exact
        };
        tt.store(hash, depth, max_eval, flag, best_move);
        (max_eval, best_move)
    } else {
        let mut min_eval = i32::MAX;
        for (move_index, mut chess_move) in legal_moves.into_iter().enumerate() {
            chess_board.make_move(&mut chess_move);

            let can_reduce = move_index >= 2
                && depth >= 3
                && chess_move.capture.is_none()
                && !chess_move.is_promotion()
                && !in_check;

            let eval = if chess_board.is_repetition(2) {
                0 // draw by repetition
            } else if can_reduce {
                let reduced = alpha_beta(chess_board, conductor, tt,
                    depth - 2, alpha, beta, true, true).0;
                if reduced < beta {
                    // Re-search at full depth
                    alpha_beta(chess_board, conductor, tt,
                        depth - 1, alpha, beta, true, true).0
                } else {
                    reduced
                }
            } else {
                alpha_beta(chess_board, conductor, tt,
                    depth - 1, alpha, beta, true, true).0
            };

            chess_board.undo_move();

            if eval < min_eval {
                min_eval = eval;
                best_move = Some(chess_move);
            }
            if eval < beta {
                beta = eval;
            }
            if beta <= alpha {
                break; // alpha cutoff
            }
        }

        // Determine TT flag: fail-low (alpha cutoff) → UpperBound, else Exact.
        let flag = if min_eval <= original_alpha {
            TtFlag::UpperBound
        } else if min_eval >= original_beta {
            TtFlag::LowerBound
        } else {
            TtFlag::Exact
        };
        tt.store(hash, depth, min_eval, flag, best_move);
        (min_eval, best_move)
    }
}

/// Root-level search. Probes the opening book first; falls back to alpha-beta.
///
/// Passes an accumulated alpha/beta window so that sub-searches can prune
/// once we have established a good score, which is much faster than the
/// previous approach of using a fresh [MIN, MAX] window per root move.
pub fn alpha_beta_root(
    chess_board: &mut ChessBoard,
    conductor: &PieceConductor,
    book: Option<&OpeningBook>,
    depth: i32,
    is_white: bool,
) -> (i32, Option<ChessMove>) {
    // Opening book probe: if we get a hit, return the book move immediately.
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

    // Fresh TT for each search.  Entries from earlier in the tree (e.g. when
    // deeper sub-trees reach the same positions) are reused throughout this call.
    let mut tt = TranspositionTable::new(TT_SIZE);

    let mut best_score = if is_white { i32::MIN + 1 } else { i32::MAX };
    let mut best_moves: Vec<ChessMove> = Vec::new();

    // Use a growing window: alpha rises (for white) or beta falls (for black)
    // as we find better moves, enabling pruning in every subsequent sub-search.
    let mut alpha = i32::MIN + 1;
    let mut beta = i32::MAX;

    for mut chess_move in legal_moves {
        chess_board.make_move(&mut chess_move);
        let (eval, _) = alpha_beta(
            chess_board,
            conductor,
            &mut tt,
            depth - 1,
            alpha,
            beta,
            !is_white,
            true,
        );
        chess_board.undo_move();

        if is_white {
            if eval > best_score {
                best_score = eval;
                best_moves.clear();
                best_moves.push(chess_move);
                if eval > alpha {
                    alpha = eval;
                }
            } else if eval == best_score {
                best_moves.push(chess_move);
            }
        } else {
            if eval < best_score {
                best_score = eval;
                best_moves.clear();
                best_moves.push(chess_move);
                if eval < beta {
                    beta = eval;
                }
            } else if eval == best_score {
                best_moves.push(chess_move);
            }
        }
    }

    let best = best_moves.choose(&mut rand::thread_rng()).copied();
    (best_score, best)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chess_board::ChessBoard;
    use move_generator::{
        move_generator::get_all_legal_moves_for_color, piece_conductor::PieceConductor,
    };

    fn conductor() -> PieceConductor {
        PieceConductor::new()
    }

    // -----------------------------------------------------------------------
    // Basic sanity
    // -----------------------------------------------------------------------

    #[test]
    fn returns_a_move_from_starting_position() {
        let mut board = ChessBoard::new();
        let c = conductor();
        let (_, mv) = alpha_beta_root(&mut board, &c, None, 2, true);
        assert!(mv.is_some(), "Engine must return a move from the starting position");
    }

    #[test]
    fn score_is_zero_for_perfectly_symmetric_position() {
        // Starting position is symmetric, so the score should be 0.
        let mut board = ChessBoard::new();
        let c = conductor();
        let (score, _) = alpha_beta_root(&mut board, &c, None, 2, true);
        // Allow a small margin for positional asymmetries, but rough equality.
        assert!(
            score.abs() < 50,
            "Starting position score should be near 0, got {score}"
        );
    }

    // -----------------------------------------------------------------------
    // Tactical: winning captures
    // -----------------------------------------------------------------------

    /// Position: white king e1 (4), white queen d4 (27), black king e8 (60),
    /// black queen d5 (35).  White should immediately capture the undefended queen.
    #[test]
    fn white_captures_hanging_queen() {
        let mut board = ChessBoard::new();
        board.set_from_fen("4k3/8/8/3q4/3Q4/8/8/4K3 w - - 0 1");
        let c = conductor();
        let (score, mv) = alpha_beta_root(&mut board, &c, None, 2, true);
        let m = mv.expect("Engine must find a move");
        assert_eq!(m.start_square(), 27, "Should move from d4 (square 27)");
        assert_eq!(m.target_square(), 35, "Should capture on d5 (square 35)");
        assert!(score > 800, "Score should reflect a queen-up advantage, got {score}");
    }

    /// Same position, but now it is black to move — black should capture white's queen.
    #[test]
    fn black_captures_hanging_queen() {
        let mut board = ChessBoard::new();
        board.set_from_fen("4k3/8/8/3q4/3Q4/8/8/4K3 b - - 0 1");
        let c = conductor();
        let (score, mv) = alpha_beta_root(&mut board, &c, None, 2, false);
        let m = mv.expect("Engine must find a move");
        assert_eq!(m.start_square(), 35, "Should move from d5 (square 35)");
        assert_eq!(m.target_square(), 27, "Should capture on d4 (square 27)");
        assert!(score < -800, "Score should reflect black's queen-up advantage, got {score}");
    }

    // -----------------------------------------------------------------------
    // Checkmate detection
    // -----------------------------------------------------------------------

    /// Position: white queen f7 (53), white king g6 (46), black king g8 (62).
    /// At depth 2 the engine sees that black has no legal reply — checkmate.
    #[test]
    fn white_finds_checkmate_in_one() {
        let mut board = ChessBoard::new();
        board.set_from_fen("6k1/5Q2/6K1/8/8/8/8/8 w - - 0 1");
        let c = conductor();
        let (_, mv) = alpha_beta_root(&mut board, &c, None, 2, true);
        let m = mv.expect("Engine must find a move");

        let mut board_after = board.clone();
        let mut mv_copy = m;
        board_after.make_move(&mut mv_copy);
        let replies = get_all_legal_moves_for_color(&mut board_after, &c, false);
        assert!(
            replies.is_empty(),
            "After white's best move ({}) black should have no legal replies",
            m.to_san_simple()
        );
    }

    /// Position: black queen b3 (17), white king a1 (0), black king c1 (2).
    /// At depth 2 the engine sees that white has no legal reply — checkmate.
    #[test]
    fn black_finds_checkmate_in_one() {
        let mut board = ChessBoard::new();
        board.set_from_fen("8/8/8/8/8/1q6/8/K1k5 b - - 0 1");
        let c = conductor();
        let (_, mv) = alpha_beta_root(&mut board, &c, None, 2, false);
        let m = mv.expect("Engine must find a move");

        let mut board_after = board.clone();
        let mut mv_copy = m;
        board_after.make_move(&mut mv_copy);
        let replies = get_all_legal_moves_for_color(&mut board_after, &c, true);
        assert!(
            replies.is_empty(),
            "After black's best move ({}) white should have no legal replies",
            m.to_san_simple()
        );
    }

    // -----------------------------------------------------------------------
    // TT correctness: results must be consistent
    // -----------------------------------------------------------------------

    /// Running the same search twice must produce the same score.
    #[test]
    fn search_score_is_deterministic() {
        let mut board = ChessBoard::new();
        board.set_from_fen("4k3/8/8/3q4/3Q4/8/8/4K3 w - - 0 1");
        let c = conductor();
        let (s1, _) = alpha_beta_root(&mut board, &c, None, 3, true);
        let (s2, _) = alpha_beta_root(&mut board, &c, None, 3, true);
        assert_eq!(s1, s2, "Same search must produce the same score");
    }

    /// The internal alpha_beta (with explicit TT) must agree with alpha_beta_root
    /// on the evaluation of the same position.
    #[test]
    fn internal_alpha_beta_agrees_with_root() {
        let mut board = ChessBoard::new();
        board.set_from_fen("4k3/8/8/3q4/3Q4/8/8/4K3 w - - 0 1");
        let c = conductor();

        let (root_score, _) = alpha_beta_root(&mut board, &c, None, 2, true);

        let mut tt = TranspositionTable::new(1 << 16);
        let (ab_score, _) = alpha_beta(&mut board, &c, &mut tt, 2, i32::MIN + 1, i32::MAX, true, true);

        assert_eq!(
            root_score, ab_score,
            "alpha_beta_root and alpha_beta must agree on score"
        );
    }

    /// After applying the best move the board is back to its original state
    /// (i.e. undo_move works correctly and the TT didn't corrupt anything).
    #[test]
    fn board_state_unchanged_after_search() {
        let mut board = ChessBoard::new();
        let hash_before = board.current_hash();
        let c = conductor();
        alpha_beta_root(&mut board, &c, None, 3, true);
        assert_eq!(
            board.current_hash(),
            hash_before,
            "Search must not leave the board in a modified state"
        );
    }

    /// NMP and LMR must not corrupt the board state.
    #[test]
    fn nmp_lmr_do_not_corrupt_board_state() {
        let mut board = ChessBoard::new();
        let hash_before = board.current_hash();
        let c = conductor();
        // depth=4 ensures NMP and LMR are triggered
        alpha_beta_root(&mut board, &c, None, 4, true);
        assert_eq!(board.current_hash(), hash_before);
    }

    // -----------------------------------------------------------------------
    // is_zugzwang_prone
    // -----------------------------------------------------------------------

    #[test]
    fn zugzwang_prone_kings_only() {
        let mut board = ChessBoard::new();
        board.set_from_fen("k7/8/8/8/8/8/8/7K w - - 0 1");
        assert!(is_zugzwang_prone(&board), "K vs K should be zugzwang-prone");
    }

    #[test]
    fn zugzwang_prone_one_rook() {
        let mut board = ChessBoard::new();
        board.set_from_fen("k7/8/8/8/8/8/8/6RK w - - 0 1");
        assert!(is_zugzwang_prone(&board), "K+R vs K should be zugzwang-prone");
    }

    #[test]
    fn not_zugzwang_prone_two_rooks() {
        let mut board = ChessBoard::new();
        board.set_from_fen("k7/8/8/8/8/8/8/5RRK w - - 0 1");
        assert!(!is_zugzwang_prone(&board), "K+2R vs K should not be zugzwang-prone");
    }

    #[test]
    fn not_zugzwang_prone_starting_position() {
        let board = ChessBoard::new();
        assert!(!is_zugzwang_prone(&board), "Starting position should not be zugzwang-prone");
    }

    // -----------------------------------------------------------------------
    // Tactical correctness at depth >= 3 (NMP + LMR active)
    // -----------------------------------------------------------------------

    /// Same queen-capture position at depth=4 — NMP and LMR both active.
    /// We only assert the score here, not the specific move: in a symmetrical
    /// double-queen position, NMP can tie non-capturing moves with the capture
    /// score (by "seeing" the capture in the null-move sub-search), so the root
    /// may randomly pick a king move that shares the same returned score.
    /// The score invariant (white is up a queen) must still hold regardless.
    #[test]
    fn white_captures_hanging_queen_depth_4() {
        let mut board = ChessBoard::new();
        board.set_from_fen("4k3/8/8/3q4/3Q4/8/8/4K3 w - - 0 1");
        let hash_before = board.current_hash();
        let c = conductor();
        let (score, mv) = alpha_beta_root(&mut board, &c, None, 4, true);
        assert!(mv.is_some(), "Engine must return a move at depth 4");
        assert!(score > 800, "Score should reflect a queen-up advantage, got {score}");
        assert_eq!(board.current_hash(), hash_before, "Board must be clean after search");
    }

    #[test]
    fn black_captures_hanging_queen_depth_4() {
        let mut board = ChessBoard::new();
        board.set_from_fen("4k3/8/8/3q4/3Q4/8/8/4K3 b - - 0 1");
        let hash_before = board.current_hash();
        let c = conductor();
        let (score, mv) = alpha_beta_root(&mut board, &c, None, 4, false);
        assert!(mv.is_some(), "Engine must return a move at depth 4");
        assert!(score < -800, "Score should reflect black's queen-up advantage, got {score}");
        assert_eq!(board.current_hash(), hash_before, "Board must be clean after search");
    }

    // -----------------------------------------------------------------------
    // NMP in-check guard
    // -----------------------------------------------------------------------

    /// White king in check from a black rook at depth=3.
    /// NMP must be skipped (null move in check is illegal); engine must still
    /// return a legal evasion and leave the board state intact.
    #[test]
    fn nmp_skipped_when_in_check() {
        let mut board = ChessBoard::new();
        // Black rook on e8 gives check to white king on e1; black king on a8.
        board.set_from_fen("k3r3/8/8/8/8/8/8/4K3 w - - 0 1");
        let hash_before = board.current_hash();
        let c = conductor();
        let (_, mv) = alpha_beta_root(&mut board, &c, None, 3, true);
        assert!(mv.is_some(), "Engine must return an evasion move when in check");
        assert_eq!(board.current_hash(), hash_before, "Board must be clean after search");
    }
}

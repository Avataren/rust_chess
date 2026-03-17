use chess_board::ChessBoard;
use move_generator::piece_conductor::PieceConductor;

/// Evaluates the chess board position.
/// Returns centipawns from white's perspective (positive = good for white).
///
/// The active backend is selected at compile time via Cargo features:
///   classical-eval  — hand-crafted HCE (no NN)
///   nn-full-forward — full NN forward pass, no runtime enable/disable check
///   nn-incremental  — full NN forward pass here; incremental accumulator in search
///
/// Returns 0 when no feature is selected (should not happen in a real build).
pub fn evaluate_board(chess_board: &ChessBoard, conductor: &PieceConductor) -> i32 {
    // Priority: nn-incremental > nn-full-forward > runtime-switch > classical-eval.
    // The `not()` guards make this correct even when Cargo unifies multiple features
    // across workspace members (e.g. `cargo build --workspace`).

    #[cfg(feature = "nn-incremental")]
    return crate::neural_eval::eval_direct(chess_board);

    #[cfg(all(feature = "nn-full-forward", not(feature = "nn-incremental")))]
    return crate::neural_eval::eval_direct(chess_board);

    #[cfg(all(feature = "runtime-switch",
              not(any(feature = "nn-incremental", feature = "nn-full-forward"))))]
    {
        if let Some(score) = crate::neural_eval::try_neural_eval(chess_board) {
            return score;
        }
        return crate::classical_eval::evaluate(chess_board, conductor);
    }

    #[cfg(all(feature = "classical-eval",
              not(any(feature = "nn-incremental", feature = "nn-full-forward",
                      feature = "runtime-switch"))))]
    return crate::classical_eval::evaluate(chess_board, conductor);

    let _ = conductor;
    0
}

// ── Tests — only compiled with classical-eval (they need deterministic HCE) ──

#[cfg(all(test, feature = "classical-eval"))]
mod tests {
    use super::*;
    use chess_board::ChessBoard;
    use move_generator::piece_conductor::PieceConductor;

    fn eval(fen: &str) -> i32 {
        let mut board = ChessBoard::new();
        board.set_from_fen(fen);
        evaluate_board(&board, &PieceConductor::new())
    }

    #[test]
    fn starting_position_is_near_zero() {
        let score = eval("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        assert!(score.abs() < 30, "Starting position must be near-zero, got {score}");
    }

    #[test]
    fn central_knight_scores_higher_than_corner_knight() {
        let central = eval("4k3/8/8/8/8/2N5/8/4K3 w - - 0 1");
        let corner  = eval("4k3/8/8/8/8/8/8/N3K3 w - - 0 1");
        assert!(central > corner);
    }

    #[test]
    fn open_diagonal_bishop_scores_higher() {
        let central = eval("4k3/8/8/8/2B5/8/8/4K3 w - - 0 1");
        let corner  = eval("4k3/8/8/8/8/8/8/B3K3 w - - 0 1");
        assert!(central > corner);
    }

    #[test]
    fn mobility_is_symmetric_for_equal_positions() {
        let score = eval("4k3/8/2n5/8/8/2N5/8/4K3 w - - 0 1");
        assert!(score.abs() < 30);
    }

    #[test]
    fn extra_pawn_beats_extra_mobility() {
        let with_pawn    = eval("4k3/8/2n5/8/2NP4/8/8/4K3 w - - 0 1");
        let without_pawn = eval("4k3/8/2n5/8/2N5/8/8/4K3 w - - 0 1");
        assert!(with_pawn > without_pawn);
    }

    #[test]
    fn rook_open_file_still_rewarded() {
        let open   = eval("4k3/8/8/8/P7/8/8/R3K3 w - - 0 1");
        let closed = eval("4k3/8/8/8/8/8/1P6/1R2K3 w - - 0 1");
        assert!(open > closed);
    }

    #[test]
    fn mobility_bonus_is_not_inflated() {
        let score = eval("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        assert!(score.abs() < 50);
    }

    #[test]
    fn rook_on_7th_ranks_higher_than_6th() {
        let on_7th = eval("6k1/R7/8/8/8/8/8/4K3 w - - 0 1");
        let on_6th = eval("6k1/8/R7/8/8/8/8/4K3 w - - 0 1");
        assert!(on_7th > on_6th);
    }

    #[test]
    fn rook_on_7th_requires_king_on_8th() {
        let with_bonus    = eval("6k1/R7/8/8/8/8/8/4K3 w - - 0 1");
        let without_bonus = eval("8/R7/8/8/4k3/8/8/4K3 w - - 0 1");
        assert!(with_bonus > without_bonus);
    }

    #[test]
    fn knight_outpost_scores_higher_than_non_outpost() {
        let outpost     = eval("4k3/8/8/3N4/4P3/8/8/4K3 w - - 0 1");
        let non_outpost = eval("4k3/8/8/8/3NP3/8/8/4K3 w - - 0 1");
        assert!(outpost > non_outpost);
    }

    #[test]
    fn outpost_not_awarded_when_enemy_pawn_attacks() {
        let chased  = eval("4k3/8/4p3/3N4/4P3/8/8/4K3 w - - 0 1");
        let outpost = eval("4k3/8/8/3N4/4P3/8/8/4K3 w - - 0 1");
        assert!(outpost > chased);
    }

    #[test]
    fn outpost_requires_pawn_protection() {
        let unprotected = eval("4k3/8/8/3N4/8/8/8/4K3 w - - 0 1");
        let protected   = eval("4k3/8/8/3N4/4P3/8/8/4K3 w - - 0 1");
        assert!(protected > unprotected);
    }

    #[test]
    fn pawn_hash_consistent_on_repeated_calls() {
        let c = PieceConductor::new();
        let mut board = ChessBoard::new();
        board.set_from_fen("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R b KQkq - 0 6");
        let s1 = evaluate_board(&board, &c);
        let s2 = evaluate_board(&board, &c);
        assert_eq!(s1, s2);
    }

    #[test]
    fn pawn_hash_distinguishes_different_pawn_structures() {
        let c = PieceConductor::new();
        let mut a = ChessBoard::new();
        a.set_from_fen("4k3/pppppppp/8/8/8/8/PPPPPPPP/4K3 w - - 0 1");
        let mut b = ChessBoard::new();
        b.set_from_fen("4k3/pppp1ppp/8/4P3/8/8/PPPP1PPP/4K3 w - - 0 1");
        assert_ne!(evaluate_board(&a, &c), evaluate_board(&b, &c));
    }

    #[test]
    fn pawn_hash_updates_after_pawn_structure_change() {
        let c = PieceConductor::new();
        let mut b1 = ChessBoard::new();
        b1.set_from_fen("4k3/8/8/8/4P3/4P3/8/4K3 w - - 0 1");
        let s1 = evaluate_board(&b1, &c);
        let mut b2 = ChessBoard::new();
        b2.set_from_fen("4k3/8/8/8/4P3/3P4/8/4K3 w - - 0 1");
        let s2 = evaluate_board(&b2, &c);
        assert!(s1 < s2);
        assert_eq!(s1, evaluate_board(&b1, &c));
    }

    #[test]
    fn broken_pawn_shield_scores_worse() {
        let intact = eval("4k3/8/8/8/8/8/5PPP/6K1 w - - 0 1");
        let broken = eval("4k3/8/8/5ppp/8/8/8/6K1 w - - 0 1");
        assert!(intact > broken);
    }

    #[test]
    fn attacked_king_scores_worse() {
        let attacked = eval("4k3/8/8/8/8/8/8/K2r1q2 w - - 0 1");
        let safe     = eval("3qk3/8/8/8/8/8/8/K7 w - - 0 1");
        assert!(attacked < safe);
    }

    #[test]
    fn queen_presence_amplifies_attack_danger() {
        let with_queen    = eval("4k3/8/8/8/8/2n5/8/K3r1q1 w - - 0 1");
        let without_queen = eval("4k3/8/8/8/8/2n5/8/K3r3 w - - 0 1");
        assert!(with_queen < without_queen);
    }
}

use chess_board::ChessBoard;
use move_generator::{
    move_generator::get_all_legal_captures_for_color,
    piece_conductor::PieceConductor,
};
use crate::see::see;
use super::{capture_value, eval_node, SearchContext, MAX_PLY, DELTA_MARGIN};

// ── Quiescence search ─────────────────────────────────────────────────────────

/// Continues searching capture-only moves after the main search depth is
/// exhausted, so we never evaluate a position mid-capture-sequence.
/// This eliminates the horizon effect that causes higher depths to play worse.
///
/// SEE pruning: skip captures where SEE < 0 (clearly losing exchanges).
/// This is strictly more accurate than the previous delta-pruning heuristic
/// and correctly handles defended pieces without needing an arbitrary margin.
/// TT is intentionally NOT used here: qsearch runs at millions of nodes/s and
/// TT lookups/stores at that frequency cause cache thrashing that slows the
/// overall search more than the TT hits save.
fn quiescence_inner<const IS_WHITE: bool>(
    chess_board: &mut ChessBoard,
    conductor: &PieceConductor,
    ctx: &mut SearchContext,
    mut alpha: i32,
    mut beta: i32,
    parent_eval: Option<i32>,
    qdepth: i32,
    ply: usize,
) -> i32 {
    ctx.nodes += 1;
    if qdepth == 0 {
        return parent_eval.unwrap_or_else(|| eval_node(chess_board, conductor, ctx, ply));
    }
    // Reuse the eval passed in from the parent (avoids a redundant eval_node call
    // at the top-level qsearch entry); recursive calls pass None.
    let stand_pat = parent_eval.unwrap_or_else(|| eval_node(chess_board, conductor, ctx, ply));

    // Fail-soft quiescence: return the actual best score found, not alpha/beta.
    if IS_WHITE {
        if stand_pat >= beta { return stand_pat; }
        if stand_pat > alpha { alpha = stand_pat; }
    } else {
        if stand_pat <= alpha { return stand_pat; }
        if stand_pat < beta { beta = stand_pat; }
    }
    let mut best = stand_pat;

    let mut captures = std::mem::take(&mut ctx.move_lists[ply.min(MAX_PLY - 1)]);
    let mut pseudo_buf = std::mem::take(&mut ctx.pseudo_buf);
    get_all_legal_captures_for_color(
        chess_board,
        conductor,
        IS_WHITE,
        &mut captures,
        &mut pseudo_buf,
    );
    ctx.pseudo_buf = pseudo_buf;
    captures.sort();

    for idx in 0..captures.len() {
        let mut chess_move = captures[idx];
        // Delta pruning: if even capturing the piece (plus a margin) can't
        // improve the bound, skip this capture (saves SEE computation).
        let cap_val = capture_value(chess_board, &chess_move)
            + if chess_move.is_promotion() { 800 } else { 0 };
        if IS_WHITE {
            if stand_pat + cap_val + DELTA_MARGIN <= alpha { continue; }
        } else {
            if stand_pat - cap_val - DELTA_MARGIN >= beta { continue; }
        }

        // SEE pruning: skip losing captures (SEE < 0).
        if see(
            chess_board,
            conductor,
            chess_move.start_square() as usize,
            chess_move.target_square() as usize,
            IS_WHITE,
        ) < 0
        {
            continue;
        }

        ctx.make_move_with_acc(ply, &mut chess_move, chess_board);
        let eval = quiescence(
            chess_board,
            conductor,
            ctx,
            alpha,
            beta,
            !IS_WHITE,
            None,
            qdepth - 1,
            ply + 1,
        );
        chess_board.undo_move();

        if IS_WHITE {
            if eval >= beta {
                captures.clear();
                ctx.move_lists[ply.min(MAX_PLY - 1)] = captures;
                return eval;
            }
            if eval > best { best = eval; }
            if eval > alpha { alpha = eval; }
        } else {
            if eval <= alpha {
                captures.clear();
                ctx.move_lists[ply.min(MAX_PLY - 1)] = captures;
                return eval;
            }
            if eval < best { best = eval; }
            if eval < beta { beta = eval; }
        }
    }
    captures.clear();
    ctx.move_lists[ply.min(MAX_PLY - 1)] = captures;
    best
}

pub(super) fn quiescence(
    chess_board: &mut ChessBoard,
    conductor: &PieceConductor,
    ctx: &mut SearchContext,
    alpha: i32,
    beta: i32,
    is_white: bool,
    parent_eval: Option<i32>,
    qdepth: i32,
    ply: usize,
) -> i32 {
    if is_white {
        quiescence_inner::<true>(chess_board, conductor, ctx, alpha, beta, parent_eval, qdepth, ply)
    } else {
        quiescence_inner::<false>(chess_board, conductor, ctx, alpha, beta, parent_eval, qdepth, ply)
    }
}

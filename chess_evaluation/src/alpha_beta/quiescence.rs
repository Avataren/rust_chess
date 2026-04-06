use chess_board::ChessBoard;
use move_generator::{
    move_generator::{get_all_legal_captures_for_color, get_all_legal_moves_for_color},
    piece_conductor::PieceConductor,
};
use crate::see::see;
use super::{capture_value, eval_node, SearchContext, MAX_PLY, MATE_BASE};

// ── Quiescence search ─────────────────────────────────────────────────────────

/// Continues searching after the main search depth is exhausted.
///
/// **Normal (not in check):** searches capture-only moves to avoid the horizon
/// effect.  Delta pruning and SEE pruning skip unpromising captures.
///
/// **In check:** generates *all* legal moves (evasions) and searches every one.
/// Stand-pat is unsound when forced — we can't pass.  If there are no legal
/// moves the position is checkmate and we return the mated score.
///
/// TT is intentionally NOT used: qsearch runs at millions of nodes/s and
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

    let in_check = conductor.is_king_in_check(chess_board, IS_WHITE);

    if qdepth == 0 {
        // Depth exhausted — return static eval regardless of check status.
        // In practice qdepth reaches 0 only after 12 consecutive reductions,
        // so being in check here is extremely rare.
        return parent_eval.unwrap_or_else(|| eval_node(chess_board, conductor, ctx, ply));
    }

    // ── Check evasions ────────────────────────────────────────────────────────
    // When in check we must try every legal move.  Stand-pat is not available.
    if in_check {
        let mut evasions = std::mem::take(&mut ctx.move_lists[ply.min(MAX_PLY - 1)]);
        let mut pseudo_buf = std::mem::take(&mut ctx.pseudo_buf);
        get_all_legal_moves_for_color(chess_board, conductor, IS_WHITE, &mut evasions, &mut pseudo_buf);
        ctx.pseudo_buf = pseudo_buf;

        if evasions.is_empty() {
            // No legal moves while in check = checkmate.
            // Score is from white's perspective: white wins if black is mated,
            // white loses if white is mated.
            evasions.clear();
            ctx.move_lists[ply.min(MAX_PLY - 1)] = evasions;
            return if IS_WHITE {
                -(MATE_BASE - ply as i32) // white is mated — bad for white
            } else {
                MATE_BASE - ply as i32    // black is mated — good for white
            };
        }

        // Worst-case initialiser: mated at this ply (will be overwritten).
        let mut best = if IS_WHITE { -(MATE_BASE - ply as i32) } else { MATE_BASE - ply as i32 };
        for idx in 0..evasions.len() {
            let mut chess_move = evasions[idx];
            ctx.make_move_with_acc(ply, &mut chess_move, chess_board);
            let eval = quiescence(
                chess_board, conductor, ctx,
                alpha, beta, !IS_WHITE,
                None, qdepth - 1, ply + 1,
            );
            chess_board.undo_move();

            if IS_WHITE {
                if eval >= beta {
                    evasions.clear();
                    ctx.move_lists[ply.min(MAX_PLY - 1)] = evasions;
                    return eval;
                }
                if eval > best { best = eval; }
                if eval > alpha { alpha = eval; }
            } else {
                if eval <= alpha {
                    evasions.clear();
                    ctx.move_lists[ply.min(MAX_PLY - 1)] = evasions;
                    return eval;
                }
                if eval < best { best = eval; }
                if eval < beta { beta = eval; }
            }
        }
        evasions.clear();
        ctx.move_lists[ply.min(MAX_PLY - 1)] = evasions;
        return best;
    }

    // ── Capture-only (normal node) ────────────────────────────────────────────
    // Reuse the eval passed in from the parent (avoids a redundant eval_node
    // call at the top-level qsearch entry); recursive calls pass None.
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

    // Score each capture by SEE (net material exchange) and sort best-first.
    //
    // Previously, captures.sort() was effectively a no-op: ChessMove::capture is
    // never set by the move generator, so every capture sorted equally (score 0),
    // producing arbitrary/generator order.  Proper SEE ordering lets alpha-beta
    // find cutoffs much earlier in qsearch — typically the most valuable winning
    // capture raises alpha/lowers beta quickly, pruning the remaining list.
    //
    // Promotions: SEE treats the attacker as a pawn throughout and misses the
    // pawn→queen gain.  We compensate by adding 800 cp (queen 900 − pawn 100)
    // so promotion captures sort to the front as they deserve.
    //
    // Buffer is reused across recursive qsearch calls via the ordering scratch:
    // each level takes the Vec, clears, populates, and returns it, so no heap
    // allocation occurs in the hot path.
    let mut scored = std::mem::take(&mut ctx.ordering_scratch.good_captures);
    scored.clear();
    for &mv in captures.iter() {
        let score = if mv.is_promotion() {
            capture_value(chess_board, &mv) + 800
        } else {
            see(
                chess_board,
                conductor,
                mv.start_square() as usize,
                mv.target_square() as usize,
                IS_WHITE,
            )
        };
        scored.push((score, mv));
    }
    scored.sort_unstable_by(|a, b| b.0.cmp(&a.0));
    captures.clear();
    ctx.move_lists[ply.min(MAX_PLY - 1)] = captures;

    for &(see_val, chess_move) in scored.iter() {
        let mut chess_move = chess_move;

        // Losing captures: break (list is sorted by SEE desc, so all remaining
        // are also losing).
        if see_val < 0 {
            break;
        }

        // Delta pruning: even gaining `see_val` plus a safety margin cannot raise
        // alpha (white) or lower beta (black).  Break because every subsequent
        // capture has see_val ≤ this one (sorted), so the condition holds for all.
        if IS_WHITE {
            if stand_pat + see_val + ctx.params.delta_margin <= alpha {
                break;
            }
        } else {
            if stand_pat - see_val - ctx.params.delta_margin >= beta {
                break;
            }
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
                ctx.ordering_scratch.good_captures = scored;
                return eval;
            }
            if eval > best { best = eval; }
            if eval > alpha { alpha = eval; }
        } else {
            if eval <= alpha {
                ctx.ordering_scratch.good_captures = scored;
                return eval;
            }
            if eval < best { best = eval; }
            if eval < beta { beta = eval; }
        }
    }
    ctx.ordering_scratch.good_captures = scored;
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

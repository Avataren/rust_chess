use chess_board::ChessBoard;
use chess_foundation::ChessMove;
use move_generator::piece_conductor::PieceConductor;
use crate::see::see;
use super::{piece_idx, ContHistTable};

// ── Move ordering helpers ─────────────────────────────────────────────────────

/// Read-only heuristic inputs bundled for `order_moves`.
/// Groups the six parameters that come from the search context's history tables,
/// reducing `order_moves` from 17 parameters to 9.
pub(super) struct MoveOrderingHeuristics<'a> {
    pub history:         &'a [[i32; 64]; 64],
    pub capture_history: &'a [[i32; 64]; 64],
    pub cont_hist_1:     &'a ContHistTable,
    pub cont_hist_2:     &'a ContHistTable,
    /// (piece_type_idx, to_sq) of the opponent's last move (1 ply back).
    pub prev1: Option<(usize, usize)>,
    /// (piece_type_idx, to_sq) of our own last move (2 plies back).
    pub prev2: Option<(usize, usize)>,
}

/// Heap-allocated scratch buffers for `order_moves` — reused each call to
/// avoid per-call allocations in the hot search path.
#[derive(Default)]
pub(super) struct OrderingScratchBuffers {
    pub good_captures:  Vec<(i32, ChessMove)>,
    pub bad_captures:   Vec<(i32, ChessMove)>,
    pub quiets:         Vec<ChessMove>,
    pub killer_entries: Vec<ChessMove>,
}

// ── Move ordering ─────────────────────────────────────────────────────────────

/// Order moves for best-first search:
///   1. TT / PV move (if present)
///   2. Winning/even captures (SEE ≥ 0), sorted by SEE score descending
///   3. Killer moves (quiet moves that caused β-cutoffs at this ply)
///   4. Countermove (quiet move that refuted the opponent's last move)
///   5. Remaining quiet moves, sorted by history score (descending)
///   6. Losing captures (SEE < 0), sorted by SEE score descending
///
/// Separating losing captures from killers/history-sorted quiets is important:
/// a quiet killer or well-history-scored move is usually better than a losing exchange.
pub(super) fn order_moves(
    moves: &mut Vec<ChessMove>,
    tt_move: Option<ChessMove>,
    killers: &[Option<ChessMove>; 2],
    countermove: Option<ChessMove>,
    h: &MoveOrderingHeuristics<'_>,
    board: &ChessBoard,
    conductor: &PieceConductor,
    is_white: bool,
    scratch: &mut OrderingScratchBuffers,
) {
    // Pull out the TT move first.
    let tt_entry = tt_move.and_then(|tt_m| {
        moves
            .iter()
            .position(|m| {
                m.start_square() == tt_m.start_square() && m.target_square() == tt_m.target_square()
            })
            .map(|i| moves.swap_remove(i))
    });

    scratch.good_captures.clear();
    scratch.bad_captures.clear();
    scratch.quiets.clear();

    for m in moves.drain(..) {
        // `m.capture` is never set by the move generator, so use board lookup.
        // EN_PASSANT_CAPTURE_FLAG marks en-passant captures (target square is empty).
        let mv_is_capture = m.has_flag(ChessMove::EN_PASSANT_CAPTURE_FLAG)
            || board.get_piece_at_square(m.target_square()).is_some();
        if mv_is_capture || m.is_promotion() {
            let mut score = see(
                board,
                conductor,
                m.start_square() as usize,
                m.target_square() as usize,
                is_white,
            );
            // Promotion: add queen-minus-pawn gain so promotions are
            // valued correctly (SEE only accounts for the pawn on the board).
            if m.is_promotion() {
                score += crate::see::SEE_QUEEN - crate::see::SEE_PAWN;
            }
            if score >= 0 {
                scratch.good_captures.push((score, m));
            } else {
                scratch.bad_captures.push((score, m));
            }
        } else {
            scratch.quiets.push(m);
        }
    }

    // Sort by SEE descending; use capture_history as tiebreaker.
    scratch.good_captures.sort_unstable_by(|a, b| {
        b.0.cmp(&a.0).then_with(|| {
            let ha = h.capture_history[a.1.start_square() as usize][a.1.target_square() as usize];
            let hb = h.capture_history[b.1.start_square() as usize][b.1.target_square() as usize];
            hb.cmp(&ha)
        })
    });
    scratch.bad_captures.sort_unstable_by(|a, b| {
        b.0.cmp(&a.0).then_with(|| {
            let ha = h.capture_history[a.1.start_square() as usize][a.1.target_square() as usize];
            let hb = h.capture_history[b.1.start_square() as usize][b.1.target_square() as usize];
            hb.cmp(&ha)
        })
    });

    // Extract killer moves from quiets, preserving killer priority order.
    scratch.killer_entries.clear();
    for killer in killers.iter().flatten() {
        if let Some(pos) = scratch.quiets.iter().position(|m| {
            m.start_square() == killer.start_square() && m.target_square() == killer.target_square()
        }) {
            let item = scratch.quiets.swap_remove(pos);
            scratch.killer_entries.push(item);
        }
    }

    // Extract countermove (if not already pulled out as TT move or killer).
    let countermove_entry: Option<ChessMove> = countermove.and_then(|cm| {
        let pos = scratch.quiets.iter().position(|m| {
            m.start_square() == cm.start_square() && m.target_square() == cm.target_square()
        })?;
        Some(scratch.quiets.swap_remove(pos))
    });

    // Sort remaining quiets by combined history + continuation history score, highest first.
    scratch.quiets.sort_by(|a, b| {
        let quiet_score = |m: &ChessMove| -> i32 {
            let from = m.start_square() as usize;
            let to = m.target_square() as usize;
            let piece = piece_idx(*m);
            let mut score = h.history[from][to];
            if let Some((pp, pt)) = h.prev1 {
                score += h.cont_hist_1.get(pp, pt, piece, to);
            }
            if let Some((pp, pt)) = h.prev2 {
                score += h.cont_hist_2.get(pp, pt, piece, to);
            }
            score
        };
        quiet_score(b).cmp(&quiet_score(a))
    });

    // Reassemble: TT → good captures → killers → countermove → history quiets → bad captures.
    if let Some(tt_m) = tt_entry {
        moves.push(tt_m);
    }
    moves.extend(scratch.good_captures.iter().map(|(_, m)| *m));
    moves.extend(scratch.killer_entries.iter().copied());
    if let Some(cm) = countermove_entry {
        moves.push(cm);
    }
    moves.extend(scratch.quiets.drain(..));
    moves.extend(scratch.bad_captures.iter().map(|(_, m)| *m));
}

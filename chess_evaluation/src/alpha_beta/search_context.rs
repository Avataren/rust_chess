use chess_board::ChessBoard;
use chess_foundation::{ChessMove, piece::PieceType};
use crate::neural_eval::{HIDDEN1 as ACCUM_DIM, KING_BUCKETS};
use super::{MAX_PLY, ACC_SIZE, OrderingScratchBuffers, search_params::SearchParams};

// ── Continuation history ──────────────────────────────────────────────────────

/// Continuation history table.
/// Indexed logically as [prev_piece_type 0-6][prev_to 0-63][curr_piece_type 0-6][curr_to 0-63].
/// Flat Vec for heap allocation (avoids stack overflow from large array).
/// Size: 7 × 64 × 7 × 64 × 4 bytes = ~784 KB per table.
const CONT_HIST_PIECE: usize = 7;
const CONT_HIST_SQ: usize = 64;
const CONT_HIST_SIZE: usize = CONT_HIST_PIECE * CONT_HIST_SQ * CONT_HIST_PIECE * CONT_HIST_SQ;

pub struct ContHistTable {
    data: Vec<i32>,
}

impl ContHistTable {
    pub fn new() -> Self {
        Self {
            data: vec![0; CONT_HIST_SIZE],
        }
    }

    #[inline(always)]
    fn idx(pp: usize, pt: usize, cp: usize, ct: usize) -> usize {
        (pp * CONT_HIST_SQ + pt) * (CONT_HIST_PIECE * CONT_HIST_SQ) + cp * CONT_HIST_SQ + ct
    }

    #[inline(always)]
    pub fn get(&self, pp: usize, pt: usize, cp: usize, ct: usize) -> i32 {
        self.data[Self::idx(pp, pt, cp, ct)]
    }

    #[inline(always)]
    pub fn get_mut(&mut self, pp: usize, pt: usize, cp: usize, ct: usize) -> &mut i32 {
        &mut self.data[Self::idx(pp, pt, cp, ct)]
    }

    pub fn age(&mut self) {
        for v in &mut self.data {
            *v >>= 1;
        }
    }
}

/// Map a ChessMove's piece to a cont-hist index (0-6).
#[inline(always)]
pub(in crate::alpha_beta) fn piece_idx(mv: ChessMove) -> usize {
    mv.chess_piece
        .map(|cp| cp.piece_type() as usize)
        .unwrap_or(0)
        .min(6)
}

// ── HalfKP piece slot helpers ─────────────────────────────────────────────────

/// Map a piece type + ownership flag to a HalfKP slot index (0-11).
/// Ours (is_ours=true): Pawn=0, Knight=1, Bishop=2, Rook=3, Queen=4, King=5.
/// Theirs (is_ours=false): same +6.
#[inline(always)]
pub(in crate::alpha_beta) fn halfkp_piece_slot(pt: PieceType, is_ours: bool) -> usize {
    let base = match pt {
        PieceType::Pawn => 0,
        PieceType::Knight => 1,
        PieceType::Bishop => 2,
        PieceType::Rook => 3,
        PieceType::Queen => 4,
        PieceType::King => 5,
        PieceType::None => 0,
    };
    if is_ours {
        base
    } else {
        base + 6
    }
}

/// Compute the HalfKP accumulator feature index for a piece on `square` with
/// king bucket `bucket`, belonging to piece-type group `slot`.
///
/// Layout: `slot * 64 * KING_BUCKETS + square * KING_BUCKETS + bucket`.
/// Centralised here so a single typo can't silently corrupt incremental updates.
#[inline(always)]
pub(in crate::alpha_beta) fn halfkp_feature_idx(slot: usize, square: usize, bucket: usize) -> usize {
    slot * 64 * KING_BUCKETS + square * KING_BUCKETS + bucket
}

// ── HalfKAv2 piece slot helpers ──────────────────────────────────────────────

/// Map a piece type + ownership flag to a HalfKAv2 slot index (0-10).
/// Own (is_ours=true): Pawn=0, Knight=1, Bishop=2, Rook=3, Queen=4. King=EXCLUDED (usize::MAX).
/// Theirs (is_ours=false): Pawn=5, Knight=6, Bishop=7, Rook=8, Queen=9, King=10.
///
/// Returns usize::MAX for own king — callers must skip features with this slot.
#[inline(always)]
pub(in crate::alpha_beta) fn halfkav2_piece_slot(pt: PieceType, is_ours: bool) -> usize {
    if is_ours {
        match pt {
            PieceType::Pawn   => 0,
            PieceType::Knight => 1,
            PieceType::Bishop => 2,
            PieceType::Rook   => 3,
            PieceType::Queen  => 4,
            PieceType::King   => usize::MAX, // own king not a feature in HalfKAv2
            PieceType::None   => 0,
        }
    } else {
        match pt {
            PieceType::Pawn   => 5,
            PieceType::Knight => 6,
            PieceType::Bishop => 7,
            PieceType::Rook   => 8,
            PieceType::Queen  => 9,
            PieceType::King   => 10,
            PieceType::None   => 5,
        }
    }
}

/// Compute the HalfKAv2 accumulator feature index.
///
/// Layout: `slot * 64 * 64 + square * 64 + king_sq`.
/// `king_sq` is the exact (possibly file-mirrored) king square, not a bucket.
#[inline(always)]
pub(in crate::alpha_beta) fn halfkav2_feature_idx(slot: usize, square: usize, king_sq: usize) -> usize {
    slot * 64 * 64 + square * 64 + king_sq
}

// ── Per-context eval cache ────────────────────────────────────────────────────

/// Number of (hash, score) slots in the per-context eval cache.
/// 16 K entries × 12 bytes = 192 KB — hot in L2/L3 after a few accesses.
/// Power of 2 so the index is a single bitwise AND.
/// Sweet spot measured empirically at depth 7: 64K > 128K > 16K > 256K.
/// Larger caches spill out of L3 and their miss-latency penalty exceeds the
/// hit-rate benefit.  768 KB fits comfortably in a typical L3 cache.
pub(super) const EVAL_CACHE_SIZE: usize = 1 << 16; // 65 536 entries × 12 B = 768 KB

// ── Search context (killers + history) ───────────────────────────────────────

/// Per-search state for move ordering heuristics.
///
/// * `killers`          — up to 2 quiet moves per ply that caused a β-cutoff.
/// * `history`          — `history[from][to]` accumulates `depth²` on β-cutoff.
/// * `capture_history`  — `capture_history[from][to]` same but for captures;
///                        used to break SEE ties in move ordering.
/// * `cont_hist_1`      — 1-ply continuation history: good responses to opponent's last move.
///                        Indexed by (prev_piece, prev_to, curr_piece, curr_to).
/// * `cont_hist_2`      — 2-ply continuation history: good follow-ups to our own last move.
/// * `countermoves`     — `countermoves[from][to]` is the quiet move that most
///                        recently refuted the opponent move (from→to).
/// * `prev_moves`       — the move played at each ply, used to index countermoves.
/// * `excluded_move`    — per-ply move excluded during singular extension search.
/// * `static_evals`    — cached static eval per ply for the improving flag.
///                       `i32::MIN` means "not computed / in check".
pub struct SearchContext {
    pub(in crate::alpha_beta) killers: [[Option<ChessMove>; 2]; MAX_PLY],
    pub(in crate::alpha_beta) history: [[i32; 64]; 64],
    pub(in crate::alpha_beta) capture_history: [[i32; 64]; 64],
    pub cont_hist_1: ContHistTable,
    pub cont_hist_2: ContHistTable,
    pub(in crate::alpha_beta) countermoves: Box<[[Option<ChessMove>; 64]; 64]>,
    pub(in crate::alpha_beta) prev_moves: [Option<ChessMove>; MAX_PLY],
    pub(in crate::alpha_beta) excluded_move: [Option<ChessMove>; MAX_PLY],
    pub(in crate::alpha_beta) static_evals: [i32; MAX_PLY],
    /// Total nodes visited (alpha_beta + quiescence calls).  Incremented at
    /// the top of each call.  Useful for NPS benchmarking.
    pub nodes: u64,

    // ── Incremental accumulator stack (Phase 4) ───────────────────────────
    // Pre-ReLU L1 accumulators for the dual-perspective neural model.
    // Stored as raw i16 (quantized, not dequantized) for SIMD efficiency.
    // Heap-allocated to avoid stack pressure (~80 KB vs ~330 KB for f32).
    // acc_white[ply] / acc_black[ply] reflect the board state at search ply `ply`.
    // acc_valid=true iff a dual model is loaded and init_accumulators has been called.
    pub acc_white: Box<[[i16; ACCUM_DIM]; ACC_SIZE]>,
    pub acc_black: Box<[[i16; ACCUM_DIM]; ACC_SIZE]>,
    pub acc_valid: bool,
    /// Per-ply move lists, reused across depths.
    pub(in crate::alpha_beta) move_lists: Vec<Vec<ChessMove>>,
    /// Scratch buffer for pseudo-legal move generation per piece.
    pub pseudo_buf: Vec<ChessMove>,
    /// Scratch buffers for move ordering — reused each call.
    pub(in crate::alpha_beta) ordering_scratch: OrderingScratchBuffers,
    /// Reusable buffer for quiet moves tried before a beta-cutoff.
    pub(in crate::alpha_beta) tried_quiets_buf: Vec<ChessMove>,
    /// Reusable buffer for capture moves tried before a beta-cutoff.
    /// Parallel to `tried_quiets_buf`: captures that failed get a capture_history malus.
    pub(in crate::alpha_beta) tried_captures_buf: Vec<ChessMove>,
    /// The noise-selected move from the most recently completed `search_root` call.
    /// Set by `search_root` when `RootNoiseConfig` has noise enabled; otherwise equals
    /// the true best move.  Used by `id_search_single` for the final move returned to
    /// the caller so that noise never degrades iterative-deepening move ordering.
    pub(in crate::alpha_beta) noisy_move: Option<ChessMove>,
    /// Lightweight eval cache: `eval_cache[hash & (EVAL_CACHE_SIZE-1)] = (hash, score)`.
    /// Avoids redundant NN forward passes for positions revisited within a search
    /// (especially qsearch transpositions).  Cleared in `init_accumulators`.
    pub(in crate::alpha_beta) eval_cache: Vec<(u64, i32)>,
    /// Tunable pruning parameters. Set via `SearchContext::with_params`; defaults
    /// reproduce the hardcoded baseline when constructed with `SearchContext::new()`.
    pub params: SearchParams,
}

impl SearchContext {
    pub fn new() -> Self {
        Self::with_params(SearchParams::default())
    }

    /// Create a context with custom pruning parameters.
    /// All heuristic tables start zeroed; `params` override the hardcoded defaults.
    pub fn with_params(params: SearchParams) -> Self {
        Self {
            killers: [[None; 2]; MAX_PLY],
            history: [[0; 64]; 64],
            capture_history: [[0; 64]; 64],
            cont_hist_1: ContHistTable::new(),
            cont_hist_2: ContHistTable::new(),
            countermoves: Box::new([[None; 64]; 64]),
            prev_moves: [None; MAX_PLY],
            excluded_move: [None; MAX_PLY],
            static_evals: [i32::MIN; MAX_PLY],
            nodes: 0,
            acc_white: Box::new([[0i16; ACCUM_DIM]; ACC_SIZE]),
            acc_black: Box::new([[0i16; ACCUM_DIM]; ACC_SIZE]),
            acc_valid: false,
            move_lists: (0..MAX_PLY + 16).map(|_| Vec::with_capacity(64)).collect(),
            pseudo_buf: Vec::with_capacity(64),
            ordering_scratch: OrderingScratchBuffers {
                good_captures:  Vec::with_capacity(32),
                bad_captures:   Vec::with_capacity(16),
                quiets:         Vec::with_capacity(64),
                killer_entries: Vec::with_capacity(4),
            },
            tried_quiets_buf: Vec::with_capacity(16),
            tried_captures_buf: Vec::with_capacity(16),
            eval_cache: vec![(0u64, 0i32); EVAL_CACHE_SIZE],
            noisy_move: None,
            params,
        }
    }

    /// Initialize accumulators from the root board position.
    /// Also clears the eval cache so stale entries don't survive a model reload.
    pub fn init_accumulators(&mut self, board: &ChessBoard) {
        self.eval_cache.fill((0, 0));
        #[cfg(feature = "nn-incremental")]
        {
            self.acc_valid = crate::neural_eval::init_accumulators_direct(
                board,
                &mut self.acc_white[0],
                &mut self.acc_black[0],
            );
            return;
        }
        #[cfg(not(feature = "nn-incremental"))]
        {
            self.acc_valid = crate::neural_eval::init_accumulators_for_board(
                board,
                &mut self.acc_white[0],
                &mut self.acc_black[0],
            );
        }
    }

    /// Push accumulator state to ply+1 with an incremental delta for the given move.
    /// Call BEFORE make_move.  Returns true when the king moved (caller must call
    /// acc_recompute after make_move since the king bucket/square changes).
    pub fn acc_push(&mut self, ply: usize, mv: &ChessMove, board: &ChessBoard) -> bool {
        if !self.acc_valid {
            return false;
        }
        let src = ply.min(ACC_SIZE - 1);
        let dst = (ply + 1).min(ACC_SIZE - 1);

        // Copy parent accumulator to child ply (arrays are Copy)
        let tmp_w = self.acc_white[src];
        self.acc_white[dst] = tmp_w;
        let tmp_b = self.acc_black[src];
        self.acc_black[dst] = tmp_b;

        // Identify the moving piece.
        let moving_piece = match mv.chess_piece {
            Some(p) => p,
            None => match board.get_piece_at_square(mv.start_square()) {
                Some(p) => p,
                None => return true, // no piece at source — shouldn't happen; full recompute
            },
        };

        let from_sq = mv.start_square() as usize;
        let to_sq = mv.target_square() as usize;
        let piece_is_white = moving_piece.is_white();

        let use_kav2 = crate::neural_eval::is_halfkav2();

        // Current king squares / buckets
        let wk_sq = (board.get_white() & board.get_kings()).0.trailing_zeros() as usize;
        let bk_sq_raw = (board.get_black() & board.get_kings()).0.trailing_zeros() as usize;
        let bk_flipped = bk_sq_raw ^ 56;

        // King move handling
        if moving_piece.piece_type() == PieceType::King {
            if use_kav2 {
                // HalfKAv2: exact king_sq appears in EVERY feature index, so any king
                // move — regardless of distance or direction — requires full recompute.
                return true;
            }
            // HalfKP: full recompute only when bucket or horizontal-mirror flag changes.
            let wk_bucket = crate::neural_eval::KING_BUCKET[wk_sq.min(63)];
            let bk_bucket = crate::neural_eval::KING_BUCKET[bk_flipped.min(63)];
            let new_own_bucket = if piece_is_white {
                crate::neural_eval::KING_BUCKET[to_sq.min(63)]
            } else {
                let flipped = to_sq ^ 56;
                crate::neural_eval::KING_BUCKET[flipped.min(63)]
            };
            let current_own_bucket = if piece_is_white { wk_bucket } else { bk_bucket };
            if new_own_bucket != current_own_bucket {
                return true;
            }
            let old_mirror = if piece_is_white {
                (wk_sq.min(63) % 8) >= 4
            } else {
                (bk_flipped.min(63) % 8) >= 4
            };
            let new_mirror = if piece_is_white {
                (to_sq.min(63) % 8) >= 4
            } else {
                ((to_sq ^ 56).min(63) % 8) >= 4
            };
            if old_mirror != new_mirror {
                return true;
            }
            // Same bucket AND same mirror: fall through to incremental update.
        }

        // Horizontal mirroring: flip piece file bits when king is on files e-h.
        let mirror_w = (wk_sq.min(63) % 8) >= 4;
        let mirror_b = (bk_flipped.min(63) % 8) >= 4;
        let w_sq = |sq: usize| if mirror_w { sq ^ 7 } else { sq };
        let b_sq = |sq: usize| {
            let r = sq ^ 56;
            if mirror_b { r ^ 7 } else { r }
        };

        let acc_w = &mut self.acc_white[dst];
        let acc_b = &mut self.acc_black[dst];

        let orig_pt = moving_piece.piece_type();
        let to_pt = if mv.is_promotion() {
            mv.promotion_piece_type().unwrap_or(PieceType::Pawn)
        } else {
            orig_pt
        };

        let mut w_cap = [0usize; 1];
        let mut b_cap = [0usize; 1];
        let cap_n;

        if use_kav2 {
            // ── HalfKAv2 incremental path ─────────────────────────────────────
            let king_w = if mirror_w { wk_sq ^ 7 } else { wk_sq };
            let king_b = if mirror_b { bk_flipped ^ 7 } else { bk_flipped };

            let slot_w = halfkav2_piece_slot(orig_pt, piece_is_white);
            let slot_b = halfkav2_piece_slot(orig_pt, !piece_is_white);
            // Own-king slot returns usize::MAX — king moves already returned above.
            let w_sub_pre = halfkav2_feature_idx(slot_w, w_sq(from_sq), king_w);
            let b_sub_pre = halfkav2_feature_idx(slot_b, b_sq(from_sq), king_b);

            let to_slot_w = halfkav2_piece_slot(to_pt, piece_is_white);
            let to_slot_b = halfkav2_piece_slot(to_pt, !piece_is_white);
            let w_add = halfkav2_feature_idx(to_slot_w, w_sq(to_sq), king_w);
            let b_add = halfkav2_feature_idx(to_slot_b, b_sq(to_sq), king_b);

            if mv.has_flag(ChessMove::EN_PASSANT_CAPTURE_FLAG) {
                let cap_sq = if piece_is_white { to_sq.wrapping_sub(8) } else { to_sq + 8 };
                w_cap[0] = halfkav2_feature_idx(halfkav2_piece_slot(PieceType::Pawn, !piece_is_white), w_sq(cap_sq), king_w);
                b_cap[0] = halfkav2_feature_idx(halfkav2_piece_slot(PieceType::Pawn,  piece_is_white), b_sq(cap_sq), king_b);
                cap_n = 1;
            } else {
                match board.get_piece_at_square(mv.target_square()) {
                    Some(cap) => {
                        // King captures are pseudo-legal (shouldn't occur in
                        // legal play). halfkav2_piece_slot returns usize::MAX
                        // for the own-king side, which overflows in the
                        // feature-index multiply. Trigger a full recompute
                        // instead — the position is a decisive win/loss anyway.
                        if cap.piece_type() == PieceType::King {
                            return true;
                        }
                        let cap_slot_w = halfkav2_piece_slot(cap.piece_type(), cap.is_white());
                        let cap_slot_b = halfkav2_piece_slot(cap.piece_type(), !cap.is_white());
                        w_cap[0] = halfkav2_feature_idx(cap_slot_w, w_sq(to_sq), king_w);
                        b_cap[0] = halfkav2_feature_idx(cap_slot_b, b_sq(to_sq), king_b);
                        cap_n = 1;
                    }
                    None => cap_n = 0,
                }
            }

            crate::neural_eval::acc_apply_deltas(acc_w, &[w_sub_pre], &[w_add], &w_cap[..cap_n]);
            crate::neural_eval::acc_apply_deltas(acc_b, &[b_sub_pre], &[b_add], &b_cap[..cap_n]);
        } else {
            // ── HalfKP incremental path ───────────────────────────────────────
            let wk_bucket = crate::neural_eval::KING_BUCKET[wk_sq.min(63)];
            let bk_bucket = crate::neural_eval::KING_BUCKET[bk_flipped.min(63)];

            let slot_w = halfkp_piece_slot(orig_pt, piece_is_white);
            let slot_b = halfkp_piece_slot(orig_pt, !piece_is_white);
            let w_sub_pre = halfkp_feature_idx(slot_w, w_sq(from_sq), wk_bucket);
            let b_sub_pre = halfkp_feature_idx(slot_b, b_sq(from_sq), bk_bucket);

            let to_slot_w = halfkp_piece_slot(to_pt, piece_is_white);
            let to_slot_b = halfkp_piece_slot(to_pt, !piece_is_white);
            let w_add = halfkp_feature_idx(to_slot_w, w_sq(to_sq), wk_bucket);
            let b_add = halfkp_feature_idx(to_slot_b, b_sq(to_sq), bk_bucket);

            if mv.has_flag(ChessMove::EN_PASSANT_CAPTURE_FLAG) {
                let cap_sq = if piece_is_white { to_sq.wrapping_sub(8) } else { to_sq + 8 };
                w_cap[0] = halfkp_feature_idx(halfkp_piece_slot(PieceType::Pawn, !piece_is_white), w_sq(cap_sq), wk_bucket);
                b_cap[0] = halfkp_feature_idx(halfkp_piece_slot(PieceType::Pawn,  piece_is_white), b_sq(cap_sq), bk_bucket);
                cap_n = 1;
            } else {
                match board.get_piece_at_square(mv.target_square()) {
                    Some(cap) => {
                        let cap_slot_w = halfkp_piece_slot(cap.piece_type(), cap.is_white());
                        let cap_slot_b = halfkp_piece_slot(cap.piece_type(), !cap.is_white());
                        w_cap[0] = halfkp_feature_idx(cap_slot_w, w_sq(to_sq), wk_bucket);
                        b_cap[0] = halfkp_feature_idx(cap_slot_b, b_sq(to_sq), bk_bucket);
                        cap_n = 1;
                    }
                    None => cap_n = 0,
                }
            }

            crate::neural_eval::acc_apply_deltas(acc_w, &[w_sub_pre], &[w_add], &w_cap[..cap_n]);
            crate::neural_eval::acc_apply_deltas(acc_b, &[b_sub_pre], &[b_add], &b_cap[..cap_n]);
        }

        false // no full recompute needed
    }

    /// Recompute accumulator at `ply` from scratch (called after king moves).
    pub fn acc_recompute(&mut self, ply: usize, board: &ChessBoard) {
        let p = ply.min(ACC_SIZE - 1);
        #[cfg(feature = "nn-incremental")]
        if !crate::neural_eval::init_accumulators_direct(
            board,
            &mut self.acc_white[p],
            &mut self.acc_black[p],
        ) {
            self.acc_valid = false;
        }
        #[cfg(not(feature = "nn-incremental"))]
        if !crate::neural_eval::init_accumulators_for_board(
            board,
            &mut self.acc_white[p],
            &mut self.acc_black[p],
        ) {
            self.acc_valid = false;
        }
    }

    /// Apply `make_move` while maintaining the incremental accumulator.
    ///
    /// Combines `acc_push → make_move → (acc_recompute if king moved)` into one
    /// call, eliminating the copy-pasted pattern across all search entry points.
    #[inline]
    pub fn make_move_with_acc(
        &mut self,
        ply: usize,
        mv: &mut ChessMove,
        board: &mut ChessBoard,
    ) {
        let king_moved = self.acc_push(ply, mv, board);
        board.make_move(mv);
        if king_moved {
            self.acc_recompute(ply + 1, board);
        }
    }

    /// Halve all history scores between ID iterations so that shallower
    /// searches don't drown out discoveries from the current depth.
    pub fn age_history(&mut self) {
        for row in &mut self.history {
            for v in row {
                *v >>= 1;
            }
        }
        for row in &mut self.capture_history {
            for v in row {
                *v >>= 1;
            }
        }
        self.cont_hist_1.age();
        self.cont_hist_2.age();
    }

    /// Record a move that caused a β-cutoff:
    /// update killer slots + history + continuation history (quiets),
    /// and the countermove table.
    pub(in crate::alpha_beta) fn record_cutoff(&mut self, ply: usize, depth: i32, mv: ChessMove) {
        let p = ply.min(MAX_PLY - 1);
        let k = &mut self.killers[p];
        // Only shift if this isn't already the first killer slot.
        if k[0].map_or(true, |k0| {
            k0.start_square() != mv.start_square() || k0.target_square() != mv.target_square()
        }) {
            k[1] = k[0];
            k[0] = Some(mv);
        }
        // Reward deeper cutoffs more, clamped to avoid history overflow.
        let bonus = depth * depth;
        let v = &mut self.history[mv.start_square() as usize][mv.target_square() as usize];
        *v = (*v + bonus).min(16_384);
        // Countermove: record mv as the refutation of the opponent's last move.
        if let Some(prev) = self.prev_moves[p] {
            self.countermoves[prev.start_square() as usize][prev.target_square() as usize] =
                Some(mv);
        }
        // Continuation history: reward this move given the previous moves.
        let mv_piece = piece_idx(mv);
        let mv_to = mv.target_square() as usize;
        // 1-ply: keyed on opponent's previous move (what they just played before us)
        if let Some(prev1) = self.prev_moves[p] {
            let pp1 = piece_idx(prev1);
            let pt1 = prev1.target_square() as usize;
            let v = self.cont_hist_1.get_mut(pp1, pt1, mv_piece, mv_to);
            *v = (*v + bonus).min(16_384);
        }
        // 2-ply: keyed on our own previous move (what we played 2 plies ago)
        if p >= 1 {
            if let Some(prev2) = self.prev_moves[p - 1] {
                let pp2 = piece_idx(prev2);
                let pt2 = prev2.target_square() as usize;
                let v = self.cont_hist_2.get_mut(pp2, pt2, mv_piece, mv_to);
                *v = (*v + bonus).min(16_384);
            }
        }
    }

    /// Apply a history malus (negative bonus) to a quiet move that was searched
    /// but failed to produce a cutoff.  Penalises moves tried before the
    /// actual β-cutoff move so they are ordered lower in future nodes.
    pub(in crate::alpha_beta) fn apply_history_malus(&mut self, ply: usize, depth: i32, mv: ChessMove) {
        let p = ply.min(MAX_PLY - 1);
        let malus = depth * depth;
        let v = &mut self.history[mv.start_square() as usize][mv.target_square() as usize];
        *v = (*v - malus).max(-16_384);
        // Also apply malus to continuation history.
        let mv_piece = piece_idx(mv);
        let mv_to = mv.target_square() as usize;
        if let Some(prev1) = self.prev_moves[p] {
            let pp1 = piece_idx(prev1);
            let pt1 = prev1.target_square() as usize;
            let v = self.cont_hist_1.get_mut(pp1, pt1, mv_piece, mv_to);
            *v = (*v - malus).max(-16_384);
        }
        if p >= 1 {
            if let Some(prev2) = self.prev_moves[p - 1] {
                let pp2 = piece_idx(prev2);
                let pt2 = prev2.target_square() as usize;
                let v = self.cont_hist_2.get_mut(pp2, pt2, mv_piece, mv_to);
                *v = (*v - malus).max(-16_384);
            }
        }
    }
}

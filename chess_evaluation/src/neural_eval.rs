//! Neural network evaluation for chess positions.
//!
//! Implements a small NNUE-like MLP trained with the Python pipeline in
//! `nn_training/`. Two model variants are supported:
//!
//! **Single-perspective (Phase 1):** `12288 → 512 → 32 → 1`
//!   Weights: `backbone_3_weight` shape (32, 512)
//!
//! **Dual-perspective (Phase 2+3):** `[12288|12288] → 512+512 → 32 → 1`
//!   Shared EmbeddingBag; two accumulators concat'd before fc2.
//!   Weights: `backbone_3_weight` shape (32, 1024)
//!   CP output is white-absolute (positive = good for white).
//!
//! Weights are loaded from an NPZ file exported by `scripts/export_weights.py`.
//!
//! ```ignore
//! init_neural_eval("path/to/nnue_like_weights.npz").unwrap();
//! set_neural_eval_enabled(true);
//! ```
//!
//! Score convention: returns centipawns from **white's perspective**
//! (positive = good for white), matching the existing `evaluate_board`.

use std::io::{Read, Seek};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::OnceLock;

use chess_board::ChessBoard;
use chess_foundation::bitboard::Bitboard;

// ── Architecture constants ────────────────────────────────────────────────

pub(crate) const HIDDEN1: usize = 512;
const HIDDEN2: usize = 32;
const HIDDEN1_DUAL: usize = HIDDEN1 * 2; // 1024

const HALFKP_FEATURE_DIM: usize = 12 * 64 * 16; // 12,288
const LEGACY_FEATURE_DIM: usize = 768;

// ── Global state ──────────────────────────────────────────────────────────

static NEURAL_ENABLED: AtomicBool = AtomicBool::new(false);
static EVALUATOR: OnceLock<NeuralEvaluator> = OnceLock::new();

/// Minimum WDL confidence to trust the NN score.
/// Default 0.0 = always trust NN (no fallback).
static CONFIDENCE_THRESHOLD: AtomicU32 = AtomicU32::new(0);

/// Load weights from an NPZ file path.
pub fn init_neural_eval(path: &str) -> Result<(), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("Failed to read {path}: {e}"))?;
    init_neural_eval_from_bytes(&bytes)
}

/// Load weights from an in-memory NPZ blob.
pub fn init_neural_eval_from_bytes(bytes: &[u8]) -> Result<(), String> {
    let evaluator = NeuralEvaluator::from_npz_bytes(bytes)?;
    EVALUATOR
        .set(evaluator)
        .map_err(|_| "Neural evaluator already initialized".into())
}

/// Enable or disable neural network evaluation at runtime.
pub fn set_neural_eval_enabled(enabled: bool) {
    NEURAL_ENABLED.store(enabled, Ordering::Relaxed);
}

/// Returns `true` if neural eval is currently enabled.
pub fn is_neural_eval_enabled() -> bool {
    NEURAL_ENABLED.load(Ordering::Relaxed)
}

/// Set the minimum WDL confidence required to use the NN score.
pub fn set_neural_confidence_threshold(threshold: f32) {
    CONFIDENCE_THRESHOLD.store(threshold.to_bits(), Ordering::Relaxed);
}

pub fn get_neural_confidence_threshold() -> f32 {
    f32::from_bits(CONFIDENCE_THRESHOLD.load(Ordering::Relaxed))
}

/// Returns `Some(score)` in centipawns from **white's perspective** when
/// neural eval is enabled, weights are loaded, and confidence passes.
#[inline]
pub fn try_neural_eval(board: &ChessBoard) -> Option<i32> {
    if !NEURAL_ENABLED.load(Ordering::Relaxed) {
        return None;
    }
    let threshold = f32::from_bits(CONFIDENCE_THRESHOLD.load(Ordering::Relaxed));
    EVALUATOR.get().and_then(|e| {
        let (score, confidence) = e.evaluate_with_confidence(board);
        if confidence < threshold {
            return None;
        }
        // Dual model returns white-absolute; single-perspective model returns stm-relative.
        if e.dual_perspective {
            Some(score) // already white-absolute
        } else {
            Some(if board.is_white_active() { score } else { -score })
        }
    })
}

/// Try neural eval using pre-computed accumulators (Phase 3 incremental path).
/// Returns white-absolute centipawns or None if unavailable.
#[inline]
pub fn try_neural_eval_accum(
    acc_white: &[f32; HIDDEN1],
    acc_black: &[f32; HIDDEN1],
    _is_white: bool,
) -> Option<i32> {
    if !NEURAL_ENABLED.load(Ordering::Relaxed) {
        return None;
    }
    let threshold = f32::from_bits(CONFIDENCE_THRESHOLD.load(Ordering::Relaxed));
    EVALUATOR.get().and_then(|e| {
        if !e.dual_perspective {
            return None;
        }
        let (score, confidence) = e.evaluate_from_accumulators(acc_white, acc_black);
        if confidence < threshold {
            return None;
        }
        Some(score) // dual model output is always white-absolute
    })
}

/// Initialize both accumulators from scratch for the given board position.
/// Returns true iff neural eval is enabled, a dual model is loaded, and
/// initialization succeeded.
pub fn init_accumulators_for_board(
    board: &ChessBoard,
    acc_white: &mut [f32; HIDDEN1],
    acc_black: &mut [f32; HIDDEN1],
) -> bool {
    if !NEURAL_ENABLED.load(Ordering::Relaxed) {
        return false;
    }
    let evaluator = match EVALUATOR.get() {
        Some(e) if e.dual_perspective => e,
        _ => return false,
    };
    let ((w_idx, wc), (b_idx, bc)) = encode_dual_halfkp(board);

    acc_white.copy_from_slice(&evaluator.b1);
    for &i in &w_idx[..wc] {
        let src = &evaluator.w1_t[i * HIDDEN1..(i + 1) * HIDDEN1];
        for j in 0..HIDDEN1 {
            acc_white[j] += src[j];
        }
    }

    acc_black.copy_from_slice(&evaluator.b1);
    for &i in &b_idx[..bc] {
        let src = &evaluator.w1_t[i * HIDDEN1..(i + 1) * HIDDEN1];
        for j in 0..HIDDEN1 {
            acc_black[j] += src[j];
        }
    }
    true
}

/// Add a feature column from w1_t into an accumulator (in-place).
/// No-op if evaluator not loaded.
#[inline]
pub fn acc_add_feature(acc: &mut [f32; HIDDEN1], feature_idx: usize) {
    if let Some(e) = EVALUATOR.get() {
        let src = &e.w1_t[feature_idx * HIDDEN1..(feature_idx + 1) * HIDDEN1];
        for j in 0..HIDDEN1 {
            acc[j] += src[j];
        }
    }
}

/// Subtract a feature column from w1_t from an accumulator (in-place).
/// No-op if evaluator not loaded.
#[inline]
pub fn acc_sub_feature(acc: &mut [f32; HIDDEN1], feature_idx: usize) {
    if let Some(e) = EVALUATOR.get() {
        let src = &e.w1_t[feature_idx * HIDDEN1..(feature_idx + 1) * HIDDEN1];
        for j in 0..HIDDEN1 {
            acc[j] -= src[j];
        }
    }
}

// ── Evaluator ─────────────────────────────────────────────────────────────

pub struct NeuralEvaluator {
    /// Number of input features: 768 (legacy) or 12,288 (HalfKP).
    feature_dim: usize,

    /// True when `backbone_3_weight` is (32, 1024) instead of (32, 512).
    pub dual_perspective: bool,

    /// Layer 1 weights stored **transposed**: [feature_dim × HIDDEN1].
    /// Shared between both perspectives in dual mode.
    w1_t: Vec<f32>,
    b1: Vec<f32>,

    /// Layer 2 weights: [HIDDEN2 × HIDDEN1] (single) or [HIDDEN2 × HIDDEN1_DUAL] (dual).
    w2: Vec<f32>,
    b2: Vec<f32>,

    /// CP head: [HIDDEN2]
    w3: Vec<f32>,
    b3: f32,

    /// WDL head: [3 × HIDDEN2]
    w_wdl: Vec<f32>,
    b_wdl: Vec<f32>,
}

impl NeuralEvaluator {
    fn from_npz_bytes(data: &[u8]) -> Result<Self, String> {
        let cursor = std::io::Cursor::new(data);
        let mut zip = zip::ZipArchive::new(cursor)
            .map_err(|e| format!("Not a valid NPZ/zip file: {e}"))?;

        let scale = read_npy_f32_scalar(&mut zip, "scale.npy")?;

        let w1_i16   = read_npy_i16(&mut zip, "backbone_0_weight.npy")?;
        let b1_i16   = read_npy_i16(&mut zip, "backbone_0_bias.npy")?;
        let w2_i16   = read_npy_i16(&mut zip, "backbone_3_weight.npy")?;
        let b2_i16   = read_npy_i16(&mut zip, "backbone_3_bias.npy")?;
        let w3_i16   = read_npy_i16(&mut zip, "cp_head_weight.npy")?;
        let b3_i16   = read_npy_i16(&mut zip, "cp_head_bias.npy")?;
        let w_wdl_i16 = read_npy_i16(&mut zip, "wdl_head_weight.npy")?;
        let b_wdl_i16 = read_npy_i16(&mut zip, "wdl_head_bias.npy")?;

        let dq = |v: &[i16]| -> Vec<f32> { v.iter().map(|&x| x as f32 / scale).collect() };

        // Detect dual from w2 size: HIDDEN2×HIDDEN1_DUAL vs HIDDEN2×HIDDEN1
        let dual = w2_i16.len() == HIDDEN2 * HIDDEN1_DUAL;
        let expected_w2 = if dual { HIDDEN2 * HIDDEN1_DUAL } else { HIDDEN2 * HIDDEN1 };
        if w2_i16.len() != expected_w2 {
            return Err(format!(
                "Unexpected backbone_3 size {} (expected {} single or {} dual)",
                w2_i16.len(), HIDDEN2 * HIDDEN1, HIDDEN2 * HIDDEN1_DUAL
            ));
        }

        // Detect feature_dim from w1 weight shape
        let feature_dim = w1_i16.len() / HIDDEN1;
        if feature_dim != LEGACY_FEATURE_DIM && feature_dim != HALFKP_FEATURE_DIM {
            return Err(format!(
                "Unexpected feature_dim {feature_dim} (expected {LEGACY_FEATURE_DIM} or {HALFKP_FEATURE_DIM})"
            ));
        }

        // Transpose w1: [HIDDEN1 × feature_dim] → [feature_dim × HIDDEN1]
        let w1_row = dq(&w1_i16);
        let mut w1_t = vec![0.0f32; feature_dim * HIDDEN1];
        for j in 0..HIDDEN1 {
            for i in 0..feature_dim {
                w1_t[i * HIDDEN1 + j] = w1_row[j * feature_dim + i];
            }
        }

        Ok(Self {
            feature_dim,
            dual_perspective: dual,
            w1_t,
            b1: dq(&b1_i16),
            w2: dq(&w2_i16),
            b2: dq(&b2_i16),
            w3: dq(&w3_i16),
            b3: dq(&b3_i16)[0],
            w_wdl: dq(&w_wdl_i16),
            b_wdl: dq(&b_wdl_i16),
        })
    }

    /// Evaluate a position from scratch.
    ///
    /// Returns `(score, confidence)` where:
    /// - For dual model: `score` is centipawns from **white's** perspective.
    /// - For single model: `score` is centipawns from **side-to-move's** perspective.
    pub fn evaluate_with_confidence(&self, board: &ChessBoard) -> (i32, f32) {
        if self.dual_perspective {
            let ((w_idx, wc), (b_idx, bc)) = encode_dual_halfkp(board);

            let mut h_w = [0.0f32; HIDDEN1];
            let mut h_b = [0.0f32; HIDDEN1];
            h_w.copy_from_slice(&self.b1);
            h_b.copy_from_slice(&self.b1);

            for &i in &w_idx[..wc] {
                let src = &self.w1_t[i * HIDDEN1..(i + 1) * HIDDEN1];
                for j in 0..HIDDEN1 {
                    h_w[j] += src[j];
                }
            }
            for &i in &b_idx[..bc] {
                let src = &self.w1_t[i * HIDDEN1..(i + 1) * HIDDEN1];
                for j in 0..HIDDEN1 {
                    h_b[j] += src[j];
                }
            }
            for v in h_w.iter_mut() {
                *v = v.max(0.0);
            }
            for v in h_b.iter_mut() {
                *v = v.max(0.0);
            }
            self.forward_l2_heads_dual(&h_w, &h_b)
        } else {
            let active = if self.feature_dim == HALFKP_FEATURE_DIM {
                encode_active_features_halfkp(board)
            } else {
                encode_active_features_legacy(board)
            };

            let mut h1 = [0.0f32; HIDDEN1];
            h1.copy_from_slice(&self.b1);
            let (active_indices, active_count) = active;
            for &i in active_indices[..active_count].iter() {
                let src = &self.w1_t[i * HIDDEN1..(i + 1) * HIDDEN1];
                for j in 0..HIDDEN1 {
                    h1[j] += src[j];
                }
            }
            for v in h1.iter_mut() {
                *v = v.max(0.0);
            }
            self.forward_l2_heads_single(&h1)
        }
    }

    /// Evaluate from pre-ReLU accumulators (Phase 3 incremental path).
    /// Only valid for dual-perspective models.
    pub fn evaluate_from_accumulators(
        &self,
        acc_white: &[f32; HIDDEN1],
        acc_black: &[f32; HIDDEN1],
    ) -> (i32, f32) {
        debug_assert!(self.dual_perspective);
        let mut h_w = *acc_white;
        let mut h_b = *acc_black;
        for v in h_w.iter_mut() {
            *v = v.max(0.0);
        }
        for v in h_b.iter_mut() {
            *v = v.max(0.0);
        }
        self.forward_l2_heads_dual(&h_w, &h_b)
    }

    /// Layer 2 + heads for dual model: input is [h_w(512) | h_b(512)].
    fn forward_l2_heads_dual(&self, h_w: &[f32; HIDDEN1], h_b: &[f32; HIDDEN1]) -> (i32, f32) {
        let mut h2 = [0.0f32; HIDDEN2];
        for j in 0..HIDDEN2 {
            let row = &self.w2[j * HIDDEN1_DUAL..(j + 1) * HIDDEN1_DUAL];
            let mut acc = self.b2[j];
            for i in 0..HIDDEN1 {
                acc += row[i] * h_w[i];
                acc += row[HIDDEN1 + i] * h_b[i];
            }
            h2[j] = acc.max(0.0);
        }
        self.forward_heads(&h2)
    }

    /// Layer 2 + heads for single-perspective model: input is h1(512).
    fn forward_l2_heads_single(&self, h1: &[f32; HIDDEN1]) -> (i32, f32) {
        let mut h2 = [0.0f32; HIDDEN2];
        for j in 0..HIDDEN2 {
            let row = &self.w2[j * HIDDEN1..(j + 1) * HIDDEN1];
            let mut acc = self.b2[j];
            for i in 0..HIDDEN1 {
                acc += row[i] * h1[i];
            }
            h2[j] = acc.max(0.0);
        }
        self.forward_heads(&h2)
    }

    /// CP head + WDL head from h2.
    #[inline]
    fn forward_heads(&self, h2: &[f32; HIDDEN2]) -> (i32, f32) {
        let mut cp = self.b3;
        for i in 0..HIDDEN2 {
            cp += self.w3[i] * h2[i];
        }

        let mut logits = [0.0f32; 3];
        for k in 0..3 {
            let row = &self.w_wdl[k * HIDDEN2..(k + 1) * HIDDEN2];
            let mut acc = self.b_wdl[k];
            for i in 0..HIDDEN2 {
                acc += row[i] * h2[i];
            }
            logits[k] = acc;
        }
        let max_l = logits[0].max(logits[1]).max(logits[2]);
        let exps = [
            (logits[0] - max_l).exp(),
            (logits[1] - max_l).exp(),
            (logits[2] - max_l).exp(),
        ];
        let sum = exps[0] + exps[1] + exps[2];
        let confidence = exps[0].max(exps[1]).max(exps[2]) / sum;
        (cp.round() as i32, confidence)
    }
}

// ── Feature encoding ──────────────────────────────────────────────────────

pub(crate) const KING_BUCKET: [usize; 64] = {
    let mut t = [0usize; 64];
    let mut sq = 0usize;
    while sq < 64 {
        let file = sq % 8;
        let rank = sq / 8;
        let file_bucket = if file <= 3 { file } else { 7 - file };
        let rank_half = if rank <= 3 { 0 } else { 1 };
        t[sq] = rank_half * 4 + file_bucket;
        sq += 1;
    }
    t
};

/// Dual HalfKP encoding: returns ((white_indices, white_count), (black_indices, black_count)).
///
/// White perspective: absolute (white king = ours, white pieces = slots 0-5).
/// Black perspective: rank-flipped (black king = ours, black pieces = slots 0-5).
/// Both use the same feature formula: slot*64*16 + sq*16 + king_bucket.
pub(crate) fn encode_dual_halfkp(
    board: &ChessBoard,
) -> (([usize; 32], usize), ([usize; 32], usize)) {
    let white_bb = board.get_white();
    let black_bb = board.get_black();

    // White perspective: white king bucket (no rank flip)
    let wk_sq = (white_bb & board.get_kings()).0.trailing_zeros() as usize;
    let wk_sq = wk_sq.min(63);
    let wk_bucket = KING_BUCKET[wk_sq];

    // Black perspective: black king rank-flipped
    let bk_sq_raw = (black_bb & board.get_kings()).0.trailing_zeros() as usize;
    let bk_sq_raw = bk_sq_raw.min(63);
    let bk_bucket = KING_BUCKET[bk_sq_raw ^ 56];

    let mut w_indices = [0usize; 32];
    let mut b_indices = [0usize; 32];
    let mut wc = 0usize;
    let mut bc = 0usize;

    macro_rules! push_white {
        ($bb:expr, $slot:expr) => {
            let mut bb: Bitboard = $bb;
            while bb.0 != 0 {
                let sq = bb.0.trailing_zeros() as usize;
                bb.0 &= bb.0 - 1;
                if wc < 32 {
                    w_indices[wc] = $slot * 64 * 16 + sq * 16 + wk_bucket;
                    wc += 1;
                }
            }
        };
    }

    macro_rules! push_black {
        ($bb:expr, $slot:expr) => {
            let mut bb: Bitboard = $bb;
            while bb.0 != 0 {
                let sq = bb.0.trailing_zeros() as usize;
                bb.0 &= bb.0 - 1;
                if bc < 32 {
                    b_indices[bc] = $slot * 64 * 16 + (sq ^ 56) * 16 + bk_bucket;
                    bc += 1;
                }
            }
        };
    }

    // White perspective: white pieces = ours (0-5), black pieces = theirs (6-11)
    push_white!(white_bb & board.get_pawns(),    0);
    push_white!(white_bb & board.get_knights(),  1);
    push_white!(white_bb & board.get_bishops(),  2);
    push_white!(white_bb & board.get_rooks(),    3);
    push_white!(white_bb & board.get_queens(),   4);
    push_white!(white_bb & board.get_kings(),    5);
    push_white!(black_bb & board.get_pawns(),    6);
    push_white!(black_bb & board.get_knights(),  7);
    push_white!(black_bb & board.get_bishops(),  8);
    push_white!(black_bb & board.get_rooks(),    9);
    push_white!(black_bb & board.get_queens(),  10);
    push_white!(black_bb & board.get_kings(),   11);

    // Black perspective: black pieces = ours (0-5), white pieces = theirs (6-11), squares rank-flipped
    push_black!(black_bb & board.get_pawns(),    0);
    push_black!(black_bb & board.get_knights(),  1);
    push_black!(black_bb & board.get_bishops(),  2);
    push_black!(black_bb & board.get_rooks(),    3);
    push_black!(black_bb & board.get_queens(),   4);
    push_black!(black_bb & board.get_kings(),    5);
    push_black!(white_bb & board.get_pawns(),    6);
    push_black!(white_bb & board.get_knights(),  7);
    push_black!(white_bb & board.get_bishops(),  8);
    push_black!(white_bb & board.get_rooks(),    9);
    push_black!(white_bb & board.get_queens(),  10);
    push_black!(white_bb & board.get_kings(),   11);

    ((w_indices, wc), (b_indices, bc))
}

/// Legacy 768-dim encoder.
fn encode_active_features_legacy(board: &ChessBoard) -> ([usize; 32], usize) {
    let white_to_move = board.is_white_active();
    let flip = !white_to_move;
    let (ours, theirs) = if white_to_move {
        (board.get_white(), board.get_black())
    } else {
        (board.get_black(), board.get_white())
    };

    let mut indices = [0usize; 32];
    let mut count = 0usize;

    macro_rules! push_bb {
        ($bb:expr, $offset:expr) => {
            let mut bb: Bitboard = $bb;
            while bb.0 != 0 {
                let sq = bb.0.trailing_zeros() as usize;
                bb.0 &= bb.0 - 1;
                let mapped = if flip { sq ^ 56 } else { sq };
                indices[count] = $offset + mapped;
                count += 1;
            }
        };
    }

    push_bb!(ours   & board.get_pawns(),    0);
    push_bb!(ours   & board.get_knights(), 64);
    push_bb!(ours   & board.get_bishops(), 128);
    push_bb!(ours   & board.get_rooks(),   192);
    push_bb!(ours   & board.get_queens(),  256);
    push_bb!(ours   & board.get_kings(),   320);
    push_bb!(theirs & board.get_pawns(),   384);
    push_bb!(theirs & board.get_knights(), 448);
    push_bb!(theirs & board.get_bishops(), 512);
    push_bb!(theirs & board.get_rooks(),   576);
    push_bb!(theirs & board.get_queens(),  640);
    push_bb!(theirs & board.get_kings(),   704);

    (indices, count)
}

/// HalfKP 12,288-dim king-bucketed encoder (single-perspective, side-to-move normalized).
fn encode_active_features_halfkp(board: &ChessBoard) -> ([usize; 32], usize) {
    let white_to_move = board.is_white_active();
    let flip = !white_to_move;
    let (ours, theirs) = if white_to_move {
        (board.get_white(), board.get_black())
    } else {
        (board.get_black(), board.get_white())
    };

    let king_raw = (ours & board.get_kings()).0.trailing_zeros() as usize;
    let king_sq = if flip { king_raw ^ 56 } else { king_raw };
    let bucket = KING_BUCKET[king_sq.min(63)];

    let mut indices = [0usize; 32];
    let mut count = 0usize;

    macro_rules! push_bb_halfkp {
        ($bb:expr, $slot:expr) => {
            let mut bb: Bitboard = $bb;
            while bb.0 != 0 {
                let sq = bb.0.trailing_zeros() as usize;
                bb.0 &= bb.0 - 1;
                let mapped = if flip { sq ^ 56 } else { sq };
                indices[count] = $slot * 64 * 16 + mapped * 16 + bucket;
                count += 1;
            }
        };
    }

    push_bb_halfkp!(ours   & board.get_pawns(),    0);
    push_bb_halfkp!(ours   & board.get_knights(),  1);
    push_bb_halfkp!(ours   & board.get_bishops(),  2);
    push_bb_halfkp!(ours   & board.get_rooks(),    3);
    push_bb_halfkp!(ours   & board.get_queens(),   4);
    push_bb_halfkp!(ours   & board.get_kings(),    5);
    push_bb_halfkp!(theirs & board.get_pawns(),    6);
    push_bb_halfkp!(theirs & board.get_knights(),  7);
    push_bb_halfkp!(theirs & board.get_bishops(),  8);
    push_bb_halfkp!(theirs & board.get_rooks(),    9);
    push_bb_halfkp!(theirs & board.get_queens(),  10);
    push_bb_halfkp!(theirs & board.get_kings(),   11);

    (indices, count)
}

/// Dense feature vector — only used by unit tests (legacy 768-dim).
#[cfg(test)]
fn encode_features_legacy(board: &ChessBoard) -> [f32; LEGACY_FEATURE_DIM] {
    let mut feat = [0.0f32; LEGACY_FEATURE_DIM];
    let (indices, count) = encode_active_features_legacy(board);
    for i in indices[..count].iter().copied() {
        feat[i] = 1.0;
    }
    feat
}

/// Dense feature vector — only used by unit tests (HalfKP 12,288-dim).
#[cfg(test)]
fn encode_features_halfkp(board: &ChessBoard) -> [f32; HALFKP_FEATURE_DIM] {
    let mut feat = [0.0f32; HALFKP_FEATURE_DIM];
    let (indices, count) = encode_active_features_halfkp(board);
    for i in indices[..count].iter().copied() {
        feat[i] = 1.0;
    }
    feat
}

// ── NPY parsing ───────────────────────────────────────────────────────────

fn read_npy_i16<R: Read + Seek>(
    zip: &mut zip::ZipArchive<R>,
    name: &str,
) -> Result<Vec<i16>, String> {
    let mut buf = Vec::new();
    zip.by_name(name)
        .map_err(|_| format!("Missing array '{name}' in NPZ"))?
        .read_to_end(&mut buf)
        .map_err(|e| format!("Read error for '{name}': {e}"))?;
    parse_npy_i16(&buf, name)
}

fn read_npy_f32_scalar<R: Read + Seek>(
    zip: &mut zip::ZipArchive<R>,
    name: &str,
) -> Result<f32, String> {
    let mut buf = Vec::new();
    zip.by_name(name)
        .map_err(|_| format!("Missing array '{name}' in NPZ"))?
        .read_to_end(&mut buf)
        .map_err(|e| format!("Read error for '{name}': {e}"))?;
    parse_npy_f32(&buf, name)?
        .into_iter()
        .next()
        .ok_or_else(|| format!("'{name}' is empty"))
}

fn parse_npy_i16(buf: &[u8], name: &str) -> Result<Vec<i16>, String> {
    let (offset, n) = parse_npy_header(buf, name, "<i2")?;
    let data = &buf[offset..];
    if data.len() < n * 2 {
        return Err(format!("'{name}': data too short ({} bytes for {n} i16)", data.len()));
    }
    Ok(data[..n * 2]
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect())
}

fn parse_npy_f32(buf: &[u8], name: &str) -> Result<Vec<f32>, String> {
    let (offset, n) = parse_npy_header(buf, name, "<f4")?;
    let data = &buf[offset..];
    if data.len() < n * 4 {
        return Err(format!("'{name}': data too short ({} bytes for {n} f32)", data.len()));
    }
    Ok(data[..n * 4]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn parse_npy_header(buf: &[u8], name: &str, expected_dtype: &str) -> Result<(usize, usize), String> {
    if buf.len() < 10 || &buf[0..6] != b"\x93NUMPY" {
        return Err(format!("'{name}' is not a valid .npy file"));
    }
    let major = buf[6];
    let (header_len, header_start) = match major {
        1 => (u16::from_le_bytes([buf[8], buf[9]]) as usize, 10usize),
        2 => {
            if buf.len() < 12 {
                return Err(format!("'{name}': truncated v2 header"));
            }
            (u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize, 12usize)
        }
        v => return Err(format!("'{name}': unsupported .npy version {v}")),
    };

    let data_offset = header_start + header_len;
    if buf.len() < data_offset {
        return Err(format!("'{name}': file truncated before data"));
    }

    let header = std::str::from_utf8(&buf[header_start..data_offset])
        .map_err(|_| format!("'{name}': header is not valid UTF-8"))?;

    if !header.contains(expected_dtype) {
        return Err(format!(
            "'{name}': expected dtype '{expected_dtype}', got header: {header}"
        ));
    }

    let n = parse_shape_product(header, name)?;
    Ok((data_offset, n))
}

fn parse_shape_product(header: &str, name: &str) -> Result<usize, String> {
    let shape_start = header
        .find("'shape'")
        .or_else(|| header.find("\"shape\""))
        .ok_or_else(|| format!("'{name}': no 'shape' key in header"))?;
    let after = &header[shape_start..];
    let open = after
        .find('(')
        .ok_or_else(|| format!("'{name}': malformed shape tuple"))?;
    let close = after
        .find(')')
        .ok_or_else(|| format!("'{name}': malformed shape tuple"))?;
    let inner = &after[open + 1..close];

    let product: usize = inner
        .split(',')
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .product();

    Ok(if product == 0 { 1 } else { product })
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use chess_board::ChessBoard;

    // ── Legacy 768-dim tests ──────────────────────────────────────────────

    #[test]
    fn test_legacy_feature_count_starting_position() {
        let board = ChessBoard::new();
        let features = encode_features_legacy(&board);
        let active: usize = features.iter().filter(|&&v| v != 0.0).count();
        assert_eq!(active, 32, "Starting position should have 32 active features");
    }

    #[test]
    fn test_legacy_feature_count_black_to_move() {
        let mut board = ChessBoard::new();
        board.toggle_turn();
        let features = encode_features_legacy(&board);
        let active: usize = features.iter().filter(|&&v| v != 0.0).count();
        assert_eq!(active, 32);
    }

    #[test]
    fn test_legacy_white_king_feature_white_to_move() {
        let board = ChessBoard::new();
        assert!(board.is_white_active());
        let features = encode_features_legacy(&board);
        assert_eq!(features[320 + 4], 1.0, "White king should be at feature 324");
    }

    #[test]
    fn test_legacy_black_king_feature_black_to_move() {
        let mut board = ChessBoard::new();
        board.toggle_turn();
        let features = encode_features_legacy(&board);
        assert_eq!(features[320 + 4], 1.0, "Black king (mirrored) should be at feature 324");
    }

    // ── HalfKP 12,288-dim tests ───────────────────────────────────────────

    #[test]
    fn test_halfkp_feature_count_starting_position() {
        let board = ChessBoard::new();
        let features = encode_features_halfkp(&board);
        let active: usize = features.iter().filter(|&&v| v != 0.0).count();
        assert_eq!(active, 32, "HalfKP starting position should have 32 active features");
    }

    #[test]
    fn test_halfkp_feature_count_black_to_move() {
        let mut board = ChessBoard::new();
        board.toggle_turn();
        let features = encode_features_halfkp(&board);
        let active: usize = features.iter().filter(|&&v| v != 0.0).count();
        assert_eq!(active, 32, "HalfKP black-to-move should have 32 active features");
    }

    #[test]
    fn test_halfkp_features_in_range() {
        let board = ChessBoard::new();
        let features = encode_features_halfkp(&board);
        for (i, &v) in features.iter().enumerate() {
            if v != 0.0 {
                assert!(i < HALFKP_FEATURE_DIM, "HalfKP feature index {i} out of range");
            }
        }
    }

    #[test]
    fn test_halfkp_king_bucket_consistency() {
        for sq in 0..64 {
            assert!(KING_BUCKET[sq] < 16, "KING_BUCKET[{sq}] = {} out of range", KING_BUCKET[sq]);
        }
    }

    // ── Dual HalfKP tests ─────────────────────────────────────────────────

    #[test]
    fn test_dual_halfkp_feature_count_starting() {
        let board = ChessBoard::new();
        let ((w_idx, wc), (b_idx, bc)) = encode_dual_halfkp(&board);
        assert_eq!(wc, 32, "Dual white perspective should have 32 features");
        assert_eq!(bc, 32, "Dual black perspective should have 32 features");
        // All indices should be in range
        for &i in &w_idx[..wc] {
            assert!(i < HALFKP_FEATURE_DIM, "Dual white feature index {i} out of range");
        }
        for &i in &b_idx[..bc] {
            assert!(i < HALFKP_FEATURE_DIM, "Dual black feature index {i} out of range");
        }
    }

    #[test]
    fn test_dual_halfkp_symmetric_sides() {
        // From starting position: dual white perspective == dual black perspective (symmetric board)
        let board = ChessBoard::new();
        let ((w_idx, wc), (b_idx, bc)) = encode_dual_halfkp(&board);
        assert_eq!(wc, bc, "Symmetric board should have same feature count for both perspectives");
        // The feature sets should be identical for the starting position (symmetric)
        let mut w_set: Vec<_> = w_idx[..wc].to_vec();
        let mut b_set: Vec<_> = b_idx[..bc].to_vec();
        w_set.sort();
        b_set.sort();
        assert_eq!(w_set, b_set, "Starting position should be symmetric across perspectives");
    }

    #[test]
    fn test_dual_halfkp_black_to_move_same_features() {
        // Dual encoding is independent of side to move
        let board_w = ChessBoard::new();
        let mut board_b = ChessBoard::new();
        board_b.toggle_turn();
        let ((w_idx_w, wc_w), (b_idx_w, bc_w)) = encode_dual_halfkp(&board_w);
        let ((w_idx_b, wc_b), (b_idx_b, bc_b)) = encode_dual_halfkp(&board_b);
        // Features must be identical regardless of side to move
        assert_eq!(wc_w, wc_b);
        assert_eq!(bc_w, bc_b);
        assert_eq!(&w_idx_w[..wc_w], &w_idx_b[..wc_b]);
        assert_eq!(&b_idx_w[..bc_w], &b_idx_b[..bc_b]);
    }
}

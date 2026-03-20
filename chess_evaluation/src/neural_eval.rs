//! Neural network evaluation for chess positions.
//!
//! Implements a small NNUE-like MLP trained with the Python pipeline in
//! `nn_training/`. Two model variants are supported:
//!
//! **Single-perspective (Phase 1):** `12288 → 512 → 32 → 1 (×N output buckets)`
//!   Weights: `backbone_3_weight` shape (32, 512)
//!
//! **Dual-perspective (Phase 2+3):** `[12288|12288] → 512+512 → 32 → 1 (×N output buckets)`
//!   Shared EmbeddingBag; two accumulators concat'd before fc2.
//!   Weights: `backbone_3_weight` shape (32, 1024)
//!   CP output is white-absolute (positive = good for white).
//!   SCReLU activation: `clamp(x,0,1)²` at every activation site.
//!   Output buckets: separate weights per game phase (2–32 pieces → bucket 0–7).
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

// ── SCReLU activation: clamp(x,0,1)² ──────────────────────────────────────

/// SCReLU for the i16→f32 accumulator path: dequantize + clamp to [0,1] + square.
#[inline(always)]
fn screlu_i16(raw: i16, scale: f32) -> f32 {
    let c = (raw.max(0) as f32).min(scale);
    let x = c / scale;
    x * x
}

/// SCReLU for the f32 scratch path: clamp(x, 0, 1)².
#[inline(always)]
fn screlu_f32(x: f32) -> f32 {
    let c = x.max(0.0).min(1.0);
    c * c
}

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

/// Returns true if weights have been loaded (via EvalFile or embedded bytes).
pub fn is_neural_eval_initialized() -> bool {
    EVALUATOR.get().is_some()
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

/// Try neural eval using pre-computed accumulators (Phase 4 i16 incremental path).
/// Returns white-absolute centipawns or None if unavailable.
#[inline]
pub fn try_neural_eval_accum(
    board: &ChessBoard,
    acc_white: &[i16; HIDDEN1],
    acc_black: &[i16; HIDDEN1],
) -> Option<i32> {
    if !NEURAL_ENABLED.load(Ordering::Relaxed) {
        return None;
    }
    let threshold = f32::from_bits(CONFIDENCE_THRESHOLD.load(Ordering::Relaxed));
    EVALUATOR.get().and_then(|e| {
        if !e.dual_perspective {
            return None;
        }
        let bucket = piece_bucket(board, e.n_output_buckets);
        let (score, confidence) = e.evaluate_from_accumulators(acc_white, acc_black, bucket);
        if confidence < threshold {
            return None;
        }
        Some(score) // dual model output is always white-absolute
    })
}

/// Initialize both accumulators from scratch for the given board position.
/// Returns true iff neural eval is enabled, a dual model is loaded, and
/// initialization succeeded.  Accumulators hold raw i16 quantized values.
pub fn init_accumulators_for_board(
    board: &ChessBoard,
    acc_white: &mut [i16; HIDDEN1],
    acc_black: &mut [i16; HIDDEN1],
) -> bool {
    if !NEURAL_ENABLED.load(Ordering::Relaxed) {
        return false;
    }
    let evaluator = match EVALUATOR.get() {
        Some(e) if e.dual_perspective => e,
        _ => return false,
    };
    let ((w_idx, wc), (b_idx, bc)) = encode_dual_halfkp(board);

    acc_white.copy_from_slice(&evaluator.b1_i16);
    for &i in &w_idx[..wc] {
        let col = &evaluator.w1_t_i16[i * HIDDEN1..(i + 1) * HIDDEN1];
        add_col(acc_white, col);
    }

    acc_black.copy_from_slice(&evaluator.b1_i16);
    for &i in &b_idx[..bc] {
        let col = &evaluator.w1_t_i16[i * HIDDEN1..(i + 1) * HIDDEN1];
        add_col(acc_black, col);
    }
    true
}

// ── Compile-time-feature direct evaluation (no NEURAL_ENABLED check) ─────────
//
// These functions are used by the nn-full-forward and nn-incremental features.
// They bypass the NEURAL_ENABLED AtomicBool and confidence-threshold checks that
// exist for runtime switching (chess_uci). The weights are embedded at startup
// so the evaluator is always present when these features are selected.

/// Direct full-forward NN evaluation — no NEURAL_ENABLED check, no confidence
/// threshold.  Panics in debug if weights are not loaded.
#[cfg(any(feature = "nn-full-forward", feature = "nn-incremental"))]
#[inline]
pub fn eval_direct(board: &ChessBoard) -> i32 {
    let e = EVALUATOR.get()
        .expect("neural eval not loaded — call init_neural_eval_from_bytes at startup");
    let (score, _) = e.evaluate_with_confidence(board);
    if e.dual_perspective {
        score
    } else {
        if board.is_white_active() { score } else { -score }
    }
}

/// Direct accumulator-based evaluation — no runtime checks.
/// Only valid for dual-perspective models (eval.npz).
#[cfg(feature = "nn-incremental")]
#[inline]
pub fn eval_accum_direct(
    board: &ChessBoard,
    acc_white: &[i16; HIDDEN1],
    acc_black: &[i16; HIDDEN1],
) -> i32 {
    let e = EVALUATOR.get()
        .expect("neural eval not loaded — call init_neural_eval_from_bytes at startup");
    let bucket = piece_bucket(board, e.n_output_buckets);
    e.evaluate_from_accumulators(acc_white, acc_black, bucket).0
}

/// Initialize accumulators from a board position — no NEURAL_ENABLED check.
/// Returns false only if the loaded model is not dual-perspective.
#[cfg(feature = "nn-incremental")]
pub fn init_accumulators_direct(
    board: &ChessBoard,
    acc_white: &mut [i16; HIDDEN1],
    acc_black: &mut [i16; HIDDEN1],
) -> bool {
    let evaluator = match EVALUATOR.get() {
        Some(e) if e.dual_perspective => e,
        _ => return false,
    };
    let ((w_idx, wc), (b_idx, bc)) = encode_dual_halfkp(board);
    acc_white.copy_from_slice(&evaluator.b1_i16);
    for &i in &w_idx[..wc] {
        let col = &evaluator.w1_t_i16[i * HIDDEN1..(i + 1) * HIDDEN1];
        add_col(acc_white, col);
    }
    acc_black.copy_from_slice(&evaluator.b1_i16);
    for &i in &b_idx[..bc] {
        let col = &evaluator.w1_t_i16[i * HIDDEN1..(i + 1) * HIDDEN1];
        add_col(acc_black, col);
    }
    true
}

/// Add a feature column into an i16 accumulator (in-place, SIMD-dispatched).
/// No-op if evaluator not loaded.
#[inline]
pub fn acc_add_feature(acc: &mut [i16; HIDDEN1], feature_idx: usize) {
    if let Some(e) = EVALUATOR.get() {
        let col = &e.w1_t_i16[feature_idx * HIDDEN1..(feature_idx + 1) * HIDDEN1];
        add_col(acc, col);
    }
}

/// Subtract a feature column from an i16 accumulator (in-place, SIMD-dispatched).
/// No-op if evaluator not loaded.
#[inline]
pub fn acc_sub_feature(acc: &mut [i16; HIDDEN1], feature_idx: usize) {
    if let Some(e) = EVALUATOR.get() {
        let col = &e.w1_t_i16[feature_idx * HIDDEN1..(feature_idx + 1) * HIDDEN1];
        sub_col(acc, col);
    }
}

// ── SIMD column accumulator operations ───────────────────────────────────
//
// Three compile-time paths selected by cfg:
//   x86_64 + avx2    → _mm256_adds_epi16  (16 i16/reg, 32 instr/col)
//   wasm32 + simd128 → i16x8_add_sat       (8 i16/reg, 64 instr/col)
//   fallback         → scalar saturating_add (1 i16/iter, 512 instr/col)
//
// `add_col` / `sub_col` are the public dispatch functions used everywhere.

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
unsafe fn add_col_avx2(acc: &mut [i16; HIDDEN1], col: &[i16]) {
    use std::arch::x86_64::*;
    for i in (0..HIDDEN1).step_by(16) {
        let a = _mm256_loadu_si256(acc[i..].as_ptr() as *const __m256i);
        let b = _mm256_loadu_si256(col[i..].as_ptr() as *const __m256i);
        _mm256_storeu_si256(acc[i..].as_mut_ptr() as *mut __m256i, _mm256_adds_epi16(a, b));
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
unsafe fn sub_col_avx2(acc: &mut [i16; HIDDEN1], col: &[i16]) {
    use std::arch::x86_64::*;
    for i in (0..HIDDEN1).step_by(16) {
        let a = _mm256_loadu_si256(acc[i..].as_ptr() as *const __m256i);
        let b = _mm256_loadu_si256(col[i..].as_ptr() as *const __m256i);
        _mm256_storeu_si256(acc[i..].as_mut_ptr() as *mut __m256i, _mm256_subs_epi16(a, b));
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
unsafe fn add_col_wasm(acc: &mut [i16; HIDDEN1], col: &[i16]) {
    use core::arch::wasm32::*;
    for i in (0..HIDDEN1).step_by(8) {
        let a = v128_load(acc[i..].as_ptr() as *const v128);
        let b = v128_load(col[i..].as_ptr() as *const v128);
        v128_store(acc[i..].as_mut_ptr() as *mut v128, i16x8_add_sat(a, b));
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
unsafe fn sub_col_wasm(acc: &mut [i16; HIDDEN1], col: &[i16]) {
    use core::arch::wasm32::*;
    for i in (0..HIDDEN1).step_by(8) {
        let a = v128_load(acc[i..].as_ptr() as *const v128);
        let b = v128_load(col[i..].as_ptr() as *const v128);
        v128_store(acc[i..].as_mut_ptr() as *mut v128, i16x8_sub_sat(a, b));
    }
}

#[allow(dead_code)]
fn add_col_scalar(acc: &mut [i16; HIDDEN1], col: &[i16]) {
    for (a, &b) in acc.iter_mut().zip(col) {
        *a = a.saturating_add(b);
    }
}

#[allow(dead_code)]
fn sub_col_scalar(acc: &mut [i16; HIDDEN1], col: &[i16]) {
    for (a, &b) in acc.iter_mut().zip(col) {
        *a = a.saturating_sub(b);
    }
}

// Compile-time dispatch: each target gets exactly one definition.

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
fn add_col(acc: &mut [i16; HIDDEN1], col: &[i16]) {
    unsafe { add_col_avx2(acc, col) }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
fn add_col(acc: &mut [i16; HIDDEN1], col: &[i16]) {
    unsafe { add_col_wasm(acc, col) }
}

#[cfg(not(any(
    all(target_arch = "x86_64", target_feature = "avx2"),
    all(target_arch = "wasm32", target_feature = "simd128"),
)))]
#[inline(always)]
fn add_col(acc: &mut [i16; HIDDEN1], col: &[i16]) {
    add_col_scalar(acc, col)
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
fn sub_col(acc: &mut [i16; HIDDEN1], col: &[i16]) {
    unsafe { sub_col_avx2(acc, col) }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
fn sub_col(acc: &mut [i16; HIDDEN1], col: &[i16]) {
    unsafe { sub_col_wasm(acc, col) }
}

#[cfg(not(any(
    all(target_arch = "x86_64", target_feature = "avx2"),
    all(target_arch = "wasm32", target_feature = "simd128"),
)))]
#[inline(always)]
fn sub_col(acc: &mut [i16; HIDDEN1], col: &[i16]) {
    sub_col_scalar(acc, col)
}

// ── Evaluator ─────────────────────────────────────────────────────────────

pub struct NeuralEvaluator {
    /// Number of input features: 768 (legacy) or 12,288 (HalfKP).
    feature_dim: usize,

    /// True when `backbone_3_weight` is (32, 1024) instead of (32, 512).
    pub dual_perspective: bool,

    /// Quantization scale: raw_i16 / scale = f32 value.
    scale: f32,

    /// Layer 1 weights stored **transposed** as f32: [feature_dim × HIDDEN1].
    /// Used by the scratch evaluation path (evaluate_with_confidence).
    w1_t: Vec<f32>,
    b1: Vec<f32>,

    /// Layer 1 weights stored **transposed** as raw i16: [feature_dim × HIDDEN1].
    /// Used by the incremental SIMD path (acc_add/sub_feature, init_accumulators).
    w1_t_i16: Vec<i16>,
    /// Raw i16 biases for Layer 1 accumulator initialization.
    b1_i16: Vec<i16>,

    /// Layer 2 weights: [HIDDEN2 × HIDDEN1] (single) or [HIDDEN2 × HIDDEN1_DUAL] (dual).
    w2: Vec<f32>,
    b2: Vec<f32>,

    /// CP head: [n_output_buckets × HIDDEN2]
    w3: Vec<f32>,
    /// CP head biases: [n_output_buckets]
    b3: Vec<f32>,

    /// WDL head: [n_output_buckets × 3 × HIDDEN2]
    w_wdl: Vec<f32>,
    /// WDL head biases: [n_output_buckets × 3]
    b_wdl: Vec<f32>,

    /// Number of output buckets: 1 for old single-bucket models, 8 for new.
    pub n_output_buckets: usize,
}

impl NeuralEvaluator {
    fn from_npz_bytes(data: &[u8]) -> Result<Self, String> {
        let cursor = std::io::Cursor::new(data);
        let mut zip = zip::ZipArchive::new(cursor)
            .map_err(|e| format!("Not a valid NPZ/zip file: {e}"))?;

        let scale = read_npy_f32_scalar(&mut zip, "scale.npy")?;

        let w1_raw    = read_npy_i16(&mut zip, "backbone_0_weight.npy")?;
        let b1_raw    = read_npy_i16(&mut zip, "backbone_0_bias.npy")?;
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
        let feature_dim = w1_raw.len() / HIDDEN1;
        if feature_dim != LEGACY_FEATURE_DIM && feature_dim != HALFKP_FEATURE_DIM {
            return Err(format!(
                "Unexpected feature_dim {feature_dim} (expected {LEGACY_FEATURE_DIM} or {HALFKP_FEATURE_DIM})"
            ));
        }

        // Auto-detect output bucket count from cp_head_weight shape.
        // Old models: w3_i16.len() == HIDDEN2 → n_output_buckets = 1.
        // New models: w3_i16.len() == N_BUCKETS * HIDDEN2 → n_output_buckets = N_BUCKETS.
        let n_output_buckets = w3_i16.len() / HIDDEN2;
        if n_output_buckets == 0 || w3_i16.len() % HIDDEN2 != 0 {
            return Err(format!(
                "Unexpected cp_head_weight size {} (not a multiple of HIDDEN2={})",
                w3_i16.len(), HIDDEN2
            ));
        }

        // Transpose w1 into two forms:
        //   w1_t     (f32, dequantized) — used by scratch evaluation path
        //   w1_t_i16 (i16, raw)         — used by SIMD incremental path
        // Input layout from NPZ: [HIDDEN1 × feature_dim] (row-major)
        let w1_row_f32 = dq(&w1_raw);
        let mut w1_t = vec![0.0f32; feature_dim * HIDDEN1];
        let mut w1_t_i16 = vec![0i16; feature_dim * HIDDEN1];
        for j in 0..HIDDEN1 {
            for i in 0..feature_dim {
                w1_t[i * HIDDEN1 + j] = w1_row_f32[j * feature_dim + i];
                w1_t_i16[i * HIDDEN1 + j] = w1_raw[j * feature_dim + i];
            }
        }

        Ok(Self {
            feature_dim,
            dual_perspective: dual,
            scale,
            w1_t,
            b1: dq(&b1_raw),
            w1_t_i16,
            b1_i16: b1_raw,
            w2: dq(&w2_i16),
            b2: dq(&b2_i16),
            w3: dq(&w3_i16),
            b3: dq(&b3_i16),
            w_wdl: dq(&w_wdl_i16),
            b_wdl: dq(&b_wdl_i16),
            n_output_buckets,
        })
    }

    /// Evaluate a position from scratch.
    ///
    /// Returns `(score, confidence)` where:
    /// - For dual model: `score` is centipawns from **white's** perspective.
    /// - For single model: `score` is centipawns from **side-to-move's** perspective.
    pub fn evaluate_with_confidence(&self, board: &ChessBoard) -> (i32, f32) {
        let bucket = piece_bucket(board, self.n_output_buckets);
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
            for v in h_w.iter_mut() { *v = screlu_f32(*v); }
            for v in h_b.iter_mut() { *v = screlu_f32(*v); }
            self.forward_l2_heads_dual(&h_w, &h_b, bucket)
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
            for v in h1.iter_mut() { *v = screlu_f32(*v); }
            self.forward_l2_heads_single(&h1, bucket)
        }
    }

    /// Evaluate from pre-activation i16 accumulators (Phase 4 incremental path).
    /// SCReLU-clamps and dequantizes each element, then runs the f32 L2 heads.
    /// Only valid for dual-perspective models.
    pub fn evaluate_from_accumulators(
        &self,
        acc_white: &[i16; HIDDEN1],
        acc_black: &[i16; HIDDEN1],
        bucket: usize,
    ) -> (i32, f32) {
        debug_assert!(self.dual_perspective);
        let s = self.scale;
        let mut h_w = [0.0f32; HIDDEN1];
        let mut h_b = [0.0f32; HIDDEN1];
        for (h, &a) in h_w.iter_mut().zip(acc_white.iter()) {
            *h = screlu_i16(a, s);
        }
        for (h, &a) in h_b.iter_mut().zip(acc_black.iter()) {
            *h = screlu_i16(a, s);
        }
        self.forward_l2_heads_dual(&h_w, &h_b, bucket)
    }

    /// Layer 2 + heads for dual model: input is [h_w(1024) | h_b(1024)].
    fn forward_l2_heads_dual(&self, h_w: &[f32; HIDDEN1], h_b: &[f32; HIDDEN1], bucket: usize) -> (i32, f32) {
        let mut h2 = [0.0f32; HIDDEN2];
        for j in 0..HIDDEN2 {
            let row = &self.w2[j * HIDDEN1_DUAL..(j + 1) * HIDDEN1_DUAL];
            let mut acc = self.b2[j];
            for i in 0..HIDDEN1 {
                acc += row[i] * h_w[i];
                acc += row[HIDDEN1 + i] * h_b[i];
            }
            h2[j] = screlu_f32(acc);
        }
        self.forward_heads(&h2, bucket)
    }

    /// Layer 2 + heads for single-perspective model: input is h1(1024).
    fn forward_l2_heads_single(&self, h1: &[f32; HIDDEN1], bucket: usize) -> (i32, f32) {
        let mut h2 = [0.0f32; HIDDEN2];
        for j in 0..HIDDEN2 {
            let row = &self.w2[j * HIDDEN1..(j + 1) * HIDDEN1];
            let mut acc = self.b2[j];
            for i in 0..HIDDEN1 {
                acc += row[i] * h1[i];
            }
            h2[j] = screlu_f32(acc);
        }
        self.forward_heads(&h2, bucket)
    }

    /// CP head + WDL head from h2, selected by output bucket.
    #[inline]
    fn forward_heads(&self, h2: &[f32; HIDDEN2], bucket: usize) -> (i32, f32) {
        let b = bucket.min(self.n_output_buckets - 1);

        // CP head: row b of w3 (shape [n_output_buckets × HIDDEN2])
        let w_cp = &self.w3[b * HIDDEN2..(b + 1) * HIDDEN2];
        let mut cp = self.b3[b];
        for i in 0..HIDDEN2 {
            cp += w_cp[i] * h2[i];
        }

        // WDL head: rows b*3 .. b*3+2 of w_wdl (shape [n_output_buckets × 3 × HIDDEN2])
        let mut logits = [0.0f32; 3];
        for k in 0..3 {
            let row_start = (b * 3 + k) * HIDDEN2;
            let row = &self.w_wdl[row_start..row_start + HIDDEN2];
            let mut acc = self.b_wdl[b * 3 + k];
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

/// Map total piece count to output bucket index.
/// Formula: clamp((total_pieces - 2) * n_buckets / 30, 0, n_buckets - 1).
/// 2 pieces → 0, 32 pieces → n_buckets - 1.
pub fn piece_bucket(board: &ChessBoard, n_buckets: usize) -> usize {
    let total = board.get_all_pieces().count_ones() as usize;
    ((total.saturating_sub(2)) * n_buckets / 30).min(n_buckets - 1)
}

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
/// Both use the same feature formula: slot*64*16 + mapped_sq*16 + king_bucket.
///
/// Horizontal mirroring: when the king is on files 4-7 (e-h), piece square files
/// are flipped (`sq ^ 7` flips bits 0-2, preserving rank bits 3-5).  This ensures
/// that king-on-a1 and king-on-h1 see identical feature distributions.
pub(crate) fn encode_dual_halfkp(
    board: &ChessBoard,
) -> (([usize; 32], usize), ([usize; 32], usize)) {
    let white_bb = board.get_white();
    let black_bb = board.get_black();

    // White perspective: white king bucket (no rank flip)
    let wk_sq = (white_bb & board.get_kings()).0.trailing_zeros() as usize;
    let wk_sq = wk_sq.min(63);
    let wk_bucket = KING_BUCKET[wk_sq];
    let mirror_w = (wk_sq % 8) >= 4;

    // Black perspective: black king rank-flipped
    let bk_sq_raw = (black_bb & board.get_kings()).0.trailing_zeros() as usize;
    let bk_sq_raw = bk_sq_raw.min(63);
    let bk_flipped = bk_sq_raw ^ 56;
    let bk_bucket = KING_BUCKET[bk_flipped];
    let mirror_b = (bk_flipped % 8) >= 4;

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
                    let mapped = if mirror_w { sq ^ 7 } else { sq };
                    w_indices[wc] = $slot * 64 * 16 + mapped * 16 + wk_bucket;
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
                    let rank_flipped = sq ^ 56;
                    let mapped = if mirror_b { rank_flipped ^ 7 } else { rank_flipped };
                    b_indices[bc] = $slot * 64 * 16 + mapped * 16 + bk_bucket;
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

    fn sorted(indices: &[usize], count: usize) -> Vec<usize> {
        let mut v = indices[..count].to_vec();
        v.sort_unstable();
        v
    }

    #[test]
    fn test_dual_white_king_feature_index() {
        // White king on e1 (sq=4). file=4 ≥ 4, so mirror_w=true, mapped=4^7=3.
        // King bucket for mapped sq=3: file=3, rank=0 → bucket = 0*4+3 = 3.
        // White king is slot 5 (ours). Expected index = 5*64*16 + 3*16 + 3 = 5171.
        let board = ChessBoard::new();
        assert!(board.is_white_active());
        let ((w_idx, wc), _) = encode_dual_halfkp(&board);
        let king_sq: usize = 4;  // e1
        let mirror_w = (king_sq % 8) >= 4;
        let mapped = if mirror_w { king_sq ^ 7 } else { king_sq }; // 3
        let file_bucket = if mapped % 8 <= 3 { mapped % 8 } else { 7 - mapped % 8 };
        let rank_half   = if mapped / 8 <= 3 { 0 } else { 1 };
        let bucket      = rank_half * 4 + file_bucket;
        let expected    = 5 * 64 * 16 + mapped * 16 + bucket;
        assert!(
            w_idx[..wc].contains(&expected),
            "White king index {expected} not found in dual white-pov features"
        );
    }

    #[test]
    fn test_dual_black_king_feature_index() {
        // Black king on e8 (sq=60). After rank-flip: 60^56=4. file=4 ≥ 4, so mirror_b=true.
        // mapped = 4^7 = 3. Bucket for sq=3: file=3, rank=0 → bucket=3.
        // Black king is slot 5 (ours in black-pov). Expected = 5*64*16 + 3*16 + 3 = 5171.
        let board = ChessBoard::new();
        let (_, (b_idx, bc)) = encode_dual_halfkp(&board);
        let bk_sq: usize = 60; // e8
        let flipped      = bk_sq ^ 56; // 4
        let mirror_b     = (flipped % 8) >= 4;
        let mapped       = if mirror_b { flipped ^ 7 } else { flipped }; // 3
        let file_bucket  = if mapped % 8 <= 3 { mapped % 8 } else { 7 - mapped % 8 };
        let rank_half    = if mapped / 8 <= 3 { 0 } else { 1 };
        let bucket       = rank_half * 4 + file_bucket;
        let expected     = 5 * 64 * 16 + mapped * 16 + bucket;
        assert!(
            b_idx[..bc].contains(&expected),
            "Black king index {expected} not found in dual black-pov features"
        );
    }

    #[test]
    fn test_dual_no_duplicate_indices() {
        let board = ChessBoard::new();
        let ((w_idx, wc), (b_idx, bc)) = encode_dual_halfkp(&board);

        let mut w = w_idx[..wc].to_vec(); w.sort_unstable();
        let mut b = b_idx[..bc].to_vec(); b.sort_unstable();

        w.windows(2).for_each(|p| assert_ne!(p[0], p[1], "Duplicate in white-pov"));
        b.windows(2).for_each(|p| assert_ne!(p[0], p[1], "Duplicate in black-pov"));
    }

    // ── Phase 4: i16 SIMD accumulator tests ──────────────────────────────

    #[test]
    fn test_add_col_correctness() {
        // Smoke test: add_col on zero accumulator equals the column.
        let mut acc = [0i16; HIDDEN1];
        let col: Vec<i16> = (0..HIDDEN1 as i16).collect();
        add_col(&mut acc, &col);
        assert_eq!(acc[0], 0);
        assert_eq!(acc[1], 1);
        assert_eq!(acc[HIDDEN1 - 1], (HIDDEN1 - 1) as i16);
    }

    #[test]
    fn test_sub_col_correctness() {
        // Smoke test: sub_col on an accumulator equal to col yields zero.
        let col: Vec<i16> = (1..=HIDDEN1 as i16).collect();
        let mut acc: [i16; HIDDEN1] = col.as_slice().try_into().unwrap();
        sub_col(&mut acc, &col);
        for v in acc.iter() {
            assert_eq!(*v, 0, "acc should be zero after subtracting itself");
        }
    }

    #[test]
    fn test_add_col_saturates_at_max() {
        let mut acc = [i16::MAX; HIDDEN1];
        let col = [1i16; HIDDEN1];
        add_col(&mut acc, &col);
        assert_eq!(acc[0], i16::MAX, "saturating add must not overflow i16::MAX");
    }

    #[test]
    fn test_sub_col_saturates_at_min() {
        let mut acc = [i16::MIN; HIDDEN1];
        let col = [1i16; HIDDEN1];
        sub_col(&mut acc, &col);
        assert_eq!(acc[0], i16::MIN, "saturating sub must not underflow i16::MIN");
    }

    #[test]
    fn test_acc_add_feature_noop_without_evaluator() {
        // Without a loaded evaluator the function must be a no-op.
        let mut acc = [42i16; HIDDEN1];
        acc_add_feature(&mut acc, 0);
        assert!(acc.iter().all(|&v| v == 42), "acc_add_feature must be no-op with no evaluator");
    }

    #[test]
    fn test_acc_sub_feature_noop_without_evaluator() {
        let mut acc = [7i16; HIDDEN1];
        acc_sub_feature(&mut acc, 0);
        assert!(acc.iter().all(|&v| v == 7), "acc_sub_feature must be no-op with no evaluator");
    }

    /// Full equivalence test: i16 incremental path vs f32 scratch path.
    ///
    /// Requires the weights file at `src/eval.npz` (relative to crate root).
    /// Run with: `cargo test -p chess_evaluation -- --include-ignored i16_accum_equivalence`
    #[test]
    #[ignore = "requires src/eval.npz — run with --include-ignored"]
    fn test_i16_accum_equivalence() {
        let bytes = match std::fs::read("src/eval.npz") {
            Ok(b) => b,
            Err(_) => { println!("skipping: src/eval.npz not found"); return; }
        };
        let eval = NeuralEvaluator::from_npz_bytes(&bytes).unwrap();
        if !eval.dual_perspective {
            println!("skipping: model is single-perspective (backbone_3 is 32×512, need 32×1024)");
            return;
        }

        let positions = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
            "r3r1k1/pp3pbp/1qp3p1/2B5/2BP2b1/Q1n2N2/P4PPP/3RR1K1 w - - 0 1",
            "8/8/4k3/8/8/4K3/8/8 w - - 0 1",
        ];

        for fen in positions {
            let mut board = ChessBoard::new();
            board.set_from_fen(fen);

            // f32 scratch path
            let (scratch_score, _) = eval.evaluate_with_confidence(&board);

            // i16 incremental path
            let ((w_idx, wc), (b_idx, bc)) = encode_dual_halfkp(&board);
            let mut acc_w = [0i16; HIDDEN1];
            let mut acc_b = [0i16; HIDDEN1];
            acc_w.copy_from_slice(&eval.b1_i16);
            acc_b.copy_from_slice(&eval.b1_i16);
            for &i in &w_idx[..wc] {
                let col = &eval.w1_t_i16[i * HIDDEN1..(i + 1) * HIDDEN1];
                add_col(&mut acc_w, col);
            }
            for &i in &b_idx[..bc] {
                let col = &eval.w1_t_i16[i * HIDDEN1..(i + 1) * HIDDEN1];
                add_col(&mut acc_b, col);
            }
            let bucket = piece_bucket(&board, eval.n_output_buckets);
            let (i16_score, _) = eval.evaluate_from_accumulators(&acc_w, &acc_b, bucket);

            let diff = (scratch_score - i16_score).unsigned_abs();
            assert!(
                diff <= 2,
                "FEN {fen}: scratch={scratch_score} i16={i16_score} diff={diff}cp (expected ≤2cp)"
            );
        }
    }

    // ── SCReLU tests ──────────────────────────────────────────────────────

    #[test]
    fn test_screlu_zero() {
        assert_eq!(screlu_i16(0, 256.0), 0.0);
        assert_eq!(screlu_f32(0.0), 0.0);
    }

    #[test]
    fn test_screlu_half() {
        let v = screlu_i16(128, 256.0);
        let expected = 0.5f32 * 0.5;
        assert!((v - expected).abs() < 1e-4, "screlu_i16(128, 256) = {v}, expected ~{expected}");
    }

    #[test]
    fn test_screlu_clamped() {
        // Values above scale clamp to 1.0
        assert_eq!(screlu_i16(512, 256.0), 1.0);
        assert_eq!(screlu_f32(2.0), 1.0);
    }

    #[test]
    fn test_screlu_negative() {
        assert_eq!(screlu_i16(-1, 256.0), 0.0);
        assert_eq!(screlu_f32(-0.5), 0.0);
    }

    // ── Output bucket tests ───────────────────────────────────────────────

    #[test]
    fn test_output_bucket_range() {
        // For any piece count 2..=32, bucket must be in [0, 7].
        let board = ChessBoard::new(); // 32 pieces
        let total = board.get_all_pieces().count_ones() as usize;
        assert_eq!(total, 32);
        for pc in 2..=32usize {
            let b = ((pc.saturating_sub(2)) * 8 / 30).min(7);
            assert!(b < 8, "piece_count={pc} → bucket={b} out of range");
        }
    }

    #[test]
    fn test_output_bucket_extremes() {
        // 2 pieces → bucket 0; 32 pieces → bucket 7
        let b_min = ((2usize.saturating_sub(2)) * 8 / 30).min(7);
        let b_max = ((32usize.saturating_sub(2)) * 8 / 30).min(7);
        assert_eq!(b_min, 0, "2 pieces must map to bucket 0");
        assert_eq!(b_max, 7, "32 pieces must map to bucket 7");
    }

    // ── Horizontal mirror symmetry test ──────────────────────────────────

    #[test]
    fn test_horizontal_mirror_symmetry() {
        // White king on a1 + pawn a2 + black king h8  (king on file 0 — no mirror)
        // vs
        // White king on h1 + pawn h2 + black king a8  (king on file 7 — mirror applied)
        // Both positions are file-mirror images; the sorted white-pov feature sets
        // must be identical after the horizontal mirroring fix.
        let mut board_a = ChessBoard::new();
        board_a.set_from_fen("7k/8/8/8/8/8/P7/K7 w - - 0 1");
        let mut board_h = ChessBoard::new();
        board_h.set_from_fen("k7/8/8/8/8/8/7P/7K w - - 0 1");

        let ((wa, wca), _) = encode_dual_halfkp(&board_a);
        let ((wh, wch), _) = encode_dual_halfkp(&board_h);

        assert_eq!(wca, wch, "Feature count must be equal for mirror positions");
        assert_eq!(
            sorted(&wa, wca),
            sorted(&wh, wch),
            "Mirrored positions (Ka1+Pa2 vs Kh1+Ph2) must produce identical white-pov feature sets"
        );
    }

    #[test]
    #[ignore = "requires src/eval.npz — run with --include-ignored"]
    fn test_incremental_screlu_matches_scratch() {
        let bytes = match std::fs::read("src/eval.npz") {
            Ok(b) => b,
            Err(_) => { println!("skipping: src/eval.npz not found"); return; }
        };
        let eval = NeuralEvaluator::from_npz_bytes(&bytes).unwrap();
        if !eval.dual_perspective {
            println!("skipping: model is single-perspective");
            return;
        }

        // Test with king on both board halves to exercise mirroring paths
        let positions = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            "r3r1k1/pp3pbp/1qp3p1/2B5/2BP2b1/Q1n2N2/P4PPP/3RR1K1 w - - 0 1",
            "7k/8/8/8/8/8/P7/K7 w - - 0 1",  // king on file 0
            "k7/8/8/8/8/8/7P/7K w - - 0 1",  // king on file 7 (mirrored)
            "8/8/4k3/8/8/4K3/8/8 w - - 0 1",
        ];

        for fen in positions {
            let mut board = ChessBoard::new();
            board.set_from_fen(fen);

            let (scratch_score, _) = eval.evaluate_with_confidence(&board);

            let ((w_idx, wc), (b_idx, bc)) = encode_dual_halfkp(&board);
            let mut acc_w = [0i16; HIDDEN1];
            let mut acc_b = [0i16; HIDDEN1];
            acc_w.copy_from_slice(&eval.b1_i16);
            acc_b.copy_from_slice(&eval.b1_i16);
            for &i in &w_idx[..wc] {
                let col = &eval.w1_t_i16[i * HIDDEN1..(i + 1) * HIDDEN1];
                add_col(&mut acc_w, col);
            }
            for &i in &b_idx[..bc] {
                let col = &eval.w1_t_i16[i * HIDDEN1..(i + 1) * HIDDEN1];
                add_col(&mut acc_b, col);
            }
            let bucket = piece_bucket(&board, eval.n_output_buckets);
            let (i16_score, _) = eval.evaluate_from_accumulators(&acc_w, &acc_b, bucket);

            let diff = (scratch_score - i16_score).unsigned_abs();
            assert!(
                diff <= 2,
                "FEN {fen}: scratch={scratch_score} i16={i16_score} diff={diff}cp (expected ≤2cp)"
            );
        }
    }
}

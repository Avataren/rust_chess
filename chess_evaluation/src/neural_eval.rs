//! Neural network evaluation for chess positions.
//!
//! Implements a small NNUE-like MLP trained with the Python pipeline in
//! `nn_training/`. Two model variants are supported:
//!
//! **Single-perspective (Phase 1):** `12288 → 512 → 32 → 1 (×N output buckets)`
//!   Weights: `backbone_3_weight` shape (32, 512)
//!
//! **Dual-perspective (Phase 2+3):** `[12288|12288] → 1024+1024 → 256 → 1 (×N output buckets)`
//!   Shared EmbeddingBag; two accumulators concat'd before fc2.
//!   Weights: `backbone_3_weight` shape (256, 2048)
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
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::OnceLock;

use chess_board::ChessBoard;
use chess_foundation::bitboard::Bitboard;

// ── Architecture constants ────────────────────────────────────────────────
//
// These must match the trained model's hidden dimensions.
// After changing, rebuild all binaries: cargo build --release
// Python config: hidden_dim must equal HIDDEN1, hidden2_dim must equal HIDDEN2.
//
//   Current: HalfKAv2 768×256 (teacher model, to be distilled)
//   Previous: 1024×64

pub(crate) const HIDDEN1: usize = 768;
const HIDDEN2: usize = 256;
const HIDDEN1_DUAL: usize = HIDDEN1 * 2; // 1536

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

pub(crate) const KING_BUCKETS: usize = 32;
const NUM_PIECE_SLOTS: usize = 12;
const HALFKP_FEATURE_DIM: usize = NUM_PIECE_SLOTS * 64 * KING_BUCKETS; // 24,576
const LEGACY_FEATURE_DIM: usize = 768;

// HalfKAv2: 11 piece types (all except own king) × 64 squares × 64 exact king squares.
// Feature index: slot * 64 * 64 + piece_sq * 64 + king_sq
// Slots 0-4: own P/N/B/R/Q; slots 5-10: their P/N/B/R/Q/K (own king excluded).
const HALFKAV2_NUM_PIECE_SLOTS: usize = 11;
pub(crate) const HALFKAV2_FEATURE_DIM: usize = HALFKAV2_NUM_PIECE_SLOTS * 64 * 64; // 45,056

// ── Global state ──────────────────────────────────────────────────────────

static NEURAL_ENABLED: AtomicBool = AtomicBool::new(false);
static EVALUATOR: OnceLock<NeuralEvaluator> = OnceLock::new();

/// Minimum WDL confidence to trust the NN score.
/// Default 0.0 = always trust NN (no fallback).
static CONFIDENCE_THRESHOLD: AtomicU32 = AtomicU32::new(0);

/// Number of positions where confidence fell below threshold and HCE was used.
/// Reset by `reset_hce_fallback_count()` before each search.
static HCE_FALLBACK_COUNT: AtomicU64 = AtomicU64::new(0);

pub fn reset_hce_fallback_count() {
    HCE_FALLBACK_COUNT.store(0, Ordering::Relaxed);
}

pub fn get_hce_fallback_count() -> u64 {
    HCE_FALLBACK_COUNT.load(Ordering::Relaxed)
}

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

/// Returns true if the loaded model uses HalfKAv2 features (input_dim = 45,056).
/// Used by search_context to select the correct incremental accumulator formula.
pub fn is_halfkav2() -> bool {
    EVALUATOR.get()
        .map(|e| e.feature_dim == HALFKAV2_FEATURE_DIM)
        .unwrap_or(false)
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

fn fill_accumulators(
    evaluator: &NeuralEvaluator,
    board: &ChessBoard,
    acc_white: &mut [i16; HIDDEN1],
    acc_black: &mut [i16; HIDDEN1],
) {
    let ((w_idx, wc), (b_idx, bc)) = if evaluator.feature_dim == HALFKAV2_FEATURE_DIM {
        encode_dual_halfkav2(board)
    } else {
        encode_dual_halfkp(board)
    };
    acc_white.copy_from_slice(&evaluator.b1_i16);
    for &i in &w_idx[..wc] {
        add_col(acc_white, &evaluator.w1_t_i16[i * HIDDEN1..(i + 1) * HIDDEN1]);
    }
    acc_black.copy_from_slice(&evaluator.b1_i16);
    for &i in &b_idx[..bc] {
        add_col(acc_black, &evaluator.w1_t_i16[i * HIDDEN1..(i + 1) * HIDDEN1]);
    }
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
    fill_accumulators(evaluator, board, acc_white, acc_black);
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

/// Like `eval_direct` but returns `None` if the evaluator has not been
/// initialized yet.  Used by `evaluate_board` so tests compiled with
/// `nn-incremental` can fall back to classical eval without loading a model.
/// Does NOT apply the confidence threshold — that would create an inconsistent
/// evaluation function inside the search tree (some nodes NN, some HCE).
#[cfg(any(feature = "nn-full-forward", feature = "nn-incremental"))]
#[inline]
pub fn try_eval_direct(board: &ChessBoard) -> Option<i32> {
    let e = EVALUATOR.get()?;
    let (score, _) = e.evaluate_with_confidence(board);
    Some(if e.dual_perspective {
        score
    } else {
        if board.is_white_active() { score } else { -score }
    })
}

/// Returns the WDL confidence (max softmax output) for `board` using a full
/// forward pass.  `None` if weights are not loaded.  Works regardless of
/// Cargo feature — intended for UCI diagnostics, not the search hot-path.
pub fn eval_position_confidence(board: &ChessBoard) -> Option<f32> {
    let e = EVALUATOR.get()?;
    let (_, confidence) = e.evaluate_with_confidence(board);
    Some(confidence)
}

/// Direct accumulator-based evaluation — no runtime checks, no confidence gate.
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

/// Like `eval_accum_direct` but returns `None` if the evaluator is not loaded.
/// Does NOT apply the confidence threshold — mixing NN and HCE within the same
/// search tree creates inconsistent evaluations and tactical blunders.
/// Confidence is tracked separately for display via `eval_position_confidence`.
#[cfg(feature = "nn-incremental")]
#[inline]
pub fn try_eval_accum_direct(
    board: &ChessBoard,
    acc_white: &[i16; HIDDEN1],
    acc_black: &[i16; HIDDEN1],
) -> Option<i32> {
    let e = EVALUATOR.get()?;
    let bucket = piece_bucket(board, e.n_output_buckets);
    Some(e.evaluate_from_accumulators(acc_white, acc_black, bucket).0)
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
    fill_accumulators(evaluator, board, acc_white, acc_black);
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

/// Apply accumulator deltas in the order: `subs_pre`, `adds`, `subs_post`.
///
/// The three-phase ordering preserves the original operation sequence:
///   sub moving-piece-from-source → add moving-piece-to-dest → sub captured-piece
/// which matters for saturating i16 arithmetic (reordering can change intermediate
/// saturation and thus produce different results).
///
/// Does a single `EVALUATOR.get()` instead of one per feature, reducing
/// atomic loads from 4–8 to 2 per search node in `acc_push`.
#[inline]
pub fn acc_apply_deltas(
    acc: &mut [i16; HIDDEN1],
    subs_pre: &[usize],
    adds: &[usize],
    subs_post: &[usize],
) {
    if let Some(e) = EVALUATOR.get() {
        for &idx in subs_pre {
            sub_col(acc, &e.w1_t_i16[idx * HIDDEN1..(idx + 1) * HIDDEN1]);
        }
        for &idx in adds {
            add_col(acc, &e.w1_t_i16[idx * HIDDEN1..(idx + 1) * HIDDEN1]);
        }
        for &idx in subs_post {
            sub_col(acc, &e.w1_t_i16[idx * HIDDEN1..(idx + 1) * HIDDEN1]);
        }
    }
}

// ── SIMD column accumulator operations ───────────────────────────────────
//
// Three compile-time paths selected by cfg:
//   x86_64 + avx2    → _mm256_adds_epi16  (16 i16/reg, 4× unrolled, 16 iters/col)
//   wasm32 + simd128 → i16x8_add_sat       (8 i16/reg, 64 instr/col)
//   fallback         → scalar saturating_add (1 i16/iter, 512 instr/col)
//
// `add_col` / `sub_col` are the public dispatch functions used everywhere.

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
unsafe fn add_col_avx2(acc: &mut [i16; HIDDEN1], col: &[i16]) {
    use std::arch::x86_64::*;
    // 4× unrolled: 16 iterations of 4 registers = 64 total, ILP-friendly
    for i in (0..HIDDEN1).step_by(64) {
        let a0 = _mm256_loadu_si256(acc[i     ..].as_ptr() as *const __m256i);
        let a1 = _mm256_loadu_si256(acc[i + 16..].as_ptr() as *const __m256i);
        let a2 = _mm256_loadu_si256(acc[i + 32..].as_ptr() as *const __m256i);
        let a3 = _mm256_loadu_si256(acc[i + 48..].as_ptr() as *const __m256i);
        let b0 = _mm256_loadu_si256(col[i     ..].as_ptr() as *const __m256i);
        let b1 = _mm256_loadu_si256(col[i + 16..].as_ptr() as *const __m256i);
        let b2 = _mm256_loadu_si256(col[i + 32..].as_ptr() as *const __m256i);
        let b3 = _mm256_loadu_si256(col[i + 48..].as_ptr() as *const __m256i);
        _mm256_storeu_si256(acc[i     ..].as_mut_ptr() as *mut __m256i, _mm256_adds_epi16(a0, b0));
        _mm256_storeu_si256(acc[i + 16..].as_mut_ptr() as *mut __m256i, _mm256_adds_epi16(a1, b1));
        _mm256_storeu_si256(acc[i + 32..].as_mut_ptr() as *mut __m256i, _mm256_adds_epi16(a2, b2));
        _mm256_storeu_si256(acc[i + 48..].as_mut_ptr() as *mut __m256i, _mm256_adds_epi16(a3, b3));
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
unsafe fn sub_col_avx2(acc: &mut [i16; HIDDEN1], col: &[i16]) {
    use std::arch::x86_64::*;
    // 4× unrolled: 16 iterations of 4 registers = 64 total, ILP-friendly
    for i in (0..HIDDEN1).step_by(64) {
        let a0 = _mm256_loadu_si256(acc[i     ..].as_ptr() as *const __m256i);
        let a1 = _mm256_loadu_si256(acc[i + 16..].as_ptr() as *const __m256i);
        let a2 = _mm256_loadu_si256(acc[i + 32..].as_ptr() as *const __m256i);
        let a3 = _mm256_loadu_si256(acc[i + 48..].as_ptr() as *const __m256i);
        let b0 = _mm256_loadu_si256(col[i     ..].as_ptr() as *const __m256i);
        let b1 = _mm256_loadu_si256(col[i + 16..].as_ptr() as *const __m256i);
        let b2 = _mm256_loadu_si256(col[i + 32..].as_ptr() as *const __m256i);
        let b3 = _mm256_loadu_si256(col[i + 48..].as_ptr() as *const __m256i);
        _mm256_storeu_si256(acc[i     ..].as_mut_ptr() as *mut __m256i, _mm256_subs_epi16(a0, b0));
        _mm256_storeu_si256(acc[i + 16..].as_mut_ptr() as *mut __m256i, _mm256_subs_epi16(a1, b1));
        _mm256_storeu_si256(acc[i + 32..].as_mut_ptr() as *mut __m256i, _mm256_subs_epi16(a2, b2));
        _mm256_storeu_si256(acc[i + 48..].as_mut_ptr() as *mut __m256i, _mm256_subs_epi16(a3, b3));
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

// ── SCReLU dequantization: i16 accumulator → f32 ─────────────────────────
//
// Converts raw i16 accumulator values to f32 with SCReLU applied:
//   clamp(x / scale, 0, 1)²
//
// AVX2 path: processes 8 i16 per iteration via cvtepi16→cvtepi32→cvtepi32_ps.
// Scalar fallback used on non-AVX2 targets.

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
unsafe fn screlu_deq_avx2(acc: &[i16], scale: f32, out: &mut [f32]) {
    use std::arch::x86_64::*;
    debug_assert_eq!(acc.len(), out.len());
    let zero   = _mm256_setzero_ps();
    let vscale = _mm256_set1_ps(scale);
    let inv_sc = _mm256_set1_ps(1.0 / scale);
    // Process 16 i16 per iteration (two 128-bit loads → two 256-bit f32 blocks)
    // HIDDEN1=512 → 32 iterations, scalar tail never runs.
    let chunks = acc.len() / 16;
    for k in 0..chunks {
        let vi16_lo = _mm_loadu_si128(acc.as_ptr().add(k * 16    ) as *const __m128i);
        let vi16_hi = _mm_loadu_si128(acc.as_ptr().add(k * 16 + 8) as *const __m128i);
        let vi32_lo = _mm256_cvtepi16_epi32(vi16_lo);
        let vi32_hi = _mm256_cvtepi16_epi32(vi16_hi);
        let vf_lo   = _mm256_cvtepi32_ps(vi32_lo);
        let vf_hi   = _mm256_cvtepi32_ps(vi32_hi);
        let clp_lo  = _mm256_min_ps(_mm256_max_ps(vf_lo, zero), vscale);
        let clp_hi  = _mm256_min_ps(_mm256_max_ps(vf_hi, zero), vscale);
        let norm_lo = _mm256_mul_ps(clp_lo, inv_sc);
        let norm_hi = _mm256_mul_ps(clp_hi, inv_sc);
        _mm256_storeu_ps(out.as_mut_ptr().add(k * 16    ), _mm256_mul_ps(norm_lo, norm_lo));
        _mm256_storeu_ps(out.as_mut_ptr().add(k * 16 + 8), _mm256_mul_ps(norm_hi, norm_hi));
    }
    for k in chunks * 16..acc.len() {
        out[k] = screlu_i16(acc[k], scale);
    }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
fn screlu_deq_avx2(acc: &[i16], scale: f32, out: &mut [f32]) {
    for (o, &a) in out.iter_mut().zip(acc.iter()) {
        *o = screlu_i16(a, scale);
    }
}

#[inline(always)]
fn screlu_deq(acc: &[i16], scale: f32, out: &mut [f32]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe { return screlu_deq_avx2(acc, scale, out); }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    for (o, &a) in out.iter_mut().zip(acc.iter()) {
        *o = screlu_i16(a, scale);
    }
}

// ── Quantized incremental L2 path ─────────────────────────────────────────
//
// The dual-perspective incremental eval spends ~95% of its time in the
// Layer-2 GEMV, streaming the 1.5 MB f32 `w2` matrix from L3 once per call —
// it is memory-bound.  Storing `w2` as raw i16 (its original quantized form)
// halves the bytes and doubles the MACs/instruction via `_mm256_madd_epi16`.
//
// SCReLU activations are effectively already quantized: the i16 accumulator
// (scale 256) clamps to [0, scale] before squaring, so the activation takes
// only ~scale+1 distinct values.  Re-quantizing the squared result to
// `ACT_Q` levels is therefore near-lossless.
//
// Overflow: each of the 8 final i32 lanes accumulates HIDDEN1_DUAL/8 = 192
// products of |w2| and q ≤ ACT_Q.  `L2_I16_MAX_W` is the largest |w2| that
// provably keeps every lane inside i32, carrying a 2× safety margin; the
// embedded weights peak near 8.6k, well under it.  Models that exceed it fall
// back to the f32 path at load time.
const ACT_Q: i32 = 512;
const ACT_Q_F: f32 = ACT_Q as f32;

const L2_I16_MAX_W: i32 =
    (i32::MAX as i64 / (2 * (HIDDEN1_DUAL as i64 / 8) * ACT_Q as i64)) as i32;

/// SCReLU + quantize an i16 accumulator to i16 activations in [0, ACT_Q].
///   out[k] = round( clamp(acc[k]/scale, 0, 1)² · ACT_Q )
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
unsafe fn screlu_quant_avx2(acc: &[i16], scale: f32, out: &mut [i16]) {
    use std::arch::x86_64::*;
    debug_assert_eq!(acc.len(), out.len());
    debug_assert_eq!(acc.len() % 16, 0);
    let zero = _mm256_setzero_ps();
    let vscale = _mm256_set1_ps(scale);
    // clamp to [0, scale], square, then scale by ACT_Q / scale²  →  [0, ACT_Q]
    let vfac = _mm256_set1_ps(ACT_Q_F / (scale * scale));
    let chunks = acc.len() / 16;
    for k in 0..chunks {
        let lo = _mm_loadu_si128(acc.as_ptr().add(k * 16) as *const __m128i);
        let hi = _mm_loadu_si128(acc.as_ptr().add(k * 16 + 8) as *const __m128i);
        let fl = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(lo));
        let fh = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(hi));
        let cl = _mm256_min_ps(_mm256_max_ps(fl, zero), vscale);
        let ch = _mm256_min_ps(_mm256_max_ps(fh, zero), vscale);
        let ql = _mm256_mul_ps(_mm256_mul_ps(cl, cl), vfac);
        let qh = _mm256_mul_ps(_mm256_mul_ps(ch, ch), vfac);
        // round-to-nearest (default MXCSR) → i32 → pack to i16
        let il = _mm256_cvtps_epi32(ql);
        let ih = _mm256_cvtps_epi32(qh);
        // packs_epi32 works per 128-bit lane; permute to restore linear order
        let packed = _mm256_permute4x64_epi64(_mm256_packs_epi32(il, ih), 0b11_01_10_00);
        _mm256_storeu_si256(out.as_mut_ptr().add(k * 16) as *mut __m256i, packed);
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
unsafe fn hsum256_epi32(v: std::arch::x86_64::__m256i) -> i32 {
    use std::arch::x86_64::*;
    let lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extracti128_si256(v, 1);
    let s = _mm_add_epi32(lo, hi);
    let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b01_00_11_10));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b00_00_00_01));
    _mm_cvtsi128_si32(s)
}

/// Dual-model Layer-2: `out[j] = b2[j] + inv_q · Σ_i w2r[j·D + i] · q[i]`,
/// where `w2r` is row-major [HIDDEN2 × D], `q` the concatenated quantized
/// activations [q_w | q_b] (D = HIDDEN1_DUAL), `inv_q = 1 / (scale · ACT_Q)`.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
unsafe fn l2_dual_i16_avx2(
    w2r: &[i16],
    q: &[i16; HIDDEN1_DUAL],
    b2: &[f32],
    inv_q: f32,
    out: &mut [f32; HIDDEN2],
) {
    use std::arch::x86_64::*;
    const D: usize = HIDDEN1_DUAL;
    debug_assert_eq!(w2r.len(), HIDDEN2 * D);
    debug_assert_eq!(D % 64, 0);
    for j in 0..HIDDEN2 {
        let base = w2r.as_ptr().add(j * D);
        let mut s0 = _mm256_setzero_si256();
        let mut s1 = _mm256_setzero_si256();
        let mut s2 = _mm256_setzero_si256();
        let mut s3 = _mm256_setzero_si256();
        let mut i = 0;
        while i < D {
            let q0 = _mm256_loadu_si256(q.as_ptr().add(i) as *const __m256i);
            let q1 = _mm256_loadu_si256(q.as_ptr().add(i + 16) as *const __m256i);
            let q2 = _mm256_loadu_si256(q.as_ptr().add(i + 32) as *const __m256i);
            let q3 = _mm256_loadu_si256(q.as_ptr().add(i + 48) as *const __m256i);
            s0 = _mm256_add_epi32(s0, _mm256_madd_epi16(_mm256_loadu_si256(base.add(i) as *const __m256i), q0));
            s1 = _mm256_add_epi32(s1, _mm256_madd_epi16(_mm256_loadu_si256(base.add(i + 16) as *const __m256i), q1));
            s2 = _mm256_add_epi32(s2, _mm256_madd_epi16(_mm256_loadu_si256(base.add(i + 32) as *const __m256i), q2));
            s3 = _mm256_add_epi32(s3, _mm256_madd_epi16(_mm256_loadu_si256(base.add(i + 48) as *const __m256i), q3));
            i += 64;
        }
        let s = _mm256_add_epi32(_mm256_add_epi32(s0, s1), _mm256_add_epi32(s2, s3));
        *out.get_unchecked_mut(j) = b2[j] + inv_q * hsum256_epi32(s) as f32;
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
fn l2_dual_i16(w2r: &[i16], q: &[i16; HIDDEN1_DUAL], b2: &[f32], inv_q: f32, out: &mut [f32; HIDDEN2]) {
    unsafe { l2_dual_i16_avx2(w2r, q, b2, inv_q, out) }
}

/// Scalar fallback (non-AVX2 targets): identical arithmetic.
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
fn l2_dual_i16(w2r: &[i16], q: &[i16; HIDDEN1_DUAL], b2: &[f32], inv_q: f32, out: &mut [f32; HIDDEN2]) {
    const D: usize = HIDDEN1_DUAL;
    for j in 0..HIDDEN2 {
        let row = &w2r[j * D..(j + 1) * D];
        let mut s: i64 = 0;
        for i in 0..D {
            s += row[i] as i64 * q[i] as i64;
        }
        out[j] = b2[j] + inv_q * s as f32;
    }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
fn screlu_quant_scalar(acc: &[i16], scale: f32, out: &mut [i16]) {
    let fac = ACT_Q_F / (scale * scale);
    for (o, &a) in out.iter_mut().zip(acc.iter()) {
        let c = (a.max(0) as f32).min(scale);
        *o = (c * c * fac).round() as i16;
    }
}

#[inline(always)]
fn screlu_quant(acc: &[i16], scale: f32, out: &mut [i16]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe { return screlu_quant_avx2(acc, scale, out); }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    screlu_quant_scalar(acc, scale, out);
}

/// Quantize already-SCReLU'd f32 activations in [0, 1] to i16 in [0, ACT_Q].
/// Used by the scratch/full-forward path, which produces f32 activations.
#[inline]
fn quant_unit_activations(h: &[f32], out: &mut [i16]) {
    debug_assert_eq!(h.len(), out.len());
    for (o, &v) in out.iter_mut().zip(h.iter()) {
        *o = (v.clamp(0.0, 1.0) * ACT_Q_F).round() as i16;
    }
}

// ── Column-major GEMV for fc2 (input_dim × HIDDEN2 = 32) ─────────────────
//
// w is stored column-major: w[i * HIDDEN2 + j] = weight for output j, input i.
// acc is pre-initialised with bias; x is the input vector.
//
// AVX2 + FMA path: holds all 32 outputs in 4 YMM registers, streams
// through x once — each input element touches its 32-wide weight column
// without evicting the accumulator registers.
//
// GEMV: acc += w × x  (w is column-major: shape [x.len() × HIDDEN2])
// Works for any HIDDEN2 that is a multiple of 8.

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn gemv_col_avx2(w: &[f32], x: &[f32], acc: &mut [f32; HIDDEN2]) {
    use std::arch::x86_64::*;
    debug_assert_eq!(HIDDEN2 % 8, 0);
    debug_assert_eq!(w.len(), x.len() * HIDDEN2);
    // Process HIDDEN2 outputs in 8-wide AVX chunks.  Since HIDDEN2 is a
    // compile-time constant, LLVM unrolls the inner while-loop and keeps the
    // acc slices in YMM registers across outer iterations (w and acc can't alias).
    for i in 0..x.len() {
        let xi  = _mm256_set1_ps(*x.get_unchecked(i));
        let col = w.as_ptr().add(i * HIDDEN2);
        let mut k = 0;
        while k < HIDDEN2 {
            let a = _mm256_loadu_ps(acc.as_ptr().add(k));
            _mm256_storeu_ps(
                acc.as_mut_ptr().add(k),
                _mm256_fmadd_ps(_mm256_loadu_ps(col.add(k)), xi, a),
            );
            k += 8;
        }
    }
}

fn gemv_col_scalar(w: &[f32], x: &[f32], acc: &mut [f32; HIDDEN2]) {
    for i in 0..x.len() {
        let xi  = x[i];
        let col = &w[i * HIDDEN2..(i + 1) * HIDDEN2];
        for j in 0..HIDDEN2 { acc[j] += col[j] * xi; }
    }
}

#[inline(always)]
fn gemv_col(w: &[f32], x: &[f32], acc: &mut [f32; HIDDEN2]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe { return gemv_col_avx2(w, x, acc); }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    gemv_col_scalar(w, x, acc)
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

    /// Layer 2 weights as raw i16, **row-major** [HIDDEN2 × HIDDEN1_DUAL] — exactly
    /// the NPZ layout.  Dual-perspective models only.  Used by the quantized
    /// incremental eval path (`evaluate_from_accumulators`), which is memory-bound
    /// on this matrix: keeping it i16 halves the bytes streamed per eval vs `w2`
    /// (f32) and lets the dot product use `_mm256_madd_epi16` (16 MACs/instr).
    w2_dual_i16: Vec<i16>,
    /// True when the quantized i16 Layer-2 path is used by
    /// `evaluate_from_accumulators`: dual model, weights provably overflow-safe,
    /// and not disabled via `XAV_L2_F32=1`.  False → legacy f32 path.
    use_i16_l2: bool,

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
        if feature_dim != LEGACY_FEATURE_DIM
            && feature_dim != HALFKP_FEATURE_DIM
            && feature_dim != HALFKAV2_FEATURE_DIM
        {
            return Err(format!(
                "Unexpected feature_dim {feature_dim} \
                 (expected {LEGACY_FEATURE_DIM}, {HALFKP_FEATURE_DIM}, or {HALFKAV2_FEATURE_DIM})"
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

        // Transpose w2 from row-major (HIDDEN2 × input_dim) to column-major
        // (input_dim × HIDDEN2) so the GEMV can stream through the input once
        // and update all HIDDEN2 outputs simultaneously — better cache utilisation.
        let input_dim_l2 = if dual { HIDDEN1_DUAL } else { HIDDEN1 };
        let w2_row = dq(&w2_i16);
        let mut w2_col = vec![0.0f32; HIDDEN2 * input_dim_l2];
        for j in 0..HIDDEN2 {
            for i in 0..input_dim_l2 {
                w2_col[i * HIDDEN2 + j] = w2_row[j * input_dim_l2 + i];
            }
        }

        // Quantized i16 Layer-2 path: dual models only, and only when the
        // weights are small enough that the i32 dot product cannot overflow.
        // `XAV_L2_F32=1` forces the legacy f32 path for A/B comparison.
        let w2_max = w2_i16.iter().map(|v| v.unsigned_abs() as i32).max().unwrap_or(0);
        let use_i16_l2 = dual
            && w2_max <= L2_I16_MAX_W
            && std::env::var_os("XAV_L2_F32").is_none();

        Ok(Self {
            feature_dim,
            dual_perspective: dual,
            scale,
            w1_t,
            b1: dq(&b1_raw),
            w1_t_i16,
            b1_i16: b1_raw,
            w2_dual_i16: if use_i16_l2 { w2_i16.clone() } else { Vec::new() },
            use_i16_l2,
            w2: w2_col,
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
            let ((w_idx, wc), (b_idx, bc)) = if self.feature_dim == HALFKAV2_FEATURE_DIM {
                encode_dual_halfkav2(board)
            } else {
                encode_dual_halfkp(board)
            };

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

        if !self.use_i16_l2 {
            // Legacy f32 Layer-2 path (non-dual weights, overflow risk, or
            // XAV_L2_F32=1).
            let mut h_w = [0.0f32; HIDDEN1];
            let mut h_b = [0.0f32; HIDDEN1];
            screlu_deq(acc_white, self.scale, &mut h_w);
            screlu_deq(acc_black, self.scale, &mut h_b);
            return self.forward_l2_heads_dual(&h_w, &h_b, bucket);
        }

        // Quantized L2 path: SCReLU→i16, single i16 GEMV over the row-major
        // `w2_dual_i16` (half the bytes of `w2`, the step this is bound on).
        let mut q = [0i16; HIDDEN1_DUAL];
        let (q_w, q_b) = q.split_at_mut(HIDDEN1);
        screlu_quant(acc_white, self.scale, q_w);
        screlu_quant(acc_black, self.scale, q_b);

        let mut h2: [f32; HIDDEN2] = self.b2[..HIDDEN2].try_into().unwrap();
        l2_dual_i16(
            &self.w2_dual_i16,
            &q,
            &self.b2,
            1.0 / (self.scale * ACT_Q_F),
            &mut h2,
        );
        for v in h2.iter_mut() { *v = screlu_f32(*v); }
        self.forward_heads(&h2, bucket)
    }

    /// Layer 2 + heads for dual model: input is [h_w | h_b], each SCReLU'd to
    /// [0, 1].  Routes through the quantized i16 GEMV when the weights allow it
    /// (see `use_i16_l2`) — that path is ~1.6× faster and CP-equivalent to
    /// within ~1 cp — otherwise the legacy f32 GEMV.
    fn forward_l2_heads_dual(&self, h_w: &[f32; HIDDEN1], h_b: &[f32; HIDDEN1], bucket: usize) -> (i32, f32) {
        let mut h2: [f32; HIDDEN2] = self.b2[..HIDDEN2].try_into().unwrap();
        if self.use_i16_l2 {
            let mut q = [0i16; HIDDEN1_DUAL];
            quant_unit_activations(h_w, &mut q[..HIDDEN1]);
            quant_unit_activations(h_b, &mut q[HIDDEN1..]);
            l2_dual_i16(&self.w2_dual_i16, &q, &self.b2, 1.0 / (self.scale * ACT_Q_F), &mut h2);
        } else {
            // w2 is column-major (HIDDEN1_DUAL × HIDDEN2), split h_w / h_b halves.
            gemv_col(&self.w2[..HIDDEN1 * HIDDEN2], h_w, &mut h2);
            gemv_col(&self.w2[HIDDEN1 * HIDDEN2..], h_b, &mut h2);
        }
        for v in h2.iter_mut() { *v = screlu_f32(*v); }
        self.forward_heads(&h2, bucket)
    }

    /// Layer 2 + heads for single-perspective model: input is h1(1024).
    fn forward_l2_heads_single(&self, h1: &[f32; HIDDEN1], bucket: usize) -> (i32, f32) {
        // w2 is column-major (HIDDEN1 × HIDDEN2).
        let mut h2: [f32; HIDDEN2] = self.b2[..HIDDEN2].try_into().unwrap();
        gemv_col(&self.w2, h1, &mut h2);
        for v in h2.iter_mut() { *v = screlu_f32(*v); }
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
        let rank_quarter = rank / 2;
        t[sq] = rank_quarter * 4 + file_bucket;
        sq += 1;
    }
    t
};

/// Dual HalfKP encoding: returns ((white_indices, white_count), (black_indices, black_count)).
///
/// White perspective: absolute (white king = ours, white pieces = slots 0-5).
/// Black perspective: rank-flipped (black king = ours, black pieces = slots 0-5).
/// Both use the same feature formula: slot*64*32 + mapped_sq*32 + king_bucket.
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
                    w_indices[wc] = $slot * 64 * KING_BUCKETS + mapped * KING_BUCKETS + wk_bucket;
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
                    b_indices[bc] = $slot * 64 * KING_BUCKETS + mapped * KING_BUCKETS + bk_bucket;
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

/// Dual HalfKAv2 encoding: returns ((white_indices, white_count), (black_indices, black_count)).
///
/// HalfKAv2 uses exact king square (0-63) instead of a coarse bucket.
/// Own king is EXCLUDED; opponent king is included as slot 10.
/// Feature index: slot * 64 * 64 + mapped_sq * 64 + king_sq.
///
/// Slots 0-4: own P/N/B/R/Q. Slots 5-10: their P/N/B/R/Q/K.
/// Horizontal mirroring: when king is on files 4-7, both king_sq and piece
/// squares have their file bits flipped (`sq ^ 7`) so left-right mirror positions
/// share the same feature space.
///
/// Must stay bit-for-bit identical to `encode_board_halfkav2_dual` in features.py.
pub(crate) fn encode_dual_halfkav2(
    board: &ChessBoard,
) -> (([usize; 32], usize), ([usize; 32], usize)) {
    let white_bb = board.get_white();
    let black_bb = board.get_black();

    // White perspective: exact white king square with optional file mirror
    let wk_sq = (white_bb & board.get_kings()).0.trailing_zeros() as usize;
    let wk_sq = wk_sq.min(63);
    let mirror_w = (wk_sq % 8) >= 4;
    let king_w = if mirror_w { wk_sq ^ 7 } else { wk_sq };

    // Black perspective: black king rank-flipped, then optionally file-mirrored
    let bk_sq_raw = (black_bb & board.get_kings()).0.trailing_zeros() as usize;
    let bk_sq_raw = bk_sq_raw.min(63);
    let bk_flipped = bk_sq_raw ^ 56;
    let mirror_b = (bk_flipped % 8) >= 4;
    let king_b = if mirror_b { bk_flipped ^ 7 } else { bk_flipped };

    let mut w_indices = [0usize; 32];
    let mut b_indices = [0usize; 32];
    let mut wc = 0usize;
    let mut bc = 0usize;

    macro_rules! push_white_kav2 {
        ($bb:expr, $slot:expr) => {
            let mut bb: Bitboard = $bb;
            while bb.0 != 0 {
                let sq = bb.0.trailing_zeros() as usize;
                bb.0 &= bb.0 - 1;
                if wc < 32 {
                    let mapped = if mirror_w { sq ^ 7 } else { sq };
                    w_indices[wc] = $slot * 64 * 64 + mapped * 64 + king_w;
                    wc += 1;
                }
            }
        };
    }

    macro_rules! push_black_kav2 {
        ($bb:expr, $slot:expr) => {
            let mut bb: Bitboard = $bb;
            while bb.0 != 0 {
                let sq = bb.0.trailing_zeros() as usize;
                bb.0 &= bb.0 - 1;
                if bc < 32 {
                    let rank_flipped = sq ^ 56;
                    let mapped = if mirror_b { rank_flipped ^ 7 } else { rank_flipped };
                    b_indices[bc] = $slot * 64 * 64 + mapped * 64 + king_b;
                    bc += 1;
                }
            }
        };
    }

    // White perspective: own (white) P/N/B/R/Q = slots 0-4 (own king excluded)
    //                    their (black) P/N/B/R/Q/K = slots 5-10
    push_white_kav2!(white_bb & board.get_pawns(),   0);
    push_white_kav2!(white_bb & board.get_knights(), 1);
    push_white_kav2!(white_bb & board.get_bishops(), 2);
    push_white_kav2!(white_bb & board.get_rooks(),   3);
    push_white_kav2!(white_bb & board.get_queens(),  4);
    // white king: excluded from white perspective
    push_white_kav2!(black_bb & board.get_pawns(),   5);
    push_white_kav2!(black_bb & board.get_knights(), 6);
    push_white_kav2!(black_bb & board.get_bishops(), 7);
    push_white_kav2!(black_bb & board.get_rooks(),   8);
    push_white_kav2!(black_bb & board.get_queens(),  9);
    push_white_kav2!(black_bb & board.get_kings(),  10);

    // Black perspective: own (black) P/N/B/R/Q = slots 0-4 (own king excluded)
    //                    their (white) P/N/B/R/Q/K = slots 5-10, rank-flipped
    push_black_kav2!(black_bb & board.get_pawns(),   0);
    push_black_kav2!(black_bb & board.get_knights(), 1);
    push_black_kav2!(black_bb & board.get_bishops(), 2);
    push_black_kav2!(black_bb & board.get_rooks(),   3);
    push_black_kav2!(black_bb & board.get_queens(),  4);
    // black king: excluded from black perspective
    push_black_kav2!(white_bb & board.get_pawns(),   5);
    push_black_kav2!(white_bb & board.get_knights(), 6);
    push_black_kav2!(white_bb & board.get_bishops(), 7);
    push_black_kav2!(white_bb & board.get_rooks(),   8);
    push_black_kav2!(white_bb & board.get_queens(),  9);
    push_black_kav2!(white_bb & board.get_kings(),  10);

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
                indices[count] = $slot * 64 * KING_BUCKETS + mapped * KING_BUCKETS + bucket;
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
        // White king is slot 5 (ours). Expected index = 5*64*32 + 3*32 + 3 = 10339.
        let board = ChessBoard::new();
        assert!(board.is_white_active());
        let ((w_idx, wc), _) = encode_dual_halfkp(&board);
        let king_sq: usize = 4;  // e1
        let mirror_w = (king_sq % 8) >= 4;
        let mapped = if mirror_w { king_sq ^ 7 } else { king_sq }; // 3
        let file_bucket  = if mapped % 8 <= 3 { mapped % 8 } else { 7 - mapped % 8 };
        let rank_quarter = (mapped / 8) / 2;
        let bucket       = rank_quarter * 4 + file_bucket;
        let expected    = 5 * 64 * KING_BUCKETS + mapped * KING_BUCKETS + bucket;
        assert!(
            w_idx[..wc].contains(&expected),
            "White king index {expected} not found in dual white-pov features"
        );
    }

    #[test]
    fn test_dual_black_king_feature_index() {
        // Black king on e8 (sq=60). After rank-flip: 60^56=4. file=4 ≥ 4, so mirror_b=true.
        // mapped = 4^7 = 3. Bucket for sq=3: file=3, rank=0 → bucket=3.
        // Black king is slot 5 (ours in black-pov). Expected = 5*64*32 + 3*32 + 3 = 10339.
        let board = ChessBoard::new();
        let (_, (b_idx, bc)) = encode_dual_halfkp(&board);
        let bk_sq: usize = 60; // e8
        let flipped      = bk_sq ^ 56; // 4
        let mirror_b     = (flipped % 8) >= 4;
        let mapped       = if mirror_b { flipped ^ 7 } else { flipped }; // 3
        let file_bucket  = if mapped % 8 <= 3 { mapped % 8 } else { 7 - mapped % 8 };
        let rank_quarter = (mapped / 8) / 2;
        let bucket       = rank_quarter * 4 + file_bucket;
        let expected     = 5 * 64 * KING_BUCKETS + mapped * KING_BUCKETS + bucket;
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

    /// The quantized i16 Layer-2 path must track the legacy f32 path closely
    /// for the embedded weights across a spread of accumulator states.
    #[test]
    #[ignore = "requires src/eval.npz — run with --include-ignored"]
    fn test_i16_l2_matches_f32_l2() {
        let bytes = match std::fs::read("src/eval.npz") {
            Ok(b) => b,
            Err(_) => { println!("skipping: src/eval.npz not found"); return; }
        };
        let mut eval = match NeuralEvaluator::from_npz_bytes(&bytes) {
            Ok(e) => e,
            Err(e) => { println!("skipping: {e}"); return; }
        };
        if !eval.dual_perspective {
            println!("skipping: single-perspective model");
            return;
        }
        assert!(eval.use_i16_l2, "embedded weights should be i16-L2 safe");

        // Deterministic pseudo-random accumulator states in a realistic range
        // (post-L1 pre-activation values are typically within a few × scale).
        let mut state = 0x9E3779B97F4A7C15u64;
        let mut rng = || {
            state ^= state << 13; state ^= state >> 7; state ^= state << 17;
            state
        };
        let span = (eval.scale * 3.0) as i32;
        let mut max_diff = 0i32;
        for _ in 0..200 {
            let mut acc_w = [0i16; HIDDEN1];
            let mut acc_b = [0i16; HIDDEN1];
            for k in 0..HIDDEN1 {
                acc_w[k] = ((rng() % (2 * span as u64 + 1)) as i32 - span) as i16;
                acc_b[k] = ((rng() % (2 * span as u64 + 1)) as i32 - span) as i16;
            }
            for bucket in 0..eval.n_output_buckets {
                eval.use_i16_l2 = true;
                let (a, _) = eval.evaluate_from_accumulators(&acc_w, &acc_b, bucket);
                eval.use_i16_l2 = false;
                let (b, _) = eval.evaluate_from_accumulators(&acc_w, &acc_b, bucket);
                max_diff = max_diff.max((a - b).abs());
            }
        }
        assert!(max_diff <= 4, "i16 vs f32 L2 max CP diff {max_diff} (expected ≤4)");
        println!("i16 vs f32 L2: max CP diff over 200×{} states = {max_diff}", eval.n_output_buckets);
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
        let eval = match NeuralEvaluator::from_npz_bytes(&bytes) {
            Ok(e) => e,
            Err(e) => { println!("skipping: {e}"); return; }
        };
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

    // ── HalfKAv2 dual-perspective tests ──────────────────────────────────

    #[test]
    fn test_halfkav2_feature_count_starting() {
        // 32 pieces total; own king excluded per perspective → 31 active features each
        let board = ChessBoard::new();
        let ((w_idx, wc), (b_idx, bc)) = encode_dual_halfkav2(&board);
        assert_eq!(wc, 31, "White perspective should have 31 active features");
        assert_eq!(bc, 31, "Black perspective should have 31 active features");
        for &i in &w_idx[..wc] {
            assert!(i < HALFKAV2_FEATURE_DIM, "White index {i} out of range");
        }
        for &i in &b_idx[..bc] {
            assert!(i < HALFKAV2_FEATURE_DIM, "Black index {i} out of range");
        }
    }

    #[test]
    fn test_halfkav2_symmetric_starting_position() {
        // Starting position is symmetric: both perspectives must have the same feature set
        let board = ChessBoard::new();
        let ((w_idx, wc), (b_idx, bc)) = encode_dual_halfkav2(&board);
        assert_eq!(wc, bc);
        let mut w_set = w_idx[..wc].to_vec();
        let mut b_set = b_idx[..bc].to_vec();
        w_set.sort_unstable();
        b_set.sort_unstable();
        assert_eq!(w_set, b_set, "Starting position must be symmetric across perspectives");
    }

    #[test]
    fn test_halfkav2_independent_of_side_to_move() {
        let board_w = ChessBoard::new();
        let mut board_b = ChessBoard::new();
        board_b.toggle_turn();
        let ((ww, wc_w), (wb, bc_w)) = encode_dual_halfkav2(&board_w);
        let ((bw, wc_b), (bb, bc_b)) = encode_dual_halfkav2(&board_b);
        assert_eq!(wc_w, wc_b);
        assert_eq!(bc_w, bc_b);
        assert_eq!(&ww[..wc_w], &bw[..wc_b], "White perspective must not depend on side to move");
        assert_eq!(&wb[..bc_w], &bb[..bc_b], "Black perspective must not depend on side to move");
    }

    #[test]
    fn test_halfkav2_no_duplicate_indices() {
        let board = ChessBoard::new();
        let ((w_idx, wc), (b_idx, bc)) = encode_dual_halfkav2(&board);
        let mut w_sorted = w_idx[..wc].to_vec();
        w_sorted.sort_unstable();
        w_sorted.dedup();
        assert_eq!(w_sorted.len(), wc, "Duplicate indices in white perspective");
        let mut b_sorted = b_idx[..bc].to_vec();
        b_sorted.sort_unstable();
        b_sorted.dedup();
        assert_eq!(b_sorted.len(), bc, "Duplicate indices in black perspective");
    }

    #[test]
    fn test_halfkav2_indices_in_range_various_positions() {
        let fens = [
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            "r3r1k1/pp3pbp/1qp3p1/2B5/2BP2b1/Q1n2N2/P4PPP/3RR1K1 w - - 0 1",
            "7k/8/8/8/8/8/P7/K7 w - - 0 1",
            "k7/8/8/8/8/8/7P/7K w - - 0 1",
            "8/8/4k3/8/8/4K3/8/8 w - - 0 1",
        ];
        for fen in fens {
            let mut board = ChessBoard::new();
            board.set_from_fen(fen);
            let ((w_idx, wc), (b_idx, bc)) = encode_dual_halfkav2(&board);
            for &i in &w_idx[..wc] {
                assert!(i < HALFKAV2_FEATURE_DIM, "FEN {fen}: white index {i} out of range");
            }
            for &i in &b_idx[..bc] {
                assert!(i < HALFKAV2_FEATURE_DIM, "FEN {fen}: black index {i} out of range");
            }
        }
    }

    #[test]
    fn test_halfkav2_minimal_kk_position() {
        // K vs K: each side only has the opponent's king → 1 feature each
        let mut board = ChessBoard::new();
        board.set_from_fen("8/8/4k3/8/8/4K3/8/8 w - - 0 1");
        let ((_, wc), (_, bc)) = encode_dual_halfkav2(&board);
        assert_eq!(wc, 1, "K vs K: white perspective should have 1 feature");
        assert_eq!(bc, 1, "K vs K: black perspective should have 1 feature");
    }

    #[test]
    fn test_halfkav2_exact_opponent_king_index_starting() {
        // White king on e1 = sq 4. file=4 >= 4 → mirror=true, king_w = 4^7 = 3.
        // Black king on e8 = sq 60, slot=10 (their king from white POV).
        // mapped_sq = 60 ^ 7 = 59 (file-mirrored because mirror=true).
        // Expected white index = 10 * 64 * 64 + 59 * 64 + 3 = 44739.
        //
        // Black perspective (rank-flipped): bk_sq_raw=60, bk_flipped=60^56=4, mirror=true, king_b=3.
        // Their king (white, sq=4) in black POV: rank_flipped=4^56=60, mapped=60^7=59.
        // Expected black index = 10 * 64 * 64 + 59 * 64 + 3 = 44739.
        let board = ChessBoard::new();
        let ((w_idx, wc), (b_idx, bc)) = encode_dual_halfkav2(&board);
        let expected: usize = 10 * 64 * 64 + 59 * 64 + 3; // 44739
        assert!(
            w_idx[..wc].contains(&expected),
            "Opponent king index {expected} not found in white perspective"
        );
        assert!(
            b_idx[..bc].contains(&expected),
            "Opponent king index {expected} not found in black perspective"
        );
    }

    #[test]
    fn test_halfkav2_mirror_symmetry_a1_vs_h1_king() {
        // King+Pawn on a1/a2 vs King+Pawn on h1/h2 are horizontal mirrors.
        // Both should produce the same feature set.
        let mut board_a = ChessBoard::new();
        let mut board_h = ChessBoard::new();
        board_a.set_from_fen("7k/8/8/8/8/8/P7/K7 w - - 0 1");
        board_h.set_from_fen("k7/8/8/8/8/8/7P/7K w - - 0 1");
        let ((w_a, wc_a), _) = encode_dual_halfkav2(&board_a);
        let ((w_h, wc_h), _) = encode_dual_halfkav2(&board_h);
        assert_eq!(wc_a, wc_h, "Feature counts must match for mirrored positions");
        let mut a_sorted = w_a[..wc_a].to_vec();
        let mut h_sorted = w_h[..wc_h].to_vec();
        a_sorted.sort_unstable();
        h_sorted.sort_unstable();
        assert_eq!(a_sorted, h_sorted, "Horizontally mirrored positions must produce identical features");
    }

    #[test]
    fn test_halfkav2_exact_own_pawn_index_starting() {
        // White king on e1 = sq 4. mirror=true, king_w = 4^7 = 3.
        // White pawn on a2 = sq 8, slot=0 (own pawn from white POV).
        // mapped_sq = 8 ^ 7 = 15 (file-mirrored).
        // Expected index = 0 * 64 * 64 + 15 * 64 + 3 = 963.
        let board = ChessBoard::new();
        let ((w_idx, wc), _) = encode_dual_halfkav2(&board);
        let expected: usize = 0 * 64 * 64 + 15 * 64 + 3; // 963
        assert!(
            w_idx[..wc].contains(&expected),
            "White pawn a2 index {expected} not found in white perspective features"
        );
    }

    #[test]
    #[ignore = "requires src/eval.npz — run with --include-ignored"]
    fn test_incremental_screlu_matches_scratch() {
        let bytes = match std::fs::read("src/eval.npz") {
            Ok(b) => b,
            Err(_) => { println!("skipping: src/eval.npz not found"); return; }
        };
        let eval = match NeuralEvaluator::from_npz_bytes(&bytes) {
            Ok(e) => e,
            Err(e) => { println!("skipping: {e}"); return; }
        };
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

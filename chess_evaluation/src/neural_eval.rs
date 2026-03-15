//! Neural network evaluation for chess positions.
//!
//! Implements a small NNUE-like MLP (768 → 512 → 32 → 1) trained with
//! the Python pipeline in `nn_training/`. Weights are loaded from an NPZ
//! file exported by `scripts/export_weights.py`.
//!
//! The evaluator is off by default. Load weights, then enable:
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

// ── Architecture constants (must match Python model config) ───────────────

const FEATURE_DIM: usize = 768;
const HIDDEN1: usize = 512;
const HIDDEN2: usize = 32;

// ── Global state ──────────────────────────────────────────────────────────

static NEURAL_ENABLED: AtomicBool = AtomicBool::new(false);
static EVALUATOR: OnceLock<NeuralEvaluator> = OnceLock::new();

/// Minimum WDL confidence to trust the NN score. When `max(softmax(wdl))`
/// is below this value the position is too uncertain and the classical
/// evaluator is used instead. Stored as f32 bits for atomic access.
/// Default 0.0 = always trust NN (no fallback). Tune after training.
static CONFIDENCE_THRESHOLD: AtomicU32 = AtomicU32::new(0);

/// Load weights from an NPZ file path.
///
/// Does NOT automatically enable neural eval — call
/// `set_neural_eval_enabled(true)` afterwards.
pub fn init_neural_eval(path: &str) -> Result<(), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("Failed to read {path}: {e}"))?;
    init_neural_eval_from_bytes(&bytes)
}

/// Load weights from an in-memory NPZ blob (e.g. `include_bytes!`).
///
/// Useful for WASM builds where file I/O is unavailable.
pub fn init_neural_eval_from_bytes(bytes: &[u8]) -> Result<(), String> {
    let evaluator = NeuralEvaluator::from_npz_bytes(bytes)?;
    EVALUATOR
        .set(evaluator)
        .map_err(|_| "Neural evaluator already initialized".into())
}

/// Enable or disable neural network evaluation at runtime.
/// Has no effect if weights were never loaded.
pub fn set_neural_eval_enabled(enabled: bool) {
    NEURAL_ENABLED.store(enabled, Ordering::Relaxed);
}

/// Returns `true` if neural eval is currently enabled.
pub fn is_neural_eval_enabled() -> bool {
    NEURAL_ENABLED.load(Ordering::Relaxed)
}

/// Set the minimum WDL confidence required to use the NN score.
///
/// After a forward pass the WDL head produces a 3-class probability
/// distribution (win / draw / loss). `max(softmax(wdl))` is the
/// confidence — how sure the model is about the outcome.
///
/// - `0.0` (default) — always use NN, never fall back.
/// - `0.4` — fall back when no single outcome exceeds 40% probability
///            (the model is nearly equally lost between all three outcomes).
/// - `0.5` — fall back unless the model leans clearly toward one outcome.
///
/// Positions that trigger fallback are unusual structures the model hasn't
/// learned well; the classical evaluator handles them more reliably.
pub fn set_neural_confidence_threshold(threshold: f32) {
    CONFIDENCE_THRESHOLD.store(threshold.to_bits(), Ordering::Relaxed);
}

pub fn get_neural_confidence_threshold() -> f32 {
    f32::from_bits(CONFIDENCE_THRESHOLD.load(Ordering::Relaxed))
}

/// Returns `Some(score)` in centipawns from **white's perspective** when
/// neural eval is enabled, weights are loaded, and the model is sufficiently
/// confident. Returns `None` to fall back to the classical evaluator.
#[inline]
pub fn try_neural_eval(board: &ChessBoard) -> Option<i32> {
    if !NEURAL_ENABLED.load(Ordering::Relaxed) {
        return None;
    }
    let threshold = f32::from_bits(CONFIDENCE_THRESHOLD.load(Ordering::Relaxed));
    EVALUATOR.get().and_then(|e| {
        let (stm_score, confidence) = e.evaluate_with_confidence(board);
        if confidence < threshold {
            return None; // uncertain position → classical eval
        }
        Some(if board.is_white_active() { stm_score } else { -stm_score })
    })
}

// ── Evaluator ─────────────────────────────────────────────────────────────

pub struct NeuralEvaluator {
    /// Layer 1 weights stored **transposed**: [FEATURE_DIM × HIDDEN1].
    ///
    /// Transposing at load time lets us accumulate sparse input features
    /// with a single contiguous slice read per active feature (~32 active
    /// out of 768), rather than doing a full dense matrix-vector product.
    w1_t: Vec<f32>, // FEATURE_DIM * HIDDEN1
    b1: Vec<f32>,   // HIDDEN1

    /// Layer 2 weights in row-major order: [HIDDEN2 × HIDDEN1].
    w2: Vec<f32>, // HIDDEN2 * HIDDEN1
    b2: Vec<f32>, // HIDDEN2

    /// CP head weights: [HIDDEN2] (the single output row, flattened).
    w3: Vec<f32>, // HIDDEN2
    b3: f32,

    /// WDL head weights: [3 × HIDDEN2] row-major.
    /// Used to compute confidence (max softmax probability).
    w_wdl: Vec<f32>, // 3 * HIDDEN2
    b_wdl: Vec<f32>, // 3
}

impl NeuralEvaluator {
    fn from_npz_bytes(data: &[u8]) -> Result<Self, String> {
        let cursor = std::io::Cursor::new(data);
        let mut zip = zip::ZipArchive::new(cursor)
            .map_err(|e| format!("Not a valid NPZ/zip file: {e}"))?;

        let scale = read_npy_f32_scalar(&mut zip, "scale.npy")?;

        let w1_i16 = read_npy_i16(&mut zip, "backbone_0_weight.npy")?;
        let b1_i16 = read_npy_i16(&mut zip, "backbone_0_bias.npy")?;
        let w2_i16 = read_npy_i16(&mut zip, "backbone_3_weight.npy")?;
        let b2_i16 = read_npy_i16(&mut zip, "backbone_3_bias.npy")?;
        let w3_i16 = read_npy_i16(&mut zip, "cp_head_weight.npy")?;
        let b3_i16 = read_npy_i16(&mut zip, "cp_head_bias.npy")?;
        let w_wdl_i16 = read_npy_i16(&mut zip, "wdl_head_weight.npy")?;
        let b_wdl_i16 = read_npy_i16(&mut zip, "wdl_head_bias.npy")?;

        let dq = |v: &[i16]| -> Vec<f32> { v.iter().map(|&x| x as f32 / scale).collect() };

        // w1 is originally [HIDDEN1 × FEATURE_DIM] row-major.
        // Transpose to [FEATURE_DIM × HIDDEN1] for cache-friendly sparse access.
        let w1_row = dq(&w1_i16);
        let mut w1_t = vec![0.0f32; FEATURE_DIM * HIDDEN1];
        for j in 0..HIDDEN1 {
            for i in 0..FEATURE_DIM {
                w1_t[i * HIDDEN1 + j] = w1_row[j * FEATURE_DIM + i];
            }
        }

        Ok(Self {
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

    /// Evaluate a position.
    ///
    /// Returns `(cp, confidence)` where:
    /// - `cp` is centipawns from the **side-to-move's** perspective.
    /// - `confidence` is `max(softmax(wdl))` — the probability the model
    ///   assigns to its most likely outcome (win/draw/loss). Range 0..1;
    ///   values below ~0.4 indicate the model is uncertain about the position.
    pub fn evaluate_with_confidence(&self, board: &ChessBoard) -> (i32, f32) {
        let features = encode_features(board);

        // ── Layer 1: sparse input accumulation (features are 0 or 1) ──────
        // Start with biases, then for each active feature add its column
        // from the transposed weight matrix. ~32 active features → ~32 × 512
        // adds instead of 768 × 512.
        let mut h1 = [0.0f32; HIDDEN1];
        h1.copy_from_slice(&self.b1);
        for i in 0..FEATURE_DIM {
            if features[i] != 0.0 {
                let src = &self.w1_t[i * HIDDEN1..(i + 1) * HIDDEN1];
                for j in 0..HIDDEN1 {
                    h1[j] += src[j];
                }
            }
        }
        for v in h1.iter_mut() {
            *v = v.max(0.0); // ReLU
        }

        // ── Layer 2: dense [HIDDEN2 × HIDDEN1] ────────────────────────────
        let mut h2 = [0.0f32; HIDDEN2];
        for j in 0..HIDDEN2 {
            let row = &self.w2[j * HIDDEN1..(j + 1) * HIDDEN1];
            let mut acc = self.b2[j];
            for i in 0..HIDDEN1 {
                acc += row[i] * h1[i];
            }
            h2[j] = acc.max(0.0); // ReLU
        }

        // ── CP head: dot(w3, h2) + b3 ─────────────────────────────────────
        let mut cp = self.b3;
        for i in 0..HIDDEN2 {
            cp += self.w3[i] * h2[i];
        }

        // ── WDL head → confidence ─────────────────────────────────────────
        // logits: [3 × HIDDEN2] row-major → win, draw, loss
        let mut logits = [0.0f32; 3];
        for k in 0..3 {
            let row = &self.w_wdl[k * HIDDEN2..(k + 1) * HIDDEN2];
            let mut acc = self.b_wdl[k];
            for i in 0..HIDDEN2 {
                acc += row[i] * h2[i];
            }
            logits[k] = acc;
        }
        // Numerically stable softmax
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

/// Encode board to a 768-dim binary feature vector from the side-to-move's
/// perspective, matching Python's `encode_board_12x64`.
///
/// Layout (after optional mirror for black-to-move):
/// - [  0.. 63] side-to-move pawns
/// - [ 64..127] side-to-move knights
/// - [128..191] side-to-move bishops
/// - [192..255] side-to-move rooks
/// - [256..319] side-to-move queens
/// - [320..383] side-to-move king
/// - [384..447] opponent pawns
/// - [448..511] opponent knights
/// - [512..575] opponent bishops
/// - [576..639] opponent rooks
/// - [640..703] opponent queens
/// - [704..767] opponent king
fn encode_features(board: &ChessBoard) -> [f32; FEATURE_DIM] {
    let mut feat = [0.0f32; FEATURE_DIM];

    let white_to_move = board.is_white_active();
    // If black to move, flip square indices vertically (rank mirror, sq ^ 56)
    // so the model always sees positions from the mover's point of view.
    let flip = !white_to_move;

    let (ours, theirs) = if white_to_move {
        (board.get_white(), board.get_black())
    } else {
        (board.get_black(), board.get_white())
    };

    set_features(&mut feat, ours & board.get_pawns(), 0, flip);
    set_features(&mut feat, ours & board.get_knights(), 64, flip);
    set_features(&mut feat, ours & board.get_bishops(), 128, flip);
    set_features(&mut feat, ours & board.get_rooks(), 192, flip);
    set_features(&mut feat, ours & board.get_queens(), 256, flip);
    set_features(&mut feat, ours & board.get_kings(), 320, flip);

    set_features(&mut feat, theirs & board.get_pawns(), 384, flip);
    set_features(&mut feat, theirs & board.get_knights(), 448, flip);
    set_features(&mut feat, theirs & board.get_bishops(), 512, flip);
    set_features(&mut feat, theirs & board.get_rooks(), 576, flip);
    set_features(&mut feat, theirs & board.get_queens(), 640, flip);
    set_features(&mut feat, theirs & board.get_kings(), 704, flip);

    feat
}

#[inline]
fn set_features(feat: &mut [f32; FEATURE_DIM], mut bb: Bitboard, offset: usize, flip: bool) {
    while bb.0 != 0 {
        let sq = bb.0.trailing_zeros() as usize;
        bb.0 &= bb.0 - 1;
        let mapped = if flip { sq ^ 56 } else { sq };
        feat[offset + mapped] = 1.0;
    }
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

/// Parse an .npy file header; return `(data_offset, n_elements)`.
/// Validates that `expected_dtype` appears in the header dict.
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

/// Extract the product of shape dimensions from an .npy header string.
/// Handles shapes like `(512, 768)`, `(512,)`, `(1,)`.
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

    // Empty tuple `()` or scalar `(1,)` with product==0 → treat as 1 element.
    Ok(if product == 0 { 1 } else { product })
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use chess_board::ChessBoard;

    /// Smoke-test: feature encoding of the starting position should have
    /// exactly 32 active features (16 white pieces + 16 black pieces).
    #[test]
    fn test_feature_count_starting_position() {
        let board = ChessBoard::new();
        let features = encode_features(&board);
        let active: usize = features.iter().filter(|&&v| v != 0.0).count();
        assert_eq!(active, 32, "Starting position should have 32 active features");
    }

    /// From black's perspective the board is mirrored, but still 32 features.
    #[test]
    fn test_feature_count_black_to_move() {
        let mut board = ChessBoard::new();
        board.toggle_turn();
        let features = encode_features(&board);
        let active: usize = features.iter().filter(|&&v| v != 0.0).count();
        assert_eq!(active, 32);
    }

    /// White king starts on e1 (square 4). When white to move (no flip),
    /// it should appear at offset 320 + 4 = 324.
    #[test]
    fn test_white_king_feature_white_to_move() {
        let board = ChessBoard::new();
        assert!(board.is_white_active());
        let features = encode_features(&board);
        // White king on e1 = square 4, side-to-move king offset = 320
        assert_eq!(features[320 + 4], 1.0, "White king should be at feature 324");
    }

    /// When it's black's turn the board is mirrored: black king on e8
    /// (square 60) maps to sq ^ 56 = 4, in side-to-move king slot (offset 320).
    #[test]
    fn test_black_king_feature_black_to_move() {
        let mut board = ChessBoard::new();
        board.toggle_turn();
        let features = encode_features(&board);
        // Black king on e8 = square 60; flipped: 60 ^ 56 = 4; offset 320
        assert_eq!(features[320 + 4], 1.0, "Black king (mirrored) should be at feature 324");
    }
}

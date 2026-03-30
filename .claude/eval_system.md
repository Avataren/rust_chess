# Evaluation System — Working Knowledge

The engine has two evaluators (NNUE neural and classical HCE) and four Cargo feature modes that control which is compiled and how it's selected at runtime. Getting this wrong produces silent mis-evaluations with no compile errors.

---

## The four Cargo features

Defined in `chess_evaluation/Cargo.toml`. Exactly one must be active — they are mutually exclusive.

| Feature | What it compiles | Runtime behaviour |
|---------|-----------------|-------------------|
| `classical-eval` | HCE only — zero NN code | Always uses HCE |
| `nn-full-forward` | NN only — no HCE fallback | Always uses NN (full forward pass per call) |
| `nn-incremental` | NN with incremental i16 accumulator + HCE fallback | Always uses NN via accumulators in search; full forward elsewhere |
| `runtime-switch` | Both HCE and NN | Switched at runtime via `set_neural_eval_enabled()` / UCI setoptions |

**Default feature** of the `chess_evaluation` crate: `runtime-switch`.

**Workspace override** (`Cargo.toml` root `[workspace.dependencies]`):
```toml
chess_evaluation = { path = "chess_evaluation", default-features = false, features = ["nn-incremental"] }
```
This is the live setting. **Everything that does `chess_evaluation = { workspace = true }` gets `nn-incremental`**, including `chess_uci` and `self_play`. The crate default (`runtime-switch`) is never used in production.

**What uses each feature:**
- `chess_uci`, `self_play`, `bench`: `nn-incremental` (via workspace dep)
- `chess` (Bevy app): `nn-incremental` (explicit override)
- Unit tests inside `chess_evaluation`: `classical-eval` is the default for the crate's own tests — fast, deterministic, no NN loading required
- `puzzle_bench`: `runtime-switch` (loaded at runtime via `--eval-file`)

---

## Embedded weights vs runtime weights

**Embedded** (`include_bytes!`):
`chess_uci/src/main.rs` embeds `chess_evaluation/src/eval.npz` at compile time:
```rust
#[cfg(any(feature = "nn-full-forward", feature = "nn-incremental"))]
static NNUE_WEIGHTS: &[u8] = include_bytes!("../../chess_evaluation/src/eval.npz");
```
This file (`chess_evaluation/src/eval.npz`) is a **static fallback** — it's whatever weights happened to be there at the last `cargo build`. It is NOT kept in sync with `nn_training/artifacts/eval.npz`.

**Runtime override** (via UCI `setoption`):
```
setoption name EvalFile value /path/to/eval.npz
setoption name NeuralEval value true
```
This overrides the embedded weights with whatever NPZ file you point to. The lichess bot config sets:
```yaml
EvalFile: /home/avataren/src/rust_chess/nn_training/artifacts/eval.npz
NeuralEval: true
```

**Critical:** `chess_evaluation/src/eval.npz` (embedded) and `nn_training/artifacts/eval.npz` (runtime) are two different files. The self-play loop updates `artifacts/eval.npz` on every promotion. The embedded file only updates when you explicitly copy it and rebuild. **Do not confuse them.**

**When `nn-incremental` is active**, the embedded weights are loaded on the first `isready` if `EvalFile` was not set:
```rust
if !is_neural_eval_initialized() {
    init_neural_eval_from_bytes(NNUE_WEIGHTS).expect("...");
}
```

---

## NNUE architecture

### Dual-perspective model (current)

Two independent encodings of the same position — one from white's king position, one from black's — are computed and concatenated. This lets the model be aware of both kings simultaneously without needing to flip and re-evaluate.

```
Board position
     │
     ├─ White HalfKP encoding → [12,288 sparse indices, max 32 active]
     │                                     │
     │                              Layer 1 (EmbeddingBag)
     │                              12,288 → 512  (i16, quantized)
     │                              acc_white[512]
     │
     └─ Black HalfKP encoding → [12,288 sparse indices, max 32 active]
                                           │
                                    Layer 1 (shared weights)
                                    12,288 → 512  (i16, quantized)
                                    acc_black[512]

     [acc_white | acc_black]  →  SCReLU  →  1024 f32 vector
                                    │
                             Layer 2: 1024 → 32
                                    │
                                  SCReLU
                                    │
                         ┌──────────┴──────────┐
                    CP head                 WDL head
                   32 → 1 cp              32 → 3 logits
                 (×8 buckets)             (×8 buckets)
```

**HalfKP feature index:**
```
feature_idx = slot * 2048 + mapped_sq * 32 + king_bucket
```
- `slot`: piece type 0–11 (white pawn=0 … white king=5, black pawn=6 … black king=11)
- `mapped_sq`: 0–63 (rank-flipped for black perspective: XOR 56 flips rank, XOR 7 flips file)
- `king_bucket`: 0–31 (32 zones derived from king position, horizontally mirrored)
- Input dim = 12 × 64 × 32 = **24,576**

**Output buckets** (8): Selected by total piece count:
```
bucket = clamp((total_pieces - 2) * 8 / 30, 0, 7)
```
Bucket 0 = 2 pieces (bare kings), bucket 7 = 32 pieces (opening). Separate CP and WDL weight matrices per bucket.

**Score convention:** Always **white-absolute** centipawns (positive = white winning). The search negates as needed based on side to move.

**Activation:** SCReLU = `clamp(x, 0, 1)²` — squared clamped ReLU. Better quantization robustness than regular ReLU.

**Quantization:** Weights stored as `i16` in the NPZ, with a `f32` scale factor. Dequantization: `value_f32 = value_i16 / scale`. The accumulators operate in `i16` space with saturating arithmetic; dequantization to `f32` happens only in the final forward pass from the accumulators.

---

## Incremental accumulator (nn-incremental feature)

Instead of recomputing Layer 1 from scratch each position, the accumulator stores the Layer 1 pre-activation and updates it incrementally as pieces move:

- **Init**: `init_accumulators_for_board(board, acc_white, acc_black)` — full encode from scratch
- **Make move**: add/subtract the feature columns for moved/captured pieces
- **Undo move**: reverse the delta

The `SearchContext` struct holds `acc_white[512]` and `acc_black[512]` as a stack (one per search depth). This is the dominant performance win — Layer 1 (the most expensive operation) becomes 2 SIMD add/subtract operations per move instead of a full GEMV.

**SIMD dispatch:**
- x86_64 + AVX2: `_mm256_adds_epi16` (16 i16/register, processes 512 elements in ~32 cycles)
- wasm32 + simd128: `i16x8_add_sat`
- Fallback: scalar `i16::saturating_add` / `saturating_sub`

**When accumulators are unavailable** (e.g. TB probe contexts, tests): falls back to full forward pass.

---

## Classical HCE (fallback / classical-eval feature)

Used when the neural eval is disabled or unavailable. Evaluates:

- **Material** (tapered MG/EG): P=82/94, N=337/281, B=365/297, R=477/512, Q=1025/936
- **Piece-square tables** for all pieces
- **Pawn structure** (isolated −15, doubled −15, passed pawn bonuses by rank) — cached via pawn hash
- **Mobility** (knights 4cp/sq, bishops 3cp/sq, rooks 2cp/sq, queens 2cp/sq; excludes own pieces and pawn-attacked squares)
- **King safety**: pawn shield (MG only, 20cp/missing pawn), attack-count table (non-linear, 0–1095cp)
- **Bishop pair** bonus: 30 MG / 50 EG
- **Rook bonuses**: open file (20/15), semi-open (10/8), 7th rank (25/35), behind passed pawn (20 EG)
- **Knight outpost**: 30 MG / 20 EG
- **Mop-up**: drives losing king to corner in won endgames; scales with 50-move clock

Score convention: **white-absolute** centipawns.

---

## Eval dispatch in the search

Entry point: `evaluate_board(board)` in `board_evaluation.rs`. Compile-time dispatch:

```
nn-incremental  →  eval_accum_direct(board, acc_white, acc_black)
                   (no runtime check — always NN; falls back to full-forward if acc not available)

nn-full-forward →  eval_direct(board)
                   (always NN, full forward, no HCE fallback)

runtime-switch  →  try_neural_eval(board)  →  Some(score) if enabled + loaded
                   else  classical_eval(board)

classical-eval  →  classical_eval(board)
```

**NeuralConfidence** (runtime-switch only): If `set_neural_confidence_threshold(t)` is set (e.g. 0.4), positions where WDL softmax confidence < t fall back to HCE. The self-play loop uses `NeuralConfidence=0.4` during training data generation.

---

## Gotchas

**The two eval.npz files are not the same thing.**
`chess_evaluation/src/eval.npz` is baked into the binary at compile time. It only updates when you copy and rebuild. `nn_training/artifacts/eval.npz` is the live promoted model, updated each iteration. The lichess bot uses the live file via `EvalFile` setoption.

**Changing the workspace feature requires a full rebuild.**
Switching between `nn-incremental`, `runtime-switch`, etc. in `Cargo.toml` will recompile `chess_evaluation` and every crate that depends on it. `cargo build --release -p chess_uci` is not enough — you need to let Cargo rebuild all affected crates.

**Tests in chess_evaluation use classical-eval by default.**
Running `cargo test -p chess_evaluation` compiles the crate with its default features (`runtime-switch`). The TB tests that probe Syzygy are `#[ignore]` and need explicit `--ignored` flag. Neural eval tests require a live NPZ file.

**Score perspective is dual-model only.**
Old single-perspective models output side-to-move centipawns. The dual-perspective model always outputs white-absolute. The dataset CP convention must match: `JsonlDualPositionDataset` converts labels to white-absolute internally. Do not mix single-perspective weights with dual-perspective training code.

**nn-incremental has no runtime enable/disable.**
Unlike `runtime-switch`, `nn-incremental` cannot be toggled via `setoption NeuralEval false`. Once the binary is built with `nn-incremental`, the NN is always used in search. `NeuralEval=true/false` setoptions are parsed but only meaningful for `runtime-switch` builds.

---

## Quick reference: changing the eval

| Goal | What to do |
|------|-----------|
| Use latest trained weights in lichess bot | Handled automatically — loop updates `artifacts/eval.npz` on promotion |
| Update embedded weights (for distribution) | Copy `artifacts/eval.npz` → `chess_evaluation/src/eval.npz`, then rebuild |
| Test with classical eval only | `cargo test -p chess_evaluation --features classical-eval` |
| Profile NN vs HCE strength difference | Build two binaries: one with `nn-incremental`, one with `classical-eval`; run self_play |
| Switch to runtime-switch (e.g. for experiments) | Edit root `Cargo.toml` `[workspace.dependencies]`, change `features = ["nn-incremental"]` to `features = ["runtime-switch"]`, rebuild |

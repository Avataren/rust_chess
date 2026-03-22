# NNUE Architecture Improvement Roadmap

## Current State

| Component | Current | Notes |
|---|---|---|
| Feature encoding | HalfKP, 12×64×16 = **12,288** features | 16 king buckets (coarse) |
| Accumulator | hidden_dim=**1024** per perspective | 2048 combined |
| Bottleneck | fc2 2048→**64** (hidden2_dim) | recently upgraded from 32 |
| Output buckets | **8** (by piece count) | cp head + WDL head each |
| Activation | SCReLU throughout | |
| Quantization | int16, scale=256 | |
| Training data | ~54M positions | 2 Lichess months, depth-14 |
| Inference | AVX2 SIMD, column-major GEMV | incremental accumulators |

---

## Impact vs Effort Matrix

| # | Improvement | ELO Impact | Effort | Data Regen? |
|---|---|---|---|---|
| A1 | hidden2_dim 64→128 | +10-20 | Low | No |
| A2 | WDL game-result blending | +15-25 | Low→Medium | Partial |
| A3 | King buckets 16→32 | +20-40 | Medium | .npy only |
| A4 | Second hidden layer (fc3) | +15-30 | Medium | No |
| A5 | Skip/direct connection | +5-15 | Medium | No |
| A6 | Lazy eval in qsearch | +5-15 NPS | Medium | No (Rust only) |
| A7 | Accumulator refresh cache | +5-10 NPS | Medium-High | No (Rust only) |
| A8 | HalfKAv2 (45,056 features) | +30-60 | High | .npy only |
| A9 | 200M+ positions (10 months) | +30-50 | High | Yes |
| A10 | Teacher→student distillation | +20-40 (student) | High | Partial |

---

## Phase 1 — Quick Wins (No Data Regeneration)

### A1. Widen hidden2_dim 64→128

The 2048→64 compression is still aggressive. Doubling it costs ~0.5MB weight and
roughly doubles fc2 inference cost — manageable given the SIMD GEMV optimizations.

**`neural_eval.rs`**: change `const HIDDEN2: usize = 64` → `128`.
Update `gemv_col32_avx2` to process 128 outputs in two passes of 8 YMM registers
each (16 YMM registers needed total — exactly the x86_64 limit; split into two loops
over HIDDEN2/2 each to stay within register budget):

```rust
// Pass 1: outputs 0..63  (8 YMM registers)
// Pass 2: outputs 64..127 (8 YMM registers)
```

**`model.py`**: `hidden2_dim: 128` in config. No code change needed.

**`configs/`**: new YAML `halfkp_dual_all_54m_1024_h128_8b.yaml`.

---

### A2. WDL Game-Result Blending

Current `cp_to_wdl_target` is a pure Stockfish-logistic function — the actual game
outcome is never used. Stockfish blends eval WDL with game result at 50/50.

**Minimum viable (no regen):** Add `wdl_lambda` to configs (default 1.0 = current
behaviour). Infrastructure ready for when result data is available.

**Full version:** Requires game outcome in dataset.

`generate_data.py` — `_iter_pgn_positions` yields `(fen, result)` where result is
0/0.5/1 from the PGN `Result` header.

`preprocess_dataset.py` — writes `.result.npy` column (uint8: 0=loss, 1=draw, 2=win).

`BinaryDualPositionDataset` — loads `.result.npy`; passes to `cp_to_wdl_target`.

`features.py` — `cp_to_wdl_target(cp, game_result=None, lambda_eval=0.7)`:
```python
eval_wdl = pure_logistic(cp)
if game_result is not None:
    game_wdl = [game_result, 1-abs(2*game_result-1), 1-game_result]
    return lambda_eval * eval_wdl + (1 - lambda_eval) * game_wdl
```

**Data regen needed**: `.result.npy` only — existing `.npy` files remain valid.
JSONL files need `result` field added (requires re-running labeling pipeline).

---

### A3. Training Improvements (No Arch Change)

- **Cosine warm restarts**: replace `CosineAnnealingLR` with `CosineAnnealingWarmRestarts(T_0=30)` in `train.py` — prevents getting stuck in local minima on long runs.
- **Label smoothing on WDL**: add `epsilon=0.01` to `soft_target_cross_entropy` — improves calibration slightly.

---

## Phase 2 — Architecture Changes (Retrain Required, No Data Regen)

### A4. Second Hidden Layer

Stockfish uses: `accumulator → fc2 → screlu → fc3 → screlu → heads`, where fc3 is
deliberately tiny to keep per-node inference cost low.

Proposed: `2048 → fc2(64) → screlu → fc3(16) → screlu → heads`

**`model.py` `EvalNetDual`**:
```python
self.fc3 = nn.Linear(hidden2_dim, hidden3_dim)
# forward:
h = screlu(self.fc3(h))
```
Add `hidden3_dim: 16` to configs.

**`neural_eval.rs`**: add `w3_fc3`, `b3_fc3` fields; add a second GEMV in
`forward_l2_heads_dual` (64→16, trivially cheap).

**`export_weights.py`**: emit `backbone_6_weight` / `backbone_6_bias`.

**Key benefit**: heads now operate on 16 values instead of 64 — cp/WDL dot products
become 4× cheaper, partially offsetting the fc3 cost.

---

### A5. Skip / Direct Connection

A direct connection from the fc2 output (64-dim) to the output heads, bypassing
screlu and fc3. Gradient shortcutting reduces vanishing gradient in early training.

**`model.py`**: `self.skip_cp = nn.Linear(hidden2_dim, n_output_buckets)` applied
to fc2 pre-activation; added to final cp logit.

**`neural_eval.rs`**: one additional 64-element dot product per eval (trivial cost).

---

### A6. Lazy Eval in Quiescence Search

Currently every qsearch node pays the full `eval_accum_direct` cost even when
stand-pat pruning would cut anyway.

**`alpha_beta.rs`** — modify `quiescence` signature to accept `parent_static_eval`:
```rust
if parent_static_eval + MAX_MATERIAL_GAIN < alpha {
    return (alpha, None);  // skip eval entirely
}
```
`MAX_MATERIAL_GAIN ≈ 1100` (queen + margin).

Expected NPS gain: **5-15%** depending on tree shape. Pure Rust change, no Python
or data implications.

---

### A7. Accumulator Refresh Cache for King Moves

King moves currently trigger full accumulator recomputation (all ~32 features
re-added from scratch). A small cache keyed by `(king_square, bucket)` avoids
this for repeated king positions (common in endgames and repetitions).

**`alpha_beta.rs`**: add `refresh_table: [[Option<([i16; HIDDEN1], [i16; HIDDEN1])>; 64]; MAX_PLY]`
or a hash-map variant. On king move: check table before calling `init_accumulators_direct`.

Expected NPS gain: **5-10%** in endgame-heavy positions.

---

## Phase 3 — Feature Set Upgrade (Requires .npy Regeneration)

### A3. King Buckets: 16 → 32

Doubles the feature space from 12,288 to 24,576 by splitting rank halves into
rank quarters (4 rank zones × 4 mirrored files = 16 → 32 buckets).

**Why**: "King on g1 (castled)" and "king on g3 (exposed)" currently share a bucket.
Finer buckets let the model distinguish king safety states it currently cannot.

**`features.py`** — `_build_king_bucket_table()`:
```python
# Change: rank_half = 0 if rank <= 3 else 1
# To:     rank_quarter = rank // 2  (gives 0,1,2,3)
file_bucket = file if file <= 3 else 7 - file
table.append(rank_quarter * 4 + file_bucket)
```
`HALFKP_FEATURE_DIM = 12 * 64 * 32 = 24576`

**`neural_eval.rs`** — update `KING_BUCKET` table and `HALFKP_FEATURE_DIM`:
```rust
let rank_quarter = rank / 2;  // was: if rank <= 3 { 0 } else { 1 }
t[sq] = rank_quarter * 4 + file_bucket;
```
Change feature index formula: `slot * 64 * 32 + sq * 32 + bucket`
(was `slot * 64 * 16 + sq * 16 + bucket`).

**`from_npz_bytes`**: add `24576` to the valid `feature_dim` check.

**Embedding size**: 24,576 × 1024 × 2 bytes = **~50MB** (doubled from ~25MB).
Accumulator update column length unchanged (still HIDDEN1=1024).

**Data regen**: run `nnue_preprocess --dual` on existing JSONL files. No relabeling.
All existing `.white_indices.npy` / `.black_indices.npy` files must be regenerated.

---

### A8. HalfKAv2 Feature Set (Full Stockfish Parity)

**Why**: Encodes the king's exact square (0-63) into every feature index rather
than bucketing. The network can learn king-position-specific patterns that are
invisible with any bucketing scheme.

**Feature formula**: `slot * 64 * 64 + piece_square * 64 + king_square`
where king_square uses horizontal mirroring (XOR 7 if king file ≥ 4).
Feature dim: 11 × 64 × 64 = **45,056** (king excluded as "own piece" = 11 slots).

**Embedding size**: 45,056 × 1024 × 2 bytes = **~92MB** (~4× current).

**`features.py`**: new `encode_board_halfkav2_dual()` function.
**`neural_eval.rs`**: new `encode_dual_halfkav2()` function.
**`preprocess_dataset.py`**: `--halfkav2` flag.

**Data regen**: run new preprocessor on existing JSONL. No relabeling.

**Recommendation**: implement 32-bucket first (A3) — gives ~80% of HalfKAv2
benefit at 25% of the complexity and embedding cost.

---

## Phase 4 — Data Quality & Scale

### Scale to 200M+ Positions

Current dataset (54M, 2 months) is modest. 10 months of 2019-2022 high-ELO
Lichess games would yield ~200M positions.

**No code changes needed.** Use existing pipeline with additional PGN files:
```bash
for month in 2021-02 2021-03 ... 2022-06; do
  bash scripts/gen_balanced_dataset.sh lichess_db_standard_rated_${month}.pgn \
    balanced_${month} --skip-games 0
done
# Combine all datasets
```

Bottleneck: Stockfish labeling at ~100-200 positions/sec/worker.
With 32 workers at depth-14: ~100M positions per ~14 hours.

### Adaptive Depth Labeling

Use depth-20 for endgame positions (≤12 pieces) and Syzygy tablebases for ≤7 pieces.

**`generate_data.py`**: accept `(fen, depth)` tuples; route by piece count:
- ≤7 pieces → Syzygy (already supported via `--syzygy-path`, depth-1)
- 8-12 pieces → depth-18
- 13+ pieces → depth-14

Requires small refactor to `_worker_init` and `_label_fen`.

---

## Phase 5 — Distillation Path

Train a large, unconstrained teacher then distill to a fast student for deployment.

### Teacher Model
- HalfKAv2 features (45,056 input)
- hidden_dim=2048 per perspective
- hidden2_dim=256, hidden3_dim=64
- 8 output buckets
- 200M+ positions

### Student Model (deployment)
- Current architecture (HalfKP, 1024-dim, h2=64)
- Trained on teacher soft labels instead of Stockfish labels
- Much lower noise floor; teacher provides richer gradient signal

### `train.py` distillation mode
```yaml
distillation:
  enabled: true
  teacher_checkpoint: artifacts/teacher.pt
  temperature: 2.0
  alpha: 0.5   # blend: alpha*distill_loss + (1-alpha)*label_loss
```

Load teacher at startup, run in `torch.no_grad()` mode per batch.
WDL loss: `alpha * KL(student || teacher) + (1-alpha) * CE(student || stockfish)`.
CP loss: `MSE(student_cp, teacher_cp)`.

~50 lines added to `train.py`. No Rust or data changes.

---

## What Requires Data Regeneration

| Change | JSONL Regen | .npy Regen | Retrain |
|---|---|---|---|
| hidden2_dim 64→128 | No | No | Yes |
| Second hidden layer | No | No | Yes |
| Skip connection | No | No | Yes |
| Lazy qsearch | No | No | No (Rust only) |
| Accumulator refresh cache | No | No | No (Rust only) |
| WDL game-result (min) | No | No | Yes |
| WDL game-result (full) | Yes | Yes (.result.npy) | Yes |
| King buckets 16→32 | No | Yes | Yes |
| HalfKAv2 features | No | Yes | Yes |
| More Lichess months | Yes | Yes | Yes |
| Adaptive depth labeling | Yes | Yes | Yes |

---

## Recommended Sequence

### This week
1. **A1**: `hidden2_dim=128` — change constant, update GEMV kernel, new config, retrain
2. **A6**: Lazy qsearch — pure Rust, benchmark NPS before/after

### Short-term (2-4 weeks)
3. **A4**: Second hidden layer (fc3: 64→16) — retrain from current checkpoint
4. **A2**: Add `result` field infrastructure to pipeline for next data run
5. Begin downloading additional Lichess months for expanded dataset

### Medium-term (1-2 months)
6. **A3**: King buckets 16→32 — regenerate `.npy`, retrain from scratch on expanded dataset
7. **A7**: Accumulator refresh cache — implement and benchmark
8. Distillation infrastructure: train teacher on full expanded dataset

### Long-term (3+ months)
9. **A8**: HalfKAv2 full feature set — requires all three: new encoding in Python/Rust, new dataset
10. Full teacher→student distillation with self-play loop

---

## Critical File Reference

| File | Relevant To |
|---|---|
| `chess_evaluation/src/neural_eval.rs` | HIDDEN2, HALFKP_FEATURE_DIM, GEMV kernel, NeuralEvaluator loading |
| `chess_evaluation/src/alpha_beta.rs` | Lazy qsearch, accumulator refresh cache, acc_push/recompute |
| `nn_training/nnue_train/model.py` | EvalNetDual: fc3, skip connections, hidden dims |
| `nn_training/nnue_train/features.py` | King bucket table — must stay identical to Rust KING_BUCKET |
| `nn_training/scripts/export_weights.py` | NPZ schema — any new layer needs a new key here |
| `nn_training/scripts/generate_data.py` | Game result extraction, adaptive depth |
| `nn_training/scripts/preprocess_dataset.py` | .result.npy column, new feature encoding flags |

> **Important**: `features.py` king bucket formula and `neural_eval.rs` KING_BUCKET
> table must always be byte-for-byte equivalent. Any bucket count change must be
> coordinated in both simultaneously, and all `.npy` files regenerated before training.

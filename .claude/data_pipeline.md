# Training Data Pipeline — Working Knowledge

Read this before writing any data generation code, touching dataset loaders, modifying CP conventions, or running preprocessing. The CP perspective convention is the most dangerous silent failure in this codebase — wrong perspective trains the model on inverted evaluations with no error or warning.

---

## The full pipeline

```
PGN / self-play games
        │
        ▼
  generate_data.py          ← FENs extracted, Stockfish labels at depth 14
        │
        ▼ JSONL (side-to-move CP)
  selfplay_pool.jsonl        ← FIFO pool, 750k positions, newest at end
        │
        ▼ split + inject 50k anchor
  data/train.jsonl           ← shuffled, ~675k positions + 50k anchor
  data/val.jsonl             ← ~75k positions (self-play only)
  data/gen_val.jsonl         ← fixed 10k real-game positions, never changes
        │
  [optional: preprocess_dataset.py]   ← converts to binary .npy for 10-20× faster loading
        │
        ▼
     train.py                ← fine-tunes model, logs val_cp_mae
        │
        ▼
  checkpoint_iterN.pt
```

---

## JSONL format and CP perspective

Every record: `{"fen": "...", "cp": N}`

**`cp` is side-to-move centipawns** — positive means the side to move is winning.

This is what `generate_data.py` writes:
```python
score = info["score"].pov(board.turn)   # python-chess: score from mover's POV
cp = float(score.score(mate_score=10000))
return {"fen": fen, "cp": cp}
```

**The model expects white-absolute CP** (positive = white winning). The conversion happens in `JsonlDualPositionDataset.__getitem__`:
```python
sign = 1.0 if board.turn == chess.WHITE else -1.0
cp = cp_raw * sign   # side-to-move → white-absolute
```

**Rule:** If you write a new data source, store side-to-move CP in the JSONL file. The loader handles the conversion. Do NOT pre-convert to white-absolute in JSONL — the loader would then double-convert and silently invert half the labels.

This was a past bug (`d3ec94c: fix: convert side-to-move CP to white-absolute in JsonlDualPositionDataset`). The test for it is the gen_val MAE: if it spikes, this is the first thing to check.

---

## Binary dataset format (produced by `preprocess_dataset.py`)

Used for 10–20× faster DataLoader throughput (memmap vs JSON parse + python-chess per sample). Training falls back to JSONL automatically if the `.npy` files don't exist.

**Dual-perspective files** (what we use — `--dual` flag):

| File | Shape | Dtype | Contents |
|------|-------|-------|----------|
| `{prefix}.white_indices.npy` | (N, 32) | uint16 | Active HalfKP feature indices, white POV |
| `{prefix}.black_indices.npy` | (N, 32) | uint16 | Active HalfKP feature indices, black POV |
| `{prefix}.counts.npy` | (N,) | uint8 | Total pieces on board (used for output bucket selection) |
| `{prefix}.cp.npy` | (N,) | float32 | **White-absolute CP** (already converted from side-to-move) |
| `{prefix}.piece_count.npy` | (N,) | uint8 | Same as counts |

**Single-perspective files** (legacy, `--no-halfkp`):

| File | Shape | Dtype | Contents |
|------|-------|-------|----------|
| `{prefix}.indices.npy` | (N, 32) | uint16 | Active feature indices, side-to-move POV |
| `{prefix}.counts.npy` | (N,) | uint8 | Active feature count |
| `{prefix}.cp.npy` | (N,) | float32 | Side-to-move CP (NOT converted) |
| `{prefix}.piece_count.npy` | (N,) | uint8 | Total pieces |

**Note:** Binary `.cp.npy` in dual mode stores white-absolute (conversion done at preprocessing time). Single-perspective `.cp.npy` stores side-to-move. Do not mix loaders and binary files from different modes.

**Sentinel/padding:** Unused slots in the 32-element index arrays are padded with `HALFKP_FEATURE_DIM` (24,576) — a sentinel value that `EmbeddingBag` ignores because it's out of range.

---

## `max_cp_abs` — clipping behaviour

Controls gradient focus: positions with large advantages (|cp| > threshold) are clipped so the model doesn't waste capacity fitting near-certain outcomes.

- **Default in dataset.py**: 1500 cp (if not overridden)
- **Actual value used in training** (`configs/finetune.yaml`): **800 cp**
- Applied at training time via `np.clip(cp_raw, -max_cp_abs, max_cp_abs)`, not at preprocessing
- `cp_raw` (unclipped) is always stored alongside `cp` (clipped) in memory
- **WDL targets always use `cp_raw`** — the full, unclipped value. Clipping the WDL targets would distort the win/draw/loss distribution. Clipping only affects the regression (CP) loss.

---

## WDL target derivation

`cp_to_wdl_target(cp)` converts a white-absolute CP value to a (win, draw, loss) probability triple used as soft targets for the WDL head:

```python
p_win  = 1 / (1 + exp(-cp / 180))       # logistic, temperature 180
p_loss = 1 - p_win
draw   = max(0, 1 - abs(cp) / 800)      # triangle: peak 1.0 at cp=0, zero at ±800
non_draw = 1 - draw
p_win  *= non_draw                        # renormalise win/loss around draw weight
p_loss *= non_draw
# then normalise so p_win + draw + p_loss = 1
```

- Temperature 180: roughly matches Stockfish's WDL model
- Draw triangle: positions beyond ±800 cp get draw probability 0; equal positions have max draw probability
- `cp_raw` is passed (unclipped); positions beyond ±800 cp get `draw=0` automatically

---

## HalfKP feature encoding

**Input dimension**: 12 × 64 × 32 = **24,576** per perspective.

**Feature index formula:**
```
idx = slot * 2048 + mapped_sq * 32 + king_bucket
```

- `slot`: piece type × colour (0=white pawn, 1=white knight, …, 5=white king, 6=black pawn, …, 11=black king)
- `mapped_sq`: 0–63; rank-flipped for black POV (`sq ^ 56`)
- `king_bucket`: 0–31; derived from king square with horizontal mirror (files e–h mirror to d–a)

**King bucket formula:**
```python
file_bucket = file if file <= 3 else 7 - file   # mirror right half
rank_quarter = rank // 2                          # 4 rank groups of 2
king_bucket = rank_quarter * 4 + file_bucket      # 0..31
```

**Dual perspective:** White and black each get their own encoding from their king's reference frame. No side-to-move normalisation — both perspectives are always computed. This encoding in Python (`dataset.py`) is identical to the Rust implementation in `neural_eval.rs`. If they diverge, training and inference will silently use different feature spaces.

Max 32 active features per perspective (one per piece, padding with sentinel for unused slots).

---

## Dataset class hierarchy

```
JsonlDualPositionDataset        ← JSONL source, converts CP on-the-fly (slow but always works)
BinaryDualPositionDataset       ← pre-processed .npy files, 10-20× faster
GPUPreloadedDualDataset         ← entire dataset in GPU VRAM, fastest possible (needs big VRAM)
```

Training uses whichever is available, in priority order (binary > JSONL). `load_dataset()` in `train.py` auto-detects based on whether the `.npy` files exist next to the `.jsonl` path.

---

## When to run `preprocess_dataset.py`

Run it when:
- Training on a large static dataset (the 69M anchor data) — 10–20× speedup worth it
- Doing many training runs on the same dataset

Don't bother for:
- The self-play loop — `train.jsonl` changes every iteration, preprocessing would add overhead with no benefit

**Command:**
```bash
cd nn_training
python3 scripts/preprocess_dataset.py \
  --input data/train.jsonl \
  --output data/train \
  --dual
```
Produces `data/train.white_indices.npy`, `data/train.black_indices.npy`, etc. alongside the original `.jsonl`.

**Note:** `--max-cp-abs` is deprecated in `preprocess_dataset.py` — clipping is handled at training time, not preprocessing.

---

## Gotchas

**CP perspective is the silent killer.**
Wrong perspective = model trained on inverted evaluations. No error is raised. The model will appear to train (loss decreases) but will play like it's trying to lose. The `gen_val_mae` check in the self-play loop is the best early detector — it will spike immediately if CP perspective is wrong.

**JSONL stores side-to-move; binary .cp.npy stores white-absolute.**
These are not interchangeable. If you mix a JSONL-produced file with a binary loader that expects white-absolute (or vice versa), evaluations for black-to-move positions will be silently inverted.

**WDL targets must use unclipped CP.**
Clipping `cp_raw` before computing WDL targets compresses the distribution and makes the model think most positions are near-equal. Always pass the raw, unclipped CP to `cp_to_wdl_target()`.

**`preprocess_dataset.py` does the CP conversion; `JsonlDualPositionDataset` also does it.**
Don't run preprocessing and then use `JsonlDualPositionDataset` on the same data — the conversion would happen twice, inverting black-to-move positions. Use `BinaryDualPositionDataset` with preprocessed files, or `JsonlDualPositionDataset` with raw JSONL. Never both.

**Pool keeps newest positions** (`lines[-pool_size:]`). Old self-play data is evicted. The anchor data (injected separately each iteration) is never in the pool — it bypasses the FIFO entirely.

**val.jsonl must not contain anchor data.** Anchor positions are injected into `train.jsonl` only. If anchor positions appear in `val.jsonl`, the validation signal is contaminated with out-of-distribution data that makes val_cp_mae look artificially better than it is.

# Adding More Training Data

## Current state

| Location | Shards | Positions | Format | Notes |
|---|---|---|---|---|
| `data/lichess_hf/shards/` | 17 | 253M | int16 indices | All shards from `Lichess/chess-position-evaluations` HF dataset |
| `data/lichess_hf/val_lichess.*` | — | 2M | int32 indices | Carved from shard_16, used for validation |

---

## The int16 situation

Shards downloaded before the int32 fix are stored with **int16 index dtype**.
This is handled transparently by `GPUPreloadedDualDataset._load_indices`:

```python
if arr.dtype == np.int16:
    return arr.view(np.uint16).astype(np.int32)   # reinterpret bits, don't sign-extend
return arr.astype(np.int32)
```

**You do not need to convert existing int16 shards.** They work correctly as-is.

**New shards** downloaded with the current `download_lichess_hf.py` are saved as int32 already. No special handling needed.

**Danger zone — manual file operations:** If you ever copy, slice, or numpy-save index arrays that are int16, use the view trick, NOT `.astype(np.int32)` directly:

```python
# WRONG — sign-extends negative int16 values, corrupts indices > 32767
fixed = arr.astype(np.int32)

# CORRECT — reinterprets bits as unsigned before widening
fixed = arr.view(np.uint16).astype(np.int32)
```

If you end up with an int32 file that has negative values (min < 0), fix it with:
```python
arr = np.load("shard_XX.white_indices.npy")
fixed = arr.astype(np.int16).view(np.uint16).astype(np.int32)
np.save("shard_XX.white_indices.npy", fixed)
# repeat for black_indices.npy
```

---

## Option A: Download more Lichess HF shards

The HF dataset `Lichess/chess-position-evaluations` has exactly 17 shards — you already have all of them.

To check for new releases:
```bash
cd nn_training
.venv/bin/python3 scripts/download_lichess_hf.py --help
```

If a newer version of the dataset is released with additional shards, download them into the same directory:
```bash
.venv/bin/python3 scripts/download_lichess_hf.py \
  --output-dir data/lichess_hf/shards \
  --workers 8
```

The script skips already-downloaded shards (checks for `.cp.npy` marker file).

---

## Option B: Add a different HF dataset

Any dataset that provides `(FEN, depth≥18 centipawn evaluation)` pairs can be preprocessed:

```bash
# 1. Convert JSONL → binary (dual HalfKAv2 format)
.venv/bin/python3 scripts/preprocess_dataset.py \
  --input  data/new_source/positions.jsonl \
  --output data/new_source/train \
  --dual --halfkav2

# 2. If the result is a single large file (>20M positions), split it
.venv/bin/python3 scripts/split_shard.py \
  --input  data/new_source/train \
  --outdir data/new_source/shards \
  --n-shards 5        # aim for ~13–15M positions per shard
```

JSONL format expected: `{"fen": "...", "cp": N}` where `cp` is **side-to-move centipawns**.
Do not pre-convert to white-absolute — `preprocess_dataset.py` handles that.

---

## Option C: Split an existing large shard

If you have a single large binary (e.g. the old 69M `train_all_69m.*`):

```bash
.venv/bin/python3 scripts/split_shard.py \
  --input  data/all_69m/train_all_69m \
  --outdir data/all_69m/shards \
  --n-shards 5
```

The split script uses mmap reads (~1.7 GB RAM per chunk regardless of source size).

---

## Updating the config after adding shards

In `configs/halfkp_1024_256_lichess_sharded.yaml`:

```yaml
data:
  train_shards:
    - data/lichess_hf/shards          # existing 17 shards
    - data/new_source/shards          # new shards directory
```

Then recalculate epochs:
- Count total shards: `ls data/lichess_hf/shards/*.cp.npy data/new_source/shards/*.cp.npy | wc -l`
- Set `epochs = total_shards × 6` (6 full passes — sweet spot before memorisation)
- Set `warmup_epochs = total_shards` (1 full pass warmup)

Example with 17 + 5 = 22 shards:
```yaml
training:
  epochs: 132        # 22 × 6
  warmup_epochs: 22
```

---

## Carving a new val set

Val should always come from the **same distribution** as training. If you add a new source, optionally carve a small slice for val:

```python
import numpy as np

SRC = "data/new_source/shards/shard_04"   # last shard of new data
VAL_OUT = "data/lichess_hf/val_lichess"   # overwrite existing val
VAL_N = 2_000_000

exts = [(".white_indices.npy", np.int32), (".black_indices.npy", np.int32),
        (".cp.npy", np.float32), (".counts.npy", np.uint8), (".piece_count.npy", np.uint8)]

for ext, dtype in exts:
    arr = np.load(SRC + ext, mmap_mode="r")
    np.save(VAL_OUT + ext, arr[:VAL_N].astype(dtype))
    np.save(SRC + ext, arr[VAL_N:].astype(dtype))
```

**If carving from an int16 shard**, use the safe conversion to avoid re-introducing the sign-extension bug:
```python
arr = np.load(SRC + ".white_indices.npy", mmap_mode="r")
if arr.dtype == np.int16:
    chunk = arr[:VAL_N].view(np.uint16).astype(np.int32)   # safe
else:
    chunk = arr[:VAL_N].astype(np.int32)
np.save(VAL_OUT + ".white_indices.npy", chunk)
```

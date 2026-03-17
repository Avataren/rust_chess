# NN Training Pipeline

End-to-end pipeline for training a chess evaluation network used by the Rust engine.

## Architecture

Current: **Dual-perspective HalfKP NNUE** (Phase 4)

- Input: 12,288 (`12 pieces × 64 squares × 16 king buckets`) sparse features, evaluated from **both** white and black's perspective simultaneously
- Hidden: 512+512 (two shared accumulators, one per perspective) → concatenated 1024 → 32
- Outputs: centipawn scalar (white-absolute) + 3-way WDL logits
- Weights exported as int16-quantized `.npz` for Rust inference

**Rust inference path:**
- Accumulators maintained incrementally during search (only ~2 column updates per move)
- Column add/subtract dispatched via AVX2 SIMD (native) or simd128 (WASM)
- Accumulator stored as raw i16; dequantized to f32 only once per eval at the ReLU boundary
- Full forward pass used as fallback for scratch evaluation and single-perspective models

Legacy: 768-dim (`12 × 64`) absolute features and single-perspective HalfKP — still supported via auto-detection in the Rust loader.

## Folder layout

```
configs/          training configs
data/             JSONL datasets, raw FEN files, and binary encoded datasets
artifacts/        checkpoints (.pt) and exported weights (.npz)
runs/             TensorBoard logs
scripts/          all pipeline scripts
nnue_train/       Python package (model, dataset, features)
```

## Setup

```bash
cd nn_training
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For ROCm/CUDA, install the appropriate PyTorch build first.

---

## Quick start — one command

```bash
# Full pipeline: PGN → FENs → label → split → encode (single + dual)
PYTHONPATH=. python3 scripts/make_dataset.py \
  --positions 20_000_000 \
  --pgn /data/lichess_db_standard_rated_2024-01.pgn \
  --depth 14 \
  --workers 32

# Re-label an existing FENs file at a different depth
PYTHONPATH=. python3 scripts/make_dataset.py \
  --positions 20_000_000 \
  --fens data/fens_20m.txt \
  --depth 14 \
  --workers 32
```

Key options: `--positions N`, `--depth D`, `--workers W`, `--pgn / --fens`,
`--min-elo`, `--no-dual`, `--skip-extract/label/split/preprocess`.

---

## Step 0 — Download and extract Lichess data

Lichess publishes monthly PGN dumps of all rated standard games at:
https://database.lichess.org/

Files are `.pgn.zst` (Zstandard compressed). Pick a month — larger/newer months have more games.
January 2021 (~31GB compressed, ~219GB uncompressed) is a good balance of size and diversity.

```bash
# Download a monthly dump (example: January 2021)
wget https://database.lichess.org/standard/lichess_db_standard_rated_2021-01.pgn.zst

# Decompress (requires zstd)
# Install: sudo pacman -S zstd  (Arch) / sudo apt install zstd  (Ubuntu)
zstd -d lichess_db_standard_rated_2021-01.pgn.zst
# Output: lichess_db_standard_rated_2021-01.pgn (~219GB)
```

You can also stream directly into pgn_extract without fully decompressing:
```bash
zstdcat lichess_db_standard_rated_2021-01.pgn.zst | ./target/release/pgn_extract \
  --input /dev/stdin \
  --output data/fens_20m.txt \
  --max-positions 20000000
```

> Tip: newer months have more games and higher average ELO. For maximum diversity,
> combine multiple months or use `--min-elo 0` to include all skill levels.

---

## Step 1 — Extract positions from PGN (Rust, fast)

The Rust `pgn_extract` binary is ~110x faster than Python for PGN scanning.

```bash
# Build once
cd /path/to/rust_chess
cargo build --release -p pgn_extract

# Extract 20M positions across all ELO levels
./target/release/pgn_extract \
  --input nn_training/lichess_db_standard_rated_2021-01.pgn \
  --output nn_training/data/fens_20m.txt \
  --max-positions 20000000 \
  --min-elo 0
```

Options: `--min-elo`, `--max-elo`, `--min-ply`, `--positions-per-game`

**Speed**: ~27,000 pos/sec. 20M positions in ~12 minutes.

---

## Step 2 — Label positions with Stockfish (parallel)

```bash
cd nn_training

# Label pre-extracted FENs (recommended — use Rust extractor first)
PYTHONPATH=. python3 scripts/generate_data.py \
  --label-engine /usr/bin/stockfish \
  --fens data/fens_20m.txt \
  --output data/d14_20m.jsonl \
  --eval-depth 14 \
  --workers 32 \
  --selfplay-games 0 \
  --max-positions 20000000
```

`--workers 32` spawns 32 parallel Stockfish instances. Each owns its own process.

Depth guide:
- `--eval-depth 8`:  ~80 pos/sec/worker,  ~70 min for 10M at 32 workers — fast, good baseline
- `--eval-depth 12`: ~20 pos/sec/worker,  ~4.5 hrs for 10M at 32 workers — better quality
- `--eval-depth 14`: ~7 pos/sec/worker,  ~12 hrs for 20M at 32 workers — recommended for production
- `--eval-depth 16`: ~4 pos/sec/worker,  marginal gain over 14, diminishing returns

### Building up a larger dataset incrementally

20M positions is a reasonable first run but on the small side for the architecture
(~6.3M parameters). 100M is a more comfortable target. Run successive 20M batches
from the **same PGN** using `--skip-games` to avoid re-sampling the same games:

```bash
# Batch 1 — games 0–20M (default)
PYTHONPATH=. python3 scripts/generate_data.py \
  --pgn data/lichess_db_standard_rated_2021-01.pgn \
  --output data/d14_20m_batch1.jsonl \
  --eval-depth 14 --workers 32 --max-positions 20000000

# Batch 2 — games 20M–40M
PYTHONPATH=. python3 scripts/generate_data.py \
  --pgn data/lichess_db_standard_rated_2021-01.pgn \
  --skip-games 20000000 \
  --output data/d14_20m_batch2.jsonl \
  --eval-depth 14 --workers 32 --max-positions 20000000

# Batch 3 — games 40M–60M
PYTHONPATH=. python3 scripts/generate_data.py \
  --pgn data/lichess_db_standard_rated_2021-01.pgn \
  --skip-games 40000000 \
  ...
```

`--skip-games N` counts **qualifying games** (those passing the Elo filter), so each
batch is guaranteed non-overlapping regardless of how many games are filtered out.

The January 2021 lichess dump (~220GB uncompressed) contains hundreds of millions of
games — enough for 5+ non-overlapping batches of 20M.

**Do not mix datasets labeled at different depths** (e.g. depth-8 and depth-14).
The label quality mismatch acts as noise and generally hurts training more than
the extra data helps. Re-label old FEN files at depth-14 instead.

**Retrain after each batch** to track improvement before committing to the full run.
Expect the biggest jump from batch 1→2; returns diminish after ~60–80M positions
for this architecture.

---

## Step 3 — Split dataset

```bash
# Shuffle and split with 1% held out for validation (200k val for 20M total)
PYTHONPATH=. python3 scripts/split_dataset.py \
  --input    data/d14_20m.jsonl \
  --train    data/train_20m.jsonl \
  --val      data/val_20m.jsonl \
  --val-ratio 0.01
```

> Note: `split_dataset.py` loads the full file into RAM for shuffling.
> A 1.7GB JSONL file requires ~4GB RAM during the shuffle step — this is fine but takes 1-2 minutes.

---

## Step 4 — Pre-encode to binary (required for dual; strongly recommended)

Convert JSONL to sparse binary format for ~10-20x faster DataLoader throughput.
Instead of parsing JSON + running python-chess per sample at training time, workers just do a
memmap read + scatter of ~32 indices. This keeps the GPU near 100% utilization.

### Dual-perspective (recommended — required for the incremental search path)

```bash
# Run train and val in parallel
PYTHONPATH=. python3 scripts/preprocess_dataset.py \
  --input data/train_20m.jsonl \
  --output data/train_20m \
  --dual > /tmp/preprocess_train.log 2>&1 &

PYTHONPATH=. python3 scripts/preprocess_dataset.py \
  --input data/val_20m.jsonl \
  --output data/val_20m \
  --dual > /tmp/preprocess_val.log 2>&1 &

wait && echo "done"
```

Output files (dual, 20M samples):
```
data/train_20m.white_indices.npy  ~1.2 GB   (N, 32) uint16 white-perspective feature indices
data/train_20m.black_indices.npy  ~1.2 GB   (N, 32) uint16 black-perspective feature indices
data/train_20m.counts.npy         ~20 MB    (N,)    uint8  active feature count
data/train_20m.cp.npy             ~80 MB    (N,)    float32 centipawn (white-absolute)
```

CP values are automatically converted to white-absolute convention during preprocessing
(positions where black is to move have their CP sign flipped).

### Single-perspective (legacy)

```bash
PYTHONPATH=. python3 scripts/preprocess_dataset.py \
  --input data/train_20m.jsonl \
  --output data/train_20m \
  --use-halfkp
```

Output files:
```
data/train_20m.indices.npy   ~1.2 GB   (N, 32) uint16 active feature indices
data/train_20m.counts.npy    ~20 MB    (N,)    uint8  active feature count
data/train_20m.cp.npy        ~80 MB    (N,)    float32 centipawn values
```

The training script auto-detects binary files: if `train_20m.white_indices.npy` (dual) or
`train_20m.indices.npy` (single) exists alongside the JSONL, the fast binary dataset is used
automatically. No config change needed.

> Note: dense float32 storage would be ~970GB for 12,288-dim HalfKP features at 20M positions.
> Sparse indices reduce this to ~2.5GB total — feasible with mmap.

---

## Step 5 — Train

### Dual-perspective (recommended)

```bash
PYTHONPATH=. python3 scripts/train.py \
  --config configs/halfkp_dual_20m.yaml \
  --out    artifacts/checkpoint_dual_20m.pt \
  --tb-logdir runs/dual_20m
```

### Single-perspective (legacy)

```bash
PYTHONPATH=. python3 scripts/train.py \
  --config configs/halfkp_10m.yaml \
  --out    artifacts/checkpoint_halfkp_10m.pt \
  --tb-logdir runs/halfkp_10m
```

### Resume / fine-tune from existing checkpoint

```bash
PYTHONPATH=. python3 scripts/train.py \
  --config configs/halfkp_dual_20m.yaml \
  --out    artifacts/checkpoint_dual_20m_ft.pt \
  --resume artifacts/checkpoint_dual_20m.pt \
  --tb-logdir runs/dual_20m_ft
```

Watch training live:
```bash
tensorboard --logdir runs
```

Key metrics to watch:
- `val/cp_mae`: main quality metric (centipawn MAE on validation set). Baseline: 132 cp (10M single-perspective)
- Train/val gap: large gap = overfitting → increase dropout or add more data
- Best checkpoint saved automatically when `val/loss` improves

---

## Step 6 — Export weights

```bash
# Dual-perspective (backbone_3_weight will be 32×1024)
PYTHONPATH=. python3 scripts/export_weights.py \
  --checkpoint artifacts/checkpoint_dual_20m.pt \
  --config     configs/halfkp_dual_20m.yaml \
  --output     artifacts/eval_halfkp_dual_20m.npz \
  --scale      256.0

# Single-perspective (backbone_3_weight will be 32×512)
PYTHONPATH=. python3 scripts/export_weights.py \
  --checkpoint artifacts/checkpoint_halfkp_10m.pt \
  --config     configs/halfkp_10m.yaml \
  --output     artifacts/eval_halfkp_10m.npz
```

---

## Step 7 — Deploy and verify

```bash
cp artifacts/eval_halfkp_dual_20m.npz ../chess_evaluation/src/eval.npz
cargo build --release -p chess_uci

# Run all engine tests
cargo test -p chess_evaluation

# Verify i16 incremental path is numerically correct (dual model only)
cargo test -p chess_evaluation -- --include-ignored test_i16_accum_equivalence
```

The equivalence test confirms the i16 SIMD accumulator path produces scores within ±2cp of
the f32 scratch path on a sample of positions. It requires a dual-perspective model to run.

---

## Configs

| Config | Input dim | Dataset | Notes |
|--------|-----------|---------|-------|
| `default.yaml` | 768 | `data/train.jsonl` | Legacy absolute features |
| `halfkp.yaml` | 12,288 | `data/train_1m.jsonl` | HalfKP single-perspective, 1M positions |
| `halfkp_10m.yaml` | 12,288 | `data/train_10m.jsonl` | HalfKP single-perspective, 10M, dropout 0.3 |
| `halfkp_dual.yaml` | 12,288 | `data/train_10m.jsonl` | Dual-perspective, 10M |
| `halfkp_dual_20m.yaml` | 12,288 | `data/train_20m.jsonl` | **Current** — dual-perspective, 20M, depth-14 labels |

---

## Self-play training loop

Continuously generates data via self-play, labels it, fine-tunes, and promotes the model if it improves:

```bash
PYTHONPATH=. python3 scripts/selfplay_loop.py \
  --engine target/release/chess_uci \
  --stockfish /usr/bin/stockfish \
  --base-checkpoint artifacts/checkpoint_halfkp.pt \
  --config configs/halfkp.yaml
```

---

## Benchmarking the engine

```bash
# NPS benchmark (depth 12, 1 thread) — do not run while labeling
cargo run --release -p chess_evaluation --bin bench -- --depth 12 --threads 1

# Neural vs classical self-play, 100 games
./target/release/self_play \
  --engine1 ./target/release/chess_uci \
  --engine2 ./target/release/chess_uci \
  --engine1-opt EvalFile=chess_evaluation/src/eval.npz \
  --engine2-opt UseNN=false \
  --games 100 \
  --threads 8
```

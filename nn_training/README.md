# NN Training Pipeline

End-to-end pipeline for training a chess evaluation network used by the Rust engine.

## Architecture

Current: **HalfKP MLP** (Phase 1 of NNUE-style upgrade)

- Input: 12,288 (`12 pieces × 64 squares × 16 king buckets`) sparse features
- Hidden: 512 → 32
- Outputs: centipawn scalar + 3-way WDL logits
- Weights exported as int16-quantized `.npz` for Rust inference

Legacy: 768-dim (`12 × 64`) absolute piece-square features — still supported via auto-detection in the Rust loader.

## Folder layout

```
configs/          training configs (default.yaml, halfkp.yaml, halfkp_10m.yaml)
data/             JSONL datasets and raw FEN files
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
  --output data/fens_10m.txt \
  --max-positions 10000000
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

# Extract 10M positions across all ELO levels
./target/release/pgn_extract \
  --input nn_training/lichess_db_standard_rated_2021-01.pgn \
  --output nn_training/data/fens_10m.txt \
  --max-positions 10000000 \
  --min-elo 0

# ELO-filtered example (high ELO only)
./target/release/pgn_extract \
  --input nn_training/lichess_db_standard_rated_2021-01.pgn \
  --output nn_training/data/fens_highelo.txt \
  --max-positions 1000000 \
  --min-elo 2200
```

Options: `--min-elo`, `--max-elo`, `--min-ply`, `--positions-per-game`

**Speed**: ~27,000 pos/sec. 10M positions in ~6 minutes.

---

## Step 2 — Label positions with Stockfish (parallel)

```bash
cd nn_training

# Label pre-extracted FENs (recommended — use Rust extractor first)
PYTHONPATH=. python3 scripts/generate_data.py \
  --label-engine /usr/bin/stockfish \
  --fens data/fens_10m.txt \
  --output data/large_multielo.jsonl \
  --eval-depth 8 \
  --workers 32 \
  --selfplay-games 0 \
  --max-positions 10000000

# Or label directly from PGN (slower scan, but single command)
PYTHONPATH=. python3 scripts/generate_data.py \
  --label-engine /usr/bin/stockfish \
  --pgn ../lichess_db_standard_rated_2021-01.pgn \
  --output data/all.jsonl \
  --min-elo 2200 \
  --eval-depth 8 \
  --workers 32 \
  --selfplay-games 0 \
  --max-positions 1000000
```

`--workers 32` spawns 32 parallel Stockfish instances. Each owns its own process.
**Speed**: ~2,500 pos/sec with 32 workers at depth 8. 10M positions in ~60-70 minutes.

Depth guide:
- `--eval-depth 8`: fast, good quality (~80 pos/sec/worker)
- `--eval-depth 12`: slower, marginal quality gain (~20 pos/sec/worker)

---

## Step 2b — Pre-encode to binary (recommended for large datasets)

After labeling, convert JSONL to sparse binary format for ~10-20x faster DataLoader throughput.
Instead of parsing JSON + running python-chess per sample at training time, workers just do a
memmap read + scatter of ~32 indices. This keeps the GPU near 100% utilization.

```bash
# Run train and val in parallel
PYTHONPATH=. python3 scripts/preprocess_dataset.py \
  --input data/train_10m.jsonl \
  --output data/train_10m \
  --use-halfkp > /tmp/preprocess_train.log 2>&1 &

PYTHONPATH=. python3 scripts/preprocess_dataset.py \
  --input data/val_10m.jsonl \
  --output data/val_10m \
  --use-halfkp > /tmp/preprocess_val.log 2>&1 &

wait && echo "done"
```

Output files (example for 10M samples):
```
data/train_10m.indices.npy   ~633 MB   (N, 32) uint16 active feature indices
data/train_10m.counts.npy    ~10 MB    (N,)    uint8  active feature count
data/train_10m.cp.npy        ~40 MB    (N,)    float32 centipawn values
```

The training script auto-detects binary files: if `train_10m.indices.npy` exists alongside
`train_10m.jsonl`, `BinaryPositionDataset` is used automatically. No config change needed.

> Note: dense float32 storage would be ~485GB for 12,288-dim HalfKP features.
> Sparse indices reduce this to ~683MB total — feasible with mmap.

---

## Step 3 — Split dataset

```bash
# Shuffle and split 90/10 train/val
shuf data/large_multielo.jsonl | split -l 9000000 - split_tmp_
mv split_tmp_aa data/train_10m.jsonl
mv split_tmp_ab data/val_10m.jsonl
```

---

## Step 4 — Train

```bash
# HalfKP model on 10M dataset
PYTHONPATH=. python3 scripts/train.py \
  --config configs/halfkp_10m.yaml \
  --out artifacts/checkpoint_halfkp_10m.pt \
  --tb-logdir runs/halfkp_10m

# Resume / fine-tune from existing checkpoint
PYTHONPATH=. python3 scripts/train.py \
  --config configs/halfkp_10m.yaml \
  --out artifacts/checkpoint_finetuned.pt \
  --resume artifacts/checkpoint_halfkp_10m.pt \
  --tb-logdir runs/finetuned
```

Watch training live:
```bash
tensorboard --logdir runs
```

Key metrics to watch:
- `val/cp_mae`: main quality metric (centipawn error on validation set)
- Train/val gap: large gap = overfitting → need more data or higher dropout
- Best checkpoint is saved automatically when `val/loss` improves

---

## Step 5 — Export weights

```bash
PYTHONPATH=. python3 scripts/export_weights.py \
  --checkpoint artifacts/checkpoint_halfkp_10m.pt \
  --config configs/halfkp_10m.yaml \
  --output artifacts/eval_halfkp_10m.npz
```

To deploy to the engine, copy to `chess_evaluation/src/`:
```bash
cp artifacts/eval_halfkp_10m.npz ../chess_evaluation/src/eval.npz
cargo build --release -p chess_uci
```

---

## Configs

| Config | Input dim | Dataset | Notes |
|--------|-----------|---------|-------|
| `default.yaml` | 768 | `data/train.jsonl` | Legacy absolute features |
| `halfkp.yaml` | 12,288 | `data/train_1m.jsonl` | HalfKP, 1M positions |
| `halfkp_10m.yaml` | 12,288 | `data/train_10m.jsonl` | HalfKP, 10M positions, dropout 0.3 |

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
# Neural vs classical, 8 threads, 100 games
./target/release/self_play \
  --engine1 ./target/release/chess_uci \
  --engine2 ./target/release/chess_uci \
  --engine1-opt EvalFile=chess_evaluation/src/eval.npz \
  --engine2-opt UseNN=false \
  --games 100 \
  --threads 8

# NPS benchmark on Sicilian position
(echo "position fen rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
 echo "go movetime 5000"
 sleep 7
 echo "quit") | ./target/release/chess_uci
```

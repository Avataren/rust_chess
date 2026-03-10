# NN Training Project (ROCm + Stockfish + Self-Play)

This subproject adds an end-to-end training pipeline for a chess evaluation network that can later be used by the Rust engine and exported for lightweight WASM inference.

## What is implemented

- Position generation from:
  - Stockfish self-play
  - Optional PGN sampling
- Labeling with Stockfish centipawn evaluations
- Dataset split utility
- PyTorch training script (ROCm/CUDA autodetect)
- Dual-head model:
  - `cp_head`: centipawn regression
  - `wdl_head`: W/D/L classification
- Quantized weight export (`.npz`) for easy Rust-side inference integration

## Folder layout

- `configs/default.yaml` - training/model/data config
- `requirements.txt` - Python deps
- `nnue_train/features.py` - 12x64 side-to-move normalized encoder + cp->WDL
- `nnue_train/dataset.py` - JSONL dataset loader
- `nnue_train/model.py` - evaluation model
- `scripts/generate_data.py` - data generation and Stockfish labels
- `scripts/split_dataset.py` - train/val split
- `scripts/train.py` - training loop
- `scripts/export_weights.py` - int16 quantized export

## Architecture choice

Current implementation uses a **small NNUE-like MLP**:

- Input: 768 (`12 x 64`) sparse board occupancy, side-to-move normalized
- Hidden: 512 -> 32
- Outputs:
  - 1 scalar cp value
  - 3-way WDL logits

This architecture is intentionally compact for eventual WASM integration.

## Setup

```bash
cd nn_training
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For ROCm, install a ROCm-enabled PyTorch build in the same env.

## 1) Generate labeled data

```bash
python scripts/generate_data.py \
  --label-engine /path/to/stockfish \
  --output data/all.jsonl \
  --selfplay-games 50000 \
  --selfplay-movetime-ms 20 \
  --eval-depth 12
```

Use a separate UCI binary for self-play trajectory generation (for example,
your own Rust engine once UCI is implemented):

```bash
python scripts/generate_data.py \
  --label-engine /path/to/stockfish \
  --selfplay-engine /path/to/rust_engine_uci \
  --output data/all.jsonl
```

Optional PGN mix:

```bash
python scripts/generate_data.py \
  --label-engine /path/to/stockfish \
  --pgn /path/to/games.pgn \
  --output data/all.jsonl
```

## 2) Split dataset

```bash
python scripts/split_dataset.py --input data/all.jsonl --train data/train.jsonl --val data/val.jsonl
```

For non-empty inputs, the splitter guarantees at least 1 validation row (and keeps at least 1 training row when possible), so small datasets still work end-to-end.


## 3) Train

```bash
PYTHONPATH=. python scripts/train.py --config configs/default.yaml --out artifacts/checkpoint.pt
```

The training loader uses `drop_last=False`, so small datasets still run (including when `len(train) < batch_size`).


By default validation workers follow training workers, but if `training.workers` is `0`, validation also uses `0` workers (no multiprocessing). You can override validation worker count with optional `training.val_workers` in the config.

The WDL head is trained against the soft WDL target distribution from the dataset (not argmax-hardened labels).

## 4) Export quantized weights

```bash
PYTHONPATH=. python scripts/export_weights.py \
  --checkpoint artifacts/checkpoint.pt \
  --config configs/default.yaml \
  --output artifacts/nnue_like_weights.npz
```

By default export fails fast on int16 overflow after scaling (`--on-overflow error`).
If you prefer saturating behavior, pass `--on-overflow clip`.


## 5) Monitor with TensorBoard

Training writes TensorBoard scalars by default to `runs/nn_training`.

```bash
tensorboard --logdir runs
```

You can override the log directory:

```bash
PYTHONPATH=. python scripts/train.py --config configs/default.yaml --out checkpoint.pt --tb-logdir runs/experiment_01
```

## Next step for Rust integration

Implement a Rust evaluator in `chess_evaluation` that:

1. loads `nnue_like_weights.npz` (or converted binary format),
2. recreates the same 12x64 feature encoding,
3. runs the quantized layers,
4. returns cp score in `evaluate_board` path.

You can initially blend NN and handcrafted eval scores while tuning search.

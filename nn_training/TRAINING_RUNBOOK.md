# NNUE Training Runbook — 1024×256 Base Model

All commands run from `nn_training/` unless otherwise noted.

---

## Quick Reference

| Phase | Command summary | Duration (est.) |
|-------|----------------|-----------------|
| 1 — base training | ph1 config, from scratch | ~8–12 hrs |
| 2 — continue Mar | ph2 config, `--resume` | ~4–6 hrs |
| 3 — continue May | ph3 config, `--resume` | ~4–6 hrs |
| 4 — Lichess calibration | ph4 config, `--resume` | ~4–6 hrs |
| Deploy | export + rebuild engine | ~5 min |
| Benchmark | puzzle bench + self-play | ~30 min |

---

## Environment

Always prefix training commands with this on the RX 7900 XTX (ROCm 7.2 / PyTorch rocm6.2 mismatch):

```bash
export PYTORCH_NO_HIPBLASLT=1
```

Or prepend it inline to each command as shown below.

---

## Phase 1 — Base pre-training (120M, from scratch)

```bash
cd nn_training
PYTORCH_NO_HIPBLASLT=1 PYTHONPATH=. python3 scripts/train.py \
  --config configs/halfkp_1024_256_ph1_t80_apr.yaml \
  --tb-logdir runs/1024x256_ph1_t80_apr
```

- Loads `data/t80_apr2024/merged.*` into GPU (~18 GB VRAM, takes ~30 s)
- 100 epochs, LR 0.001 with cosine annealing
- Best checkpoint saved to `artifacts/checkpoint.pt` automatically

---

## Phase 2 — Continue on test80 March (90M)

```bash
PYTORCH_NO_HIPBLASLT=1 PYTHONPATH=. python3 scripts/train.py \
  --config configs/halfkp_1024_256_ph2_t80_mar.yaml \
  --resume artifacts/checkpoint.pt \
  --tb-logdir runs/1024x256_ph2_t80_mar
```

- Restores **weights only** (optimizer + scheduler restart fresh — `reset_best_val: true`)
- 50 epochs, LR 0.0003

---

## Phase 3 — Continue on test80 May (90M)

```bash
PYTORCH_NO_HIPBLASLT=1 PYTHONPATH=. python3 scripts/train.py \
  --config configs/halfkp_1024_256_ph3_t80_may.yaml \
  --resume artifacts/checkpoint.pt \
  --tb-logdir runs/1024x256_ph3_t80_may
```

- 50 epochs, LR 0.0002

---

## Phase 4 — Lichess calibration (67M, final)

```bash
PYTORCH_NO_HIPBLASLT=1 PYTHONPATH=. python3 scripts/train.py \
  --config configs/halfkp_1024_256_ph4_lichess.yaml \
  --resume artifacts/checkpoint.pt \
  --tb-logdir runs/1024x256_ph4_lichess
```

- 50 epochs, LR 0.0001
- Trains on **67M Lichess + 10M SF anchor** (13% SF ratio prevents opening-bucket forgetting)
- This is where `val_cp_mae` finally becomes a meaningful signal

---

## Resuming a crashed run (same phase)

If a run crashes mid-epoch, resume it **without** `reset_best_val` (omit from config or use `--resume` only):

```bash
PYTORCH_NO_HIPBLASLT=1 PYTHONPATH=. python3 scripts/train.py \
  --config configs/halfkp_1024_256_ph1_t80_apr.yaml \
  --resume artifacts/checkpoint.pt \
  --tb-logdir runs/1024x256_ph1_t80_apr
```

This restores the full training state (optimizer, scheduler, epoch counter).

---

## Monitoring with TensorBoard

```bash
cd nn_training
tensorboard --logdir runs/
```

Then open http://localhost:6006.

**What each metric actually means by phase:**

| Metric | Phase 1–3 (SF data) | Phase 4 (Lichess) |
|--------|---------------------|-------------------|
| `train/cp_mae` | ✅ primary signal — watch this | ✅ primary signal |
| `val/cp_mae` | ❌ misleading — val is Lichess, train is SF | ✅ now meaningful |
| `val/cp_mae[b0]` | ❌ opening bucket diverges fast (SF ≠ human openings) | ✅ should converge |
| `val/loss` | ⚠️ divergence alarm only — panic if it spikes suddenly | ✅ watch normally |
| `train/lr` | ✅ confirms cosine schedule | ✅ |

**Phase 1 expected trajectory:** `train_cp_mae` ~70 → ~55–62 over 100 epochs (≈0.3–0.4 CP/epoch drop).

---

## Deploy: export weights to engine

After any phase completes (or when `artifacts/checkpoint.pt` looks good):

```bash
# 1. Export checkpoint → artifacts/eval.npz (quantised int16)
cd nn_training
PYTHONPATH=. python3 scripts/export_weights.py \
  --config configs/halfkp_1024_256_ph1_t80_apr.yaml \
  --checkpoint artifacts/checkpoint.pt \
  --output artifacts/eval.npz

# 2. Rebuild the engine from the rust_chess root
cd ..
cargo build --release -p chess_uci -p chess_evaluation -p self_play -p puzzle_bench
```

`artifacts/eval.npz` is read live by the lichess bot — no restart needed.

---

## Benchmark: puzzle solve rate

```bash
cd /home/avataren/src/rust_chess
target/release/puzzle_bench \
  --file lichess_db_puzzle.csv.zst \
  --eval-file nn_training/artifacts/eval.npz \
  --count 2000 --depth 7 --seed 42 \
  --min-rating 1500 --threads 0
```

Or use the `/puzzle` skill in Copilot CLI.

---

## Benchmark: NPS / search speed

```bash
cd /home/avataren/src/rust_chess
target/release/bench --depth 12 --threads 1
```

---

## Benchmark: self-play (model vs model)

```bash
cd /home/avataren/src/rust_chess
target/release/self_play \
  --games 150 \
  --candidate nn_training/artifacts/eval.npz \
  --baseline chess_evaluation/src/eval.npz \
  --threads 0
```

> **Note:** 150 games gives SE ≈ 4.1%. A result above 52% is a meaningful improvement.

---

## Get more training data

If you want to download additional test80-2024 months (Jan/Feb/Jun):

```bash
cd nn_training
# Edit datasets/download_t80.py to add months, then:
python3 datasets/download_t80.py

# Convert new binpack to binary (edit BINPACK/OUTPUT/MAX_POSITIONS in script):
bash scripts/binpack_to_binary.sh \
  datasets/test80-2024-XX-xxx-2tb7p.min-v2.v6.binpack.zst \
  data/t80_XXXX \
  120000000
```

---

## Rebuild C++ binpack converter (if needed)

```bash
# Clone nnue-pytorch headers (one-time)
git clone --depth 1 https://github.com/official-stockfish/nnue-pytorch /tmp/nnue-pytorch

# Build
cd /tmp/nnue-pytorch
g++ -O2 -std=c++20 \
  -Idata_loader/cpp/lib -Idata_loader/cpp \
  /home/avataren/src/rust_chess/nn_training/scripts/binpack2jsonl.cpp \
  -o /home/avataren/src/rust_chess/nn_training/scripts/binpack2jsonl \
  -lpthread
```

---

## Architecture constants (Rust)

If you change hidden layer sizes, update `chess_evaluation/src/neural_eval.rs`:

```rust
pub(crate) const HIDDEN1: usize = 1024;  // ← currently 1024
const HIDDEN2: usize = 256;              // ← currently 256
```

Then rebuild: `cargo build --release` from `rust_chess/`.

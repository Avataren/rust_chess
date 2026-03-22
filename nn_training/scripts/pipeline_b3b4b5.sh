#!/usr/bin/env bash
# pipeline_b3b4b5.sh
#
# Full pipeline:
#   1. Export current best checkpoint for lichess
#   2. Decompress May 2020 Lichess PGN
#   3. Extract 5M FENs each for b3 (14-16), b4 (17-20), b5 (21-24) pieces
#   4. Combine + label at depth 14
#   5. Split 90/10 + combine with all_41m
#   6. Binary encode → all_54m
#   7. Create training config
#   8. Update lichess-bot EvalFile
#
# Usage:
#   bash scripts/pipeline_b3b4b5.sh 2>&1 | tee logs/pipeline_b3b4b5.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NN_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$NN_ROOT/.." && pwd)"

PYTHON="$NN_ROOT/.venv/bin/python3"
FINETUNE_EXTRACT="$REPO_ROOT/target/release/finetune_extract"
NNUE_PREPROCESS="$REPO_ROOT/target/release/nnue_preprocess"
STOCKFISH=$(command -v stockfish)
WORKERS=$(nproc)

PGN_ZST="$NN_ROOT/lichess_db_standard_rated_2020-05.pgn.zst"
PGN="$NN_ROOT/lichess_db_standard_rated_2020-05.pgn"
CHECKPOINT="$NN_ROOT/artifacts/checkpoint_all_41m_1024_8b.pt"
NPZ_OUT="$NN_ROOT/artifacts/eval_all_41m_1024_8b.npz"
LICHESS_CFG="$REPO_ROOT/lichess-bot/config.yml"

FENS_DIR="$NN_ROOT/data/new_fens"
OUT_DIR="$NN_ROOT/data/all_54m"

EXISTING_TRAIN="$NN_ROOT/data/all_41m/train_all_41m.jsonl"
EXISTING_VAL="$NN_ROOT/data/all_41m/val_all_41m.jsonl"

POS_PER_BUCKET=5000000
MIN_ELO=1800

mkdir -p "$FENS_DIR" "$OUT_DIR"

echo "════════════════════════════════════════════════════════"
echo "  pipeline_b3b4b5.sh"
echo "  workers: $WORKERS"
echo "  target: ${POS_PER_BUCKET} positions per bucket (b3/b4/b5)"
echo "════════════════════════════════════════════════════════"

# ── Step 1: Decompress PGN ───────────────────────────────────────────────────

echo ""
echo "[1/7] Decompressing $(basename "$PGN_ZST")..."
if [[ -f "$PGN" ]]; then
    echo "  Already decompressed, skipping."
else
    zstd -d "$PGN_ZST" -o "$PGN" --progress
    echo "  → $(du -sh "$PGN" | cut -f1) uncompressed"
fi

# ── Step 3: Extract FENs (sequential to avoid I/O contention) ────────────────

echo ""
echo "[2/7] Extracting FENs from $(basename "$PGN")..."

echo "  b3: 14-16 pieces (late endgame)..."
"$FINETUNE_EXTRACT" \
    --input "$PGN" \
    --output "$FENS_DIR/new_b3.fens" \
    --min-pieces 14 --max-pieces 16 \
    --sample-from-last 70 \
    --min-elo "$MIN_ELO" \
    --positions-per-game 2 \
    --max-positions "$POS_PER_BUCKET"
echo "  → $(wc -l < "$FENS_DIR/new_b3.fens") b3 FENs"

echo "  b4: 17-20 pieces (transition)..."
"$FINETUNE_EXTRACT" \
    --input "$PGN" \
    --output "$FENS_DIR/new_b4.fens" \
    --min-pieces 17 --max-pieces 20 \
    --sample-from-last 50 \
    --min-elo "$MIN_ELO" \
    --positions-per-game 2 \
    --max-positions "$POS_PER_BUCKET"
echo "  → $(wc -l < "$FENS_DIR/new_b4.fens") b4 FENs"

echo "  b5: 21-24 pieces (transition/middlegame)..."
"$FINETUNE_EXTRACT" \
    --input "$PGN" \
    --output "$FENS_DIR/new_b5.fens" \
    --min-pieces 21 --max-pieces 24 \
    --sample-from-last 40 \
    --min-elo "$MIN_ELO" \
    --positions-per-game 2 \
    --max-positions "$POS_PER_BUCKET"
echo "  → $(wc -l < "$FENS_DIR/new_b5.fens") b5 FENs"

# ── Step 4: Label with Stockfish ──────────────────────────────────────────────

echo ""
echo "[3/7] Combining FENs and labeling at depth 14..."
cat "$FENS_DIR/new_b3.fens" "$FENS_DIR/new_b4.fens" "$FENS_DIR/new_b5.fens" \
    | shuf > "$FENS_DIR/new_b3b4b5.fens"
TOTAL_FENS=$(wc -l < "$FENS_DIR/new_b3b4b5.fens")
echo "  $TOTAL_FENS total FENs to label"

PYTHONPATH="$NN_ROOT" "$PYTHON" "$NN_ROOT/scripts/generate_data.py" \
    --label-engine "$STOCKFISH" \
    --fens "$FENS_DIR/new_b3b4b5.fens" \
    --output "$FENS_DIR/new_b3b4b5.jsonl" \
    --eval-depth 14 \
    --workers "$WORKERS" \
    --selfplay-games 0
echo "  → $(wc -l < "$FENS_DIR/new_b3b4b5.jsonl") labeled positions"

# ── Step 5: Split 90/10 ───────────────────────────────────────────────────────

echo ""
echo "[4/7] Splitting 90/10..."
LABELED=$(wc -l < "$FENS_DIR/new_b3b4b5.jsonl")
NEW_TRAIN_N=$(( LABELED * 9 / 10 ))
NEW_VAL_N=$(( LABELED - NEW_TRAIN_N ))
shuf "$FENS_DIR/new_b3b4b5.jsonl" > "$FENS_DIR/new_b3b4b5.shuffled.tmp"
head -n "$NEW_TRAIN_N" "$FENS_DIR/new_b3b4b5.shuffled.tmp" > "$FENS_DIR/new_b3b4b5_train.jsonl"
tail -n "$NEW_VAL_N"   "$FENS_DIR/new_b3b4b5.shuffled.tmp" > "$FENS_DIR/new_b3b4b5_val.jsonl"
rm "$FENS_DIR/new_b3b4b5.shuffled.tmp"
echo "  train: $NEW_TRAIN_N  val: $NEW_VAL_N"

# ── Step 6: Combine with all_41m ──────────────────────────────────────────────

echo ""
echo "[5/7] Combining with all_41m and shuffling..."
cat "$EXISTING_TRAIN" "$FENS_DIR/new_b3b4b5_train.jsonl" | shuf > "$OUT_DIR/train_all_54m.jsonl"
cat "$EXISTING_VAL"   "$FENS_DIR/new_b3b4b5_val.jsonl"   | shuf > "$OUT_DIR/val_all_54m.jsonl"
echo "  train: $(wc -l < "$OUT_DIR/train_all_54m.jsonl")"
echo "  val:   $(wc -l < "$OUT_DIR/val_all_54m.jsonl")"

# ── Step 7: Binary encode ─────────────────────────────────────────────────────

echo ""
echo "[6/7] Binary encoding..."
"$NNUE_PREPROCESS" --input "$OUT_DIR/train_all_54m.jsonl" --output "$OUT_DIR/train_all_54m" --dual
"$NNUE_PREPROCESS" --input "$OUT_DIR/val_all_54m.jsonl"   --output "$OUT_DIR/val_all_54m"   --dual

# ── Step 8: Create config + update lichess-bot ────────────────────────────────

echo ""
echo "[7/7] Creating training config..."

cat > "$NN_ROOT/configs/halfkp_dual_all_54m_1024_8b.yaml" << 'YAML'
seed: 137

training:
  batch_size: 32768
  epochs: 200
  lr: 0.001
  warmup_epochs: 5
  weight_decay: 0.0001
  grad_clip: 1.0
  workers: 8
  pin_memory: true
  amp: true
  persistent_workers: true
  prefetch_factor: 4

model:
  input_dim: 12288
  hidden_dim: 1024
  hidden2_dim: 32
  dropout: 0.3
  dual_perspective: true
  n_output_buckets: 8

loss:
  cp_weight: 0.3
  wdl_weight: 0.7

data:
  train_file: data/all_54m/train_all_54m.jsonl
  val_file:   data/all_54m/val_all_54m.jsonl
  max_cp_abs: 800
# ~54M positions, 8 buckets, hidden_dim=1024
YAML

# ── Verify bucket distribution ────────────────────────────────────────────────

echo ""
echo "[verify] 8-bucket distribution of all_54m train:"
"$PYTHON" - << EOF
import numpy as np
pc = np.load('$OUT_DIR/train_all_54m.piece_count.npy', mmap_mode='r').astype(int)
n = len(pc)
b = ((pc - 2) * 8 // 30).clip(0, 7)
labels = ['2-5','6-9','10-13','14-16','17-20','21-24','25-28','29-32']
for i in range(8):
    c = int((b==i).sum())
    ok = '✓' if c >= 3_000_000 else '✗'
    print(f'  b{i} ({labels[i]}): {c:>9,}  ({100*c/n:.1f}%)  {ok}')
EOF


# ── Cleanup large intermediates ───────────────────────────────────────────────

echo ""
echo "Cleaning up intermediate files..."
rm -f "$FENS_DIR/new_b3.fens" "$FENS_DIR/new_b4.fens" "$FENS_DIR/new_b5.fens"
rm -f "$FENS_DIR/new_b3b4b5.fens" "$FENS_DIR/new_b3b4b5.jsonl"
rm -f "$FENS_DIR/new_b3b4b5_train.jsonl" "$FENS_DIR/new_b3b4b5_val.jsonl"
echo "  Removing decompressed PGN to recover disk space..."
rm -f "$PGN"

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Done!"
echo ""
echo "  To start training:"
echo "  PYTHONPATH=$NN_ROOT .venv/bin/python scripts/train.py \\"
echo "    --config configs/halfkp_dual_all_54m_1024_8b.yaml \\"
echo "    --out artifacts/checkpoint_all_54m_1024_8b.pt \\"
echo "    --tb-logdir runs/all_54m_1024_8b"
echo ""
echo "  Export model when ready:"
echo "  PYTHONPATH=$NN_ROOT .venv/bin/python scripts/export_weights.py \\"
echo "    --checkpoint artifacts/checkpoint_all_54m_1024_8b.pt \\"
echo "    --output artifacts/eval_all_54m_1024_8b.npz"
echo "════════════════════════════════════════════════════════"

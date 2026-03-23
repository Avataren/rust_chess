#!/usr/bin/env bash
# puzzle_finetune.sh — Full pipeline: export → label → preprocess → finetune → export weights
#
# Usage:
#   ./scripts/puzzle_finetune.sh <puzzle_csv> [options]
#
# Examples:
#   ./scripts/puzzle_finetune.sh /data/lichess_db_puzzle.csv.zst
#   ./scripts/puzzle_finetune.sh /data/lichess_db_puzzle.csv.zst \
#       --resume artifacts/checkpoint_all_54m_1024_h64_8b_32kb_cp07.pt \
#       --min-rating 1500 --max-rating 2500 \
#       --depth 14 --epochs 30
#
# Steps:
#   1. Export all puzzle lines from the CSV (Rust, fast)
#   2. Walk each puzzle's move sequence, label positions with Stockfish (Python)
#   3. Binary-encode (nnue_preprocess)
#   4. Fine-tune from checkpoint (train.py --resume)
#   5. Export weights to eval.npz
#   6. Copy eval.npz into chess_evaluation for use by the engine
#
# The script is idempotent: already-completed steps are skipped if their
# output files already exist.  Delete specific output files to re-run a step.

set -euo pipefail

# ── Args ──────────────────────────────────────────────────────────────────────

PUZZLE_CSV="${1:-}"
if [[ -z "$PUZZLE_CSV" ]]; then
    echo "Usage: $0 <lichess_db_puzzle.csv[.zst]> [options]"
    echo ""
    echo "Options:"
    echo "  --resume PATH        checkpoint to finetune from (default: latest in artifacts/)"
    echo "  --out-name NAME      output name for data/checkpoints (default: puzzle_ft)"
    echo "  --min-rating N       lower rating filter (default: 0 = all)"
    echo "  --max-rating N       upper rating filter (default: no limit)"
    echo "  --depth N            Stockfish labeling depth (default: 14)"
    echo "  --workers N          parallel Stockfish workers (default: nproc-1)"
    echo "  --epochs N           fine-tune epochs (default: 30)"
    echo "  --lr FLOAT           fine-tune learning rate (default: 0.00005)"
    echo "  --stockfish PATH     Stockfish binary (default: auto-detect)"
    exit 1
fi
shift

# Defaults
RESUME=""
OUT_NAME="puzzle_ft"
MIN_RATING=0
MAX_RATING=99999
DEPTH=14
WORKERS=$(( $(nproc) - 1 ))
WORKERS=$(( WORKERS < 1 ? 1 : WORKERS ))
EPOCHS=30
LR="0.00005"
STOCKFISH=$(command -v stockfish 2>/dev/null || true)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume)     RESUME="$2";     shift 2 ;;
        --out-name)   OUT_NAME="$2";   shift 2 ;;
        --min-rating) MIN_RATING="$2"; shift 2 ;;
        --max-rating) MAX_RATING="$2"; shift 2 ;;
        --depth)      DEPTH="$2";      shift 2 ;;
        --workers)    WORKERS="$2";    shift 2 ;;
        --epochs)     EPOCHS="$2";     shift 2 ;;
        --lr)         LR="$2";         shift 2 ;;
        --stockfish)  STOCKFISH="$2";  shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NN_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$NN_ROOT/.." && pwd)"

PUZZLE_BENCH="$REPO_ROOT/target/release/puzzle_bench"
NNUE_PREPROCESS="$REPO_ROOT/target/release/nnue_preprocess"
GEN_FINETUNE="$SCRIPT_DIR/gen_puzzle_finetune_data.py"
TRAIN_PY="$SCRIPT_DIR/train.py"
EXPORT_WEIGHTS_PY="$SCRIPT_DIR/export_weights.py"

if [[ -x "$NN_ROOT/.venv/bin/python3" ]]; then
    PYTHON="$NN_ROOT/.venv/bin/python3"
else
    PYTHON="${PYTHON:-python3}"
fi

OUT_DIR="$NN_ROOT/data/$OUT_NAME"
EXPORT_TSV="$OUT_DIR/puzzles.tsv"
LABELED_TRAIN="$OUT_DIR/train_${OUT_NAME}.jsonl"
LABELED_VAL="$OUT_DIR/val_${OUT_NAME}.jsonl"
CHECKPOINT_OUT="$NN_ROOT/artifacts/checkpoint_${OUT_NAME}.pt"
NPZ_OUT="$NN_ROOT/artifacts/eval_${OUT_NAME}.npz"
ENGINE_NPZ="$REPO_ROOT/chess_evaluation/src/eval.npz"

# Auto-detect latest checkpoint if not specified
if [[ -z "$RESUME" ]]; then
    RESUME=$(stat -c "%Y %n" "$NN_ROOT/artifacts"/checkpoint_*.pt 2>/dev/null \
        | sort -rn | head -1 | awk '{print $2}' || true)
    if [[ -z "$RESUME" ]]; then
        echo "ERROR: No checkpoint found in artifacts/. Use --resume <path>."
        exit 1
    fi
fi

# ── Sanity checks ─────────────────────────────────────────────────────────────

[[ -f "$PUZZLE_CSV" ]]     || { echo "ERROR: puzzle CSV not found: $PUZZLE_CSV"; exit 1; }
[[ -x "$PUZZLE_BENCH" ]]   || { echo "ERROR: puzzle_bench not built. Run: cargo build --release -p chess_evaluation"; exit 1; }
[[ -x "$NNUE_PREPROCESS" ]]|| { echo "ERROR: nnue_preprocess not built. Run: cargo build --release -p nnue_preprocess"; exit 1; }
[[ -f "$RESUME" ]]         || { echo "ERROR: checkpoint not found: $RESUME"; exit 1; }
[[ -n "$STOCKFISH" && -x "$STOCKFISH" ]] || { echo "ERROR: stockfish not found. Install it or use --stockfish PATH"; exit 1; }

mkdir -p "$OUT_DIR"

# ── Banner ────────────────────────────────────────────────────────────────────

echo "════════════════════════════════════════════════════════"
echo "  puzzle_finetune.sh"
echo "  puzzle CSV : $PUZZLE_CSV"
echo "  rating     : ${MIN_RATING}–${MAX_RATING}"
echo "  output     : $OUT_DIR"
echo "  resume     : $RESUME"
echo "  stockfish  : $STOCKFISH  depth=$DEPTH  workers=$WORKERS"
echo "  finetune   : epochs=$EPOCHS  lr=$LR"
echo "════════════════════════════════════════════════════════"
echo ""

# ── Step 1: Export puzzle lines ───────────────────────────────────────────────

echo "[1/5] Exporting puzzle positions..."
if [[ -f "$EXPORT_TSV" ]]; then
    echo "  → Skipping (already exists): $EXPORT_TSV  ($(wc -l < "$EXPORT_TSV") lines)"
else
    RATING_FLAGS=""
    [[ "$MIN_RATING" -gt 0 ]]     && RATING_FLAGS="$RATING_FLAGS --min-rating $MIN_RATING"
    [[ "$MAX_RATING" -lt 99999 ]] && RATING_FLAGS="$RATING_FLAGS --max-rating $MAX_RATING"

    "$PUZZLE_BENCH" \
        --file "$PUZZLE_CSV" \
        --count 0 \
        $RATING_FLAGS \
        --export-all "$EXPORT_TSV" \
        --export-only
    echo "  → $(wc -l < "$EXPORT_TSV") puzzle lines written to $EXPORT_TSV"
fi

# ── Step 2: Label positions with Stockfish ────────────────────────────────────

echo ""
echo "[2/5] Labeling positions with Stockfish (depth=$DEPTH, workers=$WORKERS)..."
if [[ -f "$LABELED_TRAIN" && -f "$LABELED_VAL" ]]; then
    echo "  → Skipping (already exist):"
    echo "     $(wc -l < "$LABELED_TRAIN") train  $LABELED_TRAIN"
    echo "     $(wc -l < "$LABELED_VAL") val    $LABELED_VAL"
else
    PYTHONPATH="$NN_ROOT" "$PYTHON" "$GEN_FINETUNE" \
        --input    "$EXPORT_TSV" \
        --output   "$OUT_DIR" \
        --stockfish "$STOCKFISH" \
        --depth    "$DEPTH" \
        --workers  "$WORKERS"
fi

# ── Step 3: Binary encode ─────────────────────────────────────────────────────

echo ""
echo "[3/5] Binary encoding (dual perspective)..."
for SPLIT_JSONL in "$LABELED_TRAIN" "$LABELED_VAL"; do
    PREFIX="${SPLIT_JSONL%.jsonl}"
    LABEL="$(basename "$PREFIX")"
    SENTINEL="${PREFIX}.cp.npy"
    if [[ -f "$SENTINEL" ]]; then
        echo "  → Skipping $LABEL (already encoded)"
    else
        echo "  Encoding $LABEL..."
        "$NNUE_PREPROCESS" \
            --input  "$SPLIT_JSONL" \
            --output "$PREFIX" \
            --dual
    fi
done

# ── Step 4: Fine-tune ─────────────────────────────────────────────────────────

echo ""
echo "[4/5] Fine-tuning..."
if [[ -f "$CHECKPOINT_OUT" ]]; then
    echo "  → Skipping (checkpoint already exists): $CHECKPOINT_OUT"
else
    # Write a temporary config that overrides data paths, epochs, and lr.
    TMP_CONFIG="$OUT_DIR/finetune_run.yaml"
    cat > "$TMP_CONFIG" << YAML
seed: 42
training:
  batch_size: 16384
  epochs: $EPOCHS
  lr: $LR
  warmup_epochs: 2
  weight_decay: 0.0001
  grad_clip: 1.0
  workers: 8
  pin_memory: true
  amp: true
  persistent_workers: true
  prefetch_factor: 4
  compile: true
  compile_mode: "default"
model:
  input_dim: 24576
  hidden_dim: 1024
  hidden2_dim: 64
  dropout: 0.1
  dual_perspective: true
  n_output_buckets: 8
loss:
  cp_weight: 0.9
  wdl_weight: 0.1
data:
  train_file: $LABELED_TRAIN
  val_file:   $LABELED_VAL
  max_cp_abs: 1200
YAML

    PYTHONPATH="$NN_ROOT" "$PYTHON" "$TRAIN_PY" \
        --config  "$TMP_CONFIG" \
        --resume  "$RESUME" \
        --out     "$CHECKPOINT_OUT"
fi

# ── Step 5: Export weights ────────────────────────────────────────────────────

echo ""
echo "[5/5] Exporting weights to NPZ..."
if [[ -f "$NPZ_OUT" ]]; then
    echo "  → Skipping (already exists): $NPZ_OUT"
else
    PYTHONPATH="$NN_ROOT" "$PYTHON" "$EXPORT_WEIGHTS_PY" \
        --checkpoint "$CHECKPOINT_OUT" \
        --output     "$NPZ_OUT"
    echo "  → $NPZ_OUT"
fi

# Copy into engine source so the next build embeds the new weights.
echo ""
echo "  Copying $NPZ_OUT → $ENGINE_NPZ"
cp "$NPZ_OUT" "$ENGINE_NPZ"

# ── Done ──────────────────────────────────────────────────────────────────────

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Done!"
echo ""
echo "  Checkpoint : $CHECKPOINT_OUT"
echo "  NPZ        : $NPZ_OUT"
echo "  Engine NPZ : $ENGINE_NPZ  (embedded on next cargo build)"
echo ""
echo "  Rebuild engine:"
echo "    cargo build --release -p chess_evaluation"
echo ""
echo "  Run puzzle benchmark to measure improvement:"
echo "    ./target/release/puzzle_bench \\"
echo "      --file $PUZZLE_CSV \\"
echo "      --count 2000 --min-rating 1000 --max-rating 2500 --depth 7 --seed 42"
echo "════════════════════════════════════════════════════════"

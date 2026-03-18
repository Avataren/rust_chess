#!/usr/bin/env bash
# gen_balanced_dataset.sh — Build a balanced dataset by combining existing data
# with newly extracted endgame positions to hit a target endgame fraction.
#
# Usage:
#   ./scripts/gen_balanced_dataset.sh <pgn_file> <output_name> [options]
#
# Example:
#   ./scripts/gen_balanced_dataset.sh lichess_db_standard_rated_2021-01.pgn balanced_24m
#
# What it does:
#   1. Analyzes existing train + val JSONL files to find the endgame deficit
#   2. Extracts the required endgame FENs from the PGN
#   3. Labels with Stockfish depth 14
#   4. Splits new endgame data 90/10 train/val
#   5. Combines with existing train/val splits
#   6. Binary encodes the final combined datasets (dual perspective)
#
# The existing data is never modified — new files are written to data/<output_name>/.

set -euo pipefail

# ── Args ──────────────────────────────────────────────────────────────────────

PGN="${1:-}"
NAME="${2:-}"

if [[ -z "$PGN" || -z "$NAME" ]]; then
    echo "Usage: $0 <pgn_file> <output_name> [--target-fraction N] [--workers N] [--min-elo N] [--stockfish PATH] [--depth N]"
    exit 1
fi

TARGET_FRACTION=0.33
WORKERS=$(nproc)
MIN_ELO=1800
DEPTH=14
STOCKFISH=$(command -v stockfish 2>/dev/null || echo "")

shift 2
while [[ $# -gt 0 ]]; do
    case "$1" in
        --target-fraction) TARGET_FRACTION="$2"; shift 2 ;;
        --workers)         WORKERS="$2";          shift 2 ;;
        --min-elo)         MIN_ELO="$2";           shift 2 ;;
        --stockfish)       STOCKFISH="$2";         shift 2 ;;
        --depth)           DEPTH="$2";             shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$STOCKFISH" ]]; then
    echo "ERROR: stockfish not found in PATH. Use --stockfish /path/to/stockfish"
    exit 1
fi

# ── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NN_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$NN_ROOT/.." && pwd)"

FINETUNE_EXTRACT="$REPO_ROOT/target/release/finetune_extract"
NNUE_PREPROCESS="$REPO_ROOT/target/release/nnue_preprocess"
GENERATE_DATA="$NN_ROOT/scripts/generate_data.py"

if [[ -x "$NN_ROOT/.venv/bin/python3" ]]; then
    PYTHON="$NN_ROOT/.venv/bin/python3"
else
    PYTHON="${PYTHON:-python3}"
fi

OUT_DIR="$NN_ROOT/data/$NAME"
mkdir -p "$OUT_DIR"

EXISTING_TRAINS=(
    "$NN_ROOT/data/train_d14_20m.jsonl"
    "$NN_ROOT/data/finetune_1m/train_finetune_1m.jsonl"
)
EXISTING_VALS=(
    "$NN_ROOT/data/val_d14_20m.jsonl"
    "$NN_ROOT/data/finetune_1m/val_finetune_1m.jsonl"
)

NEW_ENDGAME_FENS="$OUT_DIR/new_endgame.fens"
NEW_ENDGAME_JSONL="$OUT_DIR/new_endgame.jsonl"
NEW_ENDGAME_TRAIN="$OUT_DIR/new_endgame_train.jsonl"
NEW_ENDGAME_VAL="$OUT_DIR/new_endgame_val.jsonl"
TRAIN_OUT="$OUT_DIR/train_$NAME.jsonl"
VAL_OUT="$OUT_DIR/val_$NAME.jsonl"

# ── Sanity checks ─────────────────────────────────────────────────────────────

if [[ ! -f "$PGN" ]]; then
    echo "ERROR: PGN not found: $PGN"
    exit 1
fi
if [[ ! -x "$FINETUNE_EXTRACT" ]]; then
    echo "ERROR: finetune_extract not built. Run: cargo build --release -p pgn_extract"
    exit 1
fi
if [[ ! -x "$NNUE_PREPROCESS" ]]; then
    echo "ERROR: nnue_preprocess not built. Run: cargo build --release -p nnue_preprocess"
    exit 1
fi
for f in "${EXISTING_TRAINS[@]}" "${EXISTING_VALS[@]}"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: existing dataset not found: $f"
        exit 1
    fi
done

# ── Step 1: Compute endgame deficit ───────────────────────────────────────────

echo ""
echo "[1/6] Analyzing existing data to compute endgame deficit..."

DEFICIT=$("$PYTHON" - <<EOF
import numpy as np
from pathlib import Path

TARGET = $TARGET_FRACTION

train_files = [$(printf '"%s",' "${EXISTING_TRAINS[@]}")]
val_files   = [$(printf '"%s",' "${EXISTING_VALS[@]}")]

def eg_count(paths):
    pcs = []
    for p in paths:
        prefix = str(Path(p).with_suffix(''))
        pcs.append(np.load(prefix + '.piece_count.npy', mmap_mode='r').astype(int))
    pc = np.concatenate(pcs)
    eg = int((pc <= 16).sum())
    n  = len(pc)
    extra = max(0, int((TARGET * n - eg) / (1 - TARGET)) + 1)
    return eg, n, extra

tr_eg, tr_n, tr_extra = eg_count(train_files)
va_eg, va_n, va_extra = eg_count(val_files)

# Extract total to split 90/10 and cover both deficits
# train_extra <= 0.9 * total  =>  total >= train_extra / 0.9
total = max(int(tr_extra / 0.9) + 1000, tr_extra + va_extra + 1000)

print(f"TRAIN  current={tr_n:,}  endgame={tr_eg:,} ({100*tr_eg/tr_n:.1f}%)  need={tr_extra:,}")
print(f"VAL    current={va_n:,}  endgame={va_eg:,} ({100*va_eg/va_n:.1f}%)  need={va_extra:,}")
print(f"EXTRACT {total}")
EOF
)

echo "$DEFICIT" | grep -v "^EXTRACT"
EXTRACT_N=$(echo "$DEFICIT" | grep "^EXTRACT" | awk '{print $2}')
echo "  → Extracting $EXTRACT_N new endgame FENs to reach ${TARGET_FRACTION} target"

# ── Step 2: Extract endgame FENs ──────────────────────────────────────────────

echo ""
echo "[2/6] Extracting $EXTRACT_N endgame FENs (≤16 pieces, last 40 plies, min-elo $MIN_ELO)..."
"$FINETUNE_EXTRACT" \
    --input "$PGN" \
    --output "$NEW_ENDGAME_FENS" \
    --max-positions "$EXTRACT_N" \
    --max-pieces 16 \
    --sample-from-last 40 \
    --min-elo "$MIN_ELO" \
    --positions-per-game 3

echo "  → $(wc -l < "$NEW_ENDGAME_FENS") endgame FENs extracted"

# ── Step 3: Label with Stockfish ──────────────────────────────────────────────

echo ""
echo "[3/6] Labeling with Stockfish depth $DEPTH ($WORKERS workers)..."
PYTHONPATH="$NN_ROOT" "$PYTHON" "$GENERATE_DATA" \
    --label-engine "$STOCKFISH" \
    --fens "$NEW_ENDGAME_FENS" \
    --output "$NEW_ENDGAME_JSONL" \
    --eval-depth "$DEPTH" \
    --workers "$WORKERS" \
    --selfplay-games 0 \
    --max-positions "$EXTRACT_N"

LABELED=$(wc -l < "$NEW_ENDGAME_JSONL")
echo "  → $LABELED labeled endgame positions"
rm "$NEW_ENDGAME_FENS"

# ── Step 4: Split new endgame data 90/10 ─────────────────────────────────────

echo ""
echo "[4/6] Splitting new endgame data 90/10..."
NEW_TRAIN_N=$(( LABELED * 9 / 10 ))
NEW_VAL_N=$(( LABELED - NEW_TRAIN_N ))
SHUFFLED="$OUT_DIR/new_endgame.shuffled.tmp"
shuf "$NEW_ENDGAME_JSONL" > "$SHUFFLED"
head -n "$NEW_TRAIN_N" "$SHUFFLED" > "$NEW_ENDGAME_TRAIN"
tail -n "$NEW_VAL_N"   "$SHUFFLED" > "$NEW_ENDGAME_VAL"
rm "$SHUFFLED" "$NEW_ENDGAME_JSONL"
echo "  → train: $NEW_TRAIN_N  val: $NEW_VAL_N"

# ── Step 5: Combine with existing splits ──────────────────────────────────────

echo ""
echo "[5/6] Combining with existing data and shuffling..."

echo "  Combining train..."
cat "${EXISTING_TRAINS[@]}" "$NEW_ENDGAME_TRAIN" | shuf > "$TRAIN_OUT"
rm "$NEW_ENDGAME_TRAIN"
TRAIN_TOTAL=$(wc -l < "$TRAIN_OUT")

echo "  Combining val..."
cat "${EXISTING_VALS[@]}" "$NEW_ENDGAME_VAL" | shuf > "$VAL_OUT"
rm "$NEW_ENDGAME_VAL"
VAL_TOTAL=$(wc -l < "$VAL_OUT")

echo "  → train: $TRAIN_TOTAL  val: $VAL_TOTAL"

# Verify balance
"$PYTHON" - <<EOF
import numpy as np
from pathlib import Path

for split, path in [('train', '$TRAIN_OUT'), ('val', '$VAL_OUT')]:
    # We can't read piece_count from JSONL directly - just count lines
    # Balance will be verified after binary encoding
    n = sum(1 for _ in open(path))
    print(f"  {split}: {n:,} positions")
EOF

# ── Step 6: Binary encode ─────────────────────────────────────────────────────

echo ""
echo "[6/6] Binary encoding (dual perspective)..."
for SPLIT_JSONL in "$TRAIN_OUT" "$VAL_OUT"; do
    PREFIX="${SPLIT_JSONL%.jsonl}"
    LABEL="$(basename "$PREFIX")"
    echo "  Encoding $LABEL..."
    "$NNUE_PREPROCESS" \
        --input "$SPLIT_JSONL" \
        --output "$PREFIX" \
        --dual
done

# ── Verify final balance ──────────────────────────────────────────────────────

echo ""
echo "[verify] Final bucket distribution..."
"$PYTHON" - <<EOF
import numpy as np
from pathlib import Path

for split in ['train_$NAME', 'val_$NAME']:
    pc_path = '$OUT_DIR/' + split + '.piece_count.npy'
    pc = np.load(pc_path, mmap_mode='r').astype(int)
    eg = (pc <= 16).sum()
    mg = (pc >= 17).sum()
    n  = len(pc)
    b0 = ((pc - 2) * 2 // 30).clip(0, 1)
    print(f"  {split}: {n:,}  endgame={eg:,} ({100*eg/n:.1f}%)  midgame={mg:,} ({100*mg/n:.1f}%)")
EOF

# ── Done ──────────────────────────────────────────────────────────────────────

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Done! Files in $OUT_DIR:"
for f in "$OUT_DIR"/*.jsonl; do
    [[ -f "$f" ]] && printf "    %-55s %7.0f MB\n" "$(basename "$f")" "$(( $(stat -c%s "$f") / 1000000 ))"
done
echo ""
echo "  Create a training config pointing to:"
echo "    train_file: data/$NAME/train_$NAME.jsonl"
echo "    val_file:   data/$NAME/val_$NAME.jsonl"
echo "════════════════════════════════════════════════════════"

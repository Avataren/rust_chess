#!/usr/bin/env bash
# binpack_to_binary.sh — Convert a .binpack file to binary .npy training format.
#
# No Stockfish re-labeling: uses scores already in the binpack, converting from
# SF internal units (÷2.96) to white-absolute centipawns. WDL calibration in
# features.py absorbs the ~5% rounding error.
#
# Usage:
#   ./scripts/binpack_to_binary.sh <input.binpack> <output_dir> [OPTIONS]
#
# Options:
#   --max N           Take at most N positions (default: 100000000 = 100M)
#   --shard-size N    Positions per processing shard (default: 10000000 = 10M)
#   --cp-divisor F    SF-units → centipawn divisor (default: 2.96)
#   --filter-cp N     Drop positions with |cp| > N after conversion (default: 3000)
#   --workers N       Preprocess workers (default: 8)
#   --keep-shards     Don't delete shard JSONL files after processing
#
# Example — 50M positions from T60T70wIsRightFarseer.binpack:
#   ./scripts/binpack_to_binary.sh \
#       datasets/T60T70wIsRightFarseer.binpack \
#       data/t60_t70 \
#       --max 50000000
#
# Example — full data_d9_2021 (large; use --max to cap):
#   ./scripts/binpack_to_binary.sh \
#       datasets/data_d9_2021_09_02.binpack \
#       data/d9_2021 \
#       --max 100000000
#
# Disk usage estimate per 10M positions:
#   JSONL shard: ~800 MB (deleted after preprocessing each shard)
#   Binary .npy: ~1.3 GB (kept)
# Total for 100M positions: ~13 GB binary output + peak ~0.8 GB temp JSONL.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BINPACK2JSONL="/tmp/binpack2jsonl"

# ── Defaults ─────────────────────────────────────────────────────────────────
MAX_POSITIONS=100000000
SHARD_SIZE=10000000
CP_DIVISOR=2.96
FILTER_CP=3000
WORKERS=8
KEEP_SHARDS=false

# ── Argument parsing ──────────────────────────────────────────────────────────
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <input.binpack> <output_dir> [--max N] [--shard-size N] [--cp-divisor F] [--filter-cp N] [--workers N] [--keep-shards]"
    exit 1
fi

INPUT_BINPACK="$1"; shift
OUTPUT_DIR="$1";    shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        --max)           MAX_POSITIONS="$2"; shift 2 ;;
        --shard-size)    SHARD_SIZE="$2";    shift 2 ;;
        --cp-divisor)    CP_DIVISOR="$2";    shift 2 ;;
        --filter-cp)     FILTER_CP="$2";     shift 2 ;;
        --workers)       WORKERS="$2";       shift 2 ;;
        --keep-shards)   KEEP_SHARDS=true;   shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Pre-flight checks ─────────────────────────────────────────────────────────
if [[ ! -f "$INPUT_BINPACK" ]]; then
    echo "ERROR: binpack not found: $INPUT_BINPACK"
    exit 1
fi

if [[ ! -x "$BINPACK2JSONL" ]]; then
    echo "ERROR: $BINPACK2JSONL not found or not executable."
    echo "Build it from the nnue-pytorch repo root:"
    echo "  cd /tmp/nnue-pytorch"
    echo "  g++ -O2 -std=c++17 -Idata_loader/cpp/lib -Idata_loader/cpp \\"
    echo "      $SCRIPT_DIR/binpack2jsonl.cpp -o $BINPACK2JSONL -lpthread"
    exit 1
fi

# ── Setup ─────────────────────────────────────────────────────────────────────
SHARDS_DIR="$OUTPUT_DIR/shards"
MERGED_PREFIX="$OUTPUT_DIR/merged"
mkdir -p "$SHARDS_DIR"

echo "═══════════════════════════════════════════════════"
echo " binpack_to_binary.sh"
echo "   input   : $INPUT_BINPACK"
echo "   output  : $OUTPUT_DIR"
echo "   max     : $MAX_POSITIONS positions"
echo "   shard   : $SHARD_SIZE positions/shard"
echo "   divisor : $CP_DIVISOR (SF units → centipawns)"
echo "   filter  : |cp| ≤ $FILTER_CP"
echo "═══════════════════════════════════════════════════"

# ── Step 1: Convert binpack → JSONL shards ────────────────────────────────────
echo ""
echo "[1/3] Converting binpack to JSONL shards..."

"$BINPACK2JSONL" "$INPUT_BINPACK" \
    --cp-divisor "$CP_DIVISOR" \
    --white-absolute \
    --filter-cp "$FILTER_CP" \
    --max "$MAX_POSITIONS" \
    | split --lines="$SHARD_SIZE" --numeric-suffixes=1 --suffix-length=3 \
            --additional-suffix=".jsonl" - "$SHARDS_DIR/shard_"

SHARD_COUNT=$(ls "$SHARDS_DIR"/shard_*.jsonl 2>/dev/null | wc -l)
echo "  Created $SHARD_COUNT shard(s) in $SHARDS_DIR/"

# ── Step 2: Preprocess each shard to binary ───────────────────────────────────
echo ""
echo "[2/3] Preprocessing shards to binary format..."

SHARD_NUM=0
for SHARD_JSONL in "$SHARDS_DIR"/shard_*.jsonl; do
    SHARD_NUM=$((SHARD_NUM + 1))
    SHARD_NAME="$(basename "$SHARD_JSONL" .jsonl)"
    SHARD_OUT="$SHARDS_DIR/$SHARD_NAME"
    LINES=$(wc -l < "$SHARD_JSONL")
    echo "  Shard $SHARD_NUM/$SHARD_COUNT: $SHARD_NAME ($LINES positions)..."

    PYTHONPATH="$REPO_ROOT" python3 "$SCRIPT_DIR/preprocess_dataset.py" \
        --input "$SHARD_JSONL" \
        --output "$SHARD_OUT" \
        --dual \
        --use-halfkp

    if [[ "$KEEP_SHARDS" == false ]]; then
        rm "$SHARD_JSONL"
    fi
done

# ── Step 3: Merge shards ──────────────────────────────────────────────────────
echo ""
echo "[3/3] Merging shards → $MERGED_PREFIX ..."

PYTHONPATH="$REPO_ROOT" python3 "$SCRIPT_DIR/merge_npy_shards.py" \
    --shards "$SHARDS_DIR" \
    --output "$MERGED_PREFIX"

echo ""
echo "Done. Binary dataset at: $MERGED_PREFIX.{white_indices,black_indices,cp,...}.npy"
echo ""
echo "To train a new base model on this + existing 69M data, use:"
echo "  See scripts/train_base.yaml (create a new config pointing to merged binary dataset)"

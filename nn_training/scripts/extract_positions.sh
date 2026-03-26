#!/usr/bin/env bash
# extract_positions.sh — Extract FEN positions from a Lichess PGN dump and label with Stockfish.
#
# By default uses bucket-balanced extraction: runs finetune_extract once per output
# bucket (8 total) so that each piece-count range contributes equally to the dataset.
# The bucket formula matches the model: bucket = ((piece_count - 2) * 8 // 30).clamp(0, 7)
#
# Usage:
#   ./scripts/extract_positions.sh <pgn_or_zst> <output_name> [options]
#
# Examples:
#   # Extract 15M balanced positions (skip first 10M games), merge with all_54m → all_69m
#   ./scripts/extract_positions.sh \
#       lichess_db_standard_rated_2020-05.pgn.zst \
#       all_69m \
#       --positions 15000000 \
#       --skip-games 10000000 \
#       --merge-with data/all_54m
#
#   # Unbalanced (uniform sampling), relabeling an existing FENs file, no merge
#   ./scripts/extract_positions.sh \
#       /dev/null all_69m \
#       --fens data/all_69m/all_69m.fens.txt \
#       --positions 15000000 \
#       --skip-extract \
#       --no-balance
#
# What it does:
#   1. Extract FENs via Rust finetune_extract (8 parallel jobs, one per output bucket)
#      or pgn_extract (uniform) when --no-balance is set
#   2. Label with Stockfish in parallel
#   3. Shuffle + split 90/10 into new_train / new_val
#   4. Merge new_train + base_train (and val) from --merge-with, shuffle
#   5. Binary encode (dual-perspective) with nnue_preprocess

set -euo pipefail

# ── Args ──────────────────────────────────────────────────────────────────────

PGN="${1:-}"
NAME="${2:-}"

if [[ -z "$PGN" || -z "$NAME" ]]; then
    echo "Usage: $0 <pgn_or_zst> <output_name> [--positions N] [--skip-games N] [--min-elo N] [--depth N] [--workers N] [--stockfish PATH] [--fens FILE] [--merge-with DIR] [--skip-extract] [--skip-label] [--skip-split] [--skip-merge] [--skip-preprocess] [--no-balance] [--no-dual]"
    exit 1
fi

POSITIONS=15000000
SKIP_GAMES=0
MIN_ELO=0
DEPTH=14
WORKERS=$(nproc)
STOCKFISH=$(command -v stockfish 2>/dev/null || echo "")
FENS_FILE=""
MERGE_DIR=""
MERGE_ALSO_TRAIN=()   # extra pre-labeled JSONL files to add to train (space-separated or repeated flags)
MERGE_ALSO_VAL=()     # corresponding val files
SKIP_EXTRACT=0
SKIP_LABEL=0
SKIP_SPLIT=0
SKIP_MERGE=0
SKIP_PREPROCESS=0
NO_BALANCE=0
NO_DUAL=0

shift 2
while [[ $# -gt 0 ]]; do
    case "$1" in
        --positions)          POSITIONS="$2";                    shift 2 ;;
        --skip-games)         SKIP_GAMES="$2";                   shift 2 ;;
        --min-elo)            MIN_ELO="$2";                      shift 2 ;;
        --depth)              DEPTH="$2";                        shift 2 ;;
        --workers)            WORKERS="$2";                      shift 2 ;;
        --stockfish)          STOCKFISH="$2";                    shift 2 ;;
        --fens)               FENS_FILE="$2";                    shift 2 ;;
        --merge-with)         MERGE_DIR="$2";                    shift 2 ;;
        --merge-also-train)   MERGE_ALSO_TRAIN+=("$2");          shift 2 ;;
        --merge-also-val)     MERGE_ALSO_VAL+=("$2");            shift 2 ;;
        --skip-extract)       SKIP_EXTRACT=1;                    shift   ;;
        --skip-label)         SKIP_LABEL=1;                      shift   ;;
        --skip-split)         SKIP_SPLIT=1;                      shift   ;;
        --skip-merge)         SKIP_MERGE=1;                      shift   ;;
        --skip-preprocess)    SKIP_PREPROCESS=1;                 shift   ;;
        --no-balance)         NO_BALANCE=1;                      shift   ;;
        --no-dual)            NO_DUAL=1;                         shift   ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NN_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$NN_ROOT/.." && pwd)"

PGN_EXTRACT="$REPO_ROOT/target/release/pgn_extract"
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

FENS_OUT="${FENS_FILE:-$OUT_DIR/$NAME.fens.txt}"
LABELED_OUT="$OUT_DIR/$NAME.jsonl"
# After 90/10 split, new data lands here (before optional merge with base dataset)
NEW_TRAIN_OUT="$OUT_DIR/new_train_$NAME.jsonl"
NEW_VAL_OUT="$OUT_DIR/new_val_$NAME.jsonl"
# Final train/val (either merged or just the new data)
TRAIN_OUT="$OUT_DIR/train_$NAME.jsonl"
VAL_OUT="$OUT_DIR/val_$NAME.jsonl"

# ── Output bucket piece-count ranges ─────────────────────────────────────────
# Formula: bucket = ((piece_count - 2) * 8 // 30).clamp(0, 7)
#   bucket 0: pieces  2-5   (deep endgame)
#   bucket 1: pieces  6-9
#   bucket 2: pieces 10-13
#   bucket 3: pieces 14-16
#   bucket 4: pieces 17-20
#   bucket 5: pieces 21-24
#   bucket 6: pieces 25-28
#   bucket 7: pieces 29-32  (opening / full board)
BUCKET_MIN=(2  6  10 14 17 21 25 29)
BUCKET_MAX=(5  9  13 16 20 24 28 32)

# ── Sanity checks ─────────────────────────────────────────────────────────────

if [[ $SKIP_EXTRACT -eq 0 ]]; then
    if [[ $NO_BALANCE -eq 1 && ! -x "$PGN_EXTRACT" ]]; then
        echo "ERROR: pgn_extract not built. Run: cargo build --release -p pgn_extract"
        exit 1
    fi
    if [[ $NO_BALANCE -eq 0 && ! -x "$FINETUNE_EXTRACT" ]]; then
        echo "ERROR: finetune_extract not built. Run: cargo build --release -p pgn_extract"
        exit 1
    fi
fi
if [[ $SKIP_LABEL -eq 0 && -z "$STOCKFISH" ]]; then
    echo "ERROR: stockfish not found in PATH. Use --stockfish /path/to/stockfish"
    exit 1
fi
if [[ $SKIP_PREPROCESS -eq 0 && ! -x "$NNUE_PREPROCESS" ]]; then
    echo "ERROR: nnue_preprocess not built. Run: cargo build --release -p nnue_preprocess"
    exit 1
fi
if [[ $SKIP_MERGE -eq 0 && -n "$MERGE_DIR" ]]; then
    MERGE_TRAIN=$(echo "$MERGE_DIR"/train_*.jsonl | tr ' ' '\n' | head -1)
    MERGE_VAL=$(echo "$MERGE_DIR"/val_*.jsonl   | tr ' ' '\n' | head -1)
    if [[ ! -f "$MERGE_TRAIN" ]]; then
        echo "ERROR: --merge-with set but no train_*.jsonl found in $MERGE_DIR"
        exit 1
    fi
    if [[ ! -f "$MERGE_VAL" ]]; then
        echo "ERROR: --merge-with set but no val_*.jsonl found in $MERGE_DIR"
        exit 1
    fi
fi

echo ""
echo "════════════════════════════════════════════════════════"
echo "  extract_positions.sh"
echo "  source    : $PGN"
echo "  output    : $OUT_DIR"
echo "  positions : $POSITIONS (new)"
echo "  skip-games: $SKIP_GAMES"
echo "  min-elo   : $MIN_ELO"
echo "  depth     : $DEPTH"
echo "  workers   : $WORKERS"
echo "  balanced  : $([ $NO_BALANCE -eq 1 ] && echo no || echo yes)"
echo "  merge-with: ${MERGE_DIR:-none}"
if [[ ${#MERGE_ALSO_TRAIN[@]} -gt 0 ]]; then
    for f in "${MERGE_ALSO_TRAIN[@]}"; do echo "  merge-also: $f  ($(wc -l < "$f") lines)"; done
fi
echo "  dual enc  : $([ $NO_DUAL -eq 1 ] && echo no || echo yes)"
echo "════════════════════════════════════════════════════════"

# ── Step 1: Extract FENs ──────────────────────────────────────────────────────

if [[ $SKIP_EXTRACT -eq 0 ]]; then
    echo ""
    if [[ $NO_BALANCE -eq 1 ]]; then
        # ── Uniform extraction (pgn_extract) ──────────────────────────────────
        echo "[1/4] Extracting $POSITIONS FENs (uniform, skip-games=$SKIP_GAMES)..."
        CMD=(
            "$PGN_EXTRACT"
            --input "$PGN"
            --output "$FENS_OUT"
            --max-positions "$POSITIONS"
            --skip-games "$SKIP_GAMES"
        )
        if [[ $MIN_ELO -gt 0 ]]; then
            CMD+=(--min-elo "$MIN_ELO")
        fi
        "${CMD[@]}"
    else
        # ── Bucket-balanced extraction (finetune_extract × 8, parallel) ───────
        PER_BUCKET=$(( POSITIONS / 8 ))
        # Give any remainder to bucket 7 (most abundant source)
        REMAINDER=$(( POSITIONS - PER_BUCKET * 8 ))
        echo "[1/4] Extracting ${POSITIONS} FENs balanced across 8 buckets (~${PER_BUCKET} each, skip-games=$SKIP_GAMES)..."
        echo "  Launching 8 parallel extractions..."

        PIDS=()
        BUCKET_FILES=()
        for b in 0 1 2 3 4 5 6 7; do
            MIN_PC="${BUCKET_MIN[$b]}"
            MAX_PC="${BUCKET_MAX[$b]}"
            BFILE="$OUT_DIR/bucket_${b}.fens.tmp"
            BUCKET_FILES+=("$BFILE")
            N=$(( PER_BUCKET + ( b == 7 ? REMAINDER : 0 ) ))

            CMD=(
                "$FINETUNE_EXTRACT"
                --input "$PGN"
                --output "$BFILE"
                --max-positions "$N"
                --min-pieces "$MIN_PC"
                --max-pieces "$MAX_PC"
                --skip-games "$SKIP_GAMES"
                --positions-per-game 1
            )
            if [[ $MIN_ELO -gt 0 ]]; then
                CMD+=(--min-elo "$MIN_ELO")
            fi
            # For deep endgame buckets, only sample from the last plies of each
            # game (where piece counts are lowest), greatly reducing scan time.
            if [[ $b -le 2 ]]; then
                CMD+=(--sample-from-last 100)
            fi

            LOG="$OUT_DIR/bucket_${b}.extract.log"
            "${CMD[@]}" > "$LOG" 2>&1 &
            PIDS+=($!)
            echo "  bucket $b (pieces ${MIN_PC}-${MAX_PC}, n=$N) → $BFILE  [pid ${PIDS[-1]}]"
        done

        # Wait for all extractions with progress reporting
        echo ""
        FAILED=0
        for i in "${!PIDS[@]}"; do
            pid="${PIDS[$i]}"
            b=$i
            if wait "$pid"; then
                GOT=$(wc -l < "${BUCKET_FILES[$b]}" 2>/dev/null || echo 0)
                echo "  bucket $b done: $GOT FENs"
            else
                echo "  bucket $b FAILED (exit $?)"
                FAILED=1
            fi
        done
        if [[ $FAILED -ne 0 ]]; then
            echo "ERROR: one or more bucket extractions failed. Check logs in $OUT_DIR/bucket_*.extract.log"
            exit 1
        fi

        # Merge + shuffle into single FENs file
        echo "  Merging and shuffling buckets..."
        cat "${BUCKET_FILES[@]}" | shuf > "$FENS_OUT"
        rm -f "${BUCKET_FILES[@]}"
    fi

    EXTRACTED=$(wc -l < "$FENS_OUT")
    echo "  → $EXTRACTED FENs written to $FENS_OUT"

    # Print per-bucket summary using Python
    if [[ $NO_BALANCE -eq 0 ]]; then
        "$PYTHON" - <<EOF
import sys
from collections import Counter

buckets = Counter()
with open("$FENS_OUT") as f:
    for line in f:
        fen = line.strip()
        if not fen:
            continue
        # Count pieces from FEN board part (field 0)
        board = fen.split()[0]
        pc = sum(1 for c in board if c.isalpha())
        b = min(7, max(0, (pc - 2) * 8 // 30))
        buckets[b] += 1

total = sum(buckets.values())
print(f"  Bucket distribution ({total:,} total):")
ranges = [(2,5),(6,9),(10,13),(14,16),(17,20),(21,24),(25,28),(29,32)]
for b in range(8):
    n = buckets.get(b, 0)
    bar = "█" * int(30 * n / max(buckets.values(), default=1))
    lo, hi = ranges[b]
    print(f"    bucket {b} (pc {lo:2d}-{hi:2d}): {n:>7,}  {100*n/total:5.1f}%  {bar}")
EOF
    fi

else
    if [[ ! -f "$FENS_OUT" ]]; then
        echo "ERROR: --skip-extract set but FENs file not found: $FENS_OUT"
        exit 1
    fi
    EXTRACTED=$(wc -l < "$FENS_OUT")
    echo "[1/4] Skipping extraction — using $FENS_OUT ($EXTRACTED FENs)"
fi

# ── Step 2: Label with Stockfish ──────────────────────────────────────────────

if [[ $SKIP_LABEL -eq 0 ]]; then
    echo ""
    echo "[2/4] Labeling $EXTRACTED FENs with Stockfish depth $DEPTH ($WORKERS workers)..."
    PYTHONPATH="$NN_ROOT" "$PYTHON" "$GENERATE_DATA" \
        --label-engine "$STOCKFISH" \
        --fens "$FENS_OUT" \
        --output "$LABELED_OUT" \
        --eval-depth "$DEPTH" \
        --workers "$WORKERS" \
        --selfplay-games 0 \
        --max-positions "$EXTRACTED"
    LABELED=$(wc -l < "$LABELED_OUT")
    echo "  → $LABELED labeled positions"
else
    if [[ ! -f "$LABELED_OUT" ]]; then
        echo "ERROR: --skip-label set but labeled JSONL not found: $LABELED_OUT"
        exit 1
    fi
    LABELED=$(wc -l < "$LABELED_OUT")
    echo "[2/4] Skipping labeling — using $LABELED_OUT ($LABELED positions)"
fi

# ── Step 3: Shuffle + split 90/10 ─────────────────────────────────────────────

if [[ $SKIP_SPLIT -eq 0 ]]; then
    echo ""
    echo "[3/5] Shuffling and splitting 90/10..."
    TRAIN_N=$(( LABELED * 9 / 10 ))
    VAL_N=$(( LABELED - TRAIN_N ))
    SHUFFLED="$OUT_DIR/$NAME.shuffled.tmp"
    shuf "$LABELED_OUT" > "$SHUFFLED"
    head -n "$TRAIN_N" "$SHUFFLED" > "$NEW_TRAIN_OUT"
    tail -n "$VAL_N"   "$SHUFFLED" > "$NEW_VAL_OUT"
    rm "$SHUFFLED"
    echo "  → new train: $TRAIN_N  new val: $VAL_N"
else
    for f in "$NEW_TRAIN_OUT" "$NEW_VAL_OUT"; do
        if [[ ! -f "$f" ]]; then
            echo "ERROR: --skip-split set but $f not found"
            exit 1
        fi
    done
    TRAIN_N=$(wc -l < "$NEW_TRAIN_OUT")
    VAL_N=$(wc -l < "$NEW_VAL_OUT")
    echo "[3/5] Skipping split — using existing new_train/new_val ($TRAIN_N / $VAL_N)"
fi

# ── Step 4: Merge with base dataset ───────────────────────────────────────────

if [[ $SKIP_MERGE -eq 0 ]]; then
    echo ""
    echo "[4/5] Merging datasets and shuffling..."

    # Collect all train sources: new data + optional base dataset + optional extra files
    TRAIN_SOURCES=("$NEW_TRAIN_OUT")
    VAL_SOURCES=("$NEW_VAL_OUT")

    if [[ -n "$MERGE_DIR" ]]; then
        MERGE_TRAIN=$(echo "$MERGE_DIR"/train_*.jsonl | tr ' ' '\n' | head -1)
        MERGE_VAL=$(echo  "$MERGE_DIR"/val_*.jsonl   | tr ' ' '\n' | head -1)
        TRAIN_SOURCES=("$MERGE_TRAIN" "${TRAIN_SOURCES[@]}")
        VAL_SOURCES=("$MERGE_VAL"     "${VAL_SOURCES[@]}")
    fi
    for f in "${MERGE_ALSO_TRAIN[@]}"; do TRAIN_SOURCES+=("$f"); done
    for f in "${MERGE_ALSO_VAL[@]}";   do VAL_SOURCES+=("$f");   done

    echo "  Train sources:"
    for f in "${TRAIN_SOURCES[@]}"; do printf "    %-60s  %s lines\n" "$f" "$(wc -l < "$f")"; done
    echo "  Val sources:"
    for f in "${VAL_SOURCES[@]}"; do printf "    %-60s  %s lines\n" "$f" "$(wc -l < "$f")"; done

    echo "  Merging train..."
    cat "${TRAIN_SOURCES[@]}" | shuf > "$TRAIN_OUT"
    echo "  Merging val..."
    cat "${VAL_SOURCES[@]}"   | shuf > "$VAL_OUT"

    # Clean up intermediate new_train/new_val (now merged in)
    [[ -f "$NEW_TRAIN_OUT" && "$NEW_TRAIN_OUT" != "$TRAIN_OUT" ]] && rm "$NEW_TRAIN_OUT"
    [[ -f "$NEW_VAL_OUT"   && "$NEW_VAL_OUT"   != "$VAL_OUT"   ]] && rm "$NEW_VAL_OUT"

    echo "  → train: $(wc -l < "$TRAIN_OUT")  val: $(wc -l < "$VAL_OUT")"
else
    for f in "$TRAIN_OUT" "$VAL_OUT"; do
        [[ -f "$f" ]] || { echo "ERROR: --skip-merge set but $f not found"; exit 1; }
    done
    echo "[4/5] Skipping merge"
fi

# ── Step 5: Binary encode ─────────────────────────────────────────────────────

if [[ $SKIP_PREPROCESS -eq 0 ]]; then
    echo ""
    echo "[5/5] Binary encoding..."
    for SPLIT_JSONL in "$TRAIN_OUT" "$VAL_OUT"; do
        PREFIX="${SPLIT_JSONL%.jsonl}"
        LABEL="$(basename "$PREFIX")"
        echo "  Encoding $LABEL (single-perspective)..."
        "$NNUE_PREPROCESS" \
            --input "$SPLIT_JSONL" \
            --output "$PREFIX"
        if [[ $NO_DUAL -eq 0 ]]; then
            echo "  Encoding $LABEL (dual-perspective)..."
            "$NNUE_PREPROCESS" \
                --input "$SPLIT_JSONL" \
                --output "$PREFIX" \
                --dual
        fi
    done
else
    echo "[5/5] Skipping binary encoding"
fi

# ── Done ──────────────────────────────────────────────────────────────────────

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Done! Files in $OUT_DIR:"
for f in "$OUT_DIR"/*.jsonl "$OUT_DIR"/*.txt; do
    [[ -f "$f" ]] && printf "    %-55s %7.0f MB\n" "$(basename "$f")" "$(( $(stat -c%s "$f") / 1000000 ))"
done
echo ""
echo "  To train, point your config at:"
echo "    train_file: data/$NAME/train_$NAME.jsonl"
echo "    val_file:   data/$NAME/val_$NAME.jsonl"
echo "════════════════════════════════════════════════════════"

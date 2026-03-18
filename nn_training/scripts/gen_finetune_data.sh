#!/usr/bin/env bash
# gen_finetune_data.sh — Generate a balanced finetune dataset from a Lichess PGN.
#
# Usage:
#   ./scripts/gen_finetune_data.sh <pgn_file> <output_name> [options]
#
# Examples:
#   ./scripts/gen_finetune_data.sh /data/lichess_2021.pgn finetune_1m
#   ./scripts/gen_finetune_data.sh /data/lichess_2021.pgn finetune_1m --endgame 600000 --midgame 600000
#
# Output (in data/):
#   data/<output_name>/train_<output_name>.{jsonl,white_indices.npy,...}
#   data/<output_name>/val_<output_name>.{jsonl,white_indices.npy,...}
#
# Pipeline:
#   1. Extract endgame FENs  (≤16 pieces, last 40 plies, min-elo 1800)
#   2. Extract midgame FENs  (17-32 pieces,              min-elo 1800)
#   3. Shuffle + combine
#   4. Label with Stockfish depth 14
#   5. Split 90/10 train/val
#   6. Binary encode (dual perspective, for EvalNetDual)

set -euo pipefail

# ── Args ──────────────────────────────────────────────────────────────────────

PGN="${1:-}"
NAME="${2:-}"

if [[ -z "$PGN" || -z "$NAME" ]]; then
    echo "Usage: $0 <pgn_file> <output_name> [--endgame N] [--midgame N] [--workers N] [--min-elo N] [--stockfish PATH] [--depth N]"
    exit 1
fi

# Defaults
ENDGAME_N=500000
MIDGAME_N=500000
WORKERS=$(nproc)
MIN_ELO=1800
DEPTH=14
STOCKFISH=$(command -v stockfish 2>/dev/null || echo "")

# Parse optional flags
shift 2
while [[ $# -gt 0 ]]; do
    case "$1" in
        --endgame)   ENDGAME_N="$2"; shift 2 ;;
        --midgame)   MIDGAME_N="$2"; shift 2 ;;
        --workers)   WORKERS="$2";   shift 2 ;;
        --min-elo)   MIN_ELO="$2";   shift 2 ;;
        --stockfish) STOCKFISH="$2"; shift 2 ;;
        --depth)     DEPTH="$2";     shift 2 ;;
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
# Prefer the local venv if it exists
if [[ -x "$NN_ROOT/.venv/bin/python3" ]]; then
    PYTHON="$NN_ROOT/.venv/bin/python3"
else
    PYTHON="${PYTHON:-python3}"
fi

OUT_DIR="$NN_ROOT/data/$NAME"
mkdir -p "$OUT_DIR"

ENDGAME_FENS="$OUT_DIR/endgame.fens"
MIDGAME_FENS="$OUT_DIR/midgame.fens"
ALL_FENS="$OUT_DIR/all.fens"
LABELED_JSONL="$OUT_DIR/$NAME.jsonl"
TRAIN_JSONL="$OUT_DIR/train_$NAME.jsonl"
VAL_JSONL="$OUT_DIR/val_$NAME.jsonl"
TRAIN_PREFIX="$OUT_DIR/train_$NAME"
VAL_PREFIX="$OUT_DIR/val_$NAME"

TOTAL=$(( ENDGAME_N + MIDGAME_N ))
VAL_LINES=$(( TOTAL / 10 ))
TRAIN_LINES=$(( TOTAL - VAL_LINES ))

# ── Sanity checks ─────────────────────────────────────────────────────────────

if [[ ! -f "$PGN" ]]; then
    echo "ERROR: PGN file not found: $PGN"
    exit 1
fi

if [[ ! -x "$FINETUNE_EXTRACT" ]]; then
    echo "ERROR: finetune_extract not built. Run:"
    echo "  cargo build --release -p pgn_extract"
    exit 1
fi

if [[ ! -x "$NNUE_PREPROCESS" ]]; then
    echo "ERROR: nnue_preprocess not built. Run:"
    echo "  cargo build --release -p nnue_preprocess"
    exit 1
fi

# ── Banner ────────────────────────────────────────────────────────────────────

echo "════════════════════════════════════════════════════════"
echo "  gen_finetune_data.sh"
echo "  pgn      : $PGN"
echo "  name     : $NAME"
echo "  endgame  : $ENDGAME_N positions (≤16 pieces, last 40 plies, min-elo $MIN_ELO)"
echo "  midgame  : $MIDGAME_N positions (17-32 pieces,              min-elo $MIN_ELO)"
echo "  total    : $TOTAL  →  train $TRAIN_LINES / val $VAL_LINES"
echo "  depth    : $DEPTH"
echo "  workers  : $WORKERS"
echo "  stockfish: $STOCKFISH"
echo "  output   : $OUT_DIR"
echo "════════════════════════════════════════════════════════"

# ── Step 1: Extract endgame FENs ──────────────────────────────────────────────

echo ""
echo "[1/6] Extracting $ENDGAME_N endgame FENs (≤16 pieces, last 40 plies)..."
"$FINETUNE_EXTRACT" \
    --input "$PGN" \
    --output "$ENDGAME_FENS" \
    --max-positions "$ENDGAME_N" \
    --max-pieces 16 \
    --sample-from-last 40 \
    --min-elo "$MIN_ELO" \
    --positions-per-game 2

echo "  → $(wc -l < "$ENDGAME_FENS") endgame FENs written"

# ── Step 2: Extract midgame FENs ──────────────────────────────────────────────

echo ""
echo "[2/6] Extracting $MIDGAME_N midgame FENs (17-32 pieces)..."
"$FINETUNE_EXTRACT" \
    --input "$PGN" \
    --output "$MIDGAME_FENS" \
    --max-positions "$MIDGAME_N" \
    --min-pieces 17 \
    --min-elo "$MIN_ELO"

echo "  → $(wc -l < "$MIDGAME_FENS") midgame FENs written"

# ── Step 3: Shuffle + combine ─────────────────────────────────────────────────

echo ""
echo "[3/6] Combining and shuffling..."
cat "$ENDGAME_FENS" "$MIDGAME_FENS" | shuf > "$ALL_FENS"
echo "  → $(wc -l < "$ALL_FENS") total FENs"
rm "$ENDGAME_FENS" "$MIDGAME_FENS"

# ── Step 4: Label with Stockfish ──────────────────────────────────────────────

echo ""
echo "[4/6] Labeling with Stockfish depth $DEPTH ($WORKERS workers)..."
PYTHONPATH="$NN_ROOT" "$PYTHON" "$GENERATE_DATA" \
    --label-engine "$STOCKFISH" \
    --fens "$ALL_FENS" \
    --output "$LABELED_JSONL" \
    --eval-depth "$DEPTH" \
    --workers "$WORKERS" \
    --selfplay-games 0 \
    --max-positions "$TOTAL"

echo "  → $(wc -l < "$LABELED_JSONL") labeled positions"
rm "$ALL_FENS"

# ── Step 5: Shuffle + split 90/10 ────────────────────────────────────────────

echo ""
echo "[5/6] Splitting into train ($TRAIN_LINES) / val ($VAL_LINES)..."
SHUFFLED="$OUT_DIR/$NAME.shuffled.tmp"
shuf "$LABELED_JSONL" > "$SHUFFLED"
head -n "$TRAIN_LINES" "$SHUFFLED" > "$TRAIN_JSONL"
tail -n "$VAL_LINES"   "$SHUFFLED" > "$VAL_JSONL"
rm "$SHUFFLED" "$LABELED_JSONL"
echo "  → train: $(wc -l < "$TRAIN_JSONL")  val: $(wc -l < "$VAL_JSONL")"

# ── Step 6: Binary encode (dual perspective) ──────────────────────────────────

echo ""
echo "[6/6] Binary encoding (dual perspective)..."
for SPLIT_JSONL in "$TRAIN_JSONL" "$VAL_JSONL"; do
    PREFIX="${SPLIT_JSONL%.jsonl}"
    LABEL="$(basename "$PREFIX")"
    echo "  Encoding $LABEL..."
    "$NNUE_PREPROCESS" \
        --input "$SPLIT_JSONL" \
        --output "$PREFIX" \
        --dual
done

# ── Done ──────────────────────────────────────────────────────────────────────

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Done! Files in $OUT_DIR:"
for f in "$OUT_DIR"/*.{jsonl,npy} 2>/dev/null; do
    [[ -f "$f" ]] && printf "    %-55s %7.0f MB\n" "$(basename "$f")" "$(( $(stat -c%s "$f") / 1000000 ))"
done
echo ""
echo "  Add to your finetune config:"
echo "    data:"
echo "      train_file: data/$NAME/train_$NAME.jsonl"
echo "      val_file:   data/$NAME/val_$NAME.jsonl"
echo "      max_cp_abs: 800"
echo "════════════════════════════════════════════════════════"

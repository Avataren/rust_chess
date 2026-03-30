#!/usr/bin/env bash
# Self-play improvement loop with anchor-data anti-drift protection.
# Run from anywhere:
#   /home/avataren/src/rust_chess/nn_training/start_selfplay_loop.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN_DIR="$REPO_DIR/target/release"

cd "$SCRIPT_DIR"

PYTHONPATH=. python3 scripts/selfplay_loop.py \
  --engine              "$BIN_DIR/chess_uci" \
  --stockfish           /usr/bin/stockfish \
  --initial-checkpoint  artifacts/best_checkpoint.pt \
  --puzzle-binary       "$BIN_DIR/puzzle_bench" \
  --puzzle-file         "$REPO_DIR/lichess_db_puzzle.csv.zst" \
  --selfplay-binary     "$BIN_DIR/self_play" \
  --anchor-data         data/all_69m/train_all_69m.jsonl \
  --anchor-size         50000 \
  --anchor-min-fraction 0.10 \
  --selfplay-eval-games 150 \
  --gen-val-max-increase 5.0 \
  --selfplay-noise-prob 0.05 \
  --selfplay-dirichlet-alpha 0.3 \
  --selfplay-dirichlet-amplitude 100

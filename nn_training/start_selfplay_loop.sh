#!/usr/bin/env bash
# Self-play improvement loop with anchor-data anti-drift protection.
# Run from the nn_training directory:
#   cd /home/avataren/src/rust_chess/nn_training && ./start_selfplay_loop.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHONPATH=. python3 scripts/selfplay_loop.py \
  --engine      ../../target/release/chess_uci \
  --stockfish   /usr/bin/stockfish \
  --initial-checkpoint artifacts/best_checkpoint.pt \
  --puzzle-binary  ../../target/release/puzzle_bench \
  --puzzle-file    ../lichess_db_puzzle.csv.zst \
  --selfplay-binary ../../target/release/self_play \
  --anchor-data data/all_69m/train_all_69m.jsonl \
  --anchor-size 50000

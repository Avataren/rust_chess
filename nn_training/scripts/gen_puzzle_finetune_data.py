#!/usr/bin/env python3
"""gen_puzzle_finetune_data.py — Turn exported puzzle lines into labeled training data.

Reads the tab-separated file written by puzzle_bench --export-all (or
--export-failures), walks each puzzle's full move sequence with python-chess to
generate every intermediate position, labels each with Stockfish, then writes
train/val splits ready for train.py.

Usage:
  python3 scripts/gen_puzzle_finetune_data.py \
      --input  data/puzzle_failures.tsv \
      --output data/puzzle_ft \
      --stockfish $(which stockfish) \
      [--depth 14] [--workers N] [--val-fraction 0.05]

Input format (one line per puzzle):
  <FEN>\\t<move0> <move1> <move2> ...

  move0   = opponent's setup move (applied first)
  move1   = correct solution move
  move2+  = continuation (alternating sides)

  All positions along the full solution line are collected (both sides to move),
  giving dense coverage of the tactical sequence.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import chess
import chess.engine


# ── Per-worker state ──────────────────────────────────────────────────────────

_engine: chess.engine.SimpleEngine | None = None
_depth: int = 14


def _worker_init(stockfish_path: str, depth: int) -> None:
    global _engine, _depth
    _engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    _depth = depth


def _label_fen(fen: str) -> dict | None:
    try:
        board = chess.Board(fen)
        info = _engine.analyse(board, chess.engine.Limit(depth=_depth))
        score = info["score"].pov(board.turn)
        cp = float(score.score(mate_score=10000))
        return {"fen": fen, "cp": cp}
    except Exception:
        return None


# ── Position extraction ───────────────────────────────────────────────────────

def extract_fens(line: str) -> list[str]:
    """Walk the full move sequence for one puzzle line, collecting all FENs."""
    parts = line.rstrip("\n").split("\t", 1)
    if len(parts) != 2:
        return []
    fen, moves_str = parts
    moves = moves_str.split()
    if len(moves) < 2:
        return []

    fens: list[str] = []
    board = chess.Board(fen)

    for uci in moves:
        try:
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                break  # illegal move — stop here (corrupt puzzle data)
            board.push(move)
            fens.append(board.fen())
        except Exception:
            break

    return fens


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input",        required=True,  help="TSV file from puzzle_bench --export-all")
    ap.add_argument("--output",       required=True,  help="Output directory for train/val splits")
    ap.add_argument("--stockfish",    required=True,  help="Path to Stockfish binary")
    ap.add_argument("--depth",        type=int, default=14,   help="Stockfish eval depth (default: 14)")
    ap.add_argument("--workers",      type=int, default=max(1, os.cpu_count() - 1), help="Parallel workers")
    ap.add_argument("--val-fraction", type=float, default=0.05, help="Fraction for validation set (default: 0.05)")
    ap.add_argument("--seed",         type=int, default=42)
    args = ap.parse_args()

    inp  = Path(args.input)
    out  = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    if not inp.exists():
        sys.exit(f"Input file not found: {inp}")
    if not Path(args.stockfish).exists():
        sys.exit(f"Stockfish not found: {args.stockfish}")

    name = out.name

    print(f"gen_puzzle_finetune_data")
    print(f"  input:     {inp}  ({inp.stat().st_size // 1024} KB)")
    print(f"  output:    {out}")
    print(f"  stockfish: {args.stockfish}  depth={args.depth}")
    print(f"  workers:   {args.workers}")
    print()

    # ── Step 1: Extract all FENs ──────────────────────────────────────────────

    print("[1/4] Extracting positions from puzzle lines...")
    lines = inp.read_text().splitlines()
    all_fens: list[str] = []
    for line in lines:
        all_fens.extend(extract_fens(line))

    # Deduplicate (same position can appear in multiple puzzles)
    unique_fens = list(dict.fromkeys(all_fens))
    print(f"  {len(lines)} puzzles → {len(all_fens)} positions → {len(unique_fens)} unique FENs")

    # ── Step 2: Label with Stockfish ──────────────────────────────────────────

    # Write to a temp JSONL as results arrive — avoids holding ~10 GB of dicts
    # in memory before the shuffle step.
    tmp_jsonl = out / f"_{name}_labeled.tmp.jsonl"

    print(f"[2/4] Labeling with Stockfish depth={args.depth} ({args.workers} workers)...")
    total_fens = len(unique_fens)
    n_labeled = 0
    # imap_unordered: workers pull items continuously — no batch-boundary stalls.
    # chunksize=128 keeps dispatch overhead low while maintaining fine load balance.
    with Pool(args.workers, initializer=_worker_init, initargs=(args.stockfish, args.depth)) as pool, \
         open(tmp_jsonl, "w") as tmp_f:
        for result in pool.imap_unordered(_label_fen, unique_fens, chunksize=128):
            if result is not None:
                tmp_f.write(json.dumps(result) + "\n")
                n_labeled += 1
            n_seen = n_labeled  # approximate (skipped not counted separately here)
            if n_labeled % 5000 == 0:
                pct = n_labeled * 100 / total_fens
                print(f"\r  {n_labeled}/{total_fens} labeled  ({pct:.1f}%)", end="", flush=True)

    del unique_fens  # free ~3 GB before shuffle
    print(f"\r  {n_labeled} positions labeled ({total_fens - n_labeled} skipped)    ")

    # ── Step 3: Shuffle + split ───────────────────────────────────────────────

    print("[3/4] Shuffling and splitting train/val...")
    # Read back from disk, shuffle, split — peak memory is now just the labeled rows.
    with open(tmp_jsonl) as f:
        lines = f.readlines()
    tmp_jsonl.unlink()  # free disk space

    random.seed(args.seed)
    random.shuffle(lines)

    n_val   = max(1, int(len(lines) * args.val_fraction))
    n_train = len(lines) - n_val
    print(f"  train: {n_train}   val: {n_val}")

    # ── Step 4: Write JSONL ───────────────────────────────────────────────────

    print("[4/4] Writing JSONL files...")
    train_path = out / f"train_{name}.jsonl"
    val_path   = out / f"val_{name}.jsonl"

    with open(train_path, "w") as f:
        f.writelines(lines[:n_train])
    with open(val_path, "w") as f:
        f.writelines(lines[n_train:])

    print(f"  {train_path}  ({train_path.stat().st_size // 1024} KB)")
    print(f"  {val_path}  ({val_path.stat().st_size // 1024} KB)")
    print()
    print("Done. Next steps:")
    print()
    print("  1. Preprocess (binary encode for fast training):")
    print(f"     nnue_preprocess --input {train_path} --output {out}/train_{name} --dual")
    print(f"     nnue_preprocess --input {val_path}   --output {out}/val_{name}   --dual")
    print()
    print("  2. Fine-tune from current checkpoint:")
    print(f"     python3 scripts/train.py --config configs/puzzle_finetune.yaml \\")
    print(f"       --resume artifacts/checkpoint.pt")


if __name__ == "__main__":
    main()

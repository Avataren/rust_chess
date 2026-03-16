#!/usr/bin/env python3
"""Convert JSONL position files to binary sparse-index format for fast training.

Output files:
  {output}.indices.npy  -- shape (N, MAX_ACTIVE) uint16, active feature indices
  {output}.counts.npy   -- shape (N,)            uint8,  number of active features
  {output}.cp.npy       -- shape (N,)             float32, centipawn values

For HalfKP (12,288-dim) positions typically have ~32 active features.
For legacy 12x64 (768-dim) positions have exactly 32 active features.

This replaces per-sample JSON parsing + python-chess encoding with a fast
memmap read + sparse scatter, typically 10-20x faster in the DataLoader.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import chess
import numpy as np
from tqdm import tqdm

import sys, os
sys.path.insert(0, str(Path(__file__).parent.parent))
from nnue_train.features import (
    HALFKP_FEATURE_DIM, FEATURE_DIM,
    encode_board_halfkp, encode_board_12x64,
)

MAX_ACTIVE = 32  # maximum active features per position


def count_lines(path: Path) -> int:
    count = 0
    with open(path, "rb") as f:
        for _ in f:
            count += 1
    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input JSONL file")
    ap.add_argument("--output", required=True, help="Output path prefix (no extension)")
    ap.add_argument("--use-halfkp", action="store_true", default=True,
                    help="Use HalfKP 12,288-dim features (default: true)")
    ap.add_argument("--no-halfkp", dest="use_halfkp", action="store_false")
    ap.add_argument("--max-cp-abs", type=int, default=1500)
    args = ap.parse_args()

    input_path = Path(args.input)
    out_prefix = Path(args.output)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    print(f"Counting lines in {input_path}...")
    N = count_lines(input_path)
    print(f"  {N:,} positions")

    feature_dim = HALFKP_FEATURE_DIM if args.use_halfkp else FEATURE_DIM
    encode_fn = encode_board_halfkp if args.use_halfkp else encode_board_12x64
    print(f"  feature_dim={feature_dim}, use_halfkp={args.use_halfkp}")

    indices_path = str(out_prefix) + ".indices.npy"
    counts_path  = str(out_prefix) + ".counts.npy"
    cp_path      = str(out_prefix) + ".cp.npy"

    indices_arr = np.lib.format.open_memmap(
        indices_path, mode="w+", dtype=np.uint16, shape=(N, MAX_ACTIVE)
    )
    counts_arr = np.lib.format.open_memmap(
        counts_path, mode="w+", dtype=np.uint8, shape=(N,)
    )
    cp_arr = np.lib.format.open_memmap(
        cp_path, mode="w+", dtype=np.float32, shape=(N,)
    )

    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, total=N, desc="encoding")):
            if not line.strip():
                continue
            row = json.loads(line)
            cp = float(row["cp"])
            cp = max(-args.max_cp_abs, min(args.max_cp_abs, cp))
            cp_arr[i] = cp

            board = chess.Board(row["fen"])
            x = encode_fn(board)
            active = np.where(x > 0)[0].astype(np.uint16)
            count = min(len(active), MAX_ACTIVE)
            counts_arr[i] = count
            indices_arr[i, :count] = active[:count]

    # Flush to disk
    del indices_arr, counts_arr, cp_arr

    print(f"Wrote:")
    for p in [indices_path, counts_path, cp_path]:
        size = Path(p).stat().st_size / 1e6
        print(f"  {p}  ({size:.1f} MB)")


if __name__ == "__main__":
    main()

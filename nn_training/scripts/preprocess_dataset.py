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
    encode_board_halfkp, encode_board_12x64, encode_board_halfkp_dual,
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
    ap.add_argument("--max-cp-abs", type=int, default=None,
                    help="Deprecated: cp clipping is applied at training time. This argument "
                         "is accepted for backward compatibility but has no effect.")
    ap.add_argument("--dual", action="store_true", default=False,
                    help="Generate dual-perspective files (white_indices + black_indices). "
                         "CP values are converted to white-absolute convention.")
    args = ap.parse_args()

    input_path = Path(args.input)
    out_prefix = Path(args.output)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    print(f"Counting lines in {input_path}...")
    N = count_lines(input_path)
    print(f"  {N:,} positions")

    if args.dual:
        print(f"  mode=dual (white-absolute cp, two perspective files)")
    else:
        feature_dim = HALFKP_FEATURE_DIM if args.use_halfkp else FEATURE_DIM
        encode_fn = encode_board_halfkp if args.use_halfkp else encode_board_12x64
        print(f"  feature_dim={feature_dim}, use_halfkp={args.use_halfkp}")

    counts_path      = str(out_prefix) + ".counts.npy"
    cp_path          = str(out_prefix) + ".cp.npy"
    piece_count_path = str(out_prefix) + ".piece_count.npy"

    if args.dual:
        white_indices_path = str(out_prefix) + ".white_indices.npy"
        black_indices_path = str(out_prefix) + ".black_indices.npy"
        white_indices_arr = np.lib.format.open_memmap(
            white_indices_path, mode="w+", dtype=np.uint16, shape=(N, MAX_ACTIVE)
        )
        black_indices_arr = np.lib.format.open_memmap(
            black_indices_path, mode="w+", dtype=np.uint16, shape=(N, MAX_ACTIVE)
        )
    else:
        indices_path = str(out_prefix) + ".indices.npy"
        indices_arr = np.lib.format.open_memmap(
            indices_path, mode="w+", dtype=np.uint16, shape=(N, MAX_ACTIVE)
        )

    counts_arr = np.lib.format.open_memmap(
        counts_path, mode="w+", dtype=np.uint8, shape=(N,)
    )
    cp_arr = np.lib.format.open_memmap(
        cp_path, mode="w+", dtype=np.float32, shape=(N,)
    )
    piece_count_arr = np.lib.format.open_memmap(
        piece_count_path, mode="w+", dtype=np.uint8, shape=(N,)
    )

    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, total=N, desc="encoding")):
            if not line.strip():
                continue
            row = json.loads(line)
            cp = float(row["cp"])  # stored raw; clipping applied at training time

            board = chess.Board(row["fen"])

            piece_count = bin(int(board.occupied)).count('1')
            piece_count_arr[i] = piece_count

            if args.dual:
                # Convert cp from side-to-move to white-absolute perspective
                if board.turn == chess.BLACK:
                    cp = -cp
                cp_arr[i] = cp

                w_arr, b_arr = encode_board_halfkp_dual(board)
                count = min(int(board.piece_map().__len__()), MAX_ACTIVE)
                counts_arr[i] = count
                white_indices_arr[i, :count] = w_arr[:count].astype(np.uint16)
                black_indices_arr[i, :count] = b_arr[:count].astype(np.uint16)
            else:
                cp_arr[i] = cp
                x = encode_fn(board)
                active = np.where(x > 0)[0].astype(np.uint16)
                count = min(len(active), MAX_ACTIVE)
                counts_arr[i] = count
                indices_arr[i, :count] = active[:count]

    # Flush to disk
    if args.dual:
        del white_indices_arr, black_indices_arr, counts_arr, cp_arr, piece_count_arr
    else:
        del indices_arr, counts_arr, cp_arr, piece_count_arr

    print(f"Wrote:")
    output_files = [counts_path, cp_path, piece_count_path]
    if args.dual:
        output_files = [white_indices_path, black_indices_path] + output_files
    else:
        output_files = [indices_path] + output_files
    for p in output_files:
        size = Path(p).stat().st_size / 1e6
        print(f"  {p}  ({size:.1f} MB)")


if __name__ == "__main__":
    main()

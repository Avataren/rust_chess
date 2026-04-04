#!/usr/bin/env python3
"""Split a large binary shard into N smaller shards of ~equal size.

Uses mmap for reading to avoid loading the full source into RAM.

Usage (from nn_training/):
  python3 scripts/split_shard.py \
    --input  data/all_69m/train_all_69m \
    --outdir data/all_69m/shards \
    --n-shards 5
"""
from __future__ import annotations
import argparse
import math
from pathlib import Path

import numpy as np
from tqdm import tqdm

EXTENSIONS = [
    (".white_indices.npy", np.int32),
    (".black_indices.npy", np.int32),
    (".cp.npy",            np.float32),
    (".counts.npy",        np.uint8),
    (".piece_count.npy",   np.uint8),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",    required=True, help="Source prefix (no extension)")
    ap.add_argument("--outdir",   required=True, help="Output directory for split shards")
    ap.add_argument("--n-shards", type=int, default=5, help="Number of output shards")
    args = ap.parse_args()

    src = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Determine N from the cp array (1-D, cheapest to mmap)
    cp_src = np.load(str(src) + ".cp.npy", mmap_mode="r")
    N = len(cp_src)
    shard_size = math.ceil(N / args.n_shards)
    print(f"Total positions : {N:,}")
    print(f"Shards          : {args.n_shards}  (~{shard_size:,} each)")
    print(f"Output dir      : {outdir}")

    for shard_idx in range(args.n_shards):
        lo = shard_idx * shard_size
        hi = min(lo + shard_size, N)
        name = f"shard_{shard_idx:02d}"
        print(f"\n[{shard_idx+1}/{args.n_shards}]  {name}  positions {lo:,}–{hi:,}  ({hi-lo:,} rows)")

        for ext, dtype in tqdm(EXTENSIONS, desc=name, leave=False):
            src_file = str(src) + ext
            dst_file = str(outdir / name) + ext
            arr = np.load(src_file, mmap_mode="r")
            chunk = np.array(arr[lo:hi], dtype=dtype)  # copies slice to RAM
            np.save(dst_file, chunk)
            size_mb = Path(dst_file).stat().st_size / 1e6
            tqdm.write(f"  wrote {dst_file}  ({size_mb:.0f} MB)")

    print(f"\nDone. {args.n_shards} shards written to {outdir}/")


if __name__ == "__main__":
    main()

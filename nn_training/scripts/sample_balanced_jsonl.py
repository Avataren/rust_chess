#!/usr/bin/env python3
"""
sample_balanced_jsonl.py — Bucket-balanced sampling from a labeled JSONL dataset.

Uses a pre-computed piece_count.npy sidecar (same basename, same directory) to
avoid re-parsing FENs.  Falls back to FEN parsing when the .npy is absent.

Bucket formula matches the model:  bucket = ((piece_count - 2) * 8 // 30).clamp(0, 7)

Usage examples:

  # Sample 1M per bucket (8M total) from puzzle train set
  python scripts/sample_balanced_jsonl.py \\
      --input  data/puzzle_ft/train_puzzle_ft.jsonl \\
      --output data/puzzle_ft/train_puzzle_ft_8m_balanced.jsonl \\
      --per-bucket 1000000

  # Sample a specific total (distributed equally across buckets)
  python scripts/sample_balanced_jsonl.py \\
      --input  data/puzzle_ft/train_puzzle_ft.jsonl \\
      --output data/puzzle_ft/train_puzzle_ft_8m_balanced.jsonl \\
      --total  8000000

  # After sampling, merge with game data and re-encode:
  #   cat data/all_69m/train_all_69m.jsonl train_puzzle_ft_8m_balanced.jsonl | shuf > data/all_77m/train_all_77m.jsonl
  #   nnue_preprocess --input ... --output ... --dual
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm


# ── Bucket helpers ────────────────────────────────────────────────────────────

BUCKET_RANGES = [(2,5),(6,9),(10,13),(14,16),(17,20),(21,24),(25,28),(29,32)]

def piece_count_from_fen(fen: str) -> int:
    return sum(1 for c in fen.split()[0] if c.isalpha())

def bucket_from_pc(pc: int) -> int:
    return min(7, max(0, (pc - 2) * 8 // 30))


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Bucket-balanced sampling from a labeled JSONL dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--input",  "-i", required=True, help="Input JSONL file")
    ap.add_argument("--output", "-o", required=True, help="Output JSONL file")

    size_group = ap.add_mutually_exclusive_group(required=True)
    size_group.add_argument("--per-bucket", type=int, metavar="N",
                            help="Maximum positions to sample per bucket")
    size_group.add_argument("--total", type=int, metavar="N",
                            help="Total positions to sample (distributed equally across buckets)")

    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    inp  = Path(args.input)
    out  = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    per_bucket = args.per_bucket if args.per_bucket else args.total // 8

    # ── Step 1: Load or compute piece counts ─────────────────────────────────

    pc_npy = inp.with_suffix("").with_suffix(".piece_count.npy")
    if pc_npy.exists():
        print(f"Loading piece counts from {pc_npy.name} ...")
        piece_counts = np.load(pc_npy, mmap_mode="r").astype(np.int32)
        n_total = len(piece_counts)
        buckets = np.clip((piece_counts - 2) * 8 // 30, 0, 7)
    else:
        print(f"piece_count.npy not found — scanning FENs (slower)...")
        pcs = []
        with inp.open() as f:
            for line in tqdm(f, desc="scanning"):
                try:
                    fen = json.loads(line)["fen"]
                    pcs.append(piece_count_from_fen(fen))
                except Exception:
                    pcs.append(0)
        piece_counts = np.array(pcs, dtype=np.int32)
        buckets = np.clip((piece_counts - 2) * 8 // 30, 0, 7)
        n_total = len(buckets)

    # ── Step 2: Sample indices per bucket ────────────────────────────────────

    print(f"\nDataset: {n_total:,} positions")
    print(f"Target : {per_bucket:,} per bucket  ({per_bucket * 8:,} total)\n")

    selected_indices: set[int] = set()
    bucket_counts: dict[int, int] = {}

    for b in range(8):
        lo, hi = BUCKET_RANGES[b]
        idxs = np.where(buckets == b)[0]
        available = len(idxs)
        n = min(per_bucket, available)
        chosen = np.random.choice(idxs, size=n, replace=False)
        selected_indices.update(chosen.tolist())
        bucket_counts[b] = n
        bar = "█" * int(30 * n / max(per_bucket, 1))
        print(f"  bucket {b} (pc {lo:2d}-{hi:2d}): {available:>8,} available  →  sampled {n:>7,}  {bar}")

    total_selected = len(selected_indices)
    print(f"\n  Total selected: {total_selected:,}")

    # ── Step 3: Stream JSONL, writing selected lines ──────────────────────────

    print(f"\nWriting {out} ...")
    written = 0
    with inp.open() as fin, out.open("w") as fout:
        for idx, line in enumerate(tqdm(fin, total=n_total, desc="streaming")):
            if idx in selected_indices:
                fout.write(line)
                written += 1

    print(f"Done — wrote {written:,} positions to {out}")
    print(f"\nNext steps:")
    print(f"  Merge with game data:")
    base = out.stem
    print(f"    cat <game_train.jsonl> {out.name} | shuf > <merged_train.jsonl>")
    print(f"  Then binary encode the merged file with nnue_preprocess --dual")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--val", default="data/val.jsonl")
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    lines = Path(args.input).read_text(encoding="utf-8").splitlines()
    random.shuffle(lines)
    total = len(lines)
    if total == 0:
        n_val = 0
    else:
        n_val = int(total * args.val_ratio)
        n_val = max(1, n_val)
        if total > 1:
            n_val = min(n_val, total - 1)

    val = lines[:n_val]
    train = lines[n_val:]

    Path(args.train).parent.mkdir(parents=True, exist_ok=True)
    Path(args.val).parent.mkdir(parents=True, exist_ok=True)
    Path(args.train).write_text("\n".join(train) + "\n", encoding="utf-8")
    Path(args.val).write_text("\n".join(val) + "\n", encoding="utf-8")
    print(f"train={len(train)} val={len(val)}")


if __name__ == "__main__":
    main()

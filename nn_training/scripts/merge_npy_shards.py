#!/usr/bin/env python3
"""Merge multiple binary dataset shards (.npy files) into a single dataset.

Usage:
    python3 scripts/merge_npy_shards.py --shards data/shards/ --output data/t60_merged
    python3 scripts/merge_npy_shards.py --shards data/shards/shard_0 data/shards/shard_1 --output data/merged

Each shard directory must contain files produced by preprocess_dataset.py:
    {prefix}.white_indices.npy, {prefix}.black_indices.npy,
    {prefix}.counts.npy, {prefix}.cp.npy, {prefix}.piece_count.npy

The output directory will contain identically named merged files.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def find_shard_prefixes(shard_paths: list[Path]) -> list[Path]:
    """Given a list of paths (files or dirs), return a sorted list of .cp.npy file stems."""
    prefixes = []
    for p in shard_paths:
        if p.is_dir():
            for cp_file in sorted(p.glob("*.cp.npy")):
                prefixes.append(cp_file.with_suffix("").with_suffix(""))
        elif p.suffix == ".npy" and ".cp" in p.name:
            prefixes.append(p.with_suffix("").with_suffix(""))
        else:
            # Treat as a prefix directly
            prefixes.append(p)
    return prefixes


def merge_shards(prefixes: list[Path], output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, list[np.ndarray]] = {
        "white_indices": [],
        "black_indices": [],
        "counts": [],
        "cp": [],
        "piece_count": [],
    }

    total = 0
    for pfx in prefixes:
        wi = pfx.parent / (pfx.name + ".white_indices.npy")
        bi = pfx.parent / (pfx.name + ".black_indices.npy")
        co = pfx.parent / (pfx.name + ".counts.npy")
        cp = pfx.parent / (pfx.name + ".cp.npy")
        pc = pfx.parent / (pfx.name + ".piece_count.npy")

        missing = [f for f in [wi, bi, co, cp] if not f.exists()]
        if missing:
            print(f"  SKIP {pfx.name}: missing {[f.name for f in missing]}", file=sys.stderr)
            continue

        n = len(np.load(cp, mmap_mode="r"))
        print(f"  shard {pfx.name}: {n:,} positions", flush=True)
        total += n

        arrays["white_indices"].append(np.load(wi, mmap_mode="r"))
        arrays["black_indices"].append(np.load(bi, mmap_mode="r"))
        arrays["counts"].append(np.load(co, mmap_mode="r"))
        arrays["cp"].append(np.load(cp, mmap_mode="r"))
        if pc.exists():
            arrays["piece_count"].append(np.load(pc, mmap_mode="r"))
        else:
            # Fallback: use counts as piece_count proxy
            arrays["piece_count"].append(np.load(co, mmap_mode="r"))

    print(f"\nMerging {len(prefixes)} shards → {total:,} total positions ...", flush=True)

    out = str(output_prefix)
    np.save(out + ".white_indices.npy", np.concatenate(arrays["white_indices"], axis=0))
    print("  white_indices done", flush=True)
    np.save(out + ".black_indices.npy", np.concatenate(arrays["black_indices"], axis=0))
    print("  black_indices done", flush=True)
    np.save(out + ".counts.npy",        np.concatenate(arrays["counts"],        axis=0))
    np.save(out + ".cp.npy",            np.concatenate(arrays["cp"],            axis=0))
    np.save(out + ".piece_count.npy",   np.concatenate(arrays["piece_count"],   axis=0))
    print(f"  Done → {out}.*.npy  ({total:,} positions)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge binary dataset shards into one.")
    ap.add_argument("--shards", nargs="+", required=True,
                    help="Shard directory/directories or prefix paths to merge.")
    ap.add_argument("--output", required=True,
                    help="Output prefix (e.g. data/t60_merged/train).")
    args = ap.parse_args()

    shard_paths = [Path(p) for p in args.shards]
    prefixes = find_shard_prefixes(shard_paths)
    if not prefixes:
        print("No shards found.", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(prefixes)} shard prefix(es).")
    merge_shards(prefixes, Path(args.output))


if __name__ == "__main__":
    main()

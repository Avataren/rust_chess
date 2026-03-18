#!/usr/bin/env python3
"""Analyze cp distribution of a preprocessed binary dataset.

Loads the raw cp.npy file and prints statistics to help choose max_cp_abs
for training.  Optionally plots a histogram.

Usage:
  python scripts/analyze_dataset.py data/train_d14_20m.jsonl
  python scripts/analyze_dataset.py data/train_d14_20m.jsonl --plot
  python scripts/analyze_dataset.py data/train_d14_20m.jsonl --both-splits
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


# ── WDL draw probability (mirrors cp_to_wdl_target) ─────────────────────────

def draw_prob(cp: float) -> float:
    return max(0.0, 1.0 - abs(cp) / 800.0)


# ── Analysis ─────────────────────────────────────────────────────────────────

def analyze(cp_raw: np.ndarray, label: str) -> None:
    n = len(cp_raw)
    abs_cp = np.abs(cp_raw)

    print(f"\n{'═' * 60}")
    print(f"  {label}  ({n:,} positions)")
    print(f"{'═' * 60}")

    # Basic stats
    print(f"\n  Basic statistics")
    print(f"  {'mean':>10}  {'std':>10}  {'min':>10}  {'max':>10}")
    print(f"  {cp_raw.mean():>10.1f}  {cp_raw.std():>10.1f}  {cp_raw.min():>10.1f}  {cp_raw.max():>10.1f}")

    # Percentiles of |cp|
    pcts = [50, 75, 90, 95, 99, 99.9]
    print(f"\n  Percentiles of |cp|")
    print(f"  {'p50':>8}  {'p75':>8}  {'p90':>8}  {'p95':>8}  {'p99':>8}  {'p99.9':>8}")
    vals = [float(np.percentile(abs_cp, p)) for p in pcts]
    print("  " + "  ".join(f"{v:>8.1f}" for v in vals))

    # Coverage at candidate thresholds
    thresholds = [200, 300, 400, 500, 600, 800, 1000, 1200, 1500]
    print(f"\n  Coverage at candidate max_cp_abs thresholds")
    print(f"  {'threshold':>10}  {'positions kept':>16}  {'% kept':>8}  {'draw_prob at threshold':>24}")
    for t in thresholds:
        kept = int((abs_cp <= t).sum())
        pct = 100.0 * kept / n
        dp = draw_prob(t)
        marker = "  ◀ regression ceiling (draw=0 beyond here)" if t == 800 else ""
        print(f"  {t:>10}  {kept:>16,}  {pct:>7.2f}%  {dp:>24.3f}{marker}")

    # Positions with |cp| > 800 (draw prob = 0, fully decided)
    fully_decided = int((abs_cp > 800).sum())
    print(f"\n  Fully decided (|cp| > 800, draw=0):  {fully_decided:,}  ({100.*fully_decided/n:.1f}%)")

    # Suggest max_cp_abs
    # Upper bound: 800cp — draw probability hits zero here, so regression beyond
    # this is redundant (the WDL head already handles fully-decided positions).
    # Within that bound, use p90 so that 90% of positions contribute to regression.
    WDL_CUTOFF = 800
    p90 = float(np.percentile(abs_cp, 90))
    suggested = round(min(WDL_CUTOFF, p90) / 50) * 50
    suggested = max(200, suggested)  # floor at 200
    kept_pct = 100.0 * (abs_cp <= suggested).sum() / n

    print(f"\n  {'─' * 56}")
    print(f"  Recommended max_cp_abs: {suggested}")
    print(f"    p90 of |cp| = {p90:.0f}  →  min({p90:.0f}, {WDL_CUTOFF}) = {min(WDL_CUTOFF, p90):.0f}"
          f"  →  rounded to {suggested}")
    print(f"    {kept_pct:.1f}% of positions used for cp regression loss")
    print(f"    100.0% of positions always used for WDL loss (raw cp, unclipped)")
    print(f"  {'─' * 56}")

    return suggested


def plot_histogram(cp_raw: np.ndarray, label: str, suggested: int) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  matplotlib not available — skipping plot (pip install matplotlib)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(label)

    # Full distribution
    ax = axes[0]
    ax.hist(cp_raw, bins=200, range=(-3000, 3000), color="steelblue", alpha=0.7, edgecolor="none")
    ax.axvline( suggested, color="red",    linestyle="--", label=f"+max_cp_abs ({suggested})")
    ax.axvline(-suggested, color="red",    linestyle="--")
    ax.axvline( 800,       color="orange", linestyle=":",  label="+800 (draw=0)")
    ax.axvline(-800,       color="orange", linestyle=":")
    ax.set_xlabel("cp (centipawns)")
    ax.set_ylabel("count")
    ax.set_title("cp distribution (±3000)")
    ax.legend()

    # Zoomed ±1500
    ax = axes[1]
    ax.hist(cp_raw, bins=150, range=(-1500, 1500), color="steelblue", alpha=0.7, edgecolor="none")
    ax.axvline( suggested, color="red",    linestyle="--", label=f"+max_cp_abs ({suggested})")
    ax.axvline(-suggested, color="red",    linestyle="--")
    ax.axvline( 800,       color="orange", linestyle=":",  label="+800 (draw=0)")
    ax.axvline(-800,       color="orange", linestyle=":")
    ax.set_xlabel("cp (centipawns)")
    ax.set_title("cp distribution (±1500 zoom)")
    ax.legend()

    plt.tight_layout()
    out = Path(label.split()[0]).stem + "_cp_hist.png"
    plt.savefig(out, dpi=120)
    print(f"\n  Histogram saved to: {out}")
    plt.show()


def load_cp(path: str) -> tuple[np.ndarray, str]:
    """Load raw cp values from binary .cp.npy or JSONL."""
    p = Path(path)
    cp_path = p.with_suffix("").with_suffix("") if p.suffix == ".jsonl" else p.with_suffix("")
    npy = Path(str(cp_path) + ".cp.npy")

    if npy.exists():
        print(f"  Loading {npy}")
        return np.load(npy, mmap_mode="r").astype(np.float32), str(npy)

    # Fallback: parse JSONL directly (slow)
    if p.exists() and p.suffix == ".jsonl":
        print(f"  No .cp.npy found — reading JSONL (slow): {p}")
        import json
        values = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    values.append(float(json.loads(line)["cp"]))
        return np.array(values, dtype=np.float32), str(p)

    sys.exit(f"Cannot find cp data for: {path}\n"
             f"  Expected: {npy}\n"
             f"  Run preprocess_dataset.py first, or pass the .jsonl directly.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Analyze cp distribution to choose max_cp_abs for training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("dataset", help="Path to train JSONL or binary prefix "
                                    "(e.g. data/train_d14_20m.jsonl)")
    ap.add_argument("--val", default=None,
                    help="Also analyze the val split (auto-detected if omitted "
                         "and a val_ counterpart exists).")
    ap.add_argument("--both-splits", action="store_true",
                    help="Auto-detect and analyze both train and val splits.")
    ap.add_argument("--plot", action="store_true",
                    help="Save and show a histogram (requires matplotlib).")
    args = ap.parse_args()

    paths = [args.dataset]

    if args.both_splits or args.val:
        if args.val:
            paths.append(args.val)
        else:
            # Auto-detect: train_X → val_X
            p = Path(args.dataset)
            if p.name.startswith("train_"):
                val_candidate = p.parent / p.name.replace("train_", "val_", 1)
                if val_candidate.exists():
                    paths.append(str(val_candidate))
                else:
                    # try .cp.npy
                    prefix = str(Path(str(val_candidate).replace(".jsonl", "")))
                    if Path(prefix + ".cp.npy").exists():
                        paths.append(str(val_candidate))
                    else:
                        print(f"  Could not auto-detect val split for {p.name}")

    for path in paths:
        cp_raw, label = load_cp(path)
        suggested = analyze(cp_raw, label)

        if args.plot:
            plot_histogram(cp_raw, label, suggested)

    print()


if __name__ == "__main__":
    main()

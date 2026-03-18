#!/usr/bin/env python3
"""
End-to-end dataset generation pipeline.

Steps:
  1. Extract FEN positions from a PGN file  (Rust pgn_extract, ~27k pos/sec)
  2. Label positions with Stockfish in parallel  (generate_data.py)
  3. Shuffle and split 90/10 into train/val JSONL
  4. Pre-encode binary: single-perspective (.indices.npy) and dual (.white/black_indices.npy)
     — train and val encoded in parallel

Usage examples:

  # Full pipeline from a Lichess PGN dump
  python scripts/make_dataset.py \\
    --positions 10_000_000 \\
    --pgn /data/lichess_2024-01.pgn \\
    --depth 12

  # Skip extraction — label a pre-existing FENs file, 32 workers
  python scripts/make_dataset.py \\
    --positions 10_000_000 \\
    --fens data/fens_10m.txt \\
    --depth 14 \\
    --workers 32

  # Only redo the binary preprocessing after relabeling at a new depth
  python scripts/make_dataset.py \\
    --positions 10_000_000 \\
    --name my_10m_d14 \\
    --skip-extract --skip-label --skip-split \\
    --depth 14
"""
from __future__ import annotations

import argparse
import multiprocessing
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
NN_ROOT   = REPO_ROOT / "nn_training"


# ── Subprocess helpers ────────────────────────────────────────────────────────

def run(cmd: list, *, desc: str, extra_env: dict | None = None) -> None:
    """Run a command, stream output live, abort on failure."""
    print(f"\n[make_dataset] ▶  {desc}")
    print(f"  $ {' '.join(str(c) for c in cmd)}\n")
    env = {**os.environ, **(extra_env or {})}
    t0 = time.monotonic()
    result = subprocess.run(cmd, env=env)
    elapsed = time.monotonic() - t0
    if result.returncode != 0:
        sys.exit(f"[make_dataset] FAILED (exit {result.returncode}) after {elapsed:.1f}s")
    print(f"\n[make_dataset] ✓  {elapsed:.1f}s")


def run_parallel(jobs: list[tuple[list, str, dict | None]]) -> None:
    """Run multiple commands in parallel; redirect output to log files to avoid
    interleaved tqdm bars; poll and print a single status line from the parent."""
    import tempfile

    entries = []  # (proc, desc, log_path)
    t0 = time.monotonic()

    for cmd, desc, extra_env in jobs:
        env = {**os.environ, **(extra_env or {})}
        log_fd, log_path = tempfile.mkstemp(suffix=".log", prefix="make_dataset_")
        log_file = os.fdopen(log_fd, "w")
        print(f"[make_dataset] ▶  {desc}")
        print(f"  $ {' '.join(str(c) for c in cmd)}")
        print(f"  log → {log_path}")
        proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=log_file)
        log_file.close()
        entries.append((proc, desc, log_path))

    print()
    # Poll until all done, printing a single updating status line
    while any(p.poll() is None for p, _, _ in entries):
        elapsed = time.monotonic() - t0
        still_running = [d for p, d, _ in entries if p.poll() is None]
        short = [d.split("(")[-1].rstrip(")") if "(" in d else d for d in still_running]
        print(f"\r  [{elapsed:5.0f}s]  running: {', '.join(short)}   ", end="", flush=True)
        time.sleep(2)

    print()  # newline after the status line

    # Report results
    failed = []
    for proc, desc, log_path in entries:
        if proc.returncode != 0:
            print(f"\n[make_dataset] FAILED: {desc!r}  (exit {proc.returncode})")
            try:
                lines = Path(log_path).read_text().splitlines()
                print(f"  Last lines of {log_path}:")
                for line in lines[-20:]:
                    print(f"    {line}")
            except OSError:
                pass
            failed.append(desc)
        else:
            Path(log_path).unlink(missing_ok=True)

    if failed:
        for proc, _, _ in entries:
            if proc.poll() is None:
                proc.terminate()
        sys.exit(f"[make_dataset] {len(failed)} job(s) failed")

    print(f"[make_dataset] ✓  all parallel jobs done in {time.monotonic()-t0:.1f}s")


def count_lines(path: Path) -> int:
    return int(subprocess.check_output(["wc", "-l", str(path)]).split()[0])


def find_binary(name: str, candidates: list[str]) -> str:
    for c in candidates:
        if Path(c).exists():
            return c
    found = shutil.which(name)
    if found:
        return found
    sys.exit(
        f"[make_dataset] Cannot find '{name}'.\n"
        f"  Tried: {candidates}\n"
        f"  Build: cargo build --release -p {name}"
    )


# ── Shuffle + split ───────────────────────────────────────────────────────────

def shuffle_and_split(src: Path, train_out: Path, val_out: Path, val_frac: float) -> None:
    total   = count_lines(src)
    n_val   = max(1, int(total * val_frac))
    n_train = total - n_val
    print(f"\n[make_dataset] ▶  Shuffle + split  ({total:,} → {n_train:,} train / {n_val:,} val)")

    tmp = src.with_suffix(".shuffled.tmp")
    try:
        print(f"  $ shuf {src}  (may take ~30s for 10M lines)")
        t0 = time.monotonic()
        with open(tmp, "w") as f:
            subprocess.run(["shuf", str(src)], stdout=f, check=True)
        print(f"  shuffle done in {time.monotonic()-t0:.1f}s")

        with open(train_out, "w") as f:
            subprocess.run(["head", "-n", str(n_train), str(tmp)], stdout=f, check=True)
        with open(val_out, "w") as f:
            subprocess.run(["tail", "-n", str(n_val), str(tmp)], stdout=f, check=True)
    finally:
        tmp.unlink(missing_ok=True)

    print(f"  → {train_out}  ({n_train:,})")
    print(f"  → {val_out}  ({n_val:,})")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    n_cpu = multiprocessing.cpu_count()
    default_workers = max(1, n_cpu // 2)

    ap = argparse.ArgumentParser(
        description="Generate a labeled chess position dataset end-to-end.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Required ──
    ap.add_argument(
        "--positions", "-n",
        type=lambda s: int(s.replace("_", "").replace(",", "")),
        required=True,
        metavar="N",
        help="Number of positions to extract and label (e.g. 10_000_000).",
    )

    # ── Stockfish ──
    ap.add_argument(
        "--depth", "-d", type=int, default=12,
        help="Stockfish evaluation depth per position (default: 12). "
             "Depth 8 ≈ 4× faster; depth 16 ≈ 4× slower than depth 12.",
    )
    ap.add_argument(
        "--workers", "-w", type=int, default=default_workers,
        help=f"Parallel Stockfish worker processes (default: {default_workers}).",
    )
    ap.add_argument(
        "--stockfish", default=None,
        help="Path to Stockfish binary (auto-detected if omitted).",
    )

    # ── Position source ──
    source = ap.add_mutually_exclusive_group()
    source.add_argument(
        "--pgn", default=None,
        help="Input PGN file — run pgn_extract to produce FENs.",
    )
    source.add_argument(
        "--fens", default=None,
        help="Pre-extracted FENs file (one FEN per line) — skips pgn_extract.",
    )

    # ── Filtering ──
    ap.add_argument("--min-elo", type=int, default=0,
                    help="Minimum Elo for pgn_extract (default: 0 = all).")
    ap.add_argument("--positions-per-game", type=int, default=1,
                    help="Positions sampled per game in pgn_extract (default: 1).")

    # ── Output ──
    ap.add_argument(
        "--output-dir", default=str(NN_ROOT / "data"),
        help="Output directory (default: nn_training/data).",
    )
    ap.add_argument(
        "--name", default=None,
        help="Base name for output files (default: auto from --positions and --depth).",
    )
    ap.add_argument(
        "--val-fraction", type=float, default=0.1,
        help="Fraction of data held out for validation (default: 0.1).",
    )

    # ── Pipeline control ──
    ap.add_argument("--skip-extract",    action="store_true",
                    help="Skip FEN extraction (reuse existing fens file).")
    ap.add_argument("--skip-label",      action="store_true",
                    help="Skip Stockfish labeling (reuse existing JSONL).")
    ap.add_argument("--skip-split",      action="store_true",
                    help="Skip shuffle/split (train/val JSONL already exist).")
    ap.add_argument("--skip-preprocess", action="store_true",
                    help="Skip binary pre-encoding step.")
    ap.add_argument("--no-dual",         action="store_true",
                    help="Skip dual-perspective encoding (single-perspective only).")
    ap.add_argument(
        "--pgn-extract-bin", default=None,
        help="Path to pgn_extract binary (auto-detected if omitted).",
    )

    args = ap.parse_args()

    # ── Resolve file paths ────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n    = args.positions
    d    = args.depth
    name = args.name or f"dataset_{n // 1_000_000}m_d{d}"

    fens_path    = Path(args.fens)    if args.fens    else out_dir / f"{name}.fens.txt"
    labeled_path = out_dir / f"{name}.jsonl"
    train_path   = out_dir / f"train_{name}.jsonl"
    val_path     = out_dir / f"val_{name}.jsonl"

    python     = sys.executable
    pythonpath = str(NN_ROOT)

    print(f"""
[make_dataset] ════════════════════════════════════════
  positions  : {n:,}
  eval depth : {d}
  workers    : {args.workers}
  name       : {name}
  output dir : {out_dir}
  dual enc.  : {"no" if args.no_dual else "yes"}
[make_dataset] ════════════════════════════════════════""")

    # ── Step 1: Extract FENs ──────────────────────────────────────────────────
    if not args.skip_extract:
        if args.fens:
            print(f"\n[make_dataset] Using existing FENs: {fens_path}")
        elif args.pgn:
            pgn_extract = args.pgn_extract_bin or find_binary(
                "pgn_extract",
                [
                    str(REPO_ROOT / "target" / "release" / "pgn_extract"),
                    str(REPO_ROOT / "target" / "debug"   / "pgn_extract"),
                ],
            )
            cmd = [
                pgn_extract,
                "--input",              str(args.pgn),
                "--output",             str(fens_path),
                "--max-positions",      str(n),
                "--positions-per-game", str(args.positions_per_game),
            ]
            if args.min_elo:
                cmd += ["--min-elo", str(args.min_elo)]
            run(cmd, desc=f"Extract {n:,} FENs from PGN")
        else:
            sys.exit(
                "[make_dataset] ERROR: specify --pgn <file> or --fens <file> "
                "(or --skip-extract if the FENs already exist)."
            )
    else:
        if not fens_path.exists() and not args.skip_label:
            sys.exit(f"[make_dataset] --skip-extract set but {fens_path} not found.")
        print(f"\n[make_dataset] Skipping extraction — using {fens_path}")

    # ── Step 2: Label with Stockfish ──────────────────────────────────────────
    if not args.skip_label:
        stockfish = args.stockfish or find_binary(
            "stockfish",
            ["/usr/bin/stockfish", "/usr/local/bin/stockfish"],
        )
        n_fens = count_lines(fens_path)
        approx_rate = args.workers * {8: 80, 12: 20, 14: 10, 16: 5}.get(d, 15)
        approx_min  = n_fens // approx_rate // 60 if approx_rate else "?"
        print(f"\n[make_dataset]  {n_fens:,} FENs to label at depth {d}")
        print(f"  Estimated time: ~{approx_min} minutes with {args.workers} workers")

        run(
            [
                python,
                str(NN_ROOT / "scripts" / "generate_data.py"),
                "--label-engine",   stockfish,
                "--fens",           str(fens_path),
                "--output",         str(labeled_path),
                "--eval-depth",     str(d),
                "--workers",        str(args.workers),
                "--selfplay-games", "0",
                "--max-positions",  str(n),
            ],
            desc=f"Label {n_fens:,} positions  depth={d}  workers={args.workers}",
            extra_env={"PYTHONPATH": pythonpath},
        )
    else:
        if not labeled_path.exists():
            sys.exit(f"[make_dataset] --skip-label set but {labeled_path} not found.")
        print(f"\n[make_dataset] Skipping labeling — using {labeled_path}")

    # ── Step 3: Shuffle + split ───────────────────────────────────────────────
    if not args.skip_split:
        shuffle_and_split(labeled_path, train_path, val_path, args.val_fraction)
    else:
        for p in [train_path, val_path]:
            if not p.exists():
                sys.exit(f"[make_dataset] --skip-split set but {p} not found.")
        print(f"\n[make_dataset] Skipping split — using {train_path} + {val_path}")

    # ── Step 4: Binary pre-encoding ───────────────────────────────────────────
    if not args.skip_preprocess:
        # Prefer the Rust binary (20-50× faster, no mmap → no SIGBUS).
        rust_bin = None
        for candidate in [
            REPO_ROOT / "target" / "release" / "nnue_preprocess",
            REPO_ROOT / "target" / "debug"   / "nnue_preprocess",
        ]:
            if candidate.exists():
                rust_bin = candidate
                break

        if rust_bin:
            print(f"[make_dataset] Using Rust encoder: {rust_bin}")
        else:
            print("[make_dataset] Rust encoder not found — falling back to Python.")
            print("  Build with: cargo build --release -p nnue_preprocess")

        preprocess_script = str(NN_ROOT / "scripts" / "preprocess_dataset.py")

        jobs = []
        for split_path in [train_path, val_path]:
            out_prefix = str(split_path.with_suffix(""))  # strip .jsonl
            label = split_path.stem

            if rust_bin:
                # Single-perspective
                jobs.append((
                    [str(rust_bin),
                     "--input", str(split_path),
                     "--output", out_prefix],
                    f"Encode {label} (single-perspective HalfKP)",
                    None,
                ))
                # Dual-perspective
                if not args.no_dual:
                    jobs.append((
                        [str(rust_bin),
                         "--input", str(split_path),
                         "--output", out_prefix,
                         "--dual"],
                        f"Encode {label} (dual-perspective)",
                        None,
                    ))
            else:
                # Python fallback
                jobs.append((
                    [python, preprocess_script,
                     "--input", str(split_path),
                     "--output", out_prefix,
                     "--use-halfkp"],
                    f"Encode {label} (single-perspective HalfKP)",
                    {"PYTHONPATH": pythonpath},
                ))
                if not args.no_dual:
                    jobs.append((
                        [python, preprocess_script,
                         "--input", str(split_path),
                         "--output", out_prefix,
                         "--dual"],
                        f"Encode {label} (dual-perspective)",
                        {"PYTHONPATH": pythonpath},
                    ))

        print(f"\n[make_dataset] ▶  Binary encoding  ({len(jobs)} jobs in parallel)")
        run_parallel(jobs)

    # ── Done ──────────────────────────────────────────────────────────────────
    print(f"\n[make_dataset] ═══════════════ Done ═══════════════")
    print(f"\n  Files written to {out_dir}/:")
    for f in sorted(out_dir.glob(f"*{name}*")):
        print(f"    {f.name:<55}  {f.stat().st_size / 1e6:>7.0f} MB")

    print(f"""
  Next steps:

  1. Edit configs/halfkp_dual.yaml:
       data:
         train_file: data/train_{name}.jsonl
         val_file:   data/val_{name}.jsonl

  2. Train:
       PYTHONPATH={pythonpath} python scripts/train.py \\
         --config configs/halfkp_dual.yaml \\
         --out artifacts/halfkp_dual_{name}.pt

  3. Export:
       PYTHONPATH={pythonpath} python scripts/export_weights.py \\
         --checkpoint artifacts/halfkp_dual_{name}.pt \\
         --config configs/halfkp_dual.yaml \\
         --output artifacts/eval_dual_{name}.npz
""")


if __name__ == "__main__":
    main()

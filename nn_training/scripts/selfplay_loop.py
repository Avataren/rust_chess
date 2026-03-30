#!/usr/bin/env python3
"""
Continuous self-play improvement loop.

Each iteration:
  1. Export current best checkpoint → NPZ weights
  2. Generate self-play games using the neural engine
  3. Label positions with Stockfish
  4. Append new data to the replay pool (capped at --pool-size)
  5. Fine-tune from the current best checkpoint for N epochs
  6. If val_cp_mae improved → promote to best; else discard

Usage:
  PYTHONPATH=. python3 scripts/selfplay_loop.py \
    --engine ../../target/release/chess_uci \
    --stockfish /usr/bin/stockfish \
    --initial-checkpoint artifacts/checkpoint_1m.pt
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_val_mae(checkpoint_path: Path) -> float:
    """Read val_cp_mae from a checkpoint's stored metrics."""
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # Stored as val_loss (combined); fall back to large sentinel if missing.
    return float(ck.get("val_cp_mae", ck.get("val_loss", 9999.0)))


def export_weights(checkpoint_path: Path, npz_path: Path):
    print(f"[loop] Exporting {checkpoint_path} → {npz_path}")
    subprocess.run(
        [sys.executable, "scripts/export_weights.py",
         "--checkpoint", str(checkpoint_path),
         "--output", str(npz_path)],
        check=True,
    )


def gen_val_mae(checkpoint_path: Path, gen_val_file: Path) -> float:
    """Compute CP-MAE of a checkpoint on the fixed generalisation validation set.

    Returns -1.0 if the checkpoint or data file is missing.
    This measures whether improvements generalise to real-game positions,
    not just to the self-play distribution.
    """
    if not checkpoint_path or not checkpoint_path.exists():
        return -1.0
    if not gen_val_file or not gen_val_file.exists():
        return -1.0
    result = subprocess.run(
        [sys.executable, "scripts/eval_mae.py",
         "--checkpoint", str(checkpoint_path),
         "--data",       str(gen_val_file)],
        capture_output=True, text=True,
    )
    for line in result.stdout.splitlines():
        if line.startswith("cp_mae="):
            try:
                return float(line.split("=")[1])
            except ValueError:
                pass
    return -1.0


def sample_gen_val(source_file: Path, out_file: Path, n: int, seed: int = 0):
    """Draw a fixed random sample from source_file and write to out_file.

    Called once at startup — same seed means same positions every run,
    so the generalisation metric is comparable across all iterations.
    """
    import random as _random
    rng = _random.Random(seed)
    lines = source_file.read_text(encoding="utf-8").splitlines(keepends=True)
    sample = rng.sample(lines, min(n, len(lines)))
    out_file.write_text("".join(sample), encoding="utf-8")
    print(f"[loop] Generalisation val set: {len(sample)} positions → {out_file}")


def puzzle_score(bench_binary: str, puzzle_file: str, npz_path: Path,
                 count: int, depth: int, seed: int,
                 min_rating: int = 0, max_rating: int = 0,
                 export_failures_file: str = "") -> float:
    """Run puzzle_bench and return the overall solve rate (0.0–100.0).
    Returns -1.0 if the binary or puzzle file is not available.
    When export_failures_file is set, puzzle_bench writes a TSV of failed puzzles
    that can be processed into training data via gen_puzzle_finetune_data.py."""
    if not bench_binary or not puzzle_file:
        return -1.0
    cmd = [bench_binary,
           "--file",      puzzle_file,
           "--eval-file", str(npz_path),
           "--count",     str(count),
           "--depth",     str(depth),
           "--seed",      str(seed),
           "--threads",   "0"]  # 0 = use all available CPUs
    if min_rating > 0:
        cmd += ["--min-rating", str(min_rating)]
    if max_rating > 0:
        cmd += ["--max-rating", str(max_rating)]
    if export_failures_file:
        cmd += ["--export-failures", export_failures_file]
    result = subprocess.run(cmd, capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if "Overall:" in line:
            # Format: "  Overall:  solved/total  (pct%)"
            pct_str = line.split("(")[-1].rstrip("%)").strip()
            try:
                return float(pct_str)
            except ValueError:
                pass
    return -1.0


def process_puzzle_failures(
    failures_tsv: Path,
    out_jsonl: Path,
    stockfish_path: str,
    workers: int,
) -> bool:
    """Label positions from a puzzle_bench --export-failures TSV and write JSONL.

    Calls gen_puzzle_finetune_data.py which walks every move in each failed puzzle
    to extract all intermediate FENs, labels them with Stockfish, and writes a
    train/val split.  Only the train split is kept (injected as anchor data).

    Returns True if out_jsonl was written with at least one position.
    """
    if not failures_tsv.exists() or failures_tsv.stat().st_size == 0:
        return False

    import tempfile
    tmp_dir = Path(tempfile.mkdtemp(prefix="puzzle_ft_"))
    try:
        result = subprocess.run(
            [sys.executable, "scripts/gen_puzzle_finetune_data.py",
             "--input",     str(failures_tsv),
             "--output",    str(tmp_dir),
             "--stockfish", stockfish_path,
             "--workers",   str(workers)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"[loop] WARNING: gen_puzzle_finetune_data failed:\n{result.stderr[-500:]}")
            return False

        # gen_puzzle_finetune_data writes train_<dirname>.jsonl inside tmp_dir
        train_files = list(tmp_dir.glob("train_*.jsonl"))
        if not train_files:
            return False
        train_file = train_files[0]
        if train_file.stat().st_size == 0:
            return False

        shutil.move(str(train_file), str(out_jsonl))
        n_lines = sum(1 for _ in out_jsonl.open())
        print(f"[loop] Puzzle failures → {n_lines} labeled positions → {out_jsonl}")
        return True
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# Opening lines from chess_evaluation/src/opening_book.rs (UCI move sequences).
# Each entry is played through python-chess to produce a starting FEN.
_OPENING_LINES = [
    # Ruy Lopez
    ["e2e4","e7e5","g1f3","b8c6","f1b5"],
    ["e2e4","e7e5","g1f3","b8c6","f1b5","a7a6","b5a4","g8f6"],
    ["e2e4","e7e5","g1f3","b8c6","f1b5","a7a6","b5a4","g8f6","e1g1","f8e7"],
    ["e2e4","e7e5","g1f3","b8c6","f1b5","a7a6","b5a4","g8f6","e1g1","f8e7","f1e1","b7b5","a4b3","d7d6"],
    ["e2e4","e7e5","g1f3","b8c6","f1b5","g8f6"],
    ["e2e4","e7e5","g1f3","b8c6","f1b5","g8f6","e1g1","f6e4"],
    ["e2e4","e7e5","g1f3","b8c6","f1b5","f8c5"],
    # Italian
    ["e2e4","e7e5","g1f3","b8c6","f1c4"],
    ["e2e4","e7e5","g1f3","b8c6","f1c4","f8c5"],
    ["e2e4","e7e5","g1f3","b8c6","f1c4","f8c5","c2c3","g8f6"],
    ["e2e4","e7e5","g1f3","b8c6","f1c4","f8c5","c2c3","g8f6","d2d4","e5d4","c3d4"],
    ["e2e4","e7e5","g1f3","b8c6","f1c4","f8c5","d2d3","g8f6","c2c3"],
    # Two Knights
    ["e2e4","e7e5","g1f3","b8c6","f1c4","g8f6"],
    ["e2e4","e7e5","g1f3","b8c6","f1c4","g8f6","d2d4","e5d4","e1g1"],
    # Scotch
    ["e2e4","e7e5","g1f3","b8c6","d2d4","e5d4","f3d4"],
    ["e2e4","e7e5","g1f3","b8c6","d2d4","e5d4","f3d4","g8f6"],
    # Petrov
    ["e2e4","e7e5","g1f3","g8f6","f3e5","d7d6","e5f3","f6e4"],
    ["e2e4","e7e5","g1f3","g8f6","d2d4"],
    ["e2e4","e7e5","g1f3","g8f6","f3e5","d7d6","e5f3","f6e4","d2d4"],
    # Vienna
    ["e2e4","e7e5","b1c3","g8f6","f2f4"],
    ["e2e4","e7e5","b1c3","b8c6","g1f3"],
    # King's Gambit
    ["e2e4","e7e5","f2f4","e5f4","g1f3"],
    ["e2e4","e7e5","f2f4","e5f4","g1f3","g7g5"],
    # Sicilian
    ["e2e4","c7c5","g1f3"],
    ["e2e4","c7c5","g1f3","d7d6","d2d4","c5d4","f3d4"],
    ["e2e4","c7c5","g1f3","d7d6","d2d4","c5d4","f3d4","g8f6","b1c3"],
    ["e2e4","c7c5","g1f3","d7d6","d2d4","c5d4","f3d4","g8f6","b1c3","a7a6"],
    ["e2e4","c7c5","g1f3","d7d6","d2d4","c5d4","f3d4","g8f6","b1c3","a7a6","c1e3"],
    ["e2e4","c7c5","g1f3","d7d6","d2d4","c5d4","f3d4","g8f6","b1c3","a7a6","f1e2"],
    ["e2e4","c7c5","g1f3","d7d6","d2d4","c5d4","f3d4","g8f6","b1c3","g7g6"],
    ["e2e4","c7c5","g1f3","d7d6","d2d4","c5d4","f3d4","g8f6","b1c3","g7g6","c1e3","f8g7","f2f3"],
    ["e2e4","c7c5","g1f3","b8c6","d2d4","c5d4","f3d4"],
    ["e2e4","c7c5","g1f3","b8c6","d2d4","c5d4","f3d4","g8f6","b1c3"],
    ["e2e4","c7c5","g1f3","b8c6","d2d4","c5d4","f3d4","g8f6","b1c3","e7e5"],
    ["e2e4","c7c5","g1f3","b8c6","d2d4","c5d4","f3d4","g8f6","b1c3","e7e5","d4b5","d7d6"],
    ["e2e4","c7c5","g1f3","e7e6","d2d4","c5d4","f3d4"],
    ["e2e4","c7c5","g1f3","e7e6","d2d4","c5d4","f3d4","g8f6","b1c3"],
    ["e2e4","c7c5","g1f3","e7e6","d2d4","c5d4","f3d4","a7a6"],
    ["e2e4","c7c5","b1c3","b8c6","g1f3","g7g6"],
    ["e2e4","c7c5","c2c3","g8f6","e4e5","f6d5"],
    # French
    ["e2e4","e7e6","d2d4"],
    ["e2e4","e7e6","d2d4","d7d5","b1c3"],
    ["e2e4","e7e6","d2d4","d7d5","b1c3","g8f6","e4e5"],
    ["e2e4","e7e6","d2d4","d7d5","b1c3","f8b4"],
    ["e2e4","e7e6","d2d4","d7d5","b1c3","f8b4","e4e5","c7c5"],
    ["e2e4","e7e6","d2d4","d7d5","e4e5","c7c5"],
    ["e2e4","e7e6","d2d4","d7d5","e4e5","c7c5","c2c3","b8c6","g1f3"],
    ["e2e4","e7e6","d2d4","d7d5","b1d2","g8f6"],
    ["e2e4","e7e6","d2d4","d7d5","b1d2","c7c5"],
    # Scandinavian
    ["e2e4","d7d5","e4d5"],
    ["e2e4","d7d5","e4d5","d8d5","b1c3"],
    ["e2e4","d7d5","e4d5","d8d5","b1c3","d5a5"],
    ["e2e4","d7d5","e4d5","g8f6"],
    # Caro-Kann
    ["e2e4","c7c6","d2d4","d7d5"],
    ["e2e4","c7c6","d2d4","d7d5","b1c3","d5e4","c3e4"],
    ["e2e4","c7c6","d2d4","d7d5","b1c3","d5e4","c3e4","c8f5"],
    ["e2e4","c7c6","d2d4","d7d5","b1c3","d5e4","c3e4","b8d7"],
    ["e2e4","c7c6","d2d4","d7d5","e4e5","c8f5"],
    ["e2e4","c7c6","d2d4","d7d5","e4d5","c6d5","c2c4"],
    # Alekhine
    ["e2e4","g8f6","e4e5","f6d5","d2d4"],
    ["e2e4","g8f6","e4e5","f6d5","d2d4","d7d6","g1f3"],
    # Pirc/Modern
    ["e2e4","d7d6","d2d4","g8f6","b1c3"],
    ["e2e4","g7g6","d2d4","f8g7","b1c3"],
    # Queen's Gambit
    ["d2d4","d7d5","c2c4"],
    ["d2d4","d7d5","c2c4","e7e6","b1c3"],
    ["d2d4","d7d5","c2c4","e7e6","b1c3","g8f6","g1f3"],
    ["d2d4","d7d5","c2c4","e7e6","b1c3","g8f6","c1g5"],
    ["d2d4","d7d5","c2c4","e7e6","b1c3","g8f6","c1g5","f8e7","e2e3"],
    ["d2d4","d7d5","c2c4","e7e6","b1c3","f8e7","g1f3","g8f6","c1f4"],
    # Slav
    ["d2d4","d7d5","c2c4","c7c6","b1c3","g8f6"],
    ["d2d4","d7d5","c2c4","c7c6","b1c3","g8f6","g1f3"],
    ["d2d4","d7d5","c2c4","c7c6","g1f3","g8f6","b1c3","d5c4"],
    ["d2d4","d7d5","c2c4","c7c6","g1f3","g8f6","b1c3","e7e6"],
    # QGA
    ["d2d4","d7d5","c2c4","d5c4","g1f3"],
    ["d2d4","d7d5","c2c4","d5c4","g1f3","g8f6","e2e3"],
    # Catalan
    ["d2d4","d7d5","c2c4","e7e6","g2g3"],
    ["d2d4","d7d5","c2c4","e7e6","g2g3","g8f6","f1g2"],
    ["d2d4","d7d5","c2c4","e7e6","g2g3","g8f6","f1g2","f8e7","g1f3"],
    # King's Indian
    ["d2d4","g8f6","c2c4"],
    ["d2d4","g8f6","c2c4","g7g6","b1c3"],
    ["d2d4","g8f6","c2c4","g7g6","b1c3","f8g7","e2e4"],
    ["d2d4","g8f6","c2c4","g7g6","b1c3","f8g7","e2e4","d7d6","g1f3"],
    ["d2d4","g8f6","c2c4","g7g6","b1c3","f8g7","e2e4","d7d6","g1f3","e8g8","f1e2"],
    ["d2d4","g8f6","c2c4","g7g6","b1c3","f8g7","e2e4","d7d6","f2f3"],
    # Nimzo-Indian
    ["d2d4","g8f6","c2c4","e7e6","b1c3"],
    ["d2d4","g8f6","c2c4","e7e6","b1c3","f8b4"],
    ["d2d4","g8f6","c2c4","e7e6","b1c3","f8b4","e2e3"],
    ["d2d4","g8f6","c2c4","e7e6","b1c3","f8b4","d1c2"],
    ["d2d4","g8f6","c2c4","e7e6","b1c3","f8b4","e2e3","e8g8","f1d3"],
    ["d2d4","g8f6","c2c4","e7e6","b1c3","f8b4","a2a3","b4c3","b2c3"],
    # Queen's Indian
    ["d2d4","g8f6","c2c4","e7e6","g1f3"],
    ["d2d4","g8f6","c2c4","e7e6","g1f3","b7b6"],
    ["d2d4","g8f6","c2c4","e7e6","g1f3","b7b6","g2g3"],
    ["d2d4","g8f6","c2c4","e7e6","g1f3","b7b6","g2g3","c8b7","f1g2"],
    ["d2d4","g8f6","c2c4","e7e6","g1f3","f8b4"],
    # Grünfeld
    ["d2d4","g8f6","c2c4","g7g6","b1c3","d7d5"],
    ["d2d4","g8f6","c2c4","g7g6","b1c3","d7d5","c4d5","f6d5","e2e4"],
    ["d2d4","g8f6","c2c4","g7g6","b1c3","d7d5","c4d5","f6d5","e2e4","d5c3","b2c3","f8g7"],
    # Grünfeld Russian System
    ["d2d4","g8f6","c2c4","g7g6","b1c3","d7d5","g1f3","f8g7","d1b3"],
    # Grünfeld Exchange, Bc4 line
    ["d2d4","g8f6","c2c4","g7g6","b1c3","d7d5","c4d5","f6d5","e2e4","d5c3","b2c3","f8g7","f1c4"],
    # Benoni
    ["d2d4","g8f6","c2c4","c7c5","d4d5"],
    ["d2d4","g8f6","c2c4","c7c5","d4d5","e7e6","b1c3","e6d5","c4d5","d7d6"],
    # Modern Benoni main line
    ["d2d4","g8f6","c2c4","c7c5","d4d5","e7e6","b1c3","e6d5","c4d5","d7d6","e2e4","g7g6","g1f3"],
    # Modern Benoni Fianchetto variation
    ["d2d4","g8f6","c2c4","c7c5","d4d5","e7e6","b1c3","e6d5","c4d5","d7d6","g2g3","g7g6","f1g2"],
    # Benko Gambit
    ["d2d4","g8f6","c2c4","c7c5","d4d5","b7b5","c4b5","a7a6"],
    ["d2d4","g8f6","c2c4","c7c5","d4d5","b7b5","c4b5","a7a6","b5a6","c8a6"],
    # Dutch
    ["d2d4","f7f5","g2g3","g8f6","f1g2"],
    ["d2d4","f7f5","c2c4","g8f6","g2g3"],
    # Dutch Stonewall
    ["d2d4","f7f5","g1f3","g8f6","e2e3","e7e6","f1d3","d7d5","e1g1","c7c6"],
    # Dutch Leningrad
    ["d2d4","f7f5","g2g3","g8f6","f1g2","g7g6","g1f3","f8g7","e1g1","e8g8"],
    # Trompowsky Attack
    ["d2d4","g8f6","c1g5"],
    ["d2d4","g8f6","c1g5","e7e6","e2e3"],
    ["d2d4","g8f6","c1g5","f6e4","g5f4","d7d5"],
    # Torre Attack
    ["d2d4","g8f6","g1f3","e7e6","c1g5"],
    ["d2d4","g8f6","g1f3","d7d5","c1g5"],
    ["d2d4","d7d5","g1f3","g8f6","c1g5","e7e6","e2e3"],
    # London
    ["d2d4","d7d5","g1f3","g8f6","c1f4"],
    ["d2d4","d7d5","g1f3","g8f6","c1f4","e7e6","e2e3"],
    ["d2d4","d7d5","g1f3","g8f6","c1f4","c7c5","e2e3"],
    ["d2d4","g8f6","g1f3","d7d5","c1f4","e7e6"],
    ["d2d4","g8f6","c1f4","d7d5","e2e3","e7e6","g1f3"],
    # English
    ["c2c4","e7e5"],
    ["c2c4","e7e5","b1c3","g8f6"],
    ["c2c4","e7e5","g2g3","g8f6","f1g2"],
    ["c2c4","g8f6","b1c3","e7e5"],
    ["c2c4","c7c5","g1f3","b8c6"],
    ["c2c4","g8f6","g2g3","g7g6"],
    # Réti
    ["g1f3","d7d5","g2g3"],
    ["g1f3","d7d5","g2g3","g8f6","f1g2"],
    ["g1f3","d7d5","c2c4"],
    ["g1f3","g8f6","g2g3","g7g6","f1g2","f8g7"],
]


def _opening_fens() -> list[str]:
    """Return one FEN per opening line by replaying moves with python-chess."""
    import chess
    fens = []
    for line in _OPENING_LINES:
        board = chess.Board()
        ok = True
        for uci in line:
            try:
                board.push_uci(uci)
            except Exception:
                ok = False
                break
        if ok:
            fens.append(board.fen())
    return fens


def _selfplay_chunk(selfplay_binary: str, engine_path: str,
                    candidate_npz: str, baseline_npz: str,
                    games: int, movetime_ms: int, threads_per_engine: int,
                    opening_fens_file: str = "") -> tuple[int, int, int]:
    """Run one self_play chunk and return (wins, draws, losses) for engine1 (candidate)."""
    cmd = [selfplay_binary,
         engine_path, engine_path,
         "--games",        str(games),
         "--movetime",     str(movetime_ms),
         "--no-ponder",
         "--engine1-opt",  f"EvalFile={candidate_npz}",
         "--engine1-opt",  "NeuralEval=true",
         "--engine1-opt",  f"Threads={threads_per_engine}",
         "--engine2-opt",  f"EvalFile={baseline_npz}",
         "--engine2-opt",  "NeuralEval=true",
         "--engine2-opt",  f"Threads={threads_per_engine}",
    ]
    if opening_fens_file:
        cmd += ["--opening-fens", opening_fens_file]
    result = subprocess.run(cmd, capture_output=True, text=True,)
    wins = draws = losses = 0
    wins_seen = 0
    for line in result.stdout.splitlines():
        # Final summary lines: "  chess_uci : 5 wins  (33.3%)"  (engine name = file stem,
        # not the literal "engine1"/"engine2" — detect by position: first = engine1, second = engine2)
        # or "  Draws         : 10"
        if " wins " in line and "%" in line:
            try:
                val = int(line.split(":")[1].strip().split()[0])
                if wins_seen == 0:
                    wins = val
                else:
                    losses = val
                wins_seen += 1
            except (IndexError, ValueError):
                pass
        elif line.strip().startswith("Draws"):
            try: draws = int(line.split(":")[1].strip().split()[0])
            except (IndexError, ValueError): pass
    return wins, draws, losses


def selfplay_winrate(selfplay_binary: str, engine_path: str,
                     candidate_npz: Path, baseline_npz: Path,
                     games: int, movetime_ms: int,
                     n_workers: int = 4) -> float:
    """Pit candidate vs baseline in parallel using n_workers self_play instances.

    Splits `games` across workers so all run concurrently.  Each worker gets
    `threads_per_engine = max(2, available_cpus // (2 * n_workers))` threads so
    the total CPU usage stays within the machine's core count.

    Returns candidate score (0.0–100.0, draws count 0.5) or -1.0 if unavailable.
    """
    import tempfile

    if not selfplay_binary:
        return -1.0

    cpu_count = os.cpu_count() or 4
    threads_per_engine = max(2, cpu_count // (2 * n_workers))

    # Distribute games as evenly as possible across workers.
    base, remainder = divmod(games, n_workers)
    chunk_sizes = [base + (1 if i < remainder else 0) for i in range(n_workers)]

    # Write opening FENs to a shared temp file that all worker processes can read.
    fens = _opening_fens()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tf:
        opening_fens_file = tf.name
        tf.write("\n".join(fens) + "\n")

    try:
        # subprocess.run releases the GIL so ThreadPoolExecutor is sufficient.
        # Use as_completed so the progress bar updates as each chunk finishes
        # rather than blocking until all chunks are done.
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
            future_to_chunk = {
                ex.submit(_selfplay_chunk,
                          selfplay_binary, engine_path,
                          str(candidate_npz), str(baseline_npz),
                          chunk, movetime_ms, threads_per_engine,
                          opening_fens_file): chunk
                for chunk in chunk_sizes
            }

            total_wins = total_draws = total_losses = 0
            chunk_results = []
            with tqdm(total=games, desc="[loop] self-play eval", unit="game",
                      bar_format="{l_bar}{bar}| {n}/{total} games  {postfix}") as pbar:
                for fut in concurrent.futures.as_completed(future_to_chunk):
                    w, d, l = fut.result()
                    chunk_results.append((w, d, l))
                    total_wins   += w
                    total_draws  += d
                    total_losses += l
                    done = total_wins + total_draws + total_losses
                    score_so_far = (total_wins + 0.5 * total_draws) / done * 100.0 if done else 0.0
                    pbar.set_postfix_str(
                        f"{total_wins}W/{total_draws}D/{total_losses}L  score={score_so_far:.1f}%"
                    )
                    pbar.update(future_to_chunk[fut])
    finally:
        os.unlink(opening_fens_file)

    total_games = total_wins + total_draws + total_losses
    if total_games == 0:
        return -1.0
    score = (total_wins + 0.5 * total_draws) / total_games * 100.0
    print(f"[loop]   self-play chunks: {[f'{r[0]}W/{r[1]}D/{r[2]}L' for r in chunk_results]}")
    return score


def generate_data(
    engine_path: str,
    npz_path: Path | None,
    stockfish_path: str,
    output_path: Path,
    games: int,
    positions_per_game: int,
    movetime_ms: int,
    eval_depth: int,
    workers: int,
    selfplay_threads: int,
    selfplay_parallel: int,
    opening_fens_file: str = "",
):
    print(f"[loop] Generating {games} self-play games → {output_path}")
    cmd = [sys.executable, "scripts/generate_data.py",
           "--selfplay-engine", engine_path,
           "--label-engine", stockfish_path,
           "--output", str(output_path),
           "--selfplay-games", str(games),
           "--positions-per-game", str(positions_per_game),
           "--selfplay-movetime-ms", str(movetime_ms),
           "--eval-depth", str(eval_depth),
           "--max-positions", str(games * positions_per_game),
           "--workers", str(workers),
           "--selfplay-threads", str(selfplay_threads),
           "--selfplay-parallel", str(selfplay_parallel),
           ]
    if npz_path is not None:
        cmd += [
            "--selfplay-engine-opt", f"EvalFile={npz_path}",
            "--selfplay-engine-opt", "NeuralEval=true",
        ]
    if opening_fens_file:
        cmd += ["--opening-fens-file", opening_fens_file]
    subprocess.run(cmd, check=True)


def append_to_pool(new_data: Path, pool_file: Path, pool_size: int):
    """Append new_data to pool_file, then trim to pool_size lines (keep newest)."""
    with pool_file.open("a", encoding="utf-8") as f:
        f.write(new_data.read_text(encoding="utf-8"))

    lines = pool_file.read_text(encoding="utf-8").splitlines(keepends=True)
    if len(lines) > pool_size:
        lines = lines[-pool_size:]
        pool_file.write_text("".join(lines), encoding="utf-8")
    print(f"[loop] Pool size: {len(lines)} positions")


def split_pool(pool_file: Path, train_file: Path, val_file: Path, val_fraction: float = 0.1):
    lines = pool_file.read_text(encoding="utf-8").splitlines(keepends=True)
    random.shuffle(lines)
    n_val = max(1000, int(len(lines) * val_fraction))
    n_train = len(lines) - n_val
    train_file.write_text("".join(lines[:n_train]), encoding="utf-8")
    val_file.write_text("".join(lines[n_train:]), encoding="utf-8")
    print(f"[loop] Split: {n_train} train / {n_val} val")


def inject_anchor_data(anchor_file: Path, train_file: Path, n: int,
                       extra_files: list[Path] | None = None,
                       exclude_fens: set[str] | None = None):
    """Sample n lines from anchor_file and append them to train_file.

    Injecting a fixed slice of original training data each iteration prevents
    long-term distributional drift: as the FIFO pool fills with self-play data
    the model would otherwise eventually train only on its own distribution.
    These anchor positions are never evicted from training (but not added to
    the validation set, keeping val as a clean self-play signal).

    extra_files: additional JSONL files whose entire contents are appended
    after the anchor sample (e.g. puzzle failure positions for targeted repair).
    exclude_fens: set of FEN strings to skip when sampling (prevents gen_val
    positions from leaking into training and making the gen_val gate optimistic).
    """
    lines = anchor_file.read_text(encoding="utf-8").splitlines(keepends=True)
    if not lines:
        return
    if exclude_fens:
        filtered = []
        for line in lines:
            try:
                if json.loads(line)["fen"] not in exclude_fens:
                    filtered.append(line)
            except (json.JSONDecodeError, KeyError):
                filtered.append(line)
        excluded = len(lines) - len(filtered)
        if excluded:
            print(f"[loop] Excluded {excluded} gen_val positions from anchor sample")
        lines = filtered
    sample = random.sample(lines, min(n, len(lines)))
    with train_file.open("a", encoding="utf-8") as f:
        f.writelines(sample)
    print(f"[loop] Injected {len(sample)} anchor positions into training set")

    if extra_files:
        for extra in extra_files:
            if not extra or not extra.exists() or extra.stat().st_size == 0:
                continue
            extra_lines = extra.read_text(encoding="utf-8").splitlines(keepends=True)
            with train_file.open("a", encoding="utf-8") as f:
                f.writelines(extra_lines)
            print(f"[loop] Injected {len(extra_lines)} puzzle-failure positions from {extra}")


def fine_tune(
    checkpoint_path: Path,
    out_checkpoint: Path,
    config: str,
    tb_logdir: str,
):
    print(f"[loop] Fine-tuning from {checkpoint_path} → {out_checkpoint}")
    subprocess.run(
        [sys.executable, "scripts/train.py",
         "--config", config,
         "--resume", str(checkpoint_path),
         "--reset-best-val",
         "--out", str(out_checkpoint),
         "--tb-logdir", tb_logdir,
         ],
        check=True,
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Continuous self-play training loop")
    ap.add_argument("--engine", required=True, help="Path to chess_uci binary")
    ap.add_argument("--stockfish", default="/usr/bin/stockfish")
    ap.add_argument("--initial-checkpoint", required=True,
                    help="Starting checkpoint (.pt) — must already be trained")
    ap.add_argument("--iterations", type=int, default=0,
                    help="Number of iterations (0 = run forever)")
    ap.add_argument("--games-per-iter", type=int, default=3000,
                    help="Self-play games per iteration (default 3000)")
    ap.add_argument("--positions-per-game", type=int, default=15,
                    help="Positions sampled per self-play game (default 15)")
    ap.add_argument("--movetime-ms", type=int, default=100,
                    help="Engine think time per move during self-play")
    ap.add_argument("--eval-depth", type=int, default=14,
                    help="Stockfish depth for labelling")
    ap.add_argument("--workers", type=int, default=32,
                    help="Parallel Stockfish labelling workers")
    ap.add_argument("--selfplay-threads", type=int, default=1,
                    help="Threads per self-play engine instance (default 1)")
    ap.add_argument("--selfplay-parallel", type=int, default=32,
                    help="Parallel self-play engine instances (default 32)")
    ap.add_argument("--pool-size", type=int, default=750_000,
                    help="Max positions kept in the replay pool (default 750000 — new data is ~6%% per iter)")
    ap.add_argument("--config", default="configs/finetune.yaml")
    ap.add_argument("--artifacts-dir", default="artifacts")
    # ── Evaluation ────────────────────────────────────────────────────────────
    ap.add_argument("--puzzle-binary", default="",
                    help="Path to puzzle_bench binary for promotion gating")
    ap.add_argument("--puzzle-file", default="",
                    help="Path to lichess puzzle CSV (.zst) for benchmarking")
    ap.add_argument("--puzzle-count", type=int, default=2000,
                    help="Puzzles to solve per eval (default 2000, SE≈1.1%% at 50%% baseline)")
    ap.add_argument("--puzzle-depth", type=int, default=7,
                    help="Search depth for puzzle bench (default 7)")
    ap.add_argument("--puzzle-min-rating", type=int, default=1500,
                    help="Minimum puzzle rating for eval (default 1500, filters trivial puzzles)")
    ap.add_argument("--puzzle-max-rating", type=int, default=0,
                    help="Maximum puzzle rating for eval (default 0 = no limit)")
    ap.add_argument("--puzzle-regression-tolerance", type=float, default=0.0,
                    help="Allow puzzle score to be up to this %% below best before rejecting (default 0.0 — deterministic benchmark, no noise)")
    ap.add_argument("--puzzle-seed", type=int, default=42,
                    help="Fixed RNG seed for all puzzle evaluations — same puzzles every iteration (default 42)")
    ap.add_argument("--selfplay-binary", default="",
                    help="Path to self_play binary for head-to-head eval")
    ap.add_argument("--selfplay-eval-games", type=int, default=40,
                    help="Games for head-to-head eval (default 40, SE≈7.9%% — only catches clear regressions)")
    ap.add_argument("--selfplay-eval-movetime", type=int, default=100,
                    help="Movetime ms for head-to-head eval games (default 100)")
    ap.add_argument("--selfplay-eval-workers", type=int, default=4,
                    help="Parallel self_play instances for eval (default 4); "
                         "threads per engine auto-scaled to fill available CPUs")
    ap.add_argument("--selfplay-min-winrate", type=float, default=47.0,
                    help="Minimum self-play win rate %% to count as not-a-regression (default 47.0)")
    # ── Anchor data (anti-drift) ───────────────────────────────────────────────
    ap.add_argument("--anchor-data", default="",
                    help="Path to JSONL file of original training positions to inject each iteration. "
                         "Prevents long-term distributional drift as the FIFO pool fills with self-play data.")
    ap.add_argument("--anchor-size", type=int, default=50_000,
                    help="Anchor positions sampled and appended to train set each iteration (default 50000)")
    ap.add_argument("--anchor-min-fraction", type=float, default=0.10,
                    help="Keep anchor at least this fraction of the current pool size (default 0.10 = 10%%). "
                         "Prevents anchor from becoming negligible as the pool grows. "
                         "Effective anchor = max(--anchor-size, pool_size * fraction).")
    # ── Generalisation validation (real-game positions, not self-play) ─────────
    ap.add_argument("--gen-val-size", type=int, default=10_000,
                    help="Positions to hold out from anchor data for generalisation validation (default 10000). "
                         "CP-MAE on this fixed set reveals whether improvements transfer to real-game positions.")
    ap.add_argument("--gen-val-max-increase", type=float, default=5.0,
                    help="Block promotion if gen_val CP-MAE rises more than this %% above best (default 5.0). "
                         "Catches overfitting to the self-play distribution: puzzle score can hold while the model "
                         "degrades on real-game positions. Set 0 to disable this gate.")
    args = ap.parse_args()

    artifacts = Path(args.artifacts_dir)
    artifacts.mkdir(exist_ok=True)

    best_ck = Path(args.initial_checkpoint)
    if not best_ck.exists():
        sys.exit(f"Checkpoint not found: {best_ck}")

    best_mae = load_val_mae(best_ck)
    print(f"[loop] Starting from {best_ck}  val_cp_mae={best_mae:.1f}")

    # Build (once) a fixed generalisation validation set from the anchor data.
    # Using a fixed seed means the same positions are held out every run,
    # so gen_val_mae is comparable across all iterations and restarts.
    gen_val_file = Path("data/gen_val.jsonl")
    gen_val_file.parent.mkdir(exist_ok=True)
    if args.anchor_data and Path(args.anchor_data).exists():
        if not gen_val_file.exists():
            sample_gen_val(Path(args.anchor_data), gen_val_file, args.gen_val_size, seed=0)
        else:
            print(f"[loop] Reusing existing gen_val set: {gen_val_file} ({gen_val_file.stat().st_size // 1024}KB)")
    else:
        gen_val_file = None

    # Build a set of gen_val FENs to exclude from anchor injection.
    # gen_val is sampled from the same anchor file that gets injected each iteration,
    # so without exclusion those positions gradually leak into training and make the
    # gen_val gate optimistic (it measures fit to positions the model trained on).
    gen_val_fens: set[str] = set()
    if gen_val_file and gen_val_file.exists():
        for line in gen_val_file.open(encoding="utf-8"):
            try:
                gen_val_fens.add(json.loads(line.strip())["fen"])
            except (json.JSONDecodeError, KeyError):
                pass
        if gen_val_fens:
            print(f"[loop] Loaded {len(gen_val_fens)} gen_val FENs for anchor exclusion")

    # Export initial weights and measure baseline puzzle score.
    best_npz = artifacts / "eval.npz"
    if not best_npz.exists():
        export_weights(best_ck, best_npz)
    baseline_failures_tsv = Path("data/puzzle_failures_iter0.tsv")
    best_puzzle = puzzle_score(
        args.puzzle_binary, args.puzzle_file, best_npz,
        args.puzzle_count, args.puzzle_depth, seed=args.puzzle_seed,
        min_rating=args.puzzle_min_rating, max_rating=args.puzzle_max_rating,
        export_failures_file=str(baseline_failures_tsv) if args.puzzle_binary else "",
    )
    if best_puzzle >= 0:
        print(f"[loop] Initial puzzle score: {best_puzzle:.1f}%  (seed={args.puzzle_seed})")

    best_gen_mae = gen_val_mae(best_ck, gen_val_file) if gen_val_file else -1.0
    if best_gen_mae >= 0:
        print(f"[loop] Initial gen_val CP-MAE: {best_gen_mae:.2f}cp  (real-game positions, fixed set)")

    # Clean up any stale puzzle-failure JSONL files from previous runs.
    # These are the only per-iteration artifacts that are never deleted during
    # normal operation.  On resume they won't exist yet (injection happens before
    # the file is created), so deleting them here is always safe.
    # NOTE: this deletes *.jsonl only — the baseline *.tsv above survives.
    stale = sorted(Path("data").glob("puzzle_failures_iter*.jsonl"))
    if stale:
        for f in stale:
            f.unlink()
        print(f"[loop] Removed {len(stale)} stale puzzle-failure file(s) from previous run")

    # Convert the baseline failure TSV into puzzle_failures_iter0.jsonl so that
    # iteration 1 can inject it via prev_puzzle_anchor (iter{1-1} = iter0).
    # This gives the first training run the same targeted repair that all later
    # iterations benefit from.
    baseline_puzzle_anchor = Path("data/puzzle_failures_iter0.jsonl")
    if args.puzzle_binary and args.stockfish and baseline_failures_tsv.exists() and baseline_failures_tsv.stat().st_size > 0:
        process_puzzle_failures(
            baseline_failures_tsv,
            baseline_puzzle_anchor,
            args.stockfish,
            workers=args.workers,
        )
        baseline_failures_tsv.unlink(missing_ok=True)

    # Always regenerate the opening FENs file from _OPENING_LINES so that
    # changes to the opening book take effect on the next restart without
    # requiring manual deletion of the cached file.
    opening_fens_file = Path("data/opening_fens.txt")
    opening_fens_file.parent.mkdir(exist_ok=True)
    fens = _opening_fens()
    opening_fens_file.write_text("\n".join(fens) + "\n")
    print(f"[loop] Wrote {len(fens)} theory opening FENs → {opening_fens_file}")

    pool_file = Path("data/selfplay_pool.jsonl")
    pool_file.parent.mkdir(exist_ok=True)
    if not pool_file.exists():
        pool_file.touch()

    iteration = 0
    while True:
        iteration += 1
        if args.iterations > 0 and iteration > args.iterations:
            break
        print(f"\n{'='*60}")
        print(f"[loop] Iteration {iteration}")
        print(f"{'='*60}")

        # 1. Use current best weights for self-play (eval.npz is always up to date)
        npz_path = best_npz  # artifacts/eval.npz

        # 2. Generate self-play data (skip if already done — supports resume)
        new_data = Path(f"data/selfplay_iter{iteration}.jsonl")
        if new_data.exists() and new_data.stat().st_size > 0:
            print(f"[loop] Reusing existing {new_data} ({new_data.stat().st_size // 1024}KB) — skipping generation")
        else:
            generate_data(
                engine_path=args.engine,
                npz_path=npz_path,
                stockfish_path=args.stockfish,
                output_path=new_data,
                games=args.games_per_iter,
                positions_per_game=args.positions_per_game,
                movetime_ms=args.movetime_ms,
                eval_depth=args.eval_depth,
                workers=args.workers,
                selfplay_threads=args.selfplay_threads,
                selfplay_parallel=args.selfplay_parallel,
                opening_fens_file=str(opening_fens_file),
            )

        # 3. Append to replay pool
        append_to_pool(new_data, pool_file, args.pool_size)
        new_data.unlink()  # remove per-iter file, it's in the pool now

        # 4. Split pool into train/val
        split_pool(pool_file, Path("data/train.jsonl"), Path("data/val.jsonl"))

        # 4b. Inject anchor positions from original training data (if configured).
        # This prevents the model from drifting into a narrow self-play distribution
        # after many iterations. The anchor positions are appended to train.jsonl only
        # (not val), so the validation signal stays clean.
        #
        # Dynamic scaling: as the pool grows the fixed --anchor-size would shrink to a
        # negligible fraction. effective_anchor = max(anchor_size, pool_size * fraction)
        # keeps anchors at least --anchor-min-fraction of the training set.
        # Puzzle failures written at the END of iteration N are injected at the
        # START of iteration N+1, so use iter{N-1} for injection and iter{N} for writing.
        prev_puzzle_anchor = Path(f"data/puzzle_failures_iter{iteration - 1}.jsonl")
        puzzle_anchor_jsonl = Path(f"data/puzzle_failures_iter{iteration}.jsonl")
        if args.anchor_data and Path(args.anchor_data).exists():
            pool_line_count = sum(1 for _ in pool_file.open(encoding="utf-8"))
            effective_anchor = max(
                args.anchor_size,
                int(pool_line_count * args.anchor_min_fraction),
            )
            if effective_anchor != args.anchor_size:
                print(f"[loop] Anchor scaled: {args.anchor_size} → {effective_anchor} "
                      f"({args.anchor_min_fraction*100:.0f}%% of {pool_line_count} pool lines)")
            inject_anchor_data(
                Path(args.anchor_data), Path("data/train.jsonl"), effective_anchor,
                extra_files=[prev_puzzle_anchor] if prev_puzzle_anchor.exists() else None,
                exclude_fens=gen_val_fens if gen_val_fens else None,
            )
        elif args.anchor_data:
            print(f"[loop] WARNING: --anchor-data '{args.anchor_data}' not found — skipping anchor injection")

        # 4c. Shuffle train.jsonl so anchor positions are distributed throughout.
        train_path = Path("data/train.jsonl")
        train_lines = train_path.read_text(encoding="utf-8").splitlines(keepends=True)
        random.shuffle(train_lines)
        train_path.write_text("".join(train_lines), encoding="utf-8")
        print(f"[loop] Shuffled {len(train_lines)} training positions")

        # 5. Fine-tune
        candidate_ck = artifacts / f"checkpoint_iter{iteration}.pt"
        tb_logdir = f"runs/loop_iter{iteration}"
        fine_tune(
            checkpoint_path=best_ck,
            out_checkpoint=candidate_ck,
            config=args.config,
            tb_logdir=tb_logdir,
        )

        # 6. Compare and promote
        if not candidate_ck.exists():
            # Guard against training crashes: --reset-best-val means epoch 1 always
            # saves a checkpoint, so this only fires if train.py was killed before
            # writing anything. Continue from current best so the loop can recover.
            print(f"[loop] Iteration {iteration}: training produced no checkpoint — continuing from current best.")
        else:
            candidate_mae = load_val_mae(candidate_ck)
            candidate_npz = artifacts / f"candidate_iter{iteration}.npz"
            export_weights(candidate_ck, candidate_npz)

            # ── Evaluate candidate strength ───────────────────────────────
            # Fixed seed ensures candidate is always scored on the same puzzle set
            # as the stored best_puzzle — no need to re-run on the best model.
            # Export failures for feedback: they become anchor training data next iteration.
            failures_tsv = Path(f"data/puzzle_failures_iter{iteration}.tsv")
            cand_puzzle = puzzle_score(
                args.puzzle_binary, args.puzzle_file, candidate_npz,
                args.puzzle_count, args.puzzle_depth, seed=args.puzzle_seed,
                min_rating=args.puzzle_min_rating, max_rating=args.puzzle_max_rating,
                export_failures_file=str(failures_tsv) if args.puzzle_binary else "",
            )

            # Convert puzzle failures TSV → labeled JSONL for next iteration's anchor.
            if failures_tsv.exists() and failures_tsv.stat().st_size > 0:
                process_puzzle_failures(
                    failures_tsv,
                    puzzle_anchor_jsonl,
                    args.stockfish,
                    workers=args.workers,
                )
                failures_tsv.unlink(missing_ok=True)

            # Early-exit: compute puzzle gate result immediately so we can skip
            # the expensive self-play eval when the puzzle gate already fails.
            puzzle_ok = (
                cand_puzzle >= best_puzzle - args.puzzle_regression_tolerance
                if cand_puzzle >= 0 and best_puzzle >= 0 else None
            )

            if cand_puzzle >= 0:
                gate_str = {True: "PASSED", False: "FAILED", None: "N/A"}[puzzle_ok]
                print(f"[loop] Puzzle score: candidate={cand_puzzle:.1f}%  best={best_puzzle:.1f}%  [{gate_str}]")

            if puzzle_ok is False:
                print(f"[loop] Puzzle gate failed ({cand_puzzle:.1f}% < {best_puzzle - args.puzzle_regression_tolerance:.1f}%) — skipping self-play eval")
                cand_winrate = -1.0
            else:
                cand_winrate = selfplay_winrate(
                    args.selfplay_binary, args.engine,
                    candidate_npz, best_npz,
                    args.selfplay_eval_games, args.selfplay_eval_movetime,
                    n_workers=args.selfplay_eval_workers,
                )
            cand_gen_mae = gen_val_mae(candidate_ck, gen_val_file) if gen_val_file else -1.0

            print(f"[loop] Iteration {iteration}: candidate mae={candidate_mae:.1f}cp  best mae={best_mae:.1f}cp")
            if cand_puzzle >= 0:
                print(f"[loop] Puzzle score:   candidate={cand_puzzle:.1f}%  best={best_puzzle:.1f}%")
            if cand_winrate >= 0:
                print(f"[loop] Self-play win rate (candidate vs best): {cand_winrate:.1f}%")
            if cand_gen_mae >= 0:
                delta = cand_gen_mae - best_gen_mae
                trend = f"{'▲' if delta > 0 else '▼'}{abs(delta):.2f}cp" if best_gen_mae >= 0 else "baseline"
                print(f"[loop] Gen-val CP-MAE: candidate={cand_gen_mae:.2f}cp  best={best_gen_mae:.2f}cp  ({trend})")

            # ── Promotion decision ────────────────────────────────────────
            # val_cp_mae on the self-play pool is a circular signal (model fits
            # its own data) and is NOT used for promotion.  Use external signals:
            #   - puzzle score >= best (equal or better tactical ability)
            #   - self-play win rate >= min_winrate (not a regression in play)
            #   - gen_val CP-MAE increase <= --gen-val-max-increase (real-game generalisation)
            # Both puzzle + winrate must pass when both tools are configured.
            # Falls back to each individual signal if only one is available.
            # If neither tool is configured, always promote with a warning.
            # puzzle_ok already computed above (used for self-play early-exit)
            winrate_ok  = cand_winrate >= args.selfplay_min_winrate if cand_winrate >= 0 else None

            # gen_val soft gate: block if real-game MAE rises more than threshold.
            # This catches overfitting to the self-play distribution where puzzle score
            # holds but positional understanding degrades on real-game positions.
            gen_val_ok = None
            gen_val_reason = ""
            if (cand_gen_mae >= 0 and best_gen_mae >= 0 and args.gen_val_max_increase > 0):
                gen_val_threshold = best_gen_mae * (1.0 + args.gen_val_max_increase / 100.0)
                gen_val_ok = cand_gen_mae <= gen_val_threshold
                gen_val_reason = (
                    f"gen_val {cand_gen_mae:.2f}cp <= {gen_val_threshold:.2f}cp "
                    f"({args.gen_val_max_increase:.0f}%% tolerance)"
                )
                if not gen_val_ok:
                    print(f"[loop] gen_val gate FAILED: {cand_gen_mae:.2f}cp > {gen_val_threshold:.2f}cp "
                          f"(best={best_gen_mae:.2f}cp + {args.gen_val_max_increase:.0f}%%)")

            if puzzle_ok is not None and winrate_ok is not None:
                promoted = puzzle_ok and winrate_ok
                reason = (f"puzzle {cand_puzzle:.1f}%>={best_puzzle - args.puzzle_regression_tolerance:.1f}%"
                          f" AND win rate {cand_winrate:.1f}%>={args.selfplay_min_winrate:.0f}%")
            elif puzzle_ok is not None:
                promoted = puzzle_ok
                reason = f"puzzle {cand_puzzle:.1f}% >= {best_puzzle - args.puzzle_regression_tolerance:.1f}%"
            elif winrate_ok is not None:
                promoted = winrate_ok
                reason = f"win rate {cand_winrate:.1f}% >= {args.selfplay_min_winrate:.0f}%"
            else:
                promoted = True
                reason = "no eval tools configured — always promoting"
                print("[loop] WARNING: no --puzzle-binary or --selfplay-binary configured; "
                      "promotion is unconditional.")

            # Apply gen_val gate on top of the primary decision.
            if promoted and gen_val_ok is not None and not gen_val_ok:
                promoted = False
                reason = f"{reason} BUT gen_val gate failed ({gen_val_reason})"

            if promoted:
                best_ck = candidate_ck
                best_mae = candidate_mae
                best_puzzle = cand_puzzle
                if cand_gen_mae >= 0:
                    best_gen_mae = cand_gen_mae
                shutil.copy(candidate_ck, artifacts / "best_checkpoint.pt")
                shutil.copy(candidate_npz, best_npz)  # update eval.npz in place
                candidate_npz.unlink(missing_ok=True)  # no longer needed; eval.npz has the content
                print(f"[loop] Promoted! ({reason}) — exported weights → {artifacts / 'eval.npz'}")
            else:
                not_reason = reason
                print(f"[loop] Not promoted ({not_reason}) — keeping current best as training base.")
                # Discard candidate entirely; next iteration fine-tunes from best_ck again.
                candidate_ck.unlink(missing_ok=True)
                candidate_npz.unlink(missing_ok=True)
                # puzzle_anchor_jsonl is kept for the next iteration: it will be
                # read as prev_puzzle_anchor at the start of iteration N+1.

    print(f"\n[loop] Done. Best val_cp_mae={best_mae:.1f}  checkpoint={best_ck}")


if __name__ == "__main__":
    main()

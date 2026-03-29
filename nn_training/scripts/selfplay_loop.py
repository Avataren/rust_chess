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


def puzzle_score(bench_binary: str, puzzle_file: str, npz_path: Path,
                 count: int, depth: int, seed: int,
                 min_rating: int = 0, max_rating: int = 0) -> float:
    """Run puzzle_bench and return the overall solve rate (0.0–100.0).
    Returns -1.0 if the binary or puzzle file is not available."""
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


def _selfplay_chunk(selfplay_binary: str, engine_path: str,
                    candidate_npz: str, baseline_npz: str,
                    games: int, movetime_ms: int, threads_per_engine: int) -> tuple[int, int, int]:
    """Run one self_play chunk and return (wins, draws, losses) for engine1 (candidate)."""
    result = subprocess.run(
        [selfplay_binary,
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
         ],
        capture_output=True, text=True,
    )
    wins = draws = losses = 0
    for line in result.stdout.splitlines():
        # "  engine1 : 5 wins  (50.0%)"  or  "  Draws   : 3"
        if "wins" in line and "engine1" in line:
            try: wins = int(line.split(":")[1].strip().split()[0])
            except (IndexError, ValueError): pass
        elif line.strip().startswith("Draws"):
            try: draws = int(line.split(":")[1].strip().split()[0])
            except (IndexError, ValueError): pass
        elif "wins" in line and "engine2" in line:
            try: losses = int(line.split(":")[1].strip().split()[0])
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
    if not selfplay_binary:
        return -1.0

    cpu_count = os.cpu_count() or 4
    threads_per_engine = max(2, cpu_count // (2 * n_workers))

    # Distribute games as evenly as possible across workers.
    base, remainder = divmod(games, n_workers)
    chunk_sizes = [base + (1 if i < remainder else 0) for i in range(n_workers)]

    # subprocess.run releases the GIL so ThreadPoolExecutor is sufficient.
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = [
            ex.submit(_selfplay_chunk,
                      selfplay_binary, engine_path,
                      str(candidate_npz), str(baseline_npz),
                      chunk, movetime_ms, threads_per_engine)
            for chunk in chunk_sizes
        ]
        results = [f.result() for f in futures]

    total_wins   = sum(r[0] for r in results)
    total_draws  = sum(r[1] for r in results)
    total_losses = sum(r[2] for r in results)
    total_games  = total_wins + total_draws + total_losses
    if total_games == 0:
        return -1.0
    score = (total_wins + 0.5 * total_draws) / total_games * 100.0
    print(f"[loop]   self-play chunks: {[f'{r[0]}W/{r[1]}D/{r[2]}L' for r in results]}")
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
            "--selfplay-engine-opt", "NeuralConfidence=0.4",
        ]
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


def inject_anchor_data(anchor_file: Path, train_file: Path, n: int):
    """Sample n lines from anchor_file and append them to train_file.

    Injecting a fixed slice of original training data each iteration prevents
    long-term distributional drift: as the FIFO pool fills with self-play data
    the model would otherwise eventually train only on its own distribution.
    These anchor positions are never evicted from training (but not added to
    the validation set, keeping val as a clean self-play signal).
    """
    lines = anchor_file.read_text(encoding="utf-8").splitlines(keepends=True)
    if not lines:
        return
    sample = random.sample(lines, min(n, len(lines)))
    with train_file.open("a", encoding="utf-8") as f:
        f.writelines(sample)
    print(f"[loop] Injected {len(sample)} anchor positions into training set")


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
    args = ap.parse_args()

    artifacts = Path(args.artifacts_dir)
    artifacts.mkdir(exist_ok=True)

    best_ck = Path(args.initial_checkpoint)
    if not best_ck.exists():
        sys.exit(f"Checkpoint not found: {best_ck}")

    best_mae = load_val_mae(best_ck)
    print(f"[loop] Starting from {best_ck}  val_cp_mae={best_mae:.1f}")

    # Export initial weights and measure baseline puzzle score.
    best_npz = artifacts / "eval.npz"
    if not best_npz.exists():
        export_weights(best_ck, best_npz)
    best_puzzle = puzzle_score(
        args.puzzle_binary, args.puzzle_file, best_npz,
        args.puzzle_count, args.puzzle_depth, seed=args.puzzle_seed,
        min_rating=args.puzzle_min_rating, max_rating=args.puzzle_max_rating,
    )
    if best_puzzle >= 0:
        print(f"[loop] Initial puzzle score: {best_puzzle:.1f}%  (seed={args.puzzle_seed})")

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
        if args.anchor_data and Path(args.anchor_data).exists():
            inject_anchor_data(Path(args.anchor_data), Path("data/train.jsonl"), args.anchor_size)
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
            # train.py only saves when val_loss improves over the resume point.
            # If nothing was saved, treat the resumed checkpoint as the candidate
            # so the next iteration still fine-tunes from the latest state.
            print(f"[loop] Iteration {iteration}: no improvement during fine-tune — continuing from current best.")
        else:
            candidate_mae = load_val_mae(candidate_ck)
            candidate_npz = artifacts / f"candidate_iter{iteration}.npz"
            export_weights(candidate_ck, candidate_npz)

            # ── Evaluate candidate strength ───────────────────────────────
            # Fixed seed ensures candidate is always scored on the same puzzle set
            # as the stored best_puzzle — no need to re-run on the best model.
            cand_puzzle = puzzle_score(
                args.puzzle_binary, args.puzzle_file, candidate_npz,
                args.puzzle_count, args.puzzle_depth, seed=args.puzzle_seed,
                min_rating=args.puzzle_min_rating, max_rating=args.puzzle_max_rating,
            )
            cand_winrate = selfplay_winrate(
                args.selfplay_binary, args.engine,
                candidate_npz, best_npz,
                args.selfplay_eval_games, args.selfplay_eval_movetime,
                n_workers=args.selfplay_eval_workers,
            )

            print(f"[loop] Iteration {iteration}: candidate mae={candidate_mae:.1f}cp  best mae={best_mae:.1f}cp")
            if cand_puzzle >= 0:
                print(f"[loop] Puzzle score: candidate={cand_puzzle:.1f}%  best={best_puzzle:.1f}%")
            if cand_winrate >= 0:
                print(f"[loop] Self-play win rate (candidate vs best): {cand_winrate:.1f}%")

            # ── Promotion decision ────────────────────────────────────────
            # val_cp_mae on the self-play pool is a circular signal (model fits
            # its own data) and is NOT used for promotion.  Use external signals:
            #   - puzzle score >= best (equal or better tactical ability)
            #   - self-play win rate >= min_winrate (not a regression in play)
            # Both signals must pass when both tools are configured.
            # Falls back to each individual signal if only one is available.
            # If neither tool is configured, always promote with a warning.
            puzzle_ok   = cand_puzzle  >= best_puzzle - args.puzzle_regression_tolerance  if cand_puzzle  >= 0 and best_puzzle >= 0 else None
            winrate_ok  = cand_winrate >= args.selfplay_min_winrate                      if cand_winrate >= 0                     else None

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

            if promoted:
                best_ck = candidate_ck
                best_mae = candidate_mae
                best_puzzle = cand_puzzle
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

    print(f"\n[loop] Done. Best val_cp_mae={best_mae:.1f}  checkpoint={best_ck}")


if __name__ == "__main__":
    main()

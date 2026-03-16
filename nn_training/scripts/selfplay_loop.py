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
import json
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


def generate_data(
    engine_path: str,
    npz_path: Path | None,
    stockfish_path: str,
    output_path: Path,
    games: int,
    positions_per_game: int,
    movetime_ms: int,
    eval_depth: int,
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
    n_val = max(1000, int(len(lines) * val_fraction))
    n_train = len(lines) - n_val
    train_file.write_text("".join(lines[:n_train]), encoding="utf-8")
    val_file.write_text("".join(lines[n_train:]), encoding="utf-8")
    print(f"[loop] Split: {n_train} train / {n_val} val")


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
    ap.add_argument("--games-per-iter", type=int, default=5000,
                    help="Self-play games per iteration")
    ap.add_argument("--positions-per-game", type=int, default=4,
                    help="Positions sampled per self-play game")
    ap.add_argument("--movetime-ms", type=int, default=50,
                    help="Engine think time per move during self-play")
    ap.add_argument("--eval-depth", type=int, default=10,
                    help="Stockfish depth for labelling")
    ap.add_argument("--pool-size", type=int, default=1_000_000,
                    help="Max positions kept in the replay pool")
    ap.add_argument("--neural-mae-threshold", type=float, default=80.0,
                    help="Only use neural eval for self-play once val_cp_mae is below this (default 80)")
    ap.add_argument("--config", default="configs/finetune.yaml")
    ap.add_argument("--artifacts-dir", default="artifacts")
    args = ap.parse_args()

    artifacts = Path(args.artifacts_dir)
    artifacts.mkdir(exist_ok=True)

    best_ck = Path(args.initial_checkpoint)
    if not best_ck.exists():
        sys.exit(f"Checkpoint not found: {best_ck}")

    best_mae = load_val_mae(best_ck)
    print(f"[loop] Starting from {best_ck}  val_cp_mae={best_mae:.1f}")

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

        # 1. Export current best weights (always, in case we need them)
        npz_path = artifacts / "loop_weights.npz"
        export_weights(best_ck, npz_path)

        use_neural = best_mae < args.neural_mae_threshold
        print(f"[loop] Self-play engine: {'neural' if use_neural else 'classical'} eval  (mae={best_mae:.1f}, threshold={args.neural_mae_threshold})")

        # 2. Generate self-play data
        new_data = Path(f"data/selfplay_iter{iteration}.jsonl")
        generate_data(
            engine_path=args.engine,
            npz_path=npz_path if use_neural else None,
            stockfish_path=args.stockfish,
            output_path=new_data,
            games=args.games_per_iter,
            positions_per_game=args.positions_per_game,
            movetime_ms=args.movetime_ms,
            eval_depth=args.eval_depth,
        )

        # 3. Append to replay pool
        append_to_pool(new_data, pool_file, args.pool_size)
        new_data.unlink()  # remove per-iter file, it's in the pool now

        # 4. Split pool into train/val
        split_pool(pool_file, Path("data/train.jsonl"), Path("data/val.jsonl"))

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
            print(f"[loop] Iteration {iteration}: candidate={candidate_mae:.1f}cp  best={best_mae:.1f}cp")
            # Always advance to the latest checkpoint so training accumulates.
            # Only export to eval.npz (for deployment) when we actually improve.
            best_ck = candidate_ck
            if candidate_mae < best_mae:
                best_mae = candidate_mae
                shutil.copy(candidate_ck, artifacts / "best_checkpoint.pt")
                export_weights(artifacts / "best_checkpoint.pt", artifacts / "eval.npz")
                print(f"[loop] Improved! New best={best_mae:.1f}cp — exported weights → {artifacts / 'eval.npz'}")
            else:
                print(f"[loop] No MAE improvement, but advancing training base to iteration checkpoint.")

    print(f"\n[loop] Done. Best val_cp_mae={best_mae:.1f}  checkpoint={best_ck}")


if __name__ == "__main__":
    main()

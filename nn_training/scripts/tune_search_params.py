#!/usr/bin/env python3
"""tune_search_params.py — Bayesian search-parameter tuning with Optuna.

Objective: maximise puzzle solve rate measured by puzzle_bench with
deterministic settings (fixed seed, fixed puzzle set).

Usage:
    cd nn_training
    python3 scripts/tune_search_params.py \
        --puzzle-file  /path/to/lichess_db_puzzle.csv.zst \
        --eval-file    artifacts/eval_tactical_v2_test.npz \
        --puzzle-bench ../target/release/puzzle_bench \
        [--trials 200] \
        [--count 500] \
        [--min-rating 1500] \
        [--depth 12] \
        [--seed 42] \
        [--study-name search_tuning] \
        [--storage sqlite:///tuning.db]

Results are saved to a SQLite database (or in-memory if --storage is omitted)
so a run can be resumed with the same --study-name.

After the run, the best params are printed and saved to best_search_params.json.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def build_default_params() -> dict:
    """Current SearchParams::default() in Rust — keep in sync with search_params.rs."""
    return {
        "rfp_improving_margin":      46,
        "rfp_not_improving_margin":  147,
        "rfp_max_depth":             4,
        "futility_margin":           141,
        "futility_max_depth":        1,
        "delta_margin":              350,
        "probcut_margin":            339,
        "probcut_min_depth":         3,
        "nmp_excess_divisor":        283,
        "lmp_base":                  [0, 4, 15, 22, 39],
        "aspiration_delta":          139,
        "aspiration_min_depth":      4,
        "se_margin":                 81,
        "history_pruning_threshold": 722,
        "history_pruning_max_depth": 4,
    }


def build_baseline_params() -> dict:
    """Original hardcoded baseline before Optuna tuning — matches SearchParams::baseline()."""
    return {
        "rfp_improving_margin":      65,
        "rfp_not_improving_margin":  85,
        "rfp_max_depth":             7,
        "futility_margin":           200,
        "futility_max_depth":        3,
        "delta_margin":              250,
        "probcut_margin":            200,
        "probcut_min_depth":         5,
        "nmp_excess_divisor":        200,
        "lmp_base":                  [0, 4, 8, 13, 20],
        "aspiration_delta":          50,
        "aspiration_min_depth":      3,
        "se_margin":                 50,
        "history_pruning_threshold": 256,
        "history_pruning_max_depth": 3,
    }


def run_puzzle_bench(
    bench_bin:    str,
    puzzle_file:  str,
    eval_file:    str,
    params:       dict,
    count:        int,
    min_rating:   int,
    depth:        int,
    seed:         int,
    threads:      int = 0,
    movetime_ms:  "int | None" = None,
) -> float:
    """Write params JSON, run puzzle_bench, return solve rate in [0, 1]."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(params, f)
        params_path = f.name

    try:
        cmd = [
            bench_bin,
            "--file",       puzzle_file,
            "--eval-file",  eval_file,
            "--count",      str(count),
            "--min-rating", str(min_rating),
            "--seed",       str(seed),
            "--threads",    str(threads),
            "--fresh-tt",
            "--params",     params_path,
        ]
        if movetime_ms is not None:
            cmd += ["--movetime", str(movetime_ms)]
        else:
            cmd += ["--depth", str(depth)]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,  # 2-hour hard ceiling per trial
        )
        if result.returncode != 0:
            print(f"[warn] puzzle_bench failed:\n{result.stderr[:500]}", file=sys.stderr)
            return 0.0

        # Parse "Overall:  NNN/MMM  (PP.P%)" from stdout
        for line in result.stdout.splitlines():
            if line.strip().startswith("Overall:"):
                parts = line.split()
                # parts example: ['Overall:', '505/1000', '(50.5%)']
                frac = parts[1].split("/")
                if len(frac) == 2:
                    solved = int(frac[0])
                    total  = int(frac[1])
                    return solved / total if total > 0 else 0.0
        print(f"[warn] Could not parse Overall line from:\n{result.stdout[:500]}", file=sys.stderr)
        return 0.0
    finally:
        os.unlink(params_path)


def define_search_space(trial, args) -> dict:
    """Sample one set of SearchParams from the Optuna trial."""
    import optuna  # noqa: F401  (imported inside to allow --help without optuna)

    params = {}

    # RFP
    params["rfp_improving_margin"]     = trial.suggest_int("rfp_improving_margin",     30, 120)
    params["rfp_not_improving_margin"] = trial.suggest_int("rfp_not_improving_margin", 50, 150)
    params["rfp_max_depth"]            = trial.suggest_int("rfp_max_depth",            4, 10)

    # Futility
    params["futility_margin"]     = trial.suggest_int("futility_margin",     80, 400)
    params["futility_max_depth"]  = trial.suggest_int("futility_max_depth",  1, 5)

    # Delta
    params["delta_margin"] = trial.suggest_int("delta_margin", 100, 500)

    # ProbCut
    params["probcut_margin"]    = trial.suggest_int("probcut_margin",    80, 400)
    params["probcut_min_depth"] = trial.suggest_int("probcut_min_depth", 3, 7)

    # NMP
    params["nmp_excess_divisor"] = trial.suggest_int("nmp_excess_divisor", 80, 500)

    # LMP base table
    # Sample offsets that preserve the constraint: lmp[i] <= lmp[i+1].
    lmp0 = 0
    lmp1 = trial.suggest_int("lmp_base_1", 2,  8)
    lmp2 = trial.suggest_int("lmp_base_2", lmp1 + 1, 16)
    lmp3 = trial.suggest_int("lmp_base_3", lmp2 + 1, 25)
    lmp4 = trial.suggest_int("lmp_base_4", lmp3 + 1, 40)
    params["lmp_base"] = [lmp0, lmp1, lmp2, lmp3, lmp4]

    # Aspiration
    params["aspiration_delta"]     = trial.suggest_int("aspiration_delta",     20, 150)
    params["aspiration_min_depth"] = trial.suggest_int("aspiration_min_depth", 2, 5)

    # Singular Extension
    params["se_margin"] = trial.suggest_int("se_margin", 20, 120)

    # History pruning
    params["history_pruning_threshold"] = trial.suggest_int("history_pruning_threshold", 64, 1024)
    params["history_pruning_max_depth"] = trial.suggest_int("history_pruning_max_depth",  1, 6)

    return params


def trial_params_to_dict(trial) -> dict:
    """Reconstruct the full SearchParams dict from a completed Optuna trial."""
    params = {}
    for key, val in trial.params.items():
        if not key.startswith("lmp_base_"):
            params[key] = val
    params["lmp_base"] = [
        0,
        trial.params.get("lmp_base_1", 4),
        trial.params.get("lmp_base_2", 8),
        trial.params.get("lmp_base_3", 13),
        trial.params.get("lmp_base_4", 20),
    ]
    return params


def run_self_play(
    self_play_bin: str,
    engine_bin:    str,
    eval_file:     str,
    params:        dict,
    num_games:     int,
    movetime_ms:   int,
    threads:       int,
    opening_fens:  "str | None",
) -> float:
    """Run self_play: engine1 with candidate params vs engine2 with default params.

    Returns engine1 score in [0, 1].  A score > 0.5 means the candidate beats
    the current SearchParams::default() compiled into the binary.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(params, f)
        params_path = f.name

    try:
        cmd = [
            self_play_bin,
            engine_bin, engine_bin,
            "--games",    str(num_games),
            "--movetime", str(movetime_ms),
            "--no-ponder",
            "--engine1-opt", f"SearchParamsFile={params_path}",
            "--engine1-opt", f"EvalFile={eval_file}",
            "--engine1-opt", "NeuralEval=true",
            "--engine2-opt", f"EvalFile={eval_file}",
            "--engine2-opt", "NeuralEval=true",
        ]
        if threads > 0:
            cmd += [
                "--engine1-opt", f"Threads={threads}",
                "--engine2-opt", f"Threads={threads}",
            ]
        if opening_fens:
            cmd += ["--opening-fens", opening_fens]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        if result.returncode != 0:
            print(f"[warn] self_play failed:\n{result.stderr[:400]}", file=sys.stderr)
            return 0.5

        # The last line matching "score: X.X%" is always engine1's score.
        score = 0.5
        for line in result.stdout.splitlines():
            m = re.search(r'score:\s*([\d.]+)%', line)
            if m:
                score = float(m.group(1)) / 100.0
        return score
    finally:
        os.unlink(params_path)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--puzzle-file",  required=True,
                    help="lichess_db_puzzle.csv.zst")
    ap.add_argument("--eval-file",    required=True,
                    help="NNUE .npz file to evaluate against")
    ap.add_argument("--puzzle-bench", default="../target/release/puzzle_bench",
                    help="Path to the puzzle_bench binary")
    ap.add_argument("--trials",       type=int, default=200,
                    help="Number of Optuna trials")
    ap.add_argument("--count",        type=int, default=500,
                    help="Puzzles per trial (lower = faster, noisier)")
    ap.add_argument("--min-rating",   type=int, default=1500,
                    help="Minimum puzzle rating")
    ap.add_argument("--depth",        type=int, default=12,
                    help="Search depth per puzzle (ignored when --movetime is set)")
    ap.add_argument("--movetime",     type=int, default=None,
                    help="Search time per puzzle in ms (iterative deepening until deadline). "
                         "Use this instead of --depth to optimise for NNUE speed profile. "
                         "Typical: 50-100ms for fast tuning, 200ms for high-quality signal.")
    ap.add_argument("--seed",         type=int, default=42,
                    help="Fixed seed for reproducible puzzle sampling")
    ap.add_argument("--study-name",   default="search_tuning",
                    help="Optuna study name (for resumable runs)")
    ap.add_argument("--storage",      default=None,
                    help="Optuna storage URL, e.g. sqlite:///tuning.db")
    ap.add_argument("--out",          default="best_search_params.json",
                    help="Output file for best params")
    ap.add_argument("--threads",      type=int, default=0,
                    help="CPU threads for puzzle_bench (default 0=all)")

    # ── Phase 2: self-play validation ─────────────────────────────────────────
    ap.add_argument("--self-play-bin",  default=None,
                    help="Path to self_play binary (enables Phase 2)")
    ap.add_argument("--engine-bin",     default="../target/release/chess_uci",
                    help="Path to chess_uci binary used for self-play")
    ap.add_argument("--top-k",          type=int, default=5,
                    help="Top-K Optuna candidates to validate via self-play")
    ap.add_argument("--sp-games",       type=int, default=40,
                    help="Self-play games per candidate (default 40, SE≈7.9%%)")
    ap.add_argument("--sp-movetime",    type=int, default=100,
                    help="Self-play movetime in ms (default 100)")
    ap.add_argument("--sp-threads",     type=int, default=1,
                    help="Threads per engine in self-play (default 1 for clean param signal)")
    ap.add_argument("--sp-opening-fens", default=None,
                    help="Optional FEN file for self-play openings (one FEN per line)")
    args = ap.parse_args()

    try:
        import optuna
    except ImportError:
        print("optuna not installed. Run: pip install optuna", file=sys.stderr)
        sys.exit(1)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    mode_str = f"movetime={args.movetime}ms" if args.movetime else f"depth={args.depth}"
    print(f"Measuring baseline (default params, {mode_str})...", flush=True)
    baseline = run_puzzle_bench(
        args.puzzle_bench,
        args.puzzle_file,
        args.eval_file,
        build_default_params(),
        args.count,
        args.min_rating,
        args.depth,
        args.seed,
        args.threads,
        args.movetime,
    )
    print(f"Baseline solve rate: {baseline:.3%}  ({int(baseline * args.count)}/{args.count})")
    print()

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        storage=args.storage,
        load_if_exists=True,
    )

    def objective(trial) -> float:
        params = define_search_space(trial, args)
        rate = run_puzzle_bench(
            args.puzzle_bench,
            args.puzzle_file,
            args.eval_file,
            params,
            args.count,
            args.min_rating,
            args.depth,
            args.seed,
            args.threads,
            args.movetime,
        )
        # Print progress inline
        solved = int(rate * args.count)
        delta  = rate - baseline
        sign   = "+" if delta >= 0 else ""
        print(
            f"  trial {trial.number:4d}  rate={rate:.3%} ({solved}/{args.count})"
            f"  delta={sign}{delta:.3%}",
            flush=True,
        )
        return rate

    print(f"Running {args.trials} Optuna trials...")
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)

    # ── Phase 1 results ───────────────────────────────────────────────────────
    best_puzzle_trial = study.best_trial
    print()
    print("=" * 60)
    print(f"Phase 1 best: trial #{best_puzzle_trial.number}  "
          f"solve rate: {best_puzzle_trial.value:.3%}  "
          f"(+{best_puzzle_trial.value - baseline:+.3%} vs baseline)")

    best_params = trial_params_to_dict(best_puzzle_trial)

    # ── Phase 2: self-play validation of top-K candidates ────────────────────
    if args.self_play_bin:
        print()
        print("=" * 60)
        print(f"Phase 2: self-play validation  "
              f"(top {args.top_k} candidates, {args.sp_games} games each, "
              f"{args.sp_movetime}ms/move, {args.sp_threads} thread(s)/engine)")
        print("=" * 60)

        completed = [t for t in study.trials if t.value is not None]
        top_trials = sorted(completed, key=lambda t: t.value, reverse=True)[:args.top_k]

        print(f"  Puzzle scores of candidates: "
              f"{', '.join(f'{t.value:.3%}' for t in top_trials)}")
        print()

        sp_results = []
        for rank, trial in enumerate(top_trials):
            cparams = trial_params_to_dict(trial)
            print(f"  [{rank+1}/{len(top_trials)}] trial #{trial.number} "
                  f"(puzzle {trial.value:.3%}) ... ", end="", flush=True)
            sp_score = run_self_play(
                args.self_play_bin,
                args.engine_bin,
                args.eval_file,
                cparams,
                args.sp_games,
                args.sp_movetime,
                args.sp_threads,
                args.sp_opening_fens,
            )
            sign = "+" if sp_score > 0.5 else ""
            print(f"self-play score: {sp_score:.1%}  ({sign}{sp_score - 0.5:+.1%} vs default)")
            sp_results.append((sp_score, trial, cparams))

        sp_results.sort(key=lambda x: x[0], reverse=True)
        winner_score, winner_trial, winner_params = sp_results[0]

        print()
        print(f"  Self-play winner: trial #{winner_trial.number}  "
              f"puzzle={winner_trial.value:.3%}  self-play={winner_score:.1%}")

        # Use self-play winner (best among candidates, regardless of threshold).
        # If the puzzle-best and self-play-best differ, note it.
        best_params = winner_params
        if winner_trial.number != best_puzzle_trial.number:
            print(f"  Note: self-play winner differs from puzzle-best "
                  f"(trial #{best_puzzle_trial.number})")

    # ── Print and save ────────────────────────────────────────────────────────
    print()
    print("Best params:")
    for k, v in sorted(best_params.items()):
        default_v = build_default_params().get(k, "?")
        print(f"  {k:<35} {str(v):<20}  (default: {default_v})")

    out_path = Path(args.out)
    with open(out_path, "w") as f:
        json.dump(best_params, f, indent=2)
    print()
    print(f"Saved to {out_path}")
    print()
    print("To verify with full 2000-puzzle bench:")
    bench_bin = args.puzzle_bench
    eval_file = args.eval_file
    puzzle_file = args.puzzle_file
    print(f"  {bench_bin} \\")
    print(f"    --file {puzzle_file} \\")
    print(f"    --eval-file {eval_file} \\")
    print(f"    --count 2000 --min-rating 1500 --depth 12 --seed 42 --threads 32 --fresh-tt \\")
    print(f"    --params {out_path}")


if __name__ == "__main__":
    main()

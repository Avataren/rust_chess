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
import subprocess
import sys
import tempfile
from pathlib import Path


def build_default_params() -> dict:
    """Default values matching SearchParams::default() in Rust."""
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
            "--depth",      str(depth),
            "--seed",       str(seed),
            "--threads",    "0",          # all CPUs — fresh TT per puzzle guarantees no cross-pollution
            "--fresh-tt",                 # fresh TT per puzzle for fair comparison
            "--params",     params_path,
        ]
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
                    help="Search depth for each puzzle")
    ap.add_argument("--seed",         type=int, default=42,
                    help="Fixed seed for reproducible puzzle sampling")
    ap.add_argument("--study-name",   default="search_tuning",
                    help="Optuna study name (for resumable runs)")
    ap.add_argument("--storage",      default=None,
                    help="Optuna storage URL, e.g. sqlite:///tuning.db")
    ap.add_argument("--out",          default="best_search_params.json",
                    help="Output file for best params")
    args = ap.parse_args()

    try:
        import optuna
    except ImportError:
        print("optuna not installed. Run: pip install optuna", file=sys.stderr)
        sys.exit(1)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Baseline eval: measure the default params first so we can see improvement.
    print("Measuring baseline (default params)...", flush=True)
    baseline = run_puzzle_bench(
        args.puzzle_bench,
        args.puzzle_file,
        args.eval_file,
        build_default_params(),
        args.count,
        args.min_rating,
        args.depth,
        args.seed,
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

    best = study.best_trial
    print()
    print("=" * 60)
    print(f"Best trial: #{best.number}  solve rate: {best.value:.3%}")
    print(f"Improvement over baseline: {best.value - baseline:+.3%}")
    print()
    best_params = define_search_space.__wrapped__(best) if hasattr(define_search_space, '__wrapped__') else {}
    # Re-derive best params from best trial params dict
    best_params = {}
    for key, val in best.params.items():
        if key.startswith("lmp_base_"):
            pass  # handled below
        else:
            best_params[key] = val
    # Reconstruct lmp_base from individual params
    lmp1 = best.params.get("lmp_base_1", 4)
    lmp2 = best.params.get("lmp_base_2", 8)
    lmp3 = best.params.get("lmp_base_3", 13)
    lmp4 = best.params.get("lmp_base_4", 20)
    best_params["lmp_base"] = [0, lmp1, lmp2, lmp3, lmp4]

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

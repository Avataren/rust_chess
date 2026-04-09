# Benchmarking — Working Knowledge

There are four distinct benchmarks in this project that measure completely different things. Using the wrong one to draw conclusions is a common mistake.

---

## The four benchmarks at a glance

| Benchmark | Binary | Measures | Reliability |
|-----------|--------|----------|-------------|
| `puzzle_bench` | `target/release/puzzle_bench` | Tactical correctness vs Stockfish | Strong external signal |
| `bench` | `target/release/bench` (or `cargo run -p chess_evaluation --bin bench --release`) | Raw NPS / search speed | Engine performance only, not play quality |
| `self_play` | `target/release/self_play` | Head-to-head model vs model | Noisy at ≤40 games (SE≈7.9%) |
| `eval_mae.py` | Python script | CP-MAE on a fixed real-game position set | Best generalisation signal |

---

## puzzle_bench — the primary quality gate

Runs 2000 Lichess puzzles (rating ≥ 1500) at depth 7, fixed seed 42.

```bash
target/release/puzzle_bench \
  --file lichess_db_puzzle.csv.zst \
  --eval-file nn_training/artifacts/eval.npz \
  --count 2000 \
  --depth 7 \
  --seed 42 \
  --min-rating 1500 \
  --threads 0
```

Use the `/puzzle` command to run this with defaults.

**What it actually measures:** Whether the engine finds the Stockfish-optimal first move in sharp, tactical positions drawn from real games. Strong signal for pattern recognition and tactical sharpness.

**What it does NOT measure:** Positional understanding, long-term planning, time management, endgame technique. A model can improve puzzle score while quietly degrading in quiet positions.

**Gotchas:**
- The benchmark is **fully deterministic**: same seed → same 2000 puzzles → same TT state per puzzle (`--fresh-tt` is implied by `--threads > 1`). Tolerance in the self-play loop must be **0.0**. Any score change is real, not noise.
- `--threads 0` means "all available CPUs." Pass `--threads 1` for a single-threaded run if comparing against a single-threaded baseline.
- The `--eval-file` flag overrides the embedded weights. Always pass `nn_training/artifacts/eval.npz` explicitly when evaluating the current model.
- **Do not run this while data labeling is in progress.** Stockfish + puzzle_bench competing for all CPUs will slow labeling by 5-10x and mess up timing.

**Useful variants:**
```bash
# Only hard puzzles (signal for deep search quality)
--min-rating 2000 --max-rating 2800

# Fast smoke test
--count 200 --depth 5

# Export failures for fine-tuning data
--export-failures data/puzzle_failures.jsonl
```

---

## bench — raw search speed (NPS)

12 fixed positions across opening, middlegame, pawn endgame, rook endgame. Measures nodes/second, not quality.

```bash
# Quick default (depth 7, 1 thread, deterministic node count)
cargo run -p chess_evaluation --bin bench --release

# Multi-threaded with hash sweep (useful after changing TT or SMP code)
cargo run -p chess_evaluation --bin bench --release -- --depth 12 --threads 1,4,8,16,32

# Full hash × thread grid
cargo run -p chess_evaluation --bin bench --release -- --hash-sweep
```

**What it measures:** Nodes searched per second. Useful for catching performance regressions after touching `alpha_beta`, the NNUE forward pass, move generation, or the TT.

**What it does NOT measure:** Move quality, evaluation accuracy, tactical correctness.

**Gotchas:**
- Single-threaded mode (default, 1 thread, no `--hash`) uses a **deterministic fixed-depth search** — the node count is reproducible across runs. Multi-threaded mode uses iterative deepening + Lazy SMP, which has run-to-run variance in node count.
- Do not compare single-threaded and multi-threaded numbers directly — they use different search paths.
- `--hash-sweep` takes several minutes. Use only when profiling TT behaviour.
- NPS regressions of less than ~5% are within run-to-run noise for the threaded path. The sequential path is stable.

---

## self_play — head-to-head model comparison

Pits two engine instances against each other over N games. Used in the self-play loop to gate promotion.

```bash
target/release/self_play \
  target/release/chess_uci \
  target/release/chess_uci \
  --games 40 \
  --movetime 100 \
  --no-ponder \
  --engine1-opt EvalFile=nn_training/artifacts/eval.npz \
  --engine1-opt NeuralEval=true \
  --engine2-opt EvalFile=nn_training/artifacts/other.npz \
  --engine2-opt NeuralEval=true \
  --opening-fens /tmp/opening_fens.txt
```

**What it measures:** Whether engine1 scores better than chance against engine2.

**Critical limitation — sample size:**

| Games | Standard Error | Detectable difference (80% power) |
|-------|---------------|-----------------------------------|
| 40    | ±7.9%         | ~18% gap                          |
| 100   | ±5.0%         | ~12% gap                          |
| 200   | ±3.5%         | ~8% gap                           |
| 400   | ±2.5%         | ~6% gap                           |

40 games (current default in the loop) is a **regression guard only**. It cannot confirm improvement — a genuinely stronger model will still lose this test 20% of the time.

**Opening diversity:** Without `--opening-fens`, all games start from the initial position. The self-play loop generates a temp file from the 117-line opening book (`_OPENING_LINES` in `selfplay_loop.py`, sourced from `chess_evaluation/src/opening_book.rs`). Games are paired: game N and N+1 play the same opening FEN with sides swapped, then cycle to the next opening.

**Endgame benchmark (endgame_bench.sh):** A separate script that runs the self_play binary on fixed endgame FENs to test conversion correctness. It's Linux-path incompatible as written (hardcoded `.exe`). Run manually if needed:
```bash
target/release/self_play target/release/chess_uci target/release/chess_uci \
  --games 6 --movetime 200 \
  --fen "4k3/8/8/8/8/8/8/4K2Q w - - 0 1"
```

---

## eval_mae.py — generalisation check

Evaluates a checkpoint's CP-MAE on a fixed 10k sample of real-game positions (`data/gen_val.jsonl`).

```bash
cd nn_training
python3 scripts/eval_mae.py \
  --checkpoint artifacts/best_checkpoint.pt \
  --data data/gen_val.jsonl
# prints: cp_mae=<value>
```

**What it measures:** How accurately the model predicts Stockfish evaluations on positions from real GM games — not positions the model generated itself. A rising gen_val MAE while puzzle score rises is a warning sign that improvements are not transferring to real-game distributions.

**Gotchas:**
- `data/gen_val.jsonl` is created once at loop startup from `--anchor-data` with `seed=0` and never regenerated. The same 10k positions are used for all iterations and all restarts. Do not delete it.
- This is **informational only** in the loop — not a promotion gate. Watch the trend across iterations.
- Requires the Python venv to be active (torch + nnue_train). Run from `nn_training/` with the venv.

---

## Reading the signals together

After a promotion, you want to see:
- ✅ Puzzle score: flat or up (equal or better tactics)
- ✅ Gen-val MAE: flat or down (equal or better on real positions)
- ✅ Self-play winrate: ≥ 47% (didn't catastrophically regress)
- ✅ Lichess rating: trending up over weeks (ground truth)

Warning patterns:
- Puzzle up + gen-val up → overfitting to self-play tactical distribution, not generalising
- Puzzle flat + gen-val down → training is improving the model but puzzles aren't capturing it (could be positional improvement)
- Winrate near 50% every iteration → the 40-game gate is mostly noise; don't read into individual results

---

## Search parameter tuning (Optuna)

All search pruning constants (RFP, futility, delta, ProbCut, NMP, LMP, aspiration, SE, history)
are now in `SearchParams` and can be loaded via `puzzle_bench --params params.json`.
Use `tune_search_params.py` to find better values via Bayesian optimisation.

```bash
cd nn_training
pip install optuna       # one-time
python3 scripts/tune_search_params.py \
  --puzzle-file  /path/to/lichess_db_puzzle.csv.zst \
  --eval-file    artifacts/eval_tactical_v2_test.npz \
  --puzzle-bench ../target/release/puzzle_bench \
  --trials 200 \
  --count 500 \
  --min-rating 1500 \
  --depth 12 \
  --seed 42 \
  --storage sqlite:///tuning.db   # enables resume
```

Each trial: ~16s (500 puzzles × 32ms). 200 trials ≈ 53 min. The study is resumable —
re-run with the same `--study-name` and `--storage` to continue from where it left off.

After the run, verify the winner on the full benchmark:

```bash
target/release/puzzle_bench \
  --file /path/to/lichess_db_puzzle.csv.zst \
  --eval-file nn_training/artifacts/eval_tactical_v2_test.npz \
  --count 2000 --min-rating 1500 --depth 12 --seed 42 \
  --threads 32 --fresh-tt \
  --params best_search_params.json
```

**To use the best params permanently:** copy `best_search_params.json` to a committed path
(e.g. `chess_evaluation/search_params.json`) and load it at engine startup via a new UCI
`setoption name SearchParams` option — or hard-code the values back as the new
`SearchParams::default()` once they're confirmed better.

**Design note:** `--params` makes each puzzle's search single-threaded (no Lazy SMP per puzzle),
but puzzles are still solved in parallel across Rayon workers when `--threads N > 1` is passed.
Results are collected in original order so the overall solve rate is fully deterministic at any
thread count. Use `--threads 0` (all CPUs) with `--params` to maximize tuning speed.

---

## Before running any benchmark

- Check if the self-play loop is running (`ps aux | grep selfplay_loop`). The loop already runs `puzzle_bench` internally during evaluation — running it concurrently wastes CPU and can interfere with the loop's timing.
- Make sure the binary is up to date. If you changed Rust code: `cargo build --release -p chess_evaluation` (for bench/puzzle_bench) or `-p chess_uci` / `-p self_play`.
- Always pass `--eval-file nn_training/artifacts/eval.npz` to `puzzle_bench` and `bench` when evaluating the current model. Without it they use the weights compiled into the binary at build time, which may be stale.

Run the NPS / search-speed benchmark against the current best weights.

Arguments (optional, space-separated): `$ARGUMENTS`

Parse any of the following from `$ARGUMENTS` if provided:
- `--depth N` — search depth (default 7)
- `--threads T` — thread count or comma-separated list e.g. `1,4,8,16` (default 1)
- `--hash MB` — transposition table size in MB or comma-separated list (default: engine default)
- `--hash-sweep` — run the full predefined hash × thread grid at depth 12 (ignores other flags)
- `--eval-file PATH` — weights file (default: `nn_training/artifacts/eval.npz`)

Then run using the Bash tool. Examples:

**Default (depth 7, 1 thread, deterministic):**
```
target/release/bench --eval-file nn_training/artifacts/eval.npz --depth 7
```

**Multi-threaded sweep:**
```
target/release/bench --eval-file nn_training/artifacts/eval.npz --depth 12 --threads 1,4,8,16,32
```

**Full hash × thread grid:**
```
target/release/bench --eval-file nn_training/artifacts/eval.npz --hash-sweep
```

If `target/release/bench` doesn't exist or is stale, build it first:
```
cargo build --release -p chess_evaluation
```

After running, report the total nodes, total time, and average NPS. If multiple configurations were run, highlight the best avg NPS configuration.

Note: single-threaded mode (no `--threads`, no `--hash`) uses a deterministic fixed-depth search with a stable node count. Multi-threaded mode uses iterative deepening + Lazy SMP and has run-to-run variance — differences under ~5% are within noise.

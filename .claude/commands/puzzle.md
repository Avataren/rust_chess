Run the puzzle benchmark against the current best weights (`nn_training/artifacts/eval.npz`).

Arguments (optional, space-separated): `$ARGUMENTS`

Parse any of the following from `$ARGUMENTS` if provided:
- `--count N` — number of puzzles (default 2000)
- `--depth N` — search depth (default 7)
- `--seed N` — RNG seed (default 42)
- `--min-rating N` — minimum puzzle rating (default 1500)
- `--max-rating N` — maximum puzzle rating (default 0, no limit)

Then run this command using the Bash tool:

```
/home/avataren/src/rust_chess/target/release/puzzle_bench \
  --file /home/avataren/src/rust_chess/lichess_db_puzzle.csv.zst \
  --eval-file /home/avataren/src/rust_chess/nn_training/artifacts/eval.npz \
  --count <count> \
  --depth <depth> \
  --seed <seed> \
  --min-rating <min-rating> \
  --threads 0
```

Add `--max-rating <N>` only if the user specified it.

After running, show the full output and summarise the overall solve rate.

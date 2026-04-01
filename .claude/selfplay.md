# Self-Play Improvement Loop — Working Knowledge

Read this before touching anything related to self-play training, the loop, eval metrics, or the chess engine weights.

---

## What the loop does

Each iteration:
1. Generate self-play games with the current best engine → extract FENs
2. Label positions with Stockfish at depth 14 → JSONL
3. Append to FIFO replay pool (max 750k positions, keeps newest)
4. Split pool into train/val, inject 50k anchor positions into train only
5. Shuffle train.jsonl (anchor positions must be distributed, not batched at the end)
6. Fine-tune from best checkpoint for 5 epochs
7. Evaluate candidate: puzzle score + self-play winrate + gen_val MAE
8. Promote if both puzzle score and winrate pass; discard otherwise

---

## Canonical files — know these before touching anything

| File | Role |
|------|------|
| `nn_training/artifacts/eval.npz` | **The** weights file. Used by both the loop AND the lichess bot. Never overwrite on startup if it already exists. |
| `nn_training/artifacts/best_checkpoint.pt` | Best .pt checkpoint. Copied here on every promotion. |
| `nn_training/start_selfplay_loop.sh` | How to start the loop. |
| `nn_training/scripts/selfplay_loop.py` | Main loop logic. |
| `nn_training/scripts/generate_data.py` | Training data generation (self-play games + Stockfish labeling). |
| `nn_training/scripts/eval_mae.py` | Evaluates a .pt checkpoint's CP-MAE on any JSONL file. |
| `nn_training/configs/finetune.yaml` | Fine-tune config used each iteration. |
| `data/selfplay_pool.jsonl` | FIFO replay pool. Newest positions at the end. |
| `data/gen_val.jsonl` | Fixed 10k sample from anchor data (seed=0). Never regenerated once it exists. |
| `chess_evaluation/src/opening_book.rs` | 117 opening lines used in the self-play eval (head-to-head). |

---

## Promotion criteria

- **Puzzle score** (`puzzle_bench`, 2000 puzzles, depth 7, min_rating 1500): candidate must match or beat best. Tolerance is **0.0**. The seed is randomised at startup and rotates every 5 iterations (configurable via `--puzzle-seed-rotation-interval`); on rotation the best model is re-scored first so the baseline stays valid. Both models are always scored on the same seed within an iteration.
- **Self-play winrate** (40 games, candidate vs best): must be ≥ 47%. This is a regression guard only — SE≈7.9% means it cannot detect genuine improvement.
- **Gen-val MAE** (`data/gen_val.jsonl`): informational only, not a gate. Watch the trend (▲ = generalisation degrading). If it keeps rising while puzzle score rises, the improvements are not transferring to real-game positions.
- **val_cp_mae on self-play pool**: NOT used for promotion. It's circular — the model is being evaluated on positions it generated.

---

## Gotchas that have already burned us

**eval.npz startup overwrite**
The loop must NOT re-export weights to `eval.npz` at startup if the file already exists. The lichess bot reads this file live. Overwriting it mid-game causes the bot to switch weights unexpectedly. The check: `if not best_npz.exists(): export_weights(...)`.

**random.seed(42) in generate_data.py**
Was previously hardcoded, making all self-play games identical across iterations (only color alternation varied). Fixed to `random.seed()` (OS entropy). Do not reintroduce a fixed seed in the worker init or the batch loop.

**CP perspective in JSONL**
`cp` values are **white-absolute** (positive = white winning). The `JsonlDualPositionDataset` converts to side-to-move perspective internally. If you write new data generation code, use white-absolute CP or the training will be silently wrong.

**Puzzle regression tolerance**
Must stay at `0.0`. The benchmark is deterministic for a given seed — same seed, same puzzles, same TT state. Any score drop on the same seed is real, not noise. On seed rotation the baseline is always re-scored before comparing.

**Self-play eval openings**
Without the opening book, all eval games start from the initial position → only 2 distinct games repeated N/2 times. The `self_play` binary now accepts `--opening-fens <file>` and `selfplay_winrate()` generates a temp file from `_OPENING_LINES` (copied from `opening_book.rs`). Game pairs share an opening (game N and N+1 play the same FEN, sides swapped).

**Anchor data is train-only**
Injecting anchor positions into val.jsonl would contaminate the validation signal with non-self-play positions. Only inject into train.jsonl.

**Pool keeps newest**
`append_to_pool` trims with `lines[-pool_size:]` — it keeps the MOST RECENT positions. Older self-play data is evicted first. This is intentional: newer iterations play better chess.

**Syzygy tables are not used in self-play**
The `chess_uci` binary has Syzygy support but the self-play loop does not pass `SyzygyPath` as an engine option. This is intentional — TB probing during data generation would bias positions away from normal play.

---

## Eval signal quality (honest assessment)

| Signal | What it actually measures | Reliability |
|--------|--------------------------|-------------|
| Puzzle score | Tactical sharpness on real-game positions | Good external signal |
| Self-play winrate (40 games) | Catastrophic regression guard | Very noisy, SE≈7.9% |
| Gen-val MAE | Whether improvements transfer to real-game eval | Best generalization signal |
| Val-cp-mae (self-play pool) | How well the model fits its own distribution | Circular, not used |
| Lichess bot rating | Actual playing strength | Ground truth, but slow |

---

## Before making changes to the loop

1. Check if the loop is running (`ps aux | grep selfplay_loop`). Never kill it mid-iteration — data files and checkpoints may be in partial state.
2. Read the function you're modifying fully before editing. The loop has subtle ordering dependencies (export → puzzle → winrate → promote → update best_npz).
3. If changing Rust binaries (`self_play`, `puzzle_bench`, `chess_uci`), rebuild: `cargo build --release -p <crate>`.
4. If changing the dataset format or CP convention, check `nnue_train/dataset.py` to ensure the loader matches.
5. The loop is designed to be restartable: per-iter JSONL files are reused if they exist and are non-empty.

---

## Key invariant

`artifacts/eval.npz` always reflects `artifacts/best_checkpoint.pt`. Promotion copies the candidate NPZ over eval.npz in-place (`shutil.copy`). The lichess bot picks up the new weights on the next game without restart.

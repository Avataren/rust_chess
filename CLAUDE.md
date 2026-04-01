# Project orientation

This is a UCI chess engine written in Rust with a Python-based NNUE training pipeline. The two halves are connected by a single file: `nn_training/artifacts/eval.npz`.

**Crate responsibilities:**
- `chess_foundation` — types: pieces, moves, squares, flags
- `chess_board` — board state, make/undo move, FEN parsing
- `move_generator` — legal move generation, PieceConductor
- `chess_evaluation` — search (alpha-beta, iterative deepening, Lazy SMP), NNUE eval, classical HCE, Syzygy TB, opening book
- `chess_uci` — UCI protocol, time management, the playable engine binary
- `self_play` — head-to-head binary: plays two engine instances against each other for evaluation
- `puzzle_bench` — bin inside chess_evaluation: benchmarks puzzle solve rate

**Training pipeline** (`nn_training/`):
Self-play games → Stockfish labels → FIFO pool → fine-tune → evaluate → promote → `artifacts/eval.npz`

**System seams (non-obvious connections):**
- `nn_training/artifacts/eval.npz` — the only file shared between training and the live bot; updated in-place on every promotion; read by lichess-bot at game start via UCI `EvalFile` setoption
- `chess_evaluation/src/eval.npz` — separate file, baked into the binary at compile time; only updated manually before a rebuild; do not confuse with the above
- Syzygy tables (`/home/avataren/syzygy`) — used by the live engine and lichess bot; intentionally NOT used during self-play data generation or training

**What is intentionally disconnected:**
- Self-play training does not use Syzygy (would bias position distribution)
- Gen-val MAE is a soft promotion gate (blocks if CP-MAE rises >5% above best); not a hard gate
- The self-play winrate (150 games, SE≈4.1%) is a regression guard, not a confirmation of improvement

# Code exploration

Prefer codebase-memory MCP tools over Grep+Read for open-ended exploration:

- Use `mcp__codebase-memory-mcp__search_code` when you don't know the exact file or symbol name
- Use `mcp__codebase-memory-mcp__trace_call_path` before reading intermediate files to understand call chains
- Use `mcp__codebase-memory-mcp__get_architecture` at the start of unfamiliar tasks instead of manually exploring multiple files
- Use `mcp__codebase-memory-mcp__query_graph` to find all callers/callees of a function

Fall back to Grep/Read when you already know the exact file and line range.

# Domain knowledge

Before working on the self-play training loop, engine weights, eval metrics, or anything in `nn_training/`:
→ Read `.claude/selfplay.md` first. It contains gotchas, the promotion criteria, canonical file locations, and known failure modes that are not derivable from the code alone.

Before running or interpreting any benchmark (puzzle_bench, bench NPS, self_play, eval_mae):
→ Read `.claude/benchmarking.md` first. It covers what each benchmark actually measures, its limitations, correct invocation, and how to read the signals together.

Before touching evaluation code, Cargo features, NNUE weights, the accumulator, or anything in `chess_evaluation/`:
→ Read `.claude/eval_system.md` first. It documents the four Cargo features, the two eval.npz files (embedded vs runtime), NNUE architecture, incremental accumulator, and common mistakes.

Before touching `chess_uci/src/tb.rs`, Syzygy probe logic, or endgame engine behaviour:
→ Read `.claude/syzygy.md` first. It covers the DTZ-vs-WDL distinction, the AmbiguousWdl type gotcha, castle move translation, the queen sacrifice bug and why DTZ fixes it, and the test suite.

Before writing data generation code, touching dataset loaders, or working with CP values:
→ Read `.claude/data_pipeline.md` first. The CP perspective convention (side-to-move in JSONL, white-absolute in binary) is a silent failure mode — wrong perspective trains on inverted evaluations with no warning.

Before writing any terminal progress bar or status line (tqdm, `\r`, ANSI, etc.):
→ Read `.claude/terminal_progress.md` first. The `\r` and fixed-width tqdm patterns break on terminal resize. The fixes are: `dynamic_ncols=True` on every tqdm call, and `\r\033[K` + width clamp for manual status lines.

# Reasoning approach

**GAN thinking framework** — use this when analysing complex feedback loops (training pipelines, eval systems, iterative processes):
- **Generator**: what produces the data or signal? Is it adaptive, or does it produce the same thing regardless of downstream results?
- **Discriminator**: what validates quality or gates promotion? Is it sharp enough to distinguish real improvement from noise? Does it cover all failure modes (tactical *and* positional *and* generalization)?
- **Feedback path**: does the discriminator's output flow back to improve the generator, or is information discarded after each decision?
- **Mode collapse risk**: does the system converge to a narrow distribution over time? What forces diversity?

Applied to this project: generator = self-play + Stockfish oracle; discriminator = puzzle gate + winrate gate + gen_val gate; feedback = puzzle failures injected as anchor data next iteration.

**Before proposing or implementing any change, check:**
1. **Regression risk** — does this touch a path that affects the live bot (`artifacts/eval.npz`), the promotion gate, or the data format? If so, what is the worst-case silent failure mode?
2. **Compute cost** — self-play eval (150 games), puzzle bench (2000 puzzles), and Stockfish labeling are all expensive. Don't add steps that run unconditionally when they can be short-circuited or batched. If the puzzle gate already fails, skip the winrate eval.
3. **Data invariants** — CP perspective, pool FIFO ordering, anchor-only-in-train, fixed gen_val seed. Any code touching JSONL files must respect these or training silently degrades.

**Training run commands MUST include a TensorBoard log directory.**
Always append `--tb-logdir runs/<descriptive-name>` to every `train.py` invocation. Never give the user a training command without it — a run that can't be monitored on TensorBoard has to be restarted from scratch. Example:
```bash
PYTHONPATH=. python3 scripts/train.py --config configs/halfkp_dual_selfplay_768_64_8b.yaml --tb-logdir runs/768x64_from_scratch
```
To resume with logs without changing training outcome (full state is saved — model, optimizer, scheduler, epoch):
```bash
PYTHONPATH=. python3 scripts/train.py --config <cfg> --resume artifacts/checkpoint.pt --tb-logdir runs/<name>
```

**Component boundary rules** — bugs in this codebase have repeatedly come from assuming an interface instead of verifying it:
- **Subprocess output parsers**: Read the binary source (e.g. `self_play/src/main.rs`) to find the exact output format before writing any parser. The engine name in `self_play` output is derived from `file_stem()`, not a fixed label.
- **Iteration-indexed files** (e.g. `puzzle_failures_iter{N}.jsonl`): Trace the read path and write path independently with a concrete N to confirm the offset is correct before writing.
- **Cached files** (e.g. `data/opening_fens.txt`): A cache that only regenerates when missing silently serves stale data after code changes. Always consider the "file exists but is stale" scenario.
- **Opening/move sequences**: Validate programmatically (`_opening_fens()` in the venv) before committing — illegal moves fail silently at runtime.

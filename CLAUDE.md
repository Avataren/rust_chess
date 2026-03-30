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
- Gen-val MAE is logged but not a promotion gate (observational only)
- The 40-game self-play winrate is a regression guard, not a confirmation of improvement

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

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

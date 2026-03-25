# Code exploration

Prefer codebase-memory MCP tools over Grep+Read for open-ended exploration:

- Use `mcp__codebase-memory-mcp__search_code` when you don't know the exact file or symbol name
- Use `mcp__codebase-memory-mcp__trace_call_path` before reading intermediate files to understand call chains
- Use `mcp__codebase-memory-mcp__get_architecture` at the start of unfamiliar tasks instead of manually exploring multiple files
- Use `mcp__codebase-memory-mcp__query_graph` to find all callers/callees of a function

Fall back to Grep/Read when you already know the exact file and line range.

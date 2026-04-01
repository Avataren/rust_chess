# Terminal Progress Bars — How to Do Them Correctly

Read this before writing any terminal progress bar or status line in this project.

---

## The Problem

Two failure modes, both caused by ignoring terminal width:

**`\r` wraps and orphans lines.** `\r` moves the cursor to column 0 of the *current* line.
If the previous output was longer than the terminal and wrapped, the cursor lands at column 0
of the *last wrapped line*, not the beginning of the progress line. The next update prints
from there, leaving orphaned text above it. Resize the terminal narrower than the message
and every update creates a new line instead of overwriting.

**`tqdm` without `dynamic_ncols=True` uses a stale width.** `tqdm` measures terminal width
once at construction. Resize the terminal and it over-fills (wraps) or under-fills forever.

---

## The Fix

### Rule 1 — always use `dynamic_ncols=True` with tqdm

```python
# BAD
tqdm(iterable, desc="labeling", total=n)

# GOOD
tqdm(iterable, desc="labeling", total=n, dynamic_ncols=True)
```

`dynamic_ncols=True` makes tqdm call `shutil.get_terminal_size()` on every redraw.
It re-measures after resize and redraws within the new width. Nothing else is needed.

### Rule 2 — never use bare `\r`; use `\r\033[K` + width clamp

When you need a simple one-liner status (no ETA, no bar) and don't want the tqdm import:

```python
import shutil

def _term_print(msg: str) -> None:
    """Overwrite the current line, safe under terminal resize."""
    width = shutil.get_terminal_size().columns
    # \r   — return to column 0
    # \033[K — erase from cursor to end of line (removes leftover chars from wider previous text)
    # [:width-1] — clamp so the text never wraps (wrapping breaks \r overwrite)
    print(f"\r\033[K{msg[:width - 1]}", end="", flush=True)
```

To finish the progress line (newline + optional summary):
```python
print()           # blank newline
# or
print(f"\r\033[K{summary}")  # replace with final summary, then…
print()           # …move to next line
```

**Never add trailing spaces to pad.** `\033[K` erases the remainder of the line so
padding is unnecessary and just clutters the code.

### Rule 3 — no fixed-width bars computed from a hardcoded column count

If you build a `[####    ]` bar manually, derive its width from
`shutil.get_terminal_size().columns` each time you draw, not from a constant.

---

## Rust

The project's Rust binaries (`puzzle_bench`, `self_play`) print per-line or append-only
output — no `\r` patterns. Leave them alone; they don't have resize issues.

If Rust progress is ever needed, use the [`indicatif`](https://docs.rs/indicatif) crate,
which handles resize correctly out of the box.

---

## Quick checklist

- [ ] Every `tqdm(...)` call has `dynamic_ncols=True`
- [ ] No bare `\r` — only `\r\033[K` followed by text clamped to `shutil.get_terminal_size().columns - 1`
- [ ] No manually computed fixed-width bar columns

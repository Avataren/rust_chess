# Syzygy Tablebase Integration — Working Knowledge

Read this before touching `chess_uci/src/tb.rs`, Syzygy probe logic, or any endgame-related engine behaviour.

---

## Where the code lives

- `chess_uci/src/tb.rs` — all probe logic, move translation, tests
- `chess_uci/src/main.rs` — UCI setoption handlers for `SyzygyPath` and `SyzygyProbeLimit`
- Dependency: `shakmaty-syzygy = "0.28"` (the API changed significantly from 0.x)
- Tablebase files live at `/home/avataren/syzygy` (7-piece)

---

## Initialisation

TB is stored in a global `OnceLock<Mutex<Option<Tablebase<Chess>>>>`. It's initialised when the UCI engine receives:
```
setoption name SyzygyPath value /home/avataren/syzygy
```
This loads all `.rtbz` (DTZ) and `.rtbw` (WDL) files from the directory. On success, `syzygy_probe_limit` is automatically set to `tb::max_pieces()` so positions up to the max available piece count are always probed. The probe limit can be manually overridden via `setoption name SyzygyProbeLimit value N`.

**Probe limit in self-play:** The self-play training loop does NOT pass `SyzygyPath`. This is intentional — TB probing during self-play would bias the position distribution away from normal engine play. Do not add it.

---

## Probe strategy in `probe_root`

Two-tier strategy, always DTZ-first:

### Tier 1 — DTZ (primary, requires `.rtbz` files)
```rust
if let Ok(Some(sm)) = tb.best_move(&chess_pos) { ... }
```
`best_move()` selects:
- **Winning positions**: the move that minimises Distance-To-Zero (fastest path to a pawn move or capture that resets the 50-move clock, eventually forcing checkmate)
- **Losing positions**: the move that maximises DTZ (delays the loss as long as possible)
- **Drawing positions**: any move that maintains the draw

This is the correct strategy. It prevents both piece sacrifices (a WDL-only prober picks captures first due to move ordering) and draw-by-repetition loops (which WDL cannot detect).

### Tier 2 — WDL fallback (only when `.rtbz` absent)
When `best_move()` fails (no DTZ files for this piece count), fall back to probing every legal move's resulting position with `probe_wdl_after_zeroing()`, then pick the best by:
1. Highest WDL rank (Win > CursedWin > Draw > BlessedLoss > Loss)
2. Tie-break on piece count of the resulting position — **more pieces remaining wins ties**. This prevents accidental captures in equal-WDL situations (the bug we fixed: a WDL-only prober was picking queen captures first because captures sort before quiet moves).

---

## The `AmbiguousWdl` type (shakmaty-syzygy 0.28 gotcha)

`probe_wdl` returns `AmbiguousWdl`, not `Wdl`. These are different types. `AmbiguousWdl` has extra variants that account for 50-move clock uncertainty:

| AmbiguousWdl variant | Meaning | Our rank |
|----------------------|---------|----------|
| `Win` | Definite win | 2 |
| `MaybeWin` | Win, but 50-move clock might prevent it | 1 (treat as CursedWin) |
| `CursedWin` | Technically winning but draw by 50-move rule | 1 |
| `Draw` | Draw | 0 |
| `BlessedLoss` | Technically losing but draw by 50-move rule | -1 |
| `MaybeLoss` | Loss, but 50-move clock might save it | -1 (treat as BlessedLoss) |
| `Loss` | Definite loss | -2 |

**Conservative mapping:** `MaybeWin` is treated as rank 1 (not 2), and `MaybeLoss` as rank -1 (not -2). This is intentional — when the outcome is uncertain due to the 50-move clock, we don't claim a definite win/loss. Treating `MaybeWin` as rank 2 would cause the engine to sacrifice material expecting a win it might not achieve.

If you upgrade `shakmaty-syzygy`, check whether this type changes — it has broken things before.

---

## Move translation: `match_shakmaty_move`

Converts a shakmaty `Move` to our `ChessMove` by matching against the legal move list.

**Square encoding is identical** in both libraries: `sq = rank * 8 + file`, a1=0, h8=63. No remapping needed for normal moves.

**Castle moves require special handling.** Shakmaty provides (king_square, rook_square); we need (king_from, king_to):
```rust
SMove::Castle { king, rook } => {
    let king_sq = u16::from(*king);
    let rook_file = u16::from(*rook) % 8;
    let king_rank = king_sq / 8;
    let king_to = king_rank * 8 + if rook_file == 7 { 6 } else { 2 };
    (king_sq, king_to)
}
```
- Rook on h-file (file 7) → kingside → king goes to g-file (6): `king_rank * 8 + 6`
- Rook on a-file (file 0) → queenside → king goes to c-file (2): `king_rank * 8 + 2`
- Works for both white (rank 0) and black (rank 7) castling

**Promotion moves:** Match the shakmaty `Role` (Queen/Knight/Rook/Bishop) against the corresponding `ChessMove` promotion flag. A non-promotion ChessMove must NOT match a promotion shakmaty move, and vice versa.

**En passant:** Treated identically to normal moves — the from/to squares are sufficient.

---

## Position conversion

`board_to_chess(board) -> Result<Chess, _>` converts our `ChessBoard` to shakmaty's `Chess` via FEN round-trip using `CastlingMode::Standard`. If the position is invalid (impossible piece placement, etc.) it returns an error and `probe_root` returns `None`.

---

## WDL rank → score mapping

| Rank | Score (cp) | Label |
|------|-----------|-------|
| 2 (Win) | +20,000 | "TB Win" |
| 1 (CursedWin) | +100 | "TB Cursed Win" |
| 0 (Draw) | 0 | "TB Draw" |
| -1 (BlessedLoss) | -100 | "TB Blessed Loss" |
| -2 (Loss) | -20,000 | "TB Loss" |

These scores are reported in UCI `info` output and used by the search for move ordering in TB positions.

---

## Gotchas

**The queen sacrifice bug (fixed)**
WDL-only probing iterates legal moves and captures sort first (MVV-LVA ordering in `ChessMove::Ord`). In KQvKR with a queen-can-capture-rook-but-leads-to-draw position, the WDL prober would pick the queen capture (Draw) over a quiet winning move. DTZ's `best_move()` selects the fastest-winning path and avoids this entirely. The WDL fallback now uses a piece-count tiebreak to also avoid accidental captures.

**Do not use `probe_wdl` directly on the root position for move selection.** It only tells you the outcome of the current position, not which move achieves the best outcome. You must probe after each candidate move (or use `best_move()`).

**Draw-by-repetition is invisible to WDL probing.** WDL only knows Win/Draw/Loss; it cannot detect that repeating the position three times leads to a draw even from a winning position. DTZ prevents repetition by always choosing the move that makes progress (decreasing DTZ).

**The TB files at `/home/avataren/syzygy` include 7-piece tables.** Probe limit is auto-set to `max_pieces()` on init. In 8+ piece positions, TB is not consulted and the engine plays normally.

---

## Tests

All in `chess_uci/src/tb.rs`. Unit tests (no TB files needed):
- `match_normal_move` — e2e4 (sq 12→28)
- `match_normal_does_not_match_promotion_chessmove` — type safety check
- `match_queen_promotion`, `match_knight_promotion_not_confused_with_queen_promotion`
- `match_white_kingside_castle`, `match_white_queenside_castle`, `match_black_kingside_castle`
- `match_returns_none_when_list_empty`

Integration tests (`#[ignore]`, require `/home/avataren/syzygy`):
- `tb_avoids_queen_sacrifice_that_leads_to_draw` — FEN `8/8/8/4k3/Q3r3/8/8/K7 w - - 0 1` (KQvKR): asserts the returned move is NOT a4→e4 (24→28, the losing queen capture)
- `tb_kqk_reports_win` — FEN `7k/8/3Q4/8/8/K7/8/8 w - - 0 1` (KQvK): asserts a move is found and labelled "TB Win"

Run integration tests: `cargo test -p chess_uci -- --ignored`

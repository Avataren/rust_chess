# Refactoring Findings — rust_chess Engine

Analysis of the chess engine codebase for clarity, separation of concerns, and performance opportunities.

---

## Critical Priority

### 1. White/Black Loop Duplication in `alpha_beta.rs`

**Location:** Lines ~1365–1577 (white) and ~1591–1810 (black) in `alpha_beta.rs`; repeated again in `alpha_beta_root()` (~1890–1983 white, ~1984–2078 black).

The main search loop and root search loop each have two nearly identical branches — one for white, one for black. The only differences are sign-flipped comparisons (`>= beta` vs `<= alpha`), and everything else (futility pruning, LMP, LMR, PVS, beta-cutoff recording, accumulator updates) is verbatim copy-paste.

**Estimated duplication:** ~250 lines in the main loop, ~90 lines in root search.

**Fix:** Unify with a `const IS_WHITE: bool` const-generic parameter, or a sign-flip helper that converts the perspective to always-maximizing. Either eliminates the duplication entirely.

---

### 2. `SearchContext` is Monolithic

**Location:** `alpha_beta.rs` lines ~218–253, used throughout.

`SearchContext` mixes three unrelated subsystems:
- **Move ordering heuristics:** `killers`, `history`, `capture_history`, `cont_hist_1/2`, `countermoves`
- **Neural accumulator state:** `acc_white`, `acc_black`, `acc_valid`, `prev_moves`
- **Scratch buffers:** `move_lists`, `pseudo_buf`, `good_captures_buf`, `bad_captures_buf`, etc.

Quiescence search only needs the accumulator and scratch buffers, but receives the entire context. This creates misleading coupling and makes it harder to reason about what each search phase actually uses.

**Fix:** Split into `MoveOrderingContext`, `AccumulatorState`, and `ScratchBuffers`. Quiescence takes only the latter two.

---

### 3. `order_moves()` Has 17 Parameters

**Location:** `alpha_beta.rs` lines ~717–836.

```rust
fn order_moves(
    moves, tt_move, killers, countermove,
    history, capture_history, ch1, ch2,
    prev1, prev2,
    board, conductor, is_white,
    good_captures_buf, bad_captures_buf, quiets_buf, killer_entries_buf,
)
```

Seventeen parameters is a strong signal that the function's inputs need structure. The history tables are logically one thing; the previous move info is logically one thing; the scratch buffers are logically one thing.

**Fix:**
```rust
struct MoveOrderingHeuristics<'a> {
    history: &'a [[i32; 64]; 64],
    capture_history: &'a [[i32; 64]; 64],
    cont_hist_1: &'a ContHistTable,
    cont_hist_2: &'a ContHistTable,
    prev1: Option<(usize, usize)>,
    prev2: Option<(usize, usize)>,
}

fn order_moves(
    moves, tt_move, killers, countermove,
    heuristics: &MoveOrderingHeuristics,
    board, conductor, is_white,
    scratch: &mut OrderingScratchBuffers,
)
```

Reduces to 9 parameters; each group is self-documenting.

---

## High Priority

### 4. `handle_async_moves` Has 16 Parameters and Expensive Clones

**Location:** `chess_event_handler.rs` lines ~303–320.

The Bevy system takes 16 parameters (Bevy's limit), making it hard to read and reason about. Additionally, every search dispatch clones `ChessBoard` (~1 KB) and `PieceConductor` (~30 KB with magic tables) — this happens on every AI move and on every ponder dispatch.

**Fix for parameters:** Bundle related resources:
```rust
#[derive(Resource)]
pub struct GameBoardState { pub board: ChessBoard, pub conductor: PieceConductor }

#[derive(Resource)]
pub struct GameUIState {
    pub game_over_state: GameOverState,
    pub pending_game_over: PendingGameOver,
    pub last_move: LastMove,
    pub is_ai_thinking: IsAiThinking,
    pub pending_move_sound: PendingMoveSound,
}
```

**Fix for clones:** Use `Arc<ChessBoard>` and `Arc<PieceConductor>` for async search tasks. Ponder searches can hold the Arc without deep-copying; only clone on actual board mutations.

---

### 5. Ponder State Transitions Are Implicit

**Location:** `chess_event_handler.rs` lines ~328–332, ~357–366, and scattered elsewhere.

Three fields (`ponder_active`, `main_search_active`, `stops: Vec<Arc<AtomicBool>>`) are mutated across multiple callsites with no central coordination. It is easy to update one field and forget another, leading to subtle bugs (e.g., stops not cleared, AI still thinking flag not reset).

**Fix:** Add an explicit state machine:
```rust
enum SearchState { Idle, SearchingMain, PonderingBackground }

impl PonderState {
    pub fn cancel_all(&mut self) {
        for s in &self.stops { s.store(true, Ordering::Relaxed); }
        self.stops.clear();
        self.state = SearchState::Idle;
    }
}
```

All state transitions go through methods on `PonderState`, making illegal states unrepresentable.

---

### 6. Repeated `acc_push` / `acc_recompute` Pattern

**Location:** `alpha_beta.rs` lines ~1135–1139, ~1428–1432, ~1647–1651, ~1898–1902, ~1992–1996.

This pattern appears at every `make_move` callsite:
```rust
let king_moved = ctx.acc_push(ply, &mv, board);
board.make_move(&mut mv);
if king_moved {
    ctx.acc_recompute(ply + 1, board);
}
```

Five copy-paste sites. One typo (wrong ply, wrong board) would silently corrupt the accumulator.

**Fix:**
```rust
fn make_move_with_acc(board: &mut ChessBoard, ctx: &mut SearchContext, mv: &mut ChessMove, ply: usize) {
    let king_moved = ctx.acc_push(ply, mv, board);
    board.make_move(mv);
    if king_moved { ctx.acc_recompute(ply + 1, board); }
}
```

---

## Medium Priority

### 7. Mate Distance Logic Scattered Across 4 Locations

**Location:** `alpha_beta.rs` lines ~917–925, ~145–155, ~163–181, and within the root search.

The formula `±(1_000_000 - ply)` and the threshold checks (`score > MATE_SCORE_THRESHOLD`) appear in multiple places. A future change (e.g., adjusting the mate score base) requires touching all four sites consistently.

**Fix:** A small helper module:
```rust
const MATE_BASE: i32 = 1_000_000;
const MATE_THRESHOLD: i32 = 999_000;

fn mate_score(ply: usize) -> i32 { MATE_BASE - ply as i32 }
fn is_mate_score(score: i32) -> bool { score.abs() > MATE_THRESHOLD }
fn encode_for_tt(score: i32, ply: usize) -> i32 { ... }
fn decode_from_tt(score: i32, ply: usize) -> i32 { ... }
```

---

### 8. `classical_eval.rs` Is a Monolith (810 lines)

**Location:** `chess_evaluation/src/classical_eval.rs`.

Pawn hash table logic, material + PST accumulation, king safety scoring, and passed pawn detection are all in one file with no internal module boundaries. Each concern is independently testable and tuneable, but they can't be tested in isolation.

**Fix:** Extract into submodules:
- `pawn_eval.rs` — pawn hash, structure (isolated, doubled, passed)
- `king_safety.rs` — shield, attack count, threat bonuses
- `material.rs` — piece values + PST accumulation
- `classical_eval.rs` — orchestration only

---

### 9. `chessboard.rs` Mixes Four Concerns (1065 lines)

**Location:** `chess_board/src/chessboard.rs`.

The file handles: bitboard position state, move application (make/undo), Zobrist hash management, repetition detection, and FEN parsing. These are separate responsibilities that happen to share a struct.

**Fix:** Decompose into focused types:
```rust
struct BoardPosition { /* 8 bitboards, castling, ep */ }
struct BoardHistory { /* move_history, position_history, halfmove_clock */ }
struct ZobristHasher { /* hash state, compute/update */ }

pub struct ChessBoard {
    position: BoardPosition,
    history: BoardHistory,
    hasher: ZobristHasher,
}
```

`ChessBoard` becomes a thin facade that coordinates the three. Each inner type can be tested and reasoned about independently.

---

### 10. PVS Pattern Duplicated Three Times

**Location:** `alpha_beta.rs` lines ~1512–1546 (main white), ~1728–1767 (main black), ~2016–2047 (root black).

The null-window + conditional re-search pattern:
```rust
let score = alpha_beta(..., alpha, alpha + 1, ...);
if score > alpha && score < beta {
    alpha_beta(..., alpha, beta, ...)
}
```

appears three times with only sign differences. Should be a helper.

---

## Low Priority

### 11. Duplicate Docstring in `transposition_table.rs`

**Location:** Lines ~114–128.

The docstring for `store()` is copy-pasted in full. Remove the duplicate block.

---

### 12. `AGE_COST` Rationale Not Documented

**Location:** `transposition_table.rs` line ~13.

```rust
const AGE_COST: i32 = 4;
```

The value `4` is correct (a 3-generation-old depth-12 entry becomes equivalent to depth-0) but the reasoning isn't captured. Add a comment explaining the derivation so future tuning attempts have context.

---

### 13. HalfKP Feature Index Formula Repeated in `neural_eval.rs`

**Location:** `neural_eval.rs` lines ~361–367, ~377–384, ~395–402, ~408–415.

The accumulator index calculation `slot * 64 * KING_BUCKETS + square * KING_BUCKETS + bucket` appears 4+ times. A single typo in any copy would silently corrupt incremental updates.

**Fix:** Extract `fn halfkp_feature_idx(slot, square, bucket) -> usize`.

---

### 14. `game_resources.rs` — Scattered Single-Field Resources

**Location:** `chess/src/game_resources.rs` (~261 lines).

`TimeControl`, `Strength`, `PlayerColor`, `GamePhase`, `GameOverState` are all separate resources that are almost always queried together. Grouping them into `GameSettings` and `GameState` reduces boilerplate in every system that needs more than one.

---

## Summary Table

| Priority | File | Issue | Action |
|----------|------|-------|--------|
| Critical | `alpha_beta.rs` | White/black loop duplication (~340 lines) | Const-generic or sign-flip unification |
| Critical | `alpha_beta.rs` | `SearchContext` mixes 3 concerns | Split into 3 structs |
| Critical | `alpha_beta.rs` | `order_moves` has 17 params | Bundle into 3 structs, reduce to 9 |
| High | `chess_event_handler.rs` | 16-param system + expensive clones | Bundle resources; use Arc for clones |
| High | `chess_event_handler.rs` | Implicit ponder state transitions | State machine with transition methods |
| High | `alpha_beta.rs` | `acc_push/recompute` pattern ×5 | Extract `make_move_with_acc()` |
| Medium | `alpha_beta.rs` | Mate distance logic in 4 places | Centralize in helper module |
| Medium | `classical_eval.rs` | 810-line monolith | Split into pawn/king/material modules |
| Medium | `chessboard.rs` | 1065-line mixed concerns | Decompose into Position/History/Hasher |
| Medium | `alpha_beta.rs` | PVS pattern duplicated ×3 | Extract PVS helper |
| Low | `transposition_table.rs` | Duplicate docstring | Delete duplicate |
| Low | `transposition_table.rs` | `AGE_COST` unexplained | Add derivation comment |
| Low | `neural_eval.rs` | Feature index formula ×4 | Extract `halfkp_feature_idx()` |
| Low | `game_resources.rs` | Scattered resources | Consolidate into `GameSettings`/`GameState` |

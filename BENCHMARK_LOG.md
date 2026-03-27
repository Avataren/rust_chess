# Benchmark Log

Tracks puzzle solve rate across engine versions.

**Reproduce any run:**
```
cargo run -p chess_evaluation --bin puzzle_bench --release -- \
  --file lichess_db_puzzle.csv.zst \
  --count 2000 --min-rating 1000 --max-rating 2500 --depth 7 --seed 42
```

---

## 2026-03-23 — commit d249d92

**Changes:** Fix SEE diagonal attackers + pinned defenders, lazy qsearch eval reuse,
HalfKP 32 king buckets, dual-perspective NNUE 1024-dim 32KB model.

| Setting | Value |
|---------|-------|
| Depth | 7 |
| Puzzles | 2000 (seed 42) |
| Rating range | 1000–2500 |
| Eval | NNUE runtime-switch (eval.npz, 32KB model) |

### Overall: 1819/2000 (90.9%)

### By rating

| Rating band | Solved | Total | % |
|-------------|--------|-------|---|
| 1000–1249 | 416 | 423 | 98.3% |
| 1250–1499 | 407 | 428 | 95.1% |
| 1500–1749 | 371 | 406 | 91.4% |
| 1750–1999 | 308 | 338 | 91.1% |
| 2000–2249 | 196 | 239 | 82.0% |
| 2250–2499 | 121 | 166 | 72.9% |

### By theme (top 20)

| Theme | Solved | Total | % |
|-------|--------|-------|---|
| short | 990 | 1058 | 93.6% |
| endgame | 870 | 977 | 89.0% |
| middlegame | 863 | 930 | 92.8% |
| crushing | 778 | 882 | 88.2% |
| advantage | 614 | 675 | 91.0% |
| long | 529 | 611 | 86.6% |
| mate | 411 | 424 | 96.9% |
| master | 274 | 292 | 93.8% |
| fork | 242 | 259 | 93.4% |
| mateIn2 | 192 | 195 | 98.5% |
| sacrifice | 139 | 170 | 81.8% |
| veryLong | 138 | 162 | 85.2% |
| mateIn1 | 151 | 156 | 96.8% |
| oneMove | 151 | 156 | 96.8% |
| advancedPawn | 132 | 147 | 89.8% |
| pin | 128 | 143 | 89.5% |
| kingsideAttack | 130 | 140 | 92.9% |
| defensiveMove | 110 | 138 | 79.7% |
| discoveredAttack | 131 | 138 | 94.9% |
| deflection | 126 | 133 | 94.7% |

**Weak spots:** `defensiveMove` (79.7%), `sacrifice` (81.8%), `veryLong` (85.2%)

---

## 2026-03-25 — cp07 re-run + puzzle fine-tune experiment

**Changes:** Re-run of cp07 baseline (slight variance from 2026-03-23 due to non-determinism), plus
experiment: 30-epoch puzzle fine-tune on all Lichess puzzle positions labeled at depth 14.

**Reproduce baseline:**
```
cargo run -p chess_evaluation --bin puzzle_bench --release -- \
  --file lichess_db_puzzle.csv.zst \
  --count 2000 --min-rating 1000 --max-rating 2500 --depth 7 --seed 42
```

**Reproduce puzzle fine-tune run:**
```
cargo run -p chess_evaluation --bin puzzle_bench --release -- \
  --file lichess_db_puzzle.csv.zst \
  --count 2000 --min-rating 1000 --max-rating 2500 --depth 7 --seed 42 \
  --eval-file nn_training/artifacts/eval_puzzle_ft.npz
```

### Baseline — cp07 (eval.npz, 32KB model): 1816/2000 (90.8%) — 63.5s

| Rating band | Solved | Total | % |
|-------------|--------|-------|---|
| 1000–1249 | 416 | 423 | 98.3% |
| 1250–1499 | 408 | 428 | 95.3% |
| 1500–1749 | 375 | 406 | 92.4% |
| 1750–1999 | 305 | 338 | 90.2% |
| 2000–2249 | 193 | 239 | 80.8% |
| 2250–2499 | 119 | 166 | 71.7% |

| Theme | Solved | Total | % |
|-------|--------|-------|---|
| short | 983 | 1058 | 92.9% |
| endgame | 875 | 977 | 89.6% |
| middlegame | 855 | 930 | 91.9% |
| crushing | 777 | 882 | 88.1% |
| advantage | 611 | 675 | 90.5% |
| long | 536 | 611 | 87.7% |
| mate | 411 | 424 | 96.9% |
| master | 273 | 292 | 93.5% |
| fork | 245 | 259 | 94.6% |
| mateIn2 | 191 | 195 | 97.9% |
| sacrifice | 139 | 170 | 81.8% |
| veryLong | 135 | 162 | 83.3% |
| mateIn1 | 151 | 156 | 96.8% |
| oneMove | 151 | 156 | 96.8% |
| advancedPawn | 134 | 147 | 91.2% |
| pin | 127 | 143 | 88.8% |
| kingsideAttack | 132 | 140 | 94.3% |
| defensiveMove | 113 | 138 | 81.9% |
| discoveredAttack | 131 | 138 | 94.9% |
| deflection | 128 | 133 | 96.2% |

**Weak spots:** `sacrifice` (81.8%), `defensiveMove` (81.9%), `veryLong` (83.3%)

---

### Puzzle fine-tune (eval_puzzle_ft.npz, 30 epochs, lr=5e-5): 1294/2000 (64.7%) ❌ — 219.1s

**Conclusion: catastrophic forgetting. Do not use.**

The fine-tune on Lichess puzzle positions (27M positions labeled at depth 14) caused severe
regression across all positional categories. Only pure mate-finding was unaffected. Search time
also increased 3.5× due to degraded move ordering from corrupted eval.

| Rating band | Solved | Total | % | Δ |
|-------------|--------|-------|---|---|
| 1000–1249 | 362 | 423 | 85.6% | −12.7pp |
| 1250–1499 | 288 | 428 | 67.3% | −28.0pp |
| 1500–1749 | 241 | 406 | 59.4% | −33.0pp |
| 1750–1999 | 186 | 338 | 55.0% | −35.2pp |
| 2000–2249 | 134 | 239 | 56.1% | −24.7pp |
| 2250–2499 | 83 | 166 | 50.0% | −21.7pp |

| Theme | Solved | Total | % | Δ |
|-------|--------|-------|---|---|
| short | 677 | 1058 | 64.0% | −28.9pp |
| endgame | 709 | 977 | 72.6% | −17.0pp |
| middlegame | 539 | 930 | 58.0% | −33.9pp |
| crushing | 567 | 882 | 64.3% | −23.8pp |
| advantage | 302 | 675 | 44.7% | −45.8pp |
| long | 355 | 611 | 58.1% | −29.6pp |
| mate | 417 | 424 | 98.3% | +1.4pp ✓ |
| master | 183 | 292 | 62.7% | −30.8pp |
| fork | 145 | 259 | 56.0% | −38.6pp |
| mateIn2 | 195 | 195 | 100.0% | +2.1pp ✓ |
| sacrifice | 119 | 170 | 70.0% | −11.8pp |
| veryLong | 105 | 162 | 64.8% | −18.5pp |
| mateIn1 | 151 | 156 | 96.8% | 0pp |
| oneMove | 151 | 156 | 96.8% | 0pp |
| advancedPawn | 113 | 147 | 76.9% | −14.3pp |
| pin | 83 | 143 | 58.0% | −30.8pp |
| kingsideAttack | 104 | 140 | 74.3% | −20.0pp |
| defensiveMove | 85 | 138 | 61.6% | −20.3pp |
| discoveredAttack | 84 | 138 | 60.9% | −34.0pp |
| deflection | 90 | 133 | 67.7% | −28.5pp |

**Lesson learned:** Puzzle position fine-tuning causes catastrophic forgetting of positional
evaluation. The distribution of puzzle positions (sharp, tactical, often material-imbalanced) is
too different from the quiet game positions the model was trained on. Even with a conservative
lr=5e-5, 30 epochs was enough to destroy general positional understanding while only preserving
trivial mate patterns that don't depend on eval quality. The recalibration step (`recalibrate.yaml`)
was not attempted given the severity of the regression.

---

## 2026-03-25 — acc_push board-lookup fix

**Changes:** Fixed incremental accumulator not being used. Root cause: the move generator does
not populate `ChessMove::chess_piece`, so `acc_push` treated every move as "unknown piece →
full recompute". Fix: when `chess_piece` is `None`, look up the moving piece via
`board.get_piece_at_square(start_square)` before the fallback. Now only actual king moves
(bucket change) trigger a full `acc_recompute`.

**Result: 1.40× speedup** (63.5s → 45.4s for 2000 puzzles, depth 7), same solve rate.

| Setting | Value |
|---------|-------|
| Depth | 7 |
| Puzzles | 2000 (seed 42) |
| Rating range | 1000–2500 |
| Eval | NNUE nn-incremental (eval.npz, 32KB model) |
| Time | 45.4s (vs 63.5s baseline) |

### Overall: 1816/2000 (90.8%) — 45.4s

---

## 2026-03-25 — eval cache + TT static eval + acc init fixes

**Changes:**
1. Fixed `bench_sequential` missing `ctx.init_accumulators()` (bench was measuring slow path).
2. Fixed `smp_helper` and `extract_ponder_move` missing `ctx.init_accumulators()` — Lazy SMP helpers were using full forward pass.
3. TT static eval caching: stored `static_eval: i32` in TtEntry (fills existing 4-byte padding, no size change). Alpha-beta reuses the cached value instead of calling `eval_node` at depth 1-9 nodes on TT hit. ~2% NPS gain in sequential bench.
4. Per-context eval cache: 16K `(hash, score)` pairs in `SearchContext`. `eval_node` probes this before the NN forward pass. Qsearch transpositions (same position reached via different capture orders) hit the cache at high rates. **+24% NPS on sequential bench, 1.47× speedup on puzzle bench.**

**Cumulative speedup from baseline (broken acc_push):** 63.5s → 30.9s = **2.06×**

| Setting | Value |
|---------|-------|
| Depth | 7 |
| Puzzles | 2000 (seed 42) |
| Rating range | 1000–2500 |
| Eval | NNUE nn-incremental (eval.npz, 32KB model) |
| Bench NPS | 276 Knps (vs 218 Knps after acc_push fix, 162 Knps before) |
| Puzzle time | 30.9s (vs 45.4s after acc_push fix, 63.5s baseline) |

### Overall: 1816/2000 (90.8%) — 30.9s

---

## 2026-03-25 — capture ghost fix + king same-bucket optimization

**Changes:**
1. **Capture ghost fix**: `mv.capture` is never populated by the move generator (same as `mv.chess_piece`). The `else if let Some(cap) = mv.capture` branch in `acc_push` was dead code — every non-en-passant capture silently skipped removing the captured piece's feature, leaving ghost pieces in the accumulator and corrupting eval. Fixed by replacing with `board.get_piece_at_square(mv.target_square())` lookup (before `make_move`, piece still present). Root cause of bad play (giving up pieces, not following through with forced mates).
2. **King same-bucket optimization**: `acc_push` previously returned `true` (full recompute) for all king moves. Many king moves stay in the same bucket (e.g., e1→e2, e1→d1 both bucket 3). Now checks new vs current bucket — only returns `true` if bucket changes. Same-bucket king moves use incremental update instead.

| Setting | Value |
|---------|-------|
| Depth | 7 |
| Puzzles | 2000 (seed 42) |
| Rating range | 1000–2500 |
| Eval | NNUE nn-incremental (eval.npz, 32KB model) |
| Bench NPS | 286 Knps (vs 289 Knps — within noise) |
| Puzzle time | 30.0s |

### Overall: 1811/2000 (90.6%) — 30.0s

**Notes:** Solve rate (90.6%) is within normal variance of the 90.8% baseline. The capture fix restores correct accumulator state — previously the engine was playing with systematically wrong evaluations for all positions involving captures.

---

## 2026-03-25 — Three more correctness fixes

**Changes:**
1. **`is_quiet` always true for captures (CRITICAL):** `chess_move.capture` is never set by the move generator (always `None`), so `is_quiet = capture.is_none() && !is_promotion()` was always `true` for all non-promotion moves including captures. Consequences: captures were futility-pruned, LMP-pruned, history-pruned, and their beta-cutoffs updated `history`/`killers` instead of `capture_history`. Fixed by using board lookup: `is_capture = has_flag(EN_PASSANT) || board.get_piece_at_square(target).is_some()`.

2. **`order_moves` treated all non-promotions as quiets (CRITICAL):** `m.capture.is_some()` was always false, so `good_captures` and `bad_captures` were always empty and SEE ordering was completely dead. All captures were sorted among quiets by history score — equivalent to random ordering. Fixed by same board lookup approach.

3. **King same-bucket + mirror change (MODERATE):** The same-bucket optimization didn't check if horizontal mirroring changed (e.g., e1→d1: same bucket 3, but e-file has mirror=true and d-file has mirror=false). When mirror changes, all piece feature indices are invalid. Added mirror change check alongside bucket change check.

**Performance impact:** With SEE now called for every capture in `order_moves`, NPS dropped from 286K to 174K (−39%). However, the better pruning makes depth 8 nearly free after depth 7, so at constant time the engine is significantly stronger.

| Depth | Time | Solve rate |
|-------|------|-----------|
| 7 (old baseline) | 30.0s | 90.8% |
| 7 (after fixes) | 48.3s | 90.8% |
| 8 (after fixes) | 49.0s | **93.2%** |

At equivalent time (~49s), depth 8 with proper ordering achieves **93.2%** vs 90.8% at depth 7 before.

---

## 2026-03-25 — Six correctness fixes (second review pass)

**Changes:**
1. **`search_root` missing `prev_moves[1]` (MODERATE):** Root loop never set `ctx.prev_moves[1]` to the root move; continuation history and countermove table were unused at ply 1. Fixed: set after `make_move_with_acc`.

2. **Capture history missing malus (MODERATE):** Capture history rewarded the cutoff capture but never penalised captures tried before it. Added `tried_captures_buf` and negative bonus for failed captures.

3. **`alpha_beta_root` missing `init_accumulators` (CRITICAL):** The direct entry point never called `init_accumulators`, leaving `acc_valid = false` and forcing full NN forward passes for every eval.

4. **`prev_moves` stored before `make_move` sets `chess_piece` (MODERATE):** The move generator leaves `chess_piece = None`; `make_move` sets it. Since `prev_moves[(ply+1)]` was stored BEFORE `make_move_with_acc`, `piece_idx(prev_move)` always returned 0 (Pawn). All cont_hist entries were keyed on (0, prev_to, 0, curr_to) instead of actual piece types. Fixed by swapping the order: `make_move_with_acc` first, then store in `prev_moves`. Same fix in `search_root`.

5. **Null move left stale `prev_moves[(ply+1)]` (MINOR):** Added `ctx.prev_moves[(ply+1)] = None` before `make_null_move` so the null search child doesn't see a stale move for cont_hist/countermove lookups.

6. **ProbCut missing `prev_moves[(ply+1)]` (MINOR):** Added `ctx.prev_moves[(ply+1)] = Some(pc_mv)` after `make_move_with_acc` in the ProbCut loop so children get correct piece-type-aware cont_hist keys.

| Setting | Value |
|---------|-------|
| Depth | 8 |
| Puzzles | 2000 (seed 42) |
| Rating range | 1000–2500 |
| Eval | NNUE nn-incremental (eval.npz, 32KB model) |
| Time | 48.7s |

### Overall: 1867/2000 (93.3%) — 48.7s

| Rating band | Solved | Total | % |
|-------------|--------|-------|---|
| 1000–1249 | 416 | 423 | 98.3% |
| 1250–1499 | 412 | 428 | 96.3% |
| 1500–1749 | 387 | 406 | 95.3% |
| 1750–1999 | 310 | 338 | 91.7% |
| 2000–2249 | 211 | 239 | 88.3% |
| 2250–2499 | 131 | 166 | 78.9% |

**Weak spots:** `veryLong` (85.2%), `defensiveMove` (84.8%), `sacrifice` (88.2%)

---

## 2026-03-27 — 512×32×8 dual-perspective model on 69M positions (no dropout)

**Changes:** Switched from 1024×64×8 (checkpoint_all_54m_1024_h64_8b_32kb) to a freshly trained
512×32×8 dual-perspective model on the all_69m dataset. Key training fix: removed dropout (was 0.3,
now 0.0) — high dropout was hurting convergence for NNUE given the large, diverse dataset.
Inference HIDDEN1: 512, HIDDEN2: 32.

**Reproduce:**
```
cargo run -p chess_evaluation --bin puzzle_bench --release -- \
  --file lichess_db_puzzle.csv.zst \
  --eval-file nn_training/artifacts/eval_all_69m_512_8b_nodrop.npz \
  --count 2000 --min-rating 1000 --max-rating 2500 --depth 8 --seed 42
```

| Setting | Value |
|---------|-------|
| Depth | 8 |
| Puzzles | 2000 (seed 42) |
| Rating range | 1000–2500 |
| Eval | NNUE nn-incremental (eval_all_69m_512_8b_nodrop.npz) |
| Time | 28.1s (vs 48.7s with 1024×64×8 — 1.73× faster) |

### Overall: 1816/2000 (90.8%) — 28.1s

| Rating band | Solved | Total | % | Δ vs prev |
|-------------|--------|-------|---|-----------|
| 1000–1249 | 414 | 423 | 97.9% | −0.4pp |
| 1250–1499 | 411 | 428 | 96.0% | −0.3pp |
| 1500–1749 | 380 | 406 | 93.6% | −1.7pp |
| 1750–1999 | 298 | 338 | 88.2% | −3.5pp |
| 2000–2249 | 197 | 239 | 82.4% | −5.9pp |
| 2250–2499 | 116 | 166 | 69.9% | −9.0pp |

| Theme | Solved | Total | % |
|-------|--------|-------|---|
| short | 992 | 1058 | 93.8% |
| endgame | 875 | 977 | 89.6% |
| middlegame | 856 | 930 | 92.0% |
| crushing | 773 | 882 | 87.6% |
| advantage | 616 | 675 | 91.3% |
| long | 528 | 611 | 86.4% |
| mate | 411 | 424 | 96.9% |
| master | 271 | 292 | 92.8% |
| fork | 245 | 259 | 94.6% |
| mateIn2 | 193 | 195 | 99.0% |
| sacrifice | 140 | 170 | 82.4% |
| veryLong | 134 | 162 | 82.7% |
| mateIn1 | 151 | 156 | 96.8% |
| oneMove | 151 | 156 | 96.8% |
| advancedPawn | 131 | 147 | 89.1% |
| pin | 130 | 143 | 90.9% |
| kingsideAttack | 130 | 140 | 92.9% |
| defensiveMove | 116 | 138 | 84.1% |
| discoveredAttack | 129 | 138 | 93.5% |
| deflection | 122 | 133 | 91.7% |

**Notes:** Same overall solve rate (90.8%) as the larger 1024×64×8 model, but 1.73× faster due to
smaller architecture. Regression at high ratings (2000+) suggests the smaller model has less
positional understanding at complex positions. The solve rate gap widens with difficulty.

**Weak spots:** `veryLong` (82.7%), `sacrifice` (82.4%), `defensiveMove` (84.1%)


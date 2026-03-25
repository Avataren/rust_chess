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

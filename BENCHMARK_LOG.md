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

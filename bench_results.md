# Benchmark Results

## Running the benchmark

```bash
# Standard baseline (depth 7, single-threaded, deterministic node count)
cargo run --release -p chess_evaluation --bin bench

# Fixed depth, specific thread count
cargo run --release -p chess_evaluation --bin bench -- --depth 7 --threads 4

# Multi-thread comparison (forces iterative-deepening for fair comparison)
cargo run --release -p chess_evaluation --bin bench -- --depth 7 --threads 1,2,4,8

# Hash + thread sweep — finds optimal TT size and thread count
# Runs depth 12 across all combinations, prints top-5 ranked by NPS
cargo run --release -p chess_evaluation --bin bench -- \
  --depth 12 --hash 64,128,256,512,1024 --threads 1,4,8,16

# Full predefined sweep (depth=12, hash=16–2048 MB, threads=1/4/8/16, 28 configs)
cargo run --release -p chess_evaluation --bin bench -- --hash-sweep
```

**Note:** Use `--depth 10` or higher for hash-size benchmarks — at depth 7 the TT
fills too quickly to show meaningful hit-rate differences across sizes.

---

## chunk0 — baseline (2026-03-13)

```
Position                            Nodes         ms          NPS
------------------------------------------------------------------
After 1.e4                         593551        834       711691
Ruy Lopez setup                   1102633       1475       747547
Italian Game                      4838394       6789       712681
Tactical – Sacrifice              4294504       5172       830337
Open file tension                 1809826       1332      1358728
Complex middle                    1864482       2062       904210
Central tension                   4128973       3721      1109640
K+P endgame                          8604          2      4302000
Pawn race                            9148          1      9148000
KR vs Kr                           223324         53      4213660
Active rook                         87748         28      3133857
Q vs passers                       116281         25      4651240
------------------------------------------------------------------
TOTAL / AVG NPS                  19077468      21494       887571
```

total_nodes=19,077,468  avg_nps=887,571

---

## chunk1 — Futility + RFP + Aggressive LMR (2026-03-13)

```
Position                            Nodes         ms          NPS
------------------------------------------------------------------
After 1.e4                         376875        567       664682
Ruy Lopez setup                    527910        785       672496
Italian Game                      1839524       2841       647491
Tactical – Sacrifice              1009317       1276       791000
Open file tension                  687700        598      1150000
Complex middle                     610087        818       745827
Central tension                   1154929       1198       964047
K+P endgame                          5134          2      2567000
Pawn race                            6674          1      6674000
KR vs Kr                           146840         40      3671000
Active rook                         40362         16      2522625
Q vs passers                        20837          6      3472833
------------------------------------------------------------------
TOTAL / AVG NPS                   6426189       8148       788682
```

total_nodes=6,426,189 (-66%)  avg_nps=788,682  total_ms=8,148 (-62% vs chunk0)
Changes: futility pruning (depth 1: ±200cp, depth 2: ±400cp), RFP (depth≤6, 100*depth margin),
         LMR formula R=floor(ln(depth)*ln(move_index+1)/1.5) instead of flat R=1.
Self-play vs chunk0: **66.2%** (chunk1 win rate, 40 games, 200ms/move)

---

## chunk2 — Mobility + Rook-on-7th + Knight Outposts (2026-03-13)

```
Position                            Nodes         ms          NPS
------------------------------------------------------------------
After 1.e4                         401715        500       803430
Ruy Lopez setup                    533053        641       831595
Italian Game                      1844354       2316       796353
Tactical – Sacrifice               954718        917      1041131
Open file tension                  756707        588      1286916
Complex middle                     724386        792       914628
Central tension                   1143825        951      1202760
K+P endgame                          5134          1      5134000
Pawn race                            6674          1      6674000
KR vs Kr                           150301         44      3415931
Active rook                         49429         18      2746055
Q vs passers                        20837          5      4167400
------------------------------------------------------------------
TOTAL / AVG NPS                   6591133       6774       973004
```

total_nodes=6,591,133 (+2.6% vs chunk1)  avg_nps=973,004 (+23% vs chunk1)
Self-play vs chunk1: **60.0%** (20W/8D/12L, 40 games, 200ms/move)
Changes: mobility (knight×4, bishop×3, rook×1 cp/sq; excludes pawn-attacked squares),
         rook-on-7th (+25/+35 MG/EG when enemy king on back rank),
         knight outposts (+30/+20 MG/EG on protected advanced squares).

---

## chunk3 — Delta pruning in qsearch (2026-03-13)

```
Position                            Nodes         ms          NPS
------------------------------------------------------------------
After 1.e4                         304513        421       723308
Ruy Lopez setup                    393838        551       714769
Italian Game                      1360745       2046       665075
Tactical – Sacrifice               747238        909       822044
Open file tension                  546157        491      1112336
Complex middle                     486612        668       728461
Central tension                    772041        810       953137
K+P endgame                          6359          1      6359000
Pawn race                            6078          1      6078000
KR vs Kr                           159929         47      3402744
Active rook                         42970         17      2527647
Q vs passers                        19859          5      3971800
------------------------------------------------------------------
TOTAL / AVG NPS                   4846339       5967       812190
```

total_nodes=4,846,339 (-26% vs chunk2)  avg_nps=812,190 (-17% vs chunk2)  total_ms=5,967 (-12% vs chunk2)
Self-play vs chunk2: **60.0%** (20W/8D/12L, 40 games, 200ms/move)
Changes:
- Delta pruning in qsearch: skip captures where stand_pat + piece_value + 200cp <= alpha (white)
  or stand_pat - piece_value - 200cp >= beta (black).
- TT in qsearch was tested but REMOVED: caused 70% NPS regression due to cache thrashing
  at the millions-of-nodes/s qsearch rate. Delta pruning kept; TT discarded.

---

## chunk4 — RFP/futility threshold tuning (2026-03-13)

total_nodes=5,304,764 (+9% vs chunk3)  avg_nps=832,250  total_ms=6,374
Self-play vs chunk3 (no ponder): **51.2%** (17W/7D/16L, 40 games, 200ms/move) — neutral
Changes: RFP margin 100→75*depth, RFP depth ≤6→≤9, futility extended to depth ≤3 (200*depth formula).

---

## chunk5 — SEE move ordering + qsearch pruning (2026-03-13)

```
Position                            Nodes         ms          NPS
------------------------------------------------------------------
After 1.e4                         269607        461       584830
Ruy Lopez setup                    269959        434       622025
Italian Game                       693956       1238       560546
Tactical – Sacrifice               581039        921       630878
Open file tension                  351800        387       909043
Complex middle                     728574       1315       554048
Central tension                    530400        653       812251
K+P endgame                          5122          2      2561000
Pawn race                            5480          1      5480000
KR vs Kr                           162859         54      3015907
Active rook                         31258         14      2232714
Q vs passers                        18046          6      3007666
------------------------------------------------------------------
TOTAL / AVG NPS                   3648100       5486       664983
```

total_nodes=3,648,100 (-31% vs chunk4)  avg_nps=664,983 (-20% vs chunk4)  total_ms=5,486 (-14% vs chunk4)
Self-play vs chunk4 (no ponder): **61.3%** (21W/7D/12L, 40 games, 200ms/move)
Changes:
- SEE (Static Exchange Evaluation) in see.rs: full exchange simulation using LVA loop + X-ray.
- order_moves: captures sorted by SEE score (winning/even first, losing last) instead of MVV only.
- qsearch: SEE < 0 pruning replaces delta pruning (skips clearly losing captures).
- 9 SEE unit tests covering: undefended pieces, defended pieces, equal exchanges, X-ray, symmetry.

---

## chunk6 — Singular Extensions + Countermove + IID + Pawn Hash + LMP + Capture History (2026-03-15)

```
Position                            Nodes         ms          NPS
------------------------------------------------------------------
After 1.e4                          88083         93       947129
Ruy Lopez setup                     92396        101       914811
Italian Game                       229825        257       894260
Tactical – Sacrifice               195841        188      1041707
Open file tension                  102737         90      1141522
Complex middle                     128577        134       959529
Central tension                    132116        118      1119627
K+P endgame                          5435          1      5435000
Pawn race                            5584          1      5584000
KR vs Kr                            87509         33      2651787
Active rook                         20539          8      2567375
Q vs passers                         8572          3      2857333
------------------------------------------------------------------
TOTAL / AVG NPS                   1097214       1027      1068368
```

total_nodes=1,097,214 (-70% vs chunk5)  avg_nps=1,068,368 (+60% vs chunk5)  total_ms=1,027 (-81% vs chunk5)
Changes (cumulative since chunk5):
- LMP (Late Move Pruning): skip remaining quiets after threshold at depth 1–4.
- Capture history: separate bonus/malus table for captures, used as SEE tiebreaker.
- Countermove heuristic: quiet move that refuted opponent's last move; ordered after killers.
- IID (Internal Iterative Deepening): reduced-depth search at depth≥5 when TT has no move.
- Pawn hash table: 16k-entry per-thread cache for pawn structure evaluation.
- **Singular Extensions**: at depth≥6, if TT move's exclusion search (depth/2) fails below
  tt_score−50cp, the TT move is singular and extended by one ply.
  Implementation: `excluded_move[ply]` in SearchContext; TT cutoffs guarded when excluded.

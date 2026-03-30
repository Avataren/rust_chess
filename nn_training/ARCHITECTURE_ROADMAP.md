# NNUE Architecture Improvement Roadmap

## Current State (as of 2026-03-30)

| Component | Current |
|---|---|
| Feature encoding | HalfKP dual-perspective, 12×64×32 = **24,576** features, 32 king buckets |
| Layer 1 (accumulator) | **512** per perspective, 1024 combined |
| Layer 2 (bottleneck) | 1024 → **32** |
| Output buckets | **8** (by piece count) |
| Activation | SCReLU throughout |
| Quantization | i16 weights, f32 scale, saturating arithmetic |
| SIMD | AVX2 (x86_64), simd128 (wasm32), scalar fallback |
| Anchor training data | ~69M positions, Stockfish depth 14 |
| Self-play loop | Active — FIFO pool 750k, 5-epoch fine-tune per iteration |

### Search features implemented
- Iterative deepening + aspiration windows
- Lazy SMP (multi-threaded)
- LMR with precomputed table, LMP, NMP, ProbCut
- Singular extension + double extension, IIR
- Killer moves, history, capture history, continuation history, countermove
- Incremental i16 accumulators (AVX2/wasm SIMD)
- Qsearch eval cache (avoids redundant NN passes on repeated positions)
- Syzygy tablebase probing (DTZ, up to 7 pieces)
- Opening book (131 theory lines, Zobrist hash map)

---

## Impact vs Effort Matrix

| # | Improvement | ELO Estimate | Effort | Data Regen? |
|---|---|---|---|---|
| A1 | hidden2_dim 32→64 | +10–20 | Low | No |
| A2 | WDL game-result blending | +10–20 | Medium | Partial |
| A3 | Deeper Stockfish labels (depth 14→16 for new data) | +5–15 | Low | Yes (new only) |
| A4 | Accumulator refresh cache (skip stale acc at root) | +5–10 NPS | Medium | No (Rust only) |
| A5 | Lazy eval in qsearch (HCE for non-leaf qnodes) | +10–20 NPS | Medium | No (Rust only) |
| A6 | More self-play data per iteration | +? | Low | No |
| A7 | HalfKAv2 features (45,056 input dim) | +30–60 | High | Full regen |
| A8 | Wider Layer 1: 512→768 or 1024 per perspective | +15–30 | High | Full retrain |
| A9 | 200M+ positions (longer Lichess history) | +20–40 | High | Yes |
| A10 | Teacher→student distillation | +20–40 (student) | High | Partial |

---

## A1 — Widen hidden2_dim 32→64

The 1024→32 compression is aggressive. 32 outputs is the bottleneck for both the CP and WDL heads across 8 output buckets.

**Rust**: `chess_evaluation/src/neural_eval.rs` — change `const HIDDEN2: usize = 32` → `64`.
Update the `gemv_col32_avx2` inner kernel or add a `gemv_col64` variant.

**Python**: `finetune.yaml` and any full-training configs — `hidden2_dim: 64`.

No data regeneration needed. Checkpoint is incompatible (different weight shapes) — full retrain from the 69M anchor or fine-tune from scratch on the pool.

---

## A2 — WDL Game-Result Blending

The current WDL target is derived entirely from Stockfish CP via `cp_to_wdl_target()`. Using the actual game outcome (win/draw/loss from the PGN `Result` header) blended in at ~30% weight is a known Stockfish technique that improves calibration.

**Minimum viable** (no data regen): add `wdl_lambda` parameter to training config (1.0 = current, 0.7 = blend). Infrastructure change only — actual blending requires result labels in the dataset.

**Full version** — `generate_data.py`: yield `(fen, cp, result)` where result = {0, 0.5, 1} from PGN header. Store as `{"fen": "...", "cp": N, "result": 0.5}`. Dataset loader reads `result` field if present, falls back to CP-only WDL if absent (backward compatible). Self-play positions get `result=None` since game outcomes aren't known at generation time.

---

## A3 — Deeper Stockfish Labels

Current default in the self-play loop is depth 12. The anchor data was labeled at depth 14. Increasing new self-play labels to depth 14–16 improves label quality at the cost of labeling time. Worth doing if CPU is available during the labeling phase.

In `start_selfplay_loop.sh`: add `--eval-depth 14` (or 16).

---

## A4 — Accumulator Refresh Cache

Currently accumulators are rebuilt from scratch at the root of each search and updated incrementally through the tree. Positions seen at depth 0 across multiple iterative deepening passes recompute the same root accumulator each time.

A small hash-indexed cache (e.g. 256 entries, keyed by board Zobrist hash) can return pre-computed accumulators for recently seen root positions. Negligible memory cost, ~2–5% NPS improvement at shallow depths.

Implementation: `chess_evaluation/src/alpha_beta/search_context.rs` — add `acc_cache: HashMap<u64, ([i16; 512], [i16; 512])>` (or fixed-size ring buffer to avoid allocation).

---

## A5 — Lazy Eval in Qsearch

The NN is currently called at every qsearch leaf. For non-leaf qsearch nodes (where more captures remain), a cheap HCE static eval could serve as stand-pat without calling the NN.

Only applies to `runtime-switch` builds; `nn-incremental` always uses NN. Would need a hybrid where the stand-pat is HCE and the NN is called only when the position stabilises.

Complex to implement correctly (HCE and NN score scales differ). Lower priority than A1/A2.

---

## A7 — HalfKAv2 Features

Replace HalfKP (12×64×32 = 24,576) with HalfKAv2 (10×64×64 = 40,960, or the standard 45,056 with virtual king square). This is the feature set used by modern Stockfish variants.

Benefit: the feature set explicitly encodes the attacking king's position, giving much richer pawn structure and king safety representation.

Cost: full architecture change (Rust encoder + Python dataset + new training run), 2–3× larger weight matrix for Layer 1. Requires regenerating all binary `.npy` preprocessed data.

---

## Self-Play Loop Limitations (known)

- **150 games at 100ms, SE≈4.1%**: promotion gate is a regression guard, not a confirmation of improvement. A genuine +10 ELo improvement has <50% chance of clearing the gate in one evaluation.
- **Puzzle gate blocks on exact score**: 0% regression tolerance is intentional (benchmark is deterministic) but means one tactical hole can block an otherwise strong model.
- **Gen-val is 10k positions from the 69M anchor**: a stable generalization signal, but the 69M anchor has its own distribution biases. Ground truth is the Lichess bot rating.
- **No game-outcome signal**: self-play pool positions don't carry win/draw/loss labels. WDL targets are Stockfish-derived only.

#!/usr/bin/env python3
"""Download and convert Lichess/chess-position-evaluations to binary training format.

Processes one Parquet shard at a time to stay within disk budget:
  - Downloads shard (~2.3 GB)
  - Filters by depth, converts FEN → HalfKP binary
  - Deletes the Parquet shard
  - Repeats

Disk peak per shard: ~2.3 GB parquet + ~2.7 GB binary shard output.
Total output for 17 shards at 20M cap: ~340M positions ≈ 46 GB binary.

Usage:
    cd nn_training
    PYTHONPATH=. python3 scripts/download_lichess_hf.py \
        --output data/normalized_hf \
        --min-depth 0 \
        --max-per-shard 15000000 \
        --cp-clamp 3000 \
        --workers 8

Merge is skipped by default (not needed for sharded_gpu training).
Pass --merge to produce a single merged dataset (~34 GB RAM required).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import chess
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

# Add project root to path for nnue_train imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nnue_train.features import encode_board_halfkav2_dual, HALFKAV2_FEATURE_DIM

REPO_ID   = "mateuszgrzyb/lichess-stockfish-normalized"
NUM_SHARDS = 10
MATE_SCORE = 10000   # CP value assigned to mate positions before clamping
SENTINEL   = HALFKAV2_FEATURE_DIM  # 45056 — padding value for unused index slots


def _encode_position(fen: str, cp_stm: float, cp_clamp: float):
    """Encode one FEN to dual HalfKP indices + white-absolute CP.

    Returns (white_idx, black_idx, count, cp_white, piece_count) or None on error.
    white_idx / black_idx are int32 arrays of length 32 (padded with SENTINEL=45056).
    Intermediate buffers are uint16 (values 0–45055 fit; max uint16 is 65535) then
    widened to int32 before return — saved files are always int32, never int16.
    cp_white is white-absolute, clamped to ±cp_clamp.
    """
    try:
        board = chess.Board(fen)
    except Exception:
        return None

    w_raw, b_raw = encode_board_halfkav2_dual(board)
    n_pieces = bin(int(board.occupied)).count('1')

    # Pad to exactly 32 slots
    w = np.full(32, SENTINEL, dtype=np.uint16)
    b = np.full(32, SENTINEL, dtype=np.uint16)
    wl = min(len(w_raw), 32);  w[:wl] = w_raw[:wl]
    bl = min(len(b_raw), 32);  b[:bl] = b_raw[:bl]

    # Dataset is already white-absolute (mateuszgrzyb/lichess-stockfish-normalized).
    # No sign flip needed — positive CP always means white is winning.
    cp_white = float(cp_stm)
    cp_white = float(np.clip(cp_white, -cp_clamp, cp_clamp))

    return w.astype(np.int32), b.astype(np.int32), n_pieces, cp_white, n_pieces


_CP_CLAMP_GLOBAL: float = 3000.0  # set before spawning workers


def _encode_row(args: tuple) -> tuple | None:
    """Top-level worker function (picklable). args = (fen, cp_stm)."""
    fen, cp_stm = args
    return _encode_position(fen, cp_stm, _CP_CLAMP_GLOBAL)


def _piece_count_from_fen(fen: str) -> int:
    """Count pieces on the board from a FEN string (fast, no chess.Board needed)."""
    board_part = fen.split()[0]
    return sum(1 for c in board_part if c.isalpha())


def process_shard(
    shard_idx: int,
    parquet_path: Path,
    out_dir: Path,
    min_depth: int,
    max_positions: int,
    cp_clamp: float,
    workers: int,
    max_pieces: int = 32,
    min_pieces: int = 2,
) -> int:
    """Convert one Parquet shard to binary .npy files. Returns number of positions written.

    max_pieces: skip positions with more than this many pieces on the board.
      Set to 24 to exclude opening positions (≥25 pieces) and focus on
      middlegame/endgame positions where tactical patterns occur.
      Default 32 keeps all positions (original behaviour).

    min_pieces: skip positions with fewer than this many pieces.
      Default 2 keeps all non-empty positions.
    """
    import multiprocessing as mp

    global _CP_CLAMP_GLOBAL
    _CP_CLAMP_GLOBAL = cp_clamp

    filter_desc = f"depth>={min_depth}"
    if max_pieces < 32:
        filter_desc += f", pieces<={max_pieces}"

    print(f"  Reading {parquet_path.name} (chunked, {filter_desc}) ...", flush=True)
    pfile = pq.ParquetFile(parquet_path)

    rows: list[tuple[str, float]] = []
    n_depth_skip = 0
    n_pieces_skip = 0
    for batch in pfile.iter_batches(
        batch_size=500_000,
        columns=["fen", "cp", "mate", "depth"],
    ):
        fens   = batch["fen"].to_pylist()
        cps    = batch["cp"].to_pylist()
        mates  = batch["mate"].to_pylist()
        depths = batch["depth"].to_pylist()
        for i in range(len(fens)):
            d = depths[i]
            if d is None or int(d) < min_depth:
                n_depth_skip += 1
                continue
            # Piece count filter: skip opening positions early (cheap string scan,
            # avoids constructing a chess.Board for positions we'll discard anyway).
            if max_pieces < 32 or min_pieces > 2:
                pc = _piece_count_from_fen(fens[i])
                if pc > max_pieces or pc < min_pieces:
                    n_pieces_skip += 1
                    continue
            mate_val = mates[i]
            cp_val   = cps[i]
            if mate_val is not None:
                cp_stm = MATE_SCORE if int(mate_val) > 0 else -MATE_SCORE
            elif cp_val is not None:
                cp_stm = float(cp_val)
            else:
                continue
            rows.append((fens[i], cp_stm))
            if max_positions and len(rows) >= max_positions:
                break
        if max_positions and len(rows) >= max_positions:
            break

    print(f"  {len(rows):,} positions after {filter_desc} filter", flush=True)
    if n_pieces_skip:
        print(f"  ({n_pieces_skip:,} skipped by piece count filter)", flush=True)

    if not rows:
        return 0

    # Encode in sub-batches of 1M to cap peak RAM usage.
    # Processing all 15M at once with pool.map materialises ~8 GB of result objects.
    # Processing 1M at a time keeps peak overhead to ~800 MB per sub-batch.
    SUB_BATCH = 1_000_000
    white_parts, black_parts, counts_parts, cp_parts, pc_parts = [], [], [], [], []

    print(f"  Encoding {len(rows):,} positions ({workers} workers, {SUB_BATCH//1_000_000}M sub-batches)...", flush=True)
    with mp.Pool(workers) as pool:
        for start in range(0, len(rows), SUB_BATCH):
            chunk = rows[start:start + SUB_BATCH]
            results = [r for r in pool.map(_encode_row, chunk, chunksize=2048) if r is not None]
            if not results:
                continue
            white_parts.append(np.array([r[0] for r in results], dtype=np.int32))
            black_parts.append(np.array([r[1] for r in results], dtype=np.int32))
            counts_parts.append(np.array([r[2] for r in results], dtype=np.uint8))
            cp_parts.append(np.array([r[3] for r in results], dtype=np.float32))
            pc_parts.append(np.array([r[4] for r in results], dtype=np.uint8))
            done = min(start + SUB_BATCH, len(rows))
            print(f"    {done:,}/{len(rows):,} encoded", flush=True)
            del results, chunk

    if not white_parts:
        return 0

    white_idx = np.concatenate(white_parts); del white_parts
    black_idx = np.concatenate(black_parts); del black_parts
    counts    = np.concatenate(counts_parts); del counts_parts
    cp_arr    = np.concatenate(cp_parts);    del cp_parts
    pc_arr    = np.concatenate(pc_parts);    del pc_parts
    N = len(cp_arr)
    print(f"  {N:,} positions encoded successfully", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / f"shard_{shard_idx:02d}"
    np.save(str(prefix) + ".white_indices.npy", white_idx)
    np.save(str(prefix) + ".black_indices.npy", black_idx)
    np.save(str(prefix) + ".counts.npy",        counts)
    np.save(str(prefix) + ".cp.npy",            cp_arr)
    np.save(str(prefix) + ".piece_count.npy",   pc_arr)

    size_gb = sum(
        (str(prefix) + s + ".npy").__len__()  # just filenames
        for s in [".white_indices", ".black_indices", ".counts", ".cp", ".piece_count"]
    )
    total_bytes = (white_idx.nbytes + black_idx.nbytes + counts.nbytes +
                   cp_arr.nbytes + pc_arr.nbytes)
    print(f"  Saved to {prefix}.*.npy  ({total_bytes/1e9:.2f} GB)", flush=True)
    return N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output",        default="data/normalized_hf",
                    help="Output directory for shards + merged dataset")
    ap.add_argument("--min-depth",     type=int,   default=0,
                    help="Minimum Stockfish depth to include (default 0 = keep all)")
    ap.add_argument("--max-per-shard", type=int,   default=0,
                    help="Max positions to take per shard (default 0 = no limit)")
    ap.add_argument("--cp-clamp",      type=float, default=10000.0,
                    help="Clamp CP to ±N (default 10000; preserves mate scores as ±10000 "
                         "for max tactical signal — old default was 3000 which discarded mates)")
    ap.add_argument("--max-pieces",    type=int,   default=32,
                    help="Skip positions with more than N pieces (default 32 = keep all). "
                         "Use --max-pieces 24 to exclude opening positions (≥25 pieces) "
                         "and focus on middlegame/endgame where tactical patterns occur.")
    ap.add_argument("--min-pieces",    type=int,   default=2,
                    help="Skip positions with fewer than N pieces (default 2 = keep all).")
    ap.add_argument("--workers",       type=int,   default=max(1, os.cpu_count() - 2),
                    help="Encoding worker processes")
    ap.add_argument("--shards",        default="0-9",
                    help="Shard range to process, e.g. '0-16' or '0,1,2'")
    ap.add_argument("--keep-parquet",  action="store_true",
                    help="Don't delete Parquet files after conversion")
    ap.add_argument("--merge",         action="store_true",
                    help="Merge shards into a single dataset after encoding "
                         "(requires ~34 GB RAM for 17×15M shards; not needed "
                         "for sharded_gpu training which reads shards directly)")
    args = ap.parse_args()

    # Parse shard indices
    if "-" in args.shards:
        lo, hi = args.shards.split("-")
        shard_indices = list(range(int(lo), int(hi) + 1))
    else:
        shard_indices = [int(s) for s in args.shards.split(",")]

    out_root   = Path(args.output)
    shards_dir = out_root / "shards"
    parquet_dir = out_root / "parquet"
    parquet_dir.mkdir(parents=True, exist_ok=True)

    total_positions = 0

    for idx in shard_indices:
        filename = f"train-{idx:05d}.parquet"
        shard_out = shards_dir / f"shard_{idx:02d}"

        # Skip if already converted
        done_marker = shards_dir / f"shard_{idx:02d}.cp.npy"
        if done_marker.exists():
            n = len(np.load(str(done_marker), mmap_mode="r"))
            print(f"[shard {idx:02d}] already done ({n:,} positions), skipping", flush=True)
            total_positions += n
            continue

        print(f"\n[shard {idx:02d}/{NUM_SHARDS-1}] Downloading {filename} ...", flush=True)
        parquet_path = parquet_dir / f"shard_{idx:02d}.parquet"

        if not parquet_path.exists():
            # hf_hub_download saves to local_dir/filename (preserving subdir structure)
            hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                repo_type="dataset",
                local_dir=str(parquet_dir),
                local_dir_use_symlinks=False,
            )
            downloaded = parquet_dir / filename  # e.g. parquet_dir/train-00000.parquet
            if downloaded.exists():
                downloaded.rename(parquet_path)
            else:
                raise FileNotFoundError(f"Expected download at {downloaded}")

        n = process_shard(
            shard_idx=idx,
            parquet_path=parquet_path,
            out_dir=shards_dir,
            min_depth=args.min_depth,
            max_positions=args.max_per_shard,
            cp_clamp=args.cp_clamp,
            workers=args.workers,
            max_pieces=args.max_pieces,
            min_pieces=args.min_pieces,
        )
        total_positions += n

        if not args.keep_parquet:
            parquet_path.unlink(missing_ok=True)
            # Also clean up the data/ subdir if empty
            data_subdir = parquet_dir / "data"
            if data_subdir.exists():
                try:
                    data_subdir.rmdir()
                except OSError:
                    pass
            print(f"  Deleted {parquet_path.name}", flush=True)

        print(f"[shard {idx:02d}] done. Running total: {total_positions:,}", flush=True)

    print(f"\n{'='*60}")
    print(f"All shards done. Total positions: {total_positions:,}")

    if not args.merge:
        print("Skipping merge (use --merge to produce a single merged dataset).")
        return

    # Collect shard prefixes that were actually written
    shard_prefixes = sorted(
        p.parent / p.name.replace(".cp.npy", "")
        for p in shards_dir.glob("shard_??.cp.npy")
    )

    if not shard_prefixes:
        print("No shard outputs found — nothing to merge.")
        return

    # Merge shards into a single dataset
    print(f"\nMerging {len(shard_prefixes)} shards → {out_root}/merged ...", flush=True)
    sys.path.insert(0, str(Path(__file__).parent))
    from merge_npy_shards import merge_shards
    merge_shards(
        prefixes=shard_prefixes,
        output_prefix=out_root / "merged",
    )
    print(f"Merged dataset at {out_root}/merged.*.npy")
    print(f"Total positions: {total_positions:,}")


if __name__ == "__main__":
    main()

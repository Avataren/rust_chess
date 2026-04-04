from __future__ import annotations

import json
from pathlib import Path

import chess
import numpy as np
import torch
from torch.utils.data import Dataset

from .features import (
    cp_to_wdl_target, cp_to_wdl_batch,
    encode_board_12x64, encode_board_halfkp, encode_board_halfkp_dual,
    encode_board_halfkav2_dual,
    HALFKP_FEATURE_DIM, HALFKAV2_FEATURE_DIM, FEATURE_DIM,
)


def _ply_from_fen(fen: str) -> int:
    """Extract ply from FEN fullmove counter. Falls back to 40 (middlegame)."""
    parts = fen.split()
    try:
        fullmove = int(parts[5])
        ply = (fullmove - 1) * 2 + (1 if parts[1] == "b" else 0)
        return min(max(ply, 0), 240)
    except (IndexError, ValueError):
        return 40


def _ply_from_piece_count(piece_count: int) -> int:
    """Approximate ply from piece count for binary datasets that lack FEN.

    32 pieces → ply ≈ 0 (opening)
    24 pieces → ply ≈ 32 (early middlegame)
    16 pieces → ply ≈ 64 (middlegame/endgame transition)
     8 pieces → ply ≈ 96 (endgame)
    """
    return min(240, max(0, (32 - int(piece_count)) * 4))


class BinaryPositionDataset(Dataset):
    """Fast dataset backed by pre-encoded sparse binary files.

    Use scripts/preprocess_dataset.py to generate the binary files from JSONL.
    Each __getitem__ is a memmap read + sparse scatter — no JSON parsing or
    python-chess encoding at training time, giving ~10-20x DataLoader speedup.

    Files expected (given path = "data/train_10m.jsonl"):
      data/train_10m.indices.npy      -- (N, 32) uint16
      data/train_10m.counts.npy       -- (N,)    uint8
      data/train_10m.cp.npy           -- (N,)    float32
      data/train_10m.piece_count.npy  -- (N,)    uint8   (optional, for output buckets)
    """

    def __init__(self, path: str, max_cp_abs: int = 1500, use_halfkp: bool = True):
        prefix = str(Path(path).with_suffix(""))
        self.feature_dim = HALFKP_FEATURE_DIM if use_halfkp else FEATURE_DIM

        self.indices = np.load(prefix + ".indices.npy", mmap_mode="r")
        self.counts  = np.load(prefix + ".counts.npy",  mmap_mode="r")
        self.cp_raw  = np.load(prefix + ".cp.npy",      mmap_mode="r")
        self.cp      = np.clip(self.cp_raw, -max_cp_abs, max_cp_abs)
        pc_path = prefix + ".piece_count.npy"
        if Path(pc_path).exists():
            self.piece_count = np.load(pc_path, mmap_mode="r")
        else:
            # Legacy datasets without piece_count: use active feature count as proxy
            self.piece_count = self.counts

    def __len__(self) -> int:
        return len(self.cp)

    def __getitem__(self, idx: int):
        count = int(self.counts[idx])

        # Return sparse indices padded with feature_dim (out-of-range sentinel).
        # The training loop scatters these into a dense tensor on GPU — 800x less
        # PCIe traffic than sending the full 12,288-dim dense float32 vector.
        indices = np.full(32, self.feature_dim, dtype=np.int64)
        indices[:count] = self.indices[idx, :count]

        cp_val = float(self.cp[idx])
        cp = np.array([cp_val], dtype=np.float32)
        wdl = cp_to_wdl_target(float(self.cp_raw[idx]),
                               ply=_ply_from_piece_count(self.piece_count[idx]))
        pc = np.array([int(self.piece_count[idx])], dtype=np.int64)

        return (
            torch.from_numpy(indices),
            torch.from_numpy(pc),
            torch.from_numpy(cp),
            torch.from_numpy(wdl),
        )


class BinaryDualPositionDataset(Dataset):
    """Fast dual-perspective dataset backed by pre-encoded sparse binary files.

    Use scripts/preprocess_dataset.py --dual to generate the binary files.

    Files expected (given path = "data/train_10m.jsonl"):
      data/train_10m.white_indices.npy  -- (N, 32) uint16
      data/train_10m.black_indices.npy  -- (N, 32) uint16
      data/train_10m.counts.npy         -- (N,)    uint8
      data/train_10m.cp.npy             -- (N,)    float32  (white-absolute cp)
      data/train_10m.piece_count.npy    -- (N,)    uint8    (total pieces on board)
    """

    def __init__(self, path: str, max_cp_abs: int = 1500, max_positions: int | None = None):
        prefix = str(Path(path).with_suffix(""))
        self.white_indices = np.load(prefix + ".white_indices.npy", mmap_mode="r")
        self.black_indices = np.load(prefix + ".black_indices.npy", mmap_mode="r")
        self.counts = np.load(prefix + ".counts.npy", mmap_mode="r")
        # cp_raw: unclipped values (up to ±1500) used for WDL target computation
        # so the WDL head always sees a full win/draw/loss distribution.
        # cp: clipped to max_cp_abs for the cp regression loss, focusing
        # learning on near-equal positions where evaluation quality matters.
        self.cp_raw = np.load(prefix + ".cp.npy", mmap_mode="r")
        self.cp = np.clip(self.cp_raw, -max_cp_abs, max_cp_abs)
        pc_path = prefix + ".piece_count.npy"
        if Path(pc_path).exists():
            self.piece_count = np.load(pc_path, mmap_mode="r")
        else:
            # Legacy datasets without piece_count: estimate from active feature count
            self.piece_count = self.counts
        if max_positions is not None:
            self.white_indices = self.white_indices[:max_positions]
            self.black_indices = self.black_indices[:max_positions]
            self.counts        = self.counts[:max_positions]
            self.cp_raw        = self.cp_raw[:max_positions]
            self.cp            = self.cp[:max_positions]
            self.piece_count   = self.piece_count[:max_positions]

    def __len__(self) -> int:
        return len(self.cp)

    def __getitem__(self, idx: int):
        count = int(self.counts[idx])
        SENTINEL = HALFKAV2_FEATURE_DIM  # 45056 — must match model's padding_idx

        w_idx = np.full(32, SENTINEL, dtype=np.int64)
        b_idx = np.full(32, SENTINEL, dtype=np.int64)
        w_idx[:count] = self.white_indices[idx, :count]
        b_idx[:count] = self.black_indices[idx, :count]

        cp_val = float(self.cp[idx])          # clipped — for cp regression loss
        cp_raw = float(self.cp_raw[idx])      # unclipped — for WDL target
        cp = np.array([cp_val], dtype=np.float32)
        wdl = cp_to_wdl_target(cp_raw, ply=_ply_from_piece_count(self.piece_count[idx]))
        pc = np.array([int(self.piece_count[idx])], dtype=np.int64)

        return (
            torch.from_numpy(w_idx),
            torch.from_numpy(b_idx),
            torch.from_numpy(pc),
            torch.from_numpy(cp),
            torch.from_numpy(wdl),
        )


class GPUPreloadedDualDataset:
    """Loads the entire dual-perspective dataset into GPU VRAM at startup.

    Eliminates DataLoader worker overhead by serving shuffled batches directly
    from GPU tensors with zero CPU→GPU transfer per batch. Best used when GPU
    utilization is low and VRAM is available (model is too small to saturate GPU).

    Memory: ~200 bytes/position  (indices×2 as int32, cp/wdl/pc as float32/int64).
    Example: 77M positions ≈ 15.4 GB VRAM.

    Drop-in replacement for DataLoader: supports len() and __iter__,
    yielding (w_idx, b_idx, piece_count, cp, wdl) tuples with all tensors on GPU.
    """

    def __init__(
        self,
        path: "str | list[str] | list[tuple[str, int | None]]",
        max_cp_abs: int,
        device: torch.device,
        batch_size: int,
        shuffle: bool = True,
        feature_dim: int = HALFKP_FEATURE_DIM,
    ):
        # Normalise path to list of (path_str, max_n_or_None) tuples.
        if isinstance(path, str):
            path_specs = [(path, None)]
        else:
            path_specs = []
            for entry in path:
                if isinstance(entry, str):
                    path_specs.append((entry, None))
                else:
                    path_specs.append(tuple(entry))  # (path, max_n)

        SENTINEL = feature_dim  # padding value; must match model's padding_idx

        all_white, all_black, all_counts, all_cp_raw, all_pc = [], [], [], [], []
        for p, max_n in path_specs:
            prefix = str(Path(p).with_suffix(""))
            print(f"  GPU preload: {Path(p).name}" +
                  (f"  [:{max_n:,}]" if max_n else "") + f" → {device}")

            w_np  = np.load(prefix + ".white_indices.npy", mmap_mode="r")
            b_np  = np.load(prefix + ".black_indices.npy", mmap_mode="r")
            c_np  = np.load(prefix + ".counts.npy",        mmap_mode="r")
            cp_np = np.load(prefix + ".cp.npy",            mmap_mode="r")
            pc_path = prefix + ".piece_count.npy"
            pc_np = np.load(pc_path, mmap_mode="r") if Path(pc_path).exists() else c_np

            if max_n is not None:
                w_np  = w_np[:max_n];  b_np = b_np[:max_n]
                c_np  = c_np[:max_n];  cp_np = cp_np[:max_n]; pc_np = pc_np[:max_n]

            all_white.append(w_np);  all_black.append(b_np)
            all_counts.append(c_np); all_cp_raw.append(cp_np); all_pc.append(pc_np)

        white_np  = np.concatenate(all_white,  axis=0) if len(all_white)  > 1 else all_white[0]
        black_np  = np.concatenate(all_black,  axis=0) if len(all_black)  > 1 else all_black[0]
        counts_np = np.concatenate(all_counts, axis=0) if len(all_counts) > 1 else all_counts[0]
        cp_raw_np = np.concatenate(all_cp_raw, axis=0) if len(all_cp_raw) > 1 else all_cp_raw[0]
        pc_np     = np.concatenate(all_pc,     axis=0) if len(all_pc)     > 1 else all_pc[0]

        N = len(cp_raw_np)

        # Materialise mmap → RAM as int32.
        # Files saved by download_lichess_hf.py use int16 storage for compactness,
        # but HalfKAv2 indices reach 45,055 which exceeds int16 max (32,767).
        # The values were originally uint16; viewing as uint16 recovers the original
        # values before casting to int32. HalfKP values (≤24,576) are unaffected.
        def _load_indices(arr: np.ndarray) -> np.ndarray:
            if arr.dtype == np.int16:
                return arr.view(np.uint16).astype(np.int32)
            return arr.astype(np.int32)

        white_arr  = _load_indices(white_np)
        black_arr  = _load_indices(black_np)
        counts_arr = counts_np.astype(np.int32)

        # Fill padding slots (positions ≥ count) with SENTINEL column by column.
        # preprocess_dataset.py zero-initialises the npy arrays, so unused slots
        # contain 0 — a valid feature index.  We must overwrite them with SENTINEL
        # so EmbeddingBag (padding_idx=feature_dim) ignores them correctly.
        for j in range(32):
            mask = counts_arr <= j  # (N,) bool — rows where column j is padding
            if mask.any():
                white_arr[mask, j] = SENTINEL
                black_arr[mask, j] = SENTINEL

        # Precompute WDL targets and clip cp
        cp_raw = np.asarray(cp_raw_np, dtype=np.float32)
        cp_clipped = np.clip(cp_raw, -max_cp_abs, max_cp_abs)
        # Ply proxy from piece count: (32 - n_pieces) * 4, clamped to [0, 240]
        ply_est = np.clip((32 - pc_np.astype(np.float32)) * 4, 0.0, 240.0)
        wdl_np = cp_to_wdl_batch(cp_raw, ply=ply_est)  # (N, 3) float32

        # Transfer to GPU
        print(f"  Transferring {N:,} positions to GPU...", flush=True)
        self.white_idx = torch.from_numpy(white_arr).to(device)                             # (N, 32) int32
        self.black_idx = torch.from_numpy(black_arr).to(device)                             # (N, 32) int32
        self.cp  = torch.from_numpy(cp_clipped).unsqueeze(1).to(device)                     # (N, 1)  float32
        self.wdl = torch.from_numpy(wdl_np).to(device)                                     # (N, 3)  float32
        self.pc  = torch.from_numpy(np.asarray(pc_np, dtype=np.int64)).unsqueeze(1).to(device)  # (N, 1) int64

        self.N = N
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device

        mem_bytes = (self.white_idx.nbytes + self.black_idx.nbytes +
                     self.cp.nbytes + self.wdl.nbytes + self.pc.nbytes)
        print(f"  Loaded: {N:,} positions  ({mem_bytes / 1e9:.2f} GB GPU RAM)")

    def __len__(self) -> int:
        return (self.N + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        # Generate permutation on CPU (Fisher-Yates, no GPU temp workspace) then
        # transfer.  torch.randperm on GPU uses a sort-based algorithm that needs
        # ~2× N int64 = ~1.8 GB extra VRAM for 120M positions — causes OOM when
        # the preloaded data already occupies most of VRAM.
        if self.shuffle:
            perm = torch.randperm(self.N, dtype=torch.int32).to(self.device)
        else:
            perm = torch.arange(self.N, dtype=torch.int32, device=self.device)
        for start in range(0, self.N, self.batch_size):
            idx = perm[start : start + self.batch_size].long()
            # Cast int16 → int64 on GPU (EmbeddingBag requires int64 indices)
            yield (
                self.white_idx[idx].to(torch.int64),
                self.black_idx[idx].to(torch.int64),
                self.pc[idx],
                self.cp[idx],
                self.wdl[idx],
            )


class ShardedGPUDataset:
    """GPU-resident training for datasets larger than VRAM.

    Partitions the training data into small shards that each fit comfortably in
    GPU memory (~2 GB per shard = ~15M positions).  Each call to __iter__ loads
    exactly one shard to GPU, yields all batches from it, then frees the VRAM.
    The training loop's "epoch" therefore equals one shard mini-epoch.

    After cycling through all N shards (one full pass), the shard order is
    reshuffled so consecutive passes have different orderings.

    Usage in config:
        training:
          sharded_gpu: true
        data:
          train_shards: data/lichess_hf/shards   # directory of shard_??.*.npy files
          val_file: data/all_69m/val_all_69m.jsonl

    A training run of 'epochs: 200' with 20 shards = 10 full passes over the dataset.
    """

    def __init__(
        self,
        shard_dirs: "str | list[str]",
        max_cp_abs: int,
        device: torch.device,
        batch_size: int,
        shuffle: bool = True,
        feature_dim: int = HALFKP_FEATURE_DIM,
    ):
        if isinstance(shard_dirs, str):
            shard_dirs = [shard_dirs]

        # Discover all shard prefixes.
        # Each entry can be a directory (glob for *.cp.npy) or a file prefix
        # (e.g. "data/all_69m/train_all_69m" → single shard, avoids picking up val files).
        self.shard_prefixes: list[str] = []
        for d in shard_dirs:
            p = Path(d)
            if p.is_dir():
                cp_files = sorted(p.glob("*.cp.npy"))
            elif Path(str(p) + ".cp.npy").exists():
                cp_files = [Path(str(p) + ".cp.npy")]  # explicit prefix
            else:
                raise ValueError(f"Shard path not found as directory or prefix: {d}")
            for cp_file in cp_files:
                prefix = str(cp_file).replace(".cp.npy", "")
                wi = Path(prefix + ".white_indices.npy")
                bi = Path(prefix + ".black_indices.npy")
                if wi.exists() and bi.exists():
                    self.shard_prefixes.append(prefix)

        if not self.shard_prefixes:
            raise ValueError(f"No binary shards found in: {shard_dirs}")

        self.max_cp_abs = max_cp_abs
        self.device = device
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.feature_dim = feature_dim

        # Count total positions across all shards for reporting
        shard_sizes = [len(np.load(p + ".cp.npy", mmap_mode="r"))
                       for p in self.shard_prefixes]
        self._total_N = sum(shard_sizes)
        self._avg_shard_N = self._total_N // len(self.shard_prefixes)

        # Shuffled queue: which shard to load next
        self._queue: list[int] = []

        print(f"ShardedGPUDataset: {len(self.shard_prefixes)} shards  "
              f"{self._total_N:,} total positions  "
              f"(~{self._avg_shard_N:,}/shard  "
              f"{len(self.shard_prefixes)} mini-epochs per full pass)")

    # N and len() report per-shard stats so train.py batch-count logging is sensible
    @property
    def N(self) -> int:
        return self._avg_shard_N

    def __len__(self) -> int:
        return (self._avg_shard_N + self.batch_size - 1) // self.batch_size

    def _next_shard_idx(self) -> int:
        if not self._queue:
            order = list(range(len(self.shard_prefixes)))
            if self.shuffle:
                np.random.shuffle(order)
            self._queue = order
        return self._queue.pop(0)

    def __iter__(self):
        shard_idx = self._next_shard_idx()
        prefix = self.shard_prefixes[shard_idx]
        shard_name = Path(prefix).name
        pass_n = (len(self.shard_prefixes) - len(self._queue) - 1)
        print(f"  [shard {shard_idx}/{len(self.shard_prefixes)-1}] "
              f"Loading {shard_name} ...", flush=True)

        gpu_shard = GPUPreloadedDualDataset(
            [(prefix, None)],
            max_cp_abs=self.max_cp_abs,
            device=self.device,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            feature_dim=self.feature_dim,
        )
        yield from gpu_shard

        # Explicitly free VRAM before next shard
        del gpu_shard
        torch.cuda.empty_cache()


class JsonlDualPositionDataset(Dataset):
    """Fallback dual-perspective JSONL dataset (no preprocessing required).

    Returns (x_white, x_black, piece_count, cp, wdl) matching BinaryDualPositionDataset.
    Slower than the binary dataset but works directly from JSONL files.

    Set use_halfkav2=True to encode with HalfKAv2 features (45,056-dim) instead
    of HalfKP (24,576-dim). Must match the model's input_dim.
    """

    def __init__(self, path: str, max_cp_abs: int = 1500, use_halfkav2: bool = False):
        self.path = path
        self.use_halfkav2 = use_halfkav2
        offsets = []
        cp_values = []
        cp_raw_values = []

        with open(path, "rb") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                row = json.loads(line)
                cp_raw = float(row["cp"])
                cp = max(-max_cp_abs, min(max_cp_abs, cp_raw))
                offsets.append(offset)
                cp_values.append(cp)
                cp_raw_values.append(cp_raw)

        self.offsets = np.array(offsets, dtype=np.int64)
        self.cp_values = np.array(cp_values, dtype=np.float32)
        self.cp_raw_values = np.array(cp_raw_values, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, idx: int):
        with open(self.path, "rb") as f:
            f.seek(int(self.offsets[idx]))
            line = f.readline()
        row = json.loads(line)
        board = chess.Board(row["fen"])

        if self.use_halfkav2:
            w_idx, b_idx = encode_board_halfkav2_dual(board)
        else:
            w_idx, b_idx = encode_board_halfkp_dual(board)
        piece_count = np.array([len(board.piece_map())], dtype=np.int64)
        # JSONL stores side-to-move CP; model expects white-absolute.
        sign = 1.0 if board.turn == chess.WHITE else -1.0
        cp = np.array([self.cp_values[idx] * sign], dtype=np.float32)
        wdl = cp_to_wdl_target(float(self.cp_raw_values[idx]) * sign,
                               ply=_ply_from_fen(row["fen"]))

        return (
            torch.from_numpy(w_idx),
            torch.from_numpy(b_idx),
            torch.from_numpy(piece_count),
            torch.from_numpy(cp),
            torch.from_numpy(wdl),
        )


class JsonlPositionDataset(Dataset):
    """Fallback JSONL dataset using byte offsets (no preprocessing required).

    Slower than BinaryPositionDataset but works directly from JSONL files.
    """

    def __init__(self, path: str, max_cp_abs: int = 1500, use_halfkp: bool = False):
        self.path = path
        self.use_halfkp = use_halfkp

        offsets = []
        cp_values = []
        cp_raw_values = []

        with open(path, "rb") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                row = json.loads(line)
                cp_raw = float(row["cp"])
                cp = max(-max_cp_abs, min(max_cp_abs, cp_raw))
                offsets.append(offset)
                cp_values.append(cp)
                cp_raw_values.append(cp_raw)

        self.offsets = np.array(offsets, dtype=np.int64)
        self.cp_values = np.array(cp_values, dtype=np.float32)
        self.cp_raw_values = np.array(cp_raw_values, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, idx: int):
        with open(self.path, "rb") as f:
            f.seek(int(self.offsets[idx]))
            line = f.readline()
        row = json.loads(line)
        board = chess.Board(row["fen"])

        if self.use_halfkp:
            x = encode_board_halfkp(board)
        else:
            x = encode_board_12x64(board)

        cp = np.array([self.cp_values[idx]], dtype=np.float32)
        wdl = cp_to_wdl_target(float(self.cp_raw_values[idx]),
                               ply=_ply_from_fen(row["fen"]))

        return (
            torch.from_numpy(x),
            torch.from_numpy(cp),
            torch.from_numpy(wdl),
        )

from __future__ import annotations

import json
from pathlib import Path

import chess
import numpy as np
import torch
from torch.utils.data import Dataset

from .features import cp_to_wdl_target, cp_to_wdl_batch, encode_board_12x64, encode_board_halfkp, HALFKP_FEATURE_DIM, FEATURE_DIM, encode_board_halfkp_dual


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

    def __init__(self, path: str, max_cp_abs: int = 1500):
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

    def __len__(self) -> int:
        return len(self.cp)

    def __getitem__(self, idx: int):
        count = int(self.counts[idx])
        SENTINEL = HALFKP_FEATURE_DIM  # 24576

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

    Memory: ~136 bytes/position  (indices×2 as int16, cp/wdl/pc as float32/int64).
    Example: 77M positions ≈ 10.5 GB VRAM.

    Drop-in replacement for DataLoader: supports len() and __iter__,
    yielding (w_idx, b_idx, piece_count, cp, wdl) tuples with all tensors on GPU.
    """

    def __init__(
        self,
        path: str,
        max_cp_abs: int,
        device: torch.device,
        batch_size: int,
        shuffle: bool = True,
    ):
        prefix = str(Path(path).with_suffix(""))
        SENTINEL = HALFKP_FEATURE_DIM  # 24576 — fits in int16 (max 32767)

        print(f"  GPU preload: {Path(path).name} → {device}")

        white_np = np.load(prefix + ".white_indices.npy", mmap_mode="r")  # (N, 32) uint16
        black_np = np.load(prefix + ".black_indices.npy", mmap_mode="r")  # (N, 32) uint16
        counts_np = np.load(prefix + ".counts.npy",       mmap_mode="r")  # (N,)    uint8
        cp_raw_np = np.load(prefix + ".cp.npy",           mmap_mode="r")  # (N,)    float32
        pc_path = prefix + ".piece_count.npy"
        pc_np = np.load(pc_path, mmap_mode="r") if Path(pc_path).exists() else counts_np

        N = len(cp_raw_np)

        # Materialise mmap → RAM as int16 (uint16 values ≤24576 all fit in int16)
        white_arr = white_np.astype(np.int16)
        black_arr = black_np.astype(np.int16)
        counts_arr = counts_np.astype(np.int32)

        # Fill padding slots (positions ≥ count) with SENTINEL column by column.
        # preprocess_dataset.py zero-initialises the npy arrays, so unused slots
        # contain 0 — a valid feature index.  We must overwrite them with SENTINEL
        # so EmbeddingBag (padding_idx=HALFKP_FEATURE_DIM) ignores them correctly.
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
        self.white_idx = torch.from_numpy(white_arr).to(device)                             # (N, 32) int16
        self.black_idx = torch.from_numpy(black_arr).to(device)                             # (N, 32) int16
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


class JsonlDualPositionDataset(Dataset):
    """Fallback dual-perspective JSONL dataset (no preprocessing required).

    Returns (x_white, x_black, piece_count, cp, wdl) matching BinaryDualPositionDataset.
    Slower than the binary dataset but works directly from JSONL files.
    """

    def __init__(self, path: str, max_cp_abs: int = 1500):
        self.path = path
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

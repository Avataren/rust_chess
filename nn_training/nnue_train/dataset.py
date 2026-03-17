from __future__ import annotations

import json
from pathlib import Path

import chess
import numpy as np
import torch
from torch.utils.data import Dataset

from .features import cp_to_wdl_target, encode_board_12x64, encode_board_halfkp, HALFKP_FEATURE_DIM, FEATURE_DIM, encode_board_halfkp_dual


class BinaryPositionDataset(Dataset):
    """Fast dataset backed by pre-encoded sparse binary files.

    Use scripts/preprocess_dataset.py to generate the binary files from JSONL.
    Each __getitem__ is a memmap read + sparse scatter — no JSON parsing or
    python-chess encoding at training time, giving ~10-20x DataLoader speedup.

    Files expected (given path = "data/train_10m.jsonl"):
      data/train_10m.indices.npy  -- (N, 32) uint16
      data/train_10m.counts.npy   -- (N,)    uint8
      data/train_10m.cp.npy       -- (N,)    float32
    """

    def __init__(self, path: str, max_cp_abs: int = 1500, use_halfkp: bool = True):
        prefix = str(Path(path).with_suffix(""))
        self.feature_dim = HALFKP_FEATURE_DIM if use_halfkp else FEATURE_DIM

        self.indices = np.load(prefix + ".indices.npy", mmap_mode="r")
        self.counts  = np.load(prefix + ".counts.npy",  mmap_mode="r")
        self.cp      = np.load(prefix + ".cp.npy",      mmap_mode="r")

        # Clip cp values
        self.cp = np.clip(self.cp, -max_cp_abs, max_cp_abs)

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
        wdl = cp_to_wdl_target(cp_val)

        return (
            torch.from_numpy(indices),
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
    """

    def __init__(self, path: str, max_cp_abs: int = 1500):
        prefix = str(Path(path).with_suffix(""))
        self.white_indices = np.load(prefix + ".white_indices.npy", mmap_mode="r")
        self.black_indices = np.load(prefix + ".black_indices.npy", mmap_mode="r")
        self.counts = np.load(prefix + ".counts.npy", mmap_mode="r")
        self.cp = np.clip(np.load(prefix + ".cp.npy", mmap_mode="r"), -max_cp_abs, max_cp_abs)

    def __len__(self) -> int:
        return len(self.cp)

    def __getitem__(self, idx: int):
        count = int(self.counts[idx])
        SENTINEL = HALFKP_FEATURE_DIM  # 12288

        w_idx = np.full(32, SENTINEL, dtype=np.int64)
        b_idx = np.full(32, SENTINEL, dtype=np.int64)
        w_idx[:count] = self.white_indices[idx, :count]
        b_idx[:count] = self.black_indices[idx, :count]

        cp_val = float(self.cp[idx])
        cp = np.array([cp_val], dtype=np.float32)
        wdl = cp_to_wdl_target(cp_val)

        return (
            torch.from_numpy(w_idx),
            torch.from_numpy(b_idx),
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

        with open(path, "rb") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                row = json.loads(line)
                cp = float(row["cp"])
                cp = max(-max_cp_abs, min(max_cp_abs, cp))
                offsets.append(offset)
                cp_values.append(cp)

        self.offsets = np.array(offsets, dtype=np.int64)
        self.cp_values = np.array(cp_values, dtype=np.float32)

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
        wdl = cp_to_wdl_target(float(self.cp_values[idx]))

        return (
            torch.from_numpy(x),
            torch.from_numpy(cp),
            torch.from_numpy(wdl),
        )

from __future__ import annotations

import json
from dataclasses import dataclass

import chess
import numpy as np
import torch
from torch.utils.data import Dataset

from .features import cp_to_wdl_target, encode_board_12x64, encode_board_halfkp


@dataclass
class Sample:
    fen: str
    cp: float


class JsonlPositionDataset(Dataset):
    """Loads position samples from line-delimited JSON.

    Expected format per line:
      {"fen": "...", "cp": 32.1}

    Set use_halfkp=True to use king-bucketed HalfKP features (12,288-dim)
    instead of the original 12x64 features (768-dim).
    """

    def __init__(self, path: str, max_cp_abs: int = 1500, use_halfkp: bool = False):
        self.samples: list[Sample] = []
        self.use_halfkp = use_halfkp
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                cp = float(row["cp"])
                cp = max(-max_cp_abs, min(max_cp_abs, cp))
                self.samples.append(Sample(fen=row["fen"], cp=cp))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        board = chess.Board(s.fen)
        if self.use_halfkp:
            x = encode_board_halfkp(board)
        else:
            x = encode_board_12x64(board)

        cp = np.array([s.cp], dtype=np.float32)
        wdl = cp_to_wdl_target(s.cp)

        return (
            torch.from_numpy(x),
            torch.from_numpy(cp),
            torch.from_numpy(wdl),
        )

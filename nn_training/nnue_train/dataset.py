from __future__ import annotations

import json
from dataclasses import dataclass

import chess
import numpy as np
import torch
from torch.utils.data import Dataset

from .features import cp_to_wdl_target, encode_board_12x64


@dataclass
class Sample:
    fen: str
    cp: float


class JsonlPositionDataset(Dataset):
    """Loads position samples from line-delimited JSON.

    Expected format per line:
      {"fen": "...", "cp": 32.1}
    """

    def __init__(self, path: str, max_cp_abs: int = 1500):
        self.samples: list[Sample] = []
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
        x = encode_board_12x64(board)

        cp = np.array([s.cp], dtype=np.float32)
        wdl = cp_to_wdl_target(s.cp)

        return (
            torch.from_numpy(x),
            torch.from_numpy(cp),
            torch.from_numpy(wdl),
        )

from __future__ import annotations

import torch
from torch import nn


class EvalNet(nn.Module):
    """A small NNUE-like dense model with value and WDL heads.

    Input is sparse 12x64 features. For WASM runtime, you can export and run the
    quantized linear layers directly in Rust.
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,
        hidden2_dim: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden2_dim),
            nn.ReLU(),
        )
        self.cp_head = nn.Linear(hidden2_dim, 1)
        self.wdl_head = nn.Linear(hidden2_dim, 3)

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        cp = self.cp_head(h)
        wdl_logits = self.wdl_head(h)
        return cp, wdl_logits

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class EvalNetDual(nn.Module):
    """Dual-perspective NNUE: two shared-weight EmbeddingBag lookups (white + black king
    views), each ReLU'd, then concatenated before fc2.

    Architecture: [h_white(512) | h_black(512)] → fc2(1024→32) → ReLU → heads
    Convention: always white-first concatenation regardless of side to move.

    CP targets must use white-absolute convention (positive = good for white).
    """

    def __init__(
        self,
        input_dim: int = 12288,
        hidden_dim: int = 512,
        hidden2_dim: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim

        # Shared feature transformer — both perspectives use the same weights
        self.embedding = nn.EmbeddingBag(
            input_dim + 1, hidden_dim, mode="sum", padding_idx=input_dim
        )
        self.bias1 = nn.Parameter(torch.zeros(hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden2_dim)  # 1024 → 32
        self.cp_head = nn.Linear(hidden2_dim, 1)
        self.wdl_head = nn.Linear(hidden2_dim, 3)

    def forward(self, x_white: torch.Tensor, x_black: torch.Tensor):
        h_w = F.relu(self.embedding(x_white) + self.bias1)
        h_b = F.relu(self.embedding(x_black) + self.bias1)  # shared weights
        h = F.relu(self.fc2(self.dropout(torch.cat([h_w, h_b], dim=1))))
        return self.cp_head(h), self.wdl_head(h)


class EvalNet(nn.Module):
    """NNUE-style evaluation network with optional sparse input via EmbeddingBag.

    sparse_input=True (recommended for HalfKP training):
      - First layer uses nn.EmbeddingBag: 32 row lookups directly into the weight
        matrix, never materializing the 12,288-dim dense input vector on GPU.
      - Input to forward(): (batch, 32) int64 indices, padding_idx=input_dim.
      - ~800x less PCIe traffic and no 400MB/batch GPU allocation vs dense path.

    sparse_input=False (legacy / JSONL fallback):
      - Standard nn.Linear first layer.
      - Input to forward(): (batch, input_dim) float32 dense tensor.

    Both modes export identical weight shapes to .npz — Rust inference unchanged.
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,
        hidden2_dim: int = 32,
        dropout: float = 0.0,
        sparse_input: bool = False,
    ):
        super().__init__()
        self.sparse_input = sparse_input
        self.input_dim = input_dim

        if sparse_input:
            # padding_idx=input_dim: index used to pad variable-length bags is ignored
            self.embedding = nn.EmbeddingBag(
                input_dim + 1, hidden_dim, mode="sum", padding_idx=input_dim
            )
            self.bias1 = nn.Parameter(torch.zeros(hidden_dim))
        else:
            self.fc1 = nn.Linear(input_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden2_dim)
        self.cp_head = nn.Linear(hidden2_dim, 1)
        self.wdl_head = nn.Linear(hidden2_dim, 3)

    def forward(self, x: torch.Tensor):
        if self.sparse_input:
            h = F.relu(self.embedding(x) + self.bias1)
        else:
            h = F.relu(self.fc1(x))

        h = F.relu(self.fc2(self.dropout(h)))
        return self.cp_head(h), self.wdl_head(h)

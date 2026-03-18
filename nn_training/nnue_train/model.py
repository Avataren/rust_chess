from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

N_OUTPUT_BUCKETS = 8


def screlu(x: torch.Tensor) -> torch.Tensor:
    """SCReLU activation: clamp(x, 0, 1)²."""
    return torch.clamp(x, 0.0, 1.0).pow(2)


def get_output_bucket(piece_count: torch.Tensor, n_output_buckets: int = N_OUTPUT_BUCKETS) -> torch.Tensor:
    """Map total piece count to output bucket index (same formula as Rust piece_bucket)."""
    return ((piece_count - 2) * n_output_buckets // 30).clamp(0, n_output_buckets - 1)


class EvalNetDual(nn.Module):
    """Dual-perspective NNUE: two shared-weight EmbeddingBag lookups (white + black king
    views), each SCReLU'd, then concatenated before fc2.

    Architecture: [h_white(hidden_dim) | h_black(hidden_dim)] → fc2 → SCReLU → bucketed heads
    Convention: always white-first concatenation regardless of side to move.

    CP targets must use white-absolute convention (positive = good for white).
    Output buckets: separate head weights per game phase (piece-count bucketed).
    """

    def __init__(
        self,
        input_dim: int = 12288,
        hidden_dim: int = 1024,
        hidden2_dim: int = 32,
        dropout: float = 0.0,
        n_output_buckets: int = N_OUTPUT_BUCKETS,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_output_buckets = n_output_buckets

        # Shared feature transformer — both perspectives use the same weights
        self.embedding = nn.EmbeddingBag(
            input_dim + 1, hidden_dim, mode="sum", padding_idx=input_dim
        )
        # Small init: with ~32 active features summed, default uniform[-1,1] produces
        # pre-activation magnitudes of ~32*0.5=16, saturating SCReLU hard.
        # Scaling down ensures most neurons start in the active SCReLU region [0,1].
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        self.bias1 = nn.Parameter(torch.zeros(hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden2_dim)
        # Bucketed output heads: one row per bucket
        self.cp_head = nn.Linear(hidden2_dim, n_output_buckets)
        self.wdl_head = nn.Linear(hidden2_dim, n_output_buckets * 3)

    def forward(self, x_white: torch.Tensor, x_black: torch.Tensor, piece_count: torch.Tensor):
        h_w = screlu(self.embedding(x_white) + self.bias1)
        h_b = screlu(self.embedding(x_black) + self.bias1)  # shared weights
        h = screlu(self.fc2(self.dropout(torch.cat([h_w, h_b], dim=1))))

        B = x_white.size(0)
        bucket = get_output_bucket(piece_count, self.n_output_buckets).long().view(-1)  # (B,)

        cp_all = self.cp_head(h)                                # (B, n_output_buckets)
        wdl_all = self.wdl_head(h)                             # (B, n_output_buckets*3)

        cp = cp_all[torch.arange(B, device=h.device), bucket].unsqueeze(1)   # (B, 1)
        wdl = wdl_all.view(B, self.n_output_buckets, 3)[torch.arange(B, device=h.device), bucket]  # (B, 3)
        return cp, wdl


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
        hidden_dim: int = 1024,
        hidden2_dim: int = 32,
        dropout: float = 0.0,
        sparse_input: bool = False,
        n_output_buckets: int = N_OUTPUT_BUCKETS,
    ):
        super().__init__()
        self.sparse_input = sparse_input
        self.input_dim = input_dim
        self.n_output_buckets = n_output_buckets

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
        self.cp_head = nn.Linear(hidden2_dim, n_output_buckets)
        self.wdl_head = nn.Linear(hidden2_dim, n_output_buckets * 3)

    def forward(self, x: torch.Tensor, piece_count: torch.Tensor):
        if self.sparse_input:
            h = screlu(self.embedding(x) + self.bias1)
        else:
            h = screlu(self.fc1(x))

        h = screlu(self.fc2(self.dropout(h)))

        B = x.size(0)
        bucket = get_output_bucket(piece_count, self.n_output_buckets).long().view(-1)  # (B,)

        cp_all = self.cp_head(h)
        wdl_all = self.wdl_head(h)
        cp = cp_all[torch.arange(B, device=h.device), bucket].unsqueeze(1)
        wdl = wdl_all.view(B, self.n_output_buckets, 3)[torch.arange(B, device=h.device), bucket]
        return cp, wdl

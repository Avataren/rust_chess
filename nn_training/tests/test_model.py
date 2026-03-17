"""Tests for SCReLU activation, output buckets, and model forward pass."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from nnue_train.model import screlu, get_output_bucket, EvalNetDual, N_OUTPUT_BUCKETS


def test_screlu_values():
    x = torch.tensor([0.0, 0.5, 1.0, 1.5, -0.5])
    y = screlu(x)
    expected = torch.tensor([0.0, 0.25, 1.0, 1.0, 0.0])
    assert torch.allclose(y, expected), f"screlu values wrong: {y} vs {expected}"


def test_screlu_gradient():
    # screlu(x) = clamp(x, 0, 1)^2
    # Gradient: 2*clamp(x,0,1) * d_clamp/dx
    # d_clamp/dx = 1 inside (0,1), 0 outside (saturated at boundaries)
    # x=-0.5 → clamped to 0, grad=0; x=0.5 → 2*0.5=1.0; x=1.5 → clamped to 1, d_clamp/dx=0 → grad=0
    x = torch.tensor([-0.5, 0.5, 1.5], requires_grad=True)
    y = screlu(x).sum()
    y.backward()
    expected_grad = torch.tensor([0.0, 1.0, 0.0])
    assert torch.allclose(x.grad, expected_grad), f"screlu grad wrong: {x.grad}"


def test_output_bucket_selection():
    # Verify bucket formula: clamp((pc - 2) * N // 30, 0, N-1)
    N = N_OUTPUT_BUCKETS
    for pc in range(2, 33):
        expected = min((pc - 2) * N // 30, N - 1)
        result = int(get_output_bucket(torch.tensor([pc]), N).item())
        assert result == expected, f"pc={pc}: expected bucket {expected}, got {result}"

    # Extremes
    assert int(get_output_bucket(torch.tensor([2]), N).item()) == 0
    assert int(get_output_bucket(torch.tensor([32]), N).item()) == N - 1


def test_model_forward_shape():
    B = 8
    model = EvalNetDual(input_dim=12288, hidden_dim=64, hidden2_dim=16, n_output_buckets=4)
    model.eval()
    SENTINEL = 12288
    x_white = torch.full((B, 32), SENTINEL, dtype=torch.int64)
    x_black = torch.full((B, 32), SENTINEL, dtype=torch.int64)
    piece_count = torch.randint(2, 33, (B, 1))

    with torch.no_grad():
        cp, wdl = model(x_white, x_black, piece_count)

    assert cp.shape == (B, 1), f"cp shape wrong: {cp.shape}"
    assert wdl.shape == (B, 3), f"wdl shape wrong: {wdl.shape}"


def test_model_backward_runs():
    """loss.backward() must run without error; only selected bucket gets gradient."""
    B = 4
    model = EvalNetDual(input_dim=12288, hidden_dim=32, hidden2_dim=16, n_output_buckets=N_OUTPUT_BUCKETS)
    SENTINEL = 12288
    x_white = torch.full((B, 32), SENTINEL, dtype=torch.int64)
    x_black = torch.full((B, 32), SENTINEL, dtype=torch.int64)
    piece_count = torch.full((B, 1), 20, dtype=torch.int64)  # all same bucket

    cp, wdl = model(x_white, x_black, piece_count)
    loss = cp.mean() + wdl.mean()
    loss.backward()  # must not raise

    # cp_head.weight should have a gradient (at least the selected bucket row)
    assert model.cp_head.weight.grad is not None, "cp_head.weight must have grad after backward"

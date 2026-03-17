#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from nnue_train.model import EvalNet, EvalNetDual

INT16_MIN = np.iinfo(np.int16).min
INT16_MAX = np.iinfo(np.int16).max


def quantize_int16(
    t: torch.Tensor,
    scale: float = 256.0,
    on_overflow: str = "error",
    name: str = "tensor",
) -> np.ndarray:
    arr_f = (t.detach().cpu().numpy() * scale).round()
    overflow_mask = (arr_f < INT16_MIN) | (arr_f > INT16_MAX)

    if np.any(overflow_mask):
        n_over = int(overflow_mask.sum())
        total = int(arr_f.size)
        min_v = float(arr_f.min())
        max_v = float(arr_f.max())
        msg = (
            f"Quantization overflow in {name}: {n_over}/{total} values out of int16 range "
            f"[{INT16_MIN}, {INT16_MAX}] after scaling; observed [{min_v:.1f}, {max_v:.1f}]."
        )
        if on_overflow == "error":
            raise ValueError(msg)
        if on_overflow == "clip":
            print(f"Warning: {msg} Clipping to int16 range.")
            arr_f = np.clip(arr_f, INT16_MIN, INT16_MAX)
        else:
            raise ValueError(f"Unsupported on_overflow mode: {on_overflow}")

    return arr_f.astype(np.int16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="artifacts/checkpoint.pt")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--output", default="artifacts/nnue_like_weights.npz")
    ap.add_argument("--scale", type=float, default=256.0)
    ap.add_argument(
        "--on-overflow",
        choices=["error", "clip"],
        default="error",
        help="Behavior when scaled values exceed int16 range.",
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    mcfg = cfg["model"]

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt["model_state"]

    # Auto-detect dual model: shared embedding + fc2 has 1024 input columns
    is_dual = mcfg.get("dual_perspective", False) or (
        "embedding.weight" in state and "fc2.weight" in state
        and state["fc2.weight"].shape[1] == mcfg["hidden_dim"] * 2
    )

    if is_dual:
        model = EvalNetDual(
            input_dim=mcfg["input_dim"],
            hidden_dim=mcfg["hidden_dim"],
            hidden2_dim=mcfg["hidden2_dim"],
            dropout=0.0,
        )
    else:
        model = EvalNet(
            input_dim=mcfg["input_dim"],
            hidden_dim=mcfg["hidden_dim"],
            hidden2_dim=mcfg["hidden2_dim"],
            dropout=0.0,
            sparse_input=mcfg.get("sparse_input", False),
        )

    model.load_state_dict(state)
    model.eval()

    state = model.state_dict()
    s = args.scale
    ov = args.on_overflow

    # Extract first-layer weights in unified shape (hidden, input).
    # EmbeddingBag: (input+1, hidden) → take first input_dim rows, transpose.
    # Linear: (hidden, input) → use directly.
    if is_dual or (hasattr(model, "sparse_input") and model.sparse_input):
        w1 = state["embedding.weight"][:mcfg["input_dim"]].T.contiguous()
        b1 = state["bias1"]
    else:
        w1 = state["fc1.weight"]
        b1 = state["fc1.bias"]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        scale=np.array([s], dtype=np.float32),
        backbone_0_weight=quantize_int16(w1,                  s, ov, "backbone.0.weight"),
        backbone_0_bias=  quantize_int16(b1,                  s, ov, "backbone.0.bias"),
        backbone_3_weight=quantize_int16(state["fc2.weight"],  s, ov, "backbone.3.weight"),
        backbone_3_bias=  quantize_int16(state["fc2.bias"],    s, ov, "backbone.3.bias"),
        cp_head_weight=   quantize_int16(state["cp_head.weight"],  s, ov, "cp_head.weight"),
        cp_head_bias=     quantize_int16(state["cp_head.bias"],    s, ov, "cp_head.bias"),
        wdl_head_weight=  quantize_int16(state["wdl_head.weight"], s, ov, "wdl_head.weight"),
        wdl_head_bias=    quantize_int16(state["wdl_head.bias"],   s, ov, "wdl_head.bias"),
    )

    print(f"Wrote quantized weights to {out_path}")


if __name__ == "__main__":
    main()

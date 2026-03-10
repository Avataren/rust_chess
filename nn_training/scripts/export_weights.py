#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from nnue_train.model import EvalNet


def quantize_int16(t: torch.Tensor, scale: float = 256.0):
    arr = (t.detach().cpu().numpy() * scale).round().astype(np.int16)
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="artifacts/checkpoint.pt")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--output", default="artifacts/nnue_like_weights.npz")
    ap.add_argument("--scale", type=float, default=256.0)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    mcfg = cfg["model"]

    model = EvalNet(
        input_dim=mcfg["input_dim"],
        hidden_dim=mcfg["hidden_dim"],
        hidden2_dim=mcfg["hidden2_dim"],
        dropout=0.0,
    )

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    state = model.state_dict()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        scale=np.array([args.scale], dtype=np.float32),
        backbone_0_weight=quantize_int16(state["backbone.0.weight"], args.scale),
        backbone_0_bias=quantize_int16(state["backbone.0.bias"], args.scale),
        backbone_3_weight=quantize_int16(state["backbone.3.weight"], args.scale),
        backbone_3_bias=quantize_int16(state["backbone.3.bias"], args.scale),
        cp_head_weight=quantize_int16(state["cp_head.weight"], args.scale),
        cp_head_bias=quantize_int16(state["cp_head.bias"], args.scale),
    )

    print(f"Wrote quantized weights to {out_path}")


if __name__ == "__main__":
    main()

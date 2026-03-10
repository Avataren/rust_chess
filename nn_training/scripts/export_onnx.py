#!/usr/bin/env python3
"""Export a trained EvalNet checkpoint to ONNX format for use with tract-onnx in Rust."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from nnue_train.model import EvalNet


def main():
    ap = argparse.ArgumentParser(description="Export EvalNet checkpoint to ONNX")
    ap.add_argument("--checkpoint", default="artifacts/checkpoint.pt")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--output", default="artifacts/eval.onnx")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    mcfg = cfg["model"]

    model = EvalNet(
        input_dim=mcfg["input_dim"],
        hidden_dim=mcfg["hidden_dim"],
        hidden2_dim=mcfg["hidden2_dim"],
        dropout=0.0,
    )

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dummy = torch.zeros(1, mcfg["input_dim"])

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["features"],
        output_names=["cp", "wdl"],
        opset_version=args.opset,
        dynamic_axes={
            "features": {0: "batch"},
            "cp": {0: "batch"},
            "wdl": {0: "batch"},
        },
    )

    size_kb = out_path.stat().st_size / 1024
    print(f"Exported ONNX model to {out_path}  ({size_kb:.0f} KB)")
    print(f"Input:  features [batch, {mcfg['input_dim']}]")
    print(f"Output: cp [batch, 1]  (centipawns, side-to-move perspective)")
    print(f"        wdl [batch, 3] (win/draw/loss logits, not used in Rust)")


if __name__ == "__main__":
    main()

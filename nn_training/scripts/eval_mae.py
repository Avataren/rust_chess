#!/usr/bin/env python3
"""Evaluate a checkpoint's CP-MAE on an arbitrary JSONL file.

Usage:
  python3 scripts/eval_mae.py --checkpoint artifacts/best_checkpoint.pt \
                               --data data/gen_val.jsonl
Prints a single line:
  cp_mae=<value>
so it can be consumed by the self-play loop.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch import nn
from torch.utils.data import DataLoader

from nnue_train.dataset import JsonlDualPositionDataset
from nnue_train.model import EvalNetDual


def eval_mae(checkpoint_path: str, data_path: str, batch_size: int = 4096) -> float:
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {})
    mcfg = cfg.get("model", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EvalNetDual(
        input_dim=mcfg.get("input_dim", 24576),
        hidden_dim=mcfg.get("hidden_dim", 512),
        hidden2_dim=mcfg.get("hidden2_dim", 32),
        dropout=0.0,
        n_output_buckets=mcfg.get("n_output_buckets", 1),
    ).to(device)

    state = ck["model_state"]
    # strip torch.compile prefix if present
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    max_cp_abs = cfg.get("data", {}).get("max_cp_abs", 800)
    ds = JsonlDualPositionDataset(data_path, max_cp_abs=max_cp_abs)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)

    feature_dim = mcfg.get("input_dim", 24576)
    total_mae = 0.0
    total_n = 0

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 5:
                x_white, x_black, piece_count, cp, wdl = batch
            else:
                x_white, x_black, cp, wdl = batch
                piece_count = torch.full((x_white.size(0), 1), 32, dtype=torch.int64)

            x_white = x_white.to(device)
            x_black = x_black.to(device)
            piece_count = piece_count.to(device)
            cp = cp.to(device)

            cp_pred, _ = model(x_white, x_black, piece_count)
            mae = torch.abs(cp_pred.squeeze(1) - cp.squeeze(1)).sum().item()
            total_mae += mae
            total_n += cp.size(0)

    return total_mae / max(1, total_n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--batch-size", type=int, default=4096)
    args = ap.parse_args()

    mae = eval_mae(args.checkpoint, args.data, args.batch_size)
    print(f"cp_mae={mae:.4f}")


if __name__ == "__main__":
    main()

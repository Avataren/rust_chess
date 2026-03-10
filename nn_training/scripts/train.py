#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from nnue_train.dataset import JsonlPositionDataset
from nnue_train.model import EvalNet


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_epoch(model, loader, optimizer, scaler, device, cfg):
    model.train()
    cp_loss_fn = nn.HuberLoss(delta=100.0)
    wdl_loss_fn = nn.CrossEntropyLoss()

    total = 0.0
    for x, cp, wdl in tqdm(loader, desc="train", leave=False):
        x = x.to(device, non_blocking=True)
        cp = cp.to(device, non_blocking=True)
        wdl = wdl.to(device, non_blocking=True)
        wdl_idx = torch.argmax(wdl, dim=1)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=cfg["training"]["amp"] and device.type == "cuda"):
            cp_pred, wdl_logits = model(x)
            cp_loss = cp_loss_fn(cp_pred, cp)
            wdl_loss = wdl_loss_fn(wdl_logits, wdl_idx)
            loss = cfg["loss"]["cp_weight"] * cp_loss + cfg["loss"]["wdl_weight"] * wdl_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
        scaler.step(optimizer)
        scaler.update()

        total += float(loss.item())

    return total / max(1, len(loader))


@torch.no_grad()
def eval_epoch(model, loader, device, cfg):
    model.eval()
    cp_loss_fn = nn.HuberLoss(delta=100.0)
    wdl_loss_fn = nn.CrossEntropyLoss()

    total = 0.0
    for x, cp, wdl in tqdm(loader, desc="val", leave=False):
        x = x.to(device, non_blocking=True)
        cp = cp.to(device, non_blocking=True)
        wdl = wdl.to(device, non_blocking=True)
        wdl_idx = torch.argmax(wdl, dim=1)

        cp_pred, wdl_logits = model(x)
        cp_loss = cp_loss_fn(cp_pred, cp)
        wdl_loss = wdl_loss_fn(wdl_logits, wdl_idx)
        loss = cfg["loss"]["cp_weight"] * cp_loss + cfg["loss"]["wdl_weight"] * wdl_loss
        total += float(loss.item())

    return total / max(1, len(loader))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="artifacts/checkpoint.pt")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    set_seed(cfg["seed"])

    device = detect_device()
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA/ROCm device: {torch.cuda.get_device_name(0)}")

    train_ds = JsonlPositionDataset(
        cfg["data"]["train_file"],
        max_cp_abs=cfg["data"]["max_cp_abs"],
    )
    val_ds = JsonlPositionDataset(
        cfg["data"]["val_file"],
        max_cp_abs=cfg["data"]["max_cp_abs"],
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["workers"],
        pin_memory=cfg["training"]["pin_memory"],
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=max(1, cfg["training"]["workers"] // 2),
        pin_memory=cfg["training"]["pin_memory"],
    )

    mcfg = cfg["model"]
    model = EvalNet(
        input_dim=mcfg["input_dim"],
        hidden_dim=mcfg["hidden_dim"],
        hidden2_dim=mcfg["hidden2_dim"],
        dropout=mcfg["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_val = float("inf")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        tr = train_epoch(model, train_loader, optimizer, scaler, device, cfg)
        va = eval_epoch(model, val_loader, device, cfg)
        print(f"epoch={epoch} train={tr:.4f} val={va:.4f}")

        if va < best_val:
            best_val = va
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg,
                    "val_loss": va,
                    "epoch": epoch,
                },
                args.out,
            )
            print(f"saved checkpoint: {args.out}")


if __name__ == "__main__":
    main()

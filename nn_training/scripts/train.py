#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
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

    total_loss = 0.0
    total_cp_loss = 0.0
    total_wdl_loss = 0.0
    total_cp_mae = 0.0
    total_wdl_acc = 0.0

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

        cp_mae = torch.mean(torch.abs(cp_pred - cp))
        wdl_acc = (torch.argmax(wdl_logits, dim=1) == wdl_idx).float().mean()

        total_loss += float(loss.item())
        total_cp_loss += float(cp_loss.item())
        total_wdl_loss += float(wdl_loss.item())
        total_cp_mae += float(cp_mae.item())
        total_wdl_acc += float(wdl_acc.item())

    denom = max(1, len(loader))
    return {
        "loss": total_loss / denom,
        "cp_loss": total_cp_loss / denom,
        "wdl_loss": total_wdl_loss / denom,
        "cp_mae": total_cp_mae / denom,
        "wdl_acc": total_wdl_acc / denom,
    }


@torch.no_grad()
def eval_epoch(model, loader, device, cfg):
    model.eval()
    cp_loss_fn = nn.HuberLoss(delta=100.0)
    wdl_loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_cp_loss = 0.0
    total_wdl_loss = 0.0
    total_cp_mae = 0.0
    total_wdl_acc = 0.0

    for x, cp, wdl in tqdm(loader, desc="val", leave=False):
        x = x.to(device, non_blocking=True)
        cp = cp.to(device, non_blocking=True)
        wdl = wdl.to(device, non_blocking=True)
        wdl_idx = torch.argmax(wdl, dim=1)

        cp_pred, wdl_logits = model(x)
        cp_loss = cp_loss_fn(cp_pred, cp)
        wdl_loss = wdl_loss_fn(wdl_logits, wdl_idx)
        loss = cfg["loss"]["cp_weight"] * cp_loss + cfg["loss"]["wdl_weight"] * wdl_loss

        cp_mae = torch.mean(torch.abs(cp_pred - cp))
        wdl_acc = (torch.argmax(wdl_logits, dim=1) == wdl_idx).float().mean()

        total_loss += float(loss.item())
        total_cp_loss += float(cp_loss.item())
        total_wdl_loss += float(wdl_loss.item())
        total_cp_mae += float(cp_mae.item())
        total_wdl_acc += float(wdl_acc.item())

    denom = max(1, len(loader))
    return {
        "loss": total_loss / denom,
        "cp_loss": total_cp_loss / denom,
        "wdl_loss": total_wdl_loss / denom,
        "cp_mae": total_cp_mae / denom,
        "wdl_acc": total_wdl_acc / denom,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="artifacts/checkpoint.pt")
    ap.add_argument("--tb-logdir", default="runs/nn_training", help="TensorBoard log directory")
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


    if len(train_ds) == 0:
        raise ValueError("Training dataset is empty. Provide at least one training sample.")
    if len(val_ds) == 0:
        raise ValueError("Validation dataset is empty. Provide at least one validation sample.")

    if len(train_ds) < cfg["training"]["batch_size"]:
        print(
            "Warning: train dataset is smaller than batch_size; "
            "training will use a partial batch each epoch (drop_last=False)."
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["workers"],
        pin_memory=cfg["training"]["pin_memory"],
        drop_last=False,
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
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=args.tb_logdir)
    print(f"TensorBoard logdir: {args.tb_logdir}")

    try:
        for epoch in range(1, cfg["training"]["epochs"] + 1):
            tr = train_epoch(model, train_loader, optimizer, scaler, device, cfg)
            va = eval_epoch(model, val_loader, device, cfg)

            print(
                f"epoch={epoch} "
                f"train_loss={tr['loss']:.4f} val_loss={va['loss']:.4f} "
                f"train_cp_mae={tr['cp_mae']:.2f} val_cp_mae={va['cp_mae']:.2f} "
                f"train_wdl_acc={tr['wdl_acc']:.3f} val_wdl_acc={va['wdl_acc']:.3f}"
            )

            writer.add_scalar("train/loss", tr["loss"], epoch)
            writer.add_scalar("train/cp_loss", tr["cp_loss"], epoch)
            writer.add_scalar("train/wdl_loss", tr["wdl_loss"], epoch)
            writer.add_scalar("train/cp_mae", tr["cp_mae"], epoch)
            writer.add_scalar("train/wdl_acc", tr["wdl_acc"], epoch)

            writer.add_scalar("val/loss", va["loss"], epoch)
            writer.add_scalar("val/cp_loss", va["cp_loss"], epoch)
            writer.add_scalar("val/wdl_loss", va["wdl_loss"], epoch)
            writer.add_scalar("val/cp_mae", va["cp_mae"], epoch)
            writer.add_scalar("val/wdl_acc", va["wdl_acc"], epoch)
            writer.flush()

            if va["loss"] < best_val:
                best_val = va["loss"]
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "config": cfg,
                        "val_loss": va["loss"],
                        "epoch": epoch,
                    },
                    out_path,
                )
                print(f"saved checkpoint: {out_path}")
    finally:
        writer.close()


if __name__ == "__main__":
    main()

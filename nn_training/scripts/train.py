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

from torch.utils.data import Sampler
from nnue_train.dataset import BinaryPositionDataset, BinaryDualPositionDataset, JsonlPositionDataset
from nnue_train.model import EvalNet, EvalNetDual, get_output_bucket


class StratifiedEndgameSampler(Sampler):
    """Yields indices so that `endgame_fraction` of each epoch comes from
    endgame positions (≤16 pieces), oversampling them with replacement.
    Epoch length = len(dataset) (same number of batches as unweighted)."""

    def __init__(self, piece_counts: np.ndarray, endgame_fraction: float):
        self.eg_idx = np.where(piece_counts <= 16)[0]
        self.mg_idx = np.where(piece_counts > 16)[0]
        self.endgame_fraction = endgame_fraction
        self._epoch = 0
        n = len(piece_counts)
        self._n_eg = int(n * endgame_fraction)
        self._n_mg = n - self._n_eg

    def __iter__(self):
        rng = np.random.default_rng(self._epoch)
        self._epoch += 1
        eg = rng.choice(self.eg_idx, size=self._n_eg, replace=True)
        mg = rng.choice(self.mg_idx, size=self._n_mg,
                        replace=self._n_mg > len(self.mg_idx))
        idx = np.concatenate([eg, mg])
        rng.shuffle(idx)
        return iter(idx.tolist())

    def __len__(self) -> int:
        return self._n_eg + self._n_mg


def load_dataset(path: str, max_cp_abs: int, use_halfkp: bool, dual: bool = False):
    """Use fast binary dataset if pre-processed files exist, else fall back to JSONL."""
    prefix = str(Path(path).with_suffix(""))
    if dual and all(Path(prefix + ext).exists()
                    for ext in (".white_indices.npy", ".black_indices.npy", ".counts.npy", ".cp.npy")):
        print(f"  Loading dual binary dataset: {prefix}.*")
        return BinaryDualPositionDataset(path, max_cp_abs=max_cp_abs)
    if all(Path(prefix + ext).exists() for ext in (".indices.npy", ".counts.npy", ".cp.npy")):
        print(f"  Loading binary dataset: {prefix}.*")
        return BinaryPositionDataset(path, max_cp_abs=max_cp_abs, use_halfkp=use_halfkp)
    print(f"  Loading JSONL dataset: {path}  (run preprocess_dataset.py for faster training)")
    return JsonlPositionDataset(path, max_cp_abs=max_cp_abs, use_halfkp=use_halfkp)


def _extract_dual_batch(batch):
    """Unpack a dual-perspective batch, handling both old (4-tuple) and new (5-tuple) formats."""
    if len(batch) == 5:
        x_white, x_black, piece_count, cp, wdl = batch
    else:
        # Legacy dataset without piece_count: use placeholder of 32 pieces
        x_white, x_black, cp, wdl = batch
        piece_count = torch.full((x_white.size(0), 1), 32, dtype=torch.int64)
    return x_white, x_black, piece_count, cp, wdl


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def soft_target_cross_entropy(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=1)
    return -(target_probs * log_probs).sum(dim=1).mean()


def sparse_to_dense(indices: torch.Tensor, feature_dim: int) -> torch.Tensor:
    """Convert (batch, 32) sparse index tensor to (batch, feature_dim) dense float32.

    Indices equal to feature_dim are padding sentinels and are ignored.
    Runs on whatever device `indices` lives on (GPU after .to(device)).
    """
    mask = indices < feature_dim                          # (B, 32) bool
    safe = indices.clamp(0, feature_dim - 1)             # avoid OOB scatter
    x = torch.zeros(indices.size(0), feature_dim,
                    dtype=torch.float32, device=indices.device)
    x.scatter_(1, safe, mask.float())
    return x


def _forward_batch(model, batch, device, feature_dim):
    """Run model forward pass, handling both single and dual perspective models."""
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    if isinstance(raw, EvalNetDual):
        x_white, x_black, piece_count, cp, wdl = _extract_dual_batch(batch)
        x_white = x_white.to(device, non_blocking=True)
        x_black = x_black.to(device, non_blocking=True)
        piece_count = piece_count.to(device, non_blocking=True)
        cp = cp.to(device, non_blocking=True)
        wdl = wdl.to(device, non_blocking=True)
        cp_pred, wdl_logits = model(x_white, x_black, piece_count)
    else:
        if len(batch) == 4:
            x, piece_count, cp, wdl = batch
            piece_count = piece_count.to(device, non_blocking=True)
        else:
            # JsonlPositionDataset: no piece_count — use placeholder (max bucket)
            x, cp, wdl = batch
            piece_count = torch.full((x.size(0), 1), 32, dtype=torch.int64, device=device)
        x = x.to(device, non_blocking=True)
        if x.dtype == torch.int64 and not model.sparse_input:
            x = sparse_to_dense(x, feature_dim)
        cp = cp.to(device, non_blocking=True)
        wdl = wdl.to(device, non_blocking=True)
        cp_pred, wdl_logits = model(x, piece_count)
    bucket = get_output_bucket(piece_count, (model._orig_mod if hasattr(model, "_orig_mod") else model).n_output_buckets).long().view(-1)
    return cp_pred, wdl_logits, cp, wdl, bucket


def train_epoch(model, loader, optimizer, scaler, device, cfg):
    model.train()
    cp_loss_fn = nn.HuberLoss(delta=100.0)
    feature_dim = cfg["model"]["input_dim"]
    n_buckets = (model._orig_mod if hasattr(model, "_orig_mod") else model).n_output_buckets

    total_loss = 0.0
    total_cp_loss = 0.0
    total_wdl_loss = 0.0
    total_cp_mae = 0.0
    total_wdl_acc = 0.0
    total_wdl_target_confidence = 0.0
    bucket_cp_mae_sum = [0.0] * n_buckets
    bucket_count = [0] * n_buckets

    for batch in tqdm(loader, desc="train", leave=False):
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=cfg["training"]["amp"] and device.type == "cuda"):
            cp_pred, wdl_logits, cp, wdl, bucket = _forward_batch(model, batch, device, feature_dim)
            wdl_idx = torch.argmax(wdl, dim=1)
            cp_loss = cp_loss_fn(cp_pred, cp)
            wdl_loss = soft_target_cross_entropy(wdl_logits, wdl)
            loss = cfg["loss"]["cp_weight"] * cp_loss + cfg["loss"]["wdl_weight"] * wdl_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
        scaler.step(optimizer)
        scaler.update()

        cp_mae_per = torch.abs(cp_pred.squeeze(1) - cp.squeeze(1)).detach()
        wdl_acc = (torch.argmax(wdl_logits, dim=1) == wdl_idx).float().mean()
        target_confidence = torch.gather(wdl, 1, torch.argmax(wdl_logits, dim=1, keepdim=True)).mean()

        total_loss += float(loss.item())
        total_cp_loss += float(cp_loss.item())
        total_wdl_loss += float(wdl_loss.item())
        total_cp_mae += float(cp_mae_per.mean().item())
        total_wdl_acc += float(wdl_acc.item())
        total_wdl_target_confidence += float(target_confidence.item())

        for b in range(n_buckets):
            mask = bucket == b
            if mask.any():
                bucket_cp_mae_sum[b] += float(cp_mae_per[mask].sum().item())
                bucket_count[b] += int(mask.sum().item())

    denom = max(1, len(loader))
    metrics = {
        "loss": total_loss / denom,
        "cp_loss": total_cp_loss / denom,
        "wdl_loss": total_wdl_loss / denom,
        "cp_mae": total_cp_mae / denom,
        "wdl_acc": total_wdl_acc / denom,
        "wdl_target_confidence": total_wdl_target_confidence / denom,
    }
    for b in range(n_buckets):
        metrics[f"cp_mae_b{b}"] = bucket_cp_mae_sum[b] / max(1, bucket_count[b])
    return metrics


@torch.no_grad()

def eval_epoch(model, loader, device, cfg):
    model.eval()
    cp_loss_fn = nn.HuberLoss(delta=100.0)
    feature_dim = cfg["model"]["input_dim"]
    n_buckets = (model._orig_mod if hasattr(model, "_orig_mod") else model).n_output_buckets

    total_loss = 0.0
    total_cp_loss = 0.0
    total_wdl_loss = 0.0
    total_cp_mae = 0.0
    total_wdl_acc = 0.0
    total_wdl_target_confidence = 0.0
    bucket_cp_mae_sum = [0.0] * n_buckets
    bucket_count = [0] * n_buckets

    eg_cp_mae_sum = 0.0
    eg_count = 0

    for batch in tqdm(loader, desc="val", leave=False):
        cp_pred, wdl_logits, cp, wdl, bucket = _forward_batch(model, batch, device, feature_dim)
        wdl_idx = torch.argmax(wdl, dim=1)
        cp_loss = cp_loss_fn(cp_pred, cp)
        wdl_loss = soft_target_cross_entropy(wdl_logits, wdl)
        loss = cfg["loss"]["cp_weight"] * cp_loss + cfg["loss"]["wdl_weight"] * wdl_loss

        cp_mae_per = torch.abs(cp_pred.squeeze(1) - cp.squeeze(1))
        wdl_acc = (torch.argmax(wdl_logits, dim=1) == wdl_idx).float().mean()
        target_confidence = torch.gather(wdl, 1, torch.argmax(wdl_logits, dim=1, keepdim=True)).mean()

        total_loss += float(loss.item())
        total_cp_loss += float(cp_loss.item())
        total_wdl_loss += float(wdl_loss.item())
        total_cp_mae += float(cp_mae_per.mean().item())
        total_wdl_acc += float(wdl_acc.item())
        total_wdl_target_confidence += float(target_confidence.item())

        for b in range(n_buckets):
            mask = bucket == b
            if mask.any():
                bucket_cp_mae_sum[b] += float(cp_mae_per[mask].sum().item())
                bucket_count[b] += int(mask.sum().item())

        # Track endgame (≤16 pieces) separately regardless of bucket count
        # piece_count is buried in the batch; re-derive from bucket formula inverse
        # Instead, tag via piece_count directly from the batch
        if isinstance(batch, (list, tuple)) and len(batch) >= 5:
            raw_pc = batch[2].to(cp_pred.device)
            eg_mask = raw_pc.view(-1) <= 16
            if eg_mask.any():
                eg_cp_mae_sum += float(cp_mae_per[eg_mask].sum().item())
                eg_count += int(eg_mask.sum().item())

    denom = max(1, len(loader))
    metrics = {
        "loss": total_loss / denom,
        "cp_loss": total_cp_loss / denom,
        "wdl_loss": total_wdl_loss / denom,
        "cp_mae": total_cp_mae / denom,
        "wdl_acc": total_wdl_acc / denom,
        "wdl_target_confidence": total_wdl_target_confidence / denom,
    }
    for b in range(n_buckets):
        metrics[f"cp_mae_b{b}"] = bucket_cp_mae_sum[b] / max(1, bucket_count[b])
    metrics["cp_mae_endgame"] = eg_cp_mae_sum / max(1, eg_count)
    metrics["cp_mae_endgame_n"] = eg_count
    return metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="artifacts/checkpoint.pt")
    ap.add_argument("--resume", default=None, help="Resume fine-tuning from a checkpoint (.pt)")
    ap.add_argument("--reset-best-val", action="store_true",
                    help="Reset best_val to inf when resuming (use when finetuning on a different dataset)")
    ap.add_argument("--tb-logdir", default="runs/nn_training", help="TensorBoard log directory")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    set_seed(cfg["seed"])

    device = detect_device()
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA/ROCm device: {torch.cuda.get_device_name(0)}")

    use_halfkp = cfg["model"].get("use_halfkp", False)
    dual = cfg["model"].get("dual_perspective", False)
    train_ds = load_dataset(cfg["data"]["train_file"], cfg["data"]["max_cp_abs"], use_halfkp, dual=dual)
    val_ds   = load_dataset(cfg["data"]["val_file"],   cfg["data"]["max_cp_abs"], use_halfkp, dual=dual)


    if len(train_ds) == 0:
        raise ValueError("Training dataset is empty. Provide at least one training sample.")
    if len(val_ds) == 0:
        raise ValueError("Validation dataset is empty. Provide at least one validation sample.")

    if len(train_ds) < cfg["training"]["batch_size"]:
        print(
            "Warning: train dataset is smaller than batch_size; "
            "training will use a partial batch each epoch (drop_last=False)."
        )

    train_workers = int(cfg["training"]["workers"])
    default_val_workers = 0 if train_workers == 0 else max(1, train_workers // 2)
    val_workers = int(cfg["training"].get("val_workers", default_val_workers))
    persistent = cfg["training"].get("persistent_workers", False) and train_workers > 0
    prefetch = cfg["training"].get("prefetch_factor", 2) if train_workers > 0 else None

    endgame_fraction = cfg["training"].get("endgame_fraction", None)
    if endgame_fraction is not None:
        from pathlib import Path as _Path
        _prefix = str(_Path(cfg["data"]["train_file"]).with_suffix(""))
        _pc = np.load(_prefix + ".piece_count.npy", mmap_mode="r").astype(np.int32)
        _n_eg = int((_pc <= 16).sum())
        print(f"Stratified sampling: endgame_fraction={endgame_fraction:.2f}  "
              f"endgame={_n_eg:,} ({100*_n_eg/len(_pc):.1f}%)  "
              f"midgame={len(_pc)-_n_eg:,}")
        _sampler = StratifiedEndgameSampler(_pc, endgame_fraction)
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg["training"]["batch_size"],
            sampler=_sampler,
            num_workers=train_workers,
            pin_memory=cfg["training"]["pin_memory"],
            drop_last=False,
            persistent_workers=persistent,
            prefetch_factor=prefetch,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg["training"]["batch_size"],
            shuffle=True,
            num_workers=train_workers,
            pin_memory=cfg["training"]["pin_memory"],
            drop_last=False,
            persistent_workers=persistent,
            prefetch_factor=prefetch,
        )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=val_workers,
        pin_memory=cfg["training"]["pin_memory"],
        persistent_workers=persistent,
        prefetch_factor=prefetch,
    )

    mcfg = cfg["model"]
    if mcfg.get("dual_perspective", False):
        model = EvalNetDual(
            input_dim=mcfg["input_dim"],
            hidden_dim=mcfg["hidden_dim"],
            hidden2_dim=mcfg["hidden2_dim"],
            dropout=mcfg["dropout"],
            n_output_buckets=mcfg.get("n_output_buckets", 1),
        ).to(device)
    else:
        model = EvalNet(
            input_dim=mcfg["input_dim"],
            hidden_dim=mcfg["hidden_dim"],
            hidden2_dim=mcfg["hidden2_dim"],
            dropout=mcfg["dropout"],
            sparse_input=mcfg.get("sparse_input", False),
            n_output_buckets=mcfg.get("n_output_buckets", 1),
        ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    warmup_epochs = cfg["training"].get("warmup_epochs", 0)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, cfg["training"]["epochs"] - warmup_epochs),
        eta_min=cfg["training"]["lr"] / 100,
    )
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0 / warmup_epochs, total_iters=warmup_epochs
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
        )
    else:
        scheduler = cosine
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_val = float("inf")
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])

    if cfg["training"].get("compile", False):
        print("Compiling model with torch.compile...")
        model = torch.compile(model, mode=cfg["training"].get("compile_mode", "default"))
        print("Compilation graph ready (first batch will trigger kernel build).")
        if args.reset_best_val:
            best_val = float("inf")
            print(f"Resumed from {args.resume}  (best_val reset to inf)")
        else:
            best_val = ck.get("val_loss", float("inf"))
            print(f"Resumed from {args.resume}  (val_loss={best_val:.4f})")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"DataLoader workers: train={train_workers}, val={val_workers}")

    writer = SummaryWriter(log_dir=args.tb_logdir)
    print(f"TensorBoard logdir: {args.tb_logdir}")

    try:
        for epoch in range(1, cfg["training"]["epochs"] + 1):
            tr = train_epoch(model, train_loader, optimizer, scaler, device, cfg)
            va = eval_epoch(model, val_loader, device, cfg)
            scheduler.step()

            n_buckets = (model._orig_mod if hasattr(model, "_orig_mod") else model).n_output_buckets
            bucket_str = " ".join(
                f"b{b}={va[f'cp_mae_b{b}']:.2f}" for b in range(n_buckets)
            )
            eg_str = f"  val_cp_mae_endgame={va['cp_mae_endgame']:.2f}(n={va['cp_mae_endgame_n']})" \
                     if va.get("cp_mae_endgame_n", 0) > 0 else ""
            print(
                f"epoch={epoch} "
                f"train_loss={tr['loss']:.4f} val_loss={va['loss']:.4f} "
                f"train_cp_mae={tr['cp_mae']:.2f} val_cp_mae={va['cp_mae']:.2f} "
                f"val_cp_mae[{bucket_str}]{eg_str} "
                f"train_wdl_acc={tr['wdl_acc']:.3f} val_wdl_acc={va['wdl_acc']:.3f} "
                f"train_wdl_conf={tr['wdl_target_confidence']:.3f} val_wdl_conf={va['wdl_target_confidence']:.3f}"
            )

            writer.add_scalar("train/loss", tr["loss"], epoch)
            writer.add_scalar("train/cp_loss", tr["cp_loss"], epoch)
            writer.add_scalar("train/wdl_loss", tr["wdl_loss"], epoch)
            writer.add_scalar("train/cp_mae", tr["cp_mae"], epoch)
            writer.add_scalar("train/wdl_acc", tr["wdl_acc"], epoch)
            writer.add_scalar("train/wdl_target_confidence", tr["wdl_target_confidence"], epoch)
            for b in range(n_buckets):
                writer.add_scalar(f"train/cp_mae_b{b}", tr[f"cp_mae_b{b}"], epoch)

            writer.add_scalar("val/loss", va["loss"], epoch)
            writer.add_scalar("val/cp_loss", va["cp_loss"], epoch)
            writer.add_scalar("val/wdl_loss", va["wdl_loss"], epoch)
            writer.add_scalar("val/cp_mae", va["cp_mae"], epoch)
            writer.add_scalar("val/wdl_acc", va["wdl_acc"], epoch)
            writer.add_scalar("val/wdl_target_confidence", va["wdl_target_confidence"], epoch)
            for b in range(n_buckets):
                writer.add_scalar(f"val/cp_mae_b{b}", va[f"cp_mae_b{b}"], epoch)
            if va.get("cp_mae_endgame_n", 0) > 0:
                writer.add_scalar("val/cp_mae_endgame", va["cp_mae_endgame"], epoch)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)
            writer.flush()

            if va["loss"] < best_val:
                best_val = va["loss"]
                # Unwrap compiled model to save clean state dict without _orig_mod. prefix
                raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                torch.save(
                    {
                        "model_state": raw_model.state_dict(),
                        "config": cfg,
                        "val_loss": va["loss"],
                        "val_cp_mae": va["cp_mae"],
                        "epoch": epoch,
                    },
                    out_path,
                )
                print(f"saved checkpoint: {out_path}")
    finally:
        writer.close()


if __name__ == "__main__":
    main()

"""
main.py

Usage:
    python main.py
    python main.py --arch configs/arch_groupnorm.yaml
    python main.py --arch configs/arch.yaml --input configs/input.yaml --tl configs/tl.yaml
"""

import os
import csv
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from config     import load_cfg
from model      import build_model
from losses     import build_loss
from dataset    import discover_ids, precache, IntelligentPatchDataset, build_val_set
from train      import train_one_epoch
from validation import validate_patches, validate_fullvol_objectlevel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arch",  default="configs/arch.yaml")
    p.add_argument("--input", default="configs/input.yaml")
    p.add_argument("--tl",    default="configs/tl.yaml")
    return p.parse_args()


def run(cfg):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:   {device}")
    print(f"Run:      {cfg.run_name}  |  TL={cfg.TL}")
    print(f"Arch:     norm={cfg.norm}  residual={cfg.residual}  channels={cfg.channels}")
    print(f"Loss:     {cfg.loss}")

    # ---- IDs ----
    train_ids = discover_ids(cfg.HOLO_DIR_TRAIN)
    val_ids   = discover_ids(cfg.HOLO_DIR_VAL)
    print(f"Train: {len(train_ids)} volumes  |  Val: {len(val_ids)} volumes")

    cache_train = os.path.join(cfg.cache_root, "train")
    cache_val   = os.path.join(cfg.cache_root, "val")

    # ---- Pre-cache ----
    print("\n=== Pre-caching 3D volumes ===")
    precache(train_ids, cfg.HOLO_DIR_TRAIN, cfg.BIN_DIR_TRAIN, cache_train, cfg, device, "TRAIN")
    precache(val_ids,   cfg.HOLO_DIR_VAL,   cfg.BIN_DIR_VAL,   cache_val,   cfg, device, "VAL")

    # ---- Datasets ----
    ds_train = IntelligentPatchDataset(train_ids, cfg.HOLO_DIR_TRAIN, cfg.BIN_DIR_TRAIN, cache_train, cfg, device)
    ds_val   = IntelligentPatchDataset(val_ids,   cfg.HOLO_DIR_VAL,   cfg.BIN_DIR_VAL,   cache_val,   cfg, device)
    dl_train = DataLoader(
        ds_train, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"), drop_last=True,
    )

    # ---- Deterministic val set (built once, fixed for the entire run) ----
    print(f"\nBuilding deterministic val set ({cfg.val_n_patches} patches)...")
    val_set = build_val_set(ds_val, n=cfg.val_n_patches, seed=cfg.seed)
    print("Done.")

    # ---- Loss ----
    loss_fn = build_loss(cfg)

    # ---- Model ----
    model = build_model(cfg).to(device)

    if cfg.TL:
        if os.path.exists(cfg.pretrained_ckpt):
            ckpt = torch.load(cfg.pretrained_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            print(f"Loaded pretrained: {cfg.pretrained_ckpt}  (F1={ckpt.get('best_obj_F1', '?')})")
        else:
            print(f"[TL] Checkpoint not found: {cfg.pretrained_ckpt} — training from scratch")
    else:
        print("Training from scratch")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"UNet3D: {n_params:,} parameters\n")

    # ---- Optimiser / scheduler / scaler ----
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=8, factor=0.5
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # ---- Checkpoint dir ----
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    ckpt_out = cfg.ckpt_path

    # ---- CSV logger ----
    log_csv = os.path.join(cfg.checkpoint_dir, f"training_log_{cfg.run_name}.csv")
    csv_fields = [
        "epoch", "train_loss", "train_dice", "train_prec", "train_rec",
        "val_loss", "val_dice", "val_prec", "val_rec",
        "obj_f1", "obj_prec", "obj_rec", "n_pred_mean", "n_gt_mean",
    ]
    with open(log_csv, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=csv_fields).writeheader()

    # ---- Training loop ----
    best_f1    = -float("inf")
    best_dice  = 0.0
    no_improve = 0
    obj_metrics = {}

    print("=" * 70)
    print(f"TRAINING — {cfg.run_name}  loss={cfg.loss}  norm={cfg.norm}  residual={cfg.residual}  anisotropic={cfg.anisotropic}")
    print("=" * 70)

    for epoch in range(1, cfg.epochs + 1):

        train_metrics = train_one_epoch(
            model, dl_train, optimizer, scaler, loss_fn, cfg, device, epoch
        )

        # -- patch-level val every epoch (deterministic, honest loss) --
        patch_metrics = validate_patches(model, val_set, loss_fn, device)

        # -- full-vol object-level every 5 epochs --
        if epoch % 5 == 0 or epoch == cfg.epochs:
            obj_metrics = validate_fullvol_objectlevel(
                model=model, ids=val_ids, cache_dir=cache_val,
                cfg=cfg, device=device, max_images=30,
                thr=0.5, stride_xy=96, stride_z=48, match_radius_vox=10.0,
            )

        if obj_metrics:
            scheduler.step(obj_metrics["obj_f1"])

        log = (
            f"[{epoch:03d}/{cfg.epochs}] "
            f"train loss={train_metrics['train_loss']:.4f} dice={train_metrics['train_dice']:.3f} | "
            f"val loss={patch_metrics['val_loss']:.4f} dice={patch_metrics['val_dice']:.3f} "
            f"p={patch_metrics['val_prec']:.3f} r={patch_metrics['val_rec']:.3f}"
        )
        if obj_metrics:
            log += (
                f" | OBJ f1={obj_metrics['obj_f1']:.3f} "
                f"p={obj_metrics['obj_prec']:.3f} r={obj_metrics['obj_rec']:.3f} "
                f"(Pred~{obj_metrics['n_pred_mean']:.1f} GT~{obj_metrics['n_gt_mean']:.1f})"
            )
        print(log)

        # ---- Log to CSV ----
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["train_loss"],
            "train_dice": train_metrics["train_dice"],
            "train_prec": train_metrics["train_prec"],
            "train_rec":  train_metrics["train_rec"],
            "val_loss":   patch_metrics["val_loss"],
            "val_dice":   patch_metrics["val_dice"],
            "val_prec":   patch_metrics["val_prec"],
            "val_rec":    patch_metrics["val_rec"],
            "obj_f1":     obj_metrics.get("obj_f1", ""),
            "obj_prec":   obj_metrics.get("obj_prec", ""),
            "obj_rec":    obj_metrics.get("obj_rec", ""),
            "n_pred_mean":obj_metrics.get("n_pred_mean", ""),
            "n_gt_mean":  obj_metrics.get("n_gt_mean", ""),
        }
        with open(log_csv, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=csv_fields).writerow(row)

        # ---- Checkpoint on best obj_f1 ----
        if obj_metrics and obj_metrics["obj_f1"] > best_f1:
            best_f1   = obj_metrics["obj_f1"]
            best_dice = patch_metrics["val_dice"]
            no_improve = 0
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(),
                 "best_obj_F1": best_f1, "cfg": cfg.__dict__},
                ckpt_out,
            )
            print(f"saved: {ckpt_out}")
        else:
            no_improve += 1

        if no_improve >= cfg.patience:
            print(f"Early stopping at epoch {epoch}  (best F1={best_f1:.4f}  dice={best_dice:.3f})")
            break

    print(f"\nDone.  Best F1={best_f1:.4f}  Best dice={best_dice:.3f}")


if __name__ == "__main__":
    args = parse_args()
    cfg  = load_cfg(arch=args.arch, input=args.input, tl=args.tl)
    run(cfg)

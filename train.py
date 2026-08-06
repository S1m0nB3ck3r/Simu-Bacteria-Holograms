import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from losses import voxel_metrics


def train_one_epoch(
    model: nn.Module,
    dl_train: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    loss_fn,
    cfg,
    device: torch.device,
    epoch: int,
) -> dict:
    """
    Args:
        loss_fn: callable from build_loss(cfg)

    Returns dict: train_loss, train_dice, train_prec, train_rec
    """
    model.train()
    rl, rd, rp, rr = 0.0, 0.0, 0.0, 0.0

    pbar = tqdm(dl_train, desc=f"Epoch {epoch:03d}/{cfg.epochs}", leave=False)
    for vt, st in pbar:
        vt = vt.to(device, non_blocking=True)
        st = st.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(vt)
            loss   = loss_fn(logits, st)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        d, p, r = voxel_metrics(logits.detach(), st.detach())
        rl += loss.item(); rd += d; rp += p; rr += r
        step = pbar.n + 1
        pbar.set_postfix(
            loss=f"{rl/step:.4f}", dice=f"{rd/step:.3f}",
            p=f"{rp/step:.3f}", r=f"{rr/step:.3f}",
        )

    n = max(1, len(dl_train))
    return dict(
        train_loss=rl / n, train_dice=rd / n,
        train_prec=rp / n, train_rec =rr / n,
    )
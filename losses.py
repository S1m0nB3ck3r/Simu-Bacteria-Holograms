"""
Toggle via arch.yaml:
  loss: bce_dice    → 0.4 BCE (dynamic pos_weight) + 0.6 Dice
  loss: focal_dice  → Focal Loss + Dice
  loss: tversky     → Tversky Loss (beta > alpha penalises FN more)
"""

import torch
import torch.nn.functional as F
from typing import Callable


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------

def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p     = torch.sigmoid(logits).reshape(logits.size(0), -1)
    t     = target.reshape(target.size(0), -1)
    inter = (p * t).sum(1)
    return 1 - ((2 * inter + eps) / (p.sum(1) + t.sum(1) + eps)).mean()


def bce_dynamic(
    logits: torch.Tensor, target: torch.Tensor,
    max_pw: float = 200.0, eps: float = 1e-6,
) -> torch.Tensor:
    """BCE with pos_weight auto-computed from batch imbalance."""
    pos = target.sum()
    neg = target.numel() - pos
    pw  = (neg / (pos + eps)).clamp(max=max_pw)
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)


def focal_loss(
    logits: torch.Tensor, target: torch.Tensor,
    alpha: float = 0.25, gamma: float = 2.0,
) -> torch.Tensor:
    bce  = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p    = torch.sigmoid(logits)
    pt   = p * target + (1 - p) * (1 - target)
    w    = alpha * target + (1 - alpha) * (1 - target)
    loss = w * (1 - pt) ** gamma * bce
    return loss.mean()


def tversky_loss(
    logits: torch.Tensor, target: torch.Tensor,
    alpha: float = 0.3, beta: float = 0.7, eps: float = 1e-6,
) -> torch.Tensor:
    """
    Tversky loss: alpha weights FP, beta weights FN.
    beta > alpha → penalises missed bacteria more than false alarms.
    """
    p     = torch.sigmoid(logits).reshape(logits.size(0), -1)
    t     = target.reshape(target.size(0), -1)
    tp    = (p * t).sum(1)
    fp    = (p * (1 - t)).sum(1)
    fn    = ((1 - p) * t).sum(1)
    return 1 - ((tp + eps) / (tp + alpha * fp + beta * fn + eps)).mean()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_loss(cfg) -> Callable:
    """
    Returns a loss function  f(logits, target) → scalar tensor.
    Reads cfg.loss and the relevant weight parameters.
    """
    if cfg.loss == "bce_dice":
        bw, dw = cfg.bce_weight, cfg.dice_weight
        def _bce_dice(logits, target):
            return bw * bce_dynamic(logits, target) + dw * dice_loss(logits, target)
        return _bce_dice

    elif cfg.loss == "focal_dice":
        dw = cfg.dice_weight
        a, g = cfg.focal_alpha, cfg.focal_gamma
        def _focal_dice(logits, target):
            return 0.5 * bce_dynamic(logits, target) + focal_loss(logits, target, alpha=a, gamma=g) + dw * dice_loss(logits, target)
        return _focal_dice

    elif cfg.loss == "tversky":
        a, b = cfg.tversky_alpha, cfg.tversky_beta
        def _tversky(logits, target):
            return 0.5 * bce_dynamic(logits, target) + 0.5 * tversky_loss(logits, target, alpha=a, beta=b)
        return _tversky

    else:
        raise ValueError(f"Unknown loss: {cfg.loss}. Choose: bce_dice | focal_dice | tversky")


# ---------------------------------------------------------------------------
# Metrics (always the same regardless of loss choice)
# ---------------------------------------------------------------------------

@torch.no_grad()
def voxel_metrics(
    logits: torch.Tensor, target: torch.Tensor,
    thr: float = 0.5, eps: float = 1e-8,
) -> tuple:
    """Returns (dice, precision, recall) at threshold."""
    p  = (torch.sigmoid(logits) > thr).float()
    t  = (target > 0.5).float()
    tp = (p * t).sum().item()
    fp = (p * (1 - t)).sum().item()
    fn = ((1 - p) * t).sum().item()
    prec = tp / (tp + fp + eps)
    rec  = tp / (tp + fn + eps)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    return dice, prec, rec

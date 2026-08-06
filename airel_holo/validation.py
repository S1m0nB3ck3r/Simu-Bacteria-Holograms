"""
validation.py
"""

import os
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import scipy.ndimage as ndi

from losses import voxel_metrics


# ---------------------------------------------------------------------------
# Patch-level validation  (mirrors training — honest loss)
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate_patches(
    model: nn.Module,
    val_set: list,
    loss_fn,
    device: torch.device,
) -> Dict:
    """
    Evaluate on a fixed deterministic val_set (built once via build_val_set()).
    Mirrors training exactly: patch → model → logits → loss_fn.

    Args:
        val_set: list of (vt, st) from build_val_set()
        loss_fn: callable from build_loss(cfg)

    Returns dict: val_loss, val_dice, val_prec, val_rec
    """
    model.eval()
    losses, dices, precs, recs = [], [], [], []

    for vt, st in val_set:
        vt = vt[None].to(device)
        st = st[None].to(device)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(vt)
            loss   = loss_fn(logits, st)
        d, p, r = voxel_metrics(logits, st)
        losses.append(loss.item())
        dices.append(d)
        precs.append(p)
        recs.append(r)

    return dict(
        val_loss = float(np.mean(losses)),
        val_dice = float(np.mean(dices)),
        val_prec = float(np.mean(precs)),
        val_rec  = float(np.mean(recs)),
    )


# ---------------------------------------------------------------------------
# Full-volume object-level validation  (biological metric — no loss)
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate_fullvol_objectlevel(
    model: nn.Module,
    ids: List[int],
    cache_dir: str,
    cfg,
    device: torch.device,
    max_images: int = 10,
    seed: int = 123,
    thr: float = 0.5,
    stride_xy: int = 96,
    stride_z: int = 48,
    min_voxels: int = 10,
    match_radius_vox: float = 10.0,
) -> Dict:
    """
    Stitch full-volume predictions, threshold, count bacteria, match to GT.
    No loss computation — stitched probs are not honest logits.

    Args:
        min_voxels: minimum connected-component size to count as a detection
                    (filters fragmented blobs)

    Returns dict with keys:
        voxel_dice, voxel_prec, voxel_rec,
        obj_prec, obj_rec, obj_f1,
        n_pred_mean, n_gt_mean
    """
    model.eval()
    rng  = np.random.default_rng(seed + 999)
    pick = (
        ids if len(ids) <= max_images
        else rng.choice(ids, size=max_images, replace=False).tolist()
    )

    vd_sum, vp_sum, vr_sum = 0.0, 0.0, 0.0
    tp_sum, fp_sum, fn_sum = 0, 0, 0
    n_pred_sum, n_gt_sum   = 0, 0
    n_img = 0

    for idx in pick:
        vp = os.path.join(cache_dir, f"vol_{idx}.npy")
        sp = os.path.join(cache_dir, f"seg_{idx}.npy")
        if not (os.path.exists(vp) and os.path.exists(sp)):
            continue

        vol_hwz = np.load(vp).astype(np.float32)
        gt_hwz  = (np.load(sp) > 0).astype(np.uint8)

        _, pred_hwz = sliding_window_predict_full_volume(
            model, vol_hwz, cfg, device,
            stride_xy=stride_xy, stride_z=stride_z, thr=thr,
        )

        # voxel metrics — computed directly on binary mask vs GT (no loss)
        p_t  = torch.from_numpy(pred_hwz.astype(np.float32))
        gt_t = torch.from_numpy(gt_hwz.astype(np.float32))
        tp_v = (p_t * gt_t).sum().item()
        fp_v = (p_t * (1 - gt_t)).sum().item()
        fn_v = ((1 - p_t) * gt_t).sum().item()
        eps  = 1e-8
        vd_sum += (2 * tp_v) / (2 * tp_v + fp_v + fn_v + eps)
        vp_sum += tp_v / (tp_v + fp_v + eps)
        vr_sum += tp_v / (tp_v + fn_v + eps)

        # object-level metrics
        gt_xyz, n_gt = _centers_from_3d_mask(gt_hwz, min_voxels=0)
        pr_xyz, n_pr = _centers_from_3d_mask(pred_hwz, min_voxels=min_voxels)
        tp, fp, fn   = _match_counts_by_radius(gt_xyz, pr_xyz, radius=match_radius_vox)
        tp_sum += tp; fp_sum += fp; fn_sum += fn
        n_gt_sum += n_gt; n_pred_sum += n_pr
        n_img += 1

        torch.cuda.empty_cache()

    if n_img == 0:
        nan = float("nan")
        return dict(
            voxel_dice=nan, voxel_prec=nan, voxel_rec=nan,
            obj_prec=nan, obj_rec=nan, obj_f1=nan,
            n_pred_mean=nan, n_gt_mean=nan,
        )

    obj_prec = tp_sum / (tp_sum + fp_sum + 1e-8)
    obj_rec  = tp_sum / (tp_sum + fn_sum + 1e-8)
    obj_f1   = (2 * obj_prec * obj_rec) / (obj_prec + obj_rec + 1e-8)

    return dict(
        voxel_dice  = vd_sum / n_img,
        voxel_prec  = vp_sum / n_img,
        voxel_rec   = vr_sum / n_img,
        obj_prec    = obj_prec,
        obj_rec     = obj_rec,
        obj_f1      = obj_f1,
        n_pred_mean = n_pred_sum / n_img,
        n_gt_mean   = n_gt_sum   / n_img,
    )


# ---------------------------------------------------------------------------
# Sliding-window inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def sliding_window_predict_full_volume(
    model: nn.Module,
    vol_hwz: np.ndarray,
    cfg,
    device: torch.device,
    stride_xy: int = 96,
    stride_z: int = 16,
    thr: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stitch full-volume prediction from overlapping patches (average pooling).

    Args:
        vol_hwz: (H, W, Z) float32, already normalized

    Returns:
        prob_hwz: (H, W, Z) float32 probability map  [0, 1]
        pred_hwz: (H, W, Z) uint8  binary mask
    """
    model.eval()
    H, W, Z = vol_hwz.shape
    pxy, pz = cfg.patch_xy, cfg.patch_z

    prob_acc = np.zeros((Z, H, W), dtype=np.float32)
    w_acc    = np.zeros((Z, H, W), dtype=np.float32)

    ys = list(range(0, max(1, H - pxy + 1), stride_xy))
    xs = list(range(0, max(1, W - pxy + 1), stride_xy))
    zs = list(range(0, max(1, Z - pz  + 1), stride_z))
    if ys[-1] != H - pxy: ys.append(H - pxy)
    if xs[-1] != W - pxy: xs.append(W - pxy)
    if zs[-1] != Z - pz:  zs.append(Z - pz)

    for y0 in ys:
        for x0 in xs:
            for z0 in zs:
                patch = vol_hwz[y0:y0+pxy, x0:x0+pxy, z0:z0+pz]
                vt    = torch.from_numpy(patch.transpose(2, 0, 1))[None, None].float().to(device)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    prob = torch.sigmoid(model(vt))[0, 0].float().detach().cpu().numpy()
                prob_acc[z0:z0+pz, y0:y0+pxy, x0:x0+pxy] += prob
                w_acc   [z0:z0+pz, y0:y0+pxy, x0:x0+pxy] += 1.0

    prob_acc /= np.maximum(w_acc, 1e-6)
    prob_hwz  = prob_acc.transpose(1, 2, 0)
    pred_hwz  = (prob_hwz > thr).astype(np.uint8)
    return prob_hwz, pred_hwz


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _centers_from_3d_mask(
    mask_hwz: np.ndarray,
    min_voxels: int = 10,
) -> Tuple[np.ndarray, int]:
    """
    Connected-component centers from a binary mask.

    Args:
        mask_hwz:   (H, W, Z) uint8/bool
        min_voxels: discard components smaller than this (avoids fragment noise)

    Returns:
        xyz: (N, 3) float32 array of (X, Y, Z) centers
        n:   number of valid components
    """
    lbl, n = ndi.label(mask_hwz > 0)
    if n == 0:
        return np.zeros((0, 3), np.float32), 0

    centers = []
    for i in range(1, n + 1):
        comp = lbl == i
        if comp.sum() < min_voxels:
            continue
        cy, cx, cz = ndi.center_of_mass(comp)
        centers.append([cx, cy, cz])   # store as (X, Y, Z)

    if not centers:
        return np.zeros((0, 3), np.float32), 0
    return np.array(centers, dtype=np.float32), len(centers)


def _match_counts_by_radius(
    gt_xyz: np.ndarray,
    pr_xyz: np.ndarray,
    radius: float = 10.0,
) -> Tuple[int, int, int]:
    """
    Greedy nearest-neighbour matching.
    A prediction is TP if within `radius` voxels of an unmatched GT center.

    Returns: (tp, fp, fn)
    """
    if len(gt_xyz) == 0 and len(pr_xyz) == 0:
        return 0, 0, 0
    if len(gt_xyz) == 0:
        return 0, len(pr_xyz), 0
    if len(pr_xyz) == 0:
        return 0, 0, len(gt_xyz)

    used = np.zeros(len(gt_xyz), dtype=bool)
    tp = 0
    for p in pr_xyz:
        d2 = np.sum((gt_xyz - p[None, :]) ** 2, axis=1)
        j  = int(np.argmin(d2))
        if (not used[j]) and (d2[j] <= radius ** 2):
            used[j] = True
            tp += 1
    return tp, len(pr_xyz) - tp, len(gt_xyz) - tp

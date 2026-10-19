# -*- coding: utf-8 -*-

"""
Filename: detection.py

Description:
Du volume de sortie du réseau à une liste de bactéries, et comparaison à la vérité terrain :
  - infer_volume  : inférence sur le volume entier par tuiles XY avec halo (le halo est jeté :
                    aucun effet de bord entre tuiles), fp16 ou fp32 ;
  - peaks_from_heat : maxima locaux de la carte de chaleur (boîte 3 × 5 × 5 voxels = 3 × 1,75 ×
                    1,75 µm) au-dessus d'un seuil, affinés par barycentre 3 × 3 × 3 ;
  - objects_from_mask : composantes connexes 26-voisines du masque seuillé, barycentre pondéré par la
                    probabilité, composantes de moins de min_voxels ignorées ;
  - match         : appariement hongrois, distance normalisée √((dxy/tol_xy)² + (dz/tol_z)²) ≤ 1
                    (tolérances du rapport DENSE : 3 µm en XY, 5 µm en Z) ;
  - classical_detect : méthode classique de référence (SEGMENTATION_REPORT) — volume SUM_OF_VARIANCE
                    pleine résolution (0,175 µm, pas Z 0,5 µm), maxima locaux au-dessus du seuil
                    local K = 15, boîte 2,1 × 5,3 µm.
Positions en µm, repère HOLOTRACKER (x = colonne, y = ligne).

Author: Simon BECKER
Date: 2026-10-17

License:
GNU General Public License v3.0
"""

import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

DL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(DL_DIR)


# =============================================================================
# INFÉRENCE
# =============================================================================

@torch.no_grad()
def infer_volume(model, V, tile=None, halo=32, amp=True, multiple=8):
    """
    V : (C, Z, Y, X) sur le GPU. Retourne les sorties (C_out, Z, Y, X) float32 (logits en canal 0).
    tile=None : volume entier si Z·Y·X ≤ 15 M voxels (512 × 512 × 56), sinon tuiles de 256 avec halo de 32,
    réduites à 128 si Z·(tuile + 2·halo)² dépasse 22 M voxels (fp16) ou 11 M (fp32), p. ex. 208 plans en fp32
    (mesuré, base 16, Quadro RTX 3000 : 104 plans en une passe déborde des 6 Go, 2 033 ms contre 597
    en tuiles ; 56 plans en une passe 255 ms contre 326 en tuiles).
    """
    C, Z, Y, X = V.shape
    if tile is None:
        # plus grande tuile qui tient dans le budget (halo compris) ; fp32 : activations deux fois plus lourdes
        budget = 22e6 if amp else 11e6
        tile = max(Y, X) if Z * Y * X <= 15e6 * (1 if amp else 0.5) else next(
            (t for t in (256, 128, 64) if Z * (t + 2 * halo) ** 2 <= budget), 64)
    if tile >= max(Y, X):
        halo = 0
    out = None
    pz = (-Z) % multiple
    for y0 in range(0, Y, tile):
        for x0 in range(0, X, tile):
            y1, x1 = min(y0 + tile, Y), min(x0 + tile, X)
            ya, yb = max(y0 - halo, 0), min(y1 + halo, Y)
            xa, xb = max(x0 - halo, 0), min(x1 + halo, X)
            crop = V[:, :, ya:yb, xa:xb]
            py, px = (-(yb - ya)) % multiple, (-(xb - xa)) % multiple
            crop = F.pad(crop, (0, px, 0, py, 0, pz))[None]
            if model_is_channels_last(model):
                crop = crop.contiguous(memory_format=torch.channels_last_3d)
            with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                o = model(crop)[0].float()
            if out is None:
                out = torch.empty((o.shape[0], Z, Y, X), device=V.device, dtype=torch.float32)
            out[:, :, y0:y1, x0:x1] = o[:, :Z, y0 - ya:y0 - ya + (y1 - y0), x0 - xa:x0 - xa + (x1 - x0)]
    return out


def model_is_channels_last(model):
    w = next(model.parameters())
    return w.dim() == 5 and w.is_contiguous(memory_format=torch.channels_last_3d) and not w.is_contiguous()


# =============================================================================
# POST-TRAITEMENT
# =============================================================================

def peaks_from_heat(logits, thr, step_zyx_um, z0_um, box=(3, 5, 5), return_voxels=False):
    """Retourne (n, 3) x, y, z (µm) et scores, par ordre de score décroissant (et les voxels des pics)."""
    p = torch.sigmoid(logits)
    mp = F.max_pool3d(p[None, None], box, stride=1, padding=tuple(b // 2 for b in box))[0, 0]
    iz, iy, ix = torch.nonzero((p == mp) & (p > thr), as_tuple=True)
    if iz.numel() == 0:
        return (np.zeros((0, 3)), np.zeros(0), np.zeros((0, 3), int)) if return_voxels else (np.zeros((0, 3)), np.zeros(0))
    Z, Y, X = p.shape
    acc = torch.zeros((iz.numel(), 4), device=p.device)
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                z, y, x = (iz + dz).clamp(0, Z - 1), (iy + dy).clamp(0, Y - 1), (ix + dx).clamp(0, X - 1)
                w = p[z, y, x]
                acc[:, 0] += w * z
                acc[:, 1] += w * y
                acc[:, 2] += w * x
                acc[:, 3] += w
    c = (acc[:, :3] / acc[:, 3:4]).cpu().numpy()
    pos = np.stack([c[:, 2] * step_zyx_um[2], c[:, 1] * step_zyx_um[1], z0_um + c[:, 0] * step_zyx_um[0]], axis=1)
    score = p[iz, iy, ix].cpu().numpy()
    o = np.argsort(-score)
    if return_voxels:
        vox = torch.stack([iz, iy, ix], dim=1).cpu().numpy()[o]
        return pos[o], score[o], vox
    return pos[o], score[o]


def objects_from_mask(logits, thr, step_zyx_um, z0_um, min_voxels=2, return_axes=False):
    """
    Composantes connexes (26-voisinage) de sigmoïde > thr ; barycentre pondéré par la probabilité ;
    axe = vecteur propre principal de la covariance pondérée des voxels (en µm), si return_axes.
    """
    import cupy as cp
    from cupyx.scipy import ndimage as cpnd
    p = torch.sigmoid(logits)
    binary = (p > thr)
    empty = (np.zeros((0, 3)), np.zeros(0), np.zeros((0, 3))) if return_axes else (np.zeros((0, 3)), np.zeros(0))
    if not bool(binary.any()):
        return empty
    lab, n = cpnd.label(cp.from_dlpack(binary.contiguous()), structure=cp.ones((3, 3, 3), dtype=bool))
    lab = torch.from_dlpack(lab).long()
    sel = lab > 0
    idx = lab[sel] - 1
    iz, iy, ix = torch.nonzero(sel, as_tuple=True)
    w = p[sel]
    # coordonnées en µm (x, y, z) : la covariance doit être isotrope dans l'espace physique
    cx, cy, cz = ix.float() * step_zyx_um[2], iy.float() * step_zyx_um[1], iz.float() * step_zyx_um[0]
    acc = torch.zeros((int(n), 11), device=p.device)
    acc.index_add_(0, idx, torch.stack([w * cx, w * cy, w * cz, w, torch.ones_like(w),
                                        w * cx * cx, w * cy * cy, w * cz * cz, w * cx * cy, w * cx * cz, w * cy * cz],
                                       dim=1))
    keep = acc[:, 4] >= min_voxels
    acc = acc[keep].double()
    if acc.shape[0] == 0:
        return empty
    m = acc[:, :3] / acc[:, 3:4]
    pos = m.cpu().numpy().copy()
    pos[:, 2] += z0_um
    score = (acc[:, 3] / acc[:, 4]).cpu().numpy()     # probabilité moyenne de la composante
    o = np.argsort(-score)
    if not return_axes:
        return pos[o], score[o]
    W = acc[:, 3]
    cov = torch.stack([acc[:, 5] / W - m[:, 0] ** 2, acc[:, 6] / W - m[:, 1] ** 2, acc[:, 7] / W - m[:, 2] ** 2,
                       acc[:, 8] / W - m[:, 0] * m[:, 1], acc[:, 9] / W - m[:, 0] * m[:, 2],
                       acc[:, 10] / W - m[:, 1] * m[:, 2]], dim=1).cpu().numpy()
    import holo_physics as hp
    axes = hp.axis_from_tensor(cov)
    return pos[o], score[o], axes[o]


# =============================================================================
# APPARIEMENT ET MÉTRIQUES
# =============================================================================

def match(det, gt, tol_xy=3.0, tol_z=5.0):
    """det (n, 3), gt (m, 3) -> paires (i_det, j_gt) d'appariement hongrois."""
    if len(det) == 0 or len(gt) == 0:
        return np.zeros(0, int), np.zeros(0, int)
    dxy = np.hypot(det[:, None, 0] - gt[None, :, 0], det[:, None, 1] - gt[None, :, 1])
    dz = np.abs(det[:, None, 2] - gt[None, :, 2])
    cost = np.sqrt((dxy / tol_xy) ** 2 + (dz / tol_z) ** 2)
    r, c = linear_sum_assignment(np.where(cost <= 1.0, cost, 1e6))
    ok = cost[r, c] <= 1.0
    return r[ok], c[ok]


def evaluate(det, gt_bacts, tol_xy=3.0, tol_z=5.0, lying_phi=60.0, det_axes=None):
    """
    Compteurs et erreurs pour un hologramme ; gt_bacts (m, 5) x, y, z, θ, φ.
    'hit' : bactérie détectée (m,), 'err' : erreurs de position des appariées, 'ang' : erreur angulaire
    de l'axe (degrés, insensible au signe) des appariées si det_axes est fourni.
    """
    gt = gt_bacts[:, :3]
    r, c = match(det, gt, tol_xy, tol_z)
    lying = gt_bacts[:, 4] > lying_phi
    err = det[r] - gt[c] if len(r) else np.zeros((0, 3))
    hit = np.zeros(len(gt), bool)
    hit[c] = True
    out = dict(tp=len(r), n_det=len(det), n_gt=len(gt), tp_lying=int(lying[c].sum()), n_lying=int(lying.sum()),
               err=err, hit=hit, gt_idx=c, det_idx=r)
    if det_axes is not None and len(r):
        import holo_physics as hp
        a_gt = hp.rod_axes(gt_bacts[c, 3], gt_bacts[c, 4])
        a_det = det_axes[r] / np.linalg.norm(det_axes[r], axis=1, keepdims=True)
        out['ang'] = np.degrees(np.arccos(np.clip(np.abs((a_gt * a_det).sum(axis=1)), 0, 1)))
    return out


def summarize(rows):
    tp = sum(r['tp'] for r in rows)
    nd = sum(r['n_det'] for r in rows)
    ng = sum(r['n_gt'] for r in rows)
    tl = sum(r['tp_lying'] for r in rows)
    nl = sum(r['n_lying'] for r in rows)
    err = np.concatenate([r['err'] for r in rows]) if rows else np.zeros((0, 3))
    rec = tp / ng if ng else float('nan')
    prec = tp / nd if nd else float('nan')
    f1 = 2 * rec * prec / (rec + prec) if (rec + prec) > 0 else 0.0
    s = dict(recall=rec, precision=prec, f1=f1, recall_lying=tl / nl if nl else float('nan'), tp=tp, n_det=nd, n_gt=ng)
    angs = [r['ang'] for r in rows if 'ang' in r]
    if angs:
        a = np.concatenate(angs)
        s.update(angle_median_deg=float(np.median(a)), angle_p90_deg=float(np.percentile(a, 90)))
    if len(err):
        s.update(sigma_x=float(np.std(err[:, 0])), sigma_y=float(np.std(err[:, 1])), sigma_z=float(np.std(err[:, 2])),
                 bias_z=float(np.mean(err[:, 2])), mae_xy=float(np.mean(np.hypot(err[:, 0], err[:, 1]))),
                 mae_z=float(np.mean(np.abs(err[:, 2]))))
    return s


# =============================================================================
# MÉTHODE CLASSIQUE DE RÉFÉRENCE
# =============================================================================

class ClassicalDetector:
    """
    Chaîne du SEGMENTATION_REPORT (meilleure méthode classique en milieu dense) : propagation pleine
    résolution (band-pass 15–250), volume SUM_OF_VARIANCE (noyau 11), maxima locaux au-dessus du
    seuil local K = 15 (mailles 25 × 64 × 64), boîte 2,1 × 5,3 µm, position par parabole.
    """

    def __init__(self, n=1024, dx_um=0.175, wavelength_um=0.66, index_medium=1.33, z0_um=0.0, step_um=0.5,
                 nb_planes=200, high_pass=15, low_pass=250, sum_size=11, k_local=15.0):
        import cupy as cp
        sys.path.insert(0, REPO_DIR)
        sys.path.insert(0, os.path.join(REPO_DIR, 'localisation_pipeline'))
        self.cp = cp
        self.p = dict(n=n, dx_um=dx_um, lam_m=wavelength_um * 1e-6 / index_medium, z0_um=z0_um, step_um=step_um,
                      nb=nb_planes, hp=high_pass, lp=low_pass, sum_size=sum_size, k=k_local)
        self.d_vol = cp.zeros((nb_planes, n, n), dtype=cp.float32)
        self.d_focus = cp.zeros((nb_planes, n, n), dtype=cp.float32)

    def __call__(self, h, timings=None):
        cp = self.cp
        from libs import propagation as propag
        from libs import focus as focus_lib
        from libs.focus import Focus_type
        from pipeline_holotracker_locate_local_maxima import detect_local_maxima
        p = self.p
        t0 = time.perf_counter()
        d_h = cp.asarray(np.asarray(h, dtype=np.float32) + 1.0).astype(cp.complex64)
        buf = [cp.zeros((p['n'], p['n']), cp.complex64) for _ in range(3)]
        propag.volume_propag_angular_spectrum_to_module(d_h, buf[0], buf[1], buf[2], self.d_vol, p['lam_m'], 1.0,
                                                        p['dx_um'] * 1e-6, p['n'], p['n'], p['z0_um'] * 1e-6,
                                                        p['step_um'] * 1e-6, p['nb'], p['hp'], p['lp'])
        cp.cuda.Device().synchronize()
        t1 = time.perf_counter()
        focus_lib.focus(self.d_vol, self.d_focus, p['sum_size'], Focus_type.SUM_OF_VARIANCE)
        cp.cuda.Device().synchronize()
        t2 = time.perf_counter()
        df, _, _ = detect_local_maxima(self.d_focus, p['dx_um'], p['step_um'], p['z0_um'], k_local=p['k'],
                                       mesh_z=25, mesh_xy=64, footprint_xy_um=2.1, footprint_z_um=5.3)
        cp.cuda.Device().synchronize()
        t3 = time.perf_counter()
        if timings is not None:
            timings.append(dict(propagation=t1 - t0, focus=t2 - t1, detection=t3 - t2, total=t3 - t0))
        det = df[['baryX', 'baryY', 'baryZ']].to_numpy(float)
        return det, df['snr'].to_numpy(float)

# -*- coding: utf-8 -*-

"""
Filename: seg_stage2.py

Description:
Étage 2 du U-Net 3D : segmentation d'instance fine autour de chaque détection de l'étage 1.

Grille fine = grille du masque du simulateur : pixel caméra dx (0,175 µm) en x et y, pas BPM (0,25 µm) en
z ; voxel (k, ligne, colonne) <-> (z = k · dz, y = ligne · dx, x = colonne · dx). Le bâtonnet (0,6 µm
d'épaisseur) couvre ~3,4 voxels en x, y et ~2,4 en z.

Entrée d'une fenêtre (32 × 40 × 40 voxels = 8 × 7 × 7 µm) : le champ complexe (Re, Im) refocalisé
EXACTEMENT sur la grille fine, à partir du volume grossier de l'étage 1 (pixel 0,35 µm, plans à 1 µm),
sans repropager l'hologramme entier (LocalRefocus) :
  - recadrage 64 × 64 (22,4 µm) du plan grossier le plus proche, apodisé sur les bords ;
  - propagation par spectre angulaire de Δz = z_fin − z_grossier (|Δz| ≤ 4,5 µm : la lumière s'étale
    d'au plus ~4,5 µm, la zone utile reste à 7,7 µm des bords) ;
  - suréchantillonnage ×2 par bourrage du spectre : le volume grossier contient toute la bande utile
    (rayon ≤ 250 px de Fourier du capteur < 256), le champ à 0,175 µm est donc exact.
Sortie : 3 canaux, « la bactérie de la détection » (self), « les autres bactéries » (others) — cibles =
fractions occupées du masque d'instance du simulateur (generate_dataset.py --save_masks ; un voxel peut
appartenir aux deux canaux quand deux bactéries se croisent) — et la carte fine des centres de TOUTES les
bactéries de la fenêtre (gaussienne σ 0,25 µm en XY, 0,5 µm en Z, 1 au voxel le plus proche du centre ;
loss focale de CenterNet).
Inférence en deux tours (segment_instances) : fenêtres sur les détections de l'étage 1 -> masque self et
centres vus dans chaque fenêtre ; les centres qui ne correspondent à aucune instance trouvée (bactéries
fusionnées ou manquées par l'étage 1) deviennent de nouvelles fenêtres ; doublons éliminés par recouvrement
des masques. Une fenêtre peut donc faire naître plusieurs instances.

Attribution d'une fenêtre : la bactérie la plus proche du centre (distance normalisée par 1 µm en XY et
2 µm en Z) si elle est à moins de 1 ; sinon aucune (fausse détection : self vide).

Mesures d'un masque (mask_measures), sans modèle de forme : barycentre, axe principal (ACP), longueur =
étendue des voxels le long de l'axe + empreinte d'un voxel projetée sur l'axe, épaisseur = 4·√λ⊥
(λ⊥ : moyenne des deux petites valeurs propres ; exact pour un cylindre plein), volume.

Author: Simon BECKER
Date: 2026-10-18

License:
GNU General Public License v3.0
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

FINE = dict(nz=32, nxy=40, dz_um=0.25)
TOL = dict(xy=1.0, z=2.0)            # attribution d'une fenêtre à une bactérie
HEAT = dict(sigma_xy_um=0.25, sigma_z_um=0.5)
INFER = dict(heat_thr=0.3, nms=(5, 5, 5), border=2, same_xy_um=0.5, same_z_um=1.0, vmin=20, dup_iou=0.5)


# =============================================================================
# REFOCALISATION LOCALE
# =============================================================================

class LocalRefocus:
    """
    Fenêtres fines (B, 2, nz, nxy, nxy) autour de points, à partir du volume grossier V (2, Z, m, m) de
    holo_physics.Reconstructor (même échelle : unités de bruit). La grille fine est alignée sur la grille
    du masque du simulateur : origine (k0, r0, c0) entière, voxel (k, r, c) <-> (k·dz, r·dx, c·dx).
    """

    def __init__(self, rec, nz=FINE['nz'], nxy=FINE['nxy'], dz_um=FINE['dz_um'], crop=64, chunk=32):
        if rec.decim != 2:
            raise ValueError("LocalRefocus suppose un volume grossier décimé ×2")
        self.rec, self.nz, self.nxy, self.dz, self.crop, self.chunk = rec, nz, nxy, dz_um, crop, chunk
        self.dx = rec.dx_um                       # grille fine
        self.st = rec.dx_um * rec.decim           # grille grossière
        dev = rec.device
        f = torch.fft.fftfreq(crop, d=self.st, device=dev)
        fy, fx = torch.meshgrid(f, f, indexing='ij')
        self.kz = torch.sqrt(torch.clamp(1.0 / rec.lam_um ** 2 - fx ** 2 - fy ** 2, min=0.0))
        n = torch.arange(crop, device=dev, dtype=torch.float32)
        edge = crop // 8                          # apodisation en cosinus sur 8 px (2,8 µm) de chaque bord
        w = torch.clamp(torch.minimum(n + 0.5, crop - 0.5 - n) / edge, max=1.0)
        w = 0.5 - 0.5 * torch.cos(math.pi * w)
        self.window = (w[:, None] * w[None, :]).to(torch.complex64)
        self.z_coarse = rec.z_um                  # (Z,)

    def origins(self, pts_um):
        """Origine (k0, r0, c0) de la fenêtre fine de chaque point (x, y, z en µm)."""
        p = np.asarray(pts_um, dtype=np.float64)
        c0 = np.rint(p[:, 0] / self.dx).astype(np.int64) - self.nxy // 2
        r0 = np.rint(p[:, 1] / self.dx).astype(np.int64) - self.nxy // 2
        k0 = np.rint(p[:, 2] / self.dz).astype(np.int64) - self.nz // 2
        return np.stack([k0, r0, c0], axis=1)

    @torch.no_grad()
    def __call__(self, V, pts_um):
        org = self.origins(pts_um)
        out = torch.empty((len(org), 2, self.nz, self.nxy, self.nxy), device=V.device, dtype=torch.float32)
        for b0 in range(0, len(org), self.chunk):
            out[b0:b0 + self.chunk] = self._chunk(V, org[b0:b0 + self.chunk])
        return out, org

    def _chunk(self, V, org):
        dev, B, cr, up = V.device, len(org), self.crop, 2
        Z, m = V.shape[1], V.shape[2]
        zc = self.z_coarse
        # plan grossier le plus proche du centre de la fenêtre
        zmid = torch.as_tensor((org[:, 0] + self.nz / 2) * self.dz, device=dev, dtype=torch.float32)
        kc = torch.argmin(torch.abs(zmid[:, None] - zc[None]), dim=1)                      # (B,)
        # recadrage grossier : la colonne fine c <-> colonne grossière c / 2 ; centre de la fenêtre fine
        cy = (org[:, 1] + self.nxy // 2) // 2
        cx = (org[:, 2] + self.nxy // 2) // 2
        j0y = torch.as_tensor(cy - cr // 2, device=dev)
        j0x = torch.as_tensor(cx - cr // 2, device=dev)
        ar = torch.arange(cr, device=dev)
        iy, ix = j0y[:, None] + ar[None], j0x[:, None] + ar[None]                          # (B, cr)
        vy, vx = (iy >= 0) & (iy < m), (ix >= 0) & (ix < m)
        iy, ix = iy.clamp(0, m - 1), ix.clamp(0, m - 1)
        kk = kc[:, None, None].expand(B, cr, cr)
        re = V[0][kk, iy[:, :, None].expand(B, cr, cr), ix[:, None, :].expand(B, cr, cr)]
        im = V[1][kk, iy[:, :, None].expand(B, cr, cr), ix[:, None, :].expand(B, cr, cr)]
        valid = (vy[:, :, None] & vx[:, None, :]).to(torch.float32)
        E = torch.complex(re * valid, im * valid) * self.window
        Fe = torch.fft.fft2(E)                                                               # (B, cr, cr)
        # propagation locale vers chaque plan fin
        zf = torch.as_tensor(org[:, 0], device=dev, dtype=torch.float32)[:, None] + torch.arange(self.nz, device=dev)
        dz = zf * self.dz - zc[kc][:, None]                                                  # (B, nz)
        G = Fe[:, None] * torch.exp((2j * math.pi) * dz[:, :, None, None] * self.kz[None, None])
        # suréchantillonnage ×2 : bourrage du spectre centré
        G = torch.fft.fftshift(G, dim=(-2, -1))
        p = cr // 2
        G = F.pad(torch.view_as_real(G), (0, 0, p, p, p, p))
        G = torch.view_as_complex(G.contiguous())
        e = torch.fft.ifft2(torch.fft.ifftshift(G, dim=(-2, -1))) * (up * up)              # (B, nz, 2cr, 2cr)
        # grille fine du recadrage : colonne fine 2·j0 + q ; fenêtre fine à partir de org
        qy = torch.as_tensor(org[:, 1], device=dev) - 2 * j0y
        qx = torch.as_tensor(org[:, 2], device=dev) - 2 * j0x
        a = torch.arange(self.nxy, device=dev)
        ry, rx = qy[:, None] + a[None], qx[:, None] + a[None]                                # (B, nxy)
        bi = torch.arange(B, device=dev)[:, None, None, None]
        zi = torch.arange(self.nz, device=dev)[None, :, None, None]
        sel = e[bi, zi, ry[:, None, :, None], rx[:, None, None, :]]                          # (B, nz, nxy, nxy)
        return torch.stack([sel.real, sel.imag], dim=1)


# =============================================================================
# INSTANCES DU SIMULATEUR ET CIBLES
# =============================================================================

def load_instances(item):
    """Liste des masques d'instance du simulateur : (origine (k, r, c), fractions float32 (K, R, C))."""
    lev = float(item['meta']['mask_grid']['frac_levels'])
    org, shp, data = item['mask_origin'], item['mask_shape'], item['mask_data']
    out, o = [], 0
    for (k, r, c), (K, R, C) in zip(org, shp):
        n = int(K) * int(R) * int(C)
        out.append(((int(k), int(r), int(c)), data[o:o + n].reshape(int(K), int(R), int(C)).astype(np.float32) / lev))
        o += n
    return out


def paste(dst, dst_org, src, src_org):
    """Copie (maximum) de src dans dst sur leur intersection ; origines (k, r, c) dans la même grille."""
    lo = [max(a, b) for a, b in zip(dst_org, src_org)]
    hi = [min(a + s, b + t) for a, s, b, t in zip(dst_org, dst.shape, src_org, src.shape)]
    if any(h <= l for l, h in zip(lo, hi)):
        return
    d = tuple(slice(l - o, h - o) for l, h, o in zip(lo, hi, dst_org))
    s = tuple(slice(l - o, h - o) for l, h, o in zip(lo, hi, src_org))
    np.maximum(dst[d], src[s], out=dst[d])


def assign(centers_um, pts_um, tol=TOL):
    """Indice de la bactérie attribuée à chaque fenêtre (la plus proche, distance normalisée < 1), ou −1."""
    if len(centers_um) == 0:
        return np.full(len(pts_um), -1)
    s = np.array([1 / tol['xy'], 1 / tol['xy'], 1 / tol['z']])
    d, j = cKDTree(centers_um[:, :3] * s).query(np.asarray(pts_um)[:, :3] * s)
    return np.where(d < 1.0, j, -1)


class CropTargets:
    """Cibles (B, 2, nz, nxy, nxy) d'un hologramme : self = instance attribuée, others = les autres."""

    def __init__(self, item, nz=FINE['nz'], nxy=FINE['nxy']):
        self.inst = load_instances(item)
        self.centers = item['bacts'][:, :3].astype(np.float64)
        self.tree = cKDTree(self.centers) if len(self.centers) else None
        self.shape = (nz, nxy, nxy)
        g = item['meta']['mask_grid']
        self.reach = math.sqrt((nz * g['dz_um'] / 2) ** 2 + 2 * (nxy * g['dx_um'] / 2) ** 2) + 2.0

        self.dx, self.dz = g['dx_um'], g['dz_um']

    def __call__(self, org, owner, pts_um, heat=True):
        B = len(org)
        nz, ny, nx = self.shape
        t = np.zeros((B, 3 if heat else 2) + self.shape, np.float32)
        for b in range(B):
            near = self.tree.query_ball_point(np.asarray(pts_um[b])[:3], self.reach) if self.tree is not None else []
            if heat:
                zz = (org[b, 0] + np.arange(nz)) * self.dz
                yy = (org[b, 1] + np.arange(ny)) * self.dx
                xx = (org[b, 2] + np.arange(nx)) * self.dx
            for j in near:
                o, m = self.inst[j]
                paste(t[b, 0 if j == owner[b] else 1], tuple(org[b]), m, o)
                if heat:
                    x, y, z = self.centers[j]
                    g = (np.exp(-(zz - z) ** 2 / (2 * HEAT['sigma_z_um'] ** 2))[:, None, None]
                         * np.exp(-(yy - y) ** 2 / (2 * HEAT['sigma_xy_um'] ** 2))[None, :, None]
                         * np.exp(-(xx - x) ** 2 / (2 * HEAT['sigma_xy_um'] ** 2))[None, None, :])
                    np.maximum(t[b, 2], g, out=t[b, 2])
                    k = int(round(z / self.dz)) - org[b, 0]
                    r = int(round(y / self.dx)) - org[b, 1]
                    c = int(round(x / self.dx)) - org[b, 2]
                    if 0 <= k < nz and 0 <= r < ny and 0 <= c < nx:
                        t[b, 2, k, r, c] = 1.0
        return t

    def full_volume_voxels(self, j, thr=0.5):
        """Nombre de voxels (fraction ≥ thr) de l'instance j entière (IoU exacte même si elle déborde)."""
        return int((self.inst[j][1] >= thr).sum())


# =============================================================================
# LOSS
# =============================================================================

def seg_loss(logits, target, w_others=0.5, w_heat=1.0):
    """
    Canaux 0–1 : BCE (cibles = fractions) + Dice doux par fenêtre (Dice de self seulement si la cible est non
    vide) ; canal 2 : focale de CenterNet sur la carte des centres.
    """
    from losses import centernet
    lg, tg = logits[:, :2], target[:, :2]
    bce = F.binary_cross_entropy_with_logits(lg, tg, reduction='none').mean(dim=(2, 3, 4))   # (B, 2)
    p = torch.sigmoid(lg)
    inter = (p * tg).sum(dim=(2, 3, 4))
    den = p.sum(dim=(2, 3, 4)) + tg.sum(dim=(2, 3, 4))
    dice = 1 - (2 * inter + 1.0) / (den + 1.0)
    has = (tg.sum(dim=(2, 3, 4)) > 0).float()
    dice = (dice * has).sum(dim=0) / has.sum(dim=0).clamp(min=1)
    w = torch.tensor([1.0, w_others], device=logits.device)
    loss = (bce.mean(dim=0) * w).sum() + (dice * w).sum() * 0.5
    if logits.shape[1] > 2:
        loss = loss + w_heat * centernet(logits[:, 2:3], target[:, 2:3])
    return loss


# =============================================================================
# MESURES
# =============================================================================

def mask_measures(mask, org, dx_um, dz_um):
    """
    mask : (nz, ny, nx) bool ; org : (k0, r0, c0). Retourne dict ou None si moins de 4 voxels :
    centre (x, y, z µm), axe (x, y, z), longueur, épaisseur, volume (µm³), n voxels.
    """
    k, r, c = np.nonzero(mask)
    if len(k) < 4:
        return None
    P = np.stack([(c + org[2]) * dx_um, (r + org[1]) * dx_um, (k + org[0]) * dz_um], axis=1)
    ctr = P.mean(axis=0)
    w, v = np.linalg.eigh(np.cov((P - ctr).T))
    a = v[:, -1]
    t = (P - ctr) @ a
    foot = abs(a[0]) * dx_um + abs(a[1]) * dx_um + abs(a[2]) * dz_um     # empreinte d'un voxel projetée sur l'axe
    return dict(center=ctr, axis=a, length=float(t.max() - t.min() + foot),
                thickness=float(4 * math.sqrt(max(w[:2].mean(), 0.0))), volume=float(len(k) * dx_um * dx_um * dz_um),
                n=int(len(k)))


def axis_angle_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(abs(float(np.dot(a, b))), 0, 1))))


# =============================================================================
# INFÉRENCE
# =============================================================================

@torch.no_grad()
def segment(model, V, pts_um, refocus, amp=True, batch=64, with_heat=False):
    """
    Probabilités self (B, nz, nxy, nxy) float16, origines des fenêtres autour de pts_um et, si with_heat,
    carte des centres (B, nz, nxy, nxy) float16.
    """
    shape = (0, refocus.nz, refocus.nxy, refocus.nxy)
    if len(pts_um) == 0:
        z = torch.zeros(shape, device=V.device, dtype=torch.float16)
        return (z, np.zeros((0, 3), np.int64), z) if with_heat else (z, np.zeros((0, 3), np.int64))
    X, org = refocus(V, pts_um)
    out = torch.empty((len(X),) + shape[1:], device=V.device, dtype=torch.float16)
    heat = torch.empty_like(out) if with_heat else None
    for b0 in range(0, len(X), batch):
        x = X[b0:b0 + batch].contiguous(memory_format=torch.channels_last_3d)
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            y = model(x).float()
        out[b0:b0 + batch] = torch.sigmoid(y[:, 0]).half()
        if with_heat:
            heat[b0:b0 + batch] = torch.sigmoid(y[:, 2]).half()
    return (out, org, heat) if with_heat else (out, org)


@torch.no_grad()
def window_peaks(heat, org, dx_um, dz_um, thr=INFER['heat_thr'], nms=INFER['nms'], border=INFER['border']):
    """Centres (n, 3) µm (x, y, z), scores et fenêtre d'origine des maxima locaux de la carte des centres."""
    if len(heat) == 0:
        return np.zeros((0, 3)), np.zeros(0), np.zeros(0, int)
    h = heat.float()[:, None]
    mx = F.max_pool3d(h, nms, stride=1, padding=tuple(k // 2 for k in nms))
    keep = (h == mx) & (h > thr)
    if border:
        keep[..., :border, :, :] = False
        keep[..., -border:, :, :] = False
        keep[..., :, :border, :] = False
        keep[..., :, -border:, :] = False
        keep[..., :, :, :border] = False
        keep[..., :, :, -border:] = False
    b, _, k, r, c = torch.nonzero(keep, as_tuple=True)
    sc = h[b, 0, k, r, c].cpu().numpy()
    b, k, r, c = (t.cpu().numpy() for t in (b, k, r, c))
    pos = np.stack([(c + org[b, 2]) * dx_um, (r + org[b, 1]) * dx_um, (k + org[b, 0]) * dz_um], axis=1)
    return pos, sc, b


def iou_sparse(o1, m1, o2, m2):
    """IoU de deux masques booléens dans la même grille, origines (k, r, c)."""
    lo = [max(a, b) for a, b in zip(o1, o2)]
    hi = [min(a + s, b + t) for a, s, b, t in zip(o1, m1.shape, o2, m2.shape)]
    inter = 0
    if all(h > l for l, h in zip(lo, hi)):
        a = m1[tuple(slice(l - o, h - o) for l, h, o in zip(lo, hi, o1))]
        b = m2[tuple(slice(l - o, h - o) for l, h, o in zip(lo, hi, o2))]
        inter = int((a & b).sum())
    u = int(m1.sum()) + int(m2.sum()) - inter
    return inter / u if u else 0.0


def _near(a, b, xy, z):
    """Booléen (n_a,) : chaque point de a a-t-il un point de b à moins de (xy, z) (distance normalisée) ?"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros(len(a), bool)
    s = np.array([1 / xy, 1 / xy, 1 / z])
    d, _ = cKDTree(np.asarray(b) * s).query(np.asarray(a) * s)
    return d < 1.0


def nms_points(P, scores, xy, z):
    """Indices gardés : par score décroissant, chaque point supprime ses voisins à moins de (xy, z) (normalisé)."""
    if len(P) == 0:
        return np.zeros(0, int)
    s = np.array([1 / xy, 1 / xy, 1 / z])
    tree = cKDTree(np.asarray(P) * s)
    dead = np.zeros(len(P), bool)
    keep = []
    for i in np.argsort(-np.asarray(scores)):
        if not dead[i]:
            keep.append(i)
            dead[tree.query_ball_point(P[i] * s, 1.0)] = True
    return np.array(keep, int)


def dedup_instances(inst, xy=1.5, z=3.0, iou_thr=0.5):
    """
    Doublons : instances dont les centres sont à moins de (xy, z) et dont les masques se recouvrent
    (IoU > iou_thr) ; on garde la plus grosse. Un seul arbre, IoU calculée sur les paires voisines seulement.
    """
    if len(inst) < 2:
        return inst
    C = np.array([d['center'] for d in inst]) * np.array([1 / xy, 1 / xy, 1 / z])
    nbrs = [[] for _ in inst]
    for i, j in cKDTree(C).query_pairs(1.0):
        nbrs[i].append(j)
        nbrs[j].append(i)
    removed = np.zeros(len(inst), bool)
    kept = []
    for i in np.argsort([-d['n'] for d in inst], kind='stable'):
        if removed[i]:
            continue
        kept.append(i)
        for j in nbrs[i]:
            if not removed[j] and iou_sparse(inst[i]['origin'], inst[i]['mask'], inst[j]['origin'], inst[j]['mask']) > iou_thr:
                removed[j] = True
    return [inst[i] for i in sorted(kept)]


def mask_center(mask, org, dx_um, dz_um):
    k, r, c = np.nonzero(mask)
    return np.array([(c.mean() + org[2]) * dx_um, (r.mean() + org[1]) * dx_um, (k.mean() + org[0]) * dz_um])


@torch.no_grad()
def segment_instances(model, V, pos1, refocus, cfg=INFER, rounds=2):
    """
    Instances (liste de dict : origin, mask (bool), center (µm), score, round, n) à partir des détections pos1.
    Tour 1 : fenêtres sur pos1. Tour t > 1 : fenêtres sur les centres vus (carte des centres) qui ne sont
    proches d'aucune instance déjà trouvée. Doublons : instances proches dont les masques se recouvrent
    (IoU > dup_iou) -> on garde la plus grosse.
    """
    dx, dz = refocus.dx, refocus.dz
    inst, cand = [], np.asarray(pos1, dtype=np.float64).reshape(-1, 3)
    for rnd in range(1, rounds + 1):
        if len(cand) == 0:
            break
        P, org, H = segment(model, V, cand, refocus, with_heat=True)
        Pm = P > 0.5
        nv_t = Pm.flatten(1).sum(dim=1)
        sc = ((P.float() * Pm).flatten(1).sum(dim=1) / nv_t.clamp(min=1)).cpu().numpy()   # score moyen, sur le GPU
        nv = nv_t.cpu().numpy()
        good = np.nonzero(nv >= cfg['vmin'])[0]
        Pm = Pm[torch.as_tensor(good, device=Pm.device)].cpu().numpy() if len(good) else np.zeros((0,) + tuple(P.shape[1:]), bool)
        for q, i in enumerate(good):
            inst.append(dict(origin=org[i], mask=Pm[q], center=mask_center(Pm[q], org[i], dx, dz),
                             score=float(sc[i]), round=rnd, n=int(nv[i])))
        if rnd == rounds:
            break
        peaks, psc, _ = window_peaks(H, org, dx, dz, cfg['heat_thr'])
        if len(peaks) == 0:
            break
        # centres vus dans plusieurs fenêtres : regroupés (le plus sûr d'abord)
        uniq = peaks[nms_points(peaks, psc, cfg['same_xy_um'], cfg['same_z_um'])]
        found = np.array([d['center'] for d in inst]) if inst else np.zeros((0, 3))
        cand = uniq[~_near(uniq, found, cfg['same_xy_um'] * 2, cfg['same_z_um'] * 2)]
    return dedup_instances(inst, 1.5, 3.0, cfg['dup_iou'])

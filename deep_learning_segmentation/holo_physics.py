# -*- coding: utf-8 -*-

"""
Filename: holo_physics.py

Description:
Briques physiques du U-Net 3D, en PyTorch (GPU), sans aller-retour CuPy :
  - Reconstructor : hologramme normalisé (fond ≈ 1) -> volume complexe reconstruit (Re, Im) sur une
    grille décimée. La décimation latérale se fait en recadrant le spectre (interpolation à bande
    limitée exacte) : avec decim = 2, le pixel passe de 0,175 à 0,35 µm et la propagation coûte
    4 fois moins ; la bande conservée (rayon ≤ 256 px de Fourier du capteur) contient la bande
    15–250 px de la méthode classique. Échantillon j de la grille décimée <-> x = j · decim · dx.
  - rendu des cibles sur une grille de patch : masque du bâtonnet (voxélisé, dilaté d'un
    demi-voxel pour qu'un bâtonnet de 0,6 µm couvre au moins un plan de 1 µm) et carte de chaleur
    gaussienne des centres (valeur 1 au voxel le plus proche du centre).

Repères (ceux d'HOLOTRACKER et du simulateur hybride) : x = colonne, y = ligne, en µm ; z = distance
de propagation positive qui met l'objet au point. Volumes ordonnés (Z, Y, X).

Author: Simon BECKER
Date: 2026-10-17

License:
GNU General Public License v3.0
"""

import math

import numpy as np
import torch


class Reconstructor:
    """
    Volume reconstruit par spectre angulaire, en PyTorch.
      n          : taille de l'hologramme (carré, px)
      dx_um      : pixel objet (pixel caméra / grossissement)
      decim      : décimation latérale (1, 2, 4)
      z_um       : distances des plans reconstruits
      high_pass, low_pass : filtre spectral en px de Fourier du capteur (grille n), 0 = pas de borne
    Sortie : tenseur (2, Z, n/decim, n/decim) float32, Re et Im du champ de H = I − 1, divisés par
    l'écart-type de H filtré (unités de bruit : l'échelle ne dépend ni du film ni de l'éclairage).
    """

    def __init__(self, n, dx_um, wavelength_um, index_medium, z_um, decim=2, high_pass=15, low_pass=250,
                 device='cuda'):
        self.n, self.decim, self.device = n, decim, device
        self.m = n // decim
        self.dx_um = dx_um
        self.z_um = torch.as_tensor(np.asarray(z_um, dtype=np.float32), device=device)
        lam = wavelength_um / index_medium
        self.lam_um = lam                                                # longueur d'onde dans le milieu
        f = torch.fft.fftfreq(n, d=dx_um, device=device)                 # µm⁻¹, grille complète
        keep = torch.cat([torch.arange(0, self.m // 2), torch.arange(n - self.m // 2, n)]).to(device)
        self.keep = keep                                                 # indices du spectre recadré
        fk = f[keep]
        fy, fx = torch.meshgrid(fk, fk, indexing='ij')
        rho = torch.sqrt(fx ** 2 + fy ** 2) * n * dx_um                  # px de Fourier du capteur
        arg = 1.0 / lam ** 2 - fx ** 2 - fy ** 2
        self.kz = torch.sqrt(torch.clamp(arg, min=0.0))                  # µm⁻¹
        band = arg > 0
        if high_pass:
            band &= rho >= high_pass
        if low_pass:
            band &= rho <= low_pass
        self.band = band.to(torch.complex64)
        # échelle fixe : écart-type, après filtrage, d'un bruit capteur blanc de σ = sensor_noise
        # (0,015 mesuré sur le film réel). Une échelle estimée sur l'image dépendrait de la densité.
        self.sensor_noise = 0.015
        self.scale = self.sensor_noise * math.sqrt(float(band.sum()) / n ** 2)

    @torch.no_grad()
    def __call__(self, holo_norm, z_chunk=32):
        """holo_norm : (n, n) numpy ou tenseur, fond ≈ 1. Retourne (2, Z, m, m) float32 sur le GPU."""
        h = torch.as_tensor(holo_norm, device=self.device, dtype=torch.float32) - 1.0
        F = torch.fft.fft2(h)[self.keep][:, self.keep] * self.band
        # facteur d'échelle de la décimation : même amplitude qu'à pleine résolution
        F = F / (self.decim ** 2)
        noise = self.scale
        out = torch.empty((2, len(self.z_um), self.m, self.m), device=self.device, dtype=torch.float32)
        for k0 in range(0, len(self.z_um), z_chunk):
            z = self.z_um[k0:k0 + z_chunk]
            kern = torch.exp((2j * math.pi) * z[:, None, None] * self.kz[None])
            E = torch.fft.ifft2(F[None] * kern)
            out[0, k0:k0 + len(z)] = E.real
            out[1, k0:k0 + len(z)] = E.imag
        return out / (noise + 1e-12)


# =============================================================================
# NORMALISATION LOCALE (médiane / MAD par maille, comme le seuil « local MAD » d'HOLOTRACKER)
# =============================================================================

@torch.no_grad()
def local_mad_normalize(V, mesh=(13, 32, 32), sub=(1, 4, 4)):
    """
    V : (C, Z, Y, X). Chaque canal : (V − médiane locale) / σ locale, avec σ = 1,4826 × MAD, estimés
    par maille (HOLOTRACKER code_locate/CCL3D.mailles_fond_mad : statistiques de rang, robustes aux
    objets par construction ; voxels sous-échantillonnés de `sub` avant les tris), puis interpolés
    en trilinéaire au centre des mailles. Avec 2 canaux (Re, Im), σ est commune aux deux (moyenne
    des deux MAD) pour garder la relation de phase. Maille par défaut : 13 plans × 32 × 32 px =
    13 × 11,2 × 11,2 µm (HOLOTRACKER : 25 × 64 × 64 à 0,5 µm et 0,175 µm). Z, Y, X doivent être
    multiples de la maille ; sinon le reste est couvert par la dernière maille (interpolation).
    """
    C, Z, Y, X = V.shape
    mz, my, mx = mesh
    gz, gy, gx = Z // mz, Y // my, X // mx
    t = V[:, :gz * mz:sub[0], :gy * my:sub[1], :gx * mx:sub[2]]
    t = t.reshape(C, gz, mz // sub[0], gy, my // sub[1], gx, mx // sub[2]).permute(0, 1, 3, 5, 2, 4, 6)
    t = t.reshape(C, gz, gy, gx, -1)
    med = t.median(dim=-1).values
    mad = (t - med[..., None]).abs().median(dim=-1).values * 1.4826
    sig = mad.mean(dim=0, keepdim=True).expand_as(mad) if C > 1 else mad
    up = lambda g: torch.nn.functional.interpolate(g[None], size=(Z, Y, X), mode='trilinear', align_corners=False)[0]
    return (V - up(med)) / (up(sig) + 1e-6)


# =============================================================================
# CIBLES
# =============================================================================

ORIENT_MIN = 0.3
ORIENT_CHANNELS = ('xx', 'yy', 'zz', 'xy', 'xz', 'yz')


def orient_flip(o, axis):
    """Tenseur d'orientation (6, ...) après symétrie : axis 'x' (x -> −x), 'y', ou 'xy' (échange x ↔ y)."""
    xx, yy, zz, xy, xz, yz = o
    if axis == 'x':
        return torch.stack([xx, yy, zz, -xy, -xz, yz])
    if axis == 'y':
        return torch.stack([xx, yy, zz, -xy, xz, -yz])
    return torch.stack([yy, xx, zz, xy, yz, xz])


def axis_from_tensor(t6):
    """(n, 6) -> (n, 3) vecteur propre de plus grande valeur propre (axe estimé)."""
    t6 = np.asarray(t6, dtype=np.float64)
    M = np.zeros((len(t6), 3, 3))
    M[:, 0, 0], M[:, 1, 1], M[:, 2, 2] = t6[:, 0], t6[:, 1], t6[:, 2]
    M[:, 0, 1] = M[:, 1, 0] = t6[:, 3]
    M[:, 0, 2] = M[:, 2, 0] = t6[:, 4]
    M[:, 1, 2] = M[:, 2, 1] = t6[:, 5]
    w, v = np.linalg.eigh(M)
    return v[:, :, -1]


def rod_axes(theta_deg, phi_deg):
    """Vecteur unitaire de l'axe (x, y, z) : θ azimut depuis x dans le plan image, φ angle à l'axe Z."""
    th, ph = np.radians(theta_deg), np.radians(phi_deg)
    return np.stack([np.sin(ph) * np.cos(th), np.sin(ph) * np.sin(th), np.cos(ph)], axis=-1)


def patch_grid(origin_zyx, shape_zyx, step_zyx_um, z0_um, device):
    """Coordonnées (µm) des voxels d'un patch : origin en indices de la grille du volume."""
    gz = z0_um + (origin_zyx[0] + torch.arange(shape_zyx[0], device=device)) * step_zyx_um[0]
    gy = (origin_zyx[1] + torch.arange(shape_zyx[1], device=device)) * step_zyx_um[1]
    gx = (origin_zyx[2] + torch.arange(shape_zyx[2], device=device)) * step_zyx_um[2]
    return gz, gy, gx


def render_targets(bacts, origin_zyx, shape_zyx, step_zyx_um, z0_um, length_um, thickness_um,
                   sigma_xy_um=0.5, sigma_z_um=1.0, device='cuda', want=('mask', 'heat')):
    """
    bacts : array (n, 5) x, y, z, θ, φ (µm, degrés). Retourne dict 'mask' (Z, Y, X) {0, 1},
    'heat' (Z, Y, X) [0, 1] et 'orient' (6, Z, Y, X) sur le patch.
    'orient' : tenseur a·aᵀ de l'axe a de la bactérie dont la gaussienne domine le voxel, composantes
    (xx, yy, zz, xy, xz, yz) — insensible au signe de a (un bâtonnet n'a pas de sens).
    Il est appris là où la carte de chaleur dépasse ORIENT_MIN (voisinage du centre).
    Masque : voxel retenu si la distance au segment, normalisée par (t/2 + demi-voxel) en XY et en Z,
    est ≤ 1 — le bâtonnet voxélisé couvre au moins un plan quelle que soit son orientation.
    """
    gz, gy, gx = patch_grid(origin_zyx, shape_zyx, step_zyx_um, z0_um, device)
    out = {}
    if 'mask' in want:
        out['mask'] = torch.zeros(shape_zyx, device=device)
    if 'heat' in want or 'orient' in want:
        out['heat'] = torch.zeros(shape_zyx, device=device)
    if 'orient' in want:
        out['orient'] = torch.zeros((6,) + tuple(shape_zyx), device=device)
    if len(bacts) == 0:
        return out
    half = (length_um - thickness_um) / 2.0
    margin = length_um / 2 + 4 * max(sigma_xy_um, sigma_z_um, thickness_um)
    ax = rod_axes(bacts[:, 3], bacts[:, 4])
    r_xy = thickness_um / 2 + step_zyx_um[2] / 2
    r_z = thickness_um / 2 + step_zyx_um[0] / 2
    lo = torch.tensor([float(gz[0]), float(gy[0]), float(gx[0])])
    hi = torch.tensor([float(gz[-1]), float(gy[-1]), float(gx[-1])])
    for b, a in zip(bacts, ax):
        x, y, z = b[0], b[1], b[2]
        if not (lo[0] - margin <= z <= hi[0] + margin and lo[1] - margin <= y <= hi[1] + margin
                and lo[2] - margin <= x <= hi[2] + margin):
            continue
        # sous-boîte du patch autour de l'objet
        iz = torch.nonzero((gz >= z - margin) & (gz <= z + margin)).flatten()
        iy = torch.nonzero((gy >= y - margin) & (gy <= y + margin)).flatten()
        ix = torch.nonzero((gx >= x - margin) & (gx <= x + margin)).flatten()
        if len(iz) == 0 or len(iy) == 0 or len(ix) == 0:
            continue
        sz, sy, sx = slice(int(iz[0]), int(iz[-1]) + 1), slice(int(iy[0]), int(iy[-1]) + 1), slice(int(ix[0]), int(ix[-1]) + 1)
        Z, Y, X = torch.meshgrid(gz[sz] - z, gy[sy] - y, gx[sx] - x, indexing='ij')
        if 'mask' in want:
            t = torch.clamp(X * a[0] + Y * a[1] + Z * a[2], -half, half)
            dX, dY, dZ = X - t * a[0], Y - t * a[1], Z - t * a[2]
            d = (dX ** 2 + dY ** 2) / r_xy ** 2 + dZ ** 2 / r_z ** 2
            out['mask'][sz, sy, sx] = torch.maximum(out['mask'][sz, sy, sx], (d <= 1.0).float())
        if 'heat' in want or 'orient' in want:
            g = torch.exp(-(X ** 2 + Y ** 2) / (2 * sigma_xy_um ** 2) - Z ** 2 / (2 * sigma_z_um ** 2))
            if 'orient' in want:
                own = g > out['heat'][sz, sy, sx]
                tensor = torch.tensor([a[0] * a[0], a[1] * a[1], a[2] * a[2], a[0] * a[1], a[0] * a[2], a[1] * a[2]],
                                      device=device, dtype=torch.float32)
                o = out['orient'][:, sz, sy, sx]
                out['orient'][:, sz, sy, sx] = torch.where(own[None], tensor[:, None, None, None], o)
            out['heat'][sz, sy, sx] = torch.maximum(out['heat'][sz, sy, sx], g)
            # pic exactement à 1 au voxel le plus proche du centre (loss focale type CenterNet)
            kz = int(round((z - float(gz[0])) / step_zyx_um[0]))
            ky = int(round((y - float(gy[0])) / step_zyx_um[1]))
            kx = int(round((x - float(gx[0])) / step_zyx_um[2]))
            if 0 <= kz < shape_zyx[0] and 0 <= ky < shape_zyx[1] and 0 <= kx < shape_zyx[2]:
                out['heat'][kz, ky, kx] = 1.0
    return out

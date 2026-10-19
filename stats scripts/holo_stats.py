# -*- coding: utf-8 -*-

"""
Filename: holo_stats.py

Description:
Étude statistique d'un hologramme (réel ou simulé) pour comparer simulation et mesure.
Chaîne identique à HOLOTRACKER : nettoyage par l'hologramme de référence (soustraction ou
division), propagation du volume par spectre angulaire, focus, seuil, CCL3D.

Statistiques produites à trois niveaux :
  1. hologramme : brut, référence, nettoyé (niveaux, saturation, bruit, enveloppe, spectre)
  2. volume     : module propagé et volume de focus (global, par plan, fond robuste, seuil)
  3. objets     : bactéries / micro-objets détectés (taille, pic, contraste, étendue, voisins)

Sorties (dossier --out, par défaut <hologramme>_stats_<date>/) :
  summary.json, planes.csv, objects.csv, spectrum.csv, figures PNG

Utilisation (un seul argument : le fichier YAML, qui contient tous les paramètres) :
  python stats/holo_stats.py stats/holo_stats_real.yaml
  python stats/holo_stats.py stats/holo_stats_simu.yaml

Les options possibles de chaque paramètre sont décrites en commentaire dans les YAML.
Chemins relatifs : par rapport au dossier du fichier YAML.

Author: Simon BECKER
Date: 2026-09-17

License:
GNU General Public License v3.0
"""

import datetime
import json
import os
import sys
import time

import numpy as np
import cupy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from libs import propagation as propag
from libs import focus
from libs.focus import Focus_type
from libs.CCL3D import CCL3D, CCA_CUDA_float, calc_threshold, type_threshold

PERCENTILES = [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9]
MAD_TO_SIGMA = 1.4826


# =============================================================================
# LECTURE ET NETTOYAGE
# =============================================================================

def load_image(path, size_x, size_y):
    """Lit BMP / PNG / TIFF / NPY et recadre au centre sur size_x × size_y (comme read_image)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        img = np.load(path)
    elif ext in ('.tif', '.tiff'):
        import tifffile
        img = tifffile.imread(path)
    else:
        img = np.asarray(Image.open(path))
    if img.ndim != 2:
        raise ValueError(f"{path} : image 2D attendue, forme {img.shape}")
    sy, sx = img.shape
    if sx < size_x or sy < size_y:
        raise ValueError(f"{path} : image {sx}×{sy} plus petite que {size_x}×{size_y}")
    ox, oy = (sx - size_x) // 2, (sy - size_y) // 2
    return img[oy:oy + size_y, ox:ox + size_x]


def load_reference(path, size_x, size_y, extension, mean_type='arithmetic'):
    """
    Référence = un fichier image (type natif conservé, comme HOLOTRACKER qui le lit par
    np.array(Image.open(...))), ou un dossier dont on calcule l'hologramme moyen de toutes
    les images d'extension `extension`, exactement comme calc_holo_moyen de HOLOTRACKER :
    accumulation en float64, puis conversion en float32.
      mean_type : 'arithmetic' (somme / n) ou 'geometric' (exp(moyenne(log x)), « mean_log »)
    """
    if not os.path.isdir(path):
        return load_image(path, size_x, size_y), 1
    files = sorted(f for f in os.listdir(path) if f.lower().endswith(extension.lower()))
    if not files:
        raise ValueError(f"{path} : aucune image {extension}")
    acc = np.zeros((size_y, size_x), dtype=np.float64)
    for f in files:
        img = load_image(os.path.join(path, f), size_x, size_y).astype(np.float64)
        if mean_type == 'geometric':
            acc += np.log(np.clip(img, np.finfo(np.float32).tiny, None))
        else:
            acc += img
    mean = acc / len(files)
    if mean_type == 'geometric':
        mean = np.exp(mean)
    return mean.astype(np.float32), len(files)


def check_reference(raw, ref):
    """Avertit si la référence n'est pas à l'échelle de l'hologramme (ex. moyenne ré-étirée sur 0-255)."""
    warnings = []
    if ref.min() <= 0:
        warnings.append(f"la référence contient {int((ref <= 0).sum())} pixel(s) <= 0 : la division y explose")
    ratio = raw.mean() / ref.mean() if ref.mean() != 0 else np.inf
    if not 0.8 < ratio < 1.25:
        warnings.append(f"moyenne hologramme / référence = {ratio:.3g} : échelles différentes ?")
    std_ratio = ref.std() / raw.std() if raw.std() != 0 else np.inf
    if std_ratio > 1.2:
        warnings.append(f"écart-type référence / hologramme = {std_ratio:.3g} : une moyenne devrait être "
                        f"moins dispersée que l'hologramme (référence ré-étirée ?)")
    return warnings


def clean_hologram(raw, ref, cleaning, division_power):
    """
    Nettoyage identique à HOLOTRACKER (core.py), types compris : l'hologramme brut est en
    float32 (read_image), la division se fait en float64, le résultat est en float32.
    """
    raw = raw.astype(np.float32)
    if cleaning == 'subtraction':
        cleaned = raw - ref
        cleaned = cleaned - cleaned.min()      # HOLOTRACKER : valeurs positives
    elif cleaning == 'division':
        cleaned = raw.astype(np.float64) / (ref + 1e-10)
        cleaned = np.power(cleaned, division_power)   # HOLOTRACKER : limite les extrêmes (0.8)
    else:
        cleaned = raw.copy()
    return cleaned.astype(np.float32)


# =============================================================================
# STATISTIQUES
# =============================================================================

def basic_stats(values):
    v = np.asarray(values, dtype=np.float64).ravel()
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    d = {
        'min': float(v.min()), 'max': float(v.max()),
        'mean': float(v.mean()), 'std': float(v.std()),
        'median': float(med), 'robust_std': float(MAD_TO_SIGMA * mad),
        'contrast_std_over_mean': float(v.std() / v.mean()) if v.mean() != 0 else None,
    }
    d.update({f'p{p:g}': float(x) for p, x in zip(PERCENTILES, np.percentile(v, PERCENTILES))})
    return d


def noise_estimate(img):
    """
    Bruit blanc estimé par le laplacien discret [[1,-2,1],[-2,4,-2],[1,-2,1]] (Immerkær),
    en version robuste (MAD) pour limiter la contribution des franges et des objets.
    Pour un bruit blanc d'écart-type s, la réponse a pour écart-type 6·s.
    """
    I = img.astype(np.float64)
    L = (I[:-2, :-2] - 2 * I[:-2, 1:-1] + I[:-2, 2:]
         - 2 * I[1:-1, :-2] + 4 * I[1:-1, 1:-1] - 2 * I[1:-1, 2:]
         + I[2:, :-2] - 2 * I[2:, 1:-1] + I[2:, 2:])
    return float(MAD_TO_SIGMA * np.median(np.abs(L - np.median(L))) / 6.0)


def radial_spectrum(img, dx_um):
    """Densité spectrale de puissance moyennée par anneau. Fréquences en cycles/µm."""
    I = img.astype(np.float64)
    I = I - I.mean()
    ny, nx = I.shape
    P = np.abs(np.fft.fftshift(np.fft.fft2(I))) ** 2 / (nx * ny)
    yy, xx = np.indices(P.shape)
    r = np.hypot(xx - nx // 2, yy - ny // 2).astype(int)
    rmax = min(nx, ny) // 2
    sums = np.bincount(r.ravel(), P.ravel())[:rmax]
    counts = np.bincount(r.ravel())[:rmax]
    freq = np.arange(rmax) / (nx * dx_um)
    return freq, sums / np.maximum(counts, 1)


def hologram_stats(img, dx_um, envelope_sigma_px):
    d = basic_stats(img)
    d['dtype'] = str(img.dtype)
    d['distinct_levels'] = int(np.unique(img).size)
    if img.dtype == np.uint8:
        d['saturation_low_fraction'] = float((img == 0).mean())
        d['saturation_high_fraction'] = float((img == 255).mean())
    d['noise_std_laplacian'] = noise_estimate(img)
    d['noise_std_over_mean'] = d['noise_std_laplacian'] / d['mean'] if d['mean'] != 0 else None
    env = gaussian_filter(img.astype(np.float64), envelope_sigma_px)
    d['envelope_min'] = float(env.min())
    d['envelope_max'] = float(env.max())
    d['envelope_variation_over_mean'] = float((env.max() - env.min()) / env.mean()) if env.mean() != 0 else None
    d['structure_std_without_envelope'] = float((img - env).std())
    freq, psd = radial_spectrum(img, dx_um)
    hf = psd[int(0.75 * len(psd)):]
    d['psd_high_freq_plateau'] = float(hf.mean())
    return d, (freq, psd)


def volume_stats(d_volume, subsample_xy):
    """Statistiques globales (sous-échantillon CPU) et par plan (GPU + CPU)."""
    sub = cp.asnumpy(d_volume[:, ::subsample_xy, ::subsample_xy])
    glob = basic_stats(sub)
    glob['mean_gpu_full'] = float(cp.mean(d_volume))
    glob['std_gpu_full'] = float(cp.std(d_volume))
    glob['max_gpu_full'] = float(cp.max(d_volume))
    planes = {
        'mean': cp.asnumpy(d_volume.mean(axis=(1, 2))),
        'std': cp.asnumpy(d_volume.std(axis=(1, 2))),
        'max': cp.asnumpy(d_volume.max(axis=(1, 2))),
        'median': np.median(sub, axis=(1, 2)),
        'p99': np.percentile(sub, 99, axis=(1, 2)),
    }
    return glob, planes, sub


def object_stats(d_labels, d_focus, bin_cpu, dx_um, dz_um, bg_median, bg_sigma, z0_um=0.0, d_idx=None):
    """
    Descripteurs par objet. Seuls les voxels segmentés sont rapatriés sur CPU
    (cupy.bincount est inutilisable sur cette machine).
    Z absolu = distance à l'hologramme : z0_um (distance_ini) + indice de plan × pas,
    comme le z_offset de CCA_CUDA_float dans HOLOTRACKER.

    d_idx : indices (z, y, x) des voxels segmentés déjà calculés sur le GPU (cp.nonzero(d_bin)).
            S'ils sont fournis, bin_cpu n'est pas utilisé (peut valoir None) : on évite
            np.nonzero sur le volume entier côté CPU (~0,5 s pour 200 × 1024²). Mêmes indices,
            même ordre (C) : résultats identiques.
    """
    if d_idx is None:
        zi, yi, xi = np.nonzero(bin_cpu)
        d_idx = (cp.asarray(zi), cp.asarray(yi), cp.asarray(xi))
    else:
        zi, yi, xi = (cp.asnumpy(ix) for ix in d_idx)
    if zi.size == 0:
        return {}
    lab = cp.asnumpy(d_labels[d_idx]).astype(np.int64)
    val = cp.asnumpy(d_focus[d_idx]).astype(np.float64)

    order = np.argsort(lab, kind='stable')
    lab, val, zi, yi, xi = lab[order], val[order], zi[order], yi[order], xi[order]
    starts = np.flatnonzero(np.r_[True, lab[1:] != lab[:-1]])
    ids = lab[starts]

    nb = np.diff(np.r_[starts, lab.size])
    s = np.add.reduceat(val, starts)
    peak = np.maximum.reduceat(val, starts)
    obj = {
        'label': ids,
        'nb_vox': nb,
        'bary_x_um': np.add.reduceat(val * xi, starts) / s * dx_um,
        'bary_y_um': np.add.reduceat(val * yi, starts) / s * dx_um,
        'bary_z_um': z0_um + np.add.reduceat(val * zi, starts) / s * dz_um,
        'focus_peak': peak,
        'focus_mean': s / nb,
        'focus_sum': s,
        'peak_snr_robust': (peak - bg_median) / bg_sigma if bg_sigma > 0 else np.full(ids.size, np.nan),
        'extent_x_um': (np.maximum.reduceat(xi, starts) - np.minimum.reduceat(xi, starts) + 1) * dx_um,
        'extent_y_um': (np.maximum.reduceat(yi, starts) - np.minimum.reduceat(yi, starts) + 1) * dx_um,
        'extent_z_um': (np.maximum.reduceat(zi, starts) - np.minimum.reduceat(zi, starts) + 1) * dz_um,
    }
    # plan du pic de chaque objet
    is_peak = val == np.repeat(peak, nb)
    first_peak = np.minimum.reduceat(np.where(is_peak, np.arange(val.size), val.size), starts)
    obj['z_peak_um'] = z0_um + zi[first_peak] * dz_um

    if ids.size >= 2:
        pts = np.column_stack([obj['bary_x_um'], obj['bary_y_um'], obj['bary_z_um']])
        dist, _ = cKDTree(pts).query(pts, k=2)
        obj['nearest_neighbor_um'] = dist[:, 1]
    else:
        obj['nearest_neighbor_um'] = np.full(ids.size, np.nan)
    return obj


# =============================================================================
# MODULE NORMALISÉ ET PROFILS AXIAUX
# =============================================================================

NORM_EPS = 1e-9   # « micro constante » de la normalisation, en unités du module


def propagate_module(cleaned, args, medium_wavelength, nx, ny, nz):
    """Volume du module propagé (float32, GPU), comme HOLOTRACKER (libs/propagation.py aligné)."""
    d_holo = cp.asarray(cleaned.astype(np.complex64))
    d_fft = cp.zeros((ny, nx), dtype=cp.complex64)
    d_fft_propag = cp.zeros((ny, nx), dtype=cp.complex64)
    d_kernel = cp.zeros((ny, nx), dtype=cp.complex64)
    d_volume = cp.zeros((nz, ny, nx), dtype=cp.float32)
    propag.volume_propag_angular_spectrum_to_module(
        d_holo, d_fft, d_kernel, d_fft_propag, d_volume,
        medium_wavelength, args.magnification, args.pixel_size, nx, ny,
        args.distance_ini, args.step, nz, args.high_pass, args.low_pass)
    del d_holo, d_fft, d_fft_propag, d_kernel
    return d_volume


def focus_volume_complex(cleaned, args, medium_wavelength, nx, ny, nz, sum_size, focus_type):
    """
    Volume de focus (float32, GPU) pour un critère qui exige le champ complexe
    (focus.COMPLEX_FOCUS_TYPES) : propagation plan par plan, porteuse retirée, critère calculé
    sur chaque plan complexe sans stocker le volume complexe.
    """
    plane_fns = {Focus_type.VAR_IMAG_OVER_REAL: focus.focus_plane_var_imag_over_real}
    d_holo = cp.asarray(cleaned.astype(np.complex64))
    d_volume = cp.zeros((nz, ny, nx), dtype=cp.float32)
    propag.volume_propag_angular_spectrum_to_focus_complex(
        d_holo, d_volume, plane_fns[focus_type], sum_size,
        medium_wavelength, args.magnification, args.pixel_size, nx, ny,
        args.distance_ini, args.step, nz, args.high_pass, args.low_pass)
    del d_holo
    return d_volume


def normalized_volume_stats(d_volume, mean, std, subsample_xy):
    """
    Statistiques du module normalisé voxel par voxel : V_norm = (V - moyenne) / (écart-type + ε),
    moyenne et écart-type étant ceux du volume entier. Min, max, moyenne, écart-type et
    fractions au-dessus de k σ : volume complet (GPU). Médiane, écart-type robuste et
    percentiles : sous-échantillon, sur CPU (comme les autres statistiques du volume).
    """
    scale = 1.0 / (std + NORM_EPS)
    d_norm = (d_volume - np.float32(mean)) * np.float32(scale)
    d = basic_stats(cp.asnumpy(d_norm[:, ::subsample_xy, ::subsample_xy]))
    d.update({'min': float(cp.min(d_norm)), 'max': float(cp.max(d_norm)),
              'mean': float(cp.mean(d_norm)), 'std': float(cp.std(d_norm))})
    for k in (3, 5, 10):
        d[f'fraction_above_{k}sigma'] = float(cp.mean((d_norm > k).astype(cp.float32)))
    d['normalization'] = {'volume_mean': mean, 'volume_std': std, 'epsilon': NORM_EPS}
    del d_norm
    return d


def object_z_profiles(d_volume, obj, dx_um, half=2):
    """
    Profil axial de chaque objet : pour chaque plan, maximum du volume dans une fenêtre de
    (2·half+1)² pixels centrée sur le barycentre XY de l'objet. Retour : (n_objets, n_plans).
    """
    nz, ny, nx = d_volume.shape
    cx = np.clip(np.round(obj['bary_x_um'] / dx_um).astype(int), 0, nx - 1)
    cy = np.clip(np.round(obj['bary_y_um'] / dx_um).astype(int), 0, ny - 1)
    off = np.arange(-half, half + 1)
    X = np.clip(cx[:, None, None] + off[None, None, :], 0, nx - 1)
    Y = np.clip(cy[:, None, None] + off[None, :, None], 0, ny - 1)
    X, Y = np.broadcast_arrays(X, Y)
    vals = cp.asnumpy(d_volume[:, cp.asarray(Y), cp.asarray(X)])      # (nz, n, w, w)
    return vals.max(axis=(2, 3)).T


def distribution_summary(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None
    d = {'count': int(x.size), 'mean': float(x.mean()), 'std': float(x.std()),
         'min': float(x.min()), 'max': float(x.max())}
    d.update({f'p{p:g}': float(v) for p, v in zip([5, 25, 50, 75, 95], np.percentile(x, [5, 25, 50, 75, 95]))})
    return d


# =============================================================================
# FIGURES
# =============================================================================

def save_fig(fig, out_dir, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, name), dpi=110)
    plt.close(fig)


def fig_holograms(out_dir, raw, ref, cleaned, spectra, dx_um):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, img, title in zip(axes[0], (raw, ref, cleaned), ("brut", "référence", "nettoyé")):
        lo, hi = np.percentile(img, [0.5, 99.5])
        im = ax.imshow(img, cmap='gray', vmin=lo, vmax=hi)
        ax.set_title(f"Hologramme {title}")
        fig.colorbar(im, ax=ax, fraction=0.046)
    for ax, img, title in zip(axes[1, :2], (raw, cleaned), ("brut", "nettoyé")):
        ax.hist(img.ravel(), bins=256, log=True)
        ax.set_title(f"Histogramme {title}")
    ax = axes[1, 2]
    for (freq, psd), label in zip(spectra, ("brut", "référence", "nettoyé")):
        ax.semilogy(freq[1:], psd[1:] / psd[1:].max(), label=label)
    ax.set_xlabel("fréquence (cycles/µm)")
    ax.set_title("Spectre radial (normalisé)")
    ax.legend()
    save_fig(fig, out_dir, "hologram.png")


def fig_volume(out_dir, planes_mod, planes_foc, sub_mod, sub_foc, z_um, threshold):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, planes, title in ((axes[0, 0], planes_mod, "Module propagé"), (axes[0, 1], planes_foc, "Focus")):
        for key in ('mean', 'median', 'p99', 'max'):
            ax.plot(z_um, planes[key], label=key)
        ax.fill_between(z_um, planes['mean'] - planes['std'], planes['mean'] + planes['std'], alpha=0.2, label='mean ± std')
        ax.set_xlabel("z (µm)")
        ax.set_title(f"{title} par plan")
        ax.legend()
    axes[0, 1].axhline(threshold, color='r', ls='--', label='seuil')
    axes[0, 1].set_yscale('log')
    for ax, sub, title in ((axes[1, 0], sub_mod, "Module"), (axes[1, 1], sub_foc, "Focus")):
        ax.hist(sub.ravel(), bins=500, log=True)
        ax.set_title(f"Histogramme {title} (sous-échantillon)")
    axes[1, 1].axvline(threshold, color='r', ls='--')
    save_fig(fig, out_dir, "volume.png")


def fig_projections(out_dir, mip_mod, mip_foc, bin_xy, bin_xz, obj, dx_um, dz_um):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    for ax, img, title in ((axes[0, 0], mip_mod[0], "Module MIP XY"), (axes[0, 1], mip_foc[0], "Focus MIP XY")):
        lo, hi = np.percentile(img, [1, 99.9])
        ax.imshow(img, cmap='gray', vmin=lo, vmax=hi)
        ax.set_title(title)
    axes[0, 1].contour(bin_xy, levels=[0.5], colors='r', linewidths=0.5)
    if obj:
        axes[0, 1].plot(obj['bary_x_um'] / dx_um, obj['bary_y_um'] / dx_um, 'c+', ms=4)
    for ax, img, title in ((axes[1, 0], mip_mod[1], "Module MIP XZ"), (axes[1, 1], mip_foc[1], "Focus MIP XZ")):
        lo, hi = np.percentile(img, [1, 99.9])
        ax.imshow(img, cmap='gray', vmin=lo, vmax=hi, aspect=dz_um / dx_um)
        ax.set_title(title)
    axes[1, 1].contour(bin_xz, levels=[0.5], colors='r', linewidths=0.5)
    save_fig(fig, out_dir, "projections.png")


def fig_objects(out_dir, obj):
    keys = [('nb_vox', True), ('focus_peak', True), ('peak_snr_robust', True),
            ('extent_x_um', False), ('extent_z_um', False), ('bary_z_um', False),
            ('nearest_neighbor_um', False), ('focus_sum', True)]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, (key, logx) in zip(axes.ravel(), keys):
        x = np.asarray(obj[key], dtype=np.float64)
        x = x[np.isfinite(x) & ((x > 0) if logx else True)]
        if x.size:
            bins = np.logspace(np.log10(x.min()), np.log10(x.max()) + 1e-9, 40) if logx else 40
            ax.hist(x, bins=bins)
            if logx:
                ax.set_xscale('log')
        ax.set_title(key)
    save_fig(fig, out_dir, "objects.png")


# =============================================================================
# PRINCIPAL
# =============================================================================

# Paramètres attendus dans le YAML : section -> {clé: (type, valeurs autorisées ou None)}
FOCUS_TYPES = ['TENEGRAD', 'SUM_OF_VARIANCE', 'SUM_OF_LAPLACIAN', 'SUM_OF_INTENSITY',
               'SUM_OF_GRADIENT', 'MEAN_ALL', 'MEAN_LOG_ALL',
               'VAR_IMAG_OVER_REAL']   # types gérés par libs.focus ; le dernier exige le champ complexe
CONFIG_SPEC = {
    'inputs': {'hologram': (str, None), 'reference': (str, None), 'output_dir': (str, None)},
    'cleaning': {'type': (str, ['division', 'subtraction', 'none']), 'division_power': (float, None),
                 'reference_mean': (str, ['arithmetic', 'geometric'])},
    'optics': {'wavelength': (float, None), 'medium_index': (float, None), 'pixel_size': (float, None),
               'magnification': (float, None), 'holo_size_x': (int, None), 'holo_size_y': (int, None)},
    'propagation': {'distance_ini': (float, None), 'nb_planes': (int, None), 'step': (float, None),
                    'high_pass': (int, None), 'low_pass': (int, None)},
    'detection': {'focus_type': (str, FOCUS_TYPES), 'sum_size': (int, None), 'nb_std_threshold': (float, None),
                  'connectivity': (int, [6, 18, 26]), 'min_voxel': (int, None), 'max_voxel': (int, None)},
    'analysis': {'envelope_sigma': (float, None), 'subsample_xy': (int, None)},
    'ground_truth': {'positions_csv': (str, None), 'z_offset': (float, None),
                     'tolerance_xy': (float, None), 'tolerance_z': (float, None)},
}
# Paramètres facultatifs (absents des anciens YAML) : valeur par défaut
OPTIONAL_DEFAULTS = {'reference_mean': 'arithmetic', 'positions_csv': None, 'z_offset': 0.0,
                     'tolerance_xy': 2e-6, 'tolerance_z': 5e-6}


# =============================================================================
# SIMULATION : PARAMÈTRES ET VÉRITÉ TERRAIN
# =============================================================================
# Correspondance des axes, vérifiée sur une simulation en mode liste (bactéries placées à des
# positions asymétriques connues) :
#   - le simulateur remplit le masque en [x, y, z] et propage chaque plan tel quel : le x
#     simulé devient la LIGNE de l'image, le y simulé la COLONNE ;
#   - l'analyse (comme HOLOTRACKER) prend x = colonne, y = ligne.
#   => x_image = y_simulé, y_image = x_simulé.
#   - z : le plan simulé pos_z est reconstruit à la distance pos_z + distance_volume_camera
#     de l'hologramme (le flip Z du simulateur est correct).

def load_simulation_parameters(sim_dir):
    """
    Paramètres d'une simulation, depuis son dossier de sortie :
      - GUI    : parameters_simu_bact.json
      - script : config_simu.json (écrit par main_simu_hologram.py)
    Retourne un dict homogène (unités SI) ou None si aucun fichier n'est trouvé.
    """
    for name, keys in (("parameters_simu_bact.json",
                        dict(wavelength='wavelength', index='index_milieu', pixel='pix_size', mag='grossissement',
                             size='holo_size_xy', nz='z_size', step='step_z', dvc='distance_volume_camera')),
                       ("config_simu.json",
                        dict(wavelength='wavelength', index='index_medium', pixel='pix_size', mag='magnification',
                             size='holo_size_xy', nz='z_size', step='z_step', dvc='distance_volume_camera'))):
        path = os.path.join(sim_dir, name)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            defaults = dict(step=0.5e-6, dvc=0.0)
            out = {k: raw.get(v, defaults.get(k)) for k, v in keys.items()}
            out['source'] = path
            return out
    return None


def list_simulated_holograms(sim_dir):
    """
    Hologrammes d'un dossier de simulation : {identifiant: {'formats': {ext: chemin},
    'reference': {ext: chemin}, 'positions': chemin CSV ou None}}.
    Identifiant : GUI « 0 », script « 0_az0p00_pol0p00 ».
    """
    holo_dir = os.path.join(sim_dir, "simulated_hologram")
    if not os.path.isdir(holo_dir):
        return {}
    out = {}
    for f in sorted(os.listdir(holo_dir)):
        stem, ext = os.path.splitext(f)
        if not stem.startswith("holo_") or ext.lower() not in ('.tiff', '.tif', '.npy', '.bmp'):
            continue
        hid = stem[len("holo_"):]
        entry = out.setdefault(hid, {'formats': {}, 'reference': {}, 'positions': None})
        entry['formats'][ext.lower()] = os.path.join(holo_dir, f)
    for hid, entry in out.items():
        ref_dir = os.path.join(sim_dir, "holo_ref")
        for ext in ('.tiff', '.tif', '.npy', '.bmp'):
            path = os.path.join(ref_dir, f"holo_ref_{hid}{ext}")
            if os.path.exists(path):
                entry['reference'][ext] = path
        for prefix in ("bact_", "bacteria_", "sphere_"):
            path = os.path.join(sim_dir, "object_positions", f"{prefix}{hid}.csv")
            if os.path.exists(path):
                entry['positions'] = path
                break
    return out


def load_ground_truth(csv_path, z_offset_um):
    """
    Positions simulées (CSV du simulateur : x/y/z_position_m, + orientation et taille des
    bactéries ou rayon des sphères), converties dans le repère de l'analyse, en µm :
    x_image = y_simulé, y_image = x_simulé, z = z_simulé + z_offset_um (distance volume-caméra).
    """
    data = np.genfromtxt(csv_path, delimiter=',', names=True, dtype=None, encoding='utf-8')
    data = np.atleast_1d(data)
    names = data.dtype.names
    truth = {
        'x_um': 1e6 * data['y_position_m'].astype(np.float64),
        'y_um': 1e6 * data['x_position_m'].astype(np.float64),
        'z_um': 1e6 * data['z_position_m'].astype(np.float64) + z_offset_um,
    }
    for col in ('theta_angle', 'phi_angle', 'length', 'thickness', 'radius'):
        if col in names:
            truth[col] = data[col].astype(np.float64)
    return truth


def match_ground_truth(obj, truth, tol_xy_um, tol_z_um, field_um, z_range_um):
    """
    Appariement un-à-un détections ↔ vérité par l'algorithme hongrois, sur la distance
    normalisée d = sqrt((dxy / tol_xy)² + (dz / tol_z)²), paire acceptée si d <= 1.
    Détections non appariées mais à moins de la tolérance d'un objet apparié : doublons
    (ex. une bactérie verticale donne deux blocs de part et d'autre de son centre en z).
    Objets simulés hors du champ ou de la plage z reconstruite : exclus du rappel.
    """
    from scipy.optimize import linear_sum_assignment

    n_t = truth['x_um'].size
    n_d = obj['label'].size if obj else 0
    in_volume = ((truth['x_um'] >= 0) & (truth['x_um'] < field_um[0]) & (truth['y_um'] >= 0) &
                 (truth['y_um'] < field_um[1]) & (truth['z_um'] >= z_range_um[0]) & (truth['z_um'] <= z_range_um[1]))
    truth_match = np.full(n_t, -1)          # indice de la détection appariée
    det_match = np.full(n_d, -1)            # indice de l'objet simulé apparié
    det_status = np.full(n_d, 'faux positif', dtype=object)
    if n_d and n_t:
        dxy = np.hypot(obj['bary_x_um'][:, None] - truth['x_um'][None, :],
                       obj['bary_y_um'][:, None] - truth['y_um'][None, :])
        dz = obj['bary_z_um'][:, None] - truth['z_um'][None, :]
        cost = np.hypot(dxy / tol_xy_um, dz / tol_z_um)
        rows, cols = linear_sum_assignment(np.where(cost <= 1.0, cost, 1e6))
        for r, c in zip(rows, cols):
            if cost[r, c] <= 1.0:
                det_match[r], truth_match[c] = c, r
                det_status[r] = 'apparié'
        matched_truth = truth_match >= 0
        for r in np.flatnonzero(det_match < 0):
            near = (cost[r] <= 1.0) & matched_truth
            if near.any():
                det_status[r] = 'doublon'
                det_match[r] = int(np.flatnonzero(near)[np.argmin(cost[r][near])])

    tp = int((truth_match >= 0).sum())
    fn = int(((truth_match < 0) & in_volume).sum())
    fp = int((det_status == 'faux positif').sum())
    dup = int((det_status == 'doublon').sum())
    metrics = {
        'nb_simulated': n_t, 'nb_simulated_in_volume': int(in_volume.sum()),
        'nb_detected': n_d, 'true_positives': tp, 'false_negatives': fn, 'false_positives': fp,
        'duplicates': dup,
        'recall': tp / max(int(in_volume.sum()), 1),
        'precision': tp / max(tp + fp, 1),
        'precision_strict': tp / max(n_d, 1),
        'tolerance_xy_um': tol_xy_um, 'tolerance_z_um': tol_z_um,
    }
    metrics['f1'] = (2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall'])
                     if metrics['precision'] + metrics['recall'] > 0 else 0.0)
    if tp:
        i_det = truth_match[truth_match >= 0]
        i_tru = np.flatnonzero(truth_match >= 0)
        err = {'x': obj['bary_x_um'][i_det] - truth['x_um'][i_tru],
               'y': obj['bary_y_um'][i_det] - truth['y_um'][i_tru],
               'z': obj['bary_z_um'][i_det] - truth['z_um'][i_tru]}
        for k, e in err.items():
            metrics[f'error_{k}_mean_um'] = float(e.mean())
            metrics[f'error_{k}_median_um'] = float(np.median(e))
            metrics[f'error_{k}_std_um'] = float(e.std())
        metrics['error_xy_rms_um'] = float(np.sqrt(np.mean(err['x'] ** 2 + err['y'] ** 2)))
        metrics['error_z_rms_um'] = float(np.sqrt(np.mean(err['z'] ** 2)))
    if 'phi_angle' in truth:
        for lo, hi in ((0, 30), (30, 60), (60, 90.001)):
            sel = in_volume & (truth['phi_angle'] >= lo) & (truth['phi_angle'] < hi)
            metrics[f'recall_phi_{lo}_{int(hi)}'] = float((truth_match[sel] >= 0).mean()) if sel.any() else None
    return {'truth_match': truth_match, 'det_match': det_match, 'det_status': det_status,
            'in_volume': in_volume, 'metrics': metrics}


def save_matching_csv(out_dir, truth, match, obj):
    with open(os.path.join(out_dir, "ground_truth_matching.csv"), 'w') as f:
        extra = [c for c in ('theta_angle', 'phi_angle', 'length', 'thickness', 'radius') if c in truth]
        f.write("truth_index,x_um,y_um,z_um," + ",".join(extra) + ",in_volume,detected_label,"
                "det_x_um,det_y_um,det_z_um,det_nb_vox\n")
        for i in range(truth['x_um'].size):
            j = match['truth_match'][i]
            det = (f"{obj['label'][j]},{obj['bary_x_um'][j]:.4f},{obj['bary_y_um'][j]:.4f},"
                   f"{obj['bary_z_um'][j]:.4f},{obj['nb_vox'][j]}") if j >= 0 else ",,,,"
            f.write(f"{i},{truth['x_um'][i]:.4f},{truth['y_um'][i]:.4f},{truth['z_um'][i]:.4f},"
                    + "".join(f"{truth[c][i]:.6g}," for c in extra)
                    + f"{int(match['in_volume'][i])},{det}\n")


def validate_config(raw, base_dir):
    """
    Valide un dictionnaire de configuration (structure du YAML). Tous les paramètres sont
    obligatoires. Les chemins relatifs sont résolus par rapport à base_dir.
    Retourne un SimpleNamespace à plat ; lève ValueError avec la liste de toutes les erreurs.
    """
    from types import SimpleNamespace

    errors, flat = [], {}
    for section, keys in CONFIG_SPEC.items():
        values = raw.get(section) or {}
        for key, (typ, choices) in keys.items():
            if key not in values:
                if key in OPTIONAL_DEFAULTS:
                    flat[key] = OPTIONAL_DEFAULTS[key]
                else:
                    errors.append(f"{section}.{key} manquant")
                continue
            v = values[key]
            if key in ('output_dir', 'positions_csv') and (v is None or str(v).strip() in ('', 'null', 'None')):
                flat[key] = None
                continue
            try:
                v = typ(v)   # PyYAML lit "660e-9" (sans point) comme une chaîne : conversion explicite
            except (TypeError, ValueError):
                errors.append(f"{section}.{key} = {values[key]!r} : {typ.__name__} attendu")
                continue
            if choices is not None and v not in choices:
                errors.append(f"{section}.{key} = {v!r} : valeurs possibles {choices}")
                continue
            flat[key] = v
        for key in set(values) - set(keys):
            errors.append(f"{section}.{key} : paramètre inconnu")
    for section in set(raw) - set(CONFIG_SPEC):
        errors.append(f"section inconnue : {section}")
    if errors:
        raise ValueError("Fichier de configuration invalide :\n  - " + "\n  - ".join(errors))

    for key in ('hologram', 'reference', 'output_dir', 'positions_csv'):
        if flat[key] is not None and not os.path.isabs(flat[key]):
            flat[key] = os.path.normpath(os.path.join(base_dir, flat[key]))
    flat['cleaning'] = flat.pop('type')
    flat['out'] = flat.pop('output_dir')
    return SimpleNamespace(**flat)


def load_config(yaml_path):
    """Lit et valide le fichier YAML (chemins relatifs : par rapport au dossier du YAML)."""
    import yaml
    with open(yaml_path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f) or {}
    return validate_config(raw, os.path.dirname(os.path.abspath(yaml_path)))


def extract_object_crops(d_focus, obj, dx_um, dz_um, half_xy=24, half_z=40, max_objects=5000, z0_um=0.0):
    """
    Coupes du volume de focus autour de chaque objet, pour la visualisation :
    XY au plan du pic, XZ passant par le barycentre. Tailles fixes (bords complétés par 0).
    """
    nz, ny, nx = d_focus.shape
    n = min(obj['label'].size, max_objects)
    xy = np.zeros((n, 2 * half_xy, 2 * half_xy), dtype=np.float32)
    xz = np.zeros((n, 2 * half_z, 2 * half_xy), dtype=np.float32)
    for i in range(n):
        cx = int(round(obj['bary_x_um'][i] / dx_um))
        cy = int(round(obj['bary_y_um'][i] / dx_um))
        cz = int(round((obj['z_peak_um'][i] - z0_um) / dz_um))
        x0, x1 = max(cx - half_xy, 0), min(cx + half_xy, nx)
        y0, y1 = max(cy - half_xy, 0), min(cy + half_xy, ny)
        z0, z1 = max(cz - half_z, 0), min(cz + half_z, nz)
        ox, oy, oz = x0 - (cx - half_xy), y0 - (cy - half_xy), z0 - (cz - half_z)
        xy[i, oy:oy + y1 - y0, ox:ox + x1 - x0] = cp.asnumpy(d_focus[min(max(cz, 0), nz - 1), y0:y1, x0:x1])
        xz[i, oz:oz + z1 - z0, ox:ox + x1 - x0] = cp.asnumpy(d_focus[z0:z1, min(max(cy, 0), ny - 1), x0:x1])
    return {'xy': xy, 'xz': xz, 'half_xy': half_xy, 'half_z': half_z}


def run_analysis(args, config_path=None, config_text=None, log=print):
    """
    Chaîne complète : lecture, nettoyage, propagation, focus, seuil, CCL3D, statistiques.
    Écrit les sorties dans le dossier de résultats et retourne tout ce qu'il faut pour
    afficher les résultats (images, profils, objets, coupes, résumé).

    config_path : YAML copié dans le dossier de sortie ; sinon config_text y est écrit
    sous le nom holo_stats_config.yaml.
    """
    import shutil
    t0 = time.perf_counter()

    nx, ny, nz = args.holo_size_x, args.holo_size_y, args.nb_planes
    medium_wavelength = args.wavelength / args.medium_index
    dx_um = 1e6 * args.pixel_size / args.magnification
    dz_um = 1e6 * args.step
    z_um = 1e6 * args.distance_ini + dz_um * np.arange(nz)

    out_dir = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.hologram)),
        f"{os.path.splitext(os.path.basename(args.hologram))[0]}_stats_{datetime.datetime.now():%Y_%m_%d_%H_%M_%S}")
    os.makedirs(out_dir, exist_ok=True)
    if config_path:
        shutil.copy2(config_path, os.path.join(out_dir, os.path.basename(config_path)))
    elif config_text:
        with open(os.path.join(out_dir, "holo_stats_config.yaml"), 'w', encoding='utf-8') as f:
            f.write(config_text)

    log("=" * 70)
    log("STATISTIQUES HOLOGRAMME / VOLUME / OBJETS")
    log("=" * 70)
    if config_path:
        log(f"Config     : {config_path}")
    log(f"Hologramme : {args.hologram}")
    log(f"Référence  : {args.reference}")
    log(f"Nettoyage  : {args.cleaning}")
    log(f"Sortie     : {out_dir}")

    # ---- 1. Hologrammes --------------------------------------------------------
    log("\n[1/4] Hologrammes...")
    raw = load_image(args.hologram, nx, ny)
    ref, n_ref_images = load_reference(args.reference, nx, ny, os.path.splitext(args.hologram)[1],
                                       getattr(args, 'reference_mean', 'arithmetic'))
    if n_ref_images > 1:
        log(f"  Référence = moyenne de {n_ref_images} images")
    ref_warnings = check_reference(raw, ref)
    for w in ref_warnings:
        log(f"  ATTENTION : {w}")
    cleaned = clean_hologram(raw, ref, args.cleaning, args.division_power)

    s_raw, spec_raw = hologram_stats(raw, dx_um, args.envelope_sigma)
    s_ref, spec_ref = hologram_stats(ref, dx_um, args.envelope_sigma)
    s_clean, spec_clean = hologram_stats(cleaned, dx_um, args.envelope_sigma)
    s_raw['mean_ratio_over_reference'] = float(raw.mean() / ref.mean())
    s_raw['correlation_with_reference'] = float(np.corrcoef(raw.ravel().astype(np.float64), ref.ravel())[0, 1])
    fig_holograms(out_dir, raw, ref, cleaned, (spec_raw, spec_ref, spec_clean), dx_um)
    np.savetxt(os.path.join(out_dir, "spectrum.csv"),
               np.column_stack([spec_raw[0], spec_raw[1], spec_ref[1], spec_clean[1]]),
               delimiter=',', header="freq_cycles_per_um,psd_raw,psd_reference,psd_cleaned", comments='')

    # ---- 2. Volume -------------------------------------------------------------
    log("[2/4] Propagation du volume...")
    # libs/propagation.py est aligné sur HOLOTRACKER (filtre spectral compris) : vérifié par
    # comparaison avec le code de HOLOTRACKER (même seuil, mêmes objets, mêmes voxels)
    d_volume = propagate_module(cleaned, args, medium_wavelength, nx, ny, nz)

    s_mod, planes_mod, sub_mod = volume_stats(d_volume, args.subsample_xy)
    s_norm = normalized_volume_stats(d_volume, s_mod['mean_gpu_full'], s_mod['std_gpu_full'], args.subsample_xy)
    mip_mod = (cp.asnumpy(d_volume.max(axis=0)), cp.asnumpy(d_volume.max(axis=1)), cp.asnumpy(d_volume.max(axis=2)))

    log("[3/4] Focus et détection...")
    if Focus_type[args.focus_type] in focus.COMPLEX_FOCUS_TYPES:
        # critère sur le champ complexe : le volume du module ne sert plus, on recalcule
        del d_volume
        cp.get_default_memory_pool().free_all_blocks()
        d_volume = focus_volume_complex(cleaned, args, medium_wavelength, nx, ny, nz, args.sum_size,
                                        Focus_type[args.focus_type])
    else:
        focus.focus(d_volume, d_volume, args.sum_size, Focus_type[args.focus_type])   # en place, comme HOLOTRACKER
    s_foc, planes_foc, sub_foc = volume_stats(d_volume, args.subsample_xy)
    mip_foc = (cp.asnumpy(d_volume.max(axis=0)), cp.asnumpy(d_volume.max(axis=1)), cp.asnumpy(d_volume.max(axis=2)))

    threshold = calc_threshold(d_volume, args.nb_std_threshold)
    bg_median, bg_sigma = s_foc['median'], s_foc['robust_std']
    s_foc['threshold'] = threshold
    s_foc['threshold_in_robust_sigma_above_median'] = (threshold - bg_median) / bg_sigma if bg_sigma > 0 else None

    d_bin = cp.zeros((nz, ny, nx), dtype=cp.bool_)
    d_labels, n_labels = CCL3D(d_bin, d_volume, type_threshold.THRESHOLD, threshold, args.connectivity)
    bin_cpu = cp.asnumpy(d_bin)
    s_foc['voxels_above_threshold'] = int(bin_cpu.sum())
    s_foc['fraction_above_threshold'] = s_foc['voxels_above_threshold'] / bin_cpu.size

    # ---- 3. Objets -------------------------------------------------------------
    log(f"[4/4] Analyse de {n_labels} objets...")
    obj = object_stats(d_labels, d_volume, bin_cpu, dx_um, dz_um, bg_median, bg_sigma, z0_um=z_um[0],
                       d_idx=cp.nonzero(d_bin))

    # contrôle croisé avec le calcul HOLOTRACKER (CCA_CUDA_float) : nombre de voxels et barycentres
    cca_check = None
    if n_labels > 0:
        feats = CCA_CUDA_float(d_labels, d_volume, n_labels, 0, nx, ny, nz, dx_um, dx_um, dz_um, z_offset=z_um[0])
        f = feats[obj['label'] - 1]
        cca_check = {
            'nb_objects_cca': int(n_labels),
            'total_voxels_cca': int(feats[:, 4].sum()),
            'max_barycenter_diff_um': float(np.max(np.abs(np.column_stack(
                [obj['bary_x_um'] - f[:, 1], obj['bary_y_um'] - f[:, 2], obj['bary_z_um'] - f[:, 3]])))),
        }
    if obj and (args.min_voxel > 0 or args.max_voxel > 0):
        keep = np.ones(obj['label'].size, dtype=bool)
        if args.min_voxel > 0:
            keep &= obj['nb_vox'] >= args.min_voxel
        if args.max_voxel > 0:
            keep &= obj['nb_vox'] <= args.max_voxel
        obj = {k: v[keep] for k, v in obj.items()}
        # le contour affiché ne retient que les objets conservés
        bin_cpu = np.isin(cp.asnumpy(d_labels), obj['label'])

    crops = extract_object_crops(d_volume, obj, dx_um, dz_um, z0_um=z_um[0]) if obj else None

    # Module propagé aux voxels des objets. Le volume du module a été remplacé par le focus
    # (calcul en place, comme HOLOTRACKER) : il est recalculé ici, à l'identique.
    profiles, s_norm_obj = None, None
    if obj:
        zi, yi, xi = np.nonzero(bin_cpu)
        d_idx = (cp.asarray(zi), cp.asarray(yi), cp.asarray(xi))
        vox_label = cp.asnumpy(d_labels[d_idx]).astype(np.int64)
        profiles = {'focus': object_z_profiles(d_volume, obj, dx_um), 'half_window_px': 2}
        del d_volume, d_labels, d_bin
        d_labels = d_bin = None
        cp.get_default_memory_pool().free_all_blocks()
        d_module = propagate_module(cleaned, args, medium_wavelength, nx, ny, nz)
        profiles['module'] = object_z_profiles(d_module, obj, dx_um)
        vox_norm = ((cp.asnumpy(d_module[d_idx]).astype(np.float64) - s_mod['mean_gpu_full'])
                    / (s_mod['std_gpu_full'] + NORM_EPS))
        del d_module, d_idx
        cp.get_default_memory_pool().free_all_blocks()
        s_norm_obj = basic_stats(vox_norm)
        s_norm_obj['nb_voxels'] = int(vox_norm.size)
        for k in (3, 5, 10):
            s_norm_obj[f'fraction_above_{k}sigma'] = float((vox_norm > k).mean())
        # par objet : pic et moyenne du module normalisé
        order = np.argsort(vox_label, kind='stable')
        lab_s, norm_s = vox_label[order], vox_norm[order]
        starts = np.flatnonzero(np.r_[True, lab_s[1:] != lab_s[:-1]])
        pos = np.searchsorted(lab_s[starts], obj['label'])
        obj['module_norm_peak'] = np.maximum.reduceat(norm_s, starts)[pos]
        obj['module_norm_mean'] = (np.add.reduceat(norm_s, starts) / np.diff(np.r_[starts, lab_s.size]))[pos]
        np.savez(os.path.join(out_dir, "object_z_profiles.npz"), z_um=z_um, label=obj['label'],
                 module=profiles['module'], focus=profiles['focus'], half_window_px=2)

    truth, match = None, None
    if getattr(args, 'positions_csv', None):
        truth = load_ground_truth(args.positions_csv, 1e6 * args.z_offset)
        match = match_ground_truth(obj, truth, 1e6 * args.tolerance_xy, 1e6 * args.tolerance_z,
                                   (nx * dx_um, ny * dx_um), (z_um[0], z_um[-1]))
        save_matching_csv(out_dir, truth, match, obj)
        m = match['metrics']
        log(f"  Vérité terrain : {m['true_positives']}/{m['nb_simulated_in_volume']} objets simulés retrouvés "
            f"(rappel {m['recall']:.2f}, précision {m['precision']:.2f}, {m['duplicates']} doublons, "
            f"{m['false_positives']} faux positifs)")
    bin_proj = (bin_cpu.any(axis=0), bin_cpu.any(axis=1), bin_cpu.any(axis=2))
    fig_volume(out_dir, planes_mod, planes_foc, sub_mod, sub_foc, z_um, threshold)
    fig_projections(out_dir, mip_mod, mip_foc, bin_proj[0], bin_proj[1], obj, dx_um, dz_um)
    hist_mod = np.histogram(sub_mod, bins=500)
    hist_foc = np.histogram(sub_foc, bins=500)
    d_volume = None
    del d_volume, d_bin, d_labels, bin_cpu, sub_mod, sub_foc
    cp.get_default_memory_pool().free_all_blocks()

    with open(os.path.join(out_dir, "planes.csv"), 'w') as f:
        f.write("plane,z_um," + ",".join(f"module_{k}" for k in planes_mod) + "," +
                ",".join(f"focus_{k}" for k in planes_foc) + "\n")
        for i in range(nz):
            f.write(f"{i},{z_um[i]:.4f}," + ",".join(f"{planes_mod[k][i]:.6g}" for k in planes_mod) + "," +
                    ",".join(f"{planes_foc[k][i]:.6g}" for k in planes_foc) + "\n")

    n_obj = int(obj['label'].size) if obj else 0
    if obj:
        with open(os.path.join(out_dir, "objects.csv"), 'w') as f:
            keys = list(obj.keys())
            f.write(",".join(keys) + "\n")
            for i in range(n_obj):
                f.write(",".join(f"{obj[k][i]:.6g}" for k in keys) + "\n")
        fig_objects(out_dir, obj)

    volume_um3 = (nx * dx_um) * (ny * dx_um) * (nz * dz_um)
    summary = {
        'config': config_path,
        'inputs': {'hologram': os.path.abspath(args.hologram), 'reference': os.path.abspath(args.reference),
                   'reference_nb_images': n_ref_images, 'reference_warnings': ref_warnings},
        'parameters': vars(args),
        'voxel_um': {'dx': dx_um, 'dy': dx_um, 'dz': dz_um},
        'hologram': {'raw': s_raw, 'reference': s_ref, 'cleaned': s_clean},
        'volume': {'module': s_mod, 'module_normalized': s_norm,
                   'module_normalized_objects': s_norm_obj, 'focus': s_foc},
        'objects': {
            'count': n_obj,
            'count_before_voxel_filter': int(n_labels),
            'density_per_mm3': n_obj / volume_um3 * 1e9,
            **({k: distribution_summary(v) for k, v in obj.items() if k != 'label'} if obj else {}),
        },
        'cca_check': cca_check,
        'ground_truth': ({'positions_csv': args.positions_csv, **match['metrics']} if match else None),
        'processing_time_s': time.perf_counter() - t0,
    }
    with open(os.path.join(out_dir, "summary.json"), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log_summary(summary, log)
    log(f"Terminé en {summary['processing_time_s']:.1f} s → {out_dir}")

    return {
        'out_dir': out_dir, 'summary': summary,
        'raw': raw, 'reference': ref, 'cleaned': cleaned,
        'spectra': {'freq': spec_raw[0], 'raw': spec_raw[1], 'reference': spec_ref[1], 'cleaned': spec_clean[1]},
        'z_um': z_um, 'dx_um': dx_um, 'dz_um': dz_um, 'threshold': threshold,
        'planes_module': planes_mod, 'planes_focus': planes_foc,
        'hist_module': hist_mod, 'hist_focus': hist_foc,
        'mip_module': mip_mod, 'mip_focus': mip_foc, 'bin_proj': bin_proj,
        'objects': obj, 'crops': crops, 'truth': truth, 'match': match, 'profiles': profiles,
    }


def log_summary(summary, log=print):
    """Résumé texte des statistiques principales."""
    h = summary['hologram']
    s_mod, s_foc = summary['volume']['module'], summary['volume']['focus']
    o = summary['objects']
    log("\n" + "-" * 70)
    log(f"{'':28s}{'brut':>13s}{'référence':>13s}{'nettoyé':>13s}")
    for key in ('mean', 'std', 'min', 'max', 'contrast_std_over_mean', 'noise_std_laplacian',
                'noise_std_over_mean', 'envelope_variation_over_mean', 'distinct_levels'):
        log(f"{key:28s}" + "".join(f"{s[key]:13.4g}" if s.get(key) is not None else f"{'-':>13s}"
                                  for s in (h['raw'], h['reference'], h['cleaned'])))
    log("-" * 70)
    log(f"Module  : moyenne {s_mod['mean_gpu_full']:.4g}  écart-type {s_mod['std_gpu_full']:.4g}  max {s_mod['max_gpu_full']:.4g}")
    s_n, s_no = summary['volume'].get('module_normalized'), summary['volume'].get('module_normalized_objects')
    if s_n:
        log(f"Module normalisé (V-m)/(σ+ε) : min {s_n['min']:.3g}  max {s_n['max']:.3g}  p99.9 {s_n['p99.9']:.3g}  "
            f"> 5σ : {100 * s_n['fraction_above_5sigma']:.3g} %")
    if s_no:
        log(f"  voxels des objets : min {s_no['min']:.3g}  max {s_no['max']:.3g}  moyenne {s_no['mean']:.3g}  "
            f"médiane {s_no['median']:.3g}")
    log(f"Focus   : médiane {s_foc['median']:.4g}  σ robuste {s_foc['robust_std']:.4g}  max {s_foc['max_gpu_full']:.4g}")
    log(f"Seuil   : {s_foc['threshold']:.4g}  (= {s_foc['threshold_in_robust_sigma_above_median']:.1f} σ robustes au-dessus de la médiane)")
    log(f"Voxels segmentés : {s_foc['voxels_above_threshold']}  ({100 * s_foc['fraction_above_threshold']:.4f} %)")
    log(f"Objets  : {o['count']}  (avant filtre voxels : {o['count_before_voxel_filter']})  densité {o['density_per_mm3']:.3g} /mm³")
    for key in ('nb_vox', 'focus_peak', 'peak_snr_robust', 'extent_x_um', 'extent_z_um', 'nearest_neighbor_um'):
        s = o.get(key)
        if s:
            log(f"  {key:22s} médiane {s['p50']:10.4g}   p5 {s['p5']:10.4g}   p95 {s['p95']:10.4g}")
    log("-" * 70)


def main():
    if len(sys.argv) != 2:
        print("Utilisation : python stats/holo_stats.py CONFIG.yaml")
        return 1
    config_path = os.path.abspath(sys.argv[1])
    try:
        args = load_config(config_path)
    except (OSError, ValueError) as e:
        print(e)
        return 1
    run_analysis(args, config_path=config_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

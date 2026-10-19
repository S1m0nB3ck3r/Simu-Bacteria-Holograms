# -*- coding: utf-8 -*-

"""
Filename: pipeline_holotracker_locate_local_maxima.py

Description:
Pipeline de localisation par MAXIMA LOCAUX du volume de focus : un point par objet, au lieu de
seuil + composantes connexes (CLAUDE_HOLOTRACKER_TO_DO.md, § 2.2).

    candidat = (focus == max du focus dans une boîte FOOTPRINT centrée)  ET  (focus > T(x))

  - T(x) est le seuil local de pipeline_holotracker_locate_local_threshold.py
    (fond par maille avec écrêtage κσ, σ plancher, interpolation trilinéaire) ;
  - la boîte a la taille de la résolution mesurée (largeurs à mi-hauteur du focus : 3,85 µm en
    XY, 12 µm en Z) : un seul maximum par réponse, que la bactérie soit couchée ou debout ;
  - deux objets proches restent séparés tant qu'il reste un creux entre eux, là où CCL3D les
    fusionne en une seule composante ;
  - position affinée sous le voxel par interpolation parabolique sur les 3 axes ;
  - chaque objet est décrit par son pic et son rapport signal sur bruit local
    (pic − fond local) / max(σ local, σ plancher), et non par un nombre de voxels, qui dépend de
    l'orientation à la puissance 3.

Author: Simon BECKER
Date: 2026-10-12

License:
GNU General Public License v3.0

Utilisation :
python pipeline_holotracker_locate_local_maxima.py

Fichiers requis :
- simu_holo_test.bmp (hologramme d'entrée)

Fichiers générés :
- result_local_maxima.csv (positions des objets détectés)
"""

import cupy as cp
import numpy as np
import time
import os
import sys
from cupyx.scipy import ndimage as cp_ndimage

# Add the project root and this directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from libs.traitement_holo import *
from libs import propagation as propag
from libs import focus
from libs.focus import Focus_type
from libs.CCL3D import calc_threshold, connectivity_26
from pipeline_holotracker_locate_local_threshold import local_threshold_volume
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# PARAMETRES
# =============================================================================

# Fichiers d'entrée et de sortie
HOLOGRAM_FILE = os.path.join(os.path.dirname(__file__), "simu_holo_test.bmp")
RESULT_FILE = "./result_local_maxima.csv"

# Paramètres de l'hologramme
WAVELENGTH = 660e-9  # Longueur d'onde (m)
INDEX_MEDIUM = 1.33  # Indice du milieu
MAGNIFICATION = 40.0  # Grossissement
HOLO_SIZE_XY = 1024  # Taille image (pixels)
Z_SIZE = 200  # Nombre de plans Z
PIX_SIZE = 5.5e-6  # Taille pixel (m)
STEP_Z = 0.5e-6  # Pas en Z (m)
VOLUME_CAMERA_DISTANCE = 1e-5  # Distance volume-caméra (m) = distance du premier plan reconstruit

# Paramètres du focus
SUM_SIZE = 15  # Taille du noyau de focus
FILTER_LOW = 15  # Rayon min conservé dans le spectre (px) : coupe les basses fréquences
FILTER_HIGH = 125  # Rayon max conservé dans le spectre (px) : coupe les hautes fréquences
FOCUS_TYPE = Focus_type.TENEGRAD

# Boîte des maxima locaux, en µm (résolution mesurée du volume de focus) ; convertie en voxels
FOOTPRINT_XY_UM = 3.85
FOOTPRINT_Z_UM = 12.0

# Seuil local (voir pipeline_holotracker_locate_local_threshold.py)
K_LOCAL = 8.0  # 8 : meilleur rappel à précision ≥ 95 %, benchmark du 2026-10-12
MESH_XY = 64
MESH_Z = 25
CLIP_KAPPA = 3.0
CLIP_ITERATIONS = 3
SIGMA_FLOOR = None
SIGMA_FLOOR_PERCENTILE = 10

# Comparaison avec le seuil global + CCL de pipeline_holotracker_locate_simple.py
COMPARE_GLOBAL = True
NB_STDVAR_THRESHOLD = 10

# Options d'affichage
DISPLAY_IMAGES = True


# =============================================================================
# MAXIMA LOCAUX
# =============================================================================

def footprint_voxels(dx_um, dz_um, footprint_xy_um=FOOTPRINT_XY_UM, footprint_z_um=FOOTPRINT_Z_UM):
    """Taille de la boîte (Z, Y, X) en voxels, impaire pour être centrée."""
    odd = lambda v: max(3, int(round(v)) // 2 * 2 + 1)
    return odd(footprint_z_um / dz_um), odd(footprint_xy_um / dx_um), odd(footprint_xy_um / dx_um)


def local_maxima(d_focus_volume, d_threshold, footprint):
    """
    Indices (z, y, x) des maxima locaux au-dessus du seuil : voxels égaux au maximum de leur
    boîte (filtre max séparable, bords en 'nearest') et supérieurs au seuil local.
    """
    d_max = cp_ndimage.maximum_filter(d_focus_volume, size=footprint, mode='nearest')
    d_cand = (d_focus_volume == d_max) & (d_focus_volume > d_threshold)
    del d_max
    return cp.nonzero(d_cand)


def subvoxel_offsets(d_focus_volume, iz, iy, ix):
    """
    Décalage sous le voxel, axe par axe, par la parabole passant par le maximum et ses deux
    voisins : δ = (f₋ − f₊) / (2 (f₋ − 2 f₀ + f₊)), borné à ±0,5. Nul au bord du volume.
    """
    shape = d_focus_volume.shape
    f0 = d_focus_volume[iz, iy, ix]
    out = []
    for axis, idx in enumerate((iz, iy, ix)):
        lo, hi = cp.maximum(idx - 1, 0), cp.minimum(idx + 1, shape[axis] - 1)
        im = [iz, iy, ix]; ip = [iz, iy, ix]
        im[axis], ip[axis] = lo, hi
        fm, fp = d_focus_volume[tuple(im)], d_focus_volume[tuple(ip)]
        den = fm - 2 * f0 + fp
        d = cp.where((den < 0) & (lo != idx) & (hi != idx), 0.5 * (fm - fp) / cp.where(den == 0, 1, den), 0)
        out.append(cp.clip(d, -0.5, 0.5))
    return out


def detect_local_maxima(d_focus_volume, dx_um, dz_um, z_offset_um, k_local=K_LOCAL, mesh_z=MESH_Z, mesh_xy=MESH_XY,
                        sigma_floor=SIGMA_FLOOR, sigma_floor_percentile=SIGMA_FLOOR_PERCENTILE,
                        kappa=CLIP_KAPPA, iterations=CLIP_ITERATIONS,
                        footprint_xy_um=FOOTPRINT_XY_UM, footprint_z_um=FOOTPRINT_Z_UM):
    """
    Chaîne complète sur un volume de focus (Z, Y, X) : seuil local, maxima locaux, affinage.
    Retourne un DataFrame (baryX, baryY, baryZ en µm, peak, snr, ix, iy, iz) et les infos du seuil.
    """
    d_threshold, infos = local_threshold_volume(d_focus_volume, k_local, mesh_z, mesh_xy, sigma_floor,
                                                sigma_floor_percentile, kappa, iterations)
    fp = footprint_voxels(dx_um, dz_um, footprint_xy_um, footprint_z_um)
    iz, iy, ix = local_maxima(d_focus_volume, d_threshold, fp)
    del d_threshold
    if iz.size == 0:
        return pd.DataFrame(columns=['baryX', 'baryY', 'baryZ', 'peak', 'snr', 'ix', 'iy', 'iz']), infos, fp
    oz, oy, ox = subvoxel_offsets(d_focus_volume, iz, iy, ix)
    peak = d_focus_volume[iz, iy, ix]
    # fond et σ de la maille contenant le maximum
    bg = infos['background'][iz // mesh_z, iy // mesh_xy, ix // mesh_xy]
    sg = cp.maximum(infos['sigma'][iz // mesh_z, iy // mesh_xy, ix // mesh_xy], infos['sigma_floor'])
    df = pd.DataFrame({
        'baryX': cp.asnumpy((ix + ox) * dx_um), 'baryY': cp.asnumpy((iy + oy) * dx_um),
        'baryZ': cp.asnumpy(z_offset_um + (iz + oz) * dz_um),
        'peak': cp.asnumpy(peak), 'snr': cp.asnumpy((peak - bg) / sg),
        'ix': cp.asnumpy(ix), 'iy': cp.asnumpy(iy), 'iz': cp.asnumpy(iz)})
    return df.sort_values('snr', ascending=False).reset_index(drop=True), infos, fp


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def main():
    """Pipeline principal de traitement"""

    print("=" * 60)
    print("PIPELINE DE LOCALISATION D'HOLOGRAMMES — MAXIMA LOCAUX")
    print("=" * 60)
    print(f"Fichier d'entrée : {HOLOGRAM_FILE}")
    print(f"Fichier de sortie : {RESULT_FILE}")
    print()

    if not os.path.exists(HOLOGRAM_FILE):
        print(f"❌ ERREUR : Le fichier {HOLOGRAM_FILE} n'existe pas")
        return 1

    medium_wavelength = WAVELENGTH / INDEX_MEDIUM
    dx = 1e6 * PIX_SIZE / MAGNIFICATION  # en µm
    dz = STEP_Z * 1e6  # en µm
    fp = footprint_voxels(dx, dz)
    print("Paramètres calculés :")
    print(f"  - Résolution XY : {dx:.4f} µm/pixel, Z : {dz:.2f} µm/plan")
    print(f"  - Boîte des maxima locaux : {fp[0]} plans × {fp[1]} × {fp[2]} px "
          f"({fp[0]*dz:.1f} × {fp[1]*dx:.2f} × {fp[2]*dx:.2f} µm)")
    print()

    d_fft_holo = cp.zeros(shape=(HOLO_SIZE_XY, HOLO_SIZE_XY), dtype=cp.complex64)
    d_fft_holo_propag = cp.zeros(shape=(HOLO_SIZE_XY, HOLO_SIZE_XY), dtype=cp.complex64)
    d_KERNEL = cp.zeros(shape=(HOLO_SIZE_XY, HOLO_SIZE_XY), dtype=cp.complex64)
    d_volume_module = cp.zeros(shape=(Z_SIZE, HOLO_SIZE_XY, HOLO_SIZE_XY), dtype=cp.float32)

    print(f"📁 Chargement de l'hologramme : {HOLOGRAM_FILE}")
    ini_time = time.perf_counter()
    h_holo = read_image(HOLOGRAM_FILE, HOLO_SIZE_XY, HOLO_SIZE_XY)
    if DISPLAY_IMAGES:
        display(h_holo, "Hologramme d'entrée")
    d_holo = cp.asarray(h_holo)

    print("🔄 ÉTAPE 1 : Propagation du volume...")
    t1 = time.perf_counter()
    propag.volume_propag_angular_spectrum_to_module(
        d_holo, d_fft_holo, d_KERNEL, d_fft_holo_propag, d_volume_module,
        medium_wavelength, MAGNIFICATION, PIX_SIZE, HOLO_SIZE_XY, HOLO_SIZE_XY,
        VOLUME_CAMERA_DISTANCE, STEP_Z, Z_SIZE, FILTER_LOW, FILTER_HIGH)
    cp.cuda.Device().synchronize()
    t2 = time.perf_counter()

    print("🎯 ÉTAPE 2 : Calcul du focus...")
    if FOCUS_TYPE in focus.COMPLEX_FOCUS_TYPES:
        # critère sur le champ complexe (porteuse retirée), calculé plan par plan à la propagation
        propag.volume_propag_angular_spectrum_to_focus_complex(
            d_holo.astype(cp.complex64), d_volume_module, focus.focus_plane_var_imag_over_real, SUM_SIZE,
            medium_wavelength, MAGNIFICATION, PIX_SIZE, HOLO_SIZE_XY, HOLO_SIZE_XY,
            VOLUME_CAMERA_DISTANCE, STEP_Z, Z_SIZE, FILTER_LOW, FILTER_HIGH)
    else:
        focus.focus(d_volume_module, d_volume_module, SUM_SIZE, FOCUS_TYPE)
    cp.cuda.Device().synchronize()
    t3 = time.perf_counter()

    print("📍 ÉTAPE 3 : Seuil local et maxima locaux...")
    positions, infos, _ = detect_local_maxima(d_volume_module, dx, dz, VOLUME_CAMERA_DISTANCE * 1e6)
    cp.cuda.Device().synchronize()
    t4 = time.perf_counter()
    print(f"   σ plancher : {infos['sigma_floor']:.6g}")
    print(f"   Nombre d'objets détectés (maxima locaux) : {len(positions)}")

    if COMPARE_GLOBAL:
        threshold_global = calc_threshold(d_volume_module, NB_STDVAR_THRESHOLD)
        _, n_global = cp_ndimage.label(d_volume_module > threshold_global, connectivity_26)
        print(f"   Pour comparaison, seuil global (moyenne + {NB_STDVAR_THRESHOLD} σ) + CCL : {int(n_global)} objets")

    print(f"💾 ÉTAPE 4 : Sauvegarde vers {RESULT_FILE}")
    positions.insert(0, 'i_image', 1)
    positions.drop(columns=['ix', 'iy', 'iz']).to_csv(RESULT_FILE, index=False)
    t5 = time.perf_counter()

    print()
    print("⏱️  TEMPS DE TRAITEMENT :")
    print(f"   Propagation          : {t2-t1:.3f} s")
    print(f"   Focus                : {t3-t2:.3f} s")
    print(f"   Seuil + maxima locaux: {t4-t3:.3f} s")
    print(f"   Sauvegarde           : {t5-t4:.3f} s")
    print(f"   TOTAL                : {t5-ini_time:.3f} s")
    print()
    print("=" * 60)
    print(f"Objets détectés : {len(positions)}")
    if len(positions):
        print("Aperçu (5 objets de plus fort RSB) :")
        print(positions.drop(columns=['ix', 'iy', 'iz']).head().to_string(index=False))
    print("=" * 60)

    if DISPLAY_IMAGES and len(positions):
        mip = cp.asnumpy(d_volume_module.max(axis=0))
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(mip, cmap='gray', extent=[0, HOLO_SIZE_XY * dx, HOLO_SIZE_XY * dx, 0])
        plt.scatter(positions['baryX'], positions['baryY'], s=30, facecolors='none', edgecolors='r')
        plt.title('Projection max du focus + maxima locaux')
        plt.subplot(1, 2, 2, projection='3d')
        ax = plt.gca()
        ax.scatter3D(positions['baryX'], positions['baryY'], positions['baryZ'], c=positions['snr'], cmap='viridis')
        ax.set_xlabel('X (µm)'); ax.set_ylabel('Y (µm)'); ax.set_zlabel('Z (µm)')
        ax.set_title(f'Positions 3D ({len(positions)} objets, couleur = RSB)')
        plt.tight_layout()
        plt.show()
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)

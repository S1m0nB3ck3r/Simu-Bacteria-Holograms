# -*- coding: utf-8 -*-

"""
Filename: pipeline_holotracker_locate_local_threshold.py

Description:
Pipeline de localisation identique à pipeline_holotracker_locate_simple.py, sauf l'étape de
seuillage : le seuil global (moyenne + n·σ du volume entier) est remplacé par un seuil LOCAL

    T(x) = fond_local(x) + K · max(σ_local(x), σ_plancher)

Pourquoi : le seuil global se couple à la densité. Calculé sur le volume entier, objets compris,
il est multiplié par 7,3 entre 1 et 41 bactéries, et une bactérie faible disparaît dès qu'il y en
a 6 (CLAUDE_HOLOTRACKER_TO_DO.md, § 1.3 à 2.1). Un seuil local ne monte que là où le fond monte.

Estimation du fond local, à la manière de SExtractor (Bertin & Arnouts 1996) :
  1. le volume de focus est découpé en mailles de MESH_Z plans × MESH_XY × MESH_XY pixels ;
  2. dans chaque maille, moyenne et σ par écrêtage itératif à ±CLIP_KAPPA σ : les voxels des
     objets sont exclus, sinon un objet ferait monter son propre seuil et se masquerait lui-même ;
  3. σ_plancher : borne basse de σ, pour qu'une maille très calme ne donne pas un seuil
     minuscule rempli de bruit ; à calibrer une fois sur un volume sans objet (SIGMA_FLOOR),
     sinon estimé ici comme le percentile SIGMA_FLOOR_PERCENTILE des σ de mailles ;
  4. le seuil des mailles est interpolé en trilinéaire jusqu'à la résolution du volume.

Author: Simon BECKER
Date: 2026-10-12

License:
GNU General Public License v3.0

Utilisation :
python pipeline_holotracker_locate_local_threshold.py

Fichiers requis :
- simu_holo_test.bmp (hologramme d'entrée)

Fichiers générés :
- result_local_threshold.csv (positions des objets détectés)
"""

import cupy as cp
import numpy as np
import time
import os
import sys
from cupyx.scipy import ndimage as cp_ndimage

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from libs.traitement_holo import *
from libs import propagation as propag
from libs import focus
from libs.focus import Focus_type
from libs.CCL3D import *
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# PARAMETRES
# =============================================================================

# Fichiers d'entrée et de sortie
HOLOGRAM_FILE = os.path.join(os.path.dirname(__file__), "simu_holo_test.bmp")
RESULT_FILE = "./result_local_threshold.csv"

# Paramètres de l'hologramme
WAVELENGTH = 660e-9  # Longueur d'onde (m)
INDEX_MEDIUM = 1.33  # Indice du milieu
MAGNIFICATION = 40.0  # Grossissement
HOLO_SIZE_XY = 1024  # Taille image (pixels)
Z_SIZE = 200  # Nombre de plans Z
PIX_SIZE = 5.5e-6  # Taille pixel (m)
STEP_Z = 0.5e-6  # Pas en Z (m)
VOLUME_CAMERA_DISTANCE = 1e-5  # Distance volume-caméra (m) = distance du premier plan reconstruit

# Paramètres de détection
SUM_SIZE = 15  # Taille du noyau de focus
N_CONNECTIVITY = 26  # Connectivité 3D
FILTER_LOW = 15  # Rayon min conservé dans le spectre (px) : coupe les basses fréquences
FILTER_HIGH = 125  # Rayon max conservé dans le spectre (px) : coupe les hautes fréquences
FOCUS_TYPE = Focus_type.TENEGRAD

# Seuil local : T(x) = fond_local(x) + K · max(σ_local(x), σ_plancher)
K_LOCAL = 8.0  # Nombre de σ locaux au-dessus du fond local (8 : meilleur rappel à précision ≥ 95 %, benchmark du 2026-10-12)
MESH_XY = 64  # Côté d'une maille en pixels (64 px = 8,8 µm ; doit diviser HOLO_SIZE_XY)
MESH_Z = 25  # Épaisseur d'une maille en plans (25 plans = 12,5 µm ; doit diviser Z_SIZE)
CLIP_KAPPA = 3.0  # Écrêtage à ±CLIP_KAPPA σ pour exclure les objets de l'estimation du fond
CLIP_ITERATIONS = 3  # Nombre d'itérations d'écrêtage
SIGMA_FLOOR = None  # σ plancher calibré sur un volume sans objet ; None = estimé (ligne suivante)
SIGMA_FLOOR_PERCENTILE = 10  # Si SIGMA_FLOOR est None : percentile des σ de mailles pris comme plancher

# Comparaison avec le seuil global de pipeline_holotracker_locate_simple.py
COMPARE_GLOBAL = True
NB_STDVAR_THRESHOLD = 10  # Seuil global en nombre d'écart-types (comparaison seulement)

# Options d'affichage
DISPLAY_IMAGES = True  # Mettre à True pour voir les images intermédiaires
VERBOSE = True  # Affichage des informations


# =============================================================================
# SEUIL LOCAL
# =============================================================================

def mesh_background(d_focus_volume, mesh_z, mesh_xy, kappa=3.0, iterations=3):
    """
    Fond et σ par maille, avec écrêtage itératif à ±kappa σ (les objets sont exclus).
    d_focus_volume : (Z, Y, X) ; Z divisible par mesh_z, Y et X par mesh_xy.
    Retourne deux grilles (Z/mesh_z, Y/mesh_xy, X/mesh_xy) : fond et σ.
    Traitement par tranche de mailles en Z pour limiter la mémoire GPU.
    """
    nz, ny, nx = d_focus_volume.shape
    if nz % mesh_z or ny % mesh_xy or nx % mesh_xy:
        raise ValueError(f"volume {d_focus_volume.shape} non divisible par les mailles "
                         f"({mesh_z}, {mesh_xy}, {mesh_xy})")
    gz, gy, gx = nz // mesh_z, ny // mesh_xy, nx // mesh_xy
    bg = cp.zeros((gz, gy, gx), dtype=cp.float32)
    sig = cp.zeros((gz, gy, gx), dtype=cp.float32)
    for k in range(gz):
        # tranche de mailles -> (mesh_z, gy, mesh_xy, gx, mesh_xy) -> axes de maille regroupés
        slab = d_focus_volume[k * mesh_z:(k + 1) * mesh_z].reshape(mesh_z, gy, mesh_xy, gx, mesh_xy)
        slab = slab.transpose(1, 3, 0, 2, 4).reshape(gy, gx, -1)
        m = slab.mean(axis=2)
        s = slab.std(axis=2)
        for _ in range(iterations):
            # masque en float32 : une somme de booléens par axe passe par CUB, qui exige nvcc
            # sur cette machine (voir CLAUDE.md)
            keep = (cp.abs(slab - m[:, :, None]) < kappa * s[:, :, None]).astype(cp.float32)
            n = cp.maximum(keep.sum(axis=2), 1.0)
            m = (slab * keep).sum(axis=2) / n
            s = cp.sqrt((((slab - m[:, :, None]) ** 2) * keep).sum(axis=2) / n)
            del keep
        bg[k], sig[k] = m, s
        del slab
    return bg, sig


def local_threshold_volume(d_focus_volume, k_local, mesh_z, mesh_xy, sigma_floor=None,
                           sigma_floor_percentile=10, kappa=3.0, iterations=3):
    """
    Seuil local à la résolution du volume : T = fond + k_local · max(σ, σ_plancher), calculé
    sur les mailles puis interpolé en trilinéaire (grid_mode : les valeurs sont aux centres
    des mailles, mode 'nearest' au-delà des centres des mailles du bord).
    Retourne (T, infos) ; infos contient les grilles et le σ plancher utilisé.
    """
    bg, sig = mesh_background(d_focus_volume, mesh_z, mesh_xy, kappa, iterations)
    if sigma_floor is None:
        sigma_floor = float(cp.percentile(sig.ravel(), sigma_floor_percentile))
    t_mesh = bg + k_local * cp.maximum(sig, sigma_floor)
    d_threshold = cp_ndimage.zoom(t_mesh, (mesh_z, mesh_xy, mesh_xy), order=1,
                                  grid_mode=True, mode='nearest')
    infos = {'background': bg, 'sigma': sig, 'sigma_floor': sigma_floor, 'threshold_mesh': t_mesh}
    return d_threshold, infos


def detect_local_threshold(d_focus_volume, d_threshold, n_connectivity=26):
    """Binarisation voxel par voxel (focus > T(x)) puis étiquetage en composantes connexes."""
    pattern = {26: connectivity_26, 18: connectivity_18}.get(n_connectivity, connectivity_6)
    d_bin = d_focus_volume > d_threshold
    d_labels, number_of_labels = cp_ndimage.label(d_bin, pattern)
    return d_bin, d_labels, int(number_of_labels)


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def main():
    """Pipeline principal de traitement"""

    print("=" * 60)
    print("PIPELINE DE LOCALISATION D'HOLOGRAMMES — SEUIL LOCAL")
    print("=" * 60)
    print(f"Fichier d'entrée : {HOLOGRAM_FILE}")
    print(f"Fichier de sortie : {RESULT_FILE}")
    print()

    if not os.path.exists(HOLOGRAM_FILE):
        print(f"❌ ERREUR : Le fichier {HOLOGRAM_FILE} n'existe pas")
        return 1

    medium_wavelength = WAVELENGTH / INDEX_MEDIUM
    dx = 1000000 * PIX_SIZE / MAGNIFICATION  # en µm
    dy = 1000000 * PIX_SIZE / MAGNIFICATION  # en µm
    dz = STEP_Z * 1e6  # en µm

    print("Paramètres calculés :")
    print(f"  - Longueur d'onde dans le milieu : {medium_wavelength*1e9:.1f} nm")
    print(f"  - Résolution XY : {dx:.4f} µm/pixel")
    print(f"  - Résolution Z : {dz:.2f} µm/plan")
    print(f"  - Mailles du seuil local : {MESH_Z} plans × {MESH_XY} × {MESH_XY} px "
          f"({MESH_Z*dz:.1f} × {MESH_XY*dx:.1f} × {MESH_XY*dy:.1f} µm)")
    print()

    # Allocation mémoire
    d_fft_holo = cp.zeros(shape=(HOLO_SIZE_XY, HOLO_SIZE_XY), dtype=cp.complex64)
    d_fft_holo_propag = cp.zeros(shape=(HOLO_SIZE_XY, HOLO_SIZE_XY), dtype=cp.complex64)
    d_KERNEL = cp.zeros(shape=(HOLO_SIZE_XY, HOLO_SIZE_XY), dtype=cp.complex64)
    d_volume_module = cp.zeros(shape=(Z_SIZE, HOLO_SIZE_XY, HOLO_SIZE_XY), dtype=cp.float32)

    # Chargement de l'hologramme
    print(f"📁 Chargement de l'hologramme : {HOLOGRAM_FILE}")
    ini_time = time.perf_counter()
    h_holo = read_image(HOLOGRAM_FILE, HOLO_SIZE_XY, HOLO_SIZE_XY)
    if DISPLAY_IMAGES:
        display(h_holo, "Hologramme d'entrée")
    d_holo = cp.asarray(h_holo)

    # ÉTAPE 1 : Propagation du volume par méthode du spectre angulaire
    print("🔄 ÉTAPE 1 : Propagation du volume...")
    t1 = time.perf_counter()
    propag.volume_propag_angular_spectrum_to_module(
        d_holo, d_fft_holo, d_KERNEL,
        d_fft_holo_propag, d_volume_module,
        medium_wavelength, MAGNIFICATION, PIX_SIZE, HOLO_SIZE_XY, HOLO_SIZE_XY,
        VOLUME_CAMERA_DISTANCE, STEP_Z, Z_SIZE, FILTER_LOW, FILTER_HIGH
    )
    cp.cuda.Device().synchronize()
    t2 = time.perf_counter()

    # ÉTAPE 2 : Calcul du focus
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
    if DISPLAY_IMAGES:
        display(d_volume_module.max(axis=0), "max Projection XY - Après focus")

    # ÉTAPE 3 : Seuil local
    print("📐 ÉTAPE 3 : Seuil local...")
    d_threshold, infos = local_threshold_volume(
        d_volume_module, K_LOCAL, MESH_Z, MESH_XY, SIGMA_FLOOR, SIGMA_FLOOR_PERCENTILE,
        CLIP_KAPPA, CLIP_ITERATIONS)
    cp.cuda.Device().synchronize()
    t4 = time.perf_counter()
    t_mesh = cp.asnumpy(infos['threshold_mesh'])
    print(f"   σ plancher : {infos['sigma_floor']:.6g}"
          + ("  (estimé : percentile %g des σ de mailles)" % SIGMA_FLOOR_PERCENTILE if SIGMA_FLOOR is None else ""))
    print(f"   Seuil local : min {t_mesh.min():.6g}, médiane {np.median(t_mesh):.6g}, max {t_mesh.max():.6g}")

    # ÉTAPE 4 : Détection d'objets
    print("🔍 ÉTAPE 4 : Détection d'objets...")
    d_bin, d_labels, number_of_labels = detect_local_threshold(d_volume_module, d_threshold, N_CONNECTIVITY)
    del d_threshold
    cp.cuda.Device().synchronize()
    t5 = time.perf_counter()
    print(f"   Nombre d'objets détectés (seuil local) : {number_of_labels}")

    if COMPARE_GLOBAL:
        threshold_global = calc_threshold(d_volume_module, NB_STDVAR_THRESHOLD)
        _, n_global = cp_ndimage.label(d_volume_module > threshold_global, connectivity_26)
        print(f"   Pour comparaison, seuil global (moyenne + {NB_STDVAR_THRESHOLD} σ = {threshold_global:.6g}) : "
              f"{int(n_global)} objets")

    if DISPLAY_IMAGES:
        display(projection_bool(d_bin, axis=0), "Projection XY - Objets détectés (seuil local)")
        display(projection_bool(d_bin, axis=1), "Projection XZ - Objets détectés (seuil local)")

    # ÉTAPE 5 : Analyse des objets détectés
    print("📊 ÉTAPE 5 : Analyse des objets...")
    features = CCA_CUDA_float(d_labels, d_volume_module, number_of_labels, 1, HOLO_SIZE_XY, HOLO_SIZE_XY, Z_SIZE,
                              dx, dy, dz, z_offset=VOLUME_CAMERA_DISTANCE * 1e6)
    t6 = time.perf_counter()

    # ÉTAPE 6 : Sauvegarde des résultats
    print(f"💾 ÉTAPE 6 : Sauvegarde vers {RESULT_FILE}")
    positions = pd.DataFrame(features, columns=['i_image', 'baryX', 'baryY', 'baryZ', 'nb_pix'])
    positions.to_csv(RESULT_FILE, index=False)
    t7 = time.perf_counter()

    print()
    print("⏱️  TEMPS DE TRAITEMENT :")
    print(f"   Propagation : {t2-t1:.3f} s")
    print(f"   Focus       : {t3-t2:.3f} s")
    print(f"   Seuil local : {t4-t3:.3f} s")
    print(f"   Détection   : {t5-t4:.3f} s")
    print(f"   Analyse     : {t6-t5:.3f} s")
    print(f"   Sauvegarde  : {t7-t6:.3f} s")
    print(f"   TOTAL       : {t7-ini_time:.3f} s")
    print()
    print("=" * 60)
    print(f"Objets détectés : {number_of_labels}")
    print(f"Fichier résultat : {RESULT_FILE}")
    if number_of_labels > 0:
        print("Aperçu des résultats (5 premiers objets) :")
        print(positions.head().to_string(index=False))
    print("=" * 60)

    if DISPLAY_IMAGES and number_of_labels > 0:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(t_mesh.max(axis=0), cmap='viridis')
        plt.colorbar(label='seuil')
        plt.title('Seuil local par maille (max sur Z)')
        plt.subplot(1, 2, 2, projection='3d')
        ax = plt.gca()
        ax.scatter3D(positions['baryX'], positions['baryY'], positions['baryZ'], c=positions['nb_pix'], cmap='viridis')
        ax.set_xlabel('X (µm)')
        ax.set_ylabel('Y (µm)')
        ax.set_zlabel('Z (µm)')
        ax.set_title(f'Positions 3D ({number_of_labels} objets)')
        plt.tight_layout()
        plt.show()

    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)

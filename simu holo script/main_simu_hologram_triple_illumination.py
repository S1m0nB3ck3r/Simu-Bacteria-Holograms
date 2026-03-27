# -*- coding: utf-8 -*-
"""
Script simplifié de simulation d'hologrammes de bactéries (mode "bacteria_list").
Tous les paramètres sont hardcodés pour faciliter la compréhension.
Équivalent à main_simu_hologram.py + conig_1bact_3_illuminations.json

Usage:
    python main_simu_hologram_bidouille.py
"""

# ==============================================================================
# IMPORTS
# ==============================================================================
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from libs.simu_hologram import (
    Bacterie,
    GPU_insert_bact_in_mask_volume,
    create_illumination_field_polar,
    pad_centered,
    phase_shift_through_plane,
    cross_through_plane,
    create_illumination_field,
    save_volume_as_tiff,
)
import numpy as np
import cupy as cp
import math
import datetime
import libs.propagation as propagation
import libs.traitement_holo as traitement_holo
from PIL import Image
import tifffile
import json
from cupy.fft import fft2, ifft2, fftshift, ifftshift



# ==============================================================================
# PARAMÈTRES HARDCODÉS (issus de conig_1bact_3_illuminations.json)
# ==============================================================================

# --- Répertoire de sortie ---
OUTPUT_DIR = "./output/"

# --- Nombre d'hologrammes à générer ---
NB_HOLO = 1

# --- Taille de l'hologramme (en pixels, carré) ---
HOLO_SIZE_XY = 1024

# --- Bordure ajoutée de chaque côté pour éviter les effets de bord ---
BORDER = 512

# --- Facteur de sur-échantillonnage pour l'insertion des bactéries ---
UPSCALE_FACTOR = 2

# --- Nombre de plans Z dans le volume ---
Z_SIZE = 200

# --- Pas en Z (en mètres) ---
Z_STEP = 0.5e-6  # 0.5 µm

# --- Paramètres optiques ---
PIX_SIZE = 5.5e-6         # Taille pixel caméra (m)
MAGNIFICATION = 40.0      # Grossissement du microscope
INDEX_MEDIUM = 1.33       # Indice de réfraction du milieu (eau)
INDEX_OBJECT = 1.35       # Indice de réfraction de la bactérie
WAVELENGTH = 660e-9       # Longueur d'onde laser (m) = 660 nm

# --- Illumination et bruit ---
ILLUMINATION_MEAN = 1.0  # Amplitude moyenne du champ d'illumination
NOISE_STD_MIN = 0.01      # Écart-type min du bruit
NOISE_STD_MAX = 0.02       # Écart-type max du bruit

# --- Sources d'illumination (3 ondes planes avec angles différents) ---
NUMBER_OF_SOURCES = 3
SOURCES_AZIMUTH = [0.0, 120.0, 240.0]  # Angles azimutaux (en degrés)
SOURCES_POLAR   = [30.0, 30.0, 30.0]      # Angles polaires (en degrés)

# --- Distance volume-caméra (0 = au contact) ---
DISTANCE_VOLUME_CAMERA = 0.0

# --- Options de sauvegarde ---
SAVE_HOLOGRAM_BMP = True
SAVE_HOLOGRAM_TIFF = True
SAVE_HOLOGRAM_NPY = True
SAVE_PROPAGATED_TIFF = True
SAVE_PROPAGATED_NPY = True
SAVE_SEGMENTATION_TIFF = True
SAVE_SEGMENTATION_NPY = True
SAVE_POSITIONS_CSV = True

# --- Liste de bactéries (positions en mètres, dimensions en mètres, angles en radians) ---
BACTERIA_LIST = [
    {
        "pos_x": 7.0e-5,       # Position X (m) = 70 µm
        "pos_y": 7.0e-5,       # Position Y (m) = 70 µm
        "pos_z": 5.0e-5,       # Position Z (m) = 50 µm
        "length": 3.0e-6,      # Longueur (m)   = 3 µm
        "thickness": 1.0e-6,   # Épaisseur (m)  = 1 µm
        "theta": 0.0,          # Angle theta
        "phi": 90.0,            # Angle phi
    }
]


# ==============================================================================
# PARAMÈTRES DÉRIVÉS (calculés à partir des paramètres ci-dessus)
# ==============================================================================

# Taille de l'hologramme avec bordure
holo_size_xy_w_b = HOLO_SIZE_XY + BORDER * 2   # 1024 + 512*2 = 2048

# Taille d'un voxel dans le plan XY (= taille pixel ramenée au plan objet)
vox_size_xy = PIX_SIZE / MAGNIFICATION  # 5.5e-6 / 40 = 0.1375 µm

# Taille d'un voxel en Z
vox_size_z = Z_STEP  # 0.5 µm

# Longueur d'onde dans le milieu
lambda_milieu = WAVELENGTH / INDEX_MEDIUM  # 660nm / 1.33 ≈ 496 nm

# Tailles des volumes (en voxels) : [X, Y, Z]
volume_size          = [HOLO_SIZE_XY, HOLO_SIZE_XY, Z_SIZE]                                            # [1024, 1024, 200]
volume_size_upscaled = [HOLO_SIZE_XY * UPSCALE_FACTOR, HOLO_SIZE_XY * UPSCALE_FACTOR, Z_SIZE]          # [2048, 2048, 200]

# Déphasage subi par l'onde en traversant un voxel :
#   - dans le milieu : 0 (référence)
#   - dans l'objet  : 2π * dz * Δn / λ
shift_in_env = 0.0
shift_in_obj = 2.0 * cp.pi * vox_size_z * (INDEX_OBJECT - INDEX_MEDIUM) / WAVELENGTH
transmission_in_obj = 1.0 



# ==============================================================================
# FONCTIONS UTILITAIRES
# ==============================================================================

def angle_to_str(angle: float) -> str:
    """Convertit un angle float en string safe pour les noms de fichiers.
    Exemples : -5.0 -> 'm5p0', 5.0 -> '5p0', 0.0 -> '0p0'
    """
    return f"{angle:.2f}".replace('-', 'm').replace('.', 'p')


def setup_directories(base_path):
    """Crée l'arborescence de sortie avec un sous-dossier horodaté."""
    now = datetime.datetime.now()
    formatted_date_time = now.strftime("%Y_%m_%d_%H_%M_%S")
    print(f"Date et heure actuelles: {formatted_date_time}")

    output_dir              = os.path.join(base_path, formatted_date_time)
    object_positions_dir    = os.path.join(output_dir, "object_positions")
    simulated_hologram_dir  = os.path.join(output_dir, "simulated_hologram")
    binary_volume_dir       = os.path.join(output_dir, "binary_volume")
    hologram_volume_dir     = os.path.join(output_dir, "hologram_volume")

    os.makedirs(object_positions_dir, exist_ok=True)
    os.makedirs(simulated_hologram_dir, exist_ok=True)
    os.makedirs(binary_volume_dir, exist_ok=True)
    os.makedirs(hologram_volume_dir, exist_ok=True)

    return {
        'base':                output_dir,
        'object_positions':    object_positions_dir,
        'simulated_hologram':  simulated_hologram_dir,
        'binary_volume':       binary_volume_dir,
        'hologram_volume':     hologram_volume_dir,
    }


def save_results(dirs, holo_id, intensity_image, intensity_volume,
                 bool_volume_mask, liste_bacteries):
    """Sauvegarde tous les résultats d'un hologramme."""

    chemin_holograms  = dirs['simulated_hologram']
    chemin_positions  = dirs['object_positions']
    chemin_binary     = dirs['binary_volume']
    chemin_intensity  = dirs['hologram_volume']

    # Normalisation 8 bits pour le BMP
    intensity_normalized = (
        (intensity_image - intensity_image.min())
        / (intensity_image.max() - intensity_image.min() + 1e-10)
        * 255
    ).astype(np.uint8)

    bool_volume = bool_volume_mask.astype(np.uint8)

    # --- Hologramme ---
    if SAVE_HOLOGRAM_BMP:
        path = os.path.join(chemin_holograms, f"holo_{holo_id}.bmp")
        Image.fromarray(intensity_normalized).save(path)
        print(f"    [OK] Hologram BMP: {path}")

    if SAVE_HOLOGRAM_TIFF:
        path = os.path.join(chemin_holograms, f"holo_{holo_id}.tiff")
        tifffile.imwrite(path, intensity_image.astype(np.float32))
        print(f"    [OK] Hologram TIFF: {path}")

    if SAVE_HOLOGRAM_NPY:
        path = os.path.join(chemin_holograms, f"holo_{holo_id}.npy")
        np.save(path, intensity_image.astype(np.float32))
        print(f"    [OK] Hologram NPY: {path}")

    # --- Volume propagé ---
    if SAVE_PROPAGATED_TIFF:
        path = os.path.join(chemin_intensity, f"intensity_volume_{holo_id}.tiff")
        save_volume_as_tiff(path, intensity_volume)
        print(f"    [OK] Intensity volume TIFF: {path}")

    if SAVE_PROPAGATED_NPY:
        path = os.path.join(chemin_intensity, f"intensity_volume_{holo_id}.npy")
        np.save(path, intensity_volume.astype(np.float32))
        print(f"    [OK] Intensity volume NPY: {path}")

    # --- Volume segmentation (masque binaire = vérité terrain) ---
    if SAVE_SEGMENTATION_TIFF:
        path = os.path.join(chemin_binary, f"bin_volume_{holo_id}.tiff")
        save_volume_as_tiff(path, bool_volume)
        print(f"    [OK] Bin volume TIFF: {path}")

    if SAVE_SEGMENTATION_NPY:
        path = os.path.join(chemin_binary, f"bin_volume_{holo_id}.npy")
        np.save(path, bool_volume_mask)
        print(f"    [OK] Bin volume NPY: {path}")

    # --- Positions CSV ---
    if SAVE_POSITIONS_CSV:
        path = os.path.join(chemin_positions, f"bacteria_{holo_id}.csv")
        with open(path, 'w') as f:
            f.write("thickness,length,x_position_m,y_position_m,z_position_m,"
                    "x_voxel,y_voxel,z_voxel,theta_angle,phi_angle\n")
            for bact in liste_bacteries:
                x_vox = int(bact.pos_x / vox_size_xy)
                y_vox = int(bact.pos_y / vox_size_xy)
                z_vox = int(bact.pos_z / vox_size_z)
                f.write(f"{bact.thickness},{bact.length},"
                        f"{bact.pos_x},{bact.pos_y},{bact.pos_z},"
                        f"{x_vox},{y_vox},{z_vox},"
                        f"{bact.theta},{bact.phi}\n")
        print(f"    [OK] Positions CSV: {path}")

    # --- Positions TXT (toujours sauvegardé) ---
    path = os.path.join(chemin_positions, f"bacteria_{holo_id}.txt")
    with open(path, 'w') as f:
        for bact in liste_bacteries:
            x_vox = int(bact.pos_x / vox_size_xy)
            y_vox = int(bact.pos_y / vox_size_xy)
            z_vox = int(bact.pos_z / vox_size_z)
            f.write(f"{bact.thickness} {bact.length} "
                    f"{bact.pos_x} {bact.pos_y} {bact.pos_z} "
                    f"{x_vox} {y_vox} {z_vox} "
                    f"{bact.theta} {bact.phi}\n")
    print(f"    [OK] Positions TXT: {path}")


def save_parameters(dirs):
    """Sauvegarde un fichier parameters_simu_bact.json dans le répertoire de sortie."""
    params = {
        "output_base_path": OUTPUT_DIR,
        "number_of_holograms": NB_HOLO,
        "number_of_bacteria": len(BACTERIA_LIST),
        "holo_size_xy": HOLO_SIZE_XY,
        "border": BORDER,
        "upscale_factor": UPSCALE_FACTOR,
        "z_size": Z_SIZE,
        "transmission_milieu": 1.0,
        "index_milieu": INDEX_MEDIUM,
        "index_bacterie": INDEX_OBJECT,
        "longueur_min": min(b["length"] for b in BACTERIA_LIST),
        "longueur_max": max(b["length"] for b in BACTERIA_LIST),
        "epaisseur_min": min(b["thickness"] for b in BACTERIA_LIST),
        "epaisseur_max": max(b["thickness"] for b in BACTERIA_LIST),
        "pix_size": PIX_SIZE,
        "grossissement": MAGNIFICATION,
        "vox_size_z_total": Z_SIZE * Z_STEP,
        "wavelength": WAVELENGTH,
        "illumination_mean": ILLUMINATION_MEAN,
        "ecart_type_min": NOISE_STD_MIN,
        "ecart_type_max": NOISE_STD_MAX,
        "save_hologram_bmp": SAVE_HOLOGRAM_BMP,
        "save_hologram_tiff": SAVE_HOLOGRAM_TIFF,
        "save_hologram_npy": SAVE_HOLOGRAM_NPY,
        "save_propagated_tiff": SAVE_PROPAGATED_TIFF,
        "save_propagated_npy": SAVE_PROPAGATED_NPY,
        "save_segmentation_tiff": SAVE_SEGMENTATION_TIFF,
        "save_segmentation_npy": SAVE_SEGMENTATION_NPY,
        "save_positions_csv": SAVE_POSITIONS_CSV,
        "propagation_step": Z_STEP,
        "number_of_propagation": Z_SIZE,
        "volume_sensor_distance": DISTANCE_VOLUME_CAMERA,
        "step_z": Z_STEP,
        "distance_volume_camera": DISTANCE_VOLUME_CAMERA,
    }
    path = os.path.join(dirs['base'], "parameters_simu_bact.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=4)
    print(f"    [OK] Parameters JSON: {path}")


# ==============================================================================
# PROGRAMME PRINCIPAL
# ==============================================================================

if __name__ == "__main__":

    # ------------------------------------------------------------------
    # 1) Affichage du résumé de la configuration
    # ------------------------------------------------------------------
    print("=" * 80)
    print("SIMULATION HOLOGRAMME — BACTÉRIES DEPUIS LISTE (version bidouille)")
    print("=" * 80)
    print(f"Hologramme           : {HOLO_SIZE_XY}×{HOLO_SIZE_XY} px")
    print(f"Bordure              : {BORDER} px")
    print(f"Taille avec bordure  : {holo_size_xy_w_b}×{holo_size_xy_w_b} px")
    print(f"Upscale factor       : {UPSCALE_FACTOR}")
    print(f"Plans Z              : {Z_SIZE}  (pas = {Z_STEP*1e6:.1f} µm)")
    print(f"Voxel XY             : {vox_size_xy*1e6:.4f} µm")
    print(f"Voxel Z              : {vox_size_z*1e6:.4f} µm")
    print(f"Longueur d'onde      : {WAVELENGTH*1e9:.1f} nm")
    print(f"λ dans le milieu     : {lambda_milieu*1e9:.1f} nm")
    print(f"Indice milieu        : {INDEX_MEDIUM}")
    print(f"Indice objet         : {INDEX_OBJECT}")
    print(f"Sources illumination : {NUMBER_OF_SOURCES}")
    for i in range(NUMBER_OF_SOURCES):
        print(f"  Source {i+1}: angle_azimuth = {SOURCES_AZIMUTH[i]:.2f}°  angle_polaire = {SOURCES_POLAR[i]:.2f}°")
    print(f"Nombre de bactéries  : {len(BACTERIA_LIST)}")
    for i, b in enumerate(BACTERIA_LIST):
        print(f"  Bact {i+1}: pos=({b['pos_x']*1e6:.1f}, {b['pos_y']*1e6:.1f}, {b['pos_z']*1e6:.1f}) µm  "
              f"L={b['length']*1e6:.1f} µm  e={b['thickness']*1e6:.1f} µm  "
              f"θ={b['theta']:.2f}  φ={b['phi']:.2f}")
    print(f"Sortie               : {OUTPUT_DIR}")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 2) Création des répertoires de sortie
    # ------------------------------------------------------------------
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    dirs = setup_directories(OUTPUT_DIR)

    # Sauvegarder les paramètres de simulation dans le répertoire de sortie
    save_parameters(dirs)

    # ------------------------------------------------------------------
    # 3) Allocation des buffers GPU (une seule fois, réutilisés à chaque plan Z)
    # ------------------------------------------------------------------
    # Ces buffers sont utilisés par propag_angular_spectrum pour la FFT
    d_fft_holo        = cp.zeros((holo_size_xy_w_b, holo_size_xy_w_b), dtype=cp.complex64)
    d_fft_holo_propag = cp.zeros((holo_size_xy_w_b, holo_size_xy_w_b), dtype=cp.complex64)
    d_holo_propag     = cp.zeros((holo_size_xy_w_b, holo_size_xy_w_b), dtype=cp.float32)
    d_KERNEL          = cp.zeros((holo_size_xy_w_b, holo_size_xy_w_b), dtype=cp.complex64)

    # Générateur aléatoire (pour le bruit d'illumination)
    rnd = np.random.default_rng()

    # ------------------------------------------------------------------
    # 4) Boucle sur les hologrammes à générer (ici NB_HOLO = 1)
    # ------------------------------------------------------------------
    for n in range(NB_HOLO):
        print(f"\n[{n+1}/{NB_HOLO}] Génération de l'hologramme n°{n}...")

        # ==============================================================
        # 4a) Créer les objets Bacterie à partir de la liste hardcodée
        # ==============================================================
        liste_bacteries = []
        for bact_def in BACTERIA_LIST:
            bact = Bacterie(
                pos_x=bact_def['pos_x'],
                pos_y=bact_def['pos_y'],
                pos_z=bact_def['pos_z'],
                length=bact_def['length'],
                thickness=bact_def['thickness'],
                theta=bact_def.get('theta', 0.0),
                phi=bact_def.get('phi', 0.0),
            )
            liste_bacteries.append(bact)
        print(f"  → {len(liste_bacteries)} bactérie(s) créée(s)")

        # ==============================================================
        # 4b) Créer le volume 3D de masque (sur-échantillonné)
        #     Chaque voxel vaut 0 (milieu) ou >0 (objet)
        # ==============================================================
        cp_mask_volume_upscaled = cp.full(volume_size_upscaled, fill_value=0.0, dtype=cp.float16)

        print("  → Insertion des bactéries dans le volume sur-échantillonné...")
        for bact in liste_bacteries:
            GPU_insert_bact_in_mask_volume(
                cp_mask_volume_upscaled,
                bact,
                vox_size_xy / UPSCALE_FACTOR,   # voxel XY plus fin
                vox_size_z,
            )

        # ==============================================================
        # 4c) Retourner l'axe Z (convention de propagation)
        # ==============================================================
        cp_mask_volume_upscaled = cp.flip(cp_mask_volume_upscaled, axis=2)

        # ==============================================================
        # 4d) Sous-échantillonner vers la résolution finale
        #     Moyenne des blocs (upscale × upscale) → anti-aliasing
        # ==============================================================
        cp_mask_volume = cp_mask_volume_upscaled.reshape(
            HOLO_SIZE_XY, UPSCALE_FACTOR,
            HOLO_SIZE_XY, UPSCALE_FACTOR,
            Z_SIZE,
        ).mean(axis=(1, 3))
        print(f"  → Volume final : {cp_mask_volume.shape}")

        # ==============================================================
        # 4e) Masque binaire de segmentation (vérité terrain)
        #     Calculé AVANT le lissage éventuel
        # ==============================================================
        bool_volume_mask = cp.asnumpy(cp_mask_volume > 0.0)

        # ----------------------------------------------------------
        # Créer le champ d'illumination (onde plane + bruit)
        # ----------------------------------------------------------
        ecart_type_bruit = (
            (NOISE_STD_MAX - NOISE_STD_MIN) * rnd.random() + NOISE_STD_MIN
        )
        cp_field_plane = create_illumination_field_polar(
            field_size_xy_pix=holo_size_xy_w_b,
            wavelength=WAVELENGTH,
            pixel_size=PIX_SIZE,
            magnification=MAGNIFICATION,
            medium_index=INDEX_MEDIUM,
            number_of_sources=NUMBER_OF_SOURCES,
            sources_azimuth_degree=SOURCES_AZIMUTH,
            sources_polar_degree=SOURCES_POLAR,
            noise_mean=ILLUMINATION_MEAN,
            noise_std=ecart_type_bruit,
            )

        # ----------------------------------------------------------
        # Propagation plan par plan à travers le volume
        # À chaque plan Z :
        #   1. Propager le champ d'un pas dz (spectre angulaire)
        #   2. Appliquer le déphasage dû au masque (milieu vs objet)
        # ----------------------------------------------------------
        print("  → Propagation plan par plan...")
        for i in range(Z_SIZE):
            # 1) Propagation d'un pas dz
            cp_field_plane = propagation.propag_angular_spectrum(
                cp_field_plane,
                d_fft_holo,
                d_KERNEL,
                d_fft_holo_propag,
                d_holo_propag,
                lambda_milieu,
                MAGNIFICATION,
                PIX_SIZE,
                holo_size_xy_w_b,
                holo_size_xy_w_b,
                vox_size_z,
                0, 0,   # pas de shift XY
            )
            # if i % 20 == 0:
            #     phase_unwrapped = cp.unwrap(cp.unwrap(cp.angle(cp_field_plane), axis=0), axis=1)
            #     traitement_holo.display(phase_unwrapped, title=f"Phase deroulée plan Z={i}")
            #     traitement_holo.display(cp.abs(cp_field_plane), title=f"Amplitude plan Z={i}")

            # 2) Padding du masque pour correspondre à la taille avec bordure
            cp_mask_plane_w_border = pad_centered(
                cp_mask_volume[:, :, i],
                [holo_size_xy_w_b, holo_size_xy_w_b],
            )

            # 3) Déphasage : le champ traverse le voxel
            #    - dans le milieu : shift_in_env = 0
            #    - dans l'objet   : shift_in_obj = 2π·dz·Δn/λ
            cp_field_plane = cross_through_plane(
                mask_plane=cp_mask_plane_w_border,
                plane_to_shift=cp_field_plane,
                shift_in_env = 0.0,
                shift_in_obj=shift_in_obj,
                transmission_in_obj=transmission_in_obj
            )

        # ----------------------------------------------------------
        # Recadrer (retirer les bordures) et calculer l'intensité
        # ----------------------------------------------------------
        cropped_field = cp_field_plane[
            BORDER : BORDER + HOLO_SIZE_XY,
            BORDER : BORDER + HOLO_SIZE_XY,
        ]
        intensity_image = cp.asnumpy(traitement_holo.intensite(cropped_field))

        # ----------------------------------------------------------
        # Sauvegarde de tous les résultats
            # ----------------------------------------------------------
        print("  → Sauvegarde des résultats...")
        intensity_volume = cp.asnumpy(
            traitement_holo.intensite(cp_mask_volume.astype(cp.float32))
        )
        save_results(
            dirs, 1,
            intensity_image,
            intensity_volume,
            bool_volume_mask,
            liste_bacteries,
        )

    # ------------------------------------------------------------------
    # 5) Fin
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SIMULATION TERMINÉE")
    print("=" * 80)

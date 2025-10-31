# -*- coding: utf-8 -*-

"""
Filename: processor_simu_bact.py

Description:
Script de traitement qui lit parameters_simu_bact.json et génère un hologramme de bactéries.
Génère : 1 hologramme image BMP + 2 TIFF multi-stack (segmentation et intensité)

Author: Simon BECKER
Date: 2025-10-24

License:
GNU General Public License v3.0
"""

import sys
import os
import json
import numpy as np
import cupy as cp
import datetime
import traceback

# Ajoute le répertoire parent au path pour importer les modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from simu_hologram import (
    gen_random_bacteria, GPU_insert_bact_in_mask_volume,
    pad_centered, save_holo_data, save_volume_as_tiff,
    phase_shift_through_plane
)
import propagation
import traitement_holo
from PIL import Image

def display_image(array_2d, normalize=True):
    """
    Affiche un array numpy 2D avec PIL.Image.
    Si normalize=True, l'image est mise à l'échelle sur 0-255.
    """
    import numpy as np

    arr = array_2d
    if normalize:
        arr = arr.astype(np.float32)
        min_val = arr.min()
        max_val = arr.max()
        if max_val > min_val:
            arr = (arr - min_val) / (max_val - min_val) * 255
        else:
            arr = np.zeros_like(arr)
        arr = arr.astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)
    img = Image.fromarray(arr, mode='L')
    img.show()



def update_status(status_file, step, message, progress=0, error=None):
    """Met à jour le fichier de statut pour communication avec le GUI"""
    status = {
        'step': step,
        'message': message,
        'progress': progress,
        'timestamp': datetime.datetime.now().isoformat(),
        'error': error
    }
    try:
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)
    except Exception:
        pass  # Ignore les erreurs d'écriture du statut


def load_parameters():
    """Charge les paramètres depuis le fichier JSON"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, "parameters_simu_bact.json")
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Fichier de configuration non trouvé : {config_file}")
    
    with open(config_file, 'r') as f:
        params = json.load(f)
    
    return params


def create_output_directories(base_path):
    """Crée les répertoires de sortie avec timestamp"""
    now = datetime.datetime.now()
    formatted_date_time = now.strftime("%Y_%m_%d_%H_%M_%S")
    print(f"Date et heure actuelles : {formatted_date_time}")
    
    output_dir = os.path.join(base_path, formatted_date_time)
    positions_dir = os.path.join(output_dir, "positions")
    holograms_dir = os.path.join(output_dir, "holograms")
    data_dir = os.path.join(output_dir, "data_holograms")
    
    os.makedirs(positions_dir, exist_ok=True)
    os.makedirs(holograms_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    return {
        'base': output_dir,
        'positions': positions_dir,
        'holograms': holograms_dir,
        'data': data_dir
    }


def generate_hologram(params, status_file=None):
    """Génère un hologramme de bactéries selon les paramètres"""
    
    print("="*80)
    print("DÉBUT DE LA GÉNÉRATION D'HOLOGRAMME")
    print("="*80)
    
    if status_file:
        update_status(status_file, 0, "Initialisation...", 0)
    
    # Extraction des paramètres
    number_of_bacteria = params['number_of_bacteria']
    holo_size_xy = params['holo_size_xy']
    border = params['border']
    upscale_factor = params['upscale_factor']
    z_size = params['z_size']
    
    transmission_milieu = params['transmission_milieu']
    index_milieu = params['index_milieu']
    index_bacterie = params['index_bacterie']
    
    longueur_min = params['longueur_min']
    longueur_max = params['longueur_max']
    epaisseur_min = params['epaisseur_min']
    epaisseur_max = params['epaisseur_max']
    
    pix_size = params['pix_size']
    grossissement = params['grossissement']
    vox_size_z_total = params['vox_size_z_total']
    
    wavelength = params['wavelength']
    illumination_mean = params['illumination_mean']
    ecart_type_min = params['ecart_type_min']
    ecart_type_max = params['ecart_type_max']
    
    # Calculs dérivés
    holo_size_xy_w_b = holo_size_xy + border * 2
    vox_size_xy = pix_size / grossissement
    vox_size_z = vox_size_z_total / z_size
    lambda_milieu = wavelength / index_milieu
    
    print(f"\nParamètres du volume :")
    print(f"  - Taille XY : {holo_size_xy} pixels")
    print(f"  - Taille XY avec bordure : {holo_size_xy_w_b} pixels")
    print(f"  - Nombre de plans Z : {z_size}")
    print(f"  - Taille voxel XY : {vox_size_xy*1e6:.3f} µm")
    print(f"  - Taille voxel Z : {vox_size_z*1e6:.3f} µm")
    print(f"  - Nombre de bactéries : {number_of_bacteria}")
    
    # Création des répertoires de sortie
    dirs = create_output_directories(params['output_base_path'])
    
    # Noms des fichiers de sortie
    data_file = os.path.join(dirs['data'], "data_0.npz")
    holo_file = os.path.join(dirs['holograms'], "holo_0.bmp")
    bin_tiff_file = os.path.join(dirs['holograms'], "bin_volume_0.tiff")
    intensity_tiff_file = os.path.join(dirs['holograms'], "intensity_volume_0.tiff")
    positions_file = os.path.join(dirs['positions'], "bact_0.txt")
    
    # Paramètres pour save_holo_data
    parameters_dict = {
        'holo_size_x': holo_size_xy,
        'holo_size_y': holo_size_xy,
        'holo_plane_number': z_size,
        'medium_index': index_milieu,
        'object_index': index_bacterie,
        'pix_size_cam': pix_size,
        'magnification_cam': grossissement,
        'Z_step': vox_size_z,
        'illumination_wavelength': wavelength,
        'medium_wavelength': lambda_milieu
    }
    
    # Génération du champ d'illumination
    print("\n[1/5] Création du champ d'illumination...")
    if status_file:
        update_status(status_file, 1, "Création du champ d'illumination...", 10)
    rng = np.random.default_rng()
    np_field_plane = np.full(shape=[holo_size_xy_w_b, holo_size_xy_w_b], 
                             fill_value=0.0+0.0j, dtype=np.complex64)
    ecart_type_bruit = (ecart_type_max - ecart_type_min) * rng.random() + ecart_type_min
    bruit_gaussien = np.abs(np.random.normal(illumination_mean, ecart_type_bruit, 
                                            [holo_size_xy_w_b, holo_size_xy_w_b]))
    np_field_plane.real = np.sqrt(bruit_gaussien)
    cp_field_plane = cp.asarray(np_field_plane)
    
    # Initialisation des masques
    print("[2/5] Initialisation des masques...")
    if status_file:
        update_status(status_file, 2, "Initialisation des masques...", 20)
    volume_size = [holo_size_xy, holo_size_xy, z_size]
    volume_size_upscaled = [holo_size_xy * upscale_factor, holo_size_xy * upscale_factor, z_size]
    
    cp_mask_volume = cp.full(shape=volume_size, fill_value=0, dtype=cp.float16)
    cp_mask_plane_w_border = cp.full(shape=(holo_size_xy_w_b, holo_size_xy_w_b), 
                                     fill_value=0.0, dtype=cp.float32)
    cp_mask_volume_upscaled = cp.full(shape=volume_size_upscaled, fill_value=0, dtype=cp.float16)
    
    # Allocation du volume d'intensité pour stocker la propagation
    cp_intensity_volume = cp.zeros(shape=(holo_size_xy, holo_size_xy, z_size), dtype=cp.float32)
    
    # Génération des bactéries
    print("[3/5] Génération et insertion des bactéries...")
    if status_file:
        update_status(status_file, 3, "Génération des bactéries...", 30)
    longueur_min_max = {min: longueur_min, max: longueur_max}
    epaisseur_min_max = {min: epaisseur_min, max: epaisseur_max}
    
    liste_bacteries = gen_random_bacteria(
        number_of_bact=number_of_bacteria,
        xyz_min_max=[0, holo_size_xy * vox_size_xy, 
                     0, holo_size_xy * vox_size_xy, 
                     0, z_size * vox_size_z],
        thickness_min_max=epaisseur_min_max,
        length_min_max=longueur_min_max
    )
    
    # Sauvegarde des positions des bactéries
    for bact in liste_bacteries:
        bact.to_file(positions_file)
    
    # Conversion en liste de dictionnaires pour save_holo_data
    bacteria_list = [
        {
            "thickness": b.thickness,
            "length": b.length,
            "x_position_m": b.pos_x,
            "y_position_m": b.pos_y,
            "z_position_m": b.pos_z,
            "theta_angle": b.theta,
            "phi_angle": b.phi
        }
        for b in liste_bacteries
    ]
    
    # Insertion des bactéries dans le volume
    for i, bact in enumerate(liste_bacteries):
        if (i + 1) % 50 == 0:
            print(f"  Insertion bactérie {i+1}/{number_of_bacteria}")
            if status_file:
                progress = 30 + int(20 * (i + 1) / number_of_bacteria)
                update_status(status_file, 3, f"Insertion bactérie {i+1}/{number_of_bacteria}...", progress)
        GPU_insert_bact_in_mask_volume(cp_mask_volume_upscaled, bact, 
                                      vox_size_xy / upscale_factor, vox_size_z)
    
    # Inversion de l'axe Z et downsampling
    cp_mask_volume_upscaled = cp.flip(cp_mask_volume_upscaled, axis=2)
    cp_mask_volume = cp_mask_volume_upscaled[:, :, :].reshape(
        holo_size_xy, upscale_factor, holo_size_xy, upscale_factor, z_size
    ).mean(axis=(1, 3))
    
    # Calculs de propagation
    print("[4/5] Propagation du champ...")
    if status_file:
        update_status(status_file, 4, "Propagation du champ...", 50)
    shift_in_env = 0.0
    shift_in_obj = 2.0 * cp.pi * vox_size_z * (index_bacterie - index_milieu) / wavelength
    
    # Allocations pour la propagation
    d_fft_holo = cp.zeros(shape=(holo_size_xy_w_b, holo_size_xy_w_b), dtype=cp.complex64)
    d_fft_holo_propag = cp.zeros(shape=(holo_size_xy_w_b, holo_size_xy_w_b), dtype=cp.complex64)
    d_holo_propag = cp.zeros(shape=(holo_size_xy_w_b, holo_size_xy_w_b), dtype=cp.float32)
    d_KERNEL = cp.zeros(shape=(holo_size_xy_w_b, holo_size_xy_w_b), dtype=cp.complex64)
    
    # Propagation plan par plan
    for i in range(z_size):
        if (i + 1) % 50 == 0:
            print(f"  Plan {i+1}/{z_size}")
            if status_file:
                progress = 50 + int(30 * (i + 1) / z_size)
                update_status(status_file, 4, f"Propagation plan {i+1}/{z_size}...", progress)

        # display_image(cp.asnumpy(cp.abs(cp_field_plane)**2), normalize=True)
        
        cp_field_plane = propagation.propag_angular_spectrum(
            cp_field_plane, d_fft_holo, d_KERNEL, d_fft_holo_propag, d_holo_propag,
            lambda_milieu, float(grossissement), pix_size, 
            holo_size_xy_w_b, holo_size_xy_w_b, vox_size_z, 0, 0
        )

        # display_image(cp.asnumpy(cp.abs(cp_field_plane)**2), normalize=True)

        
        # Sauvegarde de l'intensité du plan courant (sans bordure) dans le volume
        croped_plane = cp_field_plane[border:border+holo_size_xy, border:border+holo_size_xy]
        cp_intensity_volume[:, :, i] = traitement_holo.intensite(croped_plane)
        
        cp_mask_plane_w_border = pad_centered(cp_mask_volume[:, :, i], 
                                             [holo_size_xy_w_b, holo_size_xy_w_b])
        
        # Correction de phase
        cp_field_plane = phase_shift_through_plane(
            mask_plane=cp_mask_plane_w_border,
            plane_to_shift=cp_field_plane,
            shift_in_env=shift_in_env,
            shift_in_obj=shift_in_obj
        )
    
    # Recadrage du champ final pour l'hologramme 2D
    croped_field_plane = cp_field_plane[border:border+holo_size_xy, border:border+holo_size_xy]
    
    # Calcul de l'intensité de l'hologramme final (plan du capteur)
    intensity_image = cp.asnumpy(traitement_holo.intensite(croped_field_plane))
    
    # Conversion du volume d'intensité en numpy
    intensity_volume = cp.asnumpy(cp_intensity_volume)
    
    # Sauvegarde des résultats
    print("[5/5] Sauvegarde des résultats...")
    if status_file:
        update_status(status_file, 5, "Sauvegarde des résultats...", 80)
    
    # Volume binaire (segmentation)
    # cp_mask_volume contient des valeurs float16 entre 0 et 1 (après downsampling)
    # Il faut d'abord créer un masque booléen, puis convertir en uint8
    
    # Debug : affichage des stats du volume avant conversion
    print(f"\n  DEBUG cp_mask_volume AVANT conversion:")
    print(f"    - shape: {cp_mask_volume.shape}, dtype: {cp_mask_volume.dtype}")
    print(f"    - min: {float(cp.min(cp_mask_volume))}, max: {float(cp.max(cp_mask_volume))}")
    
    bool_volume_mask = cp.asnumpy(cp_mask_volume > 0.0)  # Booléen True/False
    bool_volume = bool_volume_mask.astype(np.uint8)  # 0 ou 1
    
    print(f"\n  DEBUG bool_volume APRÈS conversion:")
    print(f"    - shape: {bool_volume.shape}, dtype: {bool_volume.dtype}")
    print(f"    - min: {bool_volume.min()}, max: {bool_volume.max()}")
    print(f"    - non-zero: {np.count_nonzero(bool_volume)}/{bool_volume.size}")
    print(f"    - unique values: {np.unique(bool_volume)}")
    
    # Affiche quelques valeurs du plan central pour vérifier
    mid_z = bool_volume.shape[2] // 2
    print(f"\n  DEBUG plan central (z={mid_z}):")
    print(f"    - non-zero dans ce plan: {np.count_nonzero(bool_volume[:, :, mid_z])}")
    print(f"    - max dans ce plan: {bool_volume[:, :, mid_z].max()}")
    
    # Sauvegarde du fichier NPZ complet
    save_holo_data(data_file, bool_volume_mask, intensity_image, parameters_dict, bacteria_list)
    
    # Sauvegarde de l'image BMP de l'hologramme
    intensity_normalized = ((intensity_image - intensity_image.min()) / 
                           (intensity_image.max() - intensity_image.min()) * 255).astype(np.uint8)
    Image.fromarray(intensity_normalized).save(holo_file)
    
    # Sauvegarde des TIFF multi-stack
    save_volume_as_tiff(bin_tiff_file, bool_volume)
    save_volume_as_tiff(intensity_tiff_file, intensity_volume)
    
    if status_file:
        update_status(status_file, 6, "Génération terminée avec succès !", 100)
    
    print("\n" + "="*80)
    print("GÉNÉRATION TERMINÉE AVEC SUCCÈS !")
    print("="*80)
    print(f"\nFichiers générés dans : {dirs['base']}")
    print(f"  - Hologramme BMP : {holo_file}")
    print(f"  - Volume binaire TIFF : {bin_tiff_file}")
    print(f"  - Volume intensité TIFF : {intensity_tiff_file}")
    print(f"  - Données NPZ : {data_file}")
    print(f"  - Positions bactéries : {positions_file}")
    print("="*80 + "\n")
    
    # Retourne les chemins pour la visualisation
    return {
        'hologram': holo_file,
        'bin_volume': bin_tiff_file,
        'intensity_volume': intensity_tiff_file,
        'output_dir': dirs['base']
    }


def main():
    """Fonction principale"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    status_file = os.path.join(script_dir, "processing_status.json")
    result_file = os.path.join(script_dir, "processing_result.json")
    
    try:
        # Chargement des paramètres
        print("Chargement des paramètres...")
        update_status(status_file, 0, "Chargement des paramètres...", 0)
        params = load_parameters()
        
        # Génération de l'hologramme
        result = generate_hologram(params, status_file)
        
        # Sauvegarde du résultat pour le GUI
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        return 0
        
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"\n❌ ERREUR : {error_msg}", file=sys.stderr)
        traceback.print_exc()
        update_status(status_file, -1, "Erreur lors de la génération", 0, error_msg)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

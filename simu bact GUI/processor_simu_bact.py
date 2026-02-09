# -*- coding: utf-8 -*-

"""
Filename: processor_simu_bact.py

Description:
Script de traitement qui lit parameters_simu_bact.json et génère des hologrammes de bactéries.
Supporte les itérations multiples et les options de sauvegarde personnalisées.

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
import shutil
from PIL import Image

# Ajoute le répertoire parent au path pour importer les modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'libs'))

# Configuration de l'encodage pour éviter les erreurs sur Windows
if sys.platform.startswith('win'):
    try:
        # Essaie de configurer UTF-8 pour la console Windows
        os.system('chcp 65001 > nul')
    except:
        pass  # Ignore les erreurs si chcp n'est pas disponible

# Add parent directory to path for imports
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'libs'))

from simu_hologram import (
    gen_random_bacteria, GPU_insert_bact_in_mask_volume,
    pad_centered, save_holo_data, save_volume_as_tiff,
    phase_shift_through_plane
)
import propagation
import traitement_holo
from PIL import Image
import tifffile

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


def update_status(status_file, step, message, progress=0, error=None, stopped=False):
    """Met à jour le fichier de statut pour communication avec le GUI"""
    status = {
        'step': step,
        'message': message,
        'progress': progress,
        'timestamp': datetime.datetime.now().isoformat(),
        'error': error,
        'stopped': stopped
    }
    try:
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)
    except Exception:
        pass  # Ignore les erreurs d'écriture du statut


def check_stop_signal():
    """Vérifie si un signal d'arrêt a été envoyé"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    stop_file = os.path.join(script_dir, "processing_stop.json")
    
    if os.path.exists(stop_file):
        try:
            with open(stop_file, 'r') as f:
                stop_data = json.load(f)
            return stop_data.get("stop_requested", False)
        except Exception:
            return False
    return False


def cleanup_stop_signal():
    """Nettoie le fichier signal d'arrêt"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    stop_file = os.path.join(script_dir, "processing_stop.json")
    
    if os.path.exists(stop_file):
        try:
            os.remove(stop_file)
        except Exception:
            pass


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
    """Crée les répertoires de sortie avec timestamp et copie le fichier de paramètres"""
    now = datetime.datetime.now()
    formatted_date_time = now.strftime("%Y_%m_%d_%H_%M_%S")
    print(f"Date et heure actuelles : {formatted_date_time}")
    
    output_dir = os.path.join(base_path, formatted_date_time)
    simulated_hologram_dir = os.path.join(output_dir, "simulated_hologram")
    binary_volume_dir = os.path.join(output_dir, "binary_volume")
    hologram_volume_dir = os.path.join(output_dir, "hologram_volume")
    object_positions_dir = os.path.join(output_dir, "object_positions")
    
    os.makedirs(simulated_hologram_dir, exist_ok=True)
    os.makedirs(binary_volume_dir, exist_ok=True)
    os.makedirs(hologram_volume_dir, exist_ok=True)
    os.makedirs(object_positions_dir, exist_ok=True)
    
    # Copie du fichier de paramètres dans le répertoire de sortie
    script_dir = os.path.dirname(os.path.abspath(__file__))
    params_source = os.path.join(script_dir, "parameters_simu_bact.json")
    params_dest = os.path.join(output_dir, "parameters_simu_bact.json")
    
    try:
        if os.path.exists(params_source):
            shutil.copy2(params_source, params_dest)
            print(f"Fichier de paramètres copié vers : {params_dest}")
        else:
            print(f"Attention : Fichier de paramètres non trouvé à {params_source}")
    except Exception as e:
        print(f"Erreur lors de la copie du fichier de paramètres : {e}")
    
    return {
        'base': output_dir,
        'simulated_hologram': simulated_hologram_dir,
        'binary_volume': binary_volume_dir,
        'hologram_volume': hologram_volume_dir,
        'object_positions': object_positions_dir
    }


def generate_hologram(params, iteration, dirs, status_file=None):
    """
    Génère un hologramme de bactéries selon les paramètres
    
    Args:
        params: Dictionnaire des paramètres
        iteration: Numéro d'itération (pour nommage des fichiers)
        dirs: Dictionnaire des répertoires de sortie
        status_file: Fichier de statut pour le GUI
    """
    
    print(f"\n{'='*80}")
    print(f"GÉNÉRATION HOLOGRAMME #{iteration}")
    print(f"{'='*80}")
    
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
    distance_volume_camera = params.get('distance_volume_camera', 0.01)  # Distance volume-caméra (défaut: 1cm)
    step_z = params.get('step_z', 0.5e-6)  # Pas Z (défaut: 0.5 µm)
    
    # Calcul de vox_size_z_total à partir de step_z et z_size
    vox_size_z_total = step_z * z_size
    
    wavelength = params['wavelength']
    illumination_mean = params['illumination_mean']
    ecart_type_min = params['ecart_type_min']
    ecart_type_max = params['ecart_type_max']
    
    # Options de sauvegarde
    save_opts = {
        'hologram_bmp': params.get('save_hologram_bmp', True),
        'hologram_tiff': params.get('save_hologram_tiff', False),
        'hologram_npy': params.get('save_hologram_npy', False),
        'propagated_tiff': params.get('save_propagated_tiff', True),
        'propagated_npy': params.get('save_propagated_npy', False),
        'segmentation_tiff': params.get('save_segmentation_tiff', True),
        'segmentation_npy': params.get('save_segmentation_npy', False),
        'positions_csv': params.get('save_positions_csv', True)
    }
    
    # Calculs dérivés
    holo_size_xy_w_b = holo_size_xy + border * 2
    vox_size_xy = pix_size / grossissement
    vox_size_z = vox_size_z_total / z_size
    lambda_milieu = wavelength / index_milieu
    
    if iteration == 0:
        print(f"\nParamètres du volume :")
        print(f"  - Taille XY : {holo_size_xy} pixels")
        print(f"  - Taille XY avec bordure : {holo_size_xy_w_b} pixels")
        print(f"  - Nombre de plans Z : {z_size}")
        print(f"  - Taille voxel XY : {vox_size_xy*1e6:.3f} µm")
        print(f"  - Taille voxel Z : {vox_size_z*1e6:.3f} µm")
        print(f"  - Nombre de bactéries : {number_of_bacteria}")
    
    # Noms des fichiers de sortie avec numéro d'itération
    holo_bmp_file = os.path.join(dirs['simulated_hologram'], f"holo_{iteration}.bmp")
    holo_tiff_file = os.path.join(dirs['simulated_hologram'], f"holo_{iteration}.tiff")
    holo_npy_file = os.path.join(dirs['simulated_hologram'], f"holo_{iteration}.npy")
    bin_tiff_file = os.path.join(dirs['binary_volume'], f"bin_volume_{iteration}.tiff")
    bin_npy_file = os.path.join(dirs['binary_volume'], f"bin_volume_{iteration}.npy")
    intensity_tiff_file = os.path.join(dirs['hologram_volume'], f"intensity_volume_{iteration}.tiff")
    intensity_npy_file = os.path.join(dirs['hologram_volume'], f"intensity_volume_{iteration}.npy")
    positions_file = os.path.join(dirs['object_positions'], f"bact_{iteration}.txt")
    positions_csv_file = os.path.join(dirs['object_positions'], f"bact_{iteration}.csv")
    
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
    
    field_abs = cp.abs(cp_field_plane)
    
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
    
    print(f"DEBUG: {len(liste_bacteries)} bactéries générées")
    
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

    # array_sum = cp.asnumpy(cp_mask_volume.sum(axis=2))
    # array_sum_u8 = (array_sum / array_sum.max() * 255).astype(np.uint8)
    # Image.fromarray(array_sum_u8, mode='L').save(os.path.join(dirs['binary_volume'], f"debug_mask_sum_{iteration}.bmp"))
    
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

        
        cp_field_plane = propagation.propag_angular_spectrum(
            cp_field_plane, d_fft_holo, d_KERNEL, d_fft_holo_propag, d_holo_propag,
            lambda_milieu, float(grossissement), pix_size, 
            holo_size_xy_w_b, holo_size_xy_w_b, vox_size_z, 0, 0
        )
        
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

    # Propagation finale jusqu'au plan hologramme
    print("  Propagation finale jusqu'au plan hologramme...")
    
    # Propagation avec la distance volume-caméra
    if distance_volume_camera > 0:
        print(f"    Distance volume-caméra : {distance_volume_camera*1000:.1f} mm")
        cp_field_plane = propagation.propag_angular_spectrum(
            cp_field_plane, d_fft_holo, d_KERNEL, d_fft_holo_propag, d_holo_propag,
            lambda_milieu, float(grossissement), pix_size, 
            holo_size_xy_w_b, holo_size_xy_w_b, distance_volume_camera, 0, 0
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
        update_status(status_file, 5, f"Sauvegarde hologramme #{iteration}...", 80)
    
    # Volume binaire (segmentation)
    bool_volume_mask = cp.asnumpy(cp_mask_volume > 0.0)  # Booléen True/False
    bool_volume = bool_volume_mask.astype(np.uint8)  # 0 ou 1
    
    # Normalisation de l'hologramme pour affichage 8bits
    if intensity_image.max() > intensity_image.min():
        intensity_normalized = ((intensity_image - intensity_image.min()) / 
                               (intensity_image.max() - intensity_image.min()) * 255).astype(np.uint8)
        print(f"Intensité normalisée - min: {intensity_normalized.min()}, max: {intensity_normalized.max()}")
    else:
        print("ERREUR: Intensité constante (min == max) ! L'hologramme sera noir.")
        intensity_normalized = np.zeros_like(intensity_image, dtype=np.uint8)
    
    # Sauvegarde conditionnelle selon les options
    saved_files = []
    
    # Hologramme BMP 8bits
    if save_opts['hologram_bmp']:
        Image.fromarray(intensity_normalized).save(holo_bmp_file)
        saved_files.append(f"Hologramme BMP : {holo_bmp_file}")
    
    # Hologramme TIFF 32bits
    if save_opts['hologram_tiff']:
        tifffile.imwrite(holo_tiff_file, intensity_image.astype(np.float32))
        saved_files.append(f"Hologramme TIFF : {holo_tiff_file}")
    
    # Hologramme NPY 32bits
    if save_opts['hologram_npy']:
        np.save(holo_npy_file, intensity_image.astype(np.float32))
        saved_files.append(f"Hologramme NPY : {holo_npy_file}")
    
    # Volume propagé TIFF multistack
    if save_opts['propagated_tiff']:
        save_volume_as_tiff(intensity_tiff_file, intensity_volume)
        saved_files.append(f"Volume propagé TIFF : {intensity_tiff_file}")
    
    # Volume propagé NPY
    if save_opts['propagated_npy']:
        np.save(intensity_npy_file, intensity_volume.astype(np.float32))
        saved_files.append(f"Volume propagé NPY : {intensity_npy_file}")
    
    # Volume segmentation TIFF multistack
    if save_opts['segmentation_tiff']:
        save_volume_as_tiff(bin_tiff_file, bool_volume)
        saved_files.append(f"Volume segmentation TIFF : {bin_tiff_file}")
    
    # Volume segmentation NPY bool
    if save_opts['segmentation_npy']:
        np.save(bin_npy_file, bool_volume_mask)  # Sauvegarde en booléen
        saved_files.append(f"Volume segmentation NPY : {bin_npy_file}")
    
    # Positions bactéries - TOUJOURS sauvegarder les deux formats
    if save_opts['positions_csv']:
        # Sauvegarde en format CSV avec positions en mètres ET en voxels
        with open(positions_csv_file, 'w') as f:
            f.write("thickness,length,x_position_m,y_position_m,z_position_m,x_voxel,y_voxel,z_voxel,theta_angle,phi_angle\n")
            for bact in liste_bacteries:
                # Calcul des positions en voxels (indices du tableau)
                x_voxel = int(bact.pos_x / vox_size_xy)
                y_voxel = int(bact.pos_y / vox_size_xy)
                z_voxel = int(bact.pos_z / vox_size_z)
                f.write(f"{bact.thickness},{bact.length},{bact.pos_x},{bact.pos_y},{bact.pos_z},{x_voxel},{y_voxel},{z_voxel},{bact.theta},{bact.phi}\n")
        saved_files.append(f"Positions CSV : {positions_csv_file}")
    
    # Toujours sauvegarder aussi le fichier TXT avec TOUTES les colonnes
    with open(positions_file, 'w') as f:
        for bact in liste_bacteries:
            # Calcul des positions en voxels
            x_voxel = int(bact.pos_x / vox_size_xy)
            y_voxel = int(bact.pos_y / vox_size_xy)
            z_voxel = int(bact.pos_z / vox_size_z)
            # Format TXT avec toutes les colonnes (même ordre que CSV)
            f.write(f"{bact.thickness} {bact.length} {bact.pos_x} {bact.pos_y} {bact.pos_z} {x_voxel} {y_voxel} {z_voxel} {bact.theta} {bact.phi}\n")
    saved_files.append(f"Positions TXT : {positions_file}")
    
    print(f"\nFichiers générés pour hologramme #{iteration} :")
    for file_desc in saved_files:
        print(f"  - {file_desc}")
    
    return saved_files


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
        
        # Nombre d'hologrammes à générer
        number_of_holograms = params.get('number_of_holograms', 1)
        
        print(f"\n{'='*80}")
        print(f"DÉBUT DE LA SIMULATION - {number_of_holograms} hologramme(s)")
        print(f"{'='*80}\n")
        
        # Création des répertoires de sortie (une seule fois pour toutes les itérations)
        dirs = create_output_directories(params['output_base_path'])
        
        # Génération des hologrammes
        all_saved_files = []
        for i in range(number_of_holograms):
            # Vérification du signal d'arrêt entre chaque hologramme
            if check_stop_signal():
                print(f"\nArret demande par l'utilisateur apres {i} hologramme(s)")
                update_status(status_file, i, "Simulation interrompue par l'utilisateur", 
                            int((i / number_of_holograms) * 100), stopped=True)
                cleanup_stop_signal()
                return
            
            # Calcul de la progression globale
            overall_progress = int((i / number_of_holograms) * 100)
            update_status(status_file, i+1, f"Génération hologramme {i+1}/{number_of_holograms}...", overall_progress)
            
            # Génération d'un hologramme
            saved_files = generate_hologram(params, i, dirs, status_file)
            all_saved_files.extend(saved_files)
        
        # Sauvegarde du résultat pour le GUI
        result = {
            'output_dir': dirs['base'],
            'number_of_holograms': number_of_holograms,
            'files': all_saved_files
        }
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Nettoyage du signal d'arrêt
        cleanup_stop_signal()
        
        update_status(status_file, number_of_holograms+1, "Simulation terminée avec succès !", 100)
        
        print(f"\n{'='*80}")
        print("SIMULATION TERMINÉE AVEC SUCCÈS !")
        print(f"{'='*80}")
        print(f"\n{number_of_holograms} hologramme(s) généré(s) dans : {dirs['base']}")
        print(f"{'='*80}\n")
        
        return 0
        
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"\n❌ ERREUR : {error_msg}", file=sys.stderr)
        traceback.print_exc()
        
        # Nettoyage du signal d'arrêt en cas d'erreur aussi
        cleanup_stop_signal()
        
        update_status(status_file, -1, "Erreur lors de la simulation", 0, error_msg)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

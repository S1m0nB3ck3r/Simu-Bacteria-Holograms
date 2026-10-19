# -*- coding: utf-8 -*-

"""
Unified hologram simulation script for bacteria, spheres (random or list-based)

Usage:
    python main_simu_hologram.py configs/config_bacteria_random.json
    python main_simu_hologram.py configs/config_bacteria_list.json
    python main_simu_hologram.py configs/config_spheres_random.json
"""

# Add parent directory to path for imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from libs.simu_hologram import *
import numpy as np
import cupy as cp
from cupyx import jit
import math
import time
import matplotlib.pyplot as plt
import datetime
import libs.propagation as propagation
import libs.traitement_holo as traitement_holo
from PIL import Image  
import PIL
import argparse
import json


def load_config(config_path):
    """Load configuration from JSON file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print(f"Configuration loaded from: {config_path}")
    return config


def validate_config(config):
    """Validate configuration and set defaults"""
    required_keys = ['mode', 'nb_holo']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
    
    # Set default values for optional keys
    defaults = {
        'output_dir': None,
        'nb_objects': 50,
        'config_file': None,
        'holo_size_xy': 1024,
        'border': 256,
        'upscale_factor': 2,
        'z_size': 200,
        'z_step': 0.5e-6,
        'length_min': 1.5e-6,
        'length_max': 2.0e-6,
        'thickness_min': 0.8e-6,
        'thickness_max': 0.8e-6,
        'radius_min': 0.5e-6,
        'radius_max': 2.0e-6,
        'pix_size': 5.5e-6,
        'magnification': 40.0,
        'index_medium': 1.33,
        'index_object': 1.335,
        'wavelength': 660e-9,
        'illumination_mean': 1.0,
        'sensor_noise_std_min': 0.015,   # bruit capteur, relatif au fond (mesuré sur film réel)
        'sensor_noise_std_max': 0.015,
        'number_of_sources': 1,
        'sources_angle_azimuth': [0.0],
        'sources_angle_polar': [0.0],
        'distance_volume_camera': 0.0,   # m, object space
        'holo_ref_nb_planes': 1,
        'volume_smoothing': {
            'enabled': False,
            'method': 'gaussian',
            'sigma': 0.5,
            'sdf_width': 1.0
        },
        'save_options': {
            'hologram_bmp': True,
            'hologram_tiff': False,
            'hologram_npy': False,
            'propagated_tiff': True,
            'propagated_npy': False,
            'segmentation_tiff': True,
            'segmentation_npy': False,
            'positions_csv': True,
            'holo_ref': False
        }
    }
    
    for key, default_value in defaults.items():
        if key not in config:
            config[key] = default_value
    
    # Compatibilite ascendante : l'ancienne convention cartesienne
    # (sources_angle_degree_X/Y = inclinaisons selon X et Y) est convertie
    # vers la convention canonique azimut / polaire.
    if 'sources_angle_degree_X' in config or 'sources_angle_degree_Y' in config:
        n_src = config['number_of_sources']
        angles_x = config.get('sources_angle_degree_X', [0.0] * n_src)
        angles_y = config.get('sources_angle_degree_Y', [0.0] * n_src)
        polar, azimuth = [], []
        for a_x, a_y in zip(angles_x, angles_y):
            sin_x = math.sin(math.radians(a_x))
            sin_y = math.sin(math.radians(a_y))
            polar.append(math.degrees(math.asin(min(1.0, math.hypot(sin_x, sin_y)))))
            azimuth.append(math.degrees(math.atan2(sin_y, sin_x)))
        config['sources_angle_polar'] = polar
        config['sources_angle_azimuth'] = azimuth
        print("  [config] convention 'sources_angle_degree_X/Y' convertie en "
              "'sources_angle_azimuth' / 'sources_angle_polar'")

    # Bruit d'illumination supprimé (bruit blanc spatial sur l'amplitude : pas de phénomène physique
    # correspondant ; le fond fixe réel est retiré par division par la moyenne du film). Remplacé par
    # le bruit capteur sensor_noise_std_min/max, ajouté à l'intensité après propagation.
    for old in ('noise_std_min', 'noise_std_max'):
        if old in config:
            config.pop(old)
            print(f"  [config] '{old}' ignoré : bruit d'illumination supprimé, "
                  "utiliser 'sensor_noise_std_min' / 'sensor_noise_std_max'")
    if not 0 <= config['sensor_noise_std_min'] <= config['sensor_noise_std_max']:
        raise ValueError("sensor_noise_std_min / max : attendu 0 <= min <= max, "
                         f"got {config['sensor_noise_std_min']} / {config['sensor_noise_std_max']}")

    # Validate illumination source angles
    if len(config['sources_angle_azimuth']) != config['number_of_sources']:
        raise ValueError(f"sources_angle_azimuth must have {config['number_of_sources']} elements, "
                        f"got {len(config['sources_angle_azimuth'])}")
    if len(config['sources_angle_polar']) != config['number_of_sources']:
        raise ValueError(f"sources_angle_polar must have {config['number_of_sources']} elements, "
                        f"got {len(config['sources_angle_polar'])}")
    
    return config


def setup_directories(base_path, save_holo_ref=False):
    """Create output directories and return paths"""
    now = datetime.datetime.now()
    formatted_date_time = now.strftime("%Y_%m_%d_%H_%M_%S")
    print(f"Date et heure actuelles: {formatted_date_time}")
    
    output_dir = os.path.join(base_path, formatted_date_time)
    object_positions_dir = os.path.join(output_dir, "object_positions")
    simulated_hologram_dir = os.path.join(output_dir, "simulated_hologram")
    binary_volume_dir = os.path.join(output_dir, "binary_volume")
    hologram_volume_dir = os.path.join(output_dir, "hologram_volume")
    holo_ref_dir = os.path.join(output_dir, "holo_ref")
    
    os.makedirs(object_positions_dir, exist_ok=True)
    os.makedirs(simulated_hologram_dir, exist_ok=True)
    os.makedirs(binary_volume_dir, exist_ok=True)
    os.makedirs(hologram_volume_dir, exist_ok=True)
    if save_holo_ref:
        os.makedirs(holo_ref_dir, exist_ok=True)
    
    return {
        'base': output_dir,
        'object_positions': object_positions_dir,
        'simulated_hologram': simulated_hologram_dir,
        'binary_volume': binary_volume_dir,
        'hologram_volume': hologram_volume_dir,
        'holo_ref': holo_ref_dir
    }


def angle_to_str(angle: float) -> str:
    """Convert angle float to filename-safe string (no dot, no minus sign).
    Example: -5.0 -> 'm5p0', 5.0 -> '5p0', 0.0 -> '0p0'
    """
    return f"{angle:.2f}".replace('-', 'm').replace('.', 'p')


def load_bacteria_list_from_file(filepath):
    """Load bacteria list from file"""
    bacteria_list = []
    # TODO: Implement based on file format
    return bacteria_list


def load_sphere_list_from_file(filepath):
    """Load sphere list from file"""
    sphere_list = []
    # TODO: Implement based on file format
    return sphere_list


def save_hologram_results(dirs, n, intensity_image, 
                         intensity_volume, bool_volume_mask, liste_objects, 
                         save_options, vox_size_xy, vox_size_z, object_type='bacteria'):
    """
    Save hologram results based on save_options
    
    Args:
        dirs: Dictionary with keys 'simulated_hologram', 'binary_volume', 'hologram_volume', 'object_positions'
        n: Iteration number
        intensity_image: 2D intensity (hologram)
        intensity_volume: 3D intensity volume
        bool_volume_mask: 3D binary volume (segmentation)
        liste_objects: List of object instances
        save_options: Dict with boolean flags for what to save
        vox_size_xy, vox_size_z: Voxel sizes for coordinate conversion
        object_type: 'bacteria' or 'sphere'
    """
    import tifffile
    
    # Extract directories from dirs dictionary
    chemin_holograms = dirs['simulated_hologram']
    chemin_positions = dirs['object_positions']
    chemin_binary = dirs['binary_volume']
    chemin_intensity = dirs['hologram_volume']
    
    # Normalize hologram to 8-bit
    intensity_normalized = ((intensity_image - intensity_image.min()) / 
                           (intensity_image.max() - intensity_image.min() + 1e-10) * 255).astype(np.uint8)
    
    bool_volume = bool_volume_mask.astype(np.uint8)
    
    # Hologram BMP 8bits
    if save_options.get('hologram_bmp', False):
        holo_bmp_file = os.path.join(chemin_holograms, f"holo_{n}.bmp")
        Image.fromarray(intensity_normalized).save(holo_bmp_file)
        print(f"    [OK] Saved hologram BMP: {holo_bmp_file}")
    
    # Hologram TIFF 32bits
    if save_options.get('hologram_tiff', False):
        holo_tiff_file = os.path.join(chemin_holograms, f"holo_{n}.tiff")
        tifffile.imwrite(holo_tiff_file, intensity_image.astype(np.float32))
        print(f"    [OK] Saved hologram TIFF: {holo_tiff_file}")
    
    # Hologram NPY 32bits
    if save_options.get('hologram_npy', False):
        holo_npy_file = os.path.join(chemin_holograms, f"holo_{n}.npy")
        np.save(holo_npy_file, intensity_image.astype(np.float32))
        print(f"    [OK] Saved hologram NPY: {holo_npy_file}")
    
    # Volume propagated TIFF multistack
    if save_options.get('propagated_tiff', False):
        intensity_tiff_file = os.path.join(chemin_intensity, f"intensity_volume_{n}.tiff")
        save_volume_as_tiff(intensity_tiff_file, intensity_volume)
        print(f"    [OK] Saved intensity volume TIFF: {intensity_tiff_file}")
    
    # Volume propagated NPY
    if save_options.get('propagated_npy', False):
        intensity_npy_file = os.path.join(chemin_intensity, f"intensity_volume_{n}.npy")
        np.save(intensity_npy_file, intensity_volume.astype(np.float32))
        print(f"    [OK] Saved intensity volume NPY: {intensity_npy_file}")
    
    # Volume segmentation TIFF multistack
    if save_options.get('segmentation_tiff', False):
        bin_tiff_file = os.path.join(chemin_binary, f"bin_volume_{n}.tiff")
        save_volume_as_tiff(bin_tiff_file, bool_volume)
        print(f"    [OK] Saved bin_volume TIFF: {bin_tiff_file}")
    
    # Volume segmentation NPY bool
    if save_options.get('segmentation_npy', False):
        bin_npy_file = os.path.join(chemin_binary, f"bin_volume_{n}.npy")
        np.save(bin_npy_file, bool_volume_mask)
        print(f"    [OK] Saved bin_volume NPY: {bin_npy_file}")
    
    # Positions CSV (with both meters and voxels)
    if save_options.get('positions_csv', False):
        positions_csv_file = os.path.join(chemin_positions, f"{object_type}_{n}.csv")
        with open(positions_csv_file, 'w') as f:
            if object_type == 'bacteria':
                f.write("thickness,length,x_position_m,y_position_m,z_position_m,x_voxel,y_voxel,z_voxel,theta_angle,phi_angle\n")
                for obj in liste_objects:
                    x_voxel = int(obj.pos_x / vox_size_xy)
                    y_voxel = int(obj.pos_y / vox_size_xy)
                    z_voxel = int(obj.pos_z / vox_size_z)
                    f.write(f"{obj.thickness},{obj.length},{obj.pos_x},{obj.pos_y},{obj.pos_z},"
                           f"{x_voxel},{y_voxel},{z_voxel},{obj.theta},{obj.phi}\n")
            else:  # sphere
                f.write("radius,x_position_m,y_position_m,z_position_m,x_voxel,y_voxel,z_voxel\n")
                for obj in liste_objects:
                    x_voxel = int(obj.pos_x / vox_size_xy)
                    y_voxel = int(obj.pos_y / vox_size_xy)
                    z_voxel = int(obj.pos_z / vox_size_z)
                    f.write(f"{obj.radius},{obj.pos_x},{obj.pos_y},{obj.pos_z},"
                           f"{x_voxel},{y_voxel},{z_voxel}\n")
        print(f"    [OK] Saved positions CSV: {positions_csv_file}")
    
    # Always save TXT format as well
    positions_file = os.path.join(chemin_positions, f"{object_type}_{n}.txt")
    with open(positions_file, 'w') as f:
        if object_type == 'bacteria':
            for obj in liste_objects:
                x_voxel = int(obj.pos_x / vox_size_xy)
                y_voxel = int(obj.pos_y / vox_size_xy)
                z_voxel = int(obj.pos_z / vox_size_z)
                f.write(f"{obj.thickness} {obj.length} {obj.pos_x} {obj.pos_y} {obj.pos_z} "
                       f"{x_voxel} {y_voxel} {z_voxel} {obj.theta} {obj.phi}\n")
        else:  # sphere
            for obj in liste_objects:
                x_voxel = int(obj.pos_x / vox_size_xy)
                y_voxel = int(obj.pos_y / vox_size_xy)
                z_voxel = int(obj.pos_z / vox_size_z)
                f.write(f"{obj.radius} {obj.pos_x} {obj.pos_y} {obj.pos_z} "
                       f"{x_voxel} {y_voxel} {z_voxel}\n")
    print(f"    ✓ Saved positions TXT: {positions_file}")


def save_reference_hologram(dirs, holo_id, make_field, nb_planes, distance, intensity_image,
                            buffers, lambda_milieu, magnification, pix_size,
                            holo_size_xy, border, save_options, sensor_std, background_intensity):
    """
    Reference hologram, as the mean of nb_planes frames of a real film without object: the
    same illumination (make_field, noise-free) propagated over the same total distance as the
    simulated hologram, plus sensor noise sensor_std/sqrt(nb_planes) (the mean of N independent
    Gaussian draws is one draw of std sigma/sqrt(N)). Saved in holo_ref/ with the hologram's formats.

    Without phase masks, the z_size steps of dz compose into a single propagation.
    The BMP is normalized with the hologram's min/max (not its own) so that both BMPs
    share the same intensity scale.
    """
    import tifffile

    holo_size_xy_w_b = holo_size_xy + 2 * border
    d_fft_holo, d_KERNEL, d_fft_holo_propag, d_holo_propag = buffers
    cp_field_ref = propagation.propag_angular_spectrum(
        make_field(), d_fft_holo, d_KERNEL, d_fft_holo_propag, d_holo_propag,
        lambda_milieu, magnification, pix_size, holo_size_xy_w_b, holo_size_xy_w_b,
        distance, 0, 0
    )
    ref_image = cp.asnumpy(add_sensor_noise(
        traitement_holo.intensite(cp_field_ref[border:border+holo_size_xy, border:border+holo_size_xy]),
        sensor_std / math.sqrt(nb_planes), background_intensity))

    ref_base = os.path.join(dirs['holo_ref'], f"holo_ref_{holo_id}")
    if save_options.get('hologram_bmp', False):
        i_min, i_max = intensity_image.min(), intensity_image.max()
        ref_normalized = (np.clip((ref_image - i_min) / (i_max - i_min + 1e-10), 0.0, 1.0)
                          * 255).astype(np.uint8)
        Image.fromarray(ref_normalized).save(ref_base + ".bmp")
        print(f"    [OK] Saved reference hologram BMP: {ref_base}.bmp")
    if save_options.get('hologram_tiff', False):
        tifffile.imwrite(ref_base + ".tiff", ref_image.astype(np.float32))
        print(f"    [OK] Saved reference hologram TIFF: {ref_base}.tiff")
    if save_options.get('hologram_npy', False):
        np.save(ref_base + ".npy", ref_image.astype(np.float32))
        print(f"    [OK] Saved reference hologram NPY: {ref_base}.npy")


def simulate_bacteria(config, dirs):
    """Simulate bacteria holograms, from a random draw or from a predefined list.

    Modes pris en charge :
      - 'bacteria_random' : bacteries tirees aleatoirement dans le volume central
      - 'bacteria_list'   : bacteries decrites dans config['bacteria']

    Seule la construction de la liste d'objets differe entre les deux modes ;
    toute la chaine optique (volume, propagation, sauvegarde) est commune.
    """
    chemin_positions = dirs['object_positions']

    from_list = (config['mode'] == 'bacteria_list')

    print("\n" + "=" * 80)
    print("SIMULATION: BACTERIA (FROM LIST)" if from_list else "SIMULATION: BACTERIA (RANDOM)")
    print("=" * 80)

    # Parameters
    nb_holo_to_simulate = int(config['nb_holo'])
    holo_size_xy = int(config['holo_size_xy'])
    border = int(config['border'])
    upscale_factor = int(config['upscale_factor'])
    z_size = int(config['z_size'])

    # Optical parameters
    pix_size = config['pix_size']
    grossissement = config['magnification']
    index_milieu = config['index_medium']
    index_objet = config['index_object']
    wavelength = config['wavelength']

    # Volume parameters
    holo_size_xy_w_b = holo_size_xy + border * 2
    vox_size_xy = pix_size / grossissement
    vox_size_z = config['z_step']
    lambda_milieu = wavelength / index_milieu

    volume_size = [holo_size_xy, holo_size_xy, z_size]
    volume_size_upscaled = [holo_size_xy * upscale_factor, holo_size_xy * upscale_factor, z_size]

    # Phase shift parameters
    shift_in_env = 0.0
    shift_in_obj = 2.0 * cp.pi * vox_size_z * (index_objet - index_milieu) / wavelength

    # GPU allocations
    d_fft_holo = cp.zeros(shape=(holo_size_xy_w_b, holo_size_xy_w_b), dtype=cp.complex64)
    d_fft_holo_propag = cp.zeros(shape=(holo_size_xy_w_b, holo_size_xy_w_b), dtype=cp.complex64)
    d_holo_propag = cp.zeros(shape=(holo_size_xy_w_b, holo_size_xy_w_b), dtype=cp.float32)
    d_KERNEL = cp.zeros(shape=(holo_size_xy_w_b, holo_size_xy_w_b), dtype=cp.complex64)

    rnd = np.random.default_rng()

    # Entrees specifiques au mode
    if from_list:
        bacteria_list_config = config.get('bacteria', [])
        if not bacteria_list_config:
            print("ERROR: No bacteria defined in config['bacteria']")
            return
    else:
        number_of_bacteria = config['nb_objects']
        longueur_min_max = {min: config['length_min'], max: config['length_max']}
        epaisseur_min_max = {min: config['thickness_min'], max: config['thickness_max']}

    for n in range(nb_holo_to_simulate):
        print(f"\n[{n+1}/{nb_holo_to_simulate}] Generating hologram set {n}...")

        positions_file = os.path.join(chemin_positions, f"bact_{n}.txt")

        # Initialize volumes (once per n, shared across all illuminations)
        cp_mask_volume = cp.full(shape=volume_size, fill_value=0.0, dtype=cp.float16)
        cp_mask_volume_upscaled = cp.full(shape=volume_size_upscaled, fill_value=0.0, dtype=cp.float16)

        # Construction de la liste de bacteries
        if from_list:
            print(f"  - Creating {len(bacteria_list_config)} bacteria from list...")
            liste_bacteries = [
                Bacterie(
                    pos_x=b['pos_x'], pos_y=b['pos_y'], pos_z=b['pos_z'],
                    length=b['length'], thickness=b['thickness'],
                    theta=b.get('theta', 0.0), phi=b.get('phi', 0.0)
                )
                for b in bacteria_list_config
            ]
        else:
            print(f"  - Generating {number_of_bacteria} bacteria...")
            liste_bacteries = gen_random_bacteria(
                number_of_bact=number_of_bacteria,
                xyz_min_max=[0, holo_size_xy * vox_size_xy,
                             0, holo_size_xy * vox_size_xy,
                             0, z_size * vox_size_z],
                thickness_min_max=epaisseur_min_max,
                length_min_max=longueur_min_max
            )

        # Save bacteria positions (once per n, shared across all illuminations)
        print("  - Saving bacteria positions...")
        with open(positions_file, 'w') as f:
            for bact in liste_bacteries:
                f.write(f"{bact.pos_x}\t{bact.pos_y}\t{bact.pos_z}\t"
                        f"{bact.length}\t{bact.thickness}\t{bact.theta}\t{bact.phi}\n")

        # Insert bacteria in volume
        print("  - Inserting bacteria in volume...")
        for i, bact in enumerate(liste_bacteries):
            if (i + 1) % max(1, len(liste_bacteries) // 10) == 0:
                print(f"    - {i+1}/{len(liste_bacteries)}")
            # upscale_factor : aligne la grille fine sur la grille finale (voir upscale_grid_shift),
            # sans quoi le moyennage de blocs ci-dessous décale l'objet de (up-1)/(2*up) voxel
            GPU_insert_bact_in_mask_volume(cp_mask_volume_upscaled, bact,
                                          vox_size_xy / upscale_factor, vox_size_z, upscale_factor)


        # Flip Z axis
        cp_mask_volume_upscaled = cp.flip(cp_mask_volume_upscaled, axis=2)
        # retour à la grille finale : le bloc moyenné est décentré d'un demi-bloc par rapport au
        # point d'échantillonnage i*vox, ce que compense le décalage appliqué à l'insertion
        cp_mask_volume = cp_mask_volume_upscaled[:,:,:].reshape(
            holo_size_xy, upscale_factor, holo_size_xy, upscale_factor, z_size
        ).mean(axis=(1, 3))
        
        # Segmentation binaire AVANT lissage (vérité terrain)
        bool_volume_mask = cp.asnumpy(cp_mask_volume > 0.0)
        
        # Lissage du volume pour réduire l'aliasing des bords voxelisés
        smoothing_cfg = config.get('volume_smoothing', {})
        if smoothing_cfg.get('enabled', False):
            cp_mask_volume = smooth_volume_gpu(
                cp_mask_volume,
                method=smoothing_cfg.get('method', 'gaussian'),
                sigma=smoothing_cfg.get('sigma', 0.5),
                sdf_width=smoothing_cfg.get('sdf_width', 1.0)
            )
        
        # Loop over illumination sources (1 hologram per source)
        for src_idx in range(config['number_of_sources']):
            angle_azimuth = config['sources_angle_azimuth'][src_idx]
            angle_polar = config['sources_angle_polar'][src_idx]
            angle_suffix = f"_az{angle_to_str(angle_azimuth)}_pol{angle_to_str(angle_polar)}"
            holo_id = f"{n}{angle_suffix}"
            
            print(f"  - Source {src_idx+1}/{config['number_of_sources']}: azimuth={angle_azimuth}° polar={angle_polar}°")
            
            # Illumination without noise (plane wave); sensor noise is added to the intensity
            # after propagation, with a std drawn once for this hologram
            sensor_std = ((config['sensor_noise_std_max'] - config['sensor_noise_std_min']) * rnd.random() +
                          config['sensor_noise_std_min'])
            background_intensity = config['illumination_mean'] ** 2
            def make_illumination_field():
                field = create_illumination_field_polar(
                    field_size_xy_pix=holo_size_xy_w_b,
                    wavelength=wavelength,
                    pixel_size=pix_size,
                    magnification=grossissement,
                    medium_index=index_milieu,
                    number_of_sources=1,
                    sources_azimuth_degree=[angle_azimuth],
                    sources_polar_degree=[angle_polar],
                    noise_mean=config['illumination_mean'],
                    noise_std=0.0
                )
                # Apodisation of the border: removes the periodic-wrap phase jump of tilted waves
                return apodize_border(field, border)
            cp_field_plane = make_illumination_field()
            save_holo_ref = config.get('save_options', {}).get('holo_ref', False)
            
            # Propagation
            print("  - Propagating hologram...")
            for i in range(z_size):
                if (i + 1) % max(1, z_size // 10) == 0:
                    print(f"    - Z plane {i+1}/{z_size}")
                
                cp_field_plane = propagation.propag_angular_spectrum(
                    cp_field_plane, d_fft_holo, d_KERNEL, d_fft_holo_propag, d_holo_propag,
                    lambda_milieu, grossissement, pix_size, holo_size_xy_w_b, holo_size_xy_w_b, 
                    vox_size_z, 0, 0
                )
                
                cp_mask_plane_w_border = pad_centered(cp_mask_volume[:,:,i], 
                                                      [holo_size_xy_w_b, holo_size_xy_w_b])
                
                cp_field_plane = phase_shift_through_plane(
                    mask_plane=cp_mask_plane_w_border,
                    plane_to_shift=cp_field_plane,
                    shift_in_env=shift_in_env,
                    shift_in_obj=shift_in_obj
                )
            
            # Final propagation from the volume exit to the hologram plane
            distance_volume_camera = config['distance_volume_camera']
            if distance_volume_camera > 0:
                print(f"  - Propagating volume -> camera ({distance_volume_camera*1e6:.1f} µm)...")
                cp_field_plane = propagation.propag_angular_spectrum(
                    cp_field_plane, d_fft_holo, d_KERNEL, d_fft_holo_propag, d_holo_propag,
                    lambda_milieu, grossissement, pix_size, holo_size_xy_w_b, holo_size_xy_w_b,
                    distance_volume_camera, 0, 0
                )
            
            # Crop and save
            cropped_field_plane = cp_field_plane[border:border+holo_size_xy, 
                                                 border:border+holo_size_xy]
            
            intensity_image = cp.asnumpy(add_sensor_noise(
                traitement_holo.intensite(cropped_field_plane), sensor_std, background_intensity))
            
            if save_holo_ref:
                print(f"  - Reference hologram (no object, {config['holo_ref_nb_planes']} plane(s))...")
                save_reference_hologram(
                    dirs, holo_id, make_illumination_field, max(1, int(config['holo_ref_nb_planes'])),
                    z_size * vox_size_z + max(0.0, distance_volume_camera), intensity_image,
                    (d_fft_holo, d_KERNEL, d_fft_holo_propag, d_holo_propag),
                    lambda_milieu, grossissement, pix_size, holo_size_xy, border,
                    config.get('save_options', {}), sensor_std, background_intensity
                )
            
            print("  - Saving results...")
            save_hologram_results(
                dirs, holo_id,
                intensity_image,
                cp.asnumpy(traitement_holo.intensite(cp_mask_volume.astype(cp.float32))),
                bool_volume_mask,
                liste_bacteries,
                config.get('save_options', {}),
                vox_size_xy, vox_size_z,
                object_type='bacteria'
            )
            
            print(f"    ✓ Illumination {src_idx+1} complete (id={holo_id})")
        
        print(f"  ✓ Hologram set {n} complete ({config['number_of_sources']} illuminations)")


def simulate_sphere(config, dirs):
    """Simulate sphere holograms, from a random draw or from a predefined list.

    Modes pris en charge :
      - 'sphere_random' : spheres tirees aleatoirement dans le volume central
      - 'sphere_list'   : spheres decrites dans config['spheres']

    Seule la construction de la liste d'objets differe entre les deux modes ;
    toute la chaine optique (volume, propagation, sauvegarde) est commune.
    """
    chemin_positions = dirs['object_positions']

    from_list = (config['mode'] == 'sphere_list')

    print("\n" + "=" * 80)
    print("SIMULATION: SPHERES (FROM LIST)" if from_list else "SIMULATION: SPHERES (RANDOM)")
    print("=" * 80)

    # Parameters
    nb_holo_to_simulate = int(config['nb_holo'])
    holo_size_xy = int(config['holo_size_xy'])
    border = int(config['border'])
    upscale_factor = int(config['upscale_factor'])
    z_size = int(config['z_size'])

    # Optical parameters
    pix_size = config['pix_size']
    grossissement = config['magnification']
    index_milieu = config['index_medium']
    index_objet = config['index_object']
    wavelength = config['wavelength']

    # Volume parameters
    holo_size_xy_w_b = holo_size_xy + border * 2
    vox_size_xy = pix_size / grossissement
    vox_size_z = config['z_step']
    lambda_milieu = wavelength / index_milieu

    volume_size = [holo_size_xy, holo_size_xy, z_size]
    volume_size_upscaled = [holo_size_xy * upscale_factor, holo_size_xy * upscale_factor, z_size]

    # Phase shift parameters
    shift_in_env = 0.0
    shift_in_obj = 2.0 * cp.pi * vox_size_z * (index_objet - index_milieu) / wavelength
    transmission_obj = 0.0  # Spheres are opaque

    # GPU allocations
    d_fft_holo = cp.zeros(shape=(holo_size_xy_w_b, holo_size_xy_w_b), dtype=cp.complex64)
    d_fft_holo_propag = cp.zeros(shape=(holo_size_xy_w_b, holo_size_xy_w_b), dtype=cp.complex64)
    d_holo_propag = cp.zeros(shape=(holo_size_xy_w_b, holo_size_xy_w_b), dtype=cp.float32)
    d_KERNEL = cp.zeros(shape=(holo_size_xy_w_b, holo_size_xy_w_b), dtype=cp.complex64)

    rnd = np.random.default_rng()

    # Entrees specifiques au mode
    if from_list:
        spheres_list_config = config.get('spheres', [])
        if not spheres_list_config:
            print("ERROR: No spheres defined in config['spheres']")
            return
    else:
        number_of_spheres = config['nb_objects']
        rayon_min_max = {min: config['radius_min'], max: config['radius_max']}
        # Volume constraints - spheres inserted in central region
        volume_min_max = [0, holo_size_xy * vox_size_xy,
                          0, holo_size_xy * vox_size_xy,
                          0, z_size * vox_size_z]

    for n in range(nb_holo_to_simulate):
        print(f"\n[{n+1}/{nb_holo_to_simulate}] Generating hologram set {n}...")

        positions_file = os.path.join(chemin_positions, f"spheres_{n}.txt")

        # Initialize volumes (once per n, shared across all illuminations)
        cp_mask_volume = cp.full(shape=volume_size, fill_value=0.0, dtype=cp.float16)
        cp_mask_volume_upscaled = cp.full(shape=volume_size_upscaled, fill_value=0.0, dtype=cp.float16)

        # Construction de la liste de spheres
        if from_list:
            print(f"  - Creating {len(spheres_list_config)} spheres from list...")
            liste_spheres = [
                Sphere(pos_x=s['pos_x'], pos_y=s['pos_y'], pos_z=s['pos_z'], radius=s['radius'])
                for s in spheres_list_config
            ]
        else:
            print(f"  - Generating {number_of_spheres} random spheres...")
            liste_spheres = gen_random_sphere(number_of_spheres, volume_min_max, rayon_min_max)

        # Save sphere positions (once per n, shared across all illuminations)
        print("  - Saving sphere positions...")
        with open(positions_file, 'w') as f:
            for sphere in liste_spheres:
                f.write(f"{sphere.pos_x}\t{sphere.pos_y}\t"
                       f"{sphere.pos_z}\t{sphere.radius}\n")

        # Insert spheres in volume
        print("  - Inserting spheres in volume...")
        for i, sphere in enumerate(liste_spheres):
            # upscale_factor : aligne la grille fine sur la grille finale (voir upscale_grid_shift)
            insert_sphere_in_mask_volume(cp_mask_volume_upscaled, sphere,
                                        vox_size_xy / upscale_factor, vox_size_z, upscale_factor)


        # Flip Z axis
        cp_mask_volume_upscaled = cp.flip(cp_mask_volume_upscaled, axis=2)
        # retour à la grille finale : le bloc moyenné est décentré d'un demi-bloc par rapport au
        # point d'échantillonnage i*vox, ce que compense le décalage appliqué à l'insertion
        cp_mask_volume = cp_mask_volume_upscaled[:,:,:].reshape(
            holo_size_xy, upscale_factor, holo_size_xy, upscale_factor, z_size
        ).mean(axis=(1, 3))
        
        # Segmentation binaire AVANT lissage (vérité terrain)
        bool_volume_mask = cp.asnumpy(cp_mask_volume > 0.0)
        
        # Lissage du volume pour réduire l'aliasing des bords voxelisés
        smoothing_cfg = config.get('volume_smoothing', {})
        if smoothing_cfg.get('enabled', False):
            cp_mask_volume = smooth_volume_gpu(
                cp_mask_volume,
                method=smoothing_cfg.get('method', 'gaussian'),
                sigma=smoothing_cfg.get('sigma', 0.5),
                sdf_width=smoothing_cfg.get('sdf_width', 1.0)
            )
        
        # Loop over illumination sources (1 hologram per source)
        for src_idx in range(config['number_of_sources']):
            angle_azimuth = config['sources_angle_azimuth'][src_idx]
            angle_polar = config['sources_angle_polar'][src_idx]
            angle_suffix = f"_az{angle_to_str(angle_azimuth)}_pol{angle_to_str(angle_polar)}"
            holo_id = f"{n}{angle_suffix}"
            
            print(f"  - Source {src_idx+1}/{config['number_of_sources']}: azimuth={angle_azimuth}° polar={angle_polar}°")
            
            # Illumination without noise (plane wave); sensor noise is added to the intensity
            # after propagation, with a std drawn once for this hologram
            sensor_std = ((config['sensor_noise_std_max'] - config['sensor_noise_std_min']) * rnd.random() +
                          config['sensor_noise_std_min'])
            background_intensity = config['illumination_mean'] ** 2
            def make_illumination_field():
                field = create_illumination_field_polar(
                    field_size_xy_pix=holo_size_xy_w_b,
                    wavelength=wavelength,
                    pixel_size=pix_size,
                    magnification=grossissement,
                    medium_index=index_milieu,
                    number_of_sources=1,
                    sources_azimuth_degree=[angle_azimuth],
                    sources_polar_degree=[angle_polar],
                    noise_mean=config['illumination_mean'],
                    noise_std=0.0
                )
                # Apodisation of the border: removes the periodic-wrap phase jump of tilted waves
                return apodize_border(field, border)
            cp_field_plane = make_illumination_field()
            save_holo_ref = config.get('save_options', {}).get('holo_ref', False)
            
            # Propagation
            print("  - Propagating hologram...")
            for i in range(z_size):
                cp_field_plane = propagation.propag_angular_spectrum(
                    cp_field_plane, d_fft_holo, d_KERNEL, d_fft_holo_propag, d_holo_propag,
                    lambda_milieu, grossissement, pix_size, holo_size_xy_w_b, holo_size_xy_w_b, 
                    vox_size_z, 0, 0
                )
                
                cp_mask_plane_w_border = pad_centered(cp_mask_volume[:,:,i], 
                                                      [holo_size_xy_w_b, holo_size_xy_w_b])
                
                cp_field_plane = cross_through_plane(
                    mask_plane=cp_mask_plane_w_border,
                    plane_to_shift=cp_field_plane,
                    shift_in_env=shift_in_env,
                    shift_in_obj=shift_in_obj,
                    transmission_in_obj=transmission_obj
                )
            
            # Final propagation from the volume exit to the hologram plane
            distance_volume_camera = config['distance_volume_camera']
            if distance_volume_camera > 0:
                print(f"  - Propagating volume -> camera ({distance_volume_camera*1e6:.1f} µm)...")
                cp_field_plane = propagation.propag_angular_spectrum(
                    cp_field_plane, d_fft_holo, d_KERNEL, d_fft_holo_propag, d_holo_propag,
                    lambda_milieu, grossissement, pix_size, holo_size_xy_w_b, holo_size_xy_w_b,
                    distance_volume_camera, 0, 0
                )
            
            # Crop and save
            cropped_field_plane = cp_field_plane[border:border+holo_size_xy, 
                                                 border:border+holo_size_xy]
            
            intensity_image = cp.asnumpy(add_sensor_noise(
                traitement_holo.intensite(cropped_field_plane), sensor_std, background_intensity))
            
            if save_holo_ref:
                print(f"  - Reference hologram (no object, {config['holo_ref_nb_planes']} plane(s))...")
                save_reference_hologram(
                    dirs, holo_id, make_illumination_field, max(1, int(config['holo_ref_nb_planes'])),
                    z_size * vox_size_z + max(0.0, distance_volume_camera), intensity_image,
                    (d_fft_holo, d_KERNEL, d_fft_holo_propag, d_holo_propag),
                    lambda_milieu, grossissement, pix_size, holo_size_xy, border,
                    config.get('save_options', {}), sensor_std, background_intensity
                )
            
            print("  - Saving results...")
            save_hologram_results(
                dirs, holo_id,
                intensity_image,
                cp.asnumpy(traitement_holo.intensite(cp_mask_volume.astype(cp.float32))),
                bool_volume_mask,
                liste_spheres,
                config.get('save_options', {}),
                vox_size_xy, vox_size_z,
                object_type='sphere'
            )
            
            print(f"    ✓ Illumination {src_idx+1} complete (id={holo_id})")
        print(f"  ✓ Hologram set {n} complete ({config['number_of_sources']} illuminations)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Unified hologram simulation script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_simu_hologram.py configs/config_bacteria_random.json
  python main_simu_hologram.py configs/config_sphere_random.json
        """
    )
    
    parser.add_argument('config_file', type=str, nargs='?', help='JSON configuration file')
    args = parser.parse_args()
    
    # --- DEBUG FALLBACK : config utilisée si aucun argument n'est passé ---
    DEBUG_CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config_1bact_3_illuminations.json')
    # ----------------------------------------------------------------------
    
    if args.config_file is None:
        print(f"[DEBUG] No argument provided, using hardcoded config: {DEBUG_CONFIG_FILE}")
        args.config_file = DEBUG_CONFIG_FILE
    
    # Load and validate configuration
    try:
        config = load_config(args.config_file)
        config = validate_config(config)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        exit(1)
    
    # Setup output directory
    if config['output_dir'] is None:
        mode_prefix = config['mode'].split('_')[0]
        base_path = os.path.join(os.getcwd(), f"simu_{mode_prefix}")
    else:
        base_path = config['output_dir']
    
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    dirs = setup_directories(base_path, config['save_options'].get('holo_ref', False))

    # Configuration effective (valeurs par défaut comprises), pour relire les paramètres
    # de la simulation depuis son dossier de sortie (stats/GUI_holo_stats.py)
    with open(os.path.join(dirs['base'], "config_simu.json"), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    
    # Print configuration summary
    print("\n" + "="*80)
    print("CONFIGURATION SUMMARY")
    print("="*80)
    print(f"Mode: {config['mode']}")
    print(f"Number of holograms: {config['nb_holo']}")
    print(f"Output directory: {base_path}")
    print(f"Holo size: {config['holo_size_xy']}x{config['holo_size_xy']} pixels")
    print(f"Z planes: {config['z_size']}")
    print(f"Wavelength: {config['wavelength']*1e9:.1f} nm")
    print(f"Medium index: {config['index_medium']}")
    print(f"Object index: {config['index_object']}")
    print(f"Volume-camera distance: {config['distance_volume_camera']*1e6:.1f} µm")
    print(f"Illumination sources: {config['number_of_sources']}")
    for _i in range(config['number_of_sources']):
        print(f"  Source {_i+1}: azimuth={config['sources_angle_azimuth'][_i]:.2f}°  polar={config['sources_angle_polar'][_i]:.2f}°")
    
    smoothing = config.get('volume_smoothing', {})
    if smoothing.get('enabled', False):
        print(f"Volume smoothing: {smoothing['method']} (sigma={smoothing.get('sigma', 0.5)}, sdf_width={smoothing.get('sdf_width', 1.0)})")
    else:
        print(f"Volume smoothing: disabled")
    
    if 'bacteria' in config['mode']:
        print(f"Number of bacteria per hologram: {config['nb_objects']}")
        print(f"Bacteria length: {config['length_min']*1e6:.2f} - {config['length_max']*1e6:.2f} µm")
        print(f"Bacteria thickness: {config['thickness_min']*1e6:.3f} - {config['thickness_max']*1e6:.3f} µm")
    elif 'sphere' in config['mode']:
        print(f"Number of spheres per hologram: {config['nb_objects']}")
        print(f"Sphere radius: {config['radius_min']*1e6:.3f} - {config['radius_max']*1e6:.3f} µm")
    print("="*80 + "\n")
    
    # Run simulation based on mode
    if config['mode'] in ('bacteria_random', 'bacteria_list'):
        simulate_bacteria(config, dirs)
    elif config['mode'] in ('sphere_random', 'sphere_list'):
        simulate_sphere(config, dirs)
    else:
        raise ValueError(f"Unknown mode: {config['mode']!r} (attendu : bacteria_random, "
                         "bacteria_list, sphere_random ou sphere_list)")
    
    print("\n" + "="*80)
    print("SIMULATION COMPLETE")
    print("="*80)

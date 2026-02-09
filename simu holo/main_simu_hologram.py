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
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'libs'))

from simu_hologram import *
import numpy as np
import cupy as cp
from cupyx import jit
import math
import time
import matplotlib.pyplot as plt
import datetime
import propagation
import traitement_holo
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
        'length_min': 3.0e-6,
        'length_max': 4.0e-6,
        'thickness_min': 1.0e-6,
        'thickness_max': 2.0e-6,
        'radius_min': 0.5e-6,
        'radius_max': 2.0e-6,
        'pix_size': 5.5e-6,
        'magnification': 40.0,
        'index_medium': 1.33,
        'index_object': 1.335,
        'wavelength': 660e-9,
        'illumination_mean': 1.0,
        'noise_std_min': 0.01,
        'noise_std_max': 0.1,
        'distance_volume_camera': 0.01,
        'save_options': {
            'hologram_bmp': True,
            'hologram_tiff': False,
            'hologram_npy': False,
            'propagated_tiff': True,
            'propagated_npy': False,
            'segmentation_tiff': True,
            'segmentation_npy': False,
            'positions_csv': True
        }
    }
    
    for key, default_value in defaults.items():
        if key not in config:
            config[key] = default_value
    
    return config


def setup_directories(base_path):
    """Create output directories and return paths"""
    now = datetime.datetime.now()
    formatted_date_time = now.strftime("%Y_%m_%d_%H_%M_%S")
    print(f"Date et heure actuelles: {formatted_date_time}")
    
    output_dir = os.path.join(base_path, formatted_date_time)
    object_positions_dir = os.path.join(output_dir, "object_positions")
    simulated_hologram_dir = os.path.join(output_dir, "simulated_hologram")
    binary_volume_dir = os.path.join(output_dir, "binary_volume")
    hologram_volume_dir = os.path.join(output_dir, "hologram_volum")
    
    os.makedirs(object_positions_dir, exist_ok=True)
    os.makedirs(simulated_hologram_dir, exist_ok=True)
    os.makedirs(binary_volume_dir, exist_ok=True)
    os.makedirs(hologram_volume_dir, exist_ok=True)
    
    return {
        'base': output_dir,
        'object_positions': object_positions_dir,
        'simulated_hologram': simulated_hologram_dir,
        'binary_volume': binary_volume_dir,
        'hologram_volume': hologram_volume_dir
    }


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
        intensity_tiff_file = os.path.join(chemin_intensity, f"propagated_volume_{n}.tiff")
        save_volume_as_tiff(intensity_tiff_file, intensity_volume)
        print(f"    [OK] Saved propagated volume TIFF: {intensity_tiff_file}")
    
    # Volume propagated NPY
    if save_options.get('propagated_npy', False):
        intensity_npy_file = os.path.join(chemin_intensity, f"propagated_volume_{n}.npy")
        np.save(intensity_npy_file, intensity_volume.astype(np.float32))
        print(f"    [OK] Saved propagated volume NPY: {intensity_npy_file}")
    
    # Volume segmentation TIFF multistack
    if save_options.get('segmentation_tiff', False):
        bin_tiff_file = os.path.join(chemin_binary, f"segmentation_{n}.tiff")
        save_volume_as_tiff(bin_tiff_file, bool_volume)
        print(f"    [OK] Saved segmentation TIFF: {bin_tiff_file}")
    
    # Volume segmentation NPY bool
    if save_options.get('segmentation_npy', False):
        bin_npy_file = os.path.join(chemin_binary, f"segmentation_{n}.npy")
        np.save(bin_npy_file, bool_volume_mask)
        print(f"    [OK] Saved segmentation NPY: {bin_npy_file}")
    
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


def simulate_bacteria_random(config, dirs):
    """Simulate random bacteria holograms"""
    # Extract directories from dirs dictionary
    chemin_positions = dirs['object_positions']
    chemin_holograms = dirs['simulated_hologram']
    chemin_data_holo = dirs['hologram_volume']
    
    print("\n" + "="*80)
    print("SIMULATION: BACTERIA (RANDOM)")
    print("="*80)
    
    # Parameters
    nb_holo_to_simulate = config['nb_holo']
    number_of_bacteria = config['nb_objects']
    holo_size_xy = config['holo_size_xy']
    border = config['border']
    upscale_factor = config['upscale_factor']
    z_size = config['z_size']
    
    # Bacteria parameters
    longueur_min_max = {min: config['length_min'], max: config['length_max']}
    epaisseur_min_max = {min: config['thickness_min'], max: config['thickness_max']}
    
    # Optical parameters
    pix_size = config['pix_size']
    grossissement = config['magnification']
    index_milieu = config['index_medium']
    index_objet = config['index_object']
    wavelength = config['wavelength']
    
    # Volume parameters
    holo_size_xy_w_b = holo_size_xy + border * 2
    vox_size_xy = pix_size / grossissement
    vox_size_z = 100e-6 / z_size
    lambda_milieu = wavelength / index_milieu
    
    volume_size = [holo_size_xy, holo_size_xy, z_size]
    volume_size_w_b = [holo_size_xy_w_b, holo_size_xy_w_b, z_size]
    volume_size_upscaled = [holo_size_xy * upscale_factor, holo_size_xy * upscale_factor, z_size]
    
    # Phase shift parameters
    shift_in_env = 0.0
    shift_in_obj = 2.0 * cp.pi * vox_size_z * (index_objet - index_milieu) / wavelength
    
    # GPU allocations
    d_fft_holo = cp.zeros(shape=(holo_size_xy_w_b, holo_size_xy_w_b), dtype=cp.complex64)
    d_fft_holo_propag = cp.zeros(shape=(holo_size_xy_w_b, holo_size_xy_w_b), dtype=cp.complex64)
    d_holo_propag = cp.zeros(shape=(holo_size_xy_w_b, holo_size_xy_w_b), dtype=cp.float32)
    d_KERNEL = cp.zeros(shape=(holo_size_xy_w_b, holo_size_xy_w_b), dtype=cp.complex64)
    
    parameters = {
        'holo_size_x': holo_size_xy,
        'holo_size_y': holo_size_xy,
        'holo_plane_number': z_size,
        'medium_index': index_milieu,
        'object_index': index_objet,
        'pix_size_cam': pix_size,
        'magnification_cam': grossissement,
        'Z_step': vox_size_z,
        'illumination_wavelength': wavelength,
        'medium_wavelength': lambda_milieu
    }
    
    rnd = np.random.default_rng()
    
    for n in range(nb_holo_to_simulate):
        print(f"\n[{n+1}/{nb_holo_to_simulate}] Generating hologram...")
        
        data_file = os.path.join(chemin_data_holo, f"data_{n}.npz")
        holo_file = os.path.join(chemin_holograms, f"holo_{n}.bmp")
        bin_tiff_file = os.path.join(chemin_holograms, f"bin_volume_{n}.tiff")
        intensity_tiff_file = os.path.join(chemin_holograms, f"intensity_volume_{n}.tiff")
        positions_file = os.path.join(chemin_positions, f"bact_{n}.txt")
        
        # Create illumination field
        np_field_plane = np.full(shape=[holo_size_xy_w_b, holo_size_xy_w_b], 
                                fill_value=0.0+0.0j, dtype=cp.complex64)
        ecart_type_bruit = ((config['noise_std_max'] - config['noise_std_min']) * rnd.random() + 
                           config['noise_std_min'])
        bruit_gaussien = np.abs(np.random.normal(config['illumination_mean'], ecart_type_bruit, 
                                                 [holo_size_xy_w_b, holo_size_xy_w_b]))
        np_field_plane.real = np.sqrt(bruit_gaussien)
        cp_field_plane = cp.asarray(np_field_plane)
        
        # Initialize volumes
        cp_mask_volume = cp.full(shape=volume_size, fill_value=0, dtype=cp.float16)
        cp_mask_volume_upscaled = cp.full(shape=volume_size_upscaled, fill_value=0, dtype=cp.float16)
        
        # Generate bacteria
        print(f"  - Generating {number_of_bacteria} bacteria...")
        liste_bacteries = gen_random_bacteria(
            number_of_bact=number_of_bacteria,
            xyz_min_max=[0, holo_size_xy * vox_size_xy,
                        0, holo_size_xy * vox_size_xy,
                        0, z_size * vox_size_z],
            thickness_min_max=epaisseur_min_max,
            length_min_max=longueur_min_max
        )
        
        # Save bacteria positions
        print("  - Saving bacteria positions...")
        for bact in liste_bacteries:
            bact.to_file(positions_file)
        
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
        
        # Insert bacteria in volume
        print("  - Inserting bacteria in volume...")
        for i, bact in enumerate(liste_bacteries):
            if (i + 1) % max(1, number_of_bacteria // 10) == 0:
                print(f"    - {i+1}/{number_of_bacteria}")
            GPU_insert_bact_in_mask_volume(cp_mask_volume_upscaled, bact, 
                                          vox_size_xy / upscale_factor, vox_size_z)
        
        # Flip Z axis
        cp_mask_volume_upscaled = cp.flip(cp_mask_volume_upscaled, axis=2)
        cp_mask_volume = cp_mask_volume_upscaled[:,:,:].reshape(
            holo_size_xy, upscale_factor, holo_size_xy, upscale_factor, z_size
        ).mean(axis=(1, 3))
        
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
        
        # Crop and save
        cropped_field_plane = cp_field_plane[border:border+holo_size_xy, 
                                             border:border+holo_size_xy]
        
        intensity_image = cp.asnumpy(traitement_holo.intensite(cropped_field_plane))
        bool_volume_mask = cp.asnumpy(cp_mask_volume > 0.0)
        
        print("  - Saving results...")
        save_hologram_results(
            dirs, n,
            intensity_image,
            cp.asnumpy(traitement_holo.intensite(cp_mask_volume.astype(cp.float32))),
            bool_volume_mask,
            liste_bacteries,
            config.get('save_options', {}),
            vox_size_xy, vox_size_z,
            object_type='bacteria'
        )
        
        # Also save NPZ data file if needed
        save_holo_data(
            data_file,
            bool_volume_mask,
            intensity_image,
            parameters,
            bacteria_list
        )
        
        print(f"  ✓ Hologram {n} complete")


def simulate_bacteria_list(config, dirs):
    """Simulate bacteria holograms from a predefined list"""
    # Extract directories from dirs dictionary
    chemin_positions = dirs['object_positions']
    chemin_holograms = dirs['simulated_hologram']
    chemin_data_holo = dirs['hologram_volume']
    
    print("\n" + "="*80)
    print("SIMULATION: BACTERIA (FROM LIST)")
    print("="*80)
    
    # Parameters
    nb_holo_to_simulate = config['nb_holo']
    holo_size_xy = config['holo_size_xy']
    border = config['border']
    upscale_factor = config['upscale_factor']
    z_size = config['z_size']
    
    # Optical parameters
    pix_size = config['pix_size']
    grossissement = config['magnification']
    index_milieu = config['index_medium']
    index_objet = config['index_object']
    wavelength = config['wavelength']
    
    # Volume parameters
    holo_size_xy_w_b = holo_size_xy + border * 2
    vox_size_xy = pix_size / grossissement
    vox_size_z = 100e-6 / z_size
    lambda_milieu = wavelength / index_milieu
    
    volume_size = [holo_size_xy, holo_size_xy, z_size]
    volume_size_w_b = [holo_size_xy_w_b, holo_size_xy_w_b, z_size]
    volume_size_upscaled = [holo_size_xy * upscale_factor, holo_size_xy * upscale_factor, z_size]
    
    # Phase shift parameters
    shift_in_env = 0.0
    shift_in_obj = 2.0 * cp.pi * vox_size_z * (index_objet - index_milieu) / wavelength
    
    # GPU allocations
    d_fft_holo = cp.zeros(shape=(holo_size_xy_w_b, holo_size_xy_w_b), dtype=cp.complex64)
    d_fft_holo_propag = cp.zeros(shape=(holo_size_xy_w_b, holo_size_xy_w_b), dtype=cp.complex64)
    d_holo_propag = cp.zeros(shape=(holo_size_xy_w_b, holo_size_xy_w_b), dtype=cp.float32)
    d_KERNEL = cp.zeros(shape=(holo_size_xy_w_b, holo_size_xy_w_b), dtype=cp.complex64)
    
    parameters = {
        'holo_size_x': holo_size_xy,
        'holo_size_y': holo_size_xy,
        'holo_plane_number': z_size,
        'medium_index': index_milieu,
        'object_index': index_objet,
        'pix_size_cam': pix_size,
        'magnification_cam': grossissement,
        'Z_step': vox_size_z,
        'illumination_wavelength': wavelength,
        'medium_wavelength': lambda_milieu
    }
    
    rnd = np.random.default_rng()
    
    # Get bacteria from config
    bacteria_list_config = config.get('bacteria', [])
    if not bacteria_list_config:
        print("ERROR: No bacteria defined in config['bacteria']")
        return
    
    for n in range(nb_holo_to_simulate):
        print(f"\n[{n+1}/{nb_holo_to_simulate}] Generating hologram from bacteria list...")
        
        data_file = os.path.join(chemin_data_holo, f"data_{n}.npz")
        holo_file = os.path.join(chemin_holograms, f"holo_{n}.bmp")
        bin_tiff_file = os.path.join(chemin_holograms, f"bin_volume_{n}.tiff")
        intensity_tiff_file = os.path.join(chemin_holograms, f"intensity_volume_{n}.tiff")
        positions_file = os.path.join(chemin_positions, f"bact_{n}.txt")
        
        # Create illumination field
        np_field_plane = np.full(shape=[holo_size_xy_w_b, holo_size_xy_w_b], 
                                fill_value=0.0+0.0j, dtype=cp.complex64)
        ecart_type_bruit = ((config['noise_std_max'] - config['noise_std_min']) * rnd.random() + 
                           config['noise_std_min'])
        bruit_gaussien = np.abs(np.random.normal(config['illumination_mean'], ecart_type_bruit, 
                                                 [holo_size_xy_w_b, holo_size_xy_w_b]))
        np_field_plane.real = np.sqrt(bruit_gaussien)
        cp_field_plane = cp.asarray(np_field_plane)
        
        # Initialize volumes
        cp_mask_volume = cp.full(shape=volume_size, fill_value=0, dtype=cp.float16)
        cp_mask_volume_upscaled = cp.full(shape=volume_size_upscaled, fill_value=0, dtype=cp.float16)
        
        # Create bacteria from list config
        print(f"  - Creating {len(bacteria_list_config)} bacteria from list...")
        liste_bacteries = []
        bacteria_list_output = []
        
        for bact_config in bacteria_list_config:
            bact = Bacterie(
                pos_x=bact_config['pos_x'] + border * vox_size_xy,
                pos_y=bact_config['pos_y'] + border * vox_size_xy,
                pos_z=bact_config['pos_z'],
                length=bact_config['length'],
                thickness=bact_config['thickness'],
                theta=bact_config.get('theta', 0.0),
                phi=bact_config.get('phi', 0.0)
            )
            liste_bacteries.append(bact)
            bacteria_list_output.append({
                "thickness": bact.thickness,
                "length": bact.length,
                "x_position_m": bact.pos_x - border * vox_size_xy,
                "y_position_m": bact.pos_y - border * vox_size_xy,
                "z_position_m": bact.pos_z,
                "theta_angle": bact.theta,
                "phi_angle": bact.phi
            })
        
        # Save bacteria positions
        print("  - Saving bacteria positions...")
        with open(positions_file, 'w') as f:
            for bact in liste_bacteries:
                f.write(f"{bact.pos_x - border*vox_size_xy}\t{bact.pos_y - border*vox_size_xy}\t"
                       f"{bact.pos_z}\t{bact.length}\t{bact.thickness}\t"
                       f"{bact.theta}\t{bact.phi}\n")
        
        # Insert bacteria in volume
        print("  - Inserting bacteria in volume...")
        for i, bact in enumerate(liste_bacteries):
            GPU_insert_bact_in_mask_volume(cp_mask_volume_upscaled, bact, 
                                          vox_size_xy / upscale_factor, vox_size_z)
        
        # Flip Z axis
        cp_mask_volume_upscaled = cp.flip(cp_mask_volume_upscaled, axis=2)
        cp_mask_volume = cp_mask_volume_upscaled[:,:,:].reshape(
            holo_size_xy, upscale_factor, holo_size_xy, upscale_factor, z_size
        ).mean(axis=(1, 3))
        
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
            
            cp_field_plane = phase_shift_through_plane(
                mask_plane=cp_mask_plane_w_border,
                plane_to_shift=cp_field_plane,
                shift_in_env=shift_in_env,
                shift_in_obj=shift_in_obj
            )
        
        # Crop and save
        cropped_field_plane = cp_field_plane[border:border+holo_size_xy, 
                                             border:border+holo_size_xy]
        
        intensity_image = cp.asnumpy(traitement_holo.intensite(cropped_field_plane))
        bool_volume_mask = cp.asnumpy(cp_mask_volume > 0.0)
        
        print("  - Saving results...")
        save_hologram_results(
            dirs, n,
            intensity_image,
            cp.asnumpy(traitement_holo.intensite(cp_mask_volume.astype(cp.float32))),
            bool_volume_mask,
            liste_bacteries,
            config.get('save_options', {}),
            vox_size_xy, vox_size_z,
            object_type='bacteria'
        )
        
        # Also save NPZ data file if needed
        save_holo_data(
            data_file,
            bool_volume_mask,
            intensity_image,
            parameters,
            bacteria_list_output
        )
        
        print(f"  ✓ Hologram {n} complete")


def simulate_sphere_random(config, dirs):
    """Simulate random sphere holograms"""
    # Extract directories from dirs dictionary
    chemin_positions = dirs['object_positions']
    chemin_holograms = dirs['simulated_hologram']
    chemin_data_holo = dirs['hologram_volume']
    
    print("\n" + "="*80)
    print("SIMULATION: SPHERES (RANDOM)")
    print("="*80)
    
    # Parameters
    nb_holo_to_simulate = config['nb_holo']
    number_of_spheres = config['nb_objects']
    holo_size_xy = config['holo_size_xy']
    border = config['border']
    upscale_factor = config['upscale_factor']
    z_size = config['z_size']
    
    # Sphere parameters
    rayon_min_max = {min: config['radius_min'], max: config['radius_max']}
    
    # Optical parameters
    pix_size = config['pix_size']
    grossissement = config['magnification']
    index_milieu = config['index_medium']
    index_objet = config['index_object']
    wavelength = config['wavelength']
    
    # Volume parameters
    holo_size_xy_w_b = holo_size_xy + border * 2
    vox_size_xy = pix_size / grossissement
    vox_size_z = 100e-6 / z_size
    lambda_milieu = wavelength / index_milieu
    
    volume_size = [holo_size_xy, holo_size_xy, z_size]
    volume_size_w_b = [holo_size_xy_w_b, holo_size_xy_w_b, z_size]
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
    
    parameters = {
        'holo_size_x': holo_size_xy,
        'holo_size_y': holo_size_xy,
        'holo_plane_number': z_size,
        'medium_index': index_milieu,
        'object_index': index_objet,
        'pix_size_cam': pix_size,
        'magnification_cam': grossissement,
        'Z_step': vox_size_z,
        'illumination_wavelength': wavelength,
        'medium_wavelength': lambda_milieu
    }
    
    rnd = np.random.default_rng()
    
    # Volume constraints - spheres inserted in central region
    volume_min_max = [border*vox_size_xy, (border+holo_size_xy)*vox_size_xy,
                      border*vox_size_xy, (border+holo_size_xy)*vox_size_xy,
                      0, z_size * vox_size_z]
    
    for n in range(nb_holo_to_simulate):
        print(f"\n[{n+1}/{nb_holo_to_simulate}] Generating random sphere hologram...")
        
        data_file = os.path.join(chemin_data_holo, f"data_{n}.npz")
        holo_file = os.path.join(chemin_holograms, f"holo_{n}.bmp")
        bin_tiff_file = os.path.join(chemin_holograms, f"bin_volume_{n}.tiff")
        intensity_tiff_file = os.path.join(chemin_holograms, f"intensity_volume_{n}.tiff")
        positions_file = os.path.join(chemin_positions, f"spheres_{n}.txt")
        
        # Create illumination field
        np_field_plane = np.full(shape=[holo_size_xy_w_b, holo_size_xy_w_b], 
                                fill_value=0.0+0.0j, dtype=cp.complex64)
        ecart_type_bruit = ((config['noise_std_max'] - config['noise_std_min']) * rnd.random() + 
                           config['noise_std_min'])
        bruit_gaussien = np.abs(np.random.normal(config['illumination_mean'], ecart_type_bruit, 
                                                 [holo_size_xy_w_b, holo_size_xy_w_b]))
        np_field_plane.real = np.sqrt(bruit_gaussien)
        cp_field_plane = cp.asarray(np_field_plane)
        
        # Initialize volumes
        cp_mask_volume = cp.full(shape=volume_size, fill_value=0, dtype=cp.float16)
        cp_mask_volume_upscaled = cp.full(shape=volume_size_upscaled, fill_value=0, dtype=cp.float16)
        
        # Generate random spheres
        print(f"  - Generating {number_of_spheres} random spheres...")
        liste_spheres = gen_random_sphere(number_of_spheres, volume_min_max, rayon_min_max)
        
        # Save sphere positions
        print("  - Saving sphere positions...")
        with open(positions_file, 'w') as f:
            for sphere in liste_spheres:
                f.write(f"{sphere.pos_x - border*vox_size_xy}\t{sphere.pos_y - border*vox_size_xy}\t"
                       f"{sphere.pos_z}\t{sphere.radius}\n")
        
        spheres_list_output = [
            {
                "radius": s.radius,
                "x_position_m": s.pos_x - border*vox_size_xy,
                "y_position_m": s.pos_y - border*vox_size_xy,
                "z_position_m": s.pos_z
            }
            for s in liste_spheres
        ]
        
        # Insert spheres in volume
        print("  - Inserting spheres in volume...")
        for i, sphere in enumerate(liste_spheres):
            insert_sphere_in_mask_volume(cp_mask_volume_upscaled, sphere, 
                                        vox_size_xy / upscale_factor, vox_size_z, upscale_factor)
        
        # Flip Z axis
        cp_mask_volume_upscaled = cp.flip(cp_mask_volume_upscaled, axis=2)
        cp_mask_volume = cp_mask_volume_upscaled[:,:,:].reshape(
            holo_size_xy, upscale_factor, holo_size_xy, upscale_factor, z_size
        ).mean(axis=(1, 3))
        
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
        
        # Crop and save
        cropped_field_plane = cp_field_plane[border:border+holo_size_xy, 
                                             border:border+holo_size_xy]
        
        intensity_image = cp.asnumpy(traitement_holo.intensite(cropped_field_plane))
        bool_volume_mask = cp.asnumpy(cp_mask_volume > 0.0)
        
        print("  - Saving results...")
        save_hologram_results(
            dirs, n,
            intensity_image,
            cp.asnumpy(traitement_holo.intensite(cp_mask_volume.astype(cp.float32))),
            bool_volume_mask,
            liste_spheres,
            config.get('save_options', {}),
            vox_size_xy, vox_size_z,
            object_type='sphere'
        )
        
        # Also save NPZ data file if needed
        save_holo_data(
            data_file,
            bool_volume_mask,
            intensity_image,
            parameters,
            spheres_list_output
        )
        
        print(f"  ✓ Hologram {n} complete")


def simulate_sphere_list(config, dirs):
    """Simulate sphere holograms from a predefined list"""
    # Extract directories from dirs dictionary
    chemin_positions = dirs['object_positions']
    chemin_holograms = dirs['simulated_hologram']
    chemin_data_holo = dirs['hologram_volume']
    
    print("\n" + "="*80)
    print("SIMULATION: SPHERES (FROM LIST)")
    print("="*80)
    
    # Parameters
    nb_holo_to_simulate = config['nb_holo']
    holo_size_xy = config['holo_size_xy']
    border = config['border']
    upscale_factor = config['upscale_factor']
    z_size = config['z_size']
    
    # Optical parameters
    pix_size = config['pix_size']
    grossissement = config['magnification']
    index_milieu = config['index_medium']
    index_objet = config['index_object']
    wavelength = config['wavelength']
    
    # Volume parameters
    holo_size_xy_w_b = holo_size_xy + border * 2
    vox_size_xy = pix_size / grossissement
    vox_size_z = 100e-6 / z_size
    lambda_milieu = wavelength / index_milieu
    
    volume_size = [holo_size_xy, holo_size_xy, z_size]
    volume_size_w_b = [holo_size_xy_w_b, holo_size_xy_w_b, z_size]
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
    
    parameters = {
        'holo_size_x': holo_size_xy,
        'holo_size_y': holo_size_xy,
        'holo_plane_number': z_size,
        'medium_index': index_milieu,
        'object_index': index_objet,
        'pix_size_cam': pix_size,
        'magnification_cam': grossissement,
        'Z_step': vox_size_z,
        'illumination_wavelength': wavelength,
        'medium_wavelength': lambda_milieu
    }
    
    rnd = np.random.default_rng()
    
    # Get spheres from config
    spheres_list_config = config.get('spheres', [])
    if not spheres_list_config:
        print("ERROR: No spheres defined in config['spheres']")
        return
    
    for n in range(nb_holo_to_simulate):
        print(f"\n[{n+1}/{nb_holo_to_simulate}] Generating hologram from sphere list...")
        
        data_file = os.path.join(chemin_data_holo, f"data_{n}.npz")
        holo_file = os.path.join(chemin_holograms, f"holo_{n}.bmp")
        bin_tiff_file = os.path.join(chemin_holograms, f"bin_volume_{n}.tiff")
        intensity_tiff_file = os.path.join(chemin_holograms, f"intensity_volume_{n}.tiff")
        positions_file = os.path.join(chemin_positions, f"spheres_{n}.txt")
        
        # Create illumination field
        np_field_plane = np.full(shape=[holo_size_xy_w_b, holo_size_xy_w_b], 
                                fill_value=0.0+0.0j, dtype=cp.complex64)
        ecart_type_bruit = ((config['noise_std_max'] - config['noise_std_min']) * rnd.random() + 
                           config['noise_std_min'])
        bruit_gaussien = np.abs(np.random.normal(config['illumination_mean'], ecart_type_bruit, 
                                                 [holo_size_xy_w_b, holo_size_xy_w_b]))
        np_field_plane.real = np.sqrt(bruit_gaussien)
        cp_field_plane = cp.asarray(np_field_plane)
        
        # Initialize volumes
        cp_mask_volume = cp.full(shape=volume_size, fill_value=0, dtype=cp.float16)
        cp_mask_volume_upscaled = cp.full(shape=volume_size_upscaled, fill_value=0, dtype=cp.float16)
        
        # Create spheres from list config
        print(f"  - Creating {len(spheres_list_config)} spheres from list...")
        liste_spheres = []
        spheres_list_output = []
        
        for sphere_config in spheres_list_config:
            sphere = Sphere(
                pos_x=sphere_config['pos_x'] + border * vox_size_xy,
                pos_y=sphere_config['pos_y'] + border * vox_size_xy,
                pos_z=sphere_config['pos_z'],
                radius=sphere_config['radius']
            )
            liste_spheres.append(sphere)
            spheres_list_output.append({
                "radius": sphere.radius,
                "x_position_m": sphere.pos_x - border*vox_size_xy,
                "y_position_m": sphere.pos_y - border*vox_size_xy,
                "z_position_m": sphere.pos_z
            })
        
        # Save sphere positions
        print("  - Saving sphere positions...")
        with open(positions_file, 'w') as f:
            for sphere in liste_spheres:
                f.write(f"{sphere.pos_x - border*vox_size_xy}\t{sphere.pos_y - border*vox_size_xy}\t"
                       f"{sphere.pos_z}\t{sphere.radius}\n")
        
        # Insert spheres in volume
        print("  - Inserting spheres in volume...")
        for i, sphere in enumerate(liste_spheres):
            insert_sphere_in_mask_volume(cp_mask_volume_upscaled, sphere, 
                                        vox_size_xy / upscale_factor, vox_size_z, upscale_factor)
        
        # Flip Z axis
        cp_mask_volume_upscaled = cp.flip(cp_mask_volume_upscaled, axis=2)
        cp_mask_volume = cp_mask_volume_upscaled[:,:,:].reshape(
            holo_size_xy, upscale_factor, holo_size_xy, upscale_factor, z_size
        ).mean(axis=(1, 3))
        
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
        
        # Crop and save
        cropped_field_plane = cp_field_plane[border:border+holo_size_xy, 
                                             border:border+holo_size_xy]
        
        intensity_image = cp.asnumpy(traitement_holo.intensite(cropped_field_plane))
        bool_volume_mask = cp.asnumpy(cp_mask_volume > 0.0)
        
        print("  - Saving results...")
        save_hologram_results(
            dirs, n,
            intensity_image,
            cp.asnumpy(traitement_holo.intensite(cp_mask_volume.astype(cp.float32))),
            bool_volume_mask,
            liste_spheres,
            config.get('save_options', {}),
            vox_size_xy, vox_size_z,
            object_type='sphere'
        )
        
        # Also save NPZ data file if needed
        save_holo_data(
            data_file,
            bool_volume_mask,
            intensity_image,
            parameters,
            spheres_list_output
        )
        
        print(f"  ✓ Hologram {n} complete")


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
    
    parser.add_argument('config_file', type=str, help='JSON configuration file')
    args = parser.parse_args()
    
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
    
    dirs = setup_directories(base_path)
    
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
    
    if 'bacteria' in config['mode']:
        print(f"Number of bacteria per hologram: {config['nb_objects']}")
        print(f"Bacteria length: {config['length_min']*1e6:.2f} - {config['length_max']*1e6:.2f} µm")
        print(f"Bacteria thickness: {config['thickness_min']*1e6:.3f} - {config['thickness_max']*1e6:.3f} µm")
    elif 'sphere' in config['mode']:
        print(f"Number of spheres per hologram: {config['nb_objects']}")
        print(f"Sphere radius: {config['radius_min']*1e6:.3f} - {config['radius_max']*1e6:.3f} µm")
    print("="*80 + "\n")
    
    # Run simulation based on mode
    if config['mode'] == 'bacteria_random':
        simulate_bacteria_random(config, dirs)
    elif config['mode'] == 'bacteria_list':
        simulate_bacteria_list(config, dirs)
    elif config['mode'] == 'sphere_random':
        simulate_sphere_random(config, dirs)
    elif config['mode'] == 'sphere_list':
        simulate_sphere_list(config, dirs)
    
    print("\n" + "="*80)
    print("SIMULATION COMPLETE")
    print("="*80)

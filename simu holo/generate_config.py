#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Helper script to generate configuration JSON files for simulations
"""

import json
import os
from pathlib import Path


def create_config(
    mode="bacteria_random",
    nb_holo=100,
    nb_objects=50,
    holo_size_xy=1024,
    border=256,
    z_size=200,
    length_min=3.0e-6,
    length_max=4.0e-6,
    thickness_min=1.0e-6,
    thickness_max=2.0e-6,
    radius_min=0.5e-6,
    radius_max=2.0e-6,
    pix_size=5.5e-6,
    magnification=40.0,
    index_medium=1.33,
    index_object=1.335,
    wavelength=660e-9,
    output_dir=None,
    **kwargs
):
    """
    Create a configuration dictionary
    
    Parameters
    ----------
    mode : str
        'bacteria_random', 'bacteria_list', 'sphere_random', 'sphere_list'
    nb_holo : int
        Number of holograms to generate
    nb_objects : int
        Number of objects per hologram
    holo_size_xy : int
        Hologram XY size in pixels
    border : int
        Border size in pixels
    z_size : int
        Number of Z planes
    length_min, length_max : float
        Bacteria length range (m)
    thickness_min, thickness_max : float
        Bacteria thickness range (m)
    radius_min, radius_max : float
        Sphere radius range (m)
    pix_size : float
        Camera pixel size (m)
    magnification : float
        Microscope magnification
    index_medium : float
        Refractive index of medium
    index_object : float
        Refractive index of object
    wavelength : float
        Illumination wavelength (m)
    output_dir : str or None
        Output directory (None = auto)
    **kwargs : dict
        Additional parameters (noise_std_min, noise_std_max, etc.)
    """
    
    config = {
        "mode": mode,
        "nb_holo": nb_holo,
        "output_dir": output_dir,
        "nb_objects": nb_objects,
        "holo_size_xy": holo_size_xy,
        "border": border,
        "upscale_factor": kwargs.get('upscale_factor', 2),
        "z_size": z_size,
    }
    
    # Add bacteria parameters if relevant
    if "bacteria" in mode or "bacteria" in kwargs.get("object_type", ""):
        config.update({
            "length_min": length_min,
            "length_max": length_max,
            "thickness_min": thickness_min,
            "thickness_max": thickness_max,
        })
    
    # Add sphere parameters if relevant
    if "sphere" in mode or "sphere" in kwargs.get("object_type", ""):
        config.update({
            "radius_min": radius_min,
            "radius_max": radius_max,
        })
    
    # Add optical parameters
    config.update({
        "pix_size": pix_size,
        "magnification": magnification,
        "index_medium": index_medium,
        "index_object": index_object,
        "wavelength": wavelength,
    })
    
    # Add illumination parameters
    config.update({
        "illumination_mean": kwargs.get('illumination_mean', 1.0),
        "noise_std_min": kwargs.get('noise_std_min', 0.01),
        "noise_std_max": kwargs.get('noise_std_max', 0.1),
    })
    
    return config


def save_config(config, filename):
    """Save configuration to JSON file"""
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)
    print(f"✓ Configuration saved to: {filename}")


# Presets for common scenarios
PRESETS = {
    "bacteria_small": {
        "mode": "bacteria_random",
        "nb_holo": 10,
        "nb_objects": 10,
        "holo_size_xy": 512,
        "z_size": 100,
    },
    
    "bacteria_medium": {
        "mode": "bacteria_random",
        "nb_holo": 100,
        "nb_objects": 50,
        "holo_size_xy": 1024,
        "z_size": 200,
    },
    
    "bacteria_large": {
        "mode": "bacteria_random",
        "nb_holo": 500,
        "nb_objects": 100,
        "holo_size_xy": 2048,
        "border": 512,
        "z_size": 300,
        "upscale_factor": 2,
    },
    
    "bacteria_uv": {
        "mode": "bacteria_random",
        "nb_holo": 100,
        "nb_objects": 50,
        "wavelength": 405e-9,
        "magnification": 60.0,
    },
    
    "sphere_small": {
        "mode": "sphere_random",
        "nb_holo": 10,
        "nb_objects": 20,
        "holo_size_xy": 1024,
        "radius_min": 0.5e-6,
        "radius_max": 2.0e-6,
    },
    
    "sphere_large": {
        "mode": "sphere_random",
        "nb_holo": 100,
        "nb_objects": 50,
        "holo_size_xy": 2048,
        "radius_min": 1.0e-6,
        "radius_max": 5.0e-6,
    },
}


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python generate_config.py <preset_name> [output_file]")
        print("\nAvailable presets:")
        for preset_name in PRESETS.keys():
            print(f"  - {preset_name}")
        sys.exit(1)
    
    preset_name = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else f"config_{preset_name}.json"
    
    if preset_name not in PRESETS:
        print(f"Error: Unknown preset '{preset_name}'")
        print(f"Available presets: {', '.join(PRESETS.keys())}")
        sys.exit(1)
    
    # Create config from preset and save
    config_dict = create_config(**PRESETS[preset_name])
    save_config(config_dict, output_file)
    
    # Print config summary
    print("\nConfiguration Summary:")
    print(f"  Mode: {config_dict['mode']}")
    print(f"  Holograms: {config_dict['nb_holo']}")
    print(f"  Objects per hologram: {config_dict['nb_objects']}")
    print(f"  Hologram size: {config_dict['holo_size_xy']}x{config_dict['holo_size_xy']} pixels")
    print(f"  Z planes: {config_dict['z_size']}")
    print(f"  Wavelength: {config_dict['wavelength']*1e9:.1f} nm")

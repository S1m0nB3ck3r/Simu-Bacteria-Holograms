"""
Data splitting tool for hologram segmentation training

Splits hologram and segmentation data into train/val/test sets.

Usage:
    python split_data.py config_split.json
"""

import os
import sys
import json
import shutil
from pathlib import Path
import random
import numpy as np


def load_split_config(config_path):
    """Load split configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def verify_data_correspondence(holo_dir, seg_dir, holo_prefix='holo_', seg_prefix='segmentation_'):
    """
    Verify that hologram and segmentation files correspond
    
    Args:
        holo_dir: Directory with hologram files
        seg_dir: Directory with segmentation files
        holo_prefix: Prefix for hologram files (e.g., 'holo_')
        seg_prefix: Prefix for segmentation files (e.g., 'segmentation_')
    
    Returns:
        List of indices that exist in both directories
    """
    # Extract indices from hologram files
    holo_indices = set()
    for fname in os.listdir(holo_dir):
        if fname.endswith('.npy') and fname.startswith(holo_prefix):
            try:
                # Extract number from prefix_X.npy
                idx = fname.replace(holo_prefix, '').replace('.npy', '')
                holo_indices.add(idx)
            except:
                pass
    
    # Extract indices from segmentation files
    seg_indices = set()
    for fname in os.listdir(seg_dir):
        if fname.endswith('.npy') and fname.startswith(seg_prefix):
            try:
                # Extract number from prefix_X.npy
                idx = fname.replace(seg_prefix, '').replace('.npy', '')
                seg_indices.add(idx)
            except:
                pass
    
    # Find intersection
    common_indices = sorted(list(holo_indices.intersection(seg_indices)))
    
    print(f"\nData verification:")
    print(f"  Hologram files (prefix='{holo_prefix}'): {len(holo_indices)}")
    print(f"  Segmentation files (prefix='{seg_prefix}'): {len(seg_indices)}")
    print(f"  Matching pairs: {len(common_indices)}")
    
    if len(common_indices) == 0:
        raise ValueError(f"No matching pairs found! Check file naming convention and prefixes.")
    
    missing_in_seg = holo_indices - seg_indices
    missing_in_holo = seg_indices - holo_indices
    
    if missing_in_seg:
        print(f"  Warning: {len(missing_in_seg)} hologram files without segmentation")
    if missing_in_holo:
        print(f"  Warning: {len(missing_in_holo)} segmentation files without hologram")
    
    return common_indices


def split_indices(indices, ratios, seed=42):
    """
    Split indices into train/val/test
    
    Args:
        indices: List of indices
        ratios: List of [train_ratio, val_ratio, test_ratio]
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with 'train', 'val', 'test' lists
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Shuffle indices
    indices_shuffled = indices.copy()
    random.shuffle(indices_shuffled)
    
    n_total = len(indices_shuffled)
    n_train = int(n_total * ratios[0])
    n_val = int(n_total * ratios[1])
    
    train_indices = indices_shuffled[:n_train]
    val_indices = indices_shuffled[n_train:n_train+n_val]
    test_indices = indices_shuffled[n_train+n_val:]
    
    print(f"\nSplit (seed={seed}):")
    print(f"  Train: {len(train_indices)} samples ({len(train_indices)/n_total*100:.1f}%)")
    print(f"  Val:   {len(val_indices)} samples ({len(val_indices)/n_total*100:.1f}%)")
    print(f"  Test:  {len(test_indices)} samples ({len(test_indices)/n_total*100:.1f}%)")
    
    return {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices
    }


def copy_files(holo_dir, seg_dir, output_dir, splits, holo_prefix='holo_', seg_prefix='segmentation_', use_symlink=False):
    """
    Copy or symlink files to split directories
    
    Args:
        holo_dir: Source hologram directory
        seg_dir: Source segmentation directory
        output_dir: Output base directory
        splits: Dictionary with train/val/test indices
        holo_prefix: Prefix for hologram files
        seg_prefix: Prefix for segmentation files
        use_symlink: Use symbolic links instead of copying (saves space)
    """
    for split_name, indices in splits.items():
        # Create directories
        split_holo_dir = os.path.join(output_dir, split_name, "simulated_hologram")
        split_seg_dir = os.path.join(output_dir, split_name, "binary_volume")
        
        os.makedirs(split_holo_dir, exist_ok=True)
        os.makedirs(split_seg_dir, exist_ok=True)
        
        print(f"\nProcessing {split_name} split...")
        
        for idx in indices:
            # Hologram file
            holo_src = os.path.join(holo_dir, f"{holo_prefix}{idx}.npy")
            holo_dst = os.path.join(split_holo_dir, f"{holo_prefix}{idx}.npy")
            
            # Segmentation file
            seg_src = os.path.join(seg_dir, f"{seg_prefix}{idx}.npy")
            seg_dst = os.path.join(split_seg_dir, f"{seg_prefix}{idx}.npy")
            
            # Copy or symlink
            if use_symlink:
                if os.path.exists(holo_dst):
                    os.remove(holo_dst)
                if os.path.exists(seg_dst):
                    os.remove(seg_dst)
                os.symlink(os.path.abspath(holo_src), holo_dst)
                os.symlink(os.path.abspath(seg_src), seg_dst)
            else:
                shutil.copy2(holo_src, holo_dst)
                shutil.copy2(seg_src, seg_dst)
        
        print(f"  Copied {len(indices)} pairs to {split_name}/")
    
    print(f"\n✓ All files {'linked' if use_symlink else 'copied'} successfully!")


def save_split_info(output_dir, splits, config):
    """Save split information for reproducibility"""
    split_info = {
        'config': config,
        'splits': {k: [str(i) for i in v] for k, v in splits.items()},
        'counts': {k: len(v) for k, v in splits.items()}
    }
    
    info_path = os.path.join(output_dir, "split_info.json")
    with open(info_path, 'w') as f:
        json.dump(split_info, f, indent=4)
    
    print(f"\nSplit info saved to: {info_path}")


def main():
    # Check command line arguments
    if len(sys.argv) != 2:
        print("Usage: python split_data.py config_split.json")
        print("\nExample config_split.json:")
        print("""{
    "description": "Configuration for data splitting",
    "holo_dir": "path/to/simulated_hologram",
    "seg_dir": "path/to/binary_volume",
    "output_dir": "path/to/data_split",
    "ratios": [0.7, 0.15, 0.15],
    "seed": 42,
    "use_symlink": false,
    "hologram_prefix": "holo_",
    "segmentation_prefix": "segmentation_"
}""")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    # Load configuration
    print(f"Loading configuration from: {config_path}")
    config = load_split_config(config_path)
    
    holo_dir = config['holo_dir']
    seg_dir = config['seg_dir']
    output_dir = config['output_dir']
    ratios = config.get('ratios', [0.7, 0.15, 0.15])
    seed = config.get('seed', 42)
    use_symlink = config.get('use_symlink', False)
    holo_prefix = config.get('hologram_prefix', 'holo_')
    seg_prefix = config.get('segmentation_prefix', 'segmentation_')
    
    # Validate ratios
    if abs(sum(ratios) - 1.0) > 0.01:
        raise ValueError(f"Ratios must sum to 1.0, got {sum(ratios)}")
    
    print("="*80)
    print("DATA SPLITTING TOOL")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Hologram dir:     {holo_dir}")
    print(f"  Segmentation dir: {seg_dir}")
    print(f"  Output dir:       {output_dir}")
    print(f"  Hologram prefix:  {holo_prefix}")
    print(f"  Segmentation prefix: {seg_prefix}")
    print(f"  Ratios:           Train={ratios[0]:.2%}, Val={ratios[1]:.2%}, Test={ratios[2]:.2%}")
    print(f"  Seed:             {seed}")
    print(f"  Mode:             {'Symlink' if use_symlink else 'Copy'}")
    
    # Verify directories exist
    if not os.path.exists(holo_dir):
        raise FileNotFoundError(f"Hologram directory not found: {holo_dir}")
    if not os.path.exists(seg_dir):
        raise FileNotFoundError(f"Segmentation directory not found: {seg_dir}")
    
    # Verify data correspondence
    common_indices = verify_data_correspondence(holo_dir, seg_dir, holo_prefix, seg_prefix)
    
    # Split data
    splits = split_indices(common_indices, ratios, seed)
    
    # Copy/link files
    copy_files(holo_dir, seg_dir, output_dir, splits, holo_prefix, seg_prefix, use_symlink)
    
    # Save split info
    save_split_info(output_dir, splits, config)
    
    print("\n" + "="*80)
    print("SPLIT COMPLETE")
    print("="*80)
    print(f"\nData structure created:")
    print(f"{output_dir}/")
    print(f"├── train/")
    print(f"│   ├── simulated_hologram/")
    print(f"│   └── binary_volume/")
    print(f"├── val/")
    print(f"│   ├── simulated_hologram/")
    print(f"│   └── binary_volume/")
    print(f"├── test/")
    print(f"│   ├── simulated_hologram/")
    print(f"│   └── binary_volume/")
    print(f"└── split_info.json")


if __name__ == "__main__":
    main()

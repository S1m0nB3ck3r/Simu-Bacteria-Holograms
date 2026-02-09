"""
Model definitions and utilities for 3D hologram segmentation

Contains:
- UNet3D: 3D U-Net architecture with skip connections
- VolumeReconstructor: Hologram reconstruction using angular spectrum
- HologramToSegmentationDataset: Dataset for training
- dice_coefficient: Metric function
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import cupy as cp

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import avec le namespace libs
from libs import propagation as propag
from libs import traitement_holo


class VolumeReconstructor:
    """Reconstructs 3D volumes from holograms using angular spectrum propagation"""
    
    def __init__(self, parameters):
        self.parameters = parameters.copy()
        self.nb_pix_X = parameters["holo_size_x"]
        self.nb_pix_Y = parameters["holo_size_y"]
        self.nb_plan = parameters["holo_plane_number"]
        self.lambda_milieu = parameters["medium_wavelength"]
        self.magnification = parameters["magnification_cam"]
        self.pix_size = parameters["pix_size_cam"]
        self.dz = parameters["Z_step"]

        self.d_fft_holo = cp.zeros(shape=(self.nb_pix_X, self.nb_pix_Y), dtype=cp.complex64)
        self.d_fft_holo_propag = cp.zeros(shape=(self.nb_pix_X, self.nb_pix_Y), dtype=cp.complex64)
        self.d_KERNEL = cp.zeros(shape=(self.nb_pix_X, self.nb_pix_Y), dtype=cp.complex64)
        self.d_volume_module = cp.zeros(shape=(self.nb_pix_X, self.nb_pix_Y, self.nb_plan), dtype=cp.float32)

    def volume_reconstruction(self, hologram, parameters):
        """
        Reconstruct 3D volume from hologram using angular spectrum propagation
        
        Returns:
            torch.Tensor: Reconstructed volume with shape (1, D, H, W) where D is depth (Z)
        """
        holo_size_x = parameters['holo_size_x']
        holo_size_y = parameters['holo_size_y']
        holo_plane_number = parameters['holo_plane_number']
        
        # GPU arrays - NOTE: propagation uses (Z, Y, X) ordering
        d_HOLO = cp.array(hologram, dtype=cp.float32)
        d_FFT_HOLO = cp.zeros(shape=(holo_size_y, holo_size_x), dtype=cp.complex64)
        d_KERNEL = cp.zeros(shape=(holo_size_y, holo_size_x), dtype=cp.complex64)
        d_FFT_HOLO_PROPAG = cp.zeros(shape=(holo_size_y, holo_size_x), dtype=cp.complex64)
        
        # Volume in (Z, Y, X) order as expected by propagation
        d_HOLO_VOLUME_PROPAG_MODULE = cp.zeros(
            shape=(holo_plane_number, holo_size_y, holo_size_x), 
            dtype=cp.float32
        )
        
        # Propagation parameters
        medium_wavelength = parameters['medium_wavelength']
        magnification_cam = parameters['magnification_cam']
        pix_size_cam = parameters['pix_size_cam']
        Z_step = parameters['Z_step']
        
        # Reconstruct volume
        propag.volume_propag_angular_spectrum_to_module(
            d_HOLO, d_FFT_HOLO, d_KERNEL, d_FFT_HOLO_PROPAG, 
            d_HOLO_VOLUME_PROPAG_MODULE,
            medium_wavelength, magnification_cam, pix_size_cam,
            holo_size_x, holo_size_y, 0.0, Z_step, holo_plane_number, 0.0, 0.0
        )
        
        # Convert to numpy: shape is (Z, Y, X)
        volume_np = cp.asnumpy(d_HOLO_VOLUME_PROPAG_MODULE)
        
        # Convert to torch and add channel dimension: (1, Z, Y, X) 
        # Note: PyTorch 3D convolutions expect (C, D, H, W) format
        volume_torch = torch.from_numpy(volume_np).unsqueeze(0).float()
        
        return volume_torch


class HologramToSegmentationDataset(Dataset):
    """Dataset with pre-loaded volumes for faster training"""
    
    def __init__(
        self,
        hologram_dir,
        segmentation_dir,
        parameters,
        patch_size_XY=(128, 128),
        stride_XY=(64, 64),
        patch_size_Z=64,
        stride_Z=32,
        hologram_prefix="holo_",
        segmentation_prefix="segmentation_",
        verbose_timing=False,
    ):
        """
        Initialize dataset with pre-loading from separate hologram and segmentation files
        
        Args:
            hologram_dir: Directory containing hologram .npy files (holo_X.npy)
            segmentation_dir: Directory containing segmentation .npy files (segmentation_X.npy)
            parameters: Reconstruction parameters dict
            patch_size_XY: (width, height) of patches
            stride_XY: (stride_x, stride_y) for patch extraction
            patch_size_Z: Depth of patches
            stride_Z: Stride along Z axis
            hologram_prefix: Prefix for hologram files (e.g., "holo_")
            segmentation_prefix: Prefix for segmentation files (e.g., "bin_volume_")
            verbose_timing: If True, print detailed timing for each reconstruction
        """
        self.hologram_dir = hologram_dir
        self.segmentation_dir = segmentation_dir
        self.parameters = parameters
        self.patch_size_XY = patch_size_XY
        self.stride_XY = stride_XY
        self.patch_size_Z = patch_size_Z
        self.stride_Z = stride_Z
        self.hologram_prefix = hologram_prefix
        self.segmentation_prefix = segmentation_prefix
        self.verbose_timing = verbose_timing

        self.reconstructor = VolumeReconstructor(parameters)
        self.slice_index = []
        
        # Get list of hologram files
        holo_files = sorted([f for f in os.listdir(hologram_dir) if f.endswith('.npy') and f.startswith(hologram_prefix)])
        
        # Create mapping: file_idx -> (holo_path, seg_path)
        self.file_paths = {}
        print(f"Indexing {len(holo_files)} files...")
        
        for file_idx, holo_filename in enumerate(holo_files):
            # Extract index from filename
            idx = holo_filename.replace(hologram_prefix, '').replace('.npy', '')
            seg_filename = f"{segmentation_prefix}{idx}.npy"
            
            holo_path = os.path.join(hologram_dir, holo_filename)
            seg_path = os.path.join(segmentation_dir, seg_filename)
            
            # Store paths
            self.file_paths[file_idx] = (holo_path, seg_path)
            
            # Load one sample to get dimensions for patch indexing
            if file_idx == 0:
                bool_volume = np.load(seg_path)
                X, Y, Z = bool_volume.shape  # Stored as (X, Y, Z)
            
            # Index all possible patches for this file in (Z, Y, X) order
            for z in range(0, Z - patch_size_Z + 1, stride_Z):
                for y in range(0, Y - patch_size_XY[1] + 1, stride_XY[1]):
                    for x in range(0, X - patch_size_XY[0] + 1, stride_XY[0]):
                        self.slice_index.append((file_idx, z, y, x))
        
        print(f"Indexing complete. Total patches: {len(self.slice_index)}")
        
        # Cache for on-the-fly reconstruction (only last used volume)
        self.cached_file_idx = None
        self.cached_volume_tensor = None
        self.cached_bool_tensor = None

    def __len__(self):
        return len(self.slice_index)

    def __getitem__(self, idx):
        file_idx, z, y, x = self.slice_index[idx]  # Now in (Z, Y, X) order
        
        # Check if we need to load/reconstruct this volume
        if self.cached_file_idx != file_idx:
            if self.verbose_timing:
                import time
                load_start = time.time()
            
            # Load new volume
            holo_path, seg_path = self.file_paths[file_idx]
            
            if self.verbose_timing:
                load_time = time.time() - load_start
                print(f"    [Volume {file_idx}] Loading files: {load_time:.4f}s", end="")
                recon_start = time.time()
            
            # Load hologram and segmentation (stored as X, Y, Z)
            hologram_image = np.load(holo_path)
            bool_volume_xyz = np.load(seg_path)
            
            if self.verbose_timing:
                io_time = time.time() - recon_start
                print(f" | I/O: {io_time:.4f}s", end="")
                recon_start = time.time()
            
            # Reconstruct volume - returns (1, Z, Y, X)
            volume_tensor = self.reconstructor.volume_reconstruction(hologram_image, self.parameters)
            
            # Convert segmentation from (X, Y, Z) to (Z, Y, X) to match
            bool_volume_zyx = bool_volume_xyz.transpose(2, 1, 0)
            bool_tensor = torch.from_numpy(bool_volume_zyx).unsqueeze(0).to(torch.float32)
            
            if self.verbose_timing:
                recon_time = time.time() - recon_start
                print(f" | Reconstruction: {recon_time:.4f}s")
            
            # Update cache
            self.cached_file_idx = file_idx
            self.cached_volume_tensor = volume_tensor
            self.cached_bool_tensor = bool_tensor
        else:
            # Use cached data
            volume_tensor = self.cached_volume_tensor
            bool_tensor = self.cached_bool_tensor

        px, py = self.patch_size_XY
        pz = self.patch_size_Z

        # Extract patch in (C, Z, Y, X) order - indices already correct
        volume_patch = volume_tensor[:, z:z+pz, y:y+py, x:x+px]
        bool_patch = bool_tensor[:, z:z+pz, y:y+py, x:x+px]

        return volume_patch, bool_patch


class VolumeBatchSampler:
    """
    Custom sampler that processes all patches from one volume before moving to the next.
    Shuffles the order of volumes each epoch, but keeps patches from the same volume together.
    This maximizes cache efficiency for on-the-fly reconstruction.
    """
    
    def __init__(self, dataset, shuffle=True):
        """
        Args:
            dataset: HologramToSegmentationDataset instance
            shuffle: Whether to shuffle the order of volumes
        """
        self.dataset = dataset
        self.shuffle = shuffle
        
        # Group patch indices by file_idx
        self.volume_patches = {}
        for idx, (file_idx, x, y, z) in enumerate(dataset.slice_index):
            if file_idx not in self.volume_patches:
                self.volume_patches[file_idx] = []
            self.volume_patches[file_idx].append(idx)
        
        self.volume_ids = list(self.volume_patches.keys())
        self.epoch = 0
    
    def __iter__(self):
        # Shuffle volume order if requested
        if self.shuffle:
            volume_order = torch.randperm(len(self.volume_ids)).tolist()
        else:
            volume_order = list(range(len(self.volume_ids)))
        
        # Yield all patches from each volume in order
        for vol_idx in volume_order:
            file_idx = self.volume_ids[vol_idx]
            for patch_idx in self.volume_patches[file_idx]:
                yield patch_idx
        
        self.epoch += 1
    
    def __len__(self):
        return len(self.dataset)


class UNet3D(nn.Module):
    """3D U-Net with skip connections for volumetric segmentation"""
    
    def __init__(self, dropout_prob=0.3):
        """
        Initialize 3D U-Net
        
        Args:
            dropout_prob: Dropout probability (0 to disable)
        """
        super(UNet3D, self).__init__()
        self.dropout_prob = dropout_prob
        self.encoder1 = self.block(1, 16, dropout_prob)
        self.encoder2 = self.block(16, 32, dropout_prob)
        self.encoder3 = self.block(32, 64, dropout_prob)
        self.encoder4 = self.block(64, 128, dropout_prob)
        self.pool = nn.MaxPool3d(2)
        self.bottleneck = self.block(128, 256, dropout_prob)
        
        # Upsampling layers
        self.upconv4 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.decoder4 = self.block(256, 128, dropout_prob)  # 256 = 128 (upconv) + 128 (skip)
        
        self.upconv3 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.decoder3 = self.block(128, 64, dropout_prob)  # 128 = 64 (upconv) + 64 (skip)
        
        self.upconv2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.decoder2 = self.block(64, 32, dropout_prob)  # 64 = 32 (upconv) + 32 (skip)
        
        self.upconv1 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.decoder1 = self.block(32, 16, dropout_prob)  # 32 = 16 (upconv) + 16 (skip)
        
        self.final_conv = nn.Conv3d(16, 1, kernel_size=1)

    def block(self, in_channels, out_channels, dropout_prob=0.0):
        """Create a convolutional block with optional dropout"""
        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout_prob > 0:
            layers.append(nn.Dropout3d(dropout_prob))
        layers.extend([
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        ])
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass with skip connections
        
        Args:
            x: Input tensor of shape (B, 1, X, Y, Z)
            
        Returns:
            Output tensor of shape (B, 1, X, Y, Z)
        """
        # Encoder avec skip connections
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder avec skip connections
        d4 = self.upconv4(b)
        d4 = torch.cat([d4, e4], dim=1)  # Concatenation avec e4
        d4 = self.decoder4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)  # Concatenation avec e3
        d3 = self.decoder3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)  # Concatenation avec e2
        d2 = self.decoder2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)  # Concatenation avec e1
        d1 = self.decoder1(d1)
        
        return self.final_conv(d1)


def dice_coefficient(preds, targets, eps=1e-6):
    """
    Calculate Dice coefficient for binary segmentation
    
    Args:
        preds: Predictions (logits or probabilities)
        targets: Ground truth (0 or 1)
        eps: Small epsilon for numerical stability
        
    Returns:
        Dice coefficient (0 to 1, higher is better)
    """
    preds = (torch.sigmoid(preds) > 0.5).float()
    targets = targets.float()
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum()
    dice = (2. * intersection + eps) / (union + eps)
    return dice

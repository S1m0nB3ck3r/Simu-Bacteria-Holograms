"""
Angular Spectrum Propagation (ASP) reconstruction.
"""

import math
import numpy as np
import torch


def angular_spectrum_propagate_stack(
    holo_2d: torch.Tensor,
    wavelength: float,
    pixel_size: float,
    z_list: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Propagate a 2-D hologram to a stack of Z planes via ASP.

    Args:
        holo_2d:    (H, W) float tensor
        wavelength: illumination wavelength [m]
        pixel_size: camera pixel size [m]
        z_list:     (Z,) tensor of propagation distances [m]
        eps:        numerical stability floor

    Returns:
        (Z, H, W) float32 amplitude stack
    """
    device = holo_2d.device
    H, W = holo_2d.shape

    u0 = torch.sqrt(torch.clamp(holo_2d - holo_2d.min(), min=0.0) + eps)
    U0 = torch.fft.fft2(u0)

    fx = torch.fft.fftfreq(W, d=pixel_size, device=device)
    fy = torch.fft.fftfreq(H, d=pixel_size, device=device)
    FY, FX = torch.meshgrid(fy, fx, indexing="ij")

    root_arg = 1.0 - (wavelength * FX) ** 2 - (wavelength * FY) ** 2
    valid = root_arg >= 0
    root = torch.zeros_like(root_arg)
    root[valid] = torch.sqrt(root_arg[valid])
    k = 2.0 * math.pi / wavelength

    vol = []
    for z in z_list:
        H_z = torch.zeros((H, W), dtype=torch.complex64, device=device)
        phase = k * z * root
        H_z[valid] = torch.exp(1j * phase[valid]).to(torch.complex64)
        Uz = torch.fft.ifft2(U0 * H_z)
        vol.append(torch.abs(Uz).to(torch.float32))

    return torch.stack(vol, dim=0)  # (Z, H, W)


@torch.no_grad()
def reconstruct_volume(holo_np: np.ndarray, cfg, device: torch.device) -> np.ndarray:
    """
    Reconstruct and Z-score normalize a 3-D volume from a 2-D hologram.

    Args:
        holo_np: (H, W) float32 numpy array
        cfg:     Cfg object (uses medium_wavelength, pix_size_cam, Z_step,
                             holo_plane_number)
        device:  torch device

    Returns:
        vol_hwz: (H, W, Z) float32, Z-score normalized
    """
    holo = torch.from_numpy(holo_np).float().to(device)
    z_list = (
        torch.arange(cfg.holo_plane_number, device=device, dtype=torch.float32)
        * cfg.Z_step
    )
    vol = angular_spectrum_propagate_stack(
        holo, cfg.medium_wavelength, cfg.pix_size_cam, z_list
    )
    vol_hwz = vol.permute(1, 2, 0).cpu().numpy().astype(np.float32)
    vol_hwz = (vol_hwz - vol_hwz.mean()) / (vol_hwz.std() + 1e-6)
    return vol_hwz
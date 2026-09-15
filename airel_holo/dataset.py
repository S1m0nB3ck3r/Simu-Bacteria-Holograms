"""
Intelligent patch sampling dataset + disk cache management.
Sampling strategy:
  pos_fraction     → patch centered on a known bacterium
  hardneg_fraction → patch near-but-offset from a bacterium
  remainder        → fully random patch
"""

import os
import re
import glob
import json
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import scipy.ndimage as ndi

from reconstruction import reconstruct_volume


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def discover_ids(holo_dir: str, prefix: str = "holo_") -> List[int]:
    """Return sorted list of integer IDs from holo_*.npy files."""
    files = glob.glob(os.path.join(holo_dir, f"{prefix}*.npy"))
    ids = []
    for f in files:
        m = re.search(rf"{prefix}(\d+)\.npy$", os.path.basename(f))
        if m:
            ids.append(int(m.group(1)))
    return sorted(set(ids))


# ---------------------------------------------------------------------------
# Pre-cache
# ---------------------------------------------------------------------------

def precache(
    ids: List[int],
    holo_dir: str,
    bin_dir: str,
    cache_dir: str,
    cfg,
    device: torch.device,
    desc: str = "Caching",
) -> None:
    """Reconstruct + normalize all volumes and save to disk (skip if exists)."""
    os.makedirs(cache_dir, exist_ok=True)
    for idx in tqdm(ids, desc=desc):
        vp = os.path.join(cache_dir, f"vol_{idx}.npy")
        sp = os.path.join(cache_dir, f"seg_{idx}.npy")
        if os.path.exists(vp) and os.path.exists(sp):
            continue
        holo = np.load(os.path.join(holo_dir, f"holo_{idx}.npy"))
        seg  = np.load(os.path.join(bin_dir,  f"bin_volume_{idx}.npy"))
        vol  = reconstruct_volume(holo, cfg, device)
        np.save(vp, vol.astype(np.float32))
        np.save(sp, seg.astype(np.uint8))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class IntelligentPatchDataset(Dataset):
    """
    Patch dataset with intelligent sampling:
      - 70% positive  (centered on a bacterium center from GT)
      - 20% hard-neg  (offset from a bacterium center)
      - 10% random

    Volumes are reconstructed once and cached to disk.
    Bacteria centers are indexed once and cached to JSON.
    """

    def __init__(
        self,
        ids: List[int],
        holo_dir: str,
        bin_dir: str,
        cache_dir: str,
        cfg,
        device: torch.device,
    ):
        self.ids       = ids
        self.holo_dir  = holo_dir
        self.bin_dir   = bin_dir
        self.cache_dir = cache_dir
        self.cfg       = cfg
        self.device    = device
        self.rng       = np.random.default_rng(cfg.seed)
        os.makedirs(cache_dir, exist_ok=True)

        self.bacteria: List[Tuple[int, int, int, int]] = []
        self._build_bacteria_index()

    # ------------------------------------------------------------------
    def _build_bacteria_index(self) -> None:
        index_cache = os.path.join(
            self.cache_dir, f"_bacteria_index_{self.cfg.patch_z}.json"
        )
        if os.path.exists(index_cache):
            print(f"  Loading cached bacteria index: {index_cache}")
            with open(index_cache, "r") as f:
                self.bacteria = [tuple(x) for x in json.load(f)]
            print(f"  {len(self.bacteria)} bacteria in {len(self.ids)} volumes")
            return

        print("Indexing bacteria from GT...")
        for idx in tqdm(self.ids, desc="Bacteria index"):
            seg_path = os.path.join(self.bin_dir, f"bin_volume_{idx}.npy")
            if not os.path.exists(seg_path):
                continue
            seg = np.load(seg_path)
            labeled, n = ndi.label(seg > 0)
            if n == 0:
                continue
            for cy, cx, cz in ndi.center_of_mass(seg > 0, labeled, range(1, n + 1)):
                self.bacteria.append(
                    (idx, int(round(cy)), int(round(cx)), int(round(cz)))
                )

        with open(index_cache, "w") as f:
            json.dump(self.bacteria, f)
        print(f"  {len(self.bacteria)} bacteria indexed (cached)")

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.cfg.samples_per_epoch

    # ------------------------------------------------------------------
    def _cache_paths(self, idx: int) -> Tuple[str, str]:
        return (
            os.path.join(self.cache_dir, f"vol_{idx}.npy"),
            os.path.join(self.cache_dir, f"seg_{idx}.npy"),
        )

    def _ensure_cached(self, idx: int) -> None:
        vp, sp = self._cache_paths(idx)
        if os.path.exists(vp) and os.path.exists(sp):
            return
        holo = np.load(os.path.join(self.holo_dir, f"holo_{idx}.npy"))
        seg  = np.load(os.path.join(self.bin_dir,  f"bin_volume_{idx}.npy"))
        vol  = reconstruct_volume(holo, self.cfg, self.device)
        np.save(vp, vol.astype(np.float32))
        np.save(sp, seg.astype(np.uint8))

    def _crop(
        self, vol: np.ndarray, seg: np.ndarray, cy: int, cx: int, cz: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        H, W, Z = vol.shape
        pxy, pz = self.cfg.patch_xy, self.cfg.patch_z
        y0 = max(0, min(cy - pxy // 2, H - pxy))
        x0 = max(0, min(cx - pxy // 2, W - pxy))
        z0 = max(0, min(cz - pz  // 2, Z - pz))
        return (
            vol[y0:y0+pxy, x0:x0+pxy, z0:z0+pz].copy(),
            seg[y0:y0+pxy, x0:x0+pxy, z0:z0+pz].copy(),
        )

    # ------------------------------------------------------------------
    def __getitem__(self, _) -> Tuple[torch.Tensor, torch.Tensor]:
        c = self.cfg
        r = self.rng.random()
        H, W, Z = 512, 512, c.holo_plane_number

        has_bact = len(self.bacteria) > 0

        if r < c.pos_fraction and has_bact:
            # Positive: centered on a bacterium
            idx, cy, cx, cz = self.bacteria[self.rng.integers(0, len(self.bacteria))]

        elif r < c.pos_fraction + c.hardneg_fraction:
            # Hard negative: near a bacterium but offset
            if has_bact:
                idx, cy, cx, cz = self.bacteria[self.rng.integers(0, len(self.bacteria))]
                cy = int(np.clip(cy + self.rng.integers(-c.patch_xy, c.patch_xy), c.patch_xy // 2, H - c.patch_xy // 2))
                cx = int(np.clip(cx + self.rng.integers(-c.patch_xy, c.patch_xy), c.patch_xy // 2, W - c.patch_xy // 2))
                cz = int(np.clip(cz + self.rng.integers(-c.patch_z,  c.patch_z),  c.patch_z  // 2, Z - c.patch_z  // 2))
            else:
                idx = int(self.ids[self.rng.integers(0, len(self.ids))])
                cy  = int(self.rng.integers(c.patch_xy // 2, H - c.patch_xy // 2))
                cx  = int(self.rng.integers(c.patch_xy // 2, W - c.patch_xy // 2))
                cz  = int(self.rng.integers(c.patch_z  // 2, Z - c.patch_z  // 2))
        else:
            # Random
            idx = int(self.ids[self.rng.integers(0, len(self.ids))])
            cy  = int(self.rng.integers(c.patch_xy // 2, H - c.patch_xy // 2))
            cx  = int(self.rng.integers(c.patch_xy // 2, W - c.patch_xy // 2))
            cz  = int(self.rng.integers(c.patch_z  // 2, Z - c.patch_z  // 2))

        self._ensure_cached(idx)
        vp, sp = self._cache_paths(idx)
        vol = np.load(vp)
        seg = np.load(sp)
        vol_patch, seg_patch = self._crop(vol, seg, cy, cx, cz)

        # ---- Augmentation (controlled by cfg flags) ----
        c = self.cfg

        # 1. Random flips (XYZ axes)
        if c.aug_flips:
            for axis in range(3):
                if self.rng.random() < 0.5:
                    vol_patch = np.flip(vol_patch, axis).copy()
                    seg_patch = np.flip(seg_patch, axis).copy()

        # 2. Random 90° rotations in XY plane
        if c.aug_rotation:
            k = self.rng.integers(0, 4)   # 0, 90, 180, 270
            if k > 0:
                vol_patch = np.rot90(vol_patch, k, axes=(0, 1)).copy()
                seg_patch = np.rot90(seg_patch, k, axes=(0, 1)).copy()

        # 3. Intensity augmentation (vol only — seg is binary, unchanged)
        if c.aug_intensity:
            scale = self.rng.uniform(c.aug_scale_min, c.aug_scale_max)
            vol_patch = (vol_patch * scale).astype(np.float32)
            noise = self.rng.normal(0.0, c.aug_noise_std, vol_patch.shape).astype(np.float32)
            vol_patch = vol_patch + noise

        # (H,W,Z) → (1, Z, H, W) for 3D conv
        vt = torch.from_numpy(vol_patch.transpose(2, 0, 1))[None].float()
        st = torch.from_numpy(seg_patch.transpose(2, 0, 1))[None].float()
        return vt, st


# ---------------------------------------------------------------------------
# Deterministic validation set
# ---------------------------------------------------------------------------

def build_val_set(
    ds: IntelligentPatchDataset,
    n: int,
    seed: int = 0,
) -> list:
    """
    Pre-sample n patches from ds with a fixed seed.
    Returns list of (vt, st) tensors — same every call.

    Usage:
        val_set = build_val_set(ds_val, cfg.val_n_patches, cfg.seed)
        # then pass val_set to validate_patches()
    """
    rng_state = ds.rng  # save
    ds.rng = np.random.default_rng(seed)
    patches = [ds[0] for _ in range(n)]
    ds.rng = rng_state  # restore
    return patches

"""
config.py
=========
Usage:
    from config import load_cfg
    cfg = load_cfg()
    cfg = load_cfg(arch="configs/arch.yaml", input="configs/input.yaml", tl="configs/tl.yaml")
"""

import os
from dataclasses import dataclass
import yaml


def _load(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


@dataclass
class Cfg:
    # ---- input.yaml ----
    holo_plane_number: int   = 100
    bact_numb:         int   = 10

    HOLO_DIR_TRAIN: str = ""
    BIN_DIR_TRAIN:  str = ""
    HOLO_DIR_VAL:   str = ""
    BIN_DIR_VAL:    str = ""
    cache_root:     str = ""

    patch_xy: int = 128
    patch_z:  int = 64

    samples_per_epoch: int   = 10000
    pos_fraction:      float = 0.7
    hardneg_fraction:  float = 0.2

    medium_wavelength: float = 6.6e-7
    pix_size_cam:      float = 1.375e-7
    Z_step:            float = 5e-7

    # ---- arch.yaml ----
    channels:        list  = None
    dropout_prob:    float = 0.3
    norm:            str   = "batchnorm"
    residual:        bool  = False
    anisotropic:     bool  = False

    epochs:          int   = 80
    lr:              float = 1e-3
    batch_size:      int   = 30
    patience:        int   = 10
    num_workers:     int   = 4
    seed:            int   = 42

    loss:            str   = "bce_dice"
    bce_weight:      float = 0.4
    dice_weight:     float = 0.6
    focal_alpha:     float = 0.25
    focal_gamma:     float = 2.0
    tversky_alpha:   float = 0.3
    tversky_beta:    float = 0.7

    val_n_patches:   int   = 500
    run_name:        str   = "baseline"
    checkpoint_dir:  str   = ""

    # ---- augmentation (input.yaml) ----
    aug_flips:       bool  = True
    aug_rotation:    bool  = False
    aug_intensity:   bool  = False
    aug_noise_std:   float = 0.05
    aug_scale_min:   float = 0.8
    aug_scale_max:   float = 1.2

    # ---- tl.yaml ----
    TL:              bool  = False
    Nb_TL:           int   = 50
    plane_TL:        int   = 150
    pretrained_ckpt: str   = ""

    def __post_init__(self):
        if self.channels is None:
            self.channels = [1, 16, 32, 64, 128, 256]

    @property
    def ckpt_path(self) -> str:
        """Full path to output checkpoint file."""
        return os.path.join(self.checkpoint_dir, "best.pt")

    @property
    def ckpt_name(self) -> str:
        suffix = "_TL" if self.TL else ""
        return f"simon_unet3d_fixed_best{suffix}.pt"


_DEFAULT_ARCH  = os.path.join(os.path.dirname(__file__), "configs", "arch.yaml")
_DEFAULT_INPUT = os.path.join(os.path.dirname(__file__), "configs", "input.yaml")
_DEFAULT_TL    = os.path.join(os.path.dirname(__file__), "configs", "tl.yaml")


def load_cfg(
    arch:  str = _DEFAULT_ARCH,
    input: str = _DEFAULT_INPUT,
    tl:    str = _DEFAULT_TL,
) -> Cfg:
    merged: dict = {}
    for path in (input, arch, tl):
        if os.path.exists(path):
            merged.update(_load(path))
        else:
            print(f"[config] Warning: {path} not found, using defaults.")

    known    = {f for f in Cfg.__dataclass_fields__ if Cfg.__dataclass_fields__[f].init}
    filtered = {k: v for k, v in merged.items() if k in known}
    cfg = Cfg(**filtered)

    # validate required paths
    missing = [f for f in ("HOLO_DIR_TRAIN", "BIN_DIR_TRAIN", "HOLO_DIR_VAL",
                            "BIN_DIR_VAL", "cache_root", "checkpoint_dir")
               if not getattr(cfg, f)]
    if missing:
        raise ValueError(f"Missing required paths in yaml: {missing}")

    return cfg

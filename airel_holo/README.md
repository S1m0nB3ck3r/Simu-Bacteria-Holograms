# 3D U-Net for Bacteria Segmentation in Digital Holographic Volumes

Deep-learning pipeline for detecting and segmenting bacteria in 3D volumes
reconstructed from inline digital holograms. A 3D U-Net is trained on
Angular-Spectrum-Propagation (ASP) reconstructions to produce voxel-level
segmentation masks, from which individual bacteria are localised and counted.

The pipeline supports architecture ablations (normalisation, residual blocks,
loss functions, channel widths), a bacteria-density study (10/20/50 bacteria
per volume), and **curriculum learning** with dynamic patch scaling to extend
detection to deep volumes (up to 200 planes / 100 µm).

---

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Data Layout](#data-layout)
- [Configuration](#configuration)
- [Usage](#usage)
- [File Reference](#file-reference)
- [Method Summary](#method-summary)

---

## Overview

Each raw hologram (`512 × 512`) is numerically reconstructed into a 3D volume
(`512 × 512 × Z`) via ASP, then fed to a 3D U-Net that outputs a per-voxel
foreground probability. Connected-component analysis of the thresholded mask
yields individual bacterium detections, scored against ground truth with an
object-level F1 (greedy centroid matching within a fixed radius).

Key physical parameters (simulation): ×40 objective, effective pixel size
0.1375 µm, axial step 0.5 µm/plane, wavelength 660 nm, bacteria 2–4 µm long ×
0.5–0.7 µm thick, refractive-index contrast Δn = 0.005.

---

## Installation

```bash
python -m venv venv
source venv/bin/activate            # Linux/Mac
pip install torch numpy scipy scikit-image pyyaml tqdm matplotlib pandas openpyxl
```

Requires Python ≥ 3.10 and a CUDA-capable GPU (training is GPU-bound; a
`512×512×200` volume does not fit through the network in one pass and is
processed with a sliding window).

---

## Data Layout

Data is expected in a per-depth / per-density directory tree:

```
Z{planes}/{Nb}b/
├── data_split/
│   ├── train/
│   │   ├── simulated_hologram/    holo_0.npy, holo_1.npy, ...
│   │   └── binary_volume/         bin_volume_0.npy, ...
│   ├── val/
│   │   ├── simulated_hologram/
│   │   └── binary_volume/
│   └── test/
│       ├── simulated_hologram/
│       └── binary_volume/
├── cache_simon_fixed/             built automatically (vol_N.npy + seg_N.npy)
│   ├── train/  val/  test/
└── checkpoints/{run_name}/best.pt
```

`split_data.py` produces the 70/15/15 `data_split/` from a flat folder of
holograms (fixed seed, disjoint slicing — no leakage).

---

## Configuration

A run is defined by **two YAML files** (plus an optional transfer-learning
file). Splitting configuration this way lets one dataset (`input.yaml`) be
reused across many architectures (`architecture.yaml`).

### `architecture_*.yaml` — model + optimisation
```yaml
channels:     [1, 16, 32, 64, 128, 256]  # input + 4 encoder levels + bottleneck
dropout_prob: 0.3
norm:         batchnorm     # batchnorm | groupnorm | instancenorm
residual:     true          # residual blocks in encoder/decoder

lr:          1.0e-3
batch_size:  30
epochs:      80
patience:    10             # early stopping on object-level F1

loss:          bce_dice     # bce_dice | focal_dice | tversky
tversky_alpha: 0.3          # FP weight
tversky_beta:  0.7          # FN weight (β>α penalises missed bacteria)

run_name:       baseline_res
checkpoint_dir: "../Z10/50b/checkpoints/baseline_res"
```

### `input_*.yaml` — dataset + patches + physics
```yaml
holo_plane_number: 10       # Z (number of reconstructed planes)
bact_numb: 50               # Nb (bacteria per hologram)

HOLO_DIR_TRAIN: "../Z10/50b/data_split/train/simulated_hologram"
BIN_DIR_TRAIN:  "../Z10/50b/data_split/train/binary_volume"
HOLO_DIR_VAL:   "../Z10/50b/data_split/val/simulated_hologram"
BIN_DIR_VAL:    "../Z10/50b/data_split/val/binary_volume"
HOLO_DIR_TEST:  "../Z10/50b/data_split/test/simulated_hologram"
BIN_DIR_TEST:   "../Z10/50b/data_split/test/binary_volume"
cache_root:     "../Z10/50b/cache_simon_fixed"

patch_xy: 128
patch_z:  8                 # scale with Z for deep volumes

samples_per_epoch: 10000
pos_fraction:      0.7      # patch centred on a bacterium
hardneg_fraction:  0.2      # patch near-but-offset from a bacterium

medium_wavelength: 6.6e-7   # ASP physics (meters)
pix_size_cam:      1.375e-7
Z_step:            5.0e-7

aug_flips: true             # random X/Y/Z flips
```

### `TL.yaml` — transfer learning (optional)
```yaml
TL: false
pretrained_ckpt: ""         # e.g. "../Z100/10b/checkpoints/baseline/best.pt"
```

For curriculum learning, set `TL: true` and point `pretrained_ckpt` at the
previous stage's `best.pt`. The architecture must match the source checkpoint.

---

## Usage

### 1. Split raw data
```bash
python split_data.py
```

### 2. Train
```bash
python main.py \
    --arch  configs/architecture_base_res_z10_50b.yaml \
    --input configs/input_z10_p8_50b.yaml
```
On first run, volumes are reconstructed and cached under `cache_root/`.
Each epoch logs to a CSV inside `checkpoint_dir/`; the best-F1 model is saved
as `best.pt`.

### 3. Curriculum learning / transfer
```bash
python main.py \
    --arch  configs/architecture_tv_wide_z125.yaml \
    --input configs/input_z125.yaml \
    --tl    configs/TL.yaml            # with TL: true + pretrained_ckpt set
```

Sequential chains (e.g. weekend runs) — use `&&` so a failed stage stops the
chain:
```bash
nohup bash -c '
python main.py --arch a1.yaml --input i1.yaml &&
python main.py --arch a2.yaml --input i2.yaml
' > train.log 2>&1 &
```

### 4. Evaluate + detection figures
```bash
python plot_paper_figure.py \
    --arch  configs/architecture_base_res_z10_50b.yaml \
    --input configs/input_z10_p8_50b.yaml \
    --cache_split test \
    --out_dir figs \
    --vol_idx 133          # optional: also render this specific volume
```
Produces `fig_a_{run}_Z{Z}_Nb{Nb}.png` (detection overlay + crops),
`fig_b_...png` (per-volume TP/FP/FN), and appends a row of micro-averaged
metrics to `figs/results_summary.csv`.

Compare depths/densities in one scatter:
```bash
python plot_paper_figure.py --all \
    --configs "arch_z10.yaml,input_z10.yaml" \
              "arch_z100.yaml,input_z100.yaml" \
    --xaxis Z \
    --out_dir figs
```

---

## File Reference

| File | Role |
|------|------|
| `main.py`               | Entry point — parses YAMLs, builds model/data, runs the training loop, handles TL and early stopping. |
| `config.py`             | Loads and merges the architecture / input / TL YAMLs into a single `Cfg` dataclass. |
| `model.py`              | `UNet3D` with toggleable normalisation, residual blocks, anisotropic kernels, and dynamic pooling for variable `patch_z`. |
| `losses.py`             | BCE+Dice, Focal+Dice, and Tversky losses via a `build_loss()` factory; voxel-level metrics. |
| `dataset.py`            | Intelligent patch sampling (positive / hard-negative / random) and disk-cache management. |
| `reconstruction.py`     | Angular Spectrum Propagation — hologram → 3D volume. |
| `train.py`              | `train_one_epoch()` — one optimisation pass, returns metrics. |
| `validation.py`         | Full-volume sliding-window inference and object-level F1 (micro-averaged). |
| `plot_paper_figure.py`  | Test-set evaluation, detection-overlay figures, and CSV metric summary. |
| `split_data.py`         | 70/15/15 train/val/test split (seed 42, disjoint). |

---

## Method Summary

- **Reconstruction** — inline holograms are reconstructed to 3D via ASP;
  volumes are z-score normalised per volume and cached.
- **Sampling** — training patches are drawn 70 % centred on a bacterium,
  20 % near-but-offset (hard negatives), 10 % fully random, to counter the
  extreme foreground/background imbalance.
- **Model** — 3D U-Net, isotropic `3×3×3` kernels, batch-norm, optional
  residual blocks; best configuration uses wide channels
  `[1,24,48,96,192,384]` with a BCE+Tversky loss (α=0.3, β=0.7).
- **Evaluation** — thresholded mask → 3D connected components → centroid
  greedy matching (radius 10 vox) → micro-averaged object-level F1.
- **Curriculum learning** — to reach deep volumes (Z=200), the model is
  transferred through Z = 100 → 125 → 150 → 175 → 200; scaling `patch_z`
  with depth (dynamic patch) roughly doubles the final F1 versus a fixed
  patch.

---

## Notes

- The object count is the number of **connected components** in the mask,
  which may slightly exceed the number of simulated bacteria (a tilted
  bacterium spanning a plane boundary can split into two components). Ground
  truth and prediction pass through the identical pipeline, so the metric
  remains consistent.
- The reported F1 is **micro-averaged** over all evaluation volumes
  (aggregate TP/FP/FN, then compute F1), matching the training-time metric.
- Reduce GPU/CPU memory at inference by increasing `--stride_xy`/`--stride_z`
  (defaults to the patch size = no overlap = fastest).

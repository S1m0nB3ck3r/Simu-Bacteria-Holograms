# Z = 200 — segmentation and barycentres

Test-set results for the deepest reconstruction: **200 planes = 100 µm**.

This is the hard regime. Each bacterium's diffraction rings extend across tens
of planes, so at this depth they overlap heavily and the network struggles to
separate a genuine in-focus object from the out-of-focus signature of its
neighbours. Detection rates are correspondingly low — see
[Interpreting these results](#interpreting-these-results) before drawing
conclusions from the error figures.

Naming: `<type>_<architecture>[_<strategy>]_Z200_Nb<density>[_vol<N>].<ext>`

---

## Contents

Four architectures (`baseline`, `baseline_res`, `tversky_res`,
`tversky_wide`), three densities (`Nb10`, `Nb20`, `Nb50`), and — at the
reference density `Nb20` — three training strategies:

| Suffix | Strategy |
|---|---|
| *(none)* | trained from scratch directly at Z = 200 |
| `_tl` | curriculum learning, patch depth held fixed at `Pz = 64` |
| `_dyn_tl` | curriculum learning, patch depth scaled with Z (`Pz` 80 → 128) |

`Nb20` additionally contains a `tversky` run (BCE+Tversky loss on the narrow
architecture, no residual blocks).

Files ending `_vol55` show test volume 55 specifically, so the same hologram
can be compared across models; the others show each model's own best-scoring
volume.

---

## What is in each file type

### `barycenters_*.csv`

One row per object: every bacterium the model predicted, plus the ones it
missed.

| Column | Unit | Meaning |
|---|---|---|
| `vol_idx` | — | Index of the hologram (matches `holo_<N>.npy`) |
| `pred_id` | — | Numbering of the objects found in that volume |
| `x_px`, `y_px` | px | Lateral position |
| `z_plane` | plane | Axial position, in reconstruction planes (0–200) |
| `x_um`, `y_um`, `z_um` | µm | The same position in physical units — **the barycentre** |
| `status` | — | `TP` real bacterium detected · `FP` false detection · `FN` bacterium missed |
| `gt_id` | — | Which real bacterium it was matched to (empty for `FP`) |
| `err_xy_um` | µm | Lateral distance from the true position |
| `err_z_um` | µm | Axial distance from the true position |
| `err_um` | µm | Total 3D distance |
| `dx_um`, `dy_um`, `dz_um` | µm | The same offsets **with sign** (predicted − true) |

**Reading the file.** The coordinate columns mean different things depending on
`status`:

- `TP` and `FP` — the coordinates are what the **model predicted**.
- `FN` — the coordinates are those of the **real bacterium that was missed**;
  there is no prediction, so the error columns are empty.

Positions are converted from voxels using δ<sub>xy</sub> = 0.1375 µm laterally
and δ<sub>z</sub> = 0.5 µm axially.

### `fig_s2_projection_*.png`

Predicted mask (red) and true mask (green), outlined over the
maximum-intensity projection of the volume. Shows **one volume only**: the
best-scoring one, or volume 55 for the `_vol55` files.

### `fig_f_predvstrue_*.png`

Predicted against true coordinate for every matched bacterium, one panel per
axis, with the identity line and a least-squares fit (slope and offset
annotated). Aggregates all matched bacteria over the 30 test volumes.

---

## Reproducing

```bash
# from scratch
python3.10 plot_paper_figure.py \
    --arch  configs/Z200_20b/architecture_base_z200_zeroshot.yaml \
    --input configs/Z200_20b/input_Z200_zeroshot.yaml \
    --cache_split test --vol_idx 55 \
    --out_dir figs_Z200_zeroshot

# dynamic-patch curriculum
python3.10 plot_paper_figure.py \
    --arch  configs/Z200_20b/architecture_baseline_Z200_dynpatch.yaml \
    --input configs/Z200_20b/input_Z200_dynpatch.yaml \
    --cache_split test --vol_idx 55 \
    --out_dir figs_Z200_dynpatch
```

See the `Z_10` and `Z_100` folders for the shallower depths, where detection is
close to complete and the two error components are comparable.

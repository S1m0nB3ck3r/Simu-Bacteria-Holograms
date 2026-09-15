# Z = 10 — segmentation and barycentres

Test-set results for the shallowest reconstruction depth: **10 planes = 5 µm**.

Each file corresponds to one trained model evaluated on the 30 held-out test
volumes of one dataset. Three bacteria densities are covered
(`Nb10`, `Nb20`, `Nb50`) and four architectures
(`baseline`, `baseline_res`, `tversky_res`, `tversky_wide`), giving 12
combinations × 3 file types = 36 files.

Naming: `<type>_<architecture>_Z10_Nb<density>.<ext>`

---

## What is in each file type

### `barycenters_*.csv`

One row per object, listing the position of every bacterium the model
predicted, plus the ones it missed. This is the main quantitative output.

| Column | Unit | Meaning |
|---|---|---|
| `vol_idx` | — | Index of the hologram (matches `holo_<N>.npy`) |
| `pred_id` | — | Numbering of the objects found in that volume |
| `x_px`, `y_px` | px | Lateral position |
| `z_plane` | plane | Axial position, in reconstruction planes |
| `x_um`, `y_um`, `z_um` | µm | The same position in physical units — **the barycentre** |
| `status` | — | `TP` real bacterium detected · `FP` false detection · `FN` bacterium missed |
| `gt_id` | — | Which real bacterium it was matched to (empty for `FP`) |
| `err_xy_um` | µm | Lateral distance from the true position |
| `err_z_um` | µm | Axial distance from the true position |
| `err_um` | µm | Total 3D distance |
| `dx_um`, `dy_um`, `dz_um` | µm | The same offsets **with sign** (predicted − true) |

**Reading the file.** The meaning of the coordinate columns depends on
`status`:

- `TP` and `FP` — the coordinates are what the **model predicted**.
- `FN` — the coordinates are those of the **real bacterium that was missed**;
  there is no prediction, so the error columns are empty.

The unsigned `err_*` columns measure precision. The signed `d*_um` columns
additionally reveal a systematic shift: a mean `dz_um` far from zero means the
model places bacteria consistently too deep or too shallow, which is a
different problem from random scatter.

Positions are converted from voxels using δ<sub>xy</sub> = 0.1375 µm laterally
and δ<sub>z</sub> = 0.5 µm axially.

### `fig_s2_projection_*.png`

The predicted mask (red) and the true mask (green), outlined over the
maximum-intensity projection of the reconstructed volume, so every object in
the volume appears in a single image together with its diffraction pattern.
Outlines rather than fills, so the underlying reconstruction stays visible.

Shows **one volume** — the best-scoring one of the test set. Illustrative of
the segmentation the network produces, not representative of average
performance.

### `fig_f_predvstrue_*.png`

Predicted against true coordinate for every matched bacterium, one panel per
axis (x, y, z). Dashed line: perfect localisation. Red line: least-squares fit,
with slope and offset annotated.

- Points on the diagonal → exact localisation.
- Spread about it → random error.
- Slope ≠ 1 or offset ≠ 0 → a *systematic* effect, correctable by calibration.

Aggregates all matched bacteria over the 30 test volumes.


## Reproducing

```bash
python3.10 plot_paper_figure.py \
    --arch  configs/Z10_20b/architecture_base_z10.yaml \
    --input configs/Z10_20b/input_z10_p8.yaml \
    --cache_split test \
    --out_dir figs_Z10_20b
```

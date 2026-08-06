"""
plot_paper_figure.py
=====================
Publication figures for bacteria detection in holographic volumes.

DEFAULT MODE (single run):
    python3.10 plot_paper_figure.py \
        --arch    configs/architecture_baseline_Z100.yaml \
        --input   configs/input_test_Z100.yaml \
        --out_dir paper_fig_z100

    Saves:
        fig_a_{run}.png          -- hologram + TP/FP/FN circles + inset crops
        fig_b_{run}.png          -- TP/FP/FN bar chart per volume
        fig_c_{run}.png          -- predicted vs GT count scatter
        fig_d_loss_{run}.png     -- training/val loss curves
        fig_d_dice_{run}.png     -- training/val dice curves
        fig_d_objF1_{run}.png    -- object-level F1 vs epoch

ALL MODE (compare across Z depths):
    python3.10 plot_paper_figure.py \
        --all \
        --configs \
            "configs/arch_z10.yaml,configs/input_z10.yaml" \
            "configs/arch_z100.yaml,configs/input_z100.yaml" \
            "configs/arch_z200.yaml,configs/input_z200.yaml" \
        --labels "Z=10" "Z=100" "Z=200" \
        --out_dir paper_fig_all

    Saves:
        fig_c_all_depths.png     -- mean predicted vs GT count per Z depth
                                    with std error bars across 30 volumes
"""

import os, sys, argparse
import numpy as np
import torch
import yaml
import scipy.ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config     import Cfg
from model      import build_model
from validation import sliding_window_predict_full_volume

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#333333", "axes.labelcolor": "#111111",
    "axes.titlecolor": "#111111", "xtick.color": "#333333",
    "ytick.color": "#333333", "grid.color": "#DDDDDD",
    "grid.linestyle": "--", "grid.alpha": 0.6,
    "text.color": "#111111", "font.family": "sans-serif",
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "legend.fontsize": 8, "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

MATCH_RADIUS = 10.0
CMAP_DET = {"TP": "#2E7D32", "FP": "#E53935", "FN": "#1565C0"}


def load_cfg_inference(arch_path, input_path):
    merged = {}
    for path in (input_path, arch_path):
        if os.path.exists(path):
            with open(path) as f:
                merged.update(yaml.safe_load(f) or {})
    known = {f for f in Cfg.__dataclass_fields__ if Cfg.__dataclass_fields__[f].init}
    return Cfg(**{k: v for k, v in merged.items() if k in known})


def get_centers(mask_hwz):
    lbl, n = ndi.label(mask_hwz > 0)
    if n == 0:
        return np.zeros((0, 3)), 0
    c = ndi.center_of_mass(mask_hwz.astype(np.uint8), lbl, range(1, n+1))
    return np.array([[ci[1], ci[0], ci[2]] for ci in c], dtype=np.float32), n


def greedy_match(gt_xyz, pr_xyz, r=MATCH_RADIUS):
    if len(gt_xyz) == 0 and len(pr_xyz) == 0:
        return [], [], [], [], []
    if len(gt_xyz) == 0:
        return [], list(range(len(pr_xyz))), [], [], []
    if len(pr_xyz) == 0:
        return [], [], list(range(len(gt_xyz))), [], []
    used_gt = np.zeros(len(gt_xyz), bool)
    used_pr = np.zeros(len(pr_xyz), bool)
    pairs = []
    for i, p in enumerate(pr_xyz):
        d2 = np.sum((gt_xyz - p[None,:])**2, axis=1)
        j  = int(np.argmin(d2))
        if not used_gt[j] and d2[j] <= r**2:
            used_gt[j] = used_pr[i] = True
            pairs.append((j, i))
    fp = [i for i in range(len(pr_xyz)) if not used_pr[i]]
    fn = [j for j in range(len(gt_xyz)) if not used_gt[j]]
    return pairs, fp, fn, [p[0] for p in pairs], [p[1] for p in pairs]


def discover_ids(holo_dir):
    ids = []
    for f in sorted(os.listdir(holo_dir)):
        if f.startswith("holo_") and f.endswith(".npy"):
            try:
                ids.append(int(f.replace("holo_","").replace(".npy","")))
            except ValueError:
                pass
    return ids


def savefig(out_dir, name):
    path = os.path.join(out_dir, name)
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


@torch.no_grad()
def run_inference(cfg, arch_path, n_vols, stride_xy, stride_z, cache_split='val', vol_idx=None):
    """Run inference on val set, return list of per-volume result dicts."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = os.path.join(cfg.checkpoint_dir, "best.pt")
    run_name   = os.path.basename(cfg.checkpoint_dir)
    holo_dir   = cfg.HOLO_DIR_TEST
    cache_dir  = os.path.join(cfg.cache_root, cache_split)

    print(f"Run: {run_name}")
    print(f"Checkpoint: {checkpoint}  (F1={torch.load(checkpoint, map_location='cpu', weights_only=False).get('best_obj_F1','?')})")
    print(f"Holo dir:  {holo_dir}")
    print(f"Cache dir: {cache_dir}")

    model = build_model(cfg).to(device)
    ckpt  = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ids     = discover_ids(holo_dir)[:n_vols]
    results = []
    best_so_far = None   # best-F1 volume whose arrays we retain

    for idx in ids:
        vp = os.path.join(cache_dir, f"vol_{idx}.npy")
        sp = os.path.join(cache_dir, f"seg_{idx}.npy")
        if not (os.path.exists(vp) and os.path.exists(sp)):
            print(f"  [skip] cache missing idx={idx}")
            continue

        vol_hwz = np.load(vp).astype(np.float32)
        gt_hwz  = (np.load(sp) > 0).astype(np.uint8)
        H, W, Z = gt_hwz.shape

        sxy = stride_xy if stride_xy is not None else cfg.patch_xy
        sz  = stride_z  if stride_z  is not None else cfg.patch_z
        prob_hwz, pred_hwz = sliding_window_predict_full_volume(
            model, vol_hwz, cfg, device,
            stride_xy=sxy, stride_z=sz, thr=0.5)

        gt_xyz,   n_gt   = get_centers(gt_hwz)
        pred_xyz, n_pred = get_centers(pred_hwz)
        pairs, fp_i, fn_i, _, _ = greedy_match(gt_xyz, pred_xyz)

        tp = len(pairs); fp = len(fp_i); fn = len(fn_i)
        # per-volume F1 (for visualization/colorbar only)
        eps  = 1e-8
        prec = tp/(tp+fp+eps); rec = tp/(tp+fn+eps)
        f1   = 2*prec*rec/(prec+rec+eps)

        holo_path = os.path.join(holo_dir, f"holo_{idx}.npy")
        holo_np   = np.load(holo_path).astype(np.float32) \
                    if os.path.exists(holo_path) else vol_hwz.max(axis=2)

        rec_d = dict(
            idx=idx, n_gt=n_gt, n_pred=n_pred,
            tp=tp, fp=fp, fn=fn, f1=f1, prec=prec, rec=rec,
            gt_xyz=gt_xyz, pred_xyz=pred_xyz,
            fp_i=fp_i, fn_i=fn_i, pairs=pairs,
            vol_hwz=None, gt_hwz=None, pred_hwz=None,
            prob_hwz=None, holo_np=None,
        )

        # ── Memory guard: heavy arrays kept only for volumes we will plot ──
        # (the requested vol_idx, and the running best-F1 volume)
        keep_heavy = (vol_idx is not None and idx == vol_idx)
        if not keep_heavy:
            if best_so_far is None or f1 > best_so_far["f1"]:
                keep_heavy = True
                # release the previous best's arrays
                if best_so_far is not None and best_so_far["idx"] != vol_idx:
                    for k in ("vol_hwz","gt_hwz","pred_hwz","prob_hwz","holo_np"):
                        best_so_far[k] = None

        if keep_heavy:
            rec_d.update(vol_hwz=vol_hwz, gt_hwz=gt_hwz,
                         pred_hwz=pred_hwz, prob_hwz=prob_hwz,
                         holo_np=holo_np)
            if vol_idx is None or idx != vol_idx:
                best_so_far = rec_d
        else:
            del vol_hwz, gt_hwz, pred_hwz, prob_hwz, holo_np

        results.append(rec_d)
        print(f"  idx={idx:3d}  GT={n_gt}  Pred={n_pred}  "
              f"TP={tp} FP={fp} FN={fn}  F1={f1:.3f}")

    # ── Micro-averaged F1 (consistent with training) ─────────────────────
    total_tp = sum(x["tp"] for x in results)
    total_fp = sum(x["fp"] for x in results)
    total_fn = sum(x["fn"] for x in results)
    eps = 1e-8
    micro_prec = total_tp / (total_tp + total_fp + eps)
    micro_rec  = total_tp / (total_tp + total_fn + eps)
    micro_f1   = 2*micro_prec*micro_rec / (micro_prec + micro_rec + eps)
    print(f"\nMicro F1 = {micro_f1:.4f}  Prec = {micro_prec:.4f}  "
          f"Rec = {micro_rec:.4f}  (aggregated over {len(results)} volumes)")

    for r in results:
        r["micro_f1"]   = micro_f1
        r["micro_prec"] = micro_prec
        r["micro_rec"]  = micro_rec

    return results, run_name, micro_f1, micro_prec, micro_rec


def _render_fig_a(r, run_name, out_dir, fname):
    """Render one detection-overlay figure for a single volume result r."""
    H, W, Z = r["vol_hwz"].shape
    vol_zhw = r["vol_hwz"].transpose(2, 0, 1)

    def make_crop(xyz, i, rad=28):
        cx, cy, cz = xyz[i]
        cy, cx, cz = int(round(cy)), int(round(cx)), int(round(cz))
        y0,y1 = max(0,cy-rad), min(H,cy+rad)
        x0,x1 = max(0,cx-rad), min(W,cx+rad)
        return (vol_zhw[cz,y0:y1,x0:x1] if 0<=cz<Z else np.zeros((y1-y0,x1-x0)),
                (x0, y0, x1, y1))

    tp_crops = [(make_crop(r["gt_xyz"],   j), "TP") for j,_ in r["pairs"][:2]]
    fp_crops = [(make_crop(r["pred_xyz"], i), "FP") for i in r["fp_i"][:1]]
    fn_crops = [(make_crop(r["gt_xyz"],   j), "FN") for j in r["fn_i"][:1]]
    crops    = (tp_crops + fp_crops + fn_crops)[:4]
    n_crops  = len(crops)

    fig = plt.figure(figsize=(10, 6))
    ax_main = fig.add_axes([0.02, 0.08, 0.55, 0.82])
    ax_main.imshow(r["holo_np"], cmap="gray", interpolation="nearest")

    for j,_ in r["pairs"]:
        cx,cy = r["gt_xyz"][j,0], r["gt_xyz"][j,1]
        ax_main.add_patch(plt.Circle((cx,cy),18,color=CMAP_DET["TP"],
                                     fill=False,linewidth=1.5))
    for i in r["fp_i"]:
        cx,cy = r["pred_xyz"][i,0], r["pred_xyz"][i,1]
        ax_main.add_patch(plt.Circle((cx,cy),18,color=CMAP_DET["FP"],
                                     fill=False,linewidth=1.5))
    for j in r["fn_i"]:
        cx,cy = r["gt_xyz"][j,0], r["gt_xyz"][j,1]
        ax_main.add_patch(plt.Circle((cx,cy),18,color=CMAP_DET["FN"],
                                     fill=False,linewidth=1.5,linestyle="--"))

    ax_main.legend(
        handles=[mpatches.Patch(color=v,label=k) for k,v in CMAP_DET.items()],
        loc="lower left", fontsize=9, framealpha=0.85, edgecolor="#CCCCCC")
    ax_main.set_title(
        f"(a) {run_name}  |  Vol.{r['idx']}  F1={r['f1']:.3f}  "
        f"TP={r['tp']} FP={r['fp']} FN={r['fn']}",
        fontsize=10, fontweight="bold")
    ax_main.axis("off")

    crop_h = 0.82 / max(n_crops, 1)
    for ci, ((crop_img, bbox), label) in enumerate(crops):
        x0_f = 0.60
        y0_f = 0.08 + (n_crops - 1 - ci) * crop_h
        ax_c = fig.add_axes([x0_f, y0_f + 0.01, 0.37, crop_h - 0.02])
        ax_c.imshow(crop_img, cmap="gray", interpolation="nearest")
        ax_c.set_title(label, fontsize=9, color=CMAP_DET[label],
                       fontweight="bold", pad=2)
        for sp in ax_c.spines.values():
            sp.set_edgecolor(CMAP_DET[label]); sp.set_linewidth(2.5)
        ax_c.set_xticks([]); ax_c.set_yticks([])
        x0b, y0b, x1b, y1b = bbox
        ax_main.add_patch(plt.Rectangle((x0b, y0b), x1b-x0b, y1b-y0b,
                          linewidth=1.5, edgecolor=CMAP_DET[label],
                          facecolor="none", linestyle="-"))

    savefig(out_dir, fname)


def plot_single(cfg, results, run_name, out_dir,
               micro_f1=None, micro_prec=None, micro_rec=None, vol_idx=None):
    """Generate all figures for a single run."""

    # unique tag so Nb=10/20/50 and different Z don't overwrite each other
    tag = f"{run_name}_Z{cfg.holo_plane_number}_Nb{cfg.bact_numb}"

    mean_f1   = np.mean([x["f1"]   for x in results])
    mean_prec = np.mean([x["prec"] for x in results])
    mean_rec  = np.mean([x["rec"]  for x in results])
    n_gts     = [x["n_gt"]   for x in results]
    n_preds   = [x["n_pred"] for x in results]
    f1s       = [x["f1"]     for x in results]

    # ── Fig a (best volume) ──────────────────────────────────────────────
    # only volumes whose heavy arrays were retained can be rendered
    plottable = [x for x in results if x.get("vol_hwz") is not None]
    if plottable:
        r_best = max(plottable, key=lambda x: x["f1"])
        _render_fig_a(r_best, run_name, out_dir, f"fig_a_{tag}.png")
    else:
        print("  [warn] no volume arrays retained — skipping fig_a (best)")

    # ── Fig a (fixed volume) — same hologram across all models ───────────
    if vol_idx is not None:
        r_fixed = next((x for x in results
                        if x["idx"] == vol_idx and x.get("vol_hwz") is not None),
                       None)
        if r_fixed is not None:
            _render_fig_a(r_fixed, run_name, out_dir,
                          f"fig_a_{tag}_vol{vol_idx}.png")
        else:
            print(f"  [warn] vol_idx={vol_idx} not found in results — "
                  f"skipping fixed-volume fig_a")

    # ── Fig b: TP/FP/FN per volume ────────────────────────────────────────
    vol_ids = [x["idx"] for x in results]
    tps = [x["tp"] for x in results]
    fps = [x["fp"] for x in results]
    fns = [x["fn"] for x in results]

    fig, ax = plt.subplots(figsize=(max(8, len(results)*0.38), 5))
    x = np.arange(len(results))
    w = 0.28
    ax.bar(x-w, tps, w, label="TP", color="#2E7D32", alpha=0.9)
    ax.bar(x,   fps, w, label="FP", color="#E53935", alpha=0.9)
    ax.bar(x+w, fns, w, label="FN", color="#1565C0", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in vol_ids], rotation=90, fontsize=7.5)
    ax.set_xlabel("Volume index")
    ax.set_ylabel("Count")
    ax.set_title(f"(b) TP / FP / FN per volume - {run_name}\n"
                 f"{len(results)} volumes  |  "
                 f"total TP={sum(tps)} FP={sum(fps)} FN={sum(fns)}",
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    savefig(out_dir, f"fig_b_{tag}.png")

    # ── Save CSV summary row (append) ─────────────────────────────────────
    import csv
    csv_path = os.path.join(out_dir, "results_summary.csv")
    row = {
        "run_name":   run_name,
        "Z":          cfg.holo_plane_number,
        "Nb":         cfg.bact_numb,
        "patch_xy":   cfg.patch_xy,
        "patch_z":    cfg.patch_z,
        "loss":       cfg.loss,
        "residual":   cfg.residual,
        "channels":   str(cfg.channels),
        "n_vols":     len(results),
        "micro_F1":   round(micro_f1,   4) if micro_f1   is not None else None,
        "micro_prec": round(micro_prec, 4) if micro_prec is not None else None,
        "micro_rec":  round(micro_rec,  4) if micro_rec  is not None else None,
        "total_TP":   sum(x["tp"] for x in results),
        "total_FP":   sum(x["fp"] for x in results),
        "total_FN":   sum(x["fn"] for x in results),
    }
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"  Appended row to: {csv_path}")

    print(f"\nMicro F1={micro_f1:.4f}  Prec={micro_prec:.4f}  Rec={micro_rec:.4f}"
          if micro_f1 is not None else
          f"\nMean F1={mean_f1:.4f}")


def plot_all_depths(configs_labels, out_dir, n_vols, stride_xy, stride_z, xaxis='Z', cache_split='val'):
    """Fig c across multiple Z depths: mean pred vs mean GT with std bars."""
    points = []

    for (arch_path, input_path), label in configs_labels:
        cfg     = load_cfg_inference(arch_path, input_path)
        results, run_name, micro_f1, micro_prec, micro_rec = run_inference(
            cfg, arch_path, n_vols, stride_xy, stride_z, cache_split=cache_split)
        if not results:
            print(f"  [skip] no results for {label}")
            continue
        n_gts   = np.array([x["n_gt"]   for x in results], dtype=float)
        n_preds = np.array([x["n_pred"] for x in results], dtype=float)
        f1s     = np.array([x["f1"]     for x in results])
        n_gts_arr  = np.array([x["n_gt"]   for x in results], dtype=float)
        n_preds_arr= np.array([x["n_pred"] for x in results], dtype=float)
        f1s_arr    = np.array([x["f1"]     for x in results], dtype=float)
        points.append(dict(
            label=label,
            Z=cfg.holo_plane_number,
            mean_gt=n_gts_arr.mean(),    std_gt=n_gts_arr.std(),
            mean_pred=n_preds_arr.mean(),std_pred=n_preds_arr.std(),
            mean_nb=n_gts_arr.mean(),    std_nb=n_gts_arr.std(),
            mean_f1=f1s_arr.mean(),      std_f1=f1s_arr.std(),
            micro_f1=micro_f1,           micro_prec=micro_prec,
            micro_rec=micro_rec,
            n=len(results),
        ))
        print(f"  {label}: micro F1={micro_f1:.4f}  "
              f"Prec={micro_prec:.4f}  Rec={micro_rec:.4f}")
        print(f"  {label}: mean F1={f1s.mean():.3f}±{f1s.std():.3f}  "
              f"GT={n_gts.mean():.1f}  Pred={n_preds.mean():.1f}±{n_preds.std():.1f}")

    if not points:
        print("No results.")
        return

    fig, ax = plt.subplots(figsize=(7, 6))

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(points)))

    f1_vals = [p["mean_f1"] for p in points]
    vmin_f1 = max(0.0, min(f1_vals) - 0.05)
    norm    = plt.Normalize(vmin=vmin_f1, vmax=1.0)
    cmap    = plt.cm.RdYlGn

    all_vals = [p["mean_gt"] for p in points] + [p["mean_pred"] for p in points]
    lim = max(all_vals) + 3

    for p in points:
        c = cmap(norm(p["mean_f1"]))
        if xaxis == "Z":
            x_val = p["Z"]
            x_err = None
        else:
            x_val = p["mean_nb"]
            x_err = p["std_nb"]
        ax.errorbar(x_val, p["micro_f1"],
                    xerr=x_err, yerr=None,
                    fmt="o", color=c, markersize=10,
                    capsize=4, capthick=1.5, linewidth=1.5,
                    label=f"{p['label']}  F1={p['micro_f1']:.3f}")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="mean obj_F1", pad=0.02)

    if xaxis == "Z":
        ax.set_xlabel("Z depth (planes)")
        ax.set_ylabel("mean obj_F1 (\u00b1std over volumes)")
        ax.set_title("(c) F1 vs Z depth\neach point = mean\u00b1std over volumes",
                     fontweight="bold")
        x_vals = [p["Z"] for p in points]
        ax.set_xlim(min(x_vals)-5, max(x_vals)+5)
        ax.set_ylim(0, 1.05)
    else:
        ax.set_xlabel("Mean bacteria count Nb (\u00b1std)")
        ax.set_ylabel("mean obj_F1 (\u00b1std over volumes)")
        ax.set_title("(c) F1 vs bacteria count Nb\neach point = mean\u00b1std over volumes",
                     fontweight="bold")
        x_vals = [p["mean_nb"] for p in points]
        ax.set_xlim(max(0, min(x_vals)-3), max(x_vals)+3)
        ax.set_ylim(0, 1.05)

    ax.legend(fontsize=8.5, loc="best")
    ax.grid(True); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    savefig(out_dir, f"fig_c_all_{xaxis}_{cfg.bact_numb}.png")


@torch.no_grad()
def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    if args.all:
        # parse --configs "arch.yaml,input.yaml" pairs
        configs_labels = []
        for i, pair in enumerate(args.configs):
            parts = pair.split(",")
            if len(parts) != 2:
                raise ValueError(f"--configs expects 'arch.yaml,input.yaml' pairs, got: {pair}")
            arch_path, input_path = parts[0].strip(), parts[1].strip()
            if args.labels and i < len(args.labels):
                label = args.labels[i]
            else:
                cfg_tmp = load_cfg_inference(arch_path, input_path)
                run     = os.path.basename(cfg_tmp.checkpoint_dir)
                z       = cfg_tmp.holo_plane_number
                pxy     = cfg_tmp.patch_xy
                pz      = cfg_tmp.patch_z
                label   = f"Z{z}_{run}_{pxy}x{pz}"
            configs_labels.append(((arch_path, input_path), label))
        plot_all_depths(configs_labels, args.out_dir,
                        args.n_vols, args.stride_xy, args.stride_z,
                        xaxis=args.xaxis, cache_split=args.cache_split)

    else:
        cfg = load_cfg_inference(args.arch, args.input)
        results, run_name, micro_f1, micro_prec, micro_rec = run_inference(
        cfg, args.arch, args.n_vols, args.stride_xy, args.stride_z,
        cache_split=args.cache_split, vol_idx=args.vol_idx)
        if not results:
            print("No results — check HOLO_DIR_TEST and cache_root in input yaml.")
            return
        plot_single(cfg, results, run_name, args.out_dir,
                    micro_f1, micro_prec, micro_rec, vol_idx=args.vol_idx)


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    # single mode
    p.add_argument("--arch",  default=None, help="arch yaml (single mode)")
    p.add_argument("--input", default=None, help="input yaml (single mode)")

    # all mode
    p.add_argument("--all",     action="store_true",
                   help="compare across multiple Z depths")
    p.add_argument("--configs", nargs="+", default=None,
                   help="'arch.yaml,input.yaml' pairs (--all mode)")
    p.add_argument("--labels",  nargs="+", default=None,
                   help="display labels for each config (--all mode)")

    # shared
    p.add_argument("--out_dir",   default="paper_fig")
    p.add_argument("--n_vols",    type=int, default=30)
    p.add_argument("--stride_xy", type=int, default=None,
                   help="XY stride (default: patch_xy from yaml = no overlap)")
    p.add_argument("--stride_z",  type=int, default=None,
                   help="Z stride (default: patch_z from yaml = no overlap)")
    p.add_argument("--xaxis", choices=["Z","Nb"], default="Z",
                   help="X axis for --all mode: Z depth or bacteria count Nb")
    p.add_argument("--cache_split", choices=["val","test"], default="val",
                   help="which cache split to evaluate on (default: val)")
    p.add_argument("--vol_idx", type=int, default=None,
                   help="also save fig_a for this specific volume index "
                        "(same hologram across models for comparison)")
    args = p.parse_args()

    if args.all and not args.configs:
        p.error("--all requires --configs")
    if not args.all and not (args.arch and args.input):
        p.error("single mode requires --arch and --input")

    main(args)
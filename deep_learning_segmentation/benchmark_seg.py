# -*- coding: utf-8 -*-

"""
Filename: benchmark_seg.py

Description:
Benchmark de la chaîne complète « positions 3D + volume segmenté par instance » : étage 1 (U-Net 3D sur le
volume grossier, carte des centres ; models/stage1 par défaut, 0–208 µm) puis étage 2
(seg_stage2.segment_instances, deux tours : fenêtres sur les détections, puis sur les centres vus par
l'étage 2 et non encore segmentés ; models/stage2), sur les jeux de gen_seg_sim_data.sh (test : N = 50 à
400, 30 % des bactéries à moins de 4 µm d'une autre ; paires rapprochées). Vérité : masques d'instance du
simulateur (fraction ≥ 0,5) et positions.
  - détection : rappel, précision, F1 par position (appariement hongrois 3 µm / 5 µm), pour l'étage 1
    seul et pour les instances finales (barycentres des masques) ;
  - segmentation : appariement hongrois instances prédites / vraies maximisant l'IoU (masques entiers) ;
    PQ = SQ × RQ (vrai positif si IoU > 0,5, Kirillov et al. 2019) ; par orientation φ, par N, par
    distance à la plus proche voisine ; part des instances nées au second tour ; borne haute : étage 2
    lancé sur les vraies positions ;
  - forme, mêmes estimateurs sur les masques prédit et vrai (seg_stage2.mask_measures) : erreur d'axe,
    de longueur, d'épaisseur ; erreur de position du barycentre ;
  - paires : paire résolue si les deux bactéries ont chacune une instance d'IoU > 0,5 ;
  - temps par hologramme : reconstruction, réseau 1, pics, étage 2 (deux tours, mesures comprises).
Sorties : resultats/benchmark_segmentation/ (metrics.json, per_bacteria.csv, strata.csv, pairs*.csv, timings.csv,
figures ; examples/ : fenêtres autour des bactéries les plus collées, contours de toutes les instances
prédites et vraies ; instances/ : volume segmenté de quelques hologrammes (origine + masque de chaque
instance, mesures CSV) et projection des instances sur l'hologramme, z en µm).

Utilisation :
  python deep_learning_segmentation/benchmark_seg.py

Author: Simon BECKER
Date: 2026-10-18

License:
GNU General Public License v3.0
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

DL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DL_DIR)
import benchmark as bm
import detection as det
import seg_stage2 as s2
import train as tr
import train_stage2 as t2
from unet3d import UNet3D

COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#7a5bd6']


def load_stage2(run_dir):
    ck = torch.load(os.path.join(run_dir, 'model_best.pt'), map_location='cuda', weights_only=False)
    cfg = ck['cfg']
    m = UNet3D(in_ch=2, out_ch=3, base=cfg['base'], depth=cfg['depth'], norm='batch').cuda().eval()
    m.load_state_dict(ck['state'])
    return tr.inference_model(m), cfg


def sync():
    torch.cuda.synchronize()


def nn_dist(b):
    if len(b) < 2:
        return np.full(len(b), np.inf)
    d, _ = cKDTree(b[:, :3]).query(b[:, :3], k=2)
    return d[:, 1]


def match_instances(insts, gtm, gt_centers, radius_um=3.0):
    """Appariement hongrois maximisant l'IoU ; retourne {indice vrai : (indice prédit, IoU)}."""
    if not insts or not gtm:
        return {}
    I = np.zeros((len(insts), len(gtm)))
    tree = cKDTree(gt_centers[:, :3])
    for i, d in enumerate(insts):
        for j in tree.query_ball_point(d['center'], radius_um):
            o, m = gtm[j]
            I[i, j] = s2.iou_sparse(d['origin'], d['mask'], o, m)
    r, c = linear_sum_assignment(-I)
    return {int(j): (int(i), float(I[i, j])) for i, j in zip(r, c) if I[i, j] > 0}


def evaluate_item(it, pos1, insts):
    g = it['meta']['mask_grid']
    b = it['bacts']
    gtm = [(o, m >= 0.5) for o, m in s2.load_instances(it)]
    centers = np.array([d['center'] for d in insts]).reshape(-1, 3)
    r1, c1 = det.match(pos1, b[:, :3])
    r2, c2 = det.match(centers, b[:, :3])
    out = dict(n_gt=len(b), n_det1=len(pos1), n_inst=len(insts), tp1=len(r1), tp2=len(r2),
               n_round2=int(sum(d['round'] > 1 for d in insts)))
    e1 = pos1[r1] - b[c1, :3] if len(r1) else np.zeros((0, 3))
    m = match_instances(insts, gtm, b)
    nnd = nn_dist(b)
    rows = []
    for j in range(len(b)):
        row = dict(n=len(b), z=float(b[j, 2]), phi=float(b[j, 4]), nn=float(nnd[j]),
                   L_param=float(b[j, 5]) if b.shape[1] > 5 else np.nan,
                   T_param=float(b[j, 6]) if b.shape[1] > 6 else np.nan, iou=0.0, round=0)
        gm = s2.mask_measures(gtm[j][1], gtm[j][0], g['dx_um'], g['dz_um'])
        if gm:
            row.update(L_gt=gm['length'], T_gt=gm['thickness'], V_gt=gm['volume'])
        if j in m:
            i, iou = m[j]
            d = insts[i]
            row.update(iou=iou, round=d['round'])
            pm = s2.mask_measures(d['mask'], d['origin'], g['dx_um'], g['dz_um'])
            if pm and gm:
                e = pm['center'] - b[j, :3]
                row.update(err_xy=float(np.hypot(e[0], e[1])), err_z=float(e[2]),
                           ang=s2.axis_angle_deg(pm['axis'], gm['axis']), L_pred=pm['length'],
                           T_pred=pm['thickness'], V_pred=pm['volume'])
        rows.append(row)
    return rows, out, e1


def run(items, m1, cfg1, thr1, m2, rec, refocus, infer, timings=None, oracle=False, collect=None, rounds=2):
    all_rows, dets, errs1 = [], [], []
    for k, it in enumerate(items):
        sync(); t0 = time.perf_counter()
        V = rec(it['h'].astype(np.float32) + 1.0)
        sync(); t1 = time.perf_counter()
        if oracle:
            pos1 = it['bacts'][:, :3].astype(np.float64)
            t2_ = t3 = t1
        else:
            logits = det.infer_volume(m1, tr.to_input(V, cfg1))
            sync(); t2_ = time.perf_counter()
            pos1, _ = tr.detect(logits, cfg1['target'], thr1, rec)
            del logits
            sync(); t3 = time.perf_counter()
        insts = s2.segment_instances(m2, V, pos1, refocus, cfg=infer, rounds=rounds)
        sync(); t4 = time.perf_counter()
        rows, out, e1 = evaluate_item(it, pos1, insts)
        for r in rows:
            r['sample'] = os.path.basename(it['file'])
            r.update({kk: it['meta'].get(kk) for kk in ('pair_type', 'pair_sep_um')})
        all_rows += rows
        errs1.append(e1)
        out['sample'] = os.path.basename(it['file'])
        dets.append(out)
        if timings is not None and k > 0:
            timings.append(dict(n=len(it['bacts']), n_det=len(pos1), n_inst=len(insts), reconstruction=t1 - t0,
                                network1=t2_ - t1, peaks=t3 - t2_, stage2=t4 - t3, total=t4 - t0))
        if collect is not None and k in collect:
            b = it['bacts']
            hard = np.argsort(nn_dist(b))[:6]                        # bactéries les plus collées
            X, org = refocus(V, b[hard, :3])
            collect[k] = dict(item=it, insts=insts, hard=hard, X=X.cpu().numpy(), org=org)
        del V
    return pd.DataFrame(all_rows), pd.DataFrame(dets), np.concatenate(errs1) if errs1 else np.zeros((0, 3))


def pq(df, n_inst, thr=0.5):
    iou = df.iou.to_numpy()
    tp = int((iou > thr).sum())
    fn = len(iou) - tp
    fp = max(n_inst - tp, 0)
    sq = float(iou[iou > thr].mean()) if tp else 0.0
    rq = tp / max(tp + 0.5 * fp + 0.5 * fn, 1e-9)
    return dict(PQ=sq * rq, SQ=sq, RQ=rq, tp=tp, fp=fp, fn=fn)


def summarize(df, dets, e1):
    s = {}
    for k, n_det in (('1', 'n_det1'), ('2', 'n_inst')):
        r = dets['tp' + k].sum() / dets.n_gt.sum()
        p = dets['tp' + k].sum() / max(dets[n_det].sum(), 1)
        s.update({f'recall{k}': r, f'precision{k}': p, f'f1_{k}': 2 * r * p / (r + p) if r + p else 0.0})
    s.update(pq(df, int(dets.n_inst.sum())))
    s.update(instances_round2=int(dets.n_round2.sum()), instances=int(dets.n_inst.sum()),
             iou_mean=float(df.iou.mean()), iou_mean_matched=float(df[df.iou > 0].iou.mean()),
             seg_recall_iou50=float((df.iou > 0.5).mean()))
    if len(e1):
        s.update(sigma1_xy=float(np.sqrt(np.mean(e1[:, 0] ** 2 + e1[:, 1] ** 2))), sigma1_z=float(e1[:, 2].std()))
    d = df.dropna(subset=['ang']) if 'ang' in df else df.iloc[:0]
    if len(d):
        s.update(angle_median_deg=float(d.ang.median()), angle_p90_deg=float(d.ang.quantile(0.9)),
                 length_err_median=float((d.L_pred - d.L_gt).abs().median()), length_bias=float((d.L_pred - d.L_gt).mean()),
                 thickness_err_median=float((d.T_pred - d.T_gt).abs().median()),
                 thickness_bias=float((d.T_pred - d.T_gt).mean()),
                 sigma2_xy=float(np.sqrt(np.mean(d.err_xy ** 2))), sigma2_z=float(d.err_z.std()), bias2_z=float(d.err_z.mean()))
        if d.L_param.notna().all():
            s.update(corr_length_pred_param=float(np.corrcoef(d.L_pred, d.L_param)[0, 1]),
                     corr_length_gtmask_param=float(np.corrcoef(d.L_gt, d.L_param)[0, 1]),
                     corr_thickness_pred_param=float(np.corrcoef(d.T_pred, d.T_param)[0, 1]),
                     corr_thickness_gtmask_param=float(np.corrcoef(d.T_gt, d.T_param)[0, 1]))
    return s


def strata(df):
    out = []
    bins = dict(nn=([0, 1, 2, 4, 8, np.inf], ['< 1 µm', '1–2 µm', '2–4 µm', '4–8 µm', '> 8 µm']),
                phi=([0, 30, 60, 90.1], ['φ < 30°', '30–60°', 'φ > 60°']))
    for col, (edges, labels) in bins.items():
        cat = pd.cut(df[col], edges, labels=labels, right=False)
        for lab in labels:
            d = df[cat == lab]
            if len(d):
                out.append(dict(by=col, bin=lab, n=len(d), seg_recall_iou50=float((d.iou > 0.5).mean()),
                                iou_mean=float(d.iou.mean()), from_round2=float((d['round'] > 1).mean()),
                                angle_median=float(d.ang.median()) if 'ang' in d and d.ang.notna().any() else np.nan))
    for n in sorted(df.n.unique()):
        d = df[df.n == n]
        out.append(dict(by='N', bin=str(n), n=len(d), seg_recall_iou50=float((d.iou > 0.5).mean()),
                        iou_mean=float(d.iou.mean()), from_round2=float((d['round'] > 1).mean()),
                        angle_median=float(d.ang.median()) if 'ang' in d and d.ang.notna().any() else np.nan))
    return pd.DataFrame(out)


def _win(mask, mo, wo, shape):
    """Masque (origine mo) recadré dans une fenêtre (origine wo, forme shape)."""
    out = np.zeros(shape, bool)
    lo = [max(a, b) for a, b in zip(mo, wo)]
    hi = [min(a + s, b + t) for a, s, b, t in zip(mo, mask.shape, wo, shape)]
    if all(h > l for l, h in zip(lo, hi)):
        out[tuple(slice(l - o, h - o) for l, h, o in zip(lo, hi, wo))] = \
            mask[tuple(slice(l - o, h - o) for l, h, o in zip(lo, hi, mo))]
    return out


def plot_examples(col, out_dir):
    """Fenêtres autour des bactéries les plus collées : |E| ; contours de toutes les instances vraies (vert)
    et prédites (une couleur par instance)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d = os.path.join(out_dir, 'examples')
    os.makedirs(d, exist_ok=True)
    pal = ['#eda100', '#e87ba4', '#2a78d6', '#eb6834', '#7a5bd6', '#ffffff']
    for k, c in col.items():
        if c is None:
            continue
        it, insts, X, org, hard = c['item'], c['insts'], c['X'], c['org'], c['hard']
        g = it['meta']['mask_grid']
        gtm = [(o, m >= 0.5) for o, m in s2.load_instances(it)]
        b = it['bacts']
        nnd = nn_dist(b)
        shape = X.shape[2:]
        fig, ax = plt.subplots(len(hard), 3, figsize=(10, 3.3 * len(hard)))
        ax = np.atleast_2d(ax)
        for row, j in enumerate(hard):
            A = np.hypot(X[row, 0], X[row, 1])
            wo = org[row]
            gw = [w for w in (_win(m, o, wo, shape) for o, m in gtm) if w.any()]
            pw = [w for w in (_win(dd['mask'], dd['origin'], wo, shape) for dd in insts) if w.any()]
            ref = gw[0] if gw else np.zeros(shape, bool)
            kc = int(np.argmax(ref.sum(axis=(1, 2)))) if ref.any() else shape[0] // 2
            yc = int(np.argmax(ref.sum(axis=(0, 2)))) if ref.any() else shape[1] // 2
            views = ((lambda v: v.max(axis=0), A.max(axis=0), 'projection XY', 1),
                     (lambda v: v[kc], A[kc], f'plan z = {(wo[0] + kc) * g["dz_um"]:.2f} µm', 1),
                     (lambda v: v[:, yc, :], A[:, yc, :], 'coupe XZ (z vers le bas)', g['dz_um'] / g['dx_um']))
            for a_, (f, img, title, asp) in zip(ax[row], views):
                a_.imshow(img, cmap='gray', aspect=asp)
                for w in gw:
                    if f(w).any():
                        a_.contour(f(w).astype(float), [0.5], colors='#1baf7a', linewidths=1.6)
                for q, w in enumerate(pw):
                    if f(w).any():
                        a_.contour(f(w).astype(float), [0.5], colors=pal[q % len(pal)], linewidths=1.0, linestyles='--')
                a_.set_title(title, fontsize=8)
                a_.set_xticks([])
                a_.set_yticks([])
            ax[row, 0].set_ylabel(f"z {b[j, 2]:.1f} µm\nvoisine à {nnd[j]:.1f} µm\n{len(gw)} vraies, {len(pw)} prédites",
                                  fontsize=8)
        fig.suptitle(f"{os.path.basename(it['file'])} : vert = instances du simulateur, pointillés = instances prédites",
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(d, f"{k:03d}_{os.path.basename(it['file'])[:-4]}.png"), dpi=110)
        plt.close(fig)


def save_instances(col, out_dir):
    """Volume segmenté (instances : origine + masque) + mesures + projection des instances sur l'hologramme."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    d = os.path.join(out_dir, 'instances')
    os.makedirs(d, exist_ok=True)
    for k, c in col.items():
        if c is None:
            continue
        it, insts = c['item'], c['insts']
        g = it['meta']['mask_grid']
        meas = []
        for n_, dd in enumerate(insts):
            m = s2.mask_measures(dd['mask'], dd['origin'], g['dx_um'], g['dz_um'])
            if m:
                meas.append(dict(instance=n_ + 1, x_um=m['center'][0], y_um=m['center'][1], z_um=m['center'][2],
                                 axis_x=m['axis'][0], axis_y=m['axis'][1], axis_z=m['axis'][2], length_um=m['length'],
                                 thickness_um=m['thickness'], volume_um3=m['volume'], n_voxels=m['n'],
                                 score=dd['score'], round=dd['round']))
        name = os.path.basename(it['file'])[:-4]
        np.savez_compressed(os.path.join(d, f"{name}_instances.npz"),
                            origin=np.array([dd['origin'] for dd in insts]).reshape(-1, 3),
                            masks=np.array([dd['mask'] for dd in insts]),
                            grid=json.dumps(dict(dx_um=g['dx_um'], dz_um=g['dz_um'],
                                                 note="voxel (k, r, c) de l'instance i <-> (z, y, x) = "
                                                      "((origin_k + k)·dz, (origin_r + r)·dx, (origin_c + c)·dx)")))
        pd.DataFrame(meas).to_csv(os.path.join(d, f"{name}_instances.csv"), index=False)
        h = it['h'].astype(np.float32)
        lo, hi = np.percentile(h, [0.5, 99.5])
        lab = np.zeros(h.shape, np.int32)
        for n_, dd in enumerate(insts):
            r, cc = np.nonzero(dd['mask'].max(axis=0))
            rr, c2 = r + dd['origin'][1], cc + dd['origin'][2]
            ok = (rr >= 0) & (rr < h.shape[0]) & (c2 >= 0) & (c2 < h.shape[1])
            lab[rr[ok], c2[ok]] = n_ + 1
        fig, axs = plt.subplots(1, 2, figsize=(16, 8.4))
        field = h.shape[0] * g['dx_um']
        rng = np.random.default_rng(0)
        cmap = plt.matplotlib.colors.ListedColormap(rng.uniform(0.25, 1, (max(len(insts), 1) + 1, 3)))
        for a_, zoom in ((axs[0], False), (axs[1], True)):
            a_.imshow(h, cmap='gray', vmin=lo, vmax=hi, extent=[0, field, field, 0])
            a_.imshow(np.ma.masked_equal(lab, 0), cmap=cmap, alpha=0.8, extent=[0, field, field, 0], interpolation='nearest')
            if zoom and meas:
                xy = np.array([[m['x_um'], m['y_um']] for m in meas])
                dens = (np.abs(xy[:, None] - xy[None]).max(axis=2) < 10).sum(axis=1)     # zone la plus peuplée
                cx, cy = np.clip(xy[np.argmax(dens)], 15, field - 15)
                a_.set_xlim(cx - 15, cx + 15)
                a_.set_ylim(cy + 15, cy - 15)
                for m in meas:
                    if abs(m['x_um'] - cx) < 15 and abs(m['y_um'] - cy) < 15:
                        a_.text(m['x_um'] + 0.5, m['y_um'] - 0.5, f"{m['z_um']:.1f}", color='#eda100', fontsize=7,
                                path_effects=[pe.withStroke(linewidth=1.2, foreground='black')])
                a_.set_title('zoom 30 × 30 µm, z en µm', fontsize=10)
            else:
                for m in meas:
                    a_.text(m['x_um'] + 1, m['y_um'] - 1, f"{m['z_um']:.0f}", color='#eda100', fontsize=3.5,
                            path_effects=[pe.withStroke(linewidth=0.8, foreground='black')])
                a_.set_title(f"{name} : {len(meas)} instances (projection XY), N vrai = {len(it['bacts'])}", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(d, f"{name}_instances.png"), dpi=150)
        plt.close(fig)


def plot_summary(df, st, pairs_sum, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    for a_, by, title in ((ax[0], 'nn', 'selon la distance à la plus proche voisine'), (ax[1], 'phi', "selon l'orientation"),
                          (ax[2], 'N', 'selon le nombre de bactéries')):
        s = st[st.by == by]
        x = np.arange(len(s))
        a_.bar(x - 0.2, s.seg_recall_iou50, 0.4, color=COLORS[0], label='bactéries segmentées (IoU > 0,5)')
        a_.bar(x + 0.2, s.from_round2, 0.4, color=COLORS[3], label='… dont trouvées au second tour')
        a_.plot(x, s.iou_mean, 'o-', color=COLORS[1], lw=2, label='IoU moyenne')
        a_.set_xticks(x)
        a_.set_xticklabels(s['bin'])
        a_.set_ylim(0, 1.02)
        a_.set_title(title)
        a_.grid(alpha=0.25, axis='y')
        a_.spines[['top', 'right']].set_visible(False)
    ax[0].legend(frameon=False, fontsize=8, loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'segmentation_strates.png'), dpi=110)
    plt.close(fig)
    d = df.dropna(subset=['L_pred', 'L_gt']) if 'L_pred' in df else df.iloc[:0]
    if len(d):
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
        for a_, x, y, lab in ((ax[0], d.L_gt, d.L_pred, 'longueur (µm)'), (ax[1], d.T_gt, d.T_pred, 'épaisseur (µm)')):
            a_.plot(x, y, '.', ms=2, alpha=0.4, color=COLORS[0])
            lo_, hi_ = np.percentile(np.concatenate([x, y]), [1, 99])
            a_.plot([lo_, hi_], [lo_, hi_], color='#888888', lw=1)
            a_.set_xlabel(f'{lab}, masque du simulateur')
            a_.set_ylabel(f'{lab}, masque prédit')
            a_.spines[['top', 'right']].set_visible(False)
        ax[2].hist(d.ang.dropna(), bins=np.arange(0, 91, 3), color=COLORS[2])
        ax[2].set_xlabel("erreur d'axe (degrés)")
        ax[2].spines[['top', 'right']].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'segmentation_forme.png'), dpi=110)
        plt.close(fig)
    if pairs_sum is not None and len(pairs_sum):
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
        for a_, kind, title in ((ax[0], 'lateral', 'Écart latéral'), (ax[1], 'axial', 'Écart axial')):
            s = pairs_sum[pairs_sum.pair_type == kind]
            a_.plot(s.sep, s.segmented, '-o', lw=2, ms=6, color=COLORS[0], label='paire résolue (2 instances IoU > 0,5)')
            a_.plot(s.sep, s.iou_min, '-o', lw=2, ms=6, color=COLORS[1], label='IoU de la moins bonne des deux')
            a_.set_title(title)
            a_.set_xlabel('écart entre les centres (µm)')
            a_.set_ylim(-0.02, 1.02)
            a_.grid(alpha=0.25)
            a_.spines[['top', 'right']].set_visible(False)
        ax[0].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'segmentation_paires.png'), dpi=110)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage1', default=os.path.join(DL_DIR, 'models', 'stage1'))
    ap.add_argument('--stage2', default=os.path.join(DL_DIR, 'models', 'stage2'))
    ap.add_argument('--data', default=os.path.join(DL_DIR, 'data', 'seg_sim'))
    ap.add_argument('--out', default=os.path.join(DL_DIR, 'resultats', 'benchmark_segmentation'))
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--rounds', type=int, default=2)
    ap.add_argument('--n_examples', type=int, default=4)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    m1, cfg1, thr1 = bm.load_model(args.stage1)
    tr.GRID.update(cfg1['grid'])
    m2, cfg2 = load_stage2(args.stage2)
    infer = dict(s2.INFER, vmin=cfg2.get('vmin', s2.INFER['vmin']))
    test = t2.load_seg_split(os.path.join(args.data, 'test'), args.limit)
    pairs_dir = os.path.join(args.data, 'pairs')
    pairs = t2.load_seg_split(pairs_dir, args.limit) if os.path.isdir(pairs_dir) else []
    rec = tr.make_reconstructor(test[0]['meta'])
    refocus = s2.LocalRefocus(rec)

    res = dict(stage1=os.path.basename(args.stage1), stage2=os.path.basename(args.stage2), infer=infer)
    timings, col, seen = [], {}, set()
    for k, it in enumerate(test):     # un exemple par densité
        if it['meta']['n'] not in seen and len(col) < args.n_examples:
            seen.add(it['meta']['n'])
            col[k] = None
    print(f"chaîne complète, test ({len(test)} hologrammes)", flush=True)
    df, dets, e1 = run(test, m1, cfg1, thr1, m2, rec, refocus, infer, timings=timings, collect=col, rounds=args.rounds)
    res['test'] = summarize(df, dets, e1)
    if args.rounds > 1:
        print("un seul tour (sans les centres vus par l'étage 2)", flush=True)
        res['test_one_round'] = summarize(*run(test, m1, cfg1, thr1, m2, rec, refocus, infer, rounds=1))
    print("étage 2 sur les vraies positions (borne haute)", flush=True)
    df_o, dets_o, e_o = run(test, m1, cfg1, thr1, m2, rec, refocus, infer, oracle=True, rounds=args.rounds)
    res['test_oracle_positions'] = summarize(df_o, dets_o, e_o)
    st = strata(df)
    pairs_sum = None
    if pairs:
        print(f"paires ({len(pairs)} hologrammes)", flush=True)
        dp, detp, ep = run(pairs, m1, cfg1, thr1, m2, rec, refocus, infer, rounds=args.rounds)
        res['pairs'] = summarize(dp, detp, ep)
        rows = []
        for (smp, kind, sep), g in dp.groupby(['sample', 'pair_type', 'pair_sep_um'], sort=False):
            g = g.reset_index(drop=True)
            for q in range(len(g) // 2):
                a, b_ = g.iloc[2 * q], g.iloc[2 * q + 1]
                rows.append(dict(pair_type=kind, sep=sep, segmented=bool(a.iou > 0.5 and b_.iou > 0.5),
                                 iou_min=min(a.iou, b_.iou), from_round2=bool(a['round'] > 1 or b_['round'] > 1)))
        pr = pd.DataFrame(rows)
        pr.to_csv(os.path.join(args.out, 'pairs.csv'), index=False)
        pairs_sum = pr.groupby(['pair_type', 'sep'])[['segmented', 'iou_min', 'from_round2']].mean().reset_index()
        pairs_sum.to_csv(os.path.join(args.out, 'pairs_summary.csv'), index=False)
        print(pairs_sum.round(3).to_string(index=False))
    df.to_csv(os.path.join(args.out, 'per_bacteria.csv'), index=False)
    st.to_csv(os.path.join(args.out, 'strata.csv'), index=False)
    tm = pd.DataFrame(timings)
    tm.to_csv(os.path.join(args.out, 'timings.csv'), index=False)
    if len(tm):
        res['timings_ms_median'] = (tm.drop(columns=['n', 'n_det', 'n_inst']).median() * 1000).round(1).to_dict()
        res['timings_ms_by_n'] = (tm.groupby('n').median() * 1000).round(1).drop(columns=['n_det', 'n_inst']).reset_index().to_dict('records')
    tr_sum = os.path.join(args.stage2, 'train_summary.json')
    if os.path.exists(tr_sum):
        res['training_stage2'] = json.load(open(tr_sum, encoding='utf-8'))
    json.dump(res, open(os.path.join(args.out, 'metrics.json'), 'w', encoding='utf-8'), indent=2, default=float)
    plot_summary(df, st, pairs_sum, args.out)
    plot_examples(col, args.out)
    save_instances(col, args.out)
    pd.set_option('display.width', 200)
    print(json.dumps({k: v for k, v in res.items() if k != 'training_stage2'}, indent=1, default=float))
    print(st.round(3).to_string(index=False))


if __name__ == '__main__':
    main()

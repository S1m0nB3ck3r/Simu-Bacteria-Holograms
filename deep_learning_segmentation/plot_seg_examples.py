# -*- coding: utf-8 -*-

"""
Filename: plot_seg_examples.py

Description:
Exemples de bactéries segmentées par la chaîne complète (étage 1 + étage 2, deux tours), en trois coupes
passant par le centre de la bactérie vraie : XY, XZ, YZ (grille fine 0,175 × 0,175 × 0,25 µm, axes en µm,
z vers le bas). Fond : amplitude du champ refocalisé localement ; vert plein : masque d'instance du
simulateur ; orange pointillé : instance prédite appariée ; traits fins : autres instances vraies (vert)
et prédites (rose) de la fenêtre. Cas choisis sur les hologrammes de test : isolées debout, couchée,
inclinée ; voisine à 2–4 µm ; voisine à 1–2 µm (de préférence trouvée au second tour) ; échec à < 1 µm.
Pour chaque cas, la bactérie dont l'IoU est la plus proche de la médiane de sa catégorie (cas typique,
pas le meilleur).

Utilisation :
  python deep_learning_segmentation/plot_seg_examples.py [--holos 6] [--out resultats/benchmark_segmentation/exemples_3_plans]

Author: Simon BECKER
Date: 2026-10-18

License:
GNU General Public License v3.0
"""

import argparse
import os
import sys

import numpy as np

DL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DL_DIR)
import benchmark as bm
import benchmark_seg as bs
import detection as det
import seg_stage2 as s2
import train as tr
import train_stage2 as t2

CASES = [
    ("isolée, debout (φ < 30°)", lambda r: (r.nn > 8) & (r.phi < 30)),
    ("isolée, couchée (φ > 60°)", lambda r: (r.nn > 8) & (r.phi > 60)),
    ("isolée, inclinée (30–60°)", lambda r: (r.nn > 8) & (r.phi >= 30) & (r.phi <= 60)),
    ("voisine à 2–4 µm", lambda r: (r.nn >= 2) & (r.nn < 4)),
    ("voisine à 1–2 µm, trouvée au 2e tour", lambda r: (r.nn >= 1) & (r.nn < 2) & (r['round'] == 2)),
    ("voisine à moins de 1 µm", lambda r: r.nn < 1),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage1', default=os.path.join(DL_DIR, 'models', 'stage1'))
    ap.add_argument('--stage2', default=os.path.join(DL_DIR, 'models', 'stage2'))
    ap.add_argument('--data', default=os.path.join(DL_DIR, 'data', 'seg_sim', 'test'))
    ap.add_argument('--holos', type=int, default=6)
    ap.add_argument('--out', default=os.path.join(DL_DIR, 'resultats', 'benchmark_segmentation', 'exemples_3_plans'))
    args = ap.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    os.makedirs(args.out, exist_ok=True)
    m1, cfg1, thr1 = bm.load_model(args.stage1)
    tr.apply_model_settings(cfg1)
    m2, cfg2 = bs.load_stage2(args.stage2)
    infer = dict(s2.INFER, vmin=cfg2.get('vmin', s2.INFER['vmin']))
    items = t2.load_seg_split(args.data, args.holos)
    rec = tr.make_reconstructor(items[0]['meta'])
    fine = dict(s2.FINE, **cfg2.get('fine', {}))
    refocus = s2.LocalRefocus(rec, nz=fine['nz'], nxy=fine['nxy'], dz_um=fine['dz_um'],
                              crop=cfg2.get('refocus_crop', s2.REFOCUS_CROP))
    store, rows = {}, []
    for k, it in enumerate(items):
        V = rec(it['h'].astype(np.float32) + 1.0)
        pos1, _ = tr.detect(det.infer_volume(m1, tr.to_input(V, cfg1)), cfg1['target'], thr1, rec)
        insts = s2.segment_instances(m2, V, pos1, refocus, cfg=infer)
        r, _, _ = bs.evaluate_item(it, pos1, insts)
        gtm = [(o, m >= 0.5) for o, m in s2.load_instances(it)]
        match = bs.match_instances(insts, gtm, it['bacts'])
        for j, row in enumerate(r):
            row.update(holo=k, j=j, pred=match[j][0] if j in match else -1)
        rows += r
        # fenêtres centrées sur chaque vraie bactérie (champ refocalisé), gardées pour le tracé
        X, org = refocus(V, it['bacts'][:, :3])
        store[k] = dict(item=it, insts=insts, gtm=gtm, X=X.cpu().numpy().astype(np.float16), org=org)
        print(f"{os.path.basename(it['file'])} : N = {len(it['bacts'])}, {len(insts)} instances", flush=True)
        del V
    df = pd.DataFrame(rows)
    picks = []
    for title, sel in CASES:
        d = df[sel(df)]
        if title.startswith('voisine à 1–2') and d.empty:
            d = df[(df.nn >= 1) & (df.nn < 2)]
        if d.empty:
            continue
        med = d.iou.median()
        picks.append((title, d.iloc[int(np.argmin(np.abs(d.iou.to_numpy() - med)))], med, len(d)))
    g = items[0]['meta']['mask_grid']
    dx, dz = g['dx_um'], g['dz_um']
    fig, ax = plt.subplots(len(picks), 3, figsize=(11, 3.6 * len(picks)))
    ax = np.atleast_2d(ax)
    for row, (title, r, med, n_cat) in enumerate(picks):
        st = store[int(r.holo)]
        j = int(r.j)
        X, wo = st['X'][j].astype(np.float32), st['org'][j]
        A = np.hypot(X[0], X[1])
        shape = A.shape
        gw = [bs._win(m, o, wo, shape) for o, m in st['gtm']]
        me = gw[j]
        pi = int(r.pred)
        pw = [bs._win(d['mask'], d['origin'], wo, shape) for d in st['insts']]
        if me.any():
            kk, rr, cc = [int(round(v.mean())) for v in np.nonzero(me)]
        else:
            kk, rr, cc = shape[0] // 2, shape[1] // 2, shape[2] // 2
        hz, hy, hx = shape[0] * dz / 2, shape[1] * dx / 2, shape[2] * dx / 2
        views = ((lambda v: v[kk], A[kk], [-hx, hx, hy, -hy], f"XY (z = {(wo[0] + kk) * dz:.2f} µm)", 'x (µm)', 'y (µm)'),
                 (lambda v: v[:, rr, :], A[:, rr, :], [-hx, hx, hz, -hz], "XZ", 'x (µm)', 'z (µm)'),
                 (lambda v: v[:, :, cc], A[:, :, cc], [-hy, hy, hz, -hz], "YZ", 'y (µm)', 'z (µm)'))
        for a_, (f, img, ext, name, xl, yl) in zip(ax[row], views):
            a_.imshow(img, cmap='gray', extent=ext, interpolation='nearest')
            yy = np.linspace(ext[3], ext[2], img.shape[0])
            xx = np.linspace(ext[0], ext[1], img.shape[1])
            for q, w in enumerate(gw):
                if q != j and f(w).any():
                    a_.contour(xx, yy, f(w).astype(float), [0.5], colors='#1baf7a', linewidths=0.7, linestyles='dotted')
            for q, w in enumerate(pw):
                if q != pi and f(w).any():
                    a_.contour(xx, yy, f(w).astype(float), [0.5], colors='#e87ba4', linewidths=0.7, linestyles='dotted')
            if f(me).any():
                a_.contour(xx, yy, f(me).astype(float), [0.5], colors='#1baf7a', linewidths=1.8)
            if pi >= 0 and f(pw[pi]).any():
                a_.contour(xx, yy, f(pw[pi]).astype(float), [0.5], colors='#eda100', linewidths=1.5, linestyles='--')
            a_.set_title(name, fontsize=9)
            a_.set_xlabel(xl, fontsize=8)
            a_.set_ylabel(yl, fontsize=8)
            a_.tick_params(labelsize=7)
        ax[row, 0].annotate(f"{title}\nIoU {r.iou:.2f} (médiane de la catégorie {med:.2f}, {n_cat} bactéries)\n"
                            f"φ = {r.phi:.0f}°, z = {r.z:.1f} µm, voisine à {r.nn:.1f} µm"
                            + (f", 2e tour" if r['round'] == 2 else "") + ("" if pi >= 0 else ", NON SEGMENTÉE"),
                            xy=(0, 1.13), xycoords='axes fraction', fontsize=9, fontweight='bold', va='bottom')
    fig.suptitle("Vert : masque du simulateur ; orange pointillé : instance prédite ; traits fins : autres instances "
                 "(vert vraies, rose prédites)", fontsize=10, y=0.999)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    path = os.path.join(args.out, 'exemples_3_plans.png')
    fig.savefig(path, dpi=110)
    print(path)


if __name__ == '__main__':
    main()

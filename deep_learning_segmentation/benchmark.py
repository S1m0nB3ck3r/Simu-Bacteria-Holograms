# -*- coding: utf-8 -*-

"""
Filename: benchmark.py

Description:
Benchmark des U-Net 3D entraînés (runs/<nom>/model_best.pt) et de la méthode classique de référence
(maxima locaux SUM_OF_VARIANCE, K = 15, SEGMENTATION_REPORT), sur les mêmes hologrammes :
  - jeux de test 'hybrid' (fonds réels, images 85–99 du film) et 'sim' (fond plat + bruit), N = 10,
    50, 100, 200 bactéries : rappel, précision, F1, rappel des couchées (φ > 60°), erreurs de
    position (appariement hongrois 3 µm / 5 µm) ;
  - film réel (images 85–99) : rappel des bactéries suivies par HOLOTRACKER (seule vérité disponible)
    et nombre de détections par image ;
  - temps par hologramme 1024 × 1024 : reconstruction, réseau (fp16 et fp32), post-traitement ;
    classique : propagation, focus, détection. Mesures après chauffe, GPU synchronisé.
Seuil de chaque modèle : celui retenu en validation (aucun réglage sur le test).

Sortie : runs/benchmark/ (benchmark.json, metrics.csv, timings.csv, figures PNG).

Utilisation :
  python deep_learning_segmentation/benchmark.py --runs bce_dice focal_tversky centernet wmse

Author: Simon BECKER
Date: 2026-10-17

License:
GNU General Public License v3.0
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

DL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(DL_DIR)
sys.path.insert(0, DL_DIR)
import detection as det
import train as tr
from train import load_split, make_reconstructor, to_input, detect
from unet3d import UNet3D

REAL_FILM = os.path.join(REPO_DIR, 'REAL HOLO FILM', 'bacterias')
REAL_CSV = os.path.join(REAL_FILM, 'TRAJECTORIES_20261016_173113.csv')
REAL_FRAMES = range(85, 100)
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#7a5bd6']
PAIR_TOL = dict(tol_xy=1.0, tol_z=2.5)     # paires : une détection par membre, tolérances strictes
# balayage du seuil (cible 'heat') : rappel en fonction du nombre de détections par image, pour comparer des
# modèles à nombre de détections égal (le seuil de validation simulé n'a pas le même effet sur le film réel)
SWEEP = [0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def sync():
    torch.cuda.synchronize()


def load_model(run_dir):
    ck = torch.load(os.path.join(run_dir, 'model_best.pt'), map_location='cuda', weights_only=False)
    cfg = ck['cfg']
    model = UNet3D(in_ch=2 if cfg['input'] == 'reim' else 1, out_ch=7 if cfg.get('orient_weight', 0) > 0 else 1,
                   base=cfg['base'], depth=cfg['depth'], norm=cfg.get('norm_layer', 'group')).cuda().eval()
    model.load_state_dict(ck['state'])
    return tr.inference_model(model), cfg, ck['threshold']


def real_film_items():
    """Images 85–99 du film, normalisées par la moyenne du film entier ; vérité = points de trajectoire."""
    from PIL import Image
    paths = sorted(glob.glob(os.path.join(REAL_FILM, '*.bmp')))
    mean = np.mean([np.asarray(Image.open(p), dtype=np.float64) for p in paths], axis=0)
    d = pd.read_csv(REAL_CSV)
    items = []
    for f in REAL_FRAMES:
        raw = np.asarray(Image.open(paths[f]), dtype=np.float64)
        t = d[(d['HOLOGRAM NUMBER'] == f + 1) & (d['TRAJECTORY NUMBER'] >= 0)]
        b = np.zeros((len(t), 5), np.float32)
        b[:, 0], b[:, 1], b[:, 2] = t['X POSITION (m)'] * 1e6, t['Y POSITION (m)'] * 1e6, t['Z POSITION (m)'] * 1e6
        items.append(dict(h=(raw / mean - 1.0).astype(np.float32), bacts=b, meta=dict(n=len(t)), file=paths[f]))
    return items


def gt_table(it, r, method, dataset):
    """Une ligne par bactérie vraie : z, φ, N, distance à la plus proche voisine (XY, Z), détectée ou non."""
    b = it['bacts']
    if len(b) == 0:
        return []
    d = b[:, None, :3] - b[None, :, :3]
    dxy = np.hypot(d[..., 0], d[..., 1])
    dz = np.abs(d[..., 2])
    dn = np.sqrt((dxy / 3.0) ** 2 + (dz / 5.0) ** 2)      # distance normalisée par la tolérance
    np.fill_diagonal(dn, np.inf)
    j = np.argmin(dn, axis=1)
    rows = []
    for i in range(len(b)):
        rows.append(dict(method=method, dataset=dataset, n=it['meta']['n'], z=float(b[i, 2]), phi=float(b[i, 4]),
                         nn_xy=float(dxy[i, j[i]]) if len(b) > 1 else np.nan,
                         nn_z=float(dz[i, j[i]]) if len(b) > 1 else np.nan, hit=bool(r['hit'][i])))
    return rows


def pair_rows(it, det_pos, method):
    """Paires 2k / 2k+1 : résolue si les deux membres ont chacun une détection (tolérances strictes)."""
    r = det.evaluate(det_pos, it['bacts'], **PAIR_TOL)
    hit = r['hit']
    return [dict(method=method, pair_type=it['meta']['pair_type'], sep=it['meta']['pair_sep_um'],
                 resolved=bool(hit[2 * k] and hit[2 * k + 1]), one=bool(hit[2 * k] or hit[2 * k + 1]))
            for k in range(len(hit) // 2)]


def pick_examples(items, dataset):
    """Indices des hologrammes illustrés (les mêmes pour toutes les méthodes)."""
    if not items:
        return []
    if dataset == 'paires':
        out = []
        for kind in ('lateral', 'axial'):
            seps = sorted({it['meta']['pair_sep_um'] for it in items if it['meta']['pair_type'] == kind})
            for sep in seps[1:3]:      # 2e et 3e écarts les plus faibles
                out.append(next(k for k, it in enumerate(items)
                                if it['meta']['pair_type'] == kind and it['meta']['pair_sep_um'] == sep))
        return out
    if dataset == 'film réel':
        return [0, len(items) - 1]
    out = {}
    for k, it in enumerate(items):      # un hologramme par densité
        out.setdefault(it['meta']['n'], k)
    return sorted(out.values())[:5]


def plot_detections(it, pos, r, method, dataset, k, out_dir):
    """
    Hologramme (I − 1) avec les détections : vraies en vert (z détecté), fausses en rouge (z détecté), bactéries
    manquées en orange (z vrai). À droite, zoom de 40 µm sur la zone la plus dense. z en µm.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    halo = [pe.withStroke(linewidth=1.2, foreground='black')]
    dx = it['meta'].get('dx_um', 0.175)
    h = np.asarray(it['h'], dtype=np.float32)
    field = h.shape[0] * dx
    gt = it['bacts']
    tp = np.zeros(len(pos), bool)
    tp[r['det_idx']] = True
    miss = ~r['hit']
    # centre du zoom : bactérie vraie qui a le plus de voisines à moins de 8 µm (latéralement)
    if len(gt):
        dxy = np.hypot(gt[:, None, 0] - gt[None, :, 0], gt[:, None, 1] - gt[None, :, 1])
        c = gt[np.argmax((dxy < 8).sum(axis=1)), :2]
    else:
        c = np.array([field / 2, field / 2])
    half = 8.0 if dataset == 'paires' else 20.0
    c = np.clip(c, half, field - half)
    lo, hi = np.percentile(h, [0.5, 99.5])
    fp_name = 'non suivie' if dataset == 'film réel' else 'fausse'
    miss_name = 'trajectoire HOLOTRACKER manquée' if dataset == 'film réel' else 'manquée'

    fig, axes = plt.subplots(1, 2, figsize=(16, 8.4))
    for ax, zoom in ((axes[0], False), (axes[1], True)):
        ax.imshow(h, cmap='gray', vmin=lo, vmax=hi, extent=[0, field, field, 0], interpolation='nearest')
        fs, ms = (6.5, 9) if zoom else (3.2, 3.5)
        groups = ((pos[tp], '#1baf7a', 'o', f'détection vraie ({tp.sum()})'),
                  (pos[~tp], '#e8402a', 's', f'détection {fp_name} ({(~tp).sum()})'),
                  (gt[miss, :3], '#eda100', 'x', f'{miss_name} ({miss.sum()})'))
        for P, col, mk, lab in groups:
            if len(P) == 0:
                continue
            ax.plot(P[:, 0], P[:, 1], mk, mfc='none', mec=col, ms=ms, mew=0.9 if not zoom else 1.4, ls='none', label=lab)
            for x, y, z in P[:, :3]:
                if not zoom or (abs(x - c[0]) < half and abs(y - c[1]) < half):
                    ax.text(x + (0.25 if zoom else 1.2), y - (0.25 if zoom else 1.2), f"{z:.1f}", color=col,
                            fontsize=fs, clip_on=True, path_effects=halo)
        if zoom:
            ax.set_xlim(c[0] - half, c[0] + half)
            ax.set_ylim(c[1] + half, c[1] - half)
            ax.set_title(f"zoom {2 * half:.0f} × {2 * half:.0f} µm (z en µm)", fontsize=10)
        else:
            ax.set_xlim(0, field)
            ax.set_ylim(field, 0)
            ax.add_patch(plt.Rectangle((c[0] - half, c[1] - half), 2 * half, 2 * half, fill=False, ec='#2a78d6', lw=1))
            ax.legend(loc='upper right', fontsize=7, framealpha=0.85)
            ax.set_title(f"{dataset} — {os.path.basename(it['file'])} — N = {len(gt)}", fontsize=10)
        ax.set_xlabel('x (µm)')
        ax.set_ylabel('y (µm)')
    rec_ = r['tp'] / max(len(gt), 1)
    prec = r['tp'] / max(len(pos), 1)
    extra = (f"  ·  paire {it['meta']['pair_type']} {it['meta']['pair_sep_um']} µm"
             if it['meta'].get('pair_type') else '')
    fig.suptitle(f"{method}  —  rappel {rec_:.3f}, précision {prec:.3f}{extra}", fontsize=12)
    fig.tight_layout()
    safe = lambda s: ''.join(ch if ch.isalnum() or ch in '-_' else '_' for ch in s)
    d = os.path.join(out_dir, 'detections', safe(dataset))
    os.makedirs(d, exist_ok=True)
    fig.savefig(os.path.join(d, f"{k:03d}_N{len(gt)}__{safe(method)}.png"), dpi=170)
    plt.close(fig)


def run_model(model, cfg, thr, items, rec, timings, amp=True, tag='', gt_rows=None, dataset='', p_rows=None,
              show=(), out_dir=None, sweep_rows=None):
    rows = []
    acc = {t: [0, 0, 0] for t in SWEEP}           # seuil -> (vrais positifs, détections, vérité)
    # normalisation « film » : chaque jeu est un film, statistiques calculées sur ses hologrammes
    film = tr.film_stats_of([np.asarray(it['h'], dtype=np.float32) + 1.0 for it in items], rec, cfg) \
        if cfg.get('norm') in tr.FILM_NORMS else None
    for k, it in enumerate(items):
        sync(); t0 = time.perf_counter()
        V = to_input(rec(np.asarray(it['h'], dtype=np.float32) + 1.0), cfg, film)
        sync(); t1 = time.perf_counter()
        logits = det.infer_volume(model, V, amp=amp)
        sync(); t2 = time.perf_counter()
        pos, _, axes = detect(logits, cfg['target'], thr, rec, with_axes=True)
        sync(); t3 = time.perf_counter()
        if sweep_rows is not None and cfg['target'] == 'heat' and p_rows is None:
            pa, sa = detect(logits, cfg['target'], min(SWEEP), rec)
            sa = np.asarray(sa)
            for t in SWEEP:
                r = det.evaluate(pa[sa >= t], it['bacts'])
                acc[t][0] += r['tp']; acc[t][1] += r['n_det']; acc[t][2] += r['n_gt']
        if k > 0:      # la première sert de chauffe
            timings.append(dict(method=tag, reconstruction=t1 - t0, network=t2 - t1, postprocess=t3 - t2,
                                total=t3 - t0))
        if k in show:
            plot_detections(it, pos, det.evaluate(pos, it['bacts'], **(PAIR_TOL if p_rows is not None else {})),
                            tag.replace(' fp16', ''), dataset, k, out_dir)
        if p_rows is not None:
            p_rows += pair_rows(it, pos, tag.replace(' fp16', ''))
            continue
        r = det.evaluate(pos, it['bacts'], det_axes=axes)
        r['n'] = it['meta']['n']
        rows.append(r)
        if gt_rows is not None:
            gt_rows += gt_table(it, r, tag.replace(' fp16', ''), dataset)
    if sweep_rows is not None and acc[SWEEP[0]][2]:
        for t, (tp, nd, ng) in acc.items():
            sweep_rows.append(dict(method=tag.replace(' fp16', ''), dataset=dataset, threshold=t, recall=tp / ng,
                                   precision=tp / nd if nd else float('nan'), det_per_image=nd / len(items),
                                   gt_per_image=ng / len(items), model_threshold=thr))
    return rows


CLASSICAL = 'classique (maxima SoV, K=15)'


def run_classical(clf, items, timings, gt_rows=None, dataset='', p_rows=None, show=(), out_dir=None):
    rows = []
    for k, it in enumerate(items):
        tl = []
        pos, _ = clf(np.asarray(it['h'], dtype=np.float32), tl)
        if k > 0:
            timings.append(dict(method='classique', reconstruction=tl[0]['propagation'], network=tl[0]['focus'],
                                postprocess=tl[0]['detection'], total=tl[0]['total']))
        if k in show:
            plot_detections(it, pos, det.evaluate(pos, it['bacts'], **(PAIR_TOL if p_rows is not None else {})),
                            CLASSICAL, dataset, k, out_dir)
        if p_rows is not None:
            p_rows += pair_rows(it, pos, CLASSICAL)
            continue
        r = det.evaluate(pos, it['bacts'])
        r['n'] = it['meta']['n']
        rows.append(r)
        if gt_rows is not None:
            gt_rows += gt_table(it, r, CLASSICAL, dataset)
    return rows


def by_density(rows, method, dataset):
    out = []
    for n in sorted(set(r['n'] for r in rows)):
        s = det.summarize([r for r in rows if r['n'] == n])
        out.append(dict(method=method, dataset=dataset, n=n, **s))
    s = det.summarize(rows)
    out.append(dict(method=method, dataset=dataset, n='all', **s))
    return out


def plot_metrics(df, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for dataset in df.dataset.unique():
        d = df[(df.dataset == dataset) & (df.n != 'all')]
        if d.empty or dataset == 'film réel':
            continue
        fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
        for ax, (col, title) in zip(axes, [('recall', 'Rappel'), ('precision', 'Précision'), ('f1', 'F1'),
                                           ('recall_lying', 'Rappel des couchées (φ > 60°)')]):
            for i, m in enumerate(d.method.unique()):
                s = d[d.method == m]
                ax.plot(s.n.astype(int), s[col], '-o', lw=2, ms=6, color=COLORS[i % len(COLORS)], label=m)
            ax.set_title(title)
            ax.set_xlabel('bactéries par hologramme')
            ax.set_xscale('log')
            ns = sorted(d.n.astype(int).unique())
            ax.set_xticks(ns)
            ax.set_xticklabels([str(n) for n in ns])
            ax.minorticks_off()
            ax.grid(alpha=0.25)
            ax.spines[['top', 'right']].set_visible(False)
        axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle(f"Détection — jeu de test {dataset}")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"detection_{dataset.replace('/', '_')}.png"), dpi=110)
        plt.close(fig)


def plot_pairs(ps, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, kind, title in ((axes[0], 'lateral', 'Écart latéral (même z)'), (axes[1], 'axial', 'Écart axial (même x, y)')):
        d = ps[ps.pair_type == kind]
        for i, m in enumerate(d.method.unique()):
            s = d[d.method == m]
            ax.plot(s.sep, s.resolved, '-o', lw=2, ms=6, color=COLORS[i % len(COLORS)], label=m)
        ax.set_title(title)
        ax.set_xlabel('écart entre les deux bactéries (µm)')
        ax.set_ylabel('paires résolues (2 détections)')
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "paires.png"), dpi=110)
    plt.close(fig)


def plot_timings(tm, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    g = tm.groupby('method')[['reconstruction', 'network', 'postprocess']].median() * 1000
    g = g.loc[g.sum(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(g) + 1.5))
    left = np.zeros(len(g))
    labels = {'reconstruction': 'reconstruction / propagation', 'network': 'réseau / focus', 'postprocess': 'post-traitement / détection'}
    for i, c in enumerate(['reconstruction', 'network', 'postprocess']):
        ax.barh(g.index, g[c], left=left, color=COLORS[i], label=labels[c], edgecolor='white', linewidth=2)
        left += g[c].to_numpy()
    for y, v in enumerate(left):
        ax.text(v + 10, y, f"{v:.0f} ms", va='center', fontsize=9)
    ax.axvline(400, color='#888888', lw=1, ls='--')
    ax.text(405, len(g) - 0.6, "budget 400 ms", color='#555555', fontsize=8)
    ax.set_xlabel('temps médian par hologramme 1024 × 1024 (ms)')
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "temps_inference.png"), dpi=110)
    plt.close(fig)


def plot_training(run_dirs, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for i, rd in enumerate(run_dirs):
        hpath = os.path.join(rd, 'history.json')
        if not os.path.exists(hpath):
            continue
        h = pd.DataFrame(json.load(open(hpath, encoding='utf-8')))
        name = os.path.basename(rd)
        t = h['elapsed_s'] if 'elapsed_s' in h else h['train_time_s']
        axes[0].plot(t / 60, h.f1, '-o', lw=2, ms=5, color=COLORS[i % len(COLORS)], label=name)
        if 'train_loss' in h:
            axes[1].plot(h.it, h.train_loss / h.train_loss.iloc[0], '-', lw=2, color=COLORS[i % len(COLORS)], label=name)
    axes[0].set_xlabel("temps d'entraînement (min)")
    axes[0].set_title('F1 de validation (volumes entiers)')
    axes[1].set_xlabel('itérations')
    axes[1].set_title("loss d'entraînement (rapportée à la 1re mesure)")
    axes[1].set_yscale('log')
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "entrainement.png"), dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--datasets', nargs='+', default=['dataset_sim_200um'],
                    help="jeux (dossiers contenant test/) : chemin, ou nom dans datasets/ ou data/")
    ap.add_argument('--no_real', action='store_true')
    ap.add_argument('--no_classical', action='store_true')
    ap.add_argument('--no_pairs', action='store_true')
    ap.add_argument('--fp32_subset', type=int, default=6)
    ap.add_argument('--no_examples', action='store_true', help="pas d'images annotées (detections/)")
    ap.add_argument('--pairs_dir', default=os.path.join(DL_DIR, 'data', 'seg_sim', 'pairs'), help="jeu de paires rapprochées")
    ap.add_argument('--out', default=os.path.join(DL_DIR, 'resultats', 'benchmark_detection'))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # jeu : chemin tel quel, sinon dans deep_learning_segmentation/, datasets/ ou data/ ; nommé par son dossier
    find = lambda d: next((p for p in (d, os.path.join(DL_DIR, d), os.path.join(DL_DIR, 'datasets', d),
                                       os.path.join(DL_DIR, 'data', d)) if os.path.isdir(os.path.join(p, 'test'))), d)
    test_dirs = {os.path.basename(os.path.normpath(d)): os.path.join(find(d), 'test') for d in args.datasets}
    args.datasets = list(test_dirs)
    sets = {d: load_split(p) for d, p in test_dirs.items()}
    if not args.no_real:
        sets['film réel'] = real_film_items()
    pairs_dir = args.pairs_dir
    pairs = load_split(pairs_dir) if (not args.no_pairs and os.path.isdir(pairs_dir)) else []
    meta = next(iter(load_split(test_dirs[args.datasets[0]], 1)))['meta']
    metrics, timings, gt_rows, p_rows, sweep_rows = [], [], [], [], []
    show = {name: set(pick_examples(items, name)) if not args.no_examples else set()
            for name, items in list(sets.items()) + [('paires', pairs)]}

    if not args.no_classical:
        import cupy as cp
        clfs = {}
        for name, items in list(sets.items()) + ([('paires', pairs)] if pairs else []):
            # profondeur reconstruite : 0–100 µm (200 plans à 0,5 µm), ou 0–200 µm pour les jeux profonds
            zmax = max(float(it['meta'].get('z_range_um', [0, 95])[1]) for it in items)
            nb = 200 if zmax <= 100 else 400
            if nb not in clfs:
                clfs.clear()
                cp.get_default_memory_pool().free_all_blocks()
                clfs[nb] = det.ClassicalDetector(nb_planes=nb)
            print(f"classique / {name} ({nb} plans)", flush=True)
            if name == 'paires':
                run_classical(clfs[nb], items, [], dataset=name, p_rows=p_rows, show=show[name], out_dir=args.out)
                continue
            rows = run_classical(clfs[nb], items, timings if name == args.datasets[0] else [], gt_rows, name,
                                 show=show[name], out_dir=args.out)
            metrics += by_density(rows, CLASSICAL, name)
        clfs.clear()
        cp.get_default_memory_pool().free_all_blocks()

    # chemin tel quel, sinon relatif à deep_learning_segmentation/ (models/stage1), sinon un nom de runs/
    run_dirs = [next((d for d in (r, os.path.join(DL_DIR, r), os.path.join(DL_DIR, 'runs', r)) if os.path.isdir(d)), r)
                for r in args.runs]
    for rd in run_dirs:
        model, cfg, thr = load_model(rd)
        tr.apply_model_settings(cfg)          # grille, filtre et maxima propres au modèle
        rec = make_reconstructor(meta)
        name = os.path.basename(rd)
        for dname, items in sets.items():
            print(f"{name} / {dname} (seuil {thr})", flush=True)
            rows = run_model(model, cfg, thr, items, rec, timings if dname == args.datasets[0] else [],
                             tag=f"{name} fp16", gt_rows=gt_rows, dataset=dname, show=show[dname], out_dir=args.out,
                             sweep_rows=sweep_rows)
            metrics += by_density(rows, name, dname)
        if pairs:
            print(f"{name} / paires", flush=True)
            run_model(model, cfg, thr, pairs, rec, [], tag=f"{name} fp16", dataset='paires', p_rows=p_rows,
                      show=show['paires'], out_dir=args.out)
        # temps en fp32 sur quelques hologrammes
        try:
            run_model(model, cfg, thr, sets[args.datasets[0]][:args.fp32_subset + 1], rec, timings, amp=False,
                      tag=f"{name} fp32")
        except RuntimeError as e:      # OutOfMemoryError ou « CUDA error: out of memory » (cuDNN)
            if 'out of memory' not in str(e):
                raise
            print(f"{name} : fp32 hors mémoire, temps fp32 non mesuré", flush=True)
            torch.cuda.empty_cache()
        del model
        torch.cuda.empty_cache()

    df = pd.DataFrame(metrics)
    df.to_csv(os.path.join(args.out, 'metrics.csv'), index=False)
    if sweep_rows:
        pd.DataFrame(sweep_rows).to_csv(os.path.join(args.out, 'threshold_sweep.csv'), index=False)
    pd.DataFrame(gt_rows).to_csv(os.path.join(args.out, 'gt_table.csv'), index=False)
    if p_rows:
        pr = pd.DataFrame(p_rows)
        pr.to_csv(os.path.join(args.out, 'pairs.csv'), index=False)
        ps = pr.groupby(['method', 'pair_type', 'sep'])[['resolved', 'one']].mean().reset_index()
        ps.to_csv(os.path.join(args.out, 'pairs_summary.csv'), index=False)
        plot_pairs(ps, args.out)
        print(ps.round(3).to_string(index=False))
    tm = pd.DataFrame(timings)
    tm.to_csv(os.path.join(args.out, 'timings.csv'), index=False)
    tsum = (tm.groupby('method')[['reconstruction', 'network', 'postprocess', 'total']].median() * 1000).round(1)
    train = {}
    for rd in run_dirs:
        p = os.path.join(rd, 'train_summary.json')
        if os.path.exists(p):
            train[os.path.basename(rd)] = json.load(open(p, encoding='utf-8'))
    json.dump(dict(metrics=metrics, timings_ms=tsum.reset_index().to_dict('records'), training=train),
              open(os.path.join(args.out, 'benchmark.json'), 'w', encoding='utf-8'), indent=2, default=str)
    plot_metrics(df, args.out)
    plot_timings(tm, args.out)
    plot_training(run_dirs, args.out)
    pd.set_option('display.width', 200)
    cols = [c for c in ['method', 'dataset', 'n', 'recall', 'precision', 'f1', 'recall_lying', 'sigma_x', 'sigma_y',
                        'sigma_z', 'bias_z', 'angle_median_deg'] if c in df.columns]
    print(df[cols].round(3).to_string(index=False))
    print(tsum.to_string())


if __name__ == '__main__':
    main()

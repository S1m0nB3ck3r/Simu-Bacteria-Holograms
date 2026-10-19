# -*- coding: utf-8 -*-

"""
Filename: train_stage2.py

Description:
Entraînement de l'étage 2 (seg_stage2.py) : U-Net 3D sur des fenêtres fines (32 × 40 × 40 voxels de
0,25 × 0,175 × 0,175 µm) autour des bactéries, cibles = masques d'instance du simulateur (jeux de
gen_seg_data.sh). À chaque pas, holos_per_step hologrammes sont reconstruits (volume grossier, 1 µm,
décimé ×2, 0–208 µm) ; crops_per_holo fenêtres en sont tirées :
  - centrées sur une bactérie tirée au hasard, décalées d'une erreur gaussienne (jitter_xy, jitter_z :
    erreur de position de l'étage 1, prise large) ;
  - mid_fraction au milieu d'une bactérie et de sa plus proche voisine si elle est à moins de 3 µm (ce que
    voit l'étage 2 quand l'étage 1 a fusionné deux bactéries) ;
  - neg_fraction au hasard dans le volume (fausses détections : la cible self est le plus souvent vide).
Cibles (seg_stage2.CropTargets) : self, others, carte fine des centres.
Augmentations exactes : symétries x, y et échange x ↔ y (entrée et cibles ensemble).
Validation : fenêtres autour de toutes les bactéries des hologrammes de validation, mêmes décalages
(graine fixe) ; IoU exacte de self (prédiction > 0,5 contre fraction ≥ 0,5, instance entière même si
elle déborde de la fenêtre) ; centres : rappel et précision des maxima de la carte des centres contre les
vrais centres présents dans chaque fenêtre ; meilleur modèle = moyenne de l'IoU et du F1 des centres.
Sortie : runs/<nom>/ (model_best.pt, config.json, history.json, train_summary.json).

Utilisation :
  python deep_learning_segmentation/train_stage2.py --name stage2

Author: Simon BECKER
Date: 2026-10-18

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
import torch

DL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DL_DIR)
import holo_physics as hp
import seg_stage2 as s2
import detection as det
import train as tr
from unet3d import UNet3D, count_parameters

GRID = dict(z0_um=0.0, step_z_um=1.0, nb_planes=208, decim=2)


def load_seg_split(path, limit=None):
    items = []
    for f in sorted(glob.glob(os.path.join(path, 'sample_*.npz')))[:limit]:
        d = np.load(f)
        items.append(dict(h=d['h'], bacts=d['bacts'], meta=json.loads(str(d['meta'])), file=f,
                          mask_origin=d['mask_origin'], mask_shape=d['mask_shape'], mask_data=d['mask_data']))
    return items


def make_reconstructor(meta):
    z = GRID['z0_um'] + np.arange(GRID['nb_planes']) * GRID['step_z_um']
    return hp.Reconstructor(1024, meta['dx_um'], meta['wavelength_um'], meta['index_medium'], z, decim=GRID['decim'])


def crop_points(item, n, rng, cfg, field_um, zr):
    """Centres des fenêtres : bactéries + erreur de position, milieux de paires proches, points au hasard."""
    b = item['bacts']
    n_neg = int(round(cfg['neg_fraction'] * n)) if len(b) else n
    n_mid = 0
    mids = np.zeros((0, 3))
    if len(b) > 1 and cfg['mid_fraction'] > 0:
        from scipy.spatial import cKDTree
        d, j = cKDTree(b[:, :3]).query(b[:, :3], k=2)
        close = np.nonzero(d[:, 1] < 3.0)[0]
        if len(close):
            n_mid = int(round(cfg['mid_fraction'] * n))
            i = rng.choice(close, n_mid)
            mids = 0.5 * (b[i, :3] + b[j[i, 1], :3])
    n_pos = n - n_neg - n_mid
    j = rng.integers(0, len(b), n_pos) if len(b) else np.zeros(0, int)
    jit = np.array([cfg['jitter_xy'], cfg['jitter_xy'], cfg['jitter_z']])
    pos = b[j, :3] + rng.normal(0, 1, (n_pos, 3)) * jit
    neg = np.stack([rng.uniform(0, field_um, n_neg), rng.uniform(0, field_um, n_neg), rng.uniform(*zr, n_neg)], 1)
    return np.concatenate([pos, mids, neg]).astype(np.float64)


def augment(x, t, rng):
    for b in range(len(x)):
        for dims in ((-1,), (-2,)):
            if rng.random() < 0.5:
                x[b], t[b] = x[b].flip(dims), t[b].flip(dims)
        if rng.random() < 0.5:
            x[b], t[b] = x[b].transpose(-1, -2).clone(), t[b].transpose(-1, -2).clone()
    return x, t


@torch.no_grad()
def validate(model, val, rec, refocus, cfg, seed=123):
    model.eval()
    m = tr.inference_model(model)
    rng = np.random.default_rng(seed)
    ious, fp = [], []
    c_tp = c_gt = c_det = 0
    for it in val:
        V = rec(it['h'].astype(np.float32) + 1.0)
        tg = s2.CropTargets(it)
        b = it['bacts'][:, :3]
        pts = b + rng.normal(0, 1, b.shape) * np.array([cfg['jitter_xy'], cfg['jitter_xy'], cfg['jitter_z']])
        neg = np.stack([rng.uniform(0, 179.2, 20), rng.uniform(0, 179.2, 20), rng.uniform(2, 200, 20)], 1)
        allp = np.concatenate([pts, neg])
        P, org, H = s2.segment(m, V, allp, refocus, with_heat=True)
        owner = s2.assign(b, allp)
        # centres : maxima de la carte contre les vrais centres présents dans chaque fenêtre (hors bord)
        g = it['meta']['mask_grid']
        peaks, _, win = s2.window_peaks(H, org, g['dx_um'], g['dz_um'])
        brd = s2.INFER['border']
        for i in range(len(allp)):
            lo = (org[i] + brd) * np.array([g['dz_um'], g['dx_um'], g['dx_um']])
            hi = (org[i] + np.array(P.shape[1:]) - brd) * np.array([g['dz_um'], g['dx_um'], g['dx_um']])
            zyx = b[:, [2, 1, 0]]
            inside = b[np.all((zyx >= lo) & (zyx < hi), axis=1)]
            pk = peaks[win == i]
            c_gt += len(inside)
            c_det += len(pk)
            if len(inside) and len(pk):
                c_tp += len(det.match(pk, inside, tol_xy=0.5, tol_z=1.0)[0])
        for i in range(len(allp)):
            pred = (P[i] > 0.5).cpu().numpy()
            if owner[i] < 0:
                fp.append(pred.sum() >= cfg['vmin'])
                continue
            t = tg(org[i:i + 1], owner[i:i + 1], allp[i:i + 1], heat=False)[0, 0] >= 0.5
            inter = int((pred & t).sum())
            tot = tg.full_volume_voxels(owner[i])
            ious.append(inter / max(int(pred.sum()) + tot - inter, 1))
    model.train()
    ious = np.array(ious)
    cr, cp_ = c_tp / max(c_gt, 1), c_tp / max(c_det, 1)
    cf1 = 2 * cr * cp_ / (cr + cp_) if cr + cp_ else 0.0
    return dict(iou_mean=float(ious.mean()), iou_median=float(np.median(ious)), iou_gt50=float((ious > 0.5).mean()),
                n=int(len(ious)), neg_fp_rate=float(np.mean(fp)) if fp else float('nan'),
                center_recall=cr, center_precision=cp_, center_f1=cf1, score=0.5 * (float(ious.mean()) + cf1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--data', default=os.path.join(DL_DIR, 'data', 'seg_sim'))
    ap.add_argument('--out', default=os.path.join(DL_DIR, 'runs'))
    ap.add_argument('--iters', type=int, default=6000)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--warmup', type=int, default=200)
    ap.add_argument('--holos_per_step', type=int, default=2)
    ap.add_argument('--crops_per_holo', type=int, default=16)
    ap.add_argument('--neg_fraction', type=float, default=0.1)
    ap.add_argument('--mid_fraction', type=float, default=0.15)
    ap.add_argument('--jitter_xy', type=float, default=0.25)
    ap.add_argument('--jitter_z', type=float, default=0.5)
    ap.add_argument('--vmin', type=int, default=20, help="voxels self minimum pour garder une détection")
    ap.add_argument('--base', type=int, default=16)
    ap.add_argument('--depth', type=int, default=3)
    ap.add_argument('--eval_every', type=int, default=500)
    ap.add_argument('--val_limit', type=int, default=12)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    cfg = vars(args)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = os.path.join(args.out, args.name)
    os.makedirs(out, exist_ok=True)

    train = load_seg_split(os.path.join(args.data, 'train'))
    val = load_seg_split(os.path.join(args.data, 'val'), args.val_limit)
    meta = train[0]['meta']
    rec = make_reconstructor(meta)
    refocus = s2.LocalRefocus(rec)
    cfg.update(grid=dict(GRID), fine=dict(s2.FINE), mask_grid=meta['mask_grid'])
    model = UNet3D(in_ch=2, out_ch=3, base=args.base, depth=args.depth, norm='batch').cuda()
    model = model.to(memory_format=torch.channels_last_3d)
    cfg['n_params'] = count_parameters(model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    with open(os.path.join(out, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2)
    print(f"{len(train)} hologrammes d'entraînement, {len(val)} de validation, {cfg['n_params']} paramètres", flush=True)

    history, best, losses = [], -1.0, []
    t0, t_val, t_data, t_net = time.time(), 0.0, 0.0, 0.0
    torch.cuda.reset_peak_memory_stats()
    # arrêt propre (GUI_train.py) : fichier STOP dans le dossier du run -> dernière validation, bilan, fin
    stop_file = os.path.join(out, 'STOP')
    if os.path.exists(stop_file):
        os.remove(stop_file)
    stop_now = False
    for it in range(args.iters):
        stop_now = os.path.exists(stop_file)
        for g in opt.param_groups:
            g['lr'] = tr.lr_at(it, cfg)
        td = time.time()
        xs, ts = [], []
        for k in rng.choice(len(train), args.holos_per_step, replace=False):
            item = train[k]
            V = rec(item['h'].astype(np.float32) + 1.0)
            pts = crop_points(item, args.crops_per_holo, rng, cfg, 179.2, item['meta'].get('z_range_um', [2, 200]))
            X, org = refocus(V, pts)
            owner = s2.assign(item['bacts'][:, :3], pts)
            T = s2.CropTargets(item)(org, owner, pts)
            xs.append(X)
            ts.append(torch.from_numpy(T).cuda())
        x, t = augment(torch.cat(xs), torch.cat(ts), rng)
        torch.cuda.synchronize()
        tn = time.time()
        t_data += tn - td
        x = x.contiguous(memory_format=torch.channels_last_3d)
        with torch.autocast('cuda', dtype=torch.float16):
            logits = model(x)
        loss = s2.seg_loss(logits.float(), t)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(opt)
        scaler.update()
        torch.cuda.synchronize()
        t_net += time.time() - tn
        losses.append(float(loss))
        if (it + 1) % 100 == 0:
            n = it + 1
            print(f"it {n}/{args.iters}  loss {np.mean(losses[-100:]):.4f}  {(time.time() - t0 - t_val) / n:.3f} s/it "
                  f"(données {t_data / n:.3f}, réseau {t_net / n:.3f})", flush=True)
        if (it + 1) % args.eval_every == 0 or it + 1 == args.iters or stop_now:
            tv = time.time()
            s = validate(model, val, rec, refocus, cfg)
            t_val += time.time() - tv
            history.append(dict(it=it + 1, train_loss=float(np.mean(losses[-args.eval_every:])),
                                elapsed_s=time.time() - t0 - t_val, val_time_s=time.time() - tv, **s))
            print(f"  VALIDATION it {it + 1} : IoU moyenne {s['iou_mean']:.3f}, médiane {s['iou_median']:.3f}, "
                  f"IoU > 0,5 {s['iou_gt50']:.3f} ({s['n']} bactéries), négatifs gardés {s['neg_fp_rate']:.3f} ; "
                  f"centres rappel {s['center_recall']:.3f} précision {s['center_precision']:.3f} "
                  f"[{time.time() - tv:.0f} s]", flush=True)
            if s['score'] > best:
                best = s['score']
                torch.save(dict(state=model.state_dict(), cfg=cfg), os.path.join(out, 'model_best.pt'))
            with open(os.path.join(out, 'history.json'), 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2)
        if stop_now:
            os.remove(stop_file)
            print(f"  ARRÊT demandé (fichier STOP) à l'itération {it + 1}", flush=True)
            break
    n = it + 1
    summary = dict(name=args.name, iters=n, stopped=stop_now, n_params=cfg['n_params'], train_time_s=time.time() - t0 - t_val,
                   s_per_it=(time.time() - t0 - t_val) / n, data_s_per_it=t_data / n, net_s_per_it=t_net / n,
                   peak_gpu_mem_gb=torch.cuda.max_memory_allocated() / 1e9, best_val_score=best,
                   best=max(history, key=lambda h: h['score']), last=history[-1])
    with open(os.path.join(out, 'train_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()

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
import math
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
import simu_dataset as sd
from unet3d import UNet3D, count_parameters

# =============================================================================
# RÉGLAGES PAR DÉFAUT — options de la ligne de commande et de GUI_train.py (onglet Étage 2) ; les valeurs
# utilisées sont enregistrées dans config.json et model_best.pt, et reprises par l'inférence
# =============================================================================
DECIM = 2                   # décimation latérale du volume grossier (1, 2 ou 4) : la même que l'étage 1
HIGH_PASS = 15              # filtre passe-haut de la reconstruction (px de Fourier du capteur, 0 : aucun)
LOW_PASS = 250              # filtre passe-bas (0 : aucun)
Z_STEP_UM = 1.0             # pas Z du volume grossier d'où partent les fenêtres (µm)
FINE_NZ = 32                # fenêtre fine : nombre de plans (pas = pas des masques du jeu, 0,25 µm : 8 µm)
FINE_NXY = 40               # fenêtre fine : côté en voxels fins (pixel objet, 0,175 µm : 7 µm)
REFOCUS_CROP = 64           # carré du volume grossier refocalisé autour de chaque fenêtre (px grossiers)
HEAT_SIGMA_XY_UM = 0.25     # cible : σ des gaussiennes de la carte fine des centres en XY (µm)
HEAT_SIGMA_Z_UM = 0.5       # idem en Z (µm)
TOL_XY_UM = 1.0             # une fenêtre appartient à la bactérie la plus proche si elle est à moins de TOL_XY en XY
TOL_Z_UM = 2.0              # et TOL_Z en Z (distance normalisée < 1) ; sinon fenêtre négative (self vide)
CROPS_PER_HOLO = 16         # fenêtres tirées par hologramme reconstruit
NEG_FRACTION = 0.1          # part des fenêtres au hasard (fausses détections)
MID_FRACTION = 0.15         # part des fenêtres au milieu de deux voisines proches (détection fusionnée)
MID_PAIR_UM = 3.0           # « voisines proches » : centres à moins de MID_PAIR_UM
JITTER_XY_UM = 0.25         # décalage aléatoire (σ) des fenêtres, XY : erreur de position simulée de l'étage 1
JITTER_Z_UM = 0.5           # idem en Z
VMIN = 20                   # voxels minimum d'une instance (validation, repris à l'inférence)

GRID = dict(z0_um=0.0, step_z_um=Z_STEP_UM, nb_planes=208, decim=DECIM)
RECON = dict(high_pass=HIGH_PASS, low_pass=LOW_PASS)


def load_seg_split(path, limit=None):
    """Dossier train/ ou val/ : jeu de GUI_generate_dataset.py (simu_dataset.py) ou .npz de generate_dataset.py --save_masks."""
    if sd.is_dataset(path):
        return sd.load_split(path, limit, masks=True)
    items = []
    for f in sorted(glob.glob(os.path.join(path, 'sample_*.npz')))[:limit]:
        d = np.load(f)
        items.append(dict(h=d['h'], bacts=d['bacts'], meta=json.loads(str(d['meta'])), file=f,
                          mask_origin=d['mask_origin'], mask_shape=d['mask_shape'], mask_data=d['mask_data']))
    return items


def make_reconstructor(meta):
    z = GRID['z0_um'] + np.arange(GRID['nb_planes']) * GRID['step_z_um']
    return hp.Reconstructor(meta.get('holo_size', 1024), meta['dx_um'], meta['wavelength_um'], meta['index_medium'], z,
                            decim=GRID['decim'], high_pass=RECON['high_pass'], low_pass=RECON['low_pass'])


def field_of(meta):
    """Champ latéral (µm) et plage de profondeur des bactéries (µm) d'un hologramme."""
    return meta.get('holo_size', 1024) * meta['dx_um'], meta.get('z_range_um', [2, 200])


def crop_points(item, n, rng, cfg, field_um, zr):
    """Centres des fenêtres : bactéries + erreur de position, milieux de paires proches, points au hasard."""
    b = item['bacts']
    n_neg = int(round(cfg['neg_fraction'] * n)) if len(b) else n
    n_mid = 0
    mids = np.zeros((0, 3))
    if len(b) > 1 and cfg['mid_fraction'] > 0:
        from scipy.spatial import cKDTree
        d, j = cKDTree(b[:, :3]).query(b[:, :3], k=2)
        close = np.nonzero(d[:, 1] < cfg.get('mid_pair_um', MID_PAIR_UM))[0]
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
        tg = s2.CropTargets(it, refocus.nz, refocus.nxy, heat=cfg.get('heat'))
        b = it['bacts'][:, :3]
        pts = b + rng.normal(0, 1, b.shape) * np.array([cfg['jitter_xy'], cfg['jitter_xy'], cfg['jitter_z']])
        field, zr = field_of(it['meta'])
        neg = np.stack([rng.uniform(0, field, 20), rng.uniform(0, field, 20), rng.uniform(*zr, 20)], 1)
        allp = np.concatenate([pts, neg])
        P, org, H = s2.segment(m, V, allp, refocus, with_heat=True)
        owner = s2.assign(b, allp, tol=cfg.get('tol', s2.TOL))
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
    ap = argparse.ArgumentParser(description="Entraînement de l'étage 2 (segmentation d'instance) du U-Net 3D")
    ap.add_argument('--name', required=True, help="nom du run (sous-dossier de --out)")
    ap.add_argument('--data', default=os.path.join(DL_DIR, 'datasets', 'dataset_sim_200um_N20-400'),
                    help="dossier d'un jeu avec masques d'instance (train/, val/)")
    ap.add_argument('--z_min_um', type=float, default=None,
                    help="début du volume grossier (µm) ; défaut : distance volume-caméra du jeu (0 pour un jeu .npz)")
    ap.add_argument('--z_max_um', type=float, default=None,
                    help="fin du volume grossier (µm) ; défaut : distance + plans × pas du jeu (207 pour un jeu .npz)")
    ap.add_argument('--out', default=os.path.join(DL_DIR, 'runs'), help="dossier des runs")
    ap.add_argument('--decim', type=int, choices=[1, 2, 4], default=DECIM,
                    help="décimation latérale du volume grossier ; doit être celle de l'étage 1 utilisé avec ce modèle")
    ap.add_argument('--high_pass', type=float, default=HIGH_PASS, help="filtre passe-haut (px de Fourier du capteur)")
    ap.add_argument('--low_pass', type=float, default=LOW_PASS, help="filtre passe-bas (px de Fourier du capteur)")
    ap.add_argument('--z_step', type=float, default=Z_STEP_UM, help="pas Z du volume grossier (µm)")
    ap.add_argument('--fine_nz', type=int, default=FINE_NZ,
                    help="fenêtre fine : nombre de plans (pas = pas des masques du jeu) ; multiple de 2^niveaux")
    ap.add_argument('--fine_nxy', type=int, default=FINE_NXY,
                    help="fenêtre fine : côté en voxels fins (pixel objet) ; multiple de 2^niveaux")
    ap.add_argument('--refocus_crop', type=int, default=REFOCUS_CROP,
                    help="carré du volume grossier refocalisé autour de chaque fenêtre (px grossiers, pair)")
    ap.add_argument('--heat_sigma_xy_um', type=float, default=HEAT_SIGMA_XY_UM,
                    help="cible : σ des gaussiennes de la carte fine des centres en XY (µm)")
    ap.add_argument('--heat_sigma_z_um', type=float, default=HEAT_SIGMA_Z_UM, help="idem en Z (µm)")
    ap.add_argument('--tol_xy_um', type=float, default=TOL_XY_UM,
                    help="attribution d'une fenêtre à la bactérie la plus proche : tolérance XY (µm)")
    ap.add_argument('--tol_z_um', type=float, default=TOL_Z_UM, help="idem en Z (µm)")
    ap.add_argument('--iters', type=int, default=6000, help="itérations (pas d'optimisation)")
    ap.add_argument('--lr', type=float, default=1e-3, help="taux d'apprentissage maximal (AdamW)")
    ap.add_argument('--warmup', type=int, default=200, help="itérations de montée linéaire du taux, puis cosinus")
    ap.add_argument('--holos_per_step', type=int, default=2, help="hologrammes reconstruits par pas")
    ap.add_argument('--crops_per_holo', type=int, default=CROPS_PER_HOLO, help="fenêtres tirées de chaque hologramme")
    ap.add_argument('--neg_fraction', type=float, default=NEG_FRACTION, help="part des fenêtres au hasard (fausses détections)")
    ap.add_argument('--mid_fraction', type=float, default=MID_FRACTION,
                    help="part des fenêtres au milieu de deux voisines proches (détection fusionnée)")
    ap.add_argument('--mid_pair_um', type=float, default=MID_PAIR_UM, help="distance maximale de deux « voisines proches » (µm)")
    ap.add_argument('--jitter_xy', type=float, default=JITTER_XY_UM,
                    help="décalage aléatoire σ des fenêtres en XY (µm) : erreur de position simulée de l'étage 1")
    ap.add_argument('--jitter_z', type=float, default=JITTER_Z_UM, help="idem en Z (µm)")
    ap.add_argument('--vmin', type=int, default=VMIN, help="voxels self minimum pour garder une détection")
    ap.add_argument('--base', type=int, default=16, help="canaux du premier niveau (doublés à chaque niveau)")
    ap.add_argument('--depth', type=int, default=3, help="niveaux du U-Net (sous-échantillonnages par 2)")
    ap.add_argument('--eval_every', type=int, default=500, help="validation toutes les N itérations")
    ap.add_argument('--val_limit', type=int, default=12, help="hologrammes de validation utilisés")
    ap.add_argument('--seed', type=int, default=0, help="graine")
    args = ap.parse_args()
    cfg = vars(args)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = os.path.join(args.out, args.name)
    os.makedirs(out, exist_ok=True)

    train = load_seg_split(os.path.join(args.data, 'train'))
    val = load_seg_split(os.path.join(args.data, 'val'), args.val_limit)
    meta = train[0]['meta']
    GRID.update(step_z_um=args.z_step, decim=args.decim)
    RECON.update(high_pass=args.high_pass, low_pass=args.low_pass)
    mult = 2 ** args.depth
    if args.fine_nz % mult or args.fine_nxy % mult:
        raise ValueError(f"--fine_nz et --fine_nxy : multiples de {mult} (2^niveaux)")
    # volume grossier : de z_min à z_max (volume simulé du jeu par défaut), plans multiples de 2^depth
    args.z_min_um, args.z_max_um = sd.resolve_range(os.path.join(args.data, 'train'), args.z_min_um, args.z_max_um)
    GRID['z0_um'], GRID['nb_planes'] = sd.grid_range(args.z_min_um, args.z_max_um, GRID['step_z_um'], 2 ** args.depth)
    print(f"volume grossier : {GRID['z0_um']:.1f}–{GRID['z0_um'] + (GRID['nb_planes'] - 1) * GRID['step_z_um']:.1f} µm, "
          f"{GRID['nb_planes']} plans", flush=True)
    rec = make_reconstructor(meta)
    # grille fine = grille des masques du simulateur (pas BPM du jeu)
    fine = dict(nz=args.fine_nz, nxy=args.fine_nxy, dz_um=meta['mask_grid']['dz_um'])
    if abs(meta['mask_grid']['dx_um'] - rec.dx_um) > 1e-6:
        raise ValueError(f"grille des masques {meta['mask_grid']['dx_um']} µm ≠ pixel objet {rec.dx_um} µm")
    refocus = s2.LocalRefocus(rec, nz=fine['nz'], nxy=fine['nxy'], dz_um=fine['dz_um'], crop=args.refocus_crop)
    cfg.update(grid=dict(GRID), fine=fine, mask_grid=meta['mask_grid'],
               recon=dict(high_pass=args.high_pass, low_pass=args.low_pass, decim=args.decim),
               heat=dict(sigma_xy_um=args.heat_sigma_xy_um, sigma_z_um=args.heat_sigma_z_um),
               tol=dict(xy=args.tol_xy_um, z=args.tol_z_um),
               optics=dict(dx_um=meta['dx_um'], wavelength_um=meta['wavelength_um'], index_medium=meta['index_medium']))
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
            pts = crop_points(item, args.crops_per_holo, rng, cfg, *field_of(item['meta']))
            X, org = refocus(V, pts)
            owner = s2.assign(item['bacts'][:, :3], pts, tol=cfg['tol'])
            T = s2.CropTargets(item, fine['nz'], fine['nxy'], heat=cfg['heat'])(org, owner, pts)
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

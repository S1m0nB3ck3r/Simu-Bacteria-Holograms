# -*- coding: utf-8 -*-

"""
Filename: train.py

Description:
Entraînement du U-Net 3D (unet3d.py) sur les jeux de generate_dataset.py.
  - Les hologrammes (float16, 2 Mo) sont chargés en mémoire ; à chaque pas, `holos_per_step`
    hologrammes sont reconstruits sur le GPU (holo_physics.Reconstructor, ~40 ms chacun) et
    `patches_per_holo` patchs en sont extraits : une fraction `pos_fraction` centrée sur une bactérie
    tirée au hasard (± demi-patch), le reste au hasard — le volume compte ~10⁻⁴ de voxels positifs.
  - Cibles rendues à la volée (masque du bâtonnet ou carte de chaleur, selon la loss).
  - Augmentations exactes : symétries x, y et échange x ↔ y (le problème est invariant).
  - fp16 automatique (GradScaler), AdamW, montée linéaire puis décroissance en cosinus.
  - Validation sur les volumes entiers : détection, appariement, F1 au meilleur seuil d'une grille ;
    le meilleur modèle (F1 de validation) est conservé.
Sortie : runs/<nom>/ (model_best.pt, config.json, history.json, train_summary.json).

Utilisation :
  python deep_learning_segmentation/train.py --name centernet --loss centernet --iters 4000

Author: Simon BECKER
Date: 2026-10-17

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
import copy
from unet3d import UNet3D, count_parameters, fold_batchnorm
from losses import LOSSES, TARGET_OF_LOSS
import detection as det

GRID = dict(z0_um=0.0, step_z_um=1.0, nb_planes=104, decim=2)
THRESHOLDS = {'heat': [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
              'mask': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}


def load_split(path, limit=None):
    files = sorted(glob.glob(os.path.join(path, 'sample_*.npz')))[:limit]
    items = []
    for f in files:
        d = np.load(f)
        items.append(dict(h=d['h'], bacts=d['bacts'], meta=json.loads(str(d['meta'])), file=f))
    return items


def make_reconstructor(meta, device='cuda'):
    z = GRID['z0_um'] + np.arange(GRID['nb_planes']) * GRID['step_z_um']
    return hp.Reconstructor(1024, meta['dx_um'], meta['wavelength_um'], meta['index_medium'], z, decim=GRID['decim'],
                            device=device)


def to_input(V, cfg):
    """Canaux d'entrée ('reim' : Re, Im ; 'abs' : |E|) puis normalisation ('fixed' : unités de bruit
    capteur ; 'localmad' : médiane / MAD par maille, comme le seuil local MAD d'HOLOTRACKER)."""
    if cfg['input'] == 'abs':
        V = torch.sqrt(V[0:1] ** 2 + V[1:2] ** 2)
    elif cfg['input'] != 'reim':
        raise ValueError(cfg['input'])
    if cfg.get('norm', 'fixed') == 'localmad':
        V = hp.local_mad_normalize(V)
    return V


def step_um(rec):
    s = rec.dx_um * rec.decim
    return (GRID['step_z_um'], s, s)


def sample_patches(V, bacts, cfg, rng, rec, target_kind):
    """Patchs (C, pz, py, px) et cibles (1 ou 7, pz, py, px) d'un volume (7 : chaleur + tenseur d'orientation)."""
    C, Z, Y, X = V.shape
    pz, py, px = cfg['patch']
    st = step_um(rec)
    xs, ts = [], []
    for _ in range(cfg['patches_per_holo']):
        if len(bacts) and rng.random() < cfg['pos_fraction']:
            b = bacts[rng.integers(len(bacts))]
            cz = int((b[2] - GRID['z0_um']) / st[0]) + rng.integers(-pz // 3, pz // 3 + 1)
            cy = int(b[1] / st[1]) + rng.integers(-py // 3, py // 3 + 1)
            cx = int(b[0] / st[2]) + rng.integers(-px // 3, px // 3 + 1)
            z0, y0, x0 = cz - pz // 2, cy - py // 2, cx - px // 2
        else:
            z0, y0, x0 = rng.integers(0, Z - pz + 1), rng.integers(0, Y - py + 1), rng.integers(0, X - px + 1)
        z0, y0, x0 = int(np.clip(z0, 0, Z - pz)), int(np.clip(y0, 0, Y - py)), int(np.clip(x0, 0, X - px))
        x = V[:, z0:z0 + pz, y0:y0 + py, x0:x0 + px]
        orient = cfg.get('orient_weight', 0) > 0
        want = (target_kind, 'orient') if orient else (target_kind,)
        r = hp.render_targets(bacts, (z0, y0, x0), (pz, py, px), st, GRID['z0_um'], cfg['length_um'],
                              cfg['thickness_um'], want=want, device=V.device)
        t = torch.cat([r[target_kind][None], r['orient']]) if orient else r[target_kind][None]
        # symétries exactes (le tenseur d'orientation change de signe / s'échange avec les axes)
        for axis, dims, ok in (('x', (-1,), rng.random() < 0.5), ('y', (-2,), rng.random() < 0.5)):
            if ok:
                x, t = x.flip(dims), t.flip(dims)
                if orient:
                    t = torch.cat([t[:1], hp.orient_flip(t[1:], axis)])
        if rng.random() < 0.5:
            x, t = x.transpose(-1, -2), t.transpose(-1, -2)
            if orient:
                t = torch.cat([t[:1], hp.orient_flip(t[1:], 'xy')])
        xs.append(x)
        ts.append(t)
    return xs, ts


def detect(logits, kind, thr, rec, with_axes=False):
    """logits (C, Z, Y, X). Retourne positions (n, 3), scores et, si with_axes, axes (n, 3) ou None."""
    st = step_um(rec)
    if kind == 'heat':
        pos, score, vox = det.peaks_from_heat(logits[0], thr, st, GRID['z0_um'], return_voxels=True)
        axes = None
        if with_axes and logits.shape[0] >= 7 and len(vox):
            t6 = logits[1:7, vox[:, 0], vox[:, 1], vox[:, 2]].T.float().cpu().numpy()
            axes = hp.axis_from_tensor(t6)
        return (pos, score, axes) if with_axes else (pos, score)
    pos, score, axes = det.objects_from_mask(logits[0], thr, st, GRID['z0_um'], return_axes=True)
    return (pos, score, axes) if with_axes else (pos, score)


def loss_total(loss_fn, logits, target, cfg):
    l = loss_fn(logits[:, :1], target[:, :1])
    if cfg.get('orient_weight', 0) > 0:
        w = (target[:, :1] >= hp.ORIENT_MIN).float()
        l_o = (w * (logits[:, 1:7] - target[:, 1:7]) ** 2).sum() / torch.clamp(w.sum() * 6, min=1.0)
        l = l + cfg['orient_weight'] * l_o
    return l


def inference_model(model):
    """Copie prête pour l'inférence : eval, BatchNorm repliée, format channels_last_3d."""
    m = fold_batchnorm(copy.deepcopy(model).eval())
    return m.to(memory_format=torch.channels_last_3d)


@torch.no_grad()
def validate(model, items, rec, cfg, kind, amp=True):
    train_model, model = model, inference_model(model)
    rows = {t: [] for t in THRESHOLDS[kind]}
    for it in items:
        V = to_input(rec(it['h'].astype(np.float32) + 1.0), cfg)
        logits = det.infer_volume(model, V, amp=amp)
        for t in THRESHOLDS[kind]:
            pos, _ = detect(logits, kind, t, rec)
            rows[t].append(det.evaluate(pos, it['bacts']))
    del model
    train_model.train()
    summ = {t: det.summarize(r) for t, r in rows.items()}
    best = max(summ, key=lambda t: summ[t]['f1'])
    return best, summ[best], summ


def lr_at(it, cfg):
    w = cfg['warmup']
    if it < w:
        return cfg['lr'] * (it + 1) / w
    q = (it - w) / max(1, cfg['iters'] - w)
    return cfg['lr'] * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * q)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--loss', choices=list(LOSSES), default='centernet')
    ap.add_argument('--input', choices=['reim', 'abs'], default='reim')
    ap.add_argument('--norm', choices=['fixed', 'localmad'], default='fixed')
    ap.add_argument('--z_step', type=float, default=1.0, help="pas Z de la grille (µm)")
    ap.add_argument('--depth_um', type=float, default=208.0, help="profondeur reconstruite (µm), 0 à depth_um")
    ap.add_argument('--orient_weight', type=float, default=0.0, help="> 0 : tête d'orientation (cible chaleur)")
    ap.add_argument('--train_dir', default=os.path.join(DL_DIR, 'data', 'seg_sim', 'train'))
    ap.add_argument('--val_dir', default=os.path.join(DL_DIR, 'data', 'seg_sim', 'val'))
    ap.add_argument('--iters', type=int, default=4000)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--warmup', type=int, default=200)
    ap.add_argument('--base', type=int, default=16)
    ap.add_argument('--depth', type=int, default=3)
    ap.add_argument('--patch', type=int, nargs=3, default=[48, 96, 96],
                    help="(Z, Y, X) ; 4 patchs de 64 × 128 × 128 débordent des 6 Go (2 fois plus lent)")
    ap.add_argument('--norm_layer', choices=['batch', 'group'], default='batch')
    ap.add_argument('--holos_per_step', type=int, default=2)
    ap.add_argument('--patches_per_holo', type=int, default=2)
    ap.add_argument('--pos_fraction', type=float, default=0.7)
    ap.add_argument('--eval_every', type=int, default=500)
    ap.add_argument('--val_limit', type=int, default=24)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=os.path.join(DL_DIR, 'runs'))
    args = ap.parse_args()
    cfg = vars(args)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = 'cuda'
    out = os.path.join(args.out, args.name)
    os.makedirs(out, exist_ok=True)
    kind = TARGET_OF_LOSS[args.loss]
    if args.orient_weight > 0 and kind != 'heat':
        raise ValueError("la tête d'orientation exige une cible 'heat'")
    # grille : depth_um de profondeur (104 µm par défaut), nombre de plans multiple de 2^depth
    mult = 2 ** args.depth
    GRID['step_z_um'] = args.z_step
    GRID['nb_planes'] = int(math.ceil(args.depth_um / args.z_step / mult) * mult)
    if args.z_step != 1.0 and args.patch[0] == 48:
        args.patch[0] = int(max(mult, round(48 / args.z_step / mult) * mult))

    t_load = time.time()
    train_items = load_split(args.train_dir)
    # validation : sous-ensemble équilibré en densité (les fichiers alternent N = 10, 50, 100, 200)
    val_items = load_split(args.val_dir, args.val_limit)
    meta = train_items[0]['meta']
    cfg.update(length_um=meta['length_um'], thickness_um=meta['thickness_um'], grid=GRID, target=kind)
    print(f"{len(train_items)} hologrammes d'entraînement, {len(val_items)} de validation "
          f"(chargés en {time.time() - t_load:.0f} s)", flush=True)
    rec = make_reconstructor(meta, dev)

    model = UNet3D(in_ch=2 if args.input == 'reim' else 1, out_ch=7 if args.orient_weight > 0 else 1,
                   base=args.base, depth=args.depth, norm=args.norm_layer).to(dev)
    cfg['n_params'] = count_parameters(model)
    print(f"U-Net 3D : base {args.base}, profondeur {args.depth}, {cfg['n_params'] / 1e6:.2f} M paramètres", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    loss_fn = LOSSES[args.loss]
    with open(os.path.join(out, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2)

    history, best_f1, losses = [], -1.0, []
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    t_data = t_net = 0.0
    # arrêt propre (GUI_train.py) : fichier STOP dans le dossier du run -> dernière validation, bilan, fin
    stop_file = os.path.join(out, 'STOP')
    if os.path.exists(stop_file):
        os.remove(stop_file)
    stop_now = False
    for it in range(args.iters):
        stop_now = os.path.exists(stop_file)
        for g in opt.param_groups:
            g['lr'] = lr_at(it, cfg)
        ta = time.perf_counter()
        xs, ts = [], []
        for k in rng.choice(len(train_items), args.holos_per_step, replace=False):
            item = train_items[k]
            V = to_input(rec(item['h'].astype(np.float32) + 1.0), cfg)
            x, t = sample_patches(V, item['bacts'], cfg, rng, rec, kind)
            xs += x
            ts += t
        xb, tb = torch.stack(xs), torch.stack(ts)
        torch.cuda.synchronize()
        tb_ = time.perf_counter()
        with torch.autocast('cuda', dtype=torch.float16):
            logits = model(xb)
        loss = loss_total(loss_fn, logits.float(), tb, cfg)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(opt)
        scaler.update()
        torch.cuda.synchronize()
        tc = time.perf_counter()
        t_data += tb_ - ta
        t_net += tc - tb_
        losses.append(float(loss))
        if (it + 1) % 50 == 0:
            el = time.time() - t0
            print(f"it {it + 1}/{args.iters}  loss {np.mean(losses[-50:]):.4f}  lr {lr_at(it, cfg):.2e}  "
                  f"{el / (it + 1):.2f} s/it (données {t_data / (it + 1):.2f}, réseau {t_net / (it + 1):.2f})", flush=True)
        if (it + 1) % args.eval_every == 0 or it + 1 == args.iters or stop_now:
            tv = time.time()
            thr, s, _ = validate(model, val_items, rec, cfg, kind)
            rec_h = dict(it=it + 1, train_loss=float(np.mean(losses[-args.eval_every:])), threshold=thr,
                         val_time_s=time.time() - tv, elapsed_s=time.time() - t0, **s)
            history.append(rec_h)
            print(f"  VALIDATION it {it + 1} : F1 {s['f1']:.3f} (rappel {s['recall']:.3f}, précision {s['precision']:.3f}, "
                  f"couchées {s['recall_lying']:.3f}, σz {s.get('sigma_z', float('nan')):.2f} µm) au seuil {thr} "
                  f"[{time.time() - tv:.0f} s]", flush=True)
            if s['f1'] > best_f1:
                best_f1 = s['f1']
                torch.save(dict(state=model.state_dict(), cfg=cfg, threshold=thr), os.path.join(out, 'model_best.pt'))
            with open(os.path.join(out, 'history.json'), 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2)
        if stop_now:
            os.remove(stop_file)
            print(f"  ARRÊT demandé (fichier STOP) à l'itération {it + 1}", flush=True)
            break
    n_done = it + 1
    summary = dict(name=args.name, loss=args.loss, input=args.input, norm=args.norm, z_step=args.z_step,
                   base=args.base, orient_weight=args.orient_weight, n_params=cfg['n_params'],
                   iters=n_done, stopped=stop_now, train_time_s=time.time() - t0, s_per_it=(time.time() - t0) / n_done,
                   data_s_per_it=t_data / n_done, net_s_per_it=t_net / n_done,
                   peak_gpu_mem_gb=torch.cuda.max_memory_allocated() / 2 ** 30, best_val_f1=best_f1,
                   best=max(history, key=lambda h: h['f1']))
    with open(os.path.join(out, 'train_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()

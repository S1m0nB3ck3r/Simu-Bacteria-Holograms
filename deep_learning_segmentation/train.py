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
import simu_dataset as sd

# =============================================================================
# RÉGLAGES PAR DÉFAUT — options de la ligne de commande et de GUI_train.py (onglet Étage 1) ; les valeurs
# utilisées sont enregistrées dans config.json et model_best.pt, et reprises par l'inférence
# =============================================================================
DECIM = 2                   # décimation latérale du volume grossier (1, 2 ou 4) : voxel XY = pixel objet × DECIM
HIGH_PASS = 15              # filtre passe-haut de la reconstruction, rayon en px de Fourier du capteur (0 : aucun)
LOW_PASS = 250              # filtre passe-bas (0 : aucun) ; sans perte si ≤ taille / (2 × DECIM) (256 pour 1024 et ×2)
Z_STEP_UM = 1.0             # pas Z du volume grossier (µm)
PATCH = (48, 96, 96)        # patchs d'entraînement (Z, Y, X), voxels du volume grossier
POS_FRACTION = 0.7          # part des patchs centrés près d'une bactérie (les autres : au hasard)
HEAT_SIGMA_XY_UM = 0.5      # cible : largeur (σ) des gaussiennes de la carte des centres en XY (µm)
HEAT_SIGMA_Z_UM = 1.0       # idem en Z (µm)
PEAK_BOX = (3, 5, 5)        # fenêtre des maxima locaux de la carte des centres (Z, Y, X), voxels grossiers
FILM_STATS_HOLOS = 32       # hologrammes (régulièrement espacés) pour les statistiques d'une normalisation « film »
MAD_MESH_UM = (13.0, 11.2)  # maille de la normalisation MAD locale (µm) : Z puis XY

GRID = dict(z0_um=0.0, step_z_um=Z_STEP_UM, nb_planes=104, decim=DECIM)   # grille en cours (main, ou modèle chargé)
RECON = dict(high_pass=HIGH_PASS, low_pass=LOW_PASS)                        # filtre en cours
PEAK = dict(box=PEAK_BOX)                                                   # fenêtre des maxima en cours
THRESHOLDS = {'heat': [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
              'mask': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}


def load_split(path, limit=None):
    """Dossier train/ ou val/ : jeu de GUI_generate_dataset.py (BMP + CSV, simu_dataset.py) ou .npz de generate_dataset.py."""
    if sd.is_dataset(path):
        return sd.load_split(path, limit, masks=False)
    files = sorted(glob.glob(os.path.join(path, 'sample_*.npz')))[:limit]
    items = []
    for f in files:
        d = np.load(f)
        items.append(dict(h=d['h'], bacts=d['bacts'], meta=json.loads(str(d['meta'])), file=f))
    return items


def apply_model_settings(cfg):
    """Grille, filtre de reconstruction et fenêtre des maxima d'un modèle d'étage 1 chargé (défauts : anciens modèles)."""
    GRID.update(cfg['grid'])
    r = cfg.get('recon', {})
    RECON.update(high_pass=r.get('high_pass', 15), low_pass=r.get('low_pass', 250))
    PEAK['box'] = tuple(cfg.get('peak_box', (3, 5, 5)))


def make_reconstructor(meta, device='cuda'):
    z = GRID['z0_um'] + np.arange(GRID['nb_planes']) * GRID['step_z_um']
    return hp.Reconstructor(meta.get('holo_size', 1024), meta['dx_um'], meta['wavelength_um'], meta['index_medium'], z,
                            decim=GRID['decim'], high_pass=RECON['high_pass'], low_pass=RECON['low_pass'], device=device)


NORMS = {
    'globalmad': "MAD globale par hologramme : (V − médiane) / (1,4826 × MAD) du volume de l'hologramme",
    'filmmad': "MAD globale du film : médiane et σ du film (médianes sur ses hologrammes), les mêmes pour toutes ses images",
    'globalstd': "moyenne et écart-type du volume de chaque hologramme : (V − moyenne) / écart-type",
    'filmstd': "moyenne et écart-type du film (médianes sur ses hologrammes), les mêmes pour toutes ses images",
    'localmad': "MAD locale : médiane et σ par maille, interpolées (seuil local MAD d'HOLOTRACKER)",
    'fixed': "échelle fixe (modèles du 2026-10-18) : σ calculé du bruit capteur 0,015 et de la bande du filtre",
}
FILM_NORMS = {'filmmad': 'mad', 'filmstd': 'std'}      # normalisations « film » -> statistique par hologramme


def to_input(V, cfg, film_stats=None):
    """
    Canaux d'entrée ('reim' : Re, Im ; 'abs' : |E|) puis normalisation (NORMS). Le volume V de Reconstructor
    est déjà divisé par l'échelle fixe ; les normalisations MAD le remettent en écarts-types robustes mesurés.
    'filmmad' / 'filmstd' exigent film_stats (film_stats_of sur les hologrammes du même film).
    """
    if cfg['input'] == 'abs':
        V = torch.sqrt(V[0:1] ** 2 + V[1:2] ** 2)
    elif cfg['input'] != 'reim':
        raise ValueError(cfg['input'])
    norm = cfg.get('norm', 'fixed')
    if norm == 'localmad':
        V = hp.local_mad_normalize(V, mesh=tuple(cfg.get('mad_mesh', (13, 32, 32))))
    elif norm == 'globalmad':
        V = hp.apply_global(V, hp.global_mad_stats(V))
    elif norm == 'globalstd':
        V = hp.apply_global(V, hp.global_std_stats(V))
    elif norm in FILM_NORMS:
        if film_stats is None:
            raise ValueError(f"normalisation '{norm}' : statistiques du film manquantes")
        V = hp.apply_global(V, film_stats)
    elif norm != 'fixed':
        raise ValueError(norm)
    return V


def film_stats_of(holos, rec, cfg, n=None):
    """Statistiques du film (FILM_NORMS) : au plus n hologrammes (fond ≈ 1) régulièrement espacés de la liste holos."""
    n = int(n or cfg.get('film_holos', FILM_STATS_HOLOS))
    idx = np.unique(np.linspace(0, len(holos) - 1, min(n, len(holos))).round().astype(int))

    def volumes():
        for i in idx:
            V = rec(np.asarray(holos[i], dtype=np.float32))
            yield torch.sqrt(V[0:1] ** 2 + V[1:2] ** 2) if cfg['input'] == 'abs' else V
    return hp.film_mad_stats(volumes(), FILM_NORMS[cfg['norm']])


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
                              cfg['thickness_um'], sigma_xy_um=cfg.get('heat_sigma_xy_um', 0.5),
                              sigma_z_um=cfg.get('heat_sigma_z_um', 1.0), want=want, device=V.device)
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
        pos, score, vox = det.peaks_from_heat(logits[0], thr, st, GRID['z0_um'], box=tuple(PEAK['box']),
                                              return_voxels=True)
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
def validate(model, items, rec, cfg, kind, amp=True, film_stats=None):
    train_model, model = model, inference_model(model)
    rows = {t: [] for t in THRESHOLDS[kind]}
    for it in items:
        V = to_input(rec(it['h'].astype(np.float32) + 1.0), cfg, film_stats)
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
    ap = argparse.ArgumentParser(description="Entraînement de l'étage 1 (positions) du U-Net 3D")
    ap.add_argument('--name', required=True, help="nom du run (sous-dossier de --out)")
    ap.add_argument('--loss', choices=list(LOSSES), default='centernet',
                    help="centernet : carte des centres (retenue) ; wmse, bce_dice, focal_tversky : masque grossier")
    ap.add_argument('--input', choices=['reim', 'abs'], default='reim', help="reim : Re et Im du champ ; abs : module")
    ap.add_argument('--norm', choices=list(NORMS), default='fixed', help="; ".join(f"{k} : {v}" for k, v in NORMS.items()))
    ap.add_argument('--mad_mesh_um', type=float, nargs=2, default=list(MAD_MESH_UM),
                    help="maille de la MAD locale (µm) : Z puis XY (HOLOTRACKER : 12,5 × 11,2)")
    ap.add_argument('--film_holos', type=int, default=FILM_STATS_HOLOS,
                    help="hologrammes utilisés pour les statistiques d'une normalisation « film »")
    ap.add_argument('--decim', type=int, choices=[1, 2, 4], default=DECIM,
                    help="décimation latérale du volume reconstruit (voxel XY = pixel objet × décimation) ; "
                         "l'étage 2 doit être entraîné avec la même")
    ap.add_argument('--high_pass', type=float, default=HIGH_PASS,
                    help="filtre passe-haut de la reconstruction (rayon en px de Fourier du capteur, 0 : aucun)")
    ap.add_argument('--low_pass', type=float, default=LOW_PASS,
                    help="filtre passe-bas (0 : aucun) ; sans perte à la décimation si ≤ taille / (2 × décimation)")
    ap.add_argument('--heat_sigma_xy_um', type=float, default=HEAT_SIGMA_XY_UM,
                    help="cible : largeur σ des gaussiennes de la carte des centres en XY (µm)")
    ap.add_argument('--heat_sigma_z_um', type=float, default=HEAT_SIGMA_Z_UM, help="idem en Z (µm)")
    ap.add_argument('--peak_box', type=int, nargs=3, default=list(PEAK_BOX),
                    help="fenêtre des maxima locaux de la carte des centres (Z, Y, X), voxels grossiers impairs")
    ap.add_argument('--z_step', type=float, default=Z_STEP_UM, help="pas Z de la grille (µm)")
    ap.add_argument('--z_min_um', type=float, default=None,
                    help="début du volume reconstruit (µm) ; défaut : distance volume-caméra du jeu (0 pour un jeu .npz)")
    ap.add_argument('--z_max_um', type=float, default=None,
                    help="fin du volume reconstruit (µm) ; défaut : distance + plans × pas du jeu (207 pour un jeu .npz)")
    ap.add_argument('--depth_um', type=float, default=None, help="ancien réglage : volume de 0 à depth_um")
    ap.add_argument('--orient_weight', type=float, default=0.0, help="> 0 : tête d'orientation (cible chaleur)")
    ap.add_argument('--train_dir', default=os.path.join(DL_DIR, 'datasets', 'dataset_sim_200um_N20-400', 'train'),
                    help="dossier train/ d'un jeu (GUI_generate_dataset.py ou .npz de generate_dataset.py)")
    ap.add_argument('--val_dir', default=os.path.join(DL_DIR, 'datasets', 'dataset_sim_200um_N20-400', 'val'),
                    help="dossier val/ du même jeu")
    ap.add_argument('--iters', type=int, default=4000, help="itérations (pas d'optimisation)")
    ap.add_argument('--lr', type=float, default=1e-3, help="taux d'apprentissage maximal (AdamW)")
    ap.add_argument('--warmup', type=int, default=200, help="itérations de montée linéaire du taux, puis cosinus")
    ap.add_argument('--base', type=int, default=16, help="canaux du premier niveau (doublés à chaque niveau)")
    ap.add_argument('--depth', type=int, default=3, help="niveaux du U-Net (sous-échantillonnages par 2)")
    ap.add_argument('--patch', type=int, nargs=3, default=list(PATCH),
                    help="patchs (Z, Y, X) en voxels grossiers ; 4 patchs de 64 × 128 × 128 débordent des 6 Go")
    ap.add_argument('--norm_layer', choices=['batch', 'group'], default='batch', help="couches de normalisation du réseau")
    ap.add_argument('--holos_per_step', type=int, default=2, help="hologrammes reconstruits par pas")
    ap.add_argument('--patches_per_holo', type=int, default=2, help="patchs tirés de chaque hologramme")
    ap.add_argument('--pos_fraction', type=float, default=POS_FRACTION, help="part des patchs centrés près d'une bactérie")
    ap.add_argument('--eval_every', type=int, default=500, help="validation toutes les N itérations")
    ap.add_argument('--val_limit', type=int, default=24, help="hologrammes de validation utilisés")
    ap.add_argument('--seed', type=int, default=0, help="graine (tirages des patchs, initialisation)")
    ap.add_argument('--out', default=os.path.join(DL_DIR, 'runs'), help="dossier des runs")
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
    # grille : de z_min à z_max (volume simulé du jeu par défaut), nombre de plans multiple de 2^depth
    mult = 2 ** args.depth
    GRID['decim'] = args.decim
    RECON.update(high_pass=args.high_pass, low_pass=args.low_pass)
    PEAK['box'] = tuple(args.peak_box)
    if any(b % 2 == 0 for b in args.peak_box):
        raise ValueError("--peak_box : valeurs impaires")
    if args.depth_um is not None and args.z_max_um is None:
        args.z_min_um, args.z_max_um = args.z_min_um or 0.0, args.depth_um - args.z_step
    args.z_min_um, args.z_max_um = sd.resolve_range(args.train_dir, args.z_min_um, args.z_max_um)
    GRID['step_z_um'] = args.z_step
    GRID['z0_um'], GRID['nb_planes'] = sd.grid_range(args.z_min_um, args.z_max_um, args.z_step, mult)
    print(f"volume reconstruit : {GRID['z0_um']:.1f}–{GRID['z0_um'] + (GRID['nb_planes'] - 1) * args.z_step:.1f} µm, "
          f"{GRID['nb_planes']} plans de {args.z_step} µm", flush=True)
    if args.z_step != 1.0 and args.patch[0] == 48:
        args.patch[0] = int(max(mult, round(48 / args.z_step / mult) * mult))

    t_load = time.time()
    train_items = load_split(args.train_dir)
    # validation : sous-ensemble équilibré en densité (les fichiers alternent N = 10, 50, 100, 200)
    val_items = load_split(args.val_dir, args.val_limit)
    meta = train_items[0]['meta']
    cfg.update(length_um=meta['length_um'], thickness_um=meta['thickness_um'], grid=GRID, target=kind,
               optics=dict(dx_um=meta['dx_um'], wavelength_um=meta['wavelength_um'], index_medium=meta['index_medium']),
               recon=dict(high_pass=args.high_pass, low_pass=args.low_pass, decim=args.decim))
    z_top = GRID['z0_um'] + (GRID['nb_planes'] - 1) * GRID['step_z_um']
    if meta.get('z_range_um') and (meta['z_range_um'][1] > z_top or meta['z_range_um'][0] < GRID['z0_um']):
        print(f"ATTENTION : bactéries entre {meta['z_range_um'][0]:.0f} et {meta['z_range_um'][1]:.0f} µm, volume "
              f"reconstruit entre {GRID['z0_um']:.0f} et {z_top:.0f} µm (--z_min_um, --z_max_um)", flush=True)
    print(f"{len(train_items)} hologrammes d'entraînement, {len(val_items)} de validation "
          f"(chargés en {time.time() - t_load:.0f} s)", flush=True)
    rec = make_reconstructor(meta, dev)
    film_train = film_val = None
    if args.norm == 'localmad':
        cfg['mad_mesh'] = list(hp.mesh_voxels(args.mad_mesh_um, GRID['step_z_um'], rec.dx_um * rec.decim))
        print(f"MAD locale : maille {cfg['mad_mesh']} voxels", flush=True)
    if args.norm in FILM_NORMS:    # chaque jeu est un « film » : ses propres statistiques
        film_train = film_stats_of([it['h'].astype(np.float32) + 1.0 for it in train_items], rec, cfg)
        film_val = film_stats_of([it['h'].astype(np.float32) + 1.0 for it in val_items], rec, cfg)
        cfg['film_stats'] = dict(train=film_train, val=film_val)
        print(f"statistiques du film ({args.norm}) : entraînement {film_train}, validation {film_val}", flush=True)

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
            V = to_input(rec(item['h'].astype(np.float32) + 1.0), cfg, film_train)
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
            thr, s, _ = validate(model, val_items, rec, cfg, kind, film_stats=film_val)
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

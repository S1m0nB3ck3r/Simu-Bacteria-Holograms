# -*- coding: utf-8 -*-

"""
Filename: generate_dataset.py

Description:
Jeux d'entraînement / validation / test du U-Net 3D : hologrammes de N bactéries simulées
(bâtonnets transparents 1,8 × 0,6 µm, BPM de Hybrid_REAL_SIMU_HOLOGRAMS/real_simulation.py), à
positions et orientations aléatoires (orientation isotrope), sur deux types de fond :
  - 'hybrid' : fond RÉEL nettoyé (sortie background/ de Hybrid_REAL_SIMU_HOLOGRAMS) : vrai bruit,
    vraies franges fixes, objets réels résiduels. Hologramme = fond normalisé × hologramme simulé ;
  - 'sim'    : fond plat + bruit capteur gaussien σ = 0,015 (valeur mesurée sur le film réel).
Les images de fond sont découpées dans le temps (pas de fuite entre jeux : deux images voisines du
film se ressemblent) : entraînement 0–69, validation 70–84, test 85–99.
Δn est tiré par hologramme dans [0,032 ; 0,060] (dispersion p10–p90 de l'amplitude réelle / simulée
mesurée sur le film : 0,68–1,30 autour de Δn = 0,046, recalibré le 2026-10-18 après la correction de la
longueur du bâtonnet ; avant : [0,030 ; 0,056] autour de 0,043).

Sortie : <out>/<split>/sample_XXXXX.npz avec
  h      : (1024, 1024) float16, hologramme normalisé − 1 (fond ≈ 0)
  bacts  : (N, 5) float32, x, y, z (µm, repère HOLOTRACKER), θ, φ (degrés) ; (N, 7) avec --size_jitter :
           + longueur et épaisseur passées au simulateur (µm)
  meta   : json (N, Δn, image de fond, type de fond)
  avec --save_masks, le masque d'instance du simulateur (real_simulation.bacteria_instance_masks : même
  noyau d'insertion, même grille que la BPM, chaque bactérie écrite seule ; un voxel peut appartenir à
  deux bactéries qui se croisent) : grille pixel dx × pas BPM (meta mask_grid), fraction occupée × up²
  (moyenne des blocs up × up du simulateur, entier uint8) :
  mask_origin (N, 3) int32 : (tranche k, ligne, colonne) du coin de la boîte ; tranche k <-> z = k · pas
  mask_shape  (N, 3) int32 : (K, R, C) ; mask_data : concaténation des boîtes (uint8, ordre C)
  Contrôle sur les --check_first premiers hologrammes de chaque jeu (relire les tranches coûte ~10 s) :
  l'union binaire des instances doit être identique au masque binaire effectivement traversé par la
  BPM (simulate_frame, plane_masks), à quelques voxels d'arrondi près (≤ 0,01 % : voxels pile sur la
  surface), sinon arrêt.
  (Déduire les instances du masque binaire et des positions, à l'entraînement, est exact pour des
  bactéries isolées mais attribue mal 8 à 12 % des voxels de bactéries qui se touchent, 2026-10-18.)

Utilisation :
  python deep_learning_segmentation/generate_dataset.py --background hybrid --out deep_learning_segmentation/data/hybrid
  python deep_learning_segmentation/generate_dataset.py --background sim    --out deep_learning_segmentation/data/sim

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
import cupy as cp
from PIL import Image
import tifffile

DL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(DL_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, 'Hybrid_REAL_SIMU_HOLOGRAMS'))
import real_simulation as rs

# fonds nettoyés produits par : python Hybrid_REAL_SIMU_HOLOGRAMS/real_simulation.py deep_learning_segmentation/experiences/config_backgrounds.json
# (nettoyage complet : trajectoires, détections hors trajectoire, CLEAN des restes ; TIFF float32 normalisés)
DEFAULT_BACKGROUND_DIR = os.path.join(DL_DIR, 'data', 'backgrounds')
SPLITS = {   # images de fond (indices du film) et nombre d'hologrammes
    'train': dict(frames=range(0, 70), n=480, density='loguniform'),
    'val': dict(frames=range(70, 85), n=48, density=[10, 50, 100, 200]),
    'test': dict(frames=range(85, 100), n=80, density=[10, 50, 100, 200]),
}
OPTICS = dict(n=1024, pixel_um=7.0, magnification=40.0, wavelength_um=0.66, index_medium=1.33)
BACTERIA = dict(length_um=1.8, thickness_um=0.6, delta_n=(0.032, 0.060), z_um=(5.0, 95.0), n_min=10, n_max=250)
SENSOR_NOISE = 0.015
# jeu de test des paires : 12 paires par hologramme (grille 4 × 3), écart imposé, fonds de test
PAIRS = dict(frames=range(85, 100), holos_per_value=4, grid=(4, 3), z_um=(15.0, 85.0),
             lateral_um=[1.0, 1.5, 2.0, 3.0, 4.0, 6.0], axial_um=[1.0, 2.0, 3.0, 4.0, 6.0, 8.0])


def load_backgrounds(bg_dir):
    """
    Fonds réels nettoyés, dans l'ordre du film. bg_dir : dossier de sortie de real_simulation.py, ou
    dossier qui en contient un seul. TIFF float32 (déjà normalisés) de préférence, sinon BMP / niveau
    de gris moyen. Retourne la liste des chemins et une fonction de lecture -> fond normalisé (≈ 1).
    """
    if not os.path.isdir(os.path.join(bg_dir, 'background')):
        subs = sorted(glob.glob(os.path.join(bg_dir, '*', 'background')))
        if not subs:
            raise FileNotFoundError(f"aucun dossier background/ dans {bg_dir}")
        bg_dir = os.path.dirname(subs[-1])
    paths = sorted(glob.glob(os.path.join(bg_dir, 'background', '*.tif')))
    if paths:
        return paths, lambda p: tifffile.imread(p).astype(np.float32)
    paths = sorted(glob.glob(os.path.join(bg_dir, 'background', '*.bmp')))
    if not paths:
        raise FileNotFoundError(f"aucun fond dans {bg_dir}/background")
    gray = float(tifffile.imread(os.path.join(bg_dir, 'mean', 'mean.tif')).mean())
    return paths, lambda p: np.asarray(Image.open(p)).astype(np.float32) / gray


def random_bacteria(n, rng, field_um, z_range=None, size_jitter=0.0, cluster_fraction=0.0, cluster_um=(0.6, 4.0)):
    """
    Positions uniformes ; cluster_fraction : part des bactéries replacées près d'une autre (distance entre
    centres uniforme dans cluster_um, direction isotrope) — sinon deux bactéries ne se touchent presque
    jamais (400 bactéries dans 179 × 179 × 200 µm : plus proche voisine à ~14 µm en moyenne).
    """
    zr = z_range or BACTERIA['z_um']
    x = rng.uniform(0, field_um, n)
    y = rng.uniform(0, field_um, n)
    z = rng.uniform(*zr, n)
    n_cl = int(round(cluster_fraction * n))
    if n_cl and n > 1:
        moved = rng.choice(n, n_cl, replace=False)
        anchors = np.setdiff1d(np.arange(n), moved)
        for i in moved:
            j = rng.choice(anchors)
            d = rng.normal(size=3)
            d *= rng.uniform(*cluster_um) / np.linalg.norm(d)
            x[i] = np.clip(x[j] + d[0], 0, field_um)
            y[i] = np.clip(y[j] + d[1], 0, field_um)
            z[i] = np.clip(z[j] + d[2], *zr)
    theta = rng.uniform(0, 360, n)
    phi = np.degrees(np.arccos(rng.random(n)))     # isotrope sur la demi-sphère
    cols = [x, y, z, theta, phi]
    if size_jitter > 0:      # longueur et épaisseur indépendantes, ± size_jitter (relatif) autour du nominal
        cols += [BACTERIA['length_um'] * rng.uniform(1 - size_jitter, 1 + size_jitter, n),
                 BACTERIA['thickness_um'] * rng.uniform(1 - size_jitter, 1 + size_jitter, n)]
    return np.stack(cols, axis=1).astype(np.float32)


def check_masks(inst, planes, N):
    """Nombre de voxels où l'union binaire des instances diffère du masque traversé par la BPM."""
    acc = {}
    for (k0, r0, c0), m in inst:
        for kk in range(m.shape[0]):
            if (m[kk] > 0).any():
                a = acc.setdefault(k0 + kk, np.zeros((N, N), bool))
                a[r0:r0 + m.shape[1], c0:c0 + m.shape[2]] |= m[kk] > 0
    return sum(int((acc.get(k, np.zeros((N, N), bool)) != (planes.get(k, np.zeros((N, N))) > 0)).sum())
               for k in set(acc) | set(planes))



def pair_bacteria(kind, sep, rng, field_um):
    """12 paires : bactéries 2k et 2k+1 séparées de sep (µm) en XY ('lateral', même z) ou en Z ('axial')."""
    gx, gy = PAIRS['grid']
    out = []
    for i in range(gx):
        for j in range(gy):
            cx = (i + 0.5) * field_um / gx + rng.uniform(-4, 4)
            cy = (j + 0.5) * field_um / gy + rng.uniform(-4, 4)
            cz = rng.uniform(*PAIRS['z_um'])
            if kind == 'lateral':
                a = rng.uniform(0, 2 * np.pi)
                d = np.array([np.cos(a), np.sin(a), 0.0]) * sep / 2
            else:
                d = np.array([0.0, 0.0, sep / 2])
            for sgn in (-1, 1):
                out.append([cx + sgn * d[0], cy + sgn * d[1], cz + sgn * d[2], rng.uniform(0, 360),
                            np.degrees(np.arccos(rng.random()))])
    return np.array(out, np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--background', choices=['sim', 'hybrid'], default='sim')
    ap.add_argument('--background_dir', default=DEFAULT_BACKGROUND_DIR)
    ap.add_argument('--out', required=True)
    ap.add_argument('--splits', nargs='+', default=list(SPLITS), help="train val test, ou pairs")
    ap.add_argument('--scale', type=float, default=1.0, help="facteur sur le nombre d'hologrammes")
    ap.add_argument('--seed', type=int, default=0)
    # plages de profondeur et de densité : remplacent les valeurs par défaut
    ap.add_argument('--z_range', type=float, nargs=2, default=None, help="profondeur des bactéries (µm)")
    ap.add_argument('--n_range', type=int, nargs=2, default=None, help="N tiré en log-uniforme (entraînement)")
    ap.add_argument('--n_list', type=int, nargs='+', default=None, help="N en alternance (validation, test)")
    ap.add_argument('--counts', type=int, nargs=3, default=None, help="hologrammes train val test")
    # segmentation (masques d'instance du simulateur)
    ap.add_argument('--size_jitter', type=float, default=0.0, help="taille ± (relatif), p. ex. 0.1")
    ap.add_argument('--sim_step', type=float, default=None, help="pas BPM (µm) ; aussi pas Z du masque")
    ap.add_argument('--save_masks', action='store_true')
    ap.add_argument('--check_first', type=int, default=3, help="hologrammes contrôlés par jeu (masques)")
    ap.add_argument('--cluster_fraction', type=float, default=0.0, help="part des bactéries placées près d'une autre")
    args = ap.parse_args()
    if args.counts:
        for k, c in zip(('train', 'val', 'test'), args.counts):
            SPLITS[k] = dict(SPLITS[k], n=c)
    if args.n_list:
        for k in ('val', 'test'):
            SPLITS[k] = dict(SPLITS[k], density=list(args.n_list))
    if args.n_range:
        BACTERIA['n_min'], BACTERIA['n_max'] = args.n_range

    N = OPTICS['n']
    dx = OPTICS['pixel_um'] * 1e-6 / OPTICS['magnification']
    lam = OPTICS['wavelength_um'] * 1e-6
    field_um = N * dx * 1e6
    prop = rs.Propagator(N, 256, dx, lam / OPTICS['index_medium'])
    sc = dict(rs.DEFAULT_CONFIG['simulation'], length_um=BACTERIA['length_um'], thickness_um=BACTERIA['thickness_um'])
    if args.sim_step:
        sc['step_um'] = args.sim_step
    if args.background == 'hybrid':
        bg_paths, read_bg = load_backgrounds(args.background_dir)
    rng = np.random.default_rng(args.seed)

    for split in args.splits:
        if split == 'pairs':
            plan = [(k, v) for k in ('lateral', 'axial') for v in PAIRS[f'{k}_um'] for _ in range(PAIRS['holos_per_value'])]
            spec = dict(frames=PAIRS['frames'])
        else:
            spec = SPLITS[split]
        out = os.path.join(args.out, split)
        os.makedirs(out, exist_ok=True)
        n_holo = len(plan) if split == 'pairs' else int(round(spec['n'] * args.scale))
        frames = list(spec['frames'])
        t0 = time.time()
        for i in range(n_holo):
            if os.path.exists(os.path.join(args.out, 'STOP')):       # arrêt propre (GUI_train.py)
                os.remove(os.path.join(args.out, 'STOP'))
                print(f"ARRÊT demandé (fichier STOP) : {split} {i}/{n_holo}", flush=True)
                return
            pair = None
            if split == 'pairs':
                pair = plan[i]
                bacts = pair_bacteria(pair[0], pair[1], rng, field_um)
                if args.size_jitter > 0:
                    sz = np.stack([BACTERIA['length_um'] * rng.uniform(1 - args.size_jitter, 1 + args.size_jitter, len(bacts)),
                                   BACTERIA['thickness_um'] * rng.uniform(1 - args.size_jitter, 1 + args.size_jitter, len(bacts))], 1)
                    bacts = np.concatenate([bacts, sz.astype(np.float32)], axis=1)
                nb = len(bacts)
            else:
                if spec['density'] == 'loguniform':
                    nb = int(round(np.exp(rng.uniform(np.log(BACTERIA['n_min']), np.log(BACTERIA['n_max'])))))
                else:
                    nb = spec['density'][i % len(spec['density'])]
                bacts = random_bacteria(nb, rng, field_um, args.z_range, args.size_jitter, args.cluster_fraction)
            dn = float(rng.uniform(*BACTERIA['delta_n']))
            frame = frames[i % len(frames)]
            blist = [dict(x_um=float(b[0]), y_um=float(b[1]), z_um=float(b[2]), theta_deg=float(b[3]),
                          phi_deg=float(b[4])) for b in bacts]
            if bacts.shape[1] >= 7:
                for d, b in zip(blist, bacts):
                    d.update(length_um=float(b[5]), thickness_um=float(b[6]))
            planes = {} if (args.save_masks and i < args.check_first) else None
            sim = rs.simulate_frame(blist, prop, dict(sc, delta_n=dn), dx, lam, plane_masks=planes)
            sim = cp.asnumpy(sim)
            extra = {}
            if args.save_masks:
                inst = rs.bacteria_instance_masks(blist, sc, dx, N)
                if planes is not None:
                    bad = check_masks(inst, planes, N)
                    nvox = sum(int((p > 0).sum()) for p in planes.values())
                    print(f"  contrôle {split} {i} : union des instances / masque simulé, {bad} voxels différents "
                          f"sur {nvox}", flush=True)
                    # quelques voxels pile sur la surface du bâtonnet peuvent basculer : positions calculées depuis
                    # l'origine du volume (simulation) ou de la boîte de la bactérie (instance), arrondis différents
                    if bad > max(2, 1e-4 * nvox):
                        raise RuntimeError(f"{split} {i} : union des instances ≠ masque simulé ({bad} voxels)")
                up2 = int(sc['upscale_factor']) ** 2
                extra = dict(mask_origin=np.array([o for o, _ in inst], np.int32).reshape(-1, 3),
                             mask_shape=np.array([m.shape for _, m in inst], np.int32).reshape(-1, 3),
                             mask_data=np.concatenate([np.rint(m.astype(np.float32) * up2).astype(np.uint8).ravel()
                                                       for _, m in inst] + [np.zeros(0, np.uint8)]))
            if args.background == 'hybrid':
                bg = read_bg(bg_paths[frame])
                holo = bg * sim
            else:
                holo = sim + rng.normal(0.0, SENSOR_NOISE, sim.shape).astype(np.float32)
            meta = dict(n=nb, delta_n=dn, background=args.background,
                        pair_type=pair[0] if pair else None, pair_sep_um=pair[1] if pair else None,
                        z_range_um=list(args.z_range or BACTERIA['z_um']),
                        background_frame=frame if args.background == 'hybrid' else None,
                        length_um=BACTERIA['length_um'], thickness_um=BACTERIA['thickness_um'],
                        size_jitter=args.size_jitter, sim_step_um=sc['step_um'], cluster_fraction=args.cluster_fraction,
                        mask_grid=dict(dz_um=sc['step_um'], dx_um=dx * 1e6, frac_levels=int(sc['upscale_factor']) ** 2)
                        if args.save_masks else None,
                        dx_um=dx * 1e6, wavelength_um=OPTICS['wavelength_um'], index_medium=OPTICS['index_medium'])
            np.savez(os.path.join(out, f"sample_{i:05d}.npz"), h=(holo - 1.0).astype(np.float16), bacts=bacts,
                     meta=json.dumps(meta), **extra)
            if (i + 1) % 20 == 0 or i + 1 == n_holo:
                el = time.time() - t0
                print(f"[{split}] {i + 1}/{n_holo}  ({el / (i + 1):.2f} s/hologramme)", flush=True)
            cp.get_default_memory_pool().free_all_blocks()


if __name__ == '__main__':
    main()

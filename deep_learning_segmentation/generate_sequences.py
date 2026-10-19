# -*- coding: utf-8 -*-

"""
Filename: generate_sequences.py

Description:
Films holographiques 100 % simulés de bactéries en mouvement, dynamique calée sur les trajectoires du film
réel (stats/stat_report_trajectories.py, TRAJECTORY_REPORT/). Cadence : 100 Hz par convention, identique pour
le film réel et le film simulé ; tous les paramètres sont exprimés PAR IMAGE, la vraie cadence n'intervient
donc pas.
Modèle (valeurs par défaut = mesures du 2026-10-18 sur 57 trajectoires) :
  - nageuses (40 %) : vitesse et inclinaison initiale propres à chaque bactérie, tirées ensemble parmi celles
    des nageuses du film (inclinaison = arctan(v_xy / v_z) du MSD : les vraies nagent surtout à l'horizontale,
    ~70° de l'axe Z), azimut au hasard ; diffusion de rotation D_r = 0,0052 rad²/image (cos θ = 0,73 après 30 images),
    pas de demi-tour (les virages > 120° relevés sur le film, 0,012 par image, sont incompatibles avec cette
    persistance : artefacts de localisation), diffusion de translation 0,01 µm²/image ; le bâtonnet est
    aligné sur la direction de nage ;
  - non nageuses (60 %) : diffusion de translation 0,009 µm²/image (x, y, z), orientation aléatoire qui
    diffuse (D_r 0,01 rad²/image) ;
  - bords : en x, y, une bactérie qui sort du champ rentre du côté opposé sous une NOUVELLE identité (une
    sort, une autre entre) ; en z, réflexion sur les parois de la chambre ; tailles ± size_jitter, propres à
    chaque bactérie ; Δn tiré par film.
Chaque image est simulée par la BPM du simulateur hybride (real_simulation.simulate_frame), sur fond plat +
bruit capteur σ = 0,015, avec le masque d'instance du simulateur (même format que generate_dataset.py
--save_masks) et l'identité de chaque bactérie.

Sortie : <out>/<split>/seq_XXX/frame_YYYY.npz (h, bacts (N, 7), ids (N,), meta, mask_origin / mask_shape /
mask_data), <out>/<split>/seq_XXX/trajectories.csv (colonnes d'HOLOTRACKER + identité, population, vitesse).

Utilisation :
  python deep_learning_segmentation/generate_sequences.py --out deep_learning_segmentation/data/seq

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
import cupy as cp

DL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(DL_DIR)
sys.path.insert(0, DL_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, 'Hybrid_REAL_SIMU_HOLOGRAMS'))
import real_simulation as rs
import generate_dataset as gd

TRAJ_REPORT = os.path.join(REPO_DIR, 'TRAJECTORY_REPORT', 'per_trajectory.csv')
DYNAMICS = dict(
    fps=100.0,                      # convention : même cadence pour le film réel et le film simulé
    swimmer_fraction=0.40,
    swim_speeds_um_per_frame=None,  # None : vitesses des nageuses lues dans TRAJECTORY_REPORT (sinon liste)
    swim_rot_diff=0.0052,           # rad²/image
    swim_reversal_prob=0.0,         # par image (0,012 mesuré, mais incompatible avec la persistance : artefact)
    swim_trans_diff=0.01,           # µm²/image
    passive_trans_diff=0.009,       # µm²/image
    passive_rot_diff=0.01,          # rad²/image
)
SPLITS = {'train': dict(n_seq=8, n_frames=40, n=[50, 100, 200, 300, 400], seed=501),
          'val': dict(n_seq=2, n_frames=20, n=[100, 300], seed=502),
          'test': dict(n_seq=4, n_frames=30, n=[50, 100, 200, 400], seed=503)}


def swim_speeds(dyn):
    """(vitesse en µm/image, inclinaison φ en radians) des nageuses du film, une ligne par trajectoire."""
    if dyn['swim_speeds_um_per_frame']:
        v = np.asarray(dyn['swim_speeds_um_per_frame'], float)
        return np.stack([v, np.arccos(np.random.default_rng(0).random(len(v)))], 1)
    d = pd.read_csv(TRAJ_REPORT)
    sw = d[d.v_xy * 10 > 2 * np.sqrt(4 * d.D_xy * 10)]       # même critère que le rapport
    return np.stack([np.hypot(sw.v_xy, sw.v_z).to_numpy(), np.arctan2(sw.v_xy, sw.v_z).to_numpy()], 1)


def random_unit(rng, n):
    u = rng.normal(size=(n, 3))
    return u / np.linalg.norm(u, axis=1, keepdims=True)


def rotate_diffuse(u, D, rng):
    """Marche aléatoire sur la sphère : u + √(2D)·ξ⊥, renormalisé."""
    xi = rng.normal(size=u.shape) * np.sqrt(2 * D)
    xi -= (xi * u).sum(axis=1, keepdims=True) * u
    u = u + xi
    return u / np.linalg.norm(u, axis=1, keepdims=True)


def reflect(p, lo, hi):
    p = np.where(p < lo, 2 * lo - p, p)
    return np.where(p > hi, 2 * hi - p, p)


def axis_to_angles(u):
    """Axe -> (θ, φ) en degrés, φ ∈ [0, 90] (un bâtonnet n'a pas de sens)."""
    u = np.where(u[:, 2:3] < 0, -u, u)
    phi = np.degrees(np.arccos(np.clip(u[:, 2], -1, 1)))
    theta = np.degrees(np.arctan2(u[:, 1], u[:, 0])) % 360
    return theta, phi


def simulate_tracks(n, n_frames, rng, dyn, field_um, z_range, speeds):
    """Positions (T, n, 3), axes (T, n, 3), identités (T, n), population, vitesse."""
    lo = np.array([0.0, 0.0, z_range[0]])
    hi = np.array([field_um, field_um, z_range[1]])
    swim = rng.random(n) < dyn['swimmer_fraction']
    pick = speeds[rng.integers(0, len(speeds), n)]
    v = np.where(swim, pick[:, 0], 0.0)
    p = lo + rng.random((n, 3)) * (hi - lo)
    u = random_unit(rng, n)
    ph, az = pick[:, 1], rng.uniform(0, 2 * np.pi, n)
    up = np.stack([np.sin(ph) * np.cos(az), np.sin(ph) * np.sin(az), np.cos(ph) * rng.choice([-1, 1], n)], 1)
    u = np.where(swim[:, None], up, u)
    P, U = np.empty((n_frames, n, 3)), np.empty((n_frames, n, 3))
    ids = np.arange(n)
    I = np.empty((n_frames, n), int)
    next_id = n
    for t in range(n_frames):
        P[t], U[t], I[t] = p, u, ids
        rev = swim & (rng.random(n) < dyn['swim_reversal_prob'])
        u[rev] = -u[rev]
        D_r = np.where(swim, dyn['swim_rot_diff'], dyn['passive_rot_diff'])[:, None]
        u = rotate_diffuse(u, D_r, rng)
        D_t = np.where(swim, dyn['swim_trans_diff'], dyn['passive_trans_diff'])[:, None]
        step = v[:, None] * u + rng.normal(size=(n, 3)) * np.sqrt(2 * D_t)
        p = p + step
        # x, y : sortie du champ -> entrée du côté opposé, nouvelle identité
        left = ((p[:, :2] < lo[:2]) | (p[:, :2] > hi[:2])).any(axis=1)
        p[:, :2] = lo[:2] + np.mod(p[:, :2] - lo[:2], hi[:2] - lo[:2])
        ids = ids.copy()
        ids[left] = next_id + np.arange(left.sum())
        next_id += int(left.sum())
        # z : réflexion sur les parois ; une nageuse repart avec la composante z inversée
        below, above = p[:, 2] < lo[2], p[:, 2] > hi[2]
        p[:, 2] = np.clip(reflect(p[:, 2], lo[2], hi[2]), lo[2], hi[2])
        u[below | above, 2] *= -1
    return P, U, I, swim, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--splits', nargs='+', default=list(SPLITS))
    ap.add_argument('--z_range', type=float, nargs=2, default=[2.0, 200.0])
    ap.add_argument('--size_jitter', type=float, default=0.1)
    ap.add_argument('--sim_step', type=float, default=0.25)
    ap.add_argument('--n_seq', type=int, default=None, help="remplace le nombre de films de chaque jeu")
    ap.add_argument('--n_frames', type=int, default=None, help="remplace le nombre d'images par film")
    ap.add_argument('--n_list', type=int, nargs='+', default=None, help="remplace les densités (alternées)")
    ap.add_argument('--no_masks', action='store_true')
    args = ap.parse_args()
    N = gd.OPTICS['n']
    dx = gd.OPTICS['pixel_um'] * 1e-6 / gd.OPTICS['magnification']
    lam = gd.OPTICS['wavelength_um'] * 1e-6
    field_um = N * dx * 1e6
    prop = rs.Propagator(N, 256, dx, lam / gd.OPTICS['index_medium'])
    sc = dict(rs.DEFAULT_CONFIG['simulation'], length_um=gd.BACTERIA['length_um'],
              thickness_um=gd.BACTERIA['thickness_um'], step_um=args.sim_step)
    up2 = int(sc['upscale_factor']) ** 2
    speeds = swim_speeds(DYNAMICS)
    for split in args.splits:
        spec = dict(SPLITS[split])
        if args.n_seq:
            spec['n_seq'] = args.n_seq
        if args.n_frames:
            spec['n_frames'] = args.n_frames
        if args.n_list:
            spec['n'] = args.n_list
        rng = np.random.default_rng(spec['seed'])
        t0, done = time.time(), 0
        for s in range(spec['n_seq']):
            n = spec['n'][s % len(spec['n'])]
            P, U, I, swim, v = simulate_tracks(n, spec['n_frames'], rng, DYNAMICS, field_um, args.z_range, speeds)
            L = gd.BACTERIA['length_um'] * rng.uniform(1 - args.size_jitter, 1 + args.size_jitter, n)
            T = gd.BACTERIA['thickness_um'] * rng.uniform(1 - args.size_jitter, 1 + args.size_jitter, n)
            dn = float(rng.uniform(*gd.BACTERIA['delta_n']))
            sd = os.path.join(args.out, split, f"seq_{s:03d}")
            os.makedirs(sd, exist_ok=True)
            rows = []
            for f in range(spec['n_frames']):
                theta, phi = axis_to_angles(U[f])
                bacts = np.stack([P[f, :, 0], P[f, :, 1], P[f, :, 2], theta, phi, L, T], 1).astype(np.float32)
                blist = [dict(x_um=float(b[0]), y_um=float(b[1]), z_um=float(b[2]), theta_deg=float(b[3]),
                              phi_deg=float(b[4]), length_um=float(b[5]), thickness_um=float(b[6])) for b in bacts]
                sim = cp.asnumpy(rs.simulate_frame(blist, prop, dict(sc, delta_n=dn), dx, lam))
                holo = sim + rng.normal(0.0, gd.SENSOR_NOISE, sim.shape).astype(np.float32)
                extra = {}
                if not args.no_masks:
                    inst = rs.bacteria_instance_masks(blist, sc, dx, N)
                    extra = dict(mask_origin=np.array([o for o, _ in inst], np.int32).reshape(-1, 3),
                                 mask_shape=np.array([m.shape for _, m in inst], np.int32).reshape(-1, 3),
                                 mask_data=np.concatenate([np.rint(m.astype(np.float32) * up2).astype(np.uint8).ravel()
                                                           for _, m in inst] + [np.zeros(0, np.uint8)]))
                meta = dict(n=n, delta_n=dn, background='sim', sequence=s, frame=f, fps=DYNAMICS['fps'],
                            dynamics=DYNAMICS | {'swim_speeds_um_per_frame': 'TRAJECTORY_REPORT'},
                            z_range_um=list(args.z_range), length_um=gd.BACTERIA['length_um'],
                            thickness_um=gd.BACTERIA['thickness_um'], size_jitter=args.size_jitter,
                            sim_step_um=sc['step_um'], dx_um=dx * 1e6, wavelength_um=gd.OPTICS['wavelength_um'],
                            index_medium=gd.OPTICS['index_medium'],
                            mask_grid=None if args.no_masks else dict(dz_um=sc['step_um'], dx_um=dx * 1e6, frac_levels=up2))
                np.savez(os.path.join(sd, f"frame_{f:04d}.npz"), h=(holo - 1.0).astype(np.float16), bacts=bacts,
                         ids=I[f].astype(np.int32), meta=json.dumps(meta), **extra)
                for i in range(n):
                    rows.append({'HOLOGRAM NUMBER': f + 1, 'X POSITION (m)': P[f, i, 0] * 1e-6,
                                 'Y POSITION (m)': P[f, i, 1] * 1e-6, 'Z POSITION (m)': P[f, i, 2] * 1e-6,
                                 'TRAJECTORY NUMBER': int(I[f, i]), 'swimmer': bool(swim[i]), 'speed_um_per_frame': float(v[i]),
                                 'theta_deg': float(theta[i]), 'phi_deg': float(phi[i])})
                done += 1
                cp.get_default_memory_pool().free_all_blocks()
            pd.DataFrame(rows).to_csv(os.path.join(sd, 'trajectories.csv'), index=False)
            el = time.time() - t0
            print(f"[{split}] film {s + 1}/{spec['n_seq']} : N = {n}, {spec['n_frames']} images, "
                  f"{int(swim.sum())} nageuses ({el / done:.2f} s/image)", flush=True)


if __name__ == '__main__':
    main()

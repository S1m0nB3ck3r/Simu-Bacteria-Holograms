# -*- coding: utf-8 -*-

"""
Filename: stat_report_trajectories.py

Description:
Dynamique des bactéries du film réel, à partir des trajectoires HOLOTRACKER (TRAJECTORIES_*.csv), pour
générer ensuite des films 100 % simulés en mouvement. Unités : µm et image (cadence absente du fichier).
Les positions sont bruitées (localisation ~0,1–0,2 µm en XY, davantage en Z) : les estimateurs évitent la
vitesse image par image, que ce bruit gonfle.
  - MSD(τ) par trajectoire, en XY et en Z séparément, ajusté par MSD = v²τ² + 2dDτ + 2dσ² (d = 2 en XY,
    1 en Z ; τ ≤ 10 images) : vitesse de nage dirigée v, diffusion D, bruit de localisation σ ;
  - vitesse lissée (régression linéaire sur ± W images) : distribution, part de mouvement en Z ;
  - persistance : autocorrélation de la direction de la vitesse lissée (cos de l'angle entre t et t + τ) ;
    temps de persistance = τ où elle tombe à 1/e ;
  - angles de virage entre déplacements successifs sur k images (k = 3, 5) : forme de la distribution
    (régime « course et culbute » si bimodale) ;
  - populations : bactéries nageuses / non nageuses (v du MSD) ; répartition en z ; distance à la plus
    proche voisine parmi les bactéries suivies.
Sortie : TRAJECTORY_REPORT/ (per_trajectory.csv, summary.json, figures PNG) ; le rapport est écrit à part.

Utilisation :
  python stats/stat_report_trajectories.py ["REAL HOLO FILM/bacterias/TRAJECTORIES_20261016_173113.csv"] [dossier de sortie]

Author: Simon BECKER
Date: 2026-10-18

License:
GNU General Public License v3.0
"""

import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CSV = os.path.join(REPO, 'REAL HOLO FILM', 'bacterias', 'TRAJECTORIES_20261016_173113.csv')
OUT = os.path.join(REPO, 'TRAJECTORY_REPORT')
TAU_MAX = 10
SMOOTH_W = 3
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']


def load(csv):
    d = pd.read_csv(csv)
    d = d[d['TRAJECTORY NUMBER'] >= 0].copy()
    d['t'] = d['HOLOGRAM NUMBER'].astype(int)
    d['x'], d['y'], d['z'] = d['X POSITION (m)'] * 1e6, d['Y POSITION (m)'] * 1e6, d['Z POSITION (m)'] * 1e6
    d['traj'] = d['TRAJECTORY NUMBER'].astype(int)
    return d.sort_values(['traj', 't'])


def msd(P, t, taus):
    """MSD par composante (n_tau, 3) à partir de positions P (n, 3) aux temps entiers t (trous tolérés)."""
    idx = {int(a): i for i, a in enumerate(t)}
    out = np.full((len(taus), 3), np.nan)
    cnt = np.zeros(len(taus), int)
    for k, tau in enumerate(taus):
        pairs = [(idx[a], idx[a + tau]) for a in idx if a + tau in idx]
        if len(pairs) >= 5:
            i, j = np.array(pairs).T
            out[k] = ((P[j] - P[i]) ** 2).mean(axis=0)
            cnt[k] = len(pairs)
    return out, cnt


def fit_msd(taus, m, d):
    """Moindres carrés MSD = a τ² + b τ + c (a, b, c ≥ 0) -> v = √a, D = b / (2d), σ = √(c / 2d)."""
    ok = np.isfinite(m)
    if ok.sum() < 4:
        return np.nan, np.nan, np.nan
    A = np.stack([taus[ok] ** 2, taus[ok], np.ones(ok.sum())], 1)
    from scipy.optimize import nnls
    coef, _ = nnls(A, m[ok])
    a, b, c = coef
    return float(np.sqrt(a)), float(b / (2 * d)), float(np.sqrt(c / (2 * d)))


def smooth_velocity(P, t, w=SMOOTH_W):
    """Vitesse (n, 3) µm/image par régression linéaire sur ± w images (NaN si trop peu de points)."""
    V = np.full_like(P, np.nan)
    for i, ti in enumerate(t):
        sel = np.abs(t - ti) <= w
        if sel.sum() >= w + 1:
            tt = t[sel] - t[sel].mean()
            den = (tt ** 2).sum()
            if den > 0:
                V[i] = (tt[:, None] * (P[sel] - P[sel].mean(axis=0))).sum(axis=0) / den
    return V


def main():
    global OUT
    csv = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    if len(sys.argv) > 2:
        OUT = sys.argv[2]
    os.makedirs(OUT, exist_ok=True)
    d = load(csv)
    taus = np.arange(1, TAU_MAX + 1, dtype=float)
    rows, msd_xy_all, msd_z_all, autoc, turns = [], [], [], {k: [] for k in range(1, 31)}, {3: [], 5: []}
    vel_all = []
    for tr, g in d.groupby('traj'):
        P = g[['x', 'y', 'z']].to_numpy()
        t = g.t.to_numpy()
        if len(g) < 15:
            continue
        m, _ = msd(P, t, taus.astype(int))
        mxy, mz = m[:, 0] + m[:, 1], m[:, 2]
        msd_xy_all.append(mxy)
        msd_z_all.append(mz)
        vxy, Dxy, sxy = fit_msd(taus, mxy, 2)
        vz, Dz, sz = fit_msd(taus, mz, 1)
        V = smooth_velocity(P, t)
        sp = np.linalg.norm(V, axis=1)
        ok = np.isfinite(sp)
        vel_all.append(V[ok])
        # persistance : cos de l'angle entre les vitesses lissées à t et t + τ
        U = V / np.maximum(sp[:, None], 1e-9)
        idx = {int(a): i for i, a in enumerate(t)}
        for k in autoc:
            c = [float(U[idx[a]] @ U[idx[a + k]]) for a in idx if a + k in idx and ok[idx[a]] and ok[idx[a + k]]]
            if c:
                autoc[k].append(np.mean(c))
        for k in turns:
            # déplacements sur k images, angle entre deux déplacements consécutifs
            seg = [P[idx[a + k]] - P[idx[a]] for a in sorted(idx) if a + k in idx and (a - a % k) == a]
            for s1, s2 in zip(seg[:-1], seg[1:]):
                n1, n2 = np.linalg.norm(s1), np.linalg.norm(s2)
                if n1 > 3 * 0.2 and n2 > 3 * 0.2:          # déplacement > 3× le bruit de localisation XY
                    turns[k].append(np.degrees(np.arccos(np.clip(s1 @ s2 / (n1 * n2), -1, 1))))
        net = P[-1] - P[0]
        rows.append(dict(traj=tr, n_points=len(g), t_first=int(t[0]), t_last=int(t[-1]), z_mean=float(P[:, 2].mean()),
                         z_range=float(np.ptp(P[:, 2])), v_xy=vxy, D_xy=Dxy, sigma_xy=sxy, v_z=vz, D_z=Dz, sigma_z=sz,
                         speed_smooth_median=float(np.nanmedian(sp)), net_displacement=float(np.linalg.norm(net)),
                         straightness=float(np.linalg.norm(net) / max(np.nansum(np.linalg.norm(np.diff(P, axis=0), axis=1)), 1e-9)),
                         vz_fraction=float(np.nanmedian(np.abs(V[ok, 2]) / np.maximum(sp[ok], 1e-9))) if ok.any() else np.nan,
                         nvox_median=float(g['NUMBER OF VOXEL'].median())))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'per_trajectory.csv'), index=False)
    # populations : nageuses si la composante dirigée domine la diffusion sur 10 images
    df['swimmer'] = df.v_xy * TAU_MAX > 2 * np.sqrt(4 * df.D_xy * TAU_MAX)
    ac = {k: float(np.mean(v)) for k, v in autoc.items() if v}
    sw = df[df.swimmer]
    tau_p = next((k for k, v in ac.items() if v < np.exp(-1)), None)
    V = np.concatenate(vel_all)
    summary = dict(
        csv=csv, trajectories=int(len(df)), points=int(df.n_points.sum()),
        frames=int(d.t.nunique()), points_per_frame=float(d.groupby('t').size().mean()),
        swimmers=int(df.swimmer.sum()),
        msd_fit_median={c: float(df[c].median()) for c in ['v_xy', 'D_xy', 'sigma_xy', 'v_z', 'D_z', 'sigma_z']},
        msd_fit_swimmers_median={c: float(sw[c].median()) for c in ['v_xy', 'D_xy', 'v_z', 'D_z']} if len(sw) else {},
        speed_smooth_um_per_frame=dict(median=float(np.median(np.linalg.norm(V, axis=1))),
                                       p10=float(np.percentile(np.linalg.norm(V, axis=1), 10)),
                                       p90=float(np.percentile(np.linalg.norm(V, axis=1), 90))),
        vz_over_v_median=float(df.vz_fraction.median()),
        persistence_cos=ac, persistence_time_frames=tau_p,
        turn_angle_deg={k: dict(median=float(np.median(v)), p90=float(np.percentile(v, 90)), n=len(v),
                                frac_gt_90=float(np.mean(np.array(v) > 90))) for k, v in turns.items() if v},
        straightness_median=float(df.straightness.median()),
        z_um=dict(min=float(d.z.min()), median=float(d.z.median()), max=float(d.z.max())),
    )
    # plus proche voisine parmi les bactéries suivies
    from scipy.spatial import cKDTree
    nn = []
    for _, g in d.groupby('t'):
        if len(g) > 1:
            dist, _ = cKDTree(g[['x', 'y', 'z']].to_numpy()).query(g[['x', 'y', 'z']].to_numpy(), k=2)
            nn += list(dist[:, 1])
    summary['nearest_neighbour_um'] = dict(median=float(np.median(nn)), p10=float(np.percentile(nn, 10)),
                                           frac_lt_4=float(np.mean(np.array(nn) < 4)))
    json.dump(summary, open(os.path.join(OUT, 'summary.json'), 'w', encoding='utf-8'), indent=2)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    a = ax[0, 0]
    for m in msd_xy_all:
        a.plot(taus, m, color='#2a78d6', alpha=0.25, lw=1)
    a.plot(taus, np.nanmedian(msd_xy_all, axis=0), color='#2a78d6', lw=3, label='XY (médiane)')
    a.plot(taus, np.nanmedian(msd_z_all, axis=0), color='#eb6834', lw=3, label='Z (médiane)')
    a.set_xscale('log'); a.set_yscale('log')
    a.plot(taus, taus ** 2 * np.nanmedian(msd_xy_all, axis=0)[0], ':', color='#888888', lw=1, label='∝ τ² (balistique)')
    a.plot(taus, taus * np.nanmedian(msd_xy_all, axis=0)[0], '--', color='#888888', lw=1, label='∝ τ (diffusion)')
    a.set_xlabel('délai τ (images)'); a.set_ylabel('MSD (µm²)'); a.set_title('Déplacement quadratique moyen'); a.legend(frameon=False, fontsize=8)
    a = ax[0, 1]
    a.hist(df.v_xy, bins=20, color='#2a78d6', alpha=0.8, label='XY')
    a.hist(df.v_z, bins=20, color='#eb6834', alpha=0.6, label='Z')
    a.set_xlabel('vitesse dirigée v du MSD (µm/image)'); a.set_title('Vitesse de nage par trajectoire'); a.legend(frameon=False)
    a = ax[0, 2]
    ks = sorted(ac)
    a.plot(ks, [ac[k] for k in ks], '-o', color='#1baf7a', ms=4)
    a.axhline(np.exp(-1), color='#888888', ls='--', lw=1)
    a.set_xlabel('délai τ (images)'); a.set_ylabel('⟨cos θ(t, t + τ)⟩'); a.set_title('Persistance de la direction'); a.set_ylim(-0.1, 1.02)
    a = ax[1, 0]
    for k, c in zip(turns, COLORS):
        if turns[k]:
            a.hist(turns[k], bins=np.arange(0, 181, 10), alpha=0.6, color=c, density=True, label=f'k = {k} images')
    a.set_xlabel('angle de virage (degrés)'); a.set_title('Virages entre déplacements successifs'); a.legend(frameon=False)
    a = ax[1, 1]
    a.hist(d.z, bins=40, color='#7a5bd6')
    a.set_xlabel('z (µm)'); a.set_title('Répartition en profondeur (points suivis)')
    a = ax[1, 2]
    for tr, g in list(d.groupby('traj'))[:57]:
        a.plot(g.x, g.y, lw=0.8)
    a.set_aspect('equal'); a.set_xlabel('x (µm)'); a.set_ylabel('y (µm)'); a.set_title('Trajectoires (projection XY)')
    for a in ax.ravel():
        a.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'dynamique.png'), dpi=110)
    plt.close(fig)
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    print(df.describe().round(3).to_string())


if __name__ == '__main__':
    main()

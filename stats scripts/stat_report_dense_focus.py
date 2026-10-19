# -*- coding: utf-8 -*-

"""
Filename: stat_report_dense_focus.py

Description:
Comparaison des critères de focus en MILIEU DENSE. Pour chaque densité N (nombre de bactéries
par hologramme) :
  1. simulation de n_holo hologrammes de N bactéries aléatoires (positions, tailles,
     orientations) avec "simu holo script/main_simu_hologram.py", ou relecture des hologrammes
     déjà présents dans output_dir/holograms (simulation.reuse_holograms) ;
  2. nettoyage par division par l'hologramme de référence (comme HOLOTRACKER), propagation, puis
     volume de focus pour chacun des critères de detection.focus_types ;
  3. fond local et σ local par maille (écrêtage κσ, comme pipeline_holotracker_locate_local_threshold) ;
  4. pour chaque bactérie simulée : contraste LOCAL = (pic du focus autour de la position
     vraie − fond local) / σ local, et contraste GLOBAL (fond et σ du volume entier) ;
  5. détection par seuil local T = fond + K·max(σ, σ plancher) puis composantes connexes, pour
     chaque K de detection.k_list : rappel (toutes, debout, couchées), précision, erreur de
     position des détections appariées.

Sorties (output_dir) :
  holograms/                       hologrammes simulés (.npy, référence, .bmp, positions .csv)
  bacteries.csv                    une ligne par (hologramme, critère, bactérie) : orientation,
                                   contrastes local et global, σ local
  detection.csv                    une ligne par (hologramme, critère, K) : rappel, précision,
                                   erreurs de position
  resume_detection.csv             par (N, critère) : K retenu (F1 maximal), rappel, précision, F1
  rappel_vs_densite.png            rappel (toutes et couchées) au K retenu
  contraste_local_vs_densite.png   contraste local médian des bactéries couchées et montée du σ local
  contraste_sur_seuil_vs_densite.png  contraste local médian des couchées / K nécessaire
  erreur_Z_vs_densite.png          |erreur Z| des détections appariées au K retenu
  copie du YAML

Choix de K pour chaque (N, critère) : le K qui maximise F1 = 2·P·R / (P + R) (précision P, rappel R,
sommés sur les hologrammes). Une contrainte « précision ≥ 95 % » est inutilisable à faible densité :
avec 5 bactéries, un seul faux positif sur 3 hologrammes donne 94 %.

Utilisation :
  python stats/stat_report_dense_focus.py stats/stat_report_dense_focus.yaml
  python stats/stat_report_dense_focus.py stats/stat_report_dense_focus.yaml --figures-only
      (recalcule résumé et figures à partir de bacteries.csv et detection.csv, sans détection)

Author: Simon BECKER
Date: 2026-10-13

License:
GNU General Public License v3.0
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import cupy as cp
import pandas as pd
import yaml
from cupyx.scipy import ndimage as nd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'localisation_pipeline'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from libs import propagation as propag, focus
from libs.focus import Focus_type
from libs.CCL3D import CCA_CUDA_float, connectivity_26, threshold_center_scale
import holo_stats as hs
from pipeline_holotracker_locate_local_threshold import mesh_background


MIN_LYING = 5          # nombre minimal de bactéries couchées (φ > 60°) pour publier une valeur « couchées »


# =============================================================================
# SIMULATION
# =============================================================================

def simulate(cfg, n, out_dir):
    """Hologrammes de n bactéries (n_holo) : simulés, ou relus s'ils existent déjà. Liste de dicts."""
    sim = cfg['simulation']
    hdir = os.path.join(out_dir, 'holograms', f'N{n:03d}')
    if not (sim['reuse_holograms'] and glob.glob(os.path.join(hdir, '*', 'simulated_hologram', '*.npy'))):
        if os.path.isdir(hdir):
            shutil.rmtree(hdir)
        c = json.load(open(os.path.join(REPO, 'simu holo script', 'config_bacteria_random.json'), encoding='utf-8'))
        c.update(mode='bacteria_random', nb_holo=sim['n_holo'], nb_objects=n, output_dir=hdir,
                 holo_size_xy=sim['holo_size'], border=sim['border'], upscale_factor=sim['upscale_factor'],
                 z_size=sim['nb_planes'], z_step=sim['step_um'] * 1e-6,
                 length_min=sim['length_um'][0] * 1e-6, length_max=sim['length_um'][1] * 1e-6,
                 thickness_min=sim['thickness_um'][0] * 1e-6, thickness_max=sim['thickness_um'][1] * 1e-6,
                 pix_size=sim['pixel_size_um'] * 1e-6, magnification=sim['magnification'],
                 index_medium=sim['index_medium'], index_object=sim['index_object'],
                 wavelength=sim['wavelength_nm'] * 1e-9, illumination_mean=1.0,
                 sensor_noise_std_min=sim['sensor_noise_std'], sensor_noise_std_max=sim['sensor_noise_std'],
                 number_of_sources=1, sources_angle_azimuth=[0.0], sources_angle_polar=[0.0],
                 distance_volume_camera=sim['distance_volume_camera_um'] * 1e-6,
                 holo_ref_nb_planes=sim['reference_nb_planes'])
        c['save_options'] = dict(hologram_bmp=True, hologram_npy=True, positions_csv=True, holo_ref=True,
                                 propagated_tiff=False, segmentation_tiff=False)
        os.makedirs(hdir, exist_ok=True)
        cfg_path = os.path.join(hdir, 'config_simulation.json')
        json.dump(c, open(cfg_path, 'w', encoding='utf-8'), indent=1)
        subprocess.run([sys.executable, '-W', 'ignore', os.path.join(REPO, 'simu holo script', 'main_simu_hologram.py'),
                        cfg_path], check=True, stdout=subprocess.DEVNULL,
                       env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
    run = sorted(glob.glob(os.path.join(hdir, '*', '')))[0]
    holos = []
    for npy in sorted(glob.glob(os.path.join(run, 'simulated_hologram', '*.npy'))):
        hid = os.path.basename(npy)[5:-4]                               # holo_<id>.npy
        gt = pd.read_csv(os.path.join(run, 'object_positions', f'bacteria_{hid}.csv'))
        holos.append({
            'id': hid, 'holo': np.load(npy), 'ref': np.load(os.path.join(run, 'holo_ref', f'holo_ref_{hid}.npy')),
            # repère de la reconstruction : x_image = y_simulé, y_image = x_simulé, Z = distance
            # volume-caméra + pos_z (flip Z du simulateur)
            'gt': np.c_[gt.y_position_m * 1e6, gt.x_position_m * 1e6,
                        sim['distance_volume_camera_um'] + gt.z_position_m * 1e6],
            'phi': gt.phi_angle.to_numpy(), 'length': gt.length.to_numpy() * 1e6,
            'thickness': gt.thickness.to_numpy() * 1e6})
    return holos


# =============================================================================
# DÉTECTION
# =============================================================================

def focus_volume(cleaned, cfg, crit):
    """Volume de focus (Z, Y, X) float32 sur GPU, module ou champ complexe selon le critère."""
    sim, det = cfg['simulation'], cfg['detection']
    N, nz = sim['holo_size'], sim['nb_planes']
    lam = sim['wavelength_nm'] * 1e-9 / sim['index_medium']
    pix, M = sim['pixel_size_um'] * 1e-6, sim['magnification']
    d0, dz = sim['distance_volume_camera_um'] * 1e-6, sim['step_um'] * 1e-6
    d_holo = cp.asarray(cleaned.astype(np.complex64))
    d = cp.zeros((nz, N, N), cp.float32)
    ft = Focus_type[crit]
    if ft in focus.COMPLEX_FOCUS_TYPES:
        propag.volume_propag_angular_spectrum_to_focus_complex(
            d_holo, d, focus.focus_plane_var_imag_over_real, det['sum_size'], lam, M, pix, N, N, d0, dz, nz,
            det['high_pass'], det['low_pass'])
    else:
        b = [cp.zeros((N, N), cp.complex64) for _ in range(3)]
        propag.volume_propag_angular_spectrum_to_module(d_holo, b[0], b[1], b[2], d, lam, M, pix, N, N, d0, dz, nz,
                                                        det['high_pass'], det['low_pass'])
        focus.focus(d, d, det['sum_size'], ft)
    return d


def objects(d, d_bin, cfg):
    """Barycentres (µm) des composantes connexes, pondérés par le focus (CCA HOLOTRACKER)."""
    sim = cfg['simulation']
    lab, n = nd.label(d_bin, connectivity_26)
    n = int(n)
    if n == 0:
        return np.zeros((0, 3))
    if n > cfg['detection']['max_objects']:
        return None                                   # seuil trop bas : volume rempli de bruit
    dx = sim['pixel_size_um'] / sim['magnification']
    f = CCA_CUDA_float(lab, d, n, 1, sim['holo_size'], sim['holo_size'], sim['nb_planes'], dx, dx, sim['step_um'],
                       z_offset=sim['distance_volume_camera_um'])
    return np.asarray(f, dtype=np.float64).reshape(-1, 5)[:, 1:4]


def match(det, gt, tol_xy, tol_z):
    """Appariement glouton par distance normalisée croissante, dans les tolérances. {i_gt: j_det}."""
    cand = []
    for i, g in enumerate(gt):
        dxy = np.hypot(det[:, 0] - g[0], det[:, 1] - g[1])
        dzz = np.abs(det[:, 2] - g[2])
        for j in np.flatnonzero((dxy < tol_xy) & (dzz < tol_z)):
            cand.append((dxy[j] / tol_xy + dzz[j] / tol_z, i, j))
    m, used = {}, set()
    for _, i, j in sorted(cand):
        if i not in m and j not in used:
            m[i] = j
            used.add(j)
    return m


# =============================================================================
# FIGURES
# =============================================================================

def figures(summary, contrast, out, cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    crits = cfg['detection']['focus_types']
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    markers = ['o', 's', '^', 'v', 'D', 'P', 'X', '*']
    rng = lambda v: f"{v[0]:g}" if v[0] == v[1] else f"{v[0]:g}–{v[1]:g}"
    sub = (f"bactéries {rng(cfg['simulation']['length_um'])} × "
           f"{rng(cfg['simulation']['thickness_um'])} µm, orientations "
           f"aléatoires — {cfg['simulation']['n_holo']} hologrammes par densité — seuil local "
           f"fond + K·σ, mailles {cfg['detection']['mesh_z']} plans × {cfg['detection']['mesh_xy']} px")

    def style(ax, ylabel, log=False):
        ax.set_xscale('log')
        ax.set_xlabel("N : nombre de bactéries dans l'hologramme")
        ax.set_ylabel(ylabel)
        ax.set_xticks(cfg['simulation']['densities'])
        ax.set_xticklabels([str(v) for v in cfg['simulation']['densities']])
        if log:
            ax.set_yscale('log')
        ax.grid(alpha=0.3, which='both')
        ax.legend(fontsize=8, ncol=2)

    def lines(ax, df, ycol):
        for i, c in enumerate(crits):
            g = df[df['crit'] == c].sort_values('n')
            if not g.empty and g[ycol].notna().any():
                ax.plot(g['n'], g[ycol], '-', marker=markers[i % 8], color=colors[i % len(colors)], lw=1.5, ms=5, label=c)

    fig, axes = plt.subplots(1, 2, figsize=(17, 6.2))
    lines(axes[0], summary, 'recall')
    style(axes[0], "rappel au K retenu (F1 maximal)")
    axes[0].set_title("Bactéries retrouvées (K qui maximise F1 ; précision dans resume_detection.csv)")
    lines(axes[1], summary, 'recall_lying')
    style(axes[1], f"rappel des bactéries couchées (φ > 60°)")
    axes[1].set_title("Bactéries couchées retrouvées (même K)")
    fig.suptitle(sub, fontsize=10); fig.tight_layout(); fig.savefig(os.path.join(out, 'rappel_vs_densite.png'), dpi=130); plt.close(fig)

    lying = contrast[contrast['phi'] > 60].groupby(['n', 'crit'])['contrast_local'].agg(['median', 'size']).reset_index()
    # pas de valeur « couchées » sur moins de MIN_LYING bactéries couchées (à faible densité, le
    # tirage peut n'en donner qu'une ou deux)
    lying['contrast_local'] = lying['median'].where(lying['size'] >= MIN_LYING)
    lying = lying[['n', 'crit', 'contrast_local']]
    sig = contrast.groupby(['n', 'crit'])['sigma_local'].median().reset_index()
    nmin = sig['n'].min()
    sig['sigma_rel'] = sig.apply(lambda r: r['sigma_local'] / float(sig[(sig['n'] == nmin) & (sig['crit'] == r['crit'])]['sigma_local'].iloc[0]), axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.2))
    lines(axes[0], lying, 'contrast_local')
    style(axes[0], "contraste local médian : (pic − fond local) / σ local", log=True)
    axes[0].set_title("Contraste local des bactéries couchées (φ > 60°)")
    lines(axes[1], sig, 'sigma_rel')
    style(axes[1], f"σ local médian / σ local médian à N = {nmin}")
    axes[1].set_title("Montée du fond local avec la densité (halo des voisines)")
    fig.suptitle(sub, fontsize=10); fig.tight_layout(); fig.savefig(os.path.join(out, 'contraste_local_vs_densite.png'), dpi=130); plt.close(fig)

    ratio = lying.merge(summary[['n', 'crit', 'K']], on=['n', 'crit'])
    ratio['contrast_over_K'] = ratio['contrast_local'] / ratio['K']
    fig, ax = plt.subplots(figsize=(10, 6.2))
    lines(ax, ratio, 'contrast_over_K')
    ax.axhline(1, color='k', lw=0.7)
    style(ax, "contraste local médian des couchées / K retenu", log=True)
    ax.set_title("Bactérie couchée typique rapportée au seuil nécessaire (1 = juste au seuil)\n" + sub, fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'contraste_sur_seuil_vs_densite.png'), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6.2))
    lines(ax, summary, 'abs_err_z_um')
    style(ax, "|erreur Z| moyenne des détections appariées (µm)")
    ax.set_title("Erreur Z des détections appariées, au K retenu (F1 maximal)\n" + sub, fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'erreur_Z_vs_densite.png'), dpi=130); plt.close(fig)
    return ratio


# =============================================================================
# MAIN
# =============================================================================

def summarize(out, cfg):
    """Résumé par (N, critère) au K qui maximise F1, puis figures, à partir des CSV."""
    bact = pd.read_csv(os.path.join(out, 'bacteries.csv'))
    detc = pd.read_csv(os.path.join(out, 'detection.csv'))
    ok = detc[detc['n_det'].notna()]                  # K trop bas (volume de bruit) exclus
    g = ok.groupby(['n', 'crit', 'K'])
    agg = g[['tp', 'n_det']].sum()
    agg['n_holo'] = g.size()
    # sommes sur les hologrammes : N bactéries simulées par hologramme
    agg['recall'] = agg['tp'] / (agg.index.get_level_values('n') * agg['n_holo'])
    agg['precision'] = agg['tp'] / agg['n_det'].clip(lower=1)
    agg['f1'] = 2 * agg['precision'] * agg['recall'] / (agg['precision'] + agg['recall']).replace(0, np.nan)
    agg['fp_per_holo'] = (agg['n_det'] - agg['tp']) / agg['n_holo']
    for c in ('recall_lying', 'recall_standing', 'abs_err_z_um', 'err_xy_um'):
        agg[c] = g[c].mean()
    agg = agg.reset_index()
    best = agg.loc[agg.groupby(['n', 'crit'])['f1'].idxmax().dropna().astype(int)]
    # rappel des couchées non significatif s'il repose sur moins de MIN_LYING bactéries couchées
    n_lying = bact[bact['crit'] == bact['crit'].iloc[0]].assign(ly=lambda t: t['phi'] > 60).groupby('n')['ly'].sum()
    best['n_lying'] = best['n'].map(n_lying).fillna(0).astype(int)
    best.loc[best['n_lying'] < MIN_LYING, 'recall_lying'] = np.nan
    summary = best[['n', 'crit', 'K', 'recall', 'precision', 'f1', 'fp_per_holo', 'n_lying', 'recall_lying', 'recall_standing',
                    'abs_err_z_um', 'err_xy_um']].reset_index(drop=True)
    ratio = figures(summary, bact, out, cfg)
    summary = summary.merge(ratio[['n', 'crit', 'contrast_local', 'contrast_over_K']], on=['n', 'crit'], how='left')
    summary.rename(columns={'contrast_local': 'contrast_local_lying_median'}).to_csv(
        os.path.join(out, 'resume_detection.csv'), index=False, float_format='%.4g')
    return summary


def main():
    if len(sys.argv) not in (2, 3) or (len(sys.argv) == 3 and sys.argv[2] != '--figures-only'):
        print("Utilisation : python stats/stat_report_dense_focus.py CONFIG.yaml [--figures-only]")
        return 1
    yaml_path = os.path.abspath(sys.argv[1])
    cfg = yaml.safe_load(open(yaml_path, encoding='utf-8'))
    sim, det = cfg['simulation'], cfg['detection']
    for c in det['focus_types']:
        if c not in hs.FOCUS_TYPES:
            print(f"critère inconnu : {c}")
            return 1
    out = cfg['output_dir']
    if not os.path.isabs(out):
        out = os.path.normpath(os.path.join(os.path.dirname(yaml_path), out))
    os.makedirs(out, exist_ok=True)
    if os.path.abspath(yaml_path) != os.path.abspath(os.path.join(out, os.path.basename(yaml_path))):
        shutil.copy2(yaml_path, out)
    if len(sys.argv) == 3:
        summarize(out, cfg)
        print(f"Résumé et figures recalculés -> {out}")
        return 0
    tol_xy, tol_z =cfg['matching']['tolerance_xy_um'], cfg['matching']['tolerance_z_um']
    dx = sim['pixel_size_um'] / sim['magnification']
    hb_xy, hb_z = int(round(cfg['matching']['peak_box_xy_um'] / dx)), int(round(cfg['matching']['peak_box_z_um'] / sim['step_um']))

    rows_b, rows_d = [], []
    t0 = time.perf_counter()
    for n in sim['densities']:
        holos = simulate(cfg, n, out)
        print(f"N = {n} : {len(holos)} hologramme(s) — {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
        for h in holos:
            cleaned = hs.clean_hologram(h['holo'], h['ref'], 'division', det['division_power'])
            lying, standing = h['phi'] > 60, h['phi'] < 30
            for crit in det['focus_types']:
                d = focus_volume(cleaned, cfg, crit)
                bg, sg = mesh_background(d, det['mesh_z'], det['mesh_xy'], det['clip_kappa'], det['clip_iterations'])
                floor = float(cp.percentile(sg.ravel(), det['sigma_floor_percentile']))
                gmed, gsig = threshold_center_scale(d, 'MAD')
                bg_c, sg_c = cp.asnumpy(bg), cp.asnumpy(cp.maximum(sg, floor))
                for i, (x, y, z) in enumerate(h['gt']):
                    ix, iy, iz = int(round(x / dx)), int(round(y / dx)), int(round((z - sim['distance_volume_camera_um']) / sim['step_um']))
                    if not 0 <= iz < sim['nb_planes']:
                        continue
                    sl = (slice(max(iz - hb_z, 0), iz + hb_z + 1), slice(max(iy - hb_xy, 0), iy + hb_xy + 1),
                          slice(max(ix - hb_xy, 0), ix + hb_xy + 1))
                    pk = float(d[sl].max())
                    mz = min(iz // det['mesh_z'], bg_c.shape[0] - 1)
                    my, mx = min(iy // det['mesh_xy'], bg_c.shape[1] - 1), min(ix // det['mesh_xy'], bg_c.shape[2] - 1)
                    rows_b.append(dict(n=n, holo=h['id'], crit=crit, bact=i, phi=float(h['phi'][i]),
                                       length_um=float(h['length'][i]), thickness_um=float(h['thickness'][i]),
                                       contrast_local=(pk - bg_c[mz, my, mx]) / sg_c[mz, my, mx],
                                       contrast_global=(pk - gmed) / gsig, sigma_local=float(sg_c[mz, my, mx])))
                for k in det['k_list']:
                    thr = nd.zoom(bg + k * cp.maximum(sg, floor), (det['mesh_z'], det['mesh_xy'], det['mesh_xy']),
                                  order=1, grid_mode=True, mode='nearest')
                    dets = objects(d, d > thr, cfg)
                    del thr
                    if dets is None:
                        rows_d.append(dict(n=n, holo=h['id'], crit=crit, K=k, n_det=np.nan))
                        continue
                    m = match(dets, h['gt'], tol_xy, tol_z) if len(dets) else {}
                    hit = np.array([i in m for i in range(len(h['gt']))])
                    err = np.array([dets[j] - h['gt'][i] for i, j in m.items()]) if m else np.zeros((0, 3))
                    rows_d.append(dict(n=n, holo=h['id'], crit=crit, K=k, n_det=len(dets), tp=len(m),
                                       recall=hit.mean(), recall_lying=hit[lying].mean() if lying.any() else np.nan,
                                       recall_standing=hit[standing].mean() if standing.any() else np.nan,
                                       abs_err_z_um=np.abs(err[:, 2]).mean() if len(err) else np.nan,
                                       err_xy_um=np.hypot(err[:, 0], err[:, 1]).mean() if len(err) else np.nan))
                del d, bg, sg
                cp.get_default_memory_pool().free_all_blocks()
                print(f"   {h['id']} {crit:<18s} — {(time.perf_counter() - t0) / 60:.1f} min", flush=True)

    pd.DataFrame(rows_b).to_csv(os.path.join(out, 'bacteries.csv'), index=False, float_format='%.5g')
    pd.DataFrame(rows_d).to_csv(os.path.join(out, 'detection.csv'), index=False, float_format='%.5g')
    summarize(out, cfg)
    print(f"\nTerminé en {(time.perf_counter() - t0) / 60:.1f} min -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
stat_report_dense_segmentation.py — méthodes de segmentation / localisation en milieu dense.

Relit les hologrammes de FOCUS_REPORT/DENSE (vérité terrain : positions et orientations) et compare,
sur le même volume de focus, cinq méthodes :
    1. global_ccl    seuil global moyenne + K·σ du volume, composantes connexes, barycentre pondéré
                     par le focus (chaîne historique d'HOLOTRACKER : calc_threshold, CCL3D, CCA)
    2. local_ccl     seuil local fond + K·max(σ, σ plancher) par mailles, composantes connexes, barycentre
    3. max_parabola  maxima locaux au-dessus du seuil local, position par parabole 3 points par axe
    4. max_c_thr     mêmes maxima, position = centroïde dans la boîte, poids max(focus − seuil local, 0)
    5. max_c_bg      mêmes maxima, position = centroïde dans la boîte, poids max(focus − fond local, 0)
Pour chaque K de detection.k_list : rappel, précision, rappel des couchées (φ > 60°) et erreurs de
position de chaque bactérie retrouvée. Puis, au K « fixe » de chaque méthode : temps de calcul.

Sorties (output_dir) : detection.csv, erreurs.csv, temps.csv, resume_detection.csv,
resume_erreurs.csv, figures PNG, copie du YAML.
Lancer : python stats/stat_report_dense_segmentation.py stats/stat_report_dense_segmentation.yaml [--figures-only]
"""
import os
import sys
import time
import shutil

import cupy as cp
import numpy as np
import pandas as pd
import yaml
from cupyx.scipy import ndimage as nd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'localisation_pipeline'))
sys.path.insert(0, HERE)
from libs.CCL3D import CCA_CUDA_float, connectivity_26
import holo_stats as hs
import stat_report_dense_focus as D
from pipeline_holotracker_locate_local_threshold import mesh_background
from pipeline_holotracker_locate_local_maxima import footprint_voxels, local_maxima, subvoxel_offsets

METHODS = ['global_ccl', 'local_ccl', 'max_parabola', 'max_c_thr', 'max_c_bg']
LABELS = {'global_ccl': 'Seuil global + CCL', 'local_ccl': 'Seuil local + CCL',
          'max_parabola': 'Maxima + parabole', 'max_c_thr': 'Maxima + centroïde (focus − seuil)',
          'max_c_bg': 'Maxima + centroïde (focus − fond)'}
# palette catégorielle validée (validate_palette.js, mode clair), ordre fixe
COLORS = {'global_ccl': '#2a78d6', 'local_ccl': '#eb6834', 'max_parabola': '#1baf7a',
          'max_c_thr': '#eda100', 'max_c_bg': '#e87ba4'}
MARKERS = {'global_ccl': 'o', 'local_ccl': 's', 'max_parabola': '^', 'max_c_thr': 'D', 'max_c_bg': 'v'}
LYING_PHI = 60.0
MIN_LYING = 5


# =============================================================================
# MÉTHODES
# =============================================================================

def ccl_positions(d, d_bin, dx, dz, z0, max_objects):
    """Composantes connexes (connectivité 26) et barycentres pondérés par le focus brut (µm)."""
    lab, n = nd.label(d_bin, connectivity_26)
    n = int(n)
    if n == 0:
        return np.zeros((0, 3))
    if n > max_objects:
        return None
    f = CCA_CUDA_float(lab, d, n, 1, d.shape[2], d.shape[1], d.shape[0], dx, dx, dz, z_offset=z0)
    del lab
    return np.asarray(f, dtype=np.float64).reshape(-1, 5)[:, 1:4]


def mesh_value(grid, iz, iy, ix, mz, mxy):
    """Grille de mailles interpolée en trilinéaire au voxel (même convention que zoom grid_mode)."""
    c = cp.stack([(iz + 0.5) / mz - 0.5, (iy + 0.5) / mxy - 0.5, (ix + 0.5) / mxy - 0.5]).astype(cp.float32)
    return nd.map_coordinates(grid, c, order=1, mode='nearest')


def box_centroid(d, iz, iy, ix, base, fp):
    """Centroïde (indices flottants) dans la boîte fp centrée sur chaque sommet, poids max(d − base, 0)."""
    hz, hy, hx = fp[0] // 2, fp[1] // 2, fp[2] // 2
    oz, oy, ox = cp.meshgrid(cp.arange(-hz, hz + 1), cp.arange(-hy, hy + 1), cp.arange(-hx, hx + 1), indexing='ij')
    Z = iz[:, None] + oz.ravel()[None]
    Y = iy[:, None] + oy.ravel()[None]
    X = ix[:, None] + ox.ravel()[None]
    inside = (Z >= 0) & (Z < d.shape[0]) & (Y >= 0) & (Y < d.shape[1]) & (X >= 0) & (X < d.shape[2])
    v = d[cp.clip(Z, 0, d.shape[0] - 1), cp.clip(Y, 0, d.shape[1] - 1), cp.clip(X, 0, d.shape[2] - 1)]
    w = cp.where(inside, cp.maximum(v - base[:, None], 0), 0)
    sw = cp.maximum(w.sum(axis=1), 1e-30)
    return (w * Z).sum(axis=1) / sw, (w * Y).sum(axis=1) / sw, (w * X).sum(axis=1) / sw


def to_um(z, y, x, dx, dz, z0):
    return cp.asnumpy(cp.stack([x * dx, y * dx, z0 + z * dz], axis=1)).astype(np.float64)


def run_methods(d, k, mesh, det, dx, dz, z0, methods=METHODS):
    """Positions (µm) détectées par chaque méthode au seuil K. mesh = (fond, σ, σ plancher) ou None."""
    out = {}
    if 'global_ccl' in methods:
        t = float(cp.mean(d)) + k * float(cp.std(d))
        out['global_ccl'] = ccl_positions(d, d > t, dx, dz, z0, det['max_objects'])
    if not any(m != 'global_ccl' for m in methods):
        return out
    mz, mxy = det['mesh_z'], det['mesh_xy']
    if mesh is None:
        bg, sg = mesh_background(d, mz, mxy, det['clip_kappa'], det['clip_iterations'])
        floor = float(cp.percentile(sg.ravel(), det['sigma_floor_percentile']))
    else:
        bg, sg, floor = mesh
    tm = bg + k * cp.maximum(sg, floor)
    thr = nd.zoom(tm, (mz, mxy, mxy), order=1, grid_mode=True, mode='nearest')
    if 'local_ccl' in methods:
        out['local_ccl'] = ccl_positions(d, d > thr, dx, dz, z0, det['max_objects'])
    if any(m.startswith('max_') for m in methods):
        fp = footprint_voxels(dx, dz, det['box_xy_um'], det['box_z_um'])
        iz, iy, ix = local_maxima(d, thr, fp)
        if iz.size > det['max_objects']:
            for m in methods:
                if m.startswith('max_'):
                    out[m] = None
        elif iz.size == 0:
            for m in methods:
                if m.startswith('max_'):
                    out[m] = np.zeros((0, 3))
        else:
            if 'max_parabola' in methods:
                oz, oy, ox = subvoxel_offsets(d, iz, iy, ix)
                out['max_parabola'] = to_um(iz + oz, iy + oy, ix + ox, dx, dz, z0)
            if 'max_c_thr' in methods:
                out['max_c_thr'] = to_um(*box_centroid(d, iz, iy, ix, thr[iz, iy, ix], fp), dx, dz, z0)
            if 'max_c_bg' in methods:
                out['max_c_bg'] = to_um(*box_centroid(d, iz, iy, ix, mesh_value(bg, iz, iy, ix, mz, mxy), fp), dx, dz, z0)
    del thr
    return out


# =============================================================================
# MESURE
# =============================================================================

def measure(cfg, dcfg, holo_dir, out):
    det = cfg['detection']
    sim = dcfg['simulation']
    dx, dz, z0 = sim['pixel_size_um'] / sim['magnification'], sim['step_um'], sim['distance_volume_camera_um']
    tol_xy, tol_z = dcfg['matching']['tolerance_xy_um'], dcfg['matching']['tolerance_z_um']
    rows_d, rows_e = [], []
    t0 = time.perf_counter()
    for n in sim['densities']:
        for h in D.simulate(dcfg, n, holo_dir):
            cleaned = hs.clean_hologram(h['holo'], h['ref'], 'division', dcfg['detection']['division_power'])
            lying = h['phi'] > LYING_PHI
            for crit in det['focus_types']:
                d = D.focus_volume(cleaned, dcfg, crit)
                bg, sg = mesh_background(d, det['mesh_z'], det['mesh_xy'], det['clip_kappa'], det['clip_iterations'])
                mesh = (bg, sg, float(cp.percentile(sg.ravel(), det['sigma_floor_percentile'])))
                for k in det['k_list']:
                    for m, pos in run_methods(d, k, mesh, det, dx, dz, z0).items():
                        if pos is None:
                            continue
                        mt = D.match(pos, h['gt'], tol_xy, tol_z) if len(pos) else {}
                        hit = np.array([i in mt for i in range(len(h['gt']))])
                        rows_d.append(dict(n=n, holo=h['id'], crit=crit, method=m, K=k, n_det=len(pos), tp=len(mt),
                                           n_gt=len(h['gt']), tp_lying=int(hit[lying].sum()), n_lying=int(lying.sum())))
                        for i, j in mt.items():
                            e = pos[j] - h['gt'][i]
                            rows_e.append(dict(n=n, holo=h['id'], crit=crit, method=m, K=k, gt=i, phi=h['phi'][i],
                                               ex=e[0], ey=e[1], ez=e[2]))
                    cp.get_default_memory_pool().free_all_blocks()
                del d, bg, sg, mesh
                cp.get_default_memory_pool().free_all_blocks()
            print(f"N = {n}, hologramme {h['id']} — {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    pd.DataFrame(rows_d).to_csv(os.path.join(out, 'detection.csv'), index=False)
    pd.DataFrame(rows_e).to_csv(os.path.join(out, 'erreurs.csv'), index=False)


def fixed_k(cfg, det_df):
    """K fixe par (critère, méthode) : maximise le F1 moyen sur les densités >= fixed_k_min_density."""
    a = aggregate(det_df[det_df.n >= cfg['detection']['fixed_k_min_density']], ['crit', 'method', 'n', 'K'])
    m = a.groupby(['crit', 'method', 'K'])['f1'].mean().reset_index()
    return m.loc[m.groupby(['crit', 'method'])['f1'].idxmax()][['crit', 'method', 'K']]


def timing(cfg, dcfg, holo_dir, kfix, out):
    """Temps seuil + détection + position, au K fixe, chaque méthode calculée seule (mailles comprises)."""
    det = cfg['detection']
    sim = dcfg['simulation']
    dx, dz, z0 = sim['pixel_size_um'] / sim['magnification'], sim['step_um'], sim['distance_volume_camera_um']
    rows = []
    for n in sim['densities']:
        for h in D.simulate(dcfg, n, holo_dir):
            cleaned = hs.clean_hologram(h['holo'], h['ref'], 'division', dcfg['detection']['division_power'])
            for crit in det['focus_types']:
                d = D.focus_volume(cleaned, dcfg, crit)
                for _, r in kfix[kfix.crit == crit].iterrows():
                    for rep in range(2):                      # 1re exécution = compilation des noyaux
                        cp.cuda.Device().synchronize()
                        t = time.perf_counter()
                        run_methods(d, r.K, None, det, dx, dz, z0, methods=[r.method])
                        cp.cuda.Device().synchronize()
                        if rep == 1:
                            rows.append(dict(n=n, holo=h['id'], crit=crit, method=r.method, K=r.K,
                                             t_ms=1e3 * (time.perf_counter() - t)))
                    cp.get_default_memory_pool().free_all_blocks()
                del d
                cp.get_default_memory_pool().free_all_blocks()
    pd.DataFrame(rows).to_csv(os.path.join(out, 'temps.csv'), index=False)


# =============================================================================
# RÉSUMÉ ET FIGURES
# =============================================================================

def aggregate(df, keys):
    a = df.groupby(keys)[['n_det', 'tp', 'n_gt', 'tp_lying', 'n_lying']].sum().reset_index()
    a['recall'] = a.tp / a.n_gt
    a['precision'] = a.tp / a.n_det.clip(lower=1)
    a['f1'] = (2 * a.precision * a.recall / (a.precision + a.recall)).where(a.tp > 0, 0.0)
    a['recall_lying'] = (a.tp_lying / a.n_lying).where(a.n_lying >= MIN_LYING)
    return a


def summarize(cfg, out):
    det_df = pd.read_csv(os.path.join(out, 'detection.csv'))
    err = pd.read_csv(os.path.join(out, 'erreurs.csv'))
    kfix = fixed_k(cfg, det_df)
    a = aggregate(det_df, ['crit', 'method', 'n', 'K'])
    at_fix = a.merge(kfix, on=['crit', 'method', 'K'])
    at_fix['k_choice'] = 'fixe'
    best = a.loc[a.groupby(['crit', 'method', 'n'])['f1'].idxmax()].copy()
    best['k_choice'] = 'F1 max par densité'
    res = pd.concat([at_fix, best])
    res['det_per_holo'] = res.n_det / res.n.map(det_df.groupby('n').holo.nunique())
    res.to_csv(os.path.join(out, 'resume_detection.csv'), index=False)

    # erreurs au K fixe, sur les bactéries retrouvées par les 5 méthodes (même population)
    e = err.merge(kfix, on=['crit', 'method', 'K'])
    key = ['crit', 'n', 'holo', 'gt']
    nm = e.groupby(key).method.nunique().rename('nm').reset_index()
    e = e.merge(nm, on=key)
    rows = []
    for pop, s in [('communes', e[e.nm == len(METHODS)]), ('communes couchées', e[(e.nm == len(METHODS)) & (e.phi > LYING_PHI)]),
                   ('toutes retrouvées', e)]:
        g = s.groupby(['crit', 'method', 'n'])
        r = g.agg(n_bact=('ez', 'size'), bias_x=('ex', 'mean'), std_x=('ex', 'std'), bias_y=('ey', 'mean'),
                  std_y=('ey', 'std'), bias_z=('ez', 'mean'), std_z=('ez', 'std')).reset_index()
        med = s.assign(axy=np.hypot(s.ex, s.ey), az=s.ez.abs()).groupby(['crit', 'method', 'n'])[['axy', 'az']].median()
        r = r.merge(med.rename(columns={'axy': 'median_abs_xy', 'az': 'median_abs_z'}).reset_index(), on=['crit', 'method', 'n'])
        r.insert(0, 'population', pop)
        rows.append(r)
        # toutes densités confondues
        r2 = s.groupby(['crit', 'method']).agg(n_bact=('ez', 'size'), bias_x=('ex', 'mean'), std_x=('ex', 'std'),
                                                bias_y=('ey', 'mean'), std_y=('ey', 'std'), bias_z=('ez', 'mean'),
                                                std_z=('ez', 'std')).reset_index()
        med2 = s.assign(axy=np.hypot(s.ex, s.ey), az=s.ez.abs()).groupby(['crit', 'method'])[['axy', 'az']].median()
        r2 = r2.merge(med2.rename(columns={'axy': 'median_abs_xy', 'az': 'median_abs_z'}).reset_index(), on=['crit', 'method'])
        r2.insert(0, 'population', pop)
        r2['n'] = 'toutes'
        rows.append(r2)
    re = pd.concat(rows)
    re.to_csv(os.path.join(out, 'resume_erreurs.csv'), index=False)
    figures(cfg, res, re, out)
    return res, re


def figures(cfg, res, re, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b',
                         'xtick.color': '#52514e', 'ytick.color': '#52514e', 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb'})
    dens = sorted(res.n.unique())
    for crit in cfg['detection']['focus_types']:
        sfx = '' if len(cfg['detection']['focus_types']) == 1 else f'_{crit}'

        def panel(ax, df, col, ylabel, title, methods=METHODS, labels=LABELS, colors=COLORS):
            for m in methods:
                s = df[df.method == m].sort_values('n')
                ax.plot(s.n, s[col], color=colors[m], marker=MARKERS[m], ms=7, lw=2, label=labels[m],
                        markeredgecolor='#fcfcfb', markeredgewidth=1.5)
            ax.set_xscale('log')
            ax.set_xticks(dens)
            ax.set_xticklabels([str(v) for v in dens])
            ax.set_xlabel("N : bactéries dans l'hologramme")
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontsize=10, loc='left')
            ax.grid(True, axis='y', color='#e4e3df', lw=0.8)

        for kc, name in [('fixe', 'detection_vs_densite_K_fixe'), ('F1 max par densité', 'detection_vs_densite_K_optimal')]:
            df = res[(res.crit == crit) & (res.k_choice == kc)]
            # les 3 variantes « maxima » partagent les mêmes sommets : une seule courbe de détection
            dm = ['global_ccl', 'local_ccl', 'max_c_thr']
            dl = {'global_ccl': LABELS['global_ccl'], 'local_ccl': LABELS['local_ccl'],
                  'max_c_thr': 'Maxima locaux (détection commune aux 3 variantes)'}
            dc = {'global_ccl': COLORS['global_ccl'], 'local_ccl': COLORS['local_ccl'], 'max_c_thr': COLORS['max_parabola']}
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
            panel(axes[0], df, 'recall', 'rappel', 'Rappel (bactéries retrouvées)', dm, dl, dc)
            panel(axes[1], df, 'precision', 'précision', 'Précision (détections justes)', dm, dl, dc)
            panel(axes[2], df, 'recall_lying', 'rappel des couchées', 'Rappel des bactéries couchées (φ > 60°)', dm, dl, dc)
            if kc == 'fixe':
                kk = 'K fixe : ' + ', '.join(f"{w} {int(df[df.method == m].K.iloc[0])}"
                                              for m, w in zip(dm, ('global', 'local', 'maxima')))
            else:
                kk = 'K maximisant F1 dans chaque case'
            fig.suptitle(f"{crit} — détection selon la densité ({kk})", fontsize=11, x=0.01, ha='left')
            h, l = axes[0].get_legend_handles_labels()
            fig.legend(h, l, loc='lower center', ncol=3, frameon=False, fontsize=9)
            fig.tight_layout(rect=(0, 0.08, 1, 0.94))
            fig.savefig(os.path.join(out, f'{name}{sfx}.png'), dpi=130)
            plt.close(fig)

        df = re[(re.crit == crit) & (re.population == 'communes') & (re.n != 'toutes')].copy()
        df['n'] = df.n.astype(int)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
        panel(axes[0], df, 'std_x', 'écart-type (µm)', 'Erreur en X')
        panel(axes[1], df, 'std_y', 'écart-type (µm)', 'Erreur en Y')
        panel(axes[2], df, 'std_z', 'écart-type (µm)', 'Erreur en Z')
        ymax = max(df.std_x.max(), df.std_y.max()) * 1.15
        for ax in axes[:2]:
            ax.set_ylim(0, ymax)
        axes[2].set_ylim(0, df.std_z.max() * 1.15)
        fig.suptitle(f"{crit} — erreur de position (écart-type), K fixe, bactéries retrouvées par les 5 méthodes",
                     fontsize=11, x=0.01, ha='left')
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc='lower center', ncol=5, frameon=False, fontsize=9)
        fig.tight_layout(rect=(0, 0.08, 1, 0.94))
        fig.savefig(os.path.join(out, f'erreur_position_vs_densite{sfx}.png'), dpi=130)
        plt.close(fig)


# =============================================================================

def main():
    if len(sys.argv) not in (2, 3) or (len(sys.argv) == 3 and sys.argv[2] != '--figures-only'):
        print(__doc__)
        return 1
    yaml_path = os.path.abspath(sys.argv[1])
    ydir = os.path.dirname(yaml_path)
    cfg = yaml.safe_load(open(yaml_path, encoding='utf-8'))
    dpath = os.path.normpath(os.path.join(ydir, cfg['dense_config']))
    dcfg = yaml.safe_load(open(dpath, encoding='utf-8'))
    dcfg['simulation']['reuse_holograms'] = True
    holo_dir = os.path.normpath(os.path.join(ydir, cfg['holograms_dir']))
    out = os.path.normpath(os.path.join(ydir, cfg['output_dir']))
    os.makedirs(out, exist_ok=True)
    for p in (yaml_path, dpath):
        shutil.copy2(p, out)
    if len(sys.argv) == 2:
        measure(cfg, dcfg, holo_dir, out)
        kfix = fixed_k(cfg, pd.read_csv(os.path.join(out, 'detection.csv')))
        timing(cfg, dcfg, holo_dir, kfix, out)
    summarize(cfg, out)
    print(f"Rapport -> {out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())

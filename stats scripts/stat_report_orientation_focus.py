# -*- coding: utf-8 -*-

"""
Filename: stat_report_orientation_focus.py

Description:
Erreur de position d'une bactérie en fonction de son orientation, pour PLUSIEURS critères de
focus. Pour chaque orientation (φ, θ) et chaque répétition (nouveau tirage du bruit) :
  1. simulation d'UN hologramme contenant une bactérie (et de son hologramme de référence) ;
  2. détection sur ce même hologramme avec chacun des critères de sweep.focus_types
     (même seuillage pour tous, voir detection.threshold_type) ;
  3. appariement avec la bactérie simulée : erreurs x, y, z, distance latérale et 3D ;
  4. mesures indépendantes du seuil, autour de la position VRAIE (report_common.probe_focus) :
     fond du volume de focus (médiane, 1,4826·MAD), pic du focus dans une boîte autour de la
     bactérie, contraste (pic − fond) / σ, voxels au-dessus de fond + 5σ et + 20σ.
Tous les critères voient donc exactement les mêmes hologrammes : les écarts entre courbes ne
viennent que du critère.

φ (polaire) : angle entre l'axe de la bactérie et l'axe optique, 0° = alignée, 90° = couchée.
θ (azimut)  : direction de l'inclinaison dans le plan de l'image, depuis l'axe x.

Seuil : les seuils en nombre d'écarts-types ne sont pas comparables d'un critère à l'autre (un
critère au carré a une autre dispersion). Le YAML fourni utilise PEAK_ABOVE_BG 0,5 : avec un seul
objet dans le volume, chaque critère est segmenté à mi-hauteur de son propre pic au-dessus de son
propre fond, par la même règle.

Sorties (dossier output.output_dir, par défaut output/stat_report_orientation_focus_<date>/) :
  raw.csv                                une ligne par (hologramme, critère)
  summary.csv                            une ligne par (critère, φ) : moyennes et écarts-types
                                         sur θ et les répétitions
  erreur_Z_vs_orientation.png            erreur Z signée = f(φ), moyenne ± écart-type
  abs_erreur_Z_vs_orientation.png        |erreur Z| = f(φ)
  sensibilite_bruit_vs_orientation.png   écart-type de l'erreur Z entre tirages de bruit, et
                                         contraste (pic − fond) / σ du critère = f(φ)
  intensite_focus_vs_orientation.png     pic au-dessus du fond rapporté à φ = 0°, et voxels
                                         au-dessus de fond + 20σ = f(φ) (à 5σ, la boîte de mesure
                                         est saturée quelle que soit l'orientation)
  copie du YAML utilisé

Utilisation :
  python stats/stat_report_orientation_focus.py stats/stat_report_orientation_focus.yaml

Author: Simon BECKER
Date: 2026-10-13

License:
GNU General Public License v3.0
"""

import copy
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import report_common as rc
import holo_stats as hs

SWEEP_SPEC = {
    'z_um': (float, None),
    'phi_start_deg': (float, None), 'phi_stop_deg': (float, None), 'phi_step_deg': (float, None),
    'theta_start_deg': (float, None), 'theta_stop_deg': (float, None), 'theta_step_deg': (float, None),
    'focus_types': ('strlist', hs.FOCUS_TYPES),
}

METRICS = ('err_x_um', 'err_y_um', 'err_z_um', 'abs_err_z_um', 'err_xy_um', 'err_3d_um', 'nb_vox',
           'focus_bg_median', 'focus_bg_sigma', 'focus_peak_box', 'focus_snr_box', 'peak_above_bg',
           'vox_above_5sigma', 'vox_above_20sigma')


def angles(start, stop, step, nom):
    """Balayage d'angle start -> stop inclus, par pas de step."""
    if step <= 0:
        raise ValueError(f"{nom}_step_deg doit être > 0")
    a = np.round(np.arange(start, stop + step / 2, step), 6)
    if a.size == 0:
        raise ValueError(f"balayage vide : vérifier {nom}_start_deg, {nom}_stop_deg, {nom}_step_deg")
    return a


def aggregate(rows, path):
    """Par (critère, φ), sur θ et les répétitions : taux de détection, moyennes et écarts-types."""
    import pandas as pd
    df = pd.DataFrame(rows)
    df['err_xy_um'] = np.hypot(df['err_x_um'], df['err_y_um'])
    df['err_3d_um'] = np.sqrt(df['err_x_um'] ** 2 + df['err_y_um'] ** 2 + df['err_z_um'] ** 2)
    df['abs_err_z_um'] = df['err_z_um'].abs()
    df['peak_above_bg'] = df['focus_peak_box'] - df['focus_bg_median']
    out = []
    for (ft, phi), g in df.groupby(['focus_type', 'phi_deg'], sort=False):
        rec = {'focus_type': ft, 'phi_deg': phi, 'n_holograms': len(g), 'detection_rate': float(g['detected'].mean()),
               'n_detected_mean': float(g['n_detected'].mean())}
        for c in METRICS:
            v = pd.to_numeric(g[c], errors='coerce').dropna()
            rec[f'{c}_mean'] = float(v.mean()) if len(v) else np.nan
            rec[f'{c}_std'] = float(v.std(ddof=0)) if len(v) else np.nan
        out.append(rec)
    agg = pd.DataFrame(out).sort_values(['focus_type', 'phi_deg']).reset_index(drop=True)
    # pic au-dessus du fond, rapporté à la bactérie alignée sur l'axe (φ minimal) du même critère
    ref = agg[agg['phi_deg'] == agg['phi_deg'].min()].set_index('focus_type')['peak_above_bg_mean']
    agg['peak_rel_to_phi0'] = agg['peak_above_bg_mean'] / agg['focus_type'].map(ref)
    agg.to_csv(path, index=False, float_format='%.6g')
    return agg


def plot(agg, focus_types, title, out_dir, show):
    """Les 4 figures décrites en tête de fichier, une courbe par critère."""
    show = rc.interactive_backend(show)
    import matplotlib.pyplot as plt
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    markers = ['o', 's', '^', 'v', 'D', 'P', 'X', '*']
    xlabel = "φ : angle entre la bactérie et l'axe optique (°) — 0 = alignée, 90 = couchée"

    def curves(ax, column, band_column=None, log=False):
        """Trace agg[column] = f(φ) pour chaque critère ; bande ± agg[band_column] si fourni."""
        for i, ft in enumerate(focus_types):
            g = agg[agg['focus_type'] == ft]
            if g.empty or g[column].isna().all():
                continue
            x, y = g['phi_deg'].to_numpy(), g[column].to_numpy()
            c = colors[i % len(colors)]
            ax.plot(x, y, '-', marker=markers[i % len(markers)], ms=5, lw=1.5, color=c, label=ft)
            if band_column is not None:
                sd = g[band_column].to_numpy()
                ax.fill_between(x, y - sd, y + sd, color=c, alpha=0.08)
        if log:
            ax.set_yscale('log')
        ax.set_xlabel(xlabel)
        ax.set_xticks(np.arange(0, 91, 10))
        ax.grid(alpha=0.3, which='both')
        ax.legend(fontsize=8, ncol=2)

    figs = []
    fig, ax = plt.subplots(figsize=(10, 6.2))
    curves(ax, 'err_z_um_mean', 'err_z_um_std')
    ax.axhline(0, color='k', lw=0.7)
    ax.set_ylabel("erreur Z = Z détecté − Z simulé (µm)")
    ax.set_title("Erreur Z signée (moyenne ± écart-type sur les tirages de bruit)\n" + title, fontsize=10)
    figs.append((fig, 'erreur_Z_vs_orientation.png'))

    fig, ax = plt.subplots(figsize=(10, 6.2))
    curves(ax, 'abs_err_z_um_mean', 'abs_err_z_um_std')
    ax.set_ylabel("|erreur Z| (µm)")
    ax.set_title("Erreur Z absolue (moyenne ± écart-type sur les tirages de bruit)\n" + title, fontsize=10)
    figs.append((fig, 'abs_erreur_Z_vs_orientation.png'))

    fig, axes = plt.subplots(1, 2, figsize=(17, 6.2))
    curves(axes[0], 'err_z_um_std')
    axes[0].set_ylabel("écart-type de l'erreur Z entre tirages de bruit (µm)")
    axes[0].set_title("Sensibilité au bruit : dispersion de la position Z")
    curves(axes[1], 'focus_snr_box_mean', log=True)
    axes[1].set_ylabel("contraste du critère : (pic − fond) / σ du fond")
    axes[1].set_title("Contraste du pic de focus au-dessus du bruit du fond")
    fig.suptitle(title, fontsize=10)
    figs.append((fig, 'sensibilite_bruit_vs_orientation.png'))

    fig, axes = plt.subplots(1, 2, figsize=(17, 6.2))
    curves(axes[0], 'peak_rel_to_phi0', log=True)
    axes[0].set_ylabel("(pic − fond) rapporté à la bactérie alignée (φ = 0°)")
    axes[0].set_title("Intensité du focus selon l'orientation")
    curves(axes[1], 'vox_above_20sigma_mean', log=True)
    axes[1].set_ylabel("voxels au-dessus de fond + 20σ autour de la bactérie")
    axes[1].set_title("Volume au-dessus d'un seuil fixe (fond + 20σ)")
    fig.suptitle(title, fontsize=10)
    figs.append((fig, 'intensite_focus_vs_orientation.png'))

    for fig, name in figs:
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, name), dpi=130)
    if show:
        plt.show()
    for fig, _ in figs:
        plt.close(fig)


def main():
    if len(sys.argv) != 2:
        print("Utilisation : python stats/stat_report_orientation_focus.py CONFIG.yaml")
        return 1
    yaml_path = os.path.abspath(sys.argv[1])
    try:
        cfg = rc.load_config(yaml_path, SWEEP_SPEC, drop=('bacterium.tilt_plane', 'detection.focus_type',
                                                          'detection.auto_threshold', 'detection.auto_nb_std_min',
                                                          'detection.auto_nb_std_max', 'detection.auto_precision'))
        sw = cfg['sweep']
        phis = angles(sw.phi_start_deg, sw.phi_stop_deg, sw.phi_step_deg, 'phi')
        thetas = angles(sw.theta_start_deg, sw.theta_stop_deg, sw.theta_step_deg, 'theta')
        cfg['detection'].auto_threshold = False          # check_config / detect s'y attendent
        warnings = rc.check_config(cfg, [sw.z_um])
    except (OSError, ValueError) as e:
        print(e)
        return 1
    run, sim, bac, det = cfg['run'], cfg['simulation'], cfg['bacterium'], cfg['detection']
    out = rc.prepare_output(cfg, yaml_path, "stat_report_orientation_focus")
    n_holo = phis.size * thetas.size * run.repeats
    total = n_holo * len(sw.focus_types)
    print("=" * 70)
    print("RAPPORT : ERREUR DE POSITION EN FONCTION DE L'ORIENTATION, PAR CRITÈRE DE FOCUS")
    print("=" * 70)
    print(f"φ de {phis[0]:g}° à {phis[-1]:g}° ({phis.size} angles), θ de {thetas[0]:g}° à {thetas[-1]:g}° "
          f"({thetas.size} angles), Z = {sw.z_um:g} µm, {run.repeats} répétition(s) -> {n_holo} hologrammes")
    print(f"critères : {', '.join(sw.focus_types)} ; seuil {det.threshold_type} × {det.nb_std_threshold:g}")
    print(f"sortie : {out}")
    for w in warnings:
        print(f"ATTENTION : {w}")

    np.random.seed(run.seed)                    # phase globale de l'illumination (create_illumination_field_polar)
    rng = np.random.default_rng(run.seed)       # bruit capteur
    writer = rc.CsvWriter(os.path.join(out, "raw.csv"))
    # boîte de mesure autour de la position vraie : ± tolérances d'appariement
    probe = dict(x_um=bac.x_um, y_um=bac.y_um, z_um=sw.z_um, half_xy_um=cfg['matching'].tolerance_xy_um,
                 half_z_um=cfg['matching'].tolerance_z_um, levels_sigma=(5, 20))
    t0, done = time.perf_counter(), 0
    for theta in thetas:
        for phi in phis:
            for rep in range(run.repeats):
                s = rc.simulate_single_bacterium(sim, bac, sw.z_um, float(phi), rng, theta_deg=float(theta))
                for ft in sw.focus_types:
                    t = time.perf_counter()
                    d = copy.copy(det)
                    d.focus_type = ft
                    res = rc.detect(s['holo'], s['ref'], sim, d, probe=probe)
                    row = {'focus_type': ft,
                           **rc.evaluate(res, bac, sw.z_um, cfg['matching'],
                                         params={'phi_deg': float(phi), 'theta_deg': float(theta)}),
                           **res['probe'],
                           'repeat': rep, 'sensor_noise_std': s['sensor_noise_std'], 'truncated': s['truncated'],
                           'time_s': time.perf_counter() - t}
                    writer.write(row)
                    done += 1
                    found = (f"err ({row['err_x_um']:+.2f}, {row['err_y_um']:+.2f}, {row['err_z_um']:+.2f}) µm, "
                             f"{int(row['nb_vox'])} vox") if row['detected'] else "non détectée"
                    print(f"[{done}/{total}] φ {phi:g}° θ {theta:g}° {ft:<18s} : {found}, "
                          f"contraste {row['focus_snr_box']:.1f} σ, {row['n_detected']} objet(s) — "
                          f"{rc.eta(t0, done, total)}", flush=True)

    agg = aggregate(writer.rows, os.path.join(out, "summary.csv"))
    plot(agg, sw.focus_types,
         f"Bactérie {bac.length_um:g} × {bac.thickness_um:g} µm à Z = {sw.z_um:g} µm, n = {sim.index_object:g} "
         f"(milieu {sim.index_medium:g}) — seuil {det.threshold_type} × {det.nb_std_threshold:g} — "
         f"{thetas.size} θ × {run.repeats} tirage(s) de bruit par φ",
         out, run.show_plots)
    print(f"\nTerminé : {done} détections en {(time.perf_counter() - t0) / 60:.1f} min -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

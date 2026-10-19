# -*- coding: utf-8 -*-

"""
Filename: stat_report_orientation.py

Description:
Rapport statistique en fonction de l'orientation d'une bactérie placée à une distance Z fixe.
Les DEUX angles sont balayés :
  φ (polaire) : angle entre l'axe de la bactérie et l'axe optique — 0° = alignée, 90° = couchée ;
  θ (azimut)  : direction de l'inclinaison dans le plan de l'image, mesurée depuis l'axe x —
                0° = plan x–z, 90° = plan y–z.
Pour chaque couple (φ, θ) du balayage :
  1. simulation d'un hologramme contenant UNE bactérie (et de son hologramme de référence) ;
  2. détection avec la chaîne HOLOTRACKER ;
  3. appariement avec la bactérie simulée : erreur de position x, y, z ;
  4. module normalisé (V − moyenne) / (écart-type + ε) : min, max, moyenne, pour l'objet apparié
     et pour l'ensemble des voxels segmentés ;
  5. focus de l'objet apparié (pic, moyenne, somme) et seuil de détection ;
  6. si detection.auto_threshold : seuil le plus élevé (en nombre d'écarts-types) qui ne laisse
     QU'UN SEUL objet dans le volume, apparié à la bactérie — mesure directe de la détectabilité.
Chaque point peut être répété (nouveau tirage du bruit) pour obtenir moyenne et écart-type.

Sorties (dossier output.output_dir, par défaut output/stat_report_orientation_<date>/) :
  raw.csv       une ligne par hologramme (écrite au fil de l'eau) : position vraie (gt_x_um,
                gt_y_um, gt_z_um), angles (phi_deg, theta_deg), position détectée (det_*),
                erreurs (err_*), intensité des voxels segmentés (norm_max, norm_mean), focus de
                ces voxels (focus_peak, focus_mean), nombre de voxels (nb_vox), puis le reste
  summary.csv   une ligne par couple (θ, φ) : taux de détection, moyennes / écarts-types
  report.png    graphes (affichés à la fin si run.show_plots)
  objects/      si output.save_objects : une image par hologramme (objets détectés, erreurs)
  copie du YAML utilisé

Utilisation :
  python stats/stat_report_orientation.py stats/stat_report_orientation.yaml

Author: Simon BECKER
Date: 2026-09-18

License:
GNU General Public License v3.0
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import report_common as rc

SWEEP_SPEC = {
    'z_um': (float, None),
    'phi_start_deg': (float, None), 'phi_stop_deg': (float, None), 'phi_step_deg': (float, None),
    'theta_start_deg': (float, None), 'theta_stop_deg': (float, None), 'theta_step_deg': (float, None),
}


def angles(start, stop, step, nom):
    """Balayage d'angle start -> stop inclus, par pas de step."""
    if step <= 0:
        raise ValueError(f"{nom}_step_deg doit être > 0")
    a = np.round(np.arange(start, stop + step / 2, step), 6)
    if a.size == 0:
        raise ValueError(f"balayage vide : vérifier {nom}_start_deg, {nom}_stop_deg, {nom}_step_deg")
    return a


def main():
    if len(sys.argv) != 2:
        print("Utilisation : python stats/stat_report_orientation.py CONFIG.yaml")
        return 1
    yaml_path = os.path.abspath(sys.argv[1])
    try:
        cfg = rc.load_config(yaml_path, SWEEP_SPEC, drop=('bacterium.tilt_plane',))
        sw = cfg['sweep']
        phis = angles(sw.phi_start_deg, sw.phi_stop_deg, sw.phi_step_deg, 'phi')
        thetas = angles(sw.theta_start_deg, sw.theta_stop_deg, sw.theta_step_deg, 'theta')
        warnings = rc.check_config(cfg, [sw.z_um])
    except (OSError, ValueError) as e:
        print(e)
        return 1
    run, sim, bac, det = cfg['run'], cfg['simulation'], cfg['bacterium'], cfg['detection']
    out = rc.prepare_output(cfg, yaml_path, "stat_report_orientation")
    total = phis.size * thetas.size * run.repeats
    print("=" * 70)
    print("RAPPORT : ERREUR DE POSITION ET INTENSITÉ EN FONCTION DE L'ORIENTATION")
    print("=" * 70)
    print(f"φ de {phis[0]:g}° à {phis[-1]:g}° par pas de {sw.phi_step_deg:g}° ({phis.size} angles), "
          f"θ de {thetas[0]:g}° à {thetas[-1]:g}° par pas de {sw.theta_step_deg:g}° ({thetas.size} angles), "
          f"Z = {sw.z_um:g} µm ; {run.repeats} répétition(s) -> {total} hologrammes")
    print(f"sortie : {out}")
    for w in warnings:
        print(f"ATTENTION : {w}")

    np.random.seed(run.seed)                    # phase globale de l'illumination (create_illumination_field_polar)
    rng = np.random.default_rng(run.seed)       # bruit capteur (écart-type par hologramme, puis tirage)
    writer = rc.CsvWriter(os.path.join(out, "raw.csv"))
    t0, done = time.perf_counter(), 0
    for theta in thetas:
        for phi in phis:
            for rep in range(run.repeats):
                t = time.perf_counter()
                s = rc.simulate_single_bacterium(sim, bac, sw.z_um, float(phi), rng, theta_deg=float(theta))
                res = rc.detect(s['holo'], s['ref'], sim, det,
                                auto=(bac, sw.z_um, cfg['matching']) if det.auto_threshold else None)
                row = {**rc.evaluate(res, bac, sw.z_um, cfg['matching'],
                                     params={'phi_deg': float(phi), 'theta_deg': float(theta)}),
                       'repeat': rep, 'sensor_noise_std': s['sensor_noise_std'], 'truncated': s['truncated'],
                       'time_s': time.perf_counter() - t}
                writer.write(row)
                base = f'phi{phi:g}_theta{theta:g}_z{sw.z_um:g}_r{rep}'
                if cfg['output'].save_objects:
                    rc.plot_objects(res, bac, float(sw.z_um), float(phi), sim.pixel_size_um / sim.magnification,
                                    os.path.join(out, 'objects', base + '.png'),
                                    noise_std=s['sensor_noise_std'], theta_deg=float(theta))
                if cfg['output'].save_holograms:
                    import tifffile
                    p = os.path.join(out, "holograms", base)
                    tifffile.imwrite(p + "_holo.tiff", s['holo'].astype(np.float32))
                    tifffile.imwrite(p + "_ref.tiff", s['ref'].astype(np.float32))
                done += 1
                seuil = (f", seuil limite {row['nb_std_found']:.2f} σ" if det.auto_threshold and row['detected'] else "")
                print(f"[{done}/{total}] φ {phi:g}° θ {theta:g}° : "
                      + (f"err ({row['err_x_um']:+.2f}, {row['err_y_um']:+.2f}, {row['err_z_um']:+.2f}) µm, "
                         f"max norm. {row['norm_max']:.1f}{seuil}" if row['detected'] else "non détectée")
                      + f", {row['n_detected']} objet(s) — {rc.eta(t0, done, total)}", flush=True)

    agg = rc.aggregate(writer.rows, ['theta_deg', 'phi_deg'], os.path.join(out, "summary.csv"))
    rc.plot_report(agg, 'phi_deg', "φ : angle avec l'axe optique (°), 0 = alignée, 90 = couchée",
                   'theta_deg' if thetas.size > 1 else None, "θ (°)",
                   f"Bactérie {bac.length_um:g} × {bac.thickness_um:g} µm à Z = {sw.z_um:g} µm, "
                   f"n = {sim.index_object:g} (milieu {sim.index_medium:g}), {run.repeats} répétition(s) par angle"
                   + (" — seuil cherché automatiquement" if det.auto_threshold else ""),
                   os.path.join(out, "report.png"), run.show_plots,
                   extra_metric=(('nb_std_found', "Seuil limite de détection", "seuil (nb d'écarts-types)")
                                 if det.auto_threshold else None))
    print(f"\nTerminé : {done} hologrammes en {(time.perf_counter() - t0) / 60:.1f} min -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

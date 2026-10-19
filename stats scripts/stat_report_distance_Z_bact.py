# -*- coding: utf-8 -*-

"""
Filename: stat_report_distance_Z.py

Description:
Rapport statistique en fonction de la distance Z (bactérie ↔ hologramme).
Pour chaque orientation (φ, angle avec l'axe optique) et chaque position Z du balayage :
  1. simulation d'un hologramme contenant UNE bactérie (et de son hologramme de référence) ;
  2. détection avec la chaîne HOLOTRACKER ;
  3. appariement avec la bactérie simulée : erreur de position x, y, z ;
  4. module normalisé (V − moyenne) / (écart-type + ε) : min, max, moyenne, pour l'objet apparié
     et pour l'ensemble des voxels segmentés ;
  5. focus de l'objet apparié (pic, moyenne, somme) et seuil de détection.
Chaque point peut être répété (nouveau tirage du bruit) pour obtenir moyenne et écart-type.

Sorties (dossier output.output_dir, par défaut output/stat_report_distance_Z_<date>/) :
  raw.csv       une ligne par hologramme (écrite au fil de l'eau) : position vraie (gt_x_um,
                gt_y_um, gt_z_um), angle (phi_deg), position détectée (det_*), erreurs (err_*),
                intensité des voxels segmentés (norm_max, norm_mean), focus de ces voxels
                (focus_peak, focus_mean), nombre de voxels (nb_vox), puis le reste
  summary.csv   une ligne par (orientation, Z) : taux de détection, moyennes / écarts-types
  report.png    graphes (affichés à la fin si run.show_plots)
  objects/      si output.save_objects : une image par hologramme (objets détectés, erreurs)
  copie du YAML utilisé

Utilisation :
  python stats/stat_report_distance_Z.py stats/stat_report_distance_Z.yaml

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
    'z_start_um': (float, None), 'z_stop_um': (float, None), 'z_step_um': (float, None),
    'orientations_phi_deg': ('floatlist', None),
}


def main():
    if len(sys.argv) != 2:
        print("Utilisation : python stats/stat_report_distance_Z.py CONFIG.yaml")
        return 1
    yaml_path = os.path.abspath(sys.argv[1])
    try:
        cfg = rc.load_config(yaml_path, SWEEP_SPEC)
        sw = cfg['sweep']
        z_values = np.round(np.arange(sw.z_start_um, sw.z_stop_um + sw.z_step_um / 2, sw.z_step_um), 6)
        if z_values.size == 0:
            raise ValueError("balayage Z vide : vérifier z_start_um, z_stop_um, z_step_um")
        warnings = rc.check_config(cfg, list(z_values))
    except (OSError, ValueError) as e:
        print(e)
        return 1
    run, sim, bac, det = cfg['run'], cfg['simulation'], cfg['bacterium'], cfg['detection']
    out = rc.prepare_output(cfg, yaml_path, "stat_report_distance_Z")
    total = len(sw.orientations_phi_deg) * z_values.size * run.repeats
    print("=" * 70)
    print("RAPPORT : ERREUR DE POSITION ET INTENSITÉ EN FONCTION DE Z")
    print("=" * 70)
    print(f"Z de {z_values[0]:g} à {z_values[-1]:g} µm par pas de {sw.z_step_um:g} µm ({z_values.size} positions)")
    print(f"orientations φ : {', '.join(f'{p:g}°' for p in sw.orientations_phi_deg)} (plan {bac.tilt_plane}) ; "
          f"{run.repeats} répétition(s) -> {total} hologrammes")
    print(f"sortie : {out}")
    for w in warnings:
        print(f"ATTENTION : {w}")

    np.random.seed(run.seed)                    # phase globale de l'illumination (create_illumination_field_polar)
    rng = np.random.default_rng(run.seed)       # bruit capteur (écart-type par hologramme, puis tirage)
    writer = rc.CsvWriter(os.path.join(out, "raw.csv"))
    t0, done = time.perf_counter(), 0
    for phi in sw.orientations_phi_deg:
        for z in z_values:
            for rep in range(run.repeats):
                t = time.perf_counter()
                s = rc.simulate_single_bacterium(sim, bac, float(z), phi, rng)
                res = rc.detect(s['holo'], s['ref'], sim, det)
                row = {**rc.evaluate(res, bac, float(z), cfg['matching'], params={'phi_deg': float(phi)}),
                       'repeat': rep, 'sensor_noise_std': s['sensor_noise_std'], 'truncated': s['truncated'],
                       'time_s': time.perf_counter() - t}
                writer.write(row)
                if cfg['output'].save_objects:
                    rc.plot_objects(res, bac, float(z), float(phi), sim.pixel_size_um / sim.magnification,
                                    os.path.join(out, 'objects', f'phi{phi:g}_z{z:g}_r{rep}.png'),
                                    noise_std=s['sensor_noise_std'])
                if cfg['output'].save_holograms:
                    import tifffile
                    base = os.path.join(out, "holograms", f"phi{phi:g}_z{z:g}_r{rep}")
                    tifffile.imwrite(base + "_holo.tiff", s['holo'].astype(np.float32))
                    tifffile.imwrite(base + "_ref.tiff", s['ref'].astype(np.float32))
                done += 1
                print(f"[{done}/{total}] φ {phi:g}° Z {z:g} µm : "
                      + (f"err ({row['err_x_um']:+.2f}, {row['err_y_um']:+.2f}, {row['err_z_um']:+.2f}) µm, "
                         f"max norm. {row['norm_max']:.1f}" if row['detected'] else "non détectée")
                      + f", {row['n_detected']} objet(s) — {rc.eta(t0, done, total)}", flush=True)

    agg = rc.aggregate(writer.rows, ['phi_deg', 'gt_z_um'], os.path.join(out, "summary.csv"))
    rc.plot_report(agg, 'gt_z_um', "Z : distance bactérie ↔ hologramme (µm)", 'phi_deg', "φ (°)",
                   f"Bactérie {bac.length_um:g} × {bac.thickness_um:g} µm, n = {sim.index_object:g} "
                   f"(milieu {sim.index_medium:g}), {run.repeats} répétition(s) par point",
                   os.path.join(out, "report.png"), run.show_plots)
    print(f"\nTerminé : {done} hologrammes en {(time.perf_counter() - t0) / 60:.1f} min -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

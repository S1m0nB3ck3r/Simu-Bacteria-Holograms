# -*- coding: utf-8 -*-

"""
Filename: stat_report_distance_Z_sphere.py

Description:
Rapport statistique en fonction de la distance Z (sphère ↔ hologramme) — précision de la
détection sur des objets sphériques, donc sans effet d'orientation.
Pour chaque rayon de la liste et chaque position Z du balayage :
  1. simulation d'un hologramme contenant UNE sphère (et de son hologramme de référence) ;
  2. détection avec la chaîne HOLOTRACKER ;
  3. appariement avec la sphère simulée : erreur de position x, y, z ;
  4. module normalisé (V − moyenne) / (écart-type + ε) : min, max, moyenne, pour l'objet apparié
     et pour l'ensemble des voxels segmentés ;
  5. focus de l'objet apparié (pic, moyenne, somme) et seuil de détection.
Chaque point peut être répété (nouveau tirage du bruit) pour obtenir moyenne et écart-type.

La sphère est l'objet de référence pour mesurer la précision de localisation : symétrique, son
barycentre est sa position, sans le biais projectif d'un bâtonnet incliné.

Sorties (dossier output.output_dir, par défaut output/stat_report_distance_Z_sphere_<date>/) :
  raw.csv       une ligne par hologramme (écrite au fil de l'eau) : position vraie (gt_x_um,
                gt_y_um, gt_z_um), rayon (radius_um), position détectée (det_*), erreurs (err_*),
                intensité des voxels segmentés (norm_max, norm_mean), focus de ces voxels
                (focus_peak, focus_mean), nombre de voxels (nb_vox), puis le reste
  summary.csv   une ligne par (rayon, Z) : taux de détection, moyennes / écarts-types
  report.png    graphes (affichés à la fin si run.show_plots), une courbe par rayon
  objects/      si output.save_objects : une image par hologramme (objets détectés, erreurs)
  copie du YAML utilisé

Utilisation :
  python stats/stat_report_distance_Z_sphere.py stats/stat_report_distance_Z_sphere.yaml

Author: Simon BECKER
Date: 2026-10-12

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
    'radii_um': ('floatlist', None),
}

# section propre à ce rapport : elle remplace 'bacterium'
SPHERE_SPEC = {'x_um': (float, None), 'y_um': (float, None)}


def main():
    if len(sys.argv) != 2:
        print("Utilisation : python stats/stat_report_distance_Z_sphere.py CONFIG.yaml")
        return 1
    yaml_path = os.path.abspath(sys.argv[1])
    try:
        cfg = rc.load_config(yaml_path, SWEEP_SPEC, drop=('bacterium',), extra={'sphere': SPHERE_SPEC})
        sw = cfg['sweep']
        z_values = np.round(np.arange(sw.z_start_um, sw.z_stop_um + sw.z_step_um / 2, sw.z_step_um), 6)
        if z_values.size == 0:
            raise ValueError("balayage Z vide : vérifier z_start_um, z_stop_um, z_step_um")
        if not sw.radii_um or min(sw.radii_um) <= 0:
            raise ValueError("sweep.radii_um : liste de rayons > 0 attendue")
        warnings = rc.check_config(cfg, list(z_values), obj_section='sphere', half_um=max(sw.radii_um))
    except (OSError, ValueError) as e:
        print(e)
        return 1
    run, sim, sph, det = cfg['run'], cfg['simulation'], cfg['sphere'], cfg['detection']
    out = rc.prepare_output(cfg, yaml_path, "stat_report_distance_Z_sphere")
    total = len(sw.radii_um) * z_values.size * run.repeats
    print("=" * 70)
    print("RAPPORT SPHÈRES : ERREUR DE POSITION ET INTENSITÉ EN FONCTION DE Z")
    print("=" * 70)
    print(f"Z de {z_values[0]:g} à {z_values[-1]:g} µm par pas de {sw.z_step_um:g} µm ({z_values.size} positions)")
    print(f"rayons : {', '.join(f'{r:g} µm' for r in sw.radii_um)} ; "
          f"{run.repeats} répétition(s) -> {total} hologrammes")
    print(f"sortie : {out}")
    for w in warnings:
        print(f"ATTENTION : {w}")

    np.random.seed(run.seed)                    # phase globale de l'illumination (create_illumination_field_polar)
    rng = np.random.default_rng(run.seed)       # bruit capteur (écart-type par hologramme, puis tirage)
    writer = rc.CsvWriter(os.path.join(out, "raw.csv"))
    t0, done = time.perf_counter(), 0
    for radius in sw.radii_um:
        for z in z_values:
            for rep in range(run.repeats):
                t = time.perf_counter()
                s = rc.simulate_single_sphere(sim, sph, float(z), float(radius), rng)
                res = rc.detect(s['holo'], s['ref'], sim, det)
                row = {**rc.evaluate(res, sph, float(z), cfg['matching'], params={'radius_um': float(radius)}),
                       'repeat': rep, 'sensor_noise_std': s['sensor_noise_std'], 'truncated': s['truncated'],
                       'time_s': time.perf_counter() - t}
                writer.write(row)
                base = f'r{radius:g}_z{z:g}_r{rep}'
                if cfg['output'].save_objects:
                    rc.plot_objects(res, sph, float(z), None, sim.pixel_size_um / sim.magnification,
                                    os.path.join(out, 'objects', base + '.png'),
                                    noise_std=s['sensor_noise_std'], label=f"r = {radius:g} µm")
                if cfg['output'].save_holograms:
                    import tifffile
                    p = os.path.join(out, "holograms", base)
                    tifffile.imwrite(p + "_holo.tiff", s['holo'].astype(np.float32))
                    tifffile.imwrite(p + "_ref.tiff", s['ref'].astype(np.float32))
                done += 1
                print(f"[{done}/{total}] r {radius:g} µm Z {z:g} µm : "
                      + (f"err ({row['err_x_um']:+.2f}, {row['err_y_um']:+.2f}, {row['err_z_um']:+.2f}) µm, "
                         f"max norm. {row['norm_max']:.1f}" if row['detected'] else "non détectée")
                      + f", {row['n_detected']} objet(s) — {rc.eta(t0, done, total)}", flush=True)

    agg = rc.aggregate(writer.rows, ['radius_um', 'gt_z_um'], os.path.join(out, "summary.csv"))
    rc.plot_report(agg, 'gt_z_um', "Z : distance sphère ↔ hologramme (µm)", 'radius_um', "rayon (µm)",
                   f"Sphères de rayon {', '.join(f'{r:g}' for r in sw.radii_um)} µm, "
                   f"n = {sim.index_object:g} (milieu {sim.index_medium:g}), "
                   f"{run.repeats} répétition(s) par point",
                   os.path.join(out, "report.png"), run.show_plots)
    print(f"\nTerminé : {done} hologrammes en {(time.perf_counter() - t0) / 60:.1f} min -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

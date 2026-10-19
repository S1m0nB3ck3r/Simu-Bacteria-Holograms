# -*- coding: utf-8 -*-

"""
Filename: report_common.py

Description:
Briques communes aux rapports statistiques (stat_report_distance_Z.py, stat_report_orientation.py) :
  - simulation d'un hologramme contenant UNE bactérie, avec le modèle physique du simulateur
    (masque de la bactérie, déphasage par tranche, apodisation, bruit d'illumination, référence
    moyennée sur N illuminations) ;
  - détection identique à HOLOTRACKER (moteur de holo_stats : propagation, focus, seuil, CCL3D) ;
  - intensité normalisée des objets : (V − moyenne du volume) / (écart-type du volume + ε) ;
  - appariement avec la bactérie simulée, erreur de position ;
  - lecture / validation du YAML, écriture des CSV, graphes.

Simulation « par tranche utile » : seules les tranches qui contiennent la bactérie sont traversées
pas à pas (propagation dz + masque de phase), le vide avant et après est franchi en une seule
propagation. En espace libre, propager d1 puis d2 équivaut exactement à propager d1 + d2 : le
résultat est celui du simulateur complet (vérifié), pour une fraction du temps.

Repères : X, Y sont ceux de l'image analysée (x = colonne, y = ligne, comme HOLOTRACKER) ;
Z est la distance du centre de la bactérie à l'hologramme. Le simulateur place x en ligne de
l'image : la conversion est faite ici (x_simulé = y_image, y_simulé = x_image).

Author: Simon BECKER
Date: 2026-09-18

License:
GNU General Public License v3.0
"""

import csv
import datetime
import math
import os
import shutil
import sys
import time
from types import SimpleNamespace

import numpy as np
import cupy as cp

STATS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(STATS_DIR)
sys.path.insert(0, STATS_DIR)
sys.path.insert(0, REPO_DIR)

import holo_stats as hs
from libs import propagation
from libs import traitement_holo
from libs import focus
from libs.focus import Focus_type
from libs.CCL3D import CCL3D, calc_threshold, threshold_center_scale, type_threshold, THRESHOLD_TYPES
from libs.simu_hologram import (Bacterie, Sphere, GPU_insert_bact_in_mask_volume,
                                insert_sphere_in_mask_volume, phase_shift_through_plane,
                                pad_centered, create_illumination_field_polar, apodize_border,
                                add_sensor_noise)


# =============================================================================
# CONFIGURATION YAML
# =============================================================================
# section -> {clé: (type, valeurs autorisées ou None)} ; 'floatlist' = liste de réels ;
# 'strlist' = liste de chaînes (valeurs autorisées vérifiées élément par élément).
COMMON_SPEC = {
    'output': {'output_dir': ('path_or_null', None), 'save_holograms': (bool, None),
               'save_objects': (bool, None)},
    'simulation': {
        'holo_size': (int, None), 'border': (int, None), 'upscale_factor': (int, None),
        'pixel_size_um': (float, None), 'magnification': (float, None), 'wavelength_nm': (float, None),
        'index_medium': (float, None), 'index_object': (float, None),
        'absorption_percent_per_um': (float, None), 'sim_step_um': (float, None),
        'illumination_distance_um': (float, None), 'illumination_mean': (float, None),
        'reference_nb_planes': (int, None),
        'sensor_noise_std_min': (float, None), 'sensor_noise_std_max': (float, None),
        'source_azimuth_deg': (float, None), 'source_polar_deg': (float, None),
    },
    'bacterium': {'length_um': (float, None), 'thickness_um': (float, None),
                  'x_um': (float, None), 'y_um': (float, None), 'tilt_plane': (str, ['xz', 'yz'])},
    'detection': {
        'cleaning': (str, ['division', 'subtraction', 'none']), 'division_power': (float, None),
        'distance_ini_um': (float, None), 'nb_planes': (int, None), 'step_um': (float, None),
        'high_pass': (int, None), 'low_pass': (int, None), 'focus_type': (str, hs.FOCUS_TYPES),
        'sum_size': (int, None), 'threshold_type': (str, list(THRESHOLD_TYPES)),
        'nb_std_threshold': (float, None), 'connectivity': (int, [6, 18, 26]),
        'min_voxel': (int, None), 'max_voxel': (int, None),
        'auto_threshold': (bool, None), 'auto_nb_std_min': (float, None), 'auto_nb_std_max': (float, None),
        'auto_precision': (float, None),
    },
    'matching': {'tolerance_xy_um': (float, None), 'tolerance_z_um': (float, None)},
    'run': {'repeats': (int, None), 'seed': (int, None), 'show_plots': (bool, None)},
}


def load_config(yaml_path, sweep_spec, drop=(), extra=None):
    """
    Lit et valide le YAML : tous les paramètres sont obligatoires. Retourne {section: SimpleNamespace}.
    drop  : « section.clé » ou « section » de COMMON_SPEC sans objet pour ce script (ni attendus,
            ni acceptés dans le YAML).
    extra : sections propres au script, au même format que COMMON_SPEC (par ex. 'sphere').
    """
    import yaml
    with open(yaml_path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f) or {}
    spec = {section: dict(keys) for section, keys in COMMON_SPEC.items()}
    spec['sweep'] = sweep_spec
    for section, keys in (extra or {}).items():
        spec[section] = dict(keys)
    for d in drop:
        section, _, key = d.partition('.')
        if key:
            spec[section].pop(key, None)
        else:
            spec.pop(section, None)
    errors, out = [], {}
    for section, keys in spec.items():
        values = raw.get(section) or {}
        ns = {}
        for key, (typ, choices) in keys.items():
            if key not in values:
                errors.append(f"{section}.{key} manquant")
                continue
            v = values[key]
            try:
                if typ == 'path_or_null':
                    v = None if v is None or str(v).strip() in ('', 'null', 'None') else str(v)
                elif typ == 'floatlist':
                    v = [float(x) for x in (v if isinstance(v, (list, tuple)) else [v])]
                elif typ == 'strlist':
                    v = [str(x) for x in (v if isinstance(v, (list, tuple)) else [v])]
                    if not v:
                        raise ValueError
                elif typ is bool:
                    if not isinstance(v, bool):
                        raise ValueError
                else:
                    v = typ(v)      # PyYAML lit « 660e-9 » (sans point) comme une chaîne
            except (TypeError, ValueError):
                errors.append(f"{section}.{key} = {values[key]!r} : type incorrect")
                continue
            bad = [x for x in v if x not in choices] if (choices is not None and typ == 'strlist') else                   ([v] if (choices is not None and v not in choices) else [])
            if bad:
                errors.append(f"{section}.{key} = {bad[0] if typ == 'strlist' else v!r} : valeurs possibles {choices}")
                continue
            ns[key] = v
        for key in set(values) - set(keys):
            errors.append(f"{section}.{key} : paramètre inconnu")
        out[section] = SimpleNamespace(**ns)
    for section in set(raw) - set(spec):
        errors.append(f"section inconnue : {section}")
    if errors:
        raise ValueError("Fichier de configuration invalide :\n  - " + "\n  - ".join(errors))
    base = os.path.dirname(os.path.abspath(yaml_path))
    if out['output'].output_dir is not None and not os.path.isabs(out['output'].output_dir):
        out['output'].output_dir = os.path.normpath(os.path.join(base, out['output'].output_dir))
    return out


def check_config(cfg, z_values_um, obj_section='bacterium', half_um=None):
    """
    Contrôles de cohérence entre simulation, reconstruction et positions balayées.
    obj_section : section décrivant l'objet ('bacterium', 'sphere'…) ; elle fournit x_um et y_um.
    half_um     : demi-extension axiale de l'objet (µm) ; par défaut celle d'une bactérie.
    """
    sim, obj, det = cfg['simulation'], cfg[obj_section], cfg['detection']
    errors, warnings = [], []
    half = half_um if half_um is not None else obj.length_um / 2 + obj.thickness_um / 2
    z_rec = (det.distance_ini_um, det.distance_ini_um + (det.nb_planes - 1) * det.step_um)
    if max(z_values_um) + half + 2 * sim.sim_step_um >= sim.illumination_distance_um:
        errors.append(f"simulation.illumination_distance_um ({sim.illumination_distance_um}) doit dépasser "
                      f"Z max + demi-extension de l'objet ({max(z_values_um) + half:.1f} µm)")
    outside = [z for z in z_values_um if not z_rec[0] <= z <= z_rec[1]]
    if outside:
        warnings.append(f"{len(outside)} position(s) Z hors de la plage reconstruite [{z_rec[0]:.1f}, {z_rec[1]:.1f}] µm : "
                        "ces objets ne peuvent pas être détectés")
    if min(z_values_um) - half < 0:
        warnings.append("des objets dépassent derrière le plan de l'hologramme (Z < demi-extension) : "
                        "leur partie de distance négative n'est pas simulée (colonne 'truncated')")
    field_um = sim.holo_size * sim.pixel_size_um / sim.magnification
    if not (0 <= obj.x_um < field_um and 0 <= obj.y_um < field_um):
        errors.append(f"{obj_section}.x_um / y_um hors du champ [0, {field_um:.1f}[ µm")
    if not 0 <= sim.absorption_percent_per_um < 100:
        errors.append(f"simulation.absorption_percent_per_um = {sim.absorption_percent_per_um} : "
                      "pourcentage attendu dans [0, 100[ (100 % par µm donnerait une absorption infinie)")
    if not 0 <= sim.sensor_noise_std_min <= sim.sensor_noise_std_max:
        errors.append(f"simulation.sensor_noise_std_min / max = {sim.sensor_noise_std_min} / "
                      f"{sim.sensor_noise_std_max} : attendu 0 <= min <= max")
    if det.threshold_type in ('MAX_FOCUS', 'PEAK_ABOVE_BG'):
        # seuil = n × voxel max : n est une fraction
        if not 0 < det.nb_std_threshold <= 1:
            errors.append(f"detection.nb_std_threshold = {det.nb_std_threshold} : avec MAX_FOCUS, "
                          "fraction du voxel max attendue dans ]0, 1]")
        if det.auto_threshold and not 0 < det.auto_nb_std_min < det.auto_nb_std_max <= 1:
            errors.append(f"detection.auto_nb_std_min / max = {det.auto_nb_std_min} / {det.auto_nb_std_max} : "
                          "avec MAX_FOCUS, fractions attendues avec 0 < min < max <= 1")
    mem_gb = det.nb_planes * sim.holo_size ** 2 * 4 / 2 ** 30
    if mem_gb > 1.8:
        warnings.append(f"volume reconstruit de {mem_gb:.1f} Go (float32) : proche de la limite d'un GPU de 6 Go")
    if errors:
        raise ValueError("Configuration incohérente :\n  - " + "\n  - ".join(errors))
    return warnings


def prepare_output(cfg, yaml_path, script_name):
    out = cfg['output'].output_dir or os.path.join(
        REPO_DIR, "output", f"{script_name}_{datetime.datetime.now():%Y_%m_%d_%H_%M_%S}")
    os.makedirs(out, exist_ok=True)
    shutil.copy2(yaml_path, os.path.join(out, os.path.basename(yaml_path)))
    if cfg['output'].save_holograms:
        os.makedirs(os.path.join(out, "holograms"), exist_ok=True)
    if cfg['output'].save_objects:
        os.makedirs(os.path.join(out, "objects"), exist_ok=True)
    return out


# =============================================================================
# SIMULATION D'UNE BACTÉRIE
# =============================================================================

def simulate_single_bacterium(sim, bac, z_um, phi_deg, rng, return_mask=False, theta_deg=None):
    """
    Hologramme (intensité) et hologramme de référence d'une bactérie seule.
      z_um      : distance du centre de la bactérie à l'hologramme
      phi_deg   : angle entre l'axe de la bactérie et l'axe optique (0° = alignée, 90° = couchée)
      theta_deg : azimut, mesuré dans le plan de l'image depuis l'axe x (0° = la bactérie penche
                  vers x, plan x–z ; 90° = vers y, plan y–z). None : azimut donné par bac.tilt_plane.
    L'illumination part de la distance sim.illumination_distance_um (face lointaine de
    l'échantillon), comme dans le simulateur (qui part de z_size·dz + distance volume-caméra).
    """
    N, up = sim.holo_size, sim.upscale_factor
    vox = sim.pixel_size_um * 1e-6 / sim.magnification
    dz = sim.sim_step_um * 1e-6
    Z = z_um * 1e-6
    length, thick = bac.length_um * 1e-6, bac.thickness_um * 1e-6
    D0, m, truncated = _slab(sim, Z, length / 2 + thick / 2)

    # bactérie dans le repère du simulateur : x_simulé = y_image, y_simulé = x_image ; l'azimut du
    # simulateur est donc le complément de l'azimut image (plan x-z de l'image -> theta = 90°).
    theta_img = theta_deg if theta_deg is not None else (0.0 if getattr(bac, 'tilt_plane', 'xz') == 'xz' else 90.0)
    theta = 90.0 - theta_img
    bact = Bacterie(pos_x=bac.y_um * 1e-6, pos_y=bac.x_um * 1e-6, pos_z=Z - D0,
                    thickness=thick, length=length, theta=theta, phi=phi_deg)
    vol_up = cp.zeros((N * up, N * up, m), dtype=cp.float16)
    # up en dernier argument : aligne la grille fine sur la grille finale (voir upscale_grid_shift),
    # sans quoi le moyennage de blocs décale l'objet de (up−1)/(2·up) voxel
    GPU_insert_bact_in_mask_volume(vol_up, bact, vox / up, dz, up)
    return _hologram_from_mask(sim, vol_up, D0, m, truncated, rng, return_mask)


def simulate_single_sphere(sim, sph, z_um, radius_um, rng, return_mask=False):
    """
    Hologramme (intensité) et hologramme de référence d'une sphère seule.
      z_um      : distance du centre de la sphère à l'hologramme
      radius_um : rayon de la sphère
    Mêmes conventions et même modèle physique que simulate_single_bacterium : seul l'objet inséré
    dans le volume masque change (insert_sphere_in_mask_volume au lieu du noyau bactérie).
    """
    N, up = sim.holo_size, sim.upscale_factor
    vox = sim.pixel_size_um * 1e-6 / sim.magnification
    dz = sim.sim_step_um * 1e-6
    Z = z_um * 1e-6
    radius = radius_um * 1e-6
    D0, m, truncated = _slab(sim, Z, radius)

    # x_simulé = y_image, y_simulé = x_image, comme pour la bactérie
    sphere = Sphere(pos_x=sph.y_um * 1e-6, pos_y=sph.x_um * 1e-6, pos_z=Z - D0, radius=radius)
    vol_up = cp.zeros((N * up, N * up, m), dtype=cp.float16)
    # up en dernier argument : aligne la grille fine sur la grille finale (voir upscale_grid_shift)
    insert_sphere_in_mask_volume(vol_up, sphere, vox / up, dz, up)
    return _hologram_from_mask(sim, vol_up, D0, m, truncated, rng, return_mask)


def _slab(sim, Z, half):
    """
    Tranche utile autour d'un objet centré à la distance Z : les plans traversés pas à pas sont
    aux distances D0 + k·dz, k = 0…m−1, avec deux tranches de marge de chaque côté.
    Retourne (D0, m, truncated) ; truncated indique un objet qui dépasse derrière l'hologramme.
    """
    dz = sim.sim_step_um * 1e-6
    D0 = max(0.0, Z - half - 2 * dz)
    m = int(math.ceil((Z + half + 2 * dz - D0) / dz)) + 1
    return D0, m, (Z - half) < 0


def _hologram_from_mask(sim, vol_up, D0, m, truncated, rng, return_mask):
    """
    Partie commune à tous les objets : retour du masque sur-échantillonné à la grille finale,
    puis propagation BPM de l'illumination à travers la tranche utile, et hologramme de référence.

    Comme un film réel nettoyé par sa moyenne : l'illumination est une onde plane sans bruit (le fond
    fixe réel est éliminé par la division), commune à l'hologramme et à la référence ; seul le bruit
    du capteur est simulé, ajouté à l'intensité après propagation. Il change à chaque image : la
    référence, moyenne de sim.reference_nb_planes images, n'en garde que σ/√N.
    """
    N, B, up = sim.holo_size, sim.border, sim.upscale_factor
    NB = N + 2 * B
    pix, M = sim.pixel_size_um * 1e-6, sim.magnification
    lam = sim.wavelength_nm * 1e-9
    lam_m = lam / sim.index_medium
    dz = sim.sim_step_um * 1e-6
    L = sim.illumination_distance_um * 1e-6

    mask = vol_up.reshape(N, up, N, up, m).mean(axis=(1, 3))
    del vol_up
    # Par tranche : déphasage 2π·dz·Δn/λ (indice réel) et atténuation.
    # absorption_percent_per_um = p % d'intensité perdus par µm de matière : après un trajet L il
    # reste (1 − p/100)^L en intensité, ce qui s'écrit exp(−α·L) avec α = −ln(1 − p/100) µm⁻¹.
    # L'amplitude étant la racine de l'intensité, l'exposant par tranche est α·dz/2.
    shift_in_obj = 2.0 * math.pi * dz * (sim.index_object - sim.index_medium) / lam
    reste = 1.0 - sim.absorption_percent_per_um / 100.0
    atten_in_obj = 0.0 if reste >= 1.0 else -math.log(reste) * sim.sim_step_um / 2.0

    sensor_std = (sim.sensor_noise_std_max - sim.sensor_noise_std_min) * rng.random() + sim.sensor_noise_std_min

    # une seule illumination (onde plane sans bruit), commune à l'hologramme et à la référence
    illumination = apodize_border(create_illumination_field_polar(
        field_size_xy_pix=NB, wavelength=lam, pixel_size=pix, magnification=M,
        medium_index=sim.index_medium, number_of_sources=1,
        sources_azimuth_degree=[sim.source_azimuth_deg], sources_polar_degree=[sim.source_polar_deg],
        noise_mean=sim.illumination_mean, noise_std=0.0), B)

    buf = [cp.zeros((NB, NB), cp.complex64), cp.zeros((NB, NB), cp.complex64),
           cp.zeros((NB, NB), cp.complex64), cp.zeros((NB, NB), cp.float32)]

    def prop(f, d):
        if d <= 0:
            return f
        return propagation.propag_angular_spectrum(f, buf[0], buf[1], buf[2], buf[3], lam_m, M, pix, NB, NB, d, 0, 0)

    # hologramme : illumination -> tranche la plus lointaine -> tranches (dz + masque) -> hologramme
    f = prop(illumination, L - (D0 + (m - 1) * dz))
    for k in range(m - 1, -1, -1):
        if k != m - 1:
            f = prop(f, dz)
        plane = pad_centered(mask[:, :, k], [NB, NB])
        f = phase_shift_through_plane(mask_plane=plane, plane_to_shift=f, shift_in_env=0.0,
                                      shift_in_obj=shift_in_obj, attenuation_in_obj=atten_in_obj)
    f = prop(f, D0)
    # bruit du capteur (gaussien) sur l'intensité, après propagation ; le fond sans objet vaut
    # illumination_mean² en intensité
    bg = sim.illumination_mean ** 2
    holo = cp.asnumpy(add_sensor_noise(traitement_holo.intensite(f[B:B + N, B:B + N]), sensor_std, bg, rng))

    # référence : la même illumination propagée sans objet ; moyenne de N images -> bruit capteur
    # σ/√N (la moyenne de N tirages gaussiens indépendants est un tirage d'écart-type σ/√N)
    ref = traitement_holo.intensite(prop(illumination, L)[B:B + N, B:B + N])
    ref = cp.asnumpy(add_sensor_noise(ref, sensor_std / math.sqrt(sim.reference_nb_planes), bg, rng))
    out = {'holo': holo, 'ref': ref, 'sensor_noise_std': sensor_std, 'truncated': truncated}
    if return_mask:
        out['mask'] = cp.asnumpy(mask)
    del f, buf, mask, illumination
    cp.get_default_memory_pool().free_all_blocks()
    return out


# =============================================================================
# DÉTECTION (identique à HOLOTRACKER) ET INTENSITÉ NORMALISÉE
# =============================================================================

def detection_args(sim, det):
    return SimpleNamespace(magnification=sim.magnification, pixel_size=sim.pixel_size_um * 1e-6,
                           distance_ini=det.distance_ini_um * 1e-6, step=det.step_um * 1e-6,
                           high_pass=det.high_pass, low_pass=det.low_pass)


def objects_at_threshold(d_focus, threshold, sim, det, z0_um):
    """Objets détectés pour un seuil donné, sans les grandeurs coûteuses (pour la recherche de seuil)."""
    nz, N = d_focus.shape[0], d_focus.shape[1]
    d_bin = cp.zeros((nz, N, N), dtype=cp.bool_)
    d_labels, n_labels = CCL3D(d_bin, d_focus, type_threshold.THRESHOLD, threshold, det.connectivity)
    obj = hs.object_stats(d_labels, d_focus, None, sim.pixel_size_um / sim.magnification, det.step_um,
                          0.0, 0.0, z0_um=z0_um, d_idx=cp.nonzero(d_bin))
    if obj and (det.min_voxel > 0 or det.max_voxel > 0):
        keep = np.ones(obj['label'].size, dtype=bool)
        if det.min_voxel > 0:
            keep &= obj['nb_vox'] >= det.min_voxel
        if det.max_voxel > 0:
            keep &= obj['nb_vox'] <= det.max_voxel
        obj = {k: v[keep] for k, v in obj.items()}
    del d_bin, d_labels
    cp.get_default_memory_pool().free_all_blocks()
    return obj, int(n_labels)


def search_threshold(d_focus, sim, det, bac, z_um, matching):
    """
    Seuil le plus élevé (en nombre de dispersions du volume de focus : écart-type ou 1,4826·MAD selon
    det.threshold_type) qui donne UN SEUL objet dans
    tout le volume, et cet objet apparié à la bactérie simulée. Exiger l'unicité évite les faux
    appariements : sous le seuil limite, le volume se remplit de grains de bruit dont l'un tombe
    forcément dans la fenêtre de tolérance. Le volume de focus est déjà calculé : seuls le
    seuillage, l'étiquetage et l'appariement sont refaits à chaque essai.

    La recherche part du HAUT : la borne haute est le pic de focus au voisinage de la bactérie
    (au-delà, aucun voxel de l'objet ne passe le seuil), puis on descend par pas doublés jusqu'à
    trouver un seuil qui détecte, et on affine par dichotomie jusqu'à det.auto_precision.
    Descendre est nécessaire : un seuil trop bas ne convient pas non plus, car d'autres objets
    (bruit) apparaissent alors dans le volume.

    Bornes : det.auto_nb_std_min et det.auto_nb_std_max.
    Retourne (nb_std, seuil, nb_std_peak) ; nb_std vaut NaN si aucun seuil de l'intervalle ne
    donne un objet unique apparié à la bactérie.
    """
    nz, N = d_focus.shape[0], d_focus.shape[1]
    m, sd = threshold_center_scale(d_focus, det.threshold_type)    # centre et dispersion du seuil
    dx_um = sim.pixel_size_um / sim.magnification
    z0_um, dz_um = det.distance_ini_um, det.step_um

    # pic du focus dans une boîte autour de la bactérie (tolérances d'appariement)
    def box(center, half, n):
        lo = max(0, int(round((center - half))))
        return slice(lo, min(n, int(round(center + half)) + 1))
    bx = box(bac.x_um / dx_um, matching.tolerance_xy_um / dx_um, N)
    by = box(bac.y_um / dx_um, matching.tolerance_xy_um / dx_um, N)
    bz = box((z_um - z0_um) / dz_um, matching.tolerance_z_um / dz_um, nz)
    peak = float(cp.max(d_focus[bz, by, bx])) if (bz.stop > bz.start and by.stop > by.start and bx.stop > bx.start) else m
    nb_std_peak = (peak - m) / sd if sd > 0 else float('nan')

    truth = {'x_um': np.array([bac.x_um]), 'y_um': np.array([bac.y_um]), 'z_um': np.array([z_um])}
    z_range, field = (z0_um, z0_um + (nz - 1) * dz_um), (N * dx_um, N * dx_um)

    def found(nb_std):
        obj, _ = objects_at_threshold(d_focus, m + nb_std * sd, sim, det, z0_um)
        if not obj or obj['label'].size != 1:       # un seul objet dans tout le volume, sinon c'est du bruit
            return False
        mt = hs.match_ground_truth(obj, truth, matching.tolerance_xy_um, matching.tolerance_z_um, field, z_range)
        return bool(mt['truth_match'][0] >= 0)

    hi = min(det.auto_nb_std_max, nb_std_peak)      # au-dessus : aucun voxel de l'objet ne passe
    if not np.isfinite(hi) or hi <= det.auto_nb_std_min:
        return float('nan'), m + det.nb_std_threshold * sd, nb_std_peak

    # descente depuis le pic, par pas doublés, jusqu'au premier seuil qui détecte
    lo, step = None, det.auto_precision
    while hi - step >= det.auto_nb_std_min:
        cand = hi - step
        if found(cand):
            lo = cand
            break
        hi, step = cand, step * 2                   # cand n'a pas détecté : nouvelle borne haute
    if lo is None:
        return float('nan'), m + det.nb_std_threshold * sd, nb_std_peak

    while hi - lo > det.auto_precision:             # affinage entre lo (détecte) et hi (ne détecte pas)
        mid = 0.5 * (lo + hi)
        if found(mid):
            lo = mid
        else:
            hi = mid
    return lo, m + lo * sd, nb_std_peak


def probe_focus(d_vol, probe, dx_um, z0_um, dz_um):
    """
    Mesures du volume de focus autour de la position VRAIE de l'objet, indépendantes du seuil et
    de la détection. probe = dict(x_um, y_um, z_um, half_xy_um, half_z_um, levels_sigma).
    Fond : médiane et 1,4826·MAD du volume entier (sous-échantillon XY 1/4). Retourne le fond,
    sa dispersion, le pic du focus dans la boîte, le contraste (pic − fond) / σ, et pour chaque
    niveau k de levels_sigma le nombre de voxels de la boîte au-dessus de fond + k·σ.
    """
    nz, ny, nx = d_vol.shape
    med, sig = threshold_center_scale(d_vol, 'MAD')

    def sl(c, h, n):
        return slice(max(0, int(round(c - h))), min(n, int(round(c + h)) + 1))
    box = (sl((probe['z_um'] - z0_um) / dz_um, probe['half_z_um'] / dz_um, nz),
           sl(probe['y_um'] / dx_um, probe['half_xy_um'] / dx_um, ny),
           sl(probe['x_um'] / dx_um, probe['half_xy_um'] / dx_um, nx))
    sub = d_vol[box]
    peak = float(sub.max()) if sub.size else float('nan')
    out = {'focus_bg_median': med, 'focus_bg_sigma': sig, 'focus_peak_box': peak,
           'focus_snr_box': (peak - med) / sig if sig > 0 else float('nan')}
    for k in probe.get('levels_sigma', ()):
        out[f'vox_above_{k:g}sigma'] = int((sub > med + k * sig).astype(cp.float32).sum())
    return out


def detect(holo, ref, sim, det, auto=None, probe=None):
    """
    Chaîne HOLOTRACKER sur un hologramme : nettoyage, propagation, focus, seuil, CCL3D, objets.
    Puis module normalisé (V − moyenne) / (écart-type + ε) aux voxels des objets (le module est
    recalculé après la détection, le focus l'ayant remplacé en place).

    auto = (bac, z_um, matching) : au lieu du seuil fixe det.nb_std_threshold, search_threshold()
    cherche le seuil le plus élevé qui ne laisse qu'un seul objet dans le volume, apparié à la
    bactérie.
    probe : voir probe_focus() ; résultat dans la clé 'probe' (None sinon).
    """
    N = sim.holo_size
    nz = det.nb_planes
    a = detection_args(sim, det)
    lam_m = sim.wavelength_nm * 1e-9 / sim.index_medium
    dx_um = sim.pixel_size_um / sim.magnification
    z0_um = det.distance_ini_um
    cleaned = hs.clean_hologram(holo, ref, det.cleaning, det.division_power)

    d_vol = hs.propagate_module(cleaned, a, lam_m, N, N, nz)
    v_mean, v_std = float(cp.mean(d_vol)), float(cp.std(d_vol))
    if Focus_type[det.focus_type] in focus.COMPLEX_FOCUS_TYPES:
        # critère sur le champ complexe (porteuse retirée) : le module n'a servi qu'à v_mean / v_std
        del d_vol
        cp.get_default_memory_pool().free_all_blocks()
        d_vol = hs.focus_volume_complex(cleaned, a, lam_m, N, N, nz, det.sum_size, Focus_type[det.focus_type])
    else:
        focus.focus(d_vol, d_vol, det.sum_size, Focus_type[det.focus_type])
    probe_res = probe_focus(d_vol, probe, dx_um, z0_um, det.step_um) if probe is not None else None
    nb_std_found = nb_std_peak = float('nan')
    if auto is not None and det.auto_threshold:
        # search_threshold retombe sur det.nb_std_threshold si aucun seuil ne convient
        nb_std_found, threshold, nb_std_peak = search_threshold(d_vol, sim, det, *auto)
    else:
        threshold = calc_threshold(d_vol, det.nb_std_threshold, det.threshold_type)
    d_bin = cp.zeros((nz, N, N), dtype=cp.bool_)
    d_labels, n_labels = CCL3D(d_bin, d_vol, type_threshold.THRESHOLD, threshold, det.connectivity)
    # indices des voxels segmentés, calculés une fois sur le GPU (mêmes indices, même ordre C que
    # np.nonzero sur CPU, ~25 fois plus rapide) : servent aux descripteurs et à la normalisation
    d_idx_all = cp.nonzero(d_bin)
    obj = hs.object_stats(d_labels, d_vol, None, dx_um, det.step_um, 0.0, 0.0, z0_um=z0_um, d_idx=d_idx_all)

    norm_all = None
    if obj:
        d_idx = d_idx_all
        vox_label = cp.asnumpy(d_labels[d_idx]).astype(np.int64)
        if det.min_voxel > 0 or det.max_voxel > 0:
            keep = np.ones(obj['label'].size, dtype=bool)
            if det.min_voxel > 0:
                keep &= obj['nb_vox'] >= det.min_voxel
            if det.max_voxel > 0:
                keep &= obj['nb_vox'] <= det.max_voxel
            obj = {k: v[keep] for k, v in obj.items()}
            sel = np.isin(vox_label, obj['label'])
            vox_label = vox_label[sel]
            d_idx = tuple(ix[cp.asarray(sel)] for ix in d_idx)
        del d_vol, d_labels, d_bin
        d_vol = d_labels = d_bin = None
        cp.get_default_memory_pool().free_all_blocks()
        if obj and obj['label'].size:
            d_mod = hs.propagate_module(cleaned, a, lam_m, N, N, nz)
            vox_norm = (cp.asnumpy(d_mod[d_idx]).astype(np.float64) - v_mean) / (v_std + hs.NORM_EPS)
            del d_mod
            order = np.argsort(vox_label, kind='stable')
            lab_s, norm_s = vox_label[order], vox_norm[order]
            starts = np.flatnonzero(np.r_[True, lab_s[1:] != lab_s[:-1]])
            pos = np.searchsorted(lab_s[starts], obj['label'])
            counts = np.diff(np.r_[starts, lab_s.size])
            obj['norm_min'] = np.minimum.reduceat(norm_s, starts)[pos]
            obj['norm_max'] = np.maximum.reduceat(norm_s, starts)[pos]
            obj['norm_mean'] = (np.add.reduceat(norm_s, starts) / counts)[pos]
            norm_all = {'min': float(vox_norm.min()), 'max': float(vox_norm.max()), 'mean': float(vox_norm.mean())}
        else:
            obj = {}
    del d_vol, d_labels, d_bin
    cp.get_default_memory_pool().free_all_blocks()
    return {'objects': obj, 'cleaned': cleaned, 'threshold': threshold, 'threshold_type': det.threshold_type,
            'probe': probe_res,
            'n_labels': int(n_labels), 'volume_mean': v_mean,
            'volume_std': v_std, 'norm_all': norm_all, 'nb_std_found': nb_std_found, 'nb_std_peak': nb_std_peak,
            'z_range_um': (z0_um, z0_um + (nz - 1) * det.step_um), 'field_um': (N * dx_um, N * dx_um)}


def evaluate(res, bac, z_um, match_cfg, params=None):
    """
    Appariement avec l'objet simulé et grandeurs du rapport (une ligne de CSV).
    params : dict des paramètres balayés (par ex. {'phi_deg': …, 'theta_deg': …} ou
    {'radius_um': …}), inséré juste après la position vraie pour que le CSV se lise
    « vérité terrain, paramètres, détection, erreurs ».

    Ordre des colonnes : position vraie (gt_*), angles, position détectée (det_*), erreurs (err_*),
    intensité des voxels segmentés de l'objet apparié (module normalisé norm_max / norm_mean),
    focus de ces mêmes voxels (focus_peak / focus_mean), nombre de voxels (nb_vox), puis le reste.
    """
    truth = {'x_um': np.array([bac.x_um]), 'y_um': np.array([bac.y_um]), 'z_um': np.array([z_um])}
    obj = res['objects']
    m = hs.match_ground_truth(obj, truth, match_cfg.tolerance_xy_um, match_cfg.tolerance_z_um,
                              res['field_um'], res['z_range_um'])
    res['match'], res['truth'] = m, truth
    nan = float('nan')
    row = {'gt_x_um': bac.x_um, 'gt_y_um': bac.y_um, 'gt_z_um': z_um}
    row.update(params or {})
    row.update({k: nan for k in ('det_x_um', 'det_y_um', 'det_z_um', 'err_x_um', 'err_y_um', 'err_z_um',
                                 'norm_max', 'norm_mean', 'focus_peak', 'focus_mean', 'nb_vox')})
    row.update({'n_detected': int(obj['label'].size) if obj else 0,
                'duplicates': m['metrics']['duplicates'], 'false_positives': m['metrics']['false_positives'],
                'in_volume': bool(m['in_volume'][0]), 'detected': bool(m['truth_match'][0] >= 0),
                'threshold': res['threshold'], 'nb_std_found': res.get('nb_std_found', float('nan')),
                'nb_std_peak': res.get('nb_std_peak', float('nan'))})
    j = int(m['truth_match'][0])
    if j >= 0:
        row.update({'det_x_um': obj['bary_x_um'][j], 'det_y_um': obj['bary_y_um'][j], 'det_z_um': obj['bary_z_um'][j],
                    'err_x_um': obj['bary_x_um'][j] - bac.x_um, 'err_y_um': obj['bary_y_um'][j] - bac.y_um,
                    'err_z_um': obj['bary_z_um'][j] - z_um, 'nb_vox': int(obj['nb_vox'][j]),
                    'focus_peak': obj['focus_peak'][j], 'focus_mean': obj['focus_mean'][j],
                    'focus_sum': obj['focus_sum'][j], 'extent_z_um': obj['extent_z_um'][j],
                    'norm_min': obj['norm_min'][j], 'norm_max': obj['norm_max'][j], 'norm_mean': obj['norm_mean'][j]})
    else:
        row.update({k: nan for k in ('focus_sum', 'extent_z_um', 'norm_min')})
    na = res['norm_all'] or {}
    row.update({'all_norm_min': na.get('min', nan), 'all_norm_max': na.get('max', nan),
                'all_norm_mean': na.get('mean', nan)})
    return row


_INTERACTIVE_WARNED = False


def interactive_backend(show):
    """
    Bascule matplotlib sur un backend interactif quand un affichage est demandé.
    holo_stats force « Agg » (non interactif) à l'import, pour écrire ses PNG sans fenêtre :
    sans cette bascule, plt.show() n'affiche rien et avertit « FigureCanvasAgg is non-interactive ».
    Retourne False si aucun backend interactif n'est disponible (les PNG restent écrits).
    """
    global _INTERACTIVE_WARNED
    import matplotlib
    if not show:
        return False
    if matplotlib.get_backend().lower().startswith(('tkagg', 'qtagg', 'qt5agg', 'wxagg', 'macosx')):
        return True
    for backend in ("TkAgg", "QtAgg"):
        try:
            matplotlib.use(backend, force=True)
            return True
        except Exception:
            continue
    if not _INTERACTIVE_WARNED:
        print("  (aucun backend matplotlib interactif : les figures sont seulement enregistrées en PNG)")
        _INTERACTIVE_WARNED = True
    return False


def plot_objects(res, bac, z_um, phi_deg, dx_um, png_path, noise_std=None, theta_deg=None, label=None):
    """
    Contrôle visuel d'un hologramme, enregistré en PNG (aucune fenêtre) : hologramme nettoyé avec
    les objets détectés et l'objet simulé, vue x–z des détections, résumé chiffré.
    L'erreur de position x, y, z est rappelée dans le titre.
      vert  : objet simulé          cyan   : détection appariée
      orange: détection non appariée (taille du marqueur ∝ nombre de voxels)
    phi_deg / theta_deg valent None pour un objet sans orientation (sphère) ; label ajoute une
    mention au titre (par ex. le rayon).
    """
    import matplotlib.pyplot as plt

    obj, m = res['objects'], res.get('match')
    cleaned = res['cleaned']
    n = cleaned.shape[0]
    ext = [0, n * dx_um, n * dx_um, 0]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), gridspec_kw={'width_ratios': [1.1, 1.1, 0.8]})
    lo, hi = np.percentile(cleaned, [0.5, 99.5])
    axes[0].imshow(cleaned, cmap='gray', vmin=lo, vmax=hi, extent=ext)
    axes[0].set_title("Hologramme nettoyé + objets détectés")
    axes[0].set_xlabel("x (µm)")
    axes[0].set_ylabel("y (µm)")
    axes[1].set_title("Vue x–z des détections")
    axes[1].set_xlabel("x (µm)")
    axes[1].set_ylabel("z (µm)")
    axes[1].set_xlim(0, n * dx_um)
    axes[1].set_ylim(res['z_range_um'][1], res['z_range_um'][0])
    if obj and obj['label'].size:
        matched = (m['det_status'] == 'apparié') if m is not None else np.zeros(obj['label'].size, bool)
        size = 10 + 40 * obj['nb_vox'] / max(obj['nb_vox'].max(), 1)
        # l'objet apparié est tracé en dernier, pour rester visible au milieu des détections parasites
        for sel, color, width, lab in ((~matched, '#f39c12', 1.2, "détection non appariée"),
                                       (matched, '#5dade2', 2.2, "détection appariée")):
            if sel.any():
                for ax, yk in ((axes[0], 'bary_y_um'), (axes[1], 'bary_z_um')):
                    ax.scatter(obj['bary_x_um'][sel], obj[yk][sel], s=size[sel] + 30, facecolors='none',
                               edgecolors=color, linewidths=width, label=lab, zorder=3 if sel is matched else 2)
    for ax, yv in ((axes[0], bac.y_um), (axes[1], z_um)):
        ax.plot([bac.x_um], [yv], marker='+', color='#2ecc71', ms=14, mew=2, ls='none', label="objet simulé")
        ax.legend(fontsize=8, loc='upper right')
    axes[2].axis('off')
    tete = "   ".join(([] if phi_deg is None else [f"φ = {phi_deg:g}°"])
                      + ([] if theta_deg is None else [f"θ = {theta_deg:g}°"])
                      + ([] if label is None else [label]) + [f"Z = {z_um:g} µm"])
    lines = [tete]
    if noise_std is not None:
        lines.append(f"bruit capteur : σ = {noise_std:.4g}")
    lines += [f"objets détectés : {obj['label'].size if obj else 0}",
              f"seuil : {res['threshold']:.4g}" + (f" ({res['threshold_type']})" if res.get('threshold_type') else "")]
    if np.isfinite(res.get('nb_std_found', float('nan'))):
        lines.append(f"  = {res['nb_std_found']:.2f} σ (seuil limite cherché)")
    lines += [f"module : moyenne {res['volume_mean']:.4g}, écart-type {res['volume_std']:.4g}", ""]
    if m is not None and m['truth_match'][0] >= 0:
        j = int(m['truth_match'][0])
        lines += [f"appariée : objet #{obj['label'][j]}",
                  f"  erreur x {obj['bary_x_um'][j] - bac.x_um:+.2f} µm",
                  f"  erreur y {obj['bary_y_um'][j] - bac.y_um:+.2f} µm",
                  f"  erreur z {obj['bary_z_um'][j] - z_um:+.2f} µm",
                  f"  voxels {int(obj['nb_vox'][j])}   module max {obj['norm_max'][j]:.1f} σ"]
    else:
        lines.append("aucune détection appariée")
        if obj and obj['label'].size:
            d = np.hypot(obj['bary_x_um'] - bac.x_um, obj['bary_y_um'] - bac.y_um)
            k = int(np.argmin(d))
            lines += ["détection la plus proche :",
                      f"  écart xy {d[k]:.2f} µm, écart z {obj['bary_z_um'][k] - z_um:+.2f} µm",
                      f"  voxels {int(obj['nb_vox'][k])}"]
    if obj and obj['label'].size:
        lines += ["", f"voxels par objet : médiane {np.median(obj['nb_vox']):.0f}, max {obj['nb_vox'].max():.0f}"]
    axes[2].text(0.0, 0.98, "\n".join(lines), va='top', family='monospace', fontsize=10)
    if m is not None and m['truth_match'][0] >= 0:
        j = int(m['truth_match'][0])
        titre = (f"erreur de position : x {obj['bary_x_um'][j] - bac.x_um:+.2f}   "
                 f"y {obj['bary_y_um'][j] - bac.y_um:+.2f}   z {obj['bary_z_um'][j] - z_um:+.2f} µm")
    else:
        titre = "bactérie non détectée"
    bruit = "" if noise_std is None else f"   —   bruit capteur σ = {noise_std:.3g}"
    fig.suptitle(f"{tete}{bruit}   —   {titre}", fontsize=13)
    fig.tight_layout()
    fig.savefig(png_path, dpi=100)
    plt.close(fig)


# =============================================================================
# CSV
# =============================================================================

class CsvWriter:
    """Écrit les lignes au fil de l'eau : un arrêt en cours de route ne perd pas les résultats déjà calculés."""
    def __init__(self, path):
        self.path, self.fields, self.rows = path, None, []

    def write(self, row):
        self.rows.append(row)
        new = self.fields is None
        if new:
            self.fields = list(row.keys())
        with open(self.path, 'a' if not new else 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=self.fields)
            if new:
                w.writeheader()
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})


def aggregate(rows, group_keys, path):
    """Statistiques par point du balayage (moyenne, écart-type, médiane sur les répétitions) -> CSV."""
    import pandas as pd
    df = pd.DataFrame(rows)
    metrics = ['err_x_um', 'err_y_um', 'err_z_um', 'norm_min', 'norm_max', 'norm_mean',
               'all_norm_min', 'all_norm_max', 'all_norm_mean', 'focus_peak', 'focus_mean', 'focus_sum',
               'threshold', 'nb_std_found', 'nb_std_peak', 'nb_vox', 'extent_z_um', 'n_detected', 'duplicates']
    out = []
    for keys, g in df.groupby(group_keys, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rec = dict(zip(group_keys, keys))
        rec['n_holograms'] = len(g)
        rec['detection_rate'] = float(g['detected'].mean())
        for mcol in metrics:
            v = pd.to_numeric(g[mcol], errors='coerce').dropna()
            rec[f'{mcol}_mean'] = float(v.mean()) if len(v) else float('nan')
            rec[f'{mcol}_std'] = float(v.std(ddof=0)) if len(v) else float('nan')
            rec[f'{mcol}_median'] = float(v.median()) if len(v) else float('nan')
        for e in ('err_x_um', 'err_y_um', 'err_z_um'):
            v = pd.to_numeric(g[e], errors='coerce').dropna().abs()
            rec[f'abs_{e}_mean'] = float(v.mean()) if len(v) else float('nan')
        out.append(rec)
    agg = pd.DataFrame(out)
    agg.to_csv(path, index=False, float_format='%.6g')
    return agg


# =============================================================================
# GRAPHES
# =============================================================================

def plot_report(agg, x_col, x_label, curve_col, curve_label, title, png_path, show, extra_metric=None):
    """
    Figure 3 × 3 :
      erreurs x, y, z (moyenne ± écart-type) ;
      nombre de voxels segmentés de l'objet apparié (moyenne ± écart-type), voxel max du focus de
      l'objet apparié (avec la moyenne et le seuil de détection), taux de détection ;
      nombre d'objets détectés, module normalisé de l'objet apparié, module normalisé de tous les
      voxels segmentés.
    Une courbe par valeur de curve_col (ou une seule si curve_col est None).
    """
    show = interactive_backend(show)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 3, figsize=(17, 13))
    groups = [(None, agg)] if curve_col is None else list(agg.groupby(curve_col, sort=True))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    def triplet(ax, g, x, c, lab, keys, names):
        """max (trait plein), moyenne (tirets), min (pointillés) d'une même grandeur."""
        for key, style, name in zip(keys, ('-', '--', ':'), names):
            ax.plot(x, g[f'{key}_mean'], style, color=c, label=None if lab is None else f"{lab} — {name}")

    for gi, (val, g) in enumerate(groups):
        c = colors[gi % len(colors)]
        lab = None if val is None else f"{curve_label} = {val:g}"
        x = g[x_col].to_numpy()
        for ax, key, name in ((axes[0, 0], 'err_x_um', 'x'), (axes[0, 1], 'err_y_um', 'y'), (axes[0, 2], 'err_z_um', 'z')):
            mu, sd = g[f'{key}_mean'].to_numpy(), g[f'{key}_std'].to_numpy()
            ax.plot(x, mu, '-', color=c, lw=1.3, label=lab)
            ax.fill_between(x, mu - sd, mu + sd, color=c, alpha=0.15)
            ax.axhline(0, color='k', lw=0.5)
            ax.set_ylabel(f"erreur {name} = détecté − simulé (µm)")
            ax.set_title(f"Erreur de position {name}")
        mu, sd = g['nb_vox_mean'].to_numpy(), g['nb_vox_std'].to_numpy()
        axes[1, 0].plot(x, mu, '-o', ms=2.5, color=c, label=lab)
        axes[1, 0].fill_between(x, mu - sd, mu + sd, color=c, alpha=0.15)
        axes[1, 1].plot(x, g['focus_peak_mean'], '-o', ms=2.5, color=c,
                        label=None if lab is None else f"{lab} — voxel max")
        axes[1, 1].plot(x, g['focus_mean_mean'], '--', color=c, label=None if lab is None else f"{lab} — moyenne")
        axes[1, 1].plot(x, g['threshold_mean'], ':', color=c, lw=1,
                        label=None if lab is None else f"{lab} — seuil")
        if extra_metric is None:
            axes[1, 2].plot(x, g['detection_rate'], '-o', ms=2.5, color=c, label=lab)
        else:
            key = extra_metric[0]
            mu, sd = g[f'{key}_mean'].to_numpy(), g[f'{key}_std'].to_numpy()
            axes[1, 2].plot(x, mu, '-o', ms=2.5, color=c, label=lab)
            axes[1, 2].fill_between(x, mu - sd, mu + sd, color=c, alpha=0.15)
        axes[2, 0].plot(x, g['n_detected_mean'], '-', color=c, label=lab)
        triplet(axes[2, 1], g, x, c, lab, ('norm_max', 'norm_mean', 'norm_min'), ("max", "moyenne", "min"))
        triplet(axes[2, 2], g, x, c, lab, ('all_norm_max', 'all_norm_mean', 'all_norm_min'), ("max", "moyenne", "min"))

    axes[1, 0].set_title("Nombre de voxels segmentés de l'objet apparié")
    axes[1, 0].set_ylabel("voxels")
    axes[1, 1].set_title("Voxel max du focus de l'objet apparié\n(plein : max, tirets : moyenne, points : seuil)")
    axes[1, 1].set_ylabel("focus (unités du critère de focus)")
    if extra_metric is None:
        axes[1, 2].set_ylim(-0.05, 1.05)
        axes[1, 2].set_title("Taux de détection (objet retrouvé)")
    else:
        axes[1, 2].set_title(extra_metric[1])
        axes[1, 2].set_ylabel(extra_metric[2])
    axes[2, 0].set_title("Nombre d'objets détectés par hologramme")
    axes[2, 1].set_title("Module normalisé de l'objet apparié\n(plein : max, tirets : moyenne, points : min)")
    axes[2, 1].set_ylabel("(V − moyenne) / (écart-type + ε)")
    axes[2, 2].set_title("Module normalisé de tous les voxels segmentés\n(plein : max, tirets : moyenne, points : min)")
    axes[2, 2].set_ylabel("(V − moyenne) / (écart-type + ε)")
    for ax in axes.ravel():
        ax.set_xlabel(x_label)
        ax.grid(alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(png_path, dpi=110)
    if show:
        plt.show()
    plt.close(fig)


def eta(t0, done, total):
    el = time.perf_counter() - t0
    rest = el / done * (total - done) if done else 0
    return f"{el / 60:.1f} min écoulées, reste ~{rest / 60:.1f} min"

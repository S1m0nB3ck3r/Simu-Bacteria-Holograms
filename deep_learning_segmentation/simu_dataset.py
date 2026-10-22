# -*- coding: utf-8 -*-

"""
Filename: simu_dataset.py

Description:
Jeu d'hologrammes simulés pour le U-Net 3D, au format lisible hors Python (GUI_generate_dataset.py) :
génération, description des paramètres (partagée avec le GUI) et lecture par train.py / train_stage2.py.

Simulation : BPM du simulateur hybride (Hybrid_REAL_SIMU_HOLOGRAMS/real_simulation.simulate_frame : onde
plane, seules les tranches occupées sont traversées pas à pas), fond plat + bruit capteur gaussien ajouté à
l'intensité, SANS fond réel (ce qui généralise au film réel, RAPPORT_UNET3D.md). Masques d'instance
exportés par le simulateur lui-même (real_simulation.bacteria_instance_masks : même noyau d'insertion,
même grille), union contrôlée contre le masque traversé par la BPM sur les premiers hologrammes.

Arborescence produite :
  <dossier>/parameters.yaml                 tous les paramètres (unités SI, comme simulation_gui.py), valeurs
                                            dérivées, description des formats, état de la génération
  <dossier>/generate.log                    journal
  <dossier>/<train|val|test>/index.csv      un hologramme par ligne : nom, N, Δn, σ du bruit, part saturée
  <dossier>/<split>/holograms/holo_XXXXX.bmp      hologramme 8 bits : gris = I / I0 · gris_fond (arrondi, borné
                                                  à 0–255) ; I0 = intensité du fond
  <dossier>/<split>/positions/holo_XXXXX.csv      une bactérie par ligne (m, degrés), z = distance au capteur
  <dossier>/<split>/instances/holo_XXXXX.npz      volume de segmentation d'instance, creux : par bactérie,
                                                  boîte englobante sur la grille du simulateur
  facultatifs : holograms_tiff/holo_XXXXX.tif (intensité float32, fond = illumination moyenne × transmission²),
                instance_volume/holo_XXXXX.tif (volume d'étiquettes uint16 complet, z_size × N × N : 1,6 Go
                pour 1024 × 1024 × 800 plans).

Volume d'instance (.npz) : mask_origin (M, 3) int32 = (tranche k, ligne, colonne) du coin de la boîte,
tranche k <-> distance au capteur z = k · pas ; mask_shape (M, 3) = (K, R, C) ; mask_data = boîtes
concaténées (ordre C), fraction occupée du voxel × frac_levels (uint8, 0 = vide, frac_levels = plein) ;
z0_plane = indice de la première tranche du volume simulé (distance volume-caméra / pas). Un voxel peut
appartenir à deux bactéries qui se croisent. instance_volume() reconstruit le volume d'étiquettes complet.

Utilisation :
  python deep_learning_segmentation/simu_dataset.py <dossier contenant parameters.yaml>

Author: Simon BECKER
Date: 2026-10-19

License:
GNU General Public License v3.0
"""

import argparse
import datetime
import math
import os
import sys
import time

import numpy as np
import yaml

DL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(DL_DIR)
SPLITS = ('train', 'val', 'test')

# =============================================================================
# PARAMÈTRES : (clé, libellé, type, valeur par défaut, aide), par section — source commune du GUI et du YAML
# =============================================================================
# Valeurs par défaut = calibration du 2026-10-18 (Δn = 0,046 ± dispersion du film réel, bâtonnet 1,8 × 0,6 µm
# ± 10 %, pas BPM convergé 0,25 µm), profondeur 0–200 µm comme le jeu des modèles retenus.
LAYOUT = [
    ("CHEMINS", [
        ('output_dir', "Dossier de sortie", 'dir', os.path.join(DL_DIR, 'data', 'dataset_sim'),
         "Reçoit parameters.yaml et les dossiers train/, val/, test/"),
    ]),
    ("SIMULATION", [
        ('n_train', "Hologrammes d'entraînement", 'int', 240, ""),
        ('n_val', "Hologrammes de validation", 'int', 24, ""),
        ('n_test', "Hologrammes de test", 'int', 40, ""),
        ('seed', "Graine", 'int', 0, "Même graine et mêmes paramètres : même jeu"),
    ]),
    ("VOLUME", [
        ('number_of_bacteria_min', "Nombre de bactéries min (entraînement)", 'int', 20,
         "Tiré par hologramme en log-uniforme entre min et max"),
        ('number_of_bacteria_max', "Nombre de bactéries max (entraînement)", 'int', 400, ""),
        ('number_of_bacteria_val_test', "Nombre de bactéries (validation, test)", 'str', '100 200 300 400',
         "Valeurs prises en alternance, séparées par des espaces (validation équilibrée en densité)"),
        ('cluster_fraction', "Bactéries rapprochées (part)", 'float', 0.3,
         "Part des bactéries replacées près d'une autre, à une distance entre centres tirée entre le min et le max "
         "ci-dessous (sinon, tirées au hasard, elles ne se touchent presque jamais)"),
        ('cluster_min_um', "Rapprochées : distance min (µm)", 'float', 0.6,
         "Distance entre centres minimale d'une bactérie rapprochée et de sa voisine (tirage uniforme entre min et max)"),
        ('cluster_max_um', "Rapprochées : distance max (µm)", 'float', 4.0, ""),
        ('avoid_overlap', "Bactéries sans recouvrement", 'bool', True,
         "Deux bactéries (capsules : axe de longueur L − T, rayon T/2) ne s'interpénètrent pas : une bactérie qui en "
         "touche une déjà placée est écartée de celle-ci, dans la direction qui les sépare, jusqu'à ne plus en toucher "
         "aucune. Sans cette option, 47 % des bactéries à 1 µm l'une de l'autre se recouvrent"),
        ('min_gap_um', "Écart minimal entre surfaces (µm)", 'float', 0.0,
         "Avec « sans recouvrement » : distance minimale entre les surfaces de deux bactéries ; 0 : contact permis, "
         "recouvrement interdit ; 0,175 µm (un pixel objet) : masques disjoints même après voxelisation"),
        ('holo_size_xy', "Taille XY hologramme (pixels)", 'int', 1024, "Multiple de 16"),
        ('border', "Apodisation bordure (pixels)", 'int', 256,
         "Bordure de zéros de chaque côté pendant la propagation (pas de repliement) ; 256 suffit, 512 : 2,8× plus lent"),
        ('upscale_factor', "Facteur upscale", 'int', 2,
         "Sur-échantillonnage latéral du masque (anti-crénelage) ; fraction occupée par voxel = 1/upscale²"),
        ('z_size', "Nombre de plans Z", 'int', 800, "Profondeur du volume = plans × pas"),
        ('step_z', "Pas de simulation (m)", 'float', 0.25e-6,
         "Pas BPM et pas Z du volume d'instance ; 0,25 µm : convergé (0,5 µm surestime l'énergie de 3,5 %), "
         "même grille que l'étage 2"),
    ]),
    ("OPTIQUE", [
        ('index_milieu', "Indice milieu", 'float', 1.33, ""),
        ('index_bacterie_min', "Indice bactérie min", 'float', 1.362,
         "Tiré par hologramme entre min et max ; 1,330 + 0,046 = valeur calée sur le film réel, plage 0,032–0,060 "
         "= dispersion mesurée. Le 1,335 de simulation_gui.py donne un signal 8,6 fois trop faible"),
        ('index_bacterie_max', "Indice bactérie max", 'float', 1.390, ""),
        ('transmission_milieu', "Transmission milieu", 'float', 1.0,
         "Transmission en amplitude, uniforme : l'intensité du fond vaut illumination × transmission² (TIFF) ; "
         "le BMP est normalisé par le fond, il n'en dépend pas"),
        ('wavelength', "Longueur d'onde (m)", 'float', 660e-9, "Dans le vide"),
    ]),
    ("CAMÉRA", [
        ('pix_size', "Taille pixel caméra (m)", 'float', 7e-6, ""),
        ('grossissement', "Grossissement", 'float', 40, "Pixel objet = pixel caméra / grossissement"),
        ('distance_volume_camera', "Distance volume-caméra (m)", 'float', 0.0,
         "Distance entre le capteur (plan image ramené dans l'objet) et le début du volume ; les bactéries sont "
         "entre cette distance et distance + plans × pas"),
    ]),
    ("BACTÉRIES", [
        ('longueur_min', "Longueur min (m)", 'float', 1.62e-6, "Longueur bout à bout ; 1,8 µm ± 10 %"),
        ('longueur_max', "Longueur max (m)", 'float', 1.98e-6, ""),
        ('epaisseur_min', "Épaisseur min (m)", 'float', 0.54e-6, "0,6 µm ± 10 %"),
        ('epaisseur_max', "Épaisseur max (m)", 'float', 0.66e-6, ""),
    ]),
    ("ILLUMINATION", [
        ('illumination_mean', "Moyenne illumination", 'float', 1.0,
         "Intensité de l'onde plane (TIFF float32 seulement)"),
        ('sensor_noise_std_min', "Bruit capteur min (σ relatif)", 'float', 0.015,
         "Écart-type relatif au fond, tiré par hologramme ; 0,015 mesuré sur le film réel"),
        ('sensor_noise_std_max', "Bruit capteur max (σ relatif)", 'float', 0.015, ""),
    ]),
    ("SAUVEGARDE", [
        ('bmp_background_gray', "Niveau de gris du fond (BMP)", 'int', 128,
         "Gris = I / I0 × cette valeur ; 128 : contraste jusqu'à I = 2 I0, quantification 0,008 (bruit 0,015)"),
        ('save_hologram_tiff', "Hologramme 32 bits TIFF", 'bool', False, "En plus du BMP"),
        ('save_instance_volume_tiff', "Volume d'instance complet TIFF (lourd)", 'bool', False,
         "Étiquettes uint16 (0 = fond, i = bactérie i du CSV), z_size × N × N : 1,6 Go par hologramme pour "
         "1024 × 1024 × 800 ; le .npz creux contient la même information (et les fractions)"),
        ('check_first', "Hologrammes contrôlés par jeu", 'int', 2,
         "Union des instances comparée au masque traversé par la BPM (~10 s par hologramme)"),
    ]),
]
SPEC = {k: dict(section=s, label=l, kind=t, default=d, help=h) for s, rows in LAYOUT for k, l, t, d, h in rows}
DEFAULTS = {k: v['default'] for k, v in SPEC.items()}
POS_COLUMNS = ['object', 'x_m', 'y_m', 'z_m', 'theta_deg', 'phi_deg', 'length_m', 'thickness_m', 'x_px', 'y_px', 'z_plane']


def coerce(p):
    """Paramètres complétés par les défauts et convertis au bon type (lève ValueError)."""
    out = dict(DEFAULTS)
    out.update({k: v for k, v in (p or {}).items() if k in SPEC})
    for k, s in SPEC.items():
        v = out[k]
        if s['kind'] == 'int':
            out[k] = int(round(float(str(v).replace(',', '.'))))
        elif s['kind'] == 'float':
            out[k] = float(str(v).replace(',', '.'))
        elif s['kind'] == 'bool':
            out[k] = v if isinstance(v, bool) else str(v).lower() in ('1', 'true', 'oui', 'yes')
        else:
            out[k] = str(v)
    return out


def derived(p):
    """Grandeurs dérivées, en unités SI, et contrôle de cohérence (liste d'erreurs)."""
    dx = p['pix_size'] / p['grossissement']
    depth = p['z_size'] * p['step_z']
    margin = (p['longueur_max'] + p['epaisseur_max']) / 2
    d = dict(pixel_objet_m=dx, champ_xy_m=p['holo_size_xy'] * dx, profondeur_volume_m=depth,
             z_bacteries_m=[p['distance_volume_camera'] + margin, p['distance_volume_camera'] + depth - margin],
             delta_n=[p['index_bacterie_min'] - p['index_milieu'], p['index_bacterie_max'] - p['index_milieu']],
             longueur_onde_milieu_m=p['wavelength'] / p['index_milieu'],
             z0_plane=int(round(p['distance_volume_camera'] / p['step_z'])),
             frac_levels=p['upscale_factor'] ** 2)
    err = []
    if p['holo_size_xy'] % 16 or p['holo_size_xy'] < 64:
        err.append("taille XY : multiple de 16 (et ≥ 64)")
    if d['z_bacteries_m'][1] <= d['z_bacteries_m'][0]:
        err.append("volume moins profond qu'une bactérie")
    if not 0 <= d['delta_n'][0] <= d['delta_n'][1]:
        err.append("indice bactérie : milieu ≤ min ≤ max")
    if not 0 < p['epaisseur_min'] <= p['epaisseur_max'] or not 0 < p['longueur_min'] <= p['longueur_max']:
        err.append("tailles : 0 < min ≤ max")
    if not 0 <= p['sensor_noise_std_min'] <= p['sensor_noise_std_max']:
        err.append("bruit : 0 ≤ min ≤ max")
    if not 1 <= p['upscale_factor'] <= 15:
        err.append("facteur upscale : 1 à 15 (fractions sur 8 bits)")
    if not 1 <= p['number_of_bacteria_min'] <= p['number_of_bacteria_max']:
        err.append("nombre de bactéries : 1 ≤ min ≤ max")
    if not 1 <= p['bmp_background_gray'] <= 254:
        err.append("niveau de gris du fond : 1 à 254")
    try:
        if not [int(x) for x in p['number_of_bacteria_val_test'].split()]:
            raise ValueError
    except ValueError:
        err.append("nombre de bactéries (validation, test) : entiers séparés par des espaces")
    if p['distance_volume_camera'] < 0:
        err.append("distance volume-caméra ≥ 0")
    if not 0 <= p['cluster_min_um'] <= p['cluster_max_um']:
        err.append("bactéries rapprochées : 0 ≤ distance min ≤ distance max")
    if p['min_gap_um'] < 0:
        err.append("écart minimal entre surfaces ≥ 0")
    return d, err


# =============================================================================
# parameters.yaml
# =============================================================================

FORMATS = {
    'hologramme_bmp': "holograms/holo_XXXXX.bmp, 8 bits : gris = arrondi(I / I0 × bmp_background_gray), borné à "
                      "0–255 ; I0 = intensité du fond. Hologramme normalisé (fond 1) = gris / bmp_background_gray",
    'positions_csv': "positions/holo_XXXXX.csv : object (= étiquette du volume d'instance − 1), x_m, y_m (colonne, "
                     "ligne × pixel objet, repère HOLOTRACKER), z_m (distance au capteur, positive : plan de "
                     "refocalisation), theta_deg (azimut), phi_deg (angle à l'axe optique), length_m (bout à bout), "
                     "thickness_m, x_px, y_px, z_plane (indice dans le volume : (z_m − distance volume-caméra) / pas)",
    'instances_npz': "instances/holo_XXXXX.npz : mask_origin (M, 3) = (k, ligne, colonne) du coin de la boîte de "
                     "chaque bactérie, tranche k <-> z = k × pas (distance au capteur) ; mask_shape (M, 3) ; "
                     "mask_data uint8 = boîtes concaténées (ordre C), fraction occupée × frac_levels ; z0_plane = "
                     "première tranche du volume ; voxel de l'instance si fraction ≥ 0,5",
    'index_csv': "index.csv : hologram, name, n_bacteria, delta_n, sensor_noise_std, bmp_saturated_fraction, "
                 "n_separated (bactéries écartées pour supprimer un recouvrement), max_shift_um (plus grand déplacement)",
    'hologramme_tiff': "holograms_tiff/holo_XXXXX.tif (facultatif) : intensité float32, fond = illumination × "
                       "transmission²",
    'volume_instance_tiff': "instance_volume/holo_XXXXX.tif (facultatif) : uint16 (z_size, N, N), 0 = fond, "
                            "i = bactérie object = i − 1 (fraction ≥ 0,5 ; recouvrement : la plus grande fraction)",
}


def _round(v):
    if isinstance(v, float):
        return float(f"{v:.10g}")
    if isinstance(v, list):
        return [_round(x) for x in v]
    return v


def _scalar(v):
    return yaml.safe_dump(v, allow_unicode=True, default_flow_style=True, width=10 ** 6).strip().removesuffix('...').strip()


def write_yaml(out_dir, p, status, counts=None, extra=None):
    """parameters.yaml : une clé par ligne, libellé du GUI en commentaire."""
    p = dict(p, output_dir=os.path.abspath(out_dir))
    d, _ = derived(p)
    lines = ["# Jeu d'hologrammes simulés pour le U-Net 3D (deep_learning_segmentation/GUI_generate_dataset.py)",
             "# Unités SI (m) ; angles en degrés. Relu par train.py et train_stage2.py (simu_dataset.read_parameters).", ""]
    for sec, rows in LAYOUT:
        lines.append(f"{sec.lower()}:")
        for k, label, *_ in rows:
            lines.append(f"  {k}: {_scalar(_round(p[k]))}  # {label}")
    lines.append("")
    doc = dict(derive={k: _round(v) for k, v in d.items()})
    doc['simulation_bpm'] = dict(
        methode="BPM multi-tranches (real_simulation.simulate_frame) : onde plane, tranches occupées traversées "
                "au pas step_z, vide franchi en une propagation par spectre angulaire ; bruit capteur gaussien "
                "ajouté à l'intensité après propagation ; aucun fond réel",
        orientation="isotrope sur la demi-sphère (theta uniforme 0–360°, cos(phi) uniforme)",
        positions="x, y uniformes sur le champ, z uniforme dans derive.z_bacteries_m ; cluster_fraction replacées "
                  "près d'une autre (distance entre centres uniforme entre cluster_min_um et cluster_max_um, direction "
                  "isotrope) ; avoid_overlap : une bactérie qui en touche une déjà placée (distance des axes < somme des "
                  "rayons + min_gap_um) en est écartée jusqu'à ne plus en toucher aucune",
        tailles="longueur et épaisseur uniformes et indépendantes par bactérie (longueur ≥ épaisseur)",
        indice="Δn = indice bactérie − indice milieu, uniforme par hologramme",
        bruit="σ uniforme par hologramme entre sensor_noise_std_min et max, relatif au fond")
    doc['formats'] = FORMATS
    doc['generation'] = dict(etat=status, date=datetime.datetime.now().isoformat(timespec='seconds'),
                             hologrammes=counts or {}, **(extra or {}))
    with open(os.path.join(out_dir, 'parameters.yaml'), 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")
        yaml.safe_dump(doc, f, allow_unicode=True, sort_keys=False, default_flow_style=False, width=110)


def read_parameters(dataset_dir):
    """Paramètres à plat (clés du GUI) + 'derive' + 'generation' de <dataset_dir>/parameters.yaml."""
    with open(os.path.join(dataset_dir, 'parameters.yaml'), encoding='utf-8') as f:
        doc = yaml.safe_load(f)
    flat = {}
    for sec, rows in LAYOUT:
        flat.update(doc.get(sec.lower(), {}))
    flat.setdefault('avoid_overlap', False)       # jeux antérieurs au test de recouvrement (2026-10-22)
    p = coerce(flat)
    p['derive'] = doc.get('derive') or derived(p)[0]
    p['generation'] = doc.get('generation', {})
    return p


def is_dataset(path):
    """Vrai si path est un dossier de jeu (parameters.yaml) ou un de ses dossiers train/val/test."""
    return os.path.isfile(os.path.join(path, 'parameters.yaml')) or (
        os.path.isfile(os.path.join(path, 'index.csv')) and os.path.isfile(os.path.join(os.path.dirname(os.path.abspath(path)), 'parameters.yaml')))


def volume_range_um(path):
    """
    Plage de profondeur du volume simulé (µm, distances au capteur) : (distance volume-caméra, distance + plans × pas).
    path : dossier du jeu ou un de ses dossiers train/val/test. None si ce n'est pas un jeu de GUI_generate_dataset.py.
    """
    path = os.path.abspath(path)
    root = path if os.path.isfile(os.path.join(path, 'parameters.yaml')) else os.path.dirname(path)
    if not os.path.isfile(os.path.join(root, 'parameters.yaml')):
        return None
    p = read_parameters(root)
    z0 = p['distance_volume_camera'] * 1e6
    return z0, z0 + p['z_size'] * p['step_z'] * 1e6


def grid_range(z_min_um, z_max_um, step_um, mult):
    """Grille de reconstruction couvrant [z_min, z_max] : (z0, nombre de plans multiple de mult)."""
    return float(z_min_um), int(math.ceil(((z_max_um - z_min_um) / step_um + 1) / mult) * mult)


def resolve_range(data_path, z_min_um=None, z_max_um=None, legacy=(0.0, 207.0)):
    """z min et z max de la reconstruction : valeurs données, sinon celles du volume du jeu, sinon legacy (.npz)."""
    auto = volume_range_um(data_path) or legacy
    return (auto[0] if z_min_um is None else float(z_min_um)), (auto[1] if z_max_um is None else float(z_max_um))


def describe(dataset_dir):
    """Résumé d'une ligne ou deux d'un jeu (GUI_train.py)."""
    p = read_parameters(dataset_dir)
    d, g = p['derive'], p['generation']
    counts = ", ".join(f"{k} {v}" for k, v in (g.get('hologrammes') or {}).items())
    return (f"{counts or 'aucun hologramme'} ({g.get('etat', '?')}) ; {p['holo_size_xy']} px de {d['pixel_objet_m'] * 1e6:.3f} µm, "
            f"λ {p['wavelength'] * 1e9:.0f} nm, n {p['index_milieu']} ; z {d['z_bacteries_m'][0] * 1e6:.0f}–"
            f"{d['z_bacteries_m'][1] * 1e6:.0f} µm, pas {p['step_z'] * 1e6:g} µm ; N {p['number_of_bacteria_min']}–"
            f"{p['number_of_bacteria_max']} ; Δn {d['delta_n'][0]:.3f}–{d['delta_n'][1]:.3f} ; "
            f"bactéries {p['longueur_min'] * 1e6:.2f}–{p['longueur_max'] * 1e6:.2f} × {p['epaisseur_min'] * 1e6:.2f}–"
            f"{p['epaisseur_max'] * 1e6:.2f} µm")


# =============================================================================
# GÉNÉRATION
# =============================================================================

def rod_axes(theta_deg, phi_deg):
    """Axes unitaires (n, 3) dans le repère image (x = colonne, y = ligne, z) : même convention que le simulateur
    (real_simulation : Bacterie(pos_x = y, pos_y = x, theta = 90° − θ) ; libs/simu_hologram : axe (sin φ cos θ_s,
    sin φ sin θ_s, cos φ) dans son repère)."""
    th, ph = np.radians(theta_deg), np.radians(phi_deg)
    return np.stack([np.sin(ph) * np.cos(th), np.sin(ph) * np.sin(th), np.cos(ph)], axis=-1)


def segment_distance(c1, a1, h1, c2, a2, h2):
    """
    Distance minimale entre le segment c1 ± h1·a1 et les segments c2[k] ± h2[k]·a2[k] (vectorisé sur k), et vecteur
    (k, 3) du point le plus proche du second vers le point le plus proche du premier. Méthode d'Ericson
    (Real-Time Collision Detection, 5.1.9), segments dégénérés (h = 0 : sphère) compris.
    """
    p1, q1 = c1 - h1 * a1, c1 + h1 * a1
    p2, q2 = c2 - h2[:, None] * a2, c2 + h2[:, None] * a2
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e = float(d1 @ d1), np.einsum('ij,ij->i', d2, d2)
    f = np.einsum('ij,ij->i', d2, r)
    eps = 1e-12
    if a <= eps:
        s = np.zeros(len(c2))
        t = np.where(e > eps, np.clip(f / np.maximum(e, eps), 0, 1), 0.0)
    else:
        c = r @ d1
        b = d2 @ d1
        denom = a * e - b * b
        s = np.where(denom > eps, np.clip((b * f - c * e) / np.maximum(denom, eps), 0, 1), 0.0)
        t = np.where(e > eps, (b * s + f) / np.maximum(e, eps), 0.0)
        lo, hi = t < 0, t > 1
        t = np.clip(t, 0, 1)
        s = np.where(lo, np.clip(-c / a, 0, 1), np.where(hi, np.clip((b - c) / a, 0, 1), s))
        s = np.where(e > eps, s, np.clip(-c / a, 0, 1))
    v = (p1 + s[:, None] * d1) - (p2 + t[:, None] * d2)
    return np.linalg.norm(v, axis=1), v


def separate_bacteria(b, rng, field_um, zr, gap_um=0.0, max_moves=200):
    """
    Supprime les recouvrements (n, 7) : les bactéries sont placées une à une ; une bactérie qui touche une bactérie
    déjà placée (distance des axes < somme des rayons + gap_um) est écartée de la plus enfoncée, le long de la
    droite qui joint leurs points les plus proches, de ce qui manque (+ 0,01 µm), jusqu'à n'en toucher aucune ; si
    elle reste bloquée (bord du volume, voisines de tous côtés), elle est retirée ailleurs au hasard.
    Retourne (b corrigé, nombre de bactéries déplacées, plus grand déplacement en µm).
    """
    b = b.astype(np.float64).copy()
    ax = rod_axes(b[:, 3], b[:, 4])
    h = np.maximum((b[:, 5] - b[:, 6]) / 2, 0.0)
    rad = b[:, 6] / 2
    lo = np.array([0.0, 0.0, zr[0]])
    hi = np.array([field_um, field_um, zr[1]])
    n_moved, max_shift = 0, 0.0
    for i in range(len(b)):
        start = b[i, :3].copy()
        if i == 0:
            continue
        for move in range(max_moves * 2):
            dist, v = segment_distance(b[i, :3], ax[i], h[i], b[:i, :3], ax[:i], h[:i])
            deficit = rad[i] + rad[:i] + gap_um - dist
            k = int(np.argmax(deficit))
            if deficit[k] <= 0:
                break
            if move >= max_moves:                     # bloquée : nouvelle position au hasard
                b[i, :3] = rng.uniform(lo, hi)
                continue
            u = v[k] / dist[k] if dist[k] > 1e-9 else rng.normal(size=3)
            u /= np.linalg.norm(u)
            b[i, :3] = np.clip(b[i, :3] + u * (deficit[k] + 0.01), lo, hi)
        shift = float(np.linalg.norm(b[i, :3] - start))
        if shift > 0:
            n_moved += 1
            max_shift = max(max_shift, shift)
    return b.astype(np.float32), n_moved, max_shift


def count_overlaps(b, gap_um=0.0):
    """Nombre de paires de bactéries (n, 7) qui se recouvrent (distance des axes < somme des rayons + gap_um)."""
    ax = rod_axes(b[:, 3], b[:, 4])
    h = np.maximum((b[:, 5] - b[:, 6]) / 2, 0.0)
    n = 0
    for i in range(1, len(b)):
        dist, _ = segment_distance(b[i, :3].astype(np.float64), ax[i], h[i], b[:i, :3].astype(np.float64), ax[:i], h[:i])
        n += int(((b[i, 6] + b[:i, 6]) / 2 + gap_um - dist > 1e-6).sum())
    return n


def draw_bacteria(n, rng, p, d, stats=None):
    """
    (n, 7) : x, y, z (µm, z = distance au capteur), θ, φ (degrés), longueur, épaisseur (µm). Avec avoid_overlap,
    les recouvrements sont supprimés (separate_bacteria) ; stats (dict) reçoit n_separated et max_shift_um.
    """
    field = d['champ_xy_m'] * 1e6
    zr = [z * 1e6 for z in d['z_bacteries_m']]
    x, y, z = rng.uniform(0, field, n), rng.uniform(0, field, n), rng.uniform(*zr, n)
    n_cl = int(round(p['cluster_fraction'] * n))
    if n_cl and n > 1:
        moved = rng.choice(n, min(n_cl, n - 1), replace=False)
        anchors = np.setdiff1d(np.arange(n), moved)
        for i in moved:
            j = rng.choice(anchors)
            v = rng.normal(size=3)
            v *= rng.uniform(p['cluster_min_um'], p['cluster_max_um']) / np.linalg.norm(v)
            x[i], y[i], z[i] = np.clip(x[j] + v[0], 0, field), np.clip(y[j] + v[1], 0, field), np.clip(z[j] + v[2], *zr)
    theta = rng.uniform(0, 360, n)
    phi = np.degrees(np.arccos(rng.random(n)))
    T = rng.uniform(p['epaisseur_min'], p['epaisseur_max'], n) * 1e6
    L = np.maximum(rng.uniform(p['longueur_min'], p['longueur_max'], n) * 1e6, T)
    b = np.stack([x, y, z, theta, phi, L, T], axis=1).astype(np.float32)
    n_sep, shift = 0, 0.0
    if p.get('avoid_overlap') and n > 1:
        b, n_sep, shift = separate_bacteria(b, rng, field, zr, p.get('min_gap_um', 0.0))
    if stats is not None:
        stats.update(n_separated=n_sep, max_shift_um=round(shift, 3))
    return b


def labels_volume(inst, p, d):
    """Volume d'étiquettes uint16 (z_size, N, N) : fraction ≥ 0,5 ; recouvrement : la plus grande fraction."""
    N, Z, z0 = p['holo_size_xy'], p['z_size'], d['z0_plane']
    lab = np.zeros((Z, N, N), np.uint16)
    best = np.zeros((Z, N, N), np.float16)
    for i, ((k0, r0, c0), m) in enumerate(inst):
        if m.size == 0:
            continue
        k0 -= z0
        lo = (max(k0, 0), r0, c0)
        hi = (min(k0 + m.shape[0], Z), r0 + m.shape[1], c0 + m.shape[2])
        if hi[0] <= lo[0]:
            continue
        sub = m[lo[0] - k0:hi[0] - k0]
        sl = tuple(slice(a, b) for a, b in zip(lo, hi))
        win = (sub >= 0.5) & (sub > best[sl])
        lab[sl][win] = i + 1
        best[sl][win] = sub[win]
    return lab


def generate(out_dir, log=print):
    """Génère le jeu décrit par <out_dir>/parameters.yaml. Fichier STOP dans out_dir : arrêt propre."""
    import cupy as cp
    import pandas as pd
    from PIL import Image
    import tifffile
    sys.path.insert(0, os.path.join(REPO_DIR, 'Hybrid_REAL_SIMU_HOLOGRAMS'))
    import real_simulation as rs
    from generate_dataset import check_masks

    p = read_parameters(out_dir)
    d, err = derived(p)
    if err:
        raise ValueError("paramètres incohérents : " + " ; ".join(err))
    N = p['holo_size_xy']
    dx, lam = d['pixel_objet_m'], p['wavelength']
    prop = rs.Propagator(N, p['border'], dx, lam / p['index_milieu'])
    sc = dict(rs.DEFAULT_CONFIG['simulation'], step_um=p['step_z'] * 1e6, upscale_factor=p['upscale_factor'],
              border=p['border'], length_um=(p['longueur_min'] + p['longueur_max']) / 2 * 1e6,
              thickness_um=(p['epaisseur_min'] + p['epaisseur_max']) / 2 * 1e6)
    gray, lev = p['bmp_background_gray'], d['frac_levels']
    i0 = p['illumination_mean'] * p['transmission_milieu'] ** 2
    nlist = [int(x) for x in p['number_of_bacteria_val_test'].split()]
    rng = np.random.default_rng(p['seed'])
    stop_file = os.path.join(out_dir, 'STOP')
    counts, t_start = {}, time.time()
    log(f"pixel objet {dx * 1e6:.4f} µm, champ {d['champ_xy_m'] * 1e6:.1f} µm, z {d['z_bacteries_m'][0] * 1e6:.1f}–"
        f"{d['z_bacteries_m'][1] * 1e6:.1f} µm, Δn {d['delta_n'][0]:.3f}–{d['delta_n'][1]:.3f}, pas {p['step_z'] * 1e6:g} µm")
    write_yaml(out_dir, p, 'en cours', counts)
    for split in SPLITS:
        n_holo = p[f'n_{split}']
        if n_holo <= 0:
            continue
        sd = os.path.join(out_dir, split)
        dirs = ['holograms', 'positions', 'instances'] + (['holograms_tiff'] if p['save_hologram_tiff'] else []) \
            + (['instance_volume'] if p['save_instance_volume_tiff'] else [])
        for sub in dirs:
            os.makedirs(os.path.join(sd, sub), exist_ok=True)
        index, t0 = [], time.time()
        for i in range(n_holo):
            if os.path.exists(stop_file):
                os.remove(stop_file)
                counts[split] = i
                pd.DataFrame(index).to_csv(os.path.join(sd, 'index.csv'), index=False, float_format='%.6g')
                write_yaml(out_dir, p, 'interrompu', counts)
                log(f"ARRÊT demandé (fichier STOP) : {split} {i}/{n_holo}")
                return counts
            if split == 'train':
                nb = int(round(math.exp(rng.uniform(math.log(p['number_of_bacteria_min']),
                                                    math.log(p['number_of_bacteria_max'])))))
            else:
                nb = nlist[i % len(nlist)]
            st = {}
            bacts = draw_bacteria(nb, rng, p, d, st)
            dn = float(rng.uniform(*d['delta_n']))
            sigma = float(rng.uniform(p['sensor_noise_std_min'], p['sensor_noise_std_max']))
            blist = [dict(x_um=float(b[0]), y_um=float(b[1]), z_um=float(b[2]), theta_deg=float(b[3]),
                          phi_deg=float(b[4]), length_um=float(b[5]), thickness_um=float(b[6])) for b in bacts]
            planes = {} if i < p['check_first'] else None
            sim = cp.asnumpy(rs.simulate_frame(blist, prop, dict(sc, delta_n=dn), dx, lam, plane_masks=planes))
            holo = sim + rng.normal(0.0, sigma, sim.shape).astype(np.float32)          # fond = 1
            inst = rs.bacteria_instance_masks(blist, sc, dx, N)
            if planes is not None:
                bad = check_masks(inst, planes, N)
                nvox = sum(int((q > 0).sum()) for q in planes.values())
                log(f"  contrôle {split} {i} : union des instances / masque simulé, {bad} voxels différents sur {nvox}")
                if bad > max(2, 1e-4 * nvox):          # quelques voxels pile sur la surface (arrondis)
                    raise RuntimeError(f"{split} {i} : union des instances ≠ masque simulé ({bad} voxels)")
            name = f"holo_{i:05d}"
            g = np.rint(holo * gray)
            sat = float(((g < 0) | (g > 255)).mean())
            Image.fromarray(np.clip(g, 0, 255).astype(np.uint8)).save(os.path.join(sd, 'holograms', name + '.bmp'))
            if p['save_hologram_tiff']:
                tifffile.imwrite(os.path.join(sd, 'holograms_tiff', name + '.tif'), (holo * i0).astype(np.float32))
            pos = pd.DataFrame({'object': np.arange(nb), 'x_m': bacts[:, 0] * 1e-6, 'y_m': bacts[:, 1] * 1e-6,
                                'z_m': bacts[:, 2] * 1e-6, 'theta_deg': bacts[:, 3], 'phi_deg': bacts[:, 4],
                                'length_m': bacts[:, 5] * 1e-6, 'thickness_m': bacts[:, 6] * 1e-6,
                                'x_px': np.floor(bacts[:, 0] / (dx * 1e6)).astype(int),
                                'y_px': np.floor(bacts[:, 1] / (dx * 1e6)).astype(int),
                                'z_plane': np.floor(bacts[:, 2] / (p['step_z'] * 1e6)).astype(int) - d['z0_plane']},
                               columns=POS_COLUMNS)
            pos.to_csv(os.path.join(sd, 'positions', name + '.csv'), index=False, float_format='%.9g')
            np.savez_compressed(
                os.path.join(sd, 'instances', name + '.npz'),
                mask_origin=np.array([o for o, _ in inst], np.int32).reshape(-1, 3),
                mask_shape=np.array([m.shape for _, m in inst], np.int32).reshape(-1, 3),
                mask_data=np.concatenate([np.rint(m.astype(np.float32) * lev).astype(np.uint8).ravel() for _, m in inst]
                                         + [np.zeros(0, np.uint8)]),
                frac_levels=np.int32(lev), step_m=np.float64(p['step_z']), pixel_m=np.float64(dx),
                z0_plane=np.int32(d['z0_plane']))
            if p['save_instance_volume_tiff']:
                tifffile.imwrite(os.path.join(sd, 'instance_volume', name + '.tif'), labels_volume(inst, p, d),
                                 bigtiff=True, compression='zlib')
            index.append(dict(hologram=i, name=name, n_bacteria=nb, delta_n=dn, sensor_noise_std=sigma, **st,
                              bmp_saturated_fraction=sat))
            el = time.time() - t0
            sep = f", {st['n_separated']} écartées" if st.get('n_separated') else ''
            log(f"[{split}] {i + 1}/{n_holo}  N = {nb}, Δn = {dn:.4f}{sep}  ({el / (i + 1):.2f} s/hologramme)")
            cp.get_default_memory_pool().free_all_blocks()
        pd.DataFrame(index).to_csv(os.path.join(sd, 'index.csv'), index=False, float_format='%.6g')
        counts[split] = n_holo
        write_yaml(out_dir, p, 'en cours', counts)
    write_yaml(out_dir, p, 'terminé', counts, dict(duree_s=round(time.time() - t_start, 1)))
    log(f"terminé : {counts}, {time.time() - t_start:.0f} s")
    return counts


# =============================================================================
# LECTURE (train.py, train_stage2.py)
# =============================================================================

def instance_masks(npz_path):
    """Instances d'un hologramme : liste de (origine (k, r, c), fractions float32 (K, R, C))."""
    z = np.load(npz_path)
    lev = float(z['frac_levels'])
    out, o = [], 0
    for org, (K, R, C) in zip(z['mask_origin'], z['mask_shape']):
        n = int(K) * int(R) * int(C)
        out.append((tuple(int(v) for v in org), z['mask_data'][o:o + n].reshape(int(K), int(R), int(C)).astype(np.float32) / lev))
        o += n
    return out


def instance_volume(npz_path, dataset_dir=None):
    """Volume d'étiquettes complet uint16 (z_size, N, N) d'un hologramme (instances/holo_XXXXX.npz)."""
    dataset_dir = dataset_dir or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(npz_path))))
    p = read_parameters(dataset_dir)
    return labels_volume(instance_masks(npz_path), p, derived(p)[0])


def load_split(split_dir, limit=None, masks=True):
    """
    Hologrammes d'un dossier train/, val/ ou test/, au format des .npz de generate_dataset.py : dict h (I/I0 − 1,
    float16), bacts (N, 7) µm et degrés, meta (optique, grille des masques, tailles nominales…), et, si masks,
    mask_origin, mask_shape, mask_data (fractions × frac_levels).
    """
    import pandas as pd
    from PIL import Image
    split_dir = os.path.abspath(split_dir)
    p = read_parameters(os.path.dirname(split_dir))
    d = p['derive']
    idx = pd.read_csv(os.path.join(split_dir, 'index.csv'))
    if limit:
        idx = idx.iloc[:limit]
    gray = float(p['bmp_background_gray'])
    base = dict(holo_size=p['holo_size_xy'], dx_um=d['pixel_objet_m'] * 1e6, wavelength_um=p['wavelength'] * 1e6,
                index_medium=p['index_milieu'], background='sim', format='bmp',
                length_um=(p['longueur_min'] + p['longueur_max']) / 2 * 1e6,
                thickness_um=(p['epaisseur_min'] + p['epaisseur_max']) / 2 * 1e6,
                z_range_um=[z * 1e6 for z in d['z_bacteries_m']], sim_step_um=p['step_z'] * 1e6,
                mask_grid=dict(dz_um=p['step_z'] * 1e6, dx_um=d['pixel_objet_m'] * 1e6, frac_levels=int(d['frac_levels'])))
    items = []
    for r in idx.itertuples():
        f = os.path.join(split_dir, 'holograms', r.name + '.bmp')
        h = (np.asarray(Image.open(f), dtype=np.float32) / gray - 1.0).astype(np.float16)
        pos = pd.read_csv(os.path.join(split_dir, 'positions', r.name + '.csv'))
        bacts = np.stack([pos.x_m * 1e6, pos.y_m * 1e6, pos.z_m * 1e6, pos.theta_deg, pos.phi_deg,
                          pos.length_m * 1e6, pos.thickness_m * 1e6], axis=1).astype(np.float32).reshape(-1, 7)
        meta = dict(base, n=int(r.n_bacteria), delta_n=float(r.delta_n), sensor_noise_std=float(r.sensor_noise_std))
        it = dict(h=h, bacts=bacts, meta=meta, file=f)
        if masks:
            z = np.load(os.path.join(split_dir, 'instances', r.name + '.npz'))
            it.update(mask_origin=z['mask_origin'], mask_shape=z['mask_shape'], mask_data=z['mask_data'])
        items.append(it)
    return items


def main():
    ap = argparse.ArgumentParser(description="Génère le jeu décrit par <dossier>/parameters.yaml")
    ap.add_argument('out_dir', help="dossier du jeu : contient parameters.yaml (écrit par GUI_generate_dataset.py) ; reçoit train/, val/, test/")
    args = ap.parse_args()
    generate(args.out_dir, log=lambda m: print(m, flush=True))


if __name__ == '__main__':
    main()

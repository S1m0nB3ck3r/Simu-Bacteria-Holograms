# -*- coding: utf-8 -*-

"""
Filename: real_simulation.py

Description:
Simulations « réalistes » construites sur un film holographique réel et ses trajectoires
HOLOTRACKER :
  1. retrait de l'hologramme moyen (division ou soustraction) : I_n = fond ≈ 1 + 2·Re(u) ;
  2. nettoyage de type CLEAN des bactéries détectées, une par une, sur l'hologramme mis à jour :
       - propagation de H = I_n − 1 jusqu'au plan z de la bactérie ;
       - fenêtre circulaire (bord en cosinus) autour de (x, y), zéro ailleurs ;
       - rétropropagation de la fenêtre jusqu'au plan de l'hologramme -> p ;
       - soustraction de 2·Re(p) = p + p* : p retire l'image réelle, p* l'image jumelle ;
     puis, en option, CLEAN des restes : maxima locaux du volume de focus (SUM_OF_VARIANCE, seuil
     local) sur l'hologramme nettoyé, soustraits de la même façon, sur plusieurs itérations ;
  3. bactéries simulées (BPM du simulateur, bâtonnets) aux positions des trajectoires, orientées
     selon la direction de la trajectoire (ajustement linéaire sur ±W images) ;
  4. film final = fond nettoyé × hologramme des bactéries simulées.

Sorties (chacune optionnelle) : moyenne, film de fond (nettoyé), bactéries nettoyées seules,
bactéries simulées seules, bactéries simulées + fond ; en BMP 8 bits (normalisé × niveau de gris
moyen du film, sans réintroduire la moyenne temporelle ; lisible
par HOLOTRACKER) et/ou TIFF float32 normalisé (fond ≈ 1). Toujours : vérité terrain
(ground_truth.csv), journal du nettoyage (cleaning_log.csv), configuration utilisée.

Repères (ceux d'HOLOTRACKER) : x = colonne × dx, y = ligne × dx, z = distance de propagation
positive qui met l'objet au point. Le simulateur place x en ligne de l'image : la conversion est
faite à l'insertion (comme stats/report_common.py).

Utilisation :
  python Hybrid_REAL_SIMU_HOLOGRAMS/real_simulation.py Hybrid_REAL_SIMU_HOLOGRAMS/config_real_simulation.json
  python Hybrid_REAL_SIMU_HOLOGRAMS/GUI_real_simulation.py

Author: Simon BECKER
Date: 2026-10-17

License:
GNU General Public License v3.0
"""

import copy
import datetime
import glob
import json
import math
import os
import sys
import time

import numpy as np
import cupy as cp
import pandas as pd
import tifffile
from PIL import Image

REAL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(REAL_DIR)
sys.path.insert(0, REPO_DIR)

from libs.simu_hologram import Bacterie, GPU_insert_bact_in_mask_volume, phase_shift_through_plane, pad_centered
from libs import focus as focus_lib


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    # Entrées : chemins absolus ou relatifs à la racine du dépôt
    'film_dir': "REAL HOLO FILM/bacterias",
    'image_extension': "bmp",
    'trajectories_csv': "REAL HOLO FILM/bacterias/TRAJECTORIES_20261016_173113.csv",
    'first_hologram_number': 1,      # HOLOGRAM NUMBER de la première image (ordre alphabétique)
    'first_frame': 0,                # première image traitée (indice dans le film, 0 = début)
    'max_frames': 0,                 # nombre d'images traitées (0 = toutes)
    'output_dir': "Hybrid_REAL_SIMU_HOLOGRAMS/output",   # un sous-dossier <film>_<date> y est créé
    'optics': {
        'wavelength_nm': 660.0,
        'index_medium': 1.33,
        'pixel_size_um': 7.0,
        'magnification': 40.0,
    },
    'background': {
        'mean_file': "",             # vide : moyenne arithmétique calculée sur tout le film
        'removal': "division",       # 'division' : I / M ; 'subtraction' : (I − M) / <M> + 1
    },
    'cleaning': {
        'enabled': True,
        'border': 256,               # bordure de zéros (px) : la rétropropagation ne se replie pas
        'window_radius_um': 1.5,     # rayon de la fenêtre au plan z (pleine valeur)
        'window_taper_um': 0.5,      # largeur du bord en cosinus au-delà du rayon
        'passes': 2,                 # passes sur la même liste d'objets (CLEAN : chaque passe retire
                                     #   une part du reste ; 1 : 11 %, 2 : 6 % de signal restant)
        'high_pass': 15,             # filtre de l'extraction : rayons conservés dans le spectre
        'low_pass': 0,               #   (px de Fourier du capteur, 0 = pas de borne)
        'gain': 1.0,                 # fraction de 2·Re(p) soustraite
        'clean_unassigned': True,    # nettoie aussi les détections hors trajectoire…
        'unassigned_min_voxel': 50,  # … d'au moins ce nombre de voxels
        'min_z_um': 2.0,             # objets plus proches de l'hologramme ignorés : l'image jumelle,
                                     #   défocalisée de 2z seulement, tombe dans la fenêtre
        'measure_detections': False, # compte les maxima (paramètres de 'residual') avant / après
        'residual': {                # CLEAN des restes (bactéries non détectées par HOLOTRACKER)
            'enabled': True,
            'iterations': 2,
            'distance_ini_um': 0.0,
            'nb_planes': 200,        # multiple de mesh_z
            'step_um': 0.5,
            'high_pass': 15,
            'low_pass': 250,
            'sum_size': 11,
            'k_local': 15.0,         # K du seuil local (SEGMENTATION_REPORT : 15)
            'mesh_z': 25,
            'mesh_xy': 64,
            'max_objects': 300,      # par itération, par RSB décroissant ; les maxima du premier et
        },                           #   du dernier plan (objet hors plage) sont ignorés
    },
    'simulation': {
        'enabled': True,
        'length_um': 1.8,
        'thickness_um': 0.6,
        'delta_n': 0.046,            # indice objet − indice milieu ; 0,046 : calé sur le film de
                                     #   référence (amplitude réelle / simulée = 1 pour 1,8 × 0,6 µm bout
                                     #   à bout, pas BPM 0,25 µm, 2026-10-18 ; 0,043 avant la correction
                                     #   de la longueur du bâtonnet, qui faisait L + T/2)
        'step_um': 0.5,              # pas BPM
        'upscale_factor': 2,
        'border': 256,
        'chunk_planes': 24,          # tranches du masque traitées ensemble (mémoire GPU)
        'orientation_half_window': 5,    # ajustement de la trajectoire sur ±W images
        'min_points_orientation': 3,     # en dessous : orientation aléatoire
        'simulate_unassigned': False,    # détections hors trajectoire : orientation aléatoire
        'unassigned_min_voxel': 200,
        'measure_contrast': True,    # énergie réelle / simulée dans la fenêtre de nettoyage
        'seed': 0,
    },
    'outputs': {
        'mean': True,
        'background': True,
        'cleaned_bacteria': True,
        'simulated_bacteria': True,
        'simulated_on_background': True,
        'bmp': True,
        'tiff': False,
    },
}

OUTPUT_DIRS = {
    'background': "background",
    'cleaned_bacteria': "cleaned_bacteria",
    'simulated_bacteria': "simulated_bacteria",
    'simulated_on_background': "simulated_on_background",
}

CSV_COLS = {'frame': 'HOLOGRAM NUMBER', 'x': 'X POSITION (m)', 'y': 'Y POSITION (m)', 'z': 'Z POSITION (m)',
            'nvox': 'NUMBER OF VOXEL', 'traj': 'TRAJECTORY NUMBER'}


def merge_config(base, update):
    """Fusion récursive : les clés de update remplacent celles de base."""
    out = copy.deepcopy(base)
    for k, v in (update or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_config(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return merge_config(DEFAULT_CONFIG, json.load(f))


def resolve_path(p):
    if not p:
        return p
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(REPO_DIR, p))


def check_config(cfg):
    """Contrôles des entrées ; lève ValueError avec la liste des erreurs."""
    errors = []
    film = resolve_path(cfg['film_dir'])
    if not os.path.isdir(film):
        errors.append(f"dossier du film introuvable : {film}")
    elif not list_frames(cfg):
        errors.append(f"aucune image .{cfg['image_extension']} dans {film}")
    if not os.path.isfile(resolve_path(cfg['trajectories_csv'])):
        errors.append(f"fichier de trajectoires introuvable : {resolve_path(cfg['trajectories_csv'])}")
    mf = cfg['background']['mean_file']
    if mf and not os.path.isfile(resolve_path(mf)):
        errors.append(f"hologramme moyen introuvable : {resolve_path(mf)}")
    if cfg['background']['removal'] not in ('division', 'subtraction'):
        errors.append("background.removal : 'division' ou 'subtraction'")
    rc = cfg['cleaning']['residual']
    if rc['enabled'] and rc['nb_planes'] % rc['mesh_z']:
        errors.append(f"cleaning.residual.nb_planes ({rc['nb_planes']}) doit être multiple de mesh_z ({rc['mesh_z']})")
    o = cfg['outputs']
    if any(o[k] for k in OUTPUT_DIRS) and not (o['bmp'] or o['tiff']):
        errors.append("outputs : cocher au moins un format (bmp, tiff)")
    if errors:
        raise ValueError("Configuration invalide :\n  - " + "\n  - ".join(errors))


# =============================================================================
# FILM, MOYENNE, NORMALISATION
# =============================================================================

def list_frames(cfg):
    film = resolve_path(cfg['film_dir'])
    return sorted(glob.glob(os.path.join(film, "*." + cfg['image_extension'].lstrip('.'))))


def read_image(path):
    img = np.asarray(Image.open(path))
    if img.ndim == 3:
        img = img[..., :3].mean(axis=2)
    return img.astype(np.float32)


def compute_mean(paths, progress=None):
    """Moyenne arithmétique en float64 (pas d'arrondi 8 bits, contrairement à mean_arith.bmp)."""
    acc = None
    for i, p in enumerate(paths):
        img = read_image(p).astype(np.float64)
        acc = img if acc is None else acc + img
        if progress and (i + 1) % 20 == 0:
            progress(f"  moyenne : {i + 1}/{len(paths)} images")
    return (acc / len(paths)).astype(np.float32)


def normalize(raw, mean, removal):
    """Hologramme normalisé, fond ≈ 1, dans les deux modes."""
    if removal == 'division':
        return raw / np.maximum(mean, 1e-6)
    return (raw - mean) / float(mean.mean()) + 1.0


def save_frame(norm, name, out_dir, gray_level, fmt):
    """
    BMP 8 bits (normalisé × niveau de gris moyen du film) et/ou TIFF float32 normalisé.
    La moyenne temporelle n'est PAS réintroduite : le fond reste plat (≈ gray_level), sans le
    motif fixe du montage. La moyenne elle-même est enregistrée à part (sortie « mean »).
    """
    base = os.path.splitext(name)[0]
    if fmt['bmp']:
        img = np.clip(np.rint(norm * gray_level), 0, 255).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(out_dir, base + ".bmp"))
    if fmt['tiff']:
        tifffile.imwrite(os.path.join(out_dir, base + ".tif"), norm.astype(np.float32))


# =============================================================================
# PROPAGATION (spectre angulaire, grille avec bordure)
# =============================================================================

class Propagator:
    """
    Spectre angulaire sur une grille N + 2·B : noyau exp(2iπ·d·kz), kz = √(1/λ² − fx² − fy²),
    ondes évanescentes coupées. Même noyau que libs/propagation (d_calc_kernel_angular_spectrum_jit)
    et donc même convention de signe que HOLOTRACKER : d > 0 met au point un objet trouvé à z = d.
    La bordure de zéros évite que le champ propagé se replie sur le bord opposé.
    """

    def __init__(self, n, border, dx, lambda_medium):
        self.N, self.B = n, border
        self.NB = n + 2 * border
        f = cp.fft.fftfreq(self.NB, d=dx)
        fx, fy = cp.meshgrid(f, f, indexing='xy')
        arg = 1.0 / lambda_medium ** 2 - fx ** 2 - fy ** 2
        self.kz = cp.sqrt(cp.maximum(arg, 0.0)).astype(cp.float32)
        self.propagating = arg > 0
        # rayon en pixels de Fourier du capteur (grille N), comme les filtres d'HOLOTRACKER
        self.rho = (cp.sqrt(fx ** 2 + fy ** 2) * n * dx).astype(cp.float32)

    def kernel(self, d):
        return cp.where(self.propagating, cp.exp((2j * math.pi * d) * self.kz), 0).astype(cp.complex64)

    def bandpass(self, high_pass, low_pass):
        m = cp.ones((self.NB, self.NB), dtype=cp.float32)
        if high_pass:
            m[self.rho < high_pass] = 0
        if low_pass:
            m[self.rho > low_pass] = 0
        return m

    def pad(self, a):
        out = cp.zeros((self.NB, self.NB), dtype=cp.complex64)
        out[self.B:self.B + self.N, self.B:self.B + self.N] = a
        return out

    def crop(self, a):
        return a[self.B:self.B + self.N, self.B:self.B + self.N]

    def propagate(self, field, d):
        return cp.fft.ifft2(cp.fft.fft2(field) * self.kernel(d))


# =============================================================================
# NETTOYAGE (CLEAN)
# =============================================================================

class Cleaner:
    """Soustraction séquentielle des bactéries d'un hologramme H = I_n − 1 (CuPy, N × N)."""

    def __init__(self, prop, dx_um, cc):
        self.prop, self.dx_um, self.cc = prop, dx_um, cc
        self.bp = prop.bandpass(cc['high_pass'], cc['low_pass'])
        r_out = cc['window_radius_um'] + cc['window_taper_um']
        self.half = int(math.ceil(r_out / dx_um)) + 1

    def window(self, x_um, y_um):
        """Fenêtre (boîte de la grille avec bordure, poids) centrée en (x, y)."""
        B, NB, h = self.prop.B, self.prop.NB, self.half
        c, r = x_um / self.dx_um + B, y_um / self.dx_um + B
        r0, r1 = max(int(round(r)) - h, 0), min(int(round(r)) + h + 1, NB)
        c0, c1 = max(int(round(c)) - h, 0), min(int(round(c)) + h + 1, NB)
        yy, xx = cp.meshgrid(cp.arange(r0, r1, dtype=cp.float32), cp.arange(c0, c1, dtype=cp.float32), indexing='ij')
        dist = cp.sqrt((yy - r) ** 2 + (xx - c) ** 2) * self.dx_um
        R, T = self.cc['window_radius_um'], self.cc['window_taper_um']
        if T > 0:
            w = cp.where(dist <= R, 1.0, cp.where(dist >= R + T, 0.0,
                                                  0.5 * (1 + cp.cos(math.pi * (dist - R) / T))))
        else:
            w = (dist <= R).astype(cp.float32)
        return (r0, r1, c0, c1), w.astype(cp.float32)

    def focused(self, Hf, z_um):
        """Champ filtré au plan z (grille avec bordure)."""
        return cp.fft.ifft2(Hf * self.bp * self.prop.kernel(z_um * 1e-6))

    def window_energy(self, Hf, obj):
        (r0, r1, c0, c1), w = self.window(obj['x_um'], obj['y_um'])
        Ez = self.focused(Hf, obj['z_um'])
        return float(cp.sum(cp.abs(Ez[r0:r1, c0:c1] * w) ** 2))

    def clean(self, H, objects, removed, stop=None):
        """
        Soustrait les objets dans l'ordre donné ; H et removed sont modifiés sur place.
        Chaque objet est extrait de l'hologramme déjà nettoyé des précédents.
        Retourne l'énergie de la fenêtre extraite pour chaque objet (signal + bruit).
        """
        prop, gain = self.prop, self.cc['gain']
        Hf = cp.fft.fft2(prop.pad(H))
        out = []
        for obj in objects:
            if stop is not None and stop.is_set():
                break
            z = obj['z_um'] * 1e-6
            (r0, r1, c0, c1), w = self.window(obj['x_um'], obj['y_um'])
            Ez = self.focused(Hf, obj['z_um'])
            patch = cp.zeros_like(Ez)
            patch[r0:r1, c0:c1] = Ez[r0:r1, c0:c1] * w
            energy = float(cp.sum(cp.abs(patch) ** 2))
            p = cp.fft.ifft2(cp.fft.fft2(patch) * prop.kernel(-z))
            sub = (2.0 * gain) * cp.real(prop.crop(p))
            H -= sub
            removed += sub
            Hf = cp.fft.fft2(prop.pad(H))
            out.append(energy)
        return out

    def noise_energy(self, H):
        """Énergie attendue du bruit seul dans la fenêtre : Σw² × variance de H filtré."""
        _, w = self.window(self.half * self.dx_um, self.half * self.dx_um)
        Hb = cp.fft.ifft2(cp.fft.fft2(self.prop.pad(H)) * self.bp)
        return float(cp.sum(w ** 2)) * float(cp.var(cp.real(self.prop.crop(Hb))))


def clean_passes(cleaner, H, objects, removed, passes, stop=None):
    """Plusieurs passes sur les mêmes objets ; renvoie les énergies extraites à la première passe."""
    energies = cleaner.clean(H, objects, removed, stop)
    for _ in range(int(passes) - 1):
        cleaner.clean(H, objects, removed, stop)
    return energies


def residual_detections(H, prop, rc, dx_um, min_z_um=0.0, max_objects=None):
    """
    Maxima locaux du volume de focus SUM_OF_VARIANCE de l'hologramme (déjà nettoyé), au-dessus du
    seuil local (localisation_pipeline/pipeline_holotracker_locate_local_maxima.py).
    Le volume est calculé plan par plan : seul le volume de focus est stocké.
    Les maxima du premier et du dernier plan sont écartés : l'objet est hors de la plage (ou c'est
    l'hologramme lui-même, à z = 0), et le nettoyer à ce plan effacerait autre chose que lui.
    """
    sys.path.insert(0, os.path.join(REPO_DIR, 'localisation_pipeline'))
    from pipeline_holotracker_locate_local_maxima import detect_local_maxima

    N = prop.N
    bp = prop.bandpass(rc['high_pass'], rc['low_pass'])
    Hf = cp.fft.fft2(prop.pad(H)) * bp
    d_focus = cp.zeros((rc['nb_planes'], N, N), dtype=cp.float32)
    for i in range(rc['nb_planes']):
        d = (rc['distance_ini_um'] + i * rc['step_um']) * 1e-6
        module = cp.abs(prop.crop(cp.fft.ifft2(Hf * prop.kernel(d)))).astype(cp.float32)
        focus_lib.focus_sum_of_variance(module[None], d_focus[i:i + 1], rc['sum_size'])
    df, _, _ = detect_local_maxima(d_focus, dx_um, rc['step_um'], rc['distance_ini_um'],
                                   k_local=rc['k_local'], mesh_z=rc['mesh_z'], mesh_xy=rc['mesh_xy'])
    del d_focus, Hf
    cp.get_default_memory_pool().free_all_blocks()
    df = df[(df.iz > 0) & (df.iz < rc['nb_planes'] - 1) & (df.baryZ >= min_z_um)]
    df = df.head(rc['max_objects'] if max_objects is None else max_objects)
    return [{'x_um': r.baryX, 'y_um': r.baryY, 'z_um': r.baryZ, 'nvox': 0, 'traj': -1, 'kind': 'residual',
             'snr': r.snr} for r in df.itertuples()]


# =============================================================================
# TRAJECTOIRES ET ORIENTATIONS
# =============================================================================

def load_detections(cfg):
    """CSV HOLOTRACKER -> DataFrame en µm (x_um, y_um, z_um, frame, nvox, traj)."""
    d = pd.read_csv(resolve_path(cfg['trajectories_csv']))
    missing = [c for c in CSV_COLS.values() if c not in d.columns and c != CSV_COLS['traj']]
    if missing:
        raise ValueError(f"colonnes absentes du CSV : {missing}")
    out = pd.DataFrame({
        'frame': d[CSV_COLS['frame']].astype(int),
        'x_um': d[CSV_COLS['x']] * 1e6, 'y_um': d[CSV_COLS['y']] * 1e6, 'z_um': d[CSV_COLS['z']] * 1e6,
        'nvox': d[CSV_COLS['nvox']].astype(int),
        'traj': d[CSV_COLS['traj']].astype(int) if CSV_COLS['traj'] in d.columns else -1,
    })
    return out


def random_orientation(rng):
    """Direction isotrope sur la demi-sphère : θ uniforme, cos φ uniforme dans [0, 1]."""
    return 360.0 * rng.random(), math.degrees(math.acos(rng.random()))


def trajectory_orientations(det, half_window, min_points, rng):
    """
    Pour chaque point de trajectoire : vitesse par ajustement linéaire x(t), y(t), z(t) sur les points
    de la même trajectoire à ±half_window images. La bactérie nage selon son axe : l'axe du bâtonnet est
    la direction de la vitesse. θ = azimut dans le plan image depuis l'axe x, φ = angle à l'axe Z,
    replié sur φ ∈ [0°, 90°] (bâtonnet symétrique).
    Ajoute les colonnes theta_deg, phi_deg, speed_um_per_frame, orientation ('trajectory' ou 'random').
    """
    det = det.copy()
    det['theta_deg'], det['phi_deg'] = np.nan, np.nan
    det['speed_um_per_frame'], det['orientation'] = np.nan, ''
    for _, g in det[det.traj >= 0].groupby('traj'):
        t = g.frame.to_numpy(float)
        P = g[['x_um', 'y_um', 'z_um']].to_numpy(float)
        for idx, ti in zip(g.index, t):
            sel = np.abs(t - ti) <= half_window
            v = None
            if sel.sum() >= max(min_points, 2):
                tt = t[sel] - t[sel].mean()
                den = float((tt ** 2).sum())
                if den > 0:
                    v = (tt[:, None] * (P[sel] - P[sel].mean(axis=0))).sum(axis=0) / den
            speed = float(np.linalg.norm(v)) if v is not None else 0.0
            if speed > 0:
                theta = math.degrees(math.atan2(v[1], v[0])) % 360.0
                phi = math.degrees(math.acos(max(-1.0, min(1.0, v[2] / speed))))
                if phi > 90.0:
                    phi, theta = 180.0 - phi, (theta + 180.0) % 360.0
                det.loc[idx, ['theta_deg', 'phi_deg', 'speed_um_per_frame']] = theta, phi, speed
                det.loc[idx, 'orientation'] = 'trajectory'
            else:
                theta, phi = random_orientation(rng)
                det.loc[idx, ['theta_deg', 'phi_deg', 'speed_um_per_frame']] = theta, phi, speed
                det.loc[idx, 'orientation'] = 'random'
    return det


# =============================================================================
# SIMULATION DES BACTÉRIES (BPM, tranches utiles uniquement)
# =============================================================================

def bacteria_instance_masks(bacts, sc, dx, N):
    """
    Masque d'instance de chaque bactérie, tel que simulate_frame l'utilise : même noyau d'insertion,
    même grille (pixel i <-> i·dx en ligne / colonne, tranche k <-> distance k·step_um), même retour à
    la grille finale par moyenne des blocs up × up. Seule différence : chaque bactérie est écrite
    seule, dans sa boîte, au lieu du volume commun (où les recouvrements fusionnent).
    Retourne une liste de (origine (k, ligne, colonne) dans la grille de simulation, tableau float16
    (K, R, C) de la fraction occupée de chaque voxel ∈ [0, 1]).
    """
    up = int(sc['upscale_factor'])
    dz = sc['step_um'] * 1e-6
    out = []
    for b in bacts:
        Lb = b.get('length_um', sc['length_um']) * 1e-6
        Tb = b.get('thickness_um', sc['thickness_um']) * 1e-6
        ext = Lb / 2 + Tb / 2                        # demi-boîte du noyau d'insertion
        y, x, z = b['y_um'] * 1e-6, b['x_um'] * 1e-6, b['z_um'] * 1e-6
        r0, c0 = max(0, int(math.floor((y - ext) / dx)) - 1), max(0, int(math.floor((x - ext) / dx)) - 1)
        r1, c1 = min(N, int(math.ceil((y + ext) / dx)) + 2), min(N, int(math.ceil((x + ext) / dx)) + 2)
        k0, k1 = max(0, int(math.floor((z - ext) / dz)) - 1), int(math.ceil((z + ext) / dz)) + 2
        R, C, K = r1 - r0, c1 - c0, k1 - k0
        if R <= 0 or C <= 0:
            out.append(((k0, r0, c0), np.zeros((0, 0, 0), np.float16)))
            continue
        vol_up = cp.zeros((R * up, C * up, K + 1), dtype=cp.float16)
        bact = Bacterie(pos_x=y - r0 * dx, pos_y=x - c0 * dx, pos_z=z - k0 * dz,
                        thickness=Tb, length=Lb, theta=90.0 - b['theta_deg'], phi=b['phi_deg'])
        GPU_insert_bact_in_mask_volume(vol_up, bact, dx / up, dz, up)
        m = vol_up[:, :, :K].reshape(R, up, C, up, K).mean(axis=(1, 3))
        out.append(((k0, r0, c0), cp.asnumpy(m.transpose(2, 0, 1)).astype(np.float16)))
    return out


def simulate_frame(bacts, prop, sc, dx, lam, plane_masks=None):
    """
    Intensité (N × N, fond 1, sans bruit) d'une onde plane traversant les bactéries.
    bacts : liste de dict x_um, y_um, z_um, theta_deg, phi_deg (repère image), et facultativement
    length_um, thickness_um (taille propre à la bactérie ; sinon celle de sc).
    Seules les tranches du masque qui contiennent un objet sont traversées pas à pas ; le vide est
    franchi en une seule propagation (exact en espace libre, comme stats/report_common.py).
    Masque construit par paquets de chunk_planes tranches (mémoire GPU), du plus loin au plus près.
    plane_masks : dict facultatif, rempli {indice de tranche : masque (N, N) traversé} (contrôle).
    """
    N, up, NB = prop.N, int(sc['upscale_factor']), prop.NB
    dz = sc['step_um'] * 1e-6
    L, T = sc['length_um'] * 1e-6, sc['thickness_um'] * 1e-6
    size = lambda b: (b.get('length_um', sc['length_um']) * 1e-6, b.get('thickness_um', sc['thickness_um']) * 1e-6)
    half = max([sum(size(b)) / 2 for b in bacts] + [L / 2 + T / 2])
    shift_in_obj = 2.0 * math.pi * dz * sc['delta_n'] / lam
    field = cp.ones((NB, NB), dtype=cp.complex64)   # onde plane normale : constante, pas de bord
    if not bacts:
        return cp.ones((N, N), dtype=cp.float32)

    spans = []
    for b in bacts:
        z = b['z_um'] * 1e-6
        spans.append((max(0, int(math.floor((z - half) / dz)) - 2), int(math.ceil((z + half) / dz)) + 2))
    k_top = max(s[1] for s in spans)
    m = int(sc['chunk_planes'])
    current = None   # distance (m) de la dernière tranche appliquée
    for k0 in range((k_top // m) * m, -1, -m):
        sel = [b for b, s in zip(bacts, spans) if s[1] >= k0 and s[0] <= k0 + m - 1]
        if not sel:
            continue
        # m + 1 tranches : le noyau d'insertion exclut la dernière (indice max borné à shape − 1)
        vol_up = cp.zeros((N * up, N * up, m + 1), dtype=cp.float16)
        for b in sel:
            zr = b['z_um'] * 1e-6 - k0 * dz
            if int((zr - half) / dz) > m - 1 or math.ceil((zr + half) / dz) < 1:
                continue
            Lb, Tb = size(b)
            bact = Bacterie(pos_x=b['y_um'] * 1e-6, pos_y=b['x_um'] * 1e-6, pos_z=zr,
                            thickness=Tb, length=Lb, theta=90.0 - b['theta_deg'], phi=b['phi_deg'])
            GPU_insert_bact_in_mask_volume(vol_up, bact, dx / up, dz, up)
        mask = vol_up[:, :, :m].reshape(N, up, N, up, m).mean(axis=(1, 3))
        del vol_up
        occupied = cp.asnumpy(mask.max(axis=(0, 1)) > 0)
        for k in range(m - 1, -1, -1):
            if not occupied[k]:
                continue
            if plane_masks is not None:
                plane_masks[k0 + k] = cp.asnumpy(mask[:, :, k])
            zk = (k0 + k) * dz
            if current is not None and current > zk:
                field = prop.propagate(field, current - zk)
            current = zk
            field = phase_shift_through_plane(mask_plane=pad_centered(mask[:, :, k], [NB, NB]),
                                              plane_to_shift=field, shift_in_env=0.0, shift_in_obj=shift_in_obj)
        del mask
    if current:
        field = prop.propagate(field, current)
    intensity = cp.abs(prop.crop(field)) ** 2
    del field
    return intensity.astype(cp.float32)


# =============================================================================
# CHAÎNE COMPLÈTE
# =============================================================================

def run(cfg, progress=print, stop=None, frame_callback=None, save=True, cache=None):
    """
    Traite le film. progress(str) : journal ; stop : threading.Event ; frame_callback(dict) : images
    de l'image courante (normalized, background, cleaned_bacteria, simulated_bacteria,
    simulated_on_background), pour l'aperçu ; save=False : aucun fichier écrit (aperçu).
    cache : dict conservé par l'appelant entre deux appels (aperçus successifs) ; la moyenne et les
    orientations y sont gardées tant que le film, la moyenne et les trajectoires ne changent pas.
    Retourne le dossier de sortie (None si save=False).
    """
    cache = {} if cache is None else cache
    check_config(cfg)
    if not save:
        cfg = merge_config(cfg, {'outputs': {k: False for k in ('mean', *OUTPUT_DIRS)}})
    t0 = time.time()
    opt, bg, cc, sc, outs = cfg['optics'], cfg['background'], cfg['cleaning'], cfg['simulation'], cfg['outputs']
    dx = opt['pixel_size_um'] * 1e-6 / opt['magnification']
    dx_um = dx * 1e6
    lam = opt['wavelength_nm'] * 1e-9
    lam_m = lam / opt['index_medium']
    rng = np.random.default_rng(sc['seed'])

    paths = list_frames(cfg)
    first = int(cfg['first_frame'])
    sel_paths = paths[first:]
    if cfg['max_frames']:
        sel_paths = sel_paths[:int(cfg['max_frames'])]
    film_name = os.path.basename(os.path.normpath(resolve_path(cfg['film_dir'])))
    out_dir = None
    if save:
        out_dir = os.path.join(resolve_path(cfg['output_dir']),
                               f"{film_name}_{datetime.datetime.now():%Y_%m_%d_%H_%M_%S}")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "config_real_simulation.json"), 'w', encoding='utf-8') as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        for key, sub in OUTPUT_DIRS.items():
            if outs[key]:
                os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    progress(f"Film : {len(paths)} images, {len(sel_paths)} traitées -> {out_dir or 'aperçu (rien n est écrit)'}")

    # 1. Moyenne
    mean_key = (tuple(paths), resolve_path(bg['mean_file']) or None,
                os.path.getmtime(resolve_path(bg['mean_file'])) if bg['mean_file'] else None)
    if cache.get('mean_key') == mean_key:
        mean = cache['mean']
        progress("Moyenne : reprise du calcul précédent")
    elif bg['mean_file']:
        mean = read_image(resolve_path(bg['mean_file']))
        progress(f"Moyenne lue : {resolve_path(bg['mean_file'])}")
    else:
        progress("Calcul de la moyenne du film…")
        mean = compute_mean(paths, progress)
    cache['mean_key'], cache['mean'] = mean_key, mean
    N = mean.shape[0]
    gray_level = float(mean.mean())     # niveau du fond des BMP enregistrés
    if mean.shape[0] != mean.shape[1]:
        raise ValueError(f"image non carrée {mean.shape} : non prise en charge")
    if outs['mean']:
        os.makedirs(os.path.join(out_dir, "mean"), exist_ok=True)
        tifffile.imwrite(os.path.join(out_dir, "mean", "mean.tif"), mean)
        Image.fromarray(np.clip(np.rint(mean), 0, 255).astype(np.uint8)).save(os.path.join(out_dir, "mean", "mean.bmp"))

    # 2. Détections et orientations
    csv_path = resolve_path(cfg['trajectories_csv'])
    det_key = (csv_path, os.path.getmtime(csv_path), sc['orientation_half_window'], sc['min_points_orientation'],
               sc['seed'])
    if cache.get('det_key') == det_key:
        det = cache['det']
    else:
        det = load_detections(cfg)
        det = trajectory_orientations(det, sc['orientation_half_window'], sc['min_points_orientation'], rng)
        cache['det_key'], cache['det'] = det_key, det
    n_traj = (det.traj >= 0).sum()
    n_rand = (det.orientation == 'random').sum()
    progress(f"Détections : {len(det)} dont {n_traj} en trajectoire ({det[det.traj >= 0].traj.nunique()} "
             f"trajectoires, {n_rand} orientations aléatoires faute de points)")

    prop_clean = Propagator(N, cc['border'], dx, lam_m)
    cleaner = Cleaner(prop_clean, dx_um, cc)
    prop_sim = prop_clean if sc['border'] == cc['border'] else Propagator(N, sc['border'], dx, lam_m)

    truth_rows, log_rows, frame_rows = [], [], []
    fmt = {'bmp': outs['bmp'], 'tiff': outs['tiff']}
    for fi, path in enumerate(sel_paths):
        if stop is not None and stop.is_set():
            progress("Arrêt demandé.")
            break
        tf = time.time()
        name = os.path.basename(path)
        holo_number = first + fi + int(cfg['first_hologram_number'])
        raw = read_image(path)
        norm = normalize(raw, mean, bg['removal'])
        H = cp.asarray(norm - 1.0, dtype=cp.float32)
        removed = cp.zeros_like(H)
        frame_det = det[det.frame == holo_number]
        if cc['measure_detections']:
            H_initial = H.copy()

        # 3. Nettoyage
        if cc['enabled']:
            near = frame_det[frame_det.z_um >= cc['min_z_um']]
            objs = near[near.traj >= 0].sort_values('nvox', ascending=False)
            objs = [dict(r._asdict(), kind='trajectory') for r in objs.itertuples()]
            if cc['clean_unassigned']:
                un = near[(near.traj < 0) & (near.nvox >= cc['unassigned_min_voxel'])]
                objs += [dict(r._asdict(), kind='unassigned') for r in un.sort_values('nvox', ascending=False).itertuples()]
            energies = clean_passes(cleaner, H, objs, removed, cc['passes'], stop)
            rc = cc['residual']
            if rc['enabled']:
                for it in range(rc['iterations']):
                    res = residual_detections(H, prop_clean, rc, dx_um, cc['min_z_um'])
                    if not res:
                        break
                    progress(f"    CLEAN des restes, itération {it + 1} : {len(res)} maxima")
                    objs += res
                    energies += clean_passes(cleaner, H, res, removed, cc['passes'], stop)
            e_noise = cleaner.noise_energy(H)
            for o, e in zip(objs, energies):
                log_rows.append({'HOLOGRAM NUMBER': holo_number, 'kind': o['kind'], 'trajectory': o['traj'],
                                 'x_um': o['x_um'], 'y_um': o['y_um'], 'z_um': o['z_um'], 'nb_voxel': o['nvox'],
                                 'snr_residual': o.get('snr', float('nan')),
                                 'energy': e, 'energy_noise': e_noise})
            real_energy = {o['Index']: e for o, e in zip(objs, energies) if 'Index' in o}
        else:
            real_energy, e_noise = {}, float('nan')

        background = 1.0 + H

        # 4. Bactéries simulées
        if sc['enabled']:
            sims = frame_det[frame_det.traj >= 0]
            if sc['simulate_unassigned']:
                un = frame_det[(frame_det.traj < 0) & (frame_det.nvox >= sc['unassigned_min_voxel'])].copy()
                for idx in un.index:
                    th, ph = random_orientation(rng)
                    un.loc[idx, ['theta_deg', 'phi_deg', 'speed_um_per_frame']] = th, ph, np.nan
                    un.loc[idx, 'orientation'] = 'random'
                sims = pd.concat([sims, un])
            bacts = [dict(r._asdict()) for r in sims.itertuples()]
            simulated = simulate_frame(bacts, prop_sim, sc, dx, lam)
            sim_energy = {}
            if sc['measure_contrast'] and bacts:
                Hf_sim = cp.fft.fft2(prop_clean.pad(simulated - 1.0))
                sim_energy = {b['Index']: cleaner.window_energy(Hf_sim, b) for b in bacts}
                del Hf_sim
            for b in bacts:
                eb, es = real_energy.get(b['Index'], float('nan')), sim_energy.get(b['Index'], float('nan'))
                ratio = math.sqrt(max(eb - e_noise, 0.0) / es) if es and es > 0 and not math.isnan(eb) else float('nan')
                truth_rows.append({
                    'HOLOGRAM NUMBER': holo_number, 'image': name, 'TRAJECTORY NUMBER': b['traj'],
                    'X POSITION (m)': b['x_um'] * 1e-6, 'Y POSITION (m)': b['y_um'] * 1e-6,
                    'Z POSITION (m)': b['z_um'] * 1e-6,
                    'theta_deg': b['theta_deg'], 'phi_deg': b['phi_deg'], 'orientation': b['orientation'],
                    'speed_um_per_frame': b['speed_um_per_frame'],
                    'length_um': sc['length_um'], 'thickness_um': sc['thickness_um'], 'delta_n': sc['delta_n'],
                    'energy_real': eb, 'energy_sim': es, 'energy_noise': e_noise, 'amplitude_ratio_real_sim': ratio})
        else:
            simulated = cp.ones_like(H)

        final = background * simulated

        # Indicateur : maxima détectés avant nettoyage, dans le fond, dans le film final
        if cc['measure_detections']:
            rc = cc['residual']
            row = {'HOLOGRAM NUMBER': holo_number}
            traj_pts = frame_det[frame_det.traj >= 0]
            for key, img in (('normalized', H_initial), ('background', H), ('final', final - 1.0)):
                found = residual_detections(img, prop_clean, rc, dx_um, 0.0, max_objects=10 ** 6)
                row[f'maxima_{key}'] = len(found)
                row[f'maxima_{key}_at_trajectories'] = count_matches(found, traj_pts)
            row['trajectory_points'] = len(traj_pts)
            frame_rows.append(row)
            del H_initial

        # 5. Sauvegardes
        to_np = lambda a: cp.asnumpy(a).astype(np.float32)
        images = {'background': background, 'cleaned_bacteria': 1.0 + removed,
                  'simulated_bacteria': simulated, 'simulated_on_background': final}
        for key, img in images.items():
            if outs[key]:
                save_frame(to_np(img), name, os.path.join(out_dir, OUTPUT_DIRS[key]), gray_level, fmt)
        if frame_callback is not None:
            frame_callback({'index': fi, 'name': name, 'normalized': norm,
                            **{k: to_np(v) for k, v in images.items()}})
        n_clean = sum(1 for r in log_rows if r['HOLOGRAM NUMBER'] == holo_number)
        progress(f"[{fi + 1}/{len(sel_paths)}] {name} : {n_clean} objets nettoyés, "
                 f"{sum(1 for r in truth_rows if r['HOLOGRAM NUMBER'] == holo_number)} bactéries simulées "
                 f"({time.time() - tf:.1f} s)")
        del H, removed, background, simulated, final, images
        cp.get_default_memory_pool().free_all_blocks()

    summary = summarize(truth_rows, log_rows, frame_rows)
    if save:
        pd.DataFrame(truth_rows).to_csv(os.path.join(out_dir, "ground_truth.csv"), index=False)
        pd.DataFrame(log_rows).to_csv(os.path.join(out_dir, "cleaning_log.csv"), index=False)
        if frame_rows:
            pd.DataFrame(frame_rows).to_csv(os.path.join(out_dir, "detections_per_frame.csv"), index=False)
        with open(os.path.join(out_dir, "summary.json"), 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    for line in format_summary(summary):
        progress(line)
    progress(f"Terminé en {time.time() - t0:.0f} s" + (f" -> {out_dir}" if out_dir else ""))
    return out_dir


def count_matches(found, points, tol_xy_um=2.0, tol_z_um=5.0):
    """Nombre de points (DataFrame x_um, y_um, z_um) qui ont un maximum détecté à proximité."""
    if not found or points.empty:
        return 0
    f = np.array([[o['x_um'], o['y_um'], o['z_um']] for o in found])
    n = 0
    for p in points[['x_um', 'y_um', 'z_um']].to_numpy():
        n += bool(((np.hypot(f[:, 0] - p[0], f[:, 1] - p[1]) < tol_xy_um) & (np.abs(f[:, 2] - p[2]) < tol_z_um)).any())
    return n


def summarize(truth_rows, log_rows, frame_rows=()):
    s = {}
    if log_rows:
        lg = pd.DataFrame(log_rows)
        for kind, g in lg.groupby('kind'):
            s[f'cleaning_{kind}'] = {'count': int(len(g)),
                                     'energy_over_noise_median': float((g.energy / g.energy_noise).median())}
    if truth_rows:
        tr = pd.DataFrame(truth_rows)
        ratio = float(tr.amplitude_ratio_real_sim.median())
        s['simulated'] = {'count': int(len(tr)), 'orientation_counts': tr.orientation.value_counts().to_dict(),
                          'phi_deg_median': float(tr.phi_deg.median()),
                          'amplitude_ratio_real_sim_median': ratio,
                          # objet de phase faible : amplitude ∝ Δn (à forme égale)
                          'delta_n_equivalent': float(tr.delta_n.iloc[0] * ratio)}
    if frame_rows:
        fr = pd.DataFrame(frame_rows)
        s['detections_per_frame_mean'] = {c: float(fr[c].mean()) for c in fr.columns if c != 'HOLOGRAM NUMBER'}
    return s


def format_summary(s):
    lines = ["Bilan :"]
    for k, v in s.items():
        if k.startswith('cleaning_'):
            lines.append(f"  nettoyage {k[9:]} : {v['count']} objets, énergie extraite / bruit (médiane) "
                         f"{v['energy_over_noise_median']:.1f}")
    if 'simulated' in s:
        v = s['simulated']
        lines.append(f"  simulées : {v['count']} ({v['orientation_counts']}), φ médian {v['phi_deg_median']:.0f}°, "
                     f"amplitude réelle / simulée médiane {v['amplitude_ratio_real_sim_median']:.2f} "
                     f"-> Δn équivalent des bactéries réelles ≈ {v['delta_n_equivalent']:.4f}")
    if 'detections_per_frame_mean' in s:
        v = s['detections_per_frame_mean']
        lines.append(f"  maxima par image (moyenne) : avant nettoyage {v['maxima_normalized']:.1f} "
                     f"({v['maxima_normalized_at_trajectories']:.1f} aux trajectoires sur {v['trajectory_points']:.1f}), "
                     f"fond {v['maxima_background']:.1f} ({v['maxima_background_at_trajectories']:.1f}), "
                     f"final {v['maxima_final']:.1f} ({v['maxima_final_at_trajectories']:.1f})")
    return lines


def main():
    if len(sys.argv) < 2:
        print("Usage : python Hybrid_REAL_SIMU_HOLOGRAMS/real_simulation.py <config.json>")
        sys.exit(1)
    run(load_config(sys.argv[1]))


if __name__ == '__main__':
    main()

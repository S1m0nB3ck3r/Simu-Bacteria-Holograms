# -*- coding: utf-8 -*-

"""
Filename: inference_pipeline.py

Description:
Inférence de la chaîne U-Net 3D sur un film ou un hologramme, sans interface (utilisée par
GUI_inference.py) :
  - lecture des images (BMP, TIFF, PNG), suppression de l'hologramme moyen du film : division I / M (comme
    HOLOTRACKER et Hybrid_REAL_SIMU_HOLOGRAMS) ou soustraction (I − M) / <M> + 1 ; sans hologramme moyen :
    division par la moyenne de l'image, ou rien (image déjà ramenée à un fond de 1) ;
  - étage 1 (positions, U-Net 3D sur le volume grossier) puis, si un modèle est donné, étage 2
    (instances fines, seg_stage2.segment_instances, deux tours) ; a priori temporel facultatif : les
    centres des instances de l'image précédente sont ajoutés aux candidats de l'étage 2 (une bactérie
    bouge peu d'une image à l'autre ; l'étage 2 rejette un candidat sans bactérie) ;
  - mesures de chaque instance (barycentre, axe, longueur, épaisseur, volume) ;
  - export : CSV (colonnes d'HOLOTRACKER pour la position, en m, + mesures) et, par image, les instances
    (origine + masque, grille fine) en .npz.
run_batch(reporter, ...) est la fonction de travail du TaskRunner (gui_messages.py).

Utilisation en ligne de commande :
  python deep_learning_segmentation/inference_pipeline.py <dossier du film> <dossier de sortie> [--z_max 100]
         [--stage1 models/stage1] [--stage2 models/stage2]

Author: Simon BECKER
Date: 2026-10-18

License:
GNU General Public License v3.0
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

DL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DL_DIR)
import detection as det
import holo_physics as hp
import seg_stage2 as s2
import train as tr
from unet3d import UNet3D

IMAGE_EXT = ('bmp', 'tif', 'tiff', 'png')
# suppression de l'hologramme moyen du film (M) ; le résultat a toujours un fond ≈ 1
MEAN_REMOVAL = {'division': "division par l'hologramme moyen du film : I / M",
                'subtraction': "soustraction de l'hologramme moyen du film : (I − M) / <M> + 1",
                'image_mean': "aucune ; division par la moyenne de l'image : I / <I>",
                'none': "aucune (image déjà ramenée à un fond de 1)"}
OLD_NAMES = {'film_mean': 'division'}          # anciens réglages (« normalisation »)
# =============================================================================
# RÉGLAGES PAR DÉFAUT DE L'INFÉRENCE — options de la ligne de commande et de GUI_inference.py ; None = valeur
# enregistrée avec le modèle (son entraînement)
# =============================================================================
DEFAULT_OPTICS = dict(dx_um=0.175, wavelength_um=0.66, index_medium=1.33)
# reconstruction : plage z de recherche, pas Z, décimation, filtre (px de Fourier du capteur)
DEFAULT_RECON = dict(z_min_um=0.0, z_max_um=None, step_z_um=None, decim=None, high_pass=None, low_pass=None,
                     z_margin_um=None)
Z_MARGIN_UM = 8.0       # plans reconstruits de part et d'autre de la plage de recherche (bords du réseau)
PEAK_BOX = None         # étage 1 : fenêtre des maxima locaux (Z, Y, X), voxels grossiers ; None : celle du modèle
FILM_STATS_HOLOS = None  # images utilisées pour les statistiques d'une normalisation « film » ; None : celles du modèle
# étage 2 (seg_stage2.INFER) : seuil et fenêtre des maxima de la carte fine des centres, bord ignoré, distance de
# deux centres identiques, voxels minimum (None : celui du modèle), fusion des doublons (IoU, voisinage)
STAGE2 = dict(heat_thr=0.3, nms=(5, 5, 5), border=2, same_xy_um=0.5, same_z_um=1.0, vmin=None,
              dup_iou=0.5, dup_xy_um=1.5, dup_z_um=3.0)


# =============================================================================
# FILM
# =============================================================================

def list_frames(folder, ext='bmp'):
    exts = IMAGE_EXT if ext in ('', '*', None) else (ext.lower().lstrip('.'),)
    paths = []
    for e in exts:
        paths += glob.glob(os.path.join(folder, f'*.{e}'))
    return sorted(set(paths))


def read_image(path):
    if path.lower().endswith(('.tif', '.tiff')):
        import tifffile
        a = tifffile.imread(path)
    else:
        from PIL import Image
        a = np.asarray(Image.open(path))
    a = a.astype(np.float32)
    return a.mean(axis=2) if a.ndim == 3 else a


def film_mean(paths, reporter=None):
    acc = None
    for i, p in enumerate(paths):
        a = read_image(p).astype(np.float64)
        acc = a if acc is None else acc + a
        if reporter is not None:
            reporter.check_stop()
            if (i + 1) % 20 == 0 or i + 1 == len(paths):
                reporter.progress(i + 1, len(paths), "moyenne du film")
    return (acc / len(paths)).astype(np.float32)


def remove_mean(raw, mode, mean=None):
    """
    Suppression de l'hologramme moyen M du film (mode 'division' ou 'subtraction', même formule que
    Hybrid_REAL_SIMU_HOLOGRAMS/real_simulation.normalize), ou mise à l'échelle seule ('image_mean', 'none').
    La division retire aussi un éclairage non uniforme (facteur multiplicatif) ; la soustraction le laisse
    moduler le signal. Fond ≈ 1 dans tous les cas.
    """
    mode = OLD_NAMES.get(mode, mode)
    if mode == 'division':
        return raw / np.maximum(mean, 1e-6)
    if mode == 'subtraction':
        return (raw - mean) / max(float(mean.mean()), 1e-6) + 1.0
    if mode == 'image_mean':
        return raw / max(float(raw.mean()), 1e-6)
    return raw


def uses_film_mean(mode):
    return OLD_NAMES.get(mode, mode) in ('division', 'subtraction')


# =============================================================================
# CHAÎNE
# =============================================================================

def load_stage1(run_dir):
    ck = torch.load(os.path.join(run_dir, 'model_best.pt'), map_location='cuda', weights_only=False)
    cfg = ck['cfg']
    m = UNet3D(in_ch=2 if cfg['input'] == 'reim' else 1, out_ch=7 if cfg.get('orient_weight', 0) > 0 else 1,
               base=cfg['base'], depth=cfg['depth'], norm=cfg.get('norm_layer', 'group')).cuda().eval()
    m.load_state_dict(ck['state'])
    return tr.inference_model(m), cfg, ck['threshold']


def load_stage2(run_dir):
    ck = torch.load(os.path.join(run_dir, 'model_best.pt'), map_location='cuda', weights_only=False)
    cfg = ck['cfg']
    m = UNet3D(in_ch=2, out_ch=3, base=cfg['base'], depth=cfg['depth'], norm='batch').cuda().eval()
    m.load_state_dict(ck['state'])
    return tr.inference_model(m), cfg


class Pipeline:
    def __init__(self, stage1_dir, stage2_dir=None, optics=None, z_max_um=None, recon=None):
        self.m1, self.cfg1, self.thr1 = load_stage1(stage1_dir)
        self.model_grid = dict(self.cfg1['grid'])
        r = self.cfg1.get('recon', {})
        self.model_recon = dict(step_z_um=self.model_grid['step_z_um'], decim=self.model_grid['decim'],
                                high_pass=r.get('high_pass', 15), low_pass=r.get('low_pass', 250))
        self.m2, self.cfg2 = load_stage2(stage2_dir) if stage2_dir else (None, None)
        self.fine = dict(s2.FINE, **(self.cfg2.get('fine', {}) if self.cfg2 else {}))
        self.refocus_crop = int(self.cfg2.get('refocus_crop', s2.REFOCUS_CROP)) if self.cfg2 else s2.REFOCUS_CROP
        self.stage2_decim = int(self.cfg2.get('recon', {}).get('decim', 2)) if self.cfg2 else None
        self.model_peak_box = tuple(self.cfg1.get('peak_box', (3, 5, 5)))
        self.model_film_holos = int(self.cfg1.get('film_holos', tr.FILM_STATS_HOLOS))
        self.infer = None
        self.set_post()
        recon = dict(recon or {})
        if z_max_um is not None:
            recon['z_max_um'] = z_max_um
        self.configure(optics, recon)

    def set_post(self, peak_box=PEAK_BOX, film_holos=FILM_STATS_HOLOS, **stage2):
        """
        Post-traitement : fenêtre des maxima de l'étage 1 (None : celle du modèle), images pour les statistiques
        d'une normalisation « film » (None : celles du modèle), réglages de l'étage 2 (clés de STAGE2 ; None :
        défaut de STAGE2, vmin : celui du modèle).
        """
        self.peak_box = tuple(int(v) for v in peak_box) if peak_box else self.model_peak_box
        if any(b % 2 == 0 or b < 1 for b in self.peak_box):
            raise ValueError(f"fenêtre des maxima {self.peak_box} : entiers impairs")
        self.film_holos = int(film_holos) if film_holos else self.model_film_holos
        if self.cfg2 is not None:
            inf = dict(STAGE2, vmin=self.cfg2.get('vmin', s2.INFER['vmin']))
            inf.update({k: v for k, v in stage2.items() if v not in (None, '')})
            inf['nms'] = tuple(int(v) for v in inf['nms'])
            if any(b % 2 == 0 or b < 1 for b in inf['nms']):
                raise ValueError(f"fenêtre des maxima de l'étage 2 {inf['nms']} : entiers impairs")
            self.infer = inf

    def configure(self, optics=None, recon=None):
        """
        Optique (dx_um, wavelength_um, index_medium) et reconstruction : z_min_um, z_max_um (plage de recherche ;
        z_max None : jusqu'au bout de la plage du modèle), step_z_um, decim, high_pass, low_pass (px de Fourier
        du capteur) ; None = valeur d'entraînement du modèle. Seuls les plans utiles sont reconstruits (marge de
        8 µm de chaque côté, nombre de plans multiple de 2^niveaux), puis les détections et les instances hors de
        [z_min, z_max] sont retirées. L'étage 2 exige la décimation de son entraînement.
        """
        self.optics = dict(DEFAULT_OPTICS, **{k: v for k, v in (optics or {}).items() if v is not None})
        r = dict(DEFAULT_RECON, **(recon or {}))
        m, g0 = self.model_recon, self.model_grid
        step = float(r['step_z_um'] or m['step_z_um'])
        decim = int(r['decim'] or m['decim'])
        if self.m2 is not None and decim != self.stage2_decim:
            raise ValueError(f"décimation {decim} : l'étage 2 a été entraîné avec une décimation de {self.stage2_decim}")
        hp_ = m['high_pass'] if r['high_pass'] in (None, '') else float(r['high_pass'])
        lp_ = m['low_pass'] if r['low_pass'] in (None, '') else float(r['low_pass'])
        z_min = max(float(r['z_min_um'] or 0.0), 0.0)
        z_max = float(r['z_max_um']) if r['z_max_um'] not in (None, '', 0) else None
        if z_max is not None and z_max <= z_min:
            raise ValueError(f"profondeur : z max ({z_max}) ≤ z min ({z_min})")
        mult = 2 ** int(self.cfg1['depth'])
        margin = Z_MARGIN_UM if r['z_margin_um'] in (None, '') else float(r['z_margin_um'])
        z0 = max(0.0, float(np.floor((z_min - margin) / step) * step))
        z_end = z_max + margin if z_max is not None else g0['z0_um'] + (g0['nb_planes'] - 1) * g0['step_z_um']
        nb = int(np.ceil(((z_end - z0) / step + 1) / mult) * mult)
        self.grid = dict(g0, z0_um=z0, step_z_um=step, nb_planes=nb, decim=decim)
        self.recon = dict(z_min_um=z_min, z_max_um=z_max, step_z_um=step, decim=decim, high_pass=hp_, low_pass=lp_,
                          z_margin_um=margin)
        self.z_min_um, self.z_max_um = z_min, z_max
        self._rec = {}
        self.z_range_um = (z0, z0 + (nb - 1) * step)

    def set_z_max(self, z_max_um):
        self.configure(self.optics, dict(self.recon, z_max_um=z_max_um))

    def warnings(self):
        """Écarts entre la reconstruction choisie et celle de l'entraînement des modèles."""
        w, m = [], self.model_recon
        for k, lab in (('step_z_um', 'pas Z (µm)'), ('decim', 'décimation'), ('high_pass', 'passe-haut (px)'),
                       ('low_pass', 'passe-bas (px)')):
            if abs(float(self.recon[k]) - float(m[k])) > 1e-9:
                w.append(f"{lab} {self.recon[k]:g} au lieu de {m[k]:g} à l'entraînement")
        if self.peak_box != self.model_peak_box:
            w.append(f"fenêtre des maxima {self.peak_box} au lieu de {self.model_peak_box} à l'entraînement")
        mo = self.cfg1.get('optics')
        if mo:
            for k, lab in (('dx_um', 'pixel objet (µm)'), ('wavelength_um', "longueur d'onde (µm)"), ('index_medium', 'indice')):
                if abs(float(self.optics[k]) - float(mo[k])) > 1e-6:
                    w.append(f"{lab} {self.optics[k]:g} au lieu de {mo[k]:g} à l'entraînement")
        return w

    def describe(self):
        g, r = self.grid, self.recon
        zr = (f"recherche {self.z_min_um:.0f}–{self.z_max_um:.0f} µm" if self.z_max_um is not None
              else f"recherche à partir de {self.z_min_um:.0f} µm")
        s = (f"étage 1 : {self.cfg1.get('name', '?')} ({self.cfg1['loss']}, normalisation {self.cfg1.get('norm', 'fixed')}, "
             f"seuil {self.thr1}) ; reconstruction "
             f"z {self.z_range_um[0]:.0f}–{self.z_range_um[1]:.0f} µm ({zr}), pas {g['step_z_um']:g} µm, décimation "
             f"{g['decim']}, bande {r['high_pass']:g}–{r['low_pass']:g} px, pixel {self.optics['dx_um']:g} µm")
        if self.m2 is not None:
            s += f" ; étage 2 : {self.cfg2.get('name', '?')} (grille {self.optics['dx_um']:g} × {self.fine['dz_um']:g} µm)"
        w = self.warnings()
        if w:
            s += " ; ATTENTION : " + " ; ".join(w)
        return s

    def reconstructor(self, n):
        if n not in self._rec:
            z = self.grid['z0_um'] + np.arange(self.grid['nb_planes']) * self.grid['step_z_um']
            rec = hp.Reconstructor(n, self.optics['dx_um'], self.optics['wavelength_um'], self.optics['index_medium'], z,
                                   decim=self.grid['decim'], high_pass=self.recon['high_pass'],
                                   low_pass=self.recon['low_pass'])
            ref = None
            if self.m2 is not None:
                ref = s2.LocalRefocus(rec, nz=self.fine['nz'], nxy=self.fine['nxy'], dz_um=self.fine['dz_um'],
                                      crop=self.refocus_crop)
            self._rec[n] = (rec, ref)
        return self._rec[n]

    @torch.no_grad()
    def film_stats(self, paths, removal, mean, reporter=None, n=None):
        """Statistiques du film (modèle 'filmmad' ou 'filmstd') : n images régulièrement espacées, même reconstruction."""
        n = int(n or self.film_holos)
        idx = np.unique(np.linspace(0, len(paths) - 1, min(n, len(paths))).round().astype(int))
        holos = [remove_mean(read_image(paths[i]), removal, mean) for i in idx]
        tr.GRID.update(self.grid)
        rec, _ = self.reconstructor(holos[0].shape[0])
        if reporter is not None:
            reporter.log(f"statistiques du film ({self.cfg1['norm']}, {len(holos)} images)…")
        return tr.film_stats_of(holos, rec, self.cfg1, n=len(holos))

    def process(self, holo_norm, thr1=None, rounds=2, prior_centers=None, keep_windows=False, film_stats=None):
        """
        holo_norm : (n, n), fond ≈ 1. Retourne positions de l'étage 1, instances, mesures, temps (s) et, si
        keep_windows, l'amplitude du champ refocalisé autour de chaque instance (coupes affichées par le GUI).
        """
        if holo_norm.shape[0] != holo_norm.shape[1] or holo_norm.shape[0] % 16:
            raise ValueError(f"hologramme {holo_norm.shape} : carré, côté multiple de 16 attendu")
        tr.GRID.update(self.grid)
        tr.PEAK['box'] = self.peak_box
        rec, refocus = self.reconstructor(holo_norm.shape[0])
        t = {}
        torch.cuda.synchronize(); t0 = time.perf_counter()
        V = rec(holo_norm)
        torch.cuda.synchronize(); t1 = time.perf_counter()
        logits = det.infer_volume(self.m1, tr.to_input(V, self.cfg1, film_stats))
        torch.cuda.synchronize(); t2 = time.perf_counter()
        pos1, sc1 = tr.detect(logits, self.cfg1['target'], self.thr1 if thr1 is None else thr1, rec)
        del logits
        zhi = self.z_max_um if self.z_max_um is not None else np.inf
        if len(pos1):
            keep = (pos1[:, 2] >= self.z_min_um) & (pos1[:, 2] <= zhi)
            pos1, sc1 = pos1[keep], np.asarray(sc1)[keep]
        torch.cuda.synchronize(); t3 = time.perf_counter()
        t.update(reconstruction=t1 - t0, network1=t2 - t1, peaks=t3 - t2)
        insts, meas, windows = [], [], None
        if self.m2 is not None:
            cand = pos1
            if prior_centers is not None and len(prior_centers):
                cand = np.concatenate([pos1, np.asarray(prior_centers)[:, :3]])
            insts = s2.segment_instances(self.m2, V, cand, refocus, cfg=self.infer, rounds=rounds)
            insts = [d for d in insts if self.z_min_um <= d['center'][2] <= zhi]
            torch.cuda.synchronize(); t4 = time.perf_counter()
            for d in insts:
                m = s2.mask_measures(d['mask'], d['origin'], refocus.dx, refocus.dz)
                meas.append(m)
            t['stage2'] = t4 - t3
            if keep_windows and insts:
                X, worg = refocus(V, np.array([d['center'] for d in insts]))
                windows = (torch.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2).half().cpu().numpy(), worg)
                del X
        del V
        t['total'] = sum(t.values())
        return dict(pos1=pos1, score1=sc1, instances=insts, measures=meas, timings=t, windows=windows)

    def label_projections(self, res, n):
        """Projections des instances (étiquettes 1..m) : XY (n × n), XZ et YZ (plans fins × n)."""
        nzf = int(round((self.z_range_um[1] + 8) / self.fine['dz_um'])) + 1
        xy = np.zeros((n, n), np.int32)
        xz = np.zeros((nzf, n), np.int32)
        yz = np.zeros((nzf, n), np.int32)
        for lab, d in enumerate(res['instances'], start=1):
            m, (k0, r0, c0) = d['mask'], d['origin']
            k, r, c = np.nonzero(m)
            k, r, c = k + k0, r + r0, c + c0
            ok = (k >= 0) & (k < nzf) & (r >= 0) & (r < n) & (c >= 0) & (c < n)
            k, r, c = k[ok], r[ok], c[ok]
            xy[r, c] = lab
            xz[k, c] = lab
            yz[k, r] = lab
        return xy, xz, yz


def result_rows(frame_index, name, res, hologram_number=None):
    """
    Lignes CSV d'une image. Les 6 premières colonnes sont celles du RESULT_*.csv d'HOLOTRACKER (positions en m,
    NUMBER OF VOXEL = voxels du masque fin), suivies des mesures des instances.
    """
    rows = []
    hn = frame_index + 1 if hologram_number is None else hologram_number
    if res['instances']:
        for i, (d, m) in enumerate(zip(res['instances'], res['measures'])):
            c = m['center'] if m else d['center']
            rows.append({'HOLOGRAM NUMBER': hn, 'OBJECT NUMBER': i + 1,
                         'X POSITION (m)': c[0] * 1e-6, 'Y POSITION (m)': c[1] * 1e-6, 'Z POSITION (m)': c[2] * 1e-6,
                         'NUMBER OF VOXEL': d['n'], 'image': name,
                         'axis_x': m['axis'][0] if m else np.nan, 'axis_y': m['axis'][1] if m else np.nan,
                         'axis_z': m['axis'][2] if m else np.nan, 'length_um': m['length'] if m else np.nan,
                         'thickness_um': m['thickness'] if m else np.nan, 'volume_um3': m['volume'] if m else np.nan,
                         'score': d['score'], 'round': d['round']})
    else:
        for i, (p, s) in enumerate(zip(res['pos1'], res['score1'])):
            rows.append({'HOLOGRAM NUMBER': hn, 'OBJECT NUMBER': i + 1, 'X POSITION (m)': p[0] * 1e-6,
                         'Y POSITION (m)': p[1] * 1e-6, 'Z POSITION (m)': p[2] * 1e-6, 'NUMBER OF VOXEL': 0,
                         'image': name, 'score': float(s)})
    return rows


def save_overlay_bmp(path, h, res, dx_um, z_max, contrast=0.1):
    """
    BMP 8 bits RGB : hologramme normalisé (gris, ± contrast autour de 1) ; chaque bactérie entourée, couleur
    selon z (échelle 0 – z_max en bas à droite), z en µm écrit à côté.
    """
    import matplotlib
    from PIL import Image, ImageDraw, ImageFont
    g = np.clip((h - 1.0) / (2 * contrast) + 0.5, 0, 1)
    im = Image.fromarray((g * 255).astype(np.uint8)).convert('RGB')
    dr = ImageDraw.Draw(im)
    fdir = os.path.join(os.path.dirname(matplotlib.__file__), 'mpl-data', 'fonts', 'ttf')
    font = ImageFont.truetype(os.path.join(fdir, 'DejaVuSans.ttf'), 10)
    cmap = matplotlib.colormaps['turbo']
    if res['instances']:
        pts = np.array([m['center'] if m else d['center'] for d, m in zip(res['instances'], res['measures'])])
    else:
        pts = np.asarray(res['pos1']).reshape(-1, 3)
    col = lambda z: tuple(int(255 * v) for v in cmap(float(np.clip(z / z_max, 0, 1)))[:3])
    for x, y, z in pts:
        cx, cy = x / dx_um, y / dx_um
        c = col(z)
        dr.ellipse([cx - 7, cy - 7, cx + 7, cy + 7], outline=c, width=2)
        dr.text((cx + 8, cy - 13), f"{z:.1f}", fill=c, font=font, stroke_width=1, stroke_fill=(0, 0, 0))
    W, H = im.size
    x0, y0, bw, bh = W - 190, H - 34, 160, 10                   # échelle de couleur de z
    for i in range(bw):
        dr.line([x0 + i, y0, x0 + i, y0 + bh], fill=col(z_max * i / (bw - 1)))
    dr.rectangle([x0 - 1, y0 - 1, x0 + bw, y0 + bh], outline=(255, 255, 255))
    for v in (0, z_max / 2, z_max):
        dr.text((x0 + bw * v / z_max - 6, y0 + bh + 2), f"{v:.0f}", fill=(255, 255, 255), font=font,
                stroke_width=1, stroke_fill=(0, 0, 0))
    dr.text((x0 - 52, y0 - 1), "z (µm)", fill=(255, 255, 255), font=font, stroke_width=1, stroke_fill=(0, 0, 0))
    dr.text((8, 6), f"{len(pts)} bactéries", fill=(255, 255, 255), font=font, stroke_width=1, stroke_fill=(0, 0, 0))
    im.save(path, format='BMP')


def save_instances_npz(path, res, pipeline):
    g = dict(dx_um=pipeline.optics['dx_um'], dz_um=pipeline.fine['dz_um'])
    np.savez_compressed(path, origin=np.array([d['origin'] for d in res['instances']]).reshape(-1, 3),
                        masks=np.array([d['mask'] for d in res['instances']]),
                        grid=json.dumps(dict(dx_um=g['dx_um'], dz_um=g['dz_um'],
                                             note="voxel (k, r, c) de l'instance i <-> (z, y, x) = "
                                                  "((origin_k + k)·dz, (origin_r + r)·dx, (origin_c + c)·dx)")))


# =============================================================================
# TRAITEMENT D'UN LOT (fonction de travail du TaskRunner)
# =============================================================================

def run_batch(rep, pipeline, params, cache):
    """
    params : folder, ext, mean_removal (MEAN_REMOVAL), first, last (inclus, −1 = fin), out_dir ('' = pas d'export),
    save_npz, save_bmp (hologramme + localisations), prior, rounds, thr1 (None = seuil du modèle),
    display (bool : envoyer chaque image au GUI, avec les fenêtres des instances pour les coupes au clic).
    Optique et reconstruction (plage z, pas, décimation, bande) : celles du pipeline (Pipeline.configure). Sorties : positions_unet3d.csv,
    inference_params.json, instances/*.npz, <image>_localisations.bmp.
    cache : dict partagé (moyenne du film).
    """
    import pandas as pd
    paths = list_frames(params['folder'], params['ext'])
    if not paths:
        raise FileNotFoundError(f"aucune image *.{params['ext']} dans {params['folder']}")
    first = max(0, int(params['first']))
    last = len(paths) - 1 if int(params['last']) < 0 else min(int(params['last']), len(paths) - 1)
    mean = None
    removal = params.get('mean_removal', params.get('normalization', 'division'))
    if uses_film_mean(removal):
        key = (params['folder'], params['ext'], len(paths))
        if cache.get('mean_key') != key:
            rep.log(f"moyenne du film ({len(paths)} images)…")
            cache['mean'], cache['mean_key'] = film_mean(paths, rep), key
        mean = cache['mean']
    out = params.get('out_dir') or ''
    if out:
        os.makedirs(out, exist_ok=True)
        if params.get('save_npz') and pipeline.m2 is not None:
            os.makedirs(os.path.join(out, 'instances'), exist_ok=True)
    film = None
    if pipeline.cfg1.get('norm') in tr.FILM_NORMS:        # une échelle pour tout le film, recalculée si le film change
        key = (params['folder'], params['ext'], len(paths), removal, json.dumps(pipeline.grid, sort_keys=True),
               json.dumps(pipeline.recon, sort_keys=True), json.dumps(pipeline.optics, sort_keys=True), pipeline.film_holos)
        if cache.get('film_key') != key:
            cache['film_stats'], cache['film_key'] = pipeline.film_stats(paths, removal, mean, rep), key
        film = cache['film_stats']
    rows, times, prior = [], [], None
    todo = list(range(first, last + 1))
    for n_done, f in enumerate(todo):
        rep.check_stop()
        name = os.path.basename(paths[f])
        h = remove_mean(read_image(paths[f]), removal, mean)
        display = params.get('display', True)
        res = pipeline.process(h, thr1=params.get('thr1'), rounds=int(params.get('rounds', 2)), film_stats=film,
                               prior_centers=prior if params.get('prior') else None, keep_windows=display)
        if pipeline.m2 is not None:
            prior = np.array([m['center'] if m else d['center'] for d, m in zip(res['instances'], res['measures'])])
        rows += result_rows(f, name, res)
        times.append(res['timings'])
        if out and params.get('save_npz') and pipeline.m2 is not None:
            save_instances_npz(os.path.join(out, 'instances', f"{os.path.splitext(name)[0]}_instances.npz"), res, pipeline)
        if out and params.get('save_bmp'):
            save_overlay_bmp(os.path.join(out, f"{os.path.splitext(name)[0]}_localisations.bmp"), h, res,
                             pipeline.optics['dx_um'], pipeline.z_max_um or pipeline.z_range_um[1])
        tt = res['timings']
        n_obj = len(res['instances']) if pipeline.m2 is not None else len(res['pos1'])
        rep.log(f"[{n_done + 1}/{len(todo)}] {name} : {len(res['pos1'])} détections, "
                + (f"{len(res['instances'])} instances, " if pipeline.m2 is not None else "")
                + f"{tt['total'] * 1000:.0f} ms")
        rep.progress(n_done + 1, len(todo), name)
        if display:
            payload = dict(index=f, name=name, holo=(h - 1.0).astype(np.float16), pos1=res['pos1'], timings=tt,
                           n_objects=n_obj, z_range=pipeline.z_range_um)
            if pipeline.m2 is not None:
                xy, xz, yz = pipeline.label_projections(res, h.shape[0])
                payload.update(label_xy=xy, label_xz=xz, label_yz=yz,
                               centers=np.array([m['center'] if m else d['center'] for d, m in zip(res['instances'], res['measures'])]).reshape(-1, 3),
                               rounds=np.array([d['round'] for d in res['instances']]),
                               dz_fine=pipeline.fine['dz_um'], dx=pipeline.optics['dx_um'],
                               # coupes au clic : amplitude refocalisée autour de chaque instance + masques
                               inst_origin=np.array([d['origin'] for d in res['instances']]).reshape(-1, 3),
                               inst_masks=[d['mask'] for d in res['instances']],
                               measures=[m for m in res['measures']], scores=[d['score'] for d in res['instances']])
                if res['windows'] is not None:
                    payload.update(win_amp=res['windows'][0], win_org=res['windows'][1])
            else:
                payload['dx'] = pipeline.optics['dx_um']
            rep.frame(payload)
    summary = dict(frames=len(todo), objects=len(rows))
    if times:
        tm = pd.DataFrame(times)
        summary['timings_ms_median'] = (tm.median() * 1000).round(1).to_dict()
    if out:
        csv = os.path.join(out, 'positions_unet3d.csv')
        pd.DataFrame(rows).to_csv(csv, index=False)
        with open(os.path.join(out, 'inference_params.json'), 'w', encoding='utf-8') as fh:
            json.dump(dict(params=params, models=pipeline.describe(), optics=pipeline.optics, reconstruction=pipeline.recon,
                           grid=pipeline.grid, peak_box=pipeline.peak_box, film_holos=pipeline.film_holos,
                           stage2=pipeline.infer, summary=summary), fh, indent=2, ensure_ascii=False,
                      default=str)
        summary['csv'] = csv
    return summary


def main():
    ap = argparse.ArgumentParser(description="Inférence du U-Net 3D (positions + instances) sur un film")
    ap.add_argument('folder', help="dossier des hologrammes (une image par fichier, ordre alphabétique)")
    ap.add_argument('out_dir', help="dossier de sortie (CSV, instances .npz, BMP des localisations)")
    ap.add_argument('--ext', default='bmp', help="extension des images (bmp, tif, png ; * : toutes)")
    ap.add_argument('--stage1', default=os.path.join(DL_DIR, 'models', 'stage1'), help="run de l'étage 1 (model_best.pt)")
    ap.add_argument('--stage2', default=os.path.join(DL_DIR, 'models', 'stage2'),
                    help="run de l'étage 2 ; chaîne vide : positions seules")
    ap.add_argument('--mean_removal', choices=list(MEAN_REMOVAL), default='division',
                    help="suppression de l'hologramme moyen du film : division, subtraction ; ou image_mean, none")
    ap.add_argument('--first', type=int, default=0, help="première image (indice, 0 = première)")
    ap.add_argument('--last', type=int, default=-1, help="dernière image incluse (−1 : fin du film)")
    ap.add_argument('--prior', action='store_true', help="a priori : centres de l'image précédente ajoutés aux candidats")
    ap.add_argument('--thr1', type=float, default=None, help="seuil de l'étage 1 ; défaut : celui de la validation")
    ap.add_argument('--rounds', type=int, default=2, help="tours de l'étage 2 (2 : sépare les bactéries collées)")
    ap.add_argument('--peak_box', type=int, nargs=3, default=PEAK_BOX,
                    help="étage 1 : fenêtre des maxima locaux (Z, Y, X), voxels grossiers impairs ; défaut : celle du modèle")
    ap.add_argument('--film_holos', type=int, default=FILM_STATS_HOLOS,
                    help="images pour les statistiques d'une normalisation « film » ; défaut : celles du modèle")
    ap.add_argument('--z_margin_um', type=float, default=None,
                    help=f"plans reconstruits au-delà de la plage de recherche, de chaque côté (µm) ; défaut {Z_MARGIN_UM:g}")
    ap.add_argument('--heat_thr', type=float, default=STAGE2['heat_thr'], help="étage 2 : seuil des maxima de la carte fine des centres")
    ap.add_argument('--nms', type=int, nargs=3, default=list(STAGE2['nms']),
                    help="étage 2 : fenêtre des maxima de la carte fine des centres (Z, Y, X), voxels fins impairs")
    ap.add_argument('--border', type=int, default=STAGE2['border'], help="étage 2 : voxels ignorés au bord de chaque fenêtre")
    ap.add_argument('--same_xy_um', type=float, default=STAGE2['same_xy_um'],
                    help="étage 2 : deux centres plus proches que cela en XY sont le même (second tour)")
    ap.add_argument('--same_z_um', type=float, default=STAGE2['same_z_um'], help="idem en Z (µm)")
    ap.add_argument('--vmin', type=int, default=None, help="étage 2 : voxels minimum d'une instance ; défaut : celui du modèle")
    ap.add_argument('--dup_iou', type=float, default=STAGE2['dup_iou'], help="doublons : IoU des masques au-delà de laquelle on fusionne")
    ap.add_argument('--dup_xy_um', type=float, default=STAGE2['dup_xy_um'], help="doublons : voisinage testé en XY (µm)")
    ap.add_argument('--dup_z_um', type=float, default=STAGE2['dup_z_um'], help="doublons : voisinage testé en Z (µm)")
    ap.add_argument('--z_min', type=float, default=0.0, help="profondeur minimale de recherche (µm)")
    ap.add_argument('--z_max', type=float, default=None, help="profondeur maximale de recherche (µm)")
    ap.add_argument('--z_step', type=float, default=None, help="pas Z de reconstruction (µm) ; défaut : celui du modèle")
    ap.add_argument('--decim', type=int, default=None, help="décimation latérale ; défaut : celle du modèle (2)")
    ap.add_argument('--high_pass', type=float, default=None, help="passe-haut (px de Fourier du capteur) ; défaut 15")
    ap.add_argument('--low_pass', type=float, default=None, help="passe-bas (px de Fourier du capteur) ; défaut 250")
    ap.add_argument('--dx_um', type=float, default=None, help="pixel objet (µm) = pixel caméra / grossissement")
    ap.add_argument('--wavelength_um', type=float, default=None, help="longueur d'onde dans le vide (µm)")
    ap.add_argument('--index_medium', type=float, default=None, help="indice du milieu")
    ap.add_argument('--no_bmp', action='store_true', help="pas d'images BMP avec les localisations")
    ap.add_argument('--no_npz', action='store_true', help="pas d'instances .npz")
    args = ap.parse_args()

    class _Rep:
        def log(self, m): print(m, flush=True)
        def progress(self, *a): pass
        def frame(self, d): pass
        def check_stop(self): pass

    p = Pipeline(args.stage1, args.stage2 if args.stage2 and os.path.isdir(args.stage2) else None,
                 optics=dict(dx_um=args.dx_um, wavelength_um=args.wavelength_um, index_medium=args.index_medium),
                 recon=dict(z_min_um=args.z_min, z_max_um=args.z_max, step_z_um=args.z_step, decim=args.decim,
                            high_pass=args.high_pass, low_pass=args.low_pass, z_margin_um=args.z_margin_um))
    p.set_post(peak_box=args.peak_box, film_holos=args.film_holos, heat_thr=args.heat_thr, nms=args.nms,
               border=args.border, same_xy_um=args.same_xy_um, same_z_um=args.same_z_um, vmin=args.vmin,
               dup_iou=args.dup_iou, dup_xy_um=args.dup_xy_um, dup_z_um=args.dup_z_um)
    print(p.describe())
    s = run_batch(_Rep(), p, dict(folder=args.folder, ext=args.ext, mean_removal=args.mean_removal, first=args.first,
                                  last=args.last, out_dir=args.out_dir, save_npz=not args.no_npz,
                                  save_bmp=not args.no_bmp, prior=args.prior, rounds=args.rounds, thr1=args.thr1,
                                  display=False), {})
    print(json.dumps(s, indent=2, default=str))


if __name__ == '__main__':
    main()

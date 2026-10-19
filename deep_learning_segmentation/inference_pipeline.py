# -*- coding: utf-8 -*-

"""
Filename: inference_pipeline.py

Description:
Inférence de la chaîne U-Net 3D sur un film ou un hologramme, sans interface (utilisée par
GUI_inference.py) :
  - lecture des images (BMP, TIFF, PNG), normalisation : division par la moyenne du film (comme
    HOLOTRACKER et Hybrid_REAL_SIMU_HOLOGRAMS), par la moyenne de l'image, ou aucune (déjà normalisé) ;
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
NORMALIZATION = {'film_mean': "division par la moyenne du film", 'image_mean': "division par la moyenne de l'image",
                 'none': "aucune (image déjà normalisée, fond ≈ 1)"}
DEFAULT_OPTICS = dict(dx_um=0.175, wavelength_um=0.66, index_medium=1.33)


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


def normalize(raw, mode, mean=None):
    if mode == 'film_mean':
        return raw / np.maximum(mean, 1e-6)
    if mode == 'image_mean':
        return raw / max(float(raw.mean()), 1e-6)
    return raw


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
    def __init__(self, stage1_dir, stage2_dir=None, optics=None, z_max_um=None):
        self.m1, self.cfg1, self.thr1 = load_stage1(stage1_dir)
        self.model_grid = dict(self.cfg1['grid'])
        self.optics = dict(DEFAULT_OPTICS, **(optics or {}))
        self.m2, self.cfg2 = load_stage2(stage2_dir) if stage2_dir else (None, None)
        self.infer = dict(s2.INFER, vmin=self.cfg2.get('vmin', s2.INFER['vmin'])) if self.cfg2 else None
        self.set_z_max(z_max_um)

    def set_z_max(self, z_max_um):
        """
        Profondeur maximale de recherche (µm ; None : toute la plage du modèle). Seuls les plans utiles sont
        reconstruits (marge de 8 µm, nombre de plans multiple de 2^niveaux), puis les détections et les instances
        au-delà de z_max sont retirées.
        """
        g = dict(self.model_grid)
        self.z_max_um = float(z_max_um) if z_max_um not in (None, '', 0) else None
        if self.z_max_um is not None:
            mult = 2 ** int(self.cfg1['depth'])
            need = (self.z_max_um + 8.0 - g['z0_um']) / g['step_z_um'] + 1
            g['nb_planes'] = int(min(g['nb_planes'], np.ceil(need / mult) * mult))
        self.grid = g
        self._rec = {}
        self.z_range_um = (g['z0_um'], g['z0_um'] + (g['nb_planes'] - 1) * g['step_z_um'])

    def describe(self):
        g = self.grid
        zmax = f", recherche jusqu'à {self.z_max_um:.0f} µm" if self.z_max_um is not None else ''
        s = (f"étage 1 : {self.cfg1.get('name', '?')} ({self.cfg1['loss']}, z {self.z_range_um[0]:.0f}–{self.z_range_um[1]:.0f} µm"
             f"{zmax}, pas {g['step_z_um']} µm, seuil {self.thr1})")
        if self.m2 is not None:
            s += f" ; étage 2 : {self.cfg2.get('name', '?')} (grille {self.cfg2['mask_grid']['dx_um']} × {self.cfg2['mask_grid']['dz_um']} µm)"
        return s

    def reconstructor(self, n):
        if n not in self._rec:
            z = self.grid['z0_um'] + np.arange(self.grid['nb_planes']) * self.grid['step_z_um']
            rec = hp.Reconstructor(n, self.optics['dx_um'], self.optics['wavelength_um'], self.optics['index_medium'], z,
                                   decim=self.grid['decim'])
            self._rec[n] = (rec, s2.LocalRefocus(rec) if self.m2 is not None else None)
        return self._rec[n]

    @torch.no_grad()
    def process(self, holo_norm, thr1=None, rounds=2, prior_centers=None, keep_windows=False):
        """
        holo_norm : (n, n), fond ≈ 1. Retourne positions de l'étage 1, instances, mesures, temps (s) et, si
        keep_windows, l'amplitude du champ refocalisé autour de chaque instance (coupes affichées par le GUI).
        """
        if holo_norm.shape[0] != holo_norm.shape[1] or holo_norm.shape[0] % 16:
            raise ValueError(f"hologramme {holo_norm.shape} : carré, côté multiple de 16 attendu")
        tr.GRID.update(self.grid)
        rec, refocus = self.reconstructor(holo_norm.shape[0])
        t = {}
        torch.cuda.synchronize(); t0 = time.perf_counter()
        V = rec(holo_norm)
        torch.cuda.synchronize(); t1 = time.perf_counter()
        logits = det.infer_volume(self.m1, tr.to_input(V, self.cfg1))
        torch.cuda.synchronize(); t2 = time.perf_counter()
        pos1, sc1 = tr.detect(logits, self.cfg1['target'], self.thr1 if thr1 is None else thr1, rec)
        del logits
        if self.z_max_um is not None and len(pos1):
            keep = pos1[:, 2] <= self.z_max_um
            pos1, sc1 = pos1[keep], np.asarray(sc1)[keep]
        torch.cuda.synchronize(); t3 = time.perf_counter()
        t.update(reconstruction=t1 - t0, network1=t2 - t1, peaks=t3 - t2)
        insts, meas, windows = [], [], None
        if self.m2 is not None:
            cand = pos1
            if prior_centers is not None and len(prior_centers):
                cand = np.concatenate([pos1, np.asarray(prior_centers)[:, :3]])
            insts = s2.segment_instances(self.m2, V, cand, refocus, cfg=self.infer, rounds=rounds)
            if self.z_max_um is not None:
                insts = [d for d in insts if d['center'][2] <= self.z_max_um]
            torch.cuda.synchronize(); t4 = time.perf_counter()
            g = self.cfg2['mask_grid']
            for d in insts:
                m = s2.mask_measures(d['mask'], d['origin'], g['dx_um'], g['dz_um'])
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
        g = self.cfg2['mask_grid']
        nzf = int(round((self.z_range_um[1] + 8) / g['dz_um'])) + 1
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
    g = pipeline.cfg2['mask_grid']
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
    params : folder, ext, normalization, first, last (inclus, −1 = fin), out_dir ('' = pas d'export),
    save_npz, save_bmp (hologramme + localisations), prior, rounds, thr1 (None = seuil du modèle),
    display (bool : envoyer chaque image au GUI, avec les fenêtres des instances pour les coupes au clic).
    La profondeur maximale est celle du pipeline (Pipeline.set_z_max). Sorties : positions_unet3d.csv,
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
    if params['normalization'] == 'film_mean':
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
    rows, times, prior = [], [], None
    todo = list(range(first, last + 1))
    for n_done, f in enumerate(todo):
        rep.check_stop()
        name = os.path.basename(paths[f])
        h = normalize(read_image(paths[f]), params['normalization'], mean)
        display = params.get('display', True)
        res = pipeline.process(h, thr1=params.get('thr1'), rounds=int(params.get('rounds', 2)),
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
                               dz_fine=pipeline.cfg2['mask_grid']['dz_um'], dx=pipeline.optics['dx_um'],
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
            json.dump(dict(params=params, models=pipeline.describe(), summary=summary), fh, indent=2, ensure_ascii=False,
                      default=str)
        summary['csv'] = csv
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('folder')
    ap.add_argument('out_dir')
    ap.add_argument('--ext', default='bmp')
    ap.add_argument('--stage1', default=os.path.join(DL_DIR, 'models', 'stage1'))
    ap.add_argument('--stage2', default=os.path.join(DL_DIR, 'models', 'stage2'))
    ap.add_argument('--normalization', choices=list(NORMALIZATION), default='film_mean')
    ap.add_argument('--first', type=int, default=0)
    ap.add_argument('--last', type=int, default=-1)
    ap.add_argument('--prior', action='store_true')
    ap.add_argument('--z_max', type=float, default=None, help="profondeur maximale de recherche (µm)")
    ap.add_argument('--no_bmp', action='store_true', help="pas d'images BMP avec les localisations")
    ap.add_argument('--no_npz', action='store_true', help="pas d'instances .npz")
    args = ap.parse_args()

    class _Rep:
        def log(self, m): print(m, flush=True)
        def progress(self, *a): pass
        def frame(self, d): pass
        def check_stop(self): pass

    p = Pipeline(args.stage1, args.stage2 if args.stage2 and os.path.isdir(args.stage2) else None, z_max_um=args.z_max)
    print(p.describe())
    s = run_batch(_Rep(), p, dict(folder=args.folder, ext=args.ext, normalization=args.normalization, first=args.first,
                                  last=args.last, out_dir=args.out_dir, save_npz=not args.no_npz,
                                  save_bmp=not args.no_bmp, prior=args.prior, rounds=2, display=False), {})
    print(json.dumps(s, indent=2, default=str))


if __name__ == '__main__':
    main()

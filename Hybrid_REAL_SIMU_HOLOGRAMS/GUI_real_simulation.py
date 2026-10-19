# -*- coding: utf-8 -*-

"""
Filename: GUI_real_simulation.py

Description:
Interface graphique de real_simulation.py (simulations construites sur un film réel) :
  - à gauche, onglets de paramètres : entrées (film, trajectoires, moyenne), optique et fond,
    nettoyage, simulation, sorties (cases à cocher) ; boutons Aperçu / Lancer / Arrêter ;
  - à droite, aperçu de l'image courante, choisie dans un menu déroulant (normalisée, fond,
    bactéries nettoyées seules, bactéries simulées seules, simulées + fond ; le zoom est conservé
    d'une image à l'autre) et journal.

« Aperçu » traite l'image choisie (n°, ◀ / ▶) sans rien écrire ; la moyenne et les orientations
sont gardées d'un aperçu à l'autre ; « Lancer » traite le film et
enregistre les sorties cochées. Les derniers réglages sont mémorisés
(GUI_real_simulation_params.json, à côté du script) ; « Enregistrer config » écrit un JSON
utilisable par le script en ligne de commande. Le calcul tourne dans un thread.

Utilisation :
  python Hybrid_REAL_SIMU_HOLOGRAMS/GUI_real_simulation.py

Author: Simon BECKER
Date: 2026-10-17

License:
GNU General Public License v3.0
"""

import json
import os
import queue
import sys
import threading
import traceback

import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real_simulation as rs

PARAMS_FILE = os.path.join(rs.REAL_DIR, "GUI_real_simulation_params.json")
HOLOTRACKER_PARAMS = os.path.normpath(os.path.join(rs.REPO_DIR, "..", "HOLOTRACKER", "last_param.json"))

# -----------------------------------------------------------------------------
# Contrôles : (chemin dans la config, libellé, type, aide).
#   type : 'dir', 'file', 'str', 'int', 'float', 'bool', ou liste de choix
#   un tuple ('bool', clé) en 5e position désactive le contrôle si la case est décochée
# -----------------------------------------------------------------------------
TABS = {
    "Entrées": [
        (('film_dir',), "Dossier du film", 'dir', "Images du film réel (une par hologramme)"),
        (('image_extension',), "Extension des images", ['bmp', 'tif', 'tiff', 'png'], "Format des images du film"),
        (('trajectories_csv',), "Trajectoires (CSV)", 'file',
         "CSV HOLOTRACKER : HOLOGRAM NUMBER, X/Y/Z POSITION (m), NUMBER OF VOXEL, TRAJECTORY NUMBER (-1 : hors trajectoire)"),
        (('first_hologram_number',), "N° du 1er hologramme", 'int',
         "HOLOGRAM NUMBER de la première image du film (ordre alphabétique). HOLOTRACKER : 1"),
        (('first_frame',), "Première image traitée", 'int', "Indice dans le film (0 = début)"),
        (('max_frames',), "Nombre d'images", 'int', "0 = toutes"),
        (('output_dir',), "Dossier de sortie", 'dir', "Un sous-dossier <film>_<date> y est créé"),
        (('background', 'mean_file'), "Hologramme moyen", 'file',
         "Vide : moyenne arithmétique calculée sur tout le film (float, sans arrondi 8 bits)"),
    ],
    "Optique & fond": [
        (('optics', 'wavelength_nm'), "Longueur d'onde (nm)", 'float', "Dans le vide"),
        (('optics', 'index_medium'), "Indice du milieu", 'float', "Eau : 1,33"),
        (('optics', 'pixel_size_um'), "Pixel caméra (µm)", 'float', "Taille du pixel du capteur"),
        (('optics', 'magnification'), "Grossissement", 'float', "De l'objectif"),
        (('background', 'removal'), "Retrait de la moyenne", ['division', 'subtraction'],
         "division : I / M ; subtraction : (I − M) / <M> + 1. Fond ≈ 1 dans les deux cas"),
    ],
    "Nettoyage": [
        (('cleaning', 'enabled'), "Nettoyer les bactéries détectées", 'bool',
         "Propagation au plan z, fenêtre, rétropropagation, soustraction de 2·Re(p) (image jumelle comprise)"),
        (('cleaning', 'window_radius_um'), "Rayon de la fenêtre (µm)", 'float',
         "Optimum mesuré en simulation : 1,5 µm (plus large : l'image jumelle et les voisines entrent)", ('cleaning', 'enabled')),
        (('cleaning', 'window_taper_um'), "Bord en cosinus (µm)", 'float', "Au-delà du rayon", ('cleaning', 'enabled')),
        (('cleaning', 'passes'), "Passes", 'int', "1 : 11 % de signal restant, 2 : 6 %, 3 : 5 % (simulation)", ('cleaning', 'enabled')),
        (('cleaning', 'high_pass'), "Rayon min du spectre (px)", 'int', "Filtre de l'extraction (0 = aucun)", ('cleaning', 'enabled')),
        (('cleaning', 'low_pass'), "Rayon max du spectre (px)", 'int', "0 = aucun", ('cleaning', 'enabled')),
        (('cleaning', 'gain'), "Gain", 'float', "Fraction de 2·Re(p) soustraite", ('cleaning', 'enabled')),
        (('cleaning', 'min_z_um'), "Z minimal (µm)", 'float',
         "Objets plus proches ignorés : leur image jumelle tombe dans la fenêtre", ('cleaning', 'enabled')),
        (('cleaning', 'clean_unassigned'), "Nettoyer les détections hors trajectoire", 'bool', "", ('cleaning', 'enabled')),
        (('cleaning', 'unassigned_min_voxel'), "… d'au moins (voxels)", 'int', "", ('cleaning', 'clean_unassigned')),
        (('cleaning', 'residual', 'enabled'), "CLEAN des restes", 'bool',
         "Maxima locaux (SUM_OF_VARIANCE, seuil local) de l'hologramme nettoyé, soustraits à leur tour :\n"
         "bactéries non détectées par HOLOTRACKER, fantômes en Z", ('cleaning', 'enabled')),
        (('cleaning', 'residual', 'iterations'), "Itérations", 'int', "", ('cleaning', 'residual', 'enabled')),
        (('cleaning', 'residual', 'k_local'), "K du seuil local", 'float', "SEGMENTATION_REPORT : 15", ('cleaning', 'residual', 'enabled')),
        (('cleaning', 'residual', 'distance_ini_um'), "Distance 1er plan (µm)", 'float', "", ('cleaning', 'residual', 'enabled')),
        (('cleaning', 'residual', 'nb_planes'), "Nombre de plans", 'int', "Multiple de 25 (maille du seuil local)", ('cleaning', 'residual', 'enabled')),
        (('cleaning', 'residual', 'step_um'), "Pas Z (µm)", 'float', "", ('cleaning', 'residual', 'enabled')),
        (('cleaning', 'residual', 'max_objects'), "Maxima par itération (max)", 'int', "Par RSB décroissant", ('cleaning', 'residual', 'enabled')),
        (('cleaning', 'measure_detections'), "Compter les maxima avant / après", 'bool',
         "Indicateur de qualité (≈ 4 s de plus par image) : detections_per_frame.csv", ('cleaning', 'enabled')),
    ],
    "Simulation": [
        (('simulation', 'enabled'), "Ajouter des bactéries simulées", 'bool', "Aux positions des trajectoires"),
        (('simulation', 'length_um'), "Longueur (µm)", 'float', "", ('simulation', 'enabled')),
        (('simulation', 'thickness_um'), "Épaisseur (µm)", 'float', "", ('simulation', 'enabled')),
        (('simulation', 'delta_n'), "Δn (objet − milieu)", 'float',
         "0,043 : calé sur le film de référence (amplitude réelle / simulée = 1).\n"
         "Le bilan donne le Δn équivalent mesuré sur le film traité.", ('simulation', 'enabled')),
        (('simulation', 'step_um'), "Pas BPM (µm)", 'float', "", ('simulation', 'enabled')),
        (('simulation', 'upscale_factor'), "Sur-échantillonnage", 'int', "Du masque, en XY", ('simulation', 'enabled')),
        (('simulation', 'orientation_half_window'), "Demi-fenêtre d'orientation (images)", 'int',
         "Ajustement linéaire de la trajectoire sur ±W images (déplacement médian 0,5 µm / image)", ('simulation', 'enabled')),
        (('simulation', 'min_points_orientation'), "Points min. pour l'orientation", 'int',
         "En dessous : orientation aléatoire (isotrope)", ('simulation', 'enabled')),
        (('simulation', 'simulate_unassigned'), "Simuler aussi les détections hors trajectoire", 'bool',
         "Orientation aléatoire", ('simulation', 'enabled')),
        (('simulation', 'unassigned_min_voxel'), "… d'au moins (voxels)", 'int', "", ('simulation', 'simulate_unassigned')),
        (('simulation', 'measure_contrast'), "Mesurer l'amplitude réelle / simulée", 'bool',
         "Calibration de Δn (ground_truth.csv, summary.json)", ('simulation', 'enabled')),
        (('simulation', 'seed'), "Graine aléatoire", 'int', "", ('simulation', 'enabled')),
    ],
    "Sorties": [
        (('outputs', 'mean'), "Hologramme moyen", 'bool', "mean/mean.tif et mean.bmp"),
        (('outputs', 'background'), "Film de fond (nettoyé)", 'bool', "background/"),
        (('outputs', 'cleaned_bacteria'), "Bactéries nettoyées seules", 'bool', "cleaned_bacteria/ : ce qui a été retiré, + 1"),
        (('outputs', 'simulated_bacteria'), "Bactéries simulées seules", 'bool', "simulated_bacteria/ : fond 1, sans bruit"),
        (('outputs', 'simulated_on_background'), "Bactéries simulées + fond", 'bool', "simulated_on_background/"),
        (('outputs', 'bmp'), "Format BMP 8 bits", 'bool', "Normalisé × niveau de gris moyen du film (la moyenne temporelle n'est pas réintroduite).\n"
         "Lisible par HOLOTRACKER : fond plat, inutile de re-diviser par une moyenne"),
        (('outputs', 'tiff'), "Format TIFF float32", 'bool', "Normalisé (fond ≈ 1)"),
    ],
}
PREVIEW = [('normalized', "Normalisé"), ('background', "Fond"), ('cleaned_bacteria', "Nettoyées seules"),
           ('simulated_bacteria', "Simulées seules"), ('simulated_on_background', "Simulées + fond")]


def get_in(d, path):
    for k in path:
        d = d[k]
    return d


def set_in(d, path, v):
    for k in path[:-1]:
        d = d.setdefault(k, {})
    d[path[-1]] = v


class ToolTip:
    def __init__(self, widget, text):
        self.widget, self.text, self.tip = widget, text, None
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)

    def show(self, _event=None):
        if not self.text or self.tip:
            return
        x, y = self.widget.winfo_rootx() + 20, self.widget.winfo_rooty() + self.widget.winfo_height() + 2
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self.tip, text=self.text, justify='left', background="#ffffe0", relief='solid',
                 borderwidth=1, wraplength=420).pack()

    def hide(self, _event=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None


class RealSimulationGUI:

    def __init__(self, root):
        self.root = root
        self.root.title("Simulations sur film réel")
        self.root.geometry("1500x900")
        self.vars, self.widgets = {}, {}
        self.queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker = None

        main = ttk.PanedWindow(root, orient='horizontal')
        main.pack(fill='both', expand=True, padx=5, pady=5)
        left, right = ttk.Frame(main), ttk.Frame(main)
        main.add(left, weight=0)
        main.add(right, weight=1)
        self.init_left(left)
        self.init_right(right)
        self.set_config(self.load_params())
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.after(100, self.poll_queue)

    # ------------------------------------------------------------------ paramètres
    def init_left(self, parent):
        # boutons en haut : toujours visibles, quelle que soit la hauteur de l'écran
        bar = ttk.Frame(parent)
        bar.pack(fill='x', pady=(0, 3))
        self.btn_run = ttk.Button(bar, text="Lancer", command=lambda: self.start(save=True))
        self.btn_stop = ttk.Button(bar, text="Arrêter", command=self.stop, state='disabled')
        for b in (self.btn_run, self.btn_stop):
            b.pack(side='left', padx=3)
        # aperçu : une image du film, choisie par son indice ; ◀ / ▶ passent à la voisine et la calculent
        pbar = ttk.Frame(parent)
        pbar.pack(fill='x', pady=(0, 3))
        self.btn_preview = ttk.Button(pbar, text="Aperçu de l'image n°", command=lambda: self.start(save=False))
        self.btn_preview.pack(side='left', padx=3)
        self.preview_index = tk.StringVar(value="0")
        spin = ttk.Spinbox(pbar, from_=0, to=10 ** 6, textvariable=self.preview_index, width=6)
        spin.pack(side='left')
        spin.bind("<Return>", lambda _e: self.start(save=False))
        ToolTip(spin, "Indice de l'image dans le film (0 = première), indépendant de « Première image traitée ».\n"
                      "Entrée : calcule l'aperçu")
        self.btn_prev = ttk.Button(pbar, text="◀", width=3, command=lambda: self.step_preview(-1))
        self.btn_next = ttk.Button(pbar, text="▶", width=3, command=lambda: self.step_preview(+1))
        self.btn_prev.pack(side='left', padx=(6, 0))
        self.btn_next.pack(side='left')
        self.preview_name = ttk.Label(pbar, text="")
        self.preview_name.pack(side='left', padx=6)
        self.cache = {}     # moyenne et orientations gardées d'un aperçu à l'autre
        bar2 = ttk.Frame(parent)
        bar2.pack(fill='x', pady=(0, 5))
        ttk.Button(bar2, text="Charger config…", command=self.load_config_file).pack(side='left', padx=3)
        ttk.Button(bar2, text="Enregistrer config…", command=self.save_config_file).pack(side='left', padx=3)
        ttk.Button(bar2, text="Valeurs par défaut", command=lambda: self.set_config(rs.DEFAULT_CONFIG)).pack(side='left', padx=3)
        nb = ttk.Notebook(parent)
        nb.pack(fill='both', expand=True)
        for tab_name, rows in TABS.items():
            tab = ttk.Frame(nb, padding=8)
            nb.add(tab, text=tab_name)
            for r, spec in enumerate(rows):
                self.add_control(tab, r, *spec)
            tab.columnconfigure(1, weight=1)
            if tab_name == "Optique & fond":
                ttk.Button(tab, text="Importer last_param.json d'HOLOTRACKER…",
                           command=self.import_holotracker).grid(row=len(rows), column=0, columnspan=3,
                                                                 sticky='w', pady=10)
        self.update_states()

    def add_control(self, parent, row, path, label, kind, help_text, enable=None):
        if kind == 'bool':
            var = tk.BooleanVar()
            w = ttk.Checkbutton(parent, text=label, variable=var, command=self.update_states)
            w.grid(row=row, column=0, columnspan=3, sticky='w', pady=2)
        else:
            var = tk.StringVar()
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky='w', pady=2, padx=(0, 6))
            if isinstance(kind, list):
                w = ttk.Combobox(parent, textvariable=var, values=kind, state='readonly', width=14)
                w.grid(row=row, column=1, sticky='w')
            else:
                w = ttk.Entry(parent, textvariable=var, width=40 if kind in ('dir', 'file', 'str') else 14)
                w.grid(row=row, column=1, sticky='we' if kind in ('dir', 'file', 'str') else 'w')
            if kind in ('dir', 'file'):
                b = ttk.Button(parent, text="…", width=3, command=lambda: self.browse(path, kind))
                b.grid(row=row, column=2, padx=2)
                self.widgets[path + ('browse',)] = (b, enable)
        ToolTip(w, help_text)
        self.vars[path] = (var, kind)
        self.widgets[path] = (w, enable)

    def browse(self, path, kind):
        var = self.vars[path][0]
        initial = rs.resolve_path(var.get()) or rs.REPO_DIR
        if kind == 'dir':
            p = filedialog.askdirectory(initialdir=initial if os.path.isdir(initial) else rs.REPO_DIR)
        else:
            p = filedialog.askopenfilename(initialdir=os.path.dirname(initial) if initial else rs.REPO_DIR)
        if p:
            var.set(p)

    def update_states(self):
        for path, (w, enable) in self.widgets.items():
            if enable is None:
                continue
            on = bool(self.vars[enable][0].get()) and self._parents_on(enable)
            try:
                w.configure(state=('readonly' if isinstance(w, ttk.Combobox) else 'normal') if on else 'disabled')
            except tk.TclError:
                pass

    def _parents_on(self, path):
        enable = self.widgets.get(path, (None, None))[1]
        return enable is None or (bool(self.vars[enable][0].get()) and self._parents_on(enable))

    def get_config(self):
        cfg = json.loads(json.dumps(rs.DEFAULT_CONFIG))
        errors = []
        for path, (var, kind) in self.vars.items():
            v = var.get()
            try:
                if kind == 'int':
                    v = int(float(str(v).replace(',', '.')))
                elif kind == 'float':
                    v = float(str(v).replace(',', '.'))
            except ValueError:
                errors.append(f"{'.'.join(path)} = {v!r}")
                continue
            set_in(cfg, path, v)
        if errors:
            raise ValueError("Valeurs numériques invalides : " + ", ".join(errors))
        return cfg

    def set_config(self, cfg):
        cfg = rs.merge_config(rs.DEFAULT_CONFIG, cfg)
        for path, (var, kind) in self.vars.items():
            try:
                var.set(get_in(cfg, path))
            except (KeyError, TypeError):
                pass
        self.update_states()

    def load_params(self):
        try:
            with open(PARAMS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (OSError, ValueError):
            return rs.DEFAULT_CONFIG

    def save_params(self):
        try:
            with open(PARAMS_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.get_config(), f, indent=2, ensure_ascii=False)
        except (OSError, ValueError):
            pass

    def load_config_file(self):
        p = filedialog.askopenfilename(initialdir=rs.REAL_DIR, filetypes=[("JSON", "*.json")])
        if p:
            self.set_config(rs.load_config(p))

    def save_config_file(self):
        p = filedialog.asksaveasfilename(initialdir=rs.REAL_DIR, defaultextension=".json", filetypes=[("JSON", "*.json")])
        if p:
            with open(p, 'w', encoding='utf-8') as f:
                json.dump(self.get_config(), f, indent=2, ensure_ascii=False)
            self.log(f"Configuration enregistrée : {p}")

    def import_holotracker(self):
        p = filedialog.askopenfilename(initialdir=os.path.dirname(HOLOTRACKER_PARAMS), filetypes=[("JSON", "*.json")])
        if not p:
            return
        with open(p, 'r', encoding='utf-8') as f:
            h = json.load(f)
        values = {('optics', 'wavelength_nm'): float(h['wavelength']) * 1e9,
                  ('optics', 'index_medium'): float(h['medium_optical_index']),
                  ('optics', 'pixel_size_um'): float(h['pixel_size']) * 1e6,
                  ('optics', 'magnification'): float(h['objective_magnification'])}
        if h.get('cleaning_type') in ('division', 'subtraction'):
            values[('background', 'removal')] = h['cleaning_type']
        for path, v in values.items():
            self.vars[path][0].set(round(v, 6) if isinstance(v, float) else v)
        self.log(f"Paramètres optiques importés de {p}")

    # ------------------------------------------------------------------ aperçu et journal
    def init_right(self, parent):
        # une seule image, choisie dans un menu ; journal en bas, figure au-dessus
        ctrl = ttk.Frame(parent)
        ctrl.pack(side='top', fill='x', pady=(0, 3))
        ttk.Label(ctrl, text="Image affichée").pack(side='left')
        self.view = tk.StringVar(value=PREVIEW[0][1])
        view_box = ttk.Combobox(ctrl, textvariable=self.view, values=[t for _, t in PREVIEW], state='readonly', width=22)
        view_box.pack(side='left', padx=(4, 15))
        view_box.bind("<<ComboboxSelected>>", lambda _e: self.draw_preview())
        # Haut / Bas : type précédent / suivant, affiché tout de suite (sans ouvrir la liste)
        view_box.bind("<Up>", lambda _e: self.step_view(-1))
        view_box.bind("<Down>", lambda _e: self.step_view(+1))
        ttk.Label(ctrl, text="Contraste ±").pack(side='left')
        self.contrast = tk.StringVar(value="0.1")
        ttk.Combobox(ctrl, textvariable=self.contrast, values=["0.02", "0.05", "0.1", "0.2", "0.5"],
                     width=6).pack(side='left', padx=4)
        self.contrast.trace_add('write', lambda *_: self.draw_preview())
        self.frame_label = ttk.Label(ctrl, text="")
        self.frame_label.pack(side='right', padx=5)
        self.text = tk.Text(parent, height=10, wrap='word')
        self.text.pack(side='bottom', fill='x')
        self.fig = Figure(figsize=(5, 5))
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.axis('off')
        self.fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        NavigationToolbar2Tk(self.canvas, parent).update()
        # taille demandée réduite : sinon le volet de droite réclame la taille de la figure et la
        # fenêtre la rogne au lieu de la redimensionner
        self.canvas.get_tk_widget().configure(width=300, height=200)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        self.last_frame = None

    def log(self, msg):
        self.text.insert('end', msg + "\n")
        self.text.see('end')

    def step_view(self, step):
        titles = [t for _, t in PREVIEW]
        i = titles.index(self.view.get()) if self.view.get() in titles else 0
        self.view.set(titles[(i + step) % len(titles)])
        self.draw_preview()
        return "break"      # empêche la Combobox d'ouvrir sa liste

    def draw_preview(self):
        f = self.last_frame
        if f is None:
            return
        try:
            c = float(self.contrast.get())
        except ValueError:
            return
        key = {t: k for k, t in PREVIEW}.get(self.view.get(), PREVIEW[0][0])
        # le zoom courant est conservé d'une image à l'autre : on compare la même région
        keep = self.ax.images and self.ax.images[0].get_array().shape == f[key].shape
        lims = (self.ax.get_xlim(), self.ax.get_ylim()) if keep else None
        self.ax.clear()
        self.ax.imshow(f[key], cmap='gray', vmin=1 - c, vmax=1 + c)
        if lims:
            self.ax.set_xlim(lims[0])
            self.ax.set_ylim(lims[1])
        self.ax.set_title(self.view.get(), fontsize=10)
        self.ax.axis('off')
        self.frame_label.configure(text=f["name"])
        self.canvas.draw_idle()

    # ------------------------------------------------------------------ exécution
    def start(self, save):
        if self.worker and self.worker.is_alive():
            return
        try:
            cfg = self.get_config()
            if not save:
                n_frames = len(rs.list_frames(cfg))
                idx = min(max(int(self.preview_index.get()), 0), max(n_frames - 1, 0))
                self.preview_index.set(str(idx))
                cfg['first_frame'], cfg['max_frames'] = idx, 1
            rs.check_config(cfg if save else rs.merge_config(cfg, {'outputs': {'bmp': True}}))
        except ValueError as e:
            messagebox.showerror("Paramètres", str(e))
            return
        self.save_params()
        self.stop_event.clear()
        for b in (self.btn_preview, self.btn_prev, self.btn_next, self.btn_run):
            b.configure(state='disabled')
        self.btn_stop.configure(state='normal')
        self.worker = threading.Thread(target=self._worker, args=(cfg, save), daemon=True)
        self.worker.start()

    def step_preview(self, step):
        try:
            idx = int(self.preview_index.get()) + step
        except ValueError:
            idx = 0
        self.preview_index.set(str(max(idx, 0)))
        self.start(save=False)

    def _worker(self, cfg, save):
        try:
            out = rs.run(cfg, progress=lambda m: self.queue.put(('log', m)), stop=self.stop_event,
                         frame_callback=lambda f: self.queue.put(('frame', f)), save=save, cache=self.cache)
            self.queue.put(('done', out))
        except Exception:
            self.queue.put(('error', traceback.format_exc()))

    def stop(self):
        self.stop_event.set()
        self.log("Arrêt demandé (fin de l'objet en cours)…")

    def poll_queue(self):
        try:
            while True:
                kind, data = self.queue.get_nowait()
                if kind == 'log':
                    self.log(data)
                elif kind == 'frame':
                    self.last_frame = data
                    self.preview_name.configure(text=data['name'])
                    self.draw_preview()
                else:
                    if kind == 'error':
                        self.log(data)
                        messagebox.showerror("Erreur", data.strip().splitlines()[-1])
                    for b in (self.btn_preview, self.btn_prev, self.btn_next, self.btn_run):
                        b.configure(state='normal')
                    self.btn_stop.configure(state='disabled')
        except queue.Empty:
            pass
        self.root.after(100, self.poll_queue)

    def on_closing(self):
        self.save_params()
        self.stop_event.set()
        self.root.destroy()


def main():
    root = tk.Tk()
    RealSimulationGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()

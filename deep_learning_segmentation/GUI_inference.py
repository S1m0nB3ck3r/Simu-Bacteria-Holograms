# -*- coding: utf-8 -*-

"""
Filename: GUI_inference.py

Description:
Interface d'inférence de la chaîne U-Net 3D (inference_pipeline.py) : positions 3D (étage 1) et volume
segmenté par instance (étage 2) sur un film holographique ou un hologramme.
  - Entrée : dossier d'images (BMP, TIFF, PNG), suppression de l'hologramme moyen du film (division ou
    soustraction ; ou aucune), plage d'images ; Modèles : dossiers des runs (model_best.pt) des deux étages ; Reconstruction : pixel
    caméra, grossissement, longueur d'onde, indice du milieu, plage de profondeur (z min, z max), pas Z,
    décimation latérale, filtre passe-bande (vides : valeurs d'entraînement du modèle ; tout écart est
    signalé) ; Réglages : seuil de l'étage 1, nombre de tours de l'étage 2, a priori temporel (centres de
    l'image précédente ajoutés aux candidats) ; Sortie : CSV (colonnes d'HOLOTRACKER + mesures), instances .npz.
  - « Image n° » calcule une image seule (◀ ▶ : voisines) ; « Lancer le lot » traite la plage et exporte.
  - Vues : hologramme + détections (z en µm), instances en projection XY, XZ, YZ ; Haut / Bas dans le
    menu changent de vue ; le zoom est conservé d'une image à l'autre. Un clic sur une bactérie (hors mode
    zoom / déplacement) ouvre ses trois coupes de segmentation XY, XZ, YZ par son centre : amplitude du champ
    refocalisé sur la grille fine, masque de la bactérie, masques des voisines.
  - Recherche entre z min et z max (0–100 µm par défaut, seuls les plans utiles sont reconstruits) ; BMP des
    localisations en sortie.
Le calcul tourne hors de la boucle Tk (gui_messages.TaskRunner, fil de travail : les modèles chargés sont
partagés) ; MessagePump relève les messages par paquets bornés et n'affiche que la dernière image reçue :
l'interface reste réactive pendant un lot.
Réglages mémorisés dans GUI_inference_params.json (à côté du script).

Utilisation :
  python deep_learning_segmentation/GUI_inference.py

Author: Simon BECKER
Date: 2026-10-18

License:
GNU General Public License v3.0
"""

import json
import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
import matplotlib.patheffects as pe

DL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(DL_DIR)
sys.path.insert(0, DL_DIR)
from gui_messages import MessagePump, TaskRunner

PARAMS_FILE = os.path.join(DL_DIR, "GUI_inference_params.json")
DEFAULTS = {
    'folder': os.path.join(REPO_DIR, 'REAL HOLO FILM', 'bacterias'), 'ext': 'bmp', 'mean_removal': 'division',
    'first': 0, 'last': -1,
    'stage1': os.path.join(DL_DIR, 'models', 'stage1'), 'stage2': os.path.join(DL_DIR, 'models', 'stage2'),
    'pix_size_um': 7.0, 'magnification': 40.0, 'wavelength_um': 0.66, 'index_medium': 1.33,
    'z_min': '0', 'z_max': '100', 'z_step': '', 'decim': 'modèle', 'high_pass': '', 'low_pass': '', 'z_margin': '',
    'thr1': '', 'rounds': 2, 'prior': False, 'peak_box': '', 'film_holos': '',
    'vmin': 20, 'heat_thr': 0.3, 'nms2': '5 5 5', 'border2': 2, 'same_xy': 0.5, 'same_z': 1.0,
    'dup_iou': 0.5, 'dup_xy': 1.5, 'dup_z': 3.0, 'save_bmp': True,
    'out_dir': os.path.join(DL_DIR, 'runs_inference'), 'save_npz': True,
}
REMOVAL = {'division': "division par l'hologramme moyen", 'subtraction': "soustraction de l'hologramme moyen",
           'image_mean': "aucune (÷ moyenne de l'image)", 'none': "aucune (fond déjà à 1)"}
TABS = {
    "Entrée": [
        ('folder', "Dossier du film", 'dir', "Dossier des hologrammes (une image par fichier, ordre alphabétique)"),
        ('ext', "Extension", ['bmp', 'tif', 'tiff', 'png', '*'], "Type d'image ; * : toutes"),
        ('mean_removal', "Suppression de l'hologramme moyen", list(REMOVAL.values()),
         "Hologramme moyen M = moyenne de toutes les images du film. Division I / M (comme HOLOTRACKER) : retire "
         "aussi un éclairage non uniforme et les poussières fixes. Soustraction (I − M) / <M> + 1 : équivalente si "
         "l'éclairage est uniforme, sinon le signal reste modulé par l'éclairage. Aucune : image seule divisée par "
         "sa moyenne (hologramme isolé, BMP simulé), ou rien si le fond vaut déjà 1 (TIFF du simulateur)"),
        ('first', "Première image", 'int', "Indice (0 = première) ; aussi l'image de départ de l'aperçu"),
        ('last', "Dernière image", 'int', "Indice inclus ; −1 = fin du film"),
    ],
    "Modèles": [
        ('stage1', "Étage 1 (positions)", 'dir', "Dossier d'un run de train.py (model_best.pt)"),
        ('stage2', "Étage 2 (instances)", 'dir', "Dossier d'un run de train_stage2.py ; vide : positions seules"),
    ],
    "Reconstruction": [
        ('pix_size_um', "Pixel caméra (µm)", 'float', "Pixel objet = pixel caméra / grossissement (7 µm / 40 = 0,175 µm)"),
        ('magnification', "Grossissement", 'float', ""),
        ('wavelength_um', "Longueur d'onde (µm)", 'float', "Dans le vide"),
        ('index_medium', "Indice du milieu", 'float', ""),
        ('z_min', "Profondeur minimale (µm)", 'str', "Distance de refocalisation à partir de laquelle on cherche"),
        ('z_max', "Profondeur maximale (µm)", 'str',
         "Seuls les plans utiles sont reconstruits (z min − 8 à z max + 8 µm) ; vide : jusqu'au bout de la plage du "
         "modèle (207 µm)"),
        ('z_step', "Pas Z (µm)", 'str', "Pas des plans reconstruits ; vide : celui de l'entraînement (1 µm). Un autre pas "
                                       "change ce que voit le réseau : à valider"),
        ('decim', "Décimation latérale", ['modèle', '1', '2', '4'],
         "Pixel du volume = pixel objet × décimation, par recadrage du spectre (sans perte si le passe-bas est ≤ "
         "taille / (2 × décimation)) ; modèle : celle de l'entraînement. L'étage 2 exige celle de son entraînement"),
        ('high_pass', "Passe-haut (px de Fourier)", 'str',
         "Rayon en pixels de Fourier du capteur en dessous duquel le spectre est coupé (fond lentement variable) ; "
         "vide : valeur d'entraînement du modèle (15)"),
        ('low_pass', "Passe-bas (px de Fourier)", 'str',
         "Rayon au-delà duquel le spectre est coupé (bruit haute fréquence) ; vide : valeur d'entraînement (250)"),
        ('z_margin', "Marge de reconstruction (µm)", 'str',
         "Plans reconstruits de part et d'autre de la plage de recherche, pour que le réseau ne voie pas de bord ; "
         "vide : 8 µm"),
    ],
    "Réglages": [
        ('thr1', "Seuil de l'étage 1", 'str', "Vide : seuil retenu en validation pour ce modèle"),
        ('rounds', "Tours de l'étage 2", ['1', '2', '3'],
         "1 : une instance par détection ; 2 : les centres vus par l'étage 2 et non segmentés (bactéries "
         "collées, fusionnées par l'étage 1) deviennent de nouvelles fenêtres"),
        ('prior', "A priori de l'image précédente", 'bool',
         "Lot : les centres des instances de l'image précédente sont ajoutés aux candidats de l'étage 2 "
         "(une bactérie bouge peu d'une image à l'autre ; un candidat sans bactérie est rejeté)"),
        ('peak_box', "Fenêtre des maxima, étage 1 (Z Y X)", 'str',
         "Une détection = un maximum local de la carte des centres dans cette fenêtre (voxels grossiers, valeurs "
         "impaires, p. ex. « 3 5 5 » = 3 µm × 1,75 µm × 1,75 µm) ; plus grande : moins de doublons mais deux "
         "bactéries proches fusionnent ; vide : celle du modèle"),
        ('film_holos', "Images pour les statistiques du film", 'str',
         "Modèles à normalisation « film » (MAD ou écart-type du film) : nombre d'images, régulièrement espacées, "
         "sur lesquelles le centre et l'échelle sont mesurés ; vide : celui du modèle (32)"),
        ('vmin', "Voxels minimum d'une instance", 'int',
         "Étage 2 : une instance de moins de voxels fins est une fausse détection (une bactérie de 1,8 × 0,6 µm "
         "fait ≈ 60 voxels)"),
        ('heat_thr', "Seuil des centres (étage 2)", 'float',
         "Étage 2 : seuil des maxima de la carte fine des centres ; ces centres, s'ils ne correspondent à aucune "
         "instance, deviennent les fenêtres du second tour"),
        ('nms2', "Fenêtre des maxima, étage 2 (Z Y X)", 'str',
         "Étage 2 : fenêtre des maxima locaux de la carte fine des centres (voxels fins de 0,25 × 0,175 × 0,175 µm, "
         "valeurs impaires)"),
        ('border2', "Bord ignoré des fenêtres (voxels)", 'int',
         "Étage 2 : les maxima à moins de ce nombre de voxels du bord d'une fenêtre sont ignorés (bactérie tronquée)"),
        ('same_xy', "Même centre à moins de (XY, µm)", 'float',
         "Second tour : deux centres plus proches que cela (en XY et en Z) sont la même bactérie"),
        ('same_z', "Même centre à moins de (Z, µm)", 'float', ""),
        ('dup_iou', "Doublons : IoU de fusion", 'float',
         "Deux instances voisines dont les masques se recouvrent au-delà de cette IoU sont fusionnées (on garde la "
         "plus grosse) ; plus bas : moins de doublons, mais risque de fusionner deux bactéries collées"),
        ('dup_xy', "Doublons : voisinage XY (µm)", 'float', "Distance entre centres en dessous de laquelle l'IoU est testée"),
        ('dup_z', "Doublons : voisinage Z (µm)", 'float', ""),
    ],
    "Sortie": [
        ('out_dir', "Dossier de sortie", 'dir',
         "positions_unet3d.csv (colonnes du RESULT_*.csv d'HOLOTRACKER + mesures), inference_params.json, "
         "instances/*.npz, *_localisations.bmp"),
        ('save_npz', "Enregistrer les instances (.npz)", 'bool',
         "Origine + masque de chaque instance, grille fine (0,175 × 0,175 × 0,25 µm)"),
        ('save_bmp', "Enregistrer les BMP avec les localisations", 'bool',
         "Hologramme normalisé, chaque bactérie entourée, couleur et valeur de z (µm)"),
    ],
}
VIEWS = ["Hologramme + détections", "Instances (projection XY)", "Instances (projection XZ)", "Instances (projection YZ)"]


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


class InferenceGUI:

    def __init__(self, root):
        self.root = root
        self.root.title("U-Net 3D — inférence (positions et instances)")
        self.root.geometry("1500x900")
        self.vars = {}
        self.pipeline, self.pipeline_key = None, None
        self.cache = {}
        self.task = None
        self.last = None
        self.runner = TaskRunner('thread')
        main = ttk.PanedWindow(root, orient='horizontal')
        main.pack(fill='both', expand=True, padx=5, pady=5)
        left, right = ttk.Frame(main), ttk.Frame(main)
        main.add(left, weight=0)
        main.add(right, weight=1)
        self.init_left(left)
        self.init_right(right)
        self.set_params(self.load_params())
        self.pump = MessagePump(root, self.runner, dict(log=self.log, progress=self.on_progress, frame=self.on_frame,
                                                        done=self.on_done, error=self.on_error, stopped=self.on_stopped))
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    # ------------------------------------------------------------------ paramètres
    def init_left(self, parent):
        bar = ttk.Frame(parent)
        bar.pack(fill='x', pady=(0, 3))
        self.btn_load = ttk.Button(bar, text="Charger les modèles", command=self.load_models)
        self.btn_run = ttk.Button(bar, text="Lancer le lot", command=lambda: self.start_batch(preview=False))
        self.btn_stop = ttk.Button(bar, text="Arrêter", command=self.stop, state='disabled')
        for b in (self.btn_load, self.btn_run, self.btn_stop):
            b.pack(side='left', padx=3)
        pbar = ttk.Frame(parent)
        pbar.pack(fill='x', pady=(0, 3))
        self.btn_preview = ttk.Button(pbar, text="Image n°", command=lambda: self.start_batch(preview=True))
        self.btn_preview.pack(side='left', padx=3)
        self.preview_index = tk.StringVar(value="0")
        spin = ttk.Spinbox(pbar, from_=0, to=10 ** 6, textvariable=self.preview_index, width=6)
        spin.pack(side='left')
        spin.bind("<Return>", lambda _e: self.start_batch(preview=True))
        ToolTip(spin, "Indice de l'image dans le film (0 = première). Entrée : calcule l'image")
        self.btn_prev = ttk.Button(pbar, text="◀", width=3, command=lambda: self.step_preview(-1))
        self.btn_next = ttk.Button(pbar, text="▶", width=3, command=lambda: self.step_preview(+1))
        self.btn_prev.pack(side='left', padx=(6, 0))
        self.btn_next.pack(side='left')
        self.busy_buttons = (self.btn_load, self.btn_run, self.btn_preview, self.btn_prev, self.btn_next)
        nb = ttk.Notebook(parent)
        nb.pack(fill='both', expand=True)
        for tab_name, rows in TABS.items():
            tab = ttk.Frame(nb, padding=8)
            nb.add(tab, text=tab_name)
            for r, spec in enumerate(rows):
                self.add_control(tab, r, *spec)
            tab.columnconfigure(1, weight=1)
        self.model_label = ttk.Label(parent, text="Modèles non chargés", wraplength=430, justify='left')
        self.model_label.pack(fill='x', pady=5)

    def add_control(self, parent, row, key, label, kind, help_text):
        if kind == 'bool':
            var = tk.BooleanVar()
            w = ttk.Checkbutton(parent, text=label, variable=var)
            w.grid(row=row, column=0, columnspan=3, sticky='w', pady=2)
        else:
            var = tk.StringVar()
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky='w', pady=2, padx=(0, 6))
            if isinstance(kind, list):
                w = ttk.Combobox(parent, textvariable=var, values=kind, state='readonly', width=24)
                w.grid(row=row, column=1, sticky='w')
            else:
                w = ttk.Entry(parent, textvariable=var, width=44 if kind in ('dir', 'file', 'str') else 12)
                w.grid(row=row, column=1, sticky='we' if kind in ('dir', 'file') else 'w')
            if kind in ('dir', 'file'):
                ttk.Button(parent, text="…", width=3, command=lambda: self.browse(key, kind)).grid(row=row, column=2, padx=2)
        ToolTip(w, help_text)
        self.vars[key] = (var, kind)

    def browse(self, key, kind):
        var = self.vars[key][0]
        init = var.get() if os.path.isdir(var.get()) else DL_DIR
        p = filedialog.askdirectory(initialdir=init) if kind == 'dir' else filedialog.askopenfilename(initialdir=init)
        if p:
            var.set(p)

    def get_params(self):
        p, errors = {}, []
        for key, (var, kind) in self.vars.items():
            v = var.get()
            try:
                if kind == 'int':
                    v = int(float(str(v).replace(',', '.')))
                elif kind == 'float':
                    v = float(str(v).replace(',', '.'))
            except ValueError:
                errors.append(f"{key} = {v!r}")
                continue
            p[key] = v
        if errors:
            raise ValueError("Valeurs numériques invalides : " + ", ".join(errors))
        p['mean_removal'] = {v: k for k, v in REMOVAL.items()}.get(p['mean_removal'], 'division')
        p['rounds'] = int(p['rounds'])
        thr = str(p.get('thr1', '')).strip().replace(',', '.')
        p['thr1'] = float(thr) if thr else None
        for k in ('z_min', 'z_max', 'z_step', 'high_pass', 'low_pass', 'z_margin'):
            v = str(p.get(k, '')).strip().replace(',', '.')
            try:
                p[k] = float(v) if v else None
            except ValueError:
                raise ValueError(f"Valeur numérique invalide : {k} = {v!r}")
        p['decim'] = None if p.get('decim') in ('', 'modèle', None) else int(p['decim'])
        for k in ('peak_box', 'nms2'):          # trois entiers « Z Y X » ; vide : défaut
            v = str(p.get(k, '')).replace(',', ' ').split()
            try:
                p[k] = [int(x) for x in v] if v else None
            except ValueError:
                raise ValueError(f"{k} : trois entiers attendus ({' '.join(v)!r})")
            if p[k] is not None and len(p[k]) != 3:
                raise ValueError(f"{k} : trois entiers attendus (Z Y X)")
        v = str(p.get('film_holos', '')).strip()
        p['film_holos'] = int(v) if v else None
        if p['magnification'] <= 0:
            raise ValueError("grossissement > 0")
        p['dx_um'] = p['pix_size_um'] / p['magnification']
        return p

    @staticmethod
    def optics_recon(p):
        optics = dict(dx_um=p['dx_um'], wavelength_um=p['wavelength_um'], index_medium=p['index_medium'])
        recon = dict(z_min_um=p['z_min'] or 0.0, z_max_um=p['z_max'], step_z_um=p['z_step'], decim=p['decim'],
                     high_pass=p['high_pass'], low_pass=p['low_pass'], z_margin_um=p['z_margin'])
        return optics, recon

    def set_params(self, p):
        if 'mean_removal' not in p and 'normalization' in p:                 # anciens réglages (« normalisation »)
            p = dict(p, mean_removal={'film_mean': 'division'}.get(p['normalization'], p['normalization']))
        p = dict(DEFAULTS, **p)
        for key, (var, kind) in self.vars.items():
            v = p.get(key, '')
            if key == 'mean_removal':
                v = REMOVAL.get(v, v)
            if key in ('thr1', 'z_min', 'z_max', 'z_step', 'high_pass', 'low_pass', 'z_margin', 'peak_box', 'nms2',
                       'film_holos') and v is None:
                v = ''
            if isinstance(v, (list, tuple)):
                v = ' '.join(str(x) for x in v)
            if key == 'decim' and v in (None, ''):
                v = 'modèle'
            var.set(v)

    def load_params(self):
        try:
            with open(PARAMS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (OSError, ValueError):
            return DEFAULTS

    def save_params(self):
        try:
            p = self.get_params()
            with open(PARAMS_FILE, 'w', encoding='utf-8') as f:
                json.dump(p, f, indent=2, ensure_ascii=False)
        except (OSError, ValueError):
            pass

    # ------------------------------------------------------------------ affichage
    def init_right(self, parent):
        ctrl = ttk.Frame(parent)
        ctrl.pack(side='top', fill='x', pady=(0, 3))
        ttk.Label(ctrl, text="Vue").pack(side='left')
        self.view = tk.StringVar(value=VIEWS[0])
        vb = ttk.Combobox(ctrl, textvariable=self.view, values=VIEWS, state='readonly', width=26)
        vb.pack(side='left', padx=(4, 12))
        vb.bind("<<ComboboxSelected>>", lambda _e: self.draw())
        vb.bind("<Up>", lambda _e: self.step_view(-1))
        vb.bind("<Down>", lambda _e: self.step_view(+1))
        ttk.Label(ctrl, text="Contraste ±").pack(side='left')
        self.contrast = tk.StringVar(value="0.1")
        ttk.Combobox(ctrl, textvariable=self.contrast, values=["0.02", "0.05", "0.1", "0.2", "0.5"], width=6).pack(side='left', padx=4)
        self.contrast.trace_add('write', lambda *_: self.draw())
        self.show_z = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, text="z (µm)", variable=self.show_z, command=self.draw).pack(side='left', padx=8)
        self.show_stage1 = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctrl, text="détections de l'étage 1", variable=self.show_stage1, command=self.draw).pack(side='left')
        self.frame_label = ttk.Label(ctrl, text="")
        self.frame_label.pack(side='right', padx=5)
        prog = ttk.Frame(parent)
        prog.pack(side='bottom', fill='x')
        self.progress = ttk.Progressbar(prog, mode='determinate')
        self.progress.pack(side='left', fill='x', expand=True, padx=(0, 6))
        self.progress_label = ttk.Label(prog, text="", width=60)
        self.progress_label.pack(side='left')
        self.text = tk.Text(parent, height=9, wrap='word')
        self.text.pack(side='bottom', fill='x')
        self.timing_label = ttk.Label(parent, text="")
        self.timing_label.pack(side='bottom', fill='x')
        self.fig = Figure(figsize=(5, 5))
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.axis('off')
        self.fig.subplots_adjust(left=0.05, right=0.99, top=0.95, bottom=0.05)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.toolbar = NavigationToolbar2Tk(self.canvas, parent)
        self.toolbar.update()
        # clic sur une bactérie : ses trois coupes de segmentation
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.selected = None
        self.seg_win = None
        self.canvas.get_tk_widget().configure(width=300, height=200)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        self.cmap = ListedColormap(np.random.default_rng(0).uniform(0.3, 1.0, (4096, 3)))
        self.last_view = None

    def log(self, msg):
        self.text.insert('end', msg + "\n")
        self.text.see('end')

    def step_view(self, step):
        i = VIEWS.index(self.view.get()) if self.view.get() in VIEWS else 0
        self.view.set(VIEWS[(i + step) % len(VIEWS)])
        self.draw()
        return "break"

    def draw(self):
        f = self.last
        if f is None:
            return
        try:
            c = float(self.contrast.get())
        except ValueError:
            return
        view = self.view.get()
        keep = self.last_view == view and self.ax.images
        lims = (self.ax.get_xlim(), self.ax.get_ylim()) if keep else None
        self.ax.clear()
        dx = f['dx']
        n = f['holo'].shape[0]
        field = n * dx
        halo = [pe.withStroke(linewidth=1.0, foreground='black')]
        has_inst = 'label_xy' in f
        if view == VIEWS[0] or not has_inst:
            self.ax.imshow(f['holo'].astype(np.float32), cmap='gray', vmin=-c, vmax=c, extent=[0, field, field, 0])
            P = f['centers'] if has_inst else f['pos1']
            if self.show_stage1.get() and has_inst and len(f['pos1']):
                self.ax.plot(f['pos1'][:, 0], f['pos1'][:, 1], 'o', mfc='none', mec='#2a78d6', ms=5, mew=0.8)
            if len(P):
                rnd = f.get('rounds', np.ones(len(P)))
                for sel, col in ((rnd <= 1, '#1baf7a'), (rnd > 1, '#e87ba4')):
                    if sel.any():
                        self.ax.plot(P[sel, 0], P[sel, 1], 's' if has_inst else 'o', mfc='none', mec=col, ms=4, mew=0.9)
                if self.show_z.get():
                    for x, y, z in P:
                        self.ax.text(x + 0.8, y - 0.8, f"{z:.1f}", color='#eda100', fontsize=6, clip_on=True,
                                     path_effects=halo)
            self.ax.set_xlabel('x (µm)')
            self.ax.set_ylabel('y (µm)')
        elif view == VIEWS[1]:
            self.ax.imshow(f['holo'].astype(np.float32), cmap='gray', vmin=-c, vmax=c, extent=[0, field, field, 0])
            self.ax.imshow(np.ma.masked_equal(f['label_xy'] % 4096, 0), cmap=self.cmap, vmin=0, vmax=4095,
                           extent=[0, field, field, 0], interpolation='nearest', alpha=0.85)
            if self.show_z.get():
                for x, y, z in f['centers']:
                    self.ax.text(x + 0.8, y - 0.8, f"{z:.1f}", color='#eda100', fontsize=6, clip_on=True, path_effects=halo)
            self.ax.set_xlabel('x (µm)')
            self.ax.set_ylabel('y (µm)')
        else:
            lab = f['label_xz'] if view == VIEWS[2] else f['label_yz']
            depth = lab.shape[0] * f['dz_fine']
            self.ax.set_facecolor('black')
            self.ax.imshow(np.ma.masked_equal(lab % 4096, 0), cmap=self.cmap, vmin=0, vmax=4095, extent=[0, field, depth, 0],
                           interpolation='nearest', aspect='equal')
            self.ax.set_xlabel('x (µm)' if view == VIEWS[2] else 'y (µm)')
            self.ax.set_ylabel('z (µm)')
        if has_inst and self.selected is not None and self.selected < len(f['centers']):
            x, y, z = f['centers'][self.selected]
            px, py = {VIEWS[2]: (x, z), VIEWS[3]: (y, z)}.get(view, (x, y))
            self.ax.plot([px], [py], 'o', mfc='none', mec='#ffffff', ms=16, mew=1.8)
        if lims:
            self.ax.set_xlim(lims[0])
            self.ax.set_ylim(lims[1])
        n_inst = len(f['centers']) if has_inst else len(f['pos1'])
        extra = f", dont {int((f['rounds'] > 1).sum())} au second tour (rose)" if has_inst else ''
        self.ax.set_title(f"{f['name']}\n{len(f['pos1'])} détections"
                          + (f", {n_inst} instances{extra}" if has_inst else '')
                          + (" — cliquer sur une bactérie : ses coupes" if 'win_amp' in f else ''), fontsize=9)
        self.last_view = view
        self.canvas.draw_idle()

    # ------------------------------------------------------------------ coupes d'une bactérie
    def on_click(self, ev):
        f = self.last
        if (ev.inaxes is not self.ax or ev.button != 1 or self.toolbar.mode or f is None or 'win_amp' not in f
                or not len(f['centers'])):
            return
        view = self.view.get()
        C = f['centers']
        P = {VIEWS[2]: C[:, [0, 2]], VIEWS[3]: C[:, [1, 2]]}.get(view, C[:, :2])
        d = np.hypot(P[:, 0] - ev.xdata, P[:, 1] - ev.ydata)
        i = int(np.argmin(d))
        if d[i] > 4.0:                                  # µm : clic trop loin de toute bactérie
            return
        self.selected = i
        self.draw()
        self.show_instance(i)

    def show_instance(self, i):
        """Fenêtre : coupes XY, XZ, YZ par le centre de l'instance i (amplitude refocalisée, masques)."""
        f = self.last
        amp = f['win_amp'][i].astype(np.float32)                     # (nz, ny, nx) grille fine
        wo = f['win_org'][i]
        shape = amp.shape
        dx, dz = f['dx'], f['dz_fine']
        me = _paste_window(f['inst_masks'][i], f['inst_origin'][i], wo, shape)
        others = [w for j, (m, o) in enumerate(zip(f['inst_masks'], f['inst_origin'])) if j != i
                  for w in [_paste_window(m, o, wo, shape)] if w.any()]
        kk, rr, cc = (int(round(v.mean())) for v in np.nonzero(me)) if me.any() else (s // 2 for s in shape)
        x0, y0, z0 = wo[2] * dx, wo[1] * dx, wo[0] * dz
        ext_xy = [x0, x0 + shape[2] * dx, y0 + shape[1] * dx, y0]
        ext_xz = [x0, x0 + shape[2] * dx, z0 + shape[0] * dz, z0]
        ext_yz = [y0, y0 + shape[1] * dx, z0 + shape[0] * dz, z0]
        if self.seg_win is None or not self.seg_win.winfo_exists():
            self.seg_win = tk.Toplevel(self.root)
            self.seg_win.geometry("1150x480")
            self.seg_fig = Figure(figsize=(11, 4.4))
            self.seg_canvas = FigureCanvasTkAgg(self.seg_fig, master=self.seg_win)
            NavigationToolbar2Tk(self.seg_canvas, self.seg_win).update()
            self.seg_canvas.get_tk_widget().pack(fill='both', expand=True)
        self.seg_fig.clear()
        axs = self.seg_fig.subplots(1, 3)
        views = ((amp[kk], me[kk], [o[kk] for o in others], ext_xy, 'x (µm)', 'y (µm)', f"XY (z = {z0 + kk * dz:.2f} µm)"),
                 (amp[:, rr, :], me[:, rr, :], [o[:, rr, :] for o in others], ext_xz, 'x (µm)', 'z (µm)', f"XZ (y = {y0 + rr * dx:.2f} µm)"),
                 (amp[:, :, cc], me[:, :, cc], [o[:, :, cc] for o in others], ext_yz, 'y (µm)', 'z (µm)', f"YZ (x = {x0 + cc * dx:.2f} µm)"))
        for a, (img, m, oth, ext, xl, yl, title) in zip(axs, views):
            a.imshow(img, cmap='gray', extent=ext, interpolation='nearest')
            gx = np.linspace(ext[0], ext[1], img.shape[1])
            gy = np.linspace(ext[3], ext[2], img.shape[0])
            for o in oth:
                if o.any():
                    a.contour(gx, gy, o.astype(float), [0.5], colors='#e87ba4', linewidths=0.8, linestyles='dotted')
            if m.any():
                a.contour(gx, gy, m.astype(float), [0.5], colors='#eda100', linewidths=1.6)
            a.set_title(title, fontsize=9)
            a.set_xlabel(xl, fontsize=8)
            a.set_ylabel(yl, fontsize=8)
            a.tick_params(labelsize=7)
        mm = f['measures'][i]
        x, y, z = f['centers'][i]
        txt = f"instance {i + 1} : x {x:.2f}, y {y:.2f}, z {z:.2f} µm"
        if mm:
            ax_ = np.asarray(mm['axis'])
            phi = np.degrees(np.arccos(min(abs(ax_[2]), 1.0)))
            txt += (f" — longueur {mm['length']:.2f} µm, épaisseur {mm['thickness']:.2f} µm, φ {phi:.0f}° (0° : debout), "
                    f"{mm['n']} voxels, score {f['scores'][i]:.2f}, tour {int(f['rounds'][i])}")
        self.seg_fig.suptitle(txt + "\norange : masque de la bactérie ; pointillés roses : autres instances", fontsize=9)
        self.seg_fig.tight_layout()
        self.seg_canvas.draw_idle()
        self.seg_win.title(f"Segmentation — {f['name']} — instance {i + 1}")
        self.seg_win.lift()

    # ------------------------------------------------------------------ calcul
    def set_busy(self, busy):
        for b in self.busy_buttons:
            b.configure(state='disabled' if busy else 'normal')
        self.btn_stop.configure(state='normal' if busy else 'disabled')

    def load_models(self, then=None):
        try:
            p = self.get_params()
        except ValueError as e:
            messagebox.showerror("Paramètres", str(e))
            return
        st2 = p['stage2'] if p['stage2'] and os.path.isfile(os.path.join(p['stage2'], 'model_best.pt')) else None
        if not os.path.isfile(os.path.join(p['stage1'], 'model_best.pt')):
            messagebox.showerror("Modèles", f"pas de model_best.pt dans {p['stage1']}")
            return
        self.save_params()
        key = (p['stage1'], st2)
        optics, recon = self.optics_recon(p)
        self.task, self.then = 'load', then
        self.set_busy(True)
        self.log("Chargement des modèles…")
        self.runner.start(_load_pipeline, p['stage1'], st2, optics, recon, key)

    def start_batch(self, preview):
        if self.runner.busy():
            return
        try:
            p = self.get_params()
        except ValueError as e:
            messagebox.showerror("Paramètres", str(e))
            return
        key = (p['stage1'], p['stage2'] if p['stage2'] and os.path.isfile(os.path.join(p['stage2'], 'model_best.pt')) else None)
        if self.pipeline is None or self.pipeline_key != key:
            self.load_models(then=lambda: self.start_batch(preview))
            return
        shown = self.pipeline.describe()
        try:
            self.pipeline.set_post(peak_box=p['peak_box'], film_holos=p['film_holos'], heat_thr=p['heat_thr'],
                                   nms=p['nms2'], border=p['border2'], same_xy_um=p['same_xy'], same_z_um=p['same_z'],
                                   vmin=p['vmin'], dup_iou=p['dup_iou'], dup_xy_um=p['dup_xy'], dup_z_um=p['dup_z'])
        except ValueError as e:
            messagebox.showerror("Réglages", str(e))
            return
        optics, recon = self.optics_recon(p)
        before = (dict(self.pipeline.optics), dict(self.pipeline.recon))
        try:
            self.pipeline.configure(optics, recon)
        except ValueError as e:
            self.pipeline.configure(*before)
            messagebox.showerror("Reconstruction", str(e))
            return
        if self.pipeline.describe() != shown or self.model_label.cget('text') != shown:
            self.show_pipeline()
        if preview:
            try:
                i = max(int(self.preview_index.get()), 0)
            except ValueError:
                i = 0
            self.preview_index.set(str(i))
            p.update(first=i, last=i, out_dir='', prior=False)
        self.save_params()
        self.task = 'preview' if preview else 'batch'
        self.set_busy(True)
        self.progress.configure(value=0)
        self.runner.start(_run_batch, self.pipeline, p, self.cache)

    def step_preview(self, step):
        try:
            i = int(self.preview_index.get()) + step
        except ValueError:
            i = 0
        self.preview_index.set(str(max(i, 0)))
        self.start_batch(preview=True)

    def stop(self):
        self.runner.stop()
        self.log("Arrêt demandé (fin de l'image en cours)…")

    # messages du calcul
    def on_progress(self, data):
        i, n, txt = data
        self.progress.configure(maximum=max(n, 1), value=i)
        self.progress_label.configure(text=f"{i}/{n}  {txt}")

    def on_frame(self, f):
        if self.last is None or self.last.get('index') != f.get('index'):
            self.selected = None
        self.last = f
        t = f['timings']
        self.timing_label.configure(text="temps : " + ", ".join(f"{k} {v * 1000:.0f} ms" for k, v in t.items()))
        self.frame_label.configure(text=f"image {f['index']} : {f['name']}")
        self.draw()

    def on_done(self, result):
        task = self.task
        self.task = None
        if self.runner.worker is not None:
            self.runner.worker.join(timeout=2)      # le fil a posté « done » : il se termine
        self.set_busy(False)
        if task == 'load':
            self.pipeline, self.pipeline_key = result
            self.show_pipeline()
            then, self.then = self.then, None
            if then:
                then()
        elif task == 'batch':
            self.log(f"Lot terminé : {json.dumps(result, ensure_ascii=False, default=str)}")

    def show_pipeline(self):
        """Modèles et reconstruction en cours ; écarts avec l'entraînement en orange."""
        w = self.pipeline.warnings()
        self.model_label.configure(text=self.pipeline.describe(), foreground='#c25100' if w else '')
        self.log(self.pipeline.describe())

    def on_error(self, tb):
        self.task = None
        self.set_busy(False)
        self.log(tb)
        messagebox.showerror("Erreur", tb.strip().splitlines()[-1])

    def on_stopped(self, _):
        self.task = None
        self.set_busy(False)
        self.log("Arrêté.")

    def on_closing(self):
        self.save_params()
        self.runner.stop()
        self.pump.close()
        self.root.destroy()


def _paste_window(mask, mask_org, win_org, shape):
    """Masque booléen (origine mask_org, grille fine) recadré dans une fenêtre (origine win_org, forme shape)."""
    out = np.zeros(shape, bool)
    lo = [max(a, b) for a, b in zip(mask_org, win_org)]
    hi = [min(a + s, b + t) for a, s, b, t in zip(mask_org, mask.shape, win_org, shape)]
    if all(h > l for l, h in zip(lo, hi)):
        out[tuple(slice(l - o, h - o) for l, h, o in zip(lo, hi, win_org))] =             mask[tuple(slice(l - o, h - o) for l, h, o in zip(lo, hi, mask_org))]
    return out


def _load_pipeline(rep, stage1, stage2, optics, recon, key):
    import inference_pipeline as ip
    return ip.Pipeline(stage1, stage2, optics, recon=recon), key


def _run_batch(rep, pipeline, params, cache):
    import inference_pipeline as ip
    return ip.run_batch(rep, pipeline, params, cache)


def main():
    root = tk.Tk()
    InferenceGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()

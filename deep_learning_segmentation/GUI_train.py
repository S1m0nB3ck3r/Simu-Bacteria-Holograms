# -*- coding: utf-8 -*-

"""
Filename: GUI_train.py

Description:
Interface d'entraînement du U-Net 3D :
  - Données : dossier d'un jeu produit par GUI_generate_dataset.py (parameters.yaml, train/, val/, test/ :
    hologrammes BMP, volumes d'instance, positions CSV), résumé de ses paramètres ; un dossier de .npz de
    generate_dataset.py (ancien format, benchmarks) est aussi accepté ; « Générer un jeu… » ouvre
    GUI_generate_dataset.py ;
  - Étage 1 (positions) : train.py (loss, normalisation, profondeur, pas Z, taille du réseau, tête
    d'orientation) ;
  - Étage 2 (instances) : train_stage2.py (fenêtres fines, décalages, négatifs, milieux de paires).
Chaque tâche lance le script en PROCESSUS INDÉPENDANT, sortie écrite dans <dossier>/train.log ; un fil de travail (gui_messages.TaskRunner) lit ce journal et envoie au GUI les messages
progression / loss / validation (history.json), relevés par paquets bornés (MessagePump) : l'interface
reste réactive, et le calcul ne dépend pas d'elle. « Arrêter » dépose un fichier STOP dans le dossier : le
script termine l'itération, fait une dernière validation, enregistre le bilan et s'arrête (pas de modèle à
moitié écrit) ; « Tuer » termine le processus. Fermer le GUI pendant une tâche : la laisser tourner ou
l'arrêter proprement ; « Suivre un run… » reprend l'affichage d'un run en cours ou fini.
Réglages mémorisés dans GUI_train_params.json (à côté du script).

Utilisation :
  python deep_learning_segmentation/GUI_train.py

Author: Simon BECKER
Date: 2026-10-18

License:
GNU General Public License v3.0
"""

import json
import math
import os
import re
import subprocess
import sys
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

DL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(DL_DIR)
sys.path.insert(0, DL_DIR)
from gui_messages import MessagePump, TaskRunner
import simu_dataset as sd

PARAMS_FILE = os.path.join(DL_DIR, "GUI_train_params.json")
PY = sys.executable


def script_constants(script):
    """Constantes de tête (NOM = valeur littérale) d'un script d'entraînement, lues sans l'importer (pas de torch)."""
    import ast
    out = {}
    tree = ast.parse(open(os.path.join(DL_DIR, script), encoding='utf-8').read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id.isupper():
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return out


C1, C2 = script_constants('train.py'), script_constants('train_stage2.py')
words = lambda t: ' '.join(str(v) for v in t)
DEFAULTS = {
    # données (dossier de GUI_generate_dataset.py)
    'dataset': os.path.join(DL_DIR, 'datasets', 'dataset_sim_200um_N20-400'), 'z_min': '', 'z_max': '',
    # étage 1 (défauts : constantes de train.py)
    's1_name': 'stage1', 's1_out': os.path.join(DL_DIR, 'runs'),
    's1_loss': 'centernet', 's1_norm': 'filmmad', 's1_mesh_z': C1['MAD_MESH_UM'][0], 's1_mesh_xy': C1['MAD_MESH_UM'][1],
    's1_film_holos': C1['FILM_STATS_HOLOS'], 's1_decim': str(C1['DECIM']), 's1_high_pass': C1['HIGH_PASS'],
    's1_low_pass': C1['LOW_PASS'], 's1_z_step': C1['Z_STEP_UM'], 's1_patch': words(C1['PATCH']),
    's1_pos_frac': C1['POS_FRACTION'], 's1_heat_xy': C1['HEAT_SIGMA_XY_UM'], 's1_heat_z': C1['HEAT_SIGMA_Z_UM'],
    's1_peak_box': words(C1['PEAK_BOX']), 's1_base': 16, 's1_levels': 3, 's1_iters': 6000, 's1_lr': 0.001,
    's1_orient': False,
    # étage 2 (défauts : constantes de train_stage2.py)
    's2_name': 'stage2', 's2_out': os.path.join(DL_DIR, 'runs'),
    's2_decim': str(C2['DECIM']), 's2_high_pass': C2['HIGH_PASS'], 's2_low_pass': C2['LOW_PASS'],
    's2_z_step': C2['Z_STEP_UM'], 's2_fine_nz': C2['FINE_NZ'], 's2_fine_nxy': C2['FINE_NXY'],
    's2_crop': C2['REFOCUS_CROP'], 's2_heat_xy': C2['HEAT_SIGMA_XY_UM'], 's2_heat_z': C2['HEAT_SIGMA_Z_UM'],
    's2_tol_xy': C2['TOL_XY_UM'], 's2_tol_z': C2['TOL_Z_UM'],
    's2_iters': 6000, 's2_lr': 0.001, 's2_base': 16, 's2_levels': 3, 's2_crops': C2['CROPS_PER_HOLO'], 's2_holos': 2,
    's2_jit_xy': C2['JITTER_XY_UM'], 's2_jit_z': C2['JITTER_Z_UM'], 's2_neg': C2['NEG_FRACTION'],
    's2_mid': C2['MID_FRACTION'], 's2_mid_pair': C2['MID_PAIR_UM'], 's2_vmin': C2['VMIN'],
}
RECON_HELP = {
    'decim': "Décimation latérale du volume grossier : voxel XY = pixel objet × décimation (×2 : 0,35 µm), par "
             "recadrage du spectre, sans perte si le passe-bas est ≤ taille / (2 × décimation) (256 px pour 1024 et ×2). "
             "×1 : 4 fois plus de calcul ; ×4 : 4 fois moins, passe-bas ≤ 128. Les deux étages doivent avoir la même",
    'high_pass': "Filtre passe-haut de la reconstruction : rayon (px de Fourier du capteur) sous lequel le spectre est "
                 "coupé (fond lentement variable, résidus d'éclairage) ; 0 : aucun",
    'low_pass': "Filtre passe-bas : rayon au-delà duquel le spectre est coupé (bruit haute fréquence) ; 0 : aucun",
}
TABS = {
    "Données": [
        ('dataset', "Jeu de données", 'dir',
         "Dossier produit par GUI_generate_dataset.py (parameters.yaml, train/, val/, test/), utilisé par les deux "
         "étages ; un dossier de .npz de generate_dataset.py (train/, val/) est aussi accepté"),
        ('z_min', "Profondeur min reconstruite (µm)", 'str',
         "Début du volume reconstruit (distance au capteur), deux étages ; vide : distance volume-caméra du jeu "
         "(0 pour un jeu .npz)"),
        ('z_max', "Profondeur max reconstruite (µm)", 'str',
         "Fin du volume reconstruit ; vide : distance volume-caméra + nombre de plans × pas du jeu (207 pour un jeu .npz)"),
    ],
    "Étage 1": [
        ('s1_name', "Nom du run", 'str', "Sous-dossier créé dans le dossier des runs (model_best.pt, config.json, "
                                        "history.json, train.log)"),
        ('s1_out', "Dossier des runs", 'dir', ""),
        ('s1_loss', "Loss", ['centernet', 'wmse', 'bce_dice', 'focal_tversky'],
         "centernet : carte des centres, positions = maxima (la meilleure au benchmark) ; les autres apprennent un "
         "masque grossier de la bactérie"),
        ('s1_norm', "Normalisation", ['filmmad', 'globalmad', 'filmstd', 'globalstd', 'localmad', 'fixed'],
         "Mise à l'échelle du volume reconstruit (Re, Im) avant le réseau : (V − centre) / σ. filmmad : médiane et "
         "1,4826 × MAD du film (médianes sur ses hologrammes), les mêmes pour toutes ses images ; globalmad : celles "
         "du volume de chaque hologramme ; filmstd, globalstd : idem avec moyenne et écart-type ; localmad : par "
         "maille (taille ci-dessous), interpolées ; fixed : échelle calculée du bruit capteur (modèles du 2026-10-18)"),
        ('s1_film_holos', "Hologrammes pour les statistiques du film", 'int',
         "filmmad, filmstd : nombre d'hologrammes, régulièrement espacés, sur lesquels le centre et σ sont mesurés"),
        ('s1_mesh_z', "Maille MAD locale, Z (µm)", 'float', "localmad seulement ; HOLOTRACKER : 12,5 µm"),
        ('s1_mesh_xy', "Maille MAD locale, XY (µm)", 'float', "localmad seulement ; arrondie à 4 voxels ; HOLOTRACKER : 11,2 µm"),
        ('s1_decim', "Décimation latérale", ['1', '2', '4'], RECON_HELP['decim']),
        ('s1_high_pass', "Passe-haut (px de Fourier)", 'float', RECON_HELP['high_pass']),
        ('s1_low_pass', "Passe-bas (px de Fourier)", 'float', RECON_HELP['low_pass']),
        ('s1_z_step', "Pas Z (µm)", 'float', "Pas des plans du volume grossier ; 1 µm ; 2 µm : réseau 2,3× plus rapide, "
                                            "σz 0,28 au lieu de 0,16 µm"),
        ('s1_patch', "Patchs (Z Y X, voxels)", 'str',
         "Taille des blocs du volume grossier vus à chaque pas d'entraînement (multiples de 2^niveaux) ; 48 96 96 = "
         "48 × 33,6 × 33,6 µm à ×2 ; 64 128 128 déborde des 6 Go"),
        ('s1_pos_frac', "Patchs centrés sur une bactérie (part)", 'float',
         "Part des patchs tirés autour d'une bactérie ; les autres sont tirés au hasard (le volume est presque vide)"),
        ('s1_heat_xy', "Largeur des cibles XY (µm)", 'float',
         "Cible : chaque bactérie est une gaussienne de cet écart-type dans la carte des centres ; plus large : "
         "apprentissage plus facile, mais pics de deux voisines confondus"),
        ('s1_heat_z', "Largeur des cibles Z (µm)", 'float', "Idem en Z (la réponse est ≈ 2,5 fois plus longue en Z)"),
        ('s1_peak_box', "Fenêtre des maxima (Z Y X, voxels)", 'str',
         "Une détection = un maximum local de la carte des centres dans cette fenêtre (valeurs impaires, voxels "
         "grossiers) ; sert à la validation et devient le défaut de l'inférence"),
        ('s1_base', "Canaux de base", 'int', "Canaux du premier niveau, doublés à chaque niveau ; 16 (1,4 M paramètres) ; "
                                             "8 : 4× plus petit"),
        ('s1_levels', "Niveaux du U-Net", 'int', "Sous-échantillonnages par 2 ; 3 : champ récepteur de plusieurs "
                                                 "dizaines de voxels"),
        ('s1_iters', "Itérations", 'int', "Pas d'optimisation ; ~0,3 à 0,45 s chacun (Quadro RTX 3000)"),
        ('s1_lr', "Taux d'apprentissage", 'float', "Maximum (AdamW) : montée linéaire sur 200 itérations puis cosinus"),
        ('s1_orient', "Tête d'orientation", 'bool', "Prédit aussi l'axe de la bactérie (tenseur a·aᵀ), sans coût mesurable"),
    ],
    "Étage 2": [
        ('s2_name', "Nom du run", 'str', "Sous-dossier créé dans le dossier des runs"),
        ('s2_out', "Dossier des runs", 'dir', ""),
        ('s2_decim', "Décimation latérale", ['1', '2', '4'], RECON_HELP['decim']),
        ('s2_high_pass', "Passe-haut (px de Fourier)", 'float', RECON_HELP['high_pass']),
        ('s2_low_pass', "Passe-bas (px de Fourier)", 'float', RECON_HELP['low_pass']),
        ('s2_z_step', "Pas Z du volume grossier (µm)", 'float',
         "Plans du volume grossier d'où partent les fenêtres (le plan le plus proche est refocalisé)"),
        ('s2_fine_nz', "Fenêtre fine : plans", 'int',
         "Hauteur de la fenêtre autour de chaque détection, en plans fins (pas des masques du jeu, 0,25 µm : 32 = "
         "8 µm) ; multiple de 2^niveaux"),
        ('s2_fine_nxy', "Fenêtre fine : côté (voxels)", 'int',
         "Côté de la fenêtre en pixels objet (0,175 µm : 40 = 7 µm) ; multiple de 2^niveaux"),
        ('s2_crop', "Carré refocalisé (px grossiers)", 'int',
         "Région du plan grossier propagée vers la fenêtre fine (64 px à ×2 = 22,4 µm) : assez grande pour contenir "
         "la lumière qui s'étale sur la hauteur de la fenêtre"),
        ('s2_heat_xy', "Largeur des cibles des centres XY (µm)", 'float',
         "Cible : σ des gaussiennes de la carte fine des centres (second tour)"),
        ('s2_heat_z', "Largeur des cibles des centres Z (µm)", 'float', ""),
        ('s2_tol_xy', "Attribution d'une fenêtre, XY (µm)", 'float',
         "Une fenêtre appartient à la bactérie la plus proche de son centre si elle est à moins de cette distance "
         "(XY et Z normalisés) ; sinon c'est une fausse détection (masque vide)"),
        ('s2_tol_z', "Attribution d'une fenêtre, Z (µm)", 'float', ""),
        ('s2_iters', "Itérations", 'int', "~0,55 s par itération"),
        ('s2_lr', "Taux d'apprentissage", 'float', "Maximum (AdamW)"),
        ('s2_base', "Canaux de base", 'int', ""),
        ('s2_levels', "Niveaux du U-Net", 'int', ""),
        ('s2_crops', "Fenêtres par hologramme", 'int', "Fenêtres tirées dans chaque hologramme reconstruit"),
        ('s2_holos', "Hologrammes par pas", 'int', "Hologrammes reconstruits par pas (fenêtres par pas = produit des deux)"),
        ('s2_jit_xy', "Décalage des fenêtres XY (µm)", 'float', "σ du décalage aléatoire : erreur de position simulée de l'étage 1"),
        ('s2_jit_z', "Décalage des fenêtres Z (µm)", 'float', ""),
        ('s2_neg', "Fenêtres négatives (part)", 'float', "Au hasard dans le volume : apprend à rejeter les fausses détections"),
        ('s2_mid', "Fenêtres entre deux voisines (part)", 'float',
         "Au milieu de deux bactéries proches : ce que voit l'étage 2 quand l'étage 1 les a fusionnées"),
        ('s2_mid_pair', "Voisines « proches » à moins de (µm)", 'float', "Distance entre centres définissant ces paires"),
        ('s2_vmin', "Voxels minimum d'une instance", 'int',
         "En dessous : fausse détection (≈ 60 voxels pour une bactérie de 1,8 × 0,6 µm) ; repris à l'inférence"),
    ],
}
NOISE = ('experimental', 'Warning', 'warn(', 'cupy._util')
RE_IT = re.compile(r'^it (\d+)/(\d+)\s+(?:\[[^\]]*\]\s+)?loss ([\d.eE+-]+).*?([\d.]+) s/it')
RE_GEN = re.compile(r'^\[(\w+)\] (\d+)/(\d+)')   # progression de simu_dataset.py / generate_dataset.py


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


# =============================================================================
# FONCTION DE TRAVAIL : lance les commandes et suit leur journal
# =============================================================================

def follow(rep, cmds, workdir, log_name, box):
    """
    Exécute cmds l'une après l'autre (processus indépendants, sortie dans workdir/log_name), en lisant le
    journal au fil de l'eau. box : dict partagé avec le GUI ('proc' : processus courant, 'kill' : demande
    d'arrêt immédiat). Messages : log, progress, metrics ('loss', it, loss, s/it) et ('history', liste).
    cmds = [] : suit seulement un journal existant (run lancé par une session précédente).
    """
    os.makedirs(workdir, exist_ok=True)
    log_path = os.path.join(workdir, log_name)
    hist_path = os.path.join(workdir, 'history.json')
    env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
    # processus indépendant, sans console, en priorité inférieure : le GUI garde toujours la main sur les processeurs
    flags = (getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0) | getattr(subprocess, 'CREATE_NO_WINDOW', 0)
             | getattr(subprocess, 'BELOW_NORMAL_PRIORITY_CLASS', 0))
    rc = None
    for ci, cmd in enumerate(cmds or [None]):
        if cmd is not None:
            rep.log("$ " + " ".join(f'"{c}"' if ' ' in c else c for c in cmd))
            fh = open(log_path, 'a' if ci else 'w', encoding='utf-8')
            box['proc'] = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=REPO_DIR, env=env,
                                           creationflags=flags)
            fh.close()
        pos = 0 if (cmd is None or ci == 0) else box.get('pos', 0)
        while True:
            p = box.get('proc')
            alive = p is not None and p.poll() is None
            if box.get('kill') and alive:
                p.terminate()
                rep.log("processus terminé (Tuer)")
            try:
                with open(log_path, 'r', encoding='utf-8', errors='replace') as fh:
                    fh.seek(pos)
                    chunk = fh.read()
                    pos = fh.tell()
            except OSError:
                chunk = ''
            for line in chunk.splitlines():
                if not line.strip() or any(k in line for k in NOISE):
                    continue
                m = RE_IT.match(line.strip())
                if m:
                    it, n, loss, spi = int(m[1]), int(m[2]), float(m[3]), float(m[4])
                    rep.progress(it, n, f"{spi:.2f} s/it, reste ~{(n - it) * spi / 60:.0f} min")
                    rep.metrics(('loss', it, loss, spi))
                    continue
                g = RE_GEN.match(line.strip())
                if g:
                    rep.progress(int(g[2]), int(g[3]), f"génération {g[1]}")
                rep.log(line)
                if 'VALIDATION' in line and os.path.exists(hist_path):
                    try:
                        rep.metrics(('history', json.load(open(hist_path, encoding='utf-8'))))
                    except ValueError:
                        pass
            if cmd is None and not chunk:
                if os.path.exists(os.path.join(workdir, 'train_summary.json')) or rep.stopped():
                    break                                   # suivi d'un run fini, ou abandon du suivi
            elif cmd is not None and not alive and not chunk:
                break
            time.sleep(0.5)
        box['pos'] = pos
        if cmd is not None:
            rc = box['proc'].returncode
            if rc != 0 or box.get('kill') or os.path.exists(os.path.join(workdir, 'STOPPED_ALL')):
                break
    if os.path.exists(hist_path):
        try:
            rep.metrics(('history', json.load(open(hist_path, encoding='utf-8'))))
        except ValueError:
            pass
    return dict(returncode=rc, workdir=workdir)


# =============================================================================
# GUI
# =============================================================================

class TrainGUI:

    def __init__(self, root):
        self.root = root
        self.root.title("U-Net 3D — données et entraînement")
        self.root.geometry("1500x900")
        self.vars = {}
        self.runner = TaskRunner('thread')
        self.box = {}
        self.workdir = None
        self.following = False
        self.losses, self.history = [], []
        main = ttk.PanedWindow(root, orient='horizontal')
        main.pack(fill='both', expand=True, padx=5, pady=5)
        left, right = ttk.Frame(main), ttk.Frame(main)
        main.add(left, weight=0)
        main.add(right, weight=1)
        self.init_left(left)
        self.init_right(right)
        self.set_params(self.load_params())
        for k in ('dataset', 'z_min', 'z_max'):
            self.vars[k][0].trace_add('write', self.update_data_info)
        self.update_data_info()
        self.pump = MessagePump(root, self.runner, dict(log=self.log, progress=self.on_progress, metrics=self.on_metrics,
                                                        done=self.on_done, error=self.on_error, stopped=self.on_done))
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    # ------------------------------------------------------------------ paramètres
    def init_left(self, parent):
        bar = ttk.Frame(parent)
        bar.pack(fill='x', pady=(0, 3))
        self.btn_gen = ttk.Button(bar, text="Générer un jeu…", command=self.open_generator)
        self.btn_s1 = ttk.Button(bar, text="Entraîner l'étage 1", command=lambda: self.start_train(1))
        self.btn_s2 = ttk.Button(bar, text="Entraîner l'étage 2", command=lambda: self.start_train(2))
        for b in (self.btn_gen, self.btn_s1, self.btn_s2):
            b.pack(side='left', padx=3)
        bar2 = ttk.Frame(parent)
        bar2.pack(fill='x', pady=(0, 5))
        self.btn_stop = ttk.Button(bar2, text="Arrêter", command=self.stop, state='disabled')
        self.btn_kill = ttk.Button(bar2, text="Tuer", command=self.kill, state='disabled')
        self.btn_follow = ttk.Button(bar2, text="Suivre un run…", command=self.follow_run)
        for b in (self.btn_stop, self.btn_kill, self.btn_follow):
            b.pack(side='left', padx=3)
        ToolTip(self.btn_stop, "Arrêt propre : fin de l'itération, dernière validation, bilan enregistré")
        ToolTip(self.btn_kill, "Arrêt immédiat du processus (le dernier modèle enregistré reste valable)")
        ToolTip(self.btn_gen, "Ouvre GUI_generate_dataset.py (fenêtre indépendante)")
        self.busy_buttons = (self.btn_s1, self.btn_s2, self.btn_follow)
        nb = ttk.Notebook(parent)
        nb.pack(fill='both', expand=True)
        for tab_name, rows in TABS.items():
            holder = ttk.Frame(nb)
            nb.add(holder, text=tab_name)
            canvas = tk.Canvas(holder, highlightthickness=0, width=560)
            sb = ttk.Scrollbar(holder, orient='vertical', command=canvas.yview)
            tab = ttk.Frame(canvas, padding=8)
            tab.bind("<Configure>", lambda e, c=canvas: c.configure(scrollregion=c.bbox("all")))
            canvas.create_window((0, 0), window=tab, anchor='nw')
            canvas.configure(yscrollcommand=sb.set)
            canvas.pack(side='left', fill='both', expand=True)
            sb.pack(side='right', fill='y')
            canvas.bind("<Enter>", lambda e, c=canvas: c.bind_all(
                "<MouseWheel>", lambda ev: c.yview_scroll(int(-ev.delta / 120), "units")))
            canvas.bind("<Leave>", lambda e, c=canvas: c.unbind_all("<MouseWheel>"))
            for r, spec in enumerate(rows):
                self.add_control(tab, r, *spec)
            tab.columnconfigure(1, weight=1)
            if tab_name == "Données":
                self.data_info = ttk.Label(tab, text="", wraplength=440, justify='left')
                self.data_info.grid(row=len(rows), column=0, columnspan=3, sticky='w', pady=(8, 0))

    def add_control(self, parent, row, key, label, kind, help_text):
        if kind == 'bool':
            var = tk.BooleanVar()
            w = ttk.Checkbutton(parent, text=label, variable=var)
            w.grid(row=row, column=0, columnspan=3, sticky='w', pady=2)
        else:
            var = tk.StringVar()
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky='w', pady=2, padx=(0, 6))
            if isinstance(kind, list):
                w = ttk.Combobox(parent, textvariable=var, values=kind, state='readonly', width=16)
                w.grid(row=row, column=1, sticky='w')
            else:
                w = ttk.Entry(parent, textvariable=var, width=40 if kind in ('dir', 'str') else 12)
                w.grid(row=row, column=1, sticky='we' if kind == 'dir' else 'w')
            if kind == 'dir':
                ttk.Button(parent, text="…", width=3, command=lambda: self.browse(key)).grid(row=row, column=2, padx=2)
        ToolTip(w, help_text)
        self.vars[key] = (var, kind)

    def browse(self, key):
        var = self.vars[key][0]
        p = filedialog.askdirectory(initialdir=var.get() if os.path.isdir(var.get()) else DL_DIR)
        if p:
            var.set(p)

    def dataset_info(self, path):
        """(texte, profondeur maximale des bactéries en µm ou None) d'un dossier de données."""
        if not path or not os.path.isdir(path):
            return "Dossier introuvable", None
        if os.path.isfile(os.path.join(path, 'parameters.yaml')):
            try:
                p = sd.read_parameters(path)
                return "Jeu GUI_generate_dataset.py : " + sd.describe(path), p['derive']['z_bacteries_m'][1] * 1e6
            except (OSError, ValueError, KeyError, TypeError) as e:
                return f"parameters.yaml illisible : {e}", None
        import glob
        n = {s: len(glob.glob(os.path.join(path, s, 'sample_*.npz'))) for s in ('train', 'val', 'test')}
        if n['train']:
            return "Jeu .npz de generate_dataset.py : " + ", ".join(f"{k} {v}" for k, v in n.items()), None
        return "Ni parameters.yaml ni train/*.npz dans ce dossier", None

    def z_range(self, p=None):
        """z min, z max de la reconstruction (µm) : saisis, sinon ceux du volume du jeu."""
        raw = {k: str(self.vars[k][0].get()).strip().replace(',', '.') for k in ('z_min', 'z_max')}
        given = {k: (float(v) if v else None) for k, v in raw.items()}
        data = self.vars['dataset'][0].get()
        try:
            return sd.resolve_range(data, given['z_min'], given['z_max']), given
        except (OSError, ValueError):
            return (given['z_min'] or 0.0, given['z_max'] or 207.0), given

    def update_data_info(self, *_):
        txt, _z = self.dataset_info(self.vars['dataset'][0].get())
        try:
            (z0, z1), given = self.z_range()
            try:
                src = "volume du jeu" if sd.volume_range_um(self.vars['dataset'][0].get()) else "défaut d'un jeu .npz"
            except (OSError, ValueError):
                src = "défaut"
            auto = [k for k in ('z_min', 'z_max') if given[k] is None]
            txt += f"\nVolume reconstruit : {z0:.1f}–{z1:.1f} µm" + (
                f" ({' et '.join('min' if k == 'z_min' else 'max' for k in auto)} automatique : {src})" if auto else "")
        except ValueError:
            txt += "\nProfondeur min / max : valeur illisible"
        self.data_info.configure(text=txt)

    def open_generator(self):
        subprocess.Popen([PY, os.path.join(DL_DIR, 'GUI_generate_dataset.py')], cwd=REPO_DIR,
                         env=dict(os.environ, PYTHONIOENCODING='utf-8'))

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
        return p

    def set_params(self, p):
        p = dict(DEFAULTS, **p)
        for key, (var, kind) in self.vars.items():
            var.set(p.get(key, ''))

    def load_params(self):
        try:
            with open(PARAMS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (OSError, ValueError):
            return DEFAULTS

    def save_params(self):
        try:
            with open(PARAMS_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.get_params(), f, indent=2, ensure_ascii=False)
        except (OSError, ValueError):
            pass

    # ------------------------------------------------------------------ affichage
    def init_right(self, parent):
        self.status = ttk.Label(parent, text="", font=('TkDefaultFont', 10, 'bold'))
        self.status.pack(side='top', fill='x', pady=(0, 3))
        prog = ttk.Frame(parent)
        prog.pack(side='bottom', fill='x')
        self.progress = ttk.Progressbar(prog, mode='determinate')
        self.progress.pack(side='left', fill='x', expand=True, padx=(0, 6))
        self.progress_label = ttk.Label(prog, text="", width=50)
        self.progress_label.pack(side='left')
        self.text = tk.Text(parent, height=12, wrap='word')
        self.text.pack(side='bottom', fill='x')
        self.fig = Figure(figsize=(6, 4))
        self.ax_loss = self.fig.add_subplot(1, 2, 1)
        self.ax_val = self.fig.add_subplot(1, 2, 2)
        self.fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.12, wspace=0.25)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().configure(width=300, height=200)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        self.draw()

    def log(self, msg):
        self.text.insert('end', msg + "\n")
        self.text.see('end')
        if int(self.text.index('end-1c').split('.')[0]) > 5000:
            self.text.delete('1.0', '1000.0')

    def draw(self):
        a, b = self.ax_loss, self.ax_val
        a.clear()
        b.clear()
        if self.losses:
            it = [x[0] for x in self.losses]
            a.plot(it, [x[1] for x in self.losses], color='#2a78d6', lw=1.5)
            a.set_yscale('log')
        a.set_title("loss d'entraînement", fontsize=10)
        a.set_xlabel('itération')
        h = self.history
        if h:
            it = [x['it'] for x in h]
            keys = [('f1', 'F1'), ('recall', 'rappel'), ('recall_lying', 'rappel des couchées')] if 'f1' in h[0] else \
                [('iou_mean', 'IoU moyenne'), ('iou_gt50', 'instances IoU > 0,5'), ('center_recall', 'centres : rappel'),
                 ('center_precision', 'centres : précision')]
            cols = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
            for (k, lab), c in zip(keys, cols):
                if k in h[0]:
                    b.plot(it, [x[k] for x in h], '-o', ms=4, lw=1.5, color=c, label=lab)
            b.legend(frameon=False, fontsize=8, loc='lower right')
            b.set_ylim(0, 1.02)
        b.set_title('validation', fontsize=10)
        b.set_xlabel('itération')
        for ax in (a, b):
            ax.grid(alpha=0.25)
            ax.spines[['top', 'right']].set_visible(False)
        self.canvas.draw_idle()

    # ------------------------------------------------------------------ tâches
    def set_busy(self, busy):
        for b in self.busy_buttons:
            b.configure(state='disabled' if busy else 'normal')
        for b in (self.btn_stop, self.btn_kill):
            b.configure(state='normal' if busy else 'disabled')

    def _launch(self, cmds, workdir, log_name, title):
        self.following = not cmds
        self.save_params()
        self.losses, self.history = [], []
        self.draw()
        self.workdir = workdir
        self.box = {}
        self.status.configure(text=f"{title} — {workdir}")
        self.progress.configure(value=0)
        self.set_busy(True)
        self.runner.start(follow, cmds, workdir, log_name, self.box)

    def start_train(self, stage):
        try:
            p = self.get_params()
        except ValueError as e:
            messagebox.showerror("Paramètres", str(e))
            return
        k = f's{stage}_'
        workdir = os.path.join(p[k + 'out'], p[k + 'name'])
        data = p['dataset']
        for split in ('train', 'val'):
            d = os.path.join(data, split)
            if not (os.path.isfile(os.path.join(d, 'index.csv')) or (os.path.isdir(d) and any(
                    f.startswith('sample_') for f in os.listdir(d)))):
                messagebox.showerror("Données", f"{d} : ni index.csv (GUI_generate_dataset.py) ni sample_*.npz")
                return
        _txt, z = self.dataset_info(data)
        try:
            (z0, z1), given = self.z_range()
        except ValueError:
            messagebox.showerror("Profondeur", "profondeur min / max : nombre attendu (ou vide)")
            return
        if z1 <= z0:
            messagebox.showerror("Profondeur", f"profondeur max ({z1}) ≤ min ({z0})")
            return
        if z is not None and z > z1 and not messagebox.askyesno(
                "Profondeur", f"Bactéries jusqu'à {z:.0f} µm, volume reconstruit jusqu'à {z1:.0f} µm seulement. "
                              f"Continuer quand même ?"):
            return
        zargs = [a for k, flag in (('z_min', '--z_min_um'), ('z_max', '--z_max_um')) if given[k] is not None
                 for a in (flag, str(given[k]))]
        if os.path.exists(os.path.join(workdir, 'model_best.pt')) and not messagebox.askyesno(
                "Run existant", f"{workdir} contient déjà un modèle. L'écraser ?"):
            return
        if stage == 1:
            try:
                patch = [str(int(v)) for v in p['s1_patch'].replace(',', ' ').split()]
                box = [str(int(v)) for v in p['s1_peak_box'].replace(',', ' ').split()]
                if len(patch) != 3 or len(box) != 3:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Paramètres", "patchs et fenêtre des maxima : trois entiers « Z Y X »")
                return
            cmd = [PY, '-u', os.path.join(DL_DIR, 'train.py'), '--name', p['s1_name'], '--out', p['s1_out'],
                   '--train_dir', os.path.join(data, 'train'), '--val_dir', os.path.join(data, 'val'),
                   '--loss', p['s1_loss'], '--norm', p['s1_norm'], '--mad_mesh_um', str(p['s1_mesh_z']), str(p['s1_mesh_xy']),
                   '--film_holos', str(p['s1_film_holos']), '--decim', str(p['s1_decim']),
                   '--high_pass', str(p['s1_high_pass']), '--low_pass', str(p['s1_low_pass']), '--patch', *patch,
                   '--pos_fraction', str(p['s1_pos_frac']), '--heat_sigma_xy_um', str(p['s1_heat_xy']),
                   '--heat_sigma_z_um', str(p['s1_heat_z']), '--peak_box', *box,
                   *zargs,
                   '--z_step', str(p['s1_z_step']), '--base', str(p['s1_base']), '--depth', str(p['s1_levels']),
                   '--iters', str(p['s1_iters']), '--lr', str(p['s1_lr'])]
            if p['s1_orient']:
                cmd += ['--orient_weight', '1']
        else:
            cmd = [PY, '-u', os.path.join(DL_DIR, 'train_stage2.py'), '--name', p['s2_name'], '--out', p['s2_out'],
                   '--data', data, *zargs, '--iters', str(p['s2_iters']), '--lr', str(p['s2_lr']), '--base', str(p['s2_base']),
                   '--depth', str(p['s2_levels']), '--crops_per_holo', str(p['s2_crops']),
                   '--holos_per_step', str(p['s2_holos']), '--jitter_xy', str(p['s2_jit_xy']),
                   '--jitter_z', str(p['s2_jit_z']), '--neg_fraction', str(p['s2_neg']),
                   '--mid_fraction', str(p['s2_mid']), '--vmin', str(p['s2_vmin']),
                   '--mid_pair_um', str(p['s2_mid_pair']), '--decim', str(p['s2_decim']),
                   '--high_pass', str(p['s2_high_pass']), '--low_pass', str(p['s2_low_pass']),
                   '--z_step', str(p['s2_z_step']), '--fine_nz', str(p['s2_fine_nz']), '--fine_nxy', str(p['s2_fine_nxy']),
                   '--refocus_crop', str(p['s2_crop']), '--heat_sigma_xy_um', str(p['s2_heat_xy']),
                   '--heat_sigma_z_um', str(p['s2_heat_z']), '--tol_xy_um', str(p['s2_tol_xy']),
                   '--tol_z_um', str(p['s2_tol_z'])]
        self._launch([cmd], workdir, 'train.log', f"Entraînement de l'étage {stage}")

    def follow_run(self):
        d = filedialog.askdirectory(initialdir=DL_DIR, title="Dossier d'un run (train.log)")
        if not d:
            return
        if not os.path.exists(os.path.join(d, 'train.log')):
            messagebox.showerror("Suivi", f"pas de train.log dans {d}")
            return
        self._launch([], d, 'train.log', "Suivi")

    def stop(self):
        if self.following:                      # suivi d'un run : on arrête de suivre, pas le run
            self.runner.stop()
            self.log("Fin du suivi (le run continue).")
            return
        if self.workdir:
            open(os.path.join(self.workdir, 'STOP'), 'w').close()
            open(os.path.join(self.workdir, 'STOPPED_ALL'), 'w').close()
        self.runner.stop()
        self.log("Arrêt demandé : fin de l'itération, dernière validation, bilan…")

    def kill(self):
        if messagebox.askyesno("Tuer", "Arrêter immédiatement le processus ?"):
            self.box['kill'] = True
            self.runner.stop()

    # messages
    def on_progress(self, data):
        i, n, txt = data
        self.progress.configure(maximum=max(n, 1), value=i)
        self.progress_label.configure(text=f"{i}/{n}  {txt}")

    def on_metrics(self, m):
        if m[0] == 'loss':
            self.losses.append((m[1], m[2]))
        else:
            self.history = m[1]
            best = max(self.history, key=lambda x: x.get('f1', x.get('score', x.get('iou_mean', 0))))
            if 'f1' in best:
                txt = f"meilleur : it {best['it']}, F1 {best['f1']:.3f}"
            else:
                txt = f"meilleur : it {best['it']}, IoU {best['iou_mean']:.3f}, centres {best.get('center_recall', float('nan')):.3f}"
            self.status.configure(text=f"{self.status.cget('text').split('  |')[0]}  |  {txt}")
        self.draw()

    def on_done(self, result):
        if self.runner.worker is not None:
            self.runner.worker.join(timeout=2)
        self.set_busy(False)
        if self.workdir and os.path.exists(os.path.join(self.workdir, 'STOPPED_ALL')):
            os.remove(os.path.join(self.workdir, 'STOPPED_ALL'))
        if result:
            rc = result.get('returncode')
            self.log(f"Terminé (code {rc}) : {result.get('workdir')}" if rc is not None else "Suivi terminé")

    def on_error(self, tb):
        self.set_busy(False)
        self.log(tb)
        messagebox.showerror("Erreur", tb.strip().splitlines()[-1])

    def on_closing(self):
        self.save_params()
        if self.runner.busy() and self.box.get('proc') is not None and self.box['proc'].poll() is None:
            ans = messagebox.askyesnocancel(
                "Tâche en cours", "Une tâche tourne.\nOui : la laisser tourner (suivi possible plus tard avec « Suivre "
                                  "un run… »)\nNon : l'arrêter proprement\nAnnuler : rester")
            if ans is None:
                return
            if ans is False:
                self.stop()
        self.runner.stop()
        self.pump.close()
        self.root.destroy()


def main():
    root = tk.Tk()
    TrainGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()

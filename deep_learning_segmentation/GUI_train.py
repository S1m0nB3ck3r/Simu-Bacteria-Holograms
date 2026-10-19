# -*- coding: utf-8 -*-

"""
Filename: GUI_train.py

Description:
Interface de génération des données et d'entraînement du U-Net 3D :
  - Données : generate_dataset.py (fond réel ou simulé, profondeur, densités, tailles ± x %, part des
    bactéries rapprochées, pas BPM, masques d'instance du simulateur, paires rapprochées) ;
  - Étage 1 (positions) : train.py (loss, normalisation, profondeur, pas Z, taille du réseau, tête
    d'orientation) ;
  - Étage 2 (instances) : train_stage2.py (fenêtres fines, décalages, négatifs, milieux de paires).
Chaque tâche lance le script en PROCESSUS INDÉPENDANT, sortie écrite dans <dossier>/train.log (ou
generate.log) ; un fil de travail (gui_messages.TaskRunner) lit ce journal et envoie au GUI les messages
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

PARAMS_FILE = os.path.join(DL_DIR, "GUI_train_params.json")
PY = sys.executable
DEFAULTS = {
    # données
    'data_out': os.path.join(DL_DIR, 'data', 'seg_sim'), 'background': 'sim', 'n_train': 240, 'n_val': 24, 'n_test': 40,
    'z_min': 2.0, 'z_max': 200.0, 'n_min': 20, 'n_max': 400, 'n_list': '100 200 300 400', 'size_jitter': 0.1,
    'sim_step': 0.25, 'save_masks': True, 'cluster_fraction': 0.3, 'pairs': True, 'seed': 401,
    # étage 1
    's1_data': os.path.join(DL_DIR, 'data', 'seg_sim'), 's1_name': 'stage1', 's1_out': os.path.join(DL_DIR, 'runs'),
    's1_loss': 'centernet', 's1_norm': 'fixed', 's1_depth_um': 208.0, 's1_z_step': 1.0, 's1_base': 16, 's1_levels': 3,
    's1_iters': 6000, 's1_lr': 0.001, 's1_orient': False,
    # étage 2
    's2_data': os.path.join(DL_DIR, 'data', 'seg_sim'), 's2_name': 'stage2', 's2_out': os.path.join(DL_DIR, 'runs'),
    's2_iters': 6000, 's2_lr': 0.001, 's2_base': 16, 's2_levels': 3, 's2_crops': 16, 's2_holos': 2,
    's2_jit_xy': 0.25, 's2_jit_z': 0.5, 's2_neg': 0.1, 's2_mid': 0.15, 's2_vmin': 20,
}
TABS = {
    "Données": [
        ('data_out', "Dossier de sortie", 'dir', "Les jeux sont écrits dans <dossier>/train, val, test, pairs"),
        ('background', "Fond", ['sim', 'hybrid'], "sim : fond plat + bruit capteur (à utiliser : généralise au film réel) ; hybrid : fonds réels nettoyés du film (déconseillé, RAPPORT_UNET3D.md)"),
        ('n_train', "Hologrammes d'entraînement", 'int', ""),
        ('n_val', "Hologrammes de validation", 'int', ""),
        ('n_test', "Hologrammes de test", 'int', ""),
        ('z_min', "z minimum (µm)", 'float', "Profondeur des bactéries"),
        ('z_max', "z maximum (µm)", 'float', ""),
        ('n_min', "N minimum (entraînement)", 'int', "Nombre de bactéries tiré en log-uniforme"),
        ('n_max', "N maximum (entraînement)", 'int', ""),
        ('n_list', "N (validation, test)", 'str', "Valeurs en alternance, séparées par des espaces"),
        ('size_jitter', "Variation de taille (±, relatif)", 'float', "0,1 : longueur et épaisseur à ± 10 % de 1,8 × 0,6 µm"),
        ('sim_step', "Pas BPM (µm)", 'float', "Aussi le pas Z du masque ; 0,25 µm : plus fin que la bactérie"),
        ('save_masks', "Masques d'instance du simulateur", 'bool', "Nécessaire pour l'étage 2"),
        ('cluster_fraction', "Bactéries rapprochées (part)", 'float', "Part placée à 0,6–4 µm d'une autre"),
        ('pairs', "Jeu de paires rapprochées", 'bool', "48 hologrammes, écarts latéraux 1–6 µm et axiaux 1–8 µm"),
        ('seed', "Graine", 'int', ""),
    ],
    "Étage 1": [
        ('s1_data', "Données", 'dir', "Dossier contenant train/ et val/"),
        ('s1_name', "Nom du run", 'str', ""),
        ('s1_out', "Dossier des runs", 'dir', ""),
        ('s1_loss', "Loss", ['centernet', 'wmse', 'bce_dice', 'focal_tversky'], "centernet : la meilleure (benchmark)"),
        ('s1_norm', "Normalisation", ['fixed', 'localmad'], "fixed : unités de bruit capteur ; localmad : médiane / MAD locales"),
        ('s1_depth_um', "Profondeur reconstruite (µm)", 'float', "0 à cette valeur ; au moins z maximum des données"),
        ('s1_z_step', "Pas Z (µm)", 'float', "1 µm ; 2 µm : réseau 2,3× plus rapide, σz 0,28 au lieu de 0,16 µm"),
        ('s1_base', "Canaux de base", 'int', "16 (1,4 M paramètres) ; 8 : 4× plus petit"),
        ('s1_levels', "Niveaux du U-Net", 'int', ""),
        ('s1_iters', "Itérations", 'int', "~0,3 s par itération (Quadro RTX 3000)"),
        ('s1_lr', "Taux d'apprentissage", 'float', ""),
        ('s1_orient', "Tête d'orientation", 'bool', "Tenseur a·aᵀ de l'axe, sans coût mesurable"),
    ],
    "Étage 2": [
        ('s2_data', "Données (avec masques)", 'dir', "Dossier contenant train/ et val/ générés avec les masques d'instance"),
        ('s2_name', "Nom du run", 'str', ""),
        ('s2_out', "Dossier des runs", 'dir', ""),
        ('s2_iters', "Itérations", 'int', "~0,55 s par itération"),
        ('s2_lr', "Taux d'apprentissage", 'float', ""),
        ('s2_base', "Canaux de base", 'int', ""),
        ('s2_levels', "Niveaux du U-Net", 'int', ""),
        ('s2_crops', "Fenêtres par hologramme", 'int', ""),
        ('s2_holos', "Hologrammes par pas", 'int', ""),
        ('s2_jit_xy', "Décalage des fenêtres XY (µm)", 'float', "Erreur de position simulée de l'étage 1"),
        ('s2_jit_z', "Décalage des fenêtres Z (µm)", 'float', ""),
        ('s2_neg', "Fenêtres négatives (part)", 'float', "Au hasard dans le volume : fausses détections"),
        ('s2_mid', "Fenêtres entre deux voisines (part)", 'float', "Milieu de deux bactéries à moins de 3 µm (détection fusionnée)"),
        ('s2_vmin', "Voxels minimum d'une instance", 'int', ""),
    ],
}
NOISE = ('experimental', 'Warning', 'warn(', 'cupy._util')
RE_IT = re.compile(r'^it (\d+)/(\d+)\s+(?:\[[^\]]*\]\s+)?loss ([\d.eE+-]+).*?([\d.]+) s/it')
RE_GEN = re.compile(r'^\[(\w+)\] (\d+)/(\d+)')


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
        self.pump = MessagePump(root, self.runner, dict(log=self.log, progress=self.on_progress, metrics=self.on_metrics,
                                                        done=self.on_done, error=self.on_error, stopped=self.on_done))
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    # ------------------------------------------------------------------ paramètres
    def init_left(self, parent):
        bar = ttk.Frame(parent)
        bar.pack(fill='x', pady=(0, 3))
        self.btn_gen = ttk.Button(bar, text="Générer les données", command=self.start_generate)
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
        self.busy_buttons = (self.btn_gen, self.btn_s1, self.btn_s2, self.btn_follow)
        nb = ttk.Notebook(parent)
        nb.pack(fill='both', expand=True)
        for tab_name, rows in TABS.items():
            tab = ttk.Frame(nb, padding=8)
            nb.add(tab, text=tab_name)
            for r, spec in enumerate(rows):
                self.add_control(tab, r, *spec)
            tab.columnconfigure(1, weight=1)

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

    def start_generate(self):
        try:
            p = self.get_params()
            n_list = [str(int(x)) for x in p['n_list'].split()]
        except ValueError as e:
            messagebox.showerror("Paramètres", str(e))
            return
        g = [PY, '-u', os.path.join(DL_DIR, 'generate_dataset.py'), '--background', p['background'], '--out', p['data_out'],
             '--size_jitter', str(p['size_jitter']), '--sim_step', str(p['sim_step']), '--seed', str(p['seed'])]
        if p['save_masks']:
            g.append('--save_masks')
        main = g + ['--splits', 'train', 'val', 'test', '--z_range', str(p['z_min']), str(p['z_max']),
                    '--n_range', str(p['n_min']), str(p['n_max']), '--n_list', *n_list,
                    '--counts', str(p['n_train']), str(p['n_val']), str(p['n_test']),
                    '--cluster_fraction', str(p['cluster_fraction'])]
        cmds = [main]
        if p['pairs']:
            cmds.append(g + ['--splits', 'pairs'])
        self._launch(cmds, p['data_out'], 'generate.log', "Génération des données")

    def start_train(self, stage):
        try:
            p = self.get_params()
        except ValueError as e:
            messagebox.showerror("Paramètres", str(e))
            return
        k = f's{stage}_'
        workdir = os.path.join(p[k + 'out'], p[k + 'name'])
        data = p[k + 'data']
        for split in ('train', 'val'):
            if not os.path.isdir(os.path.join(data, split)):
                messagebox.showerror("Données", f"pas de dossier {split}/ dans {data}")
                return
        if os.path.exists(os.path.join(workdir, 'model_best.pt')) and not messagebox.askyesno(
                "Run existant", f"{workdir} contient déjà un modèle. L'écraser ?"):
            return
        if stage == 1:
            cmd = [PY, '-u', os.path.join(DL_DIR, 'train.py'), '--name', p['s1_name'], '--out', p['s1_out'],
                   '--train_dir', os.path.join(data, 'train'), '--val_dir', os.path.join(data, 'val'),
                   '--loss', p['s1_loss'], '--norm', p['s1_norm'], '--depth_um', str(p['s1_depth_um']),
                   '--z_step', str(p['s1_z_step']), '--base', str(p['s1_base']), '--depth', str(p['s1_levels']),
                   '--iters', str(p['s1_iters']), '--lr', str(p['s1_lr'])]
            if p['s1_orient']:
                cmd += ['--orient_weight', '1']
        else:
            cmd = [PY, '-u', os.path.join(DL_DIR, 'train_stage2.py'), '--name', p['s2_name'], '--out', p['s2_out'],
                   '--data', data, '--iters', str(p['s2_iters']), '--lr', str(p['s2_lr']), '--base', str(p['s2_base']),
                   '--depth', str(p['s2_levels']), '--crops_per_holo', str(p['s2_crops']),
                   '--holos_per_step', str(p['s2_holos']), '--jitter_xy', str(p['s2_jit_xy']),
                   '--jitter_z', str(p['s2_jit_z']), '--neg_fraction', str(p['s2_neg']),
                   '--mid_fraction', str(p['s2_mid']), '--vmin', str(p['s2_vmin'])]
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

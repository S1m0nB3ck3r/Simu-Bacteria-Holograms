# -*- coding: utf-8 -*-

"""
Filename: GUI_generate_dataset.py

Description:
Interface de génération d'un jeu d'hologrammes simulés pour le U-Net 3D (moteur : simu_dataset.py). Mêmes
paramètres que simu holo bact with GUI/simulation_gui.py (volume, optique, caméra, bactéries, illumination),
plus le nombre d'hologrammes par jeu, la plage du nombre de bactéries, la part de bactéries rapprochées et la
plage d'indice de la bactérie. Écrit dans le dossier de sortie :
  - parameters.yaml : tous les paramètres de la simulation, valeurs dérivées, description des formats ;
  - train/, val/, test/ : holograms/*.bmp (hologrammes 8 bits), instances/*.npz (volume de segmentation
    d'instance, creux), positions/*.csv (positions, orientations, tailles), index.csv ; en option
    holograms_tiff/*.tif (32 bits) et instance_volume/*.tif (volume d'étiquettes complet).
Ce dossier est celui que prend GUI_train.py. La génération tourne dans un PROCESSUS INDÉPENDANT
(simu_dataset.py, journal generate.log) suivi par un fil (gui_messages.TaskRunner, MessagePump) : l'interface
reste réactive. « Arrêter » : fin de l'hologramme en cours, parameters.yaml et index.csv à jour ; « Tuer » :
arrêt immédiat. Aperçu : dernier hologramme écrit, bactéries entourées (couleur = z).
Réglages mémorisés dans GUI_generate_dataset_params.json (à côté du script).

Utilisation :
  python deep_learning_segmentation/GUI_generate_dataset.py

Author: Simon BECKER
Date: 2026-10-19

License:
GNU General Public License v3.0
"""

import json
import os
import shutil
import sys
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

DL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DL_DIR)
from gui_messages import MessagePump, TaskRunner
from GUI_train import ToolTip, follow, PY
import simu_dataset as sd

PARAMS_FILE = os.path.join(DL_DIR, "GUI_generate_dataset_params.json")
ICONS = {"CHEMINS": "📁", "SIMULATION": "🔢", "VOLUME": "📦", "OPTIQUE": "🔬", "CAMÉRA": "📷", "BACTÉRIES": "🦠",
         "ILLUMINATION": "💡", "SAUVEGARDE": "💾"}


class GenerateGUI:

    def __init__(self, root):
        self.root = root
        self.root.title("U-Net 3D — génération d'un jeu d'hologrammes simulés")
        self.root.geometry("1450x920")
        self.vars = {}
        self.runner = TaskRunner('thread')
        self.box, self.workdir, self.following = {}, None, False
        self.last_preview = 0.0
        main = ttk.PanedWindow(root, orient='horizontal')
        main.pack(fill='both', expand=True, padx=5, pady=5)
        left, right = ttk.Frame(main), ttk.Frame(main)
        main.add(left, weight=0)
        main.add(right, weight=1)
        self.init_left(left)
        self.init_right(right)
        self.set_params(self.load_params())
        for var, _ in self.vars.values():
            var.trace_add('write', lambda *_: self.root.after_idle(self.update_info))
        self.update_info()
        self.pump = MessagePump(root, self.runner, dict(log=self.log, progress=self.on_progress, done=self.on_done,
                                                        error=self.on_error, stopped=self.on_done))
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    # ------------------------------------------------------------------ paramètres
    def init_left(self, parent):
        bar = ttk.Frame(parent)
        bar.pack(fill='x', pady=(0, 5))
        self.btn_run = ttk.Button(bar, text="🚀 Lancer la génération", command=self.start)
        self.btn_stop = ttk.Button(bar, text="⏹ Arrêter", command=self.stop, state='disabled')
        self.btn_kill = ttk.Button(bar, text="Tuer", command=self.kill, state='disabled')
        self.btn_load = ttk.Button(bar, text="Charger parameters.yaml…", command=self.load_yaml)
        for b in (self.btn_run, self.btn_stop, self.btn_kill, self.btn_load):
            b.pack(side='left', padx=3)
        ToolTip(self.btn_stop, "Fin de l'hologramme en cours ; parameters.yaml et index.csv mis à jour (jeu utilisable)")
        ToolTip(self.btn_kill, "Arrêt immédiat du processus (le dernier hologramme peut être incomplet)")
        ToolTip(self.btn_load, "Reprend les paramètres d'un jeu existant")
        self.busy_buttons = (self.btn_run, self.btn_load)
        # formulaire défilant, une section par bloc (comme simulation_gui.py)
        holder = ttk.Frame(parent)
        holder.pack(fill='both', expand=True)
        canvas = tk.Canvas(holder, highlightthickness=0, width=600)
        sb = ttk.Scrollbar(holder, orient='vertical', command=canvas.yview)
        form = ttk.Frame(canvas, padding=(4, 0))
        form.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=form, anchor='nw')
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side='left', fill='both', expand=True)
        sb.pack(side='right', fill='y')
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", lambda ev: canvas.yview_scroll(int(-ev.delta / 120), "units")))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))
        row = 0
        for sec, rows in sd.LAYOUT:
            if row:
                ttk.Separator(form, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky='ew', pady=5)
                row += 1
            ttk.Label(form, text=f"{ICONS.get(sec, '')} {sec}", font=('Arial', 11, 'bold'), foreground='#2c3e50').grid(
                row=row, column=0, columnspan=3, sticky='w', padx=2, pady=(4, 2))
            row += 1
            for key, label, kind, _default, help_text in rows:
                self.add_control(form, row, key, label, kind, help_text)
                row += 1
        form.columnconfigure(1, weight=1)

    def add_control(self, parent, row, key, label, kind, help_text):
        if kind == 'bool':
            var = tk.BooleanVar()
            w = ttk.Checkbutton(parent, text=label, variable=var)
            w.grid(row=row, column=0, columnspan=3, sticky='w', padx=24, pady=1)
        else:
            var = tk.StringVar()
            ttk.Label(parent, text=label + " :", width=38, anchor='w').grid(row=row, column=0, sticky='w', padx=8, pady=1)
            w = ttk.Entry(parent, textvariable=var, width=46 if kind in ('dir', 'str') else 16)
            w.grid(row=row, column=1, sticky='we' if kind == 'dir' else 'w', pady=1)
            if kind == 'dir':
                ttk.Button(parent, text="…", width=3, command=lambda: self.browse(key)).grid(row=row, column=2, padx=2)
        ToolTip(w, help_text)
        self.vars[key] = (var, kind)

    def browse(self, key):
        var = self.vars[key][0]
        p = filedialog.askdirectory(initialdir=var.get() if os.path.isdir(var.get()) else DL_DIR)
        if p:
            var.set(p)

    def raw_params(self):
        return {k: var.get() for k, (var, kind) in self.vars.items()}

    def get_params(self):
        """Paramètres typés (ValueError si une valeur est illisible ou incohérente)."""
        try:
            p = sd.coerce(self.raw_params())
        except ValueError as e:
            raise ValueError(f"valeur numérique invalide ({e})")
        _, err = sd.derived(p)
        if err:
            raise ValueError(" ; ".join(err))
        return p

    def set_params(self, p):
        p = dict(sd.DEFAULTS, **{k: v for k, v in p.items() if k in sd.SPEC})
        for key, (var, kind) in self.vars.items():
            var.set(p[key] if kind == 'bool' else str(p[key]))

    def load_params(self):
        try:
            with open(PARAMS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (OSError, ValueError):
            return sd.DEFAULTS

    def save_params(self):
        try:
            with open(PARAMS_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.raw_params(), f, indent=2, ensure_ascii=False)
        except OSError:
            pass

    def load_yaml(self):
        d = filedialog.askdirectory(initialdir=DL_DIR, title="Dossier d'un jeu (parameters.yaml)")
        if not d:
            return
        try:
            p = sd.read_parameters(d)
        except (OSError, ValueError) as e:
            messagebox.showerror("parameters.yaml", str(e))
            return
        self.set_params(p)
        self.log(f"Paramètres repris de {d}")

    def update_info(self):
        """Valeurs dérivées, place disque et durée estimées (mise à jour à chaque saisie)."""
        try:
            p = sd.coerce(self.raw_params())
            d, err = sd.derived(p)
        except ValueError:
            self.info.configure(text="valeur numérique illisible", foreground='#c0392b')
            return
        n_tot = p['n_train'] + p['n_val'] + p['n_test']
        N = p['holo_size_xy']
        try:
            nb_vt = float(np.mean([int(x) for x in p['number_of_bacteria_val_test'].split()]))
        except ValueError:
            nb_vt = 0.0
        a, b = p['number_of_bacteria_min'], max(p['number_of_bacteria_max'], p['number_of_bacteria_min'] + 1e-9)
        nb_tr = (b - a) / max(np.log(b / a), 1e-9) if b > a else a                  # moyenne log-uniforme
        nb_mean = (p['n_train'] * nb_tr + (p['n_val'] + p['n_test']) * nb_vt) / max(n_tot, 1)
        # place : BMP N² octets ; instances ~ 90 octets par bactérie (npz compressé) ; CSV ~ 150 octets par bactérie
        per = N * N + nb_mean * 240 + (4 * N * N if p['save_hologram_tiff'] else 0) \
            + (p['z_size'] * N * N * 2 * 0.02 if p['save_instance_volume_tiff'] else 0)
        # durée mesurée (Quadro RTX 3000, 1024², pas 0,25 µm) : ~1,5 s + 0,016 s par bactérie
        sec = (1.5 + 0.016 * nb_mean) * (N / 1024) ** 2 * (0.25e-6 / p['step_z'])
        txt = (f"pixel objet {d['pixel_objet_m'] * 1e6:.4f} µm, champ {d['champ_xy_m'] * 1e6:.1f} µm, profondeur "
               f"{d['profondeur_volume_m'] * 1e6:.1f} µm ; bactéries entre z = {d['z_bacteries_m'][0] * 1e6:.1f} et "
               f"{d['z_bacteries_m'][1] * 1e6:.1f} µm du capteur ; Δn {d['delta_n'][0]:.3f}–{d['delta_n'][1]:.3f} ; "
               f"grille des masques {d['pixel_objet_m'] * 1e6:.3f} × {d['pixel_objet_m'] * 1e6:.3f} × "
               f"{p['step_z'] * 1e6:g} µm, fractions sur {d['frac_levels']} niveaux.\n"
               f"{n_tot} hologrammes, {nb_mean:.0f} bactéries en moyenne : ≈ {per * n_tot / 1e6:.0f} Mo, ≈ "
               f"{sec * n_tot / 60:.0f} min (estimation).")
        if abs(p['step_z'] - 0.25e-6) > 1e-12:
            txt += "\nPas ≠ 0,25 µm : grille fine de l'étage 2 différente de celle des modèles retenus."
        if abs(d['pixel_objet_m'] - 0.175e-6) > 1e-12:
            txt += "\nPixel objet ≠ 0,175 µm : les modèles retenus ont été entraînés à 0,175 µm."
        if err:
            txt += "\nERREUR : " + " ; ".join(err)
        self.info.configure(text=txt, foreground='#c0392b' if err else '')

    # ------------------------------------------------------------------ affichage
    def init_right(self, parent):
        self.status = ttk.Label(parent, text="✅ Prêt", font=('TkDefaultFont', 10, 'bold'))
        self.status.pack(side='top', fill='x', pady=(0, 3))
        self.info = ttk.Label(parent, text="", wraplength=780, justify='left')
        self.info.pack(side='top', fill='x', pady=(0, 5))
        prog = ttk.Frame(parent)
        prog.pack(side='bottom', fill='x')
        self.progress = ttk.Progressbar(prog, mode='determinate')
        self.progress.pack(side='left', fill='x', expand=True, padx=(0, 6))
        self.progress_label = ttk.Label(prog, text="", width=40)
        self.progress_label.pack(side='left')
        self.text = tk.Text(parent, height=11, wrap='word')
        self.text.pack(side='bottom', fill='x')
        self.fig = Figure(figsize=(6, 6))
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.fig.subplots_adjust(left=0.04, right=0.84, top=0.94, bottom=0.04)
        self.cax = self.fig.add_axes([0.86, 0.2, 0.02, 0.6])
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().configure(width=300, height=300)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        self.ax.set_axis_off()
        self.cax.set_axis_off()
        self.canvas.draw_idle()

    def log(self, msg):
        self.text.insert('end', msg + "\n")
        self.text.see('end')
        if int(self.text.index('end-1c').split('.')[0]) > 5000:
            self.text.delete('1.0', '1000.0')

    def preview(self, split, i):
        """Dernier hologramme écrit (BMP) et ses bactéries (CSV), au plus une fois par seconde."""
        if time.time() - self.last_preview < 1.0 or not self.workdir:
            return
        name = f"holo_{i:05d}"
        f = os.path.join(self.workdir, split, 'holograms', name + '.bmp')
        c = os.path.join(self.workdir, split, 'positions', name + '.csv')
        if not (os.path.exists(f) and os.path.exists(c)):
            return
        try:
            g = np.asarray(Image.open(f), dtype=np.float32)
            pos = pd.read_csv(c)
            p = sd.coerce(self.raw_params())
        except (OSError, ValueError):
            return
        self.last_preview = time.time()
        dx = p['pix_size'] / p['grossissement']
        self.ax.clear()
        self.cax.clear()
        gray = p['bmp_background_gray']
        st = max(1, g.shape[0] // 512)                     # affichage allégé : au plus 512 × 512 points
        self.ax.imshow(g[::st, ::st], cmap='gray', vmin=gray * 0.85, vmax=gray * 1.15, interpolation='nearest',
                       extent=(-0.5, g.shape[1] - 0.5, g.shape[0] - 0.5, -0.5))
        if len(pos):
            import matplotlib
            z = pos.z_m.to_numpy() * 1e6
            sm = matplotlib.cm.ScalarMappable(matplotlib.colors.Normalize(z.min(), max(z.max(), z.min() + 1)), 'turbo')
            self.ax.scatter(pos.x_m / dx, pos.y_m / dx, s=60, facecolors='none', edgecolors=sm.to_rgba(z), linewidths=1.2)
            self.fig.colorbar(sm, cax=self.cax, label='z (µm)')
        self.ax.set_title(f"{split}/{name}.bmp : {len(pos)} bactéries", fontsize=10)
        self.ax.set_axis_off()
        self.canvas.draw_idle()

    # ------------------------------------------------------------------ tâches
    def set_busy(self, busy):
        for b in self.busy_buttons:
            b.configure(state='disabled' if busy else 'normal')
        for b in (self.btn_stop, self.btn_kill):
            b.configure(state='normal' if busy else 'disabled')

    def start(self):
        try:
            p = self.get_params()
        except ValueError as e:
            messagebox.showerror("Paramètres", str(e))
            return
        out = os.path.abspath(p['output_dir'])
        old = [s for s in sd.SPLITS if os.path.isdir(os.path.join(out, s))]
        if old:
            if not messagebox.askyesno("Dossier existant", f"{out} contient déjà {', '.join(old)}/.\n"
                                       "Les supprimer et générer un nouveau jeu ?"):
                return
            for s in old:
                shutil.rmtree(os.path.join(out, s))
        os.makedirs(out, exist_ok=True)
        for f in ('STOP',):
            if os.path.exists(os.path.join(out, f)):
                os.remove(os.path.join(out, f))
        sd.write_yaml(out, p, 'nouveau')
        self.save_params()
        self.workdir, self.box = out, {}
        self.status.configure(text=f"Génération — {out}")
        self.progress.configure(value=0)
        self.set_busy(True)
        cmd = [PY, '-u', os.path.join(DL_DIR, 'simu_dataset.py'), out]
        self.runner.start(follow, [cmd], out, 'generate.log', self.box)

    def stop(self):
        if self.workdir:
            open(os.path.join(self.workdir, 'STOP'), 'w').close()
        self.log("Arrêt demandé : fin de l'hologramme en cours…")

    def kill(self):
        if messagebox.askyesno("Tuer", "Arrêter immédiatement le processus ?"):
            self.box['kill'] = True
            self.runner.stop()

    def on_progress(self, data):
        i, n, txt = data
        self.progress.configure(maximum=max(n, 1), value=i)
        split = txt.replace('génération', '').strip()
        self.progress_label.configure(text=f"{split} : {i}/{n}")
        if split in sd.SPLITS:
            self.preview(split, i - 1)

    def on_done(self, result):
        if self.runner.worker is not None:
            self.runner.worker.join(timeout=2)
        self.set_busy(False)
        if self.workdir and os.path.exists(os.path.join(self.workdir, 'STOP')):
            os.remove(os.path.join(self.workdir, 'STOP'))
        try:
            state = sd.describe(self.workdir)
        except (OSError, ValueError, TypeError):
            state = ''
        rc = (result or {}).get('returncode')
        self.status.configure(text=("✅ Terminé" if rc == 0 else f"⚠ Fin (code {rc})") + f" — {self.workdir}")
        self.log(f"Fin (code {rc}) : {state}")

    def on_error(self, tb):
        self.set_busy(False)
        self.log(tb)
        messagebox.showerror("Erreur", tb.strip().splitlines()[-1])

    def on_closing(self):
        self.save_params()
        if self.runner.busy() and self.box.get('proc') is not None and self.box['proc'].poll() is None:
            ans = messagebox.askyesnocancel(
                "Génération en cours", "Une génération tourne.\nOui : la laisser tourner (journal generate.log)\n"
                                       "Non : l'arrêter proprement\nAnnuler : rester")
            if ans is None:
                return
            if ans is False:
                self.stop()
        self.runner.stop()
        self.pump.close()
        self.root.destroy()


def main():
    root = tk.Tk()
    GenerateGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()

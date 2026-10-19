# -*- coding: utf-8 -*-

"""
Filename: GUI_holo_stats.py

Description:
Interface graphique de holo_stats (statistiques hologramme / volume / objets).
Disposition inspirée de HOLOTRACKER :
  - à gauche, onglets de paramètres (fichiers, nettoyage & optique, propagation &
    détection, journal) et bouton de lancement ;
  - à droite, onglets de visualisation (hologrammes, volume, objets, distributions,
    profils Z, statistiques avec comparaison à un autre summary.json).

Tous les paramètres se règlent par des contrôles (champs numériques, menus déroulants,
cases à cocher), dans des unités pratiques (nm, µm). Les derniers réglages sont
mémorisés automatiquement (GUI_holo_stats_params.json, à côté du script).
Le calcul tourne dans un thread : l'interface reste réactive.

Utilisation :
  python stats/GUI_holo_stats.py

Author: Simon BECKER
Date: 2026-09-18

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
import ttkbootstrap as tb
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import holo_stats as hs

UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(UTILS_DIR)
PARAMS_FILE = os.path.join(UTILS_DIR, "GUI_holo_stats_params.json")

# Style sombre des figures, accordé au thème « superhero »
BG = '#2b3e50'
DARK_RC = {
    'figure.facecolor': BG, 'axes.facecolor': '#1f2d3a', 'savefig.facecolor': BG,
    'text.color': 'white', 'axes.labelcolor': 'white', 'axes.edgecolor': '#8899aa',
    'xtick.color': 'white', 'ytick.color': 'white', 'axes.titlecolor': 'white',
    'legend.facecolor': '#1f2d3a', 'legend.edgecolor': '#8899aa', 'font.size': 9,
}
INFO_COLOR = "#9fb3c8"

# -----------------------------------------------------------------------------
# Paramètres numériques et listes : une entrée par contrôle.
#   kind  : 'spin' (champ numérique à flèches), 'combo' (menu fermé), 'combo_edit' (menu + saisie)
#   scale : valeur affichée × scale = valeur transmise à holo_stats (unités SI)
#   enable: nom de la case à cocher qui active le contrôle (None = toujours actif)
# -----------------------------------------------------------------------------
NUMERIC_PARAMS = {
    # Nettoyage & optique
    'division_power': dict(label="Puissance après division", kind='spin', type=float, frm=0.05, to=3.0, inc=0.05,
                           default=0.8, help="Appliquée après holo / ref. HOLOTRACKER : 0.8 ; 1.0 = division pure"),
    'wavelength': dict(label="Longueur d'onde (nm)", kind='spin', type=float, frm=200, to=2000, inc=1, scale=1e-9,
                       default=660, help="Longueur d'onde dans le vide"),
    'medium_index': dict(label="Indice du milieu", kind='spin', type=float, frm=1.0, to=2.5, inc=0.001,
                         default=1.33, help="Eau : 1.33"),
    'pixel_size': dict(label="Pixel caméra (µm)", kind='spin', type=float, frm=0.1, to=50, inc=0.1, scale=1e-6,
                       default=7.0, help="Taille du pixel du capteur"),
    'magnification': dict(label="Grossissement", kind='combo_edit', type=float, values=[10, 20, 40, 60, 63, 100],
                          default=40, help="Grossissement de l'objectif (saisie libre possible)"),
    'holo_size_x': dict(label="Taille X (px)", kind='combo_edit', type=int, values=[256, 512, 1024, 2048],
                        default=1024, help="Zone analysée, recadrée au centre de l'image"),
    'holo_size_y': dict(label="Taille Y (px)", kind='combo_edit', type=int, values=[256, 512, 1024, 2048],
                        default=1024, help="Zone analysée, recadrée au centre de l'image"),
    # Propagation
    'distance_ini': dict(label="Distance 1er plan (µm)", kind='spin', type=float, frm=-10000, to=10000, inc=1,
                         scale=1e-6, default=0.0,
                         help="Distance du premier plan reconstruit à l'hologramme.\n"
                              "Simulation : mettre la distance volume-caméra (ex. 10 µm)"),
    'nb_planes': dict(label="Nombre de plans", kind='spin', type=int, frm=1, to=2000, inc=10, default=200,
                      help="Mémoire GPU : plans × X × Y × 4 octets"),
    'step': dict(label="Pas Z (µm)", kind='spin', type=float, frm=0.01, to=100, inc=0.1, scale=1e-6, default=0.5,
                 help="Pas entre plans reconstruits"),
    'high_pass': dict(label="Rayon min (px)", kind='spin', type=int, frm=0, to=4096, inc=1, default=15,
                      enable='use_spectral_filter', help="Rayon minimal conservé dans le spectre"),
    'low_pass': dict(label="Rayon max (px)", kind='spin', type=int, frm=1, to=4096, inc=1, default=125,
                     enable='use_spectral_filter', help="Rayon maximal conservé dans le spectre"),
    # Détection
    'focus_type': dict(label="Critère de focus", kind='combo', type=str, values=hs.FOCUS_TYPES, default='TENEGRAD',
                       help="Critère de netteté appliqué au volume"),
    'sum_size': dict(label="Noyau de focus (px)", kind='spin', type=int, frm=1, to=101, inc=2, default=11,
                     help="Taille du noyau de sommation du focus"),
    'nb_std_threshold': dict(label="Seuil (nb σ)", kind='spin', type=float, frm=0, to=100, inc=0.5, default=14.0,
                             help="Seuil = moyenne + n × écart-type du volume de focus"),
    'connectivity': dict(label="Connectivité 3D", kind='combo', type=int, values=[6, 18, 26], default=26,
                         help="Voisinage des composantes connexes"),
    'min_voxel': dict(label="Voxels min", kind='spin', type=int, frm=0, to=10**7, inc=1, default=0,
                      enable='use_voxel_filter', help="Taille minimale des objets conservés (0 = pas de borne)"),
    'max_voxel': dict(label="Voxels max", kind='spin', type=int, frm=0, to=10**7, inc=100, default=0,
                      enable='use_voxel_filter', help="Taille maximale des objets conservés (0 = pas de borne)"),
    # Analyse
    'envelope_sigma': dict(label="Lissage enveloppe (px)", kind='spin', type=float, frm=1, to=500, inc=5,
                           default=50.0, help="Écart-type du lissage qui estime l'enveloppe d'éclairage"),
    'subsample_xy': dict(label="Sous-échantillonnage XY", kind='combo', type=int, values=[1, 2, 4, 8], default=4,
                         help="Stats CPU du volume : 1 = tout le volume (lent), 4 = 1 voxel sur 16"),
    # Référence
    'reference_mean': dict(label="Type de moyenne", kind='combo', type=str, values=['arithmetic', 'geometric'],
                           default='arithmetic',
                           help="Moyenne du dossier, comme HOLOTRACKER :\n"
                                "arithmetic : somme / n (mean_arith)\n"
                                "geometric : exp(moyenne(log)) (mean_log)"),
    # Vérité terrain
    'truth_z_offset': dict(label="Décalage Z vérité (µm)", kind='spin', type=float, frm=-10000, to=10000, inc=1,
                           scale=1e-6, default=0.0, enable='use_truth',
                           help="z vérité = z simulé + décalage.\nMettre la distance volume-caméra de la simulation."),
    'tol_xy': dict(label="Tolérance XY (µm)", kind='spin', type=float, frm=0.1, to=50, inc=0.5, scale=1e-6,
                   default=2.0, enable='use_truth', help="Distance latérale maximale pour apparier détection et objet simulé"),
    'tol_z': dict(label="Tolérance Z (µm)", kind='spin', type=float, frm=0.1, to=100, inc=0.5, scale=1e-6,
                  default=5.0, enable='use_truth',
                  help="Distance axiale maximale pour apparier.\nUne bactérie verticale donne deux blocs à environ ±4 µm."),
}
CHECKBOXES = {
    'use_spectral_filter': dict(label="Filtre spectral passe-bande", default=True,
                                help="Décoché : pas de filtrage (rayons 0 et 0)"),
    'use_voxel_filter': dict(label="Filtrer les objets par taille", default=False,
                             help="Décoché : tous les objets sont conservés"),
    'output_auto': dict(label="Dossier de sortie automatique", default=True,
                        help="<dossier de l'hologramme>/<nom>_stats_<date>/"),
    'use_truth': dict(label="Comparer aux positions simulées", default=False,
                      help="Apparie les objets détectés aux objets simulés (fichier object_positions)"),
}
FILE_DEFAULTS = {
    'hologram': os.path.join(REPO_DIR, "REAL HOLO FILM", "manipe3.859alpr9.000000.bmp").replace('\\', '/'),
    'reference': os.path.join(REPO_DIR, "REAL HOLO FILM").replace('\\', '/'),
    'reference_mode': 'dir',
    'output_dir': "",
    'cleaning': 'division',
    'positions_csv': "",
    'sim_dir': "",
}
TRUTH_LABELS = [
    ('nb_simulated', "Objets simulés"), ('nb_simulated_in_volume', "  dont dans le volume reconstruit"),
    ('nb_detected', "Objets détectés"), ('true_positives', "Appariés (vrais positifs)"),
    ('false_negatives', "Manqués (faux négatifs)"), ('false_positives', "Faux positifs"),
    ('duplicates', "Doublons (2e bloc d'un objet déjà apparié)"),
    ('recall', "Rappel"), ('precision', "Précision (doublons exclus)"),
    ('precision_strict', "Précision stricte (doublons = faux)"), ('f1', "F1"),
    ('error_x_mean_um', "Erreur x moyenne (µm)"), ('error_y_mean_um', "Erreur y moyenne (µm)"),
    ('error_z_mean_um', "Erreur z moyenne (µm)"), ('error_x_std_um', "Erreur x écart-type (µm)"),
    ('error_y_std_um', "Erreur y écart-type (µm)"), ('error_z_std_um', "Erreur z écart-type (µm)"),
    ('error_xy_rms_um', "Erreur XY RMS (µm)"), ('error_z_rms_um', "Erreur Z RMS (µm)"),
    ('recall_phi_0_30', "Rappel, φ 0–30° (proche de l'axe optique)"), ('recall_phi_30_60', "Rappel, φ 30–60°"),
    ('recall_phi_60_90', "Rappel, φ 60–90° (couchée)"),
    ('tolerance_xy_um', "Tolérance XY (µm)"), ('tolerance_z_um', "Tolérance Z (µm)"),
]
IMAGE_TYPES = [("Images", "*.bmp *.png *.tif *.tiff *.npy"), ("Tous", "*.*")]

OBJECT_COLUMNS = [
    ('label', "#", 50, '{:.0f}'), ('nb_vox', "voxels", 70, '{:.0f}'),
    ('bary_x_um', "x (µm)", 70, '{:.2f}'), ('bary_y_um', "y (µm)", 70, '{:.2f}'), ('bary_z_um', "z (µm)", 70, '{:.2f}'),
    ('focus_peak', "pic", 80, '{:.4g}'), ('peak_snr_robust', "RSB pic", 70, '{:.1f}'),
    ('extent_x_um', "Δx (µm)", 65, '{:.2f}'), ('extent_y_um', "Δy (µm)", 65, '{:.2f}'), ('extent_z_um', "Δz (µm)", 65, '{:.2f}'),
    ('nearest_neighbor_um', "voisin (µm)", 80, '{:.2f}'),
    ('module_norm_peak', "pic module (σ)", 95, '{:.1f}'),
]


class ToolTip:
    """Info-bulle au survol."""
    def __init__(self, widget, text):
        self.widget, self.text, self.tip = widget, text, None
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)

    def show(self, _event=None):
        if self.tip or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 2
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self.tip, text=self.text, justify="left", background="#ffffe0", foreground="black",
                 relief="solid", borderwidth=1, font=("Segoe UI", 9)).pack(ipadx=4, ipady=2)

    def hide(self, _event=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None


class PlotPanel(ttk.Frame):
    """Figure matplotlib intégrée, avec barre d'outils (zoom, déplacement, enregistrement)."""
    def __init__(self, parent, figsize=(5, 4)):
        super().__init__(parent)
        with matplotlib.rc_context(DARK_RC):
            self.fig = Figure(figsize=figsize)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        # Windows à échelle 125 % : matplotlib multiplie la figure par tk scaling / (96/72),
        # alors que Tk (processus « DPI aware » sous ttkbootstrap) mesure déjà en pixels
        # physiques -> figure 25 % trop grande, rognée à droite et en bas. On fixe le rapport à 1.
        self.canvas.get_tk_widget().unbind("<Map>")
        self.canvas._set_device_pixel_ratio(1.0)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def draw(self, plot_function):
        """Efface la figure et appelle plot_function(fig) dans le style sombre."""
        with matplotlib.rc_context(DARK_RC):
            self.fig.clear()
            self.fig.set_facecolor(BG)
            plot_function(self.fig)
            try:
                self.fig.tight_layout()
            except Exception:
                pass
        self.canvas.draw_idle()


def percentile_limits(img, lo=0.5, hi=99.5):
    a, b = np.percentile(img, [lo, hi])
    return (a, b) if b > a else (img.min(), img.max() + 1e-12)


class HoloStatsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Holo Stats — statistiques hologramme / volume / objets")
        self.root.geometry("1500x850")
        try:
            self.root.state('zoomed')   # plein écran sous Windows (la barre des tâches reste visible)
        except tk.TclError:
            pass
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.num_vars = {}              # nom -> tk.StringVar (contrôles numériques et listes)
        self.num_widgets = {}           # nom -> widget (pour activer / désactiver)
        self.check_vars = {}            # nom -> tk.BooleanVar
        self.result = None
        self.compare_summary = None
        self.compare_path = None
        self.selected_object = None
        self.worker = None
        self.msg_queue = queue.Queue()
        self.status_var = tk.StringVar(value="Prêt")

        self.paned = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        self.paned.pack(fill="both", expand=True)
        self.left_frame = ttk.Frame(self.paned, width=430)
        self.right_frame = ttk.Frame(self.paned)
        self.paned.add(self.left_frame, weight=0)
        self.paned.add(self.right_frame, weight=1)

        self.init_left_frame()
        self.init_right_frame()
        # La molette modifie un champ numérique ou un menu déroulant Tk dès qu'elle est tournée
        # au-dessus : une valeur peut changer en faisant défiler, sans qu'on le remarque.
        for widget_class in ("TSpinbox", "TCombobox"):
            self.root.bind_class(widget_class, "<MouseWheel>", lambda e: "break")
        self.load_params()
        self._poll_id = self.root.after(100, self.poll_queue)

    # =========================================================================
    # PANNEAU GAUCHE : PARAMÈTRES
    # =========================================================================

    def init_left_frame(self):
        # bas du panneau, toujours visible : lancement et état
        bottom = ttk.Frame(self.left_frame)
        bottom.pack(side="bottom", fill="x", padx=8, pady=8)
        row = ttk.Frame(bottom)
        row.pack(fill="x", pady=(0, 6))
        self.run_button = tb.Button(row, text="▶  Lancer l'analyse", bootstyle="success", command=self.run_analysis)
        self.run_button.pack(side="left", fill="x", expand=True)
        tb.Button(row, text="↺ Défauts", bootstyle="secondary", command=self.reset_params).pack(side="left", padx=(6, 0))
        self.progress = tb.Progressbar(bottom, mode="indeterminate", bootstyle="success-striped")
        self.progress.pack(fill="x")

        self.left_notebook = ttk.Notebook(self.left_frame)
        self.left_notebook.pack(fill="both", expand=True)

        tab = ttk.Frame(self.left_notebook)
        self.left_notebook.add(tab, text="Fichiers")
        self.init_files_tab(tab)

        tab = ttk.Frame(self.left_notebook)
        self.left_notebook.add(tab, text="Simulation")
        self.init_simulation_tab(tab)

        tab = ttk.Frame(self.left_notebook)
        self.left_notebook.add(tab, text="Nettoyage & optique")
        self.init_optics_tab(tab)

        tab = ttk.Frame(self.left_notebook)
        self.left_notebook.add(tab, text="Propagation & détection")
        self.init_detection_tab(tab)

        tab = ttk.Frame(self.left_notebook)
        self.left_notebook.add(tab, text="Journal")
        self.log_tab = tab
        self.log_text = tk.Text(tab, wrap="none", font=("Consolas", 9), bg="#1f2d3a", fg="white",
                                insertbackground="white")
        ys = ttk.Scrollbar(tab, orient="vertical", command=self.log_text.yview)
        xs = ttk.Scrollbar(tab, orient="horizontal", command=self.log_text.xview)
        self.log_text.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        ys.pack(side="right", fill="y")
        xs.pack(side="bottom", fill="x")
        self.log_text.pack(fill="both", expand=True)

        for var in (list(self.num_vars.values()) + list(self.check_vars.values())
                    + [self.cleaning_var, self.reference_mode]):
            var.trace_add("write", lambda *_: self.update_states())

    # ---- onglet Fichiers -----------------------------------------------------

    def init_files_tab(self, tab):
        box = ttk.Labelframe(tab, text="Hologramme")
        box.pack(fill="x", padx=8, pady=8)
        self.hologram_var = tk.StringVar()
        row = ttk.Frame(box)
        row.pack(fill="x", padx=6, pady=6)
        ttk.Entry(row, textvariable=self.hologram_var).pack(side="left", fill="x", expand=True)
        tb.Button(row, text="📄", width=3, bootstyle="secondary", command=self.browse_hologram).pack(side="left", padx=(3, 0))
        ttk.Label(box, text=".bmp, .png, .tif, .tiff, .npy (image 2D)", foreground=INFO_COLOR).pack(anchor="w", padx=6, pady=(0, 6))

        box = ttk.Labelframe(tab, text="Référence (hologramme moyen)")
        box.pack(fill="x", padx=8, pady=8)
        self.reference_mode = tk.StringVar(value='dir')
        row = ttk.Frame(box)
        row.pack(fill="x", padx=6, pady=(6, 0))
        rb = ttk.Radiobutton(row, text="Fichier image", value='file', variable=self.reference_mode)
        rb.pack(side="left")
        ToolTip(rb, "Image à la même échelle que l'hologramme (pas une moyenne ré-étirée sur 0-255)")
        rb = ttk.Radiobutton(row, text="Dossier : moyenne de ses images", value='dir', variable=self.reference_mode)
        rb.pack(side="left", padx=(12, 0))
        ToolTip(rb, "Hologramme moyen de toutes les images du dossier ayant l'extension de l'hologramme")
        self.reference_var = tk.StringVar()
        row = ttk.Frame(box)
        row.pack(fill="x", padx=6, pady=6)
        ttk.Entry(row, textvariable=self.reference_var).pack(side="left", fill="x", expand=True)
        tb.Button(row, text="📂", width=3, bootstyle="secondary", command=self.browse_reference).pack(side="left", padx=(3, 0))
        self.add_numeric_rows(self._grid(box), ['reference_mean'])

        box = ttk.Labelframe(tab, text="Sortie")
        box.pack(fill="x", padx=8, pady=8)
        self.add_checkbox(box, 'output_auto').pack(anchor="w", padx=6, pady=(6, 0))
        self.output_var = tk.StringVar()
        row = ttk.Frame(box)
        row.pack(fill="x", padx=6, pady=6)
        self.output_entry = ttk.Entry(row, textvariable=self.output_var)
        self.output_entry.pack(side="left", fill="x", expand=True)
        self.output_button = tb.Button(row, text="📁", width=3, bootstyle="secondary", command=self.browse_output_dir)
        self.output_button.pack(side="left", padx=(3, 0))

        box = ttk.Labelframe(tab, text="Comparaison")
        box.pack(fill="x", padx=8, pady=8)
        ttk.Label(box, text="summary.json d'une autre analyse (ex. réel ↔ simulé),\n"
                            "affiché à côté des résultats dans l'onglet Statistiques.",
                  foreground=INFO_COLOR).pack(anchor="w", padx=6, pady=4)
        row = ttk.Frame(box)
        row.pack(fill="x", padx=6, pady=(0, 6))
        tb.Button(row, text="📂 Charger un summary.json", bootstyle="secondary",
                  command=self.load_comparison).pack(side="left", fill="x", expand=True)
        tb.Button(row, text="✕", width=3, bootstyle="secondary",
                  command=self.clear_comparison).pack(side="left", padx=(3, 0))
        self.compare_label = ttk.Label(box, text="aucune", wraplength=380, foreground=INFO_COLOR)
        self.compare_label.pack(anchor="w", padx=6, pady=(0, 6))

    # ---- onglet Simulation ---------------------------------------------------

    def init_simulation_tab(self, tab):
        self.sim_params, self.sim_holos = None, {}
        box = ttk.Labelframe(tab, text="Charger une simulation")
        box.pack(fill="x", padx=8, pady=8)
        ttk.Label(box, text="Dossier de sortie d'une simulation (GUI ou script) : ses paramètres\n"
                            "optiques et de volume remplissent les autres onglets, l'hologramme\n"
                            "de référence (holo_ref) et les positions des objets sont repris.",
                  foreground=INFO_COLOR, justify="left").pack(anchor="w", padx=6, pady=(4, 6))
        tb.Button(box, text="📂 Choisir le dossier de simulation…", bootstyle="info",
                  command=self.browse_simulation).pack(fill="x", padx=6)
        self.sim_dir_var = tk.StringVar()
        ttk.Label(box, textvariable=self.sim_dir_var, wraplength=390, foreground=INFO_COLOR).pack(anchor="w", padx=6, pady=4)
        grid = self._grid(box)
        ttk.Label(grid, text="Hologramme").grid(row=0, column=0, sticky="w", pady=3, padx=(0, 8))
        self.sim_holo_var = tk.StringVar()
        self.sim_holo_cb = ttk.Combobox(grid, textvariable=self.sim_holo_var, state="readonly")
        self.sim_holo_cb.grid(row=0, column=1, sticky="ew", pady=3)
        self.sim_holo_cb.bind("<<ComboboxSelected>>", lambda e: self.on_sim_hologram_selected())
        ttk.Label(grid, text="Format").grid(row=1, column=0, sticky="w", pady=3, padx=(0, 8))
        self.sim_format_var = tk.StringVar()
        self.sim_format_cb = ttk.Combobox(grid, textvariable=self.sim_format_var, state="readonly")
        self.sim_format_cb.grid(row=1, column=1, sticky="ew", pady=3)
        self.sim_format_cb.bind("<<ComboboxSelected>>", lambda e: self.apply_simulation())
        ToolTip(self.sim_format_cb, ".tiff / .npy : intensité flottante brute\n.bmp : normalisé min-max sur 0-255")
        self.sim_info = ttk.Label(box, foreground=INFO_COLOR, justify="left")
        self.sim_info.pack(anchor="w", padx=6, pady=(0, 6))

        box = ttk.Labelframe(tab, text="Vérité terrain")
        box.pack(fill="x", padx=8, pady=8)
        self.add_checkbox(box, 'use_truth').pack(anchor="w", padx=6, pady=(6, 0))
        self.positions_var = tk.StringVar()
        row = ttk.Frame(box)
        row.pack(fill="x", padx=6, pady=6)
        self.positions_entry = ttk.Entry(row, textvariable=self.positions_var)
        self.positions_entry.pack(side="left", fill="x", expand=True)
        self.positions_button = tb.Button(row, text="📄", width=3, bootstyle="secondary", command=self.browse_positions)
        self.positions_button.pack(side="left", padx=(3, 0))
        ToolTip(self.positions_entry, "CSV du simulateur (object_positions/bact_*.csv, bacteria_*.csv, sphere_*.csv)")
        self.add_numeric_rows(self._grid(box), ['truth_z_offset', 'tol_xy', 'tol_z'])
        ttk.Label(box, text="Repères : x analyse = y simulé, y analyse = x simulé\n"
                            "(le simulateur place x en ligne de l'image) ; z = z simulé + décalage.",
                  foreground=INFO_COLOR, justify="left").pack(anchor="w", padx=6, pady=(0, 6))

    def browse_simulation(self):
        path = filedialog.askdirectory(title="Dossier de sortie d'une simulation",
                                       initialdir=self._initial_dir(self.sim_dir_var.get()))
        if path:
            self.load_simulation_dir(path, apply=True)

    def load_simulation_dir(self, path, apply=True):
        params = hs.load_simulation_parameters(path)
        holos = hs.list_simulated_holograms(path)
        if apply and (params is None or not holos):
            messagebox.showerror("Simulation", "Dossier de simulation non reconnu :\n"
                                 + ("aucun fichier de paramètres (parameters_simu_bact.json ou config_simu.json)\n"
                                    if params is None else "")
                                 + ("aucun hologramme dans simulated_hologram/" if not holos else ""))
            return
        self.sim_params, self.sim_holos = params, holos
        self.sim_dir_var.set(path)
        self.sim_holo_cb.configure(values=list(holos))
        if holos:
            self.sim_holo_var.set(next(iter(holos)))
            self.on_sim_hologram_selected(apply=apply)

    def on_sim_hologram_selected(self, apply=True):
        entry = self.sim_holos.get(self.sim_holo_var.get())
        if not entry:
            return
        formats = sorted(entry['formats'], key=lambda e: ['.tiff', '.tif', '.npy', '.bmp'].index(e))
        self.sim_format_cb.configure(values=formats)
        if self.sim_format_var.get() not in formats:
            self.sim_format_var.set(formats[0])
        if apply:
            self.apply_simulation()

    def apply_simulation(self):
        entry, p = self.sim_holos.get(self.sim_holo_var.get()), self.sim_params
        if not entry or p is None:
            return
        ext = self.sim_format_var.get()
        self.hologram_var.set(entry['formats'][ext].replace('\\', '/'))
        notes = []
        if entry['reference']:
            ref_ext = ext if ext in entry['reference'] else sorted(entry['reference'])[0]
            self.reference_mode.set('file')
            self.reference_var.set(entry['reference'][ref_ext].replace('\\', '/'))
            if ref_ext != ext:
                notes.append(f"référence en {ref_ext} (pas de {ext})")
        else:
            notes.append("pas d'hologramme de référence (holo_ref) : cocher l'option dans le simulateur")
        fmt = lambda v: f"{v:g}"
        self.num_vars['wavelength'].set(fmt(p['wavelength'] * 1e9))
        self.num_vars['medium_index'].set(fmt(p['index']))
        self.num_vars['pixel_size'].set(fmt(p['pixel'] * 1e6))
        self.num_vars['magnification'].set(fmt(p['mag']))
        self.num_vars['holo_size_x'].set(str(int(p['size'])))
        self.num_vars['holo_size_y'].set(str(int(p['size'])))
        self.num_vars['distance_ini'].set(fmt(p['dvc'] * 1e6))
        self.num_vars['nb_planes'].set(str(int(p['nz'])))
        self.num_vars['step'].set(fmt(p['step'] * 1e6))
        self.num_vars['truth_z_offset'].set(fmt(p['dvc'] * 1e6))
        self.positions_var.set((entry['positions'] or "").replace('\\', '/'))
        self.check_vars['use_truth'].set(bool(entry['positions']))
        if not entry['positions']:
            notes.append("pas de fichier de positions : comparaison désactivée")
        self.sim_info.config(text=(f"λ {p['wavelength'] * 1e9:g} nm, n {p['index']:g}, pixel {p['pixel'] * 1e6:g} µm, "
                                   f"×{p['mag']:g}\n{int(p['nz'])} plans de {p['step'] * 1e6:g} µm, "
                                   f"volume à {p['dvc'] * 1e6:g} µm de la caméra\n"
                                   f"paramètres : {os.path.basename(p['source'])}"
                                   + "".join(f"\n⚠ {n}" for n in notes)))
        self.status_var.set(f"Simulation chargée : hologramme {self.sim_holo_var.get()} ({ext})")

    def browse_positions(self):
        path = filedialog.askopenfilename(title="Positions des objets simulés",
                                          initialdir=self._initial_dir(self.positions_var.get()),
                                          filetypes=[("CSV", "*.csv"), ("Tous", "*.*")])
        if path:
            self.positions_var.set(path)

    # ---- onglets de paramètres -----------------------------------------------

    def init_optics_tab(self, tab):
        box = ttk.Labelframe(tab, text="Nettoyage")
        box.pack(fill="x", padx=8, pady=8)
        grid = self._grid(box)
        self.cleaning_var = tk.StringVar(value='division')
        lab = ttk.Label(grid, text="Méthode")
        lab.grid(row=0, column=0, sticky="w", pady=3, padx=(0, 8))
        cb = ttk.Combobox(grid, textvariable=self.cleaning_var, values=['division', 'subtraction', 'none'],
                          state="readonly")
        cb.grid(row=0, column=1, sticky="ew", pady=3)
        help_text = ("division : holo / ref, puis puissance\n"
                     "subtraction : holo - ref, puis retrait du minimum\n"
                     "none : hologramme brut (référence lue pour ses statistiques)")
        ToolTip(lab, help_text)
        ToolTip(cb, help_text)
        self.add_numeric_rows(grid, ['division_power'], start_row=1)

        box = ttk.Labelframe(tab, text="Optique")
        box.pack(fill="x", padx=8, pady=8)
        self.add_numeric_rows(self._grid(box), ['wavelength', 'medium_index', 'pixel_size', 'magnification',
                                                'holo_size_x', 'holo_size_y'])
        self.optics_info = ttk.Label(box, foreground=INFO_COLOR, justify="left")
        self.optics_info.pack(anchor="w", padx=6, pady=(0, 6))

    def init_detection_tab(self, tab):
        box = ttk.Labelframe(tab, text="Propagation")
        box.pack(fill="x", padx=8, pady=(8, 4))
        grid = self._grid(box)
        self.add_numeric_rows(grid, ['distance_ini', 'nb_planes', 'step'])
        self.add_checkbox(grid, 'use_spectral_filter').grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self.add_numeric_rows(grid, ['high_pass', 'low_pass'], start_row=4)
        self.volume_info = ttk.Label(box, foreground=INFO_COLOR, justify="left")
        self.volume_info.pack(anchor="w", padx=6, pady=(0, 6))

        box = ttk.Labelframe(tab, text="Détection")
        box.pack(fill="x", padx=8, pady=4)
        grid = self._grid(box)
        self.add_numeric_rows(grid, ['focus_type', 'sum_size', 'nb_std_threshold', 'connectivity'])
        self.add_checkbox(grid, 'use_voxel_filter').grid(row=4, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self.add_numeric_rows(grid, ['min_voxel', 'max_voxel'], start_row=5)

        box = ttk.Labelframe(tab, text="Analyse")
        box.pack(fill="x", padx=8, pady=4)
        self.add_numeric_rows(self._grid(box), ['envelope_sigma', 'subsample_xy'])

    @staticmethod
    def _grid(parent):
        grid = ttk.Frame(parent)
        grid.pack(fill="x", padx=6, pady=6)
        grid.columnconfigure(1, weight=1)
        return grid

    def add_numeric_rows(self, grid, names, start_row=0):
        for r, name in enumerate(names, start=start_row):
            spec = NUMERIC_PARAMS[name]
            lab = ttk.Label(grid, text=spec['label'])
            lab.grid(row=r, column=0, sticky="w", pady=3, padx=(0, 8))
            var = tk.StringVar()
            if spec['kind'] == 'spin':
                w = ttk.Spinbox(grid, textvariable=var, from_=spec['frm'], to=spec['to'], increment=spec['inc'])
                help_text = f"{spec['help']}\nPlage : {spec['frm']} à {spec['to']}"
            else:
                w = ttk.Combobox(grid, textvariable=var, values=[str(v) for v in spec['values']],
                                 state="readonly" if spec['kind'] == 'combo' else "normal")
                help_text = spec['help']
            w.grid(row=r, column=1, sticky="ew", pady=3)
            ToolTip(lab, help_text)
            ToolTip(w, help_text)
            self.num_vars[name], self.num_widgets[name] = var, w

    def add_checkbox(self, parent, name):
        spec = CHECKBOXES[name]
        var = tk.BooleanVar(value=spec['default'])
        cb = ttk.Checkbutton(parent, text=spec['label'], variable=var, bootstyle="round-toggle")
        ToolTip(cb, spec['help'])
        self.check_vars[name] = var
        return cb

    def update_states(self):
        """Active / désactive les contrôles dépendants et met à jour les informations dérivées."""
        for name, spec in NUMERIC_PARAMS.items():
            enabled = spec.get('enable') is None or self.check_vars[spec['enable']].get()
            if name == 'division_power':
                enabled = self.cleaning_var.get() == 'division'
            if name == 'reference_mean':
                enabled = self.reference_mode.get() == 'dir'
            self.num_widgets[name].configure(state=("readonly" if spec['kind'] == 'combo' else "normal")
                                             if enabled else "disabled")
        auto = self.check_vars['output_auto'].get()
        self.output_entry.configure(state="disabled" if auto else "normal")
        self.output_button.configure(state="disabled" if auto else "normal")
        truth = self.check_vars['use_truth'].get()
        self.positions_entry.configure(state="normal" if truth else "disabled")
        self.positions_button.configure(state="normal" if truth else "disabled")

        v = {}
        for name in ('pixel_size', 'magnification', 'holo_size_x', 'holo_size_y', 'nb_planes', 'step', 'distance_ini'):
            try:
                v[name] = float(self.num_vars[name].get())
            except ValueError:
                return
        if v['magnification'] > 0:
            dx_um = v['pixel_size'] / v['magnification']
            self.optics_info.config(text=f"Pixel côté objet : {1e3 * dx_um:.1f} nm\n"
                                         f"Champ : {v['holo_size_x'] * dx_um:.1f} × {v['holo_size_y'] * dx_um:.1f} µm")
        z0, z1 = v['distance_ini'], v['distance_ini'] + (v['nb_planes'] - 1) * v['step']
        mem = v['nb_planes'] * v['holo_size_x'] * v['holo_size_y'] * 4 / 2**20
        self.volume_info.config(text=f"Plans reconstruits : z de {z0:.1f} à {z1:.1f} µm\n"
                                     f"Volume GPU : {mem:.0f} Mo (float32)")

    # ---- lecture / mémorisation des réglages ---------------------------------

    def get_params(self):
        """État de tous les contrôles, tel qu'affiché (pour la mémorisation)."""
        return {
            'hologram': self.hologram_var.get(), 'reference': self.reference_var.get(),
            'reference_mode': self.reference_mode.get(), 'output_dir': self.output_var.get(),
            'cleaning': self.cleaning_var.get(),
            'positions_csv': self.positions_var.get(), 'sim_dir': self.sim_dir_var.get(),
            'sim_hologram': self.sim_holo_var.get(), 'sim_format': self.sim_format_var.get(),
            **{k: v.get() for k, v in self.num_vars.items()},
            **{k: v.get() for k, v in self.check_vars.items()},
        }

    def set_params(self, p):
        self.hologram_var.set(p.get('hologram', FILE_DEFAULTS['hologram']))
        self.reference_var.set(p.get('reference', FILE_DEFAULTS['reference']))
        self.reference_mode.set(p.get('reference_mode', FILE_DEFAULTS['reference_mode']))
        self.output_var.set(p.get('output_dir', FILE_DEFAULTS['output_dir']))
        self.cleaning_var.set(p.get('cleaning', FILE_DEFAULTS['cleaning']))
        for name, var in self.num_vars.items():
            var.set(str(p.get(name, NUMERIC_PARAMS[name]['default'])))
        for name, var in self.check_vars.items():
            var.set(bool(p.get(name, CHECKBOXES[name]['default'])))
        self.positions_var.set(p.get('positions_csv', FILE_DEFAULTS['positions_csv']))
        self.sim_dir_var.set("")
        self.sim_holo_cb.configure(values=[])
        sim_dir = p.get('sim_dir', FILE_DEFAULTS['sim_dir'])
        if sim_dir and os.path.isdir(sim_dir):
            self.load_simulation_dir(sim_dir, apply=False)
            if p.get('sim_hologram') in self.sim_holos:
                self.sim_holo_var.set(p['sim_hologram'])
                self.on_sim_hologram_selected(apply=False)
                if p.get('sim_format') in self.sim_holos[p['sim_hologram']]['formats']:
                    self.sim_format_var.set(p['sim_format'])
        self.update_states()

    def load_params(self):
        params = {}
        if os.path.exists(PARAMS_FILE):
            try:
                with open(PARAMS_FILE, 'r', encoding='utf-8') as f:
                    params = json.load(f)
            except Exception:
                params = {}
        self.set_params(params)

    def save_params(self):
        try:
            with open(PARAMS_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.get_params(), f, indent=2, ensure_ascii=False)
        except OSError:
            pass

    def reset_params(self):
        if messagebox.askyesno("Valeurs par défaut", "Remettre tous les paramètres à leur valeur par défaut ?"):
            self.set_params({})

    def build_args(self):
        """
        Convertit les contrôles en paramètres holo_stats (unités SI), puis les valide avec
        hs.validate_config. Lève ValueError avec la liste de toutes les erreurs.
        """
        errors, vals = [], {}
        for name, spec in NUMERIC_PARAMS.items():
            text = self.num_vars[name].get().strip()
            try:
                v = spec['type'](float(text)) if spec['type'] is int else spec['type'](text)
                if spec['type'] is int and float(text) != int(float(text)):
                    raise ValueError
            except ValueError:
                errors.append(f"{spec['label']} : « {text} » n'est pas un {'entier' if spec['type'] is int else 'nombre'}")
                continue
            if spec['kind'] == 'spin' and not spec['frm'] <= v <= spec['to']:
                errors.append(f"{spec['label']} : {v} hors de la plage {spec['frm']} à {spec['to']}")
            vals[name] = v * spec.get('scale', 1) if spec['type'] is float else v
        if vals.get('magnification', 1) <= 0 or vals.get('holo_size_x', 1) <= 0 or vals.get('holo_size_y', 1) <= 0:
            errors.append("Grossissement et tailles doivent être > 0")
        if not self.check_vars['use_spectral_filter'].get():
            vals['high_pass'], vals['low_pass'] = 0, 0
        elif 'high_pass' in vals and 'low_pass' in vals and vals['high_pass'] >= vals['low_pass']:
            errors.append("Filtre spectral : le rayon min doit être < au rayon max")
        if not self.check_vars['use_voxel_filter'].get():
            vals['min_voxel'], vals['max_voxel'] = 0, 0

        hologram, reference = self.hologram_var.get().strip(), self.reference_var.get().strip()
        if not os.path.isfile(hologram):
            errors.append(f"Hologramme introuvable : {hologram}")
        if self.reference_mode.get() == 'file' and not os.path.isfile(reference):
            errors.append(f"Référence : fichier introuvable : {reference}")
        if self.reference_mode.get() == 'dir' and not os.path.isdir(reference):
            errors.append(f"Référence : dossier introuvable : {reference}")
        output = None if self.check_vars['output_auto'].get() else self.output_var.get().strip()
        if output == "":
            errors.append("Dossier de sortie vide (ou cocher « automatique »)")
        positions = self.positions_var.get().strip() if self.check_vars['use_truth'].get() else None
        if positions is not None and not os.path.isfile(positions):
            errors.append(f"Vérité terrain : fichier de positions introuvable : {positions}")
        if errors:
            raise ValueError("Paramètres invalides :\n  - " + "\n  - ".join(errors))

        raw = {
            'inputs': {'hologram': hologram, 'reference': reference, 'output_dir': output},
            'cleaning': {'type': self.cleaning_var.get(), 'division_power': vals['division_power'],
                         'reference_mean': vals['reference_mean']},
            'optics': {k: vals[k] for k in ('wavelength', 'medium_index', 'pixel_size', 'magnification',
                                            'holo_size_x', 'holo_size_y')},
            'propagation': {k: vals[k] for k in ('distance_ini', 'nb_planes', 'step', 'high_pass', 'low_pass')},
            'detection': {k: vals[k] for k in ('focus_type', 'sum_size', 'nb_std_threshold', 'connectivity',
                                               'min_voxel', 'max_voxel')},
            'analysis': {k: vals[k] for k in ('envelope_sigma', 'subsample_xy')},
            'ground_truth': {'positions_csv': positions, 'z_offset': vals['truth_z_offset'],
                             'tolerance_xy': vals['tol_xy'], 'tolerance_z': vals['tol_z']},
        }
        return hs.validate_config(raw, REPO_DIR)

    # ---- parcourir -----------------------------------------------------------

    @staticmethod
    def _initial_dir(path):
        path = path.strip()
        if os.path.isdir(path):
            return path
        return os.path.dirname(path) if path and os.path.isdir(os.path.dirname(path)) else REPO_DIR

    def browse_hologram(self):
        path = filedialog.askopenfilename(title="Hologramme", initialdir=self._initial_dir(self.hologram_var.get()),
                                          filetypes=IMAGE_TYPES)
        if path:
            self.hologram_var.set(path)

    def browse_reference(self):
        start = self._initial_dir(self.reference_var.get())
        if self.reference_mode.get() == 'file':
            path = filedialog.askopenfilename(title="Hologramme de référence", initialdir=start, filetypes=IMAGE_TYPES)
        else:
            path = filedialog.askdirectory(title="Dossier d'hologrammes à moyenner", initialdir=start)
        if path:
            self.reference_var.set(path)

    def browse_output_dir(self):
        path = filedialog.askdirectory(title="Dossier de sortie", initialdir=self._initial_dir(self.output_var.get()))
        if path:
            self.output_var.set(path)

    # =========================================================================
    # PANNEAU DROIT : VISUALISATION
    # =========================================================================

    def init_right_frame(self):
        ttk.Label(self.right_frame, textvariable=self.status_var, relief="sunken", anchor="w").pack(
            side="bottom", fill="x", padx=2, pady=2)
        self.right_notebook = ttk.Notebook(self.right_frame)
        self.right_notebook.pack(fill="both", expand=True)

        # Hologrammes
        tab = ttk.Frame(self.right_notebook)
        self.right_notebook.add(tab, text="Hologrammes")
        self.holo_panel = PlotPanel(tab)
        self.holo_panel.pack(fill="both", expand=True)

        # Volume
        tab = ttk.Frame(self.right_notebook)
        self.right_notebook.add(tab, text="Volume")
        bar = ttk.Frame(tab)
        bar.pack(fill="x", padx=6, pady=4)
        self.volume_kind = tk.StringVar(value="focus")
        ttk.Label(bar, text="Projection max. :").pack(side="left")
        for val, txt in (("focus", "Focus"), ("module", "Module propagé")):
            ttk.Radiobutton(bar, text=txt, value=val, variable=self.volume_kind,
                            command=self.draw_volume).pack(side="left", padx=6)
        self.show_detections = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Détections", variable=self.show_detections,
                        command=self.draw_volume).pack(side="left", padx=12)
        self.show_truth = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Vérité terrain", variable=self.show_truth,
                        command=self.draw_volume).pack(side="left")
        ttk.Label(bar, text="Clic sur la vue XY : sélection de l'objet le plus proche",
                  foreground="#9fb3c8").pack(side="right")
        self.volume_panel = PlotPanel(tab)
        self.volume_panel.pack(fill="both", expand=True)
        self.volume_panel.canvas.mpl_connect("button_press_event", self.on_volume_click)

        # Objets
        tab = ttk.Frame(self.right_notebook)
        self.right_notebook.add(tab, text="Objets")
        pw = ttk.PanedWindow(tab, orient=tk.VERTICAL)
        pw.pack(fill="both", expand=True)
        table_frame = ttk.Frame(pw)
        self.obj_tree = ttk.Treeview(table_frame, columns=[c[0] for c in OBJECT_COLUMNS] + ['truth'],
                                     show="headings", height=12, selectmode="browse")
        for key, title, width, _ in OBJECT_COLUMNS:
            self.obj_tree.heading(key, text=title, command=lambda k=key: self.sort_objects(k))
            self.obj_tree.column(key, width=width, anchor="e")
        self.obj_tree.heading('truth', text="vérité", command=lambda: self.sort_objects('truth'))
        self.obj_tree.column('truth', width=110, anchor="w")
        ys = ttk.Scrollbar(table_frame, orient="vertical", command=self.obj_tree.yview)
        self.obj_tree.configure(yscrollcommand=ys.set)
        ys.pack(side="right", fill="y")
        self.obj_tree.pack(fill="both", expand=True)
        self.obj_tree.bind("<<TreeviewSelect>>", self.on_object_selected)
        self.sort_state = ('label', False)
        pw.add(table_frame, weight=1)
        bottom = ttk.Frame(pw)
        self.object_info = ttk.Label(bottom, text="", font=("Consolas", 10), justify="left", anchor="nw", width=30)
        self.object_info.pack(side="right", fill="y", padx=8, pady=8)
        self.object_panel = PlotPanel(bottom)
        self.object_panel.pack(side="left", fill="both", expand=True)
        pw.add(bottom, weight=2)

        # Vérité terrain
        tab = ttk.Frame(self.right_notebook)
        self.right_notebook.add(tab, text="Vérité terrain")
        self.truth_tab = tab
        pw = ttk.PanedWindow(tab, orient=tk.HORIZONTAL)
        pw.pack(fill="both", expand=True)
        frame = ttk.Frame(pw)
        self.truth_tree = ttk.Treeview(frame, columns=('value',), show="tree headings", selectmode="none")
        self.truth_tree.heading('#0', text="Grandeur")
        self.truth_tree.heading('value', text="Valeur")
        self.truth_tree.column('#0', width=290)
        self.truth_tree.column('value', width=90, anchor="e")
        self.truth_tree.pack(fill="both", expand=True)
        pw.add(frame, weight=0)
        self.truth_panel = PlotPanel(pw)
        pw.add(self.truth_panel, weight=1)

        # Distributions
        tab = ttk.Frame(self.right_notebook)
        self.right_notebook.add(tab, text="Distributions")
        bar = ttk.Frame(tab)
        bar.pack(fill="x", padx=6, pady=4)
        feats = [c[0] for c in OBJECT_COLUMNS if c[0] != 'label'] + ['focus_mean', 'focus_sum', 'z_peak_um',
                                                                      'module_norm_mean']
        self.dist_x = tk.StringVar(value='nb_vox')
        self.dist_y = tk.StringVar(value='— histogramme —')
        self.dist_logx = tk.BooleanVar(value=True)
        self.dist_logy = tk.BooleanVar(value=False)
        ttk.Label(bar, text="X :").pack(side="left")
        cb = ttk.Combobox(bar, textvariable=self.dist_x, values=feats, state="readonly", width=20)
        cb.pack(side="left", padx=4)
        cb.bind("<<ComboboxSelected>>", lambda e: self.draw_distributions())
        ttk.Label(bar, text="Y :").pack(side="left", padx=(12, 0))
        cb = ttk.Combobox(bar, textvariable=self.dist_y, values=['— histogramme —'] + feats, state="readonly", width=20)
        cb.pack(side="left", padx=4)
        cb.bind("<<ComboboxSelected>>", lambda e: self.draw_distributions())
        ttk.Checkbutton(bar, text="log X", variable=self.dist_logx, command=self.draw_distributions).pack(side="left", padx=8)
        ttk.Checkbutton(bar, text="log Y", variable=self.dist_logy, command=self.draw_distributions).pack(side="left")
        self.dist_panel = PlotPanel(tab)
        self.dist_panel.pack(fill="both", expand=True)

        # Profils Z
        tab = ttk.Frame(self.right_notebook)
        self.right_notebook.add(tab, text="Profils Z")
        self.planes_panel = PlotPanel(tab)
        self.planes_panel.pack(fill="both", expand=True)

        # Statistiques
        tab = ttk.Frame(self.right_notebook)
        self.right_notebook.add(tab, text="Statistiques")
        self.stats_tab = tab
        bar = ttk.Frame(tab)
        bar.pack(fill="x", padx=6, pady=4)
        self.stats_title = ttk.Label(bar, text="", foreground="#9fb3c8")
        self.stats_title.pack(side="left")
        tb.Button(bar, text="📂 Ouvrir le dossier de résultats", bootstyle="secondary",
                  command=self.open_output_dir).pack(side="right")
        cols = ('value', 'compare', 'ratio')
        self.stats_tree = ttk.Treeview(tab, columns=cols, show="tree headings")
        self.stats_tree.heading('#0', text="Grandeur")
        self.stats_tree.heading('value', text="Analyse courante")
        self.stats_tree.heading('compare', text="Comparaison")
        self.stats_tree.heading('ratio', text="Courante / comparaison")
        self.stats_tree.column('#0', width=380)
        for c in cols:
            self.stats_tree.column(c, width=170, anchor="e")
        ys = ttk.Scrollbar(tab, orient="vertical", command=self.stats_tree.yview)
        self.stats_tree.configure(yscrollcommand=ys.set)
        ys.pack(side="right", fill="y")
        self.stats_tree.pack(fill="both", expand=True)

    # =========================================================================
    # ANALYSE (thread)
    # =========================================================================

    def run_analysis(self):
        if self.worker and self.worker.is_alive():
            return
        try:
            args = self.build_args()
        except ValueError as e:
            messagebox.showerror("Paramètres invalides", str(e))
            return
        self.save_params()
        self.log_text.delete("1.0", tk.END)
        self.left_notebook.select(self.log_tab)
        self.run_button.config(state="disabled")
        self.progress.start(12)
        self.status_var.set("Analyse en cours...")
        self.worker = threading.Thread(target=self._worker, args=(args,), daemon=True)
        self.worker.start()

    def _worker(self, args):
        try:
            result = hs.run_analysis(args, log=lambda m: self.msg_queue.put(('log', m)))
            self.msg_queue.put(('done', result))
        except Exception as e:
            self.msg_queue.put(('log', traceback.format_exc()))
            self.msg_queue.put(('error', f"{type(e).__name__} : {e}"))

    def poll_queue(self):
        try:
            while True:
                kind, payload = self.msg_queue.get_nowait()
                if kind == 'log':
                    self.log_text.insert(tk.END, payload + "\n")
                    self.log_text.see(tk.END)
                elif kind == 'done':
                    self.analysis_finished(payload)
                elif kind == 'error':
                    self.progress.stop()
                    self.run_button.config(state="normal")
                    self.status_var.set("Erreur : " + payload)
                    messagebox.showerror("Erreur pendant l'analyse", payload)
        except queue.Empty:
            pass
        self._poll_id = self.root.after(100, self.poll_queue)

    def analysis_finished(self, result):
        self.progress.stop()
        self.run_button.config(state="normal")
        self.result = result
        self.selected_object = None
        n = result['summary']['objects']['count']
        self.status_var.set(f"Terminé en {result['summary']['processing_time_s']:.1f} s — {n} objets — {result['out_dir']}")
        self.draw_all()
        self.right_notebook.select(1)

    def draw_all(self):
        self.draw_holograms()
        self.draw_volume()
        self.fill_object_table()
        self.draw_object()
        self.draw_distributions()
        self.draw_planes()
        self.draw_truth()
        self.fill_stats()

    # =========================================================================
    # DESSINS
    # =========================================================================

    def draw_holograms(self):
        r = self.result

        def plot(fig):
            axes = fig.subplots(2, 3)
            for ax, key, title in zip(axes[0], ('raw', 'reference', 'cleaned'), ("brut", "référence", "nettoyé")):
                img = r[key]
                lo, hi = percentile_limits(img)
                im = ax.imshow(img, cmap='gray', vmin=lo, vmax=hi)
                ax.set_title(f"Hologramme {title}")
                fig.colorbar(im, ax=ax, fraction=0.046)
            for ax, key, title in zip(axes[1, :2], ('raw', 'cleaned'), ("brut", "nettoyé")):
                ax.hist(np.asarray(r[key]).ravel(), bins=256, log=True, color='#5dade2')
                ax.set_title(f"Histogramme {title}")
            ax = axes[1, 2]
            f = r['spectra']['freq']
            for key, label in (('raw', 'brut'), ('reference', 'référence'), ('cleaned', 'nettoyé')):
                psd = r['spectra'][key]
                ax.semilogy(f[1:], psd[1:] / psd[1:].max(), label=label)
            ax.set_xlabel("fréquence (cycles/µm)")
            ax.set_title("Spectre radial (normalisé)")
            ax.legend()
        self.holo_panel.draw(plot)

    def draw_volume(self):
        r = self.result
        if r is None:
            return
        mips = r['mip_focus'] if self.volume_kind.get() == 'focus' else r['mip_module']
        obj, bins = r['objects'], r['bin_proj']
        dx, dz = r['dx_um'], r['dz_um']
        nz, ny, nx = mips[1].shape[0], mips[0].shape[0], mips[0].shape[1]
        z0 = r['z_um'][0]

        def plot(fig):
            gs = fig.add_gridspec(2, 2, width_ratios=[nx * dx, nz * dz], height_ratios=[ny * dx, nz * dz])
            ax_xy = fig.add_subplot(gs[0, 0])
            ax_yz = fig.add_subplot(gs[0, 1], sharey=ax_xy)
            ax_xz = fig.add_subplot(gs[1, 0], sharex=ax_xy)
            ext_xy = [0, nx * dx, ny * dx, 0]
            ext_xz = [0, nx * dx, z0 + nz * dz, z0]
            ext_yz = [z0, z0 + nz * dz, ny * dx, 0]
            for ax, img, ext in ((ax_xy, mips[0], ext_xy), (ax_xz, mips[1], ext_xz), (ax_yz, mips[2].T, ext_yz)):
                lo, hi = percentile_limits(img, 1, 99.9)
                ax.imshow(img, cmap='gray', vmin=lo, vmax=hi, extent=ext, aspect='auto', interpolation='nearest')
            if self.show_detections.get():
                # contour() avec origin='upper' : extent donné (gauche, droite, bas, haut) comme pour imshow
                for ax, b, ext in ((ax_xy, bins[0], ext_xy), (ax_xz, bins[1], ext_xz), (ax_yz, bins[2].T, ext_yz)):
                    if b.any():
                        ax.contour(b.astype(np.float32), levels=[0.5], colors='r', linewidths=0.6,
                                   extent=ext, origin='upper')
                if obj:
                    ax_xy.plot(obj['bary_x_um'], obj['bary_y_um'], 'c+', ms=5)
                    ax_xz.plot(obj['bary_x_um'], obj['bary_z_um'], 'c+', ms=5)
                    ax_yz.plot(obj['bary_z_um'], obj['bary_y_um'], 'c+', ms=5)
            truth, match = r.get('truth'), r.get('match')
            if truth is not None and self.show_truth.get():
                for sel, style in self._truth_styles(match):
                    ax_xy.plot(truth['x_um'][sel], truth['y_um'][sel], **style)
                    ax_xz.plot(truth['x_um'][sel], truth['z_um'][sel], **style)
                    ax_yz.plot(truth['z_um'][sel], truth['y_um'][sel], **style)
                ax_xy.legend(loc='upper right', fontsize=8, framealpha=0.6)
            if obj and self.selected_object is not None:
                i = self.selected_object
                ax_xy.plot(obj['bary_x_um'][i], obj['bary_y_um'][i], 'o', mfc='none', mec='yellow', ms=14, mew=1.5)
                ax_xz.plot(obj['bary_x_um'][i], obj['bary_z_um'][i], 'o', mfc='none', mec='yellow', ms=14, mew=1.5)
                ax_yz.plot(obj['bary_z_um'][i], obj['bary_y_um'][i], 'o', mfc='none', mec='yellow', ms=14, mew=1.5)
            ax_xy.set_xlim(0, nx * dx)
            ax_xy.set_ylim(ny * dx, 0)
            ax_xz.set_ylim(z0 + nz * dz, z0)
            ax_yz.set_xlim(z0, z0 + nz * dz)
            kind = "focus" if self.volume_kind.get() == 'focus' else "module propagé"
            ax_xy.set_title(f"Projection max. XY — {kind}")
            ax_xy.set_ylabel("y (µm)")
            ax_xz.set_xlabel("x (µm)")
            ax_xz.set_ylabel("z (µm)")
            ax_yz.set_xlabel("z (µm)")
            ax_yz.tick_params(labelleft=False)
            ax_xy.tick_params(labelbottom=False)
            self._ax_xy = ax_xy
        self.volume_panel.draw(plot)

    @staticmethod
    def _truth_styles(match):
        """Objets simulés : appariés (vert), manqués (rouge), hors du volume reconstruit (gris)."""
        found = match['truth_match'] >= 0
        inside = match['in_volume']
        return [(found, dict(ls='none', marker='o', mfc='none', mec='#2ecc71', ms=9, mew=1.2, label="simulé, trouvé")),
                (~found & inside, dict(ls='none', marker='x', color='#e74c3c', ms=8, mew=1.5, label="simulé, manqué")),
                (~found & ~inside, dict(ls='none', marker='.', color='#95a5a6', ms=5, label="simulé, hors volume"))]

    def on_volume_click(self, event):
        r = self.result
        if (r is None or not r['objects'] or event.inaxes is not getattr(self, '_ax_xy', None)
                or self.volume_panel.toolbar.mode or event.xdata is None):
            return
        obj = r['objects']
        d = np.hypot(obj['bary_x_um'] - event.xdata, obj['bary_y_um'] - event.ydata)
        self.select_object(int(np.argmin(d)))

    # ---- objets --------------------------------------------------------------

    def fill_object_table(self):
        self.obj_tree.delete(*self.obj_tree.get_children())
        obj = self.result['objects'] if self.result else None
        if not obj:
            return
        key, reverse = self.sort_state
        truth_text = [self._truth_text(i) for i in range(obj['label'].size)]
        order = (np.argsort(np.array(truth_text), kind='stable') if key == 'truth'
                 else np.argsort(obj[key], kind='stable'))
        if reverse:
            order = order[::-1]
        for i in order:
            values = [fmt.format(obj[k][i]) if np.isfinite(obj[k][i]) else "-" for k, _, _, fmt in OBJECT_COLUMNS]
            self.obj_tree.insert("", "end", iid=str(i), values=values + [truth_text[i]])

    def _truth_text(self, i):
        match = self.result.get('match') if self.result else None
        if not match:
            return ""
        status, t = match['det_status'][i], match['det_match'][i]
        return f"{status} #{t}" if t >= 0 else status

    def sort_objects(self, key):
        k, reverse = self.sort_state
        self.sort_state = (key, not reverse if k == key else False)
        self.fill_object_table()
        if self.selected_object is not None:
            self.obj_tree.selection_set(str(self.selected_object))
            self.obj_tree.see(str(self.selected_object))

    def on_object_selected(self, _event=None):
        sel = self.obj_tree.selection()
        if sel and int(sel[0]) != self.selected_object:
            self.selected_object = int(sel[0])
            self.draw_object()
            self.draw_volume()
            self.draw_planes()

    def select_object(self, i):
        self.selected_object = i
        self.obj_tree.selection_set(str(i))
        self.obj_tree.see(str(i))
        self.draw_object()
        self.draw_volume()
        self.draw_planes()

    def draw_object(self):
        r = self.result
        if r is None:
            return
        obj, crops, i = r['objects'], r['crops'], self.selected_object

        def plot(fig):
            if not obj or i is None:
                self.object_info.config(text="")
                ax = fig.add_subplot(111)
                ax.axis('off')
                ax.text(0.5, 0.5, "Sélectionner un objet dans le tableau\nou d'un clic dans l'onglet Volume",
                        ha='center', va='center', color='#9fb3c8')
                return
            dx, dz = r['dx_um'], r['dz_um']
            axes = fig.subplots(1, 3)
            # contexte : projection XY du focus autour de l'objet
            mip = r['mip_focus'][0]
            cx, cy = obj['bary_x_um'][i] / dx, obj['bary_y_um'][i] / dx
            h = 128
            x0, y0 = int(max(cx - h, 0)), int(max(cy - h, 0))
            sub = mip[y0:int(cy + h), x0:int(cx + h)]
            lo, hi = percentile_limits(mip, 1, 99.9)
            axes[0].imshow(sub, cmap='gray', vmin=lo, vmax=hi,
                           extent=[x0 * dx, (x0 + sub.shape[1]) * dx, (y0 + sub.shape[0]) * dx, y0 * dx])
            axes[0].plot(obj['bary_x_um'][i], obj['bary_y_um'][i], 'o', mfc='none', mec='yellow', ms=14)
            axes[0].set_title("Contexte (proj. max.)")
            axes[0].set_xlabel("µm")
            if crops is not None and i < crops['xy'].shape[0]:
                hxy, hz = crops['half_xy'], crops['half_z']
                c_xy, c_xz = crops['xy'][i], crops['xz'][i]
                axes[1].imshow(c_xy, cmap='magma', extent=[-hxy * dx, hxy * dx, hxy * dx, -hxy * dx])
                axes[1].set_title(f"Focus XY (z pic = {obj['z_peak_um'][i]:.1f} µm)")
                axes[2].imshow(c_xz, cmap='magma', extent=[-hxy * dx, hxy * dx, hz * dz, -hz * dz], aspect='auto')
                axes[2].set_title("Focus XZ (z relatif)")
                for ax in axes[1:3]:
                    ax.axhline(0, color='c', lw=0.5, alpha=0.6)
                    ax.axvline(0, color='c', lw=0.5, alpha=0.6)
                    ax.set_xlabel("µm")
            match, truth = r.get('match'), r.get('truth')
            lines = [f"Objet #{obj['label'][i]}", ""] + [
                f"{title:12s} {fmt.format(obj[k][i])}" for k, title, _, fmt in OBJECT_COLUMNS if k != 'label'] + [
                f"{'z pic':12s} {obj['z_peak_um'][i]:.2f} µm",
                f"{'focus moyen':12s} {obj['focus_mean'][i]:.4g}",
                f"{'focus somme':12s} {obj['focus_sum'][i]:.4g}"]
            if 'module_norm_mean' in obj:
                lines.append(f"{'module moyen':12s} {obj['module_norm_mean'][i]:.1f} σ")
            if match:
                t = match['det_match'][i]
                lines += ["", f"vérité : {self._truth_text(i)}"]
                if t >= 0:
                    lines += [f"  simulé x {truth['x_um'][t]:.2f}", f"         y {truth['y_um'][t]:.2f}",
                              f"         z {truth['z_um'][t]:.2f}",
                              f"  erreur z {obj['bary_z_um'][i] - truth['z_um'][t]:+.2f} µm"]
                    if 'phi_angle' in truth:
                        lines.append(f"  φ {truth['phi_angle'][t]:.0f}°  θ {truth['theta_angle'][t]:.0f}°")
            self.object_info.config(text="\n".join(lines))
        self.object_panel.draw(plot)

    # ---- vérité terrain ------------------------------------------------------

    def draw_truth(self):
        self.truth_tree.delete(*self.truth_tree.get_children())
        r = self.result
        truth, match = (r.get('truth'), r.get('match')) if r else (None, None)
        if match:
            m = match['metrics']
            for key, label in TRUTH_LABELS:
                if m.get(key) is not None:
                    v = m[key]
                    self.truth_tree.insert("", "end", text=label,
                                           values=(f"{v:d}" if isinstance(v, (int, np.integer)) else f"{v:.3g}",))

        def plot(fig):
            if not match:
                ax = fig.add_subplot(111)
                ax.axis('off')
                ax.text(0.5, 0.5, "Pas de vérité terrain.\nCharger une simulation (onglet Simulation à gauche)\n"
                                  "et cocher « Comparer aux positions simulées ».",
                        ha='center', va='center', color=INFO_COLOR)
                return
            obj = r['objects']
            dx, z = r['dx_um'], r['z_um']
            ny, nx = r['mip_focus'][0].shape
            axes = fig.subplots(2, 3)
            status = match['det_status']
            det_styles = [('apparié', dict(ls='none', marker='+', color='#5dade2', ms=7, label="détecté, apparié")),
                          ('doublon', dict(ls='none', marker='+', color='#bb8fce', ms=7, label="détecté, doublon")),
                          ('faux positif', dict(ls='none', marker='s', mfc='none', mec='#f39c12', ms=6,
                                                label="détecté, faux positif"))]
            for ax, (a, b), (ea, eb) in ((axes[0, 0], ('x_um', 'y_um'), ('bary_x_um', 'bary_y_um')),
                                         (axes[0, 1], ('x_um', 'z_um'), ('bary_x_um', 'bary_z_um'))):
                bg = r['mip_focus'][0] if b == 'y_um' else r['mip_focus'][1]
                lo, hi = percentile_limits(bg, 1, 99.9)
                ext = [0, nx * dx, ny * dx, 0] if b == 'y_um' else [0, nx * dx, z[-1] + r['dz_um'], z[0]]
                ax.imshow(bg, cmap='gray', vmin=lo, vmax=hi, extent=ext, aspect='auto', alpha=0.5)
                for sel, style in self._truth_styles(match):
                    ax.plot(truth[a][sel], truth[b][sel], **style)
                if obj:
                    for st, style in det_styles:
                        sel = status == st
                        ax.plot(obj[ea][sel], obj[eb][sel], **style)
                ax.set_xlabel("x (µm)")
                ax.set_ylabel("y (µm)" if b == 'y_um' else "z (µm)")
            axes[0, 0].set_title("Vue XY")
            axes[0, 1].set_title("Vue XZ")
            axes[0, 0].legend(fontsize=7, loc='upper right', framealpha=0.6)

            ax = axes[0, 2]
            found = match['truth_match'] >= 0
            if found.any():
                i_det, i_tru = match['truth_match'][found], np.flatnonzero(found)
                errs = {'x': obj['bary_x_um'][i_det] - truth['x_um'][i_tru],
                        'y': obj['bary_y_um'][i_det] - truth['y_um'][i_tru],
                        'z': obj['bary_z_um'][i_det] - truth['z_um'][i_tru]}
                lim = max(1e-3, max(np.abs(e).max() for e in errs.values()))
                bins = np.linspace(-lim, lim, 31)
                for k, c in (('x', '#5dade2'), ('y', '#58d68d'), ('z', '#f5b041')):
                    ax.hist(errs[k], bins=bins, histtype='step', lw=1.5, color=c, label=f"{k} (moy. {errs[k].mean():+.2f})")
                ax.legend(fontsize=8)
            ax.set_title("Erreur de position des appariés")
            ax.set_xlabel("détecté − simulé (µm)")

            inside = match['in_volume']

            def recall_plot(ax, values, edges, xlabel):
                idx = np.digitize(values, edges) - 1
                centers, rec, cnt = [], [], []
                for k in range(len(edges) - 1):
                    sel = inside & (idx == k)
                    if sel.any():
                        centers.append(0.5 * (edges[k] + edges[k + 1]))
                        rec.append(found[sel].mean())
                        cnt.append(sel.sum())
                ax.bar(centers, rec, width=0.8 * np.diff(edges).min(), color='#5dade2')
                for xc, rc, n in zip(centers, rec, cnt):
                    ax.text(xc, rc + 0.02, str(n), ha='center', fontsize=7, color='white')
                ax.set_ylim(0, 1.1)
                ax.set_xlabel(xlabel)
                ax.set_ylabel("rappel")

            recall_plot(axes[1, 0], truth['z_um'], np.linspace(z[0], z[-1] + r['dz_um'], 11), "z simulé (µm)")
            axes[1, 0].set_title("Rappel selon z (nombre d'objets au-dessus)")
            if 'phi_angle' in truth:
                recall_plot(axes[1, 1], truth['phi_angle'], np.arange(0, 91, 15), "φ simulé (°), 0 = axe optique")
                axes[1, 1].set_title("Rappel selon l'orientation")
                ax = axes[1, 2]
                if found.any():
                    sc = ax.scatter(truth['phi_angle'][i_tru], errs['z'], c=truth['z_um'][i_tru], cmap='viridis', s=16)
                    fig.colorbar(sc, ax=ax, label="z (µm)")
                ax.set_xlabel("φ simulé (°)")
                ax.set_ylabel("erreur z (µm)")
                ax.set_title("Erreur z selon l'orientation")
            elif 'radius' in truth:
                recall_plot(axes[1, 1], 1e6 * truth['radius'], np.linspace(0, 1e6 * truth['radius'].max() * 1.01, 7),
                            "rayon simulé (µm)")
                axes[1, 1].set_title("Rappel selon le rayon")
                axes[1, 2].axis('off')
            else:
                axes[1, 1].axis('off')
                axes[1, 2].axis('off')
        self.truth_panel.draw(plot)

    # ---- distributions -------------------------------------------------------

    def draw_distributions(self):
        r = self.result
        if r is None:
            return
        obj = r['objects']
        kx, ky = self.dist_x.get(), self.dist_y.get()
        logx, logy = self.dist_logx.get(), self.dist_logy.get()

        def plot(fig):
            ax = fig.add_subplot(111)
            if not obj:
                ax.text(0.5, 0.5, "Aucun objet", ha='center', va='center')
                return
            x = np.asarray(obj[kx], dtype=np.float64)
            if ky == '— histogramme —':
                v = x[np.isfinite(x) & ((x > 0) if logx else True)]
                if v.size:
                    bins = np.logspace(np.log10(v.min()), np.log10(v.max()) + 1e-9, 40) if logx else 40
                    ax.hist(v, bins=bins, color='#5dade2', edgecolor='#1f2d3a')
                ax.set_xlabel(kx)
                ax.set_ylabel("nombre d'objets")
                ax.set_title(f"Distribution de {kx} ({v.size} objets)")
            else:
                y = np.asarray(obj[ky], dtype=np.float64)
                sc = ax.scatter(x, y, c=obj['bary_z_um'], cmap='viridis', s=18)
                fig.colorbar(sc, ax=ax, label="z (µm)")
                if self.selected_object is not None:
                    ax.plot(x[self.selected_object], y[self.selected_object], 'o', mfc='none', mec='yellow', ms=12)
                ax.set_xlabel(kx)
                ax.set_ylabel(ky)
                ax.set_title(f"{ky} en fonction de {kx}")
                if logy:
                    ax.set_yscale('log')
            if logx:
                ax.set_xscale('log')
            if logy and ky == '— histogramme —':
                ax.set_yscale('log')
            ax.grid(alpha=0.2)
        self.dist_panel.draw(plot)

    # ---- profils Z -----------------------------------------------------------

    def draw_planes(self):
        """
        Profils axiaux de l'objet sélectionné : module propagé et focus le long de z, au maximum
        d'une fenêtre de 5 × 5 pixels centrée sur son barycentre XY. En pointillés, le fond du
        volume au même plan (médiane et 99e percentile de tous les pixels du plan).
        En bas : histogrammes du volume entier.
        """
        r = self.result
        if r is None:
            return
        obj, prof, i = r['objects'], r.get('profiles'), self.selected_object
        z = r['z_um']
        norm = r['summary']['volume'].get('module_normalized') or {}
        n_mean = norm.get('normalization', {}).get('volume_mean')
        n_std = norm.get('normalization', {}).get('volume_std')
        eps = norm.get('normalization', {}).get('epsilon', 0.0)

        def plot(fig):
            axes = fig.subplots(2, 2)
            if not obj or prof is None or i is None:
                for ax in axes[0]:
                    ax.axis('off')
                axes[0, 0].text(1.1, 0.5, "Sélectionner un objet (onglet Objets, ou clic sur la vue XY de l'onglet Volume)\n"
                                          "pour afficher son profil le long de z.",
                                ha='center', va='center', color=INFO_COLOR, transform=axes[0, 0].transAxes)
            else:
                win = 2 * prof['half_window_px'] + 1
                for ax, key, planes, title in ((axes[0, 0], 'module', r['planes_module'], "module propagé"),
                                               (axes[0, 1], 'focus', r['planes_focus'], "focus")):
                    ax.plot(z, prof[key][i], color='#f5b041', lw=1.8, label=f"objet #{obj['label'][i]}")
                    ax.plot(z, planes['median'], color='#aab7c4', ls='--', lw=1, label="fond : médiane du plan")
                    ax.plot(z, planes['p99'], color='#aab7c4', ls=':', lw=1, label="fond : p99 du plan")
                    ax.axvline(obj['bary_z_um'][i], color='#5dade2', lw=1, label="barycentre z")
                    ax.axvline(obj['z_peak_um'][i], color='#f4d03f', lw=1, ls='--', label="plan du pic de focus")
                    match, truth = r.get('match'), r.get('truth')
                    if match is not None and match['det_match'][i] >= 0:
                        ax.axvline(truth['z_um'][match['det_match'][i]], color='#2ecc71', lw=1.2, ls='-.',
                                   label="z simulé (vérité)")
                    ax.set_xlabel("z (µm)")
                    ax.set_title(f"Objet #{obj['label'][i]} — {title} le long de z (max. sur {win}×{win} px)")
                axes[0, 1].axhline(r['threshold'], color='r', ls='--', lw=1, label="seuil de détection")
                axes[0, 1].fill_between(z, r['threshold'], prof['focus'][i], where=prof['focus'][i] > r['threshold'],
                                        color='r', alpha=0.15)
                axes[0, 1].set_yscale('log')
                if n_mean is not None:
                    f = lambda v: (v - n_mean) / (n_std + eps)
                    finv = lambda u: u * (n_std + eps) + n_mean
                    sec = axes[0, 0].secondary_yaxis('right', functions=(f, finv))
                    sec.set_ylabel("module normalisé (σ)")
                for ax in axes[0]:
                    ax.legend(fontsize=7)
            for ax, (counts, edges), title in ((axes[1, 0], r['hist_module'], "Module"),
                                               (axes[1, 1], r['hist_focus'], "Focus")):
                ax.stairs(counts, edges, fill=True, color='#5dade2')
                ax.set_yscale('log')
                ax.set_title(f"Histogramme {title}, volume entier (sous-échantillon)")
            axes[1, 1].axvline(r['threshold'], color='r', ls='--')
            if obj and i is not None and prof is not None:
                axes[1, 0].axvline(prof['module'][i].max(), color='#f5b041', lw=1.2)
                axes[1, 1].axvline(prof['focus'][i].max(), color='#f5b041', lw=1.2)
        self.planes_panel.draw(plot)

    # ---- statistiques --------------------------------------------------------

    STAT_ROWS = [
        ("Hologramme brut", 'hologram.raw', ['mean', 'std', 'min', 'max', 'median', 'contrast_std_over_mean',
                                             'noise_std_laplacian', 'noise_std_over_mean', 'envelope_variation_over_mean',
                                             'structure_std_without_envelope', 'distinct_levels', 'saturation_low_fraction',
                                             'saturation_high_fraction', 'mean_ratio_over_reference',
                                             'correlation_with_reference', 'psd_high_freq_plateau']),
        ("Référence", 'hologram.reference', ['mean', 'std', 'min', 'max', 'contrast_std_over_mean',
                                             'noise_std_laplacian', 'noise_std_over_mean', 'envelope_variation_over_mean']),
        ("Hologramme nettoyé", 'hologram.cleaned', ['mean', 'std', 'min', 'max', 'median', 'robust_std',
                                                    'contrast_std_over_mean', 'noise_std_laplacian', 'noise_std_over_mean',
                                                    'envelope_variation_over_mean', 'structure_std_without_envelope',
                                                    'p1', 'p99', 'psd_high_freq_plateau']),
        ("Volume — module propagé", 'volume.module', ['mean_gpu_full', 'std_gpu_full', 'max_gpu_full', 'median',
                                                      'robust_std', 'p99', 'p99.9']),
        ("Volume — module propagé normalisé : (V − moyenne) / (écart-type + ε)", 'volume.module_normalized',
         ['min', 'max', 'mean', 'std', 'median', 'robust_std', 'p1', 'p99', 'p99.9',
          'fraction_above_3sigma', 'fraction_above_5sigma', 'fraction_above_10sigma']),
        ("Volume — module normalisé, voxels des objets détectés", 'volume.module_normalized_objects',
         ['nb_voxels', 'min', 'max', 'mean', 'std', 'median', 'robust_std', 'p5', 'p95',
          'fraction_above_3sigma', 'fraction_above_5sigma', 'fraction_above_10sigma']),
        ("Volume — focus", 'volume.focus', ['mean_gpu_full', 'std_gpu_full', 'max_gpu_full', 'median', 'robust_std',
                                            'p99', 'p99.9', 'threshold', 'threshold_in_robust_sigma_above_median',
                                            'voxels_above_threshold', 'fraction_above_threshold']),
        ("Objets", 'objects', ['count', 'count_before_voxel_filter', 'density_per_mm3']),
        ("Vérité terrain", 'ground_truth', ['nb_simulated_in_volume', 'true_positives', 'false_negatives',
                                            'false_positives', 'duplicates', 'recall', 'precision',
                                            'precision_strict', 'f1', 'error_x_mean_um', 'error_y_mean_um',
                                            'error_z_mean_um', 'error_xy_rms_um', 'error_z_rms_um',
                                            'recall_phi_0_30', 'recall_phi_30_60', 'recall_phi_60_90']),
    ]
    OBJECT_STAT_KEYS = ['nb_vox', 'focus_peak', 'focus_mean', 'focus_sum', 'peak_snr_robust', 'module_norm_peak',
                        'module_norm_mean', 'extent_x_um', 'extent_y_um', 'extent_z_um', 'bary_z_um',
                        'nearest_neighbor_um']

    @staticmethod
    def _get(summary, path, key):
        node = summary
        for p in path.split('.'):
            node = node.get(p, {}) if isinstance(node, dict) else {}
        return node.get(key) if isinstance(node, dict) else None

    @staticmethod
    def _fmt(v):
        if v is None:
            return "-"
        if isinstance(v, (int, np.integer)):
            return f"{v:d}"
        return f"{v:.5g}"

    def _stat_row(self, parent, label, a, b):
        ratio = "-"
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and b not in (0, None):
            ratio = f"{a / b:.3g}"
        self.stats_tree.insert(parent, "end", text=label, values=(self._fmt(a), self._fmt(b), ratio))

    def fill_stats(self):
        self.stats_tree.delete(*self.stats_tree.get_children())
        cur = self.result['summary'] if self.result else None
        cmp_ = self.compare_summary
        if cur is None and cmp_ is None:
            return
        cur, cmp_ = cur or {}, cmp_ or {}
        self.stats_title.config(text=f"Courante : {cur.get('inputs', {}).get('hologram', '-')}    "
                                     f"Comparaison : {self.compare_path or '-'}")
        for title, path, keys in self.STAT_ROWS:
            node = self.stats_tree.insert("", "end", text=title, open=True)
            for k in keys:
                a, b = self._get(cur, path, k), self._get(cmp_, path, k)
                if a is not None or b is not None:
                    self._stat_row(node, k, a, b)
        node = self.stats_tree.insert("", "end", text="Objets — médianes (p5 / p95)", open=True)
        for k in self.OBJECT_STAT_KEYS:
            a, b = self._get(cur, 'objects', k), self._get(cmp_, 'objects', k)
            if a or b:
                self._stat_row(node, f"{k} médiane", (a or {}).get('p50'), (b or {}).get('p50'))
                self._stat_row(node, f"{k} p5", (a or {}).get('p5'), (b or {}).get('p5'))
                self._stat_row(node, f"{k} p95", (a or {}).get('p95'), (b or {}).get('p95'))
        node = self.stats_tree.insert("", "end", text="Contrôle HOLOTRACKER (CCA)", open=False)
        for k in ('nb_objects_cca', 'total_voxels_cca', 'max_barycenter_diff_um'):
            self._stat_row(node, k, (cur.get('cca_check') or {}).get(k), (cmp_.get('cca_check') or {}).get(k))
        self._stat_row("", "Temps de calcul (s)", cur.get('processing_time_s'), cmp_.get('processing_time_s'))

    def load_comparison(self):
        path = filedialog.askopenfilename(title="summary.json à comparer", initialdir=REPO_DIR,
                                          filetypes=[("summary.json", "*.json"), ("Tous", "*.*")])
        if not path:
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                self.compare_summary = json.load(f)
        except Exception as e:
            messagebox.showerror("Comparaison", f"Lecture impossible :\n{e}")
            return
        self.compare_path = path
        self.compare_label.config(text=path)
        self.fill_stats()
        self.right_notebook.select(self.stats_tab)

    def clear_comparison(self):
        self.compare_summary, self.compare_path = None, None
        self.compare_label.config(text="aucune")
        self.fill_stats()

    def open_output_dir(self):
        if self.result and os.path.isdir(self.result['out_dir']):
            os.startfile(self.result['out_dir'])

    def on_closing(self):
        if self.worker and self.worker.is_alive():
            if not messagebox.askyesno("Analyse en cours", "Une analyse est en cours. Quitter quand même ?"):
                return
        self.save_params()
        self.root.after_cancel(self._poll_id)
        self.root.destroy()


def main():
    root = tb.Window(themename="superhero")
    HoloStatsApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

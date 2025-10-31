# -*- coding: utf-8 -*-

"""
Filename: simu_bact_gui.py

Description:
Interface graphique Tkinter pour configurer et lancer la simulation d'hologrammes de bactéries.
Chaque modification de paramètre met à jour parameters_simu_bact.json et lance la génération.

Author: Simon BECKER
Date: 2025-10-24

License:
GNU General Public License v3.0
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import os
import subprocess
import threading
import sys
import numpy as np
from PIL import Image, ImageTk
import tifffile
from scipy import ndimage

class ResultVisualizer:
    """Fenêtre de visualisation des résultats avec slider pour les plans Z"""
    def __init__(self, parent, result):
        self.parent = parent
        self.result = result
        self.current_z = 0
        
        # Chargement des données
        try:
            self.hologram = np.array(Image.open(result['hologram']))
            self.bin_volume = tifffile.imread(result['bin_volume'])
            self.intensity_volume = tifffile.imread(result['intensity_volume'])
            self.z_size = self.bin_volume.shape[2] if len(self.bin_volume.shape) == 3 else self.bin_volume.shape[0]
            
            # Debug : affiche les infos sur les volumes
            print(f"\n=== Informations de chargement ===")
            print(f"Hologramme shape: {self.hologram.shape}, dtype: {self.hologram.dtype}")
            print(f"Volume binaire shape: {self.bin_volume.shape}, dtype: {self.bin_volume.dtype}")
            print(f"  - Min: {self.bin_volume.min()}, Max: {self.bin_volume.max()}")
            print(f"  - Valeurs non-nulles: {np.count_nonzero(self.bin_volume)}/{self.bin_volume.size}")
            print(f"Volume intensité shape: {self.intensity_volume.shape}, dtype: {self.intensity_volume.dtype}")
            print(f"  - Min: {self.intensity_volume.min()}, Max: {self.intensity_volume.max()}")
            print(f"Z size: {self.z_size}")
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur de chargement des données :\n{e}")
            parent.destroy()
            return
        
        self.create_widgets()
        self.update_display()
    
    def create_widgets(self):
        """Crée l'interface de visualisation"""
        # Frame principal
        main_frame = ttk.Frame(self.parent, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Frame pour les 3 images
        images_frame = ttk.Frame(main_frame)
        images_frame.pack(fill=tk.BOTH, expand=True)
        
        # Colonne 1 : Hologramme simulé
        col1 = ttk.Frame(images_frame)
        col1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        ttk.Label(col1, text="Hologramme Simulé", font=('Arial', 12, 'bold')).pack()
        self.holo_label = ttk.Label(col1)
        self.holo_label.pack(pady=5)
        
        # Colonne 2 : Volume binaire
        col2 = ttk.Frame(images_frame)
        col2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        ttk.Label(col2, text="Segmentation (Binaire)", font=('Arial', 12, 'bold')).pack()
        self.bin_label = ttk.Label(col2)
        self.bin_label.pack(pady=5)
        
        # Colonne 3 : Volume intensité
        col3 = ttk.Frame(images_frame)
        col3.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        ttk.Label(col3, text="Volume Intensité", font=('Arial', 12, 'bold')).pack()
        self.intensity_label = ttk.Label(col3)
        self.intensity_label.pack(pady=5)
        
        # Frame pour le slider
        slider_frame = ttk.Frame(main_frame)
        slider_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(slider_frame, text="Plan Z :").pack(side=tk.LEFT, padx=5)
        
        self.z_slider = ttk.Scale(slider_frame, from_=0, to=self.z_size-1, 
                                  orient=tk.HORIZONTAL, command=self.on_slider_change)
        self.z_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.z_label = ttk.Label(slider_frame, text=f"0 / {self.z_size-1}")
        self.z_label.pack(side=tk.LEFT, padx=5)
        
        # Boutons de contrôle
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(control_frame, text="⏮ Début", command=lambda: self.set_z(0)).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="◀ Précédent", command=self.prev_z).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Suivant ▶", command=self.next_z).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Fin ⏭", command=lambda: self.set_z(self.z_size-1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="📂 Ouvrir dossier", command=self.open_folder).pack(side=tk.RIGHT, padx=5)
    
    def on_slider_change(self, value):
        """Callback du slider"""
        self.current_z = int(float(value))
        self.update_display()
    
    def set_z(self, z):
        """Définit le plan Z"""
        self.current_z = max(0, min(z, self.z_size - 1))
        self.z_slider.set(self.current_z)
        self.update_display()
    
    def prev_z(self):
        """Plan précédent"""
        self.set_z(self.current_z - 1)
    
    def next_z(self):
        """Plan suivant"""
        self.set_z(self.current_z + 1)
    
    def open_folder(self):
        """Ouvre le dossier de résultats"""
        import platform
        import subprocess
        folder = self.result['output_dir']
        if platform.system() == "Windows":
            os.startfile(folder)
        elif platform.system() == "Darwin":  # macOS
            subprocess.Popen(["open", folder])
        else:  # Linux
            subprocess.Popen(["xdg-open", folder])
    
    def update_display(self):
        """Met à jour l'affichage des images"""
        self.z_label.config(text=f"{self.current_z} / {self.z_size-1}")
        
        # Image hologramme (constante)
        holo_img = self.resize_image(self.hologram, max_size=400)
        holo_photo = ImageTk.PhotoImage(holo_img)
        self.holo_label.config(image=holo_photo)
        self.holo_label.image = holo_photo
        
        # Plan binaire
        if len(self.bin_volume.shape) == 3:
            bin_plane = self.bin_volume[:, :, self.current_z]
        else:
            bin_plane = self.bin_volume[self.current_z, :, :]
        
        # Debug du plan binaire
        print(f"\nPlan Z={self.current_z}")
        print(f"  Binaire - shape: {bin_plane.shape}, dtype: {bin_plane.dtype}")
        print(f"  Binaire - min: {bin_plane.min()}, max: {bin_plane.max()}, non-zero: {np.count_nonzero(bin_plane)}")
        
        # Le volume binaire est déjà en 0-255, pas besoin de conversion
        if bin_plane.dtype != np.uint8:
            bin_plane_uint8 = bin_plane.astype(np.uint8)
        else:
            bin_plane_uint8 = bin_plane
            
        bin_img = self.resize_image(bin_plane_uint8, max_size=400)
        bin_photo = ImageTk.PhotoImage(bin_img)
        self.bin_label.config(image=bin_photo)
        self.bin_label.image = bin_photo
        
        # Plan intensité
        if len(self.intensity_volume.shape) == 3:
            intensity_plane = self.intensity_volume[:, :, self.current_z]
        else:
            intensity_plane = self.intensity_volume[self.current_z, :, :]
        
        print(f"  Intensité - shape: {intensity_plane.shape}, dtype: {intensity_plane.dtype}")
        print(f"  Intensité - min: {intensity_plane.min()}, max: {intensity_plane.max()}")
        
        # Normalisation pour l'affichage
        intensity_min = intensity_plane.min()
        intensity_max = intensity_plane.max()
        if intensity_max > intensity_min:
            intensity_normalized = ((intensity_plane - intensity_min) / 
                                   (intensity_max - intensity_min) * 255).astype(np.uint8)
        else:
            intensity_normalized = np.zeros_like(intensity_plane, dtype=np.uint8)
        
        intensity_img = self.resize_image(intensity_normalized, max_size=400)
        intensity_photo = ImageTk.PhotoImage(intensity_img)
        self.intensity_label.config(image=intensity_photo)
        self.intensity_label.image = intensity_photo
    
    def resize_image(self, array, max_size=400):
        """Redimensionne une image pour l'affichage"""
        # Convertit en uint8
        if array.dtype != np.uint8:
            array = array.astype(np.uint8)
        
        img = Image.fromarray(array)
        
        # Redimensionne en gardant le ratio
        width, height = img.size
        if max(width, height) > max_size:
            ratio = max_size / max(width, height)
            new_size = (int(width * ratio), int(height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        return img


class SimuBactGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Simulation Bactéries - Configuration")
        self.root.geometry("900x950")
        
        # Chemin du fichier de configuration
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_file = os.path.join(script_dir, "parameters_simu_bact.json")
        self.processor_script = os.path.join(script_dir, "processor_simu_bact.py")
        
        # Paramètres par défaut
        self.default_params = {
            "output_base_path": os.path.join(os.path.dirname(script_dir), "simu_bact_random"),
            "number_of_bacteria": 200,
            "holo_size_xy": 1024,
            "border": 256,
            "upscale_factor": 2,
            "z_size": 200,
            "transmission_milieu": 1.0,
            "index_milieu": 1.33,
            "index_bacterie": 1.335,
            "longueur_min": 3.0e-6,
            "longueur_max": 4.0e-6,
            "epaisseur_min": 1.0e-6,
            "epaisseur_max": 2.0e-6,
            "pix_size": 5.5e-6,
            "grossissement": 40,
            "vox_size_z_total": 100e-6,
            "wavelength": 660e-9,
            "illumination_mean": 1.0,
            "ecart_type_min": 0.01,
            "ecart_type_max": 0.1
        }
        
        self.load_or_create_config()
        self.create_widgets()
        self.processing = False
        self.status_check_timer = None
        self.status_file = os.path.join(script_dir, "processing_status.json")
        self.result_file = os.path.join(script_dir, "processing_result.json")
        
    def load_or_create_config(self):
        """Charge la configuration existante ou crée une nouvelle"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    self.params = json.load(f)
                # Ajoute les paramètres manquants
                for key, value in self.default_params.items():
                    if key not in self.params:
                        self.params[key] = value
            except Exception as e:
                messagebox.showerror("Erreur", f"Erreur de lecture du fichier de config : {e}")
                self.params = self.default_params.copy()
        else:
            self.params = self.default_params.copy()
            self.save_config()
    
    def save_config(self):
        """Sauvegarde la configuration dans le fichier JSON"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.params, f, indent=4)
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur d'écriture du fichier de config : {e}")
    
    def create_widgets(self):
        """Crée l'interface graphique"""
        # Frame principal divisé en 2 colonnes
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Colonne gauche : Paramètres avec scrollbar
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Colonne droite : Visualisation hologramme
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        main_frame.columnconfigure(0, weight=0)  # Colonne gauche : largeur fixe
        main_frame.columnconfigure(1, weight=1)  # Colonne droite : s'étend
        main_frame.rowconfigure(0, weight=1)
        
        # Canvas et scrollbar pour les paramètres (colonne gauche)
        canvas = tk.Canvas(left_frame, bg='white', highlightthickness=0, width=550)
        scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", width=canvas.winfo_reqwidth())
        canvas.configure(yscrollcommand=scrollbar.set)
        
        row = 0
        
        # Titre
        title_label = ttk.Label(scrollable_frame, text="Simulation Hologrammes de Bactéries", 
                                font=('Arial', 16, 'bold'))
        title_label.grid(row=row, column=0, columnspan=3, pady=15)
        row += 1
        
        # Section Chemins
        self.add_section_title(scrollable_frame, row, "📁 CHEMINS")
        row += 1
        
        self.add_path_entry(scrollable_frame, row, "Dossier de sortie", "output_base_path")
        row += 1
        
        # Section Volume
        ttk.Separator(scrollable_frame, orient='horizontal').grid(row=row, column=0, columnspan=3, 
                                                                   sticky='ew', pady=5)
        row += 1
        self.add_section_title(scrollable_frame, row, "📦 VOLUME")
        row += 1
        
        self.add_entry(scrollable_frame, row, "Nombre de bactéries", "number_of_bacteria", int)
        row += 1
        self.add_entry(scrollable_frame, row, "Taille XY hologramme (pixels)", "holo_size_xy", int)
        row += 1
        self.add_entry(scrollable_frame, row, "Bordure (pixels)", "border", int)
        row += 1
        self.add_entry(scrollable_frame, row, "Facteur upscale", "upscale_factor", int)
        row += 1
        self.add_entry(scrollable_frame, row, "Nombre de plans Z", "z_size", int)
        row += 1
        
        # Section Optique
        ttk.Separator(scrollable_frame, orient='horizontal').grid(row=row, column=0, columnspan=3, 
                                                                   sticky='ew', pady=5)
        row += 1
        self.add_section_title(scrollable_frame, row, "🔬 OPTIQUE")
        row += 1
        
        self.add_entry(scrollable_frame, row, "Indice milieu", "index_milieu", float)
        row += 1
        self.add_entry(scrollable_frame, row, "Indice bactérie", "index_bacterie", float)
        row += 1
        self.add_entry(scrollable_frame, row, "Transmission milieu", "transmission_milieu", float)
        row += 1
        self.add_entry(scrollable_frame, row, "Longueur d'onde (m)", "wavelength", float)
        row += 1
        
        # Section Caméra
        ttk.Separator(scrollable_frame, orient='horizontal').grid(row=row, column=0, columnspan=3, 
                                                                   sticky='ew', pady=5)
        row += 1
        self.add_section_title(scrollable_frame, row, "📷 CAMÉRA")
        row += 1
        
        self.add_entry(scrollable_frame, row, "Taille pixel caméra (m)", "pix_size", float)
        row += 1
        self.add_entry(scrollable_frame, row, "Grossissement", "grossissement", int)
        row += 1
        self.add_entry(scrollable_frame, row, "Taille voxel Z totale (m)", "vox_size_z_total", float)
        row += 1
        
        # Section Bactéries
        ttk.Separator(scrollable_frame, orient='horizontal').grid(row=row, column=0, columnspan=3, 
                                                                   sticky='ew', pady=5)
        row += 1
        self.add_section_title(scrollable_frame, row, "🦠 BACTÉRIES")
        row += 1
        
        self.add_entry(scrollable_frame, row, "Longueur min (m)", "longueur_min", float)
        row += 1
        self.add_entry(scrollable_frame, row, "Longueur max (m)", "longueur_max", float)
        row += 1
        self.add_entry(scrollable_frame, row, "Épaisseur min (m)", "epaisseur_min", float)
        row += 1
        self.add_entry(scrollable_frame, row, "Épaisseur max (m)", "epaisseur_max", float)
        row += 1
        
        # Section Illumination
        ttk.Separator(scrollable_frame, orient='horizontal').grid(row=row, column=0, columnspan=3, 
                                                                   sticky='ew', pady=5)
        row += 1
        self.add_section_title(scrollable_frame, row, "💡 ILLUMINATION")
        row += 1
        
        self.add_entry(scrollable_frame, row, "Moyenne illumination", "illumination_mean", float)
        row += 1
        self.add_entry(scrollable_frame, row, "Écart-type min", "ecart_type_min", float)
        row += 1
        self.add_entry(scrollable_frame, row, "Écart-type max", "ecart_type_max", float)
        row += 1
        
        # Bouton génération
        ttk.Separator(scrollable_frame, orient='horizontal').grid(row=row, column=0, columnspan=3, 
                                                                   sticky='ew', pady=8)
        row += 1
        
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.grid(row=row, column=0, columnspan=3, pady=5)
        
        self.generate_btn = ttk.Button(button_frame, text="🚀 Générer Hologramme", 
                                       command=self.generate_hologram, width=30)
        self.generate_btn.pack(pady=5)
        
        self.stop_btn = ttk.Button(button_frame, text="⏹ Arrêter", 
                                   command=self.stop_processing, width=30, state='disabled')
        self.stop_btn.pack(pady=5)
        
        row += 1
        
        # Status et barre de progression
        status_frame = ttk.Frame(scrollable_frame)
        status_frame.grid(row=row, column=0, columnspan=3, sticky='ew', padx=10, pady=10)
        row += 1
        
        self.status_label = ttk.Label(status_frame, text="✅ Prêt", foreground="green", 
                                      font=('Arial', 10, 'bold'))
        self.status_label.pack()
        
        self.progress_bar = ttk.Progressbar(status_frame, mode='determinate', length=400, maximum=100)
        self.progress_bar.pack(pady=5)
        
        canvas.pack(side="left", fill="y")
        scrollbar.pack(side="right", fill="y")
        
        # Panneau de droite : image hologramme
        # Titre
        title_holo = ttk.Label(right_frame, text="Hologramme Simulé", 
                               font=('Arial', 12, 'bold'), foreground='#2c3e50')
        title_holo.pack(anchor='n', pady=5)
        
        # Image
        self.holo_display_label = ttk.Label(right_frame, text="Aucun hologramme généré", 
                                            foreground="gray", font=('Arial', 10, 'italic'))
        self.holo_display_label.pack(anchor='center', expand=True)
        
        # Bind mouse wheel
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        
    def add_section_title(self, parent, row, text):
        """Ajoute un titre de section"""
        label = ttk.Label(parent, text=text, font=('Arial', 11, 'bold'), foreground='#2c3e50')
        label.grid(row=row, column=0, columnspan=3, sticky=tk.W, padx=5, pady=(5, 2))
        
    def add_entry(self, parent, row, label_text, param_key, param_type):
        """Ajoute un champ de saisie avec label"""
        label = ttk.Label(parent, text=label_text + " :", width=30, anchor='w')
        label.grid(row=row, column=0, sticky=tk.W, padx=10, pady=1)
        
        entry = ttk.Entry(parent, width=25)
        entry.insert(0, str(self.params[param_key]))
        entry.grid(row=row, column=1, padx=5, pady=1, sticky='w')
        
        # Validation et mise à jour au changement
        entry.bind('<Return>', lambda e: self.update_param(param_key, entry.get(), param_type))
        entry.bind('<FocusOut>', lambda e: self.update_param(param_key, entry.get(), param_type))
        
        # Affichage de l'unité ou info
        if param_type == float and 'e-' in str(self.params[param_key]):
            info = ttk.Label(parent, text=f"({self.params[param_key]:.2e})", foreground='gray')
            info.grid(row=row, column=2, sticky=tk.W, padx=5)
        
    def add_path_entry(self, parent, row, label_text, param_key):
        """Ajoute un champ de saisie pour un chemin avec bouton de navigation"""
        label = ttk.Label(parent, text=label_text + " :", width=30, anchor='w')
        label.grid(row=row, column=0, sticky=tk.W, padx=10, pady=1)
        
        entry = ttk.Entry(parent, width=40)
        entry.insert(0, self.params[param_key])
        entry.grid(row=row, column=1, padx=5, pady=1, sticky='ew')
        
        entry.bind('<Return>', lambda e: self.update_param(param_key, entry.get(), str))
        entry.bind('<FocusOut>', lambda e: self.update_param(param_key, entry.get(), str))
        
        btn = ttk.Button(parent, text="📁 Parcourir", 
                        command=lambda: self.browse_folder(entry, param_key))
        btn.grid(row=row, column=2, padx=5, pady=1)
        
    def browse_folder(self, entry, param_key):
        """Ouvre un dialogue de sélection de dossier"""
        folder = filedialog.askdirectory(initialdir=self.params.get(param_key, ""))
        if folder:
            entry.delete(0, tk.END)
            entry.insert(0, folder)
            self.update_param(param_key, folder, str)
    
    def update_param(self, key, value, value_type):
        """Met à jour un paramètre et sauvegarde la configuration"""
        try:
            self.params[key] = value_type(value)
            self.save_config()
            self.status_label.config(text=f"✅ Paramètre '{key}' mis à jour", foreground="green")
        except ValueError as e:
            messagebox.showerror("Erreur de saisie", 
                               f"Valeur invalide pour '{key}'.\nType attendu : {value_type.__name__}")
            self.status_label.config(text=f"❌ Erreur : {key}", foreground="red")
    
    def generate_hologram(self):
        """Lance la génération d'hologramme dans un thread séparé"""
        if self.processing:
            messagebox.showwarning("En cours", "Une génération est déjà en cours.")
            return
        
        # Vérification que le script processor existe
        if not os.path.exists(self.processor_script):
            messagebox.showerror("Erreur", 
                               f"Le script processor_simu_bact.py n'a pas été trouvé dans :\n{self.processor_script}")
            return
        
        # Nettoyage du fichier de statut
        if os.path.exists(self.status_file):
            try:
                os.remove(self.status_file)
            except Exception:
                pass
        
        # Lance le traitement dans un thread
        self.processing = True
        self.generate_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.status_label.config(text="⏳ Génération en cours...", foreground="orange")
        self.progress_bar['value'] = 0
        
        # Démarre le monitoring du statut
        self.check_processing_status()
        
        thread = threading.Thread(target=self.run_processor, daemon=True)
        thread.start()
    
    def check_processing_status(self):
        """Vérifie régulièrement le fichier de statut"""
        if not self.processing:
            return
        
        try:
            if os.path.exists(self.status_file):
                with open(self.status_file, 'r') as f:
                    status = json.load(f)
                
                # Met à jour l'interface
                message = status.get('message', 'Traitement en cours...')
                progress = status.get('progress', 0)
                
                self.status_label.config(text=f"⏳ {message}")
                self.progress_bar['value'] = progress
        except Exception:
            pass  # Ignore les erreurs de lecture
        
        # Replanifie la vérification dans 500ms
        if self.processing:
            self.status_check_timer = self.root.after(500, self.check_processing_status)
    
    def run_processor(self):
        """Exécute le script processor"""
        try:
            # Récupère le chemin de l'interpréteur Python actuel
            python_exe = sys.executable
            
            # Lance le script processor
            result = subprocess.run(
                [python_exe, self.processor_script],
                capture_output=True,
                text=True,
                cwd=os.path.dirname(self.processor_script)
            )
            
            # Affiche le résultat dans la console
            if result.stdout:
                print("=== STDOUT ===")
                print(result.stdout)
            if result.stderr:
                print("=== STDERR ===")
                print(result.stderr)
            
            # Mise à jour de l'interface
            if result.returncode == 0:
                self.root.after(0, lambda: self.on_processing_complete(success=True))
            else:
                self.root.after(0, lambda: self.on_processing_complete(
                    success=False, 
                    error=f"Code de retour : {result.returncode}\n{result.stderr}"
                ))
                
        except Exception as e:
            self.root.after(0, lambda: self.on_processing_complete(success=False, error=str(e)))
    
    def on_processing_complete(self, success=True, error=None):
        """Callback appelé à la fin du traitement"""
        self.processing = False
        self.generate_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        
        # Annule le timer de vérification
        if self.status_check_timer:
            self.root.after_cancel(self.status_check_timer)
            self.status_check_timer = None
        
        if success:
            self.status_label.config(text="✅ Hologramme généré avec succès !", foreground="green")
            self.progress_bar['value'] = 100
            
            # Charge les résultats et affiche l'hologramme dans le panneau droit
            if os.path.exists(self.result_file):
                try:
                    with open(self.result_file, 'r') as f:
                        result = json.load(f)
                    self.display_hologram(result)
                    self.open_visualizer(result)
                except Exception as e:
                    messagebox.showinfo("Succès", "L'hologramme a été généré avec succès !")
            else:
                messagebox.showinfo("Succès", "L'hologramme a été généré avec succès !")
        else:
            self.status_label.config(text="❌ Erreur lors de la génération", foreground="red")
            self.progress_bar['value'] = 0
            messagebox.showerror("Erreur", f"Erreur lors de la génération :\n\n{error}")
    
    def display_hologram(self, result):
        """Affiche l'hologramme dans le panneau de droite"""
        try:
            # Charge l'image hologramme
            holo_img = Image.open(result['hologram'])
            
            # Redimensionne pour s'adapter au panneau (max 600x600)
            holo_img.thumbnail((600, 600), Image.Resampling.LANCZOS)
            
            # Convertit pour Tkinter
            holo_photo = ImageTk.PhotoImage(holo_img)
            
            # Affiche dans le label
            self.holo_display_label.config(image=holo_photo, text="")
            self.holo_display_label.image = holo_photo  # Garde une référence
            
        except Exception as e:
            self.holo_display_label.config(text=f"Erreur d'affichage :\n{e}", foreground="red")
    
    def open_visualizer(self, result):
        """Ouvre une fenêtre de visualisation des résultats"""
        viz_window = tk.Toplevel(self.root)
        viz_window.title("Visualisation des Résultats")
        viz_window.geometry("1400x700")
        
        visualizer = ResultVisualizer(viz_window, result)
    
    def stop_processing(self):
        """Arrête le traitement en cours (non implémenté pour subprocess)"""
        messagebox.showinfo("Information", 
                          "L'arrêt du traitement n'est pas encore implémenté.\n" +
                          "Veuillez patienter jusqu'à la fin du traitement en cours.")


def main():
    root = tk.Tk()
    app = SimuBactGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

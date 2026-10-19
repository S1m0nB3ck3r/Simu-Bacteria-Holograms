# -*- coding: utf-8 -*-

"""
Filename: simulation_gui.py

Description:
Interface graphique pour la simulation d'hologrammes de bactéries.
Permet de configurer les paramètres et de choisir les formats de sauvegarde.

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
import time
import sys

class SimulationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Simulation Hologrammes Bactéries")
        self.root.geometry("600x900")
        
        # Chemin du fichier de configuration
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_file = os.path.join(script_dir, "parameters_simu_bact.json")
        self.processor_script = os.path.join(script_dir, "processor_simu_bact.py")
        
        # Paramètres par défaut
        self.default_params = {
            "output_base_path": os.path.join(os.path.dirname(script_dir), "simu_bact_random"),
            "number_of_holograms": 1,
            "number_of_bacteria": 200,
            "holo_size_xy": 1024,
            "border": 256,
            "upscale_factor": 2,
            "z_size": 200,
            "transmission_milieu": 1.0,
            "index_milieu": 1.33,
            "index_bacterie": 1.335,
            "longueur_min": 1.5e-6,
            "longueur_max": 2.0e-6,
            "epaisseur_min": 0.8e-6,
            "epaisseur_max": 0.8e-6,
            "pix_size": 5.5e-6,
            "grossissement": 40,
            "distance_volume_camera": 0.0,  # Distance volume-caméra en mètres, espace objet
            "step_z": 0.5e-6,
            "wavelength": 660e-9,
            "illumination_mean": 1.0,
            # Bruit capteur : écart-type relatif au fond, ajouté à l'intensité après propagation
            # (mesuré sur film réel : 0,015)
            "sensor_noise_std_min": 0.015,
            "sensor_noise_std_max": 0.015,
            # Options de sauvegarde
            "save_hologram_bmp": True,
            "save_hologram_tiff": False,
            "save_hologram_npy": False,
            "save_propagated_tiff": True,
            "save_propagated_npy": False,
            "save_segmentation_tiff": True,
            "save_segmentation_npy": False,
            "save_positions_csv": True,
            "save_holo_ref": False,
            "holo_ref_nb_planes": 1
        }
        
        self.load_or_create_config()
        self.create_widgets()
        self.processing = False
        self.status_check_timer = None
        self.status_file = os.path.join(script_dir, "processing_status.json")
        self.result_file = os.path.join(script_dir, "processing_result.json")
        self.stop_file = os.path.join(script_dir, "processing_stop.json")
        
        # Gestion de la fermeture de la fenêtre
        self.root.protocol("WM_DELETE_WINDOW", self.on_window_close)
        
    def load_or_create_config(self):
        """Charge la configuration existante ou crée une nouvelle"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    self.params = json.load(f)
                # Bruit d'illumination supprimé : anciennes clés retirées
                for old in ('ecart_type_min', 'ecart_type_max'):
                    self.params.pop(old, None)
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
        # Frame principal avec scrollbar
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Canvas et scrollbar
        canvas = tk.Canvas(main_frame, bg='white', highlightthickness=0, width=550)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
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
        
        # Section Simulation
        ttk.Separator(scrollable_frame, orient='horizontal').grid(row=row, column=0, columnspan=3, 
                                                                   sticky='ew', pady=5)
        row += 1
        self.add_section_title(scrollable_frame, row, "🔢 SIMULATION")
        row += 1
        
        self.add_entry(scrollable_frame, row, "Nombre d'hologrammes", "number_of_holograms", int)
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
        self.add_entry(scrollable_frame, row, "Apodisation bordure (pixels)", "border", int)
        row += 1
        self.add_entry(scrollable_frame, row, "Facteur upscale", "upscale_factor", int)
        row += 1
        self.add_entry(scrollable_frame, row, "Nombre de plans Z", "z_size", int)
        row += 1
        self.add_entry(scrollable_frame, row, "Pas de simulation (m)", "step_z", float)
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
        self.add_entry(scrollable_frame, row, "Distance volume-caméra (m)", "distance_volume_camera", float)
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
        self.add_entry(scrollable_frame, row, "Bruit capteur min (σ relatif)", "sensor_noise_std_min", float)
        row += 1
        self.add_entry(scrollable_frame, row, "Bruit capteur max (σ relatif)", "sensor_noise_std_max", float)
        row += 1
        
        # Section SAVING
        ttk.Separator(scrollable_frame, orient='horizontal').grid(row=row, column=0, columnspan=3, 
                                                                   sticky='ew', pady=5)
        row += 1
        self.add_section_title(scrollable_frame, row, "💾 SAUVEGARDE")
        row += 1
        
        # Checkboxes pour les options de sauvegarde
        self.add_checkbox(scrollable_frame, row, "Hologramme simulé 8bits BMP", "save_hologram_bmp")
        row += 1
        self.add_checkbox(scrollable_frame, row, "Hologramme simulé 32bits TIFF", "save_hologram_tiff")
        row += 1
        self.add_checkbox(scrollable_frame, row, "Hologramme simulé 32bits NPY", "save_hologram_npy")
        row += 1
        self.add_checkbox(scrollable_frame, row, "Volume propagé TIFF multistack", "save_propagated_tiff")
        row += 1
        self.add_checkbox(scrollable_frame, row, "Volume propagé NPY", "save_propagated_npy")
        row += 1
        self.add_checkbox(scrollable_frame, row, "Volume segmentation TIFF multistack", "save_segmentation_tiff")
        row += 1
        self.add_checkbox(scrollable_frame, row, "Volume segmentation bool NPY", "save_segmentation_npy")
        row += 1
        self.add_checkbox(scrollable_frame, row, "Positions bactéries CSV", "save_positions_csv")
        row += 1
        self.add_checkbox(scrollable_frame, row,
                          "Hologramme de référence sans objet → holo_ref/ (mêmes formats que l'hologramme)",
                          "save_holo_ref")
        row += 1
        self.add_entry(scrollable_frame, row, "Nb illuminations pour la référence moyennée", "holo_ref_nb_planes", int)
        row += 1
        
        # Bouton génération
        ttk.Separator(scrollable_frame, orient='horizontal').grid(row=row, column=0, columnspan=3, 
                                                                   sticky='ew', pady=8)
        row += 1
        
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.grid(row=row, column=0, columnspan=3, pady=5)
        
        self.generate_btn = ttk.Button(button_frame, text="🚀 Lancer la Simulation", 
                                       command=self.start_simulation, width=30)
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
    
    def add_checkbox(self, parent, row, label_text, param_key):
        """Ajoute une checkbox pour une option booléenne"""
        var = tk.BooleanVar(value=self.params.get(param_key, False))
        
        checkbox = ttk.Checkbutton(parent, text=label_text, variable=var,
                                   command=lambda: self.update_param(param_key, var.get(), bool))
        checkbox.grid(row=row, column=0, columnspan=3, sticky=tk.W, padx=30, pady=2)
        
    def update_param(self, key, value, param_type):
        """Met à jour un paramètre et sauvegarde"""
        try:
            if param_type == bool:
                self.params[key] = value
            else:
                self.params[key] = param_type(value)
            self.save_config()
        except ValueError:
            messagebox.showerror("Erreur", f"Valeur invalide pour {key}")
    
    def browse_folder(self, entry, param_key):
        """Ouvre un dialogue de sélection de dossier"""
        folder = filedialog.askdirectory(initialdir=self.params[param_key])
        if folder:
            entry.delete(0, tk.END)
            entry.insert(0, folder)
            self.update_param(param_key, folder, str)
    
    def start_simulation(self):
        """Lance la simulation"""
        self.save_config()
        
        # Nettoyage des fichiers de statut et d'arrêt
        for file_path in [self.status_file, self.stop_file]:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception:
                    pass
        
        # Lance le traitement dans un thread
        self.processing = True
        self.generate_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.status_label.config(text="⏳ Simulation en cours...", foreground="orange")
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
                
                # Vérifie si le traitement a été arrêté côté processeur
                if status.get('stopped', False):
                    self.on_processing_complete(success=False, error="Simulation interrompue par l'utilisateur")
                    return
                    
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
            self.status_label.config(text="✅ Simulation terminée avec succès !", foreground="green")
            self.progress_bar['value'] = 100
            messagebox.showinfo("Succès", "La simulation s'est terminée avec succès !")
        else:
            self.status_label.config(text="❌ Erreur lors de la simulation", foreground="red")
            self.progress_bar['value'] = 0
            messagebox.showerror("Erreur", f"Erreur lors de la simulation :\n\n{error}")
    
    def stop_processing(self):
        """Arrête le traitement en cours"""
        if not self.processing:
            return
            
        # Crée un fichier signal d'arrêt
        try:
            stop_signal = {
                "stop_requested": True,
                "timestamp": time.time(),
                "message": "Arrêt demandé par l'utilisateur"
            }
            with open(self.stop_file, 'w') as f:
                json.dump(stop_signal, f)
            
            # Met à jour l'interface
            self.status_label.config(text="⏹ Arrêt en cours...", foreground="red")
            self.stop_btn.config(state='disabled')
            
            print("Signal d'arrêt envoyé au processeur")
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de l'envoi du signal d'arrêt :\n{e}")
    
    def on_window_close(self):
        """Gère la fermeture de la fenêtre"""
        if self.processing:
            # Demande confirmation si une simulation est en cours
            if messagebox.askyesno("Confirmation", 
                                 "Une simulation est en cours.\n" +
                                 "Voulez-vous vraiment fermer l'application ?\n" +
                                 "(La simulation sera interrompue)"):
                self.stop_processing()
                # Attend un petit moment pour que l'arrêt soit pris en compte
                self.root.after(1000, self.force_close)
            return
        else:
            # Pas de simulation en cours, ferme normalement
            self.root.destroy()
    
    def force_close(self):
        """Force la fermeture après un délai"""
        self.processing = False
        if self.status_check_timer:
            self.root.after_cancel(self.status_check_timer)
        self.root.destroy()


def main():
    root = tk.Tk()
    app = SimulationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

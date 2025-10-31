# -*- coding: utf-8 -*-

"""
Filename: visualizer_gui.py

Description:
Interface graphique pour visualiser les résultats de simulation d'hologrammes.
Affiche côte à côte : hologramme, volume segmenté et volume propagé avec slider Z.

Author: Simon BECKER
Date: 2025-10-24

License:
GNU General Public License v3.0
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import numpy as np
from PIL import Image, ImageTk
import tifffile


class VisualizerGUI:
    """Interface de visualisation des résultats"""
    def __init__(self, root):
        self.root = root
        self.root.title("Visualisateur Hologrammes")
        self.root.geometry("1400x800")
        
        self.current_folder = None
        self.current_z = 0
        self.hologram = None
        self.bin_volume = None
        self.propagated_volume = None
        self.z_size = 0
        
        self.create_widgets()
    
    def create_widgets(self):
        """Crée l'interface graphique"""
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Frame de sélection du dossier
        select_frame = ttk.Frame(main_frame)
        select_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(select_frame, text="Dossier de résultats :", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=5)
        
        self.folder_entry = ttk.Entry(select_frame, width=60)
        self.folder_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        ttk.Button(select_frame, text="📁 Parcourir", command=self.browse_folder).pack(side=tk.LEFT, padx=5)
        ttk.Button(select_frame, text="🔄 Charger", command=self.load_data).pack(side=tk.LEFT, padx=5)
        
        # Frame pour les 3 images
        images_frame = ttk.Frame(main_frame)
        images_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Colonne 1 : Hologramme simulé
        col1 = ttk.Frame(images_frame)
        col1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        ttk.Label(col1, text="Hologramme Simulé", font=('Arial', 12, 'bold')).pack()
        self.holo_label = ttk.Label(col1, text="Aucune donnée chargée", foreground="gray")
        self.holo_label.pack(pady=5, expand=True)
        
        # Colonne 2 : Volume binaire
        col2 = ttk.Frame(images_frame)
        col2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        ttk.Label(col2, text="Segmentation (Binaire)", font=('Arial', 12, 'bold')).pack()
        self.bin_label = ttk.Label(col2, text="Aucune donnée chargée", foreground="gray")
        self.bin_label.pack(pady=5, expand=True)
        
        # Colonne 3 : Volume propagé avec surimpression
        col3 = ttk.Frame(images_frame)
        col3.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        ttk.Label(col3, text="Volume Propagé + Segmentation", font=('Arial', 12, 'bold')).pack()
        self.propagated_label = ttk.Label(col3, text="Aucune donnée chargée", foreground="gray")
        self.propagated_label.pack(pady=5, expand=True)
        
        # Frame pour le slider
        slider_frame = ttk.Frame(main_frame)
        slider_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(slider_frame, text="Plan Z :").pack(side=tk.LEFT, padx=5)
        
        self.z_slider = ttk.Scale(slider_frame, from_=0, to=0, 
                                  orient=tk.HORIZONTAL, command=self.on_slider_change, state='disabled')
        self.z_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.z_label = ttk.Label(slider_frame, text="0 / 0")
        self.z_label.pack(side=tk.LEFT, padx=5)
        
        # Boutons de contrôle
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(control_frame, text="⏮ Début", command=lambda: self.set_z(0)).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="◀ Précédent", command=self.prev_z).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Suivant ▶", command=self.next_z).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Fin ⏭", command=lambda: self.set_z(self.z_size-1)).pack(side=tk.LEFT, padx=2)
    
    def browse_folder(self):
        """Ouvre un dialogue de sélection de dossier"""
        folder = filedialog.askdirectory(title="Sélectionner le dossier de résultats")
        if folder:
            self.folder_entry.delete(0, tk.END)
            self.folder_entry.insert(0, folder)
    
    def load_data(self):
        """Charge les données depuis le dossier sélectionné"""
        folder = self.folder_entry.get()
        if not folder or not os.path.exists(folder):
            messagebox.showerror("Erreur", "Dossier invalide ou inexistant")
            return
        
        try:
            # Définit les répertoires
            simulated_hologram_dir = os.path.join(folder, "simulated_hologram")
            binary_volume_dir = os.path.join(folder, "binary_volume")
            hologram_volume_dir = os.path.join(folder, "hologram_volume")
            
            # Vérifie l'existence du répertoire hologramme
            if not os.path.exists(simulated_hologram_dir):
                messagebox.showerror("Erreur", "Dossier 'simulated_hologram' introuvable")
                return
            
            # Charge l'hologramme (premier fichier .bmp trouvé)
            holo_files = [f for f in os.listdir(simulated_hologram_dir) if f.endswith('.bmp')]
            if holo_files:
                holo_path = os.path.join(simulated_hologram_dir, holo_files[0])
                self.hologram = np.array(Image.open(holo_path))
            else:
                self.hologram = None
            
            # Charge le volume binaire (premier fichier bin_volume*.tiff)
            if os.path.exists(binary_volume_dir):
                bin_files = [f for f in os.listdir(binary_volume_dir) if f.startswith('bin_volume') and f.endswith('.tiff')]
                if bin_files:
                    bin_path = os.path.join(binary_volume_dir, bin_files[0])
                    self.bin_volume = tifffile.imread(bin_path)
                else:
                    self.bin_volume = None
            else:
                self.bin_volume = None
            
            # Charge le volume propagé (premier fichier intensity_volume*.tiff)
            if os.path.exists(hologram_volume_dir):
                prop_files = [f for f in os.listdir(hologram_volume_dir) if f.startswith('intensity_volume') and f.endswith('.tiff')]
                if prop_files:
                    prop_path = os.path.join(hologram_volume_dir, prop_files[0])
                    self.propagated_volume = tifffile.imread(prop_path)
                else:
                    self.propagated_volume = None
            else:
                self.propagated_volume = None
            
            # Détermine la taille Z
            if self.bin_volume is not None:
                self.z_size = self.bin_volume.shape[2] if len(self.bin_volume.shape) == 3 else self.bin_volume.shape[0]
            elif self.propagated_volume is not None:
                self.z_size = self.propagated_volume.shape[2] if len(self.propagated_volume.shape) == 3 else self.propagated_volume.shape[0]
            else:
                self.z_size = 0
            
            if self.z_size > 0:
                self.z_slider.config(to=self.z_size-1, state='normal')
                self.current_z = 0
                self.z_slider.set(0)
                self.update_display()
            else:
                messagebox.showwarning("Attention", "Aucun volume trouvé dans le dossier")
                
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur de chargement :\n{e}")
    
    def on_slider_change(self, value):
        """Callback du slider"""
        self.current_z = int(float(value))
        self.update_display()
    
    def set_z(self, z):
        """Définit le plan Z"""
        if self.z_size == 0:
            return
        self.current_z = max(0, min(z, self.z_size - 1))
        self.z_slider.set(self.current_z)
        self.update_display()
    
    def prev_z(self):
        """Plan précédent"""
        self.set_z(self.current_z - 1)
    
    def next_z(self):
        """Plan suivant"""
        self.set_z(self.current_z + 1)
    
    def update_display(self):
        """Met à jour l'affichage des images"""
        self.z_label.config(text=f"{self.current_z} / {self.z_size-1}")
        
        # Image hologramme (constante)
        if self.hologram is not None:
            holo_img = self.resize_image(self.hologram, max_size=400)
            holo_photo = ImageTk.PhotoImage(holo_img)
            self.holo_label.config(image=holo_photo, text="")
            self.holo_label.image = holo_photo
        
        # Plan binaire
        if self.bin_volume is not None:
            if len(self.bin_volume.shape) == 3:
                bin_plane = self.bin_volume[:, :, self.current_z]
            else:
                bin_plane = self.bin_volume[self.current_z, :, :]
            
            # Affichage direct (déjà en 0-255)
            if bin_plane.dtype != np.uint8:
                bin_plane = bin_plane.astype(np.uint8)
                
            bin_img = self.resize_image(bin_plane, max_size=400)
            bin_photo = ImageTk.PhotoImage(bin_img)
            self.bin_label.config(image=bin_photo, text="")
            self.bin_label.image = bin_photo
        
        # Plan propagé avec surimpression de la segmentation en bleu
        if self.propagated_volume is not None:
            if len(self.propagated_volume.shape) == 3:
                prop_plane = self.propagated_volume[:, :, self.current_z]
            else:
                prop_plane = self.propagated_volume[self.current_z, :, :]
            
            # Normalisation pour l'affichage
            prop_min = prop_plane.min()
            prop_max = prop_plane.max()
            if prop_max > prop_min:
                prop_normalized = ((prop_plane - prop_min) / 
                                   (prop_max - prop_min) * 255).astype(np.uint8)
            else:
                prop_normalized = np.zeros_like(prop_plane, dtype=np.uint8)
            
            # Création de l'image RGB pour la surimpression
            prop_rgb = np.stack([prop_normalized, prop_normalized, prop_normalized], axis=-1)
            
            # Surimpression de la segmentation en bleu
            if self.bin_volume is not None:
                if len(self.bin_volume.shape) == 3:
                    bin_plane = self.bin_volume[:, :, self.current_z]
                else:
                    bin_plane = self.bin_volume[self.current_z, :, :]
                
                # Masque binaire (zones segmentées)
                mask = bin_plane > 0
                
                # Applique la couleur bleue avec transparence (overlay)
                alpha = 0.4  # Transparence de la surimpression
                prop_rgb[mask, 0] = (1 - alpha) * prop_rgb[mask, 0]  # Diminue le rouge
                prop_rgb[mask, 1] = (1 - alpha) * prop_rgb[mask, 1]  # Diminue le vert
                prop_rgb[mask, 2] = np.clip((1 - alpha) * prop_rgb[mask, 2] + alpha * 255, 0, 255).astype(np.uint8)  # Augmente le bleu
            
            prop_img = self.resize_image_rgb(prop_rgb, max_size=400)
            prop_photo = ImageTk.PhotoImage(prop_img)
            self.propagated_label.config(image=prop_photo, text="")
            self.propagated_label.image = prop_photo
    
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
    
    def resize_image_rgb(self, array, max_size=400):
        """Redimensionne une image RGB pour l'affichage"""
        # Convertit en uint8
        if array.dtype != np.uint8:
            array = array.astype(np.uint8)
        
        img = Image.fromarray(array, mode='RGB')
        
        # Redimensionne en gardant le ratio
        width, height = img.size
        if max(width, height) > max_size:
            ratio = max_size / max(width, height)
            new_size = (int(width * ratio), int(height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        return img


def main():
    root = tk.Tk()
    app = VisualizerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

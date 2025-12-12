# -*- coding: utf-8 -*-

"""
Filename: pipeline_holotracker_locate.py

Description:
Pipeline simple de localisation d'objets dans des hologrammes
Version pédagogique avec paramètres hardcodés pour les étudiants

Author: Simon BECKER
Date: 2024-07-09 / Modifié 2025-12-12

License:
GNU General Public License v3.0

Utilisation simple :
python pipeline_holotracker_locate.py

Fichiers requis :
- simu_holo_test.bmp (hologramme d'entrée)

Fichiers générés :
- result.csv (positions des objets détectés)
"""

import cupy as cp
import numpy as np
import time
import os
from PIL import Image
from traitement_holo import *
import propagation as propag
import focus 
from focus import Focus_type
from CCL3D import *
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# PARAMETRES HARDCODES (pour usage pédagogique simple)
# =============================================================================

# Fichiers d'entrée et de sortie
HOLOGRAM_FILE = "./simu_holo_test.bmp"
RESULT_FILE = "./result.csv"

# Paramètres de l'hologramme
WAVELENGTH = 660e-9  # Longueur d'onde (m)
INDEX_MEDIUM = 1.33  # Indice du milieu
MAGNIFICATION = 40.0  # Grossissement
HOLO_SIZE_XY = 1024  # Taille image (pixels)
Z_SIZE = 200  # Nombre de plans Z
PIX_SIZE = 5.5e-6  # Taille pixel (m)
STEP_Z = 0.5e-6  # Pas en Z (m)
VOLUME_CAMERA_DISTANCE = 1e-5 # Distance volume-caméra (m)

# Paramètres de détection
SUM_SIZE = 15  # Taille du noyau de focus
NB_STDVAR_THRESHOLD = 10  # Seuil en nombre d'écart-types
N_CONNECTIVITY = 26  # Connectivité 3D
FILTER_LOW = 15  # Filtre passe-bas
FILTER_HIGH = 125  # Filtre passe-haut
Focus_type = Focus_type.TENEGRAD

# Options d'affichage
DISPLAY_IMAGES = True  # Mettre à True pour voir les images intermédiaires
USE_MATPLOTLIB = True  # True pour matplotlib (titres garantis), False pour PIL (plus rapide)
VERBOSE = True  # Affichage des informations

# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def main():
    """Pipeline principal de traitement"""
    
    print("=" * 60)
    print("PIPELINE DE LOCALISATION D'HOLOGRAMMES")
    print("=" * 60)
    print(f"Fichier d'entrée : {HOLOGRAM_FILE}")
    print(f"Fichier de sortie : {RESULT_FILE}")
    print()
    
    # Vérification du fichier d'entrée
    if not os.path.exists(HOLOGRAM_FILE):
        print(f"❌ ERREUR : Le fichier {HOLOGRAM_FILE} n'existe pas")
        print("   Assurez-vous d'avoir un hologramme nommé 'simu_holo_test.bmp'")
        return 1
    
    # Calcul des paramètres dérivés
    medium_wavelength = WAVELENGTH / INDEX_MEDIUM
    dx = 1000000 * PIX_SIZE / MAGNIFICATION  # en µm
    dy = 1000000 * PIX_SIZE / MAGNIFICATION  # en µm
    dz = STEP_Z * 1e6  # conversion en µm
    
    print("Paramètres calculés :")
    print(f"  - Longueur d'onde dans le milieu : {medium_wavelength*1e9:.1f} nm")
    print(f"  - Résolution XY : {dx:.2f} µm/pixel")
    print(f"  - Résolution Z : {dz:.2f} µm/plan")
    print()
    
    # Allocation mémoire
    print("📋 Allocation de la mémoire GPU...")
    
    h_holo = np.zeros(shape=(HOLO_SIZE_XY, HOLO_SIZE_XY), dtype=np.float32)
    d_holo = cp.zeros(shape=(HOLO_SIZE_XY, HOLO_SIZE_XY), dtype=cp.float32)
    d_fft_holo = cp.zeros(shape=(HOLO_SIZE_XY, HOLO_SIZE_XY), dtype=cp.complex64)
    d_fft_holo_propag = cp.zeros(shape=(HOLO_SIZE_XY, HOLO_SIZE_XY), dtype=cp.complex64)
    d_holo_propag = cp.zeros(shape=(HOLO_SIZE_XY, HOLO_SIZE_XY), dtype=cp.float32)
    d_KERNEL = cp.zeros(shape=(HOLO_SIZE_XY, HOLO_SIZE_XY), dtype=cp.complex64)
    d_volume_module = cp.zeros(shape=(Z_SIZE, HOLO_SIZE_XY, HOLO_SIZE_XY), dtype=cp.float32)
    d_bin_volume_focus = cp.zeros(shape=(Z_SIZE, HOLO_SIZE_XY, HOLO_SIZE_XY), dtype=cp.bool_)
    
    # Chargement de l'hologramme
    print(f"📁 Chargement de l'hologramme : {HOLOGRAM_FILE}")
    
    ini_time = time.perf_counter()
    h_holo = read_image(HOLOGRAM_FILE, HOLO_SIZE_XY, HOLO_SIZE_XY)
    
    if DISPLAY_IMAGES:
        print("   📸 Affichage de l'hologramme d'entrée")
        display(h_holo, "Hologramme d'entrée")
    
    # Normalisation simple (pas de soustraction d'hologramme moyen)
    min_val = h_holo.min()
    max_val = h_holo.max()

    print(f"   Valeurs image : min={min_val:.3f}, max={max_val:.3f}")
    
    d_holo = cp.asarray(h_holo)
    
    # ÉTAPE 1 : Propagation du volume par méthode du spectre angulaire
    print("🔄 ÉTAPE 1 : Propagation du volume...")
    
    t1 = time.perf_counter()

    # Propagation initiale jusqu'au plan du volume
    propag.propag_angular_spectrum(
        d_holo, d_fft_holo, d_KERNEL,
        d_fft_holo_propag, d_holo_propag,
        medium_wavelength, MAGNIFICATION, PIX_SIZE, HOLO_SIZE_XY, HOLO_SIZE_XY, 
        VOLUME_CAMERA_DISTANCE, FILTER_LOW, FILTER_HIGH
    )

    # Reconstruction du volume par propagation plan par plan
    propag.volume_propag_angular_spectrum_to_module(
        d_holo, d_fft_holo, d_KERNEL,
        d_fft_holo_propag, d_volume_module,
        medium_wavelength, MAGNIFICATION, PIX_SIZE, HOLO_SIZE_XY, HOLO_SIZE_XY, 
        0.0, STEP_Z, Z_SIZE, FILTER_LOW, FILTER_HIGH
        )
    t2 = time.perf_counter()
    
    if DISPLAY_IMAGES:
        print("   📸 Affichage des projections du volume propagé")
        display(d_volume_module.max(axis=0), "Max Projection XY - Volume propagé")
        display(d_volume_module.max(axis=1), "Max Projection XZ - Volume propagé")
        display(d_volume_module.max(axis=2), "Max Projection YZ - Volume propagé")

    # ÉTAPE 2 : Calcul du focus
    print("🎯 ÉTAPE 2 : Calcul du focus...")
    
    focus.focus(d_volume_module, d_volume_module, SUM_SIZE, Focus_type.TENEGRAD)
    t3 = time.perf_counter()
    
    if DISPLAY_IMAGES:
        print("   📸 Affichage des projections après focus")
        display(d_volume_module.max(axis=0), "max Projection XY - Après focus")
        display(d_volume_module.max(axis=1), "max Projection XZ - Après focus")
        display(d_volume_module.max(axis=2), "max Projection YZ - Après focus")

    # ÉTAPE 3 : Détection d'objets (CCL3D)
    print("🔍 ÉTAPE 3 : Détection d'objets...")
    
    threshold = calc_threshold(d_volume_module, NB_STDVAR_THRESHOLD)
    d_labels, number_of_labels = CCL3D(d_bin_volume_focus, d_volume_module, type_threshold.THRESHOLD, threshold, N_CONNECTIVITY)
    t4 = time.perf_counter()
    
    print(f"   Seuil calculé : {threshold:.6f}")
    print(f"   Nombre d'objets détectés : {number_of_labels}")
    
    if DISPLAY_IMAGES:
        print("   📸 Affichage des projections binaires")
        display(projection_bool(d_bin_volume_focus, axis=0), "Projection XY - Objets détectés")
        display(projection_bool(d_bin_volume_focus, axis=1), "Projection XZ - Objets détectés")
        display(projection_bool(d_bin_volume_focus, axis=2), "Projection YZ - Objets détectés")

    # ÉTAPE 4 : Analyse des objets détectés
    print("📊 ÉTAPE 4 : Analyse des objets...")
    
    features = np.ndarray(shape=(number_of_labels,), dtype=dobjet)
    features = CCA_CUDA_float(d_labels, d_volume_module, number_of_labels, 1, HOLO_SIZE_XY, HOLO_SIZE_XY, Z_SIZE, dx, dy, dz)
    t5 = time.perf_counter()
    
    # ÉTAPE 5 : Sauvegarde des résultats
    print(f"💾 ÉTAPE 5 : Sauvegarde vers {RESULT_FILE}")
    
    positions = pd.DataFrame(features, columns=['i_image', 'baryX', 'baryY', 'baryZ', 'nb_pix'])
    positions.to_csv(RESULT_FILE, index=False)
    t6 = time.perf_counter()
    
    # Affichage des temps de traitement
    total_time = t6 - ini_time
    print()
    print("⏱️  TEMPS DE TRAITEMENT :")
    print(f"   Propagation : {t2-t1:.3f} s")
    print(f"   Focus       : {t3-t2:.3f} s") 
    print(f"   Détection   : {t4-t3:.3f} s")
    print(f"   Analyse     : {t5-t4:.3f} s")
    print(f"   Sauvegarde  : {t6-t5:.3f} s")
    print(f"   TOTAL       : {total_time:.3f} s")
    print()
    
    # Résumé final
    print("=" * 60)
    print("✅ TRAITEMENT TERMINÉ AVEC SUCCÈS")
    print("=" * 60)
    print(f"Objets détectés : {number_of_labels}")
    print(f"Fichier résultat : {RESULT_FILE}")
    if number_of_labels > 0:
        print()
        print("Aperçu des résultats (5 premiers objets) :")
        print(positions.head().to_string(index=False))
    print("=" * 60)
    
    # Affichage optionnel des graphiques
    if DISPLAY_IMAGES and number_of_labels > 0:
        # Histogramme des intensités
        h_intensite = cp.asnumpy(d_volume_module**2).reshape((HOLO_SIZE_XY * HOLO_SIZE_XY * Z_SIZE,))
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.hist(h_intensite, bins=1000)
        plt.yscale('log')
        plt.title('Distribution des intensités')
        plt.xlabel('Intensité')
        plt.ylabel('Nombre de voxels')
        
        # Graphique 3D des positions
        plt.subplot(1, 2, 2, projection='3d')
        ax = plt.gca()
        Z = positions['baryZ']
        Y = positions['baryY'] 
        X = positions['baryX']
        ax.scatter3D(X, Y, Z, c=positions['nb_pix'], cmap='viridis')
        ax.set_xlabel('X (µm)')
        ax.set_ylabel('Y (µm)')
        ax.set_zlabel('Z (µm)')
        ax.set_title(f'Positions 3D ({number_of_labels} objets)')
        
        plt.tight_layout()
        plt.show()
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
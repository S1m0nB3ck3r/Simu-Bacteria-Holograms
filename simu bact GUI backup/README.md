# Interface GUI pour la Simulation d'Hologrammes de Bactéries

## Description

Cette application Tkinter permet de configurer et générer des hologrammes synthétiques de bactéries de manière interactive.

## Fichiers

- **`simu_bact_gui.py`** : Interface graphique Tkinter
- **`processor_simu_bact.py`** : Script de traitement qui génère l'hologramme
- **`parameters_simu_bact.json`** : Fichier de configuration (mis à jour automatiquement)

## Utilisation

### Lancement de l'interface

```bash
python "simu bact GUI/simu_bact_gui.py"
```

### Fonctionnement

1. **Modification des paramètres** : 
   - Modifiez les valeurs dans l'interface
   - Chaque changement met automatiquement à jour `parameters_simu_bact.json`

2. **Génération d'hologramme** :
   - Cliquez sur "🚀 Générer Hologramme"
   - Le script `processor_simu_bact.py` est lancé automatiquement
   - La génération se fait en arrière-plan

3. **Résultats** :
   - Un nouveau dossier horodaté est créé dans le répertoire de sortie
   - Contient :
     - `holo_0.bmp` : Image de l'hologramme
     - `bin_volume_0.tiff` : Stack TIFF de segmentation (masque binaire)
     - `intensity_volume_0.tiff` : Stack TIFF d'intensité
     - `data_0.npz` : Données complètes (NumPy)
     - `bact_0.txt` : Positions des bactéries

## Paramètres

### 📁 Chemins
- **Dossier de sortie** : Répertoire où seront sauvegardés les résultats

### 📦 Volume
- **Nombre de bactéries** : Nombre de bactéries à générer dans le volume
- **Taille XY hologramme** : Taille en pixels (largeur/hauteur)
- **Bordure** : Taille de la bordure pour éviter les effets de bord FFT
- **Facteur upscale** : Facteur de suréchantillonnage pour la génération
- **Nombre de plans Z** : Nombre de tranches dans la profondeur

### 🔬 Optique
- **Indice milieu** : Indice de réfraction du milieu (ex: 1.33 pour l'eau)
- **Indice bactérie** : Indice de réfraction des bactéries
- **Transmission milieu** : Coefficient de transmission
- **Longueur d'onde** : Longueur d'onde de la source (en mètres)

### 📷 Caméra
- **Taille pixel caméra** : Taille physique d'un pixel (en mètres)
- **Grossissement** : Grossissement de l'objectif
- **Taille voxel Z totale** : Taille totale du volume en Z (en mètres)

### 🦠 Bactéries
- **Longueur min/max** : Plage de longueurs des bactéries (en mètres)
- **Épaisseur min/max** : Plage d'épaisseurs des bactéries (en mètres)

### 💡 Illumination
- **Moyenne illumination** : Intensité moyenne de l'illumination
- **Écart-type min/max** : Plage de bruit gaussien sur l'illumination

## Notes

- La génération peut prendre plusieurs minutes selon les paramètres
- Assurez-vous d'avoir suffisamment d'espace disque
- GPU recommandé (CuPy) pour des performances optimales
- Les fichiers TIFF peuvent être visualisés dans ImageJ/Fiji

## Dépendances

```bash
pip install numpy cupy-cuda12x tkinter pillow tifffile
```

## Auteur

Simon BECKER - 2025

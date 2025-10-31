# Quick Start Guide

## 🚀 Démarrage Rapide

### 1. Lancer la Simulation

**Windows :**
```
Double-cliquez sur launch_simulation.bat
```

**Ligne de commande :**
```bash
python simulation_gui.py
```

### 2. Configurer les Paramètres

- Ajustez les paramètres selon vos besoins
- **Important** : Section SAUVEGARDE
  - Cochez les formats que vous souhaitez sauvegarder
  - Par défaut : BMP, TIFF volumes, CSV positions
- Définissez le nombre d'hologrammes à simuler

### 3. Lancer la Simulation

- Cliquez sur "🚀 Lancer la Simulation"
- Une barre de progression indique l'avancement
- Les résultats sont sauvegardés dans le dossier configuré

### 4. Visualiser les Résultats

**Windows :**
```
Double-cliquez sur launch_visualizer.bat
```

**Ligne de commande :**
```bash
python visualizer_gui.py
```

- Cliquez sur "📁 Parcourir" et sélectionnez le dossier de résultats (YYYY_MM_DD_HH_MM_SS)
- Cliquez sur "🔄 Charger"
- Utilisez le slider pour naviguer dans les plans Z

## 📊 Formats de Sauvegarde

### Hologramme
- **BMP 8bits** : Visualisation rapide (recommandé)
- **TIFF 32bits** : Données complètes haute précision
- **NPY 32bits** : Format NumPy pour analyse Python

### Volumes
- **TIFF multistack** : Compatible ImageJ/Fiji (recommandé)
- **NPY** : Format NumPy pour traitement Python

### Positions
- **CSV** : Compatible Excel, facile à lire (recommandé)
- **TXT** : Format texte brut

## 🎯 Workflow Typique

1. **Configuration** : Ajuster les paramètres dans simulation_gui.py
2. **Génération** : Lancer 1 ou plusieurs hologrammes
3. **Visualisation** : Explorer les résultats avec visualizer_gui.py
4. **Analyse** : Charger les fichiers NPY/TIFF dans vos scripts

## ⚙️ Paramètres Importants

- **Nombre de bactéries** : 200 par défaut
- **Taille hologramme** : 1024x1024 pixels
- **Nombre de plans Z** : 200 (profondeur du volume)
- **Grossissement** : 40x
- **Longueur d'onde** : 660 nm (rouge)

## 📂 Structure des Résultats

```
output_folder/
  YYYY_MM_DD_HH_MM_SS/
    holograms/          # Hologrammes et volumes
      holo_0.bmp
      bin_volume_0.tiff
      intensity_volume_0.tiff
    positions/          # Positions des bactéries
      bact_0.csv
    data_holograms/     # Données complètes NPZ
      data_0.npz
```

## 🆘 Dépannage

**La simulation ne démarre pas :**
- Vérifiez que CuPy est installé
- Vérifiez que vous avez un GPU compatible CUDA

**Les volumes sont noirs :**
- Vérifiez que le nombre de bactéries > 0
- Utilisez le slider Z pour explorer différents plans

**Erreur de mémoire :**
- Réduisez la taille de l'hologramme
- Réduisez le nombre de plans Z

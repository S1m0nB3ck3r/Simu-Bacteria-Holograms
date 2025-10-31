# GUI Simulation et Visualisation - Hologrammes de Bactéries

## Description

Ce dossier contient deux applications séparées :

### 1. **simulation_gui.py** - Application de Simulation
Interface pour configurer et lancer les simulations d'hologrammes de bactéries.

**Fonctionnalités :**
- Configuration complète des paramètres de simulation
- Section SAUVEGARDE avec choix des formats :
  - Hologramme simulé : BMP 8bits, TIFF 32bits, NPY 32bits
  - Volume propagé : TIFF multistack, NPY
  - Volume segmentation : TIFF multistack, NPY bool
  - Positions bactéries : CSV
- Nombre d'hologrammes à simuler (itérations multiples)
- Barre de progression en temps réel
- Fichiers sauvegardés avec suffixe `_i.xxx` (i = numéro d'itération)

**Lancement :**
```bash
python simulation_gui.py
```

### 2. **visualizer_gui.py** - Application de Visualisation
Interface pour visualiser les résultats de simulation.

**Fonctionnalités :**
- Sélection du dossier de résultats
- Affichage côte à côte de :
  - Hologramme simulé
  - Volume segmenté (binaire)
  - Volume propagé (intensité)
- Navigation dans les plans Z avec slider
- Boutons de navigation rapide (début, précédent, suivant, fin)

**Lancement :**
```bash
python visualizer_gui.py
```

## Fichiers

- `simulation_gui.py` : Interface de simulation
- `visualizer_gui.py` : Interface de visualisation
- `processor_simu_bact.py` : Script de traitement (appelé par simulation_gui.py)
- `parameters_simu_bact.json` : Fichier de configuration (auto-généré)
- `processing_status.json` : Fichier de statut (auto-généré pendant l'exécution)
- `processing_result.json` : Fichier de résultats (auto-généré)

## Workflow

1. **Simulation** : Lancer `simulation_gui.py`
   - Configurer les paramètres
   - Choisir les formats de sauvegarde
   - Définir le nombre d'hologrammes
   - Lancer la simulation

2. **Visualisation** : Lancer `visualizer_gui.py`
   - Sélectionner le dossier de résultats généré
   - Explorer les volumes plan par plan

## Structure des résultats

```
output_folder/
  YYYY_MM_DD_HH_MM_SS/
    holograms/
      holo_0.bmp
      holo_1.bmp
      ...
      bin_volume_0.tiff
      bin_volume_1.tiff
      ...
      intensity_volume_0.tiff
      intensity_volume_1.tiff
      ...
    positions/
      bact_0.txt (ou .csv)
      bact_1.txt
      ...
    data_holograms/
      data_0.npz
      data_1.npz
      ...
```

## Auteur

Simon BECKER - 2025

## Licence

GNU General Public License v3.0

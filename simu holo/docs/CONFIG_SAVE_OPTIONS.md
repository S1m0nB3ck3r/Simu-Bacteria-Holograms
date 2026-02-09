# Configuration des Options de Sauvegarde

## Vue d'ensemble

Le système de simulation holographique supporte maintenant des **options de sauvegarde personnalisées** dans les fichiers de configuration JSON. Cela permet de contrôler exactement quels fichiers sont générés lors de chaque simulation.

## Options de Sauvegarde

Les options de sauvegarde sont définies dans la section `save_options` du fichier JSON:

```json
"save_options": {
    "hologram_bmp": true,
    "hologram_tiff": false,
    "hologram_npy": false,
    "propagated_tiff": true,
    "propagated_npy": false,
    "segmentation_tiff": true,
    "segmentation_npy": false,
    "positions_csv": true
}
```

### Détail des options

| Option | Type | Description | Défaut |
|--------|------|-------------|--------|
| `hologram_bmp` | bool | Hologramme final en format BMP 8-bit (pour visualisation rapide) | `true` |
| `hologram_tiff` | bool | Hologramme final en format TIFF 32-bit float (données précises) | `false` |
| `hologram_npy` | bool | Hologramme final en format NumPy NPY 32-bit float | `false` |
| `propagated_tiff` | bool | Volume 3D de propagation en format TIFF multistack | `true` |
| `propagated_npy` | bool | Volume 3D de propagation en format NumPy NPY | `false` |
| `segmentation_tiff` | bool | Segmentation binaire (objets) en format TIFF multistack | `true` |
| `segmentation_npy` | bool | Segmentation binaire (objets) en format NumPy NPY booléen | `false` |
| `positions_csv` | bool | Positions des objets en format CSV avec positions en mètres et en voxels | `true` |

## Formats de fichier

### BMP (8-bit)
- **Avantages**: Léger, visualisation rapide avec n'importe quel lecteur
- **Inconvénients**: Perte de précision, images normalisées à 0-255
- **Utilité**: Affichage/inspection rapide

### TIFF (32-bit float)
- **Avantages**: Précision complète, format standard pour imagerie scientifique
- **Inconvénients**: Fichiers plus volumineux
- **Utilité**: Analyse quantitative, traitement ultérieur

### NPY (NumPy)
- **Avantages**: Format natif NumPy, rapide à charger en Python
- **Inconvénients**: Spécifique à NumPy/SciPy
- **Utilité**: Pipelines Python, intégration directe

### CSV/TXT (positions)
- **CSV**: Format tabulaire standard avec en-têtes
- **TXT**: Format texte simple avec colonnes séparées par espaces
- **Les deux formats sont toujours sauvegardés** pour compatibilité maximale

## Colonnes des fichiers de positions

### Pour les bactéries (CSV)
```
thickness, length, x_position_m, y_position_m, z_position_m, x_voxel, y_voxel, z_voxel, theta_angle, phi_angle
```

### Pour les sphères (CSV)
```
radius, x_position_m, y_position_m, z_position_m, x_voxel, y_voxel, z_voxel
```

**Notes**:
- Positions en **mètres** (m) et en **voxels** (indices de tableau)
- Les angles theta/phi sont en **degrés**
- Utiles pour réidentifier les objets dans les volumes 3D

## Stratégies de sauvegarde recommandées

### Développement / Testing (léger)
```json
"save_options": {
    "hologram_bmp": true,
    "hologram_tiff": false,
    "hologram_npy": false,
    "propagated_tiff": false,
    "propagated_npy": false,
    "segmentation_tiff": false,
    "segmentation_npy": false,
    "positions_csv": true
}
```
**Taille par hologramme**: ~2-5 MB  
**Temps de sauvegarde**: Minimal

### Production (complet)
```json
"save_options": {
    "hologram_bmp": true,
    "hologram_tiff": true,
    "hologram_npy": false,
    "propagated_tiff": true,
    "propagated_npy": false,
    "segmentation_tiff": true,
    "segmentation_npy": false,
    "positions_csv": true
}
```
**Taille par hologramme**: ~200-500 MB  
**Temps de sauvegarde**: Modéré (quelques secondes par hologramme)

### Données de recherche (archivage)
```json
"save_options": {
    "hologram_bmp": true,
    "hologram_tiff": true,
    "hologram_npy": true,
    "propagated_tiff": true,
    "propagated_npy": true,
    "segmentation_tiff": true,
    "segmentation_npy": true,
    "positions_csv": true
}
```
**Taille par hologramme**: ~500 MB - 1 GB  
**Temps de sauvegarde**: Plus lent

## Utilisation en ligne de commande

```bash
# Configuration développement (peu de fichiers)
python main_simu_hologram.py configs/config_bacteria_random.json

# Configuration production (complet)
python main_simu_hologram.py configs/config_bacteria_list.json
```

Les options de sauvegarde sont lues automatiquement depuis le fichier JSON configuré.

## Structure des répertoires de sortie

Avec `"output_dir": null`, les résultats sont organisés comme suit:

```
simu_bacteria/
├── YYYY_MM_DD_HH_MM_SS/
│   ├── holograms/
│   │   ├── holo_0.bmp (si hologram_bmp=true)
│   │   ├── holo_0.tiff (si hologram_tiff=true)
│   │   ├── propagated_volume_0.tiff (si propagated_tiff=true)
│   │   ├── segmentation_0.tiff (si segmentation_tiff=true)
│   │   ├── holo_1.bmp
│   │   └── ...
│   ├── positions/
│   │   ├── bacteria_0.csv (si positions_csv=true)
│   │   ├── bacteria_0.txt (toujours)
│   │   ├── bacteria_1.csv
│   │   └── ...
│   └── data_holograms/
│       ├── data_0.npz
│       └── ...
```

## Conseils de performance

1. **BMP suffisant?** Pour visualisation et inspection, BMP seul est généralement suffisant
2. **Éviter NPY en production**: Les fichiers NPY sont plus lents à sauvegarder
3. **TIFF pour archivage**: Format standard, compatible avec la plupart des logiciels
4. **Positions toujours sauvegardées**: Même si `positions_csv=false`, TXT est toujours créé
5. **Volumes optionnels**: Pour économiser l'espace, décocher `propagated_tiff` et `segmentation_tiff` si non nécessaires

## Migration depuis l'ancien système

Si vous utilisiez les anciens scripts (main_simu_hologram_bacteria_list.py, etc.), les options par défaut dans les nouveaux fichiers JSON correspondent au comportement précédent (sauvegarde complète).

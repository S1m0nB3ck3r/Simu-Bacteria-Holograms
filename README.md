# Simu-Bacteria-Holograms

Simulation d'hologrammes de Gabor pour la détection et localisation 3D de bactéries et sphères. Ce projet combine des méthodes classiques de traitement d'images holographiques avec des techniques de deep learning (U-Net 3D) pour la segmentation volumétrique.

## 🎯 Fonctionnalités principales

- **Simulation d'hologrammes** : Génération d'hologrammes avec bactéries ou sphères (configuration JSON)
- **Interface graphique** : GUI interactive pour paramétrer et générer des datasets
- **Pipeline classique** : Localisation 3D par propagation angulaire, focus et CCL3D
- **Deep Learning** : Segmentation 3D avec U-Net pour l'apprentissage supervisé
- **Reconstruction volumétrique** : Méthode du spectre angulaire avec accélération GPU (CuPy/CUDA)

## 🚀 Démarrage rapide

### 1. Simulation d'hologrammes (Recommandé)

Génération d'hologrammes via fichiers de configuration JSON :

```bash
cd "simu holo"
python main_simu_hologram.py configs/config_bacteria_random.json
```

**Options disponibles** :
- `config_bacteria_random.json` : Bactéries aléatoires
- `config_bacteria_list.json` : Bactéries à positions prédéfinies
- `config_sphere_random.json` : Sphères aléatoires
- `config_sphere_list.json` : Sphères à positions prédéfinies

Voir [simu holo/README.md](simu%20holo/README.md) pour la documentation complète.

### 2. Interface graphique interactive

Génération de datasets pour l'entraînement de réseaux de neurones :

```bash
cd "simu bact GUI"
python simulation_gui.py
```

Permet de :
- Configurer les paramètres (taille, nombre d'objets, propriétés optiques)
- Générer des lots d'hologrammes
- Choisir les formats de sortie (BMP, TIFF, NPY, NPZ)
- Visualiser les résultats avec `visualizer_gui.py`

### 3. Pipeline de localisation classique

Pipeline éducatif sans IA pour comprendre les principes de reconstruction holographique :

```bash
cd localisation_pipeline
python pipeline_holotracker_locate_simple.py
```

**Étapes du pipeline** :
1. **Propagation** : Méthode du spectre angulaire pour reconstruction 3D
2. **Focus** : Calcul du critère de focus (Tenengrad)
3. **Détection** : Seuillage et composantes connexes 3D
4. **Localisation** : Extraction des coordonnées 3D (barycentres)

### 4. Deep Learning (U-Net 3D)

Segmentation volumétrique par réseau de neurones convolutif 3D :

```bash
cd deep_learning_segmentation
python train_UNET3D.py  # Entraînement
python test_UNET3D.py   # Test et évaluation
```

## 📁 Structure du projet

```
Simu-Bacteria-Holograms/
├── simu holo/                      # ⭐ PRINCIPAL: Simulation par config JSON
│   ├── main_simu_hologram.py
│   ├── configs/                    # Fichiers de configuration
│   ├── docs/                       # Documentation détaillée
│   └── examples/
│
├── simu bact GUI/                  # Interface graphique
│   ├── simulation_gui.py
│   ├── visualizer_gui.py
│   └── processor_simu_bact.py
│
├── localisation_pipeline/          # Pipelines de localisation
│   ├── pipeline_holotracker_locate_simple.py
│   └── main_reconstruction_volume.py
│
├── deep_learning_segmentation/     # Deep learning (U-Net 3D)
│   ├── train_UNET3D.py
│   ├── test_UNET3D.py
│   ├── model.py
│   └── ...
│
├── libs/                           # 📦 Modules centralisés
│   ├── simu_hologram.py           # Génération hologrammes
│   ├── propagation.py             # Propagation onde
│   ├── traitement_holo.py         # Post-processing
│   ├── typeHolo.py                # Définitions types
│   ├── CCL3D.py                   # Composantes connexes 3D
│   └── focus.py                   # Critères de focus
│
└── [Documentation]
    ├── README.md                   # Ce fichier
    ├── QUICK_START.md              # Guide de démarrage
    └── PROJECT_STRUCTURE.md        # Organisation détaillée
```

## 🔧 Prérequis

### Matériel
- **GPU NVIDIA** avec support CUDA (obligatoire pour CuPy)

### Logiciels
```bash
pip install numpy cupy-cuda11x pillow tifffile matplotlib pandas
pip install torch torchvision torchmetrics  # Pour deep learning
pip install scikit-learn scipy              # Pour CCL3D
```

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [QUICK_START.md](QUICK_START.md) | Guide de démarrage rapide |
| [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) | Organisation détaillée du projet |
| [simu holo/README.md](simu%20holo/README.md) | Documentation simulation JSON |
| [simu holo/docs/CONFIG_GUIDE.md](simu%20holo/docs/CONFIG_GUIDE.md) | Référence paramètres |
| [libs/README.md](libs/README.md) | Documentation des modules |

## 🛠️ Utilisation

### Génération de données d'entraînement

1. Créer une configuration (ou copier un template)
2. Lancer la simulation :
   ```bash
   python "simu holo/main_simu_hologram.py" "simu holo/configs/ma_config.json"
   ```
3. Les résultats sont dans `simu_bacteria/` ou `simu_sphere/`

### Test du pipeline classique

```bash
cd localisation_pipeline
python pipeline_holotracker_locate_simple.py
```

Résultats : `result.csv` avec positions (X, Y, Z) des objets détectés

### Entraînement U-Net 3D

1. Générer des données avec `simu holo/` (option `save_npz_data`)
2. Configurer `deep_learning_segmentation/config_train.json`
3. Lancer :
   ```bash
   python deep_learning_segmentation/train_UNET3D.py
   ```

## 📖 Méthodes implémentées

### Propagation
- **Spectre angulaire** : Propagation exacte dans l'espace de Fourier
- **Fresnel** : Approximation paraxiale
- **Rayleigh-Sommerfeld** : Propagation rigoureuse

### Focus
- **Tenengrad** : Gradient de Sobel au carré (recommandé)
- **Variance** : Variance locale
- **Laplacien** : Dérivée seconde

### Détection
- **CCL3D** : Composantes connexes 3D (connectivité 6, 18, 26)
- **Seuillage adaptatif** : Basé sur l'écart-type

### Deep Learning
- **U-Net 3D** : Segmentation volumétrique avec skip connections
- **Patchs 3D** : Traitement par fenêtres glissantes
- **Métriques** : Dice Score, IoU, Precision, Recall

## 📄 License

GNU General Public License v3.0 - Voir [LICENCE](LICENCE)

## 👤 Auteur

Simon BECKER - 2024-2025


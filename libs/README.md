# 📦 libs/ - Modules Centralisés

Ce répertoire contient tous les modules de base du projet Simu-Bacteria-Holograms. C'est le cœur du système, utilisé par tous les scripts de simulation, traitement et analyse.

## 📋 Contenu des Modules

### Modules Core

| Module | Lignes | Description | Fonctionnalités principales |
|--------|--------|-------------|------------------------------|
| **simu_hologram.py** | ~800 | Génération hologrammes | Génération bactéries/sphères, insertion GPU dans volumes, hologrammes de Gabor |
| **propagation.py** | ~376 | Propagation onde | Spectre angulaire, Fresnel, Rayleigh-Sommerfeld, propagation volumétrique |
| **traitement_holo.py** | ~400 | Post-processing | Intensité, normalisation, filtrage, sauvegarde images (BMP/TIFF) |
| **typeHolo.py** | ~150 | Définitions types | Classes `Bacterie`, `Sphere`, `objet`, `info_Holo` |
| **CCL3D.py** | ~365 | Composantes connexes 3D | Labeling 3D, calcul barycentres, connectivité 6/18/26 |
| **focus.py** | ~286 | Critères de focus | Tenengrad, Variance, Laplacien, traitement volumétrique |

## 🔗 Système d'Import Unifié (Février 2026)

Le projet utilise désormais un système d'imports cohérent basé sur le package `libs/`.

### Pattern Standard (Recommandé)

**Depuis `localisation_pipeline/`, `deep_learning_segmentation/`, etc.** :

```python
import sys
import os

# Ajouter la racine du projet au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Imports depuis libs/ avec préfixe
from libs.traitement_holo import *
from libs import propagation as propag
from libs.CCL3D import *
from libs import focus
from libs.focus import Focus_type
```

**Utilisé par** :
- `localisation_pipeline/pipeline_holotracker_locate_simple.py`
- `localisation_pipeline/main_reconstruction_volume.py`
- `deep_learning_segmentation/train_UNET3D.py`
- `deep_learning_segmentation/test_UNET3D.py`

### Imports Relatifs (Dans libs/)

**À l'intérieur des modules de `libs/`** :

```python
# Dans libs/propagation.py
from . import typeHolo
from .traitement_holo import *

# Dans libs/simu_hologram.py
from . import propagation
from . import traitement_holo
```

Les modules du package `libs/` s'importent entre eux avec des imports relatifs (`.`).

### Depuis simu_holo/

```python
import sys
import os

# Racine et libs
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'libs'))

from simu_hologram import *
import propagation
import traitement_holo
```

## 🔄 Dépendances Entre Modules

```
typeHolo.py
    ├─ Classes: Bacterie, Sphere, objet, info_Holo
    └─ Pas de dépendances internes

propagation.py
    ├─ Importe: typeHolo (relatif)
    ├─ Importe: traitement_holo (relatif)
    └─ Méthodes: spectre angulaire, Fresnel, RS

traitement_holo.py
    ├─ Pas de dépendances internes
    └─ Fonctions: intensité, sauvegarde, filtrage

simu_hologram.py
    ├─ Importe: propagation (relatif)
    ├─ Importe: traitement_holo (relatif)
    └─ Génère hologrammes complets

CCL3D.py
    ├─ Indépendant
    └─ Labeling 3D avec CuPy

focus.py
    ├─ Indépendant
    └─ Critères de focus volumétriques
```

## ✨ Dépendances Externes

### Essentielles
```bash
pip install numpy         # Calculs numériques
pip install cupy-cuda11x  # GPU CUDA (adapter version)
pip install pillow        # Manipulation images
pip install tifffile      # Format TIFF
pip install matplotlib    # Visualisation
```

### Machine Learning (optionnel)
```bash
pip install torch torchvision  # Deep learning
pip install torchmetrics       # Métriques ML
```

### Scientifiques
```bash
pip install scipy         # Outils scientifiques
pip install pandas        # DataFrames (pour CCL3D)
```

## 📍 Fonctionnalités par Module

### simu_hologram.py
- `generate_bacteria_random()` : Génération aléatoire de bactéries
- `generate_spheres_random()` : Génération aléatoire de sphères
- `insert_bacteria_volume()` : Insertion bactérie dans volume GPU
- `insert_sphere_volume()` : Insertion sphère dans volume GPU
- `holo_simu()` : Simulation hologramme complet

### propagation.py
- `propag_angular_spectrum()` : Propagation plan à plan
- `volume_propag_angular_spectrum_to_module()` : Reconstruction volumétrique
- `volume_propag_angular_spectrum_complex()` : Reconstruction complexe
- `calc_KERNEL_PROPAG()` : Calcul kernel de propagation

### traitement_holo.py
- `read_image()` : Lecture image (BMP, TIFF)
- `save_holo_bmp()` : Sauvegarde BMP
- `save_holo_tiff()` : Sauvegarde TIFF
- `intensite()` : Calcul intensité (|A|²)
- `display()` : Affichage matplotlib
- `normalise_to_U8()` : Normalisation 8-bit

### CCL3D.py
- `CCL3D()` : Labeling composantes connexes 3D
- `CCA_CUDA_float()` : Analyse des labels (barycentres, volumes)
- Supporte connectivité 6, 18, 26

### focus.py
- `focus()` : Calcul critère de focus sur volume
- Types disponibles : `TENEGRAD`, `VARIANCE`, `LAPLACIAN`

### typeHolo.py
- Classes de données : `Bacterie`, `Sphere`, `objet`, `info_Holo`
- Types NumPy pour export CSV

## 🔧 Vérification d'Installation

```bash
# Test imports depuis la racine
python -c "import sys, os; sys.path.insert(0, '.'); from libs.traitement_holo import *; print('✅ Imports OK')"

# Test avec CuPy
python -c "import cupy as cp; print('✅ CuPy OK -', cp.cuda.runtime.getDeviceCount(), 'GPU(s)')"

# Test imports depuis localisation_pipeline/
cd localisation_pipeline
python -c "import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..')); from libs import propagation; print('✅ OK')"
cd ..
```

## 📊 Utilisation Typique

### Pipeline de localisation classique

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from libs.traitement_holo import *
from libs import propagation as propag
from libs import focus
from libs.CCL3D import *

# 1. Charger hologramme
h_holo = read_image("hologram.bmp")

# 2. Propager volume
d_volume = cp.zeros((Z, X, Y), dtype=cp.float32)
propag.volume_propag_angular_spectrum_to_module(...)

# 3. Calculer focus
focus.focus(d_volume, d_volume, 15, focus.Focus_type.TENEGRAD)

# 4. Détecter objets
d_labels, nb_objects = CCL3D(d_volume, ...)

# 5. Analyser
positions = CCA_CUDA_float(d_labels, d_volume, ...)
```

### Simulation hologramme

```python
from libs import simu_hologram
from libs import propagation
from libs import traitement_holo

# Générer objets
bacteria_list = simu_hologram.generate_bacteria_random(...)

# Créer volume
d_volume = cp.zeros((Z, X, Y), dtype=cp.complex64)
simu_hologram.insert_bacteria_volume(bacteria, d_volume, ...)

# Simuler hologramme
d_holo = simu_hologram.holo_simu(d_volume, ...)

# Sauvegarder
traitement_holo.save_holo_bmp(d_holo, "output.bmp")
```

## 📚 Documentation Connexe

| Document | Description |
|----------|-------------|
| [../README.md](../README.md) | Vue d'ensemble du projet |
| [../QUICK_START.md](../QUICK_START.md) | Guide de démarrage |
| [../PROJECT_STRUCTURE.md](../PROJECT_STRUCTURE.md) | Structure détaillée |
| [../simu holo/README.md](../simu%20holo/README.md) | Documentation simulation |

## 🔄 Historique et Migration

### Avant (Structure Ancienne)
Les modules étaient à la racine du projet, ce qui causait :
- Confusion sur l'origine des imports
- Difficultés de navigation
- Risques de conflits de noms

### Après (Février 2026)
- ✅ Tous les modules dans `libs/`
- ✅ Imports avec préfixe `libs.`
- ✅ Imports relatifs entre modules de libs/
- ✅ Structure claire et professionnelle
- ✅ Documentation mise à jour

## ⚠️ Notes Importantes

1. **Ne pas modifier directement** : Ces modules sont utilisés par tout le projet
2. **Tester après modifications** : Vérifier tous les scripts utilisateurs
3. **Imports relatifs** : Dans `libs/`, toujours utiliser `from . import module`
4. **GPU requis** : CuPy nécessite une carte NVIDIA avec CUDA

## 🆘 Support

En cas de problème d'import :
1. Vérifier que vous êtes dans le bon répertoire
2. Vérifier que `sys.path.insert(0, ...)` pointe vers la racine
3. Consulter [QUICK_START.md](../QUICK_START.md) section "Résolution de problèmes"

---

**Dernière mise à jour** : Février 2026  
**Version** : 2.0 - Architecture modulaire avec système d'imports unifié
- **CuPy**: Calculs GPU avec CUDA
- **PIL/Pillow**: Manipulation images
- **tifffile**: Lecture/écriture TIFF

### Machine Learning
- **PyTorch**: Entraînement et inférence UNet3D
- **wandb**: Logging expériences (optionnel)

### Scientifiques
- **SciPy**: Outils via CuPy (`cupyx.scipy`)
- **Matplotlib**: Visualisation

## 🔄 Dépendances Entre Modules

```
typeHolo.py
    └─ Base des classes Bacterie, Sphere, objet

simu_hologram.py
    ├─ Utilise: typeHolo
    └─ Génère: Bacteria/Sphere objects

propagation.py
    └─ Propage ondes (angular spectrum method)

traitement_holo.py
    └─ Post-traite hologrammes

CCL3D.py
    ├─ Utilise: CuPy
    └─ Labeling composantes connexes 3D

focus.py
    ├─ Utilise: NumPy, Matplotlib
    └─ Critères focus volumétriques
```

## 📍 Historique Migration

**Avant** (modules à la racine):
```
Simu-Bacteria-Holograms/
├── simu_hologram.py
├── propagation.py
├── traitement_holo.py
├── typeHolo.py
├── CCL3D.py
├── focus.py
└── [autres fichiers]
```

**Après** (modules centralisés):
```
Simu-Bacteria-Holograms/
├── libs/
│   ├── simu_hologram.py
│   ├── propagation.py
│   ├── traitement_holo.py
│   ├── typeHolo.py
│   ├── CCL3D.py
│   ├── focus.py
│   └── README.md (ce fichier)
├── [autres fichiers]
└── [fichiers racine avec imports mis à jour]
```

**Fichiers originaux à la racine** toujours présents avec commentaire:
```python
# DEPENDENCY - Moved to libs/
```

## ⚙️ Vérification d'Importation

Pour tester que les imports fonctionnent:

```bash
# Depuis racine
python -c "import sys; sys.path.insert(0, 'libs'); from simu_hologram import *; print('OK')"

# Depuis simu_holo/
cd simu_holo
python -c "import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'libs')); from simu_hologram import *; print('OK')"
```

## 📚 Références Supplémentaires

- [QUICK_START.md](../QUICK_START.md) - Guide de démarrage rapide
- [PROJECT_STRUCTURE.md](../PROJECT_STRUCTURE.md) - Structure complète du projet
- [simu_holo/README.md](../simu_holo/README.md) - Documentation simulation

---

**Note importante**: Ces modules sont la **fondation** du projet. Toute modification doit être testée pour assurer la compatibilité avec tous les fichiers utilisateurs.

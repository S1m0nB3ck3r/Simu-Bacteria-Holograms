# Dépendances Requises

## Python
- Python 3.10 ou supérieur

## Packages Python Requis

```bash
# Installation complète
pip install numpy cupy-cuda11x pillow tifffile scipy tkinter

# Ou via requirements.txt
pip install -r requirements.txt
```

### Core
- **numpy** : Calcul numérique
- **cupy** : Calcul GPU (CUDA required)
- **pillow** : Traitement d'images
- **tifffile** : Lecture/écriture TIFF multistack

### GUI
- **tkinter** : Interface graphique (inclus avec Python)
- **scipy** : Opérations scientifiques (ndimage pour dilatation)

### Modules locaux
- **simu_hologram.py** : Fonctions de simulation (dans le dossier parent)
- **propagation.py** : Propagation du champ (dans le dossier parent)
- **traitement_holo.py** : Traitement holographique (dans le dossier parent)
- **typeHolo.py** : Types et classes (dans le dossier parent)

## Matériel Requis

### GPU
- GPU NVIDIA avec support CUDA
- CUDA Toolkit 11.x ou supérieur
- Au moins 4 GB de VRAM (8 GB recommandé)

### RAM
- Au moins 8 GB de RAM système
- 16 GB recommandé pour grandes simulations

## Vérification de l'Installation

### Test rapide
```bash
python -c "import numpy, cupy, PIL, tifffile, scipy; print('✅ Tous les packages sont installés')"
```

### Test GPU
```bash
python -c "import cupy as cp; print(f'GPU: {cp.cuda.Device().name}'); print(f'CUDA: {cp.cuda.runtime.runtimeGetVersion()}')"
```

## Problèmes Courants

### CuPy ne s'installe pas
```bash
# Installez la version correspondant à votre CUDA
pip install cupy-cuda110  # CUDA 11.0
pip install cupy-cuda111  # CUDA 11.1
pip install cupy-cuda12x  # CUDA 12.x
```

### Tkinter non disponible
```bash
# Ubuntu/Debian
sudo apt-get install python3-tk

# Windows
# Tkinter est inclus dans l'installation Python standard
```

### ImportError pour les modules locaux
Assurez-vous que les fichiers `simu_hologram.py`, `propagation.py`, `traitement_holo.py` et `typeHolo.py` sont dans le dossier parent (`..`).

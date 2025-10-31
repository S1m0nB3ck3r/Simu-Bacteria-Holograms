# Guide de Démarrage Rapide

## 🚀 Installation et Premier Lancement

### 1. Vérifier les dépendances

```bash
cd "simu bact GUI"
python test_setup.py
```

Ce script vérifie que tous les modules nécessaires sont installés.

### 2. Lancer l'interface

**Option A - Double-clic sur le fichier batch (Windows):**
```
launch_gui.bat
```

**Option B - Ligne de commande:**
```bash
python simu_bact_gui.py
```

### 3. Première génération

1. L'interface s'ouvre avec les paramètres par défaut
2. Modifiez les paramètres si nécessaire (optionnel)
3. Cliquez sur "🚀 Générer Hologramme"
4. Attendez la fin du traitement (plusieurs minutes)
5. Les résultats sont dans le dossier de sortie avec un timestamp

## 📁 Structure des Fichiers

```
simu bact GUI/
├── simu_bact_gui.py              # Interface Tkinter (à lancer)
├── processor_simu_bact.py         # Moteur de génération (appelé automatiquement)
├── parameters_simu_bact.json      # Configuration (mis à jour par l'interface)
├── launch_gui.bat                 # Lanceur Windows (double-clic)
├── test_setup.py                  # Script de vérification
├── README.md                      # Documentation complète
└── QUICKSTART.md                  # Ce fichier
```

## 🎯 Workflow Typique

1. **Ouvrir l'interface** → `launch_gui.bat` ou `python simu_bact_gui.py`
2. **Ajuster les paramètres** → Modifications automatiquement sauvegardées
3. **Générer** → Clic sur le bouton "Générer Hologramme"
4. **Visualiser** → Ouvrir les fichiers TIFF avec ImageJ/Fiji

## 📊 Résultats Générés

Chaque génération crée un dossier horodaté contenant :

```
simu_bact_random/
└── 2025_10_24_14_30_15/
    ├── holograms/
    │   ├── holo_0.bmp                  # Image hologramme (visualisation)
    │   ├── bin_volume_0.tiff           # Stack TIFF segmentation
    │   └── intensity_volume_0.tiff     # Stack TIFF intensité
    ├── data_holograms/
    │   └── data_0.npz                  # Données NumPy complètes
    └── positions/
        └── bact_0.txt                  # Positions des bactéries
```

## ⚙️ Paramètres Recommandés

### Configuration Rapide (test)
- Nombre de bactéries : 50
- Taille XY : 512
- Nombre de plans Z : 100

### Configuration Standard (production)
- Nombre de bactéries : 200
- Taille XY : 1024
- Nombre de plans Z : 200

### Configuration Haute Résolution
- Nombre de bactéries : 500
- Taille XY : 2048
- Nombre de plans Z : 300

## 🔧 Dépannage

### L'interface ne se lance pas
```bash
# Vérifier les imports
python test_setup.py

# Installer tkinter si manquant (généralement inclus avec Python)
# Sur Ubuntu/Debian : sudo apt-get install python3-tk
```

### Erreur CuPy
```bash
# Vérifier CUDA
python -c "import cupy; print(cupy.__version__)"

# Installer CuPy adapté à votre version CUDA
pip install cupy-cuda12x  # Pour CUDA 12.x
```

### Erreur de mémoire GPU
- Réduire `holo_size_xy`
- Réduire `number_of_bacteria`
- Réduire `z_size`

### Les TIFF ne s'ouvrent pas
- Utiliser ImageJ : https://imagej.net/
- Ou Fiji : https://fiji.sc/

## 💡 Astuces

1. **Test rapide** : Réduire tous les paramètres de moitié pour un test rapide
2. **Sauvegarde** : Le fichier JSON est sauvegardé automatiquement
3. **Multiples générations** : Fermer et relancer le processor pour générer plusieurs hologrammes
4. **Monitoring** : Observer la console pour suivre la progression

## 📞 Support

Pour toute question ou problème :
- Consulter le README.md complet
- Vérifier les logs dans la console
- Exécuter `test_setup.py` pour diagnostiquer

## 🎓 Pour Aller Plus Loin

Voir `README.md` pour :
- Description détaillée de tous les paramètres
- Théorie de la simulation
- Format des fichiers de sortie
- API Python pour l'automatisation

---

**Créé par Simon BECKER - 2025**

# Simulation Holographique (simu_holo)

Système complet de simulation d'hologrammes de bactéries et de sphères basé sur des fichiers de configuration JSON.

## 📁 Structure du répertoire

```
simu_holo/
├── README.md                          # Ce fichier
├── main_simu_hologram.py              # Script principal de simulation
├── generate_config.py                 # Générateur de configurations
│
├── configs/                           # Fichiers de configuration JSON
│   ├── config_bacteria_random.json    # Bactéries aléatoires
│   ├── config_bacteria_list.json      # Bactéries prédéfinies
│   ├── config_sphere_random.json      # Sphères aléatoires
│   ├── config_sphere_list.json        # Sphères prédéfinies
│   └── config_template.json           # Template de référence
│
├── docs/                              # Documentation
│   ├── CONFIG_SAVE_OPTIONS.md         # Guide des options de sauvegarde
│   ├── CONFIG_GUIDE.md                # Référence complète des paramètres
│   ├── IMPROVEMENTS_SUMMARY.md        # Résumé des améliorations
│   └── REFACTOR_JSON_CONFIG.md        # Notes sur la refonte JSON
│
└── examples/                          # Exemples d'utilisation
    └── run_examples.sh                # Scripts d'exemple
```

## 🚀 Démarrage rapide

### Installation des dépendances

```bash
# Dépendances Python requises
pip install numpy cupy pillow tifffile torchmetrics torch
```

### Utilisation basique

```bash
# Simulation bactéries aléatoires
python main_simu_hologram.py configs/config_bacteria_random.json

# Simulation bactéries prédéfinies
python main_simu_hologram.py configs/config_bacteria_list.json

# Simulation sphères aléatoires
python main_simu_hologram.py configs/config_sphere_random.json

# Simulation sphères prédéfinies
python main_simu_hologram.py configs/config_sphere_list.json
```

### Résultats

Les résultats sont organisés par défaut dans `simu_bacteria/` ou `simu_sphere/` :

```
simu_bacteria/
└── YYYY_MM_DD_HH_MM_SS/
    ├── holograms/          # Images et volumes d'hologrammes
    ├── positions/          # Fichiers de positions des objets
    └── data_holograms/     # Données NPZ (pour deep learning)
```

## 📋 Modes de simulation

### 1. **bacteria_random**
Génère des hologrammes avec bactéries aléatoires.

**Configuration**:
```json
{
    "mode": "bacteria_random",
    "nb_holo": 100,
    "nb_objects": 50,
    "length_min": 1.5e-6,
    "length_max": 2.0e-6,
    "thickness_min": 0.8e-6,
    "thickness_max": 0.8e-6
}
```

### 2. **bacteria_list**
Génère des hologrammes avec bactéries à positions prédéfinies.

**Configuration**:
```json
{
    "mode": "bacteria_list",
    "nb_holo": 10,
    "bacteria": [
        {
            "pos_x": 1.0e-5,
            "pos_y": 1.0e-5,
            "pos_z": 5.0e-5,
            "length": 1.75e-6,
            "thickness": 0.8e-6,
            "theta": 0.0,
            "phi": 0.0
        }
    ]
}
```

### 3. **sphere_random**
Génère des hologrammes avec sphères aléatoires.

**Configuration**:
```json
{
    "mode": "sphere_random",
    "nb_holo": 100,
    "nb_objects": 50,
    "radius_min": 0.5e-6,
    "radius_max": 2.0e-6
}
```

### 4. **sphere_list**
Génère des hologrammes avec sphères à positions prédéfinies.

**Configuration**:
```json
{
    "mode": "sphere_list",
    "nb_holo": 10,
    "spheres": [
        {
            "pos_x": 2.0e-5,
            "pos_y": 2.0e-5,
            "pos_z": 5.0e-5,
            "radius": 0.8e-6
        }
    ]
}
```

## ⚙️ Paramètres de configuration

### Paramètres optiques
| Paramètre | Description | Défaut |
|-----------|-------------|--------|
| `pix_size` | Taille du pixel du capteur (m) | 5.5e-6 |
| `magnification` | Grossissement optique | 40.0 |
| `wavelength` | Longueur d'onde (m) | 660e-9 |
| `index_medium` | Indice de réfraction du milieu | 1.33 |
| `index_object` | Indice de réfraction de l'objet | 1.335 |

### Paramètres géométriques
| Paramètre | Description | Défaut |
|-----------|-------------|--------|
| `holo_size_xy` | Taille de l'hologramme (pixels) | 1024 |
| `border` | Bordure anti-aliasing (pixels) | 256 |
| `z_size` | Nombre de plans de propagation | 200 |
| `upscale_factor` | Facteur d'upsampling | 2 |
| `distance_volume_camera` | Distance volume-caméra (m) | 0.01 |

### Paramètres d'illumination
| Paramètre | Description | Défaut |
|-----------|-------------|--------|
| `illumination_mean` | Niveau moyen d'illumination | 1.0 |
| `sensor_noise_std_min` | Bruit capteur minimum (std relatif au fond, ajouté à l'intensité après propagation) | 0.015 |
| `sensor_noise_std_max` | Bruit capteur maximum (std relatif au fond) | 0.015 |

### Paramètres de sauvegarde
Voir [CONFIG_SAVE_OPTIONS.md](docs/CONFIG_SAVE_OPTIONS.md)

## 📊 Options de sauvegarde

Contrôlez exactement quels fichiers sont générés:

```json
"save_options": {
    "hologram_bmp": true,           // Hologramme 2D (8-bit)
    "hologram_tiff": false,         // Hologramme 2D (32-bit)
    "hologram_npy": false,          // Hologramme 2D (NumPy)
    "propagated_tiff": true,        // Volume 3D de propagation
    "propagated_npy": false,        // Volume 3D (NumPy)
    "segmentation_tiff": true,      // Segmentation 3D binaire
    "segmentation_npy": false,      // Segmentation (NumPy)
    "positions_csv": true           // Positions en CSV
}
```

**Recommandations**:
- **Développement**: BMP + CSV (5-10 MB/hologramme)
- **Production**: BMP + TIFF + Segmentation (300-500 MB/hologramme)
- **Archivage**: Tous les formats (500 MB - 1 GB/hologramme)

## 🔧 Personnaliser les configurations

### Créer une nouvelle configuration

1. **Copier un template**:
   ```bash
   cp configs/config_template.json configs/mon_config.json
   ```

2. **Éditer les paramètres**:
   ```json
   {
       "mode": "bacteria_random",
       "nb_holo": 50,
       "nb_objects": 100,
       "pix_size": 5.5e-6,
       ...
   }
   ```

3. **Lancer la simulation**:
   ```bash
   python main_simu_hologram.py configs/mon_config.json
   ```

### Utiliser le générateur de configurations

```bash
python generate_config.py --preset bacteria_medium --output mon_config.json
```

Présets disponibles:
- `bacteria_small`: 10 bactéries
- `bacteria_medium`: 50 bactéries
- `bacteria_large`: 200 bactéries
- `bacteria_uv`: Configuration UV optimisée
- `sphere_small`: Petites sphères
- `sphere_large`: Grandes sphères

## 📖 Documentation

- **[CONFIG_SAVE_OPTIONS.md](docs/CONFIG_SAVE_OPTIONS.md)**: Guide détaillé des options de sauvegarde
- **[CONFIG_GUIDE.md](docs/CONFIG_GUIDE.md)**: Référence complète de tous les paramètres
- **[IMPROVEMENTS_SUMMARY.md](docs/IMPROVEMENTS_SUMMARY.md)**: Résumé des améliorations
- **[REFACTOR_JSON_CONFIG.md](docs/REFACTOR_JSON_CONFIG.md)**: Notes sur la refonte du système

## 🎯 Cas d'usage typiques

### Générer un dataset de test
```json
{
    "mode": "bacteria_random",
    "nb_holo": 10,
    "output_dir": null,
    "save_options": {
        "hologram_bmp": true,
        "propagated_tiff": false,
        "segmentation_tiff": false,
        "positions_csv": true
    }
}
```

### Produire des données pour deep learning
```json
{
    "mode": "bacteria_random",
    "nb_holo": 1000,
    "output_dir": "/path/to/training_data",
    "save_options": {
        "hologram_bmp": true,
        "propagated_tiff": true,
        "segmentation_tiff": true,
        "positions_csv": true
    }
}
```

### Analyser des configurations spécifiques
```json
{
    "mode": "bacteria_list",
    "nb_holo": 5,
    "save_options": {
        "hologram_bmp": true,
        "hologram_tiff": true,
        "propagated_tiff": true,
        "segmentation_tiff": true,
        "positions_csv": true
    }
}
```

## ⚡ Performance

### Temps typiques
- **Génération bactéries**: ~5 secondes par hologramme
- **Propagation**: ~30-60 secondes par hologramme  
- **Sauvegarde**: ~5-10 secondes par hologramme
- **Total**: ~45-90 secondes par hologramme

### Optimisation
- Réduire `z_size` pour accélérer (défaut 200)
- Désactiver volumes optionnels (TIFF/NPY)
- Utiliser BMP au lieu de TIFF si possible
- Augmenter `upscale_factor` pour plus de détails (plus lent)

## 🐛 Dépannage

### Erreur: "Configuration file not found"
```bash
# Vérifier le chemin du fichier config
python main_simu_hologram.py configs/config_bacteria_random.json
```

### Erreur: "Missing required configuration key"
```bash
# Vérifier que le fichier JSON contient les clés requises
# 'mode' et 'nb_holo' sont obligatoires
```

### Répertoire de sortie trop volumineux
```json
"save_options": {
    "hologram_bmp": true,
    "hologram_tiff": false,    // Désactiver
    "propagated_tiff": false,  // Désactiver
    "segmentation_tiff": false,// Désactiver
    "positions_csv": true
}
```

## 📞 Support

Pour les questions ou problèmes:
1. Consulter la documentation dans `docs/`
2. Vérifier les fichiers de configuration d'exemple dans `configs/`
3. Examiner les logs de la simulation

## 📝 Licence

Voir LICENCE au niveau du projet parent.

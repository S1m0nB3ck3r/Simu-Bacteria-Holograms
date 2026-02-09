# Structure du Projet

## Organisation des répertoires

```
Simu-Bacteria-Holograms/                     # Répertoire racine du projet
│
├── simu holo/                              # ⭐ PRINCIPAL: Simulation d'hologrammes (JSON config)
│   ├── main_simu_hologram.py               # ← SCRIPT PRINCIPAL UNIFIÉ
│   ├── generate_config.py                  # ← Générateur de config
│   ├── README.md                           # Guide complet d'utilisation
│   ├── setup.sh                            # Script de setup
│   │
│   ├── configs/                            # 📋 Fichiers de configuration JSON
│   │   ├── config_bacteria_random.json
│   │   ├── config_bacteria_list.json
│   │   ├── config_sphere_random.json
│   │   ├── config_sphere_list.json
│   │   └── config_template.json
│   │
│   ├── docs/                               # 📚 Documentation détaillée
│   │   ├── CONFIG_GUIDE.md                 # Référence complète des paramètres
│   │   ├── CONFIG_SAVE_OPTIONS.md          # Guide des 8 options de sauvegarde
│   │   ├── IMPROVEMENTS_SUMMARY.md         # Résumé des améliorations ML
│   │   └── REFACTOR_JSON_CONFIG.md         # Notes techniques de refonte
│   │
│   ├── examples/                           # 🔧 Exemples d'utilisation
│   │   └── run_examples.sh                 # 5 scénarios d'exemple
│   │
│   └── legacy/                             # 📦 Scripts obsolètes (historique)
│       ├── main_simu_hologram_bacteria_list.py
│       ├── main_simu_hologram_random_bact.py
│       ├── main_simu_hologram_random_sphere.py
│       ├── main_simu_hologram_sphere_list.py
│       └── README.md
│
├── simu bact GUI/                          # 🖥️ Interface graphique interactive
│   ├── simulation_gui.py                   # GUI principale
│   ├── processor_simu_bact.py              # Processeur en arrière-plan
│   ├── visualizer_gui.py                   # Visualiseur GUI
│   ├── parameters_simu_bact.json           # Config GUI
│   └── ...
│
├── [Modules core de simulation]
│   ├── simu_hologram.py                    # Génération hologrammes
│   ├── propagation.py                      # Propagation onde (spectre angulaire)
│   ├── traitement_holo.py                  # Traitement post-processing
│   ├── typeHolo.py                         # Types/classes hologramme
│   └── ...
│
├── [Deep learning / Machine Learning]
│   ├── test_UNET3D.py                      # ⭐ 3D U-Net (amélioré)
│   ├── CCL3D.py
│   ├── deep_segmentation_IA.py
│   ├── save_test_UNET3D.py
│   ├── pipeline_holotracker_locate_simple.py
│   └── ...
│
├── [Fichiers utils]
│   ├── focus.py
│   ├── detection_param.json
│   ├── holo_param.json
│   ├── parameters_simu_bact.json
│   └── ...
│
└── [Fichiers documentation racine]
    ├── QUICK_START.md                      # ← START HERE! Guide de démarrage
    ├── PROJECT_STRUCTURE.md                # Ce fichier
    ├── README.md                           # Historique du projet
    ├── LICENCE
    └── .git/, .gitignore
```
│   ├── config_bacteria_list.json           (copié dans simu holo/configs/)
│   ├── config_sphere_random.json           (copié dans simu holo/configs/)
│   ├── config_sphere_list.json             (copié dans simu holo/configs/)
│   ├── config_template.json                (copié dans simu holo/configs/)
│   ├── config_*.json (autres paramètres)
│   └── parameters_simu_bact.json           (GUI)
│
├── [Documentation obsolète]
│   ├── CONFIG_GUIDE.md                     (copié dans simu holo/docs/)
│   ├── CONFIG_SAVE_OPTIONS.md              (copié dans simu holo/docs/)
│   ├── IMPROVEMENTS_SUMMARY.md             (copié dans simu holo/docs/)
│   ├── REFACTOR_JSON_CONFIG.md             (copié dans simu holo/docs/)
│   ├── run_examples.sh                     (copié dans simu holo/examples/)
│   └── ...
│
├── README.md                               # Documentation racine du projet
├── LICENCE
└── .git/                                   # Historique Git
```

## Principes d'organisation

### 1. **Séparation des fonctionnalités**

#### `libs/` - Modules centralisés 📦
- **Contenu** : Tous les modules de base du projet
  - `simu_hologram.py` : Génération hologrammes et objets
  - `propagation.py` : Propagation onde (spectre angulaire)
  - `traitement_holo.py` : Post-processing hologrammes
  - `typeHolo.py` : Définitions types (Bacterie, Sphere, etc.)
  - `CCL3D.py` : Composantes connexes 3D
  - `focus.py` : Critères de focus
- **Imports** : Utilisé par tous les scripts via `from libs.module import *`
- **Avantage** : Centralisation, pas de duplication, imports cohérents

#### `simu holo/` - Simulation par configuration JSON
- ✓ Approche moderne avec fichiers de configuration
- ✓ Documentation complète et centralisée  
- ✓ Exemples pratiques
- ✓ Options de sauvegarde flexibles
- **À utiliser pour** : Nouvelles simulations, production, recherche

#### `simu bact GUI/` - Interface utilisateur
- GUI intuitive pour les utilisateurs non-techniques
- Configuration graphique directe
- Traitement en arrière-plan
- **À utiliser pour** : Tests interactifs, configuration simple

#### `localisation_pipeline/` - Pipelines de traitement
- Scripts pour tester la reconstruction et localisation
- `pipeline_holotracker_locate_simple.py` : Pipeline éducatif complet
- `main_reconstruction_volume.py` : Reconstruction volumétrique simple
- **À utiliser pour** : Tests, validation, enseignement

#### `deep_learning_segmentation/` - Deep Learning
- Scripts d'entraînement et test U-Net 3D
- Segmentation volumétrique supervisée
- Métriques et évaluation
- **À utiliser pour** : Recherche IA, comparaison avec méthodes classiques

### 2. **Flux de travail recommandé**

```
Utilisateur
    ↓
    ├─→ Nouvelle simulation?
    │   └─→ simu holo/
    │       1. Modifier configs/config_*.json
    │       2. python main_simu_hologram.py configs/config_*.json
    │       3. Résultats → simu_bacteria/YYYY_MM_DD_HH_MM_SS/
    │
    ├─→ Configuration interactive?
    │   └─→ simu bact GUI/
    │       1. python simulation_gui.py
    │       2. Configurer via interface
    │       3. Lancer simulation
    │
    ├─→ Test pipeline classique?
    │   └─→ localisation_pipeline/
    │       1. Copier un hologramme test (simu_holo_test.bmp)
    │       2. python pipeline_holotracker_locate_simple.py
    │       3. Résultats → result.csv
    │
    └─→ Deep learning?
        └─→ deep_learning_segmentation/
            1. Générer données avec simu holo/
            2. python split_data.py
            3. python train_UNET3D.py
            4. python test_UNET3D.py
```

### 3. **Système d'imports unifié**

Tous les scripts utilisent maintenant le système d'imports basé sur `libs/` :

```python
import sys
import os

# Ajouter la racine du projet au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Imports depuis libs/ (avec préfixe)
from libs.traitement_holo import *
from libs import propagation as propag
from libs.CCL3D import *
from libs import focus
```

**Imports internes dans libs/** : Les modules de `libs/` utilisent des imports relatifs :

```python
# Dans libs/propagation.py
from . import typeHolo
from .traitement_holo import *
```

**Avantages** :
- Imports cohérents dans tout le projet
- Pas de confusion sur l'origine des modules
- Facilite le debugging et la maintenance
- Structure claire et professionnelle

### 4. **Organisation des fichiers**

#### Données générées
- `simu_bacteria/YYYY_MM_DD_HH_MM_SS/` : Résultats simulations bactéries
- `simu_sphere/YYYY_MM_DD_HH_MM_SS/` : Résultats simulations sphères
- Sous-dossiers : `holograms/`, `positions/`, `data_holograms/`

#### Configurations
- `simu holo/configs/` : Configurations JSON pour simulations
- `deep_learning_segmentation/config_*.json` : Configs ML
- `simu bact GUI/parameters_*.json` : Configs GUI

#### Documentation
- Racine : README.md, QUICK_START.md, PROJECT_STRUCTURE.md
- `simu holo/docs/` : Documentation simulation
- `libs/README.md` : Documentation modules

## Migration et compatibilité

### Structure actuelle (Février 2026)

Le projet a été réorganisé pour une meilleure maintenabilité :

**Avant** :
- Modules éparpillés à la racine
- Imports directs sans namespace
- Scripts de test mélangés avec les modules

**Après** :
- `libs/` : Tous les modules centralisés
- `localisation_pipeline/` : Scripts de test/pipeline séparés
- Imports avec préfixe `libs.`
- Structure claire et modulaire

### Compatibilité

✅ Les anciens scripts dans `simu holo/legacy/` fonctionnent toujours  
✅ Les scripts racine ont été migrés vers les bons répertoires  
✅ Le système d'imports est unifié et documenté  
✅ La documentation a été mise à jour

## Avantages de cette organisation

| Aspect | Avant | Après |
|--------|-------|-------|
| **Clarté** | Code mélangé | Séparation claire |
| **Maintenabilité** | Imports confus | Système unifié avec `libs/` |
| **Navigation** | Difficile | Structure logique |
| **Documentation** | Dispersée | Centralisée et à jour |
| **Extensibilité** | Limité | Modulaire et scalable |
| **Collaboration** | Confusion | Rôles et responsabilités clairs |
- ✓ Exemples pratiques
- ✓ Options de sauvegarde flexibles
- **À utiliser pour**: Nouvelles simulations, production, recherche

#### `simu bact GUI/` - Interface utilisateur
- GUI intuitive pour les utilisateurs non-techniques
- Configuration graphique directe
- Traitement en arrière-plan
- **À utiliser pour**: Tests interactifs, configuration simple

#### Root - Modules de base
- `simu_hologram.py`: Classes et fonctions de simulation
- `propagation.py`: Propagation d'ondes lumineuses
- `traitement_holo.py`: Traitement d'images
- Importés par tous les scripts

#### Anciens scripts - Compatibilité
- Conservés pour compatibilité descendante
- Peuvent être supprimés une fois la migration complète

### 2. **Flux de travail recommandé**

```mermaid
Utilisateur
    ↓
    ├─→ Nouvelle simulation?
    │   └─→ simu holo/
    │       1. Modifier configs/config_*.json
    │       2. python main_simu_hologram.py configs/config_*.json
    │       3. Résultats → simu_bacteria/YYYY_MM_DD_HH_MM_SS/
    │
    ├─→ Configuration interactive?
    │   └─→ simu bact GUI/
    │       1. python simulation_gui.py
    │       2. Configurer via interface
    │       3. Lancer simulation
    │
    └─→ Deep learning?
        └─→ Root directory
            1. python test_UNET3D.py
            2. Utiliser données de simu holo/
```

### 3. **Chemins d'accès et imports**

Les scripts `main_simu_hologram.py` et `generate_config.py` sont à la racine mais utilisés depuis `simu holo/`.

**Option A: Symbolic links** (Unix/Linux/macOS)
```bash
cd simu holo/
./setup.sh  # Crée les liens symboliques
python main_simu_hologram.py configs/config_bacteria_random.json
```

**Option B: Chemins relatifs** (Windows/partout)
```bash
cd simu holo/
python ../main_simu_hologram.py configs/config_bacteria_random.json
```

**Option C: À partir de la racine**
```bash
python simu\ holo/main_simu_hologram.py simu\ holo/configs/config_bacteria_random.json
```

### 4. **Fichiers dupliqués (JSON + docs)**

Pour faciliter l'usage, les fichiers JSON et documentation sont dupliqués:
- **Source**: Racine du projet (historique, sauvegarde)
- **Actifs**: `simu holo/configs/` et `simu holo/docs/` (usage quotidien)

## Migration depuis l'ancienne structure

### Avant
```
root/
├── config_bacteria_random.json
├── main_simu_hologram.py
├── CONFIG_GUIDE.md
├── run_examples.sh
└── ... (mélangé avec autre code)
```

### Après
```
root/
├── simu holo/
│   ├── configs/
│   │   └── config_bacteria_random.json
│   ├── docs/
│   │   └── CONFIG_GUIDE.md
│   ├── examples/
│   │   └── run_examples.sh
│   └── main_simu_hologram.py (lien)
│
└── [fichiers source toujours à la racine pour imports]
```

## Checklist de migration

- ✓ Créer `simu holo/` avec sous-dossiers
- ✓ Copier fichiers JSON dans `configs/`
- ✓ Copier documentation dans `docs/`
- ✓ Copier examples dans `examples/`
- ✓ Créer README.md principal
- ✓ Créer setup.sh pour liens symboliques
- ⏳ Mettre à jour chemins d'import dans les scripts
- ⏳ Tester depuis le dossier `simu holo/`
- ⏳ Documenter la nouvelle organisation

## Support des anciennes structures

Pour maintenir la compatibilité:
1. Les fichiers racine ne sont pas supprimés
2. Les anciens scripts continuent de fonctionner
3. La migration est graduelle et optionnelle
4. Les utilisateurs existants ne sont pas affectés

## Avantages de cette organisation

| Aspect | Avant | Après |
|--------|-------|-------|
| **Clarté** | Code mélangé | Séparation claire |
| **Maintenabilité** | Difficile de naviguer | Structure logique |
| **Documentation** | Dispersée | Centralisée |
| **Configurations** | Fichiers racine | Dossier dédié |
| **Extensibilité** | Limité | Modulaire |
| **Collaboration** | Confusion possible | Rôles clairs |

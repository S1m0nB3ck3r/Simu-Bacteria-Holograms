# Structure du dépôt Simu-Bacteria-Holograms

État au 2026-10-19. Ce document dit, pour chaque répertoire, **à quoi il sert, ce qu'il contient, s'il est
encore actif**, et ce qui est versionné dans git. Il remplace la version de mars 2026, qui décrivait des
répertoires renommés ou supprimés depuis.

## 1. Vue d'ensemble

Le dépôt simule des hologrammes inline de bactéries pour mettre au point leur détection 3D. Il alimente aussi
HOLOTRACKER, le pipeline classique, qui partage la même physique (`libs/`).

```
                 libs/  (physique commune : propagation, focus, insertion des objets, composantes connexes)
                   │
   ┌───────────────┼──────────────────────────────┐
   ▼               ▼                              ▼
 SIMULER        DÉTECTER                       ANALYSER
 simu holo script/          localisation_pipeline/  (classique)     stats/  → FOCUS_REPORT/, SEGMENTATION_REPORT/,
 simu holo bact with GUI/   deep_learning_segmentation/ (U-Net 3D)            TRAJECTORY_REPORT/, output/
 Hybrid_REAL_SIMU_HOLOGRAMS/
 deep_learning_segmentation/generate_*.py
   │
   ▼
 DONNÉES : REAL HOLO FILM/ (film réel)   SIMU HOLO FILM/ (films simulés par le GUI)
```

## 2. Par où commencer

| Je veux… | Aller dans | Point d'entrée |
|---|---|---|
| détecter et segmenter des bactéries sur un film | `deep_learning_segmentation/` | `GUI_inference.py` |
| entraîner ou réentraîner le U-Net 3D | `deep_learning_segmentation/` | `GUI_train.py` |
| simuler un hologramme à partir d'un JSON | `simu holo script/` | `main_simu_hologram.py <config.json>` |
| générer un jeu de données avec une interface | `simu holo bact with GUI/` | `simulation_gui.py` |
| simuler des bactéries sur le film réel (fond réel) | `Hybrid_REAL_SIMU_HOLOGRAMS/` | `GUI_real_simulation.py` |
| localiser par la méthode classique (seuil, CCL) | `localisation_pipeline/` | `pipeline_holotracker_locate_*.py` |
| comparer simulé et réel, mesurer un critère de focus | `stats/` | `GUI_holo_stats.py`, `stat_report_*.py` |
| connaître les faits physiques établis | racine | `CLAUDE.md` (section Physique) |

## 3. Code

### `libs/` — physique commune *(actif, versionné)*
Modules importés par tous les autres répertoires, et partagés avec HOLOTRACKER.
- `propagation.py` : spectre angulaire, Fresnel, Rayleigh-Sommerfeld, propagation volumique.
- `simu_hologram.py` : insertion des bâtonnets et des sphères dans le volume masque (GPU), illumination, déphasage
  par tranche (BPM), bruit capteur. **Longueur du bâtonnet = longueur bout à bout depuis le 2026-10-18.**
- `focus.py` : critères de focus (TENEGRAD, SUM_OF_VARIANCE…).
- `CCL3D.py` : composantes connexes 3D.
- `traitement_holo.py`, `typeHolo.py` : intensité, normalisation, entrées/sorties ; classes `Bacterie`, `Sphere`.

### `simu holo script/` — simulateur piloté par JSON *(actif, versionné)*
- **Point d'entrée** : `main_simu_hologram.py <config.json>`, avec les configurations `config_bacteria_random.json`,
  `config_bacteria_list.json`, `config_sphere_*.json`, `config_1bact_3_illuminations.json` et
  `config_template.json`.
- **Méthode** : BPM sur toutes les tranches du volume.
- **Sorties** :
  - l'hologramme ;
  - le volume propagé ;
  - le `bin_volume`, masque binaire enregistré **après l'inversion de l'axe Z** ;
  - les positions.

  Elles vont dans `simu_bacteria/`, non versionné.
- `generate_config.py` génère des configurations ; `docs/` décrit les options ; `examples/run_examples.sh`.
- **Doublons historiques** :
  - `main_simu_hologram_old.py` : copie à 96,8 % du script principal ;
  - `main_simu_hologram_triple_illumination.py` : variante aux paramètres codés en dur ;
  - `legacy/` : les 4 anciens scripts séparés, avant leur fusion.

### `simu holo bact with GUI/` — génération de jeux par interface *(actif, versionné)*
- `simulation_gui.py` écrit `parameters_simu_bact.json`, puis lance `processor_simu_bact.py`, qui produit les
  hologrammes ; `visualizer_gui.py` affiche côte à côte l'hologramme, le volume segmenté et le volume propagé.
- Sorties dans `SIMU HOLO FILM/<date>/`.
- `processing_status.json`, `processing_result.json`, `test_interruption_config.json` : fichiers d'état et de test.

### `Hybrid_REAL_SIMU_HOLOGRAMS/` — simulations sur le film réel *(actif, versionné sauf `output/`)*
- `real_simulation.py` + `config_real_simulation.json` + `GUI_real_simulation.py`.
- **Chaîne** :
  1. retrait de la moyenne du film réel ;
  2. nettoyage des bactéries détectées par HOLOTRACKER (CLEAN, image jumelle comprise) ;
  3. bactéries simulées aux positions des trajectoires.
- **Sert aussi au U-Net** : ses fonctions `simulate_frame` (BPM sur les seules tranches occupées) et
  `bacteria_instance_masks` (masque d'instance de chaque bactérie) sont le simulateur utilisé par
  `deep_learning_segmentation/`.
- **Calibration de Δn** sur le film réel : 0,046.
- `output/bacterias_2026_10_17_16_08_17/` (≈ 400 Mo) : un traitement complet du film, avec la moyenne, le fond
  nettoyé, les bactéries nettoyées, les bactéries simulées seules et sur fond, et `ground_truth.csv`. Il date
  d'avant la correction de la longueur du bâtonnet.

### `localisation_pipeline/` — méthode classique *(actif, versionné)*
- **Trois variantes** de la chaîne propagation → focus → seuil → CCL3D :
  - `pipeline_holotracker_locate_simple.py` : seuil global ;
  - `pipeline_holotracker_locate_local_threshold.py` : seuil local par mailles ;
  - `pipeline_holotracker_locate_local_maxima.py` : maxima locaux, la meilleure (rapports du 2026-10-13).
- `main_reconstruction_volume.py` : exemple de reconstruction sur `simu_holo_test.bmp`.

### `deep_learning_segmentation/` — U-Net 3D *(actif ; code et modèles versionnés, données non)*
Positions 3D, puis segmentation d'instance de chaque bactérie, en deux étages. Détail complet dans son
[README.md](deep_learning_segmentation/README.md), schémas dans `PIPELINE_UNET3D.pdf`, résultats dans
`RAPPORT_UNET3D.md`.
- **Modèles retenus** : `models/stage1`, `models/stage2`, entraînés sur de la simulation sans fond.
- **Données** : `data/seg_sim` (jeu d'entraînement), `data/seq` (films simulés en mouvement), non versionnées.
- **Interfaces** : `GUI_inference.py` (un clic sur une bactérie montre ses coupes XY, XZ, YZ) et
  `GUI_train.py`.
- `experiences/` : scripts des expériences passées ; `resultats/` : benchmarks des modèles retenus.

### `stats/` — statistiques et rapports *(actif, versionné)*
- **Comparaison simulé / réel** : `holo_stats.py` (moteur) et `GUI_holo_stats.py`, avec `holo_stats_real.yaml`
  et `holo_stats_simu.yaml`. Statistiques hologramme, volume et objets.
- **Rapports de mesure** : `stat_report_*.py`, chacun avec son YAML, sur le moteur commun `report_common.py` :

  | Script | Mesure | Sortie |
  |---|---|---|
  | `stat_report_distance_Z_bact.py`, `_sphere.py`, `stat_report_orientation_bact.py` | erreur de position selon z ou l'orientation | `output/` |
  | `stat_report_orientation_focus.py` | critères de focus selon l'orientation | `FOCUS_REPORT/1_bacteria/` |
  | `stat_report_dense_focus.py` | critères de focus en milieu dense | `FOCUS_REPORT/DENSE/` |
  | `stat_report_dense_segmentation.py` | 5 méthodes de segmentation en milieu dense | `SEGMENTATION_REPORT/` |
  | `stat_report_trajectories.py` | dynamique des bactéries du film réel | `TRAJECTORY_REPORT/` |

## 4. Données

### `REAL HOLO FILM/` — film réel de référence *(versionné)*
- `bacterias/` : 100 hologrammes BMP 1024 × 1024 (pixel objet 0,175 µm) :
  - `RESULT_20261016_172327.csv` : détections d'HOLOTRACKER ;
  - `TRAJECTORIES_20261016_173113.csv` : trajectoires (57) ;
  - `mean/` : moyennes du film ;
  - `result/` : **inférence du U-Net du 2026-10-18**, sur 0–100 µm. Il contient `positions_unet3d.csv` aux colonnes
    du RESULT d'HOLOTRACKER, une BMP par image avec les localisations, et les masques des instances.
- `polystyrene_spheres/` : 100 hologrammes de billes, pas encore exploités.

### `SIMU HOLO FILM/` — films simulés par le GUI *(28 Go ; 172 fichiers versionnés)*
- **Deux tirages du GUI de simulation**, `2026_09_21_14_51_58/` et `2026_10_12_17_12_44/` (14 Go chacun) :
  hologrammes, volumes propagés, `binary_volume/`, positions, `parameters_simu_bact.json`.
- **Anciens paramètres** : Δn = 0,005 (signal 8,6 fois trop faible), longueurs de 1,2 à 2,5 µm, ancienne
  géométrie du bâtonnet.

## 5. Rapports et résultats

| Emplacement | Contenu | Date |
|---|---|---|
| `RAPPORT_DETECTION_SEUIL_FOCUS.md` | orientation, type de seuil, critère de focus en milieu dense | 2026-10-12/13 |
| `FOCUS_REPORT/1_bacteria/RAPPORT_FOCUS.md` | erreur en Z selon l'orientation, par critère de focus | 2026-10-13 |
| `FOCUS_REPORT/DENSE/RAPPORT_DENSE.md` | critères de focus de 5 à 200 bactéries (+ hologrammes simulés N005 à N200) | 2026-10-13 |
| `SEGMENTATION_REPORT/RAPPORT_SEGMENTATION.md` | 5 méthodes classiques de segmentation en milieu dense | 2026-10-13 |
| `UNET3D_motivation.md` | pourquoi un U-Net 3D, budget de temps | 2026-10-13 |
| `TRAJECTORY_REPORT/` | dynamique du film réel (sans rapport rédigé ; il est dans `RAPPORT_UNET3D.md`, section 7) | 2026-10-18 |
| `deep_learning_segmentation/RAPPORT_UNET3D.md` | U-Net 3D : détection, segmentation, film réel, temps | 2026-10-18 |
| `output/` | sorties brutes des `stat_report_*` (non versionné) | 2026-10-12/13 |

Les rapports du 2026-10-12 et du 2026-10-13 utilisent des bâtonnets plus longs que demandé : ils mesuraient L + T/2
bout à bout. Leurs conclusions qualitatives tiennent, les valeurs absolues portent sur ces bâtonnets.

## 6. Documents d'échange avec HOLOTRACKER (racine)

| Fichier | Sens | Contenu |
|---|---|---|
| `CLAUDE_HOLOTRACKER_TO_DO.md` | ce dépôt → HOLOTRACKER | constats mesurés et pistes de correction (orientation, seuil, densité) |
| `CLAUDE_HOLOTRACKER_DEBUG_TO_DO.md` | ce dépôt → HOLOTRACKER | défauts relevés dans le code d'HOLOTRACKER |
| `SIMULATION_TO_DO.md` | HOLOTRACKER → ce dépôt | mesure demandée au simulateur (2026-10-13) |
| `CLAUDE.md` | instructions de travail | règles, pièges de la machine, faits physiques établis, état du code |

## 7. Hors du flux principal

- **`airel_holo/`** (100 Mo, versionné) : projet U-Net 3D **antérieur et indépendant**, importé en août 2026
  (branche « airel »). Il a ses propres scripts (`main.py`, `train.py`, `model.py`, `dataset.py`), ses
  configurations YAML et ses figures. `deep_learning_segmentation/` n'en dépend pas.
- **`archive/`** (non versionné) : instantanés complets du code après chaque modification validée, avec leur
  `MODIFICATIONS.md` (règle de `CLAUDE.md`).
- **`result.csv`** (racine, versionné) : sortie de localisation ancienne (colonnes `i_image`, `baryX/Y/Z`,
  `nb_pix`), d'origine non documentée.
- **`README.md`, `QUICK_START.md`** (mars 2026) : **obsolètes**. Ils renvoient à `simu holo/`, qui n'existe plus, et
  à l'ancien pipeline U-Net.
- **`requirements.txt`**, **`LICENCE`** (GPL v3).

## 8. Ce que git suit

Non versionnés (`.gitignore`) :
- `output/` et `archive/` ;
- `simu holo script/simu_bacteria/` ;
- dans `deep_learning_segmentation/` : `data/`, `runs/`, `resultats/` et `runs_inference/` ;
- les réglages mémorisés des interfaces (`*_params.json`) ;
- les fichiers `*.npy`.

Le motif `CLAUDE*` exclut aussi `CLAUDE.md` et les deux documents `CLAUDE_HOLOTRACKER_*.md`, qui ne sont donc
**pas versionnés**.

## 9. Points de confusion, et propositions de rangement (rien n'a été modifié)

1. **`README.md` et `QUICK_START.md`** sont périmés : les réécrire, ou les réduire à un renvoi vers ce document.
2. **Simulateurs en double** : `main_simu_hologram_old.py`, `main_simu_hologram_triple_illumination.py` et
   `simu holo script/legacy/` doublent le script principal. Ils pourraient être supprimés, puisqu'ils restent
   dans git.
3. **Trois simulateurs coexistent** :
   - `simu holo script/` : JSON, toutes les tranches, masque binaire inversé en Z ;
   - `simu holo bact with GUI/` : interface ;
   - `Hybrid_REAL_SIMU_HOLOGRAMS/` : tranches occupées seulement et masques d'instance, utilisé par le U-Net.

   Ils partagent `libs/`, mais n'ont ni les mêmes paramètres par défaut ni les mêmes sorties.
4. **`SIMU HOLO FILM/`** (28 Go) date d'avant la recalibration de Δn et la correction de la longueur : à
   supprimer, ou à régénérer si on en a encore besoin.
5. **`Hybrid_REAL_SIMU_HOLOGRAMS/output/`** (400 Mo) : son seul usage, les fonds réels pour l'entraînement, a été
   abandonné.
6. **`airel_holo/`** : à garder comme référence historique, ou à sortir du dépôt.
7. **`result.csv`** et **`output/`** : sorties anciennes, sans document qui les décrive.
8. **`CLAUDE*` dans le `.gitignore`** : les documents d'échange avec HOLOTRACKER ne sont pas versionnés ; à
   vérifier si c'est voulu.

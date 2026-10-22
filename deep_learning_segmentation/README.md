# deep_learning_segmentation — U-Net 3D : positions 3D et segmentation d'instance des bactéries

Chaîne en deux étages sur des hologrammes inline 1024 × 1024 (pixel 0,175 µm, jusqu'à 207 µm de profondeur) :
**étage 1**, U-Net 3D sur le volume reconstruit grossier, donne les positions ; **étage 2**, U-Net 3D sur une
fenêtre fine autour de chaque position, donne le masque de chaque bactérie (instances), en deux tours pour
séparer les bactéries collées. Entraînement sur hologrammes **100 % simulés, sans fond** (fond plat + bruit
capteur) : c'est ce qui généralise au film réel. Schémas et mode d'emploi : [PIPELINE_UNET3D.pdf](PIPELINE_UNET3D.pdf)
(état du 2026-10-22) ; résultats : [RAPPORT_UNET3D.md](RAPPORT_UNET3D.md) ; normalisations de l'entrée :
`resultats/benchmark_grid_B/RAPPORT_NORMALISATION_ETAGE1.pdf`.

## Utiliser

```bash
python deep_learning_segmentation/GUI_generate_dataset.py   # jeu simulé : BMP + instances + CSV + parameters.yaml
python deep_learning_segmentation/GUI_train.py              # entraîner les deux étages sur ce jeu
python deep_learning_segmentation/GUI_inference.py          # film réel ou simulé -> positions, instances, CSV
python deep_learning_segmentation/inference_pipeline.py "REAL HOLO FILM/bacterias" <sortie> --mean_removal division --z_max 100
```

Aides détaillées (chaque champ des GUI, chaque option des scripts) : [HELP_GUI_GENERATE_DATASET.pdf](HELP_GUI_GENERATE_DATASET.pdf),
[HELP_TRAIN.pdf](HELP_TRAIN.pdf), [HELP_INFERENCE.pdf](HELP_INFERENCE.pdf) ; les quatre PDF se regénèrent à partir du code
(`doc_tools/make_diagrams.py`, `make_pipeline_pdf.py`, `make_help_pdfs.py`).

Réglages : leurs valeurs par défaut sont des constantes en tête de `train.py`, `train_stage2.py` et
`inference_pipeline.py` (reconstruction, cibles, fenêtres, post-traitement) ; chacun est une option de la ligne de
commande (`--help` l'explique) et un champ des GUI (infobulle). Les valeurs d'entraînement sont enregistrées avec le
modèle et reprises par défaut à l'inférence.

### Générer un jeu (`GUI_generate_dataset.py`, moteur `simu_dataset.py`)

Mêmes paramètres que `simu holo bact with GUI/simulation_gui.py`, plus le nombre d'hologrammes par jeu, la plage
du nombre de bactéries, la part et la distance des bactéries rapprochées, le test de non-recouvrement (une bactérie qui en touche une autre est écartée) et la plage d'indice (défauts calibrés : Δn 0,032–0,060,
1,8 × 0,6 µm ± 10 %, pas 0,25 µm). Même physique que `simulation_gui.py` (BPM, mêmes noyaux) ; le vide est franchi
en une propagation au lieu de plan par plan (identique à 10⁻⁵ près), l'éclairage n'est pas apodisé sur la bordure
(écart ≤ 0,001, 15 fois sous le bruit capteur), l'orientation est isotrope (cos φ uniforme).

| Fichier | Contenu |
|---|---|
| `parameters.yaml` | tous les paramètres (m, degrés), valeurs dérivées (pixel objet, plage z, Δn), formats, état de la génération |
| `<split>/holograms/holo_XXXXX.bmp` | hologramme 8 bits, gris = I / I0 × gris du fond (128 par défaut) |
| `<split>/positions/holo_XXXXX.csv` | une bactérie par ligne : x, y, z (m, z = distance au capteur), θ, φ, longueur, épaisseur, indices de voxel |
| `<split>/instances/holo_XXXXX.npz` | volume de segmentation d'instance creux (boîte de chaque bactérie, fraction occupée, grille du simulateur) ; `simu_dataset.instance_volume()` reconstruit le volume d'étiquettes complet |
| `<split>/index.csv` | un hologramme par ligne : N, Δn, σ du bruit, part saturée du BMP |
| options | `holograms_tiff/` (32 bits), `instance_volume/` (étiquettes uint16 complètes, 1,6 Go par hologramme en 1024 × 1024 × 800) |

### Entraîner (`GUI_train.py`, `train.py`, `train_stage2.py`)

Le dossier du jeu sert aux deux étages ; la taille, l'optique et le pas des masques viennent de `parameters.yaml`.
Le volume reconstruit couvre par défaut la profondeur du jeu (distance volume-caméra → distance + plans × pas ;
`--z_min_um`, `--z_max_um` pour la changer). L'ancien format `.npz` de `generate_dataset.py` reste lu.

Normalisation du volume reconstruit avant l'étage 1 (`--norm`, enregistrée avec le modèle et reprise à l'inférence) :

| `--norm` | Centre et échelle | Calculés sur |
|---|---|---|
| `filmmad` (défaut de `GUI_train.py`) | médiane ; 1,4826 × MAD | 32 hologrammes du film |
| `filmstd` | moyenne ; écart-type | 32 hologrammes du film |
| `globalmad` | médiane ; 1,4826 × MAD | le volume de chaque hologramme (+ 65 ms) |
| `globalstd` | moyenne ; écart-type | le volume de chaque hologramme |
| `localmad` | médiane ; MAD par maille, interpolées | mailles de 13 × 11,2 µm (`--mad_mesh_um`) |
| `fixed` (défaut de `train.py`, modèle retenu) | 0 ; 0,015 × √(part du spectre filtré) | rien : constante du bruit capteur |

Benchmark du 2026-10-22 : à nombre de détections égal sur le film réel, toutes se valent (rappel 0,89–0,94 à
100 détections par image). Le fond du volume suit la densité (× 2,2 de 100 à 400 bactéries) : une normalisation
par film reste constante d'une image à l'autre et s'adapte au bruit d'un autre montage, d'où le défaut.

### Inférer (`GUI_inference.py`, `inference_pipeline.py`)

- **Suppression de l'hologramme moyen** : division I / M (comme HOLOTRACKER), soustraction (I − M) / ⟨M⟩ + 1, ou
  aucune (division par la moyenne de l'image, ou image déjà à fond 1).
- **Reconstruction** (onglet du même nom) : pixel caméra, grossissement, longueur d'onde, indice, z min / z max,
  pas Z, décimation, filtre ; vides, ce sont les valeurs d'entraînement, et tout écart est signalé.
- **Seuil de l'étage 1** (onglet Réglages) : vide, celui de la validation simulée. C'est le réglage qui pèse le plus
  sur le nombre de détections d'un film réel.
- **Sorties** : `positions_unet3d.csv` (6 premières colonnes = celles du `RESULT_*.csv` d'HOLOTRACKER, positions
  en m, puis axe, longueur, épaisseur, volume, score), `<image>_localisations.bmp`, `instances/*.npz`,
  `inference_params.json` (paramètres, optique, reconstruction). Dans le GUI, un clic sur une bactérie ouvre ses
  coupes de segmentation XY, XZ, YZ.

## Structure

| Élément | Rôle |
|---|---|
| **Modèles** | |
| `models/stage1/` | étage 1 retenu (échelle fixe, entraîné le 2026-10-18 sur l'ancien jeu `data/seg_sim`) : `model_best.pt`, `config.json`, `history.json`, `train_summary.json` |
| `models/stage2/` | étage 2 retenu (même jeu, masques d'instance du simulateur) |
| **Cœur** | |
| `holo_physics.py` | reconstruction par spectre angulaire (décimation ×2 par le spectre), normalisations (MAD / écart-type de l'hologramme ou du film, MAD locale), cibles de l'étage 1 |
| `unet3d.py` | U-Net 3D résiduel (base 16, 3 niveaux, 1,42 M paramètres), repli de la BatchNorm à l'inférence |
| `losses.py` | loss focale CenterNet (retenue), MSE pondérée, BCE + Dice, focale + Tversky |
| `detection.py` | inférence par tuiles, pics, appariement hongrois, méthode classique de référence (maxima SoV) |
| `seg_stage2.py` | étage 2 : refocalisation locale exacte, cibles, loss, inférence en deux tours, dédoublonnage, mesures |
| **Données** | |
| `GUI_generate_dataset.py`, `simu_dataset.py` | génération d'un jeu (BMP + volume d'instance + positions CSV + `parameters.yaml`) et sa lecture par l'entraînement |
| `generate_dataset.py`, `gen_seg_sim_data.sh` | ancien format `.npz` : jeu `seg_sim` des modèles retenus (graines 401 à 404), paires rapprochées (`--splits pairs`) |
| `generate_sequences.py` | films 100 % simulés en mouvement (dynamique du film réel, `TRAJECTORY_REPORT/`), identités des bactéries |
| `datasets/` | jeux de `GUI_generate_dataset.py` : `dataset_sim_100um_N20-400`, `dataset_sim_200um_N20-400`, `dataset_sim_100um_N10-200` (≈ 950 Mo ; pas encore dans `.gitignore`) |
| **Entraînement** | |
| `train.py` | étage 1 (patchs 48 × 96 × 96, carte des centres, normalisation `--norm`) |
| `train_stage2.py` | étage 2 (fenêtres 32 × 40 × 40, grille fine au pas des masques du jeu) |
| **Évaluation** | |
| `benchmark.py` | étage 1 contre la méthode classique : jeux simulés (`--datasets`), film réel (trajectoires HOLOTRACKER), balayage du seuil (`threshold_sweep.csv`), paires, temps |
| `benchmark_seg.py` | chaîne complète (`--data`, `--pairs_dir`) : PQ, IoU par bactérie, second tour, forme, paires, temps, volume segmenté |
| `plot_seg_examples.py` | bactéries segmentées en coupes XY, XZ, YZ |
| `resultats/` | benchmarks et journaux (non versionné) ; `benchmark_grid_B/` : étude des normalisations, son rapport et ses scripts de figures |
| `runs/` | entraînements (non versionné) ; les 11 modèles d'étage 1 de l'étude des normalisations |
| **Interfaces** | |
| `inference_pipeline.py` | inférence sur un film, export CSV (colonnes d'HOLOTRACKER) + instances `.npz` |
| `GUI_inference.py`, `GUI_train.py`, `GUI_generate_dataset.py` | interfaces (Tk) |
| `gui_messages.py` | gestionnaire de messages : calcul hors de la boucle Tk, messages par paquets bornés, arrêt propre (fichier `STOP`) |
| **Documentation** | |
| `RAPPORT_UNET3D.md`, `RAPPORT_UNET3D_figures/` | rapport et figures |
| `PIPELINE_UNET3D.pdf` | structure, pipeline, normalisation, architecture du réseau, résultats, utilisation |
| `HELP_GUI_GENERATE_DATASET.pdf`, `HELP_TRAIN.pdf`, `HELP_INFERENCE.pdf` | aides : champs des GUI, options des scripts, sorties, conseils |
| `doc_tools/` | génération des PDF à partir du code (style commun, schémas, tableaux lus dans les GUI et les `argparse`) |
| `experiences/` | scripts des expériences du rapport (variantes de l'étage 1, fonds hybrides) ; leurs données et modèles ont été supprimés, à régénérer pour les rejouer |

Le dossier `data/` (ancien jeu `seg_sim`, films `seq`) a été supprimé ; les deux se régénèrent par
`gen_seg_sim_data.sh` et `generate_sequences.py`.

Grilles : volume grossier 0,35 × 0,35 × 1 µm ; masques et étage 2 sur la grille du simulateur
0,175 × 0,175 × 0,25 µm. Repère d'HOLOTRACKER : x = colonne, y = ligne, z = distance de refocalisation positive
(distance au capteur).

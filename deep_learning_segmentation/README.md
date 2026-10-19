# deep_learning_segmentation — U-Net 3D : positions 3D et segmentation d'instance des bactéries

Chaîne en deux étages sur des hologrammes inline 1024 × 1024 (pixel 0,175 µm, 0–207 µm de profondeur) :
**étage 1**, U-Net 3D sur le volume reconstruit grossier, donne les positions ; **étage 2**, U-Net 3D sur une
fenêtre fine autour de chaque position, donne le masque de chaque bactérie (instances), en deux tours pour
séparer les bactéries collées. Entraînement sur hologrammes **100 % simulés, sans fond** (fond plat + bruit
capteur) : c'est ce qui généralise au film réel. Schémas : [PIPELINE_UNET3D.pdf](PIPELINE_UNET3D.pdf) ; résultats :
[RAPPORT_UNET3D.md](RAPPORT_UNET3D.md).

## Utiliser

```bash
python deep_learning_segmentation/GUI_inference.py      # film réel ou simulé -> positions, instances, CSV
python deep_learning_segmentation/GUI_train.py          # générer des données, entraîner les deux étages
python deep_learning_segmentation/inference_pipeline.py "REAL HOLO FILM/bacterias" "REAL HOLO FILM/bacterias/result" --z_max 100
```

Sorties de l'inférence : `positions_unet3d.csv` (6 premières colonnes = celles du `RESULT_*.csv` d'HOLOTRACKER,
positions en m, puis axe, longueur, épaisseur, volume, score), `<image>_localisations.bmp` (hologramme
normalisé, bactéries entourées, couleur et valeur de z), `instances/*.npz` (masque fin de chaque bactérie),
`inference_params.json`. Dans `GUI_inference.py`, un clic sur une bactérie ouvre ses coupes de segmentation
XY, XZ, YZ. `--z_max` limite la recherche (seuls les plans utiles sont reconstruits).

## Structure

| Élément | Rôle |
|---|---|
| **Modèles** | |
| `models/stage1/` | étage 1 retenu (entraîné sur `data/seg_sim`) : `model_best.pt`, `config.json`, `history.json`, `train_summary.json` |
| `models/stage2/` | étage 2 retenu (même jeu, masques d'instance du simulateur) |
| **Cœur** | |
| `holo_physics.py` | reconstruction par spectre angulaire (décimation ×2 par le spectre), normalisation MAD locale (option), cibles de l'étage 1 |
| `unet3d.py` | U-Net 3D résiduel (base 16, 3 niveaux, 1,42 M paramètres), repli de la BatchNorm à l'inférence |
| `losses.py` | loss focale CenterNet (retenue), MSE pondérée, BCE + Dice, focale + Tversky |
| `detection.py` | inférence par tuiles, pics, appariement hongrois, méthode classique de référence (maxima SoV) |
| `seg_stage2.py` | étage 2 : refocalisation locale exacte, cibles, loss, inférence en deux tours, dédoublonnage, mesures |
| **Données** | |
| `generate_dataset.py` | hologrammes simulés (BPM du simulateur hybride), tailles ± x %, bactéries rapprochées, masques d'instance du simulateur |
| `gen_seg_sim_data.sh` | produit `data/seg_sim` (jeu d'entraînement actuel) |
| `generate_sequences.py` | films 100 % simulés en mouvement (dynamique du film réel, `TRAJECTORY_REPORT/`), identités des bactéries |
| `data/seg_sim/` | train 240 / val 24 / test 40 + 48 hologrammes de paires, 200 µm, N 20–400 (non versionné) |
| `data/seq/` | 14 films (train 8 × 40, val 2 × 20, test 4 × 30 images), même format + `trajectories.csv` (non versionné) |
| **Entraînement** | |
| `train.py` | étage 1 (patchs 48 × 96 × 96, carte des centres) |
| `train_stage2.py` | étage 2 (fenêtres 32 × 40 × 40) |
| **Évaluation** | |
| `benchmark.py` | étage 1 contre la méthode classique : jeux simulés, film réel (trajectoires HOLOTRACKER), paires, temps, hologrammes annotés |
| `benchmark_seg.py` | chaîne complète : PQ, IoU par bactérie, second tour, forme, paires, temps, volume segmenté |
| `plot_seg_examples.py` | bactéries segmentées en coupes XY, XZ, YZ |
| `resultats/` | sorties des benchmarks des modèles retenus et journaux (non versionné) |
| **Interfaces** | |
| `inference_pipeline.py` | inférence sur un film, export CSV (colonnes d'HOLOTRACKER) + instances `.npz` |
| `GUI_inference.py`, `GUI_train.py` | interfaces (Tk) |
| `gui_messages.py` | gestionnaire de messages : calcul hors de la boucle Tk, messages par paquets bornés, arrêt propre (fichier `STOP`) |
| **Documentation** | |
| `RAPPORT_UNET3D.md`, `RAPPORT_UNET3D_figures/` | rapport et figures |
| `PIPELINE_UNET3D.pdf` | structure, pipeline, architecture du réseau |
| `experiences/` | scripts des expériences du rapport (variantes de l'étage 1, fonds hybrides) ; leurs données et modèles ont été supprimés, à régénérer pour les rejouer |
| `runs/` | sorties des nouveaux entraînements (non versionné) |

Grilles : volume grossier 0,35 × 0,35 × 1 µm (208 plans) ; masques et étage 2 sur la grille du simulateur
0,175 × 0,175 × 0,25 µm. Repère d'HOLOTRACKER : x = colonne, y = ligne, z = distance de refocalisation positive.

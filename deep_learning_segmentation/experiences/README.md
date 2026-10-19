# Expériences du rapport (archives de scripts)

Scripts qui ont produit la section 6 de `../RAPPORT_UNET3D.md`. Leurs données (`data/hybrid`, `data/sim`,
`data/seg`, fonds nettoyés) et leurs modèles ont été supprimés le 2026-10-18 : ils sont
antérieurs à la correction de la longueur du bâtonnet et de Δn, ou utilisent des fonds réels (hybrides), abandonnés
au profit de la simulation sans fond. Pour rejouer une expérience, régénérer d'abord ses données.

| Script | Expérience |
|---|---|
| `run_experiments.sh` | 9 variantes de l'étage 1 (loss, MAD local, orientation, pas Z, taille, fond simulé) |
| `gen_seg_data.sh` | jeu de segmentation sur fonds hybrides |
| `config_backgrounds.json` | nettoyage du film réel pour les fonds hybrides (`Hybrid_REAL_SIMU_HOLOGRAMS/real_simulation.py`) |
| `run_benchmarks.sh` | benchmarks des runs de ces expériences |

Lancer depuis la racine du dépôt, par exemple `bash deep_learning_segmentation/experiences/run_experiments.sh`.

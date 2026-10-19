#!/usr/bin/env bash
# Jeux de la segmentation SANS FOND (fond plat + bruit capteur σ = 0,015), mêmes graines et paramètres que gen_seg_data.sh : 200 µm, tailles ± 10 % autour de 1,8 × 0,6 µm, pas BPM 0,25 µm,
# masques d'instance du simulateur ; 30 % des bactéries placées à 0,6–4 µm d'une autre (contacts).
# Lancer depuis la racine du dépôt.
set -eu
export PYTHONIOENCODING=utf-8
G="python -u deep_learning_segmentation/generate_dataset.py --background sim --z_range 2 200 --size_jitter 0.1 --sim_step 0.25 --save_masks --cluster_fraction 0.3"
D=deep_learning_segmentation/data/seg_sim
$G --out $D --splits train     --n_range 20 400 --counts 240 0 0 --seed 401
$G --out $D --splits val       --n_list 100 200 300 400 --counts 0 24 0 --seed 402
$G --out $D --splits test      --n_list 50 100 200 300 400 --counts 0 0 40 --seed 403
# paires rapprochées (latérales 1–6 µm, axiales 1–8 µm), fonds de test, avec masques
python -u deep_learning_segmentation/generate_dataset.py --background sim --size_jitter 0.1 --sim_step 0.25 --save_masks --out $D --splits pairs --seed 404
echo "=== données de segmentation prêtes"

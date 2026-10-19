#!/usr/bin/env bash
# Série d'entraînements du rapport RAPPORT_UNET3D.md (4000 itérations chacun, ~20 min sur Quadro RTX 3000).
# Lancer depuis la racine du dépôt : bash deep_learning_segmentation/experiences/run_experiments.sh
set -u
export PYTHONIOENCODING=utf-8
T="python -u deep_learning_segmentation/train.py --iters 4000"
LOG=deep_learning_segmentation/runs/logs
mkdir -p "$LOG"
run() { name=$1; shift; echo "=== $name : $*"; $T --name "$name" "$@" > "$LOG/$name.log" 2>&1; tail -n 3 "$LOG/$name.log" | head -n 1; }

# 1. loss (même architecture, même budget)
run centernet      --loss centernet
run wmse           --loss wmse
run bce_dice       --loss bce_dice
run focal_tversky  --loss focal_tversky
# 2. normalisation locale médiane / MAD (HOLOTRACKER)
run centernet_localmad  --loss centernet --norm localmad
# 3. tête d'orientation
run centernet_orient --loss centernet --orient_weight 1
# 4. variantes rapides (temps d'inférence)
run centernet_z2       --loss centernet --z_step 2
run centernet_z2_base8 --loss centernet --z_step 2 --base 8
# 5. écart de domaine : entraîné sur fond simulé, validé sur fond réel
run centernet_simtrain --loss centernet --train_dir deep_learning_segmentation/data/sim/train
echo "=== terminé"

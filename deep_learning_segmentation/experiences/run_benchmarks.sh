#!/usr/bin/env bash
# Benchmarks du rapport RAPPORT_UNET3D.md (section 6.1), GPU libre (sinon les temps sont faussés) :
# les modèles de run_experiments.sh + méthode classique : jeux hybride, simulé, film réel, paires.
# Lancer depuis la racine du dépôt : bash deep_learning_segmentation/experiences/run_benchmarks.sh
set -u
export PYTHONIOENCODING=utf-8
DL=deep_learning_segmentation
RUNS=""
for r in centernet wmse bce_dice focal_tversky centernet_localmad centernet_orient centernet_z2 centernet_z2_base8 centernet_simtrain; do
  [ -f $DL/runs/$r/model_best.pt ] && RUNS="$RUNS $r"
done
echo "=== benchmark :$RUNS"
python -u $DL/benchmark.py --runs $RUNS --datasets hybrid sim --pairs_dir $DL/data/hybrid/pairs \
  --out $DL/runs/benchmark > $DL/runs/benchmark.log 2>&1
echo "=== benchmark fini ($?)"

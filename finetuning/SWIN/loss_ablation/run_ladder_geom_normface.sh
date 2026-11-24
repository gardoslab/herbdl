#!/bin/bash -l
# Ladder geometry for NormFace: one eval-only pass per rung over that rung's
# validation set (saves pooled features + predictions), then the ladder
# diagnostics (margin, angle to the species' weight, Fisher) for all losses with
# features. DBM / KappaFace / ArcFace features from 2026-10-30 are reused.
set -u
for rung in n15k n32k n64k; do
    echo "__LADDERGEOM__ ===== START $rung cosine_m0 $(date '+%F %T')"
    if ! HERBDL_NO_RELOCATE=1 NPROC_PER_NODE=1 \
         CONFIG_FILE="configs_advanced/ladder_eval/${rung}_cosine_m0.yml" bash train_advanced.sh; then
        echo "__LADDERGEOM__ FAILED $rung cosine_m0"
        exit 1
    fi
    echo "__LADDERGEOM__ ===== DONE $rung cosine_m0 $(date '+%F %T')"
done
module load miniconda
module load academic-ml/fall-2026
conda activate fall-2026-pyt
python loss_ablation/ladder_diagnostics.py --out ladder_diagnostics_with_normface.json

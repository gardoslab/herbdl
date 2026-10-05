#!/bin/bash -l
# Phase 3 geometry: (1) re-run the full-validation-set eval of each model, saving pooled
# features (configs_advanced/phase3_geom/*.yml), skipping any model whose features and
# predictions already exist; (2) compute geometry metrics for all of them (CPU/GPU).

GEOM=/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/output/SWIN/PHASE3_GEOM
status=0
for name in dbm kappaface arcface dbm_seed43 kappaface_seed43 arcface_seed43 \
            subcenter_arcface arcface_tara arcface_ecc hace x2softmax; do
    if [ -f "$GEOM/$name/features.npy" ] && [ -f "$GEOM/$name/predictions.npz" ]; then
        echo "__PHASE3G__ ===== SKIP $name (features already saved)"
        continue
    fi
    echo "__PHASE3G__ ===== START $name $(date '+%F %T')"
    if HERBDL_NO_RELOCATE=1 CONFIG_FILE=configs_advanced/phase3_geom/$name.yml bash train_advanced.sh; then
        echo "__PHASE3G__ ===== OK $name $(date '+%F %T')"
    else
        echo "__PHASE3G__ ===== FAILED $name"
        status=1
    fi
done

module load miniconda
module load academic-ml/fall-2026
conda activate fall-2026-pyt
python loss_ablation/phase3_geometry.py || status=1
exit $status

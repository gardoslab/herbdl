#!/bin/bash -l
# Supplement geometry for the ConvNeXt / DINOv2 models.
# Usage (from finetuning/SWIN): bash loss_ablation/run_backbone_geom.sh <name> [<name> ...]
#   e.g. convnext_dbm dinov2_arcface ... (configs: configs_advanced/backbones/geom/<name>_{val,train}.yml)
# For each model: an eval-only pass on validation and on training images that saves pooled features
# and predictions (skipped if the feature file already exists), then backbone_geometry.py on all of them.
# No `set -e`/`set -u`: miniconda's activation scripts return non-zero / use unset variables in batch shells.
FAILED=0
for name in "$@"; do
    for split in val train; do
        root=PHASE4_GEOM; [ "$split" = train ] && root=PHASE4_GEOM_TRAIN
        if [ -f "/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/output/SWIN/$root/$name/features.npy" ]; then
            echo "__BACKGEOM__ SKIP $name $split (features exist)"
            continue
        fi
        echo "__BACKGEOM__ ===== START $name $split $(date '+%F %T')"
        if HERBDL_NO_RELOCATE=1 NPROC_PER_NODE=1 \
           CONFIG_FILE="configs_advanced/backbones/geom/${name}_${split}.yml" bash train_advanced.sh; then
            echo "__BACKGEOM__ ===== DONE $name $split $(date '+%F %T')"
        else
            echo "__BACKGEOM__ FAILED $name $split"
            FAILED=1
        fi
    done
done
bash loss_ablation/run.sh backbone_geometry.py --models "$@"
exit $FAILED

#!/bin/bash -l
# ConvNeXt / DINOv2 on a ladder rung: feature passes (val, train) then ladder_geometry_full.py.
# Usage (from finetuning/SWIN): bash loss_ablation/run_backbone_ladder_geom.sh <convnext|dinov2> <n32k|n64k> <loss> [<loss> ...]
BB=$1; RUNG=$2; shift 2
OUT=/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/output/SWIN
FAILED=0; L=()
for loss in "$@"; do
    name=${BB}_${RUNG}_$loss; L+=(${BB}_$loss)
    for split in val train; do
        root=LADDER_GEOM; [ "$split" = train ] && root=LADDER_GEOM_TRAIN
        if [ -f "$OUT/$root/$name/features.npy" ]; then echo "__LADDERGEOM__ SKIP $name $split"; continue; fi
        echo "__LADDERGEOM__ ===== START $name $split $(date '+%F %T')"
        HERBDL_NO_RELOCATE=1 NPROC_PER_NODE=1 CONFIG_FILE=configs_advanced/backbones/ladder/geom/${name}_${split}.yml \
            bash train_advanced.sh || { echo "__LADDERGEOM__ FAILED $name $split"; FAILED=1; }
    done
done
bash loss_ablation/run.sh ladder_geometry_full.py --rungs "$RUNG" --losses "${L[@]}"
exit $FAILED

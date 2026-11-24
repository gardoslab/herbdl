#!/bin/bash -l
# Ladder geometry passes + analysis.
# Usage (from finetuning/SWIN): bash loss_ablation/run_ladder_geom.sh <split: val|train> <rung> <loss> [<loss> ...]
#   val:   configs_advanced/ladder_eval/<rung>_<loss>.yml        -> LADDER_GEOM/<rung>_<loss>/features.npy
#   train: configs_advanced/ladder_geom_train/<rung>_<loss>.yml  -> LADDER_GEOM_TRAIN/<rung>_<loss>/features.npy
# Passes whose feature file exists are skipped; then ladder_geometry_full.py on both splits of these cells.
# No `set -e`/`set -u`: miniconda's activation scripts misbehave under them in batch shells.
SPLIT=$1; RUNG=$2; shift 2
OUT=/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/output/SWIN
FAILED=0
for loss in "$@"; do
    if [ "$SPLIT" = train ]; then cfg=ladder_geom_train/${RUNG}_${loss}.yml; f=$OUT/LADDER_GEOM_TRAIN/${RUNG}_${loss}/features.npy
    else cfg=ladder_eval/${RUNG}_${loss}.yml; f=$OUT/LADDER_GEOM/${RUNG}_${loss}/features.npy; fi
    if [ -f "$f" ]; then echo "__LADDERGEOM__ SKIP ${RUNG}_${loss} $SPLIT (features exist)"; continue; fi
    echo "__LADDERGEOM__ ===== START ${RUNG}_${loss} $SPLIT $(date '+%F %T')"
    if HERBDL_NO_RELOCATE=1 NPROC_PER_NODE=1 CONFIG_FILE=configs_advanced/$cfg bash train_advanced.sh; then
        echo "__LADDERGEOM__ ===== DONE ${RUNG}_${loss} $SPLIT $(date '+%F %T')"
    else
        echo "__LADDERGEOM__ FAILED ${RUNG}_${loss} $SPLIT"; FAILED=1
    fi
done
bash loss_ablation/run.sh ladder_geometry_full.py --rungs "$RUNG" --losses "$@"
exit $FAILED

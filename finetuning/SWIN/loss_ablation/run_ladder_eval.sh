#!/bin/bash -l
# Re-score finished ladder models on their rung's full validation set and save per-image
# predictions (configs_advanced/ladder_eval/*.yml). One model at a time on 1 GPU; a failure
# in one does not stop the others. Usage: bash loss_ablation/run_ladder_eval.sh [loss ...]
LOSSES=${*:-dbm}
status=0
for loss in $LOSSES; do
  for rung in n15k n32k n64k; do
    name=${rung}_${loss}
    echo "__LADDER_EVAL__ ===== START $name $(date '+%F %T')"
    if HERBDL_NO_RELOCATE=1 CONFIG_FILE=configs_advanced/ladder_eval/$name.yml bash train_advanced.sh; then
      echo "__LADDER_EVAL__ ===== OK $name $(date '+%F %T')"
    else
      echo "__LADDER_EVAL__ ===== FAILED $name"; status=1
    fi
  done
done
exit $status

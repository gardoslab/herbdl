#!/bin/bash -l
# Part 4 step 1: (a) CPU/GPU smoke test of the backbone adapters, then (b) re-score llindon's
# finished plain-CE checkpoints on the full val_2022 set with per-image predictions
# (configs_advanced/backbone_eval/*.yml, made by generate.py). One model at a time on 1 GPU;
# a failure in one does not stop the others.

module load miniconda
module load academic-ml/fall-2026
conda activate fall-2026-pyt

EVAL_ROOT=/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/output/SWIN/PHASE4_EVAL
status=0
for name in convnext_arcface convnext_dbm convnext_ce_la dinov2_arcface dinov2_dbm dinov2_ce_la; do
    echo "__PART4EVAL__ ===== START $name $(date '+%F %T')"
    if HERBDL_NO_RELOCATE=1 CONFIG_FILE=configs_advanced/backbones/eval/$name.yml bash train_advanced.sh; then
        echo "__PART4EVAL__ ===== OK $name $(date '+%F %T')"
    else
        echo "__PART4EVAL__ ===== FAILED $name"; status=1
    fi
done
for name in convnext_arcface convnext_dbm convnext_ce_la dinov2_arcface dinov2_dbm dinov2_ce_la; do
    python3 - "$EVAL_ROOT/$name" "$name" <<'PY'
import json, os, sys
d, name = sys.argv[1], sys.argv[2]
r = {}
for f in ("eval_results.json", "all_results.json"):
    p = os.path.join(d, f)
    if os.path.isfile(p):
        r = json.load(open(p)); break
print(f"__PART4EVAL_SUMMARY__ {name}: eval_accuracy={r.get('eval_accuracy')} eval_f1={r.get('eval_f1')} "
      f"predictions_saved={os.path.isfile(os.path.join(d, 'predictions.npz'))}")
PY
done
exit $status

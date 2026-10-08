#!/bin/bash -l
# Phase 3: re-evaluate every finished 384px loss model on the full validation set and
# save each image's prediction (configs_advanced/phase3_eval/*.yml, made by generate.py).
# One model at a time on 1 GPU; a failure in one model does not stop the others.
# Most important models first, so a timeout still leaves the key comparisons.

EVAL_ROOT=/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/output/SWIN/PHASE3_EVAL
status=0
for name in kappaface dbm arcface subcenter_arcface arcface_tara arcface_ecc hace x2softmax; do
    echo "__PHASE3__ ===== START $name $(date '+%F %T')"
    if HERBDL_NO_RELOCATE=1 CONFIG_FILE=configs_advanced/phase3_eval/$name.yml bash train_advanced.sh; then
        echo "__PHASE3__ ===== OK $name $(date '+%F %T')"
    else
        echo "__PHASE3__ ===== FAILED $name"
        status=1
    fi
done

# One line per model: recomputed accuracy/macro-F1 (must match the training run's own
# final numbers) and whether the predictions file exists.
for name in kappaface dbm arcface subcenter_arcface arcface_tara arcface_ecc hace x2softmax; do
    python3 - "$EVAL_ROOT/$name" "$name" <<'PY'
import json, os, sys
d, name = sys.argv[1], sys.argv[2]
r = {}
for f in ("eval_results.json", "all_results.json"):
    p = os.path.join(d, f)
    if os.path.isfile(p):
        r = json.load(open(p)); break
print(f"__PHASE3_SUMMARY__ {name}: eval_accuracy={r.get('eval_accuracy')} eval_f1={r.get('eval_f1')} "
      f"predictions_saved={os.path.isfile(os.path.join(d, 'predictions.npz'))}")
PY
done
exit $status

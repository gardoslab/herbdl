# AGENTS.md

Guidance for coding agents (Codex, Claude Code, others) working in this repository. The detailed
architecture guide is `CLAUDE.md`; the repository map is `README.md`.

## Project overview

Experiments in unsupervised and supervised learning on herbarium specimens: plant species classification with
SWIN, ConvNeXt, DINOv2, CoAtNet, EfficientNet, CLIP and BioCLIP, on Boston University's Shared Computing Cluster
(SCC). Key paths: base `/projectnb/herbdl/`, images `/projectnb/herbdl/data/kaggle-herbaria/`, checkpoints
`/projectnb/herbdl/workspaces/<username>/herbdl/finetuning/output/`.

Training goes through one script, `finetuning/SWIN/SWIN_finetuning_advanced.py`, launched with
`finetuning/SWIN/train_advanced.sh` and a YAML config (see `CLAUDE.md` for the command and environment).

## Lessons learned (loss ablations, scaling ladder, backbone repeat)

Hard-won; read before launching or analysing anything. Results and numbers live in the project's
"Loss Function Ablations" document (kept outside this repo).

### Evaluation: count the images, then trust the number
- **Always check how many validation images were scored.** `val_2022` has 167,955. Two silent bugs changed results:
  2-GPU runs were scored on rank 0's half only (fixed: predictions are gathered across ranks; macro-F1 was ~1 point
  low), and a collaborator's configs capped evaluation at `data.max_eval_samples: 50000` (accuracy barely moves but
  macro-F1 is 1.8-2.3 points lower on a subset, because species with no validation image but some predictions score F1 = 0).
  Look for `Num examples = 167955` in the log.
- An eval-only re-score (`evaluation.init_weights` + `evaluation.predictions_path`, `do_train: false`) must reproduce the
  training run's own final numbers; use it to recover a score when a job died right after saving its model, and to save
  per-image predictions for later analysis.
- Macro-F1 = mean per-class F1 over classes in the labels or the predictions. Compare models only on the same images.
  "Points" means percentage points (0.8161 -> 0.8125 is 0.36 points).
- Seed noise is about 0.1 points (two seeds of three losses only). Bootstrap intervals over validation images measure
  evaluation noise, not training noise.
- **Same-specimen copies:** about 1.3-1.7% of `val_2022` images have a copy in `train_2022` (cosine of a generic SWIN
  feature >= 0.93 flags 1.68%; by eye ~1.25%, because the 0.93-0.95 band is mostly look-alike scans). Models get ~98% of
  them right; removing them lowers scores by a few tenths and changes no ranking. The ladder's validation set is easier
  (Herbarium 2021 images) and not comparable with the ablation.
- **Geometry metrics:** the separation gap subtracts an image-to-centroid cosine from a centroid-to-centroid cosine, so it
  depends on how many images each centroid has. Compare by number of training images per species and with an equal number
  of validation images per species (`loss_ablation/geometry_fixed_n.py`). The overall average is dominated by common species
  (53% of species have 51-73 training images).

### Training and configs
- The native cross-entropy path applies label smoothing 0.1 by default (the collator default) and ignores logit adjustment;
  the loss registry heads bypass it. Run plain cross-entropy through the registry `softmax` head. Logit adjustment works on
  that head only.
- The loss ablation uses **only a random crop and flip**. Heavy augmentation (RandAugment, mixup/CutMix, label smoothing)
  exists in many older configs; applied from the first step it hurt badly (0.62 vs 0.78); ramp it in
  (`augmentation.warmup_epochs`) or start from a trained checkpoint.
- bf16: margin heads must take `acos` in fp32 (a bf16 clamp epsilon is a no-op, so `acos(>1)` gives NaN). Grep logs for
  non-finite warnings before relaunching.
- Keep the **effective batch** (per-device x accumulation x GPUs) when changing GPU count. Use
  `resume_from_checkpoint: newest`; never leave `overwrite_output_dir` on. KappaFace and ECC are 1-GPU only.
- Check a new config against the script's required keys before launching (e.g. `custom.run_notes`); a missing key kills
  the job in seconds. Compare losses within a backbone only; each backbone keeps its own tuned recipe.

### Cluster and job operations
- SGE stops jobs at 12 hours ("job died on the exec node"). Relaunch from the newest checkpoint. Elapsed time under 12 hours
  with exit status 137 means a node loss or kill: relaunch the same way. A job ending right at the limit can save the model
  but not the final summary: re-score it.
- After every launch run `orx supervise <run> --restart` and confirm the run leaves "starting" (the launcher prints "Could
  not spawn `orx supervise`" and otherwise no status updates arrive). Do not trust the run list alone: stale "running"
  records hid dead jobs for two days. Check `qstat` and checkpoint modification times. Logs of running jobs lag; read the
  job's log file on the cluster.
- The first evaluation of a run is slow (cold file reads, up to an hour), later ones take minutes. One-off NCCL watchdog
  timeouts and Weights & Biases 404s on resume happen; relaunching from the checkpoint fixes them.
- Keep at most 4 jobs queued or running unless told otherwise.

### Git
- Each experiment is an `orx/...` branch off the previous one; all are merged into `loss-ablations`. Shared single-file
  configs (`repgeom/configs/repgeom.yml`, the SWIN-L hybrid yml) were overwritten by each experiment: keep one file per
  variant instead.

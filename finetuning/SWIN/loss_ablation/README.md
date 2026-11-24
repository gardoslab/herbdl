# Loss-function ablations (SWIN-Base, 384px)

Code behind the paper *Smaller Margins, Looser Clusters* (draft in `gardoslab/project-artifacts`,
`herbdl/loss-ablations/`). Eight losses are compared on Herbarium 2022 / NAFlora-1M (15,501
species, the "base data"), the margin is varied from 0° to 17°, and the strongest losses are
followed up a species-count ladder (15.5k / 32k / 64k species from the merged FGVC8 + FGVC9 data).
All training and eval-only scoring goes through `finetuning/SWIN/SWIN_finetuning_advanced.py` +
`train_advanced.sh` with a YAML config; analyses are standalone scripts run with `run.sh`.

```bash
cd finetuning/SWIN
# train (2 GPUs) or score a finished model (1 GPU) from a config
HERBDL_NO_RELOCATE=1 NPROC_PER_NODE=2 CONFIG_FILE=configs_advanced/swin_base_384_dbm_pilot.yml bash train_advanced.sh
HERBDL_NO_RELOCATE=1 CONFIG_FILE=configs_advanced/phase3_geom/dbm.yml bash train_advanced.sh
# run an analysis
bash loss_ablation/run.sh paper_measures.py
```

KappaFace and ECC train on 1 GPU (their per-species buffers are not synced across GPUs). Long
runs resume from the newest checkpoint across 12–24 h scheduler slots.

## Model and loss code

| File | What it is |
|---|---|
| `losses.py` | Classifier heads: softmax, ArcFace / sub-center ArcFace (`m=0` is NormFace), KappaFace, X2-Softmax, DBM, sibling margin |
| `sibling_margin.py` | The sibling-margin rule (pairwise, taxonomy-aware, rarity-scaled) in NumPy; `SiblingMarginHead` calls it. Test: `sibling_margin_test.py` |
| `hierarchy.py` | HACE head and `TaxonomyTree` |
| `aux.py`, `bioclip_teacher.py` | Auxiliary losses on top of a head: ECC, TARA (TARA needs the BioCLIP teacher) |
| `model.py` | `SwinWithLoss`: backbone + any head / hierarchy / aux combination |
| `backbones.py` | Pooled-feature adapters for ConvNeXt, DINOv2, EfficientNet, timm models (cross-backbone replication) |
| `checkpoint_io.py`, `grad_modulation.py` | Geometry export for `repgeom`; FlexiGrad primitives (not wired into training) |
| `build_ladder.py` | Builds the nested 15.5k / 32k / 64k species sets from the merged 2021 + 2022 manifests |

## Configs (`finetuning/SWIN/configs_advanced/`)

| Paper element | Configs |
|---|---|
| Main table (base data) | `swin_base_384_{dbm,kappaface,subcenter_arcface,arcface_tara,arcface_ecc,hace,x2softmax}_pilot.yml`, `swin_base_384_arcface_ablation.yml`, `swin_base_384_ce.yml`, `swin_base_384_ce_la.yml`; second seed: `*_seed43.yml` |
| Margin controls | `swin_base_384_cosine_m0.yml` (NormFace), `swin_base_384_arcface_m01.yml` |
| Logit-scale control | `swin_base_384_cosine_m0_s15.yml`, `swin_base_384_arcface_s15.yml` |
| Scaling ladder | `swin_base_384_ladder_{n15k,n32k,n64k}_{dbm,kappaface,arcface,cosine_m0}.yml` (fixed 131,250 steps) |
| Epoch-matched 32k check | `swin_base_384_ladder_n32k_{cosine_m0,arcface}_e50.yml`; scored by `ladder_eval/n32k_{cosine_m0,arcface}_e50.yml` (`run_e50_score.sh`) |
| Cross-backbone replication | `backbones/` (train, incl. `{convnext,dinov2}_cosine_m0.yml`), `backbones/eval/`, `backbone_eval/` (score) |
| Cross-backbone geometry | `backbones/geom/` (base data, val and train features); `backbones/ladder/` (ConvNeXt / DINOv2 on the 32k and 64k rungs) and `backbones/ladder/geom/` |
| Ladder geometry on training images | `ladder_geom_train/`; plain-CE ladder scoring `ladder_eval/{n15k,n32k,n64k}_ce.yml` |
| Sibling margin (32k rung, not in the paper) | `swin_base_384_ladder_n32k_sibling_margin.yml` (train), `ladder_eval/n32k_sibling.yml` (score) |
| Scoring with saved predictions | `phase3_eval/` (base data), `ladder_eval/` (ladder) |
| Scoring with saved features | `phase3_geom/` (validation images), `phase3_geom_train/` (training images) |

## Analyses (all read saved predictions / features; outputs under `output/SWIN/MARGIN_DIAG` or `PHASE3_GEOM/analysis`)

| Paper element | Script |
|---|---|
| Main-table intervals, per-band F1 (base) | `phase3_per_species.py`, `main_table_ci.py` |
| Per-band comparison with NormFace (base and ladder) | `cosine_band_compare.py` |
| Margins each loss trains with; scale-normalised margins, separation gap, alignment | `scale_free_margins.py` |
| Compactness (angle) and Fisher separability, validation images | `paper_separability.py` (needs `boundary_margins.py`), `paper_measures.py` |
| Same, on training images | `train_measures.py` |
| Testing the mechanism (train vs val by band, far misses, sister-species separability) | `cone_test.py` |
| Logit-scale control | `scale15_measures.py` |
| Head swap | `head_swap.py` |
| Ladder: lead decomposition by band; KappaFace / DBM trained margins | `band_leads_by_rung.py`, `ladder_margins.py` (band helpers in `ladder_routing.py`) |
| Ladder data checks: long tail, base vs ladder 15.5k | `ladder_distribution.py`, `ladder_vs_phase2.py` |
| Same-specimen copies between training and validation | `near_duplicates.py`, `near_duplicates_precise.py`, `part4_leakage.py` |
| Bias-corrected and size-matched Fisher; compactness = within-species spread x alignment | `geometry_robustness.py`, `geometry_robustness_part2.py` |
| Geometry of the ConvNeXt / DINOv2 models (base data) | `backbone_geometry.py` (`run_backbone_geom.sh`: feature passes, then the analysis) |
| Ladder geometry, validation and training images, every loss and backbone | `ladder_diagnostics.py`, `ladder_geometry_full.py` (`run_ladder_geom.sh`, `run_ladder_geom_normface.sh`, `run_backbone_ladder_geom.sh`, `run_convnext_ladder_geom.sh`) |
| Supplement geometry table | `supplement_table.py` |
| Who pays for a margin (common-species cost by margin size) | `who_pays.py` |
| Feature geometry helpers | `phase3_geometry.py`, `repgeom_geometry.py` |

### Sibling margin (follow-up, not in the paper)

A genus-aware pairwise margin, tested on the 32k rung; it did not beat DBM (EXPERIMENT_LOG, 2026-11-23).

| Question | Script |
|---|---|
| How much can a family / genus oracle recover; error taxonomy | `sibling_ceiling.py` |
| Siblings vs the same number of nearest class templates | `sibling_matched_control.py` |
| The margin on the real training sets | `sibling_margin_distribution.py` |
| Results at 32k: per-band F1 vs NormFace and DBM, error relations, realised gaps | `sibling_results.py` (test: `sibling_results_test.py`) |

## Trainer options added for this work (`SWIN_finetuning_advanced.py`)

- `hace:` and `loss_registry:` config sections (head, head_kwargs, aux, aux_log_steps)
- `training.resume_from_checkpoint: newest` and `training.max_steps` (fixed step budget for the ladder)
- `evaluation.init_weights` / `predictions_path` / `features_path` (eval-only scoring)
- `loss_registry.margin_warmup_epochs` (registry ArcFace) and `arcface.margin_warmup_epochs` (native ArcFace)
- Logit adjustment (`long_tail.logit_adjustment`) on the registry softmax head (training only)
- Multi-GPU evaluation gathers predictions from all ranks
- `loss_registry.head: sibling` (needs `genus` and `family` columns in the dataset)
- ArcFace heads lower-bound `1 - cos^2` at 1e-7 before the square root, so the gradient stays finite as
  `cos -> 1` (without it the s=15 ArcFace run went NaN in epoch 22)

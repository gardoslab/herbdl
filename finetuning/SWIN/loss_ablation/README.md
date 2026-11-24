# Loss-function ablations (SWIN-Base, 384px)

Code behind the paper *Smaller Margins, Looser Clusters* (draft in `gardoslab/project-artifacts`,
`herbdl/loss-ablations/`). Eight losses are compared on Herbarium 2022 / NAFlora-1M (15,501
species, the "base data"), the margin is varied from 0° to 17°, and the strongest losses are
followed up a species-count ladder (15.5k / 32k / 64k species from the merged FGVC8 + FGVC9 data).
All training and eval-only scoring goes through `finetuning/SWIN/SWIN_finetuning_advanced.py` +
`train_advanced.sh` with a YAML config; analyses are standalone scripts in `analysis/`, run with `run.sh`.

```bash
cd finetuning/SWIN
# train (2 GPUs) or score a finished model (1 GPU) from a config
HERBDL_NO_RELOCATE=1 NPROC_PER_NODE=2 CONFIG_FILE=configs_advanced/swin_base_384_dbm_pilot.yml bash train_advanced.sh
HERBDL_NO_RELOCATE=1 CONFIG_FILE=configs_advanced/<eval-only copy>.yml bash train_advanced.sh   # see configs_advanced/README.md
# run an analysis
bash loss_ablation/run.sh paper_measures.py
```

KappaFace and ECC train on 1 GPU (their per-species buffers are not synced across GPUs). Long
runs resume from the newest checkpoint across 12–24 h scheduler slots.

## Model and loss code

| File | What it is |
|---|---|
| `losses.py` | Classifier heads: softmax, ArcFace / sub-center ArcFace (`m=0` is NormFace), KappaFace, X2-Softmax, DBM, sibling margin |
| `sibling_margin.py` | The sibling-margin rule (pairwise, taxonomy-aware, rarity-scaled) that `SiblingMarginHead` calls |
| `hierarchy.py` | HACE head and `TaxonomyTree` |
| `aux.py` | Auxiliary losses on top of a head: ECC, TARA. TARA needs a frozen BioCLIP teacher, which is not in this branch |
| `model.py` | `SwinWithLoss`: backbone + any head / hierarchy / aux combination |
| `backbones.py` | Pooled-feature adapters for ConvNeXt, DINOv2, EfficientNet, timm models (cross-backbone replication) |
| `checkpoint_io.py` | Geometry export read by `repgeom/` |
| `build_ladder.py` | Builds the nested 15.5k / 32k / 64k species sets from the merged 2021 + 2022 manifests |

## Configs

One training config per model, dataset and loss in `finetuning/SWIN/configs_advanced/`; its README
(section "Loss-ablation configs") lists them and explains how to change the seed, margin, scale or
training length, and how to write an eval-only scoring config.

## Analyses (`analysis/`)

Each script reads saved predictions or pooled features written by an eval-only scoring pass (see
`configs_advanced/README.md`) and writes JSON under `output/SWIN/MARGIN_DIAG`. Only the final version of each
analysis is kept; earlier versions are in the experiment branches.

| Paper element | Script |
|---|---|
| Main-table intervals, per-band F1 (base) | `phase3_per_species.py`, `main_table_ci.py` |
| Per-band comparison with NormFace (base and ladder) | `cosine_band_compare.py` |
| Margins each loss trains with (scale-normalised) | `scale_free_margins.py` |
| Compactness angle and bias-corrected Fisher: SWIN losses, scale control, cone pairs, ladder | `geometry_robustness.py`, `geometry_robustness_part2.py` |
| Same, ConvNeXt / DINOv2 on the base data | `backbone_geometry.py` |
| Same, every loss and backbone on the ladder, validation and training images | `ladder_geometry_full.py` |
| Supplement geometry table | `supplement_table.py` |
| Mechanism test (train vs val by band, far misses, sister-species separability) | `cone_test.py` |
| Head swap | `head_swap.py` |
| Same-specimen copies between training and validation; scores without them | `part4_leakage.py` |
| Shared helpers imported by the scripts above | `paper_measures.py`, `paper_separability.py`, `phase3_geometry.py`, `repgeom_geometry.py`, `boundary_margins.py`, `ladder_diagnostics.py` |

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

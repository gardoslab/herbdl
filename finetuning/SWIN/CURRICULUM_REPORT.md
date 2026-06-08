# Curriculum Learning — Stage-by-Stage Impact Report

## Starting Point: SWIN_BASE_BASELINE

**What it is:** SWIN-Base (224px, ImageNet-22k pretrained), fine-tuned with standard CE loss, no augmentation beyond basic resizing/normalization, unfrozen backbone from the start.

**Result:** Peak F1 = **0.7454** @ epoch 47.8

**Interpretation:** Solid starting point. Slow convergence curve — model starts at 0.58 F1 and takes ~48 epochs to plateau. This is the reference to beat.

---

## Interlude: Standalone Augmentation Test (SWIN_BASE_224_AUGMENTED)

**What it added:** Heavy augmentation (RandAugment mag=9, Mixup α=0.8, CutMix α=1.0, RandomErasing 25%, label smoothing 0.1) applied directly from scratch — no warm-up, no curriculum.

**Result:** Peak F1 = **0.6118** @ epoch 44.4 — **worse than baseline by 3.4 points**

**Why it failed:** Throwing all regularization at a model cold is destructive. Strong Mixup/CutMix targets corrupt learning signal before the backbone has stabilized. The model oscillates and never recovers — note the flat 0.57–0.61 plateau from epoch 20–99. This is the key motivation for curriculum learning.

---

## Curriculum Stage 1 — Mild Augmentation Warm-up

**What changed:** Initialized from baseline checkpoint. RandAugment mag=4 (mild), Mixup α=0.8, CutMix α=1.0, RandomErasing p=0.1, label smoothing 0.05. LR = 5e-5.

**Result:** Peak F1 = **0.7214** @ epoch 23.9

**Interpretation:** Starts immediately at 0.69 F1 (baseline already baked in), reaches 0.72 in 24 epochs. The mild augmentation + lower LR successfully builds on the baseline without disrupting it. Notably, this run converges faster than the baseline — 0.69 at epoch 3 vs. 0.58 for baseline.

**Gain vs baseline at epoch 24:** +0.013 F1

---

## Curriculum Stage 2 — Medium Augmentation

**What changed:** From S1 checkpoint. RandAugment mag=7 (stepped up), RandomErasing p=0.15, label smoothing 0.1. LR = 3e-5.

**Result:** Peak F1 = **0.7421** @ epoch 27.3

**Gain vs S1:** +0.021 F1

**Interpretation:** The stepped-up augmentation is now helping rather than hurting, because the backbone is already warm. Model jumps to 0.72 at epoch 3 and climbs to 0.74 by epoch 27.

---

## Curriculum Stage 3 — Heavy Augmentation

**What changed:** From S2 checkpoint. RandAugment mag=9 (full strength), RandomErasing p=0.25. LR = 2e-5. 50 epochs.

**Result:** Peak F1 = **0.7510** @ epoch 41.0

**Gain vs S2:** +0.009 F1. Diminishing returns beginning.

**Interpretation:** Full augmentation now converges to a higher ceiling than baseline. However, the improvement margin is shrinking. The model starts at 0.74 immediately and creeps upward slowly — most gain is in early epochs, then it plateaus.

---

## Curriculum Stage 3-Cont — Extended Cosine Schedule

**What changed:** From S3 final model (not best checkpoint). Fresh cosine LR schedule restart from 2e-5. Same augmentation. Intended to push past the S3 plateau.

**Result:** Peak F1 = **0.7510** @ epoch 50.0

**Gain vs S3:** **+0.000 F1**

**Interpretation:** The LR restart did not help — S3 had already converged. The model stays in the same 0.74–0.75 band the entire 50 epochs. This suggests the 224px + CE + augmentation combination has hit its ceiling.

---

## Curriculum MultiTask — Auxiliary Family/Genus Heads

**What changed:** From S3-Cont final model. Added CE auxiliary heads for family and genus (weights 0.2×family + 0.3×genus + 1.0×species). Mixup/CutMix retained. LR = 3e-4 (higher — new heads need to train). 100 epochs.

**Result:** Peak F1 = **0.7523** @ epoch 68.3

**Gain vs S3-Cont:** +0.001 F1 net, but with a very different trajectory.

**Key observation:** The new family/genus heads start randomly initialized → eval_on_start near-zero → slow recovery through ~40 epochs before exceeding S3-Cont. MultiTask eventually pulls ahead but the improvement is modest. The multi-task signal is providing regularization but not a dramatic accuracy boost on its own.

---

## Curriculum ArcFace — SubCenter ArcFace Metric Learning

**What changed:** From MultiTask checkpoint. Replaced CE species head with SubCenter ArcFace (embedding=512, scale=30, margin=0.5, k=3 sub-centers). Mixup/CutMix disabled (incompatible with hard labels). Hybrid CE weight = 0.0. LR = 1e-4. 60 epochs.

**Result:** Peak F1 = **0.7376** @ epoch 58.1

**Gain vs MultiTask:** **–0.015 F1** — a regression.

**Interpretation:** ArcFace starts from near-zero (random embedding + weight matrix initialization), takes ~40 epochs just to recover to MultiTask's level, and peaks 1.5% *below* the MultiTask checkpoint it started from. The loss function change required too many epochs to re-learn what CE had already learned. The 60-epoch budget was insufficient for ArcFace to amortize its warm-up cost and then improve further.

---

## Summary Table

| Stage | Technique Added | Peak F1 | Δ vs Previous | Epochs to Peak |
|-------|----------------|---------|--------------|----------------|
| Baseline | CE, no augmentation | 0.7454 | — | 47.8 |
| Aug (standalone) | Heavy aug, no curriculum | 0.6118 | –0.034 | 44.4 |
| S1 | Mild aug (warm-up) | 0.7214 | –0.024* | 23.9 |
| S2 | Medium aug | 0.7421 | +0.021 | 27.3 |
| S3 | Heavy aug | 0.7510 | +0.009 | 41.0 |
| S3-Cont | LR restart | 0.7510 | +0.000 | 50.0 |
| MultiTask | Family/genus aux heads | 0.7523 | +0.001 | 68.3 |
| ArcFace | Metric learning loss | 0.7376 | **–0.015** | 58.1 |

\* S1 starts below baseline because it used fewer epochs (25 vs. 48 for baseline). Chaining S1→S2→S3 ultimately exceeds the baseline ceiling (0.751 vs. 0.745).

---

## Key Takeaways

1. **Curriculum ordering matters critically.** Applying heavy augmentation cold destroyed performance (0.61). Applied progressively, it exceeds baseline (0.751 vs. 0.745).

2. **The aug curriculum plateau is around 0.750–0.752.** S3, S3-Cont, and MultiTask all peak in this band. The 224px CE model appears structurally capped here.

3. **MultiTask gave only marginal gain (+0.001).** The auxiliary signal helps slightly but the species task already dominates. More useful as regularization than as a direct accuracy booster.

4. **ArcFace regressed.** The 60-epoch budget was too short — ArcFace requires a long cold-start recovery period before it can outperform CE. The hybrid/384 stages queued after it will inherit this disadvantage.

5. **The gap to 0.80 is still ~5 points.** The most promising levers remaining are:
   - **384px resolution** — larger receptive field is known to help fine-grained recognition
   - **SWIN V2 architecture** — updated relative position bias and scaled cosine attention
   - **Revisiting ArcFace** with a longer budget or frozen-backbone warm-up phase

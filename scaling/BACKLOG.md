# Label-cardinality scaling — backlog

Deferred work for this line, in rough priority order. Items move out of here
into an experiment node when they are picked up.

## 1. Architecture ablation (explicitly deferred by the user, 2026-09-15)

Does the cardinality curve have the same shape across architecture families, or
do some degrade more gracefully as the label space grows?

Arms, all as frozen feature extractors so the existing sweep harness applies
unchanged — only `features.backbone` in `configs/scaling.yml` moves:

| Family | Candidate checkpoint | Note |
|---|---|---|
| Transformer (hierarchical) | `microsoft/swin-base-patch4-window7-224-in22k` | the current baseline |
| Transformer (plain) | `google/vit-base-patch16-224-in21k` | isolates hierarchy vs plain attention |
| CNN | `facebook/convnext-base-224-22k` | modern CNN at matched scale/pretraining |
| Hybrid | `MambaVision-B` / MaxViT | conv + attention |
| SSM / Mamba-style | `VMamba-B` | no published fine-grained result above N=10^4 exists |
| Self-supervised ViT | `facebook/dinov2-base` | no label supervision at all upstream |

Controls that matter and are easy to get wrong:
- **Match pretraining data.** IN-22k vs IN-1k vs LVD-142M is a bigger lever than
  architecture; comparing a 22k-pretrained Swin against a 1k-pretrained VMamba
  measures the pretraining corpus, not the family.
- **Match feature dimensionality**, or at least report it — a 1024-d probe and a
  768-d probe are not the same classifier.
- The claim to test is about the *slope* of the curve, not the intercept. Two
  backbones can differ by 10 points at N=50 and still degrade identically.

Prior work says to expect a null: architecture family adds no incremental
explanatory power once baseline accuracy is controlled for
(arXiv 2609.06051, and Abnar et al. arXiv 2110.02095 for the scaling analogue).
A clean null at N=15k would itself be worth reporting.

## 2. Fixed *total* image budget (the other half of the control)

The current sweep holds images-per-class fixed, so total data grows with N. The
complementary regime holds the total image count fixed and lets images-per-class
fall as 1/N. That answers the practical question — *given a fixed annotation
budget, how finely should the label space be sliced?* — which is the
"more classes or more examples per class?" ablation from
arXiv 1608.08614, never run at 15k-class fine-grained scale.

Cheap: the feature cache already covers it, only the sampler changes.

## 3. Natural long-tailed regime

The balanced regime deliberately removes imbalance so cardinality can be
isolated. The realistic setting keeps the empirical Herbarium distribution, at
which point the frequency-based head/tail split in `metrics.shot_bins`
(few <20 / medium 20–100 / many >100) becomes meaningful and can be compared
against the separability-based split used in the balanced regime.

Open question worth a node: at fixed N, how much of the macro-F1 drop is
imbalance and how much is cardinality? Running balanced and natural at the same
N ladder separates them.

## 4. Frozen probe → full fine-tuning confirmation

Everything here measures *linear separability of a fixed representation* as a
function of label cardinality. That is a well-posed question and it is the
regime the granularity literature is defined in, but it is **not** the same as
full fine-tuning, where the encoder can reshape itself around the label space.

Confirmation plan: full fine-tune SWIN at two or three points on the ladder
(N = 200, 2000, and the ceiling) and check whether the anchor-set curve has the
same sign and rough magnitude. Expensive — one GPU-day per point — so it is a
confirmation, not a sweep.

## 5. Head type

Linear softmax only, currently. ArcFace / cosine heads are standard at high
cardinality and are what the project's own SWIN runs use, so the cardinality
curve under an ArcFace head is the more directly comparable number. The
`train_probe` seam is the only thing that changes.

## 6. Taxonomy-aware sampling

Classes are currently sampled uniformly at random, so growing N adds
taxonomically scattered species. Real label-space growth is not random — it
densifies within genera and families. A "dense" sampler that fills out whole
genera before adding new ones should degrade far faster at the same N, which
would show that *N alone is the wrong x-axis* and separability is the right one.
The manifest already carries `family`/`genus`/`species` columns.

## 7. Granularity as the real independent variable

`mean_nn_centroid_distance` is already logged per task. Once the sweep has run,
regress macro-F1 on (log N) versus on separability and report which explains
more variance. If separability wins, the headline result of the whole line is
that label *count* is a nuisance parameter and label *geometry* is the signal.

# Notes — Scaling Law for Label Count

Running log of hypotheses, experiments tried, and findings for this
direction. Maintained by the exploration and experimentation agents;
reviewed by the reflection agent.

## Guiding questions

- How does fine-grained classification performance scale as the number of
  classes grows (e.g. toward 30k/40k/50k species-level labels)?
- Does the scaling behavior differ from typical coarse-grained
  classification scaling laws?
- Where do we see diminishing returns, and does that point vary by
  architecture (SWIN vs. CNN vs. hybrid)?

## Candidate methods

- Train the same architecture on nested label-count subsets (e.g. top-1k,
  5k, 10k, 20k, full label set) and fit accuracy/loss vs. label-count curves.
- Compare per-class sample efficiency as label count grows (does accuracy
  degrade primarily from more classes or from thinner per-class data?).
- Cross-reference with `interpretability/` — does representation quality per
  layer shift as label count scales up?

## Proposed dataset design: controlling for distribution shape

Prior experiments built nested subsets (top-1k, 5k, 10k, 20k) by taking the
most frequent labels from the combined Kaggle21+22 pool. This confounds
**label count (K)** with **distribution shape**: every subset shares the same
dominant head classes, and growing K just appends long-tail species with a
handful of images each. The head (which drives most of the loss/accuracy)
never changes, so performance barely moves — the scaling signal was masked
by the imbalance, not absent.

To isolate label count as a variable, decouple it from samples-per-class (n):

- **Fixed-n, random-K subsets.** For each target K, restrict to species with
  ≥ n images (n = a fixed quota, e.g. 20–50, near the count of the rarest
  class to include), then **randomly** draw K species from that eligible
  pool — not top-K by frequency, so the same head classes aren't reused at
  every step. From each chosen species sample exactly n images. Total
  N = K·n grows linearly and purely as a function of K, with per-class
  difficulty held constant, so any change in loss/accuracy is attributable
  to label count alone.
- **Balanced held-out eval set**, built the same way (disjoint images, same
  n-per-class quota) — otherwise accuracy is dominated by whichever few
  high-frequency classes survive in the eval split, hiding tail behavior.
- **Second axis: sample efficiency.** Separately fix K and sweep n (e.g.
  K=1000, n ∈ {5,10,20,50,100,...}) to get the classic samples-per-class
  curve, independent of label count. Running both sweeps lets loss/accuracy
  be fit as a function of K and n separately, and lets us check for
  interaction between them.
- **Prefer downsampling the head over bootstrapping the tail.** Upsampling
  rare classes (duplicating images) changes what's being learned — near
  duplicates let the model partially memorize rather than generalize, and
  inflate N without adding information. Capping every class at a fixed
  quota keeps "one sample = one unit of information" true across the whole
  sweep. Bootstrapping could be tested later as its own orthogonal axis, not
  as a substitute for controlling K.
- **Taxonomic stratification.** Stratify class selection by family/genus
  where possible, so an increase in K doesn't just happen to pull in one
  taxonomic group — otherwise family/genus composition becomes a second
  hidden confound alongside label count.
- **Merging Kaggle21 + Kaggle22.** `scientificNameEncoded` is not necessarily
  consistent across the two datasets — join on `scientificName` (or
  `genus`+`species`) strings when pooling, not on the encoded IDs. Also
  double check whether the existing `configs/*_15k.yml` / `_21k.yml` /
  `_50k.yml` naming refers to sample counts or class counts before reusing
  them — they likely come from the old confounded top-K scheme.

## Log

_(empty — first entries go here as experiments are run)_

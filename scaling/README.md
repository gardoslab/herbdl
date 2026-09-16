# Label-cardinality scaling

At what number of unique labels does fine-grained recognition performance
actually degrade, and is aggregate F1 hiding the answer?

The project's existing SWIN result is ~0.7 F1 at ~15k herbarium species, but
that number is carried by the well-represented classes. Two things are missing
from the literature and both are measured here:

1. **A controlled N-sweep on a fixed image pool.** Published work either varies
   the class count across *different* datasets (confounding cardinality with
   domain and with images-per-class) or varies model scale at a fixed class
   count. Nobody sweeps N with everything else pinned.
2. **Head/tail-decomposed metrics** instead of an aggregate. See
   `metrics.py`.

## What is held fixed

| Quantity | How |
|---|---|
| Representation | Frozen backbone, features extracted once and cached; every task reads the same vectors |
| Images per class | Exactly `k_train` train + `k_eval` eval at every N |
| Which images | Per-class permutation seeded by `(seed, class)` only — never by N |
| Which classes | **Nested**: the N=50 label set is a subset of N=200, ⊂ N=1000, … |
| Optimisation budget | A constant number of gradient steps, *not* epochs |

That last one is not a detail. With epochs fixed, the number of steps grows with
N (more classes at fixed images/class = a bigger dataset), the small-N probes
come out undertrained, and the accuracy curve inverts — it comes out *increasing*
in N. This was caught in the synthetic smoke test before any cluster time was
spent. `probe_train_accuracy` is logged per task, and a built-in convergence
check re-trains the hardest task with 2× the budget and prints the delta.

## The headline measurement

Nested sampling makes the *anchor set* possible: the first 50 classes of each
seed's shuffled order are present at every rung of the ladder, with byte-identical
training and evaluation images. Their macro-F1 is reported at every N.

Those classes cannot get harder for any reason except that more labels were added
around them. Same images, same features, same per-class evidence, same optimiser
budget — only the number of competing labels changes. Everything else in the
table is context for that one number.

A nearest-class-mean decision rule is evaluated alongside the learned probe on the
same centroids. NCM has no optimisation in it at all, so if the two curves have the
same shape the cause is the geometry of the label space rather than anything about
fitting a larger head.

## Metrics beyond macro-F1

Computed from per-class tp / predicted-support / true-support, so nothing ever
materialises an N×N confusion matrix (15k² float64 is 1.8 GB).

- `f1_gini` — concentration of the per-class F1 vector. 0 = every class equally
  served, →1 = the score belongs to a few classes.
- `n_eff`, `n_eff_frac` — `exp(H(p))` over the normalised per-class F1 vector.
  Reads as *"this N-way model behaves like an N_eff-way model."* Equals N when
  classes are served evenly, regardless of the absolute F1 level.
- `f1_mass` — `sum(per-class F1)` = `macro_f1 × N`. Total recognition, in units
  of perfectly-solved classes. Pairs with `n_eff`: mass says how much, `n_eff`
  says how evenly spread.
- `dead_class_rate`, `never_predicted_rate` — the fraction of the label space the
  model has effectively abandoned. Expected to be the cleanest curve in the study.
- `f1_decile_profile` — per-class F1 by decile, the direct "carried by the head"
  picture.
- `f1_by_separability_decile`, `corr_f1_separability` — in the balanced regime
  every class has equal evidence, so frequency cannot explain who wins. Classes
  are instead ordered by distance to their nearest other class centroid in
  feature space. This is the balanced-regime stand-in for head/tail, and it tests
  the dataset-granularity prediction directly.
- `macro_f1_{few,medium,many}_shot` — the conventional frequency split, activated
  automatically when the regime is imbalanced (see backlog item 3).

## Layout

```
scaling/
  run_scaling.sh        fixed run contract; never edit on a child branch
  configs/scaling.yml   what child nodes edit
  data.py               manifest loading + the census printed at run start
  features.py           frozen-backbone feature cache (shards, atomic, resumable)
  sweep.py              sampling, probes, reporting
  metrics.py            head/tail decomposition
  BACKLOG.md            deferred work, incl. the architecture ablation
```

## Reading a run log

The census block comes first and states exactly what pool was loaded — column
names, image count, class count, the images-per-class distribution, and a path
existence check. Every number below it depends on that being the intended pool.

Then one line per `(seed, N)` task, then the summary table, then the anchor-set
headline and the decile profiles.

## Scope

This measures linear separability of a fixed representation as a function of
label cardinality. That is the regime the dataset-granularity literature is
defined in, and it is what makes a 5-point × 3-seed sweep affordable at all —
but it is not full fine-tuning, where the encoder reshapes itself around the
label space. Backlog item 4 is the confirmation plan.

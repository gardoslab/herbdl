# How far does image recognition fall as you add labels?

Three runs, 16 September 2026. This document supersedes an earlier version whose
headline numbers were wrong; see "What we got wrong the first time" below.

| Node | Run | What it did |
|---|---|---|
| `d2e95774` | `09cfc736` | First sweep. Fixed 4,000 training steps at every rung. **Superseded — undertrained.** |
| `0ac5ea74` | `a9c8b04e` | Same sweep, but training budget measured rather than assumed. |
| `5bf244c6` | `6fd4250b` | Same again, using a SWIN fine-tuned on the 15k herbarium labels, scored on held-out images. |

Raw numbers on SCC under
`/projectnb/herbdl/workspaces/faridkar/herbdl/scaling/results/`.

---

## The question

We'd seen SWIN get about 0.7 F1 on roughly 15,000 plant species, and suspected
that number was flattering — that a handful of species were doing the heavy
lifting while most were being quietly missed. We wanted to know whether there's
some particular number of labels where recognition "breaks."

The problem with answering that from published work is that every study compares
*different datasets*. A 200-class bird dataset and a 15,000-class plant dataset
differ in the number of labels, yes — but also in photo quality, in how many
example images each class gets, in how similar the classes look. If accuracy is
lower on the big one, you can't tell which of those four things caused it.

So the whole point here was to change **only** the number of labels, and hold
everything else still.

---

## How the experiment was built

### Step one: turn every photo into a fingerprint

We ran every herbarium photo through a frozen SWIN model once, without modifying
it. Each photo becomes a list of 1,024 numbers — a compact numerical description
of what's in the image. Those get saved to disk.

Why do it this way? Because every experiment afterward is then just arithmetic on
saved numbers. The whole label-count comparison takes minutes instead of weeks of
retraining, which is what made it affordable to test five label counts across
three random seeds and six training budgets.

The tradeoff is real: we measure how well a *fixed* representation supports
different numbers of labels, not how a model retrained end-to-end at each rung
would behave. The third run below partly addresses this by swapping in a
representation that was trained on this task.

### Step two: build a ladder of label counts

Five versions of the same task: tell apart 50 species, then 200, then 1,000, then
5,000, then everything eligible. The code caps the top rung at however many
species actually qualify and says so in the log rather than silently substituting
a number.

### Step three — the important one: give every species the same data

In every version, every species gets exactly 10 photos to learn from and 5 to be
tested on. Not "on average 10." Exactly 10.

So moving from 50 species to 13,000 changes nothing about how much evidence each
species has. The only thing that changes is how many other species must be ruled
out.

This also means the pool is not long-tailed by construction. Worth knowing:
Herbarium-2022 isn't very long-tailed to begin with — images per species run 2 to
73, median 52, because the organisers capped the head. The top 10% of species
hold only 14.9% of the images.

### Step four: keep 50 species present at every rung

Fifty species appear in all five versions with literally the same photos each
time. Call them the anchor species. If their scores drop as the ladder climbs, we
know exactly why: nothing about them changed, only the crowd around them grew.

### Step five: repeat three times

Different random species selections per seed. Results were stable — see the
spread columns below.

---

## How we scored it

**Plain accuracy** asks what fraction of test photos we got right. The trouble is
that if 200 species are easy and 13,000 are hopeless, accuracy still looks
respectable because the easy ones are numerous in the test set.

**Macro-F1** scores each species separately, then averages with every species
counting equally. A rare species counts exactly as much as a common one. This is
the number that catches the problem we suspected, and it's the one to read
throughout.

We also track two things that describe the *shape* of the outcome rather than its
average: the **dead fraction** (species that never get a single correct
prediction) and the **Gini** of per-species scores (0 = everyone scores the same,
1 = maximally lopsided).

---

## The training budget, and why it needed measuring

The small classifier that sits on top of the fingerprints originally got 4,000
training steps at every rung. That sounds fair — identical compute for every
task.

It isn't. More species means more training photos, so the same 4,000 steps buys
very different amounts of practice:

| N | training images | passes over the data at 4k steps |
|---:|---:|---:|
| 50 | 500 | 4,000 |
| 200 | 2,000 | 2,048 |
| 1,000 | 10,000 | 410 |
| 5,000 | 50,000 | 82 |
| 13,548 | 135,480 | **30** |

(At N=50 the batch clamps to the 500 training images, so each step is one
full-batch pass.)

Fixing *epochs* instead was tried first and made the accuracy curve come out
**increasing** in N — plainly an artifact of handing larger tasks more compute.
Neither knob is neutral.

So the second run stopped choosing and **measured** instead. For each rung it
trains at 4k, 8k, 16k, 32k, 64k, 128k steps and keeps going until doubling the
budget buys less than 0.005 macro-F1.

One detail matters here. The learning rate follows a schedule that rises then
tapers to near zero at the end of whatever budget it's given. A run *planned* for
4,000 steps and a run *stopped early* at 4,000 steps are different objects — the
first has landed, the second is mid-flight. Every budget above is a complete run
from scratch, not a snapshot.

---

## Result 1 — the generic representation (ImageNet-22k SWIN)

671,817 images, 15,501 species, 13,548 eligible at 10 train + 5 eval.
Converged values, mean ± sd over three seeds:

| N | macro-F1 | anchor-50 F1 | Gini | dead frac | steps to converge |
|---:|---:|---:|---:|---:|---:|
| 50 | 0.856 ±0.041 | 0.856 ±0.041 | 0.083 | 0.000 | 8,000 |
| 200 | 0.742 ±0.024 | 0.737 ±0.051 | 0.146 | 0.003 | 29,333 |
| 1,000 | 0.595 ±0.014 | 0.587 ±0.024 | 0.227 | 0.035 | 64,000 |
| 5,000 | 0.438 ±0.002 | 0.453 ±0.059 | 0.327 | 0.107 | 128,000 |
| 13,548 | **0.336** ±0.001 | 0.358 ±0.060 | 0.414 | 0.197 | **>128,000** |

### There is no breaking point

No threshold, no cliff, no knee. The decline begins at 50 species and continues
smoothly. Every doubling of the label set costs about **11%** of macro-F1.

On a log-log plot the slope is **−0.165** with R²=0.974 — a good straight line.

### Adding competitors alone destroys performance

The anchor species — identical photos throughout — go from **0.856 alone to 0.358
among 13,548 competitors**, a 58% drop. Same species, same training photos, same
test photos, same model. The only thing that changed is how many other labels
existed to be confused with.

This is the result worth defending. It can't be explained by data quality or
class difficulty, because both were held fixed by construction. It removes the
confound that undermines every cross-dataset comparison in the literature.

### Concentration appears without any imbalance

Per-species scores at the top rung, best tenth to worst:

```
0.785  0.614  0.512  0.428  0.352  0.275  0.222  0.166  0.003  0.000
```

About a fifth of species (19.7%, or roughly 2,670) never get a single correct prediction,
and score inequality rises from 0.083 to 0.414 across the ladder — **even though
every species had exactly the same 10 training photos**. The lopsidedness is not a
long-tail effect; it emerges purely from label competition.

### Training cost grows with the label space

Look at the last column: 8,000 steps at N=50, 64,000 at N=1,000, still climbing
past 128,000 at N=13,548.

Part of what people experience as "large label spaces are harder" is not a lower
ceiling but a **much more expensive climb to it**. Those are different problems
with different fixes. On a fixed budget, a large label space looks far worse than
it fundamentally is, and you may blame your representation when your schedule is
the real constraint.

### Caveat: the top rung is a floor

At 13,548 species the probe was still improving when it hit the 128,000-step cap
(+0.009 on the final doubling). **0.336 is a lower bound**, and the true decline
is gentler still than −0.165.

---

## Result 2 — the fine-tuned representation (SWIN_BASE_BASELINE, 15,501 labels)

Same ladder, but the frozen features come from a SWIN fine-tuned on the full
Kaggle-2022 label space.

### Leakage was the central design problem

That model was fine-tuned on `train_2022.json`. The sweep originally drew both
probe-train and probe-eval images from the same pool — so a fine-tuned backbone
would have been scored on images it had memorised, flattening the curve for an
entirely uninteresting reason.

Probe-eval images therefore come only from `val_2022.json`, the split held out of
fine-tuning, and the run asserts the two pools share no image path:

```
train pool : 671,817 images / 15,501 classes  (median 52 img/class)
eval  pool : 167,955 images / 15,334 classes  (median 12 img/class)
path overlap between pools : 0 (clean)
eligible at k_train=10 / k_eval=5 : 12,689 classes
```

The top rung is 12,689 rather than 13,548 because held-out images are scarcer.

### The numbers

| N | macro-F1 | anchor-50 F1 | Gini | dead frac | steps to converge |
|---:|---:|---:|---:|---:|---:|
| 50 | 0.985 ±0.002 | 0.985 ±0.002 | 0.013 | 0.000 | 8,000 |
| 200 | 0.967 ±0.008 | 0.968 ±0.006 | 0.027 | 0.000 | 8,000 |
| 1,000 | 0.914 ±0.005 | 0.908 ±0.008 | 0.061 | 0.000 | 8,000 |
| 5,000 | 0.804 ±0.002 | 0.804 ±0.030 | 0.119 | 0.003 | 16,000 |
| 12,689 | **0.712** ±0.002 | 0.701 ±0.031 | 0.169 | 0.013 | 32,000 |

### Task-adapted features change the shape, not just the height

This is the main finding of the comparison.

| | generic IN22k | fine-tuned 15k |
|---|---:|---:|
| loss per doubling of N | 10.8% | **3.9%** |
| log-log slope | −0.165 | **−0.057** |
| gap at N=50 | — | +0.130 |
| gap at top rung | — | **+0.376** |

The two curves are not parallel lines offset by a constant. The fine-tuned one is
genuinely **flatter** — about a third the slope — which is why the gap between
them nearly triples from the bottom of the ladder to the top.

So cardinality pressure is not a fixed tax you pay regardless of representation.
A representation built for this label space absorbs most of it.

### The original 0.7 reproduces, and is healthier than feared

0.712 at 12,689 species on held-out images closely matches the ~0.7 that prompted
this whole investigation — a useful independent check that this pipeline measures
the same thing the original training did.

But the "carried by the head" worry does not hold up. Per-species deciles:

```
fine-tuned:  1.000  0.929  0.889  0.823  0.776  0.724  0.659  0.580  0.480  0.261
generic:     0.785  0.614  0.512  0.428  0.352  0.275  0.222  0.166  0.003  0.000
```

Only **1.3%** of species score zero (roughly 165 of 12,689), against 19.7% for the generic
backbone, and even the worst tenth averages 0.261. The 0.7 is a broadly competent
model, not a few winners propping up a corpse.

### Fine-tuned features are also cheaper to fit

The top rung converged at **32,000 steps**, against **>128,000** for the generic
one. Task-adapted features don't only score higher; they reach their ceiling with
far less optimisation. This reinforces the compute-to-convergence finding above.

### Two caveats

The fine-tuned curve is a **worse** power law (R²=0.896 vs 0.974) — it's flat from
50→200 then steepens, so "3.9% per doubling" averages over a curve that isn't
really straight. The comparison between arms is solid; the single exponent is a
rougher summary than it appears.

And this arm cannot be read as intrinsic task difficulty: its features were
optimised on exactly this 15k decision. It answers the practical question — does
task-adapted pretraining flatten the cardinality penalty? — and the answer is
clearly yes. Held-out eval is what keeps even that question answerable.

---

## What feature-space geometry does *not* explain

For each species we measured the distance from its average fingerprint to the
nearest other species' average, then asked whether that predicts its score.

It predicts weakly: correlation **+0.310** generic, **+0.373** fine-tuned. Sorting
species by isolation instead of by score flattens the profile dramatically
(generic: 0.481 → 0.207, versus 0.785 → 0.000 when sorted by score).

So "visually distinctive species win, crowded species lose" is not the
explanation. Something else decides winners and losers, and we don't yet know
what. This was a prediction we made and it was wrong both times.

---

## What we got wrong the first time

The first run fixed the budget at 4,000 steps, and three of its headline numbers
were substantially artifacts of undertraining. Recording them because the size of
the error is itself instructive:

| Claim | Reported | Corrected |
|---|---:|---:|
| macro-F1 at ~13.5k species | 0.159 | **0.336** (a floor) |
| loss per doubling | 18.6% | **10.8%** |
| species scoring zero | 50.9% | **19.7%** |
| anchor-50 drop | −81.7% | **−58.2%** |

One claim was pure artifact. A no-training baseline (average each species'
fingerprints, assign each test photo to the nearest average) appeared to *beat*
the trained probe at the top rung, 0.206 to 0.159. Properly trained, the probe
scores 0.336 and wins at every rung. There was no finding there — just a starved
classifier.

The earlier write-up also claimed the shape "held" at 8,000 steps. That was an
overreach from a single doubling, on a curve still climbing steeply. One extra
point does not license a stability claim.

What survived: no threshold, the anchor effect, and concentration under equal
evidence. Direction and shape held; magnitudes did not.

---

## Where this leaves us

**Findings we'd stand behind:**

1. Performance declines smoothly with label count — no threshold, no cliff — at
   about 11% per doubling for a generic representation.
2. Adding competing labels alone degrades performance on classes whose data never
   changed: 0.856 → 0.358 on identical images.
3. Severe score concentration emerges with perfectly equal per-class evidence, so
   it is not fundamentally a long-tail effect.
4. Task-adapted pretraining changes the *slope*, not just the intercept — 3.9% per
   doubling versus 10.8%.
5. Compute-to-convergence grows steeply with label count, and is separable from
   representation quality.

**Open questions:**

- Which species land in the bottom decile, given that feature-space isolation
  explains so little of it?
- Where does the generic curve actually converge at the top rung, past 128k steps?
- Does the pattern hold for architectures other than SWIN? See `BACKLOG.md`.

Note that backlog item 3 (natural long-tailed regime) is largely moot on
Herbarium-2022 given the census above — testing real imbalance needs the 2021 or
merged GBIF pool instead.

A further opportunity surfaced while resolving checkpoints: the SCC tree holds
**118 loadable checkpoints**, including LR/schedule sweeps at 15,501 labels and a
second regime at 64,497 labels. That is a ready-made axis for asking whether the
cardinality slope depends on *how well* the backbone was trained.

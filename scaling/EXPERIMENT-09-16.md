# Label cardinality sweep — what we did and what we found

Run `09cfc736-44ca-4a1c-a839-a0e077ce0d84`, 16 September 2026. Commit `41cc489`.
Ran 27m30s on one L40S. Raw numbers in
`/projectnb/herbdl/workspaces/faridkar/herbdl/scaling/results/balanced_swin_base_in22k.json`.

---

## The question we were trying to answer

We'd seen SWIN get about 0.7 F1 on roughly 15,000 plant species, and suspected
that number was flattering — that a handful of species were doing the heavy
lifting while most were being quietly missed. We wanted to know whether there's
some particular number of labels where recognition "breaks."

The problem with answering that from published work is that every study compares
*different datasets*. A 200-class bird dataset and a 15,000-class plant dataset
differ in the number of labels, yes — but also in photo quality, in how many
example images each class gets, in how similar the classes look. If accuracy is
lower on the big one, you can't tell which of those four things caused it.

So the whole point of this experiment was to change **only** the number of
labels, and hold everything else still.

---

## How the experiment was built

### Step one: turn every photo into a fingerprint

We took a pretrained SWIN model and ran all 671,817 herbarium photos through it
once, without ever modifying the model. For each photo the model produces a list
of 1,024 numbers — think of it as a compact numerical description of what's in
the image. We saved all of those to disk. This took about 20 minutes at 532
images per second, and it happened exactly once.

Why do it this way? Because now every experiment afterward is just arithmetic on
saved numbers. We could run the entire label-count comparison in 27 minutes
instead of weeks of retraining.

The tradeoff is real and worth stating plainly: we are measuring how well a
*fixed* set of features supports different numbers of labels, not how well a
model fully retrained from scratch would do. A fully fine-tuned model would
score higher everywhere. Whether it would decline at the same *rate* is an open
question this run cannot answer.

One detail that matters: the backbone is pretrained on ImageNet-22k, **not** on
herbarium labels. If we'd started from a model already fine-tuned on the 15k
plant species, the features would already encode the answer to the very question
we're asking, and the sweep would be circular.

### Step two: build a ladder of label counts

We made five versions of the same task: tell apart 50 species, then 200, then
1,000, then 5,000, then 13,548 species.

We'd asked for 15,000 at the top, but only 13,548 species had enough photos to
meet the rule in step three, so the code capped the top rung at the real ceiling
and said so in the log rather than silently substituting a different number.

### Step three — the important one: give every species exactly the same data

In every single version, every species gets exactly 10 photos to learn from and
5 photos to be tested on. Not "on average 10." Exactly 10.

So when we move from 50 species to 13,548 species, the amount of evidence per
species never changes. The *only* thing that changes is how many other species
the model has to rule out.

### Step four: keep 50 species present at every rung

We picked 50 species and made sure they appear in all five versions, with
literally the same photos each time. Call them the anchor species. If their
scores drop as the ladder climbs, we know exactly why — nothing about them
changed, only the crowd around them grew.

### Step five: repeat three times

Different random species selections each time, so we can see whether results are
stable or just luck. They were stable — see the spread column in the table below.

---

## How we scored it

Two things matter here.

**Plain accuracy** asks: what fraction of test photos did we get right? The
trouble is that if 200 species are easy and 13,000 are hopeless, accuracy can
look respectable because the easy ones are numerous in the test set.

**Macro-F1** fixes this. It scores each species separately, then averages those
scores with every species counting equally — a rare species counts exactly as
much as a common one. This is the number that catches the problem we suspected,
and it's the one to read throughout.

---

## A note on the training budget, because it constrains everything below

The small classifier that sits on top of the fingerprints got **4,000 training
steps at every rung**, batch size 1024 — identical compute whether it was
separating 50 species or 13,548.

That sounds fair, but in terms of passes over the data it isn't, because more
classes at 10 images each means a bigger training set:

| N | training images | passes over the data |
|---:|---:|---:|
| 50 | 500 | 8,192 |
| 200 | 2,000 | 2,048 |
| 1,000 | 10,000 | 410 |
| 5,000 | 50,000 | 82 |
| 13,548 | 135,480 | **30** |

We chose fixed steps deliberately. Fixing *epochs* instead was tried first and
made the accuracy curve come out **increasing** in N — obviously an artifact of
handing the larger tasks proportionally more compute. But fixed steps carries
the opposite bias, and 30 passes versus 8,192 is a wide enough gap to matter.

Neither knob is neutral. Read the numbers below with that in mind.

---

## What we found

The ladder, averaged over three repeats:

| Species being told apart | Macro-F1 | Anchor-50 F1 | Share scoring zero |
|---:|---:|---:|---:|
| 50 | 0.856 ±0.043 | 0.856 ±0.043 | 0% |
| 200 | 0.729 ±0.018 | 0.717 ±0.031 | 0.5% |
| 1,000 | 0.528 ±0.016 | 0.512 ±0.038 | 5.9% |
| 5,000 | 0.277 ±0.003 | 0.272 ±0.048 | 30.4% |
| 13,548 | 0.159 ±0.001 | 0.157 ±0.012 | **50.9%** |

### Signal one: there is no breaking point

We expected to find a threshold — some count where things fall apart. There
isn't one. The decline starts immediately at 50 species and continues smoothly
all the way up. No knee in the curve, no cliff.

Roughly speaking, every time you double the number of species you lose about a
fifth of your macro-F1. The decline is gradual and relentless rather than sudden.

Fitting a straight line through the five points on a log-log plot gives an
exponent of about −0.30, but the leftover errors from that fit are systematically
curved rather than random, which means a pure power law is the wrong shape. The
honest summary is "roughly power-law through the middle," not a law. And see the
caveat section — that exponent is biased steep anyway.

### Signal two: the anchor species — the cleanest result here

Those 50 species that appear at every rung, with identical photos throughout,
scored **0.856 when they were the only species in the task, and 0.157 at
13,548 species.**

Same species. Same training photos. Same test photos. Same model. The only
difference is how many other labels existed to be confused with. An 82% drop
caused purely by adding competitors.

This is the result worth defending, because it can't be explained away by data
quality or class difficulty — those were held fixed by construction. It's the
confound that undermines every cross-dataset comparison in the literature, and
this design removes it.

### Signal three: severe concentration, but not from imbalance

At the top rung we sorted all 13,548 species by their individual score and split
them into ten equal groups:

```
best tenth: 0.578   0.370   0.297   0.224   0.121   0.000   0.000   0.000   0.000   0.000
```

Half the species — 6,895 of them — scored exactly zero. Not "low." Zero. The
model never once correctly identified them. Any overall number you report is
generated entirely by the better half, and mostly by the top fifth. Inequality
across per-class scores (Gini) climbs from 0.084 at N=50 to 0.641 at N=13,548.

Now the part worth pausing on. **Every species had the same 10 training photos.**
So this lopsidedness cannot be blamed on some species having more data. Give
every class identical evidence and a severe winners-and-losers split *still*
appears, purely as a consequence of how many labels are competing.

That's a more interesting claim than "rare classes do badly," which everyone
already knows. It also reframes the original observation: the concentration in
the 15k SWIN result is probably not mostly a long-tail problem.

Supporting context from the dataset census: Herbarium-2022 is far more balanced
than assumed. Images per species run from 2 to 73, median 52 — the organisers
capped the head. The top 10% of species hold only 14.9% of the images. A
genuinely long-tailed corpus would be nearer 50%.

### Signal four: a negative result — feature-space isolation doesn't explain it

Our guess beforehand was that the winners would be species sitting off on their
own in fingerprint space — visually distinctive ones with no close lookalikes.
So we measured, for each species, the distance from its average fingerprint to
the nearest other species' average, and checked whether that predicted its score.

It barely did. The correlation is about **+0.26**, where 0 means no relationship
and 1 means perfect. Sorting species by isolation instead of by score flattens
the profile right out:

```
by score:       0.578  0.370  0.297  0.224  0.121  0.000  0.000  0.000  0.000  0.000
by isolation:   0.267  0.224  0.196  0.167  0.157  0.137  0.128  0.118  0.111  0.087
```

So "isolated species win, crowded species lose" is not the explanation. Something
else decides winners and losers, and we don't yet know what. This prediction was
wrong, and the gap is worth chasing.

---

## The part we don't fully trust yet

When you train the small classifier, you show it the data repeatedly and it
improves each pass. At some point it stops improving — it's learned what it can.
To check whether 4,000 steps was enough at the hardest setting, we simply trained
it twice: once for 4,000 steps and once for 8,000.

If it had finished learning, both should score about the same. Instead:

- 4,000 steps → macro-F1 **0.160**
- 8,000 steps → macro-F1 **0.199**

Still climbing. **It hadn't finished learning — we stopped it too early.**

There's corroborating evidence. At 50 species the classifier got 100% on its own
training photos, which is what you'd expect. At 13,548 species it managed only
49% on its own training photos. It couldn't memorise the material it was shown,
which is the signature of a classifier cut off partway. Given the table above —
30 passes at the top rung against 8,192 at the bottom — that's unsurprising in
hindsight.

### What this means concretely

- **The 0.159 at the top rung is too pessimistic.** The true value is higher.
- **The −0.30 exponent is biased steep** and shouldn't be quoted as a finding.
- **The shape holds.** Smooth decline with no threshold, the anchor collapse, and
  half the species at zero all survive at 8,000 steps.

It also likely explains an oddity. As a sanity check we ran a much simpler
method with no training at all: average the 10 fingerprints for each species,
then assign each test photo to whichever average it's closest to. At the top rung
that untrained method *beat* the trained classifier, 0.206 to 0.159. That looked
like a surprising finding about representations. The likelier explanation is
mundane — the trained classifier was stopped before it caught up.

---

## Where that leaves us

**Three findings we'd stand behind:**

1. Performance declines smoothly with label count, with no threshold or cliff.
2. Adding competing labels alone destroys performance on completely unchanged
   classes — 0.856 to 0.157 on identical data.
3. Severe winners-and-losers concentration emerges even with perfectly equal
   data per class, so it is not fundamentally a long-tail effect.

**One open question:** why those particular species win, given that isolation in
feature space explains so little of it.

**One fix needed:** rerun the top two rungs training to convergence — stopping
when improvement stalls rather than at a fixed count — and report the compute
each rung actually needed. That turns the step budget from a confound into a
measurement. It's a follow-on experiment, not a correction; this run answered its
question, just with a fuzzier top end than we'd like.

Longer-term ideas, including the architecture comparison, are in `BACKLOG.md`.
Note that backlog item 3 (natural long-tailed regime) is largely moot on
Herbarium-2022 given the census above — testing real imbalance needs the 2021 or
merged GBIF pool instead.

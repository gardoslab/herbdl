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

## Log

_(empty — first entries go here as experiments are run)_

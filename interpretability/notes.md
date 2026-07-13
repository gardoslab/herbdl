# Notes — Interpretability of Fine-Grained Performance

Running log of hypotheses, experiments tried, and findings for this
direction. Maintained by the exploration and experimentation agents;
reviewed by the reflection agent.

## Guiding questions

- Which heads/layers carry fine-grained discriminative signal (as opposed to
  coarse-grained/family-level signal)?
- How does fine-grained understanding differ across architecture paradigms
  (convolutional, transformer, hybrid)?
- Does linear probing at different depths/layers reveal where fine-grained
  separability emerges?

## Candidate methods

- Linear probing across layers for each architecture (SWIN, a CNN baseline,
  SWIN-CLIP hybrid), measuring fine-grained (species) vs. coarse-grained
  (family/genus) probe accuracy per layer.
- Attention head ablation/importance scoring (transformer-based models only).
- CKA / representational similarity comparisons across architectures at
  matched relative depth.

## Log

_(empty — first entries go here as experiments are run)_

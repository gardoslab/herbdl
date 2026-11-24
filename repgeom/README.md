# repgeom — post-hoc representation geometry of a 15.5k-class herbarium classifier

This line of work does **not** train anything. It takes an already fine-tuned
SWIN classifier over 15,501 Kaggle-Herbarium-2022 species and asks what its
learned representation looks like at that label cardinality:

1. **What separates well-recognised classes from badly-recognised ones?**
   Per-class recall is regressed on geometry (intra-class spread, margin to the
   nearest class, classifier-weight norm) and on corpus/taxonomy covariates
   (training frequency, number of congeneric species).
2. **Is the error structure taxonomic?** Are confusions concentrated within
   genus / within family, and does a crowded genus cost accuracy?
3. **Where on the sheet does the evidence come from?** Swin-base pools a 7x7
   final feature map by a plain mean, so the logit decomposes *exactly* into 49
   per-position contributions — a spatial evidence map with no saliency
   approximation and no extra backward pass.

Everything derives from one forward pass over the 167,955-image validation
split, cached outside the run directory so later nodes reuse it.

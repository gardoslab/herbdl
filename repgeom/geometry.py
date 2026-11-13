"""Class-manifold geometry of the fine-tuned representation.

The question this file exists to answer is "what does the feature space look
like when you ask it to hold 15,501 classes", and specifically whether the
*geometric* description of a class predicts how well it is recognised.

Two families of quantity:

  per-class   tightness (how concentrated a species' images are), margin (how
              far the nearest other species sits), and the classifier row that
              reads the class out. These join the per-class regression.

  global      neural-collapse descriptors (NC1 within/between scatter, equinorm,
              self-duality). The collapse literature is almost entirely balanced
              and <=1000 classes; whether it survives 15.5k long-tailed classes
              is an open question, so the numbers are reported plainly rather
              than argued for.

All heavy linear algebra is chunked on the GPU: a 15,501 x 15,501 centroid
similarity is 961 MB in fp32 and is never materialised whole.
"""

import numpy as np
import torch

CHUNK = 2048


def class_centroids(feat, labels, n_classes, device):
    """Per-class mean of the pooled features, plus counts and the global mean."""
    C = feat.shape[1]
    sums = torch.zeros(n_classes, C, dtype=torch.float64, device=device)
    counts = torch.zeros(n_classes, dtype=torch.float64, device=device)
    y_all = torch.as_tensor(labels, dtype=torch.long)

    for lo in range(0, len(feat), CHUNK):
        x = torch.as_tensor(feat[lo:lo + CHUNK]).to(device).double()
        y = y_all[lo:lo + CHUNK].to(device)
        sums.index_add_(0, y, x)
        counts.index_add_(0, y, torch.ones(len(y), dtype=torch.float64, device=device))

    present = counts > 0
    mu = torch.zeros_like(sums)
    mu[present] = sums[present] / counts[present][:, None]
    # Global mean over *images*, which is what the collapse definitions use.
    mu_g = sums.sum(0) / counts.sum()
    return mu.float(), counts.cpu().numpy().astype(np.int64), mu_g.float(), present.cpu().numpy()


def intra_class_spread(feat, labels, mu, n_classes, device):
    """Mean cosine to, and mean L2 from, the image's own class centroid.

    Cosine is the scale-free notion of tightness; L2 is kept because the
    classifier is affine in the un-normalised feature and so is sensitive to
    radius as well as direction.
    """
    cos_sum = torch.zeros(n_classes, dtype=torch.float64, device=device)
    l2_sum = torch.zeros(n_classes, dtype=torch.float64, device=device)
    cnt = torch.zeros(n_classes, dtype=torch.float64, device=device)
    mu_n = torch.nn.functional.normalize(mu, dim=1)
    y_all = torch.as_tensor(labels, dtype=torch.long)

    for lo in range(0, len(feat), CHUNK):
        x = torch.as_tensor(feat[lo:lo + CHUNK]).to(device).float()
        y = y_all[lo:lo + CHUNK].to(device)
        xn = torch.nn.functional.normalize(x, dim=1)
        cos = (xn * mu_n[y]).sum(1)
        l2 = (x - mu[y]).norm(dim=1)
        cos_sum.index_add_(0, y, cos.double())
        l2_sum.index_add_(0, y, l2.double())
        cnt.index_add_(0, y, torch.ones(len(y), dtype=torch.float64, device=device))

    cnt_safe = cnt.clamp(min=1)
    return ((cos_sum / cnt_safe).cpu().numpy(), (l2_sum / cnt_safe).cpu().numpy())


def nearest_neighbour(mat, present, device, k=2):
    """Top-k cosine neighbours of each row, excluding the row itself.

    Used twice: on centroids (how crowded is this species' neighbourhood in
    feature space) and on classifier rows (how similar are the read-outs).
    """
    n = mat.shape[0]
    M = torch.nn.functional.normalize(mat.to(device), dim=1)
    absent = torch.as_tensor(~present, device=device)
    sims = np.full((n, k), np.nan, dtype=np.float32)
    idxs = np.full((n, k), -1, dtype=np.int64)

    for lo in range(0, n, CHUNK):
        hi = min(lo + CHUNK, n)
        s = M[lo:hi] @ M.T                      # (chunk, n)
        s[:, absent] = -2.0                     # classes with no val images
        s[torch.arange(hi - lo, device=device), torch.arange(lo, hi, device=device)] = -2.0
        v, i = s.topk(k, dim=1)
        sims[lo:hi] = v.cpu().numpy()
        idxs[lo:hi] = i.cpu().numpy()
        del s
    sims[~present] = np.nan
    return sims, idxs


def collapse_metrics(feat, labels, mu, mu_g, W, present, device):
    """Neural-collapse descriptors, computed over classes that have val images."""
    C = feat.shape[1]
    y_all = torch.as_tensor(labels, dtype=torch.long)

    sw = torch.zeros(C, C, dtype=torch.float64, device=device)
    n_used = 0
    for lo in range(0, len(feat), CHUNK):
        x = torch.as_tensor(feat[lo:lo + CHUNK]).to(device).float()
        y = y_all[lo:lo + CHUNK].to(device)
        d = (x - mu[y]).double()
        sw += d.T @ d
        n_used += len(y)
    sw /= n_used

    p = torch.as_tensor(present, device=device)
    Mc = (mu[p] - mu_g[None, :]).double()
    sb = (Mc.T @ Mc) / Mc.shape[0]

    # NC1: how small within-class scatter is relative to between-class scatter.
    # pinv rather than inv -- Sigma_B is rank <= K-1 but K >> C here, so it is
    # usually full rank; pinv just keeps a degenerate case from throwing.
    nc1 = float(torch.trace(sw @ torch.linalg.pinv(sb)).item() / Mc.shape[0])

    wn = W.norm(dim=1)
    mn = (mu[p] - mu_g[None, :]).norm(dim=1)
    # Self-duality: does the classifier row for a class point at that class's
    # centred centroid? Perfect collapse would put this at 1.
    dual = torch.nn.functional.cosine_similarity(W[p], mu[p] - mu_g[None, :], dim=1)

    return {
        "nc1_tr_sw_pinv_sb_over_k": nc1,
        "trace_ratio_sw_over_sb": float((torch.trace(sw) / torch.trace(sb)).item()),
        "equinorm_cv_classifier": float((wn.std() / wn.mean()).item()),
        "equinorm_cv_centroid": float((mn.std() / mn.mean()).item()),
        "self_duality_mean_cos": float(dual.mean().item()),
        "self_duality_median_cos": float(dual.median().item()),
        "n_classes_used": int(p.sum().item()),
    }, dual.cpu().numpy()

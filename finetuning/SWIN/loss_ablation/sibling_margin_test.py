"""Checks for the sibling-margin head and its NumPy rule. CPU only, synthetic data, a few seconds.

Run from finetuning/SWIN:  python loss_ablation/sibling_margin_test.py
Prints ALL CHECKS PASSED, or fails on the first broken check.
"""

import io
import sys
from contextlib import redirect_stdout

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from loss_ablation import sibling_margin as sm  # noqa: E402
from loss_ablation.losses import SiblingMarginHead, build_head  # noqa: E402

torch.manual_seed(0)
rng = np.random.default_rng(0)

# ---- a small taxonomy: genera of size 1..6, families of ~25 species; two species have no genus
C, D = 160, 24
genus, family, g, i = [], [], 0, 0
while len(genus) < C:
    size = (g % 6) + 1
    genus += [f"G{g}"] * size
    g += 1
genus = genus[:C]
family = [f"F{c // 25}" for c in range(C)]
# genera must not straddle families, or the hierarchy is inconsistent: realign
for c in range(C):
    family[c] = family[[k for k in range(C) if genus[k] == genus[c]][0]]
genus[0] = genus[1] = ""  # no genus given
counts = rng.integers(2, 80, size=C)
labels_flat = np.repeat(np.arange(C), 2)
gid, fid = sm.taxonomy_arrays(labels_flat, [genus[c] for c in labels_flat], [family[c] for c in labels_flat], C)
assert (np.bincount(gid)[gid[:2]] == 1).all(), "species without a genus must have no siblings"

# ---- the NumPy rule
m_sp = sm.species_margin(counts, m_max=0.15, tau=1.0)
assert abs(m_sp[counts.argmin()] - 0.15) < 1e-12, "the rarest species gets m_max"
order = np.argsort(counts)
assert (np.diff(m_sp[order]) <= 1e-15).all(), "margin must not grow with the number of training images"
y = rng.integers(0, C, size=40)
M = sm.pair_margin(y, m_sp, gid, fid, 1.0, 0.25)
for r, yy in enumerate(y):  # brute force from the definition
    for j in range(C):
        if j == yy:
            want = 0.0
        elif gid[j] == gid[yy]:
            want = m_sp[yy]
        elif fid[j] == fid[yy]:
            want = 0.25 * m_sp[yy]
        else:
            want = 0.0
        assert abs(M[r, j] - want) < 1e-12, (r, j)
n_sib, n_fam = sm.competitor_counts(gid, fid)
assert ((M > 0).sum(1) <= n_sib[y] + n_fam[y]).all()

# ---- bad manifests
for bad in ([("a", "G1", "F1"), ("a", "G2", "F1")], [("a", "G1", "F1"), ("a", "G1", "F2")]):
    try:
        sm.taxonomy_arrays([s for s, _, _ in bad] + [1], [g_ for _, g_, _ in bad] + ["H"], [f_ for _, _, f_ in bad] + ["F"], 2)
        raise SystemExit("an inconsistent manifest must raise")
    except (ValueError, KeyError, TypeError):
        pass

# ---- the head
head = SiblingMarginHead(D, C, counts, gid, fid, s=30.0, m_max=0.15, tau=1.0, family_weight=0.25, log_every=0)
head.train()
x = torch.randn(40, D)
yt = torch.as_tensor(y)
cos = F.linear(F.normalize(x, dim=1), F.normalize(head.weight, dim=1))
logits_eval = head(x)
assert torch.allclose(logits_eval, 30.0 * cos, atol=1e-5), "inference logits must be s * cosine"
logits = head(x, yt)
margin = (logits / 30.0 - cos)
assert torch.allclose(margin, torch.as_tensor(M, dtype=torch.float32), atol=1e-5), "applied margin differs from the rule"
assert torch.allclose(logits.gather(1, yt.view(-1, 1)), 30.0 * cos.gather(1, yt.view(-1, 1)), atol=1e-5), "true logit must be untouched"
loss = F.cross_entropy(logits, yt)
assert loss.item() >= F.cross_entropy(30.0 * cos, yt).item() - 1e-6, "competitor margins cannot lower the loss"
loss.backward()
assert torch.isfinite(head.weight.grad).all() and head.weight.grad.abs().sum() > 0

zero = SiblingMarginHead(D, C, counts, gid, fid, m_max=0.0, log_every=0)
zero.weight.data.copy_(head.weight.data)
assert torch.allclose(zero(x, yt), 30.0 * cos, atol=1e-5), "m_max = 0 must be a plain cosine softmax"
head.set_margin(0.0)
assert torch.allclose(head(x, yt), 30.0 * cos, atol=1e-5), "set_margin(0) must switch the margin off"
head.set_margin(0.15)
assert torch.allclose(head(x, yt) / 30.0 - cos, torch.as_tensor(M, dtype=torch.float32), atol=1e-5)
head.eval()
assert torch.allclose(head(x, yt), head(x, yt)), "eval-mode forward with labels must still run"
assert torch.allclose(head(x), 30.0 * cos, atol=1e-5)

# ---- genus siblings only (family_weight = 0), the setting of the first training run
only = SiblingMarginHead(D, C, counts, gid, fid, m_max=0.15, family_weight=0.0, log_every=0)
only.weight.data.copy_(head.weight.data)
Mg = sm.pair_margin(y, m_sp, gid, fid, 1.0, 0.0)
assert torch.allclose(only(x, yt) / 30.0 - cos, torch.as_tensor(Mg, dtype=torch.float32), atol=1e-5)
assert ((Mg > 0).sum(1) <= n_sib[y]).all(), "genus-only margin may reach siblings and nobody else"
assert ((Mg > 0) == (gid[None, :] == gid[y][:, None]) & (np.arange(C)[None, :] != y[:, None])).all()

# registry and live statistics
h2 = build_head("sibling", D, C, class_counts=counts, genus_ids=gid, family_ids=fid, log_every=1)
h2.train()
buf = io.StringIO()
with redirect_stdout(buf):
    h2(x, yt)
line = buf.getvalue().strip()
assert line.startswith("__SIBLING__ ") and "top_is_sibling=" in line and "m_true_p50=" in line, line
st = h2.last_stats
assert 0 <= st["top_is_sibling"] <= 1 and 0 <= st["top_has_margin"] <= 1 and st["m_true_max"] <= 0.15 + 1e-6
buf = io.StringIO()
with redirect_stdout(buf):
    h2(x, yt)
assert buf.getvalue().strip().startswith("__SIBLING__"), "log_every=1 must log every call"
h3 = SiblingMarginHead(D, C, counts, gid, fid, log_every=5)
h3.train()
buf = io.StringIO()
with redirect_stdout(buf):
    for _ in range(11):
        h3(x, yt)
assert buf.getvalue().count("__SIBLING__") == 3, "log_every=5 logs calls 1, 6 and 11"

# ---- the distribution summary
d = sm.distribution(counts, gid, fid, 0.15)
assert abs(sum(b["image_share"] for b in d["bands"]) - 1) < 1e-9 or counts.max() > 10 ** 9
assert d["species_margin_p100"] == 0.15 and d["image_margin_p0"] <= d["image_margin_p100"]
print("ALL CHECKS PASSED")

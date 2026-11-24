"""Synthetic check of sibling_results.py: realised gaps and error relations against brute force."""
import sys, os, numpy as np, torch, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sibling_results as S
rng = np.random.default_rng(0)
N, D, B = 50, 16, 3000
W = torch.nn.functional.normalize(torch.randn(N, D), dim=1)
gid = rng.integers(0, 10, N); gid[0] = 99  # species 0 has no sibling
y = rng.integers(0, N, B)
feats = (W[y].numpy() + 0.9 * rng.standard_normal((B, D))).astype(np.float32)
d = tempfile.mkdtemp(); np.save(f"{d}/f.npy", feats)
S.head_rows = lambda run_dir: W
S.CHUNK = 700
gs, gf = S.sibling_gaps(f"{d}/f.npy", "x", y, gid)
cos = torch.nn.functional.normalize(torch.as_tensor(feats), dim=1) @ W.T
c = cos.numpy(); ok = 0
for i in range(B):
    sib = [j for j in range(N) if gid[j] == gid[y[i]] and j != y[i]]
    far = [j for j in range(N) if gid[j] != gid[y[i]]]
    es = c[i, y[i]] - c[i, sib].max() if sib else np.nan
    ef = c[i, y[i]] - c[i, far].max()
    assert (np.isnan(es) and np.isnan(gs[i])) or abs(es - gs[i]) < 1e-5, i
    assert abs(ef - gf[i]) < 1e-5
right = (np.nan_to_num(gs, nan=1.0) > 0) & (gf > 0)
assert np.array_equal(right, c.argmax(1) == y)
fid = gid // 3
p = rng.integers(0, N, B)
r = S.relation(y, p, gid, fid)
assert np.all((r == 0) == (gid[p] == gid[y]))
print("__SIBRES__ synthetic check passed", int(np.isnan(gs).sum()), "images without sibling")

"""CPU smoke test for species_sphere.py on synthetic data (numpy + matplotlib only).

Builds a fake 48-species dataset (4 families x 3 genera x 4 species) whose templates
cluster by genus, fakes the cached features / predictions / templates that
species_sphere.py reads on the SCC, runs its main() end to end for a plain head and a
3-sub-center head, and checks the outputs.

Usage: python loss_ablation/species_sphere_smoke.py [OUT_DIR]   (default: a temp dir)
"""

import json
import os
import sys
import tempfile
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import species_sphere as ss  # noqa: E402

N, D, N_VAL = 48, 64, 15


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def fake_dataset(root):
    rng = np.random.default_rng(0)
    family = np.arange(N) // 12
    genus = np.arange(N) // 4
    fam_vec, gen_vec, spc_vec = (rng.standard_normal((n, D)) for n in (4, 12, N))
    templates = unit(0.6 * fam_vec[family] + 0.8 * gen_vec[genus] + 0.45 * spc_vec)

    with open(f"{root}/train.json", "w") as f:
        for c in range(N):
            for _ in range(5 + c % 7):
                f.write(json.dumps({ss.LABEL: c, "genus": f"Genus{genus[c]}",
                                    "family": f"Family{family[c]}ceae",
                                    "scientificName": f"Genus{genus[c]} species{c}"}) + "\n")

    labels = np.repeat(np.arange(N), N_VAL)
    weights = {}
    for name, k in (("dbm", 1), ("subcenter", 3)):
        feats = templates[labels] + 0.12 * rng.standard_normal((len(labels), D))
        preds = (unit(feats) @ templates.T).argmax(1)
        os.makedirs(f"{root}/geom/{name}", exist_ok=True)
        np.save(f"{root}/geom/{name}/features.npy", (3.0 * feats).astype(np.float32))
        np.savez(f"{root}/geom/{name}/predictions.npz", labels=labels, predictions=preds)
        extra = rng.standard_normal((N, k - 1, D)) if k > 1 else np.zeros((N, 0, D))
        weights[name] = np.concatenate([2.0 * templates[:, None, :], extra], axis=1)
    return weights, genus, family


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="species_sphere_smoke_")
    os.makedirs(out, exist_ok=True)
    with tempfile.TemporaryDirectory() as root:
        weights, genus, family = fake_dataset(root)
        sys.modules["phase3_geometry"] = types.SimpleNamespace(
            N=N,
            MODELS={"dbm": ("dbm", "registry"), "subcenter": ("subcenter", "arcface")},
            class_directions=lambda run_dir, kind: weights[run_dir],
        )
        ss.TRAIN_JSON = f"{root}/train.json"
        ss.main(["--models", "dbm", "subcenter", "--min-val", "10",
                 "--geom", f"{root}/geom", "--out", out])

    summary = json.load(open(f"{out}/triples.json"))
    sim = [s["class"] for s in summary["similar"]]
    dis = [s["class"] for s in summary["dissimilar"]]
    assert len(set(genus[sim])) == 1, f"similar triple spans genera: {sim}"
    assert len(set(family[dis])) == 3, f"dissimilar triple repeats a family: {dis}"
    z = np.load(f"{out}/species_sphere.npz")
    for m in ("dbm", "subcenter"):
        for kind in ss.KINDS:
            t = z[f"{m}__{kind}__templates"]
            assert np.allclose(np.linalg.norm(t, axis=1), 1), "templates not unit length"
            assert len(z[f"{m}__{kind}__coords"]) == 3 * N_VAL
    sim_angle = min(summary["stats"]["dbm"]["similar"]["template_angles_deg"])
    dis_angle = min(summary["stats"]["dbm"]["dissimilar"]["template_angles_deg"])
    assert sim_angle < dis_angle, "similar templates should be closer than dissimilar ones"
    assert os.path.getsize(f"{out}/species_sphere.png") > 10_000
    print(f"SMOKE OK  ({out})")


if __name__ == "__main__":
    main()

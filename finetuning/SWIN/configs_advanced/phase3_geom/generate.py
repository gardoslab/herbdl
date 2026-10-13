"""Make phase3_geom/<name>.yml from the phase3_eval eval-only configs: same model, same
full-validation-set evaluation, but output to PHASE3_GEOM/<name> and also save pooled
features (evaluation.features_path). Run from finetuning/SWIN/configs_advanced/."""
SRC = {  # name -> (phase3_eval config to copy, weights dir to swap in or None)
    "arcface": ("arcface", None), "subcenter_arcface": ("subcenter_arcface", None),
    "hace": ("hace", None), "dbm": ("dbm", None), "x2softmax": ("x2softmax", None),
    "kappaface": ("kappaface", None), "arcface_ecc": ("arcface_ecc", None),
    "arcface_tara": ("arcface_tara", None), "dbm_seed43": ("dbm_seed43", None),
    "kappaface_seed43": ("kappaface", ("SWIN_BASE_KAPPAFACE_384_PILOT/", "SWIN_BASE_KAPPAFACE_384_PILOT_SEED43/")),
    "arcface_seed43": ("arcface", ("SWIN_BASE_ARCFACE_384/", "SWIN_BASE_ARCFACE_384_SEED43/")),
}
for name, (src, swap) in SRC.items():
    s = open(f"phase3_eval/{src}.yml").read()
    if swap:
        assert s.count(swap[0] + "model.safetensors") == 1, name
        s = s.replace(swap[0] + "model.safetensors", swap[1] + "model.safetensors")
    n = s.count(f"PHASE3_EVAL/{src}")
    assert n == 4, (name, n)  # header comment, output_dir, logging_dir, predictions_path
    s = s.replace(f"PHASE3_EVAL/{src}", f"PHASE3_GEOM/{name}")
    s = s.replace(f"  predictions_path: ", f"  features_path: /projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/output/SWIN/PHASE3_GEOM/{name}/features.npy\n  predictions_path: ")
    s = "# Phase 3 geometry: copy of phase3_eval/%s.yml%s, writing to PHASE3_GEOM/%s and\n# also saving pooled features. Made by phase3_geom/generate.py.\n" % (
        src, f" with weights from {swap[1].rstrip('/')}" if swap else "", name) + s
    open(f"phase3_geom/{name}.yml", "w").write(s)
    print(name)

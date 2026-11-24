"""Shared paths and helpers for the GBIF-F25 index / overlap jobs."""
import glob
import os
import subprocess

import pandas as pd

DATA = "/projectnb/herbdl/data/"
F25_DIR = DATA + "GBIF-F25/"
F25_DB = DATA + "GBIF-F25h/download_status.db"
F25_OCC = F25_DIR + "occurrence.txt"
F24_OCC = DATA + "GBIF-F24/occurrence.txt"
HARVARD_OCC = DATA + "harvard-herbaria/gbif/occurrence.txt"
MASTER_INDEX = DATA + "master_index.csv"

# Arrow-backed strings use far less memory than Python objects for ~20M-row tables.
try:
    import pyarrow  # noqa: F401
    STR = "string[pyarrow]"
except ImportError:
    STR = "string"

K_GRID = (20, 30, 50, 75, 100, 150, 200, 300, 500, 1000)


def cut_columns(src, wanted, dst):
    """Extract named tab-separated columns from a (huge) GBIF occurrence file with `cut`.

    Column positions are looked up in the file's own header, so the different
    GBIF exports (F24 has `species` at 202, F25 at 204) are handled the same way.
    Returns the list of column names in the order written to `dst`.
    """
    with open(src, encoding="utf-8", errors="replace") as f:
        header = f.readline().rstrip("\n").split("\t")
    pos = {name: i + 1 for i, name in enumerate(header)}
    missing = [c for c in wanted if c not in pos]
    if missing:
        raise KeyError(f"{src}: columns not in header: {missing}")
    # `cut` emits fields in file order regardless of -f order; return that order.
    order = sorted(wanted, key=lambda c: pos[c])
    fields = ",".join(str(pos[c]) for c in order)
    with open(dst, "w") as out:
        subprocess.run(["cut", "-f", fields, src], stdout=out, check=True)
    return order


def read_tsv_chunks(path, names, chunksize=2_000_000):
    """Read a `cut` output as string chunks; skip its header row; no quote handling (GBIF TSV)."""
    return pd.read_csv(path, sep="\t", names=names, header=0, dtype=STR,
                       quoting=3, chunksize=chunksize, on_bad_lines="skip")


def epithet(genus, species):
    """Species epithet from a GBIF `species` binomial ("Carex lurida" -> "lurida").

    Kaggle and Harvard store only the epithet in `species`; GBIF stores the
    binomial. Returns <NA> when the binomial does not start with the genus.
    """
    g = genus.str.strip().str.lower()
    s = species.str.strip()
    starts = pd.Series([isinstance(a, str) and isinstance(b, str) and a.lower().startswith(b + " ")
                        for a, b in zip(s, g)], index=s.index)
    return s.str.split(" ", n=1).str[1].where(starts)


def master_species_counts():
    """Images per species key ("genus epithet", lower-case) in master_index.csv, by source."""
    mi = pd.read_csv(MASTER_INDEX, usecols=["filepath", "genus", "species"], dtype="string").dropna()
    mi["src"] = mi.filepath.str.split("/").str[4]
    g, s = mi.genus.str.strip(), mi.species.str.strip()
    # GBIF-F24 rows hold the binomial in `species`; Kaggle/Harvard hold the epithet.
    ep = epithet(g, s)
    s = ep.where(ep.notna(), s)
    mi["key"] = (g + " " + s).str.lower()
    return mi.groupby(["src", "key"]).size()


def uniform_table(name, counts):
    """Print how many species have >= k images (a perfectly uniform k-per-species set)."""
    c = counts[counts > 0]
    q = c.quantile([.5, .75, .9, .99]).round().to_dict()
    print(f"\n== {name}: {len(c)} species, {int(c.sum())} images; quantiles {q}")
    print(f" {'k':>5} | {'#species>=k':>11} | {'n*k':>10}")
    for k in K_GRID:
        n = int((c >= k).sum())
        print(f" {k:>5} | {n:>11} | {n * k:>10}")


def latest_index():
    paths = sorted(glob.glob(F25_DIR + "index_snapshot_*.csv"))
    if not paths:
        raise FileNotFoundError("no GBIF-F25 index snapshot; run build_index.py first")
    return paths[-1]


def scratch():
    return os.environ.get("TMPDIR", "/tmp")

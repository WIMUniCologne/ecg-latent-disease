# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Build the ICD-to-Phecode lookup table behind the hospital diagnoses.

The ICD-coded hospital diagnoses are defined through the public Phecode map v1.2 (Wu et al.,
JMIR Medical Informatics 2019; https://phewascatalog.org); data/labels.py reads the table
written here. This script reads the ``Phecode_map`` data set of the R package
vcastro/Phecode (GitHub), which holds the rolled-up map (a parent phecode also carries the
codes of its children) for ICD-9-CM, ICD-10-CM and WHO ICD-10, and keeps the two US
vocabularies MIMIC-IV uses. Codes are written without the dot and in upper case, as MIMIC-IV
stores them; duplicate rows are dropped and the order of the source is kept.

Input:  $ECG_CACHE_DIR/phecodes/vcastro_Phecode_map.rda (needs the package pyreadr)
Output: $ECG_CACHE_DIR/phecodes/phecode_lookup.csv  (icd_version, code_dotless, phecode)

    python data/build_phecode_lookup.py

Both files are pinned by SHA-256. Another input stops the run unless
ECG_ALLOW_OTHER_PHECODES is set. With the input of the paper (SOURCE_SHA256) the output is
byte-identical to the lookup used for the paper (OUTPUT_SHA256; 158,022 rows: 23,021
ICD-9-CM and 135,001 ICD-10-CM), and the run stops if it is not.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ecg_paths import cache_path  # noqa: E402

SOURCE = cache_path("phecodes", "vcastro_Phecode_map.rda")
OUT = cache_path("phecodes", "phecode_lookup.csv")
SOURCE_SHA256 = "a2684dda7cbd48ce92c48af05ec91020a2b99764196101ea41555f13fa2c6027"
OUTPUT_SHA256 = "d3d898f6e1f4425145b34278f6cfb64575864ebb489d98c878488e7fcbcac10c"
VERSION = {"ICD9CM": 9, "ICD10CM": 10}


def _sha256(path: str) -> str:
    """Hex SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(source: str = SOURCE) -> pd.DataFrame:
    """Return the lookup table from the vcastro/Phecode map: ICD-9-CM and ICD-10-CM rows
    only, codes without dots and in upper case, duplicates dropped, source order kept."""
    import pyreadr
    m = pyreadr.read_r(source)["Phecode_map"]
    m = m[m["vocabulary_id"].isin(list(VERSION))]
    out = pd.DataFrame({
        "icd_version": m["vocabulary_id"].map(VERSION),
        "code_dotless": m["code"].astype(str).str.replace(".", "", regex=False).str.strip().str.upper(),
        "phecode": m["phecode"].astype(str),
    })
    return out.drop_duplicates()


def main():
    """Write the lookup and check that input and output are the ones used for the paper. A
    different input stops the run unless ECG_ALLOW_OTHER_PHECODES is set; the paper's input
    with a different output always stops it (the builder has changed)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=SOURCE)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    same_source = _sha256(a.source) == SOURCE_SHA256
    if not same_source:
        if not os.environ.get("ECG_ALLOW_OTHER_PHECODES"):
            raise SystemExit(f"{a.source} is not the Phecode map of the paper (sha256 {SOURCE_SHA256}); "
                             "set ECG_ALLOW_OTHER_PHECODES=1 to use it anyway")
        print(f"note: {a.source} is not the map version used for the paper", flush=True)
    lookup = build(a.source)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    lookup.to_csv(a.out, index=False)
    same_out = _sha256(a.out) == OUTPUT_SHA256
    print(f"wrote {len(lookup):,} rows to {a.out}; identical to the paper's lookup: {same_out}")
    if same_source and not same_out:
        raise SystemExit("same input but a different output: the builder has changed")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Write the D3 regression fixtures of M3c (pre-registration §21 (ii)).

Phase 1's D3 run (``R3_measured_realdepth``: measured 5K factors plus
residual, real per-class depth, on the 5K mouse bundle ``c30f7eab``) is the
regression target of M3c's R3 recipe and profile mode. This script derives
the small fixtures of ``tests/test_annotation/data/``:

- ``m3c_d3_efficiency_mouse5k.csv``: ``gene_id`` and ``log2_efficiency`` of
  all 5,006 panel genes (``sim/runs/mouse5k/D3/efficiency.csv``);
- ``m3c_d3_predictions_mouse5k.csv``: the D3 predictions of the 14 major
  classes (real share >= 1%) and ALL (``sim/tables/pred_mouse5k_3.csv``):
  ``n``, ``kish_n``, ``cov_class_rule73``, ``cov_subclass_rule73`` and the
  provisional-decision columns (information only);
- ``m3c_real_composition_mouse5k.csv``: the public section's called (class,
  subclass) cell counts (``mouse5k/qc/percell_calls_mouse5k.parquet``), the
  weights of ``analyse.py`` ``comp_weights``;
- ``m3c_d3_fixtures.json``: each source's path under the evidence root and
  sha256, the D3 manifest and the derivation.

The JSON holds no local path: the evidence root is written as the variable
``$MERXEN_EVIDENCE_ROOT``, and the bundle and test-set directories the D3
manifest records are written under their store variable
(``$MERXEN_REFERENCE_STORE_LARGE/wmb_panel/<hash>``); a manifest path under
none of the location variables is refused.

Usage::

    python scripts/acceptance/m3c_d3_fixtures.py \\
        --evidence-root "$MERXEN_EVIDENCE_ROOT"

``--evidence-root`` defaults to that ``.env`` variable (template
``.env.example``); the store variables must be set as well.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from merxen import local_config

EFFICIENCY = "5k_real/sim/runs/mouse5k/D3/efficiency.csv"
MANIFEST = "5k_real/sim/runs/mouse5k/D3/manifest.json"
PREDICTIONS = "5k_real/sim/tables/pred_mouse5k_3.csv"
CALLS = "5k_real/mouse5k/qc/percell_calls_mouse5k.parquet"
MAJOR_SHARE = 0.01
PREDICTION_COLUMNS = (
    "class",
    "n",
    "kish_n",
    "cov_class_rule73",
    "cov_subclass_rule73",
    "cov_class_prov",
    "cov_subclass_prov",
)
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "tests/test_annotation/data"
# What the record's evidence_root says: ``$A`` stands for the .env variable.
EVIDENCE_ROOT_TEXT = "$A = $MERXEN_EVIDENCE_ROOT"
LOCATION_VARIABLES = ("MERXEN_REFERENCE_STORE_LARGE", "MERXEN_REFERENCE_STORE")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def locate(value: object, key: str) -> object:
    """Return a manifest path as ``$VARIABLE/relative`` (other values as is).

    The D3 manifest's ``bundle`` and ``test_set`` entries are directories under
    the large reference store.

    Args:
        value: The manifest entry.
        key: Its key (for the message).

    Returns:
        The entry with its store prefix replaced by the variable's name.

    Raises:
        RuntimeError: If ``value`` is a path under none of the store variables
            (the message names the key, never the path).
    """
    if not isinstance(value, str) or not value.startswith("/"):
        return value
    for name in LOCATION_VARIABLES:
        root = local_config.path(name)
        if root is None:
            continue
        try:
            relative = Path(value).relative_to(root)
        except ValueError:
            continue
        return f"${name}/{relative.as_posix()}"
    raise RuntimeError(
        f"d3_manifest[{key!r}] is a path under none of {', '.join(LOCATION_VARIABLES)} "
        "(set them in .env)"
    )


def build(evidence_root: Path) -> dict[str, str]:
    """Return the fixture files (name -> text)."""
    efficiency = pd.read_csv(evidence_root / EFFICIENCY)
    predictions = pd.read_csv(evidence_root / PREDICTIONS)
    calls = pd.read_parquet(evidence_root / CALLS, columns=["class", "subclass"])
    share = calls["class"].astype(str).value_counts(normalize=True)
    major = sorted(share[share >= MAJOR_SHARE].index)
    rows = predictions[predictions["class"].isin([*major, "ALL"])]
    if len(rows) != len(major) + 1:
        raise RuntimeError(f"{PREDICTIONS}: major classes or ALL missing")
    composition = (
        calls.astype(str)
        .groupby(["class", "subclass"], sort=True)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    manifest = json.loads((evidence_root / MANIFEST).read_text(encoding="utf-8"))
    files = {
        "m3c_d3_efficiency_mouse5k.csv": efficiency[
            ["gene_id", "log2_efficiency"]
        ].to_csv(index=False, float_format="%.17g", lineterminator="\n"),
        "m3c_d3_predictions_mouse5k.csv": rows[list(PREDICTION_COLUMNS)].to_csv(
            index=False, float_format="%.17g", lineterminator="\n"
        ),
        "m3c_real_composition_mouse5k.csv": composition.to_csv(
            index=False, lineterminator="\n"
        ),
    }
    record: dict[str, Any] = {
        "purpose": "M3c pre-registration §21 (ii): the D3 regression fixtures",
        "evidence_root": EVIDENCE_ROOT_TEXT,
        "sources": {
            name: {"path": f"$A/{name}", "sha256": _sha256(evidence_root / name)}
            for name in (EFFICIENCY, MANIFEST, PREDICTIONS, CALLS)
        },
        "d3_manifest": {
            key: locate(manifest.get(key), key)
            for key in (
                "variant",
                "bundle",
                "test_set",
                "n_test_cells",
                "n_genes",
                "grid",
                "mapping",
                "efficiency",
                "efficiency_sd_ln",
                "efficiency_frac_below_0.25",
                "efficiency_frac_below_0.125",
                "recipe",
                "n_simulated",
            )
        },
        "major_classes": major,
        "derivation": {
            "m3c_d3_efficiency_mouse5k.csv": "D3 efficiency.csv gene_id and "
            "log2_efficiency (median-normalised log2 efficiency, panel order)",
            "m3c_d3_predictions_mouse5k.csv": "pred_mouse5k_3.csv rows of the "
            f"classes with real called share >= {MAJOR_SHARE} and ALL; analyse.py "
            "per_class over the called class, simulated cells weighted to the "
            "real called composition (class weight x within-class subclass "
            "weight, trimmed at 10x the class median)",
            "m3c_real_composition_mouse5k.csv": "cells per called (class, "
            "subclass) of percell_calls_mouse5k.parquet",
        },
    }
    for name, text in files.items():
        record.setdefault("fixtures", {})[name] = hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()
    files["m3c_d3_fixtures.json"] = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if not np.isfinite(efficiency["log2_efficiency"]).all():
        raise RuntimeError(f"{EFFICIENCY}: non-finite efficiency")
    return files


def main(argv: Sequence[str] | None = None) -> int:
    """Write the fixtures; ``--check`` exits 1 when a committed one is stale."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    local_config.add_location_argument(
        parser,
        "--evidence-root",
        "MERXEN_EVIDENCE_ROOT",
        "The evidence root (`$A`) holding the D3 sources.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    files = build(args.evidence_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stale = []
    for name, text in files.items():
        path = args.output_dir / name
        if path.is_file() and path.read_text(encoding="utf-8") == text:
            continue
        stale.append(name)
        if not args.check:
            path.write_text(text, encoding="utf-8")
    print(("stale: " if args.check else "wrote: ") + ", ".join(stale or ["nothing"]))
    return 1 if args.check and stale else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Mouse ``avg_correlation`` floor shadow study (plan §7.3; pre-registration §16).

Reads the M6 mouse RESOLVE label tables of ag7 proseg_hybrid and VZG2
original_seg (no correlation floor applied), applies the candidate class
floors (none, 0.40, 0.50) to the confident class calls and scores, overall
and per depth bin: class coverage of table cells, marker consistency of
the confident class calls, and the marker consistency of the calls a floor
removes. Marker pseudo-labels are MO1's own, ported verbatim:

- ag7: ``data/mouse_ag7/out/tables/C_pseudo_labels_per_cell.csv``
  (``pseudo_fine``) with ``expected_pseudo_for_mmc`` of
  ``data/mouse_ag7/scripts/ag7_sections_extra.py``;
- VZG2: ``data/mouse_vzg2/tables/C_pseudo_labels_per_cell_original_seg.csv.gz``
  (``pseudo_class``) with the six-group map of
  ``review_scientific/mouse_marker_consistency.py``.

Before any floor is scored, MO1 over all marker-pseudo-confident cells with
the unpruned MAP labels must reproduce .9114 (ag7) and .8556 (VZG2) to three
decimals; otherwise the study stops without a selection. The selection rule
is the pre-registered one (``select_floor``).

Usage::

    mouse_corr_shadow.py --ag7-labels L1 --vzg2-labels L2 --out DIR
        [--evidence-root A]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EVIDENCE_ROOT = Path("/srv/storage/MerXen/annotation_dev/evidence_20260926")
FLOORS: tuple[float | None, ...] = (None, 0.40, 0.50)
DEPTH_BINS: tuple[tuple[int, float], ...] = (
    (20, 50),
    (50, 100),
    (100, 250),
    (250, 500),
    (500, 1000),
    (1000, 2000),
    (2000, math.inf),
)
MO1_BASELINE = {"ag7": 0.9114, "vzg2": 0.8556}
MO1_TOLERANCE = 0.0005  # three decimals
# Pre-registered selection constants (§16).
MIN_GAIN = 0.005
MIN_REMOVED = 50
MIN_REMOVED_GAP = 0.10
MAX_COVERAGE_LOSS = 0.02
MAX_BIN_COVERAGE_LOSS = 0.05
MIN_BIN_CELLS = 1000
MIN_EXTRA_GAIN = 0.005
MAX_EXTRA_LOSS = 0.01
FLOAT32_TOLERANCE = 1e-6

# data/mouse_ag7/scripts/ag7_sections_extra.py (verbatim).
NN_CLASS_TO_BROAD = {
    30: "Astrocytes/Ependymal",
    31: "Oligodendrocyte lineage",
    32: "OEC",
    33: "Vascular cells",
    34: "Microglia",
}
NN_SUBCLASS_EXPECT: list[tuple[str, set[str]]] = [
    ("OPC NN", {"OPC"}),
    ("COP NN", {"OPC", "Oligo"}),
    ("NFOL NN", {"Oligo", "OPC"}),
    ("MFOL NN", {"Oligo"}),
    ("MOL NN", {"Oligo"}),
    ("Oligo NN", {"Oligo"}),
    ("Astro", {"Astro"}),
    ("Bergmann", {"Astro"}),
    ("Tanycyte", {"Astro", "Epen"}),
    ("Ependymal", {"Epen"}),
    ("Hypendymal", {"Epen"}),
    ("CHOR", {"Epen", "Astro"}),
    ("Endo NN", {"Endo"}),
    ("Peri NN", {"Peri/SMC"}),
    ("SMC NN", {"Peri/SMC"}),
    ("VLMC NN", {"VLMC"}),
    ("ABC NN", {"VLMC"}),
    ("Microglia", {"Micro"}),
    ("BAM", {"Micro"}),
    ("Monocytes", {"Micro"}),
    ("DC NN", {"Micro"}),
    ("Lymphoid", {"Micro"}),
    ("OEC", set()),
]
# review_scientific/mouse_marker_consistency.py (verbatim).
VZG2_PSEUDO_GROUP = {
    "Glut": "Neuron",
    "GABA": "Neuron",
    "Astro": "Astro",
    "EpendCP": "Astro",
    "Oligo": "OL",
    "OPC": "OL",
    "Immune": "Immune",
    "Endo": "Vascular",
    "Mural": "Vascular",
    "VLMC": "Vascular",
}


def _class_num(name: str) -> int:
    try:
        return int(str(name).split(" ")[0])
    except ValueError:
        return -1


def expected_pseudo_for_mmc(class_name: str, subclass_name: str) -> set[str] | None:
    """Marker pseudo-classes consistent with a MapMyCells label (ag7 report)."""
    n = _class_num(class_name)
    if n in NN_CLASS_TO_BROAD:
        for tok, exp in NN_SUBCLASS_EXPECT:
            if tok in str(subclass_name):
                return exp
        return {"Astro", "Epen", "Oligo", "OPC", "Micro", "Endo", "Peri/SMC", "VLMC"}
    s = str(subclass_name)
    has_g = "Glut" in s
    has_i = ("GABA" in s) or ("Gly" in s)
    if has_g and not has_i:
        return {"Glut"}
    if has_i and not has_g:
        return {"GABA"}
    return {"Glut", "GABA"}


def cls2(c: str) -> str:
    """The VZG2 six-group class of a WMB class name (MO1 baseline)."""
    code = int(c.split()[0])
    if code <= 29:
        return "Neuron"
    return {30: "Astro", 31: "OL", 32: "OEC", 33: "Vascular", 34: "Immune"}.get(
        code, "Other"
    )


@dataclass(frozen=True)
class Scored:
    """Per table cell: pseudo-confident, consistent (pruned and unpruned calls)."""

    pseudo_confident: np.ndarray
    consistent: np.ndarray
    consistent_unpruned: np.ndarray


def score_ag7(labels: pd.DataFrame, pseudo: pd.DataFrame) -> Scored:
    """Score ag7 with the ag7 report's pseudo-labels and expected classes."""
    fine = pseudo["pseudo_fine"].reindex(labels.index).astype(object)
    confident = fine.notna().to_numpy() & (fine.astype(str) != "unassigned").to_numpy()

    def consistent(class_column: str, subclass_column: str) -> np.ndarray:
        values = []
        for q, c, s in zip(
            fine.to_numpy(),
            labels[class_column].astype(object).to_numpy(),
            labels[subclass_column].astype(object).to_numpy(),
            strict=True,
        ):
            expected = expected_pseudo_for_mmc(c, s)
            values.append(expected is not None and q in expected)
        return np.asarray(values, dtype=bool)

    return Scored(
        pseudo_confident=confident,
        consistent=consistent("mmc_wmb_class_name", "mmc_wmb_subclass_name"),
        consistent_unpruned=consistent(
            "mmc_wmb_unpruned_class_name", "mmc_wmb_unpruned_subclass_name"
        ),
    )


def score_vzg2(labels: pd.DataFrame, pseudo: pd.DataFrame) -> Scored:
    """Score VZG2 with the six-group map of the MO1 baseline."""
    classes = pseudo["pseudo_class"].reindex(labels.index).astype(object)
    confident = classes.isin(list(VZG2_PSEUDO_GROUP)).to_numpy()
    group = classes.map(VZG2_PSEUDO_GROUP).to_numpy(dtype=object)

    def consistent(column: str) -> np.ndarray:
        called = labels[column].astype(object).to_numpy()
        return np.array(
            [
                isinstance(c, str) and g is not None and cls2(c) == g
                for c, g in zip(called, group, strict=True)
            ],
            dtype=bool,
        )

    return Scored(
        pseudo_confident=confident,
        consistent=consistent("mmc_wmb_class_name"),
        consistent_unpruned=consistent("mmc_wmb_unpruned_class_name"),
    )


def _share(mask: np.ndarray, members: np.ndarray) -> float | None:
    n = int(members.sum())
    return float((mask & members).sum() / n) if n else None


def floor_metrics(
    labels: pd.DataFrame, scored: Scored, floor: float | None, members: np.ndarray
) -> dict[str, Any]:
    """Coverage and consistency of one floor over a set of table cells."""
    base = labels["ct_class_status"].astype(str).to_numpy() == "confident"
    corr = labels["ct_class_corr"].to_numpy(np.float64)
    if floor is None:
        kept = base
    else:
        kept = base & (np.nan_to_num(corr, nan=-np.inf) >= floor - FLOAT32_TOLERANCE)
    removed = base & ~kept
    pseudo = scored.pseudo_confident & members
    return {
        "n_table": int(members.sum()),
        "coverage": _share(kept, members),
        "n_confident": int((kept & members).sum()),
        "n_pseudo_confident_kept": int((kept & pseudo).sum()),
        "cons": _share(scored.consistent, kept & pseudo),
        "n_removed": int((removed & members).sum()),
        "n_removed_pseudo": int((removed & pseudo).sum()),
        "rem_cons": _share(scored.consistent, removed & pseudo),
    }


def dataset_tables(
    labels: pd.DataFrame, scored: Scored
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Per floor x bin metrics and the MO1 check of one dataset."""
    table = labels["in_table"].to_numpy(bool)
    counts = labels["total_counts"].to_numpy(np.float64)
    rows = []
    bins = [("all", table)] + [
        (
            f"[{low},{'inf' if math.isinf(high) else int(high)})",
            table & (counts >= low) & (counts < high),
        )
        for low, high in DEPTH_BINS
    ]
    for floor in FLOORS:
        for name, members in bins:
            rows.append(
                {
                    "floor": "none" if floor is None else f"{floor:.2f}",
                    "depth_bin": name,
                    **floor_metrics(labels, scored, floor, members),
                }
            )
    pseudo = scored.pseudo_confident & table
    mo1 = {
        "n_pseudo_confident": int(pseudo.sum()),
        "mo1_unpruned": _share(scored.consistent_unpruned, pseudo),
        "mo1_pruned": _share(scored.consistent, pseudo),
    }
    return pd.DataFrame(rows), mo1


def _row(frame: pd.DataFrame, floor: str, depth_bin: str = "all") -> pd.Series:
    rows = frame[(frame["floor"] == floor) & (frame["depth_bin"] == depth_bin)]
    return rows.iloc[0]


def admissible(frame: pd.DataFrame, floor: str) -> tuple[bool, list[str]]:
    """The pre-registered admissibility of one floor on one dataset (§16)."""
    base = _row(frame, "none")
    row = _row(frame, floor)
    reasons: list[str] = []
    gain = row["cons"] - base["cons"]
    if not gain >= MIN_GAIN:
        reasons.append(f"(i) consistency gain {gain:+.4f} < {MIN_GAIN}")
    if row["n_removed_pseudo"] < MIN_REMOVED:
        reasons.append(
            f"(ii) {row['n_removed_pseudo']} removed pseudo-confident < {MIN_REMOVED}"
        )
    elif not row["rem_cons"] <= row["cons"] - MIN_REMOVED_GAP:
        reasons.append(
            f"(ii) removed consistency {row['rem_cons']:.4f} > kept "
            f"{row['cons']:.4f} - {MIN_REMOVED_GAP}"
        )
    loss = base["coverage"] - row["coverage"]
    if not loss <= MAX_COVERAGE_LOSS:
        reasons.append(f"(iii) coverage loss {loss:.4f} > {MAX_COVERAGE_LOSS}")
    for depth_bin in frame["depth_bin"].unique():
        if depth_bin == "all":
            continue
        base_bin = _row(frame, "none", depth_bin)
        if base_bin["n_table"] < MIN_BIN_CELLS:
            continue
        bin_loss = base_bin["coverage"] - _row(frame, floor, depth_bin)["coverage"]
        if not bin_loss <= MAX_BIN_COVERAGE_LOSS:
            reasons.append(
                f"(iii) bin {depth_bin} coverage loss {bin_loss:.4f} > "
                f"{MAX_BIN_COVERAGE_LOSS}"
            )
    return not reasons, reasons


def select_floor(frames: Mapping[str, pd.DataFrame]) -> tuple[str, dict[str, Any]]:
    """Apply the pre-registered selection rule (§16) over both datasets."""
    verdicts: dict[str, Any] = {}
    ok = {}
    for floor in ("0.40", "0.50"):
        per = {name: admissible(frame, floor) for name, frame in frames.items()}
        ok[floor] = all(passed for passed, _ in per.values())
        verdicts[floor] = {
            name: {"admissible": passed, "reasons": reasons}
            for name, (passed, reasons) in per.items()
        }
    if not ok["0.40"] and not ok["0.50"]:
        return "none", verdicts
    if ok["0.40"] != ok["0.50"]:
        return ("0.40" if ok["0.40"] else "0.50"), verdicts
    extra = {
        name: (
            _row(frame, "0.50")["cons"] - _row(frame, "0.40")["cons"],
            _row(frame, "0.40")["coverage"] - _row(frame, "0.50")["coverage"],
        )
        for name, frame in frames.items()
    }
    verdicts["both_admissible"] = {
        name: {"extra_gain": gain, "extra_loss": loss}
        for name, (gain, loss) in extra.items()
    }
    if all(
        gain >= MIN_EXTRA_GAIN and loss <= MAX_EXTRA_LOSS
        for gain, loss in extra.values()
    ):
        return "0.50", verdicts
    return "0.40", verdicts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _read_labels(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["cell_id"] = frame["cell_id"].astype(str)
    return frame.set_index("cell_id")


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def report(
    frames: Mapping[str, pd.DataFrame],
    mo1: Mapping[str, Mapping[str, Any]],
    selection: str | None,
    verdicts: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> str:
    """Return the plain-text report."""
    lines = [
        "M6 mouse avg_correlation floor shadow study (plan §7.3; pre-registration §16)",
        f"script sha256 {inputs['script_sha256']}; git {inputs['git_commit']}",
        "",
        "Inputs:",
    ]
    lines += [f"  {key}: {value}" for key, value in inputs["files"].items()]
    lines += ["", "MO1 check (unpruned MAP labels, all marker-pseudo-confident cells):"]
    for name, record in mo1.items():
        check = "PASS" if record["check"] else "FAIL"
        lines.append(
            f"  {name}: {_fmt(record['mo1_unpruned'])} "
            f"(baseline {MO1_BASELINE[name]}; n {record['n_pseudo_confident']}); "
            f"pruned labels {_fmt(record['mo1_pruned'])}; check {check}"
        )
    header = (
        "  floor  depth_bin      n_table  coverage  n_conf  cons    "
        "n_rem_pseudo  rem_cons"
    )
    for name, frame in frames.items():
        lines += ["", f"{name}: per floor and depth bin", header]
        for _, row in frame.iterrows():
            lines.append(
                f"  {row['floor']:<6} {row['depth_bin']:<14} {row['n_table']:>7}  "
                f"{_fmt(row['coverage']):>8}  {row['n_confident']:>6}  "
                f"{_fmt(row['cons']):>6}  {row['n_removed_pseudo']:>12}  "
                f"{_fmt(row['rem_cons']):>8}"
            )
    lines += ["", "Admissibility (pre-registered):"]
    for floor in ("0.40", "0.50"):
        for name, record in verdicts.get(floor, {}).items():
            verdict = "admissible" if record["admissible"] else "not admissible"
            why = "" if record["admissible"] else f" ({'; '.join(record['reasons'])})"
            lines.append(f"  {floor} {name}: {verdict}{why}")
    if "both_admissible" in verdicts:
        lines.append(f"  both admissible: {json.dumps(verdicts['both_admissible'])}")
    chosen = (
        selection if selection is not None else "none (MO1 check failed: no selection)"
    )
    lines += ["", f"SELECTION: {chosen}"]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Run the study (see the module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ag7-labels", type=Path, required=True)
    parser.add_argument("--vzg2-labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, default=EVIDENCE_ROOT)
    args = parser.parse_args(argv)
    root = args.evidence_root
    pseudo_paths = {
        "ag7": root / "data/mouse_ag7/out/tables/C_pseudo_labels_per_cell.csv",
        "vzg2": root
        / "data/mouse_vzg2/tables/C_pseudo_labels_per_cell_original_seg.csv.gz",
    }
    label_paths = {"ag7": args.ag7_labels, "vzg2": args.vzg2_labels}
    args.out.mkdir(parents=True, exist_ok=False)
    frames: dict[str, pd.DataFrame] = {}
    mo1: dict[str, dict[str, Any]] = {}
    for name in ("ag7", "vzg2"):
        labels = _read_labels(label_paths[name])
        pseudo = pd.read_csv(pseudo_paths[name], index_col=0)
        pseudo.index = pseudo.index.astype(str)
        scored = (score_ag7 if name == "ag7" else score_vzg2)(labels, pseudo)
        frame, check = dataset_tables(labels, scored)
        check["check"] = (
            check["mo1_unpruned"] is not None
            and abs(check["mo1_unpruned"] - MO1_BASELINE[name]) <= MO1_TOLERANCE
        )
        frames[name] = frame
        mo1[name] = check
        frame.to_csv(args.out / f"corr_floor_{name}.csv", index=False)
    inputs = {
        "script_sha256": _sha256(Path(__file__)),
        "git_commit": _git_commit(),
        "files": {
            **{
                f"{key}_labels": f"{path} sha256 {_sha256(path)}"
                for key, path in label_paths.items()
            },
            **{
                f"{key}_pseudo": f"{path} sha256 {_sha256(path)}"
                for key, path in pseudo_paths.items()
            },
        },
    }
    if all(record["check"] for record in mo1.values()):
        selection, verdicts = select_floor(frames)
    else:
        selection, verdicts = None, {}
    text = report(frames, mo1, selection, verdicts, inputs)
    (args.out / "CORR_SHADOW_REPORT.txt").write_text(text, encoding="utf-8")
    (args.out / "corr_shadow.json").write_text(
        json.dumps(
            {
                "selection": selection,
                "mo1": mo1,
                "verdicts": verdicts,
                "inputs": inputs,
            },
            indent=1,
            default=float,
        ),
        encoding="utf-8",
    )
    sys.stdout.write(text)
    return 0 if selection is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())

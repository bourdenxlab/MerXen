#!/usr/bin/env python
"""H10: the E1 marker referee on new-vs-legacy broad disputes (plan §14; M8).

H10 (pre-registration §2, §9): for each disputed table cell, where both
broad labels are among the 7 classes and they differ, the label whose
canonical panel markers (E1's list, ``shadow.E1_REFEREE_MARKERS``) hold the
larger fraction of the cell's counts wins; the markers must side with the
new label in >= 70% of the disputes. **New** is the RESOLVE confident broad
label (``ct_broad_status == confident``: ``ct_broad_name``; cells that are
not confident have no new label), **legacy** is the published
``obs["broad_class"]`` of the legacy clustered H5AD.

This is a standalone CLI over ``merxen.annotation.shadow.marker_referee``
that scores exactly as ``resolve_criteria.py`` does (the same label rule,
the same ``marker_class_scores`` on the legacy H5AD's ``layers["counts"]`` of
the labelled table cells, the same H10 rows through
``resolve_criteria.referee_table_rows``); ``--check-against`` compares its
rows with a ``resolve_criteria.py`` run's ``referee.csv`` and fails on any
difference. It adds the per-dispute breakdown (which class pairs the
disputes are, and who wins each), the input to the legacy comparison
(``compare_legacy.py``) and the M13 real-data QC referee.

Inputs (read-only):

- ``--resolve-root``: ``<root>/<pair>/<seg>/`` RESOLVE outputs
  (``<pair>_resolve_summary.json``, ``<plat>/<sid>_celltype_labels.parquet``);
- ``--results-root``: the published legacy clustered H5ADs
  (``<pair>/<seg>/clustering_squidpy/clustering_squidpy_out/<plat>/``).

Writes to ``--out``: ``referee.csv`` (one row per sample: disputes and wins),
``referee_disputes.csv`` (per sample and dispute class pair),
``h10_table.csv`` (the H10 criteria rows) and ``marker_referee_run.json``.

Usage::

    python scripts/acceptance/marker_referee.py --resolve-root <runs> \\
        --results-root <results> --out <dir> [--check-against <criteria>/referee.csv]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from merxen.annotation.pipeline import read_label_table
from merxen.annotation.shadow import (
    E1_REFEREE_MARKERS,
    marker_class_scores,
    marker_referee,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from resolve_criteria import confident, referee_table_rows  # noqa: E402
from shadow_baselines import PAIRS, PLATFORMS, _clustered_path  # noqa: E402

logger = logging.getLogger("marker_referee")

NEW = "new confident (RESOLVE)"
LEGACY = "legacy broad_class"
ALL_SEGMENTATIONS = ("proseg_hybrid", "reseg", "proseg_mask", "original_seg")
# The columns --check-against compares (resolve_criteria.py's referee.csv).
CHECKED = ("n_cells", "n_disputes", "first_wins", "second_wins", "undecided")


@dataclass
class RefereeSample:
    """One sample's table cells, their counts and both broad labellings.

    Attributes:
        pair: Pair id.
        segmentation: Segmentation.
        sample_id: Sample id.
        platform: ``MERSCOPE`` or ``XENIUM``.
        labels: RESOLVE label rows of the table cells (index ``cell_id``).
        counts: Raw counts of those cells (rows in ``labels`` order).
        symbols: Gene symbol per column of ``counts``.
        legacy_broad: Published legacy ``broad_class`` of those cells.
    """

    pair: str
    segmentation: str
    sample_id: str
    platform: str
    labels: pd.DataFrame
    counts: sparse.csr_matrix
    symbols: list[str]
    legacy_broad: np.ndarray


def load_referee_sample(
    resolve_dir: Path,
    results: Path,
    summary: Mapping[str, Any],
    pair: str,
    segmentation: str,
    platform: str,
) -> RefereeSample:
    """Load one sample's labels and its legacy clustered H5AD (as resolve_criteria).

    Args:
        resolve_dir: The pair x segmentation RESOLVE output directory.
        results: The published results root (legacy clustered H5ADs).
        summary: The pair's ``<pair>_resolve_summary.json``.
        pair: Pair id.
        segmentation: Segmentation.
        platform: ``MERSCOPE`` or ``XENIUM``.

    Returns:
        The sample.

    Raises:
        ValueError: If a labelled table cell is missing from the H5AD.
    """
    import anndata as ad

    sample_id = f"{pair}_{platform}"
    entry = summary["samples"][sample_id]
    labels, _ = read_label_table(resolve_dir / entry["labels"])
    labels = labels[labels["in_table"].to_numpy(bool)].set_index("cell_id")
    labels.index = labels.index.astype(str)
    adata = ad.read_h5ad(_clustered_path(results, pair, segmentation, platform))
    obs = pd.Index(adata.obs_names.astype(str))
    position = obs.get_indexer(labels.index)
    if (position < 0).any():
        raise ValueError(f"{sample_id}: labelled cells missing from the clustered H5AD")
    return RefereeSample(
        pair=pair,
        segmentation=segmentation,
        sample_id=sample_id,
        platform=platform,
        labels=labels,
        counts=sparse.csr_matrix(adata.layers["counts"])[position],
        symbols=[str(name) for name in adata.var_names],
        legacy_broad=np.asarray(adata.obs["broad_class"].astype(object))[position],
    )


def new_labels(labels: pd.DataFrame) -> np.ndarray:
    """Return the new labelling: the confident broad name, else ``None``."""
    return np.where(
        confident(labels, "broad"),
        labels["ct_broad_name"].astype(object).to_numpy(),
        None,
    )


def referee_row(sample: RefereeSample, scores: np.ndarray) -> dict[str, Any]:
    """Return the H10 referee row of one sample (``resolve_criteria`` columns).

    Args:
        sample: The sample.
        scores: ``marker_class_scores`` of its counts.

    Returns:
        The row (``first`` = new, ``second`` = legacy).
    """
    result = marker_referee(scores, new_labels(sample.labels), sample.legacy_broad)
    return {
        "pair": sample.pair,
        "segmentation": sample.segmentation,
        "sample_id": sample.sample_id,
        "first": NEW,
        "second": LEGACY,
        "n_cells": result.n_cells,
        "n_disputes": result.n_disputes,
        "dispute_share": result.n_disputes / max(result.n_cells, 1),
        "first_wins": result.first_wins,
        "second_wins": result.second_wins,
        "undecided": result.undecided,
        "consensus_top_marker_ok": result.consensus_top_marker_ok,
        "top_disputes": json.dumps(result.top_disputes),
    }


def dispute_rows(
    sample: RefereeSample,
    scores: np.ndarray,
    classes: Sequence[str] = tuple(E1_REFEREE_MARKERS),
) -> list[dict[str, Any]]:
    """Return the disputes of one sample per (new, legacy) class pair.

    A dispute is a cell whose two labels are both among ``classes`` and
    differ (``marker_referee``'s definition); the new label wins when its
    class's marker fraction is the larger one.

    Args:
        sample: The sample.
        scores: ``marker_class_scores`` of its counts, columns in
            ``classes`` order.
        classes: The referee's classes.

    Returns:
        One row per class pair, most frequent first.
    """
    column = {cls: index for index, cls in enumerate(classes)}
    new = new_labels(sample.labels)
    legacy = sample.legacy_broad
    rows: list[dict[str, Any]] = []
    tally: dict[tuple[str, str], list[int]] = {}
    for row, (a, b) in enumerate(zip(new, legacy, strict=True)):
        if a is None or b is None or a not in column or b not in column or a == b:
            continue
        score_a = scores[row, column[str(a)]]
        score_b = scores[row, column[str(b)]]
        counts = tally.setdefault((str(a), str(b)), [0, 0, 0])
        counts[0 if score_a > score_b else 1 if score_b > score_a else 2] += 1
    n_disputes = sum(sum(value) for value in tally.values())
    for (a, b), (new_wins, legacy_wins, ties) in sorted(
        tally.items(), key=lambda item: (-sum(item[1]), item[0])
    ):
        n = new_wins + legacy_wins + ties
        rows.append(
            {
                "pair": sample.pair,
                "segmentation": sample.segmentation,
                "sample_id": sample.sample_id,
                "new": a,
                "legacy": b,
                "n": n,
                "share_of_disputes": n / n_disputes,
                "new_wins": new_wins / n,
                "legacy_wins": legacy_wins / n,
                "undecided": ties / n,
            }
        )
    return rows


def _same(left: Any, right: Any) -> bool:
    try:
        a, b = float(left), float(right)
    except (TypeError, ValueError):
        return str(left) == str(right)
    if math.isnan(a) and math.isnan(b):
        return True
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


def check_against(
    rows: Iterable[Mapping[str, Any]], criteria_referee: pd.DataFrame
) -> list[str]:
    """Return every difference from a ``resolve_criteria.py`` ``referee.csv``.

    Args:
        rows: This script's referee rows.
        criteria_referee: ``referee.csv`` of a ``resolve_criteria.py`` run.

    Returns:
        One message per differing value or missing sample (empty: identical).
    """
    theirs = criteria_referee[
        (criteria_referee["first"] == NEW) & (criteria_referee["second"] == LEGACY)
    ]
    index = {
        (str(row["segmentation"]), str(row["sample_id"])): row
        for row in theirs.to_dict("records")
    }
    problems: list[str] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (str(row["segmentation"]), str(row["sample_id"]))
        seen.add(key)
        other = index.get(key)
        if other is None:
            problems.append(f"{key}: not in the criteria referee.csv")
            continue
        for column in CHECKED:
            if not _same(row[column], other[column]):
                problems.append(f"{key} {column}: {row[column]} != {other[column]}")
    for key in sorted(set(index) - seen):
        problems.append(f"{key}: only in the criteria referee.csv")
    return problems


def main(argv: list[str] | None = None) -> int:
    """Score H10 on RESOLVE outputs against the legacy broad_class."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--resolve-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pairs", default=",".join(PAIRS))
    parser.add_argument("--segmentations", default=",".join(ALL_SEGMENTATIONS))
    parser.add_argument(
        "--check-against",
        type=Path,
        help="resolve_criteria.py referee.csv; exit 1 on any difference",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    referee: list[dict[str, Any]] = []
    disputes: list[dict[str, Any]] = []
    for segmentation in [item for item in args.segmentations.split(",") if item]:
        for pair in [item for item in args.pairs.split(",") if item]:
            resolve_dir = args.resolve_root / pair / segmentation
            summary_path = resolve_dir / f"{pair}_resolve_summary.json"
            if not summary_path.is_file():
                logger.warning("no RESOLVE outputs for %s %s", pair, segmentation)
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            for platform in PLATFORMS:
                sample = load_referee_sample(
                    resolve_dir,
                    args.results_root,
                    summary,
                    pair,
                    segmentation,
                    platform,
                )
                scores, _ = marker_class_scores(sample.counts, sample.symbols)
                referee.append(referee_row(sample, scores))
                disputes += dispute_rows(sample, scores)
                logger.info("%s %s %s refereed", pair, segmentation, platform)
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(referee).to_csv(args.out / "referee.csv", index=False)
    pd.DataFrame(disputes).to_csv(args.out / "referee_disputes.csv", index=False)
    pd.DataFrame(referee_table_rows(referee)).to_csv(
        args.out / "h10_table.csv", index=False
    )
    problems: list[str] = []
    if args.check_against is not None:
        problems = check_against(referee, pd.read_csv(args.check_against))
    record = {
        "resolve_root": str(args.resolve_root),
        "results_root": str(args.results_root),
        "samples": len(referee),
        "check_against": None
        if args.check_against is None
        else str(args.check_against),
        "check_differences": problems,
    }
    (args.out / "marker_referee_run.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    for problem in problems:
        logger.error("differs from resolve_criteria: %s", problem)
    logger.info("wrote %s (%d samples)", args.out, len(referee))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

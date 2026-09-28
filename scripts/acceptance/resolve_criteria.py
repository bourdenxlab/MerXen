#!/usr/bin/env python
"""Human acceptance criteria on RESOLVE label tables (plan §14; M4 exit, stage D).

Re-measures the pre-registered human criteria with M4's final rules: the
confident statuses, gate, flags and compositions RESOLVE wrote
(``merxen annotate-resolve``: resolvability-gated emission reweighted to each
dataset, floors, the SEA-AD vote and the COP rule), instead of the M3 shadow
rules. The metric definitions are the pre-registration's §2 (the M3
acceptance scripts' functions are reused); H9, which M3 did not measure, uses
the marker pseudo-labels of the dataset reports the plan names
(``marker_pseudo_labels.py``). No threshold is changed here: each row carries
the pre-registered threshold and whether the §14 human flip rule scores it.

Inputs (all read-only):

- ``--resolve-root``: ``<root>/<pair>/<seg>/`` RESOLVE outputs
  (``<pair>_resolve_summary.json``, ``<plat>/<sid>_celltype_labels.parquet``);
- ``--runs-root``: the MAP outputs they resolved (SEA-AD tidy parquets for
  H3; panel files and ``map_manifest.json`` for H4);
- ``--results-root``: the published clustered H5ADs (counts, log-normalised
  ``X``, legacy ``broad_class``);
- ``--heldout-root`` (H4): the M3 held-out-gene WHB re-maps
  (``runs/<pair>/<sid>_set_a_heldout.parquet``) and ``--store`` (the
  bundles RESOLVE read, the store's current ones).

Writes to ``--out``:

- ``criteria_samples.csv``: per sample x segmentation, H2, H3, H5, H7, H8,
  H9 (both pseudo-label methods), H10 and the gate;
- ``criteria_jsd.csv``: H1 / H17 soft JSD rows (whole section and shared
  mask, 95% block-bootstrap CI) and the other composition kinds;
- ``criteria_h16.csv``: every flag x class x platform stratum with the H16
  marking check;
- ``referee.csv``: the E1 marker referee (new = RESOLVE confident broad);
- ``heldout_enrichment_m4.csv`` / ``heldout_h4_m4.csv``: H4 per label set,
  including RESOLVE run on the held-out re-map (``h4_resolve/``);
- ``criteria_table.csv``: one row per criterion x dataset (value,
  threshold, pass, scored by the flip rule).

Usage::

    python scripts/acceptance/resolve_criteria.py \\
        --resolve-root <m4>/shadow/runs --runs-root <shadow>/baselines/runs \\
        --results-root <results> --heldout-root <shadow>/heldout \\
        --store <annotation_references> --out <m4>/criteria
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from merxen.annotation import pipeline
from merxen.annotation.config import AnnotationConfig
from merxen.annotation.mapmycells_engine import (
    MmcBundle,
    level_frame,
    read_tidy_parquet,
)
from merxen.annotation.panel import AnnotationPanel
from merxen.annotation.pipeline import (
    annotate_resolve,
    current_store_bundles,
    load_map_manifest,
    read_label_table,
)
from merxen.annotation.schema import meets_threshold
from merxen.annotation.shadow import (
    AGREEMENT_MIN_COUNTS,
    OPC,
    argmax_broad_names,
    cop_rule_broad_names,
    evaluate_human_rules,
    heldout_enrichment,
    label_agreement,
    marker_class_scores,
    marker_referee,
    published_queries,
    rule_inputs_from_provisional,
    seaad_broad_calls,
    select_heldout_markers,
    whb_labels_from_tidy,
)
from merxen.annotation.store import ReferenceStore
from merxen.annotation.vocab import (
    COP_SUPERCLUSTER,
    load_heldout_markers,
    primary_vocab,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from heldout_genes import (  # noqa: E402
    H4_MIN_CLASSES,
    enrichment_rows,
    marker_count_frame,
    marker_ids,
)
from marker_pseudo_labels import (  # noqa: E402
    METHODS,
    h9_agreement,
    p1212_pseudo_labels,
    p7513_pseudo_labels,
    resolved_six,
)
from shadow_baselines import (  # noqa: E402
    PAIRS,
    PLATFORMS,
    SEGMENTATIONS,
    _clustered_path,
    add_fallback_arguments,
)

logger = logging.getLogger("resolve_criteria")

# Pre-registered thresholds (plan §14 = pre-registration §9; never changed here).
H1_THRESHOLDS: dict[str, float] = {
    "P7513": 0.17,
    "P1212": 0.17,
    "P7113": 0.17,
    "P5011": 0.31,
}
H17_MARGIN = 0.02
H2_MAX = 0.01
H3_THRESHOLDS: dict[str, float] = {
    "P7513": 0.90,
    "P1212": 0.80,
    "P7113": 0.80,
    "P5011": 0.80,
}
H5_COP_MAX = 0.02
H5_OPC_MAX = 0.10
H7_THRESHOLDS: dict[tuple[str, str], float] = {
    ("P7513", "MERSCOPE"): 0.62,
    ("P7513", "XENIUM"): 0.44,
    ("P7113", "MERSCOPE"): 0.67,
    ("P7113", "XENIUM"): 0.39,
    ("P1212", "MERSCOPE"): 0.34,
    ("P1212", "XENIUM"): 0.29,
    ("P5011", "MERSCOPE"): 0.30,
    ("P5011", "XENIUM"): 0.40,
}
H8_BROAD_ONLY: frozenset[str] = frozenset({"P1212_MERSCOPE", "P5011_MERSCOPE"})
# Expected warning flags: §5.4 / H8 on proseg_hybrid; on reseg (not scored by
# H8) the M3 baselines of the pre-registration §5, for information.
H8_WARNING: dict[str, frozenset[str]] = {
    "proseg_hybrid": frozenset({"P5011_MERSCOPE"}),
    "reseg": frozenset({"P1212_MERSCOPE", "P5011_MERSCOPE"}),
}
H9_THRESHOLDS: dict[tuple[str, str], float] = {
    ("P7513", "MERSCOPE"): 0.90,
    ("P7513", "XENIUM"): 0.83,
    ("P1212", "MERSCOPE"): 0.75,
    ("P1212", "XENIUM"): 0.75,
}
# The report whose pseudo-label method H9 names for each pair.
H9_METHOD: dict[str, str] = {"P7513": "p7513_c", "P1212": "p1212_4"}
H10_MIN = 0.70
# A failing class within this of its AUROC ceiling fails on detection alone.
H4_CEILING_SLACK = 0.02
H16_RATE = 0.15

DEV_PAIRS = ("P7513", "P1212")
DONOR_PAIRS = ("P7113", "P5011")
H4_PAIRS = ("P7513", "P1212", "P7113")
H17_PAIRS = ("P7513", "P1212")
DEV_CRITERIA = frozenset({"H1", "H2", "H3", "H5", "H7", "H8", "H9", "H10", "H16"})
DONOR_CRITERIA = frozenset({"H1", "H2", "H3", "H7", "H8"})
SEA_RUN = "seaad_mr_panel"
WHB_RUN = "whb_frontal_supc_clus"

# Label sets with this prefix are diagnostics, never an H4 option.
H4_DIAGNOSTIC_PREFIX = "diag_"
H4_DIAGNOSTIC_DEPTHS: tuple[int, ...] = (30, 60)
H4_LABEL_SETS: tuple[str, ...] = (
    "heldout_argmax",
    "heldout_argmax_cop_rule",
    "heldout_argmax_cop_rule_seaad",
    "heldout_whb_confident",
    "m4_resolve_heldout",
    "m4_resolve_heldout_whb_only",
    "m4_production_confident",
)


def scored(criterion: str, pair: str, segmentation: str) -> bool:
    """Whether the §14 human flip rule scores a criterion on a dataset.

    H1-H3, H5, H7-H10, H12-H16 on P7513 and P1212; H1-H3, H7, H8, H12 on
    P7113 and P5011; H4 on P7513, P1212 and P7113; H17 (H1 + 0.02, H2, H9)
    on P7513 and P1212 reseg; everything on proseg_hybrid otherwise.

    Args:
        criterion: ``H1`` ... ``H18``, ``H17/<component>`` or ``<id>/<part>``
            (scored as ``<id>``, e.g. ``H8/warning``).
        pair: Pair id.
        segmentation: Segmentation.

    Returns:
        Whether the flip rule scores it.
    """
    if criterion.startswith("H17"):
        return segmentation == "reseg" and pair in H17_PAIRS
    criterion = criterion.split("/")[0]
    if segmentation != "proseg_hybrid":
        return False
    if criterion == "H4":
        return pair in H4_PAIRS
    if pair in DEV_PAIRS:
        return criterion in DEV_CRITERIA
    if pair in DONOR_PAIRS:
        return criterion in DONOR_CRITERIA
    return False


def criterion_row(
    criterion: str,
    pair: str,
    segmentation: str,
    dataset: str,
    value: float | None,
    threshold: float | None,
    comparator: str,
    *,
    note: str = "",
    passes: bool | None = None,
) -> dict[str, Any]:
    """Return one ``criteria_table.csv`` row.

    Args:
        criterion: Criterion id (``H7``, ``H17/H1``, ``H4[m4_resolve_heldout]``).
        pair: Pair id.
        segmentation: Segmentation.
        dataset: Sample id, pair id or label.
        value: Measured value (``None`` / NaN: not measured).
        threshold: Pre-registered threshold.
        comparator: ``<=``, ``>=`` or ``==`` (mechanical checks).
        note: Free text.
        passes: Override for mechanical checks; default from the comparison.

    Returns:
        The row.
    """
    measured = value is not None and not (
        isinstance(value, float) and math.isnan(value)
    )
    if passes is None and measured and threshold is not None:
        passes = (
            bool(value <= threshold + 1e-12)  # type: ignore[operator]
            if comparator == "<="
            else bool(value >= threshold - 1e-12)  # type: ignore[operator]
        )
    base = criterion.split("[")[0]
    return {
        "criterion": criterion,
        "pair": pair,
        "segmentation": segmentation,
        "dataset": dataset,
        "value": value,
        "comparator": comparator,
        "threshold": threshold,
        "passes": passes,
        "scored": scored(base, pair, segmentation),
        "note": note,
    }


@dataclass
class ResolvedSample:
    """One sample's RESOLVE table cells and published context.

    Attributes:
        pair: Pair id.
        segmentation: Segmentation.
        sample_id: Sample id.
        platform: ``MERSCOPE`` or ``XENIUM``.
        labels: Label-table rows of the table cells (index ``cell_id``).
        summary: The sample's entry of ``<pair>_resolve_summary.json``.
        sea: E2 SEA-AD broad calls aligned with ``labels``.
        counts: Raw counts of the table cells (published order of
            ``labels``).
        lognorm: The published log-normalised ``X`` of the same cells.
        symbols: Gene symbol per column.
        legacy_broad: Published ``obs["broad_class"]``.
    """

    pair: str
    segmentation: str
    sample_id: str
    platform: str
    labels: pd.DataFrame
    summary: dict[str, Any]
    sea: pd.DataFrame
    counts: sparse.csr_matrix
    lognorm: sparse.csr_matrix
    symbols: list[str]
    legacy_broad: np.ndarray


def load_resolved_sample(
    resolve_dir: Path,
    runs_dir: Path,
    results: Path,
    summary: Mapping[str, Any],
    pair: str,
    segmentation: str,
    platform: str,
) -> ResolvedSample:
    """Load one sample's label table, SEA-AD calls and published matrix."""
    import anndata as ad

    sample_id = f"{pair}_{platform}"
    entry = summary["samples"][sample_id]
    labels, _ = read_label_table(resolve_dir / entry["labels"])
    labels = labels[labels["in_table"].to_numpy(bool)].set_index("cell_id")
    labels.index = labels.index.astype(str)
    tidy, _ = read_tidy_parquet(
        runs_dir / platform.lower() / f"{sample_id}_mmc_{SEA_RUN}.parquet"
    )
    sea = seaad_broad_calls(
        level_frame(tidy, "subclass"),
        level_frame(tidy, "supertype"),
        class_level=level_frame(tidy, "class"),
    )
    sea.index = sea.index.astype(str)
    adata = ad.read_h5ad(_clustered_path(results, pair, segmentation, platform))
    obs = pd.Index(adata.obs_names.astype(str))
    position = obs.get_indexer(labels.index)
    if (position < 0).any():
        raise ValueError(f"{sample_id}: labelled cells missing from the clustered H5AD")
    return ResolvedSample(
        pair=pair,
        segmentation=segmentation,
        sample_id=sample_id,
        platform=platform,
        labels=labels,
        summary=dict(entry),
        sea=sea.reindex(labels.index),
        counts=sparse.csr_matrix(adata.layers["counts"])[position],
        lognorm=sparse.csr_matrix(adata.X)[position],
        symbols=[str(name) for name in adata.var_names],
        legacy_broad=np.asarray(adata.obs["broad_class"].astype(object))[position],
    )


def confident(labels: pd.DataFrame, level: str) -> np.ndarray:
    """Return whether each cell's ``ct_<level>`` status is ``confident``."""
    return labels[f"ct_{level}_status"].astype(str).to_numpy() == "confident"


def cop_control(labels: pd.DataFrame) -> dict[str, float]:
    """Return the H5 shares of table cells (§2: confident COP / OPC, COP-derived)."""
    supercluster = labels["mmc_whb_supercluster_name"].astype(object).to_numpy()
    is_cop = supercluster == COP_SUPERCLUSTER
    cop = confident(labels, "supercluster") & (
        labels["ct_supercluster_name"].astype(object).to_numpy() == COP_SUPERCLUSTER
    )
    opc = confident(labels, "broad") & (
        labels["ct_broad_name"].astype(object).to_numpy() == OPC
    )
    return {
        "cop_argmax_share": float(is_cop.mean()),
        "cop_confident_supercluster_share": float(cop.mean()),
        "opc_confident_broad_share": float(opc.mean()),
        "cop_derived_share_of_confident_opc": (
            float((opc & is_cop).sum() / opc.sum()) if opc.any() else math.nan
        ),
        "cop_suppressed_share": float(labels["flag_cop_suppressed"].mean()),
    }


def pseudo_labellings(sample: ResolvedSample) -> dict[str, np.ndarray]:
    """Return both reports' resolved 6-class pseudo-labels of the cells (H9)."""
    frames = {
        "p7513_c": p7513_pseudo_labels(sample.counts, sample.lognorm, sample.symbols),
        "p1212_4": p1212_pseudo_labels(sample.counts, sample.symbols),
    }
    return {method: resolved_six(frames[method], method) for method in METHODS}


def sample_criteria(
    sample: ResolvedSample, pseudo: Mapping[str, np.ndarray] | None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return the per-sample criteria row and the H9 detail rows."""
    labels = sample.labels
    counts = labels["total_counts"].to_numpy(np.float64)
    broad_confident = confident(labels, "broad")
    whb = argmax_broad_names(labels)
    ge20 = counts >= AGREEMENT_MIN_COUNTS
    agree, n_ge20 = label_agreement(whb, sample.sea["broad"], ge20)
    agree_strict, _ = label_agreement(whb, sample.sea["broad"], ge20, strict=True)
    implausible = labels["flag_implausible"].to_numpy(bool)
    supercluster = labels["mmc_whb_supercluster_name"].astype(object).to_numpy()
    resolution = sample.summary["resolution"]
    gate = resolution["gate"]
    row: dict[str, Any] = {
        "pair": sample.pair,
        "segmentation": sample.segmentation,
        "sample_id": sample.sample_id,
        "platform": sample.platform,
        "n_table_cells": len(labels),
        "median_counts": float(np.median(counts)),
        "trust": sample.summary["trust"]["state"],
        "degraded_mode": resolution["degraded_mode"]["name"],
        "gate_level": gate["level"],
        "gate_warning": bool(gate["warning"]),
        "gate_A_frac_ge30": gate["frac_ge30"],
        "gate_reasons": " | ".join(gate["level_reasons"] + gate["warning_reasons"]),
        "H2_implausible_share": float(implausible.mean()),
        "H2_top_nodes": json.dumps(
            pd.Series(supercluster[implausible], dtype=object)
            .value_counts()
            .head(4)
            .to_dict()
        ),
        "H2_implausible_lineage_confident_share": float(
            (implausible & confident(labels, "lineage")).mean()
        ),
        "H3_whb_sea_agree_ge20": agree,
        "H3_whb_sea_agree_ge20_strict": agree_strict,
        "H3_n_ge20": n_ge20,
        "H7_broad_coverage_table": float(broad_confident.mean()),
        "H7_broad_coverage_segmented": gate["broad_coverage_segmented"],
        "lineage_coverage_table": float(confident(labels, "lineage").mean()),
        "supercluster_coverage_table": float(confident(labels, "supercluster").mean()),
        "resolvability_extrapolated_share": resolution[
            "resolvability_extrapolated_share"
        ],
        **{f"H5_{key}": value for key, value in cop_control(labels).items()},
    }
    detail: list[dict[str, Any]] = []
    if pseudo is not None:
        names = labels["ct_broad_name"].astype(object).to_numpy()
        for method, six in pseudo.items():
            result = h9_agreement(names, broad_confident, six, method)
            row[f"H9_{method}"] = result.agreement
            row[f"H9_{method}_n_scored"] = result.n_scored
            detail.append(
                {
                    "pair": sample.pair,
                    "segmentation": sample.segmentation,
                    "sample_id": sample.sample_id,
                    **result.to_row(),
                    "pseudo_resolved_share": float(
                        np.mean([value is not None for value in six])
                    ),
                }
            )
    return row, detail


def _top(values: np.ndarray, n: int = 3) -> str:
    """The ``n`` most frequent values with their shares, as JSON."""
    series = pd.Series(values, dtype=object).dropna()
    if not len(series):
        return "{}"
    shares = series.value_counts(normalize=True).head(n)
    return json.dumps(
        {str(key): round(float(value), 3) for key, value in shares.items()}
    )


def h2_breakdown(sample: ResolvedSample) -> list[dict[str, Any]]:
    """Return the H2 diagnostics of one sample (the M8 fix / exception input).

    Per implausible WHB node (and all nodes pooled): the share of table
    cells, the share whose lineage is still confident (an implausible node
    keeps its lineage when SEA-AD agrees at lineage), median counts, the
    SEA-AD subclass and the WHB runner-up of those cells, and H2 without the
    node (what a vocab change that made it plausible would give; not a
    metric, which only the user can change).
    """
    labels = sample.labels
    vocab = primary_vocab("human")
    implausible = labels["flag_implausible"].to_numpy(bool)
    node = labels["mmc_whb_supercluster_name"].astype(object).to_numpy()
    lineage = confident(labels, "lineage")
    counts = labels["total_counts"].to_numpy(np.float64)
    sea_subclass = labels["mmc_seaad_subclass_name"].astype(object).to_numpy()
    runner_up = (
        labels["mmc_whb_supercluster_runner_up_1_name"].astype(object).to_numpy()
    )
    n_table = len(labels)
    rows = []
    groups = [("all", implausible)] + [
        (str(name), implausible & (node == name))
        for name in pd.Series(node[implausible], dtype=object).value_counts().index
    ]
    for name, mask in groups:
        if not mask.any():
            continue
        kind = (
            "all"
            if name == "all"
            else "sink"
            if name in vocab and vocab.is_sink(name)
            else "region_implausible"
        )
        rows.append(
            {
                "pair": sample.pair,
                "segmentation": sample.segmentation,
                "sample_id": sample.sample_id,
                "node": name,
                "kind": kind,
                "n": int(mask.sum()),
                "share_table": float(mask.sum() / n_table),
                "lineage_confident_share": float(lineage[mask].mean()),
                "median_counts": float(np.median(counts[mask])),
                "h2_without_node": (
                    float((implausible & ~mask).sum() / n_table)
                    if name != "all"
                    else math.nan
                ),
                "seaad_subclass_top": _top(sea_subclass[mask]),
                "whb_runner_up_top": _top(runner_up[mask]),
            }
        )
    return rows


def referee_rows(sample: ResolvedSample) -> list[dict[str, Any]]:
    """Return the E1 marker-referee rows (H10: new = RESOLVE confident broad)."""
    labels = sample.labels
    scores, _ = marker_class_scores(sample.counts, sample.symbols)
    broad_confident = confident(labels, "broad")
    new = np.where(
        broad_confident, labels["ct_broad_name"].astype(object).to_numpy(), None
    )
    whb = argmax_broad_names(labels)
    sea = sample.sea["broad"].to_numpy(object)
    comparisons = {
        ("new confident (RESOLVE)", "legacy broad_class"): (new, sample.legacy_broad),
        ("WHB argmax", "SEA-AD argmax"): (whb, sea),
        ("new confident (RESOLVE)", "SEA-AD argmax"): (new, sea),
    }
    rows = []
    for (first, second), (a, b) in comparisons.items():
        result = marker_referee(scores, a, b)
        rows.append(
            {
                "pair": sample.pair,
                "segmentation": sample.segmentation,
                "sample_id": sample.sample_id,
                "first": first,
                "second": second,
                "n_cells": result.n_cells,
                "n_disputes": result.n_disputes,
                "dispute_share": result.n_disputes / max(result.n_cells, 1),
                "first_wins": result.first_wins,
                "second_wins": result.second_wins,
                "undecided": result.undecided,
                "top_disputes": json.dumps(result.top_disputes),
            }
        )
    return rows


def jsd_rows(summary: Mapping[str, Any], pair: str, segmentation: str) -> list[dict]:
    """Return the pair JSD rows RESOLVE wrote, with the H1 / H17 threshold."""
    threshold = H1_THRESHOLDS.get(pair)
    if threshold is not None and segmentation != "proseg_hybrid":
        threshold += H17_MARGIN
    rows = []
    for item in summary["pair"]["jsd"]:
        rows.append(
            {
                "pair": pair,
                "segmentation": segmentation,
                **item,
                "threshold": threshold if item["kind"] == "soft" else None,
                "mask_note": summary["pair"].get("mask_note"),
            }
        )
    return rows


def h16_rows(
    summary: Mapping[str, Any], pair: str, segmentation: str
) -> list[dict[str, Any]]:
    """Return every flag stratum with the H16 marking check.

    H16 (mechanical): realised rates for every flag x class x platform, and
    every stratum above 15% marked uninformative. A stratum is correctly
    marked when its rate is at most 15%, or it carries
    ``informative_h16 = false`` (or no rate: no null, flag not computed).
    """
    rows = []
    for sample_id, sample in summary["samples"].items():
        for stratum in sample["flags"]["strata"]:
            rate = stratum.get("rate")
            has_rate = rate is not None and not (
                isinstance(rate, float) and math.isnan(rate)
            )
            above = bool(has_rate and float(rate) > H16_RATE)
            marked = (not has_rate) or (not above) or not stratum["informative_h16"]
            rows.append(
                {
                    "pair": pair,
                    "segmentation": segmentation,
                    "sample_id": sample_id,
                    **stratum,
                    "above_h16": above,
                    "h16_marked_ok": bool(marked),
                }
            )
    return rows


def table_rows_for_sample(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the criteria-table rows of one sample."""
    pair, seg = str(row["pair"]), str(row["segmentation"])
    sid, platform = str(row["sample_id"]), str(row["platform"])
    out = [
        criterion_row("H2", pair, seg, sid, row["H2_implausible_share"], H2_MAX, "<="),
        criterion_row(
            "H3",
            pair,
            seg,
            sid,
            row["H3_whb_sea_agree_ge20"],
            H3_THRESHOLDS.get(pair),
            ">=",
        ),
        criterion_row(
            "H5",
            pair,
            seg,
            sid,
            row["H5_cop_confident_supercluster_share"],
            H5_COP_MAX,
            "<=",
            note="confident COP supercluster share",
        ),
        criterion_row(
            "H5",
            pair,
            seg,
            sid,
            row["H5_opc_confident_broad_share"],
            H5_OPC_MAX,
            "<=",
            note=(
                "confident broad OPC share; COP-derived share "
                f"{row['H5_cop_derived_share_of_confident_opc']:.3f}"
            ),
        ),
        criterion_row(
            "H7",
            pair,
            seg,
            sid,
            row["H7_broad_coverage_table"],
            H7_THRESHOLDS.get((pair, platform)),
            ">=",
        ),
    ]
    expected = "broad_only" if sid in H8_BROAD_ONLY else "full"
    warning_expected = sid in H8_WARNING.get(seg, frozenset())
    basis = "§5.4" if seg == "proseg_hybrid" else "M3 baseline"
    level_ok = row["gate_level"] == expected
    note = (
        f"level {row['gate_level']} (expected {expected}); "
        f"warning {row['gate_warning']}"
    )
    if bool(row["gate_warning"]) != warning_expected:
        note += f" (the {basis} expectation is {warning_expected})"
    out.append(
        criterion_row(
            "H8",
            pair,
            seg,
            sid,
            None,
            None,
            "==",
            note=note,
            passes=bool(level_ok and row["gate_level"] != "failed"),
        )
    )
    # The warning flag against its expectation (H8 names the levels; §5.4
    # expects the warning on P5011_MERSCOPE only): reported as its own row
    # so an unexpected warning is never hidden behind a passing level.
    out.append(
        criterion_row(
            "H8/warning",
            pair,
            seg,
            sid,
            float(row["H7_broad_coverage_segmented"]),
            None,
            "==",
            note=(
                f"warning {bool(row['gate_warning'])} (the {basis} expectation "
                f"is {warning_expected}); confident broad coverage of segmented "
                "objects (warning below 0.15)"
            ),
            passes=bool(row["gate_warning"]) == warning_expected,
        )
    )
    method = H9_METHOD.get(pair)
    if method is not None and f"H9_{method}" in row:
        criterion = "H9" if seg == "proseg_hybrid" else "H17/H9"
        out.append(
            criterion_row(
                criterion,
                pair,
                seg,
                sid,
                row[f"H9_{method}"],
                H9_THRESHOLDS.get((pair, platform)),
                ">=",
                note=(
                    f"pseudo-labels {method} "
                    f"({int(row[f'H9_{method}_n_scored'])} scored)"
                ),
            )
        )
    if seg != "proseg_hybrid":
        out.append(
            criterion_row(
                "H17/H2", pair, seg, sid, row["H2_implausible_share"], H2_MAX, "<="
            )
        )
    return out


def referee_table_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return the H10 criteria-table rows (new vs legacy referee)."""
    return [
        criterion_row(
            "H10",
            str(row["pair"]),
            str(row["segmentation"]),
            str(row["sample_id"]),
            float(row["first_wins"]),
            H10_MIN,
            ">=",
            note=f"{int(row['n_disputes'])} disputes",
        )
        for row in rows
        if row["first"] == "new confident (RESOLVE)"
        and row["second"] == "legacy broad_class"
    ]


def jsd_table_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return the H1 / H17 criteria-table rows (soft, whole section)."""
    out = []
    for row in rows:
        if row["kind"] != "soft" or row["region"] != "whole_section":
            continue
        pair, seg = str(row["pair"]), str(row["segmentation"])
        criterion = "H1" if seg == "proseg_hybrid" else "H17/H1"
        out.append(
            criterion_row(
                criterion,
                pair,
                seg,
                pair,
                float(row["jsd"]),
                row["threshold"],
                "<=",
                note=(
                    f"95% CI [{row['ci_low']:.3f}, {row['ci_high']:.3f}] "
                    f"({row['resampling']})"
                ),
            )
        )
    return out


def h16_table_rows(rows: pd.DataFrame) -> list[dict[str, Any]]:
    """Return one H16 criteria-table row per pair x segmentation."""
    out = []
    for (pair, seg), frame in rows.groupby(["pair", "segmentation"]):
        n_above = int(frame["above_h16"].sum())
        bad = int((~frame["h16_marked_ok"].astype(bool)).sum())
        out.append(
            criterion_row(
                "H16",
                str(pair),
                str(seg),
                str(pair),
                None,
                None,
                "==",
                note=(
                    f"{len(frame)} strata reported; {n_above} above 15%, "
                    f"{bad} of them not marked"
                ),
                passes=bad == 0,
            )
        )
    return out


# ---------------------------------------------------------------------------
# H4 with M4's rules


def _patched_runs(
    heldout: Mapping[str, Path], *, whb_only: bool
) -> Callable[..., dict[str, pipeline.ResolveRun]]:
    """Return a ``load_resolve_runs`` that swaps in the held-out WHB re-map."""
    original = pipeline.load_resolve_runs

    def load(
        map_dir: Path | str, sample: Any, **kwargs: Any
    ) -> dict[str, pipeline.ResolveRun]:
        runs = original(map_dir, sample, **kwargs)
        out: dict[str, pipeline.ResolveRun] = {}
        for run_id, run in runs.items():
            record = run.record
            if record.role == "primary" and "annotation" in record.purposes:
                tidy, meta = read_tidy_parquet(heldout[sample.sample_id])
                if meta.get("build_hash") != record.build_hash:
                    raise ValueError(
                        f"{heldout[sample.sample_id]} was mapped with another bundle"
                    )
                run = dataclasses.replace(run, tidy=tidy)
            elif whb_only and record.role == "secondary":
                continue
            out[run_id] = run
        return out

    return load


def resolve_heldout(
    map_dir: Path,
    heldout: Mapping[str, Path],
    out_dir: Path,
    config: AnnotationConfig,
    store: Path,
    n_segmented: Mapping[str, int],
    *,
    whb_only: bool,
) -> dict[str, pd.DataFrame]:
    """Run RESOLVE with the held-out WHB re-map as the primary call (H4, M4 rules).

    Everything else is the production RESOLVE: the current bundles'
    resolvability tables reweighted to the re-map's composition, floors, the
    gate and the COP rule. With ``whb_only`` the SEA-AD run is left out
    (degraded mode ``whb_only`` with ``allow_single_method``: WHB decides
    alone, and no method saw the held-out genes); otherwise the production
    SEA-AD calls vote (they saw the held-out genes, as option (b')).

    Returns:
        Sample id to its table cells' label rows (index ``cell_id``).
    """
    run_config = config.model_copy(update={"allow_single_method": whb_only})
    original = pipeline.load_resolve_runs
    pipeline.load_resolve_runs = _patched_runs(heldout, whb_only=whb_only)  # type: ignore[assignment]
    try:
        result = annotate_resolve(
            map_dir,
            run_config,
            output_dir=out_dir,
            bundle_finder=current_store_bundles(ReferenceStore(store)),
            n_segmented=n_segmented,
            n_bootstrap=1,
        )
    finally:
        pipeline.load_resolve_runs = original  # type: ignore[assignment]
    tables = {}
    for sample_id, resolution in result.samples.items():
        labels = resolution.labels
        labels = labels[labels["in_table"].to_numpy(bool)].set_index("cell_id")
        labels.index = labels.index.astype(str)
        tables[sample_id] = labels
    return tables


def h4_pair(
    pair: str,
    args: argparse.Namespace,
    config: AnnotationConfig,
    n_segmented: Mapping[str, int],
) -> list[dict[str, Any]]:
    """Score H4 on one pair for every label set (M3 options and M4's rules)."""
    run_dir = args.runs_root / pair / "proseg_hybrid"
    manifest = load_map_manifest(run_dir / "map_manifest.json")
    panel = AnnotationPanel.model_validate_json(
        (run_dir / "panel" / "panel_genes.json").read_text(encoding="utf-8")
    )
    clustered = {
        platform: _clustered_path(args.results_root, pair, "proseg_hybrid", platform)
        for platform in PLATFORMS
    }
    queries = published_queries(
        clustered,
        panel,
        species="human",
        gene_id_fallback_csv=args.gene_id_fallback_csv,
    )
    heldout = {
        f"{pair}_{platform}": args.heldout_root
        / "runs"
        / pair
        / f"{pair}_{platform}_set_a_heldout.parquet"
        for platform in PLATFORMS
    }
    resolved = {
        name: resolve_heldout(
            run_dir,
            heldout,
            args.out / "h4_resolve" / name / pair,
            config,
            args.store,
            n_segmented,
            whb_only=whb_only,
        )
        for name, whb_only in (
            ("m4_resolve_heldout", False),
            ("m4_resolve_heldout_whb_only", True),
        )
    }
    marker_table = load_heldout_markers("human")
    rows: list[dict[str, Any]] = []
    for platform in PLATFORMS:
        sample_id = f"{pair}_{platform}"
        record = manifest.samples[sample_id].runs[WHB_RUN]
        bundle = MmcBundle.from_dir(record.bundle_path)
        vocab = pd.read_csv(
            bundle.path / "vocab_snapshot.csv", dtype=str, keep_default_na=False
        )
        symbols = panel.symbols_by_platform.get(platform) or panel.symbols
        selection = select_heldout_markers(marker_table, symbols, platform)
        ids = marker_ids(panel, platform, selection.all_markers)
        query = queries[platform]
        tidy, _ = read_tidy_parquet(heldout[sample_id])
        total = pd.Series(query.total_counts, index=pd.Index(query.cell_ids))
        labels = whb_labels_from_tidy(tidy, vocab, total)
        cells = labels.index
        counts = marker_count_frame(query, ids, cells)
        total_counts = labels["total_counts"].to_numpy(np.float64)
        argmax = argmax_broad_names(labels)
        sea_tidy, _ = read_tidy_parquet(
            run_dir / platform.lower() / f"{sample_id}_mmc_{SEA_RUN}.parquet"
        )
        sea = seaad_broad_calls(
            level_frame(sea_tidy, "subclass"),
            level_frame(sea_tidy, "supertype"),
            class_level=level_frame(sea_tidy, "class"),
        ).reindex(cells)
        sea_confident_opc = (sea["broad"].to_numpy(object) == OPC) & meets_threshold(
            sea["broad_raw"], config.thresholds.seaad_broad
        )
        markers = selection.markers
        base = {
            "pair": pair,
            "platform": platform,
            "variant": "set_a",
            "n_cells": len(cells),
            "n_query_genes": len(query.gene_ids) - len(ids),
            "n_heldout_genes": len(ids),
        }
        rows += enrichment_rows(
            heldout_enrichment(counts, total_counts, argmax, markers),
            base,
            "heldout_argmax",
        )
        for label_set, rescue in (
            ("heldout_argmax_cop_rule", None),
            ("heldout_argmax_cop_rule_seaad", sea_confident_opc),
        ):
            cop_labels = cop_rule_broad_names(
                argmax,
                labels["mmc_whb_supercluster_name"].to_numpy(object),
                labels["mmc_whb_supercluster_bp"].to_numpy(np.float64),
                total_counts,
                platform=platform,
                sea_confident_opc=rescue,
            )
            rows += enrichment_rows(
                heldout_enrichment(counts, total_counts, cop_labels, markers),
                base,
                label_set,
            )
        whb_only = evaluate_human_rules(
            rule_inputs_from_provisional(labels), platform=platform, second_vote="none"
        )
        rows += enrichment_rows(
            heldout_enrichment(
                counts, total_counts, argmax, markers, include=whb_only.broad_confident
            ),
            base,
            "heldout_whb_confident",
        )
        production, _ = read_label_table(
            args.resolve_root
            / pair
            / "proseg_hybrid"
            / platform.lower()
            / f"{sample_id}_celltype_labels.parquet"
        )
        production = production.set_index("cell_id")
        production.index = production.index.astype(str)
        tables = {
            **{name: frames[sample_id] for name, frames in resolved.items()},
            "m4_production_confident": production,
        }
        for label_set, table in tables.items():
            table = table.reindex(cells)
            names = table["ct_broad_name"].astype(object).to_numpy()
            rows += enrichment_rows(
                heldout_enrichment(
                    counts,
                    total_counts,
                    names,
                    markers,
                    include=confident(table, "broad"),
                ),
                {**base, "n_confident": int(confident(table, "broad").sum())},
                label_set,
            )
        # Diagnostics (never an H4 option): the RESOLVE held-out labels on
        # deeper cells only, to show how far detection limits the AUROC.
        table = tables["m4_resolve_heldout"].reindex(cells)
        names = table["ct_broad_name"].astype(object).to_numpy()
        for depth in H4_DIAGNOSTIC_DEPTHS:
            include = confident(table, "broad") & (total_counts >= depth)
            rows += enrichment_rows(
                heldout_enrichment(
                    counts, total_counts, names, markers, include=include
                ),
                {**base, "n_confident": int(include.sum())},
                f"{H4_DIAGNOSTIC_PREFIX}m4_resolve_heldout_ge{depth}",
            )
        logger.info("%s %s: H4 label sets scored", pair, platform)
    return rows


def auroc_ceiling(detection_assigned: float, detection_other: float) -> float:
    """The largest AUROC the held-out fraction can reach at these detection rates.

    Cells without a held-out count all score 0: an assigned cell at 0 loses to
    every detected other cell and ties with every undetected one, so even
    with perfect labels AUROC <= d_a + (1 - d_a)(1 - d_o) / 2.

    Args:
        detection_assigned: Share of the assigned cells with >= 1 held-out
            count.
        detection_other: The same in the other cells.

    Returns:
        The bound (NaN when a rate is missing).
    """
    if not (math.isfinite(detection_assigned) and math.isfinite(detection_other)):
        return math.nan
    return detection_assigned + 0.5 * (1.0 - detection_assigned) * (
        1.0 - detection_other
    )


def h4_summary(enrichment: pd.DataFrame) -> list[dict[str, Any]]:
    """Classes passing of 7 per pair, platform and label set."""
    if "auroc_ceiling" not in enrichment:
        enrichment = enrichment.assign(
            auroc_ceiling=[
                auroc_ceiling(float(a), float(o))
                for a, o in zip(
                    enrichment["detection_assigned"],
                    enrichment["detection_other"],
                    strict=True,
                )
            ]
        )
    rows = []
    for (pair, platform, label_set), frame in enrichment.groupby(
        ["pair", "platform", "label_set"]
    ):
        n_pass = int(frame["passes"].sum())
        rows.append(
            {
                "pair": pair,
                "platform": platform,
                "label_set": label_set,
                "n_scored": len(frame),
                "n_pass": n_pass,
                "failing": ";".join(
                    f"{cls} (fold {fold:.1f}, AUROC {auroc:.2f})"
                    for cls, fold, auroc in frame.loc[
                        ~frame["passes"].astype(bool),
                        ["broad_class", "fold", "auroc"],
                    ].itertuples(index=False)
                ),
                "h4_scored_pair": pair in H4_PAIRS,
                "h4_pass": n_pass >= H4_MIN_CLASSES,
                "diagnostic": str(label_set).startswith(H4_DIAGNOSTIC_PREFIX),
                "failing_at_auroc_ceiling": ";".join(
                    str(cls)
                    for cls, auroc, ceiling in frame.loc[
                        ~frame["passes"].astype(bool),
                        ["broad_class", "auroc", "auroc_ceiling"],
                    ].itertuples(index=False)
                    if math.isfinite(auroc) and auroc >= ceiling - H4_CEILING_SLACK
                ),
            }
        )
    return rows


# The headline H4 label set: plan §5.8 asks for labels no method made with the
# held-out genes, so the headline is the WHB-only re-resolve of the held-out
# re-map; the SEA-AD-voted sets saw the held-out genes (partly circular).
H4_HEADLINE_SET = "m4_resolve_heldout_whb_only"
H4_CIRCULAR_SETS: dict[str, str] = {
    "m4_resolve_heldout": "partly circular (SEA saw held-out genes)",
    "m4_production_confident": "circular (every method saw held-out genes)",
}


def h4_table_rows(summary: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return the H4 criteria-table rows (one per label set, no diagnostics).

    Every label set gets an ``H4[<set>]`` row; the non-circular headline set
    (``H4_HEADLINE_SET``) also gives the plain ``H4`` row, and the sets whose
    SEA-AD votes saw the held-out genes are labelled circular in the note.
    """
    rows = []
    for row in summary:
        if row.get("diagnostic", False):
            continue
        label_set = str(row["label_set"])
        note = f"classes passing of {row['n_scored']}; failing: {row['failing']}"
        caveat = H4_CIRCULAR_SETS.get(label_set)
        if caveat is not None:
            note = f"{caveat}; {note}"
        names = [f"H4[{label_set}]"]
        if label_set == H4_HEADLINE_SET:
            names.insert(0, "H4")
            note = f"headline (non-circular: no method saw the held-out genes); {note}"
        for name in names:
            rows.append(
                criterion_row(
                    name,
                    str(row["pair"]),
                    "proseg_hybrid",
                    f"{row['pair']}_{row['platform']}",
                    float(row["n_pass"]),
                    float(H4_MIN_CLASSES),
                    ">=",
                    note=note,
                )
            )
    return rows


# ---------------------------------------------------------------------------


def n_segmented_table(qc_summary: Path) -> dict[tuple[str, str, str], int]:
    """Return (pair, seg, PLATFORM) -> segmented objects (``qc_summary.csv``)."""
    qc = pd.read_csv(qc_summary)
    return {
        (str(row.sample), str(row.seg), str(row.platform).upper()): int(row.n_cells)
        for row in qc.itertuples(index=False)
    }


def main(argv: list[str] | None = None) -> int:
    """Re-measure the human criteria on RESOLVE outputs."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--resolve-root", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--heldout-root", type=Path, help="M3 held-out re-maps (H4)")
    parser.add_argument("--store", type=Path, help="reference store (H4 re-resolve)")
    parser.add_argument(
        "--qc-summary",
        type=Path,
        help="research/lowcount/qc_summary.csv (segmented objects, H4 re-resolve)",
    )
    parser.add_argument("--pairs", default=",".join(PAIRS))
    parser.add_argument("--segmentations", default=",".join(SEGMENTATIONS))
    parser.add_argument("--skip-h4", action="store_true")
    parser.add_argument("--skip-h9", action="store_true")
    add_fallback_arguments(parser)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if not args.skip_h4 and (
        args.heldout_root is None or args.store is None or args.qc_summary is None
    ):
        parser.error("H4 needs --heldout-root, --store and --qc-summary (or --skip-h4)")
    pairs = [item for item in args.pairs.split(",") if item]
    sample_rows: list[dict[str, Any]] = []
    h9_rows: list[dict[str, Any]] = []
    h2_rows: list[dict[str, Any]] = []
    referee: list[dict[str, Any]] = []
    jsd: list[dict[str, Any]] = []
    h16: list[dict[str, Any]] = []
    for seg in [item for item in args.segmentations.split(",") if item]:
        for pair in pairs:
            resolve_dir = args.resolve_root / pair / seg
            summary_path = resolve_dir / f"{pair}_resolve_summary.json"
            if not summary_path.is_file():
                logger.warning("no RESOLVE outputs for %s %s", pair, seg)
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            for platform in PLATFORMS:
                sample = load_resolved_sample(
                    resolve_dir,
                    args.runs_root / pair / seg,
                    args.results_root,
                    summary,
                    pair,
                    seg,
                    platform,
                )
                pseudo = None if args.skip_h9 else pseudo_labellings(sample)
                row, detail = sample_criteria(sample, pseudo)
                sample_rows.append(row)
                h9_rows += detail
                h2_rows += h2_breakdown(sample)
                referee += referee_rows(sample)
                logger.info("%s %s %s scored", pair, seg, platform)
            jsd += jsd_rows(summary, pair, seg)
            h16 += h16_rows(summary, pair, seg)
    args.out.mkdir(parents=True, exist_ok=True)
    h16_frame = pd.DataFrame(h16)
    table: list[dict[str, Any]] = []
    for row in sample_rows:
        table += table_rows_for_sample(row)
    table += referee_table_rows(referee)
    table += jsd_table_rows(jsd)
    if len(h16_frame):
        table += h16_table_rows(h16_frame)
    outputs: dict[str, Any] = {
        "criteria_samples.csv": sample_rows,
        "h9_pseudo.csv": h9_rows,
        "h2_breakdown.csv": h2_rows,
        "referee.csv": referee,
        "criteria_jsd.csv": jsd,
        "criteria_h16.csv": h16,
    }
    if not args.skip_h4:
        config = AnnotationConfig(species="human")
        config = config.model_copy(
            update={
                "panel": config.panel.model_copy(
                    update={"gene_id_fallback_csv": args.gene_id_fallback_csv}
                )
            }
        ).coupled_to_clustering(10)
        segmented = n_segmented_table(args.qc_summary)
        enrichment: list[dict[str, Any]] = []
        for pair in pairs:
            n_segmented = {
                f"{pair}_{platform}": segmented[(pair, "proseg_hybrid", platform)]
                for platform in PLATFORMS
                if (pair, "proseg_hybrid", platform) in segmented
            }
            enrichment += h4_pair(pair, args, config, n_segmented)
        enrichment_frame = pd.DataFrame(enrichment)
        enrichment_frame["auroc_ceiling"] = [
            auroc_ceiling(float(a), float(o))
            for a, o in zip(
                enrichment_frame["detection_assigned"],
                enrichment_frame["detection_other"],
                strict=True,
            )
        ]
        enrichment = enrichment_frame.to_dict("records")
        h4 = h4_summary(enrichment_frame)
        table += h4_table_rows(h4)
        outputs["heldout_enrichment_m4.csv"] = enrichment
        outputs["heldout_h4_m4.csv"] = h4
    outputs["criteria_table.csv"] = table
    for name, rows in outputs.items():
        pd.DataFrame(rows).to_csv(args.out / name, index=False)
        logger.info("wrote %s (%d rows)", args.out / name, len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())

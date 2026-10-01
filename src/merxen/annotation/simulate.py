"""In-silico panel simulation: ``merxen annotation-panel-simulate`` (plan §8.8; M3b).

The design aid for a panel before any of its data exist (plan §8.8: "one
resolvability run for any gene list ... predicting emitted levels and weak
parents"). For a gene list (a vendor panel, a custom panel before ordering):

1. **Panel.** ``panel.panel_from_gene_list`` resolves the declared list (gene
   IDs, controls, the exact-case species test; §8.4) and its family (trust
   inheritance, §8.1).
2. **Production PREP.** Each primary / secondary reference of the species
   (human ``whb_frontal_supc_clus`` and ``seaad_mr_panel``; mouse
   ``wmb_panel``) is built through ``ReferenceStore.get_or_build`` with the
   production configuration, so the bundle is exactly what a pipeline run
   would use (large panels in the large store, with the per-parent marker
   prefilter and kept reference markers, §8.7). PREP runs the resolvability
   self-map (§8.3): held-out reference cells restricted to the panel, thinned
   binomially to the depth grid (only cells whose native counts reach D),
   contaminated with ``R1_contam_HO`` and mapped with the production
   settings, decided with the §8.3 rules (split halves, local isotonic
   thresholds, Wilson bound, point precision, pooled deep bins — the
   resolvability version 3 rules of the H18 follow-up).
3. **Prediction.** The bundle's decisions in the panel's regime (``validated``
   for a panel of a real-data-validated family, else ``provisional``, with the
   provisional margins) give the predicted levels per (level, class, depth
   bin), the trust state (``diagnostics.trust_for_panel``), weak and
   collapsed parents; optionally at an expected depth or on a depth profile.
4. **Prefilter comparison** (panels whose markers were prefiltered, §8.7):
   the unfiltered lookup is built from the self-map engine's marker
   precompute (``run_marker_steps`` on all panel genes; this is the unfiltered
   memory measurement of OD-E8), the same simulated cells (decision recipe,
   seed 0) are mapped with it, and the calls are compared per (level, class):
   agreement at bp >= 0.8 must reach 0.95 for every class with >= 50
   confident calls, and no parent may have fewer than 5 markers (NP9).
5. **Resources.** Wall time per step, peak RSS (the largest process, as the
   evidence measured it, and the summed PSS of each step's process tree),
   and disk (bundle, kept reference markers, test set).

``--gate-p`` is the M13 hook: the gate-P programme (leave-one-donor-out HO
bundles, the second mouse draw, seeds 0 / 1, stress recipes, the NP1-NP9
report) registers itself with ``register_gate_p_hook``; until then the
option is refused before any compute (``GatePUnavailableError``).
"""

from __future__ import annotations

import csv
import dataclasses
import json
import logging
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from merxen.annotation.config import AnnotationConfig, AnnotationReferenceSpec
    from merxen.annotation.panel import AnnotationPanel, PanelComputation
    from merxen.annotation.store import BundleBuilder, ReferenceStore

logger = logging.getLogger(__name__)

SIMULATE_SCHEMA_VERSION: Final = 1
REPORT_JSON: Final = "simulate_report.json"
REPORT_TXT: Final = "SIMULATE_REPORT.txt"
PREDICTED_CSV: Final = "predicted_levels.csv"
PREFILTER_CSV: Final = "prefilter_comparison.csv"
RESOURCES_CSV: Final = "resources.csv"
PANEL_DIR: Final = "panel"
# NP9 / §8.7: prefiltered vs unfiltered lookup agreement.
PREFILTER_MIN_AGREEMENT: Final = 0.95
PREFILTER_MIN_CLASS_N: Final = 50
PREFILTER_CONFIDENT_BP: Final = 0.8
PREFILTER_MIN_PARENT_MARKERS: Final = 5
PrefilterCompare = Literal["auto", "always", "off"]

# The references annotation-panel-simulate builds per species (those with a
# resolvability self-map, plan §8.3).
SELF_MAP_REFERENCES: Final[dict[str, tuple[str, ...]]] = {
    "human": ("whb_frontal_supc_clus", "seaad_mr_panel"),
    "mouse": ("wmb_panel",),
}
# Builder source name <- Nextflow param, per reference; the same map as
# AnnotationReferences.SOURCE_PARAMS (workflows/lib; string-tested).
REFERENCE_SOURCE_PARAMS: Final[dict[str, dict[str, str]]] = {
    "whb_frontal_supc_clus": {
        "region_precompute": "annotation_whb_region_precompute_source",
        "whb_h5ad_dir": "annotation_whb_h5ad_dir",
        "whb_metadata_dir": "annotation_whb_metadata_dir",
        "seaad_precomputed_stats": "annotation_seaad_precomputed_stats_path",
    },
    "seaad_mr_panel": {
        "seaad_precomputed_stats": "annotation_seaad_precomputed_stats_path",
        "seaad_metadata_dir": "annotation_seaad_metadata_dir",
        "whb_region_dir": "annotation_whb_region_precompute_source",
        "whb_h5ad_dir": "annotation_whb_h5ad_dir",
        "whb_metadata_dir": "annotation_whb_metadata_dir",
    },
    "wmb_panel": {
        "wmb_h5ad_dir": "annotation_wmb_h5ad_dir",
        "wmb_metadata_dir": "annotation_wmb_metadata_dir",
        "wmb_mapping_stats": "annotation_wmb_mapping_stats_path",
        "wmb_selfmap_test_cells": "annotation_wmb_selfmap_test_cells_path",
        "wmb_marker_gene_universe": "annotation_wmb_marker_gene_universe_path",
    },
    "wmb_region_share": {
        "merfish_ccf_metadata": "annotation_merfish_ccf_metadata_path",
    },
}


class SimulationError(RuntimeError):
    """A panel simulation cannot run."""


class GatePUnavailableError(SimulationError):
    """``--gate-p`` was asked for but no gate-P programme is registered (M13)."""


@dataclass(frozen=True)
class GatePRequest:
    """What the gate-P programme (M13, ``scripts/acceptance/new_panel.py``) gets.

    Attributes:
        panel: The declared panel.
        config: The annotation config.
        store: The reference store (for the leave-one-donor-out HO bundles).
        bundles: The production bundle directory per reference.
        out_dir: The simulation's output directory.
        report: The base simulation report.
    """

    panel: AnnotationPanel
    config: AnnotationConfig
    store: ReferenceStore
    bundles: dict[str, Path]
    out_dir: Path
    report: dict[str, Any]


GatePHook = Callable[[GatePRequest], dict[str, Any]]
_GATE_P_HOOK: GatePHook | None = None


def register_gate_p_hook(hook: GatePHook | None) -> None:
    """Register the gate-P programme (M13) that ``--gate-p`` runs.

    The hook receives the base simulation (``GatePRequest``) and returns the
    NP1-NP9 record stored under ``gate_p`` in the report. M13 registers it
    from ``scripts/acceptance/new_panel.py``; ``None`` unregisters.

    Args:
        hook: The programme, or ``None``.
    """
    global _GATE_P_HOOK
    _GATE_P_HOOK = hook


def gate_p_hook() -> GatePHook | None:
    """Return the registered gate-P programme, if any."""
    return _GATE_P_HOOK


def require_gate_p_hook() -> GatePHook:
    """Return the gate-P programme or refuse ``--gate-p`` before any compute.

    Raises:
        GatePUnavailableError: If none is registered (until M13).
    """
    hook = gate_p_hook()
    if hook is None:
        raise GatePUnavailableError(
            "--gate-p runs the gate-P programme (plan §8.8, §14: leave-one-donor-"
            "out HO bundles, the second mouse test draw, seeds 0 / 1, stress "
            "recipes, NP1-NP9), which arrives in M13 (scripts/acceptance/"
            "new_panel.py registers it with merxen.annotation.simulate."
            "register_gate_p_hook); run without --gate-p for the design-aid "
            "simulation"
        )
    return hook


def reference_sources(reference_id: str, params: Mapping[str, Path]) -> dict[str, Path]:
    """Return a reference's builder sources from Nextflow-named params.

    Args:
        reference_id: The reference.
        params: Param name (``annotation_*``) to path, for the params set.

    Returns:
        Builder source name to path.
    """
    mapping = REFERENCE_SOURCE_PARAMS.get(reference_id, {})
    return {
        name: Path(params[param])
        for name, param in sorted(mapping.items())
        if param in params
    }


def directory_bytes(path: Path | str) -> int:
    """Return the bytes of the files under a directory (symlinks not followed)."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            file_path = Path(root) / name
            try:
                if not file_path.is_symlink():
                    total += file_path.stat().st_size
            except OSError:
                continue
    return total


def _gb(size_bytes: int | float) -> float:
    return round(float(size_bytes) / 1024**3, 3)


def _read_manifest(bundle_dir: Path) -> dict[str, Any]:
    from merxen.annotation.store import BUNDLE_MANIFEST_NAME

    manifest: dict[str, Any] = json.loads(
        (bundle_dir / BUNDLE_MANIFEST_NAME).read_text(encoding="utf-8")
    )
    return manifest


def _json_native(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_native(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating | float):
        number = float(value)
        return None if not math.isfinite(number) else round(number, 6)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if value is pd.NA:
        return None
    return value


# --------------------------------------------------------------------------
# Predicted levels


def decision_rows(bundle_dir: Path) -> pd.DataFrame:
    """Return the PREP decisions of a bundle (``resolvability.parquet``).

    Args:
        bundle_dir: The bundle.

    Returns:
        The ``kind == "decision"`` rows (every regime), or an empty frame.
    """
    from merxen.annotation.resolvability import RESOLVABILITY_FILE

    path = bundle_dir / RESOLVABILITY_FILE
    if not path.is_file():
        return pd.DataFrame()
    table = pd.read_parquet(path)
    return table[table["kind"] == "decision"].reset_index(drop=True)


PREDICTED_COLUMNS: Final[tuple[str, ...]] = (
    "reference_id",
    "regime",
    "level",
    "class",
    "depth",
    "status",
    "reason",
    "n_test",
    "n_confident",
    "n_effective",
    "precision",
    "wilson_lb",
    "coverage",
    "t_star",
    "target",
    "d_max",
    "extrapolated",
    "pooled",
    "pool_min_depth",
)


def predicted_levels(
    decisions: pd.DataFrame, reference_id: str, regime: str
) -> pd.DataFrame:
    """Return the predicted emission per (level, class, depth) in one regime.

    Args:
        decisions: ``decision_rows`` output.
        reference_id: The reference (recorded).
        regime: ``validated`` or ``provisional``.

    Returns:
        ``PREDICTED_COLUMNS`` rows.
    """
    if decisions.empty:
        return pd.DataFrame(columns=list(PREDICTED_COLUMNS))
    frame = decisions[decisions["regime"] == regime].copy()
    frame["reference_id"] = reference_id
    for column in PREDICTED_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    return frame[list(PREDICTED_COLUMNS)].reset_index(drop=True)


def emission_summary(predicted: pd.DataFrame) -> dict[str, Any]:
    """Return, per level, each class's emitted depths and extrapolated share.

    Args:
        predicted: ``predicted_levels`` output of one reference.

    Returns:
        ``{level: {"classes": {class: {"emitted_depths", "first_depth",
        "n_bins", "n_extrapolated", "reasons"}}, "n_classes",
        "n_classes_emitted"}}``.
    """
    result: dict[str, Any] = {}
    for level, rows in predicted.groupby("level", sort=False, observed=True):
        classes: dict[str, Any] = {}
        for cls, group in rows.groupby("class", sort=True, observed=True):
            emitted = group[group["status"] == "emitted"]
            depths = sorted(int(value) for value in emitted["depth"])
            reasons = sorted(
                {str(value) for value in group["reason"].dropna() if str(value)}
            )
            classes[str(cls)] = {
                "emitted_depths": depths,
                "first_depth": depths[0] if depths else None,
                "n_bins": int(len(group)),
                "n_extrapolated": int(
                    emitted["extrapolated"].fillna(False).astype(bool).sum()
                ),
                "reasons": reasons,
            }
        result[str(level)] = {
            "classes": classes,
            "n_classes": len(classes),
            "n_classes_emitted": sum(
                1 for item in classes.values() if item["emitted_depths"]
            ),
        }
    return result


def depth_profile_weights(
    depths: Sequence[float], grid: Sequence[int]
) -> dict[int, float]:
    """Return the share of cells per depth bin of a depth profile.

    Cells below the first grid depth fall in no bin (they are below every
    emission floor) and are reported as ``0``.

    Args:
        depths: Per-cell panel counts of a (planned or real) dataset.
        grid: The bundle's depth grid.

    Returns:
        Share of cells per grid depth, plus ``0`` for cells below the grid.
    """
    from merxen.annotation.resolvability import depth_bin

    values = np.asarray(list(depths), dtype=np.float64)
    if values.size == 0:
        return {}
    bins = depth_bin(values, list(grid))
    shares = {0: float(np.mean(np.isnan(bins)))}
    for depth in grid:
        shares[int(depth)] = float(np.mean(bins == float(depth)))
    return shares


def read_depth_profile(path: Path | str) -> list[float]:
    """Read a depth profile: a CSV with a ``depth`` (or ``total_counts``) column.

    Args:
        path: The CSV (one row per cell; an optional ``weight`` column
            repeats a row that many times, rounded).

    Returns:
        Per-cell depths.
    """
    table = pd.read_csv(path)
    column = next(
        (name for name in ("depth", "total_counts", "counts") if name in table.columns),
        None,
    )
    if column is None:
        raise SimulationError(f"{path} needs a depth, total_counts or counts column")
    values = table[column].astype(float)
    if "weight" in table.columns:
        repeats = np.maximum(np.round(table["weight"].astype(float)).astype(int), 0)
        return [float(v) for v in np.repeat(values.to_numpy(), repeats.to_numpy())]
    return [float(v) for v in values]


def resolvable_share(
    predicted: pd.DataFrame, shares: Mapping[int, float]
) -> dict[str, dict[str, float]]:
    """Return, per level and class, the profile's share of cells in emitted bins.

    Args:
        predicted: ``predicted_levels`` output of one reference.
        shares: ``depth_profile_weights`` output.

    Returns:
        ``{level: {class: share}}``.
    """
    result: dict[str, dict[str, float]] = {}
    for (level, cls), group in predicted.groupby(
        ["level", "class"], sort=True, observed=True
    ):
        emitted = {int(d) for d in group[group["status"] == "emitted"]["depth"]}
        share = sum(value for depth, value in shares.items() if depth in emitted)
        result.setdefault(str(level), {})[str(cls)] = round(float(share), 4)
    return result


# --------------------------------------------------------------------------
# Prefilter comparison


PREFILTER_COLUMNS: Final[tuple[str, ...]] = (
    "reference_id",
    "level",
    "class",
    "emitted_level",
    "n_unfiltered_confident",
    "agreement_unfiltered_confident",
    "n_either_confident",
    "agreement_either_confident",
    "precision_prefiltered",
    "precision_unfiltered",
    "n_prefiltered_confident",
    "status",
)


def compare_calls(
    prefiltered: pd.DataFrame,
    unfiltered: pd.DataFrame,
    *,
    reference_id: str,
    emitted_levels: Sequence[str],
    recipe: str,
    confident_bp: float = PREFILTER_CONFIDENT_BP,
    min_class_n: int = PREFILTER_MIN_CLASS_N,
    min_agreement: float = PREFILTER_MIN_AGREEMENT,
) -> pd.DataFrame:
    """Compare the prefiltered and unfiltered calls per (level, class).

    The same simulated cells (decision recipe, seed 0) mapped with both
    lookups; a class is judged on the cells confident (bp >= 0.8) under the
    unfiltered lookup: the share whose prefiltered call is the same must be
    at least ``min_agreement`` when there are at least ``min_class_n`` of
    them (plan §8.7, NP9), else it is ``not_evaluable``.

    Args:
        prefiltered: The bundle's ``resolvability_cells`` rows.
        unfiltered: The unfiltered self-map's cells rows.
        reference_id: The reference (recorded).
        emitted_levels: Levels the panel emits (judged); others are reported.
        recipe: The decision recipe.
        confident_bp: Confidence threshold.
        min_class_n: Confident calls a class needs to be judged.
        min_agreement: Required agreement.

    Returns:
        ``PREFILTER_COLUMNS`` rows.
    """
    columns = ["level", "sim_id", "call", "bp", "correct", "truth_parent"]

    def select(frame: pd.DataFrame) -> pd.DataFrame:
        rows = frame[(frame["recipe"] == recipe) & (frame["seed"] == 0)]
        return rows[columns].copy()

    left = select(prefiltered)
    right = select(unfiltered)
    merged = left.merge(
        right, on=["level", "sim_id"], suffixes=("_pre", "_unf"), how="inner"
    )
    records: list[dict[str, Any]] = []
    for (level, cls), group in merged.groupby(
        ["level", "truth_parent_unf"], sort=True, observed=True
    ):
        call_pre = (
            group["call_pre"].astype(object).where(group["call_pre"].notna(), None)
        )
        call_unf = (
            group["call_unf"].astype(object).where(group["call_unf"].notna(), None)
        )
        same = np.array(
            [
                a is not None and b is not None and str(a) == str(b)
                for a, b in zip(call_pre, call_unf, strict=True)
            ],
            dtype=bool,
        )
        conf_unf = group["bp_unf"].to_numpy(np.float64) >= confident_bp
        conf_pre = group["bp_pre"].to_numpy(np.float64) >= confident_bp
        either = conf_unf | conf_pre
        n_unf = int(conf_unf.sum())
        agreement = float(same[conf_unf].mean()) if n_unf else None
        n_either = int(either.sum())
        is_emitted = str(level) in set(emitted_levels)
        if n_unf < min_class_n:
            status = "not_evaluable"
        elif agreement is not None and agreement >= min_agreement:
            status = "pass"
        else:
            status = "fail"
        records.append(
            {
                "reference_id": reference_id,
                "level": str(level),
                "class": str(cls),
                "emitted_level": is_emitted,
                "n_unfiltered_confident": n_unf,
                "agreement_unfiltered_confident": agreement,
                "n_either_confident": n_either,
                "agreement_either_confident": float(same[either].mean())
                if n_either
                else None,
                "precision_prefiltered": float(group["correct_pre"][conf_pre].mean())
                if conf_pre.any()
                else None,
                "precision_unfiltered": float(group["correct_unf"][conf_unf].mean())
                if n_unf
                else None,
                "n_prefiltered_confident": int(conf_pre.sum()),
                "status": status,
            }
        )
    return pd.DataFrame(records, columns=list(PREFILTER_COLUMNS))


def fine_levels_of(bundle_dir: Path) -> set[str]:
    """Return the report-only fine levels of a bundle's self-map (role ``fine``)."""
    from merxen.annotation.resolvability import RESOLVABILITY_SUMMARY_FILE

    path = bundle_dir / RESOLVABILITY_SUMMARY_FILE
    if not path.is_file():
        return set()
    summary = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(item.get("level"))
        for item in summary.get("levels") or []
        if item.get("role") == "fine"
    }


def judged_levels(
    predicted: pd.DataFrame,
    *,
    fine_levels: set[str],
    allow_fine_levels: bool = False,
) -> list[str]:
    """Return the levels a prefilter comparison judges (§8.7: emitted levels).

    A level is judged when the panel emits it for some class and depth; the
    report-only fine levels (WHB cluster, WMB supertype; OD-E4) only when
    ``allow_fine_levels`` turns them on. The others are reported, not judged.

    Args:
        predicted: ``predicted_levels`` output.
        fine_levels: The bundle's fine levels.
        allow_fine_levels: ``thresholds.allow_fine_levels``.

    Returns:
        Sorted level names.
    """
    emitted = {
        str(level) for level in predicted[predicted["status"] == "emitted"]["level"]
    }
    if not allow_fine_levels:
        emitted -= set(fine_levels)
    return sorted(emitted)


def prefilter_verdict(
    comparison: pd.DataFrame,
    *,
    prefiltered_min_markers: int | None,
    unfiltered_min_markers: int | None,
    prefiltered_weak_parents: Sequence[str],
    unfiltered_weak_parents: Sequence[str],
    min_parent_markers: int = PREFILTER_MIN_PARENT_MARKERS,
) -> dict[str, Any]:
    """Return the §8.7 / NP9 verdict of a prefilter comparison.

    Args:
        comparison: ``compare_calls`` output.
        prefiltered_min_markers: Fewest markers of a parent, prefiltered.
        unfiltered_min_markers: The same without the prefilter.
        prefiltered_weak_parents: Parents below ``min_parent_markers``.
        unfiltered_weak_parents: The same without the prefilter.
        min_parent_markers: Markers every parent needs (5).

    Returns:
        ``passes``: the pre-registered rule (plan §8.7, NP9): every judged
        class of an emitted level agrees and no parent of the prefiltered
        lookup is below ``min_parent_markers`` markers
        (``no_parent_below_minimum``); ``no_parent_made_weak`` reports the
        relaxed reading (only parents the unfiltered lookup keeps above the
        minimum count), which is not the criterion (changing it needs the
        user's approval, §14); the failing and unevaluable classes and the
        parents made weak by the prefilter.
    """
    judged = comparison[comparison["emitted_level"].astype(bool)]
    failing = judged[judged["status"] == "fail"]
    new_weak = sorted(set(prefiltered_weak_parents) - set(unfiltered_weak_parents))
    # Pre-registered (§8.7, NP9): no parent below the minimum at all. A parent
    # the panel already leaves weak without the prefilter fails it too; the
    # relaxed reading (no parent made weak by the prefilter) is reported only
    # (M3b review 2).
    relaxed_parents_ok = not new_weak
    strict_parents_ok = not prefiltered_weak_parents and (
        prefiltered_min_markers is None or prefiltered_min_markers >= min_parent_markers
    )
    return {
        "passes": bool(failing.empty and strict_parents_ok),
        "agreement_passes": bool(failing.empty),
        "parents_pass": bool(strict_parents_ok),
        "no_parent_below_minimum": bool(strict_parents_ok),
        "no_parent_made_weak": bool(relaxed_parents_ok),
        "passes_relaxed_parent_reading": bool(failing.empty and relaxed_parents_ok),
        "n_judged": int((judged["status"] != "not_evaluable").sum()),
        "n_pass": int((judged["status"] == "pass").sum()),
        "n_fail": int(len(failing)),
        "n_not_evaluable": int((judged["status"] == "not_evaluable").sum()),
        "failing": [
            {
                "level": row["level"],
                "class": row["class"],
                "agreement": row["agreement_unfiltered_confident"],
                "n": int(row["n_unfiltered_confident"]),
            }
            for _index, row in failing.iterrows()
        ],
        "min_agreement_judged": None
        if judged[judged["status"] != "not_evaluable"].empty
        else float(
            judged[judged["status"] != "not_evaluable"][
                "agreement_unfiltered_confident"
            ].min()
        ),
        "prefiltered_markers_per_parent_min": prefiltered_min_markers,
        "unfiltered_markers_per_parent_min": unfiltered_min_markers,
        "n_prefiltered_weak_parents": len(prefiltered_weak_parents),
        "n_unfiltered_weak_parents": len(unfiltered_weak_parents),
        "parents_weak_only_with_prefilter": new_weak,
        "rule": (
            f"agreement >= {PREFILTER_MIN_AGREEMENT} at bp >= {PREFILTER_CONFIDENT_BP} "
            f"for every class with >= {PREFILTER_MIN_CLASS_N} unfiltered confident "
            f"calls at every emitted level; no parent below {min_parent_markers} "
            "markers (plan §8.7, NP9)"
        ),
    }


# --------------------------------------------------------------------------
# One reference


@dataclass
class ReferenceSimulation:
    """Everything simulated for one reference.

    Attributes:
        reference_id: The reference.
        record: The report record.
        predicted: ``predicted_levels`` rows.
        prefilter: ``compare_calls`` rows (empty when not compared).
        resources: Resource rows (step, wall, peaks, disk).
        bundle_dir: The production bundle, if built.
    """

    reference_id: str
    record: dict[str, Any]
    predicted: pd.DataFrame
    prefilter: pd.DataFrame
    resources: list[dict[str, Any]] = field(default_factory=list)
    bundle_dir: Path | None = None


def _step_rows(
    reference_id: str, phase: str, builder_output: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ctm = builder_output.get("ctm_steps") or {}
    for step, seconds in (builder_output.get("timings_s") or {}).items():
        metrics = ctm.get(step) or {}
        rows.append(
            {
                "reference_id": reference_id,
                "phase": phase,
                "step": step,
                "wall_s": seconds,
                "peak_rss_gb": metrics.get("peak_rss_gb"),
                "peak_tree_rss_gb": metrics.get("peak_tree_rss_gb"),
                "peak_tree_pss_gb": metrics.get("peak_tree_pss_gb"),
                "n_processors": None,
                "disk_gb": None,
            }
        )
    return rows


def _mapping_rows(
    reference_id: str, phase: str, runs: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    return [
        {
            "reference_id": reference_id,
            "phase": phase,
            "step": f"map_{run.get('tag')}",
            "wall_s": run.get("wall_s"),
            "peak_rss_gb": run.get("peak_rss_gb"),
            "peak_tree_rss_gb": None,
            "peak_tree_pss_gb": None,
            "n_processors": run.get("n_processors"),
            "disk_gb": None,
        }
        for run in runs
    ]


def panel_regime(state: str | None, validation_basis: str | None) -> str:
    """Return the resolvability regime a panel's cells are emitted in (§8.2).

    ``validated`` only for a real-data-validated family; a family validated
    by simulation keeps the provisional rule and margins, as does every
    other panel.
    """
    if state == "validated" and validation_basis == "real_data":
        return "validated"
    return "provisional"


def simulate_reference(
    *,
    panel: AnnotationPanel,
    computation: PanelComputation,
    spec: AnnotationReferenceSpec,
    builder: BundleBuilder,
    config: AnnotationConfig,
    store: ReferenceStore,
    out_dir: Path,
    scratch_dir: Path,
    prefilter_compare: PrefilterCompare = "auto",
    expected_depth: int | None = None,
    depth_profile: Sequence[float] | None = None,
) -> ReferenceSimulation:
    """Build one reference on a panel and predict what it resolves.

    Args:
        panel: The declared panel.
        computation: ``panel_from_gene_list`` output (its report).
        spec: The prepared reference spec (sources completed).
        builder: Its builder.
        config: The annotation config (production settings).
        store: The reference store.
        out_dir: The reference's output directory.
        scratch_dir: Scratch for the unfiltered markers and mappings.
        prefilter_compare: ``auto`` (when the markers were prefiltered),
            ``always`` or ``off``.
        expected_depth: Median panel counts of the planned data, if known.
        depth_profile: Per-cell panel counts of a planned dataset, if known.

    Returns:
        The simulation of the reference.
    """
    from merxen.annotation.diagnostics import (
        TrustRules,
        load_validated_panels,
        panel_diagnostics,
        trust_for_panel,
    )
    from merxen.annotation.memory import ProcessTreeSampler
    from merxen.annotation.store import large_panel_refusal

    reference_id = spec.reference_id
    out_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {"reference_id": reference_id, "role": spec.role}
    refusal = large_panel_refusal(panel, config, builder)
    if refusal is not None:
        record.update({"status": "build_refused", "reason": refusal})
        return ReferenceSimulation(
            reference_id,
            record,
            pd.DataFrame(columns=list(PREDICTED_COLUMNS)),
            pd.DataFrame(columns=list(PREFILTER_COLUMNS)),
        )
    started = time.monotonic()
    with ProcessTreeSampler() as sampler:
        bundle_ref = store.get_or_build(spec, panel, builder=builder, config=config)
    build_wall = time.monotonic() - started
    build_peak = sampler.peak()
    bundle_dir = Path(bundle_ref.path)
    manifest = _read_manifest(bundle_dir)
    output = manifest.get("builder_output") or {}
    resolvability = output.get("resolvability") or {}
    test_set_ref = resolvability.get("test_set_bundle") or {}
    test_set_dir = _test_set_dir(bundle_dir)
    validated = load_validated_panels(config.panel.validated_panels_path)
    diagnostics = panel_diagnostics(
        panel, panel_report=computation.report, bundles=[manifest]
    )
    trust = trust_for_panel(
        diagnostics,
        reference_id=reference_id,
        role=spec.role,
        validated=validated,
        rules=TrustRules.from_config(config),
    )
    regime = panel_regime(trust.state, trust.validation_basis)
    decisions = decision_rows(bundle_dir)
    predicted = predicted_levels(decisions, reference_id, regime)
    markers = output.get("markers") or {}
    record.update(
        {
            "status": "built",
            "reused": bool(bundle_ref.reused),
            "build_hash": bundle_ref.build_hash,
            "bundle_dir": str(bundle_dir),
            "store_root": bundle_ref.store_root,
            "trust": {
                "state": trust.state,
                "family_id": trust.family_id,
                "family_basis": trust.family_basis,
                "validation_basis": trust.validation_basis,
                "reasons": [reason.model_dump(mode="json") for reason in trust.reasons],
                "notes": [note.model_dump(mode="json") for note in trust.notes],
            },
            "regime": regime,
            "emission": emission_summary(predicted),
            "resolvability_trust": resolvability.get("trust"),
            "test_set": {
                "bundle": test_set_ref,
                "n_test_cells": resolvability.get("n_test_cells"),
                "n_simulated_cells": resolvability.get("n_simulated_cells"),
            },
            "markers": {
                key: markers.get(key)
                for key in (
                    "n_panel_genes",
                    "n_candidate_genes",
                    "prefilter",
                    "n_marker_genes",
                    "markers_per_parent_min",
                    "markers_per_parent_median",
                    "weak_parents",
                    "n_collapsed_parents",
                    "collapsed_parents",
                    "reference_markers",
                )
                if key in markers
            },
            "panel_coverage": output.get("panel_coverage"),
            "depth_grid": output.get("depth_grid"),
        }
    )
    if expected_depth is not None and output.get("depth_grid"):
        from merxen.annotation.resolvability import depth_bin

        grid = [int(value) for value in output["depth_grid"]]
        bin_value = float(depth_bin(np.asarray([float(expected_depth)]), grid)[0])
        at_depth: dict[str, list[str]] = {}
        expected_bin = None if math.isnan(bin_value) else int(bin_value)
        if expected_bin is not None:
            rows = predicted[
                (predicted["depth"] == expected_bin)
                & (predicted["status"] == "emitted")
            ]
            for level, group in rows.groupby("level", sort=False, observed=True):
                at_depth[str(level)] = sorted(str(cls) for cls in group["class"])
        record["expected_depth"] = {
            "median_counts": int(expected_depth),
            "depth_bin": expected_bin,
            "emitted_classes": at_depth,
        }
    if depth_profile is not None and output.get("depth_grid"):
        shares = depth_profile_weights(depth_profile, output["depth_grid"])
        record["depth_profile"] = {
            "n_cells": len(depth_profile),
            "bin_shares": shares,
            "resolvable_share": resolvable_share(predicted, shares),
        }
    # Resources and disk.
    reference_markers_bytes = sum(
        int(item.get("size", 0))
        for item in markers.get("reference_markers") or []
        if item.get("kept")
    )
    disk = {
        "bundle_gb": _gb(directory_bytes(bundle_dir)),
        "kept_reference_markers_gb": _gb(reference_markers_bytes),
        "reference_markers_gb": _gb(
            sum(
                int(item.get("size", 0))
                for item in markers.get("reference_markers") or []
            )
        ),
        "test_set_bundle_gb": None
        if test_set_dir is None or not test_set_dir.is_dir()
        else _gb(directory_bytes(test_set_dir)),
    }
    record["resources"] = {
        "build_wall_s": round(build_wall, 1),
        "bundle_wall_time_s": manifest.get("wall_time_s"),
        "timings_s": output.get("timings_s"),
        "ctm_peak_rss_gb": output.get("ctm_peak_rss_gb"),
        "ctm_peak_tree_pss_gb": output.get("ctm_peak_tree_pss_gb"),
        "build_process_tree_peak": build_peak.to_json(),
        "self_map_runtime_s": (resolvability.get("runtime_s") or {}),
        "mapping_runs": resolvability.get("mapping_runs"),
        "disk": disk,
    }
    resources = _step_rows(reference_id, "prep", output)
    resources += _mapping_rows(
        reference_id, "prep_self_map", resolvability.get("mapping_runs") or []
    )
    if test_set_dir is not None and test_set_dir.is_dir():
        test_manifest = _read_manifest(test_set_dir)
        test_output = test_manifest.get("builder_output") or {}
        resources += _step_rows(
            str(test_set_ref.get("reference_id")), "prep_test_set", test_output
        )
        record["resources"]["test_set_timings_s"] = test_output.get("timings_s")
        record["resources"]["test_set_ctm_peak_rss_gb"] = test_output.get(
            "ctm_peak_rss_gb"
        )
    resources.append(
        {
            "reference_id": reference_id,
            "phase": "prep",
            "step": "bundle_disk",
            "wall_s": None,
            "peak_rss_gb": None,
            "peak_tree_rss_gb": None,
            "peak_tree_pss_gb": None,
            "n_processors": None,
            "disk_gb": disk["bundle_gb"],
        }
    )
    comparison = pd.DataFrame(columns=list(PREFILTER_COLUMNS))
    engine_prefiltered = _engine_prefilter(bundle_dir, test_set_dir, resolvability)
    should_compare = prefilter_compare == "always" or (
        prefilter_compare == "auto" and engine_prefiltered
    )
    if should_compare and decisions.empty:
        record["prefilter_comparison"] = {"status": "skipped", "reason": "no self-map"}
    elif should_compare:
        emitted_levels = judged_levels(
            predicted,
            fine_levels=fine_levels_of(bundle_dir),
            allow_fine_levels=bool(config.thresholds.allow_fine_levels),
        )
        comparison, verdict, extra_rows = compare_prefilter(
            reference_id=reference_id,
            spec=spec,
            panel=panel,
            config=config,
            bundle_dir=bundle_dir,
            test_set_dir=test_set_dir,
            resolvability=resolvability,
            emitted_levels=emitted_levels,
            out_dir=out_dir,
            scratch_dir=scratch_dir / reference_id,
            regime=regime,
            predicted=predicted,
        )
        record["prefilter_comparison"] = verdict
        resources += extra_rows
    else:
        record["prefilter_comparison"] = {
            "status": "not_run",
            "reason": "markers not prefiltered"
            if not engine_prefiltered
            else f"--prefilter-compare {prefilter_compare}",
        }
    return ReferenceSimulation(
        reference_id=reference_id,
        record=record,
        predicted=predicted,
        prefilter=comparison,
        resources=resources,
        bundle_dir=bundle_dir,
    )


def _test_set_dir(bundle_dir: Path) -> Path | None:
    """Return the self-map test-set bundle of a bundle (its summary's path)."""
    from merxen.annotation.resolvability import RESOLVABILITY_SUMMARY_FILE

    path = bundle_dir / RESOLVABILITY_SUMMARY_FILE
    if not path.is_file():
        return None
    summary = json.loads(path.read_text(encoding="utf-8"))
    location = (summary.get("test_set_bundle") or {}).get("path")
    return None if not location else Path(location)


def _engine_dir(
    bundle_dir: Path, test_set_dir: Path | None, resolvability: Mapping[str, Any]
) -> Path:
    """Return the bundle the self-map mapped onto (itself, or the HO bundle)."""
    from merxen.annotation.resolvability import load_resolvability

    tables = load_resolvability(bundle_dir)
    engine = (tables.summary.get("engine") if tables is not None else None) or {}
    if engine.get("self", True) or test_set_dir is None:
        return bundle_dir
    return test_set_dir


def _engine_prefilter(
    bundle_dir: Path, test_set_dir: Path | None, resolvability: Mapping[str, Any]
) -> bool:
    if not resolvability:
        return False
    engine_dir = _engine_dir(bundle_dir, test_set_dir, resolvability)
    output = _read_manifest(engine_dir).get("builder_output") or {}
    return bool((output.get("markers") or {}).get("prefilter"))


def compare_prefilter(
    *,
    reference_id: str,
    spec: AnnotationReferenceSpec,
    panel: AnnotationPanel,
    config: AnnotationConfig,
    bundle_dir: Path,
    test_set_dir: Path | None,
    resolvability: Mapping[str, Any],
    emitted_levels: Sequence[str],
    out_dir: Path,
    scratch_dir: Path,
    regime: str,
    predicted: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    """Map the self-map cells with the unfiltered lookup and compare (§8.7).

    The unfiltered lookup is found on the self-map engine's marker precompute
    with every panel gene it holds (``run_marker_steps``, no prefilter; the
    reference markers go to scratch and are deleted after); the decision
    recipe's simulated cells (seed 0, the same thinning and contamination)
    are mapped with it and compared with the bundle's prefiltered calls.

    Returns:
        The per-(level, class) comparison, the verdict record and resource
        rows (the unfiltered marker steps are the unfiltered memory
        measurement of OD-E8).
    """
    from merxen.annotation import reference as ref
    from merxen.annotation import resolvability as res
    from merxen.annotation.mapmycells_engine import MmcBundle
    from merxen.annotation.memory import ProcessTreeSampler
    from merxen.annotation.vocab import load_floor_table

    if test_set_dir is None or not test_set_dir.is_dir():
        raise SimulationError(
            f"{reference_id}: the self-map test set bundle is missing"
        )
    engine_dir = _engine_dir(bundle_dir, test_set_dir, resolvability)
    engine = MmcBundle.from_dir(engine_dir)
    engine_output = _read_manifest(engine_dir).get("builder_output") or {}
    marker_precompute = engine_dir / ref.MARKER_PRECOMPUTE_FILE
    if not marker_precompute.is_file():
        marker_precompute = engine_dir / ref.MAPPING_PRECOMPUTE_FILE
    available = set(ref.precompute_genes(marker_precompute))
    genes = sorted(gene for gene in panel.ensembl_ids if gene in available)
    tree = engine.tree()
    scratch_dir.mkdir(parents=True, exist_ok=True)
    timer = ref._StepTimer(out_dir / "logs")
    raw_path = scratch_dir / "unfiltered_query_markers.json"
    started = time.monotonic()
    with ProcessTreeSampler() as sampler:
        reference_paths = ref.run_marker_steps(
            marker_precompute=marker_precompute,
            candidates=genes,
            drop_level=engine.drop_level,
            n_per_utility=spec.n_per_utility,
            scratch_dir=scratch_dir,
            reference_dir=scratch_dir / "unfiltered_reference_markers",
            raw_lookup_path=raw_path,
            timer=timer,
            step_prefix="unfiltered_",
        )
    markers_wall = time.monotonic() - started
    unfiltered_reference_bytes = sum(path.stat().st_size for path in reference_paths)
    raw = ref.read_lookup(raw_path)
    validation = ref.validate_lookup(raw, tree, genes)
    lookup_path = out_dir / "unfiltered_query_markers.filtered.json"
    lookup_path.write_text(
        json.dumps(validation.lookup, indent=2) + "\n", encoding="utf-8"
    )
    for path in reference_paths:
        path.unlink(missing_ok=True)
    unfiltered_engine = dataclasses.replace(
        engine, lookup=lookup_path, lookup_sha256=ref.lookup_sha256(validation.lookup)
    )
    # The same test cells the bundle's self-map simulated (M8 D1: the human
    # held-out set without its other-region COP cells).
    test, _ = ref.self_map_test_cells(
        res.load_test_cells(test_set_dir),
        str((resolvability.get("test_set_bundle") or {}).get("reference_id") or ""),
    )
    runs: list[dict[str, Any]] = []
    grid = spec.resolved_depth_grid(panel.n_genes)
    recipes = res.simulation_recipes(config.resolvability, seed=ref.TEST_SET_SEED)[:1]
    started = time.monotonic()
    result = res.run_resolvability(
        test,
        specs=ref.level_specs_for(reference_id, unfiltered_engine, config),
        depths=grid,
        recipes=recipes,
        map_fn=ref.mmc_map_function(
            unfiltered_engine,
            spec=spec,
            config=config,
            scratch_dir=scratch_dir / "mapping",
            log_dir=out_dir / "logs" / "unfiltered_mapping",
            runs=runs,
        ),
        settings=ref.self_map_rule_settings(config),
        species=spec.species,
        floor_table=load_floor_table(spec.species),
        cells_rules=ref.cells_rules_for(reference_id, config),
    )
    mapping_wall = time.monotonic() - started
    prefiltered_cells = pd.read_parquet(bundle_dir / res.RESOLVABILITY_CELLS_FILE)
    comparison = compare_calls(
        prefiltered_cells,
        result.cells,
        reference_id=reference_id,
        emitted_levels=emitted_levels,
        recipe=recipes[0].name,
    )
    unfiltered_predicted = predicted_levels(result.decisions, reference_id, regime)
    unfiltered_predicted.to_csv(
        out_dir / "predicted_levels_unfiltered.csv", index=False
    )
    merged = predicted.merge(
        unfiltered_predicted,
        on=["level", "class", "depth"],
        suffixes=("", "_unfiltered"),
        how="outer",
    )
    differ = merged[merged["status"] != merged["status_unfiltered"]]
    engine_markers = engine_output.get("markers") or {}
    unfiltered_summary = validation.summary(config.panel.weak_parent_markers)
    verdict = prefilter_verdict(
        comparison,
        prefiltered_min_markers=engine_markers.get("markers_per_parent_min"),
        unfiltered_min_markers=unfiltered_summary.get("markers_per_parent_min"),
        prefiltered_weak_parents=[
            str(item) for item in engine_markers.get("weak_parents") or []
        ],
        unfiltered_weak_parents=[
            str(item) for item in unfiltered_summary.get("weak_parents") or []
        ],
    )
    verdict.update(
        {
            "status": "run",
            "engine": {
                "reference_id": engine.reference_id,
                "build_hash": engine.build_hash,
                "marker_precompute": marker_precompute.name,
            },
            "prefilter": engine_markers.get("prefilter"),
            "n_unfiltered_candidate_genes": len(genes),
            "n_prefiltered_candidate_genes": engine_markers.get("n_candidate_genes"),
            "emission_bins_differing": int(len(differ)),
            "emission_bins_compared": int(len(merged)),
            "emitted_bins_prefiltered": int((predicted["status"] == "emitted").sum()),
            "emitted_bins_unfiltered": int(
                (unfiltered_predicted["status"] == "emitted").sum()
            ),
            "unfiltered_lookup": str(lookup_path),
            "unfiltered_markers": {
                "wall_s": round(markers_wall, 1),
                "ctm_steps": timer.to_json()["ctm_steps"],
                "process_tree_peak": sampler.peak().to_json(),
                "reference_markers_gb": _gb(unfiltered_reference_bytes),
            },
            "unfiltered_mapping": {
                "wall_s": round(mapping_wall, 1),
                "runs": runs,
            },
        }
    )
    rows: list[dict[str, Any]] = []
    for step, metrics in timer.to_json()["ctm_steps"].items():
        rows.append(
            {
                "reference_id": reference_id,
                "phase": "prefilter_comparison",
                "step": step,
                "wall_s": metrics.get("wall_s"),
                "peak_rss_gb": metrics.get("peak_rss_gb"),
                "peak_tree_rss_gb": metrics.get("peak_tree_rss_gb"),
                "peak_tree_pss_gb": metrics.get("peak_tree_pss_gb"),
                "n_processors": None,
                "disk_gb": _gb(unfiltered_reference_bytes)
                if step.endswith("reference_markers")
                else None,
            }
        )
    rows += _mapping_rows(reference_id, "prefilter_comparison", runs)
    return comparison, verdict, rows


# --------------------------------------------------------------------------
# The simulation


@dataclass(frozen=True)
class ReferenceBuild:
    """A reference to build: its prepared spec and builder."""

    spec: AnnotationReferenceSpec
    builder: BundleBuilder


def run_panel_simulation(
    *,
    gene_list: Path,
    species: str,
    name: str,
    config: AnnotationConfig,
    store: ReferenceStore,
    builds: Callable[[AnnotationPanel], Sequence[ReferenceBuild]],
    out_dir: Path,
    scratch_dir: Path,
    platform: str | None = None,
    prefilter_compare: PrefilterCompare = "auto",
    expected_depth: int | None = None,
    depth_profile: Path | None = None,
    gate_p: bool = False,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Simulate a candidate panel end to end and write its report.

    Args:
        gene_list: The gene list (any ``read_panel_file`` format).
        species: ``human`` or ``mouse``.
        name: Report name (e.g. ``xenium_prime_5k_mouse``).
        config: The annotation config (production settings).
        store: The reference store (large panels go to its large root).
        builds: ``panel -> references to build`` (specs prepared).
        out_dir: Output directory.
        scratch_dir: Scratch (unfiltered markers, mappings).
        platform: The panel's platform, if known.
        prefilter_compare: See ``simulate_reference``.
        expected_depth: Median panel counts of the planned data.
        depth_profile: CSV of per-cell panel counts of a planned dataset.
        gate_p: Run the registered gate-P programme (M13) afterwards.
        provenance: Extra report fields (inputs, code version).

    Returns:
        The report (also written to ``simulate_report.json``).

    Raises:
        GatePUnavailableError: If ``gate_p`` and no programme is registered
            (checked before any compute).
    """
    from merxen.annotation.memory import ProcessTreeSampler
    from merxen.annotation.panel import panel_from_gene_list

    hook = require_gate_p_hook() if gate_p else None
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    sampler = ProcessTreeSampler().start()
    profile = read_depth_profile(depth_profile) if depth_profile is not None else None
    computation = panel_from_gene_list(
        gene_list,
        species,  # type: ignore[arg-type]
        output_dir=out_dir / PANEL_DIR,
        platform=platform,
        config=config,
    )
    panel = computation.panels["gene_list"].panel
    declared = (computation.report.get("declared_panels") or {}).get("gene_list") or {}
    report: dict[str, Any] = {
        "schema_version": SIMULATE_SCHEMA_VERSION,
        "name": name,
        "species": species,
        "platform": platform,
        "gene_list": str(gene_list),
        "started_at": started_at,
        "panel": {
            "status": computation.required.status,
            "reasons": list(computation.required.reasons),
            "panel_hash": panel.panel_hash,
            "n_genes": panel.n_genes,
            "large_panel": panel.n_genes > config.panel.large_panel_genes,
            "family": None
            if panel.panel_family is None
            else panel.panel_family.model_dump(mode="json"),
            "gene_ids": {
                key: declared.get(key)
                for key in (
                    "status",
                    "refusal_reasons",
                    "n_features_in",
                    "n_controls_removed",
                    "controls_removed",
                    "n_non_control",
                    "n_genes",
                    "gene_id_resolution",
                    "gene_id_resolution_share",
                    "unresolved",
                    "merged_duplicates",
                    "species_check",
                    "native_prefix_share",
                    "symbols_as_ids",
                    "release_drift",
                )
                if key in declared
            },
            "panel_files": {
                key: str(value) for key, value in computation.paths.items()
            },
        },
        "settings": {
            "prefilter_compare": prefilter_compare,
            "expected_depth": expected_depth,
            "depth_profile": None if depth_profile is None else str(depth_profile),
            "large_panel_genes": config.panel.large_panel_genes,
            "large_panel_marker_prefilter": config.panel.large_panel_marker_prefilter,
            "large_panel_prefilter_cap": config.panel.large_panel_prefilter_cap,
            "resolvability": config.resolvability.model_dump(mode="json"),
            "gate_p": gate_p,
        },
        "provenance": dict(provenance or {}),
        "references": {},
    }
    predicted_frames: list[pd.DataFrame] = []
    prefilter_frames: list[pd.DataFrame] = []
    resource_rows: list[dict[str, Any]] = []
    bundles: dict[str, Path] = {}
    if computation.required.status == "refused":
        report["status"] = "panel_refused"
    else:
        for item in builds(panel):
            logger.info(
                "annotation-panel-simulate %s: %s on %d genes",
                name,
                item.spec.reference_id,
                panel.n_genes,
            )
            simulation = simulate_reference(
                panel=panel,
                computation=computation,
                spec=item.spec,
                builder=item.builder,
                config=config,
                store=store,
                out_dir=out_dir / item.spec.reference_id,
                scratch_dir=scratch_dir,
                prefilter_compare=prefilter_compare,
                expected_depth=expected_depth,
                depth_profile=profile,
            )
            report["references"][simulation.reference_id] = simulation.record
            predicted_frames.append(simulation.predicted)
            prefilter_frames.append(simulation.prefilter)
            resource_rows += simulation.resources
            if simulation.bundle_dir is not None:
                bundles[simulation.reference_id] = simulation.bundle_dir
        report["status"] = "done"
    peak = sampler.stop()
    report["resources"] = {
        "wall_s": round(time.monotonic() - started, 1),
        "process_tree_peak": peak.to_json(),
        "n_processors": _n_processors(),
    }
    report["finished_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    if hook is not None:
        report["gate_p"] = hook(
            GatePRequest(
                panel=panel,
                config=config,
                store=store,
                bundles=bundles,
                out_dir=out_dir,
                report=report,
            )
        )
    write_report(
        out_dir,
        report,
        predicted=_concat(predicted_frames, PREDICTED_COLUMNS),
        prefilter=_concat(prefilter_frames, PREFILTER_COLUMNS),
        resources=resource_rows,
    )
    return report


def _n_processors() -> int:
    from merxen.annotation.reference import prep_resources

    return prep_resources().n_processors


def _concat(frames: Sequence[pd.DataFrame], columns: Sequence[str]) -> pd.DataFrame:
    kept = [frame for frame in frames if not frame.empty]
    if not kept:
        return pd.DataFrame(columns=list(columns))
    return pd.concat(kept, ignore_index=True)


RESOURCE_COLUMNS: Final[tuple[str, ...]] = (
    "reference_id",
    "phase",
    "step",
    "wall_s",
    "peak_rss_gb",
    "peak_tree_rss_gb",
    "peak_tree_pss_gb",
    "n_processors",
    "disk_gb",
)


def write_report(
    out_dir: Path,
    report: Mapping[str, Any],
    *,
    predicted: pd.DataFrame,
    prefilter: pd.DataFrame,
    resources: Sequence[Mapping[str, Any]],
) -> dict[str, Path]:
    """Write the JSON, text and CSV outputs of a simulation.

    Args:
        out_dir: Output directory.
        report: The report.
        predicted: Predicted levels of every reference.
        prefilter: Prefilter comparisons of every reference.
        resources: Resource rows.

    Returns:
        Path per output.
    """
    paths = {
        "json": out_dir / REPORT_JSON,
        "text": out_dir / REPORT_TXT,
        "predicted": out_dir / PREDICTED_CSV,
        "prefilter": out_dir / PREFILTER_CSV,
        "resources": out_dir / RESOURCES_CSV,
    }
    paths["json"].write_text(
        json.dumps(_json_native(report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    predicted.to_csv(paths["predicted"], index=False)
    prefilter.to_csv(paths["prefilter"], index=False)
    with paths["resources"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(RESOURCE_COLUMNS))
        writer.writeheader()
        for row in resources:
            writer.writerow({key: row.get(key) for key in RESOURCE_COLUMNS})
    paths["text"].write_text(render_text(report), encoding="utf-8")
    return paths


def _depth_ranges(depths: Sequence[int]) -> str:
    return "-" if not depths else f"{min(depths)}-{max(depths)} ({len(depths)} bins)"


def render_text(report: Mapping[str, Any]) -> str:
    """Return the plain-text report of a simulation."""
    panel = report.get("panel") or {}
    lines = [
        f"annotation-panel-simulate: {report.get('name')} ({report.get('species')}, "
        f"{report.get('platform') or 'platform unknown'})",
        f"gene list: {report.get('gene_list')}",
        f"panel: {panel.get('n_genes')} resolved genes, hash "
        f"{str(panel.get('panel_hash'))[:16]}, status {panel.get('status')}"
        + (
            f" ({'; '.join(panel.get('reasons') or [])})"
            if panel.get("reasons")
            else ""
        ),
        f"large panel (> {report.get('settings', {}).get('large_panel_genes')} genes): "
        f"{panel.get('large_panel')}",
        f"family: {(panel.get('family') or {}).get('family_id')} "
        f"({(panel.get('family') or {}).get('basis')})",
        f"started {report.get('started_at')}, finished {report.get('finished_at')}, "
        f"wall {report.get('resources', {}).get('wall_s')} s, process-tree peak "
        f"{report.get('resources', {}).get('process_tree_peak')}",
        "",
    ]
    for reference_id, record in (report.get("references") or {}).items():
        lines.append(f"== {reference_id}: {record.get('status')}")
        if record.get("status") != "built":
            lines.append(f"   {record.get('reason')}")
            lines.append("")
            continue
        trust = record.get("trust") or {}
        lines.append(
            f"   bundle {str(record.get('build_hash'))[:16]} "
            f"({'reused' if record.get('reused') else 'built'}) in "
            f"{record.get('store_root')}"
        )
        lines.append(
            f"   trust: {trust.get('state')} (family {trust.get('family_id')}, "
            f"{trust.get('family_basis')}); regime {record.get('regime')}"
        )
        markers = record.get("markers") or {}
        prefilter = markers.get("prefilter")
        lines.append(
            f"   markers: {markers.get('n_marker_genes')} marker genes from "
            f"{markers.get('n_candidate_genes')} candidates; per parent min "
            f"{markers.get('markers_per_parent_min')} / median "
            f"{markers.get('markers_per_parent_median')}; weak parents "
            f"{len(markers.get('weak_parents') or [])}; collapsed "
            f"{len(markers.get('collapsed_parents') or [])}"
        )
        if prefilter:
            lines.append(
                f"   prefilter: {prefilter.get('method')} v{prefilter.get('version')} "
                f"k={prefilter.get('k')} -> {prefilter.get('n_genes')} of "
                f"{prefilter.get('n_input_genes')} genes"
            )
        coverage = record.get("panel_coverage") or {}
        lines.append(
            f"   coverage: {coverage.get('n_query_genes_used')} panel genes in the "
            f"reference; root markers {coverage.get('root_markers')}; root children "
            f"separated {coverage.get('root_children_separated')} of "
            f"{coverage.get('root_children')}"
        )
        lines.append(f"   predicted emission ({record.get('regime')} regime):")
        for level, entry in (record.get("emission") or {}).items():
            lines.append(
                f"     {level}: {entry.get('n_classes_emitted')} of "
                f"{entry.get('n_classes')} classes emitted at some depth"
            )
            for cls, item in (entry.get("classes") or {}).items():
                depths = item.get("emitted_depths") or []
                text = f"       {cls:<28} {_depth_ranges(depths)}"
                if item.get("n_extrapolated"):
                    text += f"; extrapolated {item['n_extrapolated']}"
                reasons = ", ".join(item.get("reasons") or [])
                if reasons and not depths:
                    text += f"; not emitted: {reasons}"
                lines.append(text)
        if record.get("expected_depth"):
            item = record["expected_depth"]
            lines.append(
                f"   at the expected median depth {item.get('median_counts')} (bin "
                f"{item.get('depth_bin')}): "
                + "; ".join(
                    f"{level} {len(classes)} classes"
                    for level, classes in (item.get("emitted_classes") or {}).items()
                )
            )
        comparison = record.get("prefilter_comparison") or {}
        if comparison.get("status") == "run":
            verdict = "PASS" if comparison.get("passes") else "FAIL"
            lines.append(
                f"   prefilter comparison: {verdict} "
                f"({comparison.get('n_pass')} pass, {comparison.get('n_fail')} fail, "
                f"{comparison.get('n_not_evaluable')} not evaluable; min agreement "
                f"{comparison.get('min_agreement_judged')}; self-map engine markers "
                "per parent min "
                f"{comparison.get('prefiltered_markers_per_parent_min')} prefiltered / "
                f"{comparison.get('unfiltered_markers_per_parent_min')} unfiltered, "
                "parents weak only with the prefilter "
                f"{len(comparison.get('parents_weak_only_with_prefilter') or [])}; "
                "emission bins differing "
                f"{comparison.get('emission_bins_differing')} of "
                f"{comparison.get('emission_bins_compared')})"
            )
            for item in comparison.get("failing") or []:
                lines.append(
                    f"     fail: {item['level']} / {item['class']}: agreement "
                    f"{item['agreement']:.3f} on {item['n']} calls"
                )
            unfiltered = comparison.get("unfiltered_markers") or {}
            for step, metrics in (unfiltered.get("ctm_steps") or {}).items():
                lines.append(
                    f"     {step}: {metrics.get('wall_s')} s, peak RSS "
                    f"{metrics.get('peak_rss_gb')} GB (largest process), tree PSS "
                    f"{metrics.get('peak_tree_pss_gb')} GB"
                )
        else:
            lines.append(
                f"   prefilter comparison: {comparison.get('status')} "
                f"({comparison.get('reason')})"
            )
        resources = record.get("resources") or {}
        lines.append(
            f"   resources: build {resources.get('build_wall_s')} s; ctm peak RSS "
            f"{resources.get('ctm_peak_rss_gb')} GB (largest process), tree PSS "
            f"{resources.get('ctm_peak_tree_pss_gb')} GB; disk {resources.get('disk')}"
        )
        for step, seconds in (resources.get("timings_s") or {}).items():
            lines.append(f"     {step}: {seconds} s")
        lines.append("")
    if report.get("gate_p") is not None:
        lines.append(f"gate P: {json.dumps(_json_native(report['gate_p']))[:2000]}")
    return "\n".join(lines) + "\n"

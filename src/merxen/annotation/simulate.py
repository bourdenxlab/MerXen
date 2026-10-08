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

``--gate-p`` is the M13 hook: the gate-P programme (``gate_p.run_gate_p``:
leave-one-donor-out HO bundles, seeds 0 / 1, stress recipes, the NP1-NP9
report; the second mouse draw follows M6b) is registered with
``register_gate_p_hook`` by the ``annotation-panel-simulate --gate-p`` command
and ``scripts/acceptance/new_panel.py``; without a registered programme the
option is refused before any compute (``GatePUnavailableError``). The
programme's precheck (``GatePPlan``: the species, config, store, output and
depth settings) runs before any compute; the base report is written before the
programme runs, and a programme that raises is recorded in it.
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

from merxen.annotation import diagnostics as _diagnostics_notes

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
    """What the gate-P programme (M13, ``gate_p.run_gate_p``) gets.

    Attributes:
        panel: The declared panel.
        config: The annotation config.
        store: The reference store (for the leave-one-donor-out HO bundles).
        bundles: The production bundle directory per reference.
        out_dir: The simulation's output directory.
        report: The base simulation report.
        builds: The prepared specs and builders the base simulation built
            (the held-out test set's spec and the identity re-run of PREP
            come from them).
        scratch_dir: The simulation's scratch directory (outside the stores).
    """

    panel: AnnotationPanel
    config: AnnotationConfig
    store: ReferenceStore
    bundles: dict[str, Path]
    out_dir: Path
    report: dict[str, Any]
    builds: tuple[ReferenceBuild, ...] = ()
    scratch_dir: Path | None = None


@dataclass(frozen=True)
class GatePPlan:
    """What the gate-P programme can check before any compute (M13).

    ``run_panel_simulation`` passes it to the registered precheck before the
    panel is computed or any bundle is built, so a refusal that needs no
    bundle never follows hours of PREP.

    Attributes:
        species: The species the simulation was started for.
        config: The annotation config.
        store: The reference store the simulation builds in.
        out_dir: The simulation's output directory.
        scratch_dir: The simulation's scratch directory.
        expected_depth: ``--expected-depth``.
        depth_profile: ``--depth-profile``.
        depth_profile_asset: ``--depth-profile-asset``.
    """

    species: str
    config: AnnotationConfig
    store: ReferenceStore
    out_dir: Path
    scratch_dir: Path
    expected_depth: int | None = None
    depth_profile: Path | None = None
    depth_profile_asset: str | None = None


GatePHook = Callable[[GatePRequest], dict[str, Any]]
GatePPrecheck = Callable[[GatePPlan], None]
# The status of the report's ``gate_p`` record when the programme raised.
GATE_P_STATUS_REFUSED: Final = "refused"
GATE_P_STATUS_FAILED: Final = "failed"
_GATE_P_HOOK: GatePHook | None = None
_GATE_P_PRECHECK: GatePPrecheck | None = None


def register_gate_p_hook(
    hook: GatePHook | None, *, precheck: GatePPrecheck | None = None
) -> None:
    """Register the gate-P programme (M13) that ``--gate-p`` runs.

    The hook receives the base simulation (``GatePRequest``) and returns the
    NP1-NP9 record stored under ``gate_p`` in the report. The precheck, if
    any, receives a ``GatePPlan`` before any compute and raises to refuse the
    run there. M13 registers both from ``annotation-panel-simulate --gate-p``
    (and so from ``scripts/acceptance/new_panel.py``); ``None`` unregisters
    both.

    Args:
        hook: The programme, or ``None``.
        precheck: Its checks that need no compute, or ``None``.
    """
    global _GATE_P_HOOK, _GATE_P_PRECHECK
    _GATE_P_HOOK = hook
    _GATE_P_PRECHECK = precheck if hook is not None else None


def gate_p_hook() -> GatePHook | None:
    """Return the registered gate-P programme, if any."""
    return _GATE_P_HOOK


def gate_p_precheck() -> GatePPrecheck | None:
    """Return the registered gate-P precheck, if any."""
    return _GATE_P_PRECHECK


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
    member: str | None = None,
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
        member: For version-7 cells, the ensemble member compared
            (``R1_contam_HO@0``; the R1 members share recipe and seed).

    Returns:
        ``PREFILTER_COLUMNS`` rows.
    """
    columns = ["level", "sim_id", "call", "bp", "correct", "truth_parent"]

    def select(frame: pd.DataFrame) -> pd.DataFrame:
        rows = frame[(frame["recipe"] == recipe) & (frame["seed"] == 0)]
        if member is not None and "member" in rows.columns:
            rows = rows[rows["member"].astype(str) == member]
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
    depth_profile: Sequence[float] | Any | None = None,
    profile_mode: bool = False,
    real_composition: pd.DataFrame | None = None,
    profile_members: Sequence[str] | None = None,
    chemistry: Any | None = None,
    v7_diagnostic: bool = False,
    v7_fresh_seeds: Sequence[int] | None = None,
    v7_fresh_r3_seeds: Sequence[int] | None = None,
    v7_comparator: bool = False,
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
        expected_depth: Median panel counts of the planned data, if known
            (a secondary line; the depth profile is the headline, M3c).
        depth_profile: A ``sim_inputs.DepthProfile`` (per-class or pooled
            totals) or per-cell panel counts of a planned dataset, if known.
        profile_mode: Also map simulated cells drawn at the profile's
            per-class depth (plan §8.3 v7.5; primary references only).
        real_composition: Real called (class, subclass) cell counts to weigh
            profile-mode cells to (mouse).
        profile_members: Profile-mode members to run (default: the family's
            emission members).
        chemistry: The panel's ``sim_inputs.ChemistryResolution``.
        v7_diagnostic: For a version-6 bundle, compute the version-7
            decisions beside it (``run_v7_diagnostic``; never applied).
        v7_fresh_seeds: R1 seeds of a fresh ensemble B whose churn against
            the bundle's (or the diagnostic's) ensemble is reported.
        v7_fresh_r3_seeds: R3 seeds of ensemble B (default ``(1,)``, stage
            D's B).
        v7_comparator: Run the pre-registered comparator of the amended
            re-test of §21 (iii) as ensemble B (pre-registration §22.4).

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
    profile_resources: list[dict[str, Any]] = []
    if depth_profile is not None and output.get("depth_grid"):
        grid = [int(value) for value in output["depth_grid"]]
        profile: Any = depth_profile if hasattr(depth_profile, "by_class") else None
        totals: Any = profile.totals if profile is not None else depth_profile
        shares = depth_profile_weights(totals, grid)
        record["depth_profile"] = {
            "n_cells": len(totals),
            "bin_shares": shares,
            "resolvable_share": resolvable_share(predicted, shares),
        }
        if profile is not None:
            record["depth_profile"]["profile"] = profile.to_json()
            class_depth = class_depth_predictions(predicted, profile, grid)
            record["class_depth"] = {
                "regime": regime,
                "headline": class_depth_headline(class_depth, profile),
                "file": CLASS_DEPTH_CSV,
            }
            if not class_depth.empty:
                class_depth.insert(0, "reference_id", reference_id)
                class_depth.to_csv(out_dir / CLASS_DEPTH_CSV, index=False)
            if profile_mode and spec.role == "primary" and not decisions.empty:
                record["profile_mode"], profile_resources = run_profile_mode(
                    reference_id=reference_id,
                    spec=spec,
                    config=config,
                    bundle_dir=bundle_dir,
                    test_set_dir=test_set_dir,
                    resolvability=resolvability,
                    profile=profile,
                    chemistry=chemistry,
                    out_dir=out_dir,
                    scratch_dir=scratch_dir / reference_id / "profile_mode",
                    real_composition=real_composition,
                    members=profile_members,
                )
            elif profile_mode:
                record["profile_mode"] = {
                    "status": "not_run",
                    "reason": "no self-map decisions"
                    if decisions.empty
                    else f"{spec.role} reference (profile mode maps the primary)",
                }
    diagnostic_resources: list[dict[str, Any]] = []
    stored_version = (resolvability or {}).get("resolvability_version")
    wants_diagnostic = (
        bool(v7_fresh_seeds) or v7_comparator or (v7_diagnostic and stored_version != 7)
    )
    if wants_diagnostic and spec.role in ("primary", "secondary"):
        if decisions.empty:
            record["v7_diagnostic"] = {"status": "not_run", "reason": "no self-map"}
        else:
            record["v7_diagnostic"], diagnostic_resources = run_v7_diagnostic(
                reference_id=reference_id,
                spec=spec,
                panel=panel,
                config=config,
                bundle_dir=bundle_dir,
                test_set_dir=test_set_dir,
                resolvability=resolvability,
                chemistry=chemistry,
                out_dir=out_dir,
                scratch_dir=scratch_dir / reference_id / "v7_diagnostic",
                store_roots=store.roots,
                fresh_seeds=v7_fresh_seeds,
                fresh_r3_seeds=v7_fresh_r3_seeds,
                comparator=v7_comparator,
            )
    if chemistry is not None:
        record["panel_card_notes"] = panel_card_notes(
            str(panel.species), str(chemistry.chemistry)
        )
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
    resources += profile_resources
    resources += diagnostic_resources
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

    tables = load_resolvability(bundle_dir, allow_version_7=True)
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
    map_fn = ref.mmc_map_function(
        unfiltered_engine,
        spec=spec,
        config=config,
        scratch_dir=scratch_dir / "mapping",
        log_dir=out_dir / "logs" / "unfiltered_mapping",
        runs=runs,
    )
    specs = ref.level_specs_for(reference_id, unfiltered_engine, config)
    rules = ref.cells_rules_for(reference_id, config)
    stored = res.load_resolvability(bundle_dir, allow_version_7=True)
    member: str | None = None
    started = time.monotonic()
    if stored is not None and stored.version == res.RESOLVABILITY_VERSION_V7:
        # Version 7: re-simulate the first R1 member with its conventions on
        # the bundle's grid, compared member to member.
        first = res.EnsembleMember(
            res.member_recipe(res.DECISION_RECIPE, 0, config.resolvability),
            "emission",
        )
        member = first.name
        unfiltered_cells = res.simulate_members(
            test,
            specs=specs,
            depths=stored.depth_grid,
            members=[first],
            map_fn=map_fn,
            cells_rules=rules,
        ).cells
        unfiltered_decisions = res.decide(
            unfiltered_cells,
            stored.levels,
            stored.depth_grid,
            stored.settings,
            saturated_bp_share=res.EnsembleSettings.from_json(
                stored.summary.get("ensemble_settings")
            ).saturated_bp_share,
        )
    else:
        result = res.run_resolvability(
            test,
            specs=specs,
            depths=grid,
            recipes=recipes,
            map_fn=map_fn,
            settings=ref.self_map_rule_settings(config),
            species=spec.species,
            floor_table=load_floor_table(spec.species),
            cells_rules=rules,
        )
        unfiltered_cells, unfiltered_decisions = result.cells, result.decisions
    mapping_wall = time.monotonic() - started
    prefiltered_cells = pd.read_parquet(bundle_dir / res.RESOLVABILITY_CELLS_FILE)
    comparison = compare_calls(
        prefiltered_cells,
        unfiltered_cells,
        reference_id=reference_id,
        emitted_levels=emitted_levels,
        recipe=recipes[0].name,
        member=member,
    )
    if member is not None:
        # The member's own decisions on both lookups (the ensemble needs
        # every member).
        table = pd.read_parquet(bundle_dir / res.RESOLVABILITY_FILE)
        own = table[
            (table["kind"].astype(str) == "member_decision")
            & (table["member"].astype(str) == member)
        ]
        predicted = predicted_levels(own, reference_id, regime)
    unfiltered_predicted = predicted_levels(unfiltered_decisions, reference_id, regime)
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
    depth_profile_asset: str | None = None,
    profile_mode: bool = False,
    real_composition: Path | None = None,
    profile_members: Sequence[str] | None = None,
    gate_p: bool = False,
    provenance: Mapping[str, Any] | None = None,
    v7_diagnostic: bool = False,
    v7_fresh_seeds: Sequence[int] | None = None,
    v7_fresh_r3_seeds: Sequence[int] | None = None,
    v7_comparator: bool = False,
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
        expected_depth: Median panel counts of the planned data (reported as
            a secondary line).
        depth_profile: CSV of per-cell panel counts of a planned dataset:
            per-class (a ``class`` column, the ``5k_real/mouse5k/
            depth_profile.csv`` format) or pooled.
        depth_profile_asset: A registered ``profile`` or ``scenario``
            simulation-input asset instead of a CSV (``sim_inputs``).
        profile_mode: Map simulated cells drawn at the profile's per-class
            depth (plan §8.3 v7.5); needs a depth profile.
        real_composition: CSV of real called (class, subclass, n_cells) to
            weigh profile-mode cells to (mouse).
        profile_members: Profile-mode members (default: the family's
            emission members).
        gate_p: Run the registered gate-P programme (M13) afterwards: its
            precheck (``GatePPlan``) before any compute, the programme after
            the base report is written. A programme that raises leaves its
            refusal (``refused``) or failure (``failed``) under ``gate_p``
            in the written report, and the error is raised again.
        provenance: Extra report fields (inputs, code version).
        v7_diagnostic: ``--resolvability-version 7``: version-7 decisions of
            the version-6 families as a diagnostic (``run_v7_diagnostic``).
        v7_fresh_seeds: R1 seeds of a fresh ensemble B (churn, §21 (iii)).
        v7_fresh_r3_seeds: R3 seeds of ensemble B (default ``(1,)``).
        v7_comparator: Ensemble B = the pre-registered comparator of the
            amended re-test (pre-registration §22.4).

    Returns:
        The report (also written to ``simulate_report.json``).

    Raises:
        GatePUnavailableError: If ``gate_p`` and no programme is registered
            (checked before any compute).
        SimulationError: If the registered gate-P precheck refuses the run
            (before any compute), or the programme refuses it after the
            base simulation (recorded in the report first).
        SimulationError: If the depth profile is of another species or both
            a CSV and an asset are given (before any compute).
    """
    from merxen.annotation import sim_inputs as si
    from merxen.annotation.memory import ProcessTreeSampler
    from merxen.annotation.panel import panel_from_gene_list

    hook = require_gate_p_hook() if gate_p else None
    profile = load_simulation_profile(
        depth_profile, depth_profile_asset, species=species
    )
    if profile_mode and profile is None:
        raise SimulationError(
            "profile mode needs --depth-profile or --depth-profile-asset"
        )
    precheck = gate_p_precheck() if hook is not None else None
    if precheck is not None:
        precheck(
            GatePPlan(
                species=species,
                config=config,
                store=store,
                out_dir=out_dir,
                scratch_dir=scratch_dir,
                expected_depth=expected_depth,
                depth_profile=depth_profile,
                depth_profile_asset=depth_profile_asset,
            )
        )
    composition = (
        read_real_composition(real_composition)
        if real_composition is not None
        else None
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    sampler = ProcessTreeSampler().start()
    computation = panel_from_gene_list(
        gene_list,
        species,  # type: ignore[arg-type]
        output_dir=out_dir / PANEL_DIR,
        platform=platform,
        config=config,
    )
    panel = computation.panels["gene_list"].panel
    declared = (computation.report.get("declared_panels") or {}).get("gene_list") or {}
    chemistry = si.resolve_chemistry(
        panel.ensembl_ids,
        species=species,
        platform=platform,
        declared=str(getattr(config.panel, "panel_chemistry", "auto")),
    )
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
            "chemistry": chemistry.to_json(),
        },
        "settings": {
            "prefilter_compare": prefilter_compare,
            "expected_depth": expected_depth,
            "depth_profile": None if depth_profile is None else str(depth_profile),
            "depth_profile_asset": depth_profile_asset,
            "profile_mode": profile_mode,
            "real_composition": None
            if real_composition is None
            else str(real_composition),
            "profile_members": None
            if profile_members is None
            else list(profile_members),
            "large_panel_genes": config.panel.large_panel_genes,
            "large_panel_marker_prefilter": config.panel.large_panel_marker_prefilter,
            "large_panel_prefilter_cap": config.panel.large_panel_prefilter_cap,
            "resolvability": config.resolvability.model_dump(mode="json"),
            "gate_p": gate_p,
            "v7_diagnostic": v7_diagnostic,
            "v7_fresh_seeds": None if not v7_fresh_seeds else list(v7_fresh_seeds),
            "v7_fresh_r3_seeds": None
            if v7_fresh_r3_seeds is None
            else list(v7_fresh_r3_seeds),
            "v7_comparator": bool(v7_comparator),
        },
        "provenance": dict(provenance or {}),
        "references": {},
    }
    predicted_frames: list[pd.DataFrame] = []
    prefilter_frames: list[pd.DataFrame] = []
    resource_rows: list[dict[str, Any]] = []
    bundles: dict[str, Path] = {}
    built: list[ReferenceBuild] = []
    if computation.required.status == "refused":
        report["status"] = "panel_refused"
    else:
        built = list(builds(panel))
        for item in built:
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
                profile_mode=profile_mode,
                real_composition=composition,
                profile_members=profile_members,
                chemistry=chemistry,
                v7_diagnostic=v7_diagnostic,
                v7_fresh_seeds=v7_fresh_seeds,
                v7_fresh_r3_seeds=v7_fresh_r3_seeds,
                v7_comparator=v7_comparator,
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
    predicted = _concat(predicted_frames, PREDICTED_COLUMNS)
    prefilter = _concat(prefilter_frames, PREFILTER_COLUMNS)
    if hook is not None:
        # The base simulation's report is written before gate P runs, so a
        # gate-P refusal or failure after PREP never loses it; the refusal is
        # then recorded in it and raised.
        write_report(
            out_dir,
            report,
            predicted=predicted,
            prefilter=prefilter,
            resources=resource_rows,
        )
        try:
            report["gate_p"] = hook(
                GatePRequest(
                    panel=panel,
                    config=config,
                    store=store,
                    bundles=bundles,
                    out_dir=out_dir,
                    report=report,
                    builds=tuple(built),
                    scratch_dir=scratch_dir,
                )
            )
        except Exception as error:
            report["gate_p"] = {
                "status": GATE_P_STATUS_REFUSED
                if isinstance(error, SimulationError)
                else GATE_P_STATUS_FAILED,
                "error": type(error).__name__,
                "reason": str(error),
            }
            write_report(
                out_dir,
                report,
                predicted=predicted,
                prefilter=prefilter,
                resources=resource_rows,
            )
            raise
    write_report(
        out_dir,
        report,
        predicted=predicted,
        prefilter=prefilter,
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
        lines += render_profile_lines(record)
        if record.get("expected_depth"):
            item = record["expected_depth"]
            lines.append(
                "   secondary (one depth for every cell; read the per-bin table "
                "against the panel's per-class depth): at the expected median depth "
                f"{item.get('median_counts')} (bin {item.get('depth_bin')}): "
                + "; ".join(
                    f"{level} {len(classes)} classes"
                    for level, classes in (item.get("emitted_classes") or {}).items()
                )
            )
        for note in record.get("panel_card_notes") or []:
            lines.append(f"   note: {note}")
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


# --------------------------------------------------------------------------
# Profile mode and per-class depth (resolvability version 7, plan §8.3 v7.5;
# M3c). Phase 1 found one expected depth the largest error of the M3b
# headline (-.13 to -.22 coverage on the public 5K section): real cells carry
# a per-class depth (glia about a third of neuronal depth), so predictions
# use the per-class profile and the "expected median depth" line is kept
# only as a secondary line.

# Profile mode (phase 1's D-recipes, ``5k_real/sim/scripts/simlib.py``
# ``simulate_totals``): per truth class c, M_c = min(max(2 n_c, 2000), 8 n_c)
# simulated cells.
PROFILE_MIN_SIMS: Final = 2000
PROFILE_SIMS_PER_CELL: Final = 2
PROFILE_MAX_SIMS_PER_CELL: Final = 8
PROFILE_SPILL_FRACTION: Final = 0.25
# Rows thinned per block (exact thinning is per row, so blocks never change a
# simulated cell; they bound the memory of 50k-cell profile queries).
PROFILE_THIN_BLOCK_ROWS: Final = 20_000
# The raw rules profile mode reports (plan §7.3 mouse; §5.2 human defaults).
MOUSE_CLASS_RULE_BP: Final = 0.9
MOUSE_CLASS_RULE_MIN_COUNTS: Final = 20
MOUSE_SUBCLASS_RULE_BP: Final = 0.8
MOUSE_SUBCLASS_RULE_MIN_COUNTS: Final = 50
HUMAN_RAW_MIN_COUNTS: Final = 20
# Composition weights (phase 1's ``analyse.py`` ``comp_weights``): class
# weight x within-class subclass weight, trimmed at 10x the class median.
COMPOSITION_TRIM: Final = 10.0
PROFILE_PREDICTIONS_CSV: Final = "profile_predictions.csv"
CLASS_DEPTH_CSV: Final = "profile_class_depth.csv"
PROFILE_CELLS_FILE: Final = "profile_cells.parquet"
# Panel-card notes of the Xenium Prime 5K families (user decision 4, plan
# §8.10) live in ``diagnostics``; they state limits, never change a prediction.
GLIAL_UPPER_BOUND_NOTE: Final = _diagnostics_notes.GLIAL_UPPER_BOUND_NOTE
PRECISION_UNMEASURED_NOTE: Final = _diagnostics_notes.PRECISION_UNMEASURED_NOTE
MOUSE_PROFILE_LEVELS: Final[tuple[str, ...]] = ("broad", "class", "nt", "subclass")


def profile_sims_per_class(n_cells: int) -> int:
    """Return the simulated cells of a truth class with ``n_cells`` test cells."""
    return int(
        min(
            max(PROFILE_SIMS_PER_CELL * n_cells, PROFILE_MIN_SIMS),
            PROFILE_MAX_SIMS_PER_CELL * n_cells,
        )
    )


def choose_profile_partners(
    native: np.ndarray,
    groups: np.ndarray,
    host_groups: np.ndarray,
    amounts: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw each host's spill partner uniformly among eligible cells.

    Phase 1's ``_choose_partners``: per host spill group (sorted), a uniform
    draw among the cells of another group whose native counts reach the
    spill amount, from one generator.

    Raises:
        SimulationError: If a host has no eligible partner.
    """
    partners = np.empty(len(amounts), dtype=np.int64)
    for group in np.unique(host_groups):
        is_host = np.flatnonzero(host_groups == group)
        candidates = np.flatnonzero(groups != group)
        order = candidates[np.argsort(-native[candidates], kind="stable")]
        descending = native[order]
        reach = np.searchsorted(-descending, -amounts[is_host], side="right")
        if (reach == 0).any():
            raise SimulationError(f"no spill donor for group {group}")
        pick = np.floor(rng.random(len(is_host)) * reach).astype(np.int64)
        partners[is_host] = order[pick]
    return partners


def _thin_blocks(
    counts: Any,
    rows: np.ndarray,
    targets: np.ndarray,
    efficiency: np.ndarray,
    keys: Sequence[int],
    block_rows: int,
) -> Any:
    from scipy import sparse as sp

    from merxen.annotation.resolvability import thin_rows_exact

    parts = []
    for start in range(0, len(rows), max(1, int(block_rows))):
        stop = min(len(rows), start + int(block_rows))
        parts.append(
            thin_rows_exact(
                counts[rows[start:stop]],
                targets[start:stop],
                efficiency,
                list(keys[start:stop]),
            )
        )
    if not parts:
        return sp.csr_matrix((0, counts.shape[1]), dtype=np.float64)
    return sp.vstack(parts).tocsr()


def simulate_on_profile(
    test: Any,
    profile: Any,
    truth_class: np.ndarray,
    recipe: Any,
    grid: Sequence[int],
    *,
    efficiency: np.ndarray | None = None,
    key_name: str | None = None,
    spill_fraction: float | None = None,
    sims_per_class: Callable[[int], int] = profile_sims_per_class,
    block_rows: int = PROFILE_THIN_BLOCK_ROWS,
) -> Any:
    """Simulate test cells at real per-class TOTAL depths (profile mode, v7.5).

    The port of phase 1's D-recipe (``simlib.simulate_totals``, profile
    branch), with its draw structure: per truth class ``c`` (sorted), ``M_c``
    totals are drawn from ``c``'s profile (``DepthProfile.sample``; the pooled
    neuronal or non-neuronal profile below 100 cells) with the generator of
    ``draw_key(seed, name, "depth", c)``; each total's host is drawn
    uniformly among ``c``'s test cells whose native counts reach ``total /
    (1 + s)`` (the class's deepest cell, marked truncated, when none does);
    the host is thinned to ``total / (1 + s)`` and ``s`` times that is spilled
    from a uniform partner of another spill group whose native counts reach
    it (one generator, ``draw_key(seed, name, "partner")``); both with exact
    thinning keyed per simulated cell (``draw_key(seed, name, version,
    sim_id, stream)``). A simulated cell's bin is the grid value at or below
    its realised total. Profile mode never enters emission.

    Args:
        test: ``HeldOutCells``.
        profile: ``sim_inputs.DepthProfile``.
        truth_class: Truth class per test cell (mouse WMB class, human broad
            class), in the test cells' order.
        recipe: The member's ``SimulationRecipe`` (seed, name, version, spill).
        grid: The bundle's depth grid (for the bins).
        efficiency: Per-gene efficiency (default: ``member_efficiency``).
        key_name: The name in the draw keys (default: the recipe's; the D3
            regression passes ``R3_measured_realdepth``).
        spill_fraction: Spill fraction (default: the recipe's, else 0.25).
        sims_per_class: ``n_c -> M_c``.
        block_rows: Rows thinned per block.

    Returns:
        A ``SimulatedQuery`` (obs: ``cell_id``, ``depth`` (bin), ``partner_id``,
        ``host_counts``, ``spill_counts``, ``total_counts``, ``target_total``,
        ``host_depth``, ``rep``, ``truncated``, ``truth_class_sampled``,
        ``native_counts``, ``profile_source``, ``member``).

    Raises:
        SimulationError: If the truth classes do not match the test cells.
    """
    from merxen.annotation import resolvability as res

    classes = np.asarray([str(value) for value in truth_class], dtype=object)
    if len(classes) != len(test.obs):
        raise SimulationError(
            f"profile mode: {len(classes)} truth classes for {len(test.obs)} test cells"
        )
    name = str(key_name or recipe.name)
    seed = int(recipe.seed)
    spill = float(
        spill_fraction
        if spill_fraction is not None
        else (
            recipe.spill_fraction if recipe.spill_fraction else PROFILE_SPILL_FRACTION
        )
    )
    gene_eff = (
        res.member_efficiency(recipe, test.genes)
        if efficiency is None
        else np.asarray(efficiency, dtype=np.float64)
    )
    native = test.native_counts
    groups = test.obs[res.SPILL_GROUP_COLUMN].astype(str).to_numpy()
    cell_ids = test.obs.index.astype(str).to_numpy()
    from scipy import sparse as sp

    counts = sp.csr_matrix(test.counts)
    host_parts: list[np.ndarray] = []
    total_parts: list[np.ndarray] = []
    rep_parts: list[np.ndarray] = []
    trunc_parts: list[np.ndarray] = []
    source_parts: list[np.ndarray] = []
    for cls in sorted(set(classes.tolist())):
        members = np.flatnonzero(classes == cls)
        n_sims = int(sims_per_class(len(members)))
        rng = np.random.default_rng(res.draw_key(seed, name, "depth", cls))
        totals = profile.sample(cls, n_sims, rng).astype(np.float64)
        hosts_depth = totals / (1.0 + spill)
        order = members[np.argsort(-native[members], kind="stable")]
        descending = native[order]
        reach = np.searchsorted(-descending, -hosts_depth, side="right")
        truncated = reach == 0
        reach = np.maximum(reach, 1)
        picked = order[np.floor(rng.random(n_sims) * reach).astype(np.int64)]
        hosts_depth = np.where(truncated, native[picked], hosts_depth)
        totals = np.where(truncated, hosts_depth * (1.0 + spill), totals)
        host_parts.append(picked)
        total_parts.append(totals)
        rep_parts.append(np.arange(n_sims))
        trunc_parts.append(truncated)
        source_parts.append(np.full(n_sims, profile.used.get(cls, ""), dtype=object))
    hosts = np.concatenate(host_parts) if host_parts else np.zeros(0, dtype=np.int64)
    target_total = np.concatenate(total_parts) if total_parts else np.zeros(0)
    reps = np.concatenate(rep_parts) if rep_parts else np.zeros(0, dtype=np.int64)
    truncated_all = (
        np.concatenate(trunc_parts) if trunc_parts else np.zeros(0, dtype=bool)
    )
    sources = (
        np.concatenate(source_parts) if source_parts else np.zeros(0, dtype=object)
    )
    host_depth = target_total / (1.0 + spill)
    sim_ids = np.array(
        [
            f"{cell_ids[host]}|{name}|{int(rep)}"
            for host, rep in zip(hosts, reps, strict=True)
        ],
        dtype=object,
    )
    version = int(recipe.version)
    thin_keys = [res.draw_key(seed, name, version, sim, "thin") for sim in sim_ids]
    host_counts = _thin_blocks(
        counts, hosts, host_depth, gene_eff, thin_keys, block_rows
    )
    rng = np.random.default_rng(res.draw_key(seed, name, "partner"))
    amounts = spill * host_depth
    partners = (
        choose_profile_partners(native, groups, groups[hosts], amounts, rng)
        if len(hosts)
        else np.zeros(0, dtype=np.int64)
    )
    spill_keys = [res.draw_key(seed, name, version, sim, "spill") for sim in sim_ids]
    spill_counts = _thin_blocks(
        counts, partners, amounts, gene_eff, spill_keys, block_rows
    )
    simulated = (host_counts + spill_counts).tocsr()
    total = np.asarray(simulated.sum(axis=1)).ravel()
    bins = res.depth_bin(total, list(grid))
    obs = pd.DataFrame(
        {
            "cell_id": cell_ids[hosts],
            "depth": np.nan_to_num(bins, nan=0).astype(np.int32),
            "partner_id": cell_ids[partners].astype(object),
            "host_counts": np.asarray(host_counts.sum(axis=1)).ravel(),
            "spill_counts": np.asarray(spill_counts.sum(axis=1)).ravel(),
            "total_counts": total,
            "target_total": target_total,
            "host_depth": host_depth,
            "rep": reps,
            "truncated": truncated_all,
            "truth_class_sampled": classes[hosts],
            "native_counts": native[hosts],
            "profile_source": sources,
            "member": recipe.member,
        },
        index=pd.Index(sim_ids, name="sim_id"),
    )
    return res.SimulatedQuery(
        recipe=recipe, counts=simulated, genes=list(test.genes), obs=obs, n_by_depth={}
    )


def profile_truth_classes(
    test: Any, specs: Sequence[Any], species: str, vocab: pd.DataFrame | None
) -> np.ndarray:
    """Return the truth class of each test cell that profile mode samples by.

    Mouse: the WMB class name of ``truth__CCN20230722_CLAS``; human: the
    broad level's truth parent (``Exc``, ``Inh``, ``Astro``, ...), as phase 1.
    """
    from merxen.annotation import resolvability as res

    if species == "mouse":
        names = _node_names(vocab)
        values = test.obs[f"{res.TRUTH_PREFIX}{res.WMB_CLAS}"].astype(str)
        return np.array([names.get(value, value) for value in values], dtype=object)
    broad = next((spec for spec in specs if spec.meta.level == "broad"), None)
    if broad is None:
        raise SimulationError("profile mode needs the broad level's truth")
    truth = broad.truth(test)
    return np.asarray(truth["truth_parent"].astype(object).to_numpy(), dtype=object)


def _node_names(vocab: pd.DataFrame | None) -> dict[str, str]:
    if vocab is None:
        return {}
    frame = vocab.reset_index(drop=True)
    return dict(
        zip(frame["node"].astype(str), frame["node_name"].astype(str), strict=True)
    )


def _subclass_class_map(vocab: pd.DataFrame | None) -> dict[str, str]:
    from merxen.annotation import resolvability as res

    if vocab is None:
        return {}
    frame = vocab.reset_index(drop=True)
    rows = frame[frame["level"].astype(str) == res.WMB_SUBC]
    return dict(
        zip(rows["node_name"].astype(str), rows["key_name"].astype(str), strict=True)
    )


def emission_frame(
    summary: Mapping[str, Any], regime: str = "provisional"
) -> pd.DataFrame:
    """Return a bundle's emitted bins: level, class, bin, threshold, floor.

    From ``resolvability_summary.json`` (``emission`` and ``floors`` of the
    regime), as phase 1's ``analyse.py`` read it.
    """
    rows = []
    emission = (summary.get("emission") or {}).get(regime) or {}
    floors = (summary.get("floors") or {}).get(regime) or {}
    for level, classes in emission.items():
        for cls, bins in classes.items():
            floor = ((floors.get(level) or {}).get(cls) or {}).get("floor")
            for depth, record in bins.items():
                if not record or record.get("status") != "emitted":
                    continue
                threshold = record.get("threshold")
                if threshold is None:
                    threshold = record.get("t_star")
                if threshold is None:
                    continue
                rows.append(
                    {
                        "level": str(level),
                        "cls": str(cls),
                        "bin": float(depth),
                        "thr": float(threshold),
                        "floor": 0.0 if floor is None else float(floor),
                    }
                )
    return pd.DataFrame(rows, columns=["level", "cls", "bin", "thr", "floor"])


def provisional_ok(
    level: str,
    cls: np.ndarray,
    bins: np.ndarray,
    bp: np.ndarray,
    counts: np.ndarray,
    emitted: pd.DataFrame,
) -> np.ndarray:
    """Return which cells a bundle's decisions emit at a level.

    A cell is emitted when its (level, called class, bin of its total counts)
    is an emitted bin, its bp reaches the bin's applied threshold and its
    counts reach the class's floor.
    """
    rows = emitted[emitted["level"] == level]
    key = pd.MultiIndex.from_arrays(
        [rows["cls"].astype(str), rows["bin"].astype(float)]
    )
    threshold = pd.Series(rows["thr"].to_numpy(dtype=np.float64), index=key)
    floor = pd.Series(rows["floor"].to_numpy(dtype=np.float64), index=key)
    query = pd.MultiIndex.from_arrays(
        [pd.Series(cls).astype(str), pd.Series(bins).astype(float)]
    )
    applied = threshold.reindex(query).to_numpy(dtype=np.float64)
    floors = floor.reindex(query).to_numpy(dtype=np.float64)
    ok = (
        ~np.isnan(applied)
        & (bp >= np.nan_to_num(applied, nan=9.0))
        & (counts >= np.nan_to_num(floors, nan=np.inf))
    )
    return np.asarray(ok, dtype=bool)


def profile_cell_table(
    cells: pd.DataFrame,
    *,
    species: str,
    summary: Mapping[str, Any],
    names: Mapping[str, str],
    regime: str = "provisional",
) -> pd.DataFrame:
    """Return one row per simulated cell: calls, bp, truth and coverage flags.

    Mouse: ``cov_class_rule73`` (class bp >= 0.9, >= 20 counts),
    ``cov_subclass_rule73`` (and subclass bp >= 0.8, >= 50 counts), and
    ``cov_<level>_prov`` under the bundle's decisions (subclass also needs
    >= ``provisional_mouse_subclass_floor`` counts and the class); grouped
    by the called class. Human: ``cov_<level>_raw`` (bp >= the level's
    default, >= 20 counts) and ``cov_<level>_prov``, supercluster within
    broad; grouped by the called broad class.

    Args:
        cells: ``level_cells`` rows of one member (cells rules applied).
        species: ``human`` or ``mouse``.
        summary: The bundle's ``resolvability_summary.json``.
        names: Node -> name (the engine's vocab).
        regime: The decisions' regime.

    Returns:
        The table, indexed by ``sim_id``.
    """
    from merxen.annotation import resolvability as res

    if cells.empty:
        return pd.DataFrame()
    wide = cells.pivot(
        index="sim_id", columns="level", values=["parent", "call", "bp", "correct"]
    )
    first_level = str(cells["level"].iloc[0])
    first = cells[cells["level"] == first_level].set_index("sim_id")
    table = pd.DataFrame(index=wide.index)
    table["total_counts"] = first["total_counts"].reindex(table.index).astype(float)
    table["truth_leaf"] = (
        first[res.TRUTH_LEAF_COLUMN]
        .reindex(table.index)
        .astype(str)
        .map(lambda value: names.get(value, value))
    )
    counts = table["total_counts"].to_numpy(np.float64)
    grid = [int(value) for value in summary.get("depth_grid") or []]
    bins = res.depth_bin(counts, grid)
    emitted = emission_frame(summary, regime)
    levels = [
        str(item["level"])
        for item in summary.get("levels") or []
        if item.get("role") != "fine" and item.get("level") in wide["bp"].columns
    ]
    defaults = {
        str(item["level"]): float(item["default_threshold"])
        for item in summary.get("levels") or []
    }
    if species == "mouse":
        table["cls_call"] = wide[("parent", "class")].astype(object)
        table["truth_cls"] = (
            cells[cells["level"] == "class"]
            .set_index("sim_id")["truth_parent"]
            .reindex(table.index)
            .astype(str)
        )
        for level in levels:
            table[f"{level}_bp"] = wide[("bp", level)].astype(float)
            table[f"{level}_correct"] = wide[("correct", level)].astype(bool)
        class_bp = table["class_bp"].to_numpy(np.float64)
        table["cov_class_rule73"] = (class_bp >= MOUSE_CLASS_RULE_BP) & (
            counts >= MOUSE_CLASS_RULE_MIN_COUNTS
        )
        if "subclass" in levels:
            table["cov_subclass_rule73"] = (
                table["cov_class_rule73"].to_numpy(bool)
                & (table["subclass_bp"].to_numpy(np.float64) >= MOUSE_SUBCLASS_RULE_BP)
                & (counts >= MOUSE_SUBCLASS_RULE_MIN_COUNTS)
            )
        called = table["cls_call"].astype(str).to_numpy()
        subclass_floor = float(
            (summary.get("settings") or {}).get("provisional_mouse_subclass_floor", 60)
        )
        flags: dict[str, np.ndarray] = {}
        for level in [item for item in MOUSE_PROFILE_LEVELS if item in levels]:
            ok = provisional_ok(
                level,
                called,
                bins,
                table[f"{level}_bp"].to_numpy(np.float64),
                counts,
                emitted,
            )
            if level == "subclass":
                ok = ok & (counts >= subclass_floor) & flags.get("class", ok)
            flags[level] = ok
            table[f"cov_{level}_prov"] = ok
        return table
    broad_parent = wide[("parent", "broad")] if "broad" in levels else None
    table["cls_call"] = (
        broad_parent.astype(object) if broad_parent is not None else None
    )
    table["truth_cls"] = (
        cells[cells["level"] == "broad"]
        .set_index("sim_id")["truth_parent"]
        .reindex(table.index)
        .astype(str)
        if "broad" in levels
        else ""
    )
    for level in levels:
        parent = wide[("parent", level)].astype(object).to_numpy()
        bp = wide[("bp", level)].astype(float).to_numpy()
        table[f"{level}_correct"] = wide[("correct", level)].astype(bool).to_numpy()
        present = pd.notna(parent)
        table[f"cov_{level}_raw"] = (
            (bp >= defaults.get(level, 1.0))
            & (counts >= HUMAN_RAW_MIN_COUNTS)
            & present
        )
        table[f"cov_{level}_prov"] = (
            provisional_ok(
                level,
                np.array([str(value) for value in parent]),
                bins,
                bp,
                counts,
                emitted,
            )
            & present
        )
    if "supercluster" in levels and "broad" in levels:
        table["cov_supercluster_raw"] &= table["cov_broad_raw"]
        table["cov_supercluster_prov"] &= table["cov_broad_prov"]
    return table


def composition_weights_to_real(
    truth_sub: pd.Series,
    real_share: pd.Series,
    class_of: Mapping[str, str],
    *,
    trim: float = COMPOSITION_TRIM,
) -> np.ndarray:
    """Return simulated-cell weights to a real called composition.

    Phase 1's ``comp_weights``: ``w = [p_real(class) / p_sim(class)] x
    [p_real(sub | class) / p_sim(sub | class)]``, the within-class factor
    trimmed at ``trim`` x its class's median positive value and rescaled to
    mean 1 within the class; subclasses absent from the real data get
    weight 0.

    Args:
        truth_sub: Truth subclass name per simulated cell.
        real_share: Real called share per subclass name.
        class_of: Subclass name -> class name.
        trim: Trim factor.

    Returns:
        Weights per simulated cell.
    """
    weights = np.zeros(len(truth_sub))
    subs = truth_sub.astype(str).to_numpy()
    classes = np.array([class_of.get(value, "?") for value in subs], dtype=object)
    real_class = pd.Series(
        [class_of.get(value, "?") for value in real_share.index], index=real_share.index
    )
    p_real_class = real_share.groupby(real_class).sum()
    p_sim_class = pd.Series(classes).value_counts(normalize=True)
    for cls in np.unique(classes):
        selected = classes == cls
        sub = pd.Series(subs[selected])
        simulated_share = sub.value_counts(normalize=True)
        real = real_share[real_class == cls]
        if real.sum() <= 0:
            continue
        real = real / real.sum()
        within = (
            real.reindex(sub.to_numpy()).fillna(0.0).to_numpy()
            / simulated_share.reindex(sub.to_numpy()).to_numpy()
        )
        positive = within[within > 0]
        if len(positive):
            within = np.minimum(within, trim * np.median(positive))
            within = within / within.mean() if within.mean() > 0 else within
        weights[selected] = (
            within * float(p_real_class.get(cls, 0.0)) / float(p_sim_class[cls])
        )
    return weights


def _kish(weights: np.ndarray) -> float:
    positive = weights[weights > 0]
    if not len(positive):
        return 0.0
    return float(positive.sum() ** 2 / np.sum(positive**2))


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    ok = np.isfinite(values) & (weights > 0)
    if not ok.any():
        return None
    return float(np.sum(values[ok] * weights[ok]) / np.sum(weights[ok]))


def profile_metrics(table: pd.DataFrame) -> list[str]:
    """Return the coverage and precision metrics a profile table carries."""
    coverage = [column for column in table.columns if column.startswith("cov_")]
    precision = []
    for column in coverage:
        level = column[len("cov_") :].rsplit("_", 1)[0]
        if f"{level}_correct" in table.columns:
            precision.append("prec_" + column[len("cov_") :])
    return coverage + precision


def per_class_predictions(
    table: pd.DataFrame,
    weights: np.ndarray | None,
    metrics: Sequence[str],
    *,
    group: str = "cls_call",
) -> pd.DataFrame:
    """Return weighted coverage and precision per called class and ALL.

    Phase 1's ``analyse.py`` ``per_class``: coverage is the weighted share of
    simulated cells covered; precision the weighted share of covered cells
    whose call is correct.
    """
    if weights is None:
        weights = np.ones(len(table))
    series = pd.Series(np.asarray(weights, dtype=np.float64), index=table.index)
    rows = []
    groups = [
        (str(cls), frame)
        for cls, frame in table.groupby(group, sort=True, observed=True)
    ]
    for cls, frame in [*groups, ("ALL", table)]:
        values = series.loc[frame.index].to_numpy()
        record: dict[str, Any] = {
            "class": cls,
            "n": int(len(frame)),
            "w": float(values.sum()),
            "kish_n": _kish(values),
        }
        for metric in metrics:
            if metric.startswith("prec_"):
                name = metric[len("prec_") :]
                level = name.rsplit("_", 1)[0]
                covered = frame[f"cov_{name}"].to_numpy(bool)
                record[metric] = _weighted_mean(
                    frame[f"{level}_correct"].to_numpy(np.float64)[covered],
                    values[covered],
                )
            else:
                record[metric] = _weighted_mean(
                    frame[metric].to_numpy(np.float64), values
                )
        rows.append(
            {key: (np.nan if value is None else value) for key, value in record.items()}
        )
    return pd.DataFrame(rows)


def member_mean(predictions: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the mean over members of each per-class prediction."""
    frames = [frame.assign(member=name) for name, frame in predictions.items()]
    if not frames:
        return pd.DataFrame()
    stacked = pd.concat(frames, ignore_index=True)
    numeric = [
        column
        for column in stacked.columns
        if column not in ("class", "member")
        and pd.api.types.is_numeric_dtype(stacked[column])
    ]
    mean = stacked.groupby("class", sort=True)[numeric].mean().reset_index()
    mean["member"] = "member_mean"
    return mean


def class_depth_predictions(
    predicted: pd.DataFrame, profile: Any, grid: Sequence[int]
) -> pd.DataFrame:
    """Return the per-class depth predictor of plan §8.3 v7.5 (no mapping).

    For each (level L, class c): the profile's share ``s_c(d)`` of class-c
    cells per bin (its own totals, or the pool it falls back to), the
    resolvable share ``sum_d s_c(d)`` over emitted bins and the predicted
    coverage ``sum_d s_c(d) cov(L, c, d)``, ``cov`` the decision's coverage
    at its applied threshold. RESOLVE computes the same sums with each
    dataset's own ``s_c(d)`` (M4 follow-up).

    Args:
        predicted: ``predicted_levels`` rows of one reference and regime.
        profile: ``sim_inputs.DepthProfile``.
        grid: The bundle's depth grid.

    Returns:
        ``level, class, profile_source, n_profile_cells, profile_median,
        share_below_grid, resolvable_share, predicted_coverage,
        n_emitted_bins, n_test_cells`` rows (``n_test_cells``: the class's
        test cells at the level, the weight of a pooled profile's headline).
    """
    from merxen.annotation.resolvability import depth_bin

    if predicted.empty:
        return pd.DataFrame()
    rows = []
    cache: dict[str, dict[int, float]] = {}
    for (level, cls), group in predicted.groupby(
        ["level", "class"], sort=True, observed=True
    ):
        source = profile.source(str(cls))
        if source not in cache:
            values = profile.values(source)
            bins = depth_bin(values, list(grid))
            shares = {0: float(np.mean(np.isnan(bins)))}
            for depth in grid:
                shares[int(depth)] = float(np.mean(bins == float(depth)))
            cache[source] = shares
        shares = cache[source]
        emitted = group[group["status"] == "emitted"]
        coverage = {
            int(row["depth"]): float(row["coverage"])
            if row.get("coverage") is not None and np.isfinite(float(row["coverage"]))
            else 0.0
            for _index, row in emitted.iterrows()
        }
        values = profile.values(source)
        rows.append(
            {
                "level": str(level),
                "class": str(cls),
                "profile_source": source,
                "n_profile_cells": int(len(values)),
                "profile_median": float(np.median(values)) if len(values) else None,
                "share_below_grid": shares.get(0, 0.0),
                "resolvable_share": float(
                    sum(shares.get(depth, 0.0) for depth in coverage)
                ),
                "predicted_coverage": float(
                    sum(
                        shares.get(depth, 0.0) * value
                        for depth, value in coverage.items()
                    )
                ),
                "n_emitted_bins": int(len(coverage)),
                "n_test_cells": _max_test_cells(group),
            }
        )
    return pd.DataFrame(rows)


def _max_test_cells(group: pd.DataFrame) -> int:
    """Test cells of a (level, class): its largest ``n_test`` over the bins."""
    if "n_test" not in group.columns:
        return 0
    values = pd.to_numeric(group["n_test"], errors="coerce").dropna()
    return int(values.max()) if len(values) else 0


# How a headline weights its classes (pre-registration §22.2 D5): a real
# composition when one is given (mouse), the profile's own class composition
# for a per-class profile, the test-set composition (the self-map test cells'
# class shares) for a pooled profile or scenario without classes (the human
# lung-FFPE scenario).
HEADLINE_REAL_COMPOSITION: Final = "real_composition"
HEADLINE_PROFILE_COMPOSITION: Final = "profile_class_composition"
HEADLINE_TEST_SET_COMPOSITION: Final = "test_set_composition"
HEADLINE_LABELS: Final[dict[str, str]] = {
    HEADLINE_REAL_COMPOSITION: "weighted to the real composition",
    HEADLINE_PROFILE_COMPOSITION: "weighted to the profile's class composition",
    HEADLINE_TEST_SET_COMPOSITION: "weighted to the test-set composition",
}


def profile_class_shares(profile: Any) -> dict[str, float] | None:
    """Return a per-class profile's own class shares (``None`` if pooled)."""
    if getattr(profile, "pooled", False):
        return None
    n_total = sum(len(values) for values in profile.by_class.values())
    if n_total <= 0:
        return None
    return {str(cls): len(values) / n_total for cls, values in profile.by_class.items()}


def headline_class_shares(
    profile: Any, truth_class: Sequence[object] | np.ndarray
) -> tuple[str, dict[str, float]]:
    """Return how profile mode weights its ALL row without a real composition.

    Args:
        profile: The ``sim_inputs.DepthProfile``.
        truth_class: Truth class per test cell (``profile_truth_classes``).

    Returns:
        ``(basis, class shares)``: the profile's own class shares when it is
        per class and holds a truth class, else the test-set composition.
    """
    classes = pd.Series(np.asarray(truth_class, dtype=object)).astype(str)
    shares = profile_class_shares(profile)
    if shares and any(shares.get(cls, 0.0) > 0 for cls in set(classes)):
        return HEADLINE_PROFILE_COMPOSITION, shares
    test = classes.value_counts(normalize=True)
    return HEADLINE_TEST_SET_COMPOSITION, {
        str(cls): float(share) for cls, share in test.items()
    }


def class_share_weights(
    truth_class: Sequence[object] | pd.Series | np.ndarray,
    shares: Mapping[str, float],
) -> np.ndarray:
    """Return per-cell weights ``shares[c] / (simulated share of c)``.

    Weights the simulated cells of each truth class ``c`` to a class
    composition (0 for a class without a share; missing classes count as
    their own ``"nan"`` class).

    Args:
        truth_class: Truth class per simulated cell.
        shares: Target share per class.

    Returns:
        The weights.
    """
    classes = pd.Series(np.asarray(truth_class, dtype=object)).astype(str)
    if classes.empty:
        return np.zeros(0, dtype=np.float64)
    simulated = classes.value_counts(normalize=True)
    return np.array(
        [float(shares.get(cls, 0.0)) / float(simulated[cls]) for cls in classes],
        dtype=np.float64,
    )


def with_all_row(prediction: pd.DataFrame, weighted: pd.DataFrame) -> pd.DataFrame:
    """Return ``prediction`` with its ALL row taken from ``weighted``."""
    if prediction.empty or weighted.empty:
        return prediction
    rows = prediction[prediction["class"] != "ALL"]
    return pd.concat([rows, weighted[weighted["class"] == "ALL"]], ignore_index=True)[
        list(prediction.columns)
    ]


def class_depth_headline(
    class_depth: pd.DataFrame, profile: Any
) -> dict[str, dict[str, Any]]:
    """Return, per level, the class-composition-weighted class-depth predictor.

    A per-class profile weights each class by its own class share (classes
    the profile lacks do not enter). A pooled profile or scenario has no
    class composition (the human lung-FFPE scenario), so its classes are
    weighted by the test-set composition, each class's share of the self-map
    test cells at the level (``n_test_cells``; pre-registration §22.2 D5);
    ``weights`` records which.
    """
    result: dict[str, dict[str, Any]] = {}
    if class_depth.empty:
        return result
    shares = profile_class_shares(profile) or {}
    for level, rows in class_depth.groupby("level", sort=True):
        weights = np.array([shares.get(str(cls), 0.0) for cls in rows["class"]])
        mass = float(weights.sum())
        basis = HEADLINE_PROFILE_COMPOSITION
        if mass <= 0 and "n_test_cells" in rows.columns:
            counts = pd.to_numeric(rows["n_test_cells"], errors="coerce").fillna(0.0)
            total = float(counts.sum())
            if total > 0:
                weights = counts.to_numpy(np.float64) / total
                basis = HEADLINE_TEST_SET_COMPOSITION
        covered = float(weights.sum())
        result[str(level)] = {
            "weights": basis,
            "profile_share_covered": round(mass, 4),
            "resolvable_share": None
            if covered <= 0
            else round(float(np.sum(weights * rows["resolvable_share"]) / covered), 4),
            "predicted_coverage": None
            if covered <= 0
            else round(
                float(np.sum(weights * rows["predicted_coverage"]) / covered), 4
            ),
        }
    return result


def panel_card_notes(species: str, chemistry: str) -> list[str]:
    """Return the panel-card notes of a family (``diagnostics.panel_card_notes``).

    Xenium Prime 5K families carry the glial upper bound and the unmeasured
    precision (user decision 4), then the M3c trust rules (plan §8.10).
    """
    return _diagnostics_notes.panel_card_notes(species, chemistry)


MEMBER_TABLE_REFERENCE: Final[dict[str, str]] = {"wmb_panel": "wmb_10xv3"}


def load_simulation_profile(
    path: Path | None, asset_id: str | None, *, species: str
) -> Any | None:
    """Return the depth profile of a simulation (CSV or registered asset).

    No per-class depth crosses species (plan §8.3 v7.5): an asset of another
    species is refused, and so is a per-class CSV none of whose classes is a
    class of the panel's species.

    Args:
        path: A per-class or pooled CSV (``sim_inputs.read_depth_profile_table``).
        asset_id: A ``profile``, ``scenario`` or ``gate_p_profile`` asset id
            (a family's NP5 profile: its confident broad calls per class).
        species: The panel's species.

    Returns:
        The ``sim_inputs.DepthProfile``, or ``None`` without either.

    Raises:
        SimulationError: If both are given or the profile is of another
            species.
    """
    from merxen.annotation import sim_inputs as si

    if path is not None and asset_id is not None:
        raise SimulationError("give --depth-profile or --depth-profile-asset, not both")
    if asset_id is not None:
        asset = si.get_asset(asset_id)
        if asset.role not in ("profile", "scenario", si.GATE_P_PROFILE_ROLE):
            raise SimulationError(f"{asset_id} is a {asset.role} asset, not a profile")
        if asset.species != species:
            raise SimulationError(
                f"{asset_id} is a {asset.species} depth profile; no depth profile "
                f"crosses species ({species} panel)"
            )
        return si.profile_from_asset(asset)
    if path is None:
        return None
    try:
        profile = si.read_depth_profile_table(path, species=species)
    except si.SimInputError as error:
        raise SimulationError(str(error)) from error
    if not profile.pooled and all(
        si.is_neuronal_class(name, species) is None for name in profile.by_class
    ):
        raise SimulationError(
            f"{path}: none of its classes is a {species} class (a depth profile "
            "never crosses species; give a pooled profile without a class column)"
        )
    return profile


def read_real_composition(path: Path | str) -> pd.DataFrame:
    """Read a real called composition: ``class``, ``subclass``, ``n_cells``.

    Raises:
        SimulationError: If a column is missing.
    """
    table = pd.read_csv(path)
    missing = [
        column for column in ("class", "subclass", "n_cells") if column not in table
    ]
    if missing:
        raise SimulationError(f"{path}: columns {missing} are missing")
    return table


def _headline_rows(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {}
    row = frame[frame["class"] == "ALL"]
    if row.empty:
        return {}
    record = row.iloc[0].to_dict()
    return {
        key: (
            None
            if value is None or (isinstance(value, float) and math.isnan(value))
            else value
        )
        for key, value in record.items()
        if key not in ("class", "member")
    }


def profile_ensemble(
    summary: Mapping[str, Any],
    resolvability: Any,
    *,
    species: str,
    chemistry: str,
    member_table: Any | None,
) -> list[Any]:
    """Return the emission members profile mode re-simulates for a bundle.

    The bundle's recorded ``emission_members`` (so a bundle built before the
    amendment of 2026-09-29 keeps its stage-D members); without a record,
    the family's members of plan §8.3 v7.3 as amended (or the configured
    ``ensemble_r1_seeds`` / ``ensemble_r3_seeds``).

    Args:
        summary: The bundle's ``resolvability_summary.json``.
        resolvability: ``AnnotationConfig.resolvability``.
        species: The panel's species.
        chemistry: The panel chemistry.
        member_table: The ``member`` asset of R3 members, if any.

    Returns:
        ``resolvability.EnsembleMember`` objects (emission role).
    """
    from merxen.annotation import resolvability as res

    recorded = [str(name) for name in summary.get("emission_members") or []]
    if recorded:
        return res.members_from_names(
            recorded,
            resolvability,
            member_table=member_table,
            table_rule=resolvability.r3_table_rule,
            residual_sd_log2=resolvability.r3_residual_sd_log2,
        )
    r1_seeds = resolvability.ensemble_r1_seeds
    r3_seeds = resolvability.ensemble_r3_seeds
    return [
        item
        for item in res.ensemble_members(
            resolvability,
            species=species,
            chemistry=chemistry,
            member_table=member_table,
            r1_seeds=None if r1_seeds is None else tuple(r1_seeds),
            r3_seeds=None if r3_seeds is None else tuple(r3_seeds),
            table_rule=resolvability.r3_table_rule,
            residual_sd_log2=resolvability.r3_residual_sd_log2,
        )
        if item.role == "emission"
    ]


def weighted_profile_predictions(
    table: pd.DataFrame,
    metrics: Sequence[str],
    *,
    sampled_class: pd.Series,
    real_share: pd.Series | None = None,
    class_of: Mapping[str, str] | None = None,
    class_shares: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Return one member's profile-mode predictions with their weights (D5).

    With a real composition every row is weighted to it
    (``composition_weights_to_real``, mouse). Without one the per-class rows
    stay unweighted and the ALL row is weighted to ``class_shares`` (the
    profile's class composition, or the test-set composition for a pooled
    profile; ``headline_class_shares``).

    Args:
        table: ``profile_cell_table`` output.
        metrics: ``profile_metrics(table)``.
        sampled_class: Truth class each simulated cell was sampled for,
            aligned to ``table``.
        real_share: Real called share per subclass, if any.
        class_of: Subclass -> class names (with ``real_share``).
        class_shares: Class shares of the ALL row without a real composition.

    Returns:
        ``per_class_predictions`` rows.
    """
    if real_share is not None and not table.empty:
        weights = composition_weights_to_real(
            table["truth_leaf"], real_share, class_of or {}
        )
        return per_class_predictions(table, weights, metrics)
    prediction = per_class_predictions(table, None, metrics)
    if table.empty or not class_shares:
        return prediction
    weighted = per_class_predictions(
        table, class_share_weights(sampled_class, class_shares), metrics
    )
    return with_all_row(prediction, weighted)


def run_profile_mode(
    *,
    reference_id: str,
    spec: AnnotationReferenceSpec,
    config: AnnotationConfig,
    bundle_dir: Path,
    test_set_dir: Path | None,
    resolvability: Mapping[str, Any],
    profile: Any,
    chemistry: Any | None,
    out_dir: Path,
    scratch_dir: Path,
    real_composition: pd.DataFrame | None = None,
    members: Sequence[str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Simulate, map and tabulate the profile-mode cells of each member.

    Each emission member of the bundle (its recorded ``emission_members``;
    without a record the family's members of plan §8.3 v7.3 as amended on
    2026-09-29) draws its cells with ``simulate_on_profile`` and is mapped
    with the production configuration onto the self-map engine; its
    per-class predictions (the raw rule and the bundle's provisional
    decisions) are weighted to the real called composition when one is
    given, and the member mean is reported beside each member. Without a
    real composition the per-class rows stay unweighted and the ALL row is
    weighted to the profile's own class composition, or, for a pooled
    profile such as the human lung-FFPE scenario, to the test-set
    composition (the test cells' class shares; pre-registration §22.2 D5),
    labelled in ``all_row_weights``. Profile mode never enters emission.

    Returns:
        The report record and resource rows.

    Raises:
        SimulationError: If the test set is missing or a member is unknown.
    """
    from merxen.annotation import reference as ref
    from merxen.annotation import resolvability as res
    from merxen.annotation import sim_inputs as si
    from merxen.annotation.mapmycells_engine import MmcBundle

    if test_set_dir is None or not test_set_dir.is_dir():
        raise SimulationError(
            f"{reference_id}: the self-map test set bundle is missing"
        )
    started = time.monotonic()
    species = spec.species
    engine_dir = _engine_dir(bundle_dir, test_set_dir, resolvability)
    engine = MmcBundle.from_dir(engine_dir)
    vocab = engine.vocab()
    specs = ref.level_specs_for(reference_id, engine, config)
    rules = ref.cells_rules_for(reference_id, config)
    test = res.load_test_cells(test_set_dir)
    summary = json.loads(
        (bundle_dir / res.RESOLVABILITY_SUMMARY_FILE).read_text(encoding="utf-8")
    )
    grid = [int(value) for value in summary.get("depth_grid") or []]
    truth = profile_truth_classes(test, specs, species, vocab)
    chemistry_name = "unknown" if chemistry is None else str(chemistry.chemistry)
    reference = MEMBER_TABLE_REFERENCE.get(reference_id)
    member_table = (
        None
        if reference is None
        else si.member_table_for(species, chemistry_name, reference)
    )
    ensemble = profile_ensemble(
        summary,
        config.resolvability,
        species=species,
        chemistry=chemistry_name,
        member_table=member_table,
    )
    if members:
        known = {item.name: item for item in ensemble}
        unknown = sorted(set(members) - set(known))
        if unknown:
            raise SimulationError(
                f"profile members {unknown} are not emission members of this family "
                f"({sorted(known)})"
            )
        ensemble = [known[name] for name in members]
    names = _node_names(vocab)
    class_of = _subclass_class_map(vocab)
    real_share = None
    if real_composition is not None and species == "mouse":
        counts = real_composition.groupby("subclass")["n_cells"].sum()
        real_share = counts / counts.sum()
    all_row_basis, class_shares = headline_class_shares(profile, truth)
    if real_share is not None:
        all_row_basis = HEADLINE_REAL_COMPOSITION
    runs: list[dict[str, Any]] = []
    map_fn = ref.mmc_map_function(
        engine,
        spec=spec,
        config=config,
        scratch_dir=scratch_dir / "mapping",
        log_dir=out_dir / "logs" / "profile_mapping",
        runs=runs,
    )
    predictions: dict[str, pd.DataFrame] = {}
    member_records: dict[str, Any] = {}
    cell_frames: list[pd.DataFrame] = []
    resources: list[dict[str, Any]] = []
    for member in ensemble:
        step = time.monotonic()
        efficiency = res.member_efficiency(member.recipe, test.genes)
        query = simulate_on_profile(
            test, profile, truth, member.recipe, grid, efficiency=efficiency
        )
        simulated_s = time.monotonic() - step
        tidy = map_fn(query, f"profile_{member.name}", 0)
        cells = res.level_cells(tidy, query, test, specs, seed=0)
        for rule in rules:
            cells = rule(cells)
        cells = res.as_stored(cells)
        table = profile_cell_table(cells, species=species, summary=summary, names=names)
        prediction = weighted_profile_predictions(
            table,
            profile_metrics(table),
            sampled_class=query.obs["truth_class_sampled"].reindex(table.index)
            if not table.empty
            else pd.Series(dtype=object),
            real_share=real_share,
            class_of=class_of,
            class_shares=class_shares,
        )
        predictions[member.name] = prediction
        member_records[member.name] = {
            "recipe": member.recipe.to_json(),
            "n_simulated": int(len(query.obs)),
            "truncated_share": float(query.obs["truncated"].mean())
            if len(query.obs)
            else None,
            "simulate_s": round(simulated_s, 1),
            "wall_s": round(time.monotonic() - step, 1),
            "all": _headline_rows(prediction),
        }
        cell_frames.append(cells.assign(member=member.name))
        resources.append(
            {
                "reference_id": reference_id,
                "phase": "profile_mode",
                "step": f"simulate_{member.name}",
                "wall_s": round(simulated_s, 1),
                "peak_rss_gb": None,
                "peak_tree_rss_gb": None,
                "peak_tree_pss_gb": None,
                "n_processors": None,
                "disk_gb": None,
            }
        )
    resources += _mapping_rows(reference_id, "profile_mode", runs)
    mean = member_mean(predictions)
    rows = [frame.assign(member=name) for name, frame in predictions.items()]
    if not mean.empty:
        rows.append(mean)
    table_out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not table_out.empty:
        table_out.insert(0, "reference_id", reference_id)
        table_out.to_csv(out_dir / PROFILE_PREDICTIONS_CSV, index=False)
    if cell_frames:
        pd.concat(cell_frames, ignore_index=True).to_parquet(
            out_dir / PROFILE_CELLS_FILE, index=False
        )
    spread: dict[str, Any] = {}
    for metric in (column for column in mean.columns if column.startswith("cov_")):
        values = [
            record["all"].get(metric)
            for record in member_records.values()
            if record["all"].get(metric) is not None
        ]
        if values:
            spread[metric] = {"min": min(values), "max": max(values)}
    record = {
        "status": "run",
        "members": list(predictions),
        "profile": profile.to_json(),
        "weighted_to_real_composition": real_share is not None,
        "all_row_weights": all_row_basis,
        "member_records": member_records,
        "member_mean_all": _headline_rows(mean),
        "member_spread_all": spread,
        "files": {
            "predictions": PROFILE_PREDICTIONS_CSV,
            "cells": PROFILE_CELLS_FILE,
        },
        "mapping_runs": runs,
        "wall_s": round(time.monotonic() - started, 1),
        "note": "profile mode is a prediction only; it never enters emission",
    }
    return record, resources


def _fmt(value: Any) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "-"
    return f"{float(value):.3f}"


def render_profile_lines(record: Mapping[str, Any]) -> list[str]:
    """Return the profile headline lines of one reference (plan §8.3 v7.5)."""
    lines: list[str] = []
    profile = (record.get("depth_profile") or {}).get("profile")
    if profile:
        lines.append(
            f"   under the depth profile {profile.get('label')} "
            f"({profile.get('n_cells')} cells"
            + (f", asset {profile.get('asset')}" if profile.get("asset") else "")
            + ("; pooled" if profile.get("pooled") else "; per class")
            + "):"
        )
    class_depth = record.get("class_depth") or {}
    for level, item in (class_depth.get("headline") or {}).items():
        basis = item.get("weights", HEADLINE_PROFILE_COMPOSITION)
        weights = (
            f"{_fmt(item.get('profile_share_covered'))} of profile cells in "
            "tabulated classes"
            if basis == HEADLINE_PROFILE_COMPOSITION
            else f"{HEADLINE_LABELS.get(basis, basis)} (a pooled profile has no "
            "class composition)"
        )
        lines.append(
            f"     {level}: resolvable share {_fmt(item.get('resolvable_share'))}, "
            f"predicted coverage {_fmt(item.get('predicted_coverage'))} "
            f"(decisions x per-class depth shares, {class_depth.get('regime')} regime; "
            f"{weights})"
        )
    mode = record.get("profile_mode") or {}
    if mode.get("status") == "run":
        mean = mode.get("member_mean_all") or {}
        spread = mode.get("member_spread_all") or {}
        basis = mode.get("all_row_weights") or (
            HEADLINE_REAL_COMPOSITION
            if mode.get("weighted_to_real_composition")
            else None
        )
        weighted = (
            "unweighted"
            if basis is None
            else f"ALL {HEADLINE_LABELS.get(str(basis), str(basis))}"
            + ("" if basis == HEADLINE_REAL_COMPOSITION else ", classes unweighted")
        )
        lines.append(
            f"     profile mode ({', '.join(mode.get('members') or [])}; {weighted}; "
            "member mean, member min-max):"
        )
        for metric, value in mean.items():
            if not str(metric).startswith("cov_"):
                continue
            band = spread.get(metric) or {}
            lines.append(
                f"       {metric}: {_fmt(value)} "
                f"({_fmt(band.get('min'))}-{_fmt(band.get('max'))})"
            )
    elif mode:
        lines.append(f"     profile mode: {mode.get('status')} ({mode.get('reason')})")
    return lines


# --------------------------------------------------------------------------
# Resolvability version 7 as a diagnostic (M3c; plan §8.3 v7.1, §12 M3c (10);
# pre-registration §21 (iii) and (vii))

V7_DIAGNOSTIC_DIR: Final = "v7_diagnostic"
V7_DIAGNOSTIC_JSON: Final = "v7_diagnostic.json"
V7_DIAGNOSTIC_TXT: Final = "V7_DIAGNOSTIC.txt"


def _inside(path: Path, roots: Sequence[Path]) -> bool:
    resolved = path.resolve()
    for root in roots:
        base = Path(root).resolve()
        if resolved == base or base in resolved.parents:
            return True
    return False


def run_v7_diagnostic(
    *,
    reference_id: str,
    spec: AnnotationReferenceSpec,
    panel: AnnotationPanel,
    config: AnnotationConfig,
    bundle_dir: Path,
    test_set_dir: Path | None,
    resolvability: Mapping[str, Any],
    chemistry: Any | None,
    out_dir: Path,
    scratch_dir: Path,
    store_roots: Sequence[Path] = (),
    fresh_seeds: Sequence[int] | None = None,
    fresh_r3_seeds: Sequence[int] | None = None,
    comparator: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compute version-7 decisions beside a bundle, never applying them.

    For a version-6 family (set a, ag7, VZG2, the P5011 pin) the
    ensemble A (the version-7 emission members of plan §8.3 v7.3 as amended
    on 2026-09-29: R1 x 8 without a measured table, or
    ``ensemble_r1_seeds`` / ``ensemble_r3_seeds``) is simulated on the
    bundle's own test set (not topped up) and engine, decided by
    ``ensemble_decide`` and compared with the bundle's stored version-6
    decisions (emitted triples lost and gained per level and regime); for a
    version-7 bundle A is the bundle's own ensemble. With ``comparator`` the
    pre-registered comparator of the amended re-test of §21 (iii)
    (``default_member_seeds(..., comparator=True)``; pre-registration §22.4)
    is run as ensemble B, otherwise with ``fresh_seeds`` a fresh ensemble B
    (R1 at those seeds, R3 at ``fresh_r3_seeds``, default ``(1,)`` as in
    stage D); the churn A vs B is reported. B must share no member with A.
    Everything is written under ``out_dir/v7_diagnostic`` (never the store);
    the bundle is only read.

    Returns:
        The report record and resource rows.

    Raises:
        SimulationError: If the test set is missing, the output lies inside
            a store, the bundle has no self-map, or B shares a member with A.
    """
    from merxen.annotation import reference as ref
    from merxen.annotation import resolvability as res
    from merxen.annotation import sim_inputs as si
    from merxen.annotation.mapmycells_engine import MmcBundle
    from merxen.annotation.vocab import load_floor_table

    target = out_dir / V7_DIAGNOSTIC_DIR
    if _inside(target, store_roots):
        raise SimulationError(
            f"the version-7 diagnostic writes to {target}, inside a reference "
            "store; it is never written to the store"
        )
    if test_set_dir is None or not test_set_dir.is_dir():
        raise SimulationError(f"{reference_id}: the self-map test set is missing")
    tables = res.load_resolvability(bundle_dir, allow_version_7=True)
    if tables is None:
        raise SimulationError(f"{reference_id}: the bundle has no self-map")
    started = time.monotonic()
    target.mkdir(parents=True, exist_ok=True)
    stored_version = tables.version
    stored = decision_rows(bundle_dir)
    engine = MmcBundle.from_dir(_engine_dir(bundle_dir, test_set_dir, resolvability))
    specs = ref.level_specs_for(reference_id, engine, config)
    rules = ref.cells_rules_for(reference_id, config)
    test = res.load_test_cells(test_set_dir)
    species = spec.species
    chemistry_name = "unknown" if chemistry is None else str(chemistry.chemistry)
    reference = ref.V7_MEMBER_TABLE_REFERENCE.get(reference_id)
    member_table = (
        None
        if reference is None
        else si.member_table_for(species, chemistry_name, reference)
    )
    grid = (
        tables.depth_grid
        if stored_version == res.RESOLVABILITY_VERSION_V7
        else res.v7_depth_grid(species, panel.n_genes, spec.depth_grid)
    )
    ensemble = res.EnsembleSettings.from_config(config.resolvability)
    settings = ref.self_map_rule_settings(config)
    runs: list[dict[str, Any]] = []
    map_fn = ref.mmc_map_function(
        engine,
        spec=spec,
        config=config,
        scratch_dir=scratch_dir / "mapping",
        log_dir=target / "logs",
        runs=runs,
    )

    def emission(
        r1_seeds: Sequence[int] | None, r3_seeds: Sequence[int] | None
    ) -> list[Any]:
        return [
            member
            for member in res.ensemble_members(
                config.resolvability,
                species=species,
                chemistry=chemistry_name,
                member_table=member_table,
                r1_seeds=None if r1_seeds is None else tuple(r1_seeds),
                r3_seeds=None if r3_seeds is None else tuple(r3_seeds),
                table_rule=config.resolvability.r3_table_rule,
                residual_sd_log2=config.resolvability.r3_residual_sd_log2,
            )
            if member.role == "emission"
        ]

    def run(members: Sequence[Any], name: str) -> Any:
        result = res.run_resolvability_v7(
            test,
            specs=specs,
            depths=grid,
            members=members,
            map_fn=map_fn,
            settings=settings,
            ensemble=ensemble,
            species=species,
            floor_table=load_floor_table(species),
            cells_rules=rules,
            provenance={
                "diagnostic": True,
                "applied": False,
                "bundle": str(bundle_dir),
                "bundle_resolvability_version": stored_version,
                "diagnostic_ensemble": name,
            },
        )
        result.write(target / name)
        return result

    record: dict[str, Any] = {
        "status": "run",
        "applied": False,
        "note": "diagnostic only: written to an output directory, never to the "
        "store, never applied",
        "bundle": str(bundle_dir),
        "bundle_resolvability_version": stored_version,
        "depth_grid": list(grid),
        "test_cells": int(len(test.obs)),
        "top_up": "not applied (the bundle's own test set)"
        if stored_version != res.RESOLVABILITY_VERSION_V7
        else "the bundle's topped-up test set",
    }
    members_b: list[Any] = []
    kind_b = None
    if comparator:
        r1_b, r3_b = res.default_member_seeds(member_table is not None, comparator=True)
        members_b, kind_b = emission(r1_b, r3_b), "comparator"
    elif fresh_seeds:
        members_b = emission(
            fresh_seeds, fresh_r3_seeds if fresh_r3_seeds is not None else (1,)
        )
        kind_b = "fresh"
    first: pd.DataFrame
    if stored_version == res.RESOLVABILITY_VERSION_V7:
        names_a = [str(name) for name in tables.summary.get("emission_members") or []]
    else:
        members_a = emission(
            config.resolvability.ensemble_r1_seeds,
            config.resolvability.ensemble_r3_seeds,
        )
        names_a = [member.name for member in members_a]
    shared = sorted({member.name for member in members_b} & set(names_a))
    if shared:
        raise SimulationError(
            f"{reference_id}: ensemble B shares members {shared} with ensemble A; "
            "a churn test needs independent keyed draws"
        )
    if stored_version == res.RESOLVABILITY_VERSION_V7:
        first = stored
        record["ensemble_a"] = {"source": "bundle", "members": names_a}
    else:
        result_a = run(members_a, "ensemble_A")
        first = result_a.decisions
        record["ensemble_a"] = {
            "members": [member.name for member in members_a],
            "ensemble": result_a.summary["ensemble"],
            "files": str(target / "ensemble_A"),
        }
        record["version_6_vs_7"] = res.v7_diagnostic_comparison(stored, first)
    if members_b:
        result_b = run(members_b, "ensemble_B")
        record["ensemble_b"] = {
            "kind": kind_b,
            "members": [member.name for member in members_b],
            "ensemble": result_b.summary["ensemble"],
            "files": str(target / "ensemble_B"),
        }
        record["fresh_draw_churn"] = res.v7_diagnostic_comparison(
            first, result_b.decisions
        )
    record["mapping_runs"] = runs
    record["wall_s"] = round(time.monotonic() - started, 1)
    (target / V7_DIAGNOSTIC_JSON).write_text(
        json.dumps(_json_native(record), indent=2) + "\n", encoding="utf-8"
    )
    (target / V7_DIAGNOSTIC_TXT).write_text(
        "\n".join(render_v7_diagnostic(reference_id, record)) + "\n", encoding="utf-8"
    )
    return record, _mapping_rows(reference_id, "v7_diagnostic", runs)


def render_v7_diagnostic(reference_id: str, record: Mapping[str, Any]) -> list[str]:
    """Return the text lines of a version-7 diagnostic."""
    lines = [
        f"Version-7 diagnostic of {reference_id} (bundle version "
        f"{record.get('bundle_resolvability_version')}; not applied)",
        f"  grid {record.get('depth_grid')}; {record.get('test_cells')} test cells; "
        f"top-up: {record.get('top_up')}",
    ]
    for key, title in (
        ("version_6_vs_7", "version 6 (stored) vs version 7 (ensemble A)"),
        ("fresh_draw_churn", "ensemble A vs fresh ensemble B"),
    ):
        comparison = record.get(key)
        if not comparison:
            continue
        lines.append(f"  {title}:")
        for regime, stats in comparison.items():
            lines.append(
                f"    {regime}: {stats['n_first']} vs {stats['n_second']} emitted "
                f"triples, lost {stats['lost']}, gained {stats['gained']}, churn "
                f"{stats['churn']:.3f}"
            )
            for level, item in stats["per_level"].items():
                lines.append(
                    f"      {level}: lost {item['lost']}, gained {item['gained']}, "
                    f"churn {item['churn']:.3f}"
                )
    return lines

"""Report item 9 (plan §9; H12): cortical-depth QC of the labels.

Cortical depth comes from manual pia / white-matter boundaries
(``cortical_depth.boundaries``) and is label-independent, so it tests the
labels without circularity:

- median equivolumetric depth (0 = pia, 1 = white matter) per confident
  supercluster with a 95% spatial block-bootstrap CI (500 µm tiles);
- the ordering Upper-layer IT < Deep-layer IT < Deep-layer NP/CT/6b with
  non-overlapping CIs, per platform, and its replication across the two
  platforms (both pass; rank correlation of the supercluster medians);
- at broad level for every pair (including broad-only datasets): the
  oligodendrocyte share in white matter vs grey matter, and the astrocyte,
  neuron and oligodendrocyte depth gradients.

The white-matter test uses the depth output's ``white_matter`` label when the
brain outline was annotated. Without it, the report uses cells outside the
cortical ribbon as the white-matter proxy and says so (``basis``).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from typing import Any, Final

import numpy as np
import pandas as pd

from merxen.annotation.report_figures import forest, line_facets
from merxen.annotation.report_inputs import ReportInputs, SampleData
from merxen.annotation.report_items import (
    confident,
    names_array,
    new_item,
    table_cells,
)
from merxen.annotation.report_metrics import (
    MedianCi,
    depth_ordering,
    depth_profile,
    depth_replication,
    grid_codes,
    median_block_ci,
    profile_gradient,
    share_contrast,
)
from merxen.annotation.report_model import ItemWriter, ReportItem, ReportOptions, metric
from merxen.annotation.schema import Columns
from merxen.annotation.vocab import HUMAN_BROAD_CLASSES

logger = logging.getLogger(__name__)

DEPTH_COLUMN: Final = "equivolumetric_depth"
UPPER_IT: Final = "Upper-layer intratelencephalic"
DEEP_IT: Final = "Deep-layer intratelencephalic"
DEEP_NP_CT_6B: Final = "Deep-layer NP/CT/6b"
DEEP_GROUP_MEMBERS: Final[tuple[str, ...]] = (
    "Deep-layer near-projecting",
    "Deep-layer corticothalamic and 6b",
)
ORDER: Final[tuple[str, ...]] = (UPPER_IT, DEEP_IT, DEEP_NP_CT_6B)
OLIGODENDROCYTES: Final = "Oligodendrocytes"
GRADIENT_CLASSES: Final[tuple[str, ...]] = ("Astrocytes", "Neurons", OLIGODENDROCYTES)
MIN_WHITE_MATTER_CELLS: Final = 50
MIN_GROUP_CELLS: Final = 20


def _depth_arrays(
    sample: SampleData,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray] | None:
    """Return the table cells, their depth and their tissue label."""
    if sample.depth is None or DEPTH_COLUMN not in sample.depth.columns:
        return None
    table = table_cells(sample)
    depth_frame = sample.depth.loc[table.index]
    depth = pd.to_numeric(depth_frame[DEPTH_COLUMN], errors="coerce").to_numpy(
        dtype=np.float64
    )
    tissue = (
        depth_frame["cortical_depth_annotation"]
        .astype(object)
        .fillna("")
        .astype(str)
        .to_numpy()
        if "cortical_depth_annotation" in depth_frame.columns
        else np.full(len(table), "", dtype=object)
    )
    return table, depth, tissue


def column_codes(sample: SampleData, table: pd.DataFrame) -> np.ndarray | None:
    """Return each table cell's cortical column id (``-1`` outside a column)."""
    if sample.depth is None or "column_id" not in sample.depth.columns:
        return None
    values = pd.to_numeric(sample.depth.loc[table.index, "column_id"], errors="coerce")
    codes = values.fillna(-1).to_numpy(dtype=np.float64).astype(np.int64)
    return codes if (codes >= 0).any() else None


def group_labels(table: pd.DataFrame) -> np.ndarray:
    """Return each cell's depth group: its confident supercluster, NP/CT/6b merged."""
    labels = np.where(
        confident(table, "supercluster"),
        names_array(table, Columns.level("supercluster", "name")),
        "",
    )
    return np.where(np.isin(labels, DEEP_GROUP_MEMBERS), DEEP_NP_CT_6B, labels)


def item_cortical_depth(
    inputs: ReportInputs, writer: ItemWriter, options: ReportOptions
) -> ReportItem:
    """Build item 9: depth ordering, replication, WM > GM and gradients."""
    item = new_item(
        9,
        "cortical_depth",
        "Cortical depth QC",
        "labels that do not sit at their layer; broad classes without their depth "
        "gradients",
    )
    if inputs.species != "human":
        item.status = "not_applicable"
        item.notes.append("Human cortex only.")
        return item
    samples = [
        sample for sample in inputs.ordered_samples() if sample.depth is not None
    ]
    if not samples:
        item.status = "not_available"
        item.notes.append(
            "No cortical-depth outputs found (run_compute_cortical_depth off, or not "
            "published)."
        )
        return item
    median_rows: list[dict[str, Any]] = []
    broad_rows: list[dict[str, Any]] = []
    profile_rows: list[pd.DataFrame] = []
    contrast_rows: list[dict[str, Any]] = []
    orderings = {}
    gated_platforms: list[str] = []
    medians_by_platform: dict[str, dict[str, MedianCi]] = {}
    for sample in samples:
        arrays = _depth_arrays(sample)
        if arrays is None:
            continue
        table, depth, tissue = arrays
        xy = sample.table_xy()
        codes = grid_codes(xy, options.tile_um) if xy is not None else None
        columns = column_codes(sample, table)
        finite = np.isfinite(depth)
        groups = group_labels(table)
        medians: dict[str, MedianCi] = {}
        medians_columns: dict[str, MedianCi] = {}
        for group in sorted(set(groups[finite]) - {""}):
            selected = finite & (groups == group)
            if selected.sum() < MIN_GROUP_CELLS:
                continue
            result = median_block_ci(
                depth[selected],
                None if codes is None else codes[selected],
                n_reps=options.n_bootstrap,
                seed=options.seed,
            )
            by_column = median_block_ci(
                depth[selected],
                None if columns is None else columns[selected],
                n_reps=options.n_bootstrap,
                seed=options.seed,
            )
            medians[group] = result
            medians_columns[group] = by_column
            median_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "platform": sample.platform,
                    "level": "supercluster",
                    "group": group,
                    "median_depth": result.median,
                    "ci_low": result.ci_low,
                    "ci_high": result.ci_high,
                    "n_cells": result.n_cells,
                    "n_tiles": result.n_tiles,
                    "ci_low_columns": by_column.ci_low,
                    "ci_high_columns": by_column.ci_high,
                    "n_columns": by_column.n_tiles,
                }
            )
        ordering = depth_ordering(medians, ORDER, min_cells=MIN_GROUP_CELLS)
        orderings[sample.platform] = ordering
        medians_by_platform[sample.platform] = medians
        gate_level = str(
            ((sample.summary.get("resolution") or {}).get("gate") or {}).get("level")
        )
        # A dataset whose gate withholds supercluster labels has no ordering to
        # test: both CI methods record not_available, never a failure.
        gated = gate_level != "full" and bool(ordering.missing)
        if gated:
            gated_platforms.append(sample.platform)
        if columns is not None:
            sensitivity = depth_ordering(
                medians_columns, ORDER, min_cells=MIN_GROUP_CELLS
            )
            sensitivity_gated = gate_level != "full" and bool(sensitivity.missing)
            item.metrics.append(
                metric(
                    "H12",
                    "depth_ordering_passes",
                    None if sensitivity_gated else sensitivity.passes,
                    definition=(
                        "the H12 ordering with CIs from resampling the depth "
                        "output's cortical columns (sensitivity; primary: 500 µm tiles)"
                    ),
                    source="report_metrics.depth_ordering",
                    sample_id=sample.sample_id,
                    platform=sample.platform,
                    kind="column_bootstrap",
                    note=(
                        f"dataset gate {gate_level}: no supercluster labels; "
                        if sensitivity_gated
                        else ""
                    )
                    + (
                        f"ordered={sensitivity.ordered}; "
                        f"separated={sensitivity.separated}"
                    ),
                )
            )
        item.metrics.append(
            metric(
                "H12",
                "depth_ordering_passes",
                None if gated else ordering.passes,
                definition=(
                    "median depth Upper-layer IT < Deep-layer IT < Deep-layer "
                    "NP/CT/6b with non-overlapping 95% block-bootstrap CIs"
                ),
                source="report_metrics.depth_ordering",
                sample_id=sample.sample_id,
                platform=sample.platform,
                note=(
                    f"dataset gate {gate_level}: no supercluster labels; "
                    if gated
                    else ""
                )
                + (
                    f"ordered={ordering.ordered}; separated={ordering.separated}; "
                    f"medians={_rounded(ordering.medians)}; "
                    f"missing={list(ordering.missing)}"
                ),
            )
        )
        for group in ORDER:
            record = medians.get(group)
            item.metrics.append(
                metric(
                    "H12",
                    "median_depth",
                    None if record is None else record.median,
                    definition=(
                        "median equivolumetric depth (0 pia, 1 WM) of "
                        "confident cells of the group"
                    ),
                    source="cortical_depth+label_table",
                    sample_id=sample.sample_id,
                    platform=sample.platform,
                    level="supercluster",
                    group=group,
                    ci_low=None if record is None else record.ci_low,
                    ci_high=None if record is None else record.ci_high,
                    n=None if record is None else record.n_cells,
                )
            )
        broad = np.where(
            confident(table, "broad"),
            names_array(table, Columns.level("broad", "name")),
            "",
        )
        for name in HUMAN_BROAD_CLASSES:
            selected = finite & (broad == name)
            if not selected.any():
                continue
            result = median_block_ci(
                depth[selected],
                None if codes is None else codes[selected],
                n_reps=options.n_bootstrap,
                seed=options.seed,
            )
            broad_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "platform": sample.platform,
                    "level": "broad",
                    "group": name,
                    "median_depth": result.median,
                    "ci_low": result.ci_low,
                    "ci_high": result.ci_high,
                    "n_cells": result.n_cells,
                    "n_tiles": result.n_tiles,
                }
            )
        profile = depth_profile(depth[finite], broad[finite], GRADIENT_CLASSES)
        profile.insert(0, "platform", sample.platform)
        profile.insert(0, "sample_id", sample.sample_id)
        profile_rows.append(profile)
        gradient = profile_gradient(
            profile[["depth_bin", "depth_low", "depth_high", "category", "share"]]
        )
        for row in gradient.to_dict("records"):
            item.metrics.append(
                metric(
                    "H12",
                    "depth_gradient_spearman",
                    row["spearman_r"],
                    definition=(
                        "Spearman r of the class's share of confident broad "
                        "cells vs depth decile (grey matter)"
                    ),
                    source="report_metrics.profile_gradient",
                    sample_id=sample.sample_id,
                    platform=sample.platform,
                    level="broad",
                    group=str(row["category"]),
                )
            )
        is_oligo = broad == OLIGODENDROCYTES
        labelled = np.isin(broad, HUMAN_BROAD_CLASSES)
        white = tissue == "white_matter"
        grey = tissue == "grey_matter"
        if (white & labelled).sum() >= MIN_WHITE_MATTER_CELLS:
            basis = "white_matter_label"
            in_a, in_b = white & labelled, grey & labelled
        else:
            basis = "outside_ribbon_proxy"
            depth_frame = sample.depth
            if depth_frame is None:
                continue
            inside = (
                depth_frame.loc[table.index]["inside_cortical_ribbon"]
                .eq(True)
                .to_numpy()
                if "inside_cortical_ribbon" in depth_frame.columns
                else finite
            )
            known = depth_frame.loc[table.index].notna().any(axis=1).to_numpy()
            in_a, in_b = (~inside) & known & labelled, inside & labelled
        contrast = share_contrast(
            is_oligo, in_a, in_b, codes, n_reps=options.n_bootstrap, seed=options.seed
        )
        contrast_rows.append(
            {
                "sample_id": sample.sample_id,
                "platform": sample.platform,
                "basis": basis,
                "share_white_matter": contrast.share_a,
                "share_grey_matter": contrast.share_b,
                "difference": contrast.difference,
                "ci_low": contrast.ci_low,
                "ci_high": contrast.ci_high,
                "n_white_matter": contrast.n_a,
                "n_grey_matter": contrast.n_b,
                "greater": contrast.greater,
                "ci_excludes_zero": contrast.ci_excludes_zero,
            }
        )
        item.metrics.append(
            metric(
                "H12",
                "oligodendrocyte_wm_minus_gm_share",
                contrast.difference,
                definition=(
                    "oligodendrocyte share of confident broad cells in white "
                    "matter minus in grey matter"
                ),
                source="cortical_depth+label_table",
                sample_id=sample.sample_id,
                platform=sample.platform,
                level="broad",
                kind=basis,
                ci_low=contrast.ci_low,
                ci_high=contrast.ci_high,
                n=contrast.n_a + contrast.n_b,
                note=f"WM {contrast.share_a:.4g} vs GM {contrast.share_b:.4g}",
            )
        )
        if basis != "white_matter_label":
            item.notes.append(
                f"{sample.sample_id}: no white-matter label in the depth output; "
                "WM > GM uses cells outside the cortical ribbon as the white-matter "
                "proxy"
            )
    replication = depth_replication(medians_by_platform, orderings)
    # Replication needs the ordering on both platforms: a gated platform (no
    # supercluster labels) leaves it not_available, not failed.
    replicable = len(orderings) >= 2 and not gated_platforms
    item.metrics.append(
        metric(
            "H12",
            "depth_ordering_replicated",
            replication.replicated if replicable else None,
            definition="the depth ordering passes on both platforms of the pair",
            source="report_metrics.depth_replication",
            scope="pair",
            note=(
                f"no ordering on {gated_platforms} (dataset gate not full); "
                if gated_platforms
                else ""
            )
            + (
                f"platforms {list(replication.platforms)}; "
                f"per platform {list(replication.passes_per_platform)}"
            ),
        )
    )
    item.metrics.append(
        metric(
            "H12",
            "depth_profile_spearman_between_platforms",
            replication.spearman_r,
            definition=(
                "Spearman r of the supercluster median depths between the two platforms"
            ),
            source="report_metrics.depth_replication",
            scope="pair",
            n=replication.n_groups,
            note=f"max |median difference| {replication.max_abs_difference:.4g}"
            if math.isfinite(replication.max_abs_difference)
            else "",
        )
    )
    medians_frame = pd.DataFrame(median_rows + broad_rows)
    writer.figure(
        item,
        "median_depth",
        lambda: forest(
            medians_frame[medians_frame["level"] == "supercluster"]
            if len(medians_frame)
            else medians_frame,
            label="group",
            group="platform",
            value="median_depth",
            low="ci_low",
            high="ci_high",
            title=(
                "Median cortical depth per confident supercluster (95% "
                "block-bootstrap CI)"
            ),
            xlabel="equivolumetric depth (0 pia, 1 WM)",
        ),
        medians_frame[medians_frame["level"] == "supercluster"]
        if len(medians_frame)
        else medians_frame,
        "Median equivolumetric depth per confident supercluster (Deep-layer NP and "
        "CT/6b merged), per platform.",
    )
    writer.figure(
        item,
        "median_depth_broad",
        lambda: forest(
            medians_frame[medians_frame["level"] == "broad"]
            if len(medians_frame)
            else medians_frame,
            label="group",
            group="platform",
            value="median_depth",
            low="ci_low",
            high="ci_high",
            title="Median cortical depth per confident broad class",
            xlabel="equivolumetric depth (0 pia, 1 WM)",
        ),
        medians_frame[medians_frame["level"] == "broad"]
        if len(medians_frame)
        else medians_frame,
        "Median equivolumetric depth per confident broad class (every pair, including "
        "broad-only datasets).",
    )
    profiles = (
        pd.concat(profile_rows, ignore_index=True) if profile_rows else pd.DataFrame()
    )
    if len(profiles):
        plotted = profiles.assign(
            depth_mid=0.5 * (profiles["depth_low"] + profiles["depth_high"])
        )
        writer.figure(
            item,
            "gradients",
            lambda: line_facets(
                plotted,
                x="depth_mid",
                y="share",
                hue="category",
                column="platform",
                title="Broad-class share by cortical depth",
                xlabel="depth (0 pia, 1 WM)",
                ylabel="share of confident broad cells",
            ),
            plotted,
            "Share of astrocytes, neurons and oligodendrocytes among confident broad "
            "cells per depth decile.",
        )
    contrasts = pd.DataFrame(contrast_rows)
    writer.table(item, "wm_vs_gm", contrasts)
    writer.table(item, "replication", pd.DataFrame([replication.__dict__]))
    item.summary = contrasts
    return item


def _rounded(values: tuple[float, ...]) -> list[float | None]:
    return [round(value, 4) if math.isfinite(value) else None for value in values]


def ordering_summary(orderings: Mapping[str, Any]) -> dict[str, bool]:
    """Return each platform's ordering verdict."""
    return {platform: bool(result.passes) for platform, result in orderings.items()}


__all__ = [
    "DEEP_NP_CT_6B",
    "ORDER",
    "group_labels",
    "item_cortical_depth",
    "ordering_summary",
]

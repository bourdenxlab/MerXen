"""Report item 9 (plan §9; H12): cortical-depth QC of the labels.

Cortical depth comes from manual pia / white-matter boundaries
(``cortical_depth.boundaries``) and is label-independent, so it tests the
labels without circularity:

- **the depth input first** (M7 review): per platform the depth output's QC
  warnings, coordinate source and ribbon share; for a pair, a label-free
  check in the fixed (Xenium) frame that the two sections' ribbons and depth
  fields overlap (200 µm bins shared by both sections: ribbon-membership
  agreement >= 0.8 and per-bin mean-depth r >= 0.8). A platform whose depth
  fails it (boundaries applied in the wrong frame cut arbitrarily through the
  tissue) gets every H12 metric, and the pair its replication, as
  ``not_available`` with reason ``depth_input_invalid``, a banner, and the
  item status ``partial``;
- median equivolumetric depth (0 = pia, 1 = white matter) per confident
  supercluster with a 95% block-bootstrap CI whose unit is a 500 µm
  tangential block (a full pia-to-WM strip; ``tangential_position_um``); the
  square 500 µm tile bootstrap is reported beside it (``kind =
  square_tile_500um``). The tangential blocks are also the CI the M8 gate
  scores (``SCORED_CI``; approved by the user on 2026-10-01,
  pre-registration §20 D11, superseding the square tiles of §18 item 3);
- the ordering Upper-layer IT < Deep-layer IT < Deep-layer NP/CT/6b with
  non-overlapping CIs, per platform, and its replication across the two
  platforms (both pass; rank correlation of the supercluster medians);
- at broad level for every pair (including broad-only datasets): the
  oligodendrocyte share in white matter vs grey matter, and the astrocyte,
  neuron and oligodendrocyte depth gradients (shares of confident broad
  cells per depth decile).

The white-matter test uses the depth output's ``white_matter`` label when the
brain outline was annotated. Without it, the report uses cells outside the
cortical ribbon as the white-matter proxy and says so (``basis``).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
    DEPTH_AGREEMENT_BIN_UM,
    DEPTH_AGREEMENT_MIN_CELLS,
    DEPTH_BLOCK_UM,
    DEPTH_MIN_BIN_DEPTH_R,
    DEPTH_MIN_RIBBON_AGREEMENT,
    DepthAgreement,
    MedianCi,
    depth_input_agreement,
    depth_ordering,
    depth_profile,
    depth_replication,
    grid_codes,
    median_block_ci,
    profile_gradient,
    share_contrast,
    tangential_block_codes,
)
from merxen.annotation.report_model import (
    ItemWriter,
    MetricRecord,
    ReportItem,
    ReportOptions,
    metric,
)
from merxen.annotation.report_panel import Banner
from merxen.annotation.schema import Columns
from merxen.annotation.vocab import HUMAN_BROAD_CLASSES

logger = logging.getLogger(__name__)

DEPTH_COLUMN: Final = "equivolumetric_depth"
TANGENTIAL_COLUMN: Final = "tangential_position_um"
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
# The H12 CI methods (pre-registration doc §17): the primary unit is a 500 µm
# tangential block (a full pia-to-WM strip); square 500 µm tiles are the
# conservative sensitivity. Without tangential positions the tiles are the
# only (and so the primary) method.
PRIMARY_CI: Final = "tangential_block_500um"
SQUARE_TILE_CI: Final = "square_tile_500um"
# The CI the M8 gate scores the H12 ordering on: the tangential blocks, the
# report's primary (approved by the user on 2026-10-01, pre-registration §20
# D11; it supersedes the square tiles of §18 item 3). The scored ordering
# records are the kind-less ones (per platform and the pair's replication);
# a platform without tangential positions has only the tiles, which are then
# its primary. WM > GM keeps the square tiles: white matter has no
# tangential position (``report_scoring.score_h12``).
SCORED_CI: Final = PRIMARY_CI
INVALID_REASON: Final = "depth_input_invalid"
# A depth computed on transformed (aligned) coordinates: the frame in which
# the manual boundaries must also be (plan §9 item 9; M7 review).
TRANSFORMED_FRAME_TOKEN: Final = "_aligned"


@dataclass
class DepthInput:
    """One sample's depth output on its table cells.

    Attributes:
        sample: The sample.
        table: Its table cells.
        depth: Equivolumetric depth per table cell (NaN outside the ribbon).
        tissue: ``cortical_depth_annotation`` per table cell (``""`` if
            absent).
        inside: Whether each table cell lies inside the cortical ribbon.
        known: Whether each table cell has a row in the depth output.
        tangential: Tangential position per table cell (µm), if recorded.
    """

    sample: SampleData
    table: pd.DataFrame
    depth: np.ndarray
    tissue: np.ndarray
    inside: np.ndarray
    known: np.ndarray
    tangential: np.ndarray | None


@dataclass(frozen=True)
class DepthValidity:
    """Whether one platform's depth input is usable for H12.

    Attributes:
        sample_id: The sample.
        platform: Its platform.
        valid: ``True`` / ``False`` from the pair check, ``None`` when it
            could not run (one platform, coordinates not in one frame).
        reason: Why it is invalid, or why it was not checked.
        qc_warnings: The depth output's QC warnings.
        coordinate_source: The coordinates the depth was computed on.
        n_cells: Cells of the depth output (its table).
        n_inside: Of them inside the cortical ribbon.
        transformed: The depth was computed on transformed (aligned)
            coordinates.
    """

    sample_id: str
    platform: str
    valid: bool | None
    reason: str
    qc_warnings: tuple[str, ...]
    coordinate_source: str | None
    n_cells: int
    n_inside: int
    transformed: bool


def _depth_input(sample: SampleData) -> DepthInput | None:
    """Return a sample's depth arrays on its table cells, if any."""
    if sample.depth is None or DEPTH_COLUMN not in sample.depth.columns:
        return None
    table = table_cells(sample)
    frame = sample.depth.loc[table.index]
    depth = pd.to_numeric(frame[DEPTH_COLUMN], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if "cortical_depth_annotation" in frame.columns:
        annotation = frame["cortical_depth_annotation"].astype(object).fillna("")
        tissue = annotation.astype(str).to_numpy()
    else:
        tissue = np.full(len(table), "", dtype=object)
    inside = (
        frame["inside_cortical_ribbon"].eq(True).to_numpy()
        if "inside_cortical_ribbon" in frame.columns
        else np.isfinite(depth)
    )
    known = frame.notna().any(axis=1).to_numpy()
    tangential = (
        pd.to_numeric(frame[TANGENTIAL_COLUMN], errors="coerce").to_numpy(
            dtype=np.float64
        )
        if TANGENTIAL_COLUMN in frame.columns
        else None
    )
    if tangential is not None and not np.isfinite(tangential).any():
        tangential = None
    return DepthInput(sample, table, depth, tissue, inside, known, tangential)


def _qc_record(sample: SampleData, segmentation: str) -> dict[str, Any]:
    qc = sample.depth_qc or {}
    return dict((qc.get("tables") or {}).get(segmentation) or {})


def depth_validity(
    inputs: Sequence[DepthInput], segmentation: str
) -> tuple[DepthAgreement | None, dict[str, DepthValidity], str]:
    """Check the depth inputs of a pair without labels (item 9; M7 review).

    Adjacent sections share their anatomy: in the fixed frame their cortical
    ribbons and depth fields must overlap (``depth_input_agreement``). When
    they do not, the platform whose depth was computed on transformed
    coordinates is the one whose boundaries can be in the wrong frame (the
    fixed platform's depth and boundaries are both native); when neither or
    both are, every platform is marked invalid.

    Args:
        inputs: The pair's depth inputs (one per platform).
        segmentation: Segmentation (the depth QC record's key).

    Returns:
        The agreement (``None`` when not computed), the validity per sample
        id, and a note.
    """
    base: dict[str, DepthValidity] = {}
    for entry in inputs:
        record = _qc_record(entry.sample, segmentation)
        source = record.get("coordinate_source")
        warnings = tuple(
            str(value) for value in (entry.sample.depth_qc or {}).get("warnings") or ()
        )
        n_cells = record.get("n_cells")
        n_inside = record.get("n_inside_ribbon")
        base[entry.sample.sample_id] = DepthValidity(
            sample_id=entry.sample.sample_id,
            platform=entry.sample.platform,
            valid=None,
            reason="",
            qc_warnings=warnings,
            coordinate_source=None if source is None else str(source),
            n_cells=int(n_cells) if n_cells is not None else int(entry.known.sum()),
            n_inside=int(n_inside)
            if n_inside is not None
            else int((entry.inside & entry.known).sum()),
            transformed=TRANSFORMED_FRAME_TOKEN in str(source or ""),
        )
    if len(inputs) != 2:
        note = "not checked: the label-free depth check needs both platforms"
        return None, _with(base, None, note), note
    first, second = inputs
    xy_a, xy_b = first.sample.table_xy(), second.sample.table_xy()
    if (
        xy_a is None
        or xy_b is None
        or not first.sample.aligned_frame
        or not second.sample.aligned_frame
    ):
        note = "not checked: the two sections' coordinates are not in one frame"
        return None, _with(base, None, note), note
    keep_a = first.known & np.isfinite(xy_a).all(axis=1)
    keep_b = second.known & np.isfinite(xy_b).all(axis=1)
    agreement = depth_input_agreement(
        xy_a[keep_a],
        first.inside[keep_a],
        first.depth[keep_a],
        xy_b[keep_b],
        second.inside[keep_b],
        second.depth[keep_b],
        bin_um=DEPTH_AGREEMENT_BIN_UM,
        min_cells=DEPTH_AGREEMENT_MIN_CELLS,
    )
    verdict = agreement.valid(DEPTH_MIN_RIBBON_AGREEMENT, DEPTH_MIN_BIN_DEPTH_R)
    summary = (
        f"ribbon agreement {agreement.ribbon_agreement:.3f} (min "
        f"{DEPTH_MIN_RIBBON_AGREEMENT}) over {agreement.n_bins} shared "
        f"{agreement.bin_um:.0f} µm bins; per-bin mean-depth r "
        f"{agreement.depth_r:.3f} (min {DEPTH_MIN_BIN_DEPTH_R}) over "
        f"{agreement.n_depth_bins} bins"
    )
    if verdict is None:
        note = f"not checked: too few shared bins ({summary})"
        return agreement, _with(base, None, note), note
    if verdict:
        return agreement, _with(base, True, ""), f"valid: {summary}"
    transformed = [key for key, value in base.items() if value.transformed]
    blamed = set(transformed) if len(transformed) == 1 else set(base)
    basis = (
        "its depth was computed on transformed coordinates "
        f"({base[transformed[0]].coordinate_source}); the manual boundaries must "
        "be in that frame"
        if len(transformed) == 1
        else "the check cannot tell which platform is wrong"
    )
    out = {}
    for key, value in base.items():
        if key in blamed:
            reason = f"{INVALID_REASON}: {summary}; {basis}"
            out[key] = _replace(value, False, reason)
        else:
            out[key] = _replace(value, True, "")
    return agreement, out, f"invalid: {summary}"


def _replace(value: DepthValidity, valid: bool | None, reason: str) -> DepthValidity:
    return DepthValidity(
        sample_id=value.sample_id,
        platform=value.platform,
        valid=valid,
        reason=reason,
        qc_warnings=value.qc_warnings,
        coordinate_source=value.coordinate_source,
        n_cells=value.n_cells,
        n_inside=value.n_inside,
        transformed=value.transformed,
    )


def _with(
    base: Mapping[str, DepthValidity], valid: bool | None, reason: str
) -> dict[str, DepthValidity]:
    return {key: _replace(value, valid, reason) for key, value in base.items()}


def group_labels(table: pd.DataFrame) -> np.ndarray:
    """Return each cell's depth group: its confident supercluster, NP/CT/6b merged."""
    labels = np.where(
        confident(table, "supercluster"),
        names_array(table, Columns.level("supercluster", "name")),
        "",
    )
    return np.where(np.isin(labels, DEEP_GROUP_MEMBERS), DEEP_NP_CT_6B, labels)


def _gated(record: MetricRecord, invalid_reason: str) -> MetricRecord:
    """Return ``record`` as not available when its depth input is invalid."""
    if not invalid_reason:
        return record
    return record.model_copy(
        update={
            "value": None,
            "ci_low": None,
            "ci_high": None,
            "status": "not_available",
            "note": invalid_reason,
        }
    )


def item_cortical_depth(
    inputs: ReportInputs, writer: ItemWriter, options: ReportOptions
) -> ReportItem:
    """Build item 9: depth validity, ordering, replication, WM > GM, gradients."""
    item = new_item(
        9,
        "cortical_depth",
        "Cortical depth QC",
        "labels that do not sit at their layer; broad classes without their depth "
        "gradients; depth inputs whose boundaries do not fit the tissue",
    )
    if inputs.species != "human":
        item.status = "not_applicable"
        item.notes.append("Human cortex only.")
        return item
    depth_inputs = [
        entry
        for entry in (_depth_input(sample) for sample in inputs.ordered_samples())
        if entry is not None
    ]
    if not depth_inputs:
        item.status = "not_available"
        item.notes.append(
            "No cortical-depth outputs found (run_compute_cortical_depth off, or not "
            "published)."
        )
        return item
    agreement, validity, validity_note = depth_validity(
        depth_inputs, inputs.sources.segmentation
    )
    item.notes.append(f"Depth input check: {validity_note}.")
    _validity_metrics(item, agreement, validity)
    validity_rows = [
        {
            "sample_id": value.sample_id,
            "platform": value.platform,
            "depth_input_valid": value.valid,
            "reason": value.reason,
            "coordinate_source": value.coordinate_source,
            "qc_warnings": "; ".join(value.qc_warnings),
            "n_cells": value.n_cells,
            "n_inside_ribbon": value.n_inside,
            "inside_ribbon_share": value.n_inside / value.n_cells
            if value.n_cells
            else math.nan,
            **_agreement_columns(agreement),
        }
        for value in validity.values()
    ]
    writer.table(item, "depth_validity", pd.DataFrame(validity_rows))
    for value in validity.values():
        if value.valid is False:
            item.banners.append(
                Banner(
                    severity="error",
                    sample_id=value.sample_id,
                    code=INVALID_REASON,
                    text=(
                        f"cortical-depth input fails the label-free cross-platform "
                        f"check ({value.reason.removeprefix(INVALID_REASON + ': ')}). "
                        f"Every H12 metric of {value.platform} and the pair's "
                        "replication are not available (depth_input_invalid)."
                    ),
                )
            )
    invalid = {
        value.platform: value.reason
        for value in validity.values()
        if value.valid is False
    }
    median_rows: list[dict[str, Any]] = []
    broad_rows: list[dict[str, Any]] = []
    profile_rows: list[pd.DataFrame] = []
    contrast_rows: list[dict[str, Any]] = []
    orderings = {}
    tile_orderings = {}
    gated_platforms: list[str] = []
    medians_by_platform: dict[str, dict[str, MedianCi]] = {}
    tile_medians_by_platform: dict[str, dict[str, MedianCi]] = {}
    methods: set[str] = set()
    for entry in depth_inputs:
        sample, table, depth, tissue = (
            entry.sample,
            entry.table,
            entry.depth,
            entry.tissue,
        )
        reason = invalid.get(sample.platform, "")
        series = sample.platform + (" (depth input invalid)" if reason else "")
        xy = sample.table_xy()
        tiles = grid_codes(xy, options.tile_um) if xy is not None else None
        blocks: np.ndarray | None
        if entry.tangential is not None:
            blocks = tangential_block_codes(entry.tangential, DEPTH_BLOCK_UM)
            primary_method = PRIMARY_CI
        else:
            blocks = tiles
            primary_method = SQUARE_TILE_CI
            item.notes.append(
                f"{sample.sample_id}: no tangential positions in the depth output; "
                "the H12 CIs resample square 500 µm tiles"
            )
        methods.add(primary_method)
        finite = np.isfinite(depth)
        groups = group_labels(table)
        medians: dict[str, MedianCi] = {}
        medians_tiles: dict[str, MedianCi] = {}
        for group in sorted(set(groups[finite]) - {""}):
            selected = finite & (groups == group)
            if selected.sum() < MIN_GROUP_CELLS:
                continue
            result = median_block_ci(
                depth[selected],
                None if blocks is None else blocks[selected],
                n_reps=options.n_bootstrap,
                seed=options.seed,
            )
            by_tile = median_block_ci(
                depth[selected],
                None if tiles is None else tiles[selected],
                n_reps=options.n_bootstrap,
                seed=options.seed,
            )
            medians[group] = result
            medians_tiles[group] = by_tile
            median_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "platform": sample.platform,
                    "series": series,
                    "depth_input_valid": not reason,
                    "level": "supercluster",
                    "group": group,
                    "median_depth": result.median,
                    "ci_method": primary_method,
                    "ci_low": result.ci_low,
                    "ci_high": result.ci_high,
                    "n_cells": result.n_cells,
                    "n_blocks": result.n_tiles,
                    "ci_low_square_tiles": by_tile.ci_low,
                    "ci_high_square_tiles": by_tile.ci_high,
                    "n_square_tiles": by_tile.n_tiles,
                }
            )
        ordering = depth_ordering(medians, ORDER, min_cells=MIN_GROUP_CELLS)
        orderings[sample.platform] = ordering
        medians_by_platform[sample.platform] = medians
        # The square-tile ordering: the sensitivity beside the tangential
        # blocks (SCORED_CI), or the primary itself without tangential
        # positions (then both records hold the same verdict).
        sensitivity = (
            depth_ordering(medians_tiles, ORDER, min_cells=MIN_GROUP_CELLS)
            if primary_method == PRIMARY_CI
            else ordering
        )
        tile_orderings[sample.platform] = sensitivity
        tile_medians_by_platform[sample.platform] = medians_tiles
        gate_level = str(
            ((sample.summary.get("resolution") or {}).get("gate") or {}).get("level")
        )
        # A dataset whose gate withholds supercluster labels has no ordering to
        # test: both CI methods record not_available, never a failure.
        gated = gate_level != "full" and bool(ordering.missing)
        if gated:
            gated_platforms.append(sample.platform)
        gate_note = f"dataset gate {gate_level}: no supercluster labels; "
        item.metrics.append(
            _gated(
                metric(
                    "H12",
                    "depth_ordering_passes",
                    None if gated else sensitivity.passes,
                    definition=(
                        "the H12 ordering with CIs from resampling square 500 µm "
                        "tiles: reported beside the scored tangential blocks "
                        "(pre-registration §20 D11); conservative, a tile holds "
                        "only part of the depth range (display primary: "
                        f"{primary_method})"
                    ),
                    source="report_metrics.depth_ordering",
                    sample_id=sample.sample_id,
                    platform=sample.platform,
                    kind=SQUARE_TILE_CI,
                    note=(gate_note if gated else "")
                    + (
                        f"ordered={sensitivity.ordered}; "
                        f"separated={sensitivity.separated}; "
                        f"missing={list(sensitivity.missing)}; ci={SQUARE_TILE_CI}"
                    ),
                ),
                reason,
            )
        )
        item.metrics.append(
            _gated(
                metric(
                    "H12",
                    "depth_ordering_passes",
                    None if gated else ordering.passes,
                    definition=(
                        "median depth Upper-layer IT < Deep-layer IT < Deep-layer "
                        "NP/CT/6b with non-overlapping 95% block-bootstrap CIs "
                        f"({primary_method})"
                    ),
                    source="report_metrics.depth_ordering",
                    sample_id=sample.sample_id,
                    platform=sample.platform,
                    note=(gate_note if gated else "")
                    + (
                        f"ordered={ordering.ordered}; separated={ordering.separated}; "
                        f"medians={_rounded(ordering.medians)}; "
                        f"missing={list(ordering.missing)}; ci={primary_method}"
                    ),
                ),
                reason,
            )
        )
        for group in ORDER:
            record = medians.get(group)
            item.metrics.append(
                _gated(
                    metric(
                        "H12",
                        "median_depth",
                        None if record is None else record.median,
                        definition=(
                            "median equivolumetric depth (0 pia, 1 WM) of "
                            f"confident cells of the group; CI {primary_method}"
                        ),
                        source="cortical_depth+label_table",
                        sample_id=sample.sample_id,
                        platform=sample.platform,
                        level="supercluster",
                        group=group,
                        ci_low=None if record is None else record.ci_low,
                        ci_high=None if record is None else record.ci_high,
                        n=None if record is None else record.n_cells,
                    ),
                    reason,
                )
            )
        is_broad = confident(table, "broad")
        broad = np.where(
            is_broad, names_array(table, Columns.level("broad", "name")), ""
        )
        for name in HUMAN_BROAD_CLASSES:
            selected = finite & (broad == name)
            if not selected.any():
                continue
            result = median_block_ci(
                depth[selected],
                None if blocks is None else blocks[selected],
                n_reps=options.n_bootstrap,
                seed=options.seed,
            )
            broad_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "platform": sample.platform,
                    "series": series,
                    "depth_input_valid": not reason,
                    "level": "broad",
                    "group": name,
                    "median_depth": result.median,
                    "ci_method": primary_method,
                    "ci_low": result.ci_low,
                    "ci_high": result.ci_high,
                    "n_cells": result.n_cells,
                    "n_blocks": result.n_tiles,
                }
            )
        # Shares of confident broad cells (the definition): the cells without
        # a confident broad label are not in the denominator.
        labelled = finite & is_broad & (broad != "")
        profile = depth_profile(depth[labelled], broad[labelled], GRADIENT_CLASSES)
        profile.insert(0, "depth_input_valid", not reason)
        profile.insert(0, "series", series)
        profile.insert(0, "platform", sample.platform)
        profile.insert(0, "sample_id", sample.sample_id)
        profile_rows.append(profile)
        gradient = profile_gradient(
            profile[["depth_bin", "depth_low", "depth_high", "category", "share"]]
        )
        for row in gradient.to_dict("records"):
            item.metrics.append(
                _gated(
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
                        n=int(labelled.sum()),
                    ),
                    reason,
                )
            )
        is_oligo = broad == OLIGODENDROCYTES
        in_class = np.isin(broad, HUMAN_BROAD_CLASSES)
        white = tissue == "white_matter"
        grey = tissue == "grey_matter"
        if (white & in_class).sum() >= MIN_WHITE_MATTER_CELLS:
            basis = "white_matter_label"
            in_a, in_b = white & in_class, grey & in_class
        else:
            basis = "outside_ribbon_proxy"
            in_a = (~entry.inside) & entry.known & in_class
            in_b = entry.inside & in_class
        contrast = share_contrast(
            is_oligo, in_a, in_b, tiles, n_reps=options.n_bootstrap, seed=options.seed
        )
        contrast_rows.append(
            {
                "sample_id": sample.sample_id,
                "platform": sample.platform,
                "depth_input_valid": not reason,
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
            _gated(
                metric(
                    "H12",
                    "oligodendrocyte_wm_minus_gm_share",
                    contrast.difference,
                    definition=(
                        "oligodendrocyte share of confident broad cells in white "
                        "matter minus in grey matter (CI: square 500 µm tiles, as "
                        "white matter has no tangential position)"
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
                ),
                reason,
            )
        )
        if basis != "white_matter_label":
            item.notes.append(
                f"{sample.sample_id}: no white-matter label in the depth output; "
                "WM > GM uses cells outside the cortical ribbon as the white-matter "
                "proxy"
            )
    _pair_metrics(item, medians_by_platform, orderings, gated_platforms, invalid)
    _pair_metrics(
        item,
        tile_medians_by_platform,
        tile_orderings,
        gated_platforms,
        invalid,
        kind=SQUARE_TILE_CI,
    )
    item.metrics.append(
        metric(
            "H12",
            "depth_ci_method",
            PRIMARY_CI if methods == {PRIMARY_CI} else SQUARE_TILE_CI,
            definition=(
                "bootstrap unit of the report's H12 medians (display primary): "
                "500 µm tangential blocks (floor(tangential_position_um / 500), "
                "whole pia-to-WM strips; pre-registration doc §17); square 500 µm "
                "tiles beside them (kind square_tile_500um)"
            ),
            source="report_depth",
            scope="pair",
            note=f"methods per platform: {sorted(methods)}",
        )
    )
    item.metrics.append(
        metric(
            "H12",
            "depth_ci_scored",
            SCORED_CI,
            definition=(
                "the CI the M8 gate scores H12 on (pre-registration §20 D11): "
                "the ordering records without a kind, i.e. the tangential "
                "blocks (per platform and depth_ordering_replicated; the tiles "
                "where a platform has no tangential positions), and the "
                "square-tile CI of the WM - GM contrast; the square-tile "
                "orderings are reported beside it"
            ),
            source="report_depth",
            scope="pair",
        )
    )
    if invalid:
        item.status = "partial"
    _figures(item, writer, median_rows, broad_rows, profile_rows)
    contrasts = pd.DataFrame(contrast_rows)
    writer.table(item, "wm_vs_gm", contrasts)
    replication = depth_replication(medians_by_platform, orderings)
    writer.table(
        item,
        "replication",
        pd.DataFrame(
            [{**replication.__dict__, "depth_input_invalid": sorted(invalid)}]
        ),
    )
    item.summary = pd.DataFrame(validity_rows)[
        [
            "sample_id",
            "depth_input_valid",
            "coordinate_source",
            "qc_warnings",
            "inside_ribbon_share",
            "ribbon_agreement",
            "bin_mean_depth_r",
        ]
    ].merge(
        contrasts[["sample_id", "basis", "difference", "ci_low", "ci_high"]]
        if len(contrasts)
        else pd.DataFrame(columns=["sample_id"]),
        on="sample_id",
        how="left",
    )
    return item


def _agreement_columns(agreement: DepthAgreement | None) -> dict[str, Any]:
    """Return the pair check's columns of the validity table."""
    if agreement is None:
        return dict.fromkeys(
            ("ribbon_agreement", "n_shared_bins", "bin_mean_depth_r", "n_depth_bins")
        )
    return {
        "ribbon_agreement": agreement.ribbon_agreement,
        "n_shared_bins": agreement.n_bins,
        "bin_mean_depth_r": agreement.depth_r,
        "n_depth_bins": agreement.n_depth_bins,
    }


def _validity_metrics(
    item: ReportItem,
    agreement: DepthAgreement | None,
    validity: Mapping[str, DepthValidity],
) -> None:
    """Append the depth-input validity records (H12 inputs)."""
    rule = (
        f"valid when the two sections agree on ribbon membership in >= "
        f"{DEPTH_MIN_RIBBON_AGREEMENT:g} of the {DEPTH_AGREEMENT_BIN_UM:.0f} µm "
        f"bins of the Xenium frame holding >= {DEPTH_AGREEMENT_MIN_CELLS} cells of "
        f"each, and their per-bin mean depths correlate with r >= "
        f"{DEPTH_MIN_BIN_DEPTH_R:g} (pre-registration doc §17)"
    )
    for value in validity.values():
        base = {
            "source": "cortical_depth",
            "sample_id": value.sample_id,
            "platform": value.platform,
        }
        item.metrics.append(
            metric(
                "H12",
                "depth_input_valid",
                value.valid,
                definition=f"the platform's depth input is usable: {rule}",
                note=value.reason,
                **base,
            )
        )
        item.metrics.append(
            metric(
                "H12",
                "depth_coordinate_source",
                value.coordinate_source,
                definition="the coordinates the depth output was computed on",
                **base,
            )
        )
        item.metrics.append(
            metric(
                "H12",
                "depth_qc_warnings",
                "; ".join(value.qc_warnings) or "none",
                definition="the depth output's QC warnings",
                **base,
            )
        )
        item.metrics.append(
            metric(
                "H12",
                "depth_inside_ribbon_share",
                value.n_inside / value.n_cells if value.n_cells else None,
                definition=(
                    "cells inside the cortical ribbon / cells of the depth output"
                ),
                n=value.n_cells,
                **base,
            )
        )
    if agreement is not None:
        item.metrics.append(
            metric(
                "H12",
                "depth_ribbon_agreement",
                agreement.ribbon_agreement,
                definition=(
                    "share of shared bins where the two sections agree on ribbon "
                    f"membership (> half of the bin's cells inside); {rule}"
                ),
                source="report_metrics.depth_input_agreement",
                scope="pair",
                n=agreement.n_bins,
            )
        )
        item.metrics.append(
            metric(
                "H12",
                "depth_bin_mean_r",
                agreement.depth_r,
                definition=(
                    f"Pearson r of the per-bin mean depth of the two sections; {rule}"
                ),
                source="report_metrics.depth_input_agreement",
                scope="pair",
                n=agreement.n_depth_bins,
            )
        )


def _pair_metrics(
    item: ReportItem,
    medians_by_platform: Mapping[str, Mapping[str, MedianCi]],
    orderings: Mapping[str, Any],
    gated_platforms: Sequence[str],
    invalid: Mapping[str, str],
    *,
    kind: str | None = None,
) -> None:
    """Append the replication records (gated and depth-validity aware).

    ``kind=None`` gives the display-primary records (the replication the M8
    gate scores, §20 D11, and the between-platform rank correlation);
    ``kind=SQUARE_TILE_CI`` the replication from the square-tile orderings,
    reported beside it (the medians, and so the rank correlation, do not
    depend on the CI method).
    """
    replication = depth_replication(medians_by_platform, orderings)
    invalid_note = f"{INVALID_REASON} on {sorted(invalid)}" if invalid else ""
    # Replication needs the ordering on both platforms: a gated platform (no
    # supercluster labels) leaves it not_available, not failed.
    replicable = len(orderings) >= 2 and not gated_platforms
    replicated = metric(
        "H12",
        "depth_ordering_replicated",
        replication.replicated if replicable else None,
        definition=(
            "the depth ordering passes on both platforms of the pair"
            + (
                f" (CIs: {kind}; reported beside the scored primary)"
                if kind is not None
                else " (CIs: the display primary, depth_ci_method; the CI the M8 "
                "gate scores)"
            )
        ),
        source="report_metrics.depth_replication",
        scope="pair",
        kind=kind,
        note=(
            f"no ordering on {list(gated_platforms)} (dataset gate not full); "
            if gated_platforms
            else ""
        )
        + (
            f"platforms {list(replication.platforms)}; "
            f"per platform {list(replication.passes_per_platform)}"
        ),
    )
    between = metric(
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
    item.metrics.append(_gated(replicated, invalid_note))
    if kind is None:
        item.metrics.append(_gated(between, invalid_note))


def _figures(
    item: ReportItem,
    writer: ItemWriter,
    median_rows: list[dict[str, Any]],
    broad_rows: list[dict[str, Any]],
    profile_rows: list[pd.DataFrame],
) -> None:
    """Write item 9's figures (invalid depth inputs are labelled as such)."""
    supercluster = pd.DataFrame(median_rows)
    broad = pd.DataFrame(broad_rows)
    writer.figure(
        item,
        "median_depth",
        lambda: forest(
            supercluster,
            label="group",
            group="series",
            value="median_depth",
            low="ci_low",
            high="ci_high",
            title=(
                "Median cortical depth per confident supercluster (95% CI, 500 µm "
                "tangential blocks)"
            ),
            xlabel="equivolumetric depth (0 pia, 1 WM)",
        ),
        supercluster,
        "Median equivolumetric depth per confident supercluster (Deep-layer NP and "
        "CT/6b merged), per platform; the CSV adds the square-tile CIs. A platform "
        "marked 'depth input invalid' failed the label-free depth check.",
    )
    writer.figure(
        item,
        "median_depth_broad",
        lambda: forest(
            broad,
            label="group",
            group="series",
            value="median_depth",
            low="ci_low",
            high="ci_high",
            title="Median cortical depth per confident broad class",
            xlabel="equivolumetric depth (0 pia, 1 WM)",
        ),
        broad,
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
                column="series",
                title="Broad-class share by cortical depth",
                xlabel="depth (0 pia, 1 WM)",
                ylabel="share of confident broad cells",
            ),
            plotted,
            "Share of astrocytes, neurons and oligodendrocytes among confident broad "
            "cells per depth decile.",
        )


def _rounded(values: tuple[float, ...]) -> list[float | None]:
    return [round(value, 4) if math.isfinite(value) else None for value in values]


def ordering_summary(orderings: Mapping[str, Any]) -> dict[str, bool]:
    """Return each platform's ordering verdict."""
    return {platform: bool(result.passes) for platform, result in orderings.items()}


__all__ = [
    "DEEP_NP_CT_6B",
    "INVALID_REASON",
    "ORDER",
    "PRIMARY_CI",
    "SCORED_CI",
    "SQUARE_TILE_CI",
    "DepthInput",
    "DepthValidity",
    "depth_validity",
    "group_labels",
    "item_cortical_depth",
    "ordering_summary",
]

"""Report items 1, 2, 3, 5, 6, 8, 11 and 12 (plan §9), for both species.

Each ``item_*`` function reads ``ReportInputs``, writes its figures (PNG,
PDF and the CSV drawn) and extra tables through an ``ItemWriter``, and
returns a ``ReportItem`` with its metric records. Items that need the
clustered counts (4, 7), cortical depth (9) or the mouse region step (10)
live in ``report_expression``, ``report_depth`` and ``report_mouse``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from merxen.annotation.report_figures import (
    box_summary,
    grouped_bars,
    hist2d_panels,
    line_facets,
    stacked_bars,
    tile_maps,
)
from merxen.annotation.report_inputs import ReportInputs, SampleData
from merxen.annotation.report_metrics import (
    SelfThinningEligibility,
    agreement_by_quantile,
    block_bootstrap_shares,
    grid_codes,
    histogram_2d,
    one_hot_matrix,
    self_thinning_eligibility,
    soft_level_matrix,
    tile_mean_map,
)
from merxen.annotation.report_model import (
    ItemWriter,
    MetricRecord,
    ReportItem,
    ReportOptions,
    metric,
)
from merxen.annotation.report_panel import build_panel_card
from merxen.annotation.schema import Columns, safe_token
from merxen.annotation.vocab import (
    HUMAN_BROAD_CLASSES,
    HUMAN_LINEAGE_OF_BROAD_CLASS,
    UNASSIGNED_LABEL,
    load_floor_table,
    load_vocab,
    primary_vocab,
    seaad_broad_class,
)

logger = logging.getLogger(__name__)

LEVELS: Final[dict[str, tuple[str, ...]]] = {
    "human": ("lineage", "broad", "nt", "supercluster", "seaad_subclass"),
    "mouse": ("broad", "class", "nt", "subclass"),
}
PRIMARY_REFERENCE: Final[dict[str, str]] = {
    "human": "whb_frontal_supc_clus",
    "mouse": "wmb_panel",
}
SECONDARY_REFERENCE: Final = "seaad_mr_panel"
UNALLOCATED: Final = "unallocated"
AGREEMENT_MIN_COUNTS: Final = 20
SECOND_VOTE_BELOW: Final = 60
# The levels the WHB / SEA-AD vote checks below 60 counts, root first, and
# the statuses a failed vote leaves (plan §5.2).
VOTE_LEVELS: Final[tuple[str, ...]] = ("lineage", "broad")
VOTE_FAILURES: Final[tuple[str, ...]] = ("method_disagree", "single_method")
DEPTH_STRATUM_MIN_COUNTS: Final = 30
# The H4 label set item 8 shows first: scripts/acceptance/resolve_criteria.py's
# non-circular headline (H4_HEADLINE_SET, M4 review: the WHB-only re-resolve
# of the held-out map), which is H4's assigned class (user decision
# 2026-09-30, M8 D6); without it heldout_genes.py's WHB-only confident set,
# then any argmax set, each noted as not the scored set.
HELDOUT_HEADLINE_SET: Final = "m4_resolve_heldout_whb_only"
HELDOUT_FALLBACK_SETS: Final[tuple[str, ...]] = ("heldout_whb_confident",)
COP_SUPERCLUSTER: Final = "Committed oligodendrocyte precursor"
OPC: Final = "Oligodendrocyte precursors"
CGE_SUPERCLUSTER: Final = "CGE interneuron"
# Nodes the plan names as sinks to watch (§9 item 2: "COP / CGE / Splatter
# sinks"; Splatter is a vocab sink). Any node whose argmax share of its broad
# class exceeds its reference share of that class more than SINK_PRONE_RATIO
# times is listed too (M7 review; within the broad class because the WHB
# frontal precompute is neuron-enriched: neurons hold 90% of its cells).
SINK_PRONE_NODES: Final[tuple[str, ...]] = (COP_SUPERCLUSTER, CGE_SUPERCLUSTER)
SINK_PRONE_RATIO: Final = 3.0
SINK_COLUMNS: Final[tuple[str, ...]] = (
    "sample_id",
    "platform",
    "node",
    "broad_class",
    "sink",
    "region_plausible",
    "reasons",
    "n_argmax",
    "share_table",
    "soft_mass_share",
    "confident_share",
    "argmax_share_in_broad",
    "reference_share",
    "reference_share_in_broad",
    "argmax_over_reference_in_broad",
)
H5_DEFINITIONS: Final[dict[str, str]] = {
    "confident_cop_supercluster_share": "confident COP supercluster / table cells",
    "confident_broad_opc_share": "confident broad OPC / table cells",
    "cop_derived_opc_share": (
        "confident broad OPC cells whose WHB call is COP / confident broad OPC"
    ),
    "cop_derived_soft_opc_share": (
        "soft mass of the COP supercluster (assignment + runner-ups) / soft broad "
        "OPC mass, table cells"
    ),
    "cop_derived_argmax_opc_share": (
        "table cells whose WHB argmax is COP / table cells whose argmax broad class "
        "is OPC"
    ),
}
RAW_THRESHOLD_KEYS: Final[dict[str, dict[str, str]]] = {
    "human": {"broad": "whb_broad", "supercluster": "whb_supercluster"},
    "mouse": {"class": "wmb_class", "subclass": "wmb_subclass"},
}


# --------------------------------------------------------------------------
# Shared helpers


def table_cells(sample: SampleData) -> pd.DataFrame:
    """Return a sample's table cells (``in_table``)."""
    return sample.table()


def status_array(table: pd.DataFrame, level: str) -> np.ndarray:
    """Return a level's status per cell (``""`` when the column is absent)."""
    column = Columns.level(level, "status")
    if column not in table.columns:
        return np.full(len(table), "", dtype=object)
    return np.asarray(table[column].astype(object).fillna("").astype(str).to_numpy())


def confident(table: pd.DataFrame, level: str) -> np.ndarray:
    """Return whether each cell's ``ct_<level>_status`` is ``confident``."""
    return np.asarray(status_array(table, level) == "confident", dtype=bool)


def find_column(table: pd.DataFrame, column: str) -> str | None:
    """Return ``column``, or the one column matching it case-insensitively."""
    if column in table.columns:
        return column
    matches = [name for name in table.columns if str(name).lower() == column.lower()]
    return matches[0] if len(matches) == 1 else None


def names_array(table: pd.DataFrame, column: str) -> np.ndarray:
    """Return a label column as strings (``""`` for missing)."""
    found = find_column(table, column)
    if found is None:
        return np.full(len(table), "", dtype=object)
    column = found
    values = table[column].astype(object).where(table[column].notna(), "")
    return np.asarray(values.astype(str).to_numpy())


def tile_codes_of(sample: SampleData, options: ReportOptions) -> np.ndarray | None:
    """Return the 500 µm tile id of each table cell (``None`` without xy)."""
    xy = sample.table_xy()
    if xy is None or not np.isfinite(xy).all(axis=1).any():
        return None
    return grid_codes(xy, options.tile_um)


def shared_mask_of(inputs: ReportInputs, sample: SampleData) -> np.ndarray | None:
    """Return whether each table cell lies in the pair's shared tissue mask."""
    xy = sample.table_xy()
    if inputs.mask is None or xy is None or not sample.aligned_frame:
        return None
    finite = np.isfinite(xy).all(axis=1)
    inside = np.zeros(len(xy), dtype=bool)
    if finite.any():
        inside[finite] = inputs.mask.contains(xy[finite])
    return inside


def primary_reference(inputs: ReportInputs) -> str:
    """Return the species' primary reference id."""
    return PRIMARY_REFERENCE[inputs.species]


def whb_broad_names(table: pd.DataFrame, prefix: str = "mmc_whb") -> np.ndarray:
    """Return the broad class of each cell's assigned WHB supercluster."""
    vocab = primary_vocab("human")
    names = names_array(table, f"{prefix}_supercluster_name")
    lookup = {name: vocab.broad_class(name) for name in vocab.names}
    return np.array(
        [lookup.get(name, UNASSIGNED_LABEL) for name in names], dtype=object
    )


def seaad_broad_names(table: pd.DataFrame) -> np.ndarray:
    """Return SEA-AD's 7-class label per cell (subclass, VLMC split by supertype)."""
    subclass = names_array(table, "mmc_seaad_subclass_name")
    supertype = names_array(table, "mmc_seaad_supertype_name")
    cache: dict[tuple[str, str], str] = {}
    out = np.empty(len(subclass), dtype=object)
    for index, key in enumerate(zip(subclass, supertype, strict=True)):
        if key not in cache:
            if not key[0]:
                cache[key] = UNASSIGNED_LABEL
            else:
                try:
                    cache[key] = seaad_broad_class(key[0], key[1] or None)
                except (KeyError, ValueError):
                    cache[key] = UNASSIGNED_LABEL
        out[index] = cache[key]
    return out


def lineage_of(broad: np.ndarray) -> np.ndarray:
    """Map human broad classes to lineages (others to ``Mixed/Unknown``)."""
    return np.array(
        [
            HUMAN_LINEAGE_OF_BROAD_CLASS.get(str(value), UNASSIGNED_LABEL)
            for value in broad
        ],
        dtype=object,
    )


def _share(mask: np.ndarray, denominator: int) -> float:
    return float(np.count_nonzero(mask) / denominator) if denominator else math.nan


def _segmented(sample: SampleData) -> int:
    for key in ("n_segmented", "n_objects"):
        value = sample.summary.get(key)
        if value:
            return int(value)
    return int(len(sample.labels))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def new_item(number: int, slug: str, title: str, catches: str) -> ReportItem:
    """Return an empty item."""
    return ReportItem(
        number=number, slug=f"item{number:02d}_{slug}", title=title, catches=catches
    )


def class_categories(species: str) -> tuple[str, ...]:
    """Return the composition classes of a species (human broad, mouse WMB class)."""
    if species == "human":
        return HUMAN_BROAD_CLASSES
    return tuple(load_vocab("wmb_class").names)


def soft_matrix(table: pd.DataFrame, species: str) -> np.ndarray | None:
    """Return the table cells' soft composition rows with ``unallocated`` last.

    Human: the ``soft_broad_*`` columns (unallocated included). Mouse: the
    ``soft_class_*`` columns plus the residual ``1 - sum``.
    """
    classes = class_categories(species)
    if species == "human":
        columns = [f"{Columns.SOFT_BROAD_PREFIX}{safe_token(name)}" for name in classes]
        columns.append(Columns.SOFT_BROAD_UNALLOCATED)
    else:
        columns = [f"{Columns.SOFT_CLASS_PREFIX}{safe_token(name)}" for name in classes]
    if not set(columns) <= set(table.columns):
        return None
    values = table[columns].to_numpy(dtype=np.float64)
    values = np.nan_to_num(values)
    if species != "human":
        residual = np.clip(1.0 - values.sum(axis=1), 0.0, None)
        values = np.column_stack([values, residual])
    return np.asarray(values, dtype=np.float64)


def argmax_classes(table: pd.DataFrame, species: str) -> np.ndarray:
    """Return the class of each cell's primary assignment (argmax labels)."""
    if species == "human":
        return whb_broad_names(table)
    return names_array(table, "mmc_wmb_class_name")


def confident_class_labels(
    table: pd.DataFrame, species: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return the composition-level confident label and its confident mask."""
    level = "broad" if species == "human" else "class"
    return names_array(table, Columns.level(level, "name")), confident(table, level)


# --------------------------------------------------------------------------
# Item 1: annotatability, resolvability and the panel


def item_annotatability(
    inputs: ReportInputs, writer: ItemWriter, options: ReportOptions
) -> ReportItem:
    """Build item 1: coverage, statuses, gate, trust, flags and the panel card."""
    item = new_item(
        1,
        "annotatability",
        "Annotatability, resolvability and the panel",
        "broad-only datasets, silent degradation, over-firing flags, unsupported "
        "levels, ID mismatches",
    )
    species = inputs.species
    coverage_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    flag_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    thinning_rows: list[dict[str, Any]] = []
    validated_rows: list[dict[str, Any]] = []
    card_samples: list[dict[str, Any]] = []
    for sample in inputs.ordered_samples():
        table = table_cells(sample)
        n_table = len(table)
        n_segmented = _segmented(sample)
        resolution = sample.summary.get("resolution") or {}
        levels_record = resolution.get("levels") or {}
        for level in LEVELS[species]:
            if Columns.level(level, "status") not in sample.labels.columns:
                continue
            is_confident = confident(table, level)
            emission = (levels_record.get(level) or {}).get("emission") or {}
            row = {
                "sample_id": sample.sample_id,
                "platform": sample.platform,
                "level": level,
                "n_table": n_table,
                "n_segmented": n_segmented,
                "n_confident": int(is_confident.sum()),
                "confident_share_table": _share(is_confident, n_table),
                "confident_share_segmented": _share(is_confident, n_segmented),
                "resolvable_share": emission.get("emitted_share"),
                "extrapolated_share": emission.get("extrapolated_share"),
                "validated_share": (levels_record.get(level) or {}).get(
                    "validated_share"
                ),
                "regime": emission.get("regime"),
                "threshold_source": emission.get("threshold_source"),
            }
            coverage_rows.append(row)
            all_status = status_array(sample.labels, level)
            values, counts = np.unique(all_status, return_counts=True)
            table_status = status_array(table, level)
            for value, count in zip(values, counts, strict=True):
                status_rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "platform": sample.platform,
                        "level": level,
                        "bar": f"{sample.platform} {level}",
                        "status": value or "missing",
                        "n_objects": int(count),
                        "share_segmented": float(count / max(1, len(sample.labels))),
                        "share_table": _share(table_status == value, n_table),
                    }
                )
            item.metrics.append(
                metric(
                    "H7"
                    if level == ("broad" if species == "human" else "class")
                    else "report",
                    "confident_coverage_table",
                    row["confident_share_table"],
                    definition="confident cells at the level / table cells",
                    source="label_table",
                    sample_id=sample.sample_id,
                    platform=sample.platform,
                    level=level,
                    n=n_table,
                )
            )
            item.metrics.append(
                metric(
                    "H7"
                    if level == ("broad" if species == "human" else "class")
                    else "report",
                    "confident_coverage_segmented",
                    row["confident_share_segmented"],
                    definition="confident cells at the level / segmented objects",
                    source="label_table+resolve_summary",
                    sample_id=sample.sample_id,
                    platform=sample.platform,
                    level=level,
                    n=n_segmented,
                )
            )
            item.metrics.append(
                metric(
                    "report",
                    "resolvable_share",
                    row["resolvable_share"],
                    definition=(
                        "table cells whose (class, depth bin) the level is "
                        "emitted for (RESOLVE)"
                    ),
                    source="resolve_summary",
                    sample_id=sample.sample_id,
                    platform=sample.platform,
                    level=level,
                )
            )
        gate = dict(resolution.get("gate") or {})
        gate_rows.append(
            {
                "sample_id": sample.sample_id,
                "platform": sample.platform,
                "level": gate.get("level"),
                "warning": gate.get("warning"),
                "frac_ge30": gate.get("frac_ge30"),
                "broad_coverage_table": gate.get("broad_coverage_table"),
                "broad_coverage_segmented": gate.get("broad_coverage_segmented"),
                "level_reasons": "; ".join(
                    str(v) for v in gate.get("level_reasons") or []
                ),
                "warning_reasons": "; ".join(
                    str(v) for v in gate.get("warning_reasons") or []
                ),
                "trust_state": (sample.summary.get("trust") or {}).get("state")
                or gate.get("trust_state"),
                "validation_basis": (sample.summary.get("trust") or {}).get(
                    "validation_basis"
                ),
                "degraded_mode": (resolution.get("degraded_mode") or {}).get("name"),
            }
        )
        gate_criterion = "H8" if species == "human" else "MO7"
        for name, value, definition in (
            (
                "gate_level",
                gate.get("level"),
                "dataset gate level (full / broad_only / failed)",
            ),
            ("gate_warning", gate.get("warning"), "dataset gate warning flag"),
            (
                "gate_frac_ge30",
                gate.get("frac_ge30"),
                "A: share of table cells with >= 30 counts",
            ),
            (
                "gate_broad_coverage_table",
                gate.get("broad_coverage_table"),
                "confident broad coverage of table cells (gate input)",
            ),
            (
                "gate_broad_coverage_segmented",
                gate.get("broad_coverage_segmented"),
                "confident broad coverage of segmented objects (gate warning input)",
            ),
            (
                "gate_reasons",
                "; ".join(
                    str(v)
                    for v in list(gate.get("level_reasons") or [])
                    + list(gate.get("warning_reasons") or [])
                ),
                "gate level and warning reasons",
            ),
        ):
            if value is None and name not in ("gate_level", "gate_warning"):
                continue
            item.metrics.append(
                metric(
                    gate_criterion,
                    name,
                    value,
                    definition=definition,
                    source="resolve_summary",
                    sample_id=sample.sample_id,
                    platform=sample.platform,
                )
            )
        for stratum in (sample.summary.get("flags") or {}).get("strata") or []:
            flag_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "platform": stratum.get("platform") or sample.platform,
                    "flag": stratum.get("flag"),
                    "class": stratum.get("class"),
                    "n_cells": stratum.get("n_cells"),
                    "n_flagged": stratum.get("n_flagged"),
                    "rate": stratum.get("rate"),
                    "rate_all": stratum.get("rate_all"),
                    "informative": stratum.get("informative"),
                    "informative_h16": stratum.get("informative_h16"),
                    "reason": stratum.get("reason"),
                }
            )
        thinning = self_thinning_of(table, species)
        thinning_rows.append(
            {
                "sample_id": sample.sample_id,
                "platform": sample.platform,
                "n_deep": thinning.n_deep,
                "eligible": thinning.eligible,
                "n_labelled": thinning.n_labelled,
                "unlabelled_share": thinning.unlabelled_share,
                "dominant_label": thinning.dominant_label,
                "dominant_share": thinning.dominant_share,
                "evaluable_classes": "; ".join(thinning.evaluable_classes),
                "reliable": thinning.reliable,
                "reasons": "; ".join(thinning.reasons),
                "composition_labelled": _rounded_json(thinning.composition),
                "composition_argmax": _rounded_json(thinning.argmax_composition),
                "min_counts": thinning.min_counts,
                "min_cells": thinning.min_cells,
                "min_class_cells": thinning.min_class_cells,
            }
        )
        validated_rows.extend(validated_share_rows(sample, table, species))
        card_samples.append(
            {
                "sample_id": sample.sample_id,
                "platform": sample.platform,
                "summary": sample.summary,
                "manifest": sample.manifest,
                "n_missing_panel_genes": _missing_panel_genes(sample),
            }
        )
    flags = pd.DataFrame(flag_rows)
    for row in flag_rows:
        rate = row.get("rate")
        item.metrics.append(
            metric(
                "H16" if species == "human" else "report",
                f"flag_rate_{row['flag']}",
                rate,
                definition=(
                    "realised flag rate over the stratum's basis cells (RESOLVE)"
                ),
                source="resolve_summary",
                sample_id=row["sample_id"],
                platform=row["platform"],
                group=str(row["class"]),
                n=row.get("n_cells"),
                note=(
                    f"informative={row.get('informative')}; "
                    f"informative_h16={row.get('informative_h16')}; "
                    f"reason={row.get('reason')}"
                ),
            )
        )
    if flag_rows:
        marked = [
            bool(row.get("informative_h16"))
            == (not (_as_float(row.get("rate")) > 0.15))
            for row in flag_rows
            if row.get("rate") is not None and row.get("informative_h16") is not None
        ]
        item.metrics.append(
            metric(
                "H16" if species == "human" else "report",
                "uninformative_marking_consistent",
                all(marked) if marked else None,
                definition=(
                    "every stratum above 15% is marked uninformative and none below"
                ),
                source="resolve_summary",
                scope="pair",
                n=len(marked),
            )
        )
    card = build_panel_card(
        card_samples,
        panel_report=inputs.panel_report,
        bundles=inputs.bundles,
        primary_reference=primary_reference(inputs),
        floors=_floors(species),
    )
    item.banners.extend(card.banners)
    item.notes.extend(card.notes)
    item.metrics.extend(emission_metrics(card.emission, inputs))
    if inputs.panel_report is None:
        item.notes.append(
            "panel_report.json not found: gene-ID and control tables are from the "
            "provenance only"
        )
    for name, frame in card.tables().items():
        writer.table(item, name, frame)
    coverage = pd.DataFrame(coverage_rows)
    status = pd.DataFrame(status_rows)
    writer.figure(
        item,
        "coverage",
        lambda: grouped_bars(
            _long_coverage(coverage),
            category="level",
            group="series",
            value="share",
            facet="platform",
            ylabel="confident share",
            title="Confident fraction per level: table cells vs segmented objects",
        ),
        coverage,
        "Confident fraction per level and platform, of table cells and of segmented "
        "objects, with the resolvable share.",
    )
    writer.figure(
        item,
        "status_breakdown",
        lambda: stacked_bars(
            status,
            bar="bar",
            segment="status",
            value="share_segmented",
            title="Status breakdown (all segmented objects)",
        ),
        status,
        "Status per level (share of segmented objects; low_counts are the objects "
        "outside the table).",
    )
    writer.figure(
        item,
        "flag_rates",
        lambda: grouped_bars(
            flags.assign(
                series=flags["platform"].astype(str) + " " + flags["flag"].astype(str)
            )
            if len(flags)
            else flags,
            category="class",
            group="series",
            value="rate",
            ylabel="realised rate",
            title="Realised flag rates per class x platform (dashed: 15% H16 line)",
            horizontal_line=0.15,
        ),
        flags,
        "Realised flag rates per flag, class and platform; the flag table marks each "
        "stratum informative or not.",
    )
    curves = card.curves
    if len(curves):
        default_depths = sorted(curves["depth"].dropna().unique())
        plotted = curves[
            curves["level"].isin(["broad", "supercluster", "class", "subclass"])
        ]
        writer.figure(
            item,
            "resolvability_curves",
            lambda: line_facets(
                plotted.sort_values(["level", "depth", "class", "coverage"]),
                x="coverage",
                y="precision",
                hue="class",
                row="level",
                column="depth",
                title="Resolvability precision-coverage curves (PREP, unweighted)",
                xlabel="coverage",
                ylabel="precision",
            ),
            plotted,
            "Precision vs coverage over thresholds 0.50-0.99 per level and depth "
            f"({len(default_depths)} depth bins).",
        )
    gates = pd.DataFrame(gate_rows)
    writer.table(item, "gate", gates)
    writer.table(item, "self_thinning", pd.DataFrame(thinning_rows))
    writer.table(item, "validated_share_by_class", pd.DataFrame(validated_rows))
    for row in thinning_rows:
        base = {
            "source": "label_table",
            "sample_id": row["sample_id"],
            "platform": row["platform"],
            "n": row["n_deep"],
        }
        item.metrics.append(
            metric(
                "report",
                "self_thinning_eligible",
                row["eligible"],
                definition=">= 500 table cells with >= 200 counts",
                note=(
                    f"dominant {row['dominant_label']} {row['dominant_share']:.3f} "
                    f"of labelled; unlabelled {row['unlabelled_share']:.3f}; "
                    f"reliable={row['reliable']}"
                    if row["dominant_label"] is not None
                    else ""
                ),
                **base,
            )
        )
        item.metrics.append(
            metric(
                "report",
                "self_thinning_reliable",
                row["reliable"],
                definition=(
                    "eligible, <= 30% of the deep cells without a confident label "
                    "and no class holding >= 80% of the labelled deep cells"
                ),
                note=row["reasons"],
                **base,
            )
        )
        item.metrics.append(
            metric(
                "report",
                "self_thinning_unlabelled_share",
                row["unlabelled_share"],
                definition=(
                    "share of the deep (>= 200 counts) table cells without a "
                    "confident label at the composition level"
                ),
                **base,
            )
        )
        item.metrics.append(
            metric(
                "report",
                "self_thinning_evaluable_classes",
                row["evaluable_classes"],
                definition="classes with >= 50 labelled deep cells",
                **base,
            )
        )
    item.summary = gates
    item.notes.append(
        "Self-thinning diagnostic: eligibility and the deep cells' truth composition "
        "(confident labels; the unlabelled share and the argmax composition beside "
        "it) only; the thinned re-map itself is v1.1 (plan §5.4). Unreliable when "
        "more than 30% of the deep cells have no confident label or one class holds "
        ">= 80% of the labelled ones; classes with >= 50 labelled deep cells are "
        "evaluable."
    )
    return item


def self_thinning_of(table: pd.DataFrame, species: str) -> SelfThinningEligibility:
    """Return a sample's self-thinning eligibility (item 1).

    The truth of the diagnostic is the full-depth confident label at the
    composition level (human broad, mouse class); the deep cells without one
    are counted apart, never as a class; the engine's argmax class is
    reported beside it.

    Args:
        table: The table cells.
        species: Species.

    Returns:
        The eligibility record.
    """
    label, is_confident = confident_class_labels(table, species)
    return self_thinning_eligibility(
        table[Columns.TOTAL_COUNTS].to_numpy(dtype=np.float64),
        label,
        is_confident & (label != "") & (label != UNASSIGNED_LABEL),
        argmax=argmax_classes(table, species),
    )


def validated_share_rows(
    sample: SampleData, table: pd.DataFrame, species: str
) -> list[dict[str, Any]]:
    """Return the validated share of confident labels per level and label.

    ``ct_<L>_validated`` marks confident labels inside the panel family's
    validated region (plan §4.1, §8.2): for a family validated by simulation
    it varies per (level, class); on real-data families it is all or none.

    Args:
        sample: The sample.
        table: Its table cells.
        species: Species.

    Returns:
        Rows ``sample_id``, ``platform``, ``level``, ``label``,
        ``n_confident``, ``validated_share``.
    """
    rows = []
    for level in LEVELS[species]:
        column = Columns.level(level, "validated")
        if column not in table.columns:
            continue
        keep = confident(table, level)
        names = names_array(table, Columns.level(level, "name"))
        flags = table[column].astype(object).eq(True).to_numpy()
        for name in sorted(set(names[keep]) - {""}):
            selected = keep & (names == name)
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "platform": sample.platform,
                    "level": level,
                    "label": name,
                    "n_confident": int(selected.sum()),
                    "validated_share": float(flags[selected].mean()),
                }
            )
    return rows


def emission_metrics(
    emission: pd.DataFrame, inputs: ReportInputs
) -> list[MetricRecord]:
    """Return the H18 inputs: PREP and reweighted emission counts, raised thresholds.

    For each reference and level, in the regime RESOLVE used for the sample
    (``validated`` on validated families): the (class, depth) decisions of
    PREP's unweighted table, how many PREP emits, how many RESOLVE emitted
    after reweighting to the dataset, and how many validated thresholds the
    local rule would raise (``t* > default``).

    Args:
        emission: ``PanelCard.emission``.
        inputs: The report inputs.

    Returns:
        Metric records (criterion H18 for human, ``report`` for mouse).
    """
    records: list[MetricRecord] = []
    if emission.empty or "level" not in emission.columns:
        return records
    criterion = "H18" if inputs.species == "human" else "report"
    for sample in inputs.ordered_samples():
        levels = (sample.summary.get("resolution") or {}).get("levels") or {}
        column = f"emitted_reweighted_{sample.sample_id}"
        for (reference_id, level), part in emission.groupby(["reference_id", "level"]):
            regime = ((levels.get(level) or {}).get("emission") or {}).get("regime")
            if regime is None:
                # A level RESOLVE did not emit (a report-only fine level
                # that is off): no regime, nothing to compare.
                continue
            if "regime" in part.columns:
                part = part[part["regime"].astype(str) == str(regime)]
            if part.empty:
                continue
            status = part["status"].astype(str) if "status" in part.columns else None
            n_prep = int((status == "emitted").sum()) if status is not None else None
            reweighted = (
                int(part[column].fillna(False).astype(bool).sum())
                if column in part.columns and part[column].notna().any()
                else None
            )
            raised = None
            if {"t_star", "default_threshold"} <= set(part.columns):
                t_star = pd.to_numeric(part["t_star"], errors="coerce")
                default = pd.to_numeric(part["default_threshold"], errors="coerce")
                raised = int(((t_star - default) > 1e-9).sum())
            base: dict[str, Any] = {
                "source": "bundle+annotation_manifest",
                "sample_id": sample.sample_id,
                "platform": sample.platform,
                "level": str(level),
                "kind": f"{reference_id}:{regime}",
                "n": len(part),
            }
            records.append(
                metric(
                    criterion,
                    "prep_bins_emitted",
                    n_prep,
                    definition="PREP (unweighted) (class, depth) decisions emitted",
                    **base,
                )
            )
            records.append(
                metric(
                    criterion,
                    "reweighted_bins_emitted",
                    reweighted,
                    definition="(class, depth) bins RESOLVE emitted after reweighting",
                    **base,
                )
            )
            records.append(
                metric(
                    criterion,
                    "thresholds_raised_by_local_rule",
                    raised,
                    definition="PREP decisions whose local t* exceeds the default",
                    **base,
                )
            )
    return records


def _rounded_json(shares: Mapping[str, float]) -> str:
    return json.dumps({key: round(value, 4) for key, value in shares.items()})


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _missing_panel_genes(sample: SampleData) -> int | None:
    column = Columns.N_MISSING_PANEL_GENES
    if column in sample.labels.columns and len(sample.labels):
        values = pd.to_numeric(sample.labels[column], errors="coerce").dropna()
        if len(values):
            return int(values.max())
    panel = sample.manifest.get("panel") or {}
    value = panel.get("n_missing_panel_genes")
    return None if value is None else int(value)


def _floors(species: str) -> pd.DataFrame:
    try:
        return load_floor_table(species)  # type: ignore[arg-type]
    except (OSError, ValueError, KeyError) as error:
        logger.warning("floors of %s not loaded: %s", species, error)
        return pd.DataFrame()


def _long_coverage(coverage: pd.DataFrame) -> pd.DataFrame:
    if coverage.empty:
        return coverage
    parts = []
    for column, label in (
        ("confident_share_table", "of table cells"),
        ("confident_share_segmented", "of segmented objects"),
        ("resolvable_share", "resolvable share"),
    ):
        part = coverage[["platform", "level", column]].rename(columns={column: "share"})
        part["series"] = label
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------
# Item 2: composition per level


def _composition_rows(
    result: Any,
    *,
    sample: SampleData,
    level: str,
    kind: str,
    region: str,
    cop_share: float = math.nan,
    comparison_status: str = "comparable",
) -> list[dict[str, Any]]:
    rows = []
    for record in result.records:
        rows.append(
            {
                "sample_id": sample.sample_id,
                "platform": sample.platform,
                "level": level,
                "kind": kind,
                "region": region,
                "category": record.category,
                "share": record.share,
                "ci_low": record.ci_low,
                "ci_high": record.ci_high,
                "share_renormalised": record.share_renormalised,
                "renormalised_ci_low": record.renormalised_ci_low,
                "renormalised_ci_high": record.renormalised_ci_high,
                "mass": record.mass,
                "n_cells": result.n_cells,
                "n_tiles": result.n_tiles,
                "n_reps": result.n_reps,
                "cop_derived_share": cop_share if record.category == OPC else math.nan,
                "comparison_status": comparison_status,
            }
        )
    return rows


def _gate_level(sample: SampleData) -> str:
    gate = (sample.summary.get("resolution") or {}).get("gate") or {}
    return str(gate.get("level"))


def reference_node_shares(
    bundle: Path | None, level_suffix: str = "_SUPC"
) -> dict[str, float]:
    """Return each reference node's share of the bundle's reference cells.

    Args:
        bundle: The primary bundle directory (``profiles.parquet``).
        level_suffix: The level's token suffix (``_SUPC``: supercluster).

    Returns:
        Node name to ``n_cells / total`` over the level's nodes (empty
        without the table).
    """
    if bundle is None or not (Path(bundle) / "profiles.parquet").is_file():
        return {}
    frame = pd.read_parquet(
        Path(bundle) / "profiles.parquet", columns=["level", "node_name", "n_cells"]
    )
    frame = frame[frame["level"].astype(str).str.endswith(level_suffix)]
    counts = frame.drop_duplicates("node_name").set_index("node_name")["n_cells"]
    total = float(counts.sum())
    if total <= 0:
        return {}
    return {str(name): float(value) / total for name, value in counts.items()}


def sink_rows_of(
    sample: SampleData,
    table: pd.DataFrame,
    *,
    region_name: str,
    reference_shares: Mapping[str, float],
    soft_sc: tuple[np.ndarray, list[str]] | None,
) -> list[dict[str, Any]]:
    """Return the sinks table rows of one sample (item 2).

    A node is listed when it is a vocab sink, implausible in the region, one
    of ``SINK_PRONE_NODES`` (COP, CGE), or when its argmax share of its broad
    class exceeds its reference share of that class more than
    ``SINK_PRONE_RATIO`` times; each row gives its argmax, soft-mass and
    confident shares of the table cells and its reference shares.

    Args:
        sample: The sample.
        table: Its table cells.
        region_name: The anatomical region token.
        reference_shares: ``reference_node_shares`` of the primary bundle.
        soft_sc: ``supercluster_soft_matrix`` of the table, if any.

    Returns:
        The rows.
    """
    vocab = primary_vocab("human")
    n_table = max(1, len(table))
    argmax = names_array(table, "mmc_whb_supercluster_name")
    confident_names = np.where(
        confident(table, "supercluster"),
        names_array(table, Columns.level("supercluster", "name")),
        "",
    )
    nodes, freq = np.unique(argmax, return_counts=True)
    argmax_counts = {
        str(node): int(count) for node, count in zip(nodes, freq, strict=True) if node
    }
    broad_of = {name: vocab.broad_class(name) for name in vocab.names}
    argmax_by_broad: dict[str, int] = {}
    for node, count in argmax_counts.items():
        key = broad_of.get(node, UNASSIGNED_LABEL)
        argmax_by_broad[key] = argmax_by_broad.get(key, 0) + count
    reference_by_broad: dict[str, float] = {}
    for node, share in reference_shares.items():
        key = broad_of.get(node, UNASSIGNED_LABEL)
        reference_by_broad[key] = reference_by_broad.get(key, 0.0) + share
    soft_shares = (
        dict(zip(soft_sc[1], soft_sc[0].sum(axis=0) / n_table, strict=True))
        if soft_sc is not None
        else {}
    )
    rows = []
    candidates = sorted(set(argmax_counts) | set(SINK_PRONE_NODES))
    for node in candidates:
        if node not in vocab:
            continue
        broad = broad_of[node]
        reasons = []
        if vocab.is_sink(node):
            reasons.append("vocab_sink")
        try:
            plausible = vocab.is_region_plausible(node, region_name)
        except (KeyError, ValueError):
            plausible = True
        if not plausible:
            reasons.append("region_implausible")
        if node in SINK_PRONE_NODES:
            reasons.append("named_sink_prone")
        n_argmax = argmax_counts.get(node, 0)
        in_broad = (
            n_argmax / argmax_by_broad[broad] if argmax_by_broad.get(broad) else 0.0
        )
        reference = reference_shares.get(node, 0.0)
        reference_in_broad = (
            reference / reference_by_broad[broad]
            if reference_by_broad.get(broad)
            else 0.0
        )
        ratio = (
            in_broad / reference_in_broad
            if reference_in_broad > 0
            else (math.inf if in_broad > 0 else math.nan)
        )
        if reference_shares and ratio > SINK_PRONE_RATIO:
            reasons.append("argmax_over_reference_in_broad")
        if not reasons:
            continue
        rows.append(
            {
                "sample_id": sample.sample_id,
                "platform": sample.platform,
                "node": node,
                "broad_class": broad,
                "sink": vocab.is_sink(node),
                "region_plausible": plausible,
                "reasons": "; ".join(reasons),
                "n_argmax": n_argmax,
                "share_table": n_argmax / n_table,
                "soft_mass_share": soft_shares.get(node, math.nan),
                "confident_share": float(np.count_nonzero(confident_names == node))
                / n_table,
                "argmax_share_in_broad": in_broad,
                "reference_share": reference if reference_shares else math.nan,
                "reference_share_in_broad": reference_in_broad
                if reference_shares
                else math.nan,
                "argmax_over_reference_in_broad": ratio
                if reference_shares
                else math.nan,
            }
        )
    return rows


def supercluster_soft_matrix(
    table: pd.DataFrame, prefix: str = "mmc_whb"
) -> tuple[np.ndarray, list[str]] | None:
    """Return soft supercluster mass (assignment + runner-ups) and its columns."""
    name = f"{prefix}_supercluster_name"
    if name not in table.columns:
        return None
    names = list(primary_vocab("human").names)
    runner_names = []
    runner_bps = []
    for rank in range(1, 6):
        rank_name = f"{prefix}_supercluster_runner_up_{rank}_name"
        rank_bp = f"{prefix}_supercluster_runner_up_{rank}_bp"
        if rank_name in table.columns and rank_bp in table.columns:
            runner_names.append(names_array(table, rank_name))
            runner_bps.append(table[rank_bp].to_numpy(dtype=np.float64))
    matrix = soft_level_matrix(
        names_array(table, name),
        table[f"{prefix}_supercluster_bp"].to_numpy(dtype=np.float64),
        runner_names,
        runner_bps,
        categories=names,
    )
    return matrix, [*names, UNALLOCATED]


def item_composition(
    inputs: ReportInputs, writer: ItemWriter, options: ReportOptions
) -> ReportItem:
    """Build item 2: composition per level with block-bootstrap CIs (§5.5)."""
    item = new_item(
        2,
        "composition",
        "Composition per level",
        "abstention bias, COP / CGE / Splatter sinks",
    )
    species = inputs.species
    classes = list(class_categories(species))
    categories = [*classes, UNALLOCATED]
    rows: list[dict[str, Any]] = []
    reason_rows: list[dict[str, Any]] = []
    sink_rows: list[dict[str, Any]] = []
    depth_rows: list[dict[str, Any]] = []
    reference_shares = (
        reference_node_shares(inputs.bundles.get(primary_reference(inputs)))
        if species == "human"
        else {}
    )
    for sample in inputs.ordered_samples():
        table = table_cells(sample)
        codes = tile_codes_of(sample, options)
        inside = shared_mask_of(inputs, sample)
        regions: dict[str, np.ndarray | None] = {"whole_section": None}
        if inside is not None:
            regions["shared_mask"] = inside
        soft = soft_matrix(table, species)
        label, is_confident = confident_class_labels(table, species)
        counts = table[Columns.TOTAL_COUNTS].to_numpy(dtype=np.float64)
        kinds: dict[str, np.ndarray] = {}
        if soft is not None:
            kinds["soft"] = soft
            kinds["soft_ge30"] = np.where(
                (counts >= DEPTH_STRATUM_MIN_COUNTS)[:, None], soft, 0.0
            )
        kinds["confident"] = one_hot_matrix(label, classes, include=is_confident)
        argmax_broad = argmax_classes(table, species)
        kinds["argmax"] = one_hot_matrix(argmax_broad, classes)
        # The COP-derived part of each kind's OPC mass (human; plan §9 item 2).
        soft_sc = supercluster_soft_matrix(table) if species == "human" else None
        cop_parts: dict[str, np.ndarray] = {}
        if species == "human":
            argmax_sc = names_array(table, "mmc_whb_supercluster_name")
            is_cop = argmax_sc == COP_SUPERCLUSTER
            cop_parts["argmax"] = is_cop.astype(np.float64)
            cop_parts["confident"] = (is_confident & (label == OPC) & is_cop).astype(
                np.float64
            )
            if soft_sc is not None and soft is not None:
                cop_mass = soft_sc[0][:, soft_sc[1].index(COP_SUPERCLUSTER)]
                cop_parts["soft"] = cop_mass
                cop_parts["soft_ge30"] = np.where(
                    counts >= DEPTH_STRATUM_MIN_COUNTS, cop_mass, 0.0
                )
        leaf_status = (
            "comparable" if _gate_level(sample) == "full" else "withheld_for_comparison"
        )
        for region, keep in regions.items():
            for kind, matrix in kinds.items():
                result = block_bootstrap_shares(
                    matrix,
                    categories,
                    codes,
                    n_reps=options.n_bootstrap,
                    seed=options.seed,
                    keep=keep,
                )
                level = "broad" if species == "human" else "class"
                selected = np.ones(len(matrix), bool) if keep is None else keep
                total = float(matrix[selected].sum())
                cop_share = (
                    float(cop_parts[kind][selected].sum() / total)
                    if kind in cop_parts and total > 0
                    else math.nan
                )
                rows.extend(
                    _composition_rows(
                        result,
                        sample=sample,
                        level=level,
                        kind=kind,
                        region=region,
                        cop_share=cop_share,
                    )
                )
        if species == "human":
            conf_sc = names_array(table, Columns.level("supercluster", "name"))
            sc_names = list(primary_vocab("human").names)
            for region, keep in regions.items():
                if soft_sc is not None:
                    result = block_bootstrap_shares(
                        soft_sc[0],
                        soft_sc[1],
                        codes,
                        n_reps=options.n_bootstrap,
                        seed=options.seed,
                        keep=keep,
                    )
                    rows.extend(
                        _composition_rows(
                            result,
                            sample=sample,
                            level="supercluster",
                            kind="soft",
                            region=region,
                            comparison_status=leaf_status,
                        )
                    )
                result = block_bootstrap_shares(
                    one_hot_matrix(
                        conf_sc, sc_names, include=confident(table, "supercluster")
                    ),
                    [*sc_names, UNALLOCATED],
                    codes,
                    n_reps=options.n_bootstrap,
                    seed=options.seed,
                    keep=keep,
                )
                rows.extend(
                    _composition_rows(
                        result,
                        sample=sample,
                        level="supercluster",
                        kind="confident",
                        region=region,
                        comparison_status=leaf_status,
                    )
                )
        else:
            sub_names = sorted(
                set(names_array(table, Columns.level("subclass", "name"))) - {""}
            )
            result = block_bootstrap_shares(
                one_hot_matrix(
                    names_array(table, Columns.level("subclass", "name")),
                    sub_names,
                    include=confident(table, "subclass"),
                ),
                [*sub_names, UNALLOCATED],
                codes,
                n_reps=options.n_bootstrap,
                seed=options.seed,
            )
            rows.extend(
                _composition_rows(
                    result,
                    sample=sample,
                    level="subclass",
                    kind="confident",
                    region="whole_section",
                    comparison_status=leaf_status,
                )
            )
        # Mixed/Unknown by reason: the first level's status of cells without
        # any confident level.
        first_level = LEVELS[species][0]
        final = names_array(table, Columns.CT_FINAL_LEVEL)
        unknown = final == "none"
        reasons = status_array(table, first_level)[unknown]
        values, freq = np.unique(reasons, return_counts=True)
        for value, count in zip(values, freq, strict=True):
            reason_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "platform": sample.platform,
                    "reason": f"{first_level}:{value or 'missing'}",
                    "n_cells": int(count),
                    "share_table": float(count / max(1, len(table))),
                }
            )
        if soft is not None:
            reason_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "platform": sample.platform,
                    "reason": "soft_unallocated_mass",
                    "n_cells": len(table),
                    "share_table": float(soft[:, -1].sum() / max(1.0, soft.sum())),
                }
            )
        if species == "human":
            implausible = (
                table[Columns.FLAG_IMPLAUSIBLE].to_numpy(dtype=bool)
                if Columns.FLAG_IMPLAUSIBLE in table
                else np.zeros(len(table), bool)
            )
            region_name = str(
                sample.manifest.get("anatomical_region") or "frontal_cortex"
            )
            sink_rows.extend(
                sink_rows_of(
                    sample,
                    table,
                    region_name=region_name,
                    reference_shares=reference_shares,
                    soft_sc=soft_sc,
                )
            )
            argmax = names_array(table, "mmc_whb_supercluster_name")
            item.metrics.append(
                metric(
                    "H2",
                    "flag_implausible_share",
                    _share(implausible, len(table)),
                    definition="flag_implausible share of table cells",
                    source="label_table",
                    sample_id=sample.sample_id,
                    platform=sample.platform,
                    n=len(table),
                )
            )
            cop = dict(
                (sample.summary.get("resolution") or {}).get("cop_control") or {}
            )
            opc_confident = confident(table, "broad") & (
                names_array(table, Columns.level("broad", "name")) == OPC
            )
            is_cop = argmax == COP_SUPERCLUSTER
            recomputed = {
                "confident_cop_supercluster_share": _share(
                    confident(table, "supercluster")
                    & (
                        names_array(table, Columns.level("supercluster", "name"))
                        == COP_SUPERCLUSTER
                    ),
                    len(table),
                ),
                "confident_broad_opc_share": _share(opc_confident, len(table)),
                "cop_derived_opc_share": (
                    float((opc_confident & is_cop).sum() / opc_confident.sum())
                    if opc_confident.any()
                    else math.nan
                ),
            }
            for name, value in recomputed.items():
                recorded = cop.get(name)
                item.metrics.append(
                    metric(
                        "H5",
                        name,
                        value,
                        definition=H5_DEFINITIONS[name],
                        source="label_table",
                        sample_id=sample.sample_id,
                        platform=sample.platform,
                        n=len(table),
                        note=""
                        if recorded is None
                        else f"RESOLVE recorded {float(recorded):.6g}",
                    )
                )
            # The same question on the soft (headline) and argmax
            # compositions: how much of their OPC is COP (report records).
            opc_column = classes.index(OPC)
            soft_opc = float(soft[:, opc_column].sum()) if soft is not None else 0.0
            argmax_opc = int(np.count_nonzero(argmax_broad == OPC))
            for name, numerator, denominator, kind in (
                (
                    "cop_derived_soft_opc_share",
                    float(cop_parts["soft"].sum()) if "soft" in cop_parts else math.nan,
                    soft_opc,
                    "soft",
                ),
                (
                    "cop_derived_argmax_opc_share",
                    float(is_cop.sum()),
                    float(argmax_opc),
                    "argmax",
                ),
            ):
                item.metrics.append(
                    metric(
                        "H5",
                        name,
                        numerator / denominator if denominator > 0 else math.nan,
                        definition=H5_DEFINITIONS[name],
                        source="label_table",
                        sample_id=sample.sample_id,
                        platform=sample.platform,
                        kind=kind,
                        n=len(table),
                        note=(
                            f"COP {numerator / max(1, len(table)):.4g} of OPC "
                            f"{denominator / max(1, len(table)):.4g} (shares of "
                            "table cells)"
                        ),
                    )
                )
        if Columns.DEPTH_BIN in table.columns and soft is not None:
            bins = pd.to_numeric(table[Columns.DEPTH_BIN], errors="coerce").to_numpy()
            for depth in sorted(np.unique(bins[np.isfinite(bins)])):
                selected = bins == depth
                shares = soft[selected].sum(axis=0) / max(1e-12, soft[selected].sum())
                for category, share in zip(categories, shares, strict=True):
                    depth_rows.append(
                        {
                            "sample_id": sample.sample_id,
                            "platform": sample.platform,
                            "depth_bin": int(depth),
                            "category": category,
                            "share": float(share),
                            "n_cells": int(selected.sum()),
                        }
                    )
    composition = pd.DataFrame(rows)
    headline = (
        composition[
            (composition["level"] == ("broad" if species == "human" else "class"))
            & (composition["region"] == "whole_section")
            & (composition["kind"].isin(["soft", "confident", "argmax", "soft_ge30"]))
        ]
        if len(composition)
        else composition
    )
    for row in (
        headline[headline["kind"] == "soft"].to_dict("records") if len(headline) else []
    ):
        # Human: share of all mass (the unallocated share is its own row).
        # Mouse (MO4, G4): share of the allocated mass, as mouse_resolve's
        # class_shares and §5.5 renormalise; unallocated is its own record.
        renormalised = species != "human"
        if renormalised and row["category"] == UNALLOCATED:
            continue
        item.metrics.append(
            metric(
                "report" if species == "human" else "MO4",
                "soft_share",
                row["share_renormalised"] if renormalised else row["share"],
                definition=(
                    "soft composition share of the allocated mass (table cells, "
                    "whole section; unallocated renormalised away)"
                    if renormalised
                    else "soft composition share of all mass (table cells, whole "
                    "section)"
                ),
                source="report_metrics.block_bootstrap_shares",
                sample_id=row["sample_id"],
                platform=row["platform"],
                level=row["level"],
                kind="soft",
                group=row["category"],
                ci_low=row["renormalised_ci_low"] if renormalised else row["ci_low"],
                ci_high=row["renormalised_ci_high"] if renormalised else row["ci_high"],
                n=row["n_cells"],
            )
        )
    plotted = headline.copy()
    if len(plotted):
        plotted["series"] = (
            plotted["platform"].astype(str) + " " + plotted["kind"].astype(str)
        )
        if species == "mouse":
            keep = plotted.groupby("category")["share"].max()
            plotted = plotted[plotted["category"].isin(keep[keep >= 0.005].index)]
    writer.figure(
        item,
        "broad" if species == "human" else "class",
        lambda: grouped_bars(
            plotted,
            category="category",
            group="series",
            value="share",
            low="ci_low",
            high="ci_high",
            ylabel="share of mass",
            title="Composition (whole section; soft with 95% block-bootstrap CI)",
            overlay="cop_derived_share" if species == "human" else None,
            overlay_label="COP-derived part of OPC",
        ),
        plotted,
        "Soft (headline, 95% CI over 500 µm tiles), soft >= 30 counts, confident-only "
        "and argmax composition. Hatched inside the OPC bars: the part carried by "
        "the COP supercluster (cop_derived_share), a sink while COP mass dominates "
        "OPC.",
    )
    if "shared_mask" in set(composition.get("region", pd.Series(dtype=str))):
        regions_plot = composition[
            (composition["kind"] == "soft")
            & (composition["level"] == ("broad" if species == "human" else "class"))
        ].copy()
        regions_plot["series"] = (
            regions_plot["platform"].astype(str)
            + " "
            + regions_plot["region"].astype(str)
        )
        writer.figure(
            item,
            "whole_vs_mask",
            lambda: grouped_bars(
                regions_plot,
                category="category",
                group="series",
                value="share",
                low="ci_low",
                high="ci_high",
                ylabel="share of mass",
                title="Soft composition: whole section vs shared tissue mask",
            ),
            regions_plot,
            "Soft composition on the whole section and inside the pair's shared "
            "tissue mask.",
        )
    else:
        item.notes.append(
            "Shared tissue mask not applied (absent, or coordinates not in the fixed "
            "frame)."
        )
    fine = (
        composition[
            (
                composition["level"]
                == ("supercluster" if species == "human" else "subclass")
            )
            & (composition["region"] == "whole_section")
        ]
        if len(composition)
        else composition
    )
    if len(fine):
        top = fine.groupby("category")["share"].max().sort_values(ascending=False)
        fine_plot = fine[fine["category"].isin(top.head(25).index)].copy()
        fine_plot["series"] = (
            fine_plot["platform"].astype(str) + " " + fine_plot["kind"].astype(str)
        )
        fine_plot["not_comparable"] = (
            fine_plot["comparison_status"] == "withheld_for_comparison"
        )
        writer.figure(
            item,
            "supercluster" if species == "human" else "subclass",
            lambda: grouped_bars(
                fine_plot,
                category="category",
                group="series",
                value="share",
                low="ci_low",
                high="ci_high",
                ylabel="share of mass",
                title="Leaf-level composition (top 25)",
                hatched="not_comparable",
                hatched_label="not_attempted_gate: not comparable",
            ),
            fine_plot,
            "Soft and confident composition at the leaf level (supercluster / "
            "subclass). Hatched: a sample whose dataset gate is not full "
            "(comparison_status withheld_for_comparison); its leaf-level shares "
            "are not compared across platforms (plan §5.4).",
        )
        withheld = sorted(
            set(fine.loc[fine["comparison_status"] != "comparable", "sample_id"])
        )
        if withheld:
            item.notes.append(
                "Leaf-level composition of "
                + ", ".join(withheld)
                + " is withheld for comparison (dataset gate not full): drawn "
                "hatched, not comparable with the other platform."
            )
    writer.table(item, "composition", composition)
    writer.table(item, "mixed_unknown_reasons", pd.DataFrame(reason_rows))
    writer.table(item, "depth_bins", pd.DataFrame(depth_rows))
    sinks = pd.DataFrame(sink_rows, columns=list(SINK_COLUMNS))
    writer.table(item, "sinks_implausible", sinks)
    item.summary = pd.DataFrame(reason_rows)
    return item


# --------------------------------------------------------------------------
# Item 3: confidence vs counts


def item_confidence_vs_counts(
    inputs: ReportInputs, writer: ItemWriter, options: ReportOptions
) -> ReportItem:
    """Build item 3: 2D histograms of raw bp and avg_correlation vs counts."""
    item = new_item(
        3,
        "confidence_vs_counts",
        "Confidence vs counts",
        "flat bp (P1212, P5011), saturation (mouse, large panels)",
    )
    species = inputs.species
    levels = ("broad", "supercluster") if species == "human" else ("class", "subclass")
    x_edges = np.linspace(1.0, 3.5, 26)
    y_edges = np.linspace(0.0, 1.0, 21)
    bp_panels: dict[str, pd.DataFrame] = {}
    corr_panels: dict[str, pd.DataFrame] = {}
    thresholds: dict[str, list[float]] = {}
    rows = []
    median_rows = []
    for sample in inputs.ordered_samples():
        table = table_cells(sample)
        counts = np.log10(
            np.clip(table[Columns.TOTAL_COUNTS].to_numpy(dtype=np.float64), 1.0, None)
        )
        values = (sample.manifest.get("thresholds") or {}).get("values") or {}
        for level in levels:
            raw = Columns.level(level, "raw")
            corr = Columns.level(level, "corr")
            if raw not in table.columns:
                continue
            name = f"{sample.platform} {level}"
            hist = histogram_2d(
                counts, table[raw].to_numpy(dtype=np.float64), x_edges, y_edges
            )
            bp_panels[name] = hist
            key = RAW_THRESHOLD_KEYS[species].get(level)
            if key and values.get(key) is not None:
                thresholds[name] = [float(values[key])]
            rows.append(
                hist.assign(
                    sample_id=sample.sample_id,
                    platform=sample.platform,
                    level=level,
                    metric="raw_bp",
                )
            )
            if corr in table.columns:
                chist = histogram_2d(
                    counts, table[corr].to_numpy(dtype=np.float64), x_edges, y_edges
                )
                corr_panels[name] = chist
                rows.append(
                    chist.assign(
                        sample_id=sample.sample_id,
                        platform=sample.platform,
                        level=level,
                        metric="avg_correlation",
                    )
                )
            if Columns.DEPTH_BIN in table.columns:
                grouped = table.groupby(Columns.DEPTH_BIN, observed=True)
                for depth, part in grouped:
                    median_rows.append(
                        {
                            "sample_id": sample.sample_id,
                            "platform": sample.platform,
                            "level": level,
                            "depth_bin": depth,
                            "n_cells": len(part),
                            "median_raw_bp": float(
                                np.nanmedian(part[raw].to_numpy(dtype=np.float64))
                            )
                            if len(part)
                            else math.nan,
                            "median_avg_correlation": (
                                float(
                                    np.nanmedian(part[corr].to_numpy(dtype=np.float64))
                                )
                                if corr in part and len(part)
                                else math.nan
                            ),
                            "share_raw_ge_0_9": float(
                                np.mean(part[raw].to_numpy(dtype=np.float64) >= 0.9)
                            )
                            if len(part)
                            else math.nan,
                        }
                    )
    data = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    writer.figure(
        item,
        "raw_bp",
        lambda: hist2d_panels(
            bp_panels,
            xlabel="log10 total counts",
            ylabel="raw bootstrap probability",
            thresholds=thresholds,
            title="Raw bp vs counts (dashed: raw threshold)",
        ),
        data[data["metric"] == "raw_bp"] if len(data) else data,
        "2D histogram of the raw engine probability vs log10 total counts per "
        "platform and level; dashed = the raw threshold.",
    )
    writer.figure(
        item,
        "avg_correlation",
        lambda: hist2d_panels(
            corr_panels,
            xlabel="log10 total counts",
            ylabel="avg_correlation",
            title="avg_correlation vs counts",
        ),
        data[data["metric"] == "avg_correlation"] if len(data) else data,
        "2D histogram of MapMyCells avg_correlation vs log10 total counts.",
    )
    medians = pd.DataFrame(median_rows)
    writer.table(item, "by_depth_bin", medians)
    item.summary = medians
    item.notes.append("v1.1 adds calibration curves (calibration = none in v1).")
    return item


# --------------------------------------------------------------------------
# Item 5: negative-marker purity


def item_purity(
    inputs: ReportInputs, writer: ItemWriter, options: ReportOptions
) -> ReportItem:
    """Build item 5: contamination score per label and platform, flag rate."""
    item = new_item(
        5,
        "negative_marker_purity",
        "Negative-marker purity",
        "AQP4 / GJA1 in 66-74% of Xenium Exc cells (P7513 F3)",
    )
    rows = []
    for sample in inputs.ordered_samples():
        table = table_cells(sample)
        if Columns.CONTAMINATION_SCORE not in table.columns:
            continue
        label = names_array(table, Columns.level("broad", "name"))
        keep = confident(table, "broad")
        score = table[Columns.CONTAMINATION_SCORE].to_numpy(dtype=np.float64)
        flag = (
            table[Columns.FLAG_CONTAMINATED]
            if Columns.FLAG_CONTAMINATED in table
            else pd.Series(np.nan, index=table.index)
        )
        for name in sorted(set(label[keep]) - {""}):
            selected = keep & (label == name)
            values = score[selected]
            values = values[np.isfinite(values)]
            flags = pd.to_numeric(
                flag[selected].astype("object"), errors="coerce"
            ).astype(float)
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "platform": sample.platform,
                    "label": name,
                    "n_cells": int(selected.sum()),
                    "q10": float(np.quantile(values, 0.10))
                    if values.size
                    else math.nan,
                    "q25": float(np.quantile(values, 0.25))
                    if values.size
                    else math.nan,
                    "q50": float(np.quantile(values, 0.50))
                    if values.size
                    else math.nan,
                    "q75": float(np.quantile(values, 0.75))
                    if values.size
                    else math.nan,
                    "q90": float(np.quantile(values, 0.90))
                    if values.size
                    else math.nan,
                    "mean": float(values.mean()) if values.size else math.nan,
                    "flag_rate": float(flags.mean())
                    if flags.notna().any()
                    else math.nan,
                    "n_flag_null": int(flags.isna().sum()),
                }
            )
    frame = pd.DataFrame(rows)
    for row in rows:
        item.metrics.append(
            metric(
                "report",
                "contamination_score_median",
                row["q50"],
                definition=(
                    "median contamination_score of confident broad cells of the label"
                ),
                source="label_table",
                sample_id=row["sample_id"],
                platform=row["platform"],
                group=row["label"],
                n=row["n_cells"],
            )
        )
        item.metrics.append(
            metric(
                "report",
                "flag_contaminated_rate",
                row["flag_rate"],
                definition=(
                    "flag_contaminated rate over confident broad cells with a "
                    "non-null flag"
                ),
                source="label_table",
                sample_id=row["sample_id"],
                platform=row["platform"],
                group=row["label"],
                n=row["n_cells"] - row["n_flag_null"],
            )
        )
    if frame.empty:
        item.status = "not_available"
        item.notes.append("No contamination scores in the label tables.")
    writer.figure(
        item,
        "contamination",
        lambda: box_summary(
            frame,
            category="label",
            group="platform",
            title="Contamination score per confident broad label (q10-q90)",
            ylabel="contamination_score",
        ),
        frame,
        "Negative-marker contamination score (counts on the class's negative genes / "
        "total counts) per label and platform; boxes q25-q75, whiskers q10-q90.",
    )
    item.summary = frame
    return item


# --------------------------------------------------------------------------
# Item 6: method agreement


def item_method_agreement(
    inputs: ReportInputs, writer: ItemWriter, options: ReportOptions
) -> ReportItem:
    """Build item 6: WHB vs SEA-AD agreement, tiers, spatial tier map, below-60 cost."""
    item = new_item(
        6,
        "method_agreement",
        "Method agreement",
        "method disagreement hidden in the headline; the coverage cost of the "
        "below-60 rule",
    )
    if inputs.species != "human":
        item.status = "not_applicable"
        item.notes.append("Mouse has no second vote (tier 1 everywhere; plan §7.7).")
        return item
    agreement_rows = []
    tier_rows = []
    cost_rows = []
    map_rows = []
    for sample in inputs.ordered_samples():
        table = table_cells(sample)
        if find_column(table, "mmc_seaad_subclass_name") is None:
            item.notes.append(f"{sample.sample_id}: no SEA-AD calls in the label table")
            continue
        whb = whb_broad_names(table)
        sea = seaad_broad_names(table)
        counts = table[Columns.TOTAL_COUNTS].to_numpy(dtype=np.float64)
        for level, first, second, classes in (
            ("7-class", whb, sea, HUMAN_BROAD_CLASSES),
            (
                "lineage",
                lineage_of(whb),
                lineage_of(sea),
                tuple(sorted(set(HUMAN_LINEAGE_OF_BROAD_CLASS.values()))),
            ),
        ):
            frame = agreement_by_quantile(
                first, second, counts, classes=classes, unassigned=UNASSIGNED_LABEL
            )
            frame.insert(0, "level", level)
            frame.insert(0, "platform", sample.platform)
            frame.insert(0, "sample_id", sample.sample_id)
            agreement_rows.append(frame)
        h3 = agreement_by_quantile(
            whb,
            sea,
            counts,
            classes=HUMAN_BROAD_CLASSES,
            unassigned=UNASSIGNED_LABEL,
            mask=counts >= AGREEMENT_MIN_COUNTS,
        )
        h3_strict = agreement_by_quantile(
            whb,
            sea,
            counts,
            classes=HUMAN_BROAD_CLASSES,
            unassigned=UNASSIGNED_LABEL,
            mask=counts >= AGREEMENT_MIN_COUNTS,
            strict=True,
        )
        item.metrics.append(
            metric(
                "H3",
                "whb_sea_agreement_ge20",
                h3.iloc[0]["agreement"],
                definition=(
                    "WHB 7-class (argmax supercluster -> broad) = SEA-AD "
                    "7-class (subclass -> broad, VLMC split by supertype), table "
                    "cells >= 20 counts; out-of-class labels agree"
                ),
                source="label_table",
                sample_id=sample.sample_id,
                platform=sample.platform,
                n=int(h3.iloc[0]["n_cells"]),
                note=(
                    "strict (out-of-class disagree) "
                    f"{h3_strict.iloc[0]['agreement']:.6g}"
                ),
            )
        )
        tiers = (
            pd.to_numeric(table[Columns.CT_CONSENSUS_TIER], errors="coerce").to_numpy()
            if Columns.CT_CONSENSUS_TIER in table
            else np.array([])
        )
        values, freq = np.unique(tiers[np.isfinite(tiers)], return_counts=True)
        for value, count in zip(values, freq, strict=True):
            tier_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "platform": sample.platform,
                    "tier": int(value),
                    "n_cells": int(count),
                    "share": float(count / max(1, len(table))),
                }
            )
        xy = sample.table_xy()
        if xy is not None and len(tiers):
            tiles = tile_mean_map(
                xy, np.where(tiers >= 0, tiers, np.nan), tile_um=options.tile_um
            )
            tiles.insert(0, "platform", sample.platform)
            map_rows.append(tiles)
        below = counts < SECOND_VOTE_BELOW
        # A cell lost to the vote at a level is parent_unresolved below it, so
        # the rule's cost at a level is cumulative: its first vote failure at
        # that level or at any ancestor (lineage -> broad).
        lost_so_far = np.zeros(len(table), dtype=bool)
        for level in VOTE_LEVELS:
            status = status_array(table, level)
            at_level = below & np.isin(status, list(VOTE_FAILURES))
            lost_so_far |= at_level
            cost_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "platform": sample.platform,
                    "level": level,
                    "n_below60": int(below.sum()),
                    "n_lost_below60": int(lost_so_far.sum()),
                    "share_table": _share(lost_so_far, len(table)),
                    "n_lost_at_level": int(at_level.sum()),
                    "share_table_at_level": _share(at_level, len(table)),
                }
            )
            item.metrics.append(
                metric(
                    "report",
                    "below60_rule_cost",
                    _share(lost_so_far, len(table)),
                    definition=(
                        "table cells < 60 counts whose first vote failure "
                        "(method_disagree or single_method) is at the level or an "
                        "ancestor level (lineage), / table cells: the labels the "
                        "below-60 rule costs at the level"
                    ),
                    source="label_table",
                    sample_id=sample.sample_id,
                    platform=sample.platform,
                    level=level,
                    n=len(table),
                    note=(
                        f"at the level itself {_share(at_level, len(table)):.6g} "
                        f"({int(at_level.sum())} cells)"
                    ),
                )
            )
    agreement = (
        pd.concat(agreement_rows, ignore_index=True)
        if agreement_rows
        else pd.DataFrame()
    )
    tiers_frame = pd.DataFrame(tier_rows)
    if len(agreement):
        plotted = agreement[agreement["quantile"] > 0].copy()
        plotted["series"] = (
            plotted["platform"].astype(str) + " " + plotted["level"].astype(str)
        )
        plotted["quantile_label"] = "Q" + plotted["quantile"].astype(int).astype(str)
        writer.figure(
            item,
            "agreement_by_depth",
            lambda: grouped_bars(
                plotted,
                category="quantile_label",
                group="series",
                value="agreement",
                ylabel="WHB-SEA agreement",
                title="WHB vs SEA-AD agreement by depth quartile",
            ),
            plotted,
            "Share of table cells where WHB and SEA-AD agree at 7 classes and at "
            "lineage, per total-count quartile.",
        )
    writer.figure(
        item,
        "tiers",
        lambda: grouped_bars(
            tiers_frame.assign(tier_label="tier " + tiers_frame["tier"].astype(str))
            if len(tiers_frame)
            else tiers_frame,
            category="tier_label",
            group="platform",
            value="share",
            ylabel="share of table cells",
            title="Consensus tier fractions",
        ),
        tiers_frame,
        "Consensus tier (agreement, not accuracy): 2 = WHB and SEA agree, 1 = one "
        "informative, 0 = confident disagreement, -1 = none informative.",
    )
    maps = pd.concat(map_rows, ignore_index=True) if map_rows else pd.DataFrame()
    if len(maps):
        writer.figure(
            item,
            "tier_map",
            lambda: tile_maps(
                maps,
                panel="platform",
                colour_label="mean tier",
                title="Spatial tier map (500 µm tiles)",
            ),
            maps,
            "Mean consensus tier per 500 µm tile (cells with tier >= 0).",
        )
    writer.table(item, "below60_cost", pd.DataFrame(cost_rows))
    if len(agreement):
        writer.table(item, "agreement", agreement)
    item.summary = pd.DataFrame(cost_rows)
    item.notes.append("SEA-AD vs WHB + likelihood (LL) agreement is v1.1.")
    return item


# --------------------------------------------------------------------------
# Item 8: held-out-gene enrichment


def item_heldout(
    inputs: ReportInputs, writer: ItemWriter, options: ReportOptions
) -> ReportItem:
    """Build item 8: the held-out-gene enrichment summary when it is present."""
    item = new_item(
        8,
        "heldout_genes",
        "Held-out-gene enrichment",
        "labels that the panel's own markers support only circularly",
    )
    if inputs.species == "mouse":
        rows = []
        for sample in inputs.ordered_samples():
            checks = sample.summary.get("spillover_checks") or {}
            heldout = checks.get("heldout") or {}
            fpr = checks.get("astrocyte_fpr") or {}
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "platform": sample.platform,
                    "heldout_evaluated": heldout.get("evaluated"),
                    "heldout_reason": heldout.get("reason"),
                    "enrichment": heldout.get("enrichment") or heldout.get("fold"),
                    "n_genes": checks.get("n_genes"),
                    "astrocyte_fpr": fpr.get("fpr"),
                    "n_marker_astrocytes": fpr.get("n_marker_astrocytes"),
                }
            )
            item.metrics.append(
                metric(
                    "MO5",
                    "spillover_heldout_enrichment",
                    rows[-1]["enrichment"],
                    definition=(
                        "held-out microglial-gene enrichment, flagged vs "
                        "unflagged cells (E3 test)"
                    ),
                    source="resolve_summary",
                    sample_id=sample.sample_id,
                    platform=sample.platform,
                    note=str(heldout.get("reason") or ""),
                )
            )
            item.metrics.append(
                metric(
                    "MO5",
                    "spillover_astrocyte_fpr",
                    fpr.get("fpr"),
                    definition="spill-over flag rate among marker astrocytes",
                    source="resolve_summary",
                    sample_id=sample.sample_id,
                    platform=sample.platform,
                    n=fpr.get("n_marker_astrocytes"),
                )
            )
        frame = pd.DataFrame(rows)
        writer.table(item, "spillover_checks", frame)
        item.summary = frame
        item.notes.append(
            "Class-level held-out-gene enrichment runs in the acceptance scripts (M9)."
        )
        return item
    if inputs.heldout is None or inputs.heldout.empty:
        item.status = "not_available"
        item.notes.append(
            "No held-out-gene acceptance output given "
            "(scripts/acceptance/heldout_genes.py; "
            "pass --heldout-csv)."
        )
        return item
    frame = inputs.heldout.copy()
    label_set: str | None = None
    if "label_set" in frame.columns and len(frame["label_set"].unique()) > 1:
        names = [str(name) for name in frame["label_set"].unique()]
        preferred = (
            [name for name in names if name == HELDOUT_HEADLINE_SET]
            + [name for name in HELDOUT_FALLBACK_SETS if name in names]
            + [name for name in names if "argmax" in name]
        )
        label_set = preferred[0] if preferred else names[0]
        item.notes.append(
            f"label set shown: {label_set} (all label sets in the table CSV)"
        )
        shown = frame[frame["label_set"].astype(str) == label_set]
    else:
        shown = frame
        if "label_set" in frame.columns and len(frame):
            label_set = str(frame["label_set"].iloc[0])
    if label_set is not None and label_set != HELDOUT_HEADLINE_SET:
        item.notes.append(
            f"{label_set} is not H4's scored label set (M8 D6: the WHB-only "
            f"held-out re-resolve, {HELDOUT_HEADLINE_SET}, from "
            "scripts/acceptance/resolve_criteria.py); shown for information"
        )
    for row in shown.to_dict("records"):
        for name in ("fold", "auroc"):
            item.metrics.append(
                metric(
                    "H4",
                    f"heldout_{name}",
                    row.get(name),
                    definition=(
                        f"held-out-gene enrichment {name} per class and platform (§5.8)"
                    ),
                    source="heldout_csv",
                    sample_id=f"{inputs.sources.pair_id}_{row.get('platform')}",
                    platform=str(row.get("platform")),
                    group=str(row.get("broad_class")),
                    kind=str(row.get("label_set", "")) or None,
                    n=row.get("n_assigned"),
                )
            )
    long = pd.concat(
        [
            shown.assign(metric="fold", value=shown["fold"]),
            shown.assign(metric="auroc", value=shown["auroc"]),
        ],
        ignore_index=True,
    )
    long["series"] = long["platform"].astype(str) + " " + long["metric"].astype(str)
    writer.figure(
        item,
        "enrichment",
        lambda: grouped_bars(
            long,
            category="broad_class",
            group="series",
            value="value",
            facet="metric",
            ylabel="value",
            title="Held-out-gene fold and AUROC",
        ),
        long,
        "Held-out marker fold enrichment and AUROC per class and platform "
        "(non-circular; §5.8).",
    )
    writer.table(item, "heldout", frame)
    item.summary = shown
    return item


# --------------------------------------------------------------------------
# Item 11: AD and OOD panel


def depth_matched_ratio(
    real: np.ndarray,
    real_bins: np.ndarray,
    simulated: np.ndarray,
    simulated_bins: np.ndarray,
) -> tuple[float, int]:
    """Return the real / in-silico median ratio, matched on depth bins.

    Per depth bin with both real and simulated values, the ratio of medians;
    the result weights them by the real cells per bin.

    Args:
        real: Real values (e.g. avg_correlation of a class's confident cells).
        real_bins: Their depth bins.
        simulated: In-silico values of the same class.
        simulated_bins: Their depth bins.

    Returns:
        ``(ratio, n_real)``: the weighted ratio (NaN without a shared bin)
        and the real cells in shared bins.
    """
    real = np.asarray(real, dtype=np.float64)
    simulated = np.asarray(simulated, dtype=np.float64)
    ratios = []
    weights = []
    for depth in np.unique(real_bins):
        a = real[(real_bins == depth) & np.isfinite(real)]
        b = simulated[(simulated_bins == depth) & np.isfinite(simulated)]
        if a.size and b.size and np.median(b) > 0:
            ratios.append(float(np.median(a) / np.median(b)))
            weights.append(a.size)
    if not ratios:
        return math.nan, 0
    return float(np.average(ratios, weights=weights)), int(sum(weights))


def item_ad_ood(
    inputs: ReportInputs, writer: ItemWriter, options: ReportOptions
) -> ReportItem:
    """Build item 11: real / in-silico correlation, flag_ood, disease supertypes."""
    item = new_item(
        11,
        "ad_ood",
        "AD and OOD panel",
        "disease-state and out-of-distribution cells read as neurotypical types",
    )
    if inputs.species != "human":
        item.status = "not_applicable"
        item.notes.append("Human only (SEA-AD). Mouse OOD flag rates are in item 1.")
        return item
    bundle = inputs.bundles.get(primary_reference(inputs))
    simulated = None
    if bundle is not None and (bundle / "resolvability_cells.parquet").is_file():
        try:
            simulated = pd.read_parquet(
                bundle / "resolvability_cells.parquet",
                columns=["recipe", "level", "depth", "truth", "correct", "corr"],
            )
            simulated = simulated[
                (simulated["level"] == "broad") & simulated["correct"].astype(bool)
            ]
            summary = (
                json.loads((bundle / "resolvability_summary.json").read_text())
                if (bundle / "resolvability_summary.json").is_file()
                else {}
            )
            recipe = summary.get("decision_recipe")
            if recipe is not None:
                simulated = simulated[simulated["recipe"] == recipe]
        except (ValueError, KeyError, OSError) as error:
            item.notes.append(f"in-silico cells not read: {error}")
            simulated = None
    rows = []
    disease_rows = []
    for sample in inputs.ordered_samples():
        table = table_cells(sample)
        label = names_array(table, Columns.level("broad", "name"))
        keep = confident(table, "broad")
        corr = (
            table[Columns.level("broad", "corr")].to_numpy(dtype=np.float64)
            if Columns.level("broad", "corr") in table
            else np.full(len(table), np.nan)
        )
        bins = (
            pd.to_numeric(table[Columns.DEPTH_BIN], errors="coerce").to_numpy()
            if Columns.DEPTH_BIN in table
            else np.zeros(len(table))
        )
        ood = (
            pd.to_numeric(table[Columns.FLAG_OOD].astype("object"), errors="coerce")
            .astype(float)
            .to_numpy()
            if Columns.FLAG_OOD in table
            else np.full(len(table), np.nan)
        )
        for name in HUMAN_BROAD_CLASSES:
            selected = keep & (label == name)
            ratio, n_real = (math.nan, 0)
            if simulated is not None:
                sim = simulated[simulated["truth"].astype(str) == name]
                ratio, n_real = depth_matched_ratio(
                    corr[selected],
                    bins[selected],
                    sim["corr"].to_numpy(dtype=np.float64),
                    sim["depth"].to_numpy(dtype=np.float64),
                )
            flags = ood[selected]
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "platform": sample.platform,
                    "class": name,
                    "n_confident": int(selected.sum()),
                    "median_corr_real": float(np.nanmedian(corr[selected]))
                    if selected.any()
                    else math.nan,
                    "corr_ratio_real_insilico": ratio,
                    "n_depth_matched": n_real,
                    "flag_ood_rate": float(np.nanmean(flags))
                    if np.isfinite(flags).any()
                    else math.nan,
                }
            )
            item.metrics.append(
                metric(
                    "report",
                    "corr_ratio_real_insilico",
                    ratio,
                    definition=(
                        "median avg_correlation of confident real cells / of "
                        "correct in-silico calls (bundle, decision recipe), per depth "
                        "bin, weighted by real cells"
                    ),
                    source="label_table+bundle",
                    sample_id=sample.sample_id,
                    platform=sample.platform,
                    group=name,
                    n=n_real,
                )
            )
        sea = seaad_broad_names(table)
        supertype = names_array(table, "mmc_seaad_supertype_name")
        disease = np.char.endswith(supertype.astype(str), "-SEAAD")
        for name in HUMAN_BROAD_CLASSES:
            selected = sea == name
            disease_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "platform": sample.platform,
                    "class": name,
                    "n_cells": int(selected.sum()),
                    "disease_supertype_share": _share(
                        disease & selected, int(selected.sum())
                    ),
                    "insilico_baseline": math.nan,
                    "calibrated": False,
                }
            )
            item.metrics.append(
                metric(
                    "report",
                    "seaad_disease_supertype_share",
                    disease_rows[-1]["disease_supertype_share"],
                    definition=(
                        "share of table cells of the SEA-AD class whose "
                        "supertype is a -SEAAD (disease) supertype; population level"
                    ),
                    source="label_table",
                    sample_id=sample.sample_id,
                    platform=sample.platform,
                    group=name,
                    n=int(selected.sum()),
                )
            )
    frame = pd.DataFrame(rows)
    disease_frame = pd.DataFrame(disease_rows)
    writer.figure(
        item,
        "corr_ratio",
        lambda: grouped_bars(
            frame,
            category="class",
            group="platform",
            value="corr_ratio_real_insilico",
            ylabel="real / in-silico",
            title="avg_correlation: real / in-silico (depth-matched)",
            horizontal_line=1.0,
        ),
        frame,
        "Per class, the depth-matched ratio of the real confident cells' median "
        "avg_correlation to the in-silico one; with flag_ood rates in the CSV.",
    )
    writer.figure(
        item,
        "disease_supertypes",
        lambda: grouped_bars(
            disease_frame,
            category="class",
            group="platform",
            value="disease_supertype_share",
            ylabel="share",
            title="SEA-AD disease supertypes per class (population level)",
        ),
        disease_frame,
        "Share of each SEA-AD class's cells in -SEAAD disease supertypes (calibrated "
        "= false; population level only).",
    )
    item.notes.append(
        "In-silico disease-supertype baseline: not in the bundles (E1 §6(3): 7-27% of "
        "in-silico "
        "microglia vs 24-56% real). No neurotypical composition is assumed."
    )
    item.notes.append("SEA-AD glial supertypes (OD-B3): off in v1.")
    if simulated is None:
        item.status = "partial"
        item.notes.append(
            "No in-silico cells in the primary bundle: the ratio is not available."
        )
    item.summary = frame
    return item


# --------------------------------------------------------------------------
# Item 12: provenance footer


def provenance_footer(inputs: ReportInputs) -> dict[str, Any]:
    """Return the provenance footer (item 12) as a JSON-ready dict."""
    samples = {}
    references: dict[str, dict[str, Any]] = {}
    for sample in inputs.ordered_samples():
        manifest = sample.manifest
        panel = manifest.get("panel") or {}
        engine = manifest.get("engine") or {}
        thresholds = manifest.get("thresholds") or {}
        for reference_id, record in (manifest.get("references") or {}).items():
            references.setdefault(
                str(reference_id),
                {
                    "role": record.get("role"),
                    "taxonomy_id": record.get("taxonomy_id"),
                    "build_hash": record.get("build_hash"),
                    "bundle_path": record.get("bundle_path"),
                    "lookup_sha256": (record.get("markers") or {}).get("lookup_sha256"),
                    "n_query_genes_used": record.get("n_query_genes_used"),
                    "panel_trust": record.get("panel_trust"),
                },
            )
        resolvability = {
            key: {
                "sha256": value.get("sha256"),
                "recipe": value.get("recipe"),
                "recipe_version": value.get("recipe_version"),
            }
            for key, value in (manifest.get("resolvability") or {}).items()
        }
        samples[sample.sample_id] = {
            "platform": sample.platform,
            "labels_sha256": sample.summary.get("labels_sha256"),
            "panel_hash": panel.get("panel_hash"),
            "panel_family": panel.get("panel_family"),
            "panel_mode": panel.get("panel_mode"),
            "trust_state": (sample.summary.get("trust") or {}).get("state")
            or panel.get("panel_trust"),
            "validation_basis": panel.get("validation_basis"),
            "validated_panels_sha256": panel.get("validated_panels_sha256"),
            "validated_panel_levels_sha256": panel.get("validated_panel_levels_sha256"),
            "ctm_version": engine.get("ctm_version"),
            "ctm_commit": engine.get("ctm_commit"),
            "engine_seed": engine.get("rng_seed"),
            "bootstrap_factor": engine.get("bootstrap_factor"),
            "bootstrap_iteration": engine.get("bootstrap_iteration"),
            "n_processors": engine.get("n_processors"),
            "engine_wall_time_s": engine.get("wall_time_s"),
            "calibration": thresholds.get("calibration"),
            "threshold_source": thresholds.get("threshold_source"),
            "floor_source": thresholds.get("floor_source"),
            "floors_sha256": thresholds.get("floors_sha256"),
            "resolvability": resolvability,
            "merxen_version": manifest.get("merxen_version"),
            "label_table_version": manifest.get("label_table_version"),
            "coordinate_source": sample.coordinate_source,
            "cortical_depth": None
            if sample.depth_path is None
            else str(sample.depth_path),
            "clustered_h5ad": None
            if sample.clustered_path is None
            else str(sample.clustered_path),
            "mender": _mender_digest(sample.mender_manifest),
        }
    summary_path = (
        inputs.sources.resolve_dir / f"{inputs.sources.pair_id}_resolve_summary.json"
    )
    map_manifest = inputs.map_manifest or {}
    return {
        "species": inputs.species,
        "pair_id": inputs.sources.pair_id,
        "segmentation": inputs.sources.segmentation,
        "resolve_summary_sha256": _file_sha256(summary_path)
        if summary_path.is_file()
        else None,
        "resolve_schema_version": inputs.summary.get("schema_version"),
        "map_manifest_sha256": inputs.summary.get("map_manifest_sha256"),
        "map_ctm_version": map_manifest.get("ctm_version"),
        "map_ctm_commit": map_manifest.get("ctm_commit"),
        "map_wall_time_s": map_manifest.get("wall_time_s"),
        "panel_mode": inputs.summary.get("panel_mode"),
        "references": references,
        "samples": samples,
        "thresholds": inputs.summary.get("thresholds"),
        "report_seed": 0,
        "sources": {
            key: value
            for key, value in inputs.sources.describe().items()
            if key not in ("results_root",)
        },
    }


def _mender_digest(manifest: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not manifest:
        return None
    return {
        "status": manifest.get("status", "complete"),
        "cell_state_key": manifest.get("cell_state_key"),
        "unassigned_state_policy": manifest.get("unassigned_state_policy"),
        "n_cells": manifest.get("n_cells"),
        "n_domains": manifest.get("n_domains") or _count_domains(manifest),
        "cross_platform_comparable": manifest.get("cross_platform_comparable"),
    }


def _count_domains(manifest: Mapping[str, Any]) -> int | None:
    counts = manifest.get("domain_counts")
    if isinstance(counts, Mapping):
        return len(counts)
    raw = manifest.get("domain_counts_json")
    if isinstance(raw, str):
        try:
            return len(json.loads(raw))
        except ValueError:
            return None
    return None


def item_provenance(
    inputs: ReportInputs, writer: ItemWriter, options: ReportOptions
) -> tuple[ReportItem, dict[str, Any]]:
    """Build item 12 (the provenance footer) and the H13 / H14 contract metrics."""
    item = new_item(
        12, "provenance", "Provenance", "results that cannot be traced to their inputs"
    )
    footer = provenance_footer(inputs)
    footer["report_seed"] = options.seed
    footer["report_n_bootstrap"] = options.n_bootstrap
    rows = []
    for sample_id, record in footer["samples"].items():
        for key, value in record.items():
            rows.append(
                {
                    "sample_id": sample_id,
                    "field": key,
                    "value": json.dumps(value, sort_keys=True)
                    if isinstance(value, dict | list)
                    else value,
                }
            )
    for reference_id, record in footer["references"].items():
        for key, value in record.items():
            rows.append(
                {"sample_id": f"reference:{reference_id}", "field": key, "value": value}
            )
    frame = pd.DataFrame(rows)
    writer.table(item, "provenance", frame)
    for sample in inputs.ordered_samples():
        table = table_cells(sample)
        ids_match = None
        if sample.clustered_path is not None:
            from merxen.annotation.report_inputs import read_clustered_table

            clustered = read_clustered_table(sample.clustered_path, with_counts=False)
            ids_match = set(clustered.obs_names.astype(str)) == set(
                table[Columns.CELL_ID].astype(str)
            )
        item.metrics.append(
            metric(
                "H13" if inputs.species == "human" else "report",
                "table_ids_equal_clustered",
                ids_match,
                definition=(
                    "the label table's in_table ids equal the clustered H5AD's "
                    "obs_names"
                ),
                source="label_table+clustered_h5ad",
                sample_id=sample.sample_id,
                platform=sample.platform,
                n=len(table),
            )
        )
        mender = footer["samples"][sample.sample_id]["mender"]
        item.metrics.append(
            metric(
                "H13" if inputs.species == "human" else "report",
                "mender_completed",
                None if mender is None else bool(mender.get("n_cells")),
                definition="a MENDER manifest exists for the sample with cells",
                source="mender_manifest",
                sample_id=sample.sample_id,
                platform=sample.platform,
                note="" if mender is None else json.dumps(mender, sort_keys=True),
            )
        )
        item.metrics.append(
            metric(
                "H14" if inputs.species == "human" else "MO8",
                "engine_wall_time_s",
                (sample.manifest.get("engine") or {}).get("wall_time_s"),
                definition="MapMyCells wall time of the primary run (provenance)",
                source="annotation_manifest",
                sample_id=sample.sample_id,
                platform=sample.platform,
            )
        )
    if inputs.map_manifest and inputs.map_manifest.get("wall_time_s") is not None:
        item.metrics.append(
            metric(
                "H14" if inputs.species == "human" else "MO8",
                "map_wall_time_s",
                inputs.map_manifest.get("wall_time_s"),
                definition="MAP task wall time (map_manifest.json)",
                source="map_manifest",
                scope="pair",
            )
        )
    item.summary = (
        frame[
            frame["field"].isin(
                [
                    "trust_state",
                    "panel_family",
                    "labels_sha256",
                    "ctm_version",
                    "calibration",
                ]
            )
        ]
        if len(frame)
        else frame
    )
    return item, footer


__all__ = [
    "LEVELS",
    "PRIMARY_REFERENCE",
    "argmax_classes",
    "class_categories",
    "confident",
    "confident_class_labels",
    "depth_matched_ratio",
    "item_ad_ood",
    "item_annotatability",
    "item_composition",
    "item_confidence_vs_counts",
    "item_heldout",
    "item_method_agreement",
    "item_provenance",
    "item_purity",
    "lineage_of",
    "names_array",
    "new_item",
    "provenance_footer",
    "seaad_broad_names",
    "shared_mask_of",
    "soft_matrix",
    "status_array",
    "supercluster_soft_matrix",
    "table_cells",
    "tile_codes_of",
    "whb_broad_names",
]

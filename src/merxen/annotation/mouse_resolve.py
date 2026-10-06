"""RESOLVE for one mouse sample (plan §3.4, §7.2-§7.7, §4.1-§4.6; M6).

``resolve_mouse_sample`` is the mouse counterpart of
``pipeline.resolve_human_sample``. It reads the pruned view of the WMB
primary run (``load_resolve_runs``: the region step's re-mapped cells merged
into the unpruned calls, plan §7.2), and in this order:

1. **Trust** of the primary panel (§8.2) and the calls: class and subclass
   with their bootstrap probabilities, ``avg_correlation`` and runner-ups;
   broad and NT aggregated over the class runner-ups (vocab); the level the
   region step dropped from each re-mapped cell's unpruned call.
2. **Label-free gate signals** (§7.6; ``mouse_gate``): G1 from the M0a
   registration check the caller gives; G2 the derived marker referee on
   the query counts; G3 the pre-pruning T2 share from the region record and
   its ``wmb_region_share`` bundle; G4 the soft class composition against
   the MERFISH window of ``MouseGateConfig.g4_window_sections`` (none until
   M6b's AP estimate); G5 the spill-over flag rate
   (``mouse_flags.microglial_spillover``, computed here first because it
   does not depend on the statuses). The verdict precedes the statuses.
3. **Statuses** (``consensus.resolve_mouse``; §7.3) with resolvability-gated
   emission reweighted to the dataset's soft subclass composition, the
   mouse floors and, for real-data-validated families only, the class
   ``avg_correlation`` floor ``wmb_class_min_corr`` (pre-registration §16).
4. **Flags** (§4.3): contamination, diffuse and OOD as for human on the six
   mouse broad classes; spill-over; F1 region coherence
   (``mouse_flags.region_incoherent``); Astro-Epen low count.
5. **Composition** (§5.5): ``soft_class_<token>`` rows over the 34 WMB
   classes (class bootstrap probabilities of the call and its runner-ups);
   the section's soft, confident and argmax class shares.
6. The **label table** (§4.1, validated), with the raw engine columns and
   the region columns (``mmc_wmb_unpruned_*``, ``region_pruned_changed``,
   ``inferred_region``), and the **provenance** (§4.6, ``mouse_gate``).

Pair statistics (cross-platform JSD) are not computed for mouse pairs in
v1: the only mouse data are single-platform MERSCOPE sections (M6).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd

from merxen.annotation.mapmycells_engine import level_frame, runner_up_column
from merxen.annotation.pipeline import (
    MISSING_INSTANCE_ID,
    LoadedSample,
    MapManifest,
    MapSampleRecord,
    ResolveError,
    ResolveRun,
    SampleResolution,
    aggregated_scores,
    build_sample_query,
    current_merxen_version,
    engine_columns,
    masked_objects,
    reference_provenance,
    refused_trust,
    resolvability_provenance,
    resolve_threshold_values,
    round_share,
    run_for_role,
    trust_for_run,
    version_7_outputs,
    vocab_lookup,
)
from merxen.annotation.schema import Columns, safe_token
from merxen.annotation.vocab import (
    NEURONS,
    UNASSIGNED_LABEL,
    UNRESOLVED_LABEL,
    Species,
)

if TYPE_CHECKING:
    from merxen.annotation.config import AnnotationConfig
    from merxen.annotation.consensus import LevelCall, MouseCalls
    from merxen.annotation.diagnostics import TrustDecision
    from merxen.annotation.mouse_flags import MouseFlagProfiles
    from merxen.annotation.mouse_gate import MerfishWindow, RegistrationSignal
    from merxen.annotation.mouse_regions import RegionShareBundle
    from merxen.annotation.panel import AnnotationPanel

logger = logging.getLogger(__name__)

WMB_CLASS_LEVEL: Final = "CCN20230722_CLAS"
WMB_SUBCLASS_LEVEL: Final = "CCN20230722_SUBC"
WMB_SUPERTYPE_LEVEL: Final = "CCN20230722_SUPT"
MOUSE_COMPOSITION_KINDS: Final[tuple[str, ...]] = ("soft", "confident", "argmax")


@dataclass(frozen=True)
class MouseCallInputs:
    """The per-object calls of one mouse sample, with its level frames.

    Attributes:
        calls: ``consensus.MouseCalls`` (every object).
        class_frame: The primary's class level of the table cells (tidy rows
            indexed by cell id), or ``None`` without a primary run.
        subclass_frame: Its subclass level of the table cells.
        unpruned_subclass: The unpruned subclass name per table cell.
        table_rows: Object position of each table cell.
    """

    calls: MouseCalls
    class_frame: pd.DataFrame | None
    subclass_frame: pd.DataFrame | None
    unpruned_subclass: np.ndarray
    table_rows: np.ndarray


def _levels_of(run: ResolveRun) -> tuple[str, str | None]:
    levels = list(run.bundle.levels)
    class_level = WMB_CLASS_LEVEL if WMB_CLASS_LEVEL in levels else levels[0]
    subclass_level = (
        WMB_SUBCLASS_LEVEL
        if WMB_SUBCLASS_LEVEL in levels
        else (levels[1] if len(levels) > 1 else None)
    )
    return class_level, subclass_level


def _level_call(frame: pd.DataFrame, mapped: np.ndarray) -> LevelCall:
    from merxen.annotation.consensus import LevelCall

    runner = pd.to_numeric(
        frame[runner_up_column(1, "probability")], errors="coerce"
    ).to_numpy(np.float64)
    bp = frame["bp"].to_numpy(np.float64)
    return LevelCall.of(
        masked_objects(mapped, frame["name"].astype(object).to_numpy()),
        np.where(mapped, bp, np.nan),
        corr=frame["avg_correlation"].to_numpy(np.float64),
        runner_up=frame[runner_up_column(1, "name")].astype(object).to_numpy(),
        margin=bp - np.nan_to_num(runner, nan=0.0),
    )


def marker_unsupported_keys(run: ResolveRun) -> set[tuple[str, str]]:
    """Return the ``(level, node)`` pairs a bundle lists as marker-unsupported.

    Args:
        run: The primary run (its bundle's ``builder_output``).

    Returns:
        The nodes without a marker of their own (plan §7.8, R15).
    """
    output = run.bundle.manifest.get("builder_output") or {}
    record = output.get("marker_unsupported_nodes") or {}
    details = record.get("details") if isinstance(record, Mapping) else None
    return {
        (str(item.get("level")), str(item.get("node")))
        for item in details or []
        if isinstance(item, Mapping)
    }


def _unsupported(
    frame: pd.DataFrame, level: str, nodes: set[tuple[str, str]]
) -> np.ndarray:
    labels = frame["assignment"].astype(object).to_numpy()
    return np.array(
        [
            value is not None and not pd.isna(value) and (level, str(value)) in nodes
            for value in labels
        ],
        dtype=bool,
    )


def mouse_calls_from_runs(
    loaded: LoadedSample,
    primary: ResolveRun | None,
    *,
    allow_fine_levels: bool = False,
) -> MouseCallInputs:
    """Build ``resolve_mouse``'s inputs from the (pruned) WMB tidy table (§7.3).

    Args:
        loaded: The sample's counts (every object).
        primary: The WMB annotation run (``load_resolve_runs``' pruned view).
        allow_fine_levels: Also read the report-only supertype level.

    Returns:
        The calls.

    Raises:
        ResolveError: If the run names cells absent from the sample.
    """
    from merxen.annotation.consensus import (
        LevelCall,
        LevelScores,
        MouseCalls,
        WmbCalls,
    )

    obs = pd.Index(loaded.obs_names.astype(str))
    table_rows = np.flatnonzero(loaded.in_table)
    n = len(obs)
    if primary is None:
        calls = MouseCalls(
            total_counts=np.asarray(loaded.total_counts, dtype=np.int64),
            in_table=np.asarray(loaded.in_table, dtype=bool),
            wmb=None,
        )
        return MouseCallInputs(
            calls=calls,
            class_frame=None,
            subclass_frame=None,
            unpruned_subclass=np.full(len(table_rows), None, dtype=object),
            table_rows=table_rows,
        )
    class_level, subclass_level = _levels_of(primary)
    klass = level_frame(primary.tidy, class_level)
    klass.index = klass.index.astype(str)
    unknown = klass.index.difference(obs)
    if len(unknown):
        raise ResolveError(
            f"{loaded.sample.sample_id}: run {primary.run_id} names {len(unknown)} "
            f"cells absent from the sample, e.g. {list(unknown[:3])}"
        )
    frame = klass.reindex(obs)
    mapped = frame["assignment"].notna().to_numpy()
    vocab = primary.bundle.vocab()
    broad_of = vocab_lookup(vocab, class_level, "broad_class")
    nt_of = vocab_lookup(vocab, class_level, "nt")
    nt_groups = {
        node: (value if broad_of.get(node) == NEURONS else None)
        for node, value in nt_of.items()
    }
    corr = frame["avg_correlation"].to_numpy(np.float64)
    aggregated: dict[str, LevelCall] = {}
    for key, groups in (("broad", broad_of), ("nt", nt_groups)):
        names, raw, runner, margin = aggregated_scores(frame, groups)
        raw = np.where(mapped, raw, np.nan)
        aggregated[key] = LevelCall(
            name=masked_objects(mapped, names),
            scores=LevelScores.of(raw, corr=corr, runner_up=runner, margin=margin),
        )
    subclass_call = None
    subclass_table = None
    if subclass_level is not None:
        sub = level_frame(primary.tidy, subclass_level)
        sub.index = sub.index.astype(str)
        sub_frame = sub.reindex(obs)
        subclass_call = _level_call(
            sub_frame, sub_frame["assignment"].notna().to_numpy()
        )
        subclass_table = sub.reindex(obs[table_rows])
    supertype_call = None
    if allow_fine_levels:
        level_keys = set(primary.tidy["level"].astype(str)) | set(
            primary.tidy["level_name"].astype(str)
        )
        token = (
            WMB_SUPERTYPE_LEVEL if WMB_SUPERTYPE_LEVEL in level_keys else "supertype"
        )
        if token in level_keys:
            fine = level_frame(primary.tidy, token)
            fine.index = fine.index.astype(str)
            fine = fine.reindex(obs)
            supertype_call = _level_call(fine, fine["assignment"].notna().to_numpy())
    region_dropped = None
    unpruned_frame = subclass_table
    if primary.regions is not None:
        region_dropped = primary.regions.dropped_level(obs)
        if subclass_level is not None:
            unpruned = level_frame(primary.regions.unpruned, subclass_level)
            unpruned.index = unpruned.index.astype(str)
            unpruned_frame = unpruned.reindex(obs[table_rows])
    unpruned_subclass = (
        np.full(len(table_rows), None, dtype=object)
        if unpruned_frame is None
        else np.array(
            [
                None if pd.isna(value) else str(value)
                for value in unpruned_frame["name"].astype(object)
            ],
            dtype=object,
        )
    )
    unsupported_nodes = marker_unsupported_keys(primary)
    marker_unsupported = {"class": _unsupported(frame, class_level, unsupported_nodes)}
    if subclass_level is not None:
        marker_unsupported["subclass"] = _unsupported(
            level_frame(primary.tidy, subclass_level).reindex(obs),
            subclass_level,
            unsupported_nodes,
        )
    calls = MouseCalls(
        total_counts=np.asarray(loaded.total_counts, dtype=np.int64),
        in_table=np.asarray(loaded.in_table, dtype=bool),
        marker_unsupported=marker_unsupported,
        wmb=WmbCalls(
            klass=_level_call(frame, mapped),
            broad=aggregated["broad"],
            nt=aggregated["nt"],
            subclass=subclass_call,
            supertype=supertype_call,
        ),
        region_dropped=region_dropped,
    )
    assert len(calls) == n
    return MouseCallInputs(
        calls=calls,
        class_frame=klass.reindex(obs[table_rows]),
        subclass_frame=subclass_table,
        unpruned_subclass=unpruned_subclass,
        table_rows=table_rows,
    )


# --------------------------------------------------------------------------
# Composition (§5.5)


def soft_class_matrix(
    class_frame: pd.DataFrame | None, n_rows: int, class_names: Sequence[str]
) -> np.ndarray:
    """Return per-cell soft class mass over the WMB classes (§5.5).

    Args:
        class_frame: The class level of the table cells (``None``: no call).
        n_rows: Table cells.
        class_names: The vocab's classes (column order).

    Returns:
        ``(n, len(class_names))`` bootstrap-probability mass of the call and
        its runner-ups (rows summing to at most 1; the rest is unallocated).
    """
    from merxen.annotation.composition import level_candidates

    matrix = np.zeros((n_rows, len(class_names)), dtype=np.float64)
    if class_frame is None or not n_rows:
        return matrix
    names, probabilities = level_candidates(class_frame)
    column_of = {name: index for index, name in enumerate(class_names)}
    rows = np.arange(n_rows)
    for column in range(names.shape[1]):
        index = np.array(
            [
                column_of.get(str(value), -1) if value is not None else -1
                for value in names[:, column]
            ],
            dtype=np.int64,
        )
        values = np.clip(np.nan_to_num(probabilities[:, column], nan=0.0), 0.0, 1.0)
        keep = index >= 0
        np.add.at(matrix, (rows[keep], index[keep]), values[keep])
    total = matrix.sum(axis=1)
    over = total > 1.0
    if over.any():
        matrix[over] /= total[over, None]
    return matrix


def soft_class_columns(
    matrix: np.ndarray, rows: np.ndarray, n_objects: int, class_names: Sequence[str]
) -> dict[str, np.ndarray]:
    """Return the ``soft_class_<token>`` label-table columns (§4.1).

    Args:
        matrix: ``soft_class_matrix`` rows (table cells).
        rows: Object position of each row.
        n_objects: Objects of the label table.
        class_names: Column classes.

    Returns:
        float32 columns, NaN outside the table.
    """
    columns: dict[str, np.ndarray] = {}
    for index, name in enumerate(class_names):
        column = np.full(n_objects, np.nan, dtype=np.float32)
        column[rows] = np.clip(matrix[:, index], 0.0, 1.0).astype(np.float32)
        columns[f"{Columns.SOFT_CLASS_PREFIX}{safe_token(name)}"] = column
    return columns


def class_shares(
    soft: np.ndarray,
    class_names: Sequence[str],
    *,
    argmax: Sequence[object],
    confident: Sequence[object],
) -> dict[str, dict[str, float]]:
    """Return the section's soft, confident and argmax class shares.

    Args:
        soft: Soft class rows of the table cells.
        class_names: The classes.
        argmax: The class call per table cell (``None``: none).
        confident: The confident class per table cell (``None``: none).

    Returns:
        Kind to class name to share: ``soft`` over the allocated mass
        (renormalised), ``confident`` over confident cells, ``argmax`` over
        cells with a call; ``n_cells`` and the soft ``unallocated`` share.
    """
    shares: dict[str, dict[str, float]] = {}
    n = soft.shape[0]
    mass = soft.sum(axis=0)
    total = float(mass.sum())
    shares["soft"] = {
        name: float(mass[index] / total) if total > 0 else math.nan
        for index, name in enumerate(class_names)
    }
    shares["soft"]["unallocated"] = float(1.0 - total / n) if n else math.nan
    for kind, labels in (("confident", confident), ("argmax", argmax)):
        values = pd.Series(
            [value for value in labels if value is not None], dtype=object
        )
        counts = values.value_counts()
        denominator = float(counts.sum())
        shares[kind] = {
            name: float(counts.get(name, 0) / denominator) if denominator else math.nan
            for name in class_names
        }
    for record in shares.values():
        record["n_cells"] = float(n)
    return shares


def _composition_record(
    shares: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    return {
        kind: {
            ("n_cells" if name == "n_cells" else f"share_{safe_token(name)}"): round(
                float(value), 6
            )
            for name, value in record.items()
            if value is not None and math.isfinite(float(value))
        }
        for kind, record in shares.items()
    }


# --------------------------------------------------------------------------
# Inputs of the flags and the gate


def mouse_flag_profiles(
    run: ResolveRun, gene_ids: Sequence[str], symbols: Sequence[str]
) -> MouseFlagProfiles | None:
    """Return the primary bundle's class and subclass profiles on the query genes.

    Args:
        run: The WMB primary run.
        gene_ids: Query genes.
        symbols: Their symbols.

    Returns:
        The profiles, or ``None`` when the bundle has no ``profiles.parquet``.
    """
    from merxen.annotation.mouse_flags import MouseFlagProfiles

    path = run.bundle.path / "profiles.parquet"
    if not path.is_file():
        return None
    class_level, subclass_level = _levels_of(run)
    if subclass_level is None:
        return None
    subclass_class = subclass_classes(run)
    if not subclass_class:
        return None
    table = pd.read_parquet(
        path, columns=["level", "node_name", "gene_id", "expected_fraction"]
    )
    try:
        return MouseFlagProfiles.from_table(
            table,
            gene_ids=gene_ids,
            symbols=symbols,
            class_level=class_level,
            subclass_level=subclass_level,
            subclass_class=subclass_class,
        )
    except ValueError as error:
        logger.warning("no mouse flag profiles: %s", error)
        return None


def _symbols_of(loaded: LoadedSample, gene_ids: Sequence[str]) -> list[str]:
    symbol_of: dict[str, str] = {}
    for name, gene_id in zip(loaded.feature_names, loaded.feature_ids, strict=True):
        if gene_id and gene_id not in symbol_of:
            symbol_of[gene_id] = str(name)
    return [symbol_of.get(gene_id, gene_id) for gene_id in gene_ids]


def subclass_classes(run: ResolveRun) -> dict[str, str]:
    """Return subclass name to class name of a WMB run's mapping tree.

    Args:
        run: The WMB primary run.

    Returns:
        The map (empty when the tree has no class and subclass levels).
    """
    from merxen.annotation.mouse_regions import MouseRegionError, WmbTaxonomy

    class_level, subclass_level = _levels_of(run)
    if subclass_level is None:
        return {}
    try:
        taxonomy = WmbTaxonomy.from_tree(
            run.bundle.tree(), class_level=class_level, subclass_level=subclass_level
        )
    except MouseRegionError as error:
        logger.warning("no WMB class of the subclasses: %s", error)
        return {}
    return {
        subclass: cls
        for cls, subclasses in taxonomy.subclasses.items()
        for subclass in subclasses
    }


def _region_share_bundle(
    run: ResolveRun | None,
) -> tuple[RegionShareBundle | None, str | None]:
    """The region-share bundle ``load_resolve_runs`` opened (and why not)."""
    if run is None or run.regions is None:
        return None, "no region step"
    record = run.regions.record
    if run.regions.region_share is not None:
        return run.regions.region_share, None
    if run.regions.region_share_problem is not None:
        return None, run.regions.region_share_problem
    return None, f"region step {record.status}: no region-share bundle"


def _t2_signal(
    run: ResolveRun | None,
    unpruned_subclass: np.ndarray,
    bundle: RegionShareBundle | None,
    reason: str | None,
    config: AnnotationConfig,
) -> tuple[float | None, str | None]:
    from merxen.annotation.mouse_gate import t2_share
    from merxen.annotation.mouse_regions import EVALUATED_STATUSES

    if run is None or run.regions is None:
        return None, reason or "no region step"
    record = run.regions.record
    if record.status not in EVALUATED_STATUSES:
        return None, f"region step {record.status}"
    if bundle is None:
        return None, reason
    return (
        t2_share(
            list(unpruned_subclass),
            record.present_regions,
            bundle.shares,
            subclass_class=subclass_classes(run),
            never_drop_classes=config.mouse_regions.never_drop_classes,
            min_merfish_cells=config.mouse_regions.min_merfish_cells_subclass,
            max_present_share=config.mouse_gate.g3_t2_max_present_share,
        ),
        None,
    )


def _g4_signal(
    shares: Mapping[str, float],
    bundle: RegionShareBundle | None,
    reason: str | None,
    config: AnnotationConfig,
) -> tuple[float | None, float | None, MerfishWindow | None, str | None]:
    from merxen.annotation.mouse_gate import composition_offsets, merfish_window

    sections = list(config.mouse_gate.g4_window_sections)
    if not sections:
        return None, None, None, "no MERFISH window (AP estimate: M6b)"
    if bundle is None:
        return None, None, None, reason or "no region-share bundle"
    path = Path(bundle.path) / "section_composition.parquet"
    if not path.is_file():
        return None, None, None, "region-share bundle has no section_composition"
    try:
        window = merfish_window(pd.read_parquet(path), sections)
    except ValueError as error:
        return None, None, None, str(error)
    astro, immune = composition_offsets(shares, window)
    return astro, immune, window, None


def class_corr_floor(
    config: AnnotationConfig, trust: TrustDecision | None
) -> tuple[float | None, str]:
    """Return the class ``avg_correlation`` floor RESOLVE applies, and why.

    Args:
        config: ``thresholds.wmb_class_min_corr``.
        trust: The primary panel's trust decision.

    Returns:
        ``(floor or None, note)``: the floor applies only to
        real-data-validated families (pre-registration §16).
    """
    floor = config.thresholds.wmb_class_min_corr
    if floor is None:
        return None, "no class correlation floor configured"
    if (
        trust is not None
        and trust.state == "validated"
        and trust.validation_basis == "real_data"
    ):
        return float(floor), "applied: real-data-validated family"
    return None, (
        "not applied: the floor was measured on the real-data-validated mouse "
        "families only (other panels need a simulation-derived floor, M10)"
    )


# --------------------------------------------------------------------------
# The sample


def resolve_mouse_sample(
    loaded: LoadedSample,
    record: MapSampleRecord,
    runs: Mapping[str, ResolveRun],
    config: AnnotationConfig,
    *,
    manifest: MapManifest,
    panels: Mapping[str, AnnotationPanel],
    panel_report: Mapping[str, Any] | None = None,
    trust_overrides: Mapping[str, TrustDecision] | None = None,
    xy: np.ndarray | None = None,
    registration: RegistrationSignal | None = None,
    validate: bool = True,
    seed: int = 0,
) -> SampleResolution:
    """Resolve one mouse sample (see the module docstring).

    Args:
        loaded: The sample's counts (every object, control-free).
        record: Its ``map_manifest.json`` record.
        runs: Its MAP runs (``load_resolve_runs``; pruned primary view).
        config: The annotation config (coupled to ``min_counts``).
        manifest: The pair's MAP manifest.
        panels: Panel hash to annotation panel.
        panel_report: ``panel_report.json`` content.
        trust_overrides: Trust decision per reference id.
        xy: Coordinates of every object (F1; G4 needs none).
        registration: The M0a registration check of the segmentation (G1).
        validate: Check the table with ``schema.validate_label_table``.
        seed: Seed of the diffuse-flag simulation.

    Returns:
        The sample's resolution.

    Raises:
        ResolveError: If the primary run is not the mouse primary, or the MAP
            output predates the mouse region step.
    """
    from merxen.annotation import consensus as cs
    from merxen.annotation import flags as fl
    from merxen.annotation.composition import dataset_type_composition
    from merxen.annotation.mouse_flags import (
        class_coherence_summary,
        microglial_spillover,
        null_spillover,
        region_incoherent,
    )
    from merxen.annotation.mouse_gate import (
        MouseGateSignals,
        evaluate_mouse_gate,
        marker_referee,
    )
    from merxen.annotation.mouse_regions import (
        REGION_SHARE_REFERENCE_ID,
        region_step_warnings,
    )
    from merxen.annotation.provenance import (
        AnnotationProvenance,
        ConsensusProvenance,
        EngineProvenance,
        PanelProvenance,
        ReferenceProvenance,
        ThresholdProvenance,
    )
    from merxen.annotation.resolvability import load_resolvability
    from merxen.annotation.schema import coerce_label_table_dtypes, validate_label_table
    from merxen.annotation.thresholds import EmissionPlan, FloorPlan
    from merxen.annotation.vocab import (
        MOUSE_BROAD_CLASSES,
        load_class_home_coherence,
        load_state_gene_ids,
        load_vocab,
    )

    species: Species = "mouse"
    sample_id = loaded.sample.sample_id
    platform = loaded.sample.platform.upper()
    min_counts = config.require_min_counts()
    overrides = dict(trust_overrides or {})
    primary = run_for_role(runs, "primary")
    n_objects = loaded.n_objects
    table = np.asarray(loaded.in_table, dtype=bool)
    counts = np.asarray(loaded.total_counts, dtype=np.float64)
    wmb_vocab = load_vocab("wmb_class")
    class_names = list(wmb_vocab.names)
    group_of = {
        str(name): str(value)
        for name, value in wmb_vocab.frame["broad_class"].items()
        if str(value) in MOUSE_BROAD_CLASSES
    }

    # 1. Trust and calls.
    primary_id = config.primary_reference().reference_id
    if primary is not None and primary.record.reference_id != primary_id:
        raise ResolveError(
            f"{sample_id}: primary run {primary.run_id} maps "
            f"{primary.record.reference_id}, not the mouse primary {primary_id}"
        )
    if primary is not None and primary.regions is None:
        raise ResolveError(
            f"{sample_id}: the MAP output predates the mouse region step (run "
            f"{primary.run_id} has no mouse_regions record); re-run merxen annotate"
        )
    trust: TrustDecision | None
    if primary is None:
        trust = overrides.get(primary_id) or refused_trust(
            primary_id,
            species,
            record.declared_panel_hash,
            manifest.panel_reasons
            or [f"sample panel status {record.panel_status}: no primary run"],
        )
    else:
        panel = panels.get(str(primary.record.panel_hash))
        trust = overrides.get(primary.record.reference_id) or trust_for_run(
            primary, panel, config, panel_report=panel_report
        )
    inputs = mouse_calls_from_runs(
        loaded, primary, allow_fine_levels=config.thresholds.allow_fine_levels
    )
    table_rows = inputs.table_rows
    class_call = (
        np.full(n_objects, None, dtype=object)
        if inputs.calls.wmb is None
        else inputs.calls.wmb.klass.name
    )
    class_table = class_call[table_rows]
    class_bp_table = (
        np.full(len(table_rows), np.nan)
        if inputs.calls.wmb is None
        else inputs.calls.wmb.klass.scores.raw[table_rows]
    )

    # Query counts and profiles (flags, G2).
    query_counts = None
    gene_ids: tuple[str, ...] = ()
    profiles = None
    if primary is not None:
        panel = panels.get(str(primary.record.panel_hash))
        if panel is not None:
            query = build_sample_query(loaded, panel)
            query_counts = query.counts
            gene_ids = tuple(query.gene_ids)
            profiles = mouse_flag_profiles(
                primary, gene_ids, _symbols_of(loaded, gene_ids)
            )
        else:
            logger.warning(
                "%s: no panel file for %s; mouse flags and G2 are null",
                sample_id,
                str(primary.record.panel_hash)[:16],
            )

    # 2. Label-free gate signals (§7.6).
    enabled = config.flags.microglial_spillover_enabled
    spill = (
        microglial_spillover(query_counts, profiles, config.flags)
        if enabled is not False
        else null_spillover(len(table_rows), "disabled")
    )
    referee = marker_referee(
        query_counts,
        profiles,
        list(class_table),
        group_of,
        config.flags,
        config.mouse_gate,
    )
    region_bundle, region_reason = _region_share_bundle(primary)
    t2, t2_reason = _t2_signal(
        primary, inputs.unpruned_subclass, region_bundle, region_reason, config
    )
    soft = soft_class_matrix(inputs.class_frame, len(table_rows), class_names)
    soft_only = class_shares(
        soft, class_names, argmax=list(class_table), confident=[None] * len(table_rows)
    )["soft"]
    astro_points, immune_points, window, g4_reason = _g4_signal(
        {key: value for key, value in soft_only.items() if key in set(class_names)},
        region_bundle,
        region_reason,
        config,
    )
    signals = MouseGateSignals(
        registration=registration,
        referee=referee,
        t2_share=t2,
        t2_reason=t2_reason,
        astro_epen_points=astro_points,
        immune_points=immune_points,
        window=window,
        g4_reason=g4_reason,
        spillover_rate=spill.rate,
        g5_reason=spill.null_reason,
    )
    regions = None if primary is None else primary.regions
    region_record = None if regions is None else regions.record
    gate = evaluate_mouse_gate(
        signals,
        config.mouse_gate,
        trust=trust,
        extra_warnings=(
            []
            if regions is None or region_record is None
            else region_step_warnings(
                region_record.status,
                region_record.reasons,
                share_problem=regions.region_share_problem,
            )
        ),
    )

    # 3. Statuses (§7.3).
    # Version-7 bundles too (M3c follow-up): the reweighted ensemble's
    # decisions, after the saturated-bp rule and the monotone fill.
    tables = (
        None
        if primary is None
        else load_resolvability(primary.bundle.path, allow_version_7=True)
    )
    composition = None
    reweight = bool(
        tables is not None
        and config.resolvability.reweight_to_composition
        and inputs.subclass_frame is not None
        and len(inputs.subclass_frame)
    )
    if reweight:
        assert tables is not None and inputs.subclass_frame is not None
        composition = dataset_type_composition(
            inputs.subclass_frame,
            counts[table_rows],
            tables.depth_grid,
            min_bin_mass=float(config.resolvability.composition_min_bin_cells),
        )
    emission = EmissionPlan.from_tables(
        tables,
        species=species,
        thresholds=config.thresholds,
        trust=trust,
        composition=composition,
        seed_stability_max_change=config.resolvability.seed_stability_max_change,
    )
    floors = FloorPlan.build(
        species=species,
        platform=platform,
        hard_floor=min_counts,
        trust=trust,
        thresholds=config.thresholds,
        emission=emission,
    )
    corr_floor, corr_note = class_corr_floor(config, trust)
    settings = cs.MouseResolveSettings(
        platform=platform,
        min_counts=min_counts,
        emission=emission,
        floors=floors,
        thresholds=config.thresholds,
        trust=trust,
        class_min_corr=corr_floor,
        allow_fine_levels=config.thresholds.allow_fine_levels,
        allow_table_below_min_counts=loaded.sample.source == "clustered",
    )
    resolution = cs.resolve_mouse(inputs.calls, settings, gate)

    # 4. Flags (§4.3).
    coherence = region_incoherent(
        None if xy is None else np.asarray(xy, dtype=np.float64)[table_rows],
        list(class_table),
        class_bp_table,
        config.mouse_regions,
    )
    broad = resolution.levels["broad"]
    assigned = (
        np.full(n_objects, None, dtype=object)
        if inputs.calls.wmb is None
        else np.array(
            [
                value if value in set(MOUSE_BROAD_CLASSES) else None
                for value in inputs.calls.wmb.broad.name
            ],
            dtype=object,
        )
    )
    negatives = None
    profiles_by_class = None
    if primary is not None and gene_ids:
        negatives_path = primary.bundle.path / "negative_genes.parquet"
        if negatives_path.is_file():
            try:
                negatives = fl.NegativeGeneSet.from_table(
                    pd.read_parquet(negatives_path),
                    gene_ids,
                    classes=MOUSE_BROAD_CLASSES,
                    state_gene_ids=load_state_gene_ids(species),
                    max_fraction=config.flags.negative_gene_max_fraction,
                )
            except (KeyError, ValueError) as error:
                logger.warning("%s: no negative genes (%s)", sample_id, error)
        profiles_path = primary.bundle.path / "profiles.parquet"
        if profiles_path.is_file():
            class_level, _ = _levels_of(primary)
            profiles_by_class = fl.profiles_for_classes(
                pd.read_parquet(profiles_path),
                level=class_level,
                gene_ids=gene_ids,
                node_class=group_of,
                classes=MOUSE_BROAD_CLASSES,
            )
    corr = (
        inputs.calls.wmb.klass.scores.corr
        if inputs.calls.wmb is not None
        and inputs.calls.wmb.klass.scores.corr is not None
        else np.full(n_objects, np.nan)
    )
    flag_set = fl.compute_flags(
        fl.FlagInputs(
            species=species,
            platform=platform,
            total_counts=counts,
            in_table=table,
            assigned_class=assigned,
            confident=broad.confident,
            method_disagree=resolution.flags[Columns.FLAG_METHOD_DISAGREE],
            corr=corr,
            depth_bin=resolution.depth_bin,
            exclude_hard=resolution.flags[Columns.EXCLUDE_HARD],
            query_counts=query_counts,
            query_rows=table_rows if query_counts is not None else None,
            gene_ids=gene_ids,
            negatives=negatives,
            profiles_by_class=profiles_by_class,
            wmb_class=class_call,
            spillover=spill if query_counts is not None else None,
            coherence=coherence,
        ),
        config.flags,
        class_names=MOUSE_BROAD_CLASSES,
        seed=seed,
    )

    # Version 7: the report-only non-neuronal high-depth flag and the
    # class-depth prediction at the dataset's own per-class depth.
    version_7 = version_7_outputs(
        tables, emission, resolution.levels, counts, table, config
    )

    # 5. Composition (§5.5).
    klass = resolution.levels["class"]
    shares = class_shares(
        soft,
        class_names,
        argmax=list(class_table),
        confident=[
            name if ok else None
            for name, ok in zip(
                klass.name[table_rows], klass.confident[table_rows], strict=True
            )
        ],
    )

    # 6. The label table (§4.1).
    branch = np.where(klass.confident, klass.name, UNASSIGNED_LABEL).astype(object)
    subclass = resolution.levels["subclass"]
    leaf = np.where(subclass.confident, subclass.name, UNRESOLVED_LABEL).astype(object)
    primary_panel_hash = (
        primary.record.panel_hash
        if primary is not None and primary.record.panel_hash
        else record.declared_panel_hash
    )
    n_missing = primary.record.n_missing_panel_genes if primary is not None else 0
    instance_ids = (
        np.asarray(loaded.instance_ids, dtype=np.int64)
        if loaded.instance_ids is not None
        else np.full(n_objects, MISSING_INSTANCE_ID, dtype=np.int64)
    )
    n_genes = np.asarray(loaded.n_genes, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        genes_per_count = np.where(counts > 0, n_genes / counts, np.nan)
    data: dict[str, Any] = {
        Columns.CELL_ID: loaded.obs_names.astype(str),
        Columns.INSTANCE_ID: instance_ids,
        Columns.PAIR_ID: [manifest.pair_id or ""] * n_objects,
        Columns.SAMPLE_ID: [sample_id] * n_objects,
        Columns.PLATFORM: [platform] * n_objects,
        Columns.SEGMENTATION: [manifest.segmentation or ""] * n_objects,
        Columns.SPECIES: [species] * n_objects,
        Columns.ANATOMICAL_REGION: [None] * n_objects,
        Columns.PANEL_HASH: [primary_panel_hash] * n_objects,
        Columns.TOTAL_COUNTS: counts.astype(np.int32),
        Columns.N_GENES: n_genes.astype(np.int32),
        Columns.GENES_PER_COUNT: np.clip(genes_per_count, 0.0, 1.0).astype(np.float32),
        Columns.IN_TABLE: table,
        Columns.N_MISSING_PANEL_GENES: np.full(n_objects, n_missing, dtype=np.int32),
        **resolution.to_columns(),
        Columns.CT_BRANCH: branch,
        Columns.CT_LEAF: leaf,
        Columns.CT_MENDER_STATE: branch,
        **flag_set.columns,
        Columns.FLAG_NONNEURONAL_HIGH_DEPTH: version_7.flag,
        **soft_class_columns(soft, table_rows, n_objects, class_names),
    }
    frame = pd.DataFrame(data, index=pd.RangeIndex(n_objects))
    obs = pd.Index(loaded.obs_names.astype(str))
    engine_frames = []
    for run in runs.values():
        engine = engine_columns(
            run.tidy,
            run.prefix,
            obs,
            runner_up_levels=(
                [run.leaf_level_name] if run.record.role == "primary" else []
            ),
        )
        engine_frames.append(engine.reset_index(drop=True))
        if run.regions is not None:
            class_level, subclass_level = _levels_of(run)
            if subclass_level is not None:
                engine_frames.append(
                    run.regions.engine_columns(
                        obs,
                        prefix=run.prefix,
                        class_level=class_level,
                        subclass_level=subclass_level,
                    ).reset_index(drop=True)
                )
    if engine_frames:
        frame = pd.concat([frame, *engine_frames], axis=1)
    frame = coerce_label_table_dtypes(
        frame, species, include_fine_levels=config.thresholds.allow_fine_levels
    )
    if validate:
        validate_label_table(
            frame,
            species,
            h5ad_index=(
                loaded.obs_names.astype(str)
                if loaded.sample.source == "clustered"
                else None
            ),
            include_fine_levels=config.thresholds.allow_fine_levels,
        )

    # Provenance (§4.6).
    summary = resolution.summary()
    references = {}
    resolvability = {}
    if primary is not None:
        references[primary.record.reference_id] = reference_provenance(primary, trust)
        resolvability[primary.record.reference_id] = resolvability_provenance(
            tables, emission, resolution, primary.bundle, reweighted=reweight
        )
    panel_prov = None
    if trust is not None:
        panel_prov = PanelProvenance(
            panel_hash=record.declared_panel_hash,
            panel_trust=trust.state,
            trust_reasons=trust.reason_codes,
            banner=trust.banner,
            n_missing_panel_genes=n_missing,
        )
    engine_record = primary.record if primary is not None else None
    params = engine_record.engine_params if engine_record is not None else {}
    sources = sorted(
        {
            emission.threshold_source(level)
            for level in ("broad", "class", "nt", "subclass")
        }
    )
    floor_sources = sorted({floors.policy(level) for level in ("class", "subclass")})
    if region_bundle is not None:
        references[REGION_SHARE_REFERENCE_ID] = ReferenceProvenance(
            reference_id=REGION_SHARE_REFERENCE_ID,
            role="region_share",
            bundle_path=str(region_bundle.path),
            build_hash=region_bundle.build_hash,
        )
    region_notes = (
        []
        if region_record is None
        else [
            f"region_step: {region_record.status} ({region_record.source}; "
            f"{len(region_record.nodes_to_drop)} node(s) dropped, "
            f"{region_record.n_cells_region_dropped} cell(s) re-mapped)",
            *(f"region_step: {reason}" for reason in region_record.reasons),
        ]
    )
    tiers, tier_counts = np.unique(
        resolution.consensus_tier[table].astype(int), return_counts=True
    )
    provenance = AnnotationProvenance(
        species=species,
        merxen_version=current_merxen_version(),
        panel=panel_prov,
        references=references,
        engine=EngineProvenance(
            ctm_version=None if engine_record is None else engine_record.ctm_version,
            ctm_commit=None if engine_record is None else engine_record.ctm_commit,
            bootstrap_factor=params.get("bootstrap_factor"),
            bootstrap_iteration=params.get("bootstrap_iteration"),
            rng_seed=params.get("rng_seed"),
            n_processors=params.get("n_processors"),
            wall_time_s=None if engine_record is None else engine_record.wall_s,
        ),
        resolvability=resolvability,
        thresholds=ThresholdProvenance(
            mode=config.thresholds.mode,
            values=resolve_threshold_values(config),
            threshold_source=";".join(sources),
            floors_sha256=floors.table.sha256,
            floor_source=";".join(str(item) for item in floor_sources),
            calibration="none",
        ),
        flags=flag_set.provenance(),
        mouse_gate=gate.provenance(
            notes=region_notes,
            section_regions=(
                [] if region_record is None else list(region_record.present_regions)
            ),
            region_source=(
                None
                if region_record is None
                else (
                    "explicit"
                    if region_record.source == "override"
                    else ("none" if region_record.source == "none" else "auto")
                )
            ),
            n_assigned_tiles=(
                None if region_record is None else region_record.n_assigned_tiles
            ),
            drop_list_size=(
                None if region_record is None else len(region_record.nodes_to_drop)
            ),
            drop_list_sha256=(
                None if region_record is None else region_record.drop_list_sha256
            ),
            n_remapped=(
                None if region_record is None else region_record.n_cells_region_dropped
            ),
            rule_variant=None if region_record is None else region_record.rule_variant,
        ),
        consensus=ConsensusProvenance(
            degraded_mode="wmb_only",
            methods=["wmb"] if primary is not None else [],
            max_tier=1,
            tier_counts={
                str(int(tier)): int(count)
                for tier, count in zip(tiers, tier_counts, strict=True)
            },
        ),
        composition=_composition_record(shares),
        confident_fraction_table={
            level: round(float(value["confident_share_table"]), 6)
            for level, value in summary["levels"].items()
            if value["confident_share_table"] is not None
        },
        confident_fraction_segmented={
            level: round(float(value["confident_share_segmented"]), 6)
            for level, value in summary["levels"].items()
            if value["confident_share_segmented"] is not None
        },
    )
    sample_summary = {
        "sample_id": sample_id,
        "platform": platform,
        "n_objects": n_objects,
        "n_table": int(table.sum()),
        "trust": None if trust is None else trust.to_json(),
        "banner": None if trust is None else trust.banner,
        "bundles": {
            run.run_id: {
                "reference_id": run.record.reference_id,
                "mapped_build_hash": run.record.build_hash,
                "resolved_build_hash": run.bundle.build_hash,
                "overridden": run.overridden,
                "same_lookup": run.same_lookup,
            }
            for run in runs.values()
        },
        "reweighted_to_composition": reweight,
        "class_corr_floor": {"value": corr_floor, "note": corr_note},
        "marker_unsupported_calls": {
            level: int(np.asarray(values, dtype=bool)[table].sum())
            for level, values in inputs.calls.marker_unsupported.items()
        },
        "resolution": summary,
        "mouse_gate": gate.to_json(),
        "flags": flag_set.summary(),
        "spillover_checks": spill.checks,
        "region_coherence": {
            "rate": round_share(coherence.rate),
            "restricted_classes": list(coherence.restricted_classes),
            "null_reason": coherence.null_reason,
            # Plan §7.2 step 5: class coherence vs intrinsic (MERFISH) coherence.
            "per_class": class_coherence_summary(
                coherence.coherence, list(class_table), load_class_home_coherence()
            ),
        },
        "region_step": (
            None
            if region_record is None
            else {
                "status": region_record.status,
                "source": region_record.source,
                "reasons": list(region_record.reasons),
                "present_regions": list(region_record.present_regions),
                "n_nodes_dropped": len(region_record.nodes_to_drop),
                "drop_list_sha256": region_record.drop_list_sha256,
                "n_cells_region_dropped": region_record.n_cells_region_dropped,
                "region_share_build_hash": (
                    None if region_bundle is None else region_bundle.build_hash
                ),
                "region_share_problem": (
                    None if regions is None else regions.region_share_problem
                ),
            }
        ),
        "composition_run": None if primary is None else primary.run_id,
        "composition": _composition_record(shares),
        "cross_platform": {},
    }
    if version_7.summary is not None:
        sample_summary["resolvability_v7"] = version_7.summary
    return SampleResolution(
        sample_id=sample_id,
        platform=platform,
        labels=frame,
        provenance=provenance,
        summary=sample_summary,
    )

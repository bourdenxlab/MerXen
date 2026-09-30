"""Report items 4 and 7 (plan §9): reference expectations, cross-platform concordance.

Both read the clustered H5ADs' counts (``layers['counts']``, the table cells
after control removal):

- **Item 4:** per confident leaf label, the observed fraction-positive and
  mean ``log2(CPM + 1)`` on panel genes next to the bundle's
  ``profiles.parquet``, on the label's most specific panel genes; the
  pseudobulk-centroid correlation and a correlation re-mapping of each
  label's pseudobulk to its nearest reference centroid.
- **Item 7 (human pairs):** soft broad and supercluster JSD (whole section,
  shared mask; 95% joint block bootstrap), set c (or the intersection run of
  a per-platform pair) beside set a; per-type density correlation in
  200 µm bins of the Xenium frame; per-gene platform log2 ratios within
  confident labels (``<pair>_platform_gene_factors.csv``); per-label
  pseudobulk correlation. Every statement follows RESOLVE's cross-platform
  scope (§8.5): withheld levels and kinds are recorded as ``withheld``.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from merxen.annotation.composition import (
    COMPOSITION_KINDS,
    block_bootstrap_jsd,
    pair_jsd,
    section_composition,
    shared_tile_codes,
    tile_codes,
    tile_sums,
)
from merxen.annotation.report_figures import dotplot, grouped_bars, line_facets
from merxen.annotation.report_inputs import (
    ClusteredTable,
    ReportInputs,
    SampleData,
    read_clustered_table,
    resolved_gene_ids,
)
from merxen.annotation.report_items import (
    confident,
    names_array,
    new_item,
    primary_reference,
    shared_mask_of,
    soft_matrix,
    supercluster_soft_matrix,
    table_cells,
    whb_broad_names,
)
from merxen.annotation.report_metrics import (
    aligned_bin_density_correlation,
    group_mean_counts,
    log_cpm,
    nearest_centroid,
    one_hot_matrix,
    pearson_r,
    platform_gene_log2_ratios,
    soft_level_matrix,
)
from merxen.annotation.report_model import ItemWriter, ReportItem, ReportOptions, metric
from merxen.annotation.schema import Columns
from merxen.annotation.vocab import HUMAN_BROAD_CLASSES, primary_vocab

logger = logging.getLogger(__name__)

LEAF_LEVEL: Final[dict[str, str]] = {"human": "supercluster", "mouse": "subclass"}
PROFILE_LEVEL_SUFFIX: Final[dict[str, str]] = {
    "supercluster": "_SUPC",
    "subclass": "_SUBC",
    "class": "_CLAS",
}
MIN_LABEL_CELLS: Final = 20
SETC_PREFIX: Final = "mmc_whb_setc"
XPANEL_PREFIX: Final = "mmc_whb_xpanel"
# RESOLVE's jsd_purpose of a per_platform pair (merxen.annotation.pipeline
# XPANEL_PURPOSE) and the kinds it compares (XPANEL_KINDS).
XPANEL_PURPOSE: Final = "intersection_xpanel"
XPANEL_KINDS: Final[tuple[str, ...]] = ("soft", "soft_ge30", "argmax")
OWN_PANEL_REASON: Final = "own_panel_not_comparable"
SENSITIVITY_PREFIXES: Final[tuple[tuple[str, str], ...]] = (
    (SETC_PREFIX, "set_c"),
    (XPANEL_PREFIX, "intersection_run"),
)
# Set c (and any other sensitivity run) has no resolved labels: every kind
# but confident, beside set a (plan §9 item 7).
SENSITIVITY_KINDS: Final[tuple[str, ...]] = ("soft", "soft_ge30", "argmax")


def _counts_for(
    sample: SampleData,
    table: pd.DataFrame,
    cache: dict[str, ClusteredTable],
    lookup: Mapping[str, str] | None = None,
) -> tuple[Any, list[str], np.ndarray] | None:
    """Return the table cells' counts (``table`` order), gene ids and found mask."""
    if sample.clustered_path is None:
        return None
    if sample.sample_id not in cache:
        cache[sample.sample_id] = read_clustered_table(sample.clustered_path)
    clustered = cache[sample.sample_id]
    position = clustered.obs_names.get_indexer(table[Columns.CELL_ID].astype(str))
    found = position >= 0
    if not found.any():
        return None
    matrix = clustered.counts[position[found]]
    return matrix, resolved_gene_ids(clustered.gene_ids, lookup or {}), found


def _cell_log_cpm_stats(
    matrix: Any, groups: np.ndarray, order: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-group fraction positive and mean per-cell ``log2(CPM + 1)``."""
    from scipy import sparse

    counts = sparse.csr_matrix(matrix, dtype=np.float64)
    totals = np.asarray(counts.sum(axis=1)).ravel()
    scale = np.divide(1e6, totals, out=np.zeros_like(totals), where=totals > 0)
    cpm = sparse.diags(scale) @ counts
    logged = cpm.copy()
    logged.data = np.log2(logged.data + 1.0)
    positive = counts.copy()
    positive.data = (positive.data > 0).astype(np.float64)
    fraction, _ = group_mean_counts(positive, groups, order)
    mean_log, _ = group_mean_counts(logged, groups, order)
    return fraction, mean_log


def _profiles(bundle: Path | None, level: str) -> pd.DataFrame | None:
    if bundle is None or not (bundle / "profiles.parquet").is_file():
        return None
    frame = pd.read_parquet(bundle / "profiles.parquet")
    suffix = PROFILE_LEVEL_SUFFIX.get(level)
    if suffix is not None and "level" in frame.columns:
        frame = frame[frame["level"].astype(str).str.endswith(suffix)]
    return frame if len(frame) else None


def specific_genes(
    profiles: pd.DataFrame, labels: Sequence[str], genes: Sequence[str], per_label: int
) -> list[str]:
    """Return each label's most specific panel genes (reference detection).

    Specificity = the label's detection fraction minus the largest detection
    fraction of any other node, over the genes present in the data.

    Args:
        profiles: ``profiles.parquet`` rows of one level.
        labels: Node names to pick genes for.
        genes: Gene ids present in the data.
        per_label: Genes per label.

    Returns:
        The union of the picked genes, in label order.
    """
    table = profiles[profiles["gene_id"].isin(set(genes))].pivot_table(
        index="node_name",
        columns="gene_id",
        values="detection_fraction",
        aggfunc="mean",
    )
    picked: list[str] = []
    for label in labels:
        if label not in table.index:
            continue
        others = table.drop(index=label)
        specificity = table.loc[label] - (others.max(axis=0) if len(others) else 0.0)
        for gene in specificity.sort_values(ascending=False).index[:per_label]:
            if gene not in picked:
                picked.append(str(gene))
    return picked


def item_reference_expectation(
    inputs: ReportInputs, writer: ItemWriter, options: ReportOptions
) -> ReportItem:
    """Build item 4: observed vs reference dotplots, pseudobulk r and re-mapping."""
    item = new_item(
        4,
        "reference_expectation",
        "Reference-expectation dotplots",
        "Xenium 'Microglia' lacking P2RY12; astrocyte spill into neurons",
    )
    if not options.read_expression:
        item.status = "not_available"
        item.notes.append("Expression not read (read_expression off).")
        return item
    level = LEAF_LEVEL[inputs.species]
    bundle = inputs.bundles.get(primary_reference(inputs))
    profiles = _profiles(bundle, level)
    if profiles is None:
        item.status = "not_available"
        item.notes.append(f"No {level} profiles in the primary bundle.")
        return item
    cache: dict[str, ClusteredTable] = {}
    pseudo_rows = []
    any_sample = False
    ref_nodes = sorted(profiles["node_name"].astype(str).unique())
    for sample in inputs.ordered_samples():
        table = table_cells(sample)
        loaded = _counts_for(sample, table, cache, inputs.gene_lookup)
        if loaded is None:
            item.notes.append(f"{sample.sample_id}: no clustered H5AD counts")
            continue
        matrix, gene_ids, found = loaded
        labels = np.where(
            confident(table, level),
            names_array(table, Columns.level(level, "name")),
            "",
        )[found]
        values, freq = np.unique(labels[labels != ""], return_counts=True)
        order = [
            str(v)
            for v, n in sorted(
                zip(values, freq, strict=True), key=lambda kv: (-kv[1], kv[0])
            )
            if n >= MIN_LABEL_CELLS
        ]
        order = [name for name in order if name in set(ref_nodes)][
            : options.max_dotplot_labels
        ]
        if not order:
            item.notes.append(
                f"{sample.sample_id}: no confident {level} label with >= "
                f"{MIN_LABEL_CELLS} cells and a reference profile"
            )
            continue
        any_sample = True
        genes = specific_genes(profiles, order, gene_ids, options.genes_per_label)
        gene_index = {gene: index for index, gene in enumerate(gene_ids)}
        fraction, mean_log = _cell_log_cpm_stats(matrix, labels, order)
        ref = profiles.set_index(["node_name", "gene_id"])
        symbol_of = dict(
            zip(gene_ids, _symbols_for(sample, cache, gene_ids), strict=True)
        )
        rows = []
        for li, label in enumerate(order):
            for gene in genes:
                gi = gene_index[gene]
                key = (label, gene)
                rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "platform": sample.platform,
                        "label": label,
                        "gene_id": gene,
                        "gene": symbol_of.get(gene, gene),
                        "obs_fraction": float(fraction[li, gi]),
                        "obs_mean_log2cpm": float(mean_log[li, gi]),
                        "ref_fraction": float(ref.loc[key, "detection_fraction"])
                        if key in ref.index
                        else math.nan,
                        "ref_mean_log2cpm": float(ref.loc[key, "mean_log2cpm"])
                        if key in ref.index
                        else math.nan,
                    }
                )
        frame = pd.DataFrame(rows)
        writer.figure(
            item,
            f"dotplot_{sample.platform.lower()}",
            lambda frame=frame, platform=sample.platform, sid=sample.sample_id: dotplot(
                frame,
                label="label",
                gene="gene",
                size_columns=("obs_fraction", "ref_fraction"),
                colour_columns=("obs_mean_log2cpm", "ref_mean_log2cpm"),
                panel_titles=(f"{platform} observed", "reference profile"),
                title=f"{sid}: confident {level} labels on their specific genes",
            ),
            frame,
            f"{sample.platform}: observed fraction positive (size) and mean "
            f"log2(CPM+1) (colour) per confident {level} label, beside the "
            "reference profile.",
        )
        # Pseudobulk vs reference centroids over all shared genes.
        shared = [gene for gene in gene_ids if gene in set(profiles["gene_id"])]
        means, n_cells = group_mean_counts(matrix, labels, order)
        columns = [gene_index[gene] for gene in shared]
        pseudobulk = log_cpm(means[:, columns])
        centroids = (
            profiles[profiles["gene_id"].isin(set(shared))]
            .pivot_table(
                index="node_name", columns="gene_id", values="mean_cpm", aggfunc="mean"
            )
            .reindex(columns=shared)
        )
        centroid_log = log_cpm(centroids.to_numpy(dtype=np.float64))
        centroid_names = [str(name) for name in centroids.index]
        for li, label in enumerate(order):
            own = centroid_names.index(label) if label in centroid_names else None
            r_own = (
                pearson_r(pseudobulk[li], centroid_log[own])
                if own is not None
                else math.nan
            )
            best, r_best = nearest_centroid(
                pseudobulk[li], centroid_log, centroid_names
            )
            pseudo_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "platform": sample.platform,
                    "label": label,
                    "n_cells": int(n_cells[li]),
                    "n_genes": len(shared),
                    "pseudobulk_centroid_r": r_own,
                    "nearest_centroid": best,
                    "nearest_centroid_r": r_best,
                    "remaps_to_itself": best == label,
                }
            )
            item.metrics.append(
                metric(
                    "report",
                    "pseudobulk_centroid_r",
                    r_own,
                    definition=(
                        "Pearson r of log2(CPM+1) of the label's pseudobulk "
                        "and its reference centroid over shared panel genes"
                    ),
                    source="clustered_h5ad+bundle",
                    sample_id=sample.sample_id,
                    platform=sample.platform,
                    level=level,
                    group=label,
                    n=int(n_cells[li]),
                    note=f"nearest centroid {best} (r {r_best:.4g})",
                )
            )
    pseudo = pd.DataFrame(pseudo_rows)
    if len(pseudo):
        writer.figure(
            item,
            "pseudobulk_r",
            lambda: grouped_bars(
                pseudo,
                category="label",
                group="platform",
                value="pseudobulk_centroid_r",
                ylabel="Pearson r",
                title="Pseudobulk vs reference centroid",
            ),
            pseudo,
            "Per confident label, the correlation of its pseudobulk with its "
            "reference centroid; the CSV adds the nearest-centroid re-mapping.",
        )
        remapped = pseudo["remaps_to_itself"].astype(bool)
        item.notes.append(
            "pseudobulk re-mapping (nearest reference centroid by correlation): "
            f"{int(remapped.sum())} of {len(pseudo)} labels map to themselves"
        )
    if not any_sample:
        item.status = "not_available"
    item.summary = pseudo
    return item


def _symbols_for(
    sample: SampleData, cache: dict[str, ClusteredTable], gene_ids: list[str]
) -> list[str]:
    clustered = cache.get(sample.sample_id)
    if clustered is None or len(clustered.gene_symbols) != len(gene_ids):
        return list(gene_ids)
    return list(clustered.gene_symbols)


# --------------------------------------------------------------------------
# Item 7: cross-platform concordance (human)


def _scope(inputs: ReportInputs) -> Any:
    from merxen.clustering.cross_platform import scope_from_resolve_summary

    return scope_from_resolve_summary(inputs.summary)


def _section(
    sample: SampleData,
    table: pd.DataFrame,
    soft: np.ndarray,
    prefix: str = "mmc_whb",
    *,
    with_confident: bool = True,
) -> Any:
    """Return a section's composition rows on one engine run's columns.

    ``with_confident=False`` leaves the confident kind empty: an
    intersection-panel run has no resolved (confident) labels, and the
    own-panel ones must not be compared across platforms (plan §8.5).
    """
    n_cells = len(table)
    return section_composition(
        sample.sample_id,
        sample.platform,
        soft,
        total_counts=table[Columns.TOTAL_COUNTS].to_numpy(dtype=np.float64),
        argmax_broad=whb_broad_names(table, prefix),
        confident_broad=names_array(table, Columns.level("broad", "name"))
        if with_confident
        else np.full(n_cells, None, dtype=object),
        confident=confident(table, "broad")
        if with_confident
        else np.zeros(n_cells, dtype=bool),
        xy=sample.table_xy(),
        aligned_frame=sample.aligned_frame,
    )


def _soft_broad_from_prefix(table: pd.DataFrame, prefix: str) -> np.ndarray | None:
    """Soft broad rows of an engine run's supercluster columns (set c, xpanel)."""
    if f"{prefix}_supercluster_name" not in table.columns:
        return None
    vocab = primary_vocab("human")
    mapping = {name: vocab.broad_class(name) for name in vocab.names}
    runner_names = []
    runner_bps = []
    for rank in range(1, 6):
        name = f"{prefix}_supercluster_runner_up_{rank}_name"
        bp = f"{prefix}_supercluster_runner_up_{rank}_bp"
        if name in table.columns and bp in table.columns:
            runner_names.append(names_array(table, name))
            runner_bps.append(table[bp].to_numpy(dtype=np.float64))
    return soft_level_matrix(
        names_array(table, f"{prefix}_supercluster_name"),
        table[f"{prefix}_supercluster_bp"].to_numpy(dtype=np.float64),
        runner_names,
        runner_bps,
        categories=HUMAN_BROAD_CLASSES,
        mapping=mapping,
    )


def _jsd_rows(records: Sequence[Any], variant: str, level: str) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        payload = record.to_json()
        payload["variant"] = variant
        payload["level"] = level
        rows.append(payload)
    return rows


@dataclass(frozen=True)
class CrossPlatformBasis:
    """Which engine run feeds a pair's cross-platform statements (§8.5).

    Attributes:
        per_platform: The pair is annotated on two different panels.
        prefix: Label-table column prefix of the headline run
            (``mmc_whb`` for a same-panel pair, ``mmc_whb_xpanel``: the WHB
            run on the intersection panel).
        variant: Its name in the report (``set_a`` / ``intersection_run``).
        kinds: The composition kinds it compares (RESOLVE's).
        label_kind: The per-cell labels of the density correlation, gene
            factors and pseudobulk r (``confident`` broad on a same-panel
            pair; the intersection run's ``argmax`` broad class otherwise).
    """

    per_platform: bool
    prefix: str
    variant: str
    kinds: tuple[str, ...]
    label_kind: str


def cross_platform_basis(scope: Any) -> CrossPlatformBasis:
    """Return the run a pair's cross-platform statements come from.

    Args:
        scope: The pair's ``CrossPlatformScope``.

    Returns:
        The intersection-panel run for a ``per_platform`` pair (panel mode
        or RESOLVE's ``jsd_purpose``), the annotation run otherwise.
    """
    per_platform = (
        scope.panel_mode == "per_platform" or scope.jsd_purpose == XPANEL_PURPOSE
    )
    if per_platform:
        kinds = tuple(kind for kind in scope.kinds if kind != "confident")
        return CrossPlatformBasis(
            True, XPANEL_PREFIX, "intersection_run", kinds or XPANEL_KINDS, "argmax"
        )
    return CrossPlatformBasis(
        False, "mmc_whb", "set_a", tuple(scope.kinds) or COMPOSITION_KINDS, "confident"
    )


def supercluster_pair_jsd(
    first: tuple[np.ndarray, np.ndarray | None, bool],
    second: tuple[np.ndarray, np.ndarray | None, bool],
    *,
    keep_a: np.ndarray | None,
    keep_b: np.ndarray | None,
    tile_um: float,
    n_reps: int,
    seed: int,
) -> dict[str, Any]:
    """Return the soft supercluster JSD of a pair with its joint block-bootstrap CI.

    Args:
        first: ``(matrix, xy, aligned)`` of the first section (supercluster
            columns plus ``unallocated`` last).
        second: The same for the second section.
        keep_a: Cells of the first section in the region.
        keep_b: Cells of the second section in the region.
        tile_um: Tile edge.
        n_reps: Replicates.
        seed: Seed.

    Returns:
        ``jsd``, ``ci_low``, ``ci_high``, ``resampling``, ``n_tiles``.
    """
    matrix_a, xy_a, aligned_a = first
    matrix_b, xy_b, aligned_b = second
    keep_a = np.ones(len(matrix_a), bool) if keep_a is None else keep_a
    keep_b = np.ones(len(matrix_b), bool) if keep_b is None else keep_b
    columns = list(range(matrix_a.shape[1] - 1))
    a = matrix_a[keep_a]
    b = matrix_b[keep_b]
    from merxen.annotation.report_metrics import jensen_shannon_distance

    point = jensen_shannon_distance(
        a[:, columns].sum(axis=0), b[:, columns].sum(axis=0)
    )
    out: dict[str, Any] = {
        "jsd": point,
        "ci_low": math.nan,
        "ci_high": math.nan,
        "resampling": None,
        "n_tiles": 0,
    }
    if xy_a is None or xy_b is None:
        return out
    if aligned_a and aligned_b:
        codes_a, codes_b, n_tiles = shared_tile_codes(
            xy_a[keep_a], xy_b[keep_b], tile_um
        )
        result = block_bootstrap_jsd(
            tile_sums(a, codes_a, n_tiles),
            tile_sums(b, codes_b, n_tiles),
            n_reps=n_reps,
            seed=seed,
            columns=columns,
        )
    else:
        result = block_bootstrap_jsd(
            tile_sums(a, tile_codes(xy_a[keep_a], tile_um)),
            tile_sums(b, tile_codes(xy_b[keep_b], tile_um)),
            n_reps=n_reps,
            seed=seed,
            columns=columns,
            resampling="independent",
        )
    out.update(
        ci_low=result.ci_low,
        ci_high=result.ci_high,
        resampling=result.resampling,
        n_tiles=result.n_locations,
    )
    return out


def _withheld_row(
    variant: str, level: str, kind: str, region: str, note: str
) -> dict[str, Any]:
    return {
        "kind": kind,
        "region": region,
        "variant": variant,
        "level": level,
        "jsd": None,
        "allowed": False,
        "note": note,
    }


def _broad_jsd_rows(
    inputs: ReportInputs,
    scope: Any,
    basis: CrossPlatformBasis,
    pair: tuple[SampleData, SampleData],
    tables: tuple[pd.DataFrame, pd.DataFrame],
    mask: Any,
    options: ReportOptions,
) -> list[dict[str, Any]]:
    """Return the broad JSD rows: the headline run, withheld rows, sensitivities."""
    first, second = pair
    table_a, table_b = tables
    regions = ["whole_section"] + (["shared_mask"] if mask is not None else [])
    recorded = {
        (row.get("kind"), row.get("region")): row
        for row in (inputs.summary.get("pair") or {}).get("jsd") or []
    }
    if basis.per_platform:
        soft_a = _soft_broad_from_prefix(table_a, basis.prefix)
        soft_b = _soft_broad_from_prefix(table_b, basis.prefix)
    else:
        soft_a, soft_b = soft_matrix(table_a, "human"), soft_matrix(table_b, "human")
    rows: list[dict[str, Any]] = []
    if soft_a is None or soft_b is None:
        return rows
    records, _ = pair_jsd(
        _section(
            first, table_a, soft_a, basis.prefix, with_confident=not basis.per_platform
        ),
        _section(
            second,
            table_b,
            soft_b,
            basis.prefix,
            with_confident=not basis.per_platform,
        ),
        mask=mask,
        kinds=basis.kinds,
        tile_um=options.tile_um,
        n_reps=options.n_bootstrap,
        seed=options.seed,
    )
    headline = _jsd_rows(records, basis.variant, "broad")
    for row in headline:
        # RESOLVE's pair JSD comes from the same run as the headline (the
        # intersection run of a per_platform pair; §8.5).
        resolve = recorded.get((row["kind"], row["region"]))
        row["resolve_jsd"] = None if resolve is None else resolve.get("jsd")
        row["resolve_ci_low"] = None if resolve is None else resolve.get("ci_low")
        row["resolve_ci_high"] = None if resolve is None else resolve.get("ci_high")
        row["allowed"] = bool(scope.allows_kind(row["kind"])) or not scope.kinds
    rows.extend(headline)
    if basis.per_platform:
        for region in regions:
            rows.append(
                _withheld_row(
                    basis.variant,
                    "broad",
                    "confident",
                    region,
                    "withheld: the intersection run has no confident labels and "
                    "the own-panel ones rest on different gene sets (§8.5)",
                )
            )
            for kind in COMPOSITION_KINDS:
                rows.append(
                    _withheld_row(
                        "set_a",
                        "broad",
                        kind,
                        region,
                        f"{OWN_PANEL_REASON}: a per_platform pair's own-panel "
                        "compositions are never compared across platforms (§8.5)",
                    )
                )
    for prefix, variant in SENSITIVITY_PREFIXES:
        if prefix == basis.prefix or (basis.per_platform and variant == "set_c"):
            continue
        sens_a = _soft_broad_from_prefix(table_a, prefix)
        sens_b = _soft_broad_from_prefix(table_b, prefix)
        if sens_a is None or sens_b is None:
            continue
        records, _ = pair_jsd(
            _section(first, table_a, sens_a, prefix, with_confident=False),
            _section(second, table_b, sens_b, prefix, with_confident=False),
            mask=mask,
            kinds=SENSITIVITY_KINDS,
            tile_um=options.tile_um,
            n_reps=options.n_bootstrap,
            seed=options.seed,
        )
        rows.extend(_jsd_rows(records, variant, "broad"))
    return rows


def _supercluster_jsd_rows(
    scope: Any,
    basis: CrossPlatformBasis,
    pair: tuple[SampleData, SampleData],
    tables: tuple[pd.DataFrame, pd.DataFrame],
    masks: tuple[np.ndarray | None, np.ndarray | None] | None,
    options: ReportOptions,
) -> list[dict[str, Any]]:
    """Return the soft supercluster JSD rows of the headline run and set c."""
    first, second = pair
    table_a, table_b = tables
    runs = [(basis.prefix, basis.variant)]
    if not basis.per_platform:
        runs.append((SETC_PREFIX, "set_c"))
    regions: dict[str, tuple[np.ndarray | None, np.ndarray | None]] = {
        "whole_section": (None, None)
    }
    if masks is not None:
        regions["shared_mask"] = masks
    rows: list[dict[str, Any]] = []
    for prefix, variant in runs:
        sc_a = supercluster_soft_matrix(table_a, prefix)
        sc_b = supercluster_soft_matrix(table_b, prefix)
        if sc_a is None or sc_b is None:
            continue
        for region, (keep_a, keep_b) in regions.items():
            if not scope.allows("supercluster"):
                rows.append(
                    _withheld_row(
                        variant,
                        "supercluster",
                        "soft",
                        region,
                        "withheld by the cross-platform scope",
                    )
                )
                continue
            result = supercluster_pair_jsd(
                (sc_a[0], first.table_xy(), first.aligned_frame),
                (sc_b[0], second.table_xy(), second.aligned_frame),
                keep_a=keep_a,
                keep_b=keep_b,
                tile_um=options.tile_um,
                n_reps=options.n_bootstrap,
                seed=options.seed,
            )
            rows.append(
                {
                    "kind": "soft",
                    "region": region,
                    "variant": variant,
                    "level": "supercluster",
                    **result,
                    "allowed": True,
                }
            )
    if basis.per_platform and supercluster_soft_matrix(table_a) is not None:
        for region in regions:
            rows.append(
                _withheld_row(
                    "set_a",
                    "supercluster",
                    "soft",
                    region,
                    f"{OWN_PANEL_REASON}: own-panel runs of a per_platform pair",
                )
            )
    return rows


def _pair_labels(table: pd.DataFrame, basis: CrossPlatformBasis) -> np.ndarray:
    """Return each cell's broad label for the density, factor and pseudobulk rows."""
    if basis.label_kind == "argmax":
        return np.asarray(whb_broad_names(table, basis.prefix), dtype=object)
    return np.where(
        confident(table, "broad"),
        names_array(table, Columns.level("broad", "name")),
        "",
    )


def item_cross_platform(
    inputs: ReportInputs, writer: ItemWriter, options: ReportOptions
) -> ReportItem:
    """Build item 7: cross-platform concordance of a human pair.

    A ``per_platform`` pair (plan §8.5) takes every cross-platform statement
    from the WHB run on the intersection panel (``mmc_whb_xpanel``): the
    headline JSD (H1, with RESOLVE's value from the same run), the density
    correlation, the gene factors and the pseudobulk r (on its argmax broad
    labels); the own-panel comparisons are recorded as withheld
    (``own_panel_not_comparable``). A same-panel pair uses set a, with set c
    beside it.
    """
    item = new_item(
        7,
        "cross_platform",
        "Cross-platform concordance",
        "platform-specific calling, anatomy mismatch, gene-level platform offsets",
    )
    samples = inputs.ordered_samples()
    if inputs.species != "human":
        item.status = "not_applicable"
        item.notes.append("Human pairs only (mouse: MERFISH composition, item 10).")
        return item
    if len(samples) != 2:
        item.status = "not_applicable"
        item.notes.append("Needs a pair of sections (MERSCOPE and Xenium).")
        return item
    scope = _scope(inputs)
    basis = cross_platform_basis(scope)
    item.notes.append(
        f"cross-platform scope: {scope.statistics_level} (flag {scope.flag}; "
        f"reasons {', '.join(scope.reasons) or 'none'}); panel mode "
        f"{scope.panel_mode or 'unknown'}; headline run {basis.prefix} "
        f"({basis.variant}; RESOLVE jsd_purpose {scope.jsd_purpose or 'unknown'}, "
        f"runs {', '.join(scope.jsd_runs) or 'unknown'})"
    )
    first, second = samples
    table_a, table_b = table_cells(first), table_cells(second)
    mask_a, mask_b = shared_mask_of(inputs, first), shared_mask_of(inputs, second)
    mask = inputs.mask if mask_a is not None and mask_b is not None else None
    jsd_rows: list[dict[str, Any]] = []
    if scope.allows("broad"):
        jsd_rows.extend(
            _broad_jsd_rows(
                inputs,
                scope,
                basis,
                (first, second),
                (table_a, table_b),
                mask,
                options,
            )
        )
        if basis.per_platform and not any(
            row["variant"] == basis.variant and row.get("jsd") is not None
            for row in jsd_rows
        ):
            item.notes.append(
                "per_platform pair without an intersection-panel run on both "
                "platforms: no cross-platform composition (plan §8.5)"
            )
    else:
        item.notes.append("broad-level cross-platform statistics withheld by the scope")
    jsd_rows.extend(
        _supercluster_jsd_rows(
            scope,
            basis,
            (first, second),
            (table_a, table_b),
            (mask_a, mask_b) if mask is not None else None,
            options,
        )
    )
    jsd = pd.DataFrame(jsd_rows)
    _jsd_metrics(item, jsd_rows, basis)
    plotted = jsd[jsd["jsd"].notna()].copy() if len(jsd) else jsd
    if len(plotted):
        plotted["series"] = (
            plotted["variant"].astype(str) + " " + plotted["region"].astype(str)
        )
        plotted["category"] = (
            plotted["level"].astype(str) + " " + plotted["kind"].astype(str)
        )
    writer.figure(
        item,
        "jsd",
        lambda: grouped_bars(
            plotted,
            category="category",
            group="series",
            value="jsd",
            low="ci_low",
            high="ci_high",
            ylabel="JSD (base 2)",
            title="Cross-platform composition JSD (95% joint block bootstrap)",
        ),
        plotted,
        f"Soft broad and supercluster JSD (whole section, shared mask) of the "
        f"headline run ({basis.variant}), set c beside set a on a same-panel pair, "
        "other composition kinds.",
    )
    _density_item(item, writer, options, basis, scope, (first, second))
    _factor_item(item, writer, inputs, options, basis, scope, (first, second))
    variants = set(jsd.get("variant", pd.Series(dtype=str)))
    if not basis.per_platform and "set_c" not in variants:
        item.notes.append("no set c run in the label tables")
    item.notes.append("X1 (platform rescaling) is not enabled in production.")
    writer.table(item, "jsd", jsd)
    item.summary = (
        jsd[
            [
                c
                for c in (
                    "variant",
                    "level",
                    "kind",
                    "region",
                    "jsd",
                    "ci_low",
                    "ci_high",
                    "resolve_jsd",
                    "allowed",
                )
                if c in jsd.columns
            ]
        ]
        if len(jsd)
        else jsd
    )
    return item


def _jsd_metrics(
    item: ReportItem, jsd_rows: Sequence[Mapping[str, Any]], basis: CrossPlatformBasis
) -> None:
    """Append the JSD records (H1: the headline run's soft broad JSD)."""
    for row in jsd_rows:
        withheld = not row.get("allowed", True)
        criterion = (
            "H1"
            if (
                row.get("variant") == basis.variant
                and row.get("level") == "broad"
                and row.get("kind") == "soft"
            )
            else "report"
        )
        from_resolve = row.get("resolve_jsd") is not None
        item.metrics.append(
            metric(
                criterion,
                f"{row.get('level')}_jsd",
                None
                if withheld
                else (row.get("resolve_jsd") if from_resolve else row.get("jsd")),
                definition=(
                    "base-2 Jensen-Shannon distance of the renormalised "
                    "compositions, MERSCOPE vs Xenium; 95% CI from 200 joint 500 µm "
                    "tile resamples"
                ),
                source="resolve_summary" if from_resolve else "report_metrics",
                scope="pair",
                region=row.get("region"),
                level=row.get("level"),
                kind=f"{row.get('variant')}:{row.get('kind')}",
                ci_low=None
                if withheld
                else (row.get("resolve_ci_low") if from_resolve else row.get("ci_low")),
                ci_high=None
                if withheld
                else (
                    row.get("resolve_ci_high") if from_resolve else row.get("ci_high")
                ),
                status="withheld" if withheld else None,
                note=(
                    f"report recomputation {float(row['jsd']):.6g} "
                    f"({row.get('variant')})"
                    if from_resolve and row.get("jsd") is not None
                    else str(row.get("note", ""))
                ),
            )
        )


def _density_item(
    item: ReportItem,
    writer: ItemWriter,
    options: ReportOptions,
    basis: CrossPlatformBasis,
    scope: Any,
    pair: tuple[SampleData, SampleData],
) -> None:
    """Write the aligned-bin density correlation (item 7)."""
    first, second = pair
    table_a, table_b = table_cells(first), table_cells(second)
    xy_a, xy_b = first.table_xy(), second.table_xy()
    if not (
        xy_a is not None
        and xy_b is not None
        and first.aligned_frame
        and second.aligned_frame
        and scope.allows("broad")
    ):
        item.notes.append(
            "aligned-bin concordance not computed (coordinates not in one frame, or "
            "the scope withholds it)"
        )
        return
    if basis.per_platform:
        soft_a = _soft_broad_from_prefix(table_a, basis.prefix)
        soft_b = _soft_broad_from_prefix(table_b, basis.prefix)
    else:
        soft_a, soft_b = soft_matrix(table_a, "human"), soft_matrix(table_b, "human")
    label_matrices = {
        basis.label_kind: (
            one_hot_matrix(_pair_labels(table_a, basis), HUMAN_BROAD_CLASSES)[:, :-1],
            one_hot_matrix(_pair_labels(table_b, basis), HUMAN_BROAD_CLASSES)[:, :-1],
        )
    }
    matrices: dict[str, tuple[np.ndarray | None, np.ndarray | None]] = {
        **label_matrices,
        "soft": (
            None if soft_a is None else soft_a[:, :-1],
            None if soft_b is None else soft_b[:, :-1],
        ),
    }
    ok_a, ok_b = np.isfinite(xy_a).all(axis=1), np.isfinite(xy_b).all(axis=1)
    parts = []
    for kind, (m_a, m_b) in matrices.items():
        if m_a is None or m_b is None:
            continue
        frame = aligned_bin_density_correlation(
            xy_a[ok_a],
            m_a[ok_a],
            xy_b[ok_b],
            m_b[ok_b],
            HUMAN_BROAD_CLASSES,
            bin_um=options.density_bin_um,
        )
        frame.insert(0, "kind", kind)
        parts.append(frame)
    density = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    for row in density.to_dict("records"):
        item.metrics.append(
            metric(
                "report",
                "density_correlation",
                row["pearson_r"],
                definition=(
                    "Pearson r of per-type densities in 200 µm bins shared "
                    "by both sections (Xenium frame)"
                ),
                source="report_metrics.aligned_bin_density_correlation",
                scope="pair",
                kind=row["kind"],
                group=row["category"],
                n=row["n_bins"],
                note=(
                    f"run {basis.prefix}"
                    + (
                        f"; spearman {row['spearman_r']:.4g}"
                        if row["spearman_r"] == row["spearman_r"]
                        else ""
                    )
                ),
            )
        )
    if basis.per_platform:
        item.metrics.append(
            metric(
                "report",
                "density_correlation",
                None,
                definition="confident-label densities of a per_platform pair",
                source="report_expression",
                scope="pair",
                kind="confident",
                status="withheld",
                note=f"{OWN_PANEL_REASON}: own-panel confident labels (§8.5)",
            )
        )
    writer.figure(
        item,
        "density_correlation",
        lambda: grouped_bars(
            density,
            category="category",
            group="kind",
            value="pearson_r",
            ylabel="Pearson r",
            title=(
                "Per-type density correlation in "
                f"{options.density_bin_um:.0f} µm aligned bins"
            ),
        ),
        density,
        "Per-type density correlation of the two sections in aligned bins of the "
        f"Xenium frame (bins with >= 5 cells in both; run {basis.prefix}).",
    )


def _factor_item(
    item: ReportItem,
    writer: ItemWriter,
    inputs: ReportInputs,
    options: ReportOptions,
    basis: CrossPlatformBasis,
    scope: Any,
    pair: tuple[SampleData, SampleData],
) -> None:
    """Write the per-gene platform factors and the per-label pseudobulk r."""
    if not (options.read_expression and scope.allows("broad")):
        return
    first, second = pair
    table_a, table_b = table_cells(first), table_cells(second)
    cache: dict[str, ClusteredTable] = {}
    loaded_a = _counts_for(first, table_a, cache, inputs.gene_lookup)
    loaded_b = _counts_for(second, table_b, cache, inputs.gene_lookup)
    if loaded_a is None or loaded_b is None:
        item.notes.append(
            "no clustered counts for both sections: gene factors and pseudobulk r "
            "not computed"
        )
        return
    labels_basis = (
        f"the {basis.variant} argmax broad class"
        if basis.label_kind == "argmax"
        else "the confident broad label"
    )
    factors, pseudo = _gene_factors(
        first,
        _pair_labels(table_a, basis),
        _pair_labels(table_b, basis),
        loaded_a,
        loaded_b,
        cache,
    )
    path = writer.out_dir / f"{inputs.sources.pair_id}_platform_gene_factors.csv"
    from merxen.annotation.report_figures import write_csv

    write_csv(factors.assign(label_basis=basis.label_kind), path)
    item.tables["platform_gene_factors"] = path
    if len(factors):
        spread = (
            factors.groupby("label")["log2_ratio_centred"]
            .quantile([0.1, 0.5, 0.9])
            .unstack()
            .reset_index()
            .rename(columns={0.1: "q10", 0.5: "q50", 0.9: "q90"})
        )
        for row in spread.to_dict("records"):
            item.metrics.append(
                metric(
                    "report",
                    "gene_log2_ratio_q90_minus_q10",
                    float(row["q90"] - row["q10"]),
                    definition=(
                        "spread of per-gene median-centred log2(Xenium / MERSCOPE "
                        f"mean counts) within {labels_basis}"
                    ),
                    source="clustered_h5ad",
                    scope="pair",
                    kind=basis.label_kind,
                    group=str(row["label"]),
                )
            )
    for row in pseudo.to_dict("records"):
        item.metrics.append(
            metric(
                "report",
                "label_pseudobulk_r",
                row["pearson_r"],
                definition=(
                    f"Pearson r of log2(CPM+1) pseudobulks of {labels_basis}, "
                    "MERSCOPE vs Xenium, shared genes"
                ),
                source="clustered_h5ad",
                scope="pair",
                kind=basis.label_kind,
                group=str(row["label"]),
                n=int(row["n_cells_a"] + row["n_cells_b"]),
            )
        )
    if basis.per_platform:
        for name in ("gene_log2_ratio_q90_minus_q10", "label_pseudobulk_r"):
            item.metrics.append(
                metric(
                    "report",
                    name,
                    None,
                    definition="the same on the own-panel confident labels",
                    source="report_expression",
                    scope="pair",
                    kind="confident",
                    status="withheld",
                    note=f"{OWN_PANEL_REASON}: own-panel confident labels (§8.5)",
                )
            )
    writer.figure(
        item,
        "pseudobulk_r",
        lambda: grouped_bars(
            pseudo.assign(series="MERSCOPE vs Xenium"),
            category="label",
            group="series",
            value="pearson_r",
            ylabel="Pearson r",
            title="Per-label pseudobulk correlation across platforms",
        ),
        pseudo,
        f"Per broad label ({labels_basis}), the correlation of the two platforms' "
        "pseudobulks; per-gene log2 ratios in the platform gene factors CSV.",
    )
    if len(factors):
        ranked = factors.sort_values(["label", "log2_ratio_centred"]).copy()
        ranked["rank"] = ranked.groupby("label").cumcount()
        writer.figure(
            item,
            "gene_factors",
            lambda: line_facets(
                ranked,
                x="rank",
                y="log2_ratio_centred",
                hue="label",
                title="Per-gene platform log2 ratio within labels (sorted)",
                xlabel="gene rank",
                ylabel="log2 Xenium / MERSCOPE (centred)",
            ),
            ranked,
            "Median-centred per-gene log2(Xenium / MERSCOPE mean counts) within "
            f"each broad label ({labels_basis}), sorted.",
        )


def _gene_factors(
    first: SampleData,
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    loaded_a: tuple[Any, list[str], np.ndarray],
    loaded_b: tuple[Any, list[str], np.ndarray],
    cache: dict[str, ClusteredTable],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-gene platform factors and pseudobulk r per broad label.

    Args:
        first: The first (MERSCOPE) sample (for gene symbols).
        labels_a: Broad label per table cell of the first section (``""``:
            none).
        labels_b: The same for the second section.
        loaded_a: ``_counts_for`` of the first section.
        loaded_b: ``_counts_for`` of the second section.
        cache: Clustered-table cache.

    Returns:
        ``(factors, pseudobulk)``.
    """
    matrix_a, genes_a, found_a = loaded_a
    matrix_b, genes_b, found_b = loaded_b
    shared = [gene for gene in genes_a if gene in set(genes_b)]
    index_a = [genes_a.index(gene) for gene in shared]
    position_b = {gene: index for index, gene in enumerate(genes_b)}
    index_b = [position_b[gene] for gene in shared]
    symbols = dict(zip(genes_a, _symbols_for(first, cache, genes_a), strict=True))
    labels_a = np.asarray(labels_a, dtype=object)[found_a]
    labels_b = np.asarray(labels_b, dtype=object)[found_b]
    order = [
        name
        for name in HUMAN_BROAD_CLASSES
        if (labels_a == name).sum() >= MIN_LABEL_CELLS
        and (labels_b == name).sum() >= MIN_LABEL_CELLS
    ]
    means_a, n_a = group_mean_counts(matrix_a[:, index_a], labels_a, order)
    means_b, n_b = group_mean_counts(matrix_b[:, index_b], labels_b, order)
    factor_rows = []
    pseudo_rows = []
    for li, label in enumerate(order):
        ratios, median = platform_gene_log2_ratios(means_a[li], means_b[li])
        for gi, gene in enumerate(shared):
            factor_rows.append(
                {
                    "label": label,
                    "gene_id": gene,
                    "gene": symbols.get(gene, gene),
                    "mean_counts_merscope": float(means_a[li, gi]),
                    "mean_counts_xenium": float(means_b[li, gi]),
                    "log2_ratio_centred": float(ratios[gi]),
                    "label_median_log2_ratio": median,
                    "n_cells_merscope": int(n_a[li]),
                    "n_cells_xenium": int(n_b[li]),
                }
            )
        pseudo_rows.append(
            {
                "label": label,
                "n_cells_a": int(n_a[li]),
                "n_cells_b": int(n_b[li]),
                "n_genes": len(shared),
                "pearson_r": pearson_r(log_cpm(means_a[li]), log_cpm(means_b[li])),
            }
        )
    return pd.DataFrame(factor_rows), pd.DataFrame(pseudo_rows)


__all__ = [
    "OWN_PANEL_REASON",
    "CrossPlatformBasis",
    "cross_platform_basis",
    "item_cross_platform",
    "item_reference_expectation",
    "specific_genes",
    "supercluster_pair_jsd",
]

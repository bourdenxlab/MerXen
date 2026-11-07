"""Tissue-level marked point-pattern analysis of individual transcripts."""

from __future__ import annotations

import logging
import zlib
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import hypergeom
from shapely import STRtree, contains_xy, points
from shapely.geometry import MultiPolygon, Polygon
from statsmodels.stats.multitest import multipletests

from merxen.config import SpatialGeneAnalysisConfig, SpatialGeneAnalysisSampleConfig
from merxen.control_features import control_token_mask
from merxen.cortical_depth.boundaries import load_boundary_annotations
from merxen.cortical_depth.tissue import build_full_tissue_polygon
from merxen.io.transcript_io import iter_points_chunks, resolve_col
from merxen.memory import log_status

logger = logging.getLogger(__name__)

COMPARTMENTS = ("nuclear", "cytoplasmic", "extracellular")
COMPARTMENT_CODES = {name: index for index, name in enumerate(COMPARTMENTS)}


@dataclass
class TranscriptPatternData:
    """Compact arrays retained for transcript point-pattern statistics."""

    coordinates: np.ndarray
    gene_codes: np.ndarray
    gene_names: list[str]
    compartments: np.ndarray
    signed_cell_distance_um: np.ndarray
    signed_nucleus_distance_um: np.ndarray
    cell_overlap_count: np.ndarray
    nucleus_overlap_count: np.ndarray
    n_input: int
    n_outside_tissue: int
    n_invalid_coordinates: int
    n_controls_excluded: int
    gene_order: np.ndarray | None = None
    gene_offsets: np.ndarray | None = None


@dataclass
class TranscriptPatternResults:
    """Tables produced by transcript-coordinate spatial analysis."""

    summary: pd.DataFrame
    signed_distance: pd.DataFrame
    paircorr: pd.DataFrame
    rankings: pd.DataFrame
    data: TranscriptPatternData
    tissue_polygon: Polygon | MultiPolygon


def run_transcript_pattern_analysis(
    *,
    sdata_obj: Any,
    sample: SpatialGeneAnalysisSampleConfig,
    config: SpatialGeneAnalysisConfig,
) -> TranscriptPatternResults:
    """Analyze gene marks directly on observed transcript coordinates."""
    point_key = _native_transcript_point_key(sdata_obj)
    if sample.shape_key is None or sample.shape_key not in sdata_obj.shapes:
        raise KeyError(
            f"Active cell shape {sample.shape_key!r} not found; available="
            f"{list(sdata_obj.shapes.keys())}"
        )
    if sample.nuclei_shape_key not in sdata_obj.shapes:
        raise KeyError(
            f"Nuclei shape {sample.nuclei_shape_key!r} not found; rerun SEGMENT "
            "and ENRICH to create DAPI-only Cellpose nuclei."
        )

    tissue_polygon = build_analysis_tissue_polygon(sample)
    data = load_and_classify_transcripts(
        sdata_obj.points[point_key],
        cell_shapes=sdata_obj.shapes[sample.shape_key],
        nuclei_shapes=sdata_obj.shapes[sample.nuclei_shape_key],
        tissue_polygon=tissue_polygon,
        chunk_rows=config.transcript_chunk_rows,
        drop_control_features=config.drop_control_features,
        sample_id=sample.sample_id,
    )
    _prepare_gene_index(data)
    counts = gene_pattern_counts(data, config=config)
    summary = summarize_gene_pattern_counts(counts, config=config)
    signed_distance = signed_distance_enrichment_from_counts(counts, config=config)
    summary = add_signed_distance_summary(summary, signed_distance)
    paircorr = compute_multiscale_pair_correlation(data, config=config)
    summary = add_paircorr_summary(summary, paircorr)
    summary["pattern_label"] = classify_gene_patterns(summary)
    rankings = rank_transcript_patterns(summary, top_n=config.top_n)
    return TranscriptPatternResults(
        summary=summary,
        signed_distance=signed_distance,
        paircorr=paircorr,
        rankings=rankings,
        data=data,
        tissue_polygon=tissue_polygon,
    )


def build_analysis_tissue_polygon(
    sample: SpatialGeneAnalysisSampleConfig,
) -> Polygon | MultiPolygon:
    """Build bounded tissue polygons from pial and tissue-edge annotations."""
    annotations = load_boundary_annotations(
        pial_path=sample.pial_boundary_path,
        wm_path=sample.wm_boundary_path,
        side_boundary_path=sample.side_boundary_path,
        exclusion_path=sample.exclusion_path,
        ribbon_path=sample.ribbon_path,
        annotation_path=sample.annotation_path,
    )
    return build_full_tissue_polygon(
        annotations,
        require_tissue_edge=False,
        allow_ribbon_fallback=True,
    )


def load_and_classify_transcripts(
    points_obj: Any,
    *,
    cell_shapes: Any,
    nuclei_shapes: Any,
    tissue_polygon: Polygon | MultiPolygon,
    chunk_rows: int,
    drop_control_features: bool,
    sample_id: str,
) -> TranscriptPatternData:
    """Stream points and classify them without using transcript assignments."""
    x_col = resolve_col(points_obj, ["x", "global_x", "x_location"])
    y_col = resolve_col(points_obj, ["y", "global_y", "y_location"])
    gene_col = resolve_col(points_obj, ["gene", "feature_name", "target"])
    assert x_col is not None and y_col is not None and gene_col is not None

    cell_geometries = _valid_geometries(cell_shapes)
    nuclei_geometries = _valid_geometries(nuclei_shapes)
    cell_tree = STRtree(cell_geometries)
    nucleus_tree = STRtree(nuclei_geometries)
    cell_boundary_tree = STRtree(np.asarray([g.boundary for g in cell_geometries]))
    nucleus_boundary_tree = STRtree(np.asarray([g.boundary for g in nuclei_geometries]))

    coords_parts: list[np.ndarray] = []
    gene_parts: list[np.ndarray] = []
    compartment_parts: list[np.ndarray] = []
    cell_distance_parts: list[np.ndarray] = []
    nucleus_distance_parts: list[np.ndarray] = []
    cell_overlap_parts: list[np.ndarray] = []
    nucleus_overlap_parts: list[np.ndarray] = []
    gene_lookup: dict[str, int] = {}
    gene_names: list[str] = []
    n_input = 0
    n_outside_tissue = 0
    n_invalid_coordinates = 0
    n_controls_excluded = 0

    for chunk_index, chunk in enumerate(
        _iter_bounded_point_chunks(
            points_obj,
            [x_col, y_col, gene_col],
            chunk_rows=chunk_rows,
            desc=f"[{sample_id}] transcript compartments",
        ),
        start=1,
    ):
        n_input += len(chunk)
        x = pd.to_numeric(chunk[x_col], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(chunk[y_col], errors="coerce").to_numpy(dtype=float)
        genes = chunk[gene_col].astype(str).to_numpy()
        finite = np.isfinite(x) & np.isfinite(y)
        n_invalid_coordinates += int((~finite).sum())
        in_tissue = np.zeros(len(chunk), dtype=bool)
        in_tissue[finite] = contains_xy(tissue_polygon, x[finite], y[finite])
        n_outside_tissue += int((finite & ~in_tissue).sum())
        valid = finite & in_tissue
        if drop_control_features:
            controls = control_token_mask(genes)
            n_controls_excluded += int((valid & controls).sum())
            valid &= ~controls
        if not valid.any():
            continue
        x, y, genes = x[valid], y[valid], genes[valid]
        coordinates = np.column_stack([x, y])
        point_geometries = points(coordinates)

        cell_inside, cell_overlap = _tree_membership(cell_tree, point_geometries)
        nucleus_inside, nucleus_overlap = _tree_membership(
            nucleus_tree, point_geometries
        )
        cell_distance = _nearest_distances(cell_boundary_tree, point_geometries)
        nucleus_distance = _nearest_distances(nucleus_boundary_tree, point_geometries)
        cell_distance[cell_inside] *= 1.0
        cell_distance[~cell_inside] *= -1.0
        nucleus_distance[nucleus_inside] *= 1.0
        nucleus_distance[~nucleus_inside] *= -1.0

        compartments = np.full(
            len(coordinates),
            COMPARTMENT_CODES["extracellular"],
            dtype=np.uint8,
        )
        compartments[cell_inside] = COMPARTMENT_CODES["cytoplasmic"]
        compartments[nucleus_inside] = COMPARTMENT_CODES["nuclear"]

        unique_genes, inverse = np.unique(genes, return_inverse=True)
        unique_codes = np.empty(len(unique_genes), dtype=np.uint16)
        for index, gene in enumerate(unique_genes):
            code = gene_lookup.get(gene)
            if code is None:
                code = len(gene_names)
                if code > np.iinfo(np.uint16).max:
                    raise ValueError(
                        "More than 65,536 transcript features are unsupported."
                    )
                gene_lookup[gene] = code
                gene_names.append(gene)
            unique_codes[index] = code
        codes = unique_codes[inverse]

        coords_parts.append(coordinates.astype(np.float32))
        gene_parts.append(codes)
        compartment_parts.append(compartments)
        cell_distance_parts.append(cell_distance.astype(np.float32))
        nucleus_distance_parts.append(nucleus_distance.astype(np.float32))
        cell_overlap_parts.append(cell_overlap.astype(np.uint8))
        nucleus_overlap_parts.append(nucleus_overlap.astype(np.uint8))
        if chunk_index % 5 == 0:
            log_status(
                f"[{sample_id}] classified {n_input:,} input transcripts; "
                f"retained {sum(len(part) for part in coords_parts):,} in tissue"
            )

    if not coords_parts:
        raise ValueError(f"[{sample_id}] No transcripts remained inside tissue.")
    return TranscriptPatternData(
        coordinates=np.concatenate(coords_parts),
        gene_codes=np.concatenate(gene_parts),
        gene_names=gene_names,
        compartments=np.concatenate(compartment_parts),
        signed_cell_distance_um=np.concatenate(cell_distance_parts),
        signed_nucleus_distance_um=np.concatenate(nucleus_distance_parts),
        cell_overlap_count=np.concatenate(cell_overlap_parts),
        nucleus_overlap_count=np.concatenate(nucleus_overlap_parts),
        n_input=n_input,
        n_outside_tissue=n_outside_tissue,
        n_invalid_coordinates=n_invalid_coordinates,
        n_controls_excluded=n_controls_excluded,
    )


@dataclass
class GenePatternCounts:
    """Per-gene counts that the compartment and signed-distance statistics need.

    Both transcript engines reduce their transcripts to these arrays: the
    polygon engine from the retained points (:func:`gene_pattern_counts`), the
    label-raster engine (:mod:`merxen.analysis.transcript_patterns_raster`) by
    streaming accumulation over every transcript. The statistics functions
    below only read these arrays, so both engines share them.

    Attributes:
        gene_names: Gene of each row, in output order.
        compartment_counts: ``(n_genes, len(COMPARTMENTS))`` transcripts per
            compartment (nuclear, cytoplasmic, extracellular).
        distance_bin_counts: Boundary (``"cell"``, ``"nucleus"``) to
            ``(n_genes, len(edges) + 1)`` transcripts per signed-distance bin
            (bins of ``np.searchsorted(edges, d, side="right")``).
        median_signed_cell_distance_um: Median signed cell distance per gene.
        median_signed_nucleus_distance_um: Median signed nucleus distance.
        n_pericellular: Transcripts with ``-pericellular <= d_cell < 0``.
        n_membrane_proximal: Cytoplasmic transcripts with
            ``d_cell <= membrane_distance_um``.
        n_cell_overlap_ambiguous: Transcripts inside more than one cell.
        n_nucleus_overlap_ambiguous: Transcripts inside more than one nucleus.
    """

    gene_names: list[str]
    compartment_counts: np.ndarray
    distance_bin_counts: dict[str, np.ndarray]
    median_signed_cell_distance_um: np.ndarray
    median_signed_nucleus_distance_um: np.ndarray
    n_pericellular: np.ndarray
    n_membrane_proximal: np.ndarray
    n_cell_overlap_ambiguous: np.ndarray
    n_nucleus_overlap_ambiguous: np.ndarray

    @property
    def n_transcripts(self) -> np.ndarray:
        """Return the transcripts of each gene (``int64``)."""
        totals: np.ndarray = np.asarray(self.compartment_counts, dtype=np.int64).sum(
            axis=1
        )
        return totals


SIGNED_DISTANCE_BOUNDARIES: tuple[str, ...] = ("cell", "nucleus")
"""Boundaries of the signed-distance statistics, in output order."""


def gene_pattern_counts(
    data: TranscriptPatternData,
    *,
    config: SpatialGeneAnalysisConfig,
) -> GenePatternCounts:
    """Reduce the retained transcripts of the polygon engine to per-gene counts.

    Args:
        data: Classified transcripts (:func:`load_and_classify_transcripts`).
        config: Analysis parameters (signed-distance edges, proximity widths).

    Returns:
        The counts, one row per gene of ``data.gene_names`` in that order.
    """
    edges = np.asarray(config.signed_distance_edges_um, dtype=float)
    n_genes = len(data.gene_names)
    compartment_counts = np.zeros((n_genes, len(COMPARTMENTS)), dtype=np.int64)
    bin_counts = {
        boundary: np.zeros((n_genes, len(edges) + 1), dtype=np.int64)
        for boundary in SIGNED_DISTANCE_BOUNDARIES
    }
    all_bins = {
        "cell": _signed_distance_bin_codes(data.signed_cell_distance_um, edges),
        "nucleus": _signed_distance_bin_codes(data.signed_nucleus_distance_um, edges),
    }
    medians = {boundary: np.zeros(n_genes) for boundary in SIGNED_DISTANCE_BOUNDARIES}
    n_pericellular = np.zeros(n_genes, dtype=np.int64)
    n_membrane = np.zeros(n_genes, dtype=np.int64)
    n_cell_overlap = np.zeros(n_genes, dtype=np.int64)
    n_nucleus_overlap = np.zeros(n_genes, dtype=np.int64)
    for code in range(n_genes):
        gene_indices = _gene_indices(data, code)
        compartments = data.compartments[gene_indices]
        compartment_counts[code] = np.bincount(
            compartments, minlength=len(COMPARTMENTS)
        )
        for boundary in SIGNED_DISTANCE_BOUNDARIES:
            bin_counts[boundary][code] = np.bincount(
                all_bins[boundary][gene_indices], minlength=len(edges) + 1
            )
        cell_dist = data.signed_cell_distance_um[gene_indices]
        nucleus_dist = data.signed_nucleus_distance_um[gene_indices]
        medians["cell"][code] = float(np.median(cell_dist))
        medians["nucleus"][code] = float(np.median(nucleus_dist))
        n_pericellular[code] = np.count_nonzero(
            (cell_dist < 0) & (cell_dist >= -config.pericellular_distance_um)
        )
        n_membrane[code] = np.count_nonzero(
            (compartments == COMPARTMENT_CODES["cytoplasmic"])
            & (cell_dist <= config.membrane_distance_um)
        )
        n_cell_overlap[code] = np.count_nonzero(
            data.cell_overlap_count[gene_indices] > 1
        )
        n_nucleus_overlap[code] = np.count_nonzero(
            data.nucleus_overlap_count[gene_indices] > 1
        )
    return GenePatternCounts(
        gene_names=list(data.gene_names),
        compartment_counts=compartment_counts,
        distance_bin_counts=bin_counts,
        median_signed_cell_distance_um=medians["cell"],
        median_signed_nucleus_distance_um=medians["nucleus"],
        n_pericellular=n_pericellular,
        n_membrane_proximal=n_membrane,
        n_cell_overlap_ambiguous=n_cell_overlap,
        n_nucleus_overlap_ambiguous=n_nucleus_overlap,
    )


def summarize_signed_distance_enrichment(
    data: TranscriptPatternData,
    *,
    config: SpatialGeneAnalysisConfig,
) -> pd.DataFrame:
    """Summarize compartment membership and signed-distance enrichment by gene."""
    return summarize_gene_pattern_counts(
        gene_pattern_counts(data, config=config), config=config
    )


def summarize_gene_pattern_counts(
    counts: GenePatternCounts,
    *,
    config: SpatialGeneAnalysisConfig,
) -> pd.DataFrame:
    """Summarize compartment membership and proximity by gene from counts.

    Vectorised over genes (whole-transcriptome panels have 18,000), with the
    same arithmetic per value as the former per-row loop, so the polygon
    engine's tables are unchanged (``tests/test_analysis/
    test_transcript_patterns_raster.py`` checks them byte for byte).

    Args:
        counts: Per-gene counts of either transcript engine.
        config: Analysis parameters (``transcript_min_count``).

    Returns:
        One row per gene: compartment counts, fractions, log2 odds against all
        other genes, hypergeometric tails with BH FDR, the median signed
        distances, the proximity fractions and the overlap counters.
    """
    compartment_counts = np.asarray(counts.compartment_counts, dtype=np.int64)
    compartment_counts = compartment_counts.reshape(-1, len(COMPARTMENTS))
    total_by_compartment = compartment_counts.sum(axis=0)
    n_by_gene = compartment_counts.sum(axis=1)
    n_total = int(n_by_gene.sum())
    eligible = n_by_gene >= config.transcript_min_count
    columns: dict[str, Any] = {
        "gene": pd.Series(list(counts.gene_names), dtype=object),
        "n_transcripts": n_by_gene,
    }
    with np.errstate(divide="ignore", invalid="ignore"):
        for compartment_code, compartment in enumerate(COMPARTMENTS):
            a = compartment_counts[:, compartment_code]
            total = int(total_by_compartment[compartment_code])
            b = total - a
            c = n_by_gene - a
            d = (n_total - total) - c
            columns[f"n_{compartment}"] = a
            columns[f"fraction_{compartment}"] = np.where(
                n_by_gene > 0, a / np.maximum(n_by_gene, 1), np.nan
            )
            columns[f"{compartment}_enrichment_log2_odds"] = _log2_odds(a, b, c, d)
            columns[f"{compartment}_pval_clustered"] = _hypergeom_tail(
                "sf", a - 1, n_total, total, n_by_gene, eligible
            )
            columns[f"{compartment}_pval_depleted"] = _hypergeom_tail(
                "cdf", a, n_total, total, n_by_gene, eligible
            )
        columns["median_signed_cell_distance_um"] = np.asarray(
            counts.median_signed_cell_distance_um, dtype=np.float64
        )
        columns["median_signed_nucleus_distance_um"] = np.asarray(
            counts.median_signed_nucleus_distance_um, dtype=np.float64
        )
        for name, values in (
            ("fraction_pericellular", counts.n_pericellular),
            ("fraction_membrane_proximal", counts.n_membrane_proximal),
        ):
            numerator = np.asarray(values, dtype=np.int64)
            columns[name] = np.where(
                n_by_gene > 0, numerator / np.maximum(n_by_gene, 1), np.nan
            )
    columns["n_cell_overlap_ambiguous"] = np.asarray(
        counts.n_cell_overlap_ambiguous, dtype=np.int64
    )
    columns["n_nucleus_overlap_ambiguous"] = np.asarray(
        counts.n_nucleus_overlap_ambiguous, dtype=np.int64
    )
    columns["signed_distance_eligible"] = eligible.astype(bool)
    result = pd.DataFrame(columns)
    for compartment in COMPARTMENTS:
        for tail in ("clustered", "depleted"):
            pval_col = f"{compartment}_pval_{tail}"
            qval_col = f"{compartment}_pval_{tail}_fdr_bh"
            result[qval_col] = _fdr(result[pval_col].to_numpy(float))
    return result


def _log2_odds(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray
) -> np.ndarray:
    """Return ``log2((a + .5)(d + .5) / ((b + .5)(c + .5)))`` element-wise."""
    a_f, b_f, c_f, d_f = (np.asarray(v, dtype=np.float64) + 0.5 for v in (a, b, c, d))
    odds: np.ndarray = np.log2((a_f * d_f) / (b_f * c_f))
    return odds


def _hypergeom_tail(
    tail: str,
    k: np.ndarray,
    population: int,
    successes: np.ndarray | int,
    draws: np.ndarray,
    eligible: np.ndarray,
) -> np.ndarray:
    """Return hypergeometric ``sf`` or ``cdf`` values where eligible, else NaN."""
    k = np.asarray(k, dtype=np.int64)
    successes_arr = np.broadcast_to(np.asarray(successes, dtype=np.int64), k.shape)
    draws = np.broadcast_to(np.asarray(draws, dtype=np.int64), k.shape)
    result = np.full(k.shape, np.nan, dtype=np.float64)
    take = np.asarray(eligible, dtype=bool)
    if take.any():
        function = hypergeom.sf if tail == "sf" else hypergeom.cdf
        result[take] = function(
            k[take], int(population), successes_arr[take], draws[take]
        )
    return result


def compute_signed_distance_enrichment(
    data: TranscriptPatternData,
    *,
    config: SpatialGeneAnalysisConfig,
) -> pd.DataFrame:
    """Compare each gene's signed boundary-distance distribution to background."""
    return signed_distance_enrichment_from_counts(
        gene_pattern_counts(data, config=config), config=config
    )


def signed_distance_enrichment_from_counts(
    counts: GenePatternCounts,
    *,
    config: SpatialGeneAnalysisConfig,
) -> pd.DataFrame:
    """Compare each gene's signed-distance bin counts to all transcripts.

    Vectorised over genes and bins with the per-value arithmetic of the
    former row loop (rows ordered boundary, gene, bin).

    Args:
        counts: Per-gene counts of either transcript engine.
        config: Analysis parameters (edges, ``transcript_min_count``).

    Returns:
        A long table: gene x boundary x bin, with observed and background
        fractions, log2 odds, hypergeometric tails and their BH FDR per
        boundary and bin.
    """
    edges = np.asarray(config.signed_distance_edges_um, dtype=float)
    n_bins = len(edges) + 1
    n_by_gene = counts.n_transcripts
    n_genes = len(counts.gene_names)
    n_total = int(n_by_gene.sum())
    eligible_gene = n_by_gene >= config.transcript_min_count
    bin_index = np.arange(n_bins, dtype=np.int64)
    distance_min = np.asarray(
        [-np.inf if i == 0 else edges[i - 1] for i in range(n_bins)], dtype=np.float64
    )
    distance_max = np.asarray(
        [np.inf if i == len(edges) else edges[i] for i in range(n_bins)],
        dtype=np.float64,
    )
    labels = [_signed_bin_label(i, edges) for i in range(n_bins)]
    parts: list[pd.DataFrame] = []
    for boundary in SIGNED_DISTANCE_BOUNDARIES:
        gene_bins = np.asarray(counts.distance_bin_counts[boundary], dtype=np.int64)
        gene_bins = gene_bins.reshape(n_genes, n_bins)
        total_counts = gene_bins.sum(axis=0)
        a = gene_bins.reshape(-1)
        n_gene = np.repeat(n_by_gene, n_bins)
        background = np.tile(total_counts, n_genes)
        eligible = np.repeat(eligible_gene, n_bins)
        b = background - a
        c = n_gene - a
        d = (n_total - background) - c
        with np.errstate(divide="ignore", invalid="ignore"):
            parts.append(
                pd.DataFrame(
                    {
                        "gene": pd.Series(
                            np.repeat(
                                np.asarray(list(counts.gene_names), dtype=object),
                                n_bins,
                            ),
                            dtype=object,
                        ),
                        "boundary": pd.Series([boundary] * a.size, dtype=object),
                        "bin_index": np.tile(bin_index, n_genes),
                        "distance_min_um": np.tile(distance_min, n_genes),
                        "distance_max_um": np.tile(distance_max, n_genes),
                        "bin_label": pd.Series(labels * n_genes, dtype=object),
                        "n_transcripts": n_gene,
                        "eligible": eligible,
                        "observed_count": a,
                        "observed_fraction": a / n_gene,
                        "background_count": background,
                        "background_fraction": background / n_total,
                        "enrichment_log2_odds": _log2_odds(a, b, c, d),
                        "pval_enriched": _hypergeom_tail(
                            "sf", a - 1, n_total, background, n_gene, eligible
                        ),
                        "pval_depleted": _hypergeom_tail(
                            "cdf", a, n_total, background, n_gene, eligible
                        ),
                    }
                )
            )
    result = pd.concat(parts, ignore_index=True)
    result["pval_enriched_fdr_bh"] = np.nan
    result["pval_depleted_fdr_bh"] = np.nan
    for _, indices in result.groupby(
        ["boundary", "bin_index"],
        sort=False,
    ).groups.items():
        loc = list(indices)
        result.loc[loc, "pval_enriched_fdr_bh"] = _fdr(
            result.loc[loc, "pval_enriched"].to_numpy(float)
        )
        result.loc[loc, "pval_depleted_fdr_bh"] = _fdr(
            result.loc[loc, "pval_depleted"].to_numpy(float)
        )
    return result


def add_signed_distance_summary(
    summary: pd.DataFrame,
    signed_distance: pd.DataFrame,
) -> pd.DataFrame:
    """Add transparent per-bin signed-distance metrics to the gene summary.

    One column triple per boundary and bin, in the order the bins first
    appear in ``signed_distance``; genes without a row get NaN.
    """
    result = summary.copy()
    if signed_distance.empty:
        return result
    new_columns: dict[str, pd.Series] = {}
    genes = result["gene"]
    for (boundary, label), group in signed_distance.groupby(
        ["boundary", "bin_label"], sort=False
    ):
        prefix = f"signed_distance_{boundary}_{label}"
        by_gene = group.drop_duplicates("gene", keep="last").set_index("gene")
        for source, suffix in (
            ("enrichment_log2_odds", "enrichment_log2_odds"),
            ("pval_enriched_fdr_bh", "enriched_fdr_bh"),
            ("pval_depleted_fdr_bh", "depleted_fdr_bh"),
        ):
            values = genes.map(by_gene[source]).astype(np.float64)
            new_columns[f"{prefix}_{suffix}"] = values
    added = pd.DataFrame(new_columns, index=result.index)
    added = added.loc[:, [c for c in added.columns if c not in result.columns]]
    return pd.concat([result, added], axis=1)


def compute_multiscale_pair_correlation(
    data: TranscriptPatternData,
    *,
    config: SpatialGeneAnalysisConfig,
) -> pd.DataFrame:
    """Compute thinned marked pair-correlation against nested label nulls."""
    edges = np.asarray(config.paircorr_distance_edges_um, dtype=float)
    index_dtype = (
        np.uint32 if len(data.coordinates) <= np.iinfo(np.uint32).max else np.int64
    )
    global_pool = np.arange(len(data.coordinates), dtype=index_dtype)
    compartment_pools = {
        code: np.flatnonzero(data.compartments == code).astype(index_dtype)
        for code in range(len(COMPARTMENTS))
    }
    eligible = [
        (code, gene)
        for code, gene in enumerate(data.gene_names)
        if len(_gene_indices(data, code)) >= config.paircorr_min_count
    ]
    rows: list[dict[str, Any]] = []
    worker = partial(
        _paircorr_rows_for_gene,
        data=data,
        config=config,
        edges=edges,
        global_pool=global_pool,
        compartment_pools=compartment_pools,
    )
    max_workers = min(int(config.paircorr_n_jobs), len(eligible)) if eligible else 1
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        gene_results = executor.map(worker, eligible)
        for gene_rows, gene, n_selected, n_total_gene in gene_results:
            rows.extend(gene_rows)
            log_status(
                f"Pair correlation {gene}: used {n_selected:,}/{n_total_gene:,} "
                f"transcripts with {config.paircorr_permutations} null draws"
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["pval_clustered_fdr_bh"] = np.nan
    result["pval_disperse_fdr_bh"] = np.nan
    group_columns = ["null_model", "band_index"]
    for _, indices in result.groupby(group_columns, sort=False).groups.items():
        loc = list(indices)
        result.loc[loc, "pval_clustered_fdr_bh"] = _fdr(
            result.loc[loc, "pval_clustered"].to_numpy(float)
        )
        result.loc[loc, "pval_disperse_fdr_bh"] = _fdr(
            result.loc[loc, "pval_disperse"].to_numpy(float)
        )
    return result


def _paircorr_rows_for_gene(
    code_and_gene: tuple[int, str],
    *,
    data: TranscriptPatternData,
    config: SpatialGeneAnalysisConfig,
    edges: np.ndarray,
    global_pool: np.ndarray,
    compartment_pools: dict[int, np.ndarray],
) -> tuple[list[dict[str, Any]], str, int, int]:
    """Compute both nested nulls for one gene; safe for a worker thread."""
    code, gene = code_and_gene
    gene_indices = _gene_indices(data, code)
    rng = np.random.default_rng(_gene_seed(config.paircorr_seed, gene))
    selected = _sample_without_replacement(
        rng,
        gene_indices,
        min(len(gene_indices), config.paircorr_max_transcripts_per_gene),
    )
    observed = _annular_pair_counts(data.coordinates[selected], edges)
    global_null = np.zeros((config.paircorr_permutations, len(edges) - 1))
    stratified_null = np.zeros_like(global_null)
    compartment_counts = np.bincount(
        data.compartments[selected],
        minlength=len(COMPARTMENTS),
    )
    for permutation in range(config.paircorr_permutations):
        global_indices = _sample_without_replacement(rng, global_pool, len(selected))
        global_null[permutation] = _annular_pair_counts(
            data.coordinates[global_indices],
            edges,
        )
        stratified_indices = np.concatenate(
            [
                _sample_without_replacement(
                    rng,
                    compartment_pools[compartment_code],
                    int(count),
                )
                for compartment_code, count in enumerate(compartment_counts)
                if count > 0
            ]
        )
        stratified_null[permutation] = _annular_pair_counts(
            data.coordinates[stratified_indices],
            edges,
        )

    rows: list[dict[str, Any]] = []
    for null_name, null_values in (
        ("global", global_null),
        ("compartment_stratified", stratified_null),
    ):
        null_mean = null_values.mean(axis=0)
        null_low = np.quantile(null_values, 0.025, axis=0)
        null_high = np.quantile(null_values, 0.975, axis=0)
        for band in range(len(edges) - 1):
            obs = float(observed[band])
            mean = float(null_mean[band])
            rows.append(
                {
                    "gene": gene,
                    "null_model": null_name,
                    "band_index": band,
                    "distance_min_um": float(edges[band]),
                    "distance_max_um": float(edges[band + 1]),
                    "band_label": _band_label(edges[band], edges[band + 1]),
                    "n_transcripts_total": int(len(gene_indices)),
                    "n_transcripts_used": int(len(selected)),
                    "thinning_fraction": float(len(selected) / len(gene_indices)),
                    "observed_pair_count": obs,
                    "null_pair_count_mean": mean,
                    "null_pair_count_low": float(null_low[band]),
                    "null_pair_count_high": float(null_high[band]),
                    "paircorr_enrichment": (float(obs / mean) if mean > 0 else np.nan),
                    "pval_clustered": float(
                        (1 + np.count_nonzero(null_values[:, band] >= obs))
                        / (config.paircorr_permutations + 1)
                    ),
                    "pval_disperse": float(
                        (1 + np.count_nonzero(null_values[:, band] <= obs))
                        / (config.paircorr_permutations + 1)
                    ),
                }
            )
    return rows, gene, len(selected), len(gene_indices)


def add_paircorr_summary(
    summary: pd.DataFrame,
    paircorr: pd.DataFrame,
) -> pd.DataFrame:
    """Add transparent per-band pair-correlation columns to the gene table."""
    result = summary.copy()
    if paircorr.empty:
        return result
    for row in paircorr.itertuples(index=False):
        null = "global" if row.null_model == "global" else "compartment"
        band = str(row.band_label)
        take = result["gene"] == row.gene
        prefix = f"paircorr_{null}_{band}"
        result.loc[take, f"{prefix}_enrichment"] = row.paircorr_enrichment
        result.loc[take, f"{prefix}_clustered_fdr_bh"] = row.pval_clustered_fdr_bh
        result.loc[take, f"{prefix}_disperse_fdr_bh"] = row.pval_disperse_fdr_bh
    return result


def classify_gene_patterns(summary: pd.DataFrame) -> pd.Series:
    """Assign a conservative primary spatial pattern label."""
    labels: list[str] = []
    for row in summary.to_dict(orient="records"):
        if not bool(row.get("signed_distance_eligible", False)):
            labels.append("insufficient_transcripts")
            continue
        significant_clusters = [
            key
            for key, value in row.items()
            if key.startswith("paircorr_compartment_")
            and key.endswith("_clustered_fdr_bh")
            and pd.notna(value)
            and float(value) < 0.05
        ]
        significant_disperse = [
            key
            for key, value in row.items()
            if key.startswith("paircorr_compartment_")
            and key.endswith("_disperse_fdr_bh")
            and pd.notna(value)
            and float(value) < 0.05
        ]
        compartment_scores = {
            compartment: float(row.get(f"{compartment}_enrichment_log2_odds", 0.0))
            for compartment in COMPARTMENTS
        }
        best_compartment = max(
            compartment_scores,
            key=lambda name: compartment_scores[name],
        )
        best_q = float(row.get(f"{best_compartment}_pval_clustered_fdr_bh", 1.0))
        if significant_clusters:
            labels.append("residual_multiscale_clustered")
        elif significant_disperse:
            labels.append("spatially_disperse")
        elif best_q < 0.05 and compartment_scores[best_compartment] > 0:
            labels.append(f"{best_compartment}_enriched")
        elif float(row.get("fraction_pericellular", 0.0)) >= 0.25:
            labels.append("pericellular")
        elif float(row.get("fraction_membrane_proximal", 0.0)) >= 0.25:
            labels.append("membrane_proximal")
        else:
            labels.append("diffuse_or_unresolved")
    return pd.Series(labels, index=summary.index, dtype="string")


def rank_transcript_patterns(summary: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    """Build separate, interpretable per-pattern gene rankings."""
    specs: list[tuple[str, bool]] = [
        (f"{compartment}_enrichment_log2_odds", False) for compartment in COMPARTMENTS
    ]
    specs.extend(
        [
            ("fraction_pericellular", False),
            ("fraction_membrane_proximal", False),
        ]
    )
    specs.extend(
        (column, False)
        for column in summary.columns
        if column.startswith("paircorr_") and column.endswith("_enrichment")
    )
    specs.extend(
        (column, False)
        for column in summary.columns
        if column.startswith("signed_distance_")
        and column.endswith("_enrichment_log2_odds")
    )
    rankings = []
    eligible = summary[summary["signed_distance_eligible"]].copy()
    for metric, ascending in specs:
        if metric not in eligible.columns:
            continue
        ranked = (
            eligible.dropna(subset=[metric])
            .sort_values(metric, ascending=ascending)
            .head(int(top_n))
            .copy()
        )
        ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
        ranked.insert(0, "metric", metric)
        rankings.append(ranked)
    return pd.concat(rankings, ignore_index=True) if rankings else pd.DataFrame()


def transcript_indices_for_gene(
    data: TranscriptPatternData,
    gene: str,
) -> np.ndarray:
    """Return transcript row indices for a gene without rescanning all labels."""
    try:
        code = data.gene_names.index(str(gene))
    except ValueError:
        return np.empty(0, dtype=np.int64)
    return _gene_indices(data, code)


def _native_transcript_point_key(sdata_obj: Any) -> str:
    keys = [str(key) for key in sdata_obj.points]
    native = [key for key in keys if not key.endswith("_aligned_nonrigid")]
    for key in native:
        if "transcript" in key.lower():
            return key
    if native:
        return native[0]
    raise KeyError("SpatialData object has no native transcript points element.")


def _valid_geometries(shapes: Any) -> np.ndarray:
    geometries = np.asarray(shapes.geometry, dtype=object)
    valid = np.asarray(
        [geometry is not None and not geometry.is_empty for geometry in geometries]
    )
    geometries = geometries[valid]
    if len(geometries) == 0:
        raise ValueError("Shape layer contains no valid geometries.")
    return np.asarray(geometries, dtype=object)


def _iter_bounded_point_chunks(
    points_obj: Any,
    columns: list[str],
    *,
    chunk_rows: int,
    desc: str,
) -> Iterator[pd.DataFrame]:
    """Split large Dask partitions so geometry work stays row bounded."""
    for partition in iter_points_chunks(
        points_obj,
        columns,
        chunk_rows=chunk_rows,
        desc=desc,
    ):
        for start in range(0, len(partition), int(chunk_rows)):
            yield partition.iloc[start : start + int(chunk_rows)].copy()


def _prepare_gene_index(data: TranscriptPatternData) -> None:
    """Build one compact transcript ordering shared by all per-gene analyses."""
    if data.gene_order is not None and data.gene_offsets is not None:
        return
    counts = np.bincount(data.gene_codes, minlength=len(data.gene_names))
    offsets = np.empty(len(counts) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    order = np.argsort(data.gene_codes, kind="stable")
    if len(data.gene_codes) <= np.iinfo(np.uint32).max:
        order = order.astype(np.uint32)
    data.gene_order = order
    data.gene_offsets = offsets


def _gene_indices(data: TranscriptPatternData, code: int) -> np.ndarray:
    _prepare_gene_index(data)
    assert data.gene_order is not None and data.gene_offsets is not None
    start = int(data.gene_offsets[code])
    stop = int(data.gene_offsets[code + 1])
    return data.gene_order[start:stop]


def _tree_membership(
    tree: STRtree,
    point_geometries: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pairs = tree.query(point_geometries, predicate="covered_by")
    overlap = np.zeros(len(point_geometries), dtype=np.uint16)
    if pairs.size:
        np.add.at(overlap, pairs[0], 1)
    return overlap > 0, np.minimum(overlap, 255).astype(np.uint8)


def _nearest_distances(tree: STRtree, point_geometries: np.ndarray) -> np.ndarray:
    indices, distances = tree.query_nearest(
        point_geometries,
        return_distance=True,
        all_matches=False,
    )
    result = np.full(len(point_geometries), np.nan, dtype=float)
    result[indices[0]] = distances
    return result


def _sample_without_replacement(
    rng: np.random.Generator,
    pool: np.ndarray,
    count: int,
) -> np.ndarray:
    if count > len(pool):
        raise ValueError(f"Cannot draw {count} coordinates from pool of {len(pool)}")
    if count == len(pool):
        return pool.copy()
    return rng.choice(pool, size=int(count), replace=False)


def _annular_pair_counts(coordinates: np.ndarray, edges: np.ndarray) -> np.ndarray:
    tree = cKDTree(np.asarray(coordinates, dtype=float))
    cumulative = np.asarray(tree.count_neighbors(tree, edges), dtype=np.int64)
    # Self-query counts each non-self pair twice and includes n self-pairs.
    unordered = (cumulative - len(coordinates)) // 2
    unordered[0] = 0
    return np.diff(unordered).astype(float)


def _gene_seed(base_seed: int, gene: str) -> int:
    return int((int(base_seed) + zlib.crc32(gene.encode("utf-8"))) % (2**32))


def _band_label(left: float, right: float) -> str:
    def token(value: float) -> str:
        if float(value).is_integer():
            return str(int(value))
        return str(value).replace(".", "p")

    return f"{token(left)}_{token(right)}um"


def _signed_distance_bin_codes(
    distances: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    return np.searchsorted(edges, np.asarray(distances, dtype=float), side="right")


def _signed_bin_label(bin_index: int, edges: np.ndarray) -> str:
    if bin_index == 0:
        return f"lt_{_signed_distance_token(edges[0])}um"
    if bin_index == len(edges):
        return f"ge_{_signed_distance_token(edges[-1])}um"
    return (
        f"{_signed_distance_token(edges[bin_index - 1])}_"
        f"{_signed_distance_token(edges[bin_index])}um"
    )


def _signed_distance_token(value: float) -> str:
    prefix = "neg" if value < 0 else ""
    magnitude = abs(float(value))
    token = (
        str(int(magnitude))
        if magnitude.is_integer()
        else str(magnitude).replace(".", "p")
    )
    return f"{prefix}{token}"


def _fdr(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    if finite.any():
        result[finite] = multipletests(values[finite], method="fdr_bh")[1]
    return result

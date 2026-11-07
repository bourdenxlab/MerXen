"""Verbatim copy of the per-row transcript statistics before AT-8.

The label-raster engine (AT-8) moved these statistics onto per-gene count
arrays and vectorised them. The polygon engine's XENIUM and MERSCOPE outputs
must stay byte-identical, so ``test_transcript_patterns_raster.py`` compares
the new functions with this frozen copy. Do not edit it.
"""

# ruff: noqa: E501

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

from merxen.analysis.transcript_spatial_patterns import (
    COMPARTMENT_CODES,
    COMPARTMENTS,
    TranscriptPatternData,
    _fdr,
    _gene_indices,
    _signed_bin_label,
    _signed_distance_bin_codes,
)
from merxen.config import SpatialGeneAnalysisConfig


def legacy_summarize_signed_distance_enrichment(
    data: TranscriptPatternData,
    *,
    config: SpatialGeneAnalysisConfig,
) -> pd.DataFrame:
    """Summarize compartment membership and signed-distance enrichment by gene."""
    total_by_compartment = np.bincount(
        data.compartments,
        minlength=len(COMPARTMENTS),
    )
    rows: list[dict[str, Any]] = []
    n_total = len(data.coordinates)
    for code, gene in enumerate(data.gene_names):
        gene_indices = _gene_indices(data, code)
        n_gene = len(gene_indices)
        eligible = n_gene >= config.transcript_min_count
        row: dict[str, Any] = {"gene": gene, "n_transcripts": n_gene}
        counts = np.bincount(
            data.compartments[gene_indices],
            minlength=len(COMPARTMENTS),
        )
        for compartment_code, compartment in enumerate(COMPARTMENTS):
            count = int(counts[compartment_code])
            row[f"n_{compartment}"] = count
            row[f"fraction_{compartment}"] = count / n_gene if n_gene else np.nan
            a = count
            b = int(total_by_compartment[compartment_code] - count)
            c = n_gene - count
            d = int((n_total - total_by_compartment[compartment_code]) - c)
            row[f"{compartment}_enrichment_log2_odds"] = float(
                np.log2(((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5)))
            )
            row[f"{compartment}_pval_clustered"] = (
                float(
                    hypergeom.sf(
                        a - 1,
                        n_total,
                        int(total_by_compartment[compartment_code]),
                        n_gene,
                    )
                )
                if eligible
                else np.nan
            )
            row[f"{compartment}_pval_depleted"] = (
                float(
                    hypergeom.cdf(
                        a,
                        n_total,
                        int(total_by_compartment[compartment_code]),
                        n_gene,
                    )
                )
                if eligible
                else np.nan
            )

        cell_dist = data.signed_cell_distance_um[gene_indices]
        nucleus_dist = data.signed_nucleus_distance_um[gene_indices]
        row["median_signed_cell_distance_um"] = float(np.median(cell_dist))
        row["median_signed_nucleus_distance_um"] = float(np.median(nucleus_dist))
        row["fraction_pericellular"] = float(
            np.mean((cell_dist < 0) & (cell_dist >= -config.pericellular_distance_um))
        )
        row["fraction_membrane_proximal"] = float(
            np.mean(
                (data.compartments[gene_indices] == COMPARTMENT_CODES["cytoplasmic"])
                & (cell_dist <= config.membrane_distance_um)
            )
        )
        row["n_cell_overlap_ambiguous"] = int(
            np.count_nonzero(data.cell_overlap_count[gene_indices] > 1)
        )
        row["n_nucleus_overlap_ambiguous"] = int(
            np.count_nonzero(data.nucleus_overlap_count[gene_indices] > 1)
        )
        row["signed_distance_eligible"] = eligible
        rows.append(row)
    result = pd.DataFrame(rows)
    for compartment in COMPARTMENTS:
        for tail in ("clustered", "depleted"):
            pval_col = f"{compartment}_pval_{tail}"
            qval_col = f"{compartment}_pval_{tail}_fdr_bh"
            result[qval_col] = _fdr(result[pval_col].to_numpy(float))
    return result


def legacy_compute_signed_distance_enrichment(
    data: TranscriptPatternData,
    *,
    config: SpatialGeneAnalysisConfig,
) -> pd.DataFrame:
    """Compare each gene's signed boundary-distance distribution to background."""
    edges = np.asarray(config.signed_distance_edges_um, dtype=float)
    rows: list[dict[str, Any]] = []
    n_total = len(data.coordinates)
    for boundary, all_distances in (
        ("cell", data.signed_cell_distance_um),
        ("nucleus", data.signed_nucleus_distance_um),
    ):
        all_bins = _signed_distance_bin_codes(all_distances, edges)
        total_counts = np.bincount(all_bins, minlength=len(edges) + 1)
        for code, gene in enumerate(data.gene_names):
            gene_indices = _gene_indices(data, code)
            n_gene = len(gene_indices)
            eligible = n_gene >= config.transcript_min_count
            gene_counts = np.bincount(
                all_bins[gene_indices],
                minlength=len(edges) + 1,
            )
            for bin_index, count in enumerate(gene_counts):
                background_count = int(total_counts[bin_index])
                a = int(count)
                b = background_count - a
                c = n_gene - a
                d = (n_total - background_count) - c
                rows.append(
                    {
                        "gene": gene,
                        "boundary": boundary,
                        "bin_index": bin_index,
                        "distance_min_um": (
                            -np.inf if bin_index == 0 else edges[bin_index - 1]
                        ),
                        "distance_max_um": (
                            np.inf if bin_index == len(edges) else edges[bin_index]
                        ),
                        "bin_label": _signed_bin_label(bin_index, edges),
                        "n_transcripts": n_gene,
                        "eligible": eligible,
                        "observed_count": a,
                        "observed_fraction": a / n_gene,
                        "background_count": background_count,
                        "background_fraction": background_count / n_total,
                        "enrichment_log2_odds": float(
                            np.log2(((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5)))
                        ),
                        "pval_enriched": (
                            float(
                                hypergeom.sf(
                                    a - 1,
                                    n_total,
                                    background_count,
                                    n_gene,
                                )
                            )
                            if eligible
                            else np.nan
                        ),
                        "pval_depleted": (
                            float(
                                hypergeom.cdf(
                                    a,
                                    n_total,
                                    background_count,
                                    n_gene,
                                )
                            )
                            if eligible
                            else np.nan
                        ),
                    }
                )
    result = pd.DataFrame(rows)
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


def legacy_add_signed_distance_summary(
    summary: pd.DataFrame,
    signed_distance: pd.DataFrame,
) -> pd.DataFrame:
    """Add transparent per-bin signed-distance metrics to the gene summary."""
    result = summary.copy()
    for row in signed_distance.itertuples(index=False):
        take = result["gene"] == row.gene
        prefix = f"signed_distance_{row.boundary}_{row.bin_label}"
        result.loc[take, f"{prefix}_enrichment_log2_odds"] = row.enrichment_log2_odds
        result.loc[take, f"{prefix}_enriched_fdr_bh"] = row.pval_enriched_fdr_bh
        result.loc[take, f"{prefix}_depleted_fdr_bh"] = row.pval_depleted_fdr_bh
    return result

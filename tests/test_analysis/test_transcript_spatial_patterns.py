"""Tests for transcript-coordinate tissue spatial pattern analysis."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from merxen.analysis.transcript_spatial_patterns import (
    COMPARTMENT_CODES,
    TranscriptPatternData,
    build_analysis_tissue_polygon,
    compute_multiscale_pair_correlation,
    compute_signed_distance_enrichment,
    load_and_classify_transcripts,
    summarize_signed_distance_enrichment,
)
from merxen.config import SpatialGeneAnalysisConfig, SpatialGeneAnalysisSampleConfig


def _config(**overrides: object) -> SpatialGeneAnalysisConfig:
    return SpatialGeneAnalysisConfig(
        pair_id="toy",
        output_dir=Path("."),
        samples=[],
        **overrides,
    )


def test_transcripts_are_classified_from_geometry_not_assignments() -> None:
    """Nuclear/cytoplasmic/extracellular calls should use point-in-polygon."""
    points = pd.DataFrame(
        {
            "x": [3.0, 8.0, 15.0],
            "y": [3.0, 8.0, 5.0],
            "gene": ["A", "A", "B"],
            # Deliberately contradictory and unused instrument assignments.
            "cell_id": [0, 0, 42],
        }
    )
    cells = gpd.GeoDataFrame(geometry=[box(1.0, 1.0, 10.0, 10.0)])
    nuclei = gpd.GeoDataFrame(geometry=[box(2.0, 2.0, 5.0, 5.0)])

    data = load_and_classify_transcripts(
        points,
        cell_shapes=cells,
        nuclei_shapes=nuclei,
        tissue_polygon=box(0.0, 0.0, 20.0, 20.0),
        chunk_rows=2,
        drop_control_features=True,
        sample_id="toy",
    )

    assert data.compartments.tolist() == [
        COMPARTMENT_CODES["nuclear"],
        COMPARTMENT_CODES["cytoplasmic"],
        COMPARTMENT_CODES["extracellular"],
    ]
    assert data.signed_cell_distance_um.tolist() == [2.0, 2.0, -5.0]
    assert data.signed_nucleus_distance_um[0] > 0
    assert np.all(data.signed_nucleus_distance_um[1:] < 0)


def test_signed_distance_enrichment_is_reported_by_boundary_and_bin() -> None:
    """Distance enrichment should compare gene fractions with transcript background."""
    data = TranscriptPatternData(
        coordinates=np.column_stack([np.arange(20), np.zeros(20)]).astype(np.float32),
        gene_codes=np.repeat(np.array([0, 1], dtype=np.uint16), 10),
        gene_names=["inside", "outside"],
        compartments=np.repeat(
            np.array(
                [
                    COMPARTMENT_CODES["nuclear"],
                    COMPARTMENT_CODES["extracellular"],
                ],
                dtype=np.uint8,
            ),
            10,
        ),
        signed_cell_distance_um=np.r_[np.full(10, 3.0), np.full(10, -8.0)],
        signed_nucleus_distance_um=np.r_[np.full(10, 1.0), np.full(10, -10.0)],
        cell_overlap_count=np.ones(20, dtype=np.uint8),
        nucleus_overlap_count=np.ones(20, dtype=np.uint8),
        n_input=20,
        n_outside_tissue=0,
        n_invalid_coordinates=0,
        n_controls_excluded=0,
    )
    config = _config(
        transcript_min_count=2,
        signed_distance_edges_um=[-20.0, -2.0, 0.0, 2.0, 20.0],
    )

    compartment = summarize_signed_distance_enrichment(data, config=config)
    signed = compute_signed_distance_enrichment(data, config=config)

    inside = compartment.set_index("gene").loc["inside"]
    assert inside["nuclear_enrichment_log2_odds"] > 0
    row = signed[
        (signed["gene"] == "inside")
        & (signed["boundary"] == "nucleus")
        & (signed["distance_min_um"] == 0.0)
        & (signed["distance_max_um"] == 2.0)
    ].iloc[0]
    assert row["observed_fraction"] == 1.0
    assert row["enrichment_log2_odds"] > 0


def test_pair_correlation_uses_both_nulls_and_deterministic_thinning() -> None:
    """The two nested random-label nulls should be reproducible per gene."""
    rng = np.random.default_rng(4)
    clustered = rng.normal(loc=(10.0, 10.0), scale=0.3, size=(40, 2))
    diffuse = rng.uniform(0.0, 20.0, size=(40, 2))
    coordinates = np.vstack([clustered, diffuse]).astype(np.float32)
    data = TranscriptPatternData(
        coordinates=coordinates,
        gene_codes=np.repeat(np.array([0, 1], dtype=np.uint16), 40),
        gene_names=["clustered", "diffuse"],
        compartments=np.tile(np.array([0, 1], dtype=np.uint8), 40),
        signed_cell_distance_um=np.zeros(80),
        signed_nucleus_distance_um=np.zeros(80),
        cell_overlap_count=np.ones(80, dtype=np.uint8),
        nucleus_overlap_count=np.ones(80, dtype=np.uint8),
        n_input=80,
        n_outside_tissue=0,
        n_invalid_coordinates=0,
        n_controls_excluded=0,
    )
    config = _config(
        paircorr_min_count=10,
        paircorr_max_transcripts_per_gene=20,
        paircorr_distance_edges_um=[0.0, 1.0, 5.0, 20.0],
        paircorr_permutations=9,
        paircorr_seed=13,
        paircorr_n_jobs=2,
    )

    first = compute_multiscale_pair_correlation(data, config=config)
    second = compute_multiscale_pair_correlation(data, config=config)

    pd.testing.assert_frame_equal(first, second)
    assert set(first["null_model"]) == {"global", "compartment_stratified"}
    assert set(first["n_transcripts_used"]) == {20}
    assert set(first["thinning_fraction"]) == {0.5}


def test_tissue_polygon_uses_pial_and_tissue_edge_annotations(
    tmp_path: Path,
) -> None:
    """A combined pia/edge annotation should define the transcript support."""
    annotation_path = tmp_path / "boundaries.geojson"
    annotation_path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"role": "pia"},
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [[0, 0], [10, 0]],
                        },
                    },
                    {
                        "type": "Feature",
                        "properties": {"role": "tissue_edge"},
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [
                                [0, 0],
                                [10, 0],
                                [10, 10],
                                [0, 10],
                                [0, 0],
                            ],
                        },
                    },
                ],
            }
        )
    )
    sample = SpatialGeneAnalysisSampleConfig(
        sample_id="toy",
        platform="MERSCOPE",
        zarr_path=tmp_path / "toy.zarr",
        annotation_path=annotation_path,
    )

    polygon = build_analysis_tissue_polygon(sample)

    assert np.isclose(polygon.area, 100.0)
    assert polygon.contains(box(4.0, 4.0, 6.0, 6.0))

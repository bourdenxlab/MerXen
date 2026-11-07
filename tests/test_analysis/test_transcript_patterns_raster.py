"""Tests for the label-raster transcript-pattern engine (AT-8)."""

from __future__ import annotations

import io
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from shapely.geometry import box

from merxen.analysis import transcript_patterns_raster as raster
from merxen.analysis.transcript_patterns_raster import (
    DISTANCE_CLIP_CODE,
    PAIRCORR_STATUS,
    RasterEngineSettings,
    RasterInputs,
    build_distance_raster,
    density_tissue_mask,
    distance_encoding,
    gene_diagnostic_rasters,
    polygon_tissue_mask,
    run_label_raster_patterns,
)
from merxen.analysis.transcript_spatial_patterns import (
    COMPARTMENTS,
    TranscriptPatternData,
    add_signed_distance_summary,
    compute_signed_distance_enrichment,
    gene_pattern_counts,
    load_and_classify_transcripts,
    summarize_signed_distance_enrichment,
)
from merxen.config import SpatialGeneAnalysisConfig, SpatialGeneAnalysisSampleConfig
from merxen.enrichment.label_lookup import PixelFrame

from .legacy_transcript_statistics import (
    legacy_add_signed_distance_summary,
    legacy_compute_signed_distance_enrichment,
    legacy_summarize_signed_distance_enrichment,
)

UM_PER_PX = 0.5
"""Exact in binary, so pixel edges are exact micron values."""


def _config(**overrides: object) -> SpatialGeneAnalysisConfig:
    values: dict[str, object] = {
        "pair_id": "toy",
        "output_dir": Path("."),
        "samples": [],
        "transcript_min_count": 5,
    }
    values.update(overrides)
    return SpatialGeneAnalysisConfig.model_validate(values)


# --------------------------------------------------------------------------
# Count-based statistics: byte-identical to the former per-row loops
# --------------------------------------------------------------------------


def _random_pattern_data(seed: int) -> TranscriptPatternData:
    rng = np.random.default_rng(seed)
    gene_sizes = np.concatenate([rng.integers(1, 9, 6), rng.integers(10, 400, 34)])
    n = int(gene_sizes.sum())
    codes = np.repeat(np.arange(gene_sizes.size, dtype=np.uint16), gene_sizes)
    order = rng.permutation(n)
    return TranscriptPatternData(
        coordinates=rng.uniform(0, 500, size=(n, 2)).astype(np.float32),
        gene_codes=codes[order],
        gene_names=[f"G{i:02d}" for i in range(gene_sizes.size)],
        compartments=rng.integers(0, 3, n).astype(np.uint8),
        signed_cell_distance_um=rng.normal(0, 40, n).astype(np.float32),
        signed_nucleus_distance_um=rng.normal(-5, 60, n).astype(np.float32),
        cell_overlap_count=rng.integers(0, 3, n).astype(np.uint8),
        nucleus_overlap_count=rng.integers(0, 3, n).astype(np.uint8),
        n_input=n,
        n_outside_tissue=0,
        n_invalid_coordinates=0,
        n_controls_excluded=0,
    )


def _table_bytes(frame: pd.DataFrame) -> tuple[str, bytes]:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return frame.to_csv(index=False), buffer.getvalue()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_count_statistics_are_byte_identical_to_the_former_loops(seed: int) -> None:
    """XENIUM/MERSCOPE (polygon engine) tables do not change in AT-8."""
    data = _random_pattern_data(seed)
    config = _config(transcript_min_count=50)

    summary = summarize_signed_distance_enrichment(data, config=config)
    legacy_summary = legacy_summarize_signed_distance_enrichment(data, config=config)
    signed = compute_signed_distance_enrichment(data, config=config)
    legacy_signed = legacy_compute_signed_distance_enrichment(data, config=config)

    pd.testing.assert_frame_equal(summary, legacy_summary, check_exact=True)
    pd.testing.assert_frame_equal(signed, legacy_signed, check_exact=True)
    assert _table_bytes(summary) == _table_bytes(legacy_summary)
    assert _table_bytes(signed) == _table_bytes(legacy_signed)
    merged = add_signed_distance_summary(summary, signed)
    legacy_merged = legacy_add_signed_distance_summary(legacy_summary, legacy_signed)
    pd.testing.assert_frame_equal(merged, legacy_merged, check_exact=True)
    assert merged.to_csv(index=False) == legacy_merged.to_csv(index=False)


# --------------------------------------------------------------------------
# A pixel-aligned fixture shared by both engines
# --------------------------------------------------------------------------


def _fixture(seed: int = 0, n_per_gene: int = 400) -> dict[str, object]:
    """Rectangular cells and nuclei on whole pixels, and transcripts.

    Gene A sits in nuclei, B in cytoplasm, C outside cells, D everywhere,
    plus control rows by ``is_gene`` and by name.
    """
    rng = np.random.default_rng(seed)
    height, width = 240, 320  # pixels: 120 x 160 um
    cell_labels = np.zeros((height, width), dtype=np.uint32)
    nucleus_labels = np.zeros((height, width), dtype=np.uint32)
    cells: list[tuple[int, int, int, int]] = []
    nuclei: list[tuple[int, int, int, int]] = []
    label = 0
    for r0 in range(20, 200, 50):
        for c0 in range(20, 280, 60):
            label += 1
            r1, c1 = r0 + 36, c0 + 44
            cell_labels[r0:r1, c0:c1] = label
            cells.append((r0, r1, c0, c1))
            nr0, nc0 = r0 + 10, c0 + 12
            nucleus_labels[nr0 : nr0 + 14, nc0 : nc0 + 16] = label
            nuclei.append((nr0, nr0 + 14, nc0, nc0 + 16))

    def _to_um(rect: tuple[int, int, int, int]) -> object:
        r0, r1, c0, c1 = rect
        return box(c0 * UM_PER_PX, r0 * UM_PER_PX, c1 * UM_PER_PX, r1 * UM_PER_PX)

    def _sample_in(mask: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
        rows, cols = np.nonzero(mask)
        pick = rng.integers(0, rows.size, size)
        # Stay away from pixel edges so float rules cannot disagree.
        x = (cols[pick] + rng.uniform(0.05, 0.95, size)) * UM_PER_PX
        y = (rows[pick] + rng.uniform(0.05, 0.95, size)) * UM_PER_PX
        return x, y

    parts = []
    nuclear = nucleus_labels > 0
    cytoplasm = (cell_labels > 0) & ~nuclear
    outside = cell_labels == 0
    for gene, mask in (
        ("A", nuclear),
        ("B", cytoplasm),
        ("C", outside),
        ("D", np.ones_like(nuclear)),
    ):
        x, y = _sample_in(mask, n_per_gene)
        parts.append(
            pd.DataFrame({"x": x, "y": y, "feature_name": gene, "is_gene": True})
        )
    x, y = _sample_in(np.ones_like(nuclear), 30)
    parts.append(
        pd.DataFrame(
            {"x": x, "y": y, "feature_name": "Intergenic_Region_1", "is_gene": False}
        )
    )
    x, y = _sample_in(np.ones_like(nuclear), 20)
    parts.append(
        pd.DataFrame(
            {"x": x, "y": y, "feature_name": "NegControlProbe_7", "is_gene": None}
        )
    )
    points = pd.concat(parts, ignore_index=True)
    points = points.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    points["feature_name"] = points["feature_name"].astype("category")
    points["is_gene"] = points["is_gene"].astype("boolean")
    return {
        "cell_labels": cell_labels,
        "nucleus_labels": nucleus_labels,
        "cell_polygons": [_to_um(rect) for rect in cells],
        "nucleus_polygons": [_to_um(rect) for rect in nuclei],
        "points": points,
        "tissue": box(0.0, 0.0, 160.0, 120.0),
    }


def _write_points(points: pd.DataFrame, directory: Path, n_files: int) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    files = []
    for index, part in enumerate(np.array_split(np.arange(len(points)), n_files)):
        path = directory / f"part.{index}.parquet"
        table = pa.Table.from_pandas(points.iloc[part], preserve_index=False)
        pq.write_table(table, path, row_group_size=97)
        files.append(path)
    return files


def _inputs(fixture: dict[str, object], files: list[Path]) -> RasterInputs:
    return RasterInputs(
        files=files,
        cell_labels=fixture["cell_labels"],
        nucleus_labels=fixture["nucleus_labels"],
        frame=PixelFrame(UM_PER_PX),
        convention="corner",
        platform="ATERA",
    )


def _tissue(fixture: dict[str, object]) -> raster.TissueMask:
    return polygon_tissue_mask(
        fixture["tissue"], origin_um=(0.0, 0.0), bin_um=10.0, shape=(12, 16)
    )


def _run(
    fixture: dict[str, object],
    tmp_path: Path,
    *,
    n_files: int = 1,
    settings: RasterEngineSettings | None = None,
    **overrides: object,
) -> raster.RasterTranscriptPatternResults:
    files = _write_points(fixture["points"], tmp_path / f"points_{n_files}", n_files)
    return run_label_raster_patterns(
        _inputs(fixture, files),
        tissue=_tissue(fixture),
        config=_config(**overrides),
        scratch_dir=tmp_path / "scratch",
        sample_id="toy",
        settings=settings or RasterEngineSettings(n_workers=1),
    )


def test_label_engine_matches_polygon_engine_on_pixel_aligned_shapes(
    tmp_path: Path,
) -> None:
    """Compartment counts agree exactly; distances agree within one pixel."""
    fixture = _fixture()
    points = fixture["points"]
    config = _config()
    result = _run(fixture, tmp_path)

    polygon_data = load_and_classify_transcripts(
        points.rename(columns={"feature_name": "gene"}).assign(
            gene=lambda frame: frame["gene"].astype(str)
        ),
        cell_shapes=gpd.GeoDataFrame(geometry=fixture["cell_polygons"]),
        nuclei_shapes=gpd.GeoDataFrame(geometry=fixture["nucleus_polygons"]),
        tissue_polygon=fixture["tissue"],
        chunk_rows=1000,
        drop_control_features=True,
        sample_id="toy",
    )
    polygon = gene_pattern_counts(polygon_data, config=config)
    polygon_by_gene = dict(
        zip(polygon.gene_names, polygon.compartment_counts, strict=True)
    )
    raster_by_gene = dict(
        zip(result.counts.gene_names, result.counts.compartment_counts, strict=True)
    )
    assert sorted(raster_by_gene) == ["A", "B", "C", "D"]
    for gene in raster_by_gene:
        np.testing.assert_array_equal(raster_by_gene[gene], polygon_by_gene[gene])

    # Per-transcript distances: the raster value at each transcript's pixel
    # is within one pixel of the polygon engine's point-to-boundary distance.
    encoding = distance_encoding(config, UM_PER_PX)
    genes = points[points["feature_name"].isin(["A", "B", "C", "D"])]
    column = np.floor(genes["x"].to_numpy() / UM_PER_PX).astype(int)
    row = np.floor(genes["y"].to_numpy() / UM_PER_PX).astype(int)
    for boundary, labels, polygon_values in (
        ("cell", fixture["cell_labels"], polygon_data.signed_cell_distance_um),
        ("nucleus", fixture["nucleus_labels"], polygon_data.signed_nucleus_distance_um),
    ):
        out = np.zeros(labels.shape, dtype=np.uint16)
        build_distance_raster(
            labels, out, um_per_px=UM_PER_PX, encoding=encoding, tile_px=64, n_workers=1
        )
        inside = labels[row, column] > 0
        decoded = encoding.decode(out[row, column])
        signed = np.where(inside, decoded, -decoded)
        # Polygon order follows the stream; match rows by coordinates.
        polygon_frame = pd.DataFrame(
            {
                "x": polygon_data.coordinates[:, 0],
                "y": polygon_data.coordinates[:, 1],
                "d": polygon_values,
            }
        )
        raster_frame = pd.DataFrame(
            {
                "x": genes["x"].to_numpy(dtype=np.float32),
                "y": genes["y"].to_numpy(dtype=np.float32),
                "r": signed,
            }
        )
        joined = raster_frame.merge(polygon_frame, on=["x", "y"])
        assert len(joined) == len(genes)
        assert np.all(np.sign(joined["r"]) == np.sign(joined["d"]))
        assert np.max(np.abs(joined["r"] - joined["d"])) <= UM_PER_PX + 1e-6, boundary

    # The medians agree within a pixel and the fine bin.
    polygon_median = dict(
        zip(polygon.gene_names, polygon.median_signed_cell_distance_um, strict=True)
    )
    for gene, median in zip(
        result.counts.gene_names,
        result.counts.median_signed_cell_distance_um,
        strict=True,
    ):
        assert abs(median - polygon_median[gene]) <= UM_PER_PX + 0.05 + 1e-6


def test_results_do_not_depend_on_batch_size_files_or_tiles(tmp_path: Path) -> None:
    """Batches, files, distance tiles and threads only change the schedule."""
    fixture = _fixture(seed=3)
    reference = _run(fixture, tmp_path / "a", n_files=1, transcript_chunk_rows=10_000)
    for n_files, rows, tile, workers in ((3, 7, 64, 1), (5, 113, 4096, 4)):
        other = _run(
            fixture,
            tmp_path / f"b{n_files}",
            n_files=n_files,
            transcript_chunk_rows=rows,
            settings=RasterEngineSettings(distance_tile_px=tile, n_workers=workers),
        )
        pd.testing.assert_frame_equal(
            reference.summary, other.summary, check_exact=True
        )
        pd.testing.assert_frame_equal(
            reference.signed_distance, other.signed_distance, check_exact=True
        )
        assert reference.totals == other.totals


def test_every_transcript_is_used_and_controls_are_counted(tmp_path: Path) -> None:
    """Nothing is thinned: every in-tissue gene transcript enters the counts."""
    fixture = _fixture(seed=5)
    result = _run(fixture, tmp_path)
    points = fixture["points"]
    n_controls = int(
        points["feature_name"].isin(["Intergenic_Region_1", "NegControlProbe_7"]).sum()
    )

    assert result.totals["n_input"] == len(points)
    assert result.totals["n_controls_excluded"] == n_controls
    assert result.totals["n_used"] == len(points) - n_controls
    assert result.totals["n_used"] + n_controls == result.totals["n_input"]
    assert int(result.summary["n_transcripts"].sum()) == result.totals["n_used"]
    expected = points["feature_name"].astype(str).value_counts()
    for gene, n in zip(
        result.summary["gene"], result.summary["n_transcripts"], strict=True
    ):
        assert n == expected[gene]
    # No pair correlation on this engine, and the summary says so.
    assert result.paircorr.empty
    assert set(result.summary["paircorr_status"]) == {PAIRCORR_STATUS}
    assert result.record["paircorr"]["status"] == PAIRCORR_STATUS
    # Every gene transcript on a cell label is counted, as the vendor would.
    on_cell = (
        fixture["cell_labels"][
            np.floor(points["y"] / UM_PER_PX).astype(int),
            np.floor(points["x"] / UM_PER_PX).astype(int),
        ]
        > 0
    )
    is_gene = ~points["feature_name"].isin(["Intergenic_Region_1", "NegControlProbe_7"])
    assert result.totals["n_gene_transcripts_on_cell_label"] == int(
        (on_cell & is_gene).sum()
    )


def test_diagnostic_raster_sums_equal_gene_totals(tmp_path: Path) -> None:
    """The figures bin every transcript of a gene: no point is dropped."""
    fixture = _fixture(seed=7)
    result = _run(fixture, tmp_path, transcript_diagnostic_window_um=40.0)
    config = _config(transcript_diagnostic_window_um=40.0)
    rasters = gene_diagnostic_rasters(result, ["A", "C", "D"], config=config)

    totals = dict(
        zip(result.summary["gene"], result.summary["n_transcripts"], strict=True)
    )
    points = fixture["points"]
    for gene, item in rasters.items():
        assert int(item.tissue_counts.sum()) == totals[gene]
        x0, y0 = item.window_origin_um
        mine = points[points["feature_name"] == gene]
        in_window = (
            (mine["x"] >= x0)
            & (mine["x"] < x0 + item.window_um)
            & (mine["y"] >= y0)
            & (mine["y"] < y0 + item.window_um)
        )
        assert sum(int(v.sum()) for v in item.window_counts.values()) == int(
            in_window.sum()
        )
    # Gene A is nuclear: its window raster has no extracellular transcript.
    assert int(rasters["A"].window_counts["extracellular"].sum()) == 0
    assert int(rasters["A"].window_counts["nuclear"].sum()) > 0


def test_distance_raster_clips_beyond_the_halo() -> None:
    """Distances beyond the exact range are encoded as the clip code."""
    labels = np.zeros((64, 400), dtype=np.uint32)
    labels[:, :10] = 1
    config = _config(signed_distance_edges_um=[-20.0, 0.0, 20.0])
    encoding = distance_encoding(config, UM_PER_PX)
    out = np.zeros(labels.shape, dtype=np.uint16)
    build_distance_raster(
        labels, out, um_per_px=UM_PER_PX, encoding=encoding, tile_px=32, n_workers=2
    )
    decoded = encoding.decode(out[0])
    # Column 10 is the first pixel off the label: half a pixel from the edge.
    assert decoded[10] == pytest.approx(0.5 * UM_PER_PX)
    assert decoded[9] == pytest.approx(0.5 * UM_PER_PX)
    assert decoded[30] == pytest.approx(20.5 * UM_PER_PX)
    assert np.isinf(decoded[200])
    assert out[0, 200] == DISTANCE_CLIP_CODE


# --------------------------------------------------------------------------
# Tissue support
# --------------------------------------------------------------------------


def test_density_mask_finds_tissue_fills_holes_and_drops_debris() -> None:
    """Without annotations the tissue comes from the all-transcript density."""
    rng = np.random.default_rng(0)
    counts = rng.poisson(1.5, size=(120, 160))  # sparse off-tissue background
    yy, xx = np.mgrid[:120, :160]
    blob = ((yy - 60) ** 2 / 40**2 + (xx - 70) ** 2 / 50**2) <= 1.0
    hole = ((yy - 60) ** 2 + (xx - 70) ** 2) <= 6**2
    counts[blob] += rng.poisson(600, size=int(blob.sum()))
    counts[hole] = 0  # e.g. a vessel: no transcripts, still tissue
    counts[5:7, 150:152] += 900  # debris: 4 bins (400 um2)

    mask = density_tissue_mask(
        counts, origin_um=(0.0, 0.0), bin_um=10.0, min_component_area_um2=5_000.0
    )

    assert mask.method == "density"
    assert mask.inside[blob].all()
    assert mask.inside[hole].all()
    assert not mask.inside[5:7, 150:152].any()
    assert mask.inside.sum() <= blob.sum() * 1.1
    assert mask.parameters["n_components_removed"] >= 1
    assert mask.parameters["threshold_rule"] in {"otsu_log10", "fraction_of_p99"}
    assert mask.record()["area_um2"] == pytest.approx(mask.inside.sum() * 100.0)


def test_density_mask_inside_tissue_crop_keeps_sparse_tissue() -> None:
    """A window with no background is not split into dense and sparse tissue."""
    rng = np.random.default_rng(1)
    counts = rng.poisson(800, size=(80, 80))
    counts[:, 40:] = rng.poisson(120, size=(80, 40))  # a sparse layer

    mask = density_tissue_mask(counts, origin_um=(0.0, 0.0), bin_um=10.0)

    assert mask.inside.all()


def test_annotation_mask_is_exact_at_its_boundary() -> None:
    """Transcripts in boundary bins get the exact polygon test."""
    polygon = box(3.0, 3.0, 47.0, 31.0)
    mask = polygon_tissue_mask(polygon, origin_um=(0.0, 0.0), bin_um=10.0, shape=(4, 5))
    x = np.array([2.9, 3.1, 25.0, 46.9, 47.1, 25.0])
    y = np.array([10.0, 10.0, 20.0, 20.0, 20.0, 31.2])
    np.testing.assert_array_equal(
        mask.contains(x, y), [False, True, True, True, False, False]
    )
    assert mask.method == "annotation"
    assert mask.area_um2 == pytest.approx(polygon.area)


# --------------------------------------------------------------------------
# Engine selection and the stage on a native ATERA store
# --------------------------------------------------------------------------


def test_engine_selection_is_keyed_on_capability(tmp_path: Path) -> None:
    """Label elements -> raster; else a wide panel is refused; else polygon."""
    from merxen.analysis.spatial_gene_analysis import select_transcript_engine

    xenium = SpatialGeneAnalysisSampleConfig(
        sample_id="X", platform="XENIUM", zarr_path=tmp_path / "x.zarr"
    )
    assert select_transcript_engine(xenium, n_panel_genes=300) == "polygon"
    assert select_transcript_engine(xenium, n_panel_genes=5_000) == "polygon"
    with pytest.raises(ValueError, match="5,001 genes"):
        select_transcript_engine(xenium, n_panel_genes=5_001)

    store = tmp_path / "atera.zarr"
    for key in ("cell_labels", "nucleus_labels", "proseg_cell_labels"):
        (store / "labels" / key).mkdir(parents=True)
    for segmentation in ("original_seg", "reseg"):
        atera = SpatialGeneAnalysisSampleConfig(
            sample_id="A",
            platform="ATERA",
            zarr_path=store,
            segmentation=segmentation,
        )
        assert select_transcript_engine(atera, n_panel_genes=18_049) == "label_raster"
    assert raster.layer_label_keys(store, "ATERA", "reseg") == (
        "proseg_cell_labels",
        "nucleus_labels",
    )
    missing = SpatialGeneAnalysisSampleConfig(
        sample_id="A",
        platform="ATERA",
        zarr_path=tmp_path / "no_labels.zarr",
        segmentation="original_seg",
    )
    with pytest.raises(ValueError, match="no cell and"):
        select_transcript_engine(missing, n_panel_genes=18_049)


@pytest.mark.integration
def test_native_atera_store_runs_the_label_raster_engine(tmp_path: Path) -> None:
    """The stage on a native store: density mask, every transcript, outputs."""
    import anndata as ad
    from test_io.atera_native_fixture import make_native_reference

    from merxen.analysis.spatial_gene_analysis import run_spatial_gene_analysis

    _, latest = make_native_reference(
        tmp_path / "ref", n_cells=20, tiles={"0,0": 1500, "1,0": 1500}
    )
    config = SpatialGeneAnalysisConfig.model_validate(
        {
            "pair_id": "ATERA_SYNTH",
            "output_dir": tmp_path / "sga_out",
            "samples": [
                {
                    "sample_id": "ATERA_SYNTH",
                    "platform": "ATERA",
                    "zarr_path": latest,
                    "segmentation": "original_seg",
                    "table_key": "table_original",
                    "shape_key": "atera_cell_boundaries",
                    "nuclei_shape_key": "atera_nucleus",
                }
            ],
            "min_cells": 1,
            "top_n": 2,
            "figure_dpi": 72,
            "transcript_min_count": 2,
            "transcript_diagnostic_max_genes": 2,
        }
    )

    results = run_spatial_gene_analysis(
        config,
        raster_settings=RasterEngineSettings(tissue_min_component_area_um2=0.0),
    )["ATERA_SYNTH"]

    manifest = json.loads(Path(results["manifest"]).read_text())
    engine = manifest["transcript_engine"]
    assert engine["engine"] == "label_raster"
    assert engine["tissue"]["method"] == "density"
    assert engine["paircorr"]["status"] == PAIRCORR_STATUS
    assert engine["cell_labels"] == "cell_labels"
    counts = manifest["transcript_counts"]
    assert counts["n_transcripts_used"] > 0
    assert counts["n_transcripts_used"] == sum(
        counts[f"n_{name}_transcripts"] for name in COMPARTMENTS
    )
    table = ad.read_zarr(Path(latest) / "tables" / "table_original")
    assert counts["n_gene_transcripts_on_cell_label"] == int(table.X.sum())
    summary = pd.read_csv(results["transcript_summary_csv"])
    assert (summary["paircorr_status"] == PAIRCORR_STATUS).all()
    assert pd.read_parquet(results["paircorr_parquet"]).empty
    assert Path(results["transcript_tissue_mask_npz"]).is_file()
    assert list(Path(results["transcript_diagnostics_dir"]).glob("*.png"))
    assert not list((tmp_path / "sga_out").parent.glob("merxen_transcript_raster_*"))

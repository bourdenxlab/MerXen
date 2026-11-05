"""Atera readers on the public 10x dataset (slow; skipped without the data).

The data lives under ``$MERXEN_ATERA_RAW_ROOT`` (``.env``); the tests only read
it. They check what the synthetic fixture cannot prove: that the real format
matches the assumptions of :mod:`merxen.io.atera_reader`.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import anndata as ad
import dask
import numpy as np
import pandas as pd
import pytest
import shapely
from scipy import sparse

from merxen.gene_ids import is_ensembl_gene_id
from merxen.io import atera_reader as ar
from merxen.segmentation.mask_geometry import masks_to_labeled_polygons

from .atera_real import extracted_outs

pytestmark = pytest.mark.slow

SMALL_TILE = "5,4"


@pytest.fixture
def bundle(real_path: Callable[[str], Path]) -> ar.AteraBundle:
    """The extracted public dataset, resolved."""
    outs = extracted_outs(real_path("MERXEN_ATERA_RAW_ROOT"))
    return ar.resolve_atera_bundle(outs, species="human")


def test_real_bundle_header(bundle: ar.AteraBundle) -> None:
    assert bundle.analysis_sw_version.startswith("atera-")
    assert bundle.pixel_size_um == pytest.approx(0.21249222)
    assert bundle.morphology_channel_names[0] == "DAPI"
    assert "18S" in bundle.morphology_channel_names
    assert len(bundle.morphology_2d_files) == len(bundle.morphology_channel_names)
    cells = ar.read_atera_cells(bundle)
    assert cells.mask_shape == bundle.morphology_shape_yx
    assert cells.source_cell_id[0].endswith("-1")
    np.testing.assert_array_equal(
        np.bincount(cells.nucleus_cell_index, minlength=cells.n_cells),
        cells.summary["nucleus_count"].to_numpy(),
    )


def test_real_small_tile(bundle: ar.AteraBundle) -> None:
    tiles = {tile.key: tile for tile in ar.read_atera_tile_index(bundle)}
    tile = tiles[SMALL_TILE]
    names = ar.read_atera_gene_names(bundle)
    with ar.open_zip_group(bundle.transcripts_zip) as root:
        mapping = np.asarray(root.attrs["codeword_gene_mapping"])
        is_gene = ar._category_column(root, "gene_category", "is_gene")
    with ar.open_zip_group(bundle.cells_zip) as root:
        cell_id = np.asarray(root["cell_id"][:])
    frame = ar.read_transcript_partition(
        bundle,
        ar.PartitionSpec(SMALL_TILE, 0, tile.n_rows, tile.global_offset),
        gene_dtype=pd.CategoricalDtype(list(names)),
        is_gene=is_gene,
        cell_lookup=ar.CellKeyLookup.from_cell_ids(cell_id[:, 0], cell_id[:, 1]),
        min_qv=None,
        tile_dtype=pd.CategoricalDtype(list(tiles)),
        emit_codeword_index=True,
    )
    assert len(frame) == tile.n_rows
    codes = frame["feature_name"].cat.codes.to_numpy()
    np.testing.assert_array_equal(codes, mapping[frame["codeword_index"].to_numpy()])
    assigned = frame["original_assignment"].notna().to_numpy()
    assert 0 < assigned.sum() < len(frame)
    assert not (assigned & (frame["qv"].to_numpy() < 20)).any()
    assert not (frame["overlaps_nucleus"].to_numpy() & ~assigned).any()
    assert frame["original_assignment"].max() <= len(cell_id)
    np.testing.assert_array_equal(
        frame["transcript_id"].to_numpy(),
        tile.global_offset + 1 + np.arange(tile.n_rows, dtype=np.uint64),
    )


def test_real_features_and_cell_order(bundle: ar.AteraBundle) -> None:
    features = ar.read_atera_features(bundle)
    assert tuple(features["feature_name"]) == ar.read_atera_gene_names(bundle)
    genes = features[features["feature_type"] == ar.GENE_EXPRESSION_TYPE]
    assert genes["feature_name"].is_unique
    assert all(is_ensembl_gene_id(value) for value in genes.index)
    ensg_shaped = genes["feature_name"].map(is_ensembl_gene_id)
    assert 0 < int(ensg_shaped.sum()) < len(genes)
    cells = ar.read_atera_cells(bundle)
    with ar.open_zip_group(bundle.cell_feature_matrix_zip) as root:
        packed = np.asarray(ad.io.read_elem(root["obs/cell_id"]), dtype=np.int64)
        barcodes = np.asarray(ad.io.read_elem(root["obs/barcode"])).astype(object)
    np.testing.assert_array_equal(packed, cells.vendor_cell_id)
    np.testing.assert_array_equal(barcodes, cells.source_cell_id)


def test_real_metadata_is_kept_whole(bundle: ar.AteraBundle, tmp_path: Path) -> None:
    stored = {sidecar.stored_path: sidecar for sidecar in bundle.sidecars}
    assert {
        "experiment.spatial",
        "metrics_summary.csv",
        "analysis_summary.html",
        "qc_images.html",
        "panel_config.json",
        "morphology_fov_locations.json",
        "overview_scan_fov_locations.json",
        "overview_scan.png",
        "diffexp_genes/gene_expression_graphclust.csv",
    } <= set(stored)
    records = ar.copy_atera_sidecars(bundle, tmp_path)
    assert len(records) == len(bundle.sidecars)
    for sidecar in bundle.sidecars:
        copied = tmp_path / ar.ATERA_SIDECAR_DIR / sidecar.stored_path
        assert copied.read_bytes() == sidecar.source.read_bytes()
    attrs = ar.atera_metadata_attrs(bundle, records)
    json.dumps(attrs, allow_nan=False)
    assert attrs["experiment"] == json.loads(
        (bundle.root / "experiment.spatial").read_text(encoding="utf-8")
    )
    metrics = stored["metrics_summary.csv"].source
    header = metrics.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert list(attrs["metrics_summary"][0]) == header
    assert attrs["panel_config"]["payloads"][0]["num_targets"] > 18000
    explorer = attrs["experiment"]["explorer_files"]
    assert ar.stable_sidecar_name(explorer["transcripts_viz_zarr_filepath"]) == (
        "binned_transcripts.zarr.zip"
    )
    assert ar.stable_sidecar_name(explorer["cell_feature_viz_zarr_filepath"]) == (
        "csc_cell_feature_matrix.zarr.zip"
    )


def test_real_default_simplification_keeps_centroids(bundle: ar.AteraBundle) -> None:
    cells = ar.read_atera_cells(bundle)
    r0, c0, size = 20000, 30000, 1500
    crop = np.asarray(
        ar.read_atera_labels(bundle)["cell_labels"][r0 : r0 + size, c0 : c0 + size]
    ).astype(np.int64)
    border = np.unique(np.concatenate([crop[0], crop[-1], crop[:, 0], crop[:, -1]]))
    interior = np.where(np.isin(crop, border), 0, crop).astype(np.int32)
    labeled = masks_to_labeled_polygons(interior, n_jobs=1)
    assert len(labeled) > 20
    rows = np.asarray([label for label, _ in labeled]) - 1
    shifted = shapely.transform(
        np.asarray([geometry for _, geometry in labeled], dtype=object),
        lambda coords: coords + np.array([c0, r0]),
    )
    shapes = ar._pixel_polygons_to_microns(
        shifted, bundle.pixel_size_um, ar.DEFAULT_SIMPLIFY_PX
    )
    centroids = shapely.centroid(shapes)
    summary = cells.summary
    error = np.hypot(
        shapely.get_x(centroids) - summary["cell_centroid_x"].to_numpy()[rows],
        shapely.get_y(centroids) - summary["cell_centroid_y"].to_numpy()[rows],
    )
    assert error.max() < 0.01


#: Tile whose interior cells are checked against the vendor values (plan §2).
FIDELITY_TILE = "6,1"


def test_real_crop_matches_the_vendor_cell_values(bundle: ar.AteraBundle) -> None:
    """Labels, nucleus owners and Q20 counts agree with the vendor's own values.

    The crop is one transcript tile's pixel window; only cells that do not
    touch its border are compared, so all their transcripts lie in the tile.
    """
    ps = bundle.pixel_size_um
    cells = ar.read_atera_cells(bundle)
    summary = cells.summary
    tile = {item.key: item for item in ar.read_atera_tile_index(bundle)}[FIDELITY_TILE]
    x0, x1, y0, y1 = tile.bounds_um
    height, width = bundle.morphology_shape_yx
    c0, c1 = max(0, int(x0 / ps)), min(width, int(x1 / ps))
    r0, r1 = max(0, int(y0 / ps)), min(height, int(y1 / ps))
    labels = ar.read_atera_labels(bundle)
    cell_mask = np.asarray(labels["cell_labels"][r0:r1, c0:c1]).astype(np.int64)
    nucleus_mask = np.asarray(labels["nucleus_labels"][r0:r1, c0:c1]).astype(np.int64)
    border = np.unique(
        np.concatenate([cell_mask[0], cell_mask[-1], cell_mask[:, 0], cell_mask[:, -1]])
    )
    interior = np.setdiff1d(np.unique(cell_mask), np.concatenate([[0], border]))
    assert interior.size > 100
    rows = interior - 1

    # Cell label = row + 1: areas and corner-convention centroids.
    yy, xx = np.nonzero(cell_mask)
    values = cell_mask[yy, xx]
    n_labels = int(values.max()) + 1
    area = np.bincount(values, minlength=n_labels)[interior]
    cx = np.bincount(values, weights=xx, minlength=n_labels)[interior] / area
    cy = np.bincount(values, weights=yy, minlength=n_labels)[interior] / area
    np.testing.assert_allclose(
        area * ps * ps, summary["cell_area"].to_numpy()[rows], rtol=1e-5
    )
    np.testing.assert_allclose(
        (cx + c0 + 0.5) * ps, summary["cell_centroid_x"].to_numpy()[rows], atol=0.01
    )
    np.testing.assert_allclose(
        (cy + r0 + 0.5) * ps, summary["cell_centroid_y"].to_numpy()[rows], atol=0.01
    )

    # Nucleus label = nucleus row + 1, owned through polygon_sets/0/cell_index.
    on_nucleus = nucleus_mask > 0
    owners = cells.nucleus_cell_index[nucleus_mask[on_nucleus] - 1] + 1
    assert np.mean(cell_mask[on_nucleus] == owners) > 0.99

    # Per-cell QV >= 20 counts reproduce the vendor matrix exactly.
    with dask.config.set(scheduler="threads", num_workers=8):
        points, _, _ = ar.read_atera_transcripts(
            bundle, min_qv=20.0, tile_allowlist=[FIDELITY_TILE]
        )
        frame = points.compute()
    assigned = (
        frame["original_assignment"].to_numpy(dtype=np.float64, na_value=0)
    ).astype(np.int64)
    keep = np.isin(assigned, interior)
    order = {int(cell): k for k, cell in enumerate(interior)}
    counted = sparse.csr_matrix(
        (
            np.ones(int(keep.sum())),
            (
                np.asarray([order[int(cell)] for cell in assigned[keep]]),
                frame["feature_name"].cat.codes.to_numpy()[keep],
            ),
        ),
        shape=(interior.size, len(ar.read_atera_gene_names(bundle))),
    )
    with ar.open_zip_group(bundle.cell_feature_matrix_zip) as group:
        vendor = sparse.csr_matrix(ad.io.sparse_dataset(group["X"])[rows])
    assert abs(counted - vendor).sum() == 0

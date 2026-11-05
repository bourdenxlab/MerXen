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
import numpy as np
import pandas as pd
import pytest

from merxen.gene_ids import is_ensembl_gene_id
from merxen.io import atera_reader as ar

pytestmark = pytest.mark.slow

PUBLIC_DATASET = "Atera_v1_WTA_S4A_Brain_Hippocampus_ND_FF"
SMALL_TILE = "5,4"


@pytest.fixture
def bundle(real_path: Callable[[str], Path]) -> ar.AteraBundle:
    """The extracted public dataset, resolved."""
    outs = real_path("MERXEN_ATERA_RAW_ROOT") / PUBLIC_DATASET / "outs"
    if not (outs / ".extract_complete").exists():
        pytest.skip("the public Atera bundle is not extracted on this host")
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


def test_real_metadata_is_sanitised(bundle: ar.AteraBundle) -> None:
    identifying = [
        str(bundle.experiment[key])
        for key in ar.IDENTIFYING_EXPERIMENT_FIELDS
        if bundle.experiment.get(key) not in (None, "")
    ]
    assert identifying
    sanitized = json.dumps(ar.sanitize_experiment_spatial(bundle.experiment))
    assert not [value for value in identifying if value in sanitized]
    metrics = next(p for p in bundle.sidecars if p.name.startswith("metrics_summary"))
    assert ar.neutral_sidecar_name(metrics.name) == "metrics_summary.csv"
    clean = ar.sanitize_metrics_summary(metrics.read_text(encoding="utf-8"))
    header = clean.splitlines()[0].split(",")
    assert "run_name" not in header and "slide_id" not in header
    assert bundle.panel_config_file is not None
    panel = ar.sanitize_panel_config(
        json.loads(bundle.panel_config_file.read_text(encoding="utf-8"))
    )
    assert all("customer" not in payload for payload in panel["payloads"])

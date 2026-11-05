"""Equivalence tests for the per-shape count accumulation in enrichment."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from shapely.geometry import box

from merxen.enrichment.assignment import compute_table_from_points_for_shape

_GENES = ["GeneA", "GeneB", "GeneC", "GeneD", "GeneE"]


def _reference_counts(
    points: pd.DataFrame,
    shapes: gpd.GeoDataFrame,
    gene_list: list[str],
    chunk_rows: int,
) -> sparse.csr_matrix:
    """Count points per shape and gene with the original accumulation.

    This keeps the implementation that summed one CSR matrix per chunk, so the
    streamed accumulation can be compared against it.
    """
    gdf_shapes = shapes[["instance_id", "geometry"]].copy()
    gdf_shapes = gdf_shapes[
        gdf_shapes.geometry.notna() & ~gdf_shapes.geometry.is_empty
    ].copy()
    gdf_shapes["instance_id"] = gdf_shapes["instance_id"].astype("uint64")
    cell_ids = gdf_shapes["instance_id"].astype("uint64").tolist()
    cell_to_idx = {int(cell_id): i for i, cell_id in enumerate(cell_ids)}
    genes = [str(g) for g in gene_list]
    gene_to_idx = {gene: i for i, gene in enumerate(genes)}
    counts_csr = sparse.csr_matrix((len(cell_ids), len(genes)), dtype=np.int64)

    for start in range(0, len(points), chunk_rows):
        chunk = points.iloc[start : start + chunk_rows]
        xv = pd.to_numeric(chunk["x"], errors="coerce").to_numpy(np.float64)
        yv = pd.to_numeric(chunk["y"], errors="coerce").to_numpy(np.float64)
        gv = chunk["gene"].astype(str).to_numpy(dtype=object)
        valid = np.isfinite(xv) & np.isfinite(yv) & pd.notna(gv) & (gv != "")
        if not np.any(valid):
            continue
        point_frame = gpd.GeoDataFrame(
            {"gene": pd.Series(gv[valid], dtype=str)},
            geometry=gpd.points_from_xy(xv[valid], yv[valid]),
            crs=gdf_shapes.crs,
        ).reset_index(drop=True)
        joined = gpd.sjoin(
            point_frame,
            gdf_shapes.reset_index(drop=True),
            how="left",
            predicate="within",
        )
        cell_series = pd.to_numeric(joined["instance_id"], errors="coerce").astype(
            "UInt64"
        )
        assigned_mask = cell_series.notna()
        if not assigned_mask.any():
            continue
        assigned_cells = cell_series.loc[assigned_mask].to_numpy(dtype=np.uint64)
        assigned_genes = (
            joined.loc[assigned_mask, "gene"].astype(str).to_numpy(dtype=object)
        )
        cidx = np.fromiter(
            (cell_to_idx.get(int(cell), -1) for cell in assigned_cells),
            dtype=np.int64,
            count=len(assigned_cells),
        )
        gidx = np.fromiter(
            (gene_to_idx.get(gene, -1) for gene in assigned_genes),
            dtype=np.int64,
            count=len(assigned_genes),
        )
        keep = (cidx >= 0) & (gidx >= 0)
        if np.any(keep):
            chunk_mat = sparse.coo_matrix(
                (np.ones(int(np.sum(keep)), dtype=np.int64), (cidx[keep], gidx[keep])),
                shape=(len(cell_ids), len(genes)),
            ).tocsr()
            counts_csr = counts_csr + chunk_mat
    return counts_csr


def _grid_shapes(n_side: int, *, ids: list[int] | None = None) -> gpd.GeoDataFrame:
    """Return unit squares on a grid with a gap between neighbours."""
    geometries = [
        box(col * 2.0, row * 2.0, col * 2.0 + 1.5, row * 2.0 + 1.5)
        for row in range(n_side)
        for col in range(n_side)
    ]
    instance_ids = ids if ids is not None else list(range(1, len(geometries) + 1))
    return gpd.GeoDataFrame(
        {"instance_id": instance_ids, "geometry": geometries}, geometry="geometry"
    )


def _random_points(n_points: int, extent: float, seed: int) -> pd.DataFrame:
    """Return points with unknown genes, blank genes and missing coordinates."""
    rng = np.random.default_rng(seed)
    gene_pool = np.asarray([*_GENES, "NotInPanel", ""], dtype=object)
    points = pd.DataFrame(
        {
            "x": rng.uniform(-1.0, extent, n_points),
            "y": rng.uniform(-1.0, extent, n_points),
            "gene": rng.choice(gene_pool, n_points),
        }
    )
    points.loc[rng.choice(n_points, n_points // 50, replace=False), "x"] = np.nan
    return points


def _assert_same_csr(actual: object, expected: sparse.csr_matrix) -> None:
    """Assert identical sparse format, dtypes, structure and values."""
    assert isinstance(actual, sparse.csr_matrix)
    assert actual.shape == expected.shape
    assert actual.data.dtype == expected.data.dtype
    assert actual.indices.dtype == expected.indices.dtype
    assert actual.indptr.dtype == expected.indptr.dtype
    np.testing.assert_array_equal(actual.indptr, expected.indptr)
    np.testing.assert_array_equal(actual.indices, expected.indices)
    np.testing.assert_array_equal(actual.data, expected.data)


def _table_for(
    points: pd.DataFrame | dd.DataFrame,
    shapes: gpd.GeoDataFrame,
    chunk_rows: int,
    gene_list: list[str] = _GENES,
) -> ad.AnnData:
    table, _ = compute_table_from_points_for_shape(
        dataset_name="S1_XENIUM",
        points_obj=points,
        shape_gdf=shapes,
        shape_id_col="instance_id",
        shape_key="MOSAIK_proseg",
        gene_list=gene_list,
        chunk_rows=chunk_rows,
    )
    return table


@pytest.mark.parametrize("chunk_rows", [97, 10_000])
def test_counts_match_the_reference_accumulation(chunk_rows: int) -> None:
    """Many chunks and one chunk both reproduce the original count matrix."""
    shapes = _grid_shapes(6)
    points = _random_points(1500, extent=12.0, seed=1)

    table = _table_for(points, shapes, chunk_rows)

    _assert_same_csr(table.X, _reference_counts(points, shapes, _GENES, chunk_rows))
    assert table.X.nnz > 0


def test_counts_match_the_reference_for_a_whole_transcriptome_panel() -> None:
    """Thousands of genes over many chunks give the original matrix."""
    gene_list = [f"Gene{index:05d}" for index in range(20_000)]
    rng = np.random.default_rng(6)
    shapes = _grid_shapes(6)
    points = pd.DataFrame(
        {
            "x": rng.uniform(0.0, 12.0, 3000),
            "y": rng.uniform(0.0, 12.0, 3000),
            "gene": rng.choice(np.asarray([*gene_list[::7], "Unlisted"]), 3000),
        }
    )

    table = _table_for(points, shapes, chunk_rows=250, gene_list=gene_list)

    _assert_same_csr(
        table.X, _reference_counts(points, shapes, gene_list, chunk_rows=250)
    )
    assert table.n_vars == 20_000


def test_counts_match_the_reference_with_overlapping_shapes() -> None:
    """A point inside two shapes is counted for both, as before."""
    shapes = pd.concat(
        [
            _grid_shapes(3),
            gpd.GeoDataFrame(
                {
                    "instance_id": [40, 41],
                    "geometry": [box(0, 0, 6, 6), box(1, 1, 3, 3)],
                },
                geometry="geometry",
            ),
        ],
        ignore_index=True,
    )
    points = _random_points(3000, extent=6.0, seed=2)

    table = _table_for(points, shapes, chunk_rows=211)

    expected = _reference_counts(points, shapes, _GENES, chunk_rows=211)
    _assert_same_csr(table.X, expected)
    assert int(expected.sum()) > int(table.n_obs)


def test_counts_match_the_reference_with_sparse_unsorted_ids() -> None:
    """Rows follow the shape order with non-contiguous, unsorted IDs."""
    ids = [90, 3, 17, 5, 1000, 42, 8, 61, 2]
    shapes = _grid_shapes(3, ids=ids)
    points = _random_points(2000, extent=6.0, seed=3)

    table = _table_for(points, shapes, chunk_rows=150)

    _assert_same_csr(table.X, _reference_counts(points, shapes, _GENES, 150))
    assert table.obs["instance_id"].tolist() == ids


def test_counts_match_the_reference_from_dask_partitions() -> None:
    """Partitioned points accumulate to the same matrix as one frame."""
    shapes = _grid_shapes(5)
    points = _random_points(3000, extent=10.0, seed=4)
    partitioned = dd.from_pandas(points, npartitions=4)

    table = _table_for(partitioned, shapes, chunk_rows=1_000_000)

    expected = _reference_counts(points, shapes, _GENES, chunk_rows=len(points))
    _assert_same_csr(table.X, expected)


def test_counts_match_the_reference_when_nothing_is_assigned() -> None:
    """Points outside every shape give the original empty matrix."""
    shapes = _grid_shapes(2)
    points = pd.DataFrame({"x": [50.0, 60.0], "y": [50.0, 60.0], "gene": "GeneA"})

    table = _table_for(points, shapes, chunk_rows=1)

    _assert_same_csr(table.X, _reference_counts(points, shapes, _GENES, 1))
    assert table.X.nnz == 0


def test_written_count_matrix_is_byte_identical(tmp_path: Path) -> None:
    """The count matrix serialises to the same bytes as the original one."""
    shapes = _grid_shapes(6)
    points = _random_points(1500, extent=12.0, seed=5)
    table = _table_for(points, shapes, chunk_rows=313)
    reference = table.copy()
    reference.X = _reference_counts(points, shapes, _GENES, chunk_rows=313)

    table.write_zarr(tmp_path / "streamed.zarr")
    reference.write_zarr(tmp_path / "reference.zarr")

    streamed_files = sorted(
        path.relative_to(tmp_path / "streamed.zarr")
        for path in (tmp_path / "streamed.zarr" / "X").rglob("*")
        if path.is_file()
    )
    assert streamed_files
    for relative in streamed_files:
        assert (tmp_path / "streamed.zarr" / relative).read_bytes() == (
            tmp_path / "reference.zarr" / relative
        ).read_bytes()

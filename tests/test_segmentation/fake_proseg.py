#!/usr/bin/env python3
"""A stand-in ProSeg 3.2.0 for the ATERA ProSeg route tests (plan AT-9).

It accepts the command of ``merxen.segmentation.atera_proseg.runner`` and:

- prints ``proseg 3.2.0`` for ``--version`` (``FAKE_PROSEG_VERSION`` changes it);
- reads the Xenium-layout parquet and **asserts** its schema (plain strings,
  uint64 ids, uint8 compartment, float32 coordinates and QV, constant fov);
- makes one cell per non-zero label of the prior mask, keyed by the label
  (``original_cell_id``), with the pixel outline of the label as its polygon;
- assigns each row to its ``cell_id`` (``FAKE_PROSEG_MODE=identity``) or also
  gives unassigned rows within ``FAKE_PROSEG_GREEDY_UM`` microns (default 2) of
  a cell's pixels to the nearest cell (``greedy_halo``, which makes two tiles
  claim one halo row);
- prints the dataset summary with ProSeg's prior-weight scaling
  (``FAKE_PROSEG_WEIGHT_FACTOR`` skews it) and writes the cell metadata,
  transcript metadata, polygons and ``proseg.zarr`` count table
  (``FAKE_PROSEG_CORRUPT_X=1`` adds one count to the first cell).
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

EXPECTED = {
    "transcript_id": pa.uint64(),
    "x_location": pa.float32(),
    "y_location": pa.float32(),
    "z_location": pa.float32(),
    "feature_name": pa.string(),
    "cell_id": pa.string(),
    "overlaps_nucleus": pa.uint8(),
    "qv": pa.float32(),
    "fov_name": pa.string(),
}


def _log(message: str) -> None:
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{stamp} INFO  proseg] {message}", file=sys.stderr, flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--version", action="store_true")
    parser.add_argument("inputs", nargs="*")
    for flag in (
        "-x",
        "-y",
        "-z",
        "--gene-column",
        "--cell-id-column",
        "--cell-id-unassigned",
        "--transcript-id-column",
        "--qv-column",
        "--min-qv",
        "--compartment-column",
        "--compartment-nuclear",
        "--fov-column",
        "--voxel-size",
        "--burnin-voxel-size",
        "--voxel-layers",
        "--nuclear-reassignment-prob",
        "--diffusion-probability",
        "--cell-compactness",
        "--burnin-samples",
        "--samples",
        "--hillclimb",
        "--recorded-samples",
        "--prior-weight",
        "--cellpose-masks",
        "--nthreads",
        "--output-path",
        "--output-spatialdata",
        "--output-cell-metadata",
        "--output-transcript-metadata",
        "--output-cell-polygons",
        "--prior-seg-reassignment-prob",
    ):
        parser.add_argument(flag)
    parser.add_argument("--cellpose-x-transform", nargs=3, type=float)
    parser.add_argument("--cellpose-y-transform", nargs=3, type=float)
    for flag in (
        "--exclude-spatialdata-transcripts",
        "--overwrite",
        "--use-cell-initialization",
    ):
        parser.add_argument(flag, action="store_true")
    return parser


def _outlines(
    prior: np.ndarray, labels: np.ndarray, x_t: list[float], y_t: list[float]
) -> dict[int, dict]:
    from rasterio import features
    from shapely.geometry import mapping, shape
    from shapely.ops import unary_union

    found: dict[int, list] = {int(label): [] for label in labels}
    for geometry, value in features.shapes(prior.astype(np.int32), mask=prior > 0):
        found[int(value)].append(shape(geometry))
    result = {}
    for label, parts in found.items():
        merged = unary_union(parts)

        def transform(coords: np.ndarray) -> np.ndarray:
            cols, rows = coords[:, 0], coords[:, 1]
            return np.column_stack(
                [
                    x_t[0] * cols + x_t[1] * rows + x_t[2],
                    y_t[0] * cols + y_t[1] * rows + y_t[2],
                ]
            )

        from shapely import transform as shp_transform

        result[label] = mapping(shp_transform(merged, transform))
    return result


def main() -> int:
    args = _parser().parse_args()
    if args.version:
        print(f"proseg {os.environ.get('FAKE_PROSEG_VERSION', '3.2.0')}")
        return 0
    path, layout = args.inputs
    assert layout == "xenium", layout
    table = pq.read_table(path)
    schema = {field.name: field.type for field in table.schema}
    assert schema == EXPECTED, schema
    assert set(table.column("fov_name").to_pylist()) == {"0"}
    _log("loaded transcripts: 0.01s")
    prior = np.load(args.cellpose_masks)
    labels = np.unique(prior)
    labels = labels[labels != 0]
    cell_of_label = {int(label): index for index, label in enumerate(labels)}
    x_t, y_t = args.cellpose_x_transform, args.cellpose_y_transform
    _log("initialized voxels: 0.01s")
    n_rows = table.num_rows
    n_cells = len(labels)
    weight = float(args.prior_weight) * max(n_rows / max(n_cells, 1) / 100.0, 1.0)
    weight *= float(os.environ.get("FAKE_PROSEG_WEIGHT_FACTOR", "1"))
    genes = sorted(set(table.column("feature_name").to_pylist()))
    print("Using 1 threads\nFinished reading input\nRead dataset:")
    print(f"{n_rows:>9} transcripts\n{n_cells:>9} cells\n{len(genes):>9} genes")
    print(f"{1:>9} fovs")
    if weight > 1.0:
        print(f"{weight:>9.2f} prior weight", flush=True)
    iterations = int(args.burnin_samples) + int(args.samples) + int(args.hillclimb)
    for _ in range(min(iterations, 5)):
        _log("sampling times: morphology: 1.0ms, repo: 2.0ms, params: 3.0ms")

    cell_id = np.asarray(table.column("cell_id").to_pylist(), dtype=object)
    x = table.column("x_location").to_numpy()
    y = table.column("y_location").to_numpy()
    unassigned = np.iinfo(np.uint32).max
    assignment = np.full(n_rows, unassigned, dtype=np.uint32)
    for row, value in enumerate(cell_id):
        if value != "0" and int(value) in cell_of_label:
            assignment[row] = cell_of_label[int(value)]
    if os.environ.get("FAKE_PROSEG_MODE", "identity") == "greedy_halo" and n_cells:
        reach = float(os.environ.get("FAKE_PROSEG_GREEDY_UM", "2"))
        rows_px, cols_px = np.nonzero(prior)
        px_x = x_t[0] * (cols_px + 0.5) + x_t[2]
        px_y = y_t[1] * (rows_px + 0.5) + y_t[2]
        px_label = prior[rows_px, cols_px]
        for row in np.nonzero(assignment == unassigned)[0]:
            distance = np.hypot(px_x - x[row], px_y - y[row])
            nearest = int(np.argmin(distance))
            if distance[nearest] <= reach:
                assignment[row] = cell_of_label[int(px_label[nearest])]
    is_assigned = assignment != unassigned
    background = ~is_assigned

    out = Path(args.output_path)
    out.mkdir(parents=True, exist_ok=True)
    centroids = np.zeros((n_cells, 2), dtype=np.float32)
    for index, label in enumerate(labels):
        rr, cc = np.nonzero(prior == label)
        centroids[index] = (
            x_t[0] * (cc.mean() + 0.5) + x_t[2],
            y_t[1] * (rr.mean() + 0.5) + y_t[2],
        )
    pq.write_table(
        pa.table(
            {
                "cell": pa.array(np.arange(n_cells, dtype=np.uint32)),
                "original_cell_id": pa.array([str(int(v)) for v in labels]),
                "centroid_x": pa.array(centroids[:, 0]),
                "centroid_y": pa.array(centroids[:, 1]),
                "centroid_z": pa.array(np.zeros(n_cells, np.float32)),
                "cluster": pa.array(np.zeros(n_cells, np.uint16)),
                "volume": pa.array(np.full(n_cells, 10.0, np.float32)),
                "surface_area": pa.array(np.full(n_cells, 20, np.uint32)),
                "scale": pa.array(np.ones(n_cells, np.float32)),
            }
        ),
        out / args.output_cell_metadata,
    )
    gene_index = {name: i for i, name in enumerate(genes)}
    gene_codes = np.asarray(
        [gene_index[name] for name in table.column("feature_name").to_pylist()],
        dtype=np.int32,
    )
    pq.write_table(
        pa.table(
            {
                "transcript_id": table.column("transcript_id"),
                "qv": table.column("qv"),
                "x": table.column("x_location"),
                "y": table.column("y_location"),
                "z": table.column("z_location"),
                "observed_x": table.column("x_location"),
                "observed_y": table.column("y_location"),
                "observed_z": table.column("z_location"),
                "gene": pa.DictionaryArray.from_arrays(
                    pa.array(gene_codes), pa.array(genes)
                ),
                "assignment": pa.array(
                    np.where(is_assigned, assignment, 0),
                    pa.uint32(),
                    mask=~is_assigned,
                ),
                "background": pa.array(background),
            }
        ),
        out / args.output_transcript_metadata,
    )
    outlines = _outlines(prior, labels, x_t, y_t)
    features_json = [
        {
            "type": "Feature",
            "properties": {"cell": index},
            "geometry": outlines[int(label)],
        }
        for index, label in enumerate(labels)
    ]
    with gzip.open(out / args.output_cell_polygons, "wt", encoding="utf-8") as handle:
        json.dump({"type": "FeatureCollection", "features": features_json}, handle)

    import anndata as ad
    import pandas as pd
    from scipy import sparse

    counted = is_assigned & ~background
    matrix = sparse.coo_matrix(
        (
            np.ones(int(counted.sum()), np.uint32),
            (assignment[counted].astype(np.int64), gene_codes[counted]),
        ),
        shape=(n_cells, len(genes)),
    ).tocsr()
    if os.environ.get("FAKE_PROSEG_CORRUPT_X") == "1" and n_cells:
        matrix = matrix.tolil()
        matrix[0, 0] = matrix[0, 0] + 1
        matrix = matrix.tocsr()
    obs = pd.DataFrame(
        {
            "cell": np.arange(n_cells, dtype=np.uint32),
            "original_cell_id": [str(int(v)) for v in labels],
        },
        index=[str(i) for i in range(n_cells)],
    )
    var = pd.DataFrame({"gene": genes}, index=genes)
    ad.AnnData(X=matrix, obs=obs, var=var).write_zarr(
        out / args.output_spatialdata / "tables" / "table"
    )
    _log("write SpatialData: 0.01s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

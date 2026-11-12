"""Run the pinned pciSeq API without importing MerXen's spatial I/O stack."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import uuid
from collections.abc import Iterable, Iterator
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from merxen.pciseq_records import write_json_frames

UPSTREAM_COMMIT = "5e2c0fd1841f8a4022a2fa341ece965687e343dd"
_CELL_COLUMNS = {"Cell_Num", "ClassName", "Prob"}
_SPOT_COLUMNS = {"spot_id", "gene_name", "neighbour"}
logger = logging.getLogger(__name__)


def _load_prepared(
    prepared_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, list[sparse.coo_matrix]]:
    manifest = json.loads((prepared_dir / "manifest.json").read_text())
    if not isinstance(manifest, dict):
        raise ValueError("The prepared pciSeq manifest must be a JSON object")
    spots = pd.read_csv(prepared_dir / "spots.csv", dtype={"gene_name": str})
    reference = pd.read_csv(prepared_dir / "reference.csv", index_col=0)
    reference.index = reference.index.astype(str)
    reference.columns = reference.columns.astype(str)
    paths = sorted(
        (prepared_dir / "masks").glob("plane_*.npz"),
        key=lambda path: int(path.stem.removeprefix("plane_")),
    )
    if not paths:
        raise ValueError("Prepared pciSeq input contains no segmentation planes")
    if [int(path.stem.removeprefix("plane_")) for path in paths] != list(
        range(len(paths))
    ):
        raise ValueError("Prepared segmentation planes must start at zero with no gaps")
    planes = [sparse.load_npz(path).tocoo() for path in paths]
    return manifest, spots, reference, planes


def _validate_prepared(
    manifest: dict[str, Any],
    spots: pd.DataFrame,
    reference: pd.DataFrame,
    planes: list[sparse.coo_matrix],
) -> tuple[int, dict[str, Any]]:
    if not {"gene_name", "x", "y"}.issubset(spots.columns) or spots.empty:
        raise ValueError("Prepared spots must contain gene_name, x, y and rows")
    if reference.empty or not reference.index.is_unique:
        raise ValueError("The reference must contain rows with unique gene names")
    if not reference.columns.is_unique or "Zero" in reference.columns:
        raise ValueError("Reference classes must be unique and cannot include Zero")
    expression = reference.to_numpy(dtype=float)
    if not np.isfinite(expression).all() or (expression < 0).any():
        raise ValueError("Reference expression must be finite and nonnegative")
    if (expression.sum(axis=0) <= 0).any():
        raise ValueError("Each reference cell type must have positive expression")
    if (
        spots["gene_name"].isna().any()
        or not spots["gene_name"].isin(reference.index).all()
    ):
        raise ValueError("All prepared spot genes must occur in the reference")

    if len(planes) > 1 and "z_plane" not in spots:
        raise ValueError("3D prepared spots require z_plane indices")
    if "z_plane" not in spots:
        spots["z_plane"] = 0.0
    coordinates = spots[["x", "y", "z_plane"]].to_numpy(dtype=float)
    shape = planes[0].shape
    if any(plane.shape != shape for plane in planes):
        raise ValueError("All segmentation planes must have the same shape")
    maxima = np.asarray([shape[1] - 1, shape[0] - 1, len(planes) - 1])
    if (
        not np.isfinite(coordinates).all()
        or (coordinates < 0).any()
        or (coordinates > maxima).any()
    ):
        raise ValueError("Prepared spot coordinates must be finite and in bounds")

    present: set[int] = set()
    for plane in planes:
        values = plane.data
        if (
            not np.isfinite(values).all()
            or (values < 0).any()
            or (values != np.floor(values)).any()
            or (values > np.iinfo(np.uint32).max).any()
        ):
            raise ValueError("Mask labels must be nonnegative uint32 integers")
        present.update(int(value) for value in np.unique(values) if value > 0)
    count = manifest.get("cell_count")
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count < 1
        or present != set(range(1, count + 1))
    ):
        raise ValueError("Mask labels must match manifest cell_count and be dense")
    voxel_size = manifest.get("voxel_size")
    if (
        not isinstance(voxel_size, list)
        or len(voxel_size) != 3
        or not np.isfinite(np.asarray(voxel_size, dtype=float)).all()
        or (np.asarray(voxel_size, dtype=float) <= 0).any()
    ):
        raise ValueError("voxel_size must contain three positive finite sizes")
    # Upstream looks up masks after scaling Y into X units, so only square XY
    # pixels preserve raster containment. Z anisotropy is handled independently.
    if not np.isclose(voxel_size[0], voxel_size[1], rtol=1e-9, atol=0):
        raise ValueError("pciSeq requires equal X/Y voxel sizes; resample the volume")
    options = manifest.get("options", {})
    if not isinstance(options, dict):
        raise ValueError("Prepared pciSeq options must be a JSON object")
    if spots["gene_name"].isin(options.get("exclude_genes", [])).any():
        raise ValueError("Excluded genes must be removed during preparation")
    neighbors = options.get("nNeighbors", 9)
    if isinstance(neighbors, bool) or not isinstance(neighbors, int) or neighbors < 1:
        raise ValueError("nNeighbors must be a positive integer")
    max_iter = options.get("max_iter", 1000)
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    return count, dict(options)


def _validate_results(
    cells: pd.DataFrame,
    genes: pd.DataFrame,
    *,
    cell_count: int,
    spots: pd.DataFrame,
    class_names: pd.Index,
) -> None:
    if not isinstance(cells, pd.DataFrame) or not _CELL_COLUMNS.issubset(cells):
        raise ValueError("pciSeq returned malformed cellData columns")
    if not isinstance(genes, pd.DataFrame) or not _SPOT_COLUMNS.issubset(genes):
        raise ValueError("pciSeq returned malformed geneData columns")
    _validate_cells(cells, cell_count=cell_count, class_names=class_names)
    if genes["spot_id"].duplicated().any() or set(genes["spot_id"]) != set(
        range(len(spots))
    ):
        raise ValueError("pciSeq changed the prepared transcript identifiers")
    if not genes["neighbour"].isin(range(cell_count + 1)).all():
        raise ValueError("pciSeq returned unknown parent-cell identifiers")
    by_id = genes.set_index("spot_id")["gene_name"].reindex(spots.index)
    if not np.array_equal(by_id.to_numpy(), spots["gene_name"].to_numpy()):
        raise ValueError("pciSeq changed the prepared transcript gene mapping")


def _validate_cells(
    cells: pd.DataFrame, *, cell_count: int, class_names: pd.Index
) -> None:
    """Validate cell records independently of the number of transcript records."""
    if not isinstance(cells, pd.DataFrame) or not _CELL_COLUMNS.issubset(cells):
        raise ValueError("pciSeq returned malformed cellData columns")
    if cells["Cell_Num"].duplicated().any() or set(cells["Cell_Num"]) != set(
        range(1, cell_count + 1)
    ):
        raise ValueError("pciSeq changed the prepared segmentation cell identifiers")
    allowed_classes = set(class_names) | {"Zero"}
    for names, probabilities in zip(cells["ClassName"], cells["Prob"], strict=True):
        if (
            not isinstance(names, (list, np.ndarray))
            or not isinstance(probabilities, (list, np.ndarray))
            or len(names) != len(probabilities)
            or not set(names).issubset(allowed_classes)
        ):
            raise ValueError("pciSeq returned invalid class probability summaries")
        values = np.asarray(probabilities, dtype=float)
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise ValueError("pciSeq returned invalid class probability values")


def _validated_spot_frames(
    frames: Iterable[pd.DataFrame], *, spots: pd.DataFrame, cell_count: int
) -> Iterator[pd.DataFrame]:
    """Check every streamed transcript against the dense prepared row contract."""
    position = 0
    for frame in frames:
        if not isinstance(frame, pd.DataFrame) or not _SPOT_COLUMNS.issubset(frame):
            raise ValueError("pciSeq returned malformed geneData columns")
        stop = position + len(frame)
        if stop > len(spots) or not np.array_equal(
            frame.spot_id.to_numpy(), np.arange(position, stop)
        ):
            raise ValueError("pciSeq changed the prepared transcript identifiers")
        if not frame.neighbour.isin(range(cell_count + 1)).all():
            raise ValueError("pciSeq returned unknown parent-cell identifiers")
        if not np.array_equal(
            frame.gene_name.to_numpy(), spots.gene_name.iloc[position:stop].to_numpy()
        ):
            raise ValueError("pciSeq changed the prepared transcript gene mapping")
        position = stop
        yield frame
    if position != len(spots):
        raise ValueError("pciSeq changed the prepared transcript identifiers")


def _encode_class_names(
    reference: pd.DataFrame, options: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, str], list[str]]:
    # Upstream natural-sorts labels, then asserts lexical order. Equal-width
    # tokens agree under both orders while preserving every expression column.
    width = max(6, len(str(len(reference.columns) - 1)))
    token_map = {
        f"merxen_class_{index:0{width}d}": name
        for index, name in enumerate(reference.columns)
    }
    encode = {name: token for token, name in token_map.items()}
    encode["Zero"] = "Zero"
    encoded = reference.copy()
    encoded.columns = list(token_map)
    upstream_options = deepcopy(options)
    ignored_weights: list[str] = []
    weights = upstream_options.get("cell_type_weights")
    if isinstance(weights, dict):
        mapped_weights = {}
        for name, weight in weights.items():
            if name not in encode:
                ignored_weights.append(str(name))
                logger.warning("Ignoring cell_type_weights for unknown class %r", name)
            else:
                mapped_weights[encode[name]] = weight
        upstream_options["cell_type_weights"] = mapped_weights
    groups = upstream_options.get("mrf_pooled_classes")
    if isinstance(groups, list):
        mapped_groups = []
        for group in groups:
            if not isinstance(group, list) or any(
                not isinstance(name, str) or name not in encode for name in group
            ):
                raise ValueError("mrf_pooled_classes contains unknown cell classes")
            mapped_groups.append([encode[name] for name in group])
        upstream_options["mrf_pooled_classes"] = mapped_groups
    return encoded, upstream_options, token_map, ignored_weights


def _restore_class_names(
    cells: pd.DataFrame, token_map: dict[str, str]
) -> pd.DataFrame:
    if not isinstance(cells, pd.DataFrame) or "ClassName" not in cells:
        raise ValueError("pciSeq returned malformed cellData columns")
    decode = {**token_map, "Zero": "Zero"}
    restored = []
    for names in cells["ClassName"]:
        if not isinstance(names, (list, np.ndarray)) or any(
            not isinstance(name, str) or name not in decode for name in names
        ):
            raise ValueError("pciSeq returned unknown internal class tokens")
        restored.append([decode[name] for name in names])
    cells = cells.copy()
    cells["ClassName"] = restored
    return cells


def run_pciseq_compute(
    prepared_dir: Path,
    output_dir: Path,
    *,
    memory_mode: str = "standard",
    cell_batch_size: int = 128,
    spot_batch_size: int = 20_000,
    scratch_dir: Path | None = None,
) -> Path:
    """Fit pciSeq to validated prepared input and write its reported results.

    Args:
        prepared_dir: Directory produced by MerXen's pciSeq preparation stage.
        output_dir: Destination for raw cell/spot records and run metadata.

    Returns:
        Path to the computation manifest.

    Raises:
        ValueError: Prepared input or returned identifier contracts are invalid.
        ImportError: The isolated pciSeq runtime has not been installed.
    """
    if memory_mode not in {"standard", "batched"}:
        raise ValueError("pciSeq memory_mode must be standard or batched")
    if memory_mode == "batched" and min(cell_batch_size, spot_batch_size) < 1:
        raise ValueError("pciSeq batch sizes must be positive")
    manifest, spots, reference, planes = _load_prepared(prepared_dir)
    count, options = _validate_prepared(manifest, spots, reference, planes)
    requested_options = deepcopy(options)
    output_dir.mkdir(parents=True, exist_ok=True)
    options.update(
        {
            "save_data": False,
            "realtime_viewer": False,
            "remove_flat_cells": False,
            "output_path": str(output_dir.resolve()),
            "voxel_size": manifest["voxel_size"],
            "nNeighbors": min(options.get("nNeighbors", 9), count),
        }
    )
    try:
        pciseq = importlib.import_module("pciSeq")
    except ImportError as error:
        raise ImportError(
            "pciSeq is unavailable; use the isolated pciseq-runtime environment"
        ) from error
    encoded_reference, upstream_options, token_map, ignored_weights = (
        _encode_class_names(reference, options)
    )
    execution: dict[str, Any] = {"memory_mode": memory_mode}
    if memory_mode == "batched":
        from merxen.pciseq_batched import fit_batched

        with fit_batched(
            spots=spots.copy(),
            planes=[plane.copy() for plane in planes],
            reference=encoded_reference,
            options=deepcopy(upstream_options),
            expected_commit=UPSTREAM_COMMIT,
            scratch_dir=scratch_dir
            or output_dir.parent / f".pciseq-batched-{uuid.uuid4().hex}",
            cell_batch_size=cell_batch_size,
            spot_batch_size=spot_batch_size,
        ) as (cells, gene_frames, execution):
            cells = _restore_class_names(cells, token_map)
            _validate_cells(cells, cell_count=count, class_names=reference.columns)
            cells.to_json(
                output_dir / "cell_data.json", orient="records", double_precision=15
            )
            write_json_frames(
                output_dir / "gene_data.json",
                _validated_spot_frames(gene_frames, spots=spots, cell_count=count),
            )
    else:
        # fit mutates its inputs. Only task-local copies cross this boundary.
        cells, genes = pciseq.fit(
            spots=spots.copy(),
            coo=[plane.copy() for plane in planes],
            scRNAseq=encoded_reference,
            opts=deepcopy(upstream_options),
        )
        cells = _restore_class_names(cells, token_map)
        _validate_results(
            cells, genes, cell_count=count, spots=spots, class_names=reference.columns
        )
        cells.to_json(
            output_dir / "cell_data.json", orient="records", double_precision=15
        )
        genes.to_json(
            output_dir / "gene_data.json", orient="records", double_precision=15
        )
    metadata = {
        "prepared_manifest_sha256": hashlib.sha256(
            (prepared_dir / "manifest.json").read_bytes()
        ).hexdigest(),
        "upstream_repository": "https://github.com/acycliq/pciSeq_3d",
        "expected_upstream_commit": UPSTREAM_COMMIT,
        "upstream_version": str(getattr(pciseq, "__version__", "unknown")),
        "upstream_commit": str(getattr(pciseq, "__commit__", "unknown")),
        "geometry_mode": manifest.get("geometry_mode"),
        "cell_count": count,
        "spot_count": len(spots),
        "requested_options": requested_options,
        "resolved_options": options,
        "upstream_options": upstream_options,
        "class_token_map": token_map,
        "ignored_cell_type_weight_names": ignored_weights,
        "has_converged": None,
        "convergence_status": "not_exposed_by_fit",
        "probability_semantics": (
            "Upstream reported summaries: classes with probability >0.001 retained; "
            "probabilities truncated to three decimals. Missing entries are not "
            "measured zeros. fit returns normally when max_iter is exhausted."
        ),
    }
    if memory_mode == "batched":
        # Only the experimental mode records its execution; standard manifests
        # keep main's keys, so their published fingerprints do not change.
        metadata["has_converged"] = execution.get("has_converged")
        metadata["convergence_status"] = "exposed_by_pinned_model"
        metadata["execution"] = execution
    path = output_dir / "compute_manifest.json"
    path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return path


def main() -> None:
    """Run the isolated pciSeq computation from prepared files."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--memory-mode", choices=("standard", "batched"), default="standard"
    )
    parser.add_argument("--cell-batch-size", type=int, default=128)
    parser.add_argument("--spot-batch-size", type=int, default=20_000)
    parser.add_argument("--scratch-dir", type=Path)
    arguments = parser.parse_args()
    run_pciseq_compute(
        arguments.prepared_dir,
        arguments.output_dir,
        memory_mode=arguments.memory_mode,
        cell_batch_size=arguments.cell_batch_size,
        spot_batch_size=arguments.spot_batch_size,
        scratch_dir=arguments.scratch_dir,
    )


if __name__ == "__main__":
    main()

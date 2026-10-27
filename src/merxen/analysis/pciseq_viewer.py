"""Publish exact-ID pciSeq assignments as additive SpatialData viewer sources."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import dask
import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_float_dtype, is_integer_dtype

from merxen.config import PciSeqConfig
from merxen.io.spatialdata_schema import MERXEN_SCHEMA_ATTR

_BRANCHES = {
    "reseg": "proseg",
    "original_seg": "original",
    "proseg_mask": "cellpose",
    "proseg_hybrid": "proseg_hybrid",
}
_LABELS = {
    "proseg": "ProSeg",
    "original": "Original",
    "cellpose": "Cellpose",
    "proseg_hybrid": "ProSeg hybrid",
}
_STATUSES = ["assigned", "background", "not_modelled"]
_MARKER = "merxen_pciseq_viewer"
_CHUNK_ROWS = 500_000


@dataclass(frozen=True)
class _SpotIndex:
    transcript_ids: np.ndarray
    assignments: np.ndarray
    gene_codes: np.ndarray
    genes: tuple[str, ...]


def _ids(
    values: Any, field: str, *, nullable: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Parse unsigned IDs without routing nullable integers through floats."""
    series = pd.Series(values, copy=False)
    missing = series.isna().to_numpy()
    if missing.any() and not nullable:
        raise ValueError(f"{field} contains missing identifiers")
    present = series[~missing]
    if is_bool_dtype(series.dtype):
        raise ValueError(f"{field} contains boolean identifiers")
    result = np.zeros(len(series), dtype=np.uint64)
    if is_integer_dtype(series.dtype):
        if (present <= 0).any():
            raise ValueError(f"{field} requires positive identifiers")
        result[~missing] = present.to_numpy(dtype=np.uint64)
    elif is_float_dtype(series.dtype):
        numbers = present.to_numpy(dtype=float)
        if (
            not np.isfinite(numbers).all()
            or (numbers <= 0).any()
            or (numbers > 2**53).any()
            or (numbers != np.floor(numbers)).any()
        ):
            raise ValueError(f"{field} contains unsafe or noninteger identifiers")
        result[~missing] = numbers.astype(np.uint64)
    else:
        strings = present.astype(str)
        if not strings.str.fullmatch(r"[0-9]+").all():
            raise ValueError(f"{field} contains noninteger identifiers")
        try:
            result[~missing] = np.fromiter(
                (int(value) for value in strings), dtype=np.uint64, count=len(strings)
            )
        except (OverflowError, ValueError) as error:
            raise ValueError(f"{field} exceeds the uint64 identifier range") from error
        if (result[~missing] == 0).any():
            raise ValueError(f"{field} requires positive identifiers")
    return result, missing


def _unique(values: np.ndarray, field: str) -> np.ndarray:
    ordered = np.sort(values)
    if len(ordered) > 1 and np.any(ordered[1:] == ordered[:-1]):
        raise ValueError(f"{field} contains duplicate identifiers")
    return ordered


def _matches(
    sorted_ids: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.searchsorted(sorted_ids, values)
    matched = positions < len(sorted_ids)
    matched[matched] &= sorted_ids[positions[matched]] == values[matched]
    return positions, matched


def _spot_index(path: Path, frame: pd.DataFrame | None) -> _SpotIndex:
    columns = ["transcript_id", "instance_id", "gene_name"]
    if frame is None:
        chunks = pd.read_csv(
            path,
            usecols=columns,
            dtype={column: "string" for column in columns},
            keep_default_na=False,
            na_values={"instance_id": [""]},
            chunksize=_CHUNK_ROWS,
        )
    else:
        if not set(columns).issubset(frame):
            raise ValueError("Spot assignments lack exact transcript/cell IDs or genes")
        chunks = (
            frame.iloc[start : start + _CHUNK_ROWS][columns]
            for start in range(0, len(frame), _CHUNK_ROWS)
        )
    ids, assignments, gene_codes = [], [], []
    genes: dict[str, int] = {}
    for chunk in chunks:
        transcript_ids, _ = _ids(chunk.transcript_id, "spot assignment transcript_id")
        cell_ids, _ = _ids(
            chunk.instance_id, "spot assignment instance_id", nullable=True
        )
        if chunk.gene_name.isna().any() or chunk.gene_name.eq("").any():
            raise ValueError("Modeled transcripts require gene names")
        names = chunk.gene_name.astype(str)
        for name in names.unique():
            if name not in genes:
                genes[name] = len(genes)
        ids.append(transcript_ids)
        assignments.append(cell_ids)
        gene_codes.append(names.map(genes).to_numpy(dtype=np.uint32))
    if not ids:
        raise ValueError("pciSeq viewer publication requires modeled transcripts")
    transcript_ids = np.concatenate(ids)
    cell_ids = np.concatenate(assignments)
    codes = np.concatenate(gene_codes)
    order = np.argsort(transcript_ids)
    transcript_ids = transcript_ids[order]
    if len(transcript_ids) > 1 and np.any(transcript_ids[1:] == transcript_ids[:-1]):
        raise ValueError("Spot assignments contain duplicate transcript identifiers")
    return _SpotIndex(transcript_ids, cell_ids[order], codes[order], tuple(genes))


def _join_partition(
    part: pd.DataFrame, *, index: _SpotIndex, gene_column: str
) -> pd.DataFrame:
    result = part.copy()
    native_ids, _ = _ids(result.transcript_id, "native transcript_id")
    positions, matched = _matches(index.transcript_ids, native_ids)
    selected = positions[matched]
    names = np.asarray(index.genes, dtype=object)[index.gene_codes[selected]]
    if not np.array_equal(
        result.loc[matched, gene_column].astype(str).to_numpy(), names
    ):
        raise ValueError("pciSeq transcript IDs resolve to different native genes")
    values = np.zeros(len(result), dtype=np.uint64)
    values[matched] = index.assignments[selected]
    background = matched & (values == 0)
    result["transcript_id"] = native_ids
    result["assignment"] = pd.arrays.IntegerArray(values, values == 0)
    result["background"] = background
    status = np.full(len(result), "not_modelled", dtype=object)
    status[matched] = "assigned"
    status[background] = "background"
    result["pciseq_assignment_status"] = pd.Categorical(status, categories=_STATUSES)
    return result


def _cells_table(
    ids: np.ndarray, shape_key: str, cells: pd.DataFrame, metadata: dict[str, Any]
) -> ad.AnnData:
    from spatialdata.models import TableModel

    cell_ids, _ = _ids(cells.instance_id, "cell assignment instance_id")
    _unique(cell_ids, "cell assignments")
    _, known = _matches(_unique(ids, "native shapes"), cell_ids)
    if not known.all():
        raise ValueError("Cell assignments reference unknown native cells")
    columns = [
        "pciseq_cell_type",
        "pciseq_probability",
        "pciseq_status",
        "pciseq_exclusion_reason",
        "reported_class_names",
        "reported_probabilities",
    ]
    if not {"pciseq_cell_type", "pciseq_probability", "pciseq_status"}.issubset(cells):
        raise ValueError("Cell assignments lack pciSeq annotations")
    indexed = cells.copy()
    indexed.index = pd.Index(cell_ids, dtype="uint64")
    obs = indexed.reindex(pd.Index(ids, dtype="uint64"))
    obs = obs[[column for column in columns if column in obs]].copy()
    obs["pciseq_status"] = obs.pciseq_status.astype("string").fillna("not_modelled")
    obs["pciseq_exclusion_reason"] = obs.get(
        "pciseq_exclusion_reason", pd.Series("", index=obs.index)
    ).fillna("")
    obs["pciseq_probability"] = pd.to_numeric(obs.pciseq_probability, errors="raise")
    valid = obs.pciseq_probability.dropna().to_numpy(dtype=float)
    if not np.isfinite(valid).all() or (valid < 0).any() or (valid > 1).any():
        raise ValueError("Cell assignment probabilities must be within [0, 1]")
    classes = [*metadata["reference_classes"], "Zero"]
    if not obs.pciseq_cell_type.dropna().isin([*classes, "Uncertain"]).all():
        raise ValueError("Cell assignments contain unknown reference classes")
    excluded = obs.pciseq_status.eq("excluded_overlap")
    if (
        obs.loc[excluded, "pciseq_cell_type"].notna().any()
        or obs.loc[excluded, "pciseq_probability"].notna().any()
    ):
        raise ValueError("Excluded cells cannot carry a cell type or probability")
    for column in ["pciseq_cell_type", "pciseq_status", "pciseq_exclusion_reason"]:
        obs[column] = pd.Categorical(obs[column])
    for column in ["reported_class_names", "reported_probabilities"]:
        if column in obs:
            obs[column] = obs[column].fillna("[]").astype(str)
    obs["instance_id"] = ids
    obs["region"] = pd.Categorical([shape_key] * len(obs))
    obs.index = pd.Index([f"cell_{value}" for value in ids])
    table = ad.AnnData(obs=obs)
    table.uns["merxen_pciseq"] = {"reported_probability_classes": classes}
    return TableModel.parse(
        table, region=shape_key, region_key="region", instance_key="instance_id"
    )


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _source_fingerprint(path: Path, points_key: str) -> str:
    """Fingerprint native element files without reading full coordinate payloads."""
    root = path / "points" / points_key
    files: list[tuple[Any, ...]] = []
    for item in sorted(root.rglob("*")):
        if not item.is_file():
            continue
        info = item.stat()
        if item.suffix == ".parquet":
            fingerprint: Any = (info.st_size, info.st_mtime_ns)
        elif item.name in {"zarr.json", ".zattrs", ".zgroup"}:
            fingerprint = json.loads(item.read_text())
        else:
            fingerprint = (info.st_size, info.st_mtime_ns)
        files.append((str(item.relative_to(root)), fingerprint))
    return hashlib.sha256(json.dumps(files).encode()).hexdigest()


def _write_elements(
    sdata: Any, elements: list[tuple[str, str, Any]], schema: dict[str, Any]
) -> None:
    """Stage independent elements and roll back a failed multi-element refresh."""
    from merxen.io.spatialdata_io import (
        write_or_replace_element,
        write_spatialdata_metadata,
    )

    path = Path(sdata.path)
    token = uuid.uuid4().hex
    root_attrs = deepcopy(sdata.attrs)
    originals = {
        (group, key): getattr(sdata, group).get(key) for group, key, _ in elements
    }
    staged: list[tuple[str, str, str, Any]] = []
    backups: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        # Synchronous partition writes bound RAM independently of host CPU count.
        with dask.config.set(scheduler="synchronous"):
            for group, key, value in elements:
                temporary = f"_pciseq_stage_{token}_{key}"
                staged.append((group, key, temporary, value))
                write_or_replace_element(
                    sdata, temporary, group, value, overwrite=False
                )
                if group == "points":
                    import zarr

                    zarr.open_group(str(path / group / temporary), mode="a").attrs[
                        _MARKER
                    ] = value.attrs[_MARKER]
        for group, key, temporary, value in staged:
            target = path / group / key
            if target.exists():
                backup = target.with_name(f".{key}.pciseq-backup-{token}")
                os.replace(target, backup)
                backups.append((target, backup))
            os.replace(path / group / temporary, target)
            published.append(target)
            container = getattr(sdata, group)
            del container[temporary]
            if key in container:
                del container[key]
            container[key] = value
        sdata.attrs[MERXEN_SCHEMA_ATTR] = schema
        # Commit registry only after every referenced element has been written.
        write_spatialdata_metadata(sdata)
    except Exception:
        for target in reversed(published):
            shutil.rmtree(target)
        for target, backup in reversed(backups):
            os.replace(backup, target)
        for group, key, temporary, _ in staged:
            shutil.rmtree(path / group / temporary, ignore_errors=True)
            container = getattr(sdata, group)
            if temporary in container:
                del container[temporary]
            if key in container:
                del container[key]
            original = originals[(group, key)]
            if original is not None:
                container[key] = original
        sdata.attrs.clear()
        sdata.attrs.update(root_attrs)
        write_spatialdata_metadata(sdata)
        raise
    for _, backup in backups:
        shutil.rmtree(backup)
    # Remove backup groups from consolidated discovery metadata after cleanup.
    sdata.write_consolidated_metadata()


def publish_pciseq_viewer(
    config: PciSeqConfig,
    output_dir: Path,
    *,
    cell_assignments: pd.DataFrame | None = None,
    spot_assignments: pd.DataFrame | None = None,
    annotated: ad.AnnData | None = None,
) -> dict[str, Any]:
    """Publish a viewer source from finished exports without refitting pciSeq.

    Native measured points are left joined by exact uint64 transcript IDs. The
    derived points retain native transformations and distinguish modeled
    background from transcripts absent from the model. Every native shape gets
    an annotation row independently of expression QC. Store reads, staging,
    replacement and the final registry commit share the existing writer lock.

    Args:
        config: Settings identifying the native store and segmentation.
        output_dir: Finished export directory containing manifests and CSVs.
        cell_assignments: Optional existing finalization frame, avoiding a CSV read.
        spot_assignments: Optional existing finalization frame, avoiding a CSV read.
        annotated: Optional existing QC-filtered finalization table.

    Returns:
        Source registry metadata, coverage counts and artifact fingerprints; also
        saved as ``viewer_publication.json`` beside the finished exports.
    """
    import spatialdata as sd
    import zarr
    from spatialdata.models import PointsModel, TableModel
    from spatialdata.transformations import get_transformation

    from merxen.analysis.pciseq import _resolve_native_keys
    from merxen.io.spatialdata_io import spatialdata_write_lock
    from merxen.io.spatialdata_schema import canonical_shape_instance_series
    from merxen.io.transcript_io import resolve_col

    output_dir = Path(output_dir)
    metadata = json.loads((output_dir / "manifest.json").read_text())
    compute = json.loads((output_dir / "compute_manifest.json").read_text())
    for field in ["pair_id", "sample_id", "platform", "segmentation"]:
        if metadata.get(field) != getattr(config, field):
            raise ValueError(f"Exported pciSeq {field} differs from publication config")
    artifacts = {
        name: _sha(output_dir / name)
        for name in [
            "manifest.json",
            "compute_manifest.json",
            "cell_assignments.csv",
            "spot_assignments.csv",
            "annotated.h5ad",
        ]
    }
    expected = compute.get("prepared_manifest_sha256")
    if expected is not None and expected != artifacts["manifest.json"]:
        raise ValueError("Exported compute and prepared manifest fingerprints differ")
    branch = _BRANCHES[config.segmentation]
    source_key = f"pciseq_{branch}"
    points_key = f"transcripts_pciseq_{branch}"
    with spatialdata_write_lock(config.spatialdata_path):
        sdata = sd.read_zarr(config.spatialdata_path)
        shape_key, native_key, native_table = _resolve_native_keys(sdata, config)
        if shape_key != metadata.get("shape_key") or native_key != metadata.get(
            "points_key"
        ):
            raise ValueError(
                "Exported pciSeq inputs differ from the selected native elements"
            )
        source_table = str(metadata.get("source_table_key") or native_table or "")
        if not source_table:
            raise ValueError(
                "pciSeq viewer publication requires a native source table key"
            )
        table_key, embedding_key = (
            f"{source_table}_pciseq_cells",
            f"{source_table}_pciseq",
        )
        native = sdata.points[native_key]
        if "transcript_id" not in native.columns:
            raise ValueError("Native points require stable transcript_id identifiers")
        shape_ids = canonical_shape_instance_series(sdata.shapes[shape_key]).to_numpy(
            dtype=np.uint64
        )
        _unique(shape_ids, "native shape instance_id")
        source_fingerprint = {
            "points_files_stat_sha256": _source_fingerprint(
                Path(config.spatialdata_path).resolve(), native_key
            ),
            "cell_instance_ids_sha256": hashlib.sha256(shape_ids.tobytes()).hexdigest(),
        }
        provenance = {
            "version": 1,
            "artifacts_sha256": artifacts,
            "native": source_fingerprint,
            "shape": shape_key,
            "source_points": native_key,
            "source_table": source_table,
        }
        analysis_id = hashlib.sha256(
            json.dumps(provenance, sort_keys=True).encode()
        ).hexdigest()
        schema = deepcopy(sdata.attrs.get(MERXEN_SCHEMA_ATTR, {}))
        sources = schema.setdefault("analysis_sources", {})
        previous = sources.get(source_key, {})
        point_marker: dict[str, Any] = {}
        if points_key in sdata.points:
            stored_marker = zarr.open_group(
                str(Path(config.spatialdata_path) / "points" / points_key), mode="r"
            ).attrs.get(_MARKER, {})
            if isinstance(stored_marker, dict):
                point_marker = stored_marker
        if (
            previous.get("analysis_id") == analysis_id
            and points_key in sdata.points
            and table_key in sdata.tables
            and embedding_key in sdata.tables
            and point_marker.get("analysis_id") == analysis_id
            and sdata.tables[table_key].uns.get(_MARKER, {}).get("analysis_id")
            == analysis_id
            and sdata.tables[embedding_key].uns.get(_MARKER, {}).get("analysis_id")
            == analysis_id
        ):
            publication = {
                **previous,
                "source_key": source_key,
                "reused": True,
                "integrity": provenance,
                "coverage": previous.get("coverage", {}),
            }
        else:
            index = _spot_index(output_dir / "spot_assignments.csv", spot_assignments)
            if len(index.transcript_ids) != metadata.get(
                "spot_count", len(index.transcript_ids)
            ):
                raise ValueError(
                    "Modeled transcript count differs from prepared metadata"
                )
            native_series = native["transcript_id"].compute()
            native_ids, _ = _ids(native_series, "native transcript_id")
            native_count = len(native_ids)
            ordered_native = _unique(native_ids, "native transcript_id")
            _, present = _matches(ordered_native, index.transcript_ids)
            if not present.all():
                raise ValueError("Modeled transcripts are absent from native points")
            del native_series, native_ids, ordered_native
            assigned = index.assignments[index.assignments != 0]
            _, known = _matches(np.sort(shape_ids), assigned)
            if not known.all():
                raise ValueError(
                    "Transcript assignments reference unknown native cell IDs"
                )
            cells = cell_assignments
            if cells is None:
                cells = pd.read_csv(
                    output_dir / "cell_assignments.csv",
                    dtype={"instance_id": "string"},
                    keep_default_na=False,
                    na_values={"pciseq_cell_type": [""], "pciseq_probability": [""]},
                )
            all_cells = _cells_table(shape_ids, shape_key, cells, metadata)
            excluded_ids = all_cells.obs.loc[
                all_cells.obs.pciseq_status.eq("excluded_overlap"), "instance_id"
            ].to_numpy(dtype=np.uint64)
            if len(excluded_ids) and np.isin(assigned, excluded_ids).any():
                raise ValueError(
                    "Transcripts cannot be assigned to cells excluded from fitting"
                )
            marker = {"analysis_id": analysis_id, "source_key": source_key}
            all_cells.uns[_MARKER] = marker
            gene_column = str(
                resolve_col(native, ["gene", "feature_name", "target", "gene_name"])
            )
            coordinates = [axis for axis in ["x", "y", "z"] if axis in native.columns]
            if not {"x", "y"}.issubset(coordinates):
                raise ValueError("Native measured point coordinates x/y are required")
            selected = native[[*coordinates, gene_column, "transcript_id"]]
            meta = _join_partition(selected._meta, index=index, gene_column=gene_column)
            joined = selected.map_partitions(
                _join_partition, index=index, gene_column=gene_column, meta=meta
            )
            joined.attrs.pop("transform", None)
            with dask.config.set(scheduler="synchronous"):
                points = PointsModel.parse(
                    joined,
                    coordinates={axis: axis for axis in coordinates},
                    feature_key=gene_column,
                    instance_key="assignment",
                    transformations=deepcopy(get_transformation(native, get_all=True)),
                )
            points.attrs[_MARKER] = marker
            elements = [
                ("points", points_key, points),
                ("tables", table_key, all_cells),
            ]
            embedding = (
                annotated.copy()
                if annotated is not None
                else ad.read_h5ad(output_dir / "annotated.h5ad")
            )
            attrs = dict(embedding.uns.pop("spatialdata_attrs", {}))
            region = attrs.get("region")
            if isinstance(region, (list, tuple, np.ndarray)):
                region = region[0] if len(region) == 1 else None
            if not isinstance(region, str) or region not in sdata.shapes:
                raise ValueError("QC-filtered annotations require their original shape")
            variants = schema.get("segmentations", {})
            registered_variant = any(
                entry.get("shape") == region
                and entry.get("coordinate_variant_of") == branch
                for entry in variants.values()
                if isinstance(entry, dict)
            )
            if (
                region not in {shape_key, f"{shape_key}_aligned_nonrigid"}
                and not registered_variant
            ):
                raise ValueError(
                    "QC annotations link to an unrelated segmentation shape"
                )
            instance_key = attrs.get("instance_key", "instance_id")
            if instance_key not in embedding.obs:
                raise ValueError("QC-filtered annotations lack canonical instance IDs")
            embedding_ids, _ = _ids(
                embedding.obs[instance_key], "QC annotation instance_id"
            )
            _unique(embedding_ids, "QC annotation instance_id")
            _, native_coverage = _matches(np.sort(shape_ids), embedding_ids)
            linked_ids = canonical_shape_instance_series(sdata.shapes[region]).to_numpy(
                dtype=np.uint64
            )
            _, linked_coverage = _matches(
                _unique(linked_ids, "QC linked shape instance_id"), embedding_ids
            )
            if not native_coverage.all() or not linked_coverage.all():
                raise ValueError(
                    "QC cell IDs are not covered by native and linked shapes"
                )
            embedding = TableModel.parse(
                embedding,
                region=region,
                region_key=attrs.get("region_key", "region"),
                instance_key=instance_key,
            )
            embedding.uns[_MARKER] = marker
            elements.append(("tables", embedding_key, embedding))
            background_count = int(np.count_nonzero(index.assignments == 0))
            coverage = {
                "native_transcripts": native_count,
                "modelled_transcripts": len(index.transcript_ids),
                "assigned_transcripts": len(index.transcript_ids) - background_count,
                "background_transcripts": background_count,
                "not_modelled_transcripts": native_count - len(index.transcript_ids),
                "native_cells": len(shape_ids),
                "excluded_cells": int(
                    all_cells.obs.pciseq_status.eq("excluded_overlap").sum()
                ),
                "not_modelled_cells": int(
                    all_cells.obs.pciseq_status.eq("not_modelled").sum()
                ),
            }
            entry = {
                "method": "pciseq",
                "label": f"pciSeq · {_LABELS[branch]}",
                "segmentation": branch,
                "points": points_key,
                "source_points": native_key,
                "assignment_column": "assignment",
                "background_column": "background",
                "status_column": "pciseq_assignment_status",
                "shape": shape_key,
                "table": table_key,
                "embedding_table": embedding_key,
                "embedding_shape": region,
                "instance_key": "instance_id",
                "analysis_id": analysis_id,
                "coverage": coverage,
            }
            sources[source_key] = entry
            _write_elements(sdata, elements, schema)
            publication = {
                **entry,
                "source_key": source_key,
                "reused": False,
                "integrity": provenance,
            }
    target = output_dir / "viewer_publication.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(publication, indent=2) + "\n")
    os.replace(temporary, target)
    return publication

"""Modern-environment preparation, finalization, and import for MENDER."""

from __future__ import annotations

import fcntl
import json
import logging
import math
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from merxen.config import MenderConfig
from merxen.io.spatialdata_io import write_or_replace_element
from merxen.memory import force_release, log_status

logger = logging.getLogger(__name__)

CELL_ID_CANDIDATES = (
    "instance_id",
    "cell_id",
    "cell",
    "cells",
    "cell_ID",
    "EntityID",
    "cell_labels",
    "label_id",
)


def clustering_request(config: MenderConfig | dict[str, Any]) -> float | int:
    """Return the signed argument expected by ``run_clustering_normal``."""
    mode = (
        config.clustering_mode
        if isinstance(config, MenderConfig)
        else str(config["clustering_mode"])
    )
    if mode == "resolution":
        resolution = (
            config.leiden_resolution
            if isinstance(config, MenderConfig)
            else float(config["leiden_resolution"])
        )
        return -float(resolution)
    target_k = (
        config.target_k if isinstance(config, MenderConfig) else config.get("target_k")
    )
    if target_k is None or int(target_k) < 2:
        raise ValueError("MENDER target-K mode requires target_k >= 2")
    return int(target_k)


def resolve_cell_ids(adata: ad.AnnData) -> pd.Index:
    """Resolve and validate immutable cell IDs from an AnnData object."""
    spatial_attrs = dict(adata.uns.get("spatialdata_attrs", {}))
    instance_key = spatial_attrs.get("instance_key")
    if isinstance(instance_key, str) and instance_key in adata.obs:
        raw_ids = adata.obs[instance_key]
    else:
        key = next((name for name in CELL_ID_CANDIDATES if name in adata.obs), None)
        raw_ids = adata.obs[key] if key is not None else adata.obs_names
    cell_ids = pd.Index(pd.Series(raw_ids, copy=False).astype(str), name="cell_id")
    invalid = cell_ids.str.strip().isin({"", "nan", "none", "<na>"})
    if bool(invalid.any()):
        raise ValueError(f"Found {int(invalid.sum())} empty immutable cell IDs")
    if cell_ids.has_duplicates:
        duplicated = cell_ids[cell_ids.duplicated()].unique().tolist()[:5]
        raise ValueError(f"Immutable cell IDs are not unique; examples: {duplicated}")
    return cell_ids


def extract_native_centroids(shapes: Any) -> pd.DataFrame:
    """Return one native centroid per immutable cell ID from a shapes element."""
    frame = shapes.compute() if hasattr(shapes, "compute") else shapes.copy()
    if "geometry" not in frame.columns:
        raise ValueError("Native SpatialData shape has no geometry column")
    valid = frame.geometry.notna() & ~frame.geometry.is_empty
    if not bool(valid.all()):
        raise ValueError(
            f"Native SpatialData shape contains {int((~valid).sum())} empty geometries"
        )
    id_key = next((name for name in CELL_ID_CANDIDATES if name in frame), None)
    raw_ids = frame.index if id_key is None else frame[id_key]
    cell_ids = pd.Index(pd.Series(raw_ids, copy=False).astype(str), name="cell_id")
    if cell_ids.has_duplicates:
        duplicated = cell_ids[cell_ids.duplicated()].unique().tolist()[:5]
        raise ValueError(
            f"Native shape cell IDs are not unique; examples: {duplicated}"
        )
    centroids = frame.geometry.centroid
    result = pd.DataFrame(
        {
            "native_x": centroids.x.to_numpy(dtype=float),
            "native_y": centroids.y.to_numpy(dtype=float),
        },
        index=cell_ids,
    )
    finite = np.isfinite(result[["native_x", "native_y"]].to_numpy()).all(axis=1)
    if not bool(finite.all()):
        raise ValueError(
            f"Native shape contains {int((~finite).sum())} invalid centroids"
        )
    return result


def _validate_cell_states(
    adata: ad.AnnData,
    config: MenderConfig,
) -> tuple[pd.Series, np.ndarray]:
    key = config.cell_state_key
    if key not in adata.obs:
        raise KeyError(
            f"MENDER cell-state column {key!r} is absent from clustered H5AD; "
            "ordinary Leiden labels are not used as a fallback"
        )
    states = adata.obs[key]
    if not isinstance(states.dtype, pd.CategoricalDtype):
        raise TypeError(
            f"MENDER cell-state column {key!r} must be categorical, got {states.dtype}"
        )
    values = states.astype("string")
    missing = values.isna() | values.str.strip().eq("")
    if bool(missing.any()) and config.missing_state_policy == "error":
        raise ValueError(
            f"MENDER cell-state column {key!r} contains {int(missing.sum())} "
            "missing or empty values"
        )
    keep = ~missing.to_numpy()
    cleaned = pd.Series(
        pd.Categorical(values[keep].astype(str)),
        index=states.index[keep],
        name="cell_state",
    )
    if cleaned.empty or len(cleaned.cat.categories) == 0:
        raise ValueError(f"MENDER cell-state column {key!r} has no non-empty states")
    return cleaned, keep


def _source_table_ids(table: ad.AnnData) -> pd.Index:
    ids = resolve_cell_ids(table)
    return pd.Index(ids.astype(str), name="cell_id")


def _manifest_settings(config: MenderConfig) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "nn_mode": config.nn_mode,
        "radius_um": float(config.radius_um),
        "n_scales": int(config.n_scales),
        "count_rep": config.count_rep,
        "include_self": bool(config.include_self),
        "clustering_mode": config.clustering_mode,
        "leiden_resolution": float(config.leiden_resolution),
        "clustering_request": clustering_request(config),
        "random_seed": int(config.random_seed),
        "run_umap": bool(config.run_umap),
    }
    if config.target_k is not None:
        settings["target_k"] = int(config.target_k)
    return settings


def prepare_mender(config: MenderConfig, output_dir: Path | str) -> Path:
    """Validate modern inputs and export MENDER's minimal portable table."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clustered = ad.read_h5ad(config.source_h5ad)
    cell_ids = resolve_cell_ids(clustered)
    states, keep = _validate_cell_states(clustered, config)
    selected_ids = cell_ids[keep]

    import spatialdata as sd

    sdata_obj = sd.read_zarr(config.spatialdata_path)
    try:
        if config.source_spatialdata_table not in sdata_obj.tables:
            raise KeyError(
                f"Clustered SpatialData table {config.source_spatialdata_table!r} "
                f"is absent from {config.spatialdata_path}"
            )
        table_ids = _source_table_ids(sdata_obj.tables[config.source_spatialdata_table])
        if set(table_ids) != set(cell_ids):
            missing = cell_ids.difference(table_ids).tolist()[:5]
            extra = table_ids.difference(cell_ids).tolist()[:5]
            raise ValueError(
                "Clustered H5AD and clustered SpatialData table cell IDs disagree; "
                f"missing={missing}, extra={extra}"
            )
        if config.native_shape_key not in sdata_obj.shapes:
            raise KeyError(
                f"Native shape {config.native_shape_key!r} is absent from "
                f"{config.spatialdata_path}; available={list(sdata_obj.shapes)}"
            )
        centroids = extract_native_centroids(sdata_obj.shapes[config.native_shape_key])
    finally:
        del sdata_obj
        force_release(note=f"after preparing MENDER input {config.sample_id}")

    unmatched = selected_ids.difference(centroids.index)
    if len(unmatched) > 0:
        raise ValueError(
            f"Native shape {config.native_shape_key!r} has no centroid for "
            f"{len(unmatched)} MENDER cells; examples: {unmatched.tolist()[:5]}"
        )
    portable = pd.DataFrame(
        {
            "cell_id": selected_ids.astype(str),
            "native_x": centroids.loc[selected_ids, "native_x"].to_numpy(float),
            "native_y": centroids.loc[selected_ids, "native_y"].to_numpy(float),
            "cell_state": pd.Categorical(states.astype(str).to_numpy()),
        }
    )
    if portable["cell_id"].duplicated().any():
        raise ValueError("Portable MENDER table contains duplicate cell IDs")
    portable_path = output_dir / "mender_input.parquet"
    portable.to_parquet(portable_path, index=False)

    state_counts = portable["cell_state"].value_counts(sort=False)
    manifest = {
        "pair_id": config.pair_id,
        "sample_id": config.sample_id,
        "platform": config.platform,
        "segmentation": config.segmentation,
        "source_h5ad": str((config.source_h5ad_origin or config.source_h5ad).resolve()),
        "spatialdata_path": str(config.spatialdata_path.resolve()),
        "source_spatialdata_table": config.source_spatialdata_table,
        "native_shape_key": config.native_shape_key,
        "cell_state_key": config.cell_state_key,
        "missing_state_policy": config.missing_state_policy,
        "n_source_cells": int(clustered.n_obs),
        "n_cells": int(len(portable)),
        "n_missing_states": int((~keep).sum()),
        "state_counts": {str(key): int(value) for key, value in state_counts.items()},
        "coordinate_range": {
            "native_x_min": float(portable["native_x"].min()),
            "native_x_max": float(portable["native_x"].max()),
            "native_y_min": float(portable["native_y"].min()),
            "native_y_max": float(portable["native_y"].max()),
        },
        "settings": _manifest_settings(config),
        "portable_table": portable_path.name,
    }
    manifest_path = output_dir / "input_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info(
        "[%s] Prepared %d cells across %d cell states from native shape %s",
        config.sample_id,
        len(portable),
        len(state_counts),
        config.native_shape_key,
    )
    del clustered
    force_release(note=f"after writing MENDER portable input {config.sample_id}")
    return manifest_path


def _read_compute_domains(computed_dir: Path) -> pd.DataFrame:
    domains = pd.read_parquet(computed_dir / "mender_domains.parquet")
    required = {"cell_id", "mender_domain"}
    if not required.issubset(domains.columns):
        raise ValueError(
            f"MENDER domain output lacks columns {required - set(domains)}"
        )
    domains = domains.loc[:, ["cell_id", "mender_domain"]].copy()
    domains["cell_id"] = domains["cell_id"].astype(str)
    if domains["cell_id"].duplicated().any():
        raise ValueError("MENDER domain output contains duplicate cell IDs")
    if domains["mender_domain"].isna().any():
        raise ValueError("MENDER domain output contains missing domain values")
    domains["mender_domain"] = pd.Categorical(domains["mender_domain"].astype(str))
    if len(domains["mender_domain"].cat.categories) < 2:
        raise ValueError("MENDER produced fewer than two non-empty domains")
    return domains


def _add_domain_statistics(cells: pd.DataFrame, tables_dir: Path) -> None:
    contingency = (
        cells.groupby(["cell_state", "mender_domain"], observed=False)
        .size()
        .rename("cell_count")
        .reset_index()
    )
    totals = contingency.groupby("mender_domain", observed=False)[
        "cell_count"
    ].transform("sum")
    contingency["fraction_within_domain"] = contingency["cell_count"] / totals
    contingency.to_csv(tables_dir / "state_by_domain.tsv", sep="\t", index=False)

    records: list[dict[str, Any]] = []
    for domain, frame in contingency.groupby("mender_domain", observed=False):
        nonzero = frame.loc[frame["cell_count"] > 0].copy()
        fractions = nonzero["fraction_within_domain"].to_numpy(float)
        entropy = float(-(fractions * np.log2(fractions)).sum())
        dominant = nonzero.sort_values("cell_count", ascending=False).iloc[0]
        records.append(
            {
                "mender_domain": str(domain),
                "cell_count": int(nonzero["cell_count"].sum()),
                "state_entropy_bits": entropy,
                "normalized_state_entropy": (
                    entropy / math.log2(len(fractions)) if len(fractions) > 1 else 0.0
                ),
                "dominant_state": str(dominant["cell_state"]),
                "dominant_state_fraction": float(dominant["fraction_within_domain"]),
                "n_states_present": int(len(fractions)),
            }
        )
    pd.DataFrame(records).sort_values("mender_domain").to_csv(
        tables_dir / "domain_sizes.tsv", sep="\t", index=False
    )


def _save_figure(fig: plt.Figure, stem: Path, dpi: int) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_spatial_domains(cells: pd.DataFrame, plots_dir: Path, dpi: int) -> None:
    domains = pd.Categorical(cells["mender_domain"])
    fig, ax = plt.subplots(figsize=(9, 8))
    scatter = ax.scatter(
        cells["native_x"],
        cells["native_y"],
        c=domains.codes,
        cmap="tab20",
        s=2,
        linewidths=0,
        rasterized=True,
    )
    handles, _labels = scatter.legend_elements()
    ax.legend(
        handles,
        [str(value) for value in domains.categories],
        title="MENDER domain",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        markerscale=3,
    )
    ax.set(xlabel="Native x (µm)", ylabel="Native y (µm)", title="MENDER domains")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    _save_figure(fig, plots_dir / "spatial_domains", dpi)


def _plot_context_umap(
    context: ad.AnnData,
    domains: pd.DataFrame,
    plots_dir: Path,
    dpi: int,
) -> None:
    key = "X_MENDERMAP2D" if "X_MENDERMAP2D" in context.obsm else "X_umap"
    if key in context.obsm:
        coordinates = np.asarray(context.obsm[key], dtype=float)
        axis_labels = ("MENDER UMAP 1", "MENDER UMAP 2")
    elif "X_pca" in context.obsm and context.obsm["X_pca"].shape[1] >= 2:
        coordinates = np.asarray(context.obsm["X_pca"][:, :2], dtype=float)
        axis_labels = ("MENDER PC 1", "MENDER PC 2")
    else:
        coordinates = np.asarray(context.obsm["spatial"], dtype=float)
        axis_labels = ("Native x (µm)", "Native y (µm)")
    domain_lookup = domains.set_index("cell_id")["mender_domain"]
    ordered = pd.Categorical(domain_lookup.loc[context.obs_names.astype(str)])
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=ordered.codes,
        cmap="tab20",
        s=3,
        linewidths=0,
        rasterized=True,
    )
    ax.set(
        xlabel=axis_labels[0], ylabel=axis_labels[1], title="MENDER context embedding"
    )
    _save_figure(fig, plots_dir / "context_umap", dpi)


def _plot_state_domain_heatmap(
    cells: pd.DataFrame,
    plots_dir: Path,
    dpi: int,
) -> None:
    matrix = pd.crosstab(
        cells["cell_state"], cells["mender_domain"], normalize="columns"
    )
    width = max(7.0, 0.55 * matrix.shape[1] + 3.0)
    height = max(5.0, 0.28 * matrix.shape[0] + 2.0)
    fig, ax = plt.subplots(figsize=(width, height))
    sns.heatmap(matrix, cmap="viridis", ax=ax, cbar_kws={"label": "Domain fraction"})
    ax.set(title="Cell-state composition by MENDER domain", xlabel="MENDER domain")
    fig.tight_layout()
    _save_figure(fig, plots_dir / "state_domain_heatmap", dpi)


def _provenance(config: MenderConfig, input_manifest: dict[str, Any]) -> dict[str, Any]:
    result = {
        "pair_id": config.pair_id,
        "sample_id": config.sample_id,
        "platform": config.platform,
        "segmentation": config.segmentation,
        "source_h5ad": str(config.source_h5ad_origin or config.source_h5ad),
        "source_spatialdata_table": config.source_spatialdata_table,
        "native_shape_key": config.native_shape_key,
        "cell_state_key": config.cell_state_key,
        "n_cells": int(input_manifest["n_cells"]),
        "state_counts": dict(input_manifest["state_counts"]),
        **_manifest_settings(config),
    }
    return result


def finalize_mender(
    config: MenderConfig,
    prepared_dir: Path | str,
    computed_dir: Path | str,
    output_dir: Path | str,
) -> dict[str, Path]:
    """Join MENDER domains to the clustered H5AD and generate standalone QC."""
    prepared_dir = Path(prepared_dir)
    computed_dir = Path(computed_dir)
    output_root = Path(output_dir)
    sample_dir = output_root / config.platform.lower()
    input_dir = sample_dir / "input"
    tables_dir = sample_dir / "tables"
    plots_dir = sample_dir / "plots"
    for path in (input_dir, tables_dir, plots_dir):
        path.mkdir(parents=True, exist_ok=True)

    portable = pd.read_parquet(prepared_dir / "mender_input.parquet")
    portable["cell_id"] = portable["cell_id"].astype(str)
    portable["cell_state"] = pd.Categorical(portable["cell_state"].astype(str))
    domains = _read_compute_domains(computed_dir)
    if set(portable["cell_id"]) != set(domains["cell_id"]):
        raise ValueError("MENDER domain IDs do not exactly match portable input IDs")
    cells = portable.merge(domains, on="cell_id", how="left", validate="one_to_one")
    if cells["mender_domain"].isna().any():
        raise ValueError("At least one portable input cell has no MENDER domain")
    cells["mender_domain"] = pd.Categorical(cells["mender_domain"].astype(str))

    original = ad.read_h5ad(config.source_h5ad)
    original_ids = resolve_cell_ids(original)
    domain_lookup = domains.set_index("cell_id")["mender_domain"]
    missing = original_ids.difference(domain_lookup.index)
    if len(missing) > 0:
        raise ValueError(
            f"Cannot annotate source H5AD: {len(missing)} cells lack MENDER domains"
        )
    original.obs["mender_domain"] = pd.Categorical(
        domain_lookup.loc[original_ids].astype(str).to_numpy()
    )
    input_manifest = json.loads((prepared_dir / "input_manifest.json").read_text())
    provenance = _provenance(config, input_manifest)
    original.uns["merxen_mender"] = provenance

    annotated_path = sample_dir / f"{config.sample_id}_mender_annotated.h5ad"
    context_path = sample_dir / f"{config.sample_id}_mender_context.h5ad"
    cells_path = sample_dir / f"{config.sample_id}_mender_cells.parquet"
    original.write_h5ad(annotated_path)
    shutil.copy2(computed_dir / "mender_context.h5ad", context_path)
    cells.to_parquet(cells_path, index=False)
    shutil.copy2(
        prepared_dir / "input_manifest.json", input_dir / "input_manifest.json"
    )
    shutil.copy2(
        computed_dir / "scale_neighbour_summary.tsv",
        tables_dir / "scale_neighbour_summary.tsv",
    )
    _add_domain_statistics(cells, tables_dir)

    context = ad.read_h5ad(context_path)
    _plot_spatial_domains(cells, plots_dir, config.figure_dpi)
    _plot_context_umap(context, domains, plots_dir, config.figure_dpi)
    _plot_state_domain_heatmap(cells, plots_dir, config.figure_dpi)

    output_manifest = {
        **provenance,
        "domain_counts": {
            str(key): int(value)
            for key, value in cells["mender_domain"].value_counts(sort=False).items()
        },
        "artifacts": {
            "annotated_h5ad": annotated_path.name,
            "context_h5ad": context_path.name,
            "cells_parquet": cells_path.name,
            "input_manifest": "input/input_manifest.json",
            "domain_sizes": "tables/domain_sizes.tsv",
            "state_by_domain": "tables/state_by_domain.tsv",
            "scale_neighbour_summary": "tables/scale_neighbour_summary.tsv",
        },
    }
    manifest_path = sample_dir / "mender_manifest.json"
    manifest_path.write_text(json.dumps(output_manifest, indent=2) + "\n")
    del original, context
    force_release(note=f"after finalizing MENDER output {config.sample_id}")
    return {
        "annotated_h5ad": annotated_path,
        "context_h5ad": context_path,
        "cells_parquet": cells_path,
        "manifest": manifest_path,
    }


@contextmanager
def spatialdata_write_lock(zarr_path: Path | str) -> Iterator[Path]:
    """Acquire the shared MerXen writer lock for one SpatialData store."""
    lock_path = Path(f"{Path(zarr_path)}.merxen-write.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield lock_path
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _spatialdata_table_contract(table: ad.AnnData) -> tuple[str, str, str]:
    attrs = dict(table.uns.get("spatialdata_attrs", {}))
    region_key = str(attrs.get("region_key", "region"))
    instance_key = attrs.get("instance_key")
    if not isinstance(instance_key, str) or instance_key not in table.obs:
        instance_key = next(
            (name for name in CELL_ID_CANDIDATES if name in table.obs),
            None,
        )
    if instance_key is None:
        raise ValueError("Clustered SpatialData table has no immutable instance key")
    region = attrs.get("region")
    if isinstance(region, list | tuple):
        if len(region) != 1:
            raise ValueError("MENDER import requires a single-region clustered table")
        region = region[0]
    if not isinstance(region, str) or not region:
        if region_key not in table.obs:
            raise ValueError("Clustered SpatialData table has no region metadata")
        regions = table.obs[region_key].astype(str).unique()
        if len(regions) != 1:
            raise ValueError("MENDER import requires exactly one table region")
        region = str(regions[0])
    return str(region), region_key, str(instance_key)


def import_mender_spatialdata(
    config: MenderConfig,
    finalized_dir: Path | str,
    manifest_path: Path | str,
) -> Path:
    """Import only ``mender_domain`` and its provenance into the clustered table."""
    output_manifest_path = Path(manifest_path)
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    finalized_path = Path(finalized_dir)
    sample_dir = (
        finalized_path
        if finalized_path.name == config.platform.lower()
        else finalized_path / config.platform.lower()
    )
    annotated_path = sample_dir / f"{config.sample_id}_mender_annotated.h5ad"
    annotated = ad.read_h5ad(annotated_path)
    annotated_ids = resolve_cell_ids(annotated)
    domains = pd.Series(
        annotated.obs["mender_domain"].astype(str).to_numpy(),
        index=annotated_ids,
        name="mender_domain",
    )
    provenance = dict(annotated.uns["merxen_mender"])

    imported = False
    lock_path = Path(f"{config.spatialdata_path}.merxen-write.lock")
    if config.write_spatialdata_table:
        import spatialdata as sd
        from spatialdata.models import TableModel

        with spatialdata_write_lock(config.spatialdata_path) as acquired_lock:
            lock_path = acquired_lock
            sdata_obj = sd.read_zarr(config.spatialdata_path)
            try:
                if config.source_spatialdata_table not in sdata_obj.tables:
                    table_key = config.source_spatialdata_table
                    raise KeyError(
                        f"Clustered SpatialData table {table_key!r} "
                        "disappeared before MENDER import"
                    )
                table = sdata_obj.tables[config.source_spatialdata_table].copy()
                region, region_key, instance_key = _spatialdata_table_contract(table)
                table_ids = pd.Index(
                    table.obs[instance_key].astype(str), name="cell_id"
                )
                if set(table_ids) != set(domains.index):
                    raise ValueError(
                        "MENDER annotated H5AD and clustered SpatialData table "
                        "cell IDs do not match"
                    )
                table.obs["mender_domain"] = pd.Categorical(
                    domains.loc[table_ids].astype(str).to_numpy()
                )
                table.uns["merxen_mender"] = provenance
                table.uns.pop("spatialdata_attrs", None)
                parsed = TableModel.parse(
                    table,
                    region=region,
                    region_key=region_key,
                    instance_key=instance_key,
                )
                write_or_replace_element(
                    sdata_obj,
                    config.source_spatialdata_table,
                    "tables",
                    parsed,
                    overwrite=True,
                )
                imported = True
            finally:
                del sdata_obj
                force_release(note=f"after importing MENDER table {config.sample_id}")

    result = {
        "sample_id": config.sample_id,
        "platform": config.platform,
        "segmentation": config.segmentation,
        "spatialdata_path": str(config.spatialdata_path.resolve()),
        "spatialdata_table": config.source_spatialdata_table,
        "annotated_h5ad": str(annotated_path.resolve()),
        "write_spatialdata_table": bool(config.write_spatialdata_table),
        "imported": imported,
        "lock_path": str(lock_path),
        "n_cells": int(len(domains)),
        "domain_counts": {
            str(key): int(value) for key, value in domains.value_counts().items()
        },
    }
    output_manifest_path.write_text(json.dumps(result, indent=2) + "\n")
    log_status(
        f"[{config.sample_id}] MENDER SpatialData import "
        f"{'completed' if imported else 'disabled'}"
    )
    del annotated
    return output_manifest_path

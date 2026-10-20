"""Pinned-environment CPU wrapper around MENDER's single-slice model."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


def build_minimal_anndata(portable: pd.DataFrame) -> ad.AnnData:
    """Construct the expression-free AnnData consumed by MENDER."""
    required = ["cell_id", "native_x", "native_y", "cell_state"]
    missing = [column for column in required if column not in portable]
    if missing:
        raise ValueError(f"Portable MENDER input lacks columns: {missing}")
    frame = portable.loc[:, required].copy()
    frame["cell_id"] = frame["cell_id"].astype(str)
    if frame["cell_id"].duplicated().any():
        raise ValueError("Portable MENDER input contains duplicate cell IDs")
    if frame["cell_state"].isna().any():
        raise ValueError("Portable MENDER input contains missing cell states")
    frame["cell_state"] = pd.Categorical(frame["cell_state"].astype(str))
    coordinates = frame[["native_x", "native_y"]].to_numpy(dtype=float)
    if not np.isfinite(coordinates).all():
        raise ValueError("Portable MENDER input contains invalid native coordinates")
    obs = pd.DataFrame(
        {"cell_state": frame["cell_state"].to_numpy()},
        index=pd.Index(frame["cell_id"], name="cell_id"),
    )
    obs["cell_state"] = pd.Categorical(obs["cell_state"].astype(str))
    adata = ad.AnnData(
        X=sparse.csr_matrix((len(frame), 0), dtype=np.float32),
        obs=obs,
    )
    adata.obsm["spatial"] = coordinates
    return adata


def clustering_request(config: dict[str, Any]) -> float | int:
    """Return MENDER's negative-resolution or positive-target-K argument."""
    mode = str(config.get("clustering_mode", "resolution"))
    if mode == "resolution":
        resolution = float(config.get("leiden_resolution", 0.8))
        if resolution <= 0:
            raise ValueError("MENDER Leiden resolution must be positive")
        return -resolution
    if mode != "target_k":
        raise ValueError(f"Unknown MENDER clustering mode: {mode!r}")
    target_k = config.get("target_k")
    if target_k is None or int(target_k) < 2:
        raise ValueError("MENDER target-K mode requires target_k >= 2")
    return int(target_k)


def validate_mender_result(
    model: Any,
    input_adata: ad.AnnData,
) -> ad.AnnData:
    """Enforce MENDER postconditions instead of trusting printed errors."""
    context = getattr(model, "adata_MENDER", None)
    if not isinstance(context, ad.AnnData):
        raise RuntimeError("MENDER did not create an adata_MENDER result")
    if "MENDER" not in context.obs:
        raise RuntimeError("MENDER result is missing obs['MENDER']")
    input_ids = pd.Index(input_adata.obs_names.astype(str))
    result_ids = pd.Index(context.obs_names.astype(str))
    if result_ids.has_duplicates:
        raise RuntimeError("MENDER result contains duplicate cell IDs")
    if len(result_ids) != len(input_ids) or set(result_ids) != set(input_ids):
        dropped = input_ids.difference(result_ids).tolist()[:5]
        added = result_ids.difference(input_ids).tolist()[:5]
        raise RuntimeError(
            "MENDER result cell IDs do not round-trip exactly; "
            f"dropped={dropped}, added={added}"
        )
    domains = context.obs["MENDER"]
    missing = domains.isna() | domains.astype("string").str.strip().eq("")
    if bool(missing.any()):
        raise RuntimeError(f"MENDER produced {int(missing.sum())} missing domains")
    domains = pd.Categorical(domains.astype(str))
    if len(domains.categories) < 2:
        raise RuntimeError("MENDER produced fewer than two non-empty domains")
    context.obs["MENDER"] = domains
    if "spatial" not in context.obsm:
        raise RuntimeError("MENDER context result lost native coordinates")
    input_lookup = pd.DataFrame(
        np.asarray(input_adata.obsm["spatial"], dtype=float),
        index=input_ids,
    )
    result_lookup = pd.DataFrame(
        np.asarray(context.obsm["spatial"], dtype=float),
        index=result_ids,
    )
    if not np.array_equal(
        input_lookup.loc[result_ids].to_numpy(),
        result_lookup.to_numpy(),
    ):
        raise RuntimeError("MENDER changed native spatial coordinates")
    return context


def _scale_neighbour_summary(model: Any, n_scales: int) -> pd.DataFrame:
    records: list[dict[str, float | int]] = []
    for scale in range(n_scales):
        key = f"scale{scale}"
        if key not in model.adata.obsm:
            raise RuntimeError(
                f"MENDER result is missing per-scale representation {key}"
            )
        counts = np.asarray(model.adata.obsm[key]).sum(axis=1).astype(float)
        records.append(
            {
                "scale": scale,
                "cell_count_min": float(np.min(counts)),
                "cell_count_median": float(np.median(counts)),
                "cell_count_mean": float(np.mean(counts)),
                "cell_count_max": float(np.max(counts)),
            }
        )
    return pd.DataFrame(records)


def run_mender_compute(
    config_path: Path | str,
    prepared_dir: Path | str,
    output_dir: Path | str,
) -> dict[str, Path]:
    """Run one CPU-only ``MENDER_single`` acquisition."""
    config_path = Path(config_path)
    prepared_dir = Path(prepared_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text())
    portable = pd.read_parquet(prepared_dir / "mender_input.parquet")
    adata = build_minimal_anndata(portable)

    mender_module = importlib.import_module("MENDER")
    model = mender_module.MENDER_single(
        adata,
        ct_obs="cell_state",
        random_seed=int(config.get("random_seed", 666)),
    )
    model.set_MENDER_para(
        nn_mode=str(config.get("nn_mode", "radius")),
        nn_para=float(config.get("radius_um", 20.0)),
        count_rep=str(config.get("count_rep", "s")),
        include_self=bool(config.get("include_self", False)),
        n_scales=int(config.get("n_scales", 5)),
    )
    model.run_representation()
    request = clustering_request(config)
    model.run_clustering_normal(
        request,
        run_umap=bool(config.get("run_umap", True)),
    )
    context = validate_mender_result(model, adata)

    context_path = output_dir / "mender_context.h5ad"
    domains_path = output_dir / "mender_domains.parquet"
    scale_path = output_dir / "scale_neighbour_summary.tsv"
    manifest_path = output_dir / "compute_manifest.json"
    context.write_h5ad(context_path)
    pd.DataFrame(
        {
            "cell_id": context.obs_names.astype(str),
            "mender_domain": pd.Categorical(context.obs["MENDER"].astype(str)),
        }
    ).to_parquet(domains_path, index=False)
    _scale_neighbour_summary(model, int(config.get("n_scales", 5))).to_csv(
        scale_path,
        sep="\t",
        index=False,
    )
    manifest = {
        "sample_id": config["sample_id"],
        "platform": config["platform"],
        "segmentation": config["segmentation"],
        "n_cells": int(context.n_obs),
        "n_context_features": int(context.n_vars),
        "clustering_mode": config.get("clustering_mode", "resolution"),
        "clustering_request": request,
        "domain_counts": {
            str(key): int(value)
            for key, value in context.obs["MENDER"].value_counts().items()
        },
        "cpu_only": True,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "context_h5ad": context_path,
        "domains": domains_path,
        "scale_neighbour_summary": scale_path,
        "manifest": manifest_path,
    }


def main() -> None:
    """Run the isolated MENDER compute wrapper."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_mender_compute(args.config, args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()

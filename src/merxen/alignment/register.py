"""Cross-section registration using Spateo plus MerXen transform helpers."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

from merxen.alignment.features import (
    build_alignment_adata,
    prepare_spateo_features,
    shared_gene_subset,
)
from merxen.alignment.transforms import (
    NonRigidTransform,
    apply_affine_matrix,
    fit_affine_matrix,
    fit_nonrigid_transform,
    transform_displacement_summary,
)
from merxen.config import AlignmentConfig, SpateoAlignmentConfig

SPATEO_INSTALL_MESSAGE = (
    "Spateo alignment requires the optional alignment dependencies. "
    "Install them with: pip install spateo-release==1.1.1 "
    '&& pip install "anndata>=0.12.10"'
)

SpateoRunner = Callable[
    [ad.AnnData, ad.AnnData, SpateoAlignmentConfig],
    tuple[ad.AnnData, ad.AnnData],
]


@dataclass(frozen=True)
class TransformResult:
    """Container for registration outputs.

    Attributes:
        merscope_to_common: Transform object mapping MERSCOPE coordinates to a
            shared reference space.
        xenium_to_common: Transform object mapping Xenium coordinates to the
            same shared reference space.
        metadata: Additional implementation-specific information.
    """

    merscope_to_common: dict[str, Any]
    xenium_to_common: dict[str, Any]
    metadata: dict[str, Any]
    coordinate_tables: dict[str, pd.DataFrame] | None = None
    nonrigid_transform: NonRigidTransform | None = None


def register_pair(
    merscope_sdata: Any,
    xenium_sdata: Any,
    config: Any,
    *,
    spateo_runner: SpateoRunner | None = None,
) -> TransformResult:
    """Register paired MERSCOPE and Xenium sections to Xenium coordinate space."""
    cfg = _coerce_alignment_config(config)
    spateo_cfg = cfg.spateo

    fixed_raw = build_alignment_adata(
        xenium_sdata,
        platform="XENIUM",
        include_image_features=spateo_cfg.include_image_features,
    )
    moving_raw = build_alignment_adata(
        merscope_sdata,
        platform="MERSCOPE",
        include_image_features=spateo_cfg.include_image_features,
    )
    fixed_raw, moving_raw = shared_gene_subset(fixed_raw, moving_raw)

    fixed_for_spateo = prepare_spateo_features(
        fixed_raw,
        normalize_total=spateo_cfg.normalize_total,
        log1p=spateo_cfg.log1p,
        use_hvg=spateo_cfg.use_hvg,
        n_top_genes=spateo_cfg.n_top_genes,
    )
    moving_for_spateo = prepare_spateo_features(
        moving_raw,
        normalize_total=spateo_cfg.normalize_total,
        log1p=spateo_cfg.log1p,
        use_hvg=spateo_cfg.use_hvg,
        n_top_genes=spateo_cfg.n_top_genes,
    )
    fixed_for_spateo, moving_for_spateo = shared_gene_subset(
        fixed_for_spateo, moving_for_spateo
    )

    fixed_aligned, moving_aligned, tuning_records = _run_spateo_candidates(
        fixed_for_spateo,
        moving_for_spateo,
        spateo_cfg,
        spateo_runner=spateo_runner,
    )

    moving_source = np.asarray(moving_aligned.obsm["spatial"], dtype=np.float64)
    fixed_source = np.asarray(fixed_aligned.obsm["spatial"], dtype=np.float64)
    moving_rigid = _obsm_or(moving_aligned, "align_spatial_rigid", "align_spatial")
    moving_nonrigid = _obsm_or(
        moving_aligned,
        "align_spatial_nonrigid",
        "align_spatial",
        fallback=moving_rigid,
    )
    fixed_common = _obsm_or(fixed_aligned, "align_spatial", "spatial")

    affine = fit_affine_matrix(moving_source, moving_rigid)
    nonrigid = fit_nonrigid_transform(
        moving_source,
        moving_nonrigid,
        affine_matrix=affine,
        neighbors=spateo_cfg.rbf_neighbors,
        smoothing=spateo_cfg.rbf_smoothing,
        max_anchors=spateo_cfg.max_nonrigid_anchors,
    )

    coordinate_tables = {
        "merscope": _coords_table(
            moving_aligned.obs_names.astype(str),
            raw=moving_source,
            rigid=moving_rigid,
            nonrigid=moving_nonrigid,
        ),
        "xenium": _coords_table(
            fixed_aligned.obs_names.astype(str),
            raw=fixed_source,
            rigid=_obsm_or(fixed_aligned, "align_spatial_rigid", "spatial"),
            nonrigid=fixed_common,
        ),
    }

    moving_target = (
        moving_nonrigid if spateo_cfg.selected_mode == "nonrigid" else moving_rigid
    )
    metadata = {
        "method": "spateo_morpho_align",
        "pair_id": cfg.pair_id,
        "fixed_platform": cfg.fixed_platform,
        "moving_platform": cfg.moving_platform,
        "selected_mode": spateo_cfg.selected_mode,
        "n_alignment_cells": {
            "xenium": int(fixed_aligned.n_obs),
            "merscope": int(moving_aligned.n_obs),
        },
        "n_alignment_features": int(fixed_aligned.n_vars),
        "spateo": spateo_cfg.model_dump(),
        "tuning": tuning_records,
        "displacement": {
            "rigid": transform_displacement_summary(moving_source, moving_rigid),
            "nonrigid": transform_displacement_summary(moving_source, moving_nonrigid),
            "selected": transform_displacement_summary(moving_source, moving_target),
        },
    }

    return TransformResult(
        merscope_to_common={
            "type": "affine_plus_rbf",
            "selected_mode": spateo_cfg.selected_mode,
            "rigid_affine_matrix": affine.tolist(),
            "nonrigid_support_radius": nonrigid.support_radius,
            "rbf_neighbors": nonrigid.neighbors,
            "rbf_smoothing": nonrigid.smoothing,
        },
        xenium_to_common={
            "type": "identity",
            "affine_matrix": np.eye(3, dtype=float).tolist(),
        },
        metadata=metadata,
        coordinate_tables=coordinate_tables,
        nonrigid_transform=nonrigid,
    )


def run_spateo_alignment(
    fixed: ad.AnnData,
    moving: ad.AnnData,
    config: SpateoAlignmentConfig,
) -> tuple[ad.AnnData, ad.AnnData]:
    """Run Spateo's morpho_align and return fixed/moving aligned AnnData."""
    _apply_spateo_import_shims()
    try:
        import spateo as st
        from spateo.alignment.morpho_alignment import Morpho_pairwise
    except ImportError as exc:
        raise RuntimeError(SPATEO_INSTALL_MESSAGE) from exc

    device = _resolve_device(config.device)
    aligned_result = st.align.morpho_align(
        models=[fixed.copy(), moving.copy()],
        spatial_key="spatial",
        key_added="align_spatial",
        mode=config.mode,
        device=device,
        dtype=config.dtype,
        max_iter=config.max_iter,
        verbose=True,
        **_spateo_pairwise_kwargs(config, Morpho_pairwise),
    )
    aligned = (
        aligned_result[0]
        if isinstance(aligned_result, tuple) and len(aligned_result) > 0
        else aligned_result
    )
    if len(aligned) != 2:
        raise RuntimeError(
            f"Expected two aligned slices from Spateo, got {len(aligned)}"
        )
    return aligned[0], aligned[1]


def _spateo_pairwise_kwargs(
    config: SpateoAlignmentConfig,
    pairwise_cls: Any,
) -> dict[str, Any]:
    """Build Spateo kwargs supported by the installed pairwise API."""
    supported = set(inspect.signature(pairwise_cls.__init__).parameters)
    candidates: dict[str, Any] = {
        "beta": config.beta,
        "lambdaVF": config.lambda_vf,
        "K": config.k,
        "partial_robust_level": config.partial_robust_level,
        "SVI_mode": config.SVI_mode,
        "sparse_calculation_mode": config.sparse_calculation_mode,
        "use_chunk": config.use_chunk,
        "chunk_capacity": config.chunk_capacity,
    }
    if "n_sampling" in supported:
        candidates["n_sampling"] = config.n_sampling
    elif "batch_size" in supported:
        candidates["batch_size"] = config.n_sampling
    return {key: value for key, value in candidates.items() if key in supported}


def _apply_spateo_import_shims() -> None:
    """Patch narrow compatibility symbols Spateo imports at module import time.

    Spateo 1.1.1 imports its broader segmentation/data I/O stack even when only
    ``spateo.align`` is needed. Current MerXen dependencies keep modern
    AnnData/Cellpose for SpatialData and segmentation, so we provide the legacy
    names Spateo imports without downgrading those core packages.
    """
    try:
        import anndata

        if not hasattr(anndata, "read") and hasattr(anndata, "read_h5ad"):
            anndata.read = anndata.read_h5ad
    except Exception:  # noqa: BLE001
        pass

    try:
        import cellpose.models as cellpose_models

        if not hasattr(cellpose_models, "Cellpose") and hasattr(
            cellpose_models, "CellposeModel"
        ):
            cellpose_models.Cellpose = cellpose_models.CellposeModel
    except Exception:  # noqa: BLE001
        pass


def _run_spateo_candidates(
    fixed: ad.AnnData,
    moving: ad.AnnData,
    config: SpateoAlignmentConfig,
    *,
    spateo_runner: SpateoRunner | None,
) -> tuple[ad.AnnData, ad.AnnData, list[dict[str, Any]]]:
    runner = run_spateo_alignment if spateo_runner is None else spateo_runner
    candidates = _candidate_spateo_configs(config)
    best_fixed: ad.AnnData | None = None
    best_moving: ad.AnnData | None = None
    best_score = -np.inf
    records: list[dict[str, Any]] = []

    for idx, candidate in enumerate(candidates):
        aligned_fixed, aligned_moving = runner(fixed, moving, candidate)
        score, metrics = _score_aligned_candidate(
            aligned_fixed,
            aligned_moving,
            selected_mode=candidate.selected_mode,
        )
        record = {
            "candidate": int(idx),
            "score": float(score),
            "params": _candidate_param_delta(config, candidate),
            "metrics": metrics,
        }
        records.append(record)
        if score > best_score:
            best_score = float(score)
            best_fixed = aligned_fixed
            best_moving = aligned_moving

    if best_fixed is None or best_moving is None:
        raise RuntimeError("No Spateo alignment candidates were evaluated")
    return best_fixed, best_moving, records


def _candidate_spateo_configs(
    config: SpateoAlignmentConfig,
) -> list[SpateoAlignmentConfig]:
    if not config.tune:
        return [config]

    grid = config.param_grid or [
        {},
        {"partial_robust_level": 25},
        {"partial_robust_level": 75},
        {"SVI_mode": True, "n_sampling": 20_000},
    ]
    base = config.model_dump()
    return [SpateoAlignmentConfig.model_validate(base | params) for params in grid]


def _score_aligned_candidate(
    fixed: ad.AnnData,
    moving: ad.AnnData,
    *,
    selected_mode: str,
) -> tuple[float, dict[str, Any]]:
    from merxen.alignment.qc import compute_grid_alignment_metrics

    fixed_eval = fixed.copy()
    moving_eval = moving.copy()
    fixed_eval.obsm["spatial"] = _obsm_or(fixed, "align_spatial", "spatial")
    moving_key = (
        "align_spatial_nonrigid"
        if selected_mode == "nonrigid"
        else "align_spatial_rigid"
    )
    moving_eval.obsm["spatial"] = _obsm_or(
        moving,
        moving_key,
        "align_spatial",
        fallback=moving.obsm["spatial"],
    )
    metrics = compute_grid_alignment_metrics(fixed_eval, moving_eval)
    score = (
        _finite_or_zero(metrics.get("gene_grid_pearson"))
        + _finite_or_zero(metrics.get("grid_cosine"))
        + 0.1 * _finite_or_zero(metrics.get("grid_mutual_information"))
        - 0.001 * _finite_or_zero(metrics.get("centroid_assd"))
    )
    return float(score), metrics


def _candidate_param_delta(
    base: SpateoAlignmentConfig,
    candidate: SpateoAlignmentConfig,
) -> dict[str, Any]:
    base_dump = base.model_dump()
    candidate_dump = candidate.model_dump()
    return {
        key: value
        for key, value in candidate_dump.items()
        if base_dump.get(key) != value or key in {"tune", "param_grid"}
    }


def _finite_or_zero(value: Any) -> float:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return 0.0
    return val if np.isfinite(val) else 0.0


def transform_xy_for_result(result: TransformResult, coords: Any) -> np.ndarray:
    """Transform MERSCOPE xy coordinates with the selected result transform."""
    selected = result.merscope_to_common.get("selected_mode", "nonrigid")
    if selected == "nonrigid" and result.nonrigid_transform is not None:
        return result.nonrigid_transform.transform(coords)
    matrix = result.merscope_to_common["rigid_affine_matrix"]
    return apply_affine_matrix(coords, matrix)


def _coerce_alignment_config(config: Any) -> AlignmentConfig:
    if isinstance(config, AlignmentConfig):
        return config
    if isinstance(config, dict):
        return AlignmentConfig.model_validate(config)
    raise TypeError(
        "register_pair expects an AlignmentConfig or dict. "
        f"Got {type(config).__name__}."
    )


def _resolve_device(device: str) -> str:
    normalized = str(device).strip().lower()
    if normalized in {"", "auto"}:
        try:
            import torch

            return "0" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            return "cpu"
    if normalized == "cpu":
        return "cpu"
    if normalized == "cuda":
        return "0"
    if normalized.startswith("cuda:"):
        cuda_device = normalized.removeprefix("cuda:").strip()
        return cuda_device or "0"
    return str(device).strip()


def _obsm_or(
    adata: ad.AnnData,
    preferred: str,
    fallback_key: str,
    *,
    fallback: Any = None,
) -> np.ndarray:
    if preferred in adata.obsm:
        return np.asarray(adata.obsm[preferred], dtype=np.float64)
    if fallback_key in adata.obsm:
        return np.asarray(adata.obsm[fallback_key], dtype=np.float64)
    if fallback is not None:
        return np.asarray(fallback, dtype=np.float64)
    raise KeyError(
        f"AnnData does not contain obsm['{preferred}'] or obsm['{fallback_key}']"
    )


def _coords_table(
    cell_ids: Any,
    *,
    raw: np.ndarray,
    rigid: np.ndarray,
    nonrigid: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cell_id": list(cell_ids),
            "raw_x": raw[:, 0],
            "raw_y": raw[:, 1],
            "rigid_x": rigid[:, 0],
            "rigid_y": rigid[:, 1],
            "nonrigid_x": nonrigid[:, 0],
            "nonrigid_y": nonrigid[:, 1],
        }
    )

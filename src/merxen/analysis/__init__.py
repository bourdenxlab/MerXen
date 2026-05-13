"""Downstream analysis modules for MerXen."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "add_qc_metrics": "merxen.analysis.clustering_squidpy",
    "adata_from_spatialdata": "merxen.analysis.clustering_squidpy",
    "load_spatialdata_adata": "merxen.analysis.clustering_squidpy",
    "plot_qc_histograms": "merxen.analysis.clustering_squidpy",
    "plot_spatial_scatter": "merxen.analysis.clustering_squidpy",
    "plot_umap": "merxen.analysis.clustering_squidpy",
    "remove_control_features": "merxen.analysis.clustering_squidpy",
    "run_clustering_squidpy": "merxen.analysis.clustering_squidpy",
    "run_scanpy_clustering": "merxen.analysis.clustering_squidpy",
    "save_clustered_adata": "merxen.analysis.clustering_squidpy",
    "save_qc_metrics": "merxen.analysis.clustering_squidpy",
    "annotate_h5ad_with_mapmycells": "merxen.analysis.mapmycells",
    "build_mapmycells_command": "merxen.analysis.mapmycells",
    "prepare_region_mapmycells_reference": "merxen.analysis.mapmycells",
    "prepare_mapmycells_query": "merxen.analysis.mapmycells",
    "read_mapmycells_csv": "merxen.analysis.mapmycells",
    "run_mapmycells": "merxen.analysis.mapmycells",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    """Lazily expose analysis functions without importing every dependency."""
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORT_MODULES[name])
    value = getattr(module, name)
    globals()[name] = value
    return value

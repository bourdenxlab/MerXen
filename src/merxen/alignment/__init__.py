"""MerXen alignment subpackage.

The public names load on first use (PEP 562). Importing a light submodule such as
``merxen.alignment.image_warp`` or ``merxen.alignment.manifest`` (the manual
registration and the enrichment do) must not import VALIS, PyTorch and the rest of
the alignment pipeline, which cost about 0.6 GB of resident memory and several
seconds.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from merxen.alignment.bundle import DisplacementField, ValisTransformBundle
    from merxen.alignment.pipeline import run_alignment_pipeline
    from merxen.alignment.qc import compute_grid_alignment_metrics, run_alignment_qc
    from merxen.alignment.register import TransformResult, register_pair
    from merxen.alignment.transforms import (
        NonRigidTransform,
        apply_affine_matrix,
        fit_affine_matrix,
        fit_nonrigid_transform,
    )

_EXPORTS = {
    "DisplacementField": "merxen.alignment.bundle",
    "ValisTransformBundle": "merxen.alignment.bundle",
    "run_alignment_pipeline": "merxen.alignment.pipeline",
    "compute_grid_alignment_metrics": "merxen.alignment.qc",
    "run_alignment_qc": "merxen.alignment.qc",
    "TransformResult": "merxen.alignment.register",
    "register_pair": "merxen.alignment.register",
    "NonRigidTransform": "merxen.alignment.transforms",
    "apply_affine_matrix": "merxen.alignment.transforms",
    "fit_affine_matrix": "merxen.alignment.transforms",
    "fit_nonrigid_transform": "merxen.alignment.transforms",
}

__all__ = [
    "DisplacementField",
    "NonRigidTransform",
    "TransformResult",
    "ValisTransformBundle",
    "apply_affine_matrix",
    "compute_grid_alignment_metrics",
    "fit_affine_matrix",
    "fit_nonrigid_transform",
    "register_pair",
    "run_alignment_pipeline",
    "run_alignment_qc",
]


def __getattr__(name: str) -> Any:
    """Import the module that defines a public name the first time it is used."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value

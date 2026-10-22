"""Optional removal of coverslip fiducial beads from MERSCOPE image planes.

The "psf_subtract" method models every bead with an empirical radial PSF fitted
across the z planes and subtracts that model from the PolyT and DAPI planes
before the max projection. It is off by default and never imports the GPU code.

The package is lazy: importing it (or one of its light submodules such as
``provenance``, ``params`` or ``backend``, which the default build, segmentation,
enrichment and quantification runs import for the image-preprocessing record)
loads none of the correction code. The public names below are resolved on first
use (PEP 562), so a default run does not import scikit-image, the fitting code or
the driver; import the submodules (``detect``, ``model``, ...) for the
internals.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from merxen.io.bead_removal.apply import apply_bead_model, correct_window
    from merxen.io.bead_removal.calibrate import (
        BeadCalibration,
        BeadCalibrationError,
        calibrate_bead_model,
    )
    from merxen.io.bead_removal.detect import ClassifierThresholds
    from merxen.io.bead_removal.driver import BeadRemovalRunResult, run_bead_removal
    from merxen.io.bead_removal.params import (
        BEAD_REMOVAL_VERSION,
        DEFAULT_PARAMS,
        PsfSubtractParams,
        params_sha256,
    )
    from merxen.io.bead_removal.psf import PsfLibrary
    from merxen.io.bead_removal.qc import QC_DIRNAME, write_bead_removal_qc
    from merxen.io.bead_removal.reader import MosaicPlaneReader, list_mosaic_planes

# Public name -> the submodule that defines it.
_EXPORTS = {
    "BEAD_REMOVAL_VERSION": "params",
    "DEFAULT_PARAMS": "params",
    "BeadCalibration": "calibrate",
    "BeadCalibrationError": "calibrate",
    "BeadRemovalRunResult": "driver",
    "ClassifierThresholds": "detect",
    "MosaicPlaneReader": "reader",
    "PsfLibrary": "psf",
    "PsfSubtractParams": "params",
    "QC_DIRNAME": "qc",
    "apply_bead_model": "apply",
    "calibrate_bead_model": "calibrate",
    "correct_window": "apply",
    "list_mosaic_planes": "reader",
    "params_sha256": "params",
    "run_bead_removal": "driver",
    "write_bead_removal_qc": "qc",
}
_SUBMODULES = frozenset(_EXPORTS.values()) | {
    "backend",
    "fill",
    "gpu",
    "model",
    "provenance",
    "qc_plots",
}

__all__ = [
    "BEAD_REMOVAL_VERSION",
    "DEFAULT_PARAMS",
    "BeadCalibration",
    "BeadCalibrationError",
    "BeadRemovalRunResult",
    "ClassifierThresholds",
    "MosaicPlaneReader",
    "PsfLibrary",
    "PsfSubtractParams",
    "QC_DIRNAME",
    "apply_bead_model",
    "calibrate_bead_model",
    "correct_window",
    "list_mosaic_planes",
    "params_sha256",
    "run_bead_removal",
    "write_bead_removal_qc",
]


def __getattr__(name: str) -> Any:
    """Import a public name, or a submodule used as an attribute, on first use."""
    if name in _EXPORTS:
        module = importlib.import_module(f"{__name__}.{_EXPORTS[name]}")
        value = getattr(module, name)
    elif name in _SUBMODULES:
        value = importlib.import_module(f"{__name__}.{name}")
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS, *_SUBMODULES})

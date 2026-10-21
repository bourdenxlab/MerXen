"""Optional removal of coverslip fiducial beads from MERSCOPE image planes.

The "psf_subtract" method models every bead with an empirical radial PSF fitted
across the z planes and subtracts that model from the PolyT and DAPI planes
before the max projection. It is off by default and never imports the GPU code.

The public names are re-exported here as the pieces land; import the submodules
(``detect``, ``model``, ...) for the internals.
"""

from merxen.io.bead_removal.apply import apply_bead_model, correct_window
from merxen.io.bead_removal.calibrate import (
    BeadCalibration,
    BeadCalibrationError,
    calibrate_bead_model,
)
from merxen.io.bead_removal.detect import ClassifierThresholds
from merxen.io.bead_removal.params import (
    BEAD_REMOVAL_VERSION,
    DEFAULT_PARAMS,
    PsfSubtractParams,
    params_sha256,
)
from merxen.io.bead_removal.psf import PsfLibrary
from merxen.io.bead_removal.reader import MosaicPlaneReader, list_mosaic_planes

__all__ = [
    "BEAD_REMOVAL_VERSION",
    "DEFAULT_PARAMS",
    "BeadCalibration",
    "BeadCalibrationError",
    "ClassifierThresholds",
    "MosaicPlaneReader",
    "PsfLibrary",
    "PsfSubtractParams",
    "apply_bead_model",
    "calibrate_bead_model",
    "correct_window",
    "list_mosaic_planes",
    "params_sha256",
]

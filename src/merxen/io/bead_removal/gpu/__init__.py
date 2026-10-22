"""Torch/CUDA backend of the bead-removal window correction.

The default options are the production configuration. This subpackage is the only
place that imports ``torch`` and ``triton``, and it is loaded only through
:func:`merxen.io.bead_removal.backend.load_gpu_backend`, so the CPU path never
imports either library.

Public names: :func:`correct_window` (same contract as
:func:`merxen.io.bead_removal.apply.correct_window` plus a device-side max
projection), :class:`FitOpts`, :func:`triton_smoke_test`,
:class:`GpuSmokeTestError` and the :mod:`ops` helpers.
"""

from . import ops
from .fit import FitOpts
from .smoke import GpuSmokeTestError, triton_smoke_test
from .window import correct_window

__all__ = [
    "FitOpts",
    "GpuSmokeTestError",
    "correct_window",
    "ops",
    "triton_smoke_test",
]

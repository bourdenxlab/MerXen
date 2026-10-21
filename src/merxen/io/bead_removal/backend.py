"""Choice of the device that runs the bead-removal window correction.

The numpy implementation (``"cpu"``) is the specification, the test oracle and
the only path on hosts without a GPU. The GPU implementation (``"cuda"``,
PyTorch with fused Triton kernels) is an optional backend loaded from the
``gpu`` subpackage.

``torch`` and ``triton`` are imported only when a GPU is actually requested,
and only inside this module's helpers or the ``gpu`` subpackage. Importing this
module, resolving ``"cpu"`` and running the CPU path never import either.

The device is resolved once per sample, before the first window, and is never
changed mid-sample: there is no per-window mixing and no timeout fallback.
"""

from __future__ import annotations

import importlib
import logging
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

from merxen.gpu_lock import GPU_LOCK_ENV_VAR, gpu_lock_path_from_env

logger = logging.getLogger(__name__)

VALID_DEVICES = ("auto", "cpu", "cuda")
GPU_BACKEND_MODULE = "merxen.io.bead_removal.gpu"
TRITON_CACHE_ENV_VAR = "TRITON_CACHE_DIR"
# Order in which overlapping beads are updated: the CPU algorithm fits one
# source at a time in brightness order, the GPU fits colour classes in turn.
UPDATE_ORDER = {"cpu": "brightness", "cuda": "colour"}
_REMEDY = "Set merscope_bead_removal_device to cpu to use the CPU backend"

# Result of the Triton smoke test, kept for the process: the library versions
# it reports are recorded in the provenance.
_SMOKE_RECORD: dict[str, Any] | None = None


class BeadRemovalBackendError(RuntimeError):
    """The requested bead-removal device cannot be used on this host."""


def resolve_device(
    requested: str,
    *,
    fill_enabled: bool = False,
) -> Literal["cpu", "cuda"]:
    """Resolve the configured device to the one that will run the correction.

    Rules:
      * the fill has no GPU implementation, so with the fill on the device is
        ``"cpu"`` (a WARNING when something else was requested);
      * ``"cpu"`` is always ``"cpu"``;
      * ``"cuda"`` needs a CUDA device and a Triton kernel that compiles and
        gives the right numbers (a 64 px smoke test), otherwise it raises;
        without a configured GPU lock it runs unlocked, with a WARNING;
      * ``"auto"`` is ``"cuda"`` only when CUDA is available, the smoke test
        passes and ``MERXEN_GPU_LOCK_FILE`` is set, so a GPU is never used
        outside the host's lock protocol by accident; otherwise ``"cpu"``.

    Args:
        requested: ``"auto"``, ``"cpu"`` or ``"cuda"``.
        fill_enabled: Whether the residual-footprint fill is on.

    Returns:
        ``"cpu"`` or ``"cuda"``.

    Raises:
        ValueError: ``requested`` is not one of the valid devices.
        BeadRemovalBackendError: ``"cuda"`` was requested but is not usable.
    """
    if requested not in VALID_DEVICES:
        raise ValueError(
            f"unknown bead-removal device {requested!r}; expected one of "
            f"{list(VALID_DEVICES)}."
        )
    if fill_enabled:
        if requested != "cpu":
            logger.warning(
                "[BEADS] The bead-removal fill runs on the CPU backend only; "
                "using device cpu instead of %s.",
                requested,
            )
        return "cpu"
    if requested == "cpu":
        return "cpu"

    lock_file = gpu_lock_path_from_env()
    if requested == "auto":
        return _resolve_auto(lock_file)
    return _resolve_cuda(lock_file)


def _resolve_auto(lock_file: Path | None) -> Literal["cpu", "cuda"]:
    """Use the GPU only when it is available and the host lock is configured."""
    if lock_file is None:
        logger.info(
            "[BEADS] Device auto: %s is not set, using the CPU backend.",
            GPU_LOCK_ENV_VAR,
        )
        return "cpu"
    reason = _cuda_unavailable_reason()
    if reason is not None:
        logger.info("[BEADS] Device auto: %s; using the CPU backend.", reason)
        return "cpu"
    try:
        _run_smoke_test()
    except BeadRemovalBackendError as exc:
        logger.warning(
            "[BEADS] Device auto: the GPU backend is not usable (%s); using the "
            "CPU backend.",
            exc,
        )
        return "cpu"
    logger.info("[BEADS] Device auto: using the GPU backend.")
    return "cuda"


def _resolve_cuda(lock_file: Path | None) -> Literal["cuda"]:
    """Require a working GPU backend or raise with the remedy."""
    reason = _cuda_unavailable_reason()
    if reason is not None:
        raise BeadRemovalBackendError(
            f"merscope_bead_removal_device is cuda but {reason}. {_REMEDY}."
        )
    _run_smoke_test()
    if lock_file is None:
        logger.warning(
            "[BEADS] %s is not set: the GPU backend runs without the host GPU "
            "lock and may collide with other GPU processes.",
            GPU_LOCK_ENV_VAR,
        )
    return "cuda"


def _ensure_triton_cache_dir() -> None:
    """Point the Triton JIT cache at the task's temporary directory.

    Triton otherwise caches under the home directory, which is shared between
    tasks and may be read-only inside a container. The kernels then compile
    once per task (a few seconds).
    """
    if os.environ.get(TRITON_CACHE_ENV_VAR, "").strip():
        return
    cache_dir = Path(tempfile.gettempdir()) / f"merxen_triton_cache_{os.getuid()}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ[TRITON_CACHE_ENV_VAR] = str(cache_dir)


def _cuda_unavailable_reason() -> str | None:
    """Explain why CUDA cannot be used, or ``None`` when it can.

    Imports torch, so it is called only once a GPU has been requested.
    """
    try:
        import torch
    except ImportError as exc:
        return f"torch cannot be imported ({exc})"
    if not torch.cuda.is_available():
        return "torch reports no CUDA device"
    return None


def load_gpu_backend() -> ModuleType:
    """Import the GPU subpackage.

    Returns:
        The ``merxen.io.bead_removal.gpu`` module. It provides
        ``triton_smoke_test``, ``GpuSmokeTestError`` and ``correct_window``.

    Raises:
        BeadRemovalBackendError: The subpackage or one of its dependencies
            cannot be imported.
    """
    _ensure_triton_cache_dir()
    try:
        return importlib.import_module(GPU_BACKEND_MODULE)
    except ImportError as exc:
        raise BeadRemovalBackendError(
            f"the GPU backend cannot be imported ({exc}). {_REMEDY}."
        ) from exc


def _run_smoke_test() -> dict[str, Any]:
    """Run the Triton smoke test once per process and keep its record.

    The smoke test compiles the fused kernels for the 64 px specialisation
    only: the first production window compiles its own and adds a few
    seconds.

    Raises:
        BeadRemovalBackendError: The test failed (no CUDA, Triton or compiler
            problem, wrong numbers); the message carries the cause and the
            remedy.
    """
    global _SMOKE_RECORD
    if _SMOKE_RECORD is not None:
        return _SMOKE_RECORD
    gpu = load_gpu_backend()
    smoke_error = gpu.GpuSmokeTestError
    try:
        record = gpu.triton_smoke_test()
    except smoke_error as exc:
        raise BeadRemovalBackendError(
            f"the GPU backend failed its smoke test: {exc}. {_REMEDY}."
        ) from exc
    _SMOKE_RECORD = dict(record)
    return _SMOKE_RECORD


def gpu_library_versions() -> dict[str, str | None]:
    """Report the torch, triton and CUDA versions of this environment.

    Returns:
        ``torch``, ``triton`` and ``cuda`` version strings; an entry is
        ``None`` when that library is not available. Imports torch.
    """
    versions: dict[str, str | None] = {"torch": None, "triton": None, "cuda": None}
    try:
        import torch

        versions["torch"] = str(torch.__version__)
        cuda = torch.version.cuda
        versions["cuda"] = None if cuda is None else str(cuda)
    except ImportError:
        return versions
    try:
        import triton

        versions["triton"] = str(triton.__version__)
    except ImportError:
        pass
    return versions


def backend_record(device: str) -> dict[str, Any]:
    """Describe the backend that ran (or will run) the correction.

    Args:
        device: Resolved device, ``"cpu"`` or ``"cuda"``.

    Returns:
        ``device`` and ``update_order``; for ``"cuda"`` also the ``torch``,
        ``triton`` and ``cuda`` versions and the ``gpu_name`` the smoke test
        reported.

    Raises:
        ValueError: The device is not ``"cpu"`` or ``"cuda"``.
    """
    if device not in UPDATE_ORDER:
        raise ValueError(
            f"unknown bead-removal device {device!r}; expected one of "
            f"{sorted(UPDATE_ORDER)}."
        )
    record: dict[str, Any] = {"device": device, "update_order": UPDATE_ORDER[device]}
    if device == "cpu":
        return record
    source: Mapping[str, Any] = (
        _SMOKE_RECORD if _SMOKE_RECORD is not None else gpu_library_versions()
    )
    for key in ("torch", "triton", "cuda", "gpu_name"):
        record[key] = source.get(key)
    return record


def requested_backend_identity(
    requested: str,
    *,
    fill_enabled: bool = False,
) -> dict[str, Any]:
    """Describe the backend a request would use, without resolving it.

    The cache-reuse guard compares this with the recorded backend. It must not
    start a GPU or run the smoke test, so ``"auto"`` has no known device and
    gives an empty identity (nothing to compare).

    Args:
        requested: ``"auto"``, ``"cpu"`` or ``"cuda"``.
        fill_enabled: Whether the fill is on (it forces the CPU backend).

    Returns:
        ``{"device": "cpu"}``; for ``"cuda"`` also the library versions of this
        environment; ``{}`` for ``"auto"``.
    """
    if fill_enabled or requested == "cpu":
        return {"device": "cpu"}
    if requested == "cuda":
        versions = gpu_library_versions()
        return {
            "device": "cuda",
            **{key: value for key, value in versions.items() if value is not None},
        }
    return {}

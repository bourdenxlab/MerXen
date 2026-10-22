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

Resolving the device creates no CUDA context in the calling process. The cheap
checks (``torch`` imports, ``torch.cuda.is_available``) come first; the Triton
smoke test, which compiles and launches kernels, runs in a short-lived child
process while the host GPU lock is held, so a build that waits for the GPU or
does other work holds neither the GPU nor a context of its own.
"""

from __future__ import annotations

import concurrent.futures as cf
import importlib
import logging
import multiprocessing as mp
import os
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

from merxen.gpu_lock import GPU_LOCK_ENV_VAR, gpu_lock_path_from_env, gpu_process_lock

logger = logging.getLogger(__name__)

VALID_DEVICES = ("auto", "cpu", "cuda")
GPU_BACKEND_MODULE = "merxen.io.bead_removal.gpu"
TRITON_CACHE_ENV_VAR = "TRITON_CACHE_DIR"
# Order in which overlapping beads are updated: the CPU algorithm fits one
# source at a time in brightness order, the GPU fits colour classes in turn.
UPDATE_ORDER = {"cpu": "brightness", "cuda": "colour"}
DEVICE_REMEDY = "Set merscope_bead_removal_device to cpu to use the CPU backend"
# The only fill mode with a GPU implementation.
HYBRID_FILL_MODE = "hybrid"

# Result of the Triton smoke test, kept for the process: the library versions
# it reports are recorded in the provenance.
_SMOKE_RECORD: dict[str, Any] | None = None


class BeadRemovalBackendError(RuntimeError):
    """The requested bead-removal device cannot be used on this host."""


def fill_forces_cpu(fill_enabled: bool, fill_mode: str) -> bool:
    """Tell whether a fill configuration can only run on the CPU backend.

    The hybrid fill has a GPU implementation; the flat fill does not.

    Args:
        fill_enabled: Whether the residual-footprint fill is on.
        fill_mode: Mode of the fill (``"flat"`` or ``"hybrid"``); read only
            when the fill is on.

    Returns:
        True when the fill is on in a mode without a GPU implementation.
    """
    return fill_enabled and fill_mode != HYBRID_FILL_MODE


def resolve_device(
    requested: str,
    *,
    fill_enabled: bool = False,
    fill_mode: str = "flat",
) -> Literal["cpu", "cuda"]:
    """Resolve the configured device to the one that will run the correction.

    Rules:
      * the flat fill has no GPU implementation, so with the flat fill on the
        device is ``"cpu"`` (a WARNING when something else was requested); the
        hybrid fill runs on either device;
      * ``"cpu"`` is always ``"cpu"``;
      * ``"cuda"`` needs a CUDA device and a Triton kernel that compiles and
        gives the right numbers (a 64 px smoke test, run in a child process
        under the host GPU lock), otherwise it raises; without a configured GPU
        lock it runs unlocked, with a WARNING;
      * ``"auto"`` is ``"cuda"`` only when CUDA is available, the smoke test
        passes and ``MERXEN_GPU_LOCK_FILE`` is set, so a GPU is never used
        outside the host's lock protocol by accident; otherwise ``"cpu"``.

    Args:
        requested: ``"auto"``, ``"cpu"`` or ``"cuda"``.
        fill_enabled: Whether the residual-footprint fill is on.
        fill_mode: Mode of the fill (``"flat"`` or ``"hybrid"``); read only when
            the fill is on. The default is the mode without a GPU
            implementation, so a caller that omits it is kept on the CPU.

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
    if fill_forces_cpu(fill_enabled, fill_mode):
        if requested != "cpu":
            logger.warning(
                "[BEADS] The flat bead-removal fill runs on the CPU backend only; "
                "using device cpu instead of %s (the hybrid fill runs on the GPU).",
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
            f"merscope_bead_removal_device is cuda but {reason}. {DEVICE_REMEDY}."
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
    """Point the Triton JIT cache at a per-user folder in the temporary directory.

    Triton otherwise caches under the home directory, which may be read-only
    inside a container. The folder is ``merxen_triton_cache_<uid>`` under
    ``tempfile.gettempdir()``: it is shared by every process of the user that
    sees the same temporary directory (on the workstation profile ``TMPDIR`` is
    one persistent folder, so every task and run), so a kernel compiles once
    (a few seconds) and is then reused from the cache.
    """
    if os.environ.get(TRITON_CACHE_ENV_VAR, "").strip():
        return
    cache_dir = Path(tempfile.gettempdir()) / f"merxen_triton_cache_{os.getuid()}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ[TRITON_CACHE_ENV_VAR] = str(cache_dir)


def _cuda_unavailable_reason() -> str | None:
    """Explain why CUDA cannot be used, or ``None`` when it can.

    Imports torch, so it is called only once a GPU has been requested. It
    creates no CUDA context. Importing torch raises ``OSError`` when a CUDA
    library cannot be loaded, so any exception is a reason, not a crash.
    """
    try:
        import torch

        available = bool(torch.cuda.is_available())
    except Exception as exc:
        return f"torch cannot be imported ({exc})"
    if not available:
        return "torch reports no CUDA device"
    return None


def load_gpu_backend() -> ModuleType:
    """Import the GPU subpackage.

    Returns:
        The ``merxen.io.bead_removal.gpu`` module. It provides
        ``triton_smoke_test``, ``GpuSmokeTestError`` and ``correct_window``.

    Raises:
        BeadRemovalBackendError: The subpackage or one of its dependencies
            cannot be imported (an ``ImportError``, or an ``OSError`` for a CUDA
            library that cannot be loaded).
    """
    _ensure_triton_cache_dir()
    try:
        return importlib.import_module(GPU_BACKEND_MODULE)
    except Exception as exc:
        raise BeadRemovalBackendError(
            f"the GPU backend cannot be imported ({exc}). {DEVICE_REMEDY}."
        ) from exc


def _smoke_test_body() -> dict[str, Any]:
    """Load the GPU backend and run its smoke test in this process.

    Creates a CUDA context, so :func:`_run_smoke_test` runs it in a child
    process.

    Raises:
        BeadRemovalBackendError: The GPU backend cannot be loaded or the test
            failed; the message carries the cause and the remedy.
    """
    gpu = load_gpu_backend()
    smoke_error = gpu.GpuSmokeTestError
    try:
        return dict(gpu.triton_smoke_test())
    except smoke_error as exc:
        raise BeadRemovalBackendError(
            f"the GPU backend failed its smoke test: {exc}. {DEVICE_REMEDY}."
        ) from exc
    except Exception as exc:
        raise BeadRemovalBackendError(
            f"the GPU backend failed its smoke test ({type(exc).__name__}: "
            f"{exc}). {DEVICE_REMEDY}."
        ) from exc


def _execute_smoke_test(target: Callable[[], Any] | None = None) -> Any:
    """Run the smoke test (or ``target``) in a fresh child process.

    A spawned child keeps the CUDA context (a few hundred MiB) and the Triton
    compiler out of the calling process, which holds neither when the child
    exits. The child's exception is re-raised here.
    """
    with cf.ProcessPoolExecutor(
        max_workers=1, mp_context=mp.get_context("spawn")
    ) as executor:
        return executor.submit(target or _smoke_test_body).result()


def _run_smoke_test() -> dict[str, Any]:
    """Run the Triton smoke test once per process and keep its record.

    The test compiles the fused kernels for the 64 px specialisation only (the
    first production window compiles its own and adds a few seconds) and takes
    a few seconds. It uses the GPU, so it runs under the host GPU lock like
    every other GPU use, and in a child process so that this process creates no
    CUDA context before the window loop.

    Raises:
        BeadRemovalBackendError: The test failed (no CUDA, Triton or compiler
            problem, wrong numbers); the message carries the cause and the
            remedy.
    """
    global _SMOKE_RECORD
    if _SMOKE_RECORD is not None:
        return _SMOKE_RECORD
    _ensure_triton_cache_dir()
    with gpu_process_lock(gpu_lock_path_from_env()):
        record = _execute_smoke_test()
    _SMOKE_RECORD = dict(record)
    return _SMOKE_RECORD


def gpu_library_versions() -> dict[str, str | None]:
    """Report the torch, triton and CUDA versions of this environment.

    Returns:
        ``torch``, ``triton`` and ``cuda`` version strings; an entry is
        ``None`` when that library is not available (an import that raises
        any exception, such as the ``OSError`` of a missing CUDA library, counts
        as not available). Imports torch.
    """
    versions: dict[str, str | None] = {"torch": None, "triton": None, "cuda": None}
    try:
        import torch

        versions["torch"] = str(torch.__version__)
        cuda = torch.version.cuda
        versions["cuda"] = None if cuda is None else str(cuda)
    except Exception:
        return versions
    try:
        import triton

        versions["triton"] = str(triton.__version__)
    except Exception:
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
    fill_mode: str = "flat",
) -> dict[str, Any]:
    """Describe the backend a request would use, without resolving it.

    The cache-reuse guard compares this with the recorded backend. It must not
    start a GPU or run the smoke test, so ``"auto"`` has no known device and
    gives an empty identity (nothing to compare).

    Args:
        requested: ``"auto"``, ``"cpu"`` or ``"cuda"``.
        fill_enabled: Whether the fill is on.
        fill_mode: Mode of the fill; the flat fill forces the CPU backend, the
            hybrid fill does not.

    Returns:
        ``{"device": "cpu"}``; for ``"cuda"`` also the library versions of this
        environment; ``{}`` for ``"auto"``.
    """
    if fill_forces_cpu(fill_enabled, fill_mode) or requested == "cpu":
        return {"device": "cpu"}
    if requested == "cuda":
        versions = gpu_library_versions()
        return {
            "device": "cuda",
            **{key: value for key, value in versions.items() if value is not None},
        }
    return {}

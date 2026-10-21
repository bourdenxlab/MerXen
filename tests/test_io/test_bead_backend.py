"""Tests for the bead-removal device resolution (``backend.py``).

No GPU is needed: CUDA, the GPU subpackage and its smoke test are replaced by
stand-ins (the smoke test runs in this process instead of a child). The tests
that touch real imports check that the CPU path never loads torch or triton and
that the child-process mechanism works.
"""

from __future__ import annotations

import fcntl
import functools
import importlib
import logging
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from typing import Any

import pytest

from merxen.config import MerscopeBeadRemovalConfig
from merxen.io.bead_removal import backend
from merxen.io.bead_removal.backend import (
    BeadRemovalBackendError,
    backend_record,
    gpu_library_versions,
    load_gpu_backend,
    requested_backend_identity,
    resolve_device,
)

_LOGGER = "merxen.io.bead_removal.backend"
_REAL_EXECUTE_SMOKE_TEST = backend._execute_smoke_test
_SMOKE = {
    "torch": "2.14.0+cu130",
    "triton": "3.8.0",
    "cuda": "13.0",
    "gpu_name": "Test GPU",
    "seconds": 4.2,
}


class _SmokeError(RuntimeError):
    """Stand-in for the GPU package's smoke-test error."""


class _FakeGpu:
    """Stand-in for the GPU subpackage."""

    GpuSmokeTestError = _SmokeError

    def __init__(self, error: Exception | None = None) -> None:
        self.calls = 0
        self.lock_held: list[bool | None] = []
        self._error = error

    def triton_smoke_test(self) -> dict[str, Any]:
        self.calls += 1
        lock = backend.gpu_lock_path_from_env()
        self.lock_held.append(None if lock is None else _lock_is_held(lock))
        if self._error is not None:
            raise self._error
        return dict(_SMOKE)


def _lock_is_held(path: Path) -> bool:
    """Whether another open file description holds the lock (a try-lock)."""
    if not path.exists():
        return False
    descriptor = os.open(path, os.O_RDONLY)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        return False
    finally:
        os.close(descriptor)


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No lock, no Triton cache, no remembered smoke test, tmp under tmp_path.

    The smoke test runs in this process: the stand-in GPU package is not
    visible to a spawned child.
    """
    monkeypatch.delenv("MERXEN_GPU_LOCK_FILE", raising=False)
    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
    monkeypatch.setattr(backend, "_SMOKE_RECORD", None)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(
        backend,
        "_execute_smoke_test",
        lambda target=None: (target or backend._smoke_test_body)(),
    )


@pytest.fixture
def lock_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    path = tmp_path / "gpu.lock"
    monkeypatch.setenv("MERXEN_GPU_LOCK_FILE", str(path))
    return path


@pytest.fixture
def fake_gpu(monkeypatch: pytest.MonkeyPatch) -> _FakeGpu:
    """A usable CUDA device and GPU package."""
    gpu = _FakeGpu()
    monkeypatch.setattr(backend, "_cuda_unavailable_reason", lambda: None)
    monkeypatch.setattr(backend, "load_gpu_backend", lambda: gpu)
    return gpu


def _no_cuda(monkeypatch: pytest.MonkeyPatch, reason: str) -> None:
    monkeypatch.setattr(backend, "_cuda_unavailable_reason", lambda: reason)
    monkeypatch.setattr(
        backend,
        "load_gpu_backend",
        lambda: pytest.fail("the GPU package must not load without CUDA"),
    )


def _torch_must_not_be_consulted(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail() -> str | None:
        pytest.fail("CUDA must not be queried")

    monkeypatch.setattr(backend, "_cuda_unavailable_reason", _fail)
    monkeypatch.setattr(
        backend,
        "load_gpu_backend",
        lambda: pytest.fail("the GPU package must not load"),
    )


# --- the resolution matrix ---------------------------------------------------


def test_an_unknown_device_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown bead-removal device 'tpu'"):
        resolve_device("tpu")


def test_cpu_never_touches_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    _torch_must_not_be_consulted(monkeypatch)

    assert resolve_device("cpu", fill_enabled=False) == "cpu"


@pytest.mark.parametrize("requested", ["auto", "cuda"])
def test_the_fill_forces_the_cpu_with_a_warning(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    lock_env: Path,
    requested: str,
) -> None:
    _torch_must_not_be_consulted(monkeypatch)

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        assert resolve_device(requested, fill_enabled=True) == "cpu"

    assert "fill runs on the CPU backend only" in caplog.text
    assert requested in caplog.text


def test_the_fill_with_cpu_is_silent(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _torch_must_not_be_consulted(monkeypatch)

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        assert resolve_device("cpu", fill_enabled=True) == "cpu"

    assert caplog.text == ""


def test_cuda_without_a_cuda_device_raises_with_the_remedy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_cuda(monkeypatch, "torch reports no CUDA device")

    with pytest.raises(BeadRemovalBackendError) as caught:
        resolve_device("cuda", fill_enabled=False)

    message = str(caught.value)
    assert "torch reports no CUDA device" in message
    assert "merscope_bead_removal_device to cpu" in message


def test_cuda_without_torch_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # A None entry makes ``import torch`` raise ImportError.
    monkeypatch.setitem(sys.modules, "torch", None)

    with pytest.raises(BeadRemovalBackendError, match="torch cannot be imported"):
        resolve_device("cuda")


def test_cuda_smoke_failure_is_mapped_to_a_backend_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cause = _SmokeError("no C compiler found")
    gpu = _FakeGpu(error=cause)
    monkeypatch.setattr(backend, "_cuda_unavailable_reason", lambda: None)
    monkeypatch.setattr(backend, "load_gpu_backend", lambda: gpu)

    with pytest.raises(BeadRemovalBackendError, match="no C compiler found") as caught:
        resolve_device("cuda")

    assert caught.value.__cause__ is cause
    assert "merscope_bead_removal_device to cpu" in str(caught.value)


def test_a_missing_gpu_package_is_a_backend_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(backend, "_cuda_unavailable_reason", lambda: None)

    def _import(name: str) -> types.ModuleType:
        raise ModuleNotFoundError(f"No module named {name!r}", name=name)

    monkeypatch.setattr(importlib, "import_module", _import)

    with pytest.raises(BeadRemovalBackendError, match="cannot be imported"):
        resolve_device("cuda")


def test_cuda_with_a_lock_resolves_quietly(
    fake_gpu: _FakeGpu, lock_env: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        assert resolve_device("cuda") == "cuda"

    assert fake_gpu.calls == 1
    assert caplog.text == ""


def test_cuda_without_a_lock_runs_unlocked_with_a_warning(
    fake_gpu: _FakeGpu, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        assert resolve_device("cuda") == "cuda"

    assert "MERXEN_GPU_LOCK_FILE is not set" in caplog.text
    assert "without the host GPU lock" in caplog.text


def test_auto_without_a_lock_is_cpu_and_does_not_query_cuda(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _torch_must_not_be_consulted(monkeypatch)

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        assert resolve_device("auto") == "cpu"

    assert "MERXEN_GPU_LOCK_FILE is not set" in caplog.text


def test_auto_with_a_lock_but_no_cuda_is_cpu(
    monkeypatch: pytest.MonkeyPatch, lock_env: Path
) -> None:
    _no_cuda(monkeypatch, "torch reports no CUDA device")

    assert resolve_device("auto") == "cpu"


def test_auto_with_a_lock_and_cuda_is_cuda(fake_gpu: _FakeGpu, lock_env: Path) -> None:
    assert resolve_device("auto") == "cuda"
    assert fake_gpu.calls == 1


def test_auto_falls_back_to_cpu_with_a_warning_when_the_smoke_test_fails(
    monkeypatch: pytest.MonkeyPatch, lock_env: Path, caplog: pytest.LogCaptureFixture
) -> None:
    gpu = _FakeGpu(error=_SmokeError("kernel did not compile"))
    monkeypatch.setattr(backend, "_cuda_unavailable_reason", lambda: None)
    monkeypatch.setattr(backend, "load_gpu_backend", lambda: gpu)

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        assert resolve_device("auto") == "cpu"

    assert "kernel did not compile" in caplog.text
    assert "using the CPU backend" in caplog.text


def test_the_smoke_test_runs_once_per_process(
    fake_gpu: _FakeGpu, lock_env: Path
) -> None:
    resolve_device("cuda")
    resolve_device("auto")

    assert fake_gpu.calls == 1


def test_the_smoke_test_runs_only_while_the_host_lock_is_held(
    fake_gpu: _FakeGpu, lock_env: Path
) -> None:
    """The smoke test compiles and launches GPU kernels: it needs the lock."""
    assert not _lock_is_held(lock_env)

    assert resolve_device("cuda") == "cuda"

    assert fake_gpu.lock_held == [True]
    assert not _lock_is_held(lock_env), "the lock must be released afterwards"


def test_a_failing_smoke_test_releases_the_lock(
    monkeypatch: pytest.MonkeyPatch, lock_env: Path
) -> None:
    gpu = _FakeGpu(error=_SmokeError("kernel did not compile"))
    monkeypatch.setattr(backend, "_cuda_unavailable_reason", lambda: None)
    monkeypatch.setattr(backend, "load_gpu_backend", lambda: gpu)

    with pytest.raises(BeadRemovalBackendError):
        resolve_device("cuda")

    assert gpu.lock_held == [True]
    assert not _lock_is_held(lock_env)


def test_the_smoke_test_waits_for_another_holder_of_the_lock(
    fake_gpu: _FakeGpu, lock_env: Path
) -> None:
    import threading
    import time

    lock_env.touch()
    holder = os.open(lock_env, os.O_RDONLY)
    fcntl.flock(holder, fcntl.LOCK_EX)
    done = threading.Event()

    def _resolve() -> None:
        resolve_device("auto")
        done.set()

    thread = threading.Thread(target=_resolve)
    thread.start()
    try:
        time.sleep(0.5)
        assert not done.is_set() and fake_gpu.calls == 0
    finally:
        fcntl.flock(holder, fcntl.LOCK_UN)
        os.close(holder)
        thread.join(timeout=30)
    assert done.is_set() and fake_gpu.calls == 1


def test_an_unusable_cuda_never_takes_the_lock(
    monkeypatch: pytest.MonkeyPatch, lock_env: Path
) -> None:
    """The cheap checks come first: no CUDA means no lock and no smoke test."""
    _no_cuda(monkeypatch, "torch reports no CUDA device")

    assert resolve_device("auto") == "cpu"
    with pytest.raises(BeadRemovalBackendError):
        resolve_device("cuda")

    assert not lock_env.exists()


def test_the_smoke_test_runs_in_a_child_process() -> None:
    assert _REAL_EXECUTE_SMOKE_TEST(target=os.getpid) != os.getpid()


def test_an_error_of_the_smoke_test_child_is_raised_in_the_parent() -> None:
    with pytest.raises(ValueError, match="invalid literal"):
        _REAL_EXECUTE_SMOKE_TEST(target=functools.partial(int, "x"))


# --- torch that cannot be loaded -----------------------------------------------


@pytest.fixture
def broken_torch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``import torch`` raises OSError, as with a CUDA library that cannot load."""
    package = tmp_path / "stub" / "torch"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "raise OSError('libcudart.so.13: cannot open shared object file')\n"
    )
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.syspath_prepend(str(tmp_path / "stub"))
    monkeypatch.setitem(sys.modules, "triton", None)


def test_torch_that_raises_oserror_on_import_is_a_reason_not_a_crash(
    broken_torch: None, lock_env: Path
) -> None:
    assert resolve_device("auto") == "cpu"
    with pytest.raises(
        BeadRemovalBackendError, match="cannot open shared object"
    ) as caught:
        resolve_device("cuda")
    assert "merscope_bead_removal_device to cpu" in str(caught.value)


def test_the_library_versions_survive_torch_that_cannot_be_loaded(
    broken_torch: None,
) -> None:
    assert gpu_library_versions() == {"torch": None, "triton": None, "cuda": None}
    assert requested_backend_identity("cuda") == {"device": "cuda"}


def test_a_gpu_package_that_raises_oserror_is_a_backend_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _import(name: str) -> types.ModuleType:
        raise OSError("libcudart.so.13: cannot open shared object file")

    monkeypatch.setattr(importlib, "import_module", _import)

    with pytest.raises(BeadRemovalBackendError, match="cannot be imported") as caught:
        load_gpu_backend()

    assert "merscope_bead_removal_device to cpu" in str(caught.value)


def test_an_unexpected_error_of_the_smoke_test_is_a_backend_error(
    monkeypatch: pytest.MonkeyPatch, lock_env: Path
) -> None:
    """A CUDA out-of-memory error is not the package's own smoke error."""
    gpu = _FakeGpu(error=MemoryError("CUDA out of memory"))
    monkeypatch.setattr(backend, "_cuda_unavailable_reason", lambda: None)
    monkeypatch.setattr(backend, "load_gpu_backend", lambda: gpu)

    with pytest.raises(BeadRemovalBackendError, match="out of memory"):
        resolve_device("cuda")
    assert resolve_device("auto") == "cpu"


# --- the Triton cache ----------------------------------------------------------


def test_the_triton_cache_goes_under_the_temporary_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[str | None] = []

    def _load() -> _FakeGpu:
        # The variable is set before the GPU package (and so Triton) is loaded.
        seen.append(backend.os.environ.get("TRITON_CACHE_DIR"))
        return _FakeGpu()

    monkeypatch.setattr(backend, "_cuda_unavailable_reason", lambda: None)
    real_load = backend.load_gpu_backend
    monkeypatch.setattr(
        backend,
        "importlib",
        types.SimpleNamespace(import_module=lambda name: _load()),
    )

    assert real_load() is not None

    cache = Path(seen[0] or "")
    assert cache.parent == tmp_path
    assert cache.is_dir()


def test_an_existing_triton_cache_setting_is_kept(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "mine"))
    monkeypatch.setattr(
        backend,
        "importlib",
        types.SimpleNamespace(import_module=lambda name: types.ModuleType(name)),
    )

    load_gpu_backend()

    assert backend.os.environ["TRITON_CACHE_DIR"] == str(tmp_path / "mine")
    assert not (tmp_path / "mine").exists()


# --- the backend record ----------------------------------------------------------


def test_the_cpu_record_is_the_default_backend_form() -> None:
    assert backend_record("cpu") == {"device": "cpu", "update_order": "brightness"}


def test_the_cuda_record_carries_the_library_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(backend, "_SMOKE_RECORD", dict(_SMOKE))

    assert backend_record("cuda") == {
        "device": "cuda",
        "update_order": "colour",
        "torch": "2.14.0+cu130",
        "triton": "3.8.0",
        "cuda": "13.0",
        "gpu_name": "Test GPU",
    }


def test_an_unknown_record_device_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown bead-removal device"):
        backend_record("auto")


def test_library_versions_come_from_torch_and_triton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = types.ModuleType("torch")
    torch.__version__ = "2.6.0"  # type: ignore[attr-defined]
    torch.version = types.SimpleNamespace(cuda="12.4")  # type: ignore[attr-defined]
    triton = types.ModuleType("triton")
    triton.__version__ = "3.2.0"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "triton", triton)

    assert gpu_library_versions() == {
        "torch": "2.6.0",
        "triton": "3.2.0",
        "cuda": "12.4",
    }


def test_library_versions_without_torch_are_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)

    assert gpu_library_versions() == {"torch": None, "triton": None, "cuda": None}


def test_the_requested_identity_does_not_resolve_anything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _torch_must_not_be_consulted(monkeypatch)

    assert requested_backend_identity("cpu") == {"device": "cpu"}
    assert requested_backend_identity("cuda", fill_enabled=True) == {"device": "cpu"}
    # An auto request has no device before it is resolved: nothing to compare.
    assert requested_backend_identity("auto") == {}


def test_the_requested_cuda_identity_names_the_installed_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        backend,
        "gpu_library_versions",
        lambda: {"torch": "2.6.0", "triton": None, "cuda": "12.4"},
    )

    assert requested_backend_identity("cuda") == {
        "device": "cuda",
        "torch": "2.6.0",
        "cuda": "12.4",
    }


# --- the config field ---------------------------------------------------------------


def test_the_config_device_defaults_to_cpu_and_is_validated() -> None:
    assert MerscopeBeadRemovalConfig().device == "cpu"
    for device in ("auto", "cpu", "cuda"):
        assert MerscopeBeadRemovalConfig(device=device).device == device  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="device"):
        MerscopeBeadRemovalConfig(device="tpu")  # type: ignore[arg-type]


def test_the_build_json_form_with_a_device_parses() -> None:
    """The workflow adds the device key only off the default."""
    from merxen.config import SpatialDataBuildConfig

    for bead in ({"enabled": True}, {"enabled": True, "device": "cuda"}):
        config = SpatialDataBuildConfig.model_validate(
            {
                "dataset_name": "P1_MERSCOPE",
                "platform": "MERSCOPE",
                "input_path": "raw",
                "output_path": "out.zarr",
                "merscope": {"z_layers": [1, 2], "bead_removal": bead},
            }
        )
        assert config.merscope.bead_removal is not None
        assert config.merscope.bead_removal.device == bead.get("device", "cpu")


# --- CPU-only hosts and CI never import torch or triton -------------------------


def test_the_cpu_path_never_imports_torch_or_triton(tmp_path: Path) -> None:
    """Import the package, resolve the CPU device and run a window on the CPU."""
    tests_io = Path(__file__).resolve().parent
    code = f"""
import sys
sys.path.insert(0, {str(tests_io)!r})

import numpy as np

import merxen.io.bead_removal as bead_removal
from merxen.io.bead_removal import backend, driver, provenance
from merxen.io.bead_removal.backend import resolve_device
from merxen.gpu_lock import gpu_process_lock
from merxen.config import MerscopeBeadRemovalConfig
import bead_synth as bs

assert MerscopeBeadRemovalConfig().device == "cpu"
assert resolve_device("cpu") == "cpu"
assert resolve_device("auto") == "cpu"          # no lock configured
assert resolve_device("cuda", fill_enabled=True) == "cpu"
assert backend.backend_record("cpu")["device"] == "cpu"
assert backend.requested_backend_identity("auto") == {{}}
assert driver.describe_backend("cpu")["update_order"] == "brightness"
with gpu_process_lock({str(tmp_path / "gpu.lock")!r}):
    pass

calibration = bs.make_calibration()
stack = bs.make_stack(size=320, n_beads=3, seed=2, n_nuclei=2)
stacks = {{key: stack[key] for key in ("PolyT", "DAPI")}}
corrected, _model, _stats = bead_removal.correct_window(
    stacks, origin_yx=(0, 0), calibration=calibration, z_layers=[1, 2, 3]
)
assert set(corrected) == {{"PolyT", "DAPI"}}

gpu_libraries = ("torch", "triton")
loaded = sorted(n for n in sys.modules if n.split(".")[0] in gpu_libraries)
assert not loaded, loaded
print("ok")
"""
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )

    assert completed.returncode == 0, completed.stderr[-2000:]
    assert completed.stdout.strip() == "ok"

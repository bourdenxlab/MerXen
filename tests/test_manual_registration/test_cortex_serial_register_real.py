"""Registration of the real cortex pair through the landmark field (slow; never in CI).

Labellings always come from the snapshots under the evidence root (``.env``), passed
as ``labelling_payload=`` so that nothing is read from the live store's labelling; the
prepared store is read only; outputs go under ``MERXEN_TASK_TMP_DIR``. When the
snapshot's content digest is not the pinned one (the user relabelled), the tests skip.
Every measured number is logged at INFO
(``pytest -o log_cli=true --log-cli-level=INFO``).

The peak memory is the process-wide maximum, which never falls, so the runs go from
the lightest to the heaviest settings and each limit is checked right after its run.
"""

from __future__ import annotations

import json
import logging
import resource
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from merxen.manual_registration.check import execute_check
from merxen.manual_registration.config import (
    ManualRegistrationCheckConfig,
    ManualRegistrationRunConfig,
)
from merxen.manual_registration.register import run_manual_registration
from merxen.manual_registration.schema import labelling_content_sha256

pytestmark = pytest.mark.slow

logger = logging.getLogger(__name__)

ROLES = "at8=AF568;map2=AF647;dapi=DAPI"
LCM2_SNAPSHOT = "LCM2_labelling_rev2.json"
LCM2_PINNED_DIGEST = (
    "sha256:f0aa28b9e53c6cd7808dc5835ba53677a61d8b0e4ecf08aaa138d66ee8b46be5"
)
LANDMARKS_ONLY = "field_window_observations=false;field_image_check=false"
NO_WINDOWS = "field_window_observations=false"
# (wall minutes, peak GB) limits of the three settings.
LIMITS = {
    "landmarks_only": (6.0, 2.5),
    "no_windows": (12.0, 8.0),
    "default": (30.0, 10.0),
}
MAX_STORE_GB = 1.0


def _evidence(real_path: Callable[[str], Path]) -> Path:
    return real_path("MERXEN_EVIDENCE_ROOT") / "manual_registration"


def _store(real_path: Callable[[str], Path]) -> Path:
    path = (
        _evidence(real_path)
        / "runs"
        / "out"
        / "LCM2"
        / "prepared"
        / "LCM2_prepared.zarr"
    )
    if not path.exists():
        pytest.skip("the prepared store of LCM2 is not on this host")
    return path


@pytest.fixture(scope="module")
def payload(real_path: Callable[[str], Path]) -> dict[str, Any]:
    path = _evidence(real_path) / "cortex_serial" / "snapshots" / LCM2_SNAPSHOT
    if not path.is_file():
        pytest.skip(f"snapshot {LCM2_SNAPSHOT} is missing")
    document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if labelling_content_sha256(document) != LCM2_PINNED_DIGEST:
        pytest.skip(f"snapshot {LCM2_SNAPSHOT} is not the pinned labelling")
    return document


def _peak_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0**2


def _directory_gb(path: Path) -> float:
    total = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    return total / 1024.0**3


def _register(
    real_path: Callable[[str], Path],
    payload: dict[str, Any],
    folder: str,
    row_settings: str,
) -> tuple[dict[str, Path], dict[str, Any], float]:
    base = real_path("MERXEN_TASK_TMP_DIR") / "cortex_serial_register_real" / folder
    config = ManualRegistrationRunConfig(
        dataset_name="LCM2",
        preset="cortex_serial",
        prepared_store_path=_store(real_path),
        result_dir=base / "result",
        output_dir=base / "out",
        channel_roles=ROLES,
        row_settings=row_settings,
        force=True,
    )
    started = time.perf_counter()
    outputs = run_manual_registration(config, labelling_payload=payload)
    minutes = (time.perf_counter() - started) / 60.0
    summary: dict[str, Any] = json.loads(outputs["summary_json"].read_text())
    return outputs, summary, minutes


def _check_run(
    name: str,
    outputs: dict[str, Path],
    summary: dict[str, Any],
    minutes: float,
) -> None:
    limit_minutes, limit_gb = LIMITS[name]
    peak = _peak_gb()
    store_gb = _directory_gb(outputs["result_store"])
    by_role = summary["tissue_qc"]["by_role"]
    logger.info(
        "LCM2 register %s: %.1f min (limit %.0f), peak %.2f GB (limit %.1f), "
        "store %.3f GB, field LOO median %.1f / p90 %.1f um, windows %d, "
        "flow grid factor %d, max displacement %.0f um, tissue Pearson final "
        "(at8 %.3f, map2 %.3f, dapi %.3f)",
        name,
        minutes,
        limit_minutes,
        peak,
        limit_gb,
        store_gb,
        summary["field"]["loo_um"]["median"],
        summary["field"]["loo_um"]["p90"],
        summary["field"]["n_windows"],
        summary["field"]["flow"]["grid_factor"],
        summary["field"]["flow"]["max_displacement_um"],
        by_role["at8"]["pearson"]["final"],
        by_role["map2"]["pearson"]["final"],
        by_role["dapi"]["pearson"]["final"],
    )
    assert minutes <= limit_minutes
    assert peak <= limit_gb
    assert store_gb <= MAX_STORE_GB
    assert summary["field"]["materialize"]["mode"] == "preview"
    for role in ("at8", "map2"):
        pearson = by_role[role]["pearson"]
        assert pearson["final"] >= pearson["landmark"], role


def test_landmarks_only_register_matches_the_landmarks_only_check(
    real_path: Callable[[str], Path], payload: dict[str, Any]
) -> None:
    outputs, summary, minutes = _register(
        real_path, payload, "landmarks_only", LANDMARKS_ONLY
    )
    _check_run("landmarks_only", outputs, summary, minutes)
    check = execute_check(
        ManualRegistrationCheckConfig(
            dataset_name="LCM2",
            prepared_store_path=_store(real_path),
            output_dir=real_path("MERXEN_TASK_TMP_DIR")
            / "cortex_serial_register_real"
            / "landmarks_only_check",
            channel_roles=ROLES,
            image_check=False,
            windows=False,
            write_attr=False,
        ),
        labelling_payload=payload,
    )
    assert summary["field"]["model_digest"] == check.report.field.digest
    register_loo = {
        record["id"]: record["residual_um"]["loo"]
        for record in summary["landmarks"]
        if record["used_in_fit"]
    }
    check_loo = {
        int(pair_id): record["loo_um"]
        for pair_id, record in check.payload["pairs"].items()
        if record["used"]
    }
    assert register_loo.keys() == check_loo.keys()
    ids = sorted(register_loo)
    np.testing.assert_allclose(
        [register_loo[i] for i in ids], [check_loo[i] for i in ids], rtol=1e-6
    )


def test_register_without_windows_stays_within_its_limits(
    real_path: Callable[[str], Path], payload: dict[str, Any]
) -> None:
    outputs, summary, minutes = _register(real_path, payload, "no_windows", NO_WINDOWS)
    _check_run("no_windows", outputs, summary, minutes)


def test_default_register_stays_within_its_limits(
    real_path: Callable[[str], Path], payload: dict[str, Any]
) -> None:
    outputs, summary, minutes = _register(real_path, payload, "default", "")
    _check_run("default", outputs, summary, minutes)

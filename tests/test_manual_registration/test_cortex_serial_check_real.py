"""The landmark check on the real cortex pairs (slow; read-only; never in CI).

Labellings always come from the snapshots under the evidence root (``.env``), passed
as ``labelling_payload=`` so that no attribute is written; the prepared stores are
read only; outputs go under ``MERXEN_TASK_TMP_DIR``. When a snapshot's content digest
is not the pinned one (the user relabelled), the test skips. Every measured number is
logged at INFO (``pytest -o log_cli=true --log-cli-level=INFO``).
"""

from __future__ import annotations

import copy
import json
import logging
import resource
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from merxen.manual_registration.check import CheckRun, execute_check
from merxen.manual_registration.config import ManualRegistrationCheckConfig
from merxen.manual_registration.landmark_field import (
    LandmarkField,
    fit_landmark_field,
)
from merxen.manual_registration.schema import labelling_content_sha256

pytestmark = pytest.mark.slow

logger = logging.getLogger(__name__)

ROLES = "at8=AF568;map2=AF647;dapi=DAPI"
LCM2_SNAPSHOT = "LCM2_labelling_rev2.json"
LCM2_PINNED_DIGEST = (
    "sha256:f0aa28b9e53c6cd7808dc5835ba53677a61d8b0e4ecf08aaa138d66ee8b46be5"
)
LCM1_SNAPSHOT = "LCM1_labelling_rev1.json"
LCM1_PINNED_DIGEST = (
    "sha256:f47ba4e621c85dad5e9110e19a1128c454436bfe14d565178b4f4e9da6d50d52"
)
RIGID_FLAGGED = (13, 16, 51, 57, 58, 59, 60, 61, 62, 64, 65, 66, 69)
IMAGE_CONFIRMED_EXPECTED = (1, 2, 7, 8, 9, 14, 23)
GROSS_DISABLED = (11, 12, 18, 20)
MAX_MINUTES = 25.0
MAX_PEAK_GB = 10.0
HELD_OUT_WINDOWS = 12
WINDOW_EXCLUSION_UM = 1000.0
BLOCK_RADIUS_UM = 1500.0
N_BLOCKS = 8


def _evidence(real_path: Callable[[str], Path]) -> Path:
    return real_path("MERXEN_EVIDENCE_ROOT") / "manual_registration"


def _store(real_path: Callable[[str], Path], name: str) -> Path:
    path = (
        _evidence(real_path)
        / "runs"
        / "out"
        / name
        / "prepared"
        / f"{name}_prepared.zarr"
    )
    if not path.exists():
        pytest.skip(f"the prepared store of {name} is not on this host")
    return path


def _snapshot(
    real_path: Callable[[str], Path], name: str, pinned: str
) -> dict[str, Any]:
    path = _evidence(real_path) / "cortex_serial" / "snapshots" / name
    if not path.is_file():
        pytest.skip(f"snapshot {name} is missing")
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if labelling_content_sha256(payload) != pinned:
        pytest.skip(f"snapshot {name} is not the pinned labelling")
    return payload


def _config(
    real_path: Callable[[str], Path],
    dataset: str,
    folder: str,
    **overrides: Any,
) -> ManualRegistrationCheckConfig:
    out = real_path("MERXEN_TASK_TMP_DIR") / "cortex_serial_check_real" / folder
    values: dict[str, Any] = {
        "dataset_name": dataset,
        "prepared_store_path": _store(real_path, dataset),
        "output_dir": out,
        "channel_roles": ROLES,
        "write_attr": False,  # a real-data check never writes the store
    }
    values.update(overrides)
    return ManualRegistrationCheckConfig(**values)


def _peak_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0**2


def _pair(run: CheckRun, pair_id: int) -> dict[str, Any]:
    record: dict[str, Any] = run.payload["pairs"][str(pair_id)]
    return record


# ---------------------------------------------------------------------------
# A3: how much the exemption of strong confirmations changes the field
# ---------------------------------------------------------------------------


def _reference_residual_um(
    field: LandmarkField, reference: np.ndarray, moving: np.ndarray
) -> np.ndarray:
    """Residual magnitude in reference micrometres (the inverse trend Jacobian)."""
    residual = moving - field.mean(reference)
    jacobian = field.jacobian(reference)
    vectors = np.linalg.solve(jacobian, residual[:, :, None])[:, :, 0]
    return np.asarray(np.linalg.norm(vectors, axis=1))


def _held_out_windows_um(field: LandmarkField) -> np.ndarray:
    """Error at windows held out with everything within 1000 um of them."""
    observations = field.observations
    windows = np.flatnonzero([c == "window" for c in observations.obs_class])
    if not len(windows):
        return np.zeros(0)
    # Farthest-point sample of the windows, so the same windows serve both runs.
    positions = observations.reference_xy_um
    chosen = [int(windows[0])]
    distance = np.linalg.norm(positions[windows] - positions[chosen[0]], axis=1)
    while len(chosen) < min(HELD_OUT_WINDOWS, len(windows)):
        nxt = int(windows[int(np.argmax(distance))])
        chosen.append(nxt)
        distance = np.minimum(
            distance, np.linalg.norm(positions[windows] - positions[nxt], axis=1)
        )
    errors = []
    for index in chosen:
        near = (
            np.linalg.norm(positions - positions[index], axis=1) <= WINDOW_EXCLUSION_UM
        )
        refit = fit_landmark_field(observations.subset(~near), field.options)
        errors.append(
            float(
                _reference_residual_um(
                    refit,
                    positions[index : index + 1],
                    observations.moving_xy_um[index : index + 1],
                )[0]
            )
        )
    return np.asarray(errors)


def _block_held_out_clicks_um(field: LandmarkField) -> np.ndarray:
    """Error at clicks held out in spatial blocks (clicks and windows of the block)."""
    observations = field.observations
    clicks = np.flatnonzero(
        [c in ("click", "confirmed") for c in observations.obs_class]
    )
    positions = observations.reference_xy_um
    chosen = [int(clicks[0])]
    distance = np.linalg.norm(positions[clicks] - positions[chosen[0]], axis=1)
    while len(chosen) < min(N_BLOCKS, len(clicks)):
        nxt = int(clicks[int(np.argmax(distance))])
        chosen.append(nxt)
        distance = np.minimum(
            distance, np.linalg.norm(positions[clicks] - positions[nxt], axis=1)
        )
    errors: list[float] = []
    for centre in chosen:
        block = np.linalg.norm(positions - positions[centre], axis=1) <= BLOCK_RADIUS_UM
        refit = fit_landmark_field(observations.subset(~block), field.options)
        held = np.flatnonzero(block & np.isin(np.arange(observations.n), clicks))
        errors.extend(
            _reference_residual_um(
                refit, positions[held], observations.moving_xy_um[held]
            ).tolist()
        )
    return np.asarray(errors)


@pytest.fixture(scope="module")
def lcm2_payload(real_path: Callable[[str], Path]) -> dict[str, Any]:
    return _snapshot(real_path, LCM2_SNAPSHOT, LCM2_PINNED_DIGEST)


@pytest.fixture(scope="module")
def lcm2(real_path: Callable[[str], Path], lcm2_payload: dict[str, Any]) -> CheckRun:
    """The default check (image check and windows on, strong confirmations trusted)."""
    return execute_check(
        _config(real_path, "LCM2", "lcm2_default"), labelling_payload=lcm2_payload
    )


def test_lcm2_check_meets_the_m1_acceptance(lcm2: CheckRun) -> None:
    field = lcm2.payload["field"]
    report = lcm2.report
    logger.info("LCM2 field: %s", json.dumps(field, sort_keys=True))
    logger.info("LCM2 timings: %s peak %.1f GB", report.timings_s, _peak_gb())

    # 1. The clicks alone.
    clicks_only = field["loo_um_clicks_only"]
    assert clicks_only is not None
    assert clicks_only["median"] <= 45.0
    assert clicks_only["p90"] <= 105.0
    # 2. With the windows.
    assert field["n_windows"] > 0
    assert field["loo_um"]["median"] <= 38.0
    # 3. The misclicks the clicks alone expose.
    for pair_id in (13, 71):
        pair = _pair(lcm2, pair_id)
        assert pair["clicks_only"]["p_value"] < 0.01, pair_id
        assert pair["class"] in {"likely_misclick", "check"}, (pair_id, pair["class"])
    assert _pair(lcm2, 71)["clicks_only"]["weight"] < 0.15
    # 4. Pairs the image confirms.
    for pair_id in (2, 3):
        assert _pair(lcm2, pair_id)["class"] != "likely_misclick", pair_id
    # 5. The rigid fit's flags.
    flagged = sorted(
        int(key) for key, pair in lcm2.payload["pairs"].items() if pair["rigid_outlier"]
    )
    assert flagged == list(RIGID_FLAGGED)
    fine = [
        pair_id
        for pair_id in RIGID_FLAGGED
        if _pair(lcm2, pair_id)["class"] in {"ok", "kept_confirmed"}
    ]
    logger.info("rigid-flagged pairs that are ok or kept: %s", fine)
    assert len(fine) >= 9
    # 7. The image check.
    verdicts = {key: pair["verdict"] for key, pair in lcm2.payload["pairs"].items()}
    confirmed = [i for i in IMAGE_CONFIRMED_EXPECTED if verdicts[str(i)] == "confirmed"]
    logger.info("confirmed of %s: %s", IMAGE_CONFIRMED_EXPECTED, confirmed)
    assert len(confirmed) >= 4
    offsets = [
        pair["image_offset_um"]
        for pair in lcm2.payload["pairs"].values()
        if pair["verdict"] == "confirmed" and pair["image_offset_um"] is not None
    ]
    assert float(np.median(offsets)) <= 10.0
    enabled = sum(1 for pair in lcm2.payload["pairs"].values() if pair["enabled"])
    decisive = sum(
        1
        for pair in lcm2.payload["pairs"].values()
        if pair["enabled"] and pair["verdict"] in {"confirmed", "contradicted", "ridge"}
    )
    logger.info("decisive verdicts: %d of %d enabled pairs", decisive, enabled)
    assert decisive >= 0.2 * enabled
    # Cost.
    assert report.timings_s["total"] <= MAX_MINUTES * 60.0
    assert _peak_gb() <= MAX_PEAK_GB
    for name, value in sorted(lcm2.payload["field"]["class_counts"].items()):
        logger.info("class %s: %d", name, value)


def test_lcm2_trusting_confirmed_clicks_helps_the_strong_ones(
    real_path: Callable[[str], Path], lcm2: CheckRun, lcm2_payload: dict[str, Any]
) -> None:
    off = execute_check(
        _config(
            real_path,
            "LCM2",
            "lcm2_trust_off",
            row_settings="field_trust_confirmed_clicks=false",
        ),
        labelling_payload=lcm2_payload,
    )
    summary: dict[str, dict[str, float]] = {}
    for label, run in (("on", lcm2), ("off", off)):
        report = run.report
        observations = report.field.observations
        row_of = {
            int(i): row
            for row, i in enumerate(observations.obs_id)
            if observations.obs_class[row] != "window"
        }
        in_sample = _reference_residual_um(
            report.field, observations.reference_xy_um, observations.moving_xy_um
        )
        strong_ids = [
            int(report.pair_ids[i])
            for i, strength in enumerate(report.confirm_strength)
            if strength == "strong" and int(report.pair_ids[i]) in row_of
        ]
        assert strong_ids, "no strong confirmation: nothing to compare"
        residuals = [float(in_sample[row_of[i]]) for i in strong_ids]
        named = {
            i: round(float(in_sample[row_of[i]]), 1)
            for i in (62, 67, 68)
            if i in row_of
        }
        window_errors = _held_out_windows_um(report.field)
        block_errors = _block_held_out_clicks_um(report.field)
        summary[label] = {
            "strong_in_sample_median": float(np.median(residuals)),
            "window_holdout_median": (
                float(np.median(window_errors)) if len(window_errors) else 0.0
            ),
            "block_holdout_median": float(np.median(block_errors)),
        }
        logger.info(
            "A3 trust %s: %d strong clicks, in-sample residual median %.1f um "
            "(62/67/68: %s), held-out windows (%d, %g um exclusion) median %.1f um, "
            "block hold-out median %.1f um",
            label,
            len(strong_ids),
            summary[label]["strong_in_sample_median"],
            named,
            len(window_errors),
            WINDOW_EXCLUSION_UM,
            summary[label]["window_holdout_median"],
            summary[label]["block_holdout_median"],
        )
    assert summary["on"]["strong_in_sample_median"] <= 30.0
    assert (
        summary["on"]["window_holdout_median"]
        <= summary["off"]["window_holdout_median"] + 3.0
    )


def test_lcm2_with_every_pair_enabled_exposes_the_gross_pairs(
    real_path: Callable[[str], Path], lcm2_payload: dict[str, Any]
) -> None:
    payload = copy.deepcopy(lcm2_payload)
    for pair in payload["landmarks"]["pairs"]:
        pair["enabled"] = True
    run = execute_check(
        _config(
            real_path,
            "LCM2",
            "lcm2_all_pairs",
            channel_roles="",
            image_check=False,
            windows=False,
        ),
        labelling_payload=payload,
    )
    weights = {
        pair_id: _pair(run, pair_id)["weight"] for pair_id in (*GROSS_DISABLED, 17)
    }
    logger.info("all 71 pairs enabled, weights: %s", weights)
    for pair_id in GROSS_DISABLED:
        assert weights[pair_id] < 0.05, pair_id
    assert weights[17] > 0.5


def test_lcm2_uploaded_file_and_snapshot_give_the_same_check(
    real_path: Callable[[str], Path], lcm2_payload: dict[str, Any]
) -> None:
    upload = _evidence(real_path) / "incoming" / "LCM2_zarr.json"
    if not upload.is_file():
        pytest.skip("the uploaded LCM2 file is not on this host")
    from merxen.manual_registration.check import read_labelling_file

    uploaded, _store_attr = read_labelling_file(upload)
    if labelling_content_sha256(uploaded) != LCM2_PINNED_DIGEST:
        pytest.skip("the uploaded file does not hold the pinned revision-2 labelling")
    kwargs: dict[str, Any] = {
        "channel_roles": "",
        "image_check": False,
        "windows": False,
    }
    from_file = execute_check(
        _config(real_path, "LCM2", "lcm2_file", labelling_path=upload, **kwargs)
    )
    from_snapshot = execute_check(
        _config(real_path, "LCM2", "lcm2_snapshot_lm", **kwargs),
        labelling_payload=lcm2_payload,
    )
    for key in ("pairs", "field", "suggestions"):
        assert from_file.payload[key] == from_snapshot.payload[key], key
    assert from_file.payload["labelling_source"] == "file"
    assert from_file.attr_written is False


def test_lcm1_check_on_the_first_revision(
    real_path: Callable[[str], Path],
) -> None:
    payload = _snapshot(real_path, LCM1_SNAPSHOT, LCM1_PINNED_DIGEST)
    run = execute_check(
        _config(real_path, "LCM1", "lcm1_default"), labelling_payload=payload
    )
    field = run.payload["field"]
    logger.info("LCM1 field: %s", json.dumps(field, sort_keys=True))
    logger.info("LCM1 timings: %s", run.report.timings_s)
    assert field["loo_um"]["median"] <= 25.0
    assert _pair(run, 16)["class"] != "ok"

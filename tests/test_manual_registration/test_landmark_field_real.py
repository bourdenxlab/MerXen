"""Robust landmark field on the real cortex labellings (slow; never in CI).

Reads the labelling snapshots under the evidence root (``.env``), never the live
stores, and writes nothing. When a snapshot's content digest is not the pinned one
(the user relabelled) only a relative check runs.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from merxen.manual_registration import landmark_field as lf
from merxen.manual_registration.landmark_fit import leave_one_out_residuals
from merxen.manual_registration.schema import labelling_content_sha256

pytestmark = pytest.mark.slow

logger = logging.getLogger(__name__)

LCM2_SNAPSHOT = "LCM2_labelling_rev2.json"
LCM2_PINNED_DIGEST = (
    "sha256:f0aa28b9e53c6cd7808dc5835ba53677a61d8b0e4ecf08aaa138d66ee8b46be5"
)
LCM1_SNAPSHOT = "LCM1_labelling_rev1.json"
LCM1_PINNED_DIGEST = (
    "sha256:f47ba4e621c85dad5e9110e19a1128c454436bfe14d565178b4f4e9da6d50d52"
)
SIMILARITY_FLAGGED = (13, 16, 51, 57, 58, 59, 60, 61, 62, 64, 65, 66, 69)
GROSS_DISABLED = (11, 12, 18, 20)


def _load(real_path: Callable[[str], Path], name: str) -> dict[str, Any]:
    path = (
        real_path("MERXEN_EVIDENCE_ROOT")
        / "manual_registration"
        / "cortex_serial"
        / "snapshots"
        / name
    )
    if not path.is_file():
        pytest.skip(f"snapshot {name} is missing")
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def _clicks(
    payload: dict[str, Any], *, include_disabled: bool = False
) -> lf.FieldObservations:
    reference_scale = payload["images"]["reference"]["pixel_size_um"]
    moving_scale = payload["images"]["moving"]["pixel_size_um"]
    pairs = [
        pair
        for pair in payload["landmarks"]["pairs"]
        if include_disabled or pair["enabled"]
    ]
    return lf.FieldObservations.clicks(
        np.array([pair["reference_xy_px"] for pair in pairs]) * reference_scale,
        np.array([pair["moving_xy_px"] for pair in pairs]) * moving_scale,
        [pair["id"] for pair in pairs],
    )


def _row(observations: lf.FieldObservations, pair_id: int) -> int:
    return int(np.flatnonzero(observations.obs_id == pair_id)[0])


def _similarity_loo_median_um(observations: lf.FieldObservations) -> float:
    medians = [
        float(
            np.median(
                leave_one_out_residuals(
                    observations.moving_xy_um,
                    observations.reference_xy_um,
                    reflection=reflection,
                )
            )
        )
        for reflection in (False, True)
    ]
    return min(medians)


def test_lcm2_clicks_only_field_on_the_real_labelling(
    real_path: Callable[[str], Path],
) -> None:
    payload = _load(real_path, LCM2_SNAPSHOT)
    observations = _clicks(payload)
    started = time.perf_counter()
    field = lf.fit_landmark_field(observations)
    seconds = time.perf_counter() - started
    loo = field.loo()
    magnitude = loo.magnitude_reference_um
    median, p90 = float(np.median(magnitude)), float(np.percentile(magnitude, 90))
    rigid_median = _similarity_loo_median_um(observations)
    logger.info(
        "LCM2 clicks only: n=%d LOO median %.1f p90 %.1f um (similarity median %.1f), "
        "length %.0f um, signal %.0f um, click noise %.1f um, fit %.1f s",
        observations.n,
        median,
        p90,
        rigid_median,
        field.hyperparameters.long_length_um,
        field.hyperparameters.long_signal_um,
        field.hyperparameters.noise_um["click"],
        seconds,
    )
    if labelling_content_sha256(payload) != LCM2_PINNED_DIGEST:
        assert median <= 0.6 * rigid_median
        return

    assert median <= 45.0
    assert p90 <= 105.0
    assert seconds <= 30.0
    p_value = {
        int(i): float(p) for i, p in zip(observations.obs_id, loo.p_value, strict=True)
    }
    for pair_id in (13, 71):
        assert p_value[pair_id] < 0.01, pair_id
    row71 = _row(observations, 71)
    assert field.weights[row71] < 0.15
    assert p_value[71] < 1e-6
    unflagged = [pair for pair in SIMILARITY_FLAGGED if p_value[pair] >= 0.01]
    logger.info("similarity-flagged pairs not flagged by the field: %s", unflagged)
    assert len(unflagged) >= 9

    everything = _clicks(payload, include_disabled=True)
    disabled = [
        pair["id"] for pair in payload["landmarks"]["pairs"] if not pair["enabled"]
    ]
    for pair_id in GROSS_DISABLED:
        assert pair_id in disabled
    pairs = {pair["id"]: pair for pair in payload["landmarks"]["pairs"]}
    reference_scale = payload["images"]["reference"]["pixel_size_um"]
    moving_scale = payload["images"]["moving"]["pixel_size_um"]
    for pair_id in (*GROSS_DISABLED, 17):
        pair = pairs[pair_id]
        residual = field.predict_residual(
            np.array(pair["reference_xy_px"]) * reference_scale,
            np.array(pair["moving_xy_px"]) * moving_scale,
        )
        logger.info(
            "disabled pair %d: |residual| %.0f um, p %.3g",
            pair_id,
            float(residual.magnitude_reference_um[0]),
            float(residual.p_value[0]),
        )
        if pair_id == 17:
            assert residual.p_value[0] > 0.01
        else:
            assert residual.magnitude_reference_um[0] > 900.0
            assert residual.p_value[0] < 1e-6

    all_pairs = lf.fit_landmark_field(everything)
    for pair_id in GROSS_DISABLED:
        assert all_pairs.weights[_row(everything, pair_id)] < 0.05, pair_id
    assert all_pairs.weights[_row(everything, 17)] > 0.5
    logger.info(
        "all %d pairs: weights 11 %.3f 12 %.3f 18 %.3f 20 %.3f, 17 %.3f",
        everything.n,
        *(all_pairs.weights[_row(everything, i)] for i in (11, 12, 18, 20, 17)),
    )


def test_lcm1_field_on_the_derived_landmarks(
    real_path: Callable[[str], Path],
) -> None:
    payload = _load(real_path, LCM1_SNAPSHOT)
    observations = _clicks(payload)
    field = lf.fit_landmark_field(observations)
    loo = field.loo()
    median = float(np.median(loo.magnitude_reference_um))
    rigid_median = _similarity_loo_median_um(observations)
    logger.info(
        "LCM1: n=%d LOO median %.1f um (similarity median %.1f), length %.0f um",
        observations.n,
        median,
        rigid_median,
        field.hyperparameters.long_length_um,
    )
    if labelling_content_sha256(payload) != LCM1_PINNED_DIGEST:
        assert median <= 0.6 * rigid_median
        return

    assert median <= 25.0
    if 16 in observations.obs_id:
        p_value = float(loo.p_value[_row(observations, 16)])
    else:
        pair = next(p for p in payload["landmarks"]["pairs"] if p["id"] == 16)
        p_value = float(
            field.predict_residual(
                np.array(pair["reference_xy_px"])
                * payload["images"]["reference"]["pixel_size_um"],
                np.array(pair["moving_xy_px"])
                * payload["images"]["moving"]["pixel_size_um"],
            ).p_value[0]
        )
    logger.info("LCM1 pair 16 p-value %.3g", p_value)
    assert p_value < 1e-4

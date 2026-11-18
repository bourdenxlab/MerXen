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

from ._acceptance import Acceptance

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
    acceptance = Acceptance(logger, "LCM2 clicks-only field")
    if labelling_content_sha256(payload) != LCM2_PINNED_DIGEST:
        acceptance.at_most(
            "LOO median relative to the similarity fit's", median / rigid_median, 0.6
        )
        acceptance.finish()
        return

    # Every quantity is measured and logged before the first assertion.
    p_value = {
        int(i): float(p) for i, p in zip(observations.obs_id, loo.p_value, strict=True)
    }
    row71 = _row(observations, 71)
    unflagged = [pair for pair in SIMILARITY_FLAGGED if p_value[pair] >= 0.01]
    logger.info("similarity-flagged pairs not flagged by the field: %s", unflagged)

    everything = _clicks(payload, include_disabled=True)
    disabled = [
        pair["id"] for pair in payload["landmarks"]["pairs"] if not pair["enabled"]
    ]
    pairs = {pair["id"]: pair for pair in payload["landmarks"]["pairs"]}
    reference_scale = payload["images"]["reference"]["pixel_size_um"]
    moving_scale = payload["images"]["moving"]["pixel_size_um"]
    residuals: dict[int, tuple[float, float]] = {}
    for pair_id in (*GROSS_DISABLED, 17):
        pair = pairs[pair_id]
        residual = field.predict_residual(
            np.array(pair["reference_xy_px"]) * reference_scale,
            np.array(pair["moving_xy_px"]) * moving_scale,
        )
        residuals[pair_id] = (
            float(residual.magnitude_reference_um[0]),
            float(residual.p_value[0]),
        )
        logger.info(
            "disabled pair %d: |residual| %.0f um, p %.3g",
            pair_id,
            *residuals[pair_id],
        )
    all_pairs = lf.fit_landmark_field(everything)
    all_weights = {
        pair_id: float(all_pairs.weights[_row(everything, pair_id)])
        for pair_id in (11, 12, 18, 20, 17)
    }
    logger.info(
        "all %d pairs: weights 11 %.3f 12 %.3f 18 %.3f 20 %.3f, 17 %.3f",
        everything.n,
        *all_weights.values(),
    )

    acceptance.at_most("LOO median", median, 45.0, " um")
    acceptance.at_most("LOO p90", p90, 105.0, " um")
    acceptance.at_most("fit time", seconds, 30.0, " s")
    for pair_id in (13, 71):
        acceptance.at_most(f"pair {pair_id} p", p_value[pair_id], 0.01)
    acceptance.at_most("pair 71 weight", float(field.weights[row71]), 0.15)
    acceptance.at_most("pair 71 p", p_value[71], 1e-6)
    acceptance.at_least("similarity-flagged pairs not flagged", len(unflagged), 9)
    for pair_id in GROSS_DISABLED:
        acceptance.holds(
            f"pair {pair_id} disabled in the labelling", pair_id in disabled
        )
        acceptance.at_least(
            f"pair {pair_id} |residual|", residuals[pair_id][0], 900.0, " um"
        )
        acceptance.at_most(f"pair {pair_id} residual p", residuals[pair_id][1], 1e-6)
    acceptance.at_least("pair 17 residual p", residuals[17][1], 0.01)
    for pair_id in GROSS_DISABLED:
        acceptance.at_most(f"all-pairs weight of {pair_id}", all_weights[pair_id], 0.05)
    acceptance.at_least("all-pairs weight of 17", all_weights[17], 0.5)
    acceptance.finish()


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
    acceptance = Acceptance(logger, "LCM1 field")
    if labelling_content_sha256(payload) != LCM1_PINNED_DIGEST:
        acceptance.at_most(
            "LOO median relative to the similarity fit's", median / rigid_median, 0.6
        )
        acceptance.finish()
        return

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
    acceptance.at_most("LOO median", median, 25.0, " um")
    acceptance.at_most("pair 16 p", p_value, 1e-4)
    acceptance.finish()

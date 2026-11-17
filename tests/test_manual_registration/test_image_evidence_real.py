"""Image evidence on a real serial-section cortex pair (slow; read-only).

Reads the prepared store of the LCM2 pair and the pinned labelling snapshot, never
writes into either, and checks the click verdicts and the window observations
against the numbers of the planning prototype. Every measured number is logged at
INFO (``pytest -o log_cli=true --log-cli-level=INFO``).
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

from merxen.manual_registration import image_evidence as ie
from merxen.manual_registration.geometry import snap_crop
from merxen.manual_registration.schema import (
    STORE_ATTR,
    labelling_content_sha256,
    parse_labelling_attrs,
    parse_store_attrs,
    read_root_attrs,
)

logger = logging.getLogger(__name__)

# Digest of the labelling content of the LCM2 revision-2 snapshot (66 enabled pairs).
PINNED_LABELLING_DIGEST = (
    "sha256:f0aa28b9e53c6cd7808dc5835ba53677a61d8b0e4ecf08aaa138d66ee8b46be5"
)
ROLES_TEXT = "at8=AF568;map2=AF647;dapi=DAPI"
CONFIRMED_EXPECTED = (1, 2, 7, 8, 9, 14, 23)
_TRUST_SCALE_UM = 10000.0


def _quadratic_features(xy_um: np.ndarray) -> np.ndarray:
    x = xy_um[:, 0] / _TRUST_SCALE_UM
    y = xy_um[:, 1] / _TRUST_SCALE_UM
    return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=1)


def _fit_quadratic(
    reference_um: np.ndarray, moving_um: np.ndarray
) -> Callable[[np.ndarray], np.ndarray]:
    """Least-squares quadratic map, reference um to moving um (independent of WP1)."""
    coefficients, *_ = np.linalg.lstsq(
        _quadratic_features(reference_um), moving_um, rcond=None
    )

    def pull(points_um: np.ndarray) -> np.ndarray:
        return np.asarray(
            _quadratic_features(np.asarray(points_um, float)) @ coefficients
        )

    return pull


def _peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0**2


@pytest.mark.slow
def test_lcm2_image_evidence_matches_the_planning_prototype(
    real_path: Callable[[str], Path], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    evidence = real_path("MERXEN_EVIDENCE_ROOT")
    store_path = (
        evidence
        / "manual_registration"
        / "runs"
        / "out"
        / "LCM2"
        / "prepared"
        / "LCM2_prepared.zarr"
    )
    snapshot_path = (
        evidence
        / "manual_registration"
        / "cortex_serial"
        / "snapshots"
        / "LCM2_labelling_rev2.json"
    )
    if not store_path.exists() or not snapshot_path.exists():
        pytest.skip("the prepared store or the labelling snapshot is not on this host")
    payload: dict[str, Any] = json.loads(snapshot_path.read_text())
    if labelling_content_sha256(payload) != PINNED_LABELLING_DIGEST:
        pytest.skip("the snapshot is not the pinned revision-2 labelling")

    labelling = parse_labelling_attrs(payload)
    store = parse_store_attrs(read_root_attrs(store_path)[STORE_ATTR])
    crops = {
        key: snap_crop(getattr(labelling.crops, key), store.images[key].shape_yx)
        for key in ("reference", "moving")
    }
    roles = ie.resolve_channel_roles(
        ROLES_TEXT,
        reference_channels=store.images["reference"].canonical_channel_names
        or store.images["reference"].channel_names,
        moving_channels=store.images["moving"].canonical_channel_names
        or store.images["moving"].channel_names,
    )
    pairs = labelling.enabled_pairs()
    pixel_reference = store.images["reference"].pixel_size_um
    pixel_moving = store.images["moving"].pixel_size_um
    ids = [pair.id for pair in pairs]
    reference_um = np.array([pair.reference_xy_px for pair in pairs]) * pixel_reference
    moving_um = np.array([pair.moving_xy_px for pair in pairs]) * pixel_moving
    pull = _fit_quadratic(reference_um, moving_um)
    # Leave-one-out predictions of the same quadratic fit.
    predicted = np.empty_like(moving_um)
    for index in range(len(pairs)):
        rest = np.arange(len(pairs)) != index
        predicted[index] = _fit_quadratic(reference_um[rest], moving_um[rest])(
            reference_um[index : index + 1]
        )[0]

    timings: dict[str, float] = {}
    started = time.perf_counter()

    def lap(name: str, since: float) -> float:
        timings[name] = time.perf_counter() - since
        logger.info(
            "step %s: %.1f s (peak RSS %.2f GB)", name, timings[name], _peak_rss_gb()
        )
        return time.perf_counter()

    t = time.perf_counter()
    images = ie.load_role_images(store, roles, crops=crops)
    reference = ie.compute_representation(images["reference"])
    t = lap("load and reference representation", t)
    warped_images = ie.warp_role_images(
        images["moving"], reference=images["reference"], pull_um=pull
    )
    t = lap("warp", t)
    moving = ie.compute_representation(warped_images)
    t = lap("moving representation", t)

    checks = ie.check_clicks(
        reference,
        moving,
        pair_ids=ids,
        reference_xy_um=reference_um,
        moving_xy_um=moving_um,
        predicted_moving_xy_um=predicted,
        pull_um=pull,
    )
    t = lap("click checks", t)
    verdicts = {check.pair_id: check for check in checks}
    for check in checks:
        logger.info(
            "pair %d: %s (%s) strength=%s offset=%s",
            check.pair_id,
            check.verdict,
            check.reason,
            check.strength,
            None if check.image_offset_um is None else round(check.image_offset_um, 1),
        )
    for pair_id in (3, 62, 67, 68):
        if pair_id in verdicts:
            logger.info(
                "pair %d explicitly: %s, strength %s",
                pair_id,
                verdicts[pair_id].verdict,
                verdicts[pair_id].strength,
            )
    confirmed = [c for c in checks if c.verdict == "confirmed"]
    contradicted = [c for c in checks if c.verdict == "contradicted"]
    logger.info(
        "verdict counts: %s",
        {
            v: sum(c.verdict == v for c in checks)
            for v in (
                "confirmed",
                "contradicted",
                "ridge",
                "inconclusive",
                "not_checked",
            )
        },
    )

    observations = ie.tissue_window_observations(reference, moving, pull_um=pull)
    t = lap("window observations", t)
    summary = observations.summary
    logger.info("window summary: %s", json.dumps(summary, default=float))
    total_s = time.perf_counter() - started
    rss_gb = _peak_rss_gb()
    logger.info("total %.1f s, peak RSS %.2f GB", total_s, rss_gb)

    expected_confirmed = sum(
        verdicts[i].verdict == "confirmed" for i in CONFIRMED_EXPECTED if i in verdicts
    )
    assert expected_confirmed >= 4
    for pair_id in (2, 3):
        if pair_id in verdicts:
            assert verdicts[pair_id].verdict != "contradicted"
    offsets = [c.image_offset_um for c in confirmed if c.image_offset_um is not None]
    assert offsets and float(np.median(offsets)) <= 10.0
    assert (len(confirmed) + len(contradicted)) / len(checks) >= 0.20

    by_size = summary["by_size"]
    assert by_size["800"]["accepted_fraction"] >= 0.12
    assert summary["n_thinned"] >= 120
    assert 0.20 <= by_size["800"]["null_p99"] <= 0.35
    assert summary["peak_sd_um"]["minor"]["median"] <= 15.0
    assert total_s <= 20 * 60
    assert rss_gb <= 10.0

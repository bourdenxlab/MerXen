"""The acceptance bookkeeping of the slow real-data tests."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest

from ._acceptance import Acceptance, nearest_confident_fused_um, seven_pair_report


def test_acceptance_logs_every_item_and_raises_once_with_all_failures(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("acceptance-test")
    caplog.set_level(logging.INFO, logger="acceptance-test")
    acceptance = Acceptance(logger, "demo")

    assert acceptance.at_most("peak memory", 2.88, 2.5, " GB") is False
    assert acceptance.at_least("pearson", 0.4, 0.3) is True
    assert acceptance.holds("digest equal", False, "differs") is False
    assert acceptance.holds("store written", True) is True

    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 4
    assert "peak memory: measured 2.88 GB | limit <= 2.5 GB | FAIL" in messages[0]
    assert "pearson: measured 0.4 | limit >= 0.3 | PASS" in messages[1]
    with pytest.raises(AssertionError) as raised:
        acceptance.finish()
    text = str(raised.value)
    assert "2 of 4 acceptance items failed" in text
    assert "peak memory" in text and "digest equal" in text
    assert "pearson" not in text


def test_acceptance_without_failures_finishes_quietly() -> None:
    acceptance = Acceptance(logging.getLogger("acceptance-test"), "demo")
    acceptance.at_most("time", 1.0, 2.0)

    acceptance.finish()


def test_seven_pair_report_counts_contradicted_and_near_estimates() -> None:
    verdicts: dict[int, tuple[str, float | None]] = {
        1: ("confirmed", 8.0),
        2: ("confirmed", 11.0),
        7: ("inconclusive", 20.3),
        8: ("inconclusive", 15.4),
        9: ("inconclusive", 30.0),  # confident estimate, but beyond 25 um
        14: ("inconclusive", None),  # no confident fused estimate
        23: ("contradicted", 12.0),  # a contradiction never counts as near
    }

    report = seven_pair_report(verdicts, (1, 2, 7, 8, 9, 14, 23))

    assert report.n_contradicted == 1
    assert report.n_near == 4
    assert report.rows[0] == "pair 1: confirmed, offset 8.0 um"
    assert report.rows[5] == "pair 14: inconclusive, offset none"


def test_seven_pair_report_tolerates_a_pair_missing_from_the_check() -> None:
    report = seven_pair_report({1: ("confirmed", 5.0)}, (1, 99))

    assert (report.n_contradicted, report.n_near) == (0, 1)
    assert report.rows[1] == "pair 99: not in the check"


def test_acceptance_records_a_missing_measurement_as_a_failure() -> None:
    acceptance = Acceptance(logging.getLogger("acceptance-test"), "demo")

    assert acceptance.at_most("peak SD", None, 15.0, " um") is False
    assert acceptance.at_least("windows", None, 1) is False
    assert acceptance.at_most("time", 1.0, 2.0) is True
    with pytest.raises(AssertionError, match="2 of 3 acceptance items failed"):
        acceptance.finish()


def test_nearest_confident_fused_um_ignores_the_best_peak_and_other_estimates() -> None:
    def estimate(representation: str, confident: bool, x: float) -> SimpleNamespace:
        return SimpleNamespace(
            representation=representation,
            confident=confident,
            image_moving_xy_um=np.array([x, 0.0]),
        )

    estimates = [
        estimate("fused", True, 32.0),  # the higher peak, farther away
        estimate("fused", True, 18.0),
        estimate("fused", False, 3.0),
        estimate("bp20", True, 1.0),
    ]

    assert nearest_confident_fused_um(estimates, (0.0, 0.0)) == pytest.approx(18.0)
    assert nearest_confident_fused_um(estimates[2:], (0.0, 0.0)) is None

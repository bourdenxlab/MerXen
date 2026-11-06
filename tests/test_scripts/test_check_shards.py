"""Tests for the shard report checks (scripts/ci/check_shards.py)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from ._ci_scripts import load_ci_script

check: ModuleType = load_ci_script("check_shards")
shard: ModuleType = load_ci_script("pytest_shard")

IDS = [f"tests/test_{f}.py::test_{t}" for f in "abcde" for t in range(4)]


def _reports(count: int = 3) -> list[dict[str, Any]]:
    shards = shard.partition(IDS, {}, count)
    return [
        {
            "shard_index": index,
            "shard_count": count,
            "all_ids_count": len(IDS),
            "all_ids_sha256": shard.ids_digest(IDS),
            "durations_sha256": "abc",
            "assigned": sorted(members),
            "outcomes": {n: "passed" for n in members},
            "durations": {n: 0.5 for n in members},
        }
        for index, members in enumerate(shards, start=1)
    ]


def test_complete_run_passes() -> None:
    assert check.verify(_reports(), expected_count=3) == []


def test_missing_shard_is_reported() -> None:
    problems = check.verify(_reports()[:2], expected_count=3)
    assert any("no report from shard(s) [3]" in p for p in problems)
    assert any("are not the 20 collected tests" in p for p in problems)


def test_dropped_test_is_reported() -> None:
    reports = _reports()
    reports[0]["assigned"].pop()
    problems = check.verify(reports)
    assert any("are not the 20 collected tests" in p for p in problems)


def test_duplicated_test_is_reported() -> None:
    reports = _reports()
    reports[1]["assigned"].append(reports[0]["assigned"][0])
    reports[1]["outcomes"][reports[0]["assigned"][0]] = "passed"
    problems = check.verify(reports)
    assert any("in several shards" in p for p in problems)


def test_shards_that_collected_differently_are_reported() -> None:
    reports = _reports()
    reports[2]["all_ids_sha256"] = "different"
    reports[1]["durations_sha256"] = "other"
    problems = check.verify(reports)
    assert any("all_ids_sha256" in p for p in problems)
    assert any("durations_sha256" in p for p in problems)


def test_assigned_test_without_outcome_is_reported() -> None:
    reports = _reports()
    del reports[0]["outcomes"][reports[0]["assigned"][0]]
    assert any("reported no outcome" in p for p in check.verify(reports))


def test_shard_count_mismatch_is_reported() -> None:
    assert check.verify(_reports(), expected_count=5)
    mixed = _reports()
    mixed[0]["shard_count"] = 4
    assert check.verify(mixed) == ["shards disagree on the shard count: [3, 4]"]
    assert check.verify([]) == ["no shard reports"]


def test_shard_without_collection_is_reported() -> None:
    reports = _reports()
    reports[0] = {"shard_index": 1, "shard_count": 3}
    assert any("did not collect" in p for p in check.verify(reports))


def test_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    paths = []
    for report in _reports():
        path = tmp_path / f"shard-{report['shard_index']}.json"
        path.write_text(json.dumps(report))
        paths.append(str(path))
    summary = tmp_path / "summary.md"
    assert (
        check.main(
            ["verify", *paths, "--expected-count", "3", "--summary", str(summary)]
        )
        == 0
    )
    assert "| Shard |" in summary.read_text()
    out = tmp_path / "durations" / "test-durations.json"
    assert check.main(["merge-durations", *paths, "--out", str(out)]) == 0
    merged = json.loads(out.read_text())
    assert merged == {check.duration_key(n): 0.5 for n in IDS}
    broken = copy.deepcopy(_reports())
    broken[0]["assigned"] = []
    (tmp_path / "shard-1.json").write_text(json.dumps(broken[0]))
    assert check.main(["verify", *paths]) == 1
    assert "check_shards:" in capsys.readouterr().err

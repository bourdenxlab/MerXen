"""Tests for the duration-balanced shard plugin (scripts/ci/pytest_shard.py)."""

from __future__ import annotations

import json
import os
import random
from types import ModuleType

import pytest

from ._ci_scripts import CI_DIR, load_ci_script

pytest_plugins = ["pytester"]

shard: ModuleType = load_ci_script("pytest_shard")
check: ModuleType = load_ci_script("check_shards")


def _keyed(durations: dict[str, float]) -> dict[str, float]:
    """Durations by node id, keyed as in a durations file."""
    return {shard.duration_key(n): v for n, v in durations.items()}


def _suite(seed: int) -> tuple[list[str], dict[str, float]]:
    rng = random.Random(seed)
    ids = [
        f"tests/test_{f}.py::test_{t}"
        for f in range(rng.randint(1, 30))
        for t in range(rng.randint(1, 25))
    ]
    durations = {
        n: rng.expovariate(1.0) * rng.choice([0.1, 1, 10])
        for n in ids
        if rng.random() > 0.15
    }
    durations[ids[0]] = 300.0  # one test far heavier than the rest
    return ids, _keyed(durations)


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("count", range(1, 8))
def test_every_test_lands_in_exactly_one_shard(seed: int, count: int) -> None:
    ids, durations = _suite(seed)
    shards = shard.partition(ids, durations, count)
    assert len(shards) == count
    flat = [n for s in shards for n in s]
    assert sorted(flat) == sorted(ids)
    assert len(flat) == len(set(flat))
    # Deterministic, whatever order the durations were recorded in.
    shuffled = dict(
        sorted(durations.items(), key=lambda kv: random.Random(seed).random())
    )
    assert shard.partition(ids, shuffled, count) == shards
    # Shards keep collection order.
    position = {n: i for i, n in enumerate(ids)}
    for members in shards:
        assert members == sorted(members, key=position.__getitem__)


def test_files_stay_together_unless_heavy() -> None:
    others = [f"tests/test_{i}.py::t" for i in range(30)]
    light = [f"tests/test_a.py::t{i}" for i in range(3)]
    shards = shard.partition(light + others, {}, 3)
    assert len({i for i, s in enumerate(shards) for n in s if n in light}) == 1
    # Above half a shard's share (here 300 s of 330 s), the file is split.
    heavy = _keyed({n: 1.0 for n in others} | {n: 100.0 for n in light})
    shards = shard.partition(light + others, heavy, 3)
    assert len({i for i, s in enumerate(shards) for n in s if n in light}) == 3


def test_balance_puts_the_heaviest_test_alone() -> None:
    ids = ["tests/test_e2e.py::big"] + [f"tests/test_{i}.py::t" for i in range(40)]
    durations = {n: 5.0 for n in ids}
    durations["tests/test_e2e.py::big"] = 200.0
    shards = shard.partition(ids, _keyed(durations), 4)
    loads = sorted(sum(durations[n] for n in s) for s in shards)
    assert loads[-1] == 200.0
    assert loads[0] >= 65.0


def test_more_shards_than_units_leaves_shards_empty() -> None:
    shards = shard.partition(["tests/test_a.py::t"], {}, 3)
    assert sorted(len(s) for s in shards) == [0, 0, 1]


def test_unknown_durations_use_the_file_then_the_suite_median() -> None:
    ids = ["a.py::x", "a.py::y", "a.py::new", "b.py::x", "c.py::new"]
    weight = shard.weights(
        ids, _keyed({"a.py::x": 2.0, "a.py::y": 4.0, "b.py::x": 10.0})
    )
    assert weight["a.py::new"] == 3.0
    assert weight["c.py::new"] == 4.0  # median of 2, 4 and 10
    assert shard.weights(["a.py::x"], {})["a.py::x"] == shard.DEFAULT_WEIGHT


def test_heavy_tests_start_first() -> None:
    weight = {"a": 1.0, "b": 30.0, "c": 2.0, "d": 12.0}
    assert shard.front_loaded(["a", "b", "c", "d"], weight) == ["b", "d", "a", "c"]


def test_durations_file_hides_test_names() -> None:
    nodeid = "tests/test_a.py::test_x[private-value]"
    key = shard.duration_key(nodeid)
    assert key.startswith(shard.file_key("tests/test_a.py") + ":")
    assert "test_a" not in key
    assert "private" not in key
    assert key == check.duration_key(nodeid)
    text = shard.format_durations({"b.py::t": 1.26, nodeid: 0.01})
    assert text == check.format_durations({"b.py::t": 1.26, nodeid: 0.01})
    assert json.loads(text) == {key: 0.0, shard.duration_key("b.py::t"): 1.3}
    assert "private" not in text


# Real pytest runs.

SUITE = {
    "test_one.py": (
        "import pytest\n"
        "def test_a(): pass\n"
        "def test_b(): pass\n"
        "@pytest.mark.slow\n"
        "def test_slow(): pass\n"
        "@pytest.mark.parametrize('x', range(4))\n"
        "def test_p(x): pass\n"
    ),
    "test_two.py": (
        "import pytest\n"
        "class TestGroup:\n"
        "    def test_x(self): pass\n"
        "    def test_y(self): pytest.skip('no')\n"
        "def test_z(): pass\n"
    ),
    "test_three.py": "def test_c(): pass\ndef test_d(): pass\n",
    "test_four.py": "def test_e(): pass\n",
}


@pytest.fixture
def suite(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> pytest.Pytester:
    pytester.makepyfile(**{name[:-3]: text for name, text in SUITE.items()})
    pytester.makeini("[pytest]\nmarkers =\n    slow: slow\n")
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join(filter(None, [str(CI_DIR), os.environ.get("PYTHONPATH")])),
    )
    monkeypatch.syspath_prepend(CI_DIR)
    return pytester


def _collected(pytester: pytest.Pytester, *args: str) -> set[str]:
    result = pytester.runpytest("--collect-only", "-q", *args)
    return {line for line in result.outlines if "::" in line}


def test_shards_reproduce_the_unsharded_selection(suite: pytest.Pytester) -> None:
    expected = _collected(suite, "-m", "not slow")
    durations = suite.path / "durations.json"
    durations.write_text(
        json.dumps(
            _keyed({"test_one.py::test_a": 5.0, "test_two.py::TestGroup::test_x": 2.0})
        )
    )
    reports = []
    for index in (1, 2, 3):
        report = suite.path / f"shard-{index}.json"
        result = suite.runpytest(
            "-p", "pytest_shard", "-m", "not slow",
            "--shard-count", "3", "--shard-index", str(index),
            "--shard-durations", str(durations), "--shard-report", str(report),
        )  # fmt: skip
        assert result.ret == 0
        reports.append(report)
    loaded = check.load_reports(reports)
    assert check.verify(loaded, expected_count=3) == []
    assert {n for r in loaded for n in r["assigned"]} == expected
    assert loaded[0]["outcomes"]
    assert all(r["durations_sha256"] for r in loaded)


def test_xdist_workers_agree_with_a_plain_run(suite: pytest.Pytester) -> None:
    plain = suite.path / "plain.json"
    workers = suite.path / "xdist.json"
    common = ("-p", "pytest_shard", "--shard-count", "2", "--shard-index", "2")
    assert (
        suite.runpytest_subprocess(
            *common, "-p", "no:xdist", "--shard-report", str(plain)
        ).ret
        == 0
    )
    assert (
        suite.runpytest_subprocess(
            *common, "-n", "2", "--shard-report", str(workers)
        ).ret
        == 0
    )
    plain_report = json.loads(plain.read_text())
    xdist_report = json.loads(workers.read_text())
    assert plain_report["assigned"] == xdist_report["assigned"]
    assert set(xdist_report["outcomes"]) == set(xdist_report["assigned"])
    assert "worker_mismatch" not in xdist_report


def test_empty_shard_passes_but_an_empty_suite_fails(suite: pytest.Pytester) -> None:
    common = ("-p", "pytest_shard", "--shard-count", "3", "test_four.py")
    results = [suite.runpytest(*common, "--shard-index", str(i)).ret for i in (1, 2, 3)]
    assert sorted(results) == [0, 0, 0]
    empty = suite.runpytest(
        "-p",
        "pytest_shard",
        "--shard-count",
        "2",
        "--shard-index",
        "1",
        "-k",
        "nothing",
    )
    assert empty.ret == pytest.ExitCode.NO_TESTS_COLLECTED


def test_store_durations(suite: pytest.Pytester) -> None:
    out = suite.path / "measured.json"
    assert (
        suite.runpytest("-p", "pytest_shard", "--shard-store-durations", str(out)).ret
        == 0
    )
    measured = json.loads(out.read_text())
    assert shard.duration_key("test_one.py::test_slow") in measured
    assert len(measured) == 13


def test_bad_index_is_a_usage_error(suite: pytest.Pytester) -> None:
    result = suite.runpytest(
        "-p", "pytest_shard", "--shard-count", "2", "--shard-index", "3"
    )
    assert result.ret == pytest.ExitCode.USAGE_ERROR

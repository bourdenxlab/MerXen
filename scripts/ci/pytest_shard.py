"""Pytest plugin: run one shard of the suite, balanced by recorded durations.

Load it with ``PYTHONPATH=scripts/ci pytest -p pytest_shard`` and pick the
shard with ``--shard-count=N --shard-index=I`` (1-based). Write the path
options as ``--shard-durations=PATH`` (one argument; see
:func:`_require_joined_path_options`). Every process that
collects (the plain run, or each pytest-xdist worker) computes the same
partition from the collected node ids and the durations file, so the xdist
workers agree on the tests and every collected test lands in exactly one
shard. ``scripts/ci/check_shards.py verify`` proves that from the reports.

The partition works on test files, so module-scoped fixtures run in one shard
only. A file heavier than half a shard's fair share is split into single
tests. Units go longest first to the least loaded shard, and inside a shard
the tests that take 10 s or more start first, so the longest one does not
start last. Tests without a recorded duration weigh the median of their file,
or of the suite.

The plugin is stdlib-only and kept out of ``src`` on purpose: it is CI
tooling, not part of the package.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import statistics
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import pytest

#: Tests at least this heavy start first within their shard.
FRONT_LOAD_SECONDS = 10.0
#: Weight of every test when no duration at all is known.
DEFAULT_WEIGHT = 1.0
_PARTITION_KEY = "pytest_shard_partition"


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the shard options."""
    group = parser.getgroup("shard", "duration-balanced sharding (scripts/ci)")
    group.addoption("--shard-count", type=int, default=1, help="number of shards")
    group.addoption("--shard-index", type=int, default=1, help="this shard, 1-based")
    group.addoption(
        "--shard-durations",
        default=None,
        help="durations file (scripts/ci/test_durations.json format)",
    )
    group.addoption("--shard-report", default=None, help="write the shard report here")
    group.addoption(
        "--shard-store-durations",
        default=None,
        help="write the measured durations of this run here, as a durations file",
    )


def file_of(nodeid: str) -> str:
    """The test file of a node id."""
    return nodeid.split("::", 1)[0]


def file_key(path: str) -> str:
    """The digest that stands for a test file in a durations file."""
    return hashlib.sha256(path.encode()).hexdigest()[:12]


def duration_key(nodeid: str) -> str:
    """The key of a test in a durations file: digests of its file and the rest.

    Test names and parametrize ids can hold the privacy checker's own test
    inputs (host paths, sample-like names), and a local alias table can flag
    a test file's name, so the committed file spells out neither."""
    path, _, rest = nodeid.partition("::")
    return f"{file_key(path)}:{hashlib.sha256(rest.encode()).hexdigest()[:12]}"


def weights(nodeids: Sequence[str], durations: dict[str, float]) -> dict[str, float]:
    """Weight per node id: its duration, else its file's median, else the suite's.

    ``durations`` is keyed by :func:`duration_key`."""
    recorded = {n: durations[k] for n in nodeids if (k := duration_key(n)) in durations}
    suite = statistics.median(recorded.values()) if recorded else DEFAULT_WEIGHT
    per_file: dict[str, list[float]] = {}
    for nodeid, seconds in recorded.items():
        per_file.setdefault(file_of(nodeid), []).append(seconds)
    file_median = {f: statistics.median(v) for f, v in per_file.items()}
    return {
        n: float(recorded[n]) if n in recorded else file_median.get(file_of(n), suite)
        for n in nodeids
    }


def partition(
    nodeids: Sequence[str], durations: dict[str, float], count: int
) -> list[list[str]]:
    """Split node ids into ``count`` shards of similar total weight.

    Deterministic for the same inputs. Shards keep collection order; a shard
    beyond the number of units is empty.
    """
    if count < 1:
        raise ValueError(f"shard count must be at least 1, got {count}")
    weight = weights(nodeids, durations)
    total = sum(weight.values())
    files: dict[str, list[str]] = {}
    for nodeid in nodeids:
        files.setdefault(file_of(nodeid), []).append(nodeid)
    units: list[tuple[float, str, list[str]]] = []
    split_above = total / (2 * count) if count > 1 else float("inf")
    for path, members in files.items():
        file_weight = sum(weight[n] for n in members)
        if file_weight > split_above and len(members) > 1:
            units.extend((weight[n], n, [n]) for n in members)
        else:
            units.append((file_weight, path, members))
    units.sort(key=lambda unit: (-unit[0], unit[1]))
    heap = [(0.0, index) for index in range(count)]
    assigned: list[set[str]] = [set() for _ in range(count)]
    for unit_weight, _key, members in units:
        load, index = heapq.heappop(heap)
        assigned[index].update(members)
        heapq.heappush(heap, (load + unit_weight, index))
    return [[n for n in nodeids if n in shard] for shard in assigned]


def front_loaded(nodeids: Sequence[str], weight: dict[str, float]) -> list[str]:
    """Heavy tests first (longest first), the rest in their original order."""
    heavy = [n for n in nodeids if weight.get(n, 0.0) >= FRONT_LOAD_SECONDS]
    heavy.sort(key=lambda n: -weight[n])
    chosen = set(heavy)
    return heavy + [n for n in nodeids if n not in chosen]


def ids_digest(nodeids: Iterable[str]) -> str:
    """Order-independent digest of a set of node ids."""
    return hashlib.sha256("\n".join(sorted(nodeids)).encode()).hexdigest()


def load_durations(path: str | None) -> tuple[dict[str, float], str | None]:
    """Durations from a JSON file and the file's digest (empty when absent)."""
    if not path:
        return {}, None
    data = Path(path).read_bytes()
    return (
        {str(k): float(v) for k, v in json.loads(data).items()},
        hashlib.sha256(data).hexdigest(),
    )


class ShardPlugin:
    """Select this shard's tests and record what ran."""

    def __init__(self, config: pytest.Config) -> None:
        self.config = config
        _require_joined_path_options(config.invocation_params.args)
        self.count = int(config.getoption("shard_count"))
        self.index = int(config.getoption("shard_index"))
        if self.count < 1 or not 1 <= self.index <= self.count:
            raise pytest.UsageError(
                f"--shard-index must be within 1..{self.count}, got {self.index}"
            )
        self.durations, self.durations_digest = load_durations(
            config.getoption("shard_durations")
        )
        self.partition: dict[str, Any] | None = None
        self.measured: dict[str, float] = {}
        self.outcomes: dict[str, str] = {}

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(
        self, config: pytest.Config, items: list[pytest.Item]
    ) -> None:
        """Keep this shard's tests (after -m, -k and --deselect)."""
        nodeids = [item.nodeid for item in items]
        shards = partition(nodeids, self.durations, self.count)
        mine = shards[self.index - 1]
        weight = weights(nodeids, self.durations)
        order = {n: i for i, n in enumerate(front_loaded(mine, weight))}
        keep = [item for item in items if item.nodeid in order]
        keep.sort(key=lambda item: order[item.nodeid])
        drop = [item for item in items if item.nodeid not in order]
        if drop:
            config.hook.pytest_deselected(items=drop)
        items[:] = keep
        self.partition = {
            "shard_index": self.index,
            "shard_count": self.count,
            "all_ids_count": len(nodeids),
            "all_ids_sha256": ids_digest(nodeids),
            "assigned": sorted(mine),
            "estimated_seconds": round(sum(weight[n] for n in mine), 1),
            "durations_sha256": self.durations_digest,
        }
        workeroutput = getattr(config, "workeroutput", None)
        if workeroutput is not None:
            workeroutput[_PARTITION_KEY] = json.dumps(self.partition)

    @pytest.hookimpl(optionalhook=True)
    def pytest_testnodedown(self, node: Any, error: Any) -> None:
        """Take the partition a pytest-xdist worker computed."""
        output = getattr(node, "workeroutput", None) or {}
        if _PARTITION_KEY in output:
            received = json.loads(output[_PARTITION_KEY])
            if self.partition is None:
                self.partition = received
            elif received != self.partition:
                self.partition["worker_mismatch"] = True

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        """Sum setup, call and teardown time and keep the worst outcome."""
        nodeid = report.nodeid
        self.measured[nodeid] = self.measured.get(nodeid, 0.0) + float(report.duration)
        previous = self.outcomes.get(nodeid)
        if report.failed:
            outcome = "failed" if report.when == "call" else "error"
        elif report.skipped:
            outcome = "skipped"
        else:
            outcome = "passed"
        rank = {None: -1, "passed": 0, "skipped": 1, "failed": 2, "error": 3}
        if rank[outcome] > rank[previous]:
            self.outcomes[nodeid] = outcome

    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:
        """Write the report; an empty shard of a non-empty suite passes."""
        if hasattr(self.config, "workerinput"):
            return
        if (
            exitstatus == pytest.ExitCode.NO_TESTS_COLLECTED
            and self.partition is not None
            and self.partition["all_ids_count"] > 0
            and not self.partition["assigned"]
        ):
            session.exitstatus = pytest.ExitCode.OK
        report_path = self.config.getoption("shard_report")
        if report_path:
            report = dict(
                self.partition or {"shard_index": self.index, "shard_count": self.count}
            )
            report["outcomes"] = dict(sorted(self.outcomes.items()))
            report["durations"] = {
                k: round(v, 3) for k, v in sorted(self.measured.items())
            }
            report["exitstatus"] = int(session.exitstatus)
            Path(report_path).parent.mkdir(parents=True, exist_ok=True)
            Path(report_path).write_text(json.dumps(report, indent=1) + "\n")
        store = self.config.getoption("shard_store_durations")
        if store:
            Path(store).parent.mkdir(parents=True, exist_ok=True)
            Path(store).write_text(format_durations(self.measured))


_PATH_OPTIONS = ("--shard-durations", "--shard-report", "--shard-store-durations")


def _require_joined_path_options(args: Sequence[str]) -> None:
    """Refuse ``--shard-durations PATH`` given as two arguments.

    Pytest picks the rootdir and the config file before it loads ``-p``
    plugins, so it reads a separate PATH of an existing file as a test path:
    the rootdir moves up to the common ancestor, pyproject.toml's pytest
    settings are lost and every node id (and durations key) changes."""
    for argument in args:
        if argument in _PATH_OPTIONS:
            raise pytest.UsageError(
                f"write {argument}=PATH as one argument; given separately, pytest "
                "also reads PATH as a test path when it picks the rootdir"
            )


def format_durations(durations: dict[str, float]) -> str:
    """Durations by node id as a durations file: keyed by :func:`duration_key`,
    sorted, one test per line, rounded to 0.1 s."""
    keyed: dict[str, float] = {}
    for nodeid, seconds in durations.items():
        key = duration_key(nodeid)
        keyed[key] = keyed.get(key, 0.0) + seconds
    lines = [f"{json.dumps(k)}: {round(v, 1)}" for k, v in sorted(keyed.items())]
    return "{\n" + ",\n".join(lines) + "\n}\n"


def pytest_configure(config: pytest.Config) -> None:
    """Register the plugin when sharding or reporting is asked for."""
    if (
        config.getoption("shard_count") != 1
        or config.getoption("shard_report")
        or config.getoption("shard_durations")
        or config.getoption("shard_store_durations")
    ):
        config.pluginmanager.register(ShardPlugin(config), "pytest_shard_plugin")

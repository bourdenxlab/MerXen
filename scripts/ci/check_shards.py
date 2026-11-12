#!/usr/bin/env python3
"""Check the shard reports of one CI run and merge their durations.

``verify`` proves that the shards together ran the suite exactly once: every
shard saw the same collected tests, the shards' assignments are disjoint and
cover all of them, and every assigned test reported an outcome.

``merge-durations`` writes the measured durations of all shards as one file
in the format of ``scripts/ci/test_durations.json``.

Stdlib only: the aggregate CI job runs it without installing anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _digest(nodeids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(nodeids)).encode()).hexdigest()


def load_reports(paths: Sequence[Path]) -> list[dict[str, Any]]:
    """Read shard reports, ordered by shard index."""
    reports = [json.loads(Path(p).read_text()) for p in paths]
    return sorted(reports, key=lambda r: int(r.get("shard_index", 0)))


def verify(
    reports: Sequence[dict[str, Any]], expected_count: int | None = None
) -> list[str]:
    """Problems with a set of shard reports (empty when the run is complete)."""
    problems: list[str] = []
    if not reports:
        return ["no shard reports"]
    counts = {int(r.get("shard_count", 0)) for r in reports}
    if len(counts) != 1:
        return [f"shards disagree on the shard count: {sorted(counts)}"]
    count = counts.pop()
    if expected_count is not None and count != expected_count:
        problems.append(f"reports say {count} shards, the plan said {expected_count}")
    indices = [int(r.get("shard_index", 0)) for r in reports]
    missing_shards = sorted(set(range(1, count + 1)) - set(indices))
    if missing_shards:
        problems.append(f"no report from shard(s) {missing_shards}")
    duplicated_shards = sorted({i for i in indices if indices.count(i) > 1})
    if duplicated_shards:
        problems.append(f"several reports from shard(s) {duplicated_shards}")
    for report in reports:
        if "all_ids_sha256" not in report:
            problems.append(
                f"shard {report.get('shard_index')} did not collect any test"
            )
        if report.get("worker_mismatch"):
            problems.append(
                f"shard {report.get('shard_index')}: "
                "xdist workers partitioned differently"
            )
    complete = [r for r in reports if "all_ids_sha256" in r]
    if not complete:
        return problems
    for key in ("all_ids_sha256", "all_ids_count", "durations_sha256"):
        values = {r.get(key) for r in complete}
        if len(values) != 1:
            problems.append(f"shards saw different {key}: {sorted(map(str, values))}")
    owner: dict[str, int] = {}
    duplicated: list[str] = []
    for report in complete:
        for nodeid in report["assigned"]:
            if nodeid in owner:
                duplicated.append(
                    f"{nodeid} (shards {owner[nodeid]} and {report['shard_index']})"
                )
            owner[nodeid] = int(report["shard_index"])
    if duplicated:
        problems.append(
            f"{len(duplicated)} test(s) in several shards: "
            + "; ".join(duplicated[:20])
        )
    all_count = int(complete[0]["all_ids_count"])
    if len(owner) != all_count or _digest(list(owner)) != complete[0]["all_ids_sha256"]:
        problems.append(
            f"the shards' tests ({len(owner)}) are not the {all_count} collected tests"
        )
    for report in complete:
        outcomes = report.get("outcomes", {})
        silent = [n for n in report["assigned"] if n not in outcomes]
        if silent:
            problems.append(
                f"shard {report['shard_index']}: {len(silent)} assigned test(s) "
                "reported no outcome: " + ", ".join(silent[:10])
            )
    return problems


def merge_durations(reports: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Measured seconds per node id across all shards."""
    merged: dict[str, float] = {}
    for report in reports:
        for nodeid, seconds in report.get("durations", {}).items():
            merged[nodeid] = merged.get(nodeid, 0.0) + float(seconds)
    return merged


def duration_key(nodeid: str) -> str:
    """A test's key in a durations file (the same as pytest_shard.duration_key)."""
    path, _, rest = nodeid.partition("::")
    file_digest = hashlib.sha256(path.encode()).hexdigest()[:12]
    return f"{file_digest}:{hashlib.sha256(rest.encode()).hexdigest()[:12]}"


def format_durations(durations: dict[str, float]) -> str:
    """Durations by node id as a durations file (as pytest_shard writes it):
    keyed by :func:`duration_key`, sorted, one per line, rounded to 0.1 s."""
    keyed: dict[str, float] = {}
    for nodeid, seconds in durations.items():
        key = duration_key(nodeid)
        keyed[key] = keyed.get(key, 0.0) + seconds
    lines = [f"{json.dumps(k)}: {round(v, 1)}" for k, v in sorted(keyed.items())]
    return "{\n" + ",\n".join(lines) + "\n}\n"


def _summary(reports: Sequence[dict[str, Any]]) -> str:
    rows = [
        "| Shard | Tests | Estimated s | Measured s | Failed |",
        "|---|---|---|---|---|",
    ]
    for report in reports:
        outcomes = report.get("outcomes", {})
        failed = sum(1 for v in outcomes.values() if v in ("failed", "error"))
        measured = sum(report.get("durations", {}).values())
        rows.append(
            f"| {report.get('shard_index')} | {len(report.get('assigned', []))} | "
            f"{report.get('estimated_seconds', '')} | {measured:.0f} | {failed} |"
        )
    return "\n".join(rows) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("verify", help="check that the shards ran every test once")
    check.add_argument("reports", nargs="+", type=Path)
    check.add_argument("--expected-count", type=int)
    check.add_argument("--summary", type=Path, help="append a Markdown table here")
    merge = sub.add_parser(
        "merge-durations", help="merge the shards' measured durations"
    )
    merge.add_argument("reports", nargs="+", type=Path)
    merge.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    reports = load_reports(args.reports)
    if args.command == "verify":
        problems = verify(reports, args.expected_count)
        if args.summary is not None:
            with open(args.summary, "a", encoding="utf-8") as handle:
                handle.write("### Shards\n\n" + _summary(reports))
        if problems:
            for problem in problems:
                print(f"check_shards: {problem}", file=sys.stderr)
            return 1
        total = sum(len(r["assigned"]) for r in reports)
        print(
            f"check_shards: {len(reports)} shard(s) ran all {total} collected "
            "tests once"
        )
        return 0
    merged = merge_durations(reports)
    if not merged:
        print("check_shards: no durations in the reports", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(format_durations(merged))
    print(f"check_shards: wrote {len(merged)} durations to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

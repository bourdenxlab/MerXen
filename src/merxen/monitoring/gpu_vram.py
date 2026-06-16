"""Sample NVIDIA GPU VRAM use for a running process tree.

The monitor intentionally shells out to ``nvidia-smi`` instead of importing NVML
bindings so it can run in the same lightweight Nextflow environment as the task.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from typing import Any, cast

GPU_QUERY_FIELDS = (
    "index",
    "uuid",
    "name",
    "memory.used",
    "memory.total",
    "utilization.gpu",
)
APP_QUERY_FIELDS = (
    "gpu_uuid",
    "pid",
    "process_name",
    "used_memory",
)
SAMPLE_HEADER = (
    "epoch_s",
    "iso_time",
    "gpu_index",
    "gpu_uuid",
    "gpu_name",
    "gpu_memory_used_mib",
    "gpu_memory_total_mib",
    "gpu_utilization_pct",
    "task_pids",
    "task_gpu_memory_used_mib",
    "compute_apps_json",
)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _iso_now() -> str:
    return _utc_now().isoformat(timespec="seconds")


def _parse_int(value: str | None, *, default: int = 0) -> int:
    if value is None:
        return default
    match = re.search(r"-?\d+", value.replace(",", ""))
    if match is None:
        return default
    return int(match.group(0))


def _read_csv_rows(raw: str) -> list[list[str]]:
    if not raw.strip():
        return []
    return [
        [field.strip() for field in row]
        for row in csv.reader(StringIO(raw), skipinitialspace=True)
        if row
    ]


def _run_nvidia_smi(
    nvidia_smi: str,
    fields: Sequence[str],
    kind: str,
) -> list[list[str]]:
    cmd = [
        nvidia_smi,
        f"--query-{kind}={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0:
        return []
    return _read_csv_rows(proc.stdout)


def _find_nvidia_smi() -> str | None:
    return shutil.which("nvidia-smi")


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _child_pids(pid: int) -> list[int]:
    task_dir = Path("/proc") / str(pid) / "task"
    if not task_dir.exists():
        return []
    children: list[int] = []
    for child_file in task_dir.glob("*/children"):
        try:
            raw = child_file.read_text().strip()
        except OSError:
            continue
        if not raw:
            continue
        children.extend(int(child) for child in raw.split() if child.isdigit())
    return children


def process_tree_pids(root_pid: int) -> set[int]:
    """Return ``root_pid`` plus descendants visible through ``/proc``."""
    seen: set[int] = set()
    pending = [root_pid]
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        pending.extend(child for child in _child_pids(pid) if child not in seen)
    return seen


def _gpu_rows(nvidia_smi: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _run_nvidia_smi(nvidia_smi, GPU_QUERY_FIELDS, "gpu"):
        padded = row + [""] * (len(GPU_QUERY_FIELDS) - len(row))
        rows.append(
            {
                "index": padded[0],
                "uuid": padded[1],
                "name": padded[2],
                "memory_used_mib": _parse_int(padded[3]),
                "memory_total_mib": _parse_int(padded[4]),
                "utilization_pct": _parse_int(padded[5]),
            }
        )
    return rows


def _compute_app_rows(nvidia_smi: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _run_nvidia_smi(nvidia_smi, APP_QUERY_FIELDS, "compute-apps"):
        padded = row + [""] * (len(APP_QUERY_FIELDS) - len(row))
        rows.append(
            {
                "gpu_uuid": padded[0],
                "pid": _parse_int(padded[1], default=-1),
                "process_name": padded[2],
                "used_memory_mib": _parse_int(padded[3]),
            }
        )
    return rows


def _write_unavailable_summary(
    *,
    target_pid: int,
    interval_seconds: float,
    samples_path: Path,
    summary_path: Path,
    reason: str,
) -> None:
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    samples_path.write_text("\t".join(SAMPLE_HEADER) + "\n")
    summary_path.write_text(
        json.dumps(
            {
                "monitor_available": False,
                "reason": reason,
                "target_pid": target_pid,
                "sample_interval_seconds": interval_seconds,
                "sample_count": 0,
                "gpu_sample_rows": 0,
                "started_at": _iso_now(),
                "ended_at": _iso_now(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def monitor_process(
    *,
    target_pid: int,
    interval_seconds: float,
    samples_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    """Monitor GPU VRAM while ``target_pid`` is alive."""
    nvidia_smi = _find_nvidia_smi()
    if nvidia_smi is None:
        _write_unavailable_summary(
            target_pid=target_pid,
            interval_seconds=interval_seconds,
            samples_path=samples_path,
            summary_path=summary_path,
            reason="nvidia-smi not found",
        )
        return cast(dict[str, Any], json.loads(summary_path.read_text()))

    samples_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    started_at = _iso_now()
    sample_count = 0
    gpu_sample_rows = 0
    peak_task: dict[str, Any] = {"memory_used_mib": 0}
    peak_total: dict[str, Any] = {"memory_used_mib": 0}

    interval_seconds = max(float(interval_seconds), 0.1)

    with samples_path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(SAMPLE_HEADER)

        while True:
            alive = _pid_exists(target_pid)
            epoch = time.time()
            iso_time = _utc_now().isoformat(timespec="seconds")
            task_pids = process_tree_pids(target_pid)
            apps = _compute_app_rows(nvidia_smi)
            gpus = _gpu_rows(nvidia_smi)
            apps_by_gpu: dict[str, list[dict[str, Any]]] = {}
            task_memory_by_gpu: dict[str, int] = {}
            for app in apps:
                gpu_uuid = str(app["gpu_uuid"])
                apps_by_gpu.setdefault(gpu_uuid, []).append(app)
                if int(app["pid"]) in task_pids:
                    task_memory_by_gpu[gpu_uuid] = task_memory_by_gpu.get(
                        gpu_uuid,
                        0,
                    ) + int(app["used_memory_mib"])

            for gpu in gpus:
                gpu_uuid = str(gpu["uuid"])
                task_used = task_memory_by_gpu.get(gpu_uuid, 0)
                gpu_used = int(gpu["memory_used_mib"])
                writer.writerow(
                    [
                        f"{epoch:.3f}",
                        iso_time,
                        gpu["index"],
                        gpu_uuid,
                        gpu["name"],
                        gpu_used,
                        gpu["memory_total_mib"],
                        gpu["utilization_pct"],
                        ",".join(str(pid) for pid in sorted(task_pids)),
                        task_used,
                        json.dumps(
                            apps_by_gpu.get(gpu_uuid, []),
                            separators=(",", ":"),
                        ),
                    ]
                )
                gpu_sample_rows += 1
                if task_used > int(peak_task["memory_used_mib"]):
                    peak_task = {
                        "memory_used_mib": task_used,
                        "gpu_index": gpu["index"],
                        "gpu_uuid": gpu_uuid,
                        "gpu_name": gpu["name"],
                        "sampled_at": iso_time,
                    }
                if gpu_used > int(peak_total["memory_used_mib"]):
                    peak_total = {
                        "memory_used_mib": gpu_used,
                        "gpu_index": gpu["index"],
                        "gpu_uuid": gpu_uuid,
                        "gpu_name": gpu["name"],
                        "sampled_at": iso_time,
                    }
            handle.flush()
            sample_count += 1
            if not alive:
                break
            time.sleep(interval_seconds)

    ended_at = _iso_now()
    summary: dict[str, Any] = {
        "monitor_available": True,
        "target_pid": target_pid,
        "sample_interval_seconds": interval_seconds,
        "sample_count": sample_count,
        "gpu_sample_rows": gpu_sample_rows,
        "started_at": started_at,
        "ended_at": ended_at,
        "peak_task_gpu_memory_used_mib": int(peak_task["memory_used_mib"]),
        "peak_task_gpu_memory_used_gib": round(
            int(peak_task["memory_used_mib"]) / 1024,
            3,
        ),
        "peak_task_gpu_index": peak_task.get("gpu_index"),
        "peak_task_gpu_uuid": peak_task.get("gpu_uuid"),
        "peak_task_gpu_name": peak_task.get("gpu_name"),
        "peak_task_gpu_sampled_at": peak_task.get("sampled_at"),
        "peak_total_gpu_memory_used_mib": int(peak_total["memory_used_mib"]),
        "peak_total_gpu_memory_used_gib": round(
            int(peak_total["memory_used_mib"]) / 1024,
            3,
        ),
        "peak_total_gpu_index": peak_total.get("gpu_index"),
        "peak_total_gpu_uuid": peak_total.get("gpu_uuid"),
        "peak_total_gpu_name": peak_total.get("gpu_name"),
        "peak_total_gpu_sampled_at": peak_total.get("sampled_at"),
        "notes": (
            "Task GPU memory sums nvidia-smi compute-apps used_memory for the "
            "target process and visible descendants. Total GPU memory is full "
            "device memory.used and includes unrelated processes."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True, help="Root process PID.")
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=2.0,
        help="Sampling interval in seconds.",
    )
    parser.add_argument(
        "--samples-path",
        type=Path,
        required=True,
        help="TSV path for per-sample GPU memory rows.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        required=True,
        help="JSON path for peak GPU memory summary.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    monitor_process(
        target_pid=args.pid,
        interval_seconds=args.interval_seconds,
        samples_path=args.samples_path,
        summary_path=args.summary_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

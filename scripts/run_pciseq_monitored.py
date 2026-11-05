#!/usr/bin/env python3
"""Run a command in a bounded user service, retaining independent diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType
from typing import Any, NoReturn, TextIO

_GIB = 1024**3
_ENVIRONMENT = ("PATH", "JAVA_HOME", "NXF_JAVA_HOME", "PYTHONPATH", "TMPDIR")
_PROPERTIES = (
    "LoadState,ActiveState,SubState,Result,ExecMainStatus,ExecMainCode,"
    "ControlGroup,MemoryCurrent,MemoryPeak,CPUUsageNSec"
)


def unit_name(value: str) -> str:
    """Validate a literal service name without systemd glob or path semantics."""
    name = value.removesuffix(".service")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", name) or name.endswith(
        (".scope", ".slice", ".service")
    ):
        raise argparse.ArgumentTypeError("unit must be a simple unique service name")
    return f"{name}.service"


def parse_counters(text: str) -> dict[str, int]:
    """Parse Linux key/value counter files without retaining unrelated content."""
    result = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 2:
            try:
                result[fields[0].rstrip(":")] = int(fields[1])
            except ValueError:
                continue
    return result


def available_memory() -> int:
    """Return host available memory in bytes from the Linux kernel."""
    return parse_counters(Path("/proc/meminfo").read_text())["MemAvailable"] * 1024


def systemctl(unit: str) -> dict[str, str]:
    """Read only the workload unit's resource and terminal-state properties."""
    result = subprocess.run(
        ["systemctl", "--user", "show", unit, f"--property={_PROPERTIES}"],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if result.returncode not in (0, 4):
        raise RuntimeError(f"Cannot inspect user service: {result.stderr.strip()}")
    return dict(
        line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
    )


def service_command(
    unit: str,
    memory_gib: int,
    cpus: int,
    command: list[str],
    environment: dict[str, str],
) -> list[str]:
    """Construct an argument-only service invocation with explicit resource limits."""
    properties = {
        "MemoryMax": f"{memory_gib}G",
        "MemorySwapMax": "0",
        "MemoryAccounting": "yes",
        "CPUAccounting": "yes",
        "CPUQuota": f"{cpus * 100}%",
        "OOMPolicy": "kill",
        "KillMode": "control-group",
    }
    return [
        "systemd-run",
        "--user",
        f"--unit={unit}",
        "--pipe",
        "--wait",
        "--service-type=exec",
        f"--working-directory={Path.cwd()}",
        *(f"--property={key}={value}" for key, value in properties.items()),
        *(
            f"--setenv={key}={environment[key]}"
            for key in _ENVIRONMENT
            if key in environment
        ),
        "--",
        *command,
    ]


def snapshot(unit: str) -> dict[str, Any]:
    """Sample independently of the workload, including reclaimable file-cache use."""
    result: dict[str, Any] = {
        "utc": datetime.now(UTC).isoformat(),
        "host_available_bytes": available_memory(),
    }
    try:
        properties = systemctl(unit)
        result["systemd"] = properties
        group = properties.get("ControlGroup", "")
        if group.startswith("/") and ".." not in Path(group).parts:
            root = Path("/sys/fs/cgroup") / group.lstrip("/")
            for name in (
                "memory.current",
                "memory.peak",
                "memory.events",
                "memory.stat",
                "cpu.stat",
            ):
                try:
                    text = (root / name).read_text()
                    values: int | dict[str, int] = (
                        int(text)
                        if name in {"memory.current", "memory.peak"}
                        else parse_counters(text)
                    )
                    if name == "memory.stat" and isinstance(values, dict):
                        values = {
                            key: values[key]
                            for key in ("anon", "file")
                            if key in values
                        }
                    result[name] = values
                except (OSError, ValueError) as error:
                    result[name] = {"unavailable": str(error)}
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        result["monitor_error"] = str(error)
    return result


def _tee(stream: TextIO, destination: Path) -> None:
    with destination.open("w") as output:
        for line in stream:
            output.write(line)
            output.flush()
            try:
                sys.stdout.write(line)
                sys.stdout.flush()
            except OSError:
                # A disconnected terminal must not interrupt the durable log.
                continue


def _save(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def _observed_peak(value: dict[str, Any]) -> int:
    candidates = [value.get("memory.peak"), value.get("systemd", {}).get("MemoryPeak")]
    return max((int(item) for item in candidates if str(item).isdecimal()), default=0)


def _runner_report(path: Path) -> dict[str, Any]:
    """Retain the runner's rounded peak when a short-lived cgroup is already gone."""
    result: dict[str, Any] = {}
    with path.open() as stream:
        for line in stream:
            if line.lstrip().startswith("Finished with result:"):
                result["systemd_run_finished_result"] = line.split(":", 1)[1].strip()
            if not line.lstrip().startswith("Memory peak:"):
                continue
            result["systemd_run_memory_peak_report"] = line.rstrip("\r\n")
            result.pop("systemd_run_memory_peak_approx_bytes", None)
            match = re.match(
                r"\s*Memory peak:\s*([0-9]+(?:\.[0-9]+)?)"
                r"\s*([KMGTPE]?)(?:i?B)?(?:\s|$)",
                line,
            )
            if match:
                scale = " KMGTPE".index(match[2] or " ")
                result["systemd_run_memory_peak_approx_bytes"] = int(
                    float(match[1]) * 1024**scale
                )
    return result


def _interrupt(signum: int, frame: FrameType | None) -> NoReturn:
    raise KeyboardInterrupt


def run(
    unit: str, log_dir: Path, memory_gib: int, cpus: int, command: list[str]
) -> int:
    """Launch one new service and preserve observations even after an OOM kill."""
    if memory_gib <= 0 or cpus <= 0 or not command:
        raise ValueError("memory, CPUs and a command are required")
    unit = unit_name(unit)
    if systemctl(unit).get("LoadState") != "not-found":
        raise ValueError("Service already exists; choose a new unique unit name")
    if available_memory() < (memory_gib + 32) * _GIB:
        raise ValueError(
            "Insufficient available RAM: requested budget plus 32 GiB required"
        )
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / "run_spec.json").open("x") as stream:
        json.dump(
            {
                "unit": unit,
                "memory_gib": memory_gib,
                "cpus": cpus,
                "command": command,
                "cwd": str(Path.cwd()),
                "monitor_pid": os.getpid(),
                "started_utc": datetime.now(UTC).isoformat(),
            },
            stream,
            indent=2,
        )
    started = time.monotonic()
    process = subprocess.Popen(
        service_command(unit, memory_gib, cpus, command, dict(os.environ)),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    tee = threading.Thread(
        target=_tee, args=(process.stdout, log_dir / "workflow.log"), daemon=True
    )
    tee.start()
    interrupted = False
    stop_error = None
    peak = 0
    with (log_dir / "monitor.jsonl").open("a") as observations:
        try:
            while True:
                observation = snapshot(unit)
                peak = max(peak, _observed_peak(observation))
                observations.write(json.dumps(observation) + "\n")
                observations.flush()
                os.fsync(observations.fileno())
                if process.poll() is not None:
                    break
                time.sleep(5)
        except KeyboardInterrupt:
            interrupted = True
            try:
                subprocess.run(
                    ["systemctl", "--user", "stop", "--no-block", unit],
                    check=True,
                    timeout=15,
                )
            except (OSError, subprocess.SubprocessError) as error:
                stop_error = str(error)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                subprocess.run(
                    ["systemctl", "--user", "kill", "--signal=KILL", unit],
                    check=False,
                    timeout=15,
                )
                process.wait(timeout=30)
        finally:
            final = snapshot(unit)
            peak = max(peak, _observed_peak(final))
            final.update(
                {
                    "elapsed_seconds": time.monotonic() - started,
                    "returncode": process.returncode,
                    "interrupted": interrupted,
                    "memory_peak_includes_file_cache": True,
                    "peak_observed_cgroup_memory_bytes": peak,
                    "stop_error": stop_error,
                    "state": "interrupted"
                    if interrupted
                    else ("complete" if process.returncode == 0 else "failed"),
                }
            )
            _save(log_dir / "result.json", final)
            with (log_dir / "unit-journal.log").open("w") as journal:
                argv = [
                    "journalctl",
                    "--user",
                    "-u",
                    unit,
                    "--no-pager",
                    "--output=short-iso",
                ]
                try:
                    subprocess.run(
                        argv,
                        stdout=journal,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=30,
                    )
                except (OSError, subprocess.SubprocessError) as error:
                    final["journal_error"] = str(error)
                    _save(log_dir / "result.json", final)
    tee.join(timeout=5)
    final["console_log_complete"] = not tee.is_alive()
    try:
        final.update(_runner_report(log_dir / "workflow.log"))
    except OSError as error:
        final["console_report_error"] = str(error)
    _save(log_dir / "result.json", final)
    return 130 if interrupted else int(process.returncode or 0)


def main() -> int:
    """Parse the launcher settings and an uninterpreted command after ``--``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", type=unit_name, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--memory-gib", type=int, default=450)
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, _interrupt)
    try:
        return run(args.unit, args.log_dir, args.memory_gib, args.cpus, command)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    raise SystemExit(main())

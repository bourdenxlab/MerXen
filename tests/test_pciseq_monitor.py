"""Resource limits, independent observations and safe interruption of one unit."""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "pciseq_monitor", Path(__file__).parents[1] / "scripts" / "run_pciseq_monitored.py"
)
assert _SPEC is not None and _SPEC.loader is not None
monitor = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(monitor)


@pytest.mark.parametrize(
    "value",
    [
        "../other",
        "a/b",
        "*",
        "--other",
        "a b",
        ".service",
        "x.scope",
        "x.service.service",
    ],
)
def test_unit_names_reject_paths_globs_options_and_other_unit_types(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        monitor.unit_name(value)
    assert monitor.unit_name("pciseq-toy-123") == "pciseq-toy-123.service"


def test_command_has_hard_limits_and_only_allowlisted_environment() -> None:
    command = ["nextflow", "run", "workflow.nf", "--argument", "literal; $(text)"]
    argv = monitor.service_command(
        "pciseq-toy.service",
        450,
        8,
        command,
        {
            "PATH": "runtime/bin",
            "JAVA_HOME": "java",
            "PYTHONPATH": "source",
            "SECRET": "hidden",
        },
    )
    assert argv[argv.index("--") + 1 :] == command
    assert "--wait" in argv and "--pipe" in argv
    assert "--collect" not in argv
    for value in (
        "MemoryMax=450G",
        "MemorySwapMax=0",
        "CPUQuota=800%",
        "MemoryAccounting=yes",
        "CPUAccounting=yes",
        "OOMPolicy=kill",
        "KillMode=control-group",
    ):
        assert f"--property={value}" in argv
    assert "--setenv=PATH=runtime/bin" in argv
    assert "--setenv=PYTHONPATH=source" in argv
    assert f"--working-directory={Path.cwd()}" in argv
    assert not any("SECRET" in value or "hidden" in value for value in argv)


def test_kernel_counter_parsing_and_snapshot_keep_anon_file_and_oom_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert monitor.parse_counters("MemAvailable: 123 kB\nbad value\n") == {
        "MemAvailable": 123
    }
    files = {
        "memory.current": "101",
        "memory.peak": "199",
        "memory.events": "low 0\noom 1\noom_kill 1\n",
        "memory.stat": "anon 70\nfile 31\nslab 5\n",
        "cpu.stat": "usage_usec 112\nuser_usec 100\nsystem_usec 12\n",
    }
    monkeypatch.setattr(monitor, "available_memory", lambda: 500 * 1024**3)
    monkeypatch.setattr(
        monitor,
        "systemctl",
        lambda unit: {"ControlGroup": "/user.slice/toy.service", "Result": "oom-kill"},
    )
    monkeypatch.setattr(Path, "read_text", lambda path: files[path.name])
    result = monitor.snapshot("toy.service")
    assert result["memory.peak"] == 199
    assert result["memory.stat"] == {"anon": 70, "file": 31}
    assert result["memory.events"]["oom_kill"] == 1
    assert result["cpu.stat"]["usage_usec"] == 112


def test_gone_cgroup_does_not_hide_saved_systemd_peak() -> None:
    assert (
        monitor._observed_peak(
            {
                "memory.peak": {"unavailable": "removed"},
                "systemd": {"MemoryPeak": "321"},
            }
        )
        == 321
    )
    assert monitor._observed_peak({"systemd": {"MemoryPeak": "[not set]"}}) == 0


class _Process:
    def __init__(self, returncode: int = 0) -> None:
        self.stdout = io.StringIO("workload diagnostic\n")
        self.returncode: int | None = None
        self.final_code = returncode
        self.polls = 0

    def poll(self) -> int | None:
        self.polls += 1
        if self.polls > 1:
            self.returncode = self.final_code
        return self.returncode

    def wait(self, timeout: int) -> int:
        self.returncode = self.final_code
        return self.final_code


def _mock_services(
    monkeypatch: pytest.MonkeyPatch, *, returncode: int = 0
) -> list[list[str]]:
    calls: list[list[str]] = []
    monkeypatch.setattr(monitor, "systemctl", lambda unit: {"LoadState": "not-found"})
    monkeypatch.setattr(monitor, "available_memory", lambda: 500 * 1024**3)
    monkeypatch.setattr(
        monitor,
        "snapshot",
        lambda unit: {
            "memory.peak": 123,
            "systemd": {"Result": "success" if returncode == 0 else "oom-kill"},
        },
    )
    monkeypatch.setattr(monitor.time, "sleep", lambda seconds: None)

    def popen(argv: list[str], **kwargs: Any) -> _Process:
        calls.append(argv)
        assert kwargs["stdin"] == subprocess.DEVNULL
        return _Process(returncode)

    def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        if argv[0] == "journalctl":
            kwargs["stdout"].write("preserved unit journal\n")
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(monitor.subprocess, "Popen", popen)
    monkeypatch.setattr(monitor.subprocess, "run", run)
    return calls


@pytest.mark.parametrize("returncode", [0, 137])
def test_monitor_preserves_logs_and_terminal_metrics_after_completion_or_oom(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, returncode: int
) -> None:
    calls = _mock_services(monkeypatch, returncode=returncode)
    assert monitor.run("pciseq-toy", tmp_path, 450, 8, ["program", "arg"]) == returncode
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["returncode"] == returncode
    assert result["state"] == ("complete" if returncode == 0 else "failed")
    assert result["peak_observed_cgroup_memory_bytes"] == 123
    assert result["memory_peak_includes_file_cache"] is True
    assert len((tmp_path / "monitor.jsonl").read_text().splitlines()) == 2
    assert (tmp_path / "workflow.log").read_text() == "workload diagnostic\n"
    assert "preserved unit journal" in (tmp_path / "unit-journal.log").read_text()
    assert not any("stop" in call for call in calls)


def test_interrupt_stops_only_its_new_unit_and_retains_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _mock_services(monkeypatch, returncode=143)

    def interrupt(seconds: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(monitor.time, "sleep", interrupt)
    assert monitor.run("pciseq-owned", tmp_path, 450, 8, ["program"]) == 130
    stops = [call for call in calls if "stop" in call]
    assert stops == [
        ["systemctl", "--user", "stop", "--no-block", "pciseq-owned.service"]
    ]
    assert json.loads((tmp_path / "result.json").read_text())["interrupted"] is True


def test_failed_journal_collection_does_not_replace_workload_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_services(monkeypatch, returncode=137)

    def missing_journal(argv: list[str], **kwargs: Any) -> None:
        raise subprocess.TimeoutExpired(argv, 30)

    monkeypatch.setattr(monitor.subprocess, "run", missing_journal)
    assert monitor.run("pciseq-owned", tmp_path, 450, 8, ["program"]) == 137
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["returncode"] == 137
    assert "journal_error" in result


def test_short_success_retains_rounded_runner_peak_after_unit_garbage_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_services(monkeypatch)
    monkeypatch.setattr(
        monitor, "snapshot", lambda unit: {"systemd": {"LoadState": "not-found"}}
    )
    process = _Process()
    report = "Memory peak: 588.0K (swap: 0B)"
    process.stdout = io.StringIO(f"Finished with result: success\n{report}\n")
    monkeypatch.setattr(monitor.subprocess, "Popen", lambda *args, **kwargs: process)
    assert monitor.run("pciseq-short", tmp_path, 450, 8, ["program"]) == 0
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["peak_observed_cgroup_memory_bytes"] == 0
    assert result["systemd_run_memory_peak_report"] == report
    assert result["systemd_run_memory_peak_approx_bytes"] == 602112
    assert result["systemd_run_finished_result"] == "success"
    assert result["console_log_complete"] is True


def test_missing_runner_peak_remains_unknown_after_unit_garbage_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_services(monkeypatch)
    monkeypatch.setattr(
        monitor, "snapshot", lambda unit: {"systemd": {"LoadState": "not-found"}}
    )
    assert monitor.run("pciseq-short", tmp_path, 450, 8, ["program"]) == 0
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["peak_observed_cgroup_memory_bytes"] == 0
    assert "systemd_run_memory_peak_report" not in result
    assert "systemd_run_memory_peak_approx_bytes" not in result


@pytest.mark.parametrize("fault", ["existing_unit", "low_ram", "existing_log"])
def test_preflight_rejects_unsafe_launch_without_starting_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    calls = _mock_services(monkeypatch)
    if fault == "existing_unit":
        monkeypatch.setattr(monitor, "systemctl", lambda unit: {"LoadState": "loaded"})
    elif fault == "low_ram":
        monkeypatch.setattr(monitor, "available_memory", lambda: 481 * 1024**3)
    else:
        (tmp_path / "run_spec.json").write_text("old run\n")
    with pytest.raises((ValueError, FileExistsError)):
        monitor.run("pciseq-owned", tmp_path, 450, 8, ["program"])
    assert not calls

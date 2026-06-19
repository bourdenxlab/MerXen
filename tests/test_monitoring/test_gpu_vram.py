"""Tests for GPU VRAM monitoring helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from merxen.monitoring import gpu_vram


def test_monitor_process_writes_unavailable_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing nvidia-smi should produce useful artifacts without failing."""
    samples_path = tmp_path / "samples.tsv"
    summary_path = tmp_path / "summary.json"
    monkeypatch.setattr(gpu_vram, "_find_nvidia_smi", lambda: None)

    summary = gpu_vram.monitor_process(
        target_pid=os.getpid(),
        interval_seconds=0.1,
        samples_path=samples_path,
        summary_path=summary_path,
    )

    assert summary["monitor_available"] is False
    assert summary["reason"] == "nvidia-smi not found"
    assert samples_path.read_text().startswith("epoch_s\tiso_time\tgpu_index")
    assert json.loads(summary_path.read_text())["sample_count"] == 0

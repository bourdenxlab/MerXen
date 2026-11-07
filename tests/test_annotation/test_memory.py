"""Tests for the process-tree memory sampler (plan §8.7)."""

from __future__ import annotations

import subprocess
import sys
import time

from merxen.annotation.memory import ProcessTreeSampler


def test_sampler_measures_the_current_process() -> None:
    with ProcessTreeSampler(interval_s=0.05) as sampler:
        time.sleep(0.2)
    peak = sampler.peak()
    assert peak.n_samples >= 1
    assert peak.peak_rss_gb > 0.0
    assert peak.peak_processes >= 1
    record = peak.to_json()
    assert set(record) == {
        "peak_tree_rss_gb",
        "peak_tree_pss_gb",
        "peak_tree_processes",
        "tree_samples",
        "tree_sample_interval_s",
    }


def test_sampler_counts_child_processes() -> None:
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time; x = bytearray(80 * 2**20); time.sleep(1.5)",
        ]
    )
    try:
        sampler = ProcessTreeSampler(child.pid, interval_s=0.1).start()
        time.sleep(1.0)
        peak = sampler.stop()
    finally:
        child.wait()
    # The child holds an 80 MiB buffer.
    assert peak.peak_rss_gb > 0.07


def test_sampler_follows_the_descendants_of_its_root() -> None:
    # The OD-E8 process-tree peaks sample the PREP process with its ctm
    # children: the root's descendants must be summed.
    baseline, _, _ = ProcessTreeSampler().sample()
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys, time; x = bytearray(80 * 2**20); print('ready', flush=True); "
            "time.sleep(5)",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None and child.stdout.readline().strip() == "ready"
        with ProcessTreeSampler(interval_s=0.05) as sampler:
            time.sleep(0.3)
        peak = sampler.peak()
    finally:
        child.kill()
        child.wait()
    assert peak.peak_processes >= 2
    # The child's 80 MiB buffer is in the tree's RSS.
    assert peak.peak_rss_gb - baseline / 2**30 > 0.07


def test_sampler_of_a_finished_process_reports_nothing() -> None:
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    sampler = ProcessTreeSampler(child.pid)
    assert sampler.sample() == (0, None, 0)

"""Tests for ProSeg subprocess orchestration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from merxen.segmentation.proseg import run_proseg_refinement


class _FakePopen:
    """Minimal subprocess.Popen test double."""

    def __init__(self: _FakePopen, cmd: list[str], **_: object) -> None:
        self.cmd = cmd
        self.stdout = iter(["starting\n", "finished\n"])

    def wait(self: _FakePopen) -> int:
        return 0


def test_run_proseg_refinement_builds_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The wrapper should pass core flags and return the output path."""
    captured: dict[str, list[str]] = {}

    def _fake_check(_: str | Path) -> str:
        return "3.1.0"

    def _fake_popen(cmd: list[str], **kwargs: object) -> _FakePopen:
        del kwargs
        captured["cmd"] = cmd
        return _FakePopen(cmd)

    monkeypatch.setattr(
        "merxen.segmentation.proseg._check_proseg_available", _fake_check
    )
    monkeypatch.setattr("merxen.segmentation.proseg.subprocess.Popen", _fake_popen)

    output_path = tmp_path / "out.zarr"
    output_path.mkdir()
    transcripts = pd.DataFrame(
        {
            "x": [1.0, 2.0],
            "y": [1.5, 2.5],
            "z": [0.0, 0.0],
            "feature_name": ["A", "B"],
            "cell_id": [0, 1],
        }
    )

    out = run_proseg_refinement(
        transcripts_df=transcripts,
        output_path=output_path,
        proseg_binary="/usr/bin/proseg",
        samples=10,
        num_threads=2,
    )

    assert out == output_path
    assert "--output-spatialdata" in captured["cmd"]
    assert "--recorded-samples" in captured["cmd"]
    assert "--samples" in captured["cmd"]


def test_run_proseg_refinement_requires_cellpose_transform(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Supplying masks without a scale/affine should raise a ValueError."""
    monkeypatch.setattr(
        "merxen.segmentation.proseg._check_proseg_available",
        lambda _: "3.1.0",
    )

    transcripts = pd.DataFrame(
        {
            "x": [1.0],
            "y": [2.0],
            "z": [0.0],
            "feature_name": ["A"],
            "cell_id": [0],
        }
    )

    with pytest.raises(ValueError, match="Cellpose masks were supplied"):
        run_proseg_refinement(
            transcripts_df=transcripts,
            output_path=tmp_path / "out.zarr",
            proseg_binary="/usr/bin/proseg",
            cellpose_masks=np.zeros((2, 2), dtype=np.uint32),
        )


def test_run_proseg_refinement_validates_required_columns(tmp_path: Path) -> None:
    """Missing required transcript columns should fail fast."""
    transcripts = pd.DataFrame({"x": [1.0], "y": [2.0]})

    with pytest.raises(ValueError, match="Missing required columns"):
        run_proseg_refinement(
            transcripts_df=transcripts,
            output_path=tmp_path / "out.zarr",
            proseg_binary="/usr/bin/proseg",
        )

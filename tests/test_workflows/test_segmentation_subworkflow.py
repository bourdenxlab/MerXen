"""Workflow smoke tests for split Cellpose and ProSeg execution."""

from __future__ import annotations

from pathlib import Path


def test_segmentation_routes_cellpose_to_gpu_and_proseg_to_cpu() -> None:
    """The segmentation subworkflow should expose independent scheduler jobs."""
    repo_root = Path(__file__).resolve().parents[2]
    config_text = (repo_root / "workflows" / "nextflow.config").read_text()
    module_text = (repo_root / "workflows" / "modules" / "segmentation.nf").read_text()

    for expected in [
        "process CELLPOSE_SEGMENT",
        "merxen cellpose-segment",
        "process PROSEG_SEGMENT",
        "merxen proseg-segment",
        "cellpose_cellprobs_tiled.npy",
        '--cellpose-cellprob "${cellpose_cellprob}"',
        "workflow SEGMENT",
    ]:
        assert expected in module_text

    cellpose_resources = config_text.split('withName: "CELLPOSE_SEGMENT"', maxsplit=1)[
        1
    ].split('withName: "PROSEG_SEGMENT"', maxsplit=1)[0]
    proseg_resources = config_text.split('withName: "PROSEG_SEGMENT"', maxsplit=1)[
        1
    ].split('withName: "ENSURE_PROSEG"', maxsplit=1)[0]

    assert "--gpus-per-node=1" in cellpose_resources
    assert "gpu_process_lock_enabled" in cellpose_resources
    assert "queue = 'htc'" in proseg_resources
    assert "gpus-per-node" not in proseg_resources
    assert "gpu_process_lock" not in proseg_resources

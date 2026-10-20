"""Workflow smoke tests for split Cellpose and ProSeg execution."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest


def test_segmentation_routes_cellpose_to_gpu_and_proseg_to_cpu(
    dwight_config_text: str,
) -> None:
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
    assert "gpu_process_lock_enabled" in dwight_config_text
    assert 'withName: "CELLPOSE_SEGMENT"' in dwight_config_text
    assert "queue = 'htc'" in proseg_resources
    assert "gpus-per-node" not in proseg_resources
    assert "gpu_process_lock" not in proseg_resources


def test_model_sidecars_publish_separately_from_segmentation_tuples() -> None:
    """Separate metadata outputs must not enter the existing ProSeg input tuple."""
    repo_root = Path(__file__).resolve().parents[2]
    module_text = (repo_root / "workflows" / "modules" / "segmentation.nf").read_text()
    main_text = (repo_root / "workflows" / "main.nf").read_text()
    assert 'path("segment_out/cellpose_model_identity.json"), emit: model_identity' in (
        module_text
    )
    assert (
        'path("segment_out/cellpose_nuclei_model_identity.json"), emit: model_identity'
        in module_text
    )
    assert "CELLPOSE_SEGMENT(segment_inputs).segmentation" in module_text
    assert "CELLPOSE_NUCLEI_SEGMENT(nuclei_segment_inputs_ch).segmentation" in main_text
    config_text = (repo_root / "workflows" / "nextflow.config").read_text()
    gpu_config = config_text.split("gpu {", 1)[1]
    for process_name in ("CELLPOSE_SEGMENT", "CELLPOSE_NUCLEI_SEGMENT"):
        block = gpu_config.split(f'withName: "{process_name}"', 1)[1].split("}", 1)[0]
        assert "--nv" in block


@pytest.mark.parametrize(
    ("process_name", "expected_hash"),
    [
        (
            "CELLPOSE_SEGMENT",
            "5d2f43c3027fb465bcaeeba2fae5fc5e28a0b1bae763aa66e4e7cb73e5162914",
        ),
        (
            "CELLPOSE_NUCLEI_SEGMENT",
            "4765231f42d0db23cb27bb7f4ce157295b54c1d75c47a3dc1eb6c375e8c0dced",
        ),
        (
            "PROSEG_SEGMENT",
            "fd0852c639c08ad80df883cd822ec1f21e3398b5be389978bb00d18a38b74d4e",
        ),
    ],
)
def test_hitl0_preserves_segmentation_process_scripts(
    process_name: str, expected_hash: str
) -> None:
    """Adding metadata outputs must preserve the pre-HITL inference commands."""
    repo_root = Path(__file__).resolve().parents[2]
    module = (repo_root / "workflows" / "modules" / "segmentation.nf").read_text()
    block = module.split(f"process {process_name} {{", 1)[1].split("\nprocess ", 1)[0]
    match = re.search(r'script:.*?"""(.*?)"""', block, re.DOTALL)
    assert match is not None
    assert hashlib.sha256(match.group(1).encode()).hexdigest() == expected_hash


@pytest.mark.parametrize(
    ("field", "expected_hash"),
    [
        (
            "cellpose",
            "79eabbc4fc2bb162e2d50b7d7bde5cfe5f81b1fdf1e6b589c7eac89016fb529a",
        ),
        (
            "nuclei_cellpose",
            "03408af57ae0d4e9a153e3df074b0b29a6347d66227f4f6db2e264ec6339a173",
        ),
    ],
)
def test_hitl0_preserves_legacy_cellpose_config_json(
    field: str, expected_hash: str
) -> None:
    """The Python model default must not add new keys to legacy Nextflow JSON."""
    repo_root = Path(__file__).resolve().parents[2]
    main_text = (repo_root / "workflows" / "main.nf").read_text()
    match = re.search(
        rf"^                {field}: \[(.*?)^                \],",
        main_text,
        re.MULTILINE | re.DOTALL,
    )
    assert match is not None
    assert hashlib.sha256(match.group(1).encode()).hexdigest() == expected_hash

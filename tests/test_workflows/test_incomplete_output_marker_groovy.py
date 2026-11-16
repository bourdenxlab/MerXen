"""The published-output routes of main.nf refuse outputs an interrupted step left.

``requireExistingPath`` reads an upstream stage's published output when that stage
is skipped (``--start_stage``). An output with a ``.merxen-incomplete`` marker
beside it (``merxen.segmentation.output_markers``) was not finished and must be
refused, not used. The helper is run in Nextflow, from the text of main.nf.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from merxen.path_utils import mark_incomplete
from merxen.segmentation.output_markers import (
    cell_mask_marker_path,
    nuclei_mask_marker_path,
    store_marker_path,
)

_HERE = Path(__file__).resolve().parent
_WORKFLOWS = _HERE.parents[1] / "workflows"


def _golden_module() -> ModuleType:
    name = "_merxen_platform_config_golden"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, _HERE / "test_platform_config_golden.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _layout(root: Path) -> dict[str, Path]:
    """Published outputs of one sample, some of them left by interrupted steps."""
    seg = root / "complete" / "segmentation"
    seg.mkdir(parents=True)
    paths = {
        "complete_mask": seg / "cellpose_masks_tiled.npy",
        "complete_nuclei": seg / "cellpose_nuclei_masks_tiled.npy",
        "complete_latest": root / "complete" / "latest" / "latest_spatialdata.zarr",
    }
    bad = root / "interrupted" / "segmentation"
    bad.mkdir(parents=True)
    paths |= {
        "interrupted_csv": bad / "transcripts_for_proseg.csv",
        "interrupted_nuclei": bad / "cellpose_nuclei_masks_tiled.npy",
        "interrupted_latest": root
        / "interrupted"
        / "latest"
        / "latest_spatialdata.zarr",
        "interrupted_latest_link": bad / "proseg_base_latest.zarr",
    }
    for key, path in paths.items():
        if key.endswith("latest"):
            path.mkdir(parents=True)
        elif not key.endswith("_link"):
            path.write_text("x")
    paths["interrupted_latest_link"].symlink_to(
        paths["interrupted_latest"], target_is_directory=True
    )
    mark_incomplete(cell_mask_marker_path(bad / "cellpose_masks_tiled.npy"), step="t")
    mark_incomplete(
        nuclei_mask_marker_path(bad / "cellpose_nuclei_masks_tiled.npy"), step="t"
    )
    mark_incomplete(store_marker_path(paths["interrupted_latest"]), step="t")
    return paths


def test_require_existing_path_refuses_marked_outputs(tmp_path: Path) -> None:
    nextflow = shutil.which("nextflow")
    if nextflow is None:
        pytest.skip("nextflow is not installed")
    golden = _golden_module()
    paths = _layout(tmp_path / "results")
    checks = "\n".join(
        f"    check({json.dumps(str(path))}, {json.dumps(key)})"
        for key, path in paths.items()
    )
    tail = f"""
def check(String path, String label) {{
    try {{
        requireExistingPath(path, label)
        println "RESULT\\t${{label}}\\tok"
    }} catch (IllegalArgumentException error) {{
        println "RESULT\\t${{label}}\\t${{error.message.readLines()[0]}}"
    }}
}}

workflow {{
{checks}
    println "DONE"
}}
"""
    work_dir = tmp_path / "harness"
    work_dir.mkdir()
    (work_dir / "h.nf").write_text("\n".join([golden.workflow_functions()[0], tail]))
    (work_dir / "lib").symlink_to(_WORKFLOWS / "lib", target_is_directory=True)
    result = subprocess.run(
        [
            nextflow,
            "run",
            "h.nf",
            "-c",
            str(_WORKFLOWS / "nextflow.config"),
            "--outdir",
            "results",
            "-ansi-log",
            "false",
        ],
        cwd=work_dir,
        capture_output=True,
        text=True,
        env=golden.hermetic_env(),
        timeout=300,
        check=False,
    )
    assert result.returncode == 0 and "DONE" in result.stdout, (
        f"{result.stdout[-3000:]}\n{result.stderr[-2000:]}"
    )
    outcome = dict(
        line.split("\t")[1:3]
        for line in result.stdout.splitlines()
        if line.startswith("RESULT\t")
    )

    assert {key: outcome[key] for key in paths if key.startswith("complete")} == {
        "complete_mask": "ok",
        "complete_nuclei": "ok",
        "complete_latest": "ok",
    }
    for key in paths:
        if key.startswith("interrupted"):
            assert outcome[key].startswith(f"Incomplete {key}:"), (key, outcome[key])

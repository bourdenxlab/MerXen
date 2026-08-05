"""Optional real-environment smoke tests for MENDER and its CPU container."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from merxen.mender_compute import run_mender_compute


def _portable_grid(path: Path) -> None:
    rows = []
    for y_index in range(6):
        for x_index in range(6):
            rows.append(
                {
                    "cell_id": f"cell_{x_index}_{y_index}",
                    "native_x": float(x_index * 10),
                    "native_y": float(y_index * 10),
                    "cell_state": "left" if x_index < 3 else "right",
                }
            )
    frame = pd.DataFrame(rows)
    frame["cell_state"] = pd.Categorical(frame["cell_state"])
    frame.to_parquet(path, index=False)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("clustering_mode", "leiden_resolution", "target_k"),
    [
        pytest.param("resolution", 1.5, None, id="resolution-1.5"),
        pytest.param("target_k", 1.5, 2, id="target-k-2"),
    ],
)
def test_real_mender_synthetic_grid_smoke(
    tmp_path: Path,
    clustering_mode: str,
    leiden_resolution: float,
    target_k: int | None,
) -> None:
    """Run the pinned MENDER package when this test uses its dedicated env."""
    pytest.importorskip("MENDER", reason="run inside environment.mender.yml")
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    _portable_grid(prepared / "mender_input.parquet")
    config = {
        "sample_id": "synthetic_MERSCOPE",
        "platform": "MERSCOPE",
        "segmentation": "proseg_hybrid",
        "random_seed": 666,
        "nn_mode": "radius",
        "radius_um": 15.0,
        "n_scales": 6,
        "count_rep": "s",
        "include_self": True,
        "clustering_mode": clustering_mode,
        "leiden_resolution": leiden_resolution,
        "target_k": target_k,
        "run_umap": True,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    outputs = run_mender_compute(config_path, prepared, tmp_path / "computed")
    assert outputs["context_h5ad"].exists()
    domains = pd.read_parquet(outputs["domains"])
    assert len(domains) == 36
    assert domains["cell_id"].is_unique
    assert domains["mender_domain"].notna().all()
    assert domains["mender_domain"].nunique() >= 2
    manifest = json.loads(outputs["manifest"].read_text())
    expected_request = -1.5 if clustering_mode == "resolution" else 2
    assert manifest["clustering_request"] == expected_request


@pytest.mark.slow
def test_cpu_only_apptainer_mender_smoke() -> None:
    """Import MENDER in a configured SIF without exposing NVIDIA devices."""
    apptainer = shutil.which("apptainer")
    container = os.environ.get("MERXEN_MENDER_CONTAINER")
    if apptainer is None or not container:
        pytest.skip("set MERXEN_MENDER_CONTAINER on a host with Apptainer")
    completed = subprocess.run(
        [
            apptainer,
            "exec",
            "--cleanenv",
            "--env",
            "CUDA_VISIBLE_DEVICES=",
            container,
            "python",
            "-c",
            (
                "import os, MENDER; "
                "assert os.environ.get('CUDA_VISIBLE_DEVICES', '') == ''"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr

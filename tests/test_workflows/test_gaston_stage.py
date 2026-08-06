"""Static workflow contracts for the GASTON terminal stage."""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_gaston_defaults_and_independent_segmentation_selector() -> None:
    config = (REPO_ROOT / "workflows" / "nextflow.config").read_text()
    main = (REPO_ROOT / "workflows" / "main.nf").read_text()
    assert 'analysis_segmentation = "both"' in config
    assert 'gaston_segmentations = "proseg_hybrid"' in config
    assert 'return selected ?: ["proseg_hybrid"]' in main
    assert '"all": ["reseg", "original_seg", "proseg_mask", "proseg_hybrid"]' in main
    assert 'raw.split(",")' in main
    assert "requiredClusteringSegmentations = (" in main
    assert "analysisSegmentations + gastonSegmentations" in main


def test_gaston_enable_disable_row_overrides_and_stage_extension() -> None:
    main = (REPO_ROOT / "workflows" / "main.nf").read_text()
    assert 'rowFieldOrDefault(row, "gaston_enabled", params.gaston_enabled)' in main
    assert 'rowFieldOrDefault(row, "gaston_epochs", params.gaston_epochs)' in main
    assert 'rowFieldOrDefault(row, "gaston_use_gpu", params.gaston_use_gpu)' in main
    assert (
        'rowFieldOrDefault(\n            row,\n            "gaston_segmentations"'
        in main
    )
    assert 'stopStage = "gaston"' in main
    assert "autoExtendedToGaston = true" in main
    assert "runMapMyCells = false" in main
    assert 'stages += ["gaston"]' in main


def test_gaston_extra_segmentations_do_not_activate_unrelated_branches() -> None:
    main = (REPO_ROOT / "workflows" / "main.nf").read_text()
    assert "settings.analysis_layer_segmentations.collect" in main
    assert "settings.analysis_segmentations.contains(_segmentation)" in main
    assert "!settings.analysis_segmentations.contains(_segmentation)" in main
    assert "settings.required_clustering_segmentations" in main


def test_gaston_published_output_preflight_is_exact() -> None:
    main = (REPO_ROOT / "workflows" / "main.nf").read_text()
    assert "settings.gaston_published_output_mode" in main
    assert "Missing GASTON clustered H5AD for" in main
    assert "Missing GASTON latest SpatialData Zarr for" in main
    assert "Missing GASTON clustered SpatialData table" in main
    assert "gaston_published_artifacts_ch" in main
    assert "gaston_bypass_terminal_gate" in main


def test_gaston_per_pair_gate_has_known_cardinality_and_excludes_cohort() -> None:
    main = (REPO_ROOT / "workflows" / "main.nf").read_text()
    assert "currentPairTerminalStage" in main
    assert "pair_terminal_token_ch" in main
    assert "groupKey(pairId, expectedCount as int)" in main
    assert "DISTANCE_FROM_OBJECT_COHORT is intentionally outside this barrier" in main
    assert "settings.active_platforms.size() + extraClustering.size()" in main
    assert 'terminalStage != "clustering_squidpy"' in main
    assert ".join(pair_terminal_token_ch)" in main


def test_gaston_process_graph_scatter_and_resume_contract() -> None:
    main = (REPO_ROOT / "workflows" / "main.nf").read_text()
    module = (REPO_ROOT / "workflows" / "modules" / "gaston.nf").read_text()
    assert "GASTON_PREPARE(gaston_inputs_ch)" in main
    assert "GASTON_GLM_PCA(gaston_prepared_ch)" in main
    assert "GASTON_TRAIN(gaston_train_inputs_ch)" in main
    assert "GASTON_POSTPROCESS(gaston_postprocess_inputs_ch)" in main
    assert "GASTON_IMPORT(gaston_import_inputs_ch)" in main
    assert "gastonTrainingConfigJson(" in main
    assert 'trainingGaston.domain_mode = "auto"' in main
    assert "trainingGaston.num_domains = null" in main
    assert "gaston_postprocess_config_ch" in main
    assert 'path("gaston_postprocess_config.json")' in module
    assert "groupKey(branchKey, nRestarts as int)" in main
    assert "(0..<nRestarts).collect { seed ->" in main
    assert 'path("seed_${seed}")' in module
    assert module.count('export PYTHONPATH="${projectDir}/../src:') == 5
    assert "from gaston import neural_net" in module
    assert "torch.cuda.is_available()" in module
    assert "from gaston import dp_related, model_selection" in module


def test_gaston_resources_and_apptainer_cuda_scope() -> None:
    config = (REPO_ROOT / "workflows" / "nextflow.config").read_text()
    dwight = (REPO_ROOT / "workflows" / "conf" / "dwight.config").read_text()
    assert 'withName: "GASTON_PREPARE"' in config
    assert 'memory = "64 GB"' in config
    assert 'withName: "GASTON_GLM_PCA"' in config
    assert 'memory = "160 GB"' in config
    assert 'withName: "GASTON_TRAIN"' in config
    assert 'memory = "48 GB"' in config
    assert (
        'gaston_container = "file:///nfsdata/apptainer/merxen_gaston_79577c4.sif"'
    ) in config
    assert (
        "containerOptions = '-B /data,/nfsdata -C --no-home --home $PWD --nv'" in config
    )
    assert "gaston_train_max_forks = 1" in dwight
    assert "params.gpu_process_lock_file" in dwight


def test_gaston_environment_is_pinned_and_separate() -> None:
    environment = (REPO_ROOT / "environment.gaston.yml").read_text()
    dockerfile = (REPO_ROOT / "Dockerfile.gaston").read_text()
    base_environment = (REPO_ROOT / "environment.yml").read_text()
    for dependency in ("glmpca", "kneed", "scanpy", "squidpy"):
        assert dependency in environment.lower()
    assert "79577c45111b3442b808c8e620711b822965493b" in environment
    assert "79577c45111b3442b808c8e620711b822965493b" in dockerfile
    assert "cuda-version=11.8" in environment
    assert "mkl=2024.0.0" in environment
    assert "setuptools=80.9.0" in environment
    assert "dask[array,dataframe]>=2024.4.1,<2025" in environment
    assert "from gaston import dp_related, model_selection, neural_net" in dockerfile
    assert "gaston-spatial @ git+https://github.com/raphael-group/GASTON.git@" in (
        environment
    )
    assert "gaston-spatial @ git+https://github.com/raphael-group/GASTON.git@" in (
        dockerfile
    )
    assert "gaston @ git+https://github.com/raphael-group/GASTON.git@" not in (
        environment
    )
    assert "gaston @ git+https://github.com/raphael-group/GASTON.git@" not in (
        dockerfile
    )
    assert "gaston" not in base_environment.lower()


@pytest.mark.parametrize(
    ("selector", "expected_segmentations"),
    [
        (None, ["proseg_hybrid"]),
        ("reseg,proseg_mask", ["reseg", "proseg_mask"]),
        (
            "all",
            ["reseg", "original_seg", "proseg_mask", "proseg_hybrid"],
        ),
    ],
)
def test_gaston_only_stage_preview_normalizes_segmentations_and_row_overrides(
    tmp_path: Path,
    selector: str | None,
    expected_segmentations: list[str],
) -> None:
    """Compile published-output mode with only the selected artifacts present."""
    if shutil.which("nextflow") is None:
        pytest.skip("nextflow is not installed")
    outdir = tmp_path / "results"
    pair_id = "GASTON_PREVIEW"
    latest_zarr = outdir / pair_id / "xenium" / "latest" / "latest_spatialdata.zarr"
    table_names = {
        "reseg": "table_MOSAIK_proseg_clustering_squidpy",
        "original_seg": "table_original_clustering_squidpy",
        "proseg_mask": "table_MOSAIK_cellpose_clustering_squidpy",
        "proseg_hybrid": "table_MOSAIK_proseg_hybrid_clustering_squidpy",
    }
    for segmentation in expected_segmentations:
        h5ad = (
            outdir
            / pair_id
            / segmentation
            / "clustering_squidpy"
            / "clustering_squidpy_out"
            / "xenium"
            / f"{pair_id}_XENIUM_clustered.h5ad"
        )
        h5ad.parent.mkdir(parents=True, exist_ok=True)
        h5ad.touch()
        (latest_zarr / "tables" / table_names[segmentation]).mkdir(
            parents=True,
            exist_ok=True,
        )

    samplesheet = tmp_path / "samplesheet.csv"
    fieldnames = [
        "pair_id",
        "analysis_mode",
        "only_stage",
        "gaston_enabled",
        "gaston_segmentations",
        "xenium_spatialdata_path",
    ]
    with samplesheet.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "pair_id": pair_id,
                "analysis_mode": "xenium",
                "only_stage": "gaston",
                "gaston_enabled": "true",
                "gaston_segmentations": selector or "",
                "xenium_spatialdata_path": str(latest_zarr),
            }
        )
    result = subprocess.run(
        [
            "nextflow",
            "run",
            str(REPO_ROOT / "workflows" / "main.nf"),
            "-c",
            str(REPO_ROOT / "workflows" / "nextflow.config"),
            "-profile",
            "conda",
            "-preview",
            "--samplesheet",
            str(samplesheet),
            "--outdir",
            str(outdir),
            "--gaston_enabled",
            "false",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.slow
def test_gaston_apptainer_import_and_cuda_smoke() -> None:
    """Validate the built site image when it is available on a GPU host."""
    if shutil.which("apptainer") is None:
        pytest.skip("apptainer is not installed")
    image_value = os.environ.get(
        "MERXEN_GASTON_APPTAINER_IMAGE",
        "/nfsdata/apptainer/merxen_gaston_79577c4.sif",
    )
    image = Path(image_value.removeprefix("file://"))
    if not image.is_file():
        pytest.skip(f"GASTON Apptainer image is unavailable: {image}")
    result = subprocess.run(
        [
            "apptainer",
            "exec",
            "--nv",
            str(image),
            "python3",
            "-c",
            (
                "import gaston, glmpca, kneed, scanpy, squidpy, torch; "
                "assert torch.cuda.is_available(); "
                "print(torch.cuda.get_device_name(torch.cuda.current_device()))"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr

"""Behavioral contracts for the optional pciSeq workflow stage."""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_LAYERS = {
    "reseg": ("table_MOSAIK_proseg", "MOSAIK_proseg"),
    "original_seg": ("table_original", None),
    "proseg_mask": ("table_MOSAIK_cellpose", "MOSAIK_cellpose"),
    "proseg_hybrid": ("table_MOSAIK_proseg_hybrid", "MOSAIK_proseg_hybrid"),
}


def _write_samplesheet(path: Path, rows: list[dict[str, str]]) -> None:
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _run_nextflow(tmp_path: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    if shutil.which("nextflow") is None:
        pytest.skip("nextflow is not installed")
    environment = os.environ.copy()
    environment["NXF_ANSI_LOG"] = "false"
    return subprocess.run(
        [
            "nextflow",
            "run",
            str(REPO_ROOT / "workflows" / "main.nf"),
            "-profile",
            "conda",
            "-without-conda",
            "--samplesheet",
            str(tmp_path / "samplesheet.csv"),
            "--outdir",
            str(tmp_path / "results"),
            *arguments,
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


def _published_input(
    outdir: Path, pair_id: str, platform: str, segmentation: str, zarr: Path
) -> None:
    h5ad = (
        outdir
        / pair_id
        / segmentation
        / "clustering_squidpy"
        / "clustering_squidpy_out"
        / platform.lower()
        / f"{pair_id}_{platform}_clustered.h5ad"
    )
    h5ad.parent.mkdir(parents=True, exist_ok=True)
    h5ad.touch()
    table, shape = _LAYERS[segmentation]
    (zarr / "tables" / table).mkdir(parents=True, exist_ok=True)
    if shape is None:
        shape = (
            "merscope_cell_boundaries"
            if platform == "MERSCOPE"
            else "xenium_cell_boundaries"
        )
    (zarr / "shapes" / shape).mkdir(parents=True, exist_ok=True)


def test_pciseq_is_optional_and_immediately_follows_clustering() -> None:
    """Only the compute task receives the separate pciSeq environment."""
    main = (REPO_ROOT / "workflows" / "main.nf").read_text()
    config = (REPO_ROOT / "workflows" / "nextflow.config").read_text()
    module = (REPO_ROOT / "workflows" / "modules" / "pciseq.nf").read_text()
    assert "pciseq_enabled = false" in config
    assert 'pciseq_overlap_policy = "background"' in config
    assert "overlap_policy: params.pciseq_overlap_policy" in main
    assert re.search(
        r'stages \+= \["clustering_squidpy"\]\s*'
        r'if \(pciseqEnabled\) \{\s*stages \+= \["pciseq"\]',
        main,
    )
    compute = config.split('withName: "PCISEQ_COMPUTE"', 1)[1].split(
        'withName: "PCISEQ_FINALIZE"', 1
    )[0]
    assert "conda = params.pciseq_conda" in compute
    assert "container = params.pciseq_container" in compute
    assert 'export CUDA_VISIBLE_DEVICES=""' in module
    assert "--prepared-dir" in module
    assert "--computed-dir" in module


def test_downstream_stages_wait_for_every_pciseq_finalization() -> None:
    """Both fresh clustering and standalone restarts share the completion gate."""
    main = (REPO_ROOT / "workflows" / "main.nf").read_text()
    gate = main[
        main.index("pciseq_completion_expectations_ch =") : main.index(
            "// Cortical depth runs after clustering so"
        )
    ]
    assert (
        "settings.active_platforms.size() * settings.analysis_segmentations.size()"
        in gate
    )
    assert "groupKey(pairId, expectedCount as int)" in gate
    assert ".join(pciseq_done_per_pair_ch)" in gate
    assert ".combine(post_pciseq_pair_gate_ch, by: 0)" in gate
    downstream = main[main.index("// Cortical depth runs after clustering so") :]
    assert "= sample_rows_ch" not in downstream
    assert "= clustering_results_ch" not in downstream
    assert (
        "mapmycells_from_clustering_ch = clustering_downstream_results_ch" in downstream
    )
    assert (
        "gaston_current_artifacts_ch = clustering_downstream_results_ch" in downstream
    )


@pytest.mark.nextflow
def test_pciseq_completion_gate_keeps_partial_pairs_blocked(tmp_path: Path) -> None:
    """One pair's partial output must neither release itself nor block another."""
    if shutil.which("nextflow") is None:
        pytest.skip("nextflow is not installed")
    main = (REPO_ROOT / "workflows" / "main.nf").read_text()
    gate = main[
        main.index("    pciseq_completion_expectations_ch =") : main.index(
            "    // Cortical depth runs after clustering so"
        )
    ]
    harness = """
nextflow.enable.dsl = 2
workflow {
    def gateSettings = [run_pciseq: true, active_platforms: ['MERSCOPE', 'XENIUM'],
        analysis_segmentations: ['reseg', 'original_seg']]
    sample_rows_ch = Channel.of(tuple('READY', [:], gateSettings),
        tuple('PARTIAL', [:], gateSettings),
        tuple('DISABLED', [:], gateSettings + [run_pciseq: false]))
    pciseq_results_ch = Channel.fromList(
        ['reseg', 'original_seg'].collectMany { seg ->
            ['MERSCOPE', 'XENIUM'].collect { platform ->
                tuple('READY', seg, platform, 'output')
            }
        } + [tuple('PARTIAL', 'reseg', 'XENIUM', 'output')]
    )
    clustering_results_ch = Channel.fromList(
        ['READY', 'PARTIAL', 'DISABLED'].collectMany { pair ->
            ['reseg', 'original_seg'].collect { seg ->
                tuple(pair, seg, '[]', 'clustered')
            }
        }
    )
"""
    assertions = """
    post_pciseq_sample_rows_ch.map { pair, row, settings -> pair }.collect().view {
        pairs -> new File('released.json').text =
            groovy.json.JsonOutput.toJson(pairs.sort())
    }
    clustering_downstream_results_ch.map { pair, seg, samples, out -> "${pair}|${seg}" }
        .collect().view { branches ->
            new File('branches.json').text =
                groovy.json.JsonOutput.toJson(branches.sort())
        }
}
"""
    (tmp_path / "gate.nf").write_text(harness + gate + assertions)

    result = subprocess.run(
        ["nextflow", "run", "gate.nf", "-ansi-log", "false"],
        cwd=tmp_path,
        env={**os.environ, "NXF_OFFLINE": "true"},
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads((tmp_path / "released.json").read_text()) == [
        "DISABLED",
        "READY",
    ]
    assert json.loads((tmp_path / "branches.json").read_text()) == [
        "DISABLED|original_seg",
        "DISABLED|reseg",
        "READY|original_seg",
        "READY|reseg",
    ]


@pytest.mark.nextflow
def test_pciseq_standalone_runs_each_platform_and_segmentation(tmp_path: Path) -> None:
    """Published inputs yield eight independent tasks and separate output paths."""
    outdir = tmp_path / "results"
    reference = tmp_path / "reference.csv"
    reference.write_text("gene,type_a\nG1,1\n")
    row = {
        "pair_id": "PCI_PAIR",
        "analysis_mode": "paired",
        "analysis_segmentation": "all",
        "only_stage": "pci_seq",
        "pciseq_enabled": "true",
        "pciseq_reference_path": str(reference),
    }
    for platform in ("MERSCOPE", "XENIUM"):
        # Explicit overrides prove standalone mode does not require a latest
        # symlink when the source store is supplied in the samplesheet.
        zarr = tmp_path / f"{platform.lower()}_input.zarr"
        row[f"{platform.lower()}_spatialdata_path"] = str(zarr)
        for segmentation in _LAYERS:
            _published_input(outdir, row["pair_id"], platform, segmentation, zarr)
    _write_samplesheet(tmp_path / "samplesheet.csv", [row])

    result = _run_nextflow(tmp_path, "-stub-run", "--pciseq_enabled", "false")

    assert result.returncode == 0, result.stdout + result.stderr
    manifests = list(outdir.glob("PCI_PAIR/*/pciseq/pciseq_out/*/stub_manifest.json"))
    assert len(manifests) == 8
    trace = (outdir / "nextflow" / "trace.tsv").read_text()
    for process in ("PCISEQ_PREPARE", "PCISEQ_COMPUTE", "PCISEQ_FINALIZE"):
        assert len(re.findall(rf"\t{process}\t", trace)) == 8
    run_log = (tmp_path / ".nextflow.log").read_text()
    assert "selected stages=pciseq" in run_log
    assert "submitted process > CLUSTERING_SQUIDPY" not in run_log


@pytest.mark.nextflow
def test_pciseq_standalone_preflight_reports_missing_native_shape(
    tmp_path: Path,
) -> None:
    """A cached clustered H5AD alone cannot replace the native seed geometry."""
    reference = tmp_path / "reference.csv"
    reference.write_text("gene,type_a\nG1,1\n")
    zarr = tmp_path / "input.zarr"
    _published_input(tmp_path / "results", "PCI_SHAPE", "XENIUM", "proseg_mask", zarr)
    (zarr / "shapes" / "MOSAIK_cellpose").rmdir()
    _write_samplesheet(
        tmp_path / "samplesheet.csv",
        [
            {
                "pair_id": "PCI_SHAPE",
                "analysis_mode": "xenium",
                "analysis_segmentation": "cellpose",
                "only_stage": "pciseq",
                "pciseq_enabled": "true",
                "xenium_spatialdata_path": str(zarr),
            }
        ],
    )

    result = _run_nextflow(tmp_path, "--pciseq_reference_path", str(reference))

    assert result.returncode != 0
    log = (tmp_path / ".nextflow.log").read_text()
    assert (
        "Missing PCISEQ native shapes 'MOSAIK_cellpose' "
        "for PCI_SHAPE:XENIUM:proseg_mask" in log
    )
    assert "Missing PCISEQ clustered H5AD" not in log
    assert not list((tmp_path / "work").glob("*/*"))


@pytest.mark.nextflow
def test_pciseq_disabled_only_stage_explains_how_to_enable(tmp_path: Path) -> None:
    """Selecting an inactive optional stage fails before any tasks start."""
    _write_samplesheet(
        tmp_path / "samplesheet.csv",
        [
            {
                "pair_id": "PCI_DISABLED",
                "analysis_mode": "xenium",
                "only_stage": "pciseq",
            }
        ],
    )

    result = _run_nextflow(tmp_path)

    assert result.returncode != 0
    assert "Pass --pciseq_enabled true" in result.stdout + result.stderr
    assert not list((tmp_path / "work").glob("*/*"))


@pytest.mark.nextflow
def test_pciseq_default_stop_extension_and_reference_preflight(tmp_path: Path) -> None:
    """Enabling pciSeq extends the default stop while row false remains exact."""
    _write_samplesheet(
        tmp_path / "samplesheet.csv",
        [
            {
                "pair_id": "PCI_OFF",
                "analysis_mode": "xenium",
                "start_stage": "clustering_squidpy",
                "pciseq_enabled": "false",
            },
            {
                "pair_id": "PCI_ON",
                "analysis_mode": "xenium",
                "start_stage": "clustering_squidpy",
                "pciseq_enabled": "true",
            },
        ],
    )

    result = _run_nextflow(
        tmp_path, "--clustering_squidpy_hierarchical_enabled", "false"
    )

    assert result.returncode != 0
    log = (tmp_path / ".nextflow.log").read_text()
    stages = dict(
        re.findall(r"Sample (PCI_\w+): .*; selected stages=(.*)$", log, re.MULTILINE)
    )
    assert stages == {
        "PCI_OFF": "clustering_squidpy",
        "PCI_ON": "clustering_squidpy -> pciseq",
    }
    assert (
        "Missing required path parameter for PCISEQ reference expression matrix "
        "for PCI_ON" in log
    )
    assert not list((tmp_path / "work").glob("*/*"))

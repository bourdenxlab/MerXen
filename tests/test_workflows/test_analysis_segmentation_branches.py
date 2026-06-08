"""Workflow text checks for analysis segmentation branch wiring."""

from __future__ import annotations

from pathlib import Path


def test_nextflow_exposes_analysis_segmentation_branches() -> None:
    """The pipeline should route downstream modules through segmentation branches."""
    repo_root = Path(__file__).resolve().parents[2]
    main_text = (repo_root / "workflows" / "main.nf").read_text()
    config_text = (repo_root / "workflows" / "nextflow.config").read_text()

    for expected in [
        'analysis_segmentation = "both"',
        "normalizeAnalysisSegmentation",
        '"table_MOSAIK_proseg"',
        '"table_original"',
        '"merscope_cell_boundaries"',
        '"xenium_cell_boundaries"',
        "samplesJsonForSegmentation",
    ]:
        assert expected in main_text or expected in config_text


def test_downstream_modules_publish_under_segmentation_branch() -> None:
    """Branch-specific downstream modules should include segmentation in outputs."""
    repo_root = Path(__file__).resolve().parents[2]
    module_dir = repo_root / "workflows" / "modules"

    expectations = {
        "qc.nf": "/${segmentation}/qc",
        "comparison.nf": "/${segmentation}/comparison",
        "visualization.nf": "/${segmentation}/visualization",
        "clustering_squidpy.nf": "/${segmentation}/clustering_squidpy",
        "mapmycells.nf": "/${segmentation}/mapmycells",
    }
    for filename, expected in expectations.items():
        assert expected in (module_dir / filename).read_text()

"""Workflow text checks for the distance-from-object stage."""

from __future__ import annotations

from pathlib import Path


def test_distance_from_object_processes_and_three_branches_are_wired() -> None:
    """Nextflow should annotate samples then aggregate cohorts by platform."""
    repo_root = Path(__file__).resolve().parents[2]
    main_text = (repo_root / "workflows" / "main.nf").read_text()
    config_text = (repo_root / "workflows" / "nextflow.config").read_text()
    module_text = (
        repo_root / "workflows" / "modules" / "distance_from_object.nf"
    ).read_text()

    for expected in [
        "DISTANCE_FROM_OBJECT_ANNOTATE",
        "DISTANCE_FROM_OBJECT_COHORT",
        '"table_MOSAIK_proseg"',
        '"table_original"',
        '"table_MOSAIK_cellpose"',
        '"cellpose"',
        "settings.distance_from_object_segmentations",
        "distance_from_object_after_cortical_ch",
        "distance_from_object_annotation_results_ch",
        ".groupTuple()",
        "appendDistanceFromObjectPreflightChecks",
        "distanceFromObjectAnnotationPath",
    ]:
        assert expected in main_text

    for expected in [
        "distance_from_object_enabled = false",
        'distance_from_object_segmentations = ["proseg", "original", "cellpose"]',
        "distance_from_object_near_distance_um = 50.0",
        "distance_from_object_far_distance_um = 100.0",
        "distance_from_object_max_distance_um = 200.0",
        'withName: "DISTANCE_FROM_OBJECT_ANNOTATE"',
        'withName: "DISTANCE_FROM_OBJECT_COHORT"',
    ]:
        assert expected in config_text

    for expected in [
        "process DISTANCE_FROM_OBJECT_ANNOTATE",
        "merxen distance-from-object --config",
        "process DISTANCE_FROM_OBJECT_COHORT",
        "merxen distance-from-object-cohort --config",
        'stageAs: "pair_outputs/dir??/*"',
    ]:
        assert expected in module_text


def test_cortical_depth_expands_to_distance_tables_when_both_stages_run() -> None:
    """Tissue labels must be written for every requested distance table."""
    repo_root = Path(__file__).resolve().parents[2]
    main_text = (repo_root / "workflows" / "main.nf").read_text()

    expected_union = (
        "corticalSegmentations + settings.distance_from_object_segmentations"
    )
    assert main_text.count(expected_union) == 2
    assert "compute_cortical_depth_results_ch = COMPUTE_CORTICAL_DEPTH" in main_text
    assert "compute_cortical_depth_results_ch" in main_text

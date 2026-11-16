"""Workflow smoke tests for default VALIS and legacy alignment wiring."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


def test_alignment_workflow_defaults_to_valis_and_preserves_legacy_backend() -> None:
    """Nextflow must select VALIS by default without deleting legacy settings."""
    repo_root = Path(__file__).resolve().parents[2]
    module_text = (repo_root / "workflows" / "modules" / "alignment.nf").read_text()
    config_text = (repo_root / "workflows" / "nextflow.config").read_text()

    for expected in [
        'alignment_backend = "valis"',
        '"backend": "${params.alignment_backend}"',
        '"fixed_platform"',
        '"moving_platform"',
        '"merscope_image"',
        '"xenium_image"',
        '"tissue_annotation_path"',
        (
            "path(merscope_tissue_annotation, stageAs: "
            '"merscope_tissue_annotation.geojson")'
        ),
        'path(xenium_tissue_annotation, stageAs: "xenium_tissue_annotation.geojson")',
        '"valis"',
        '"legacy_spateo"',
        '"resume": ${params.alignment_resume}',
        'path("align_out")',
        "check-alignment-deps --backend valis",
        "check-alignment-deps --backend legacy_spateo",
        '"valis-wsi==1.2.0"',
        "alignment_allow_reflection = true",
        'alignment_reflection_mode = "auto"',
        "alignment_reflection_minimum_score_improvement = 0.01",
        "alignment_orientation_translation_candidates_per_angle = 3",
        "alignment_orientation_local_fine_search_enabled = true",
        "alignment_orientation_local_fine_angle_radius_degrees = 2.5",
        "alignment_orientation_local_fine_translation_radius_um = 500.0",
        '"reflection_mode": "${params.alignment_reflection_mode}"',
        '"local_fine_search_enabled"',
        '"initial_angle_degrees"',
        'alignment_background_boundary_mode = "mirror"',
        "alignment_partial_overlap_enabled = true",
        "alignment_edge_taper_um = 150.0",
        '"partial_overlap"',
        'alignment_valis_global_transform = "rigid"',
        'alignment_valis_non_rigid_backend = "optical_flow"',
        "alignment_qc_non_rigid_minimum_nmi_improvement = 0.0",
        "alignment_qc_non_rigid_maximum_coherent_rotation_degrees = 0.25",
        "alignment_qc_non_rigid_maximum_coherent_translation_um = 25.0",
        "process MATERIALIZE_ALIGNMENT",
        "cache false",
        "materialize-alignment",
        "--summary materialization_summary.json",
        "alignment_materialization_enabled = true",
        "alignment_materialization_max_forks = 1",
        'withName: "MATERIALIZE_ALIGNMENT"',
        'memory = "120 GB"',
    ]:
        assert expected in module_text or expected in config_text

    main_text = (repo_root / "workflows" / "main.nf").read_text()
    assert 'alignOut.resolve("alignment_transform.json")' in main_text
    assert 'alignOut.resolve("alignment_coords")' in main_text
    assert "appendAlignmentAnnotationPreflightChecks" in main_text
    assert "alignmentTissueAnnotationPath(row, platform)" in main_text
    assert "settings.alignment_annotation_paths.MERSCOPE" in main_text
    assert "settings.alignment_annotation_paths.XENIUM" in main_text
    assert "tuple(pairId, merscopeLatest, xeniumLatest, alignOut)" in main_text
    assert "MATERIALIZE_ALIGNMENT(alignment_registration_results_ch)" in main_text
    assert (
        '"transform_json_path": "${align_out}/alignment_transform.json"' in module_text
    )


def test_alignment_environment_uses_the_generated_runtime_lock() -> None:
    """The isolated stage environment must consume the VALIS runtime lock."""
    repo_root = Path(__file__).resolve().parents[2]
    environment_text = (repo_root / "envs" / "environment.alignment.yml").read_text()
    lock_text = (repo_root / "requirements" / "requirements.alignment.lock").read_text()
    config_text = (repo_root / "workflows" / "nextflow.config").read_text()

    assert "-r ../requirements/requirements.alignment.lock" in environment_text
    # The lock's checksum in the env file's text makes Nextflow rebuild the
    # cached ALIGN env whenever the lock changes.
    header = re.search(
        r"^# requirements.lock sha256: ([0-9a-f]{64})$", environment_text, re.M
    )
    assert header is not None
    lock_path = repo_root / "requirements" / "requirements.alignment.lock"
    assert header.group(1) == hashlib.sha256(lock_path.read_bytes()).hexdigest()
    # The env is Python 3.12; a lock resolved for a newer Python pins wheels
    # pip cannot install there (sphinx 9.2 needs 3.13).
    assert "python=3.12" in environment_text
    assert "--python-version 3.12" in lock_text.splitlines()[1]
    # pip installs the lock and rejects a git package pinned to a commit when
    # its dependent asks for another URL (ResolutionImpossible), so the lock
    # leaves the git packages out and pip takes them from their dependents.
    for package in (
        "cell-type-mapper",
        "mmc-gene-mapper",
        "abc-atlas-access",
        "bkbit",
        "schemasheets",
    ):
        assert f"--no-emit-package {package}" in lock_text.splitlines()[1]
    assert " @ git+" not in lock_text
    # The PyTorch advisories fixed in 2.13: same torch series as the base lock.
    assert re.search(r"^torch==2\.14\.\d+$", lock_text, re.M)
    assert "openjdk=11" in environment_text
    assert "libvips>=8.11" in environment_text
    assert "numpy==2.2.6" in lock_text
    assert "opencv-contrib-python-headless==4.12.0.88" in lock_text
    assert "conda.enabled = true" in config_text
    assert "conda = params.alignment_conda" in config_text

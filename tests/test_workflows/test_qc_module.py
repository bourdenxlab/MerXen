"""Workflow text checks for the QC registration guard."""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_qc_process_passes_registration_settings() -> None:
    """QC must compare against the platform's cells and honour the strict flag."""
    qc_text = (_REPO_ROOT / "workflows" / "modules" / "qc.nf").read_text()

    assert '"registration_reference_shape_key": "${registrationReferenceShapeKey}"' in (
        qc_text
    )
    assert '"registration_strict": ${registrationStrict}' in qc_text
    assert '"merscope_cell_boundaries"' in qc_text
    assert '"xenium_cell_boundaries"' in qc_text
    assert "params.qc_registration_strict" in qc_text


def test_registration_strict_defaults_to_warning_only() -> None:
    config_text = (_REPO_ROOT / "workflows" / "nextflow.config").read_text()

    assert "qc_registration_strict = false" in config_text

"""Workflow text checks for the SpatialData build configuration."""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_transform_override_mismatch_opt_out_defaults_off() -> None:
    config_text = (_REPO_ROOT / "workflows" / "nextflow.config").read_text()

    assert "merscope_allow_transform_override_mismatch = false" in config_text


def test_transform_override_mismatch_opt_out_only_set_when_requested() -> None:
    """The default MERSCOPE build JSON (and so its task hash) must not change."""
    main_text = (_REPO_ROOT / "workflows" / "main.nf").read_text()
    build_config = main_text[
        main_text.index("def buildConfigForPlatform(") : main_text.index(
            'if (platform == "XENIUM")',
            main_text.index("def buildConfigForPlatform("),
        )
    ]

    assert "params.merscope_allow_transform_override_mismatch" in build_config
    assert "merscopeOptions.allow_transform_override_mismatch = true" in build_config
    assert "merscope: merscopeOptions," in build_config

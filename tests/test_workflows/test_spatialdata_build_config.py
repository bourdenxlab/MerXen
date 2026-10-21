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


def _main_text() -> str:
    return (_REPO_ROOT / "workflows" / "main.nf").read_text()


def _function_text(main_text: str, name: str) -> str:
    """Return the source of one top-level Groovy ``def`` up to the next one."""
    start = main_text.index(f"def {name}(")
    end = main_text.find("\ndef ", start + 1)
    return main_text[start:end]


def test_bead_removal_params_default_off() -> None:
    config_text = (_REPO_ROOT / "workflows" / "nextflow.config").read_text()

    assert "merscope_bead_removal = false" in config_text
    assert "merscope_bead_removal_fill = false" in config_text


def test_bead_removal_keys_only_set_inside_the_requested_block() -> None:
    """Default MERSCOPE build JSONs (and task hashes) must not gain bead keys."""
    build_config = _function_text(_main_text(), "buildConfigForPlatform")
    block_start = build_config.index("if (merscopeBeadRemovalRequested(")
    block_end = build_config.index("return [", block_start)
    bead_block = build_config[block_start:block_end]

    assert build_config.count("merscopeOptions.bead_removal") == 1
    assert "merscopeOptions.bead_removal = beadRemoval" in bead_block
    assert "merscope: merscopeOptions," in build_config
    assert "bead_removal" not in build_config[:block_start]
    assert "bead_removal" not in build_config[block_end:].split("merscope:")[0]


def test_bead_removal_fill_key_is_nested_inside_the_bead_block() -> None:
    build_config = _function_text(_main_text(), "buildConfigForPlatform")
    block_start = build_config.index("if (merscopeBeadRemovalRequested(")
    block_end = build_config.index("return [", block_start)
    bead_block = build_config[block_start:block_end]

    assert "[enabled: true]" in bead_block
    assert "params.merscope_bead_removal_fill" in bead_block
    assert "beadRemoval.fill = true" in bead_block
    assert build_config.count("fill") == bead_block.count("fill")


def test_bead_removal_helper_reads_row_then_param_as_a_boolean() -> None:
    helper = _function_text(_main_text(), "merscopeBeadRemovalRequested")

    assert (
        'rowFieldOrDefault(row, "merscope_bead_removal", params.merscope_bead_removal)'
        in helper
    )
    assert "boolOrDefault(" in helper
    assert "false," in helper


def test_bead_removal_preflight_is_called_and_checks_planes_and_input_folder() -> None:
    main_text = _main_text()
    run_checks = _function_text(main_text, "runPreflightChecks")
    preflight = _function_text(main_text, "appendMerscopeBeadRemovalPreflightChecks")

    assert (
        "appendMerscopeBeadRemovalPreflightChecks(errors, row, settings, params)"
        in run_checks
    )
    assert "merscopeBeadRemovalRequested(row, settings.pair_id, params)" in preflight
    assert '"MERSCOPE" in settings.active_platforms' in preflight
    assert "zRange[0] < 1" in preflight
    assert "zRange[1] > 7" in preflight
    assert ".toFile().isDirectory()" in preflight
    assert "is not a directory" in preflight


def test_bead_removal_fill_logs_once_in_the_workflow_block() -> None:
    main_text = _main_text()
    workflow_block = main_text[main_text.index("\nworkflow {") :]

    assert workflow_block.count("merscope_bead_removal_fill is on") == 1
    assert "params.merscope_bead_removal_fill" in workflow_block

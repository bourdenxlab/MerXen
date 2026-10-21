"""Workflow text checks for the SpatialData build configuration."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

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


def test_bead_removal_device_param_defaults_to_cpu() -> None:
    config_text = (_REPO_ROOT / "workflows" / "nextflow.config").read_text()

    assert 'merscope_bead_removal_device = "cpu"' in config_text


def test_bead_removal_device_key_is_set_only_when_not_cpu() -> None:
    """CPU bead build JSONs (and task hashes) keep the form without a device."""
    main_text = _main_text()
    build_config = _function_text(main_text, "buildConfigForPlatform")
    block_start = build_config.index("if (merscopeBeadRemovalRequested(")
    block_end = build_config.index("return [", block_start)
    bead_block = build_config[block_start:block_end]

    assert build_config.count("beadRemoval.device") == 1
    assert "beadRemoval.device = beadDevice" in bead_block
    assert 'if (beadDevice != "cpu" && !beadFill) {' in bead_block
    assert "merscopeBeadRemovalDevice(params)" in bead_block
    # The key sits behind the guard, not next to the unconditional keys.
    guard = bead_block.index('if (beadDevice != "cpu" && !beadFill) {')
    assert guard < bead_block.index("beadRemoval.device = beadDevice")
    assert bead_block.index("beadRemoval.device = beadDevice") < bead_block.index(
        "merscopeOptions.bead_removal = beadRemoval"
    )
    assert "device" not in build_config[:block_start]


def test_bead_removal_device_helper_defaults_to_cpu_and_normalises() -> None:
    helper = _function_text(_main_text(), "merscopeBeadRemovalDevice")

    assert "params.merscope_bead_removal_device" in helper
    assert 'return "cpu"' in helper
    assert ".toLowerCase()" in helper


def test_bead_removal_preflight_checks_the_device_value() -> None:
    preflight = _function_text(_main_text(), "appendMerscopeBeadRemovalPreflightChecks")

    assert "merscopeBeadRemovalDevice(params)" in preflight
    assert '["auto", "cpu", "cuda"]' in preflight
    assert "merscope_bead_removal_device must be auto, cpu or cuda" in preflight


def test_bead_removal_device_key_is_not_written_with_the_fill() -> None:
    """The fill forces the CPU, so the device would only change the task hash."""
    build_config = _function_text(_main_text(), "buildConfigForPlatform")

    assert 'if (beadDevice != "cpu" && !beadFill) {' in build_config


def test_bead_removal_fill_mode_param_defaults_to_flat() -> None:
    config_text = (_REPO_ROOT / "workflows" / "nextflow.config").read_text()

    assert 'merscope_bead_removal_fill_mode = "flat"' in config_text
    # The comment above the param says what the modes are.
    line = config_text.index('merscope_bead_removal_fill_mode = "flat"')
    assert "hybrid" in config_text[line - 400 : line]


def test_bead_removal_fill_mode_key_only_in_the_fill_branch_off_default() -> None:
    """Flat-fill build JSONs (and task hashes) must keep their form."""
    build_config = _function_text(_main_text(), "buildConfigForPlatform")
    block_start = build_config.index("if (merscopeBeadRemovalRequested(")
    block_end = build_config.index("return [", block_start)
    bead_block = build_config[block_start:block_end]

    assert build_config.count("beadRemoval.fill_mode") == 1
    assert "fill_mode" not in build_config[:block_start]
    assert "fill_mode" not in build_config[block_end:]
    fill_branch = bead_block.index("if (beadFill) {")
    guard = bead_block.index('if (beadFillMode != "flat") {')
    key = bead_block.index("beadRemoval.fill_mode = beadFillMode")
    after_fill_branch = bead_block.index("def beadDevice")
    # The key sits behind the fill branch and the non-default guard.
    assert fill_branch < bead_block.index("beadRemoval.fill = true") < guard < key
    assert key < after_fill_branch


def test_bead_removal_fill_mode_helper_defaults_to_flat_and_normalises() -> None:
    helper = _function_text(_main_text(), "merscopeBeadRemovalFillMode")

    assert "params.merscope_bead_removal_fill_mode" in helper
    assert 'return "flat"' in helper
    assert ".toLowerCase()" in helper


def test_bead_removal_preflight_checks_the_fill_mode_value() -> None:
    preflight = _function_text(_main_text(), "appendMerscopeBeadRemovalPreflightChecks")

    assert "merscopeBeadRemovalFillMode(params)" in preflight
    assert '["flat", "hybrid"]' in preflight
    assert "merscope_bead_removal_fill_mode must be flat or hybrid" in preflight
    assert "only applies to the " in preflight


def test_published_source_zarr_fill_mode_is_compared() -> None:
    published = _function_text(_main_text(), "appendPublishedBeadRemovalChecks")
    state = _function_text(_main_text(), "publishedBeadRemovalState")

    assert "recorded.fillMode != requestedFillMode" in published
    assert 'fill.get("mode")' in state and '"flat"' in state


def test_bead_removal_z_range_is_only_checked_with_a_raw_folder() -> None:
    """A zarr-only reuse row projects nothing, so its blank z range is fine."""
    preflight = _function_text(_main_text(), "appendMerscopeBeadRemovalPreflightChecks")

    assert preflight.index("if (!merscopeDir) {") < preflight.index("zRange[0] < 1")


def test_bead_removal_fill_log_waits_for_a_row_that_requests_bead_removal() -> None:
    main_text = _main_text()
    workflow_block = main_text[main_text.index("\nworkflow {") :]

    assert "beadRequested.any" in workflow_block
    assert workflow_block.index("beadRequested.any") < workflow_block.index(
        "merscope_bead_removal_fill is on"
    )


def test_published_source_zarr_is_checked_when_the_build_is_skipped() -> None:
    main_text = _main_text()
    preflight = _function_text(main_text, "appendMerscopeBeadRemovalPreflightChecks")
    published = _function_text(main_text, "appendPublishedBeadRemovalChecks")

    assert (
        "appendPublishedBeadRemovalChecks(errors, row, settings, params)" in preflight
    )
    assert preflight.index("!settings.run_build") < preflight.index(
        "appendPublishedBeadRemovalChecks("
    )
    assert "settings.need_build_results" in published
    assert "spatialdata/spatialdata_out/source_spatialdata.zarr" in published
    assert "recorded.enabled != requested" in published
    assert "recorded.fill != requestedFill" in published


_BEAD_HARNESS_FUNCTIONS = (
    "boolOrDefault",
    "rowFieldOrDefault",
    "chooseField",
    "parseRange",
    "isBlankPath",
    "normalizedPath",
    "publishedDatasetPath",
    "merscopeBeadRemovalRequested",
    "merscopeBeadRemovalDevice",
    "merscopeBeadRemovalFillMode",
    "appendMerscopeBeadRemovalPreflightChecks",
    "onOffText",
    "publishedBeadRemovalState",
    "appendPublishedBeadRemovalChecks",
    "buildConfigForPlatform",
    "requirePlatformInput",
)

_BEAD_HARNESS = Path(__file__).parent / "harness" / "bead_removal_preflight.nf"


def _run_bead_harness(tmp_path: Path, *params: str) -> dict[str, str]:
    nextflow = shutil.which("nextflow")
    assert nextflow is not None
    main_text = _main_text()
    functions = "\n".join(
        _function_text(main_text, name) for name in _BEAD_HARNESS_FUNCTIONS
    )
    script = tmp_path / "harness.nf"
    script.write_text(functions + "\n" + _BEAD_HARNESS.read_text())
    root = tmp_path / "published"
    completed = subprocess.run(
        [
            nextflow,
            "run",
            str(script),
            "--root",
            str(root),
            "--outdir",
            str(root),
            *params,
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=120,
        env={**os.environ, "NXF_OFFLINE": "true"},
    )
    assert completed.returncode == 0, (
        completed.stdout[-3000:] + completed.stderr[-2000:]
    )
    results = {
        line.split(":")[0].removeprefix("CASE "): line.split(": ", 1)[1]
        for line in completed.stdout.splitlines()
        if line.startswith("CASE ")
    }
    config = next(
        line for line in completed.stdout.splitlines() if line.startswith("CONFIG ")
    )
    results["config"] = config.removeprefix("CONFIG ")
    return results


@pytest.mark.skipif(shutil.which("nextflow") is None, reason="nextflow is unavailable")
def test_a_published_source_zarr_must_match_the_requested_bead_removal(
    tmp_path: Path,
) -> None:
    """The skipped-build path refuses a mismatch instead of running uncorrected."""
    results = _run_bead_harness(
        tmp_path,
        "--merscope_bead_removal_fill",
        "false",
        "--merscope_bead_removal_device",
        "cuda",
    )

    assert results["legacy_off"] == "OK"
    assert results["legacy_on"] == "ERROR"
    assert results["on_on"] == "OK"
    assert results["on_off"] == "ERROR"
    assert results["zattrs_on_off"] == "ERROR"
    # The fill of the zarr differs from the (off) request.
    assert results["fill_on"] == "ERROR"
    # Not needed, or not MERSCOPE: no check.
    assert results["on_on_not_needed"] == "OK"
    assert results["on_on_xenium_only"] == "OK"
    # The z range matters only when a raw folder is projected.
    assert results["blank_z_zarr_only"] == "OK"
    assert results["blank_z_with_dir"] == "ERROR"
    assert results["z_1_6_with_dir"] == "OK"
    assert json.loads(results["config"]) == {"enabled": True, "device": "cuda"}


@pytest.mark.skipif(shutil.which("nextflow") is None, reason="nextflow is unavailable")
def test_the_fill_request_is_compared_and_replaces_the_device_key(
    tmp_path: Path,
) -> None:
    results = _run_bead_harness(
        tmp_path,
        "--merscope_bead_removal_fill",
        "true",
        "--merscope_bead_removal_device",
        "cuda",
    )

    assert results["on_on"] == "ERROR"  # the zarr has no fill, the request does
    assert results["fill_on"] == "OK"
    assert json.loads(results["config"]) == {"enabled": True, "fill": True}


@pytest.mark.skipif(shutil.which("nextflow") is None, reason="nextflow is unavailable")
def test_the_hybrid_fill_mode_is_compared_and_added_to_the_build_json(
    tmp_path: Path,
) -> None:
    results = _run_bead_harness(
        tmp_path,
        "--merscope_bead_removal_fill",
        "true",
        "--merscope_bead_removal_fill_mode",
        "Hybrid",
    )

    # A zarr with a flat fill (no mode recorded) does not match a hybrid request.
    assert results["fill_on"] == "ERROR"
    assert results["fill_hybrid"] == "OK"
    assert results["z_1_6_with_dir"] == "OK"
    assert json.loads(results["config"]) == {
        "enabled": True,
        "fill": True,
        "fill_mode": "hybrid",
    }


@pytest.mark.skipif(shutil.which("nextflow") is None, reason="nextflow is unavailable")
@pytest.mark.parametrize(
    ("fill", "mode", "expected"),
    [
        ("false", "hybrid", "ERROR"),  # the mode needs the fill
        ("true", "bogus", "ERROR"),  # not flat or hybrid
        ("true", "hybrid", "OK"),
        ("false", "flat", "OK"),
    ],
)
def test_the_fill_mode_value_is_checked_before_a_build(
    tmp_path: Path, fill: str, mode: str, expected: str
) -> None:
    results = _run_bead_harness(
        tmp_path,
        "--merscope_bead_removal_fill",
        fill,
        "--merscope_bead_removal_fill_mode",
        mode,
    )

    # run_build is on in this case, so the preflight checks the request itself.
    assert results["blank_z_zarr_only"] == expected

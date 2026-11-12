"""Golden configs rendered by the platform-dependent Groovy builders.

The workflow hands each task a config JSON built in Groovy (BUILD, SEGMENT,
ENRICH, VIEWER_CACHE, the QC heredoc, the clustering/MECR sample lists, the
pciSeq config) and picks its ``path`` inputs from platform-dependent samplesheet
fields (restart zarrs, transforms). Those values feed the task hashes, so a
refactor must render them byte for byte as before.

The harness concatenates the Groovy functions of ``workflows/main.nf``,
top-level wrappers around the inline builders of its workflow block (extracted
between text anchors) and the QC ``script:`` section, then
``harness/platform_configs.nf``, and runs it with the real
``workflows/nextflow.config`` (its defaults and their Groovy types). Global
params need a launch each: L0 runs every case on the defaults, L1 and L2 rerun
the MERSCOPE BUILD and SEGMENT cases with the bead-removal and transform params
off their defaults, L3 reruns them with bead removal switched on globally
(``--merscope_bead_removal true``) instead of per row, and L4 reruns the ATERA
BUILD cases with the ATERA params off their defaults (AT-4 added the ATERA
cases; the XENIUM and MERSCOPE cases are unchanged). AT-6 added the ATERA
native-route cases (the ``native`` file and ATERA ids in ``enrich``,
``viewer_cache`` and ``analysis_layers``); again only case ids were added.
AT-9 added the ``atera_proseg`` file (the ATERA ProSeg route) and changed the
error text of two ``native`` cases (the list of sources): the ATERA-only
``atera_cellpose_only_error`` and the platform-independent ``unknown_error``,
so an unknown ``segmentation_source`` on a XENIUM or MERSCOPE row now lists
``native_proseg`` among the valid values. Only the message changes; no task
input or hash does.

Goldens hold the compact ``JsonOutput.toJson`` text of each case (the task input
is its deterministic ``prettyPrint``) with the launch directory written as
``<WORK>``. They are tied to the ``nextflow.config`` defaults on purpose: a
changed default changes task hashes. Regenerate after an intended, announced
change with::

    MERXEN_UPDATE_GOLDEN=1 pytest tests/test_workflows/test_platform_config_golden.py
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOWS = _REPO_ROOT / "workflows"
_GOLDEN_DIR = Path(__file__).parent / "golden"
_HARNESS_TAIL = Path(__file__).parent / "harness" / "platform_configs.nf"
_UPDATE = os.environ.get("MERXEN_UPDATE_GOLDEN") == "1"
_WORK = "<WORK>"
# Absolute paths that are defaults of the default profile (conf/dwight.config)
# inside the container, not host paths.
_CONTAINER_PATHS = frozenset({"/usr/local/bin/proseg"})

GOLDEN_FILES: tuple[str, ...] = (
    "build",
    "segment",
    "enrich",
    "viewer_cache",
    "qc",
    "analysis_layers",
    "modes",
    "pciseq",
    "native",
    "mask_image_quantification",
    "atera_proseg",
    "atera_cellpose_proseg",
)

# Launch id -> extra command-line params (params are global per run).
LAUNCHES: dict[str, tuple[str, ...]] = {
    "L0": (),
    "L1": (
        "--merscope_bead_removal_fill",
        "false",
        "--merscope_bead_removal_device",
        "cuda",
    ),
    "L2": (
        "--merscope_bead_removal_fill_mode",
        "flat",
        "--merscope_bead_removal_device",
        "cuda",
        "--merscope_allow_transform_override_mismatch",
        "true",
        "--merscope_bead_mask_veto",
        "false",
    ),
    "L3": ("--merscope_bead_removal", "true"),
    "L4": (
        "--atera_min_qv",
        "30",
        "--atera_build_max_rss_gb",
        "64",
        "--species",
        "mouse",
    ),
}


@dataclass(frozen=True)
class Span:
    """An inline builder of the workflow block, wrapped in a top-level function.

    Attributes:
        signature: Groovy signature of the wrapper.
        after: Text that precedes the builder (its channel assignment).
        start: Text the builder starts with.
        end: Text right after the builder.
        token: Text the builder must contain (guards against anchor drift).
        result: Variable the wrapper returns.
    """

    signature: str
    after: str
    start: str
    end: str
    token: str
    result: str


SPANS: tuple[Span, ...] = (
    Span(
        "renderSegment(key, pairId, platform, sourceSpatialdata, meta, "
        "runSegmentNuclei, runSegment)",
        "segmentation_config_inputs_ch = build_results_ch",
        "def segmentConfig = segmentationConfigMap(",
        "\n            tuple(",
        "segmentationConfigMap(",
        "segmentConfig",
    ),
    Span(
        "renderEnrich(key, pairId, platform, latestZarr, maskPath, nucleiMaskPath, "
        "originalDataPath, rowSource = null)",
        "enrich_inputs_ch = segment_results_ch",
        "def persistentLatestZarrPath",
        "\n            tuple(",
        "def enrichConfig = [",
        "enrichConfig",
    ),
    Span(
        "enrichRestartField(row, platform)",
        "enrich_published_results_ch =",
        "def explicitLatestZarr",
        "\n                // Cross-pair",
        "def explicitLatestZarr =",
        "explicitLatestZarr",
    ),
    Span(
        "viewerTransformField(row, platform)",
        "viewer_cache_transform_ch =",
        "def transformPath",
        "\n                tuple(",
        "def transformPath =",
        "transformPath",
    ),
    Span(
        "renderViewerCache(key, pairId, platform, enrichedLatestZarr, transformPath)",
        "viewer_cache_inputs_ch = post_registration_zarrs_ch",
        "def resolvedTransform",
        "\n            tuple(",
        "def viewerCacheConfig = [",
        "viewerCacheConfig",
    ),
    Span(
        "renderMaskImageQuantification(key, pairId, platform, enrichedLatestZarr, "
        "maskPath, rowSource = null)",
        "mask_image_quantification_inputs_ch = post_enrich_zarrs_ch",
        "def quantConfig",
        "\n            tuple(",
        "def quantConfig = [",
        "quantConfig",
    ),
)

_QC_SIGNATURE = (
    "renderQcScript(pair_id, platform, segmentation, latest_zarr, table_key, "
    "shape_key, task)"
)

_SourceParser = Callable[[str], dict[str, dict[str, str]]]


def extract_span(main_text: str, span: Span) -> str:
    """Return the builder text of one span of the workflow block.

    Args:
        main_text: Source of ``workflows/main.nf``.
        span: The anchors of the builder.

    Returns:
        The text from ``span.start`` up to (not including) ``span.end``.

    Raises:
        AssertionError: If an anchor is missing, lies in another channel
            expression or the builder lost its token.
    """
    workflow_start = main_text.index("\nworkflow {")
    assert main_text.count(span.after) == 1, f"anchor not unique: {span.after!r}"
    after = main_text.index(span.after)
    assert after > workflow_start, f"anchor outside the workflow: {span.after!r}"
    start = main_text.index(span.start, after)
    assert "_ch = " not in main_text[after + len(span.after) : start], (
        f"{span.start!r} is not in the expression of {span.after!r}"
    )
    end = main_text.index(span.end, start)
    text = main_text[start:end]
    assert text.strip(), f"empty span after {span.after!r}"
    assert span.token in text, f"span after {span.after!r} lost {span.token!r}"
    assert "_ch = " not in text, f"span after {span.after!r} runs past its channel"
    return text


def workflow_functions() -> list[str]:
    """Return the Groovy functions of main.nf, then one wrapper per span.

    The registry test (``test_platform_registry_groovy.py``) builds its harness
    from the same parts.

    Returns:
        The helper functions above the workflow block, then the span wrappers.
    """
    main_text = (_WORKFLOWS / "main.nf").read_text()
    helpers = main_text[
        main_text.index("def parseChannels") : main_text.index("\nworkflow {")
    ]
    wrappers = [
        f"def {span.signature} {{\n{extract_span(main_text, span)}\n"
        f"    return {span.result}\n}}\n"
        for span in SPANS
    ]
    return [helpers, *wrappers]


def harness_script(parse: _SourceParser) -> str:
    """Return the full harness: main.nf helpers, span wrappers and the tail."""
    qc_script = parse((_WORKFLOWS / "modules" / "qc.nf").read_text())["QC"]["script"]
    qc_wrapper = f"def {_QC_SIGNATURE} {{\n{qc_script}}}\n"
    return "\n".join([*workflow_functions(), qc_wrapper, _HARNESS_TAIL.read_text()])


def _parse_cases(stdout: str, work_dir: Path) -> dict[str, dict[str, str]]:
    """Return ``golden file -> case id -> normalised compact JSON``."""
    cases: dict[str, dict[str, str]] = {name: {} for name in GOLDEN_FILES}
    roots = sorted({str(work_dir), str(work_dir.resolve())}, key=len, reverse=True)
    for line in stdout.splitlines():
        if not line.startswith("CASE\t"):
            continue
        _, group, case_id, value = line.split("\t", 3)
        for root in roots:
            value = value.replace(root, _WORK)
        assert group in cases, f"unknown golden file {group!r}"
        assert case_id not in cases[group], f"duplicate case {group}/{case_id}"
        cases[group][case_id] = value
    return cases


def hermetic_env() -> dict[str, str]:
    """The test environment without MerXen overrides of config defaults.

    ``nextflow.config`` reads a few ``MERXEN_*`` variables (model store,
    containers); the goldens hold the defaults, so none of them is passed on.
    """
    env = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("MERXEN_")
    }
    env.update(
        {
            "MERXEN_SITE_CONFIG": "/dev/null",
            "MERXEN_DWIGHT_SITE_CONFIG": "/dev/null",
            "NXF_OFFLINE": "true",
        }
    )
    return env


def _render_all(tmp_root: Path, script: str) -> dict[str, dict[str, str]]:
    """Run every launch side by side and merge their cases."""
    nextflow = shutil.which("nextflow")
    assert nextflow is not None
    running: dict[str, tuple[Path, subprocess.Popen[str]]] = {}
    for launch, extra in LAUNCHES.items():
        work_dir = tmp_root / launch
        work_dir.mkdir()
        (work_dir / "h.nf").write_text(script)
        (work_dir / "lib").symlink_to(_WORKFLOWS / "lib", target_is_directory=True)
        running[launch] = (
            work_dir,
            subprocess.Popen(
                [
                    nextflow,
                    "run",
                    "h.nf",
                    "-c",
                    str(_WORKFLOWS / "nextflow.config"),
                    "--outdir",
                    "results",
                    "-ansi-log",
                    "false",
                    "--golden_launch",
                    launch,
                    *extra,
                ],
                cwd=work_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=hermetic_env(),
            ),
        )
    merged: dict[str, dict[str, str]] = {name: {} for name in GOLDEN_FILES}
    for launch, (work_dir, process) in running.items():
        stdout, stderr = process.communicate(timeout=300)
        assert process.returncode == 0 and f"DONE {launch}" in stdout, (
            f"launch {launch} failed:\n{stdout[-3000:]}\n{stderr[-2000:]}"
        )
        for group, cases in _parse_cases(stdout, work_dir).items():
            merged[group].update(cases)
    return merged


@pytest.fixture(scope="module")
def rendered(
    tmp_path_factory: pytest.TempPathFactory, process_sections: _SourceParser
) -> dict[str, dict[str, str]]:
    """Cases of every launch; rewrites the goldens under MERXEN_UPDATE_GOLDEN."""
    if shutil.which("nextflow") is None:
        pytest.skip("nextflow is not installed")
    cases = _render_all(
        tmp_path_factory.mktemp("platform_configs"), harness_script(process_sections)
    )
    if _UPDATE:
        _GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        for name, values in cases.items():
            (_GOLDEN_DIR / f"{name}.json").write_text(
                json.dumps(values, indent=2, sort_keys=True) + "\n"
            )
    return cases


def test_spans_are_found_in_the_workflow_block(
    process_sections: _SourceParser,
) -> None:
    """Every anchor still matches one builder (fast tier: no Nextflow)."""
    script = harness_script(process_sections)
    for span in SPANS:
        assert f"def {span.signature} {{\n" in script
    assert f"def {_QC_SIGNATURE} {{\n" in script
    assert "<<JSON\n" in script


@pytest.mark.nextflow
@pytest.mark.parametrize("name", GOLDEN_FILES)
def test_rendered_configs_match_golden(
    name: str, rendered: dict[str, dict[str, str]]
) -> None:
    """Every rendered XENIUM/MERSCOPE value is byte-identical to its golden."""
    path = _GOLDEN_DIR / f"{name}.json"
    assert path.is_file(), (
        f"{path.name} is missing: generate it with MERXEN_UPDATE_GOLDEN=1"
    )
    golden: dict[str, str] = json.loads(path.read_text())
    current = rendered[name]
    assert current, f"no case rendered for {name}"
    added = sorted(set(current) - set(golden))
    removed = sorted(set(golden) - set(current))
    assert not added and not removed, (
        f"{name}: cases not in the golden: {added}; golden cases gone: {removed}"
    )
    changed = sorted(key for key in golden if golden[key] != current[key])
    assert not changed, (
        f"{name}: rendered config changed for {changed}; this changes task hashes. "
        f"First: golden {golden[changed[0]]} != rendered {current[changed[0]]}. "
        "Regenerate with MERXEN_UPDATE_GOLDEN=1 only for an announced change."
    )


_ROWS_WITHOUT_INPUTS = frozenset({"missing_inputs", "blank_inputs"})


def _expects_error(name: str, case_id: str) -> bool:
    """Whether a case must raise: ``*_error`` ids and BUILD without inputs."""
    segments = case_id.split("/")
    return segments[-1].endswith("_error") or (
        name == "build" and not _ROWS_WITHOUT_INPUTS.isdisjoint(segments)
    )


@pytest.mark.nextflow
def test_rendered_values_are_json_and_errors_are_expected(
    rendered: dict[str, dict[str, str]],
) -> None:
    """Every value parses as JSON and only the expected cases raise."""
    for name, cases in rendered.items():
        for case_id, value in cases.items():
            parsed = json.loads(value)
            is_error = isinstance(parsed, dict) and "error_class" in parsed
            assert is_error == _expects_error(name, case_id), (
                f"{name}/{case_id}: {value}"
            )


@pytest.mark.nextflow
def test_rendered_values_hold_no_host_path(
    rendered: dict[str, dict[str, str]],
) -> None:
    """Absolute paths are all under the launch directory, written ``<WORK>``."""
    home = str(Path.home())
    config_text = (_WORKFLOWS / "conf" / "dwight.config").read_text()
    for path in _CONTAINER_PATHS:
        assert f'"{path}"' in config_text, f"{path} is no longer a config default"
    for name, cases in rendered.items():
        for case_id, value in cases.items():
            assert home not in value, f"{name}/{case_id} holds a home path"
            residue = set(re.findall(r'(?:"|\\")(/[^"\\]*)', value))
            residue -= _CONTAINER_PATHS
            assert not residue, f"{name}/{case_id} holds absolute paths {residue}"


@pytest.mark.nextflow
def test_paired_mode_renders_the_single_platform_configs(
    rendered: dict[str, dict[str, str]],
) -> None:
    """A platform's BUILD and SEGMENT JSON do not depend on the analysis mode."""
    modes = rendered["modes"]
    for platform, single in (("MERSCOPE", "merscope"), ("XENIUM", "xenium")):
        for builder in ("build", "segment"):
            paired = modes[f"L0/paired_row/paired/{platform}/{builder}"]
            assert paired == modes[f"L0/paired_row/{single}/{platform}/{builder}"]
    assert "L0/paired_row/merscope/XENIUM/build" not in modes
    assert "L0/paired_row/xenium/MERSCOPE/build" not in modes


@pytest.mark.nextflow
def test_bead_removal_params_reach_only_bead_rows(
    rendered: dict[str, dict[str, str]],
) -> None:
    """L1/L2 change the bead block (and L2 the transform opt-out) only."""
    build = rendered["build"]
    for launch in ("L1", "L2"):
        for case_id in [key for key in build if key.startswith(f"{launch}/")]:
            base = build[case_id.replace(f"{launch}/", "L0/", 1)]
            if case_id.endswith("/bead_removal") or (
                launch == "L2" and not _expects_error("build", case_id)
            ):
                assert build[case_id] != base, case_id
            else:
                assert build[case_id] == base, case_id

"""The Groovy platform registry agrees with ``merxen.platforms``.

``workflows/lib/PlatformSpec.groovy`` is the workflow's twin of
``src/merxen/platforms.py``. The test launches Nextflow once on the Groovy
functions of ``workflows/main.nf`` (plus wrappers around the inline builders the
golden test extracts) and ``harness/platform_registry.nf``, which prints the
Groovy registry and the error of every converted call site for a platform that
is not registered.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from merxen.platforms import PLATFORM_SPECS

_HERE = Path(__file__).parent
_WORKFLOWS = _HERE.parents[1] / "workflows"
_HARNESS_TAIL = _HERE / "harness" / "platform_registry.nf"
_PYTHON_FIELDS = (
    "name",
    "family",
    "native_image_key",
    "original_cell_shape_key",
    "original_nucleus_shape_key",
    "default_channels",
    "segmentation_sources",
    "default_segmentation_source",
)
# Samplesheet fields of each platform, as main.nf read them before the registry.
_SAMPLESHEET_FIELDS: dict[str, dict[str, list[str]]] = {
    "MERSCOPE": {
        "input_dir_fields": ["merscope_dir"],
        "spatialdata_path_fields": ["merscope_spatialdata_path", "merscope_zarr_path"],
        "transform_path_fields": ["merscope_transform_path"],
    },
    "XENIUM": {
        "input_dir_fields": ["xenium_dir"],
        "spatialdata_path_fields": ["xenium_spatialdata_path"],
        "transform_path_fields": ["xenium_spec_path"],
    },
    "ATERA": {
        "input_dir_fields": ["atera_dir"],
        "spatialdata_path_fields": ["atera_spatialdata_path"],
        "transform_path_fields": ["atera_spec_path"],
    },
}


def _golden_module() -> ModuleType:
    """Load the golden test module for its harness parts (no sibling import).

    It is loaded under a private name, so its module code runs a second time;
    pytest still collects (and runs) its tests once, from the file itself.
    """
    name = "_merxen_platform_config_golden"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, _HERE / "test_platform_config_golden.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _harness_script() -> str:
    """Return main.nf helpers, the span wrappers and the registry tail."""
    parts: list[str] = _golden_module().workflow_functions()
    return "\n".join([*parts, _HARNESS_TAIL.read_text()])


@pytest.fixture(scope="module")
def groovy_output(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Run the harness and return its ``REGISTRY`` and ``UNKNOWN`` records."""
    nextflow = shutil.which("nextflow")
    if nextflow is None:
        pytest.skip("nextflow is not installed")
    work_dir = tmp_path_factory.mktemp("platform_registry")
    (work_dir / "h.nf").write_text(_harness_script())
    (work_dir / "lib").symlink_to(_WORKFLOWS / "lib", target_is_directory=True)
    golden = _golden_module()
    result = subprocess.run(
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
        ],
        cwd=work_dir,
        capture_output=True,
        text=True,
        env=golden.hermetic_env(),
        timeout=300,
        check=False,
    )
    assert result.returncode == 0 and "DONE" in result.stdout, (
        f"{result.stdout[-3000:]}\n{result.stderr[-2000:]}"
    )
    output: dict[str, Any] = {"registry": None, "unknown": {}}
    for line in result.stdout.splitlines():
        if line.startswith("REGISTRY\t"):
            output["registry"] = json.loads(line.split("\t", 1)[1])
        elif line.startswith("UNKNOWN\t"):
            _, case_id, value = line.split("\t", 2)
            output["unknown"][case_id] = json.loads(value)
    return output


def test_harness_spans_are_found() -> None:
    """The harness builds without Nextflow (fast tier: anchors still match)."""
    script = _harness_script()
    for span in _golden_module().SPANS:
        assert f"def {span.signature} {{\n" in script


@pytest.mark.nextflow
def test_groovy_registry_matches_python(groovy_output: dict[str, Any]) -> None:
    """Both registries list the same platforms, in order, with the same values."""
    groovy = groovy_output["registry"]
    assert groovy is not None
    assert list(groovy) == list(PLATFORM_SPECS)
    for name, spec in PLATFORM_SPECS.items():
        python = dataclasses.asdict(spec)
        python["default_channels"] = list(spec.default_channels)
        python["segmentation_sources"] = list(spec.segmentation_sources)
        shared = {field: groovy[name][field] for field in _PYTHON_FIELDS}
        assert shared == {field: python[field] for field in _PYTHON_FIELDS}, name


@pytest.mark.nextflow
def test_groovy_registry_holds_the_samplesheet_fields(
    groovy_output: dict[str, Any],
) -> None:
    """The workflow-only fields equal the lists main.nf used to spell out."""
    groovy = groovy_output["registry"]
    for name, fields in _SAMPLESHEET_FIELDS.items():
        assert {field: groovy[name][field] for field in fields} == fields, name


_UNKNOWN_SITES = (
    *(
        f"analysis_layer_keys/{segmentation}"
        for segmentation in (
            "proseg",
            "reseg",
            "original",
            "original_seg",
            "proseg_hybrid",
            "cellpose",
            "proseg_mask",
            "proseg_geometry_assignment",
        )
    ),
    "pciseq_spatialdata_path",
    "enrich_restart_field",
    "viewer_transform_field",
    "require_platform_input",
    "build_config",
    "segment_meta",
    "channels_for_row",
    "channels_for_row_override",
)


@pytest.mark.nextflow
@pytest.mark.parametrize("platform", ["COSMX", "merscope"])
def test_unknown_platform_raises_at_every_converted_site(
    groovy_output: dict[str, Any], platform: str
) -> None:
    """An unregistered platform raises instead of taking another one's branch.

    Before the registry, ``analysisLayerKeys("ATERA", "original_seg")`` (ATERA
    was not registered then) returned the Xenium shape key, and the restart,
    transform and channel lookups silently used Xenium (or MERSCOPE) fields.
    """
    unknown = groovy_output["unknown"]
    for site in _UNKNOWN_SITES:
        record = unknown[f"{platform}/{site}"]
        assert record.get("error_class") == "java.lang.IllegalArgumentException", (
            f"{platform}/{site}: {record}"
        )
        assert record["message"].startswith(f"Unknown platform: {platform}"), (
            f"{platform}/{site}: {record}"
        )

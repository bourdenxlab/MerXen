"""Execute optional model settings against the actual legacy config builder."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from merxen.config import CellposeConfig, MaskFilterConfig, TilingConfig
from merxen.segmentation.model_identity import (
    CELLPOSE_VERSION,
    CPSAM_V2_SHA256,
    TORCH_MINOR,
    CellposeInferenceContext,
    CellposeModelIdentity,
)

ROOT = Path(__file__).resolve().parents[2]


def _defaults() -> dict:
    values = {}
    text = (ROOT / "workflows/nextflow.config").read_text()
    for name, raw in re.findall(r"^    (\w+) = (.+)$", text, re.MULTILINE):
        try:
            values[name] = json.loads(raw)
        except json.JSONDecodeError:
            continue
    values["cellpose_model_store"] = None
    return values


def _run_cases(tmp_path: Path, cases: list[dict]) -> list[dict]:
    text = (ROOT / "workflows/main.nf").read_text()
    block = re.search(
        r"def baseConfig = \[.*?^            \]", text, re.MULTILINE | re.DOTALL
    )
    assert block is not None
    base = block.group().replace("params.", "opts.")
    hook = text.split("// hitl-hook:C4\n", 1)[1].split("def segmentConfig =", 1)[0]
    (tmp_path / "lib").symlink_to(ROOT / "workflows/lib", target_is_directory=True)
    (tmp_path / "cases.json").write_text(json.dumps(cases))
    script = tmp_path / "check.nf"
    script.write_text(
        """nextflow.enable.dsl = 2
workflow {
    def cases = new groovy.json.JsonSlurper().parse(new File('cases.json'))
    def results = cases.collect { entry ->
        try {
            def opts = entry.options
            def chosen = CellposeSettings.settings(entry.row, 'XENIUM', opts)
            def meta = [cellpose_settings: chosen]
            """
        + base
        + """
            def before = groovy.json.JsonOutput.prettyPrint(
                groovy.json.JsonOutput.toJson(baseConfig))
            """
        + hook
        + """
            if (entry.check_preflight) {
                CellposeSettings.preflight(entry.row,
                    [active_platforms: ['XENIUM'], species: 'human', pair_id: 'P1',
                     run_segment: entry.containsKey('run_segment')
                         ? entry.run_segment : true,
                     stage_order: ['build_spatialdata', 'segment_nuclei',
                         'segment', 'enrich', 'qc'],
                     start_stage: entry.start_stage ?: 'enrich'],
                    opts)
            }
            return [before: before, after: groovy.json.JsonOutput.prettyPrint(
                groovy.json.JsonOutput.toJson(baseConfig)), selected: chosen]
        } catch (IllegalArgumentException error) {
            return [error: error.message]
        }
    }
    new File('results.json').text = groovy.json.JsonOutput.toJson(results)
}
"""
    )
    result = subprocess.run(
        ["nextflow", "run", str(script), "-ansi-log", "false"],
        cwd=tmp_path,
        env={**os.environ, "NXF_OFFLINE": "true"},
        capture_output=True,
        text=True,
        check=False,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads((tmp_path / "results.json").read_text())


def test_required_hitl_hooks_occur_once() -> None:
    files = {
        "C1": "src/merxen/config.py",
        "C2": "src/merxen/segmentation/cellpose.py",
        "C3": "src/merxen/segmentation/pipeline.py",
        "C4": "workflows/main.nf",
        "C5": "src/merxen/segmentation/pipeline.py",
        "C7": "src/merxen/segmentation/cellpose.py",
    }
    for hook, name in files.items():
        assert (ROOT / name).read_text().count(f"hitl-hook:{hook}") == 1
    library = (ROOT / "workflows/lib/CellposeSettings.groovy").read_text()
    assert CPSAM_V2_SHA256 in library
    assert CELLPOSE_VERSION in library
    assert f"startsWith('{TORCH_MINOR}.')" in library


@pytest.mark.skipif(shutil.which("nextflow") is None, reason="nextflow unavailable")
def test_full_pipeline_preview_loads_model_library_without_scheduling(
    tmp_path: Path,
) -> None:
    """Compile the real entry workflow with its library and an inert stock row."""
    raw = tmp_path / "raw-xenium"
    raw.mkdir()
    samplesheet = tmp_path / "samplesheet.csv"
    samplesheet.write_text(f"pair_id,xenium_dir\nHD1,{raw}\n")
    output = tmp_path / "arm"
    result = subprocess.run(
        [
            "nextflow",
            "run",
            str(ROOT / "workflows/main.nf"),
            "-profile",
            "conda",
            "-preview",
            "-ansi-log",
            "false",
            "-work-dir",
            str(tmp_path / "work"),
            "--samplesheet",
            str(samplesheet),
            "--analysis_mode",
            "xenium",
            "--only_stage",
            "build_spatialdata",
            "--outdir",
            str(output),
        ],
        cwd=tmp_path,
        env={**os.environ, "NXF_OFFLINE": "true"},
        capture_output=True,
        text=True,
        check=False,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    # The entry workflow initializes its output/log directory during preview.
    # Actual build or segmentation artifacts would mean a process was scheduled.
    assert not list(output.rglob("source_spatialdata.zarr"))
    assert not list(output.rglob("*.npy"))


@pytest.mark.skipif(shutil.which("nextflow") is None, reason="nextflow unavailable")
def test_default_config_and_individual_platform_overrides(tmp_path: Path) -> None:
    cases = [{"options": _defaults(), "row": {}}]
    overrides = [
        ("xenium_cellpose_cellprob", "cellpose", "cellprob_threshold", 0),
        ("xenium_cellpose_flow_threshold", "cellpose", "flow_threshold", 0.5),
        ("xenium_cellpose_diameter", "cellpose", "diameter", 60),
        (
            "xenium_cellpose_final_max_area_um2",
            "mask_filter",
            "final_max_area_um2",
            800,
        ),
        ("xenium_mask_max_eccentricity", "mask_filter", "max_eccentricity", 1.0),
    ]
    for name, _group, _field, value in overrides:
        cases.append({"options": {**_defaults(), name: value}, "row": {}})
    cases.append(
        {"options": {**_defaults(), "merscope_cellpose_cellprob": 0}, "row": {}}
    )
    cases.append(
        {
            "options": {**_defaults(), "xenium_cellpose_cellprob": -2},
            "row": {"xenium_cellpose_cellprob": "0"},
        }
    )
    results = _run_cases(tmp_path, cases)
    assert results[0]["before"] == results[0]["after"]
    assert results[0]["selected"] == {}
    assert results[-2]["before"] == results[-2]["after"]
    for result, (_name, group, field, value) in zip(
        results[1:6], overrides, strict=True
    ):
        after = json.loads(result["after"])
        assert after[group][field] == value
        assert (
            after["nuclei_cellpose"] == json.loads(result["before"])["nuclei_cellpose"]
        )
    assert results[-1]["selected"]["cellpose"]["cellprob_threshold"] == 0


@pytest.mark.skipif(shutil.which("nextflow") is None, reason="nextflow unavailable")
def test_model_preflight_and_protected_outdir(tmp_path: Path) -> None:
    model = tmp_path / "bundle/model"
    model.parent.mkdir()
    model.write_bytes(b"test only")
    metadata = model.with_name("model.json")
    metadata.write_text(
        json.dumps(
            {
                "weights_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                "platform": "XENIUM",
                "species": "human",
                "channels": ["DAPI", "18S"],
                "tile_size": 6144,
                "status": "candidate",
                "image_key": "morphology_focus",
                "pixel_size_um": 0.2125,
                "normalization_route": "A",
            }
        )
    )
    baseline = tmp_path / "published"
    baseline.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(baseline, target_is_directory=True)
    options = {
        **_defaults(),
        "xenium_cellpose_model": str(model.parent),
        "cellpose_protected_output_roots": [str(baseline)],
        "cellpose_tile_size_candidates": [6144],
        "allow_candidate_model": True,
        "outdir": str(tmp_path / "arm"),
    }
    cases = [{"options": options, "row": {}, "check_preflight": True}]
    store = tmp_path / "store"
    store.mkdir()
    (store / ("a" * 64)).symlink_to(model.parent, target_is_directory=True)
    for selector in [str(model), str(metadata), "a" * 64]:
        cases.append(
            {
                "options": {
                    **options,
                    "xenium_cellpose_model": selector,
                    "cellpose_model_store": str(store),
                },
                "row": {},
                "check_preflight": True,
            }
        )
    linked_bundle = tmp_path / "linked-bundle"
    linked_bundle.mkdir()
    (linked_bundle / "model").symlink_to(model)
    (linked_bundle / "model.json").write_text(metadata.read_text())
    cases.append(
        {
            "options": {**options, "xenium_cellpose_model": str(linked_bundle)},
            "row": {},
            "check_preflight": True,
        }
    )
    cases.append(
        {
            "options": {**options, "xenium_cellpose_model": "missing"},
            "row": {"xenium_cellpose_model": str(model.parent)},
            "check_preflight": True,
        }
    )
    for stage in ("build_spatialdata", "segment_nuclei"):
        cases.append(
            {
                "options": options,
                "row": {},
                "check_preflight": True,
                "run_segment": False,
                "start_stage": stage,
            }
        )
    success_count = len(cases)
    old_work = tmp_path / "old-work"
    old_work.mkdir()
    original = old_work / "published.npy"
    original.write_bytes(b"baseline")
    published_link = baseline / "published.npy"
    published_link.symlink_to(original)
    linked_arm = tmp_path / "linked-arm"
    latest = linked_arm / "P1/xenium/latest"
    latest.mkdir(parents=True)
    (latest / "latest_spatialdata.zarr").symlink_to(original)
    for changes in [
        {"allow_candidate_model": False},
        {"outdir": str(baseline / "trial")},
        {"outdir": str(alias / "trial")},
        {"cellpose_tile_size_candidates": [6144, 4096]},
        {"outdir": str(linked_arm)},
    ]:
        cases.append(
            {"options": {**options, **changes}, "row": {}, "check_preflight": True}
        )
    stock_options = {
        **options,
        "xenium_cellpose_model": None,
        "xenium_cellpose_cellprob": 0,
    }
    cases.append({"options": stock_options, "row": {}, "check_preflight": True})
    results = _run_cases(tmp_path, cases)
    assert all("error" not in result for result in results[:success_count]), results
    assert results[0]["selected"]["cellpose"]["pretrained_model"] == str(model)
    for result, message in zip(
        results[success_count:-1],
        ["candidate", "protected", "protected", "single", "protected"],
        strict=True,
    ):
        assert message in result["error"]
    assert list(baseline.iterdir()) == [published_link]
    assert original.read_bytes() == b"baseline"
    assert "error" not in results[-1]
    assert results[-1]["selected"]["cellpose_model_policy"]["experimental_arm"]


@pytest.mark.skipif(shutil.which("nextflow") is None, reason="nextflow unavailable")
def test_downstream_selector_requires_matching_published_identity(
    tmp_path: Path,
) -> None:
    """Skipping segmentation must not silently relabel stock masks as custom."""
    model = tmp_path / "bundle/model"
    model.parent.mkdir()
    model.write_bytes(b"fixture")
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    metadata = model.with_name("model.json")
    metadata.write_text(
        json.dumps(
            {
                "weights_sha256": digest,
                "platform": "XENIUM",
                "species": "human",
                "channels": ["DAPI", "18S"],
                "tile_size": 6144,
                "status": "accepted",
                "cellpose_version": "4.2.1.1",
                "torch_version": "2.14.0+cu130",
                "image_key": "morphology_focus",
                "pixel_size_um": 0.2125,
                "normalization_route": "A",
            }
        )
    )
    arm = tmp_path / "arm"
    directory = arm / "P1/xenium/segmentation"
    directory.mkdir(parents=True)
    (directory / "cellpose_masks_tiled.npy").write_bytes(b"synthetic artifact")
    identity = CellposeModelIdentity(
        status="verified",
        model_name=str(model),
        weights_sha256=digest,
        cellpose_version="4.2.1.1",
        torch_version="2.14.0+cu130",
        cellpose_config=CellposeConfig(
            pretrained_model=str(model),
            model_sha256=digest,
            model_metadata_path=metadata,
            model_type="cyto3",
            gpu=False,
            flow_threshold=0.7,
        ),
        inference_context=CellposeInferenceContext(
            platform="XENIUM",
            species="human",
            channels=["DAPI", "18S"],
            image_key="morphology_focus",
            pixel_size_um=(0.2125, 0.2125),
            mask_filter=MaskFilterConfig(),
            tiling=TilingConfig(tile_size_candidates=[6144]),
        ),
    )
    sidecar = directory / "cellpose_model_identity.json"
    sidecar.write_text(identity.model_dump_json())
    stats = directory / "cellpose_stitching_stats.json"
    stats.write_text(json.dumps({"model_identity": identity.model_dump(mode="json")}))
    options = {
        **_defaults(),
        "xenium_cellpose_model": str(model.parent),
        "cellpose_protected_output_roots": [str(tmp_path / "published")],
        "cellpose_tile_size_candidates": [6144],
        "outdir": str(arm),
    }
    stock_arm = tmp_path / "stock-arm"
    stock_dir = stock_arm / "P1/xenium/segmentation"
    stock_dir.mkdir(parents=True)
    (stock_dir / "cellpose_masks_tiled.npy").write_bytes(b"stock artifact")
    stock_identity = identity.model_copy(deep=True)
    stock_identity.model_name = "cpsam_v2"
    stock_identity.weights_sha256 = CPSAM_V2_SHA256
    stock_identity.cellpose_config = CellposeConfig(
        gpu=False,
        model_type="cyto3",
        flow_threshold=0.7,
        cellprob_threshold=0,
    )
    (stock_dir / "cellpose_model_identity.json").write_text(
        stock_identity.model_dump_json()
    )
    (stock_dir / "cellpose_stitching_stats.json").write_text(
        json.dumps({"model_identity": stock_identity.model_dump(mode="json")})
    )
    stock_options = {
        **options,
        "xenium_cellpose_model": None,
        "xenium_cellpose_cellprob": 0,
        "outdir": str(stock_arm),
    }
    cases = [
        {"options": options, "row": {}, "check_preflight": True, "run_segment": False},
        {
            "options": {**options, "xenium_cellpose_cellprob": 0},
            "row": {},
            "check_preflight": True,
            "run_segment": False,
        },
    ]
    before = sidecar.read_bytes(), stats.read_bytes()
    cases.extend(
        [
            {
                "options": stock_options,
                "row": {},
                "check_preflight": True,
                "run_segment": False,
            },
            {
                "options": {**stock_options, "xenium_cellpose_final_max_area_um2": 800},
                "row": {},
                "check_preflight": True,
                "run_segment": False,
            },
        ]
    )
    results = _run_cases(tmp_path, cases)
    assert "error" not in results[0]
    assert "identity mismatch" in results[1]["error"]
    assert "error" not in results[2], results[2]
    assert "identity mismatch" in results[3]["error"]
    assert (sidecar.read_bytes(), stats.read_bytes()) == before

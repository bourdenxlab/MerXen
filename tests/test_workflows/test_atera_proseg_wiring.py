"""Channel wiring of the ATERA ProSeg route (``native_proseg``) in ``main.nf``.

Runs the real workflow on one ATERA row with a stand-in ``merxen`` executable
that only creates the outputs each command declares and a stand-in ProSeg that
only reports its version, so the test checks the task graph (NATIVE_LATEST,
the plan, one tile task per planned tile, the stitch in the SEGMENT slot,
ENRICH and MIQ configs, the restart arm, ``-resume`` and a tile subset), not
the Python stages. The configs are validated against their Pydantic models.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from merxen.config import (
    AteraProsegPlanConfig,
    AteraProsegStitchConfig,
    AteraProsegTileConfig,
    EnrichmentConfig,
    MaskImageQuantificationConfig,
    NativeLatestConfig,
    ViewerCacheConfig,
)

_WORKFLOWS = Path(__file__).resolve().parents[2] / "workflows"

_FAKE_MERXEN = textwrap.dedent(
    """\
    #!{python}
    import json, os, sys
    from pathlib import Path
    command = sys.argv[1]
    config = json.loads(Path(sys.argv[sys.argv.index("--config") + 1]).read_text())
    call = {{"command": command, "config": config, "argv": sys.argv[1:]}}
    Path("merxen_call.json").write_text(json.dumps(call))
    if command == "build-spatialdata":
        Path(config["output_path"]).mkdir(parents=True)
    elif command == "native-latest":
        latest = Path(config["persistent_latest_zarr_path"])
        latest.mkdir(parents=True, exist_ok=True)
        out = Path(config["output_dir"])
        out.mkdir(exist_ok=True)
        os.symlink(latest, out / "native_latest.zarr")
        (out / "native_latest_report.json").write_text("{{}}")
    elif command == "atera-proseg-plan":
        out = Path(config["output_dir"])
        out.mkdir(exist_ok=True)
        tiles = []
        for tile_id, run in (("c1_r0", True), ("c0_r0", True), ("c0_r1", False)):
            tiles.append({{
                "tile_id": tile_id, "grid": [0, 0], "core_um": [0, 0, 2000, 2000],
                "window_px": [0, 100, 0, 100], "n_owned": 2 if run else 0,
                "owned_labels": [1, 2] if run else [], "est_rows": 10, "run": run,
                "reason": None if run else "no vendor cell centroid in the core",
            }})
        plan = {{
            "format_version": 1, "source_build_fingerprint": "f" * 64,
            "pixel_size_um": 0.2125, "mask_shape_px": [100, 100],
            "tile_subset": config.get("tile_subset"), "tiles": tiles,
        }}
        if config.get("tile_subset"):
            for tile in tiles:
                tile["run"] = tile["run"] and tile["tile_id"] in config["tile_subset"]
        (out / "tile_plan.json").write_text(json.dumps(plan))
    elif command == "atera-proseg-tile":
        out = Path(config["output_dir"])
        out.mkdir(exist_ok=True)
        (out / "tile_manifest.json").write_text("{{}}")
    elif command == "atera-proseg-stitch":
        out = Path(config["output_dir"])
        out.mkdir(exist_ok=True)
        os.symlink(config["persistent_latest_zarr_path"], out / "proseg_latest.zarr")
        (out / "atera_proseg_report.json").write_text("{{}}")
    else:
        Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
    """
)

_FAKE_PROSEG = "#!/bin/sh\necho 'proseg 3.2.0'\n"

_MODELS = {
    "native-latest": NativeLatestConfig,
    "atera-proseg-plan": AteraProsegPlanConfig,
    "atera-proseg-tile": AteraProsegTileConfig,
    "atera-proseg-stitch": AteraProsegStitchConfig,
    "enrich": EnrichmentConfig,
    "build-viewer-caches": ViewerCacheConfig,
    "mask-image-quantification": MaskImageQuantificationConfig,
}


def _run(
    tmp_path: Path, rows: str, name: str, *extra: str, resume: bool = False
) -> tuple[str, dict[str, list[dict]]]:
    nextflow = shutil.which("nextflow")
    assert nextflow is not None
    sheet = tmp_path / f"{name}.csv"
    sheet.write_text(rows)
    env = {k: v for k, v in os.environ.items() if not k.startswith("MERXEN_")}
    env.update(
        {
            "PATH": f"{tmp_path / 'bin'}{os.pathsep}{env.get('PATH', '')}",
            "MERXEN_SITE_CONFIG": "/dev/null",
            "MERXEN_DWIGHT_SITE_CONFIG": "/dev/null",
            "NXF_OFFLINE": "true",
            "NXF_ANSI_LOG": "false",
        }
    )
    work = tmp_path / "work"
    result = subprocess.run(
        [
            nextflow,
            "run",
            str(_WORKFLOWS / "main.nf"),
            "--samplesheet",
            str(sheet),
            "--outdir",
            str(tmp_path / "results"),
            "--proseg_binary",
            str(tmp_path / "bin" / "proseg"),
            "--proseg_auto_install",
            "false",
            "-w",
            str(work),
            *extra,
            *(["-resume"] if resume else []),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, f"{result.stdout[-3000:]}\n{result.stderr[-2000:]}"
    calls: dict[str, list[dict]] = {}
    for path in sorted(work.glob("*/*/merxen_call.json")):
        if path.stat().st_mtime < _start_times.get(name, 0):
            continue
        call = json.loads(path.read_text())
        calls.setdefault(call["command"], []).append(call["config"])
    return result.stdout, calls


_start_times: dict[str, float] = {}


def _fresh_calls(tmp_path: Path, name: str, *args: str, **kwargs: bool) -> tuple:
    import time

    _start_times[name] = time.time()
    return _run(tmp_path, *args, **kwargs)


@pytest.mark.nextflow
@pytest.mark.skipif(shutil.which("nextflow") is None, reason="nextflow is unavailable")
def test_native_proseg_route_and_its_restart_arm(tmp_path: Path) -> None:
    (tmp_path / "bin").mkdir()
    fake = tmp_path / "bin" / "merxen"
    fake.write_text(_FAKE_MERXEN.format(python=sys.executable))
    fake.chmod(0o755)
    proseg = tmp_path / "bin" / "proseg"
    proseg.write_text(_FAKE_PROSEG)
    proseg.chmod(0o755)
    (tmp_path / "raw" / "atera").mkdir(parents=True)
    row = (
        "pair_id,analysis_mode,atera_dir,segmentation_source,stop_stage\n"
        "PAIR_A,atera,raw/atera,native_proseg,mask_image_quantification\n"
    )
    stdout, calls = _fresh_calls(tmp_path, "full", row, "full")
    assert "segmentation_source=native_proseg" in stdout
    assert "analysis segmentations=reseg, original_seg" in stdout
    assert sorted(calls) == sorted(["build-spatialdata", *_MODELS])
    for command, model in _MODELS.items():
        for config in calls[command]:
            model.model_validate(config)
    tiles = sorted(config["tile_id"] for config in calls["atera-proseg-tile"])
    assert tiles == ["c0_r0", "c1_r0"]
    (stitch,) = calls["atera-proseg-stitch"]
    assert stitch["tile_dirs"] == ["tile_c0_r0", "tile_c1_r0"]
    latest = tmp_path / "results" / "PAIR_A" / "atera" / "latest"
    assert stitch["persistent_latest_zarr_path"] == str(
        (latest / "latest_spatialdata.zarr").resolve()
    )
    assert calls["enrich"][0]["segmentation_source"] == "native_proseg"
    (quant,) = calls["mask-image-quantification"]
    assert quant["mask_label_key"] == "cell_labels"
    assert quant["extra_layers"] == [
        {
            "mask_label_key": "proseg_cell_labels",
            "shape_key": "MOSAIK_proseg",
            "table_key": "table_MOSAIK_proseg_image_quantification",
        }
    ]
    tile_config = calls["atera-proseg-tile"][0]
    assert tile_config["model"]["prior_weight_effective"] == 140.0
    assert tile_config["num_threads"] == 16
    stitch_out = (
        tmp_path / "results" / "PAIR_A" / "atera" / "segmentation" / "stitch_out"
    )
    for name in ("NO_CELLPOSE_MASK", "NO_TRANSCRIPTS_CSV", "NO_NUCLEI_MASK"):
        assert (stitch_out / name).stat().st_size == 0

    # -resume: the tiles stay cached.
    stdout, calls = _fresh_calls(tmp_path, "resume", row, "resume", resume=True)
    assert "atera-proseg-tile" not in calls
    assert "atera-proseg-plan" not in calls

    # Restart at enrich: the published stitch output is the latest store.
    _, calls = _fresh_calls(
        tmp_path,
        "restart_enrich",
        "pair_id,analysis_mode,atera_dir,segmentation_source,start_stage,stop_stage\n"
        "PAIR_A,atera,raw/atera,native_proseg,enrich,mask_image_quantification\n",
        "restart_enrich",
    )
    assert sorted(calls) == [
        "build-viewer-caches",
        "enrich",
        "mask-image-quantification",
    ]
    assert calls["enrich"][0]["segmentation_source"] == "native_proseg"


@pytest.mark.nextflow
@pytest.mark.skipif(shutil.which("nextflow") is None, reason="nextflow is unavailable")
def test_a_tile_subset_runs_its_tiles_without_a_stitch(tmp_path: Path) -> None:
    (tmp_path / "bin").mkdir()
    fake = tmp_path / "bin" / "merxen"
    fake.write_text(_FAKE_MERXEN.format(python=sys.executable))
    fake.chmod(0o755)
    proseg = tmp_path / "bin" / "proseg"
    proseg.write_text(_FAKE_PROSEG)
    proseg.chmod(0o755)
    (tmp_path / "raw" / "atera").mkdir(parents=True)
    _, calls = _fresh_calls(
        tmp_path,
        "subset",
        "pair_id,analysis_mode,atera_dir,segmentation_source,stop_stage\n"
        "PAIR_A,atera,raw/atera,native_proseg,segment\n",
        "subset",
        "--atera_proseg_tile_subset",
        "c1_r0",
    )
    assert [config["tile_id"] for config in calls["atera-proseg-tile"]] == ["c1_r0"]
    assert calls["atera-proseg-plan"][0]["tile_subset"] == ["c1_r0"]
    assert "atera-proseg-stitch" not in calls

"""Channel wiring of the native (vendor-segmentation) route in ``main.nf``.

Runs the real workflow on one ATERA row with a stand-in ``merxen`` executable
that only creates the outputs each command declares, so the test checks the
task graph (NATIVE_LATEST in the SEGMENT slot, the placeholders, the restart
arms) and the rendered configs, not the Python stages. The configs are
validated against their Pydantic models.
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
    call = {{"command": command, "config": config}}
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
    else:
        Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
    """
)

_MODELS = {
    "native-latest": NativeLatestConfig,
    "enrich": EnrichmentConfig,
    "build-viewer-caches": ViewerCacheConfig,
    "mask-image-quantification": MaskImageQuantificationConfig,
}


def _run(tmp_path: Path, rows: str, name: str) -> tuple[str, dict[str, list[dict]]]:
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
    work = tmp_path / f"work_{name}"
    result = subprocess.run(
        [
            nextflow,
            "run",
            str(_WORKFLOWS / "main.nf"),
            "--samplesheet",
            str(sheet),
            "--outdir",
            str(tmp_path / "results"),
            "-w",
            str(work),
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
        call = json.loads(path.read_text())
        calls.setdefault(call["command"], []).append(call["config"])
    # On the GitHub runners the workflow's log.info lines can be missing from
    # stdout (as test_gaston_stage.py found), so read this launch's run log too.
    run_log = tmp_path / ".nextflow.log"
    log_text = run_log.read_text() if run_log.is_file() else ""
    return result.stdout + result.stderr + log_text, calls


@pytest.mark.nextflow
@pytest.mark.skipif(shutil.which("nextflow") is None, reason="nextflow is unavailable")
def test_native_route_and_its_restart_arms(tmp_path: Path) -> None:
    (tmp_path / "bin").mkdir()
    fake = tmp_path / "bin" / "merxen"
    fake.write_text(_FAKE_MERXEN.format(python=sys.executable))
    fake.chmod(0o755)
    (tmp_path / "raw" / "atera").mkdir(parents=True)

    stdout, calls = _run(
        tmp_path,
        "pair_id,analysis_mode,atera_dir,stop_stage\n"
        "PAIR_A,atera,raw/atera,mask_image_quantification\n",
        "full",
    )
    assert "selected stages=build_spatialdata -> segment -> enrich -> " in stdout
    assert sorted(calls) == sorted(["build-spatialdata", *_MODELS])
    for command, model in _MODELS.items():
        (config,) = calls[command]
        model.model_validate(config)
    assert calls["enrich"][0]["segmentation_source"] == "native"
    assert calls["mask-image-quantification"][0]["mask_label_key"] == "cell_labels"
    assert calls["build-viewer-caches"][0]["base_image_key"] == "morphology_focus"
    latest = tmp_path / "results" / "PAIR_A" / "atera" / "latest"
    assert calls["native-latest"][0]["persistent_latest_zarr_path"] == str(
        (latest / "latest_spatialdata.zarr").resolve()
    )
    native_out = (
        tmp_path / "results" / "PAIR_A" / "atera" / "segmentation" / "native_out"
    )
    for name in ("NO_CELLPOSE_MASK", "NO_TRANSCRIPTS_CSV", "NO_NUCLEI_MASK"):
        assert (native_out / name).stat().st_size == 0

    # Restart at enrich: the published NATIVE_LATEST outputs are read, and the
    # row's atera_spatialdata_path (the source store) is never used as latest.
    _, calls = _run(
        tmp_path,
        "pair_id,analysis_mode,atera_dir,start_stage,stop_stage,atera_spatialdata_path\n"
        "PAIR_A,atera,raw/atera,enrich,mask_image_quantification,raw/missing.zarr\n",
        "restart_enrich",
    )
    assert sorted(calls) == [
        "build-viewer-caches",
        "enrich",
        "mask-image-quantification",
    ]

    _, calls = _run(
        tmp_path,
        "pair_id,analysis_mode,atera_dir,only_stage,atera_spatialdata_path\n"
        "PAIR_A,atera,raw/atera,mask_image_quantification,raw/missing.zarr\n",
        "restart_quant",
    )
    assert sorted(calls) == ["mask-image-quantification"]

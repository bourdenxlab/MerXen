"""The standalone manual-registration Nextflow workflows.

Text contracts (fast tier) pin the module, the two scripts, the params, the
resources and the CLI registration. The ``nextflow`` tests launch the real
scripts: the preflight stops a bad sheet before any task, the register workflow
fans out per preset in a stub run, and one tiny synthetic pair goes through both
workflows (prepare, labelling, register, compare) as ``integration``.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from merxen import privacy
from merxen.config import ViewerCacheConfig
from merxen.manual_registration.config import (
    ManualRegistrationCompareConfig,
    ManualRegistrationImageInput,
    ManualRegistrationPrepareConfig,
    ManualRegistrationRunConfig,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / "workflows"
MODULE_TEXT = (WORKFLOWS / "modules" / "manual_registration.nf").read_text()
LIB_TEXT = (WORKFLOWS / "lib" / "ManualRegistrationSheet.groovy").read_text()
PREPARE_TEXT = (WORKFLOWS / "manual_registration_prepare.nf").read_text()
REGISTER_TEXT = (WORKFLOWS / "manual_registration_register.nf").read_text()
NEXTFLOW = shutil.which("nextflow")
needs_nextflow = pytest.mark.skipif(NEXTFLOW is None, reason="nextflow is unavailable")

STORE_ATTR = "merxen_manual_registration_store"
LABELLING_ATTR = "merxen_manual_registration"
PROCESSES = {
    "MANUAL_REGISTRATION_PREPARE": (
        "merxen manual-registration-prepare --config prepare_config.json"
    ),
    "MANUAL_REGISTRATION_REGISTER": (
        "merxen manual-registration-register --config register_config.json"
    ),
    "MANUAL_REGISTRATION_COMPARE": (
        "merxen manual-registration-compare --config compare_config.json"
    ),
}


def _groovy_map_keys(text: str, start_marker: str) -> set[str]:
    """Return the keys of the first Groovy map literal after ``start_marker``."""
    start = text.index(start_marker)
    open_index = text.index("[", start)
    depth = 0
    for index in range(open_index, len(text)):
        depth += {"[": 1, "]": -1}.get(text[index], 0)
        if depth == 0:
            body = text[open_index + 1 : index]
            break
    return set(re.findall(r"^\s{4,}([a-z_]+):", body, re.MULTILINE))


# --- text contracts (fast tier) --------------------------------------------


def test_module_declares_three_cache_false_processes_without_publish_dir() -> None:
    bodies = {
        chunk.split(" {", 1)[0]: chunk for chunk in MODULE_TEXT.split("process ")[1:]
    }

    assert set(bodies) == set(PROCESSES)
    for name, command in PROCESSES.items():
        body = bodies[name]
        for expected in (
            "cache false",
            "stub:",
            "<<'JSON'",
            'export DASK_NUM_WORKERS="${task.cpus}"',
            command,
        ):
            assert expected in body, (name, expected)
    assert "publishDir" not in MODULE_TEXT
    assert "conda" not in MODULE_TEXT
    prepare = bodies["MANUAL_REGISTRATION_PREPARE"]
    assert "ManualRegistrationSheet.cziPython(params)" in prepare
    assert "export MERXEN_CZI_PYTHON=" in prepare


def test_workflows_are_standalone() -> None:
    main_text = (WORKFLOWS / "main.nf").read_text()

    assert "manual_registration" not in main_text
    assert "MANUAL_REGISTRATION" not in main_text
    for text in (PREPARE_TEXT, REGISTER_TEXT):
        for expected in (
            "SiteConfigCheck.problems(",
            "ManualRegistrationSheet.parse(",
            '"Preflight checks failed:\\n"',
        ):
            assert expected in text
    assert 'include { VIEWER_CACHE } from "./modules/viewer_cache"' in PREPARE_TEXT
    assert "VIEWER_CACHE as" not in PREPARE_TEXT
    assert "ManualRegistrationSheet.registerPreflight(" in REGISTER_TEXT
    assert ".groupTuple()" in REGISTER_TEXT


def test_emitted_configs_match_the_pydantic_contract() -> None:
    prepare_keys = _groovy_map_keys(LIB_TEXT, "def prepareConfig = [")
    assert prepare_keys <= set(ManualRegistrationPrepareConfig.model_fields)
    assert prepare_keys == {
        "dataset_name",
        "reference",
        "moving",
        "prepared_dir",
        "output_dir",
        "force",
        "workers",
    }
    inputs = set(ManualRegistrationImageInput.model_fields)
    assert _groovy_map_keys(LIB_TEXT, "def referenceInput = [") == inputs
    assert _groovy_map_keys(LIB_TEXT, "def movingInput = [") == inputs
    assert _groovy_map_keys(LIB_TEXT, "def registerConfig = [") == set(
        ManualRegistrationRunConfig.model_fields
    )
    assert _groovy_map_keys(LIB_TEXT, "def compareConfig = [") == set(
        ManualRegistrationCompareConfig.model_fields
    )
    viewer_keys = _groovy_map_keys(LIB_TEXT, "def viewerCacheConfig = [")
    assert viewer_keys <= set(ViewerCacheConfig.model_fields)
    assert {"image_only", "image_keys"} <= viewer_keys


def _block(text: str, process: str) -> str:
    match = re.search(
        rf'withName: "{process}" \{{(.*?)\n    \}}', text, flags=re.DOTALL
    )
    assert match, process
    return match.group(1)


def test_params_resources_and_dwight_forks(
    workflow_config_params: dict[str, dict[str, str]],
) -> None:
    base = workflow_config_params["nextflow.config"]
    expected = {
        "manual_registration_presets": '"same_section"',
        "manual_registration_settings": '""',
        "manual_registration_force_prepare": "false",
        "manual_registration_force_register": "false",
        "manual_registration_czi_python": "null",
    }
    for name, value in expected.items():
        assert base[name] == value, name
    site = workflow_config_params["conf/site.config.example"]
    assert site["manual_registration_czi_python"].strip('"').startswith("/path/to/")
    for relative, params in workflow_config_params.items():
        if relative == "nextflow.config":
            continue
        defined = {name for name in params if name.startswith("manual_registration_")}
        if relative == "conf/site.config.example":
            assert defined == {"manual_registration_czi_python"}
        elif relative == "conf/dwight.config":
            assert defined == {
                "manual_registration_prepare_max_forks",
                "manual_registration_register_max_forks",
            }
        else:
            assert not defined, relative

    config = (WORKFLOWS / "nextflow.config").read_text()
    for process, cpus, memory in (
        ("MANUAL_REGISTRATION_PREPARE", 8, "16 GB"),
        ("MANUAL_REGISTRATION_REGISTER", 8, "48 GB"),
        ("MANUAL_REGISTRATION_COMPARE", 1, "4 GB"),
    ):
        block = _block(config, process)
        assert re.search(rf"cpus = {cpus}\b", block), process
        assert f'memory = "{memory}"' in block, process
        assert "conda" not in block

    dwight = (WORKFLOWS / "conf" / "dwight.config").read_text()
    assert (
        workflow_config_params["conf/dwight.config"][
            "manual_registration_prepare_max_forks"
        ]
        == "2"
    )
    assert (
        workflow_config_params["conf/dwight.config"][
            "manual_registration_register_max_forks"
        ]
        == "2"
    )
    assert "maxForks = params.manual_registration_prepare_max_forks" in _block(
        dwight, "MANUAL_REGISTRATION_PREPARE"
    )
    assert "maxForks = params.manual_registration_register_max_forks" in _block(
        dwight, "MANUAL_REGISTRATION_REGISTER"
    )


def test_example_samplesheet_is_tracked_and_clean() -> None:
    gitignore = (REPO_ROOT / ".gitignore").read_text().splitlines()
    example = WORKFLOWS / "samplesheet.manual_registration.example.csv"
    text = example.read_text()
    allowed_block = LIB_TEXT.split("ALLOWED_COLUMNS = [", 1)[1].split("]", 1)[0]
    allowed = set(re.findall(r'"([a-z_]+)"', allowed_block))
    required = {"id", "reference_image", "moving_image"}
    header = set(text.splitlines()[0].split(","))

    assert "!workflows/samplesheet.manual_registration.example.csv" in gitignore
    assert required <= header <= allowed
    assert privacy.scan_text(text) == []
    tracked = subprocess.run(
        ["git", "check-ignore", "-q", str(example)],
        cwd=REPO_ROOT,
        check=False,
    )
    assert tracked.returncode == 1  # not ignored


def test_cli_registers_the_manual_registration_commands() -> None:
    from merxen.cli import main

    for name in (
        "convert-czi",
        "manual-registration-prepare",
        "manual-registration-register",
        "manual-registration-fit",
        "manual-registration-compare",
    ):
        assert name in main.commands


def test_cli_import_does_not_load_czifile() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, merxen.cli; assert 'czifile' not in sys.modules",
        ],
        check=True,
    )


# --- Nextflow launches -----------------------------------------------------


def _nextflow_env(tmp_path: Path) -> dict[str, str]:
    env = {
        **os.environ,
        "NXF_ANSI_LOG": "false",
        "NXF_OFFLINE": "true",
        "NXF_DISABLE_CHECK_LATEST": "true",
    }
    cores = min(4, os.cpu_count() or 1)
    jvm = f"-XX:TieredStopAtLevel=1 -XX:ActiveProcessorCount={cores}"
    env["NXF_OPTS"] = f"{env.get('NXF_OPTS', '')} {jvm}".strip()
    # The `merxen` CLI of this interpreter (the CI script does not activate its venv).
    env["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{env['PATH']}"
    (tmp_path / "tmp").mkdir(exist_ok=True)
    env["TMPDIR"] = str(tmp_path / "tmp")
    return env


def _run(
    tmp_path: Path,
    script: str,
    sheet: Path,
    outdir: Path,
    *extra: str,
    launch_name: str = "launch",
    timeout: int = 900,
) -> subprocess.CompletedProcess[str]:
    assert NEXTFLOW is not None
    launch = tmp_path / launch_name
    launch.mkdir(parents=True, exist_ok=True)
    limits = tmp_path / "limits.config"
    limits.write_text("process.resourceLimits = [cpus: 2, memory: 4.GB]\n")
    return subprocess.run(
        [
            NEXTFLOW,
            "-log",
            str(launch / "nextflow.log"),
            "run",
            str(WORKFLOWS / script),
            "-c",
            str(limits),
            "-profile",
            "conda",
            "-without-conda",
            "--samplesheet",
            str(sheet),
            "--outdir",
            str(outdir),
            *extra,
        ],
        cwd=launch,
        env=_nextflow_env(tmp_path),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _write_sheet(path: Path, header: list[str], rows: list[list[str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def _fake_store(
    path: Path,
    *,
    kind: str = "prepared",
    status: str | None = "finished",
    n_enabled: int = 8,
    n_disabled: int = 0,
    store_version: int = 1,
) -> Path:
    """A directory with the root attributes of a store (no arrays)."""
    path.mkdir(parents=True, exist_ok=True)
    attrs: dict[str, Any] = {
        STORE_ATTR: {"schema_version": store_version, "kind": kind}
    }
    if status is not None:
        pairs = [
            {"id": index + 1, "enabled": index < n_enabled}
            for index in range(n_enabled + n_disabled)
        ]
        attrs[LABELLING_ATTR] = {
            "schema_version": 1,
            "status": status,
            "landmarks": {"pairs": pairs},
        }
    (path / "zarr.json").write_text(
        json.dumps({"zarr_format": 3, "node_type": "group", "attributes": attrs})
    )
    return path


def _inputs(tmp_path: Path) -> None:
    img = tmp_path / "img"
    img.mkdir(exist_ok=True)
    (img / "ref.ome.tif").write_bytes(b"")
    (img / "mov.czi").write_bytes(b"")
    (img / "mov.ome.tif").write_bytes(b"")


def _preflight_output(completed: subprocess.CompletedProcess[str]) -> str:
    text = completed.stdout + completed.stderr
    assert completed.returncode != 0, text
    assert text.count("Preflight checks failed:") == 1, text
    return text


def _no_task_ran(tmp_path: Path, launch_name: str, outdir: Path) -> None:
    work = tmp_path / launch_name / "work"
    assert not work.exists() or not any(work.iterdir())
    trace = outdir / "nextflow" / "trace.tsv"
    if trace.exists():
        assert len(trace.read_text().strip().splitlines()) <= 1


@needs_nextflow
@pytest.mark.nextflow
def test_register_preflight_stops_before_any_task(tmp_path: Path) -> None:
    _inputs(tmp_path)
    outdir = tmp_path / "out"
    header = ["id", "reference_image", "moving_image"]
    rows = [
        [ident, "../img/ref.ome.tif", "../img/mov.czi"]
        for ident in ("SYN1", "SYN2", "SYN3", "SYN4")
    ]
    sheet = _write_sheet(tmp_path / "sheets" / "a.csv", header, rows)
    _fake_store(outdir / "SYN2" / "prepared" / "SYN2_prepared.zarr", status="draft")
    _fake_store(outdir / "SYN3" / "prepared" / "SYN3_prepared.zarr", kind="result")
    _fake_store(
        outdir / "SYN4" / "prepared" / "SYN4_prepared.zarr", n_enabled=7, n_disabled=1
    )

    first = _run(tmp_path, "manual_registration_register.nf", sheet, outdir)

    text = _preflight_output(first)
    for expected in (
        "SYN1: prepared store not found",
        "SYN2: the landmarks in",
        "is a result store",
        "SYN3:",
        "SYN4 (preset same_section): 7 enabled landmark pairs",
    ):
        assert expected in text, expected
    _no_task_ran(tmp_path, "launch", outdir)

    sheet_b = _write_sheet(
        tmp_path / "sheets" / "b.csv",
        [*header, "presets", "registration_settings"],
        [
            ["SYN5", "../img/ref.ome.tif", "../img/mov.czi", "fast", ""],
            [
                "SYN6",
                "../img/ref.ome.tif",
                "../img/mov.czi",
                "",
                "serial_section:Bad=1",
            ],
        ],
    )
    second = _run(
        tmp_path,
        "manual_registration_register.nf",
        sheet_b,
        outdir,
        launch_name="launch_b",
    )

    text = _preflight_output(second)
    assert "unknown preset 'fast'" in text
    assert "key 'Bad' must match" in text
    _no_task_ran(tmp_path, "launch_b", outdir)


@needs_nextflow
@pytest.mark.nextflow
def test_prepare_preflight_reports_every_sheet_error(tmp_path: Path) -> None:
    _inputs(tmp_path)
    outdir = tmp_path / "out"
    in_process = ("--manual_registration_czi_python", "in_process")
    sheet = _write_sheet(
        tmp_path / "sheets" / "a.csv",
        ["id", "reference_image", "moving_image", "reference_scene", "foo"],
        [["SYN1", "../img/ref.ome.tif", "../img/mov.czi", "S0", "1"]],
    )

    first = _run(tmp_path, "manual_registration_prepare.nf", sheet, outdir, *in_process)

    text = _preflight_output(first)
    assert "unknown column(s): foo" in text
    assert text.count("\n - ") == 1
    _no_task_ran(tmp_path, "launch", outdir)

    sheet_b = _write_sheet(
        tmp_path / "sheets" / "b.csv",
        ["id", "reference_image", "moving_image", "reference_scene"],
        [
            ["SYN1", "../img/ref.ome.tif", "../img/mov.czi", ""],
            ["SYN1", "../img/ref.ome.tif", "../img/missing.czi", ""],
            ["SYN2", "../img/ref.ome.tif", "../img/mov.ome.tif", "S0"],
        ],
    )
    second = _run(
        tmp_path,
        "manual_registration_prepare.nf",
        sheet_b,
        outdir,
        *in_process,
        launch_name="launch_b",
    )

    text = _preflight_output(second)
    assert "id 'SYN1' is also used in row 2" in text
    assert "moving_image not found" in text
    assert "reference_scene is only valid for a .czi reference_image" in text
    assert text.count("\n - ") == 3
    _no_task_ran(tmp_path, "launch_b", outdir)

    sheet_c = _write_sheet(
        tmp_path / "sheets" / "c.csv",
        ["id", "reference_image", "moving_image"],
        [["SYN1", "../img/ref.ome.tif", "../img/mov.czi"]],
    )
    third = _run(
        tmp_path,
        "manual_registration_prepare.nf",
        sheet_c,
        outdir,
        launch_name="launch_c",
    )

    text = _preflight_output(third)
    assert (
        "moving_image is a .czi file but --manual_registration_czi_python is not set"
        in text
    )
    assert text.count("\n - ") == 1
    _no_task_ran(tmp_path, "launch_c", outdir)


def _trace_rows(outdir: Path) -> list[dict[str, str]]:
    text = (outdir / "nextflow" / "trace.tsv").read_text()
    return list(csv.DictReader(text.splitlines(), delimiter="\t"))


def _task_dir(row: dict[str, str]) -> Path:
    return Path(row["workdir"])


@needs_nextflow
@pytest.mark.nextflow
def test_register_stub_run_fans_out_presets_and_compares(tmp_path: Path) -> None:
    _inputs(tmp_path)
    outdir = tmp_path / "out"
    for ident in ("SYN_A", "SYN_B", "SYN_C"):
        _fake_store(outdir / ident / "prepared" / f"{ident}_prepared.zarr")
    sheet = _write_sheet(
        tmp_path / "sheets" / "a.csv",
        ["id", "reference_image", "moving_image", "presets", "run_label"],
        [
            [
                "SYN_A",
                "../img/ref.ome.tif",
                "../img/mov.ome.tif",
                "same_section;serial_section",
                "",
            ],
            ["SYN_B", "../img/ref.ome.tif", "../img/mov.ome.tif", "same_section", ""],
            [
                "SYN_C",
                "../img/ref.ome.tif",
                "../img/mov.ome.tif",
                "serial_section",
                "trial",
            ],
        ],
    )

    completed = _run(
        tmp_path, "manual_registration_register.nf", sheet, outdir, "-stub-run"
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    trace = (outdir / "nextflow" / "trace.tsv").read_text()
    assert len(re.findall(r"\tMANUAL_REGISTRATION_REGISTER\t", trace)) == 4
    assert len(re.findall(r"\tMANUAL_REGISTRATION_COMPARE\t", trace)) == 1
    rows = _trace_rows(outdir)
    tags = {row["tag"] for row in rows if row["process"].endswith("REGISTER")}
    assert tags == {
        "SYN_A_same_section",
        "SYN_A_serial_section",
        "SYN_B_same_section",
        "SYN_C_serial_section_trial",
    }
    outdir_abs = outdir.resolve()
    (syn_c,) = [row for row in rows if row["tag"] == "SYN_C_serial_section_trial"]
    script = (_task_dir(syn_c) / ".command.sh").read_text()
    assert (
        f'"result_dir": "{outdir_abs}/SYN_C/registered/serial_section_trial"' in script
    )
    assert '"run_label": "trial"' in script
    (compare,) = [row for row in rows if row["process"].endswith("COMPARE")]
    compare_script = (_task_dir(compare) / ".command.sh").read_text()
    registered = f"{outdir_abs}/SYN_A/registered"
    same = (
        f"{registered}/same_section/SYN_A_same_section_manual_registration_summary.json"
    )
    serial = (
        f"{registered}/serial_section/"
        "SYN_A_serial_section_manual_registration_summary.json"
    )
    assert same in compare_script and serial in compare_script
    assert compare_script.index(same) < compare_script.index(serial)


# --- tiny end to end through both workflows --------------------------------


@needs_nextflow
@pytest.mark.nextflow
@pytest.mark.integration
def test_tiny_end_to_end_through_both_workflows(tmp_path: Path) -> None:
    from merxen.image_registration import load_alignment_matrix
    from merxen.manual_registration.schema import (
        RESULT_ATTR,
        canonical_json,
        read_root_attrs,
    )
    from merxen.manual_registration.simulate import (
        inject_labelling,
        make_synthetic_pair,
        simulate_landmark_pairs,
    )

    pair = make_synthetic_pair(
        tmp_path / "data", reference_shape_yx=(1280, 1280), seed=0
    )
    outdir = tmp_path / "out"
    sheet = _write_sheet(
        tmp_path / "sheet.csv",
        ["id", "reference_image", "moving_image", "presets"],
        [
            [
                "SYN1",
                str(pair.reference_path),
                str(pair.moving_path),
                "same_section;serial_section",
            ]
        ],
    )
    first = _run(
        tmp_path,
        "manual_registration_prepare.nf",
        sheet,
        outdir,
        launch_name="launch_prepare",
    )
    assert first.returncode == 0, first.stdout + first.stderr
    store = outdir / "SYN1" / "prepared" / "SYN1_prepared.zarr"
    assert read_root_attrs(store)[STORE_ATTR]["kind"] == "prepared"
    for element in (
        "reference",
        "moving",
        "_napari_compare_imgpyr__reference__ds4",
        "_napari_compare_imgpyr__moving__ds4",
    ):
        assert (store / "images" / element).is_dir(), element
    cache = outdir / "SYN1" / "prepared" / "viewer_cache" / "viewer_cache_out"
    assert (cache / "syn1_prepared_viewer_cache_summary.json").is_file()

    # Moving points: a 5 x 4 grid over the central 60 % of the moving image.
    height, width = pair.moving_shape_yx
    xs = np.linspace(0.2 * width, 0.8 * width, 5)
    ys = np.linspace(0.2 * height, 0.8 * height, 4)
    points = np.array([(x, y) for y in ys for x in xs])
    mapped = pair.moving_to_reference(points)
    inside = np.all((mapped >= 64) & (mapped < 1280 - 64), axis=1)
    points = points[inside][:12]
    assert len(points) == 12
    landmarks = simulate_landmark_pairs(
        points,
        pair.moving_to_reference,
        noise_um=0.5,
        reference_pixel_size_um=2.0,
        seed=0,
        disabled=[11, 12],
    )
    inject_labelling(store, landmarks)
    before = canonical_json(read_root_attrs(store)[LABELLING_ATTR])

    second = _run(
        tmp_path,
        "manual_registration_register.nf",
        sheet,
        outdir,
        launch_name="launch_register",
    )
    assert second.returncode == 0, second.stdout + second.stderr
    corners = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], float
    )

    def apply(matrix: np.ndarray, xy: np.ndarray) -> np.ndarray:
        return xy @ matrix[:2, :2].T + matrix[:2, 2]

    summaries = []
    for preset in ("same_section", "serial_section"):
        run_id = f"SYN1_{preset}"
        folder = outdir / "SYN1" / "registered" / preset
        result_store = folder / f"{run_id}_registered.zarr"
        attrs = read_root_attrs(result_store)
        assert attrs[STORE_ATTR]["kind"] == "result"
        assert attrs[RESULT_ATTR]["status"] == "complete"
        landmark = load_alignment_matrix(folder / f"{run_id}_landmark_matrix.csv")
        load_alignment_matrix(folder / f"{run_id}_final_matrix.csv")
        with (folder / f"{run_id}_landmarks.csv").open() as handle:
            assert len(list(csv.DictReader(handle))) == 12
        assert (folder / f"{run_id}_manual_registration_summary.json").is_file()
        assert (folder / f"{run_id}_tissue_qc.png").is_file()
        error = np.linalg.norm(
            apply(landmark, corners) - apply(pair.true_matrix, corners), axis=1
        ).max()
        assert error <= 2.0, (preset, error)
        summaries.append(preset)
    comparison = outdir / "SYN1" / "registered" / "SYN1_run_comparison.csv"
    with comparison.open() as handle:
        compared = list(csv.DictReader(handle))
    assert {row["preset"] for row in compared} == set(summaries)
    assert len(compared) == 2
    assert list(compared[0])[:3] == ["run_id", "preset", "run_label"]
    for suffix in (".json", ".png"):
        assert comparison.with_suffix(suffix).is_file()
    rows = _trace_rows(outdir)
    assert sum(r["process"].endswith("REGISTER") for r in rows) == 2
    assert sum(r["process"].endswith("COMPARE") for r in rows) == 1

    third = _run(
        tmp_path,
        "manual_registration_prepare.nf",
        sheet,
        outdir,
        launch_name="launch_prepare_again",
    )
    assert third.returncode == 0, third.stdout + third.stderr
    after = canonical_json(read_root_attrs(store)[LABELLING_ATTR])
    assert after == before

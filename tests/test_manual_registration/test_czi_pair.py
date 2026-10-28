"""Test landmark derivation and the Zeiss CZI pair validation (real data, slow).

``test_czi_pair_both_presets`` runs the two manual-registration workflows through
Nextflow on a real Zeiss CZI pair (no ground truth): prepare, landmarks derived by
``scripts/manual_registration/derive_test_landmarks.py``, register with both
presets, compare. The gates of the ``serial_section`` run are plausibility checks
of the fit and the QC; ``same_section`` must complete, its numbers are recorded.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import spatialdata as sd

from merxen.io.spatialdata_schema import validate_merxen_schema
from merxen.manual_registration.config import (
    ManualRegistrationImageInput,
    ManualRegistrationPrepareConfig,
)
from merxen.manual_registration.geometry import points_outside_crop, wrap_deg
from merxen.manual_registration.prepare import run_manual_registration_prepare
from merxen.manual_registration.presets import PRESETS
from merxen.manual_registration.schema import (
    LABELLING_ATTR,
    STORE_ATTR,
    read_root_attrs,
)
from merxen.manual_registration.simulate import make_synthetic_pair

from ._mr_helpers import A2_COARSE, a2_pairs, a2_store
from ._scripts import load_script

derive_script = load_script("derive_test_landmarks")
REPO_ROOT = Path(__file__).resolve().parents[2]


def test_farthest_point_order_spreads_points() -> None:
    points = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [5.0, 0.0], [0.0, 9.0]])

    order = derive_script.farthest_point_order(points)

    # Nearest the centroid (3), the farthest from it (4), then the tie between 0 and
    # 2 (both 5 from the chosen set) goes to the lowest index.
    assert order.tolist() == [3, 4, 0, 2, 1]
    assert derive_script.farthest_point_order(np.zeros((0, 2))).size == 0


def test_tissue_mask_finds_the_bright_disc() -> None:
    yy, xx = np.mgrid[:200, :200]
    image = np.where((yy - 100) ** 2 + (xx - 100) ** 2 < 60**2, 500.0, 5.0)
    image = image + np.random.default_rng(0).normal(0, 1, image.shape)

    mask = derive_script.tissue_mask(np.clip(image, 0, None))

    assert 0.25 < mask.mean() < 0.35
    assert mask[100, 100] and not mask[5, 5]


def test_scripts_refuse_an_output_inside_the_repository() -> None:
    inside = REPO_ROOT / "derived_landmarks.json"
    with pytest.raises(SystemExit, match="repository"):
        derive_script.main(["--store", "unused.zarr", "--out", str(inside)])
    with pytest.raises(SystemExit, match="repository"):
        load_script("export_reference_image").main(
            ["--store", "unused.zarr", "--out", str(REPO_ROOT / "export.ome.tif")]
        )


@pytest.mark.integration
def test_derive_test_landmarks_recovers_a_rotated_mirrored_pair(
    tmp_path: Path,
) -> None:
    pair = make_synthetic_pair(
        tmp_path,
        reference_shape_yx=(2048, 2048),
        rotation_deg=178.0,
        mirror_x=True,
        warp_amplitude_um=0.0,
    )
    outputs = run_manual_registration_prepare(
        ManualRegistrationPrepareConfig(
            dataset_name="SYN1",
            reference=ManualRegistrationImageInput(path=pair.reference_path),
            moving=ManualRegistrationImageInput(path=pair.moving_path),
            prepared_dir=tmp_path / "prepared",
            output_dir=tmp_path / "prepare_out",
            workers=1,
        )
    )

    derived = derive_script.derive(
        outputs["store"],
        n_pairs=12,
        n_disabled=4,
        working_pixel_size_um=8.0,
        min_inliers=12,
    )

    assert derived.coarse.mirror_x is True
    assert abs(wrap_deg(derived.coarse.rotation_deg - 178.0)) <= 10.0
    height, width = pair.moving_shape_yx
    corners = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], float
    )

    def apply(matrix: np.ndarray, xy: np.ndarray) -> np.ndarray:
        return xy @ matrix[:2, :2].T + matrix[:2, 2]

    error = np.linalg.norm(
        apply(derived.matrix_full, corners) - apply(pair.true_matrix, corners), axis=1
    ).max()
    assert error <= 4.0, error
    assert len(derived.pairs) == 12
    assert [p.id for p in derived.pairs if not p.enabled] == [3, 6, 9, 12]
    reference = np.array([p.reference_xy_px for p in derived.pairs])
    moving = np.array([p.moving_xy_px for p in derived.pairs])
    assert not points_outside_crop(reference, derived.crops["reference"]).any()
    assert not points_outside_crop(moving, derived.crops["moving"]).any()


# --------------------------------------------------------------------------
# --inject never replaces a labelling that was not written by simulate
# --------------------------------------------------------------------------


def _derived(*, disabled: tuple[int, ...] = (3, 6)) -> Any:
    return derive_script.DerivedLandmarks(
        matrix_full=np.eye(3),
        coarse=A2_COARSE,
        scale_ratio=1.0,
        crops={"reference": None, "moving": None},
        pairs=a2_pairs(disabled=disabled),
        diagnostics={},
    )


def _labelling(store: Path) -> dict[str, Any]:
    return read_root_attrs(store)[LABELLING_ATTR]  # type: ignore[no-any-return]


def _backups(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.previous_labelling.*.json"))


@pytest.mark.parametrize("status", ["finished", "draft"])
def test_inject_refuses_a_labelling_written_elsewhere(
    tmp_path: Path, status: str
) -> None:
    # a2_store writes a labelling with another written_by.software (a viewer save).
    store = a2_store(tmp_path, disabled=(1,), status=status)
    before = _labelling(store)
    out = tmp_path / "landmarks.json"

    with pytest.raises(derive_script.ForeignLabellingError, match="--force"):
        derive_script._inject(store, _derived(), out)

    assert _labelling(store) == before
    assert _backups(tmp_path) == []


def test_inject_force_saves_the_old_labelling_before_replacing_it(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    store = a2_store(tmp_path, disabled=(1,))
    before = _labelling(store)
    out = tmp_path / "runs" / "landmarks.json"

    with caplog.at_level("WARNING"):
        revision = derive_script._inject(store, _derived(), out, force=True)

    (backup,) = _backups(out.parent)
    assert backup.parent == out.parent
    assert backup.name.startswith("landmarks.previous_labelling.20")
    assert backup.name.endswith("Z.json")
    assert json.loads(backup.read_text(encoding="utf-8")) == before
    assert str(backup) in caplog.text
    after = _labelling(store)
    assert revision == before["revision"] + 1 == after["revision"]
    assert after["written_by"]["software"] == derive_script.SIMULATE_SOFTWARE
    assert after["status"] == "finished"
    assert [p["id"] for p in after["landmarks"]["pairs"] if not p["enabled"]] == [3, 6]


def test_inject_replaces_its_own_labelling_without_force_or_backup(
    tmp_path: Path,
) -> None:
    store = a2_store(tmp_path)
    out = tmp_path / "landmarks.json"
    derive_script._inject(store, _derived(), out, force=True)  # now simulate's
    (first_backup,) = _backups(tmp_path)
    first_backup.unlink()

    assert derive_script._inject(store, _derived(disabled=(2,)), out) == 3
    assert (
        _labelling(store)["written_by"]["software"] == derive_script.SIMULATE_SOFTWARE
    )
    assert _backups(tmp_path) == []


def test_inject_leaves_an_identical_finished_labelling_alone(tmp_path: Path) -> None:
    store = a2_store(tmp_path, disabled=(3, 6), coarse=A2_COARSE)
    before = _labelling(store)

    # The content equals the derived one, so nothing is written even though the
    # labelling is not simulate's and --force was not given.
    assert derive_script._inject(store, _derived(), tmp_path / "landmarks.json") is None

    assert _labelling(store) == before
    assert _backups(tmp_path) == []


def test_main_refuses_a_foreign_labelling_before_deriving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = a2_store(tmp_path, disabled=(1,))
    before = _labelling(store)
    out = tmp_path / "landmarks.json"

    def no_derivation(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("the derivation must not start")

    monkeypatch.setattr(derive_script, "derive", no_derivation)
    with pytest.raises(SystemExit, match="--force"):
        derive_script.main(["--store", str(store), "--out", str(out), "--inject"])
    assert _labelling(store) == before and not out.exists()

    with pytest.raises(SystemExit) as parsed:
        derive_script.main(["--store", str(store), "--out", str(out), "--force"])
    assert parsed.value.code == 2


def test_main_force_replaces_after_saving_the_old_labelling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = a2_store(tmp_path, disabled=(1,))
    before = _labelling(store)
    out = tmp_path / "landmarks.json"
    monkeypatch.setattr(derive_script, "derive", lambda *_a, **_k: _derived())
    monkeypatch.setattr(
        derive_script, "_output_document", lambda *_a, **_k: {"schema": "x"}
    )

    status = derive_script.main(
        ["--store", str(store), "--out", str(out), "--inject", "--force"]
    )

    assert status == 0
    (backup,) = _backups(tmp_path)
    assert json.loads(backup.read_text(encoding="utf-8")) == before
    assert (
        _labelling(store)["written_by"]["software"] == derive_script.SIMULATE_SOFTWARE
    )


# --------------------------------------------------------------------------
# The real CZI pair through Nextflow (slow; never in CI)
# --------------------------------------------------------------------------

WORKFLOWS = REPO_ROOT / "workflows"
SAME_SECTION_OVERRIDE = "same_section:matrix_scale_tolerance=0.05"


def _nextflow(
    script: str,
    *,
    launch: Path,
    sheet: Path,
    outdir: Path,
    czi_python: Path,
    task_tmp: Path,
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    nextflow = shutil.which("nextflow")
    if nextflow is None:
        pytest.skip("nextflow is unavailable")
    launch.mkdir(parents=True, exist_ok=True)
    environment = {
        **os.environ,
        "NXF_ANSI_LOG": "false",
        "NXF_DISABLE_CHECK_LATEST": "true",
        "TMPDIR": str(task_tmp),
        "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}",
    }
    return subprocess.run(
        [
            nextflow,
            "-log",
            str(launch / "nextflow.log"),
            "run",
            str(WORKFLOWS / script),
            "-profile",
            "conda",
            "-without-conda",
            "--samplesheet",
            str(sheet),
            "--outdir",
            str(outdir),
            "--manual_registration_czi_python",
            str(czi_python),
        ],
        cwd=launch,
        env=environment,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _held_out(summary: dict[str, Any], stage: str) -> float | None:
    return summary["residual_summary_um"]["held_out"][stage]["median"]  # type: ignore[no-any-return]


@pytest.mark.slow
def test_czi_pair_both_presets(real_path: Callable[[str], Path]) -> None:  # noqa: C901
    test_dir = real_path("MERXEN_MANUAL_REGISTRATION_TEST_DIR")
    evidence = real_path("MERXEN_EVIDENCE_ROOT")
    task_tmp = real_path("MERXEN_TASK_TMP_DIR")
    czi_python = real_path("MERXEN_CZI_PYTHON")
    manifest_path = evidence / "manual_registration" / "czi_test_pairs.json"
    if not manifest_path.is_file():
        pytest.skip(
            "the CZI test manifest does not exist (docs/stages/manual-registration.md)"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    alias = manifest["default"]
    entry = manifest["pairs"][alias]
    runs = evidence / "manual_registration" / "runs"
    outdir = runs / "out"
    runs.mkdir(parents=True, exist_ok=True)
    sheet = runs / "samplesheet.csv"
    with sheet.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "id",
                "reference_image",
                "moving_image",
                "presets",
                "registration_settings",
            ]
        )
        writer.writerow(
            [
                alias,
                str(test_dir / entry["reference"]),
                str(test_dir / entry["moving"]),
                "same_section;serial_section",
                SAME_SECTION_OVERRIDE,
            ]
        )

    # 1. Prepare through Nextflow (the real resource requests apply).
    prepared = _nextflow(
        "manual_registration_prepare.nf",
        launch=runs / "launch_prepare",
        sheet=sheet,
        outdir=outdir,
        czi_python=czi_python,
        task_tmp=task_tmp,
        timeout=6 * 3600,
    )
    assert prepared.returncode == 0, prepared.stdout[-3000:] + prepared.stderr[-3000:]
    prepared_dir = outdir / alias / "prepared"
    store = prepared_dir / f"{alias}_prepared.zarr"
    summary = json.loads(
        (prepared_dir / f"{alias}_prepare_summary.json").read_text(encoding="utf-8")
    )
    for role in ("reference", "moving"):
        assert summary["images"][role]["czi"] is not None, role
        provenance = json.loads(
            (prepared_dir / f"{alias}_{role}.prepare.json").read_text(encoding="utf-8")
        )
        assert provenance["conversion"]["converter"].startswith("czifile=="), role

    # 2. Derive and inject test landmarks, unless the store holds someone's own.
    if derive_script.is_foreign_labelling(read_root_attrs(store).get(LABELLING_ATTR)):
        pytest.skip(
            f"the prepared store of {alias} holds a labelling that "
            f"{derive_script.SIMULATE_SOFTWARE} did not write (landmarks placed in "
            "the viewer): this test does not replace it. To run the test on derived "
            "landmarks, run scripts/manual_registration/derive_test_landmarks.py "
            "--inject --force on the store (the old labelling is saved first)"
        )
    derived = derive_script.derive(store, n_pairs=24, n_disabled=8)
    landmarks_json = runs / f"{alias}_test_landmarks.json"
    landmarks_json.write_text(
        json.dumps(derive_script._output_document(store, derived), indent=2) + "\n",
        encoding="utf-8",
    )
    derive_script._inject(store, derived, landmarks_json)

    # 3. Register both presets through Nextflow.
    registered = _nextflow(
        "manual_registration_register.nf",
        launch=runs / "launch_register",
        sheet=sheet,
        outdir=outdir,
        czi_python=czi_python,
        task_tmp=task_tmp,
        timeout=12 * 3600,
    )
    assert registered.returncode == 0, (
        registered.stdout[-3000:] + registered.stderr[-3000:]
    )

    # 4. Read the results.
    summaries: dict[str, dict[str, Any]] = {}
    for preset in ("same_section", "serial_section"):
        path = (
            outdir
            / alias
            / "registered"
            / preset
            / f"{alias}_{preset}_manual_registration_summary.json"
        )
        summaries[preset] = json.loads(path.read_text(encoding="utf-8"))
    comparison = outdir / alias / "registered" / f"{alias}_run_comparison.csv"
    with comparison.open(encoding="utf-8") as handle:
        compared = list(csv.DictReader(handle))

    measured: dict[str, Any] = {
        "alias": alias,
        "derive": derived.diagnostics,
        "scale_ratio_matrix": derived.scale_ratio,
        "runs": {
            preset: {
                "fit": result["fit"],
                "refinement": result["refinement"],
                "nonrigid": result["nonrigid"],
                "tissue_pearson": result["tissue_qc"]["pearson"],
                "held_out_median_um": {
                    stage: _held_out(result, stage)
                    for stage in ("landmark", "affine", "final")
                },
                "nucleus_qc": result["nucleus_qc"],
                "resources": result["resources"],
                "warnings": result["warnings"],
            }
            for preset, result in summaries.items()
        },
    }
    failures: list[str] = []

    def gate(name: str, passed: bool, detail: Any) -> None:
        measured.setdefault("gates", {})[name] = {
            "passed": bool(passed),
            "detail": detail,
        }
        if not passed:
            failures.append(f"{name}: {detail}")

    # The override is the documented way to run same_section on a serial pair. These
    # are gates like the others: a failure is recorded and asserted after the
    # numbers are written, so a failed run keeps its measurements.
    gate(
        "0_override",
        all(
            result["settings"]["matrix_scale_tolerance"] == 0.05
            for result in summaries.values()
        )
        and PRESETS["same_section"].get("matrix_scale_tolerance", 0.02) != 0.05,
        {
            preset: result["settings"]["matrix_scale_tolerance"]
            for preset, result in summaries.items()
        },
    )
    gate(
        "0_complete",
        all(result["status"] == "complete" for result in summaries.values()),
        {preset: result["status"] for preset, result in summaries.items()},
    )

    serial = summaries["serial_section"]
    fit = serial["fit"]
    gate(
        "1_fit",
        fit["n_used"] >= 0.75 * fit["n_enabled"]
        and abs(fit["scale_ratio"] - 1.0) <= 0.05
        and fit["reflected"] == derived.coarse.mirror_x
        and fit["other_class_median_um"] >= 2.0 * fit["chosen_class_median_um"]
        and fit["spread_ratio"] >= 0.2,
        {
            key: fit[key]
            for key in (
                "n_used",
                "n_enabled",
                "scale_ratio",
                "reflected",
                "chosen_class_median_um",
                "other_class_median_um",
                "spread_ratio",
            )
        },
    )
    pearson = serial["tissue_qc"]["pearson"]
    gate(
        "2_tissue_pearson",
        pearson["affine"] >= pearson["landmark"] - 0.01
        and pearson["final"] >= pearson["affine"] - 0.01,
        pearson,
    )
    held_final = _held_out(serial, "final")
    held_landmark = _held_out(serial, "landmark")
    gate(
        "3_held_out",
        held_final is not None
        and held_landmark is not None
        and held_final <= held_landmark + 5.0,
        {"final": held_final, "landmark": held_landmark},
    )
    nonrigid = serial["nonrigid"]
    if nonrigid["status"] == "applied":
        flow = nonrigid["final_flow"]
        gate(
            "4_nonrigid",
            flow["min_jacobian"] >= 0.67 and flow["max_jacobian"] <= 1.5,
            flow,
        )
    else:
        gate("4_nonrigid", nonrigid["reason"] is not None, nonrigid)
    stores = {
        preset: outdir
        / alias
        / "registered"
        / preset
        / f"{alias}_{preset}_registered.zarr"
        for preset in summaries
    }
    ok = len(set(stores.values())) == 2
    for result_store in stores.values():
        ok = ok and read_root_attrs(result_store)[STORE_ATTR]["kind"] == "result"
        validate_merxen_schema(sd.read_zarr(result_store, selection=("images",)))
    ok = ok and len(compared) == 2
    ok = ok and comparison.with_suffix(".png").is_file()
    gate("5_stores_and_comparison", ok, {"rows": len(compared)})
    gate(
        "6_resources",
        all(r["resources"]["peak_rss_gb"] <= 48.0 for r in summaries.values()),
        {p: r["resources"] for p, r in summaries.items()},
    )

    (runs / f"{alias}_validation.json").write_text(
        json.dumps(measured, indent=2, default=float) + "\n", encoding="utf-8"
    )
    assert not failures, failures

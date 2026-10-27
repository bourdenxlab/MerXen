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
from merxen.manual_registration.schema import STORE_ATTR, read_root_attrs
from merxen.manual_registration.simulate import make_synthetic_pair

from ._scripts import load_script

derive_script = load_script("derive_test_landmarks")


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
# The real CZI pair through Nextflow (slow; never in CI)
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
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

    # 2. Derive and inject test landmarks.
    derived = derive_script.derive(store, n_pairs=24, n_disabled=8)
    landmarks_json = runs / f"{alias}_test_landmarks.json"
    landmarks_json.write_text(
        json.dumps(derive_script._output_document(store, derived), indent=2) + "\n",
        encoding="utf-8",
    )
    derive_script._inject(store, derived)

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

    # The override is the documented way to run same_section on a serial pair.
    for preset, result in summaries.items():
        assert result["settings"]["matrix_scale_tolerance"] == 0.05, preset
    assert PRESETS["same_section"].get("matrix_scale_tolerance", 0.02) != 0.05
    assert summaries["same_section"]["status"] == "complete"

    serial = summaries["serial_section"]
    assert serial["status"] == "complete"
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

"""Ground truth of the manual registration on HD1 (real data) and the export script.

HD1 has a post-Xenium IF image that the main pipeline registered with the Xenium
Explorer matrix ``E``, a DAPI affine refinement (final matrix ``F``) and an
optical flow. The ``slow`` test reproduces that registration from simulated
landmarks (``E`` plus noise, one outlier) and measures the manual registration
against HD1's own result. The unit tests cover ``export_reference_image.py``,
which turns the Xenium ``morphology_focus`` DAPI into the reference OME-TIFF.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import spatialdata as sd
import xarray as xr
from skimage.filters import threshold_otsu
from spatialdata.models import Image2DModel
from spatialdata.transformations import Identity, Scale

from merxen import privacy
from merxen.image_registration import (
    check_matrix_scale,
    load_alignment_matrix,
    push_points_through_registration,
)
from merxen.io.ome_tiff import open_ome_tiff_level, read_ome_tiff_info
from merxen.manual_registration.config import (
    ManualRegistrationImageInput,
    ManualRegistrationPrepareConfig,
    ManualRegistrationRunConfig,
    ManualRegistrationSettings,
)
from merxen.manual_registration.geometry import coarse_from_matrix
from merxen.manual_registration.prepare import run_manual_registration_prepare
from merxen.manual_registration.presets import FORBIDDEN_OVERRIDES
from merxen.manual_registration.register import run_manual_registration
from merxen.manual_registration.schema import (
    STORE_ATTR,
    CropRect,
    LandmarkPair,
    pair_colour,
    read_root_attrs,
)
from merxen.manual_registration.simulate import (
    inject_labelling,
    sample_tissue_points,
)

from ._scripts import load_script

export_script = load_script("export_reference_image")
derive_script = load_script("derive_test_landmarks")


def _write_store(
    path: Path, transformation: Any, scale_factors: list[int] | None = None
) -> np.ndarray:
    data = np.random.default_rng(0).integers(
        0, 4000, size=(2, 1300, 1100), dtype=np.uint16
    )
    image = Image2DModel.parse(
        data,
        dims=("c", "y", "x"),
        c_coords=["DAPI", "Other"],
        scale_factors=scale_factors,
        transformations={"global": transformation},
    )
    sd.SpatialData(images={"morphology_focus": image}).write(path)
    return data


@pytest.mark.integration
@pytest.mark.parametrize("scale_factors", [None, [2, 2]])
def test_export_reference_image_round_trip(
    tmp_path: Path, scale_factors: list[int] | None
) -> None:
    store = tmp_path / "store.zarr"
    data = _write_store(store, Scale([0.5, 0.5], axes=("x", "y")), scale_factors)
    out = tmp_path / "reference.ome.tif"

    export_script.export_channel(
        store,
        image_key="morphology_focus",
        channel="DAPI",
        out=out,
        pixel_size_um=None,
        workers=1,
    )

    info = read_ome_tiff_info(out)
    assert info.channel_names == ("DAPI",)
    assert info.pixel_size_um == (0.5, 0.5)
    level0 = open_ome_tiff_level(info, 0).values
    np.testing.assert_array_equal(level0[0], data[0])
    # A second call finds the export up to date.
    assert (
        export_script.export_channel(
            store,
            image_key="morphology_focus",
            channel="dapi",
            out=out,
            pixel_size_um=None,
            workers=1,
        )
        is None
    )


@pytest.mark.integration
def test_export_reference_image_identity_transform(tmp_path: Path) -> None:
    store = tmp_path / "store.zarr"
    _write_store(store, Identity())
    out = tmp_path / "reference.ome.tif"

    with pytest.raises(ValueError, match="--pixel-size-um"):
        export_script.export_channel(
            store,
            image_key="morphology_focus",
            channel="DAPI",
            out=out,
            pixel_size_um=None,
            workers=1,
        )
    assert not out.exists()

    export_script.export_channel(
        store,
        image_key="morphology_focus",
        channel="DAPI",
        out=out,
        pixel_size_um=0.2125,
        workers=1,
    )
    assert read_ome_tiff_info(out).pixel_size_um == pytest.approx((0.2125, 0.2125))

    second = tmp_path / "second.ome.tif"
    status = export_script.main(
        [
            "--store",
            str(store),
            "--channel",
            "DAPI",
            "--out",
            str(second),
            "--pixel-size-um",
            "0.2125",
            "--workers",
            "1",
        ]
    )
    assert status == 0 and second.is_file()


# --------------------------------------------------------------------------
# HD1 ground truth (real data; never in CI)
# --------------------------------------------------------------------------


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return pts @ matrix[:2, :2].T + matrix[:2, 2]


def _downsampled_dapi(sdata: Any, key: str, channel: str) -> np.ndarray:
    element = sdata.images[key]
    if isinstance(element, xr.DataArray):
        level0 = element
    else:
        dataset = element["scale0"].ds
        level0 = dataset[next(iter(dataset.data_vars))]
    names = [str(name) for name in level0.coords["c"].values]
    plane = level0.isel(c=names.index(channel))
    return np.asarray(
        plane.coarsen(y=16, x=16, boundary="trim").mean().data.compute(),
        dtype=np.float64,
    )


@pytest.mark.slow
def test_hd1_ground_truth(  # noqa: C901
    real_path: Callable[[str], Path], real_aliases: Path
) -> None:
    results_root = real_path("MERXEN_RESULTS_ROOT")
    task_tmp = real_path("MERXEN_TASK_TMP_DIR")
    evidence = real_path("MERXEN_EVIDENCE_ROOT")
    pid = privacy.to_real("HD1")
    source_store = results_root / pid / "xenium" / "latest" / "latest_spatialdata.zarr"
    entry = read_root_attrs(source_store)["merxen_registered_images"]["post_xenium_if"]
    explorer = np.asarray(entry["provenance"]["explorer_matrix"], dtype=np.float64)
    final_truth = np.asarray(entry["final_matrix"], dtype=np.float64)
    summary_hd1 = entry["summary"]
    p_r = float(summary_hd1["reference_pixel_size_um"])
    p_m = float(summary_hd1["image_pixel_size_um"])
    work = evidence / "manual_registration" / "runs" / "HD1"
    work.mkdir(parents=True, exist_ok=True)
    measured: dict[str, Any] = {}
    failures: list[str] = []

    def gate(name: str, value: Any, passed: bool, text: str) -> None:
        measured[name] = {"value": value, "gate": text, "passed": bool(passed)}
        if not passed:
            failures.append(f"{name}: {value} (gate: {text})")

    # 1. Export the reference DAPI and prepare.
    export = task_tmp / "manual_registration" / "HD1_reference_dapi.ome.tif"
    export_script.export_channel(
        source_store,
        image_key="morphology_focus",
        channel="DAPI",
        out=export,
        pixel_size_um=p_r,
        workers=8,
        tmp_dir=task_tmp,
    )
    assert read_ome_tiff_info(export).pixel_size_um == pytest.approx((p_r, p_r))

    # 2. Prepare.
    provenance = entry["provenance"]
    prepare_outputs = run_manual_registration_prepare(
        ManualRegistrationPrepareConfig(
            dataset_name="HD1",
            reference=ManualRegistrationImageInput(path=export),
            moving=ManualRegistrationImageInput(
                path=Path(provenance["image_path"]),
                channel_names=list(entry["channel_names"]),
                registration_channel=provenance.get("registration_channel", "DAPI"),
            ),
            prepared_dir=work / "prepared",
            output_dir=work / "prepare_out",
        )
    )
    store = prepare_outputs["store"]
    attrs = read_root_attrs(store)[STORE_ATTR]
    shapes = {
        role: tuple(attrs["images"][role]["shape_yx"]) for role in attrs["images"]
    }

    # 3. Simulated landmarks from the Explorer matrix.
    working, _ = derive_script.load_working_dapi(store, "moving", pixel_size_um=10.0)
    mask = derive_script.tissue_mask(working.pixels)
    moving_points = sample_tissue_points(
        working.pixels,
        working.to_level0,
        n=16,
        seed=0,
        min_separation_px=1000.0 / p_m,
    )
    rng = np.random.default_rng(0)
    reference_points = _apply(explorer, moving_points) + rng.normal(
        0.0, 1.0 / p_r, size=moving_points.shape
    )
    reference_points[2, 0] += 40.0 / p_r
    pairs = [
        LandmarkPair(
            id=number,
            colour=pair_colour(number, 0),
            enabled=number < 13,
            reference_xy_px=(
                round(float(reference_points[number - 1, 0]), 3),
                round(float(reference_points[number - 1, 1]), 3),
            ),
            moving_xy_px=(
                round(float(moving_points[number - 1, 0]), 3),
                round(float(moving_points[number - 1, 1]), 3),
            ),
        )
        for number in range(1, 17)
    ]
    coarse = coarse_from_matrix(
        explorer,
        moving_shape_yx=shapes["moving"],
        reference_shape_yx=shapes["reference"],
        moving_pixel_size_um=p_m,
        reference_pixel_size_um=p_r,
    )[0]
    inject_labelling(store, pairs, coarse=coarse, crops=None)

    # 4. Register with HD1's recorded settings.
    allowed = set(ManualRegistrationSettings.model_fields) - FORBIDDEN_OVERRIDES
    row_settings = "; ".join(
        f"{key}={_fmt(value)}"
        for key, value in sorted(provenance["settings"].items())
        if key in allowed and value is not None
    )

    def register(label: str) -> tuple[dict[str, Path], dict[str, Any]]:
        outputs = run_manual_registration(
            ManualRegistrationRunConfig(
                dataset_name="HD1",
                preset="same_section",
                run_label=label,
                prepared_store_path=store,
                result_dir=work / "registered" / f"same_section_{label}",
                output_dir=work / f"register_out_{label}",
                row_settings=row_settings,
            )
        )
        return outputs, json.loads(outputs["summary_json"].read_text())

    full_outputs, full = register("full")

    # 5. Crop variant: the inner 80 % of the reference.
    height, width = shapes["reference"]
    crop = CropRect(
        x0=round(0.1 * width),
        y0=round(0.1 * height),
        x1=round(0.9 * width),
        y1=round(0.9 * height),
    )
    inject_labelling(
        store, pairs, coarse=coarse, crops={"reference": crop, "moving": None}
    )
    crop_outputs, cropped = register("crop80")

    # Formulas.
    rows, cols = np.nonzero(mask)
    corners = _apply(
        working.to_level0,
        np.array(
            [
                [cols.min(), rows.min()],
                [cols.max(), rows.min()],
                [cols.max(), rows.max()],
                [cols.min(), rows.max()],
            ],
            dtype=np.float64,
        ),
    )
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    grid = np.array(
        [
            (x, y)
            for y in np.linspace(lo[1], hi[1], 20)
            for x in np.linspace(lo[0], hi[0], 20)
        ]
    )
    working_xy = _apply(np.linalg.inv(working.to_level0), grid)
    columns = np.round(working_xy[:, 0]).astype(int)
    row_index = np.round(working_xy[:, 1]).astype(int)
    inside = (
        (columns >= 0)
        & (columns < mask.shape[1])
        & (row_index >= 0)
        & (row_index < mask.shape[0])
    )
    keep = np.zeros(len(grid), dtype=bool)
    keep[inside] = mask[row_index[inside], columns[inside]]
    grid = grid[keep]

    a_full = np.asarray(full["matrices"]["final_affine_full"], dtype=np.float64)
    m_landmark = np.asarray(full["matrices"]["landmark_full"], dtype=np.float64)
    a_crop = np.asarray(cropped["matrices"]["final_affine_full"], dtype=np.float64)

    flow_name = Path(summary_hd1["outputs"]["nonrigid_flow"]).name
    flow_path = Path(summary_hd1["outputs"]["nonrigid_flow"])
    if not flow_path.is_file():
        found = list((results_root / pid).rglob(flow_name))
        if not found:
            pytest.fail(f"HD1's stored flow {flow_name} was not found (gate 6)")
        flow_path = found[0]
    with np.load(flow_path) as archive:
        flow_yx = np.asarray(archive["flow_yx"])
        working_to_reference = np.asarray(archive["working_to_reference"])
    truth_final, converged = push_points_through_registration(
        moving_points,
        final_matrix=final_truth,
        flow_yx=flow_yx,
        working_to_reference=working_to_reference,
    )
    assert converged.all()

    # Checks.
    fit = full["fit"]
    records = {record["id"]: record for record in full["landmarks"]}
    gate(
        "1_fit_outlier_and_used",
        {
            "n_outliers": fit["n_outliers"],
            "n_used": fit["n_used"],
            "id3_outlier": records[3]["outlier"],
            "id3_used": records[3]["used_in_fit"],
        },
        fit["n_outliers"] == 1
        and records[3]["outlier"]
        and not records[3]["used_in_fit"]
        and fit["n_used"] == 11,
        "n_outliers 1, pair 3 outlier and unused, n_used 11",
    )
    bbox_error = float(
        np.linalg.norm(
            _apply(m_landmark, corners) - _apply(explorer, corners), axis=1
        ).max()
        * p_r
    )
    gate("2_landmark_vs_explorer_corners_um", bbox_error, bbox_error <= 3.0, "<= 3 um")
    gate(
        "3_refinement_status",
        full["refinement"]["status"],
        full["refinement"]["status"] == "applied",
        "applied",
    )
    distance = (
        np.linalg.norm(_apply(a_full, grid) - _apply(final_truth, grid), axis=1) * p_r
    )
    gate(
        "4_affine_vs_hd1_final_um",
        {"mean": float(distance.mean()), "max": float(distance.max())},
        distance.mean() <= 0.5 and distance.max() <= 1.0,
        "mean <= 0.5, max <= 1.0 um",
    )
    nonrigid = full["nonrigid"]
    hd1_nonrigid = summary_hd1["nonrigid"]
    confident = float(nonrigid.get("confident_fraction", float("nan")))
    pearson = float(nonrigid.get("pearson_after", float("nan")))
    gate(
        "5_nonrigid",
        {
            "status": nonrigid["status"],
            "confident_fraction": confident,
            "pearson_after": pearson,
        },
        nonrigid["status"] == "applied"
        and abs(confident - float(hd1_nonrigid["confident_fraction"])) <= 0.05
        and abs(pearson - float(hd1_nonrigid["pearson_after"])) <= 0.01,
        "applied; |confident - HD1| <= 0.05; |pearson_after - HD1| <= 0.01",
    )
    held = [13, 14, 15, 16]
    e_final = np.array(
        [
            np.linalg.norm(
                np.asarray(records[i]["moving_in_reference_xy_px"]["final"])
                - truth_final[i - 1]
            )
            * p_r
            for i in held
        ]
    )
    e_affine = (
        np.linalg.norm(
            _apply(a_full, moving_points[12:])
            - _apply(final_truth, moving_points[12:]),
            axis=1,
        )
        * p_r
    )
    e_landmark = (
        np.linalg.norm(
            _apply(m_landmark, moving_points[12:])
            - _apply(final_truth, moving_points[12:]),
            axis=1,
        )
        * p_r
    )
    measured["6_recorded_only"] = {
        "e_landmark_um": e_landmark.tolist(),
        "residual_um_vs_simulated": {str(i): records[i]["residual_um"] for i in held},
    }
    gate(
        "6_held_out_ids_13_16",
        {
            "median_e_final_um": float(np.median(e_final)),
            "e_affine_mean_um": float(e_affine.mean()),
            "e_affine_max_um": float(e_affine.max()),
        },
        np.median(e_final) <= 1.0 and e_affine.mean() <= 0.5 and e_affine.max() <= 1.0,
        "median e_final <= 1.0 um; e_affine mean <= 0.5, max <= 1.0 um",
    )

    result_store = sd.read_zarr(
        Path(full_outputs["result_store"]), selection=("images",)
    )
    xenium = sd.read_zarr(source_store, selection=("images",))
    registered = _downsampled_dapi(result_store, "moving_registered", "DAPI")
    truth_image = _downsampled_dapi(xenium, "post_xenium_if", "DAPI")
    tissue = truth_image > threshold_otsu(truth_image[truth_image > 0])
    valid = tissue & (registered > 0) & (truth_image > 0)
    pearson_image = float(
        np.corrcoef(np.log1p(registered[valid]), np.log1p(truth_image[valid]))[0, 1]
    )
    gate("7_image_pearson", pearson_image, pearson_image >= 0.98, ">= 0.98")

    statuses = []
    for outputs in (full_outputs, crop_outputs):
        for key in ("landmark_matrix_csv", "final_matrix_csv"):
            matrix = load_alignment_matrix(outputs[key])
            check_matrix_scale(
                matrix,
                image_pixel_size_um=p_m,
                reference_pixel_size_um=p_r,
                tolerance=0.02,
            )
            statuses.append(key)
    gate("8_matrix_csvs", statuses, len(statuses) == 4, "load and scale check pass")

    effective = cropped["crops"]["reference"]["effective"]
    in_crop = (
        (_apply(a_full, grid)[:, 0] >= effective["x0"])
        & (_apply(a_full, grid)[:, 0] < effective["x1"])
        & (_apply(a_full, grid)[:, 1] >= effective["y0"])
        & (_apply(a_full, grid)[:, 1] < effective["y1"])
    )
    crop_difference = (
        np.linalg.norm(
            _apply(a_crop, grid[in_crop]) - _apply(a_full, grid[in_crop]), axis=1
        )
        * p_r
    )
    gate(
        "9_crop80_vs_full_um",
        {"mean": float(crop_difference.mean()), "max": float(crop_difference.max())},
        crop_difference.mean() <= 0.5 and crop_difference.max() <= 1.0,
        "mean <= 0.5, max <= 1.0 um",
    )
    resources = {
        "full": full["resources"],
        "crop80": cropped["resources"],
    }
    gate(
        "10_resources",
        resources,
        all(
            math.isfinite(r["wall_time_s"]) and r["peak_rss_gb"] <= 48.0
            for r in resources.values()
        ),
        "peak RSS <= 48 GB",
    )

    (work / "hd1_validation.json").write_text(
        json.dumps(measured, indent=2, default=float) + "\n", encoding="utf-8"
    )
    assert not failures, failures

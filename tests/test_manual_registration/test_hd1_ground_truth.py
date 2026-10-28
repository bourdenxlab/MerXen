"""Ground truth of the manual registration on HD1 (real data) and the export script.

HD1 has a post-Xenium IF image that the main pipeline registered with the Xenium
Explorer matrix ``E``, a DAPI affine refinement (final matrix ``F``) and an
optical flow. The ``slow`` test simulates landmarks from ``E`` (plus noise and one
outlier), registers them, and measures the manual registration against a control
run made with the current code from ``E`` itself (noise-free landmarks): the
registration stored in HD1's Xenium store was written by older flow code, so its
comparison with the manual registration is recorded, not gated. The unit tests
cover the comparison helpers and ``export_reference_image.py``, which turns the
Xenium ``morphology_focus`` DAPI into the reference OME-TIFF.
"""

from __future__ import annotations

import contextlib
import functools
import json
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import spatialdata as sd
import xarray as xr
from skimage.filters import threshold_otsu
from spatialdata.models import Image2DModel
from spatialdata.transformations import Identity, Scale, Sequence, Translation

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


@pytest.mark.integration
@pytest.mark.parametrize(
    "transformation",
    [
        Translation([10.0, 20.0], axes=("x", "y")),
        Scale([0.5, 0.25], axes=("x", "y")),
        Sequence(
            [
                Scale([0.5, 0.5], axes=("x", "y")),
                Translation([3.0, 4.0], axes=("x", "y")),
            ]
        ),
    ],
    ids=["translation", "anisotropic", "scale_and_translation"],
)
def test_export_reference_image_given_pixel_size_beats_a_non_scale_transformation(
    tmp_path: Path, transformation: Any
) -> None:
    store = tmp_path / "store.zarr"
    data = _write_store(store, transformation)
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

    # The transformation is only logged when the pixel size is given.
    export_script.export_channel(
        store,
        image_key="morphology_focus",
        channel="DAPI",
        out=out,
        pixel_size_um=0.2125,
        workers=1,
    )
    info = read_ome_tiff_info(out)
    assert info.pixel_size_um == pytest.approx((0.2125, 0.2125))
    np.testing.assert_array_equal(open_ome_tiff_level(info, 0).values[0], data[0])


@pytest.mark.integration
def test_export_reference_image_writes_the_stores_channel_name(
    tmp_path: Path,
) -> None:
    store = tmp_path / "store.zarr"
    _write_store(store, Scale([0.5, 0.5], axes=("x", "y")))
    out = tmp_path / "reference.ome.tif"

    # The channel is matched case-insensitively; the file keeps the store's name.
    export_script.export_channel(
        store,
        image_key="morphology_focus",
        channel="dapi",
        out=out,
        pixel_size_um=None,
        workers=1,
    )

    assert read_ome_tiff_info(out).channel_names == ("DAPI",)
    sidecar = json.loads(out.with_suffix(".export.json").read_text(encoding="utf-8"))
    assert sidecar["key"]["channel"] == "DAPI"


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


def _block_mean(array: np.ndarray, factor: int) -> np.ndarray:
    """Mean over ``factor`` x ``factor`` blocks; an edge that fills none is cut."""
    if factor == 1:
        return array
    rows, cols = (size // factor * factor for size in array.shape)
    return (
        array[:rows, :cols]
        .reshape(rows // factor, factor, cols // factor, factor)
        .mean(axis=(1, 3))
    )


def _image_pearson(
    registered: np.ndarray,
    truth: np.ndarray,
    reference_dapi: np.ndarray,
    *,
    block: int = 1,
) -> float:
    """Pearson of ``log1p`` DAPI of two registered images over the Xenium tissue.

    The tissue is the Otsu mask of the downsampled Xenium ``morphology_focus`` DAPI
    (``reference_dapi``, the same grid as the two images), where both images are
    positive; the IF image's own tissue would let IF-only structure decide it. With
    ``block`` > 1 the images are averaged over ``block`` x ``block`` blocks of that
    grid first, and a block is tissue when more than half of its pixels are.
    """
    tissue = reference_dapi > threshold_otsu(reference_dapi[reference_dapi > 0])
    tissue = _block_mean(tissue.astype(np.float64), block) > 0.5
    registered = _block_mean(registered, block)
    truth = _block_mean(truth, block)
    valid = tissue & (registered > 0) & (truth > 0)
    return float(np.corrcoef(np.log1p(registered[valid]), np.log1p(truth[valid]))[0, 1])


def _same_grid(images: Mapping[str, np.ndarray]) -> dict[str, Any] | None:
    """None when the images share one grid, else the shape of each."""
    shapes = {name: tuple(image.shape) for name, image in images.items()}
    return None if len(set(shapes.values())) == 1 else shapes


def _pearsons_against(
    images: Mapping[str, np.ndarray],
    reference_dapi: np.ndarray,
    *,
    blocks: Iterable[int] = (1, 2, 4),
) -> dict[str, dict[str, float]]:
    """Pearson of each image against the Xenium DAPI, per block size.

    The keys are the block size times the x16 grid (``x16``, ``x32``, ``x64``).
    """
    return {
        f"x{16 * block}": {
            name: _image_pearson(image, reference_dapi, reference_dapi, block=block)
            for name, image in images.items()
        }
        for block in blocks
    }


def _landmark_pairs(
    moving_xy: np.ndarray, reference_xy: np.ndarray, *, n_enabled: int
) -> list[LandmarkPair]:
    """Labelling pairs numbered from 1; the first ``n_enabled`` are enabled."""
    return [
        LandmarkPair(
            id=number,
            colour=pair_colour(number, 0),
            enabled=number <= n_enabled,
            reference_xy_px=(
                round(float(reference_xy[number - 1, 0]), 3),
                round(float(reference_xy[number - 1, 1]), 3),
            ),
            moving_xy_px=(
                round(float(moving_xy[number - 1, 0]), 3),
                round(float(moving_xy[number - 1, 1]), 3),
            ),
        )
        for number in range(1, len(moving_xy) + 1)
    ]


STORED_REGISTRATION_NOTE = (
    "HD1's stored registration predates the current non-rigid settings "
    "(no nonrigid_highpass_sigma_um in its recorded settings)"
)


def _stored_registration_note(settings: Mapping[str, Any]) -> str | None:
    """Why the stored registration is not a gate, or None when its code is current."""
    return (
        None if "nonrigid_highpass_sigma_um" in settings else STORED_REGISTRATION_NOTE
    )


@dataclass(frozen=True)
class _FinalMapping:
    """A registration's final mapping, as the comparison helpers read it.

    ``affine`` maps moving to reference level-0 pixels; ``final_xy`` holds the
    points the mapping was built for, pushed through the affine and the flow (what
    the registered image shows); ``converged`` flags the points whose flow
    inversion converged; ``nonrigid`` is the non-rigid summary of the run.
    """

    affine: np.ndarray
    final_xy: np.ndarray
    converged: np.ndarray
    nonrigid: Mapping[str, Any]


def _final_mapping(
    affine: np.ndarray,
    points_moving_xy: np.ndarray,
    *,
    nonrigid: Mapping[str, Any],
    flow_path: Path | None,
) -> _FinalMapping:
    flow_yx = working_to_reference = None
    if flow_path is not None:
        with np.load(flow_path) as archive:
            flow_yx = np.asarray(archive["flow_yx"])
            working_to_reference = np.asarray(archive["working_to_reference"])
    final_xy, converged = push_points_through_registration(
        points_moving_xy,
        final_matrix=affine,
        flow_yx=flow_yx,
        working_to_reference=working_to_reference,
    )
    return _FinalMapping(
        np.asarray(affine, dtype=np.float64), final_xy, converged, nonrigid
    )


def _result_mapping(
    summary: Mapping[str, Any], points_moving_xy: np.ndarray
) -> _FinalMapping:
    """The final mapping of a register run without crops, read from its summary.

    The flow is used only when the non-rigid stage was applied, as the run's own
    ``final`` positions do. Without crops the full-frame affine and the flow share
    one frame (a crop run would need its origins).
    """
    nonrigid = summary["nonrigid"]
    flow_path = None
    if nonrigid["status"] == "applied":
        flow_path = Path(summary["outputs"]["core_qc"]["nonrigid_flow_npz"])
    return _final_mapping(
        np.asarray(summary["matrices"]["final_affine_full"], dtype=np.float64),
        points_moving_xy,
        nonrigid=nonrigid,
        flow_path=flow_path,
    )


def _find_stored_flow(
    stored_summary: Mapping[str, Any], search_root: Path
) -> Path | None:
    """The flow archive a stored registration summary names, or None if it is gone."""
    recorded = Path(stored_summary["outputs"]["nonrigid_flow"])
    if recorded.is_file():
        return recorded
    found = sorted(search_root.rglob(recorded.name))
    return found[0] if found else None


def _matrix_gap_um(
    matrix: np.ndarray,
    other: np.ndarray,
    points_xy: np.ndarray,
    pixel_size_um: float,
) -> np.ndarray:
    """Distance between two moving-to-reference affines at the points, in um."""
    return (
        np.linalg.norm(_apply(matrix, points_xy) - _apply(other, points_xy), axis=1)
        * pixel_size_um
    )


def _distance_stats(distance: np.ndarray) -> dict[str, float]:
    return {"mean": float(distance.mean()), "max": float(distance.max())}


def _affine_gate_passed(stats: Mapping[str, float]) -> bool:
    return bool(stats["mean"] <= 0.5 and stats["max"] <= 1.0)


def _nonrigid_agreement(
    candidate: Mapping[str, Any], other: Mapping[str, Any], *, other_label: str
) -> dict[str, Any]:
    """Status, confident fraction and ``pearson_after`` of two non-rigid summaries.

    A metric a summary lacks, or holds as null (the summary writer's way of saving
    a non-finite statistic), is NaN, and no gate passes on a NaN.
    """

    def metric(summary: Mapping[str, Any], key: str) -> float:
        value = summary.get(key)
        return float("nan") if value is None else float(value)

    confident, pearson = (
        metric(candidate, "confident_fraction"),
        metric(candidate, "pearson_after"),
    )
    other_confident, other_pearson = (
        metric(other, "confident_fraction"),
        metric(other, "pearson_after"),
    )
    return {
        "status": candidate.get("status"),
        "confident_fraction": confident,
        "pearson_after": pearson,
        f"{other_label}_confident_fraction": other_confident,
        f"{other_label}_pearson_after": other_pearson,
        "confident_fraction_difference": abs(confident - other_confident),
        "pearson_after_difference": abs(pearson - other_pearson),
    }


def _nonrigid_gate_passed(agreement: Mapping[str, Any]) -> bool:
    return bool(
        agreement["status"] == "applied"
        and agreement["confident_fraction_difference"] <= 0.05
        and agreement["pearson_after_difference"] <= 0.01
    )


def _held_out_errors(
    candidate: _FinalMapping,
    other: _FinalMapping,
    moving_points: np.ndarray,
    held: Iterable[int],
    pixel_size_um: float,
    *,
    labels: tuple[str, str],
) -> dict[str, Any]:
    """Final and affine disagreement of two mappings at the held-out pairs (1-based).

    Only the held-out pairs count, convergence of the flow inversion included; the
    ids of the held-out pairs it failed at are listed under each mapping's label.
    """
    held = list(held)
    index = [number - 1 for number in held]
    e_final = (
        np.linalg.norm(candidate.final_xy[index] - other.final_xy[index], axis=1)
        * pixel_size_um
    )
    e_affine = _matrix_gap_um(
        candidate.affine, other.affine, moving_points[index], pixel_size_um
    )
    not_converged = {
        label: [number for number in held if not mapping.converged[number - 1]]
        for label, mapping in zip(labels, (candidate, other), strict=True)
    }
    return {
        "median_e_final_um": float(np.median(e_final)),
        "e_affine_mean_um": float(e_affine.mean()),
        "e_affine_max_um": float(e_affine.max()),
        "flow_converged": not any(not_converged.values()),
        "not_converged_ids": not_converged,
    }


def _held_out_gate_passed(errors: Mapping[str, Any]) -> bool:
    return bool(
        errors["flow_converged"]
        and errors["median_e_final_um"] <= 1.0
        and errors["e_affine_mean_um"] <= 0.5
        and errors["e_affine_max_um"] <= 1.0
    )


def _record(target: dict[str, Any], key: str, compute: Callable[[], Any]) -> None:
    """Store ``compute()`` under ``key``; a recorded value never fails the test.

    An exception is recorded as ``{"error": repr(exc)}`` instead, so the comparison
    with HD1's stored registration (which may be missing or in an older format)
    cannot cost the run's other measurements.
    """
    try:
        target[key] = compute()
    except Exception as exc:
        target[key] = {"error": repr(exc)}


def _gate_outcome(compute: Callable[[], tuple[Any, bool]]) -> tuple[Any, bool]:
    """The ``(value, passed)`` of a gate; one whose computation raises fails.

    The value is then ``{"error": repr(exc)}``, so the failure is recorded and the
    gates after it still run and the validation JSON is still written.
    """
    try:
        return compute()
    except Exception as exc:
        return {"error": repr(exc)}, False


def test_image_pearson_takes_its_tissue_from_the_xenium_dapi() -> None:
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:120, :160]
    xenium_tissue = (yy - 60) ** 2 + (xx - 40) ** 2 < 35**2
    if_only_blob = (yy - 60) ** 2 + (xx - 120) ** 2 < 25**2
    reference_dapi = np.where(xenium_tissue, 800.0, 5.0) + rng.uniform(0, 4, yy.shape)
    signal = rng.uniform(100.0, 1000.0, yy.shape)
    truth = np.where(xenium_tissue, signal, 5.0)
    registered = np.where(xenium_tissue, signal * 1.1, 5.0)
    # Structure only the IF image has: bright in both, but uncorrelated between them.
    truth = np.where(if_only_blob, rng.uniform(100.0, 1000.0, yy.shape), truth)
    registered = np.where(
        if_only_blob, rng.uniform(100.0, 1000.0, yy.shape), registered
    )

    value = _image_pearson(registered, truth, reference_dapi)

    assert value > 0.999
    # The IF image's own tissue mask (the old choice) takes the blob in and fails.
    own = truth > threshold_otsu(truth[truth > 0])
    assert own[if_only_blob].any()
    valid = own & (registered > 0) & (truth > 0)
    old = np.corrcoef(np.log1p(registered[valid]), np.log1p(truth[valid]))[0, 1]
    assert old < 0.95


def test_image_pearson_blocks_average_the_images_before_correlating() -> None:
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:80, :120]
    tissue = (yy - 40) ** 2 + (xx - 60) ** 2 < 30**2
    xenium = np.where(tissue, 800.0, 5.0) + rng.uniform(0, 4, yy.shape)
    # A signal constant over 4 x 4 pixels, seen through independent pixel noise.
    signal = np.kron(rng.uniform(100.0, 1000.0, (20, 30)), np.ones((4, 4)))
    noisy = np.clip(signal + rng.normal(0.0, 400.0, signal.shape), 1.0, None)

    values = [_image_pearson(noisy, signal, xenium, block=b) for b in (1, 2, 4)]

    assert values[0] == _image_pearson(noisy, signal, xenium)
    assert values[0] < values[1] < values[2]
    assert values[0] < 0.6 and values[2] > 0.85


def test_image_pearson_block_is_tissue_only_when_most_of_it_is() -> None:
    rng = np.random.default_rng(1)
    reference_dapi = np.full((40, 60), 5.0)
    reference_dapi[:, :41] = 800.0
    reference_dapi += rng.uniform(0, 4, reference_dapi.shape)
    truth = rng.uniform(100.0, 1000.0, reference_dapi.shape)
    registered = truth * 1.1
    # Column 40 is the last tissue column: a pixel of tissue at x1, half a block at x2.
    registered[:, 40] = rng.uniform(100.0, 1000.0, 40)

    assert _image_pearson(registered, truth, reference_dapi) < 0.99
    for block in (2, 4):
        assert _image_pearson(registered, truth, reference_dapi, block=block) > 0.999


def test_pearsons_against_the_xenium_dapi_per_block_size() -> None:
    rng = np.random.default_rng(2)
    yy, xx = np.mgrid[:96, :128]
    tissue = (yy - 48) ** 2 + (xx - 64) ** 2 < 35**2
    signal = np.kron(rng.uniform(100.0, 1000.0, (24, 32)), np.ones((4, 4)))
    xenium = np.where(tissue, signal, 5.0)
    images = {
        "same": xenium,
        "noisy": np.clip(xenium + rng.normal(0.0, 300.0, xenium.shape), 1.0, None),
    }

    values = _pearsons_against(images, xenium)

    assert list(values) == ["x16", "x32", "x64"]
    assert all(block["same"] == pytest.approx(1.0) for block in values.values())
    assert set(values["x16"]) == {"same", "noisy"}
    # The noise averages out over larger blocks of the signal that the images share.
    assert values["x16"]["noisy"] < values["x32"]["noisy"] < values["x64"]["noisy"]
    assert _pearsons_against(images, xenium, blocks=(2,)) == {"x32": values["x32"]}


def test_same_grid_reports_the_shapes_when_they_differ() -> None:
    assert _same_grid({"a": np.zeros((4, 5)), "b": np.ones((4, 5))}) is None
    assert _same_grid({"a": np.zeros((4, 5)), "b": np.ones((4, 6))}) == {
        "a": (4, 5),
        "b": (4, 6),
    }


def test_landmark_pairs_number_round_and_enable() -> None:
    moving = np.array([[10.0004, 20.0006], [30.0, 40.0], [50.0, 60.0]])
    reference = moving + 1.23456

    pairs = _landmark_pairs(moving, reference, n_enabled=2)

    assert [pair.id for pair in pairs] == [1, 2, 3]
    assert [pair.enabled for pair in pairs] == [True, True, False]
    assert pairs[0].moving_xy_px == (10.0, 20.001)
    assert pairs[0].reference_xy_px == (11.235, 21.235)
    assert [pair.colour for pair in pairs] == [pair_colour(n, 0) for n in (1, 2, 3)]


def test_the_control_labelling_is_the_explorer_matrix_applied_exactly() -> None:
    explorer = np.array([[0.9, 0.1, 120.0], [-0.1, 0.9, -40.0], [0.0, 0.0, 1.0]])
    moving = np.random.default_rng(3).uniform(0.0, 5000.0, size=(16, 2))

    pairs = _landmark_pairs(moving, _apply(explorer, moving), n_enabled=len(moving))

    assert all(pair.enabled for pair in pairs)
    placed = np.array([pair.reference_xy_px for pair in pairs])
    # Only the labelling's rounding to 0.001 px remains.
    np.testing.assert_allclose(placed, _apply(explorer, moving), atol=5e-4)
    np.testing.assert_allclose(
        np.array([pair.moving_xy_px for pair in pairs]), moving, atol=5e-4
    )


def test_stored_registration_note_names_the_missing_current_setting() -> None:
    old = {"nonrigid_grid_um": 40.0, "refine_affine": True}
    current = {**old, "nonrigid_highpass_sigma_um": 30.0}

    assert _stored_registration_note(old) == (
        "HD1's stored registration predates the current non-rigid settings "
        "(no nonrigid_highpass_sigma_um in its recorded settings)"
    )
    assert _stored_registration_note(current) is None
    # The key being present is what counts, whatever its value.
    assert _stored_registration_note({"nonrigid_highpass_sigma_um": None}) is None


def _write_flow(path: Path, *, dy: float, dx: float, shape: tuple[int, int]) -> None:
    np.savez(
        path,
        flow_yx=np.stack([np.full(shape, dy), np.full(shape, dx)]).astype(np.float32),
        working_to_reference=np.eye(3),
    )


def _result_summary(
    affine: np.ndarray, *, status: str, flow: Path | None
) -> dict[str, Any]:
    return {
        "matrices": {"final_affine_full": affine.tolist()},
        "nonrigid": {"status": status, "confident_fraction": 0.8},
        "outputs": {
            "core_qc": {} if flow is None else {"nonrigid_flow_npz": str(flow)}
        },
    }


def test_result_mapping_pushes_the_points_through_an_applied_flow(
    tmp_path: Path,
) -> None:
    affine = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, -3.0], [0.0, 0.0, 1.0]])
    points = np.array([[20.0, 10.0], [30.0, 20.0], [40.0, 25.0]])
    flow = tmp_path / "flow.npz"
    _write_flow(flow, dy=0.5, dx=2.0, shape=(60, 80))

    applied = _result_mapping(
        _result_summary(affine, status="applied", flow=flow), points
    )

    # The flow pulls: a point lands where the affine puts it, less the displacement.
    np.testing.assert_allclose(
        applied.final_xy, _apply(affine, points) - [2.0, 0.5], atol=1e-3
    )
    np.testing.assert_array_equal(applied.affine, affine)
    assert applied.converged.all()
    assert applied.nonrigid["status"] == "applied"

    # A flow that was computed but rejected is not part of the registration.
    rejected = _result_mapping(
        _result_summary(affine, status="rejected", flow=flow), points
    )
    np.testing.assert_allclose(rejected.final_xy, _apply(affine, points))
    # ... and so is a run that has none.
    skipped = _result_mapping(
        _result_summary(affine, status="skipped", flow=None), points
    )
    np.testing.assert_allclose(skipped.final_xy, _apply(affine, points))


def test_find_stored_flow_looks_by_name_when_the_recorded_path_is_gone(
    tmp_path: Path,
) -> None:
    root = tmp_path / "results"
    (root / "run" / "qc").mkdir(parents=True)
    recorded = tmp_path / "moved" / "flow.npz"
    summary = {"outputs": {"nonrigid_flow": str(recorded)}}

    assert _find_stored_flow(summary, root) is None

    found = root / "run" / "qc" / "flow.npz"
    found.write_bytes(b"")
    assert _find_stored_flow(summary, root) == found

    recorded.parent.mkdir()
    recorded.write_bytes(b"")
    assert _find_stored_flow(summary, root) == recorded


def test_affine_comparison_helpers_and_their_gate() -> None:
    identity = np.eye(3)
    shifted = np.array([[1.0, 0.0, 0.3], [0.0, 1.0, 0.4], [0.0, 0.0, 1.0]])
    points = np.array([[0.0, 0.0], [100.0, 50.0], [-20.0, 7.0]])

    # A 0.5 px shift on a 2 um pixel is 1 um everywhere.
    gap = _matrix_gap_um(identity, shifted, points, 2.0)
    np.testing.assert_allclose(gap, [1.0, 1.0, 1.0])
    np.testing.assert_allclose(_matrix_gap_um(shifted, identity, points, 2.0), gap)
    assert _distance_stats(gap) == {
        "mean": pytest.approx(1.0),
        "max": pytest.approx(1.0),
    }

    assert _affine_gate_passed({"mean": 0.5, "max": 1.0})
    assert not _affine_gate_passed({"mean": 0.51, "max": 1.0})
    assert not _affine_gate_passed({"mean": 0.1, "max": 1.01})
    assert not _affine_gate_passed({"mean": float("nan"), "max": float("nan")})


def test_nonrigid_agreement_compares_two_summaries() -> None:
    candidate = {"status": "applied", "confident_fraction": 0.80, "pearson_after": 0.90}
    other = {"confident_fraction": 0.84, "pearson_after": 0.905, "status": "applied"}

    agreement = _nonrigid_agreement(candidate, other, other_label="control")

    assert agreement["status"] == "applied"
    assert agreement["confident_fraction"] == 0.80
    assert agreement["control_confident_fraction"] == 0.84
    assert agreement["control_pearson_after"] == 0.905
    assert agreement["confident_fraction_difference"] == pytest.approx(0.04)
    assert agreement["pearson_after_difference"] == pytest.approx(0.005)
    assert _nonrigid_gate_passed(agreement)


_APPLIED = {"status": "applied", "confident_fraction": 0.8, "pearson_after": 0.9}


@pytest.mark.parametrize(
    ("candidate", "other"),
    [
        ({**_APPLIED, "status": "rejected"}, _APPLIED),
        (_APPLIED, {**_APPLIED, "confident_fraction": 0.86}),
        (_APPLIED, {**_APPLIED, "pearson_after": 0.92}),
        # A summary without the metrics (no flow was computed) never agrees.
        (_APPLIED, {"status": "skipped"}),
        ({"status": "skipped"}, _APPLIED),
        # A metric the summary writer saved as null (it was not finite) is NaN.
        ({**_APPLIED, "pearson_after": None}, _APPLIED),
        (_APPLIED, {**_APPLIED, "confident_fraction": None}),
    ],
    ids=[
        "status",
        "confident",
        "pearson",
        "other_has_no_metrics",
        "no_metrics",
        "null_pearson",
        "other_null_confident",
    ],
)
def test_nonrigid_gate_fails_outside_its_tolerances(
    candidate: dict[str, Any], other: dict[str, Any]
) -> None:
    agreement = _nonrigid_agreement(candidate, other, other_label="stored")

    assert not _nonrigid_gate_passed(agreement)


def test_nonrigid_agreement_reads_a_null_metric_as_nan() -> None:
    nulled = {"status": "applied", "confident_fraction": None, "pearson_after": 0.9}

    agreement = _nonrigid_agreement(nulled, _APPLIED, other_label="control")

    assert math.isnan(agreement["confident_fraction"])
    assert math.isnan(agreement["confident_fraction_difference"])
    assert agreement["pearson_after"] == 0.9
    assert agreement["pearson_after_difference"] == pytest.approx(0.0)
    assert agreement["control_confident_fraction"] == 0.8


_HELD = [13, 14, 15, 16]
_PIXEL_UM = 2.0


def _mapping(
    final_xy: np.ndarray,
    *,
    affine: np.ndarray | None = None,
    converged: bool | np.ndarray = True,
) -> _FinalMapping:
    return _FinalMapping(
        np.eye(3) if affine is None else affine,
        final_xy,
        np.array(np.broadcast_to(converged, (len(final_xy),))),
        {"status": "applied"},
    )


def _held_out_case(
    *,
    held_shift_px: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4),
    held_x_px: tuple[float, ...] = (100.0, 200.0, 300.0, 400.0),
    scale_x: float = 1.0005,
    candidate_not_converged: Iterable[int] = (),
    other_not_converged: Iterable[int] = (),
) -> dict[str, Any]:
    """The held-out errors of a candidate and an other mapping of 16 pairs.

    The other mapping is the identity. The candidate's final position is off by
    ``held_shift_px`` in x at pairs 13-16 and its affine scales x by ``scale_x``, so
    its affine gap grows with x (``held_x_px`` for pairs 13-16, around 900 px for
    pairs 1-12). Pairs 1-11 are 3 px off in the final mapping, a gap that no held-out
    pair has; pair 12, next to the held-out ones, agrees exactly.
    """
    points = np.zeros((16, 2))
    points[:12, 0] = 900.0 + 10.0 * np.arange(12)
    points[12:, 0] = held_x_px
    points[:, 1] = 40.0 * np.arange(16)
    shift = np.zeros(16)
    shift[:11] = 3.0
    shift[12:] = held_shift_px
    final = points + np.stack([shift, np.zeros(16)], axis=1)

    def converged(ids: Iterable[int]) -> np.ndarray:
        flags = np.ones(16, dtype=bool)
        flags[[number - 1 for number in ids]] = False
        return flags

    candidate = _mapping(
        final,
        affine=np.diag([scale_x, 1.0, 1.0]),
        converged=converged(candidate_not_converged),
    )
    other = _mapping(points, converged=converged(other_not_converged))
    return _held_out_errors(
        candidate, other, points, _HELD, _PIXEL_UM, labels=("full", "control")
    )


def test_held_out_errors_use_only_the_held_out_pairs() -> None:
    errors = _held_out_case()

    # Final: 0.1 to 0.4 px on a 2 um pixel. Affine: the x scale 1.0005 moves the
    # points at x = 100 to 400 px by 0.05 to 0.2 px, twice that in um.
    assert errors == {
        "median_e_final_um": pytest.approx(0.5),
        "e_affine_mean_um": pytest.approx(0.25),
        "e_affine_max_um": pytest.approx(0.4),
        "flow_converged": True,
        "not_converged_ids": {"full": [], "control": []},
    }
    assert _held_out_gate_passed(errors)


def test_held_out_errors_list_the_held_out_pairs_the_flow_inversion_missed() -> None:
    errors = _held_out_case(candidate_not_converged=[14], other_not_converged=[15, 16])

    assert errors["not_converged_ids"] == {"full": [14], "control": [15, 16]}
    assert errors["flow_converged"] is False
    assert not _held_out_gate_passed(errors)

    # A pair outside the held-out set is no concern of this gate.
    outside = _held_out_case(candidate_not_converged=[1], other_not_converged=[12])
    assert outside["not_converged_ids"] == {"full": [], "control": []}
    assert outside["flow_converged"] is True
    assert _held_out_gate_passed(outside)


@pytest.mark.parametrize(
    "case",
    [
        # Median final error 1.2 um.
        {"held_shift_px": (0.6, 0.6, 0.6, 0.6)},
        # Affine gap 0.56 um at every held-out point: mean out, max in.
        {"held_x_px": (400.0,) * 4, "scale_x": 1.0007},
        # Affine gaps 0.014 um at three points and 1.4 um at one: mean 0.36 um
        # is in range, max out.
        {"held_x_px": (10.0, 10.0, 10.0, 1000.0), "scale_x": 1.0007},
        # The flow inversion missed a held-out pair of the candidate, or of the other.
        {"candidate_not_converged": [14]},
        {"other_not_converged": [16]},
    ],
    ids=[
        "final_median",
        "affine_mean",
        "affine_max_only",
        "candidate_not_converged",
        "other_not_converged",
    ],
)
def test_held_out_gate_fails_outside_its_tolerances(case: dict[str, Any]) -> None:
    errors = _held_out_case(**case)

    assert not _held_out_gate_passed(errors)


def test_held_out_gate_passes_exactly_at_its_tolerances() -> None:
    at_limit = {
        "flow_converged": True,
        "median_e_final_um": 1.0,
        "e_affine_mean_um": 0.5,
        "e_affine_max_um": 1.0,
    }

    assert _held_out_gate_passed(at_limit)
    for key in ("median_e_final_um", "e_affine_mean_um", "e_affine_max_um"):
        assert not _held_out_gate_passed({**at_limit, key: at_limit[key] + 0.01})
    assert not _held_out_gate_passed({**at_limit, "flow_converged": False})


def test_a_recorded_value_that_raises_is_recorded_as_an_error() -> None:
    target: dict[str, Any] = {}

    _record(target, "fine", lambda: {"mean": 0.1})
    _record(target, "broken", lambda: 1 / 0)
    _record(target, "missing", lambda: dict[str, int]()["no such key"])

    assert target == {
        "fine": {"mean": 0.1},
        "broken": {"error": repr(ZeroDivisionError("division by zero"))},
        "missing": {"error": repr(KeyError("no such key"))},
    }


def test_a_gate_whose_computation_raises_fails_with_the_error() -> None:
    def broken() -> tuple[Any, bool]:
        raise ZeroDivisionError("division by zero")

    assert _gate_outcome(lambda: (0.5, True)) == (0.5, True)
    assert _gate_outcome(lambda: (0.5, False)) == (0.5, False)
    assert _gate_outcome(broken) == (
        {"error": repr(ZeroDivisionError("division by zero"))},
        False,
    )


@pytest.mark.slow
def test_hd1_ground_truth(  # noqa: C901
    request: pytest.FixtureRequest,
    real_path: Callable[[str], Path],
    real_aliases: Path,
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
    measured: dict[str, Any] = {"runs": {}}
    failures: list[str] = []

    def write_validation() -> None:
        (work / "hd1_validation.json").write_text(
            json.dumps(measured, indent=2, default=float) + "\n", encoding="utf-8"
        )

    # The measurements are saved whatever happens from here on.
    request.addfinalizer(write_validation)

    def gate(name: str, text: str, compute: Callable[[], tuple[Any, bool]]) -> None:
        """Record a gate; one whose computation raises is recorded as failed."""
        value, passed = _gate_outcome(compute)
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
    pairs = _landmark_pairs(moving_points, reference_points, n_enabled=12)
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
        # A run whose inputs did not change is not registered again: its task
        # summary says so.
        _record(
            measured["runs"],
            label,
            lambda: {
                "reused": bool(
                    json.loads(outputs["task_summary_json"].read_text()).get(
                        "reused", False
                    )
                )
            },
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

    # 6. Control: the current code from the Explorer matrix itself. HD1's stored
    # registration was written by older flow code, so the gates compare with this
    # run (the pipeline's own registration from the Explorer matrix, through the
    # shared core): the same tissue points mapped through the matrix exactly, no
    # noise, no outlier, every pair enabled, the same coarse and no crops.
    control_pairs = _landmark_pairs(
        moving_points,
        _apply(explorer, moving_points),
        n_enabled=len(moving_points),
    )
    inject_labelling(store, control_pairs, coarse=coarse, crops=None)
    control_outputs, control = register("explorer")
    _record(
        measured,
        "stored_registration_note",
        lambda: _stored_registration_note(provenance["settings"]),
    )
    _record(
        measured,
        "control_run",
        lambda: {
            "label": "explorer",
            "n_used": control["fit"]["n_used"],
            "n_outliers": control["fit"]["n_outliers"],
            "refinement_status": control["refinement"]["status"],
            "nonrigid_status": control["nonrigid"]["status"],
        },
    )

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

    # The final mapping of each registration: its affine and flow pushed through.
    @functools.cache
    def full_map() -> _FinalMapping:
        return _result_mapping(full, moving_points)

    @functools.cache
    def control_map() -> _FinalMapping:
        return _result_mapping(control, moving_points)

    @functools.cache
    def stored_map() -> _FinalMapping | None:
        stored_flow = _find_stored_flow(summary_hd1, results_root / pid)
        if stored_flow is None:
            return None
        return _final_mapping(
            final_truth,
            moving_points,
            nonrigid=summary_hd1["nonrigid"],
            flow_path=stored_flow,
        )

    # Checks. A gate whose computation raises is recorded as failed, and a recorded
    # value whose computation raises as an error, so the JSON always gets written.
    def fit_outlier_and_used() -> tuple[Any, bool]:
        fit = full["fit"]
        record = {item["id"]: item for item in full["landmarks"]}[3]
        value = {
            "n_outliers": fit["n_outliers"],
            "n_used": fit["n_used"],
            "id3_outlier": record["outlier"],
            "id3_used": record["used_in_fit"],
        }
        return value, bool(
            fit["n_outliers"] == 1
            and record["outlier"]
            and not record["used_in_fit"]
            and fit["n_used"] == 11
        )

    gate(
        "1_fit_outlier_and_used",
        "n_outliers 1, pair 3 outlier and unused, n_used 11",
        fit_outlier_and_used,
    )

    def landmark_vs_explorer_corners() -> tuple[Any, bool]:
        m_landmark = np.asarray(full["matrices"]["landmark_full"], dtype=np.float64)
        bbox_error = float(
            np.linalg.norm(
                _apply(m_landmark, corners) - _apply(explorer, corners), axis=1
            ).max()
            * p_r
        )
        return bbox_error, bbox_error <= 3.0

    gate("2_landmark_vs_explorer_corners_um", "<= 3 um", landmark_vs_explorer_corners)

    def refinement_status() -> tuple[Any, bool]:
        status = full["refinement"]["status"]
        return status, status == "applied"

    gate("3_refinement_status", "applied", refinement_status)

    # Gates 4 to 7 compare the full run with the control run; the same comparisons
    # with HD1's stored registration are recorded only.
    def affine_vs_control() -> tuple[Any, bool]:
        stats = _distance_stats(
            _matrix_gap_um(full_map().affine, control_map().affine, grid, p_r)
        )
        return stats, _affine_gate_passed(stats)

    gate(
        "4_affine_vs_control_final_um",
        "mean <= 0.5, max <= 1.0 um (full vs control)",
        affine_vs_control,
    )
    _record(
        measured,
        "4_vs_stored",
        lambda: _distance_stats(
            _matrix_gap_um(full_map().affine, final_truth, grid, p_r)
        ),
    )

    def nonrigid_vs_control() -> tuple[Any, bool]:
        agreement = _nonrigid_agreement(
            full["nonrigid"], control["nonrigid"], other_label="control"
        )
        return agreement, _nonrigid_gate_passed(agreement)

    gate(
        "5_nonrigid",
        "applied; |confident - control| <= 0.05; |pearson_after - control| <= 0.01",
        nonrigid_vs_control,
    )
    _record(
        measured,
        "5_vs_stored",
        lambda: _nonrigid_agreement(
            full["nonrigid"], summary_hd1["nonrigid"], other_label="stored"
        ),
    )

    held = [13, 14, 15, 16]

    def landmarks_vs_the_simulated_truth() -> dict[str, Any]:
        m_landmark = np.asarray(full["matrices"]["landmark_full"], dtype=np.float64)
        records = {item["id"]: item for item in full["landmarks"]}
        return {
            "e_landmark_um": _matrix_gap_um(
                m_landmark, final_truth, moving_points[12:], p_r
            ).tolist(),
            "residual_um_vs_simulated": {
                str(i): records[i]["residual_um"] for i in held
            },
        }

    _record(measured, "6_recorded_only", landmarks_vs_the_simulated_truth)

    def held_out_vs_control() -> tuple[Any, bool]:
        errors = _held_out_errors(
            full_map(),
            control_map(),
            moving_points,
            held,
            p_r,
            labels=("full", "control"),
        )
        return errors, _held_out_gate_passed(errors)

    gate(
        "6_held_out_ids_13_16",
        "flow inversion converged at pairs 13-16 in both runs; median e_final <= "
        "1.0 um; e_affine mean <= 0.5, max <= 1.0 um (full vs control)",
        held_out_vs_control,
    )

    def held_out_vs_stored() -> dict[str, Any]:
        stored = stored_map()
        if stored is None:
            return {"unavailable": "the stored flow archive was not found"}
        return _held_out_errors(
            full_map(), stored, moving_points, held, p_r, labels=("full", "stored")
        )

    _record(measured, "6_vs_stored", held_out_vs_stored)

    control_vs_stored: dict[str, Any] = {}
    measured["control_vs_stored"] = control_vs_stored
    _record(
        control_vs_stored,
        "final_affine_over_grid_um",
        lambda: _distance_stats(
            _matrix_gap_um(control_map().affine, final_truth, grid, p_r)
        ),
    )

    @functools.cache
    def xenium_images() -> Any:
        return sd.read_zarr(source_store, selection=("images",))

    @functools.cache
    def registered_images() -> dict[str, np.ndarray]:
        stores = {"full": full_outputs, "control": control_outputs}
        return {
            name: _downsampled_dapi(
                sd.read_zarr(Path(outputs["result_store"]), selection=("images",)),
                "moving_registered",
                "DAPI",
            )
            for name, outputs in stores.items()
        }

    @functools.cache
    def reference_dapi() -> np.ndarray:
        return _downsampled_dapi(xenium_images(), "morphology_focus", "DAPI")

    @functools.cache
    def stored_image() -> np.ndarray:
        return _downsampled_dapi(xenium_images(), "post_xenium_if", "DAPI")

    def image_pearson() -> tuple[Any, bool]:
        images = registered_images()
        grids = _same_grid({**images, "xenium": reference_dapi()})
        if grids is not None:
            return grids, False
        value = _image_pearson(images["full"], images["control"], reference_dapi())
        return value, value >= 0.98

    gate(
        "7_image_pearson",
        ">= 0.98 (full vs control; the images share the Xenium DAPI's grid)",
        image_pearson,
    )

    def pearson_with_stored(name: str) -> Any:
        grids = _same_grid({"stored": stored_image(), "xenium": reference_dapi()})
        if grids is not None:
            return {
                "unavailable": "the stored image is on another grid",
                "grids": grids,
            }
        return _image_pearson(
            registered_images()[name], stored_image(), reference_dapi()
        )

    _record(measured, "7_vs_stored", lambda: pearson_with_stored("full"))
    _record(
        control_vs_stored, "image_pearson_x16", lambda: pearson_with_stored("control")
    )

    def pearsons_vs_xenium_dapi() -> Any:
        compared = dict(registered_images())
        with contextlib.suppress(Exception):  # 7_vs_stored says why it is missing
            stored = stored_image()
            if _same_grid({"stored": stored, "xenium": reference_dapi()}) is None:
                compared["stored"] = stored
        return _pearsons_against(compared, reference_dapi())

    _record(measured, "7_vs_xenium_dapi", pearsons_vs_xenium_dapi)

    def matrix_csvs() -> tuple[Any, bool]:
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
        return statuses, len(statuses) == 4

    gate("8_matrix_csvs", "load and scale check pass", matrix_csvs)

    def crop80_vs_full() -> tuple[Any, bool]:
        a_full = np.asarray(full["matrices"]["final_affine_full"], dtype=np.float64)
        a_crop = np.asarray(cropped["matrices"]["final_affine_full"], dtype=np.float64)
        effective = cropped["crops"]["reference"]["effective"]
        mapped = _apply(a_full, grid)
        in_crop = (
            (mapped[:, 0] >= effective["x0"])
            & (mapped[:, 0] < effective["x1"])
            & (mapped[:, 1] >= effective["y0"])
            & (mapped[:, 1] < effective["y1"])
        )
        stats = _distance_stats(_matrix_gap_um(a_crop, a_full, grid[in_crop], p_r))
        return stats, _affine_gate_passed(stats)

    gate("9_crop80_vs_full_um", "mean <= 0.5, max <= 1.0 um", crop80_vs_full)

    def resources_within_limits() -> tuple[Any, bool]:
        resources = {
            "full": full["resources"],
            "crop80": cropped["resources"],
            "explorer": control["resources"],
        }
        return resources, all(
            math.isfinite(resources[run]["wall_time_s"])
            and resources[run]["peak_rss_gb"] <= 48.0
            for run in ("full", "crop80")
        )

    gate(
        "10_resources",
        "peak RSS <= 48 GB (full and crop80; the control run is recorded)",
        resources_within_limits,
    )

    assert not failures, failures

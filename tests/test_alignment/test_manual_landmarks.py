"""Real Zarr export, frame conversion and guards of paired manual registration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import spatialdata as sd
from spatialdata.models import Image2DModel

from merxen.alignment.bundle import ValisTransformBundle
from merxen.alignment.landmark_prepare import (
    PAIRED_LANDMARK_ATTR,
    prepare_paired_landmark_store,
)
from merxen.alignment.manual import _inverse
from merxen.alignment.mapping import load_alignment_mapper
from merxen.alignment.pipeline import run_alignment_pipeline
from merxen.config import AlignmentConfig
from merxen.io.image_source import image_to_cyx
from merxen.io.spatialdata_io import spatialdata_write_lock
from merxen.manual_registration.landmark_field import (
    FieldHyperparameters,
    FieldObservations,
    FieldOptions,
    LandmarkField,
)
from merxen.manual_registration.schema import (
    LABELLING_ATTR,
    STORE_ATTR,
    read_root_attrs,
    write_root_attr,
)


@pytest.fixture
def paired_package(tmp_path: Path) -> tuple[AlignmentConfig, Path]:
    """Native frames differ in scale, shear and origin; package offsets matter."""
    rng = np.random.default_rng(12)
    for platform in ("merscope", "xenium"):
        im = Image2DModel.parse(
            rng.integers(0, 4000, (2, 512, 512), dtype=np.uint16),
            dims=("c", "y", "x"),
            c_coords=["other", "DAPI"],
        )
        sd.SpatialData(images={"dapi": im}, attrs={"keep": "native"}).write(
            tmp_path / f"{platform}.zarr"
        )
    cfg = AlignmentConfig(
        pair_id="HD1",
        merscope_zarr_path=tmp_path / "merscope.zarr",
        xenium_zarr_path=tmp_path / "xenium.zarr",
        output_dir=tmp_path / "align",
        valis={"preprocessing": {"mask_min_area_um2": 1.0, "mask_downsample": 1}},
        merscope_image={
            "image_key": "dapi",
            "dataset_to_image_matrix": [[4.0, 0.02, 15], [0, 4.02, 8], [0, 0, 1]],
        },
        xenium_image={
            "image_key": "dapi",
            "dataset_to_image_matrix": [[3.9, 0, -4], [0, 3.9, 6], [0, 0, 1]],
        },
    )
    package = prepare_paired_landmark_store(
        cfg, tmp_path / "HD1_prepared.zarr", target_pixel_size_um=0.6
    )
    return cfg, package


def _label(package: Path, *, reflect: bool = False, status: str = "finished") -> dict:
    attrs = read_root_attrs(package)
    store = attrs[STORE_ATTR]
    points = [
        (x + i * 1.3, y - i * 0.7)
        for i, (x, y) in enumerate(
            (x, y) for y in (45.0, 120.0, 210.0) for x in (45.0, 120.0, 210.0)
        )
    ]
    payload = {
        "schema": "merxen.manual_registration.labelling",
        "schema_version": 1,
        "status": status,
        "revision": 1,
        "updated_at": "2026-01-01T00:00:00Z",
        "written_by": {"software": "test", "version": "1"},
        "store_grid_digest": store["grid_digest"],
        "images": {
            r: {k: store["images"][r][k] for k in ("shape_yx", "pixel_size_um")}
            for r in ("reference", "moving")
        },
        "coarse": {"mirror_x": reflect, "rotation_deg": 0.0, "offset_um": [0.0, 0.0]},
        "crops": {"reference": None, "moving": None},
        "landmarks": {
            "next_id": 10,
            "colour_seed": 0,
            "pairs": [
                {
                    "id": i + 1,
                    "colour": "#00ff00",
                    "enabled": True,
                    "reference_xy_px": [x, y],
                    "moving_xy_px": [255 - x if reflect else x + 4, y + 6],
                }
                for i, (x, y) in enumerate(points)
            ],
        },
    }
    with spatialdata_write_lock(package):
        write_root_attr(package, LABELLING_ATTR, payload)
    return payload


def test_package_contains_only_dapi_and_preserves_calibration_and_landmarks(
    paired_package: tuple[AlignmentConfig, Path],
) -> None:
    cfg, package = paired_package
    sdata = sd.read_zarr(package)
    assert set(sdata.images) == {"reference", "moving"}
    assert not sdata.points and not sdata.tables and not sdata.shapes
    assert image_to_cyx(sdata.images["moving"]).coords["c"].values.tolist() == ["DAPI"]
    # Block centre: package pixel 0 samples native pixel .5, not pixel 0.
    source = sdata.attrs[PAIRED_LANDMARK_ATTR]["sources"]["moving"]
    np.testing.assert_allclose(
        source["package_to_native_image"], [[2, 0, 0.5], [0, 2, 0.5], [0, 0, 1]]
    )
    original = sd.read_zarr(cfg.merscope_zarr_path)
    expected = image_to_cyx(original.images["dapi"]).data[1, :2, :2].compute().mean()
    assert image_to_cyx(sdata.images["moving"]).data[0, 0, 0].compute() == int(expected)
    label = _label(package)
    assert (
        prepare_paired_landmark_store(cfg, package, target_pixel_size_um=0.6) == package
    )
    assert read_root_attrs(package)[LABELLING_ATTR] == label
    assert sd.read_zarr(cfg.merscope_zarr_path).attrs["keep"] == "native"


@pytest.mark.parametrize("reflect", [False, True])
def test_manual_field_maps_native_frames_and_reloads_from_store(
    paired_package: tuple[AlignmentConfig, Path], reflect: bool
) -> None:
    cfg, package = paired_package
    _label(package, reflect=reflect)
    cfg = cfg.model_copy(
        update={
            "backend": "manual_landmarks",
            "manual": cfg.manual.model_copy(
                update={"prepared_store": package, "grid_spacing_um": 5.0}
            ),
        }
    )
    run_alignment_pipeline(cfg)
    bundle = ValisTransformBundle.load(cfg.output_dir / "transform_chain.json")
    ref_px = np.array([[90.0, 100.0], [150.0, 130.0], [180.0, 180.0]])
    mov_px = ref_px.copy()
    mov_px[:, 0] = 255 - ref_px[:, 0] if reflect else ref_px[:, 0] + 4
    mov_px[:, 1] += 6
    sources = read_root_attrs(package)[PAIRED_LANDMARK_ATTR]["sources"]

    def dataset(role: str, xy: np.ndarray) -> np.ndarray:
        frame = sources[role]
        matrix = (
            np.linalg.inv(frame["native"]["dataset_to_image"])
            @ frame["package_to_native_image"]
        )
        return (np.column_stack([xy, np.ones(len(xy))]) @ matrix.T)[:, :2]

    expected, moving = dataset("reference", ref_px), dataset("moving", mov_px)
    # The two nominal package pixel sizes differ: the field learns their ratio.
    np.testing.assert_allclose(bundle.transform(moving), expected, atol=0.03)
    np.testing.assert_allclose(
        bundle.fixed_dataset_to_moving_dataset(expected), moving, atol=0.03
    )
    native_m = sd.read_zarr(cfg.merscope_zarr_path)
    native_x = sd.read_zarr(cfg.xenium_zarr_path)
    assert native_m.attrs["keep"] == "native"
    assert (
        native_m.attrs["merxen_alignment"]["transform"]["backend"] == "manual_landmarks"
    )
    # Materialization must also work after the task output has been moved away.
    materialize_cfg = cfg.model_copy(update={"output_dir": cfg.output_dir / "absent"})
    mapper = load_alignment_mapper(
        materialize_cfg,
        moving_sdata=native_m,
        fixed_sdata=native_x,
        source_image_key="dapi",
        fixed_image_key="dapi",
        fixed_shape_rc=(512, 512),
        require_raster_inverse=True,
    )
    np.testing.assert_allclose(
        mapper.moving_dataset_to_fixed_dataset(moving), expected, atol=0.03
    )


@pytest.mark.parametrize(
    "problem", ["draft", "wrong_pair", "stale_grid", "changed_native"]
)
def test_invalid_landmarks_leave_native_stores_untouched(
    paired_package: tuple[AlignmentConfig, Path], problem: str
) -> None:
    cfg, package = paired_package
    label = _label(package)
    if problem == "draft":
        label["status"] = "draft"
    elif problem == "stale_grid":
        label["store_grid_digest"] = "sha256:" + "0" * 64
    elif problem == "wrong_pair":
        cfg = cfg.model_copy(update={"pair_id": "HD2"})
    elif problem == "changed_native":
        cfg = cfg.model_copy(
            update={
                "merscope_image": cfg.merscope_image.model_copy(
                    update={
                        "dataset_to_image_matrix": [[4, 0, 99], [0, 4, 8], [0, 0, 1]]
                    }
                )
            }
        )
    with spatialdata_write_lock(package):
        write_root_attr(package, LABELLING_ATTR, label)
    cfg = cfg.model_copy(
        update={
            "backend": "manual_landmarks",
            "manual": cfg.manual.model_copy(update={"prepared_store": package}),
        }
    )
    expected = {
        "draft": "draft",
        "wrong_pair": "for this pair",
        "stale_grid": "grid digest",
        "changed_native": "changed since",
    }
    with pytest.raises(ValueError, match=expected[problem]):
        run_alignment_pipeline(cfg)
    assert "merxen_alignment" not in sd.read_zarr(cfg.merscope_zarr_path).attrs


def test_uploaded_root_json_is_used_without_overwriting_server_package(
    paired_package: tuple[AlignmentConfig, Path],
) -> None:
    cfg, package = paired_package
    _label(package)
    uploaded = package.parent / "landmarks.json"
    uploaded.write_text(json.dumps({"attributes": read_root_attrs(package)}))
    # Local server copy may still have a draft. The uploaded file is authoritative.
    draft = read_root_attrs(package)[LABELLING_ATTR] | {"status": "draft"}
    with spatialdata_write_lock(package):
        write_root_attr(package, LABELLING_ATTR, draft)
    cfg = cfg.model_copy(
        update={
            "backend": "manual_landmarks",
            "manual": cfg.manual.model_copy(
                update={
                    "prepared_store": package,
                    "labelling_path": uploaded,
                    "grid_spacing_um": 5.0,
                }
            ),
        }
    )
    run_alignment_pipeline(cfg)
    assert read_root_attrs(package)[LABELLING_ATTR]["status"] == "draft"
    assert (cfg.output_dir / "manual_dapi_overlay.png").exists()


@pytest.mark.parametrize("reflect", [False, True])
def test_inverse_follows_curved_field_with_rotation_and_reflection(
    tmp_path: Path, reflect: bool
) -> None:
    """Inverting the affine alone must not substitute for the actual GP warp."""
    reference = np.array(
        [
            (x, y)
            for x in (100.0, 1400.0, 2600.0, 3800.0)
            for y in (200.0, 1500.0, 3000.0)
        ]
    )
    theta = np.deg2rad(120)
    linear = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    if reflect:
        linear[:, 0] *= -1
    moving = reference @ linear.T + [500.0, 1000.0]
    moving[:, 0] += 140 * np.sin(reference[:, 1] / 1000)
    moving[:, 1] += 100 * np.cos(reference[:, 0] / 1400)
    field = LandmarkField(
        FieldObservations.clicks(reference, moving, np.arange(len(reference))),
        FieldHyperparameters(1000.0, 150.0, None, None, {"click": 5.0}),
        np.ones(len(reference)),
        FieldOptions(components=1),
        1,
    )
    cfg = AlignmentConfig(
        pair_id="HD1",
        merscope_zarr_path=tmp_path / "moving",
        xenium_zarr_path=tmp_path / "fixed",
        output_dir=tmp_path / "out",
        manual={"inverse_tolerance_um": 0.001},
    )
    probes = np.array([[500.0, 800.0], [1700.0, 2000.0], [3200.0, 1200.0]])
    pulled = field.mean(probes)
    recovered = _inverse(field, pulled, cfg)
    np.testing.assert_allclose(recovered, probes, atol=0.01)

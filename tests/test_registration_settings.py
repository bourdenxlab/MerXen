"""Tests for the RegistrationSettings split and the image-only ViewerCacheConfig."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from merxen.config import (
    ImageRegistrationConfig,
    RegisteredImageSpec,
    RegistrationSettings,
    ViewerCacheConfig,
)
from merxen.image_registration import _provenance

SHARED_FIELDS = [
    "reference_channel",
    "reference_pixel_size_um",
    "matrix_scale_tolerance",
    "refine_affine",
    "registration_pixel_size_um",
    "refinement_window_um",
    "min_refinement_windows",
    "max_refinement_shift_um",
    "max_refinement_rotation_deg",
    "max_refinement_scale_change",
    "nonrigid_enabled",
    "nonrigid_pixel_size_um",
    "nonrigid_attachment",
    "nonrigid_tightness",
    "nonrigid_num_warp",
    "nonrigid_num_iter",
    "nonrigid_smoothing_sigma_um",
    "nonrigid_max_displacement_um",
    "nonrigid_max_cycle_error_um",
    "nonrigid_min_local_correlation",
    "nonrigid_min_jacobian",
    "nonrigid_max_jacobian",
    "nonrigid_min_confident_fraction",
    "nonrigid_residual_passes",
    "nonrigid_min_residual_correlation_improvement",
    "nonrigid_min_residual_initial_confidence",
    "tile_size",
    "chunk_size",
    "build_viewer_pyramid",
]
CONFIG_ONLY_FIELDS = [
    "dataset_name",
    "platform",
    "latest_zarr_path",
    "output_dir",
    "images",
    "reference_image_key",
    "registration_quality_enabled",
    "registration_quality_pixel_size_um",
    "registration_quality_smoothing_sigma_um",
    "registration_quality_cell_label_key",
    "registration_quality_nuclei_label_key",
]


def _registration_payload(tmp_path: Path) -> dict[str, object]:
    return {
        "dataset_name": "SYNTH_XENIUM",
        "platform": "XENIUM",
        "latest_zarr_path": str(tmp_path / "s.zarr"),
        "output_dir": str(tmp_path / "out"),
        "images": [
            {
                "image_key": "post_if",
                "image_path": str(tmp_path / "img.bin"),
                "alignment_matrix_path": str(tmp_path / "mat.csv"),
                "channel_names": ["DAPI", "p62"],
            }
        ],
    }


def test_image_registration_config_field_set_is_unchanged() -> None:
    """The split keeps the same 40 field names on ImageRegistrationConfig."""
    assert set(ImageRegistrationConfig.model_fields) == set(
        SHARED_FIELDS + CONFIG_ONLY_FIELDS
    )
    assert len(ImageRegistrationConfig.model_fields) == 40


def test_registration_settings_holds_the_shared_fields() -> None:
    """RegistrationSettings carries the 29 shared fields in their original order."""
    assert list(RegistrationSettings.model_fields) == SHARED_FIELDS
    assert issubclass(ImageRegistrationConfig, RegistrationSettings)


def test_field_signatures_unchanged_by_the_split() -> None:
    """Annotations, defaults and constraints are the pre-split ones."""
    sig = {
        name: [
            repr(field.annotation),
            repr(field.default),
            [repr(item) for item in field.metadata],
        ]
        for name, field in ImageRegistrationConfig.model_fields.items()
    }
    # Computed on main before the split with pydantic 2.13.0. If a pydantic
    # upgrade changes the reprs, recompute it on the pre-split commit, never on
    # the new code.
    assert (
        hashlib.sha256(json.dumps(sig, sort_keys=True).encode()).hexdigest()
        == "4038b7439dedba462df34670237dfa6544822876561938bd5d12d0b0290c1dc9"
    )


@pytest.mark.parametrize(
    ("nonrigid_enabled", "expected"),
    [
        (
            False,
            "f77cde35b2d9f0aae16cedc46a62cffdee38bcddaf8428e3b452e4693dad2f3a",
        ),
        (
            True,
            "ae035f23c8719903971a280775557711e7d91d98ce61026b5863f83ddedd3ede",
        ),
    ],
)
def test_provenance_settings_unchanged_by_the_split(
    tmp_path: Path, nonrigid_enabled: bool, expected: str
) -> None:
    """Registered-image provenance (hence resume behaviour) is unchanged."""
    image = tmp_path / "img.bin"
    image.write_bytes(b"x")
    matrix = tmp_path / "mat.csv"
    matrix.write_text("1,0,0\n0,1,0\n0,0,1\n")
    config = ImageRegistrationConfig(
        dataset_name="SYNTH_XENIUM",
        platform="XENIUM",
        latest_zarr_path=tmp_path / "s.zarr",
        output_dir=tmp_path / "out",
        images=[
            RegisteredImageSpec(
                image_key="post_if",
                image_path=image,
                alignment_matrix_path=matrix,
                channel_names=["DAPI", "p62"],
            )
        ],
        nonrigid_enabled=nonrigid_enabled,
    )
    provenance = _provenance(config.images[0], np.eye(3), ["DAPI", "p62"], config)
    for key in ("image_path", "image_size_bytes", "image_mtime_ns"):
        provenance.pop(key)
    digest = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode())
    assert digest.hexdigest() == expected


def test_image_registration_config_still_ignores_unknown_keys(
    tmp_path: Path,
) -> None:
    """The default extra='ignore' behaviour is kept on both classes."""
    payload = _registration_payload(tmp_path)
    config = ImageRegistrationConfig.model_validate({**payload, "unknown": 1})
    assert config.dataset_name == "SYNTH_XENIUM"
    assert RegistrationSettings.model_validate({"unknown": 1}).reference_channel == (
        "DAPI"
    )


def test_viewer_cache_config_field_order() -> None:
    """The new fields are appended; the existing ones keep their positions."""
    assert list(ViewerCacheConfig.model_fields) == [
        "dataset_name",
        "platform",
        "latest_zarr_path",
        "original_data_path",
        "output_dir",
        "transform_path",
        "downsample",
        "label_chunk_size",
        "contour_width",
        "min_size",
        "shape_keys",
        "build_image_pyramid",
        "image_only",
        "image_keys",
    ]


def test_viewer_cache_config_modes(tmp_path: Path) -> None:
    """Main mode needs platform and original_data_path; image-only does not."""
    base = {
        "dataset_name": "SYNTH",
        "latest_zarr_path": str(tmp_path / "s.zarr"),
        "output_dir": str(tmp_path / "out"),
    }
    main = {
        **base,
        "platform": "XENIUM",
        "original_data_path": str(tmp_path / "raw"),
    }
    message = "platform and original_data_path are required unless image_only is true"

    assert ViewerCacheConfig.model_validate(main).image_only is False
    without_platform = {k: v for k, v in main.items() if k != "platform"}
    with pytest.raises(ValidationError, match=message):
        ViewerCacheConfig.model_validate(without_platform)
    without_path = {k: v for k, v in main.items() if k != "original_data_path"}
    with pytest.raises(ValidationError, match=message):
        ViewerCacheConfig.model_validate(without_path)

    image_only = ViewerCacheConfig.model_validate({**base, "image_only": True})
    assert image_only.platform is None
    assert image_only.original_data_path is None
    assert image_only.image_keys is None

    with_keys = ViewerCacheConfig.model_validate(
        {**base, "image_only": True, "image_keys": ["reference"]}
    )
    assert with_keys.image_keys == ["reference"]

    with pytest.raises(ValidationError, match="only used when image_only"):
        ViewerCacheConfig.model_validate({**main, "image_keys": ["reference"]})
    for bad_keys in ([], ["a", "a"]):
        with pytest.raises(ValidationError, match="non-empty and unique"):
            ViewerCacheConfig.model_validate(
                {**base, "image_only": True, "image_keys": bad_keys}
            )

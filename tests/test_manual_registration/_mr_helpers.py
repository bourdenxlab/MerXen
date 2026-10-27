"""Shared builders of store and labelling payloads for the register tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from merxen.manual_registration.geometry import coarse_matrix_xy
from merxen.manual_registration.schema import (
    CONVENTIONS,
    LABELLING_ATTR,
    LABELLING_SCHEMA,
    STORE_ATTR,
    STORE_SCHEMA,
    CoarseTransform,
    CropRect,
    LandmarkPair,
    file_fingerprint,
    labelling_content_sha256,
    pair_colour,
)

DIGEST_A = "sha256:" + "a" * 64
DIGEST_B = "sha256:" + "b" * 64
DIGEST_C = "sha256:" + "c" * 64

# The header rows of the CSV outputs, spelled out (master design 3.8) so that a
# change of the column tuples in the code cannot go unnoticed.
LANDMARKS_CSV_HEADER = (
    "id,colour,enabled,used_in_fit,outlier,outside_crop,reference_x_px,"
    "reference_y_px,moving_x_px,moving_y_px,landmark_x_px,landmark_y_px,"
    "affine_x_px,affine_y_px,final_x_px,final_y_px,residual_landmark_um,"
    "residual_loo_um,residual_affine_um,residual_final_um,final_includes_flow,"
    "push_converged"
)
COMPARISON_CSV_HEADER = (
    "run_id,preset,run_label,fit_n_enabled,fit_n_used,fit_rms_um,"
    "fit_scale_ratio,fit_rotation_deg,refinement_status,refinement_max_shift_um,"
    "refinement_rotation_deg,refinement_scale_change,nonrigid_status,"
    "nonrigid_confident_fraction,nonrigid_pearson_before,nonrigid_pearson_after,"
    "jacobian_min,jacobian_max,tissue_pearson_coarse,tissue_pearson_landmark,"
    "tissue_pearson_affine,tissue_pearson_final,used_median_landmark_um,"
    "used_median_affine_um,used_median_final_um,held_out_median_landmark_um,"
    "held_out_median_affine_um,held_out_median_final_um,nucleus_qc_median_um,"
    "nucleus_qc_pearson,wall_time_s,peak_rss_gb"
)

# Appendix A.2 of the design: a 10-pair robust-fit example with a planted outlier.
A2_PIXEL_SIZE = 0.34511
A2_MOVING_SHAPE = (48000, 72000)
A2_REFERENCE_SHAPE = (61000, 80000)
A2_COARSE = CoarseTransform(
    rotation_deg=179.0, mirror_x=False, offset_um=(150.0, -80.0)
)
A2_MOVING = [
    (5000, 4000),
    (30000, 6000),
    (65000, 9000),
    (12000, 24000),
    (40000, 22000),
    (68000, 30000),
    (8000, 42000),
    (36000, 44000),
    (60000, 40000),
    (20000, 12000),
]
A2_REFERENCE = [
    (71081.083, 50804.36),
    (46116.596, 48370.654),
    (11176.784, 44760.977),
    (64428.598, 30684.939),
    (36517.762, 32197.367),
    (8542.641, 23707.218),
    (68745.332, 12759.59),
    (40781.701, 10270.827),
    (16716.747, 13849.06),
    (56219.587, 42546.592),
]


def a2_truth_matrix() -> np.ndarray:
    """The matrix the A.2 reference points were generated from (without noise)."""
    return coarse_matrix_xy(
        A2_COARSE,
        moving_shape_yx=A2_MOVING_SHAPE,
        reference_shape_yx=A2_REFERENCE_SHAPE,
        moving_pixel_size_um=A2_PIXEL_SIZE,
        reference_pixel_size_um=A2_PIXEL_SIZE,
    )


def make_pairs(
    moving: Sequence[Sequence[float]],
    reference: Sequence[Sequence[float]],
    *,
    disabled: Sequence[int] = (),
) -> list[LandmarkPair]:
    """Landmark pairs with ids 1..N."""
    return [
        LandmarkPair(
            id=i + 1,
            colour=pair_colour(i + 1, 0),
            enabled=(i + 1) not in disabled,
            reference_xy_px=(float(reference[i][0]), float(reference[i][1])),
            moving_xy_px=(float(moving[i][0]), float(moving[i][1])),
        )
        for i in range(len(moving))
    ]


def a2_pairs(*, disabled: Sequence[int] = ()) -> list[LandmarkPair]:
    return make_pairs(A2_MOVING, A2_REFERENCE, disabled=disabled)


def store_attr_dict(
    ref_shape: tuple[int, int],
    mov_shape: tuple[int, int],
    p_r: float,
    p_m: float,
    ome_paths: Mapping[str, Path] | None = None,
    *,
    kind: str = "prepared",
    dataset_name: str = "SYN1",
    schema_version: int = 1,
) -> dict[str, Any]:
    """A ``STORE_ATTR`` payload that passes ``StoreAttrs`` (prepared by default)."""
    images: dict[str, Any] = {}
    for role, shape, pixel in (
        ("reference", ref_shape, p_r),
        ("moving", mov_shape, p_m),
    ):
        path = (ome_paths or {}).get(role)
        fingerprint = file_fingerprint(path) if path is not None else DIGEST_C
        images[role] = {
            "element": role,
            "role": role,
            "shape_yx": list(shape),
            "dtype": "uint16",
            "pixel_size_um": pixel,
            "channel_names": ["DAPI", "AF568"],
            "registration_channel": "DAPI",
            "source_channel_names": ["DAPI", "AF568"],
            "canonical_channel_names": ["DAPI", "AF568"],
            "ome_tiff": {
                "path": str(path) if path is not None else f"/none/{role}.ome.tif",
                "fingerprint": fingerprint,
                "written_by_prepare": True,
                "level_shapes_yx": [list(shape)],
                "exact_level_factors": False,
            },
        }
    payload: dict[str, Any] = {
        "schema": STORE_SCHEMA,
        "schema_version": schema_version,
        "kind": kind,
        "dataset_name": dataset_name,
        "created_at": "2026-01-01T00:00:00Z",
        "merxen_version": "0",
        "conventions": dict(CONVENTIONS),
        "images": images,
        "channel_matching": {
            "reference_kept": ["DAPI", "AF568"],
            "reference_dropped": [],
            "moving_unmatched": [],
        },
        "grid_digest": DIGEST_A,
        "content_digest": DIGEST_B,
    }
    if kind == "result":
        payload.update(
            {
                "run_id": f"{dataset_name}_same_section",
                "preset": "same_section",
                "prepared_store": {
                    "path": "/none/prepared.zarr",
                    "content_digest": DIGEST_B,
                    "grid_digest": DIGEST_A,
                },
            }
        )
    return payload


def labelling_dict(
    pairs: Sequence[LandmarkPair],
    coarse: CoarseTransform,
    crops: Mapping[str, CropRect | None] | None = None,
    status: str = "finished",
    *,
    ref_shape: tuple[int, int] = A2_REFERENCE_SHAPE,
    mov_shape: tuple[int, int] = A2_MOVING_SHAPE,
    p_r: float = A2_PIXEL_SIZE,
    p_m: float = A2_PIXEL_SIZE,
    outlier_floor_um: float | None = None,
    schema_version: int = 1,
    revision: int = 1,
    grid_digest: str | None = DIGEST_A,
) -> dict[str, Any]:
    """A ``LABELLING_ATTR`` payload that passes ``LabellingAttrs``."""
    crop_map = dict(crops or {})
    ids = [p.id for p in pairs]
    payload: dict[str, Any] = {
        "schema": LABELLING_SCHEMA,
        "schema_version": schema_version,
        "status": status,
        "revision": revision,
        "updated_at": "2026-01-01T00:00:00Z",
        "finished_at": "2026-01-01T00:00:00Z" if status == "finished" else None,
        "written_by": {"software": "test", "version": "0"},
        "store_grid_digest": grid_digest,
        "images": {
            "reference": {"shape_yx": list(ref_shape), "pixel_size_um": p_r},
            "moving": {"shape_yx": list(mov_shape), "pixel_size_um": p_m},
        },
        "coarse": {
            "rotation_deg": coarse.rotation_deg,
            "mirror_x": coarse.mirror_x,
            "offset_um": list(coarse.offset_um),
        },
        "crops": {
            role: None if crop_map.get(role) is None else crop_map[role].model_dump()
            for role in ("reference", "moving")
        },
        "landmarks": {
            "next_id": max(ids) + 1 if ids else 1,
            "colour_seed": 0,
            "pairs": [p.model_dump(mode="json") for p in pairs],
        },
    }
    if outlier_floor_um is not None:
        payload["outlier_floor_um"] = outlier_floor_um
    payload["content_sha256"] = labelling_content_sha256(payload)
    return payload


def fake_store(path: Path, attrs: Mapping[str, Any]) -> Path:
    """An empty Zarr group with the given root attributes (enough for attrs readers)."""
    group = zarr.open_group(str(path), mode="w")
    group.attrs.update(dict(attrs))
    return path


def a2_store(
    tmp_path: Path,
    *,
    disabled: Sequence[int] = (),
    coarse: CoarseTransform = A2_COARSE,
    crops: Mapping[str, CropRect | None] | None = None,
    status: str = "finished",
    outlier_floor_um: float | None = None,
    name: str = "SYN1_prepared.zarr",
) -> Path:
    """A fake prepared store holding the A.2 labelling."""
    store = store_attr_dict(
        A2_REFERENCE_SHAPE, A2_MOVING_SHAPE, A2_PIXEL_SIZE, A2_PIXEL_SIZE
    )
    labelling = labelling_dict(
        a2_pairs(disabled=disabled),
        coarse,
        crops,
        status,
        outlier_floor_um=outlier_floor_um,
    )
    return fake_store(tmp_path / name, {STORE_ATTR: store, LABELLING_ATTR: labelling})

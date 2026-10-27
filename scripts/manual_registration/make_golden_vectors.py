#!/usr/bin/env python3
"""Generate the cross-repo golden vectors of manual landmark registration.

The output (``tests/test_data/manual_registration/golden_v1.json``) pins the
constants, the coarse-transform and crop geometry, the similarity fits, the
canonical digests and example store/labelling payloads. The napari viewer copies
the file byte-for-byte and pins its SHA-256; its own implementation of the maths
must reproduce every stored output from the stored inputs.

Usage:
    python scripts/manual_registration/make_golden_vectors.py [--output PATH] [--check]

Without ``--check`` the file is written and its SHA-256 printed. With ``--check``
the vectors are rebuilt in memory and compared with the file (floats with
``rtol = atol = 1e-9``); the exit status is 1 on the first difference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from merxen.manual_registration import geometry, landmark_fit, schema

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "tests/test_data/manual_registration/golden_v1.json"
GENERATED_BY = "scripts/manual_registration/make_golden_vectors.py"
FLOAT_RTOL = 1e-9
FLOAT_ATOL = 1e-9

CONSTANT_NAMES = (
    "SCHEMA_VERSION",
    "STORE_ATTR",
    "LABELLING_ATTR",
    "RESULT_ATTR",
    "LAYER_METADATA_KEY",
    "MIN_LANDMARK_PAIRS",
    "DEFAULT_OUTLIER_FLOOR_UM",
    "OUTLIER_K",
    "MAX_OUTLIER_FRACTION",
    "MIN_USED_PAIRS",
    "CROP_SNAP_PX",
    "CROP_MIN_SIZE_PX",
)

# Appendix A of the work-package spec: a 10-pair data set with one displaced pair.
PIXEL_UM = 0.34511
APPENDIX_COARSE = {
    "rotation_deg": 179.0,
    "mirror_x": False,
    "offset_um": [150.0, -80.0],
}
APPENDIX_MOVING_SHAPE = (48000, 72000)
APPENDIX_REFERENCE_SHAPE = (61000, 80000)
APPENDIX_MOVING = [
    (5000.0, 4000.0),
    (30000.0, 6000.0),
    (65000.0, 9000.0),
    (12000.0, 24000.0),
    (40000.0, 22000.0),
    (68000.0, 30000.0),
    (8000.0, 42000.0),
    (36000.0, 44000.0),
    (60000.0, 40000.0),
    (20000.0, 12000.0),
]
APPENDIX_REFERENCE = [
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
APPENDIX_NOISE = [
    (1.2, -0.8),
    (-2.0, 1.5),
    (0.5, 2.2),
    (-1.4, -1.1),
    (2.5, 0.3),
    (-0.6, -2.4),
    (1.8, 1.0),
    (-1.0, 0.6),
    (0.2, -1.7),
    (-2.2, 2.0),
]
LABELLING_CONTENT_SHA256 = (
    "sha256:f6b16674a10854ab06ff385a6a349ce325d874abf0e6c9d78e0a3b7683e328ad"
)
WRAP_EXAMPLES = [
    (0.0, 0.0),
    (180.0, 180.0),
    (-180.0, 180.0),
    (190.0, -170.0),
    (-190.0, 170.0),
    (540.0, 180.0),
    (-540.0, 180.0),
    (359.5, -0.5),
    (720.0, 0.0),
]


# ----------------------------------------------------------------------------
# helpers


def _plain(value: Any) -> Any:
    """Convert numpy data to plain JSON types; NaN becomes ``None``."""
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, bool) or value is None or isinstance(value, str | int):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) else value
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, Sequence):
        return [_plain(item) for item in value]
    raise TypeError(f"cannot render {type(value).__name__}")


def _coarse_dict(coarse: schema.CoarseTransform) -> dict[str, Any]:
    return {
        "rotation_deg": coarse.rotation_deg,
        "mirror_x": coarse.mirror_x,
        "offset_um": list(coarse.offset_um),
    }


def _crop_dict(crop: schema.CropRect | None) -> dict[str, int] | None:
    if crop is None:
        return None
    return {"x0": crop.x0, "y0": crop.y0, "x1": crop.x1, "y1": crop.y1}


def _crop(values: Sequence[int] | None) -> schema.CropRect | None:
    if values is None:
        return None
    return schema.CropRect(x0=values[0], y0=values[1], x1=values[2], y1=values[3])


def _placeholder_digest(label: str) -> str:
    return schema.canonical_sha256({"golden": label})


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(f"golden vector self-check failed: {message}")


def _close(actual: Any, expected: Any, atol: float) -> bool:
    return bool(
        np.allclose(
            np.asarray(actual, dtype=float),
            np.asarray(expected, dtype=float),
            rtol=0.0,
            atol=atol,
        )
    )


# ----------------------------------------------------------------------------
# coarse transform cases


def _coarse_cases() -> list[dict[str, Any]]:
    specs: list[
        tuple[
            str, schema.CoarseTransform, tuple[int, int], tuple[int, int], float, float
        ]
    ] = [
        ("identity", schema.CoarseTransform(), (1000, 2000), (1000, 2000), 0.5, 0.5),
        (
            "rotated_and_scaled",
            schema.CoarseTransform(rotation_deg=30, offset_um=(12.5, -40)),
            (1001, 1500),
            (2000, 3001),
            0.3434,
            0.2125,
        ),
        (
            "mirrored",
            schema.CoarseTransform(
                rotation_deg=-125, mirror_x=True, offset_um=(-100, 250)
            ),
            (48000, 72000),
            (61000, 80000),
            0.34511,
            0.34511,
        ),
        (
            "near_half_turn",
            schema.CoarseTransform(rotation_deg=178.5, offset_um=(120, -35.5)),
            (31347, 33453),
            (51313, 47618),
            0.3434,
            0.2125,
        ),
        (
            "hand_computed",
            schema.CoarseTransform(rotation_deg=90, mirror_x=True),
            (3, 4),
            (5, 6),
            1.0,
            1.0,
        ),
        (
            "negative_quarter_turn_mixed_pixel_sizes",
            schema.CoarseTransform(rotation_deg=-90, offset_um=(-7.25, 3.5)),
            (1024, 1023),
            (2047, 2048),
            0.5,
            0.25,
        ),
        (
            "half_turn_mirrored_large",
            schema.CoarseTransform(
                rotation_deg=180, mirror_x=True, offset_um=(100, -100)
            ),
            (61003, 80112),
            (55012, 76544),
            0.34511,
            0.34511,
        ),
        (
            "labelling_example",
            schema.CoarseTransform(rotation_deg=179.2, offset_um=(152.4, -88.0)),
            (61003, 80112),
            (55012, 76544),
            0.34511,
            0.34511,
        ),
    ]
    cases = []
    for name, coarse, moving, reference, p_m, p_r in specs:
        matrix = geometry.coarse_matrix_xy(
            coarse,
            moving_shape_yx=moving,
            reference_shape_yx=reference,
            moving_pixel_size_um=p_m,
            reference_pixel_size_um=p_r,
        )
        recovered, scale_ratio = geometry.coarse_from_matrix(
            matrix,
            moving_shape_yx=moving,
            reference_shape_yx=reference,
            moving_pixel_size_um=p_m,
            reference_pixel_size_um=p_r,
        )
        cases.append(
            {
                "name": name,
                "coarse": _coarse_dict(coarse),
                "moving_shape_yx": list(moving),
                "reference_shape_yx": list(reference),
                "moving_pixel_size_um": p_m,
                "reference_pixel_size_um": p_r,
                "matrix_xy": matrix,
                "from_matrix": {
                    "coarse": _coarse_dict(recovered),
                    "scale_ratio": scale_ratio,
                },
            }
        )
    return cases


def _coarse_transitions() -> list[dict[str, Any]]:
    transitions = []
    for theta in (0.0, 30.0, -125.0, 178.5, 180.0):
        for mirror in (False, True):
            start = schema.CoarseTransform(
                rotation_deg=theta, mirror_x=mirror, offset_um=(12.5, -40.0)
            )
            actions: list[tuple[str, Any, schema.CoarseTransform]] = [
                ("flip_horizontal", None, geometry.flipped_horizontal(start)),
                ("flip_vertical", None, geometry.flipped_vertical(start)),
                ("rotate", 37.0, geometry.rotated(start, 37.0)),
                ("rotate", -200.0, geometry.rotated(start, -200.0)),
                (
                    "set_rotation",
                    270.0,
                    schema.CoarseTransform(
                        rotation_deg=270.0,
                        mirror_x=start.mirror_x,
                        offset_um=start.offset_um,
                    ),
                ),
                ("shift", [10.0, -5.0], geometry.shifted(start, 10.0, -5.0)),
                ("reset", None, schema.CoarseTransform()),
            ]
            for action, argument, result in actions:
                transitions.append(
                    {
                        "start": _coarse_dict(start),
                        "action": action,
                        "argument": argument,
                        "result": _coarse_dict(result),
                    }
                )
    return transitions


def _snap_crop_cases() -> list[dict[str, Any]]:
    big = (55012, 76544)
    small = (6000, 6000)
    specs: list[tuple[str, Sequence[int] | None, tuple[int, int], int]] = [
        ("no_crop", None, big, 64),
        ("interior", (1000, 500, 71000, 50500), big, 64),
        ("edge_unchanged", (0, 0, 76544, 55012), big, 64),
        ("near_edges", (63, 65, 76543, 55011), big, 64),
        ("exact_minimum_width", (10, 0, 1090, 5000), small, 64),
        ("below_minimum", (65, 0, 1090, 5000), small, 64),
        ("exceeds_image", (0, 0, 6001, 10), small, 64),
        ("multiple_one", (3, 5, 1500, 1700), (2000, 2000), 1),
    ]
    cases = []
    for name, crop, shape, multiple in specs:
        try:
            effective: dict[str, int] | None = _crop_dict(
                geometry.snap_crop(
                    _crop(crop), shape, multiple=multiple, min_size_px=1024
                )
            )
            error = False
        except ValueError:
            effective, error = None, True
        cases.append(
            {
                "name": name,
                "crop": None if crop is None else _crop_dict(_crop(crop)),
                "shape_yx": list(shape),
                "multiple": multiple,
                "min_size_px": 1024,
                "effective": effective,
                "error": error,
            }
        )
    return cases


def _crop_frames(coarse_cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frames = []
    for case_index, origin_ref, origin_mov in (
        (2, (1024.0, 512.0), (64.0, 128.0)),
        (1, (0.0, 0.0), (64.0, 0.0)),
    ):
        matrix_full = np.asarray(coarse_cases[case_index]["matrix_xy"])
        frames.append(
            {
                "matrix_full": matrix_full,
                "reference_origin_xy": list(origin_ref),
                "moving_origin_xy": list(origin_mov),
                "matrix_crop": geometry.crop_matrix(
                    matrix_full,
                    reference_origin_xy=origin_ref,
                    moving_origin_xy=origin_mov,
                ),
            }
        )
    return frames


def _points_outside_crop() -> list[dict[str, Any]]:
    points = [
        (9.5, 20.0),
        (9.4999, 25.0),
        (29.4999, 39.4999),
        (29.5, 30.0),
        (20.0, 39.5),
        (20.0, 19.5),
    ]
    entries = []
    for crop in (schema.CropRect(x0=10, y0=20, x1=30, y1=40), None):
        entries.append(
            {
                "crop": _crop_dict(crop),
                "points_xy": points,
                "outside": geometry.points_outside_crop(points, crop),
            }
        )
    return entries


def _napari_affine(coarse_cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries = []
    for scale, case_index in ((0.5, 4), (0.34511, 2)):
        reference_base = np.diag([scale, scale, 1.0])
        matrix = np.asarray(coarse_cases[case_index]["matrix_xy"])
        entries.append(
            {
                "reference_base_rc": reference_base,
                "matrix_xy": matrix,
                "affine_rc": geometry.napari_affine_rc(reference_base, matrix),
            }
        )
    return entries


# ----------------------------------------------------------------------------
# fits


def _truth_matrix(
    theta: float, scale: float, translation: tuple[float, float]
) -> np.ndarray:
    return (
        geometry.translation_xy(*translation)
        @ geometry.rotation_matrix_xy(theta)
        @ np.diag([scale, scale, 1.0])
    )


def _fit_entry(
    name: str, moving: np.ndarray, reference: np.ndarray, reflection: bool
) -> dict[str, Any]:
    fit = landmark_fit.fit_similarity(moving, reference, reflection=reflection)
    loo = landmark_fit.leave_one_out_residuals(moving, reference, reflection=reflection)
    return {
        "name": name,
        "moving_xy": moving,
        "reference_xy": reference,
        "reflection": reflection,
        "matrix": fit.matrix,
        "scale": fit.scale,
        "rotation_deg": fit.rotation_deg,
        "reflected": fit.reflected,
        "residuals_px": fit.residuals_px,
        "loo_px": loo,
    }


def _fits(coarse_cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries = []

    rng = np.random.default_rng(73)
    moving = rng.uniform(0.0, 20000.0, size=(12, 2))
    truth = _truth_matrix(73.0, 1.37, (1200.0, -300.0))
    reference = geometry.apply_matrix_xy(truth, moving) + rng.normal(
        0.0, 0.5, size=(12, 2)
    )
    entries.append(_fit_entry("proper_theta73_scale1.37", moving, reference, False))

    rng = np.random.default_rng(125)
    moving = rng.uniform(0.0, 1.0, size=(12, 2)) * np.array([72000.0, 48000.0])
    truth = np.asarray(coarse_cases[2]["matrix_xy"])
    reference = geometry.apply_matrix_xy(truth, moving) + rng.normal(
        0.0, 0.5, size=(12, 2)
    )
    entries.append(_fit_entry("reflected_case3", moving, reference, True))

    rng = np.random.default_rng(180)
    moving = rng.uniform(0.0, 20000.0, size=(12, 2))
    truth = _truth_matrix(180.0, 0.6, (-500.0, 800.0))
    reference = geometry.apply_matrix_xy(truth, moving) + rng.normal(
        0.0, 0.5, size=(12, 2)
    )
    entries.append(_fit_entry("half_turn_scale0.6", moving, reference, False))

    moving = np.array(APPENDIX_MOVING)
    reference = np.array(APPENDIX_REFERENCE)
    entries.append(_fit_entry("appendix_a_proper", moving, reference, False))
    entries.append(_fit_entry("appendix_a_wrong_class", moving, reference, True))
    return entries


def _spread_ratios() -> list[dict[str, Any]]:
    clouds = [
        [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)],
        [(0.0, 0.0), (4.0, 0.0), (0.0, 1.0), (4.0, 1.0)],
        [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)],
    ]
    entries = []
    for points in clouds:
        value = landmark_fit.spread_ratio(points)
        entries.append({"points_xy": points, "value": value})
    _require(abs(entries[0]["value"] - 1.0) < 1e-12, "square spread ratio")
    _require(abs(entries[1]["value"] - 0.25) < 1e-12, "rectangle spread ratio")
    _require(entries[2]["value"] < 1e-12, "collinear spread ratio")
    return entries


def _robust_entry(
    name: str,
    moving: np.ndarray,
    reference: np.ndarray,
    *,
    reflection: bool,
    p_m: float,
    p_r: float,
    options: landmark_fit.RobustFitOptions,
) -> tuple[dict[str, Any], landmark_fit.RobustSimilarityFit]:
    result = landmark_fit.robust_similarity(
        moving,
        reference,
        reflection=reflection,
        moving_pixel_size_um=p_m,
        reference_pixel_size_um=p_r,
        options=options,
    )
    first_pass = (
        landmark_fit.leave_one_out_residuals(moving, reference, reflection=reflection)
        * p_r
    )
    entry = {
        "name": name,
        "moving_xy": moving,
        "reference_xy": reference,
        "reflection": reflection,
        "moving_pixel_size_um": p_m,
        "reference_pixel_size_um": p_r,
        "options": {
            "floor_um": options.floor_um,
            "k": options.k,
            "max_outlier_fraction": options.max_outlier_fraction,
            "min_used_pairs": options.min_used_pairs,
            "trim_outliers": options.trim_outliers,
        },
        "matrix": result.fit.matrix,
        "scale": result.fit.scale,
        "rotation_deg": result.fit.rotation_deg,
        "used": result.used,
        "outlier": result.outlier,
        "residuals_um": result.residuals_um,
        "loo_um": result.loo_um,
        "chosen_class_median_um": result.chosen_class_median_um,
        "other_class_median_um": result.other_class_median_um,
        "spread_ratio": result.spread_ratio,
        "scale_ratio": result.scale_ratio,
        "first_pass_loo_um": first_pass,
    }
    return entry, result


def _robust_fits() -> list[dict[str, Any]]:
    moving = np.array(APPENDIX_MOVING)
    reference = np.array(APPENDIX_REFERENCE)
    default = landmark_fit.RobustFitOptions()
    entries = []

    def add(
        name: str,
        mov: np.ndarray,
        ref: np.ndarray,
        options: landmark_fit.RobustFitOptions,
        *,
        p: float = PIXEL_UM,
    ) -> landmark_fit.RobustSimilarityFit:
        entry, result = _robust_entry(
            name, mov, ref, reflection=False, p_m=p, p_r=p, options=options
        )
        entries.append(entry)
        return result

    base = add("appendix_a_default", moving, reference, default)
    _require(np.flatnonzero(~base.used).tolist() == [4], "appendix A trims pair 4")
    untrimmed = add(
        "appendix_a_no_trimming",
        moving,
        reference,
        landmark_fit.RobustFitOptions(trim_outliers=False),
    )
    _require(bool(untrimmed.used.all()), "no trimming keeps every pair")
    _require(np.flatnonzero(untrimmed.outlier).tolist() == [4], "pair 4 flagged")

    three = reference.copy()
    three[1, 1] += 60.0 / PIXEL_UM
    three[7, 1] += 80.0 / PIXEL_UM
    result = add("three_outliers_stop_at_fraction", moving, three, default)
    _require(set(np.flatnonzero(~result.used).tolist()) == {1, 7}, "trim stops at 25%")

    one_more = reference.copy()
    one_more[7, 1] += 80.0 / PIXEL_UM
    limited = add(
        "min_used_pairs_stops_trimming",
        moving,
        one_more,
        landmark_fit.RobustFitOptions(min_used_pairs=9),
    )
    _require(
        np.flatnonzero(~limited.used).tolist() == [7], "min_used_pairs stops trimming"
    )
    both = add("two_outliers_default", moving, one_more, default)
    _require(
        set(np.flatnonzero(~both.used).tolist()) == {4, 7}, "default trims 4 and 7"
    )

    nan_moving = np.array([(1000.0, 1000.0)] * 6 + [(9000.0, 4000.0)])
    nan_reference = geometry.apply_matrix_xy(
        geometry.translation_xy(500.0, -300.0) @ geometry.rotation_matrix_xy(30.0),
        nan_moving,
    )
    nan_reference[:5] += np.array(
        [(0.4, -0.2), (-0.3, 0.5), (0.1, 0.2), (-0.2, -0.4), (0.3, 0.1)]
    )
    nan_reference[5, 0] += 200.0
    trimmed = add("nan_loo_trimmed", nan_moving, nan_reference, default, p=0.5)
    _require(np.flatnonzero(~trimmed.used).tolist() == [5], "NaN-LOO case trims pair 6")
    _require(np.flatnonzero(trimmed.outlier).tolist() == [5], "NaN-LOO case outlier")
    _require(bool(np.isnan(trimmed.loo_um[6])), "NaN-LOO case keeps NaN for pair 7")
    flagged_only = add(
        "nan_loo_no_trimming",
        nan_moving,
        nan_reference,
        landmark_fit.RobustFitOptions(trim_outliers=False),
        p=0.5,
    )
    _require(
        np.flatnonzero(flagged_only.outlier).tolist() == [5], "NaN-LOO flags pair 6"
    )
    _require(bool(flagged_only.used.all()), "NaN-LOO without trimming uses all pairs")
    return entries


# ----------------------------------------------------------------------------
# colours, canonical JSON, examples


def _pair_colours() -> list[dict[str, Any]]:
    return [
        {"pair_id": pair_id, "seed": seed, "colour": schema.pair_colour(pair_id, seed)}
        for seed in (0, 4211)
        for pair_id in range(1, 13)
    ]


def _canonical() -> list[dict[str, Any]]:
    objects: list[Any] = [
        {"b": 1, "a": [1.5, None, True], "c": "é"},
        {"z": {"y": [3, 2.5e-7, -0.0]}, "a": "x"},
    ]
    return [
        {
            "object": obj,
            "canonical_json": schema.canonical_json(obj),
            "sha256": schema.canonical_sha256(obj),
        }
        for obj in objects
    ]


def _store_image(role: str, *, moving: bool) -> dict[str, Any]:
    shape = [61003, 80112] if moving else [55012, 76544]
    levels = (
        [
            [61003, 80112],
            [30502, 40056],
            [15251, 20028],
            [7626, 10014],
            [3813, 5007],
            [1907, 2504],
            [954, 1252],
            [477, 626],
        ]
        if moving
        else [
            [55012, 76544],
            [27506, 38272],
            [13753, 19136],
            [6877, 9568],
            [3439, 4784],
            [1720, 2392],
            [860, 1196],
            [430, 598],
        ]
    )
    names = ["DAPI", "AF568", "AF647"]
    return {
        "element": role,
        "role": role,
        "shape_yx": shape,
        "dtype": "uint16",
        "pixel_size_um": 0.34511,
        "pixel_size_source": "metadata",
        "channel_names": names,
        "registration_channel": "DAPI",
        "source_channel_names": names,
        "canonical_channel_names": names,
        "name_sources": ["Name", "Name", "Name"],
        "ome_tiff": {
            "path": f"/path/to/LCM1_{role}.ome.tif",
            "fingerprint": _placeholder_digest(f"{role}_ome_tiff"),
            "written_by_prepare": True,
            "level_shapes_yx": levels,
            "exact_level_factors": True,
        },
        "source": {
            "format": "czi",
            "path": f"/path/to/{role}.czi",
            "fingerprint": _placeholder_digest(f"{role}_source"),
            "size_bytes": 0,
        },
        "czi": {
            "scene_index": 0,
            "scene_name": "ScanRegion0",
            "bbox_xywh": [-98304, 4096, 80112, 61003]
            if moving
            else [-131315, 20480, 76544, 55012],
            "z_planes_projected": 1,
            "pixel_type": "Gray16",
            "compression": "JpegXr" if moving else "Uncompressed",
        },
    }


def _store_header(kind: str) -> dict[str, Any]:
    return {
        "schema": schema.STORE_SCHEMA,
        "schema_version": schema.SCHEMA_VERSION,
        "kind": kind,
        "dataset_name": "LCM1",
        "created_at": "2026-11-02T10:15:00Z",
        "merxen_version": "0.0.0+golden",
        "conventions": dict(schema.CONVENTIONS),
    }


def _store_attr_prepared() -> dict[str, Any]:
    return {
        **_store_header("prepared"),
        "images": {
            "reference": _store_image("reference", moving=False),
            "moving": _store_image("moving", moving=True),
        },
        "channel_matching": {
            "reference_kept": ["DAPI", "AF568", "AF647"],
            "reference_dropped": [],
            "moving_unmatched": [],
        },
        "grid_digest": _placeholder_digest("grid_digest"),
        "content_digest": _placeholder_digest("content_digest"),
    }


def _store_attr_result() -> dict[str, Any]:
    names = ["DAPI", "AF568", "AF647"]
    return {
        **_store_header("result"),
        "run_id": "LCM1_serial_section",
        "preset": "serial_section",
        "run_label": None,
        "prepared_store": {
            "path": "/path/to/LCM1_prepared.zarr",
            "content_digest": _placeholder_digest("content_digest"),
            "grid_digest": _placeholder_digest("grid_digest"),
        },
        "images": {
            "reference": {
                "element": "reference",
                "role": "reference",
                "shape_yx": [49984, 69952],
                "full_shape_yx": [55012, 76544],
                "crop": {"x0": 1024, "y0": 512, "x1": 70976, "y1": 50496},
                "pixel_size_um": 0.34511,
                "dtype": "uint16",
                "channel_names": names,
                "registration_channel": "DAPI",
            },
            "moving": {
                "element": "moving_registered",
                "role": "moving",
                "shape_yx": [49984, 69952],
                "pixel_size_um": 0.34511,
                "dtype": "uint16",
                "channel_names": names,
                "registration_channel": "DAPI",
                "source_shape_yx": [61003, 80112],
                "source_pixel_size_um": 0.34511,
                "source_crop": {"x0": 0, "y0": 0, "x1": 80112, "y1": 61003},
            },
        },
    }


def _labelling(coarse_cases: list[dict[str, Any]]) -> dict[str, Any]:
    matrix = [
        [round(float(value), 7) for value in row]
        for row in np.asarray(coarse_cases[7]["matrix_xy"])
    ]
    payload: dict[str, Any] = {
        "schema": schema.LABELLING_SCHEMA,
        "schema_version": schema.SCHEMA_VERSION,
        "status": "finished",
        "revision": 9,
        "updated_at": "2026-11-02T14:02:11Z",
        "finished_at": "2026-11-02T14:02:11Z",
        "written_by": {
            "software": "napari-compare-xenium-merscope",
            "version": "0.4.0",
        },
        "store_grid_digest": _placeholder_digest("grid_digest"),
        "images": {
            "reference": {"shape_yx": [55012, 76544], "pixel_size_um": 0.34511},
            "moving": {"shape_yx": [61003, 80112], "pixel_size_um": 0.34511},
        },
        "coarse": {
            "rotation_deg": 179.2,
            "mirror_x": False,
            "offset_um": [152.4, -88.0],
            "matrix_xy": matrix,
        },
        "crops": {
            "reference": {"x0": 1000, "y0": 500, "x1": 71000, "y1": 50500},
            "moving": None,
        },
        "landmarks": {
            "next_id": 13,
            "colour_seed": 4211,
            "pairs": [
                {
                    "id": 1,
                    "colour": "#fff400",
                    "enabled": True,
                    "reference_xy_px": [12034.5, 8811.0],
                    "moving_xy_px": [64310.25, 49702.75],
                },
                {
                    "id": 2,
                    "colour": "#b219ff",
                    "enabled": False,
                    "reference_xy_px": [40211.0, 30122.5],
                    "moving_xy_px": [36420.0, 28533.0],
                },
            ],
        },
        "outlier_floor_um": 30.0,
    }
    payload["content_sha256"] = schema.labelling_content_sha256(payload)
    return payload


def _examples(coarse_cases: list[dict[str, Any]]) -> dict[str, Any]:
    labelling = _labelling(coarse_cases)
    return {
        "store_attr_prepared": _store_attr_prepared(),
        "store_attr_result": _store_attr_result(),
        "labelling": labelling,
        "labelling_content_sha256": schema.labelling_content_sha256(labelling),
    }


# ----------------------------------------------------------------------------
# self-checks of the Appendix-A literals


def _check_appendix_literals(data: dict[str, Any]) -> None:
    cases = {case["name"]: case for case in data["coarse_cases"]}
    expected_rows = {
        "identity": [[1, 0, 0], [0, 1, 0]],
        "rotated_and_scaled": [
            [1.3994970525, 0.808, 105.9004885513],
            [-0.808, 1.3994970525, 717.1121796245],
        ],
        "mirrored": [
            [0.5735764364, -0.8191520443, 38720.5118821327],
            [-0.8191520443, -0.5735764364, 74478.5184134923],
        ],
        "near_half_turn": [
            [-1.6154462372, 0.0423019485, 50730.1612067993],
            [-0.0423019485, -1.6154462372, 51515.3724415215],
        ],
        "hand_computed": [[0, 1, 1.5], [1, 0, 0.5]],
    }
    for name, rows in expected_rows.items():
        _require(
            _close(cases[name]["matrix_xy"][:2], rows, 1e-9), f"coarse case {name}"
        )
    _require(
        _close(
            cases["labelling_example"]["matrix_xy"][:2],
            [[-0.9999025, 0.0139622, 78338.8334], [-0.0139622, -0.9999025, 58307.7978]],
            1e-4,
        ),
        "labelling example matrix",
    )
    _require(
        _close(
            data["crop_frames"][0]["matrix_crop"][:2],
            [
                [0.5735764364, -0.8191520443, 37628.369312],
                [-0.8191520443, -0.5735764364, 73840.674899],
            ],
            1e-6,
        ),
        "crop frame example",
    )
    _require(
        _close(
            data["napari_affine"][0]["affine_rc"],
            [[0, 0.5, 0.25], [0.5, 0, 0.75], [0, 0, 1]],
            1e-12,
        ),
        "napari affine hand case",
    )
    for entry in data["snap_crop"]:
        expected = {
            "no_crop": [0, 0, 76544, 55012],
            "interior": [1024, 512, 70976, 50496],
            "edge_unchanged": [0, 0, 76544, 55012],
            "near_edges": [64, 128, 76480, 54976],
            "exact_minimum_width": [64, 0, 1088, 4992],
            "multiple_one": [3, 5, 1500, 1700],
        }.get(entry["name"])
        if expected is None:
            _require(entry["error"], f"snap_crop {entry['name']} must be an error")
        else:
            effective = entry["effective"]
            _require(
                [effective[k] for k in ("x0", "y0", "x1", "y1")] == expected,
                f"snap_crop {entry['name']}",
            )
    for entry in data["wrap_deg"]:
        _require(
            abs(entry["output"] - dict(WRAP_EXAMPLES)[entry["input"]]) < 1e-12,
            "wrap_deg",
        )
    robust = {entry["name"]: entry for entry in data["robust_fits"]}
    default = robust["appendix_a_default"]
    _require(
        _close(
            default["matrix"][:2],
            [[-0.999861, 0.017468, 76009.005205], [-0.017468, -0.999861, 54892.785195]],
            1e-5,
        ),
        "appendix A robust matrix",
    )
    _require(abs(default["scale"] - 1.00001339) < 1e-8, "appendix A scale")
    _require(abs(default["rotation_deg"] - 178.99913) < 1e-5, "appendix A rotation")
    _require(
        _close(
            default["residuals_um"],
            [
                0.7702,
                0.6055,
                0.9654,
                0.7053,
                41.0328,
                0.668,
                0.5918,
                0.4061,
                0.4827,
                0.8113,
            ],
            1e-3,
        ),
        "appendix A residuals",
    )
    _require(
        _close(
            default["loo_um"],
            [
                1.0707,
                0.7173,
                1.3361,
                0.8578,
                41.0328,
                0.9301,
                0.7922,
                0.49,
                0.6404,
                0.9614,
            ],
            1e-3,
        ),
        "appendix A leave-one-out",
    )
    _require(
        abs(default["chosen_class_median_um"] - 0.6680) < 1e-3, "appendix A median"
    )
    _require(
        abs(default["other_class_median_um"] - 8278.383) < 1e-2,
        "appendix A other median",
    )
    _require(
        _close(
            default["first_pass_loo_um"],
            [
                3.8451,
                5.459,
                6.4563,
                4.6403,
                41.0328,
                7.0818,
                3.4237,
                5.1563,
                6.1226,
                5.2289,
            ],
            1e-3,
        ),
        "appendix A first-pass leave-one-out",
    )
    colours = {(e["seed"], e["pair_id"]): e["colour"] for e in data["pair_colour"]}
    _require(
        colours[(4211, 1)] == "#fff400" and colours[(0, 1)] == "#004aff", "pair colours"
    )
    labelling = data["examples"]["labelling"]
    _require(
        labelling["content_sha256"] == LABELLING_CONTENT_SHA256, "labelling digest"
    )
    _require(
        data["examples"]["labelling_content_sha256"] == LABELLING_CONTENT_SHA256,
        "labelling digest",
    )
    _require(
        data["canonical"][0]["sha256"]
        == "sha256:5474a01554484b7b914824bd129578c9ad3b12149779e565293fc0a01b137a53",
        "canonical digest",
    )


def _check_appendix_data() -> None:
    truth = geometry.coarse_matrix_xy(
        schema.CoarseTransform(**APPENDIX_COARSE),
        moving_shape_yx=APPENDIX_MOVING_SHAPE,
        reference_shape_yx=APPENDIX_REFERENCE_SHAPE,
        moving_pixel_size_um=PIXEL_UM,
        reference_pixel_size_um=PIXEL_UM,
    )
    noise = np.array(APPENDIX_NOISE)
    noise[4, 0] += 40.0 / PIXEL_UM
    rebuilt = np.round(
        geometry.apply_matrix_xy(truth, np.array(APPENDIX_MOVING)) + noise, 3
    )
    _require(_close(rebuilt, APPENDIX_REFERENCE, 1e-9), "appendix A reference points")


# ----------------------------------------------------------------------------
# build, render, compare


def build_golden_vectors() -> dict[str, Any]:
    """Build every golden vector with the MerXen implementation.

    Returns:
        The JSON-ready dictionary (numpy data already converted).

    Raises:
        AssertionError: If an implementation result disagrees with a literal of
            the specification (so a wrong implementation cannot write a golden file).
    """
    _check_appendix_data()
    coarse_cases = _coarse_cases()
    data: dict[str, Any] = {
        "schema": "merxen.manual_registration.golden",
        "golden_version": 1,
        "generated_by": GENERATED_BY,
        "tolerances": {
            "matrix_rtol": 1e-9,
            "matrix_atol": 1e-9,
            "residual_atol_um": 1e-6,
            "angle_atol_deg": 1e-9,
        },
        "constants": {name: getattr(schema, name) for name in CONSTANT_NAMES},
        "conventions": dict(schema.CONVENTIONS),
        "wrap_deg": [
            {"input": angle, "output": geometry.wrap_deg(angle)}
            for angle, _ in WRAP_EXAMPLES
        ],
        "coarse_cases": coarse_cases,
        "coarse_transitions": _coarse_transitions(),
        "snap_crop": _snap_crop_cases(),
        "crop_frames": _crop_frames(coarse_cases),
        "points_outside_crop": _points_outside_crop(),
        "napari_affine": _napari_affine(coarse_cases),
        "fits": _fits(coarse_cases),
        "spread_ratio": _spread_ratios(),
        "robust_fits": _robust_fits(),
        "pair_colour": _pair_colours(),
        "canonical": _canonical(),
        "examples": _examples(coarse_cases),
    }
    data = _plain(data)
    _check_appendix_literals(data)
    return data


def render(data: dict[str, Any]) -> str:
    """Render the vectors as the exact text of the golden file."""
    return (
        json.dumps(
            _plain(data), indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False
        )
        + "\n"
    )


def first_difference(actual: Any, expected: Any, path: str = "$") -> str | None:
    """Return the path of the first difference between two JSON structures.

    Floats compare with ``rtol = atol = 1e-9``; strings, bools and ints exactly.
    """
    if isinstance(actual, bool) or isinstance(expected, bool):
        return None if actual is expected else path
    if isinstance(actual, Mapping) and isinstance(expected, Mapping):
        if set(actual) != set(expected):
            return f"{path} (keys {sorted(set(actual) ^ set(expected))})"
        for key in sorted(actual):
            found = first_difference(actual[key], expected[key], f"{path}.{key}")
            if found:
                return found
        return None
    if isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            return f"{path} (length {len(actual)} != {len(expected)})"
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            found = first_difference(left, right, f"{path}[{index}]")
            if found:
                return found
        return None
    if isinstance(actual, int | float) and isinstance(expected, int | float):
        if isinstance(actual, int) and isinstance(expected, int):
            return None if actual == expected else path
        close = math.isclose(actual, expected, rel_tol=FLOAT_RTOL, abs_tol=FLOAT_ATOL)
        return None if close else path
    return None if actual == expected else path


def main(argv: Sequence[str] | None = None) -> int:
    """Write or check the golden file."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare the rebuilt vectors with the file instead of writing it",
    )
    args = parser.parse_args(argv)

    data = build_golden_vectors()
    if args.check:
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        difference = first_difference(json.loads(render(data)), existing)
        if difference is not None:
            print(f"golden file differs at {difference}", file=sys.stderr)
            return 1
        print(f"{args.output} is up to date")
        return 0
    text = render(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = text.encode("utf-8")
    args.output.write_bytes(payload)
    print(hashlib.sha256(payload).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

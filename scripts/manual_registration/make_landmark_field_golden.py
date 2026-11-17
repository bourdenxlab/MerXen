#!/usr/bin/env python3
"""Generate the golden vectors of the robust landmark field.

The output (``tests/test_data/manual_registration/landmark_field_golden_v1.json``)
pins the numbers of ``merxen.manual_registration.landmark_field`` on small
synthetic inputs: a fixed-hyperparameter block (the contract the viewer's later
numerical mirror reproduces), a fitted block, and greedy suggestions. The file is
deterministic for a given numpy/scipy build and its SHA-256 is pinned by the test.

Usage:
    python scripts/manual_registration/make_landmark_field_golden.py \
        [--output PATH] [--check]

Without ``--check`` the file is written and its SHA-256 printed. With ``--check``
the vectors are rebuilt and compared with the file within the tolerances stored in
it (the fit block is compared loosely: see ``tolerances``); the exit status is 1 on
the first difference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from merxen.manual_registration import landmark_field as lf
from merxen.manual_registration.simulate import FieldPhantom, make_field_phantom

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    REPO_ROOT / "tests/test_data/manual_registration/landmark_field_golden_v1.json"
)
GENERATED_BY = "scripts/manual_registration/make_landmark_field_golden.py"
TOLERANCES = {
    "fixed_rtol": 1e-8,
    "fixed_atol": 1e-8,
    "fit_objective_rtol": 1e-6,
    "fit_mean_atol_um": 0.05,
    "fit_loo_atol_um": 0.05,
    "fit_weight_atol": 2e-3,
}
FIXED_SEED = 11
FIT_SEED = 12
QUERY_SEED = 100
N_QUERIES = 12
FIXED_HYPERPARAMETERS = lf.FieldHyperparameters(
    long_length_um=20000.0,
    long_signal_um=80.0,
    short_length_um=1200.0,
    short_signal_um=30.0,
    noise_um={"click": 30.0, "window": 15.0},
)
GROSS_WEIGHT = 0.05


def _clicks(phantom: FieldPhantom) -> lf.FieldObservations:
    shapes = np.array(
        [
            lf.ridge_shape(direction) if np.any(direction) else np.eye(2)
            for direction in phantom.ridge_direction_xy
        ]
    )
    return lf.FieldObservations.clicks(
        phantom.reference_xy_um,
        phantom.moving_xy_um,
        np.arange(1, len(phantom.reference_xy_um) + 1),
        shape_cov=shapes,
    )


def _queries(extent_um: tuple[float, float]) -> np.ndarray:
    rng = np.random.default_rng(QUERY_SEED)
    low = 0.2 * np.array(extent_um)
    high = 0.8 * np.array(extent_um)
    return np.asarray(rng.uniform(low, high, size=(N_QUERIES, 2)))


def _loo_block(loo: lf.LooResult) -> dict[str, Any]:
    return {
        "residual_um": loo.residual_um.tolist(),
        "covariance_um2": loo.covariance_um2.tolist(),
        "latent_covariance_um2": loo.latent_covariance_um2.tolist(),
        "d2": loo.d2.tolist(),
        "p_value": loo.p_value.tolist(),
        "residual_reference_um": loo.residual_reference_um.tolist(),
    }


def _fixed_block() -> dict[str, Any]:
    phantom = make_field_phantom(
        seed=FIXED_SEED, n_clicks=14, n_gross=1, n_medium=1, n_windows=10
    )
    clicks = _clicks(phantom)
    windows = lf.FieldObservations.windows(
        phantom.window_reference_xy_um,
        phantom.window_moving_xy_um,
        phantom.window_peak_cov_um2,
        first_id=1000,
    )
    observations = clicks.concat(windows)
    weights = np.ones(observations.n)
    gross = [i for i, kind in enumerate(phantom.kind) if kind == "gross"]
    weights[gross] = GROSS_WEIGHT
    field = lf.landmark_field_from_hyperparameters(
        observations, FIXED_HYPERPARAMETERS, weights
    )
    queries = _queries((22000.0, 16000.0))
    extra_reference = np.array([[9000.0, 7000.0], [13000.0, 9500.0]])
    extra_moving = extra_reference + np.array([[120.0, -60.0], [-40.0, 90.0]])
    ridge = lf.ridge_shape([1.0, 2.0])
    isotropic = np.eye(2)
    extra_shapes = np.stack([isotropic, ridge])
    residual = field.predict_residual(
        extra_reference, extra_moving, obs_class="click", shape_cov=extra_shapes
    )
    return {
        "observations": {
            "reference_xy_um": observations.reference_xy_um.tolist(),
            "moving_xy_um": observations.moving_xy_um.tolist(),
            "shape_cov": observations.shape_cov.tolist(),
            "fixed_cov_um2": observations.fixed_cov_um2.tolist(),
            "class": list(observations.obs_class),
            "id": observations.obs_id.tolist(),
            "weight": weights.tolist(),
        },
        "hyperparameters": {
            "long_length_um": FIXED_HYPERPARAMETERS.long_length_um,
            "long_signal_um": FIXED_HYPERPARAMETERS.long_signal_um,
            "short_length_um": FIXED_HYPERPARAMETERS.short_length_um,
            "short_signal_um": FIXED_HYPERPARAMETERS.short_signal_um,
            "noise_um": dict(FIXED_HYPERPARAMETERS.noise_um),
        },
        "outputs": {
            "trend_coefficients": field.trend_coefficients.tolist(),
            "trend_matrix": field.trend_matrix().tolist(),
            "objective": field.objective(),
            "query_reference_xy_um": queries.tolist(),
            "mean_um": field.mean(queries).tolist(),
            "latent_covariance_um2": field.covariance(queries).tolist(),
            "loo": _loo_block(field.loo()),
            "extra_points": {
                "reference_xy_um": extra_reference.tolist(),
                "moving_xy_um": extra_moving.tolist(),
                "shape_cov": extra_shapes.tolist(),
                "residual_um": residual.residual_um.tolist(),
                "covariance_um2": residual.covariance_um2.tolist(),
                "d2": residual.d2.tolist(),
                "p_value": residual.p_value.tolist(),
            },
            "payload": field.to_dict(),
            "digest": field.digest,
        },
    }


def _fit_block() -> dict[str, Any]:
    phantom = make_field_phantom(
        seed=FIT_SEED,
        n_clicks=30,
        short_signal_um=0.0,
        ridge_fraction=0.0,
        n_medium=0,
        n_gross=1,
    )
    observations = lf.FieldObservations.clicks(
        phantom.reference_xy_um,
        phantom.moving_xy_um,
        np.arange(1, 31),
    )
    field = lf.fit_landmark_field(observations)
    queries = _queries((22000.0, 16000.0))
    hyper = field.hyperparameters
    return {
        "observations": {
            "reference_xy_um": observations.reference_xy_um.tolist(),
            "moving_xy_um": observations.moving_xy_um.tolist(),
            "id": observations.obs_id.tolist(),
        },
        "outputs": {
            "objective": field.objective(),
            "query_reference_xy_um": queries.tolist(),
            "mean_um": field.mean(queries).tolist(),
            "loo_residual_um": field.loo().residual_um.tolist(),
            "weights": field.weights.tolist(),
        },
        "informative": {
            "hyperparameters": {
                "long_length_um": hyper.long_length_um,
                "long_signal_um": hyper.long_signal_um,
                "noise_um": dict(hyper.noise_um),
            },
            "em_iterations": field.em_iterations,
        },
    }


def _suggestions_block() -> dict[str, Any]:
    fixed = _fixed_block()
    stored = fixed["observations"]
    observations = lf.FieldObservations(
        reference_xy_um=np.array(stored["reference_xy_um"]),
        moving_xy_um=np.array(stored["moving_xy_um"]),
        shape_cov=np.array(stored["shape_cov"]),
        fixed_cov_um2=np.array(stored["fixed_cov_um2"]),
        obs_class=tuple(stored["class"]),
        obs_id=np.array(stored["id"]),
    )
    field = lf.landmark_field_from_hyperparameters(
        observations, FIXED_HYPERPARAMETERS, np.array(stored["weight"])
    )
    rng = np.random.default_rng(QUERY_SEED + 1)
    candidates = rng.uniform([2000.0, 1500.0], [20000.0, 14500.0], size=(40, 2))
    candidate_weights = rng.choice([0.0, 0.5, 1.0], size=40, p=[0.15, 0.35, 0.5])
    suggestions = lf.suggest_landmarks(
        field,
        candidates,
        candidate_weights,
        n=4,
        min_spacing_um=1500.0,
        exclusion_radius_um=500.0,
    )
    return {
        "candidates_reference_um": candidates.tolist(),
        "candidate_weights": candidate_weights.tolist(),
        "n": 4,
        "min_spacing_um": 1500.0,
        "exclusion_radius_um": 500.0,
        "suggestions": [
            {
                "rank": s.rank,
                "reference_xy_um": s.reference_xy_um.tolist(),
                "predicted_moving_xy_um": s.predicted_moving_xy_um.tolist(),
                "sd_um": s.sd_um,
                "covariance_um2": s.covariance_um2.tolist(),
                "weight": s.weight,
                "score": s.score,
            }
            for s in suggestions
        ],
    }


def build_golden_vectors() -> dict[str, Any]:
    """Compute every golden vector from its fixed inputs."""
    return {
        "generated_by": GENERATED_BY,
        "schema": lf.FIELD_SCHEMA,
        "schema_version": lf.FIELD_SCHEMA_VERSION,
        "algorithm": lf.FIELD_ALGORITHM,
        "constants": {
            "TREND_SCALE_UM": lf.TREND_SCALE_UM,
            "JITTER_UM2": lf.JITTER_UM2,
            "MAX_SHAPE_ANISOTROPY": lf.MAX_SHAPE_ANISOTROPY,
            "RIDGE_SHAPE_RATIO": lf.RIDGE_SHAPE_RATIO,
        },
        "tolerances": dict(TOLERANCES),
        "fixed": _fixed_block(),
        "fit": _fit_block(),
        "suggestions": _suggestions_block(),
    }


def render(data: dict[str, Any]) -> str:
    """Canonical text of the golden file (sorted keys, trailing newline)."""
    return (
        json.dumps(data, sort_keys=True, indent=1, ensure_ascii=True, allow_nan=False)
        + "\n"
    )


def _tolerance_for(path: str) -> tuple[float, float]:
    """``(rtol, atol)`` of a leaf, by its path in the golden document."""
    if path.startswith("$.fit.informative"):
        return math.inf, math.inf
    if path.startswith("$.fit.outputs.objective"):
        return TOLERANCES["fit_objective_rtol"], 0.0
    if path.startswith("$.fit.outputs.mean_um"):
        return 0.0, TOLERANCES["fit_mean_atol_um"]
    if path.startswith("$.fit.outputs.loo_residual_um"):
        return 0.0, TOLERANCES["fit_loo_atol_um"]
    if path.startswith("$.fit.outputs.weights"):
        return 0.0, TOLERANCES["fit_weight_atol"]
    return TOLERANCES["fixed_rtol"], TOLERANCES["fixed_atol"]


def first_difference(actual: Any, expected: Any, path: str = "$") -> str | None:
    """Return the path of the first leaf that differs beyond its tolerance."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            return path
        for key in sorted(expected):
            found = first_difference(actual[key], expected[key], f"{path}.{key}")
            if found is not None:
                return found
        return None
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return path
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            found = first_difference(left, right, f"{path}[{index}]")
            if found is not None:
                return found
        return None
    if isinstance(expected, bool | str) or expected is None:
        return None if actual == expected else path
    rtol, atol = _tolerance_for(path)
    if math.isinf(rtol):
        return None
    return (
        None
        if math.isclose(float(actual), float(expected), rel_tol=rtol, abs_tol=atol)
        else path
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Write or check the golden file."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
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
    payload = render(data).encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    print(hashlib.sha256(payload).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

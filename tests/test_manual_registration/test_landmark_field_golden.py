"""Golden vectors of the robust landmark field (the contract of a later mirror).

The viewer's numerical mirror (milestone M4) copies
``landmark_field_golden_v1.json`` byte-for-byte and pins the same SHA-256; after any
regeneration the new digest must be announced to the viewer work.

The digests inside the file (``digest`` leaves) hash computed floats whose last bits
follow the CPU's BLAS kernel. The mirror therefore verifies the digest of the stored
payload (``canonical_sha256`` of the payload without it) and compares recomputed
numbers within tolerance; it never recomputes a digest bit for bit.

The trend centre is the mean of the click-like observations (classes ``click`` and
``confirmed``), not of ``click`` alone; no golden case holds a confirmed observation, so
the mirror must read ``trend.centre_um`` from the payload rather than recompute it.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from merxen.manual_registration import landmark_field as lf
from merxen.manual_registration import schema

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    REPO_ROOT / "scripts" / "manual_registration" / "make_landmark_field_golden.py"
)
GOLDEN_PATH = (
    REPO_ROOT
    / "tests"
    / "test_data"
    / "manual_registration"
    / "landmark_field_golden_v1.json"
)
GOLDEN_SHA256 = "03eaa3815278ef8dd974753a4949fcc5152c49a94abcbc933c80648b39942ac4"


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "make_landmark_field_golden_under_test", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def golden() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    return data


@pytest.fixture(scope="module")
def rebuilt(script: ModuleType) -> dict[str, Any]:
    return json.loads(script.render(script.build_golden_vectors()))


def _arr(values: Any) -> np.ndarray:
    return np.array(values, dtype=float)


def test_golden_file_bytes_are_pinned() -> None:
    assert hashlib.sha256(GOLDEN_PATH.read_bytes()).hexdigest() == GOLDEN_SHA256


def test_golden_file_is_canonical_json_under_the_size_limit(
    golden: dict[str, Any],
) -> None:
    text = GOLDEN_PATH.read_text(encoding="utf-8")
    assert (
        text
        == json.dumps(
            golden, sort_keys=True, indent=1, ensure_ascii=True, allow_nan=False
        )
        + "\n"
    )
    assert len(text) < 500_000
    assert golden["schema"] == lf.FIELD_SCHEMA
    assert golden["algorithm"] == lf.FIELD_ALGORITHM


def test_golden_constants_match_the_module(golden: dict[str, Any]) -> None:
    assert golden["constants"] == {
        "TREND_SCALE_UM": lf.TREND_SCALE_UM,
        "JITTER_UM2": lf.JITTER_UM2,
        "MAX_SHAPE_ANISOTROPY": lf.MAX_SHAPE_ANISOTROPY,
        "RIDGE_SHAPE_RATIO": lf.RIDGE_SHAPE_RATIO,
    }


def test_rebuilt_vectors_match_the_file_within_tolerances(
    script: ModuleType, golden: dict[str, Any], rebuilt: dict[str, Any]
) -> None:
    assert script.first_difference(rebuilt, golden) is None


def test_fixed_block_reproduces_from_its_stored_inputs(
    script: ModuleType, golden: dict[str, Any]
) -> None:
    tolerances = golden["tolerances"]
    rtol, atol = tolerances["fixed_rtol"], tolerances["fixed_atol"]
    block = golden["fixed"]
    stored = block["observations"]
    observations = lf.FieldObservations(
        reference_xy_um=_arr(stored["reference_xy_um"]),
        moving_xy_um=_arr(stored["moving_xy_um"]),
        shape_cov=_arr(stored["shape_cov"]),
        fixed_cov_um2=_arr(stored["fixed_cov_um2"]),
        obs_class=tuple(stored["class"]),
        obs_id=np.array(stored["id"]),
    )
    hyper = lf.FieldHyperparameters(**block["hyperparameters"])
    field = lf.landmark_field_from_hyperparameters(
        observations, hyper, _arr(stored["weight"])
    )
    out = block["outputs"]
    queries = _arr(out["query_reference_xy_um"])
    np.testing.assert_allclose(
        field.trend_coefficients, _arr(out["trend_coefficients"]), rtol=rtol, atol=atol
    )
    assert field.objective() == pytest.approx(out["objective"], rel=rtol, abs=atol)
    np.testing.assert_allclose(
        field.mean(queries), _arr(out["mean_um"]), rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        field.covariance(queries),
        _arr(out["latent_covariance_um2"]),
        rtol=rtol,
        atol=atol,
    )
    loo = field.loo()
    np.testing.assert_allclose(
        loo.residual_um, _arr(out["loo"]["residual_um"]), rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        loo.covariance_um2, _arr(out["loo"]["covariance_um2"]), rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(loo.d2, _arr(out["loo"]["d2"]), rtol=rtol, atol=atol)
    np.testing.assert_allclose(
        loo.p_value, _arr(out["loo"]["p_value"]), rtol=rtol, atol=atol
    )
    extra = out["extra_points"]
    residual = field.predict_residual(
        _arr(extra["reference_xy_um"]),
        _arr(extra["moving_xy_um"]),
        shape_cov=_arr(extra["shape_cov"]),
    )
    np.testing.assert_allclose(
        residual.residual_um, _arr(extra["residual_um"]), rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(residual.d2, _arr(extra["d2"]), rtol=rtol, atol=atol)
    # The digest hashes BLAS-computed floats, so it is not compared bit for bit
    # across CPUs: the rebuilt payload matches the stored one within tolerance
    # and the stored digest is checked against the stored payload.
    stored_payload = out["payload"]
    rebuilt_payload = field.to_dict()
    assert rebuilt_payload["digest"] == schema.canonical_sha256(
        {key: value for key, value in rebuilt_payload.items() if key != "digest"}
    )
    assert (
        script.first_difference(json.loads(json.dumps(rebuilt_payload)), stored_payload)
        is None
    )
    assert out["payload"]["digest"] == out["digest"]
    assert lf.LandmarkField.from_dict(stored_payload).digest == out["digest"]


def test_fixed_payload_is_a_valid_c1_document(golden: dict[str, Any]) -> None:
    payload = golden["fixed"]["outputs"]["payload"]
    body = {key: value for key, value in payload.items() if key != "digest"}
    assert payload["digest"] == schema.canonical_sha256(body)
    rebuilt = lf.LandmarkField.from_dict(payload)
    assert rebuilt.digest == payload["digest"]
    assert rebuilt.components == 2
    # The gross click carries the low weight stored with the observations.
    assert min(payload["observations"]["weight"]) == pytest.approx(0.05)


def test_fit_block_reproduces_within_its_tolerances(golden: dict[str, Any]) -> None:
    tolerances = golden["tolerances"]
    block = golden["fit"]
    stored = block["observations"]
    observations = lf.FieldObservations.clicks(
        _arr(stored["reference_xy_um"]),
        _arr(stored["moving_xy_um"]),
        np.array(stored["id"]),
    )
    field = lf.fit_landmark_field(observations)
    out = block["outputs"]
    assert field.components == 1
    assert field.objective() == pytest.approx(
        out["objective"], rel=tolerances["fit_objective_rtol"]
    )
    np.testing.assert_allclose(
        field.mean(_arr(out["query_reference_xy_um"])),
        _arr(out["mean_um"]),
        rtol=0.0,
        atol=tolerances["fit_mean_atol_um"],
    )
    np.testing.assert_allclose(
        field.loo().residual_um,
        _arr(out["loo_residual_um"]),
        rtol=0.0,
        atol=tolerances["fit_loo_atol_um"],
    )
    np.testing.assert_allclose(
        field.weights,
        _arr(out["weights"]),
        rtol=0.0,
        atol=tolerances["fit_weight_atol"],
    )
    assert math.isfinite(block["informative"]["hyperparameters"]["long_length_um"])


def test_suggestions_reproduce_from_the_fixed_model(golden: dict[str, Any]) -> None:
    rtol = golden["tolerances"]["fixed_rtol"]
    atol = golden["tolerances"]["fixed_atol"]
    stored = golden["fixed"]["observations"]
    observations = lf.FieldObservations(
        reference_xy_um=_arr(stored["reference_xy_um"]),
        moving_xy_um=_arr(stored["moving_xy_um"]),
        shape_cov=_arr(stored["shape_cov"]),
        fixed_cov_um2=_arr(stored["fixed_cov_um2"]),
        obs_class=tuple(stored["class"]),
        obs_id=np.array(stored["id"]),
    )
    field = lf.landmark_field_from_hyperparameters(
        observations,
        lf.FieldHyperparameters(**golden["fixed"]["hyperparameters"]),
        _arr(stored["weight"]),
    )
    block = golden["suggestions"]
    picks = lf.suggest_landmarks(
        field,
        _arr(block["candidates_reference_um"]),
        _arr(block["candidate_weights"]),
        n=block["n"],
        min_spacing_um=block["min_spacing_um"],
        exclusion_radius_um=block["exclusion_radius_um"],
    )
    assert len(picks) == len(block["suggestions"]) == 4
    for pick, expected in zip(picks, block["suggestions"], strict=True):
        assert pick.rank == expected["rank"]
        np.testing.assert_allclose(
            pick.reference_xy_um,
            _arr(expected["reference_xy_um"]),
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            pick.covariance_um2, _arr(expected["covariance_um2"]), rtol=rtol, atol=atol
        )
        assert pick.sd_um == pytest.approx(expected["sd_um"], rel=rtol, abs=atol)
        assert pick.score == pytest.approx(expected["score"], rel=rtol, abs=atol)

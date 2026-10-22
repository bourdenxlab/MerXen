"""Tests of the fixed parameter set of the psf_subtract bead-removal method."""

from __future__ import annotations

import dataclasses
import json

import pytest

from merxen.io.bead_removal.params import (
    BEAD_REMOVAL_VERSION,
    BETA,
    DEFAULT_PARAMS,
    ENVELOPE_BLOCK_PX,
    FLUX_TARGET,
    NZ_MAX,
    PsfSubtractParams,
    params_sha256,
)


def test_params_hash_is_stable_and_sensitive() -> None:
    same = dataclasses.replace(DEFAULT_PARAMS)
    assert params_sha256(DEFAULT_PARAMS) == params_sha256(same)
    other = dataclasses.replace(DEFAULT_PARAMS, k_sigma=6.0)
    assert params_sha256(DEFAULT_PARAMS) != params_sha256(other)
    # A tuple-valued parameter is part of the hash too.
    longer = dataclasses.replace(DEFAULT_PARAMS, sel_pct=(10.0, 92.0))
    assert params_sha256(DEFAULT_PARAMS) != params_sha256(longer)
    assert len(params_sha256(DEFAULT_PARAMS)) == 64


def test_params_are_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        DEFAULT_PARAMS.k_sigma = 1.0  # type: ignore[misc]


def test_params_dump_is_json_serialisable() -> None:
    # The provenance stores the dataclass; an infinite sat_level must survive.
    dumped = json.dumps(dataclasses.asdict(DEFAULT_PARAMS))
    assert json.loads(dumped)["k_sigma"] == DEFAULT_PARAMS.k_sigma


def test_per_plane_constants_cover_every_plane() -> None:
    assert NZ_MAX == 8
    for values in (
        FLUX_TARGET,
        BETA,
        DEFAULT_PARAMS.r_sup,
        DEFAULT_PARAMS.smooth_px,
        DEFAULT_PARAMS.g_prior_sd,
    ):
        assert len(values) == NZ_MAX
    # Support radii grow with the defocus radius.
    assert list(DEFAULT_PARAMS.r_sup) == sorted(DEFAULT_PARAMS.r_sup)


def test_version_and_grid_constants() -> None:
    assert BEAD_REMOVAL_VERSION == 1
    # The envelope grid is absolute: window origins are multiples of this.
    assert ENVELOPE_BLOCK_PX == 16
    assert DEFAULT_PARAMS.stretch_block_px % ENVELOPE_BLOCK_PX == 0
    assert isinstance(DEFAULT_PARAMS, PsfSubtractParams)

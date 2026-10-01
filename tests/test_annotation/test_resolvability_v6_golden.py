"""Golden hashes of the resolvability version-6 path (pre-registration §21 (i-a)).

Milestone M3c adds resolvability version 7 for the families outside
``validated_panels.csv`` and keeps version 6 byte-identical for the families
whose decisions are pre-registered for gates H and M (plan §8.3 v7.1). This
test runs the whole version-6 path on two synthetic panels (a human-like and a
mouse-like one; the synthetic reference cells and the deterministic bootstrap
mapper of ``test_resolvability.py``): test cells, ``thin_and_contaminate`` for
``R1_contam_HO`` and ``clean``, mapping, ``level_cells``, ``decide``, floors,
trust constraint, ``resolvability_table`` and ``build_summary``, plus the
``build_hash`` of a version-6 bundle spec for every seeded panel of
``validated_panels.csv``. It hashes

1. each recipe's simulated query: the CSR ``data`` (float64), ``indices``
   (int32) and ``indptr`` (int64) as little-endian bytes and the obs index
   (and, tighter than pre-registered, the obs table);
2. the decisions of every regime as canonical CSV (rows sorted by regime,
   level, class and depth; columns in ``decide``'s order; floats ``%.17g``;
   missing values empty);
3. the ``resolvability.parquet`` table as canonical CSV (rows sorted by every
   column) and, tighter than pre-registered, the stored cells table;
4. the summary JSON without ``runtime_s``, keys sorted;
5. ``compute_build_hash`` of the version-6 bundle spec of each seeded panel
   hash, for every reference built on it (primary, secondary and test set).

The golden values were computed with the code of ``83e81e3`` (a ``git
archive`` export) before any M3c code change and must hold unchanged at every
later M3c commit. Any difference fails M3c.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from merxen.annotation import resolvability as res
from merxen.annotation.config import (
    KNOWN_REFERENCES,
    AnnotationConfig,
    AnnotationReferenceSpec,
    AnnotationResolvabilityConfig,
)
from merxen.annotation.vocab import load_floor_table

from .test_resolvability import bootstrap_mapper, make_test_cells, synthetic_specs

# Computed with the code of 83e81e3 (a git archive export), 2026-09-28.
GOLDEN: dict[str, str] = {
    "build_hash/human_set_a_296/seaad_mr_panel": (
        "054bf5e33bfbbf2863d6d47a9b8915a92c8156b27e16c9f549da1025bea12e4c"
    ),
    "build_hash/human_set_a_296/whb_frontal_supc_clus": (
        "aa98b2ff04cec0c7d7b22cd375fbc95e4c6592c1606b4c29ccc2010600c09842"
    ),
    "build_hash/human_set_a_296/whb_frontal_supc_clus_ho": (
        "4638e344e567541fd0d864765901a6ac91ea9944f08b750e29417578f8cc6d34"
    ),
    "build_hash/human_set_a_297/seaad_mr_panel": (
        "5c4141ded9f2f5be82c91f60f0ed8b4eb56429ba824e8013e84abd85e48d8efc"
    ),
    "build_hash/human_set_a_297/whb_frontal_supc_clus": (
        "9d6dc74dec83afce9e148477da16e27bf84ca0bd8d0ed226481e908a6a3be8fd"
    ),
    "build_hash/human_set_a_297/whb_frontal_supc_clus_ho": (
        "0aa7ff3d658b474b7530e88220f1ea7c358de5e114e18fe014d45c5017d106a6"
    ),
    "build_hash/human_set_c_264/seaad_mr_panel": (
        "d513dd08b7b98c5007278c2ca67b2eb0695b46c25797a115d6339076ec171931"
    ),
    "build_hash/human_set_c_264/whb_frontal_supc_clus": (
        "92ce72d35b5ddfaba16dfe1a1eb3ef48b1dfb9952ecdc18fcdb7f7e39ded421e"
    ),
    "build_hash/human_set_c_264/whb_frontal_supc_clus_ho": (
        "075181f249c0231d9e567f8584be88b20569b1fae33891a6161906d738bfe8ea"
    ),
    "build_hash/human_set_c_265/seaad_mr_panel": (
        "286e741af46d4c905d8bc2998d122d2d7e0ee54a129a10d27e70f0e01fd6e171"
    ),
    "build_hash/human_set_c_265/whb_frontal_supc_clus": (
        "757f22818c02c6db8dd9e5c476a5bc3c66289cf67c7f1bec764a061a27ae913d"
    ),
    "build_hash/human_set_c_265/whb_frontal_supc_clus_ho": (
        "6d323487ff0c49cc7cbd78dcaf118c11564b0735107b7b3a307e2662f5aba711"
    ),
    "build_hash/mouse_ag7_500/wmb_panel": (
        "47edbff9e8494a8b51a536bbd1c53cafff3a9e395b0a62caa9423d699f18dee8"
    ),
    "build_hash/mouse_ag7_500/wmb_selfmap_testset": (
        "fc46d2f11d7741b736882627b158f52222b07592029464e412f15c650c2a1895"
    ),
    "build_hash/mouse_vzg2_815/wmb_panel": (
        "88c769cf7b8cf243d148398c279fd62ed292392e1d4f3870112650cc3660bf05"
    ),
    "build_hash/mouse_vzg2_815/wmb_selfmap_testset": (
        "f4cbe68e77bd2c0ea659603e53f91a50b8ee8c8eedc04b1e92607e57158ea701"
    ),
    "human_like/bundle_output_recipe": (
        "3e176cf636d1bad5fac5573e4d8302f52acb7b139cc3dedcd9a8d2a4493bb3c6"
    ),
    "human_like/cells": (
        "761067f401a9e681a8c5b0a0b424243545f69098b06dfa9b9393aed263870761"
    ),
    "human_like/decisions": (
        "6f5e13a4f27f02d2fc143d951850b70a924524fb9bf1d55ffa8ca3e5abd8e04a"
    ),
    "human_like/floors": (
        "8a2790274efefda800fdc28594fe50f73859cf5dd766f0016322eabff16bcdb2"
    ),
    "human_like/query/R1_contam_HO/csr": (
        "fdc2bf96ab4b5b8da3f98c3a9df3506232e62756e1bfb387415663bcb25a6acc"
    ),
    "human_like/query/R1_contam_HO/index": (
        "7685185f46c28e4f7a2b7361a53970f5ea31cb12c1da971a12fb32e1eaa8655a"
    ),
    "human_like/query/R1_contam_HO/obs": (
        "484a0acf050f8d958e7a07fa52b5c2ddb2641e8213b89c3cd4956adf0359a02a"
    ),
    "human_like/query/clean/csr": (
        "dc06b0df95cf04a1a8f9343b2cd535f3c9850697365d4551982b0befefc56275"
    ),
    "human_like/query/clean/index": (
        "7685185f46c28e4f7a2b7361a53970f5ea31cb12c1da971a12fb32e1eaa8655a"
    ),
    "human_like/query/clean/obs": (
        "faff9907605eca6e2f992091171242fede5d53c78a5ba3430bad6fa0cae92d3b"
    ),
    "human_like/summary": (
        "8efdd5b187736ebf3c59b3939a7a5a73cde9e0239e4c326ceaa8f5f9e6f9473a"
    ),
    "human_like/table": (
        "0edf475d90ef350d0016d83193d7ab65c7b02dd7c203f1d4109f446c9d3d0975"
    ),
    "human_like/trust": (
        "fa4b38cee47dbef83659fd1841d5bed4ccd0f3f87ffa68e8ff2639befbab4afd"
    ),
    "mouse_like/bundle_output_recipe": (
        "3e176cf636d1bad5fac5573e4d8302f52acb7b139cc3dedcd9a8d2a4493bb3c6"
    ),
    "mouse_like/cells": (
        "09dd47fa695f7db59915f6b1688438615932f291ad1f09b4aa054ad6dd26d9a8"
    ),
    "mouse_like/decisions": (
        "487966c912dcece3d0b5337fd585641d526b631d946ea63516227f858e4fa027"
    ),
    "mouse_like/floors": (
        "a1b04f68929b635ed821aefc28570ce0ac3b040460f9ad1fe0a1f4737c39043e"
    ),
    "mouse_like/query/R1_contam_HO/csr": (
        "92dd8c30b37c9cc6f38f00aec0b81e09f3962c13259b6bb1752eb53d567e2804"
    ),
    "mouse_like/query/R1_contam_HO/index": (
        "ab67cb2a3bf10b07c970a01af7912681bbe0e5558b37dce106921b27060501a1"
    ),
    "mouse_like/query/R1_contam_HO/obs": (
        "18d20d9dec9f0acb493db99c64851f71e39b7ec088dcfe3e35cee8f5e0801212"
    ),
    "mouse_like/query/clean/csr": (
        "fb985526912dfc78b1aea16f77ccdf32476c2448aee71e05b1d19b8c66371ad1"
    ),
    "mouse_like/query/clean/index": (
        "ab67cb2a3bf10b07c970a01af7912681bbe0e5558b37dce106921b27060501a1"
    ),
    "mouse_like/query/clean/obs": (
        "6e7105819b8759d3e11120f9d86faf4295a0e8f66a22ff980da54ad24f18914e"
    ),
    "mouse_like/summary": (
        "eb90046020132a76f0ad12aecb595d1440292757d69cc4074f69318c64d46d67"
    ),
    "mouse_like/table": (
        "c3182b278eeed2ddeeead2a55fe640200ab46addb195004b1275ad268787a0ac"
    ),
    "mouse_like/trust": (
        "2477a4363fea6cf8597c179efe62f6f49d8b58c89bd6af78d32d65500efca1f2"
    ),
}

# Synthetic panels: (species, depth grid, cells per cluster, cell seed).
PANELS: dict[str, tuple[str, tuple[int, ...], int, int]] = {
    "human_like": ("human", (10, 30, 100), 60, 0),
    "mouse_like": ("mouse", (10, 20, 50, 100, 250), 60, 1),
}
# References built on a seeded panel of each species (v6 builder params).
SEEDED_REFERENCES: dict[str, tuple[str, ...]] = {
    "human": ("whb_frontal_supc_clus", "seaad_mr_panel", "whb_frontal_supc_clus_ho"),
    "mouse": ("wmb_panel", "wmb_selfmap_testset"),
}
VALIDATED_PANELS: Path = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "merxen"
    / "assets"
    / "annotation"
    / "validated_panels.csv"
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _cell(value: Any) -> str:
    """Return the canonical text of one table value."""
    if value is None or value is pd.NA:
        return ""
    if isinstance(value, bool | np.bool_):
        return "True" if bool(value) else "False"
    if isinstance(value, int | np.integer):
        return str(int(value))
    if isinstance(value, float | np.floating):
        number = float(value)
        return "" if math.isnan(number) else f"{number:.17g}"
    if isinstance(value, list | tuple | dict | np.ndarray):
        items = value.tolist() if isinstance(value, np.ndarray) else value
        return json.dumps(items, sort_keys=True, default=str)
    return str(value)


def canonical_csv(frame: pd.DataFrame, sort_by: Sequence[str] | None = None) -> str:
    """Return a frame as canonical CSV text (column order kept).

    Args:
        frame: The table.
        sort_by: Columns to sort by (their values, stable); ``None`` sorts the
            rows by the canonical text of every column.

    Returns:
        Header and rows, comma-separated, one row per line.
    """
    table = frame.reset_index(drop=True)
    if sort_by is not None:
        table = table.sort_values(list(sort_by), kind="mergesort").reset_index(
            drop=True
        )
    text = [[_cell(value) for value in row] for row in table.itertuples(index=False)]
    if sort_by is None:
        text.sort()
    lines = [",".join(str(column) for column in table.columns)]
    lines += [",".join(row) for row in text]
    return "\n".join(lines) + "\n"


def query_hashes(query: res.SimulatedQuery) -> dict[str, str]:
    """Return the sha256 of a simulated query's CSR arrays, index and obs."""
    matrix = query.counts.tocsr()
    payload = (
        np.ascontiguousarray(matrix.data, dtype="<f8").tobytes()
        + np.ascontiguousarray(matrix.indices, dtype="<i4").tobytes()
        + np.ascontiguousarray(matrix.indptr, dtype="<i8").tobytes()
    )
    index = "\n".join(str(value) for value in query.obs.index).encode("utf-8")
    return {
        "csr": _sha256(payload),
        "index": _sha256(index),
        "obs": _sha256(canonical_csv(query.obs.reset_index()).encode("utf-8")),
    }


def _fake_source(name: str) -> Any:
    from merxen.annotation.store import FileIdentity, SourceRecord

    path = f"/golden/{name}"
    return SourceRecord(
        name=name,
        path=path,
        kind="file",
        files=(FileIdentity(path=path, size=1, mtime_ns=0, sha256_head_tail="0" * 64),),
    )


def build_hashes() -> dict[str, str]:
    """Return ``compute_build_hash`` of each seeded panel's version-6 bundles."""
    from merxen.annotation.reference import builder_for
    from merxen.annotation.store import build_hash_payload, compute_build_hash

    rows = pd.read_csv(VALIDATED_PANELS)
    result: dict[str, str] = {}
    ctm = {"version": "1.7.2", "commit": None}
    for row in rows.itertuples(index=False):
        species = str(row.species)
        config = AnnotationConfig(species=cast("Any", species))
        panel = SimpleNamespace(
            panel_hash=str(row.panel_hash), n_genes=int(row.n_genes), species=species
        )
        for reference_id in SEEDED_REFERENCES[species]:
            spec = AnnotationReferenceSpec(
                reference_id=reference_id, **KNOWN_REFERENCES[reference_id]
            )
            builder = builder_for(spec, config)
            sources = {name: _fake_source(name) for name in ("a", "b")}
            payload = build_hash_payload(
                spec,
                cast("Any", panel),
                builder=builder,
                sources=sources,
                config=config,
                ctm=ctm,
            )
            result[f"build_hash/{row.panel_id}/{reference_id}"] = compute_build_hash(
                payload
            )
    return result


def run_panel(name: str) -> dict[str, str]:
    """Run the version-6 path on one synthetic panel and hash its products."""
    species, grid, n_per_cluster, seed = PANELS[name]
    test = make_test_cells(n_per_cluster, seed=seed)
    recipes = res.simulation_recipes(AnnotationResolvabilityConfig(), seed=0)
    result: dict[str, str] = {}
    for recipe in recipes:
        query = res.thin_and_contaminate(test, grid, recipe)
        for key, value in query_hashes(query).items():
            result[f"{name}/query/{recipe.name}/{key}"] = value
    run = res.run_resolvability(
        test,
        specs=synthetic_specs(),
        depths=grid,
        recipes=recipes,
        map_fn=bootstrap_mapper,
        settings=res.RuleSettings(),
        species=species,
        floor_table=load_floor_table(cast("Any", species)),
    )
    result[f"{name}/decisions"] = _sha256(
        canonical_csv(
            run.decisions, sort_by=("regime", "level", "class", "depth")
        ).encode("utf-8")
    )
    result[f"{name}/table"] = _sha256(canonical_csv(run.table).encode("utf-8"))
    result[f"{name}/cells"] = _sha256(
        canonical_csv(res.coerce_cells(run.cells)).encode("utf-8")
    )
    result[f"{name}/floors"] = _sha256(canonical_csv(run.floors).encode("utf-8"))
    result[f"{name}/trust"] = _sha256(
        json.dumps(run.trust.to_json(), sort_keys=True).encode("utf-8")
    )
    summary = {key: value for key, value in run.summary.items() if key != "runtime_s"}
    result[f"{name}/summary"] = _sha256(
        json.dumps(res._json_native(summary), sort_keys=True, allow_nan=False).encode(
            "utf-8"
        )
    )
    result[f"{name}/bundle_output_recipe"] = _sha256(
        json.dumps(
            res._json_native(run.bundle_output()["recipe"]), sort_keys=True
        ).encode("utf-8")
    )
    return result


def v6_hashes(
    panels: Sequence[str] = tuple(PANELS),
    extra: Callable[[], Mapping[str, str]] | None = build_hashes,
) -> dict[str, str]:
    """Return every golden hash (panel runs, then the build hashes)."""
    result: dict[str, str] = {}
    for name in panels:
        result.update(run_panel(name))
    if extra is not None:
        result.update(extra())
    return dict(sorted(result.items()))


@pytest.fixture(scope="module")
def current_hashes() -> dict[str, str]:
    return v6_hashes()


def test_golden_hashes_are_complete() -> None:
    assert GOLDEN, "golden hashes missing"
    names = set(GOLDEN)
    for panel in PANELS:
        assert f"{panel}/decisions" in names
        assert f"{panel}/query/R1_contam_HO/csr" in names
        assert f"{panel}/query/clean/csr" in names
    assert sum(1 for name in names if name.startswith("build_hash/")) == 16


def test_version_6_path_is_byte_identical(current_hashes: dict[str, str]) -> None:
    differ = {
        name: (GOLDEN.get(name), value)
        for name, value in current_hashes.items()
        if GOLDEN.get(name) != value
    }
    missing = sorted(set(GOLDEN) - set(current_hashes))
    assert not differ and not missing, (
        f"version-6 outputs changed (pre-registration §21 (i) fails): {differ}; "
        f"missing {missing}"
    )

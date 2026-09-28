"""The version-7 diagnostic of a version-6 family (plan §8.3 v7.1; M3c (10)).

``annotation-panel-simulate --resolvability-version 7`` computes the
version-7 decisions and the fresh-draw churn beside a version-6 bundle: it
writes only to the output directory and never changes the bundle or its
emitted decisions.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from merxen.annotation import resolvability as res
from merxen.annotation import simulate
from merxen.annotation.simulate import (
    ReferenceBuild,
    SimulationError,
    run_panel_simulation,
)

from .test_simulate import _large_whb_simulation, small_resources  # noqa: F401


def bundle_digest(bundle_dir: Path) -> dict[str, str]:
    return {
        str(path.relative_to(bundle_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(bundle_dir.rglob("*"))
        if path.is_file()
    }


def test_the_version_7_diagnostic_never_changes_a_version_6_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    small_resources: Any,  # noqa: F811
) -> None:
    from merxen.annotation.reference import builder_for, prepare_reference_spec

    config, store, spec, gene_list, _ = _large_whb_simulation(tmp_path, monkeypatch)
    # A version-6 family (the synthetic panel is pinned to version 6 here).
    config = config.model_copy(
        update={"resolvability": config.resolvability.model_copy(update={"version": 6})}
    )
    build = ReferenceBuild(
        spec=prepare_reference_spec(spec), builder=builder_for(spec, config)
    )

    def simulate_once(out: str, **options: Any) -> dict[str, Any]:
        return run_panel_simulation(
            gene_list=gene_list,
            species="human",
            name="whb_v6",
            config=config,
            store=store,
            builds=lambda panel: [build],
            out_dir=tmp_path / out,
            scratch_dir=tmp_path / f"{out}_scratch",
            platform="XENIUM",
            prefilter_compare="off",
            **options,
        )

    first = simulate_once("plain")
    record = first["references"]["whb_frontal_supc_clus"]
    bundle_dir = Path(record["bundle_dir"])
    summary = json.loads((bundle_dir / res.RESOLVABILITY_SUMMARY_FILE).read_text())
    assert summary["resolvability_version"] == 6
    before = bundle_digest(bundle_dir)
    predicted_before = pd.read_csv(tmp_path / "plain" / simulate.PREDICTED_CSV)
    second = simulate_once("diag", v7_diagnostic=True, v7_fresh_seeds=[3, 4, 5])
    diag_record = second["references"]["whb_frontal_supc_clus"]
    assert Path(diag_record["bundle_dir"]) == bundle_dir
    assert diag_record["reused"] is True
    # The bundle and its emitted decisions are unchanged.
    assert bundle_digest(bundle_dir) == before
    predicted_after = pd.read_csv(tmp_path / "diag" / simulate.PREDICTED_CSV)
    pd.testing.assert_frame_equal(predicted_before, predicted_after)
    diagnostic = diag_record["v7_diagnostic"]
    assert diagnostic["applied"] is False
    assert diagnostic["bundle_resolvability_version"] == 6
    assert diagnostic["ensemble_a"]["members"] == [
        "R1_contam_HO@0",
        "R1_contam_HO@1",
        "R1_contam_HO@2",
    ]
    assert diagnostic["ensemble_b"]["members"] == [
        "R1_contam_HO@3",
        "R1_contam_HO@4",
        "R1_contam_HO@5",
    ]
    for key in ("version_6_vs_7", "fresh_draw_churn"):
        assert {"validated", "provisional"} <= set(diagnostic[key])
        assert 0.0 <= diagnostic[key]["provisional"]["churn"] <= 1.0
    target = tmp_path / "diag" / "whb_frontal_supc_clus" / simulate.V7_DIAGNOSTIC_DIR
    assert (target / simulate.V7_DIAGNOSTIC_JSON).is_file()
    assert (target / "ensemble_A" / res.CLASS_DEPTH_FILE).is_file()
    assert "not applied" in (target / simulate.V7_DIAGNOSTIC_TXT).read_text()
    assert not any(
        path.is_relative_to(tmp_path / "diag")
        for root in store.roots
        for path in root.rglob("*")
    )
    stored = json.loads((bundle_dir / res.RESOLVABILITY_SUMMARY_FILE).read_text())
    assert stored == summary


def test_the_diagnostic_refuses_to_write_inside_a_store(tmp_path: Path) -> None:
    with pytest.raises(SimulationError, match="never written to the store"):
        simulate.run_v7_diagnostic(
            reference_id="whb_frontal_supc_clus",
            spec=None,  # type: ignore[arg-type]
            panel=None,  # type: ignore[arg-type]
            config=None,  # type: ignore[arg-type]
            bundle_dir=tmp_path,
            test_set_dir=tmp_path,
            resolvability={},
            chemistry=None,
            out_dir=tmp_path / "store" / "reports",
            scratch_dir=tmp_path / "scratch",
            store_roots=[tmp_path / "store"],
        )


def test_the_prefilter_comparison_keeps_the_version_6_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    small_resources: Any,  # noqa: F811
) -> None:
    from merxen.annotation.reference import builder_for, prepare_reference_spec

    config, store, spec, gene_list, unfiltered_calls = _large_whb_simulation(
        tmp_path, monkeypatch
    )
    config = config.model_copy(
        update={"resolvability": config.resolvability.model_copy(update={"version": 6})}
    )
    build = ReferenceBuild(
        spec=prepare_reference_spec(spec), builder=builder_for(spec, config)
    )
    report = run_panel_simulation(
        gene_list=gene_list,
        species="human",
        name="whb_v6",
        config=config,
        store=store,
        builds=lambda panel: [build],
        out_dir=tmp_path / "out",
        scratch_dir=tmp_path / "scratch_sim",
        platform="XENIUM",
    )
    record = report["references"]["whb_frontal_supc_clus"]
    assert record["prefilter_comparison"]["status"] == "run"
    assert unfiltered_calls == ["R1_contam_HO"]

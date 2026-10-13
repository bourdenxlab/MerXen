"""Tests for the panel card and trust banners (``report_panel``; plan §8.2)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from merxen.annotation import report_panel as rp


def _bundle(root: Path, *, version: int = 6, notes: list[str] | None = None) -> Path:
    """A minimal bundle: markers, lookup, profiles, resolvability tables."""
    bundle = root / "whb_frontal_supc_clus" / "b"
    bundle.mkdir(parents=True)
    (bundle / "bundle.json").write_text(
        json.dumps(
            {
                "builder_output": {
                    "markers": {
                        "lookup_file": "query_markers.filtered.json",
                        "weak_parents": ["SUPC/N2"],
                        "collapsed_parents": [{"parent": "SUPC/N3"}],
                    },
                    "marker_unsupported_nodes": {"n_nodes": 2},
                }
            }
        )
    )
    (bundle / "query_markers.filtered.json").write_text(
        json.dumps(
            {
                "None": ["g1", "g2", "g3"],
                "SUPC/N2": ["g1"],
                "SUPC/N3": [],
                "metadata": {},
            }
        )
    )
    pd.DataFrame(
        {"node": ["N2", "N3"], "node_name": ["Astrocyte", "Microglia"]}
    ).to_parquet(bundle / "profiles.parquet")
    rows: list[dict[str, Any]] = []
    for depth in (10, 30):
        for threshold in (0.5, 0.7, 0.9):
            rows.append(
                {
                    "kind": "curve",
                    "recipe": "R1",
                    "level": "broad",
                    "class": "Astro",
                    "depth": depth,
                    "threshold": threshold,
                    "precision": 0.8 + threshold / 10,
                    "coverage": 1 - threshold,
                    "n_confident": 100,
                }
            )
            rows.append(
                {
                    "kind": "curve",
                    "recipe": "clean",
                    "level": "broad",
                    "class": "Astro",
                    "depth": depth,
                    "threshold": threshold,
                    "precision": 1.0,
                    "coverage": 1.0,
                    "n_confident": 100,
                }
            )
        for regime, status, t_star in (
            ("validated", "emitted", 0.8),
            ("provisional", "not_resolvable", 0.95),
        ):
            rows.append(
                {
                    "kind": "decision",
                    "recipe": "R1",
                    "regime": regime,
                    "level": "broad",
                    "class": "Astro",
                    "depth": depth,
                    "status": status,
                    "t_star": t_star,
                    "default_threshold": 0.73,
                    "precision": 0.95,
                    "wilson_lb": 0.9,
                    "n_confident": 60,
                }
            )
    pd.DataFrame(rows).to_parquet(bundle / "resolvability.parquet")
    summary: dict[str, Any] = {
        "decision_recipe": "R1",
        "resolvability_version": version,
    }
    if notes is not None:
        summary["notes"] = notes
    (bundle / "resolvability_summary.json").write_text(json.dumps(summary))
    return bundle


def _sample(
    state: str = "validated", gate: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "sample_id": "PX_MERSCOPE",
        "platform": "MERSCOPE",
        "n_missing_panel_genes": 3,
        "summary": {
            "trust": {
                "state": state,
                "validation_basis": "real_data" if state == "validated" else None,
            },
            "resolution": {
                "gate": gate or {"level": "full", "warning": False},
                "levels": {
                    "broad": {
                        "emission": {
                            "regime": "validated",
                            "threshold_source": "validated_default",
                        }
                    }
                },
            },
        },
        "manifest": {
            "panel": {
                "panel_hash": "abc",
                "panel_family": "human_set_a",
                "gene_id_resolution": {"native": 290},
            },
            "references": {
                "whb_frontal_supc_clus": {
                    "markers": {"markers_per_parent_min": 1},
                    "n_query_genes_used": 290,
                }
            },
            "resolvability": {
                "whb_frontal_supc_clus": {
                    "d_max": {"astro": 120},
                    "extrapolated_share": {"astro": 0.1},
                    "emitted_depth_bins": {"broad": {"astro": [30]}},
                }
            },
            "thresholds": {
                "values": {"whb_broad": 0.73},
                "threshold_source": "validated_default",
                "floor_source": "packaged",
            },
        },
    }


PANEL_REPORT = {
    "declared_panels": {
        "PX_MERSCOPE": {
            "panel_hash": "declared",
            "n_features_in": 350,
            "n_controls_removed": 50,
            "controls_removed": {"name_pattern": ["Blank-1", "Blank-2"]},
            "gene_id_resolution": {"native": 296, "fallback_table": 4},
            "unresolved": {"MAPT3R": "not_in_fallback_table"},
            "species_check": {"status": "pass"},
        }
    }
}


@pytest.mark.parametrize(
    ("state", "code", "severity"),
    [
        ("refused", "panel_refused", "error"),
        ("broad_only", "panel_broad_only", "warning"),
        ("provisional", "panel_provisional", "warning"),
    ],
)
def test_trust_banners_name_the_panel_state(
    state: str, code: str, severity: str
) -> None:
    banners = rp.trust_banners("S", {"state": state}, {"level": "full"})
    assert [(banner.code, banner.severity) for banner in banners] == [(code, severity)]
    assert banners[0].text == rp.BANNER_TEXT[state]


def test_a_validated_panel_with_a_full_gate_has_no_banner() -> None:
    assert (
        rp.trust_banners(
            "S", {"state": "validated"}, {"level": "full", "warning": False}
        )
        == []
    )


def test_gate_banners_come_after_errors_first() -> None:
    banners = rp.trust_banners(
        "S",
        {"state": "provisional"},
        {
            "level": "failed",
            "level_reasons": ["panel_refused"],
            "warning": True,
            "warning_reasons": ["low"],
        },
        real_data_qc={"downgrades": ["broad_only"]},
    )
    codes = [banner.code for banner in banners]
    assert codes[0] == "gate_failed"
    assert set(codes) == {
        "gate_failed",
        "panel_provisional",
        "gate_warning",
        "real_data_qc_downgrade",
    }
    assert "panel_refused" in banners[0].text
    broad = rp.trust_banners(
        "S", {"state": "validated"}, {"level": "broad_only", "level_reasons": ["A<0.3"]}
    )
    assert [banner.code for banner in broad] == ["gate_broad_only"]
    unknown = rp.trust_banners("S", None, None)
    assert [banner.code for banner in unknown] == ["panel_trust_unknown"]


def test_v7_notes_only_for_version_7_bundles(tmp_path: Path) -> None:
    assert rp.v7_notes(_bundle(tmp_path / "v6", version=6)) == []
    generic = rp.v7_notes(_bundle(tmp_path / "v7", version=7))
    assert generic == [rp.GENERIC_V7_NOTE.format(version=7)]
    assert "upper bound" in generic[0]
    recorded = rp.v7_notes(_bundle(tmp_path / "v7n", version=7, notes=["glial note A"]))
    assert recorded == ["glial note A"]
    assert rp.v7_notes(tmp_path / "missing") == []


def test_bundle_marker_rows_mark_weak_and_collapsed_parents(tmp_path: Path) -> None:
    rows = {
        row["parent"]: row for row in rp.bundle_marker_rows("whb", _bundle(tmp_path))
    }
    assert set(rows) == {"None", "SUPC/N2", "SUPC/N3"}
    assert rows["None"]["parent_name"] == "root" and rows["None"]["n_markers"] == 3
    assert rows["SUPC/N2"]["weak"] and rows["SUPC/N2"]["parent_name"] == "Astrocyte"
    assert rows["SUPC/N3"]["collapsed"] and rows["SUPC/N3"]["n_markers"] == 0


def test_resolvability_tables_use_the_decision_recipe(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    curves = rp.resolvability_curves("whb", bundle)
    assert len(curves) == 6 and set(curves["depth"]) == {10, 30}
    assert (curves["precision"] < 1.0).all()  # the clean recipe is left out
    decisions = rp.resolvability_bins("whb", bundle)
    assert set(decisions["regime"]) == {"validated", "provisional"}
    assert len(decisions) == 4
    assert rp.resolvability_curves("whb", tmp_path / "none").empty


def test_build_panel_card_collects_ids_controls_markers_and_emission(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    floors = pd.DataFrame(
        {
            "level": ["broad", "broad"],
            "floor_class": ["Astro", "Astro"],
            "platform": ["MERSCOPE", "XENIUM"],
            "panel_family": ["human_set_a", "human_set_a"],
            "min_counts": [10, 10],
        }
    )
    card = rp.build_panel_card(
        [_sample()],
        panel_report=PANEL_REPORT,
        bundles={"whb_frontal_supc_clus": bundle},
        primary_reference="whb_frontal_supc_clus",
        floors=floors,
    )
    summary = card.summary.iloc[0]
    assert (
        summary["trust_state"] == "validated"
        and summary["validation_basis"] == "real_data"
    )
    assert summary["n_controls_removed"] == 50 and summary["n_unresolved"] == 1
    assert summary["n_missing_panel_genes"] == 3
    assert dict(
        zip(card.gene_ids["source"], card.gene_ids["n_genes"], strict=True)
    ) == {"fallback_table": 4, "native": 296}
    assert card.controls.iloc[0]["n_features"] == 2
    assert card.unresolved.iloc[0]["feature"] == "MAPT3R"
    assert list(card.floors["platform"]) == ["MERSCOPE"]
    assert card.classes.iloc[0]["d_max"] == 120
    emitted = card.emission[f"emitted_reweighted_{'PX_MERSCOPE'}"]
    by_depth = dict(
        zip(
            zip(card.emission["regime"], card.emission["depth"], strict=True),
            emitted,
            strict=True,
        )
    )
    assert bool(by_depth[("validated", 30)])
    assert not bool(by_depth[("validated", 10)])
    assert any("too few markers" in note for note in card.notes)
    assert any("3 declared panel genes absent" in note for note in card.notes)
    assert card.banners == []
    tables = card.tables()
    assert set(tables) >= {
        "panel_summary",
        "panel_resolvability_curves",
        "panel_emission",
        "panel_floors",
    }


def test_build_panel_card_without_records_still_builds(tmp_path: Path) -> None:
    card = rp.build_panel_card(
        [
            _sample(
                state="refused",
                gate={"level": "failed", "level_reasons": ["panel_refused"]},
            )
        ],
        panel_report=None,
        bundles={},
        primary_reference="whb_frontal_supc_clus",
    )
    assert card.curves.empty and card.markers.empty
    codes = {banner.code for banner in card.banners}
    assert codes == {"panel_refused", "gate_failed"}
    assert card.gene_ids.iloc[0]["n_genes"] == 290  # from the provenance
    assert isinstance(card.emission, pd.DataFrame)
    assert pd.isna(card.summary.iloc[0]["n_features_in"])

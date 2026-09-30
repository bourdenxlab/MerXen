"""Cross-platform scope of a pair from RESOLVE's pair summary (plan §5.5, §8.5)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from merxen.annotation import pipeline as pipeline_module
from merxen.annotation.composition import COMPOSITION_KINDS
from merxen.annotation.schema import GATE_LEVELS
from merxen.clustering import cross_platform as cross_platform_module
from merxen.clustering.cross_platform import (
    NO_PAIR_RECORD,
    SOURCE_MISSING,
    CrossPlatformScope,
    dataset_gate_levels,
    load_cross_platform_scope,
    resolve_summary_filename,
    scope_from_resolve_summary,
)

SAME_PANEL = {
    "panel_mode": "intersection",
    "jsd_run": ["whb_frontal_supc_clus"],
    "jsd_purpose": "annotation",
    "kinds": list(COMPOSITION_KINDS),
    "omitted_kinds": {},
    "statistics_level": "full",
    "flag": False,
    "reasons": [],
}
PER_PLATFORM = {
    "panel_mode": "per_platform",
    "jsd_run": ["whb_frontal_supc_clus_xpanel"],
    "jsd_purpose": "intersection_xpanel",
    "kinds": list(pipeline_module.XPANEL_KINDS),
    "omitted_kinds": {"confident": "RESOLVE does not resolve the intersection run"},
    "statistics_level": "full",
    "flag": False,
    "reasons": [],
}


def _summary(
    cross_platform: dict[str, Any] | None, gates: dict[str, str] | None = None
) -> dict[str, Any]:
    pair: dict[str, Any] = {"jsd": [], "mask_note": None}
    if cross_platform is not None:
        pair["cross_platform"] = cross_platform
    samples = {
        sample_id: {"resolution": {"gate": {"level": level, "warning": False}}}
        for sample_id, level in (gates or {}).items()
    }
    return {
        "pair_id": "P1",
        "panel_mode": "per_platform",
        "samples": samples,
        "pair": pair,
    }


def test_gate_levels_match_the_schema() -> None:
    assert cross_platform_module.GATE_LEVELS == GATE_LEVELS
    assert set(cross_platform_module.GATE_STATISTICS_CAP) == set(GATE_LEVELS)


def test_suffix_matches_the_resolve_step() -> None:
    assert (
        cross_platform_module.RESOLVE_SUMMARY_SUFFIX
        == pipeline_module.RESOLVE_SUMMARY_SUFFIX
    )
    assert resolve_summary_filename("P1") == pipeline_module.resolve_summary_filename(
        "P1"
    )


def test_same_panel_pair_allows_every_level() -> None:
    scope = scope_from_resolve_summary(
        _summary(SAME_PANEL, {"P1_MERSCOPE": "full", "P1_XENIUM": "full"})
    )

    assert scope.statistics_level == "full"
    assert scope.flag is False
    assert scope.jsd_runs == ("whb_frontal_supc_clus",)
    assert scope.allows("supercluster") and scope.allows("broad")
    assert all(scope.allows_kind(kind) for kind in COMPOSITION_KINDS)
    assert scope.dataset_gates == (("P1_MERSCOPE", "full"), ("P1_XENIUM", "full"))


def test_a_broad_only_dataset_restricts_a_same_panel_pair_to_broad() -> None:
    """P1212: same panel, RESOLVE records full, but MERSCOPE is broad_only (§5.4)."""
    scope = scope_from_resolve_summary(
        _summary(SAME_PANEL, {"P1_MERSCOPE": "broad_only", "P1_XENIUM": "full"})
    )

    assert scope.resolve_statistics_level == "full"
    assert scope.statistics_level == "broad_only"
    assert scope.flag is True
    assert scope.reasons == ("dataset_gate:P1_MERSCOPE:broad_only",)
    assert not scope.allows("supercluster")
    assert not scope.allows("nt")
    assert scope.allows("broad") and scope.allows("lineage")
    assert scope.allows_kind("soft")


def test_a_failed_dataset_gate_allows_no_statement() -> None:
    scope = scope_from_resolve_summary(
        _summary(
            PER_PLATFORM | {"statistics_level": "broad_only", "flag": True},
            {"P1_MERSCOPE": "failed", "P1_XENIUM": "broad_only"},
        )
    )

    assert scope.statistics_level == "none"
    assert scope.reasons == (
        "dataset_gate:P1_MERSCOPE:failed",
        "dataset_gate:P1_XENIUM:broad_only",
    )
    assert not scope.allows("broad")
    assert not scope.allows_kind("soft")


def test_gates_never_widen_the_panel_level() -> None:
    scope = scope_from_resolve_summary(
        _summary(
            PER_PLATFORM
            | {
                "statistics_level": "none",
                "flag": True,
                "reasons": ["no_intersection_run"],
            },
            {"P1_MERSCOPE": "full", "P1_XENIUM": "broad_only"},
        )
    )

    assert scope.statistics_level == "none"
    assert scope.reasons == ("no_intersection_run", "dataset_gate:P1_XENIUM:broad_only")


def test_a_per_platform_pair_refuses_confident_label_comparisons() -> None:
    """RESOLVE leaves the confident kind out of a per_platform pair (§8.5)."""
    scope = scope_from_resolve_summary(
        _summary(PER_PLATFORM, {"P1_MERSCOPE": "full", "P1_XENIUM": "full"})
    )

    assert scope.allows("supercluster")
    assert scope.omitted_kinds == ("confident",)
    assert not scope.allows_kind("confident")
    assert all(scope.allows_kind(kind) for kind in pipeline_module.XPANEL_KINDS)


def test_a_scope_without_recorded_kinds_compares_none() -> None:
    record = {key: value for key, value in SAME_PANEL.items() if key != "kinds"}
    scope = scope_from_resolve_summary(_summary(record))

    assert scope.allows("supercluster")
    assert not scope.allows_kind("soft")


def test_an_unknown_gate_level_is_refused() -> None:
    with pytest.raises(ValueError, match="gate level"):
        dataset_gate_levels(_summary(SAME_PANEL, {"P1_MERSCOPE": "partial"}))


def test_broad_only_pair_restricts_statements_to_broad_levels() -> None:
    """A per_platform pair with a small intersection compares broad levels only."""
    scope = scope_from_resolve_summary(
        _summary(
            {
                "panel_mode": "per_platform",
                "jsd_run": ["whb_frontal_supc_clus_xpanel"],
                "jsd_purpose": "intersection_xpanel",
                "statistics_level": "broad_only",
                "flag": True,
                "reasons": ["intersection_genes:80<100"],
            }
        )
    )

    assert scope.flag is True
    assert scope.reasons == ("intersection_genes:80<100",)
    assert scope.allows("broad") and scope.allows("lineage")
    assert not scope.allows("supercluster")
    assert not scope.allows("nt")


def test_no_intersection_run_allows_no_statement() -> None:
    scope = scope_from_resolve_summary(
        _summary(
            {
                "panel_mode": "per_platform",
                "jsd_run": [],
                "statistics_level": "none",
                "flag": True,
                "reasons": ["no_intersection_run"],
            }
        )
    )

    assert not scope.allows("broad")


def test_a_summary_without_a_pair_record_allows_no_statement() -> None:
    scope = scope_from_resolve_summary(_summary(None))

    assert scope.statistics_level == "none"
    assert scope.reasons == (NO_PAIR_RECORD,)
    assert not scope.allows("broad")


def test_missing_summary_allows_no_statement(tmp_path: Path) -> None:
    scope = load_cross_platform_scope(tmp_path, "P1")

    assert scope.source == SOURCE_MISSING
    assert scope.flag is True
    assert not scope.allows("broad")


def test_scope_loads_from_a_resolve_output_and_round_trips(tmp_path: Path) -> None:
    record = {
        **PER_PLATFORM,
        "jsd_run": ["a_xpanel", "b_xpanel"],
        "statistics_level": "broad_only",
        "flag": True,
        "reasons": ["intersection_trust:broad_only"],
    }
    (tmp_path / resolve_summary_filename("P1")).write_text(
        json.dumps(_summary(record, {"P1_XENIUM": "full", "P1_MERSCOPE": "broad_only"}))
    )

    scope = load_cross_platform_scope(tmp_path, "P1")

    assert scope.jsd_runs == ("a_xpanel", "b_xpanel")
    assert CrossPlatformScope.from_dict(scope.to_dict()) == scope
    payload = json.loads(json.dumps(scope.to_dict()))
    assert payload["reasons"] == [
        "intersection_trust:broad_only",
        "dataset_gate:P1_MERSCOPE:broad_only",
    ]
    assert payload["dataset_gates"] == {
        "P1_MERSCOPE": "broad_only",
        "P1_XENIUM": "full",
    }
    assert payload["omitted_kinds"] == ["confident"]


def test_an_unknown_level_is_refused() -> None:
    with pytest.raises(ValueError, match="statistics_level"):
        CrossPlatformScope(statistics_level="partial", flag=False)

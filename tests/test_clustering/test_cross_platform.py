"""Cross-platform scope of a pair from RESOLVE's pair summary (plan §5.5, §8.5)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from merxen.annotation import pipeline as pipeline_module
from merxen.clustering import cross_platform as cross_platform_module
from merxen.clustering.cross_platform import (
    NO_PAIR_RECORD,
    SOURCE_MISSING,
    CrossPlatformScope,
    load_cross_platform_scope,
    resolve_summary_filename,
    scope_from_resolve_summary,
)


def _summary(cross_platform: dict[str, Any] | None) -> dict[str, Any]:
    pair: dict[str, Any] = {"jsd": [], "mask_note": None}
    if cross_platform is not None:
        pair["cross_platform"] = cross_platform
    return {"pair_id": "P1", "panel_mode": "per_platform", "pair": pair}


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
        _summary(
            {
                "panel_mode": "intersection",
                "jsd_run": ["whb_frontal_supc_clus"],
                "jsd_purpose": "annotation",
                "statistics_level": "full",
                "flag": False,
                "reasons": [],
            }
        )
    )

    assert scope.statistics_level == "full"
    assert scope.jsd_runs == ("whb_frontal_supc_clus",)
    assert scope.allows("supercluster") and scope.allows("broad")


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
        "panel_mode": "per_platform",
        "jsd_run": ["a_xpanel", "b_xpanel"],
        "jsd_purpose": "intersection_xpanel",
        "statistics_level": "broad_only",
        "flag": True,
        "reasons": ["intersection_trust:broad_only"],
    }
    (tmp_path / resolve_summary_filename("P1")).write_text(json.dumps(_summary(record)))

    scope = load_cross_platform_scope(tmp_path, "P1")

    assert scope.jsd_runs == ("a_xpanel", "b_xpanel")
    assert CrossPlatformScope.from_dict(scope.to_dict()) == scope
    assert json.loads(json.dumps(scope.to_dict()))["reasons"] == [
        "intersection_trust:broad_only"
    ]


def test_an_unknown_level_is_refused() -> None:
    with pytest.raises(ValueError, match="statistics_level"):
        CrossPlatformScope(statistics_level="partial", flag=False)

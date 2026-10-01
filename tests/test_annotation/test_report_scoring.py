"""Tests for the M8 gate scoring rules (``report_scoring``; prereg §18, §20)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from merxen.annotation import report_scoring as sc
from merxen.annotation.report_depth import PRIMARY_CI, SQUARE_TILE_CI


def _record(name: str, value: Any, **fields: Any) -> dict[str, Any]:
    return {
        "criterion": "H12",
        "name": name,
        "value": value,
        "status": "measured" if value is not None else "not_available",
        "kind": None,
        "platform": None,
        "note": "",
        **fields,
    }


def _h12_records(
    tile: dict[str, bool | None],
    tangential: dict[str, bool | None],
    wm_gm: dict[str, tuple[float, float]],
) -> list[dict[str, Any]]:
    records = [_record("depth_ci_method", PRIMARY_CI, scope="pair")]
    for platform, value in tile.items():
        records.append(
            _record(
                "depth_ordering_passes", value, platform=platform, kind=SQUARE_TILE_CI
            )
        )
    for platform, value in tangential.items():
        records.append(_record("depth_ordering_passes", value, platform=platform))
    for platform, (value, low) in wm_gm.items():
        records.append(
            _record(
                "oligodendrocyte_wm_minus_gm_share",
                value,
                platform=platform,
                kind="outside_ribbon_proxy",
                ci_low=low,
                ci_high=value + 0.05,
            )
        )
    return records


def test_score_h12_scores_the_tangential_blocks_not_the_tiles() -> None:
    """The P7513_XENIUM case of §17: tiles fail, blocks pass (§20 D11)."""
    records = _h12_records(
        {"MERSCOPE": True, "XENIUM": False},
        {"MERSCOPE": True, "XENIUM": True},
        {"MERSCOPE": (0.5, 0.45), "XENIUM": (0.5, 0.46)},
    )
    records.append(
        _record("depth_ordering_replicated", False, scope="pair", kind=SQUARE_TILE_CI)
    )
    score = sc.score_h12(records, pair_id="P7513")
    assert score.verdict == sc.PASS and score.ci_scored == PRIMARY_CI
    assert score.ordering == {"MERSCOPE": sc.PASS, "XENIUM": sc.PASS}
    assert score.reported_beside["ordering"] == {
        "MERSCOPE": sc.PASS,
        "XENIUM": sc.FAIL,
    }
    assert score.reported_beside["replicated"] == [False]
    record = score.to_json()
    assert record["ci_scored"] == PRIMARY_CI
    assert record["ci_reported_beside"] == SQUARE_TILE_CI
    # The blocks failing fail the pair, whatever the tiles say.
    failing = _h12_records(
        {"MERSCOPE": True, "XENIUM": True},
        {"MERSCOPE": True, "XENIUM": False},
        {"MERSCOPE": (0.5, 0.45), "XENIUM": (0.5, 0.46)},
    )
    assert sc.score_h12(failing, pair_id="P7513").verdict == sc.FAIL
    # A scored replication record that contradicts its platforms is an error.
    records.append(_record("depth_ordering_replicated", False, scope="pair"))
    with pytest.raises(ValueError, match="depth_ordering_replicated"):
        sc.score_h12(records, pair_id="P7513")


def test_score_h12_wm_gm_needs_the_tile_ci_above_zero_on_both_platforms() -> None:
    records = _h12_records(
        {"MERSCOPE": True, "XENIUM": True},
        {},
        {"MERSCOPE": (0.5, 0.45), "XENIUM": (0.02, -0.01)},
    )
    score = sc.score_h12(records, pair_id="P5011")
    assert not score.ordering_required
    assert score.wm_gm["XENIUM"]["verdict"] == sc.FAIL
    assert score.verdict == sc.FAIL
    # An invalid depth input (not_available) is not a failure.
    records[-1]["status"] = "not_available"
    records[-1]["value"] = None
    assert sc.score_h12(records, pair_id="P5011").verdict == sc.NOT_AVAILABLE


def test_score_h12_never_scores_the_tiles_beside_the_blocks() -> None:
    records = _h12_records(
        {"MERSCOPE": True, "XENIUM": True}, {}, {"MERSCOPE": (0.5, 0.4)}
    )
    score = sc.score_h12(records, pair_id="P7513")
    assert score.ordering_verdict == sc.NOT_AVAILABLE
    assert score.verdict == sc.NOT_AVAILABLE
    # A report whose only CI was the tiles is scored on its primary records.
    records = _h12_records(
        {}, {"MERSCOPE": True, "XENIUM": True}, {"MERSCOPE": (0.5, 0.4)}
    )
    records[0]["value"] = SQUARE_TILE_CI
    tiles_only = sc.score_h12(records, pair_id="P7513")
    assert tiles_only.ordering_verdict == sc.PASS
    assert tiles_only.ci_scored == SQUARE_TILE_CI
    assert tiles_only.ci_reported_beside is None
    assert any("no tangential positions" in note for note in tiles_only.notes)


def test_score_h12_names_a_mixed_ci_per_platform() -> None:
    """One platform without tangential positions: each is scored on its own."""
    records = _h12_records(
        {"MERSCOPE": True, "XENIUM": False},
        {"MERSCOPE": True, "XENIUM": False},
        {"MERSCOPE": (0.5, 0.4), "XENIUM": (0.5, 0.4)},
    )
    records[0]["value"] = SQUARE_TILE_CI  # the pair-level method record
    for record in records:
        if record["name"] == "depth_ordering_passes" and record["kind"] is None:
            method = PRIMARY_CI if record["platform"] == "MERSCOPE" else SQUARE_TILE_CI
            record["note"] = f"ordered=True; ci={method}"
    score = sc.score_h12(records, pair_id="P7513")
    assert score.ci_scored == "mixed"
    assert score.ordering == {"MERSCOPE": sc.PASS, "XENIUM": sc.FAIL}
    assert any("mixed CIs" in note for note in score.notes)


def test_d3_holds_only_at_the_text_value_and_level() -> None:
    kwargs = {"segmentation": "proseg_hybrid", "dataset": "P1212_MERSCOPE"}
    held = sc.check_d3(value=0.1482, level="broad_only", **kwargs)
    assert held is not None and held.in_scope
    assert held.verdict == "EXCEPTION (D3)"
    moved = sc.check_d3(value=0.130, level="broad_only", **kwargs)
    assert moved is not None and not moved.in_scope
    assert moved.verdict == "EXCEPTION-RECHECK (D3)"
    assert not sc.check_d3(value=0.148, level="full", **kwargs).in_scope
    assert (
        sc.check_d3(
            segmentation="proseg_hybrid",
            dataset="P5011_MERSCOPE",
            value=0.148,
            level="broad_only",
        )
        is None
    )


def test_d5_checks_the_value_and_the_confident_labels_of_the_flagged_cells() -> None:
    base = {
        "criterion": "H2",
        "segmentation": "proseg_hybrid",
        "dataset": "P5011_MERSCOPE",
        "n_confident_broad": 0,
        "n_confident_supercluster": 0,
    }
    assert sc.check_d5(value=0.010177, **base).in_scope
    high = sc.check_d5(value=0.0112, **base)
    assert high is not None and not high.in_scope
    labelled = sc.check_d5(value=0.0101, **{**base, "n_confident_broad": 3})
    assert labelled is not None and not labelled.in_scope
    assert "confident broad" in labelled.reasons[0]
    unmeasured = sc.check_d5(value=0.0101, **{**base, "n_confident_broad": None})
    assert unmeasured is not None and not unmeasured.in_scope
    reseg = sc.check_d5(
        **{
            **base,
            "criterion": "H17/H2",
            "segmentation": "reseg",
            "dataset": "P1212_MERSCOPE",
        },
        value=0.011311,
    )
    assert reseg is not None and reseg.in_scope
    # §20 D13 moved the reseg text value to the pipeline's 1.13%.
    above = sc.check_d5(
        **{
            **base,
            "criterion": "H17/H2",
            "segmentation": "reseg",
            "dataset": "P1212_MERSCOPE",
        },
        value=0.0119,
    )
    assert above is not None and not above.in_scope
    # D5 names H17/H2 for P1212_MERSCOPE reseg, not proseg_hybrid H2.
    assert sc.check_d5(**{**base, "dataset": "P1212_MERSCOPE"}, value=0.0103) is None


def _h4_classes(**changes: tuple[float, float, float, bool]) -> pd.DataFrame:
    rows = {
        "Neurons": (11.3, 0.819, 0.841, True),
        "Astrocytes": (15.7, 0.870, 0.881, True),
        "Oligodendrocytes": (10.8, 0.6458, 0.6535, False),
        "Oligodendrocyte precursors": (28.3, 0.861, 0.868, True),
        "Microglia": (5.85, 0.5400, 0.5409, False),
        "Vascular cells": (9.22, 0.6626, 0.6685, False),
        "Fibroblasts": (12.7, 0.927, 0.956, True),
    }
    rows.update(changes)
    return pd.DataFrame(
        [
            {
                "broad_class": name,
                "fold": fold,
                "auroc": auroc,
                "auroc_ceiling": ceiling,
                "passes": passes,
            }
            for name, (fold, auroc, ceiling, passes) in rows.items()
        ]
    )


def test_d7_needs_the_texts_classes_at_their_auroc_ceilings() -> None:
    kwargs = {"segmentation": "proseg_hybrid", "dataset": "P1212_MERSCOPE"}
    held = sc.check_d7(classes=_h4_classes(), **kwargs)
    assert held is not None and held.in_scope
    other = sc.check_d7(
        classes=_h4_classes(Neurons=(11.3, 0.60, 0.84, False)), **kwargs
    )
    assert other is not None and not other.in_scope
    assert any("Neurons fails" in reason for reason in other.reasons)
    off_ceiling = sc.check_d7(
        classes=_h4_classes(Microglia=(5.85, 0.52, 0.56, False)), **kwargs
    )
    assert off_ceiling is not None and not off_ceiling.in_scope
    low_fold = sc.check_d7(
        classes=_h4_classes(Microglia=(2.5, 0.540, 0.541, False)), **kwargs
    )
    assert low_fold is not None and not low_fold.in_scope
    assert (
        sc.check_d7(
            classes=_h4_classes(),
            segmentation="proseg_hybrid",
            dataset="P7113_MERSCOPE",
        )
        is None
    )


def test_d4_h18_scope_is_the_named_bins_of_the_named_datasets() -> None:
    def check(dataset: str, level: str, cls: str, missing: set[int]) -> str | None:
        result = sc.check_d4_h18(dataset=dataset, level=level, cls=cls, missing=missing)
        return None if result is None else result.verdict

    assert check("P7513_MERSCOPE", "broad", "OPC", {15, 120}) == "EXCEPTION (D4)"
    assert check("P7513_MERSCOPE", "broad", "OPC", {15, 30}) == (
        "EXCEPTION-RECHECK (D4)"
    )
    # P7113_XENIUM emits broad OPC at 120 counts: only 15 is covered there.
    assert check("P7113_XENIUM", "broad", "OPC", {15}) == "EXCEPTION (D4)"
    assert check("P7113_XENIUM", "broad", "OPC", {15, 120}) == (
        "EXCEPTION-RECHECK (D4)"
    )
    assert check("P5011_MERSCOPE", "supercluster", "Immune", {60}) == ("EXCEPTION (D4)")
    assert check("P5011_MERSCOPE", "broad", "Immune", {30, 60}) == (
        "EXCEPTION-RECHECK (D4)"
    )
    assert check("P5011_XENIUM", "broad", "Immune", {60}) is None
    assert check("P7513_MERSCOPE", "broad", "Astro", {15}) is None
    assert check("PREP[MERSCOPE floors]", "broad", "OPC", {15}) is None


def test_d4_would_raise_lists_are_the_stage_a2_lists() -> None:
    prep = sc.check_d4_would_raise(
        dataset="PREP[MERSCOPE floors]", bins=sc.D4_PREP_WOULD_RAISE, reference=None
    )
    assert prep is not None and prep.in_scope
    assert len(sc.D4_PREP_WOULD_RAISE) == 15
    added = sc.check_d4_would_raise(
        dataset="PREP[XENIUM floors]",
        bins=[*sc.D4_PREP_WOULD_RAISE, ("broad", "Oligo", 30)],
        reference=None,
    )
    assert added is not None and not added.in_scope
    assert "broad Oligo 30" in added.reasons[0]
    reference = [("broad", "OPC", 15), ("broad", "Astro", 30)]
    fewer = sc.check_d4_would_raise(
        dataset="P7513_MERSCOPE", bins=reference[:1], reference=reference
    )
    assert fewer is not None and fewer.in_scope
    unlisted = sc.check_d4_would_raise(
        dataset="P7513_MERSCOPE", bins=reference, reference=None
    )
    assert unlisted is not None and not unlisted.in_scope
    assert (
        sc.check_d4_would_raise(dataset="P7513_MERSCOPE", bins=[], reference=None)
        is None
    )


def test_verdict_labels_the_row() -> None:
    held = sc.ExceptionVerdict("D5", True)
    moved = sc.ExceptionVerdict("D5", False, ("above the text",))
    assert sc.verdict(True, True, None) == sc.PASS
    assert sc.verdict(False, False, None) == "INFO fail"
    assert sc.verdict(False, True, held) == "EXCEPTION (D5)"
    assert sc.verdict(False, True, moved) == "EXCEPTION-RECHECK (D5)"
    assert sc.verdict(False, True, None) == sc.FAIL_OUTSIDE
    assert set(sc.TOLERANCES) == {"D3", "D4", "D5", "D7", "D10", "D12"}
    assert sc.TOLERANCES["D10"]["n_rows"] == 70


def test_d10_covers_only_the_stage_b_rows_of_each_dataset() -> None:
    hybrid = {"segmentation": "proseg_hybrid"}
    assert sc.check_d10(dataset="P7513_MERSCOPE", failing=[], **hybrid) is None
    fibro = ("supercluster", "Fibroblast", 120)
    held = sc.check_d10(
        dataset="P5011_XENIUM",
        failing=[
            ("broad", "OPC", 15, 0.8932),
            (*fibro, 0.78),
            ("supercluster", "Oligodendrocyte precursor", 60, 0.6726),
        ],
        **hybrid,
    )
    assert held is not None and held.in_scope
    assert held.verdict == "EXCEPTION (D10)"
    # Broad OPC 60 failed on P7113_MERSCOPE only; Fibroblast 120 on
    # P5011_XENIUM only; Astro is not named.
    for dataset, row in (
        ("P7513_MERSCOPE", ("broad", "OPC", 60, 0.8)),
        ("P7513_XENIUM", (*fibro, 0.78)),
        ("P7513_XENIUM", ("broad", "Astro", 15, 0.8)),
    ):
        moved = sc.check_d10(dataset=dataset, failing=[row], **hybrid)
        assert moved is not None and not moved.in_scope
        assert moved.verdict == "EXCEPTION-RECHECK (D10)"
    # A named row whose precision falls more than the tolerance below stage
    # B's value goes back to the user.
    floor = sc.D10_FAILING[(*fibro, "P5011_XENIUM")]
    low = sc.check_d10(
        dataset="P5011_XENIUM", failing=[(*fibro, floor - 0.02)], **hybrid
    )
    assert low is not None and not low.in_scope
    assert "below stage B" in low.reasons[0]
    # H6 is scored on proseg_hybrid only.
    assert (
        sc.check_d10(
            dataset="P5011_XENIUM", failing=[(*fibro, 0.78)], segmentation="reseg"
        )
        is None
    )


def test_d12_needs_no_switch_and_the_texts_threshold_share() -> None:
    held = sc.check_d12(pair="P1212", share_threshold=0.13403, n_switched=0)
    assert held is not None and held.in_scope
    assert held.verdict == "EXCEPTION (D12)"
    assert not sc.check_d12(
        pair="P1212", share_threshold=0.13403, n_switched=2
    ).in_scope
    assert not sc.check_d12(pair="P7513", share_threshold=0.0712, n_switched=0).in_scope
    assert not sc.check_d12(
        pair="P7513", share_threshold=0.07, n_switched=None
    ).in_scope
    assert sc.check_d12(pair="P7113", share_threshold=0.07, n_switched=0) is None
    # A cell missing from one seed's table would count as a crossing.
    assert not sc.check_d12(
        pair="P7513", share_threshold=0.07, n_switched=0, n_missing=3
    ).in_scope

"""Regional policies refuse changed artifacts and unsupported fine resolution."""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from merxen.annotation.hippocampal_profile import (
    ConfidenceObjective,
    DepthRule,
    HippocampalPolicy,
    ReferenceBinding,
    calibrate_rules,
    ordered_genes_sha256,
    resolve_reference_calls,
    wilson_lower,
)


def objective() -> ConfidenceObjective:
    """Explicit conservative example; no production/default precision inferred."""
    return ConfidenceObjective(
        min_precision=0.99,
        min_wilson_lower=0.95,
        wilson_z=1.6448536269514722,
        bootstrap_cuts=(0.0, 0.5, 0.9, 0.95, 1.0),
    )


def binding() -> ReferenceBinding:
    return ReferenceBinding(
        reference_id="seaad_hippocampal",
        taxonomy_id="regional",
        regions=("HIP", "MEC", "LEC"),
        levels=("parent", "child"),
        panel_hash="a" * 64,
        query_genes_sha256=ordered_genes_sha256(genes()),
        reference_sha256="b" * 64,
        lookup_sha256="c" * 64,
        tree_sha256="d" * 64,
    )


def genes() -> list[str]:
    return [f"gene_{i}" for i in range(18049)]


def rule(level: str, label: str, start: int = 100) -> DepthRule:
    return DepthRule(
        level=level,
        assignment=label,
        min_counts=start,
        min_bootstrap=0.95,
        n_evaluated=100,
        n_accepted=100,
        n_correct=100,
        precision=1.0,
        wilson_lower=wilson_lower(100, 100, objective().wilson_z),
    )


def policy() -> HippocampalPolicy:
    return HippocampalPolicy(
        binding=binding(),
        objective=objective(),
        calibration_sha256="e" * 64,
        evidence_basis="existing_atlas_diagnostics_reused",
        depth_edges=(100,),
        rules=(rule("parent", "P"), rule("child", "A")),
    )


def predictions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("a", "parent", "P", "Parent", 1.0, True),
            ("b", "parent", "P", "Parent", 0.7, True),
            ("c", "parent", "P", "Parent", 1.0, True),
            ("a", "child", "A", "Child A", 1.0, True),
            ("b", "child", "A", "Child A", 1.0, True),
            ("c", "child", "A", "Child A", 1.0, False),
        ],
        columns=["cell_id", "level", "assignment", "name", "bp", "marker_supported"],
    )


def objects() -> pd.DataFrame:
    return pd.DataFrame(
        {"cell_id": ["c", "empty", "a", "b"], "n_counts": [50, 0, 500, 500]}
    )


def resolve(pred: pd.DataFrame | None = None) -> pd.DataFrame:
    return resolve_reference_calls(
        objects(),
        predictions() if pred is None else pred,
        policy=policy(),
        binding=binding(),
        parent_of={("child", "A"): "P"},
    )


def test_parent_abstention_and_all_empty_and_sparse_objects_retained() -> None:
    result = resolve().set_index(["cell_id", "level"])
    assert len(result) == 8
    assert result.loc[("a", "child"), "name"] == "Child A"
    assert result.loc[("b", "child"), "resolution_status"] == "parent_unresolved"
    assert pd.isna(result.loc[("b", "child"), "name"])
    assert result.loc[("b", "child"), "raw_name"] == "Child A"
    assert result.loc[("empty", "child"), "resolution_status"] == "no_expression"
    assert result.loc[("c", "child"), "resolution_status"] == "outside_calibrated_depth"
    assert (
        result.loc[("a", "child"), "evidence_basis"]
        == "existing_atlas_diagnostics_reused"
    )


@pytest.mark.parametrize(
    "change",
    ["missing_cell", "duplicate_cell", "unexpected_cell", "missing_level", "bad_bp"],
)
def test_lost_duplicated_or_invalid_predictions_are_refused(change: str) -> None:
    pred = predictions()
    if change == "missing_cell":
        pred = pred.iloc[1:]
    elif change == "duplicate_cell":
        pred = pd.concat([pred, pred.iloc[:1]])
    elif change == "unexpected_cell":
        pred.loc[0, "cell_id"] = "other"
    elif change == "missing_level":
        pred = pred.loc[pred.level.eq("parent")]
    else:
        pred.loc[0, "bp"] = np.inf
    with pytest.raises(ValueError):
        resolve(pred)


@pytest.mark.parametrize(
    "field", ["reference_sha256", "tree_sha256", "lookup_sha256", "panel_hash"]
)
def test_modified_reference_or_panel_binding_is_refused(field: str) -> None:
    changed = ReferenceBinding.model_validate(
        {**binding().model_dump(), field: "f" * 64}
    )
    with pytest.raises(ValueError, match="binding changed"):
        resolve_reference_calls(
            objects(),
            predictions(),
            policy=policy(),
            binding=changed,
            parent_of={("child", "A"): "P"},
        )


def test_full_query_panel_must_preserve_columns_order_and_denominator() -> None:
    b = binding()
    b.check_query(b.panel_hash, genes())
    for altered in [genes()[:-1], genes()[::-1], genes()[:-1] + [genes()[0]]]:
        with pytest.raises(ValueError, match="full panel"):
            b.check_query(b.panel_hash, altered)
    with pytest.raises(ValueError):
        b.check_query("f" * 64, genes())


def test_marker_unsupported_and_unknown_classes_keep_raw_calls() -> None:
    pred = predictions()
    pred.loc[pred.cell_id.eq("a") & pred.level.eq("child"), "marker_supported"] = False
    r = resolve(pred).set_index(["cell_id", "level"])
    assert r.loc[("a", "child"), "resolution_status"] == "marker_unsupported"
    assert pd.isna(r.loc[("a", "child"), "assignment"])
    assert r.loc[("a", "child"), "raw_assignment"] == "A"
    pred.loc[pred.level.eq("child"), "assignment"] = "uncalibrated"
    r = resolve(pred).set_index(["cell_id", "level"])
    assert not r.loc[(slice(None), "child"), "accepted"].any()


def test_inconsistent_parent_chain_is_refused() -> None:
    with pytest.raises(ValueError, match="taxonomy parent"):
        resolve_reference_calls(
            objects(),
            predictions(),
            policy=policy(),
            binding=binding(),
            parent_of={("child", "A"): "WRONG"},
        )


def calibration() -> pd.DataFrame:
    return pd.DataFrame(
        [
            (
                f"c{i}",
                "depth_500",
                "parent",
                "P" if i < 100 else "other",
                "P",
                0.95 if i < 100 else 0.5,
                500,
                True,
            )
            for i in range(110)
        ],
        columns=[
            "cell_id",
            "query_id",
            "level",
            "truth",
            "assignment",
            "bp",
            "n_counts",
            "marker_supported",
        ],
    )


def test_rule_selects_supported_coverage_and_is_order_independent() -> None:
    data = calibration()
    result = calibrate_rules(
        data, levels=("parent",), depth_edges=(100,), objective=objective()
    )
    assert result == calibrate_rules(
        data.sample(frac=1, random_state=3),
        levels=("parent",),
        depth_edges=(100,),
        objective=objective(),
    )
    assert len(result) == 1 and result[0].min_bootstrap == 0.95
    assert result[0].n_evaluated == 110 and result[0].n_accepted == 100
    assert result[0].precision == 1.0


def test_repeated_thinnings_are_not_independent_cells() -> None:
    data = calibration().iloc[:1].copy()
    data = pd.concat([data.assign(query_id=f"q{i}") for i in range(100)])
    result = calibrate_rules(
        data, levels=("parent",), depth_edges=(100,), objective=objective()
    )
    assert result[0].n_evaluated == 1 and result[0].min_bootstrap is None


def test_depth_bins_use_lowest_depth_without_borrowing_evidence() -> None:
    first = calibration().iloc[:100].assign(query_id="low", n_counts=100)
    second = first.assign(query_id="high", n_counts=200, bp=1.0)
    rules = calibrate_rules(
        pd.concat([second, first]),
        levels=("parent",),
        depth_edges=(100, 500),
        objective=objective(),
    )
    assert len(rules) == 1 and rules[0].max_counts == 500
    assert rules[0].min_bootstrap == 0.95 and rules[0].n_evaluated == 100


@pytest.mark.parametrize(
    "change",
    ["inconsistent_truth", "duplicate", "negative_depth", "missing_marker_support"],
)
def test_invalid_calibration_is_refused(change: str) -> None:
    data = calibration()
    if change == "inconsistent_truth":
        data = pd.concat(
            [data, data.iloc[:1].assign(query_id="other", truth="different")]
        )
    elif change == "duplicate":
        data = pd.concat([data, data.iloc[:1]])
    elif change == "negative_depth":
        data.loc[0, "n_counts"] = -1
    else:
        data["marker_supported"] = data["marker_supported"].astype(object)
        data.loc[0, "marker_supported"] = None
    with pytest.raises(ValueError):
        calibrate_rules(
            data, levels=("parent",), depth_edges=(100,), objective=objective()
        )


def test_policy_roundtrip_hash_and_forged_evidence_refusal() -> None:
    original = policy()
    assert (
        HippocampalPolicy.model_validate_json(original.model_dump_json()).digest()
        == original.digest()
    )
    forged = original.model_dump()
    forged["rules"][0]["wilson_lower"] = 1.0
    with pytest.raises(ValidationError, match="Wilson bound differs"):
        HippocampalPolicy.model_validate(forged)

"""Tests for the mouse avg_correlation shadow study script (pre-registration §16)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts" / "acceptance"


def _load() -> ModuleType:
    name = "mouse_corr_shadow"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _frame(rows: dict[str, dict[str, float]]) -> pd.DataFrame:
    """One dataset's table: floor -> (coverage, cons, n_removed_pseudo, rem_cons)."""
    records = []
    for floor, values in rows.items():
        records.append(
            {
                "floor": floor,
                "depth_bin": "all",
                "n_table": 10_000,
                "coverage": values["coverage"],
                "cons": values["cons"],
                "n_removed_pseudo": int(values.get("n_rem", 0)),
                "rem_cons": values.get("rem_cons"),
            }
        )
        records.append(
            {
                "floor": floor,
                "depth_bin": "[20,50)",
                "n_table": 2_000,
                "coverage": values.get("bin_coverage", values["coverage"]),
                "cons": values["cons"],
                "n_removed_pseudo": 0,
                "rem_cons": None,
            }
        )
    return pd.DataFrame(records)


GOOD = {
    "none": {"coverage": 0.80, "cons": 0.90},
    "0.40": {"coverage": 0.79, "cons": 0.91, "n_rem": 200, "rem_cons": 0.60},
    "0.50": {"coverage": 0.785, "cons": 0.912, "n_rem": 300, "rem_cons": 0.62},
}


def test_the_ported_expected_classes_follow_the_ag7_report() -> None:
    module = _load()
    assert module.expected_pseudo_for_mmc("30 Astro-Epen", "319 Astro-TE NN") == {
        "Astro"
    }
    assert module.expected_pseudo_for_mmc("01 IT-ET Glut", "007 L2/3 IT CTX Glut") == {
        "Glut"
    }
    assert module.expected_pseudo_for_mmc("21 MB Dopa", "215 SNc-VTA Dopa") == {
        "Glut",
        "GABA",
    }
    assert module.expected_pseudo_for_mmc("32 OEC", "325 OEC NN") == set()
    assert (
        module.cls2("34 Immune") == "Immune" and module.cls2("12 HY GABA") == "Neuron"
    )


def test_selection_prefers_040_unless_050_gains_enough() -> None:
    module = _load()
    selection, verdicts = module.select_floor({"a": _frame(GOOD), "b": _frame(GOOD)})
    assert selection == "0.40"  # 0.50 adds only +.002
    assert verdicts["0.40"]["a"]["admissible"] and verdicts["0.50"]["b"]["admissible"]
    better = {
        **GOOD,
        "0.50": {"coverage": 0.785, "cons": 0.92, "n_rem": 300, "rem_cons": 0.62},
    }
    assert module.select_floor({"a": _frame(better), "b": _frame(better)})[0] == "0.50"


@pytest.mark.parametrize(
    "change",
    [
        {"0.40": {"coverage": 0.79, "cons": 0.903, "n_rem": 200, "rem_cons": 0.60}},
        {"0.40": {"coverage": 0.79, "cons": 0.91, "n_rem": 20, "rem_cons": 0.60}},
        {"0.40": {"coverage": 0.79, "cons": 0.91, "n_rem": 200, "rem_cons": 0.85}},
        {"0.40": {"coverage": 0.77, "cons": 0.91, "n_rem": 200, "rem_cons": 0.60}},
        {
            "0.40": {
                "coverage": 0.79,
                "cons": 0.91,
                "n_rem": 200,
                "rem_cons": 0.60,
                "bin_coverage": 0.70,
            }
        },
    ],
    ids=["gain", "removed_n", "removed_gap", "coverage", "bin_coverage"],
)
def test_each_admissibility_clause_can_reject_a_floor(change: dict) -> None:
    module = _load()
    rows = {**GOOD, **change, "0.50": GOOD["0.40"] | change["0.40"]}
    passed, reasons = module.admissible(_frame(rows), "0.40")
    assert not passed and reasons
    # A floor must be admissible on both datasets.
    selection, _ = module.select_floor({"a": _frame(rows), "b": _frame(GOOD)})
    assert selection == "none"


def test_floor_metrics_count_kept_and_removed_calls() -> None:
    module = _load()
    labels = pd.DataFrame(
        {
            "ct_class_status": [
                "confident",
                "confident",
                "confident",
                "low_confidence",
            ],
            "ct_class_corr": np.array([0.35, 0.45, 0.55, 0.9], dtype=np.float32),
        }
    )
    scored = module.Scored(
        pseudo_confident=np.array([True, True, True, True]),
        consistent=np.array([False, True, True, True]),
        consistent_unpruned=np.array([False, True, True, True]),
    )
    members = np.ones(4, dtype=bool)
    none = module.floor_metrics(labels, scored, None, members)
    assert (none["coverage"], none["cons"]) == (0.75, pytest.approx(2 / 3))
    floored = module.floor_metrics(labels, scored, 0.40, members)
    assert floored["n_confident"] == 2 and floored["cons"] == 1.0
    assert floored["n_removed_pseudo"] == 1 and floored["rem_cons"] == 0.0
    # float32 0.45 at a 0.45 floor is kept (tolerance).
    assert module.floor_metrics(labels, scored, 0.45, members)["n_confident"] == 2

"""Reference validation keeps donor leakage and missing predictions visible."""

import numpy as np
import pandas as pd
import pytest

from merxen.annotation.reference_validation import donor_holdout, score_holdout


def metadata() -> pd.DataFrame:
    """Two training donors and a held-out donor with an unseen class."""
    return pd.DataFrame(
        [
            ("a0", "A", "core", "x"),
            ("a1", "A", "core", "x"),
            ("a2", "A", "core", "unseen"),
            ("a3", "A", "context", "x"),
            ("b0", "B", "core", "x"),
            ("b1", "B", "core", "x"),
            ("b2", "B", "core", "rare"),
            ("c0", "C", "context", "x"),
            ("c1", "C", "context", "x"),
        ],
        columns=["cell_id", "donor", "region", "leaf"],
    )


def test_donor_excluded_from_every_training_region_and_unseen_test_retained() -> None:
    split = donor_holdout(
        metadata(),
        heldout_donor="A",
        training_regions=["core", "context"],
        test_regions=["core"],
    )
    assert set(split.training["cell_id"]) == {"b0", "b1", "c0", "c1"}
    assert set(split.test["cell_id"]) == {"a0", "a1", "a2"}
    assert split.audit.n_dropped_training_leaves == 1
    assert split.audit.training_donors == ("B", "C")


def test_sampling_is_order_independent_and_does_not_cap_training() -> None:
    options = dict(
        heldout_donor="A",
        training_regions=["core", "context"],
        test_regions=["core"],
        max_test_cells_per_group=1,
    )
    first = donor_holdout(metadata(), **options)
    # Duplicate dataframe index values are not duplicate cell IDs.
    reordered = metadata().sample(frac=1, random_state=3)
    reordered.index = [0] * len(reordered)
    second = donor_holdout(reordered, **options)
    assert first.audit == second.audit
    assert len(first.training) == 4 and len(first.test) == 2
    assert "unseen" in set(first.test["leaf"])


@pytest.mark.parametrize("invalid", ["duplicate", "null", "missing", "empty"])
def test_invalid_holdout_keys_or_partitions_are_refused(invalid: str) -> None:
    cells = metadata()
    if invalid == "duplicate":
        cells.loc[1, "cell_id"] = cells.loc[0, "cell_id"]
    elif invalid == "null":
        cells.loc[1, "donor"] = None
    elif invalid == "missing":
        cells = cells.drop(columns="region")
    else:
        cells = cells.loc[cells.donor.ne("A")]
    with pytest.raises(ValueError):
        donor_holdout(
            cells, heldout_donor="A", training_regions=["core"], test_regions=["core"]
        )


def test_accuracy_includes_unseen_truth_and_unassigned_calls() -> None:
    truth = pd.DataFrame(
        {"cell_id": ["a", "b", "c", "d"], "truth": ["X", "X", "Y", "Z"]}
    )
    calls = pd.DataFrame(
        {
            "cell_id": ["d", "b", "a", "c"],
            "assignment": [None, "Y", "X", "Y"],
            "bp": [np.nan, 0.99, 0.9, 0.6],
        }
    )
    result = score_holdout(truth, calls, training_labels=["X", "Y"])
    assert result.n_cells == 4 and result.accuracy == 0.5
    assert result.n_unseen_truth == 1
    assert result.accuracy_seen_truth == pytest.approx(2 / 3)
    assert result.macro_f1_truth_classes == pytest.approx(4 / 9)
    assert result.n_missing_support == 1
    high = next(s for s in result.support if s.threshold == 0.9)
    assert high.n_called == 2 and high.coverage == 0.5 and high.precision == 0.5


def test_support_without_an_assignment_is_not_a_confident_call() -> None:
    truth = pd.DataFrame({"cell_id": ["a", "b"], "truth": ["X", "Y"]})
    calls = pd.DataFrame(
        {"cell_id": ["a", "b"], "assignment": ["X", None], "bp": [0.95, 0.99]}
    )
    result = score_holdout(truth, calls, training_labels=["X", "Y"])
    high = next(s for s in result.support if s.threshold == 0.9)
    assert high.n_called == 1 and high.coverage == 0.5 and high.precision == 1.0
    assert result.accuracy == 0.5


@pytest.mark.parametrize("invalid", ["missing", "extra", "duplicate", "bad_support"])
def test_prediction_integrity_errors_cannot_improve_measured_accuracy(
    invalid: str,
) -> None:
    truth = pd.DataFrame({"cell_id": ["a", "b"], "truth": ["X", "Y"]})
    calls = pd.DataFrame(
        {"cell_id": ["a", "b"], "assignment": ["X", "Y"], "bp": [0.9, 0.9]}
    )
    if invalid == "missing":
        calls = calls.iloc[:1]
    elif invalid == "extra":
        calls.loc[2] = ["c", "X", 0.9]
    elif invalid == "duplicate":
        calls.loc[1, "cell_id"] = "a"
    else:
        calls.loc[1, "bp"] = 1.1
    with pytest.raises(ValueError):
        score_holdout(truth, calls, training_labels=["X", "Y"])

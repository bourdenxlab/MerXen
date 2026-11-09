"""Donor-separated reference validation without inherited panel trust rules.

These helpers support the Atera annotation experiments. Bootstrap support is
reported as support, not assumed to be a calibrated probability of correctness.
All held-out truth labels remain in the accuracy denominator, including labels
absent from training. Test sampling is explicit and independent of row order;
reference training cells are never capped.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HoldoutAudit:
    """Counts and identifiers describing a donor-separated experiment."""

    heldout_donor: str
    training_donors: tuple[str, ...]
    n_training_before_leaf_filter: int
    n_training: int
    n_test_before_sampling: int
    n_test: int
    n_dropped_training_leaves: int
    training_ids_sha256: str
    test_ids_sha256: str


@dataclass
class DonorHoldout:
    """Training metadata, held-out metadata and their audit record."""

    training: pd.DataFrame
    test: pd.DataFrame
    audit: HoldoutAudit


def _ids_hash(values: pd.Series) -> str:
    payload = "\n".join(sorted(values.astype(str)))
    return hashlib.sha256(payload.encode()).hexdigest()


def donor_holdout(
    cells: pd.DataFrame,
    *,
    heldout_donor: str,
    training_regions: Sequence[str],
    test_regions: Sequence[str],
    min_training_cells_per_leaf: int = 2,
    max_test_cells_per_group: int | None = None,
    test_group_column: str = "leaf",
    seed: int = 0,
) -> DonorHoldout:
    """Split metadata by donor before any reference statistics are computed.

    Args:
        cells: Unique ``cell_id`` with ``donor``, ``region`` and ``leaf``.
        heldout_donor: Donor excluded from every training region.
        training_regions: Regions eligible for training.
        test_regions: Regions eligible for held-out evaluation.
        min_training_cells_per_leaf: Minimum reference support per leaf.
        max_test_cells_per_group: Explicit evaluation sample size, or all cells.
        test_group_column: Column used to balance the evaluation sample.
        seed: Seed of the deterministic cell-ID ranking.

    Returns:
        Disjoint training and test frames. Unseen test labels are retained.

    Raises:
        ValueError: For invalid keys, empty partitions or invalid limits.
    """
    required = {"cell_id", "donor", "region", "leaf", test_group_column}
    missing = required - set(cells)
    if missing:
        raise ValueError(f"Missing holdout metadata columns: {sorted(missing)}")
    if cells[list(required)].isna().any().any():
        raise ValueError("Holdout metadata contains null keys")
    if not cells["cell_id"].astype(str).is_unique:
        raise ValueError("Holdout cell IDs must be globally unique")
    cells = cells.reset_index(drop=True)
    if min_training_cells_per_leaf < 1 or (
        max_test_cells_per_group is not None and max_test_cells_per_group < 1
    ):
        raise ValueError("Training and test limits must be positive")
    train = cells.loc[
        cells["donor"].ne(heldout_donor) & cells["region"].isin(training_regions)
    ].copy()
    test = cells.loc[
        cells["donor"].eq(heldout_donor) & cells["region"].isin(test_regions)
    ].copy()
    n_training_before = len(train)
    n_test_before = len(test)
    support = train["leaf"].value_counts()
    dropped = support.index[support < min_training_cells_per_leaf]
    train = train.loc[~train["leaf"].isin(dropped)].copy()
    if max_test_cells_per_group is not None:
        # Ranking by ID hash avoids acquisition-order and dataframe-order bias.
        rank = test["cell_id"].map(
            lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()
        )
        test = test.loc[rank.sort_values(kind="stable").index]
        test = test.groupby(test_group_column, sort=False).head(
            max_test_cells_per_group
        )
    train = train.sort_values("cell_id").reset_index(drop=True)
    test = test.sort_values("cell_id").reset_index(drop=True)
    if train.empty or test.empty:
        raise ValueError("Reference validation needs nonempty training and test sets")
    audit = HoldoutAudit(
        heldout_donor=heldout_donor,
        training_donors=tuple(sorted(train["donor"].astype(str).unique())),
        n_training_before_leaf_filter=n_training_before,
        n_training=len(train),
        n_test_before_sampling=n_test_before,
        n_test=len(test),
        n_dropped_training_leaves=len(dropped),
        training_ids_sha256=_ids_hash(train["cell_id"]),
        test_ids_sha256=_ids_hash(test["cell_id"]),
    )
    return DonorHoldout(train, test, audit)


@dataclass(frozen=True)
class ClassScore:
    """Per-class support and performance, including unseen truth classes."""

    label: str
    n_truth: int
    n_called: int
    n_correct: int
    seen_in_training: bool
    precision: float | None
    recall: float | None
    f1: float


@dataclass(frozen=True)
class SupportScore:
    """Observed precision and coverage at one bootstrap-support threshold."""

    threshold: float
    n_called: int
    coverage: float
    precision: float | None


@dataclass(frozen=True)
class ValidationScore:
    """Full-denominator accuracy and explicitly defined class-balanced metrics."""

    n_cells: int
    accuracy: float
    macro_f1_truth_classes: float
    n_unseen_truth: int
    accuracy_seen_truth: float | None
    n_missing_support: int
    classes: tuple[ClassScore, ...]
    support: tuple[SupportScore, ...]


def score_holdout(
    truth: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    training_labels: Sequence[str],
    support_thresholds: Sequence[float] = (0.5, 0.7, 0.9, 0.95),
) -> ValidationScore:
    """Score a complete one-to-one held-out prediction table.

    Args:
        truth: Unique ``cell_id`` and non-null ``truth`` label per test cell.
        predictions: Unique ``cell_id``, nullable ``assignment`` and ``bp``.
        training_labels: Labels represented in the training taxonomy.
        support_thresholds: Descriptive cuts, not production acceptance gates.

    Returns:
        Accuracy on all test cells, macro F1 over truth-present classes, and
        per-class/support diagnostics. Unassigned predictions count as errors.

    Raises:
        ValueError: If IDs are lost/duplicated or support is outside [0, 1].
    """
    for frame, columns in (
        (truth, {"cell_id", "truth"}),
        (predictions, {"cell_id", "assignment", "bp"}),
    ):
        if not columns <= set(frame):
            raise ValueError(f"Missing score columns: {sorted(columns - set(frame))}")
        if frame["cell_id"].isna().any() or not frame["cell_id"].is_unique:
            raise ValueError("Scoring cell IDs must be unique and non-null")
    if truth.empty or truth["truth"].isna().any():
        raise ValueError("Scoring requires nonempty, non-null truth")
    if set(truth["cell_id"]) != set(predictions["cell_id"]):
        raise ValueError("Prediction IDs must match all held-out truth IDs exactly")
    joined = truth[["cell_id", "truth"]].merge(
        predictions[["cell_id", "assignment", "bp"]],
        on="cell_id",
        validate="one_to_one",
    )
    bp = joined["bp"].to_numpy(dtype=float)
    if np.isinf(bp).any() or ((bp < 0) | (bp > 1)).any():
        raise ValueError("Bootstrap support must be in [0, 1] or missing")
    if any(not 0 <= cut <= 1 for cut in support_thresholds):
        raise ValueError("Support thresholds must be in [0, 1]")
    actual = joined["truth"].astype(str)
    calls = joined["assignment"].astype("string")
    correct = calls.eq(actual).fillna(False).to_numpy(dtype=bool)
    seen = actual.isin(training_labels).to_numpy()
    classes = []
    for label in sorted(set(actual) | set(calls.dropna())):
        is_truth = actual.eq(label).to_numpy()
        is_call = calls.eq(label).fillna(False).to_numpy(dtype=bool)
        n_truth, n_called = int(is_truth.sum()), int(is_call.sum())
        n_correct = int((is_truth & is_call).sum())
        classes.append(
            ClassScore(
                label=label,
                n_truth=n_truth,
                n_called=n_called,
                n_correct=n_correct,
                seen_in_training=label in training_labels,
                precision=n_correct / n_called if n_called else None,
                recall=n_correct / n_truth if n_truth else None,
                f1=2 * n_correct / (n_truth + n_called),
            )
        )
    support = []
    for cut in support_thresholds:
        confident = calls.notna().to_numpy() & np.isfinite(bp) & (bp >= cut)
        n_called = int(confident.sum())
        support.append(
            SupportScore(
                threshold=float(cut),
                n_called=n_called,
                coverage=n_called / len(joined),
                precision=float(correct[confident].mean()) if n_called else None,
            )
        )
    return ValidationScore(
        n_cells=len(joined),
        accuracy=float(correct.mean()),
        macro_f1_truth_classes=float(np.mean([c.f1 for c in classes if c.n_truth])),
        n_unseen_truth=int((~seen).sum()),
        accuracy_seen_truth=float(correct[seen].mean()) if seen.any() else None,
        n_missing_support=int(np.isnan(bp).sum()),
        classes=tuple(classes),
        support=tuple(support),
    )

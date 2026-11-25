"""Select the cells a user trusts from a label table (always-label plan, M5).

Since label table version 2 every cell keeps the labels its own evidence
supports, whatever the dataset gate, the panel trust or real-data QC say
(docs/plans/annotation-always-label-plan.md). What those checks found is in
each cell's metadata instead:

- ``ct_<level>_status``: the cell's own call at that level (``confident``,
  ``low_confidence``, ``below_floor``, ...);
- ``qc_dataset_level``, ``qc_dataset_max_level``, ``qc_dataset_warnings``,
  ``qc_panel_trust`` and the numeric ``qc_<signal>`` columns: the dataset's
  QC, the same for every cell of one sample and segmentation;
- the ``flag_*`` columns: per-cell warnings;
- the ``*_strict`` columns: what an enforced gate would have left (the
  label table version 1 answer).

``select_cells`` combines them into one boolean mask, so analyses state the
label level and the confidence they rely on in one call. It works on a label
table or on the ``obs`` of a clustered table (which carries the same
columns).
"""

from __future__ import annotations

from collections.abc import Collection

import numpy as np
import pandas as pd

from merxen.annotation.schema import CellStatus, Columns
from merxen.annotation.vocab import FINAL_LEVELS

__all__ = ["select_cells"]

CONFIDENT_ONLY: tuple[str, ...] = (CellStatus.CONFIDENT.value,)


def _status_column(labels: pd.DataFrame, level: str, *, strict: bool) -> str:
    column = Columns.strict_status(level) if strict else Columns.level(level, "status")
    if column not in labels.columns:
        raise KeyError(
            f"no {column!r} column: is {level!r} a level of this label table"
            + (" (version 2, with strict columns)?" if strict else "?")
        )
    return column


# The leaf level of each species: a table without a ``species`` column (a
# clustered ``obs``) is told apart by which leaf status column it holds.
_LEAF_LEVELS: dict[str, str] = {"human": "supercluster", "mouse": "subclass"}


def _species_levels(labels: pd.DataFrame) -> tuple[str, ...]:
    if Columns.SPECIES not in labels.columns:
        found = [
            species
            for species, leaf in _LEAF_LEVELS.items()
            if Columns.level(leaf, "status") in labels.columns
        ]
        if len(found) != 1:
            raise KeyError(
                f"no {Columns.SPECIES!r} column, and the leaf status columns do "
                "not tell the species apart"
            )
        return FINAL_LEVELS[found[0]]
    species = {str(value) for value in labels[Columns.SPECIES].dropna().unique()}
    if len(species) != 1 or next(iter(species)) not in FINAL_LEVELS:
        raise ValueError(f"expected one known species, got {sorted(species)}")
    return FINAL_LEVELS[next(iter(species))]


def _within_dataset_support(labels: pd.DataFrame, level: str) -> np.ndarray:
    if Columns.QC_DATASET_MAX_LEVEL not in labels.columns:
        raise KeyError(
            f"no {Columns.QC_DATASET_MAX_LEVEL!r} column (a label table before "
            "version 2 has no dataset QC columns)"
        )
    chain = _species_levels(labels)
    if level not in chain:
        raise ValueError(
            f"within_dataset_support needs a chain level {chain[1:]}, got {level!r}"
        )
    rank = {name: index for index, name in enumerate(chain)}
    supported = labels[Columns.QC_DATASET_MAX_LEVEL].astype(object).map(rank)
    ranks = np.asarray(supported.fillna(-1).to_numpy(dtype=int))
    return np.asarray(ranks >= rank[level], dtype=bool)


def _has_warning(labels: pd.DataFrame, patterns: Collection[str]) -> np.ndarray:
    if Columns.QC_DATASET_WARNINGS not in labels.columns:
        raise KeyError(
            f"no {Columns.QC_DATASET_WARNINGS!r} column (a label table before "
            "version 2 has no dataset QC columns)"
        )
    wanted = [str(pattern) for pattern in patterns]

    def matches(text: object) -> bool:
        if not isinstance(text, str) or not text:
            return False
        return any(
            token == pattern or token.endswith(pattern) or token.startswith(pattern)
            for token in text.split(";")
            for pattern in wanted
        )

    values = labels[Columns.QC_DATASET_WARNINGS].astype(object)
    return np.asarray(values.map(matches).to_numpy(dtype=bool), dtype=bool)


def select_cells(
    labels: pd.DataFrame,
    *,
    level: str,
    statuses: Collection[str] = CONFIDENT_ONLY,
    within_dataset_support: bool = False,
    exclude_dataset_warnings: Collection[str] = (),
    exclude_flags: Collection[str] = (),
    exclude_hard: bool = True,
    strict: bool = False,
) -> pd.Series:
    """Return which cells to keep for an analysis at one label level.

    Example: confident subclass labels, in datasets whose QC supports the
    subclass level and tripped no fail rule, without spill-over::

        keep = select_cells(
            obs,
            level="subclass",
            within_dataset_support=True,
            exclude_dataset_warnings=["_fail"],
            exclude_flags=["flag_microglial_spillover"],
        )
        obs[keep]

    Args:
        labels: A label table or a clustered table's ``obs``.
        level: The label level the analysis uses (``broad``, ``class``,
            ``nt``, ``subclass``, ``lineage``, ``supercluster``,
            ``seaad_subclass``, ...).
        statuses: The ``ct_<level>_status`` values to keep (default
            ``confident`` only).
        within_dataset_support: Keep only cells of datasets whose QC supports
            ``level`` (``level`` no deeper than ``qc_dataset_max_level``).
        exclude_dataset_warnings: Drop the cells of datasets with a matching
            ``qc_dataset_warnings`` token. A pattern matches a token equal
            to it, ending with it (``"_fail"``: any fail rule) or starting
            with it (``"g1_"``: any G1 token).
        exclude_flags: Boolean ``flag_*`` columns whose true values drop a
            cell; a missing value (flag not evaluated) keeps it.
        exclude_hard: Drop ``exclude_hard`` cells (off-table, or implausible
            at every level they were called).
        strict: Read the ``*_strict`` columns: the cells an enforced dataset
            gate, panel trust and real-data QC would have kept (label table
            version 1 behaviour).

    Returns:
        A boolean mask aligned with ``labels.index``.

    Raises:
        KeyError: If a needed column is missing.
        ValueError: If ``within_dataset_support`` is given a level outside the
            species' chain, or the table holds several species.
    """
    status = labels[_status_column(labels, level, strict=strict)].astype(object)
    keep = status.isin([str(value) for value in statuses]).to_numpy(dtype=bool)
    if within_dataset_support:
        keep &= _within_dataset_support(labels, level)
    if exclude_dataset_warnings:
        keep &= ~_has_warning(labels, exclude_dataset_warnings)
    for flag in exclude_flags:
        if flag not in labels.columns:
            raise KeyError(f"no {flag!r} column")
        values = pd.Series(labels[flag]).astype("boolean").fillna(False)
        keep &= ~values.to_numpy(dtype=bool)
    if exclude_hard:
        column = Columns.EXCLUDE_HARD_STRICT if strict else Columns.EXCLUDE_HARD
        if column not in labels.columns:
            raise KeyError(f"no {column!r} column")
        keep &= ~labels[column].to_numpy(dtype=bool)
    return pd.Series(keep, index=labels.index, name="keep")

"""Mouse AP axis: AP bins, the AP estimate and out-of-sample region tables.

Plan §7.8 (M6b):

- **AP bins** of the coronal scope (AP 2.4–10.4 mm, CCF ``x_ccf``):
  [2.4, 3.0), [3.0, 4.5), [4.5, 6.0), [6.0, 7.5), [7.5, 9.0), [9.0, 10.4]
  (``ap_bin``), the bins of the per-AP-bin harm table (§7.8, MO10).
- **AP estimate** (§7.8 item 4; E7 §5 step 1, prototyped in
  ``data/mouse_vzg2/scripts/merfish_section_match.py`` and
  ``data/mouse_ag7/scripts/ag7_sections_extra.py``): the neuron-subclass
  composition of a section's confident neurons is matched to every
  MERFISH-C57BL6J-638850 section of the ``wmb_region_share`` bundle
  (``section_composition.parquet``, subclass rows) by the Bhattacharyya
  coefficient; the best-matching section gives the AP estimate and its AP
  bin, and it and its immediate neighbours in the AP order of all sections
  are G4's composition window (``estimate_ap``).
- **Leave-neighbours-out region tables** (§7.8 item 1): the division shares,
  cell numbers and homes the region rule reads, recomputed from the MERFISH
  CCF view without a section and its immediate neighbours (± 1 section in
  the AP order of all sections; one neighbour at either end), plus the
  structure-level subclass homes of variant (a) (``MerfishRegionCounts``).
  With nothing excluded the division tables equal the bundle's
  (``reference.region_share_tables``).

Imports: numpy, pandas and the annotation modules.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pandas as pd

from merxen.annotation.mouse_regions import RegionShares
from merxen.annotation.reference import GREY_REGIONS

logger = logging.getLogger(__name__)

# Plan §7.8 AP bins (mm); the last bin is closed at the scope's end.
AP_BIN_EDGES_MM: Final[tuple[float, ...]] = (2.4, 3.0, 4.5, 6.0, 7.5, 9.0, 10.4)
AP_BIN_LABELS: Final[tuple[str, ...]] = (
    "2.4-3.0",
    "3.0-4.5",
    "4.5-6.0",
    "6.0-7.5",
    "7.5-9.0",
    "9.0-10.4",
)
# Non-neuronal WMB subclass names end with " NN" (classes 30-34).
NON_NEURONAL_SUFFIX: Final = " NN"
# The AP estimate needs at least this many confident neurons.
AP_ESTIMATE_MIN_CELLS: Final = 100
AP_ESTIMATE_TOP_K: Final = 5
# Softmax temperature of the top-k weighted AP (the ag7 prototype's 0.01).
AP_ESTIMATE_TEMPERATURE: Final = 0.01
WINDOW_HALF_WIDTH: Final = 1
# CCF-view columns the region tables read.
SECTION_COLUMN: Final = "brain_section_label"
STRUCTURE_COLUMN: Final = "parcellation_structure"


class ApEstimateError(ValueError):
    """The AP estimate cannot be computed from its inputs."""


# --------------------------------------------------------------------------
# AP bins and section order


def ap_bin(ap_mm: float | None) -> str | None:
    """Return the plan §7.8 AP bin of an AP position.

    Args:
        ap_mm: CCF AP (``x_ccf``, mm).

    Returns:
        The bin label (``AP_BIN_LABELS``), or ``None`` outside AP 2.4–10.4 mm
        (or for a missing value).
    """
    if ap_mm is None or not math.isfinite(float(ap_mm)):
        return None
    value = float(ap_mm)
    edges = AP_BIN_EDGES_MM
    if value < edges[0] or value > edges[-1]:
        return None
    for index in range(len(edges) - 1):
        if value < edges[index + 1]:
            return AP_BIN_LABELS[index]
    return AP_BIN_LABELS[-1]


def section_order(section_ap: Mapping[str, float] | pd.Series) -> tuple[str, ...]:
    """Return sections in AP order (ties: by label).

    Args:
        section_ap: AP (mm) per section label.

    Returns:
        The labels, anterior first.
    """
    items = [(float(ap), str(section)) for section, ap in dict(section_ap).items()]
    return tuple(section for _, section in sorted(items))


def neighbours_out(
    order: Sequence[str], section: str, *, half_width: int = WINDOW_HALF_WIDTH
) -> tuple[str, ...]:
    """Return a section and its neighbours in the AP order (leave-out set).

    Args:
        order: All sections in AP order (``section_order``).
        section: The held-out section.
        half_width: Neighbours on each side (1: ± 1 section; one neighbour at
            either end of the series).

    Returns:
        The excluded sections, in AP order.

    Raises:
        KeyError: If the section is not in the order.
    """
    sections = list(order)
    if section not in sections:
        raise KeyError(f"section {section!r} is not in the AP order")
    index = sections.index(section)
    return tuple(sections[max(0, index - half_width) : index + half_width + 1])


def section_steps(order: Sequence[str], section_ap: Mapping[str, float]) -> pd.Series:
    """Return each section's section step: its largest AP gap to a neighbour.

    Args:
        order: Sections in AP order.
        section_ap: AP per section.

    Returns:
        Step (mm) per section (the gap to its only neighbour at the ends).
    """
    ap = [float(section_ap[section]) for section in order]
    steps = {}
    for index, section in enumerate(order):
        gaps = []
        if index > 0:
            gaps.append(ap[index] - ap[index - 1])
        if index + 1 < len(ap):
            gaps.append(ap[index + 1] - ap[index])
        steps[section] = max(gaps) if gaps else 0.0
    return pd.Series(steps, dtype=np.float64)


# --------------------------------------------------------------------------
# AP estimate (composition matching; §7.8 item 4)


@dataclass(frozen=True)
class ApEstimate:
    """The AP estimate of one query section.

    Attributes:
        section: Best-matching MERFISH section.
        ap_mm: Its AP (the estimate).
        similarity: Its Bhattacharyya coefficient (1 = identical).
        weighted_ap_mm: AP averaged over the top ``top_k`` sections with
            softmax weights (temperature 0.01; diagnostic).
        ranking: Top sections as ``(section, ap_mm, similarity)``.
        n_cells: Query cells matched (confident neurons with a subclass).
        n_sections: MERFISH sections compared.
        ap_bin: The AP bin of ``ap_mm`` (``None`` outside the scope).
        in_scope: Whether ``ap_mm`` lies in the coronal scope.
        window_sections: The best section and its neighbours in the AP order
            of all compared sections (G4's window).
    """

    section: str
    ap_mm: float
    similarity: float
    weighted_ap_mm: float
    ranking: tuple[tuple[str, float, float], ...]
    n_cells: int
    n_sections: int
    ap_bin: str | None
    in_scope: bool
    window_sections: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        """Return the estimate as JSON (gate signals, reports)."""
        return {
            "section": self.section,
            "ap_mm": round(self.ap_mm, 4),
            "similarity": round(self.similarity, 6),
            "weighted_ap_mm": round(self.weighted_ap_mm, 4),
            "ranking": [
                {"section": section, "ap_mm": round(ap, 4), "similarity": round(bc, 6)}
                for section, ap, bc in self.ranking
            ],
            "n_cells": self.n_cells,
            "n_sections": self.n_sections,
            "ap_bin": self.ap_bin,
            "in_scope": self.in_scope,
            "window_sections": list(self.window_sections),
        }


def is_neuronal_subclass(name: object) -> bool:
    """Return whether a WMB subclass name is neuronal (no `` NN`` suffix)."""
    return name is not None and not str(name).endswith(NON_NEURONAL_SUFFIX)


def reference_compositions(
    section_composition: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return the neuron-subclass counts and AP of every MERFISH section.

    Args:
        section_composition: ``section_composition.parquet`` of the
            region-share bundle (section, ap_ccf_mm, n_cells, level,
            node_name, n, freq).

    Returns:
        ``(counts, ap)``: section x neuronal subclass counts, AP per section.

    Raises:
        ApEstimateError: If the table has no subclass rows.
    """
    needed = {"section", "ap_ccf_mm", "level", "node_name", "n"}
    missing = sorted(needed - set(section_composition.columns))
    if missing:
        raise ApEstimateError(f"section_composition lacks columns {missing}")
    rows = section_composition[section_composition["level"].astype(str) == "subclass"]
    if rows.empty:
        raise ApEstimateError("section_composition has no subclass rows")
    names = rows["node_name"].astype(str)
    rows = rows[names.map(is_neuronal_subclass).to_numpy(dtype=bool)]
    counts = (
        rows.assign(
            section=rows["section"].astype(str), node=rows["node_name"].astype(str)
        )
        .pivot_table(index="section", columns="node", values="n", aggfunc="sum")
        .fillna(0.0)
        .astype(np.float64)
    )
    ap = (
        section_composition.assign(section=section_composition["section"].astype(str))
        .groupby("section")["ap_ccf_mm"]
        .first()
        .astype(np.float64)
    )
    return counts, ap.reindex(counts.index)


def bhattacharyya(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return the Bhattacharyya coefficient of a composition vs each row.

    Args:
        query: Non-negative counts (``m``).
        reference: Non-negative counts (``k x m``).

    Returns:
        ``k`` coefficients in [0, 1] (0 for an empty row).
    """
    q = np.asarray(query, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    q_total = q.sum()
    if q_total <= 0:
        raise ApEstimateError("the query composition is empty")
    totals = r.sum(axis=1, keepdims=True)
    safe = np.where(totals > 0, totals, 1.0)
    result: np.ndarray = np.sqrt(r / safe) @ np.sqrt(q / q_total)
    return np.where(totals[:, 0] > 0, result, 0.0)


def estimate_ap(
    query_subclasses: Iterable[object],
    section_composition: pd.DataFrame,
    *,
    exclude_sections: Iterable[str] = (),
    top_k: int = AP_ESTIMATE_TOP_K,
    temperature: float = AP_ESTIMATE_TEMPERATURE,
    window_half_width: int = WINDOW_HALF_WIDTH,
    scope_mm: tuple[float, float] = (AP_BIN_EDGES_MM[0], AP_BIN_EDGES_MM[-1]),
    min_cells: int = AP_ESTIMATE_MIN_CELLS,
) -> ApEstimate:
    """Estimate a section's AP position by composition matching (§7.8 item 4).

    The query composition is the subclass counts of the given cells (the
    caller passes its confident neurons; non-neuronal subclass names are
    ignored); each MERFISH section's composition is its neuronal subclass
    counts over all its cells. The best Bhattacharyya coefficient chooses
    the section (ties: the first in AP order).

    Args:
        query_subclasses: Subclass name per query cell.
        section_composition: The bundle's ``section_composition`` table.
        exclude_sections: MERFISH sections left out of the comparison (held-
            out validation).
        top_k: Sections of the weighted AP.
        temperature: Softmax temperature of the weighted AP.
        window_half_width: Neighbours on each side in G4's window.
        scope_mm: The coronal scope ``(min, max)`` (``ap_scope_*_mm``).
        min_cells: Fewer query cells refuse the estimate.

    Returns:
        The estimate.

    Raises:
        ApEstimateError: With fewer than ``min_cells`` query neurons or no
            reference section.
    """
    names = [str(name) for name in query_subclasses if is_neuronal_subclass(name)]
    names = [name for name in names if name not in ("nan", "None")]
    if len(names) < min_cells:
        raise ApEstimateError(
            f"{len(names)} confident neurons (< {min_cells}): no AP estimate"
        )
    counts, ap = reference_compositions(section_composition)
    excluded = {str(section) for section in exclude_sections}
    keep = [section for section in counts.index if section not in excluded]
    if not keep:
        raise ApEstimateError("no MERFISH section left to compare")
    counts = counts.loc[keep]
    ap = ap.loc[keep]
    order = section_order(ap)
    counts = counts.loc[list(order)]
    query = pd.Series(names).value_counts()
    columns = counts.columns.union(query.index)
    reference = counts.reindex(columns=columns, fill_value=0.0).to_numpy()
    vector = query.reindex(columns, fill_value=0).to_numpy(np.float64)
    similarity = bhattacharyya(vector, reference)
    best = int(np.argmax(similarity))
    best_section = order[best]
    ranked = np.argsort(-similarity, kind="stable")[: max(1, top_k)]
    weights = np.exp((similarity[ranked] - similarity[best]) / temperature)
    ap_values = ap.loc[list(order)].to_numpy(np.float64)
    weighted = float(np.sum(weights * ap_values[ranked]) / np.sum(weights))
    best_ap = float(ap_values[best])
    window = order[max(0, best - window_half_width) : best + window_half_width + 1]
    return ApEstimate(
        section=best_section,
        ap_mm=best_ap,
        similarity=float(similarity[best]),
        weighted_ap_mm=weighted,
        ranking=tuple(
            (order[index], float(ap_values[index]), float(similarity[index]))
            for index in ranked
        ),
        n_cells=len(names),
        n_sections=len(order),
        ap_bin=ap_bin(best_ap),
        in_scope=scope_mm[0] <= best_ap <= scope_mm[1],
        window_sections=tuple(window),
    )


# --------------------------------------------------------------------------
# Leave-neighbours-out region tables (§7.8 item 1)


@dataclass(frozen=True)
class MerfishRegionCounts:
    """Per-section MERFISH counts behind the region tables.

    Attributes:
        grey: Grey-matter cells per (level, section, node, region) (the E7
            v2 regions, ``reference.merfish_region``).
        structure: Grey-matter cells per (section, subclass, structure).
        all_cells: All cells per (level, section, node).
        section_ap: Median ``x_ccf`` per section.
    """

    grey: pd.Series
    structure: pd.Series
    all_cells: pd.Series
    section_ap: pd.Series

    @classmethod
    def from_cells(cls, cells: pd.DataFrame) -> MerfishRegionCounts:
        """Count the cells of the MERFISH CCF view.

        Args:
            cells: One row per cell with ``brain_section_label``, ``class``,
                ``subclass``, ``x_ccf``, ``region`` (``merfish_region``) and
                ``parcellation_structure``.

        Returns:
            The counts.

        Raises:
            KeyError: If a column is missing.
        """
        needed = {SECTION_COLUMN, "class", "subclass", "x_ccf", "region"}
        missing = sorted(needed - set(cells.columns))
        if missing:
            raise KeyError(f"MERFISH cells lack columns {missing}")
        frame = pd.DataFrame(
            {
                "section": cells[SECTION_COLUMN].astype(str).to_numpy(),
                "class": cells["class"].astype(str).to_numpy(),
                "subclass": cells["subclass"].astype(str).to_numpy(),
                "region": cells["region"].astype(str).to_numpy(),
                "structure": (
                    cells[STRUCTURE_COLUMN].astype(str).to_numpy()
                    if STRUCTURE_COLUMN in cells
                    else np.full(len(cells), "unknown", dtype=object)
                ),
            }
        )
        is_grey = frame["region"].isin(GREY_REGIONS).to_numpy()
        grey = frame[is_grey]
        grey_parts = []
        all_parts = []
        for level in ("class", "subclass"):
            counts = grey.groupby(["section", level, "region"]).size()
            counts.index = counts.index.set_names(["section", "node", "region"])
            grey_parts.append(pd.concat({level: counts}, names=["level"]))
            totals = frame.groupby(["section", level]).size()
            totals.index = totals.index.set_names(["section", "node"])
            all_parts.append(pd.concat({level: totals}, names=["level"]))
        structure = grey.groupby(["section", "subclass", "structure"]).size()
        structure.index = structure.index.set_names(["section", "node", "structure"])
        section_ap = (
            pd.Series(cells["x_ccf"].to_numpy(np.float64), index=frame["section"])
            .groupby(level=0)
            .median()
        )
        return cls(
            grey=pd.concat(grey_parts).astype(np.int64),
            structure=structure.astype(np.int64),
            all_cells=pd.concat(all_parts).astype(np.int64),
            section_ap=section_ap,
        )

    @property
    def order(self) -> tuple[str, ...]:
        """All sections in AP order."""
        return section_order(self.section_ap)

    def neighbours_out(self, section: str) -> tuple[str, ...]:
        """Return a section and its ± 1 neighbours (the leave-out set)."""
        return neighbours_out(self.order, section)

    def tables(self, exclude: Iterable[str] = ()) -> dict[str, pd.DataFrame]:
        """Return the region tables without the cells of some sections.

        With nothing excluded, ``region_share`` and ``region_home`` equal
        ``reference.region_share_tables`` of the same cells.

        Args:
            exclude: Sections left out.

        Returns:
            ``region_share`` (level, node_name, region, n, share, n_grey,
            n_all), ``region_home`` (level, node_name, home, home_share,
            n_grey) and ``structure_home`` (subclass rows; the structure with
            the most grey cells, ties to the first in sorted order).
        """
        excluded = {str(section) for section in exclude}
        share_frames: list[pd.DataFrame] = []
        home_frames: list[pd.DataFrame] = []
        for level in ("class", "subclass"):
            grey = self.grey.xs(level, level="level")
            kept = ~grey.index.get_level_values("section").isin(excluded)
            counts = (
                grey[kept]
                .groupby(level=["node", "region"])
                .sum()
                .unstack(fill_value=0)
                .reindex(columns=list(GREY_REGIONS), fill_value=0)
                .astype(np.int64)
            )
            counts = counts[counts.sum(axis=1) > 0].sort_index()
            counts.index = counts.index.rename(level)
            counts.columns = counts.columns.rename("region")
            everything = self.all_cells.xs(level, level="level")
            all_kept = ~everything.index.get_level_values("section").isin(excluded)
            n_all = everything[all_kept].groupby(level="node").sum()
            n_grey = counts.sum(axis=1)
            shares = counts.div(n_grey.replace(0, np.nan), axis=0).fillna(0.0)
            long = (
                counts.stack()
                .rename("n")
                .reset_index()
                .rename(columns={level: "node_name"})
            )
            long["share"] = shares.stack().to_numpy()
            long["n_grey"] = long["node_name"].map(n_grey).astype(int)
            long["n_all"] = long["node_name"].map(n_all).fillna(0).astype(int)
            long.insert(0, "level", level)
            share_frames.append(long)
            home_frames.append(
                pd.DataFrame(
                    {
                        "level": level,
                        "node_name": shares.index.astype(str),
                        "home": shares.idxmax(axis=1).to_numpy(),
                        "home_share": shares.max(axis=1).to_numpy(),
                        "n_grey": n_grey.to_numpy(),
                    }
                )
            )
        kept = ~self.structure.index.get_level_values("section").isin(excluded)
        structure = (
            self.structure[kept]
            .groupby(level=["node", "structure"])
            .sum()
            .unstack(fill_value=0)
        )
        structure = structure[structure.sum(axis=1) > 0].sort_index()
        structure = structure.reindex(columns=sorted(structure.columns))
        totals = structure.sum(axis=1)
        values = structure.to_numpy()
        best = values.argmax(axis=1)
        structure_home = pd.DataFrame(
            {
                "level": "subclass",
                "node_name": structure.index.astype(str),
                "home": np.asarray(structure.columns, dtype=object)[best],
                "home_share": values[np.arange(len(values)), best] / totals.to_numpy(),
                "n_grey": totals.to_numpy(np.int64),
            }
        )
        return {
            "region_share": pd.concat(share_frames, ignore_index=True),
            "region_home": pd.concat(home_frames, ignore_index=True),
            "structure_home": structure_home,
        }

    def shares(self, exclude: Iterable[str] = ()) -> RegionShares:
        """Return the region shares the rule reads, without some sections.

        Args:
            exclude: Sections left out (``neighbours_out``).

        Returns:
            Division shares, cell numbers, division homes and structure
            homes.
        """
        tables = self.tables(exclude)
        return RegionShares.from_tables(
            tables["region_share"], tables["region_home"], tables["structure_home"]
        )

    def section_region_shares(self, section: str) -> pd.Series:
        """Return a section's grey-matter cell share per division.

        Args:
            section: A section label.

        Returns:
            Share per division (``GREY_REGIONS`` with cells in the section).
        """
        grey = self.grey.xs("class", level="level")
        rows = grey[grey.index.get_level_values("section") == str(section)]
        by_region = rows.groupby(level="region").sum()
        total = by_region.sum()
        if total <= 0:
            return pd.Series(dtype=np.float64)
        return (by_region / total).reindex(
            [region for region in GREY_REGIONS if region in by_region.index]
        )

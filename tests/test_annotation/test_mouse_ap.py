"""Tests for the mouse AP axis: bins, AP estimate, leave-neighbours-out tables."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from merxen.annotation.mouse_ap import (
    AP_BIN_LABELS,
    ApEstimateError,
    MerfishRegionCounts,
    ap_bin,
    bhattacharyya,
    estimate_ap,
    is_neuronal_subclass,
    neighbours_out,
    reference_compositions,
    section_order,
    section_steps,
)
from merxen.annotation.reference import region_share_tables

# --------------------------------------------------------------------------
# AP bins and the section order


@pytest.mark.parametrize(
    ("ap", "expected"),
    [
        (2.4, "2.4-3.0"),
        (2.999, "2.4-3.0"),
        (3.0, "3.0-4.5"),
        (4.49, "3.0-4.5"),
        (6.0, "6.0-7.5"),
        (9.0, "9.0-10.4"),
        (10.37, "9.0-10.4"),
        (10.4, "9.0-10.4"),
        (10.41, None),
        (2.39, None),
        (float("nan"), None),
        (None, None),
    ],
)
def test_ap_bins_follow_the_plan(ap: float | None, expected: str | None) -> None:
    assert ap_bin(ap) == expected


def test_ap_bin_labels_cover_the_scope() -> None:
    assert AP_BIN_LABELS[0].startswith("2.4") and AP_BIN_LABELS[-1].endswith("10.4")
    assert len(AP_BIN_LABELS) == 6


def test_neighbours_out_is_the_section_and_its_neighbours() -> None:
    order = section_order({"s3": 3.0, "s1": 1.0, "s2": 2.0, "s4": 4.5, "s0": 0.2})
    assert order == ("s0", "s1", "s2", "s3", "s4")
    assert neighbours_out(order, "s2") == ("s1", "s2", "s3")
    assert neighbours_out(order, "s0") == ("s0", "s1")  # one neighbour at an end
    assert neighbours_out(order, "s4") == ("s3", "s4")
    assert neighbours_out(order, "s2", half_width=2) == order
    with pytest.raises(KeyError):
        neighbours_out(order, "s9")
    steps = section_steps(
        order, {"s0": 0.2, "s1": 1.0, "s2": 2.0, "s3": 3.0, "s4": 4.5}
    )
    assert steps["s0"] == pytest.approx(0.8)
    assert steps["s3"] == pytest.approx(1.5)
    assert steps["s2"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Leave-neighbours-out tables


def _cells() -> pd.DataFrame:
    """Five sections along AP; one subclass only in sections 2 and 3."""
    rows = []

    def add(
        section: str,
        ap: float,
        klass: str,
        sub: str,
        region: str,
        n: int,
        structure: str = "X",
    ) -> None:
        for _ in range(n):
            rows.append(
                {
                    "brain_section_label": section,
                    "x_ccf": ap,
                    "class": klass,
                    "subclass": sub,
                    "region": region,
                    "parcellation_structure": structure,
                }
            )

    for index, ap in enumerate((1.0, 2.0, 3.0, 4.0, 5.0)):
        section = f"S{index}"
        add(section, ap, "01 A Glut", "001 A1 Glut", "Isocortex", 10 + index, "MOp")
        add(section, ap, "01 A Glut", "002 A2 Glut", "OLF", 5, "AON")
        add(section, ap, "01 A Glut", "002 A2 Glut", "STR", 2 + index, "CP")
        add(section, ap, "30 Astro-Epen", "300 Astro NN", "fibre", 3)
        add(section, ap, "30 Astro-Epen", "300 Astro NN", "STR", 4)
    add("S2", 3.0, "05 OB-IMN GABA", "050 OB Gaba", "OB", 7, "MOB")
    add("S3", 4.0, "05 OB-IMN GABA", "050 OB Gaba", "OB", 3, "MOB")
    add("S3", 4.0, "05 OB-IMN GABA", "050 OB Gaba", "fibre", 2, "lfbs")
    return pd.DataFrame(rows)


def _long(table: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    frame = table.copy()
    for key in keys:
        frame[key] = frame[key].astype(str)
    return frame.sort_values(keys).reset_index(drop=True)


@pytest.mark.parametrize("exclude", [(), ("S1", "S2", "S3"), ("S3", "S4")])
def test_counts_reproduce_region_share_tables(exclude: tuple[str, ...]) -> None:
    cells = _cells()
    counts = MerfishRegionCounts.from_cells(cells)
    ours = counts.tables(exclude)
    theirs = region_share_tables(cells[~cells["brain_section_label"].isin(exclude)])
    share_keys = ["level", "node_name", "region"]
    left = _long(ours["region_share"], share_keys)
    right = _long(theirs["region_share"], share_keys)[left.columns]
    pd.testing.assert_frame_equal(left, right, check_dtype=False)
    home_keys = ["level", "node_name"]
    pd.testing.assert_frame_equal(
        _long(ours["region_home"], home_keys),
        _long(theirs["region_home"], home_keys)[ours["region_home"].columns],
        check_dtype=False,
    )


def test_leave_neighbours_out_excludes_the_section_and_its_neighbours() -> None:
    counts = MerfishRegionCounts.from_cells(_cells())
    assert counts.order == ("S0", "S1", "S2", "S3", "S4")
    assert counts.neighbours_out("S2") == ("S1", "S2", "S3")
    # 050 OB Gaba lives only in S2 and S3: gone when S2 is held out.
    held = counts.shares(counts.neighbours_out("S2"))
    assert "050 OB Gaba" not in held.subclass_n_grey.index
    assert "05 OB-IMN GABA" not in held.class_share.index
    assert "050 OB Gaba" not in held.subclass_home.index
    # Held out S4 (neighbour S3): S2's seven OB cells remain.
    tail = counts.shares(counts.neighbours_out("S4"))
    assert tail.subclass_n_grey["050 OB Gaba"] == 7
    assert tail.subclass_share.loc["050 OB Gaba", "OB"] == 1.0
    # Every count drops by exactly the excluded sections' cells.
    full = counts.shares(())
    assert full.subclass_n_grey["001 A1 Glut"] == 10 + 11 + 12 + 13 + 14
    assert held.subclass_n_grey["001 A1 Glut"] == 10 + 14
    # S0 and S4 remain: OLF 5 + 5, STR 2 + 6.
    assert held.subclass_share.loc["002 A2 Glut", "OLF"] == pytest.approx(10 / 18)
    # Only grey-matter cells count (the fibre Astro-Epen cells do not).
    assert full.subclass_n_grey["300 Astro NN"] == 20


def test_structure_homes_follow_the_grey_cells_and_ties_sort() -> None:
    cells = _cells()
    counts = MerfishRegionCounts.from_cells(cells)
    homes = counts.tables(())["structure_home"].set_index("node_name")
    assert homes.loc["001 A1 Glut", "home"] == "MOp"
    # 002 A2: 25 AON vs 20 CP cells over all sections.
    assert homes.loc["002 A2 Glut", "home"] == "AON"
    assert homes.loc["002 A2 Glut", "home_share"] == pytest.approx(25 / 45)
    # Leaving S0-S1 out: 15 AON vs 15 CP -> tie -> "AON" < "CP".
    tied = counts.tables(("S0", "S1"))["structure_home"].set_index("node_name")
    assert tied.loc["002 A2 Glut", "home"] == "AON"
    # The fibre-tract OB cells do not vote for "lfbs".
    assert homes.loc["050 OB Gaba", "home"] == "MOB"
    shares = counts.shares(())
    assert shares.subclass_structure_home is not None
    assert shares.subclass_structure_home["002 A2 Glut"] == "AON"


def test_section_region_shares() -> None:
    counts = MerfishRegionCounts.from_cells(_cells())
    shares = counts.section_region_shares("S2")
    # Class-level grey cells of S2: 12 Isocortex, 5 OLF, 4 + 4 STR, 7 OB.
    assert shares["OB"] == pytest.approx(7 / 32)
    assert shares.sum() == pytest.approx(1.0)
    assert counts.section_region_shares("nope").empty


# --------------------------------------------------------------------------
# AP estimate on held-out MERFISH-like sections


def _series(n_sections: int = 25, cells: int = 4000, seed: int = 0) -> pd.DataFrame:
    """Sections along AP (about 0.4 mm apart) whose neuron mix drifts.

    120 neuron subclasses with Gaussian AP profiles (sd 0.5 mm) and one
    non-neuronal subclass, sampled per section like a MERFISH section.
    """
    rng = np.random.default_rng(seed)
    subclasses = [f"{index:03d} T{index} Glut" for index in range(120)]
    centres = np.linspace(0.5, 11.5, len(subclasses))
    ap = np.linspace(1.0, 10.6, n_sections) + rng.uniform(-0.05, 0.05, n_sections)
    rows = []
    for index, position in enumerate(ap):
        weights = np.exp(-0.5 * ((centres - position) / 0.5) ** 2) + 1e-4
        draws = rng.multinomial(cells, weights / weights.sum())
        section = f"C57BL6J-638850.{70 - index}"
        for name, n in zip(subclasses, draws, strict=True):
            if n:
                rows.append((section, position, name, int(n)))
        rows.append((section, position, "319 Astro-TE NN", cells // 2))
    frame = pd.DataFrame(rows, columns=["section", "ap_ccf_mm", "node_name", "n"])
    frame["level"] = "subclass"
    frame["n_cells"] = frame.groupby("section")["n"].transform("sum")
    frame["freq"] = frame["n"] / frame["n_cells"]
    return frame


def _query(frame: pd.DataFrame, section: str, cells: int, seed: int) -> list[str]:
    rows = frame[frame["section"] == section]
    rng = np.random.default_rng(seed)
    probabilities = rows["n"].to_numpy(float) / rows["n"].sum()
    draws = rng.multinomial(cells, probabilities)
    return list(np.repeat(rows["node_name"].to_numpy(), draws))


def test_ap_estimate_on_held_out_sections_is_within_one_step() -> None:
    frame = _series()
    counts, ap = reference_compositions(frame)
    order = section_order(ap)
    steps = section_steps(order, ap)
    for index, section in enumerate(order):
        query = _query(frame, section, 3000, seed=index)
        estimate = estimate_ap(query, frame, exclude_sections=[section])
        assert estimate.section != section
        assert abs(estimate.ap_mm - ap[section]) <= steps[section] + 1e-9, section
        assert estimate.n_sections == len(order) - 1
        assert section not in estimate.window_sections


def test_ap_estimate_finds_the_section_itself_when_not_held_out() -> None:
    frame = _series()
    _, ap = reference_compositions(frame)
    order = section_order(ap)
    section = order[10]
    estimate = estimate_ap(_query(frame, section, 5000, seed=1), frame)
    assert estimate.section == section
    assert estimate.ap_mm == pytest.approx(ap[section])
    assert estimate.window_sections == order[9:12]
    assert estimate.ranking[0][0] == section
    assert len(estimate.ranking) == 5
    assert estimate.ap_bin == ap_bin(ap[section])
    assert estimate.in_scope == (2.4 <= ap[section] <= 10.4)
    assert abs(estimate.weighted_ap_mm - ap[section]) < 1.0
    payload = estimate.to_json()
    assert payload["section"] == section and payload["window_sections"] == list(
        order[9:12]
    )


def test_ap_estimate_window_uses_all_sections_and_edges() -> None:
    frame = _series()
    _, ap = reference_compositions(frame)
    order = section_order(ap)
    first = estimate_ap(_query(frame, order[0], 5000, seed=2), frame)
    assert first.section == order[0]
    assert first.window_sections == order[:2]
    assert not first.in_scope or ap[order[0]] >= 2.4


def test_ap_estimate_ignores_non_neurons_and_needs_cells() -> None:
    frame = _series()
    _, ap = reference_compositions(frame)
    section = section_order(ap)[5]
    query = _query(frame, section, 2000, seed=3)
    neurons = [name for name in query if is_neuronal_subclass(name)]
    with_glia = estimate_ap(query, frame)
    assert with_glia.n_cells == len(neurons)
    assert with_glia.section == estimate_ap(neurons, frame).section
    with pytest.raises(ApEstimateError, match="confident neurons"):
        estimate_ap(neurons[:50], frame)
    with pytest.raises(ApEstimateError, match="no MERFISH section"):
        estimate_ap(neurons, frame, exclude_sections=list(ap.index))
    with pytest.raises(ApEstimateError, match="subclass rows"):
        estimate_ap(neurons, frame.assign(level="class"))
    assert not is_neuronal_subclass("319 Astro-TE NN")
    assert is_neuronal_subclass("007 L2/3 IT CTX Glut")
    assert not is_neuronal_subclass(None)


def test_bhattacharyya_coefficient() -> None:
    reference = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    result = bhattacharyya(np.array([2.0, 2.0]), reference)
    assert result[0] == pytest.approx(math.sqrt(0.5))
    assert result[1] == pytest.approx(1.0)
    assert result[2] == 0.0
    with pytest.raises(ApEstimateError):
        bhattacharyya(np.zeros(2), reference)

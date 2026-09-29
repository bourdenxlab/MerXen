"""Tests for the soft composition estimator and its block-bootstrap CIs (§5.5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from merxen.annotation import composition as comp
from merxen.annotation import shadow
from merxen.annotation.schema import Columns, soft_columns
from merxen.annotation.vocab import HUMAN_BROAD_CLASSES, primary_vocab

VOCAB = primary_vocab("human")
EXC = "Upper-layer intratelencephalic"
ASTRO = "Astrocyte"
OLIGO = "Oligodendrocyte"
MICRO = "Microglia"


def _frame(names: list[list[object]], probabilities: list[list[float]]) -> pd.DataFrame:
    """A tidy level frame: assigned node then runner-ups."""
    record: dict[str, list[object]] = {
        "name": [row[0] for row in names],
        "assignment": [None if row[0] is None else f"L_{row[0]}" for row in names],
        "bp": [row[0] for row in probabilities],
    }
    width = max(len(row) for row in names)
    for rank in range(1, width):
        record[f"runner_up_{rank}_name"] = [
            row[rank] if rank < len(row) else None for row in names
        ]
        record[f"runner_up_{rank}_assignment"] = [
            None if rank >= len(row) or row[rank] is None else f"L_{row[rank]}"
            for row in names
        ]
        record[f"runner_up_{rank}_probability"] = [
            row[rank] if rank < len(row) else np.nan for row in probabilities
        ]
    return pd.DataFrame(record, index=[f"c{index}" for index in range(len(names))])


def test_shadow_reexports_the_production_estimator() -> None:
    assert shadow.soft_broad_matrix is comp.soft_broad_matrix
    assert shadow.block_bootstrap_jsd is comp.block_bootstrap_jsd
    assert shadow.COMPOSITION_COLUMNS == comp.COMPOSITION_COLUMNS


def test_soft_vectors_sum_to_one_with_sinks_missing_nodes_and_overflow() -> None:
    rng = np.random.default_rng(3)
    nodes = [EXC, ASTRO, OLIGO, MICRO, "Splatter", "not a node", None]
    names = [
        [rng.choice(np.array(nodes, dtype=object)) for _ in range(6)]
        for _ in range(400)
    ]
    probabilities = [list(rng.uniform(0, 0.6, 6)) for _ in range(400)]
    probabilities[0] = [np.nan] * 6
    matrix = comp.soft_broad_from_level(_frame(names, probabilities))
    assert matrix.shape == (400, len(HUMAN_BROAD_CLASSES) + 1)
    np.testing.assert_allclose(matrix.sum(axis=1), 1.0)
    assert (matrix >= 0).all() and (matrix <= 1).all()
    assert matrix[0, -1] == 1.0  # no probability: all unallocated


def test_soft_vector_aggregates_runner_ups_through_the_vocab() -> None:
    frame = _frame(
        [[EXC, ASTRO, "Deep-layer intratelencephalic", "Splatter"]],
        [[0.6, 0.2, 0.1, 0.05]],
    )
    row = comp.soft_broad_from_level(frame)[0]
    shares = dict(zip(comp.COMPOSITION_COLUMNS, row, strict=True))
    assert shares["Neurons"] == pytest.approx(0.7)
    assert shares["Astrocytes"] == pytest.approx(0.2)
    # The sink and the residual go to unallocated.
    assert shares[comp.UNALLOCATED] == pytest.approx(0.1)


def test_soft_columns_are_float32_and_null_outside_the_table() -> None:
    matrix = np.tile(np.eye(8)[0], (2, 1))
    columns = comp.soft_broad_columns(matrix, np.array([0, 2]), 4)
    assert set(columns) == set(soft_columns("human"))
    for values in columns.values():
        assert values.dtype == np.float32
        assert np.isnan(values[[1, 3]]).all()
    assert columns[f"{Columns.SOFT_BROAD_PREFIX}neurons"][0] == 1.0
    with pytest.raises(ValueError, match="one row per position"):
        comp.soft_broad_columns(matrix, np.array([0]), 4)


def test_dataset_type_composition_keys_labels_per_depth_bin() -> None:
    frame = _frame(
        [[EXC, ASTRO], [ASTRO, None], [OLIGO, EXC]],
        [[0.8, 0.2], [1.0, np.nan], [0.5, 0.5]],
    )
    composition = comp.dataset_type_composition(
        frame, [12, 40, 300], [10, 15, 30, 60, 120, 250], min_bin_mass=0.0
    )
    assert composition.overall == pytest.approx(
        {f"L_{EXC}": 1.3, f"L_{ASTRO}": 1.2, f"L_{OLIGO}": 0.5}
    )
    assert composition.by_depth[10] == pytest.approx(
        {f"L_{EXC}": 0.8, f"L_{ASTRO}": 0.2}
    )
    assert composition.by_depth[250] == pytest.approx(
        {f"L_{OLIGO}": 0.5, f"L_{EXC}": 0.5}
    )
    assert composition.bin_mass[30] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="one value per row"):
        comp.dataset_type_composition(frame, [1, 2], [10])


def _section(
    platform: str,
    n: int,
    shares: np.ndarray,
    *,
    seed: int,
    aligned: bool = True,
    extent: float = 3000.0,
) -> comp.SectionComposition:
    rng = np.random.default_rng(seed)
    classes = rng.choice(len(HUMAN_BROAD_CLASSES), size=n, p=shares)
    soft = np.zeros((n, 8))
    soft[np.arange(n), classes] = 0.9
    soft[:, -1] = 0.1
    xy = rng.uniform(0, extent, size=(n, 2))
    names = np.array(HUMAN_BROAD_CLASSES, dtype=object)[classes]
    return comp.section_composition(
        f"S_{platform}",
        platform,
        soft,
        total_counts=rng.integers(10, 200, n),
        argmax_broad=names,
        confident_broad=names,
        confident=rng.uniform(size=n) < 0.6,
        xy=xy,
        aligned_frame=aligned,
    )


SHARES = np.array([0.5, 0.15, 0.15, 0.05, 0.05, 0.05, 0.05])


class _HalfMask:
    """A shared tissue mask covering x < 1500 µm."""

    fixed_platform = "XENIUM"

    def contains(self, xy: np.ndarray) -> np.ndarray:
        return np.asarray(xy)[:, 0] < 1500.0


def test_pair_jsd_ci_shapes_regions_and_reproducibility() -> None:
    first = _section("MERSCOPE", 3000, SHARES, seed=1)
    other = np.array([0.4, 0.2, 0.2, 0.05, 0.05, 0.05, 0.05])
    second = _section("XENIUM", 2500, other, seed=2)
    records, note = comp.pair_jsd(first, second, mask=_HalfMask(), n_reps=50)
    assert note == "applied"
    assert {(item.kind, item.region) for item in records} == {
        (kind, region)
        for kind in comp.COMPOSITION_KINDS
        for region in (comp.WHOLE_SECTION, comp.SHARED_MASK)
    }
    for item in records:
        assert item.resampling == "joint"
        assert item.n_reps == 50
        assert item.ci_low is not None and item.ci_high is not None
        assert item.ci_low <= item.ci_high
        assert 0.0 <= item.ci_low <= 1.0 and item.jsd is not None
        assert item.ci_low_independent is not None
        assert item.n_tile_locations >= max(item.n_tiles_a, item.n_tiles_b)
        payload = item.to_json()
        assert set(payload) >= {"kind", "region", "jsd", "ci_low", "ci_high"}
    soft = next(
        item
        for item in records
        if item.kind == "soft" and item.region == comp.WHOLE_SECTION
    )
    expected = comp.jensen_shannon_distance(
        first.matrices["soft"].sum(0), second.matrices["soft"].sum(0)
    )
    assert soft.jsd == pytest.approx(expected, abs=1e-6)
    masked = next(
        item
        for item in records
        if item.kind == "soft" and item.region == comp.SHARED_MASK
    )
    assert masked.n_cells_a < soft.n_cells_a
    again, _ = comp.pair_jsd(first, second, mask=_HalfMask(), n_reps=50)
    assert [item.to_json() for item in again] == [item.to_json() for item in records]
    reseeded, _ = comp.pair_jsd(first, second, mask=_HalfMask(), n_reps=50, seed=7)
    assert reseeded[0].jsd == records[0].jsd
    assert (reseeded[0].ci_low, reseeded[0].ci_high) != (
        records[0].ci_low,
        records[0].ci_high,
    )


def test_identical_sections_have_zero_jsd() -> None:
    first = _section("MERSCOPE", 500, SHARES, seed=5)
    records, _ = comp.pair_jsd(first, first, n_reps=20)
    assert records[0].jsd == pytest.approx(0.0, abs=1e-9)


def test_pair_jsd_without_a_shared_frame_resamples_independently() -> None:
    first = _section("MERSCOPE", 800, SHARES, seed=1, aligned=False)
    second = _section("XENIUM", 800, SHARES, seed=2)
    records, note = comp.pair_jsd(first, second, mask=_HalfMask(), n_reps=20)
    assert note == "coordinates not in the mask's fixed frame"
    assert {item.region for item in records} == {comp.WHOLE_SECTION}
    assert all(item.resampling == "independent" for item in records)
    assert all("independent" in item.note for item in records)


def test_pair_jsd_without_coordinates_gives_the_point_estimate_only() -> None:
    first = _section("MERSCOPE", 300, SHARES, seed=1)
    second = _section("XENIUM", 300, SHARES, seed=2)
    bare = comp.SectionComposition(
        first.sample_id, first.platform, first.matrices, xy=None
    )
    records, note = comp.pair_jsd(bare, second, n_reps=20)
    assert note == "no shared tissue mask"
    assert all(item.ci_low is None and item.resampling is None for item in records)
    assert all(item.jsd is not None for item in records)


def test_section_kinds_stratify_depth_and_confidence() -> None:
    section = _section("XENIUM", 400, SHARES, seed=9)
    deep = section.matrices["soft_ge30"].sum(axis=1) > 0
    assert deep.sum() < 400
    confident = section.matrices["confident"].sum(axis=1)
    assert set(np.unique(confident)) <= {0.0, 1.0}
    shares = comp.section_shares(section)
    assert set(shares[comp.WHOLE_SECTION]) == set(comp.COMPOSITION_KINDS)
    soft = shares[comp.WHOLE_SECTION]["soft"]
    total7 = sum(soft[f"share7_{token}"] for token in ("neurons", "astrocytes"))
    assert 0 < total7 < 1
    assert soft["share_unallocated"] == pytest.approx(0.1)
    with pytest.raises(ValueError, match="one value per soft row"):
        comp.section_composition(
            "x",
            "XENIUM",
            np.zeros((2, 8)),
            total_counts=[1],
            argmax_broad=["Neurons", "Neurons"],
            confident_broad=["Neurons", "Neurons"],
            confident=[True, True],
        )


def test_whb_broad_of_maps_sinks_to_unassigned() -> None:
    broad_of = comp.whb_broad_of()
    assert broad_of["Splatter"] == "Mixed/Unknown"
    assert broad_of[EXC] == "Neurons"
    assert set(broad_of) == set(VOCAB.names)

"""Tests for the cortical-depth item (``report_depth``; H12).

The synthetic pair is a straight cortex in the fixed frame: tangential axis
``x`` (0-6000 µm), pia at ``y = 0``, white matter from ``y = 2000``; upper
IT, deep IT and deep NP/CT/6b cells sit at their layers, oligodendrocytes in
white matter. A "mirrored" MERSCOPE depth applies the boundaries upside down
(the M7 review's failure: native-frame boundaries on aligned coordinates).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from merxen.annotation.acceptance_scoring import FAIL, NOT_AVAILABLE, PASS, score_h12
from merxen.annotation.report_depth import (
    DEEP_NP_CT_6B,
    INVALID_REASON,
    ORDER,
    PRIMARY_CI,
    SCORED_CI,
    SQUARE_TILE_CI,
    group_labels,
    item_cortical_depth,
)
from merxen.annotation.report_inputs import ReportInputs, ReportSources, SampleData
from merxen.annotation.report_model import ItemWriter, ReportItem, ReportOptions

OPTIONS = ReportOptions(n_bootstrap=30)
STEM = "item09_cortical_depth"
N_CELLS = 9000
THICKNESS_UM = 2000.0


def test_group_labels_merge_near_projecting_and_ct_6b_confident_cells() -> None:
    table = pd.DataFrame(
        {
            "ct_supercluster_name": [
                "Upper-layer intratelencephalic",
                "Deep-layer near-projecting",
                "Deep-layer corticothalamic and 6b",
                "Deep-layer intratelencephalic",
                "Deep-layer near-projecting",
            ],
            "ct_supercluster_status": ["confident"] * 4 + ["low_confidence"],
        }
    )
    groups = group_labels(table)
    np.testing.assert_array_equal(
        groups,
        [ORDER[0], DEEP_NP_CT_6B, DEEP_NP_CT_6B, ORDER[1], ""],
    )
    assert ORDER == (
        "Upper-layer intratelencephalic",
        "Deep-layer intratelencephalic",
        DEEP_NP_CT_6B,
    )


def _layer_cells(
    platform: str, seed: int, *, mirrored: bool = False, unconfident_deep: bool = False
) -> SampleData:
    """One section of the synthetic cortex with its depth output."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 6000, N_CELLS)
    y = rng.uniform(0, 3000, N_CELLS)
    true_depth = y / THICKNESS_UM
    supercluster = np.select(
        [
            true_depth < 0.4,
            true_depth < 0.7,
            true_depth < 1.0,
        ],
        [
            "Upper-layer intratelencephalic",
            "Deep-layer intratelencephalic",
            "Deep-layer near-projecting",
        ],
        default="Oligodendrocyte",
    )
    astro = rng.uniform(size=N_CELLS) < 0.25
    supercluster = np.where(astro & (true_depth < 1.0), "Astrocyte", supercluster)
    broad = np.select(
        [supercluster == "Astrocyte", supercluster == "Oligodendrocyte"],
        ["Astrocytes", "Oligodendrocytes"],
        default="Neurons",
    )
    broad_status = np.full(N_CELLS, "confident", dtype=object)
    if unconfident_deep:
        # Confidence falls with depth: half of the deep grey-matter cells lose
        # their broad label (the gradient must count confident cells only).
        lost = (true_depth > 0.5) & (true_depth < 1.0)
        lost &= rng.uniform(size=N_CELLS) < 0.5
        broad_status = np.where(lost, "low_confidence", broad_status)
    labels = pd.DataFrame(
        {
            "cell_id": [f"{platform[0]}{index}" for index in range(N_CELLS)],
            "in_table": np.ones(N_CELLS, dtype=bool),
            "ct_supercluster_name": supercluster,
            "ct_supercluster_status": np.where(
                broad_status == "confident", "confident", "parent_unresolved"
            ),
            "ct_broad_name": broad,
            "ct_broad_status": broad_status,
        }
    )
    if mirrored:
        # The boundaries upside down: pia at y = 3000, WM at y = 1000.
        depth = (3000.0 - y) / THICKNESS_UM
        source = "shapes:MOSAIK_proseg_hybrid_aligned_nonrigid"
    else:
        depth = true_depth
        source = "shapes:MOSAIK_proseg_hybrid"
    inside = depth < 1.0
    frame = pd.DataFrame(
        {
            "x": x,
            "y": y,
            "inside_cortical_ribbon": inside,
            "cortical_depth_annotation": np.where(inside, "grey_matter", ""),
            "equivolumetric_depth": np.where(inside, depth, np.nan),
            "tangential_position_um": np.where(inside, x, np.nan),
            "column_id": np.where(inside, np.floor(x / 50), np.nan),
        },
        index=labels.index,
    )
    return SampleData(
        sample_id=f"PX_{platform}",
        platform=platform,
        labels=labels,
        summary={"resolution": {"gate": {"level": "full"}}},
        manifest={},
        labels_path=Path("labels.parquet"),
        xy=np.column_stack([x, y]),
        aligned_frame=True,
        coordinate_source="clustered:test",
        depth=frame,
        depth_path=Path("depth.parquet"),
        depth_qc={
            "warnings": ["abnormal_local_thickness_variation"] if mirrored else [],
            "tables": {
                "proseg_hybrid": {
                    "coordinate_source": source,
                    "n_cells": N_CELLS,
                    "n_inside_ribbon": int(inside.sum()),
                }
            },
        },
    )


def _inputs(*samples: SampleData) -> ReportInputs:
    return ReportInputs(
        sources=ReportSources(
            species="human",
            pair_id="PX",
            segmentation="proseg_hybrid",
            resolve_dir=Path("resolve_out"),
        ),
        summary={},
        samples={sample.sample_id: sample for sample in samples},
    )


def _item(inputs: ReportInputs, root: Path) -> ReportItem:
    return item_cortical_depth(inputs, ItemWriter(root, make_figures=False), OPTIONS)


def _records(item: object, name: str, **fields: object) -> list[object]:
    return [
        record
        for record in item.metrics  # type: ignore[attr-defined]
        if record.name == name
        and all(getattr(record, key) == value for key, value in fields.items())
    ]


def test_a_matching_pair_passes_the_depth_check_and_the_ordering(
    tmp_path: Path,
) -> None:
    inputs = _inputs(_layer_cells("MERSCOPE", 1), _layer_cells("XENIUM", 2))
    item = item_cortical_depth(
        inputs, ItemWriter(tmp_path, make_figures=False), OPTIONS
    )
    assert item.status == "ok" and not item.banners
    for sample_id in ("PX_MERSCOPE", "PX_XENIUM"):
        (valid,) = _records(item, "depth_input_valid", sample_id=sample_id)
        assert valid.value is True and valid.status == "measured"
        (ordering,) = _records(
            item, "depth_ordering_passes", sample_id=sample_id, kind=None
        )
        assert ordering.value is True and f"ci={PRIMARY_CI}" in ordering.note
        assert _records(
            item, "depth_ordering_passes", sample_id=sample_id, kind=SQUARE_TILE_CI
        )
    (agreement,) = _records(item, "depth_ribbon_agreement")
    (depth_r,) = _records(item, "depth_bin_mean_r")
    assert agreement.value > 0.95 and depth_r.value > 0.95
    (method,) = _records(item, "depth_ci_method")
    assert method.value == PRIMARY_CI
    # The primary CI resamples 500 µm tangential blocks, the sensitivity tiles.
    from merxen.annotation.report_metrics import (
        grid_codes,
        median_block_ci,
        tangential_block_codes,
    )

    sample = inputs.samples["PX_MERSCOPE"]
    assert sample.depth is not None
    depth = sample.depth["equivolumetric_depth"].to_numpy()
    upper = np.isfinite(depth) & (
        group_labels(sample.labels) == "Upper-layer intratelencephalic"
    )
    blocks = tangential_block_codes(
        sample.depth["tangential_position_um"].to_numpy()[upper], 500.0
    )
    expected = median_block_ci(depth[upper], blocks, n_reps=30, seed=0)
    (record,) = _records(item, "median_depth", sample_id="PX_MERSCOPE", group=ORDER[0])
    assert record.ci_low == pytest.approx(expected.ci_low, rel=1e-5)
    assert record.ci_high == pytest.approx(expected.ci_high, rel=1e-5)
    tiles = median_block_ci(
        depth[upper], grid_codes(np.asarray(sample.xy)[upper], 500.0), n_reps=30
    )
    assert (tiles.ci_low, tiles.ci_high) != (expected.ci_low, expected.ci_high)
    replicated = _records(item, "depth_ordering_replicated")
    assert {record.kind for record in replicated} == {None, SQUARE_TILE_CI}
    assert all(record.value is True for record in replicated)
    medians = pd.read_csv(
        tmp_path / "figures" / "item09_cortical_depth_median_depth.csv"
    )
    assert set(medians["ci_method"]) == {PRIMARY_CI}
    assert {"ci_low_square_tiles", "n_blocks"} <= set(medians.columns)


def test_a_mirrored_boundary_makes_every_h12_metric_of_its_platform_not_available(
    tmp_path: Path,
) -> None:
    """The M7 review's blocker: boundaries in the wrong frame are caught."""
    merscope = _layer_cells("MERSCOPE", 1, mirrored=True)
    inputs = _inputs(merscope, _layer_cells("XENIUM", 2))
    item = item_cortical_depth(
        inputs, ItemWriter(tmp_path, make_figures=False), OPTIONS
    )
    assert item.status == "partial"
    assert [banner.code for banner in item.banners] == [INVALID_REASON]
    assert item.banners[0].sample_id == "PX_MERSCOPE"
    (agreement,) = _records(item, "depth_ribbon_agreement")
    (depth_r,) = _records(item, "depth_bin_mean_r")
    assert agreement.value < 0.8 and depth_r.value < 0
    (invalid,) = _records(item, "depth_input_valid", sample_id="PX_MERSCOPE")
    assert invalid.value is False and invalid.note.startswith(INVALID_REASON)
    assert "transformed coordinates" in invalid.note
    (valid,) = _records(item, "depth_input_valid", sample_id="PX_XENIUM")
    assert valid.value is True
    measured_names = {
        "depth_ordering_passes",
        "median_depth",
        "depth_gradient_spearman",
        "oligodendrocyte_wm_minus_gm_share",
    }
    merscope_records = [
        record
        for record in item.metrics
        if record.sample_id == "PX_MERSCOPE" and record.name in measured_names
    ]
    assert {record.name for record in merscope_records} == measured_names
    for record in merscope_records:
        assert record.status == "not_available" and record.value is None
        assert record.ci_low is None and record.note.startswith(INVALID_REASON)
    xenium_orderings = _records(item, "depth_ordering_passes", sample_id="PX_XENIUM")
    assert all(record.status == "measured" for record in xenium_orderings)
    for name, n_records in (
        ("depth_ordering_replicated", 2),  # display primary and scored tiles
        ("depth_profile_spearman_between_platforms", 1),
    ):
        records = _records(item, name)
        assert len(records) == n_records
        for record in records:
            assert record.status == "not_available" and record.value is None
            assert record.note.startswith(f"{INVALID_REASON} on ['MERSCOPE']")
    table = pd.read_csv(
        tmp_path / "tables" / "item09_cortical_depth__depth_validity.csv"
    )
    assert list(table["depth_input_valid"]) == [False, True]
    medians = pd.read_csv(
        tmp_path / "figures" / "item09_cortical_depth_median_depth.csv"
    )
    assert "MERSCOPE (depth input invalid)" in set(medians["series"])


def test_without_a_transformed_platform_both_platforms_are_marked(
    tmp_path: Path,
) -> None:
    merscope = _layer_cells("MERSCOPE", 1, mirrored=True)
    assert merscope.depth_qc is not None
    tables = merscope.depth_qc["tables"]
    tables["proseg_hybrid"]["coordinate_source"] = "obsm:spatial"
    inputs = _inputs(merscope, _layer_cells("XENIUM", 2))
    item = item_cortical_depth(
        inputs, ItemWriter(tmp_path, make_figures=False), OPTIONS
    )
    flags = [record.value for record in _records(item, "depth_input_valid")]
    assert flags == [False, False]
    assert "cannot tell which platform" in item.banners[0].text


def test_a_single_platform_is_not_checked_and_stays_measured(tmp_path: Path) -> None:
    item = item_cortical_depth(
        _inputs(_layer_cells("XENIUM", 2)),
        ItemWriter(tmp_path, make_figures=False),
        OPTIONS,
    )
    (valid,) = _records(item, "depth_input_valid")
    assert valid.value is None and valid.status == "not_available"
    assert valid.note.startswith("not checked")
    (ordering,) = _records(item, "depth_ordering_passes", kind=None)
    assert ordering.status == "measured" and item.status == "ok"
    assert not _records(item, "depth_ribbon_agreement")


def test_depth_gradients_are_shares_of_confident_broad_cells(tmp_path: Path) -> None:
    """Regression (M7 review): non-confident cells are not in the denominator."""
    inputs = _inputs(
        _layer_cells("MERSCOPE", 1, unconfident_deep=True), _layer_cells("XENIUM", 2)
    )
    item_cortical_depth(inputs, ItemWriter(tmp_path, make_figures=False), OPTIONS)
    profile = pd.read_csv(tmp_path / "figures" / "item09_cortical_depth_gradients.csv")
    merscope = profile[profile["platform"] == "MERSCOPE"]
    # Every confident grey-matter cell is an astrocyte or a neuron: their
    # shares sum to 1 in every depth bin, although half of the deep cells
    # have no confident broad label.
    totals = merscope.groupby("depth_bin")["share"].sum()
    np.testing.assert_allclose(totals.to_numpy(), 1.0)
    labels = inputs.samples["PX_MERSCOPE"].labels
    confident = labels["ct_broad_status"].to_numpy() == "confident"
    depth = inputs.samples["PX_MERSCOPE"].depth
    assert depth is not None
    finite = np.isfinite(depth["equivolumetric_depth"].to_numpy())
    assert merscope.groupby("depth_bin")["n_cells"].first().sum() == int(
        (confident & finite).sum()
    )
    assert int((~confident & finite).sum()) > 1000


def test_tangential_blocks_keep_every_depth_slice_in_a_replicate() -> None:
    """Square tiles resample depth slices; full-depth strips do not."""
    from merxen.annotation.report_metrics import (
        grid_codes,
        median_block_ci,
        tangential_block_codes,
    )

    sample = _layer_cells("XENIUM", 3)
    assert sample.depth is not None
    depth = sample.depth["equivolumetric_depth"].to_numpy()
    tangential = sample.depth["tangential_position_um"].to_numpy()
    keep = np.isfinite(depth) & (
        sample.labels["ct_broad_name"].to_numpy() == "Astrocytes"
    )
    blocks = tangential_block_codes(tangential[keep], 500.0)
    assert set(np.unique(blocks)) == set(range(12))
    tiles = grid_codes(np.asarray(sample.xy)[keep], 500.0)
    by_block = median_block_ci(depth[keep], blocks, n_reps=200)
    by_tile = median_block_ci(depth[keep], tiles, n_reps=200)
    assert by_block.median == pytest.approx(by_tile.median)
    assert (by_block.ci_high - by_block.ci_low) < (by_tile.ci_high - by_tile.ci_low)


def _close_deep_layers(platform: str, seed: int) -> SampleData:
    """A section whose deep IT and NP/CT/6b medians are close (P7513_XENIUM).

    The deep grey-matter cells are labelled by a noisy depth threshold, so
    the two groups' medians are about .08 apart: the tangential-block CIs
    (whole pia-to-WM strips) separate them, the square-tile CIs overlap.
    """
    sample = _layer_cells(platform, seed)
    assert sample.depth is not None
    rng = np.random.default_rng(seed + 100)
    depth = sample.depth["equivolumetric_depth"].to_numpy()
    names = sample.labels["ct_supercluster_name"].to_numpy().astype(object)
    deep = np.isfinite(depth) & np.isin(
        names, ["Deep-layer intratelencephalic", "Deep-layer near-projecting"]
    )
    noisy = depth[deep] + rng.normal(0.0, 0.8, int(deep.sum()))
    names[deep] = np.where(
        noisy > 0.7, "Deep-layer near-projecting", "Deep-layer intratelencephalic"
    )
    sample.labels["ct_supercluster_name"] = names
    return sample


def test_h12_is_scored_on_the_square_tiles_where_the_tangential_blocks_pass(
    tmp_path: Path,
) -> None:
    """M8 review: the scored H12 is the tile CI (pre-registration §18 item 3)."""
    inputs = _inputs(_layer_cells("MERSCOPE", 1), _close_deep_layers("XENIUM", 2))
    item = _item(inputs, tmp_path)
    tile = {
        record.platform: record.value
        for record in _records(item, "depth_ordering_passes", kind=SQUARE_TILE_CI)
    }
    primary = {
        record.platform: record.value
        for record in _records(item, "depth_ordering_passes", kind=None)
    }
    assert primary == {"MERSCOPE": True, "XENIUM": True}
    assert tile == {"MERSCOPE": True, "XENIUM": False}
    (shown,) = _records(item, "depth_ordering_replicated", kind=None)
    (scored,) = _records(item, "depth_ordering_replicated", kind=SQUARE_TILE_CI)
    assert shown.value is True and scored.value is False
    assert "the CI the M8 gate scores" in scored.definition
    # The between-platform rank correlation does not depend on the CI.
    assert len(_records(item, "depth_profile_spearman_between_platforms")) == 1
    (method,) = _records(item, "depth_ci_method")
    (scored_ci,) = _records(item, "depth_ci_scored")
    assert method.value == PRIMARY_CI  # the display primary is unchanged
    assert scored_ci.value == SCORED_CI == SQUARE_TILE_CI
    score = score_h12(item.metrics, pair_id="P7513")
    assert score.ci_scored == SQUARE_TILE_CI
    assert score.ci_reported_beside == PRIMARY_CI
    assert score.ordering == {"MERSCOPE": PASS, "XENIUM": FAIL}
    assert score.ordering_verdict == FAIL and score.verdict == FAIL
    assert score.reported_beside["ordering"] == {"MERSCOPE": PASS, "XENIUM": PASS}
    assert score.to_json()["ci_scored"] == SQUARE_TILE_CI
    # P1212 is scored on WM > GM alone (plan §14 H12).
    other = score_h12(item.metrics, pair_id="P1212")
    assert not other.ordering_required and other.wm_gm_verdict == PASS
    assert other.verdict == PASS


def test_h12_on_a_gated_platform_is_not_available_under_the_scored_ci(
    tmp_path: Path,
) -> None:
    xenium = _close_deep_layers("XENIUM", 2)
    xenium.labels["ct_supercluster_status"] = "withheld_for_comparison"
    xenium.summary["resolution"]["gate"]["level"] = "broad_only"
    item = _item(_inputs(_layer_cells("MERSCOPE", 1), xenium), tmp_path)
    (tile,) = _records(
        item, "depth_ordering_passes", sample_id="PX_XENIUM", kind=SQUARE_TILE_CI
    )
    assert tile.status == "not_available" and tile.value is None
    (scored,) = _records(item, "depth_ordering_replicated", kind=SQUARE_TILE_CI)
    assert scored.status == "not_available"
    score = score_h12(item.metrics, pair_id="P7513")
    assert score.ordering == {"MERSCOPE": PASS, "XENIUM": NOT_AVAILABLE}
    assert score.verdict == NOT_AVAILABLE


def test_without_tangential_positions_the_scored_record_is_the_primary(
    tmp_path: Path,
) -> None:
    samples = [_layer_cells("MERSCOPE", 1), _close_deep_layers("XENIUM", 2)]
    for sample in samples:
        assert sample.depth is not None
        sample.depth = sample.depth.drop(columns=["tangential_position_um"])
    item = _item(_inputs(*samples), tmp_path)
    for sample in samples:
        (primary,) = _records(
            item, "depth_ordering_passes", sample_id=sample.sample_id, kind=None
        )
        (tile,) = _records(
            item,
            "depth_ordering_passes",
            sample_id=sample.sample_id,
            kind=SQUARE_TILE_CI,
        )
        assert primary.value == tile.value
    (method,) = _records(item, "depth_ci_method")
    assert method.value == SQUARE_TILE_CI
    assert score_h12(item.metrics, pair_id="P7513").verdict == FAIL

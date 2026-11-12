"""Tests for regionprops-based mask filtering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from merxen.segmentation.mask_filter import (
    drop_labels_from_mask,
    filter_cell_by_regionprops,
    filter_labeled_mask_by_area,
)


def test_filter_cell_by_regionprops_drops_eccentric_region() -> None:
    """Elongated high-eccentricity components should be removed."""
    seg = np.zeros((32, 32), dtype=np.uint8)
    seg[2:10, 2:10] = 1
    seg[20:21, 4:28] = 1

    out = filter_cell_by_regionprops(
        seg,
        max_eccentricity=0.90,
        n_jobs=1,
        min_area_percentile=0.0,
    )

    labels = np.unique(out)
    assert labels.tolist() == [0, 1]
    assert int((out == 1).sum()) == 64


def test_filter_cell_by_regionprops_returns_empty_for_empty_input() -> None:
    """Empty masks should produce an all-zero output mask."""
    seg = np.zeros((16, 16), dtype=np.uint8)
    out = filter_cell_by_regionprops(seg, n_jobs=1)
    assert out.shape == seg.shape
    assert out.dtype == np.int32
    assert int(out.sum()) == 0


def test_filter_labeled_mask_by_area_removes_small_and_large_masks(
    tmp_path: Path,
) -> None:
    """Absolute micron-area filtering should stream, remove, and relabel masks."""
    mask = np.zeros((64, 64), dtype=np.uint32)
    mask[1:3, 1:3] = 1
    mask[5:10, 5:10] = 2
    mask[12:31, 12:31] = 3
    mask[36:58, 36:58] = 4

    mask_path = tmp_path / "mask.npy"
    np.save(mask_path, mask)

    stats = filter_labeled_mask_by_area(
        mask_path,
        pixel_area_um2=1.0,
        min_area_um2=5.0,
        max_area_um2=400.0,
        chunk_mb=1,
        show_progress=False,
    )

    out = np.load(mask_path)
    assert stats["n_labels"] == 4
    assert stats["n_kept"] == 2
    assert stats["n_removed_small"] == 1
    assert stats["n_removed_large"] == 1
    assert np.unique(out).tolist() == [0, 1, 2]
    assert int((out == 1).sum()) == 25
    assert int((out == 2).sum()) == 361


def test_drop_labels_from_mask_relabels_the_kept_labels_densely(
    tmp_path: Path,
) -> None:
    mask = np.zeros((8, 12), dtype=np.uint32)
    mask[0:2, 0:2] = 1
    mask[0:2, 4:6] = 3
    mask[4:6, 0:2] = 4
    mask[4:6, 8:10] = 7
    path = tmp_path / "masks.npy"
    np.save(path, mask)

    # Label 3 and 7 are dropped; 9 is absent and 0 is the background (ignored).
    summary = drop_labels_from_mask(path, [3, 7, 9, 0], chunk_mb=1)

    out = np.load(path)
    assert summary == {"n_labels": 4, "n_kept": 2, "n_dropped": 2, "written": True}
    assert out.dtype == np.uint32
    assert np.array_equal(out == 1, mask == 1)
    assert np.array_equal(out == 2, mask == 4)
    assert sorted(np.unique(out)) == [0, 1, 2]
    assert [p.name for p in tmp_path.iterdir()] == ["masks.npy"]


def test_drop_labels_from_mask_leaves_the_file_alone_when_nothing_is_dropped(
    tmp_path: Path,
) -> None:
    mask = np.zeros((6, 6), dtype=np.int32)
    mask[1:3, 1:3] = 2  # not dense: it must stay as it is when nothing is dropped
    path = tmp_path / "masks.npy"
    np.save(path, mask)
    before = path.read_bytes()

    summary = drop_labels_from_mask(path, [5, 0])

    assert summary == {"n_labels": 1, "n_kept": 1, "n_dropped": 0, "written": False}
    assert path.read_bytes() == before
    assert drop_labels_from_mask(path, []) == summary


def test_geometry_filter_removes_crescent_and_preserves_round_cells_and_processes(
    tmp_path: Path,
) -> None:
    """Only whole small/thin/concave objects are removed, across rewrite chunks."""
    from merxen.config import MaskGeometryFilterConfig

    yy, xx = np.indices((240, 280))
    mask = np.zeros(yy.shape, dtype=np.uint32)
    crescent = ((yy - 35) ** 2 + (xx - 35) ** 2 <= 12**2) & (
        (yy - 35) ** 2 + (xx - 40) ** 2 > 11**2
    )
    circle = (yy - 35) ** 2 + (xx - 85) ** 2 <= 3**2
    neuron = (yy - 135) ** 2 + (xx - 150) ** 2 <= 30**2
    mask[crescent] = 2
    mask[circle] = 5
    mask[neuron] = 9
    mask[134:137, 178:220] = 9  # A real process belongs to the wider soma.
    path = tmp_path / "mask.npy"
    np.save(path, mask)
    stats = filter_labeled_mask_by_area(
        path,
        pixel_area_um2=0.25,
        min_area_um2=5,
        max_area_um2=2000,
        geometry_filter=MaskGeometryFilterConfig(),
        pixel_size_um=(0.5, 0.5),
        chunk_mb=1,
    )
    out = np.load(path)
    assert stats["n_removed_geometry"] == 1
    assert stats["n_kept"] == 2
    assert not out[crescent].any()
    assert np.array_equal(out == 1, mask == 5)
    assert np.array_equal(out == 2, mask == 9)


def test_geometry_filter_uses_physical_thickness_and_protects_clipped_objects(
    tmp_path: Path,
) -> None:
    """A long thin U is removed only below the physical thickness threshold."""
    from merxen.config import MaskGeometryFilterConfig

    mask = np.zeros((80, 100), dtype=np.uint32)
    mask[20:40, 20:22] = 1
    mask[20:22, 20:40] = 1
    mask[38:40, 20:40] = 1
    mask[0:20, 60:62] = 2
    mask[0:2, 60:80] = 2
    mask[18:20, 60:80] = 2
    path = tmp_path / "mask.npy"
    for spacing, expected_removed in [((0.5, 0.5), 1), ((2.0, 2.0), 0)]:
        np.save(path, mask)
        stats = filter_labeled_mask_by_area(
            path,
            pixel_area_um2=spacing[0] * spacing[1],
            geometry_filter=MaskGeometryFilterConfig(max_area_um2=1000),
            pixel_size_um=spacing,
        )
        assert stats["n_removed_geometry"] == expected_removed
        assert np.count_nonzero(np.load(path)[mask == 2]) == np.count_nonzero(mask == 2)


def test_geometry_filter_can_be_disabled_and_leaves_source_unchanged(
    tmp_path: Path,
) -> None:
    """An absent geometry policy retains the previous area-only behaviour."""
    mask = np.zeros((30, 30), dtype=np.uint32)
    mask[5:20, 5] = 4
    mask[5, 5:20] = 4
    path = tmp_path / "mask.npy"
    np.save(path, mask)
    before = path.read_bytes()
    stats = filter_labeled_mask_by_area(path, pixel_area_um2=1, min_area_um2=5)
    assert stats["n_removed_geometry"] == 0
    assert path.read_bytes() == before


def test_geometry_identity_distinguishes_legacy_and_active_policies() -> None:
    """Old provenance must not silently claim the new cleanup was applied."""
    from merxen.config import MaskFilterConfig, MaskGeometryFilterConfig

    legacy = MaskFilterConfig()
    assert "geometry_filter" not in legacy.model_dump(mode="json")
    current = MaskFilterConfig(geometry_filter=MaskGeometryFilterConfig())
    assert current != legacy
    assert current.model_dump(mode="json")["geometry_filter"]["max_solidity"] == 0.75
    assert MaskFilterConfig.model_validate_json(current.model_dump_json()) == current


@pytest.mark.parametrize(
    "limits",
    [
        {"max_area_um2": 1},
        {"max_thickness_um": 0.5},
        {"max_solidity": 0.01},
    ],
)
def test_geometry_requires_all_three_conditions(tmp_path: Path, limits: dict) -> None:
    """Passing any one threshold protects an otherwise small thin concave cell."""
    from merxen.config import MaskGeometryFilterConfig

    mask = np.zeros((40, 40), dtype=np.uint32)
    mask[10:30, 10] = 1
    mask[10, 10:25] = 1
    mask[29, 10:25] = 1
    path = tmp_path / "mask.npy"
    np.save(path, mask)
    stats = filter_labeled_mask_by_area(
        path, pixel_area_um2=0.25, geometry_filter=MaskGeometryFilterConfig(**limits)
    )
    assert stats["n_removed_geometry"] == 0
    np.testing.assert_array_equal(np.load(path), mask)

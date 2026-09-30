"""Cortical depth reads the cells in the frame the boundaries are drawn in."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import spatialdata as sd
from pydantic import ValidationError
from shapely.geometry import Point
from spatialdata import SpatialData
from spatialdata.models import ShapesModel, TableModel

from merxen.alignment import manifest as alignment_manifest
from merxen.config import CorticalDepthConfig, CorticalDepthTableConfig
from merxen.cortical_depth import frames
from merxen.cortical_depth.frames import (
    ALIGNED_ELEMENT,
    DEPTH_PROVENANCE_UNS_KEY,
    FIXED_REFERENCE_NATIVE_ELEMENT,
    NATIVE_ELEMENT,
    TABLE_SPATIAL_ONLY,
    BoundaryFrameMismatchError,
    is_alignment_fixed_reference,
    resolve_cell_coordinate_frame,
)
from merxen.cortical_depth.pipeline import run_cortical_depth

NATIVE = "MOSAIK_proseg_hybrid"
ALIGNED = f"{NATIVE}_aligned_nonrigid"
TABLE = "table_MOSAIK_proseg_hybrid"
MOVING_MANIFEST = {"merxen_alignment": {"roles": {"fixed": "XENIUM"}}}
XENIUM_MOVING_MANIFEST = {"merxen_alignment": {"roles": {"fixed": "MERSCOPE"}}}
PAIR_REFERENCE = {"merxen_alignment_pair_reference": {"pair_id": "P1"}}
# Aligned = native mirrored in x plus a large offset, like the real MERSCOPE
# native -> Xenium affine (x flip, ~11 mm).
MIRROR_X_UM = 5000.0
OFFSET_Y_UM = 3000.0
PIA_Y_UM = 0.0
WM_Y_UM = 400.0


def _store(shapes: list[str], attrs: dict[str, Any] | None = None) -> SimpleNamespace:
    return SimpleNamespace(shapes={key: None for key in shapes}, attrs=attrs or {})


def _table(region: str | None = NATIVE) -> ad.AnnData:
    table = ad.AnnData(np.zeros((2, 1), dtype=np.float32))
    if region is not None:
        table.uns["spatialdata_attrs"] = {"region": region}
    return table


def _resolve(
    store: SimpleNamespace,
    *,
    platform: str,
    boundary_frame: str,
    requested: str | None = NATIVE,
    table_key: str = TABLE,
    region: str | None = NATIVE,
) -> frames.CellCoordinateFrame:
    return resolve_cell_coordinate_frame(
        store,
        table=_table(region),
        table_key=table_key,
        requested_shape_key=requested,
        platform=platform,
        boundary_frame=boundary_frame,
    )


# platform, aligned variant present, boundary frame, store attrs ->
# (shape key, obsm['spatial'] usable, resolution) or the refusal.
TRUTH_TABLE = [
    # The fix: native boundaries never read the aligned variant.
    ("MERSCOPE", True, "native", MOVING_MANIFEST, (NATIVE, True, NATIVE_ELEMENT)),
    ("MERSCOPE", True, "aligned", MOVING_MANIFEST, (ALIGNED, False, ALIGNED_ELEMENT)),
    ("MERSCOPE", False, "native", {}, (NATIVE, True, NATIVE_ELEMENT)),
    ("MERSCOPE", False, "aligned", {}, BoundaryFrameMismatchError),
    (
        "MERSCOPE",
        False,
        "aligned",
        PAIR_REFERENCE,
        (NATIVE, True, FIXED_REFERENCE_NATIVE_ELEMENT),
    ),
    ("XENIUM", False, "native", PAIR_REFERENCE, (NATIVE, True, NATIVE_ELEMENT)),
    ("XENIUM", False, "native", {}, (NATIVE, True, NATIVE_ELEMENT)),
    (
        "XENIUM",
        False,
        "aligned",
        PAIR_REFERENCE,
        (NATIVE, True, FIXED_REFERENCE_NATIVE_ELEMENT),
    ),
    ("XENIUM", False, "aligned", {}, BoundaryFrameMismatchError),
    (
        "XENIUM",
        True,
        "native",
        XENIUM_MOVING_MANIFEST,
        (NATIVE, True, NATIVE_ELEMENT),
    ),
    (
        "XENIUM",
        True,
        "aligned",
        XENIUM_MOVING_MANIFEST,
        (ALIGNED, False, ALIGNED_ELEMENT),
    ),
]


@pytest.mark.parametrize(
    ("platform", "has_aligned", "boundary_frame", "attrs", "expected"), TRUTH_TABLE
)
def test_frame_selection_truth_table(
    platform: str,
    has_aligned: bool,
    boundary_frame: str,
    attrs: dict[str, Any],
    expected: tuple[str, bool, str] | type[Exception],
) -> None:
    """Every platform x aligned-variant x boundary-frame cell picks one element."""
    shapes = [NATIVE, ALIGNED] if has_aligned else [NATIVE]
    store = _store(shapes, attrs)

    if isinstance(expected, type):
        with pytest.raises(expected, match="boundary_frame='aligned'"):
            _resolve(store, platform=platform, boundary_frame=boundary_frame)
        return
    frame = _resolve(store, platform=platform, boundary_frame=boundary_frame)

    shape_key, use_table_spatial, resolution = expected
    assert frame.shape_key == shape_key
    assert frame.use_table_spatial is use_table_spatial
    assert frame.resolution == resolution
    assert frame.boundary_frame == boundary_frame
    # A native table keeps annotating its native element.
    assert frame.region_key == NATIVE


def _legacy_resolve_shape_key(
    store: SimpleNamespace, *, table: ad.AnnData, requested: str | None, platform: str
) -> str | None:
    """Copy of the pre-fix cortical_depth.pipeline._resolve_shape_key."""
    if len(store.shapes) == 0:
        return None
    if requested is not None:
        aligned = f"{requested}_aligned_nonrigid"
        if platform.upper() == "MERSCOPE" and aligned in store.shapes:
            return aligned
        if requested not in store.shapes:
            raise KeyError(requested)
        return requested
    region = table.uns.get("spatialdata_attrs", {}).get("region")
    if region is not None and region in store.shapes:
        return str(region)
    if region is not None and f"{region}_aligned_nonrigid" in store.shapes:
        return f"{region}_aligned_nonrigid"
    return str(list(store.shapes.keys())[0])


@pytest.mark.parametrize("platform", ["XENIUM", "MERSCOPE"])
@pytest.mark.parametrize("requested", [NATIVE, None])
@pytest.mark.parametrize("region", [NATIVE, "other_shapes", None])
@pytest.mark.parametrize(
    "shapes",
    [[NATIVE], ["cell_boundaries", NATIVE], [NATIVE, ALIGNED], ["cell_boundaries"]],
)
def test_native_frame_matches_legacy_selection_without_merscope_aligned_variant(
    platform: str, requested: str | None, region: str | None, shapes: list[str]
) -> None:
    """Xenium, and stores without an aligned variant, keep the old element."""
    store = _store(shapes)
    table = _table(region)
    # The pre-fix resolver swapped only a configured MERSCOPE shape key for its
    # aligned variant.
    swapped = platform == "MERSCOPE" and requested is not None and ALIGNED in shapes
    try:
        legacy: str | None | type[Exception] = _legacy_resolve_shape_key(
            store, table=table, requested=requested, platform=platform
        )
    except KeyError:
        legacy = KeyError
    if legacy is KeyError:
        with pytest.raises(KeyError):
            resolve_cell_coordinate_frame(
                store,
                table=table,
                table_key=TABLE,
                requested_shape_key=requested,
                platform=platform,
                boundary_frame="native",
            )
        return
    frame = resolve_cell_coordinate_frame(
        store,
        table=table,
        table_key=TABLE,
        requested_shape_key=requested,
        platform=platform,
        boundary_frame="native",
    )
    if swapped:
        assert legacy == ALIGNED
        assert frame.shape_key == NATIVE
    else:
        assert frame.shape_key == legacy
        assert frame.region_key == legacy


def test_native_frame_refuses_when_only_the_aligned_element_exists() -> None:
    """Native boundaries cannot be matched to a store that lost its native cells."""
    with pytest.raises(BoundaryFrameMismatchError, match="only the aligned element"):
        _resolve(
            _store([ALIGNED], MOVING_MANIFEST),
            platform="MERSCOPE",
            boundary_frame="native",
        )


@pytest.mark.parametrize("boundary_frame", ["native", "aligned"])
def test_aligned_table_or_shape_keys_are_refused(boundary_frame: str) -> None:
    """Configured keys must be native; the frame setting picks the variant."""
    store = _store([NATIVE, ALIGNED], MOVING_MANIFEST)
    with pytest.raises(BoundaryFrameMismatchError, match="aligned table"):
        _resolve(
            store,
            platform="MERSCOPE",
            boundary_frame=boundary_frame,
            table_key=f"{TABLE}_aligned_nonrigid",
        )
    with pytest.raises(BoundaryFrameMismatchError, match="names an aligned element"):
        _resolve(
            store, platform="MERSCOPE", boundary_frame=boundary_frame, requested=ALIGNED
        )


def test_unknown_boundary_frame_is_refused() -> None:
    """A typo in the frame name fails instead of defaulting."""
    with pytest.raises(ValueError, match="Unknown cortical-depth boundary_frame"):
        _resolve(_store([NATIVE]), platform="XENIUM", boundary_frame="xenium")


def test_store_without_shapes_uses_table_coordinates_only_in_a_matching_frame() -> None:
    """Native table coordinates are native; aligned needs the fixed section."""
    native = _resolve(_store([]), platform="MERSCOPE", boundary_frame="native")
    assert (native.shape_key, native.use_table_spatial) == (None, True)
    assert native.resolution == TABLE_SPATIAL_ONLY
    fixed = _resolve(
        _store([], PAIR_REFERENCE), platform="XENIUM", boundary_frame="aligned"
    )
    assert fixed.resolution == TABLE_SPATIAL_ONLY
    with pytest.raises(BoundaryFrameMismatchError, match="no shapes"):
        _resolve(
            _store([], MOVING_MANIFEST), platform="MERSCOPE", boundary_frame="aligned"
        )


def test_native_region_of_a_retargeted_table_is_recovered() -> None:
    """A native table whose region was rewritten to the aligned element still
    resolves to its native element when no shape key is configured."""
    frame = _resolve(
        _store([NATIVE, ALIGNED], MOVING_MANIFEST),
        platform="MERSCOPE",
        boundary_frame="native",
        requested=None,
        region=ALIGNED,
    )
    assert frame.shape_key == NATIVE


def test_fixed_reference_detection_reads_the_alignment_attrs() -> None:
    """The fixed section is named by the pair reference or the manifest roles."""
    assert is_alignment_fixed_reference(_store([], PAIR_REFERENCE), "XENIUM")
    assert not is_alignment_fixed_reference(_store([], MOVING_MANIFEST), "MERSCOPE")
    assert is_alignment_fixed_reference(_store([], MOVING_MANIFEST), "XENIUM")
    assert not is_alignment_fixed_reference(_store([]), "XENIUM")


def test_frame_constants_match_the_alignment_manifest() -> None:
    """The duplicated alignment constants stay in sync with their source."""
    assert frames.ALIGNED_ELEMENT_SUFFIX == alignment_manifest.NONRIGID_ELEMENT_SUFFIX
    assert frames.ALIGNMENT_MANIFEST_ATTR == alignment_manifest.ALIGNMENT_MANIFEST_ATTR
    assert (
        frames.ALIGNMENT_PAIR_REFERENCE_ATTR
        == alignment_manifest.ALIGNMENT_PAIR_REFERENCE_ATTR
    )


def test_config_boundary_frame_defaults_to_native_and_rejects_unknown(
    tmp_path: Path,
) -> None:
    """The config field is explicit, defaults to native and is validated."""
    config = _config(tmp_path, tmp_path / "a.geojson", platform="MERSCOPE")
    assert config.boundary_frame == "native"
    with pytest.raises(ValidationError):
        _config(
            tmp_path, tmp_path / "a.geojson", platform="MERSCOPE", boundary_frame="xy"
        )


def test_mirrored_aligned_frame_does_not_leak_into_native_depth(
    tmp_path: Path,
) -> None:
    """MERSCOPE store with a mirrored aligned variant: native boundaries give
    the native depth. The pre-fix resolver read the aligned cells, which all
    fall outside the native ribbon."""
    zarr_path = _write_store(tmp_path, platform="MERSCOPE", attrs=MOVING_MANIFEST)
    boundaries = _write_boundaries(tmp_path / "native.geojson", mirrored=False)

    run_cortical_depth(
        _config(tmp_path / "native_out", boundaries, zarr_path=zarr_path)
    )

    cells, summary = _read_outputs(tmp_path / "native_out")
    expected = (cells["cell_id"].map(_native_xy).str[1] - PIA_Y_UM) / (
        WM_Y_UM - PIA_Y_UM
    )
    assert bool(cells["inside_cortical_ribbon"].all())
    np.testing.assert_allclose(cells["laplace_depth"], expected, atol=0.05)
    np.testing.assert_allclose(
        cells["x"], cells["cell_id"].map(_native_xy).str[0], atol=1e-6
    )
    table = summary["tables"]["proseg_hybrid"]
    assert table["coordinate_source"] == f"shapes:{NATIVE}"
    assert table["shape_key"] == NATIVE
    assert table["boundary_frame"] == "native"
    assert table["cell_coordinate_frame"] == "native"
    assert table["frame_resolution"] == NATIVE_ELEMENT
    assert summary["boundary_frame"] == "native"


def test_aligned_boundaries_read_the_aligned_cells_and_give_the_same_depth(
    tmp_path: Path,
) -> None:
    """The same boundaries mirrored into the aligned frame, with
    boundary_frame='aligned', give each cell the same depth."""
    zarr_path = _write_store(tmp_path, platform="MERSCOPE", attrs=MOVING_MANIFEST)
    native_boundaries = _write_boundaries(tmp_path / "native.geojson", mirrored=False)
    aligned_boundaries = _write_boundaries(tmp_path / "aligned.geojson", mirrored=True)

    run_cortical_depth(
        _config(tmp_path / "native_out", native_boundaries, zarr_path=zarr_path)
    )
    run_cortical_depth(
        _config(
            tmp_path / "aligned_out",
            aligned_boundaries,
            zarr_path=zarr_path,
            boundary_frame="aligned",
        )
    )

    native_cells, _ = _read_outputs(tmp_path / "native_out")
    aligned_cells, summary = _read_outputs(tmp_path / "aligned_out")
    merged = native_cells.merge(aligned_cells, on="cell_id", suffixes=("_n", "_a"))
    assert len(merged) == len(native_cells)
    assert bool(merged["inside_cortical_ribbon_a"].all())
    np.testing.assert_allclose(
        merged["laplace_depth_a"], merged["laplace_depth_n"], atol=0.03
    )
    np.testing.assert_allclose(merged["x_a"], MIRROR_X_UM - merged["x_n"], atol=1e-6)
    table = summary["tables"]["proseg_hybrid"]
    assert table["coordinate_source"] == f"shapes:{ALIGNED}"
    assert table["boundary_frame"] == "aligned"
    assert table["frame_resolution"] == ALIGNED_ELEMENT


def test_aligned_boundaries_without_an_aligned_element_are_refused(
    tmp_path: Path,
) -> None:
    """An unaligned MERSCOPE store cannot serve aligned-frame boundaries."""
    zarr_path = _write_store(
        tmp_path, platform="MERSCOPE", attrs={}, with_aligned=False
    )
    boundaries = _write_boundaries(tmp_path / "aligned.geojson", mirrored=True)

    with pytest.raises(BoundaryFrameMismatchError, match="no aligned element"):
        run_cortical_depth(
            _config(
                tmp_path / "out",
                boundaries,
                zarr_path=zarr_path,
                boundary_frame="aligned",
            )
        )


def test_xenium_table_coordinates_and_region_are_unchanged(tmp_path: Path) -> None:
    """Xenium keeps obsm['spatial'] and its native region on write-back."""
    zarr_path = _write_store(
        tmp_path,
        platform="XENIUM",
        attrs=PAIR_REFERENCE,
        with_aligned=False,
        with_table_spatial=True,
    )
    boundaries = _write_boundaries(tmp_path / "native.geojson", mirrored=False)

    run_cortical_depth(
        _config(
            tmp_path / "out",
            boundaries,
            zarr_path=zarr_path,
            platform="XENIUM",
            write_spatialdata_table=True,
        )
    )

    _, summary = _read_outputs(tmp_path / "out")
    table = summary["tables"]["proseg_hybrid"]
    assert table["coordinate_source"] == "obsm:spatial"
    assert table["shape_key"] == NATIVE
    written = sd.read_zarr(zarr_path).tables[TABLE]
    assert written.uns["spatialdata_attrs"]["region"] == NATIVE
    assert "laplace_depth" in written.obs.columns
    assert dict(written.uns[DEPTH_PROVENANCE_UNS_KEY]) == {
        "boundary_frame": "native",
        "frame_resolution": NATIVE_ELEMENT,
        "coordinate_source": "obsm:spatial",
        "shape_key": NATIVE,
    }


def test_merscope_write_back_keeps_the_native_region_and_records_the_frame(
    tmp_path: Path,
) -> None:
    """Aligned MERSCOPE store: the written-back native table keeps its native
    region and carries the frame the depth columns were computed in. The
    pre-fix code retargeted the table to the aligned element and wrote no
    provenance."""
    zarr_path = _write_store(tmp_path, platform="MERSCOPE", attrs=MOVING_MANIFEST)
    boundaries = _write_boundaries(tmp_path / "native.geojson", mirrored=False)

    run_cortical_depth(
        _config(
            tmp_path / "out",
            boundaries,
            zarr_path=zarr_path,
            write_spatialdata_table=True,
        )
    )

    store = sd.read_zarr(zarr_path)
    written = store.tables[TABLE]
    assert written.uns["spatialdata_attrs"]["region"] == NATIVE
    assert set(written.obs["region"].astype(str)) == {NATIVE}
    assert dict(written.uns[DEPTH_PROVENANCE_UNS_KEY]) == {
        "boundary_frame": "native",
        "frame_resolution": NATIVE_ELEMENT,
        "coordinate_source": f"shapes:{NATIVE}",
        "shape_key": NATIVE,
    }
    assert bool(written.obs["inside_cortical_ribbon"].all())
    cells, _ = _read_outputs(tmp_path / "out")
    by_cell = written.obs.set_index(written.obs["instance_id"].astype(str))
    np.testing.assert_allclose(
        by_cell.loc[cells["cell_id"], "laplace_depth"].to_numpy(float),
        cells["laplace_depth"].to_numpy(float),
    )
    assert ALIGNED in store.shapes


def _native_cells() -> dict[str, tuple[float, float]]:
    xs = np.arange(100.0, 1000.0, 100.0)
    ys = np.arange(40.0, WM_Y_UM, 40.0)
    return {
        f"c{index}": (float(x), float(y))
        for index, (x, y) in enumerate((x, y) for y in ys for x in xs)
    }


def _native_xy(cell_id: str) -> tuple[float, float]:
    return _native_cells()[cell_id]


def _mirror(x: float, y: float) -> tuple[float, float]:
    return MIRROR_X_UM - x, y + OFFSET_Y_UM


def _shapes(cells: dict[str, tuple[float, float]]) -> Any:
    gdf = gpd.GeoDataFrame(
        {"instance_id": list(cells)},
        geometry=[Point(xy).buffer(5.0) for xy in cells.values()],
        index=pd.Index(list(cells)),
    )
    return ShapesModel.parse(gdf)


def _write_store(
    tmp_path: Path,
    *,
    platform: str,
    attrs: dict[str, Any],
    with_aligned: bool = True,
    with_table_spatial: bool = False,
) -> Path:
    cells = _native_cells()
    shapes = {NATIVE: _shapes(cells)}
    if with_aligned:
        shapes[ALIGNED] = _shapes({key: _mirror(*xy) for key, xy in cells.items()})
    obs = pd.DataFrame(
        {
            "instance_id": list(cells),
            "region": pd.Categorical([NATIVE] * len(cells)),
        },
        index=pd.Index(list(cells)),
    )
    table = ad.AnnData(np.ones((len(cells), 1), dtype=np.float32), obs=obs)
    if with_table_spatial:
        table.obsm["spatial"] = np.asarray(list(cells.values()), dtype=float)
    parsed = TableModel.parse(
        table, region=NATIVE, region_key="region", instance_key="instance_id"
    )
    zarr_path = tmp_path / f"{platform.lower()}.zarr"
    SpatialData(shapes=shapes, tables={TABLE: parsed}, attrs=attrs).write(zarr_path)
    return zarr_path


def _write_boundaries(path: Path, *, mirrored: bool) -> Path:
    def line(points: list[tuple[float, float]]) -> list[list[float]]:
        return [list(_mirror(*p) if mirrored else p) for p in points]

    features = [
        {
            "type": "Feature",
            "properties": {"role": "pial_boundary"},
            "geometry": {
                "type": "LineString",
                "coordinates": line([(0.0, PIA_Y_UM), (1000.0, PIA_Y_UM)]),
            },
        },
        {
            "type": "Feature",
            "properties": {"role": "grey_white_boundary"},
            "geometry": {
                "type": "LineString",
                "coordinates": line([(0.0, WM_Y_UM), (1000.0, WM_Y_UM)]),
            },
        },
    ]
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    return path


def _config(
    output_dir: Path,
    annotation_path: Path,
    *,
    zarr_path: Path | None = None,
    platform: str = "MERSCOPE",
    write_spatialdata_table: bool = False,
    **extra: Any,
) -> CorticalDepthConfig:
    return CorticalDepthConfig(
        dataset_name=f"S1_{platform}",
        platform=platform,
        latest_zarr_path=zarr_path or output_dir / "missing.zarr",
        output_dir=output_dir,
        tables=[
            CorticalDepthTableConfig(
                segmentation="proseg_hybrid", table_key=TABLE, shape_key=NATIVE
            )
        ],
        annotation_path=annotation_path,
        raster_resolution_um=10.0,
        streamline_spacing_um=100.0,
        write_spatialdata_table=write_spatialdata_table,
        **extra,
    )


def _read_outputs(output_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    cells = pd.read_parquet(
        next((output_dir / "proseg_hybrid").glob("*_cells_with_cortical_depth.parquet"))
    )
    summary = json.loads((output_dir / "cortical_depth_qc_summary.json").read_text())
    return cells, summary

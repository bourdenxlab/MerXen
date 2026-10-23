"""Known-coordinate and stable-identifier checks for optional pciSeq typing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from shapely.geometry import Polygon, box
from spatialdata.models import PointsModel, ShapesModel, TableModel
from spatialdata.transformations import Affine, Identity

from merxen.analysis.pciseq import (
    finalize_pciseq,
    prepare_pciseq,
    prepare_pciseq_inputs,
)
from merxen.config import PciSeqConfig
from merxen.io.spatialdata_schema import MERXEN_SCHEMA_ATTR

LARGE_ID = 2**53 + 7


def _config(tmp_path: Path, **changes: object) -> PciSeqConfig:
    values: dict[str, object] = {
        "pair_id": "toy_pair",
        "sample_id": "toy_sample",
        "platform": "XENIUM",
        "segmentation": "reseg",
        "source_h5ad": tmp_path / "source.h5ad",
        "spatialdata_path": tmp_path / "spatialdata.zarr",
        "reference_path": tmp_path / "reference.csv",
        "write_spatialdata_table": False,
    }
    values.update(changes)
    return PciSeqConfig.model_validate(values)


def _reference() -> pd.DataFrame:
    return pd.DataFrame(
        {"type_a": [20.0, 1.0], "type_b": [1.0, 20.0]}, index=["A", "B"]
    )


def _data(
    *,
    point_frame: pd.DataFrame | None = None,
    polygons: list[object] | None = None,
    point_affine: object | None = None,
    shape_affine: object | None = None,
) -> SimpleNamespace:
    frame = (
        point_frame
        if point_frame is not None
        else pd.DataFrame(
            {
                "x": [1.0, 7.0, 12.0, 2.0, 3.0, np.nan],
                "y": [1.0, 1.0, 1.0, 2.0, 2.0, 1.0],
                "gene": ["A", "B", "A", "Blank-1", "absent", "A"],
                "transcript_id": np.arange(21, 27, dtype=np.uint64),
                "assignment": [17, LARGE_ID, None, 17, 17, 17],
            }
        )
    )
    points = PointsModel.parse(
        frame,
        coordinates={axis: axis for axis in ("x", "y", "z") if axis in frame},
        feature_key="gene",
        transformations={"global": point_affine or Identity()},
    )
    geometry = gpd.GeoDataFrame(
        {"instance_id": np.array([17, LARGE_ID], dtype=np.uint64)},
        index=pd.Index([17, LARGE_ID], dtype=np.uint64),
        geometry=polygons or [box(0, 0, 4, 4), box(6, 0, 10, 4)],
    )
    shapes = ShapesModel.parse(
        geometry, transformations={"global": shape_affine or Identity()}
    )
    registry = {
        branch: {
            "shape": f"shapes_{branch}",
            "points": "transcripts",
            "table": f"table_{branch}",
        }
        for branch in ("proseg", "original", "cellpose", "proseg_hybrid")
    }
    return SimpleNamespace(
        points={"transcripts": points},
        shapes={entry["shape"]: shapes for entry in registry.values()},
        attrs={
            MERXEN_SCHEMA_ATTR: {
                "primary_points": "transcripts",
                "segmentations": registry,
            }
        },
    )


def _source(path: Path, *, region: str = "shapes_proseg") -> ad.AnnData:
    source = ad.AnnData(
        X=np.array([[3, 7]]),
        obs=pd.DataFrame(
            {
                "instance_id": np.array([LARGE_ID], dtype=np.uint64),
                "region": pd.Categorical([region]),
                "leiden": ["2"],
            },
            index=["large_cell"],
        ),
        var=pd.DataFrame(index=["A", "B"]),
    )
    source.layers["counts"] = source.X.copy()
    source.obsm["X_umap"] = np.array([[1.0, 2.0]])
    source = TableModel.parse(
        source, region=region, region_key="region", instance_key="instance_id"
    )
    source.write_h5ad(path)
    return source


def test_projection_retains_unassigned_spots_and_large_stable_ids(
    tmp_path: Path,
) -> None:
    """Expression QC and ProSeg assignment must not remove model transcripts."""
    inputs = prepare_pciseq_inputs(
        _data(), config=_config(tmp_path), reference=_reference()
    )
    assert inputs.cell_id_map.instance_id.tolist() == [17, LARGE_ID]
    assert inputs.cell_id_map.label.tolist() == [1, 2]
    assert inputs.spot_id_map.transcript_id.tolist() == [21, 22, 23]
    assert inputs.spots.gene_name.tolist() == ["A", "B", "A"]
    assert inputs.metadata["control_spots"] == 1
    assert inputs.metadata["unmatched_gene_spots"] == 1
    assert inputs.metadata["invalid_spots"] == 1
    assert "z_plane" not in inputs.spots
    pixels = inputs.planes[0].tocsr()
    assert pixels[int(inputs.spots.y.iloc[0]), int(inputs.spots.x.iloc[0])] == 1
    assert pixels[int(inputs.spots.y.iloc[1]), int(inputs.spots.x.iloc[1])] == 2
    assert pixels[int(inputs.spots.y.iloc[2]), int(inputs.spots.x.iloc[2])] == 0


@pytest.mark.parametrize(
    "segmentation,branch",
    [
        ("reseg", "proseg"),
        ("original_seg", "original"),
        ("proseg_mask", "cellpose"),
        ("proseg_hybrid", "proseg_hybrid"),
    ],
)
def test_projection_selects_matching_native_registry_branch(
    tmp_path: Path, segmentation: str, branch: str
) -> None:
    inputs = prepare_pciseq_inputs(
        _data(),
        config=_config(tmp_path, segmentation=segmentation),
        reference=_reference(),
    )
    assert inputs.metadata["shape_key"] == f"shapes_{branch}"
    assert inputs.metadata["source_table_key"] == f"table_{branch}"


def test_separate_affines_align_points_and_shapes(tmp_path: Path) -> None:
    """Different local frames map independently into the same physical frame."""
    point_affine = Affine(
        np.array([[2, 0, 20], [0, 3, -10], [0, 0, 1]]),
        input_axes=("x", "y"),
        output_axes=("x", "y"),
    )
    shape_affine = Affine(
        np.array([[2, 0, 18], [0, 3, -13], [0, 0, 1]]),
        input_axes=("x", "y"),
        output_axes=("x", "y"),
    )
    data = _data(
        point_affine=point_affine,
        shape_affine=shape_affine,
        polygons=[box(1, 1, 5, 5), box(7, 1, 11, 5)],
    )
    inputs = prepare_pciseq_inputs(
        data, config=_config(tmp_path), reference=_reference()
    )
    pixels = inputs.planes[0].tocsr()
    assert pixels[int(inputs.spots.y.iloc[0]), int(inputs.spots.x.iloc[0])] == 1
    assert pixels[int(inputs.spots.y.iloc[1]), int(inputs.spots.x.iloc[1])] == 2


@pytest.mark.parametrize("platform,scale", [("XENIUM", 4.7), ("MERSCOPE", 9.25)])
def test_intrinsic_vendor_boundaries_keep_native_micron_spot_alignment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
    scale: float,
) -> None:
    """Legacy image-grid metadata must not rescale explicitly native inputs."""
    import spatialdata as sd

    vendor_affine = np.diag([scale, scale, 1.0])
    data = _data(
        shape_affine=Affine(
            vendor_affine, input_axes=("x", "y"), output_axes=("x", "y")
        )
    )
    shape_key = f"{platform.lower()}_cell_boundaries"
    data.shapes[shape_key] = data.shapes.pop("shapes_original")
    data.attrs[MERXEN_SCHEMA_ATTR]["segmentations"]["original"]["shape"] = shape_key
    config = _config(
        tmp_path,
        platform=platform,
        segmentation="original_seg",
        coordinate_system="intrinsic",
    )
    _source(config.source_h5ad, region=shape_key)
    _reference().to_csv(config.reference_path)
    monkeypatch.setattr(sd, "read_zarr", lambda path: data)
    prepared = tmp_path / "prepared"
    prepare_pciseq(config, prepared)

    manifest = json.loads((prepared / "manifest.json").read_text())
    assert manifest["coordinate_system"] == "intrinsic"
    np.testing.assert_array_equal(manifest["point_affine"], np.eye(3))
    np.testing.assert_array_equal(manifest["shape_affine"], np.eye(3))
    spots = pd.read_csv(prepared / "spots.csv")
    pixels = sparse.load_npz(prepared / "masks" / "plane_0000.npz").tocsr()
    assert pixels[int(spots.y.iloc[0]), int(spots.x.iloc[0])] == 1
    assert pixels[int(spots.y.iloc[1]), int(spots.x.iloc[1])] == 2

    transformed = prepare_pciseq_inputs(
        data,
        config=config.model_copy(update={"coordinate_system": "global"}),
        reference=_reference(),
    )
    np.testing.assert_array_equal(transformed.metadata["shape_affine"], vendor_affine)
    global_pixels = transformed.planes[0].tocsr()
    assert (
        global_pixels[
            int(transformed.spots.y.iloc[1]), int(transformed.spots.x.iloc[1])
        ]
        == 1
    )


def test_polygon_holes_and_shared_edges_are_preserved(tmp_path: Path) -> None:
    first = Polygon(
        [(0, 0), (4, 0), (4, 4), (0, 4)], holes=[[(1, 1), (3, 1), (3, 3), (1, 3)]]
    )
    data = _data(polygons=[first, box(4, 0, 10, 4)])
    inputs = prepare_pciseq_inputs(
        data, config=_config(tmp_path), reference=_reference()
    )
    origin = inputs.metadata["origin_um"]
    mask = inputs.planes[0].tocsr()
    assert mask[int(2 - origin[1]), int(2 - origin[0])] == 0
    assert inputs.metadata["raster_boundary_ties"] > 0


def test_overlapping_geometry_and_large_raster_fail_before_compute(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="overlap"):
        prepare_pciseq_inputs(
            _data(polygons=[box(0, 0, 7, 4), box(6, 0, 10, 4)]),
            config=_config(tmp_path, overlap_policy="error"),
            reference=_reference(),
        )
    with pytest.raises(ValueError, match="max_pixels"):
        prepare_pciseq_inputs(
            _data(), config=_config(tmp_path, max_pixels=10), reference=_reference()
        )


def test_feature_type_can_keep_control_named_real_gene(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "x": [1.0],
            "y": [1.0],
            "gene": ["antisense_A"],
            "transcript_id": [1],
            "is_gene": [True],
        }
    )
    reference = pd.DataFrame({"type_a": [1.0]}, index=["antisense_A"])
    inputs = prepare_pciseq_inputs(
        _data(point_frame=frame), config=_config(tmp_path), reference=reference
    )
    assert inputs.spots.gene_name.tolist() == ["antisense_A"]


def test_real_3d_volume_maps_measured_coordinates_without_extrusion(
    tmp_path: Path,
) -> None:
    volume = np.zeros((3, 6, 12), dtype=np.uint64)
    volume[:, 1:4, 1:4] = 17
    volume[:, 1:4, 7:10] = LARGE_ID
    volume_path = tmp_path / "labels.npy"
    np.save(volume_path, volume)
    frame = pd.DataFrame(
        {
            "x": [11.0, 17.0, 11.0],
            "y": [21.0, 21.0, 21.0],
            "z": [32.0, 34.0, 39.0],
            "gene": ["A", "B", "A"],
            "transcript_id": [51, 52, 53],
        }
    )
    config = _config(
        tmp_path,
        geometry_mode="labels_3d",
        labels_path=volume_path,
        origin_um=[10, 20, 30],
        voxel_size_um=[1, 1, 2],
    )
    inputs = prepare_pciseq_inputs(
        _data(point_frame=frame), config=config, reference=_reference()
    )
    assert inputs.spots.z_plane.tolist() == [1.0, 2.0]
    assert inputs.spots.x.tolist() == [1.0, 7.0]
    assert inputs.spot_id_map.transcript_id.tolist() == [51, 52]
    assert inputs.metadata["out_of_bounds_spots"] == 1
    assert len(inputs.planes) == 3
    assert inputs.cell_id_map.instance_id.tolist() == [17, LARGE_ID]
    no_z = frame.drop(columns="z")
    with pytest.raises(KeyError, match="z_location"):
        prepare_pciseq_inputs(
            _data(point_frame=no_z), config=config, reference=_reference()
        )


def test_intrinsic_3d_grid_uses_native_xyz_without_declared_point_transform(
    tmp_path: Path,
) -> None:
    """A native micron volume and measured XYZ share the chosen intrinsic frame."""
    volume = np.zeros((3, 6, 12), dtype=np.uint64)
    volume[:, 1:4, 1:4] = 17
    volume[:, 1:4, 7:10] = LARGE_ID
    path = tmp_path / "labels.npy"
    np.save(path, volume)
    frame = pd.DataFrame(
        {
            "x": [11.0, 17.0],
            "y": [21.0, 21.0],
            "z": [32.0, 34.0],
            "gene": ["A", "B"],
            "transcript_id": [51, 52],
        }
    )
    data = _data(
        point_frame=frame,
        point_affine=Affine(
            np.diag([10.0, 10.0, 10.0, 1.0]),
            input_axes=("x", "y", "z"),
            output_axes=("x", "y", "z"),
        ),
    )
    config = _config(
        tmp_path,
        coordinate_system="intrinsic",
        geometry_mode="labels_3d",
        labels_path=path,
        origin_um=[10, 20, 30],
        voxel_size_um=[1, 1, 2],
    )
    inputs = prepare_pciseq_inputs(data, config=config, reference=_reference())
    assert inputs.spots.x.tolist() == [1.0, 7.0]
    assert inputs.spots.y.tolist() == [1.0, 1.0]
    assert inputs.spots.z_plane.tolist() == [1.0, 2.0]
    assert inputs.metadata["coordinate_system"] == "intrinsic"
    np.testing.assert_array_equal(inputs.metadata["point_affine"], np.eye(4))
    assert inputs.metadata["out_of_bounds_spots"] == 0


def test_volume_rejects_mismatched_ids(tmp_path: Path) -> None:
    path = tmp_path / "labels.npy"
    np.save(path, np.full((2, 2, 2), 99, dtype=np.uint32))
    frame = pd.DataFrame(
        {"x": [1.0], "y": [1.0], "z": [1.0], "gene": ["A"], "transcript_id": [1]}
    )
    with pytest.raises(ValueError, match="exactly match"):
        prepare_pciseq_inputs(
            _data(point_frame=frame),
            config=_config(tmp_path, geometry_mode="labels_3d", labels_path=path),
            reference=_reference(),
        )


def _prepare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[PciSeqConfig, Path]:
    import spatialdata as sd

    config = _config(tmp_path)
    _source(config.source_h5ad)
    _reference().to_csv(config.reference_path)
    monkeypatch.setattr(sd, "read_zarr", lambda path: _data())
    prepared = tmp_path / "prepared"
    prepare_pciseq(config, prepared)
    return config, prepared


def _computed(tmp_path: Path, *, empty: bool = False) -> Path:
    result = tmp_path / "computed"
    result.mkdir()
    cells = [
        {
            "Cell_Num": 2,
            "ClassName": [] if empty else ["type_b", "type_a"],
            "Prob": [] if empty else [0.8, 0.199],
        },
        {"Cell_Num": 1, "ClassName": ["Zero"], "Prob": [0.999]},
    ]
    spots = [
        {"spot_id": 2, "gene_name": "A", "neighbour": 0},
        {"spot_id": 0, "gene_name": "A", "neighbour": 1},
        {"spot_id": 1, "gene_name": "B", "neighbour": 2},
    ]
    (result / "cell_data.json").write_text(json.dumps(cells))
    (result / "gene_data.json").write_text(json.dumps(spots))
    (result / "compute_manifest.json").write_text(json.dumps({"has_converged": None}))
    return result


def test_prepare_and_finalize_preserve_clustered_subset_and_stable_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, prepared = _prepare(tmp_path, monkeypatch)
    assert sparse.load_npz(prepared / "masks" / "plane_0000.npz").nnz > 0
    manifest = json.loads((prepared / "manifest.json").read_text())
    assert manifest["cell_count"] == 2
    assert manifest["clustered_cell_count"] == 1
    result = finalize_pciseq(config, prepared, _computed(tmp_path), tmp_path / "final")
    annotated = ad.read_h5ad(result["annotated_h5ad"])
    assert annotated.obs.instance_id.tolist() == [LARGE_ID]
    assert annotated.obs.pciseq_cell_type.tolist() == ["type_b"]
    assert annotated.obs.leiden.tolist() == ["2"]
    assert annotated.X.tolist() == [[3, 7]]
    assert annotated.layers["counts"].tolist() == [[3, 7]]
    assert annotated.obsm["X_umap"].tolist() == [[1.0, 2.0]]
    assert annotated.obsm["pciseq_reported_probabilities"].shape == (1, 3)
    assert annotated.uns["spatialdata_attrs"]["region"] == "shapes_proseg"
    spot_ids = pd.read_csv(
        result["spot_assignments"],
        dtype={"instance_id": "UInt64", "transcript_id": "uint64"},
    ).set_index("spot_id")
    assert int(spot_ids.loc[1, "instance_id"]) == LARGE_ID
    assert int(spot_ids.loc[1, "transcript_id"]) == 22
    assert pd.isna(spot_ids.loc[2, "instance_id"])


def test_reference_preparation_manifest_is_embedded_and_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Atlas and regional provenance survives both model process boundaries."""
    import spatialdata as sd

    origin_path = tmp_path / "reference_origin.json"
    config = _config(tmp_path, reference_manifest_path=origin_path)
    _source(config.source_h5ad)
    _reference().to_csv(config.reference_path)
    origin = {
        "source": "whb",
        "region_labels": ["Human ACC"],
        "level": "cluster",
        "donor_balance": True,
        "reference_sha256": hashlib.sha256(
            config.reference_path.read_bytes()
        ).hexdigest(),
    }
    origin_path.write_text(json.dumps(origin, indent=2) + "\n")
    monkeypatch.setattr(sd, "read_zarr", lambda path: _data())
    prepared = tmp_path / "prepared"

    prepare_pciseq(config, prepared)

    prepared_metadata = json.loads((prepared / "manifest.json").read_text())
    assert prepared_metadata["reference_preparation"] == origin
    assert (
        prepared / "reference_manifest.json"
    ).read_bytes() == origin_path.read_bytes()
    result = finalize_pciseq(config, prepared, _computed(tmp_path), tmp_path / "final")
    assert (tmp_path / "final" / "reference_manifest.json").read_bytes() == (
        origin_path.read_bytes()
    )
    annotated = ad.read_h5ad(result["annotated_h5ad"])
    embedded = json.loads(annotated.uns["merxen_pciseq"]["prepared_manifest_json"])
    assert embedded["reference_preparation"] == origin


@pytest.mark.parametrize("checksum", [None, "not-a-sha256", "0" * 64])
def test_reference_manifest_checksum_fails_before_spatialdata_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checksum: str | None,
) -> None:
    """A missing, malformed or mismatched origin cannot label another matrix."""
    import spatialdata as sd

    origin_path = tmp_path / "reference_origin.json"
    config = _config(tmp_path, reference_manifest_path=origin_path)
    _source(config.source_h5ad)
    _reference().to_csv(config.reference_path)
    origin = {} if checksum is None else {"reference_sha256": checksum}
    origin_path.write_text(json.dumps(origin))

    def forbidden_read(path: Path) -> None:
        pytest.fail("SpatialData was opened before checking reference provenance")

    monkeypatch.setattr(sd, "read_zarr", forbidden_read)

    with pytest.raises(ValueError):
        prepare_pciseq(config, tmp_path / "prepared")
    assert not (tmp_path / "prepared").exists()


def test_empty_reported_class_summary_becomes_uncertain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, prepared = _prepare(tmp_path, monkeypatch)
    result = finalize_pciseq(
        config, prepared, _computed(tmp_path, empty=True), tmp_path / "final"
    )
    annotated = ad.read_h5ad(result["annotated_h5ad"])
    assert annotated.obs.pciseq_status.tolist() == ["uncertain"]
    assert np.isnan(annotated.obs.pciseq_probability.iloc[0])


def test_reference_validation_rejects_duplicate_classes(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.reference_path.write_text("gene,type,type\nA,1,2\n")
    with pytest.raises(ValueError, match="unique cell type"):
        prepare_pciseq(config, tmp_path / "prepared")


def test_finalize_adds_separate_table_with_aligned_region(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import spatialdata as sd

    from merxen.io import spatialdata_io

    config, prepared = _prepare(tmp_path, monkeypatch)
    config.write_spatialdata_table = True
    source = ad.read_h5ad(prepared / "source.h5ad")
    aligned = "shapes_proseg_aligned_nonrigid"
    source.uns["spatialdata_attrs"]["region"] = aligned
    source.obs["region"] = pd.Categorical([aligned])
    source.write_h5ad(prepared / "source.h5ad")
    store = SimpleNamespace(shapes={aligned: object()}, labels={})
    monkeypatch.setattr(sd, "read_zarr", lambda path: store)
    writes: list[tuple[str, ad.AnnData]] = []
    monkeypatch.setattr(
        spatialdata_io,
        "write_or_replace_element",
        lambda sdata, key, kind, table, **kwargs: writes.append((key, table)),
    )
    finalize_pciseq(config, prepared, _computed(tmp_path), tmp_path / "final")
    assert writes[0][0] == "table_proseg_pciseq"
    assert writes[0][1].uns["spatialdata_attrs"]["region"] == aligned
    assert writes[0][1].obs.pciseq_cell_type.tolist() == ["type_b"]


def test_real_spatialdata_write_preserves_source_table_and_aligned_link(
    tmp_path: Path,
) -> None:
    """Round-trip the derived table through a real store and its writer lock."""
    import spatialdata as sd

    config = _config(tmp_path, write_spatialdata_table=True)
    source = _source(config.source_h5ad)
    data = _data()
    native = data.shapes["shapes_proseg"]
    aligned = "shapes_proseg_aligned_nonrigid"
    store = sd.SpatialData(
        points={"transcripts": data.points["transcripts"]},
        shapes={"shapes_proseg": native, aligned: native.copy()},
        tables={"table_proseg": source},
        attrs={
            MERXEN_SCHEMA_ATTR: {
                "primary_points": "transcripts",
                "segmentations": {
                    "proseg": data.attrs[MERXEN_SCHEMA_ATTR]["segmentations"]["proseg"]
                },
            }
        },
    )
    store.write(config.spatialdata_path)
    source.uns["spatialdata_attrs"]["region"] = aligned
    source.obs["region"] = pd.Categorical([aligned])
    source.write_h5ad(config.source_h5ad)
    _reference().to_csv(config.reference_path)
    prepared = tmp_path / "prepared"
    prepare_pciseq(config, prepared)
    paths = finalize_pciseq(config, prepared, _computed(tmp_path), tmp_path / "final")
    restored = sd.read_zarr(config.spatialdata_path)
    original = restored.tables["table_proseg"]
    derived = restored.tables["table_proseg_pciseq"]
    assert "pciseq_cell_type" not in original.obs
    assert derived.obs.pciseq_cell_type.tolist() == ["type_b"]
    assert derived.obs.instance_id.tolist() == [LARGE_ID]
    assert derived.uns["spatialdata_attrs"]["region"] == aligned
    np.testing.assert_array_equal(original.X, derived.X)
    assert (paths["summary"].parent / "cell_id_map.csv").exists()
    assert (paths["summary"].parent / "spot_id_map.csv").exists()
    manifest = json.loads((prepared / "manifest.json").read_text())
    assert len(manifest["reference_sha256"]) == 64


@pytest.mark.parametrize("corruption", ["cell_id", "spot_gene", "fingerprint"])
def test_finalize_rejects_corrupted_input_pairings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corruption: str
) -> None:
    config, prepared = _prepare(tmp_path, monkeypatch)
    computed = _computed(tmp_path)
    if corruption == "cell_id":
        path = computed / "cell_data.json"
        cells = json.loads(path.read_text())
        cells[0]["Cell_Num"] = 99
        path.write_text(json.dumps(cells))
    elif corruption == "spot_gene":
        path = computed / "gene_data.json"
        genes = json.loads(path.read_text())
        genes[0]["gene_name"] = "B"
        path.write_text(json.dumps(genes))
    else:
        (computed / "compute_manifest.json").write_text(
            json.dumps({"prepared_manifest_sha256": "incorrect"})
        )
    with pytest.raises(ValueError, match="identifiers|gene mapping|different prepared"):
        finalize_pciseq(config, prepared, computed, tmp_path / "final")


def test_prepare_rejects_class_without_retained_gene_expression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import spatialdata as sd

    config = _config(tmp_path)
    _source(config.source_h5ad)
    pd.DataFrame(
        {"type_a": [10, 0], "type_unmeasured": [0, 20]}, index=["A", "unmeasured"]
    ).to_csv(config.reference_path)
    monkeypatch.setattr(sd, "read_zarr", lambda path: _data())
    with pytest.raises(ValueError, match="positive expression over retained"):
        prepare_pciseq(config, tmp_path / "prepared")


def test_default_hybrid_overlap_becomes_background_prior(tmp_path: Path) -> None:
    """Keep both hybrid cells and let the model infer spots in contested regions."""
    data = _data(polygons=[box(0, 0, 7, 4), box(6, 0, 10, 4)])
    inputs = prepare_pciseq_inputs(
        data,
        config=_config(tmp_path, segmentation="proseg_hybrid"),
        reference=_reference(),
    )
    mask = inputs.planes[0].tocsr()
    origin = inputs.metadata["origin_um"]
    assert mask[int(1 - origin[1]), int(6 - origin[0])] == 0
    assert set(mask.data) == {1, 2}
    assert inputs.metadata["overlap_policy"] == "background"
    assert inputs.metadata["geometric_overlap_pair_count"] == 1
    assert inputs.metadata["ambiguous_raster_pixels"] > 0
    assert data.shapes["shapes_proseg_hybrid"].geometry.iloc[0].area == 28


def test_fully_overlapped_cell_has_actionable_seed_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no unambiguous seed pixels"):
        prepare_pciseq_inputs(
            _data(polygons=[box(0, 0, 10, 4), box(6, 1, 7, 2)]),
            config=_config(tmp_path),
            reference=_reference(),
        )

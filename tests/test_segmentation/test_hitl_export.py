"""Native crop geometry, normalization, prefill, packaging and output protection."""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest
import spatialdata as sd
import tifffile
from spatialdata.models import Image2DModel, Labels2DModel
from spatialdata.transformations import Identity, Scale

from merxen.io.image_source import prepare_cellpose_input
from merxen.segmentation.cellpose import iter_core_tiles
from merxen.segmentation.hitl import export
from merxen.segmentation.hitl.export import (
    CropBox,
    HitlExportConfig,
    export_annotation_pilot,
    normalize_training_tile,
    prepare_viewer_roi_context,
    split_crop_labels,
)


@pytest.fixture
def pilot_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> HitlExportConfig:
    """A native two-channel source with a disconnected prefill label."""
    yy, xx = np.indices((1024, 1024))
    image = np.stack([(yy + xx) % 100, (yy * 3 + xx) % 300]).astype(np.uint16)
    labels = np.zeros((1024, 1024), np.uint32)
    labels[100:140, 100:140] = 51
    labels[160:190, 160:190] = 51
    labels[450:480, 100:140] = 99
    store = tmp_path / "source.zarr"
    sd.SpatialData(
        images={
            "morphology_focus": Image2DModel.parse(
                image,
                dims=("c", "y", "x"),
                c_coords=["DAPI", "18S"],
                transformations={"global": Identity()},
            )
        },
        labels={
            "cellpose_cell_labels": Labels2DModel.parse(
                labels, dims=("y", "x"), transformations={"global": Identity()}
            )
        },
        attrs={"merxen_atera": {"experiment": {"pixel_size": 0.21249222}}},
    ).write(store)
    stats = tmp_path / "stats.json"
    stats.write_text(
        json.dumps(
            {"height": 1024, "width": 1024, "tile_size": 768, "stitch_overlap_px": 32}
        )
    )
    monkeypatch.setattr(
        export, "require_cellpose_runtime", lambda: ("4.2.1.1", "2.14.0+cu130")
    )
    return HitlExportConfig(
        source_zarr=store,
        baseline_stats_path=stats,
        output_dir=tmp_path / "pilot",
        protected_roots=[tmp_path / "published"],
        crop_size_px=256,
        margin_px=32,
        boxes=[CropBox(y0=64, x0=64, size=256), CropBox(y0=400, x0=64, size=256)],
    )


def _file_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_pilot_exports_native_geometry_and_exact_halo_input(
    pilot_config: HitlExportConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = _file_hashes(pilot_config.source_zarr)
    original_read = sd.read_zarr
    selections = []

    def read(path: Path, **kwargs: object) -> sd.SpatialData:
        selections.append(kwargs.get("selection"))
        return original_read(path, **kwargs)

    monkeypatch.setattr(export.sd, "read_zarr", read)
    result = export_annotation_pilot(pilot_config)
    assert result.crop_count == 2
    assert selections == [("images", "labels")]
    assert _file_hashes(pilot_config.source_zarr) == before
    manifest = json.loads((result.output_dir / "manifest.json").read_text())
    assert manifest["evaluation_eligible"] is False
    assert manifest["training_normalize"] is False
    assert manifest["pixel_size_um"] == 0.21249222
    assert manifest["baseline_model_identity"]["status"] == "unverified"
    row = manifest["crops"][0]
    crop_id = row["crop_id"]
    saved = np.load(result.output_dir / f"{crop_id}_seg.npy", allow_pickle=True).item()
    assert set(np.unique(saved["masks"])) == {0, 1, 2}
    assert len(saved["ismanual"]) == 2 and not saved["ismanual"].any()
    assert tifffile.imread(result.output_dir / f"{crop_id}.tif").shape == (2, 256, 256)
    source = original_read(pilot_config.source_zarr, selection=("images",))
    native = source.images["morphology_focus"].data.compute()
    _, prepared, _ = prepare_cellpose_input(np.moveaxis(native[:, :736, :736], 0, -1))
    normalized, _ = normalize_training_tile(prepared)
    np.testing.assert_array_equal(
        np.load(result.output_dir / "model_input" / f"{crop_id}.npy"),
        np.moveaxis(normalized[64:320, 64:320], -1, 0),
    )
    for filename, digest in row["files_sha256"].items():
        assert (
            hashlib.sha256((result.output_dir / filename).read_bytes()).hexdigest()
            == digest
        )
    with tarfile.open(result.package_path) as archive:
        names = archive.getnames()
        assert "annotation-pilot/annotation-guide.md" in names
        assert "annotation-pilot/review.tsv" in names
        assert "annotation-pilot/context/overview.png" in names
        assert not any(
            Path(name).is_absolute() or ".." in Path(name).parts for name in names
        )
    (result.output_dir / f"{crop_id}_seg.npy").write_bytes(b"human edits")
    with pytest.raises(ValueError, match="edits are never replaced"):
        export_annotation_pilot(pilot_config)
    assert (result.output_dir / f"{crop_id}_seg.npy").read_bytes() == b"human edits"


def test_pilot_refuses_protected_alias_before_source_read(
    pilot_config: HitlExportConfig, tmp_path: Path
) -> None:
    protected = pilot_config.protected_roots[0]
    protected.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(protected, target_is_directory=True)
    pilot_config.output_dir = alias / "pilot"
    pilot_config.source_zarr = tmp_path / "missing.zarr"
    with pytest.raises(ValueError, match="separate from source/protected"):
        export_annotation_pilot(pilot_config)
    assert not (protected / "pilot").exists()


def test_boundary_stain_is_display_only_when_model_channels_are_dapi_and_18s(
    pilot_config: HitlExportConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = sd.read_zarr(pilot_config.source_zarr, selection=("images", "labels"))
    native = source.images["morphology_focus"].data.compute()
    boundary = np.full((1, *native.shape[1:]), 60000, dtype=native.dtype)
    source.images["morphology_focus"] = Image2DModel.parse(
        np.concatenate([native, boundary]),
        dims=("c", "y", "x"),
        c_coords=["DAPI", "18S", "ATP1A1/CD45/E-Cadherin"],
        transformations={"global": Identity()},
    )
    monkeypatch.setattr(export.sd, "read_zarr", lambda *a, **k: source)
    result = export_annotation_pilot(pilot_config)
    manifest = json.loads((result.output_dir / "manifest.json").read_text())
    assert manifest["model_channels"] == ["DAPI", "18S"]
    assert manifest["view_channels"] == ["DAPI", "18S", "ATP1A1/CD45/E-Cadherin"]
    crop_id = manifest["crops"][0]["crop_id"]
    display = tifffile.imread(result.output_dir / f"{crop_id}.tif")
    assert np.all(display[2] == 60000)
    _, prepared, _ = prepare_cellpose_input(np.moveaxis(native[:, :736, :736], 0, -1))
    normalized, _ = normalize_training_tile(prepared)
    model = np.load(result.output_dir / "model_input" / f"{crop_id}.npy")
    np.testing.assert_array_equal(model, np.moveaxis(normalized[64:320, 64:320], -1, 0))
    assert not model[2].any()


def test_pilot_refuses_changed_grid_before_writing(
    pilot_config: HitlExportConfig,
) -> None:
    stats = json.loads(pilot_config.baseline_stats_path.read_text())
    stats["width"] += 1
    pilot_config.baseline_stats_path.write_text(json.dumps(stats))
    with pytest.raises(ValueError, match="grid dimensions"):
        export_annotation_pilot(pilot_config)
    assert not pilot_config.output_dir.exists()


def test_pilot_refuses_translated_label_grid(
    pilot_config: HitlExportConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = sd.read_zarr(pilot_config.source_zarr)
    from spatialdata.transformations import set_transformation

    set_transformation(
        source.labels["cellpose_cell_labels"],
        Scale([2, 2], axes=("x", "y")),
        to_coordinate_system="global",
    )
    monkeypatch.setattr(export.sd, "read_zarr", lambda *a, **k: source)
    with pytest.raises(ValueError, match="different coordinate grids"):
        export_annotation_pilot(pilot_config)
    assert not pilot_config.output_dir.exists()


def test_tissue_spread_is_deterministic_and_never_uses_prefill(
    pilot_config: HitlExportConfig,
) -> None:
    pilot_config.boxes = None
    pilot_config.n_crops = 3
    thumbnail = np.full((64, 64), 20, np.float32)
    thumbnail[4:60, 4:60] = 100
    tiles = iter_core_tiles(2048, 2048, 768, 32)
    chosen = export._choose_boxes(pilot_config, tiles, thumbnail, (2048, 2048))
    assert chosen == export._choose_boxes(pilot_config, tiles, thumbnail, (2048, 2048))
    assert len(chosen) == 3
    # The weak scanner background must not count as tissue after log contrast.
    assert all(box.y0 >= 128 and box.x0 >= 128 for box, _ in chosen)
    with pytest.raises(ValueError, match="no usable tissue"):
        export._choose_boxes(pilot_config, tiles, np.zeros((64, 64)), (2048, 2048))


def test_normalization_preserves_zero_padding_and_records_percentiles() -> None:
    image = np.arange(40 * 45 * 2, dtype=np.uint16).reshape(40, 45, 2)
    _, prepared, _ = prepare_cellpose_input(image)
    normalized, record = normalize_training_tile(prepared)
    assert normalized.dtype == np.float32
    assert not normalized[..., 2].any()
    assert record["channels"][0]["stride"] == [1, 1, 1]
    assert record["batch_axis"]


def test_prefill_components_stay_separate_when_different_cells_touch() -> None:
    mask = np.array([[1, 1, 2, 2], [0, 0, 0, 0], [1, 1, 0, 0]], dtype=np.uint32)
    assert set(np.unique(split_crop_labels(mask))) == {0, 1, 2, 3}
    with pytest.raises(ValueError, match="integer mask"):
        split_crop_labels(mask.astype(float))


def test_normalization_reconstructs_fractional_percentiles_without_rounding() -> None:
    ramp = np.arange(11 * 17, dtype=np.uint8).reshape(11, 17)
    prepared = np.stack([ramp, ramp[::-1], np.zeros_like(ramp)], axis=-1)
    normalized, record = normalize_training_tile(prepared)
    assert record["channels"][0]["low"] == pytest.approx(1.86)
    assert normalized.shape == prepared.shape


def test_viewer_selections_export_exact_prefilled_crops_and_keep_roi_ids(
    pilot_config: HitlExportConfig, tmp_path: Path
) -> None:
    from merxen.segmentation.hitl.regions import write_viewer_regions

    path = tmp_path / "context.json"
    context = prepare_viewer_roi_context(pilot_config, path)
    rois = tmp_path / "selected.geojson"
    write_viewer_regions(
        rois, context, [np.array([[64, 64], [64, 320], [320, 320], [320, 64]], float)]
    )
    config = pilot_config.model_copy(update={"boxes": None, "roi_geojson_path": rois})
    result = export_annotation_pilot(config)
    manifest = json.loads((result.output_dir / "manifest.json").read_text())
    assert result.crop_count == 1
    assert manifest["sampling"]["method"] == "viewer_rois"
    assert manifest["crops"][0]["roi_id"] == "roi_001"
    assert manifest["crops"][0]["bbox_y0_y1_x0_x1"] == [64, 320, 64, 320]
    assert (
        result.output_dir / "selected_regions.geojson"
    ).read_bytes() == rois.read_bytes()
    stats = json.loads(pilot_config.baseline_stats_path.read_text())
    stats["tile_size"] = 800
    pilot_config.baseline_stats_path.write_text(json.dumps(stats))
    config.output_dir = tmp_path / "wrong-grid-export"
    with pytest.raises(ValueError, match="different source"):
        export_annotation_pilot(config)
    assert not config.output_dir.exists()


def test_roi_context_cannot_be_written_into_a_source_store(
    pilot_config: HitlExportConfig,
) -> None:
    with pytest.raises(ValueError, match="outside source/protected"):
        prepare_viewer_roi_context(
            pilot_config, pilot_config.source_zarr / "context.json"
        )


def test_density_halo_inputs_match_inference_before_cropping(
    pilot_config: HitlExportConfig,
) -> None:
    import pandas as pd
    from spatialdata.models import PointsModel

    from merxen.config import AteraTranscriptDensityConfig
    from merxen.segmentation.transcript_density import (
        create_transcript_density,
        image_with_density,
    )

    source = sd.read_zarr(pilot_config.source_zarr)
    ps = 0.21249222
    rng = np.random.default_rng(4)
    points = pd.DataFrame(
        {
            "x": rng.uniform(0, 1024 * ps, 50000).astype(np.float32),
            "y": rng.uniform(0, 1024 * ps, 50000).astype(np.float32),
            "qv": np.full(50000, 40, np.float32),
            "is_gene": np.ones(50000, bool),
        }
    )
    source.points["transcripts"] = PointsModel.parse(
        points,
        coordinates={"x": "x", "y": "y"},
        transformations={"global": Scale([1 / ps, 1 / ps], axes=("x", "y"))},
    )
    source.write_element("transcripts")
    density = create_transcript_density(
        AteraTranscriptDensityConfig(
            source_zarr_path=pilot_config.source_zarr,
            output_dir=pilot_config.output_dir.parent / "density",
        )
    )
    pilot_config.transcript_density_path = density
    result = export_annotation_pilot(pilot_config)
    manifest = json.loads((result.output_dir / "manifest.json").read_text())
    assert manifest["model_channels"] == ["DAPI", "18S", "transcript_density"]
    assert manifest["view_channels"] == manifest["model_channels"]
    assert manifest["transcript_density"]["profile"]["sigma_um"] == 0.15
    native = image_with_density(
        source, pilot_config.source_zarr, density
    ).data.compute()
    _, prepared, _ = prepare_cellpose_input(
        np.moveaxis(native[:, :736, :736], 0, -1), normalization="per_channel"
    )
    normalized, _ = normalize_training_tile(prepared)
    row = manifest["crops"][0]
    saved = np.load(result.output_dir / "model_input" / f"{row['crop_id']}.npy")
    np.testing.assert_array_equal(saved, np.moveaxis(normalized[64:320, 64:320], -1, 0))
    assert saved[2].any()


def test_multiple_tile_crops_keep_the_normalization_mode(
    pilot_config: HitlExportConfig,
) -> None:
    pilot_config.boxes = [
        CropBox(y0=64, x0=64, size=256),
        CropBox(y0=64, x0=736, size=256),
    ]
    result = export_annotation_pilot(pilot_config)
    manifest = json.loads((result.output_dir / "manifest.json").read_text())
    assert result.crop_count == 2
    assert manifest["input_normalization"] == "joint"
    assert len({row["tile_index"] for row in manifest["crops"]}) == 2

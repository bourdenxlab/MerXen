"""Xenium viewer density stays outside model inputs and preserves annotations."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import spatialdata as sd
from spatialdata.models import Labels2DModel

from merxen.config import TranscriptDensityConfig
from merxen.io.points_aggregates import read_root_attrs
from merxen.segmentation.density_storage import embedded_density_record
from merxen.segmentation.hitl.export import HitlExportConfig, viewer_export_settings
from merxen.segmentation.hitl.preparation import (
    HitlViewerPreparationConfig,
    prepare_hitl_viewer,
)
from merxen.segmentation.transcript_density import density_image_key
from test_segmentation.test_transcript_density import density_config as density_config


def test_prepare_xenium_preserves_masks_and_annotations_and_uses_two_model_channels(
    density_config: TranscriptDensityConfig, tmp_path: Path
) -> None:
    native = density_config.source_zarr_path
    (native / "experiment.xenium").write_text('{"pixel_size": 1.0}')
    store = sd.read_zarr(native)
    store.attrs["user_annotations"] = {"cortex": ["reviewed-region"]}
    store.write_attrs()
    labels = Labels2DModel.parse(np.ones((64, 64), np.uint32), dims=("y", "x"))
    store.labels["MOSAIK_cellpose_labels"] = labels
    store.write_element("MOSAIK_cellpose_labels")
    store.write_consolidated_metadata()
    source = tmp_path / "review.zarr"
    shutil.copytree(native, source)
    (source / "experiment.xenium").unlink()
    annotations = source / "annotations.geojson"
    annotations.write_text('{"type":"FeatureCollection","features":[]}')
    stats = tmp_path / "baseline.json"
    stats.write_text(
        json.dumps(
            {"height": 64, "width": 64, "tile_size": 6144, "stitch_overlap_px": 256}
        )
    )
    config = HitlViewerPreparationConfig(
        dataset_alias="HD1",
        source_zarr=source,
        native_source_zarr=native,
        baseline_stats_path=stats,
        output_dir=tmp_path / "prepared",
    )
    result = prepare_hitl_viewer(config)
    manifest = json.loads(result.read_text())
    assert manifest["protected_data_before"] == manifest["protected_data_after"]
    assert manifest["model_channels"] == ["DAPI", "18S"]
    assert manifest["density_usage"] == "viewer_only"
    record = embedded_density_record(source)
    assert record["platform"] == "XENIUM"
    assert record["totals"]["n_binned"] == 3
    assert (
        density_image_key("XENIUM")
        in sd.read_zarr(source, selection=("images",)).images
    )
    assert read_root_attrs(source)["user_annotations"] == {
        "cortex": ["reviewed-region"]
    }
    assert annotations.read_text() == '{"type":"FeatureCollection","features":[]}'
    assert (config.output_dir / "annotation_backup/annotations.geojson").is_file()
    settings = json.loads((source / "hitl_export_settings.json").read_text())
    assert settings["model_channels"] == ["DAPI", "18S"]
    assert settings["transcript_density"] is None
    context = json.loads((source / "merxen_hitl_roi_context.json").read_text())
    assert context["platform"] == "XENIUM"
    assert context["pixel_size_um"] == 1.0
    assert prepare_hitl_viewer(config) == result
    # Two-stain Xenium export must also remain the default without an explicit
    # channels field, even when density is present in the review store.
    export = HitlExportConfig(
        source_zarr=source,
        baseline_stats_path=stats,
        output_dir=tmp_path / "export",
        protected_roots=[source],
        platform="XENIUM",
    )
    assert viewer_export_settings(export)["model_channels"] == ["DAPI", "18S"]
    assert viewer_export_settings(export)["transcript_density"] is None


def test_density_key_rejects_unsupported_platform() -> None:
    with pytest.raises(ValueError, match="supports"):
        density_image_key("MERSCOPE")

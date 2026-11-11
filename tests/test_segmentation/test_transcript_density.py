"""Native pixel alignment, quality filtering, conservation and density identity."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import spatialdata as sd
from spatialdata.models import Image2DModel, PointsModel
from spatialdata.transformations import Identity, Scale

from merxen.config import AteraTranscriptDensityConfig
from merxen.segmentation.transcript_density import (
    DENSITY_CHANNEL,
    DensityCounts,
    DensityGeometry,
    accumulate_density_counts,
    create_transcript_density,
    density_profile,
    image_with_density,
    native_density_geometry,
    read_density_record,
    reconstruct_density_window,
    smooth_density_counts,
)


@pytest.fixture
def density_config(tmp_path: Path) -> AteraTranscriptDensityConfig:
    source = tmp_path / "source.zarr"
    frame = pd.DataFrame(
        {
            "x": np.array([2.1, 2.9, 4.1, 5.1, 7.0, -0.1, 64.0, np.nan], np.float32),
            "y": np.array([3.1, 3.9, 5.1, 6.1, 8.0, 3.1, 3.1, 3.1], np.float32),
            "z": [0, 9, 1, 2, 1, 0, 0, 0],
            "qv": [20, 40, 40, 19, 40, 40, 40, 40],
            "is_gene": [True, True, False, True, True, True, True, True],
            "original_assignment": [0] * 8,
        }
    )
    sd.SpatialData(
        images={
            "morphology_focus": Image2DModel.parse(
                np.ones((2, 64, 64), np.uint16),
                dims=("c", "y", "x"),
                c_coords=["DAPI", "18S"],
                transformations={"global": Identity()},
            )
        },
        points={
            "transcripts": PointsModel.parse(
                frame,
                coordinates={"x": "x", "y": "y", "z": "z"},
                transformations={"global": Scale([1, 1], axes=("x", "y"))},
            )
        },
        attrs={"merxen_atera": {"experiment": {"pixel_size": 1.0}}},
    ).write(source)
    return AteraTranscriptDensityConfig(
        source_zarr_path=source,
        output_dir=tmp_path / "density",
        sigma_um=1,
        bin_size_px=2,
        batch_rows=2,
        chunk_size_px=64,
    )


def test_counts_include_unassigned_all_z_and_exclude_controls_low_qv_and_outside(
    density_config: AteraTranscriptDensityConfig,
) -> None:
    geometry = native_density_geometry(density_config.source_zarr_path)
    assert geometry == DensityGeometry(64, 64, 1)
    counts = accumulate_density_counts(
        density_config.source_zarr_path, geometry, density_config
    )
    assert counts.n_rows == 8
    assert counts.n_eligible == 6
    assert counts.n_invalid_coordinates == 1
    assert counts.n_outside_image == 2
    assert counts.n_binned == int(counts.counts.sum()) == 3
    assert counts.counts[1, 1] == 2
    assert counts.counts[4, 3] == 1
    changed = density_config.model_copy(update={"batch_rows": 1000})
    other = accumulate_density_counts(changed.source_zarr_path, geometry, changed)
    np.testing.assert_array_equal(counts.counts, other.counts)
    assert density_profile(changed) == density_profile(density_config)


def test_global_smoothing_and_reconstruction_have_no_chunk_seams(
    density_config: AteraTranscriptDensityConfig,
) -> None:
    geometry = native_density_geometry(density_config.source_zarr_path)
    counts = accumulate_density_counts(
        density_config.source_zarr_path, geometry, density_config
    )
    density = smooth_density_counts(counts, geometry, density_config)
    assert density.sum() * 4 == pytest.approx(3, rel=1e-6)
    whole = reconstruct_density_window(
        density, origin=(0, 0), bin_size_px=2, window=(0, 64, 0, 64)
    )
    strips = [
        reconstruct_density_window(
            density, origin=(0, 0), bin_size_px=2, window=(y, y + 8, 0, 64)
        )
        for y in range(0, 64, 8)
    ]
    np.testing.assert_array_equal(whole, np.concatenate(strips))
    assert whole.dtype == np.float32
    # The first coarse centre is at native coordinate 0.5; preserve the phase.
    centres = np.arange(16, dtype=np.float32).reshape(4, 4)
    sampled = reconstruct_density_window(
        centres, origin=(0, 0), bin_size_px=2, window=(0, 8, 0, 8)
    )
    assert sampled[1, 1] == pytest.approx(1.25)


def test_density_store_preserves_source_and_binds_auxiliary_image(
    density_config: AteraTranscriptDensityConfig,
) -> None:
    import hashlib

    source = density_config.source_zarr_path
    before = {
        str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in source.rglob("*")
        if p.is_file()
    }
    target = create_transcript_density(density_config)
    record = read_density_record(target)
    assert record["totals"]["n_binned"] == 3
    assert record["profile"]["units"] == "transcripts_per_um2"
    sdata = sd.read_zarr(source, selection=("images",))
    image = image_with_density(sdata, source, target)
    assert image.shape == (3, 64, 64)
    assert list(image.c.values) == ["DAPI", "18S", DENSITY_CHANNEL]
    assert np.max(image.sel(c=DENSITY_CHANNEL).data.compute()) > 0
    after = {
        str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in source.rglob("*")
        if p.is_file()
    }
    assert before == after
    with pytest.raises(ValueError, match="new or empty"):
        create_transcript_density(density_config)
    payload = json.loads((target / "transcript_density.json").read_text())
    payload["profile"]["sigma_um"] = 2
    (target / "transcript_density.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="manifest identity"):
        read_density_record(target)


def test_per_channel_conversion_preserves_density_and_legacy_joint_inputs() -> None:
    from merxen.io.image_source import prepare_cellpose_input

    yy, xx = np.indices((64, 72))
    stains = np.stack([10000 + yy * 100 + xx * 50, 30000 + xx * 100], -1).astype(
        np.float32
    )
    density = ((xx + yy) % 13).astype(np.float32) / 1000
    combined = np.concatenate([stains, density[..., None]], -1)
    _, historical, _ = prepare_cellpose_input(combined)
    _, prepared, _ = prepare_cellpose_input(combined, normalization="per_channel")
    assert not historical[..., 2].any()
    assert prepared[..., 2].max() >= 254
    low, high = np.percentile(stains, (2, 98))
    expected = (np.clip((stains - low) / (high - low + 1e-8), 0, 1) * 255).astype(
        np.uint8
    )
    _, legacy, _ = prepare_cellpose_input(stains)
    np.testing.assert_array_equal(legacy[..., :2], expected)
    assert not legacy[..., 2].any()


def test_partial_image_edge_bins_keep_physical_area(
    density_config: AteraTranscriptDensityConfig,
) -> None:
    geometry = DensityGeometry(3, 5, 1)
    counts = DensityCounts(np.array([[4, 4, 2], [2, 2, 1]], np.uint64), 0, 0)
    density = smooth_density_counts(counts, geometry, density_config)
    np.testing.assert_allclose(density, np.ones((2, 3)), rtol=1e-6)


def test_density_cli_runs_the_registered_stage(
    density_config: AteraTranscriptDensityConfig, tmp_path: Path
) -> None:
    from click.testing import CliRunner

    from merxen.cli import main

    path = tmp_path / "density.json"
    path.write_text(density_config.model_dump_json())
    result = CliRunner().invoke(
        main, ["atera-transcript-density", "--config", str(path)]
    )
    assert result.exit_code == 0, result.output
    assert (density_config.output_dir / "transcript_density.json").is_file()
    from merxen.segmentation.density_storage import has_embedded_density

    assert has_embedded_density(density_config.source_zarr_path)
    assert (density_config.output_dir / "transcript_density.zarr").is_symlink()


def test_stock_density_masks_require_exact_input_identity(tmp_path: Path) -> None:
    from merxen.config import CellposeConfig, SegmentationConfig
    from merxen.segmentation.hitl.registry import require_matching_cached_model
    from merxen.segmentation.model_identity import (
        CPSAM_V2_SHA256,
        CellposeInferenceContext,
        CellposeModelIdentity,
        TranscriptDensityInputIdentity,
        write_model_identity,
    )

    config = SegmentationConfig.model_validate(
        {
            "dataset": {
                "name": "ATERA_SYNTH",
                "platform": "ATERA",
                "data_path": tmp_path / "source.zarr",
                "output_dir": tmp_path / "out",
                "channels": ["DAPI", "18S", "transcript_density"],
                "image_key": "morphology_focus",
                "proseg_transcripts": "none",
                "transcript_density_path": tmp_path / "density.zarr",
            },
            "cellpose": {"input_normalization": "per_channel"},
        }
    )
    expected = CellposeModelIdentity(
        status="verified",
        model_name="cpsam_v2",
        weights_sha256=CPSAM_V2_SHA256,
        cellpose_version="4.2.1.1",
        torch_version="2.14.1+cu130",
        cellpose_config=config.cellpose,
        inference_context=CellposeInferenceContext(
            platform="ATERA",
            species="human",
            channels=config.dataset.channels,
            image_key="morphology_focus",
            pixel_size_um=(1, 1),
            mask_filter=config.mask_filter,
            tiling=config.tiling,
            transcript_density=TranscriptDensityInputIdentity(
                fingerprint="1" * 64,
                profile={"sigma_um": 0.15},
                raster_path=config.dataset.transcript_density_path,
            ),
        ),
    )
    mask = tmp_path / "cellpose_masks_tiled.npy"
    np.save(mask, np.ones((2, 2), np.uint32))
    old = expected.model_copy(
        update={"inference_context": None, "cellpose_config": CellposeConfig()}
    )
    write_model_identity(mask, old)
    with pytest.raises(ValueError, match="identity mismatch"):
        require_matching_cached_model(mask, config, expected)
    write_model_identity(mask, expected)
    (tmp_path / "cellpose_stitching_stats.json").write_text(
        json.dumps({"model_identity": expected.model_dump(mode="json")})
    )
    require_matching_cached_model(mask, config, expected)
    relocated = expected.model_copy(deep=True)
    assert relocated.inference_context is not None
    assert relocated.inference_context.transcript_density is not None
    relocated.inference_context.transcript_density.raster_path = (
        tmp_path / "embedded.zarr"
    )
    require_matching_cached_model(mask, config, relocated)
    relocated.inference_context.transcript_density.fingerprint = "2" * 64
    with pytest.raises(ValueError, match="identity mismatch"):
        require_matching_cached_model(mask, config, relocated)
    two = config.model_copy(deep=True)
    two.dataset.channels = ["DAPI", "18S"]
    two.dataset.transcript_density_path = None
    two.cellpose.input_normalization = "joint"
    with pytest.raises(ValueError, match="use transcript density"):
        require_matching_cached_model(mask, two)

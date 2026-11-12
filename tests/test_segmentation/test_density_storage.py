"""Permanent density survives relocation, cache building and late enrichment."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import dask.array as da
import pytest
import spatialdata as sd
from spatialdata.models import Image2DModel
from spatialdata.transformations import Identity

from merxen.config import AteraTranscriptDensityConfig, EnrichmentConfig
from merxen.segmentation import transcript_density
from merxen.segmentation.density_storage import (
    DENSITY_IMAGE_KEY,
    embed_transcript_density,
    embedded_density_record,
    has_embedded_density,
    prepare_embedded_density,
)
from test_segmentation.test_transcript_density import density_config as density_config


def _files(root: Path) -> dict[str, tuple[str, int]]:
    return {
        str(path.relative_to(root)): (
            hashlib.sha256(path.read_bytes()).hexdigest(),
            path.stat().st_mtime_ns,
        )
        for path in root.rglob("*")
        if path.is_file()
    }


def test_embedded_density_is_independent_and_reusable_without_regeneration(
    density_config: AteraTranscriptDensityConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = density_config.source_zarr_path
    image_before = _files(source / "images/morphology_focus")
    points_before = _files(source / "points")
    alias = prepare_embedded_density(density_config)
    assert alias.resolve() == source.resolve()
    assert has_embedded_density(source)
    record = embedded_density_record(source)
    assert record["totals"]["n_binned"] == 3
    assert DENSITY_IMAGE_KEY in sd.read_zarr(source, selection=("images",)).images
    assert _files(source / "images/morphology_focus") == image_before
    assert _files(source / "points") == points_before
    from merxen.segmentation.hitl.export import HitlExportConfig, viewer_export_settings

    stats = density_config.output_dir / "baseline.json"
    stats.write_text("{}")
    pilot = HitlExportConfig(
        source_zarr=source,
        baseline_stats_path=stats,
        output_dir=density_config.output_dir.with_name("pilot"),
        protected_roots=[source],
    )
    settings = viewer_export_settings(pilot)
    assert settings["model_channels"] == ["DAPI", "18S", "transcript_density"]
    assert settings["transcript_density"]["fingerprint"] == record["fingerprint"]
    legacy = viewer_export_settings(
        pilot.model_copy(update={"channels": ["DAPI", "18S"]})
    )
    assert legacy["model_channels"] == ["DAPI", "18S"]
    assert legacy["transcript_density"] is None
    before = _files(source)

    def refuse_regeneration(*args: object, **kwargs: object) -> None:
        raise AssertionError("An existing matching raster must be reused")

    monkeypatch.setattr(
        transcript_density, "create_transcript_density", refuse_regeneration
    )
    second = prepare_embedded_density(
        density_config.model_copy(
            update={"output_dir": density_config.output_dir.with_name("resume-output")}
        )
    )
    assert second.resolve() == source.resolve()
    assert _files(source) == before
    shutil.rmtree(density_config.output_dir)
    assert embedded_density_record(source)["fingerprint"] == record["fingerprint"]
    chunk = next(
        path
        for path in (source / "images" / DENSITY_IMAGE_KEY).rglob("*")
        if path.is_file() and path.name != "zarr.json"
    )
    chunk.write_bytes(b"changed payload")
    with pytest.raises(ValueError, match="payload changed"):
        embedded_density_record(source)


def test_density_installation_is_additive_and_refuses_different_recipe(
    density_config: AteraTranscriptDensityConfig,
) -> None:
    raster = transcript_density.create_transcript_density(density_config)
    source = density_config.source_zarr_path
    from click.testing import CliRunner

    from merxen.cli import main

    result = CliRunner().invoke(
        main,
        [
            "embed-transcript-density",
            "--spatialdata",
            str(source),
            "--density",
            str(raster),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (
        embedded_density_record(source)["fingerprint"]
        == transcript_density.read_density_record(raster)["fingerprint"]
    )
    before = _files(source)
    embed_transcript_density(source, raster)
    assert _files(source) == before
    with pytest.raises(ValueError, match="different density recipe"):
        prepare_embedded_density(
            density_config.model_copy(
                update={
                    "sigma_um": 0.5,
                    "output_dir": density_config.output_dir.with_name("different"),
                }
            )
        )
    assert _files(source) == before


def test_density_and_complete_viewer_pyramid_travel_with_spatialdata(
    density_config: AteraTranscriptDensityConfig,
    tmp_path: Path,
) -> None:
    original = sd.read_zarr(density_config.source_zarr_path)
    source = tmp_path / "large.zarr"
    sd.SpatialData(
        images={
            "morphology_focus": Image2DModel.parse(
                da.zeros((2, 1028, 1028), chunks=(2, 256, 256), dtype="uint16"),
                dims=("c", "y", "x"),
                c_coords=["DAPI", "18S"],
                transformations={"global": Identity()},
            )
        },
        points=original.points,
        attrs=original.attrs,
    ).write(source)
    prepare_embedded_density(
        density_config.model_copy(
            update={
                "source_zarr_path": source,
                "output_dir": tmp_path / "large-density",
            }
        )
    )
    copied = tmp_path / "laptop.zarr"
    shutil.copytree(source, copied)
    stored = sd.read_zarr(copied, selection=("images",))
    assert DENSITY_IMAGE_KEY in stored.images
    caches = [key for key in stored.images if key.startswith("_napari_compare_")]
    assert len(caches) == 1
    import zarr

    from merxen.viewer_cache.format import DERIVED_CACHE_ATTR

    assert zarr.open_group(str(copied / "images" / caches[0])).attrs[
        DERIVED_CACHE_ATTR
    ]["complete"]
    assert (
        embedded_density_record(copied)["fingerprint"]
        == embedded_density_record(source)["fingerprint"]
    )
    prepared = transcript_density.image_with_density(stored, copied, copied)
    assert prepared.shape == (3, 1028, 1028)


def test_enrichment_installs_density_when_latest_was_created_before_it(
    density_config: AteraTranscriptDensityConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from merxen.enrichment import enrich

    source = density_config.source_zarr_path
    latest = tmp_path / "latest.zarr"
    shutil.copytree(source, latest)
    prepare_embedded_density(density_config)
    assert not has_embedded_density(latest)
    monkeypatch.setattr(enrich, "_enrich_native_in_place", lambda *args, **kwargs: None)
    config = EnrichmentConfig(
        dataset_name="PILOT",
        platform="ATERA",
        latest_zarr_path=latest,
        original_data_path=source,
        output_dir=tmp_path / "enriched",
        mask_path=None,
        nuclei_mask_path=None,
        segmentation_source="native",
        transcript_density_source_path=source,
    )
    enrich.enrich_single_latest(config)
    assert has_embedded_density(latest)
    assert (
        embedded_density_record(latest)["fingerprint"]
        == embedded_density_record(source)["fingerprint"]
    )

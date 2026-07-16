"""CLI command for pre-building the napari viewer's derived caches."""

from __future__ import annotations

import json
from pathlib import Path

import click

from merxen.config import ViewerCacheConfig, load_config_from_json
from merxen.viewer_cache import build_viewer_caches
from merxen.viewer_cache.build import ViewerCacheParams


@click.command(name="build-viewer-caches")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Path to JSON config validated against ViewerCacheConfig.",
)
@click.option(
    "--force-rerun",
    is_flag=True,
    default=False,
    help="Rebuild caches even when an up-to-date marker is already present.",
)
def build_viewer_caches_command(config_path: Path, force_rerun: bool) -> None:
    """Materialize label masks, pyramids, and outlines into the enriched zarr."""
    cfg = load_config_from_json(config_path, ViewerCacheConfig)
    assert isinstance(cfg, ViewerCacheConfig)

    params = ViewerCacheParams(
        downsample=cfg.downsample,
        label_chunk_size=cfg.label_chunk_size,
        contour_width=cfg.contour_width,
        min_size=cfg.min_size,
        shape_keys=tuple(cfg.shape_keys) if cfg.shape_keys is not None else None,
        build_image_pyramid=cfg.build_image_pyramid,
        force=force_rerun,
    )
    summary = build_viewer_caches(
        zarr_path=cfg.latest_zarr_path,
        platform=cfg.platform,
        original_data_path=cfg.original_data_path,
        transform_path=cfg.transform_path,
        params=params,
    )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = (
        cfg.output_dir / f"{cfg.dataset_name.lower()}_viewer_cache_summary.json"
    )
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")

    for shape_key, mask_result in summary.get("masks", {}).items():
        click.echo(f"{shape_key}: {mask_result}")
    click.echo(f"image_pyramid: {summary.get('image_pyramid')}")
    click.echo(f"Saved summary: {summary_path}")
    click.echo(f"Viewer caches complete: {cfg.latest_zarr_path}")

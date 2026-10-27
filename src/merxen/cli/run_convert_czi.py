"""CLI: convert one scene of a Zeiss CZI to a tiled pyramidal OME-TIFF."""

from __future__ import annotations

import json
import os
from pathlib import Path

import click

from merxen.io.channel_names import resolve_channel_names
from merxen.io.czi import (
    CZI_PYTHON_ENV,
    CziRuntimeUnavailableError,
    czi_channel_candidates,
    resolve_czi_pixel_size,
    resolve_scene,
    run_convert_czi_scene,
    run_read_czi_info,
)

_OME_SUFFIXES = (".ome.tif", ".ome.tiff")


def _default_workers() -> int:
    configured = int(os.environ.get("DASK_NUM_WORKERS") or 0)
    return configured or min(8, os.cpu_count() or 1)


@click.command(name="convert-czi")
@click.option(
    "--input",
    "input_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Zeiss CZI file.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path, dir_okay=False),
    required=True,
    help="Output .ome.tif / .ome.tiff path.",
)
@click.option(
    "--scene",
    default=None,
    help="Scene index or name (required when the CZI has several).",
)
@click.option(
    "--channel-names",
    default=None,
    help='Override all channel names, ";"-separated, in C order.',
)
@click.option(
    "--pixel-size-um",
    type=click.FloatRange(min=0.0, min_open=True),
    default=None,
    help="Pixel size in micrometres (default: the CZI's Scaling metadata).",
)
@click.option(
    "--image-name",
    default="image",
    show_default=True,
    help="OME image name written into the file.",
)
@click.option(
    "--workers",
    type=click.IntRange(min=1),
    default=None,
    help="Reader/compression threads; default $DASK_NUM_WORKERS or min(8, CPU count).",
)
@click.option(
    "--tile-size", type=click.IntRange(min=16), default=1024, show_default=True
)
@click.option(
    "--compression",
    type=click.Choice(["zlib", "zstd", "none"]),
    default="zlib",
    show_default=True,
)
@click.option(
    "--summary-json",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Write a JSON summary of the conversion here.",
)
@click.option(
    "--czi-python",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
    help=(
        f"Python of the CZI runtime; default ${CZI_PYTHON_ENV} when czifile is "
        "not installed here."
    ),
)
def convert_czi_command(
    input_path: Path,
    output_path: Path,
    scene: str | None,
    channel_names: str | None,
    pixel_size_um: float | None,
    image_name: str,
    workers: int | None,
    tile_size: int,
    compression: str,
    summary_json: Path | None,
    czi_python: Path | None,
) -> None:
    """Convert one scene of a Zeiss CZI to a tiled pyramidal OME-TIFF."""
    if not output_path.name.lower().endswith(_OME_SUFFIXES):
        raise click.BadParameter(
            f"{output_path.name!r} must end with .ome.tif or .ome.tiff",
            param_hint="--output",
        )
    try:
        info = run_read_czi_info(input_path, python=czi_python)
        scene_index = resolve_scene(info.scenes, scene)
        override = (
            [part.strip() for part in channel_names.split(";")]
            if channel_names is not None
            else None
        )
        resolved = resolve_channel_names(
            czi_channel_candidates(info), override=override
        )
        for warning in resolved.warnings:
            click.echo(warning, err=True)
        pixel_size, pixel_size_source = resolve_czi_pixel_size(info, pixel_size_um)
        n_workers = workers or _default_workers()
        band_height = -(-max(2048, tile_size) // tile_size) * tile_size
        result = run_convert_czi_scene(
            input_path,
            output_path,
            python=czi_python,
            scene=scene_index,
            channel_names=list(resolved.names),
            pixel_size_um=pixel_size,
            image_name=image_name,
            tile_size=tile_size,
            band_height=band_height,
            compression=compression,
            workers=n_workers,
        )
    except (ValueError, OSError, CziRuntimeUnavailableError) as exc:
        raise click.ClickException(str(exc)) from exc

    n_c, height, width = result.shape_cyx
    click.echo(
        f"Converted scene {scene_index} ({width}×{height} px, {n_c} channels: "
        f"{', '.join(resolved.names)}) -> {output_path}"
    )
    levels = ", ".join(f"{w}×{h}" for h, w in result.level_shapes_yx)
    click.echo(f"Levels ({len(result.level_shapes_yx)}): {levels}")

    if summary_json is not None:
        summary = {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "scene_index": result.scene_index,
            "scene_name": result.scene_name,
            "bbox_xywh": list(result.bbox_xywh),
            "shape_cyx": list(result.shape_cyx),
            "level_shapes_yx": [list(s) for s in result.level_shapes_yx],
            "channel_names": list(resolved.names),
            "name_sources": list(resolved.sources),
            "channel_name_warnings": list(resolved.warnings),
            "pixel_size_um": pixel_size,
            "pixel_size_source": pixel_size_source,
            "metadata_pixel_size_um": (
                None if info.pixel_size_um is None else list(info.pixel_size_um)
            ),
            "z_planes_projected": result.z_planes_projected,
            "pixel_type": result.pixel_type,
            "compression": result.compression,
            "czifile_version": result.czifile_version,
            "seconds": result.seconds,
        }
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

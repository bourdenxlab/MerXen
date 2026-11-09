#!/usr/bin/env python
"""Wrap verified stock weights as a candidate fixture to test custom-model plumbing."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from pathlib import Path

from merxen.config import CellposeModelPolicy, SegmentationConfig
from merxen.platforms import platform_spec
from merxen.segmentation.hitl.registry import (
    CellposeModelMetadata,
    require_safe_custom_outputs,
)
from merxen.segmentation.model_identity import CPSAM_V2_SHA256, require_cellpose_runtime
from merxen.segmentation.pipeline import _derive_cellpose_transforms

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Write a model.json, safe arm config and Nextflow parameters for testing.

    Args:
        argv: Command arguments, defaulting to the process command line.

    Returns:
        Zero after writing the complete fixture.

    Raises:
        ValueError: If weights are not stock, or output is unsafe/nonempty.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protected-root", type=Path, required=True)
    parser.add_argument("--source-zarr", type=Path, required=True)
    parser.add_argument("--pixel-size-um", type=float)
    parser.add_argument("--tile-size", type=int)
    parser.add_argument("--baseline-stats", type=Path)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    protected = args.protected_root.resolve()
    if output.is_relative_to(protected) or (
        output.exists() and (not output.is_dir() or any(output.iterdir()))
    ):
        raise ValueError(
            "Fixture output must be new/empty and outside the protected root"
        )
    config = SegmentationConfig.model_validate_json(args.config.read_text())
    config.dataset.data_path = args.source_zarr.resolve(strict=True)
    spec = platform_spec(config.dataset.platform)
    config.dataset.image_key = spec.native_image_key
    pixel_size = args.pixel_size_um
    if pixel_size is None:
        x_transform, y_transform = _derive_cellpose_transforms(config)
        sizes = [math.hypot(x_transform[i], y_transform[i]) for i in (0, 1)]
        if not math.isclose(*sizes, rel_tol=0.01):
            raise ValueError("Fixture requires isotropic native pixels")
        pixel_size = sizes[0]
    tile_size = args.tile_size
    if args.baseline_stats is not None:
        stats = json.loads(args.baseline_stats.read_text())
        if tile_size is not None and tile_size != stats["tile_size"]:
            raise ValueError("Fixture tile size differs from the baseline grid")
        tile_size = int(stats["tile_size"])
        config.tiling.stitch_overlap_px = int(stats["stitch_overlap_px"])
    if tile_size is None:
        if len(config.tiling.tile_size_candidates) != 1:
            raise ValueError("Specify --baseline-stats or --tile-size to pin the grid")
        tile_size = config.tiling.tile_size_candidates[0]
    weights = args.weights.resolve(strict=True)
    with weights.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != CPSAM_V2_SHA256:
        raise ValueError("Fixture requires the verified stock cpsam_v2 checkpoint")
    cellpose_version, torch_version = require_cellpose_runtime()
    metadata = CellposeModelMetadata(
        weights_sha256=digest,
        platform=config.dataset.platform,
        species=config.cellpose_model_policy.species,
        channels=config.dataset.channels,
        image_key=spec.native_image_key,
        pixel_size_um=pixel_size,
        tile_size=tile_size,
        normalization_route="A",
        annotation_style="stock-plumbing-fixture-only",
        annotation_guide_version="none-no-annotations",
        license_note="Stock cpsam_v2 testing fixture; not a trained or accepted model",
        status="candidate",
        cellpose_version=cellpose_version,
        torch_version=torch_version,
        evaluated_cellpose=config.cellpose.model_copy(deep=True),
        evaluated_mask_filter=config.mask_filter.model_copy(deep=True),
    )
    config.dataset.output_dir = output / "arm"
    for name in type(config.dataset).model_fields:
        if name.startswith("persistent_"):
            setattr(config.dataset, name, None)
    config.cellpose.pretrained_model = str(output / "model")
    config.cellpose.model_sha256 = digest
    config.cellpose.model_metadata_path = output / "model.json"
    config.tiling.tile_size_candidates = [tile_size]
    config.cellpose_model_policy = CellposeModelPolicy(
        species=config.cellpose_model_policy.species,
        experimental_arm=True,
        allow_candidate_model=True,
        protected_output_roots=[protected],
    )
    require_safe_custom_outputs(config, include_source=False)
    output.mkdir(parents=True, exist_ok=True)
    # Only weights are linked, read-only by all inference callers. Published
    # masks, CSVs and zarrs are never linked into the experimental arm.
    (output / "model").symlink_to(weights)
    (output / "model.json").write_text(metadata.model_dump_json(indent=2) + "\n")
    (output / "segment_config.json").write_text(config.model_dump_json(indent=2) + "\n")
    (output / "nextflow-params.json").write_text(
        json.dumps(
            {
                f"{spec.name.lower()}_cellpose_model": str(output),
                "allow_candidate_model": True,
                "cellpose_tile_size_candidates": [tile_size],
                "cellpose_protected_output_roots": [str(protected)],
                **(
                    {"segmentation_source": "cellpose_proseg"}
                    if spec.name == "ATERA"
                    else {}
                ),
            },
            indent=2,
        )
        + "\n"
    )
    logger.info("Stock candidate fixture ready: %s", output)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())

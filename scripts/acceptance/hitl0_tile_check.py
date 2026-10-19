#!/usr/bin/env python
"""Compare one production-context Cellpose tile with a read-only published mask.

Run through the accompanying Nextflow workflow so GPU inference holds the same
host lock as production. This is a drift diagnostic, not HITL model evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from merxen.config import SegmentationConfig
from merxen.segmentation.cellpose import (
    build_cellpose_affine_to_microns,
    build_cellpose_model,
    iter_core_tiles,
    run_cellpose_tile,
)
from merxen.segmentation.hitl.registry import (
    has_custom_model,
    is_experimental_arm,
    preflight_custom_model,
    preflight_stock_arm,
)
from merxen.segmentation.mask_filter import (
    filter_labeled_mask_by_area,
)
from merxen.segmentation.model_identity import (
    identify_cellpose_model,
    write_model_identity,
)
from merxen.segmentation.pipeline import _load_dataset_sdata

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TileAgreement:
    """Label-independent agreement with the published stock segmentation."""

    reference_cells: int
    predicted_cells: int
    true_positive: int
    false_positive: int
    false_negative: int
    ap_at_05: float
    mean_matched_iou: float
    foreground_iou: float
    minimum_agreement: float
    passed: bool


def _dense_labels(mask: np.ndarray) -> np.ndarray:
    labels, inverse = np.unique(mask, return_inverse=True)
    result: np.ndarray = inverse.reshape(mask.shape).astype(np.uint32)
    if labels[0] != 0:
        result += 1
    return result


def _exclude_border_labels(mask: np.ndarray) -> np.ndarray:
    border = np.unique(np.concatenate((mask[0], mask[-1], mask[:, 0], mask[:, -1])))
    return _dense_labels(np.where(np.isin(mask, border[border > 0]), 0, mask))


def score_tile_agreement(
    reference: np.ndarray, prediction: np.ndarray, minimum_agreement: float
) -> TileAgreement:
    """Score complete interior objects after dense relabeling.

    Args:
        reference: Published labels in the scoring window.
        prediction: Recomputed labels in the same scoring window.
        minimum_agreement: Required AP at IoU 0.5 and mean matched IoU.

    Returns:
        Counts, overlap scores and the explicit pass/fail decision.

    Raises:
        ValueError: If inputs differ in shape or the window has no reference cells.
    """
    from cellpose import metrics

    if reference.shape != prediction.shape or reference.ndim != 2:
        raise ValueError("Reference and prediction must have the same 2D shape")
    if not 0 < minimum_agreement <= 1:
        raise ValueError("minimum_agreement must be in (0, 1]")
    reference = _exclude_border_labels(reference)
    prediction = _exclude_border_labels(prediction)
    reference_cells = int(reference.max())
    predicted_cells = int(prediction.max())
    if reference_cells == 0:
        raise ValueError("Scoring window contains no complete reference cells")
    ap, true_positive, false_positive, false_negative = metrics.average_precision(
        reference, prediction, threshold=[0.5]
    )
    if predicted_cells:
        ious, _ = metrics.mask_ious(reference, prediction)
        matched = ious[ious >= 0.5]
    else:
        matched = np.asarray([], dtype=float)
    mean_iou = float(matched.mean()) if matched.size else 0.0
    union = np.count_nonzero((reference > 0) | (prediction > 0))
    foreground_iou = float(np.count_nonzero((reference > 0) & (prediction > 0)) / union)
    ap_value = float(ap[0])
    return TileAgreement(
        reference_cells=reference_cells,
        predicted_cells=predicted_cells,
        true_positive=int(true_positive[0]),
        false_positive=int(false_positive[0]),
        false_negative=int(false_negative[0]),
        ap_at_05=ap_value,
        mean_matched_iou=mean_iou,
        foreground_iou=foreground_iou,
        minimum_agreement=minimum_agreement,
        passed=ap_value >= minimum_agreement and mean_iou >= minimum_agreement,
    )


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _require_fresh_output(output: Path, protected_root: Path) -> None:
    if output.resolve().is_relative_to(protected_root.resolve()):
        raise ValueError(
            "Acceptance output must be outside the protected baseline root"
        )
    if output.exists() and any(output.iterdir()):
        raise ValueError("Acceptance output directory must be new or empty")


def main(argv: list[str] | None = None) -> int:
    """Run the read-only tile drift check and write its report.

    Args:
        argv: Command arguments; defaults to the process command line.

    Returns:
        Zero on acceptance; one on drift or a changed baseline checksum.
        Report-only mode returns zero after writing any completed report.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "config",
        "source-zarr",
        "baseline-mask",
        "baseline-stats",
        "output-dir",
        "protected-root",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--image-key", required=True)
    parser.add_argument("--tile-index", type=int, required=True)
    parser.add_argument("--margin-px", type=int, default=656)
    parser.add_argument("--minimum-agreement", type=float, default=0.99)
    parser.add_argument("--min-free-gpu-gb", type=float, default=16.0)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)
    _require_fresh_output(args.output_dir, args.protected_root)
    for input_path in (
        args.config,
        args.source_zarr,
        args.baseline_mask,
        args.baseline_stats,
    ):
        if not input_path.exists():
            raise FileNotFoundError(input_path)
    if args.margin_px < 0:
        raise ValueError("margin-px must be non-negative")
    if not 0 < args.minimum_agreement <= 1:
        raise ValueError("minimum-agreement must be in (0, 1]")
    started = time.monotonic()
    config = SegmentationConfig.model_validate_json(args.config.read_text())
    config.dataset.data_path = args.source_zarr
    config.dataset.output_dir = args.output_dir
    config.cellpose.gpu = True
    stats = json.loads(args.baseline_stats.read_text())
    tile_size = int(stats["tile_size"])
    overlap = int(stats["stitch_overlap_px"])
    logger.info("Hashing the read-only published mask before inference")
    before = _sha256(args.baseline_mask)
    stats_sha = _sha256(args.baseline_stats)
    sdata, fetch, height, width, matrix, _points = _load_dataset_sdata(config)
    selected_key = config.dataset.image_key or (
        "morphology_focus" if is_experimental_arm(config) else list(sdata.images)[0]
    )
    if selected_key != args.image_key:
        raise ValueError("Production loader selected a different native image element")
    if args.image_key.endswith("_aligned_nonrigid"):
        raise ValueError("Acceptance must use the native image element")
    if (height, width) != (int(stats["height"]), int(stats["width"])):
        raise ValueError("Native image dimensions differ from the published tile grid")
    tiles = iter_core_tiles(height, width, tile_size, overlap)
    if not 0 <= args.tile_index < len(tiles):
        raise ValueError("tile-index is outside the published tile grid")
    tile = tiles[args.tile_index]
    score_bbox = (
        tile["core_y0"] + args.margin_px,
        tile["core_y1"] - args.margin_px,
        tile["core_x0"] + args.margin_px,
        tile["core_x1"] - args.margin_px,
    )
    y0, y1, x0, x1 = score_bbox
    if y0 >= y1 or x0 >= x1:
        raise ValueError("Scoring margin leaves no tile interior")
    preflight = (
        preflight_custom_model if has_custom_model(config) else preflight_stock_arm
    )
    expected_identity = (
        preflight(
            config,
            transform_matrix=matrix,
            image_key=args.image_key,
            allow_readonly_source=True,
        )
        if is_experimental_arm(config)
        else None
    )
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("The tile acceptance check requires CUDA")
    free, _total = torch.cuda.mem_get_info()
    if free / 1024**3 < args.min_free_gpu_gb:
        raise RuntimeError("Insufficient free GPU memory; retry in an idle window")
    model = build_cellpose_model(config.cellpose)
    identity = identify_cellpose_model(model, config.cellpose)
    if expected_identity is not None:
        identity.inference_context = expected_identity.inference_context
        if identity != expected_identity:
            raise ValueError("Loaded tile-check model differs from completed preflight")
    if model.device.type != "cuda":
        raise RuntimeError("Cellpose did not load onto CUDA")
    torch.cuda.reset_peak_memory_stats()
    logger.info("Running recorded production tile %d", args.tile_index)
    result = run_cellpose_tile(
        fetch,
        tile,
        model,
        config.cellpose,
        config.mask_filter,
        filter_per_tile=config.tiling.filter_per_tile,
        include_cellprobs=False,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "cellpose_masks_tiled.npy"
    np.save(prediction_path, result.filtered_masks.astype(np.uint32))
    x_affine, y_affine = build_cellpose_affine_to_microns(matrix, 1.0)
    pixel_area = abs(x_affine[0] * y_affine[1] - x_affine[1] * y_affine[0])
    area_stats = filter_labeled_mask_by_area(
        prediction_path,
        pixel_area_um2=pixel_area,
        min_area_um2=config.mask_filter.final_min_area_um2,
        max_area_um2=config.mask_filter.final_max_area_um2,
        show_progress=False,
    )
    prediction = np.load(prediction_path, mmap_mode="r")
    reference = np.load(args.baseline_mask, mmap_mode="r")
    local_y0, local_y1 = y0 - tile["tile_y0"], y1 - tile["tile_y0"]
    local_x0, local_x1 = x0 - tile["tile_x0"], x1 - tile["tile_x0"]
    agreement = score_tile_agreement(
        reference[y0:y1, x0:x1],
        prediction[local_y0:local_y1, local_x0:local_x1],
        args.minimum_agreement,
    )
    logger.info(
        "Tile agreement: AP@0.5=%.6f, mean matched IoU=%.6f, passed=%s",
        agreement.ap_at_05,
        agreement.mean_matched_iou,
        agreement.passed,
    )
    write_model_identity(prediction_path, identity)
    logger.info("Hashing the published mask again to check baseline preservation")
    after = _sha256(args.baseline_mask)
    baseline_unchanged = before == after and stats_sha == _sha256(args.baseline_stats)
    report = {
        "schema_version": 1,
        "dataset": config.dataset.name,
        "source_zarr": str(args.source_zarr),
        "image_key": args.image_key,
        "channels": config.dataset.channels,
        "tile_index": args.tile_index,
        "tile_size": tile_size,
        "stitch_overlap_px": overlap,
        "tile": tile,
        "scoring_bbox_y0_y1_x0_x1": score_bbox,
        "margin_px": args.margin_px,
        "border_rule": "Exclude labels touching the scoring-window border in each mask",
        "model_identity": identity.model_dump(mode="json"),
        "area_filter": area_stats,
        "agreement": asdict(agreement),
        "baseline_mask_sha256_before": before,
        "baseline_mask_sha256_after": after,
        "baseline_stats_sha256": stats_sha,
        "baseline_unchanged": baseline_unchanged,
        "peak_gpu_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "elapsed_seconds": time.monotonic() - started,
        "passed": agreement.passed and baseline_unchanged,
    }
    (args.output_dir / "tile_check.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    logger.info("Acceptance report: %s", args.output_dir / "tile_check.json")
    return 0 if report["passed"] or args.report_only else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())

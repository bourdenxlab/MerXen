"""Side-by-side comparison of several manual-registration runs of one pair.

Reads the summary JSON of each run (the ``RESULT_ATTR`` payload), tabulates the
fit, refinement, flow, tissue and landmark metrics, and draws an overview overlay
per run from the coarsest level of the result store's ds4 viewer caches.
"""

from __future__ import annotations

import csv
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from merxen.manual_registration.config import ManualRegistrationCompareConfig
from merxen.manual_registration.register import _plain_json, _publish
from merxen.manual_registration.schema import (
    RESULT_SCHEMA,
    STORE_ATTR,
    read_root_attrs,
    utc_now,
)
from merxen.plotting import prepare_plot_output, save_figure
from merxen.viewer_cache.format import derived_image_pyramid_cache_key

logger = logging.getLogger(__name__)

COMPARISON_SCHEMA = "merxen.manual_registration.comparison"
COMPARISON_COLUMNS: tuple[str, ...] = (
    "run_id",
    "preset",
    "run_label",
    "fit_n_enabled",
    "fit_n_used",
    "fit_rms_um",
    "fit_scale_ratio",
    "fit_rotation_deg",
    "refinement_status",
    "refinement_max_shift_um",
    "refinement_rotation_deg",
    "refinement_scale_change",
    "nonrigid_status",
    "nonrigid_confident_fraction",
    "nonrigid_pearson_before",
    "nonrigid_pearson_after",
    "jacobian_min",
    "jacobian_max",
    "tissue_pearson_coarse",
    "tissue_pearson_landmark",
    "tissue_pearson_affine",
    "tissue_pearson_final",
    "used_median_landmark_um",
    "used_median_affine_um",
    "used_median_final_um",
    "held_out_median_landmark_um",
    "held_out_median_affine_um",
    "held_out_median_final_um",
    "nucleus_qc_median_um",
    "nucleus_qc_pearson",
    "wall_time_s",
    "peak_rss_gb",
)
_PANELS_PER_ROW = 4


def _get(summary: Mapping[str, Any], *path: str) -> Any:
    value: Any = summary
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def comparison_row(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Pick the comparison metrics out of one run's summary (missing: ``None``)."""
    row: dict[str, Any] = {
        "run_id": summary.get("run_id"),
        "preset": summary.get("preset"),
        "run_label": summary.get("run_label"),
        "fit_n_enabled": _get(summary, "fit", "n_enabled"),
        "fit_n_used": _get(summary, "fit", "n_used"),
        "fit_rms_um": _get(summary, "fit", "rms_um"),
        "fit_scale_ratio": _get(summary, "fit", "scale_ratio"),
        "fit_rotation_deg": _get(summary, "fit", "rotation_deg"),
        "refinement_status": _get(summary, "refinement", "status"),
        "refinement_max_shift_um": _get(summary, "refinement", "max_shift_um"),
        "refinement_rotation_deg": _get(summary, "refinement", "rotation_deg"),
        "refinement_scale_change": _get(summary, "refinement", "scale_change"),
        "nonrigid_status": _get(summary, "nonrigid", "status"),
        "nonrigid_confident_fraction": _get(summary, "nonrigid", "confident_fraction"),
        "nonrigid_pearson_before": _get(summary, "nonrigid", "pearson_before"),
        "nonrigid_pearson_after": _get(summary, "nonrigid", "pearson_after"),
        "jacobian_min": _get(summary, "nonrigid", "final_flow", "min_jacobian"),
        "jacobian_max": _get(summary, "nonrigid", "final_flow", "max_jacobian"),
        "nucleus_qc_median_um": _get(summary, "nucleus_qc", "median_um"),
        "nucleus_qc_pearson": _get(
            summary, "nucleus_qc", "pearson_log_registration_channel"
        ),
        "wall_time_s": _get(summary, "resources", "wall_time_s"),
        "peak_rss_gb": _get(summary, "resources", "peak_rss_gb"),
    }
    for stage in ("coarse", "landmark", "affine", "final"):
        row[f"tissue_pearson_{stage}"] = _get(summary, "tissue_qc", "pearson", stage)
    for stage in ("landmark", "affine", "final"):
        row[f"used_median_{stage}_um"] = _get(
            summary, "residual_summary_um", "used", stage, "median"
        )
        row[f"held_out_median_{stage}_um"] = _get(
            summary, "residual_summary_um", "held_out", stage, "median"
        )
    return {column: row.get(column) for column in COMPARISON_COLUMNS}


def _load_summaries(config: ManualRegistrationCompareConfig) -> list[dict[str, Any]]:
    summaries = []
    for path in config.summary_paths:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"run summary not found: {path}")
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("schema") != RESULT_SCHEMA:
            raise ValueError(f"{path} is not a manual-registration result summary")
        if summary.get("status") != "complete":
            raise ValueError(
                f"{path}: the run is {summary.get('status')!r}, not complete"
            )
        if summary.get("dataset_name") != config.dataset_name:
            raise ValueError(
                f"{path} belongs to dataset {summary.get('dataset_name')!r}, not "
                f"{config.dataset_name!r}"
            )
        summary["_summary_dir"] = str(path.parent)
        summaries.append(summary)
    run_ids = [summary["run_id"] for summary in summaries]
    duplicates = sorted({run_id for run_id in run_ids if run_ids.count(run_id) > 1})
    if duplicates:
        raise ValueError(f"run ids must be unique, got duplicates {duplicates}")
    return sorted(summaries, key=lambda summary: summary["run_id"])


def _overview(summary: Mapping[str, Any]) -> np.ndarray | None:
    """Reference green / registered-moving magenta overlay from the ds4 caches."""
    import spatialdata as sd

    store_text = _get(summary, "outputs", "result_store")
    store = (
        Path(store_text)
        if store_text
        else Path(summary["_summary_dir"]) / f"{summary['run_id']}_registered.zarr"
    )
    if not store.exists():
        return None
    try:
        attrs = read_root_attrs(store)[STORE_ATTR]
        sdata = sd.read_zarr(store, selection=("images",))
        planes = []
        for role, key in (("reference", "reference"), ("moving", "moving_registered")):
            record = attrs["images"][role]
            tree = sdata.images[derived_image_pyramid_cache_key(key, 4)]
            image = _coarsest_level(tree)
            names = [str(name) for name in image.coords["c"].values]
            plane = np.asarray(
                image.isel(c=names.index(record["registration_channel"])).values,
                dtype=np.float32,
            )
            planes.append(plane)
    except (
        KeyError,
        ValueError,
        OSError,
        IndexError,
        TypeError,
        AttributeError,
    ) as exc:
        logger.warning("%s: no overview (%s)", summary.get("run_id"), exc)
        return None
    reference, moving = (_normalise(plane) for plane in planes)
    height = min(reference.shape[0], moving.shape[0])
    width = min(reference.shape[1], moving.shape[1])
    reference, moving = reference[:height, :width], moving[:height, :width]
    return np.dstack([moving, reference, moving])


def _coarsest_level(element: Any) -> Any:
    """The coarsest ``(c, y, x)`` DataArray of a (multi-scale) image element."""
    if not hasattr(element, "keys"):
        return element
    node = element[list(element.keys())[-1]]
    return node.to_dataset()["image"] if hasattr(node, "to_dataset") else node


def _normalise(plane: np.ndarray) -> np.ndarray:
    logged = np.log1p(np.maximum(plane, 0.0))
    low, high = np.percentile(logged, [1.0, 99.5])
    if high <= low:
        return np.zeros(plane.shape, dtype=np.float32)
    return np.asarray(
        np.clip((logged - low) / (high - low), 0.0, 1.0), dtype=np.float32
    )


def _fmt(value: Any, spec: str = ".2f") -> str:
    return "n/a" if value is None else format(value, spec)


def _write_figure(
    summaries: list[dict[str, Any]], rows: list[dict[str, Any]], path: Path
) -> Path:
    path = prepare_plot_output(path)
    n_cols = min(_PANELS_PER_ROW, len(summaries))
    n_rows = -(-len(summaries) // n_cols)
    figure, axes = plt.subplots(
        n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows), squeeze=False
    )
    for axis in axes.ravel():
        axis.set_axis_off()
    for axis, summary, row in zip(axes.ravel(), summaries, rows, strict=False):
        overlay = _overview(summary)
        if overlay is None:
            axis.add_patch(plt.Rectangle((0, 0), 1, 1, color="0.8"))
            axis.set_xlim(0, 1)
            axis.set_ylim(0, 1)
            axis.text(0.5, 0.5, "result store not found", ha="center", va="center")
        else:
            axis.imshow(overlay, interpolation="nearest")
        axis.set_title(
            f"{row['run_id']}\nrefine {row['refinement_status']} · "
            f"flow {row['nonrigid_status']} · "
            f"tissue r {_fmt(row['tissue_pearson_final'])}",
            fontsize=9,
        )
    figure.tight_layout()
    save_figure(figure, path, dpi=110)
    plt.close(figure)
    return path


def compare_runs(config: ManualRegistrationCompareConfig) -> dict[str, Path]:
    """Tabulate and plot several runs of one pair.

    Args:
        config: The summaries to compare and the output directories.

    Returns:
        ``comparison_json``, ``comparison_csv`` and ``comparison_png`` (durable
        paths in ``config.output_dir``).

    Raises:
        FileNotFoundError: If a summary is missing.
        ValueError: If a summary is not a complete result of this dataset, or run
            ids repeat.
    """
    summaries = _load_summaries(config)
    rows = [comparison_row(summary) for summary in summaries]
    task_dir = Path(config.task_output_dir)
    task_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(config.output_dir)
    prefix = config.dataset_name

    json_path = task_dir / f"{prefix}_run_comparison.json"
    payload = {
        "schema": COMPARISON_SCHEMA,
        "schema_version": 1,
        "dataset_name": config.dataset_name,
        "created_at": utc_now(),
        "runs": [
            {**row, "warnings": summary.get("warnings", [])}
            for row, summary in zip(rows, summaries, strict=True)
        ],
    }
    json_path.write_text(
        json.dumps(_plain_json(payload), indent=2) + "\n", encoding="utf-8"
    )
    csv_path = task_dir / f"{prefix}_run_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(COMPARISON_COLUMNS)
        for row in rows:
            writer.writerow(
                ["" if row[c] is None else row[c] for c in COMPARISON_COLUMNS]
            )
    png_path = _write_figure(summaries, rows, task_dir / f"{prefix}_run_comparison.png")

    published = {}
    for name, path in (
        ("comparison_json", json_path),
        ("comparison_csv", csv_path),
        ("comparison_png", png_path),
    ):
        published[name] = _publish(path, output_dir / path.name)
    _publish(
        png_path.with_suffix(".pdf"), output_dir / png_path.with_suffix(".pdf").name
    )
    return published

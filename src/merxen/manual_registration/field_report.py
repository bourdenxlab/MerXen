"""Figure of the landmark check: where the clicks sit, how sure the field is, residuals.

The figure of a register run (the non-affine displacement panel) is added with the
register path of the robust field.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from merxen.manual_registration.field_fit import FieldFitReport, SuggestionRecord
from merxen.manual_registration.image_evidence import RoleImages
from merxen.manual_registration.landmark_field import evaluate_on_grid
from merxen.manual_registration.schema import CropRect, StoreAttrs
from merxen.plotting import prepare_plot_output, save_figure

SD_GRID_STEP_UM = 200.0
VECTOR_SCALE = 5.0
_CLASS_COLOURS: dict[str, str] = {
    "ok": "#2ca02c",
    "kept_confirmed": "#17becf",
    "check": "#ff7f0e",
    "likely_misclick": "#d62728",
    "excluded": "#7f7f7f",
    "disabled": "#bbbbbb",
}
_CLASS_LABELS: dict[str, str] = {
    "ok": "ok",
    "kept_confirmed": "kept: image confirms",
    "check": "check",
    "likely_misclick": "likely misclick",
    "excluded": "excluded",
    "disabled": "off",
}
_MAX_TABLE_ROWS = 28


def _thumbnail(
    overview: RoleImages | None,
) -> tuple[np.ndarray, tuple[float, ...]] | None:
    """The first role image of the overview, log-scaled, with its um extent."""
    if overview is None:
        return None
    pixels = np.asarray(next(iter(overview.images.values())), dtype=np.float32)
    logged = np.log1p(np.clip(pixels, 0.0, None))
    low, high = np.percentile(logged, [1.0, 99.5])
    scaled = np.clip((logged - low) / max(high - low, 1e-6), 0.0, 1.0)
    height, width = pixels.shape
    corners = np.array([[-0.5, -0.5], [width - 0.5, height - 0.5]])
    level0 = corners @ overview.to_level0[:2, :2].T + overview.to_level0[:2, 2]
    extent_um = level0 * overview.level0_pixel_size_um
    return scaled, (extent_um[0, 0], extent_um[1, 0], extent_um[1, 1], extent_um[0, 1])


def _draw_clicks(axis: Axes, report: FieldFitReport, store: StoreAttrs) -> None:
    pixel = store.images["reference"].pixel_size_um
    pairs = report.similarity.pairs
    positions = np.array([p.reference_xy_px for p in pairs], dtype=float) * pixel
    jacobian = report.field.jacobian(positions)
    vectors = np.linalg.solve(jacobian, report.loo_vector_moving_um[:, :, None])[
        :, :, 0
    ]
    for name, colour in _CLASS_COLOURS.items():
        chosen = np.array([c == name for c in report.pair_class])
        if not chosen.any():
            continue
        axis.scatter(
            positions[chosen, 0],
            positions[chosen, 1],
            s=22,
            color=colour,
            edgecolors="black",
            linewidths=0.3,
            label=f"{_CLASS_LABELS[name]} ({int(chosen.sum())})",
            zorder=3,
        )
        if name != "disabled":
            axis.quiver(
                positions[chosen, 0],
                positions[chosen, 1],
                VECTOR_SCALE * vectors[chosen, 0],
                VECTOR_SCALE * vectors[chosen, 1],
                color=colour,
                angles="xy",
                scale_units="xy",
                scale=1.0,
                width=0.002,
                zorder=2,
            )
    for pair, point in zip(pairs, positions, strict=True):
        axis.annotate(
            str(pair.id), point, fontsize=5, xytext=(2, 2), textcoords="offset points"
        )
    axis.legend(fontsize=6, loc="lower left")


def write_field_figure(
    path: Path,
    *,
    report: FieldFitReport,
    store: StoreAttrs,
    crops: Mapping[str, CropRect],
    suggestions: Sequence[SuggestionRecord],
    kind: Literal["check"] = "check",
    overview: RoleImages | None = None,
) -> Path:
    """Write the landmark-check figure (PNG, plus a PDF twin) and return the PNG path.

    Panels: the reference tissue with every click coloured by class and its
    leave-one-out vector (x5; disabled pairs grey); the posterior SD per axis on a
    200 um grid over the reference crop, with the accepted windows as dots and the
    numbered suggestions; histograms of the robust (final and clicks-only) and rigid
    leave-one-out residuals; and a table of the pairs that are not ``ok``.

    Args:
        path: Output PNG path.
        report: The field fit.
        store: Attrs of the prepared store.
        crops: Effective crops by ``reference`` and ``moving``.
        suggestions: The suggested next clicks.
        kind: ``check`` (the only kind so far).
        overview: Coarse reference role images for the tissue thumbnail, or None.

    Returns:
        The PNG path.

    Raises:
        ValueError: For a kind other than ``check``.
    """
    if kind != "check":
        raise ValueError(f"unknown figure kind {kind!r}")
    path = prepare_plot_output(path)
    pixel = store.images["reference"].pixel_size_um
    crop = crops["reference"]
    x0, x1 = crop.x0 * pixel, crop.x1 * pixel
    y0, y1 = crop.y0 * pixel, crop.y1 * pixel
    figure, axes = plt.subplots(2, 2, figsize=(16, 14))
    tissue, thumb_axis, sd_axis = axes[0, 0], axes[0, 0], axes[0, 1]
    histogram_axis, table_axis = axes[1, 0], axes[1, 1]

    thumb = _thumbnail(overview)
    if thumb is not None:
        image, extent = thumb
        tissue.imshow(image, extent=extent, cmap="gray", interpolation="nearest")
    _draw_clicks(thumb_axis, report, store)
    tissue.set_xlim(x0, x1)
    tissue.set_ylim(y1, y0)
    tissue.set_aspect("equal")
    tissue.set_title(f"Clicks by class, leave-one-out vectors x{VECTOR_SCALE:g} (um)")

    step = SD_GRID_STEP_UM
    shape = (
        max(int(np.ceil((y1 - y0) / step)), 1),
        max(int(np.ceil((x1 - x0) / step)), 1),
    )
    grid = evaluate_on_grid(
        report.field, origin_um=(x0, y0), step_um=step, shape_yx=shape
    )
    assert grid.sd_um is not None
    shown = sd_axis.imshow(
        grid.sd_um,
        extent=(
            x0 - step / 2,
            x0 + step * (shape[1] - 0.5),
            y0 + step * (shape[0] - 0.5),
            y0 - step / 2,
        ),
        cmap="viridis",
        interpolation="nearest",
    )
    figure.colorbar(shown, ax=sd_axis, shrink=0.7, label="posterior SD per axis (um)")
    if report.windows is not None and len(report.windows.reference_xy_um):
        sd_axis.scatter(
            report.windows.reference_xy_um[:, 0],
            report.windows.reference_xy_um[:, 1],
            s=4,
            color="white",
            label=f"windows ({report.n_windows})",
        )
    for record in suggestions:
        point = record.suggestion.reference_xy_um
        sd_axis.scatter(point[0], point[1], s=40, facecolors="none", edgecolors="red")
        sd_axis.annotate(
            str(record.suggestion.rank),
            point,
            color="red",
            fontsize=7,
            xytext=(3, 3),
            textcoords="offset points",
        )
    sd_axis.set_xlim(x0, x1)
    sd_axis.set_ylim(y1, y0)
    sd_axis.set_aspect("equal")
    sd_axis.set_title("Field uncertainty, windows and suggested clicks")
    if report.windows is not None:
        sd_axis.legend(fontsize=7, loc="lower left")

    fit_index = np.asarray(report.similarity.fit_index, dtype=int)
    series: list[tuple[str, np.ndarray]] = [
        ("robust field", report.loo_reference_um[fit_index]),
    ]
    if report.has_second_pass:
        series.append(("clicks only", report.clicks_only_loo_reference_um[fit_index]))
    series.append(
        ("rigid", np.asarray(report.similarity.loo_um[fit_index], dtype=float))
    )
    finite = [v[np.isfinite(v)] for _, v in series]
    upper = max((float(v.max()) for v in finite if len(v)), default=1.0)
    bins = np.linspace(0.0, max(upper, 1.0), 30)
    for (label, _), values in zip(series, finite, strict=True):
        histogram_axis.hist(
            np.clip(values, 0.0, bins[-1]),
            bins=bins,
            histtype="step",
            linewidth=1.5,
            label=f"{label} (median {np.median(values):.0f})" if len(values) else label,
        )
    histogram_axis.set_xlabel("leave-one-out residual (reference um)")
    histogram_axis.set_ylabel("clicks")
    histogram_axis.legend(fontsize=8)
    histogram_axis.set_title("Leave-one-out residuals")

    table_axis.set_axis_off()
    p_star = np.minimum(report.p_value, report.clicks_only_p_value)
    w_star = np.fmin(report.weight, report.clicks_only_weight)
    rows = []
    for i, pair in enumerate(report.similarity.pairs):
        if report.pair_class[i] == "ok":
            continue
        weight = "" if not np.isfinite(w_star[i]) else f"{w_star[i]:.2f}"
        rows.append(
            [
                str(pair.id),
                _CLASS_LABELS[report.pair_class[i]],
                report.verdict[i],
                f"{report.loo_reference_um[i]:.0f}",
                f"{p_star[i]:.1e}",
                weight,
            ]
        )
    if rows:
        shown_rows = rows[:_MAX_TABLE_ROWS]
        table = table_axis.table(
            cellText=shown_rows,
            colLabels=["id", "class", "verdict", "LOO um", "p*", "w*"],
            loc="upper center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.0, 1.2)
        extra = len(rows) - len(shown_rows)
        title = "Pairs that are not ok" + (
            f" ({extra} more in the JSON)" if extra else ""
        )
    else:
        title = "Every pair is ok"
    table_axis.set_title(title)
    figure.suptitle(f"{store.dataset_name}: landmark check")
    figure.tight_layout()
    save_figure(figure, path, dpi=110)
    plt.close(figure)
    return path

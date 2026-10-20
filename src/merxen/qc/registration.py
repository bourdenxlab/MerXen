"""Check that a segmentation is registered to its transcripts.

A segmentation whose masks were converted to microns with the wrong
pixel-to-micron transform still looks plausible on its own: cell sizes,
counts and shapes are unchanged, only every cell sits at a fixed offset from
the tissue it was drawn on. Two label-free signals expose this:

- **Transcript density at centroids.** Cell centroids of a registered
  segmentation sit on transcript-dense cell bodies, so the number of
  transcripts within a few microns of a centroid is well above that of random
  points in the same tissue windows. A misregistered segmentation drops to
  roughly the random level.
- **Offset against the platform's own segmentation.** Cross-correlating the
  centroid density maps of the segmentation and of the platform's cells
  (Vizgen or Xenium Onboard Analysis) peaks at the misregistration vector.

Coordinates are compared in each element's intrinsic frame. MerXen keeps
transcript points and cell shapes in microns, so no SpatialData coordinate
system is applied (the platform shapes may carry a micron-to-mosaic affine
that the points do not).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd
import shapely
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree

from merxen.io.transcript_io import first_existing_col

logger = logging.getLogger(__name__)

# Blank and negative-control probes are not tied to cell bodies.
CONTROL_FEATURE_PATTERN = r"^(Blank|NegControl|Unassigned|Deprecated|Intergenic)"

Window = tuple[float, float, float, float]


class RegistrationStatus(StrEnum):
    """Outcome of a segmentation registration check."""

    PASS = "pass"
    WARN = "warn"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class RegistrationCheckParams:
    """Thresholds and sampling settings for the registration check.

    Attributes:
        radius_um: Radius for counting transcripts around a point.
        min_density_ratio: Warn when centroid density divided by random-point
            density falls below this.
        max_shift_um: Warn when the offset against the platform segmentation
            is larger than this.
        n_windows: Number of square tissue windows sampled around random cells.
        window_size_um: Side length of each window.
        n_random_per_window: Random reference points drawn per window.
        bin_um: Histogram bin size for the cross-correlation.
        max_search_um: Largest offset searched by the cross-correlation.
        min_peak_z: Minimum z-score of a cross-correlation peak for its window
            to contribute to the offset estimate.
        min_centroids: Minimum number of centroids inside the windows.
        exclude_control_features: Drop blank / negative-control probes before
            counting. Off by default: they are under 1% of transcripts and
            spread evenly, so they do not move the ratio (VZG2: identical to
            two decimals), while reading every feature name slows the points
            pass by about 1.7x.
        seed: Seed for window and random-point sampling.
    """

    radius_um: float = 4.0
    min_density_ratio: float = 1.5
    max_shift_um: float = 5.0
    n_windows: int = 8
    window_size_um: float = 400.0
    n_random_per_window: int = 1000
    bin_um: float = 2.0
    max_search_um: float = 150.0
    min_peak_z: float = 4.0
    min_centroids: int = 50
    exclude_control_features: bool = False
    seed: int = 0


@dataclass
class RegistrationCheckResult:
    """Result of :func:`segmentation_registration_check`.

    Attributes:
        status: ``pass``, ``warn`` or ``skipped``.
        reasons: Human-readable reasons behind a ``warn`` or ``skipped`` status.
        n_windows: Number of windows evaluated.
        n_centroids: Centroids inside the window interiors.
        n_random_points: Random reference points evaluated.
        mean_transcripts_at_centroids: Mean transcripts within ``radius_um``.
        mean_transcripts_at_random: Same, for random points.
        density_ratio: Ratio of the two means.
        shift_x_um: Median offset (reference minus segmentation), x.
        shift_y_um: Median offset (reference minus segmentation), y.
        shift_um: Length of the median offset.
        n_shift_windows: Windows with a usable cross-correlation peak.
        params: Settings used for the check.
        per_window: Per-window diagnostics.
    """

    status: RegistrationStatus
    reasons: list[str]
    n_windows: int
    n_centroids: int
    n_random_points: int
    mean_transcripts_at_centroids: float | None
    mean_transcripts_at_random: float | None
    density_ratio: float | None
    shift_x_um: float | None
    shift_y_um: float | None
    shift_um: float | None
    n_shift_windows: int
    params: RegistrationCheckParams
    per_window: list[dict[str, float | None]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        value = asdict(self)
        value["status"] = str(self.status)
        return value


def sample_registration_windows(
    centroids: np.ndarray,
    params: RegistrationCheckParams | None = None,
) -> list[Window]:
    """Place square windows around randomly chosen cells.

    Windows are centred on cells so that they fall on tissue, where the random
    reference points measure the local tissue density.

    Args:
        centroids: ``(n, 2)`` cell centroids in microns.
        params: Check settings; defaults are used when omitted.

    Returns:
        Windows as ``(x_min, x_max, y_min, y_max)`` tuples.
    """
    params = params or RegistrationCheckParams()
    points = _finite_xy(centroids)
    if len(points) == 0:
        return []
    rng = np.random.default_rng(params.seed)
    n_windows = min(params.n_windows, len(points))
    chosen = points[rng.choice(len(points), size=n_windows, replace=False)]
    half = params.window_size_um / 2.0
    return [(x - half, x + half, y - half, y + half) for x, y in chosen]


def transcripts_in_windows(
    points_obj: Any,
    windows: list[Window],
    *,
    x_col: str,
    y_col: str,
    gene_col: str | None = None,
    excluded_gene_pattern: str | None = CONTROL_FEATURE_PATTERN,
) -> np.ndarray:
    """Collect transcript coordinates that fall inside any window.

    Dask-backed points are filtered partition by partition, so only the
    transcripts inside the windows are materialised.

    Args:
        points_obj: SpatialData points (Dask or pandas DataFrame).
        windows: Windows from :func:`sample_registration_windows`.
        x_col: Column holding x in microns.
        y_col: Column holding y in microns.
        gene_col: Optional feature column used to drop control probes.
        excluded_gene_pattern: Regular expression of feature names to drop.

    Returns:
        ``(m, 2)`` float64 array of transcript coordinates.
    """
    if not windows:
        return np.empty((0, 2), dtype=np.float64)
    columns = [x_col, y_col]
    if gene_col is not None and excluded_gene_pattern is not None:
        columns.append(gene_col)
    else:
        excluded_gene_pattern = None
    bounds = np.asarray(windows, dtype=np.float64)

    def _filter(frame: pd.DataFrame) -> pd.DataFrame:
        x = pd.to_numeric(frame[x_col], errors="coerce").to_numpy(np.float64)
        y = pd.to_numeric(frame[y_col], errors="coerce").to_numpy(np.float64)
        keep = np.zeros(len(frame), dtype=bool)
        for x_min, x_max, y_min, y_max in bounds:
            keep |= (x >= x_min) & (x < x_max) & (y >= y_min) & (y < y_max)
        if excluded_gene_pattern is not None and gene_col is not None:
            # Match names only for the few rows inside the windows; converting
            # every feature name of a 200M-transcript partition is costly.
            rows = np.flatnonzero(keep)
            genes = frame[gene_col].iloc[rows].astype(str)
            is_control = genes.str.match(excluded_gene_pattern, na=False)
            keep[rows[is_control.to_numpy(dtype=bool)]] = False
        return pd.DataFrame({"x": x[keep], "y": y[keep]})

    subset = points_obj[columns]
    if hasattr(subset, "map_partitions"):
        meta = pd.DataFrame(
            {"x": pd.Series(dtype="float64"), "y": pd.Series(dtype="float64")}
        )
        frame: pd.DataFrame = subset.map_partitions(_filter, meta=meta).compute()
    else:
        frame = _filter(pd.DataFrame(subset))
    return np.asarray(frame[["x", "y"]].to_numpy(dtype=np.float64))


def segmentation_registration_check(
    centroids: np.ndarray,
    transcripts: np.ndarray,
    windows: list[Window],
    *,
    reference_centroids: np.ndarray | None = None,
    params: RegistrationCheckParams | None = None,
) -> RegistrationCheckResult:
    """Check that cell centroids sit on transcripts and match the platform cells.

    Args:
        centroids: ``(n, 2)`` centroids of the segmentation under test, microns.
        transcripts: ``(m, 2)`` transcript coordinates inside ``windows``.
        windows: Tissue windows from :func:`sample_registration_windows`.
        reference_centroids: Optional ``(k, 2)`` centroids of the platform's own
            segmentation. When given, the offset between the two centroid
            density maps is estimated by cross-correlation.
        params: Check settings; defaults are used when omitted.

    Returns:
        The check result. ``warn`` means the density ratio is below
        ``params.min_density_ratio`` or the offset exceeds
        ``params.max_shift_um``.
    """
    params = params or RegistrationCheckParams()
    rng = np.random.default_rng(params.seed + 1)
    cells = _finite_xy(centroids)
    tx = _finite_xy(transcripts)
    reference = (
        _finite_xy(reference_centroids) if reference_centroids is not None else None
    )
    # Only transcripts inside the windows are known, so points closer than
    # the counting radius to a window edge would be undercounted.
    inset = params.radius_um + 1.0

    tree = cKDTree(tx) if len(tx) else None
    cell_counts: list[np.ndarray] = []
    random_counts: list[np.ndarray] = []
    shifts: list[tuple[float, float]] = []
    per_window: list[dict[str, float | None]] = []
    for x_min, x_max, y_min, y_max in windows:
        inner = (x_min + inset, x_max - inset, y_min + inset, y_max - inset)
        inside = _in_box(cells, inner)
        random_points = np.column_stack(
            [
                rng.uniform(inner[0], inner[1], params.n_random_per_window),
                rng.uniform(inner[2], inner[3], params.n_random_per_window),
            ]
        )
        at_cells = _count_within(tree, cells[inside], params.radius_um)
        at_random = _count_within(tree, random_points, params.radius_um)
        cell_counts.append(at_cells)
        random_counts.append(at_random)
        row: dict[str, float | None] = {
            "x_min": float(x_min),
            "y_min": float(y_min),
            "n_centroids": float(len(at_cells)),
            "mean_transcripts_at_centroids": _mean_or_none(at_cells),
            "mean_transcripts_at_random": _mean_or_none(at_random),
            "shift_x_um": None,
            "shift_y_um": None,
            "shift_peak_z": None,
        }
        if reference is not None:
            shift = _cross_correlation_shift(
                cells[_in_box(cells, (x_min, x_max, y_min, y_max))],
                reference[_in_box(reference, (x_min, x_max, y_min, y_max))],
                (x_min, x_max, y_min, y_max),
                params,
            )
            if shift is not None:
                row["shift_x_um"], row["shift_y_um"], row["shift_peak_z"] = shift
                if shift[2] >= params.min_peak_z:
                    shifts.append((shift[0], shift[1]))
        per_window.append(row)

    all_cells = np.concatenate(cell_counts) if cell_counts else np.empty(0)
    all_random = np.concatenate(random_counts) if random_counts else np.empty(0)
    mean_cells = _mean_or_none(all_cells)
    mean_random = _mean_or_none(all_random)
    density_ratio: float | None = None
    if mean_cells is not None and mean_random is not None and mean_random > 0:
        density_ratio = mean_cells / mean_random

    shift_x = shift_y = shift_norm = None
    if shifts:
        shift_x, shift_y = (float(value) for value in np.median(shifts, axis=0))
        shift_norm = float(np.hypot(shift_x, shift_y))

    reasons: list[str] = []
    status = RegistrationStatus.PASS
    if not windows or len(tx) == 0:
        status = RegistrationStatus.SKIPPED
        reasons.append("no transcripts inside the sampled windows")
    elif len(all_cells) < params.min_centroids:
        status = RegistrationStatus.SKIPPED
        reasons.append(
            f"only {len(all_cells)} centroids inside the sampled windows "
            f"(need {params.min_centroids})"
        )
    elif density_ratio is None:
        status = RegistrationStatus.SKIPPED
        reasons.append("random points found no transcripts")
    else:
        if density_ratio < params.min_density_ratio:
            status = RegistrationStatus.WARN
            reasons.append(
                f"transcript density at centroids is {density_ratio:.2f}x that of "
                f"random points (< {params.min_density_ratio})"
            )
        if shift_norm is not None and shift_norm > params.max_shift_um:
            status = RegistrationStatus.WARN
            reasons.append(
                f"centroids are offset by ({shift_x:.1f}, {shift_y:.1f}) um from the "
                f"platform segmentation (> {params.max_shift_um} um)"
            )
    if reference is not None and not shifts and status != RegistrationStatus.SKIPPED:
        reasons.append(
            "cross-correlation against the platform segmentation was inconclusive"
        )

    return RegistrationCheckResult(
        status=status,
        reasons=reasons,
        n_windows=len(windows),
        n_centroids=int(len(all_cells)),
        n_random_points=int(len(all_random)),
        mean_transcripts_at_centroids=mean_cells,
        mean_transcripts_at_random=mean_random,
        density_ratio=density_ratio,
        shift_x_um=shift_x,
        shift_y_um=shift_y,
        shift_um=shift_norm,
        n_shift_windows=len(shifts),
        params=params,
        per_window=per_window,
    )


def shape_centroids(shapes: Any) -> np.ndarray:
    """Return ``(n, 2)`` centroids of non-empty geometries in intrinsic units.

    Args:
        shapes: GeoDataFrame or GeoSeries of cell polygons.

    Returns:
        Centroid coordinates as float64.
    """
    geometry = shapes.geometry if hasattr(shapes, "geometry") else shapes
    values = np.asarray(geometry.values if hasattr(geometry, "values") else geometry)
    keep = ~(shapely.is_missing(values) | shapely.is_empty(values))
    return np.asarray(
        shapely.get_coordinates(shapely.centroid(values[keep])), dtype=np.float64
    )


def compute_segmentation_registration_qc(
    sdata_obj: Any,
    *,
    shape_key: str,
    points_key: str,
    reference_shape_key: str | None = None,
    params: RegistrationCheckParams | None = None,
) -> RegistrationCheckResult:
    """Run :func:`segmentation_registration_check` on one SpatialData layer.

    Args:
        sdata_obj: SpatialData object holding the shapes and points.
        shape_key: Shapes of the segmentation under test.
        points_key: Transcript points element.
        reference_shape_key: Shapes of the platform's own segmentation. It is
            ignored when missing or equal to ``shape_key``.
        params: Check settings; defaults are used when omitted.

    Returns:
        The check result.
    """
    params = params or RegistrationCheckParams()
    centroids = shape_centroids(sdata_obj.shapes[shape_key])
    reference = None
    if reference_shape_key is not None and reference_shape_key != shape_key:
        if reference_shape_key in sdata_obj.shapes:
            reference = shape_centroids(sdata_obj.shapes[reference_shape_key])
        else:
            logger.info(
                "Registration check: reference shapes %r not found; skipping the "
                "cross-correlation offset.",
                reference_shape_key,
            )
    points_obj = sdata_obj.points[points_key]
    x_col = first_existing_col(points_obj, ["x", "global_x", "x_location"])
    y_col = first_existing_col(points_obj, ["y", "global_y", "y_location"])
    gene_col = (
        first_existing_col(points_obj, ["feature_name", "gene", "target"])
        if params.exclude_control_features
        else None
    )
    windows = sample_registration_windows(centroids, params)
    if x_col is None or y_col is None:
        transcripts = np.empty((0, 2), dtype=np.float64)
    else:
        transcripts = transcripts_in_windows(
            points_obj,
            windows,
            x_col=x_col,
            y_col=y_col,
            gene_col=gene_col,
        )
    return segmentation_registration_check(
        centroids,
        transcripts,
        windows,
        reference_centroids=reference,
        params=params,
    )


def _finite_xy(values: np.ndarray | None) -> np.ndarray:
    if values is None:
        return np.empty((0, 2), dtype=np.float64)
    array = np.asarray(values, dtype=np.float64).reshape(-1, 2)
    return array[np.isfinite(array).all(axis=1)]


def _in_box(points: np.ndarray, box: Window) -> np.ndarray:
    x_min, x_max, y_min, y_max = box
    return (
        (points[:, 0] >= x_min)
        & (points[:, 0] < x_max)
        & (points[:, 1] >= y_min)
        & (points[:, 1] < y_max)
    )


def _count_within(
    tree: cKDTree | None, points: np.ndarray, radius: float
) -> np.ndarray:
    if tree is None or len(points) == 0:
        return np.zeros(len(points), dtype=np.float64)
    return np.asarray(
        tree.query_ball_point(points, r=radius, return_length=True), dtype=np.float64
    )


def _mean_or_none(values: np.ndarray) -> float | None:
    return float(np.mean(values)) if len(values) else None


def _cross_correlation_shift(
    cells: np.ndarray,
    reference: np.ndarray,
    window: Window,
    params: RegistrationCheckParams,
) -> tuple[float, float, float] | None:
    """Return (dx, dy, peak z) aligning ``cells`` onto ``reference`` in a window."""
    if len(cells) < 10 or len(reference) < 10:
        return None
    x_min, x_max, y_min, y_max = window
    n_bins = max(2, int(round((x_max - x_min) / params.bin_um)))
    hist_range = [[x_min, x_max], [y_min, y_max]]
    reference_map, _, _ = np.histogram2d(
        reference[:, 0], reference[:, 1], bins=n_bins, range=hist_range
    )
    cell_map, _, _ = np.histogram2d(
        cells[:, 0], cells[:, 1], bins=n_bins, range=hist_range
    )
    reference_map = gaussian_filter(reference_map, 1.0)
    cell_map = gaussian_filter(cell_map, 1.0)
    reference_map -= reference_map.mean()
    cell_map -= cell_map.mean()
    # Full (zero-padded) correlation so large offsets do not wrap around.
    correlation = fftconvolve(reference_map, cell_map[::-1, ::-1], mode="full")
    centre = n_bins - 1
    radius = min(centre, int(round(params.max_search_um / params.bin_um)))
    search = correlation[
        centre - radius : centre + radius + 1, centre - radius : centre + radius + 1
    ]
    spread = float(search.std())
    if spread <= 0:
        return None
    peak_x, peak_y = np.unravel_index(int(np.argmax(search)), search.shape)
    peak_z = float((search[peak_x, peak_y] - search.mean()) / spread)
    dx = float((peak_x - radius) * params.bin_um)
    dy = float((peak_y - radius) * params.bin_um)
    return dx, dy, peak_z

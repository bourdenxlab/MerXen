"""Synthetic image pairs and landmark simulation for manual registration.

Used by the tests, the Nextflow end-to-end test and the validation scripts. A
synthetic pair is two OME-TIFF pyramids rendered from one shared tissue field
through a known similarity plus a smooth warp, so the true moving-to-reference
mapping is known everywhere. ``inject_labelling`` writes landmark pairs into a
prepared store with the viewer's protocol (lock, ``revision``, validated
payload).

Frames: tissue coordinates ``u`` are reference micrometres (``u = p_r * r`` for
reference pixels ``r``); points are ``(x, y)`` in level-0 pixels of the full image
they belong to; matrices map moving to reference.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import (
    distance_transform_edt,
    gaussian_filter,
    map_coordinates,
)
from skimage.filters import threshold_otsu

import merxen
from merxen.io.ome_tiff_writer import write_pyramidal_ome_tiff
from merxen.io.spatialdata_io import spatialdata_write_lock
from merxen.manual_registration.geometry import (
    apply_matrix_xy,
    coarse_from_matrix,
    coarse_matrix_xy,
)
from merxen.manual_registration.landmark_fit import fit_similarity
from merxen.manual_registration.schema import (
    LABELLING_ATTR,
    LABELLING_SCHEMA,
    SCHEMA_VERSION,
    STORE_ATTR,
    CoarseTransform,
    CropRect,
    LandmarkPair,
    labelling_content_sha256,
    pair_colour,
    parse_labelling_attrs,
    parse_store_attrs,
    read_root_attrs,
    utc_now,
    write_root_attr,
)

logger = logging.getLogger(__name__)

_CANVAS_UM_PER_PX = 1.0
_CANVAS_MARGIN_UM = 200.0
_NUCLEUS_SIGMA_UM = 3.0
_NUCLEUS_SPACING_UM = 25.0
_VESSEL_WIDTH_UM = 25.0
_VESSEL_FACTOR = 0.35
_MAX_CANDIDATES = 20000


@dataclass(frozen=True)
class SyntheticPair:
    """A synthetic reference/moving image pair with known ground truth.

    Attributes:
        reference_path: Reference OME-TIFF.
        moving_path: Moving OME-TIFF.
        reference_pixel_size_um: Reference pixel size.
        moving_pixel_size_um: Moving pixel size.
        true_matrix: Similarity from moving pixels to reference pixels (without
            the smooth warp).
        moving_to_reference: The true map of ``(N, 2)`` moving points to reference
            points, including the smooth warp.
        reference_shape_yx: Reference ``(H, W)``.
        moving_shape_yx: Moving ``(H, W)``.
    """

    reference_path: Path
    moving_path: Path
    reference_pixel_size_um: float
    moving_pixel_size_um: float
    true_matrix: np.ndarray
    moving_to_reference: Callable[[np.ndarray], np.ndarray]
    reference_shape_yx: tuple[int, int]
    moving_shape_yx: tuple[int, int]


def _canvas_extent(
    shape_yx: tuple[int, int], pixel_size_um: float
) -> tuple[float, float]:
    return (shape_yx[1] * pixel_size_um, shape_yx[0] * pixel_size_um)


def _bezier(
    p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, count: int = 600
) -> np.ndarray:
    t = np.linspace(0.0, 1.0, count)[:, None]
    return np.asarray((1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2)


def _shared_background(
    extent_um: tuple[float, float], seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Render the tissue mask, layer pattern, vessel mask and background."""
    rng = np.random.default_rng(seed)
    c = _CANVAS_UM_PER_PX
    width = int(math.ceil((extent_um[0] + 2 * _CANVAS_MARGIN_UM) / c))
    height = int(math.ceil((extent_um[1] + 2 * _CANVAS_MARGIN_UM) / c))
    yy, xx = np.indices((height, width), dtype=np.float32)
    ux = xx * c - _CANVAS_MARGIN_UM
    uy = yy * c - _CANVAS_MARGIN_UM
    del xx, yy
    dx, dy = ux - extent_um[0] / 2.0, uy - extent_um[1] / 2.0
    radius = np.sqrt(
        (dx / (0.42 * extent_um[0])) ** 2 + (dy / (0.38 * extent_um[1])) ** 2
    )
    boundary = 1.0 + 0.05 * np.sin(5.0 * np.arctan2(dy, dx) + 0.7)
    tissue = gaussian_filter((radius <= boundary).astype(np.float32), 10.0 / c)
    del dx, dy, radius, boundary
    angle = math.radians(20.0)
    layer = (
        0.5
        + 0.5
        * np.sin(2.0 * math.pi * (ux * math.cos(angle) + uy * math.sin(angle)) / 400.0)
    ).astype(np.float32)
    del ux, uy

    vessels = np.zeros((height, width), dtype=bool)
    inside = np.argwhere(tissue > 0.5)
    for _ in range(6):
        picks = inside[rng.integers(0, len(inside), size=3)]
        control = picks[:, ::-1].astype(np.float64)  # (x, y) canvas px
        curve = _bezier(control[0], control[1], control[2])
        x0, x1 = int(curve[:, 0].min()) - 40, int(curve[:, 0].max()) + 41
        y0, y1 = int(curve[:, 1].min()) - 40, int(curve[:, 1].max()) + 41
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, width), min(y1, height)
        line = np.zeros((y1 - y0, x1 - x0), dtype=bool)
        xs = np.clip(np.round(curve[:, 0]).astype(int) - x0, 0, x1 - x0 - 1)
        ys = np.clip(np.round(curve[:, 1]).astype(int) - y0, 0, y1 - y0 - 1)
        line[ys, xs] = True
        vessels[y0:y1, x0:x1] |= distance_transform_edt(~line) <= (
            _VESSEL_WIDTH_UM / 2.0 / c
        )
    vessel_factor = np.where(vessels, _VESSEL_FACTOR, 1.0).astype(np.float32)
    background = (tissue * (60.0 + 40.0 * layer) * vessel_factor).astype(np.float32)
    return background, tissue, layer, vessels


def _nuclei(
    tissue: np.ndarray, layer: np.ndarray, vessels: np.ndarray, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    c = _CANVAS_UM_PER_PX
    height, width = tissue.shape
    count = int(rng.poisson(height * width * c * c / _NUCLEUS_SPACING_UM**2))
    xs = rng.integers(0, width, size=count)
    ys = rng.integers(0, height, size=count)
    accept = (
        (tissue[ys, xs] > 0.5)
        & ~vessels[ys, xs]
        & (rng.uniform(size=count) < 0.5 + 0.5 * layer[ys, xs])
    )
    amplitude = rng.uniform(800.0, 1600.0, size=count)
    seeds = np.zeros((height, width), dtype=np.float32)
    np.add.at(seeds, (ys[accept], xs[accept]), amplitude[accept].astype(np.float32))
    sigma = _NUCLEUS_SIGMA_UM / c
    return np.asarray(
        gaussian_filter(seeds, sigma) * (2.0 * math.pi * sigma**2), dtype=np.float32
    )


def _render(
    field: np.ndarray,
    to_tissue_um: Callable[[np.ndarray], np.ndarray],
    shape_yx: tuple[int, int],
    *,
    block_rows: int = 256,
) -> np.ndarray:
    """Sample the canvas at the tissue position (um) of every output pixel."""
    height, width = shape_yx
    columns = np.arange(width, dtype=np.float64)
    out = np.zeros((height, width), dtype=np.float32)
    for start in range(0, height, block_rows):
        stop = min(start + block_rows, height)
        jj, ii = np.meshgrid(columns, np.arange(start, stop, dtype=np.float64))
        u = to_tissue_um(np.column_stack([jj.ravel(), ii.ravel()]))
        canvas_x = (u[:, 0] + _CANVAS_MARGIN_UM) / _CANVAS_UM_PER_PX
        canvas_y = (u[:, 1] + _CANVAS_MARGIN_UM) / _CANVAS_UM_PER_PX
        out[start:stop] = map_coordinates(
            field, [canvas_y, canvas_x], order=1, mode="constant", cval=0.0
        ).reshape(stop - start, width)
    return out


def _channels(
    dapi: np.ndarray, n_channels: int, rng: np.random.Generator
) -> np.ndarray:
    planes = []
    for k in range(n_channels):
        if k == 0:
            plane = dapi
        elif k == 1:
            plane = 0.5 * dapi + 20.0
        else:
            plane = 0.25 * dapi + 10.0 * k
        plane = plane + rng.normal(0.0, 5.0, dapi.shape)
        planes.append(np.rint(np.clip(plane, 0.0, 65535.0)).astype(np.uint16))
    return np.stack(planes)


def _write_ome(
    array: np.ndarray,
    path: Path,
    *,
    channel_names: Sequence[str],
    pixel_size_um: float,
    image_name: str,
) -> None:
    write_pyramidal_ome_tiff(
        lambda c, y0, h: array[c, y0 : y0 + h],
        shape_cyx=(int(array.shape[0]), int(array.shape[1]), int(array.shape[2])),
        dtype="uint16",
        channel_names=list(channel_names),
        pixel_size_um=pixel_size_um,
        image_name=image_name,
        output_path=path,
        tile_size=256,
        band_height=512,
        min_level_size=256,
        workers=1,
    )


def make_synthetic_pair(
    out_dir: Path,
    *,
    reference_shape_yx: tuple[int, int] = (2048, 2048),
    reference_pixel_size_um: float = 2.0,
    moving_pixel_size_um: float = 1.6,
    rotation_deg: float = 178.0,
    mirror_x: bool = False,
    offset_um: tuple[float, float] = (0.0, 0.0),
    warp_amplitude_um: float = 3.0,
    warp_wavelength_um: float = 300.0,
    channel_names: Sequence[str] = ("DAPI", "AF568"),
    seed: int = 0,
    independent_nuclei: bool = False,
) -> SyntheticPair:
    """Render a synthetic reference/moving pair with a known ground truth.

    Both images show the same low-frequency structure (tissue outline, layers,
    vessels) and, unless ``independent_nuclei``, the same nuclei, so they register
    like a same-section pair; with independent nuclei only the tissue-scale
    structure is shared, like adjacent serial sections. The moving image is the
    reference field seen through a similarity (``rotation_deg``, ``mirror_x``,
    ``offset_um`` about the image centres, see ``coarse_matrix_xy``) plus a smooth
    sinusoidal warp of amplitude ``warp_amplitude_um`` and wavelength
    ``warp_wavelength_um``. Output is uint16 with Gaussian noise (sigma 5), tiled
    and pyramidal with levels down to 256 px (so prepare uses the files in place).

    Args:
        out_dir: Directory for ``synthetic_reference.ome.tif`` and
            ``synthetic_moving.ome.tif``.
        reference_shape_yx: Reference ``(H, W)``.
        reference_pixel_size_um: Reference pixel size.
        moving_pixel_size_um: Moving pixel size.
        rotation_deg: Coarse rotation (counter-clockwise on screen).
        mirror_x: Whether the moving image is mirrored.
        offset_um: Coarse offset of the moving centre from the reference centre.
        warp_amplitude_um: Amplitude of the smooth warp.
        warp_wavelength_um: Wavelength of the smooth warp.
        channel_names: Channel names of both images (the first is the DAPI-like
            channel).
        seed: Random seed; the output is deterministic in it.
        independent_nuclei: Draw the moving nuclei independently.

    Returns:
        The two paths and the ground truth.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p_r, p_m = float(reference_pixel_size_um), float(moving_pixel_size_um)
    height_r, width_r = (int(v) for v in reference_shape_yx)
    extent = _canvas_extent((height_r, width_r), p_r)
    background, tissue, layer, vessels = _shared_background(extent, seed)
    nuclei_reference = _nuclei(tissue, layer, vessels, seed + 1)
    nuclei_moving = (
        _nuclei(tissue, layer, vessels, seed + 2)
        if independent_nuclei
        else nuclei_reference
    )
    field_reference = background + nuclei_reference
    field_moving = background + nuclei_moving

    moving_shape = (
        math.ceil(height_r * p_r / p_m),
        math.ceil(width_r * p_r / p_m),
    )
    true_matrix = coarse_matrix_xy(
        CoarseTransform(
            rotation_deg=rotation_deg, mirror_x=mirror_x, offset_um=offset_um
        ),
        moving_shape_yx=moving_shape,
        reference_shape_yx=(height_r, width_r),
        moving_pixel_size_um=p_m,
        reference_pixel_size_um=p_r,
    )
    amplitude_px = warp_amplitude_um / p_r

    def moving_to_reference(points: np.ndarray) -> np.ndarray:
        r0 = apply_matrix_xy(true_matrix, np.asarray(points, dtype=np.float64))
        shift = np.stack(
            [
                np.sin(2.0 * math.pi * p_r * r0[..., 1] / warp_wavelength_um + 0.3),
                np.sin(2.0 * math.pi * p_r * r0[..., 0] / warp_wavelength_um + 1.1),
            ],
            axis=-1,
        )
        return np.asarray(r0 + amplitude_px * shift, dtype=np.float64)

    n_channels = len(channel_names)
    rng = np.random.default_rng(seed + 3)
    reference_dapi = _render(field_reference, lambda r: p_r * r, (height_r, width_r))
    moving_dapi = _render(
        field_moving, lambda m: p_r * moving_to_reference(m), moving_shape
    )
    reference_path = out_dir / "synthetic_reference.ome.tif"
    moving_path = out_dir / "synthetic_moving.ome.tif"
    _write_ome(
        _channels(reference_dapi, n_channels, rng),
        reference_path,
        channel_names=channel_names,
        pixel_size_um=p_r,
        image_name="synthetic_reference",
    )
    _write_ome(
        _channels(moving_dapi, n_channels, rng),
        moving_path,
        channel_names=channel_names,
        pixel_size_um=p_m,
        image_name="synthetic_moving",
    )
    return SyntheticPair(
        reference_path=reference_path,
        moving_path=moving_path,
        reference_pixel_size_um=p_r,
        moving_pixel_size_um=p_m,
        true_matrix=true_matrix,
        moving_to_reference=moving_to_reference,
        reference_shape_yx=(height_r, width_r),
        moving_shape_yx=moving_shape,
    )


def sample_tissue_points(
    dapi: np.ndarray,
    to_level0: np.ndarray,
    *,
    n: int,
    seed: int,
    min_separation_px: float,
) -> np.ndarray:
    """Sample well-separated points inside the tissue of a working image.

    The tissue mask is the Otsu threshold of the smoothed log image; candidates
    lie at least 3 px inside it. Farthest-point sampling spreads the points over
    the tissue (distances in level-0 pixels).

    Args:
        dapi: 2-D working image whose pixel ``(j, i)`` maps to level-0 via
            ``to_level0``.
        to_level0: 3x3 working-pixel to level-0 matrix (pixel-centre convention).
        n: Number of points.
        seed: Random seed.
        min_separation_px: Smallest accepted distance between points (level-0 px).

    Returns:
        ``(n, 2)`` float64 level-0 ``(x, y)`` points.

    Raises:
        ValueError: If fewer than ``n`` points fit at the requested separation.
    """
    image = np.asarray(dapi, dtype=np.float64)
    smoothed = gaussian_filter(np.log1p(np.maximum(image, 0.0)), 2.0)
    mask = smoothed > threshold_otsu(smoothed)
    inside = distance_transform_edt(mask) >= 3
    rows, cols = np.nonzero(inside)
    rng = np.random.default_rng(seed)
    if len(rows) > _MAX_CANDIDATES:
        pick = rng.choice(len(rows), size=_MAX_CANDIDATES, replace=False)
        rows, cols = rows[pick], cols[pick]
    if len(rows) == 0:
        raise ValueError("only 0 tissue points at the requested separation")
    candidates = apply_matrix_xy(
        np.asarray(to_level0, dtype=np.float64),
        np.column_stack([cols, rows]).astype(np.float64),
    )
    chosen = [int(rng.integers(0, len(candidates)))]
    distance = np.linalg.norm(candidates - candidates[chosen[0]], axis=1)
    while len(chosen) < n:
        nxt = int(np.argmax(distance))
        if distance[nxt] < min_separation_px:
            raise ValueError(
                f"only {len(chosen)} tissue points at >= {min_separation_px:g} px "
                "separation"
            )
        chosen.append(nxt)
        distance = np.minimum(
            distance, np.linalg.norm(candidates - candidates[nxt], axis=1)
        )
    return np.asarray(candidates[chosen], dtype=np.float64)


def simulate_landmark_pairs(
    moving_xy: np.ndarray,
    moving_to_reference: Callable[[np.ndarray], np.ndarray],
    *,
    noise_um: float,
    reference_pixel_size_um: float,
    seed: int,
    colour_seed: int = 0,
    outliers: Mapping[int, tuple[float, float]] | None = None,
    disabled: Sequence[int] = (),
) -> list[LandmarkPair]:
    """Simulate landmark pairs from moving points and the true mapping.

    The reference point of pair ``id`` is the true image of the moving point plus
    Gaussian noise (``noise_um``); ``outliers`` adds a fixed offset in reference
    micrometres to the given ids. Coordinates are rounded to 3 decimals.

    Args:
        moving_xy: ``(N, 2)`` moving level-0 points; pair ids are ``1..N``.
        moving_to_reference: True map of moving points to reference points.
        noise_um: Standard deviation of the placement noise in reference um.
        reference_pixel_size_um: Reference pixel size.
        seed: Random seed.
        colour_seed: Seed of the pair colours.
        outliers: ``{id: (dx_um, dy_um)}`` offsets.
        disabled: Ids to disable.
    """
    moving = np.asarray(moving_xy, dtype=np.float64).reshape(-1, 2)
    rng = np.random.default_rng(seed)
    reference = np.asarray(moving_to_reference(moving), dtype=np.float64)
    reference = reference + rng.normal(
        0.0, noise_um / reference_pixel_size_um, reference.shape
    )
    for pair_id, (dx_um, dy_um) in (outliers or {}).items():
        reference[pair_id - 1] += np.array([dx_um, dy_um]) / reference_pixel_size_um
    disabled_ids = set(disabled)
    pairs = []
    for index in range(len(moving)):
        pair_id = index + 1
        pairs.append(
            LandmarkPair(
                id=pair_id,
                colour=pair_colour(pair_id, colour_seed),
                enabled=pair_id not in disabled_ids,
                reference_xy_px=(
                    round(float(reference[index, 0]), 3),
                    round(float(reference[index, 1]), 3),
                ),
                moving_xy_px=(
                    round(float(moving[index, 0]), 3),
                    round(float(moving[index, 1]), 3),
                ),
            )
        )
    return pairs


def _crop_dict(crop: CropRect | None) -> dict[str, int] | None:
    return None if crop is None else crop.model_dump()


def inject_labelling(
    store: Path,
    pairs: Sequence[LandmarkPair],
    *,
    coarse: CoarseTransform | None = None,
    crops: Mapping[str, CropRect | None] | None = None,
    status: str = "finished",
) -> int:
    """Write a labelling into a prepared store, as the viewer's save does.

    Holds the store's write lock, increments the stored ``revision``, validates
    the payload against ``LabellingAttrs`` and sets only ``LABELLING_ATTR``.

    Args:
        store: A prepared store (``STORE_ATTR.kind == "prepared"``).
        pairs: The landmark pairs.
        coarse: The coarse state; ``None`` fits both reflection classes to the
            enabled pairs and keeps the better one.
        crops: ``{"reference": CropRect | None, "moving": CropRect | None}``.
        status: ``"draft"`` or ``"finished"``.

    Returns:
        The new revision.

    Raises:
        ValueError: On a bad status or a store that is not a prepared store.
        UnsupportedSchemaVersionError: If the store attrs are from a newer writer.
    """
    if status not in ("draft", "finished"):
        raise ValueError(f"status must be 'draft' or 'finished', got {status!r}")
    store = Path(store)
    with spatialdata_write_lock(store):
        attrs = read_root_attrs(store)
        if STORE_ATTR not in attrs:
            raise ValueError(f"{store} is not a manual-registration store")
        store_attr = parse_store_attrs(attrs[STORE_ATTR])
        if store_attr.kind != "prepared":
            raise ValueError(
                f"{store} is a {store_attr.kind} store, not a prepared one"
            )
        old = attrs.get(LABELLING_ATTR)
        revision = int(old["revision"]) + 1 if old else 1
        reference = store_attr.images["reference"]
        moving = store_attr.images["moving"]

        def matrix_of(state: CoarseTransform) -> np.ndarray:
            return coarse_matrix_xy(
                state,
                moving_shape_yx=moving.shape_yx,
                reference_shape_yx=reference.shape_yx,
                moving_pixel_size_um=moving.pixel_size_um,
                reference_pixel_size_um=reference.pixel_size_um,
            )

        if coarse is None:
            enabled = [p for p in pairs if p.enabled]
            mov = np.array([p.moving_xy_px for p in enabled], dtype=np.float64)
            ref = np.array([p.reference_xy_px for p in enabled], dtype=np.float64)
            fits = [fit_similarity(mov, ref, reflection=flag) for flag in (False, True)]
            best = min(fits, key=lambda fit: float(np.median(fit.residuals_px)))
            coarse = coarse_from_matrix(
                best.matrix,
                moving_shape_yx=moving.shape_yx,
                reference_shape_yx=reference.shape_yx,
                moving_pixel_size_um=moving.pixel_size_um,
                reference_pixel_size_um=reference.pixel_size_um,
            )[0]
        coarse = CoarseTransform(
            rotation_deg=round(coarse.rotation_deg, 4),
            mirror_x=coarse.mirror_x,
            offset_um=(round(coarse.offset_um[0], 4), round(coarse.offset_um[1], 4)),
        )
        crop_map = dict(crops or {})
        now = utc_now()
        ids = [p.id for p in pairs]
        payload = {
            "schema": LABELLING_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "revision": revision,
            "updated_at": now,
            "finished_at": now
            if status == "finished"
            else (old or {}).get("finished_at"),
            "written_by": {
                "software": "merxen.manual_registration.simulate",
                "version": merxen.__version__,
            },
            "store_grid_digest": store_attr.grid_digest,
            "images": {
                "reference": {
                    "shape_yx": list(reference.shape_yx),
                    "pixel_size_um": reference.pixel_size_um,
                },
                "moving": {
                    "shape_yx": list(moving.shape_yx),
                    "pixel_size_um": moving.pixel_size_um,
                },
            },
            "coarse": {
                "rotation_deg": coarse.rotation_deg,
                "mirror_x": coarse.mirror_x,
                "offset_um": list(coarse.offset_um),
                "matrix_xy": matrix_of(coarse).tolist(),
            },
            "crops": {
                "reference": _crop_dict(crop_map.get("reference")),
                "moving": _crop_dict(crop_map.get("moving")),
            },
            "landmarks": {
                "next_id": (max(ids) + 1) if ids else 1,
                "colour_seed": 0,
                "pairs": [p.model_dump(mode="json") for p in pairs],
            },
        }
        payload["content_sha256"] = labelling_content_sha256(payload)
        parse_labelling_attrs(payload)
        write_root_attr(store, LABELLING_ATTR, payload)
    return revision

"""Registry of the spatial transcriptomics platforms MerXen supports.

Each supported platform has one :class:`PlatformSpec` that holds the names and
defaults the pipeline used to spell out in binary ``MERSCOPE``/``XENIUM``
branches: the native morphology image, the original (vendor) shape layers, the
default Cellpose channels, the quality-score column alias and the pixel-size
fallback. Code that needs one of these values looks it up with
:func:`platform_spec`, which raises :class:`UnknownPlatformError` for a
platform that is not registered instead of silently treating it like one of
the others.

The Nextflow workflow keeps a twin of this registry in
``workflows/lib/PlatformSpec.groovy``; ``tests/test_platforms.py`` and
``tests/test_workflows/test_platform_registry_groovy.py`` check that the two
agree. Adding a platform means adding an entry to both.

This module imports the standard library only, so every MerXen module
(configuration, enrichment, segmentation) can import it without a cycle.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal

PlatformName = Literal["MERSCOPE", "XENIUM", "ATERA"]
"""Names of the registered platforms.

Each configuration ``platform`` Literal is widened to this set only when its
stage handles every platform; until then it lists the platforms it supports.
"""

PixelConvention = Literal["center", "corner"]
"""How a micron coordinate maps to a vendor mask pixel.

``"center"`` rounds to the nearest pixel centre (``rint``), ``"corner"`` takes
the pixel whose ``[i * ps, (i + 1) * ps)`` interval holds it (``floor``).
"""

PlatformFamily = Literal["10X", "MERSCOPE"]
"""Vendor family of a platform; members of a family share readers and layouts."""

SegmentationSource = Literal["native", "cellpose_only", "cellpose_proseg"]
"""Where the cells of a row come from.

``"native"`` keeps the vendor segmentation (masks, shapes and table) and runs
no Cellpose or ProSeg; ``"cellpose_only"`` runs Cellpose without ProSeg
(reserved for AT-7); ``"cellpose_proseg"`` is the Cellpose prior refined by
ProSeg that XENIUM and MERSCOPE rows run.
"""

_SEGMENTATION_LAYERS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "native": ("original_seg",),
        "cellpose_proseg": ("reseg", "original_seg", "proseg_mask", "proseg_hybrid"),
        "cellpose_only": ("proseg_mask", "original_seg"),
    }
)


class UnknownPlatformError(ValueError):
    """Raised when a platform name is not in the registry.

    It subclasses ``ValueError`` (not ``FileNotFoundError``), so code that
    recovers from a missing transform file does not swallow it.
    """


@dataclass(frozen=True)
class PlatformSpec:
    """Names and defaults of one supported platform.

    Attributes:
        name: Canonical upper-case platform name, e.g. ``"XENIUM"``.
        family: Vendor family of the platform.
        native_image_key: SpatialData image element of the native morphology
            image used for Cellpose segmentation.
        original_cell_shape_key: Shapes element of the vendor cell segmentation.
        original_nucleus_shape_key: Shapes element of the vendor nucleus
            segmentation, or ``None`` when the platform ships none.
        default_channels: Cellpose channel names used when a samplesheet row
            does not override them.
        quality_column_alias: Transcript column renamed to the shared quality
            column of the latest store, or ``None`` when no alias is needed.
        fallback_um_per_px: Isotropic microns per morphology pixel used when no
            transform or spec file can be found.
        nucleus_ids_are_cell_ids: Whether the vendor nucleus layer is keyed by
            cell IDs. When False (ATERA: one row per nucleus), nucleus
            consumers join through ``cell_instance_id``.
        spec_sidecar_names: Spec files holding ``pixel_size``, relative to a
            built store or raw folder, in lookup order (``find_um_per_px``).
        mask_pixel_convention: Pixel convention of micron lookups into the
            vendor masks.
        segmentation_sources: Segmentation sources a row of this platform may
            use (``segmentation_source``); others are refused at config time.
        default_segmentation_source: Source used when a row names none.
        original_label_keys: Vendor label elements aligned with
            :attr:`original_shape_keys` (cell labels, then nucleus labels), or
            ``()`` when the build writes none.
        rasterised_transcript_plots: Whether static transcript figures are
            drawn as density rasters over every transcript instead of point
            scatters (whole-transcriptome panels).
    """

    name: str
    family: PlatformFamily
    native_image_key: str
    original_cell_shape_key: str
    original_nucleus_shape_key: str | None
    default_channels: tuple[str, ...]
    quality_column_alias: str | None
    fallback_um_per_px: float
    nucleus_ids_are_cell_ids: bool = True
    spec_sidecar_names: tuple[str, ...] = ()
    mask_pixel_convention: PixelConvention = "center"
    segmentation_sources: tuple[SegmentationSource, ...] = ("cellpose_proseg",)
    default_segmentation_source: SegmentationSource = "cellpose_proseg"
    original_label_keys: tuple[str, ...] = ()
    rasterised_transcript_plots: bool = False

    @property
    def original_shape_keys(self) -> tuple[str, ...]:
        """Return the original cell shape key, then the nucleus key if any."""
        if self.original_nucleus_shape_key is None:
            return (self.original_cell_shape_key,)
        return (self.original_cell_shape_key, self.original_nucleus_shape_key)


PLATFORM_SPECS: Final[Mapping[str, PlatformSpec]] = MappingProxyType(
    {
        "MERSCOPE": PlatformSpec(
            name="MERSCOPE",
            family="MERSCOPE",
            native_image_key="MERSCOPE_z_projection",
            original_cell_shape_key="merscope_cell_boundaries",
            original_nucleus_shape_key=None,
            default_channels=("DAPI", "PolyT"),
            quality_column_alias="transcript_score",
            fallback_um_per_px=0.108,
        ),
        "XENIUM": PlatformSpec(
            name="XENIUM",
            family="10X",
            native_image_key="morphology_focus",
            original_cell_shape_key="xenium_cell_boundaries",
            original_nucleus_shape_key="xenium_nucleus",
            default_channels=("DAPI", "18S"),
            quality_column_alias=None,
            fallback_um_per_px=0.2125,
            spec_sidecar_names=(
                "experiment.xenium",
                "specs.json",
                "specs/specs.json",
            ),
        ),
        "ATERA": PlatformSpec(
            name="ATERA",
            family="10X",
            native_image_key="morphology_focus",
            original_cell_shape_key="atera_cell_boundaries",
            original_nucleus_shape_key="atera_nucleus",
            default_channels=("DAPI", "18S"),
            quality_column_alias=None,
            fallback_um_per_px=0.21249222,
            nucleus_ids_are_cell_ids=False,
            # A spec override (kept beside the bundle's manifest) set the pixel
            # size the build used, so it wins over the original.
            spec_sidecar_names=(
                "atera/experiment_override.spatial",
                "atera/experiment.spatial",
                "experiment.spatial",
            ),
            mask_pixel_convention="corner",
            segmentation_sources=("native",),
            default_segmentation_source="native",
            original_label_keys=("cell_labels", "nucleus_labels"),
            rasterised_transcript_plots=True,
        ),
    }
)
"""Every registered platform by canonical name: MERSCOPE, XENIUM, ATERA."""

ORIGINAL_CELL_SHAPE_KEYS: Final[frozenset[str]] = frozenset(
    spec.original_cell_shape_key for spec in PLATFORM_SPECS.values()
)
"""Original (vendor) cell shape keys of every supported platform."""

ORIGINAL_NUCLEUS_SHAPE_KEYS: Final[frozenset[str]] = frozenset(
    spec.original_nucleus_shape_key
    for spec in PLATFORM_SPECS.values()
    if spec.original_nucleus_shape_key is not None
)
"""Original (vendor) nucleus shape keys of every supported platform."""


def supported_platforms() -> tuple[str, ...]:
    """Return the canonical names of the supported platforms in registry order.

    Returns:
        The platform names, e.g. ``("MERSCOPE", "XENIUM", "ATERA")``.
    """
    return tuple(PLATFORM_SPECS)


def segmentation_layers(source: SegmentationSource | str) -> tuple[str, ...]:
    """Return the analysis layers a segmentation source produces.

    Args:
        source: A segmentation source (``native``, ``cellpose_only`` or
            ``cellpose_proseg``).

    Returns:
        The analysis segmentations (``analysis_segmentation`` values) the
        source makes available, in the workflow's default order.

    Raises:
        ValueError: If the source is not a known segmentation source.
    """
    layers = _SEGMENTATION_LAYERS.get(str(source))
    if layers is None:
        raise ValueError(
            f"Unknown segmentation source: {source!r}. "
            f"Known: {', '.join(_SEGMENTATION_LAYERS)}"
        )
    return layers


def _canonical_name(platform: object) -> str:
    """Return the registry key a platform name is looked up by."""
    return str(platform).strip().upper()


def platform_spec(platform: str) -> PlatformSpec:
    """Return the registry entry of a platform.

    Args:
        platform: Platform name; surrounding whitespace and case are ignored.

    Returns:
        The platform's :class:`PlatformSpec`.

    Raises:
        UnknownPlatformError: If the platform is not registered. The message
            names the supported platforms.
    """
    spec = PLATFORM_SPECS.get(_canonical_name(platform))
    if spec is None:
        raise UnknownPlatformError(
            f"Unknown platform: {platform!r}. "
            f"Supported: {', '.join(supported_platforms())}"
        )
    return spec


def find_platform_spec(platform: str | None) -> PlatformSpec | None:
    """Return the registry entry of a platform, or ``None`` if it has none.

    Tolerant lookup for display code that also handles labels which are not
    platforms. Pipeline logic uses :func:`platform_spec`, which raises.

    Args:
        platform: Platform name or another label; ``None`` is allowed.

    Returns:
        The platform's :class:`PlatformSpec`, or ``None``.
    """
    if platform is None:
        return None
    return PLATFORM_SPECS.get(_canonical_name(platform))


def spec_file_pixel_size(path: Path) -> float | None:
    """Read the microns per pixel from a JSON spec file.

    Args:
        path: A spec file such as ``experiment.xenium`` or ``experiment.spatial``.

    Returns:
        Its ``pixel_size`` (else ``microns_per_pixel``) as a float, or ``None``
        when the file is missing, is not a JSON object or holds neither key.
    """
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    for key in ("pixel_size", "microns_per_pixel"):
        if key in data:
            try:
                return float(data[key])
            except (TypeError, ValueError):
                return None
    return None


def find_um_per_px(root: Path, platform: str) -> float:
    """Return the microns per morphology pixel recorded under a store or folder.

    The platform's ``spec_sidecar_names`` are tried in order under ``root``
    (a built SpatialData store or a raw output folder); the first file that
    yields a pixel size (:func:`spec_file_pixel_size`) wins.

    Args:
        root: Built store or raw folder.
        platform: Registered platform name.

    Returns:
        Microns per level-0 morphology pixel.

    Raises:
        UnknownPlatformError: If the platform is not registered.
        FileNotFoundError: If no spec file under ``root`` gives a pixel size
            (MERSCOPE stores record a transform matrix instead).
    """
    spec = platform_spec(platform)
    for name in spec.spec_sidecar_names:
        value = spec_file_pixel_size(Path(root, *name.split("/")))
        if value is not None:
            return value
    raise FileNotFoundError(
        f"No {spec.name} pixel size under the store: looked for "
        f"{', '.join(spec.spec_sidecar_names) or 'no spec file'}"
    )

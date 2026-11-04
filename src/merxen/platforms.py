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

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

PlatformName = Literal["MERSCOPE", "XENIUM"]
"""Names of the supported platforms (the configuration Literals use the same set)."""

PlatformFamily = Literal["10X", "MERSCOPE"]
"""Vendor family of a platform; members of a family share readers and layouts."""


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
    """

    name: str
    family: PlatformFamily
    native_image_key: str
    original_cell_shape_key: str
    original_nucleus_shape_key: str | None
    default_channels: tuple[str, ...]
    quality_column_alias: str | None
    fallback_um_per_px: float

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
        ),
    }
)
"""Every supported platform by canonical name, in the order MERSCOPE, XENIUM."""

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
        The platform names, e.g. ``("MERSCOPE", "XENIUM")``.
    """
    return tuple(PLATFORM_SPECS)


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

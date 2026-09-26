"""Optional annotation columns of the pipeline samplesheet (plan §3.7, hook H9).

Two optional per-row columns override the global annotation params:

- ``anatomical_region`` (human; default ``annotation_human_region``);
- ``mouse_section_regions`` (mouse; ``auto``, ``none`` or ``;``-separated CCF
  divisions; default ``annotation_mouse_section_regions``).

Blank or missing values inherit the global param, so existing samplesheets
parse unchanged. ``merxen.io.samplesheet`` calls ``parse_optional_columns``
at hook H9, and ``ClusteringSquidpySampleConfig`` types the per-row values
with ``AnatomicalRegionValue`` and ``MouseSectionRegionsValue`` (hook H8).
Species-specific checks (only ``frontal_cortex`` for human) run in
``AnnotationConfig``, which knows the run species.

This module imports only the standard library and pydantic.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Annotated, Any, Final

from pydantic import BeforeValidator

from merxen.annotation.config import normalise_mouse_section_regions

ANATOMICAL_REGION_COLUMN: Final = "anatomical_region"
MOUSE_SECTION_REGIONS_COLUMN: Final = "mouse_section_regions"
OPTIONAL_ANNOTATION_COLUMNS: Final[tuple[str, ...]] = (
    ANATOMICAL_REGION_COLUMN,
    MOUSE_SECTION_REGIONS_COLUMN,
)
_REGION_TOKEN_PATTERN: Final = re.compile(r"^[a-z0-9_]+$")


@dataclass(frozen=True)
class OptionalAnnotationColumns:
    """Parsed optional annotation columns of one samplesheet row.

    Attributes:
        anatomical_region: Region token (e.g. ``frontal_cortex``), or ``None``
            to inherit the global param.
        mouse_section_regions: ``auto``, ``none`` or canonical
            ``;``-separated divisions, or ``None`` to inherit.
    """

    anatomical_region: str | None = None
    mouse_section_regions: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        """Return the fields as a plain mapping (for ``SamplePair`` kwargs).

        Returns:
            Column name to parsed value.
        """
        return asdict(self)


def parse_anatomical_region(value: str | None) -> str | None:
    """Normalise an ``anatomical_region`` cell.

    Args:
        value: Raw cell text; spaces and hyphens become underscores, case is
            folded (``"Frontal cortex"`` → ``"frontal_cortex"``).

    Returns:
        The region token, or ``None`` for a blank cell.

    Raises:
        ValueError: If the value is not a region token after normalisation.
    """
    if value is None or not str(value).strip():
        return None
    token = re.sub(r"[\s\-]+", "_", str(value).strip().lower())
    if not _REGION_TOKEN_PATTERN.fullmatch(token):
        raise ValueError(
            f"invalid anatomical_region {value!r}: use a token such as 'frontal_cortex'"
        )
    return token


def parse_mouse_section_regions(value: str | None) -> str | None:
    """Normalise a ``mouse_section_regions`` cell.

    Args:
        value: Raw cell text: ``auto``, ``none`` or ``;``-separated CCF
            divisions (case insensitive).

    Returns:
        The canonical value, or ``None`` for a blank cell.

    Raises:
        ValueError: On an unknown division.
    """
    if value is None or not str(value).strip():
        return None
    return normalise_mouse_section_regions(str(value))


def _parse_anatomical_region_value(value: Any) -> str | None:
    return None if value is None else parse_anatomical_region(str(value))


def _parse_mouse_section_regions_value(value: Any) -> str | None:
    return None if value is None else parse_mouse_section_regions(str(value))


# Pydantic field types for the per-row values carried in ``samples_json``:
# blank means "inherit the global param" (``None``), others are normalised.
AnatomicalRegionValue = Annotated[
    str | None, BeforeValidator(_parse_anatomical_region_value)
]
MouseSectionRegionsValue = Annotated[
    str | None, BeforeValidator(_parse_mouse_section_regions_value)
]


def parse_optional_columns(
    row: Mapping[str, str | None],
) -> OptionalAnnotationColumns:
    """Parse the optional annotation columns of one samplesheet row.

    Args:
        row: One ``csv.DictReader`` row; absent columns count as blank.

    Returns:
        The parsed columns (``None`` where the row inherits the global param).

    Raises:
        ValueError: If a present value is invalid.
    """
    return OptionalAnnotationColumns(
        anatomical_region=parse_anatomical_region(row.get(ANATOMICAL_REGION_COLUMN)),
        mouse_section_regions=parse_mouse_section_regions(
            row.get(MOUSE_SECTION_REGIONS_COLUMN)
        ),
    )


def split_section_regions(value: str) -> list[str] | None:
    """Split a canonical ``mouse_section_regions`` value into divisions.

    Args:
        value: ``auto``, ``none`` or canonical ``;``-separated divisions.

    Returns:
        ``None`` for ``auto`` (infer regions), ``[]`` for ``none`` (no
        pruning), else the divisions.
    """
    canonical = normalise_mouse_section_regions(value)
    if canonical == "auto":
        return None
    if canonical == "none":
        return []
    return canonical.split(";")

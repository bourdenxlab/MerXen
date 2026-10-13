"""Optional annotation columns of the pipeline samplesheet (plan §3.7, hook H9).

Three optional per-row columns override the global annotation params:

- ``anatomical_region`` (human; default ``annotation_human_region``);
- ``mouse_section_regions`` (mouse; ``auto``, ``none`` or ``;``-separated CCF
  divisions; default ``annotation_mouse_section_regions``);
- ``clustering_squidpy_mode`` (``legacy`` or ``map_first``; default the run's
  mode, ``clustering_squidpy_mode*``; pre-registration §20 D15). Groovy's
  ``AnnotationDefaults.resolveRowMode`` applies it in the pipeline.

Blank or missing values inherit the global param, so existing samplesheets
parse unchanged. ``merxen.io.samplesheet`` calls ``parse_optional_columns``
at hook H9, and ``ClusteringSquidpySampleConfig`` types the per-row values
with ``AnatomicalRegionValue`` and ``MouseSectionRegionsValue`` (hook H8).
Species-specific checks (only ``frontal_cortex`` for human) run in
``AnnotationConfig``, which knows the run species.

The MAP step reads each sample's ``mouse_section_regions`` from the
``samples`` of ``clustering_squidpy_config.json`` (``samples_json``;
``section_regions_by_sample``) and ``effective_section_regions`` gives a
sample its value: its own, else the global default (M6, plan §7.2).

This module imports only the standard library, numpy, pandas and pydantic
(numpy and pandas through ``merxen.annotation.config``).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Annotated, Any, Final

from pydantic import BeforeValidator

from merxen.annotation.config import (
    CLUSTERING_MODES,
    ROW_MODE_COLUMN,
    normalise_mouse_section_regions,
)

ANATOMICAL_REGION_COLUMN: Final = "anatomical_region"
MOUSE_SECTION_REGIONS_COLUMN: Final = "mouse_section_regions"
CLUSTERING_MODE_COLUMN: Final = ROW_MODE_COLUMN
OPTIONAL_ANNOTATION_COLUMNS: Final[tuple[str, ...]] = (
    ANATOMICAL_REGION_COLUMN,
    MOUSE_SECTION_REGIONS_COLUMN,
    CLUSTERING_MODE_COLUMN,
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
        clustering_squidpy_mode: ``legacy`` or ``map_first``, or ``None``
            for the run's mode.
    """

    anatomical_region: str | None = None
    mouse_section_regions: str | None = None
    clustering_squidpy_mode: str | None = None

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


def parse_clustering_mode(value: str | None) -> str | None:
    """Normalise a ``clustering_squidpy_mode`` cell (as Groovy ``checkedMode``).

    Args:
        value: Raw cell text, case insensitive.

    Returns:
        ``legacy`` or ``map_first``, or ``None`` for a blank cell.

    Raises:
        ValueError: On another value.
    """
    if value is None or not str(value).strip():
        return None
    mode = str(value).strip().lower()
    if mode not in CLUSTERING_MODES:
        raise ValueError(
            f"invalid clustering_squidpy_mode {value!r}: use one of "
            f"{', '.join(CLUSTERING_MODES)}"
        )
    return mode


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
        clustering_squidpy_mode=parse_clustering_mode(row.get(CLUSTERING_MODE_COLUMN)),
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


def section_regions_by_sample(
    samples: Iterable[Mapping[str, Any]] | None,
) -> dict[str, str]:
    """Return the per-sample ``mouse_section_regions`` of a sample list.

    Args:
        samples: ``samples`` entries of ``clustering_squidpy_config.json``
            (``ClusteringSquidpySampleConfig`` dicts; ``samples_json``).

    Returns:
        Sample id to its canonical value, for the samples that set one
        (blank and absent values inherit the global param and are left out).

    Raises:
        ValueError: If a value is invalid or a sample has no ``sample_id``.
    """
    values: dict[str, str] = {}
    for sample in samples or ():
        if not isinstance(sample, Mapping):
            continue
        raw = sample.get(MOUSE_SECTION_REGIONS_COLUMN)
        if raw is None or not str(raw).strip():
            continue
        sample_id = str(sample.get("sample_id") or "").strip()
        if not sample_id:
            raise ValueError("a sample with mouse_section_regions has no sample_id")
        parsed = parse_mouse_section_regions(str(raw))
        if parsed is not None:
            values[sample_id] = parsed
    return values


def effective_section_regions(sample_value: str | None, default: str = "auto") -> str:
    """Return a sample's ``mouse_section_regions``: its own, else the default.

    Args:
        sample_value: The sample's value (``None`` or blank: inherit).
        default: The global value (``annotation_mouse_section_regions``).

    Returns:
        The canonical value.

    Raises:
        ValueError: If the chosen value is invalid.
    """
    own = parse_mouse_section_regions(sample_value)
    return own if own is not None else normalise_mouse_section_regions(default)

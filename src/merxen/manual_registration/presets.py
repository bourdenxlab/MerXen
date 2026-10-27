"""Registration presets, the settings-override grammar and settings resolution.

A preset is a named set of ``ManualRegistrationSettings`` values. ``same_section``
is the settings' defaults (equal to the ``REGISTER_IMAGES`` defaults except that
the non-rigid stage is on); ``serial_section`` is tuned for adjacent serial
sections, where only tissue-scale structure matches. Overrides come from the
Nextflow params and from the samplesheet row as ``"key=value;preset:key=value"``
text and are validated against the settings model.

This module stays light (pydantic and the config models); it imports no
SpatialData or plotting code. The two tuning dataclasses are imported inside
``registration_tuning``, because their modules load both.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from merxen.manual_registration.config import ManualRegistrationSettings
from merxen.manual_registration.schema import CROP_SNAP_PX

if TYPE_CHECKING:
    from merxen.image_registration import RegistrationTuning

logger = logging.getLogger(__name__)

# Changing any value below requires bumping its PRESET_VERSIONS entry: the version
# is part of the provenance digest and of the result attrs.
PRESETS: dict[str, dict[str, Any]] = {
    "same_section": {},
    "serial_section": {
        "matrix_scale_tolerance": 0.05,
        "registration_pixel_size_um": 6.0,
        "refinement_window_um": 1200.0,
        "min_refinement_windows": 12,
        "max_refinement_shift_um": 200.0,
        "max_refinement_rotation_deg": 5.0,
        "max_refinement_scale_change": 0.05,
        "refinement_min_window_quality": 0.2,
        "refinement_require_moving_tissue": True,
        "nonrigid_enabled": True,
        "nonrigid_pixel_size_um": 12.0,
        "nonrigid_smoothing_sigma_um": 100.0,
        "nonrigid_max_displacement_um": 200.0,
        "nonrigid_max_cycle_error_um": 30.0,
        "nonrigid_min_local_correlation": 0.15,
        "nonrigid_min_jacobian": 0.67,
        "nonrigid_max_jacobian": 1.5,
        "nonrigid_highpass_sigma_um": 120.0,
        "nonrigid_local_correlation_sigma_um": 80.0,
        "nonrigid_local_residual_tiles": False,
        "nonrigid_local_residual_tile_um": 1500.0,
        "landmark_outlier_floor_um": 30.0,
        "nucleus_qc": True,
    },
}
PRESET_VERSIONS: dict[str, str] = {
    "same_section": "same_section_v1",
    "serial_section": "serial_section_v1",
}
# Fixed by the samplesheet and the prepared store, never by a settings override.
FORBIDDEN_OVERRIDES: frozenset[str] = frozenset(
    {"reference_pixel_size_um", "reference_channel"}
)

_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def _preset_list() -> str:
    return ", ".join(sorted(PRESETS))


def parse_settings_overrides(text: str) -> dict[str | None, dict[str, str]]:
    """Parse ``"key=value;preset:key=value"`` override text.

    Items are separated by ``;`` (empty items are skipped). An item is
    ``[preset:]key=value``; the key must match ``^[a-z][a-z0-9_]*$`` and the value
    must be non-empty. Values stay strings; pydantic coerces them later.

    Args:
        text: The override text; empty or blank gives no overrides.

    Returns:
        ``{scope: {key: value}}`` where the scope is ``None`` for unprefixed items
        and the preset name for prefixed ones; only non-empty scopes are present.

    Raises:
        ValueError: On a malformed item, an unknown preset prefix or a key given
            twice in one scope. Messages start with ``registration settings: ``.
    """
    result: dict[str | None, dict[str, str]] = {}
    for raw in text.split(";"):
        item = raw.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"registration settings: {item!r} is not of the form key=value"
            )
        left, value = item.split("=", 1)
        left, value = left.strip(), value.strip()
        preset: str | None = None
        key = left
        if ":" in left:
            prefix, key = (part.strip() for part in left.split(":", 1))
            preset = prefix
            if preset not in PRESETS:
                raise ValueError(
                    f"registration settings: unknown preset {preset!r} in {item!r}; "
                    f"presets: {_preset_list()}"
                )
        if not key or not value:
            raise ValueError(f"registration settings: empty key or value in {item!r}")
        if not _KEY_PATTERN.match(key):
            raise ValueError(
                f"registration settings: invalid key {key!r} in {item!r} "
                "(lower-case letters, digits and underscores, starting with a letter)"
            )
        scope = result.setdefault(preset, {})
        if key in scope:
            where = "all presets" if preset is None else preset
            raise ValueError(
                f"registration settings: {key!r} given twice for scope {where}"
            )
        scope[key] = value
    return result


def resolve_settings(
    preset: str,
    *,
    reference_pixel_size_um: float,
    reference_channel: str,
    global_settings: str = "",
    row_settings: str = "",
) -> ManualRegistrationSettings:
    """Resolve the settings of one run.

    Precedence, lowest to highest: the preset, global unprefixed, global
    ``<preset>:``, row unprefixed, row ``<preset>:``. The reference pixel size and
    channel always come from the prepared store.

    Args:
        preset: ``same_section`` or ``serial_section``.
        reference_pixel_size_um: Reference level-0 pixel size of the store.
        reference_channel: Reference registration channel of the store.
        global_settings: Override text from the workflow params.
        row_settings: Override text from the samplesheet row.

    Returns:
        The validated settings.

    Raises:
        ValueError: On an unknown preset, a malformed or unknown override, a
            forbidden key, a value that fails validation or a ``crop_snap_px``
            below 64 (the factor the cropped pyramid levels must divide).
    """
    if preset not in PRESETS:
        raise ValueError(
            f"registration settings: unknown preset {preset!r}; "
            f"presets: {_preset_list()}"
        )
    glob = parse_settings_overrides(global_settings)
    row = parse_settings_overrides(row_settings)
    allowed = set(ManualRegistrationSettings.model_fields) - FORBIDDEN_OVERRIDES
    for source_name, scopes in (("global settings", glob), ("row settings", row)):
        keys = {key for scope in scopes.values() for key in scope}
        for key in sorted(keys & FORBIDDEN_OVERRIDES):
            raise ValueError(
                f"registration settings: {key!r} cannot be overridden; set pixel "
                "sizes / registration channels in the samplesheet and re-run prepare"
            )
        bad = sorted(keys - allowed)
        if bad:
            raise ValueError(
                f"registration settings: unknown key(s) {bad} in {source_name}; "
                f"allowed: {', '.join(sorted(allowed))}"
            )
    merged: dict[str, Any] = dict(PRESETS[preset])
    from_overrides: dict[str, Any] = {}
    for scope in (
        glob.get(None, {}),
        glob.get(preset, {}),
        row.get(None, {}),
        row.get(preset, {}),
    ):
        merged.update(scope)
        from_overrides.update(scope)
    merged["reference_pixel_size_um"] = reference_pixel_size_um
    merged["reference_channel"] = reference_channel
    try:
        settings = ManualRegistrationSettings.model_validate(merged)
    except ValidationError as exc:
        raise ValueError(
            f"registration settings for preset {preset!r} are invalid: {exc}"
        ) from exc
    if settings.crop_snap_px < CROP_SNAP_PX:
        raise ValueError(
            f"registration settings: crop_snap_px must be at least {CROP_SNAP_PX} "
            f"(got {settings.crop_snap_px}): the pyramid levels the registration "
            f"reads are cropped at factors that divide {CROP_SNAP_PX}, so a finer "
            "snap cannot keep their pixel grids aligned; larger powers of two are "
            "allowed"
        )
    if from_overrides:
        logger.info(
            "Settings overrides for preset %s: %s",
            preset,
            {key: getattr(settings, key) for key in from_overrides},
        )
    return settings


def registration_tuning(settings: ManualRegistrationSettings) -> RegistrationTuning:
    """Return the registration-core tuning that goes with a run's settings."""
    from merxen.image_registration import RegistrationTuning
    from merxen.nonrigid_registration import NonrigidTuning

    return RegistrationTuning(
        refinement_min_window_quality=settings.refinement_min_window_quality,
        refinement_require_moving_tissue=settings.refinement_require_moving_tissue,
        nonrigid=NonrigidTuning(
            highpass_sigma_um=settings.nonrigid_highpass_sigma_um,
            local_correlation_sigma_um=settings.nonrigid_local_correlation_sigma_um,
            local_residual_tiles=settings.nonrigid_local_residual_tiles,
            local_residual_tile_um=settings.nonrigid_local_residual_tile_um,
        ),
        qc_reference_label="Reference DAPI",
        qc_moving_label="Moving DAPI",
        qc_initial_matrix_label="Landmark matrix",
    )

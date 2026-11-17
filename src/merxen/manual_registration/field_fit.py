"""Robust landmark field of a pair: settings rules, fit, classes and the check payload.

The field itself lives in ``landmark_field`` and the image evidence in
``image_evidence``; this module joins them with the saved landmarks (the check of
``merxen manual-registration-check`` and, later, the register path).
"""

from __future__ import annotations

from typing import Literal

from merxen.manual_registration.config import ManualRegistrationSettings
from merxen.manual_registration.image_evidence import (
    ChannelRoles,
    resolve_channel_roles,
)
from merxen.manual_registration.schema import StoreAttrs

ROLES_EXAMPLE = "at8=AF568;map2=AF647;dapi=DAPI"


def resolve_store_roles(text: str, store: StoreAttrs) -> ChannelRoles | None:
    """Resolve a channel-roles text against the channels of a prepared store.

    Args:
        text: The roles text (C6 grammar); blank gives no roles.
        store: Attrs of the prepared store.

    Returns:
        The resolved roles, or None when the text is blank.

    Raises:
        ValueError: On a roles error; the message starts with ``channel roles: ``.
    """
    if not text.strip():
        return None

    def names(key: str) -> list[str]:
        record = store.images[key]
        return list(record.canonical_channel_names or record.channel_names)

    return resolve_channel_roles(
        text, reference_channels=names("reference"), moving_channels=names("moving")
    )


def validate_field_settings(
    settings: ManualRegistrationSettings,
    *,
    preset: str,
    roles: ChannelRoles | None,
    run_id: str,
    purpose: Literal["register", "check"],
) -> None:
    """Check the cross-field rules of the robust-field settings.

    Args:
        settings: The resolved settings.
        preset: The preset name.
        roles: The resolved channel roles, or None when none were given.
        run_id: Prefix of every message.
        purpose: ``register`` for a registration run, ``check`` for the landmark
            check (which always fits the robust field and never registers, so the
            refinement, flow and materialize settings do not matter).

    Raises:
        ValueError: With the ``[run_id]`` prefix when a rule is broken.
    """
    pfx = f"[{run_id}]"
    if purpose == "check":
        if (settings.field_image_check or settings.field_window_observations) and (
            roles is None
        ):
            raise ValueError(
                f"{pfx} the image check and tissue windows need channel roles: pass "
                f"--channel-roles '{ROLES_EXAMPLE}', or --no-image-check --no-windows "
                "for a landmarks-only check"
            )
        return
    is_field = settings.field_model == "robust_gp"
    if settings.materialize != "full" and not is_field:
        raise ValueError(
            f"{pfx} materialize={settings.materialize} (preview and none) is only "
            "for robust-field runs (field_model=robust_gp)"
        )
    if is_field and (settings.refine_affine or settings.nonrigid_enabled):
        raise ValueError(
            f"{pfx} cortex field runs skip the DAPI refinement and TV-L1 (plan D4); "
            "set refine_affine=false and nonrigid_enabled=false"
        )
    if preset == "cortex_serial" and roles is None:
        raise ValueError(
            f"{pfx} preset cortex_serial needs channel roles: set the samplesheet "
            f"column channel_roles, e.g. {ROLES_EXAMPLE}"
        )
    if (
        is_field
        and (settings.field_image_check or settings.field_window_observations)
        and roles is None
    ):
        raise ValueError(
            f"{pfx} the robust field with the image check or tissue windows needs "
            f"channel roles: set the samplesheet column channel_roles, e.g. "
            f"{ROLES_EXAMPLE}"
        )

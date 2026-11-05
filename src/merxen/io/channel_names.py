"""Channel-name normalisation for registration inputs.

Pure helpers (standard library only) that turn the channel metadata of an
image file (for example a Zeiss CZI) into the channel names used by manual
registration: detect uninformative names, recognise the nuclear stain, resolve
the registration channel and pick the reference channels the moving image also
has.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

NUCLEAR_STAIN_KEY: Final = "dapi"

_TRACK_SUFFIX = re.compile(r"[-_ ]t\d+$")
_ALEXA = re.compile(r"alexa\s*fluor|alexafluor|alexa")
_NON_ALNUM = re.compile(r"[\W_]+")
_NUCLEAR = re.compile(
    r"dapi|hoechst(33258|33342|34580)?|h(33258|33342|34580)|nucblue|bisbenzimide?"
)
_GENERIC = re.compile(
    r"(c|ch|chan|channel|chs|chs\d*t)?[:#_ -]?\d+(-?t\d+)?", re.IGNORECASE
)


def stain_key(text: str) -> str:
    """Return a comparison key for a channel or dye name.

    The key is case-, spacing- and punctuation-insensitive, drops a ZEN track
    suffix (``-T1``), abbreviates Alexa Fluor to ``af`` and maps the common
    nuclear dyes (DAPI, Hoechst, NucBlue, bisbenzimide) to ``"dapi"``.

    Args:
        text: A channel name, fluorophore or dye name.

    Returns:
        The normalised key.
    """
    key = unicodedata.normalize("NFKC", text).casefold().strip()
    key = _TRACK_SUFFIX.sub("", key)
    key = _ALEXA.sub("af", key)
    key = _NON_ALNUM.sub("", key)
    return NUCLEAR_STAIN_KEY if _NUCLEAR.fullmatch(key) else key


def is_generic_channel_name(text: str | None) -> bool:
    """Return whether a name carries no information (``None``, blank, ``Ch1``)."""
    if text is None or not text.strip():
        return True
    return _GENERIC.fullmatch(text.strip()) is not None


def is_nuclear_stain(text: str | None) -> bool:
    """Return whether a name denotes the nuclear stain (DAPI, Hoechst, ...)."""
    return (
        text is not None
        and bool(text.strip())
        and (stain_key(text) == NUCLEAR_STAIN_KEY)
    )


@dataclass(frozen=True)
class ChannelNameCandidates:
    """The name-like metadata fields of one channel (``None`` when absent)."""

    name: str | None
    fluor: str | None = None
    dye_name: str | None = None
    short_name: str | None = None

    def labelled(self) -> tuple[tuple[str, str | None], ...]:
        """Return ``(label, value)`` pairs in order of preference."""
        return (
            ("Name", self.name),
            ("Fluor", self.fluor),
            ("DyeName", self.dye_name),
            ("ShortName", self.short_name),
        )


@dataclass(frozen=True)
class ResolvedChannelNames:
    """Channel names with the metadata field each came from and any warnings."""

    names: tuple[str, ...]
    sources: tuple[str, ...]  # Name | Fluor | DyeName | ShortName | fallback | override
    warnings: tuple[str, ...]


def _describe(candidates: Sequence[ChannelNameCandidates]) -> str:
    parts = []
    for index, candidate in enumerate(candidates):
        fields = ", ".join(
            f"{label}={value!r}"
            for label, value in candidate.labelled()
            if value is not None
        )
        parts.append(f"{index}: {fields}")
    return "; ".join(parts)


def _resolve_override(
    candidates: Sequence[ChannelNameCandidates], override: Sequence[str]
) -> ResolvedChannelNames:
    names = [str(name).strip() for name in override]
    problem = None
    if len(names) != len(candidates):
        problem = (
            f"channel_names has {len(names)} entries but the image has "
            f"{len(candidates)} channels"
        )
    elif any(not name for name in names):
        problem = "channel_names must be non-empty strings"
    elif len(set(names)) != len(names):
        problem = f"channel_names must be unique, got {names}"
    if problem is not None:
        raise ValueError(f"{problem}; detected candidates: {_describe(candidates)}")
    return ResolvedChannelNames(
        names=tuple(names), sources=("override",) * len(names), warnings=()
    )


def resolve_channel_names(
    candidates: Sequence[ChannelNameCandidates],
    *,
    override: Sequence[str] | None = None,
) -> ResolvedChannelNames:
    """Choose one informative name per channel.

    With ``override`` the names are taken verbatim (stripped, checked for count,
    emptiness and uniqueness). Otherwise each channel takes the first
    non-generic of Name, Fluor, DyeName and ShortName (``c<index>`` when none);
    a single nuclear-stain channel is renamed ``DAPI``; repeated names get a
    numeric suffix.

    Args:
        candidates: Name-like metadata of every channel, in channel order.
        override: Verbatim channel names, or ``None`` to detect them.

    Returns:
        The names, their sources and human-readable warnings.

    Raises:
        ValueError: If there are no channels or an override is inconsistent.
    """
    if len(candidates) == 0:
        raise ValueError("an image needs at least one channel")
    if override is not None:
        return _resolve_override(candidates, override)

    names: list[str] = []
    sources: list[str] = []
    for index, candidate in enumerate(candidates):
        for label, value in candidate.labelled():
            if value is not None and not is_generic_channel_name(value):
                names.append(value.strip())
                sources.append(label)
                break
        else:
            names.append(f"c{index}")
            sources.append("fallback")

    warnings: list[str] = []
    nuclear = [
        index
        for index, candidate in enumerate(candidates)
        if any(is_nuclear_stain(value) for _, value in candidate.labelled())
    ]
    if len(nuclear) == 1:
        index = nuclear[0]
        if names[index] != "DAPI":
            warnings.append(
                f"channel {index} {names[index]!r} is a nuclear stain; named 'DAPI'"
            )
            names[index] = "DAPI"
    elif len(nuclear) > 1:
        warnings.append(
            f"channels {nuclear} all look like nuclear stains; names kept, "
            "set the registration channel explicitly"
        )
    elif len(candidates) == 1 and sources[0] == "fallback":
        names[0] = "DAPI"
        warnings.append(
            "the only channel has no informative name; assuming it is the "
            "nuclear stain 'DAPI'"
        )

    original = set(names)
    seen: set[str] = set()
    for index, name in enumerate(names):
        if name in seen:
            suffix = 2
            while f"{name}_{suffix}" in original or f"{name}_{suffix}" in seen:
                suffix += 1
            renamed = f"{name}_{suffix}"
            warnings.append(
                f"channel {index}: duplicate name {name!r} renamed {renamed!r}"
            )
            names[index] = renamed
            name = renamed
        seen.add(name)

    return ResolvedChannelNames(
        names=tuple(names), sources=tuple(sources), warnings=tuple(warnings)
    )


def resolve_registration_channel(names: Sequence[str], requested: str) -> str:
    """Map a requested registration channel onto one of ``names``.

    Matching order: exact, unique case-insensitive, then (for a DAPI-like
    request) the only nuclear-stain channel.

    Raises:
        ValueError: If the request is blank, ambiguous or matches nothing.
    """
    req = requested.strip()
    if not req:
        raise ValueError("registration channel must not be blank")
    if req in names:
        return req
    matches = [name for name in names if name.casefold() == req.casefold()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"registration channel {requested!r} matches several channels "
            f"ignoring case: {matches}"
        )
    if stain_key(req) == NUCLEAR_STAIN_KEY:
        nuclear = [name for name in names if is_nuclear_stain(name)]
        if len(nuclear) == 1:
            return nuclear[0]
    raise ValueError(
        f"registration channel {requested!r} is not one of the channels "
        f"{list(names)}; set the registration channel or the channel names in "
        "the samplesheet"
    )


def shared_channel_subset(
    reference: Sequence[str],
    moving: Sequence[str],
    *,
    reference_registration_channel: str,
) -> list[int]:
    """Return the reference channel indices to keep.

    A reference channel is kept when it is the registration channel or when the
    moving image has a channel with the same stain key.

    Raises:
        ValueError: If the registration channel is not a reference channel.
    """
    if reference_registration_channel not in reference:
        raise ValueError(
            f"reference registration channel {reference_registration_channel!r} "
            f"is not one of {list(reference)}"
        )
    keys = {stain_key(name) for name in moving}
    return [
        index
        for index, name in enumerate(reference)
        if name == reference_registration_channel or stain_key(name) in keys
    ]

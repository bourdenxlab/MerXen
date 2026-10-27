"""Tests for channel-name normalisation of registration inputs."""

from __future__ import annotations

import pytest

from merxen.io.channel_names import (
    ChannelNameCandidates,
    is_generic_channel_name,
    is_nuclear_stain,
    resolve_channel_names,
    resolve_registration_channel,
    shared_channel_subset,
    stain_key,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Alexa Fluor 488", "af488"),
        ("AF488", "af488"),
        ("AF 488-T1", "af488"),
        ("AlexaFluor647", "af647"),
        ("Alexa647", "af647"),
        ("  AF568 ", "af568"),
        ("Hoechst 33342", "dapi"),
        ("H33342", "dapi"),
        ("NucBlue", "dapi"),
        ("Bisbenzimid", "dapi"),
        ("dapi-T2", "dapi"),
        ("DAPI", "dapi"),
        ("ATP1A1/CD45/E-Cadherin", "atp1a1cd45ecadherin"),
        ("FITC", "fitc"),
        ("Cy5_T1", "cy5"),
    ],
)
def test_stain_key_table(text: str, expected: str) -> None:
    assert stain_key(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        None,
        "",
        "  ",
        "1",
        "C1",
        "Ch 2",
        "ch-3",
        "Channel:1",
        "Chan#4",
        "ChS1-T1",
        "ChS1T1",
        "c0",
        "Ch1-T2",
        "channel 12",
    ],
)
def test_generic_channel_names_are_detected(text: str | None) -> None:
    assert is_generic_channel_name(text)


@pytest.mark.parametrize("text", ["T1", "AF568", "DAPI", "Hoechst 33342"])
def test_informative_channel_names_are_not_generic(text: str) -> None:
    assert not is_generic_channel_name(text)


def test_is_nuclear_stain() -> None:
    assert is_nuclear_stain("Hoechst 33342")
    assert is_nuclear_stain("dapi")
    assert not is_nuclear_stain("AF568")
    assert not is_nuclear_stain(None)
    assert not is_nuclear_stain("  ")


def test_czi_metadata_of_the_zeiss_scans_resolves_to_the_names() -> None:
    candidates = [
        ChannelNameCandidates("DAPI", "DAPI", None, "DAPI"),
        ChannelNameCandidates("AF568", "Alexa Fluor 568", None, "AF568"),
        ChannelNameCandidates("AF647", "Alexa Fluor 647", None, "AF647"),
    ]
    resolved = resolve_channel_names(candidates)
    assert resolved.names == ("DAPI", "AF568", "AF647")
    assert resolved.sources == ("Name",) * 3
    assert resolved.warnings == ()
    assert shared_channel_subset(
        resolved.names, resolved.names, reference_registration_channel="DAPI"
    ) == [0, 1, 2]


def test_generic_name_falls_back_to_fluor() -> None:
    resolved = resolve_channel_names(
        [
            ChannelNameCandidates("Ch1", "Hoechst 33342"),
            ChannelNameCandidates("Ch2", "Alexa Fluor 568"),
        ]
    )
    assert resolved.names == ("DAPI", "Alexa Fluor 568")
    assert resolved.sources == ("Fluor", "Fluor")
    assert len(resolved.warnings) == 1
    assert "Hoechst 33342" in resolved.warnings[0]


def test_all_generic_names_fall_back_to_indices() -> None:
    resolved = resolve_channel_names(
        [ChannelNameCandidates(f"Ch{i}") for i in range(1, 4)]
    )
    assert resolved.names == ("c0", "c1", "c2")
    assert resolved.sources == ("fallback",) * 3
    assert resolved.warnings == ()
    with pytest.raises(ValueError, match="c0"):
        resolve_registration_channel(resolved.names, "DAPI")


def test_single_generic_channel_is_assumed_to_be_dapi() -> None:
    resolved = resolve_channel_names([ChannelNameCandidates(None)])
    assert resolved.names == ("DAPI",)
    assert len(resolved.warnings) == 1
    assert "only channel" in resolved.warnings[0]


def test_no_channels_is_an_error() -> None:
    with pytest.raises(ValueError, match="at least one channel"):
        resolve_channel_names([])


def test_two_nuclear_channels_keep_their_names() -> None:
    resolved = resolve_channel_names(
        [ChannelNameCandidates("DAPI"), ChannelNameCandidates("Hoechst 33342")]
    )
    assert resolved.names == ("DAPI", "Hoechst 33342")
    assert len(resolved.warnings) == 1
    assert "nuclear stains" in resolved.warnings[0]


def test_duplicate_names_get_the_smallest_free_suffix() -> None:
    resolved = resolve_channel_names(
        [
            ChannelNameCandidates("AF568"),
            ChannelNameCandidates("AF568"),
            ChannelNameCandidates("AF568_2"),
        ]
    )
    assert resolved.names == ("AF568", "AF568_3", "AF568_2")
    assert len(resolved.warnings) == 1


def test_override_is_verbatim_and_not_canonicalised() -> None:
    candidates = [ChannelNameCandidates(f"x{i}") for i in range(3)]
    resolved = resolve_channel_names(candidates, override=["a", "b", "c"])
    assert resolved.names == ("a", "b", "c")
    assert resolved.sources == ("override",) * 3
    assert resolved.warnings == ()
    kept = resolve_channel_names(candidates, override=["Hoechst", "x", "y"])
    assert kept.names == ("Hoechst", "x", "y")


@pytest.mark.parametrize(
    "override", [["a", "b"], ["a", "b", ""], ["a", "b", "a"], ["a", "b", "c", "d"]]
)
def test_bad_override_lists_the_detected_candidates(override: list[str]) -> None:
    candidates = [ChannelNameCandidates("DAPI", fluor="DAPI")] + [
        ChannelNameCandidates(f"x{i}") for i in range(2)
    ]
    with pytest.raises(ValueError, match="detected candidates") as excinfo:
        resolve_channel_names(candidates, override=override)
    assert "0: Name='DAPI', Fluor='DAPI'" in str(excinfo.value)


def test_resolve_registration_channel_matching_order() -> None:
    assert resolve_registration_channel(["DAPI", "AF568"], "DAPI") == "DAPI"
    assert resolve_registration_channel(["DAPI", "AF568"], "dapi") == "DAPI"
    assert (
        resolve_registration_channel(["Hoechst 33342", "AF568"], "DAPI")
        == "Hoechst 33342"
    )
    with pytest.raises(ValueError, match="several channels"):
        resolve_registration_channel(["DAPI", "dapi"], "Dapi")
    with pytest.raises(ValueError, match="must not be blank"):
        resolve_registration_channel(["DAPI"], "  ")
    with pytest.raises(ValueError, match="AF568"):
        resolve_registration_channel(["AF568", "AF647"], "DAPI")


def test_shared_channel_subset() -> None:
    assert shared_channel_subset(
        ["DAPI", "AF488", "AF568", "AF647"],
        ["Hoechst 33342", "Alexa Fluor 568"],
        reference_registration_channel="DAPI",
    ) == [0, 2]
    assert shared_channel_subset(
        ["DAPI", "AF647"], ["AF568"], reference_registration_channel="DAPI"
    ) == [0]
    with pytest.raises(ValueError, match="reference registration channel"):
        shared_channel_subset(
            ["AF568"], ["AF568"], reference_registration_channel="DAPI"
        )

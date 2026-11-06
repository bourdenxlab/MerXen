"""The clustered table-key suffix and its inverse (plan §4.8)."""

from __future__ import annotations

import pytest

from merxen.clustering.map_first import check_clustered_table_target
from merxen.table_keys import clustered_table_key, clustered_table_key_suffix

MAP_FIRST_UNS = {
    "merxen_hierarchical_clustering": {
        "mode": "map_first",
        "table_key_suffix": "mapfirst",
    }
}
FLIPPED_UNS = {
    "merxen_hierarchical_clustering": {"mode": "map_first", "table_key_suffix": ""}
}
LEGACY_UNS = {"merxen_hierarchical_clustering": {"mode": "legacy"}}


@pytest.mark.parametrize("suffix", ["", "mapfirst", "trial_2"])
@pytest.mark.parametrize(
    ("source", "segmentation"),
    [
        ("table_MOSAIK_proseg_hybrid", "proseg_hybrid"),
        ("table_MOSAIK_proseg", "reseg"),
        ("table_original", "original_seg"),
    ],
)
def test_the_suffix_parser_inverts_the_key_builder(
    source: str, segmentation: str, suffix: str
) -> None:
    key = clustered_table_key(source, segmentation, suffix)
    assert clustered_table_key_suffix(key) == suffix


@pytest.mark.parametrize(
    "key",
    [
        "table",
        "table_MOSAIK_proseg_hybrid",
        "_clustering_squidpy",
        "x_clustering_squidpyX",
    ],
)
def test_keys_without_the_clustered_tag_have_no_suffix(key: str) -> None:
    assert clustered_table_key_suffix(key) is None


def test_a_map_first_table_goes_only_to_its_suffixed_key() -> None:
    suffixed = "table_MOSAIK_proseg_hybrid_clustering_squidpy_mapfirst"
    unsuffixed = "table_MOSAIK_proseg_hybrid_clustering_squidpy"
    check_clustered_table_target(MAP_FIRST_UNS, suffixed, consumer="T")
    check_clustered_table_target(FLIPPED_UNS, unsuffixed, consumer="T")
    for uns, key in (
        (MAP_FIRST_UNS, unsuffixed),
        (MAP_FIRST_UNS, "table_MOSAIK_proseg_hybrid_clustering_squidpy_trial"),
        (MAP_FIRST_UNS, "table"),
        (FLIPPED_UNS, suffixed),
    ):
        with pytest.raises(ValueError, match="map_first table built for"):
            check_clustered_table_target(uns, key, consumer="T")


def test_a_legacy_table_never_goes_to_a_suffixed_key() -> None:
    for uns in ({}, LEGACY_UNS):
        check_clustered_table_target(
            uns, "table_MOSAIK_proseg_hybrid_clustering_squidpy", consumer="T"
        )
        # Keys outside the clustered naming stay allowed (legacy behaviour).
        check_clustered_table_target(uns, "table", consumer="T")
        with pytest.raises(ValueError, match="no map_first record"):
            check_clustered_table_target(
                uns,
                "table_MOSAIK_proseg_hybrid_clustering_squidpy_mapfirst",
                consumer="T",
            )

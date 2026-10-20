"""Golden-value guard for the viewer cache format.

These assertions pin the exact element keys, attribute names, versions, and
marker schemas the napari comparison viewer trusts. If the viewer's format
changes (e.g. it bumps ``DERIVED_CACHE_VERSION`` or renames a marker field),
these tests fail -- forcing the reimplementation in ``merxen.viewer_cache.format``
to be updated in lockstep before a drifted cache is silently rebuilt by the
viewer. The golden values were captured from
``napari-compare-xenium-merscope`` (utils.py / viewer.py).
"""

from __future__ import annotations

from merxen.viewer_cache import format as fmt


def test_prefixes_attrs_and_versions_are_pinned() -> None:
    assert fmt.DERIVED_CACHE_PREFIX == "_napari_compare_"
    assert fmt.DERIVED_CACHE_ATTR == "napari_compare_derived_cache"
    assert fmt.LABEL_CACHE_ATTR == "napari_compare_label_cache"
    assert fmt.VIEWER_DERIVED_CACHE_VERSION == 2
    assert fmt.LABEL_CACHE_VERSION == 2
    # Viewer: SYNTHETIC_IMAGE_PYRAMID_MIN_SIZE / LABEL_OUTLINE_PYRAMID_MIN_SIZE.
    assert fmt.PYRAMID_MIN_SIZE == 1024
    assert fmt.PYRAMID_MAX_LEVELS == 10
    # Viewer: RASTER_DISPLAY_TILE_SIZE / OUTLINE_COVERAGE_MAX.
    assert fmt.PYRAMID_TILE == 1024
    assert fmt.OUTLINE_COVERAGE_MAX == 255
    assert fmt.OUTLINE_PYRAMID_MODE == "coverage_mean_v1"
    assert fmt.OUTLINE_SOURCE == "coverage_mean"


def test_config_default_min_size_tracks_the_format_constant() -> None:
    """ViewerCacheConfig cannot import PYRAMID_MIN_SIZE (circular), so pin it here."""
    from merxen.config import ViewerCacheConfig

    assert ViewerCacheConfig.model_fields["min_size"].default == fmt.PYRAMID_MIN_SIZE, (
        "config.py's literal min_size default drifted from PYRAMID_MIN_SIZE"
    )


def test_cache_keys_match_viewer_golden_values() -> None:
    assert (
        fmt.derived_label_pyramid_cache_key("MOSAIK_proseg_labels", 4)
        == "_napari_compare_labelpyr__MOSAIK_proseg_labels__ds4"
    )
    assert (
        fmt.derived_outline_cache_key("MOSAIK_proseg_labels", 1)
        == "_napari_compare_outline__MOSAIK_proseg_labels__w1"
    )
    assert (
        fmt.derived_image_pyramid_cache_key("MERSCOPE_z_projection", 4)
        == "_napari_compare_imgpyr__MERSCOPE_z_projection__ds4"
    )


def test_safe_cache_token_matches_viewer_hashing() -> None:
    # Clean tokens pass through unchanged.
    assert fmt.safe_cache_token("MOSAIK_proseg_labels") == "MOSAIK_proseg_labels"
    # Unsafe characters collapse to "_" and a blake2s digest suffix is appended
    # (this exact digest was produced by the viewer's _safe_cache_token).
    assert (
        fmt.safe_cache_token("weird/key with spaces")
        == "weird_key_with_spaces__h6d7b929030"
    )


def test_label_key_for_shape_key() -> None:
    assert fmt.label_key_for_shape_key("MOSAIK_proseg", set()) == "MOSAIK_proseg_labels"
    # An already-present label element of the same name is reused verbatim.
    assert (
        fmt.label_key_for_shape_key("custom_labels", {"custom_labels"})
        == "custom_labels"
    )


def test_marker_schemas_match_viewer() -> None:
    assert fmt.label_cache_marker(
        source_shape_key="MOSAIK_proseg", shape=(100, 200), chunks=(2048, 2048)
    ) == {
        "version": 2,
        "complete": True,
        "source_shape_key": "MOSAIK_proseg",
        "shape": [100, 200],
        "chunks": [2048, 2048],
    }
    assert fmt.label_pyramid_marker(
        source_label_key="MOSAIK_proseg_labels", downsample=4, min_size=1024, levels=2
    ) == {
        "version": 2,
        "complete": True,
        "kind": "label_pyramid",
        "source_label_key": "MOSAIK_proseg_labels",
        "downsample": 4,
        "min_size": 1024,
        "levels": 2,
    }
    assert fmt.image_pyramid_marker(
        source_image_key="MERSCOPE_z_projection", downsample=4, min_size=1024, levels=2
    ) == {
        "version": 2,
        "complete": True,
        "kind": "image_pyramid",
        "source_image_key": "MERSCOPE_z_projection",
        "downsample": 4,
        "min_size": 1024,
        "levels": 2,
    }
    assert fmt.outline_marker(
        source_label_key="MOSAIK_proseg_labels",
        width=1,
        levels=3,
        source_shapes=[[160, 160]],
    ) == {
        "version": 2,
        "complete": True,
        "kind": "label_outline",
        "source_label_key": "MOSAIK_proseg_labels",
        "width": 1,
        "min_size": 1024,
        "tile_size": 1024,
        "pyramid_mode": "coverage_mean_v1",
        "value_max": 255,
        "source": "coverage_mean",
        "levels": 3,
        "source_shapes": [[160, 160]],
    }


def test_outline_marker_satisfies_viewer_expected_keys() -> None:
    """The viewer discards a cache unless EVERY key in its ``expected`` dict matches.

    Captured verbatim from ``_ensure_label_outline_cache`` in the viewer. A marker
    that drops one of these (as the pre-coverage_mean_v1 marker did) makes the
    viewer rebuild the pyramid it was handed, which is the whole failure this
    module exists to prevent.
    """
    marker = fmt.outline_marker(
        source_label_key="MOSAIK_proseg_labels",
        width=1,
        levels=8,
        source_shapes=[[113164, 98317]],
    )
    expected = {
        "kind": "label_outline",
        "source_label_key": "MOSAIK_proseg_labels",
        "width": 1,
        "min_size": 1024,
        "tile_size": 1024,
        "pyramid_mode": "coverage_mean_v1",
        "value_max": 255,
    }
    assert marker["version"] == fmt.VIEWER_DERIVED_CACHE_VERSION
    assert marker["complete"] is True
    for key, value in expected.items():
        assert marker[key] == value, (
            f"viewer compares {key!r}; marker would be rejected"
        )


def test_pyramid_markers_satisfy_viewer_expected_keys() -> None:
    """Same guard for the image and label pyramid ``expected`` dicts."""
    image = fmt.image_pyramid_marker(
        source_image_key="MERSCOPE_z_projection", downsample=4, min_size=1024, levels=3
    )
    for key, value in {
        "kind": "image_pyramid",
        "source_image_key": "MERSCOPE_z_projection",
        "downsample": 4,
        "min_size": 1024,
    }.items():
        assert image[key] == value, f"viewer compares {key!r}; marker would be rejected"

    label = fmt.label_pyramid_marker(
        source_label_key="MOSAIK_proseg_labels", downsample=4, min_size=1024, levels=3
    )
    for key, value in {
        "kind": "label_pyramid",
        "source_label_key": "MOSAIK_proseg_labels",
        "downsample": 4,
        "min_size": 1024,
    }.items():
        assert label[key] == value, f"viewer compares {key!r}; marker would be rejected"

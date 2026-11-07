"""Tests for the optional samplesheet columns (annotation.samplesheet_columns)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from merxen.annotation.samplesheet_columns import (
    ANATOMICAL_REGION_COLUMN,
    MOUSE_SECTION_REGIONS_COLUMN,
    OPTIONAL_ANNOTATION_COLUMNS,
    OptionalAnnotationColumns,
    effective_section_regions,
    parse_anatomical_region,
    parse_mouse_section_regions,
    parse_optional_columns,
    section_regions_by_sample,
    split_section_regions,
)
from merxen.config import ClusteringSquidpySampleConfig
from merxen.io.samplesheet import parse_samplesheet

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_missing_and_blank_columns_inherit_the_params() -> None:
    assert parse_optional_columns({}) == OptionalAnnotationColumns()
    blank = {ANATOMICAL_REGION_COLUMN: " ", MOUSE_SECTION_REGIONS_COLUMN: ""}
    assert parse_optional_columns(blank) == OptionalAnnotationColumns()
    assert parse_optional_columns({ANATOMICAL_REGION_COLUMN: None}).as_dict() == {
        "anatomical_region": None,
        "mouse_section_regions": None,
        "clustering_squidpy_mode": None,
    }
    assert OPTIONAL_ANNOTATION_COLUMNS == (
        "anatomical_region",
        "mouse_section_regions",
        "clustering_squidpy_mode",
    )


def test_a_rows_clustering_mode_is_checked_and_normalised() -> None:
    """The per-row mode column (pre-registration §20 D15)."""
    parsed = parse_optional_columns({"clustering_squidpy_mode": " Legacy "})
    assert parsed.clustering_squidpy_mode == "legacy"
    assert parse_optional_columns({"clustering_squidpy_mode": " "}) == (
        OptionalAnnotationColumns()
    )
    with pytest.raises(ValueError, match="invalid clustering_squidpy_mode"):
        parse_optional_columns({"clustering_squidpy_mode": "map-first"})


def test_example_samplesheet_rows_parse_unchanged() -> None:
    path = REPO_ROOT / "workflows" / "samplesheet.example.csv"
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    for row in rows:
        assert parse_optional_columns(row) == OptionalAnnotationColumns()


def test_values_are_normalised() -> None:
    parsed = parse_optional_columns(
        {
            ANATOMICAL_REGION_COLUMN: " Frontal cortex ",
            MOUSE_SECTION_REGIONS_COLUMN: "isocortex;hpf; TH",
        }
    )
    assert parsed.anatomical_region == "frontal_cortex"
    assert parsed.mouse_section_regions == "Isocortex;HPF;TH"
    assert parse_anatomical_region("frontal-cortex") == "frontal_cortex"
    assert parse_mouse_section_regions("AUTO") == "auto"
    assert parse_mouse_section_regions("none") == "none"


@pytest.mark.parametrize("value", ["frontal/cortex", "cortex (BA9)"])
def test_invalid_anatomical_region(value: str) -> None:
    with pytest.raises(ValueError, match="invalid anatomical_region"):
        parse_anatomical_region(value)


def test_invalid_section_regions() -> None:
    with pytest.raises(ValueError, match="unknown mouse section region"):
        parse_optional_columns({MOUSE_SECTION_REGIONS_COLUMN: "Isocortex;Striatum"})


def test_split_section_regions() -> None:
    assert split_section_regions("auto") is None
    assert split_section_regions("NONE") == []
    assert split_section_regions("Isocortex;HPF") == ["Isocortex", "HPF"]
    assert split_section_regions("th") == ["TH"]


def test_samplesheet_column_reaches_the_sample_config(tmp_path: Path) -> None:
    sheet = tmp_path / "samplesheet.csv"
    with sheet.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pair_id", "mouse_section_regions"])
        writer.writeheader()
        writer.writerow({"pair_id": "AG", "mouse_section_regions": "isocortex;hpf"})
        writer.writerow({"pair_id": "VZ", "mouse_section_regions": " "})
        writer.writerow({"pair_id": "NO", "mouse_section_regions": "None"})
    pairs = {pair.pair_id: pair for pair in parse_samplesheet(sheet)}
    assert pairs["AG"].mouse_section_regions == "Isocortex;HPF"
    assert pairs["VZ"].mouse_section_regions is None
    assert pairs["NO"].mouse_section_regions == "none"

    samples = [
        ClusteringSquidpySampleConfig(
            sample_id=f"{pair_id}_MERSCOPE",
            platform="MERSCOPE",
            zarr_path=tmp_path / "x.zarr",
            mouse_section_regions=pair.mouse_section_regions,
        ).model_dump(mode="json")
        for pair_id, pair in pairs.items()
    ]
    by_sample = section_regions_by_sample(samples)
    assert by_sample == {"AG_MERSCOPE": "Isocortex;HPF", "NO_MERSCOPE": "none"}
    assert effective_section_regions(by_sample.get("VZ_MERSCOPE"), "auto") == "auto"
    assert effective_section_regions("th", "none") == "TH"
    with pytest.raises(ValueError, match="unknown mouse section region"):
        section_regions_by_sample([{"sample_id": "S", "mouse_section_regions": "X"}])
    with pytest.raises(ValueError, match="no sample_id"):
        section_regions_by_sample([{"mouse_section_regions": "auto"}])
    with pytest.raises(ValueError):
        ClusteringSquidpySampleConfig(
            sample_id="S",
            platform="MERSCOPE",
            zarr_path=tmp_path / "x.zarr",
            mouse_section_regions="Striatum",
        )

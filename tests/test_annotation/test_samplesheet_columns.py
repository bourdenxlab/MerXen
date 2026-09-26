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
    parse_anatomical_region,
    parse_mouse_section_regions,
    parse_optional_columns,
    split_section_regions,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_missing_and_blank_columns_inherit_the_params() -> None:
    assert parse_optional_columns({}) == OptionalAnnotationColumns()
    blank = {ANATOMICAL_REGION_COLUMN: " ", MOUSE_SECTION_REGIONS_COLUMN: ""}
    assert parse_optional_columns(blank) == OptionalAnnotationColumns()
    assert parse_optional_columns({ANATOMICAL_REGION_COLUMN: None}).as_dict() == {
        "anatomical_region": None,
        "mouse_section_regions": None,
    }
    assert OPTIONAL_ANNOTATION_COLUMNS == ("anatomical_region", "mouse_section_regions")


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

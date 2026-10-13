"""The ``acceptance_metrics.json`` contract (``report_model``; M8 and M9 read it).

If one of these field sets changes, bump ``REPORT_SCHEMA_VERSION`` and update
the contract tables in ``docs/stages/annotation.md`` in the same commit.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import get_args

from merxen.annotation.report_model import (
    REPORT_SCHEMA_VERSION,
    AcceptanceMetrics,
    CriterionCoverage,
    ItemStatus,
    MetricRecord,
    MetricStatus,
)

DOCS = Path(__file__).resolve().parents[2] / "docs" / "stages" / "annotation.md"
DOCUMENTED_TOP_LEVEL = {
    "schema_version",
    "kind",
    "species",
    "pair_id",
    "segmentation",
    "metrics",
    "criteria",
    "datasets",
    "items",
    "provenance",
}
DOCUMENTED_RECORD_FIELDS = {
    "criterion",
    "name",
    "scope",
    "sample_id",
    "platform",
    "region",
    "level",
    "kind",
    "group",
    "value",
    "ci_low",
    "ci_high",
    "n",
    "definition",
    "source",
    "status",
    "note",
}


def _contract_section() -> str:
    text = DOCS.read_text()
    start = text.index("**The `acceptance_metrics.json` contract**")
    end = text.index("**Definitions chosen here", start)
    return text[start:end]


def test_the_document_and_record_fields_match_the_documented_contract() -> None:
    # Changing either set is a schema change: bump REPORT_SCHEMA_VERSION.
    assert REPORT_SCHEMA_VERSION == 1
    assert set(AcceptanceMetrics.model_fields) == DOCUMENTED_TOP_LEVEL
    assert set(MetricRecord.model_fields) == DOCUMENTED_RECORD_FIELDS
    assert set(CriterionCoverage.model_fields) == {
        "criterion",
        "source",
        "description",
        "n_records",
    }
    assert set(get_args(MetricStatus)) == {"measured", "not_available", "withheld"}
    assert set(get_args(ItemStatus)) == {
        "ok",
        "partial",
        "not_available",
        "not_applicable",
        "failed",
    }


def test_the_docs_table_names_every_field_and_status() -> None:
    section = _contract_section()
    names = set(re.findall(r"^\| `([a-z_]+)`", section, re.M))
    names |= {"ci_high"} if "`ci_low`, `ci_high`" in section else set()
    assert names >= DOCUMENTED_TOP_LEVEL
    assert names >= DOCUMENTED_RECORD_FIELDS
    for status in ("measured", "not_available", "withheld"):
        assert f"`{status}`" in section, status
    assert f"`REPORT_SCHEMA_VERSION` {REPORT_SCHEMA_VERSION}" in section

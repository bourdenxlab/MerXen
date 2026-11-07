"""Tests for the M3c no-promotion script's sample names (privacy rules R2, R5)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from merxen.annotation.diagnostics import load_validated_panels

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts" / "acceptance"


def _load() -> ModuleType:
    name = "m3c_no_promotion"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


M = _load()


def _ms1_hash() -> str:
    record = next(
        item
        for item in load_validated_panels().records
        if item.family_id == "mouse_ms1"
    )
    return str(record.panel_hash)


def _manifest(panel_hash: str) -> dict[str, Any]:
    return {
        "species": "mouse",
        "reference_id": "wmb_panel",
        "build_hash": "b" * 64,
        "panel": {"panel_hash": panel_hash, "n_genes": 500},
        "built_from_panel": {"platforms": ["MERSCOPE"]},
        "builder_output": {
            "panel_coverage": {
                "n_panel_genes": 500,
                "n_query_genes_used": 500,
                "root_markers": 40,
            }
        },
    }


def test_trust_records_a_listed_family_by_its_display_name(
    fake_aliases: Path,
) -> None:
    """R5: the trust file names the family as before the aliases."""
    from test_privacy.synthetic import MOUSE

    record = M.bundle_trust(_manifest(_ms1_hash()))
    assert record["family_id"] == f"mouse_{MOUSE}"
    assert record["family_basis"] == "listed"


def test_trust_keeps_the_alias_with_the_table_disabled() -> None:
    assert M.bundle_trust(_manifest(_ms1_hash()))["family_id"] == "mouse_ms1"


def _row(build_hash: str, family_id: str, state: str) -> dict[str, Any]:
    return {
        "store": "/path/to/store",
        "reference_id": "wmb_panel",
        "build_hash": build_hash,
        "family_id": family_id,
        "family_basis": "listed",
        "state": state,
        "validation_basis": "real_data",
        "validated_max_level": "subclass",
        "resolvability_version": 7,
    }


def test_check_matches_a_recorded_real_family_with_its_alias(
    tmp_path: Path, fake_aliases: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R2: a baseline recorded with the real family name still matches."""
    from test_privacy.synthetic import MOUSE

    monkeypatch.setattr(
        M,
        "check_tables",
        lambda: {"validated_panels.csv": {"identical": True, "sha256_m3c": "0" * 64}},
    )
    monkeypatch.setattr(
        M, "run_property_tests", lambda out_dir: {"passed": True, "summary": "ok"}
    )
    prime = [
        _row(str(index) * 64, family, "provisional")
        for index, family in enumerate(M.PRIME_FAMILIES)
    ]
    baseline = tmp_path / "base.json"
    baseline.write_text(
        json.dumps(
            {"bundles": [_row("a" * 64, f"mouse_{MOUSE}", "provisional"), *prime]}
        )
    )
    new_prime = [
        _row(str(index + 5) * 64, family, "provisional")
        for index, family in enumerate(M.PRIME_FAMILIES)
    ]
    current = tmp_path / "current.json"
    current.write_text(
        json.dumps(
            {
                "bundles": [
                    _row("a" * 64, "mouse_ms1", "provisional"),
                    _row("c" * 64, "mouse_ms1", "provisional"),
                    *prime,
                    *new_prime,
                ]
            }
        )
    )
    out = tmp_path / "out"
    assert M.command_check(baseline, current, out) == 0
    report = json.loads((out / "NO_PROMOTION.json").read_text())
    ms1 = [item for item in report["new_bundles"] if item["new_bundle"] == "c" * 12]
    assert ms1 and ms1[0]["ok"] and ms1[0]["states_83e81e3"] == ["provisional"]
    assert report["passes"]

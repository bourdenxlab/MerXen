"""Redistributed notices retain their bytes and required packages cannot vanish."""

from __future__ import annotations

import hashlib
import json
import runpy
from importlib import metadata
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
COLLECT = runpy.run_path(str(ROOT / "scripts/collect_gaston_licenses.py"))[
    "collect_licenses"
]


def _distribution(root: Path, name: str) -> metadata.PathDistribution:
    info = root / f"{name}-1.0.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(f"Name: {name}\nVersion: 1.0\nLicense: BSD\n")
    (info / "licenses").mkdir()
    # A nested upstream notice need not have LICENSE as its basename.
    (info / "licenses" / "upstream.txt").write_bytes(b"copyright\r\nconditions\r\n")
    (info / "RECORD").write_text(
        f"{info.name}/licenses/upstream.txt,,\n{info.name}/METADATA,,\n"
    )
    return metadata.PathDistribution(info)


def test_collection_preserves_notices_and_inventory(tmp_path: Path) -> None:
    distributions = [
        _distribution(tmp_path, name) for name in ("gaston-spatial", "glmpca", "kneed")
    ]
    output = tmp_path / "output"
    with patch("importlib.metadata.distributions", return_value=distributions):
        COLLECT(output)
    inventory = json.loads((output / "inventory.json").read_text())
    assert {row["name"] for row in inventory} == {"gaston-spatial", "glmpca", "kneed"}
    for row in inventory:
        package = output / f"{row['name']}-1.0"
        assert (package / row["notices"][0]).read_bytes() == (
            b"copyright\r\nconditions\r\n"
        )
        assert f"Name: {row['name']}" in (package / "METADATA").read_text()


def test_collection_refuses_missing_required_notice(tmp_path: Path) -> None:
    distribution = _distribution(tmp_path, "gaston-spatial")
    with (
        patch("importlib.metadata.distributions", return_value=[distribution]),
        pytest.raises(RuntimeError, match="glmpca.*kneed"),
    ):
        COLLECT(tmp_path / "output")
    assert not (tmp_path / "output" / "inventory.json").exists()


def test_glmpca_source_is_the_exact_upstream_release() -> None:
    archive = ROOT / "licenses/sources/glmpca-0.1.0.tar.gz"
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == (
        "02267dc81b1417235d95152a2395d928bff3b503edc5b8d364ebbd1045575118"
    )

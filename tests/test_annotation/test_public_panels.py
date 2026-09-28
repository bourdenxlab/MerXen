"""Tests for the pinned public panel gene lists (plan §8.8; M3b)."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from click.testing import CliRunner

from merxen.annotation import public_panels
from merxen.annotation.public_panels import (
    PUBLIC_PANEL_LISTS,
    PublicPanelError,
    PublicPanelList,
    fetch_public_panel,
    parse_panel_table,
)
from merxen.cli import main as cli_main


def vendor_table(n_genes: int, *, bom: bool = False) -> bytes:
    rows = ["Genes,Ensembl_ID,Num_Probesets,Codewords,Annotation"]
    rows += [f"Gene{index},ENSMUSG{index:011d},8,1,x" for index in range(n_genes)]
    text = "\n".join(rows) + "\n"
    return (("﻿" if bom else "") + text).encode("utf-8")


def pinned(data: bytes, n_genes: int) -> PublicPanelList:
    return PublicPanelList(
        key="test_panel",
        title="Test panel",
        species="mouse",
        platform="XENIUM",
        url="https://example.invalid/test.csv",
        file_name="test.csv",
        size=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        n_genes=n_genes,
        symbol_column="Genes",
        id_column="Ensembl_ID",
    )


def test_the_four_plan_panels_are_pinned() -> None:
    assert set(PUBLIC_PANEL_LISTS) == {
        "xenium_prime_5k_human",
        "xenium_prime_5k_mouse",
        "xenium_human_brain_v1",
        "xenium_mouse_brain_v1",
    }
    counts = {key: item.n_genes for key, item in PUBLIC_PANEL_LISTS.items()}
    assert counts == {
        "xenium_prime_5k_human": 5001,
        "xenium_prime_5k_mouse": 5006,
        "xenium_human_brain_v1": 266,
        "xenium_mouse_brain_v1": 248,
    }
    for item in PUBLIC_PANEL_LISTS.values():
        assert item.url.startswith("https://cdn.10xgenomics.com/")
        assert len(item.sha256) == 64 and item.size > 0
        assert item.species in ("human", "mouse") and item.platform == "XENIUM"


def test_parse_drops_the_byte_order_mark_and_checks_the_count() -> None:
    data = vendor_table(3, bom=True)
    item = pinned(data, 3)
    assert parse_panel_table(data, item) == [
        ("Gene0", "ENSMUSG00000000000"),
        ("Gene1", "ENSMUSG00000000001"),
        ("Gene2", "ENSMUSG00000000002"),
    ]
    with pytest.raises(PublicPanelError, match="pinned 4"):
        parse_panel_table(data, replace(item, n_genes=4))
    with pytest.raises(PublicPanelError, match="no column"):
        parse_panel_table(data, replace(item, id_column="gene_id"))


def test_fetch_verifies_writes_a_gene_table_and_reuses_the_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = vendor_table(5)
    item = pinned(data, 5)
    monkeypatch.setitem(PUBLIC_PANEL_LISTS, item.key, item)
    calls: list[str] = []

    def downloader(url: str) -> bytes:
        calls.append(url)
        return data

    fetched = fetch_public_panel(item.key, tmp_path, downloader=downloader)
    assert fetched.downloaded and calls == [item.url]
    assert fetched.raw_path.read_bytes() == data
    with fetched.gene_list_path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert [row["gene_id"] for row in rows][:2] == [
        "ENSMUSG00000000000",
        "ENSMUSG00000000001",
    ]
    assert list(rows[0]) == ["gene_symbol", "gene_id"]
    manifest = json.loads(fetched.manifest_path.read_text())
    assert manifest["url"] == item.url and manifest["sha256"] == item.sha256
    assert manifest["n_genes"] == 5
    assert manifest["downloaded"] is True
    first = manifest["first_retrieved_at"]
    assert first is not None and manifest["last_verified_at"] == first
    # A verified copy is reused without downloading; the manifest keeps the
    # first retrieval and records the new check.
    (tmp_path / f"{item.key}.manifest.json").write_text(
        json.dumps({**manifest, "last_verified_at": "2000-01-01T00:00:00+00:00"})
    )
    again = fetch_public_panel(item.key, tmp_path, downloader=downloader)
    assert not again.downloaded and calls == [item.url]
    kept = json.loads(again.manifest_path.read_text())
    assert kept["downloaded"] is True and kept["first_retrieved_at"] == first
    assert kept["last_verified_at"] != "2000-01-01T00:00:00+00:00"
    # A copy found without an earlier manifest has no known retrieval time.
    again.manifest_path.unlink()
    orphan = json.loads(
        fetch_public_panel(
            item.key, tmp_path, downloader=downloader
        ).manifest_path.read_text()
    )
    assert orphan["downloaded"] is False and orphan["first_retrieved_at"] is None


def test_fetch_refuses_a_changed_vendor_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    item = pinned(vendor_table(5), 5)
    monkeypatch.setitem(PUBLIC_PANEL_LISTS, item.key, item)
    with pytest.raises(PublicPanelError, match="vendor file changed"):
        fetch_public_panel(item.key, tmp_path, downloader=lambda url: vendor_table(6))
    assert not (tmp_path / "raw" / item.file_name).exists()
    with pytest.raises(PublicPanelError, match="unknown public panel"):
        fetch_public_panel("nope", tmp_path)


def test_cli_fetch_uses_the_helper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = vendor_table(4)
    item = pinned(data, 4)
    monkeypatch.setitem(PUBLIC_PANEL_LISTS, item.key, item)
    # Seed a verified copy: the command reuses it without any download.
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / item.file_name).write_bytes(data)
    monkeypatch.setattr(public_panels, "download_url", _no_download)
    result = CliRunner().invoke(
        cli_main,
        ["annotation-panel-fetch", "--panel", item.key, "--out-dir", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    assert "4 genes" in result.output and "verified copy" in result.output
    assert (tmp_path / f"{item.key}.gene_list.csv").is_file()


def _no_download(url: str) -> bytes:
    raise AssertionError(f"unexpected download of {url}")

"""Tests for the M6b input script (gene sets, subsamples, prepared queries; §17)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts" / "acceptance"


def _load() -> ModuleType:
    name = "mouse_ap_axis_inputs"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_section_number_and_ids() -> None:
    module = _load()
    assert module.section_number("C57BL6J-638850.57") == 57
    assert module.pair_id_for("C57BL6J-638850.05") == "MF638850_05"
    assert module.sample_id_for("C57BL6J-638850.61") == "MF638850_61_MERSCOPE"
    with pytest.raises(ValueError, match="section label"):
        module.section_number("C57BL6J-638850")


def test_draw_subsample_is_seeded_sorted_and_order_free() -> None:
    module = _load()
    labels = [f"cell{i:05d}" for i in range(1000)]
    first = module.draw_subsample(labels, "C57BL6J-638850.57", n_cells=100)
    again = module.draw_subsample(labels[::-1], "C57BL6J-638850.57", n_cells=100)
    other = module.draw_subsample(labels, "C57BL6J-638850.56", n_cells=100)
    assert first == again
    assert first == sorted(first)
    assert len(set(first)) == 100
    assert first != other
    # The pre-registered rule: rng([0, NN]) over the label-sorted cells.
    rng = np.random.default_rng([0, 57])
    expected = [labels[i] for i in np.sort(rng.choice(1000, 100, replace=False))]
    assert first == expected


def test_draw_subsample_takes_all_when_fewer_and_refuses_duplicates() -> None:
    module = _load()
    assert module.draw_subsample(["b", "a"], "C57BL6J-638850.10", n_cells=5) == [
        "a",
        "b",
    ]
    with pytest.raises(ValueError, match="duplicate"):
        module.draw_subsample(["a", "a"], "C57BL6J-638850.10")


def test_in_scope_sections_excludes_cb_ob_dominated_and_orders_by_ap() -> None:
    module = _load()
    table = pd.DataFrame(
        {
            "CB": [0.0, 0.6, 0.26, 0.0],
            "OB": [0.1, 0.0, 0.2, 0.5],
            "ap": [5.0, 12.0, 10.0, 1.0],
        },
        index=["s.3", "s.1", "s.2", "s.9"],
    )
    assert module.in_scope_sections(table) == ["s.3", "s.2"]


def test_build_gene_sets_intersects_on_ids_in_merfish_order() -> None:
    module = _load()
    genes = pd.DataFrame(
        {
            "gene_identifier": ["ENSMUSG3", "ENSMUSG1", "Blank-1", "ENSMUSG2"],
            "gene_symbol": ["C", "A", "Blank-1", "B"],
        }
    )
    sets = module.build_gene_sets(
        genes, {"ag7": ["ENSMUSG2", "ENSMUSG3", "X"], "vzg2": ["ENSMUSG1"]}
    )
    assert sets["merfish500"]["gene_ids"].tolist() == [
        "ENSMUSG3",
        "ENSMUSG1",
        "ENSMUSG2",
    ]
    assert sets["merfish_ag7"]["gene_symbol"].tolist() == ["C", "B"]
    assert sets["merfish_vzg2"]["gene_ids"].tolist() == ["ENSMUSG1"]


def test_csr_rows_matches_scipy_row_indexing() -> None:
    module = _load()
    rng = np.random.default_rng(1)
    dense = rng.poisson(0.5, size=(40, 7)).astype(float)
    dense[5] = 0.0
    matrix = sparse.csr_matrix(dense)
    rows = np.array([5, 3, 39, 0, 3])
    data, indices, indptr = module.csr_rows(
        matrix.data, matrix.indices, matrix.indptr, rows
    )
    gathered = sparse.csr_matrix((data, indices, indptr), shape=(len(rows), 7))
    np.testing.assert_array_equal(gathered.toarray(), dense[rows])
    empty = module.csr_rows(matrix.data, matrix.indices, matrix.indptr, np.array([5]))
    assert empty[2].tolist() == [0, 0]


def test_rows_of_refuses_missing_cells() -> None:
    module = _load()
    assert module.rows_of(["a", "b", "c"], ["c", "a"]).tolist() == [2, 0]
    with pytest.raises(KeyError, match="missing"):
        module.rows_of(["a"], ["z"])


def test_prepared_query_is_readable_by_prepared_samples(tmp_path: Path) -> None:
    pytest.importorskip("anndata")
    module = _load()
    from merxen.annotation.panel import prepared_samples, raw_panel_from_h5ad

    cells = pd.DataFrame(
        {
            "cell_label": ["c1", "c2"],
            "brain_section_label": ["C57BL6J-638850.40"] * 2,
            "x_section": [1.0, 2.0],
            "y_section": [3.0, 4.0],
        }
    )
    genes = pd.DataFrame(
        {
            "gene_ids": ["ENSMUSG00000000001", "ENSMUSG00000000002"],
            "gene_symbol": ["A", "B"],
        }
    )
    counts = sparse.csr_matrix(np.array([[1.0, 9.0], [0.0, 3.0]], dtype=np.float32))
    module._write_prepared(
        tmp_path / "pair",
        section="C57BL6J-638850.40",
        counts=counts,
        cells=cells,
        genes=genes,
    )
    config = json.loads(
        (tmp_path / "pair" / "clustering_squidpy_config.json").read_text()
    )
    assert config["pair_id"] == "MF638850_40"
    assert config["min_counts"] == 10
    samples = prepared_samples(
        tmp_path / "pair" / "clustering_prepare_out", clustering_config=config
    )
    assert [(s.sample_id, s.platform) for s in samples] == [
        ("MF638850_40_MERSCOPE", "MERSCOPE")
    ]
    raw = raw_panel_from_h5ad(samples[0].h5ad_path)
    assert raw.native_id_column == "gene_ids"

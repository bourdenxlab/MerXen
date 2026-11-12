"""Pinned full-fit and export parity for the optional disk-backed adapter."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from merxen.pciseq_batched import _contributing_spots
from merxen.pciseq_compute import run_pciseq_compute


@pytest.fixture
def upstream_runtime(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Use the optional runtime while keeping the synthetic fit's CPU bounded."""
    package = pytest.importorskip("pciSeq")
    monkeypatch.setenv("LOKY_MAX_CPU_COUNT", "2")
    return package


@pytest.fixture
def prepared_fit(tmp_path: Path) -> Path:
    """A real mask and measured spots exercise validation, geometry and exports."""
    prepared = tmp_path / "prepared"
    (prepared / "masks").mkdir(parents=True)
    labels = np.zeros((81, 81), dtype=np.uint32)
    for row in range(4):
        for column in range(4):
            y, x = 3 + row * 19, 3 + column * 19
            labels[y : y + 13, x : x + 13] = row * 4 + column + 1
    sparse.save_npz(prepared / "masks/plane_0000.npz", sparse.coo_matrix(labels))
    rng = np.random.default_rng(131)
    pd.DataFrame(
        {
            "gene_name": np.resize(["gene_a", "gene_b", "gene_c"], 600),
            "x": rng.uniform(0, 80, 600),
            "y": rng.uniform(0, 80, 600),
        }
    ).to_csv(prepared / "spots.csv", index=False)
    # Natural/lexical ordering differs, and NA is a valid literal class name.
    # The normal compute adapter's reversible tokens must survive both modes.
    pd.DataFrame(
        {
            "Type 2": [30.0, 2.0, 1.0],
            "Type 10": [1.0, 27.0, 3.0],
            "NA": [2.0, 3.0, 29.0],
        },
        index=["gene_a", "gene_b", "gene_c"],
    ).to_csv(prepared / "reference.csv")
    (prepared / "manifest.json").write_text(
        json.dumps(
            {
                "cell_count": 16,
                "voxel_size": [1.0, 1.0, 1.0],
                "geometry_mode": "project_2d",
                "options": {
                    "nNeighbors": 3,
                    "max_iter": 3,
                    "CellCallTolerance": 0.0,
                    "InsideCellBonus": 1.7,
                    "MisreadDensity": {"default": 1e-6},
                    "cell_centroid_prior": {"default": 10},
                    "cell_cov_prior": {"default": 10},
                    "cell_type_prior": "weighted",
                    "cell_type_weights": {"Type 2": 0.2, "Zero": 0.1},
                    "mrf_pooled_classes": [["Type 2", "Type 10"]],
                },
            }
        )
    )
    return prepared


@pytest.mark.parametrize("cell_batch", [1, 7])
@pytest.mark.parametrize("spot_batch", [1, 23, 1000])
def test_full_fit_exports_are_byte_identical_to_pinned_upstream(
    tmp_path: Path,
    prepared_fit: Path,
    upstream_runtime: Any,
    cell_batch: int,
    spot_batch: int,
) -> None:
    """Compare every nested record, not just the final winning assignments."""
    ordinary, bounded = tmp_path / "standard", tmp_path / "bounded"
    run_pciseq_compute(prepared_fit, ordinary)
    scratch = tmp_path / "disk_tensors"
    run_pciseq_compute(
        prepared_fit,
        bounded,
        memory_mode="batched",
        cell_batch_size=cell_batch,
        spot_batch_size=spot_batch,
        scratch_dir=scratch,
    )
    for name in ("cell_data.json", "gene_data.json"):
        assert (bounded / name).read_bytes() == (ordinary / name).read_bytes()
    exported = json.loads((bounded / "cell_data.json").read_text())
    assert any(record["spot_id"] for record in exported)
    assert all(
        name in {"Type 2", "Type 10", "NA", "Zero"}
        for record in exported
        for name in record["ClassName"]
    )
    execution = json.loads((bounded / "compute_manifest.json").read_text())["execution"]
    assert execution["iteration_count"] == 3
    assert len(execution["iteration_deltas"]) == 3
    assert execution["adapter_version"] == "disk_backed_exact_v1"
    source = Path(__file__).resolve().parents[1] / "src/merxen"
    for name, digest in execution["adapter_source_sha256"].items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == digest
    assert not scratch.exists()


def test_unverified_revision_is_rejected(upstream_runtime: Any) -> None:
    """An updated optional runtime must not silently enter the internal adapter."""
    from merxen.pciseq_batched import verify_upstream

    with pytest.raises(RuntimeError, match="verified upstream revision"):
        verify_upstream("0" * 40)


def test_edited_source_with_unchanged_commit_is_rejected(
    upstream_runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Commit metadata alone cannot identify locally edited upstream equations."""
    from merxen import pciseq_batched
    from merxen.pciseq_compute import UPSTREAM_COMMIT

    monkeypatch.setattr(pciseq_batched, "_VERIFIED_SOURCE_SHA256", {"app.py": "0" * 64})
    with pytest.raises(RuntimeError, match="source differs"):
        pciseq_batched.verify_upstream(UPSTREAM_COMMIT)


@pytest.mark.parametrize("chunk", [1, 7, 1000])
@pytest.mark.parametrize("large_ids", [False, True])
def test_disk_contribution_lists_match_upstream_order_and_threshold(
    tmp_path: Path, upstream_runtime: Any, chunk: int, large_ids: bool
) -> None:
    """Repeated candidate cells, empty groups and exact threshold ties remain."""
    from pciSeq.src.core.summary import get_contributing_spots_2

    rng = np.random.default_rng(415)
    count, candidates = 37, 4
    parent = rng.integers(0, 6, (count, candidates), dtype=np.uint32)
    parent[:, -1] = 0
    probabilities = rng.uniform(0.0, 1.0, parent.shape)
    probabilities[::3, 1] = 1e-4
    probabilities[::5, 2] = 0.0
    offset = 2**63 + 17 if large_ids else 0
    ids = np.arange(count, dtype=np.uint64) + np.uint64(offset)
    spots = SimpleNamespace(
        nS=count,
        data=pd.DataFrame({"gene_name": ["g"] * count}, index=pd.Index(ids)),
        gene_id=rng.integers(0, 3, count, dtype=np.int32),
        parent_cell_id=parent,
        parent_cell_prob=probabilities,
    )
    cells, genes = SimpleNamespace(nC=7), SimpleNamespace(nG=4)
    expected = get_contributing_spots_2(cells, spots, genes, 1e-4)
    actual = _contributing_spots(
        cells, spots, genes, 1e-4, scratch_dir=tmp_path, chunk_rows=chunk
    )
    assert actual.shape == expected.shape
    for key in np.ndindex(actual.shape):
        assert list(actual[key]) == list(expected[key])
    assert all(len(value) == 0 for value in actual[:, 3])
    assert all(len(value) == 0 for value in actual[-1])

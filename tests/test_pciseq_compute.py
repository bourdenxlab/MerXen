"""Contracts around the optional, process-isolated pciSeq dependency."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from merxen.pciseq_compute import UPSTREAM_COMMIT, main, run_pciseq_compute


@pytest.fixture
def prepared_input(tmp_path: Path) -> Path:
    """Create the serialized boundary without importing SpatialData."""
    prepared = tmp_path / "prepared"
    (prepared / "masks").mkdir(parents=True)
    (prepared / "manifest.json").write_text(
        json.dumps(
            {
                "cell_count": 2,
                "voxel_size": [1.0, 1.0, 1.0],
                "geometry_mode": "project_2d",
                "options": {},
            }
        )
    )
    pd.DataFrame(
        {"gene_name": ["gene_a", "gene_b"], "x": [1.0, 3.0], "y": [1.0, 1.0]}
    ).to_csv(prepared / "spots.csv", index=False)
    pd.DataFrame(
        {"type_a": [10, 1], "type_b": [1, 10]}, index=["gene_a", "gene_b"]
    ).to_csv(prepared / "reference.csv")
    sparse.save_npz(
        prepared / "masks" / "plane_0000.npz",
        sparse.coo_matrix(np.asarray([[0, 0, 0, 0], [0, 1, 0, 2]], dtype=np.uint32)),
    )
    return prepared


@pytest.fixture
def fake_upstream(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Return a controlled public API without the optional runtime installed."""
    calls: list[dict[str, Any]] = []

    def fit(**kwargs: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
        calls.append(kwargs)
        return (
            pd.DataFrame(
                {
                    "Cell_Num": [2, 1],
                    "ClassName": [["type_b", "Zero"], ["type_a"]],
                    "Prob": [[0.8, 0.199], [0.999]],
                }
            ),
            pd.DataFrame(
                {
                    "spot_id": [1, 0],
                    "gene_name": ["gene_b", "gene_a"],
                    "neighbour": [2, 0],
                }
            ),
        )

    module = SimpleNamespace(
        fit=fit,
        calls=calls,
        __version__="0.1.1",
        __commit__=UPSTREAM_COMMIT[:7],
    )
    monkeypatch.setitem(sys.modules, "pciSeq", module)
    return module


def test_compute_calls_public_api_and_preserves_reported_results(
    prepared_input: Path, fake_upstream: SimpleNamespace, tmp_path: Path
) -> None:
    """Reordered rows retain IDs, background and probability truncation."""
    output = tmp_path / "computed"
    path = run_pciseq_compute(prepared_input, output)
    manifest = json.loads(path.read_text())
    call = fake_upstream.calls[0]
    assert call["opts"]["nNeighbors"] == 2
    assert call["opts"]["save_data"] is False
    assert call["opts"]["realtime_viewer"] is False
    assert call["opts"]["remove_flat_cells"] is False
    assert list(call["spots"].index) == [0, 1]
    assert call["spots"]["z_plane"].tolist() == [0.0, 0.0]
    assert manifest["upstream_version"] == "0.1.1"
    assert (
        manifest["prepared_manifest_sha256"]
        == hashlib.sha256((prepared_input / "manifest.json").read_bytes()).hexdigest()
    )
    assert manifest["has_converged"] is None
    assert manifest["convergence_status"] == "not_exposed_by_fit"
    assert "truncated" in manifest["probability_semantics"]
    assert json.loads((output / "cell_data.json").read_text())[0]["Prob"] == [
        0.8,
        0.199,
    ]
    assert json.loads((output / "gene_data.json").read_text())[1]["neighbour"] == 0


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("gene_name", "unknown_gene", "occur in the reference"),
        ("x", -1.0, "in bounds"),
        ("y", np.nan, "in bounds"),
        ("z_plane", 2.0, "in bounds"),
    ],
)
def test_compute_rejects_filtering_that_would_invalidate_spot_ids(
    prepared_input: Path,
    fake_upstream: SimpleNamespace,
    tmp_path: Path,
    column: str,
    value: str | float,
    message: str,
) -> None:
    """Every spot reaching fit must survive upstream preprocessing."""
    spots = pd.read_csv(prepared_input / "spots.csv")
    spots.loc[0, column] = value
    spots.to_csv(prepared_input / "spots.csv", index=False)
    with pytest.raises(ValueError, match=message):
        run_pciseq_compute(prepared_input, tmp_path / "computed")
    assert not fake_upstream.calls


def test_compute_rejects_excluded_genes_before_fit(
    prepared_input: Path, fake_upstream: SimpleNamespace, tmp_path: Path
) -> None:
    """Filtering belongs in preparation alongside the transcript ID map."""
    path = prepared_input / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["options"] = {"exclude_genes": ["gene_a"]}
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="removed during preparation"):
        run_pciseq_compute(prepared_input, tmp_path / "computed")
    assert not fake_upstream.calls


def test_compute_forces_headless_export_free_options(
    prepared_input: Path, fake_upstream: SimpleNamespace, tmp_path: Path
) -> None:
    """Unsafe viewer/export options cannot cross the isolated boundary."""
    path = prepared_input / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["options"] = {
        "save_data": True,
        "realtime_viewer": True,
        "remove_flat_cells": True,
        "voxel_size": [5, 5, 5],
        "max_iter": 2,
    }
    path.write_text(json.dumps(manifest))
    run_pciseq_compute(prepared_input, tmp_path / "computed")
    options = fake_upstream.calls[0]["opts"]
    assert options["save_data"] is False
    assert options["realtime_viewer"] is False
    assert options["remove_flat_cells"] is False
    assert options["voxel_size"] == [1, 1, 1]
    assert options["max_iter"] == 2


@pytest.mark.parametrize(
    "change", ["cell_ids", "spot_ids", "gene_mapping", "probability"]
)
def test_compute_rejects_invalid_upstream_output(
    prepared_input: Path,
    fake_upstream: SimpleNamespace,
    tmp_path: Path,
    change: str,
) -> None:
    """A dependency/API drift cannot silently mislabel cells or transcripts."""
    original_fit = fake_upstream.fit

    def invalid_fit(**kwargs: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
        cells, spots = original_fit(**kwargs)
        if change == "cell_ids":
            cells.loc[0, "Cell_Num"] = 0
        elif change == "spot_ids":
            spots.loc[0, "spot_id"] = 0
        elif change == "gene_mapping":
            spots.loc[0, "gene_name"] = "gene_a"
        else:
            cells.at[0, "Prob"] = [float("nan"), 0.1]
        return cells, spots

    fake_upstream.fit = invalid_fit
    output = tmp_path / "computed"
    with pytest.raises(ValueError, match="pciSeq"):
        run_pciseq_compute(prepared_input, output)
    assert not (output / "compute_manifest.json").exists()


def test_compute_accepts_empty_thresholded_class_summary(
    prepared_input: Path, fake_upstream: SimpleNamespace, tmp_path: Path
) -> None:
    """Very diffuse posteriors can have no class above upstream's cutoff."""
    original_fit = fake_upstream.fit

    def empty_summary(**kwargs: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
        cells, spots = original_fit(**kwargs)
        cells.at[0, "ClassName"] = []
        cells.at[0, "Prob"] = []
        return cells, spots

    fake_upstream.fit = empty_summary
    run_pciseq_compute(prepared_input, tmp_path / "computed")


@pytest.mark.parametrize("value", [float("nan"), -1.0])
def test_compute_rejects_invalid_reference_expression(
    prepared_input: Path,
    fake_upstream: SimpleNamespace,
    tmp_path: Path,
    value: float,
) -> None:
    """Invalid reference means must fail before entering model code."""
    reference = pd.read_csv(prepared_input / "reference.csv", index_col=0)
    reference.iloc[0, 0] = value
    reference.to_csv(prepared_input / "reference.csv")
    with pytest.raises(ValueError, match="finite and nonnegative"):
        run_pciseq_compute(prepared_input, tmp_path / "computed")
    assert not fake_upstream.calls


def test_compute_cli_uses_only_prepared_files(
    prepared_input: Path,
    fake_upstream: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The process entry point needs neither configuration nor SpatialData I/O."""
    output = tmp_path / "computed"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pciseq_compute",
            "--prepared-dir",
            str(prepared_input),
            "--output-dir",
            str(output),
        ],
    )
    main()
    assert (output / "compute_manifest.json").exists()
    assert len(fake_upstream.calls) == 1


def test_compute_explains_missing_optional_runtime(
    prepared_input: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An absent dependency should identify the isolated environment to use."""
    monkeypatch.setitem(sys.modules, "pciSeq", None)
    with pytest.raises(ImportError, match="pciseq-runtime"):
        run_pciseq_compute(prepared_input, tmp_path / "computed")


@pytest.mark.parametrize("options", [{"nNeighbors": 0}, {"max_iter": 0}])
def test_compute_rejects_degenerate_runtime_options(
    prepared_input: Path,
    fake_upstream: SimpleNamespace,
    tmp_path: Path,
    options: dict[str, int],
) -> None:
    """Zero iterations or neighbors must fail before upstream initializes state."""
    path = prepared_input / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["options"] = options
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="positive integer"):
        run_pciseq_compute(prepared_input, tmp_path / "computed")
    assert not fake_upstream.calls


def test_compute_rejects_unequal_xy_voxel_sizes(
    prepared_input: Path, fake_upstream: SimpleNamespace, tmp_path: Path
) -> None:
    """Scaled Y would otherwise index the original unscaled mask upstream."""
    path = prepared_input / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["voxel_size"] = [1.0, 2.0, 1.0]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="equal X/Y"):
        run_pciseq_compute(prepared_input, tmp_path / "computed")
    assert not fake_upstream.calls


def test_compute_rejects_missing_segmentation_plane(
    prepared_input: Path, fake_upstream: SimpleNamespace, tmp_path: Path
) -> None:
    """A gap must fail before it changes the interpretation of z_plane."""
    source = prepared_input / "masks" / "plane_0000.npz"
    source.rename(source.with_name("plane_0001.npz"))
    with pytest.raises(ValueError, match="start at zero with no gaps"):
        run_pciseq_compute(prepared_input, tmp_path / "computed")
    assert not fake_upstream.calls

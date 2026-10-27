"""Contracts around the optional, process-isolated pciSeq dependency."""

from __future__ import annotations

import hashlib
import json
import re
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
        classes = sorted(
            kwargs["scRNAseq"].columns,
            key=lambda name: [
                int(part) if part.isdigit() else part.lower()
                for part in re.split(r"(\d+)", name)
            ],
        )
        assert classes == sorted(classes, key=str.lower)
        return (
            pd.DataFrame(
                {
                    "Cell_Num": [2, 1],
                    "ClassName": [[classes[1], "Zero"], [classes[0]]],
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
    assert json.loads((output / "cell_data.json").read_text())[0]["ClassName"] == [
        "type_b",
        "Zero",
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
    "change", ["cell_ids", "spot_ids", "gene_mapping", "probability", "class_token"]
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
        elif change == "class_token":
            cells.at[0, "ClassName"] = ["merxen_class_999999", "Zero"]
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


@pytest.mark.parametrize(
    "names",
    [
        ["type2", "type10"],
        ["L2/3 IT 2", "l2/3 IT 10"],
        ["Alpha", "alpha"],
        ["zero", "ZERO"],
        ["merxen_class_000000", "Zero-like"],
    ],
)
def test_compute_class_tokens_preserve_values_names_and_prepared_files(
    prepared_input: Path,
    fake_upstream: SimpleNamespace,
    tmp_path: Path,
    names: list[str],
) -> None:
    """Natural/lexical conflicts and case collisions do not alter class identity."""
    path = prepared_input / "reference.csv"
    reference = pd.read_csv(path, index_col=0)
    reference.columns = names
    reference.to_csv(path)
    original_bytes = path.read_bytes()
    output = tmp_path / "computed"
    manifest = json.loads(run_pciseq_compute(prepared_input, output).read_text())
    tokens = manifest["class_token_map"]
    assert list(tokens.values()) == names
    assert list(tokens) == sorted(tokens, key=str.lower)
    passed = fake_upstream.calls[0]["scRNAseq"]
    pd.testing.assert_frame_equal(passed.rename(columns=tokens), reference)
    assert path.read_bytes() == original_bytes
    cells = json.loads((output / "cell_data.json").read_text())
    assert cells[0]["ClassName"] == [names[1], "Zero"]
    assert cells[1]["ClassName"] == [names[0]]
    assert [cell["Cell_Num"] for cell in cells] == [2, 1]
    spots = json.loads((output / "gene_data.json").read_text())
    assert [spot["spot_id"] for spot in spots] == [1, 0]
    assert [spot["neighbour"] for spot in spots] == [2, 0]


def test_compute_remaps_class_options_and_ignores_unknown_token_like_weights(
    prepared_input: Path,
    fake_upstream: SimpleNamespace,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Weight and MRF associations survive while unknown names remain ignored."""
    path = prepared_input / "manifest.json"
    manifest = json.loads(path.read_text())
    options = {
        "cell_type_weights": {
            "type_a": 0.1,
            "type_b": 0.2,
            "Zero": 0.3,
            "merxen_class_000000": 0.4,
        },
        "mrf_pooled_classes": [["type_b", "type_a"]],
    }
    manifest["options"] = options
    path.write_text(json.dumps(manifest))
    original_bytes = path.read_bytes()
    result = json.loads(
        run_pciseq_compute(prepared_input, tmp_path / "computed").read_text()
    )
    encode = {name: token for token, name in result["class_token_map"].items()}
    passed_options = fake_upstream.calls[0]["opts"]
    assert passed_options["cell_type_weights"] == {
        encode["type_a"]: 0.1,
        encode["type_b"]: 0.2,
        "Zero": 0.3,
    }
    assert passed_options["mrf_pooled_classes"] == [
        [encode["type_b"], encode["type_a"]]
    ]
    assert result["requested_options"] == options
    assert (
        result["resolved_options"]["cell_type_weights"] == options["cell_type_weights"]
    )
    assert (
        result["resolved_options"]["mrf_pooled_classes"]
        == options["mrf_pooled_classes"]
    )
    assert result["upstream_options"] == passed_options
    assert result["ignored_cell_type_weight_names"] == ["merxen_class_000000"]
    assert "unknown class" in caplog.text
    assert path.read_bytes() == original_bytes


def test_compute_does_not_mutate_recorded_options_when_upstream_mutates_arguments(
    prepared_input: Path, fake_upstream: SimpleNamespace, tmp_path: Path
) -> None:
    """Nested model options and expression copies cannot rewrite provenance."""
    manifest_path = prepared_input / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["options"] = {"cell_type_weights": {"type_a": 0.25, "Zero": 0.5}}
    manifest_path.write_text(json.dumps(manifest))
    original_reference = (prepared_input / "reference.csv").read_bytes()
    original_fit = fake_upstream.fit

    def mutating_fit(**kwargs: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
        cells, spots = original_fit(**kwargs)
        kwargs["opts"]["cell_type_weights"]["Zero"] = 0.99
        kwargs["scRNAseq"].iloc[0, 0] = 999
        return cells, spots

    fake_upstream.fit = mutating_fit
    result = json.loads(
        run_pciseq_compute(prepared_input, tmp_path / "computed").read_text()
    )
    assert result["requested_options"]["cell_type_weights"]["Zero"] == 0.5
    assert result["resolved_options"]["cell_type_weights"]["Zero"] == 0.5
    assert result["upstream_options"]["cell_type_weights"]["Zero"] == 0.5
    assert (prepared_input / "reference.csv").read_bytes() == original_reference


def test_compute_rejects_unknown_pooled_class_even_when_it_matches_a_token(
    prepared_input: Path, fake_upstream: SimpleNamespace, tmp_path: Path
) -> None:
    path = prepared_input / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["options"] = {"mrf_pooled_classes": [["type_a", "merxen_class_000001"]]}
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unknown cell classes"):
        run_pciseq_compute(prepared_input, tmp_path / "computed")
    assert not fake_upstream.calls


def test_compute_rejects_reserved_zero_reference_class_before_fit(
    prepared_input: Path, fake_upstream: SimpleNamespace, tmp_path: Path
) -> None:
    path = prepared_input / "reference.csv"
    reference = pd.read_csv(path, index_col=0).rename(columns={"type_a": "Zero"})
    reference.to_csv(path)
    with pytest.raises(ValueError, match="cannot include Zero"):
        run_pciseq_compute(prepared_input, tmp_path / "computed")
    assert not fake_upstream.calls

"""Atera build acceptance script on the synthetic bundle (plan §1.1, §7.4).

One store is built from the synthetic bundle per module; tests that break the
store or the bundle work on copies.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import zarr
from scipy import sparse

from merxen.config import AteraBuildConfig
from merxen.io.builders import atera as atera_builder

from .atera_synth import AteraSynth, rewrite_zip_store, write_atera_bundle

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/acceptance/atera_build.py"
FAST = AteraBuildConfig(
    transcript_partition_rows=1_048_576, polygon_n_jobs=2, morphology_levels=3
)


@pytest.fixture(scope="module")
def acceptance() -> ModuleType:
    """Load the standalone acceptance script without running it."""
    spec = importlib.util.spec_from_file_location("atera_acceptance", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Dataclasses look their module up in sys.modules while it executes.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


@dataclass(frozen=True)
class Built:
    synth: AteraSynth
    store: Path


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Built:
    root = tmp_path_factory.mktemp("atera_acceptance")
    synth = write_atera_bundle(root / "roi")
    store = atera_builder.write_atera_spatialdata(
        input_path=synth.root,
        output_path=root / "source.zarr",
        build_config=FAST,
        n_workers=2,
    )
    return Built(synth, store)


@pytest.fixture
def copied(built: Built, tmp_path: Path) -> tuple[Built, Path]:
    """A private copy of the bundle and the store (mtimes kept)."""
    roi = tmp_path / "roi"
    store = tmp_path / "source.zarr"
    shutil.copytree(built.synth.root, roi)
    shutil.copytree(built.store, store)
    return Built(built.synth, store), roi


def _run(module: ModuleType, store: Path, roi: Path, **kwargs: object) -> dict:
    return module.run_acceptance(store, roi, alias="SYN1_ATERA", n_workers=2, **kwargs)


def test_every_criterion_passes_on_a_fresh_build(
    acceptance: ModuleType, built: Built
) -> None:
    report = _run(acceptance, built.store, built.synth.root)
    assert report["passed"], json.dumps(report["criteria"], indent=1, default=str)
    assert list(report["criteria"]) == list(acceptance.CRITERIA)
    for entry in report["criteria"].values():
        assert entry["seconds"] >= 0 and entry["peak_rss_gb_after"] > 0
    matrix = report["criteria"]["points_vs_matrix"]["details"]
    synth = built.synth
    assert matrix["vendor_total"] == matrix["points_total"] == synth.cfm_counts.sum()
    assert matrix["points_gene_total"] == synth.cfm_counts[:, synth.is_gene].sum()
    schema = report["criteria"]["schema"]["details"]
    kept = synth.transcripts["qv"].to_numpy(np.float64) >= 20.0
    assert schema["points_rows"] == int(kept.sum())
    labels = report["criteria"]["labels_vs_cell_summary"]["details"]
    assert labels["max_cell_label"] == synth.n_cells
    assert labels["labelled_pixels"] == int((synth.cell_mask > 0).sum())
    containment = report["criteria"]["nucleus_containment"]["details"]
    assert containment["containment_violations"] == 0
    assert containment["max_nucleus_label"] == len(synth.nucleus_cell_index)
    assert report["element_sizes_bytes"]["total"] > 0
    # Aliases only: no host path in the report.
    assert str(built.store.parent) not in json.dumps(report)


def test_matrix_comparison_allows_only_the_deprecated_codeword_excess(
    acceptance: ModuleType,
) -> None:
    names = ["GENE1", "DeprecatedCodeword_0001", "NegControlProbe_1", "GENE2"]
    types = [
        "Gene Expression",
        "Deprecated Codeword",
        "Negative Control Probe",
        "Gene Expression",
    ]
    points = sparse.csr_matrix(np.array([[3, 0, 1, 0], [0, 0, 0, 2]], dtype=np.int64))
    vendor = points.copy().tolil()
    vendor[1, 1] = 5  # no transcripts in this column at all: the vendor quirk
    vendor = vendor.tocsr()
    gene_counts = np.array([10, 0, 4, 2])
    ok = acceptance.compare_matrix_counts(
        vendor,
        points,
        feature_names=names,
        feature_types=types,
        gene_counts_all_qv=gene_counts,
        expected_excess=5,
    )
    assert ok.passed
    assert ok.details["excess_by_column"] == {"DeprecatedCodeword_0001": 5}
    assert ok.details["vendor_total"] == 11 and ok.details["points_total"] == 6
    assert ok.details["vendor_gene_total"] == ok.details["points_gene_total"] == 5

    wrong_total = acceptance.compare_matrix_counts(
        vendor,
        points,
        feature_names=names,
        feature_types=types,
        gene_counts_all_qv=gene_counts,
        expected_excess=84,
    )
    assert not wrong_total.passed

    # An excess on a column without transcripts that is not a deprecated codeword.
    other = gene_counts.copy()
    other[2] = 0
    vendor_other = vendor.tolil()
    vendor_other[0, 2] = 2
    vendor_other = vendor_other.tocsr()
    points_other = points.tolil()
    points_other[0, 2] = 0
    bad = acceptance.compare_matrix_counts(
        vendor_other,
        points_other.tocsr(),
        feature_names=names,
        feature_types=types,
        gene_counts_all_qv=other,
    )
    assert not bad.passed
    assert bad.details["excess_on_non_deprecated_columns"] == ["NegControlProbe_1"]

    # Any difference on a column with transcripts fails, whatever its sign.
    off = vendor.tolil()
    off[0, 0] = 2
    mismatch = acceptance.compare_matrix_counts(
        off.tocsr(),
        points,
        feature_names=names,
        feature_types=types,
        gene_counts_all_qv=gene_counts,
    )
    assert not mismatch.passed
    assert mismatch.details["mismatched_entries_on_columns_with_transcripts"] == 1
    assert mismatch.details["first_mismatches"][0]["feature"] == "GENE1"


def test_a_vendor_matrix_count_without_a_point_fails(
    acceptance: ModuleType, copied: tuple[Built, Path]
) -> None:
    built, roi = copied
    cfm = next(roi.glob("*cell_feature_matrix.zarr.zip"))

    def bump(group: zarr.Group) -> None:
        data = group["X/data"]
        values = np.asarray(data[:])
        values[0] += 1
        data[:] = values

    rewrite_zip_store(cfm, bump)
    report = _run(
        acceptance, built.store, roi, criteria=["points_vs_matrix", "build_marker"]
    )
    matrix = report["criteria"]["points_vs_matrix"]
    assert not matrix["passed"]
    assert matrix["details"]["mismatched_entries_on_columns_with_transcripts"] == 1
    assert matrix["details"]["vendor_total"] == matrix["details"]["points_total"] + 1
    # The edited bundle no longer matches the store's fingerprint.
    assert not report["criteria"]["build_marker"]["passed"]


def test_an_expected_excess_the_matrix_lacks_fails(
    acceptance: ModuleType, built: Built
) -> None:
    report = _run(
        acceptance,
        built.store,
        built.synth.root,
        criteria=["points_vs_matrix"],
        expected_vendor_excess=84,
    )
    entry = report["criteria"]["points_vs_matrix"]
    assert not entry["passed"]
    assert entry["details"]["excess_on_columns_without_transcripts"] == 0


def test_label_and_containment_errors_are_counted(
    acceptance: ModuleType, copied: tuple[Built, Path]
) -> None:
    built, roi = copied
    synth = built.synth
    nuclei = zarr.open_array(
        built.store / "labels" / "nucleus_labels" / "s0", mode="r+"
    )
    cells = zarr.open_array(built.store / "labels" / "cell_labels" / "s0", mode="r+")
    owner = synth.nucleus_cell_index
    # A nucleus pixel relabelled to a nucleus of another cell.
    row, col = np.argwhere(synth.nucleus_mask == 1)[0]
    other = int(np.flatnonzero(owner != owner[0])[0]) + 1
    nucleus_values = np.asarray(nuclei[:])
    nucleus_values[row, col] = other
    nuclei[:] = nucleus_values
    # A background pixel given to cell 1 changes its area and centroid.
    background = np.argwhere(synth.cell_mask == 0)[-1]
    cell_values = np.asarray(cells[:])
    cell_values[tuple(background)] = 1
    cells[:] = cell_values

    report = _run(
        acceptance,
        built.store,
        roi,
        criteria=["labels_vs_cell_summary", "nucleus_containment"],
    )
    labels = report["criteria"]["labels_vs_cell_summary"]
    assert not labels["passed"]
    assert labels["details"]["area_over_tolerance"] == 1
    containment = report["criteria"]["nucleus_containment"]
    assert not containment["passed"]
    assert containment["details"]["containment_violations"] == 1


def test_metadata_and_marker_errors_fail(
    acceptance: ModuleType, copied: tuple[Built, Path]
) -> None:
    built, roi = copied
    (built.store / "atera" / "experiment.spatial").write_text("{}")
    (built.store / atera_builder.BUILD_MARKER_NAME).unlink()
    report = _run(acceptance, built.store, roi, criteria=["metadata", "build_marker"])
    metadata = report["criteria"]["metadata"]
    assert not metadata["passed"]
    assert any("experiment.spatial" in item for item in metadata["details"]["failures"])
    assert not report["criteria"]["build_marker"]["passed"]


def test_cli_writes_the_report_and_sets_the_exit_status(
    acceptance: ModuleType, built: Built, tmp_path: Path
) -> None:
    report_path = tmp_path / "evidence" / "report.json"
    argv = [
        "--store",
        str(built.store),
        "--bundle",
        str(built.synth.root),
        "--alias",
        "SYN1_ATERA",
        "--report",
        str(report_path),
        "--workers",
        "2",
        "--criteria",
        "schema",
        "um_per_px",
    ]
    assert acceptance.main(argv) == 0
    report = json.loads(report_path.read_text())
    assert report["passed"] and set(report["criteria"]) == {"schema", "um_per_px"}
    assert (
        acceptance.main(
            [
                *argv[:-3],
                "--criteria",
                "points_vs_matrix",
                "--expected-vendor-excess",
                "1",
            ]
        )
        == 1
    )

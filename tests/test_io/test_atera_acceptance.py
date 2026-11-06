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
    # The synthetic matrix has no deprecated-codeword excess.
    kwargs.setdefault("expected_vendor_excess", 0)
    return module.run_acceptance(store, roi, alias="SYN1_ATERA", n_workers=2, **kwargs)


def _failures(report: dict, name: str) -> str:
    entry = report["criteria"][name]
    assert not entry["passed"], name
    return entry["error"]


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
    assert labels["pixels_differing_from_vendor_mask"] == 0
    containment = report["criteria"]["nucleus_containment"]["details"]
    assert containment["containment_violations"] == 0
    assert containment["pixels_differing_from_vendor_mask"] == 0
    assert containment["max_nucleus_label"] == len(synth.nucleus_cell_index)
    table = report["criteria"]["table_vs_vendor"]["details"]
    assert table["x_differing_entries"] == 0
    assert table["x_vendor_gene_total"] == synth.cfm_counts[:, synth.is_gene].sum()
    assert table["n_de_columns"] > 0
    assert set(table["obsm_checked"]) == {"X_pca", "X_umap"}
    assert table["colors_checked"]
    assert "nucleus_centroid_x" in table["cell_summary_columns_checked"]
    assert report["criteria"]["build_options"]["details"]["recorded"]["min_qv"] == 20.0
    assert set(report["code_revision"]) == {"commit", "dirty"}
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
    assert labels["details"]["pixels_differing_from_vendor_mask"] == 1
    containment = report["criteria"]["nucleus_containment"]
    assert not containment["passed"]
    assert containment["details"]["containment_violations"] == 1
    assert containment["details"]["pixels_differing_from_vendor_mask"] == 1


def test_a_label_swap_that_keeps_the_statistics_fails_on_the_vendor_mask(
    acceptance: ModuleType, copied: tuple[Built, Path]
) -> None:
    built, roi = copied
    cells = zarr.open_array(built.store / "labels" / "cell_labels" / "s0", mode="r+")
    values = np.asarray(cells[:])
    # Swapping one pixel of cell 1 with one of cell 2 keeps both pixel counts
    # (areas); only the pixel comparison with masks/1 names both pixels.
    one = np.argwhere(values == 1)
    two = np.argwhere(values == 2)
    values[tuple(one[0])], values[tuple(two[0])] = 2, 1
    cells[:] = values
    report = _run(acceptance, built.store, roi, criteria=["labels_vs_cell_summary"])
    details = report["criteria"]["labels_vs_cell_summary"]["details"]
    assert details["pixels_differing_from_vendor_mask"] == 2
    assert "masks/1" in _failures(report, "labels_vs_cell_summary")


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


def test_label_statistics_do_not_depend_on_blocks_and_strips(
    acceptance: ModuleType, built: Built
) -> None:
    synth = built.synth
    whole = acceptance.stream_label_stats(
        built.store,
        n_cells=synth.n_cells,
        nucleus_owner=synth.nucleus_cell_index,
        n_workers=1,
    )
    pieces = acceptance.stream_label_stats(
        built.store,
        n_cells=synth.n_cells,
        nucleus_owner=synth.nucleus_cell_index,
        n_workers=3,
        block=(13, 11),
        strip_rows=5,
    )
    rows, cols = np.nonzero(synth.cell_mask == 1)
    assert whole.cell_counts[1] == len(rows)
    assert whole.cell_row_sums[1] == rows.sum()
    assert whole.cell_col_sums[1] == cols.sum()
    for name in ("cell_counts", "cell_row_sums", "cell_col_sums", "nucleus_counts"):
        np.testing.assert_array_equal(getattr(whole, name), getattr(pieces, name))
    assert whole.containment_violations == pieces.containment_violations == 0


def test_a_tile_row_count_off_by_one_fails_the_schema_check(
    acceptance: ModuleType, built: Built, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = acceptance.raw_rows_per_tile

    def one_more(*args: object, **kwargs: object) -> dict[str, int]:
        rows = real(*args, **kwargs)
        key = next(key for key, value in sorted(rows.items()) if value)
        return {**rows, key: rows[key] + 1}

    monkeypatch.setattr(acceptance, "raw_rows_per_tile", one_more)
    report = _run(acceptance, built.store, built.synth.root, criteria=["schema"])
    assert not report["criteria"]["schema"]["passed"]


def _rewrite_shapes(store: Path, key: str, edit: object) -> None:
    import geopandas as gpd

    path = store / "shapes" / key / "shapes.parquet"
    frame = gpd.read_parquet(path)
    frame = edit(frame)  # type: ignore[operator]
    frame.to_parquet(path)


def test_a_shape_centroid_off_by_more_than_the_tolerance_fails(
    acceptance: ModuleType, copied: tuple[Built, Path]
) -> None:
    import shapely

    built, roi = copied

    def shift(frame: object) -> object:
        frame = frame.copy()  # type: ignore[attr-defined]
        first = frame.index[0]
        frame.loc[first, "geometry"] = shapely.affinity.translate(
            frame.geometry.loc[first], xoff=0.05
        )
        return frame

    _rewrite_shapes(built.store, atera_builder.CELL_SHAPES_KEY, shift)
    report = _run(acceptance, built.store, roi, criteria=["shape_centroids"])
    details = report["criteria"]["shape_centroids"]["details"]
    assert details["over_tolerance"] == 1
    assert "centroids differ" in _failures(report, "shape_centroids")


def test_a_duplicated_cell_shape_fails(
    acceptance: ModuleType, copied: tuple[Built, Path]
) -> None:
    import pandas as pd

    built, roi = copied
    _rewrite_shapes(
        built.store,
        atera_builder.CELL_SHAPES_KEY,
        lambda frame: pd.concat([frame, frame.iloc[:1]]),
    )
    report = _run(acceptance, built.store, roi, criteria=["shape_centroids"])
    assert "exactly one shape" in _failures(report, "shape_centroids")


def test_build_options_other_than_section_1_1_fail(
    acceptance: ModuleType, built: Built
) -> None:
    report = _run(
        acceptance,
        built.store,
        built.synth.root,
        criteria=["build_options"],
        expected_options=acceptance.ExpectedOptions(boundary_simplify_px=0.5),
    )
    assert "boundary_simplify_px" in _failures(report, "build_options")
    report = _run(
        acceptance,
        built.store,
        built.synth.root,
        criteria=["build_options"],
        expected_options=acceptance.ExpectedOptions(min_qv=10.0),
    )
    assert "min_qv" in _failures(report, "build_options")


def test_an_unpinned_vendor_excess_fails_unless_allowed(
    acceptance: ModuleType, built: Built
) -> None:
    unpinned = _run(
        acceptance,
        built.store,
        built.synth.root,
        criteria=["points_vs_matrix"],
        expected_vendor_excess=None,
    )
    assert "not pinned" in _failures(unpinned, "points_vs_matrix")
    allowed = _run(
        acceptance,
        built.store,
        built.synth.root,
        criteria=["points_vs_matrix"],
        expected_vendor_excess=None,
        allow_unpinned_excess=True,
    )
    assert allowed["passed"]
    with pytest.raises(SystemExit):
        acceptance.parse_args(
            ["--store", "s", "--bundle", "b", "--alias", "A", "--report", "r.json"]
        )


def test_a_leftover_partial_build_folder_fails_the_marker_check(
    acceptance: ModuleType, copied: tuple[Built, Path]
) -> None:
    built, roi = copied
    (built.store.parent / f"{built.store.name}.partial-1234").mkdir()
    report = _run(acceptance, built.store, roi, criteria=["build_marker"])
    assert report["criteria"]["build_marker"]["details"]["leftover_partials"] == 1
    assert "partial" in _failures(report, "build_marker")


def test_table_edits_fail_the_table_check(
    acceptance: ModuleType, copied: tuple[Built, Path]
) -> None:
    built, roi = copied
    table = built.store / "tables" / "table"
    de = zarr.open_group(table / "varm" / "vendor_de", mode="r+")
    name = next(key for key in de.array_keys() if key.startswith("de_"))
    values = np.asarray(de[name][:])
    values[0] = values[0] + 1.0
    de[name][:] = values
    umap = zarr.open_array(table / "obsm" / "X_umap_vendor", mode="r+")
    coords = np.asarray(umap[:])
    coords[0, 0] += 1.0
    umap[:] = coords
    report = _run(acceptance, built.store, roi, criteria=["table_vs_vendor"])
    failures = report["criteria"]["table_vs_vendor"]["details"]["failures"]
    assert any("vendor_de" in item for item in failures)
    assert any("X_umap" in item for item in failures)


def test_a_metrics_value_edit_fails_the_metadata_check(
    acceptance: ModuleType, copied: tuple[Built, Path]
) -> None:
    built, roi = copied
    root = zarr.open_group(built.store, mode="r+")
    attrs = dict(root.attrs[atera_builder.ATERA_ATTRS_KEY])
    metrics = [dict(row) for row in attrs["metrics_summary"]]
    name = next(key for key, value in metrics[0].items() if isinstance(value, int))
    metrics[0][name] += 1
    attrs["metrics_summary"] = metrics
    root.attrs[atera_builder.ATERA_ATTRS_KEY] = attrs
    report = _run(acceptance, built.store, roi, criteria=["metadata"])
    failures = report["criteria"]["metadata"]["details"]["failures"]
    assert any(name in item for item in failures)

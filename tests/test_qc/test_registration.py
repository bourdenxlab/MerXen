"""Tests for the segmentation-to-transcript registration check."""

from __future__ import annotations

import json
from pathlib import Path

import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from click.testing import CliRunner
from spatialdata import SpatialData
from spatialdata.models import PointsModel, ShapesModel
from spatialdata.transformations import Affine

from merxen.cli.run_qc import qc_command
from merxen.qc.metrics import compute_dataset_qc, save_dataset_qc
from merxen.qc.registration import (
    RegistrationCheckParams,
    RegistrationCheckResult,
    RegistrationStatus,
    compute_segmentation_registration_qc,
    sample_registration_windows,
    segmentation_registration_check,
    transcripts_in_windows,
)

# The VZG2 misregistration: segmentation minus truth, in microns.
VZG2_OFFSET = np.array([21.86, 111.78])
TISSUE_UM = 1200.0


def _tissue(
    seed: int = 0,
    *,
    n_cells: int = 4000,
    transcripts_per_cell: int = 40,
    background_per_um2: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (cell centres, transcripts) for a synthetic tissue in microns."""
    rng = np.random.default_rng(seed)
    centres = rng.uniform(0.0, TISSUE_UM, size=(n_cells, 2))
    cell_tx = np.repeat(centres, transcripts_per_cell, axis=0) + rng.normal(
        0.0, 1.5, size=(n_cells * transcripts_per_cell, 2)
    )
    n_background = int(background_per_um2 * TISSUE_UM**2)
    background = rng.uniform(0.0, TISSUE_UM, size=(n_background, 2))
    return centres, np.vstack([cell_tx, background])


def _run(
    centroids: np.ndarray,
    transcripts: np.ndarray,
    reference: np.ndarray | None,
    params: RegistrationCheckParams | None = None,
) -> RegistrationCheckResult:
    params = params or RegistrationCheckParams()
    windows = sample_registration_windows(centroids, params)
    in_windows = transcripts_in_windows(
        pd.DataFrame({"x": transcripts[:, 0], "y": transcripts[:, 1]}),
        windows,
        x_col="x",
        y_col="y",
    )
    return segmentation_registration_check(
        centroids,
        in_windows,
        windows,
        reference_centroids=reference,
        params=params,
    )


def test_registered_segmentation_passes_with_zero_offset() -> None:
    centres, transcripts = _tissue()
    rng = np.random.default_rng(1)
    segmentation = centres + rng.normal(0.0, 0.5, size=centres.shape)

    result = _run(segmentation, transcripts, reference=centres)

    assert result.status == RegistrationStatus.PASS
    assert result.density_ratio is not None and result.density_ratio > 2.0
    assert result.shift_um is not None and result.shift_um <= 2.0
    assert result.n_shift_windows >= 4
    assert result.reasons == []


def test_misregistered_segmentation_warns_and_recovers_the_offset() -> None:
    centres, transcripts = _tissue()

    result = _run(centres + VZG2_OFFSET, transcripts, reference=centres)

    assert result.status == RegistrationStatus.WARN
    assert result.density_ratio is not None and result.density_ratio < 1.5
    # Reported as reference minus segmentation, on a 2 um grid.
    assert result.shift_x_um == pytest.approx(-VZG2_OFFSET[0], abs=2.0)
    assert result.shift_y_um == pytest.approx(-VZG2_OFFSET[1], abs=2.0)
    assert len(result.reasons) == 2


def test_density_alone_flags_misregistration_without_reference() -> None:
    centres, transcripts = _tissue()

    result = _run(centres + VZG2_OFFSET, transcripts, reference=None)

    assert result.status == RegistrationStatus.WARN
    assert result.shift_um is None
    assert result.n_shift_windows == 0


def test_small_offset_is_caught_by_cross_correlation() -> None:
    """A 6 um offset keeps some density but exceeds the offset threshold."""
    centres, transcripts = _tissue()

    result = _run(centres + np.array([6.0, 0.0]), transcripts, reference=centres)

    assert result.status == RegistrationStatus.WARN
    assert result.shift_x_um == pytest.approx(-6.0, abs=2.0)
    assert any("offset" in reason for reason in result.reasons)


def test_check_skips_when_too_few_cells() -> None:
    centres, transcripts = _tissue(n_cells=20)

    result = _run(centres, transcripts, reference=None)

    assert result.status == RegistrationStatus.SKIPPED
    assert "centroids" in result.reasons[0]


def test_check_skips_without_transcripts() -> None:
    centres, _ = _tissue()

    result = _run(centres, np.empty((0, 2)), reference=None)

    assert result.status == RegistrationStatus.SKIPPED


def test_transcripts_in_windows_filters_dask_partitions_and_controls() -> None:
    frame = pd.DataFrame(
        {
            "x": [1.0, 5.0, 50.0, 6.0, 7.0],
            "y": [1.0, 5.0, 50.0, 6.0, 7.0],
            "feature_name": ["GeneA", "Blank-3", "GeneA", "NegControlProbe_1", "GeneB"],
        }
    )
    points = dd.from_pandas(frame, npartitions=2)

    xy = transcripts_in_windows(
        points,
        [(0.0, 10.0, 0.0, 10.0)],
        x_col="x",
        y_col="y",
        gene_col="feature_name",
    )

    np.testing.assert_allclose(xy, [[1.0, 1.0], [7.0, 7.0]])


def test_result_serialises_to_json() -> None:
    centres, transcripts = _tissue()

    result = _run(centres, transcripts, reference=centres)
    payload = json.loads(json.dumps(result.to_dict()))

    assert payload["status"] == "pass"
    assert payload["params"]["radius_um"] == 4.0
    assert len(payload["per_window"]) == result.n_windows


def _circles(xy: np.ndarray, radius: float = 4.0) -> np.ndarray:
    return shapely.buffer(shapely.points(xy), radius, quad_segs=4)


def _xenium_like_sdata(segmentation_offset: np.ndarray) -> SpatialData:
    """Xenium-style layout: micron points with qv and control probes."""
    centres, transcripts = _tissue(seed=3)
    rng = np.random.default_rng(4)
    n_tx = len(transcripts)
    genes = rng.choice(["GeneA", "GeneB", "GeneC"], size=n_tx).astype(object)
    controls = rng.random(n_tx) < 0.02
    genes[controls] = "NegControlProbe_00042"
    points = pd.DataFrame(
        {
            "x": transcripts[:, 0],
            "y": transcripts[:, 1],
            "z": rng.uniform(0.0, 10.0, n_tx),
            "feature_name": pd.Categorical(genes),
            "qv": rng.uniform(20.0, 40.0, n_tx),
            "assignment": np.zeros(n_tx, dtype=np.int64),
        }
    )
    platform_cells = gpd.GeoDataFrame(
        {"geometry": _circles(centres + rng.normal(0.0, 0.4, centres.shape))}
    )
    segmented = gpd.GeoDataFrame(
        {"geometry": _circles(centres + segmentation_offset, radius=5.0)}
    )
    return SpatialData(
        points={
            "transcripts": PointsModel.parse(
                points,
                coordinates={"x": "x", "y": "y", "z": "z"},
                feature_key="feature_name",
            )
        },
        shapes={
            "xenium_cell_boundaries": ShapesModel.parse(platform_cells),
            "MOSAIK_proseg": ShapesModel.parse(segmented),
        },
    )


def test_xenium_like_layer_passes_through_compute_dataset_qc(tmp_path: Path) -> None:
    zarr_path = tmp_path / "latest.zarr"
    _xenium_like_sdata(np.zeros(2)).write(zarr_path)

    result = compute_dataset_qc(
        zarr_path,
        "S1_XENIUM",
        shape_key="MOSAIK_proseg",
        registration_reference_shape_key="xenium_cell_boundaries",
    )
    paths = save_dataset_qc(result, tmp_path / "qc_out", "S1_XENIUM")

    summary = result["summary"]
    assert summary["registration_status"] == "pass"
    assert summary["registration_density_ratio"] > 2.0
    assert summary["registration_shift_um"] <= 2.0
    saved = json.loads(paths["registration_json"].read_text())
    assert saved["status"] == "pass"
    written = pd.read_csv(paths["summary_csv"])
    assert written.loc[0, "registration_status"] == "pass"


def test_platform_shapes_with_mosaic_affine_are_compared_in_microns() -> None:
    """MERSCOPE platform shapes carry a micron-to-pixel affine; points do not."""
    sdata = _xenium_like_sdata(np.zeros(2))
    reference = sdata.shapes["xenium_cell_boundaries"]
    sdata.shapes["merscope_cell_boundaries"] = ShapesModel.parse(
        gpd.GeoDataFrame({"geometry": reference.geometry.to_numpy()}),
        transformations={
            "global": Affine(
                np.array([[9.259, 0.0, 202.4], [0.0, 9.259, 1035.0], [0.0, 0.0, 1.0]]),
                input_axes=("x", "y"),
                output_axes=("x", "y"),
            )
        },
    )

    result = compute_segmentation_registration_qc(
        sdata,
        shape_key="MOSAIK_proseg",
        points_key="transcripts",
        reference_shape_key="merscope_cell_boundaries",
    )

    assert result.status == RegistrationStatus.PASS
    assert result.shift_um is not None and result.shift_um <= 2.0


def test_reference_equal_to_shape_key_skips_cross_correlation() -> None:
    sdata = _xenium_like_sdata(np.zeros(2))

    result = compute_segmentation_registration_qc(
        sdata,
        shape_key="xenium_cell_boundaries",
        points_key="transcripts",
        reference_shape_key="xenium_cell_boundaries",
    )

    assert result.status == RegistrationStatus.PASS
    assert result.shift_um is None


def _write_qc_config(tmp_path: Path, zarr_path: Path, *, strict: bool) -> Path:
    config_path = tmp_path / f"qc_config_{strict}.json"
    config_path.write_text(
        json.dumps(
            {
                "dataset_name": "S1_XENIUM",
                "latest_zarr_path": str(zarr_path),
                "output_dir": str(tmp_path / f"qc_out_{strict}"),
                "shape_key": "MOSAIK_proseg",
                "registration_reference_shape_key": "xenium_cell_boundaries",
                "registration_strict": strict,
            }
        )
    )
    return config_path


def test_qc_cli_warns_by_default_and_fails_when_strict(tmp_path: Path) -> None:
    zarr_path = tmp_path / "latest.zarr"
    _xenium_like_sdata(VZG2_OFFSET).write(zarr_path)
    runner = CliRunner()

    lenient = runner.invoke(
        qc_command,
        ["--config", str(_write_qc_config(tmp_path, zarr_path, strict=False))],
    )
    strict = runner.invoke(
        qc_command,
        ["--config", str(_write_qc_config(tmp_path, zarr_path, strict=True))],
    )

    assert lenient.exit_code == 0, lenient.output
    assert "registration check warning" in lenient.output
    saved = json.loads(
        (tmp_path / "qc_out_False" / "s1_xenium_registration_qc.json").read_text()
    )
    assert saved["status"] == "warn"
    assert strict.exit_code != 0
    assert "registration_strict" in strict.output
    # Outputs are still written before the strict failure.
    assert (tmp_path / "qc_out_True" / "s1_xenium_registration_qc.json").is_file()

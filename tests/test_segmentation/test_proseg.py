"""Tests for ProSeg subprocess orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import geopandas
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import shapely
import zarr

from merxen.path_utils import remove_path
from merxen.segmentation.proseg import (
    COORDINATE_SHIFT_SIDECAR,
    ProsegCoordinateShift,
    proseg_safe_affine,
    run_proseg_refinement,
)


class _FakePopen:
    """Minimal subprocess.Popen test double."""

    def __init__(self: _FakePopen, cmd: list[str], **_: object) -> None:
        self.cmd = cmd
        self.stdout = iter(["starting\n", "finished\n"])

    def wait(self: _FakePopen) -> int:
        return 0


def test_run_proseg_refinement_builds_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The wrapper should pass core flags and return the output path."""
    captured: dict[str, list[str]] = {}

    def _fake_check(_: str | Path) -> str:
        return "3.1.0"

    def _fake_popen(cmd: list[str], **kwargs: object) -> _FakePopen:
        del kwargs
        captured["cmd"] = cmd
        return _FakePopen(cmd)

    monkeypatch.setattr(
        "merxen.segmentation.proseg._check_proseg_available", _fake_check
    )
    monkeypatch.setattr("merxen.segmentation.proseg.subprocess.Popen", _fake_popen)

    output_path = tmp_path / "out.zarr"
    output_path.mkdir()
    transcripts = pd.DataFrame(
        {
            "transcript_id": [10, 11],
            "x": [1.0, 2.0],
            "y": [1.5, 2.5],
            "z": [0.0, 0.0],
            "feature_name": ["A", "B"],
            "cell_id": [0, 1],
            "qv": [0.95, 0.9],
        }
    )

    out = run_proseg_refinement(
        transcripts_df=transcripts,
        output_path=output_path,
        proseg_binary="/usr/bin/proseg",
        samples=10,
        num_threads=2,
        transcript_id_col="transcript_id",
        qv_col="qv",
        cellpose_masks=np.ones((2, 2), dtype=np.uint32),
        cellpose_cellprobs=np.asarray([[2.0, -1.0], [0.5, 3.0]], dtype=np.float32),
        cellpose_scale=1.0,
    )

    assert out == output_path
    assert "--output-spatialdata" in captured["cmd"]
    assert "--recorded-samples" in captured["cmd"]
    assert "--samples" in captured["cmd"]
    assert "--transcript-id-column" in captured["cmd"]
    assert "--qv-column" in captured["cmd"]
    assert "--cellpose-masks" in captured["cmd"]
    assert "--cellpose-cellprobs" in captured["cmd"]


def test_run_proseg_refinement_requires_cellpose_transform(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Supplying masks without a scale/affine should raise a ValueError."""
    monkeypatch.setattr(
        "merxen.segmentation.proseg._check_proseg_available",
        lambda _: "3.1.0",
    )

    transcripts = pd.DataFrame(
        {
            "x": [1.0],
            "y": [2.0],
            "z": [0.0],
            "feature_name": ["A"],
            "cell_id": [0],
        }
    )

    with pytest.raises(ValueError, match="Cellpose masks were supplied"):
        run_proseg_refinement(
            transcripts_df=transcripts,
            output_path=tmp_path / "out.zarr",
            proseg_binary="/usr/bin/proseg",
            cellpose_masks=np.zeros((2, 2), dtype=np.uint32),
        )


def test_run_proseg_refinement_validates_required_columns(tmp_path: Path) -> None:
    """Missing required transcript columns should fail fast."""
    transcripts = pd.DataFrame({"x": [1.0], "y": [2.0]})

    with pytest.raises(ValueError, match="Missing required columns"):
        run_proseg_refinement(
            transcripts_df=transcripts,
            output_path=tmp_path / "out.zarr",
            proseg_binary="/usr/bin/proseg",
        )


def test_proseg_safe_affine_keeps_non_negative_values_unchanged() -> None:
    """Non-negative affines must give the same command text as before."""
    x_transform = (0.108, 0.0, 12.5)
    y_transform = (0.0, 0.108, 1444.8)

    safe_x, safe_y, shift = proseg_safe_affine(x_transform, y_transform)

    assert shift.is_identity
    assert [str(v) for v in safe_x] == [str(v) for v in x_transform]
    assert [str(v) for v in safe_y] == [str(v) for v in y_transform]


def test_proseg_safe_affine_lifts_negative_translation() -> None:
    """A negative translation is lifted by whole 100 um steps on that axis only."""
    safe_x, safe_y, shift = proseg_safe_affine(
        (0.108, 0.0, -78.114), (0.0, 0.108, -250.0)
    )

    assert shift == ProsegCoordinateShift(x=100.0, y=300.0)
    assert safe_x == pytest.approx((0.108, 0.0, 21.886))
    assert safe_y == pytest.approx((0.0, 0.108, 50.0))


def test_proseg_safe_affine_zeroes_negative_zero_and_float_noise() -> None:
    """-0.0 and inversion noise would be read as flags, so they become 0.0."""
    safe_x, safe_y, shift = proseg_safe_affine(
        (0.108, -0.0, -0.0), (-1e-18, 0.108, 5.0)
    )

    assert shift.is_identity
    assert [str(v) for v in safe_x] == ["0.108", "0.0", "0.0"]
    assert [str(v) for v in safe_y] == ["0.0", "0.108", "5.0"]


def test_proseg_safe_affine_rejects_flipped_axis() -> None:
    """A flipped or rotated mask cannot be fixed by a translation."""
    with pytest.raises(ValueError, match="negative linear term"):
        proseg_safe_affine((-0.108, 0.0, 10.0), (0.0, 0.108, 5.0))


def _write_fake_proseg_store(transcript_csv: Path, store: Path) -> None:
    """Write a ProSeg-3-shaped store whose geometry follows the input coordinates.

    Like ProSeg, every output position is a function of the transcript
    coordinates it was given, so a shifted run differs only by the shift.
    """
    rows = pd.read_csv(transcript_csv)
    x = rows["x_micron"].to_numpy(np.float64)
    y = rows["y_micron"].to_numpy(np.float64)
    points = pa.table(
        {
            "transcript_id": pa.array(rows["transcript_id"], type=pa.uint64()),
            "x": pa.array((x + 0.25).astype(np.float32)),
            "y": pa.array((y - 0.5).astype(np.float32)),
            "z": pa.array(rows["z_micron"].to_numpy(np.float32)),
            "observed_x": pa.array(x.astype(np.float32)),
            "observed_y": pa.array(y.astype(np.float32)),
            "observed_z": pa.array(rows["z_micron"].to_numpy(np.float32)),
            "assignment": pa.array(rows["cell_id"].to_numpy(np.float64)),
        }
    )
    points_dir = store / "points" / "transcripts" / "points.parquet"
    points_dir.mkdir(parents=True)
    pq.write_table(points, points_dir / "part.0.parquet")

    cells = rows.loc[rows["cell_id"] > 0].groupby("cell_id")
    centroid_x = cells["x_micron"].mean().to_numpy(np.float64)
    centroid_y = cells["y_micron"].mean().to_numpy(np.float64)
    polygons = [
        shapely.MultiPolygon([shapely.box(cx - 2.0, cy - 1.5, cx + 2.0, cy + 1.5)])
        for cx, cy in zip(centroid_x, centroid_y, strict=True)
    ]
    shapes = pa.table(
        {
            "cell": pa.array(np.arange(len(polygons), dtype=np.uint32)),
            "geometry": pa.array(list(shapely.to_wkb(polygons)), type=pa.binary()),
        }
    )
    shapes_dir = store / "shapes" / "cell_boundaries"
    shapes_dir.mkdir(parents=True)
    # ProSeg puts the GeoParquet metadata at file level only, outside the
    # embedded Arrow schema.
    geo = {
        "primary_column": "geometry",
        "columns": {"geometry": {"encoding": "WKB", "geometry_types": []}},
        "version": "1.0.0",
    }
    with pq.ParquetWriter(shapes_dir / "shapes.parquet", shapes.schema) as writer:
        writer.write_table(shapes)
        writer.add_key_value_metadata({"geo": json.dumps(geo)})

    table = zarr.open_group(store / "tables" / "table", mode="w", zarr_format=2)
    obs = table.require_group("obs")
    obs.create_array("centroid_x", data=centroid_x.astype(np.float32))
    obs.create_array("centroid_y", data=centroid_y.astype(np.float32))
    obs.create_array("centroid_z", data=np.ones(len(polygons), dtype=np.float32))
    table.require_group("obsm").create_array(
        "spatial",
        data=np.column_stack([centroid_x, centroid_y]).astype(np.float32),
    )


def _read_fake_proseg_store(store: Path) -> dict[str, Any]:
    """Collect every coordinate the fake store holds."""
    table = zarr.open_group(store / "tables" / "table", mode="r")
    shapes = geopandas.read_parquet(
        store / "shapes" / "cell_boundaries" / "shapes.parquet"
    )
    return {
        "points": pq.read_table(
            store / "points" / "transcripts" / "points.parquet" / "part.0.parquet"
        ).to_pandas(),
        "polygons": shapes.geometry.to_numpy(),
        "centroid_x": np.asarray(table["obs/centroid_x"][:]),
        "centroid_y": np.asarray(table["obs/centroid_y"][:]),
        "centroid_z": np.asarray(table["obs/centroid_z"][:]),
        "spatial": np.asarray(table["obsm/spatial"][:]),
    }


def _write_negative_frame_transcripts(path: Path) -> None:
    """Write transcripts whose micron frame starts at negative x."""
    rng = np.random.default_rng(7)
    count = 60
    pd.DataFrame(
        {
            "transcript_id": np.arange(count),
            "x_micron": rng.uniform(-78.0, 40.0, count).round(5),
            "y_micron": rng.uniform(1450.0, 1500.0, count).round(5),
            "z_micron": rng.uniform(0.0, 6.0, count).round(3),
            "feature_name": rng.choice(["Gad1", "Slc17a7"], count),
            "cell_id": np.repeat(np.arange(6), 10),
        }
    ).to_csv(path, index=False)


def _install_fake_proseg(
    monkeypatch: pytest.MonkeyPatch,
    captured: dict[str, Any],
    write_store: Any = _write_fake_proseg_store,
) -> None:
    """Replace the ProSeg binary with a fake that rejects negative values."""

    def _fake_popen(cmd: list[str], **_: object) -> _FakePopen:
        for flag in ("--cellpose-x-transform", "--cellpose-y-transform"):
            start = cmd.index(flag) + 1
            values = cmd[start : start + 3]
            if any(value.startswith("-") for value in values):
                raise AssertionError(f"ProSeg would reject {flag} {values}")
        captured["cmd"] = cmd
        captured["proseg_csv"] = pd.read_csv(cmd[1], dtype=str)
        write_store(Path(cmd[1]), Path(cmd[cmd.index("--output-spatialdata") + 1]))
        return _FakePopen(cmd)

    monkeypatch.setattr(
        "merxen.segmentation.proseg._check_proseg_available", lambda _: "3.2.0"
    )
    monkeypatch.setattr("merxen.segmentation.proseg.subprocess.Popen", _fake_popen)


def _run_with_affine(
    transcripts_csv: Path,
    output_path: Path,
    x_transform: tuple[float, float, float],
    y_transform: tuple[float, float, float],
) -> Path:
    return run_proseg_refinement(
        transcripts_df=transcripts_csv,
        output_path=output_path,
        proseg_binary="/usr/local/bin/proseg",
        x_col="x_micron",
        y_col="y_micron",
        z_col="z_micron",
        transcript_id_col="transcript_id",
        samples=10,
        cellpose_masks=np.ones((4, 4), dtype=np.uint32),
        cellpose_x_transform=x_transform,
        cellpose_y_transform=y_transform,
        temp_root=output_path.parent / "tmp",
    )


def test_run_proseg_refinement_command_unchanged_without_negative_terms(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Non-negative affines pass the original CSV, values and output path."""
    captured: dict[str, Any] = {}
    _install_fake_proseg(monkeypatch, captured)
    transcripts_csv = tmp_path / "transcripts_for_proseg.csv"
    _write_negative_frame_transcripts(transcripts_csv)
    output_path = tmp_path / "proseg_base_raw.zarr"

    _run_with_affine(
        transcripts_csv, output_path, (0.108, 0.0, 12.5), (0.0, 0.108, 1444.8)
    )

    cmd = captured["cmd"]
    assert cmd[1] == str(transcripts_csv)
    x_start = cmd.index("--cellpose-x-transform") + 1
    y_start = cmd.index("--cellpose-y-transform") + 1
    assert cmd[x_start : x_start + 3] == ["0.108", "0.0", "12.5"]
    assert cmd[y_start : y_start + 3] == ["0.0", "0.108", "1444.8"]
    assert cmd[cmd.index("--output-spatialdata") + 1] == str(output_path)
    assert not (output_path / COORDINATE_SHIFT_SIDECAR).exists()


def test_run_proseg_refinement_negative_translation_round_trip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Outputs of a shifted run match an unshifted reference in the input frame."""
    captured: dict[str, Any] = {}
    _install_fake_proseg(monkeypatch, captured)
    transcripts_csv = tmp_path / "transcripts_for_proseg.csv"
    _write_negative_frame_transcripts(transcripts_csv)
    original_csv_bytes = transcripts_csv.read_bytes()
    reference_store = tmp_path / "reference.zarr"
    _write_fake_proseg_store(transcripts_csv, reference_store)
    output_path = tmp_path / "proseg_base_raw.zarr"

    out = _run_with_affine(
        transcripts_csv,
        output_path,
        (0.10799795369783013, 0.0, -78.11423964909234),
        (0.0, 0.1079976867408331, 1444.8012796051598),
    )

    assert out == output_path
    assert transcripts_csv.read_bytes() == original_csv_bytes
    original = pd.read_csv(transcripts_csv, dtype=str)
    proseg_csv = captured["proseg_csv"]
    np.testing.assert_allclose(
        proseg_csv["x_micron"].astype(float),
        original["x_micron"].astype(float) + 100.0,
    )
    for column in ("y_micron", "z_micron", "cell_id", "transcript_id"):
        assert proseg_csv[column].tolist() == original[column].tolist()

    restored = _read_fake_proseg_store(output_path)
    reference = _read_fake_proseg_store(reference_store)
    for column in ("x", "y", "observed_x", "observed_y"):
        np.testing.assert_allclose(
            restored["points"][column], reference["points"][column], atol=1e-4
        )
    for column in ("z", "observed_z", "transcript_id", "assignment"):
        np.testing.assert_array_equal(
            restored["points"][column], reference["points"][column]
        )
    assert restored["points"]["x"].dtype == np.float32
    assert all(shapely.equals_exact(restored["polygons"], reference["polygons"], 1e-9))
    for key in ("centroid_x", "centroid_y", "spatial"):
        np.testing.assert_allclose(restored[key], reference[key], atol=1e-4)
    np.testing.assert_array_equal(restored["centroid_z"], reference["centroid_z"])

    record = json.loads((output_path / COORDINATE_SHIFT_SIDECAR).read_text())
    assert record["shift_um"] == {"x": 100.0, "y": 0.0}
    assert record["proseg_x_transform"][2] == pytest.approx(21.88576035090766)
    assert not (tmp_path / ".proseg_base_raw.zarr.proseg-shifted").exists()


def test_run_proseg_refinement_leaves_no_output_when_restore_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A store that cannot be restored must not be published as the output."""

    def _write_store_without_shapes(transcript_csv: Path, store: Path) -> None:
        _write_fake_proseg_store(transcript_csv, store)
        remove_path(store / "shapes")

    _install_fake_proseg(monkeypatch, {}, _write_store_without_shapes)
    transcripts_csv = tmp_path / "transcripts_for_proseg.csv"
    _write_negative_frame_transcripts(transcripts_csv)
    output_path = tmp_path / "proseg_base_raw.zarr"

    with pytest.raises(RuntimeError, match="lacks cell polygons"):
        _run_with_affine(
            transcripts_csv, output_path, (0.108, 0.0, -78.1), (0.0, 0.108, 5.0)
        )

    assert not output_path.exists()
    assert not (tmp_path / ".proseg_base_raw.zarr.proseg-shifted").exists()

"""Synthetic end to end of registration through the robust landmark field.

One pair is rendered with a 40 um / 3 mm warp (the field recovers it, a similarity
cannot) and labelled with one gross misclick; the reference crop is cut so that the
preview store has a non-zero SpatialData offset. The runs: ``preview`` (the preset
default, windows off), ``none``, ``full`` and with tissue windows.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import spatialdata as sd
from scipy.ndimage import gaussian_filter, map_coordinates
from spatialdata.transformations import get_transformation

from merxen.image_registration import pull_points_through_registration
from merxen.io.spatialdata_io import spatialdata_write_lock
from merxen.manual_registration import field_fit
from merxen.manual_registration import register as register_module
from merxen.manual_registration.compare import compare_runs
from merxen.manual_registration.config import (
    ManualRegistrationCompareConfig,
    ManualRegistrationImageInput,
    ManualRegistrationPrepareConfig,
    ManualRegistrationRunConfig,
)
from merxen.manual_registration.geometry import coarse_from_matrix
from merxen.manual_registration.landmark_field import LandmarkField
from merxen.manual_registration.prepare import run_manual_registration_prepare
from merxen.manual_registration.register import run_manual_registration
from merxen.manual_registration.schema import (
    LABELLING_ATTR,
    RESULT_ATTR,
    STORE_ATTR,
    CoarseTransform,
    CropRect,
    LandmarkPair,
    parse_store_attrs,
    read_root_attrs,
    write_root_attr,
)
from merxen.manual_registration.simulate import (
    SyntheticPair,
    inject_labelling,
    make_synthetic_pair,
    sample_tissue_points,
    simulate_landmark_pairs,
)
from merxen.viewer_cache.format import derived_image_pyramid_cache_key

from .test_check import NAME, P_MOV, P_REF, ROLES, WORKING_TO_LEVEL0, _moving_level

pytestmark = pytest.mark.integration

CROP = CropRect(x0=128, y0=64, x1=1984, y1=1984)
MISCLICK_ID = 5
# The preview level: the coarsest reference level not coarser than 5.5 um / 2 um.
PREVIEW_FACTOR = 2


@dataclass
class Prepared:
    """A prepared synthetic pair with its labelling."""

    root: Path
    pair: SyntheticPair
    store: Path
    pairs: list[LandmarkPair]


def _prepare(root: Path, moving_channel_names: list[str] | None = None) -> Prepared:
    pair = make_synthetic_pair(
        root / "images",
        channel_names=("DAPI", "AF568", "AF647"),
        independent_nuclei=True,
        warp_amplitude_um=40.0,
        warp_wavelength_um=3000.0,
        rotation_deg=178.0,
        offset_um=(40.0, -25.0),
        seed=3,
    )
    outputs = run_manual_registration_prepare(
        ManualRegistrationPrepareConfig(
            dataset_name=NAME,
            reference=ManualRegistrationImageInput(path=pair.reference_path),
            moving=ManualRegistrationImageInput(
                path=pair.moving_path, channel_names=moving_channel_names
            ),
            prepared_dir=root / NAME / "prepared",
            output_dir=root / "prepare_out",
            workers=1,
        )
    )
    points = sample_tissue_points(
        _moving_level(pair, 1),
        WORKING_TO_LEVEL0,
        n=36,
        seed=1,
        min_separation_px=150.0 / P_MOV,
    )
    pairs = simulate_landmark_pairs(
        points,
        pair.moving_to_reference,
        noise_um=2.0,
        reference_pixel_size_um=P_REF,
        seed=1,
        outliers={MISCLICK_ID: (1000.0, 0.0)},
        disabled=(35, 36),
    )
    fitted = coarse_from_matrix(
        pair.true_matrix,
        moving_shape_yx=pair.moving_shape_yx,
        reference_shape_yx=pair.reference_shape_yx,
        moving_pixel_size_um=P_MOV,
        reference_pixel_size_um=P_REF,
    )[0]
    coarse = CoarseTransform(
        rotation_deg=fitted.rotation_deg + 2.0,
        mirror_x=fitted.mirror_x,
        offset_um=(fitted.offset_um[0] + 50.0, fitted.offset_um[1] - 30.0),
    )
    inject_labelling(
        outputs["store"],
        pairs,
        coarse=coarse,
        crops={"reference": CROP, "moving": None},
    )
    return Prepared(root, pair, outputs["store"], pairs)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory) -> Prepared:
    return _prepare(tmp_path_factory.mktemp("manual_registration_field"))


def _config(
    prepared: Prepared, result: Path, out: Path, **overrides: Any
) -> ManualRegistrationRunConfig:
    values: dict[str, Any] = {
        "dataset_name": NAME,
        "preset": "cortex_serial",
        "prepared_store_path": prepared.store,
        "result_dir": result,
        "output_dir": out,
        "channel_roles": ROLES,
        "row_settings": "field_window_observations=false",
    }
    values.update(overrides)
    return ManualRegistrationRunConfig(**values)


def _summary(outputs: dict[str, Path]) -> dict[str, Any]:
    document: dict[str, Any] = json.loads(outputs["summary_json"].read_text())
    return document


@pytest.fixture(scope="module")
def preview(
    prepared: Prepared, tmp_path_factory: pytest.TempPathFactory
) -> tuple[Path, dict[str, Path], dict[str, Any]]:
    base = tmp_path_factory.mktemp("field_preview")
    config = _config(prepared, base / "result", base / "out")
    outputs = run_manual_registration(config)
    return base, outputs, _summary(outputs)


def _element(store: Path, key: str) -> Any:
    return sd.read_zarr(store, selection=("images",)).images[key]


def _base_array(element: Any) -> np.ndarray:
    data = element.data if hasattr(element, "data") else element
    return np.asarray(data.compute() if hasattr(data, "compute") else data)


# ---------------------------------------------------------------------------
# preview
# ---------------------------------------------------------------------------


def test_preview_store_elements_follow_the_materialization_contract(
    preview: tuple[Path, dict[str, Path], dict[str, Any]], prepared: Prepared
) -> None:
    _base, outputs, summary = preview
    store = outputs["result_store"]
    shape = (CROP.height() // PREVIEW_FACTOR, CROP.width() // PREVIEW_FACTOR)
    attrs = read_root_attrs(store)
    parsed = parse_store_attrs(attrs[STORE_ATTR])
    for role, key in (("reference", "reference"), ("moving", "moving_registered")):
        record = parsed.images[role]
        assert record.materialization is not None
        assert record.materialization.mode == "preview"
        assert record.materialization.level_factor == PREVIEW_FACTOR
        assert record.materialization.pixel_size_um == pytest.approx(
            PREVIEW_FACTOR * P_REF
        )
        # The record keeps the level-0 pixel size and the crop (contract C4).
        assert record.pixel_size_um == P_REF
        assert record.shape_yx == shape
        element = _element(store, key)
        assert tuple(element.shape[1:]) == shape
        matrix = get_transformation(element, "global").to_affine_matrix(
            ("x", "y"), ("x", "y")
        )
        np.testing.assert_allclose(
            matrix,
            [
                [PREVIEW_FACTOR * P_REF, 0.0, P_REF * CROP.x0],
                [0.0, PREVIEW_FACTOR * P_REF, P_REF * CROP.y0],
                [0.0, 0.0, 1.0],
            ],
        )
    assert parsed.images["reference"].crop == CROP
    assert summary["field"]["materialize"] == {
        "mode": "preview",
        "level_factor": PREVIEW_FACTOR,
        "pixel_size_um": PREVIEW_FACTOR * P_REF,
    }
    assert attrs[RESULT_ATTR]["provenance_digest"] == summary["provenance_digest"]


def test_preview_reference_is_the_block_mean_of_the_level_zero_crop(
    preview: tuple[Path, dict[str, Path], dict[str, Any]], prepared: Prepared
) -> None:
    _base, outputs, _summary_ = preview
    level_zero = _base_array(_element(prepared.store, "reference"))[
        :, CROP.y0 : CROP.y1, CROP.x0 : CROP.x1
    ].astype(np.float64)
    channels, height, width = level_zero.shape
    blocks = level_zero.reshape(
        channels,
        height // PREVIEW_FACTOR,
        PREVIEW_FACTOR,
        width // PREVIEW_FACTOR,
        PREVIEW_FACTOR,
    ).mean(axis=(2, 4))
    stored = _base_array(_element(outputs["result_store"], "reference")).astype(
        np.float64
    )
    # The OME-TIFF level is the integer mean (sum + 2) // 4, so it differs from the
    # float mean by less than one grey level.
    assert np.abs(stored - blocks).max() <= 1.0


def test_preview_registration_recovers_the_warp_the_similarity_cannot(
    preview: tuple[Path, dict[str, Path], dict[str, Any]],
) -> None:
    _base, outputs, summary = preview
    rows = pd.read_csv(outputs["landmarks_csv"])
    clean = rows[
        rows["enabled"] & (rows["field_class"] == "ok") & ~rows["outside_crop"]
    ]
    assert len(clean) >= 20
    assert clean["residual_final_um"].max() <= 10.0
    assert (
        clean["residual_final_um"].median() < 0.5 * clean["residual_affine_um"].median()
    )
    assert clean["final_includes_flow"].all() and clean["push_converged"].all()
    misclick = rows[rows["id"] == MISCLICK_ID].iloc[0]
    assert misclick["field_class"] == "likely_misclick"
    assert bool(misclick["outlier"]) is True
    assert bool(misclick["used_in_fit"]) is True
    disabled = rows[~rows["enabled"]]
    assert (disabled["field_class"] == "disabled").all()
    assert disabled["field_weight"].isna().all()
    pearson = summary["tissue_qc"]["pearson"]
    assert pearson["final"] > pearson["affine"] > pearson["coarse"]
    assert summary["field"]["flow"]["max_displacement_um"] > 10.0
    # Contract C5: records, matrices, refinement and nonrigid blocks.
    first = summary["landmarks"][0]
    assert set(first["field"]) == {
        "weight",
        "p_value",
        "clicks_only",
        "verdict",
        "class",
        "loo_vector_moving_um",
    }
    assert summary["refinement"] == {
        "status": "disabled",
        "reason": "cortex field run",
    }
    assert summary["nonrigid"]["enabled"] is False
    assert (
        summary["matrices"]["field_base_full"]
        == summary["matrices"]["final_affine_full"]
    )
    assert summary["algorithm"]["landmark_field"] == "robust_gp_matern52_v1"
    assert summary["algorithm"]["image_evidence"] == "image_evidence_bp20_v1"
    assert summary["fit"]["model"] == "similarity"


def test_preview_tissue_qc_has_a_block_per_role(
    preview: tuple[Path, dict[str, Path], dict[str, Any]],
) -> None:
    _base, outputs, summary = preview
    by_role = summary["tissue_qc"]["by_role"]
    assert set(by_role) == {"at8", "map2", "dapi"}
    assert by_role["map2"]["channel"] == "AF647"
    for block in by_role.values():
        assert set(block["pearson"]) == {"coarse", "landmark", "affine", "final"}
        assert block["pearson"]["final"] > block["pearson"]["affine"]
    assert outputs["field_qc_png"].exists()
    assert outputs["field_qc_png"].with_suffix(".pdf").exists()
    assert outputs["tissue_qc_png"].exists()


def test_the_model_json_reproduces_its_digest(
    preview: tuple[Path, dict[str, Path], dict[str, Any]],
) -> None:
    _base, outputs, summary = preview
    payload = json.loads(outputs["landmark_field_json"].read_text())
    field = LandmarkField.from_dict(payload)
    assert field.digest == summary["field"]["model_digest"] == payload["digest"]
    assert summary["field"]["model_json"] == outputs["landmark_field_json"].name


def test_the_flow_file_pulls_like_the_pull_function_and_the_field(
    preview: tuple[Path, dict[str, Path], dict[str, Any]], prepared: Prepared
) -> None:
    _base, outputs, summary = preview
    flow = np.load(outputs["field_flow_npz"])
    assert set(flow.files) == {
        "flow_yx",
        "working_to_reference",
        "base_matrix_crop",
        "base_matrix_full",
        "reference_origin_xy",
        "moving_origin_xy",
        "sd_um",
        "sd_working_to_reference",
        "algorithm",
        "model_digest",
    }
    assert flow["flow_yx"].dtype == np.float32
    assert str(flow["model_digest"]) == summary["field"]["model_digest"]
    assert str(flow["algorithm"]) == summary["algorithm"]["landmark_field"]
    np.testing.assert_array_equal(flow["reference_origin_xy"], [CROP.x0, CROP.y0])
    np.testing.assert_allclose(
        flow["base_matrix_full"], summary["matrices"]["final_affine_full"]
    )
    assert summary["field"]["flow"]["shape_yx"] == list(flow["flow_yx"].shape[1:])
    rng = np.random.default_rng(0)
    points = rng.uniform([0, 0], [CROP.width() - 1, CROP.height() - 1], (500, 2))
    pulled = pull_points_through_registration(
        points,
        final_matrix=flow["base_matrix_crop"],
        flow_yx=flow["flow_yx"],
        working_to_reference=flow["working_to_reference"],
    )
    field = LandmarkField.from_dict(
        json.loads(outputs["landmark_field_json"].read_text())
    )
    origin_reference = flow["reference_origin_xy"]
    expected = (
        field.mean(P_REF * (points + origin_reference)) / P_MOV
        - flow["moving_origin_xy"]
    )
    assert np.abs(pulled - expected).max() < 0.05
    assert flow["sd_um"].dtype == np.float32 and np.isfinite(flow["sd_um"]).all()
    assert summary["field"]["sd_um"] is not None


def test_a_rerun_is_reused_and_leaves_the_store_alone(
    preview: tuple[Path, dict[str, Path], dict[str, Any]], prepared: Prepared
) -> None:
    base, outputs, summary = preview
    store = outputs["result_store"]
    before = (store / "zarr.json").read_bytes()
    again = run_manual_registration(_config(prepared, base / "result", base / "out2"))
    task = json.loads(again["task_summary_json"].read_text())
    assert task["reused"] is True
    assert task["provenance_digest"] == summary["provenance_digest"]
    assert again["result_store"] == store
    assert again["landmark_field_json"] == outputs["landmark_field_json"]
    assert again["field_flow_npz"] == outputs["field_flow_npz"]
    assert (store / "zarr.json").read_bytes() == before


def test_a_changed_field_setting_registers_again(
    preview: tuple[Path, dict[str, Path], dict[str, Any]], prepared: Prepared
) -> None:
    base, _outputs, summary = preview
    config = _config(
        prepared,
        base / "other_result",
        base / "out3",
        row_settings="field_window_observations=false;field_outlier_p=0.02",
    )
    summary_other = _summary(run_manual_registration(config))
    assert summary_other["provenance_digest"] != summary["provenance_digest"]
    assert "reused" not in summary_other


# ---------------------------------------------------------------------------
# none and full
# ---------------------------------------------------------------------------


def test_materialize_none_writes_no_store_and_reuses_from_the_summary(
    prepared: Prepared, tmp_path: Path
) -> None:
    result = tmp_path / "result"
    config = _config(
        prepared,
        result,
        tmp_path / "out",
        row_settings="field_window_observations=false;materialize=none",
    )
    outputs = run_manual_registration(config)
    summary = _summary(outputs)
    assert "result_store" not in outputs
    assert not any(path.is_dir() for path in result.iterdir())
    assert "result_store" not in summary["outputs"]
    assert summary["field"]["materialize"] == {
        "mode": "none",
        "level_factor": None,
        "pixel_size_um": None,
    }
    assert (
        summary["tissue_qc"]["pearson"]["final"]
        > summary["tissue_qc"]["pearson"]["affine"]
    )
    again = run_manual_registration(
        config.model_copy(update={"output_dir": tmp_path / "o2"})
    )
    assert json.loads(again["task_summary_json"].read_text())["reused"] is True
    assert "result_store" not in again
    # force registers again.
    forced = run_manual_registration(config, force_rerun=True)
    assert "reused" not in json.loads(forced["task_summary_json"].read_text())


def test_materialize_none_supersedes_and_removes_an_older_store(
    prepared: Prepared, tmp_path: Path
) -> None:
    result = tmp_path / "result"
    first = run_manual_registration(_config(prepared, result, tmp_path / "out"))
    store = first["result_store"]
    assert store.exists()
    run_manual_registration(
        _config(
            prepared,
            result,
            tmp_path / "out2",
            row_settings="field_window_observations=false;materialize=none",
        )
    )
    assert not store.exists()
    assert not any(path.is_dir() for path in result.iterdir())


def test_full_materialization_agrees_with_the_preview(
    preview: tuple[Path, dict[str, Path], dict[str, Any]],
    prepared: Prepared,
    tmp_path: Path,
) -> None:
    _base, preview_outputs, _preview_summary = preview
    config = _config(
        prepared,
        tmp_path / "result",
        tmp_path / "out",
        row_settings="field_window_observations=false;materialize=full",
    )
    outputs = run_manual_registration(config)
    summary = _summary(outputs)
    assert summary["field"]["materialize"]["mode"] == "full"
    assert summary["field"]["materialize"]["level_factor"] == 1
    parsed = parse_store_attrs(read_root_attrs(outputs["result_store"])[STORE_ATTR])
    assert parsed.images["moving"].materialization is not None
    assert parsed.images["moving"].materialization.level_factor == 1
    images = sd.read_zarr(outputs["result_store"], selection=("images",)).images
    for key in ("reference", "moving_registered"):
        # The viewer pyramids of both elements (the comparison overview reads them).
        assert derived_image_pyramid_cache_key(key, 4) in images
    full = _base_array(_element(outputs["result_store"], "moving_registered"))
    assert full.shape[1:] == (CROP.height(), CROP.width())
    matrix = get_transformation(
        _element(outputs["result_store"], "moving_registered"), "global"
    ).to_affine_matrix(("x", "y"), ("x", "y"))
    np.testing.assert_allclose(matrix[:2, 2], [P_REF * CROP.x0, P_REF * CROP.y0])
    assert matrix[0, 0] == P_REF
    channels, height, width = full.shape
    blocks = (
        full.astype(np.float64)
        .reshape(
            channels,
            height // PREVIEW_FACTOR,
            PREVIEW_FACTOR,
            width // PREVIEW_FACTOR,
            PREVIEW_FACTOR,
        )
        .mean(axis=(2, 4))
    )
    coarse = _base_array(
        _element(preview_outputs["result_store"], "moving_registered")
    ).astype(np.float64)
    inside = (blocks > 0) & (coarse > 0)
    assert inside.mean() > 0.3
    correlation = np.corrcoef(blocks[inside], coarse[inside])[0, 1]
    assert correlation > 0.97
    assert np.abs(blocks[inside] - coarse[inside]).mean() < 0.1 * blocks[inside].mean()


# ---------------------------------------------------------------------------
# windows, labelling payload, errors
# ---------------------------------------------------------------------------


def test_windows_enter_the_registered_field(
    prepared: Prepared, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(field_fit, "WINDOW_SIZES_UM", (200.0, 300.0))
    config = _config(
        prepared,
        tmp_path / "result",
        tmp_path / "out",
        row_settings=(
            "field_window_observations=true;field_window_step_um=150;"
            "field_window_search_um=150"
        ),
    )
    summary = _summary(run_manual_registration(config))
    block = summary["field"]
    assert block["windows"] is not None
    assert block["n_windows"] == block["n_observations"].get("window", 0)
    assert block["loo_um_clicks_only"] is not None or block["n_windows"] == 0
    assert summary["algorithm"]["image_evidence"] == "image_evidence_bp20_v1"
    assert (
        summary["tissue_qc"]["pearson"]["final"]
        > summary["tissue_qc"]["pearson"]["affine"]
    )


def test_a_labelling_payload_equal_to_the_stores_gives_the_same_result(
    preview: tuple[Path, dict[str, Path], dict[str, Any]], prepared: Prepared
) -> None:
    base, outputs, summary = preview
    payload = read_root_attrs(prepared.store)[LABELLING_ATTR]
    again = run_manual_registration(
        _config(prepared, base / "result", base / "out4"), labelling_payload=payload
    )
    task = json.loads(again["task_summary_json"].read_text())
    assert task["reused"] is True
    assert task["provenance_digest"] == summary["provenance_digest"]


def test_a_labelling_payload_replaces_the_stores_labelling(
    preview: tuple[Path, dict[str, Path], dict[str, Any]],
    prepared: Prepared,
    tmp_path: Path,
) -> None:
    _base, _outputs, summary = preview
    payload = json.loads(json.dumps(read_root_attrs(prepared.store)[LABELLING_ATTR]))
    for pair in payload["landmarks"]["pairs"]:
        if pair["id"] == MISCLICK_ID:
            pair["enabled"] = False
    payload["revision"] += 1
    config = _config(prepared, tmp_path / "result", tmp_path / "out")
    other = _summary(run_manual_registration(config, labelling_payload=payload))
    assert other["provenance_digest"] != summary["provenance_digest"]
    assert other["labelling"]["revision"] == payload["revision"]
    record = next(r for r in other["landmarks"] if r["id"] == MISCLICK_ID)
    assert record["enabled"] is False and record["field"]["class"] == "disabled"


def test_a_draft_labelling_payload_is_refused(
    prepared: Prepared, tmp_path: Path
) -> None:
    payload = json.loads(json.dumps(read_root_attrs(prepared.store)[LABELLING_ATTR]))
    payload["status"] = "draft"
    with pytest.raises(ValueError, match="draft"):
        run_manual_registration(
            _config(prepared, tmp_path / "result", tmp_path / "out"),
            labelling_payload=payload,
        )
    assert not (tmp_path / "result").exists()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"channel_roles": ""}, "preset cortex_serial needs channel roles"),
        (
            {"row_settings": "field_window_observations=false;refine_affine=true"},
            "skip the DAPI refinement",
        ),
        (
            {
                "preset": "same_section",
                "row_settings": "materialize=preview",
                "channel_roles": "",
            },
            "only for robust-field runs",
        ),
    ],
)
def test_register_refuses_bad_field_settings_before_writing(
    prepared: Prepared, tmp_path: Path, overrides: dict[str, Any], message: str
) -> None:
    config = _config(prepared, tmp_path / "result", tmp_path / "out", **overrides)
    with pytest.raises(ValueError, match=message):
        run_manual_registration(config)
    assert not any(p.suffix == ".zarr" for p in (tmp_path / "result").iterdir())


def test_the_prepared_store_is_not_written(
    prepared: Prepared, preview: tuple[Path, dict[str, Path], dict[str, Any]]
) -> None:
    attrs = read_root_attrs(prepared.store)
    assert "merxen_manual_registration_result" not in attrs


def _strip_to_the_pre_field_layout(outputs: dict[str, Path]) -> None:
    """Rewrite a similarity result the way the code before the field wrote it."""

    def strip(document: dict[str, Any]) -> dict[str, Any]:
        document.pop("field", None)
        for record in document["landmarks"]:
            record.pop("field", None)
        return document

    summary_path = outputs["summary_json"]
    summary_path.write_text(
        json.dumps(strip(json.loads(summary_path.read_text())), indent=2) + "\n"
    )
    store = outputs["result_store"]
    attr = strip(dict(read_root_attrs(store)[RESULT_ATTR]))
    with spatialdata_write_lock(store):
        write_root_attr(store, RESULT_ATTR, attr)
    frame = pd.read_csv(outputs["landmarks_csv"])
    frame.drop(columns=[c for c in frame.columns if c.startswith("field_")]).to_csv(
        outputs["landmarks_csv"], index=False
    )


def test_a_result_from_before_the_field_is_reused_and_compared(
    preview: tuple[Path, dict[str, Path], dict[str, Any]],
    prepared: Prepared,
    tmp_path: Path,
) -> None:
    _base, field_outputs, _field_summary = preview
    config = _config(
        prepared,
        tmp_path / "result",
        tmp_path / "out",
        preset="same_section",
        channel_roles="",
        row_settings="refine_affine=false;nonrigid_enabled=false",
    )
    outputs = run_manual_registration(config)
    first = _summary(outputs)
    assert first["field"] is None
    assert all(record["field"] is None for record in first["landmarks"])
    csv_rows = pd.read_csv(outputs["landmarks_csv"])
    assert csv_rows["field_class"].isna().all()
    _strip_to_the_pre_field_layout(outputs)
    again = run_manual_registration(
        config.model_copy(update={"output_dir": tmp_path / "out2"})
    )
    task = json.loads(again["task_summary_json"].read_text())
    assert task["reused"] is True
    assert "field" not in task
    assert again["result_store"] == outputs["result_store"]

    compared = compare_runs(
        ManualRegistrationCompareConfig(
            dataset_name=NAME,
            summary_paths=[again["summary_json"], field_outputs["summary_json"]],
            output_dir=tmp_path / "durable",
            task_output_dir=tmp_path / "task",
        )
    )
    rows = {
        row["run_id"]: row
        for row in csv.DictReader(compared["comparison_csv"].read_text().splitlines())
    }
    old, new = rows[f"{NAME}_same_section"], rows[f"{NAME}_cortex_serial"]
    assert old["materialize_mode"] == "full" and old["field_model"] == ""
    assert old["field_n_windows"] == "" and old["field_loo_median_um"] == ""
    assert new["materialize_mode"] == "preview" and new["field_model"] == "robust_gp"
    assert float(new["field_loo_median_um"]) > 0.0
    assert new["field_n_windows"] == "0"
    assert compared["comparison_png"].exists()


# ---------------------------------------------------------------------------
# pixels and channels
# ---------------------------------------------------------------------------


def test_stored_moving_image_is_the_moving_level_pulled_through_the_field(
    preview: tuple[Path, dict[str, Path], dict[str, Any]], prepared: Prepared
) -> None:
    _base, outputs, _summary_ = preview
    flow = np.load(outputs["field_flow_npz"])
    stored = _base_array(_element(outputs["result_store"], "moving_registered"))[0]
    reference = _base_array(_element(outputs["result_store"], "reference"))[0]
    height, width = stored.shape
    rows, cols = np.meshgrid(
        np.arange(0, height, 4), np.arange(0, width, 4), indexing="ij"
    )
    # Preview pixel centre -> reference crop level-0 pixel (factor 2).
    points = np.stack(
        [cols.ravel() * PREVIEW_FACTOR + 0.5, rows.ravel() * PREVIEW_FACTOR + 0.5], 1
    ).astype(np.float64)
    moving = gaussian_filter(_moving_level(prepared.pair, 0), 1.5)

    def sample(matrix: np.ndarray, flow_yx: np.ndarray | None) -> np.ndarray:
        if flow_yx is None:
            xy = points @ matrix[:2, :2].T + matrix[:2, 2]
        else:
            xy = pull_points_through_registration(
                points,
                final_matrix=matrix,
                flow_yx=flow_yx,
                working_to_reference=flow["working_to_reference"],
            )
        xy = xy + flow["moving_origin_xy"]
        return map_coordinates(moving, [xy[:, 1], xy[:, 0]], order=1, cval=-1.0)

    with_flow = sample(flow["base_matrix_crop"], flow["flow_yx"])
    affine_only = sample(flow["base_matrix_crop"], None)
    stored_smooth = gaussian_filter(stored.astype(np.float64), 1.5)[::4, ::4].ravel()
    inside = (with_flow >= 0) & (affine_only >= 0) & (stored_smooth > 0)
    assert inside.mean() > 0.3
    # Registered pixels equal the moving level sampled at the composed pull (up to
    # the block-mean level and interpolation: 0.88 against 0.01
    # measured) and not the base affine alone.
    with_flow_fit = np.corrcoef(with_flow[inside], stored_smooth[inside])[0, 1]
    affine_fit = np.corrcoef(affine_only[inside], stored_smooth[inside])[0, 1]
    assert with_flow_fit > 0.8
    assert with_flow_fit > affine_fit + 0.5
    assert reference.shape == stored.shape


def test_preview_keeps_only_the_channels_prepare_kept(tmp_path: Path) -> None:
    # The moving image lacks the reference's AF647 stain, so prepare keeps two
    # channels while the reference OME-TIFF still holds three.
    prepared_missing = _prepare(tmp_path, ["DAPI", "AF568", "AF488"])
    config = _config(
        prepared_missing,
        tmp_path / "result",
        tmp_path / "out",
        channel_roles="at8=AF568;dapi=DAPI;reference:map2=AF647;moving:map2=AF488",
    )
    outputs = run_manual_registration(config)
    element = _element(outputs["result_store"], "reference")
    assert element.shape[0] == 2
    assert list(element.coords["c"].values) == ["DAPI", "AF568"]
    # The moving element keeps all of the moving image's own channels.
    moving_element = _element(outputs["result_store"], "moving_registered")
    assert list(moving_element.coords["c"].values) == ["DAPI", "AF568", "AF488"]


def test_materialize_none_is_not_reused_after_an_interrupted_publish(
    prepared: Prepared, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = "field_window_observations=false;materialize=none"
    result = tmp_path / "result"
    config = _config(prepared, result, tmp_path / "out", row_settings=settings)
    first = run_manual_registration(config)
    summary_before = first["summary_json"].read_text()
    payload = json.loads(json.dumps(read_root_attrs(prepared.store)[LABELLING_ATTR]))
    for pair in payload["landmarks"]["pairs"]:
        if pair["id"] == MISCLICK_ID:
            pair["enabled"] = False
    payload["revision"] += 1

    real_publish = register_module._publish

    def publish_without_the_summary(source: Path, target: Path) -> Path:
        if target.name.endswith("_manual_registration_summary.json"):
            raise OSError("simulated kill before the summary")
        return real_publish(source, target)

    with monkeypatch.context() as patch:
        patch.setattr(register_module, "_publish", publish_without_the_summary)
        with pytest.raises(OSError, match="simulated kill"):
            run_manual_registration(
                config.model_copy(update={"output_dir": tmp_path / "out_b"}),
                labelling_payload=payload,
            )
    assert first["summary_json"].read_text() == summary_before

    # The first run's inputs again: the CSV on disk is the interrupted run's, so
    # the summary is no proof and the run registers again.
    again = run_manual_registration(
        config.model_copy(update={"output_dir": tmp_path / "out_c"})
    )
    assert "reused" not in json.loads(again["task_summary_json"].read_text())
    rows = pd.read_csv(again["landmarks_csv"])
    assert bool(rows[rows["id"] == MISCLICK_ID].iloc[0]["enabled"]) is True

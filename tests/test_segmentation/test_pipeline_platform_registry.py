"""Platform-registry lookups in the segmentation pipeline.

``run_proseg_segmentation`` passes the platform's quality-column alias to every
store conversion, and ``_preflight_model_selection`` checks custom arms against
the platform's native image. Both values come from
``merxen.platforms.platform_spec``; these tests pin them per platform.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from merxen.config import SegmentationConfig
from merxen.segmentation import pipeline

_PLATFORMS = [("MERSCOPE", "transcript_score"), ("XENIUM", None)]
_NATIVE = [("MERSCOPE", "MERSCOPE_z_projection"), ("XENIUM", "morphology_focus")]


def _config(
    tmp_path: Path,
    platform: str,
    *,
    image_key: str | None = None,
    experimental_arm: bool = False,
) -> SegmentationConfig:
    """Return a minimal segmentation config whose outputs live under tmp_path."""
    dataset: dict[str, Any] = {
        "name": f"DS_{platform}",
        "platform": platform,
        "data_path": str(tmp_path / "source.zarr"),
        "channels": ["DAPI"],
        "output_dir": str(tmp_path / "work" / "segment_out"),
    }
    if image_key is not None:
        dataset["image_key"] = image_key
    return SegmentationConfig.model_validate(
        {
            "dataset": dataset,
            "proseg_hybrid": {"enabled": False},
            "cellpose_model_policy": {
                "experimental_arm": experimental_arm,
                "protected_output_roots": [str(tmp_path / "protected")],
            },
        }
    )


@pytest.fixture
def conversions(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, Any]]]:
    """Stub the ProSeg run and record every store conversion's keyword arguments."""
    calls: list[tuple[str, dict[str, Any]]] = []

    def _upgrade(path: Path, **kwargs: Any) -> bool:
        calls.append(("upgrade_spatialdata_contract", kwargs))
        return False

    def _convert(raw: Path, latest: Path, **kwargs: Any) -> Path:
        calls.append(("convert_to_latest_zarr", kwargs))
        return Path(latest)

    def _proseg(**kwargs: Any) -> Path:
        raw = Path(kwargs["output_path"])
        raw.mkdir(parents=True, exist_ok=True)
        return raw

    def _noop(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(pipeline, "upgrade_spatialdata_contract", _upgrade)
    monkeypatch.setattr(pipeline, "convert_to_latest_zarr", _convert)
    monkeypatch.setattr(pipeline, "run_proseg_refinement", _proseg)
    for name in (
        "require_matching_cached_model",
        "check_store_segmentation_sidecar",
        "write_store_segmentation_sidecar",
        "force_release",
        "log_status",
    ):
        monkeypatch.setattr(pipeline, name, _noop)
    return calls


def _run_proseg(tmp_path: Path, config: SegmentationConfig) -> None:
    """Run the ProSeg stage with placeholder inputs."""
    transcripts = tmp_path / "transcripts.csv"
    transcripts.write_text("x_micron,y_micron,z_micron,feature_name,cell_id\n")
    transforms = tmp_path / "transforms.json"
    transforms.write_text(
        json.dumps({"x_transform": [1.0, 0.0, 0.0], "y_transform": [0.0, 1.0, 0.0]})
    )
    pipeline.run_proseg_segmentation(
        config,
        transcripts_csv=transcripts,
        cellpose_mask_path=tmp_path / "masks.npy",
        cellpose_cellprob_path=tmp_path / "cellprobs.npy",
        transforms_path=transforms,
    )


@pytest.mark.parametrize(("platform", "alias"), _PLATFORMS)
def test_fresh_run_converts_with_the_platform_alias(
    tmp_path: Path,
    conversions: list[tuple[str, dict[str, Any]]],
    platform: str,
    alias: str | None,
) -> None:
    """A fresh ProSeg run converts its raw output with the platform's alias."""
    _run_proseg(tmp_path, _config(tmp_path, platform))
    assert conversions == [
        (
            "convert_to_latest_zarr",
            {"quality_column_alias": alias, "platform": platform},
        )
    ]


@pytest.mark.parametrize(("platform", "alias"), _PLATFORMS)
def test_resumed_raw_output_converts_with_the_platform_alias(
    tmp_path: Path,
    conversions: list[tuple[str, dict[str, Any]]],
    platform: str,
    alias: str | None,
) -> None:
    """A run resumed from a raw ProSeg output converts it with the alias."""
    config = _config(tmp_path, platform)
    (config.dataset.output_dir / "proseg_base_raw.zarr").mkdir(parents=True)
    _run_proseg(tmp_path, config)
    assert conversions == [
        (
            "convert_to_latest_zarr",
            {"quality_column_alias": alias, "platform": platform},
        )
    ]


@pytest.mark.parametrize(("platform", "alias"), _PLATFORMS)
def test_existing_latest_store_is_upgraded_with_the_platform_alias(
    tmp_path: Path,
    conversions: list[tuple[str, dict[str, Any]]],
    platform: str,
    alias: str | None,
) -> None:
    """An existing latest store is upgraded in place with the alias."""
    config = _config(tmp_path, platform)
    latest = config.dataset.output_dir / "proseg_base_latest.zarr"
    latest.mkdir(parents=True)
    (latest / "zarr.json").write_text("{}")
    _run_proseg(tmp_path, config)
    assert conversions == [
        (
            "upgrade_spatialdata_contract",
            {"quality_column_alias": alias, "platform": platform},
        )
    ]


@pytest.mark.parametrize(("platform", "native"), _NATIVE)
def test_custom_arm_rejects_another_platforms_native_image(
    tmp_path: Path, platform: str, native: str
) -> None:
    """A custom arm must segment the platform's own native image."""
    other = dict(_NATIVE)["XENIUM" if platform == "MERSCOPE" else "MERSCOPE"]
    config = _config(tmp_path, platform, image_key=other, experimental_arm=True)
    with pytest.raises(ValueError, match="Custom models require the native image"):
        pipeline._preflight_model_selection(config, check_cell_cache=False)


@pytest.mark.parametrize(("platform", "native"), _NATIVE)
@pytest.mark.parametrize("explicit_image_key", [False, True])
def test_custom_arm_preflights_the_platform_native_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
    native: str,
    explicit_image_key: bool,
) -> None:
    """The stock-arm preflight receives the platform's native image key."""
    config = _config(
        tmp_path,
        platform,
        image_key=native if explicit_image_key else None,
        experimental_arm=True,
    )
    matrix = np.eye(3)
    seen: dict[str, Any] = {}

    def _load(_config: SegmentationConfig) -> tuple[Any, ...]:
        return None, None, 0, 0, matrix, None

    def _preflight(
        _config: SegmentationConfig, *, transform_matrix: Any, image_key: str
    ) -> str:
        seen["image_key"] = image_key
        seen["matrix"] = transform_matrix
        return "identity"

    monkeypatch.setattr(pipeline, "_load_dataset_sdata", _load)
    monkeypatch.setattr(pipeline, "preflight_stock_arm", _preflight)
    identity = pipeline._preflight_model_selection(config, check_cell_cache=False)
    assert identity == "identity"
    assert seen == {"image_key": native, "matrix": matrix}

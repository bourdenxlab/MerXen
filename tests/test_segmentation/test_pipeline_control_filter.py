"""Tests for control-feature filtering of the ProSeg transcript input."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from merxen.config import SegmentationConfig
from merxen.segmentation.pipeline import run_cellpose_segmentation


def _run_with_points(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    platform: str,
    points: pd.DataFrame,
) -> pd.DataFrame:
    dataset: dict[str, object] = {
        "name": f"P1_{platform}",
        "platform": platform,
        "data_path": str(tmp_path / "input.zarr"),
        "channels": ["DAPI"],
        "output_dir": str(tmp_path / "segment_out"),
    }
    if platform == "XENIUM":
        dataset["min_qv"] = 20.0
    cfg = SegmentationConfig.model_validate(
        {
            "dataset": dataset,
            "mask_filter": {
                "final_min_area_um2": None,
                "final_max_area_um2": None,
            },
        }
    )

    def fake_cellpose(
        *,
        output_mask_path: Path,
        output_cellprob_path: Path | None = None,
        output_stitching_stats_path: Path | None = None,
        **kwargs: object,
    ) -> Path:
        del kwargs
        mask = np.zeros((16, 16), dtype=np.uint32)
        mask[0:8, 0:8] = 1
        output_mask_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_mask_path, mask)
        if output_cellprob_path is not None:
            np.save(output_cellprob_path, np.ones(mask.shape, dtype=np.float32))
        if output_stitching_stats_path is not None:
            output_stitching_stats_path.write_text('{"final_labels": 1}\n')
        return output_mask_path

    monkeypatch.setattr(
        "merxen.segmentation.pipeline._load_dataset_sdata",
        lambda config: (object(), object(), 16, 16, np.eye(3), points),
    )
    monkeypatch.setattr(
        "merxen.segmentation.pipeline.run_tiled_cellpose",
        fake_cellpose,
    )
    monkeypatch.setattr(
        "merxen.segmentation.pipeline.build_cellpose_affine_to_microns",
        lambda *args, **kwargs: ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    )

    outputs = run_cellpose_segmentation(cfg, force_rerun=True)
    return pd.read_csv(outputs["transcripts_csv"])


def test_xenium_proseg_input_drops_controls_by_is_gene_and_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Xenium ProSeg input keeps only genes, including 5K and legacy controls."""
    names = [
        "GFAP",
        "BLANK_0006",
        "Intergenic_Region_4",
        "NegControlProbe_00002",
        "UnlistedControl_1",
        "SNAP25",
    ]
    points = pd.DataFrame(
        {
            "x": np.arange(len(names), dtype=float),
            "y": np.ones(len(names)),
            "z": np.zeros(len(names)),
            "feature_name": pd.Categorical(names),
            "is_gene": [name in {"GFAP", "SNAP25"} for name in names],
            "codeword_category": ["predesigned_gene"] * len(names),
            "qv": np.full(len(names), 30.0),
        }
    )

    written = _run_with_points(monkeypatch, tmp_path, platform="XENIUM", points=points)

    assert written["feature_name"].tolist() == ["GFAP", "SNAP25"]


def test_merscope_proseg_input_drops_blank_codewords(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """MERSCOPE ProSeg input no longer carries ``Blank-<N>`` codewords."""
    genes = ["Gad1", "Blank-3", "Nkx6-1", "Blank-41"]
    points = pd.DataFrame(
        {
            "global_x": np.arange(len(genes), dtype=float),
            "global_y": np.ones(len(genes)),
            "gene": genes,
            "transcript_score": np.full(len(genes), 0.9),
        }
    )

    written = _run_with_points(
        monkeypatch, tmp_path, platform="MERSCOPE", points=points
    )

    assert written["feature_name"].tolist() == ["Gad1", "Nkx6-1"]
    assert written["qv"].tolist() == pytest.approx([0.9, 0.9])

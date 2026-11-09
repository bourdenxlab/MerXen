"""Saved bootstrap metadata can be added without altering scientific payloads."""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import zarr

from merxen.annotation.prediction_metadata import (
    PREDICTION_FIELDS,
    enrich_observations,
    enrich_store,
    prediction_columns,
)


def _resolved(level: str) -> pd.DataFrame:
    frame = pd.DataFrame({"cell_id": ["19", "9007199254740993", "empty"]})
    frame["level"] = level
    for column in PREDICTION_FIELDS:
        if column.endswith(("name", "assignment")):
            frame[column] = ["type <A>", "type B", None]
        else:
            frame[column] = [0.9, 0.4, np.nan]
    frame["directly_assigned"] = pd.array([True, False, None], dtype="boolean")
    return frame


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    paths = tmp_path / "seaad.parquet", tmp_path / "whb.parquet"
    for path, level in zip(
        paths, ["CCN20260630_LEVEL_1", "CCN202210140_SUPC"], strict=True
    ):
        _resolved(level).to_parquet(path)
    return paths


def test_exact_id_join_is_order_independent_and_keeps_existing_labels() -> None:
    predictions = prediction_columns(
        _resolved("CCN20260630_LEVEL_1"), _resolved("CCN202210140_SUPC")
    )
    obs = pd.DataFrame(
        {"ct_seaad_subclass_name": pd.Categorical([None, "accepted", None])},
        index=["9007199254740993", "19", "empty"],
    )
    enriched = enrich_observations(obs, predictions)
    pd.testing.assert_frame_equal(enriched[obs.columns], obs)
    assert enriched.ct_seaad_subclass_raw_name.tolist()[:2] == ["type B", "type <A>"]
    assert pd.isna(enriched.loc["empty", "ct_whb_supercluster_bp"])
    pd.testing.assert_frame_equal(enrich_observations(enriched, predictions), enriched)
    enriched.loc["19", "ct_seaad_subclass_bp"] = 0.1
    with pytest.raises(ValueError, match="conflicts"):
        enrich_observations(enriched, predictions)


def test_incomplete_and_duplicate_saved_predictions_are_refused() -> None:
    seaad, whb = _resolved("CCN20260630_LEVEL_1"), _resolved("CCN202210140_SUPC")
    with pytest.raises(ValueError, match="unique cell IDs"):
        prediction_columns(pd.concat([seaad, seaad]), whb)
    with pytest.raises(ValueError, match="different cells"):
        prediction_columns(seaad, whb.iloc[:1])
    with pytest.raises(ValueError, match="Missing saved"):
        prediction_columns(seaad.drop(columns="runner_up_5_name"), whb)
    predictions = prediction_columns(seaad, whb)
    with pytest.raises(ValueError, match="covered"):
        enrich_observations(pd.DataFrame(index=["other"]), predictions)


def _store(tmp_path: Path) -> Path:
    store = tmp_path / "store.zarr"
    root = zarr.open_group(store, mode="w", zarr_format=2)
    root.create_group("tables")
    for name in ["one", "two"]:
        data = ad.AnnData(
            np.array([[1, 2], [0, 0]], dtype=np.int32),
            obs=pd.DataFrame(
                {"ct_seaad_subclass_name": pd.Categorical(["old", None])},
                index=["19", "empty"],
            ),
        )
        data.obsm["X_umap"] = np.array([[1, 2], [3, 4]], dtype=np.float32)
        data.write_zarr(store / "tables" / name)
    zarr.consolidate_metadata(store)
    return store


def test_store_enrichment_changes_only_observations_and_is_repeatable(
    tmp_path: Path,
) -> None:
    seaad, whb = _inputs(tmp_path)
    store = _store(tmp_path)
    before = {
        p.relative_to(store): p.read_bytes()
        for p in store.rglob("*")
        if p.is_file() and "/obs/" not in str(p) and p.name != ".zmetadata"
    }
    assert enrich_store(store, ["one", "two"], seaad, whb) == {"one": 2, "two": 2}
    enrich_store(store, ["one", "two"], seaad, whb)
    for relative, content in before.items():
        assert (store / relative).read_bytes() == content
    root = zarr.open_group(store, mode="r", use_consolidated=True)
    obs = ad.io.read_elem(root["tables/one/obs"])
    assert obs.ct_seaad_subclass_name.tolist()[0] == "old"
    assert obs.ct_whb_supercluster_runner_up_5_name.tolist()[0] == "type <A>"
    assert obs.ct_whb_supercluster_directly_assigned.dtype == pd.BooleanDtype()
    assert obs.ct_whb_supercluster_directly_assigned.iloc[0]
    assert pd.isna(obs.ct_whb_supercluster_directly_assigned.iloc[1])


def test_failed_replacement_rolls_back_previously_changed_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from merxen.annotation import prediction_metadata as module

    seaad, whb = _inputs(tmp_path)
    store = _store(tmp_path)
    before = {
        name: ad.read_zarr(store / "tables" / name).obs for name in ["one", "two"]
    }
    write = module.ad.io.write_elem
    calls = 0

    def fail_second(*args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated storage failure")
        write(*args, **kwargs)

    monkeypatch.setattr(module.ad.io, "write_elem", fail_second)
    with pytest.raises(RuntimeError, match="simulated"):
        enrich_store(store, ["one", "two"], seaad, whb)
    for name, obs in before.items():
        pd.testing.assert_frame_equal(ad.read_zarr(store / "tables" / name).obs, obs)

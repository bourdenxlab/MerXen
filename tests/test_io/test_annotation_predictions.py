"""Saved bootstrap metadata can be added without altering scientific payloads."""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import zarr

from merxen.io.annotation_predictions import (
    MAP_FIRST_ACCEPTANCE_FIELDS,
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
    from merxen.io import annotation_predictions as module

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


# --------------------------------------------------------------------------
# map_first: MAP tidy parquets and the RESOLVE label table


def _tidy(levels: dict[str, list[str]]) -> pd.DataFrame:
    """A MAP tidy frame: one row per (cell, level), as mapmycells_engine writes."""
    rows = []
    for level, cells in levels.items():
        for rank, cell in enumerate(cells):
            row: dict[str, object] = {
                "cell_id": cell,
                "level": level,
                "level_name": level.lower(),
                "assignment": f"{level}_id{rank}",
                "name": f"type {rank}",
                "bp": 0.9 - 0.1 * rank,
                "aggregate_probability": 0.8 - 0.1 * rank,
                "avg_correlation": 0.5,
                "directly_assigned": True,
                "n_runners_up": 5,
            }
            for runner in range(1, 6):
                row[f"runner_up_{runner}_assignment"] = f"alt{runner}"
                row[f"runner_up_{runner}_name"] = f"alt type {runner}"
                row[f"runner_up_{runner}_probability"] = 0.01 * runner
                row[f"runner_up_{runner}_correlation"] = 0.1
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame["level"] = frame["level"].astype("category")
    return frame


def test_map_first_predictions_read_the_tidy_levels_and_label_status() -> None:
    from merxen.io.annotation_predictions import map_first_predictions

    seaad = _tidy({"CCN20260630_LEVEL_0": ["19"], "CCN20260630_LEVEL_1": ["19"]})
    whb = _tidy({"CCN202210140_SUPC": ["19", "7"], "CCN202210140_CLUS": ["19", "7"]})
    labels = pd.DataFrame(
        {
            "cell_id": ["19", "7"],
            "ct_seaad_subclass_name": ["type 0", None],
            "ct_seaad_subclass_status": ["confident", "parent_unresolved"],
            # Cell 7's supercluster call is kept although not accepted.
            "ct_supercluster_name": ["type 0", "alt type 5"],
            "ct_supercluster_status": ["confident", "low_bootstrap"],
        }
    )
    predictions = map_first_predictions(seaad, whb, labels)
    assert set(predictions.index) == {"19", "7"}
    assert predictions.loc["19", "ct_seaad_subclass_raw_name"] == "type 0"
    assert predictions.loc["19", "ct_seaad_subclass_raw_assignment"] == (
        "CCN20260630_LEVEL_1_id0"
    )
    assert predictions.loc["7", "ct_whb_supercluster_bp"] == pytest.approx(0.8)
    assert predictions.loc["7", "ct_whb_supercluster_runner_up_5_name"] == (
        "alt type 5"
    )
    assert predictions.loc["7", "ct_whb_supercluster_resolution_status"] == (
        "low_bootstrap"
    )
    # The acceptance fields of the viewer contract, for both atlases.
    for stem in ("ct_seaad_subclass", "ct_whb_supercluster"):
        for field in MAP_FIRST_ACCEPTANCE_FIELDS:
            assert f"{stem}_{field}" in predictions
    assert predictions.loc["19", "ct_seaad_subclass_accepted"]
    assert predictions.loc["19", "ct_seaad_subclass_name"] == "type 0"
    assert predictions.loc["19", "ct_seaad_subclass_assignment"] == (
        "CCN20260630_LEVEL_1_id0"
    )
    assert not predictions.loc["7", "ct_seaad_subclass_accepted"]
    assert pd.isna(predictions.loc["7", "ct_seaad_subclass_assignment"])
    assert predictions.loc["19", "ct_whb_supercluster_accepted"]
    assert predictions.loc["19", "ct_whb_supercluster_assignment"] == (
        "CCN202210140_SUPC_id0"
    )
    assert not predictions.loc["7", "ct_whb_supercluster_accepted"]
    assert predictions.loc["7", "ct_whb_supercluster_name"] == "alt type 5"
    # A runner-up name maps to its id too.
    assert predictions.loc["7", "ct_whb_supercluster_assignment"] == "alt5"
    assert predictions["ct_whb_supercluster_accepted"].dtype == pd.BooleanDtype()
    # Without the label table only the prediction fields are returned.
    assert "ct_whb_supercluster_accepted" not in map_first_predictions(seaad, whb)
    # SEA-AD did not map cell 7: its fields stay missing.
    assert pd.isna(predictions.loc["7", "ct_seaad_subclass_raw_name"])
    assert pd.isna(predictions.loc["7", "ct_seaad_subclass_directly_assigned"])
    assert predictions["ct_whb_supercluster_directly_assigned"].dtype == (
        pd.BooleanDtype()
    )
    for field in PREDICTION_FIELDS:
        assert f"ct_seaad_subclass_{field}" in predictions
        assert f"ct_whb_supercluster_{field}" in predictions
    with pytest.raises(ValueError, match="unique cell IDs"):
        map_first_predictions(pd.concat([seaad, seaad]), whb)
    with pytest.raises(ValueError, match="Missing MAP"):
        map_first_predictions(seaad.drop(columns="bp"), whb)
    with pytest.raises(ValueError, match="label table"):
        map_first_predictions(seaad, whb, labels.drop(columns="cell_id"))
    with pytest.raises(ValueError, match="label table"):
        map_first_predictions(seaad, whb, labels.drop(columns="ct_supercluster_name"))


def test_map_first_enrichment_leaves_unmapped_observations_missing(
    tmp_path: Path,
) -> None:
    from merxen.io.annotation_predictions import main

    seaad = tmp_path / "S_mmc_seaad_mr_panel.parquet"
    whb = tmp_path / "S_mmc_whb_hippocampal_supc_clus.parquet"
    labels = tmp_path / "S_celltype_labels.parquet"
    _tidy({"CCN20260630_LEVEL_1": ["19"]}).to_parquet(seaad)
    _tidy({"CCN202210140_SUPC": ["19"]}).to_parquet(whb)
    # The store's ct_seaad_subclass_name is the label table's (the hierarchy
    # step copies it), so the derived name agrees with it.
    label_table = pd.DataFrame(
        {
            "cell_id": ["19", "empty"],
            "ct_seaad_subclass_name": ["old", None],
            "ct_seaad_subclass_status": ["low_confidence", "low_counts"],
            "ct_supercluster_name": ["type 0", None],
            "ct_supercluster_status": ["confident", "low_counts"],
        }
    )
    label_table.to_parquet(labels)
    store = _store(tmp_path)
    main(
        [
            "--store",
            str(store),
            "--table",
            "one",
            "--seaad-mmc",
            str(seaad),
            "--whb-mmc",
            str(whb),
            "--labels",
            str(labels),
        ]
    )
    root = zarr.open_group(store, mode="r", use_consolidated=True)
    obs = ad.io.read_elem(root["tables/one/obs"])
    assert obs.ct_seaad_subclass_name.tolist()[0] == "old"
    assert obs.loc["19", "ct_whb_supercluster_raw_name"] == "type 0"
    assert obs.loc["19", "ct_seaad_subclass_resolution_status"] == "low_confidence"
    assert not obs.loc["19", "ct_seaad_subclass_accepted"]
    assert obs.loc["19", "ct_whb_supercluster_accepted"]
    assert obs.loc["19", "ct_whb_supercluster_name"] == "type 0"
    assert obs.loc["19", "ct_whb_supercluster_assignment"] == "CCN202210140_SUPC_id0"
    assert obs.loc["empty", "ct_whb_supercluster_resolution_status"] == "low_counts"
    assert pd.isna(obs.loc["empty", "ct_whb_supercluster_bp"])
    # A label table that disagrees with the store's labels is refused.
    stale = tmp_path / "stale_labels.parquet"
    label_table.assign(ct_seaad_subclass_name=["new", None]).to_parquet(stale)
    with pytest.raises(ValueError, match="conflicts: ct_seaad_subclass_name"):
        enrich_store(store, ["two"], seaad, whb, map_first=True, labels=stale)
    with pytest.raises(ValueError, match="label table"):
        enrich_store(store, ["two"], seaad, whb, map_first=True)
    # The whole-panel route still requires every observation to be covered.
    resolved = tmp_path / "resolved.parquet"
    _tidy({"CCN20260630_LEVEL_1": ["19"]}).to_parquet(resolved)
    with pytest.raises(ValueError):
        enrich_store(store, ["two"], resolved, resolved)


def test_prediction_cli_inputs_are_one_route_or_the_other(tmp_path: Path) -> None:
    from merxen.io.annotation_predictions import main

    path = str(tmp_path / "x.parquet")
    base = ["--store", str(tmp_path), "--table", "one"]
    for argv in (
        [*base, "--seaad-resolved", path, "--seaad-mmc", path, "--whb-mmc", path],
        [*base, "--seaad-mmc", path, "--whb-resolved", path],
        [*base, "--seaad-resolved", path, "--whb-resolved", path, "--labels", path],
        [*base, "--seaad-mmc", path, "--whb-mmc", path],
    ):
        with pytest.raises(SystemExit):
            main(argv)

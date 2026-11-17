"""Expose saved bootstrap predictions without re-running annotation.

Only observation metadata is enriched. Accepted labels, expression, embeddings,
spatial elements and transcript assignments retain their existing meaning.

Two sources are read: the whole-panel hippocampal route's resolved Parquet
files (``seaad_resolved.parquet`` / ``whb_resolved.parquet``), or the map_first
route's MAP tidy outputs (``<sample>_mmc_seaad_mr_panel.parquet`` and
``<sample>_mmc_<WHB primary>.parquet``, five runners-up at every level) with
the RESOLVE label table for the acceptance fields (``name``, ``assignment``,
``accepted``, ``resolution_status``).
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import anndata as ad
import pandas as pd
import zarr

from merxen.io.spatialdata_io import spatialdata_write_lock

PREDICTION_LEVELS = (
    ("seaad", "CCN20260630_LEVEL_1", "subclass"),
    ("whb", "CCN202210140_SUPC", "supercluster"),
)
PREDICTION_FIELDS = (
    "raw_assignment",
    "raw_name",
    "bp",
    "aggregate_probability",
    "avg_correlation",
    "directly_assigned",
    "n_runners_up",
    *(
        f"runner_up_{rank}_{field}"
        for rank in range(1, 6)
        for field in ("assignment", "name", "probability", "correlation")
    ),
)


def prediction_columns(seaad: pd.DataFrame, whb: pd.DataFrame) -> pd.DataFrame:
    """Return saved top calls and up to five supported, conditional alternatives.

    Bootstrap fractions are stability measures, not biological posterior
    probabilities. Subclass runners compete within the selected parent class.
    Unexpressed objects retain missing predictions and scores.
    """
    frames = []
    for atlas, level, token in PREDICTION_LEVELS:
        source = seaad if atlas == "seaad" else whb
        required = {"cell_id", "level", *PREDICTION_FIELDS}
        if missing := required - set(source):
            raise ValueError(
                f"Missing saved {atlas} prediction fields: {sorted(missing)}"
            )
        frame = source.loc[source.level.eq(level)].set_index("cell_id")
        frame.index = frame.index.astype(str)
        if frame.empty or not frame.index.is_unique:
            raise ValueError(f"Saved {atlas} predictions need unique cell IDs")
        frames.append(frame[list(PREDICTION_FIELDS)].add_prefix(f"ct_{atlas}_{token}_"))
    if set(frames[0].index) != set(frames[1].index):
        raise ValueError("Saved atlas predictions cover different cells")
    return pd.concat(frames, axis=1)


# map_first: the RESOLVE label-table level of each prediction level; its
# ``_name`` and ``_status`` columns give the acceptance fields.
MAP_FIRST_LABEL_LEVELS = {"seaad": "ct_seaad_subclass", "whb": "ct_supercluster"}
# The label-table status of an accepted call (schema.CellStatus.CONFIDENT).
MAP_FIRST_ACCEPTED_STATUS = "confident"
# MAP tidy column -> the prediction field it holds.
MAP_FIRST_RENAMES = {"assignment": "raw_assignment", "name": "raw_name"}
# The acceptance fields map_first adds next to PREDICTION_FIELDS, as the
# whole-panel route writes them.
MAP_FIRST_ACCEPTANCE_FIELDS = ("name", "assignment", "accepted", "resolution_status")


def _name_to_assignment(frame: pd.DataFrame) -> dict[str, str]:
    """Return each taxonomy name of a level's tidy rows -> its node id.

    Names come from the raw call and the runners-up; a name seen with two ids
    is left out (its id stays missing rather than guessed).
    """
    pairs = [frame[["raw_name", "raw_assignment"]].to_numpy()]
    for rank in range(1, 6):
        pairs.append(
            frame[[f"runner_up_{rank}_name", f"runner_up_{rank}_assignment"]].to_numpy()
        )
    ids: dict[str, set[str]] = {}
    for name, assignment in (row for block in pairs for row in block):
        if pd.isna(name) or pd.isna(assignment):
            continue
        ids.setdefault(str(name), set()).add(str(assignment))
    return {name: next(iter(found)) for name, found in ids.items() if len(found) == 1}


def _acceptance_fields(
    rows: pd.DataFrame, labels: pd.DataFrame, atlas: str
) -> pd.DataFrame:
    """Return the label table's acceptance fields of one level, by cell id.

    ``name`` is the label table's call at that level (also for cells it did not
    accept, as in the map_first store); ``accepted`` is ``status ==
    confident``; ``assignment`` is the node id of ``name`` from the level's
    tidy ``rows``; ``resolution_status`` is the label-table status
    (``schema.CellStatus``). Every label-table cell is kept, so the name
    matches the one the hierarchy step copied into the store.
    """
    stem = MAP_FIRST_LABEL_LEVELS[atlas]
    columns = ["cell_id", f"{stem}_name", f"{stem}_status"]
    if missing := [column for column in columns if column not in labels]:
        raise ValueError(f"The label table lacks {missing}")
    by_cell = labels[columns].copy()
    by_cell["cell_id"] = by_cell["cell_id"].astype(str)
    if not by_cell["cell_id"].is_unique:
        raise ValueError("The label table needs unique cell IDs")
    by_cell = by_cell.set_index("cell_id")
    status = by_cell[f"{stem}_status"].astype("string")
    names = by_cell[f"{stem}_name"].astype("string")
    ids = _name_to_assignment(rows)
    return pd.DataFrame(
        {
            "name": names,
            "assignment": names.map(ids, na_action="ignore").astype("string"),
            "accepted": status.eq(MAP_FIRST_ACCEPTED_STATUS).astype("boolean"),
            "resolution_status": status,
        },
        index=by_cell.index,
    )


def map_first_predictions(
    seaad: pd.DataFrame, whb: pd.DataFrame, labels: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Return the prediction fields of a map_first sample from MAP's tidy output.

    MAP's tidy parquets hold the raw MapMyCells call (``assignment``,
    ``name``), its bootstrap fraction (``bp``, conditional on the parent
    choice), ``aggregate_probability`` (the product along the hierarchy),
    correlation and five runners-up at every level, for every table cell.
    Only the SEA-AD subclass and WHB supercluster levels are kept. Cells one
    atlas did not map (e.g. a refused SEA-AD bundle) keep missing values.

    With the label table, each level also gets the acceptance fields of the
    viewer contract (``MAP_FIRST_ACCEPTANCE_FIELDS``) from
    ``ct_seaad_subclass_*`` and ``ct_supercluster_*``: ``name`` (the label
    table's call at that level, which map_first keeps for cells it did not
    accept), ``assignment`` (that node's id), ``accepted`` (status
    ``confident``) and ``resolution_status`` (the label-table status, a
    ``schema.CellStatus`` value).

    Args:
        seaad: The SEA-AD tidy frame (``<sample>_mmc_seaad_mr_panel.parquet``).
        whb: The WHB primary's tidy frame.
        labels: The RESOLVE label table (``cell_id`` and the level names and
            statuses); without it only the prediction fields are returned.

    Returns:
        ``ct_seaad_subclass_*`` and ``ct_whb_supercluster_*`` columns indexed
        by cell id (the union of both atlases' and the label table's cells).

    Raises:
        ValueError: If a field is missing, cell ids repeat, or the label
            table lacks a name or status column.
    """
    frames = []
    for atlas, level, token in PREDICTION_LEVELS:
        source = (seaad if atlas == "seaad" else whb).rename(columns=MAP_FIRST_RENAMES)
        required = {"cell_id", "level", *PREDICTION_FIELDS}
        if missing := required - set(source):
            raise ValueError(
                f"Missing MAP {atlas} prediction fields: {sorted(missing)}"
            )
        rows = source.loc[source.level.astype(str).eq(level)].set_index("cell_id")
        rows.index = rows.index.astype(str)
        if not rows.index.is_unique:
            raise ValueError(f"MAP {atlas} predictions need unique cell IDs")
        frame = rows[list(PREDICTION_FIELDS)].copy()
        # Nullable dtypes, so cells the other atlas alone covers stay missing
        # rather than turning the column into strings.
        frame["directly_assigned"] = frame["directly_assigned"].astype("boolean")
        frame["n_runners_up"] = pd.to_numeric(frame["n_runners_up"]).astype("Int64")
        if labels is not None:
            frame = pd.concat(
                [frame, _acceptance_fields(rows, labels, atlas)], axis=1, join="outer"
            )
        frames.append(frame.add_prefix(f"ct_{atlas}_{token}_"))
    return pd.concat(frames, axis=1, join="outer")


def enrich_observations(
    obs: pd.DataFrame, predictions: pd.DataFrame, *, require_coverage: bool = True
) -> pd.DataFrame:
    """Join by exact observation ID and add missing fields, refusing conflicts.

    Args:
        obs: The table's observations.
        predictions: Prediction fields indexed by cell id.
        require_coverage: Refuse observations without predictions (the
            whole-panel route); ``False`` leaves them missing (map_first,
            where an atlas may not cover every table cell).

    Returns:
        The enriched observations.

    Raises:
        ValueError: For repeated ids, uncovered ids (when required) or a
            conflicting existing field.
    """
    ids = obs.index.astype(str)
    if not ids.is_unique or (
        require_coverage and not set(ids) <= set(predictions.index)
    ):
        raise ValueError(
            "Table cell IDs must be unique and covered by saved predictions"
        )
    if not require_coverage and not set(ids) & set(predictions.index):
        raise ValueError("No table cell ID has a saved prediction")
    aligned = predictions.reindex(ids)
    enriched = obs.copy()
    for column in aligned:
        values = aligned[column]
        if column in obs:
            old = obs[column].reset_index(drop=True).astype(object)
            new = values.reset_index(drop=True).astype(object)
            same = old.isna().equals(new.isna()) and old[old.notna()].equals(
                new[new.notna()]
            )
            if not same:
                raise ValueError(f"Existing prediction metadata conflicts: {column}")
            continue
        if (
            pd.api.types.is_object_dtype(values)
            or isinstance(values.dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(values)
        ):
            enriched[column] = pd.Categorical(values.astype("string"))
        else:
            # Preserve nullable booleans; coercing them to object arrays makes
            # some AnnData writers serialize True/<NA> as literal strings.
            enriched[column] = values.array
    return enriched


def enrich_store(
    store: Path,
    tables: list[str],
    seaad: Path,
    whb: Path,
    *,
    map_first: bool = False,
    labels: Path | None = None,
) -> dict[str, int]:
    """Replace only obs groups under the store lock, with rollback on failure.

    All joins are checked before the first replacement. The replacement groups
    are staged beside the store on the same filesystem; prior obs groups remain
    available for rollback until metadata consolidation succeeds. Repeating this
    operation with identical predictions is safe.

    Args:
        store: The SpatialData zarr store.
        tables: Table names whose obs gets the fields.
        seaad: SEA-AD predictions (resolved Parquet, or MAP tidy Parquet).
        whb: WHB predictions (resolved Parquet, or MAP tidy Parquet).
        map_first: Read MAP tidy Parquets (``map_first_predictions``);
            observations without a prediction stay missing.
        labels: map_first only, and required there: the RESOLVE label table
            (the acceptance fields ``name``, ``assignment``, ``accepted``,
            ``resolution_status``).

    Returns:
        Observations per enriched table.
    """
    if not tables or len(set(tables)) != len(tables):
        raise ValueError("Specify distinct table names")
    if any(Path(table).name != table for table in tables):
        raise ValueError("Table names must be single path components")
    if labels is not None and not map_first:
        raise ValueError("A label table is read only with map_first predictions")
    if map_first:
        if labels is None:
            raise ValueError("map_first predictions need the RESOLVE label table")
        predictions = map_first_predictions(
            pd.read_parquet(seaad), pd.read_parquet(whb), pd.read_parquet(labels)
        )
    else:
        predictions = prediction_columns(pd.read_parquet(seaad), pd.read_parquet(whb))
    store = store.resolve(strict=True)
    with spatialdata_write_lock(store):
        root = zarr.open_group(store, mode="r+", use_consolidated=False)
        observations = {
            table: enrich_observations(
                ad.io.read_elem(root[f"tables/{table}/obs"]),
                predictions,
                require_coverage=not map_first,
            )
            for table in tables
        }
        with tempfile.TemporaryDirectory(
            prefix=".prediction-metadata-", dir=store.parent
        ) as tmp:
            staging = Path(tmp)
            replaced = []
            try:
                for index, (table, obs) in enumerate(observations.items()):
                    target = store / "tables" / table / "obs"
                    group = zarr.open_group(
                        staging / str(index),
                        mode="w",
                        zarr_format=root.metadata.zarr_format,
                    )
                    ad.io.write_elem(group, "obs", obs)
                    backup = staging / f"backup-{index}"
                    os.replace(target, backup)
                    replaced.append((target, backup))
                    os.replace(staging / str(index) / "obs", target)
                zarr.consolidate_metadata(store)
            except Exception:
                import shutil

                for target, backup in reversed(replaced):
                    if target.exists():
                        shutil.rmtree(target)
                    os.replace(backup, target)
                zarr.consolidate_metadata(store)
                raise
    return {table: len(obs) for table, obs in observations.items()}


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser (whole-panel or map_first inputs)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--table", action="append", required=True)
    seaad = parser.add_mutually_exclusive_group(required=True)
    seaad.add_argument("--seaad-resolved", type=Path)
    seaad.add_argument(
        "--seaad-mmc", type=Path, help="map_first: <sample>_mmc_seaad_mr_panel.parquet"
    )
    whb = parser.add_mutually_exclusive_group(required=True)
    whb.add_argument("--whb-resolved", type=Path)
    whb.add_argument(
        "--whb-mmc", type=Path, help="map_first: <sample>_mmc_<WHB primary>.parquet"
    )
    parser.add_argument(
        "--labels",
        type=Path,
        help="map_first (required): <sample>_celltype_labels.parquet",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Enrich explicitly selected tables using two saved prediction files."""
    parser = build_parser()
    args = parser.parse_args(argv)
    map_first = args.seaad_mmc is not None or args.whb_mmc is not None
    if map_first and (args.seaad_mmc is None or args.whb_mmc is None):
        parser.error("--seaad-mmc and --whb-mmc go together")
    if args.labels is not None and not map_first:
        parser.error("--labels needs --seaad-mmc and --whb-mmc")
    if map_first and args.labels is None:
        parser.error("--seaad-mmc and --whb-mmc need --labels")
    counts = enrich_store(
        args.store,
        args.table,
        args.seaad_mmc if map_first else args.seaad_resolved,
        args.whb_mmc if map_first else args.whb_resolved,
        map_first=map_first,
        labels=args.labels,
    )
    for table, count in counts.items():
        print(f"{table}: {count} cells; saved bootstrap metadata added")


if __name__ == "__main__":
    main()

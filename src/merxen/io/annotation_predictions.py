"""Expose saved bootstrap predictions without re-running annotation.

Only observation metadata is enriched. Accepted labels, expression, embeddings,
spatial elements and transcript assignments retain their existing meaning.
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


def enrich_observations(obs: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Join by exact observation ID and add missing fields, refusing conflicts."""
    ids = obs.index.astype(str)
    if not ids.is_unique or not set(ids) <= set(predictions.index):
        raise ValueError(
            "Table cell IDs must be unique and covered by saved predictions"
        )
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
    store: Path, tables: list[str], seaad: Path, whb: Path
) -> dict[str, int]:
    """Replace only obs groups under the store lock, with rollback on failure.

    All joins are checked before the first replacement. The replacement groups
    are staged beside the store on the same filesystem; prior obs groups remain
    available for rollback until metadata consolidation succeeds. Repeating this
    operation with identical predictions is safe.
    """
    if not tables or len(set(tables)) != len(tables):
        raise ValueError("Specify distinct table names")
    if any(Path(table).name != table for table in tables):
        raise ValueError("Table names must be single path components")
    predictions = prediction_columns(pd.read_parquet(seaad), pd.read_parquet(whb))
    store = store.resolve(strict=True)
    with spatialdata_write_lock(store):
        root = zarr.open_group(store, mode="r+", use_consolidated=False)
        observations = {
            table: enrich_observations(
                ad.io.read_elem(root[f"tables/{table}/obs"]), predictions
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


def main() -> None:
    """Enrich explicitly selected tables using two saved resolved Parquet files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--table", action="append", required=True)
    parser.add_argument("--seaad-resolved", type=Path, required=True)
    parser.add_argument("--whb-resolved", type=Path, required=True)
    args = parser.parse_args()
    counts = enrich_store(
        args.store, args.table, args.seaad_resolved, args.whb_resolved
    )
    for table, count in counts.items():
        print(f"{table}: {count} cells; saved bootstrap metadata added")


if __name__ == "__main__":
    main()

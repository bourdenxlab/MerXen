"""Build region-filtered, donor-balanced linear expression priors for pciSeq."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from merxen import local_config
from merxen.analysis.allen_reference_assets import AtlasAssets, ensure_atlas_assets
from merxen.analysis.pciseq import _read_reference
from merxen.config import PciSeqReferenceConfig

_ALGORITHM_VERSION = "linear_donor_mean_v1"
_AUTO_LEVELS = {"whb": "cluster", "wmb": "subclass", "seaad": "subclass"}


@dataclass
class _ExpressionPlan:
    path: Path
    row_indices: np.ndarray
    metadata_positions: np.ndarray


@dataclass
class _ReferenceMetadata:
    cells: pd.DataFrame
    selected: pd.DataFrame
    level: str
    level_label: str
    inventory: list[dict[str, Any]]
    diagnostics: dict[str, Any]


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _read_csv(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns
    selected = (
        None if columns is None else [column for column in columns if column in header]
    )
    return pd.read_csv(path, usecols=selected, dtype=str, keep_default_na=False)


def _unique(frame: pd.DataFrame, column: str, label: str) -> None:
    if (
        column not in frame
        or frame[column].eq("").any()
        or frame[column].duplicated().any()
    ):
        raise ValueError(f"{label} requires unique nonempty {column} values")


def _resolve_level(
    assets: AtlasAssets, requested: str
) -> tuple[pd.DataFrame, str, str]:
    membership = _read_csv(
        assets.metadata_paths["cluster_to_cluster_annotation_membership"]
    )
    required = {
        "cluster_alias",
        "cluster_annotation_term_set_name",
        "cluster_annotation_term_set_label",
        "cluster_annotation_term_label",
        "cluster_annotation_term_name",
    }
    if not required.issubset(membership):
        raise ValueError("Atlas taxonomy membership lacks required annotation fields")
    levels = membership[
        ["cluster_annotation_term_set_name", "cluster_annotation_term_set_label"]
    ].drop_duplicates()
    if levels["cluster_annotation_term_set_name"].duplicated().any():
        raise ValueError("Atlas taxonomy has ambiguous level names")
    choice = (
        _AUTO_LEVELS[assets.source] if requested.casefold() == "auto" else requested
    )
    if choice.casefold() == "finest":
        terms = _read_csv(assets.metadata_paths["cluster_annotation_term"])
        ordered = terms.loc[
            terms.cluster_annotation_term_set_name.str.casefold() != "neurotransmitter"
        ].copy()
        ordered["term_set_order"] = pd.to_numeric(
            ordered.term_set_order, errors="raise"
        )
        choice = str(
            ordered.sort_values("term_set_order")
            .iloc[-1]
            .cluster_annotation_term_set_name
        )
    matches = levels.loc[
        (levels.cluster_annotation_term_set_name.str.casefold() == choice.casefold())
        | (levels.cluster_annotation_term_set_label.str.casefold() == choice.casefold())
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Unknown {assets.source} taxonomy level {requested!r}; "
            "available native levels: "
            f"{levels.cluster_annotation_term_set_name.tolist()}"
        )
    level, native = matches.iloc[0].tolist()
    selected = membership.loc[
        membership.cluster_annotation_term_set_label == native
    ].copy()
    _unique(selected, "cluster_alias", "Taxonomy level membership")
    selected = selected.set_index("cluster_alias")[
        ["cluster_annotation_term_label", "cluster_annotation_term_name"]
    ].rename(
        columns={
            "cluster_annotation_term_label": "_type_id",
            "cluster_annotation_term_name": "_type_name",
        }
    )
    return selected, str(level), str(native)


def _region_aliases(
    assets: AtlasAssets, cells: pd.DataFrame
) -> tuple[dict[str, set[str]], list[dict[str, Any]]]:
    counts = cells["_region"].value_counts()
    aliases: dict[str, set[str]] = {}
    displays: dict[str, set[str]] = {str(value): set() for value in counts.index}
    for value in counts.index:
        aliases.setdefault(str(value).casefold(), set()).add(str(value))
    key = {
        "wmb": "region_of_interest_metadata",
        "whb": "region_of_interest_structure_map",
        "seaad": "library",
    }[assets.source]
    if key in assets.metadata_paths:
        table = _read_csv(assets.metadata_paths[key])
        canonical_col = (
            "acronym" if assets.source == "wmb" else "region_of_interest_label"
        )
        name_columns = {
            "wmb": ("label", "name"),
            "whb": ("structure_symbol", "structure_name"),
            "seaad": ("region_of_interest_name", "Brain Region"),
        }[assets.source]
        if canonical_col in table:
            for _, row in table.iterrows():
                value = row[canonical_col]
                if value not in displays:
                    continue
                for column in name_columns:
                    if column in table and row[column]:
                        aliases.setdefault(
                            str(row[column]).strip().casefold(), set()
                        ).add(value)
                        displays[value].add(str(row[column]))
    inventory = [
        {
            "label": str(label),
            "names": sorted(displays[str(label)]),
            "cell_count": int(count),
        }
        for label, count in counts.sort_index().items()
    ]
    return aliases, inventory


def _atlas_metadata(
    assets: AtlasAssets, config: PciSeqReferenceConfig
) -> _ReferenceMetadata:
    columns = [
        "cell_label",
        "cluster_alias",
        "donor_label",
        "library_label",
        "region_of_interest_label",
        "region_of_interest_acronym",
        "feature_matrix_label",
    ]
    cells = _read_csv(assets.metadata_paths["cell_metadata"], columns)
    _unique(cells, "cell_label", "Atlas cell metadata")
    cells = cells.set_index("cell_label")
    diagnostic: dict[str, Any] = {"metadata_cell_count": len(cells)}
    if assets.source == "seaad":
        membership = _read_csv(
            assets.metadata_paths["cell_to_cluster_membership"],
            ["cell_label", "cluster_alias"],
        )
        _unique(membership, "cell_label", "Cell-to-cluster membership")
        cells = cells.join(
            membership.set_index("cell_label"), how="left", validate="one_to_one"
        )
        library = _read_csv(
            assets.metadata_paths["library"],
            ["library_label", "region_of_interest_label", "donor_label"],
        )
        _unique(library, "library_label", "Atlas library metadata")
        library = library.set_index("library_label").rename(
            columns={"donor_label": "_library_donor"}
        )
        cells = cells.join(
            library, on="library_label", how="left", validate="many_to_one"
        )
        if "donor_label" not in cells:
            cells["donor_label"] = cells["_library_donor"]
        elif (
            cells["donor_label"].ne(cells["_library_donor"])
            & cells["_library_donor"].notna()
        ).any():
            raise ValueError("SEA-AD cell and library donor metadata disagree")
    region_col = (
        "region_of_interest_acronym"
        if assets.source == "wmb"
        else "region_of_interest_label"
    )
    if (
        region_col not in cells
        or cells[region_col].isna().any()
        or cells[region_col].eq("").any()
    ):
        raise ValueError("Atlas metadata requires an explicit region for every cell")
    cells["_region"] = cells[region_col]
    mapping, level, native_level = _resolve_level(assets, config.level)
    cells = cells.join(mapping, on="cluster_alias", how="left", validate="many_to_one")
    aliases, inventory = _region_aliases(assets, cells)
    diagnostic["metadata_without_type"] = int(cells["_type_id"].isna().sum())
    selected = cells.loc[cells["_type_id"].notna()].copy()
    if config.region_mode == "required" and not config.region_labels:
        raise ValueError(
            "This reference requires explicit region_labels; "
            "inspect the native ROI inventory"
        )
    regions: set[str] = set()
    for requested in config.region_labels:
        matches = aliases.get(requested.strip().casefold(), set())
        if len(matches) != 1:
            raise ValueError(
                f"Unknown or ambiguous {assets.source} region {requested!r}; "
                f"use native ROI labels: {[row['label'] for row in inventory]}"
            )
        regions.update(matches)
    if regions:
        selected = selected.loc[selected["_region"].isin(regions)].copy()
    diagnostic.update(
        region_column=region_col,
        requested_regions=config.region_labels,
        resolved_regions=sorted(regions),
        cells_after_region_filter=len(selected),
    )
    if assets.source == "seaad":
        disease = _read_csv(assets.metadata_paths["disease"])
        _unique(disease, "donor_label", "SEA-AD disease metadata")
        if config.seaad_cohort == "control":
            donors = disease.loc[
                disease["Overall AD neuropathological Change"].eq("Not AD")
                & disease["Cognitive Status"].eq("No dementia"),
                "donor_label",
            ]
            criterion = (
                "AD neuropathological change = Not AD AND "
                "cognitive status = No dementia"
            )
        elif config.seaad_cohort == "ad":
            donors = disease.loc[
                disease["Overall AD neuropathological Change"].isin(
                    ["Intermediate", "High"]
                ),
                "donor_label",
            ]
            criterion = "AD neuropathological change in Intermediate, High"
        else:
            donors = disease["donor_label"]
            criterion = "All donors in the released disease metadata"
        if not selected["donor_label"].isin(disease["donor_label"]).all():
            raise ValueError(
                "SEA-AD cells reference donors missing from disease metadata"
            )
        selected = selected.loc[selected["donor_label"].isin(donors)].copy()
        diagnostic.update(
            seaad_cohort=config.seaad_cohort,
            cohort_definition=criterion,
            released_donor_count=len(disease),
            eligible_donor_count=len(donors),
            cells_after_cohort_filter=len(selected),
        )
    if selected.empty:
        raise ValueError(
            "No annotated reference cells remain after ROI and cohort selection"
        )
    if "donor_label" not in selected:
        if config.donor_balance:
            raise ValueError(
                "Donor-balanced reference preparation requires donor_label"
            )
        selected["donor_label"] = "unavailable"
    if selected["donor_label"].isna().any() or selected["donor_label"].eq("").any():
        raise ValueError("Reference cells have missing donor identifiers")
    return _ReferenceMetadata(
        cells, selected, level, native_level, inventory, diagnostic
    )


def _strings(node: Any) -> np.ndarray:
    if isinstance(node, h5py.Group):
        categories = _strings(node["categories"])
        codes = np.asarray(node["codes"])
        if (codes < 0).any():
            raise ValueError("Reference identifiers contain missing categorical values")
        return np.asarray(categories[codes])
    values = np.asarray(node)
    return np.asarray(
        [
            value.decode() if isinstance(value, bytes) else str(value)
            for value in values
        ],
        dtype=object,
    )


def _axis_ids(handle: h5py.File, axis: str, preferred: str) -> np.ndarray:
    group = handle[axis]
    key = preferred if preferred in group else group.attrs.get("_index", "_index")
    return _strings(group[key])


def _matrix_shape(handle: h5py.File) -> tuple[int, int]:
    matrix = handle["X"]
    if isinstance(matrix, h5py.Dataset):
        return int(matrix.shape[0]), int(matrix.shape[1])
    if matrix.attrs.get("encoding-type") != "csr_matrix":
        raise ValueError("Reference X must use CSR sparse storage or a dense matrix")
    shape = matrix.attrs["shape"]
    return int(shape[0]), int(shape[1])


def _expression_plan(
    assets: AtlasAssets, metadata: _ReferenceMetadata, chunk_rows: int
) -> tuple[list[_ExpressionPlan], np.ndarray, dict[str, int]]:
    seen = np.zeros(len(metadata.cells), dtype=bool)
    found = np.zeros(len(metadata.selected), dtype=bool)
    plans: list[_ExpressionPlan] = []
    counts = {"raw_cells": 0, "raw_without_metadata": 0, "empty_library_cells": 0}
    for path in assets.expression_paths:
        with h5py.File(path, "r") as handle:
            ids = _axis_ids(handle, "obs", "cell_label")
            shape = _matrix_shape(handle)
            if len(ids) != shape[0] or pd.Index(ids).has_duplicates:
                raise ValueError(
                    "Reference matrix cell identifiers are duplicated or inconsistent"
                )
            counts["raw_cells"] += len(ids)
            full_positions = metadata.cells.index.get_indexer(ids)
            matched = full_positions[full_positions >= 0]
            if seen[matched].any():
                raise ValueError("Expression shards contain duplicate reference cells")
            seen[matched] = True
            counts["raw_without_metadata"] += int((full_positions < 0).sum())
            positions = metadata.selected.index.get_indexer(ids)
            if isinstance(handle["X"], h5py.Group):
                nonempty = np.diff(np.asarray(handle["X/indptr"])) > 0
            else:
                nonempty = np.zeros(len(ids), dtype=bool)
                for start in range(0, len(ids), chunk_rows):
                    nonempty[start : start + chunk_rows] = (
                        np.asarray(handle["X"][start : start + chunk_rows]).sum(axis=1)
                        > 0
                    )
            counts["empty_library_cells"] += int(((positions >= 0) & ~nonempty).sum())
            rows = np.flatnonzero((positions >= 0) & nonempty)
            found[positions[rows]] = True
            plans.append(_ExpressionPlan(path, rows, positions[rows]))
    counts["selected_metadata_without_expression"] = int((~found).sum())
    return plans, found, counts


def _support_and_weights(
    selected: pd.DataFrame, found: np.ndarray, minimum: int, balance: bool
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    actual = selected.loc[found]
    eligible = (
        selected.groupby(["_type_id", "_type_name"], sort=True)
        .size()
        .rename("n_metadata_cells")
    )
    actual_support = actual.groupby(["_type_id", "_type_name"], sort=True).agg(
        n_cells=("donor_label", "size"), n_donors=("donor_label", "nunique")
    )
    support = (
        eligible.to_frame()
        .join(actual_support)
        .fillna(0)
        .reset_index()
        .rename(columns={"_type_id": "type_id", "_type_name": "type_name"})
    )
    support[["n_cells", "n_donors"]] = support[["n_cells", "n_donors"]].astype(int)
    support["status"] = np.where(
        support.n_cells >= minimum, "retained", "below_min_cells"
    )
    support.loc[support.n_cells == 0, "status"] = "no_expression_cells"
    valid = support.loc[support.status == "retained", "type_id"].tolist()
    if not valid:
        raise ValueError("No reference cell types meet min_cells_per_type")
    keep = found & selected["_type_id"].isin(valid).to_numpy()
    cells = selected.loc[keep]
    weights = np.zeros(len(selected), dtype=float)
    codes = np.full(len(selected), -1, dtype=np.int64)
    if balance:
        donor_counts = cells.groupby(["_type_id", "donor_label"])["_type_id"].transform(
            "size"
        )
        donors = cells.groupby("_type_id")["donor_label"].transform("nunique")
        weights[keep] = 1.0 / (donor_counts.to_numpy() * donors.to_numpy())
    else:
        weights[keep] = (
            1.0 / cells.groupby("_type_id")["_type_id"].transform("size").to_numpy()
        )
    codes[keep] = pd.Index(valid).get_indexer(cells["_type_id"])
    return support, weights, codes


def _gene_profile(
    assets: AtlasAssets, config: PciSeqReferenceConfig
) -> tuple[pd.DataFrame, list[str]]:
    genes = _read_csv(assets.metadata_paths["gene"], ["gene_identifier", "gene_symbol"])
    _unique(genes, "gene_identifier", "Atlas gene metadata")
    genes = genes.set_index("gene_identifier")
    column = "gene_symbol" if config.gene_id_mode == "symbol" else None
    genes["_profile_gene"] = genes[column] if column is not None else genes.index
    available = list(
        dict.fromkeys(genes.loc[genes["_profile_gene"].ne(""), "_profile_gene"])
    )
    if config.gene_names is None:
        output = available
    else:
        requested = list(dict.fromkeys(config.gene_names))
        output = [gene for gene in requested if gene in set(available)]
    if not output:
        raise ValueError(
            "No requested reference genes are available in the chosen identifier mode"
        )
    return genes, output


def _raw_chunk(handle: h5py.File, start: int, stop: int) -> sparse.csr_matrix:
    matrix = handle["X"]
    if isinstance(matrix, h5py.Dataset):
        result = sparse.csr_matrix(np.asarray(matrix[start:stop]), dtype=np.float64)
    else:
        pointers = np.asarray(matrix["indptr"][start : stop + 1])
        left, right = int(pointers[0]), int(pointers[-1])
        result = sparse.csr_matrix(
            (
                np.asarray(matrix["data"][left:right], dtype=np.float64),
                np.asarray(matrix["indices"][left:right]),
                pointers - left,
            ),
            shape=(stop - start, _matrix_shape(handle)[1]),
        )
    if (
        not np.isfinite(result.data).all()
        or (result.data < 0).any()
        or not np.equal(result.data, np.floor(result.data)).all()
    ):
        raise ValueError(
            "Reference X must contain finite nonnegative raw integer counts; "
            "log-scaled data cannot be averaged as counts"
        )
    return result


def _aggregate(
    assets: AtlasAssets, config: PciSeqReferenceConfig, metadata: _ReferenceMetadata
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    plans, found, counts = _expression_plan(assets, metadata, config.chunk_rows)
    support, weights, codes = _support_and_weights(
        metadata.selected, found, config.min_cells_per_type, config.donor_balance
    )
    valid_support = support.loc[support.status == "retained"]
    genes, profile_genes = _gene_profile(assets, config)
    means = np.zeros((len(valid_support), len(profile_genes)), dtype=np.float64)
    for plan in plans:
        if not len(plan.row_indices):
            continue
        with h5py.File(plan.path, "r") as handle:
            if "uns/normalization" in handle:
                normalization = handle["uns/normalization"][()]
                normalization = (
                    normalization.decode()
                    if isinstance(normalization, bytes)
                    else str(normalization)
                )
                if normalization.casefold() not in {"raw", "counts", "raw_counts"}:
                    raise ValueError(
                        "Reference matrix declares unsupported normalization "
                        f"{normalization!r}; supply official raw counts"
                    )
            ids = _axis_ids(handle, "var", "gene_identifier")
            if len(ids) != _matrix_shape(handle)[1] or pd.Index(ids).has_duplicates:
                raise ValueError(
                    "Reference matrix gene identifiers are duplicated or inconsistent"
                )
            if not pd.Index(ids).isin(genes.index).all():
                raise ValueError(
                    "Reference matrix genes are absent from the official gene metadata"
                )
            names = genes["_profile_gene"].reindex(ids).to_numpy()
            output_codes = pd.Index(profile_genes).get_indexer(names)
            source_columns = np.flatnonzero(output_codes >= 0)
            projection = sparse.csr_matrix(
                (
                    np.ones(len(source_columns)),
                    (np.arange(len(source_columns)), output_codes[source_columns]),
                ),
                shape=(len(source_columns), len(profile_genes)),
            )
            for start in range(0, _matrix_shape(handle)[0], config.chunk_rows):
                stop = start + config.chunk_rows
                a, b = np.searchsorted(plan.row_indices, [start, stop])
                positions = plan.metadata_positions[a:b]
                keep = codes[positions] >= 0
                if not keep.any():
                    continue
                rows = plan.row_indices[a:b][keep] - start
                positions = positions[keep]
                matrix = _raw_chunk(handle, start, min(stop, _matrix_shape(handle)[0]))[
                    rows
                ]
                library = np.asarray(matrix.sum(axis=1)).ravel()
                if (library <= 0).any():
                    raise ValueError(
                        "Reference contains zero-count rows with stored zeros; "
                        "eliminate those rows before preparing profiles"
                    )
                matrix = matrix[:, source_columns] @ projection
                factors = config.normalization_target_sum * weights[positions] / library
                reducer = sparse.csr_matrix(
                    (factors, (codes[positions], np.arange(len(positions)))),
                    shape=(len(valid_support), len(positions)),
                )
                update = (reducer @ matrix).tocoo()
                update.sum_duplicates()
                means[update.row, update.col] += update.data
    names = valid_support.type_name.astype(str).tolist()
    if len(set(names)) != len(names):
        names = [
            f"{name} [{identifier}]"
            for name, identifier in zip(names, valid_support.type_id, strict=True)
        ]
    reference = pd.DataFrame(
        means.T, index=pd.Index(profile_genes, name="gene"), columns=names
    )
    positive = (reference.sum(axis=0) > 0).to_numpy()
    support.loc[support.status == "retained", "status"] = np.where(
        positive, "retained", "zero_expression"
    )
    reference = reference.loc[:, positive]
    if reference.empty or not np.isfinite(reference.to_numpy()).all():
        raise ValueError(
            "No finite positive reference profiles remain after expression filtering"
        )
    diagnostics: dict[str, Any] = dict(counts)
    diagnostics.update(
        selected_cells=int((codes >= 0).sum()),
        represented_donor_count=int(
            metadata.selected.loc[codes >= 0, "donor_label"].nunique()
        ),
        retained_types=reference.shape[1],
        output_genes=reference.shape[0],
        donor_balance=config.donor_balance,
        normalization="Arithmetic mean of per-cell linear library-normalized counts",
        normalization_target_sum=config.normalization_target_sum,
        library_sum_scope=(
            "All original matrix genes, before panel restriction and "
            "duplicate symbol collapse"
        ),
        requested_genes_absent=[]
        if config.gene_names is None
        else sorted(set(config.gene_names) - set(profile_genes)),
    )
    return reference, support, {**metadata.diagnostics, **diagnostics}


def _custom(
    config: PciSeqReferenceConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    assert config.reference_path is not None
    reference = _read_reference(config.reference_path)
    original_classes = reference.columns.tolist()
    if config.region_mode == "required" and not config.region_labels:
        raise ValueError(
            "Custom human references require explicit region_labels "
            "and class/region metadata"
        )
    inventory: list[str] = []
    if config.region_labels:
        if config.custom_metadata_path is None:
            raise ValueError(
                "Filtering a prebuilt reference by ROI requires "
                "custom_metadata_path with cell_type and region columns"
            )
        metadata = _read_csv(config.custom_metadata_path)
        type_col = "cell_type" if "cell_type" in metadata else "type_name"
        if type_col not in metadata or "region" not in metadata:
            raise ValueError(
                "Custom ROI metadata requires cell_type and region columns"
            )
        inventory = sorted(metadata.region.unique().tolist())
        unknown = set(config.region_labels) - set(inventory)
        if unknown:
            raise ValueError(
                f"Unknown custom regions: {sorted(unknown)}; available: {inventory}"
            )
        if not metadata[type_col].isin(reference.columns).all():
            raise ValueError(
                "Custom ROI metadata names classes absent from the reference"
            )
        types = metadata.loc[
            metadata.region.isin(config.region_labels), type_col
        ].unique()
        reference = reference.loc[:, reference.columns.isin(types)]
    if config.gene_names is not None:
        requested = list(dict.fromkeys(config.gene_names))
        reference = reference.reindex(
            [gene for gene in requested if gene in reference.index]
        )
    reference = reference.loc[:, reference.sum(axis=0) > 0]
    if reference.empty:
        raise ValueError(
            "Custom reference is empty after ROI, gene, "
            "or positive-expression filtering"
        )
    support = pd.DataFrame(
        {
            "type_id": original_classes,
            "type_name": original_classes,
            "status": [
                "retained" if name in reference.columns else "filtered"
                for name in original_classes
            ],
        }
    )
    return (
        reference,
        support,
        {
            "region_inventory": inventory,
            "resolved_regions": config.region_labels,
            "normalization": (
                "Supplied precomputed linear means; "
                "no implicit expression transformation"
            ),
            "donor_balance": "Already aggregated; not applied",
            "support_policy": (
                "Cell and donor support unavailable for precomputed means"
            ),
            "retained_types": reference.shape[1],
            "output_genes": reference.shape[0],
        },
    )


def _source_spec(
    config: PciSeqReferenceConfig, source: str, fingerprints: Any
) -> dict[str, Any]:
    spec = config.model_dump(
        mode="json",
        exclude={
            "output_dir",
            "reference_root",
            "auto_download",
            "verify_checksums",
            "reference_path",
            "custom_metadata_path",
        },
    )
    spec["region_labels"] = sorted(spec["region_labels"])
    return {
        "algorithm_version": _ALGORITHM_VERSION,
        "source": source,
        "configuration": spec,
        "source_fingerprints": fingerprints,
    }


def _cached_paths(directory: Path) -> dict[str, Path]:
    return {
        "reference_csv": directory / "reference.csv",
        "manifest": directory / "reference_manifest.json",
        "support": directory / "class_support.csv",
    }


def _valid_cache(directory: Path) -> bool:
    paths = _cached_paths(directory)
    if not all(path.is_file() for path in paths.values()):
        return False
    try:
        manifest = json.loads(paths["manifest"].read_text())
        if not isinstance(manifest, dict):
            return False
        checksums = manifest.get("output_checksums")
        if (
            not isinstance(checksums, dict)
            or set(checksums) != {"reference.csv", "class_support.csv"}
            or manifest.get("algorithm_version") != _ALGORITHM_VERSION
            or manifest.get("cache_key") != directory.name
            or manifest.get("reference_sha256") != checksums["reference.csv"]
        ):
            return False
        for filename, digest in checksums.items():
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                or _sha256(directory / filename) != digest
            ):
                return False
        return True
    except (OSError, ValueError, KeyError):
        return False


def prepare_pciseq_reference(
    config: PciSeqReferenceConfig, output_dir: Path | None = None
) -> dict[str, Path]:
    """Create a reproducible linear gene-by-type reference from explicit atlas cells.

    Args:
        config: Source, ROI, taxonomy, cohort, normalization, and cache settings.
        output_dir: Optional publication directory overriding the config value.

    Returns:
        Paths to reference CSV, its provenance manifest, and class support counts.

    Raises:
        ValueError: Filters or input schemas cannot produce a valid reference.
    """
    source = (
        ("wmb" if config.species == "mouse" else "whb")
        if config.source == "auto"
        else config.source
    )
    root = config.reference_root or local_config.require_path(
        "MERXEN_REFERENCES_DIR", "pciSeq reference cache"
    )
    root = Path(root)
    assets: AtlasAssets | None = None
    if source == "custom":
        if config.reference_path is None:
            raise ValueError("Custom reference preparation requires reference_path")
        fingerprints = {"reference_sha256": _sha256(config.reference_path)}
        if config.custom_metadata_path is not None:
            fingerprints["metadata_sha256"] = _sha256(config.custom_metadata_path)
        provenance: dict[str, Any] = {
            "reference_path": str(config.reference_path.resolve())
        }
    else:
        assets = ensure_atlas_assets(
            root,
            source,
            auto_download=config.auto_download,
            verify_checksums=config.verify_checksums,
        )
        fingerprints = assets.provenance["source_fingerprint"]
        provenance = assets.provenance
    spec = _source_spec(config, source, fingerprints)
    cache_key = hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    cache_root = root / "pciseq_prepared"
    cache_root.mkdir(parents=True, exist_ok=True)
    cached = cache_root / cache_key
    with (cache_root / f"{cache_key}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not _valid_cache(cached):
            if cached.exists():
                cached.rename(cache_root / f".{cache_key}.invalid-{uuid.uuid4().hex}")
            staging = Path(tempfile.mkdtemp(prefix=f".{cache_key}.", dir=cache_root))
            try:
                if assets is None:
                    reference, support, diagnostics = _custom(config)
                    resolved_level = {
                        "name": "precomputed",
                        "native_label": "precomputed",
                    }
                    inventory = diagnostics.get("region_inventory", [])
                else:
                    metadata = _atlas_metadata(assets, config)
                    reference, support, diagnostics = _aggregate(
                        assets, config, metadata
                    )
                    resolved_level = {
                        "name": metadata.level,
                        "native_label": metadata.level_label,
                    }
                    inventory = metadata.inventory
                reference.to_csv(staging / "reference.csv")
                support.to_csv(staging / "class_support.csv", index=False)
                manifest = {
                    "format_version": 1,
                    "algorithm_version": _ALGORITHM_VERSION,
                    "cache_key": cache_key,
                    "resolved_source": source,
                    "reference_sha256": _sha256(staging / "reference.csv"),
                    "species": config.species,
                    "resolved_taxonomy_level": resolved_level,
                    "taxonomy_level_policy": (
                        "Auto selects native WMB subclass, WHB cluster, and SEA-AD "
                        "Subclass. These are broader native levels, not a claim "
                        "that atlas ontologies are equivalent."
                    ),
                    "region_inventory": inventory,
                    "specification": spec,
                    "source_provenance": provenance,
                    "diagnostics": diagnostics,
                    "output_checksums": {
                        filename: _sha256(staging / filename)
                        for filename in ("reference.csv", "class_support.csv")
                    },
                }
                (staging / "reference_manifest.json").write_text(
                    json.dumps(manifest, indent=2) + "\n"
                )
                os.replace(staging, cached)
            except BaseException:
                shutil.rmtree(staging, ignore_errors=True)
                raise
    destination = Path(output_dir or config.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    outputs = _cached_paths(destination)
    inputs = (
        [config.reference_path, config.custom_metadata_path]
        if assets is None
        else [*assets.expression_paths, *assets.metadata_paths.values()]
    )
    input_paths = {path.resolve() for path in inputs if path is not None}
    if any(path.resolve() in input_paths for path in outputs.values()):
        raise ValueError(
            "Reference output directory would overwrite an input source file"
        )
    for key, source_path in _cached_paths(cached).items():
        if source_path.resolve() != outputs[key].resolve():
            shutil.copy2(source_path, outputs[key])
    return outputs

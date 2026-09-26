"""The annotation MAP step, per pair x segmentation (plan §3.3, §4.1, §4.4).

``annotate_map`` runs ``CLUSTERING_SQUIDPY_ANNOTATE_MAP``'s work for each
sample of one pair x segmentation, and the standalone ``merxen annotate``
command runs the same code on published ``*_clustered.h5ad`` files:

1. **Load** the prepared H5AD (raw counts in ``X``, every segmented object)
   or a published clustered H5AD (``layers["counts"]``, table cells only);
   remove control features with the shared registry
   (``annotation.panel.ControlRegistry``, identical to
   ``remove_control_features`` on the current panels); take ``total_counts``
   and ``n_genes`` from ``merxen.clustering.cellset.select_table_cells``;
   resolve gene IDs as ``ANNOTATE_PANEL`` does (native ID, pair lookup,
   configured fallback table).
2. **Map table cells only** (``total_counts >= min_counts``), restricted to
   each bundle's panel genes present in the dataset; a missing marker gene
   restricts the lookup (``validate_lookup`` with auto-collapse) and is
   recorded as ``n_missing_panel_genes``. Subset bundles for > 1% missing
   genes are M3b.
3. **MMC per bundle** (``mapmycells_engine.run_mmc``): every required
   primary / secondary bundle, plus the human set-c run on the segmentations
   ``annotation_xplat_sensitivity_segmentations`` names and, for
   ``per_platform`` pairs, the intersection-panel run. The mouse region
   inference and pruned re-map are M6: mouse maps unpruned here.
4. **Write** ``<plat>/<sid>_mmc_<run_id>.parquet`` (tidy, per cell x level)
   and ``map_manifest.json`` (query fingerprint, bundle hashes, engine
   parameters, ctm version, wall time). With ``annotation_reuse_published``
   a run is skipped when a published manifest has the same query
   fingerprint, ``build_hash``, engine parameters and ctm version.
5. **Provisional labels** ``<plat>/<sid>_ct_provisional.parquet``: ``ct_*``
   columns from the raw thresholds alone (hard floor = ``min_counts``, sinks
   and region plausibility from the bundle vocabulary, the parent chain),
   plus the raw engine columns ``mmc_*`` (§4.1). They are for inspection and
   the M3 shadow evaluation only: no floors, resolvability, dataset gate,
   second vote or COP rule (RESOLVE, M4), so they are marked provisional in
   the file name, the parquet metadata and the manifest, and never feed the
   clustered H5AD.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from merxen.annotation.config import AnnotationConfig, AnnotationReferenceSpec
from merxen.annotation.mapmycells_engine import (
    MmcBundle,
    MmcEngineParams,
    aggregate_parent_probability,
    check_ctm_version,
    ctm_commit,
    level_frame,
    query_fingerprint,
    read_tidy_parquet,
    restrict_lookup,
    run_mmc,
    runner_up_column,
    write_query_h5ad,
)
from merxen.annotation.panel import (
    REQUIRED_BUNDLES_FILE,
    SPECIES_ID_PATTERNS,
    AnnotationPanel,
    ControlRegistry,
    DeclaredPanel,
    PanelSource,
    RawPanel,
    RequiredBundles,
    declared_panel,
    load_annotation_panel,
    load_fallback_table,
    pair_symbol_lookup,
    raw_panel_from_var,
)
from merxen.annotation.schema import CellStatus, Columns
from merxen.annotation.store import (
    ANNOTATION_BUILDER_VERSION,
    BUNDLE_MANIFEST_NAME,
    STORE_SCHEMA_VERSION,
    BundleRef,
    ReferenceStore,
    file_sha256,
)
from merxen.annotation.vocab import (
    NEURONS,
    UNASSIGNED_LABEL,
    Species,
)
from merxen.clustering.cellset import select_table_cells

if TYPE_CHECKING:
    from scipy import sparse

logger = logging.getLogger(__name__)

MAP_MANIFEST_NAME: Final = "map_manifest.json"
MAP_MANIFEST_SCHEMA_VERSION: Final = 1
MMC_PARQUET_TEMPLATE: Final = "{sample_id}_mmc_{run_id}.parquet"
PROVISIONAL_SUFFIX: Final = "_ct_provisional.parquet"
PROVISIONAL_METADATA_KEY: Final = b"merxen_ct_provisional"
PROVISIONAL_RULES: Final = (
    "PROVISIONAL (M3): raw bootstrap thresholds only, with the hard floor "
    "(min_counts), sinks and region plausibility from the bundle vocabulary "
    "and the parent chain. No floors, resolvability, dataset gate, second "
    "vote or COP rule: the RESOLVE step (M4) replaces these labels."
)
# Roles MAP maps onto; resolvability and likelihood references are used by
# PREP / v1.1, region shares by the mouse region step (M6).
MAPPED_ROLES: Final[frozenset[str]] = frozenset({"primary", "secondary", "sensitivity"})
RUN_SUFFIXES: Final[dict[str, str]] = {
    "annotation": "",
    "setc_sensitivity": "_setc",
    "intersection_xpanel": "_xpanel",
}
# Short reference names of the raw engine columns (plan §4.1).
ENGINE_PREFIXES: Final[dict[str, str]] = {
    "whb_frontal_supc_clus": "whb",
    "seaad_mr_panel": "seaad",
    "wmb_panel": "wmb",
    "whb_whole_ctx_panel": "whb_wholectx",
}
N_PROCESSORS_ENV: Final = "MERXEN_ANNOTATION_MAP_N_PROCESSORS"
DEFAULT_N_PROCESSORS: Final = 6
SECOND_VOTE_COUNTS: Final = 60
MOUSE_REGION_STEP: Final = "not_run: mouse region inference and pruned re-map are M6"

SampleSource = Literal["prepared", "clustered"]


class MapError(RuntimeError):
    """The MAP step cannot run on its inputs."""


# --------------------------------------------------------------------------
# Inputs


@dataclass(frozen=True)
class MapSample:
    """One sample to map.

    Attributes:
        sample_id: Sample id (``<sid>``).
        platform: ``"MERSCOPE"`` or ``"XENIUM"``.
        h5ad_path: Prepared or published clustered H5AD.
        source: ``"prepared"`` (counts in ``X``, every segmented object) or
            ``"clustered"`` (``layers["counts"]``, table cells only).
    """

    sample_id: str
    platform: str
    h5ad_path: Path
    source: SampleSource


@dataclass(frozen=True)
class MapBundle:
    """One MMC run of a pair x segmentation: a bundle on an annotation panel.

    Attributes:
        run_id: File token: the reference id, ``_setc`` / ``_xpanel`` for the
            set-c and cross-platform runs.
        reference_id: Store id.
        role: Reference role.
        purposes: Every use the run serves (``RequiredBundle.uses``).
        panel_name: Annotation panel name.
        panel: The annotation panel (its genes restrict the query).
        bundle: The bundle.
        spec: The reference spec (bootstrap settings and seed).
    """

    run_id: str
    reference_id: str
    role: str
    purposes: tuple[str, ...]
    panel_name: str | None
    panel: AnnotationPanel
    bundle: MmcBundle
    spec: AnnotationReferenceSpec

    def applies_to(self, platform: str) -> bool:
        """Whether this run maps a sample of a platform."""
        return not self.panel.platforms or platform in self.panel.platforms


def default_n_processors() -> int:
    """Return MAP's worker processes: the env override or 6 (plan §3.3)."""
    value = os.environ.get(N_PROCESSORS_ENV)
    if value:
        try:
            return max(1, int(value))
        except ValueError:
            logger.warning("Ignoring non-integer %s=%r", N_PROCESSORS_ENV, value)
    return DEFAULT_N_PROCESSORS


def reference_spec_for(
    config: AnnotationConfig, reference_id: str
) -> AnnotationReferenceSpec:
    """Return the config's spec of a reference, else its known spec.

    Args:
        config: The annotation config.
        reference_id: Store id.

    Returns:
        The spec.

    Raises:
        MapError: If the reference is unknown.
    """
    from merxen.annotation.config import KNOWN_REFERENCES

    for spec in config.references:
        if spec.reference_id == reference_id:
            return spec
    known = KNOWN_REFERENCES.get(reference_id)
    if known is None:
        raise MapError(f"unknown reference {reference_id!r}")
    return AnnotationReferenceSpec(reference_id=reference_id, **known)


def map_bundles(
    required: RequiredBundles,
    panel_dir: Path | str,
    bundles: Mapping[tuple[str, str | None], MmcBundle],
    config: AnnotationConfig,
    *,
    references: Sequence[str] | None = None,
) -> list[MapBundle]:
    """Return the MMC runs of a pair x segmentation.

    Args:
        required: ``required_bundles.json`` of the pair x segmentation.
        panel_dir: The ``ANNOTATE_PANEL`` output directory (panel files).
        bundles: Bundles by ``(reference_id, panel_hash)``.
        config: The annotation config.
        references: Map only these reference ids (default: every mapped role).

    Returns:
        One run per required bundle with a mapped role, primary first.

    Raises:
        MapError: If a required bundle is missing, or its build is for
            another panel.
    """
    wanted = set(references) if references is not None else None
    runs: list[MapBundle] = []
    for item in required.bundles:
        if item.role not in MAPPED_ROLES or item.panel_file is None:
            continue
        if wanted is not None and item.reference_id not in wanted:
            continue
        bundle = bundles.get((item.reference_id, item.panel_hash))
        if bundle is None:
            raise MapError(
                f"no bundle for {item.reference_id} on panel "
                f"{(item.panel_hash or '-')[:16]} ({item.panel_name}); build it "
                "with merxen annotation-reference-prep"
            )
        if bundle.panel_hash != item.panel_hash:
            raise MapError(
                f"bundle {bundle.path} is for panel {str(bundle.panel_hash)[:16]}, "
                f"not {str(item.panel_hash)[:16]}"
            )
        panel = load_annotation_panel(Path(panel_dir) / item.panel_file)
        purposes = tuple(use.purpose for use in item.uses)
        runs.append(
            MapBundle(
                run_id=item.reference_id + RUN_SUFFIXES.get(purposes[0], ""),
                reference_id=item.reference_id,
                role=item.role,
                purposes=purposes,
                panel_name=item.panel_name,
                panel=panel,
                bundle=bundle,
                spec=reference_spec_for(config, item.reference_id),
            )
        )
    runs.sort(key=lambda run: (run.role != "primary", run.role, run.run_id))
    return runs


def locate_bundle(
    store: ReferenceStore, reference_id: str, panel_hash: str | None
) -> MmcBundle:
    """Find the current-builder bundle of a reference on a panel in a store.

    Standalone runs do not know the source files PREP hashed, so the bundle
    is found by ``(reference_id, panel_hash)`` among the complete bundles
    built by the current builder and store schema versions.

    Args:
        store: The reference store.
        reference_id: Store id.
        panel_hash: Declared-panel hash.

    Returns:
        The bundle.

    Raises:
        MapError: If there is none, or several (pass the bundle explicitly).
    """
    candidates = []
    for entry in store.list():
        if (
            entry.kind != "bundle"
            or entry.reference_id != reference_id
            or entry.panel_hash != panel_hash
        ):
            continue
        manifest = json.loads(
            (entry.path / BUNDLE_MANIFEST_NAME).read_text(encoding="utf-8")
        )
        if (
            manifest.get("builder_version") == ANNOTATION_BUILDER_VERSION
            and manifest.get("schema_version") == STORE_SCHEMA_VERSION
        ):
            candidates.append(entry)
    if not candidates:
        raise MapError(
            f"the store has no builder-v{ANNOTATION_BUILDER_VERSION} bundle of "
            f"{reference_id} on panel {(panel_hash or '-')[:16]}; build it with "
            "merxen annotation-reference-prep"
        )
    if len(candidates) > 1:
        raise MapError(
            f"{len(candidates)} bundles of {reference_id} on panel "
            f"{(panel_hash or '-')[:16]}: "
            + ", ".join(str(entry.path) for entry in candidates)
            + "; pass one with --bundle"
        )
    return MmcBundle.from_dir(candidates[0].path)


# --------------------------------------------------------------------------
# Loading


@dataclass
class LoadedSample:
    """One sample's counts after control removal, with its table cells.

    Attributes:
        sample: The sample.
        obs_names: Object ids, in file order.
        instance_ids: SpatialData instance ids, when the file has them.
        counts: Objects x non-control features (CSR).
        feature_names: Non-control feature names.
        feature_ids: Resolved Ensembl ID per non-control feature (``""``
            when unresolved).
        declared: The sample's declared panel (``ANNOTATE_PANEL`` rules).
        total_counts: Counts per object after control removal.
        n_genes: Detected non-control features per object.
        in_table: Table-cell flag per object.
        min_counts: Table-cell threshold.
        n_below_min_in_clustered: Published table cells below ``min_counts``
            (clustered inputs only: their genes were ``min_cells``-filtered).
    """

    sample: MapSample
    obs_names: pd.Index
    instance_ids: np.ndarray | None
    counts: sparse.csr_matrix
    feature_names: list[str]
    feature_ids: list[str]
    declared: DeclaredPanel
    total_counts: np.ndarray
    n_genes: np.ndarray
    in_table: np.ndarray
    min_counts: int
    n_below_min_in_clustered: int = 0

    @property
    def n_objects(self) -> int:
        """Return the number of objects."""
        return len(self.obs_names)

    @property
    def n_table_cells(self) -> int:
        """Return the number of table cells."""
        return int(np.count_nonzero(self.in_table))

    def sample_fingerprint(self) -> str:
        """Return the fingerprint of all objects on all resolved features."""
        return query_fingerprint(
            self.obs_names,
            self.total_counts,
            [
                gene_id or f"unresolved:{name}"
                for name, gene_id in zip(
                    self.feature_names, self.feature_ids, strict=True
                )
            ],
        )


def _read_elem(node: Any) -> Any:
    try:
        from anndata.io import read_elem
    except ImportError:  # anndata < 0.11
        from anndata.experimental import read_elem
    return read_elem(node)


def read_h5ad_counts(
    path: Path | str, source: SampleSource
) -> tuple[pd.Index, pd.DataFrame, sparse.csr_matrix, np.ndarray | None]:
    """Read ids, ``var`` and raw counts of an H5AD without loading the rest.

    Args:
        path: The H5AD.
        source: ``"prepared"`` (counts in ``X``) or ``"clustered"``
            (``layers["counts"]``; ``X`` there is log-normalised).

    Returns:
        ``(obs_names, var, counts, instance_ids)``.

    Raises:
        MapError: If a clustered file has no ``counts`` layer.
    """
    import h5py
    from scipy import sparse

    with h5py.File(path, "r") as handle:
        obs = handle["obs"]
        index_key = obs.attrs.get("_index", "_index")
        obs_names = pd.Index([str(value) for value in _read_elem(obs[index_key])])
        var = pd.DataFrame(_read_elem(handle["var"]))
        if source == "clustered":
            if "layers" not in handle or "counts" not in handle["layers"]:
                raise MapError(f"{path} has no layers['counts'] (raw counts)")
            matrix = _read_elem(handle["layers"]["counts"])
        else:
            matrix = _read_elem(handle["X"])
        instance_ids = None
        instance_key = None
        if "uns" in handle and "spatialdata_attrs" in handle["uns"]:
            attrs = handle["uns"]["spatialdata_attrs"]
            if "instance_key" in attrs:
                instance_key = str(_read_elem(attrs["instance_key"]))
        if instance_key and instance_key in obs:
            instance_ids = np.asarray(_read_elem(obs[instance_key]))
    counts = sparse.csr_matrix(matrix)
    if counts.shape != (len(obs_names), len(var)):
        raise MapError(f"{path}: counts shape {counts.shape} does not fit obs x var")
    return obs_names, var, counts, instance_ids


def _raw_panel(var: pd.DataFrame, path: Path) -> RawPanel:
    return raw_panel_from_var(
        var, source=PanelSource(kind="prepared_h5ad_var", path=str(path))
    )


def load_samples(
    samples: Sequence[MapSample],
    config: AnnotationConfig,
    *,
    min_counts: int,
) -> list[LoadedSample]:
    """Load the samples of a pair x segmentation for mapping (step 1).

    Controls are removed with the shared registry and gene IDs resolved as in
    ``ANNOTATE_PANEL`` (native ID, the pair's symbol lookup, the configured
    fallback table), so the query genes are the panel's IDs.

    Args:
        samples: The samples.
        config: The annotation config (panel settings, species).
        min_counts: Table-cell threshold (the clustering ``min_counts``).

    Returns:
        The loaded samples, in input order.
    """
    import anndata as ad

    registry = ControlRegistry.from_config(config.panel)
    species: Species = config.species
    pattern = SPECIES_ID_PATTERNS[species]
    reads = []
    for sample in samples:
        obs_names, var, counts, instance_ids = read_h5ad_counts(
            sample.h5ad_path, sample.source
        )
        reads.append((sample, obs_names, var, counts, instance_ids))
    raws = [_raw_panel(var, sample.h5ad_path) for sample, _, var, _, _ in reads]
    lookup = pair_symbol_lookup(raws, species)
    fallback = load_fallback_table(config.panel.gene_id_fallback_csv, species)
    loaded: list[LoadedSample] = []
    for (sample, obs_names, var, counts, instance_ids), raw in zip(
        reads, raws, strict=True
    ):
        if len(raw.features) != len(var):
            raise MapError(f"{sample.h5ad_path}: var has unnamed features")
        declared = declared_panel(
            raw,
            species=species,
            platform=sample.platform,
            sample_id=sample.sample_id,
            registry=registry,
            pair_lookup=lookup,
            fallback=fallback,
        )
        keep: list[int] = []
        names: list[str] = []
        ids: list[str] = []
        for position, row in enumerate(raw.features.itertuples(index=False)):
            reason = registry.control_reason(
                str(row.name),
                platform=sample.platform,
                feature_type=str(row.feature_type),
                has_native_id=bool(row.native_id),
            )
            if reason is not None:
                continue
            native = str(row.native_id)
            gene_id = (
                native
                if native and pattern.fullmatch(native)
                else declared.symbol_to_id.get(str(row.symbol), "")
            )
            keep.append(position)
            names.append(str(row.name))
            ids.append(gene_id)
        gene_counts = counts[:, keep].tocsr()
        selection = select_table_cells(
            ad.AnnData(X=gene_counts, obs=pd.DataFrame(index=obs_names)),
            min_counts,
            compute_n_genes=True,
        )
        assert selection.n_genes is not None
        in_table = selection.is_table_cell.copy()
        n_below = 0
        if sample.source == "clustered":
            # The published table is the table: its objects passed min_counts
            # before min_cells dropped rare genes, so a few may now sum lower.
            n_below = int(np.count_nonzero(~in_table))
            if n_below:
                logger.warning(
                    "%s: %d published table cell(s) sum below min_counts %d on "
                    "the min_cells-filtered genes; they stay table cells",
                    sample.sample_id,
                    n_below,
                    min_counts,
                )
            in_table = np.ones(len(obs_names), dtype=bool)
        logger.info(
            "%s: %d objects, %d table cells, %d controls removed, %d of %d "
            "features resolved",
            sample.sample_id,
            len(obs_names),
            int(in_table.sum()),
            len(var) - len(keep),
            sum(1 for gene_id in ids if gene_id),
            len(keep),
        )
        loaded.append(
            LoadedSample(
                sample=sample,
                obs_names=obs_names,
                instance_ids=instance_ids,
                counts=gene_counts,
                feature_names=names,
                feature_ids=ids,
                declared=declared,
                total_counts=np.asarray(selection.total_counts),
                n_genes=np.asarray(selection.n_genes),
                in_table=in_table,
                min_counts=int(min_counts),
                n_below_min_in_clustered=n_below,
            )
        )
    return loaded


@dataclass(frozen=True)
class SampleQuery:
    """The query of one sample on one panel (table cells x panel genes).

    Attributes:
        counts: Table cells x query genes (CSR).
        cell_ids: Table-cell ids.
        total_counts: Their total counts (all non-control features).
        gene_ids: Query gene IDs (panel order).
        missing_gene_ids: Panel genes the dataset lacks.
        fingerprint: ``query_fingerprint`` of the query.
    """

    counts: sparse.csr_matrix
    cell_ids: pd.Index
    total_counts: np.ndarray
    gene_ids: list[str]
    missing_gene_ids: list[str]
    fingerprint: str


def build_sample_query(loaded: LoadedSample, panel: AnnotationPanel) -> SampleQuery:
    """Restrict a sample's table cells to a panel's genes (step 2).

    Features that resolve to one ID are summed; panel genes absent from the
    dataset are listed.

    Args:
        loaded: The loaded sample.
        panel: The annotation panel.

    Returns:
        The query.

    Raises:
        MapError: If no panel gene is present.
    """
    from scipy import sparse

    columns: dict[str, list[int]] = {}
    for position, gene_id in enumerate(loaded.feature_ids):
        if gene_id:
            columns.setdefault(gene_id, []).append(position)
    present = [gene_id for gene_id in panel.ensembl_ids if gene_id in columns]
    missing = [gene_id for gene_id in panel.ensembl_ids if gene_id not in columns]
    if not present:
        raise MapError(
            f"{loaded.sample.sample_id}: none of the {panel.n_genes} panel genes "
            f"of {panel.name} is in the dataset"
        )
    rows, cols = [], []
    for column, gene_id in enumerate(present):
        for position in columns[gene_id]:
            rows.append(position)
            cols.append(column)
    selector = sparse.csr_matrix(
        (np.ones(len(rows), dtype=loaded.counts.dtype), (rows, cols)),
        shape=(len(loaded.feature_ids), len(present)),
    )
    table = np.flatnonzero(loaded.in_table)
    query = (loaded.counts[table] @ selector).tocsr()
    cell_ids = loaded.obs_names[table]
    total_counts = loaded.total_counts[table]
    return SampleQuery(
        counts=query,
        cell_ids=cell_ids,
        total_counts=total_counts,
        gene_ids=present,
        missing_gene_ids=missing,
        fingerprint=query_fingerprint(cell_ids, total_counts, present),
    )


# --------------------------------------------------------------------------
# Manifest


class _MapModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MapRunRecord(_MapModel):
    """One MMC run of one sample in ``map_manifest.json``.

    Attributes:
        run_id: File token (reference id + ``_setc`` / ``_xpanel``).
        reference_id: Store id.
        role: Reference role.
        purposes: Uses the run serves.
        panel_name: Annotation panel name.
        panel_hash: Panel hash.
        n_panel_genes: Genes of the panel.
        build_hash: Bundle ``build_hash``.
        bundle_path: Bundle directory.
        builder_version: Builder version of the bundle.
        bundle_lookup_sha256: Marker-content sha256 of the bundle's lookup.
        lookup_sha256: That of the lookup mapped with.
        lookup_restricted: Whether missing genes restricted the lookup.
        collapsed_parents: Parents the restriction auto-collapsed.
        query_fingerprint: ``query_fingerprint`` (reuse key).
        n_cells: Mapped (table) cells.
        n_query_genes: Query genes.
        n_missing_panel_genes: Panel genes absent from the dataset.
        missing_panel_genes: Their IDs.
        engine_params: ``MmcEngineParams`` (reuse key).
        ctm_version: ctm version of the mapping.
        ctm_commit: ctm commit, when known.
        effective_config: Settings the extended JSON recorded.
        wall_s: Mapper wall time (of the original run when reused).
        peak_rss_gb: Mapper peak RSS, when measured.
        parquet: Tidy parquet, relative to the manifest's directory.
        parquet_sha256: Its sha256.
        extended_json: Kept gzipped extended JSON (relative), if any.
        reused: Whether this run was taken from a published manifest.
        reused_from: The published parquet it came from.
    """

    run_id: str
    reference_id: str
    role: str
    purposes: list[str]
    panel_name: str | None
    panel_hash: str | None
    n_panel_genes: int | None
    build_hash: str
    bundle_path: str
    builder_version: int | None
    bundle_lookup_sha256: str | None
    lookup_sha256: str | None
    lookup_restricted: bool
    collapsed_parents: list[str] = Field(default_factory=list)
    query_fingerprint: str
    n_cells: int
    n_query_genes: int
    n_missing_panel_genes: int
    missing_panel_genes: list[str] = Field(default_factory=list)
    engine_params: dict[str, Any]
    ctm_version: str | None
    ctm_commit: str | None = None
    effective_config: dict[str, Any] = Field(default_factory=dict)
    wall_s: float | None = None
    peak_rss_gb: float | None = None
    parquet: str
    parquet_sha256: str
    extended_json: str | None = None
    reused: bool = False
    reused_from: str | None = None


class MapSampleRecord(_MapModel):
    """One sample in ``map_manifest.json``.

    Attributes:
        sample_id: Sample id.
        platform: Platform.
        source: ``"prepared"`` or ``"clustered"``.
        h5ad_path: Input H5AD.
        h5ad_size: Its size in bytes.
        h5ad_mtime_ns: Its modification time.
        sample_fingerprint: Fingerprint of all objects on all features.
        n_objects: Objects in the input.
        n_table_cells: Table cells (mapped).
        min_counts: Table-cell threshold.
        n_below_min_in_clustered: Published table cells below it (clustered).
        declared_panel_hash: Hash of the sample's declared panel.
        n_features: Non-control features.
        n_resolved_features: Features with an Ensembl ID.
        controls_removed: Control features removed, per reason.
        runs: MMC runs by run id.
        provisional_labels: Provisional ``ct_*`` parquet (relative), if any.
        provisional_summary: Confident share of table cells per level.
    """

    sample_id: str
    platform: str
    source: SampleSource
    h5ad_path: str
    h5ad_size: int | None = None
    h5ad_mtime_ns: int | None = None
    sample_fingerprint: str
    n_objects: int
    n_table_cells: int
    min_counts: int
    n_below_min_in_clustered: int = 0
    declared_panel_hash: str
    n_features: int
    n_resolved_features: int
    controls_removed: dict[str, int] = Field(default_factory=dict)
    runs: dict[str, MapRunRecord] = Field(default_factory=dict)
    provisional_labels: str | None = None
    provisional_summary: dict[str, float] = Field(default_factory=dict)


class MapManifest(_MapModel):
    """``map_manifest.json`` of one pair x segmentation (plan §3.3 step 6).

    Attributes:
        schema_version: ``MAP_MANIFEST_SCHEMA_VERSION``.
        pair_id: Pair id.
        segmentation: Segmentation.
        species: Species.
        anatomical_region: Human region.
        created_at: ISO time.
        ctm_version: Installed ctm version.
        ctm_commit: ctm commit.
        n_processors: Worker processes.
        min_counts: Table-cell threshold.
        mouse_region_step: Status of the mouse region step (M6).
        panel_status: ``required_bundles.json`` status: ``"refused"`` when
            no panel can be annotated, so nothing was mapped (RESOLVE writes
            statuses only; plan §3.3 step 2).
        panel_reasons: Why the panel was refused.
        provisional_rules: What the provisional labels apply.
        thresholds: The raw thresholds the provisional labels used.
        wall_time_s: Wall time of the MAP call.
        samples: Samples by id.
    """

    schema_version: int = MAP_MANIFEST_SCHEMA_VERSION
    pair_id: str | None
    segmentation: str | None
    species: Species
    anatomical_region: str | None
    created_at: str
    ctm_version: str | None
    ctm_commit: str | None
    n_processors: int
    min_counts: int
    mouse_region_step: str | None = None
    panel_status: Literal["ok", "refused"] = "ok"
    panel_reasons: list[str] = Field(default_factory=list)
    provisional_rules: str = PROVISIONAL_RULES
    thresholds: dict[str, Any] = Field(default_factory=dict)
    wall_time_s: float | None = None
    samples: dict[str, MapSampleRecord] = Field(default_factory=dict)

    def write(self, path: Path | str) -> Path:
        """Write the manifest atomically.

        Args:
            path: Output file.

        Returns:
            The written path.
        """
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".partial")
        temporary.write_text(
            json.dumps(self.model_dump(mode="json"), indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
        return output


def load_map_manifest(path: Path | str) -> MapManifest:
    """Read a ``map_manifest.json``.

    Args:
        path: The file.

    Returns:
        The manifest.
    """
    return MapManifest.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _reusable_parquet(
    published: MapManifest | None,
    published_dir: Path | None,
    *,
    sample_id: str,
    run_id: str,
    fingerprint: str,
    build_hash: str,
    engine_params: Mapping[str, Any],
    ctm_version: str | None,
) -> tuple[Path, MapRunRecord] | None:
    """Return the published parquet of an identical run, if any."""
    if published is None or published_dir is None:
        return None
    sample = published.samples.get(sample_id)
    record = sample.runs.get(run_id) if sample is not None else None
    if record is None:
        return None
    reasons = []
    if record.query_fingerprint != fingerprint:
        reasons.append("query fingerprint")
    if record.build_hash != build_hash:
        reasons.append("build_hash")
    if record.engine_params != dict(engine_params):
        reasons.append("engine parameters")
    if record.ctm_version != ctm_version:
        reasons.append("ctm version")
    path = published_dir / record.parquet
    if not reasons and (
        not path.is_file() or file_sha256(path) != record.parquet_sha256
    ):
        reasons.append("parquet missing or changed")
    if reasons:
        logger.info(
            "%s %s: published run not reused (%s differ)",
            sample_id,
            run_id,
            ", ".join(reasons),
        )
        return None
    return path, record


# --------------------------------------------------------------------------
# Provisional labels and raw engine columns


def _engine_prefix(run: MapBundle) -> str:
    base = ENGINE_PREFIXES.get(run.reference_id, run.reference_id)
    return f"mmc_{base}{RUN_SUFFIXES.get(run.purposes[0], '')}"


def engine_columns(
    tidy: pd.DataFrame,
    prefix: str,
    obs_names: pd.Index,
    *,
    runner_up_levels: Iterable[str] = (),
) -> pd.DataFrame:
    """Return the raw engine columns of one run (plan §4.1), one row per object.

    Args:
        tidy: The run's tidy table.
        prefix: Column prefix, e.g. ``"mmc_whb"``.
        obs_names: Object ids (rows; unmapped objects get nulls).
        runner_up_levels: Level names that also get
            ``<prefix>_<level>_runner_up_<k>_{name,bp}``.

    Returns:
        ``<prefix>_<level>_{label,name,bp,agg,corr}`` per level.
    """
    runner_levels = set(runner_up_levels)
    columns: dict[str, Any] = {}
    for level_name in tidy["level_name"].astype(str).unique():
        frame = level_frame(tidy, level_name).reindex(obs_names)
        stem = f"{prefix}_{level_name}"
        columns[f"{stem}_label"] = frame["assignment"].astype(object)
        columns[f"{stem}_name"] = frame["name"].astype(object)
        columns[f"{stem}_bp"] = frame["bp"].astype(np.float32)
        columns[f"{stem}_agg"] = frame["aggregate_probability"].astype(np.float32)
        columns[f"{stem}_corr"] = frame["avg_correlation"].astype(np.float32)
        if level_name in runner_levels:
            for rank in range(1, 6):
                columns[f"{stem}_runner_up_{rank}_name"] = frame[
                    runner_up_column(rank, "name")
                ].astype(object)
                columns[f"{stem}_runner_up_{rank}_bp"] = frame[
                    runner_up_column(rank, "probability")
                ].astype(np.float32)
    result = pd.DataFrame(columns, index=obs_names)
    for column in result.columns:
        if column.endswith(("_label", "_name")):
            result[column] = result[column].astype("category")
    return result


def _vocab_lookup(
    vocab: pd.DataFrame | None, level: str, column: str
) -> dict[str, str | None]:
    if vocab is None or column not in vocab.columns:
        return {}
    rows = vocab[vocab["level"] == level]
    return {
        str(node): (str(value) if str(value) not in {"", "nan", "None"} else None)
        for node, value in zip(rows["node"], rows[column], strict=True)
    }


def _vocab_flag(
    vocab: pd.DataFrame | None, level: str, column: str, default: bool
) -> dict[str, bool]:
    values = _vocab_lookup(vocab, level, column)
    return {
        node: default if value is None else value.strip().lower() == "true"
        for node, value in values.items()
    }


@dataclass
class _LevelColumns:
    """The provisional ``ct_<level>_*`` values of one level (all objects)."""

    name: np.ndarray
    raw: np.ndarray
    corr: np.ndarray
    runner_up: np.ndarray
    margin: np.ndarray
    status: np.ndarray

    @classmethod
    def empty(cls, n_objects: int) -> _LevelColumns:
        return cls(
            name=np.full(n_objects, None, dtype=object),
            raw=np.full(n_objects, np.nan, dtype=np.float64),
            corr=np.full(n_objects, np.nan, dtype=np.float64),
            runner_up=np.full(n_objects, None, dtype=object),
            margin=np.full(n_objects, np.nan, dtype=np.float64),
            status=np.full(n_objects, CellStatus.LOW_COUNTS.value, dtype=object),
        )

    def to_columns(self, level: str) -> dict[str, Any]:
        return {
            Columns.level(level, "name"): pd.Categorical(self.name),
            Columns.level(level, "raw"): self.raw.astype(np.float32),
            Columns.level(level, "corr"): self.corr.astype(np.float32),
            Columns.level(level, "runner_up"): pd.Categorical(self.runner_up),
            Columns.level(level, "margin"): self.margin.astype(np.float32),
            Columns.level(level, "status"): pd.Categorical(self.status),
        }


def _level_values(
    frame: pd.DataFrame,
    positions: np.ndarray,
    n_objects: int,
    *,
    group_of: Mapping[str, str | None] | None,
    threshold: float | np.ndarray,
    parent_confident: np.ndarray | None,
    implausible: np.ndarray | None = None,
    applicable: np.ndarray | None = None,
) -> _LevelColumns:
    """Fill one level from a tidy level frame (rows = mapped cells).

    ``group_of`` aggregates the level to a coarser class (lineage, broad, NT);
    without it the level is the node itself.
    """
    values = _LevelColumns.empty(n_objects)
    if group_of is None:
        names = frame["name"].astype(object).to_numpy()
        raw = frame["bp"].to_numpy(np.float64)
        runner = frame[runner_up_column(1, "name")].astype(object).to_numpy()
        runner_probability = frame[runner_up_column(1, "probability")].to_numpy(
            np.float64
        )
        margin = raw - np.nan_to_num(runner_probability, nan=0.0)
    else:
        aggregated = aggregate_parent_probability(frame, group_of)
        names = aggregated.classes
        raw = aggregated.probability
        runner = aggregated.runner_up_class
        margin = raw - aggregated.runner_up_probability
    corr = frame["avg_correlation"].to_numpy(np.float64)
    threshold_values = (
        np.asarray(threshold, dtype=np.float64)
        if np.ndim(threshold)
        else np.full(len(frame), float(threshold))
    )
    status = np.where(
        np.nan_to_num(raw, nan=-1.0) >= threshold_values,
        CellStatus.CONFIDENT.value,
        CellStatus.LOW_CONFIDENCE.value,
    ).astype(object)
    if implausible is not None:
        status[implausible] = CellStatus.IMPLAUSIBLE.value
    if parent_confident is not None:
        status[~parent_confident] = CellStatus.PARENT_UNRESOLVED.value
    if applicable is not None:
        status[~applicable] = CellStatus.NOT_APPLICABLE.value
        names = np.asarray(names, dtype=object).copy()
        names[~applicable] = None
        runner = np.asarray(runner, dtype=object).copy()
        runner[~applicable] = None
        raw = np.where(applicable, raw, np.nan)
        margin = np.where(applicable, margin, np.nan)
    values.name[positions] = names
    values.raw[positions] = raw
    values.corr[positions] = corr
    values.runner_up[positions] = runner
    values.margin[positions] = margin
    values.status[positions] = status
    return values


def _confident(values: _LevelColumns, positions: np.ndarray) -> np.ndarray:
    return np.asarray(values.status[positions] == CellStatus.CONFIDENT.value)


def provisional_labels(
    loaded: LoadedSample,
    runs: Sequence[tuple[MapBundle, pd.DataFrame]],
    config: AnnotationConfig,
    *,
    pair_id: str | None,
    segmentation: str | None,
) -> pd.DataFrame:
    """Return provisional ``ct_*`` labels from the raw thresholds (inspection).

    Human: lineage, broad and NT aggregate the WHB supercluster bootstrap
    probabilities over the assigned node's class (E1 / E2); lineage is
    ``implausible`` for sinks and region-implausible superclusters; broad,
    NT and supercluster need a confident parent; ``seaad_subclass`` applies
    SEA-AD's raw thresholds (0.55 below 60 counts, 0.45 from 60) under a
    confident WHB broad. Mouse: broad and NT aggregate the WMB class level,
    class and subclass use their bootstrap probabilities. Objects below
    ``min_counts`` are ``low_counts``. See ``PROVISIONAL_RULES`` for what is
    not applied.

    Args:
        loaded: The sample.
        runs: ``(run, tidy)`` of the sample's MMC runs.
        config: The annotation config (thresholds, region).
        pair_id: Pair id.
        segmentation: Segmentation.

    Returns:
        One row per object: identity, ``ct_<level>_{name,raw,corr,runner_up,
        margin,status}``, ``ct_final_level`` / ``ct_final_name`` and the raw
        engine columns of every run.
    """
    obs_names = loaded.obs_names
    n_objects = loaded.n_objects
    thresholds = config.thresholds
    region = config.anatomical_region
    species = config.species
    frame: dict[str, Any] = {
        Columns.CELL_ID: obs_names.astype(str),
        Columns.PAIR_ID: pair_id,
        Columns.SAMPLE_ID: loaded.sample.sample_id,
        Columns.PLATFORM: loaded.sample.platform,
        Columns.SEGMENTATION: segmentation,
        Columns.SPECIES: species,
        Columns.TOTAL_COUNTS: loaded.total_counts.astype(np.int64),
        Columns.N_GENES: loaded.n_genes.astype(np.int64),
        Columns.IN_TABLE: loaded.in_table,
    }
    if loaded.instance_ids is not None:
        frame[Columns.INSTANCE_ID] = loaded.instance_ids
    primary = next(
        (
            (run, tidy)
            for run, tidy in runs
            if run.role == "primary" and "annotation" in run.purposes
        ),
        None,
    )
    secondary = next(
        (
            (run, tidy)
            for run, tidy in runs
            if run.role == "secondary" and "annotation" in run.purposes
        ),
        None,
    )
    levels: dict[str, _LevelColumns] = {}
    order: list[str] = []
    if primary is not None:
        run, tidy = primary
        position_of = pd.Index(obs_names)
        vocab = run.bundle.vocab()
        leaf_level = run.bundle.levels[0]
        mapped = level_frame(tidy, leaf_level)
        positions = position_of.get_indexer(mapped.index)
        if (positions < 0).any():
            raise MapError("MMC results name cells absent from the sample")
        labels = mapped["assignment"].astype(object).to_numpy()
        broad_of = _vocab_lookup(vocab, leaf_level, "broad_class")
        nt_of = _vocab_lookup(vocab, leaf_level, "nt")
        if species == "human":
            lineage_of = _vocab_lookup(vocab, leaf_level, "lineage")
            sink = _vocab_flag(vocab, leaf_level, "sink", False)
            plausible = _vocab_flag(
                vocab, leaf_level, f"region_plausible_{region}", True
            )
            implausible = np.array(
                [
                    sink.get(str(label), False) or not plausible.get(str(label), True)
                    for label in labels
                ],
                dtype=bool,
            )
            lineage = _level_values(
                mapped,
                positions,
                n_objects,
                group_of=lineage_of,
                threshold=thresholds.whb_broad,
                parent_confident=None,
                implausible=implausible,
            )
            broad = _level_values(
                mapped,
                positions,
                n_objects,
                group_of=broad_of,
                threshold=thresholds.whb_broad,
                parent_confident=_confident(lineage, positions),
            )
            is_neuron = np.array(
                [broad_of.get(str(label)) == NEURONS for label in labels], dtype=bool
            )
            nt_groups = {
                node: (nt if broad_of.get(node) == NEURONS else None)
                for node, nt in nt_of.items()
            }
            nt = _level_values(
                mapped,
                positions,
                n_objects,
                group_of=nt_groups,
                threshold=thresholds.whb_broad,
                parent_confident=_confident(broad, positions),
                applicable=is_neuron,
            )
            supercluster = _level_values(
                mapped,
                positions,
                n_objects,
                group_of=None,
                threshold=thresholds.whb_supercluster,
                parent_confident=_confident(broad, positions),
            )
            levels.update(
                lineage=lineage, broad=broad, nt=nt, supercluster=supercluster
            )
            order = ["lineage", "broad", "nt", "supercluster"]
            if secondary is not None:
                sea_run, sea_tidy = secondary
                sea_levels = sea_run.bundle.levels
                sea_level = sea_levels[min(1, len(sea_levels) - 1)]
                sea = level_frame(sea_tidy, sea_level)
                sea_positions = position_of.get_indexer(sea.index)
                counts = loaded.total_counts[sea_positions]
                sea_threshold = np.where(
                    counts < SECOND_VOTE_COUNTS,
                    thresholds.seaad_subclass_below60,
                    thresholds.seaad_subclass_from60,
                )
                broad_ok = broad.status[sea_positions] == CellStatus.CONFIDENT.value
                levels["seaad_subclass"] = _level_values(
                    sea,
                    sea_positions,
                    n_objects,
                    group_of=None,
                    threshold=sea_threshold,
                    parent_confident=broad_ok,
                )
        else:
            broad = _level_values(
                mapped,
                positions,
                n_objects,
                group_of=broad_of,
                threshold=thresholds.wmb_class,
                parent_confident=None,
            )
            cls = _level_values(
                mapped,
                positions,
                n_objects,
                group_of=None,
                threshold=thresholds.wmb_class,
                parent_confident=_confident(broad, positions),
            )
            is_neuron = np.array(
                [broad_of.get(str(label)) == NEURONS for label in labels], dtype=bool
            )
            nt = _level_values(
                mapped,
                positions,
                n_objects,
                group_of={
                    node: (value if broad_of.get(node) == NEURONS else None)
                    for node, value in nt_of.items()
                },
                threshold=thresholds.wmb_class,
                parent_confident=_confident(cls, positions),
                applicable=is_neuron,
            )
            levels.update(broad=broad, **{"class": cls}, nt=nt)
            order = ["broad", "class", "nt", "subclass"]
            if len(run.bundle.levels) > 1:
                sub_level = run.bundle.levels[1]
                sub = level_frame(tidy, sub_level)
                sub_positions = position_of.get_indexer(sub.index)
                levels["subclass"] = _level_values(
                    sub,
                    sub_positions,
                    n_objects,
                    group_of=None,
                    threshold=thresholds.wmb_subclass,
                    parent_confident=cls.status[sub_positions]
                    == CellStatus.CONFIDENT.value,
                )
    for level, values in levels.items():
        frame.update(values.to_columns(level))
    final_level = np.full(n_objects, "none", dtype=object)
    final_name = np.full(n_objects, UNASSIGNED_LABEL, dtype=object)
    for level in order:
        level_values = levels.get(level)
        if level_values is None:
            continue
        confident = level_values.status == CellStatus.CONFIDENT.value
        final_level[confident] = level
        final_name[confident] = level_values.name[confident]
    frame[Columns.CT_FINAL_LEVEL] = pd.Categorical(final_level)
    frame[Columns.CT_FINAL_NAME] = pd.Categorical(final_name)
    result = pd.DataFrame(frame, index=pd.RangeIndex(n_objects))
    for column in (
        Columns.PAIR_ID,
        Columns.SAMPLE_ID,
        Columns.PLATFORM,
        Columns.SEGMENTATION,
        Columns.SPECIES,
    ):
        result[column] = result[column].astype("category")
    engine_frames = []
    for run, tidy in runs:
        runner_levels = (
            [str(tidy["level_name"].astype(str).iloc[0])]
            if run.role == "primary" and len(tidy)
            else []
        )
        engine = engine_columns(
            tidy, _engine_prefix(run), obs_names, runner_up_levels=runner_levels
        )
        engine_frames.append(engine.reset_index(drop=True))
    if engine_frames:
        result = pd.concat([result, *engine_frames], axis=1)
    return result


def write_provisional_parquet(
    labels: pd.DataFrame, path: Path | str, metadata: Mapping[str, Any]
) -> Path:
    """Write the provisional labels, marked provisional in the parquet schema.

    Args:
        labels: ``provisional_labels`` output.
        path: Output ``<sid>_ct_provisional.parquet``.
        metadata: JSON metadata (rules, thresholds, bundle hashes).

    Returns:
        The written path.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(labels, preserve_index=False)
    schema_metadata = dict(table.schema.metadata or {})
    schema_metadata[PROVISIONAL_METADATA_KEY] = json.dumps(
        {"provisional": True, "rules": PROVISIONAL_RULES, **dict(metadata)},
        sort_keys=True,
    ).encode("utf-8")
    table = table.replace_schema_metadata(schema_metadata)
    temporary = output.with_name(output.name + ".partial")
    pq.write_table(table, temporary)
    temporary.replace(output)
    return output


def _confident_shares(labels: pd.DataFrame) -> dict[str, float]:
    in_table = labels[Columns.IN_TABLE].to_numpy(bool)
    denominator = max(int(in_table.sum()), 1)
    shares: dict[str, float] = {}
    for column in labels.columns:
        if column.startswith("ct_") and column.endswith("_status"):
            level = column.removeprefix("ct_").removesuffix("_status")
            confident = labels[column].astype(str).to_numpy() == "confident"
            shares[level] = round(float((confident & in_table).sum()) / denominator, 6)
    return shares


# --------------------------------------------------------------------------
# annotate_map


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def annotate_map(
    samples: Sequence[MapSample],
    runs: Sequence[MapBundle],
    config: AnnotationConfig,
    *,
    output_dir: Path | str,
    pair_id: str | None,
    segmentation: str | None,
    n_processors: int | None = None,
    work_dir: Path | str | None = None,
    reuse_from: Path | str | None = None,
    write_provisional: bool = True,
) -> MapManifest:
    """Run the MAP step for the samples of one pair x segmentation.

    Args:
        samples: The samples (prepared or published clustered H5ADs).
        runs: The MMC runs (``map_bundles``).
        config: The annotation config, coupled to the clustering
            ``min_counts`` (``require_min_counts``).
        output_dir: ``annotation_out`` (``<plat>/`` parquets and
            ``map_manifest.json``).
        pair_id: Pair id.
        segmentation: Segmentation.
        n_processors: MMC worker processes (default: ``default_n_processors``).
        work_dir: Task-local scratch for queries and extended JSONs (default:
            ``<output_dir>/.work``, removed at the end).
        reuse_from: Directory of a published ``map_manifest.json`` whose
            identical runs are reused (``annotation_reuse_published``);
            ``None`` disables reuse.
        write_provisional: Write the provisional ``ct_*`` parquet.

    Returns:
        The manifest (also written to ``<output_dir>/map_manifest.json``).

    Raises:
        CtmVersionError: If the installed ctm is not ``config.ctm_version``.
        MapError: If the inputs cannot be mapped.
        MmcEngineError: If a mapping fails.
    """
    start = time.monotonic()
    min_counts = config.require_min_counts()
    installed = check_ctm_version(config.ctm_version)
    processes = n_processors or default_n_processors()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    owns_work_dir = work_dir is None
    scratch = Path(work_dir) if work_dir is not None else output / ".work"
    scratch.mkdir(parents=True, exist_ok=True)
    published: MapManifest | None = None
    published_dir: Path | None = None
    if reuse_from is not None:
        candidate = Path(reuse_from) / MAP_MANIFEST_NAME
        if candidate.is_file():
            published = load_map_manifest(candidate)
            published_dir = Path(reuse_from)
    manifest = MapManifest(
        pair_id=pair_id,
        segmentation=segmentation,
        species=config.species,
        anatomical_region=config.anatomical_region,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        ctm_version=installed,
        ctm_commit=ctm_commit(),
        n_processors=processes,
        min_counts=min_counts,
        mouse_region_step=MOUSE_REGION_STEP if config.species == "mouse" else None,
        thresholds=config.thresholds.model_dump(mode="json"),
    )
    try:
        loaded_samples = load_samples(samples, config, min_counts=min_counts)
        for loaded in loaded_samples:
            record = _map_sample(
                loaded,
                runs,
                config,
                output=output,
                scratch=scratch,
                processes=processes,
                installed=installed,
                published=published,
                published_dir=published_dir,
                pair_id=pair_id,
                segmentation=segmentation,
                write_provisional=write_provisional,
            )
            manifest.samples[loaded.sample.sample_id] = record
            # Written after every sample, so an interrupted run keeps the
            # finished samples reusable.
            manifest.wall_time_s = round(time.monotonic() - start, 3)
            manifest.write(output / MAP_MANIFEST_NAME)
    finally:
        if owns_work_dir:
            shutil.rmtree(scratch, ignore_errors=True)
    manifest.wall_time_s = round(time.monotonic() - start, 3)
    manifest.write(output / MAP_MANIFEST_NAME)
    return manifest


def _map_sample(
    loaded: LoadedSample,
    runs: Sequence[MapBundle],
    config: AnnotationConfig,
    *,
    output: Path,
    scratch: Path,
    processes: int,
    installed: str,
    published: MapManifest | None,
    published_dir: Path | None,
    pair_id: str | None,
    segmentation: str | None,
    write_provisional: bool,
) -> MapSampleRecord:
    sample = loaded.sample
    sample_dir = output / sample.platform.lower()
    sample_dir.mkdir(parents=True, exist_ok=True)
    stat = sample.h5ad_path.stat()
    record = MapSampleRecord(
        sample_id=sample.sample_id,
        platform=sample.platform,
        source=sample.source,
        h5ad_path=str(sample.h5ad_path),
        h5ad_size=stat.st_size,
        h5ad_mtime_ns=stat.st_mtime_ns,
        sample_fingerprint=loaded.sample_fingerprint(),
        n_objects=loaded.n_objects,
        n_table_cells=loaded.n_table_cells,
        min_counts=loaded.min_counts,
        n_below_min_in_clustered=loaded.n_below_min_in_clustered,
        declared_panel_hash=loaded.declared.panel_hash,
        n_features=len(loaded.feature_ids),
        n_resolved_features=sum(1 for gene_id in loaded.feature_ids if gene_id),
        controls_removed={
            reason: len(names)
            for reason, names in loaded.declared.controls_removed.items()
        },
    )
    tidies: list[tuple[MapBundle, pd.DataFrame]] = []
    for run in runs:
        if not run.applies_to(sample.platform):
            continue
        params = MmcEngineParams.from_reference_spec(run.spec, n_processors=processes)
        query = build_sample_query(loaded, run.panel)
        parquet = sample_dir / MMC_PARQUET_TEMPLATE.format(
            sample_id=sample.sample_id, run_id=run.run_id
        )
        reusable = (
            _reusable_parquet(
                published,
                published_dir,
                sample_id=sample.sample_id,
                run_id=run.run_id,
                fingerprint=query.fingerprint,
                build_hash=run.bundle.build_hash,
                engine_params=params.reuse_key(),
                ctm_version=installed,
            )
            if config.reuse_published
            else None
        )
        if reusable is not None:
            source_path, old = reusable
            if source_path.resolve() != parquet.resolve():
                shutil.copy2(source_path, parquet)
            logger.info("%s %s: reused %s", sample.sample_id, run.run_id, source_path)
            run_record = old.model_copy(
                update={
                    "parquet": _relative(parquet, output),
                    "reused": True,
                    "reused_from": str(source_path.resolve()),
                    "bundle_path": str(run.bundle.path),
                }
            )
        else:
            run_record = _run_one(
                loaded,
                run,
                query,
                params,
                config,
                parquet=parquet,
                output=output,
                scratch=scratch / f"{sample.sample_id}_{run.run_id}",
            )
        record.runs[run.run_id] = run_record
        tidy, _ = read_tidy_parquet(parquet)
        tidies.append((run, tidy))
    if write_provisional and tidies:
        labels = provisional_labels(
            loaded, tidies, config, pair_id=pair_id, segmentation=segmentation
        )
        path = sample_dir / f"{sample.sample_id}{PROVISIONAL_SUFFIX}"
        write_provisional_parquet(
            labels,
            path,
            {
                "pair_id": pair_id,
                "segmentation": segmentation,
                "sample_id": sample.sample_id,
                "thresholds": config.thresholds.model_dump(mode="json"),
                "runs": {run.run_id: run.bundle.build_hash for run, _ in tidies},
            },
        )
        record.provisional_labels = _relative(path, output)
        record.provisional_summary = _confident_shares(labels)
    return record


def _run_one(
    loaded: LoadedSample,
    run: MapBundle,
    query: SampleQuery,
    params: MmcEngineParams,
    config: AnnotationConfig,
    *,
    parquet: Path,
    output: Path,
    scratch: Path,
) -> MapRunRecord:
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        query_path = write_query_h5ad(
            query.counts,
            query.cell_ids,
            query.gene_ids,
            scratch / "query.h5ad",
        )
        restricted = restrict_lookup(
            run.bundle, query.gene_ids, scratch / "lookup.restricted.json"
        )
        if query.missing_gene_ids:
            logger.warning(
                "%s %s: %d of %d panel genes absent from the dataset: %s",
                loaded.sample.sample_id,
                run.run_id,
                len(query.missing_gene_ids),
                run.panel.n_genes,
                ", ".join(query.missing_gene_ids[:10]),
            )
        result = run_mmc(
            query_path,
            run.bundle,
            params,
            output_parquet=parquet,
            work_dir=scratch,
            expected_ctm_version=config.ctm_version,
            lookup_path=restricted.path,
            keep_extended_json=config.keep_extended_json,
            log_dir=output / "logs",
            run_metadata={
                "run_id": run.run_id,
                "sample_id": loaded.sample.sample_id,
                "query_fingerprint": query.fingerprint,
                "purposes": list(run.purposes),
            },
        )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    collapsed = (
        [parent.key for parent in restricted.validation.collapsed]
        if restricted.validation is not None
        else []
    )
    return MapRunRecord(
        run_id=run.run_id,
        reference_id=run.reference_id,
        role=run.role,
        purposes=list(run.purposes),
        panel_name=run.panel_name,
        panel_hash=run.panel.panel_hash,
        n_panel_genes=run.panel.n_genes,
        build_hash=run.bundle.build_hash,
        bundle_path=str(run.bundle.path),
        builder_version=run.bundle.builder_version,
        bundle_lookup_sha256=run.bundle.lookup_sha256,
        lookup_sha256=restricted.lookup_sha256,
        lookup_restricted=restricted.path is not None,
        collapsed_parents=collapsed,
        query_fingerprint=query.fingerprint,
        n_cells=result.n_cells,
        n_query_genes=result.n_query_genes,
        n_missing_panel_genes=len(query.missing_gene_ids),
        missing_panel_genes=query.missing_gene_ids,
        engine_params=params.reuse_key(),
        ctm_version=result.ctm_version,
        ctm_commit=result.ctm_commit,
        effective_config=result.effective_config,
        wall_s=result.wall_s,
        peak_rss_gb=result.peak_rss_gb,
        parquet=_relative(parquet, output),
        parquet_sha256=file_sha256(parquet),
        extended_json=(
            _relative(result.extended_json, output)
            if result.extended_json is not None
            else None
        ),
    )


# --------------------------------------------------------------------------
# Standalone inputs (merxen annotate)


@dataclass(frozen=True)
class PublishedLayout:
    """Where a published clustered H5AD sits in a results tree.

    Attributes:
        results_root: ``<root>`` of ``<root>/<pair>/<seg>/clustering_squidpy/
            clustering_squidpy_out/<platform>/<sid>_clustered.h5ad``.
        pair_id: The pair.
        segmentation: The segmentation.
        platform: The platform (upper case).
        sample_id: ``<sid>``.
    """

    results_root: Path | None
    pair_id: str | None
    segmentation: str | None
    platform: str | None
    sample_id: str


CLUSTERED_SUFFIX: Final = "_clustered.h5ad"


def published_layout(path: Path | str) -> PublishedLayout:
    """Parse the results-tree position of a published clustered H5AD.

    Args:
        path: ``…/<pair>/<seg>/clustering_squidpy/clustering_squidpy_out/
            <platform>/<sid>_clustered.h5ad`` (other layouts give ``None``
            fields except the sample id).

    Returns:
        The layout.
    """
    resolved = Path(path).resolve()
    sample_id = resolved.name.removesuffix(CLUSTERED_SUFFIX).removesuffix(".h5ad")
    parents = resolved.parents
    in_tree = (
        len(parents) > 5
        and parents[1].name == "clustering_squidpy_out"
        and parents[2].name == "clustering_squidpy"
    )
    platform = parents[0].name.upper() if len(parents) else None
    if platform not in {"MERSCOPE", "XENIUM"}:
        platform = next(
            (
                name
                for name in ("MERSCOPE", "XENIUM")
                if sample_id.upper().endswith("_" + name)
            ),
            None,
        )
    return PublishedLayout(
        results_root=parents[5] if in_tree else None,
        pair_id=parents[4].name if in_tree else None,
        segmentation=parents[3].name if in_tree else None,
        platform=platform,
        sample_id=sample_id,
    )


def check_output_outside_inputs(output_dir: Path | str, inputs: Iterable[Path]) -> None:
    """Refuse an output directory inside a results tree or an input's folder.

    The standalone ``merxen annotate`` never writes into published results
    (R3): not below the results root of a published clustered H5AD, and not
    in any input's directory.

    Args:
        output_dir: The requested output directory.
        inputs: Input files.

    Raises:
        MapError: If the output lies inside a protected directory.
    """
    target = Path(output_dir).resolve()
    for path in inputs:
        layout = published_layout(path)
        protected = [Path(path).resolve().parent]
        if layout.results_root is not None:
            protected.append(layout.results_root)
        for directory in protected:
            if target == directory or directory in target.parents:
                raise MapError(
                    f"output {target} lies inside {directory}, which holds "
                    f"published inputs ({path}); write annotation outputs "
                    "elsewhere (never into the results tree)"
                )


def write_view_manifest(samples: Sequence[MapSample], directory: Path) -> Path:
    """Write a ``manifest.json`` that lets ``compute_panel`` read the inputs.

    ``prepared_samples`` joins absolute manifest paths as they are, so the
    view points at the published files without copying them.

    Args:
        samples: The samples.
        directory: Where ``manifest.json`` goes.

    Returns:
        The directory.
    """
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "samples": {
            sample.sample_id: str(sample.h5ad_path.resolve()) for sample in samples
        }
    }
    (directory / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    return directory


def load_required(
    panel_dir: Path | str, *, allow_refused: bool = False
) -> RequiredBundles:
    """Read ``required_bundles.json`` of an ``ANNOTATE_PANEL`` output.

    Args:
        panel_dir: The directory.
        allow_refused: Return a refused panel instead of raising (pipeline
            runs, which record the refusal with ``write_refused_manifest``).

    Returns:
        The required bundles.

    Raises:
        MapError: If the directory has none, or the panel was refused and
            ``allow_refused`` is false.
    """
    path = Path(panel_dir) / REQUIRED_BUNDLES_FILE
    if not path.is_file():
        raise MapError(f"{path} is missing (run merxen annotation-panel first)")
    required = RequiredBundles.model_validate_json(path.read_text(encoding="utf-8"))
    if required.status != "ok" and not allow_refused:
        raise MapError(
            f"the panel of {required.pair_id} {required.segmentation} was refused: "
            + "; ".join(required.reasons)
        )
    return required


def write_refused_manifest(
    required: RequiredBundles,
    config: AnnotationConfig,
    *,
    output_dir: Path | str,
    pair_id: str | None,
    segmentation: str | None,
    n_processors: int | None = None,
) -> MapManifest:
    """Record a refused panel: ``map_manifest.json`` without any run.

    A refused panel is a data-quality outcome, not an infrastructure error
    (plan §3.1 failure semantics): MAP maps nothing and writes this manifest,
    and RESOLVE (M4) writes statuses only.

    Args:
        required: The refused ``required_bundles.json``.
        config: The annotation config, coupled to the clustering
            ``min_counts``.
        output_dir: ``annotation_out``.
        pair_id: Pair id (default: the panel's).
        segmentation: Segmentation (default: the panel's).
        n_processors: MMC worker processes the task reserved.

    Returns:
        The written manifest.

    Raises:
        MapError: If the panel was not refused.
    """
    from merxen.annotation.mapmycells_engine import installed_ctm_version

    if required.status != "refused":
        raise MapError("write_refused_manifest needs a refused panel")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = MapManifest(
        pair_id=pair_id or required.pair_id,
        segmentation=segmentation or required.segmentation,
        species=config.species,
        anatomical_region=config.anatomical_region,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        ctm_version=installed_ctm_version(),
        ctm_commit=ctm_commit(),
        n_processors=n_processors or default_n_processors(),
        min_counts=config.require_min_counts(),
        mouse_region_step=MOUSE_REGION_STEP if config.species == "mouse" else None,
        panel_status="refused",
        panel_reasons=list(required.reasons),
        thresholds=config.thresholds.model_dump(mode="json"),
        wall_time_s=0.0,
    )
    manifest.write(output / MAP_MANIFEST_NAME)
    logger.warning(
        "%s %s: panel refused, nothing mapped: %s",
        manifest.pair_id,
        manifest.segmentation,
        "; ".join(required.reasons) or "no reason recorded",
    )
    return manifest


def bundle_ref_bundle(ref_path: Path | str) -> tuple[tuple[str, str | None], MmcBundle]:
    """Read a ``bundle_ref.json`` and its bundle.

    Args:
        ref_path: The file.

    Returns:
        ``((reference_id, panel_hash), bundle)``.

    Raises:
        MmcEngineError: If the bundle does not match the reference.
    """
    ref = BundleRef.model_validate_json(Path(ref_path).read_text(encoding="utf-8"))
    return (ref.reference_id, ref.panel_hash), MmcBundle.from_bundle_ref(ref)

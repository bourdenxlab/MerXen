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
   resolve gene IDs as ``ANNOTATE_PANEL`` does (``annotation.gene_ids``:
   native ID, pair lookup, the local gene table, aliases, overrides). A
   published clustered H5AD declares the features of its control-filter
   record, so ``min_cells`` filtering never changes its declared panel.
2. **Map table cells only** (``total_counts >= min_counts``), restricted to
   each bundle's panel genes present in the dataset; a missing marker gene
   restricts the lookup (``validate_lookup`` with auto-collapse) and is
   recorded as ``n_missing_panel_genes``. When a missing gene is a root
   marker, a parent is left with fewer than ``weak_parent_markers`` markers
   or more than ``subset_bundle_missing_frac`` of the panel is missing
   (``panel.subset_bundle_trigger``), the run needs a **subset bundle**: MAP
   writes the subset panel (``subset_panels/<sid>_<run_id>.panel_genes.json``,
   the parent's family, or its own above ``own_family_missing_frac``) and
   maps with the store's bundle on it when one exists; otherwise it maps
   with the restricted lookup and records the request
   (``MapRunRecord.subset_bundle``) for ``annotation-reference-prep``.
3. **MMC per bundle use** (``mapmycells_engine.run_mmc``): every use of a
   required primary / secondary bundle (``RequiredBundle.uses``) is a run on
   that use's panel file: the annotation panel of each platform, the human
   set-c run on the segmentations ``annotation_xplat_sensitivity_segmentations``
   names and, for ``per_platform`` pairs, the intersection-panel run
   (``_xpanel``). Two uses on the same gene set are mapped once and recorded
   under both run ids. The mouse region inference and pruned re-map are M6:
   mouse maps unpruned here.
4. **Write** ``<plat>/<sid>_mmc_<run_id>.parquet`` (tidy, per cell x level)
   and ``map_manifest.json`` (query fingerprint, bundle hashes, engine
   parameters, ctm version, wall time). With ``annotation_reuse_published``
   a run is skipped when a published manifest has the same query
   fingerprint, ``build_hash``, engine parameters, ctm version, tidy schema
   version and (restricted) lookup; an unreadable published manifest only
   disables reuse.
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
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from merxen.annotation.config import AnnotationConfig, AnnotationReferenceSpec
from merxen.annotation.gene_ids import (
    ResolutionRules,
    gene_id_sources,
    summing_matrix,
)
from merxen.annotation.mapmycells_engine import (
    EXTENDED_JSON_GZ_SUFFIX,
    TIDY_SCHEMA_VERSION,
    MmcBundle,
    MmcEngineParams,
    RestrictedLookup,
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
    write_tidy_parquet,
)
from merxen.annotation.panel import (
    REQUIRED_BUNDLES_FILE,
    AnnotationPanel,
    ControlRegistry,
    DeclaredPanel,
    RawPanel,
    RequiredBundles,
    SubsetBundleTrigger,
    declared_native_ids,
    declared_panel,
    feature_columns,
    load_annotation_panel,
    pair_symbol_lookup,
    raw_panel_from_h5ad,
    subset_panel,
    subset_trigger_for_config,
)
from merxen.annotation.provenance import annotation_manifest_filename
from merxen.annotation.reference import read_lookup
from merxen.annotation.schema import (
    CellStatus,
    Columns,
    label_table_filename,
    meets_threshold,
    safe_token,
)
from merxen.annotation.store import (
    ANNOTATION_BUILDER_VERSION,
    BUNDLE_MANIFEST_NAME,
    STORE_SCHEMA_VERSION,
    BundleRef,
    ReferenceStore,
    StoreEntry,
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

    from merxen.annotation.composition import SectionComposition
    from merxen.annotation.consensus import HumanCalls, HumanResolution
    from merxen.annotation.diagnostics import TrustDecision
    from merxen.annotation.provenance import (
        AnnotationProvenance,
        ReferenceProvenance,
        ResolvabilityProvenance,
    )
    from merxen.annotation.resolvability import ResolvabilityTables
    from merxen.annotation.thresholds import EmissionPlan

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
MOUSE_REGION_STEP: Final = "not_run: mouse region inference and pruned re-map are M6"

SUBSET_PANELS_DIR: Final = "subset_panels"
SUBSET_PANEL_TEMPLATE: Final = "{sample_id}_{run_id}.panel_genes.json"

SampleSource = Literal["prepared", "clustered"]
# (reference_id, subset panel_hash) -> the store's subset bundle, if built.
SubsetBundleFinder = Callable[[str, str], MmcBundle | None]


class MapError(RuntimeError):
    """The MAP step cannot run on its inputs."""


class AmbiguousSubsetBundleError(MapError):
    """The store holds several current-builder bundles on one subset panel.

    Attributes:
        candidates: The bundle directories.
    """

    def __init__(self, message: str, candidates: Sequence[str]) -> None:
        super().__init__(message)
        self.candidates = list(candidates)


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
        declared_ids_file: A platform panel file (e.g. the Xenium
            ``gene_panel.json``) giving the native IDs of the declared
            features a clustered H5AD's ``min_cells`` filter dropped
            (``raw_panel_from_h5ad``).
    """

    sample_id: str
    platform: str
    h5ad_path: Path
    source: SampleSource
    declared_ids_file: Path | None = None


@dataclass(frozen=True)
class MapBundle:
    """One MMC run of a pair x segmentation: a bundle on an annotation panel.

    Attributes:
        run_id: File token: the reference id, ``_setc`` / ``_xpanel`` for the
            set-c and cross-platform runs.
        reference_id: Store id.
        role: Reference role.
        purposes: The use this run serves (one ``RequiredBundle.uses``
            purpose; ``map_bundles`` makes one run per use).
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


def _platform_sets_overlap(first: AnnotationPanel, second: AnnotationPanel) -> bool:
    if not first.platforms or not second.platforms:
        return True
    return bool(set(first.platforms) & set(second.platforms))


def map_bundles(
    required: RequiredBundles,
    panel_dir: Path | str,
    bundles: Mapping[tuple[str, str | None], MmcBundle],
    config: AnnotationConfig,
    *,
    references: Sequence[str] | None = None,
) -> list[MapBundle]:
    """Return the MMC runs of a pair x segmentation, one per bundle use.

    ``required_bundles.json`` lists a bundle once per ``(reference,
    panel_hash)`` and names every purpose in ``uses`` (panel.py
    ``RequiredBundle``). MAP selects runs by use: each use with a mapped
    purpose (``RUN_SUFFIXES``) becomes its own run, on that use's panel file
    (whose platforms decide which samples it maps) and with that purpose's
    run id. Two uses of one bundle on the same gene set (the intersection
    panel equal to a platform panel of a ``per_platform`` pair, set c equal
    to set a) therefore give two run ids; ``annotate_map`` maps their query
    once and records it under both.

    Args:
        required: ``required_bundles.json`` of the pair x segmentation.
        panel_dir: The ``ANNOTATE_PANEL`` output directory (panel files).
        bundles: Bundles by ``(reference_id, panel_hash)``.
        config: The annotation config.
        references: Map only these reference ids (default: every mapped role).

    Returns:
        One run per mapped use of a required bundle, primary first.

    Raises:
        MapError: If a required bundle is missing, its build is for another
            panel, a use's panel file has another hash, or two runs would
            write one run id for the same platform.
    """
    wanted = set(references) if references is not None else None
    runs: list[MapBundle] = []
    panels: dict[str, AnnotationPanel] = {}
    for item in required.bundles:
        if item.role not in MAPPED_ROLES:
            continue
        if wanted is not None and item.reference_id not in wanted:
            continue
        uses = [
            use
            for use in item.uses
            if use.purpose in RUN_SUFFIXES and use.panel_file is not None
        ]
        if not uses:
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
        spec = reference_spec_for(config, item.reference_id)
        for use in uses:
            assert use.panel_file is not None
            if use.panel_file not in panels:
                panels[use.panel_file] = load_annotation_panel(
                    Path(panel_dir) / use.panel_file
                )
            panel = panels[use.panel_file]
            if panel.panel_hash != item.panel_hash:
                raise MapError(
                    f"{use.panel_file} has panel hash {panel.panel_hash[:16]}, but "
                    f"required_bundles.json lists {str(item.panel_hash)[:16]} for "
                    f"{item.reference_id} ({use.purpose})"
                )
            run = MapBundle(
                run_id=item.reference_id + RUN_SUFFIXES[use.purpose],
                reference_id=item.reference_id,
                role=item.role,
                purposes=(use.purpose,),
                panel_name=use.panel_name,
                panel=panel,
                bundle=bundle,
                spec=spec,
            )
            for other in runs:
                if other.run_id == run.run_id and _platform_sets_overlap(
                    other.panel, run.panel
                ):
                    raise MapError(
                        f"two uses of {item.reference_id} give run id "
                        f"{run.run_id} for the same platform ({other.panel_name}, "
                        f"{run.panel_name})"
                    )
            runs.append(run)
    runs.sort(key=lambda run: (run.role != "primary", run.role, run.run_id))
    return runs


def _prefilter_matches(manifest: Mapping[str, Any], config: AnnotationConfig) -> bool:
    """Whether a bundle's large-panel prefilter is the one a run config asks for.

    A large panel may have two bundles, built with and without the per-parent
    marker prefilter (its method, version and settings enter
    ``build_hash_payload["large_panel_prefilter"]``; ``None`` without it).
    The run config asks for none when ``large_panel_marker_prefilter`` is
    ``none`` or the panel has at most ``large_panel_genes`` genes, else for
    its method and cap (M3b review 2).
    """
    payload = manifest.get("build_hash_payload") or {}
    prefilter = payload.get("large_panel_prefilter")
    panel = payload.get("panel") or {}
    n_genes = panel.get("n_genes") if isinstance(panel, Mapping) else None
    wanted = (
        config.panel.large_panel_marker_prefilter != "none"
        and isinstance(n_genes, int)
        and n_genes > config.panel.large_panel_genes
    )
    if not wanted:
        return prefilter is None
    if not isinstance(prefilter, Mapping):
        return False
    settings = prefilter.get("settings") or {}
    return bool(
        prefilter.get("method") == config.panel.large_panel_marker_prefilter
        and isinstance(settings, Mapping)
        and settings.get("cap") == config.panel.large_panel_prefilter_cap
    )


def _current_bundles(
    store: ReferenceStore,
    reference_id: str,
    panel_hash: str | None,
    config: AnnotationConfig | None = None,
) -> list[StoreEntry]:
    """Complete current-builder bundles of a reference on a panel (store entries).

    With ``config``, only bundles built with the large-panel prefilter the
    config asks for (``_prefilter_matches``).
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
            and (config is None or _prefilter_matches(manifest, config))
        ):
            candidates.append(entry)
    return candidates


def _resolvability_version(entry: StoreEntry) -> int | None:
    """The resolvability version a bundle's self-map tables were written with."""
    manifest = json.loads((entry.path / BUNDLE_MANIFEST_NAME).read_text("utf-8"))
    record = (manifest.get("builder_output") or {}).get("resolvability") or {}
    value = record.get("resolvability_version") if isinstance(record, dict) else None
    return value if isinstance(value, int) else None


def _prefer_current_resolvability(candidates: list[StoreEntry]) -> list[StoreEntry]:
    """Among several bundles, keep those with the current self-map tables.

    A ``RESOLVABILITY_VERSION`` bump gives rebuilt bundles a new
    ``build_hash`` next to the old ones on the same panel; a standalone run
    then takes the current tables. Bundles without a self-map, or several
    current ones, stay ambiguous. When no candidate has the current tables
    (a panel not rebuilt since the bump), the older ones are kept and a
    warning names their versions: their self-map tables are stale until the
    panel is rebuilt (a pipeline run's PREP rebuilds them).
    """
    from merxen.annotation.resolvability import RESOLVABILITY_VERSION

    versions = [_resolvability_version(entry) for entry in candidates]
    current = [
        entry
        for entry, version in zip(candidates, versions, strict=True)
        if version == RESOLVABILITY_VERSION
    ]
    if current:
        return current if len(candidates) > 1 else candidates
    stale = [
        (entry, version)
        for entry, version in zip(candidates, versions, strict=True)
        if version is not None
    ]
    if stale:
        logger.warning(
            "no bundle of %s on panel %s has the current self-map tables "
            "(resolvability version %d); %s: their resolvability tables are "
            "stale until the panel is rebuilt with merxen "
            "annotation-reference-prep",
            candidates[0].reference_id,
            str(candidates[0].panel_hash or "-")[:16],
            RESOLVABILITY_VERSION,
            ", ".join(
                f"{entry.path.name[:16]} has version {version}"
                for entry, version in stale
            ),
        )
    return candidates


def locate_bundle(
    store: ReferenceStore,
    reference_id: str,
    panel_hash: str | None,
    *,
    config: AnnotationConfig | None = None,
) -> MmcBundle:
    """Find the current-builder bundle of a reference on a panel in a store.

    Standalone runs do not know the source files PREP hashed, so the bundle
    is found by ``(reference_id, panel_hash)`` among the complete bundles
    built by the current builder and store schema versions (and, with
    ``config``, with the large-panel prefilter it asks for: a 5K panel has a
    bundle with and one without it); of several, the one whose self-map
    tables have the current ``RESOLVABILITY_VERSION``.

    Args:
        store: The reference store.
        reference_id: Store id.
        panel_hash: Declared-panel hash.
        config: The run's annotation config (large-panel prefilter).

    Returns:
        The bundle.

    Raises:
        MapError: If there is none, or several (pass the bundle explicitly).
    """
    candidates = _prefer_current_resolvability(
        _current_bundles(store, reference_id, panel_hash, config)
    )
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
        undetected_declared_ids: Declared panel genes absent from ``var``
            (a clustered H5AD's ``min_cells`` filter dropped them): present
            in the panel with zero counts, never missing (M3b review 2).
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
    undetected_declared_ids: tuple[str, ...] = ()

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


def _raw_panel(var: pd.DataFrame, sample: MapSample) -> RawPanel:
    declared_ids = (
        declared_native_ids(sample.declared_ids_file)
        if sample.declared_ids_file is not None
        else None
    )
    return raw_panel_from_h5ad(sample.h5ad_path, var=var, declared_ids=declared_ids)


def load_samples(
    samples: Sequence[MapSample],
    config: AnnotationConfig,
    *,
    min_counts: int,
) -> list[LoadedSample]:
    """Load the samples of a pair x segmentation for mapping (step 1).

    Controls are removed with the shared registry and gene IDs resolved as in
    ``ANNOTATE_PANEL`` (``annotation.gene_ids``), so the query genes are the
    panel's IDs and each feature gets its declared decision.

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
    reads = []
    for sample in samples:
        obs_names, var, counts, instance_ids = read_h5ad_counts(
            sample.h5ad_path, sample.source
        )
        reads.append((sample, obs_names, var, counts, instance_ids))
    raws = [_raw_panel(var, sample) for sample, _, var, _, _ in reads]
    lookup = pair_symbol_lookup(raws, species)
    sources = gene_id_sources(config.panel, species, pair_lookup=lookup)
    rules = ResolutionRules.from_config(config.panel)
    loaded: list[LoadedSample] = []
    for (sample, obs_names, var, counts, instance_ids), raw in zip(
        reads, raws, strict=True
    ):
        declared = declared_panel(
            raw,
            species=species,
            platform=sample.platform,
            sample_id=sample.sample_id,
            registry=registry,
            sources=sources,
            rules=rules,
        )
        try:
            is_gene, column_ids = feature_columns(
                var, declared=declared, platform=sample.platform, registry=registry
            )
        except ValueError as error:
            raise MapError(f"{sample.h5ad_path}: {error}") from error
        keep = [int(position) for position in np.flatnonzero(is_gene)]
        var_names = [str(name) for name in var.index]
        names = [var_names[position] for position in keep]
        ids = [column_ids[position] for position in keep]
        undetected = tuple(sorted(set(declared.ensembl_ids) - set(ids)))
        if undetected:
            logger.info(
                "%s: %d declared panel gene(s) absent from var (dropped by "
                "min_cells) are mapped with zero counts: %s",
                sample.sample_id,
                len(undetected),
                ", ".join(undetected[:10]),
            )
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
                undetected_declared_ids=undetected,
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
        undetected_gene_ids: Query genes with zero counts because a
            clustered H5AD's ``min_cells`` filter dropped them.
    """

    counts: sparse.csr_matrix
    cell_ids: pd.Index
    total_counts: np.ndarray
    gene_ids: list[str]
    missing_gene_ids: list[str]
    fingerprint: str
    undetected_gene_ids: list[str] = field(default_factory=list)


def build_sample_query(loaded: LoadedSample, panel: AnnotationPanel) -> SampleQuery:
    """Restrict a sample's table cells to a panel's genes (step 2).

    Features that resolve to one ID are summed; panel genes absent from the
    dataset are listed. Declared genes a clustered H5AD's ``min_cells``
    filter dropped (``LoadedSample.undetected_declared_ids``) are query
    genes with zero counts, as the prepared H5AD would give them almost
    everywhere, so they neither restrict the lookup nor trigger a subset
    bundle.

    Args:
        loaded: The loaded sample.
        panel: The annotation panel.

    Returns:
        The query.

    Raises:
        MapError: If no panel gene is present.
    """
    detected = {gene_id for gene_id in loaded.feature_ids if gene_id}
    in_data = detected | set(loaded.undetected_declared_ids)
    present = [gene_id for gene_id in panel.ensembl_ids if gene_id in in_data]
    missing = [gene_id for gene_id in panel.ensembl_ids if gene_id not in in_data]
    undetected = [gene_id for gene_id in present if gene_id not in detected]
    if not present:
        raise MapError(
            f"{loaded.sample.sample_id}: none of the {panel.n_genes} panel genes "
            f"of {panel.name} is in the dataset"
        )
    # One column per present panel gene, the sum of every feature resolving
    # to it (e.g. H2AX and H2AFX; plan §8.4).
    selector = summing_matrix(
        list(loaded.feature_ids), present, dtype=loaded.counts.dtype
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
        undetected_gene_ids=undetected,
    )


# --------------------------------------------------------------------------
# Manifest


class _MapModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SubsetBundleRecord(_MapModel):
    """A run's subset-bundle decision (plan §3.3 step 1).

    Attributes:
        trigger: ``panel.subset_bundle_trigger`` on the bundle's panel and
            the sample's genes.
        status: ``used`` (the store had the subset bundle and the run mapped
            with it), ``requested`` (the run mapped with the parent bundle,
            its lookup restricted to the genes present; build the subset
            panel with ``merxen annotation-reference-prep --panel-genes``;
            always so in a pipeline task, whose bundles come from PREP) or
            ``ambiguous`` (the store holds several subset bundles; mapped
            as ``requested``).
        subset_panel_file: The subset panel, relative to the manifest.
        subset_panel_hash: Its ``panel_hash``.
        parent_panel_hash: The bundle panel it was cut from.
        parent_build_hash: The parent bundle.
        subset_build_hash: The subset bundle mapped with (``used``).
        candidates: The store's subset bundles (``ambiguous``).
    """

    trigger: SubsetBundleTrigger
    status: Literal["used", "requested", "ambiguous"]
    subset_panel_file: str
    subset_panel_hash: str
    parent_panel_hash: str
    parent_build_hash: str
    subset_build_hash: str | None = None
    candidates: list[str] = Field(default_factory=list)


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
        undetected_declared_genes: Panel genes of the declared panel that a
            clustered H5AD's ``min_cells`` filter dropped, mapped with zero
            counts (not missing).
        engine_params: ``MmcEngineParams`` (reuse key).
        ctm_version: ctm version of the mapping.
        ctm_commit: ctm commit, when known.
        effective_config: Settings the extended JSON recorded.
        wall_s: Mapper wall time (of the original run when reused).
        peak_rss_gb: Mapper peak RSS, when measured.
        parquet: Tidy parquet, relative to the manifest's directory.
        parquet_sha256: Its sha256.
        tidy_schema_version: ``TIDY_SCHEMA_VERSION`` of the parquet (reuse
            key; ``None`` in manifests written before it was recorded, which
            are never reused).
        extended_json: Kept gzipped extended JSON (relative), if any.
        reused: Whether this run was taken from a published manifest.
        reused_from: The published parquet it came from.
        same_mapping_as: Run id of this sample whose mapping this run shares
            (two uses of one bundle on the same query are mapped once).
        subset_bundle: The subset-bundle decision when the sample lacks
            enough panel genes (``None`` when no subset bundle is needed).
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
    undetected_declared_genes: list[str] = Field(default_factory=list)
    engine_params: dict[str, Any]
    ctm_version: str | None
    ctm_commit: str | None = None
    effective_config: dict[str, Any] = Field(default_factory=dict)
    wall_s: float | None = None
    peak_rss_gb: float | None = None
    parquet: str
    parquet_sha256: str
    tidy_schema_version: int | None = None
    extended_json: str | None = None
    reused: bool = False
    reused_from: str | None = None
    same_mapping_as: str | None = None
    subset_bundle: SubsetBundleRecord | None = None


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
        panel_status: ``"refused"`` when the sample's own panel was refused
            (a ``per_platform`` pair can refuse one platform's panel), so it
            has no run.
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
    panel_status: Literal["ok", "refused"] = "ok"
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


def load_published_manifest(directory: Path | str) -> MapManifest | None:
    """Read a published ``map_manifest.json`` for reuse, or ``None``.

    Reuse is an optimisation, so a manifest that cannot serve it never fails
    the task: a missing, unreadable or invalid file (the ``-stub-run``
    manifest, an older or newer layout) or another
    ``MAP_MANIFEST_SCHEMA_VERSION`` disables reuse with a warning.

    Args:
        directory: The published ``annotation_map_out`` directory.

    Returns:
        The manifest, or ``None`` when it cannot be reused.
    """
    path = Path(directory) / MAP_MANIFEST_NAME
    if not path.is_file():
        return None
    try:
        manifest = load_map_manifest(path)
    except (ValueError, OSError) as error:
        first_line = str(error).splitlines()[0] if str(error) else type(error).__name__
        logger.warning(
            "published manifest %s unreadable, reuse disabled (%s)", path, first_line
        )
        return None
    if manifest.schema_version != MAP_MANIFEST_SCHEMA_VERSION:
        logger.warning(
            "published manifest %s has schema version %s, not %s; reuse disabled",
            path,
            manifest.schema_version,
            MAP_MANIFEST_SCHEMA_VERSION,
        )
        return None
    return manifest


@dataclass(frozen=True)
class _Reusable:
    """A published run identical to the one requested."""

    parquet: Path
    record: MapRunRecord
    extended_json: Path | None


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
    lookup_sha256: str | None,
    keep_extended_json: bool,
) -> _Reusable | None:
    """Return the published parquet of an identical run, if any.

    Identical means the same query fingerprint, ``build_hash``, engine
    parameters, ctm version, tidy schema version and lookup (marker-content
    sha256 of the restricted lookup), with an unchanged parquet; when the
    extended JSON is to be kept, the published run must have kept it.
    """
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
    if record.tidy_schema_version != TIDY_SCHEMA_VERSION:
        reasons.append("tidy schema version")
    if record.lookup_sha256 != lookup_sha256:
        reasons.append("lookup")
    extended: Path | None = None
    if keep_extended_json:
        extended = (
            published_dir / record.extended_json
            if record.extended_json is not None
            else None
        )
        if extended is None or not extended.is_file():
            reasons.append("kept extended JSON")
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
    return _Reusable(parquet=path, record=record, extended_json=extended)


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
    probability: Literal["bp", "aggregate_probability"] = "bp",
) -> _LevelColumns:
    """Fill one level from a tidy level frame (rows = mapped cells).

    ``group_of`` aggregates the level to a coarser class (lineage, broad, NT);
    without it the level is the node itself, scored on ``probability`` (the
    bootstrap probability, or ctm's ``aggregate_probability``: the product
    down the hierarchy, E2's SEA-AD subclass definition).
    """
    values = _LevelColumns.empty(n_objects)
    if group_of is None:
        names = frame["name"].astype(object).to_numpy()
        raw = frame[probability].to_numpy(np.float64)
        runner = frame[runner_up_column(1, "name")].astype(object).to_numpy()
        runner_probability = frame[runner_up_column(1, "probability")].to_numpy(
            np.float64
        )
        # Runner-ups carry bootstrap probabilities only, so the margin is
        # always on the bootstrap probability.
        margin = frame["bp"].to_numpy(np.float64) - np.nan_to_num(
            runner_probability, nan=0.0
        )
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
        meets_threshold(raw, threshold_values),
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
    SEA-AD's raw thresholds (0.55 below ``second_vote_below_counts``, 0.45
    from it) to the subclass ``aggregate_probability`` (E2's definition)
    under a confident WHB broad. Mouse: broad and NT aggregate the WMB class level,
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
                    counts < thresholds.second_vote_below_counts,
                    thresholds.seaad_subclass_below60,
                    thresholds.seaad_subclass_from60,
                )
                broad_ok = broad.status[sea_positions] == CellStatus.CONFIDENT.value
                # E2 derived 0.55 / 0.45 on the subclass aggregate_probability
                # (class bp x subclass bp; exp/E2/build_tables.py sea_subclass_p).
                levels["seaad_subclass"] = _level_values(
                    sea,
                    sea_positions,
                    n_objects,
                    group_of=None,
                    threshold=sea_threshold,
                    parent_confident=broad_ok,
                    probability="aggregate_probability",
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
    refused_platforms: Iterable[str] = (),
    find_subset_bundle: SubsetBundleFinder | None = None,
) -> MapManifest:
    """Run the MAP step for the samples of one pair x segmentation.

    Args:
        samples: The samples (prepared or published clustered H5ADs).
        runs: The MMC runs (``map_bundles``).
        config: The annotation config, coupled to the clustering
            ``min_counts`` (``require_min_counts``).
        output_dir: ``annotation_map_out`` (``<plat>/`` parquets and
            ``map_manifest.json``).
        pair_id: Pair id.
        segmentation: Segmentation.
        n_processors: MMC worker processes (default: ``default_n_processors``).
        work_dir: Task-local scratch for queries and extended JSONs (default:
            ``<output_dir>/.work``, removed at the end).
        reuse_from: Directory of a published ``map_manifest.json`` whose
            identical runs are reused (``annotation_reuse_published``);
            ``None`` disables reuse, and an unreadable manifest disables it
            with a warning (``load_published_manifest``).
        write_provisional: Write the provisional ``ct_*`` parquet.
        refused_platforms: Platforms whose own panel was refused
            (``refused_platforms``); their samples are recorded without runs.
            Any other sample that no run applies to is an error.
        find_subset_bundle: Returns the store's bundle of a reference on a
            subset panel hash, or ``None`` (``store_subset_bundle_finder``);
            without it every needed subset bundle is only requested.

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
    if reuse_from is not None and config.reuse_published:
        published = load_published_manifest(reuse_from)
        published_dir = Path(reuse_from) if published is not None else None
    refused = {platform.upper() for platform in refused_platforms}
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
                panel_refused=loaded.sample.platform.upper() in refused,
                find_subset_bundle=find_subset_bundle,
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
    panel_refused: bool = False,
    find_subset_bundle: SubsetBundleFinder | None = None,
) -> MapSampleRecord:
    """Map one sample onto every run that applies to its platform.

    Runs on one bundle with the same query (two uses of one gene set) are
    mapped once; the later run ids get a copy of the tidy table
    (``same_mapping_as``).

    Raises:
        MapError: If no run applies to the sample and its panel was not
            refused.
    """
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
        panel_status="refused" if panel_refused else "ok",
    )
    applicable = [run for run in runs if run.applies_to(sample.platform)]
    if not applicable:
        if panel_refused:
            logger.warning(
                "%s: the %s panel was refused; the sample is not mapped",
                sample.sample_id,
                sample.platform,
            )
            return record
        raise MapError(
            f"{sample.sample_id}: no MMC run applies to platform "
            f"{sample.platform} (runs: "
            + ", ".join(f"{run.run_id} on {run.panel_name}" for run in runs)
            + ")"
        )
    tidies: list[tuple[MapBundle, pd.DataFrame]] = []
    mapped: dict[tuple[str, str], tuple[str, Path, MapRunRecord]] = {}
    for planned in applicable:
        run, subset_record = _subset_bundle_for(
            loaded,
            planned,
            config,
            output=output,
            find_subset_bundle=find_subset_bundle,
        )
        params = MmcEngineParams.from_reference_spec(run.spec, n_processors=processes)
        query = build_sample_query(loaded, run.panel)
        parquet = sample_dir / MMC_PARQUET_TEMPLATE.format(
            sample_id=sample.sample_id, run_id=run.run_id
        )
        key = (run.bundle.build_hash, query.fingerprint)
        run_scratch = scratch / f"{sample.sample_id}_{run.run_id}"
        try:
            earlier = mapped.get(key)
            if earlier is not None:
                run_record = _same_mapping(run, earlier, parquet=parquet, output=output)
            else:
                restricted = restrict_lookup(
                    run.bundle, query.gene_ids, run_scratch / "lookup.restricted.json"
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
                        lookup_sha256=restricted.lookup_sha256,
                        keep_extended_json=config.keep_extended_json,
                    )
                    if config.reuse_published
                    else None
                )
                if reusable is not None:
                    run_record = _reused_run(
                        run, reusable, parquet=parquet, output=output, config=config
                    )
                else:
                    run_record = _run_one(
                        loaded,
                        run,
                        query,
                        params,
                        config,
                        restricted=restricted,
                        parquet=parquet,
                        output=output,
                        scratch=run_scratch,
                    )
        finally:
            shutil.rmtree(run_scratch, ignore_errors=True)
        if subset_record is not None or run_record.subset_bundle is not None:
            run_record = run_record.model_copy(update={"subset_bundle": subset_record})
        mapped.setdefault(key, (run.run_id, parquet, run_record))
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


def _subset_bundle_for(
    loaded: LoadedSample,
    run: MapBundle,
    config: AnnotationConfig,
    *,
    output: Path,
    find_subset_bundle: SubsetBundleFinder | None,
) -> tuple[MapBundle, SubsetBundleRecord | None]:
    """Apply the subset-bundle trigger to one run of one sample (§3.3 step 1).

    Returns:
        The run to map (on the store's subset bundle and the subset panel
        when one exists, else unchanged) and the decision record (``None``
        when no subset bundle is needed).
    """
    # Declared genes dropped by min_cells are present with zero counts
    # (build_sample_query), as on the prepared H5AD.
    present = {gene_id for gene_id in loaded.feature_ids if gene_id} | set(
        loaded.undetected_declared_ids
    )
    if set(run.panel.ensembl_ids) <= present:
        return run, None
    trigger = subset_trigger_for_config(
        run.panel.ensembl_ids, present, read_lookup(run.bundle.lookup), config.panel
    )
    if not trigger.needs_subset or trigger.subset_panel_hash is None:
        return run, None
    subset = subset_panel(run.panel, present, trigger)
    path = (
        output
        / SUBSET_PANELS_DIR
        / SUBSET_PANEL_TEMPLATE.format(
            sample_id=loaded.sample.sample_id, run_id=run.run_id
        )
    )
    subset.write(path)
    found: MmcBundle | None = None
    candidates: list[str] = []
    if find_subset_bundle is not None:
        try:
            found = find_subset_bundle(run.reference_id, subset.panel_hash)
        except AmbiguousSubsetBundleError as error:
            candidates = error.candidates
            logger.warning(
                "%s %s: %s; mapping with the parent bundle",
                loaded.sample.sample_id,
                run.run_id,
                error,
            )
    if found is not None and found.panel_hash != subset.panel_hash:
        raise MapError(
            f"subset bundle {found.path} is for panel {str(found.panel_hash)[:16]}, "
            f"not {subset.panel_hash[:16]}"
        )
    status: Literal["used", "requested", "ambiguous"] = (
        "used" if found is not None else "ambiguous" if candidates else "requested"
    )
    record = SubsetBundleRecord(
        trigger=trigger,
        status=status,
        subset_panel_file=_relative(path, output),
        subset_panel_hash=subset.panel_hash,
        parent_panel_hash=run.panel.panel_hash,
        parent_build_hash=run.bundle.build_hash,
        subset_build_hash=None if found is None else found.build_hash,
        candidates=candidates,
    )
    reasons = ", ".join(trigger.reasons) or trigger.action
    if found is None:
        logger.warning(
            "%s %s: %d of %d panel genes missing (%s): subset bundle %s needed "
            "(%s); mapping with the parent bundle. Build it with merxen "
            "annotation-reference-prep --reference-id %s --panel-genes %s",
            loaded.sample.sample_id,
            run.run_id,
            trigger.n_missing,
            trigger.n_panel_genes,
            reasons,
            subset.panel_hash[:16],
            trigger.action,
            run.reference_id,
            path,
        )
        return run, record
    logger.info(
        "%s %s: %d of %d panel genes missing (%s); mapping with subset bundle %s",
        loaded.sample.sample_id,
        run.run_id,
        trigger.n_missing,
        trigger.n_panel_genes,
        reasons,
        found.path,
    )
    return replace(run, bundle=found, panel=subset, panel_name=subset.name), record


def store_subset_bundle_finder(
    store: ReferenceStore, config: AnnotationConfig | None = None
) -> SubsetBundleFinder:
    """Return a finder of subset bundles in a store (standalone MAP only).

    A pipeline MAP task never uses it (``--require-bundle-refs``): its
    bundles come from PREP as bundle refs that Nextflow stages and tracks.

    Args:
        store: The reference store.
        config: The run's annotation config (large-panel prefilter).

    Returns:
        ``(reference_id, panel_hash) -> MmcBundle | None``: the store's one
        current-builder bundle on that panel, opened through its store entry
        like a bundle ref (``MmcBundle.from_bundle_ref``: the directory's
        build hash must match its ``bundle.json``), or ``None`` without one;
        several raise ``AmbiguousSubsetBundleError``, which MAP records as
        ``ambiguous`` instead of failing.
    """

    def find(reference_id: str, panel_hash: str) -> MmcBundle | None:
        candidates = _prefer_current_resolvability(
            _current_bundles(store, reference_id, panel_hash, config)
        )
        if not candidates:
            return None
        if len(candidates) > 1:
            raise AmbiguousSubsetBundleError(
                f"{len(candidates)} subset bundles of {reference_id} on panel "
                f"{panel_hash[:16]}: "
                + ", ".join(str(entry.path) for entry in candidates),
                [str(entry.path) for entry in candidates],
            )
        return _bundle_from_entry(candidates[0])

    return find


def _bundle_from_entry(entry: StoreEntry) -> MmcBundle:
    """Open a store entry as a bundle ref would (build hash checked)."""
    manifest = json.loads((entry.path / BUNDLE_MANIFEST_NAME).read_text("utf-8"))
    ref = BundleRef(
        reference_id=str(manifest.get("reference_id", entry.reference_id)),
        species=str(manifest.get("species", "")),
        role=manifest.get("role", "primary"),
        panel_hash=entry.panel_hash,
        # The directory name is the content address: bundle.json must agree.
        build_hash=entry.path.name,
        path=str(entry.path),
        store_root=str(entry.store_root),
    )
    return MmcBundle.from_bundle_ref(ref)


def _reused_run(
    run: MapBundle,
    reusable: _Reusable,
    *,
    parquet: Path,
    output: Path,
    config: AnnotationConfig,
) -> MapRunRecord:
    """Copy an identical published run (parquet and, if kept, extended JSON)."""
    if reusable.parquet.resolve() != parquet.resolve():
        shutil.copy2(reusable.parquet, parquet)
    extended: str | None = None
    if config.keep_extended_json and reusable.extended_json is not None:
        target = parquet.with_name(
            parquet.name.removesuffix(".parquet") + EXTENDED_JSON_GZ_SUFFIX
        )
        if reusable.extended_json.resolve() != target.resolve():
            shutil.copy2(reusable.extended_json, target)
        extended = _relative(target, output)
    logger.info("%s: reused %s", run.run_id, reusable.parquet)
    return reusable.record.model_copy(
        update={
            "parquet": _relative(parquet, output),
            "reused": True,
            "reused_from": str(reusable.parquet.resolve()),
            "bundle_path": str(run.bundle.path),
            "extended_json": extended,
            "same_mapping_as": None,
        }
    )


def _same_mapping(
    run: MapBundle,
    earlier: tuple[str, Path, MapRunRecord],
    *,
    parquet: Path,
    output: Path,
) -> MapRunRecord:
    """Record a run whose bundle and query equal an earlier run of the sample."""
    earlier_id, earlier_parquet, earlier_record = earlier
    tidy, metadata = read_tidy_parquet(earlier_parquet)
    metadata.pop("tidy_schema_version", None)
    metadata.update({"run_id": run.run_id, "purposes": list(run.purposes)})
    write_tidy_parquet(tidy, parquet, metadata)
    logger.info(
        "%s: same bundle and query as %s; its mapping is recorded under both",
        run.run_id,
        earlier_id,
    )
    return earlier_record.model_copy(
        update={
            "run_id": run.run_id,
            "purposes": list(run.purposes),
            "panel_name": run.panel_name,
            "parquet": _relative(parquet, output),
            "parquet_sha256": file_sha256(parquet),
            "same_mapping_as": earlier_id,
        }
    )


def _run_one(
    loaded: LoadedSample,
    run: MapBundle,
    query: SampleQuery,
    params: MmcEngineParams,
    config: AnnotationConfig,
    *,
    restricted: RestrictedLookup,
    parquet: Path,
    output: Path,
    scratch: Path,
) -> MapRunRecord:
    scratch.mkdir(parents=True, exist_ok=True)
    query_path = write_query_h5ad(
        query.counts,
        query.cell_ids,
        query.gene_ids,
        scratch / "query.h5ad",
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
        undetected_declared_genes=query.undetected_gene_ids,
        engine_params=params.reuse_key(),
        ctm_version=result.ctm_version,
        ctm_commit=result.ctm_commit,
        effective_config=result.effective_config,
        wall_s=result.wall_s,
        peak_rss_gb=result.peak_rss_gb,
        parquet=_relative(parquet, output),
        parquet_sha256=file_sha256(parquet),
        tidy_schema_version=TIDY_SCHEMA_VERSION,
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


def results_root_of(path: Path | str) -> Path | None:
    """Return the results root that holds a file, if it sits in a results tree.

    A published clustered H5AD gives its ``<root>`` directly
    (``published_layout``). Any other file (a prepared H5AD, a view manifest
    target) is placed by its nearest ``clustering_squidpy`` ancestor:
    ``<root>/<pair>/<seg>/clustering_squidpy/...``.

    Args:
        path: An input file.

    Returns:
        ``<root>``, or ``None`` outside a results tree.
    """
    layout = published_layout(path)
    if layout.results_root is not None:
        return layout.results_root
    resolved = Path(path).resolve()
    for ancestor in resolved.parents:
        if ancestor.name == "clustering_squidpy" and len(ancestor.parents) > 2:
            return ancestor.parents[2]
    return None


def check_output_outside_inputs(
    output_dir: Path | str,
    inputs: Iterable[Path],
    *,
    protected_roots: Iterable[Path | str] = (),
    what: str = "output",
) -> None:
    """Refuse an output directory inside a results tree or an input's folder.

    The standalone ``merxen annotate`` never writes into published results
    (R3): not below the results root of an input (``results_root_of``), not
    below a ``protected_roots`` directory (``--results-root``), and not in
    any input's directory.

    Args:
        output_dir: The requested directory (``--out`` or ``--work-dir``).
        inputs: Input files.
        protected_roots: Further directories to stay out of.
        what: The directory's role, for the message.

    Raises:
        MapError: If the directory lies inside a protected directory.
    """
    target = Path(output_dir).resolve()
    guarded: list[tuple[Path, str]] = [
        (Path(root).resolve(), "a protected results root") for root in protected_roots
    ]
    for path in inputs:
        guarded.append((Path(path).resolve().parent, f"published inputs ({path})"))
        root = results_root_of(path)
        if root is not None:
            guarded.append((root, f"the results tree of {path}"))
    for directory, reason in guarded:
        if target == directory or directory in target.parents:
            raise MapError(
                f"{what} {target} lies inside {directory}, which holds "
                f"{reason}; write annotation outputs elsewhere (never into the "
                "results tree)"
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


def refused_platforms(required: RequiredBundles) -> list[str]:
    """Return the platforms whose own ``per_platform`` panel was refused.

    ``compute_panel`` records each refused panel as ``"<name>: <reason>"``;
    a ``per_platform`` pair names its platform panels after the platform.

    Args:
        required: ``required_bundles.json`` of the pair x segmentation.

    Returns:
        Upper-case platform names (empty for other panel modes).
    """
    if required.panel_mode != "per_platform":
        return []
    names = {reason.split(":", 1)[0].strip().upper() for reason in required.reasons}
    return sorted(names & {"MERSCOPE", "XENIUM"})


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
        output_dir: ``annotation_map_out``.
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


# --------------------------------------------------------------------------
# annotate_resolve (CLUSTERING_SQUIDPY_ANNOTATE_RESOLVE, plan §3.4; M4)

RESOLVE_SUMMARY_SUFFIX: Final = "_resolve_summary.json"
RESOLVE_SUMMARY_SCHEMA_VERSION: Final = 1
LABELS_METADATA_KEY: Final = b"merxen_annotation"
RESOLVE_STEP: Final = "annotate_resolve"
MISSING_INSTANCE_ID: Final = -1


class ResolveError(RuntimeError):
    """RESOLVE cannot run on its inputs (an infrastructure error)."""


def resolve_summary_filename(pair_id: str | None) -> str:
    """Return ``<pair>_resolve_summary.json`` (without a pair: no prefix)."""
    return f"{pair_id}{RESOLVE_SUMMARY_SUFFIX}" if pair_id else "resolve_summary.json"


@dataclass(frozen=True)
class ResolveRun:
    """One MAP run as RESOLVE reads it: its record, bundle and tidy table.

    Attributes:
        record: The run in ``map_manifest.json``.
        bundle: The bundle RESOLVE reads (the run's, or an override).
        tidy: The run's tidy MMC table.
        overridden: Whether the bundle is not the one the run mapped with.
        same_lookup: Whether the bundle's marker lookup equals the one the
            run mapped with (an override with another lookup is refused).
    """

    record: MapRunRecord
    bundle: MmcBundle
    tidy: pd.DataFrame
    overridden: bool = False
    same_lookup: bool = True

    @property
    def run_id(self) -> str:
        """Return the run id."""
        return self.record.run_id

    @property
    def prefix(self) -> str:
        """Return the raw engine column prefix (``mmc_whb``, ``mmc_whb_setc``)."""
        base = ENGINE_PREFIXES.get(self.record.reference_id, self.record.reference_id)
        purpose = self.record.purposes[0] if self.record.purposes else "annotation"
        return f"mmc_{base}{RUN_SUFFIXES.get(purpose, '')}"

    @property
    def leaf_level(self) -> str:
        """Return the bundle's first mapping level (WHB supercluster)."""
        return self.bundle.levels[0]

    @property
    def leaf_level_name(self) -> str:
        """Return the tidy table's token of the leaf level (``supercluster``)."""
        rows = self.tidy[self.tidy["level"].astype(str) == self.leaf_level]
        if len(rows):
            return str(rows["level_name"].astype(str).iloc[0])
        return self.leaf_level


# (reference_id, panel_hash) -> the bundle directory RESOLVE should read, or
# None to keep the run's (``current_store_bundles``).
BundleFinder = Callable[[str, str | None], Path | None]


def current_store_bundles(store: ReferenceStore) -> BundleFinder:
    """Return a finder of the store's current bundle per (reference, panel).

    RESOLVE on published MAP outputs can read the store's current bundles
    (current builder, current resolvability tables; ``locate_bundle``) instead
    of the ones the runs mapped with, when their marker lookups are the same.

    Args:
        store: The reference store.

    Returns:
        The finder.
    """

    def find(reference_id: str, panel_hash: str | None) -> Path | None:
        return locate_bundle(store, reference_id, panel_hash).path

    return find


def _resolve_bundle(
    record: MapRunRecord,
    overrides: Mapping[str, Path],
    finder: BundleFinder | None,
) -> tuple[MmcBundle, bool]:
    path = overrides.get(record.run_id) or overrides.get(record.reference_id)
    if path is None and finder is not None:
        path = finder(record.reference_id, record.panel_hash)
    if path is None:
        return MmcBundle.from_dir(record.bundle_path), False
    bundle = MmcBundle.from_dir(path)
    if bundle.reference_id != record.reference_id:
        raise ResolveError(
            f"bundle {path} is a {bundle.reference_id} bundle, not "
            f"{record.reference_id} (run {record.run_id})"
        )
    if record.panel_hash is not None and bundle.panel_hash != record.panel_hash:
        raise ResolveError(
            f"bundle {path} was built on panel {str(bundle.panel_hash)[:16]}, but run "
            f"{record.run_id} mapped panel {record.panel_hash[:16]}"
        )
    return bundle, Path(path).resolve() != Path(record.bundle_path).resolve()


def load_resolve_runs(
    map_dir: Path | str,
    sample: MapSampleRecord,
    *,
    bundle_overrides: Mapping[str, Path] | None = None,
    bundle_finder: BundleFinder | None = None,
    check_sha256: bool = True,
) -> dict[str, ResolveRun]:
    """Read one sample's MAP runs: tidy parquets and the bundles to resolve with.

    Args:
        map_dir: The directory holding ``map_manifest.json``.
        sample: The sample's manifest record.
        bundle_overrides: Run id or reference id to a bundle directory.
        bundle_finder: Picks a bundle per (reference, panel) when no override
            names the run (``current_store_bundles``).
        check_sha256: Check each parquet against its recorded sha256.

    Returns:
        Run id to run.

    Raises:
        ResolveError: If a parquet changed, or an override's reference, panel
            or marker lookup differs from what the run mapped with.
    """
    root = Path(map_dir)
    overrides = dict(bundle_overrides or {})
    runs: dict[str, ResolveRun] = {}
    for run_id, record in sample.runs.items():
        parquet = root / record.parquet
        if check_sha256 and file_sha256(parquet) != record.parquet_sha256:
            raise ResolveError(
                f"{parquet} differs from the sha256 map_manifest.json recorded"
            )
        tidy, _ = read_tidy_parquet(parquet)
        bundle, overridden = _resolve_bundle(record, overrides, bundle_finder)
        same_lookup = bundle.lookup_sha256 == (
            record.lookup_sha256 or record.bundle_lookup_sha256
        )
        if overridden and not same_lookup and not record.lookup_restricted:
            raise ResolveError(
                f"run {run_id} mapped with lookup "
                f"{str(record.lookup_sha256)[:16]}, but bundle {bundle.path} has "
                f"lookup {str(bundle.lookup_sha256)[:16]}: its tables do not "
                "describe that mapping"
            )
        if overridden:
            logger.info(
                "%s %s: resolving with bundle %s (mapped with %s)",
                sample.sample_id,
                run_id,
                bundle.build_hash[:16],
                record.build_hash[:16],
            )
        runs[run_id] = ResolveRun(
            record=record,
            bundle=bundle,
            tidy=tidy,
            overridden=overridden,
            same_lookup=same_lookup,
        )
    return runs


def _run_for(
    runs: Mapping[str, ResolveRun], role: str, purpose: str = "annotation"
) -> ResolveRun | None:
    return next(
        (
            run
            for run in runs.values()
            if run.record.role == role and purpose in run.record.purposes
        ),
        None,
    )


def _level_or_empty(
    tidy: pd.DataFrame, level: str, index: pd.Index
) -> pd.DataFrame | None:
    names = set(tidy["level"].astype(str)) | set(tidy["level_name"].astype(str))
    if level not in names:
        return None
    return level_frame(tidy, level).reindex(index)


def _aggregated_scores(
    frame: pd.DataFrame, group_of: Mapping[str, str | None]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    aggregated = aggregate_parent_probability(frame, group_of)
    margin = aggregated.probability - np.nan_to_num(
        aggregated.runner_up_probability, nan=0.0
    )
    return (
        aggregated.classes,
        aggregated.probability,
        aggregated.runner_up_class,
        margin,
    )


def _masked_objects(mask: np.ndarray, values: Any) -> np.ndarray:
    """Return ``values`` as objects with ``None`` where ``mask`` is false."""
    output = np.asarray(values, dtype=object).copy()
    output[~np.asarray(mask, dtype=bool)] = None
    return output


@dataclass(frozen=True)
class HumanCallInputs:
    """The per-object calls of one human sample, with the leaf frame.

    Attributes:
        calls: ``consensus.HumanCalls`` (every object).
        leaf: The primary leaf level of the table cells (tidy rows,
            indexed by table-cell id), or ``None`` without a primary run.
        table_rows: Object position of each table cell.
    """

    calls: HumanCalls
    leaf: pd.DataFrame | None
    table_rows: np.ndarray


def human_calls_from_runs(
    loaded: LoadedSample,
    primary: ResolveRun | None,
    secondary: ResolveRun | None,
    *,
    allow_fine_levels: bool = False,
) -> HumanCallInputs:
    """Build ``resolve_human``'s inputs from the MAP tidy tables (§5.2).

    WHB: the supercluster call with its bootstrap probability,
    ``avg_correlation``, best runner-up and margin; lineage, broad and NT
    aggregate the supercluster bootstrap probabilities over the bundle's
    vocab snapshot (E1 / E2, as MAP's provisional labels); the WHB cluster
    only with ``allow_fine_levels``. SEA-AD: E2's 7-class label and broad
    probability (``shadow.seaad_broad_calls`` with the class level) and the
    subclass ``aggregate_probability``. Objects the runs did not map
    (outside the table) have no call.

    Args:
        loaded: The sample's counts (every object).
        primary: The WHB annotation run.
        secondary: The SEA-AD annotation run.
        allow_fine_levels: Also read the WHB cluster level.

    Returns:
        The calls.
    """
    from merxen.annotation.consensus import (
        HumanCalls,
        LevelCall,
        LevelScores,
        SeaCalls,
        WhbCalls,
    )
    from merxen.annotation.shadow import seaad_broad_calls

    obs = pd.Index(loaded.obs_names.astype(str))
    table_rows = np.flatnonzero(loaded.in_table)
    whb: WhbCalls | None = None
    leaf_table: pd.DataFrame | None = None
    if primary is not None:
        vocab = primary.bundle.vocab()
        leaf_level = primary.leaf_level
        leaf = level_frame(primary.tidy, leaf_level)
        leaf.index = leaf.index.astype(str)
        unknown = leaf.index.difference(obs)
        if len(unknown):
            raise ResolveError(
                f"{loaded.sample.sample_id}: run {primary.run_id} names "
                f"{len(unknown)} cells absent from the sample, e.g. {list(unknown[:3])}"
            )
        frame = leaf.reindex(obs)
        leaf_table = leaf.reindex(obs[table_rows])
        mapped = frame["assignment"].notna().to_numpy()
        lineage_of = _vocab_lookup(vocab, leaf_level, "lineage")
        broad_of = _vocab_lookup(vocab, leaf_level, "broad_class")
        nt_of = _vocab_lookup(vocab, leaf_level, "nt")
        nt_groups = {
            node: (nt if broad_of.get(node) == NEURONS else None)
            for node, nt in nt_of.items()
        }
        runner_probability = pd.to_numeric(
            frame[runner_up_column(1, "probability")], errors="coerce"
        ).to_numpy(np.float64)
        names = frame["name"].astype(object).to_numpy()
        names = _masked_objects(mapped, names)
        supercluster = LevelCall.of(
            names,
            frame["bp"].to_numpy(np.float64),
            corr=frame["avg_correlation"].to_numpy(np.float64),
            runner_up=frame[runner_up_column(1, "name")].astype(object).to_numpy(),
            margin=frame["bp"].to_numpy(np.float64)
            - np.nan_to_num(runner_probability, nan=0.0),
        )
        corr = frame["avg_correlation"].to_numpy(np.float64)
        scores = {}
        for key, groups in (
            ("lineage", lineage_of),
            ("broad", broad_of),
            ("nt", nt_groups),
        ):
            _classes, raw, runner, margin = _aggregated_scores(frame, groups)
            raw = np.where(mapped, raw, np.nan)
            scores[key] = LevelScores.of(
                raw, corr=corr, runner_up=runner, margin=margin
            )
        cluster: LevelCall | None = None
        if allow_fine_levels and len(primary.bundle.levels) > 1:
            fine = _level_or_empty(primary.tidy, primary.bundle.levels[1], obs)
            if fine is not None:
                fine_runner = pd.to_numeric(
                    fine[runner_up_column(1, "probability")], errors="coerce"
                ).to_numpy(np.float64)
                cluster = LevelCall.of(
                    fine["name"].astype(object).to_numpy(),
                    fine["bp"].to_numpy(np.float64),
                    corr=fine["avg_correlation"].to_numpy(np.float64),
                    runner_up=fine[runner_up_column(1, "name")].astype(object),
                    margin=fine["bp"].to_numpy(np.float64)
                    - np.nan_to_num(fine_runner, nan=0.0),
                )
        whb = WhbCalls(
            supercluster=supercluster,
            lineage=scores["lineage"],
            broad=scores["broad"],
            nt=scores["nt"],
            cluster=cluster,
        )
    sea: SeaCalls | None = None
    if secondary is not None:
        tidy = secondary.tidy
        levels = secondary.bundle.levels
        class_level = _level_or_empty(tidy, levels[0], obs) if levels else None
        subclass = _level_or_empty(tidy, "subclass", obs)
        if subclass is None and len(levels) > 1:
            subclass = _level_or_empty(tidy, levels[1], obs)
        supertype = _level_or_empty(tidy, "supertype", obs)
        if supertype is None and len(levels) > 2:
            supertype = _level_or_empty(tidy, levels[2], obs)
        if subclass is None:
            raise ResolveError(
                f"{loaded.sample.sample_id}: run {secondary.run_id} has no subclass "
                "level"
            )
        broad = seaad_broad_calls(subclass, supertype, class_level=class_level)
        mapped_sea = subclass["assignment"].notna().to_numpy()
        broad_names = _masked_objects(mapped_sea, broad["broad"].astype(object))
        sea = SeaCalls(
            broad=LevelCall.of(
                broad_names,
                np.where(mapped_sea, broad["broad_raw"].to_numpy(np.float64), np.nan),
            ),
            subclass=LevelCall.of(
                _masked_objects(mapped_sea, subclass["name"].astype(object)),
                subclass["aggregate_probability"].to_numpy(np.float64),
                corr=subclass["avg_correlation"].to_numpy(np.float64),
                runner_up=subclass[runner_up_column(1, "name")].astype(object),
                margin=subclass["bp"].to_numpy(np.float64)
                - np.nan_to_num(
                    pd.to_numeric(
                        subclass[runner_up_column(1, "probability")], errors="coerce"
                    ).to_numpy(np.float64),
                    nan=0.0,
                ),
            ),
        )
    calls = HumanCalls(
        total_counts=np.asarray(loaded.total_counts, dtype=np.int64),
        in_table=np.asarray(loaded.in_table, dtype=bool),
        whb=whb,
        sea=sea,
    )
    return HumanCallInputs(calls=calls, leaf=leaf_table, table_rows=table_rows)


def _panel_files(panel_dir: Path | None) -> dict[str, Path]:
    """Map panel hash to its ``panel_genes*.json`` in an ANNOTATE_PANEL output."""
    if panel_dir is None or not panel_dir.is_dir():
        return {}
    files: dict[str, Path] = {}
    for path in sorted(panel_dir.glob("panel_genes*.json")):
        try:
            panel = load_annotation_panel(path)
        except (ValueError, OSError) as error:
            logger.warning("skipping unreadable panel file %s: %s", path, error)
            continue
        files.setdefault(panel.panel_hash, path)
    return files


def current_family(panel: AnnotationPanel, config: AnnotationConfig) -> AnnotationPanel:
    """Return the panel with its family re-derived from the validated tables.

    RESOLVE decides trust with the current ``validated_panels.csv`` (§8.2),
    so a panel file written before its family was listed (e.g. an M3 shadow
    run) gets the family ``ANNOTATE_PANEL`` would give it now
    (``panel.panel_family``). Subset panels keep their parent's family.

    Args:
        panel: The annotation panel.
        config: The annotation config (validated tables, ``family_min_jaccard``).

    Returns:
        The panel, with an updated family when it differs (logged).
    """
    from merxen.annotation.diagnostics import load_validated_panels
    from merxen.annotation.panel import panel_family

    if panel.panel_family is not None and panel.panel_family.basis == "subset":
        return panel
    validated = load_validated_panels(config.panel.validated_panels_path)
    family = panel_family(
        panel.ensembl_ids,
        species=panel.species,
        platforms=panel.platforms,
        known_families=validated.known_families(panel.species),
        min_jaccard=config.panel.family_min_jaccard,
    )
    if family == panel.panel_family:
        return panel
    logger.warning(
        "panel %s (%s): family %s re-derived as %s (%s) from the current "
        "validated panels",
        panel.name,
        panel.panel_hash[:16],
        None if panel.panel_family is None else panel.panel_family.family_id,
        family.family_id,
        family.basis,
    )
    return panel.model_copy(update={"panel_family": family})


def trust_for_run(
    run: ResolveRun,
    panel: AnnotationPanel | None,
    config: AnnotationConfig,
    *,
    panel_report: Mapping[str, Any] | None = None,
) -> TrustDecision:
    """Return the trust decision of one run's (reference, panel) (§8.2).

    The same decision PREP and ``annotation-panel-simulate`` take:
    ``diagnostics.panel_diagnostics`` of the annotation panel with the
    bundle's ``bundle.json``, then ``trust_for_panel``.

    Args:
        run: The run (its bundle's manifest).
        panel: The run's annotation panel (``panel_genes*.json``).
        config: The annotation config (trust rules, validated panels).
        panel_report: ``panel_report.json`` content (gene-ID diagnostics).

    Returns:
        The decision.

    Raises:
        ResolveError: If the panel is unknown.
    """
    from merxen.annotation.diagnostics import (
        TrustRules,
        load_validated_panels,
        panel_diagnostics,
        trust_for_panel,
    )

    if panel is None:
        raise ResolveError(
            f"run {run.run_id}: no panel_genes*.json with panel hash "
            f"{str(run.record.panel_hash)[:16]} (pass --panel-dir)"
        )
    diagnostics = panel_diagnostics(
        panel, panel_report=panel_report, bundles=[run.bundle.manifest]
    )
    return trust_for_panel(
        diagnostics,
        reference_id=run.record.reference_id,
        role=run.record.role,  # type: ignore[arg-type]
        validated=load_validated_panels(config.panel.validated_panels_path),
        rules=TrustRules.from_config(config),
    )


def refused_trust(
    reference_id: str, species: Species, panel_hash: str, reasons: Sequence[str]
) -> TrustDecision:
    """Return a refused primary trust decision (MAP skipped a refused panel).

    Args:
        reference_id: The primary reference.
        species: Species.
        panel_hash: The sample's declared panel hash.
        reasons: The panel's refusal reasons.

    Returns:
        The decision (state ``refused``).
    """
    from merxen.annotation.diagnostics import TrustDecision, TrustReason

    return TrustDecision(
        reference_id=reference_id,
        role="primary",
        species=species,
        panel_hash=panel_hash,
        state="refused",
        reasons=tuple(
            TrustReason(code="panel_refused", detail=str(reason))
            for reason in (reasons or ["the panel was refused before mapping"])
        ),
    )


def refused_sample_record(loaded: LoadedSample, fingerprint: str) -> MapSampleRecord:
    """Return the manifest record of a sample MAP skipped (refused panel).

    Args:
        loaded: The sample's counts.
        fingerprint: ``loaded.sample_fingerprint()``.

    Returns:
        A record without runs and with ``panel_status="refused"``.
    """
    return MapSampleRecord(
        sample_id=loaded.sample.sample_id,
        platform=loaded.sample.platform,
        source=loaded.sample.source,
        h5ad_path=str(loaded.sample.h5ad_path),
        sample_fingerprint=fingerprint,
        n_objects=loaded.n_objects,
        n_table_cells=loaded.n_table_cells,
        min_counts=loaded.min_counts,
        declared_panel_hash=loaded.declared.panel_hash,
        n_features=len(loaded.feature_names),
        n_resolved_features=sum(1 for gene_id in loaded.feature_ids if gene_id),
        panel_status="refused",
    )


def read_spatial(path: Path | str) -> tuple[np.ndarray | None, str | None]:
    """Read ``obsm['spatial']`` and the shape key of an H5AD (composition CIs).

    Args:
        path: A prepared or clustered H5AD.

    Returns:
        ``(xy, shape_key)``: ``(n, 2)`` coordinates in µm (``None`` when the
        file has none) and ``uns['merxen_clustering_squidpy']['shape_key']``
        (``*_aligned_nonrigid`` for MERSCOPE coordinates in the Xenium frame).
    """
    import h5py

    with h5py.File(path, "r") as handle:
        xy = None
        if "obsm" in handle and "spatial" in handle["obsm"]:
            xy = np.asarray(handle["obsm"]["spatial"][()], dtype=np.float64)[:, :2]
        key = "uns/merxen_clustering_squidpy/shape_key"
        shape_key = None
        if key in handle:
            value = _read_elem(handle[key])
            shape_key = value.decode() if isinstance(value, bytes) else str(value)
    return xy, shape_key


def in_fixed_frame(platform: str, shape_key: str | None) -> bool:
    """Whether a section's coordinates are in the pair's fixed (Xenium) frame."""
    return str(platform).upper() == "XENIUM" or "_aligned" in str(shape_key or "")


def default_alignment_dir(h5ad_path: Path | str, pair_id: str | None) -> Path | None:
    """Return ``<results>/<pair>/alignment/align_out`` of an input, if it exists.

    Args:
        h5ad_path: An input H5AD in a results tree.
        pair_id: The pair.

    Returns:
        The directory, or ``None``.
    """
    root = results_root_of(h5ad_path)
    if root is None or not pair_id:
        return None
    candidate = root / pair_id / "alignment" / "align_out"
    return candidate if candidate.is_dir() else None


def load_pair_mask(alignment_dir: Path | None) -> Any | None:
    """Load the pair's shared tissue mask from an ``align_out`` directory.

    Args:
        alignment_dir: ``align_out`` (``shared_tissue_mask.npy`` and
            ``registration_summary.json``), or ``None``.

    Returns:
        ``panel.SharedTissueMask``, or ``None`` when absent or unreadable.
    """
    from merxen.annotation.panel import load_shared_tissue_mask

    if alignment_dir is None:
        return None
    mask_path = alignment_dir / "shared_tissue_mask.npy"
    summary_path = alignment_dir / "registration_summary.json"
    if not mask_path.is_file() or not summary_path.is_file():
        return None
    try:
        return load_shared_tissue_mask(mask_path, summary_path)
    except (ValueError, OSError) as error:
        logger.warning("shared tissue mask in %s not used: %s", alignment_dir, error)
        return None


def human_branch_columns(
    resolution: HumanResolution,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``ct_branch`` and ``ct_leaf`` of a human resolution (§4.1).

    The branch is the map_first broad label (``vocab.broad_class_for_map_first``:
    the confident broad class, else the ``Oligodendrocyte lineage`` / ``Neurons``
    lineage fallback, else ``Mixed/Unknown``), with neurons split by a
    confident NT (``Neurons/Excitatory`` / ``Neurons/Inhibitory``, else
    ``Neurons/unresolved``) and the lineage fallback as
    ``Oligodendrocyte lineage/unresolved``. The leaf is the confident WHB
    supercluster, else ``unresolved``.

    Args:
        resolution: ``resolve_human`` output.

    Returns:
        ``(branch, leaf)`` per object.
    """
    from merxen.annotation.schema import CellStatus
    from merxen.annotation.vocab import (
        NT_EXCITATORY,
        NT_INHIBITORY,
        OLIGODENDROCYTE_LINEAGE,
        UNRESOLVED_LABEL,
        broad_classes_for_map_first,
    )

    labels = broad_classes_for_map_first(
        resolution.final_level,
        resolution.levels["broad"].name,
        resolution.levels["lineage"].name,
        species="human",
    ).astype(str)
    nt = resolution.levels["nt"]
    nt_confident = nt.status == CellStatus.CONFIDENT.value
    branch = np.asarray(labels.to_numpy(), dtype=object).copy()
    neurons = branch == NEURONS
    excitatory = neurons & nt_confident & (nt.name == NT_EXCITATORY)
    inhibitory = neurons & nt_confident & (nt.name == NT_INHIBITORY)
    branch[neurons] = "Neurons/unresolved"
    branch[excitatory] = "Neurons/Excitatory"
    branch[inhibitory] = "Neurons/Inhibitory"
    branch[branch == OLIGODENDROCYTE_LINEAGE] = (
        f"{OLIGODENDROCYTE_LINEAGE}/{UNRESOLVED_LABEL}"
    )
    supercluster = resolution.levels["supercluster"]
    leaf = np.where(supercluster.confident, supercluster.name, UNRESOLVED_LABEL)
    return branch, np.asarray(leaf, dtype=object)


@dataclass
class SampleResolution:
    """RESOLVE's result for one sample.

    Attributes:
        sample_id: Sample id.
        platform: Platform.
        labels: The label table (§4.1).
        provenance: ``AnnotationProvenance`` (§4.6).
        summary: The sample's resolve-summary record.
        section: Its composition rows (pair JSD).
        labels_path: Where the table was written.
        manifest_path: Where the provenance was written.
    """

    sample_id: str
    platform: str
    labels: pd.DataFrame
    provenance: AnnotationProvenance
    summary: dict[str, Any]
    section: SectionComposition | None = None
    labels_path: Path | None = None
    manifest_path: Path | None = None


def _round_share(value: float | None) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), 6)


def _floats_only(record: Mapping[str, Any]) -> dict[str, float]:
    return {
        str(key): float(value)
        for key, value in record.items()
        if isinstance(value, int | float)
        and not isinstance(value, bool)
        and np.isfinite(value)
    }


def _threshold_values(config: AnnotationConfig) -> dict[str, float]:
    return _floats_only(config.thresholds.model_dump(mode="json"))


def _emitted_bins(
    emission: EmissionPlan, levels: Sequence[str]
) -> dict[str, dict[str, list[int]]]:
    decisions = emission.decisions
    if decisions is None:
        return {}
    output: dict[str, dict[str, list[int]]] = {}
    for level in levels:
        table_level = emission.table_level(level)
        regime = emission.regime(table_level)
        rows = decisions[
            (decisions["level"].astype(str) == table_level)
            & (decisions["regime"].astype(str) == regime)
            & (decisions["status"].astype(str) == "emitted")
        ]
        per_class: dict[str, list[int]] = {}
        for cls, group in rows.groupby(rows["class"].astype(str)):
            per_class[safe_token(str(cls))] = sorted(
                int(depth) for depth in group["depth"]
            )
        output[safe_token(level)] = per_class
    return output


def _resolvability_provenance(
    tables: ResolvabilityTables | None,
    emission: EmissionPlan,
    resolution: HumanResolution,
    bundle: MmcBundle,
    *,
    reweighted: bool,
) -> ResolvabilityProvenance:
    from merxen.annotation.provenance import ResolvabilityProvenance
    from merxen.annotation.resolvability import RESOLVABILITY_FILE

    table = resolution.in_table
    resolvable: dict[str, float] = {}
    for level, item in resolution.emissions.items():
        share = item.summary(table)["emitted_share"]
        if share is not None:
            resolvable[safe_token(level)] = round(float(share), 6)
    if tables is None:
        return ResolvabilityProvenance(
            reweighted_to_composition=False, resolvable_share=resolvable
        )
    summary = tables.summary
    recipes = {item.get("name"): item for item in summary.get("recipes", [])}
    decision_recipe = str(summary.get("decision_recipe"))
    version = (recipes.get(decision_recipe) or {}).get("version")
    files = {
        str(item.get("path")): item.get("sha256")
        for item in bundle.manifest.get("files", []) or []
    }
    d_max: dict[str, int] = {}
    for cls, depth in (summary.get("d_max") or {}).get("broad", {}).items():
        if depth is not None:
            d_max[safe_token(str(cls))] = int(depth)
    broad = resolution.levels["broad"]
    extrapolated: dict[str, float] = {}
    keys = np.asarray(broad.class_key, dtype=object)
    for cls in sorted({str(key) for key in keys[table] if key is not None}):
        members = table & (keys == cls)
        if members.any():
            extrapolated[safe_token(cls)] = round(
                float(resolution.resolvability_extrapolated[members].mean()), 6
            )
    return ResolvabilityProvenance(
        recipe=decision_recipe,
        recipe_version=None if version is None else int(version),
        sha256=files.get(RESOLVABILITY_FILE),
        emitted_depth_bins=_emitted_bins(emission, list(resolution.emissions)),
        d_max=d_max,
        extrapolated_share=extrapolated,
        reweighted_to_composition=reweighted,
        resolvability_inherited=bool(
            (bundle.manifest.get("builder_output") or {}).get(
                "resolvability_inherited", False
            )
        ),
        resolvable_share=resolvable,
    )


def _reference_provenance(
    run: ResolveRun, trust: TrustDecision | None
) -> ReferenceProvenance:
    from merxen.annotation.diagnostics import CoverageDiagnostics
    from merxen.annotation.provenance import MarkerProvenance, ReferenceProvenance

    manifest = run.bundle.manifest
    output = manifest.get("builder_output") or {}
    markers_record = output.get("markers") or {}
    try:
        coverage = CoverageDiagnostics.from_bundle_manifest(manifest)
        markers = coverage.marker_provenance(
            lookup_sha256=run.bundle.lookup_sha256,
            n_per_utility=markers_record.get("n_per_utility"),
        )
        n_used: int | None = coverage.n_query_genes_used
        absent: int | None = len(coverage.absent_from_reference)
    except ValueError:
        markers = MarkerProvenance(lookup_sha256=run.bundle.lookup_sha256)
        n_used, absent = run.record.n_query_genes, None
    return ReferenceProvenance(
        reference_id=run.record.reference_id,
        role=run.record.role,  # type: ignore[arg-type]
        taxonomy_id=run.bundle.taxonomy_id,
        levels=list(run.bundle.levels),
        collapsed_parents=list(run.record.collapsed_parents),
        drop_level=run.bundle.drop_level,
        bundle_path=str(run.bundle.path),
        build_hash=run.bundle.build_hash,
        n_query_genes_used=n_used,
        markers=markers,
        panel_trust=None if trust is None else trust.state,
        trust_reasons=[] if trust is None else trust.reason_codes,
        n_panel_genes_absent=absent,
    )


def _composition_provenance(
    shares: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    return {
        safe_token(kind): {
            key: float(value)
            for key, value in record.items()
            if key.startswith("share") and value is not None
        }
        for kind, record in shares.items()
    }


def _assigned_broad(names: np.ndarray) -> np.ndarray:
    """The WHB vocab broad class of each supercluster name (sinks: ``None``)."""
    from merxen.annotation.composition import whb_broad_of
    from merxen.annotation.vocab import HUMAN_BROAD_CLASSES

    broad_of = whb_broad_of()
    classes = set(HUMAN_BROAD_CLASSES)
    output = np.full(len(names), None, dtype=object)
    for index, name in enumerate(names):
        if name is None:
            continue
        broad = broad_of.get(str(name))
        if broad in classes:
            output[index] = broad
    return output


def resolve_human_sample(
    loaded: LoadedSample,
    record: MapSampleRecord,
    runs: Mapping[str, ResolveRun],
    config: AnnotationConfig,
    *,
    manifest: MapManifest,
    panels: Mapping[str, AnnotationPanel],
    panel_report: Mapping[str, Any] | None = None,
    n_segmented: int | None = None,
    trust_overrides: Mapping[str, TrustDecision] | None = None,
    xy: np.ndarray | None = None,
    aligned_frame: bool = False,
    panel_mode: str | None = None,
    validate: bool = True,
    seed: int = 0,
) -> SampleResolution:
    """Resolve one human sample (plan §3.4; §4.1, §4.3, §4.6, §5.2-§5.6, §8.2-§8.3).

    Args:
        loaded: The sample's counts (every object, control-free).
        record: Its ``map_manifest.json`` record.
        runs: Its MAP runs (``load_resolve_runs``).
        config: The annotation config (coupled to ``min_counts``).
        manifest: The pair's MAP manifest.
        panels: Panel hash to annotation panel.
        panel_report: ``panel_report.json`` content.
        n_segmented: Segmented objects (the gate warning's denominator);
            default: the objects of the input.
        trust_overrides: Trust decision per reference id instead of the
            diagnostics (tests, replays).
        xy: Coordinates of every object (composition CIs).
        aligned_frame: Whether ``xy`` is in the pair's fixed frame.
        panel_mode: The pair's resolved panel mode.
        validate: Check the table with ``schema.validate_label_table``.
        seed: Seed of the diffuse-flag simulation.

    Returns:
        The sample's resolution.

    Raises:
        ResolveError: If the inputs do not fit together.
    """
    from merxen.annotation import consensus as cs
    from merxen.annotation import flags as fl
    from merxen.annotation.composition import (
        dataset_type_composition,
        section_composition,
        section_shares,
        soft_broad_columns,
        soft_broad_from_level,
    )
    from merxen.annotation.provenance import (
        AnnotationProvenance,
        ConsensusProvenance,
        DatasetGateProvenance,
        EngineProvenance,
        ThresholdProvenance,
    )
    from merxen.annotation.resolvability import load_resolvability
    from merxen.annotation.schema import (
        coerce_label_table_dtypes,
        validate_label_table,
    )
    from merxen.annotation.thresholds import EmissionPlan, FloorPlan
    from merxen.annotation.vocab import HUMAN_BROAD_CLASSES, load_state_gene_ids

    species: Species = "human"
    sample_id = loaded.sample.sample_id
    platform = loaded.sample.platform.upper()
    min_counts = config.require_min_counts()
    overrides = dict(trust_overrides or {})
    primary = _run_for(runs, "primary")
    secondary = _run_for(runs, "secondary")
    n_objects = loaded.n_objects
    table = np.asarray(loaded.in_table, dtype=bool)

    # Trust decisions (§8.2).
    primary_id = config.primary_reference().reference_id
    trust: TrustDecision | None
    secondary_trust: TrustDecision | None = None
    if primary is None:
        trust = overrides.get(primary_id) or refused_trust(
            primary_id,
            species,
            record.declared_panel_hash,
            manifest.panel_reasons
            or [f"sample panel status {record.panel_status}: no primary run"],
        )
    else:
        panel = panels.get(str(primary.record.panel_hash))
        trust = overrides.get(primary.record.reference_id) or trust_for_run(
            primary, panel, config, panel_report=panel_report
        )
    if secondary is not None:
        panel = panels.get(str(secondary.record.panel_hash))
        secondary_trust = overrides.get(secondary.record.reference_id) or trust_for_run(
            secondary, panel, config, panel_report=panel_report
        )
    elif overrides.get("seaad_mr_panel") is not None:
        secondary_trust = overrides["seaad_mr_panel"]

    inputs = human_calls_from_runs(
        loaded,
        primary,
        secondary,
        allow_fine_levels=config.thresholds.allow_fine_levels,
    )
    counts = np.asarray(loaded.total_counts, dtype=np.float64)

    # Emission reweighted to the dataset's soft composition (§8.3).
    tables = None if primary is None else load_resolvability(primary.bundle.path)
    composition = None
    reweight = bool(
        tables is not None
        and config.resolvability.reweight_to_composition
        and inputs.leaf is not None
        and len(inputs.leaf)
    )
    if reweight:
        assert tables is not None and inputs.leaf is not None
        composition = dataset_type_composition(
            inputs.leaf,
            counts[inputs.table_rows],
            tables.depth_grid,
            min_bin_mass=float(config.resolvability.composition_min_bin_cells),
        )
    emission = EmissionPlan.from_tables(
        tables,
        species=species,
        thresholds=config.thresholds,
        trust=trust,
        composition=composition,
        seed_stability_max_change=config.resolvability.seed_stability_max_change,
    )
    floors = FloorPlan.build(
        species=species,
        platform=platform,
        hard_floor=min_counts,
        trust=trust,
        thresholds=config.thresholds,
        emission=emission,
    )
    settings = cs.HumanResolveSettings.from_config(
        config,
        platform=platform,
        emission=emission,
        floors=floors,
        trust=trust,
        secondary_trust=secondary_trust,
        n_segmented=n_segmented if n_segmented is not None else n_objects,
    )
    if loaded.sample.source == "clustered":
        settings = replace(settings, allow_table_below_min_counts=True)
    resolution = cs.resolve_human(inputs.calls, settings)

    # Flags (§4.3, §5.6).
    whb = inputs.calls.whb
    names = (
        whb.supercluster.name
        if whb is not None
        else np.full(n_objects, None, dtype=object)
    )
    assigned = _assigned_broad(names)
    query_counts = None
    query_rows = None
    gene_ids: tuple[str, ...] = ()
    negatives = None
    profiles_by_class = None
    if primary is not None:
        panel = panels.get(str(primary.record.panel_hash))
        if panel is not None:
            query = build_sample_query(loaded, panel)
            query_counts = query.counts
            query_rows = np.flatnonzero(table)
            gene_ids = tuple(query.gene_ids)
        else:
            logger.warning(
                "%s: no panel file for %s; contamination and diffuse flags are null",
                sample_id,
                str(primary.record.panel_hash)[:16],
            )
        bundle_dir = primary.bundle.path
        negatives_path = bundle_dir / "negative_genes.parquet"
        profiles_path = bundle_dir / "profiles.parquet"
        if gene_ids and negatives_path.is_file():
            negatives = fl.NegativeGeneSet.from_table(
                pd.read_parquet(negatives_path),
                gene_ids,
                classes=HUMAN_BROAD_CLASSES,
                state_gene_ids=load_state_gene_ids(species),
                max_fraction=config.flags.negative_gene_max_fraction,
            )
        if gene_ids and profiles_path.is_file():
            from merxen.annotation.composition import whb_broad_of

            profiles_by_class = fl.profiles_for_classes(
                pd.read_parquet(profiles_path),
                level=primary.leaf_level,
                gene_ids=gene_ids,
                node_class=whb_broad_of(),
                classes=HUMAN_BROAD_CLASSES,
            )
    flag_set = fl.compute_flags(
        fl.FlagInputs(
            species=species,
            platform=platform,
            total_counts=counts,
            in_table=table,
            assigned_class=assigned,
            confident=resolution.levels["broad"].confident,
            method_disagree=resolution.flags[Columns.FLAG_METHOD_DISAGREE],
            corr=(
                whb.supercluster.scores.corr
                if whb is not None and whb.supercluster.scores.corr is not None
                else np.full(n_objects, np.nan)
            ),
            depth_bin=resolution.depth_bin,
            exclude_hard=resolution.flags[Columns.EXCLUDE_HARD],
            query_counts=query_counts,
            query_rows=query_rows,
            gene_ids=gene_ids,
            negatives=negatives,
            profiles_by_class=profiles_by_class,
        ),
        config.flags,
        class_names=HUMAN_BROAD_CLASSES,
        seed=seed,
    )

    # Soft composition (§5.5).
    soft_rows = (
        soft_broad_from_level(inputs.leaf)
        if inputs.leaf is not None
        else np.tile(
            np.eye(len(HUMAN_BROAD_CLASSES) + 1)[-1], (len(inputs.table_rows), 1)
        )
    )
    table_rows = inputs.table_rows
    broad = resolution.levels["broad"]
    section = section_composition(
        sample_id,
        platform,
        soft_rows,
        total_counts=counts[table_rows],
        argmax_broad=np.where(
            assigned[table_rows] == None,  # noqa: E711
            UNASSIGNED_LABEL,
            assigned[table_rows],
        ),
        confident_broad=broad.name[table_rows],
        confident=broad.confident[table_rows],
        xy=None if xy is None else np.asarray(xy)[table_rows],
        aligned_frame=aligned_frame,
    )
    shares = section_shares(section)["whole_section"]

    # The label table (§4.1).
    branch, leaf = human_branch_columns(resolution)
    primary_panel_hash = (
        primary.record.panel_hash
        if primary is not None and primary.record.panel_hash
        else record.declared_panel_hash
    )
    n_missing = primary.record.n_missing_panel_genes if primary is not None else 0
    instance_ids = (
        np.asarray(loaded.instance_ids, dtype=np.int64)
        if loaded.instance_ids is not None
        else np.full(n_objects, MISSING_INSTANCE_ID, dtype=np.int64)
    )
    if loaded.instance_ids is None:
        logger.warning(
            "%s: the input has no instance ids; instance_id is %d",
            sample_id,
            MISSING_INSTANCE_ID,
        )
    n_genes = np.asarray(loaded.n_genes, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        genes_per_count = np.where(counts > 0, n_genes / counts, np.nan)
    data: dict[str, Any] = {
        Columns.CELL_ID: loaded.obs_names.astype(str),
        Columns.INSTANCE_ID: instance_ids,
        Columns.PAIR_ID: [manifest.pair_id or ""] * n_objects,
        Columns.SAMPLE_ID: [sample_id] * n_objects,
        Columns.PLATFORM: [platform] * n_objects,
        Columns.SEGMENTATION: [manifest.segmentation or ""] * n_objects,
        Columns.SPECIES: [species] * n_objects,
        Columns.ANATOMICAL_REGION: [config.anatomical_region] * n_objects,
        Columns.PANEL_HASH: [primary_panel_hash] * n_objects,
        Columns.TOTAL_COUNTS: counts.astype(np.int32),
        Columns.N_GENES: n_genes.astype(np.int32),
        Columns.GENES_PER_COUNT: np.clip(genes_per_count, 0.0, 1.0).astype(np.float32),
        Columns.IN_TABLE: table,
        Columns.N_MISSING_PANEL_GENES: np.full(n_objects, n_missing, dtype=np.int32),
        **resolution.to_columns(),
        Columns.CT_BRANCH: branch,
        Columns.CT_LEAF: leaf,
        Columns.CT_MENDER_STATE: branch,
        **flag_set.columns,
        **soft_broad_columns(soft_rows, table_rows, n_objects),
    }
    frame = pd.DataFrame(data, index=pd.RangeIndex(n_objects))
    engine_frames = []
    for run in runs.values():
        engine = engine_columns(
            run.tidy,
            run.prefix,
            pd.Index(loaded.obs_names.astype(str)),
            runner_up_levels=(
                [run.leaf_level_name] if run.record.role == "primary" else []
            ),
        )
        engine_frames.append(engine.reset_index(drop=True))
    if engine_frames:
        frame = pd.concat([frame, *engine_frames], axis=1)
    frame = coerce_label_table_dtypes(
        frame, species, include_fine_levels=config.thresholds.allow_fine_levels
    )
    if validate:
        validate_label_table(
            frame,
            species,
            h5ad_index=(
                loaded.obs_names.astype(str)
                if loaded.sample.source == "clustered"
                else None
            ),
            include_fine_levels=config.thresholds.allow_fine_levels,
        )

    # Provenance (§4.6).
    summary = resolution.summary()
    references: dict[str, ReferenceProvenance] = {}
    for item in (primary, secondary):
        if item is None:
            continue
        item_trust = trust if item is primary else secondary_trust
        references[item.record.reference_id] = _reference_provenance(item, item_trust)
    resolvability = {}
    if primary is not None:
        resolvability[primary.record.reference_id] = _resolvability_provenance(
            tables, emission, resolution, primary.bundle, reweighted=reweight
        )
    from merxen.annotation.diagnostics import (
        panel_diagnostics,
        panel_provenance,
    )

    panel_prov = None
    if primary is not None and trust is not None:
        primary_panel = panels.get(str(primary.record.panel_hash))
        diagnostics = None
        if primary_panel is not None:
            try:
                diagnostics = panel_diagnostics(
                    primary_panel,
                    panel_report=panel_report,
                    bundles=[primary.bundle.manifest],
                )
            except ValueError as error:
                logger.warning("%s: no panel diagnostics (%s)", sample_id, error)
        if diagnostics is not None:
            panel_prov = panel_provenance(
                trust,
                diagnostics,
                panel_mode=panel_mode,  # type: ignore[arg-type]
                validated_share={
                    level: float(value["validated_share"])
                    for level, value in summary["levels"].items()
                    if value["validated_share"] is not None
                },
                n_missing_panel_genes=n_missing,
            )
    if panel_prov is None and trust is not None:
        from merxen.annotation.provenance import PanelProvenance

        panel_prov = PanelProvenance(
            panel_hash=record.declared_panel_hash,
            panel_trust=trust.state,
            trust_reasons=trust.reason_codes,
            banner=trust.banner,
            n_missing_panel_genes=n_missing,
        )
    engine_record = primary.record if primary is not None else None
    params = engine_record.engine_params if engine_record is not None else {}
    sources = sorted(
        {
            emission.threshold_source(level)
            for level in ("lineage", "broad", "nt", "supercluster")
        }
    )
    floor_sources = sorted(
        {floors.policy(level) for level in ("broad", "supercluster")}
    )
    gate = resolution.gate
    tiers, tier_counts = np.unique(
        resolution.consensus_tier[table].astype(int), return_counts=True
    )
    provenance = AnnotationProvenance(
        species=species,
        anatomical_region=config.anatomical_region,
        merxen_version=_merxen_version(),
        panel=panel_prov,
        references=references,
        engine=EngineProvenance(
            ctm_version=None if engine_record is None else engine_record.ctm_version,
            ctm_commit=None if engine_record is None else engine_record.ctm_commit,
            bootstrap_factor=params.get("bootstrap_factor"),
            bootstrap_iteration=params.get("bootstrap_iteration"),
            rng_seed=params.get("rng_seed"),
            n_processors=params.get("n_processors"),
            wall_time_s=None if engine_record is None else engine_record.wall_s,
        ),
        resolvability=resolvability,
        thresholds=ThresholdProvenance(
            mode=config.thresholds.mode,
            values=_threshold_values(config),
            threshold_source=";".join(sources),
            floors_sha256=floors.table.sha256,
            floor_source=";".join(str(item) for item in floor_sources),
            calibration="none",
        ),
        flags=flag_set.provenance(),
        gate=DatasetGateProvenance(
            frac_ge30=_round_share(gate.frac_ge30),
            table_broad_coverage=_round_share(gate.broad_coverage_table),
            segmented_broad_coverage=_round_share(gate.broad_coverage_segmented),
            level=gate.level,
            warning=gate.warning,
            reasons=list(gate.reasons),
        ),
        consensus=ConsensusProvenance(
            degraded_mode=resolution.mode.name,
            methods=sorted(resolution.mode.methods),
            max_tier=resolution.mode.max_tier,
            single_method_override=resolution.single_method_override,
            likelihood_vote=False,
            tier_counts={
                str(int(tier)): int(count)
                for tier, count in zip(tiers, tier_counts, strict=True)
            },
            cop_suppressed=int(
                resolution.flags[Columns.FLAG_COP_SUPPRESSED][table].sum()
            ),
        ),
        composition=_composition_provenance(shares),
        confident_fraction_table={
            level: round(float(value["confident_share_table"]), 6)
            for level, value in summary["levels"].items()
            if value["confident_share_table"] is not None
        },
        confident_fraction_segmented={
            level: round(float(value["confident_share_segmented"]), 6)
            for level, value in summary["levels"].items()
            if value["confident_share_segmented"] is not None
        },
    )
    sample_summary = {
        "sample_id": sample_id,
        "platform": platform,
        "n_objects": n_objects,
        "n_table": int(table.sum()),
        "n_segmented": n_segmented,
        "trust": None if trust is None else trust.to_json(),
        "secondary_trust": (
            None if secondary_trust is None else secondary_trust.to_json()
        ),
        "banner": None if trust is None else trust.banner,
        "bundles": {
            run.run_id: {
                "reference_id": run.record.reference_id,
                "mapped_build_hash": run.record.build_hash,
                "resolved_build_hash": run.bundle.build_hash,
                "overridden": run.overridden,
                "same_lookup": run.same_lookup,
            }
            for run in runs.values()
        },
        "reweighted_to_composition": reweight,
        "resolution": summary,
        "flags": flag_set.summary(),
        "composition": shares,
    }
    return SampleResolution(
        sample_id=sample_id,
        platform=platform,
        labels=frame,
        provenance=provenance,
        summary=sample_summary,
        section=section,
    )


def _merxen_version() -> str | None:
    try:
        from merxen import __version__

        return str(__version__)
    except ImportError:  # pragma: no cover - the package always has it
        return None


def write_label_table(
    labels: pd.DataFrame, path: Path | str, provenance_json: str
) -> Path:
    """Write ``<sid>_celltype_labels.parquet`` atomically, provenance in the schema.

    Args:
        labels: The label table.
        path: Output file.
        provenance_json: ``AnnotationProvenance.to_uns_json()``.

    Returns:
        The written path.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(labels, preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata[LABELS_METADATA_KEY] = provenance_json.encode("utf-8")
    table = table.replace_schema_metadata(metadata)
    temporary = output.with_name(output.name + ".partial")
    pq.write_table(table, temporary)
    temporary.replace(output)
    return output


def read_label_table(
    path: Path | str,
) -> tuple[pd.DataFrame, AnnotationProvenance | None]:
    """Read a label table and the provenance stored in its schema.

    Args:
        path: ``<sid>_celltype_labels.parquet``.

    Returns:
        ``(labels, provenance)`` (``None`` without stored provenance).
    """
    import pyarrow.parquet as pq

    from merxen.annotation.provenance import AnnotationProvenance

    table = pq.read_table(path)
    metadata = table.schema.metadata or {}
    raw = metadata.get(LABELS_METADATA_KEY)
    provenance = (
        None if raw is None else AnnotationProvenance.from_uns_json(raw.decode("utf-8"))
    )
    return table.to_pandas(), provenance


def _write_json(payload: Mapping[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=str)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _json_safe(value: Any) -> Any:
    """Replace non-finite floats by ``None`` (JSON has no NaN)."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float | np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


@dataclass
class ResolveResult:
    """What ``annotate_resolve`` wrote for one pair x segmentation.

    Attributes:
        samples: Sample id to its resolution.
        summary: The ``<pair>_resolve_summary.json`` content.
        summary_path: Where it was written.
    """

    samples: dict[str, SampleResolution]
    summary: dict[str, Any]
    summary_path: Path


def annotate_resolve(
    map_dir: Path | str,
    config: AnnotationConfig,
    *,
    output_dir: Path | str,
    panel_dir: Path | str | None = None,
    samples: Sequence[MapSample] | None = None,
    bundle_overrides: Mapping[str, Path] | None = None,
    bundle_finder: BundleFinder | None = None,
    n_segmented: Mapping[str, int] | None = None,
    alignment_dir: Path | str | None = None,
    trust_overrides: Mapping[str, TrustDecision] | None = None,
    platforms: Sequence[str] | None = None,
    tile_um: float = 500.0,
    n_bootstrap: int = 200,
    seed: int = 0,
    validate: bool = True,
) -> ResolveResult:
    """Run the RESOLVE step for one pair x segmentation (plan §3.4).

    Reads ``map_manifest.json`` and its tidy MMC parquets, the bundles
    (profiles, negative genes, resolvability tables, trust) and the samples'
    counts; applies resolvability-gated emission reweighted to each dataset's
    soft composition with its local thresholds, the floors, the dataset gate,
    the degraded-mode consensus and the flags; and writes, per sample,
    ``<plat>/<sid>_celltype_labels.parquet`` (§4.1, validated) and
    ``<plat>/<sid>_annotation_manifest.json`` (§4.6), and the pair's
    ``<pair>_resolve_summary.json`` (gate levels and warnings, trust,
    realised flag rates, coverage, resolvable share per level, compositions
    and the pair JSD with block-bootstrap CIs).

    Args:
        map_dir: ``annotation_map_out`` (``map_manifest.json`` + parquets).
        config: The annotation config, coupled to the clustering
            ``min_counts``.
        output_dir: Where the outputs go.
        panel_dir: ``ANNOTATE_PANEL`` output (default: ``<map_dir>/panel``).
        samples: Where the counts come from (default: the manifest's inputs).
        bundle_overrides: Run id or reference id to a bundle directory.
        bundle_finder: Picks a bundle per (reference, panel) when no override
            names the run (``current_store_bundles``).
        n_segmented: Segmented objects per sample id (default: the objects
            of the input; a published clustered H5AD holds table cells only,
            so the segmented-object gate warning then uses table cells).
        alignment_dir: ``align_out`` of the pair (shared tissue mask; default:
            ``<results>/<pair>/alignment/align_out`` when the inputs sit in a
            results tree).
        trust_overrides: Trust decision per reference id.
        platforms: Resolve only these platforms.
        tile_um: Block-bootstrap tile edge.
        n_bootstrap: Block-bootstrap replicates.
        seed: Bootstrap and diffuse-simulation seed.
        validate: Validate every label table.

    Returns:
        The result.

    Raises:
        ResolveError: If the inputs do not fit together.
        NotImplementedError: For mouse (M6).
    """
    from merxen.annotation.composition import (
        pair_jsd,
        section_shares,
        shared_mask_regions,
    )

    start = time.monotonic()
    root = Path(map_dir)
    manifest = load_map_manifest(root / MAP_MANIFEST_NAME)
    if manifest.species != config.species:
        raise ResolveError(
            f"{root / MAP_MANIFEST_NAME} is a {manifest.species} run, the config "
            f"{config.species}"
        )
    if config.species != "human":
        raise NotImplementedError("mouse RESOLVE rules are M6 (plan §12)")
    min_counts = config.require_min_counts()
    if manifest.min_counts != min_counts:
        raise ResolveError(
            f"MAP used min_counts {manifest.min_counts}, the config {min_counts}"
        )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    panel_root = Path(panel_dir) if panel_dir is not None else root / "panel"
    panels = {
        panel_hash: current_family(load_annotation_panel(path), config)
        for panel_hash, path in _panel_files(panel_root).items()
    }
    report_path = panel_root / "panel_report.json"
    panel_report = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if report_path.is_file()
        else None
    )
    required_path = panel_root / REQUIRED_BUNDLES_FILE
    panel_mode = None
    if required_path.is_file():
        panel_mode = RequiredBundles.model_validate_json(
            required_path.read_text(encoding="utf-8")
        ).panel_mode
    wanted = {platform.upper() for platform in platforms} if platforms else None
    given = {sample.sample_id: sample for sample in samples or ()}
    records: list[MapSampleRecord | None] = [
        item
        for item in manifest.samples.values()
        if wanted is None or item.platform.upper() in wanted
    ]
    if records:
        inputs = [
            given.get(item.sample_id)
            or MapSample(
                sample_id=item.sample_id,
                platform=item.platform,
                h5ad_path=Path(item.h5ad_path),
                source=item.source,
            )
            for item in records
            if item is not None
        ]
    elif manifest.panel_status == "refused" and given:
        # A refused panel: MAP mapped nothing and recorded no sample, so
        # RESOLVE writes statuses only for the given inputs (§3.1).
        inputs = [
            sample
            for sample in given.values()
            if wanted is None or sample.platform.upper() in wanted
        ]
        records = [None] * len(inputs)
    else:
        raise ResolveError(
            f"{root / MAP_MANIFEST_NAME} has no sample to resolve"
            + (" (a refused panel needs the prepared inputs)" if not given else "")
        )
    loaded_samples = load_samples(inputs, config, min_counts=min_counts)
    segmented = dict(n_segmented or {})
    results: dict[str, SampleResolution] = {}
    for loaded, maybe_record in zip(loaded_samples, records, strict=True):
        fingerprint = loaded.sample_fingerprint()
        if maybe_record is None:
            record = refused_sample_record(loaded, fingerprint)
        else:
            record = maybe_record
            if fingerprint != record.sample_fingerprint:
                raise ResolveError(
                    f"{record.sample_id}: the counts of {loaded.sample.h5ad_path} "
                    "differ from the ones MAP mapped (sample fingerprint)"
                )
        runs = load_resolve_runs(
            root,
            record,
            bundle_overrides=bundle_overrides,
            bundle_finder=bundle_finder,
        )
        xy, shape_key = read_spatial(loaded.sample.h5ad_path)
        if xy is not None and len(xy) != loaded.n_objects:
            logger.warning(
                "%s: obsm['spatial'] does not fit the objects; no coordinates",
                record.sample_id,
            )
            xy = None
        n_objects_segmented = segmented.get(record.sample_id)
        if n_objects_segmented is None and loaded.sample.source == "clustered":
            logger.warning(
                "%s: a clustered H5AD holds table cells only; the segmented-object "
                "gate warning uses them unless n_segmented is given",
                record.sample_id,
            )
        result = resolve_human_sample(
            loaded,
            record,
            runs,
            config,
            manifest=manifest,
            panels=panels,
            panel_report=panel_report,
            n_segmented=n_objects_segmented,
            trust_overrides=trust_overrides,
            xy=xy,
            aligned_frame=in_fixed_frame(record.platform, shape_key),
            panel_mode=panel_mode,
            validate=validate,
            seed=seed,
        )
        folder = output / record.platform.lower()
        provenance_json = result.provenance.to_uns_json()
        result.labels_path = write_label_table(
            result.labels,
            folder / label_table_filename(record.sample_id),
            provenance_json,
        )
        result.manifest_path = _write_json(
            json.loads(provenance_json),
            folder / annotation_manifest_filename(record.sample_id),
        )
        result.summary["labels"] = _relative(result.labels_path, output)
        result.summary["labels_sha256"] = file_sha256(result.labels_path)
        result.summary["annotation_manifest"] = _relative(result.manifest_path, output)
        results[record.sample_id] = result
        logger.info(
            "%s: wrote %s (%d objects)",
            record.sample_id,
            result.labels_path,
            len(result.labels),
        )

    # Pair composition statistics (§5.5; H1).
    pair: dict[str, Any] = {"jsd": [], "mask_note": None}
    by_platform = {item.platform: item for item in results.values()}
    if {"MERSCOPE", "XENIUM"} <= set(by_platform):
        first = by_platform["MERSCOPE"].section
        second = by_platform["XENIUM"].section
        assert first is not None and second is not None
        align: Path | None = None
        if alignment_dir is not None:
            align = Path(alignment_dir)
        else:
            source_path = next(
                (item.h5ad_path for item in inputs if item.platform == "MERSCOPE"),
                None,
            )
            if source_path is not None:
                align = default_alignment_dir(source_path, manifest.pair_id)
        mask = load_pair_mask(align)
        records_jsd, note = pair_jsd(
            first, second, mask=mask, tile_um=tile_um, n_reps=n_bootstrap, seed=seed
        )
        regions, _ = shared_mask_regions(first, second, mask)
        compositions = {
            section.platform: section_shares(
                section,
                regions={name: keep[index] for name, keep in regions.items()},
            )
            for index, section in enumerate((first, second))
        }
        pair = {
            "jsd": [item.to_json() for item in records_jsd],
            "mask_note": note,
            "alignment_dir": None if align is None else str(align),
            "tile_um": tile_um,
            "n_bootstrap": n_bootstrap,
            "compositions": compositions,
        }
    summary = _json_safe(
        {
            "schema_version": RESOLVE_SUMMARY_SCHEMA_VERSION,
            "step": RESOLVE_STEP,
            "pair_id": manifest.pair_id,
            "segmentation": manifest.segmentation,
            "species": manifest.species,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "map_manifest": str((root / MAP_MANIFEST_NAME).resolve()),
            "panel_status": manifest.panel_status,
            "panel_mode": panel_mode,
            "thresholds": config.thresholds.model_dump(mode="json"),
            "flags_config": config.flags.model_dump(mode="json"),
            "samples": {key: value.summary for key, value in results.items()},
            "pair": pair,
            "wall_time_s": round(time.monotonic() - start, 3),
        }
    )
    summary_path = _write_json(
        summary, output / resolve_summary_filename(manifest.pair_id)
    )
    return ResolveResult(samples=results, summary=summary, summary_path=summary_path)

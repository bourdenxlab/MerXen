"""MapMyCells engine of the annotation MAP step (plan §3.3, §4.1, D-A1, D-A8).

``run_mmc`` maps one query H5AD onto one reference bundle with the validated
configuration and returns a tidy per-cell x level table:

* ``cell_type_mapper`` runs in a subprocess of the existing
  ``merxen.analysis.mapmycells_entrypoint`` with ``bootstrap_factor`` 0.5,
  ``bootstrap_iteration`` 100, ``rng_seed`` 0, ``n_processors`` = the task's
  CPUs, ``normalization raw`` and ``cloud_safe False``;
* ``OMP`` / ``OPENBLAS`` / ``MKL`` / ``NUMBA`` / ``NUMEXPR_NUM_THREADS`` are 1
  in the subprocess (6 workers x 6 BLAS threads oversubscribe the task; R-eng)
  and ``CUDA_VISIBLE_DEVICES`` is empty (CPU-only process);
* a bundle with a ``drop_level`` (WMB: ``CCN20230722_SUPT``) passes it, with
  the bundle's mapping precompute (the Allen means) and its panel lookup
  filtered to the mapping tree;
* the installed ``cell_type_mapper`` must be the configured version
  (``annotation_ctm_version``, "1.7.2"); the extended JSON is parsed at once
  into a tidy parquet (assignment, name, bootstrap probability, aggregate
  probability, ``avg_correlation``, runner-ups 1-5 with their probabilities
  and correlations, ``directly_assigned``) and deleted, or kept gzipped when
  ``keep_extended_json`` is set. Nothing goes into ``uns`` (plan §3.3).

The module also holds the small helpers MAP needs around the engine: the
query writer, the query fingerprint used for published-output reuse
(``annotation_reuse_published``), lookup restriction for datasets that miss
panel genes, and the coarse-level probability aggregation (E1 / E2: the
bootstrap probability of the assigned node plus that of the runner-ups in the
same class).

``cell_type_mapper`` is never imported here; ``anndata`` and ``pyarrow`` are
imported inside the functions that need them.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from merxen.annotation.reference import (
    LookupValidation,
    TaxonomyTreeView,
    lookup_sha256,
    read_lookup,
    validate_lookup,
)
from merxen.annotation.store import BUNDLE_MANIFEST_NAME, BUNDLE_STATUS_COMPLETE

if TYPE_CHECKING:
    from merxen.annotation.config import AnnotationReferenceSpec
    from merxen.annotation.store import BundleRef

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Constants

MMC_ENTRYPOINT_MODULE: Final = "merxen.analysis.mapmycells_entrypoint"
CTM_DISTRIBUTION: Final = "cell_type_mapper"
# One BLAS / numba / numexpr thread per ctm worker: ``n_processors`` workers
# already fill the task's CPUs (plan §3.3; ctm itself warns about the first
# three when they are unset).
MMC_SINGLE_THREAD_ENV: Final[tuple[str, ...]] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
N_RUNNERS_UP: Final = 5
TIDY_SCHEMA_VERSION: Final = 1
TIDY_METADATA_KEY: Final = b"merxen_mmc"
QUERY_FINGERPRINT_VERSION: Final = 1
EXTENDED_JSON_GZ_SUFFIX: Final = ".extended.json.gz"
GNU_TIME: Final = Path("/usr/bin/time")

TIDY_BASE_COLUMNS: Final[tuple[str, ...]] = (
    "cell_id",
    "level",
    "level_name",
    "assignment",
    "name",
    "bp",
    "aggregate_probability",
    "avg_correlation",
    "directly_assigned",
    "n_runners_up",
)
RUNNER_UP_FIELDS: Final[tuple[str, ...]] = (
    "assignment",
    "name",
    "probability",
    "correlation",
)


def runner_up_column(rank: int, field_name: str) -> str:
    """Return the tidy column of one runner-up field.

    Args:
        rank: Runner-up rank, 1 (best alternative) to ``N_RUNNERS_UP``.
        field_name: One of ``RUNNER_UP_FIELDS``.

    Returns:
        e.g. ``"runner_up_1_probability"``.

    Raises:
        ValueError: If the rank or field is out of range.
    """
    if not 1 <= rank <= N_RUNNERS_UP:
        raise ValueError(f"runner-up rank must be 1..{N_RUNNERS_UP}, got {rank}")
    if field_name not in RUNNER_UP_FIELDS:
        raise ValueError(f"field must be one of {RUNNER_UP_FIELDS}, got {field_name!r}")
    return f"runner_up_{rank}_{field_name}"


TIDY_COLUMNS: Final[tuple[str, ...]] = (
    *TIDY_BASE_COLUMNS,
    *(
        runner_up_column(rank, field_name)
        for rank in range(1, N_RUNNERS_UP + 1)
        for field_name in RUNNER_UP_FIELDS
    ),
)
_CATEGORY_COLUMNS: Final[frozenset[str]] = frozenset(
    {"level", "level_name", "assignment", "name"}
    | {
        runner_up_column(rank, field_name)
        for rank in range(1, N_RUNNERS_UP + 1)
        for field_name in ("assignment", "name")
    }
)


class MmcEngineError(RuntimeError):
    """The MapMyCells engine could not produce a mapping."""


class CtmVersionError(MmcEngineError):
    """The installed ``cell_type_mapper`` is not the configured version."""


class ExtendedJsonError(MmcEngineError):
    """A MapMyCells extended JSON is malformed or does not match its query."""


# --------------------------------------------------------------------------
# Engine parameters and bundles


class MmcEngineParams(BaseModel):
    """Engine parameters of one MapMyCells run (plan §3.3, D-A8).

    Every field is part of the published-output reuse key: the seed changes
    5.5% of SEA-AD subclass labels and the process count about 3% at D15
    (E5, R12), so a run is reused only under identical parameters.

    Attributes:
        bootstrap_factor: Fraction of markers per bootstrap iteration.
        bootstrap_iteration: Bootstrap iterations.
        rng_seed: Seed.
        n_processors: Worker processes (the task's CPUs).
        normalization: Query normalization; the pipeline maps raw counts.
        cloud_safe: ctm's ``cloud_safe`` (off: local paths in the JSON).
        n_runners_up: Runner-ups kept per level (5).
        chunk_size: ctm chunk size; ``None`` keeps ctm's default (10,000).
        max_gb: ctm memory bound for CSC conversion; ``None`` keeps ctm's.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    bootstrap_factor: float = Field(default=0.5, gt=0.0, le=1.0)
    bootstrap_iteration: int = Field(default=100, ge=1)
    rng_seed: int = 0
    n_processors: int = Field(default=6, ge=1)
    normalization: Literal["raw", "log2CPM"] = "raw"
    cloud_safe: bool = False
    n_runners_up: int = Field(default=N_RUNNERS_UP, ge=0, le=N_RUNNERS_UP)
    chunk_size: int | None = Field(default=None, ge=1)
    max_gb: int | None = Field(default=None, ge=1)

    @classmethod
    def from_reference_spec(
        cls, spec: AnnotationReferenceSpec, *, n_processors: int
    ) -> MmcEngineParams:
        """Return the engine parameters of a reference spec.

        Args:
            spec: The reference spec (bootstrap settings and seed).
            n_processors: Worker processes (the task's CPUs).

        Returns:
            The parameters, raw normalization, ``cloud_safe`` off.
        """
        return cls(
            bootstrap_factor=spec.bootstrap_factor,
            bootstrap_iteration=spec.bootstrap_iteration,
            rng_seed=spec.rng_seed,
            n_processors=n_processors,
        )

    def reuse_key(self) -> dict[str, Any]:
        """Return the JSON dict compared for published-output reuse."""
        return self.model_dump(mode="json")


@dataclass(frozen=True)
class MmcBundle:
    """What MAP reads from one complete reference bundle (plan §3.2).

    Attributes:
        reference_id: Store id.
        role: Reference role.
        species: Species.
        build_hash: The bundle's ``build_hash``.
        path: Bundle directory.
        panel_hash: Declared-panel hash the bundle was built on.
        n_panel_genes: Genes of that panel.
        taxonomy_id: Allen taxonomy id.
        levels: Levels of the mapping tree, coarse to fine.
        drop_level: Level ctm drops from the mapping precompute (WMB SUPT).
        mapping_precompute: The mapping precompute (``--precomputed_stats``).
        lookup: The panel lookup filtered to the mapping tree.
        lookup_sha256: Marker-content sha256 of ``lookup`` (``bundle.json``).
        mapping_tree: ``mapping_tree.json`` (the tree ctm maps onto).
        vocab_snapshot: ``vocab_snapshot.csv`` (node classes), if present.
        builder_version: ``ANNOTATION_BUILDER_VERSION`` of the build.
        manifest: The full ``bundle.json``.
    """

    reference_id: str
    role: str
    species: str
    build_hash: str
    path: Path
    panel_hash: str | None
    n_panel_genes: int | None
    taxonomy_id: str | None
    levels: tuple[str, ...]
    drop_level: str | None
    mapping_precompute: Path
    lookup: Path
    lookup_sha256: str | None
    mapping_tree: Path
    vocab_snapshot: Path | None
    builder_version: int | None
    manifest: dict[str, Any] = field(repr=False, compare=False)

    @classmethod
    def from_dir(cls, path: Path | str) -> MmcBundle:
        """Read a complete bundle directory.

        Every file ``bundle.json`` lists must exist with its recorded size;
        the marker lookup is also checked against its recorded digest.

        Args:
            path: ``<store>/<reference_id>/<build_hash>``.

        Returns:
            The bundle.

        Raises:
            MmcEngineError: If the directory is not a complete, intact bundle
                or lacks a mapping precompute or lookup.
        """
        bundle_dir = Path(path)
        manifest_path = bundle_dir / BUNDLE_MANIFEST_NAME
        try:
            manifest: object = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise MmcEngineError(f"cannot read {manifest_path}: {error}") from error
        if not isinstance(manifest, dict):
            raise MmcEngineError(f"{manifest_path} does not hold a JSON object")
        if manifest.get("status") != BUNDLE_STATUS_COMPLETE:
            raise MmcEngineError(
                f"bundle {bundle_dir} is not complete (status "
                f"{manifest.get('status')!r})"
            )
        problems = []
        for record in manifest.get("files", []) or []:
            file_path = bundle_dir / str(record.get("path", ""))
            if not file_path.is_file() or file_path.is_symlink():
                problems.append(f"missing {record.get('path')!r}")
            elif file_path.stat().st_size != record.get("size"):
                problems.append(f"size of {record.get('path')!r} changed")
        if problems:
            raise MmcEngineError(
                f"bundle {bundle_dir} is not intact: {'; '.join(problems)}"
            )
        output = manifest.get("builder_output") or {}
        markers = output.get("markers") or {}
        precompute = (output.get("mapping_precompute") or {}).get("file")
        lookup_file = markers.get("lookup_file")
        if not precompute or not lookup_file:
            raise MmcEngineError(
                f"bundle {bundle_dir} records no mapping precompute or marker "
                "lookup; it cannot be mapped onto (panel-independent reference?)"
            )
        panel = manifest.get("panel") or {}
        vocab_file = output.get("vocab_snapshot_file")
        bundle = cls(
            reference_id=str(manifest.get("reference_id")),
            role=str(manifest.get("role")),
            species=str(manifest.get("species")),
            build_hash=str(manifest.get("build_hash")),
            path=bundle_dir,
            panel_hash=panel.get("panel_hash") if isinstance(panel, dict) else None,
            n_panel_genes=panel.get("n_genes") if isinstance(panel, dict) else None,
            taxonomy_id=manifest.get("taxonomy_id"),
            levels=tuple(str(level) for level in output.get("levels") or ()),
            drop_level=output.get("drop_level"),
            mapping_precompute=bundle_dir / str(precompute),
            lookup=bundle_dir / str(lookup_file),
            lookup_sha256=markers.get("lookup_sha256"),
            mapping_tree=bundle_dir / str(output.get("mapping_tree_file")),
            vocab_snapshot=bundle_dir / str(vocab_file) if vocab_file else None,
            builder_version=manifest.get("builder_version"),
            manifest=manifest,
        )
        if bundle.lookup_sha256 is not None:
            actual = lookup_sha256(read_lookup(bundle.lookup))
            if actual != bundle.lookup_sha256:
                raise MmcEngineError(
                    f"marker lookup {bundle.lookup} does not match the digest "
                    "recorded in bundle.json"
                )
        return bundle

    @classmethod
    def from_bundle_ref(cls, ref: BundleRef) -> MmcBundle:
        """Read the bundle a ``bundle_ref.json`` points to.

        Args:
            ref: The bundle reference from ``ANNOTATE_REFERENCE_PREP``.

        Returns:
            The bundle.

        Raises:
            MmcEngineError: If the bundle does not match the reference.
        """
        bundle = cls.from_dir(ref.path)
        if bundle.build_hash != ref.build_hash:
            raise MmcEngineError(
                f"bundle {ref.path} has build_hash {bundle.build_hash[:16]}, "
                f"bundle_ref.json says {ref.build_hash[:16]}"
            )
        return bundle

    def tree(self) -> TaxonomyTreeView:
        """Return the mapping tree (``mapping_tree.json``)."""
        payload = json.loads(self.mapping_tree.read_text(encoding="utf-8"))
        return TaxonomyTreeView.from_tree_dict(payload)

    def vocab(self) -> pd.DataFrame | None:
        """Return the bundle's node vocabulary, or ``None`` without one.

        Returns:
            ``vocab_snapshot.csv`` (one row per level x node) indexed by
            ``(level, node)``.
        """
        if self.vocab_snapshot is None or not self.vocab_snapshot.is_file():
            return None
        frame = pd.read_csv(self.vocab_snapshot, dtype=str, keep_default_na=False)
        return frame.set_index(["level", "node"], drop=False)


# --------------------------------------------------------------------------
# Environment and version


def installed_ctm_version() -> str | None:
    """Return the installed ``cell_type_mapper`` version (``None`` if absent).

    The mapper subprocess runs with ``sys.executable``, so this interpreter's
    distribution is the one that maps.
    """
    try:
        return importlib.metadata.version(CTM_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        return None


def check_ctm_version(expected: str) -> str:
    """Assert the installed ``cell_type_mapper`` is the configured version.

    A stale Nextflow conda env once shipped ctm 1.5.5 (R13); the lock-hash
    header forces a rebuild and this check is the second line of defence.

    Args:
        expected: ``AnnotationConfig.ctm_version`` (``annotation_ctm_version``).

    Returns:
        The installed version.

    Raises:
        CtmVersionError: If ctm is missing or another version.
    """
    installed = installed_ctm_version()
    if installed != expected:
        raise CtmVersionError(
            f"cell_type_mapper {installed or 'is not installed'} in "
            f"{sys.executable}; annotation needs {expected} "
            "(annotation_ctm_version). Rebuild the environment from "
            "requirements/requirements.lock."
        )
    return installed


def ctm_commit() -> str | None:
    """Return the VCS commit of the installed ``cell_type_mapper``, if known."""
    try:
        distribution = importlib.metadata.distribution(CTM_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        return None
    text = distribution.read_text("direct_url.json")
    if not text:
        return None
    try:
        payload: object = json.loads(text)
    except ValueError:
        return None
    vcs_info = payload.get("vcs_info") if isinstance(payload, dict) else None
    if isinstance(vcs_info, dict) and vcs_info.get("commit_id"):
        return str(vcs_info["commit_id"])
    return None


def _package_src_dir() -> Path:
    import merxen

    return Path(merxen.__file__).resolve().parents[1]


def mmc_environment(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return the environment of the mapper subprocess.

    One BLAS / numba / numexpr thread per worker, no GPU, and this process's
    ``merxen`` source first on ``PYTHONPATH`` so the subprocess runs the same
    code (the Nextflow processes export ``PYTHONPATH`` for the same reason).

    Args:
        base: Environment to start from (default: ``os.environ``).

    Returns:
        The environment.
    """
    environment = dict(os.environ if base is None else base)
    for name in MMC_SINGLE_THREAD_ENV:
        environment[name] = "1"
    environment["CUDA_VISIBLE_DEVICES"] = ""
    source = str(_package_src_dir())
    existing = [item for item in environment.get("PYTHONPATH", "").split(":") if item]
    if source not in existing:
        existing.insert(0, source)
    environment["PYTHONPATH"] = ":".join(existing)
    return environment


def _bool_arg(value: bool) -> str:
    return "True" if value else "False"


def build_mmc_command(
    query_h5ad: Path,
    bundle: MmcBundle,
    params: MmcEngineParams,
    *,
    extended_json: Path,
    log_path: Path,
    lookup_path: Path | None = None,
    tmp_dir: Path | None = None,
    nodes_to_drop: Sequence[tuple[str, str]] = (),
) -> list[str]:
    """Build the mapper command line (``from_specified_markers`` options).

    Args:
        query_h5ad: Query H5AD (raw counts in ``X``, Ensembl IDs as
            ``var_names``).
        bundle: The reference bundle.
        params: Engine parameters.
        extended_json: Where ctm writes the extended JSON.
        log_path: ctm's log file.
        lookup_path: Marker lookup to use instead of the bundle's (a lookup
            restricted to the genes a dataset has).
        tmp_dir: ctm scratch directory.
        nodes_to_drop: ``(level, node)`` pairs removed from the tree (the
            mouse pruned re-map, M6).

    Returns:
        The command.
    """
    command = [
        sys.executable,
        "-m",
        MMC_ENTRYPOINT_MODULE,
        "--query_path",
        str(query_h5ad),
        "--extended_result_path",
        str(extended_json),
        "--log_path",
        str(log_path),
        "--cloud_safe",
        _bool_arg(params.cloud_safe),
        "--verbose_stdout",
        "False",
        "--query_markers.serialized_lookup",
        str(lookup_path or bundle.lookup),
        "--precomputed_stats.path",
        str(bundle.mapping_precompute),
        "--type_assignment.normalization",
        params.normalization,
        "--type_assignment.bootstrap_iteration",
        str(params.bootstrap_iteration),
        "--type_assignment.bootstrap_factor",
        str(params.bootstrap_factor),
        "--type_assignment.rng_seed",
        str(params.rng_seed),
        "--type_assignment.n_processors",
        str(params.n_processors),
        "--type_assignment.n_runners_up",
        str(params.n_runners_up),
    ]
    if bundle.drop_level is not None:
        command += ["--drop_level", bundle.drop_level]
    if nodes_to_drop:
        command += [
            "--nodes_to_drop",
            json.dumps([[str(level), str(node)] for level, node in nodes_to_drop]),
        ]
    if params.chunk_size is not None:
        command += ["--type_assignment.chunk_size", str(params.chunk_size)]
    if params.max_gb is not None:
        command += ["--max_gb", str(params.max_gb)]
    if tmp_dir is not None:
        command += ["--tmp_dir", str(tmp_dir)]
    return command


# --------------------------------------------------------------------------
# Query


def write_query_h5ad(
    counts: Any,
    obs_names: Sequence[str],
    var_names: Sequence[str],
    path: Path | str,
) -> Path:
    """Write a minimal MapMyCells query: raw counts, cell ids, gene IDs.

    Args:
        counts: Cells x genes counts (dense or scipy sparse).
        obs_names: Cell ids (unique).
        var_names: Ensembl gene IDs (unique).
        path: Output ``.h5ad``.

    Returns:
        The written path.

    Raises:
        ValueError: If the names are not unique or do not fit the matrix.
    """
    import anndata as ad
    from scipy import sparse

    obs = pd.Index([str(name) for name in obs_names])
    var = pd.Index([str(name) for name in var_names])
    if not obs.is_unique or not var.is_unique:
        raise ValueError("query cell ids and gene IDs must be unique")
    matrix = sparse.csr_matrix(counts)
    if matrix.shape != (len(obs), len(var)):
        raise ValueError(
            f"counts shape {matrix.shape} does not fit {len(obs)} cells x "
            f"{len(var)} genes"
        )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    adata = ad.AnnData(
        X=matrix,
        obs=pd.DataFrame(index=obs),
        var=pd.DataFrame(index=var),
    )
    adata.write_h5ad(output)
    return output


def query_fingerprint(
    obs_names: Sequence[str] | pd.Index,
    total_counts: Sequence[float] | np.ndarray,
    var_names: Sequence[str] | pd.Index,
) -> str:
    """Return the query fingerprint used for published-output reuse.

    sha256 over the cell ids, their total counts (as float64, so integer and
    float count matrices with equal values agree) and the gene IDs, each in
    order (plan §3.1: ``annotation_reuse_published``).

    Args:
        obs_names: Mapped cell ids.
        total_counts: Their total counts after control removal.
        var_names: Query gene IDs.

    Returns:
        The hex digest.

    Raises:
        ValueError: If ``total_counts`` does not have one value per cell.
    """
    cells = [str(name) for name in obs_names]
    genes = [str(name) for name in var_names]
    counts = np.asarray(total_counts, dtype="<f8").ravel()
    if counts.size != len(cells):
        raise ValueError(
            f"{counts.size} total counts for {len(cells)} cells in the fingerprint"
        )
    digest = hashlib.sha256()
    digest.update(f"merxen-mmc-query-v{QUERY_FINGERPRINT_VERSION}\n".encode())
    for label, names in (("obs_names", cells), ("var_names", genes)):
        encoded = json.dumps(names, ensure_ascii=False).encode("utf-8")
        digest.update(f"{label}:{len(names)}:{len(encoded)}\n".encode())
        digest.update(encoded)
    digest.update(f"total_counts:{counts.size}\n".encode())
    digest.update(counts.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class RestrictedLookup:
    """A bundle lookup restricted to the genes one dataset has.

    Attributes:
        path: The written lookup (``None`` when no gene is missing and the
            bundle's lookup is used as is).
        validation: ``validate_lookup`` diagnostics (``None`` when unused).
        lookup_sha256: Marker-content sha256 of the lookup used.
    """

    path: Path | None
    validation: LookupValidation | None
    lookup_sha256: str | None


def restrict_lookup(
    bundle: MmcBundle,
    query_genes: Iterable[str],
    output: Path | str,
) -> RestrictedLookup:
    """Restrict a bundle's lookup to the genes a dataset has (plan §3.3 step 1).

    Parents left without markers are auto-collapsed and recorded
    (``validate_lookup``); the lookup is written only when a marker is lost.

    Args:
        bundle: The bundle.
        query_genes: Gene IDs of the query.
        output: Where a restricted lookup is written.

    Returns:
        The lookup to map with.
    """
    genes = set(query_genes)
    lookup = read_lookup(bundle.lookup)
    markers = {
        gene
        for key, values in lookup.items()
        if key not in {"metadata", "log"}
        for gene in values
    }
    if markers <= genes:
        return RestrictedLookup(
            path=None, validation=None, lookup_sha256=bundle.lookup_sha256
        )
    validation = validate_lookup(lookup, bundle.tree(), genes)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(validation.lookup) + "\n", encoding="utf-8")
    logger.warning(
        "%s: %d marker gene(s) absent from the query; lookup restricted "
        "(%d parent(s) collapsed) -> %s",
        bundle.reference_id,
        len(markers - genes),
        len(validation.collapsed),
        path,
    )
    return RestrictedLookup(
        path=path,
        validation=validation,
        lookup_sha256=lookup_sha256(validation.lookup),
    )


# --------------------------------------------------------------------------
# Extended JSON -> tidy table


def _level_token(value: str) -> str:
    token = "".join(ch if ch.isalnum() else "_" for ch in str(value).strip().lower())
    return "_".join(part for part in token.split("_") if part) or "level"


def _float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number


def _names_from_tree(tree: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    names: dict[str, dict[str, str]] = {}
    mapper = tree.get("name_mapper") or {}
    if not isinstance(mapper, Mapping):
        return names
    for level, entries in mapper.items():
        if not isinstance(entries, Mapping):
            continue
        names[str(level)] = {
            str(node): str(value["name"])
            for node, value in entries.items()
            if isinstance(value, Mapping) and value.get("name") is not None
        }
    return names


def parse_extended_json_tidy(
    source: Path | str | Mapping[str, Any],
    *,
    cell_order: Sequence[str] | None = None,
    names: Mapping[str, Mapping[str, str]] | None = None,
) -> pd.DataFrame:
    """Parse a MapMyCells extended JSON into a tidy per-cell x level table.

    Args:
        source: The extended JSON file, or its decoded payload.
        cell_order: Query cell ids; the table follows this order and every
            id must appear exactly once in the results.
        names: Fallback node names by level (e.g. the bundle's mapping
            tree), used where the JSON's ``name_mapper`` has none.

    Returns:
        One row per cell x level (levels in taxonomy order): ``TIDY_COLUMNS``
        with category, float32, int8 and nullable-boolean dtypes. Runner-ups
        beyond those ctm reported are null.

    Raises:
        ExtendedJsonError: If the payload has no results, a level entry is
            malformed, or the cells do not match ``cell_order``.
    """
    if isinstance(source, Mapping):
        payload: Any = source
    else:
        try:
            payload = json.loads(Path(source).read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ExtendedJsonError(
                f"cannot read extended JSON {source}: {error}"
            ) from error
    if not isinstance(payload, Mapping):
        raise ExtendedJsonError("the extended JSON does not hold an object")
    results = payload.get("results")
    if not isinstance(results, list):
        raise ExtendedJsonError("the extended JSON has no results list")
    tree = payload.get("taxonomy_tree")
    tree = tree if isinstance(tree, Mapping) else {}
    hierarchy = [str(level) for level in tree.get("hierarchy") or []]
    level_mapper = tree.get("hierarchy_mapper")
    level_mapper = level_mapper if isinstance(level_mapper, Mapping) else {}
    json_names = _names_from_tree(tree)
    fallback_names = {str(k): dict(v) for k, v in (names or {}).items()}

    present: list[str] = []
    for result in results[:1]:
        if isinstance(result, Mapping):
            present = [
                str(key)
                for key, value in result.items()
                if key != "cell_id" and isinstance(value, Mapping)
            ]
    levels = [level for level in hierarchy if level in present]
    levels += [level for level in present if level not in levels]

    cell_ids = [str(result.get("cell_id", "")) for result in results]
    if any(not cell_id for cell_id in cell_ids):
        raise ExtendedJsonError("an extended JSON result has no cell_id")
    if len(set(cell_ids)) != len(cell_ids):
        raise ExtendedJsonError("the extended JSON repeats cell ids")
    if cell_order is not None:
        order = [str(cell) for cell in cell_order]
        position = {cell: index for index, cell in enumerate(cell_ids)}
        missing = [cell for cell in order if cell not in position]
        if missing or len(order) != len(cell_ids):
            raise ExtendedJsonError(
                f"extended JSON cells do not match the query: {len(missing)} "
                f"query cell(s) missing, {len(cell_ids)} results for "
                f"{len(order)} query cells"
            )
        results = [results[position[cell]] for cell in order]
        cell_ids = order

    n_cells = len(cell_ids)
    frames = []
    for level in levels:
        name_of = {**fallback_names.get(level, {}), **json_names.get(level, {})}
        assignment = np.empty(n_cells, dtype=object)
        bp = np.full(n_cells, np.nan, dtype=np.float64)
        aggregate = np.full(n_cells, np.nan, dtype=np.float64)
        correlation = np.full(n_cells, np.nan, dtype=np.float64)
        direct = np.empty(n_cells, dtype=object)
        n_runners = np.zeros(n_cells, dtype=np.int8)
        ru_assignment = np.full((N_RUNNERS_UP, n_cells), None, dtype=object)
        ru_probability = np.full((N_RUNNERS_UP, n_cells), np.nan, dtype=np.float64)
        ru_correlation = np.full((N_RUNNERS_UP, n_cells), np.nan, dtype=np.float64)
        for index, result in enumerate(results):
            entry = result.get(level)
            if not isinstance(entry, Mapping):
                raise ExtendedJsonError(
                    f"cell {cell_ids[index]} has no {level} entry in the extended JSON"
                )
            label = entry.get("assignment")
            assignment[index] = None if label is None else str(label)
            bp[index] = _float(entry.get("bootstrapping_probability"))
            aggregate[index] = _float(entry.get("aggregate_probability"))
            correlation[index] = _float(entry.get("avg_correlation"))
            flag = entry.get("directly_assigned")
            direct[index] = bool(flag) if isinstance(flag, bool) else None
            runners = entry.get("runner_up_assignment") or []
            probabilities = entry.get("runner_up_probability") or []
            correlations = entry.get("runner_up_correlation") or []
            count = min(len(runners), N_RUNNERS_UP)
            n_runners[index] = count
            for rank in range(count):
                ru_assignment[rank, index] = str(runners[rank])
                if rank < len(probabilities):
                    ru_probability[rank, index] = _float(probabilities[rank])
                if rank < len(correlations):
                    ru_correlation[rank, index] = _float(correlations[rank])
        columns: dict[str, Any] = {
            "cell_id": cell_ids,
            "level": [level] * n_cells,
            "level_name": [_level_token(level_mapper.get(level, level))] * n_cells,
            "assignment": assignment,
            "name": [
                None if label is None else name_of.get(label, label)
                for label in assignment
            ],
            "bp": bp,
            "aggregate_probability": aggregate,
            "avg_correlation": correlation,
            "directly_assigned": direct,
            "n_runners_up": n_runners,
        }
        for rank in range(N_RUNNERS_UP):
            labels = ru_assignment[rank]
            columns[runner_up_column(rank + 1, "assignment")] = labels
            columns[runner_up_column(rank + 1, "name")] = [
                None if label is None else name_of.get(label, label) for label in labels
            ]
            columns[runner_up_column(rank + 1, "probability")] = ru_probability[rank]
            columns[runner_up_column(rank + 1, "correlation")] = ru_correlation[rank]
        frames.append(pd.DataFrame(columns))
    if frames:
        tidy = pd.concat(frames, ignore_index=True)
    else:
        tidy = pd.DataFrame({column: [] for column in TIDY_COLUMNS})
    return coerce_tidy_dtypes(tidy[list(TIDY_COLUMNS)])


def coerce_tidy_dtypes(tidy: pd.DataFrame) -> pd.DataFrame:
    """Return the tidy table with its contract dtypes.

    Args:
        tidy: A tidy MMC table (``TIDY_COLUMNS``).

    Returns:
        A copy: ``cell_id`` string, labels and names category, probabilities
        and correlations float32, ``n_runners_up`` int8, ``directly_assigned``
        nullable boolean.
    """
    result = tidy.copy()
    result["cell_id"] = result["cell_id"].astype(str)
    for column in result.columns:
        if column in _CATEGORY_COLUMNS:
            result[column] = result[column].astype("category")
        elif column in {"cell_id"}:
            continue
        elif column == "n_runners_up":
            result[column] = result[column].astype(np.int8)
        elif column == "directly_assigned":
            result[column] = result[column].astype("boolean")
        else:
            result[column] = result[column].astype(np.float32)
    return result


def extended_json_config(source: Path | str | Mapping[str, Any]) -> dict[str, Any]:
    """Return the effective mapping settings an extended JSON records.

    Args:
        source: The extended JSON file or payload.

    Returns:
        ``type_assignment`` settings (seed, bootstrap, processes, runner-ups,
        chunk size, normalization), ``drop_level``, ``nodes_to_drop``, ctm's
        version from ``metadata`` and ``n_unmapped_genes``.
    """
    if isinstance(source, Mapping):
        payload: Any = source
    else:
        payload = json.loads(Path(source).read_text(encoding="utf-8"))
    config = payload.get("config") if isinstance(payload, Mapping) else None
    config = config if isinstance(config, Mapping) else {}
    assignment = config.get("type_assignment")
    assignment = assignment if isinstance(assignment, Mapping) else {}
    metadata = payload.get("metadata") if isinstance(payload, Mapping) else None
    metadata = metadata if isinstance(metadata, Mapping) else {}
    keys = (
        "rng_seed",
        "bootstrap_factor",
        "bootstrap_iteration",
        "n_processors",
        "n_runners_up",
        "chunk_size",
        "normalization",
        "algorithm",
    )
    return {
        "type_assignment": {key: assignment.get(key) for key in keys},
        "drop_level": config.get("drop_level"),
        "nodes_to_drop": config.get("nodes_to_drop"),
        "ctm_version": metadata.get("version"),
        "n_unmapped_genes": payload.get("n_unmapped_genes")
        if isinstance(payload, Mapping)
        else None,
    }


def _check_effective_config(
    effective: Mapping[str, Any], params: MmcEngineParams, bundle: MmcBundle
) -> None:
    """Refuse a mapping whose recorded settings differ from the requested ones.

    Only settings the JSON records are compared (fake mappers in tests may
    record none).
    """
    recorded = effective.get("type_assignment") or {}
    expected = {
        "rng_seed": params.rng_seed,
        "bootstrap_factor": params.bootstrap_factor,
        "bootstrap_iteration": params.bootstrap_iteration,
        "n_processors": params.n_processors,
        "n_runners_up": params.n_runners_up,
        "normalization": params.normalization,
    }
    mismatched: dict[str, tuple[Any, Any]] = {
        key: (recorded.get(key), value)
        for key, value in expected.items()
        if recorded.get(key) is not None and recorded.get(key) != value
    }
    drop_level = effective.get("drop_level")
    if drop_level is not None and drop_level != bundle.drop_level:
        mismatched["drop_level"] = (drop_level, bundle.drop_level)
    if mismatched:
        raise MmcEngineError(
            "cell_type_mapper recorded other settings than requested "
            f"(recorded, requested): {mismatched}"
        )


# --------------------------------------------------------------------------
# Tidy parquet


def write_tidy_parquet(
    tidy: pd.DataFrame, path: Path | str, metadata: Mapping[str, Any]
) -> Path:
    """Write a tidy MMC table with its run metadata in the parquet schema.

    Args:
        tidy: The tidy table.
        path: Output ``.parquet``.
        metadata: JSON-serialisable run metadata (reference, build hash,
            fingerprint, engine parameters), stored under ``merxen_mmc``.

    Returns:
        The written path.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(coerce_tidy_dtypes(tidy), preserve_index=False)
    schema_metadata = dict(table.schema.metadata or {})
    schema_metadata[TIDY_METADATA_KEY] = json.dumps(
        {"tidy_schema_version": TIDY_SCHEMA_VERSION, **dict(metadata)},
        sort_keys=True,
    ).encode("utf-8")
    table = table.replace_schema_metadata(schema_metadata)
    temporary = output.with_name(output.name + ".partial")
    pq.write_table(table, temporary)
    temporary.replace(output)
    return output


def read_tidy_parquet(path: Path | str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read a tidy MMC table and its run metadata.

    Args:
        path: A ``<sid>_mmc_<run_id>.parquet``.

    Returns:
        ``(tidy, metadata)``.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    raw = (table.schema.metadata or {}).get(TIDY_METADATA_KEY)
    metadata: dict[str, Any] = json.loads(raw.decode("utf-8")) if raw else {}
    return table.to_pandas(), metadata


def level_frame(tidy: pd.DataFrame, level: str) -> pd.DataFrame:
    """Return one level of a tidy table, indexed by cell id.

    Args:
        tidy: The tidy table.
        level: A taxonomy level key (``level``) or its token (``level_name``).

    Returns:
        The level's rows, indexed by ``cell_id``.

    Raises:
        KeyError: If the table has no such level.
    """
    mask = (tidy["level"].astype(str) == level) | (
        tidy["level_name"].astype(str) == level
    )
    if not bool(mask.any()):
        raise KeyError(f"the MMC table has no level {level!r}")
    return tidy.loc[mask].set_index("cell_id")


@dataclass(frozen=True)
class AggregatedProbability:
    """Coarse-class probabilities of one level (``aggregate_parent_probability``).

    Attributes:
        classes: The assigned node's class per cell (``None`` if it has none).
        probability: Aggregated probability of that class (NaN without one).
        runner_up_class: Best other class per cell (``None`` when none).
        runner_up_probability: Its summed runner-up probability (0 if none).
    """

    classes: np.ndarray
    probability: np.ndarray
    runner_up_class: np.ndarray
    runner_up_probability: np.ndarray


def _is_missing_label(label: Any) -> bool:
    return label is None or (isinstance(label, float) and math.isnan(label))


def aggregate_parent_probability(
    frame: pd.DataFrame, group_of: Mapping[str, str | None]
) -> AggregatedProbability:
    """Aggregate a level's bootstrap probabilities to a coarser class.

    The coarse probability of the assigned node's class is its bootstrap
    probability plus that of the runner-ups in the same class (E1 / E2
    definition, capped at 1); the best other class is the one with the
    largest summed probability among the runner-ups.

    Args:
        frame: One level of a tidy table (``level_frame``).
        group_of: Node label to class; nodes mapped to ``None`` or missing
            count toward no class.

    Returns:
        The aggregated probabilities.
    """
    n_cells = len(frame)
    assigned = frame["assignment"].astype(object).to_numpy()
    classes = np.array(
        [
            None if _is_missing_label(label) else group_of.get(str(label))
            for label in assigned
        ],
        dtype=object,
    )
    probability = frame["bp"].to_numpy(dtype=np.float64).copy()
    other_totals: list[dict[str, float]] = [{} for _ in range(n_cells)]
    for rank in range(1, N_RUNNERS_UP + 1):
        labels = frame[runner_up_column(rank, "assignment")].astype(object).to_numpy()
        values = frame[runner_up_column(rank, "probability")].to_numpy(np.float64)
        for index in range(n_cells):
            label = labels[index]
            value = values[index]
            if _is_missing_label(label) or not math.isfinite(value):
                continue
            group = group_of.get(str(label))
            if group is None:
                continue
            if group == classes[index]:
                probability[index] += value
            else:
                totals = other_totals[index]
                totals[group] = totals.get(group, 0.0) + value
    probability = np.minimum(probability, 1.0)
    probability[np.array([group is None for group in classes], dtype=bool)] = np.nan
    runner_class = np.empty(n_cells, dtype=object)
    runner_probability = np.zeros(n_cells, dtype=np.float64)
    for index, totals in enumerate(other_totals):
        if totals:
            best = max(sorted(totals), key=lambda group: totals[group])
            runner_class[index] = best
            runner_probability[index] = min(totals[best], 1.0)
        else:
            runner_class[index] = None
    return AggregatedProbability(
        classes=classes,
        probability=probability,
        runner_up_class=runner_class,
        runner_up_probability=runner_probability,
    )


# --------------------------------------------------------------------------
# run_mmc


@dataclass(frozen=True)
class MmcRunResult:
    """What one ``run_mmc`` call produced.

    Attributes:
        parquet: The tidy parquet.
        n_cells: Mapped cells.
        n_query_genes: Query genes.
        levels: Levels in the tidy table.
        wall_s: Mapper wall time in seconds.
        peak_rss_gb: Largest process RSS of the mapper, when measured.
        command: The mapper command.
        ctm_version: ctm version recorded by the mapper (else installed).
        ctm_commit: ctm commit, when known.
        effective_config: Settings the extended JSON recorded.
        extended_json: The kept gzipped extended JSON, or ``None``.
        stdout_log: Mapper stdout.
        stderr_log: Mapper stderr.
        ctm_log: ctm's own log file.
    """

    parquet: Path
    n_cells: int
    n_query_genes: int
    levels: tuple[str, ...]
    wall_s: float
    peak_rss_gb: float | None
    command: list[str]
    ctm_version: str | None
    ctm_commit: str | None
    effective_config: dict[str, Any]
    extended_json: Path | None
    stdout_log: Path
    stderr_log: Path
    ctm_log: Path


def _query_shape(query_h5ad: Path) -> tuple[list[str], int]:
    import h5py

    try:
        from anndata.io import read_elem
    except ImportError:  # anndata < 0.11
        from anndata.experimental import read_elem
    with h5py.File(query_h5ad, "r") as handle:
        obs = handle["obs"]
        index_key = obs.attrs.get("_index", "_index")
        cells = [str(value) for value in read_elem(obs[index_key])]
        var = handle["var"]
        var_index_key = var.attrs.get("_index", "_index")
        n_genes = len(read_elem(var[var_index_key]))
    return cells, n_genes


def _tail(path: Path, n_lines: int = 40) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-n_lines:])


def _read_peak_rss_gb(rusage_path: Path) -> float | None:
    try:
        fields = rusage_path.read_text(encoding="utf-8").split()
        return round(float(fields[-2]) / 1024**2, 3)
    except (OSError, IndexError, ValueError):
        return None


def run_mmc(
    query_h5ad: Path | str,
    bundle: MmcBundle,
    params: MmcEngineParams,
    *,
    output_parquet: Path | str,
    work_dir: Path | str,
    expected_ctm_version: str,
    lookup_path: Path | None = None,
    nodes_to_drop: Sequence[tuple[str, str]] = (),
    keep_extended_json: bool = False,
    log_dir: Path | str | None = None,
    run_metadata: Mapping[str, Any] | None = None,
    measure_peak_rss: bool = True,
) -> MmcRunResult:
    """Map one query onto one bundle and write the tidy parquet.

    Args:
        query_h5ad: Query H5AD (``write_query_h5ad``).
        bundle: The reference bundle.
        params: Engine parameters.
        output_parquet: The tidy parquet to write.
        work_dir: Task-local scratch (extended JSON, ctm tmp); files this run
            writes there are removed.
        expected_ctm_version: ``AnnotationConfig.ctm_version``.
        lookup_path: Restricted lookup (``restrict_lookup``), if any.
        nodes_to_drop: Nodes removed from the tree (mouse pruned re-map, M6).
        keep_extended_json: Keep the extended JSON gzipped next to the
            parquet (``annotation_keep_extended_json``).
        log_dir: Where stdout / stderr / ctm logs go (default: next to the
            parquet, in ``logs/``).
        run_metadata: Extra JSON metadata stored in the parquet.
        measure_peak_rss: Wrap the mapper in ``/usr/bin/time`` when present.

    Returns:
        The run result.

    Raises:
        CtmVersionError: If the installed ctm is not ``expected_ctm_version``.
        MmcEngineError: If the mapper fails or records other settings.
        ExtendedJsonError: If its output does not match the query.
    """
    installed = check_ctm_version(expected_ctm_version)
    query = Path(query_h5ad)
    output = Path(output_parquet)
    scratch = Path(work_dir)
    scratch.mkdir(parents=True, exist_ok=True)
    logs = Path(log_dir) if log_dir is not None else output.parent / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stem = output.name.removesuffix(".parquet")
    extended_json = scratch / f"{stem}.extended.json"
    tmp_dir = scratch / f"{stem}.ctm_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = logs / f"{stem}.stdout.log"
    stderr_log = logs / f"{stem}.stderr.log"
    ctm_log = logs / f"{stem}.ctm.log"
    rusage_path = scratch / f"{stem}.rusage.txt"
    cells, n_genes = _query_shape(query)
    command = build_mmc_command(
        query,
        bundle,
        params,
        extended_json=extended_json,
        log_path=ctm_log,
        lookup_path=lookup_path,
        tmp_dir=tmp_dir,
        nodes_to_drop=nodes_to_drop,
    )
    launched = command
    if measure_peak_rss and GNU_TIME.is_file() and os.access(GNU_TIME, os.X_OK):
        launched = [str(GNU_TIME), "-f", "%M %e", "-o", str(rusage_path), *command]
    logger.info(
        "MapMyCells %s (%s) on %d cells x %d genes, %d processes",
        bundle.reference_id,
        bundle.build_hash[:12],
        len(cells),
        n_genes,
        params.n_processors,
    )
    start = time.monotonic()
    try:
        with stdout_log.open("w") as stdout, stderr_log.open("w") as stderr:
            stdout.write("$ " + " ".join(command) + "\n\n")
            stdout.flush()
            completed = subprocess.run(
                launched,
                check=False,
                stdout=stdout,
                stderr=stderr,
                env=mmc_environment(),
                text=True,
            )
    except FileNotFoundError as error:
        raise MmcEngineError(f"could not start MapMyCells: {error}") from error
    wall_s = round(time.monotonic() - start, 3)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    peak_rss_gb = _read_peak_rss_gb(rusage_path) if launched is not command else None
    rusage_path.unlink(missing_ok=True)
    if completed.returncode != 0:
        raise MmcEngineError(
            f"MapMyCells {bundle.reference_id} failed with exit code "
            f"{completed.returncode}; stderr tail ({stderr_log}):\n"
            f"{_tail(stderr_log)}"
        )
    if not extended_json.is_file():
        raise MmcEngineError(
            f"MapMyCells {bundle.reference_id} wrote no extended JSON ({extended_json})"
        )
    payload: Any = json.loads(extended_json.read_text(encoding="utf-8"))
    effective = extended_json_config(payload)
    _check_effective_config(effective, params, bundle)
    tree_names = bundle.tree().names if bundle.mapping_tree.is_file() else {}
    tidy = parse_extended_json_tidy(payload, cell_order=cells, names=tree_names)
    del payload
    ctm_version = effective.get("ctm_version") or installed
    metadata = {
        "reference_id": bundle.reference_id,
        "build_hash": bundle.build_hash,
        "panel_hash": bundle.panel_hash,
        "engine_params": params.reuse_key(),
        "ctm_version": ctm_version,
        "n_cells": len(cells),
        "n_query_genes": n_genes,
        **dict(run_metadata or {}),
    }
    write_tidy_parquet(tidy, output, metadata)
    kept: Path | None = None
    if keep_extended_json:
        kept = output.with_name(stem + EXTENDED_JSON_GZ_SUFFIX)
        with extended_json.open("rb") as raw, gzip.open(kept, "wb") as packed:
            shutil.copyfileobj(raw, packed)
    extended_json.unlink(missing_ok=True)
    levels = tuple(str(level) for level in tidy["level"].astype(str).unique())
    logger.info(
        "MapMyCells %s done in %.1f s (peak RSS %s GB) -> %s",
        bundle.reference_id,
        wall_s,
        peak_rss_gb,
        output,
    )
    return MmcRunResult(
        parquet=output,
        n_cells=len(cells),
        n_query_genes=n_genes,
        levels=levels,
        wall_s=wall_s,
        peak_rss_gb=peak_rss_gb,
        command=command,
        ctm_version=ctm_version,
        ctm_commit=ctm_commit(),
        effective_config=effective,
        extended_json=kept,
        stdout_log=stdout_log,
        stderr_log=stderr_log,
        ctm_log=ctm_log,
    )

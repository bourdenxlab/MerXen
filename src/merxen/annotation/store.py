"""Content-addressed store of annotation reference bundles (plan §3.2).

A bundle is everything MAP and RESOLVE need from one reference on one declared
panel (markers, profiles, negative genes, diagnostics, resolvability, trust
state). ``ReferenceStore.get_or_build(spec, panel)`` returns the bundle whose
``build_hash`` matches the request, building it when it does not exist yet.

Layout (one store root, plus an optional large-panel root, §8.7)::

    <store>/<reference_id>/.lock          flock target, one per reference
    <store>/<reference_id>/<build_hash>/  a complete bundle (bundle.json last)
    <store>/.tmp-<uuid>/                  a bundle being built
    <store>/.failed-<uuid>/               a build that raised, kept for review

Rules (plan §3.2, R1, R20, OD-D4):

* ``build_hash`` is the sha256 of a canonical JSON over everything that
  changes a bundle's content: the store schema version,
  ``ANNOTATION_BUILDER_VERSION`` (bumped only when builder logic changes,
  never the git commit or package version), the reference id, role and
  taxonomy, each source file's identity ``{path, size, mtime_ns,
  sha256(first + last 64 MB)}``, the panel hash, hierarchy, nodes to drop,
  drop level, marker and mapping settings, the large-panel prefilter, the
  depth grid, the resolvability recipe and the ``cell_type_mapper`` version
  and commit. The full sha256 of every source file is recorded in
  ``bundle.json``.
* A bundle is built in ``<store>/.tmp-<uuid>/`` on the same filesystem and
  renamed to ``<store>/<reference_id>/<build_hash>/`` in one ``rename(2)``
  while holding ``fcntl.flock`` on ``<store>/<reference_id>/.lock``, so
  concurrent callers (processes or Nextflow launches) build once and readers
  only ever see a missing or a complete bundle.
* Sources are copied into the bundle with a checksum, never symlinked; a
  bundle holding a symlink is refused.
* Nothing is ever deleted. A build that raises is renamed to
  ``.failed-<uuid>`` with its traceback; one that is killed leaves its
  ``.tmp-<uuid>``. ``list`` shows both and ``prune`` only lists candidates.

This module imports only the standard library and pydantic (plus the light
annotation contract modules), so the CLI stays fast; builders import
``cell_type_mapper`` lazily.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import logging
import os
import shutil
import socket
import tempfile
import time
import traceback
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

from pydantic import BaseModel, ConfigDict

from merxen.annotation.config import (
    LARGE_PANEL_GENES,
    AnnotationConfig,
    AnnotationReferenceSpec,
)
from merxen.annotation.schema import PanelTrust, ReferenceRole

if TYPE_CHECKING:
    from merxen.annotation.panel import AnnotationPanel

logger = logging.getLogger(__name__)

# Bumped only when a builder's logic changes what a bundle contains, so every
# bundle built by the old logic gets a new build_hash (plan §3.2, E-M7).
ANNOTATION_BUILDER_VERSION: Final = 1
# Version of the build-hash payload and of bundle.json.
STORE_SCHEMA_VERSION: Final = 1
# Simulation recipes of the resolvability self-map and their versions (§8.3).
RESOLVABILITY_RECIPE_VERSIONS: Final[dict[str, int]] = {"R1_contam_HO": 1}

HEAD_TAIL_BYTES: Final = 64 * 1024 * 1024
_READ_CHUNK_BYTES: Final = 8 * 1024 * 1024
BUNDLE_MANIFEST_NAME: Final = "bundle.json"
BUNDLE_REF_NAME: Final = "bundle_ref.json"
BUILD_MARKER_NAME: Final = ".building.json"
BUILD_ERROR_NAME: Final = "build_error.txt"
LOCK_FILE_NAME: Final = ".lock"
TMP_DIR_PREFIX: Final = ".tmp-"
FAILED_DIR_PREFIX: Final = ".failed-"
BUNDLE_STATUS_COMPLETE: Final = "complete"
# References whose bundle does not depend on a panel (mouse region shares).
PANEL_INDEPENDENT_ROLES: Final[frozenset[str]] = frozenset({"region_share"})
# Result files whose JSON records the bundles a run used (prune --unreferenced-by).
REFERENCE_RECORD_NAMES: Final[frozenset[str]] = frozenset(
    {BUNDLE_REF_NAME, "map_manifest.json", "required_bundles.json"}
)
REFERENCE_RECORD_SUFFIXES: Final[tuple[str, ...]] = ("_annotation_manifest.json",)
# Directories never searched for reference records (zarr stores hold millions
# of chunk files; Nextflow work dirs are not results).
_SKIPPED_SCAN_DIR_SUFFIXES: Final[tuple[str, ...]] = (".zarr",)
_SKIPPED_SCAN_DIR_NAMES: Final[frozenset[str]] = frozenset({"work", ".nextflow"})
_SHA256_HEX_LENGTH: Final = 64


class StoreError(RuntimeError):
    """Base class of reference-store errors."""


class BundleIntegrityError(StoreError):
    """A bundle directory exists but is not a valid, complete bundle."""


class UnknownBuilderError(StoreError):
    """No bundle builder is registered for a reference id."""


class PruneRefusedError(StoreError):
    """``prune`` was asked to delete; deletion is always manual (OD-D4)."""


# --------------------------------------------------------------------------
# Hashing and file identities


def canonical_json(payload: Any) -> str:
    """Return the canonical JSON text used for hashing.

    Args:
        payload: JSON-serialisable data (dicts, lists, strings, numbers).

    Returns:
        Compact JSON with sorted keys, ASCII only, and no NaN or infinity.
    """
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def sha256_hex(text: str) -> str:
    """Return the sha256 hex digest of a UTF-8 string.

    Args:
        text: Text to hash.

    Returns:
        64 lower-case hex characters.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path | str) -> str:
    """Return the sha256 hex digest of a whole file.

    Args:
        path: File to read.

    Returns:
        64 lower-case hex characters.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(_READ_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def head_tail_sha256(path: Path | str, block_bytes: int = HEAD_TAIL_BYTES) -> str:
    """Return the sha256 of a file's first and last ``block_bytes``.

    Files no larger than two blocks are hashed whole, so the digest equals
    ``file_sha256`` for them. Larger files hash the first block followed by
    the last block; the size, which ``FileIdentity`` records beside this
    digest, covers the part in between.

    Args:
        path: File to read.
        block_bytes: Bytes read from each end (64 MB, plan §3.2).

    Returns:
        64 lower-case hex characters.
    """
    file_path = Path(path)
    size = file_path.stat().st_size
    if size <= 2 * block_bytes:
        return file_sha256(file_path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for offset in (0, size - block_bytes):
            handle.seek(offset)
            remaining = block_bytes
            while remaining > 0:
                chunk = handle.read(min(_READ_CHUNK_BYTES, remaining))
                if not chunk:
                    break
                digest.update(chunk)
                remaining -= len(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class FileIdentity:
    """Identity of one source file (plan §3.2 ``build_hash``).

    Attributes:
        path: Absolute path (symlinks resolved).
        size: Size in bytes.
        mtime_ns: Modification time in nanoseconds.
        sha256_head_tail: sha256 of the first + last 64 MB.
    """

    path: str
    size: int
    mtime_ns: int
    sha256_head_tail: str

    def hash_payload(self) -> dict[str, Any]:
        """Return the identity as it enters ``build_hash``."""
        return {
            "path": self.path,
            "size": self.size,
            "mtime_ns": self.mtime_ns,
            "sha256_head_tail": self.sha256_head_tail,
        }


def file_identity(path: Path | str) -> FileIdentity:
    """Return the identity of a source file.

    Args:
        path: Existing file.

    Returns:
        Its resolved path, size, mtime and head/tail digest.

    Raises:
        FileNotFoundError: If ``path`` is not an existing file.
    """
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"reference source {path} is not a file")
    stat = resolved.stat()
    return FileIdentity(
        path=str(resolved),
        size=int(stat.st_size),
        mtime_ns=int(stat.st_mtime_ns),
        sha256_head_tail=head_tail_sha256(resolved),
    )


@dataclass(frozen=True)
class SourceRecord:
    """Identity of one named reference source: a file or a directory of files.

    Attributes:
        name: Source name from ``AnnotationReferenceSpec.sources``.
        path: Absolute path (symlinks resolved).
        kind: ``"file"`` or ``"directory"``.
        files: Identities of the file, or of every file under the directory
            in sorted relative-path order (hidden entries skipped).
    """

    name: str
    path: str
    kind: Literal["file", "directory"]
    files: tuple[FileIdentity, ...]

    def hash_payload(self) -> dict[str, Any]:
        """Return the record as it enters ``build_hash``."""
        return {
            "kind": self.kind,
            "path": self.path,
            "files": [identity.hash_payload() for identity in self.files],
        }


def source_record(name: str, path: Path | str) -> SourceRecord:
    """Return the identity of a named source file or directory.

    Args:
        name: Source name.
        path: Existing file or directory.

    Returns:
        The source record.

    Raises:
        FileNotFoundError: If ``path`` does not exist, or a directory holds no
            file.
    """
    resolved = Path(path).resolve()
    if resolved.is_file():
        return SourceRecord(
            name=name,
            path=str(resolved),
            kind="file",
            files=(file_identity(resolved),),
        )
    if not resolved.is_dir():
        raise FileNotFoundError(f"reference source {name}={path} does not exist")
    files: list[FileIdentity] = []
    for directory, dir_names, file_names in os.walk(resolved):
        dir_names[:] = sorted(name for name in dir_names if not name.startswith("."))
        for file_name in sorted(file_names):
            if file_name.startswith("."):
                continue
            file_path = Path(directory) / file_name
            if file_path.is_file():
                files.append(file_identity(file_path))
    if not files:
        raise FileNotFoundError(f"reference source directory {name}={path} is empty")
    files.sort(key=lambda identity: identity.path)
    return SourceRecord(
        name=name, path=str(resolved), kind="directory", files=tuple(files)
    )


def ctm_provenance() -> dict[str, str | None]:
    """Return the installed ``cell_type_mapper`` version and VCS commit.

    Reads package metadata only, so ``cell_type_mapper`` is not imported.

    Returns:
        ``{"version": ..., "commit": ...}``; ``None`` values when the package
        or its ``direct_url.json`` is missing.
    """
    try:
        distribution = importlib.metadata.distribution("cell_type_mapper")
    except importlib.metadata.PackageNotFoundError:
        return {"version": None, "commit": None}
    commit: str | None = None
    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text:
        try:
            direct_url: object = json.loads(direct_url_text)
        except ValueError:
            direct_url = None
        vcs_info = direct_url.get("vcs_info") if isinstance(direct_url, dict) else None
        if isinstance(vcs_info, dict) and vcs_info.get("commit_id"):
            commit = str(vcs_info["commit_id"])
    return {"version": distribution.version, "commit": commit}


# --------------------------------------------------------------------------
# Builders


@dataclass(frozen=True)
class CopiedFile:
    """A source file copied into a bundle.

    Attributes:
        source_path: Absolute path of the source.
        bundle_path: Path relative to the bundle directory.
        size: Size in bytes.
        sha256: Full sha256 of the copy, equal to the source's.
    """

    source_path: str
    bundle_path: str
    size: int
    sha256: str


@dataclass
class BuildContext:
    """What a builder gets from the store for one build.

    Attributes:
        spec: The reference spec.
        panel: The declared panel, or ``None`` for panel-independent
            references.
        build_hash: The bundle's ``build_hash``.
        work_dir: The ``.tmp-<uuid>`` directory; every bundle file goes here.
        final_dir: Where the bundle will live after the rename.
        scratch_dir: Task-local scratch outside the store, removed after the
            build (ctm ``tmp_dir``, reference markers never kept).
        sources: Source records by name.
        config: The annotation config, when the caller has one.
        copied: Files copied with ``copy_source`` so far.
    """

    spec: AnnotationReferenceSpec
    panel: AnnotationPanel | None
    build_hash: str
    work_dir: Path
    final_dir: Path
    scratch_dir: Path
    sources: dict[str, SourceRecord]
    config: AnnotationConfig | None = None
    copied: list[CopiedFile] = field(default_factory=list)

    def copy_source(self, source: str | Path, bundle_path: str | Path) -> CopiedFile:
        """Copy a source file into the bundle and verify its checksum.

        The copy is a regular file, never a symlink, so bundles never depend
        on a path other code may delete (R1, R-eng M7).

        Args:
            source: A source name (``spec.sources`` key) naming a file, or a
                file path.
            bundle_path: Destination relative to the bundle directory.

        Returns:
            The copy's record (also appended to ``copied``).

        Raises:
            FileNotFoundError: If the source is not a file.
            ValueError: If ``bundle_path`` leaves the bundle directory.
            BundleIntegrityError: If the copy's checksum differs from the
                source's.
        """
        if isinstance(source, str) and source in self.sources:
            record = self.sources[source]
            if record.kind != "file":
                raise FileNotFoundError(
                    f"source {source!r} is a directory; copy its files by path"
                )
            source_path = Path(record.path)
        else:
            source_path = Path(source).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"cannot copy {source}: not a file")
        destination = _inside(self.work_dir, bundle_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        source_digest = hashlib.sha256()
        with source_path.open("rb") as reader, destination.open("xb") as writer:
            while chunk := reader.read(_READ_CHUNK_BYTES):
                source_digest.update(chunk)
                writer.write(chunk)
            writer.flush()
            os.fsync(writer.fileno())
        copy_digest = file_sha256(destination)
        if copy_digest != source_digest.hexdigest():
            raise BundleIntegrityError(
                f"checksum of the copy of {source_path} differs from the source"
            )
        copied = CopiedFile(
            source_path=str(source_path),
            bundle_path=str(destination.relative_to(self.work_dir)),
            size=destination.stat().st_size,
            sha256=copy_digest,
        )
        self.copied.append(copied)
        return copied


BundleBuildFunction = Callable[[BuildContext], Mapping[str, Any] | None]


@dataclass(frozen=True)
class BundleBuilder:
    """How to build the bundle of one reference.

    Attributes:
        name: Builder name (recorded, and part of ``build_hash``).
        build: Writes the bundle files into ``context.work_dir`` and returns
            JSON-serialisable metadata for ``bundle.json`` (levels, node
            counts, markers per parent, collapsed parents, diagnostics). A
            ``"panel_trust"`` key becomes the bundle's trust state.
        taxonomy_id: Allen taxonomy id, e.g. ``"CCN202210140"``.
        params: Further content-affecting builder parameters (hashed).
        uses_panel: Whether the bundle depends on the panel; ``False`` for
            panel-independent references such as the mouse region shares.
    """

    name: str
    build: BundleBuildFunction
    taxonomy_id: str | None = None
    params: Mapping[str, Any] = field(default_factory=dict)
    uses_panel: bool = True


BuilderFactory = Callable[
    [AnnotationReferenceSpec, AnnotationConfig | None], BundleBuilder
]
_BUILDER_FACTORIES: dict[str, BuilderFactory] = {}
# The builders (plan §3.2 table) live in this module and register on import.
BUILDERS_MODULE: Final = "merxen.annotation.reference"


def register_builder(reference_id: str, factory: BuilderFactory) -> None:
    """Register the builder factory of a reference id.

    Args:
        reference_id: Store id, e.g. ``"whb_frontal_supc_clus"``.
        factory: Returns the builder for a spec and config.
    """
    _BUILDER_FACTORIES[reference_id] = factory


def resolve_builder(
    spec: AnnotationReferenceSpec, config: AnnotationConfig | None = None
) -> BundleBuilder:
    """Return the registered builder of a reference spec.

    Imports ``merxen.annotation.reference`` first (when it exists), whose
    builders register themselves on import.

    Args:
        spec: The reference spec.
        config: The annotation config, if any.

    Returns:
        The builder.

    Raises:
        UnknownBuilderError: If no builder is registered for the reference.
    """
    if (
        spec.reference_id not in _BUILDER_FACTORIES
        and importlib.util.find_spec(BUILDERS_MODULE) is not None
    ):
        importlib.import_module(BUILDERS_MODULE)
    factory = _BUILDER_FACTORIES.get(spec.reference_id)
    if factory is None:
        raise UnknownBuilderError(
            f"no bundle builder is registered for reference {spec.reference_id!r}; "
            f"known: {sorted(_BUILDER_FACTORIES)}"
        )
    return factory(spec, config)


# --------------------------------------------------------------------------
# Bundle records


class _StoreModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BundleRef(_StoreModel):
    """What ``ANNOTATE_REFERENCE_PREP`` emits for one bundle (``bundle_ref.json``).

    Attributes:
        reference_id: Store id.
        species: ``"human"`` or ``"mouse"``.
        role: Reference role.
        panel_hash: Declared-panel hash, ``None`` for panel-independent
            references.
        build_hash: The bundle's ``build_hash``.
        path: Bundle directory.
        store_root: Store root the bundle lives in.
        panel_trust: Trust state recorded by the builder (M3b), if any.
        reused: Whether the bundle existed before this call.
    """

    reference_id: str
    species: str
    role: ReferenceRole
    panel_hash: str | None
    build_hash: str
    path: str
    store_root: str
    panel_trust: PanelTrust | None = None
    reused: bool

    def write(self, path: Path | str) -> Path:
        """Write the reference as JSON.

        Args:
            path: Output file (usually ``bundle_ref.json``).

        Returns:
            The written path.
        """
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.model_dump(mode="json"), indent=2) + "\n")
        return output


@dataclass(frozen=True)
class BuildRequest:
    """Everything ``build_hash`` covers for one (reference, panel) request.

    Attributes:
        payload: The canonical build-hash payload.
        build_hash: sha256 of ``canonical_json(payload)``.
        sources: Source records by name.
    """

    payload: dict[str, Any]
    build_hash: str
    sources: dict[str, SourceRecord]


def build_hash_payload(
    spec: AnnotationReferenceSpec,
    panel: AnnotationPanel | None,
    *,
    builder: BundleBuilder,
    sources: Mapping[str, SourceRecord],
    config: AnnotationConfig | None = None,
    ctm: Mapping[str, str | None] | None = None,
    large_panel_genes: int = LARGE_PANEL_GENES,
) -> dict[str, Any]:
    """Return the build-hash payload of a request.

    Args:
        spec: The reference spec.
        panel: The declared panel; ignored when the builder does not use one.
        builder: The builder (name, taxonomy, params, panel use).
        sources: Source records by name.
        config: The annotation config; supplies the large-panel prefilter and
            the resolvability recipe.
        ctm: ``cell_type_mapper`` provenance; defaults to ``ctm_provenance()``.
        large_panel_genes: Panels above this size use the prefilter.

    Returns:
        The payload, with sorted, JSON-native values only.

    Raises:
        ValueError: If a panel-dependent builder gets no panel, or the panel's
            species differs from the spec's.
    """
    panel_payload: dict[str, Any] | None = None
    n_panel_genes: int | None = None
    if builder.uses_panel:
        if panel is None:
            raise ValueError(
                f"reference {spec.reference_id!r} needs a panel to build its bundle"
            )
        if panel.species != spec.species:
            raise ValueError(
                f"panel species {panel.species!r} differs from reference "
                f"{spec.reference_id!r} species {spec.species!r}"
            )
        n_panel_genes = panel.n_genes
        panel_payload = {
            "panel_hash": panel.panel_hash,
            "n_genes": panel.n_genes,
            "symbols_sha256": panel.symbols_sha256(),
        }
    prefilter: dict[str, Any] | None = None
    if (
        config is not None
        and n_panel_genes is not None
        and n_panel_genes > large_panel_genes
        and config.panel.large_panel_marker_prefilter != "none"
    ):
        prefilter = {
            "method": config.panel.large_panel_marker_prefilter,
            "cap": config.panel.large_panel_prefilter_cap,
        }
    resolvability: dict[str, Any] | None = None
    if config is not None and spec.role == "primary" and config.resolvability.enabled:
        recipe = config.resolvability.recipe
        resolvability = {
            "recipe": recipe,
            "recipe_version": RESOLVABILITY_RECIPE_VERSIONS.get(recipe),
            "settings": config.resolvability.model_dump(
                mode="json",
                exclude={
                    name
                    for name in type(config.resolvability).model_fields
                    if name.startswith("gate_p_")
                },
            ),
        }
    return {
        "schema_version": STORE_SCHEMA_VERSION,
        "builder_version": ANNOTATION_BUILDER_VERSION,
        "builder": builder.name,
        "builder_params": _json_native(dict(builder.params)),
        "reference_id": spec.reference_id,
        "species": spec.species,
        "role": spec.role,
        "taxonomy_id": builder.taxonomy_id,
        "sources": {
            name: record.hash_payload() for name, record in sorted(sources.items())
        },
        "panel": panel_payload,
        "hierarchy": list(spec.hierarchy),
        "nodes_to_drop": sorted(spec.nodes_to_drop),
        "drop_level": spec.drop_level,
        "n_per_utility": spec.n_per_utility,
        "max_cells_per_cluster": spec.max_cells_per_cluster,
        "mapping": {
            "bootstrap_factor": spec.bootstrap_factor,
            "bootstrap_iteration": spec.bootstrap_iteration,
            "rng_seed": spec.rng_seed,
        },
        "large_panel_prefilter": prefilter,
        "depth_grid": spec.resolved_depth_grid(n_panel_genes),
        "resolvability": resolvability,
        "ctm": dict(ctm if ctm is not None else ctm_provenance()),
    }


def compute_build_hash(payload: Mapping[str, Any]) -> str:
    """Return ``build_hash`` = sha256 of the canonical JSON of a payload.

    Args:
        payload: Output of ``build_hash_payload``.

    Returns:
        64 lower-case hex characters.
    """
    return sha256_hex(canonical_json(payload))


@dataclass(frozen=True)
class StoreEntry:
    """One directory found by ``ReferenceStore.list``.

    Attributes:
        kind: ``"bundle"`` (complete), ``"invalid"`` (a build-hash directory
            without a valid ``bundle.json``), ``"tmp"`` (a build in progress
            or killed) or ``"failed"`` (a build that raised).
        path: The directory.
        store_root: The store root it belongs to.
        reference_id: Reference id, when known.
        build_hash: Build hash, when known.
        panel_hash: Panel hash of the bundle, when known.
        n_panel_genes: Panel size, when known.
        created_at: ISO time of completion (bundles) or start (tmp, failed).
        size_bytes: Sum of the file sizes.
        builder_pid: Builder process id (tmp, failed).
        builder_host: Builder host name (tmp, failed).
        builder_alive: Whether that process still runs on this host (tmp).
    """

    kind: Literal["bundle", "invalid", "tmp", "failed"]
    path: Path
    store_root: Path
    reference_id: str | None = None
    build_hash: str | None = None
    panel_hash: str | None = None
    n_panel_genes: int | None = None
    created_at: str | None = None
    size_bytes: int = 0
    builder_pid: int | None = None
    builder_host: str | None = None
    builder_alive: bool | None = None

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return {
            "kind": self.kind,
            "path": str(self.path),
            "store_root": str(self.store_root),
            "reference_id": self.reference_id,
            "build_hash": self.build_hash,
            "panel_hash": self.panel_hash,
            "n_panel_genes": self.n_panel_genes,
            "created_at": self.created_at,
            "size_bytes": self.size_bytes,
            "builder_pid": self.builder_pid,
            "builder_host": self.builder_host,
            "builder_alive": self.builder_alive,
        }


@dataclass(frozen=True)
class PruneReport:
    """What ``prune(dry_run=True)`` found; nothing is deleted.

    Attributes:
        results_root: The results tree that was searched for references.
        referenced: Bundles some result file references.
        candidates: Bundles no result references, plus failed and dead
            temporary builds: what an operator may delete by hand.
        in_progress: Temporary builds whose builder still runs.
        n_record_files: Result files read for references.
        referenced_build_hashes: Every build hash found in them.
    """

    results_root: Path
    referenced: list[StoreEntry]
    candidates: list[StoreEntry]
    in_progress: list[StoreEntry]
    n_record_files: int
    referenced_build_hashes: frozenset[str]

    @property
    def candidate_bytes(self) -> int:
        """Return the total size of the candidates in bytes."""
        return sum(entry.size_bytes for entry in self.candidates)

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return {
            "dry_run": True,
            "results_root": str(self.results_root),
            "n_record_files": self.n_record_files,
            "n_referenced": len(self.referenced),
            "n_candidates": len(self.candidates),
            "candidate_bytes": self.candidate_bytes,
            "referenced": [entry.to_json() for entry in self.referenced],
            "candidates": [entry.to_json() for entry in self.candidates],
            "in_progress": [entry.to_json() for entry in self.in_progress],
        }


# --------------------------------------------------------------------------
# The store


class ReferenceStore:
    """Content-addressed, append-only store of reference bundles.

    Args:
        root: Store root (``annotation_reference_store``).
        large_root: Store for panels above ``large_panel_genes``
            (``annotation_reference_store_large``); ``None`` uses ``root``.
        large_panel_genes: Panel size above which bundles go to
            ``large_root``.
        scratch_root: Parent of the per-build scratch directories (outside
            the store); ``None`` uses the system temporary directory.

    Raises:
        ValueError: If ``scratch_root`` lies inside a store root: scratch
            directories are removed after each build, and nothing inside a
            store is ever removed.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        large_root: Path | str | None = None,
        large_panel_genes: int = LARGE_PANEL_GENES,
        scratch_root: Path | str | None = None,
    ) -> None:
        self.root = Path(root)
        self.large_root = Path(large_root) if large_root is not None else None
        self.large_panel_genes = int(large_panel_genes)
        self.scratch_root = Path(scratch_root) if scratch_root is not None else None
        if self.scratch_root is not None:
            scratch = self.scratch_root.resolve()
            for store_root in self.roots:
                resolved_root = store_root.resolve()
                if scratch == resolved_root or resolved_root in scratch.parents:
                    raise ValueError(
                        f"scratch_root {self.scratch_root} lies inside the "
                        f"reference store {store_root}"
                    )

    @property
    def roots(self) -> list[Path]:
        """Return the distinct store roots, the main root first."""
        roots = [self.root]
        if self.large_root is not None and self.large_root != self.root:
            roots.append(self.large_root)
        return roots

    def root_for(self, panel: AnnotationPanel | None) -> Path:
        """Return the store root of a panel's bundles.

        Args:
            panel: The declared panel, or ``None``.

        Returns:
            ``large_root`` for panels above ``large_panel_genes`` (when
            configured), else ``root``.
        """
        if (
            panel is not None
            and self.large_root is not None
            and panel.n_genes > self.large_panel_genes
        ):
            return self.large_root
        return self.root

    def prepare_request(
        self,
        spec: AnnotationReferenceSpec,
        panel: AnnotationPanel | None,
        *,
        builder: BundleBuilder,
        config: AnnotationConfig | None = None,
    ) -> BuildRequest:
        """Identify the sources and compute the ``build_hash`` of a request.

        Args:
            spec: The reference spec.
            panel: The declared panel.
            builder: The builder.
            config: The annotation config, if any.

        Returns:
            The request.
        """
        # A missing config means the species defaults, so a caller that
        # passes none and one that passes the defaults share one build_hash.
        config = config or AnnotationConfig(species=spec.species)
        sources = {
            name: source_record(name, path)
            for name, path in sorted(spec.sources.items())
        }
        payload = build_hash_payload(
            spec,
            panel if builder.uses_panel else None,
            builder=builder,
            sources=sources,
            config=config,
            large_panel_genes=self.large_panel_genes,
        )
        return BuildRequest(
            payload=payload,
            build_hash=compute_build_hash(payload),
            sources=sources,
        )

    def bundle_dir(
        self,
        reference_id: str,
        build_hash: str,
        panel: AnnotationPanel | None = None,
    ) -> Path:
        """Return where a bundle lives (whether or not it exists).

        Args:
            reference_id: Store id.
            build_hash: The bundle's ``build_hash``.
            panel: The declared panel (selects the large store).

        Returns:
            ``<root>/<reference_id>/<build_hash>``.
        """
        return self.root_for(panel) / reference_id / build_hash

    def find(
        self,
        spec: AnnotationReferenceSpec,
        panel: AnnotationPanel | None,
        *,
        builder: BundleBuilder | None = None,
        config: AnnotationConfig | None = None,
    ) -> BundleRef | None:
        """Return the existing bundle of a request, without building.

        Args:
            spec: The reference spec.
            panel: The declared panel.
            builder: The builder; ``None`` resolves the registered one.
            config: The annotation config, if any.

        Returns:
            The bundle reference, or ``None`` when it does not exist.
        """
        config = config or AnnotationConfig(species=spec.species)
        builder = builder or resolve_builder(spec, config)
        request = self.prepare_request(spec, panel, builder=builder, config=config)
        effective_panel = panel if builder.uses_panel else None
        final_dir = self.bundle_dir(
            spec.reference_id, request.build_hash, effective_panel
        )
        manifest = self._read_complete_bundle(final_dir, request.build_hash)
        if manifest is None:
            return None
        return self._bundle_ref(spec, effective_panel, final_dir, manifest, reused=True)

    def get_or_build(
        self,
        spec: AnnotationReferenceSpec,
        panel: AnnotationPanel | None,
        *,
        builder: BundleBuilder | None = None,
        config: AnnotationConfig | None = None,
    ) -> BundleRef:
        """Return the bundle of a reference on a panel, building it if needed.

        Concurrent calls for the same bundle, from threads, processes or
        separate Nextflow launches, build it once: the build runs under an
        exclusive ``flock`` on ``<store>/<reference_id>/.lock`` and every
        waiting caller re-checks for the bundle after taking the lock.

        Args:
            spec: The reference spec (``AnnotationReferenceSpec``).
            panel: The declared panel (``panel_genes.json``); ``None`` only
                for panel-independent references.
            builder: The builder; ``None`` resolves the registered one.
            config: The annotation config, if any.

        Returns:
            The bundle reference (``bundle_ref.json`` content).

        Raises:
            BundleIntegrityError: If the bundle directory exists but is not a
                valid bundle (it is never replaced or deleted), or the build
                wrote a symlink or a source changed during the build.
            UnknownBuilderError: If no builder is registered.
        """
        config = config or AnnotationConfig(species=spec.species)
        builder = builder or resolve_builder(spec, config)
        request = self.prepare_request(spec, panel, builder=builder, config=config)
        effective_panel = panel if builder.uses_panel else None
        root = self.root_for(effective_panel)
        final_dir = self.bundle_dir(
            spec.reference_id, request.build_hash, effective_panel
        )
        manifest = self._read_complete_bundle(final_dir, request.build_hash)
        if manifest is not None:
            logger.info("Reusing reference bundle %s", final_dir)
            return self._bundle_ref(
                spec, effective_panel, final_dir, manifest, reused=True
            )
        reference_dir = root / spec.reference_id
        reference_dir.mkdir(parents=True, exist_ok=True)
        with _exclusive_lock(reference_dir / LOCK_FILE_NAME):
            manifest = self._read_complete_bundle(final_dir, request.build_hash)
            if manifest is not None:
                logger.info("Reusing reference bundle %s (built meanwhile)", final_dir)
                return self._bundle_ref(
                    spec, effective_panel, final_dir, manifest, reused=True
                )
            manifest = self._build(
                spec,
                effective_panel,
                builder=builder,
                request=request,
                root=root,
                final_dir=final_dir,
                config=config,
            )
        return self._bundle_ref(
            spec, effective_panel, final_dir, manifest, reused=False
        )

    def _build(
        self,
        spec: AnnotationReferenceSpec,
        panel: AnnotationPanel | None,
        *,
        builder: BundleBuilder,
        request: BuildRequest,
        root: Path,
        final_dir: Path,
        config: AnnotationConfig | None,
    ) -> dict[str, Any]:
        """Build a bundle in a temporary directory and rename it into place.

        Call only while holding the reference lock.
        """
        build_id = uuid.uuid4().hex
        work_dir = root / f"{TMP_DIR_PREFIX}{build_id}"
        work_dir.mkdir()
        started_at = _now()
        marker = {
            "reference_id": spec.reference_id,
            "build_hash": request.build_hash,
            "panel_hash": None if panel is None else panel.panel_hash,
            "n_panel_genes": None if panel is None else panel.n_genes,
            "final_dir": str(final_dir),
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "started_at": started_at,
        }
        (work_dir / BUILD_MARKER_NAME).write_text(json.dumps(marker, indent=2) + "\n")
        logger.info(
            "Building reference bundle %s for %s in %s",
            request.build_hash,
            spec.reference_id,
            work_dir,
        )
        scratch_dir = Path(
            tempfile.mkdtemp(
                prefix=f"merxen-bundle-{spec.reference_id}-",
                dir=None if self.scratch_root is None else str(self.scratch_root),
            )
        )
        start = time.monotonic()
        try:
            context = BuildContext(
                spec=spec,
                panel=panel,
                build_hash=request.build_hash,
                work_dir=work_dir,
                final_dir=final_dir,
                scratch_dir=scratch_dir,
                sources=request.sources,
                config=config,
            )
            builder_output = dict(builder.build(context) or {})
            _check_sources_unchanged(request.sources)
            manifest = _bundle_manifest(
                spec,
                panel,
                builder=builder,
                request=request,
                context=context,
                builder_output=builder_output,
                started_at=started_at,
                wall_time_s=time.monotonic() - start,
            )
            manifest_path = work_dir / BUNDLE_MANIFEST_NAME
            with manifest_path.open("x", encoding="utf-8") as handle:
                handle.write(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            _fsync_directory(work_dir)
            # rename(2) is atomic within one filesystem, and <root>/.tmp-* and
            # <root>/<reference_id>/ share it. The lock guarantees final_dir
            # was absent when checked; rename fails rather than merge if not.
            os.rename(work_dir, final_dir)
            _fsync_directory(final_dir.parent)
        except BaseException as error:
            _mark_failed(work_dir, root, build_id, error)
            raise
        finally:
            # The scratch directory is outside the store and private to this
            # build; nothing in it is ever part of a bundle.
            shutil.rmtree(scratch_dir, ignore_errors=True)
        logger.info(
            "Wrote reference bundle %s (%.1f s)", final_dir, time.monotonic() - start
        )
        return manifest

    def _read_complete_bundle(
        self, bundle_dir: Path, build_hash: str
    ) -> dict[str, Any] | None:
        """Return the manifest of a complete bundle, or ``None`` when absent.

        Raises:
            BundleIntegrityError: If the directory exists but its manifest is
                missing, unreadable, for another build or lists a missing file.
        """
        if not bundle_dir.exists():
            return None
        manifest = _read_manifest(bundle_dir / BUNDLE_MANIFEST_NAME)
        problems: list[str] = []
        if manifest is None:
            problems.append("bundle.json is missing or unreadable")
        else:
            if manifest.get("build_hash") != build_hash:
                problems.append(
                    f"bundle.json records build_hash {manifest.get('build_hash')!r}"
                )
            if manifest.get("status") != BUNDLE_STATUS_COMPLETE:
                problems.append(f"status is {manifest.get('status')!r}")
            for record in manifest.get("files", []):
                file_path = bundle_dir / str(record.get("path", ""))
                if not file_path.is_file() or file_path.is_symlink():
                    problems.append(f"missing file {record.get('path')!r}")
                elif file_path.stat().st_size != record.get("size"):
                    problems.append(f"size of {record.get('path')!r} changed")
        if problems:
            raise BundleIntegrityError(
                f"reference bundle {bundle_dir} is not a valid complete bundle "
                f"({'; '.join(problems)}); it is left untouched. Inspect it and "
                "move it aside by hand to rebuild."
            )
        return manifest

    def _bundle_ref(
        self,
        spec: AnnotationReferenceSpec,
        panel: AnnotationPanel | None,
        bundle_dir: Path,
        manifest: Mapping[str, Any],
        *,
        reused: bool,
    ) -> BundleRef:
        trust = manifest.get("panel_trust")
        return BundleRef(
            reference_id=spec.reference_id,
            species=spec.species,
            role=spec.role,
            panel_hash=None if panel is None else panel.panel_hash,
            build_hash=str(manifest["build_hash"]),
            path=str(bundle_dir),
            store_root=str(bundle_dir.parent.parent),
            panel_trust=trust,
            reused=reused,
        )

    def list(self) -> list[StoreEntry]:
        """List every bundle and build directory in the store roots.

        Returns:
            Complete bundles, invalid build-hash directories, temporary
            builds and failed builds, sorted by root, reference and hash.
            Nothing is modified.
        """
        entries: list[StoreEntry] = []
        for root in self.roots:
            if not root.is_dir():
                continue
            for child in sorted(root.iterdir()):
                if child.is_symlink() or not child.is_dir():
                    continue
                if child.name.startswith(TMP_DIR_PREFIX):
                    entries.append(_build_dir_entry(child, root, "tmp"))
                elif child.name.startswith(FAILED_DIR_PREFIX):
                    entries.append(_build_dir_entry(child, root, "failed"))
                elif not child.name.startswith("."):
                    entries.extend(_reference_entries(child, root))
        return entries

    def prune(
        self, *, unreferenced_by: Path | str, dry_run: bool = True
    ) -> PruneReport:
        """List the bundles no result under ``unreferenced_by`` refers to.

        A bundle is referenced when its ``build_hash`` appears in any
        ``bundle_ref.json``, ``map_manifest.json``, ``required_bundles.json``
        or ``*_annotation_manifest.json`` below the results root. Nothing is
        ever deleted (OD-D4): the report lists what an operator may delete by
        hand.

        Args:
            unreferenced_by: Results root to search.
            dry_run: Must be ``True``.

        Returns:
            The report.

        Raises:
            PruneRefusedError: If ``dry_run`` is ``False``.
            FileNotFoundError: If the results root is not a directory.
        """
        if not dry_run:
            raise PruneRefusedError(
                "the reference store never deletes bundles; prune only lists "
                "candidates (dry run). Delete by hand after review (OD-D4)."
            )
        results_root = Path(unreferenced_by)
        if not results_root.is_dir():
            raise FileNotFoundError(f"results root {results_root} is not a directory")
        referenced_hashes, n_record_files = referenced_build_hashes(results_root)
        referenced: list[StoreEntry] = []
        candidates: list[StoreEntry] = []
        in_progress: list[StoreEntry] = []
        for entry in self.list():
            if entry.kind == "bundle":
                if entry.build_hash in referenced_hashes:
                    referenced.append(entry)
                else:
                    candidates.append(entry)
            elif entry.kind == "tmp" and entry.builder_alive is not False:
                in_progress.append(entry)
            else:
                candidates.append(entry)
        return PruneReport(
            results_root=results_root,
            referenced=referenced,
            candidates=candidates,
            in_progress=in_progress,
            n_record_files=n_record_files,
            referenced_build_hashes=frozenset(referenced_hashes),
        )


# --------------------------------------------------------------------------
# Helpers


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _json_native(value: Any) -> Any:
    """Round-trip through JSON so a payload holds JSON-native values only."""
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def _inside(directory: Path, relative: str | Path) -> Path:
    candidate = (directory / relative).resolve()
    base = directory.resolve()
    if candidate == base or base not in candidate.parents:
        raise ValueError(f"{relative} is not a path inside the bundle")
    return candidate


@contextmanager
def _exclusive_lock(lock_path: Path) -> Iterator[None]:
    """Hold an exclusive ``flock`` on a lock file (created if needed, never removed).

    The lock is released by the kernel when the process dies, so a killed
    builder never blocks the next one.
    """
    # On NFS, Linux emulates flock() with a POSIX lock, which needs a
    # descriptor opened for writing; fall back to read-only only when another
    # store user created the lock file without write permission for us.
    try:
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o666)
    except PermissionError:
        descriptor = os.open(lock_path, os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def _fsync_directory(directory: Path) -> None:
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _mark_failed(
    work_dir: Path, root: Path, build_id: str, error: BaseException
) -> None:
    """Rename a failed build's directory to ``.failed-<id>``, never deleting it."""
    if not work_dir.is_dir():
        return
    try:
        (work_dir / BUILD_ERROR_NAME).write_text(
            "".join(traceback.format_exception(error)), encoding="utf-8"
        )
        failed_dir = root / f"{FAILED_DIR_PREFIX}{build_id}"
        os.rename(work_dir, failed_dir)
        logger.error(
            "Reference bundle build failed; its files are kept in %s", failed_dir
        )
    except OSError as rename_error:
        logger.error(
            "Reference bundle build failed and %s could not be renamed: %s",
            work_dir,
            rename_error,
        )


def _check_sources_unchanged(sources: Mapping[str, SourceRecord]) -> None:
    """Fail when a source changed size or mtime while the bundle was built."""
    for record in sources.values():
        for identity in record.files:
            stat = Path(identity.path).stat()
            if (
                int(stat.st_size) != identity.size
                or int(stat.st_mtime_ns) != identity.mtime_ns
            ):
                raise BundleIntegrityError(
                    f"reference source {identity.path} changed during the build"
                )


def _bundle_files(work_dir: Path) -> list[dict[str, Any]]:
    """Return ``{path, size, sha256}`` of every bundle file; refuse symlinks."""
    records: list[dict[str, Any]] = []
    for directory, dir_names, file_names in os.walk(work_dir):
        base = Path(directory)
        for name in [*dir_names, *file_names]:
            if (base / name).is_symlink():
                raise BundleIntegrityError(
                    f"bundle file {base / name} is a symlink; bundles hold copies "
                    "only (plan §3.2)"
                )
        dir_names.sort()
        for file_name in sorted(file_names):
            file_path = base / file_name
            relative = file_path.relative_to(work_dir)
            if str(relative) in (BUILD_MARKER_NAME, BUNDLE_MANIFEST_NAME):
                continue
            records.append(
                {
                    "path": str(relative),
                    "size": file_path.stat().st_size,
                    "sha256": file_sha256(file_path),
                }
            )
    return records


def _bundle_manifest(
    spec: AnnotationReferenceSpec,
    panel: AnnotationPanel | None,
    *,
    builder: BundleBuilder,
    request: BuildRequest,
    context: BuildContext,
    builder_output: dict[str, Any],
    started_at: str,
    wall_time_s: float,
) -> dict[str, Any]:
    copied_by_source = {copied.source_path: copied.sha256 for copied in context.copied}
    sources: dict[str, Any] = {}
    for name, record in sorted(request.sources.items()):
        files = []
        for identity in record.files:
            full = copied_by_source.get(identity.path) or file_sha256(identity.path)
            files.append({**identity.hash_payload(), "sha256": full})
        sources[name] = {"kind": record.kind, "path": record.path, "files": files}
    panel_trust = builder_output.pop("panel_trust", None)
    return {
        "schema_version": STORE_SCHEMA_VERSION,
        "status": BUNDLE_STATUS_COMPLETE,
        "build_hash": request.build_hash,
        "build_hash_payload": request.payload,
        "reference_id": spec.reference_id,
        "species": spec.species,
        "role": spec.role,
        "builder": builder.name,
        "builder_version": ANNOTATION_BUILDER_VERSION,
        "taxonomy_id": builder.taxonomy_id,
        "panel": None
        if panel is None
        else {
            "panel_hash": panel.panel_hash,
            "n_genes": panel.n_genes,
            "name": panel.name,
            "kind": panel.kind,
            "platforms": list(panel.platforms),
        },
        "panel_trust": panel_trust,
        "sources": sources,
        "copied_sources": [
            {
                "source_path": copied.source_path,
                "bundle_path": copied.bundle_path,
                "size": copied.size,
                "sha256": copied.sha256,
            }
            for copied in context.copied
        ],
        "files": _bundle_files(context.work_dir),
        "builder_output": _json_native(builder_output),
        "started_at": started_at,
        "created_at": _now(),
        "wall_time_s": round(wall_time_s, 3),
        "host": socket.gethostname(),
    }


def _read_manifest(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        manifest: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        logger.warning("Could not read bundle manifest %s: %s", path, error)
        return None
    return manifest if isinstance(manifest, dict) else None


def _directory_size(directory: Path) -> int:
    total = 0
    for base, _dir_names, file_names in os.walk(directory):
        for file_name in file_names:
            file_path = Path(base) / file_name
            if not file_path.is_symlink():
                try:
                    total += file_path.stat().st_size
                except OSError:
                    continue
    return total


def _process_alive(pid: int | None, host: str | None) -> bool | None:
    if pid is None or host != socket.gethostname():
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as error:
        return error.errno != errno.ESRCH
    return True


def _build_dir_entry(
    directory: Path, root: Path, kind: Literal["tmp", "failed"]
) -> StoreEntry:
    marker = _read_manifest(directory / BUILD_MARKER_NAME) or {}
    pid = marker.get("pid")
    host = marker.get("host")
    pid_value = int(pid) if isinstance(pid, int) else None
    host_value = str(host) if isinstance(host, str) else None
    return StoreEntry(
        kind=kind,
        path=directory,
        store_root=root,
        reference_id=marker.get("reference_id"),
        build_hash=marker.get("build_hash"),
        panel_hash=marker.get("panel_hash"),
        n_panel_genes=marker.get("n_panel_genes"),
        created_at=marker.get("started_at"),
        size_bytes=_directory_size(directory),
        builder_pid=pid_value,
        builder_host=host_value,
        builder_alive=_process_alive(pid_value, host_value) if kind == "tmp" else None,
    )


def _reference_entries(reference_dir: Path, root: Path) -> list[StoreEntry]:
    entries: list[StoreEntry] = []
    for bundle_dir in sorted(reference_dir.iterdir()):
        if (
            bundle_dir.is_symlink()
            or not bundle_dir.is_dir()
            or bundle_dir.name.startswith(".")
        ):
            continue
        manifest = _read_manifest(bundle_dir / BUNDLE_MANIFEST_NAME)
        is_complete = (
            manifest is not None
            and manifest.get("status") == BUNDLE_STATUS_COMPLETE
            and manifest.get("build_hash") == bundle_dir.name
        )
        panel = (manifest or {}).get("panel") or {}
        files = (manifest or {}).get("files", [])
        entries.append(
            StoreEntry(
                kind="bundle" if is_complete else "invalid",
                path=bundle_dir,
                store_root=root,
                reference_id=reference_dir.name,
                build_hash=bundle_dir.name,
                panel_hash=panel.get("panel_hash") if isinstance(panel, dict) else None,
                n_panel_genes=panel.get("n_genes") if isinstance(panel, dict) else None,
                created_at=(manifest or {}).get("created_at"),
                size_bytes=(
                    sum(int(record.get("size", 0)) for record in files)
                    if is_complete
                    else _directory_size(bundle_dir)
                ),
            )
        )
    return entries


def _is_reference_record(file_name: str) -> bool:
    return file_name in REFERENCE_RECORD_NAMES or file_name.endswith(
        REFERENCE_RECORD_SUFFIXES
    )


def _collect_hashes(value: Any, found: set[str]) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if (
                key == "build_hash"
                and isinstance(item, str)
                and len(item) == _SHA256_HEX_LENGTH
            ):
                found.add(item)
            else:
                _collect_hashes(item, found)
    elif isinstance(value, list | tuple):
        for item in value:
            _collect_hashes(item, found)


def referenced_build_hashes(results_root: Path | str) -> tuple[set[str], int]:
    """Return every ``build_hash`` recorded in a results tree.

    Searches ``bundle_ref.json``, ``map_manifest.json``,
    ``required_bundles.json`` and ``*_annotation_manifest.json`` files,
    skipping zarr stores, Nextflow work dirs and symlinked directories.

    Args:
        results_root: Directory to search.

    Returns:
        The build hashes found and the number of record files read.
    """
    found: set[str] = set()
    n_files = 0
    for directory, dir_names, file_names in os.walk(results_root):
        base = Path(directory)
        dir_names[:] = [
            name
            for name in dir_names
            if name not in _SKIPPED_SCAN_DIR_NAMES
            and not name.endswith(_SKIPPED_SCAN_DIR_SUFFIXES)
            and not (base / name).is_symlink()
        ]
        for file_name in file_names:
            if not _is_reference_record(file_name):
                continue
            record = _read_manifest(base / file_name)
            n_files += 1
            if record is not None:
                _collect_hashes(record, found)
    return found, n_files


def spec_sources_from_pairs(pairs: Sequence[str]) -> dict[str, Path]:
    """Parse ``NAME=PATH`` pairs (CLI ``--source``) into a sources mapping.

    Args:
        pairs: ``NAME=PATH`` strings.

    Returns:
        Paths by source name.

    Raises:
        ValueError: If a pair has no ``=`` or repeats a name.
    """
    sources: dict[str, Path] = {}
    for pair in pairs:
        name, separator, path = pair.partition("=")
        if not separator or not name.strip() or not path.strip():
            raise ValueError(f"source {pair!r} must be NAME=PATH")
        if name.strip() in sources:
            raise ValueError(f"source {name.strip()!r} given twice")
        sources[name.strip()] = Path(path.strip())
    return sources

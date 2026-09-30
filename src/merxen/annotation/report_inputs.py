"""Inputs of the annotation report: locating and reading published outputs.

The report (plan §9, §3.6) reads what the annotation stages published for one
human pair × segmentation, or one mouse section × segmentation, and never
writes next to them:

- RESOLVE (``annotation_resolve_out``): ``<pair>_resolve_summary.json``, the
  label tables ``<plat>/<sid>_celltype_labels.parquet`` and the annotation
  manifests ``<plat>/<sid>_annotation_manifest.json``;
- MAP (``annotation_map_out``): ``map_manifest.json``, the tidy SEA-AD
  parquets (H3), and the mouse ``<sid>_mouse_regions.parquet``;
- PANEL (``annotation_panel_out/panel_report.json``);
- COMPUTE_CPU / FINALIZE (``clustering_squidpy[_<suffix>]``): the clustered
  H5ADs, for coordinates, counts and the hierarchy record;
- cortical depth (``<pair>/<plat>/compute_cortical_depth[_mapfirst]``), the
  shared tissue mask (``<pair>/alignment/align_out``) and MENDER
  (``mender[_mapfirst]``) when present;
- the bundles named by the provenance (read-only), and an optional
  held-out-gene acceptance CSV (§5.8).

``discover_sources`` finds these in a results tree; every path can also be
given explicitly (``ReportSources``). ``load_report_inputs`` reads them.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import pandas as pd

from merxen.annotation.schema import Columns, label_table_filename

logger = logging.getLogger(__name__)

Species = Literal["human", "mouse"]

RESOLVE_SUBDIR: Final = Path("annotation_resolve") / "annotation_resolve_out"
MAP_SUBDIR: Final = Path("annotation_map") / "annotation_map_out"
PANEL_SUBDIR: Final = Path("annotation_panel") / "annotation_panel_out"
CLUSTERING_SUBDIRS: Final[tuple[str, ...]] = (
    "clustering_squidpy_mapfirst",
    "clustering_squidpy",
)
CORTICAL_DEPTH_SUBDIRS: Final[tuple[str, ...]] = (
    "compute_cortical_depth_mapfirst",
    "compute_cortical_depth",
)
MENDER_SUBDIRS: Final[tuple[str, ...]] = ("mender_mapfirst", "mender")
RESOLVE_SUMMARY_SUFFIX: Final = "_resolve_summary.json"
MANIFEST_SUFFIX: Final = "_annotation_manifest.json"
MAP_MANIFEST_NAME: Final = "map_manifest.json"
PANEL_REPORT_NAME: Final = "panel_report.json"
SEA_RUN_ID: Final = "seaad_mr_panel"
DEPTH_CELLS_SUFFIX: Final = "_cells_with_cortical_depth.parquet"
DEPTH_QC_NAME: Final = "cortical_depth_qc_summary.json"
DEPTH_COLUMNS: Final[tuple[str, ...]] = (
    "cell_id",
    "x",
    "y",
    "inside_cortical_ribbon",
    "cortical_depth_annotation",
    "equivolumetric_depth",
    "laplace_depth",
    "cortical_depth_qc_flag",
    "column_id",
)
MOUSE_REGIONS_SUFFIX: Final = "_mouse_regions.parquet"


class ReportInputError(ValueError):
    """The report inputs are missing or inconsistent."""


@dataclass(frozen=True)
class ReportSources:
    """Where the report reads its inputs (all read-only).

    Attributes:
        species: ``human`` or ``mouse``.
        pair_id: Pair (human) or section (mouse) id, as RESOLVE names it.
        segmentation: Segmentation.
        resolve_dir: ``annotation_resolve_out``.
        map_dir: ``annotation_map_out`` (SEA-AD tidy runs, mouse regions).
        panel_dir: ``annotation_panel_out`` (``panel_report.json``).
        clustered_h5ad: Sample id to its clustered H5AD.
        cortical_depth: Sample id to its cortical-depth cell parquet.
        alignment_dir: The pair's ``align_out`` (shared tissue mask).
        mender: Sample id to its ``mender_manifest.json``.
        heldout_csv: A held-out-gene enrichment CSV (§5.8), if any.
        store_root: A reference store, to find bundles whose recorded path
            moved.
        results_root: The results root the sources were found in, if any.
        notes: How each source was found (for the provenance footer).
    """

    species: Species
    pair_id: str
    segmentation: str
    resolve_dir: Path
    map_dir: Path | None = None
    panel_dir: Path | None = None
    clustered_h5ad: Mapping[str, Path] = field(default_factory=dict)
    cortical_depth: Mapping[str, Path] = field(default_factory=dict)
    alignment_dir: Path | None = None
    mender: Mapping[str, Path] = field(default_factory=dict)
    heldout_csv: Path | None = None
    store_root: Path | None = None
    results_root: Path | None = None
    notes: tuple[str, ...] = ()

    def input_roots(self: ReportSources) -> list[Path]:
        """Return every directory the report reads from."""
        roots: list[Path] = [self.resolve_dir]
        for item in (self.map_dir, self.panel_dir, self.alignment_dir):
            if item is not None:
                roots.append(item)
        for mapping in (self.clustered_h5ad, self.cortical_depth, self.mender):
            roots.extend(Path(path).parent for path in mapping.values())
        if self.results_root is not None:
            roots.append(self.results_root)
        return roots

    def describe(self: ReportSources) -> dict[str, Any]:
        """Return the sources as JSON (paths as strings)."""
        return {
            "species": self.species,
            "pair_id": self.pair_id,
            "segmentation": self.segmentation,
            "resolve_dir": str(self.resolve_dir),
            "map_dir": None if self.map_dir is None else str(self.map_dir),
            "panel_dir": None if self.panel_dir is None else str(self.panel_dir),
            "clustered_h5ad": {
                k: str(v) for k, v in sorted(self.clustered_h5ad.items())
            },
            "cortical_depth": {
                k: str(v) for k, v in sorted(self.cortical_depth.items())
            },
            "alignment_dir": (
                None if self.alignment_dir is None else str(self.alignment_dir)
            ),
            "mender": {k: str(v) for k, v in sorted(self.mender.items())},
            "heldout_csv": None if self.heldout_csv is None else str(self.heldout_csv),
            "store_root": None if self.store_root is None else str(self.store_root),
            "results_root": (
                None if self.results_root is None else str(self.results_root)
            ),
            "notes": list(self.notes),
        }


def _first_existing(candidates: Iterable[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _platform_dir(platform: str) -> str:
    return str(platform).lower()


def read_json(path: Path | str) -> dict[str, Any]:
    """Read a JSON object.

    Args:
        path: A JSON file.

    Returns:
        The object.

    Raises:
        ReportInputError: If the file does not hold a JSON object.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ReportInputError(f"{path} does not hold a JSON object")
    return payload


def resolve_summary_path(resolve_dir: Path, pair_id: str) -> Path:
    """Return ``<resolve_dir>/<pair>_resolve_summary.json``."""
    return Path(resolve_dir) / f"{pair_id}{RESOLVE_SUMMARY_SUFFIX}"


def _summary_samples(resolve_dir: Path, pair_id: str) -> dict[str, dict[str, Any]]:
    path = resolve_summary_path(resolve_dir, pair_id)
    if not path.is_file():
        raise ReportInputError(f"no RESOLVE summary at {path}")
    samples = read_json(path).get("samples") or {}
    return {str(key): dict(value) for key, value in samples.items()}


def discover_sources(
    results_root: Path | str,
    pair_id: str,
    segmentation: str,
    *,
    species: Species = "human",
    resolve_dir: Path | str | None = None,
    map_dir: Path | str | None = None,
    panel_dir: Path | str | None = None,
    clustering_subdirs: Sequence[str] = CLUSTERING_SUBDIRS,
    cortical_depth_subdirs: Sequence[str] = CORTICAL_DEPTH_SUBDIRS,
    mender_subdirs: Sequence[str] = MENDER_SUBDIRS,
    use_cortical_depth: bool = True,
    use_alignment: bool = True,
    use_mender: bool = True,
    heldout_csv: Path | str | None = None,
    store_root: Path | str | None = None,
) -> ReportSources:
    """Find a pair × segmentation's published annotation outputs.

    Layout (plan §3.6): ``<root>/<pair>/<seg>/{annotation_resolve,
    annotation_map, annotation_panel, clustering_squidpy[_<suffix>],
    mender[_<suffix>]}``, ``<root>/<pair>/<plat>/compute_cortical_depth
    [_mapfirst]/compute_cortical_depth_out/<seg>/*_cells_with_cortical_depth
    .parquet`` and ``<root>/<pair>/alignment/align_out``. The first existing
    candidate of each suffixed directory wins (map_first before legacy).

    Args:
        results_root: The results root.
        pair_id: Pair (or mouse section) id.
        segmentation: Segmentation.
        species: Species of the run.
        resolve_dir: Explicit RESOLVE output (default: found in the tree).
        map_dir: Explicit MAP output.
        panel_dir: Explicit PANEL output.
        clustering_subdirs: Clustering directory names, in order.
        cortical_depth_subdirs: Cortical-depth directory names, in order.
        mender_subdirs: MENDER directory names, in order.
        use_cortical_depth: Read cortical depth when present.
        use_alignment: Read the shared tissue mask when present.
        use_mender: Read the MENDER manifests when present.
        heldout_csv: A held-out-gene enrichment CSV.
        store_root: A reference store.

    Returns:
        The sources.

    Raises:
        ReportInputError: If no RESOLVE summary is found.
    """
    root = Path(results_root)
    base = root / pair_id / segmentation
    notes: list[str] = []
    resolve = Path(resolve_dir) if resolve_dir is not None else base / RESOLVE_SUBDIR
    samples = _summary_samples(resolve, pair_id)
    mapped = Path(map_dir) if map_dir is not None else base / MAP_SUBDIR
    panel = Path(panel_dir) if panel_dir is not None else base / PANEL_SUBDIR
    clustered: dict[str, Path] = {}
    depth: dict[str, Path] = {}
    mender: dict[str, Path] = {}
    for sample_id, entry in sorted(samples.items()):
        platform = _platform_dir(str(entry.get("platform", "")))
        found = _first_existing(
            base
            / name
            / "clustering_squidpy_out"
            / platform
            / f"{sample_id}_clustered.h5ad"
            for name in clustering_subdirs
        )
        if found is not None:
            clustered[sample_id] = found
            notes.append(
                f"clustered H5AD of {sample_id}: {found.parent.parent.parent.name}"
            )
        if use_cortical_depth and species == "human":
            for name in cortical_depth_subdirs:
                directory = (
                    root / pair_id / platform / name / "compute_cortical_depth_out"
                )
                hits = sorted((directory / segmentation).glob(f"*{DEPTH_CELLS_SUFFIX}"))
                if hits:
                    depth[sample_id] = hits[0]
                    notes.append(f"cortical depth of {sample_id}: {name}")
                    break
        if use_mender:
            found = _first_existing(
                base / name / "mender_out" / platform / "mender_manifest.json"
                for name in mender_subdirs
            )
            if found is not None:
                mender[sample_id] = found
    alignment = root / pair_id / "alignment" / "align_out"
    return ReportSources(
        species=species,
        pair_id=pair_id,
        segmentation=segmentation,
        resolve_dir=resolve,
        map_dir=mapped if mapped.is_dir() else None,
        panel_dir=panel if panel.is_dir() else None,
        clustered_h5ad=clustered,
        cortical_depth=depth,
        alignment_dir=alignment if use_alignment and alignment.is_dir() else None,
        mender=mender,
        heldout_csv=None if heldout_csv is None else Path(heldout_csv),
        store_root=None if store_root is None else Path(store_root),
        results_root=root,
        notes=tuple(notes),
    )


# --------------------------------------------------------------------------
# Loaded data


@dataclass
class ClusteredTable:
    """What the report reads from a clustered H5AD.

    Attributes:
        path: The file.
        obs_names: Cell ids (table cells).
        gene_ids: Ensembl id per gene (``var['ensembl_id']`` when present,
            else the var names).
        gene_symbols: Symbol per gene.
        counts: ``layers['counts']`` (or ``X``) as a scipy CSR matrix.
        xy: ``obsm['spatial']`` (µm), if present.
        shape_key: ``uns['merxen_clustering_squidpy']['shape_key']``.
        hierarchy: Flat scalars of ``uns['merxen_hierarchical_clustering']``.
    """

    path: Path
    obs_names: pd.Index
    gene_ids: list[str]
    gene_symbols: list[str]
    counts: Any
    xy: np.ndarray | None
    shape_key: str | None
    hierarchy: dict[str, Any]


def _read_string_array(node: Any) -> list[str]:
    values = node[()]
    return [
        value.decode() if isinstance(value, bytes) else str(value) for value in values
    ]


def _read_index(group: Any) -> list[str]:
    key = group.attrs.get("_index", "_index")
    key = key.decode() if isinstance(key, bytes) else str(key)
    return _read_string_array(group[key])


def _read_column(group: Any, name: str) -> list[str] | None:
    if name not in group:
        return None
    node = group[name]
    if hasattr(node, "keys") and "categories" in node:
        categories = _read_string_array(node["categories"])
        codes = node["codes"][()]
        return [categories[code] if code >= 0 else "" for code in codes]
    return _read_string_array(node)


def _read_scalar(node: Any) -> Any:
    value = node[()]
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def read_clustered_table(
    path: Path | str, *, with_counts: bool = True
) -> ClusteredTable:
    """Read a clustered H5AD's ids, genes, counts, coordinates and records.

    Args:
        path: The clustered (or prepared) H5AD.
        with_counts: Read the count matrix (``False``: ids and coordinates
            only).

    Returns:
        The table.
    """
    import h5py
    from scipy import sparse

    with h5py.File(path, "r") as handle:
        obs_names = pd.Index(_read_index(handle["obs"]), dtype=str)
        var = handle["var"]
        symbols = _read_index(var)
        ids = _read_column(var, "ensembl_id") or list(symbols)
        ids = [gene_id or symbol for gene_id, symbol in zip(ids, symbols, strict=True)]
        counts = None
        if with_counts:
            node = (
                handle["layers"]["counts"]
                if "layers" in handle and "counts" in handle["layers"]
                else handle["X"]
            )
            if hasattr(node, "keys"):
                shape = tuple(int(v) for v in node.attrs["shape"])
                counts = sparse.csr_matrix(
                    (node["data"][()], node["indices"][()], node["indptr"][()]),
                    shape=shape,
                )
            else:
                counts = sparse.csr_matrix(node[()])
        xy = None
        if "obsm" in handle and "spatial" in handle["obsm"]:
            xy = np.asarray(handle["obsm"]["spatial"][()], dtype=np.float64)[:, :2]
        shape_key = None
        key = "uns/merxen_clustering_squidpy/shape_key"
        if key in handle:
            shape_key = str(_read_scalar(handle[key]))
        hierarchy: dict[str, Any] = {}
        group_key = "uns/merxen_hierarchical_clustering"
        if group_key in handle:
            for name, node in handle[group_key].items():
                if not hasattr(node, "keys"):
                    try:
                        hierarchy[name] = _read_scalar(node)
                    except (TypeError, ValueError):
                        continue
    return ClusteredTable(
        path=Path(path),
        obs_names=obs_names,
        gene_ids=[str(value) for value in ids],
        gene_symbols=[str(value) for value in symbols],
        counts=counts,
        xy=xy,
        shape_key=shape_key,
        hierarchy=hierarchy,
    )


@dataclass
class SampleData:
    """One sample's report inputs.

    Attributes:
        sample_id: Sample id.
        platform: ``MERSCOPE`` or ``XENIUM``.
        labels: The full label table (every segmented object).
        summary: Its entry of the RESOLVE summary.
        manifest: Its annotation manifest (provenance JSON).
        labels_path: The label table file.
        xy: Per-object coordinates (µm; NaN where unknown), aligned to
            ``labels``.
        aligned_frame: Whether ``xy`` is in the pair's fixed (Xenium) frame.
        coordinate_source: Where ``xy`` came from.
        depth: Per-object cortical depth columns aligned to ``labels`` (or
            ``None``).
        depth_path: The cortical-depth parquet.
        depth_qc: Its QC summary.
        clustered_path: The clustered H5AD, if found.
        mender_manifest: The MENDER manifest, if found.
        map_record: The sample's MAP manifest record.
        mouse_regions: The mouse region-step cell table, if any.
        sea_tidy_path: The SEA-AD tidy parquet of the MAP run, if found.
    """

    sample_id: str
    platform: str
    labels: pd.DataFrame
    summary: dict[str, Any]
    manifest: dict[str, Any]
    labels_path: Path
    xy: np.ndarray | None = None
    aligned_frame: bool = False
    coordinate_source: str | None = None
    depth: pd.DataFrame | None = None
    depth_path: Path | None = None
    depth_qc: dict[str, Any] | None = None
    clustered_path: Path | None = None
    mender_manifest: dict[str, Any] | None = None
    map_record: dict[str, Any] | None = None
    mouse_regions: pd.DataFrame | None = None
    sea_tidy_path: Path | None = None

    @property
    def in_table(self: SampleData) -> np.ndarray:
        """Return whether each object is a table cell."""
        return np.asarray(self.labels[Columns.IN_TABLE].to_numpy(dtype=bool))

    def table(self: SampleData) -> pd.DataFrame:
        """Return the table cells' rows."""
        return self.labels[self.in_table]

    def table_xy(self: SampleData) -> np.ndarray | None:
        """Return the table cells' coordinates, if known."""
        if self.xy is None:
            return None
        return np.asarray(self.xy[self.in_table])


@dataclass
class ReportInputs:
    """Everything one report reads.

    Attributes:
        sources: Where it came from.
        summary: The RESOLVE pair summary.
        samples: Sample id to its data (sorted by platform).
        panel_report: ``panel_report.json``, if present.
        map_manifest: ``map_manifest.json``, if present.
        bundles: Reference id to its bundle directory (existing ones only).
        mask: The pair's shared tissue mask (``panel.SharedTissueMask``).
        heldout: The held-out-gene enrichment rows of this pair, if given.
    """

    sources: ReportSources
    summary: dict[str, Any]
    samples: dict[str, SampleData]
    panel_report: dict[str, Any] | None = None
    map_manifest: dict[str, Any] | None = None
    bundles: dict[str, Path] = field(default_factory=dict)
    mask: Any | None = None
    heldout: pd.DataFrame | None = None

    @property
    def species(self: ReportInputs) -> Species:
        """Return the species."""
        return self.sources.species

    def ordered_samples(self: ReportInputs) -> list[SampleData]:
        """Return the samples, MERSCOPE first."""
        return sorted(
            self.samples.values(), key=lambda item: (item.platform, item.sample_id)
        )


def _align_xy(labels: pd.DataFrame, ids: pd.Index, xy: np.ndarray) -> np.ndarray:
    out = np.full((len(labels), 2), np.nan)
    position = ids.get_indexer(labels[Columns.CELL_ID].astype(str))
    found = position >= 0
    out[found] = xy[position[found]]
    return out


def _load_depth(
    labels: pd.DataFrame, path: Path
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    import pyarrow.parquet as pq

    names = set(pq.read_schema(path).names)
    columns = [name for name in DEPTH_COLUMNS if name in names]
    frame = pd.read_parquet(path, columns=columns)
    frame["cell_id"] = frame["cell_id"].astype(str)
    frame = frame.drop_duplicates("cell_id").set_index("cell_id")
    aligned = frame.reindex(labels[Columns.CELL_ID].astype(str).to_numpy())
    aligned.index = labels.index
    qc_path = _first_existing(
        [path.parent / DEPTH_QC_NAME, path.parent.parent / DEPTH_QC_NAME]
    )
    qc = read_json(qc_path) if qc_path is not None else None
    return aligned, qc


def _find_bundle(
    reference_id: str, record: Mapping[str, Any], store_root: Path | None
) -> Path | None:
    recorded = record.get("bundle_path")
    if recorded and Path(str(recorded)).is_dir():
        return Path(str(recorded))
    build_hash = record.get("build_hash")
    if store_root is not None and build_hash:
        candidate = store_root / reference_id / str(build_hash)
        if candidate.is_dir():
            return candidate
    return None


def _map_samples(map_manifest: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not map_manifest:
        return {}
    samples = map_manifest.get("samples") or {}
    if isinstance(samples, list):
        return {str(item.get("sample_id")): dict(item) for item in samples}
    return {str(key): dict(value) for key, value in samples.items()}


def load_heldout(path: Path, pair_id: str) -> pd.DataFrame | None:
    """Read the held-out-gene enrichment rows of a pair (§5.8).

    Accepts the ``heldout_enrichment*.csv`` of ``heldout_genes.py`` /
    ``resolve_criteria.py`` (columns ``pair``, ``platform``, ``broad_class``,
    ``fold``, ``auroc`` and optionally ``label_set``).

    Args:
        path: The CSV.
        pair_id: Pair to keep.

    Returns:
        The pair's rows, or ``None`` when the file has none.
    """
    frame = pd.read_csv(path)
    required = {"pair", "platform", "broad_class", "fold", "auroc"}
    missing = required - set(frame.columns)
    if missing:
        raise ReportInputError(f"{path} lacks the columns {sorted(missing)}")
    frame = frame[frame["pair"].astype(str) == str(pair_id)]
    return frame.reset_index(drop=True) if len(frame) else None


def load_report_inputs(sources: ReportSources) -> ReportInputs:
    """Read every input of a report.

    Args:
        sources: Where to read.

    Returns:
        The loaded inputs.

    Raises:
        ReportInputError: If the RESOLVE outputs are missing or inconsistent.
    """
    from merxen.annotation.pipeline import (
        in_fixed_frame,
        load_pair_mask,
        read_label_table,
    )

    summary_path = resolve_summary_path(sources.resolve_dir, sources.pair_id)
    if not summary_path.is_file():
        raise ReportInputError(f"no RESOLVE summary at {summary_path}")
    summary = read_json(summary_path)
    species = str(summary.get("species", sources.species))
    if species != sources.species:
        raise ReportInputError(
            f"{summary_path} is a {species} RESOLVE output, not {sources.species}"
        )
    map_manifest = None
    if sources.map_dir is not None and (sources.map_dir / MAP_MANIFEST_NAME).is_file():
        map_manifest = read_json(sources.map_dir / MAP_MANIFEST_NAME)
    map_samples = _map_samples(map_manifest)
    panel_report = None
    if (
        sources.panel_dir is not None
        and (sources.panel_dir / PANEL_REPORT_NAME).is_file()
    ):
        panel_report = read_json(sources.panel_dir / PANEL_REPORT_NAME)
    samples: dict[str, SampleData] = {}
    bundles: dict[str, Path] = {}
    for sample_id, entry in sorted((summary.get("samples") or {}).items()):
        platform = str(entry.get("platform", "")).upper()
        labels_rel = entry.get("labels") or (
            Path(_platform_dir(platform)) / label_table_filename(sample_id)
        )
        labels_path = sources.resolve_dir / str(labels_rel)
        if not labels_path.is_file():
            raise ReportInputError(f"label table of {sample_id} missing: {labels_path}")
        labels, _ = read_label_table(labels_path)
        labels = labels.reset_index(drop=True)
        manifest_rel = entry.get("annotation_manifest") or (
            Path(_platform_dir(platform)) / f"{sample_id}{MANIFEST_SUFFIX}"
        )
        manifest_path = sources.resolve_dir / str(manifest_rel)
        manifest = read_json(manifest_path) if manifest_path.is_file() else {}
        data = SampleData(
            sample_id=str(sample_id),
            platform=platform,
            labels=labels,
            summary=dict(entry),
            manifest=manifest,
            labels_path=labels_path,
            map_record=map_samples.get(str(sample_id)),
        )
        clustered = sources.clustered_h5ad.get(str(sample_id))
        if clustered is not None and Path(clustered).is_file():
            table = read_clustered_table(clustered, with_counts=False)
            if table.xy is not None:
                data.xy = _align_xy(labels, table.obs_names, table.xy)
                data.aligned_frame = in_fixed_frame(platform, table.shape_key)
                data.coordinate_source = f"clustered:{table.shape_key or 'spatial'}"
            data.clustered_path = Path(clustered)
        depth_path = sources.cortical_depth.get(str(sample_id))
        if depth_path is not None and Path(depth_path).is_file():
            data.depth, data.depth_qc = _load_depth(labels, Path(depth_path))
            data.depth_path = Path(depth_path)
            if data.xy is None and {"x", "y"} <= set(data.depth.columns):
                data.xy = data.depth[["x", "y"]].to_numpy(dtype=np.float64)
                key = str(
                    (data.depth_qc or {})
                    .get("tables", {})
                    .get(sources.segmentation, {})
                    .get("shape_key", "")
                )
                data.aligned_frame = in_fixed_frame(platform, key)
                data.coordinate_source = f"cortical_depth:{key or 'xy'}"
        mender = sources.mender.get(str(sample_id))
        if mender is not None and Path(mender).is_file():
            data.mender_manifest = read_json(mender)
        if sources.map_dir is not None:
            sea = (
                sources.map_dir
                / _platform_dir(platform)
                / f"{sample_id}_mmc_{SEA_RUN_ID}.parquet"
            )
            data.sea_tidy_path = sea if sea.is_file() else None
            regions = (
                sources.map_dir
                / _platform_dir(platform)
                / f"{sample_id}{MOUSE_REGIONS_SUFFIX}"
            )
            if regions.is_file():
                frame = pd.read_parquet(regions)
                frame["cell_id"] = frame["cell_id"].astype(str)
                data.mouse_regions = frame
        for reference_id, record in (manifest.get("references") or {}).items():
            if reference_id in bundles:
                continue
            found = _find_bundle(str(reference_id), record, sources.store_root)
            if found is not None:
                bundles[str(reference_id)] = found
        samples[str(sample_id)] = data
    if not samples:
        raise ReportInputError(f"{summary_path} lists no samples")
    mask = None
    if sources.alignment_dir is not None:
        mask = load_pair_mask(sources.alignment_dir)
    heldout = None
    if sources.heldout_csv is not None:
        heldout = load_heldout(sources.heldout_csv, sources.pair_id)
    return ReportInputs(
        sources=sources,
        summary=summary,
        samples=samples,
        panel_report=panel_report,
        map_manifest=map_manifest,
        bundles=bundles,
        mask=mask,
        heldout=heldout,
    )


def with_sources(inputs: ReportInputs, **changes: Any) -> ReportInputs:
    """Return a copy of ``inputs`` whose sources carry ``changes``."""
    return replace(inputs, sources=replace(inputs.sources, **changes))


__all__ = [
    "CLUSTERING_SUBDIRS",
    "CORTICAL_DEPTH_SUBDIRS",
    "MENDER_SUBDIRS",
    "ClusteredTable",
    "ReportInputError",
    "ReportInputs",
    "ReportSources",
    "SampleData",
    "discover_sources",
    "load_heldout",
    "load_report_inputs",
    "read_clustered_table",
    "read_json",
    "resolve_summary_path",
    "with_sources",
]

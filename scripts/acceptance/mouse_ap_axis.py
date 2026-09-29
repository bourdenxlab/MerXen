#!/usr/bin/env python
"""M6b AP-axis scoring: the region rule on MERFISH-638850, out of sample (§17).

Pre-registration §17 (``docs/acceptance/annotation-v1-preregistration.md``),
plan §7.8 items 1, 2 and 4, MO10 (§14). The inputs are stage A's unpruned
production WMB mappings of the 38 in-scope sections' 30k subsamples on the
three gene sets (``--mouse-section-regions none``) and the M6 MAP outputs of
ag7 proseg_hybrid and VZG2 original_seg. Subcommands, each writing under
``--out``:

1. ``tables``: the MERFISH CCF view's per-section counts
   (``mouse_ap.MerfishRegionCounts``), the check that with nothing left out
   they reproduce the store's ``wmb_region_share`` tables exactly, the
   leave-neighbours-out sets (each section and its ± 1 neighbours in the AP
   order of all 53 sections), each section's CCF regions (>= 1% and > 0.2%
   of its grey-matter cells) and the Allen confident-neuron subclass counts
   per section (AP-estimate validation).
2. ``plan``: per section x gene set (and ag7 / VZG2) and per candidate rule
   (v1 plus the 95 candidates of (a)-(d) x (f)), the production region step
   (``mouse_regions.plan_region_step``) with the leave-neighbours-out shares:
   inferred, CCF, missed and extra regions, the drop list, the cells to
   re-map, the **direct harm** (table cells with ``average_correlation_score
   >= 0.6``, unpruned class bp >= 0.9 and an Allen class or subclass in the
   drop list, over table cells), the confident-cell harm and the E7
   oracle-relative harm (diagnostic); and one re-map job per distinct drop
   list of a dataset.
3. ``remap``: the production pruned re-map of each job (the cells whose
   unpruned class or subclass is dropped, the section's bundle, the pruned
   lookup and ``--nodes_to_drop``; as ``pipeline._remap_cells``).
4. ``summarise``: T2 after pruning (``mouse_gate.t2_share`` on the pruned
   subclass calls with the candidate's present regions and never-drop
   nodes), relabelled cells, MO2 on ag7 and VZG2, the per-AP-bin tables, the
   candidate table and the pre-registered selection (lowest maximum direct
   harm over the 38 sections and 3 gene sets, subject to MO2 on ag7 and VZG2
   and the bin-mean T2 <= 0.1% in every bin; ties: fewer changes from v1,
   then fewer drop decisions differing from v1, then the label).
5. ``mo10``: MO10 on the selected candidate, per gene set; refuses to run
   unless the pre-registration names the selected candidate (§17 "order of
   record").
6. ``ap-estimate``: ``mouse_ap.estimate_ap`` on held-out MERFISH sections
   (each section left out of the reference): with the Allen labels of every
   section and with our mappings of the in-scope sections; the error
   against one section step.

Usage::

    mouse_ap_axis.py tables --ccf-metadata VIEW.csv --region-share-bundle DIR \
        --section-table section_region_share.csv --out OUT
    mouse_ap_axis.py plan --out OUT --mapping-done DONE_DIR --subsamples DIR \
        --real ag7=MAP_DIR --real vzg2=MAP_DIR --region-share-bundle DIR
    mouse_ap_axis.py remap --out OUT --gene-id-fallback-csv gene.csv \
        [--worker I --n-workers N --n-processors P]
    mouse_ap_axis.py summarise --out OUT
    mouse_ap_axis.py mo10 --out OUT --preregistration PREREG.md
    mouse_ap_axis.py ap-estimate --out OUT --region-share-bundle DIR \
        --mapping-done DONE_DIR --subsamples DIR

Imports: numpy, pandas and merxen (the merxen analysis env; ``remap`` needs
``cell_type_mapper``).
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import os
import shutil
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from merxen.annotation.config import MouseRegionConfig
from merxen.annotation.mouse_ap import (
    AP_BIN_LABELS,
    ApEstimateError,
    MerfishRegionCounts,
    ap_bin,
    estimate_ap,
    is_neuronal_subclass,
    section_steps,
)
from merxen.annotation.mouse_gate import t2_share
from merxen.annotation.mouse_regions import (
    DroppedNode,
    RegionShares,
    SectionRegionsRequest,
    WmbTaxonomy,
    effective_never_drop,
    plan_region_step,
    two_tier_drop_list,
)
from merxen.annotation.reference import (
    GREY_REGIONS,
    REGION_HOME_FILE,
    REGION_SHARE_FILE,
    SECTION_COMPOSITION_FILE,
    merfish_region,
)
from merxen.annotation.schema import meets_threshold

logger = logging.getLogger("mouse_ap_axis")

GENE_SETS: tuple[str, ...] = ("merfish500", "merfish_ag7", "merfish_vzg2")
REAL_DATASETS: tuple[str, ...] = ("ag7", "vzg2")
CLASS_LEVEL = "CCN20230722_CLAS"
SUBCLASS_LEVEL = "CCN20230722_SUBC"
PRIMARY_RUN = "wmb_panel"
# Harm (§17): confident Allen label and our class bp.
ALLEN_ACS_MIN = 0.6
CLASS_BP_MIN = 0.9
# CCF truth per section (E7 09_autoregion_v2.py): present >= 1%, any > 0.2%.
CCF_PRESENT_SHARE = 0.01
CCF_ANY_SHARE = 0.002
# E7's MERFISH confident neurons (Allen labels): average correlation >= 0.5.
ALLEN_NEURON_ACS_MIN = 0.5
IN_SCOPE_MAX_CB_OB = 0.5
# MO2 (§9) and MO10 (§9) limits, as fractions.
MO2_T2_MAX = 0.001
MO2_CLASS_OUTSIDE_MAX = 0.0025
MO2_SUBCLASS_OUTSIDE_MAX = 0.015
MO10_BIN_MEAN_HARM_MAX = 0.001
MO10_SECTION_MAX_HARM = 0.003
MO10_T2_BIN_MEAN_MAX = 0.001
MO10_MISSED_HARM_MAX = 0.001
SECTION_ABOVE = 0.001
# E7 PRESENT_MANUAL: the manual anatomy of ag7 and VZG2 (MO3).
MANUAL_REGIONS: dict[str, tuple[str, ...]] = {
    "ag7": ("Isocortex", "OLF", "HPF", "CTXsp", "TH", "HY", "MB"),
    "vzg2": ("Isocortex", "OLF", "HPF", "CTXsp", "TH", "HY", "MB"),
}
CCF_COLUMNS: tuple[str, ...] = (
    "cell_label",
    "brain_section_label",
    "class",
    "subclass",
    "average_correlation_score",
    "x_ccf",
    "parcellation_division",
    "parcellation_structure",
)
SECTION_PREFIX = "C57BL6J-638850."

# --------------------------------------------------------------------------
# Candidate rules (§17 "Candidates")

COMPONENTS: dict[str, dict[str, Any]] = {
    "a": {"coupled_regions": {"OB": ["OLF"]}},
    "b": {
        "conditional_never_drop": [
            {
                "level": "class",
                "name": "05 OB-IMN GABA",
                "when_any_present": ["STR", "OLF"],
            }
        ]
    },
    "c": {
        "conditional_never_drop": [
            {
                "level": "subclass",
                "name": "025 CA2-FC-IG Glut",
                "when_all_present": ["Isocortex", "STR"],
            }
        ]
    },
    "d": {"region_min_tile_fraction": {"OB": 0.005}},
}
TILE_SETTINGS: tuple[tuple[float, int], ...] = (
    (150.0, 3),
    (100.0, 3),
    (200.0, 3),
    (150.0, 2),
    (100.0, 2),
    (200.0, 2),
)
V1_TILE = (150.0, 3)


@dataclass(frozen=True)
class Candidate:
    """One candidate region rule (§17).

    Attributes:
        components: Of ``a``-``d``, sorted.
        tile_um: Tile edge (µm).
        min_neurons: Confident neurons per assigned tile.
    """

    components: tuple[str, ...]
    tile_um: float
    min_neurons: int

    @property
    def is_v1(self) -> bool:
        """Whether this is the unchanged v1 rule (the reference)."""
        return not self.components and (self.tile_um, self.min_neurons) == V1_TILE

    @property
    def label(self) -> str:
        """Canonical label (``v1``, ``a+b``, ``a+b+f:tile100``, ...)."""
        if self.is_v1:
            return "v1"
        parts = list(self.components)
        tile: list[str] = []
        if self.tile_um != V1_TILE[0]:
            tile.append(f"tile{self.tile_um:g}")
        if self.min_neurons != V1_TILE[1]:
            tile.append(f"nn{self.min_neurons}")
        if tile:
            parts.append("f:" + ",".join(tile))
        return "+".join(parts)

    @property
    def n_changes(self) -> int:
        """Changes from v1: each of (a)-(d) 1, each changed tile parameter 1."""
        return (
            len(self.components)
            + int(self.tile_um != V1_TILE[0])
            + int(self.min_neurons != V1_TILE[1])
        )

    def config(self) -> MouseRegionConfig:
        """Return the rule's ``MouseRegionConfig`` (M6 values elsewhere)."""
        settings: dict[str, Any] = {
            "tile_um": self.tile_um,
            "min_neurons_per_tile": self.min_neurons,
        }
        keep: list[dict[str, Any]] = []
        for component in self.components:
            fields = COMPONENTS[component]
            keep.extend(fields.get("conditional_never_drop", []))
            settings.update(
                {k: v for k, v in fields.items() if k != "conditional_never_drop"}
            )
        if keep:
            settings["conditional_never_drop"] = keep
        if self.components:
            settings["rule_variant"] = "m6b"
        return MouseRegionConfig.model_validate(settings)


def all_candidates() -> list[Candidate]:
    """Return v1 followed by the 95 candidates (§17)."""
    result = [Candidate((), *V1_TILE)]
    for size in range(len(COMPONENTS) + 1):
        for components in itertools.combinations(sorted(COMPONENTS), size):
            for tile_um, min_neurons in TILE_SETTINGS:
                candidate = Candidate(tuple(components), tile_um, min_neurons)
                if not candidate.is_v1:
                    result.append(candidate)
    return result


def candidate_by_label(label: str) -> Candidate:
    """Return the candidate with a canonical label."""
    for candidate in all_candidates():
        if candidate.label == label:
            return candidate
    raise KeyError(f"unknown candidate {label!r}")


# --------------------------------------------------------------------------
# Helpers


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str))
    temporary.replace(path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(text)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def drop_sha(nodes: Sequence[DroppedNode]) -> str:
    """Return the sha256 of a drop list's ``[level, node]`` pairs (§4.6)."""
    pairs = [list(node.as_pair()) for node in nodes]
    return hashlib.sha256(json.dumps(pairs, sort_keys=True).encode("utf-8")).hexdigest()


def section_number(section: str) -> int:
    """Return NN of ``C57BL6J-638850.NN``."""
    return int(str(section).rsplit(".", 1)[-1])


def section_label(pair: str) -> str:
    """Return the section label of a pair id ``MF638850_NN``."""
    return f"{SECTION_PREFIX}{int(pair.rsplit('_', 1)[-1])}"


def ccf_regions(shares: pd.Series, minimum: float, *, strict: bool) -> set[str]:
    """Return the divisions holding more (strict) or at least a share."""
    if strict:
        return {str(region) for region, value in shares.items() if value > minimum}
    return {str(region) for region, value in shares.items() if value >= minimum}


def _join(regions: Iterable[str]) -> str:
    order = {region: index for index, region in enumerate(GREY_REGIONS)}
    return ";".join(sorted(set(regions), key=lambda region: order.get(region, 99)))


def _split(value: object) -> list[str]:
    text = "" if value is None or value != value else str(value)
    return [part for part in text.split(";") if part]


# --------------------------------------------------------------------------
# 1. tables


def read_ccf_view(path: Path) -> pd.DataFrame:
    """Read the CCF view's columns and add the E7 v2 region."""
    cells = pd.read_csv(path, usecols=list(CCF_COLUMNS), engine="pyarrow")
    cells["region"] = merfish_region(
        cells["parcellation_division"], cells["parcellation_structure"], cells["x_ccf"]
    )
    return cells


def compare_tables(
    tables: Mapping[str, pd.DataFrame], bundle_dir: Path
) -> dict[str, Any]:
    """Compare the no-exclusion tables with a ``wmb_region_share`` bundle."""
    result: dict[str, Any] = {"bundle": str(bundle_dir)}
    for key, name, columns in (
        ("region_share", REGION_SHARE_FILE, ["n", "share", "n_grey", "n_all"]),
        ("region_home", REGION_HOME_FILE, ["home", "home_share", "n_grey"]),
    ):
        ours = tables[key].copy()
        theirs = pd.read_parquet(bundle_dir / name)
        keys = ["level", "node_name"] + (["region"] if key == "region_share" else [])
        for frame in (ours, theirs):
            for column in keys:
                frame[column] = frame[column].astype(str)
        merged = ours.merge(theirs, on=keys, how="outer", suffixes=("", "_b"))
        entry: dict[str, Any] = {
            "rows_ours": len(ours),
            "rows_bundle": len(theirs),
            "rows_unmatched": int(
                merged[[f"{c}_b" for c in columns]].isna().any(axis=1).sum()
                + merged[columns].isna().any(axis=1).sum()
            ),
        }
        for column in columns:
            left, right = merged[column], merged[f"{column}_b"]
            if column == "home":
                entry[f"{column}_differ"] = int(
                    (left.astype(str) != right.astype(str)).sum()
                )
            else:
                diff = (pd.to_numeric(left) - pd.to_numeric(right)).abs()
                entry[f"{column}_max_abs_diff"] = float(diff.max())
        result[key] = entry
    result["equal"] = all(
        value["rows_unmatched"] == 0
        and value.get("home_differ", 0) == 0
        and all(value[key] <= 1e-12 for key in value if key.endswith("_max_abs_diff"))
        for value in (result["region_share"], result["region_home"])
    )
    return result


def tables_command(args: argparse.Namespace) -> int:
    """Write the count tables, the checks and the section table."""
    out = Path(args.out) / "tables"
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    cells = read_ccf_view(Path(args.ccf_metadata))
    counts = MerfishRegionCounts.from_cells(cells)
    counts.grey.rename("n").to_frame().to_parquet(out / "grey.parquet")
    counts.structure.rename("n").to_frame().to_parquet(out / "structure.parquet")
    counts.all_cells.rename("n").to_frame().to_parquet(out / "all_cells.parquet")
    counts.section_ap.rename("ap_ccf_mm").to_frame().to_parquet(
        out / "section_ap.parquet"
    )
    full = counts.tables(())
    check = compare_tables(full, Path(args.region_share_bundle))
    check["region_share_bundle_sha256"] = {
        name: _sha256(Path(args.region_share_bundle) / name)
        for name in (REGION_SHARE_FILE, REGION_HOME_FILE, SECTION_COMPOSITION_FILE)
    }
    full["structure_home"].to_parquet(out / "structure_home_full.parquet")
    _write_json(out / "tables_check.json", check)
    scope = pd.read_csv(args.section_table, index_col=0)
    in_scope = set(
        scope.index[(scope["CB"] + scope["OB"]) < IN_SCOPE_MAX_CB_OB].astype(str)
    )
    order = counts.order
    steps = section_steps(order, counts.section_ap)
    rows = []
    for section in order:
        shares = counts.section_region_shares(section)
        excluded = counts.neighbours_out(section)
        rows.append(
            {
                "section": section,
                "ap_ccf_mm": float(counts.section_ap[section]),
                "ap_bin": ap_bin(float(counts.section_ap[section])),
                "in_scope": section in in_scope,
                "excluded": ";".join(excluded),
                "section_step_mm": float(steps[section]),
                "ccf_present": _join(
                    ccf_regions(shares, CCF_PRESENT_SHARE, strict=False)
                ),
                "ccf_any": _join(ccf_regions(shares, CCF_ANY_SHARE, strict=True)),
                **{
                    f"share_{region}": float(shares.get(region, 0.0))
                    for region in GREY_REGIONS
                },
            }
        )
    sections = pd.DataFrame(rows)
    sections.to_csv(out / "sections.csv", index=False)
    neurons = cells[
        cells["subclass"].map(is_neuronal_subclass).to_numpy(dtype=bool)
        & (cells["average_correlation_score"] >= ALLEN_NEURON_ACS_MIN).to_numpy()
    ]
    allen = (
        neurons.groupby(["brain_section_label", "subclass"])
        .size()
        .rename("n")
        .reset_index()
    )
    allen.to_parquet(out / "allen_confident_neurons.parquet")
    summary = {
        "n_cells": int(len(cells)),
        "n_sections": len(order),
        "n_in_scope": int(sections["in_scope"].sum()),
        "in_scope_per_bin": sections[sections["in_scope"]]
        .groupby("ap_bin")
        .size()
        .to_dict(),
        "tables_equal_bundle": check["equal"],
        "wall_s": round(time.monotonic() - start, 1),
    }
    _write_json(out / "tables_summary.json", summary)
    logger.info("tables: %s", summary)
    return 0 if check["equal"] else 1


def load_counts(out: Path) -> MerfishRegionCounts:
    """Read the count tables ``tables`` wrote."""
    tables = out / "tables"
    return MerfishRegionCounts(
        grey=pd.read_parquet(tables / "grey.parquet")["n"],
        structure=pd.read_parquet(tables / "structure.parquet")["n"],
        all_cells=pd.read_parquet(tables / "all_cells.parquet")["n"],
        section_ap=pd.read_parquet(tables / "section_ap.parquet")["ap_ccf_mm"],
    )


def load_sections(out: Path) -> pd.DataFrame:
    """Read the section table ``tables`` wrote."""
    return pd.read_csv(out / "tables" / "sections.csv").set_index("section")


def section_shares(
    counts: MerfishRegionCounts, sections: pd.DataFrame, section: str
) -> RegionShares:
    """Return a MERFISH section's leave-neighbours-out region shares (§17).

    Args:
        counts: The CCF-view count tables.
        sections: The section table (``excluded``: the section and its ± 1
            neighbours in the AP order of all sections).
        section: The scored section.

    Returns:
        The shares without the excluded sections' cells.

    Raises:
        ValueError: If the excluded set does not contain the section or is
            not its neighbourhood in the count tables' AP order.
    """
    excluded = tuple(_split(sections.loc[section, "excluded"]))
    if excluded != counts.neighbours_out(section):
        raise ValueError(f"{section}: excluded {excluded} is not its neighbourhood")
    return counts.shares(excluded)


# --------------------------------------------------------------------------
# Datasets


@dataclass
class Dataset:
    """One scored query: a MERFISH section x gene set, or ag7 / VZG2.

    Attributes:
        key: ``<gene_set>/<section>`` or ``real/<name>``.
        kind: ``merfish`` or ``real``.
        gene_set: The gene set (``real`` for ag7 / VZG2).
        section: The MERFISH section (the dataset name for real data).
        map_dir: The MAP output directory.
        bundle_path: The primary run's bundle.
        cell_ids: Table cells (tidy order).
        class_names, class_labels, class_bp: Unpruned class calls.
        subclass_names, subclass_labels, subclass_bp: Unpruned subclass calls.
        xy: Coordinates (µm).
        allen_class, allen_subclass, acs: Allen labels (MERFISH only).
    """

    key: str
    kind: str
    gene_set: str
    section: str
    map_dir: str
    bundle_path: str
    cell_ids: pd.Index
    class_names: np.ndarray
    class_labels: np.ndarray
    class_bp: np.ndarray
    subclass_names: np.ndarray
    subclass_labels: np.ndarray
    subclass_bp: np.ndarray
    xy: np.ndarray
    allen_class: np.ndarray | None = None
    allen_subclass: np.ndarray | None = None
    acs: np.ndarray | None = None


def _primary(map_dir: Path) -> tuple[Any, Any, Any]:
    from merxen.annotation.pipeline import MAP_MANIFEST_NAME, load_map_manifest

    manifest = load_map_manifest(map_dir / MAP_MANIFEST_NAME)
    if len(manifest.samples) != 1:
        raise ValueError(f"{map_dir}: expected one sample")
    (record,) = manifest.samples.values()
    return manifest, record, record.runs[PRIMARY_RUN]


def _calls(map_dir: Path, run: Any) -> tuple[pd.Index, dict[str, np.ndarray]]:
    from merxen.annotation.mapmycells_engine import level_frame, read_tidy_parquet

    tidy, _ = read_tidy_parquet(map_dir / run.parquet)
    klass = level_frame(tidy, CLASS_LEVEL)
    klass.index = klass.index.astype(str)
    sub = level_frame(tidy, SUBCLASS_LEVEL)
    sub.index = sub.index.astype(str)
    cell_ids = pd.Index(tidy["cell_id"].astype(str).unique())
    klass = klass.reindex(cell_ids)
    sub = sub.reindex(cell_ids)
    return cell_ids, {
        "class_names": klass["name"].astype(object).to_numpy(),
        "class_labels": klass["assignment"].astype(object).to_numpy(),
        "class_bp": klass["bp"].to_numpy(np.float64),
        "subclass_names": sub["name"].astype(object).to_numpy(),
        "subclass_labels": sub["assignment"].astype(object).to_numpy(),
        "subclass_bp": sub["bp"].to_numpy(np.float64),
    }


def load_merfish_dataset(done: Mapping[str, Any], subsamples: Path) -> Dataset:
    """Load a validated stage A mapping with its subsample's Allen labels."""
    map_dir = Path(done["out"])
    _, record, run = _primary(map_dir)
    cell_ids, calls = _calls(map_dir, run)
    section = section_label(str(done["pair"]))
    sub = pd.read_parquet(subsamples / f"{section}.parquet")
    sub["cell_label"] = sub["cell_label"].astype(str)
    sub = sub.set_index("cell_label")
    missing = cell_ids.difference(sub.index)
    if len(missing):
        raise ValueError(
            f"{done['job']}: {len(missing)} table cells not in the subsample"
        )
    sub = sub.reindex(cell_ids)
    xy = sub[["x_section", "y_section"]].to_numpy(np.float64) * 1000.0
    return Dataset(
        key=f"{done['gene_set']}/{section}",
        kind="merfish",
        gene_set=str(done["gene_set"]),
        section=section,
        map_dir=str(map_dir),
        bundle_path=str(run.bundle_path),
        cell_ids=cell_ids,
        xy=xy,
        allen_class=sub["class"].astype(str).to_numpy(),
        allen_subclass=sub["subclass"].astype(str).to_numpy(),
        acs=sub["average_correlation_score"].to_numpy(np.float64),
        **calls,
    )


def load_real_dataset(name: str, map_dir: Path) -> Dataset:
    """Load an M6 MAP output (ag7 / VZG2) with its table cells' coordinates.

    The coordinates are the H5AD's ``obsm["spatial"]`` matched by cell id,
    as ``pipeline._table_xy``.
    """
    import anndata as ad

    from merxen.annotation.pipeline import read_spatial

    _, record, run = _primary(map_dir)
    cell_ids, calls = _calls(map_dir, run)
    xy_all, _ = read_spatial(record.h5ad_path)
    if xy_all is None:
        raise ValueError(f"{record.h5ad_path} has no obsm['spatial']")
    backed = ad.read_h5ad(record.h5ad_path, backed="r")
    try:
        names = pd.Index(backed.obs_names.astype(str))
    finally:
        backed.file.close()
    if len(names) != len(xy_all):
        raise ValueError(f"{record.h5ad_path}: obsm['spatial'] does not fit obs")
    positions = names.get_indexer(cell_ids)
    if (positions < 0).any():
        raise ValueError(f"{name}: table cells absent from {record.h5ad_path}")
    return Dataset(
        key=f"real/{name}",
        kind="real",
        gene_set="real",
        section=name,
        map_dir=str(map_dir),
        bundle_path=str(run.bundle_path),
        cell_ids=cell_ids,
        xy=np.asarray(xy_all[positions], dtype=np.float64),
        **calls,
    )


_TAXONOMY_CACHE: dict[str, WmbTaxonomy] = {}


def taxonomy_for(bundle_path: str) -> WmbTaxonomy:
    """Return (cached) the class / subclass taxonomy of a bundle's tree."""
    if bundle_path not in _TAXONOMY_CACHE:
        from merxen.annotation.mapmycells_engine import MmcBundle

        tree = MmcBundle.from_dir(bundle_path).tree()
        _TAXONOMY_CACHE[bundle_path] = WmbTaxonomy.from_tree(
            tree, class_level=CLASS_LEVEL, subclass_level=SUBCLASS_LEVEL
        )
    return _TAXONOMY_CACHE[bundle_path]


def full_shares(bundle_dir: Path, structure_home: pd.DataFrame) -> RegionShares:
    """The store bundle's shares with the full structure homes (ag7 / VZG2)."""
    return RegionShares.from_tables(
        pd.read_parquet(bundle_dir / REGION_SHARE_FILE),
        pd.read_parquet(bundle_dir / REGION_HOME_FILE),
        structure_home,
    )


# --------------------------------------------------------------------------
# 2. plan


def node_names(nodes: Sequence[DroppedNode]) -> set[str]:
    """Return the names of a drop list's nodes (class and subclass names)."""
    return {node.name for node in nodes}


def in_nodes(
    allen_class: np.ndarray, allen_subclass: np.ndarray, names: set[str]
) -> np.ndarray:
    """Cells whose Allen class or subclass is among the node names."""
    listed = list(names)
    return np.isin(allen_class, listed) | np.isin(allen_subclass, listed)


def evaluate_candidate(
    data: Dataset,
    shares: RegionShares,
    taxonomy: WmbTaxonomy,
    candidate: Candidate,
    *,
    ccf_present: set[str] | None = None,
    ccf_any: set[str] | None = None,
) -> tuple[dict[str, Any], list[DroppedNode], list[str]]:
    """Run the production region step of one candidate on one dataset.

    Args:
        data: The dataset.
        shares: Its region shares (leave-neighbours-out for MERFISH).
        taxonomy: Its bundle's taxonomy.
        candidate: The rule.
        ccf_present: The section's CCF divisions with >= 1% (MERFISH).
        ccf_any: Those with > 0.2% (MERFISH; oracle regions, extras).

    Returns:
        ``(row, nodes, remap_cell_ids)``.
    """
    config = candidate.config()
    plan = plan_region_step(
        SectionRegionsRequest.parse("auto"),
        cell_ids=data.cell_ids,
        class_names=data.class_names,
        class_labels=data.class_labels,
        class_bp=data.class_bp,
        subclass_names=data.subclass_names,
        subclass_labels=data.subclass_labels,
        subclass_bp=data.subclass_bp,
        xy=data.xy,
        shares=shares,
        taxonomy=taxonomy,
        config=config,
    )
    inference = plan.inference
    inferred = tuple(inference.inferred_regions) if inference else ()
    nodes = list(plan.nodes)
    remap_ids = [str(cell) for cell in plan.remap_cell_ids]
    n = len(data.cell_ids)
    levels = plan.dropped_level
    # T2 regions: the present regions; a skipped section's inferred ones.
    t2_regions = (
        plan.present_regions if plan.status != "skipped_few_tiles" else inferred
    )
    unpruned_t2, _ = t2_values(
        data.subclass_names, t2_regions, shares, taxonomy, config
    )
    row: dict[str, Any] = {
        "dataset": data.key,
        "kind": data.kind,
        "gene_set": data.gene_set,
        "section": data.section,
        "candidate": candidate.label,
        "n_changes": candidate.n_changes,
        "status": plan.status,
        "n_table_cells": n,
        "n_confident_neurons": inference.n_confident_neurons if inference else 0,
        "n_assigned_tiles": inference.n_assigned_tiles if inference else 0,
        "tile_counts": json.dumps(inference.tile_counts if inference else {}),
        "inferred": _join(inferred),
        "coupled": _join(inference.coupled_regions if inference else ()),
        "coupling_tiles": json.dumps(
            inference.coupling_tile_counts if inference else {}
        ),
        "present": _join(plan.present_regions),
        "t2_regions": _join(t2_regions),
        "n_nodes": len(nodes),
        "n_class_nodes": sum(node.kind == "class" for node in nodes),
        "nodes": json.dumps([node.model_dump(mode="json") for node in nodes]),
        "drop_sha": drop_sha(nodes),
        "n_remap": int(sum(value is not None for value in levels)),
        "n_remap_class": int(sum(value == "class" for value in levels)),
        "t2_unpruned": unpruned_t2,
    }
    if data.kind == "merfish":
        assert data.allen_class is not None and data.allen_subclass is not None
        assert data.acs is not None and ccf_present is not None and ccf_any is not None
        names = node_names(nodes)
        dropped = in_nodes(data.allen_class, data.allen_subclass, names)
        confident = (data.acs >= ALLEN_ACS_MIN) & np.asarray(
            meets_threshold(np.nan_to_num(data.class_bp, nan=0.0), CLASS_BP_MIN),
            dtype=bool,
        )
        oracle = two_tier_drop_list(sorted(ccf_any), shares, taxonomy, config)
        wrong = names - node_names(oracle)
        row.update(
            {
                "ccf_present": _join(ccf_present),
                "ccf_any": _join(ccf_any),
                "missed": _join(set(ccf_present) - set(inferred)),
                "extra": _join(set(inferred) - set(ccf_any)),
                "n_harm": int((dropped & confident).sum()),
                "harm": float((dropped & confident).sum() / n) if n else 0.0,
                "n_confident_allen": int(confident.sum()),
                "harm_confident": (
                    float((dropped & confident).sum() / confident.sum())
                    if confident.any()
                    else 0.0
                ),
                "indrop_all": float(dropped.mean()) if n else 0.0,
                "oracle_harm": float(
                    in_nodes(data.allen_class, data.allen_subclass, wrong).mean()
                )
                if n
                else 0.0,
                "n_oracle_nodes": len(oracle),
            }
        )
    return row, nodes, remap_ids


def t2_values(
    subclasses: Sequence[object] | np.ndarray,
    regions: Sequence[str],
    shares: RegionShares,
    taxonomy: WmbTaxonomy,
    config: MouseRegionConfig,
) -> tuple[float, float]:
    """Return T2 (MO2's definition) and T2 without the conditionally kept nodes.

    T2 is ``mouse_gate.t2_share`` with the static never-drop classes (MO2 /
    G3 as pre-registered, §17), so a (b) or (c) keep that retains
    region-implausible calls counts; the second value leaves the section's
    kept nodes out as well (``effective_never_drop``; diagnostic).

    Args:
        subclasses: Subclass call per table cell.
        regions: The section's present (T2) regions.
        shares: Its region shares.
        taxonomy: Its bundle's taxonomy.
        config: The candidate rule.

    Returns:
        ``(t2, t2_kept_excluded)``.
    """
    subclass_class = {
        sub: cls for cls, subs in taxonomy.subclasses.items() for sub in subs
    }
    calls = list(subclasses)
    minimum = config.min_merfish_cells_subclass
    static = t2_share(
        calls,
        regions,
        shares,
        subclass_class=subclass_class,
        never_drop_classes=list(config.never_drop_classes),
        min_merfish_cells=minimum,
    )
    classes, kept = effective_never_drop(config, regions)
    kept_excluded = t2_share(
        calls,
        regions,
        shares,
        subclass_class=subclass_class,
        never_drop_classes=sorted(classes),
        min_merfish_cells=minimum,
        never_drop_subclasses=sorted(kept),
    )
    return static, kept_excluded


def _plan_one(task: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    """Plan every candidate on one dataset (a worker process)."""
    kind, spec = task
    out = Path(spec["out"])
    plan_dir = out / "plan"
    counts = load_counts(out)
    ccf_present: set[str] | None = None
    ccf_any: set[str] | None = None
    if kind == "merfish":
        data = load_merfish_dataset(spec["done"], Path(spec["subsamples"]))
        sections = load_sections(out)
        entry = sections.loc[data.section]
        shares = section_shares(counts, sections, data.section)
        ccf_present, ccf_any = (
            set(_split(entry["ccf_present"])),
            set(_split(entry["ccf_any"])),
        )
    else:
        data = load_real_dataset(spec["name"], Path(spec["map_dir"]))
        structure = pd.read_parquet(out / "tables" / "structure_home_full.parquet")
        shares = full_shares(Path(spec["region_share_bundle"]), structure)
    taxonomy = taxonomy_for(data.bundle_path)
    rows: list[dict[str, Any]] = []
    jobs: dict[str, dict[str, Any]] = {}
    for candidate in all_candidates():
        row, nodes, remap_ids = evaluate_candidate(
            data, shares, taxonomy, candidate, ccf_present=ccf_present, ccf_any=ccf_any
        )
        rows.append(row)
        if remap_ids and row["drop_sha"] not in jobs:
            jobs[row["drop_sha"]] = {
                "dataset": data.key,
                "map_dir": data.map_dir,
                "drop_sha": row["drop_sha"],
                "nodes": [node.model_dump(mode="json") for node in nodes],
                "cell_ids": remap_ids,
            }
    safe = data.key.replace("/", "__")
    frame = pd.DataFrame(rows)
    frame.to_parquet(plan_dir / "rows" / f"{safe}.parquet")
    _write_json(plan_dir / "jobs" / f"{safe}.json", list(jobs.values()))
    return {"dataset": data.key, "rows": len(rows), "jobs": len(jobs)}


def plan_command(args: argparse.Namespace) -> int:
    """Plan every candidate on every dataset (parallel over datasets)."""
    out = Path(args.out)
    (out / "plan" / "rows").mkdir(parents=True, exist_ok=True)
    (out / "plan" / "jobs").mkdir(parents=True, exist_ok=True)
    tasks: list[tuple[str, dict[str, Any]]] = []
    done_files = sorted(Path(args.mapping_done).glob("*.json"))
    for path in done_files:
        done = json.loads(path.read_text())
        if args.gene_set and done["gene_set"] not in args.gene_set:
            continue
        if args.section and section_label(done["pair"]) not in args.section:
            continue
        tasks.append(
            ("merfish", {"out": str(out), "done": done, "subsamples": args.subsamples})
        )
    for item in args.real or []:
        name, _, map_dir = item.partition("=")
        tasks.append(
            (
                "real",
                {
                    "out": str(out),
                    "name": name,
                    "map_dir": map_dir,
                    "region_share_bundle": args.region_share_bundle,
                },
            )
        )
    start = time.monotonic()
    results = []
    with ProcessPoolExecutor(max_workers=args.processes) as pool:
        for result in pool.map(_plan_one, tasks):
            results.append(result)
            logger.info("planned %s", result)
    _write_json(
        out / "plan" / "plan_summary.json",
        {
            "datasets": results,
            "n_candidates": len(all_candidates()),
            "wall_s": round(time.monotonic() - start, 1),
        },
    )
    return 0


# --------------------------------------------------------------------------
# 3. remap


def remap_dir(out: Path, dataset: str, sha: str) -> Path:
    """Return the directory of one re-map job."""
    return out / "remap" / dataset.replace("/", "__") / sha[:16]


def run_remap_jobs(
    jobs: Sequence[Mapping[str, Any]],
    out: Path,
    *,
    gene_csv: str,
    n_processors: int | None,
    scratch: Path,
) -> list[dict[str, Any]]:
    """Run the production pruned re-maps of one dataset's jobs.

    As ``pipeline._remap_cells``: the query of the MAP run (fingerprint
    checked), the re-map cells' rows, the pruned lookup of the run's bundle
    and ``--nodes_to_drop``, and the run's engine parameters **including its
    process count**: ``cell_type_mapper``'s bootstrap draws depend on it, so
    only the MAP run's ``n_processors`` reproduces the production re-map
    (``n_processors`` overrides it for tests only).
    """
    from merxen.annotation.config import AnnotationConfig
    from merxen.annotation.mapmycells_engine import (
        MmcBundle,
        MmcEngineParams,
        run_mmc,
        write_query_h5ad,
    )
    from merxen.annotation.mouse_regions import pruned_lookup
    from merxen.annotation.panel import load_annotation_panel
    from merxen.annotation.pipeline import MapSample, build_sample_query, load_samples
    from merxen.annotation.reference import lookup_sha256, read_lookup

    pending = [
        job
        for job in jobs
        if not (remap_dir(out, job["dataset"], job["drop_sha"]) / "done.json").is_file()
    ]
    if not pending:
        return []
    map_dir = Path(pending[0]["map_dir"])
    manifest, record, run = _primary(map_dir)
    config = AnnotationConfig.model_validate(
        {"species": "mouse", "panel": {"gene_id_fallback_csv": gene_csv}}
    ).coupled_to_clustering(manifest.min_counts)
    (loaded,) = load_samples(
        [
            MapSample(
                record.sample_id, record.platform, Path(record.h5ad_path), record.source
            )
        ],
        config,
        min_counts=manifest.min_counts,
    )
    panel = load_annotation_panel(map_dir / "panel" / "panel_genes.json")
    query = build_sample_query(loaded, panel)
    if query.fingerprint != run.query_fingerprint:
        raise ValueError(f"{map_dir}: the query differs from the MAP run's")
    bundle = MmcBundle.from_dir(run.bundle_path)
    if bundle.build_hash != run.build_hash:
        raise ValueError(f"{map_dir}: bundle build_hash differs from the run's")
    lookup = read_lookup(bundle.lookup)
    tree = bundle.tree()
    engine = dict(run.engine_params)
    if n_processors is not None:
        logger.warning(
            "re-mapping with %d processes, not the MAP run's %s: not the "
            "production re-map",
            n_processors,
            engine.get("n_processors"),
        )
        engine["n_processors"] = n_processors
    params = MmcEngineParams.model_validate(engine)
    results = []
    for job in pending:
        target = remap_dir(out, job["dataset"], job["drop_sha"])
        target.mkdir(parents=True, exist_ok=True)
        nodes = [DroppedNode.model_validate(node) for node in job["nodes"]]
        remap_ids = pd.Index([str(cell) for cell in job["cell_ids"]])
        positions = query.cell_ids.get_indexer(remap_ids)
        if (positions < 0).any():
            raise ValueError(f"{job['dataset']}: re-map cells are not table cells")
        validation = pruned_lookup(lookup, tree, nodes, query.gene_ids)
        work = (
            scratch / f"{remap_dir(out, job['dataset'], job['drop_sha']).parent.name}_"
            f"{job['drop_sha'][:16]}_{os.getpid()}"
        )
        work.mkdir(parents=True, exist_ok=False)
        lookup_path = work / "lookup.pruned.json"
        lookup_path.write_text(json.dumps(validation.lookup) + "\n", "utf-8")
        query_path = write_query_h5ad(
            query.counts[positions], remap_ids, query.gene_ids, work / "query.h5ad"
        )
        parquet = target / "remap.parquet"
        result = run_mmc(
            query_path,
            bundle,
            params,
            output_parquet=parquet,
            work_dir=work / "ctm",
            expected_ctm_version=config.ctm_version,
            lookup_path=lookup_path,
            nodes_to_drop=[node.as_pair() for node in nodes],
            log_dir=target / "logs",
            run_metadata={
                "purpose": "m6b_ap_axis_pruned_remap",
                "dataset": job["dataset"],
                "nodes_to_drop": [list(node.as_pair()) for node in nodes],
            },
        )
        shutil.rmtree(work, ignore_errors=True)
        payload = {
            "dataset": job["dataset"],
            "drop_sha": job["drop_sha"],
            "n_cells": result.n_cells,
            "n_nodes": len(nodes),
            "lookup_sha256": lookup_sha256(validation.lookup),
            "collapsed_parents": [parent.key for parent in validation.collapsed],
            "parquet_sha256": _sha256(parquet),
            "wall_s": result.wall_s,
            "peak_rss_gb": result.peak_rss_gb,
            "ctm_version": result.ctm_version,
        }
        _write_json(target / "done.json", payload)
        results.append(payload)
        logger.info(
            "re-mapped %s %s: %d cells",
            job["dataset"],
            job["drop_sha"][:12],
            result.n_cells,
        )
    return results


def remap_command(args: argparse.Namespace) -> int:
    """Run this worker's share of the re-map jobs (datasets round-robin)."""
    out = Path(args.out)
    job_files = sorted((out / "plan" / "jobs").glob("*.json"))
    mine = [
        path
        for index, path in enumerate(job_files)
        if index % args.n_workers == args.worker
    ]
    scratch = Path(args.scratch)
    for path in mine:
        if (out / "STOP").exists():
            logger.info("STOP file present")
            return 0
        jobs = json.loads(path.read_text())
        try:
            run_remap_jobs(
                jobs,
                out,
                gene_csv=args.gene_id_fallback_csv,
                n_processors=args.n_processors,
                scratch=scratch,
            )
        except Exception:  # noqa: BLE001 - logged; the next dataset still runs
            logger.exception("re-map jobs of %s failed", path.name)
    return 0


# --------------------------------------------------------------------------
# 4. summarise


def pruned_subclasses(
    data: Dataset, remap: pd.DataFrame | None
) -> tuple[np.ndarray, np.ndarray]:
    """Return the pruned class and subclass names (re-mapped cells replaced)."""
    from merxen.annotation.mapmycells_engine import level_frame

    classes = data.class_names.copy()
    subclasses = data.subclass_names.copy()
    if remap is None or remap.empty:
        return classes, subclasses
    for level, target in ((CLASS_LEVEL, classes), (SUBCLASS_LEVEL, subclasses)):
        frame = level_frame(remap, level)
        frame.index = frame.index.astype(str)
        positions = data.cell_ids.get_indexer(frame.index)
        if (positions < 0).any():
            raise ValueError(f"{data.key}: re-map names unknown cells")
        target[positions] = frame["name"].astype(object).to_numpy()
    return classes, subclasses


def read_remap(out: Path, dataset: str, sha: str) -> pd.DataFrame | None:
    """Return a finished re-map's tidy table (``None`` if not run)."""
    from merxen.annotation.mapmycells_engine import read_tidy_parquet

    target = remap_dir(out, dataset, sha)
    if not (target / "done.json").is_file():
        return None
    tidy, _ = read_tidy_parquet(target / "remap.parquet")
    return tidy


def _summarise_one(task: dict[str, Any]) -> list[dict[str, Any]]:
    """T2 after pruning and relabelled cells of one dataset's candidates."""
    out = Path(task["out"])
    rows = pd.read_parquet(task["rows"])
    first = rows.iloc[0]
    counts = load_counts(out)
    if first["kind"] == "merfish":
        sections = load_sections(out)
        shares = section_shares(counts, sections, str(first["section"]))
        done = json.loads(Path(task["done"]).read_text())
        data = load_merfish_dataset(done, Path(task["subsamples"]))
    else:
        structure = pd.read_parquet(out / "tables" / "structure_home_full.parquet")
        shares = full_shares(Path(task["region_share_bundle"]), structure)
        data = load_real_dataset(str(first["section"]), Path(task["map_dir"]))
    taxonomy = taxonomy_for(data.bundle_path)
    cache: dict[str, tuple[np.ndarray, np.ndarray, bool, np.ndarray]] = {}
    class_labels = data.class_labels.astype(str)
    subclass_labels = data.subclass_labels.astype(str)
    result = []
    for record in rows.to_dict("records"):
        sha = record["drop_sha"]
        nodes = [
            DroppedNode.model_validate(node) for node in json.loads(record["nodes"])
        ]
        if sha not in cache:
            remap = read_remap(out, data.key, sha) if record["n_remap"] else None
            missing = bool(record["n_remap"]) and remap is None
            classes, subclasses = pruned_subclasses(data, remap)
            # Cells whose unpruned class or subclass node is dropped (the
            # re-mapped cells; ``mouse_regions.dropped_levels``, vectorised).
            in_drop = np.isin(
                class_labels, [node.node for node in nodes if node.kind == "class"]
            ) | np.isin(
                subclass_labels,
                [node.node for node in nodes if node.kind == "subclass"],
            )
            if int(in_drop.sum()) != int(record["n_remap"]):
                raise ValueError(f"{data.key} {sha[:12]}: re-map cell count differs")
            cache[sha] = (classes, subclasses, missing, in_drop)
        classes, subclasses, missing, in_drop = cache[sha]
        candidate = candidate_by_label(str(record["candidate"]))
        config = candidate.config()
        regions = _split(record["t2_regions"])
        record["remap_missing"] = missing
        t2, t2_kept = (
            (np.nan, np.nan)
            if missing
            else t2_values(subclasses, regions, shares, taxonomy, config)
        )
        record["t2_after"] = t2
        record["t2_after_kept_excluded"] = t2_kept
        class_changed = classes.astype(object) != data.class_names.astype(object)
        subclass_changed = subclasses.astype(object) != data.subclass_names.astype(
            object
        )
        n = len(data.cell_ids)
        record["n_class_relabelled"] = int((class_changed & in_drop).sum())
        record["n_subclass_relabelled"] = int((subclass_changed & in_drop).sum())
        record["class_changed_outside"] = float((class_changed & ~in_drop).sum() / n)
        record["subclass_changed_outside"] = float(
            (subclass_changed & ~in_drop).sum() / n
        )
        record["relabelled"] = record["n_remap"] / n
        record["pruned_in_dropped_node"] = (
            int(
                (
                    in_nodes(
                        classes.astype(str), subclasses.astype(str), node_names(nodes)
                    )
                    & in_drop
                ).sum()
            )
            if not missing
            else -1
        )
        result.append(record)
    return result


def per_bin_summary(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    above: float = SECTION_ABOVE,
) -> pd.DataFrame:
    """Return per gene set x candidate x AP bin: mean, max, sections above.

    Args:
        frame: One row per section (``gene_set``, ``candidate``, ``ap_bin``,
            ``missed``, and the value columns).
        columns: Value columns (fractions).
        above: The "sections above" threshold (0.1%).

    Returns:
        One row per group and bin with ``n_sections``,
        ``n_sections_missing_region``, and ``<column>_mean``, ``_max``,
        ``_n_above`` per column.
    """
    rows = []
    keys = ["gene_set", "candidate", "ap_bin"]
    for group, part in frame.groupby(keys, sort=False):
        row: dict[str, Any] = dict(zip(keys, group, strict=True))
        row["n_sections"] = len(part)
        row["n_sections_missing_region"] = int((part["missed"].fillna("") != "").sum())
        for column in columns:
            values = pd.to_numeric(part[column], errors="coerce")
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_max"] = float(values.max())
            row[f"{column}_n_above"] = int((values > above).sum())
        rows.append(row)
    order = {label: index for index, label in enumerate(AP_BIN_LABELS)}
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["_order"] = table["ap_bin"].map(order)
    return table.sort_values(["gene_set", "candidate", "_order"]).drop(columns="_order")


def candidate_table(
    merfish: pd.DataFrame, real: pd.DataFrame, bins: pd.DataFrame
) -> pd.DataFrame:
    """Return one row per candidate: S, the constraints and the tie keys."""
    v1 = merfish[merfish["candidate"] == "v1"].set_index(["dataset"])["nodes"]
    v1_pairs = {
        dataset: {
            tuple(item[k] for k in ("level", "node")) for item in json.loads(nodes)
        }
        for dataset, nodes in v1.items()
    }
    rows = []
    for label, part in merfish.groupby("candidate", sort=False):
        candidate = candidate_by_label(str(label))
        differing = 0
        for record in part[["dataset", "nodes"]].to_dict("records"):
            pairs = {
                tuple(item[k] for k in ("level", "node"))
                for item in json.loads(record["nodes"])
            }
            differing += len(pairs ^ v1_pairs.get(record["dataset"], set()))
        cand_bins = bins[bins["candidate"] == label]
        t2_ok = (
            bool((cand_bins["t2_after_mean"] <= MO10_T2_BIN_MEAN_MAX).all())
            and not cand_bins["t2_after_mean"].isna().any()
        )
        real_part = real[real["candidate"] == label]
        mo2: dict[str, bool | None] = {}
        for name in REAL_DATASETS:
            entry = real_part[real_part["section"] == name]
            if entry.empty:
                mo2[name] = None
                continue
            item = entry.iloc[0]
            mo2[name] = bool(
                item["t2_after"] <= MO2_T2_MAX
                and item["class_changed_outside"] <= MO2_CLASS_OUTSIDE_MAX
                and item["subclass_changed_outside"] <= MO2_SUBCLASS_OUTSIDE_MAX
            )
        worst = part.loc[part["harm"].idxmax()]
        rows.append(
            {
                "candidate": label,
                "is_reference": label == "v1",
                "S_max_harm": float(part["harm"].max()),
                "S_at": f"{worst['gene_set']}/{worst['section']}",
                "mean_harm": float(part["harm"].mean()),
                "max_oracle_harm": float(part["oracle_harm"].max()),
                "max_t2_bin_mean": float(cand_bins["t2_after_mean"].max()),
                "t2_ok": t2_ok,
                "mo2_ag7": mo2.get("ag7"),
                "mo2_vzg2": mo2.get("vzg2"),
                "mo2_ok": all(value is True for value in mo2.values()),
                "admissible": t2_ok and all(value is True for value in mo2.values()),
                "n_changes": candidate.n_changes,
                "n_drop_decisions_differing": differing,
                "n_sections_skipped": int(
                    (part["status"] == "skipped_few_tiles").sum()
                ),
                "n_sections_missed": int((part["missed"].fillna("") != "").sum()),
                "n_remap_missing": int(
                    part.get("remap_missing", pd.Series(False)).sum()
                ),
                "ag7_regions": real_part[real_part["section"] == "ag7"][
                    "inferred"
                ].tolist(),
                "vzg2_regions": real_part[real_part["section"] == "vzg2"][
                    "inferred"
                ].tolist(),
            }
        )
    return pd.DataFrame(rows)


def select_candidate(table: pd.DataFrame) -> dict[str, Any]:
    """Apply the pre-registered selection rule (§17) to the candidate table.

    Returns:
        ``{"selected": label or None, "reason": ..., "ties": [...]}``.
    """
    candidates = table[~table["is_reference"]]
    admissible = candidates[candidates["admissible"].astype(bool)]
    if admissible.empty:
        return {
            "selected": None,
            "reason": (
                "no admissible candidate: v1 stays and the choice goes to the user"
            ),
            "ties": [],
        }
    best = admissible["S_max_harm"].min()
    tied = admissible[admissible["S_max_harm"] == best].sort_values(
        ["n_changes", "n_drop_decisions_differing", "candidate"]
    )
    chosen = tied.iloc[0]
    return {
        "selected": str(chosen["candidate"]),
        "S_max_harm": float(best),
        "reason": "lowest S among admissible candidates"
        + (
            f"; tie among {len(tied)} broken by fewer changes, fewer differing drop "
            "decisions, then the label"
            if len(tied) > 1
            else ""
        ),
        "ties": tied["candidate"].tolist(),
    }


def summarise_command(args: argparse.Namespace) -> int:
    """Compute T2 after pruning, the bin tables, the candidates, the selection."""
    out = Path(args.out)
    summary = out / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    done_by_key = {}
    for path in sorted(Path(args.mapping_done).glob("*.json")):
        done = json.loads(path.read_text())
        done_by_key[f"{done['gene_set']}/{section_label(done['pair'])}"] = str(path)
    real_dirs = dict(item.partition("=")[::2] for item in (args.real or []))
    tasks = []
    for path in sorted((out / "plan" / "rows").glob("*.parquet")):
        key = path.stem.replace("__", "/")
        task: dict[str, Any] = {"out": str(out), "rows": str(path)}
        if key.startswith("real/"):
            name = key.split("/", 1)[1]
            task.update(
                {
                    "map_dir": real_dirs[name],
                    "region_share_bundle": args.region_share_bundle,
                }
            )
        else:
            task.update({"done": done_by_key[key], "subsamples": args.subsamples})
        tasks.append(task)
    records: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.processes) as pool:
        for part in pool.map(_summarise_one, tasks):
            records.extend(part)
    frame = pd.DataFrame(records)
    sections = load_sections(out)
    merfish = frame[frame["kind"] == "merfish"].copy()
    merfish["ap_ccf_mm"] = merfish["section"].map(sections["ap_ccf_mm"])
    merfish["ap_bin"] = merfish["section"].map(sections["ap_bin"])
    merfish["in_scope"] = merfish["section"].map(sections["in_scope"]).astype(bool)
    merfish = merfish[merfish["in_scope"]]
    real = frame[frame["kind"] == "real"].copy()
    merfish.to_parquet(summary / "sections.parquet")
    merfish.drop(columns=["nodes"]).to_csv(summary / "sections.csv", index=False)
    real.to_csv(summary / "real.csv", index=False)
    bins = per_bin_summary(
        merfish,
        ["harm", "harm_confident", "oracle_harm", "t2_after", "relabelled", "n_nodes"],
    )
    bins.to_csv(summary / "bins.csv", index=False)
    table = candidate_table(merfish, real, bins)
    table.to_csv(summary / "candidates.csv", index=False)
    selection = select_candidate(table)
    selection["n_sections"] = int(merfish["section"].nunique())
    selection["gene_sets"] = sorted(merfish["gene_set"].unique())
    selection["n_rows"] = len(merfish)
    selection["complete"] = bool(
        selection["n_sections"] == 38
        and len(selection["gene_sets"]) == 3
        and not merfish["remap_missing"].any()
        and not real["remap_missing"].any()
    )
    _write_json(summary / "selection.json", selection)
    _write_text(summary / "CANDIDATES.txt", candidates_report(table, selection, bins))
    logger.info("selection: %s", selection)
    return 0


def candidates_report(
    table: pd.DataFrame, selection: Mapping[str, Any], bins: pd.DataFrame
) -> str:
    """Plain-text candidate table and selection."""
    lines = [
        "M6b candidate table (pre-registration §17; S = max direct harm over "
        "38 sections x 3 gene sets)",
        f"selection: {selection.get('selected')} ({selection.get('reason')}); "
        f"complete={selection.get('complete')}",
        "",
    ]
    view = table.sort_values(
        ["admissible", "S_max_harm", "n_changes"], ascending=[False, True, True]
    )
    columns = [
        "candidate",
        "S_max_harm",
        "S_at",
        "mean_harm",
        "max_t2_bin_mean",
        "t2_ok",
        "mo2_ag7",
        "mo2_vzg2",
        "admissible",
        "n_changes",
        "n_drop_decisions_differing",
        "n_sections_skipped",
        "n_sections_missed",
        "max_oracle_harm",
    ]
    lines.append(
        view[columns].to_string(index=False, float_format=lambda v: f"{v:.6f}")
    )
    chosen = selection.get("selected") or "v1"
    lines += ["", f"per-AP-bin table of {chosen} and v1:"]
    lines.append(
        bins[bins["candidate"].isin([chosen, "v1"])].to_string(
            index=False, float_format=lambda v: f"{v:.5f}"
        )
    )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# 5. mo10


def mo10_verdict(sections: pd.DataFrame, bins: pd.DataFrame) -> dict[str, Any]:
    """Return MO10 (§9) for one candidate, per gene set and overall.

    Args:
        sections: The candidate's section rows (all gene sets).
        bins: Its per-bin rows.

    Returns:
        Per gene set: bin means, section maximum, T2 bin means, missed
        regions with harm above 0.1%; ``pass`` overall when every gene set
        passes.
    """
    result: dict[str, Any] = {"gene_sets": {}}
    for gene_set, part in sections.groupby("gene_set"):
        gene_bins = bins[bins["gene_set"] == gene_set]
        bad_missed = part[
            (part["missed"].fillna("") != "") & (part["harm"] > MO10_MISSED_HARM_MAX)
        ]
        checks = {
            "bin_mean_harm_ok": bool(
                (gene_bins["harm_mean"] <= MO10_BIN_MEAN_HARM_MAX).all()
            ),
            "section_max_harm_ok": bool(part["harm"].max() <= MO10_SECTION_MAX_HARM),
            "t2_bin_mean_ok": bool(
                (gene_bins["t2_after_mean"] <= MO10_T2_BIN_MEAN_MAX).all()
            ),
            "missed_only_where_harm_small": bad_missed.empty,
        }
        result["gene_sets"][gene_set] = {
            **checks,
            "pass": all(checks.values()),
            "max_section_harm": float(part["harm"].max()),
            "max_bin_mean_harm": float(gene_bins["harm_mean"].max()),
            "max_t2_bin_mean": float(gene_bins["t2_after_mean"].max()),
            "missed_with_harm_above": bad_missed[["section", "missed", "harm"]].to_dict(
                "records"
            ),
            "max_oracle_harm": float(part["oracle_harm"].max()),
        }
    result["pass"] = bool(result["gene_sets"]) and all(
        value["pass"] for value in result["gene_sets"].values()
    )
    return result


def mo10_command(args: argparse.Namespace) -> int:
    """Score MO10 on the selected candidate, once the pre-registration names it."""
    out = Path(args.out)
    selection = json.loads((out / "summary" / "selection.json").read_text())
    chosen = selection.get("selected") or "v1"
    text = Path(args.preregistration).read_text()
    marker = f"Selected candidate (M6b): `{chosen}`"
    if marker not in text:
        raise SystemExit(
            f"the pre-registration does not record {marker!r}; commit the "
            "selection first (§17)"
        )
    sections = pd.read_parquet(out / "summary" / "sections.parquet")
    bins = pd.read_csv(out / "summary" / "bins.csv")
    verdict = mo10_verdict(
        sections[sections["candidate"] == chosen], bins[bins["candidate"] == chosen]
    )
    verdict["candidate"] = chosen
    real = pd.read_csv(out / "summary" / "real.csv")
    rows = real[real["candidate"] == chosen]
    verdict["mo2"] = rows[
        ["section", "t2_after", "class_changed_outside", "subclass_changed_outside"]
    ].to_dict("records")
    verdict["mo3"] = {
        str(row["section"]): set(_split(row["inferred"]))
        == set(MANUAL_REGIONS[str(row["section"])])
        for row in rows.to_dict("records")
    }
    _write_json(out / "mo10" / "mo10.json", verdict)
    logger.info("MO10: %s", verdict["pass"])
    return 0


# --------------------------------------------------------------------------
# 6. ap-estimate


def ap_estimate_command(args: argparse.Namespace) -> int:
    """Validate the AP estimate on held-out MERFISH sections."""
    out = Path(args.out)
    target = out / "ap_estimate"
    target.mkdir(parents=True, exist_ok=True)
    composition = pd.read_parquet(
        Path(args.region_share_bundle) / SECTION_COMPOSITION_FILE
    )
    sections = load_sections(out)
    steps = sections["section_step_mm"]
    rows = []
    allen = pd.read_parquet(out / "tables" / "allen_confident_neurons.parquet")
    for section, part in allen.groupby("brain_section_label"):
        names = np.repeat(part["subclass"].to_numpy(), part["n"].to_numpy())
        rows.append(
            _estimate_row("allen", str(section), names, composition, sections, steps)
        )
    for path in sorted(Path(args.mapping_done).glob("*.json")):
        done = json.loads(path.read_text())
        data = load_merfish_dataset(done, Path(args.subsamples))
        from merxen.annotation.mouse_regions import confident_neurons

        mask = confident_neurons(
            data.class_names, data.class_bp, data.subclass_bp, min_bp=0.9
        )
        rows.append(
            _estimate_row(
                data.gene_set,
                data.section,
                data.subclass_names[mask],
                composition,
                sections,
                steps,
            )
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(target / "heldout.csv", index=False)
    summary = (
        frame.groupby(["query", "ap_bin"], dropna=False)
        .agg(
            n=("section", "size"),
            mean_abs_error_mm=("abs_error_mm", "mean"),
            max_abs_error_mm=("abs_error_mm", "max"),
            within_step=("within_step", "mean"),
            bin_correct=("bin_correct", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(target / "heldout_summary.csv", index=False)
    _write_text(
        target / "AP_ESTIMATE.txt",
        "AP estimate on held-out MERFISH-638850 sections\n"
        + summary.to_string(index=False)
        + "\n",
    )
    return 0


def _estimate_row(
    query: str,
    section: str,
    names: Sequence[object] | np.ndarray,
    composition: pd.DataFrame,
    sections: pd.DataFrame,
    steps: pd.Series,
) -> dict[str, Any]:
    truth = float(sections.loc[section, "ap_ccf_mm"])
    try:
        found = estimate_ap(names, composition, exclude_sections=[section])
    except ApEstimateError as problem:
        return {
            "query": query,
            "section": section,
            "ap_ccf_mm": truth,
            "error": str(problem),
        }
    error = found.ap_mm - truth
    return {
        "query": query,
        "section": section,
        "ap_ccf_mm": truth,
        "ap_bin": ap_bin(truth),
        "in_scope": bool(sections.loc[section, "in_scope"]),
        "estimate_section": found.section,
        "estimate_ap_mm": found.ap_mm,
        "weighted_ap_mm": found.weighted_ap_mm,
        "similarity": found.similarity,
        "n_cells": found.n_cells,
        "error_mm": error,
        "abs_error_mm": abs(error),
        "section_step_mm": float(steps[section]),
        "within_step": abs(error) <= float(steps[section]) + 1e-9,
        "bin_correct": found.ap_bin == ap_bin(truth),
    }


# --------------------------------------------------------------------------
# CLI


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)

    tables = commands.add_parser("tables", help="count tables and checks")
    tables.add_argument("--ccf-metadata", required=True)
    tables.add_argument("--region-share-bundle", required=True)
    tables.add_argument("--section-table", required=True)
    tables.add_argument("--out", required=True)

    plan = commands.add_parser("plan", help="region step of every candidate")
    plan.add_argument("--out", required=True)
    plan.add_argument("--mapping-done", required=True)
    plan.add_argument("--subsamples", required=True)
    plan.add_argument("--real", action="append", help="NAME=MAP_DIR (ag7, vzg2)")
    plan.add_argument("--region-share-bundle", required=True)
    plan.add_argument("--gene-set", action="append")
    plan.add_argument("--section", action="append")
    plan.add_argument("--processes", type=int, default=6)

    remap = commands.add_parser("remap", help="production pruned re-maps")
    remap.add_argument("--out", required=True)
    remap.add_argument("--gene-id-fallback-csv", required=True)
    remap.add_argument("--worker", type=int, default=0)
    remap.add_argument("--n-workers", type=int, default=1)
    remap.add_argument(
        "--n-processors",
        type=int,
        default=None,
        help="override the MAP run's process count (not the production re-map)",
    )
    remap.add_argument("--scratch", required=True)

    summarise = commands.add_parser("summarise", help="T2, bins, candidates, selection")
    summarise.add_argument("--out", required=True)
    summarise.add_argument("--mapping-done", required=True)
    summarise.add_argument("--subsamples", required=True)
    summarise.add_argument("--real", action="append")
    summarise.add_argument("--region-share-bundle", required=True)
    summarise.add_argument("--processes", type=int, default=6)

    mo10 = commands.add_parser("mo10", help="MO10 on the recorded selection")
    mo10.add_argument("--out", required=True)
    mo10.add_argument("--preregistration", required=True)

    estimate = commands.add_parser("ap-estimate", help="held-out AP estimate")
    estimate.add_argument("--out", required=True)
    estimate.add_argument("--region-share-bundle", required=True)
    estimate.add_argument("--mapping-done", required=True)
    estimate.add_argument("--subsamples", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a subcommand."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = build_parser().parse_args(argv)
    handlers = {
        "tables": tables_command,
        "plan": plan_command,
        "remap": remap_command,
        "summarise": summarise_command,
        "mo10": mo10_command,
        "ap-estimate": ap_estimate_command,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())

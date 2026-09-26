#!/usr/bin/env python
"""Build MerXen's packaged annotation vocabulary and floor tables.

Reads the Allen taxonomy CSVs (the paths are arguments; nothing local is
hard-coded) and ``src/merxen/assets/annotation/overrides.yaml``, and writes:

- ``whb_supercluster_vocab.csv``: the 31 WHB superclusters (``label, name,
  broad_class, lineage, nt, sink, region_plausible_frontal_cortex``);
- ``seaad_mr_subclass_vocab.csv``: the 29 SEA-AD Multiregion subclasses plus
  supertype-prefix overrides (``label, subclass, supertype_prefix,
  broad_class, broad_class_any, lineage, nt, sink,
  region_plausible_frontal_cortex``);
- ``wmb_class_vocab.csv``: the 34 WMB classes (``label, class, broad_class,
  nt, never_drop``);
- ``floors_human.csv``, ``floors_mouse.csv``: the §5.4 v1 count floors;
- ``NOTICE``: provenance (taxonomy releases and input checksums) and licences.

Usage::

    python scripts/annotation/build_vocab_tables.py \\
        --whb-taxonomy-dir <whb_abc>/metadata/WHB-taxonomy/20240330 \\
        --seaad-term-csv <seaad_terms_csv> \\
        --wmb-taxonomy-dir <wmb_abc>/metadata/WMB-taxonomy/20231215

where ``<seaad_terms_csv>`` is the ABC
``metadata/SEA-AD-Multiregion-taxonomy/20260711/cluster_annotation_term.csv``
(a local copy may have another name). The NOTICE records each input by its
ABC path, size and sha256, never by its local path or name, so ``--check``
depends only on the content of the inputs.

``--check`` writes nothing and exits non-zero when a committed file differs
from what the inputs produce. The generator fails loudly when a taxonomy term
has no curated mapping (or a mapping names a term the taxonomy lacks), so a new
Allen release cannot silently change the vocabulary.

Needs PyYAML. It is not declared in ``pyproject.toml`` yet; the pinned
environment gets it through dask and pre-commit, and it is to be declared at
the next planned lockfile regeneration (regenerating the lock now would
invalidate legacy ``-resume``; ``docs/development.md``). The import is
deferred to ``build_all`` so a missing PyYAML fails with that explanation.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import logging
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from merxen.annotation.vocab import (
    BROAD_CLASS_ANY_COLUMN,
    BROAD_CLASS_ANY_SEPARATOR,
    BROAD_CLASSES,
    HUMAN_BROAD_CLASSES,
    HUMAN_FLOOR_CLASSES,
    HUMAN_LINEAGE_OF_BROAD_CLASS,
    HUMAN_LINEAGES,
    NEURONS,
    NT_OTHER,
    UNASSIGNED_LABEL,
    classify_neurotransmitter,
)

logger = logging.getLogger("build_vocab_tables")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASSET_DIR = REPO_ROOT / "src" / "merxen" / "assets" / "annotation"
REGION = "frontal_cortex"
PLATFORMS = ("MERSCOPE", "XENIUM")
WHB_TERM_FILE = "cluster_annotation_term.csv"
WHB_MEMBERSHIP_FILE = "cluster_to_cluster_annotation_membership.csv"
WHB_CLUSTER_FILE = "cluster.csv"
WMB_TERM_FILE = "cluster_annotation_term.csv"
WMB_MEMBERSHIP_FILE = "cluster_to_cluster_annotation_membership.csv"
# ABC file name of the SEA-AD terms; a local copy may be named otherwise.
SEAAD_TERM_FILE = "cluster_annotation_term.csv"
WHB_COUNT = 31
SEAAD_SUBCLASS_COUNT = 29
WMB_CLASS_COUNT = 34


class VocabBuildError(ValueError):
    """The taxonomy inputs and ``overrides.yaml`` do not agree."""


@dataclass(frozen=True)
class InputFile:
    """Identity of one taxonomy input, recorded in the NOTICE.

    Attributes:
        role: What the file provides, e.g. ``"WHB terms"``.
        name: The file's ABC path below ``metadata/``
            (``<taxonomy>/<release>/<file>``), taken from ``overrides.yaml``
            and not from the local path, so the NOTICE depends only on the
            file's content.
        size: Size in bytes.
        sha256: Hex sha256 of the file.
    """

    role: str
    name: str
    size: int
    sha256: str


@dataclass(frozen=True)
class BuildResult:
    """Generated file contents, keyed by file name.

    Attributes:
        files: File name to text.
        inputs: Identities of the taxonomy inputs used.
    """

    files: dict[str, str]
    inputs: tuple[InputFile, ...]


def _identify(
    path: Path, role: str, spec: Mapping[str, Any], abc_file_name: str
) -> InputFile:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    name = f"{spec['taxonomy']}/{spec['release']}/{abc_file_name}"
    return InputFile(role, name, path.stat().st_size, digest)


def _require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Taxonomy input not found: {path}")
    return path


def _bool_text(values: pd.Series) -> pd.Series:
    return values.map(lambda value: "true" if bool(value) else "false")


def _to_csv(frame: pd.DataFrame) -> str:
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False, lineterminator="\n")
    return buffer.getvalue()


def _check_set(kind: str, expected: set[str], curated: set[str]) -> None:
    missing = sorted(expected - curated)
    extra = sorted(curated - expected)
    if missing or extra:
        raise VocabBuildError(
            f"{kind}: taxonomy terms without a curated mapping {missing}; "
            f"curated names absent from the taxonomy {extra}"
        )


def _check_mapping(kind: str, name: str, broad: str, lineage: str | None) -> None:
    allowed_broad = {*BROAD_CLASSES["human"], UNASSIGNED_LABEL}
    if broad not in allowed_broad:
        raise VocabBuildError(f"{kind} {name!r}: broad_class {broad!r} not allowed")
    if lineage is None:
        return
    if lineage not in {*HUMAN_LINEAGES, UNASSIGNED_LABEL}:
        raise VocabBuildError(f"{kind} {name!r}: lineage {lineage!r} not allowed")
    if broad != UNASSIGNED_LABEL and HUMAN_LINEAGE_OF_BROAD_CLASS[broad] != lineage:
        raise VocabBuildError(
            f"{kind} {name!r}: lineage {lineage!r} does not contain {broad!r}"
        )


def _broad_class_any(broad: str, override_broads: Sequence[str]) -> str:
    """Return a SEA-AD row's ``broad_class_any`` value.

    A row whose broad class is one of the seven counts toward that class
    only. A base row that the supertype overrides split counts toward every
    class of its overrides (in the order of ``HUMAN_BROAD_CLASSES``), so its
    probability mass is not lost when the supertype is unknown. Other rows
    (sinks, ``Mixed/Unknown``) count toward no class.
    """
    if broad in HUMAN_BROAD_CLASSES:
        return broad
    if broad != UNASSIGNED_LABEL:
        return ""
    classes = set(override_broads) & set(HUMAN_BROAD_CLASSES)
    ordered = [name for name in HUMAN_BROAD_CLASSES if name in classes]
    return BROAD_CLASS_ANY_SEPARATOR.join(ordered)


def majority_nt(
    nt_terms: pd.Series,
    weights: pd.Series,
    *,
    min_share: float,
) -> tuple[str | None, float]:
    """Return a node's majority NT class and its NT-annotated cell share.

    Args:
        nt_terms: Allen NT term per cluster (NaN when unannotated).
        weights: Cells per cluster, aligned with ``nt_terms``.
        min_share: The winning class must hold more than this share of the
            NT-annotated cells, else the result is ``"Other"``.

    Returns:
        ``(nt, annotated_share)``: the NT class (``None`` when no cluster is
        annotated) and the share of cells whose cluster has an NT term.
    """
    classes = nt_terms.map(classify_neurotransmitter)
    annotated = classes.notna()
    total = float(weights.sum())
    annotated_total = float(weights[annotated].sum())
    share = annotated_total / total if total else 0.0
    if not annotated_total:
        return None, share
    by_class = weights[annotated].groupby(classes[annotated]).sum() / annotated_total
    best = str(by_class.idxmax())
    if float(by_class.max()) <= min_share:
        return NT_OTHER, share
    return best, share


def build_whb(
    taxonomy_dir: Path, spec: Mapping[str, Any]
) -> tuple[pd.DataFrame, list[InputFile]]:
    """Build the WHB supercluster vocabulary.

    Args:
        taxonomy_dir: WHB-taxonomy release directory.
        spec: The ``whb`` section of ``overrides.yaml``.

    Returns:
        The table and the identities of the inputs read.

    Raises:
        VocabBuildError: If the taxonomy and the curated mapping disagree.
    """
    term_path = _require(taxonomy_dir / WHB_TERM_FILE)
    membership_path = _require(taxonomy_dir / WHB_MEMBERSHIP_FILE)
    cluster_path = _require(taxonomy_dir / WHB_CLUSTER_FILE)
    terms = pd.read_csv(term_path)
    membership = pd.read_csv(membership_path)
    clusters = pd.read_csv(cluster_path)
    level = spec["supercluster_term_set"]
    supers = terms[terms["cluster_annotation_term_set_label"] == level].sort_values(
        "term_order"
    )
    if len(supers) != WHB_COUNT:
        raise VocabBuildError(
            f"WHB: expected {WHB_COUNT} superclusters, got {len(supers)}"
        )
    curated: Mapping[str, Mapping[str, str]] = spec["superclusters"]
    _check_set("WHB superclusters", set(supers["name"]), set(curated))
    sinks = set(spec["sinks"])
    _check_set("WHB sinks", sinks & set(supers["name"]), sinks)
    implausible = set(spec["regions"][REGION]["implausible_labels"])
    _check_set(
        "WHB implausible labels", implausible & set(supers["label"]), implausible
    )

    set_label = membership["cluster_annotation_term_set_label"]
    super_of = membership.loc[
        set_label == level, ["cluster_alias", "cluster_annotation_term_name"]
    ].rename(columns={"cluster_annotation_term_name": "supercluster"})
    nt_of = membership.loc[
        set_label == spec["neurotransmitter_term_set"],
        ["cluster_alias", "cluster_annotation_term_name"],
    ].rename(columns={"cluster_annotation_term_name": "nt_term"})
    joined = super_of.merge(nt_of, on="cluster_alias", how="left").merge(
        clusters[["cluster_alias", "number_of_cells"]], on="cluster_alias", how="left"
    )
    if joined["number_of_cells"].isna().any():
        raise VocabBuildError("WHB: clusters without a cell count in cluster.csv")

    rows = []
    for _, term in supers.iterrows():
        name = str(term["name"])
        mapping = curated[name]
        broad, lineage = mapping["broad_class"], mapping["lineage"]
        _check_mapping("WHB supercluster", name, broad, lineage)
        group = joined[joined["supercluster"] == name]
        nt, share = majority_nt(
            group["nt_term"],
            group["number_of_cells"].astype(float),
            min_share=float(spec["nt_majority_min_share"]),
        )
        is_neuronal = lineage == NEURONS
        if is_neuronal != (share > 0.5):
            raise VocabBuildError(
                f"WHB supercluster {name!r}: lineage {lineage!r} but "
                f"{share:.1%} of its cells carry an NT annotation"
            )
        rows.append(
            {
                "label": str(term["label"]),
                "name": name,
                "broad_class": broad,
                "lineage": lineage,
                "nt": nt if is_neuronal else "",
                "sink": name in sinks,
                f"region_plausible_{REGION}": str(term["label"]) not in implausible,
            }
        )
    frame = pd.DataFrame(rows)
    for column in ("sink", f"region_plausible_{REGION}"):
        frame[column] = _bool_text(frame[column])
    inputs = [
        _identify(term_path, "WHB terms", spec, WHB_TERM_FILE),
        _identify(membership_path, "WHB cluster membership", spec, WHB_MEMBERSHIP_FILE),
        _identify(cluster_path, "WHB clusters", spec, WHB_CLUSTER_FILE),
    ]
    return frame, inputs


def build_seaad(
    term_csv: Path, spec: Mapping[str, Any]
) -> tuple[pd.DataFrame, list[InputFile]]:
    """Build the SEA-AD Multiregion subclass vocabulary with supertype overrides.

    Args:
        term_csv: SEA-AD Multiregion ``cluster_annotation_term.csv``.
        spec: The ``seaad`` section of ``overrides.yaml``.

    Returns:
        The table and the identity of the input read.

    Raises:
        VocabBuildError: If the taxonomy and the curated mapping disagree.
    """
    terms = pd.read_csv(_require(term_csv))
    set_label = terms["cluster_annotation_term_set_label"]
    subclasses = terms[set_label == spec["subclass_term_set"]].sort_values("term_order")
    supertypes = terms[set_label == spec["supertype_term_set"]]
    if len(subclasses) != SEAAD_SUBCLASS_COUNT:
        raise VocabBuildError(
            f"SEA-AD: expected {SEAAD_SUBCLASS_COUNT} subclasses, got {len(subclasses)}"
        )
    neuronal_classes = set(spec["neuronal_classes"])
    curated: Mapping[str, Mapping[str, str]] = spec["subclasses"]
    non_neuronal = set(
        subclasses.loc[~subclasses["parent_term_name"].isin(neuronal_classes), "name"]
    )
    _check_set("SEA-AD non-neuronal subclasses", non_neuronal, set(curated))
    sinks = set(spec["sinks"])
    implausible = set(spec["regions"][REGION]["implausible_subclasses"])
    names = set(subclasses["name"])
    _check_set("SEA-AD sinks", sinks & names, sinks)
    _check_set("SEA-AD implausible subclasses", implausible & names, implausible)
    overrides: Mapping[str, Mapping[str, Mapping[str, str]]] = spec[
        "supertype_overrides"
    ]
    _check_set("SEA-AD override subclasses", set(overrides) & names, set(overrides))

    rows = []
    for _, term in subclasses.iterrows():
        name = str(term["name"])
        parent = str(term["parent_term_name"])
        if parent in neuronal_classes:
            broad, lineage = NEURONS, NEURONS
            nt = classify_neurotransmitter(parent) or ""
        else:
            broad = curated[name]["broad_class"]
            lineage = curated[name]["lineage"]
            nt = ""
        _check_mapping("SEA-AD subclass", name, broad, lineage)
        prefixes = overrides.get(name, {})
        base = {
            "label": str(term["label"]),
            "subclass": name,
            "supertype_prefix": "",
            "broad_class": broad,
            BROAD_CLASS_ANY_COLUMN: _broad_class_any(
                broad, [mapping["broad_class"] for mapping in prefixes.values()]
            ),
            "lineage": lineage,
            "nt": nt,
            "sink": name in sinks,
            f"region_plausible_{REGION}": name not in implausible,
        }
        rows.append(base)
        if not prefixes:
            continue
        children = supertypes.loc[
            supertypes["parent_term_label"] == term["label"], "name"
        ]
        for child in children:
            matches = [prefix for prefix in prefixes if str(child).startswith(prefix)]
            if len(matches) != 1:
                raise VocabBuildError(
                    f"SEA-AD supertype {child!r} of {name!r} matches override "
                    f"prefixes {matches}; expected exactly one"
                )
        for prefix, mapping in prefixes.items():
            if not any(str(child).startswith(prefix) for child in children):
                raise VocabBuildError(
                    f"SEA-AD override prefix {prefix!r} matches no supertype"
                )
            _check_mapping(
                "SEA-AD override", prefix, mapping["broad_class"], mapping["lineage"]
            )
            rows.append(
                {
                    **base,
                    "supertype_prefix": prefix,
                    "broad_class": mapping["broad_class"],
                    BROAD_CLASS_ANY_COLUMN: _broad_class_any(
                        mapping["broad_class"], []
                    ),
                    "lineage": mapping["lineage"],
                }
            )
    frame = pd.DataFrame(rows)
    for column in ("sink", f"region_plausible_{REGION}"):
        frame[column] = _bool_text(frame[column])
    return frame, [
        _identify(term_csv, "SEA-AD Multiregion terms", spec, SEAAD_TERM_FILE)
    ]


def build_wmb(
    taxonomy_dir: Path, spec: Mapping[str, Any]
) -> tuple[pd.DataFrame, list[InputFile]]:
    """Build the WMB class vocabulary.

    Args:
        taxonomy_dir: WMB-taxonomy release directory.
        spec: The ``wmb`` section of ``overrides.yaml``.

    Returns:
        The table and the identities of the inputs read.

    Raises:
        VocabBuildError: If the taxonomy and the curated mapping disagree.
    """
    term_path = _require(taxonomy_dir / WMB_TERM_FILE)
    membership_path = _require(taxonomy_dir / WMB_MEMBERSHIP_FILE)
    terms = pd.read_csv(term_path)
    membership = pd.read_csv(membership_path)
    level = spec["class_term_set"]
    classes = terms[terms["cluster_annotation_term_set_label"] == level].sort_values(
        "term_order"
    )
    if len(classes) != WMB_CLASS_COUNT:
        raise VocabBuildError(
            f"WMB: expected {WMB_CLASS_COUNT} classes, got {len(classes)}"
        )
    names = set(classes["name"])
    non_neuronal: Mapping[str, str] = spec["non_neuronal_classes"]
    never_drop = set(spec["never_drop"])
    _check_set("WMB non-neuronal classes", set(non_neuronal) & names, set(non_neuronal))
    _check_set("WMB never-drop classes", never_drop & names, never_drop)
    allowed_broad = set(BROAD_CLASSES["mouse"]) - {NEURONS}
    bad = sorted(set(non_neuronal.values()) - allowed_broad)
    if bad:
        raise VocabBuildError(f"WMB: non-neuronal broad classes {bad} not allowed")

    set_label = membership["cluster_annotation_term_set_label"]
    class_of = membership.loc[
        set_label == level, ["cluster_alias", "cluster_annotation_term_name"]
    ].rename(columns={"cluster_annotation_term_name": "class"})
    nt_of = membership.loc[
        set_label == spec["neurotransmitter_term_set"],
        ["cluster_alias", "cluster_annotation_term_name"],
    ].rename(columns={"cluster_annotation_term_name": "nt_term"})
    cells = membership.loc[
        membership["cluster_annotation_term_set_name"] == "cluster",
        ["cluster_alias", "number_of_cells"],
    ]
    joined = class_of.merge(nt_of, on="cluster_alias", how="left").merge(
        cells, on="cluster_alias", how="left"
    )
    if joined["number_of_cells"].isna().any():
        raise VocabBuildError("WMB: clusters without a cell count")

    rows = []
    for _, term in classes.iterrows():
        name = str(term["name"])
        broad = non_neuronal.get(name, NEURONS)
        group = joined[joined["class"] == name]
        nt, share = majority_nt(
            group["nt_term"],
            group["number_of_cells"].astype(float),
            min_share=float(spec["nt_majority_min_share"]),
        )
        is_neuronal = broad == NEURONS
        if is_neuronal != (share > 0.5):
            raise VocabBuildError(
                f"WMB class {name!r}: broad class {broad!r} but {share:.1%} of "
                "its cells carry an NT annotation"
            )
        rows.append(
            {
                "label": str(term["label"]),
                "class": name,
                "broad_class": broad,
                "nt": nt if is_neuronal else "",
                "never_drop": name in never_drop,
            }
        )
    frame = pd.DataFrame(rows)
    frame["never_drop"] = _bool_text(frame["never_drop"])
    inputs = [
        _identify(term_path, "WMB terms", spec, WMB_TERM_FILE),
        _identify(membership_path, "WMB cluster membership", spec, WMB_MEMBERSHIP_FILE),
    ]
    return frame, inputs


def build_human_floors(spec: Mapping[str, Any]) -> pd.DataFrame:
    """Expand the ``floors.human`` section into one row per floor key.

    Args:
        spec: The ``floors.human`` section of ``overrides.yaml``.

    Returns:
        Rows ``level, floor_class, platform, panel_family, min_counts,
        floor_source, inherited_from, note``.

    Raises:
        VocabBuildError: On unknown floor classes, levels or platforms.
    """
    rows = []
    for family in spec["panel_families"]:
        for entry in spec["rows"]:
            level, floor_class = entry["level"], entry["floor_class"]
            if level not in ("broad", "supercluster"):
                raise VocabBuildError(f"human floors: level {level!r} not allowed")
            if floor_class not in HUMAN_FLOOR_CLASSES:
                raise VocabBuildError(
                    f"human floors: class {floor_class!r} not allowed"
                )
            if floor_class == "COP" and level != "supercluster":
                raise VocabBuildError("human floors: COP is keyed at supercluster only")
            inherited: Mapping[str, str] = entry.get("inherited", {})
            for platform in PLATFORMS:
                rows.append(
                    {
                        "level": level,
                        "floor_class": floor_class,
                        "platform": platform,
                        "panel_family": family,
                        "min_counts": int(entry[platform]),
                        "floor_source": spec["floor_source"],
                        "inherited_from": inherited.get(platform, ""),
                        "note": entry.get("note", ""),
                    }
                )
    return pd.DataFrame(rows)


def build_mouse_floors(spec: Mapping[str, Any], wmb: pd.DataFrame) -> pd.DataFrame:
    """Expand the ``floors.mouse`` section over every WMB class.

    Args:
        spec: The ``floors.mouse`` section of ``overrides.yaml``.
        wmb: The WMB class vocabulary.

    Returns:
        Rows with the same columns as the human floors.
    """
    rows = []
    for family in spec["panel_families"]:
        for platform in spec["platforms"]:
            for level, min_counts in spec["min_counts"].items():
                for name in wmb["class"]:
                    rows.append(
                        {
                            "level": level,
                            "floor_class": name,
                            "platform": platform,
                            "panel_family": family,
                            "min_counts": int(min_counts),
                            "floor_source": spec["floor_source"],
                            "inherited_from": "",
                            "note": "",
                        }
                    )
    return pd.DataFrame(rows)


def render_notice(overrides: Mapping[str, Any], inputs: Sequence[InputFile]) -> str:
    """Render the NOTICE file.

    Args:
        overrides: The parsed ``overrides.yaml``.
        inputs: Identities of the taxonomy inputs.

    Returns:
        The NOTICE text.
    """
    whb, seaad, wmb = overrides["whb"], overrides["seaad"], overrides["wmb"]
    lines = [
        "MerXen annotation assets: provenance and licences",
        "=================================================",
        "",
        "Generated by scripts/annotation/build_vocab_tables.py; do not edit.",
        "",
        "Generated from Allen Institute taxonomies (term labels, names, order and",
        "neurotransmitter annotations) and the curated overrides.yaml:",
        "",
        f"- whb_supercluster_vocab.csv: {whb['taxonomy']} {whb['release']}",
        f"  ({whb['ccn']}; human whole brain, Siletti et al. 2023, Science).",
        f"- seaad_mr_subclass_vocab.csv: {seaad['taxonomy']} {seaad['release']}",
        f"  ({seaad['ccn']}; Seattle Alzheimer's Disease Brain Cell Atlas).",
        f"- wmb_class_vocab.csv: {wmb['taxonomy']} {wmb['release']}",
        f"  ({wmb['ccn']}; mouse whole brain, Yao et al. 2023, Nature).",
        "- floors_human.csv: plan §5.4 v1 floors from E2 real self-thinning",
        "  (review_robustness/floors_rederived.csv, precision-only column, with",
        "  the n >= 50 inheritance); floor_source real_e2.",
        "- floors_mouse.csv: plan §5.4 mouse v1 design values [L]; floor_source",
        "  design_v1.",
        "",
        "NT (nt column): the Excitatory / Inhibitory class holding more than half of",
        "a neuronal node's NT-annotated cells (cell-weighted over clusters), else",
        "Other; non-neuronal nodes have no NT. Region plausibility for frontal",
        "cortex follows the 16 superclusters pruned in E1 (ii).",
        "",
        "Broad classes follow E1's seven-class scheme: nodes outside the seven",
        "classes (WHB Splatter, Miscellaneous, Ependymal, Choroid plexus and",
        "Bergmann glia; SEA-AD Ependymal) are Mixed/Unknown, so their mass stays",
        "unallocated. SEA-AD broad_class_any lists the classes a node's probability",
        "counts toward in the SEA-AD broad score: VLMC & Perivascular without a",
        "supertype counts toward Vascular cells and Fibroblasts (E1 'Vascular/Fibro').",
        "",
        "Taxonomy inputs (ABC metadata path, bytes, sha256):",
        "",
    ]
    lines += [
        f"- {item.role}: {item.name}, {item.size}, {item.sha256}" for item in inputs
    ]
    lines += [
        "",
        "Curated by hand (reviewed in PRs, each row with a rationale):",
        "",
        "- state_genes_human.csv, state_genes_mouse.csv: disease / state genes",
        "  excluded from negative-gene lists (plan §4.7, §5.6) [L; reviewed in M4].",
        "- heldout_markers_human.csv, heldout_markers_mouse.csv: independent",
        "  held-out markers for the non-circular enrichment check (plan §5.8).",
        "  Human: two canonical markers per class; SST deliberately absent; GAD2",
        "  and P2RY12 avoided on MERSCOPE. Mouse: the E3 VZG2 held-out microglial",
        "  genes; class-level markers are still to be added (plan §5.8 step 4).",
        "",
        "Licences",
        "--------",
        "",
        "The Allen Brain Cell Atlas taxonomy metadata these tables derive from are",
        "distributed by the Allen Institute under the Creative Commons",
        "Attribution-NonCommercial 4.0 International licence (CC BY-NC 4.0) unless",
        "a release states otherwise; attribution: Allen Institute for Brain Science,",
        "Allen Brain Cell Atlas.",
        "Check each release's terms before redistributing derived tables.",
        "",
        "cell_type_mapper (MapMyCells), which uses these taxonomies at run time, is",
        "distributed under the Allen Institute Software License.",
        "",
        "MerXen's own code is MIT-licensed (see LICENSE); these notices apply to",
        "the derived data above.",
        "",
    ]
    return "\n".join(lines)


def load_overrides(path: Path) -> dict[str, Any]:
    """Parse ``overrides.yaml``.

    Args:
        path: The overrides file.

    Returns:
        The parsed mapping.

    Raises:
        ModuleNotFoundError: If PyYAML is not installed.
    """
    try:
        import yaml  # type: ignore[import-untyped]
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "build_vocab_tables needs PyYAML, which pyproject.toml does not "
            "declare yet (the pinned environment gets it through dask and "
            "pre-commit); install requirements/requirements.lock"
        ) from error
    loaded: dict[str, Any] = yaml.safe_load(path.read_text())
    return loaded


def build_all(
    *,
    whb_taxonomy_dir: Path,
    seaad_term_csv: Path,
    wmb_taxonomy_dir: Path,
    overrides_path: Path,
) -> BuildResult:
    """Build every generated file from the taxonomy inputs.

    Args:
        whb_taxonomy_dir: WHB-taxonomy release directory.
        seaad_term_csv: SEA-AD Multiregion ``cluster_annotation_term.csv``.
        wmb_taxonomy_dir: WMB-taxonomy release directory.
        overrides_path: ``overrides.yaml``.

    Returns:
        The generated files and the input identities.
    """
    overrides = load_overrides(overrides_path)
    if overrides.get("schema_version") != 1:
        raise VocabBuildError("overrides.yaml: unsupported schema_version")
    whb, whb_inputs = build_whb(whb_taxonomy_dir, overrides["whb"])
    seaad, seaad_inputs = build_seaad(seaad_term_csv, overrides["seaad"])
    wmb, wmb_inputs = build_wmb(wmb_taxonomy_dir, overrides["wmb"])
    floors_human = build_human_floors(overrides["floors"]["human"])
    floors_mouse = build_mouse_floors(overrides["floors"]["mouse"], wmb)
    inputs = (*whb_inputs, *seaad_inputs, *wmb_inputs)
    files = {
        "whb_supercluster_vocab.csv": _to_csv(whb),
        "seaad_mr_subclass_vocab.csv": _to_csv(seaad),
        "wmb_class_vocab.csv": _to_csv(wmb),
        "floors_human.csv": _to_csv(floors_human),
        "floors_mouse.csv": _to_csv(floors_mouse),
        "NOTICE": render_notice(overrides, inputs),
    }
    return BuildResult(files=files, inputs=inputs)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--whb-taxonomy-dir", type=Path, required=True)
    parser.add_argument("--seaad-term-csv", type=Path, required=True)
    parser.add_argument("--wmb-taxonomy-dir", type=Path, required=True)
    parser.add_argument(
        "--overrides", type=Path, default=DEFAULT_ASSET_DIR / "overrides.yaml"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ASSET_DIR)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Write nothing; exit 1 if a committed file is stale.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the generator.

    Args:
        argv: Command-line arguments (``sys.argv[1:]`` when ``None``).

    Returns:
        The process exit code.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv)
    result = build_all(
        whb_taxonomy_dir=args.whb_taxonomy_dir,
        seaad_term_csv=args.seaad_term_csv,
        wmb_taxonomy_dir=args.wmb_taxonomy_dir,
        overrides_path=args.overrides,
    )
    stale = []
    for name, text in result.files.items():
        path = args.output_dir / name
        current = path.read_text() if path.is_file() else None
        if current == text:
            logger.info("%s is up to date", name)
            continue
        stale.append(name)
        if not args.check:
            path.write_text(text)
            logger.info("Wrote %s", path)
    if args.check and stale:
        logger.error("Stale generated files: %s", ", ".join(stale))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

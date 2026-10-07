"""Mouse dataset gate G1-G5 from label-free signals (plan §7.6, §8.6).

Per-cell confidence cannot catch invalid data (class bootstrap >= 0.9 covers
.742 of the invalid VZG2 proseg_hybrid vs .749 of ag7; R-rob), so the mouse
gate reads signals that do not trust the labels. Its verdict has the human
gate's form (§5.4): a **level** (``full``, ``broad_only``, ``failed``) and a
separate **warning flag** with reasons.

- **G1 registration** (M0a ``segmentation_registration_check``: the QC
  stage's ``*_registration_qc.json`` or ``*_qc_summary.csv``): failed if
  the density ratio < 1.5 or the shift against the platform's own
  segmentation > 5 µm; warning if the ratio < 2.0, or no check was given,
  or the check skipped.
- **G2 marker referee** (class-group marker consistency over
  marker-pseudo-confident cells, marker sets derived per panel): failed
  < 0.70; warning < 0.80 or not evaluable.
- **G3 implausibility** (pre-pruning T2 share, E7's definition: unpruned
  subclasses with >= 20 MERFISH grey-matter cells, outside the never-drop
  classes, with < 25% of those cells in the present divisions): warning
  > 3%.
- **G4 composition** (soft class shares vs a MERFISH window of
  ``wmb_region_share`` sections): warning when Astro-Epen is outside ±5
  points or Immune outside ±1 point.
- **G5 spill-over** (``flag_microglial_spillover`` rate): warning > 15%.

The primary panel's trust state caps the level as for human (``refused`` ->
``failed``, ``broad_only`` -> ``broad_only``) and a provisional panel sets
the warning. A family validated by simulation warns, as for human, when
more than ``warn_unvalidated_share`` (10%) of the confident labels at a
chain level (broad, class, nt, subclass) fall outside its validated region
(§8.2; M13 decision D15 (a)); RESOLVE evaluates the gate again after
``resolve_mouse`` with those shares, and the level cannot change. A signal
that cannot be evaluated is recorded with its reason; G1 and G2 then warn
(the gate cannot confirm the data are valid), G3-G5 do not (G3: pruning
disabled or skipped, which the region step reports; G4: no AP window until
M6b's ``estimate_ap``; G5: the flag is null with a reason).

**G2's referee** (§8.6, [L]): six class groups (the WMB vocab's broad
classes: Neurons, Astrocytes/Ependymal, Oligodendrocyte lineage, OEC,
Vascular cells, Microglia), each with the panel genes whose group profile
(the mean of its classes' profiles) is >= ``specific_gene_ratio`` (20)
times the largest profile outside the group and >= 1/1000 of it
(immediate-early genes excluded); groups with fewer than 3 genes are left
out. Cells are pseudo-labelled with ``data/P1212``'s rule (each gene's
counts over its mean positive count in the dataset; group score = the sum;
label when the top group has >= 1.5 units and >= 60% of the summed
scores), and the consistency is the share of pseudo-labelled table cells
whose MapMyCells class (pruned primary call, whatever its confidence) is
in the same group. The 0.70 / 0.80 thresholds were set on MO1's
hand-listed referee (.911 ag7, .856 VZG2); this derived referee reads
.80 / .82 there (M6 stage B), so its warning band is provisional [L].

**Failed registration** (§7.5): annotation still runs, the gate is
``failed``, every cell is ``not_attempted_gate`` and ``exclude_hard``.

The module needs numpy and pandas.
"""

from __future__ import annotations

import csv
import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd

from merxen.annotation.mouse_flags import (
    IMMEDIATE_EARLY_GENES,
    MouseFlagProfiles,
    SpecificGenes,
    specific_genes,
)
from merxen.annotation.schema import GateLevel

if TYPE_CHECKING:
    from scipy import sparse

    from merxen.annotation.config import AnnotationFlagsConfig, MouseGateConfig
    from merxen.annotation.diagnostics import TrustDecision
    from merxen.annotation.mouse_regions import RegionShares
    from merxen.annotation.provenance import MouseGateProvenance

logger = logging.getLogger(__name__)

SIGNAL_KEYS: Final[tuple[str, ...]] = (
    "g1_density_ratio",
    "g1_shift_um",
    "g2_marker_consistency",
    "g3_implausible_share",
    "g4_astro_epen_points",
    "g4_immune_points",
    "g5_spillover_rate",
)
ASTRO_EPEN_SUFFIX: Final = "Astro-Epen"
IMMUNE_SUFFIX: Final = "Immune"
REGISTRATION_JSON_SUFFIX: Final = "_registration_qc.json"
QC_SUMMARY_SUFFIX: Final = "_qc_summary.csv"
_SEVERITY: Final[dict[str, int]] = {"full": 0, "broad_only": 1, "failed": 2}


class SignalStatus(StrEnum):
    """Outcome of one gate signal."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    NOT_EVALUATED = "not_evaluated"


# --------------------------------------------------------------------------
# G1: registration


@dataclass(frozen=True)
class RegistrationSignal:
    """The M0a registration check of one segmentation.

    Attributes:
        density_ratio: Transcripts within the radius of centroids / of
            random points.
        shift_um: Offset against the platform's own segmentation (``None``
            when not measured).
        status: M0a's own status (``pass``, ``warn``, ``skipped``).
        source: The file it was read from.
        reasons: M0a's reasons.
    """

    density_ratio: float | None
    shift_um: float | None
    status: str | None = None
    source: str | None = None
    reasons: tuple[str, ...] = ()

    @classmethod
    def from_file(cls, path: Path | str) -> RegistrationSignal:
        """Read a registration check (QC stage JSON or summary CSV).

        Args:
            path: The QC stage's registration JSON (``RegistrationCheckResult
                .to_dict``) or its summary CSV (``registration_status``,
                ``registration_density_ratio``, ``registration_shift_um``).

        Returns:
            The signal.

        Raises:
            ValueError: If the file holds no registration check.
        """
        source = Path(path)
        if source.suffix == ".json":
            payload = json.loads(source.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or "density_ratio" not in payload:
                raise ValueError(f"{source} holds no registration check")
            return cls(
                density_ratio=_number(payload.get("density_ratio")),
                shift_um=_number(payload.get("shift_um")),
                status=None
                if payload.get("status") is None
                else str(payload["status"]),
                source=str(source),
                reasons=tuple(str(item) for item in payload.get("reasons") or ()),
            )
        with source.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or "registration_density_ratio" not in rows[0]:
            raise ValueError(f"{source} holds no registration columns")
        row = rows[0]
        return cls(
            density_ratio=_number(row.get("registration_density_ratio")),
            shift_um=_number(row.get("registration_shift_um")),
            status=row.get("registration_status") or None,
            source=str(source),
        )


def find_registration_check(
    directories: Sequence[Path | str], sample_id: str
) -> RegistrationSignal | None:
    """Find a sample's M0a registration check in QC stage outputs (G1).

    The QC stage writes ``<dataset>_registration_qc.json`` and
    ``<dataset>_qc_summary.csv`` with ``dataset`` the lower-cased sample id
    (``merxen.qc.metrics.save_qc_results``). The JSON wins; a summary
    without registration columns (a QC run before M0a) does not count.

    Args:
        directories: QC output directories, searched recursively.
        sample_id: The annotation sample id (``<pair>_<PLATFORM>``).

    Returns:
        The signal, or ``None`` when no check is found.

    Raises:
        ValueError: If a name matches several files.
    """
    stem = sample_id.lower()
    for suffix in (REGISTRATION_JSON_SUFFIX, QC_SUMMARY_SUFFIX):
        name = f"{stem}{suffix}"
        # One file per resolved target (a published symlink and its work-dir
        # file are one check); the source keeps the path as found.
        found: dict[Path, Path] = {}
        for root in directories:
            for path in sorted(Path(root).rglob(name)):
                found.setdefault(path.resolve(), path)
        matches = sorted(found.values())
        if len(matches) > 1:
            raise ValueError(
                f"several {name} under the QC directories: {[str(m) for m in matches]}"
            )
        if matches:
            try:
                return RegistrationSignal.from_file(matches[0])
            except ValueError:
                if suffix == REGISTRATION_JSON_SUFFIX:
                    raise
                logger.info("%s holds no registration check", matches[0])
    return None


def _number(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(str(value))
    except ValueError:
        return None
    return number if math.isfinite(number) else None


# --------------------------------------------------------------------------
# G2: derived marker referee


@dataclass(frozen=True)
class MarkerReferee:
    """G2: class-group marker consistency of one sample.

    Attributes:
        consistency: Share of pseudo-labelled table cells whose MapMyCells
            class is in the pseudo-label's group (``None`` if too few).
        n_pseudo_confident: Pseudo-labelled table cells with a class call.
        n_cells: Table cells scored.
        markers: Group to its marker symbols.
        groups_without_markers: Groups left out (fewer than the minimum).
        recall: Per pseudo-label group, the share whose class agrees.
        reason: Why the consistency is ``None``.
    """

    consistency: float | None
    n_pseudo_confident: int
    n_cells: int
    markers: dict[str, list[str]]
    groups_without_markers: list[str] = field(default_factory=list)
    recall: dict[str, float] = field(default_factory=dict)
    reason: str | None = None

    def to_json(self) -> dict[str, Any]:
        """Return the referee as JSON (resolve summary)."""
        return {
            "consistency": _round(self.consistency),
            "n_pseudo_confident": self.n_pseudo_confident,
            "n_cells": self.n_cells,
            "markers": self.markers,
            "groups_without_markers": self.groups_without_markers,
            "recall": {key: _round(value) for key, value in self.recall.items()},
            "reason": self.reason,
        }


def _round(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(float(value), 6)


def class_group_markers(
    profiles: MouseFlagProfiles,
    group_of: Mapping[str, str],
    config: AnnotationFlagsConfig,
    *,
    min_genes: int = 3,
) -> tuple[dict[str, SpecificGenes], list[str]]:
    """Return each class group's panel markers (G2; the E3 specificity rule).

    Args:
        profiles: Class profiles on the query genes.
        group_of: WMB class name to its group (the vocab's broad class).
        config: ``specific_gene_ratio`` and ``specific_gene_min_share``.
        min_genes: Groups with fewer markers are left out.

    Returns:
        ``(markers per group, groups left out)``.
    """
    groups = np.array(
        [group_of.get(name) for name in profiles.class_names], dtype=object
    )
    markers: dict[str, SpecificGenes] = {}
    left_out: list[str] = []
    for group in sorted({value for value in groups if value is not None}):
        inside = groups == group
        if not (~inside).any():
            continue
        genes = specific_genes(
            profiles.class_profiles[inside].mean(axis=0),
            profiles.class_profiles[~inside],
            gene_ids=profiles.gene_ids,
            symbols=profiles.symbols,
            ratio=config.specific_gene_ratio,
            min_share=config.specific_gene_min_share,
            target=str(group),
            exclude_symbols=IMMEDIATE_EARLY_GENES,
        )
        if len(genes) >= min_genes:
            markers[str(group)] = genes
        else:
            left_out.append(str(group))
    return markers, left_out


def marker_pseudo_labels(
    counts: sparse.spmatrix | np.ndarray,
    markers: Mapping[str, SpecificGenes],
    *,
    min_units: float = 1.5,
    min_share: float = 0.6,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``data/P1212``'s marker pseudo-labels on derived marker sets.

    Args:
        counts: Cells x query genes.
        markers: Group to its markers.
        min_units: Units the top group needs.
        min_share: Its share of the summed units.

    Returns:
        ``(label per cell, pseudo-confident per cell)``.
    """
    from scipy import sparse as sp

    matrix = sp.csc_matrix(counts, dtype=np.float64)
    names = list(markers)
    units = np.zeros((matrix.shape[0], len(names)), dtype=np.float64)
    for column, name in enumerate(names):
        for gene in markers[name].indices:
            values = matrix[:, int(gene)].toarray().reshape(-1)
            positive = values[values > 0]
            if len(positive):
                units[:, column] += values / positive.mean()
    if not names:
        return np.full(matrix.shape[0], None, dtype=object), np.zeros(
            matrix.shape[0], dtype=bool
        )
    top = units.argmax(axis=1)
    best = units.max(axis=1)
    total = units.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        confident = (best >= min_units) & (best >= min_share * total) & (total > 0)
    labels = np.array(names, dtype=object)[top]
    return labels, np.asarray(confident, dtype=bool)


def marker_referee(
    counts: sparse.spmatrix | np.ndarray | None,
    profiles: MouseFlagProfiles | None,
    mmc_class: Sequence[object],
    group_of: Mapping[str, str],
    flags_config: AnnotationFlagsConfig,
    gate_config: MouseGateConfig,
) -> MarkerReferee:
    """Return G2 for one sample's table cells.

    Args:
        counts: Table cells x query genes (``None``: not evaluable).
        profiles: The primary bundle's profiles on the query genes.
        mmc_class: WMB class per table cell (pruned primary call).
        group_of: WMB class name to group.
        flags_config: The specificity rule.
        gate_config: G2 settings (pseudo-label rule, minimum cells).

    Returns:
        The referee.
    """
    n = len(mmc_class)
    if counts is None or profiles is None:
        return MarkerReferee(
            consistency=None,
            n_pseudo_confident=0,
            n_cells=n,
            markers={},
            reason="no query counts or profiles",
        )
    markers, left_out = class_group_markers(
        profiles, group_of, flags_config, min_genes=gate_config.g2_min_group_markers
    )
    labels, confident = marker_pseudo_labels(
        counts,
        markers,
        min_units=gate_config.g2_min_marker_units,
        min_share=gate_config.g2_min_marker_share,
    )
    mmc_group = np.array(
        [None if name is None else group_of.get(str(name)) for name in mmc_class],
        dtype=object,
    )
    scored = confident & np.array([value is not None for value in mmc_group])
    agree = scored & (labels == mmc_group)
    recall = {
        group: float(agree[scored & (labels == group)].mean())
        for group in markers
        if (scored & (labels == group)).any()
    }
    marker_symbols = {group: list(genes.symbols) for group, genes in markers.items()}
    n_scored = int(scored.sum())
    if len(markers) < 2:
        return MarkerReferee(
            consistency=None,
            n_pseudo_confident=n_scored,
            n_cells=n,
            markers=marker_symbols,
            groups_without_markers=left_out,
            reason=(
                f"{len(markers)} group(s) with >= "
                f"{gate_config.g2_min_group_markers} markers"
            ),
        )
    if n_scored < gate_config.g2_min_pseudo_confident:
        return MarkerReferee(
            consistency=None,
            n_pseudo_confident=n_scored,
            n_cells=n,
            markers=marker_symbols,
            groups_without_markers=left_out,
            recall=recall,
            reason=(
                f"{n_scored} pseudo-confident cells < "
                f"{gate_config.g2_min_pseudo_confident}"
            ),
        )
    return MarkerReferee(
        consistency=float(agree.sum() / n_scored),
        n_pseudo_confident=n_scored,
        n_cells=n,
        markers=marker_symbols,
        groups_without_markers=left_out,
        recall=recall,
    )


# --------------------------------------------------------------------------
# G3: pre-pruning implausibility


def t2_subclasses(
    present_regions: Sequence[str],
    shares: RegionShares,
    *,
    subclass_class: Mapping[str, str],
    never_drop_classes: Sequence[str],
    min_merfish_cells: int = 20,
    max_present_share: float = 0.25,
) -> set[str]:
    """Return E7's region-implausible (T2) subclasses of a section.

    As ``e7lib.implausible_subclasses(present, 0.25)``: a subclass with at
    least ``min_merfish_cells`` MERFISH grey-matter cells, outside the
    never-drop classes, with less than ``max_present_share`` of those cells
    in the present divisions. Pruning never removes the others (never-drop
    classes; small subclasses follow their class), so counting them would
    not measure the rule.

    Args:
        present_regions: The section's present divisions.
        shares: The region-share bundle's shares.
        subclass_class: Subclass name to its class name (mapping tree).
        never_drop_classes: ``MouseRegionConfig.never_drop_classes``.
        min_merfish_cells: ``MouseRegionConfig.min_merfish_cells_subclass``.
        max_present_share: E7's 0.25 (``g3_t2_max_present_share``).

    Returns:
        The T2 subclass names.
    """
    present = shares.present_share("subclass", present_regions)
    n_grey = shares.subclass_n_grey.reindex(present.index).fillna(0)
    never = {str(name) for name in never_drop_classes}
    in_never = np.array(
        [str(subclass_class.get(str(name), "")) in never for name in present.index],
        dtype=bool,
    )
    eligible = (n_grey.to_numpy() >= min_merfish_cells) & ~in_never
    return {
        str(name)
        for name in present.index[eligible & (present.to_numpy() < max_present_share)]
    }


def t2_share(
    unpruned_subclass: Sequence[object],
    present_regions: Sequence[str],
    shares: RegionShares,
    *,
    subclass_class: Mapping[str, str],
    never_drop_classes: Sequence[str],
    min_merfish_cells: int = 20,
    max_present_share: float = 0.25,
) -> float:
    """Return the T2 share (E7): table cells called to a T2 subclass.

    G3 and MO2 use this one definition (``t2_subclasses``).

    Args:
        unpruned_subclass: The subclass call per table cell (unpruned for
            G3; the pruned call for MO2's "T2 after pruning").
        present_regions: The section's present divisions.
        shares: The region-share bundle's shares.
        subclass_class: Subclass name to its class name.
        never_drop_classes: Classes never counted.
        min_merfish_cells: Subclasses with fewer MERFISH cells are not counted.
        max_present_share: E7's 0.25.

    Returns:
        The share of table cells whose subclass is T2.
    """
    t2 = t2_subclasses(
        present_regions,
        shares,
        subclass_class=subclass_class,
        never_drop_classes=never_drop_classes,
        min_merfish_cells=min_merfish_cells,
        max_present_share=max_present_share,
    )
    names = [None if value is None else str(value) for value in unpruned_subclass]
    if not names:
        return 0.0
    return float(np.mean([name in t2 for name in names]))


# --------------------------------------------------------------------------
# G4: composition vs a MERFISH window


@dataclass(frozen=True)
class MerfishWindow:
    """Class shares of pooled MERFISH sections (the G4 expectation).

    Attributes:
        sections: Section labels pooled.
        shares: WMB class name to its share of the pooled cells.
        n_cells: Pooled cells.
        ap_min_mm: Smallest AP of the sections.
        ap_max_mm: Largest AP of the sections.
    """

    sections: tuple[str, ...]
    shares: dict[str, float]
    n_cells: int
    ap_min_mm: float | None = None
    ap_max_mm: float | None = None

    def to_json(self) -> dict[str, Any]:
        """Return the window as JSON."""
        return {
            "sections": list(self.sections),
            "n_cells": self.n_cells,
            "ap_min_mm": _round(self.ap_min_mm),
            "ap_max_mm": _round(self.ap_max_mm),
            "astro_epen_share": _round(_suffix_share(self.shares, ASTRO_EPEN_SUFFIX)),
            "immune_share": _round(_suffix_share(self.shares, IMMUNE_SUFFIX)),
        }


def _suffix_share(shares: Mapping[str, float], suffix: str) -> float | None:
    values = [value for name, value in shares.items() if str(name).endswith(suffix)]
    return float(sum(values)) if values else None


def merfish_window(
    section_composition: pd.DataFrame, sections: Sequence[str]
) -> MerfishWindow:
    """Pool the class composition of MERFISH sections (``section_composition.parquet``).

    Args:
        section_composition: ``section, ap_ccf_mm, n_cells, level, node_name,
            n, freq`` rows of the region-share bundle.
        sections: The sections to pool.

    Returns:
        The window.

    Raises:
        ValueError: If a section is absent.
    """
    wanted = [str(section) for section in sections]
    rows = section_composition[
        (section_composition["level"].astype(str) == "class")
        & section_composition["section"].astype(str).isin(wanted)
    ]
    missing = sorted(set(wanted) - set(rows["section"].astype(str)))
    if missing:
        raise ValueError(
            f"MERFISH sections {missing} are not in the region-share bundle"
        )
    totals = rows.groupby("node_name", observed=True)["n"].sum()
    n_cells = int(rows.drop_duplicates("section")["n_cells"].astype(np.int64).sum())
    shares = {str(name): float(value / n_cells) for name, value in totals.items()}
    ap = pd.to_numeric(rows["ap_ccf_mm"], errors="coerce")
    return MerfishWindow(
        sections=tuple(wanted),
        shares=shares,
        n_cells=n_cells,
        ap_min_mm=float(ap.min()) if ap.notna().any() else None,
        ap_max_mm=float(ap.max()) if ap.notna().any() else None,
    )


def composition_offsets(
    shares: Mapping[str, float], window: MerfishWindow
) -> tuple[float | None, float | None]:
    """Return G4's Astro-Epen and Immune offsets in percentage points.

    Args:
        shares: The sample's soft class shares (class name to share).
        window: The MERFISH window.

    Returns:
        ``(astro_epen_points, immune_points)``: sample minus window.
    """
    result = []
    for suffix in (ASTRO_EPEN_SUFFIX, IMMUNE_SUFFIX):
        observed = _suffix_share(shares, suffix)
        expected = _suffix_share(window.shares, suffix)
        result.append(
            None
            if observed is None or expected is None
            else 100.0 * (observed - expected)
        )
    return result[0], result[1]


# --------------------------------------------------------------------------
# The verdict


@dataclass(frozen=True)
class MouseGateSignals:
    """The label-free signals of one mouse sample (§7.6).

    Attributes:
        registration: G1 (``None``: no check given).
        referee: G2 (``None``: not computed).
        t2_share: G3 (``None``: not evaluated; see ``t2_reason``).
        t2_reason: Why G3 is not evaluated.
        astro_epen_points: G4 Astro-Epen offset (``None``: not evaluated).
        immune_points: G4 Immune offset.
        window: The MERFISH window of G4.
        g4_reason: Why G4 is not evaluated.
        spillover_rate: G5 (``None``: the flag is null).
        g5_reason: Why G5 is not evaluated.
    """

    registration: RegistrationSignal | None = None
    referee: MarkerReferee | None = None
    t2_share: float | None = None
    t2_reason: str | None = None
    astro_epen_points: float | None = None
    immune_points: float | None = None
    window: MerfishWindow | None = None
    g4_reason: str | None = None
    spillover_rate: float | None = None
    g5_reason: str | None = None

    def values(self) -> dict[str, float | None]:
        """Return the signals keyed ``SIGNAL_KEYS`` (provenance ``signals``)."""
        registration = self.registration
        return {
            "g1_density_ratio": _round(
                None if registration is None else registration.density_ratio
            ),
            "g1_shift_um": _round(
                None if registration is None else registration.shift_um
            ),
            "g2_marker_consistency": _round(
                None if self.referee is None else self.referee.consistency
            ),
            "g3_implausible_share": _round(self.t2_share),
            "g4_astro_epen_points": _round(self.astro_epen_points),
            "g4_immune_points": _round(self.immune_points),
            "g5_spillover_rate": _round(self.spillover_rate),
        }


@dataclass(frozen=True)
class MouseGateVerdict:
    """The mouse gate verdict: a level plus a warning flag (§7.6).

    Attributes:
        level: ``full``, ``broad_only`` or ``failed``.
        warning: The warning flag (never lowers the level).
        level_reasons: Why the level is below ``full``.
        warning_reasons: Why the warning flag is set.
        signal_status: Per signal (G1-G5): ``SignalStatus`` value.
        notes: Signals not evaluated and why (no warning).
        signals: The signals.
        trust_state: The primary panel's trust state.
    """

    level: GateLevel
    warning: bool
    level_reasons: tuple[str, ...]
    warning_reasons: tuple[str, ...]
    signal_status: dict[str, str]
    notes: tuple[str, ...]
    signals: MouseGateSignals
    trust_state: str | None = None

    @property
    def reasons(self) -> tuple[str, ...]:
        """Level reasons followed by warning reasons."""
        return (*self.level_reasons, *self.warning_reasons)

    @property
    def attempts_leaf(self) -> bool:
        """Whether the subclass (leaf) level is attempted (``full``)."""
        return self.level == "full"

    def to_json(self) -> dict[str, Any]:
        """Return the verdict as JSON (resolve summary)."""
        signals = self.signals
        return {
            "level": self.level,
            "warning": self.warning,
            "level_reasons": list(self.level_reasons),
            "warning_reasons": list(self.warning_reasons),
            "signal_status": dict(self.signal_status),
            "notes": list(self.notes),
            "signals": signals.values(),
            "registration_source": (
                None if signals.registration is None else signals.registration.source
            ),
            "referee": None if signals.referee is None else signals.referee.to_json(),
            "window": None if signals.window is None else signals.window.to_json(),
            "trust_state": self.trust_state,
        }

    def provenance(
        self, *, notes: Sequence[str] = (), **regions: Any
    ) -> MouseGateProvenance:
        """Return ``AnnotationProvenance.mouse_gate`` (§4.6).

        Args:
            notes: Further reasons recorded after the level and warning
                reasons (the region step's status and reasons).
            **regions: ``MouseGateProvenance`` region fields
                (``section_regions``, ``region_source``, ...).

        Returns:
            The provenance record.
        """
        from merxen.annotation.provenance import MouseGateProvenance

        return MouseGateProvenance(
            signals=self.signals.values(),
            level=self.level,
            warning=self.warning,
            reasons=[*self.reasons, *(str(note) for note in notes)],
            **regions,
        )


def _worse(current: GateLevel, candidate: GateLevel) -> GateLevel:
    return candidate if _SEVERITY[candidate] > _SEVERITY[current] else current


def evaluate_mouse_gate(
    signals: MouseGateSignals,
    config: MouseGateConfig,
    *,
    trust: TrustDecision | None = None,
    validated_share: Mapping[str, float | None] | None = None,
    max_unvalidated_share: float | None = None,
    extra_warnings: Sequence[str] = (),
) -> MouseGateVerdict:
    """Return the mouse gate verdict (§7.6 truth table; see the module docstring).

    Args:
        signals: The sample's signals.
        config: ``AnnotationConfig.mouse_gate``.
        trust: The primary panel's trust decision (caps the level; a
            provisional panel warns).
        validated_share: Share of confident labels inside the validated
            region per chain level (``consensus.chain_validated_share`` after
            ``resolve_mouse``); a simulation-validated family warns where
            more than ``max_unvalidated_share`` fall outside it (§8.2). It
            never changes the level.
        max_unvalidated_share: ``AnnotationGate.warn_unvalidated_share``
            (default: its pre-registered value, 0.10).
        extra_warnings: Further warning reasons.

    Returns:
        The verdict.
    """
    level: GateLevel = "full"
    level_reasons: list[str] = []
    warnings: list[str] = []
    notes: list[str] = []
    status: dict[str, str] = {}

    # G1 registration.
    registration = signals.registration
    if registration is None or registration.density_ratio is None:
        status["g1"] = SignalStatus.NOT_EVALUATED.value
        detail = (
            "no registration check given"
            if registration is None
            else f"registration check {registration.status or 'without a ratio'}"
        )
        warnings.append(f"g1_registration: not evaluated ({detail})")
    else:
        ratio = registration.density_ratio
        shift = registration.shift_um
        failed = []
        if ratio < config.g1_density_ratio_fail:
            failed.append(f"density ratio {ratio:.3f} < {config.g1_density_ratio_fail}")
        if shift is not None and shift > config.g1_shift_fail_um:
            failed.append(
                f"shift {shift:.1f} um > {config.g1_shift_fail_um} um against the "
                "platform segmentation"
            )
        if failed:
            status["g1"] = SignalStatus.FAIL.value
            level = _worse(level, "failed")
            level_reasons.append("g1_registration: " + "; ".join(failed))
        elif ratio < config.g1_density_ratio_warn:
            status["g1"] = SignalStatus.WARN.value
            warnings.append(
                f"g1_registration: density ratio {ratio:.3f} < "
                f"{config.g1_density_ratio_warn}"
            )
        else:
            status["g1"] = SignalStatus.PASS.value
        if shift is None:
            notes.append("g1_registration: no shift against the platform segmentation")

    # G2 marker referee.
    referee = signals.referee
    consistency = None if referee is None else referee.consistency
    if consistency is None:
        status["g2"] = SignalStatus.NOT_EVALUATED.value
        reason = "not computed" if referee is None else referee.reason
        warnings.append(f"g2_marker_referee: not evaluated ({reason})")
    elif consistency < config.g2_marker_consistency_fail:
        status["g2"] = SignalStatus.FAIL.value
        level = _worse(level, "failed")
        level_reasons.append(
            f"g2_marker_referee: consistency {consistency:.3f} < "
            f"{config.g2_marker_consistency_fail}"
        )
    elif consistency < config.g2_marker_consistency_warn:
        status["g2"] = SignalStatus.WARN.value
        warnings.append(
            f"g2_marker_referee: consistency {consistency:.3f} < "
            f"{config.g2_marker_consistency_warn}"
        )
    else:
        status["g2"] = SignalStatus.PASS.value

    # G3 pre-pruning implausibility.
    if signals.t2_share is None:
        status["g3"] = SignalStatus.NOT_EVALUATED.value
        notes.append(f"g3_implausibility: not evaluated ({signals.t2_reason})")
    elif signals.t2_share > config.g3_implausible_warn:
        status["g3"] = SignalStatus.WARN.value
        warnings.append(
            f"g3_implausibility: pre-pruning T2 share {signals.t2_share:.4f} > "
            f"{config.g3_implausible_warn}"
        )
    else:
        status["g3"] = SignalStatus.PASS.value

    # G4 composition.
    if signals.astro_epen_points is None and signals.immune_points is None:
        status["g4"] = SignalStatus.NOT_EVALUATED.value
        notes.append(f"g4_composition: not evaluated ({signals.g4_reason})")
    else:
        off = []
        astro = signals.astro_epen_points
        immune = signals.immune_points
        if astro is not None and abs(astro) > config.g4_astro_epen_band_points:
            off.append(
                f"Astro-Epen {astro:+.2f} points "
                f"(band ±{config.g4_astro_epen_band_points})"
            )
        if immune is not None and abs(immune) > config.g4_immune_band_points:
            off.append(
                f"Immune {immune:+.2f} points (band ±{config.g4_immune_band_points})"
            )
        window = signals.window
        where = "" if window is None else f" vs MERFISH {', '.join(window.sections)}"
        if off:
            status["g4"] = SignalStatus.WARN.value
            warnings.append(f"g4_composition: {'; '.join(off)}{where}")
        else:
            status["g4"] = SignalStatus.PASS.value

    # G5 spill-over.
    if signals.spillover_rate is None:
        status["g5"] = SignalStatus.NOT_EVALUATED.value
        notes.append(f"g5_spillover: not evaluated ({signals.g5_reason})")
    elif signals.spillover_rate > config.g5_spillover_warn:
        status["g5"] = SignalStatus.WARN.value
        warnings.append(
            f"g5_spillover: flag rate {signals.spillover_rate:.4f} > "
            f"{config.g5_spillover_warn}"
        )
    else:
        status["g5"] = SignalStatus.PASS.value

    # Trust state (§8.2), as the human gate.
    if trust is not None:
        cap = trust.gate_level_cap
        if cap is not None:
            level = _worse(level, cap)
            level_reasons.append(
                f"{trust.gate_reason}: primary panel trust {trust.state} "
                f"({', '.join(trust.reason_codes) or 'no reason'})"
            )
        if trust.gate_warning:
            warnings.append(
                f"panel_provisional: primary panel trust {trust.state} (banner; "
                "never a lower gate level)"
            )
        if validated_share:
            # The human gate's rule and wording (thresholds.dataset_gate).
            from merxen.annotation.config import AnnotationGate

            limit = (
                AnnotationGate().warn_unvalidated_share
                if max_unvalidated_share is None
                else max_unvalidated_share
            )
            _, reasons = trust.unvalidated_share_warning(
                validated_share, max_share=limit
            )
            warnings.extend(
                f"{reason}: > {limit} of confident labels outside the validated region"
                for reason in reasons
            )
    warnings.extend(str(item) for item in extra_warnings)
    verdict = MouseGateVerdict(
        level=level,
        warning=bool(warnings),
        level_reasons=tuple(level_reasons),
        warning_reasons=tuple(warnings),
        signal_status=status,
        notes=tuple(notes),
        signals=signals,
        trust_state=None if trust is None else trust.state,
    )
    logger.info(
        "Mouse gate: %s%s (%s)",
        verdict.level,
        " + warning" if verdict.warning else "",
        ", ".join(f"{key} {value}" for key, value in status.items()),
    )
    return verdict

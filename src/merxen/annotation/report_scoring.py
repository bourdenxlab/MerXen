"""M8 gate scoring rules shared by the acceptance scripts (pre-registration §18).

The annotation report records metrics, never verdicts (``report_model``);
the acceptance scripts score them. This module holds the two parts of the
M8 scoring protocol that must not live only in an evidence script:

- **H12** (protocol item 3 as amended by §20 D11): the ordering is scored on
  the tangential-block CI, the report's primary (``report_depth.SCORED_CI``;
  approved by the user on 2026-10-01). The ordering verdict of a platform is
  its kind-less ``depth_ordering_passes`` record (the tiles where a platform
  has no tangential positions); the pair's ordering is the AND over both
  platforms (cross-checked against the pair's kind-less
  ``depth_ordering_replicated`` record); oligodendrocytes WM > GM needs the
  square-tile CI of ``oligodendrocyte_wm_minus_gm_share`` above 0 on both
  platforms (white matter has no tangential position). The square-tile
  ordering records are returned beside it as information.
- **The approved exceptions' scopes** (protocol item 6; §20): D3, D4, D5, D7,
  D10 and D12 cover the datasets, bins and classes their texts name. Each
  check here
  tests the conditions the text states, with the tolerances below, and
  returns ``EXCEPTION (Dn)`` when they hold and ``EXCEPTION-RECHECK (Dn)``
  (back to the user) when the value moved outside the text's scope. A
  failure no decision names is ``FAIL-OUTSIDE`` (``verdict``).

The tolerances were stated at the M8 review (2026-09-30), before stage B
(D10's and D12's at the review of 2026-10-01, before the re-score; §20);
they only narrow the approved exceptions (a failure outside them
goes back to the user), so they loosen nothing.

It is a ``report*`` module on purpose: like the report, it reads what
RESOLVE wrote and RESOLVE never imports it, so it stays out of the RESOLVE
rules fingerprint (``AnnotationReferences.RESOLVE_RULE_EXCLUDES``) and an
edit to the scoring does not re-run RESOLVE in the pipeline.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import pandas as pd

from merxen.annotation.report_depth import SCORED_CI, SQUARE_TILE_CI

PASS: Final = "PASS"
FAIL: Final = "FAIL"
NOT_AVAILABLE: Final = "NOT_AVAILABLE"
FAIL_OUTSIDE: Final = "FAIL-OUTSIDE"

# --------------------------------------------------------------------------
# H12 (plan §14; pre-registration §9, §17, §18 item 3)

# "P7513, P7113: Upper-layer IT < Deep-layer IT < Deep-layer NP/CT/6b medians
# with non-overlapping 95% CIs on both platforms. All 4 pairs incl. P1212,
# P5011, at broad level: oligodendrocytes WM > GM on both platforms."
H12_ORDERING_PAIRS: Final[frozenset[str]] = frozenset({"P7513", "P7113"})
H12_ORDERING: Final = "depth_ordering_passes"
H12_REPLICATED: Final = "depth_ordering_replicated"
H12_WM_GM: Final = "oligodendrocyte_wm_minus_gm_share"
H12_CI_METHOD: Final = "depth_ci_method"
_CI_NOTE: Final = re.compile(r"\bci=([a-z0-9_]+)")


def _get(record: Any, name: str) -> Any:
    if isinstance(record, Mapping):
        return record.get(name)
    return getattr(record, name, None)


def _h12(records: Iterable[Any], name: str) -> list[Any]:
    return [
        record
        for record in records
        if _get(record, "criterion") == "H12" and _get(record, "name") == name
    ]


def _bool_verdict(record: Any) -> tuple[str, str]:
    """Return a boolean record's verdict and its note."""
    value = _get(record, "value")
    note = str(_get(record, "note") or "")
    if _get(record, "status") != "measured" or not isinstance(value, bool):
        return NOT_AVAILABLE, note
    return (PASS if value else FAIL), note


def _pair_verdict(verdicts: Sequence[str], n_platforms: int) -> str:
    """AND over both platforms: a failure is known, else a gap is a gap."""
    if FAIL in verdicts:
        return FAIL
    if n_platforms < 2 or NOT_AVAILABLE in verdicts or not verdicts:
        return NOT_AVAILABLE
    return PASS


@dataclass(frozen=True)
class H12Score:
    """The scored H12 verdict of one pair (pre-registration §18 item 3, §20 D11).

    Attributes:
        pair_id: The pair.
        verdict: ``PASS``, ``FAIL`` or ``NOT_AVAILABLE``.
        ci_scored: The ordering CI the verdict used (``tangential_block_500um``;
            ``square_tile_500um`` when the pair has no tangential positions;
            ``mixed`` when only one platform has them).
        ci_reported_beside: The square tiles, not scored (``None`` when they
            are the scored primary).
        ordering_required: Whether §14 requires the ordering on this pair.
        ordering: Per platform, the scored ordering verdict.
        ordering_verdict: The pair's ordering (both platforms).
        wm_gm: Per platform: value, CI, basis and verdict of WM > GM.
        wm_gm_verdict: The pair's WM > GM (both platforms).
        reported_beside: The square-tile ordering records (information).
        notes: What a reader must know.
    """

    pair_id: str
    verdict: str
    ci_scored: str
    ci_reported_beside: str | None
    ordering_required: bool
    ordering: dict[str, str]
    ordering_verdict: str
    wm_gm: dict[str, dict[str, Any]]
    wm_gm_verdict: str
    reported_beside: dict[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def to_json(self: H12Score) -> dict[str, Any]:
        """Return the summary.json record."""
        return {
            "criterion": "H12",
            "pair_id": self.pair_id,
            "verdict": self.verdict,
            "ci_scored": self.ci_scored,
            "ci_reported_beside": self.ci_reported_beside,
            "ordering_required": self.ordering_required,
            "ordering": dict(self.ordering),
            "ordering_verdict": self.ordering_verdict,
            "wm_gm": {key: dict(value) for key, value in self.wm_gm.items()},
            "wm_gm_verdict": self.wm_gm_verdict,
            "reported_beside": dict(self.reported_beside),
            "notes": list(self.notes),
        }


def score_h12(records: Iterable[Any], *, pair_id: str) -> H12Score:
    """Score H12 of one pair from its ``acceptance_metrics.json`` records.

    The ordering of each platform is its kind-less ``depth_ordering_passes``
    record: the report's primary, the tangential blocks (``SCORED_CI``,
    §20 D11), or the square tiles on a platform without tangential positions
    (``depth_ci_method``). The pair's kind-less replication record, when
    present, must agree with the per-platform AND (else ``ValueError``: the
    report is inconsistent). WM > GM passes on a platform when the
    square-tile CI of the WM - GM oligodendrocyte share lies above 0. The
    square-tile ordering records are reported beside the score.

    Args:
        records: The pair's metric records (dicts or ``MetricRecord``).
        pair_id: The pair.

    Returns:
        The score.

    Raises:
        ValueError: If a replication record contradicts its platforms.
    """
    records = list(records)
    notes: list[str] = []
    methods = {_get(record, "value") for record in _h12(records, H12_CI_METHOD)}
    orderings = _h12(records, H12_ORDERING)
    scored = [record for record in orderings if _get(record, "kind") is None]
    beside = [record for record in orderings if _get(record, "kind") == SQUARE_TILE_CI]
    ordering: dict[str, str] = {}
    platform_methods: dict[str, str] = {}
    for record in scored:
        verdict, note = _bool_verdict(record)
        platform = str(_get(record, "platform"))
        ordering[platform] = verdict
        # Each primary record names its own CI (``ci=<method>`` in its note).
        found = _CI_NOTE.search(note)
        if found:
            platform_methods[platform] = found.group(1)
    used = set(platform_methods.values()) or methods
    tiles_only = used == {SQUARE_TILE_CI}
    if tiles_only:
        notes.append("no tangential positions: the scored primary is the tiles")
    elif len(used) > 1:
        notes.append(
            "mixed CIs: a platform without tangential positions is scored on "
            f"its tiles ({dict(sorted(platform_methods.items()))})"
        )
    ci_scored = (
        SQUARE_TILE_CI if tiles_only else SCORED_CI if len(used) <= 1 else "mixed"
    )
    platforms = {
        str(_get(record, "platform"))
        for record in records
        if _get(record, "criterion") == "H12" and _get(record, "platform")
    }
    for platform in sorted(platforms - set(ordering)):
        ordering[platform] = NOT_AVAILABLE
    ordering_verdict = _pair_verdict(list(ordering.values()), len(ordering))
    for record in _h12(records, H12_REPLICATED):
        if _get(record, "kind") is not None:
            continue
        value = _get(record, "value")
        if isinstance(value, bool) and value != (ordering_verdict == PASS):
            raise ValueError(
                f"{pair_id}: depth_ordering_replicated ({ci_scored}) is {value} but "
                f"the platforms' orderings are {ordering}"
            )
    wm_gm: dict[str, dict[str, Any]] = {}
    for record in _h12(records, H12_WM_GM):
        value = _get(record, "value")
        low = _get(record, "ci_low")
        measured = (
            _get(record, "status") == "measured"
            and isinstance(value, int | float)
            and isinstance(low, int | float)
            and math.isfinite(float(low))
        )
        verdict = (
            NOT_AVAILABLE
            if not measured
            else PASS
            if float(value) > 0 and float(low) > 0
            else FAIL
        )
        wm_gm[str(_get(record, "platform"))] = {
            "value": value,
            "ci_low": low,
            "ci_high": _get(record, "ci_high"),
            "basis": _get(record, "kind"),
            "ci": SQUARE_TILE_CI,
            "verdict": verdict,
        }
    for platform in sorted(platforms - set(wm_gm)):
        wm_gm[platform] = {"verdict": NOT_AVAILABLE}
    wm_gm_verdict = _pair_verdict(
        [value["verdict"] for value in wm_gm.values()], len(wm_gm)
    )
    required = pair_id in H12_ORDERING_PAIRS
    parts = [wm_gm_verdict] + ([ordering_verdict] if required else [])
    verdict = (
        FAIL if FAIL in parts else NOT_AVAILABLE if NOT_AVAILABLE in parts else PASS
    )
    reported: dict[str, Any] = {
        "ordering": {
            str(_get(record, "platform")): _bool_verdict(record)[0] for record in beside
        },
        "replicated": [
            _get(record, "value")
            for record in _h12(records, H12_REPLICATED)
            if _get(record, "kind") == SQUARE_TILE_CI
        ],
        "note": "square tiles, not scored (§20 D11: the tangential blocks are scored)",
    }
    return H12Score(
        pair_id=pair_id,
        verdict=verdict,
        ci_scored=ci_scored,
        ci_reported_beside=None if tiles_only else SQUARE_TILE_CI,
        ordering_required=required,
        ordering=ordering,
        ordering_verdict=ordering_verdict,
        wm_gm=wm_gm,
        wm_gm_verdict=wm_gm_verdict,
        reported_beside=reported,
        notes=tuple(notes),
    )


# --------------------------------------------------------------------------
# The approved exceptions (pre-registration §18 D3, D4, D5, D7; item 6)

# D3: "P1212_MERSCOPE carries the segmented-object coverage warning (.148 of
# 296,487 segmented objects ...) in addition to its expected broad_only level."
D3_DATASET: Final = ("proseg_hybrid", "P1212_MERSCOPE")
D3_TEXT_VALUE: Final = 0.148
D3_TOLERANCE: Final = 0.005
D3_LEVEL: Final = "broad_only"
# D5: "H2 exceeds 1% on P5011_MERSCOPE proseg_hybrid (1.02%) and
# P1212_MERSCOPE reseg (1.13%) ... none receives a broad or supercluster
# label". Scope: the value may exceed the text's by at most the tolerance.
# The reseg value was 1.04% (M4) until the user extended D5 to the
# pipeline's stage B value on 2026-10-01 (§20 D13).
D5_TEXT_VALUES: Final[dict[tuple[str, str], float]] = {
    ("proseg_hybrid", "P5011_MERSCOPE"): 0.0102,
    ("reseg", "P1212_MERSCOPE"): 0.0113,
}
D5_TOLERANCE: Final = 0.0005
D5_CRITERIA: Final[dict[tuple[str, str], str]] = {
    ("proseg_hybrid", "P5011_MERSCOPE"): "H2",
    ("reseg", "P1212_MERSCOPE"): "H17/H2",
}
# D7: "On P1212_MERSCOPE ... H4 passes 4 of 7 classes on the WHB-only
# held-out label set ... the three failing classes (Oligodendrocytes,
# Microglia, Vascular cells) reach the maximum AUROC their held-out-marker
# detection permits". Scope: the failing classes are among the three, each
# fails on AUROC alone (fold >= H4's 3) within the tolerance of its ceiling,
# and at least 4 classes pass.
D7_DATASET: Final = ("proseg_hybrid", "P1212_MERSCOPE")
D7_CLASSES: Final[frozenset[str]] = frozenset(
    {"Oligodendrocytes", "Microglia", "Vascular cells"}
)
D7_CEILING_TOLERANCE: Final = 0.01
D7_MIN_PASS: Final = 4
H4_MIN_FOLD: Final = 3.0
# D4: "Broad OPC is not emitted at 15 counts on the eight human datasets ...,
# nor at 120 counts on seven of them (... P7113_XENIUM emits it ...). Broad
# and supercluster Immune are not emitted at 60 counts on P5011_MERSCOPE."
D4_DATASETS: Final[frozenset[str]] = frozenset(
    f"{pair}_{platform}"
    for pair in ("P7513", "P1212", "P7113", "P5011")
    for platform in ("MERSCOPE", "XENIUM")
)
D4_OPC_MISSING: Final[frozenset[int]] = frozenset({15, 120})
D4_OPC_MISSING_BY_DATASET: Final[dict[str, frozenset[int]]] = {
    "P7113_XENIUM": frozenset({15}),
}
D4_IMMUNE_DATASET: Final = "P5011_MERSCOPE"
D4_IMMUNE_MISSING: Final[frozenset[int]] = frozenset({60})
# D4's would-raise list at PREP (the rebuilt set a WHB self-map, C11: "set a
# PREP 15"; broad from 15 and supercluster from 30 counts).
D4_PREP_WOULD_RAISE: Final[frozenset[tuple[str, str, int]]] = frozenset(
    [("broad", "Astro", depth) for depth in (15, 30, 60, 120)]
    + [("broad", "Immune", depth) for depth in (15, 30, 60)]
    + [("broad", "OPC", depth) for depth in (15, 30, 120)]
    + [("supercluster", "Astro", depth) for depth in (30, 60, 120)]
    + [("supercluster", "Immune", depth) for depth in (30, 60)]
)
# D10 (§20): H6's per-class precision may miss its target in exactly the
# (level, class, depth) rows stage B failed, on the datasets that failed
# them (``tables/h6_precision.csv``): OPC, COP and deep-layer CT / 6b, and
# supercluster Fibroblast 120 on P5011_XENIUM (extended by the user,
# 2026-10-01). Each row keeps stage B's precision as its floor, less
# ``D10_TOLERANCE``; a row outside the list or below its floor goes back to
# the user. Scored on proseg_hybrid only (H6's scored segmentation).
D10_SEGMENTATION: Final = "proseg_hybrid"
D10_TOLERANCE: Final = 0.01
D10_FAILING: Final[dict[tuple[str, str, int, str], float]] = {
    ("broad", "OPC", 15, "P1212_MERSCOPE"): 0.8396,
    ("broad", "OPC", 15, "P1212_XENIUM"): 0.7670,
    ("broad", "OPC", 15, "P5011_MERSCOPE"): 0.8875,
    ("broad", "OPC", 15, "P5011_XENIUM"): 0.8932,
    ("broad", "OPC", 15, "P7113_MERSCOPE"): 0.6888,
    ("broad", "OPC", 15, "P7113_XENIUM"): 0.7512,
    ("broad", "OPC", 15, "P7513_MERSCOPE"): 0.7680,
    ("broad", "OPC", 15, "P7513_XENIUM"): 0.8068,
    ("broad", "OPC", 30, "P1212_XENIUM"): 0.8644,
    ("broad", "OPC", 30, "P7113_MERSCOPE"): 0.8112,
    ("broad", "OPC", 30, "P7113_XENIUM"): 0.8539,
    ("broad", "OPC", 30, "P7513_MERSCOPE"): 0.8651,
    ("broad", "OPC", 30, "P7513_XENIUM"): 0.8895,
    ("broad", "OPC", 60, "P7113_MERSCOPE"): 0.8831,
    ("broad", "OPC", 120, "P1212_XENIUM"): 0.8564,
    ("broad", "OPC", 120, "P7113_MERSCOPE"): 0.7989,
    ("broad", "OPC", 120, "P7113_XENIUM"): 0.8454,
    ("broad", "OPC", 120, "P7513_MERSCOPE"): 0.8567,
    ("broad", "OPC", 120, "P7513_XENIUM"): 0.8848,
    (
        "supercluster",
        "Committed oligodendrocyte precursor",
        30,
        "P1212_MERSCOPE",
    ): 0.8017,
    ("supercluster", "Committed oligodendrocyte precursor", 30, "P1212_XENIUM"): 0.6926,
    (
        "supercluster",
        "Committed oligodendrocyte precursor",
        30,
        "P5011_MERSCOPE",
    ): 0.8499,
    ("supercluster", "Committed oligodendrocyte precursor", 30, "P5011_XENIUM"): 0.8279,
    (
        "supercluster",
        "Committed oligodendrocyte precursor",
        30,
        "P7113_MERSCOPE",
    ): 0.6106,
    ("supercluster", "Committed oligodendrocyte precursor", 30, "P7113_XENIUM"): 0.6591,
    (
        "supercluster",
        "Committed oligodendrocyte precursor",
        30,
        "P7513_MERSCOPE",
    ): 0.6914,
    ("supercluster", "Committed oligodendrocyte precursor", 30, "P7513_XENIUM"): 0.7171,
    ("supercluster", "Committed oligodendrocyte precursor", 60, "P1212_XENIUM"): 0.7786,
    (
        "supercluster",
        "Committed oligodendrocyte precursor",
        60,
        "P7113_MERSCOPE",
    ): 0.7050,
    ("supercluster", "Committed oligodendrocyte precursor", 60, "P7113_XENIUM"): 0.7503,
    (
        "supercluster",
        "Committed oligodendrocyte precursor",
        60,
        "P7513_MERSCOPE",
    ): 0.7756,
    ("supercluster", "Committed oligodendrocyte precursor", 60, "P7513_XENIUM"): 0.8030,
    (
        "supercluster",
        "Committed oligodendrocyte precursor",
        120,
        "P1212_MERSCOPE",
    ): 0.7885,
    (
        "supercluster",
        "Committed oligodendrocyte precursor",
        120,
        "P1212_XENIUM",
    ): 0.6785,
    (
        "supercluster",
        "Committed oligodendrocyte precursor",
        120,
        "P5011_MERSCOPE",
    ): 0.8413,
    (
        "supercluster",
        "Committed oligodendrocyte precursor",
        120,
        "P5011_XENIUM",
    ): 0.8335,
    (
        "supercluster",
        "Committed oligodendrocyte precursor",
        120,
        "P7113_MERSCOPE",
    ): 0.5891,
    (
        "supercluster",
        "Committed oligodendrocyte precursor",
        120,
        "P7113_XENIUM",
    ): 0.6432,
    (
        "supercluster",
        "Committed oligodendrocyte precursor",
        120,
        "P7513_MERSCOPE",
    ): 0.6746,
    (
        "supercluster",
        "Committed oligodendrocyte precursor",
        120,
        "P7513_XENIUM",
    ): 0.7098,
    ("supercluster", "Deep-layer corticothalamic and 6b", 30, "P1212_MERSCOPE"): 0.7685,
    ("supercluster", "Deep-layer corticothalamic and 6b", 30, "P1212_XENIUM"): 0.7796,
    ("supercluster", "Deep-layer corticothalamic and 6b", 30, "P7113_MERSCOPE"): 0.8202,
    ("supercluster", "Deep-layer corticothalamic and 6b", 30, "P7513_MERSCOPE"): 0.7807,
    ("supercluster", "Deep-layer corticothalamic and 6b", 30, "P7513_XENIUM"): 0.8446,
    ("supercluster", "Fibroblast", 120, "P5011_XENIUM"): 0.7829,
    ("supercluster", "Oligodendrocyte precursor", 30, "P1212_MERSCOPE"): 0.2360,
    ("supercluster", "Oligodendrocyte precursor", 30, "P1212_XENIUM"): 0.4950,
    ("supercluster", "Oligodendrocyte precursor", 30, "P5011_MERSCOPE"): 0.4117,
    ("supercluster", "Oligodendrocyte precursor", 30, "P5011_XENIUM"): 0.6395,
    ("supercluster", "Oligodendrocyte precursor", 30, "P7113_MERSCOPE"): 0.4610,
    ("supercluster", "Oligodendrocyte precursor", 30, "P7113_XENIUM"): 0.5910,
    ("supercluster", "Oligodendrocyte precursor", 30, "P7513_MERSCOPE"): 0.5261,
    ("supercluster", "Oligodendrocyte precursor", 30, "P7513_XENIUM"): 0.6270,
    ("supercluster", "Oligodendrocyte precursor", 60, "P1212_MERSCOPE"): 0.2635,
    ("supercluster", "Oligodendrocyte precursor", 60, "P1212_XENIUM"): 0.5317,
    ("supercluster", "Oligodendrocyte precursor", 60, "P5011_MERSCOPE"): 0.4477,
    ("supercluster", "Oligodendrocyte precursor", 60, "P5011_XENIUM"): 0.6726,
    ("supercluster", "Oligodendrocyte precursor", 60, "P7113_MERSCOPE"): 0.4976,
    ("supercluster", "Oligodendrocyte precursor", 60, "P7113_XENIUM"): 0.6260,
    ("supercluster", "Oligodendrocyte precursor", 60, "P7513_MERSCOPE"): 0.5626,
    ("supercluster", "Oligodendrocyte precursor", 60, "P7513_XENIUM"): 0.6607,
    ("supercluster", "Oligodendrocyte precursor", 120, "P1212_MERSCOPE"): 0.3665,
    ("supercluster", "Oligodendrocyte precursor", 120, "P1212_XENIUM"): 0.6474,
    ("supercluster", "Oligodendrocyte precursor", 120, "P5011_MERSCOPE"): 0.5673,
    ("supercluster", "Oligodendrocyte precursor", 120, "P5011_XENIUM"): 0.7687,
    ("supercluster", "Oligodendrocyte precursor", 120, "P7113_MERSCOPE"): 0.6157,
    ("supercluster", "Oligodendrocyte precursor", 120, "P7113_XENIUM"): 0.7303,
    ("supercluster", "Oligodendrocyte precursor", 120, "P7513_MERSCOPE"): 0.6753,
    ("supercluster", "Oligodendrocyte precursor", 120, "P7513_XENIUM"): 0.7590,
}
# D12 (b) (§20): "Re-running MAP with MapMyCells rng_seed 1 changes 7.0%
# (P7513) and 13.4% (P1212) of the seed-0 confident broad labels. None is a
# switch between two confident labels". Scope: no switch (the scored
# reading, D12 (a)), and the threshold-crossing share at most the text's
# value plus D5's tolerance.
D12_TEXT_VALUES: Final[dict[str, float]] = {"P7513": 0.070, "P1212": 0.134}
D12_TOLERANCE: Final = 0.0005
TOLERANCES: Final[dict[str, Any]] = {
    "D3": {"text_value": D3_TEXT_VALUE, "band": D3_TOLERANCE, "level": D3_LEVEL},
    "D4": {
        "opc_missing": sorted(D4_OPC_MISSING),
        "opc_missing_P7113_XENIUM": sorted(D4_OPC_MISSING_BY_DATASET["P7113_XENIUM"]),
        "immune": {"dataset": D4_IMMUNE_DATASET, "missing": sorted(D4_IMMUNE_MISSING)},
        "would_raise": "PREP within the text's 15 bins; per dataset within the "
        "stage A2 list (each added bin itemised and rechecked)",
    },
    "D5": {
        "text_values": {f"{s}/{d}": v for (s, d), v in D5_TEXT_VALUES.items()},
        "above_text_at_most": D5_TOLERANCE,
        "confident_broad_or_supercluster": 0,
    },
    "D7": {
        "classes": sorted(D7_CLASSES),
        "auroc_below_ceiling_at_most": D7_CEILING_TOLERANCE,
        "min_classes_passing": D7_MIN_PASS,
        "min_fold_failing": H4_MIN_FOLD,
    },
    "D10": {
        "failing_rows": "the 70 stage B rows (level, class, depth, dataset)",
        "segmentation": D10_SEGMENTATION,
        "n_rows": len(D10_FAILING),
        "precision_below_stage_b_at_most": D10_TOLERANCE,
    },
    "D12": {
        "text_values": dict(D12_TEXT_VALUES),
        "above_text_at_most": D12_TOLERANCE,
        "switches": 0,
    },
}


@dataclass(frozen=True)
class ExceptionVerdict:
    """Whether a failing row lies inside an approved exception's scope.

    Attributes:
        decision: ``D3``, ``D4``, ``D5``, ``D7``, ``D10`` or ``D12``.
        in_scope: Every condition of the text holds.
        reasons: The conditions that failed (empty when in scope), or what
            was checked.
    """

    decision: str
    in_scope: bool
    reasons: tuple[str, ...] = ()

    @property
    def verdict(self: ExceptionVerdict) -> str:
        """``EXCEPTION (Dn)`` or ``EXCEPTION-RECHECK (Dn)`` (back to the user)."""
        kind = "EXCEPTION" if self.in_scope else "EXCEPTION-RECHECK"
        return f"{kind} ({self.decision})"


def _result(decision: str, failed: list[str], checked: str) -> ExceptionVerdict:
    return ExceptionVerdict(
        decision, not failed, tuple(failed) if failed else (checked,)
    )


def check_d3(
    *, segmentation: str, dataset: str, value: float | None, level: str | None
) -> ExceptionVerdict | None:
    """D3: the P1212_MERSCOPE H8 warning (``None``: D3 names no such row)."""
    if (segmentation, dataset) != D3_DATASET:
        return None
    failed: list[str] = []
    if level != D3_LEVEL:
        failed.append(f"gate level {level} is not {D3_LEVEL}")
    if value is None or not math.isfinite(float(value)):
        failed.append("no warning value")
    elif abs(float(value) - D3_TEXT_VALUE) > D3_TOLERANCE + 1e-12:
        failed.append(
            f"warning value {float(value):.4f} outside "
            f"{D3_TEXT_VALUE} +- {D3_TOLERANCE}"
        )
    return _result(
        "D3", failed, f"value {value} within {D3_TEXT_VALUE} +- {D3_TOLERANCE}"
    )


def check_d5(
    *,
    criterion: str,
    segmentation: str,
    dataset: str,
    value: float | None,
    n_confident_broad: int | None,
    n_confident_supercluster: int | None,
) -> ExceptionVerdict | None:
    """D5: H2 on P5011_MERSCOPE proseg_hybrid and P1212_MERSCOPE reseg."""
    key = (segmentation, dataset)
    if D5_CRITERIA.get(key) != criterion:
        return None
    text = D5_TEXT_VALUES[key]
    failed: list[str] = []
    if value is None or not math.isfinite(float(value)):
        failed.append("no H2 value")
    elif float(value) > text + D5_TOLERANCE + 1e-12:
        failed.append(
            f"H2 {float(value):.4%} above the text's {text:.2%} + {D5_TOLERANCE:.2%}"
        )
    for name, count in (
        ("broad", n_confident_broad),
        ("supercluster", n_confident_supercluster),
    ):
        if count is None:
            failed.append(f"confident {name} labels of the flagged cells not measured")
        elif int(count) > 0:
            failed.append(f"{int(count)} flagged cells carry a confident {name} label")
    return _result(
        "D5",
        failed,
        f"H2 {value} <= {text} + {D5_TOLERANCE}; no confident broad / supercluster",
    )


def check_d7(
    *, segmentation: str, dataset: str, classes: pd.DataFrame
) -> ExceptionVerdict | None:
    """D7: H4 on P1212_MERSCOPE (the WHB-only held-out label set).

    Args:
        segmentation: The segmentation.
        dataset: The sample id.
        classes: The scored label set's rows of that dataset
            (``broad_class``, ``fold``, ``auroc``, ``auroc_ceiling``,
            ``passes``).

    Returns:
        The verdict, or ``None`` when D7 names no such row.
    """
    if (segmentation, dataset) != D7_DATASET:
        return None
    failed: list[str] = []
    passes = classes["passes"].astype(bool)
    if int(passes.sum()) < D7_MIN_PASS:
        failed.append(f"{int(passes.sum())} classes pass (< {D7_MIN_PASS})")
    for row in classes[~passes].to_dict("records"):
        name = str(row["broad_class"])
        if name not in D7_CLASSES:
            failed.append(f"{name} fails (not one of the text's classes)")
            continue
        fold = float(row["fold"])
        auroc = float(row["auroc"])
        ceiling = float(row.get("auroc_ceiling", math.nan))
        if not fold >= H4_MIN_FOLD:
            failed.append(f"{name} fold {fold:.2f} < {H4_MIN_FOLD:g}")
        if not math.isfinite(ceiling) or ceiling - auroc > D7_CEILING_TOLERANCE:
            failed.append(
                f"{name} AUROC {auroc:.3f} more than {D7_CEILING_TOLERANCE} below its "
                f"detection ceiling {ceiling:.3f}"
            )
    return _result(
        "D7",
        failed,
        "failing classes among the text's three, each at its AUROC ceiling",
    )


def check_d4_h18(
    *, dataset: str, level: str, cls: str, missing: Iterable[int]
) -> ExceptionVerdict | None:
    """D4: a failing H18 (dataset, level, class) row of the scored draw.

    PREP rows are not covered (D4 names no PREP bin): ``None``, a failure
    outside the texts. So is a class, or an Immune row on a dataset, D4 does
    not name. Broad OPC may miss 15 and 120 counts (P7113_XENIUM: 15 only);
    broad and supercluster Immune on P5011_MERSCOPE 60 counts; another
    missing depth of a named row is sent back to the user.
    """
    gone = {int(depth) for depth in missing}
    if dataset not in D4_DATASETS:
        return None
    if level == "broad" and cls == "OPC":
        allowed = D4_OPC_MISSING_BY_DATASET.get(dataset, D4_OPC_MISSING)
        extra = sorted(gone - allowed)
        return _result(
            "D4",
            [f"broad OPC also missing {extra} (the text: {sorted(allowed)})"]
            if extra
            else [],
            f"broad OPC missing {sorted(gone)} within {sorted(allowed)}",
        )
    if cls == "Immune" and level in ("broad", "supercluster"):
        if dataset != D4_IMMUNE_DATASET:
            return None  # the text names P5011_MERSCOPE only
        extra = sorted(gone - D4_IMMUNE_MISSING)
        return _result(
            "D4",
            [f"{level} Immune also missing {extra}"] if extra else [],
            f"{level} Immune missing {sorted(gone)} within {sorted(D4_IMMUNE_MISSING)}",
        )
    return None


def check_d4_would_raise(
    *,
    dataset: str,
    bins: Iterable[tuple[str, str, int]],
    reference: Iterable[tuple[str, str, int]] | None,
) -> ExceptionVerdict | None:
    """D4's would-raise list of one decision set (``None``: nothing to list).

    PREP (a dataset name starting with ``PREP``) is compared with the text's
    list (``D4_PREP_WOULD_RAISE``), a dataset with the stage A2 list of the
    scored draw (``reference``). Fewer bins stay in scope; each added bin is
    itemised and sent back to the user.
    """
    raised = {(str(level), str(cls), int(depth)) for level, cls, depth in bins}
    if not raised:
        return None
    if dataset.startswith("PREP"):
        allowed: set[tuple[str, str, int]] | None = set(D4_PREP_WOULD_RAISE)
    else:
        allowed = None if reference is None else set(reference)
    if allowed is None:
        return _result("D4", [f"no stage A2 would-raise list for {dataset}"], "")
    added = sorted(raised - allowed)
    return _result(
        "D4",
        [
            "would raise bins not listed at stage A2: "
            + ", ".join(f"{level} {cls} {depth}" for level, cls, depth in added)
        ]
        if added
        else [],
        f"{len(raised)} bins, all in D4's list "
        + ("(the text's)" if dataset.startswith("PREP") else "(stage A2's)"),
    )


def check_d10(
    *,
    segmentation: str,
    dataset: str,
    failing: Iterable[tuple[str, str, int, float]],
) -> ExceptionVerdict | None:
    """D10: H6's failing (level, class, depth, precision) rows of one dataset.

    ``None`` when nothing fails or on another segmentation. In scope when
    every failing row is one stage B failed on that dataset
    (``D10_FAILING``) and its precision is at most ``D10_TOLERANCE`` below
    stage B's value; otherwise the row is itemised and sent back.
    """
    rows = [
        (str(level), str(cls), int(depth), float(precision))
        for level, cls, depth, precision in failing
    ]
    if not rows or segmentation != D10_SEGMENTATION:
        return None
    failed: list[str] = []
    for level, cls, depth, precision in sorted(rows):
        floor = D10_FAILING.get((level, cls, depth, dataset))
        if floor is None:
            failed.append(f"{level} {cls} D{depth} fails outside D10")
        elif not precision >= floor - D10_TOLERANCE - 1e-12:
            failed.append(
                f"{level} {cls} D{depth} precision {precision:.3f} more than "
                f"{D10_TOLERANCE} below stage B's {floor:.3f}"
            )
    return _result(
        "D10",
        failed,
        f"{len(rows)} failing rows, all in D10's stage B list within "
        f"{D10_TOLERANCE} of their precision",
    )


def check_d12(
    *,
    pair: str,
    share_threshold: float | None,
    n_switched: int | None,
    n_missing: int | None = 0,
) -> ExceptionVerdict | None:
    """D12 (b): H15/seed1's threshold crossings on P7513 and P1212.

    ``None`` on another pair. In scope when no seed-0 confident label
    switches to another confident name, no cell is missing from either
    seed's table (a missing cell would count as a crossing), and the
    threshold-crossing share is at most the text's value plus
    ``D12_TOLERANCE``.
    """
    if pair not in D12_TEXT_VALUES:
        return None
    text = D12_TEXT_VALUES[pair]
    failed: list[str] = []
    if n_missing is None:
        failed.append("cells missing from the seed-1 table not counted")
    elif int(n_missing) > 0:
        failed.append(f"{int(n_missing)} cells missing from one seed's table")
    if n_switched is None:
        failed.append("label switches not measured")
    elif int(n_switched) > 0:
        failed.append(f"{int(n_switched)} confident labels switch name")
    if share_threshold is None or not math.isfinite(float(share_threshold)):
        failed.append("no threshold-crossing share")
    elif float(share_threshold) > text + D12_TOLERANCE + 1e-12:
        failed.append(
            f"threshold crossings {float(share_threshold):.4%} above the text's "
            f"{text:.1%} + {D12_TOLERANCE:.2%}"
        )
    return _result(
        "D12",
        failed,
        f"no switch; threshold crossings {share_threshold} <= {text} + {D12_TOLERANCE}",
    )


def verdict(passes: bool, scored: bool, exception: ExceptionVerdict | None) -> str:
    """The row's verdict: PASS / INFO / EXCEPTION[-RECHECK] (Dn) / FAIL-OUTSIDE."""
    if passes:
        return PASS if scored else "INFO pass"
    if not scored:
        return "INFO fail"
    return exception.verdict if exception is not None else FAIL_OUTSIDE


__all__ = [
    "D3_TEXT_VALUE",
    "D4_PREP_WOULD_RAISE",
    "D5_TEXT_VALUES",
    "D7_CLASSES",
    "D10_FAILING",
    "D12_TEXT_VALUES",
    "FAIL",
    "FAIL_OUTSIDE",
    "H12_ORDERING_PAIRS",
    "NOT_AVAILABLE",
    "PASS",
    "TOLERANCES",
    "ExceptionVerdict",
    "H12Score",
    "check_d3",
    "check_d4_h18",
    "check_d4_would_raise",
    "check_d5",
    "check_d7",
    "check_d10",
    "check_d12",
    "score_h12",
    "verdict",
]

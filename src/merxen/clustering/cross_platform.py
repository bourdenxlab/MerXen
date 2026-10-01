"""Which cross-platform statements a pair supports (plan §5.5, §8.5).

RESOLVE records in ``<pair>_resolve_summary.json`` which runs feed a pair's
cross-platform statistics and at which level (``pair.cross_platform``):

- same-panel pairs compare their annotation runs (``statistics_level``
  ``full``);
- ``per_platform`` pairs compare their WHB runs on the intersection panel
  (``_xpanel``); an intersection with fewer than 100 genes or a broad-only,
  refused or unknown intersection trust restricts them to the broad level
  (``broad_only``, flagged), and without an intersection run there are none
  (``none``).

RESOLVE builds that record from the panels alone. The scope a consumer
applies also folds in each sample's dataset gate
(``samples[<id>].resolution.gate.level``, plan §5.4): a ``broad_only``
dataset is excluded from supercluster-level cross-platform statistics, so
it caps the pair at ``broad_only`` (flagged, reason
``dataset_gate:<sample>:broad_only``), and a ``failed`` gate allows no
cross-platform statement (``none``, reason ``dataset_gate:<sample>:failed``).
RESOLVE's ``kinds`` / ``omitted_kinds`` say which composition kinds its pair
statistics compare: a ``per_platform`` pair compares its ``_xpanel`` runs
without the confident-only kind, which RESOLVE does not resolve.

Per-sample compositions and ``soft_broad_*`` rest on each platform's own
panel and are never compared across platforms. Every cross-platform
statement downstream of RESOLVE (the map_first hierarchy's records, the
downstream manifests, the end-of-run summary, the report) therefore reads
this scope first: ``CrossPlatformScope.allows(level)`` says whether a
statement at a label level may be made and ``allows_kind(kind)`` whether a
composition kind may be compared.

Imports: the standard library only (GPU-env safe; plan §3.5, §11.2).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

logger = logging.getLogger(__name__)

# merxen.annotation.pipeline.RESOLVE_SUMMARY_SUFFIX (test-enforced equal).
RESOLVE_SUMMARY_SUFFIX: Final = "_resolve_summary.json"
STATISTICS_LEVELS: Final[tuple[str, ...]] = ("full", "broad_only", "none")
# Label levels a ``broad_only`` pair may still compare (plan §8.5: "a
# broad_only level restricts them to broad"): the broad class and its
# lineage (``ct_final_level`` values).
BROAD_STATEMENT_LEVELS: Final[frozenset[str]] = frozenset({"lineage", "broad"})
SOURCE_RESOLVE_SUMMARY: Final = "resolve_summary"
SOURCE_MISSING: Final = "missing"
NO_PAIR_RECORD: Final = "no_pair_record"
# Dataset gate levels (merxen.annotation.schema.GATE_LEVELS, test-enforced
# equal) and the pair statistics level each allows at most (plan §5.4).
GATE_LEVELS: Final[tuple[str, ...]] = ("full", "broad_only", "failed")
GATE_STATISTICS_CAP: Final[dict[str, str]] = {
    "full": "full",
    "broad_only": "broad_only",
    "failed": "none",
}
DATASET_GATE_REASON: Final = "dataset_gate"


@dataclass(frozen=True)
class CrossPlatformScope:
    """The cross-platform statements a pair supports.

    Attributes:
        statistics_level: ``full`` (any level), ``broad_only`` (broad class
            and lineage only) or ``none`` (no cross-platform statement).
        flag: Whether RESOLVE flagged the pair's cross-platform statistics.
        reasons: RESOLVE's reasons (``intersection_genes:<n><100``,
            ``intersection_trust:<state>``, ``no_intersection_run``), or
            ``no_pair_record`` when the summary holds none (one platform).
        panel_mode: The pair's panel mode (``intersection`` /
            ``per_platform``), if known.
        jsd_runs: The run ids that feed the pair statistics.
        jsd_purpose: ``annotation`` or ``intersection_xpanel``.
        source: ``resolve_summary``, or ``missing`` without a summary.
        kinds: The composition kinds RESOLVE's pair statistics compare
            (``soft``, ``soft_ge30``, ``confident``, ``argmax``); empty
            when none is recorded.
        omitted_kinds: Kinds RESOLVE left out of them (``confident`` for a
            ``per_platform`` pair).
        resolve_statistics_level: The level RESOLVE recorded from the
            panels, before the dataset gates were folded in (``None``
            without a pair record).
        dataset_gates: ``(sample id, gate level)`` of each sample, sorted.
    """

    statistics_level: str
    flag: bool
    reasons: tuple[str, ...] = ()
    panel_mode: str | None = None
    jsd_runs: tuple[str, ...] = field(default_factory=tuple)
    jsd_purpose: str | None = None
    source: str = SOURCE_RESOLVE_SUMMARY
    kinds: tuple[str, ...] = field(default_factory=tuple)
    omitted_kinds: tuple[str, ...] = field(default_factory=tuple)
    resolve_statistics_level: str | None = None
    dataset_gates: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    def __post_init__(self: CrossPlatformScope) -> None:
        """Check the level.

        Raises:
            ValueError: On an unknown ``statistics_level``.
        """
        for level in (self.statistics_level, self.resolve_statistics_level):
            if level is not None and level not in STATISTICS_LEVELS:
                raise ValueError(
                    f"cross-platform statistics_level must be one of "
                    f"{STATISTICS_LEVELS}, got {level!r}"
                )

    def allows(self: CrossPlatformScope, level: str) -> bool:
        """Return whether a cross-platform statement at ``level`` may be made.

        Args:
            level: A label level (``ct_final_level`` value, e.g. ``broad``,
                ``supercluster``).

        Returns:
            ``True`` for any level when ``full``, for the broad class and
            lineage when ``broad_only``, never when ``none``.
        """
        if self.statistics_level == "full":
            return True
        if self.statistics_level == "broad_only":
            return str(level) in BROAD_STATEMENT_LEVELS
        return False

    def allows_kind(self: CrossPlatformScope, kind: str) -> bool:
        """Return whether a composition kind may be compared across platforms.

        Args:
            kind: A composition kind (``soft``, ``soft_ge30``, ``confident``,
                ``argmax``).

        Returns:
            ``True`` only when some statement is allowed
            (``statistics_level`` not ``none``) and RESOLVE's pair
            statistics compare ``kind``: never for a kind RESOLVE omitted
            (``confident`` of a ``per_platform`` pair) or for any kind when
            none is recorded.
        """
        if self.statistics_level == "none":
            return False
        return str(kind) in self.kinds and str(kind) not in self.omitted_kinds

    def to_dict(self: CrossPlatformScope) -> dict[str, Any]:
        """Return a JSON-ready dict (lists for tuples, a dict of gates)."""
        payload = asdict(self)
        payload["reasons"] = list(self.reasons)
        payload["jsd_runs"] = list(self.jsd_runs)
        payload["kinds"] = list(self.kinds)
        payload["omitted_kinds"] = list(self.omitted_kinds)
        payload["dataset_gates"] = dict(self.dataset_gates)
        return payload

    @classmethod
    def from_dict(
        cls: type[CrossPlatformScope], payload: Mapping[str, Any]
    ) -> CrossPlatformScope:
        """Rebuild a scope from ``to_dict`` output.

        Args:
            payload: The dict.

        Returns:
            The scope.
        """
        return cls(
            statistics_level=str(payload["statistics_level"]),
            flag=bool(payload.get("flag", False)),
            reasons=tuple(str(item) for item in payload.get("reasons") or ()),
            panel_mode=_text_or_none(payload.get("panel_mode")),
            jsd_runs=tuple(str(item) for item in payload.get("jsd_runs") or ()),
            jsd_purpose=_text_or_none(payload.get("jsd_purpose")),
            source=str(payload.get("source", SOURCE_RESOLVE_SUMMARY)),
            kinds=tuple(str(item) for item in payload.get("kinds") or ()),
            omitted_kinds=tuple(
                str(item) for item in payload.get("omitted_kinds") or ()
            ),
            resolve_statistics_level=_text_or_none(
                payload.get("resolve_statistics_level")
            ),
            dataset_gates=_gate_pairs(payload.get("dataset_gates")),
        )


def _text_or_none(value: Any) -> str | None:
    return None if value is None else str(value)


def _gate_pairs(value: Any) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, Mapping):
        return ()
    return tuple(sorted((str(key), str(level)) for key, level in value.items()))


def _omitted_kinds(value: Any) -> tuple[str, ...]:
    """Return RESOLVE's omitted kinds (a dict of reasons, or a list)."""
    if value is None:
        return ()
    if isinstance(value, Mapping):
        return tuple(sorted(str(key) for key in value))
    if isinstance(value, str):
        return (value,)
    return tuple(sorted(str(item) for item in value))


def dataset_gate_levels(summary: Mapping[str, Any]) -> dict[str, str]:
    """Return each sample's dataset gate level from a RESOLVE pair summary.

    Args:
        summary: A parsed ``<pair>_resolve_summary.json``.

    Returns:
        ``samples[<id>].resolution.gate.level`` per sample that records one.

    Raises:
        ValueError: On a gate level outside ``GATE_LEVELS``.
    """
    samples = summary.get("samples")
    if not isinstance(samples, Mapping):
        return {}
    levels: dict[str, str] = {}
    for sample_id, sample in samples.items():
        resolution = sample.get("resolution") if isinstance(sample, Mapping) else None
        gate = resolution.get("gate") if isinstance(resolution, Mapping) else None
        level = gate.get("level") if isinstance(gate, Mapping) else None
        if level is None:
            continue
        if str(level) not in GATE_LEVELS:
            raise ValueError(
                f"{sample_id}: dataset gate level must be one of {GATE_LEVELS}, "
                f"got {level!r}"
            )
        levels[str(sample_id)] = str(level)
    return levels


def _capped_level(level: str, cap: str) -> str:
    """Return the more restrictive of two statistics levels."""
    return max(level, cap, key=STATISTICS_LEVELS.index)


def resolve_summary_filename(pair_id: str) -> str:
    """Return RESOLVE's pair summary file name.

    Args:
        pair_id: Pair id.

    Returns:
        ``<pair_id>_resolve_summary.json``.
    """
    return f"{pair_id}{RESOLVE_SUMMARY_SUFFIX}"


def scope_from_resolve_summary(summary: Mapping[str, Any]) -> CrossPlatformScope:
    """Return the cross-platform scope a RESOLVE pair summary records.

    Args:
        summary: A parsed ``<pair>_resolve_summary.json``.

    Returns:
        The scope of ``summary["pair"]["cross_platform"]`` with the samples'
        dataset gates folded in: any ``broad_only`` gate caps the level at
        ``broad_only`` and any ``failed`` gate sets ``none``, each flagged
        with reason ``dataset_gate:<sample>:<level>`` (plan §5.4). ``none``
        with reason ``no_pair_record`` when the pair has no such record (a
        single-platform sample, or a pair with one platform resolved).
    """
    gates = dataset_gate_levels(summary)
    gate_pairs = tuple(sorted(gates.items()))
    pair = summary.get("pair")
    record = pair.get("cross_platform") if isinstance(pair, Mapping) else None
    if not isinstance(record, Mapping):
        return CrossPlatformScope(
            statistics_level="none",
            flag=False,
            reasons=(NO_PAIR_RECORD,),
            panel_mode=_text_or_none(summary.get("panel_mode")),
            dataset_gates=gate_pairs,
        )
    runs = record.get("jsd_run")
    if runs is None:
        run_ids: tuple[str, ...] = ()
    elif isinstance(runs, str):
        run_ids = (runs,)
    else:
        run_ids = tuple(str(item) for item in runs)
    recorded = str(record.get("statistics_level") or "none")
    level = recorded
    flag = bool(record.get("flag", False))
    reasons = [str(item) for item in record.get("reasons") or ()]
    for sample_id, gate_level in gate_pairs:
        cap = GATE_STATISTICS_CAP[gate_level]
        if cap == "full":
            continue
        level = _capped_level(level, cap)
        flag = True
        reasons.append(f"{DATASET_GATE_REASON}:{sample_id}:{gate_level}")
    return CrossPlatformScope(
        statistics_level=level,
        flag=flag,
        reasons=tuple(reasons),
        panel_mode=_text_or_none(record.get("panel_mode")),
        jsd_runs=run_ids,
        jsd_purpose=_text_or_none(record.get("jsd_purpose")),
        kinds=tuple(str(item) for item in record.get("kinds") or ()),
        omitted_kinds=_omitted_kinds(record.get("omitted_kinds")),
        resolve_statistics_level=recorded,
        dataset_gates=gate_pairs,
    )


def load_cross_platform_scope(
    labels_dir: Path | str, pair_id: str
) -> CrossPlatformScope:
    """Read a pair's cross-platform scope from a RESOLVE output directory.

    Args:
        labels_dir: ``annotation_resolve_out``.
        pair_id: Pair id.

    Returns:
        The recorded scope, or ``none`` (source ``missing``) when the
        directory holds no pair summary: no cross-platform statement is made
        without RESOLVE's record.
    """
    path = Path(labels_dir) / resolve_summary_filename(pair_id)
    if not path.is_file():
        logger.warning(
            "%s: no RESOLVE pair summary at %s; no cross-platform statement "
            "will be made (plan §8.5)",
            pair_id,
            path,
        )
        return CrossPlatformScope(
            statistics_level="none",
            flag=True,
            reasons=(NO_PAIR_RECORD,),
            source=SOURCE_MISSING,
        )
    summary = json.loads(path.read_text(encoding="utf-8"))
    return scope_from_resolve_summary(summary)

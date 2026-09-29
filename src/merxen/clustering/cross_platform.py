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

Per-sample compositions and ``soft_broad_*`` rest on each platform's own
panel and are never compared across platforms. Every cross-platform
statement downstream of RESOLVE (the map_first hierarchy's records, the
downstream manifests, the end-of-run summary, the report) therefore reads
this record first: ``CrossPlatformScope.allows(level)`` says whether a
statement at a label level may be made.

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
    """

    statistics_level: str
    flag: bool
    reasons: tuple[str, ...] = ()
    panel_mode: str | None = None
    jsd_runs: tuple[str, ...] = field(default_factory=tuple)
    jsd_purpose: str | None = None
    source: str = SOURCE_RESOLVE_SUMMARY

    def __post_init__(self: CrossPlatformScope) -> None:
        """Check the level.

        Raises:
            ValueError: On an unknown ``statistics_level``.
        """
        if self.statistics_level not in STATISTICS_LEVELS:
            raise ValueError(
                f"cross-platform statistics_level must be one of "
                f"{STATISTICS_LEVELS}, got {self.statistics_level!r}"
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

    def to_dict(self: CrossPlatformScope) -> dict[str, Any]:
        """Return a JSON-ready dict (lists for tuples)."""
        payload = asdict(self)
        payload["reasons"] = list(self.reasons)
        payload["jsd_runs"] = list(self.jsd_runs)
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
        )


def _text_or_none(value: Any) -> str | None:
    return None if value is None else str(value)


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
        The scope of ``summary["pair"]["cross_platform"]``; ``none`` with
        reason ``no_pair_record`` when the pair has no such record (a
        single-platform sample, or a pair with one platform resolved).
    """
    pair = summary.get("pair")
    record = pair.get("cross_platform") if isinstance(pair, Mapping) else None
    if not isinstance(record, Mapping):
        return CrossPlatformScope(
            statistics_level="none",
            flag=False,
            reasons=(NO_PAIR_RECORD,),
            panel_mode=_text_or_none(summary.get("panel_mode")),
        )
    runs = record.get("jsd_run")
    if runs is None:
        run_ids: tuple[str, ...] = ()
    elif isinstance(runs, str):
        run_ids = (runs,)
    else:
        run_ids = tuple(str(item) for item in runs)
    return CrossPlatformScope(
        statistics_level=str(record.get("statistics_level") or "none"),
        flag=bool(record.get("flag", False)),
        reasons=tuple(str(item) for item in record.get("reasons") or ()),
        panel_mode=_text_or_none(record.get("panel_mode")),
        jsd_runs=run_ids,
        jsd_purpose=_text_or_none(record.get("jsd_purpose")),
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

"""Acceptance bookkeeping for the slow real-data tests (not a test module).

A slow test measures everything first, records each acceptance item here (one INFO
line with the measured value and the limit), and calls :meth:`Acceptance.finish`
last, so that a failed performance limit never hides a correctness result: the
failures of all items come out in one assertion message.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

# A fused estimate this close to the click confirms the click's location for the
# image-check acceptance item (the verdict's own ``confirm_um`` is stricter).
IMAGE_NEAR_UM = 25.0


@dataclass
class Acceptance:
    """Collects acceptance items, logs each one and fails once at the end."""

    logger: logging.Logger
    title: str
    failures: list[str] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)

    def record(self, item: str, measured: object, limit: str, passed: bool) -> bool:
        """Log one item (``measured`` against ``limit``) and remember a failure."""
        line = (
            f"{self.title} | {item}: measured {measured} | limit {limit} | "
            f"{'PASS' if passed else 'FAIL'}"
        )
        self.lines.append(line)
        self.logger.info(line)
        if not passed:
            self.failures.append(f"{item}: measured {measured}, limit {limit}")
        return passed

    def at_most(self, item: str, measured: float, limit: float, unit: str = "") -> bool:
        """Record ``measured <= limit``."""
        return self.record(
            item, f"{measured:.3g}{unit}", f"<= {limit:.3g}{unit}", measured <= limit
        )

    def at_least(
        self, item: str, measured: float, limit: float, unit: str = ""
    ) -> bool:
        """Record ``measured >= limit``."""
        return self.record(
            item, f"{measured:.3g}{unit}", f">= {limit:.3g}{unit}", measured >= limit
        )

    def holds(self, item: str, passed: bool, measured: object = "") -> bool:
        """Record a yes/no item (``measured`` says what was seen)."""
        return self.record(item, measured if measured != "" else passed, "true", passed)

    def finish(self) -> None:
        """Raise one AssertionError naming every failed item (none: return)."""
        if self.failures:
            raise AssertionError(
                f"{self.title}: {len(self.failures)} of {len(self.lines)} acceptance "
                "items failed:\n  " + "\n  ".join(self.failures)
            )


@dataclass(frozen=True)
class SevenPairReport:
    """The image verdicts of the edge and interior pairs the acceptance item names.

    Attributes:
        rows: One log line per pair (verdict and offset).
        n_contradicted: Pairs with a ``contradicted`` verdict.
        n_near: Pairs with a confident fused estimate within ``near_um`` of the
            click: confirmed, or inconclusive with such an estimate.
        near_um: The distance used.
    """

    rows: tuple[str, ...]
    n_contradicted: int
    n_near: int
    near_um: float


def seven_pair_report(
    verdicts: Mapping[int, tuple[str, float | None]],
    pair_ids: Iterable[int],
    *,
    near_um: float = IMAGE_NEAR_UM,
) -> SevenPairReport:
    """Summarise the image verdicts of ``pair_ids`` for the acceptance item.

    Args:
        verdicts: Pair id to ``(verdict, image_offset_um)``; the offset is the
            distance of the best confident fused estimate from the click, or None.
        pair_ids: The pairs of the item; one that is not in ``verdicts`` counts as
            neither contradicted nor near.
        near_um: Distance within which an estimate counts as near the click.

    Returns:
        The per-pair lines and the two counts.
    """
    rows: list[str] = []
    n_contradicted = 0
    n_near = 0
    for pair_id in pair_ids:
        if pair_id not in verdicts:
            rows.append(f"pair {pair_id}: not in the check")
            continue
        verdict, offset = verdicts[pair_id]
        is_near = (
            verdict in {"confirmed", "inconclusive"}
            and offset is not None
            and offset <= near_um
        )
        n_contradicted += verdict == "contradicted"
        n_near += is_near
        shown = "none" if offset is None else f"{offset:.1f} um"
        rows.append(f"pair {pair_id}: {verdict}, offset {shown}")
    return SevenPairReport(tuple(rows), n_contradicted, n_near, near_um)

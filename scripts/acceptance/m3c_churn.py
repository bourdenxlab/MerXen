#!/usr/bin/env python
"""M3c pre-registration §21 (iii): stability of the emitted triples (pass / fail).

On a version-7 5K mouse bundle: ensemble A = the bundle's emission members;
ensemble B = the ``ensemble_B`` of ``m3c_v7_diagnostic.py`` (same test cells,
grid, mapping configuration and mapping seed 0; an output directory, never
the store). Stage D (failed, on record): A = {R1_contam_HO@0, @1, @2,
R3_measured_HO@0}, B = {R1_contam_HO@3, @4, @5, R3_measured_HO@1}. The
amended re-test (pre-registration §22.4, 2026-09-29): A = the rebuilt
bundle's {R1_contam_HO@0, @6-@10, R3_measured_HO@2, @3}, B = the comparator
{R1_contam_HO@20-@25, R3_measured_HO@20, @21}; the two must share no member.
Triples = the (level, class, bin) emitted in the provisional regime of the
unweighted PREP decisions, after the saturated-bp rule and the monotone
fill, over every level the bundle decides (supertype included). churn =
(|A \\ B| + |B \\ A|) / |A u B|; pass: churn <= 0.02. Reported beside it:
the broad-to-subclass churn (broad, class, NT and subclass together), the
churn per level and for the 14 major classes, the single-member churn of the
two ensembles' first R1 members under the version-7 conventions (phase 1's
analogue: 47 / 465 = 0.101) and each ensemble's emitted bins by route with
its spread-margin failures.

Usage::

    python scripts/acceptance/m3c_churn.py --bundle <v7 wmb_panel bundle> \\
        --diag-dir $A/m3c/.../prime5k_mouse/v7_diagnostic --out-dir $A/m3c/churn
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

TOLERANCE = 0.02
# The broad-to-subclass churn reported beside the test (D1 of 2026-09-29).
BROAD_TO_SUBCLASS = ("broad", "class", "nt", "subclass")
ROUTE_KEYS = (
    "n_emitted",
    "n_unanimous",
    "n_spread",
    "n_filled",
    "n_saturated_emitted",
    "n_spread_failed",
    "n_spread_margin_failed",
    "n_ensemble_e1_failed",
)
PHASE1 = {"lost": 32, "gained": 15, "union": 465, "churn": 47 / 465}
MAJOR = (
    "01 IT-ET Glut",
    "02 NP-CT-L6b Glut",
    "04 DG-IMN Glut",
    "06 CTX-CGE GABA",
    "07 CTX-MGE GABA",
    "09 CNU-LGE GABA",
    "11 CNU-HYa GABA",
    "12 HY GABA",
    "13 CNU-HYa Glut",
    "18 TH Glut",
    "30 Astro-Epen",
    "31 OPC-Oligo",
    "33 Vascular",
    "34 Immune",
)


def _table(directory: Path) -> pd.DataFrame:
    from merxen.annotation import resolvability as res

    return pd.read_parquet(directory / res.RESOLVABILITY_FILE)


def ensemble_rows(table: pd.DataFrame) -> pd.DataFrame:
    """The ensemble decision rows of a version-7 table."""
    return table[table["kind"].astype(str) == "decision"]


def member_rows(table: pd.DataFrame, member: str) -> pd.DataFrame:
    """One member's own decision rows of a version-7 table."""
    from merxen.annotation import resolvability as res

    frame = table[table["kind"].astype(str) == "member_decision"]
    return frame[frame[res.MEMBER_COLUMN].astype(str) == member]


def first_r1(members: Sequence[str]) -> str:
    """The first ``R1_contam_HO`` member of an ensemble."""
    for name in members:
        if str(name).startswith("R1_contam_HO@"):
            return str(name)
    raise SystemExit(f"no R1_contam_HO member in {list(members)}")


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    from merxen.annotation import resolvability as res

    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--diag-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table_a = _table(args.bundle)
    table_b = _table(args.diag_dir / "ensemble_B")
    summary_a = json.loads((args.bundle / res.RESOLVABILITY_SUMMARY_FILE).read_text())
    summary_b = json.loads(
        (args.diag_dir / "ensemble_B" / res.RESOLVABILITY_SUMMARY_FILE).read_text()
    )
    members_a = [str(name) for name in summary_a.get("emission_members") or []]
    members_b = [str(name) for name in summary_b.get("emission_members") or []]
    shared = sorted(set(members_a) & set(members_b))
    if shared:
        raise SystemExit(f"A and B share members {shared}: not independent draws")
    a = res.emitted_triples(ensemble_rows(table_a), "provisional")
    b = res.emitted_triples(ensemble_rows(table_b), "provisional")
    churn = res.triple_churn(a, b)
    broad_to_subclass = res.triple_churn(
        {triple for triple in a if triple[0] in BROAD_TO_SUBCLASS},
        {triple for triple in b if triple[0] in BROAD_TO_SUBCLASS},
    )
    routes = {
        label: {
            key: ((summary.get("ensemble") or {}).get("regimes") or {})
            .get("provisional", {})
            .get(key)
            for key in ROUTE_KEYS
        }
        for label, summary in (("A", summary_a), ("B", summary_b))
    }
    per_class = {}
    for cls in MAJOR:
        first = {triple for triple in a if triple[1] == cls}
        second = {triple for triple in b if triple[1] == cls}
        per_class[cls] = res.triple_churn(first, second)
    single_names = (first_r1(members_a), first_r1(members_b))
    single_a = res.emitted_triples(member_rows(table_a, single_names[0]), "provisional")
    single_b = res.emitted_triples(member_rows(table_b, single_names[1]), "provisional")
    single = res.triple_churn(single_a, single_b)
    passes = churn["churn"] <= TOLERANCE + 1e-12
    record: dict[str, Any] = {
        "test": "pre-registration §21 (iii)",
        "bundle": str(args.bundle),
        "diagnostic": str(args.diag_dir),
        "members_a": members_a,
        "members_b": members_b,
        "levels": sorted({triple[0] for triple in a | b}),
        "churn": churn,
        "broad_to_subclass": broad_to_subclass,
        "per_major_class": per_class,
        "single_member": {"members": list(single_names), **single},
        "routes_provisional": routes,
        "phase1_single_draws": PHASE1,
        "tolerance": TOLERANCE,
        "passes": bool(passes),
    }
    (args.out_dir / "CHURN.json").write_text(json.dumps(record, indent=2) + "\n")
    lines = [
        "M3c pre-registration §21 (iii): emitted-triple churn, fresh keyed ensemble",
        f"bundle {args.bundle}",
        f"A = {', '.join(summary_a.get('emission_members') or [])}",
        f"B = {', '.join(summary_b.get('emission_members') or [])}",
        f"levels {record['levels']}; provisional regime, unweighted PREP decisions "
        "(saturated-bp rule and monotone fill applied)",
        f"RESULT: {'PASS' if passes else 'FAIL'}: churn = ({churn['lost']} + "
        f"{churn['gained']}) / {churn['union']} = {churn['churn']:.4f} "
        f"(tolerance {TOLERANCE}); |A| {churn['n_first']}, |B| {churn['n_second']}",
        f"broad to subclass (reported): ({broad_to_subclass['lost']} + "
        f"{broad_to_subclass['gained']}) / {broad_to_subclass['union']} = "
        f"{broad_to_subclass['churn']:.4f}",
        "",
        "per level:",
        *[
            f"  {level:10s} A {item['n_first']:4d} B {item['n_second']:4d} lost "
            f"{item['lost']:3d} gained {item['gained']:3d} churn {item['churn']:.4f}"
            for level, item in churn["per_level"].items()
        ],
        "per major class:",
        *[
            f"  {cls:22s} A {item['n_first']:4d} B {item['n_second']:4d} lost "
            f"{item['lost']:3d} gained {item['gained']:3d} churn {item['churn']:.4f}"
            for cls, item in per_class.items()
        ],
        "",
        f"single member {single_names[0]} vs {single_names[1]} (version-7 "
        f"conventions): ({single['lost']} + {single['gained']}) / "
        f"{single['union']} = {single['churn']:.4f}; phase 1 (two single R1 "
        "draws): 47 / 465 = 0.101",
        "emitted bins by route (provisional regime):",
        *[
            f"  {label}: "
            + ", ".join(f"{key[2:]} {value}" for key, value in counts.items())
            for label, counts in routes.items()
        ],
    ]
    (args.out_dir / "CHURN.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if passes else 1


if __name__ == "__main__":
    sys.exit(main())

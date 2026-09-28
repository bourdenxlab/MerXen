#!/usr/bin/env python
"""M3c pre-registration §14 (v): no family is promoted by any M3c change.

Three parts (pass / fail):

1. ``validated_panels.csv`` and ``validated_panel_levels.csv`` are
   byte-identical to ``83e81e3`` (``git show``), and M3c adds no row.
2. For every self-map bundle in the stores (the SSD1 store and the large
   store: WHB, SEA-AD and WMB bundles of every family), the trust state is
   computed with ``diagnostics.trust_state`` from the bundle's own coverage
   and resolvability records and the packaged validated tables, once under
   the ``83e81e3`` code and once under the M3c code (``trust`` subcommand run
   with each tree on ``PYTHONPATH``); ``check`` requires, per bundle, the M3c
   state to be at most the ``83e81e3`` state (refused < broad_only <
   provisional < validated) with the validation basis and
   ``validated_max_level`` unchanged, and, per family, the state of its
   M3c version-7 bundle (where built: the Xenium Prime 5K mouse and human
   families) to be at most the family's ``83e81e3`` state; both 5K families
   must be ``provisional`` before and after.
3. The property tests of the fast suite (``test_real_qc.py``:
   ``apply_qc_outcomes`` and ``coverage_vs_simulation`` never raise a state;
   ``test_resolvability_v7_decisions.py``: simulation-input assets never
   raise the trust constraint, filled bins stay out of the trust tests) are
   run by ``check`` with pytest and must pass.

The family of a bundle is ``listed`` when its panel hash is a row of
``validated_panels.csv`` for its species and platforms (``built_from_panel``),
else its own: no bundle outside the listed hashes is within Jaccard 0.95 of a
listed panel (the 5K, v1 and custom panels), so inheritance does not arise.
Read-only on the stores.

Usage::

    PYTHONPATH=<83e81e3 export>/src python m3c_no_promotion.py trust --out base.json
    PYTHONPATH=src python m3c_no_promotion.py trust --out m3c.json
    PYTHONPATH=src python m3c_no_promotion.py check --baseline base.json \\
        --current m3c.json --out-dir $A/m3c/no_promotion
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger("m3c_no_promotion")

STORES = (
    Path("/media/mathieubo/SSD1/MerXen/annotation_references"),
    Path("/srv/storage/MerXen/annotation_references_large"),
)
SELF_MAP_REFERENCES = ("whb_frontal_supc_clus", "seaad_mr_panel", "wmb_panel")
BASE_COMMIT = "83e81e3"
TABLES = ("validated_panels.csv", "validated_panel_levels.csv")
PRIME_FAMILIES = ("mouse_xenium_4bc22b961aca", "human_xenium_93ecc58ed5c4")
PROPERTY_TESTS = (
    "tests/test_annotation/test_real_qc.py::test_apply_qc_outcomes_never_raises_a_trust_state",
    "tests/test_annotation/test_real_qc.py::test_coverage_vs_simulation_only_ever_warns",
    "tests/test_annotation/test_real_qc.py::test_qc_summary_never_promotes",
    "tests/test_annotation/test_resolvability_v7_decisions.py::"
    "test_simulation_input_assets_never_raise_the_trust_constraint",
    "tests/test_annotation/test_resolvability_v7_decisions.py::"
    "test_the_trust_guard_is_a_no_op_without_asset_members",
    "tests/test_annotation/test_resolvability_v7_decisions.py::"
    "test_filled_bins_take_no_part_in_the_trust_tests_or_floors",
    "tests/test_annotation/test_sim_inputs.py",
)
ORDER = ("refused", "broad_only", "provisional", "validated")


def bundle_trust(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return the trust state of one bundle under the code on the path."""
    from merxen.annotation import diagnostics as dg
    from merxen.annotation.config import AnnotationConfig
    from merxen.annotation.panel import PanelFamily

    species = str(manifest["species"])
    panel = manifest.get("panel") or {}
    panel_hash = str(panel.get("panel_hash"))
    built = manifest.get("built_from_panel") or {}
    platforms = sorted(str(item).upper() for item in built.get("platforms") or [])
    validated = dg.load_validated_panels()
    listed = [
        family
        for family in validated.known_families(species)
        if family.panel_hash == panel_hash
        and platforms
        and frozenset(platforms) <= family.platforms
    ]
    if listed:
        family = PanelFamily(
            family_id=listed[0].family_id,
            basis="listed",
            reference_panel_hash=panel_hash,
            jaccard=1.0,
            matched_platforms=platforms,
        )
    else:
        token = "_".join(item.lower() for item in platforms) or "any"
        family = PanelFamily(
            family_id=f"{species}_{token}_{panel_hash[:12]}",
            basis="own",
            reference_panel_hash=panel_hash,
            jaccard=1.0,
        )
    decision = dg.trust_state(
        reference_id=str(manifest["reference_id"]),
        role=cast("Any", manifest.get("role") or "primary"),
        species=cast("Any", species),
        panel_hash=panel_hash,
        n_panel_genes=int(panel.get("n_genes") or 0),
        family=family,
        validated=validated,
        rules=dg.TrustRules.from_config(AnnotationConfig(species=cast("Any", species))),
        coverage=dg.CoverageDiagnostics.from_bundle_manifest(manifest),
        resolvability=dg.ResolvabilityTrust.from_bundle_manifest(manifest),
    )
    resolvability = (manifest.get("builder_output") or {}).get("resolvability") or {}
    return {
        "family_id": family.family_id,
        "family_basis": family.basis,
        "state": decision.state,
        "validation_basis": decision.validation_basis,
        "validated_max_level": getattr(decision, "validated_max_level", None),
        "resolvability_version": resolvability.get("resolvability_version"),
        "resolvability_constraint": (resolvability.get("trust") or {}).get("state"),
    }


def command_trust(out: Path) -> int:
    """Compute the trust state of every self-map bundle (code on the path)."""
    import merxen

    rows = []
    for store in STORES:
        for reference_id in SELF_MAP_REFERENCES:
            directory = store / reference_id
            if not directory.is_dir():
                continue
            for bundle in sorted(directory.iterdir()):
                path = bundle / "bundle.json"
                if not path.is_file():
                    continue
                manifest = json.loads(path.read_text())
                try:
                    record = bundle_trust(manifest)
                except Exception as error:  # noqa: BLE001 - recorded per bundle
                    record = {"error": f"{type(error).__name__}: {error}"}
                rows.append(
                    {
                        "store": str(store),
                        "reference_id": reference_id,
                        "build_hash": manifest["build_hash"],
                        "created_at": manifest.get("created_at"),
                        "panel_hash": (manifest.get("panel") or {}).get("panel_hash"),
                        "n_genes": (manifest.get("panel") or {}).get("n_genes"),
                        **record,
                    }
                )
    out.write_text(
        json.dumps(
            {"code": str(Path(merxen.__file__).resolve().parents[1]), "bundles": rows},
            indent=2,
        )
        + "\n"
    )
    return 0


def _rank(state: str | None) -> int:
    return ORDER.index(state) if state in ORDER else -1


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def check_tables() -> dict[str, Any]:
    """Compare the validated tables with ``83e81e3``."""
    from merxen.annotation.vocab import ASSET_DIR

    result = {}
    for name in TABLES:
        current = (Path(ASSET_DIR) / name).read_bytes()
        base = subprocess.run(
            [
                "git",
                "-C",
                str(_repo_root()),
                "show",
                f"{BASE_COMMIT}:src/merxen/assets/annotation/{name}",
            ],
            capture_output=True,
            check=True,
        ).stdout
        result[name] = {
            "sha256_83e81e3": hashlib.sha256(base).hexdigest(),
            "sha256_m3c": hashlib.sha256(current).hexdigest(),
            "identical": base == current,
        }
    return result


def run_property_tests(out_dir: Path) -> dict[str, Any]:
    """Run the no-promotion property tests with pytest."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            *PROPERTY_TESTS,
        ],
        cwd=str(_repo_root()),
        capture_output=True,
        text=True,
        check=False,
    )
    (out_dir / "property_tests.log").write_text(result.stdout + result.stderr)
    tail = [line for line in result.stdout.splitlines() if line.strip()][-1:]
    return {"passed": result.returncode == 0, "summary": tail[0] if tail else ""}


def command_check(baseline: Path, current: Path, out_dir: Path) -> int:
    """Run the three parts and write the report."""
    out_dir.mkdir(parents=True, exist_ok=True)
    base = {
        row["build_hash"]: row for row in json.loads(baseline.read_text())["bundles"]
    }
    now_rows = json.loads(current.read_text())["bundles"]
    tables = check_tables()
    per_bundle = []
    for row in now_rows:
        before = base.get(row["build_hash"])
        if before is None:
            continue
        per_bundle.append(
            {
                "reference_id": row["reference_id"],
                "build_hash": row["build_hash"][:12],
                "family_id": row.get("family_id"),
                "state_83e81e3": before.get("state"),
                "state_m3c": row.get("state"),
                "basis_83e81e3": before.get("validation_basis"),
                "basis_m3c": row.get("validation_basis"),
                "max_level_equal": before.get("validated_max_level")
                == row.get("validated_max_level"),
                "ok": _rank(row.get("state")) <= _rank(before.get("state"))
                and before.get("validation_basis") == row.get("validation_basis")
                and before.get("validated_max_level") == row.get("validated_max_level")
                and "error" not in row,
            }
        )
    new_bundles = [row for row in now_rows if row["build_hash"] not in base]
    families = []
    for row in new_bundles:
        family = row.get("family_id")
        before = [
            item
            for item in base.values()
            if item.get("family_id") == family
            and item.get("reference_id") == row["reference_id"]
        ]
        best_before = max((_rank(item.get("state")) for item in before), default=-1)
        families.append(
            {
                "family_id": family,
                "reference_id": row["reference_id"],
                "new_bundle": row["build_hash"][:12],
                "resolvability_version": row.get("resolvability_version"),
                "state_m3c": row.get("state"),
                "states_83e81e3": sorted({str(item.get("state")) for item in before}),
                "ok": bool(before) and _rank(row.get("state")) <= best_before,
            }
        )
    prime = {
        family: {
            "before": sorted(
                {
                    str(item.get("state"))
                    for item in base.values()
                    if item.get("family_id") == family
                }
            ),
            "after_new_bundles": sorted(
                {
                    str(item.get("state"))
                    for item in new_bundles
                    if item.get("family_id") == family
                }
            ),
        }
        for family in PRIME_FAMILIES
    }
    prime_ok = all(
        record["before"] == ["provisional"]
        and record["after_new_bundles"] == ["provisional"]
        for record in prime.values()
    )
    properties = run_property_tests(out_dir)
    passes = bool(
        all(item["identical"] for item in tables.values())
        and all(item["ok"] for item in per_bundle)
        and all(item["ok"] for item in families)
        and prime_ok
        and properties["passed"]
    )
    report = {
        "test": "pre-registration §14 (v)",
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "tables": tables,
        "bundles": per_bundle,
        "new_bundles": families,
        "prime_families": prime,
        "property_tests": properties,
        "passes": passes,
    }
    (out_dir / "NO_PROMOTION.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "M3c pre-registration §14 (v): no family promoted by any M3c change",
        f"RESULT: {'PASS' if passes else 'FAIL'}",
        "",
        "1. validated tables vs 83e81e3:",
    ]
    for name, item in tables.items():
        lines.append(
            f"   {name}: {'identical' if item['identical'] else 'CHANGED'} "
            f"(sha256 {item['sha256_m3c'][:16]})"
        )
    n_ok = sum(1 for item in per_bundle if item["ok"])
    lines.append(
        f"2. {len(per_bundle)} existing self-map bundles: trust under M3c <= 83e81e3 "
        f"with basis and validated_max_level unchanged in {n_ok}"
    )
    counts: dict[tuple[str, str], int] = {}
    for item in per_bundle:
        key = (str(item["state_83e81e3"]), str(item["state_m3c"]))
        counts[key] = counts.get(key, 0) + 1
    for (before, after), count in sorted(counts.items()):
        lines.append(f"   {before} -> {after}: {count}")
    for item in per_bundle:
        if not item["ok"]:
            lines.append(f"   NOT OK: {item}")
    lines.append("   new (version-7) bundles vs their family's 83e81e3 bundles:")
    for item in families:
        lines.append(
            f"   {item['family_id']} {item['reference_id']} {item['new_bundle']} "
            f"(v{item['resolvability_version']}): {item['state_m3c']} vs "
            f"{item['states_83e81e3']} -> {'ok' if item['ok'] else 'NOT OK'}"
        )
    for family, record in prime.items():
        lines.append(
            f"   {family}: before {record['before']}, "
            f"after {record['after_new_bundles']}"
        )
    lines.append(
        f"3. property tests: {'PASS' if properties['passed'] else 'FAIL'} "
        f"({properties['summary']})"
    )
    (out_dir / "NO_PROMOTION.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if passes else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    trust = sub.add_parser("trust", help="trust state of every bundle")
    trust.add_argument("--out", type=Path, required=True)
    check = sub.add_parser("check", help="tables, states and property tests")
    check.add_argument("--baseline", type=Path, required=True)
    check.add_argument("--current", type=Path, required=True)
    check.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    if args.command == "trust":
        return command_trust(args.out)
    return command_check(args.baseline, args.current, args.out_dir)


if __name__ == "__main__":
    sys.exit(main())

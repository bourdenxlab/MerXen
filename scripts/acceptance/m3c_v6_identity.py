#!/usr/bin/env python
"""M3c pre-registration §14 (i-b): the stored version-6 bundles stay byte-identical.

For each version-6 production bundle of the SSD1 store (WHB set a 297
``b6bfe83d``, WHB set c 265 ``5137090d``, SEA-AD ``d70cda25``, WMB ag7
``5a032858``, WMB VZG2 ``daa8c4a6``; the list is confirmed from each
``bundle.json``: resolvability version 6 and a panel of ``validated_panels.csv``)
this script checks, with the M3c code and without any mapping:

1. **build hash** -- the bundle's spec, re-read from ``bundle.json`` (reference
   settings, sources, panel, ctm), gives the stored ``build_hash_payload`` and
   ``build_hash`` through ``store.build_hash_payload`` (so no rebuild is
   triggered and the bundle is reused);
2. **decisions** -- ``decide()`` re-run on the stored
   ``resolvability_cells.parquet`` with the stored settings
   (``load_resolvability(...).decisions()``) reproduces the stored decision
   rows of ``resolvability.parquet`` (canonical CSV sha256, every regime);
3. **simulated query** -- re-simulating each recipe (``R1_contam_HO`` and
   ``clean``, seed 0) from the bundle's stored test set gives the same CSR
   arrays, index and obs (sha256 as in (i-a) (1)) as the ``83e81e3`` code on
   the same inputs.

Step 3 runs this script twice: ``hashes`` under each code tree (``PYTHONPATH``
of the ``83e81e3`` export, then of the M3c tree), and ``check`` compares.

Read-only on the stores; writes only ``--out`` / ``--out-dir``.

Usage::

    PYTHONPATH=<83e81e3 export>/src python m3c_v6_identity.py hashes --out base.json
    PYTHONPATH=src python m3c_v6_identity.py hashes --out m3c.json
    PYTHONPATH=src python m3c_v6_identity.py check --baseline base.json \\
        --current m3c.json --out-dir $A/m3c/v6_identity
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd

logger = logging.getLogger("m3c_v6_identity")

STORE = Path("/media/mathieubo/SSD1/MerXen/annotation_references")
# The version-6 production bundles of 2026-09-28 (pre-registration §14 (i-b)).
PREREGISTERED: dict[str, tuple[str, str]] = {
    "whb_set_a_297": ("whb_frontal_supc_clus", "b6bfe83d"),
    "whb_set_c_265": ("whb_frontal_supc_clus", "5137090d"),
    "seaad_set_a": ("seaad_mr_panel", "d70cda25"),
    "wmb_ag7": ("wmb_panel", "5a032858"),
    "wmb_vzg2": ("wmb_panel", "daa8c4a6"),
}
SELF_MAP_REFERENCES = ("whb_frontal_supc_clus", "seaad_mr_panel", "wmb_panel")


def validated_panels_path() -> Path:
    """Return the packaged ``validated_panels.csv`` of the code on the path."""
    from merxen.annotation.vocab import ASSET_DIR

    return Path(ASSET_DIR) / "validated_panels.csv"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _cell(value: Any) -> str:
    """Return the canonical text of one table value (as the (i-a) test)."""
    if value is None or value is pd.NA:
        return ""
    if isinstance(value, bool | np.bool_):
        return "True" if bool(value) else "False"
    if isinstance(value, int | np.integer):
        return str(int(value))
    if isinstance(value, float | np.floating):
        number = float(value)
        return "" if math.isnan(number) else f"{number:.17g}"
    if isinstance(value, list | tuple | dict | np.ndarray):
        items = value.tolist() if isinstance(value, np.ndarray) else value
        return json.dumps(items, sort_keys=True, default=str)
    return str(value)


def canonical_csv(frame: pd.DataFrame, sort_by: Sequence[str] | None = None) -> str:
    """Return a frame as canonical CSV text (the (i-a) test's convention)."""
    table = frame.reset_index(drop=True)
    if sort_by is not None:
        table = table.sort_values(list(sort_by), kind="mergesort").reset_index(
            drop=True
        )
    text = [[_cell(value) for value in row] for row in table.itertuples(index=False)]
    if sort_by is None:
        text.sort()
    lines = [",".join(str(column) for column in table.columns)]
    lines += [",".join(row) for row in text]
    return "\n".join(lines) + "\n"


def query_hashes(query: Any) -> dict[str, str]:
    """Return the sha256 of a simulated query's CSR arrays, index and obs."""
    matrix = query.counts.tocsr()
    payload = (
        np.ascontiguousarray(matrix.data, dtype="<f8").tobytes()
        + np.ascontiguousarray(matrix.indices, dtype="<i4").tobytes()
        + np.ascontiguousarray(matrix.indptr, dtype="<i8").tobytes()
    )
    index = "\n".join(str(value) for value in query.obs.index).encode("utf-8")
    return {
        "csr": _sha256(payload),
        "index": _sha256(index),
        "obs": _sha256(canonical_csv(query.obs.reset_index()).encode("utf-8")),
        "n_cells": str(len(query.obs)),
    }


def find_bundle(reference_id: str, prefix: str) -> Path:
    """Return the store directory of a bundle given its hash prefix."""
    matches = sorted((STORE / reference_id).glob(f"{prefix}*"))
    if len(matches) != 1:
        raise SystemExit(f"{reference_id}/{prefix}: {len(matches)} matches")
    return matches[0]


def read_manifest(bundle: Path) -> dict[str, Any]:
    return cast("dict[str, Any]", json.loads((bundle / "bundle.json").read_text()))


def test_set_dir(bundle: Path, manifest: dict[str, Any]) -> Path:
    """Return the test-set bundle of a self-map bundle (same store root)."""
    record = (manifest.get("builder_output") or {}).get("resolvability") or {}
    test = record.get("test_set_bundle") or {}
    return STORE / str(test["reference_id"]) / str(test["build_hash"])


def list_v6_bundles() -> list[dict[str, Any]]:
    """Return every self-map bundle of the store with resolvability version 6."""
    listed = pd.read_csv(validated_panels_path())
    hashes = {
        str(value): str(pid)
        for value, pid in zip(listed["panel_hash"], listed["panel_id"], strict=True)
    }
    rows: list[dict[str, Any]] = []
    for reference_id in SELF_MAP_REFERENCES:
        for bundle in sorted((STORE / reference_id).iterdir()):
            if not (bundle / "bundle.json").is_file():
                continue
            manifest = read_manifest(bundle)
            summary_path = bundle / "resolvability_summary.json"
            version = None
            if summary_path.is_file():
                version = json.loads(summary_path.read_text()).get(
                    "resolvability_version"
                )
            panel_hash = str((manifest.get("panel") or {}).get("panel_hash"))
            rows.append(
                {
                    "reference_id": reference_id,
                    "build_hash": manifest["build_hash"],
                    "panel_hash": panel_hash,
                    "panel_id": hashes.get(panel_hash),
                    "n_genes": (manifest.get("panel") or {}).get("n_genes"),
                    "resolvability_version": version,
                    "builder_version": manifest.get("builder_version"),
                    "created_at": manifest.get("created_at"),
                }
            )
    return rows


# --------------------------------------------------------------------------
# hashes: simulated queries under the code tree on PYTHONPATH


def command_hashes(out: Path) -> int:
    """Hash each pre-registered bundle's re-simulated recipes (this code tree)."""
    from merxen.annotation import resolvability as res
    from merxen.annotation.config import AnnotationResolvabilityConfig

    code_root = Path(res.__file__).resolve().parents[3]
    started = time.monotonic()
    result: dict[str, Any] = {"code_root": str(code_root), "bundles": {}}
    cache: dict[str, dict[str, Any]] = {}
    for key, (reference_id, prefix) in PREREGISTERED.items():
        bundle = find_bundle(reference_id, prefix)
        manifest = read_manifest(bundle)
        grid = [int(value) for value in manifest["build_hash_payload"]["depth_grid"]]
        test_dir = test_set_dir(bundle, manifest)
        recipes = res.simulation_recipes(AnnotationResolvabilityConfig(), seed=0)
        cache_key = f"{test_dir}|{grid}"
        if cache_key not in cache:
            test = res.load_test_cells(test_dir)
            entry: dict[str, Any] = {
                "test_set": str(test_dir),
                "n_test_cells": len(test.obs),
                "grid": grid,
                "recipes": {},
            }
            for recipe in recipes:
                step = time.monotonic()
                query = res.thin_and_contaminate(test, grid, recipe)
                entry["recipes"][recipe.name] = {
                    **query_hashes(query),
                    "recipe": recipe.to_json(),
                    "seconds": round(time.monotonic() - step, 1),
                }
                logger.info(
                    "%s %s: %s cells, csr %s",
                    key,
                    recipe.name,
                    entry["recipes"][recipe.name]["n_cells"],
                    entry["recipes"][recipe.name]["csr"][:12],
                )
            cache[cache_key] = entry
        result["bundles"][key] = {
            "bundle": str(bundle),
            **cache[cache_key],
            "stored_recipes": (
                manifest["build_hash_payload"]["builder_params"]
                .get("resolvability", {})
                .get("recipes")
            ),
        }
    result["wall_s"] = round(time.monotonic() - started, 1)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


# --------------------------------------------------------------------------
# check: build hashes and decisions under the M3c code, and the hash comparison


def _source_records(payload: dict[str, Any]) -> dict[str, Any]:
    from merxen.annotation.store import FileIdentity, SourceRecord

    records = {}
    for name, item in payload["sources"].items():
        records[name] = SourceRecord(
            name=name,
            path=str(item["path"]),
            kind=str(item["kind"]),
            files=tuple(FileIdentity(**dict(entry)) for entry in item["files"]),
            pattern=item.get("pattern"),
        )
    return records


def check_build_hash(bundle: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Recompute a bundle's build-hash payload with this code (no source read)."""
    from merxen.annotation.config import (
        KNOWN_REFERENCES,
        AnnotationConfig,
        AnnotationReferenceSpec,
    )
    from merxen.annotation.reference import builder_for
    from merxen.annotation.store import build_hash_payload, compute_build_hash

    stored = manifest["build_hash_payload"]
    reference_id = str(stored["reference_id"])
    species = str(stored["species"])
    resolvability = stored["builder_params"].get("resolvability") or {}
    config = AnnotationConfig(
        species=cast("Any", species),
        anatomical_region=resolvability.get("anatomical_region"),
    )
    known = dict(KNOWN_REFERENCES[reference_id])
    spec = AnnotationReferenceSpec(
        reference_id=reference_id,
        **{
            **known,
            "hierarchy": list(stored["hierarchy"]),
            "nodes_to_drop": list(stored["nodes_to_drop"]),
            "drop_level": stored["drop_level"],
            "n_per_utility": stored["n_per_utility"],
            "max_cells_per_cluster": stored["max_cells_per_cluster"],
            "sources": {name: item["path"] for name, item in stored["sources"].items()},
        },
    )
    panel = SimpleNamespace(
        panel_hash=str(stored["panel"]["panel_hash"]),
        n_genes=int(stored["panel"]["n_genes"]),
        species=species,
    )
    builder = builder_for(spec, config)
    payload = build_hash_payload(
        spec,
        cast("Any", panel),
        builder=builder,
        sources=_source_records(stored),
        config=config,
        ctm=stored["ctm"],
    )
    recomputed = compute_build_hash(payload)
    differing = sorted(
        key
        for key in set(payload) | set(stored)
        if json.dumps(payload.get(key), sort_keys=True, default=str)
        != json.dumps(stored.get(key), sort_keys=True, default=str)
    )
    return {
        "stored_build_hash": manifest["build_hash"],
        "recomputed_build_hash": recomputed,
        "equal": recomputed == manifest["build_hash"] and not differing,
        "differing_payload_keys": differing,
    }


def check_decisions(bundle: Path) -> dict[str, Any]:
    """Re-derive the unweighted decisions from the stored cells and compare."""
    from merxen.annotation import resolvability as res

    tables = res.load_resolvability(bundle)
    if tables is None:
        return {"equal": False, "reason": "no resolvability tables"}
    started = time.monotonic()
    rederived = tables.decisions()
    stored_table = pd.read_parquet(bundle / res.RESOLVABILITY_FILE)
    stored = stored_table[stored_table["kind"].astype(str) == "decision"]
    # The table stores the decision columns of TABLE_COLUMNS (the others are
    # kept in the summary), so the comparison is on those.
    columns = [column for column in res.DECISION_COLUMNS if column in stored.columns]
    keys = ["regime", "level", "class", "depth"]
    left = canonical_csv(stored[columns], sort_by=keys)
    right = canonical_csv(rederived[columns], sort_by=keys)
    per_regime = {}
    for regime in sorted(set(stored["regime"].astype(str))):
        a = canonical_csv(stored[stored["regime"].astype(str) == regime][columns], keys)
        b = canonical_csv(
            rederived[rederived["regime"].astype(str) == regime][columns], keys
        )
        per_regime[regime] = {
            "stored_sha256": _sha256(a.encode()),
            "rederived_sha256": _sha256(b.encode()),
            "equal": a == b,
            "n_rows": int((stored["regime"].astype(str) == regime).sum()),
        }
    return {
        "resolvability_version": tables.version,
        "stored_sha256": _sha256(left.encode()),
        "rederived_sha256": _sha256(right.encode()),
        "equal": left == right,
        "n_rows": len(stored),
        "columns": columns,
        "per_regime": per_regime,
        "seconds": round(time.monotonic() - started, 1),
    }


def compare_hashes(baseline: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    """Compare the simulated-query hashes of the two code trees."""
    result: dict[str, Any] = {}
    for key in PREREGISTERED:
        base = baseline["bundles"].get(key)
        now = current["bundles"].get(key)
        if base is None or now is None:
            result[key] = {"equal": False, "reason": "missing"}
            continue
        recipes = {}
        for name in sorted(set(base["recipes"]) | set(now["recipes"])):
            a = base["recipes"].get(name) or {}
            b = now["recipes"].get(name) or {}
            recipes[name] = {
                field: {"83e81e3": a.get(field), "m3c": b.get(field)}
                for field in ("csr", "index", "obs", "n_cells")
            }
            recipes[name]["equal"] = all(
                a.get(field) == b.get(field) and a.get(field) is not None
                for field in ("csr", "index", "obs", "n_cells")
            )
            recipes[name]["recipe_equal"] = a.get("recipe") == b.get("recipe")
        result[key] = {
            "equal": all(
                item["equal"] and item["recipe_equal"] for item in recipes.values()
            ),
            "recipes": recipes,
            "stored_recipes_match": base.get("stored_recipes")
            == [item["recipe"] for item in now["recipes"].values()],
        }
    return result


def _git_head() -> str | None:
    """Return the commit of the ``merxen`` package on the path, if in a repo."""
    import merxen

    root = Path(merxen.__file__).resolve().parents[2]
    try:
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def command_check(baseline: Path, current: Path, out_dir: Path) -> int:
    """Run checks 1-2 and compare the hashes of check 3; write the report."""
    out_dir.mkdir(parents=True, exist_ok=True)
    listing = list_v6_bundles()
    base = json.loads(baseline.read_text())
    now = json.loads(current.read_text())
    queries = compare_hashes(base, now)
    bundles: dict[str, Any] = {}
    for key, (reference_id, prefix) in PREREGISTERED.items():
        bundle = find_bundle(reference_id, prefix)
        manifest = read_manifest(bundle)
        summary = json.loads((bundle / "resolvability_summary.json").read_text())
        version = summary.get("resolvability_version")
        panel_hash = str(manifest["panel"]["panel_hash"])
        listed = panel_hash in set(
            pd.read_csv(validated_panels_path())["panel_hash"].astype(str)
        )
        logger.info("%s: build hash", key)
        build = check_build_hash(bundle, manifest)
        logger.info("%s: decisions", key)
        decisions = check_decisions(bundle)
        bundles[key] = {
            "bundle": str(bundle),
            "resolvability_version": version,
            "listed_family": listed,
            "confirmed": version == 6 and listed,
            "build_hash": build,
            "decisions": decisions,
            "simulated_query": queries[key],
            "passes": bool(
                version == 6
                and listed
                and build["equal"]
                and decisions["equal"]
                and queries[key]["equal"]
            ),
        }
    passes = all(item["passes"] for item in bundles.values())
    report = {
        "test": "pre-registration §14 (i-b)",
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "code_commit": _git_head(),
        "baseline_code": base.get("code_root"),
        "current_code": now.get("code_root"),
        "store": str(STORE),
        "v6_bundles_in_store": [
            row for row in listing if row["resolvability_version"] == 6
        ],
        "bundles": bundles,
        "passes": passes,
    }
    (out_dir / "V6_IDENTITY.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "M3c pre-registration §14 (i-b): stored version-6 bundles under the M3c code",
        f"code {report['code_commit']}; baseline {report['baseline_code']}; "
        f"store {STORE}",
        f"RESULT: {'PASS' if passes else 'FAIL'}",
        "",
    ]
    for key, item in bundles.items():
        build = item["build_hash"]
        decisions = item["decisions"]
        lines.append(
            f"{key}: {'PASS' if item['passes'] else 'FAIL'} "
            f"(v{item['resolvability_version']}, listed {item['listed_family']}); "
            f"build_hash {build['stored_build_hash'][:12]} "
            f"{'==' if build['equal'] else '!='} {build['recomputed_build_hash'][:12]}"
            + (
                f" differing {build['differing_payload_keys']}"
                if build["differing_payload_keys"]
                else ""
            )
        )
        lines.append(
            f"   decisions {decisions.get('n_rows')} rows: stored "
            f"{str(decisions.get('stored_sha256'))[:12]} vs re-derived "
            f"{str(decisions.get('rederived_sha256'))[:12]} "
            f"({'equal' if decisions.get('equal') else 'DIFFERENT'})"
        )
        for name, recipe in item["simulated_query"].get("recipes", {}).items():
            lines.append(
                f"   {name}: {recipe['n_cells']['m3c']} simulated cells; csr "
                f"{str(recipe['csr']['83e81e3'])[:12]} (83e81e3) vs "
                f"{str(recipe['csr']['m3c'])[:12]} (M3c); "
                f"{'equal' if recipe['equal'] else 'DIFFERENT'} (csr, index, obs); "
                f"recipe record {'equal' if recipe['recipe_equal'] else 'DIFFERENT'}"
            )
    lines.append("")
    lines.append(
        "version-6 self-map bundles in the store: "
        + ", ".join(
            f"{row['reference_id']}/{row['build_hash'][:8]} ({row['panel_id']})"
            for row in report["v6_bundles_in_store"]
        )
    )
    (out_dir / "V6_IDENTITY.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if passes else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    hashes = sub.add_parser("hashes", help="hash the re-simulated recipes")
    hashes.add_argument("--out", type=Path, required=True)
    check = sub.add_parser("check", help="build hashes, decisions, comparison")
    check.add_argument("--baseline", type=Path, required=True)
    check.add_argument("--current", type=Path, required=True)
    check.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    if args.command == "hashes":
        return command_hashes(args.out)
    return command_check(args.baseline, args.current, args.out_dir)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""M3c pre-registration §14 (iii) and (vii): version-7 ensemble draws beside a bundle.

For each target bundle, ``simulate.run_v7_diagnostic`` is run on the bundle's
own test set and engine and written to ``--out-dir/<target>/v7_diagnostic``
(never to a store; the bundle is only read):

* **(vii) the version-6 families** (``whb_set_a_297``, ``whb_set_c_265``,
  ``seaad_set_a``, ``wmb_ag7``, ``wmb_vzg2``; the SSD1 bundles of (i-b)):
  ensemble A = ``R1_contam_HO@0``, ``@1``, ``@2`` with the version-7
  conventions (no top-up: the bundle's own test set), compared with the
  stored version-6 decisions (emitted triples lost and gained per level and
  regime), and a fresh ensemble B = ``R1_contam_HO@3``, ``@4``, ``@5`` (the
  version-7 churn under a fresh draw). Diagnostic only, never applied.
* **(iii) the version-7 5K mouse bundle** (``prime5k_mouse``; ``--bundle``):
  ensemble A is the bundle's own (``R1_contam_HO@0-2`` and
  ``R3_measured_HO@0`` on the topped-up test set), ensemble B the fresh keyed
  draw ``R1_contam_HO@3``, ``@4``, ``@5`` and ``R3_measured_HO@1``.

Each target's wall time and process-tree peak (PSS and RSS, sampled every
5 s) are recorded in ``<target>/resources.json``. MapMyCells runs on
``--n-processors`` (6) processes with the production mapping configuration
and mapping seed 0.

Usage::

    python scripts/acceptance/m3c_v7_diagnostic.py --out-dir $A/m3c/v7_diagnostic \\
        --targets whb_set_a_297,whb_set_c_265,seaad_set_a,wmb_ag7,wmb_vzg2

    python scripts/acceptance/m3c_v7_diagnostic.py --out-dir $A/m3c/v7_churn \\
        --targets prime5k_mouse --bundle /srv/storage/.../wmb_panel/<hash>
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd

logger = logging.getLogger("m3c_v7_diagnostic")

STORE = Path("/media/mathieubo/SSD1/MerXen/annotation_references")
LARGE = Path("/srv/storage/MerXen/annotation_references_large")
# name -> (reference id, bundle hash prefix, validated_panel_genes.csv panel id,
# platform of the chemistry resolution)
V6_TARGETS: dict[str, tuple[str, str, str, str | None]] = {
    "whb_set_a_297": ("whb_frontal_supc_clus", "b6bfe83d", "human_set_a_297", None),
    "whb_set_c_265": ("whb_frontal_supc_clus", "5137090d", "human_set_c_265", None),
    "seaad_set_a": ("seaad_mr_panel", "d70cda25", "human_set_a_297", None),
    "wmb_ag7": ("wmb_panel", "5a032858", "mouse_ag7_500", "MERSCOPE"),
    "wmb_vzg2": ("wmb_panel", "daa8c4a6", "mouse_vzg2_815", "MERSCOPE"),
}
FRESH_R1_SEEDS: tuple[int, ...] = (3, 4, 5)
FRESH_R3_SEED = 1


def _bundle(reference_id: str, prefix: str) -> Path:
    matches = sorted((STORE / reference_id).glob(f"{prefix}*"))
    if len(matches) != 1:
        raise SystemExit(f"{reference_id}/{prefix}: {len(matches)} bundles")
    return matches[0]


def _test_set_dir(bundle: Path, manifest: dict[str, Any]) -> Path:
    record = (manifest.get("builder_output") or {}).get("resolvability") or {}
    test = record.get("test_set_bundle") or {}
    return bundle.parent.parent / str(test["reference_id"]) / str(test["build_hash"])


def _panel_genes(panel_id: str) -> list[str]:
    from merxen.annotation.vocab import ASSET_DIR

    table = pd.read_csv(Path(ASSET_DIR) / "validated_panel_genes.csv")
    return sorted(table.loc[table["panel_id"] == panel_id, "ensembl_id"].astype(str))


def run_target(
    name: str,
    bundle: Path,
    *,
    genes: Sequence[str] | None,
    platform: str | None,
    out_dir: Path,
    scratch_dir: Path,
) -> dict[str, Any]:
    """Run the version-7 diagnostic of one bundle and record its resources."""
    from merxen.annotation import sim_inputs as si
    from merxen.annotation import simulate
    from merxen.annotation.config import (
        KNOWN_REFERENCES,
        AnnotationConfig,
        AnnotationReferenceSpec,
    )
    from merxen.annotation.memory import ProcessTreeSampler

    manifest = json.loads((bundle / "bundle.json").read_text())
    payload = manifest["build_hash_payload"]
    reference_id = str(manifest["reference_id"])
    species = str(manifest["species"])
    resolvability_params = payload["builder_params"].get("resolvability") or {}
    config = AnnotationConfig(
        species=cast("Any", species),
        anatomical_region=resolvability_params.get("anatomical_region"),
    )
    mapping = (manifest.get("recorded_settings") or {}).get("mapping") or {}
    spec = AnnotationReferenceSpec(
        reference_id=reference_id,
        **{
            **dict(KNOWN_REFERENCES[reference_id]),
            "hierarchy": list(payload["hierarchy"]),
            "nodes_to_drop": list(payload["nodes_to_drop"]),
            "drop_level": payload["drop_level"],
            "n_per_utility": payload["n_per_utility"],
            "max_cells_per_cluster": payload["max_cells_per_cluster"],
            **{
                key: mapping[key]
                for key in ("bootstrap_factor", "bootstrap_iteration", "rng_seed")
                if key in mapping
            },
        },
    )
    panel_genes = list(genes) if genes is not None else []
    if not panel_genes:
        test_dir = _test_set_dir(bundle, manifest)
        from merxen.annotation import resolvability as res

        panel_genes = list(res.load_test_cells(test_dir).genes)
    panel = SimpleNamespace(
        panel_hash=str(manifest["panel"]["panel_hash"]),
        n_genes=int(manifest["panel"]["n_genes"]),
        species=species,
        ensembl_ids=panel_genes,
    )
    chemistry = si.resolve_chemistry(panel_genes, species=species, platform=platform)
    target_out = out_dir / name
    target_out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    with ProcessTreeSampler(interval_s=5.0) as sampler:
        record, _rows = simulate.run_v7_diagnostic(
            reference_id=reference_id,
            spec=spec,
            panel=cast("Any", panel),
            config=config,
            bundle_dir=bundle,
            test_set_dir=_test_set_dir(bundle, manifest),
            resolvability=(manifest.get("builder_output") or {}).get("resolvability")
            or {},
            chemistry=chemistry,
            out_dir=target_out,
            scratch_dir=scratch_dir / name,
            store_roots=[STORE, LARGE],
            fresh_seeds=FRESH_R1_SEEDS,
            fresh_r3_seed=FRESH_R3_SEED,
        )
    peak = sampler.peak()
    resources = {
        "target": name,
        "bundle": str(bundle),
        "started_at": started_at,
        "wall_s": round(time.monotonic() - started, 1),
        "peak_tree_pss_gb": None
        if peak.peak_pss_gb is None
        else round(peak.peak_pss_gb, 3),
        "peak_tree_rss_gb": round(peak.peak_rss_gb, 3),
        "peak_processes": peak.peak_processes,
        "n_samples": peak.n_samples,
        "sample_interval_s": peak.interval_s,
        "chemistry": chemistry.chemistry,
        "mapping_runs": [
            {
                key: item.get(key)
                for key in ("tag", "n_cells", "wall_s", "peak_rss_gb", "n_processors")
                if key in item
            }
            for item in record.get("mapping_runs") or []
        ],
    }
    (target_out / "resources.json").write_text(json.dumps(resources, indent=2) + "\n")
    logger.info(
        "%s done in %.0f s (peak tree PSS %s GB)",
        name,
        resources["wall_s"],
        resources["peak_tree_pss_gb"],
    )
    return {"record_file": str(target_out / "v7_diagnostic"), **resources}


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, default=None)
    parser.add_argument("--targets", required=True)
    parser.add_argument(
        "--bundle", type=Path, default=None, help="the prime5k_mouse bundle"
    )
    parser.add_argument("--n-processors", type=int, default=6)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    from merxen.annotation.reference import set_prep_resources

    set_prep_resources(n_processors=args.n_processors)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    scratch = args.scratch_dir or args.out_dir / "scratch"
    results = []
    for name in [item.strip() for item in args.targets.split(",") if item.strip()]:
        if name == "prime5k_mouse":
            if args.bundle is None:
                raise SystemExit("prime5k_mouse needs --bundle")
            results.append(
                run_target(
                    name,
                    args.bundle,
                    genes=None,
                    platform="XENIUM",
                    out_dir=args.out_dir,
                    scratch_dir=scratch,
                )
            )
            continue
        reference_id, prefix, panel_id, platform = V6_TARGETS[name]
        results.append(
            run_target(
                name,
                _bundle(reference_id, prefix),
                genes=_panel_genes(panel_id),
                platform=platform,
                out_dir=args.out_dir,
                scratch_dir=scratch,
            )
        )
        (args.out_dir / "RESOURCES.json").write_text(
            json.dumps(results, indent=2) + "\n"
        )
    (args.out_dir / "RESOURCES.json").write_text(json.dumps(results, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

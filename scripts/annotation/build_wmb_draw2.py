#!/usr/bin/env python
"""Build gate P's second disjoint WMB test draw for a mouse panel (M13b C20a).

Gate P on a mouse family scores "the two disjoint test draws" (plan §14 NP4,
C_P): draw 1 is the test set the production ``wmb_panel`` self-map built,
draw 2 its own ``wmb_selfmap_testset`` bundle built with
``resolvability.gate_p_wmb_draw`` 2 (``reference.wmb_second_draw``). This
script builds draw 2 next to draw 1 in a reference store, from the same
inputs the pipeline's ``ANNOTATE_REFERENCE_PREP`` passes for ``wmb_panel``
(``merxen annotation-reference-prep --reference-id wmb_panel``): the same
annotation config, panel file, sources and download cache. It derives the
test set's spec from the prepared ``wmb_panel`` spec
(``reference.wmb_test_set_spec``), exactly as gate P does after its PREP, so
gate P finds the same ``build_hash``.

Before building it checks that draw 1 is in the store, complete, with the
``build_hash`` this spec gives (``--expect-draw1`` and ``--expect-panel-hash``
also pin the expected values), so draw 2 is drawn against the draw 1 the
production self-map used; draw 2 recomputes draw 1's cells itself and records
their digest, and this script compares that digest with the stored draw 1's
cells after the build. ``bundle_ref.json`` of draw 2 and a small JSON record
(``--record``: both hashes, the digests and the per-class availability) are
written.

Usage (the ag7 panel, as B1's PREP)::

    python scripts/annotation/build_wmb_draw2.py \\
        --annotation-config annotation_config.json \\
        --panel-genes panel_genes.json \\
        --store /srv/storage/MerXen/annotation_references \\
        --store-large /srv/storage/MerXen/annotation_references_large \\
        --scratch-dir prep_scratch \\
        --source wmb_h5ad_dir=<WMB-10Xv3 dir> \\
        --source wmb_mapping_stats=<precomputed stats h5> \\
        --source wmb_metadata_dir=<ABC metadata dir> \\
        --source wmb_selfmap_test_cells=<truth csv> \\
        --expect-panel-hash f4c291864ba1c911... \\
        --output bundle_ref_draw2.json --record draw2_record.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _key_value_paths(values: Sequence[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        key, separator, path = value.partition("=")
        if not separator or not key.strip() or not path.strip():
            raise SystemExit(f"--source {value!r} must be NAME=PATH")
        parsed[key.strip()] = Path(path.strip())
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argv: Arguments (default: ``sys.argv[1:]``).

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--annotation-config", type=Path, required=True)
    parser.add_argument("--panel-genes", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--store-large", type=Path, default=None)
    parser.add_argument("--scratch-dir", type=Path, default=None)
    parser.add_argument("--download-dir", type=Path, default=None)
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--n-processors", type=int, default=None)
    parser.add_argument("--max-gb", type=int, default=None)
    parser.add_argument(
        "--expect-panel-hash",
        default=None,
        help="Refuse unless the panel's hash starts with this.",
    )
    parser.add_argument(
        "--expect-draw1",
        default=None,
        help="Refuse unless draw 1's build_hash starts with this.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Build (or reuse) draw 2 and check it against the stored draw 1.

    Args:
        argv: Arguments (default: ``sys.argv[1:]``).

    Returns:
        The exit status (0 when draw 2 exists and matches draw 1).
    """
    from merxen.annotation import reference as ref
    from merxen.annotation import resolvability as res
    from merxen.annotation.config import AnnotationConfig
    from merxen.annotation.panel import load_annotation_panel
    from merxen.annotation.store import BUNDLE_MANIFEST_NAME, ReferenceStore
    from merxen.cli.run_annotation import _reference_spec

    args = parse_args(argv)
    config = AnnotationConfig.model_validate_json(
        args.annotation_config.read_text(encoding="utf-8")
    )
    if config.species != "mouse":
        raise SystemExit(f"{args.annotation_config} is not a mouse config")
    if int(config.resolvability.gate_p_wmb_draw) != ref.WMB_DRAW_1:
        raise SystemExit("give the production config (gate_p_wmb_draw 1)")
    panel = load_annotation_panel(args.panel_genes)
    if args.expect_panel_hash and not panel.panel_hash.startswith(
        args.expect_panel_hash
    ):
        raise SystemExit(
            f"panel hash {panel.panel_hash[:16]} is not {args.expect_panel_hash}"
        )
    if args.n_processors is not None or args.max_gb is not None:
        ref.set_prep_resources(n_processors=args.n_processors, max_gb=args.max_gb)
    # The spec annotation-reference-prep builds wmb_panel from.
    primary = _reference_spec(config, "wmb_panel", _key_value_paths(args.source))
    primary = ref.prepare_reference_spec(
        primary,
        ref.SourceOptions(
            download_dir=args.download_dir or args.store / ".downloads",
            auto_download=False,
            resolvability=config.resolvability.enabled,
        ),
    )
    if args.scratch_dir is not None:
        args.scratch_dir.mkdir(parents=True, exist_ok=True)
    store = ReferenceStore(
        args.store,
        large_root=args.store_large,
        large_panel_genes=config.panel.large_panel_genes,
        scratch_root=args.scratch_dir,
    )
    spec = ref.wmb_test_set_spec(primary, config)
    first_request = store.prepare_request(
        spec, panel, builder=ref.builder_for(spec, config), config=config
    )
    first_dir = store.bundle_dir(spec.reference_id, first_request.build_hash, panel)
    first_manifest = (
        json.loads((first_dir / BUNDLE_MANIFEST_NAME).read_text(encoding="utf-8"))
        if (first_dir / BUNDLE_MANIFEST_NAME).is_file()
        else {}
    )
    if first_manifest.get("status") != "complete":
        raise SystemExit(
            f"draw 1 ({first_request.build_hash[:16]}) is not a complete bundle in "
            f"{store.root_for(panel)}: build wmb_panel there first, with these inputs"
        )
    if args.expect_draw1 and not first_request.build_hash.startswith(args.expect_draw1):
        raise SystemExit(
            f"draw 1's build_hash {first_request.build_hash[:16]} is not "
            f"{args.expect_draw1}: the inputs differ from the production PREP's"
        )
    draw2_config = config.model_copy(
        update={
            "resolvability": config.resolvability.model_copy(
                update={"gate_p_wmb_draw": ref.WMB_DRAW_2}
            )
        }
    )
    bundle = store.get_or_build(
        spec, panel, builder=ref.builder_for(spec, draw2_config), config=draw2_config
    )
    bundle.write(args.output)
    second_dir = Path(bundle.path)
    manifest = json.loads(
        (second_dir / BUNDLE_MANIFEST_NAME).read_text(encoding="utf-8")
    )
    test_set = (manifest.get("builder_output") or {}).get("test_set") or {}
    draw: dict[str, Any] = test_set.get("gate_p_draw") or {}
    first_cells = {str(cell) for cell in res.load_test_cells(first_dir).obs.index}
    second_cells = {str(cell) for cell in res.load_test_cells(second_dir).obs.index}
    digest = ref.test_cell_digest(first_cells)
    checks = {
        "draw1_digest_matches": (draw.get("first_draw") or {}).get("cell_digest")
        == digest,
        "disjoint_from_draw1": not first_cells & second_cells,
        "disjoint_from_training": bool(draw.get("disjoint_from_training")),
        "panel_hash_matches": manifest.get("panel", {}).get("panel_hash")
        == panel.panel_hash,
    }
    record = {
        "panel_hash": panel.panel_hash,
        "draw1": {"build_hash": first_request.build_hash, "path": str(first_dir)},
        "draw2": {
            "build_hash": bundle.build_hash,
            "path": str(second_dir),
            "reused": bool(bundle.reused),
        },
        "checks": checks,
        "gate_p_draw": draw,
        "timings_s": (manifest.get("builder_output") or {}).get("timings_s"),
    }
    if args.record is not None:
        args.record.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    short = draw.get("classes_short_of_target") or []
    print(
        f"build_wmb_draw2.py: draw 2 {bundle.build_hash[:16]} "
        f"({'reused' if bundle.reused else 'built'}; draw 1 "
        f"{first_request.build_hash[:16]}); {draw.get('n_cells')} cells; "
        f"{len(draw.get('classes_reaching_target') or [])} of {draw.get('n_classes')} "
        f"classes reach the target, short: {len(short)}; checks {checks}"
    )
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

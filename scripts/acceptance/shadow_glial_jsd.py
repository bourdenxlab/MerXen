#!/usr/bin/env python
"""WHB vs SEA-AD glial JSD difference (plan §12 M3 item 7, §5.1; OD-B2 decided).

For information only: OD-B2 (WHB frontal names every class; SEA-AD is the
second vote) is decided, and changing the naming reference would need a new
decision. For each human pair and segmentation, the MERSCOPE-vs-Xenium JSD
of the glial composition (astrocytes, oligodendrocytes, OPC, microglia,
renormalised) and of the seven broad classes is computed with WHB's soft
composition (§5.5) and with SEA-AD's (class bootstrap probability x the
subclass bootstrap probabilities plus runner-ups through the SEA-AD vocab,
E2's root-level definition; ``seaad_soft_broad_matrix``), and with both
argmax labellings. The difference WHB - SEA-AD gets a paired spatial
block-bootstrap CI: the two sections share one 500 µm tile grid in the
Xenium frame, and each replicate applies one draw of tile locations to both
sections and both references (200 replicates); the independent per-section
resampling is reported as a sensitivity. Writes ``glial_jsd.csv`` and
``glial_compositions.csv``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from merxen.annotation.mapmycells_engine import level_frame, read_tidy_parquet
from merxen.annotation.shadow import (
    GLIAL_CLASSES,
    GLIAL_COLUMNS,
    argmax_broad_names,
    composition_shares,
    one_hot_broad_matrix,
    paired_block_bootstrap_jsd_difference,
    seaad_soft_broad_matrix,
    soft_matrix_from_provisional,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shadow_baselines import (  # noqa: E402
    HELD_OUT_PAIRS,
    PAIRS,
    PLATFORMS,
    SectionTiles,
    load_sample,
)

logger = logging.getLogger("shadow_glial_jsd")

SCOPES = {"glia": GLIAL_COLUMNS, "broad7": None}


def main(argv: list[str] | None = None) -> int:
    """Compute the WHB and SEA-AD platform JSDs and their paired difference."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pairs", default=",".join(PAIRS))
    parser.add_argument("--segmentations", default="proseg_hybrid,reseg")
    parser.add_argument("--n-bootstrap", type=int, default=200)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    rows: list[dict[str, Any]] = []
    comp_rows: list[dict[str, Any]] = []
    for seg in [s for s in args.segmentations.split(",") if s]:
        for pair in [p for p in args.pairs.split(",") if p]:
            run_dir = args.runs_root / pair / seg
            if not (run_dir / "map_manifest.json").is_file():
                continue
            per_platform: dict[str, dict[tuple[str, str], np.ndarray]] = {}
            samples = {}
            for platform in PLATFORMS:
                sample = load_sample(
                    run_dir, args.results_root, pair, seg, platform, None
                )
                samples[platform] = sample
                tidy, _ = read_tidy_parquet(
                    run_dir
                    / platform.lower()
                    / f"{sample.sample_id}_mmc_seaad_mr_panel.parquet"
                )
                subclass = level_frame(tidy, "subclass").reindex(sample.labels.index)
                supertype = level_frame(tidy, "supertype").reindex(sample.labels.index)
                class_level = level_frame(tidy, "class").reindex(sample.labels.index)
                matrices = {
                    ("WHB", "soft"): soft_matrix_from_provisional(sample.labels),
                    ("SEA-AD", "soft"): seaad_soft_broad_matrix(
                        subclass, supertype, class_level=class_level
                    ),
                    ("WHB", "argmax"): one_hot_broad_matrix(
                        argmax_broad_names(sample.labels)
                    ),
                    ("SEA-AD", "argmax"): one_hot_broad_matrix(
                        sample.sea["broad"].to_numpy(object)
                    ),
                }
                per_platform[platform] = matrices
                for (reference, kind), matrix in matrices.items():
                    shares = composition_shares(matrix)
                    glial_total = sum(shares[cls] for cls in GLIAL_CLASSES)
                    comp_rows.append(
                        {
                            "pair": pair,
                            "segmentation": seg,
                            "platform": platform,
                            "reference": reference,
                            "kind": kind,
                            **{f"share_{k}": v for k, v in shares.items()},
                            **{
                                f"glial_share_{cls}": shares[cls] / glial_total
                                for cls in GLIAL_CLASSES
                            },
                        }
                    )
            merscope, xenium = samples["MERSCOPE"], samples["XENIUM"]
            grid = SectionTiles.of(
                merscope,
                xenium,
                np.ones(len(merscope.labels), bool),
                np.ones(len(xenium.labels), bool),
            )
            tiles: dict[tuple[str, str, str], np.ndarray] = {}
            for key in per_platform["MERSCOPE"]:
                tiles_m, tiles_x = grid.sums(
                    per_platform["MERSCOPE"][key], per_platform["XENIUM"][key]
                )
                tiles[(*key, "MERSCOPE")] = tiles_m
                tiles[(*key, "XENIUM")] = tiles_x
            for kind in ("soft", "argmax"):
                for scope, columns in SCOPES.items():
                    quads = (
                        tiles[("WHB", kind, "MERSCOPE")],
                        tiles[("WHB", kind, "XENIUM")],
                        tiles[("SEA-AD", kind, "MERSCOPE")],
                        tiles[("SEA-AD", kind, "XENIUM")],
                    )
                    independent = paired_block_bootstrap_jsd_difference(
                        *quads,
                        columns=columns,
                        n_reps=args.n_bootstrap,
                        resampling="independent",
                    )
                    result = (
                        paired_block_bootstrap_jsd_difference(
                            *quads, columns=columns, n_reps=args.n_bootstrap
                        )
                        if grid.joint
                        else independent
                    )
                    rows.append(
                        {
                            "pair": pair,
                            "segmentation": seg,
                            "held_out": pair in HELD_OUT_PAIRS
                            or seg != "proseg_hybrid",
                            "kind": kind,
                            "scope": scope,
                            "jsd_whb": result.first_jsd,
                            "jsd_whb_ci_low": result.first_ci[0],
                            "jsd_whb_ci_high": result.first_ci[1],
                            "jsd_seaad": result.second_jsd,
                            "jsd_seaad_ci_low": result.second_ci[0],
                            "jsd_seaad_ci_high": result.second_ci[1],
                            "difference_whb_minus_seaad": result.difference,
                            "difference_ci_low": result.ci_low,
                            "difference_ci_high": result.ci_high,
                            "share_replicates_whb_higher": result.share_positive,
                            "resampling": result.resampling,
                            "difference_ci_low_independent": independent.ci_low,
                            "difference_ci_high_independent": independent.ci_high,
                        }
                    )
            logger.info("%s %s done", pair, seg)
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out / "glial_jsd.csv", index=False)
    pd.DataFrame(comp_rows).to_csv(args.out / "glial_compositions.csv", index=False)
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

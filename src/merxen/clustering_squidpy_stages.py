"""Process-isolated entry points for the Squidpy clustering workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from merxen.analysis.clustering_squidpy import (
    compute_clustering_squidpy,
    finalize_clustering_squidpy,
    prepare_clustering_squidpy,
)
from merxen.config import ClusteringSquidpyConfig, load_config_from_json


def _load_config(path: Path) -> ClusteringSquidpyConfig:
    config = load_config_from_json(path, ClusteringSquidpyConfig)
    assert isinstance(config, ClusteringSquidpyConfig)
    return config


def main() -> None:
    """Run one isolated clustering stage."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "compute", "finalize"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config = _load_config(args.config)
    if args.stage == "prepare":
        prepare_clustering_squidpy(config, args.output_dir)
        return
    if args.input_dir is None:
        parser.error(f"--input-dir is required for {args.stage}")
    if args.stage == "compute":
        compute_clustering_squidpy(config, args.input_dir, args.output_dir)
        return

    config.output_dir = args.output_dir
    finalize_clustering_squidpy(config, args.input_dir)


if __name__ == "__main__":
    main()

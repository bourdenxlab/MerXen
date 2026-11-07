"""Process-isolated entry points for the Squidpy clustering workflow.

``compute`` also runs the map_first hierarchy (CLUSTERING_SQUIDPY_COMPUTE_CPU,
plan §3.5): ``--mode map_first --labels-dir <annotation_resolve_out>
--table-key-suffix <suffix>`` override the fields of the clustering config
that CLUSTERING_SQUIDPY_PREPARE wrote (its script is the legacy one), and
``--mender-unassigned-state-policy`` records the run's MENDER policy in each
clustered table. Without these options every stage runs as in legacy runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from merxen.analysis.clustering_squidpy import (
    MapFirstComputeOptions,
    compute_clustering_squidpy,
    finalize_clustering_squidpy,
    prepare_clustering_squidpy,
)
from merxen.config import ClusteringSquidpyConfig, load_config_from_json


def _load_config(path: Path) -> ClusteringSquidpyConfig:
    config = load_config_from_json(path, ClusteringSquidpyConfig)
    assert isinstance(config, ClusteringSquidpyConfig)
    return config


def apply_mode_overrides(
    config: ClusteringSquidpyConfig,
    *,
    mode: str | None,
    labels_dir: Path | None,
    table_key_suffix: str | None,
) -> ClusteringSquidpyConfig:
    """Return ``config`` with the command-line clustering-mode fields applied.

    Args:
        config: The clustering config read from ``--config``.
        mode: ``--mode`` (``None`` keeps the config's).
        labels_dir: ``--labels-dir`` (``None`` keeps the config's).
        table_key_suffix: ``--table-key-suffix`` (``None`` keeps the config's).

    Returns:
        The re-validated config (the model's mode checks run again).
    """
    updates: dict[str, Any] = {}
    if mode is not None:
        updates["mode"] = mode
    if labels_dir is not None:
        updates["labels_dir"] = labels_dir
    if table_key_suffix is not None:
        updates["table_key_suffix"] = table_key_suffix
    if not updates:
        return config
    fields = config.model_dump(exclude_unset=True)
    return ClusteringSquidpyConfig.model_validate({**fields, **updates})


def main() -> None:
    """Run one isolated clustering stage."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "compute", "finalize"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("legacy", "map_first"),
        help="compute: clustering mode (default: the config's, legacy)",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        help="compute, map_first: RESOLVE's annotation_resolve_out",
    )
    parser.add_argument(
        "--table-key-suffix",
        help="compute, map_first: clustered table-key suffix of the run",
    )
    parser.add_argument(
        "--mender-unassigned-state-policy",
        choices=("state", "exclude_from_features"),
        help="compute, map_first: MENDER policy recorded in each clustered table",
    )
    args = parser.parse_args()

    map_first_args = (
        args.mode,
        args.labels_dir,
        args.table_key_suffix,
        args.mender_unassigned_state_policy,
    )
    if args.stage != "compute" and any(value is not None for value in map_first_args):
        parser.error(
            "--mode, --labels-dir, --table-key-suffix and "
            "--mender-unassigned-state-policy apply to compute only"
        )
    config = apply_mode_overrides(
        _load_config(args.config),
        mode=args.mode,
        labels_dir=args.labels_dir,
        table_key_suffix=args.table_key_suffix,
    )
    if args.stage == "prepare":
        prepare_clustering_squidpy(config, args.output_dir)
        return
    if args.input_dir is None:
        parser.error(f"--input-dir is required for {args.stage}")
    if args.stage == "compute":
        if config.mode == "map_first" and config.labels_dir is None:
            parser.error("--mode map_first needs --labels-dir")
        compute_clustering_squidpy(
            config,
            args.input_dir,
            args.output_dir,
            map_first_options=MapFirstComputeOptions(
                mender_unassigned_state_policy=args.mender_unassigned_state_policy
            ),
        )
        return

    config.output_dir = args.output_dir
    finalize_clustering_squidpy(config, args.input_dir)


if __name__ == "__main__":
    main()

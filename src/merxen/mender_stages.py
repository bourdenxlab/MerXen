"""Process-isolated entry points for modern-environment MENDER stages."""

from __future__ import annotations

import argparse
from pathlib import Path

from merxen.analysis.mender import (
    finalize_mender,
    import_mender_spatialdata,
    prepare_mender,
)
from merxen.config import MenderConfig, load_config_from_json


def _load_config(path: Path) -> MenderConfig:
    config = load_config_from_json(path, MenderConfig)
    assert isinstance(config, MenderConfig)
    return config


def main() -> None:
    """Run one modern-environment MENDER stage."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "finalize", "import"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path)
    parser.add_argument("--computed-dir", type=Path)
    parser.add_argument("--finalized-dir", type=Path)
    parser.add_argument("--source-h5ad", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = _load_config(args.config)
    if args.source_h5ad is not None:
        config.source_h5ad = args.source_h5ad
    if args.stage == "prepare":
        prepare_mender(config, args.output_dir)
        return
    if args.stage == "finalize":
        if args.prepared_dir is None or args.computed_dir is None:
            parser.error("--prepared-dir and --computed-dir are required for finalize")
        finalize_mender(
            config,
            args.prepared_dir,
            args.computed_dir,
            args.output_dir,
        )
        return
    if args.finalized_dir is None:
        parser.error("--finalized-dir is required for import")
    import_mender_spatialdata(
        config,
        args.finalized_dir,
        args.output_dir / "spatialdata_import_manifest.json",
    )


if __name__ == "__main__":
    main()

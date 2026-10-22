"""Process-isolated command-line entry points for the GASTON stage graph."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from merxen.analysis.gaston import (
    import_gaston_annotations,
    postprocess_gaston,
    prepare_gaston_input,
    run_gaston_glmpca,
    run_gaston_training,
)
from merxen.config import GastonConfig, load_config_from_json


def _load_config(path: Path) -> GastonConfig:
    config = load_config_from_json(path, GastonConfig)
    assert isinstance(config, GastonConfig)
    return config


def main() -> None:
    """Run one isolated GASTON process stage."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=("prepare", "glmpca", "train", "postprocess", "import"),
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path)
    parser.add_argument("--glmpca-dir", type=Path)
    parser.add_argument("--seed-dir", type=Path, action="append", default=[])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--standalone-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = _load_config(args.config)

    if args.stage == "prepare":
        prepare_gaston_input(config, args.output_dir)
        return
    if args.stage == "glmpca":
        if args.bundle_dir is None:
            parser.error("--bundle-dir is required for glmpca")
        run_gaston_glmpca(config, args.bundle_dir, args.output_dir)
        return
    if args.stage == "train":
        if args.bundle_dir is None or args.glmpca_dir is None or args.seed is None:
            parser.error("--bundle-dir, --glmpca-dir and --seed are required for train")
        try:
            run_gaston_training(
                config,
                args.bundle_dir,
                args.glmpca_dir,
                args.seed,
                args.output_dir,
            )
        except Exception as exc:  # noqa: BLE001 - one failed restart is rankable
            traceback.print_exc()
            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir / "seed_manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "status": "failed",
                        "seed": int(args.seed),
                        "minimum_loss": None,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
        return
    if args.stage == "postprocess":
        if args.bundle_dir is None or args.glmpca_dir is None or not args.seed_dir:
            parser.error(
                "--bundle-dir, --glmpca-dir and one or more --seed-dir values "
                "are required for postprocess"
            )
        postprocess_gaston(
            config,
            args.bundle_dir,
            args.glmpca_dir,
            args.seed_dir,
            args.output_dir,
        )
        return
    if args.standalone_dir is None:
        parser.error("--standalone-dir is required for import")
    import_gaston_annotations(config, args.standalone_dir, args.output_dir)


if __name__ == "__main__":
    main()

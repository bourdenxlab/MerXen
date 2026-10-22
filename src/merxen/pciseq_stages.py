"""Modern-environment entry points for pciSeq preparation and finalization."""

from __future__ import annotations

import argparse
from pathlib import Path

from merxen.config import PciSeqConfig, load_config_from_json


def main() -> None:
    """Run one pciSeq stage that reads or writes MerXen SpatialData."""
    from merxen.analysis.pciseq import finalize_pciseq, prepare_pciseq

    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "finalize"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path)
    parser.add_argument("--computed-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = load_config_from_json(args.config, PciSeqConfig)
    assert isinstance(config, PciSeqConfig)
    if args.stage == "prepare":
        prepare_pciseq(config, args.output_dir)
        return
    if args.prepared_dir is None or args.computed_dir is None:
        parser.error("--prepared-dir and --computed-dir are required for finalize")
    finalize_pciseq(config, args.prepared_dir, args.computed_dir, args.output_dir)


if __name__ == "__main__":
    main()

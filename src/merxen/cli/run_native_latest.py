"""CLI command for NATIVE_LATEST (the latest store of a native row)."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import click

from merxen.config import NativeLatestConfig, load_config_from_json
from merxen.segmentation.native import build_native_latest


@click.command(name="native-latest")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Path to JSON config validated against NativeLatestConfig.",
)
def native_latest_command(config_path: Path) -> None:
    """Build the latest store of a native-segmentation row from its source store."""
    cfg = load_config_from_json(config_path, NativeLatestConfig)
    assert isinstance(cfg, NativeLatestConfig)
    report = build_native_latest(cfg)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = cfg.output_dir / "native_latest_report.json"
    report_path.write_text(json.dumps(asdict(report), indent=2, default=str) + "\n")
    click.echo(f"Native latest store ready: {report.latest_path}")

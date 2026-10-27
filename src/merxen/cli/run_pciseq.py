"""CLI for optional pciSeq cell typing and its isolated process stages."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import click

from merxen.config import PciSeqConfig, load_config_from_json


@click.command(name="pciseq")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Path to JSON configuration validated against PciSeqConfig.",
)
@click.option(
    "--compute-python",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
    help="Python interpreter in the isolated pciSeq runtime; defaults to this Python.",
)
def pciseq_command(config_path: Path, compute_python: Path | None) -> None:
    """Assign cell types from native RNA spots and a supplied reference."""
    from merxen.analysis.pciseq import finalize_pciseq, prepare_pciseq

    config = load_config_from_json(config_path, PciSeqConfig)
    assert isinstance(config, PciSeqConfig)
    prepared_dir = config.output_dir / "prepared"
    computed_dir = config.output_dir / "computed"
    prepare_pciseq(config, prepared_dir)
    compute_env = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[2])
    compute_env["PYTHONPATH"] = os.pathsep.join(
        value for value in (source_root, compute_env.get("PYTHONPATH")) if value
    )
    subprocess.run(
        [
            str(compute_python or sys.executable),
            "-m",
            "merxen.pciseq_compute",
            "--prepared-dir",
            str(prepared_dir),
            "--output-dir",
            str(computed_dir),
        ],
        check=True,
        env=compute_env,
    )
    results = finalize_pciseq(config, prepared_dir, computed_dir, config.output_dir)
    click.echo("pciSeq complete:")
    for key, path in results.items():
        click.echo(f"- {key}: {path}")


@click.command(name="publish-pciseq-viewer")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="The PciSeqConfig JSON for the completed sample and segmentation.",
)
@click.option(
    "--results-dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    required=True,
    help="Completed pciSeq exports containing assignments and manifests.",
)
def publish_pciseq_viewer_command(config_path: Path, results_dir: Path) -> None:
    """Publish completed assignments to SpatialData without running the model."""
    from merxen.analysis.pciseq_viewer import publish_pciseq_viewer

    config = load_config_from_json(config_path, PciSeqConfig)
    assert isinstance(config, PciSeqConfig)
    result = publish_pciseq_viewer(config, results_dir)
    click.echo(json.dumps(result, indent=2, default=str))

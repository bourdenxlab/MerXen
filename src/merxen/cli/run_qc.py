"""CLI command for per-dataset QC metrics."""

from __future__ import annotations

from pathlib import Path

import click

from merxen.config import QCConfig, load_config_from_json
from merxen.qc.metrics import compute_dataset_qc, save_dataset_qc


@click.command(name="qc")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Path to JSON config validated against QCConfig.",
)
def qc_command(config_path: Path) -> None:
    """Compute and save geometry/assignment QC metrics for one dataset."""
    cfg = load_config_from_json(config_path, QCConfig)
    assert isinstance(cfg, QCConfig)

    qc_result = compute_dataset_qc(
        cfg.latest_zarr_path,
        cfg.dataset_name,
        table_key=cfg.table_key,
        shape_key=cfg.shape_key,
    )
    paths = save_dataset_qc(qc_result, cfg.output_dir, cfg.dataset_name)

    click.echo("QC complete:")
    for key, value in paths.items():
        click.echo(f"- {key}: {value}")

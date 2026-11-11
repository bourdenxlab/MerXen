"""CLI command for SpatialData enrichment + per-shape assignment."""

from __future__ import annotations

from pathlib import Path

import click
import pandas as pd

from merxen.config import EnrichmentConfig, load_config_from_json
from merxen.enrichment.assignment import run_per_shape_assignment_for_dataset
from merxen.enrichment.enrich import enrich_single_latest


@click.command(name="enrich")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Path to JSON config validated against EnrichmentConfig.",
)
@click.option(
    "--force-rerun",
    is_flag=True,
    default=False,
    help="Rebuild enrichment tables/layers even when already present.",
)
def enrich_command(config_path: Path, force_rerun: bool) -> None:
    """Enrich latest zarr and generate per-shape assignment tables."""
    cfg = load_config_from_json(config_path, EnrichmentConfig)
    assert isinstance(cfg, EnrichmentConfig)

    enriched_path = enrich_single_latest(cfg, force_rerun=force_rerun)
    summaries = run_per_shape_assignment_for_dataset(
        dataset_name=cfg.dataset_name,
        latest_path=enriched_path,
        force_rerun=force_rerun,
        platform=cfg.platform,
        segmentation_source=cfg.segmentation_source,
    )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = (
        cfg.output_dir / f"{cfg.dataset_name.lower()}_per_shape_assignment_summary.csv"
    )
    if summaries:
        pd.DataFrame(summaries).to_csv(summary_csv, index=False)
        click.echo(f"Saved assignment summary: {summary_csv}")
    else:
        click.echo("No per-shape assignment updates were required.")

    click.echo(f"Enrichment complete: {enriched_path}")

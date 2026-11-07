"""``merxen annotation-report``: build the annotation QC report (plan §9, M7).

Reads the published RESOLVE / MAP / PANEL / COMPUTE_CPU / cortical-depth /
MENDER outputs of one pair × segmentation (or one mouse section) and writes
``report.html``, PNG + PDF figures with one CSV each, the item tables and
``acceptance_metrics.json`` into ``--out``. It never writes into a results
tree or an input directory unless ``--allow-results-output`` is given.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

import click

logger = logging.getLogger(__name__)


def _pairs(values: tuple[str, ...], option: str) -> dict[str, Path]:
    """Parse repeated ``SAMPLE_ID=PATH`` values."""
    parsed: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise click.BadParameter(
                f"expected SAMPLE_ID=PATH, got {value!r}", param_hint=option
            )
        key, path = value.split("=", 1)
        parsed[key.strip()] = Path(path.strip())
    return parsed


def _items(value: str | None) -> list[int] | None:
    if not value:
        return None
    try:
        return [int(item) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise click.BadParameter(
            f"--items must be comma-separated integers: {error}"
        ) from error


@click.command(name="annotation-report")
@click.option(
    "--pair",
    "pair_id",
    required=True,
    help="Pair id (mouse: the section id RESOLVE used).",
)
@click.option("--segmentation", required=True, help="Segmentation, e.g. proseg_hybrid.")
@click.option(
    "--species",
    type=click.Choice(["human", "mouse"]),
    default="human",
    show_default=True,
)
@click.option(
    "--results-root",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=None,
    help="Results tree to find the outputs in (<root>/<pair>/<seg>/annotation_*, ...).",
)
@click.option(
    "--resolve-dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=None,
    help="annotation_resolve_out (default: found in --results-root).",
)
@click.option(
    "--map-dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=None,
    help="annotation_map_out (SEA-AD runs, mouse region step).",
)
@click.option(
    "--panel-dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=None,
    help="annotation_panel_out (panel_report.json).",
)
@click.option(
    "--clustered-h5ad",
    "clustered_values",
    multiple=True,
    help="SAMPLE_ID=PATH of a clustered H5AD (repeatable; overrides the tree lookup).",
)
@click.option(
    "--cortical-depth",
    "depth_values",
    multiple=True,
    help="SAMPLE_ID=PATH of a *_cells_with_cortical_depth.parquet (repeatable).",
)
@click.option(
    "--cortical-depth-dir",
    "depth_dir_values",
    multiple=True,
    help=(
        "SAMPLE_ID=DIR of a compute_cortical_depth_out: the report reads "
        "DIR/<segmentation>/*_cells_with_cortical_depth.parquet when present "
        "(repeatable; --cortical-depth wins)."
    ),
)
@click.option(
    "--mender-manifest",
    "mender_values",
    multiple=True,
    help="SAMPLE_ID=PATH of a mender_manifest.json (repeatable).",
)
@click.option(
    "--alignment-dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=None,
    help="The pair's align_out (shared tissue mask).",
)
@click.option("--no-cortical-depth", is_flag=True, help="Build without cortical depth.")
@click.option(
    "--no-alignment", is_flag=True, help="Build without the shared tissue mask."
)
@click.option("--no-mender", is_flag=True, help="Do not read MENDER manifests.")
@click.option(
    "--heldout-csv",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
    help="Held-out-gene enrichment CSV (scripts/acceptance/heldout_genes.py).",
)
@click.option(
    "--store",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=None,
    help="Reference store, for bundles whose recorded path moved (read-only).",
)
@click.option(
    "--out",
    "out_dir",
    required=True,
    type=click.Path(path_type=Path, file_okay=False),
    help="A fresh output directory.",
)
@click.option(
    "--allow-results-output",
    is_flag=True,
    help="Allow --out inside the results tree or an input directory (publish).",
)
@click.option(
    "--overwrite", is_flag=True, help="Allow a non-empty --out (files are replaced)."
)
@click.option(
    "--n-bootstrap", type=click.IntRange(min=1), default=200, show_default=True
)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option(
    "--tile-um", type=click.FloatRange(min=1.0), default=500.0, show_default=True
)
@click.option(
    "--density-bin-um", type=click.FloatRange(min=1.0), default=200.0, show_default=True
)
@click.option("--no-figures", is_flag=True, help="Write CSVs and placeholders only.")
@click.option(
    "--no-expression", is_flag=True, help="Skip the items that read counts (4, 7)."
)
@click.option(
    "--items",
    "items_value",
    default=None,
    help="Comma-separated §9 items (default: all).",
)
@click.option(
    "--strict", is_flag=True, help="Fail on an item error instead of recording it."
)
def annotation_report_command(
    pair_id: str,
    segmentation: str,
    species: str,
    results_root: Path | None,
    resolve_dir: Path | None,
    map_dir: Path | None,
    panel_dir: Path | None,
    clustered_values: tuple[str, ...],
    depth_values: tuple[str, ...],
    depth_dir_values: tuple[str, ...],
    mender_values: tuple[str, ...],
    alignment_dir: Path | None,
    no_cortical_depth: bool,
    no_alignment: bool,
    no_mender: bool,
    heldout_csv: Path | None,
    store: Path | None,
    out_dir: Path,
    allow_results_output: bool,
    overwrite: bool,
    n_bootstrap: int,
    seed: int,
    tile_um: float,
    density_bin_um: float,
    no_figures: bool,
    no_expression: bool,
    items_value: str | None,
    strict: bool,
) -> None:
    """Build the annotation QC report of one pair × segmentation (plan §9)."""
    from merxen.annotation.report import ReportOutputError, build_annotation_report
    from merxen.annotation.report_inputs import (
        ReportInputError,
        ReportSources,
        depth_cells_file,
        discover_sources,
    )
    from merxen.annotation.report_model import ReportOptions

    try:
        if results_root is not None:
            sources = discover_sources(
                results_root,
                pair_id,
                segmentation,
                species=species,  # type: ignore[arg-type]
                resolve_dir=resolve_dir,
                map_dir=map_dir,
                panel_dir=panel_dir,
                use_cortical_depth=not no_cortical_depth,
                use_alignment=not no_alignment,
                use_mender=not no_mender,
                heldout_csv=heldout_csv,
                store_root=store,
            )
        elif resolve_dir is None:
            raise click.UsageError("give --results-root or --resolve-dir")
        else:
            sources = ReportSources(
                species=species,  # type: ignore[arg-type]
                pair_id=pair_id,
                segmentation=segmentation,
                resolve_dir=resolve_dir,
                map_dir=map_dir,
                panel_dir=panel_dir,
                heldout_csv=heldout_csv,
                store_root=store,
            )
        clustered = {
            **sources.clustered_h5ad,
            **_pairs(clustered_values, "--clustered-h5ad"),
        }
        depth_from_dirs: dict[str, Path] = {}
        for sample_id, directory in _pairs(
            depth_dir_values, "--cortical-depth-dir"
        ).items():
            found = depth_cells_file(directory, segmentation)
            if found is not None:
                depth_from_dirs[sample_id] = found
        depth = (
            {}
            if no_cortical_depth
            else {
                **sources.cortical_depth,
                **depth_from_dirs,
                **_pairs(depth_values, "--cortical-depth"),
            }
        )
        mender = (
            {}
            if no_mender
            else {**sources.mender, **_pairs(mender_values, "--mender-manifest")}
        )
        alignment = None if no_alignment else (alignment_dir or sources.alignment_dir)
        sources = replace(
            sources,
            clustered_h5ad=clustered,
            cortical_depth=depth,
            mender=mender,
            alignment_dir=alignment,
        )
        result = build_annotation_report(
            sources,
            out_dir,
            options=ReportOptions(
                n_bootstrap=n_bootstrap,
                seed=seed,
                tile_um=tile_um,
                density_bin_um=density_bin_um,
                read_expression=not no_expression,
            ),
            allow_results_output=allow_results_output,
            overwrite=overwrite,
            make_figures=not no_figures,
            strict=strict,
            items=_items(items_value),
        )
    except (ReportInputError, ReportOutputError) as error:
        raise click.ClickException(str(error)) from error
    statuses = ", ".join(f"{item.number}:{item.status}" for item in result.items)
    click.echo(
        f"annotation report: {result.html} ({statuses}; {result.wall_time_s:.1f} s)"
    )


__all__ = ["annotation_report_command"]

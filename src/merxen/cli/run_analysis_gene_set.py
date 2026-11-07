"""CLI command ``merxen analysis-gene-set`` (Atera plan AT-8 §1b).

Builds the analysis gene set of a panel (``merxen.analysis_gene_set``) and
writes ``analysis_gene_set.json``, ``analysis_gene_set.csv`` (the layout
``merxen annotation-panel --panel-genes-path`` reads) and
``analysis_gene_set_missing.csv``. The module is imported inside the command,
so ``merxen`` starts without loading the annotation code.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import click

from merxen.platforms import supported_platforms


@click.command(name="analysis-gene-set")
@click.option(
    "--source",
    "source_path",
    type=click.Path(path_type=Path, exists=True),
    required=True,
    help="SpatialData store or table group (its table var), or a panel file "
    "annotation-panel reads (.h5ad, gene table, gene_panel.json, codebook).",
)
@click.option(
    "--table-key",
    default=None,
    help="Table of a SpatialData store (default: table_original).",
)
@click.option("--species", type=click.Choice(["human", "mouse"]), required=True)
@click.option(
    "--platform",
    type=click.Choice(supported_platforms(), case_sensitive=False),
    default=None,
    help="The panel's platform (control name patterns; recorded).",
)
@click.option(
    "--recipe",
    default=None,
    help="all_panel_genes (default for every platform), "
    "intersect_public_panel:<key> or explicit:<path>.",
)
@click.option("--name", default=None, help="Gene-set name (default: from the recipe).")
@click.option("--version", "set_version", type=click.IntRange(min=1), default=1)
@click.option(
    "--public-panel-dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=None,
    help="annotation-panel-fetch output for intersect_public_panel "
    "(default: the packaged Prime 5K lists).",
)
@click.option(
    "--annotation-config",
    "annotation_config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
    help="annotation_config.json whose panel settings resolve gene IDs "
    "(default: species defaults).",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
)
def analysis_gene_set_command(
    source_path: Path,
    table_key: str | None,
    species: str,
    platform: str | None,
    recipe: str | None,
    name: str | None,
    set_version: int,
    public_panel_dir: Path | None,
    annotation_config_path: Path | None,
    output_dir: Path,
) -> None:
    """Build the analysis gene set of a panel (joined on Ensembl gene IDs)."""
    from merxen.analysis_gene_set import (
        GeneSetError,
        build_analysis_gene_set,
        default_recipe,
        read_source_panel,
        write_analysis_gene_set,
    )
    from merxen.annotation.config import AnnotationConfig
    from merxen.annotation.vocab import Species

    typed_species = cast("Species", species)
    config = None
    if annotation_config_path is not None:
        config = AnnotationConfig.model_validate_json(
            annotation_config_path.read_text(encoding="utf-8")
        )
        if config.species != species:
            raise click.BadParameter(
                f"{annotation_config_path} is a {config.species} annotation config",
                param_hint="--annotation-config",
            )
    platform_value = platform.upper() if platform else None
    try:
        source = read_source_panel(
            source_path,
            species=typed_species,
            platform=platform_value,
            table_key=table_key,
            config=config,
        )
        gene_set = build_analysis_gene_set(
            source,
            recipe or default_recipe(platform_value),
            species=typed_species,
            name=name,
            version=set_version,
            public_panel_dir=public_panel_dir,
        )
    except (GeneSetError, FileNotFoundError) as error:
        raise click.ClickException(f"{type(error).__name__}: {error}") from error
    paths = write_analysis_gene_set(gene_set, output_dir)
    missing = (
        ""
        if gene_set.n_requested is None
        else f"; {len(gene_set.missing_genes)} of {gene_set.n_requested} "
        "requested genes missing"
    )
    click.echo(
        f"analysis-gene-set {gene_set.name} ({gene_set.recipe}): "
        f"{gene_set.n_genes} of {gene_set.source_panel.n_genes} panel genes{missing}; "
        f"gene_set_hash {gene_set.gene_set_hash} -> {paths.record}"
    )

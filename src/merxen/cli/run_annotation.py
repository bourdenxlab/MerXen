"""CLI commands for reference-based annotation (plan §3.2, §11.3).

* ``merxen annotation-panel``: ``ANNOTATE_PANEL`` for one pair x
  segmentation (declared panels, set a / set c, ``required_bundles.json``).
* ``merxen annotation-reference-prep``: ``ANNOTATE_REFERENCE_PREP`` for one
  (reference, panel): ``ReferenceStore.get_or_build`` and ``bundle_ref.json``.
* ``merxen annotation-store list`` and ``prune --unreferenced-by … --dry-run``:
  manual store maintenance; nothing is ever deleted (OD-D4).

The annotation modules are imported inside the commands, so ``merxen``
starts without loading them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import click

if TYPE_CHECKING:
    from merxen.annotation.config import AnnotationConfig, AnnotationReferenceSpec
    from merxen.annotation.vocab import Species

_SPECIES = click.Choice(["human", "mouse"])


def _load_annotation_config(path: Path | None, species: str) -> AnnotationConfig:
    from merxen.annotation.config import AnnotationConfig

    if path is None:
        return AnnotationConfig(species=cast("Species", species))
    config = AnnotationConfig.model_validate_json(path.read_text(encoding="utf-8"))
    if config.species != species:
        raise click.BadParameter(
            f"{path} is a {config.species} annotation config, not {species}",
            param_hint="--annotation-config",
        )
    return config


def _key_value_paths(values: tuple[str, ...], option: str) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        key, separator, path = value.partition("=")
        if not separator or not key.strip() or not path.strip():
            raise click.BadParameter(f"{value!r} must be KEY=PATH", param_hint=option)
        parsed[key.strip()] = Path(path.strip())
    return parsed


def _read_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise click.BadParameter(f"{path} does not hold a JSON object")
    return payload


@click.command(name="annotation-panel")
@click.option(
    "--prepared-dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=None,
    help="CLUSTERING_SQUIDPY_PREPARE output (manifest.json + prepared H5ADs).",
)
@click.option(
    "--panel-genes-path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
    help="Declared gene list for prepare-only runs (annotation_panel_genes_path).",
)
@click.option("--species", type=_SPECIES, required=True)
@click.option(
    "--platforms",
    default=None,
    help="Comma-separated platforms to use (default: every prepared sample).",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
)
@click.option(
    "--annotation-config",
    "annotation_config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
    help="annotation_config.json (AnnotationConfig); default: species defaults.",
)
@click.option(
    "--clustering-config",
    "clustering_config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
    help="clustering_squidpy_config.json (pair id, platforms, min_counts).",
)
@click.option("--pair-id", default=None)
@click.option("--segmentation", default=None)
@click.option(
    "--panel-file",
    "panel_file_values",
    multiple=True,
    help="KEY=PATH declared-panel file per sample id or platform "
    "(Xenium gene_panel.json, MERSCOPE codebook or a gene table).",
)
@click.option(
    "--shared-tissue-mask",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
    help="align_out/shared_tissue_mask.npy for the set-c pseudobulk.",
)
@click.option(
    "--registration-summary",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
    help="align_out/registration_summary.json (mask coordinate frame).",
)
@click.option("--min-counts", type=click.IntRange(min=0), default=None)
@click.option(
    "--gene-id-fallback-csv",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
    help="Local gene table for symbol -> Ensembl fallback (M0e).",
)
@click.option(
    "--panel-mode",
    type=click.Choice(["auto", "intersection", "per_platform"]),
    default=None,
    help="Override annotation_panel_mode.",
)
def annotation_panel_command(
    prepared_dir: Path | None,
    panel_genes_path: Path | None,
    species: str,
    platforms: str | None,
    output_dir: Path,
    annotation_config_path: Path | None,
    clustering_config_path: Path | None,
    pair_id: str | None,
    segmentation: str | None,
    panel_file_values: tuple[str, ...],
    shared_tissue_mask: Path | None,
    registration_summary: Path | None,
    min_counts: int | None,
    gene_id_fallback_csv: Path | None,
    panel_mode: str | None,
) -> None:
    """Resolve the declared panels of a pair and list its required bundles."""
    from merxen.annotation.panel import (
        compute_panel,
        load_shared_tissue_mask,
        panel_from_gene_list,
    )

    if (prepared_dir is None) == (panel_genes_path is None):
        raise click.UsageError(
            "give exactly one of --prepared-dir or --panel-genes-path"
        )
    config = _load_annotation_config(annotation_config_path, species)
    panel_updates: dict[str, Any] = {}
    if gene_id_fallback_csv is not None:
        panel_updates["gene_id_fallback_csv"] = gene_id_fallback_csv
    if panel_mode is not None:
        panel_updates["panel_mode"] = panel_mode
    if panel_updates:
        config = config.model_copy(
            update={"panel": config.panel.model_copy(update=panel_updates)}
        )
    platform_list = (
        [item.strip().upper() for item in platforms.split(",") if item.strip()]
        if platforms
        else None
    )
    if panel_genes_path is not None:
        result = panel_from_gene_list(
            panel_genes_path,
            cast("Species", species),
            output_dir=output_dir,
            platform=platform_list[0]
            if platform_list and len(platform_list) == 1
            else None,
            config=config,
            pair_id=pair_id,
            segmentation=segmentation,
        )
    else:
        assert prepared_dir is not None
        if (shared_tissue_mask is None) != (registration_summary is None):
            raise click.UsageError(
                "--shared-tissue-mask and --registration-summary go together"
            )
        mask = (
            load_shared_tissue_mask(shared_tissue_mask, registration_summary)
            if shared_tissue_mask is not None and registration_summary is not None
            else None
        )
        result = compute_panel(
            prepared_dir,
            cast("Species", species),
            platform_list,
            output_dir=output_dir,
            config=config,
            pair_id=pair_id,
            segmentation=segmentation,
            clustering_config=_read_json(clustering_config_path),
            panel_files=_key_value_paths(panel_file_values, "--panel-file"),
            shared_mask=mask,
            min_counts=min_counts,
        )
    click.echo(
        f"annotation-panel {result.required.status}: panel mode "
        f"{result.report['panel_mode']}, {len(result.panels)} panel(s), "
        f"{result.required.n_required} required bundle(s)"
    )
    for name, item in sorted(result.panels.items()):
        click.echo(
            f"- {name}: {item.panel.n_genes} genes, panel_hash "
            f"{item.panel.panel_hash[:16]} -> {item.file_name}"
        )


def _reference_spec(
    config: AnnotationConfig, reference_id: str, sources: dict[str, Path]
) -> AnnotationReferenceSpec:
    from merxen.annotation.config import KNOWN_REFERENCES, AnnotationReferenceSpec

    spec = next(
        (item for item in config.references if item.reference_id == reference_id),
        None,
    )
    if spec is None:
        known = KNOWN_REFERENCES.get(reference_id)
        if known is None:
            raise click.BadParameter(
                f"unknown reference {reference_id!r}: not in the annotation config "
                "or the known references",
                param_hint="--reference-id",
            )
        spec = AnnotationReferenceSpec(reference_id=reference_id, **known)
    if spec.species != config.species:
        raise click.BadParameter(
            f"reference {reference_id!r} is a {spec.species} reference",
            param_hint="--reference-id",
        )
    if sources:
        spec = spec.model_copy(update={"sources": {**spec.sources, **sources}})
    return spec


@click.command(name="annotation-reference-prep")
@click.option("--reference-id", required=True)
@click.option("--species", type=_SPECIES, required=True)
@click.option(
    "--panel-genes",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
    help="panel_genes*.json from annotation-panel (omit for panel-independent "
    "references).",
)
@click.option(
    "--store",
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
    help="Reference store root (annotation_reference_store).",
)
@click.option(
    "--store-large",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Store for panels above 1,000 genes (annotation_reference_store_large).",
)
@click.option(
    "--annotation-config",
    "annotation_config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
)
@click.option(
    "--source",
    "source_values",
    multiple=True,
    help="NAME=PATH reference source (added to the spec's sources).",
)
@click.option(
    "--scratch-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Parent of the build scratch directory (outside the store).",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path, dir_okay=False),
    required=True,
    help="bundle_ref.json to write.",
)
@click.option(
    "--download-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Cache of pinned reference downloads (default: <store>/.downloads).",
)
@click.option(
    "--auto-download/--no-auto-download",
    default=False,
    show_default=True,
    help="Download missing pinned reference files (annotation_auto_download).",
)
@click.option(
    "--download-seed",
    "download_seed_values",
    multiple=True,
    help="KEY=PATH local copy of a pinned file (used only if its sha256 matches).",
)
@click.option(
    "--n-processors",
    type=click.IntRange(min=1),
    default=None,
    help="Processes of the cell_type_mapper steps (default 8 or the CPU count).",
)
@click.option(
    "--max-gb",
    type=click.IntRange(min=1),
    default=None,
    help="Memory bound of the reference-marker step in GB (default 40).",
)
def annotation_reference_prep_command(
    reference_id: str,
    species: str,
    panel_genes: Path | None,
    store: Path,
    store_large: Path | None,
    annotation_config_path: Path | None,
    source_values: tuple[str, ...],
    scratch_dir: Path | None,
    output: Path,
    download_dir: Path | None,
    auto_download: bool,
    download_seed_values: tuple[str, ...],
    n_processors: int | None,
    max_gb: int | None,
) -> None:
    """Get or build one reference bundle and write its bundle_ref.json."""
    from merxen.annotation.panel import load_annotation_panel
    from merxen.annotation.store import ReferenceStore, resolve_builder

    config = _load_annotation_config(annotation_config_path, species)
    spec = _reference_spec(
        config, reference_id, _key_value_paths(source_values, "--source")
    )
    panel = load_annotation_panel(panel_genes) if panel_genes is not None else None
    if scratch_dir is not None:
        scratch_dir.mkdir(parents=True, exist_ok=True)
    reference_store = ReferenceStore(
        store,
        large_root=store_large,
        large_panel_genes=config.panel.large_panel_genes,
        scratch_root=scratch_dir,
    )
    builder = resolve_builder(spec, config)
    if builder.prepare_spec is not None:
        from merxen.annotation.reference import SourceOptions

        spec = builder.prepare_spec(
            spec,
            SourceOptions(
                download_dir=download_dir or store / ".downloads",
                auto_download=auto_download,
                seeds=_key_value_paths(download_seed_values, "--download-seed"),
            ),
        )
    if n_processors is not None or max_gb is not None:
        from merxen.annotation.reference import set_prep_resources

        set_prep_resources(n_processors=n_processors, max_gb=max_gb)
    bundle_ref = reference_store.get_or_build(
        spec, panel, builder=builder, config=config
    )
    bundle_ref.write(output)
    click.echo(
        f"annotation-reference-prep: {'reused' if bundle_ref.reused else 'built'} "
        f"{reference_id} {bundle_ref.build_hash[:16]} -> {bundle_ref.path}"
    )


@click.group(name="annotation-store")
def annotation_store_group() -> None:
    """Inspect the annotation reference store (never deletes anything)."""


def _store(store: Path, store_large: Path | None) -> Any:
    from merxen.annotation.store import ReferenceStore

    return ReferenceStore(store, large_root=store_large)


def _format_size(size_bytes: int) -> str:
    return f"{size_bytes / 1024**3:.2f} GB"


@annotation_store_group.command(name="list")
@click.option(
    "--store", type=click.Path(path_type=Path, file_okay=False), required=True
)
@click.option(
    "--store-large", type=click.Path(path_type=Path, file_okay=False), default=None
)
@click.option("--json", "as_json", is_flag=True, help="Print JSON.")
def annotation_store_list_command(
    store: Path, store_large: Path | None, as_json: bool
) -> None:
    """List bundles, temporary builds and failed builds."""
    entries = _store(store, store_large).list()
    if as_json:
        click.echo(json.dumps([entry.to_json() for entry in entries], indent=2))
        return
    if not entries:
        click.echo("annotation-store: no bundles")
        return
    for entry in entries:
        click.echo(
            "\t".join(
                [
                    entry.kind,
                    entry.reference_id or "-",
                    (entry.build_hash or "-")[:16],
                    (entry.panel_hash or "-")[:16],
                    str(
                        entry.n_panel_genes if entry.n_panel_genes is not None else "-"
                    ),
                    _format_size(entry.size_bytes),
                    entry.created_at or "-",
                    str(entry.path),
                ]
            )
        )


@annotation_store_group.command(name="prune")
@click.option(
    "--store", type=click.Path(path_type=Path, file_okay=False), required=True
)
@click.option(
    "--store-large", type=click.Path(path_type=Path, file_okay=False), default=None
)
@click.option(
    "--unreferenced-by",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    required=True,
    help="Results root whose manifests reference bundles.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Required: prune only lists candidates; deletion is manual (OD-D4).",
)
@click.option("--json", "as_json", is_flag=True, help="Print JSON.")
def annotation_store_prune_command(
    store: Path,
    store_large: Path | None,
    unreferenced_by: Path,
    dry_run: bool,
    as_json: bool,
) -> None:
    """List bundles no result references (dry run only; nothing is deleted)."""
    if not dry_run:
        raise click.UsageError(
            "annotation-store prune only lists candidates: pass --dry-run. "
            "Bundles are deleted by hand after review (OD-D4)."
        )
    report = _store(store, store_large).prune(
        unreferenced_by=unreferenced_by, dry_run=True
    )
    if as_json:
        click.echo(json.dumps(report.to_json(), indent=2))
        return
    click.echo(
        f"annotation-store prune (dry run): {len(report.referenced)} referenced, "
        f"{len(report.candidates)} candidate(s) "
        f"({_format_size(report.candidate_bytes)}), "
        f"{len(report.in_progress)} build(s) in progress; nothing deleted"
    )
    for entry in report.candidates:
        click.echo(
            f"candidate\t{entry.kind}\t{entry.reference_id or '-'}\t"
            f"{_format_size(entry.size_bytes)}\t{entry.path}"
        )

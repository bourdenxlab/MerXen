"""CLI commands for reference-based annotation (plan §3.2, §11.3).

* ``merxen annotation-panel``: ``ANNOTATE_PANEL`` for one pair x
  segmentation (declared panels, set a / set c, ``required_bundles.json``).
* ``merxen annotation-reference-prep``: ``ANNOTATE_REFERENCE_PREP`` for one
  (reference, panel): ``ReferenceStore.get_or_build`` and ``bundle_ref.json``.
* ``merxen annotation-store list`` and ``prune --unreferenced-by … --dry-run``:
  manual store maintenance; nothing is ever deleted (OD-D4).
* ``merxen annotate``: the MAP step (``annotation.pipeline.annotate_map``) on
  prepared H5ADs or, standalone, on published ``*_clustered.h5ad`` files
  (table cells, ``layers["counts"]``), writing only to ``--out``.

The annotation modules are imported inside the commands, so ``merxen``
starts without loading them. Store, builder and input errors end the command
with a one-line message (``click.ClickException``), not a traceback.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import click

if TYPE_CHECKING:
    from merxen.annotation.config import AnnotationConfig, AnnotationReferenceSpec
    from merxen.annotation.vocab import Species

_SPECIES = click.Choice(["human", "mouse"])


@contextmanager
def _clean_errors() -> Iterator[None]:
    """Turn expected store, builder and input errors into one-line CLI errors."""
    from merxen.annotation.store import StoreError

    try:
        yield
    except (StoreError, ValueError, FileNotFoundError) as error:
        raise click.ClickException(f"{type(error).__name__}: {error}") from error


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
@click.option(
    "--require-shared-tissue-mask",
    is_flag=True,
    help="Refuse a label-free set c without the shared tissue mask (aligned "
    "pairs in pipeline runs); a curated set c needs no mask.",
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
    require_shared_tissue_mask: bool,
) -> None:
    """Resolve the declared panels of a pair and list its required bundles."""
    with _clean_errors():
        _annotation_panel(
            prepared_dir=prepared_dir,
            panel_genes_path=panel_genes_path,
            species=species,
            platforms=platforms,
            output_dir=output_dir,
            annotation_config_path=annotation_config_path,
            clustering_config_path=clustering_config_path,
            pair_id=pair_id,
            segmentation=segmentation,
            panel_file_values=panel_file_values,
            shared_tissue_mask=shared_tissue_mask,
            registration_summary=registration_summary,
            min_counts=min_counts,
            gene_id_fallback_csv=gene_id_fallback_csv,
            panel_mode=panel_mode,
            require_shared_tissue_mask=require_shared_tissue_mask,
        )


def _annotation_panel(
    *,
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
    require_shared_tissue_mask: bool,
) -> None:
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
            require_shared_mask=require_shared_tissue_mask,
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
    with _clean_errors():
        _annotation_reference_prep(
            reference_id=reference_id,
            species=species,
            panel_genes=panel_genes,
            store=store,
            store_large=store_large,
            annotation_config_path=annotation_config_path,
            source_values=source_values,
            scratch_dir=scratch_dir,
            output=output,
            download_dir=download_dir,
            auto_download=auto_download,
            download_seed_values=download_seed_values,
            n_processors=n_processors,
            max_gb=max_gb,
        )


def _annotation_reference_prep(
    *,
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
    # bundle_ref.json holds only the bundle's identity, so a re-run writes the
    # same bytes; whether it was reused is logged here instead.
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
    "--store",
    type=click.Path(path_type=Path, file_okay=False, exists=True),
    required=True,
)
@click.option(
    "--store-large",
    type=click.Path(path_type=Path, file_okay=False, exists=True),
    default=None,
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
    "--store",
    type=click.Path(path_type=Path, file_okay=False, exists=True),
    required=True,
)
@click.option(
    "--store-large",
    type=click.Path(path_type=Path, file_okay=False, exists=True),
    default=None,
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


def _bundle_overrides(values: tuple[str, ...]) -> dict[str, Path]:
    """Parse ``--bundle KEY=DIR`` (``KEY`` = reference id or run id)."""
    return _key_value_paths(values, "--bundle")


@click.command(name="annotate")
@click.option(
    "--from-clustered-h5ad",
    "clustered_h5ads",
    multiple=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help="Published <sid>_clustered.h5ad (table cells, layers['counts']); "
    "repeat once per platform of the pair.",
)
@click.option(
    "--prepared-dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=None,
    help="CLUSTERING_SQUIDPY_PREPARE output (manifest.json + prepared H5ADs).",
)
@click.option("--species", type=_SPECIES, required=True)
@click.option("--pair-id", default=None, help="Default: from the results path.")
@click.option("--segmentation", default=None, help="Default: from the results path.")
@click.option(
    "--store",
    type=click.Path(path_type=Path, file_okay=False, exists=True),
    default=None,
    help="Reference store (default: the annotation config's reference_store).",
)
@click.option(
    "--store-large",
    type=click.Path(path_type=Path, file_okay=False, exists=True),
    default=None,
    help="Store for panels above 1,000 genes.",
)
@click.option(
    "--references",
    default=None,
    help="Comma-separated reference ids to map (default: every primary and "
    "secondary reference the panel requires).",
)
@click.option(
    "--bundle",
    "bundle_values",
    multiple=True,
    help="REFERENCE_ID=BUNDLE_DIR (or RUN_ID=BUNDLE_DIR, e.g. "
    "whb_frontal_supc_clus_setc=...) instead of the store lookup.",
)
@click.option(
    "--bundle-ref",
    "bundle_ref_paths",
    multiple=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help="bundle_ref.json from annotation-reference-prep (repeatable).",
)
@click.option(
    "--panel-dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=None,
    help="annotation-panel output (panel_genes*.json, required_bundles.json); "
    "default: computed from the inputs into <out>/panel.",
)
@click.option(
    "--out",
    "output_dir",
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
    help="Output directory (never inside a results tree).",
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
    help="clustering_squidpy_config.json of the prepared directory (pair id, "
    "sample platforms, min_counts).",
)
@click.option(
    "--min-counts",
    type=click.IntRange(min=0),
    default=None,
    help="Table-cell threshold (the clustering min_counts; default: the "
    "--clustering-config value, else 10).",
)
@click.option(
    "--n-processors",
    type=click.IntRange(min=1),
    default=None,
    help="MapMyCells processes (default: $MERXEN_ANNOTATION_MAP_N_PROCESSORS or 6).",
)
@click.option(
    "--work-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Scratch for queries and extended JSONs (default: <out>/.work).",
)
@click.option(
    "--keep-extended-json/--no-keep-extended-json",
    default=None,
    help="Keep the gzipped extended JSON (default: the config's).",
)
@click.option(
    "--reuse/--no-reuse",
    default=None,
    help="Reuse identical published runs (default: the config's reuse_published).",
)
@click.option(
    "--reuse-from",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=None,
    help="Directory of a published map_manifest.json (default: --out).",
)
@click.option(
    "--gene-id-fallback-csv",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
    help="Local gene table for symbol -> Ensembl fallback (M0e).",
)
@click.option(
    "--platforms",
    default=None,
    help="Comma-separated platforms to map (default: all inputs).",
)
@click.option(
    "--no-provisional",
    is_flag=True,
    help="Skip the provisional raw-threshold ct_* parquet.",
)
@click.option(
    "--allow-refused-panel",
    is_flag=True,
    help="For a refused panel, write map_manifest.json without runs and exit "
    "0 instead of failing (pipeline runs: RESOLVE writes statuses only).",
)
@click.option(
    "--require-bundle-refs",
    is_flag=True,
    help="Map only the bundles given with --bundle-ref / --bundle: fail "
    "instead of looking a missing one up in the store (pipeline runs).",
)
def annotate_command(
    clustered_h5ads: tuple[Path, ...],
    prepared_dir: Path | None,
    species: str,
    pair_id: str | None,
    segmentation: str | None,
    store: Path | None,
    store_large: Path | None,
    references: str | None,
    bundle_values: tuple[str, ...],
    bundle_ref_paths: tuple[Path, ...],
    panel_dir: Path | None,
    output_dir: Path,
    annotation_config_path: Path | None,
    clustering_config_path: Path | None,
    min_counts: int | None,
    n_processors: int | None,
    work_dir: Path | None,
    keep_extended_json: bool | None,
    reuse: bool | None,
    reuse_from: Path | None,
    gene_id_fallback_csv: Path | None,
    platforms: str | None,
    no_provisional: bool,
    allow_refused_panel: bool,
    require_bundle_refs: bool,
) -> None:
    """Map published or prepared samples with MapMyCells (MAP step, plan §3.3).

    Writes <out>/<platform>/<sid>_mmc_<run_id>.parquet (tidy per cell x
    level), <sid>_ct_provisional.parquet (raw-threshold labels, provisional
    until RESOLVE) and map_manifest.json; never writes into the inputs'
    results tree.
    """
    from merxen.annotation.mapmycells_engine import MmcEngineError
    from merxen.annotation.pipeline import MapError

    try:
        with _clean_errors():
            _annotate(
                clustered_h5ads=clustered_h5ads,
                prepared_dir=prepared_dir,
                species=species,
                pair_id=pair_id,
                segmentation=segmentation,
                store=store,
                store_large=store_large,
                references=references,
                bundle_values=bundle_values,
                bundle_ref_paths=bundle_ref_paths,
                panel_dir=panel_dir,
                output_dir=output_dir,
                annotation_config_path=annotation_config_path,
                clustering_config_path=clustering_config_path,
                min_counts=min_counts,
                n_processors=n_processors,
                work_dir=work_dir,
                keep_extended_json=keep_extended_json,
                reuse=reuse,
                reuse_from=reuse_from,
                gene_id_fallback_csv=gene_id_fallback_csv,
                platforms=platforms,
                write_provisional=not no_provisional,
                allow_refused_panel=allow_refused_panel,
                require_bundle_refs=require_bundle_refs,
            )
    except (MapError, MmcEngineError) as error:
        raise click.ClickException(f"{type(error).__name__}: {error}") from error


def _annotate(
    *,
    clustered_h5ads: tuple[Path, ...],
    prepared_dir: Path | None,
    species: str,
    pair_id: str | None,
    segmentation: str | None,
    store: Path | None,
    store_large: Path | None,
    references: str | None,
    bundle_values: tuple[str, ...],
    bundle_ref_paths: tuple[Path, ...],
    panel_dir: Path | None,
    output_dir: Path,
    annotation_config_path: Path | None,
    clustering_config_path: Path | None,
    min_counts: int | None,
    n_processors: int | None,
    work_dir: Path | None,
    keep_extended_json: bool | None,
    reuse: bool | None,
    reuse_from: Path | None,
    gene_id_fallback_csv: Path | None,
    platforms: str | None,
    write_provisional: bool,
    allow_refused_panel: bool,
    require_bundle_refs: bool,
) -> None:
    from merxen.annotation.mapmycells_engine import MmcBundle
    from merxen.annotation.panel import (
        DEFAULT_MIN_COUNTS,
        compute_panel,
        prepared_samples,
    )
    from merxen.annotation.pipeline import (
        MAPPED_ROLES,
        RUN_SUFFIXES,
        MapSample,
        annotate_map,
        check_output_outside_inputs,
        load_required,
        locate_bundle,
        map_bundles,
        published_layout,
        write_refused_manifest,
        write_view_manifest,
    )
    from merxen.annotation.store import BundleRef, ReferenceStore

    if bool(clustered_h5ads) == (prepared_dir is not None):
        raise click.UsageError(
            "give --from-clustered-h5ad (one per platform) or --prepared-dir"
        )
    clustering = _read_json(clustering_config_path) or {}
    if clustering and prepared_dir is None:
        raise click.UsageError("--clustering-config goes with --prepared-dir")
    configured_min_counts = clustering.get("min_counts")
    if configured_min_counts is not None:
        if min_counts is not None and min_counts != int(configured_min_counts):
            raise click.UsageError(
                f"--min-counts {min_counts} differs from the clustering config's "
                f"min_counts {configured_min_counts}: the table cells must be the "
                "clustering run's"
            )
        min_counts = int(configured_min_counts)
    if min_counts is None:
        min_counts = DEFAULT_MIN_COUNTS
    configured_pair = clustering.get("pair_id")
    if pair_id and configured_pair and str(configured_pair) != pair_id:
        raise click.UsageError(
            f"--pair-id {pair_id} differs from the clustering config's pair_id "
            f"{configured_pair}"
        )
    pair_id = pair_id or (str(configured_pair) if configured_pair else None)
    config = _load_annotation_config(annotation_config_path, species)
    updates: dict[str, Any] = {}
    if keep_extended_json is not None:
        updates["keep_extended_json"] = keep_extended_json
    if reuse is not None:
        updates["reuse_published"] = reuse
    if gene_id_fallback_csv is not None:
        updates["panel"] = config.panel.model_copy(
            update={"gene_id_fallback_csv": gene_id_fallback_csv}
        )
    if updates:
        config = config.model_copy(update=updates)
    config = config.coupled_to_clustering(min_counts)
    wanted_platforms = (
        {item.strip().upper() for item in platforms.split(",") if item.strip()}
        if platforms
        else None
    )

    samples: list[MapSample] = []
    if clustered_h5ads:
        for path in clustered_h5ads:
            layout = published_layout(path)
            if layout.platform is None:
                raise click.BadParameter(
                    f"cannot tell the platform of {path}",
                    param_hint="--from-clustered-h5ad",
                )
            pair_id = pair_id or layout.pair_id
            segmentation = segmentation or layout.segmentation
            samples.append(
                MapSample(
                    sample_id=layout.sample_id,
                    platform=layout.platform,
                    h5ad_path=path.resolve(),
                    source="clustered",
                )
            )
    else:
        assert prepared_dir is not None
        for prepared in prepared_samples(prepared_dir, clustering_config=clustering):
            samples.append(
                MapSample(
                    sample_id=prepared.sample_id,
                    platform=prepared.platform,
                    h5ad_path=prepared.h5ad_path.resolve(),
                    source="prepared",
                )
            )
    check_output_outside_inputs(output_dir, [sample.h5ad_path for sample in samples])
    output_dir.mkdir(parents=True, exist_ok=True)

    if panel_dir is None:
        panel_dir = output_dir / "panel"
        view = write_view_manifest(samples, panel_dir / "inputs")
        compute_panel(
            view,
            cast("Species", species),
            output_dir=panel_dir,
            config=config,
            pair_id=pair_id,
            segmentation=segmentation,
            min_counts=min_counts,
        )
    required = load_required(panel_dir, allow_refused=allow_refused_panel)
    if required.status == "refused":
        manifest = write_refused_manifest(
            required,
            config,
            output_dir=output_dir,
            pair_id=pair_id,
            segmentation=segmentation,
            n_processors=n_processors,
        )
        click.echo(
            f"annotate: panel of {manifest.pair_id} {manifest.segmentation} "
            f"refused, nothing mapped ({'; '.join(manifest.panel_reasons)}) "
            f"-> {output_dir}"
        )
        return
    reference_ids = (
        [item.strip() for item in references.split(",") if item.strip()]
        if references
        else None
    )

    store_root = store or config.reference_store
    reference_store = (
        ReferenceStore(
            store_root,
            large_root=store_large or config.reference_store_large,
            large_panel_genes=config.panel.large_panel_genes,
        )
        if store_root is not None
        else None
    )
    overrides = _bundle_overrides(bundle_values)
    # Refs are read here but opened only for the bundles MAP maps: a
    # pipeline task also stages the refs of unmapped roles (the mouse
    # region shares, M6), which hold no mapping precompute.
    from_refs: dict[tuple[str, str | None], BundleRef] = {}
    for ref_path in bundle_ref_paths:
        ref = BundleRef.model_validate_json(ref_path.read_text(encoding="utf-8"))
        from_refs[(ref.reference_id, ref.panel_hash)] = ref
    bundles: dict[tuple[str, str | None], MmcBundle] = {}
    for item in required.bundles:
        if item.role not in MAPPED_ROLES:
            continue
        if reference_ids is not None and item.reference_id not in reference_ids:
            continue
        key = (item.reference_id, item.panel_hash)
        run_key = item.reference_id + RUN_SUFFIXES.get(item.purpose, "")
        if run_key in overrides or (
            item.purpose == "annotation" and item.reference_id in overrides
        ):
            bundles[key] = MmcBundle.from_dir(
                overrides.get(run_key) or overrides[item.reference_id]
            )
        elif key in from_refs:
            bundles[key] = MmcBundle.from_bundle_ref(from_refs[key])
        elif require_bundle_refs:
            raise click.UsageError(
                f"no --bundle-ref for {item.reference_id} on panel "
                f"{(item.panel_hash or 'panel-independent')[:16]} "
                "(--require-bundle-refs)"
            )
        elif reference_store is not None:
            bundles[key] = locate_bundle(
                reference_store, item.reference_id, item.panel_hash
            )
        else:
            raise click.UsageError(
                f"no bundle for {item.reference_id}: give --store, --bundle or "
                "--bundle-ref"
            )
    runs = map_bundles(required, panel_dir, bundles, config, references=reference_ids)
    if not runs:
        raise click.UsageError("no reference to map (check --references)")
    if wanted_platforms is not None:
        samples = [sample for sample in samples if sample.platform in wanted_platforms]
    manifest = annotate_map(
        samples,
        runs,
        config,
        output_dir=output_dir,
        pair_id=pair_id,
        segmentation=segmentation,
        n_processors=n_processors,
        work_dir=work_dir,
        reuse_from=reuse_from or output_dir,
        write_provisional=write_provisional,
    )
    click.echo(
        f"annotate: {len(manifest.samples)} sample(s), "
        f"{sum(len(s.runs) for s in manifest.samples.values())} run(s) in "
        f"{manifest.wall_time_s:.0f} s -> {output_dir}"
    )
    for sample_id, record in manifest.samples.items():
        for run_id, run in record.runs.items():
            click.echo(
                f"- {sample_id} {run_id}: {run.n_cells} cells x {run.n_query_genes} "
                f"genes, {'reused' if run.reused else f'{run.wall_s:.0f} s'}"
            )

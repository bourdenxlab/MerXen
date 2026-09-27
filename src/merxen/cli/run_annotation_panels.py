"""CLI commands for new-panel design (plan §8.8; M3b).

* ``merxen annotation-panel-fetch``: fetch the pinned public vendor panel
  gene lists (``merxen.annotation.public_panels``) into a directory outside
  the repository and write their normalised gene tables and manifests.
* ``merxen annotation-panel-simulate``: the in-silico design aid
  (``merxen.annotation.simulate``): build the species' self-map references
  on a gene list with the production configuration and report the predicted
  levels per class and depth, trust, weak parents, the large-panel prefilter
  comparison, runtime, disk and peak memory. ``--gate-p`` is the M13 hook.

The annotation modules are imported inside the commands, so ``merxen``
starts without loading them.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any, cast

import click

from merxen.cli.run_annotation import (
    _SPECIES,
    _clean_errors,
    _key_value_paths,
    _load_annotation_config,
    _reference_spec,
)

logger = logging.getLogger(__name__)


@click.command(name="annotation-panel-fetch")
@click.option(
    "--panel",
    "panels",
    multiple=True,
    help="Public panel key (repeat); default: every pinned list.",
)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
    help="Directory for the lists (outside the repository).",
)
def annotation_panel_fetch_command(panels: tuple[str, ...], out_dir: Path) -> None:
    """Fetch pinned public panel gene lists (URL, size and sha256 checked)."""
    from merxen.annotation.public_panels import (
        PUBLIC_PANEL_LISTS,
        PublicPanelError,
        fetch_public_panel,
    )

    keys = list(panels) or sorted(PUBLIC_PANEL_LISTS)
    for key in keys:
        try:
            fetched = fetch_public_panel(key, out_dir)
        except PublicPanelError as error:
            raise click.ClickException(str(error)) from error
        click.echo(
            f"annotation-panel-fetch: {key}: {fetched.item.n_genes} genes, sha256 "
            f"{fetched.item.sha256[:16]} "
            f"({'downloaded' if fetched.downloaded else 'verified copy'}) -> "
            f"{fetched.gene_list_path}"
        )


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() or None


@click.command(name="annotation-panel-simulate")
@click.option(
    "--gene-list",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
    help="Gene list (vendor panel file, gene table or gene_panel.json).",
)
@click.option(
    "--public-panel",
    default=None,
    help="Pinned public panel key; fetched or verified into --panel-dir.",
)
@click.option(
    "--panel-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Directory of the public panel lists (with --public-panel).",
)
@click.option("--species", type=_SPECIES, default=None)
@click.option("--platform", default=None, help="Panel platform, e.g. XENIUM.")
@click.option("--name", default=None, help="Report name (default: the list's stem).")
@click.option(
    "--references",
    default=None,
    help="Comma-separated references (default: the species' self-map "
    "references: human whb_frontal_supc_clus,seaad_mr_panel; mouse wmb_panel).",
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
    "--param",
    "param_values",
    multiple=True,
    help="NAME=PATH reference source by its Nextflow param name, e.g. "
    "annotation_wmb_h5ad_dir=/path (repeat).",
)
@click.option(
    "--scratch-dir",
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
    help="Scratch outside the stores (build scratch, unfiltered markers).",
)
@click.option(
    "--download-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Cache of pinned reference downloads (default: <store>/.downloads).",
)
@click.option("--auto-download/--no-auto-download", default=False, show_default=True)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
    help="Report directory.",
)
@click.option("--n-processors", type=click.IntRange(min=1), default=None)
@click.option(
    "--max-gb",
    type=click.IntRange(min=1),
    default=None,
    help="Memory bound of the reference-marker steps in GB (default 40).",
)
@click.option(
    "--prefilter-compare",
    type=click.Choice(["auto", "always", "off"]),
    default="auto",
    show_default=True,
    help="Map the self-map cells with the unfiltered lookup too (auto: when "
    "the markers were prefiltered).",
)
@click.option(
    "--expected-depth",
    type=click.IntRange(min=1),
    default=None,
    help="Median panel counts per cell of the planned data.",
)
@click.option(
    "--depth-profile",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
    help="CSV of per-cell panel counts (column depth or total_counts).",
)
@click.option(
    "--gate-p",
    is_flag=True,
    default=False,
    help="Run the gate-P programme (NP1-NP9) after the simulation; M13 "
    "registers it (merxen.annotation.simulate.register_gate_p_hook), refused "
    "until then.",
)
def annotation_panel_simulate_command(**options: Any) -> None:
    """Simulate a candidate panel: predicted levels, trust, prefilter, resources."""
    from merxen.annotation.simulate import SimulationError

    with _clean_errors():
        try:
            _annotation_panel_simulate(**options)
        except SimulationError as error:
            raise click.ClickException(f"{type(error).__name__}: {error}") from error


def _annotation_panel_simulate(
    *,
    gene_list: Path | None,
    public_panel: str | None,
    panel_dir: Path | None,
    species: str | None,
    platform: str | None,
    name: str | None,
    references: str | None,
    store: Path,
    store_large: Path | None,
    annotation_config_path: Path | None,
    param_values: tuple[str, ...],
    scratch_dir: Path,
    download_dir: Path | None,
    auto_download: bool,
    out_dir: Path,
    n_processors: int | None,
    max_gb: int | None,
    prefilter_compare: str,
    expected_depth: int | None,
    depth_profile: Path | None,
    gate_p: bool,
) -> None:
    from merxen.annotation.reference import SourceOptions, set_prep_resources
    from merxen.annotation.simulate import (
        SELF_MAP_REFERENCES,
        ReferenceBuild,
        reference_sources,
        require_gate_p_hook,
        run_panel_simulation,
    )
    from merxen.annotation.store import ReferenceStore, resolve_builder

    if gate_p:
        require_gate_p_hook()
    public_record: dict[str, Any] | None = None
    if public_panel is not None:
        from merxen.annotation.public_panels import (
            PublicPanelError,
            fetch_public_panel,
        )

        if panel_dir is None:
            raise click.BadParameter(
                "--public-panel needs --panel-dir", param_hint="--panel-dir"
            )
        try:
            fetched = fetch_public_panel(public_panel, panel_dir)
        except PublicPanelError as error:
            raise click.ClickException(str(error)) from error
        gene_list = fetched.gene_list_path
        species = species or fetched.item.species
        platform = platform or fetched.item.platform
        name = name or fetched.item.key
        public_record = {
            "key": fetched.item.key,
            "url": fetched.item.url,
            "sha256": fetched.item.sha256,
            "manifest": str(fetched.manifest_path),
        }
    if gene_list is None:
        raise click.BadParameter(
            "give --gene-list or --public-panel", param_hint="--gene-list"
        )
    if species is None:
        raise click.BadParameter("--species is required", param_hint="--species")
    config = _load_annotation_config(annotation_config_path, species)
    if n_processors is not None or max_gb is not None:
        set_prep_resources(n_processors=n_processors, max_gb=max_gb)
    params = _key_value_paths(param_values, "--param")
    reference_ids = (
        [item.strip() for item in references.split(",") if item.strip()]
        if references
        else list(SELF_MAP_REFERENCES[species])
    )
    scratch_dir.mkdir(parents=True, exist_ok=True)
    build_scratch = scratch_dir / "build"
    build_scratch.mkdir(parents=True, exist_ok=True)
    reference_store = ReferenceStore(
        store,
        large_root=store_large or config.reference_store_large,
        large_panel_genes=config.panel.large_panel_genes,
        scratch_root=build_scratch,
    )
    options = SourceOptions(
        download_dir=download_dir or store / ".downloads",
        auto_download=auto_download,
        resolvability=config.resolvability.enabled,
    )

    def builds(_panel: Any) -> list[ReferenceBuild]:
        items: list[ReferenceBuild] = []
        for reference_id in reference_ids:
            spec = _reference_spec(
                config, reference_id, reference_sources(reference_id, params)
            )
            builder = resolve_builder(spec, config)
            if builder.prepare_spec is not None:
                spec = builder.prepare_spec(spec, options)
            items.append(ReferenceBuild(spec=spec, builder=builder))
        return items

    report = run_panel_simulation(
        gene_list=gene_list,
        species=species,
        name=name or gene_list.stem,
        config=config,
        store=reference_store,
        builds=builds,
        out_dir=out_dir,
        scratch_dir=scratch_dir / "simulate",
        platform=platform,
        prefilter_compare=cast("Any", prefilter_compare),
        expected_depth=expected_depth,
        depth_profile=depth_profile,
        gate_p=gate_p,
        provenance={
            "public_panel": public_record,
            "code_commit": _git_commit(),
            "references": reference_ids,
            "params": {key: str(value) for key, value in sorted(params.items())},
            "store": str(store),
            "store_large": None
            if (store_large or config.reference_store_large) is None
            else str(store_large or config.reference_store_large),
        },
    )
    click.echo(
        f"annotation-panel-simulate: {report['name']} ({report['status']}) -> {out_dir}"
    )

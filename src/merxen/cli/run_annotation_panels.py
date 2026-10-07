"""CLI commands for new-panel design (plan §8.8; M3b).

* ``merxen annotation-panel-fetch``: fetch the pinned public vendor panel
  gene lists (``merxen.annotation.public_panels``) into a directory outside
  the repository and write their normalised gene tables and manifests.
* ``merxen annotation-panel-simulate``: the in-silico design aid
  (``merxen.annotation.simulate``): build the species' self-map references
  on a gene list with the production configuration and report the predicted
  levels per class and depth, trust, weak parents, the large-panel prefilter
  comparison, runtime, disk and peak memory. ``--gate-p`` then runs the
  gate-P programme (M13; ``merxen.annotation.gate_p_run``) on the family.

The annotation modules are imported inside the commands, so ``merxen``
starts without loading them.
"""

from __future__ import annotations

import logging
import re
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


# The root of the source tree this module runs from (it lies in
# ``src/merxen/cli``): a git worktree, or a ``git archive`` export, which has
# no ``.git`` and names its commit in a ``COMMIT`` file at its root (as the
# M13 dry-run script's export step writes it).
_SOURCE_ROOT = Path(__file__).resolve().parents[3]
CODE_COMMIT_FILE = "COMMIT"
_FULL_COMMIT_ID = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")


def _git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() or None


def code_commit(source_root: Path | None = None) -> tuple[str | None, str | None]:
    """Return the commit of the code that runs, and where it was read.

    The ``COMMIT`` file at the tree's root is read first: an exported tree
    has no ``.git``, so ``git rev-parse`` there finds no repository, or the
    wrong one when the export lies inside another repository. A worktree
    has no such file and is asked with ``git rev-parse HEAD``.

    Args:
        source_root: The tree's root (default: the tree this module runs
            from).

    Returns:
        ``(commit, source)`` with source ``export`` (the ``COMMIT`` file) or
        ``git``; ``(None, None)`` when neither gives a commit.

    Raises:
        click.ClickException: If the ``COMMIT`` file does not hold one full
            commit id (the run would record a commit nobody can check).
    """
    root = _SOURCE_ROOT if source_root is None else source_root
    marker = root / CODE_COMMIT_FILE
    if marker.is_file():
        text = marker.read_text(encoding="utf-8").strip()
        if not _FULL_COMMIT_ID.fullmatch(text):
            raise click.ClickException(
                f"{marker} does not hold one full commit id (got {text[:80]!r}); "
                "export the code again"
            )
        return text, "export"
    commit = _git_commit(root)
    return (commit, "git") if commit else (None, None)


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
    help="CSV of per-cell total counts (column total_counts or depth), per class "
    "with a class column (the phase-1 depth_profile.csv format) or pooled; the "
    "report's headline (plan §8.3 v7.5).",
)
@click.option(
    "--depth-profile-asset",
    default=None,
    help="A registered simulation-input depth profile instead of a CSV, e.g. "
    "depth__xenium_prime__mouse_brain_ff (public 5K section) or "
    "depth__xenium_prime__human_lung_ffpe (lung FFPE scenario).",
)
@click.option(
    "--profile-mode/--no-profile-mode",
    default=True,
    show_default=True,
    help="With a depth profile: also map cells simulated at the profile's "
    "per-class depth for each emission member (primary reference).",
)
@click.option(
    "--real-composition",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
    help="CSV of real called class, subclass, n_cells to weigh profile-mode "
    "cells to (mouse).",
)
@click.option(
    "--profile-members",
    default=None,
    help="Comma-separated profile-mode members (default: the bundle's emission "
    "members, e.g. R1_contam_HO@0,R1_contam_HO@6,...,R3_measured_HO@3).",
)
@click.option(
    "--resolvability-version",
    "resolvability_version",
    type=click.Choice(["auto", "7"]),
    default="auto",
    show_default=True,
    help="7: also compute the resolvability version-7 decisions of a version-6 "
    "family (set a, ag7, VZG2, the pinned P5011 family) as a diagnostic under "
    "<out-dir>/<reference>/v7_diagnostic: never written to the store, never "
    "applied (plan §8.3 v7.1). Version-7 families build version 7 anyway.",
)
@click.option(
    "--v7-fresh-seeds",
    default=None,
    help="Comma-separated R1 seeds of a fresh version-7 ensemble B (e.g. 3,4,5) "
    "whose emitted-triple churn against ensemble A is reported (diagnostic).",
)
@click.option(
    "--v7-fresh-r3-seeds",
    "--v7-fresh-r3-seed",
    "v7_fresh_r3_seeds",
    default="1",
    show_default=True,
    help="Comma-separated R3 seeds of the fresh ensemble B (where a measured "
    "table exists).",
)
@click.option(
    "--v7-comparator",
    is_flag=True,
    default=False,
    help="Run the pre-registered comparator ensemble of the amended re-test of "
    "the M3c churn test as ensemble B (R1_contam_HO@20-25 + R3_measured_HO@20, "
    "@21; R1_contam_HO@20-27 without a measured table; pre-registration §22.4).",
)
@click.option(
    "--gate-p",
    is_flag=True,
    default=False,
    help="Run the gate-P programme (NP1-NP9; merxen.annotation.gate_p_run) after "
    "the simulation, writing <out-dir>/gate_p. Needs --species (M13 D4); gate P "
    "builds its leave-one-donor-out bundles in --store. The human path only: a "
    "mouse family waits for the second WMB test draw (M13 C20).",
)
@click.option(
    "--gate-p-other-region",
    type=click.Choice(["donor_own", "shared"]),
    default="donor_own",
    show_default=True,
    help="Other-region test cells of the leave-one-donor-out sets: donor_own "
    "(D2 (d): each donor draws its own) or shared (the fallback (c)).",
)
@click.option(
    "--gate-p-accept-small-pools",
    is_flag=True,
    default=False,
    help="Run D2 (d) although a donor's own pool cannot meet the per-class "
    "top-up rule (otherwise gate P stops after the pool-size report).",
)
@click.option(
    "--gate-p-version",
    type=click.Choice(["auto", "7"]),
    default="auto",
    show_default=True,
    help="7: score a version-6 family's version-7 ensemble (M13 D5 (c); never "
    "written to a store).",
)
@click.option(
    "--gate-p-dry-run",
    is_flag=True,
    default=False,
    help="Also score the seeded family's dry-run rule (M13 D28).",
)
@click.option(
    "--gate-p-time-reference-seconds",
    type=click.FloatRange(min=0.0, min_open=True),
    default=None,
    help="NP9's time reference of a version-6 family (plan §8.7 / §10); not "
    "used for a version-7 family, whose reference is M13 D10 (a)'s.",
)
@click.option(
    "--gate-p-time-reference-basis",
    default="",
    help="Where --gate-p-time-reference-seconds comes from (reported).",
)
@click.option(
    "--gate-p-dry-run-seconds",
    type=click.FloatRange(min=0.0, min_open=True),
    default=None,
    help="The set a version-7 dry run's measured_seconds (NP9's version-7 "
    "reference, M13 D10 (a)).",
)
@click.option(
    "--gate-p-dry-run-simulated-cells",
    type=click.IntRange(min=1),
    default=None,
    help="That dry run's simulated_cells.",
)
@click.option(
    "--gate-p-unresolved-reviewed",
    is_flag=True,
    default=False,
    help="The user reviewed NP1's unresolved list in the gate-P PR.",
)
@click.option(
    "--gate-p-accepted-parents",
    default=None,
    help="Comma-separated weak or collapsed parents the user accepted in the "
    "gate-P PR (NP2).",
)
@click.option(
    "--gate-p-skip-prep-identity",
    is_flag=True,
    default=False,
    help="Do not rebuild PREP for NP9's identity part (NP9 is then not evaluable).",
)
@click.option(
    "--gate-p-x1-factors",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
    help="The X1 factor table, reported beside NP6's offsets (M13 D7 (b)).",
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
    depth_profile_asset: str | None,
    profile_mode: bool,
    real_composition: Path | None,
    profile_members: str | None,
    gate_p: bool,
    resolvability_version: str = "auto",
    v7_fresh_seeds: str | None = None,
    v7_fresh_r3_seeds: str = "1",
    v7_comparator: bool = False,
    gate_p_other_region: str = "donor_own",
    gate_p_accept_small_pools: bool = False,
    gate_p_version: str = "auto",
    gate_p_dry_run: bool = False,
    gate_p_time_reference_seconds: float | None = None,
    gate_p_time_reference_basis: str = "",
    gate_p_dry_run_seconds: float | None = None,
    gate_p_dry_run_simulated_cells: int | None = None,
    gate_p_unresolved_reviewed: bool = False,
    gate_p_accepted_parents: str | None = None,
    gate_p_skip_prep_identity: bool = False,
    gate_p_x1_factors: Path | None = None,
) -> None:
    from merxen.annotation.reference import SourceOptions, set_prep_resources
    from merxen.annotation.simulate import (
        SELF_MAP_REFERENCES,
        ReferenceBuild,
        reference_sources,
        register_gate_p_hook,
        require_gate_p_hook,
        run_panel_simulation,
    )
    from merxen.annotation.store import ReferenceStore, resolve_builder

    # Before any compute: a malformed COMMIT file of an export is refused.
    commit, commit_source = code_commit()
    gate_p_programme: Any = None
    gate_p_checks: Any = None
    if gate_p:
        # M13 D4: the species is selected when a new panel check starts, so
        # gate P never takes it from a public panel's record.
        if species is None:
            raise click.BadParameter(
                "--gate-p needs --species: the species is selected when a new "
                "panel check starts (M13 D4; a human family is dry-run on set a "
                "only)",
                param_hint="--species",
            )
        from merxen.annotation.gate_p_run import (
            GatePOptions,
            gate_p_hook,
            gate_p_precheck,
        )

        gate_p_options = GatePOptions(
            species=cast("Any", species),
            other_region=gate_p_other_region,
            accept_small_pools=gate_p_accept_small_pools,
            resolvability_version=7 if gate_p_version == "7" else "auto",
            dry_run=gate_p_dry_run,
            time_reference_seconds=gate_p_time_reference_seconds,
            time_reference_basis=gate_p_time_reference_basis,
            dry_run_seconds=gate_p_dry_run_seconds,
            dry_run_simulated_cells=gate_p_dry_run_simulated_cells,
            unresolved_reviewed=gate_p_unresolved_reviewed,
            accepted_parents=tuple(
                item.strip()
                for item in (gate_p_accepted_parents or "").split(",")
                if item.strip()
            ),
            prep_identity=not gate_p_skip_prep_identity,
            x1_factors=gate_p_x1_factors,
        )
        gate_p_programme = gate_p_hook(gate_p_options)
        # run_panel_simulation runs the precheck before any compute (the
        # depth source, store, output, donors and seeds).
        gate_p_checks = gate_p_precheck(gate_p_options)
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

    if gate_p_programme is not None:
        register_gate_p_hook(gate_p_programme, precheck=gate_p_checks)
    try:
        if gate_p:
            require_gate_p_hook()
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
            depth_profile_asset=depth_profile_asset,
            profile_mode=profile_mode
            and (depth_profile is not None or depth_profile_asset is not None),
            real_composition=real_composition,
            profile_members=None
            if not profile_members
            else [item.strip() for item in profile_members.split(",") if item.strip()],
            gate_p=gate_p,
            v7_diagnostic=resolvability_version == "7",
            v7_fresh_seeds=None
            if not v7_fresh_seeds
            else [int(item) for item in v7_fresh_seeds.split(",") if item.strip()],
            v7_fresh_r3_seeds=[
                int(item) for item in v7_fresh_r3_seeds.split(",") if item.strip()
            ],
            v7_comparator=v7_comparator,
            provenance={
                "public_panel": public_record,
                "code_commit": commit,
                "code_commit_source": commit_source,
                "references": reference_ids,
                "params": {key: str(value) for key, value in sorted(params.items())},
                "store": str(store),
                "store_large": None
                if (store_large or config.reference_store_large) is None
                else str(store_large or config.reference_store_large),
            },
        )
    finally:
        if gate_p_programme is not None:
            register_gate_p_hook(None)
    click.echo(
        f"annotation-panel-simulate: {report['name']} ({report['status']}) -> {out_dir}"
    )
    gate_p_record = report.get("gate_p")
    if gate_p and isinstance(gate_p_record, dict):
        click.echo(
            f"annotation-panel-simulate: gate P {gate_p_record.get('status')}"
            + (
                f", passes {gate_p_record.get('passes')}, validated_max_level "
                f"{gate_p_record.get('validated_max_level')}"
                if gate_p_record.get("status") == "scored"
                else f": {gate_p_record.get('reason')}"
            )
            + f" -> {gate_p_record.get('run')}"
        )

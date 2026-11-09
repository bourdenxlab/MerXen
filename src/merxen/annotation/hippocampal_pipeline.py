"""Whole-panel hippocampal RCA stages, separate from frontal trust defaults.

The resource manifest is local configuration. It binds measured reference
artifacts, full-panel identity, frozen policies and downgrade-only diagnostics.
No published store is changed by these stages; FINALIZE owns table publication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import anndata as ad
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict
from scipy import sparse

from merxen.annotation import mapmycells_engine as mmc
from merxen.annotation import reference as ref
from merxen.annotation.hippocampal_profile import (
    HippocampalPolicy,
    PolicyQualification,
    ReferenceBinding,
    ordered_genes_sha256,
)
from merxen.annotation.panel import AnnotationPanel, load_annotation_panel


class ReferenceResources(BaseModel):
    """Local paths plus explicit biological identity of a measured reference."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    reference_id: str
    taxonomy_id: str
    regions: tuple[str, ...]
    precompute: Path
    lookup: Path
    tree: Path
    policy: Path
    qualification: Path


class HippocampalResources(BaseModel):
    """Explicit resources for the full-panel human hippocampal profile."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    profile: Literal["human_hippocampus_whole_panel_v1"]
    panel: Path
    seaad: ReferenceResources
    whb: ReferenceResources
    ctm_version: Literal["1.7.2"]


def file_sha256(path: Path) -> str:
    """Hash an actual artifact without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Atomically write finite JSON for deterministic stage handoffs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def load_resources(path: Path) -> HippocampalResources:
    """Read local configuration, resolving relative paths beside its manifest."""
    path = path.resolve()
    resources = HippocampalResources.model_validate_json(path.read_text())
    content = resources.model_dump()
    content["panel"] = (path.parent / resources.panel).resolve()
    for atlas in ("seaad", "whb"):
        for field in ("precompute", "lookup", "tree", "policy", "qualification"):
            content[atlas][field] = (path.parent / content[atlas][field]).resolve()
    return HippocampalResources.model_validate(content)


def bound_reference(
    resources: ReferenceResources, panel: AnnotationPanel
) -> tuple[
    mmc.MmcBundle,
    HippocampalPolicy,
    PolicyQualification,
    dict[tuple[str, str], str],
    set[str],
]:
    """Verify actual reference artifacts and qualification before any emission."""
    if panel.species != "human" or panel.n_genes != 18049:
        raise ValueError("Hippocampal reference requires the full 18,049-gene panel")
    policy = HippocampalPolicy.model_validate_json(resources.policy.read_text())
    qualification = PolicyQualification.model_validate_json(
        resources.qualification.read_text()
    )
    tree = ref.TaxonomyTreeView.from_tree_dict(json.loads(resources.tree.read_text()))
    binding = ReferenceBinding(
        reference_id=resources.reference_id,
        taxonomy_id=resources.taxonomy_id,
        regions=resources.regions,
        levels=tuple(tree.hierarchy),
        panel_hash=panel.panel_hash,
        query_genes_sha256=ordered_genes_sha256(panel.ensembl_ids),
        n_query_genes=18049,
        reference_sha256=file_sha256(resources.precompute),
        lookup_sha256=file_sha256(resources.lookup),
        tree_sha256=file_sha256(resources.tree),
    )
    binding.check_query(panel.panel_hash, panel.ensembl_ids)
    if binding != policy.binding:
        raise ValueError(
            "Hippocampal resources differ from the measured policy binding"
        )
    qualification.allowed_rules(policy)
    lookup = ref.read_lookup(resources.lookup)
    available = set(ref.precompute_genes(resources.precompute)) & set(panel.ensembl_ids)
    unsupported = set(
        ref.marker_unsupported_nodes(
            tree, ref.validate_lookup(lookup, tree, available)
        )["nodes"]
    )
    bundle = mmc.MmcBundle(
        reference_id=binding.reference_id,
        role="hippocampal",
        species="human",
        build_hash=binding.reference_sha256,
        path=resources.precompute.parent,
        panel_hash=panel.panel_hash,
        n_panel_genes=panel.n_genes,
        taxonomy_id=binding.taxonomy_id,
        levels=binding.levels,
        drop_level=None,
        mapping_precompute=resources.precompute,
        lookup=resources.lookup,
        lookup_sha256=ref.lookup_sha256(lookup),
        mapping_tree=resources.tree,
        vocab_snapshot=None,
        builder_version=None,
        manifest={"profile": "human_hippocampus_whole_panel_v1"},
    )
    parents = {
        (tree.hierarchy[i + 1], child): parent
        for i, level in enumerate(tree.hierarchy[:-1])
        for parent, children in tree.children[level].items()
        for child in children
    }
    return bundle, policy, qualification, parents, unsupported


def full_panel_counts(adata: ad.AnnData, panel: AnnotationPanel) -> sparse.csr_matrix:
    """Reorder complete raw gene columns; refuse subsets and duplicate IDs."""
    candidates = [adata.var_names.astype(str)]
    candidates.extend(
        adata.var[key].astype(str)
        for key in (
            "ensembl_id",
            "gene_id",
            "gene_ids",
            "ensembl_gene_id",
            "feature_id",
        )
        if key in adata.var
    )
    wanted = set(panel.ensembl_ids)
    if not any(
        len(candidate) == panel.n_genes
        and pd.Index(candidate).is_unique
        and set(candidate) == wanted
        for candidate in candidates
    ):
        # Gene releases can replace a native ID. Replay the panel's recorded
        # resolver, and require its full feature-resolution digest to match.
        from merxen.annotation.gene_ids import (
            GeneIdSources,
            load_gene_table,
            load_overrides,
            resolve_gene_ids,
        )

        record = getattr(panel, "declared_resolutions", {}).get("atera")
        table_path = record.gene_tables.get("human") if record is not None else None
        if record is not None and table_path and "gene_ids" in adata.var:
            table = load_gene_table(table_path, "human")
            if table is None:
                raise ValueError("Declared human gene-resolution table is unavailable")
            features = pd.DataFrame(
                {
                    "name": adata.var_names.astype(str),
                    "symbol": adata.var_names.astype(str),
                    "native_value": adata.var.gene_ids.astype(str).to_numpy(),
                }
            )
            resolution = resolve_gene_ids(
                features,
                "human",
                GeneIdSources(
                    species="human",
                    gene_tables={"human": table},
                    overrides=load_overrides("human"),
                ),
                native_id_column="gene_ids",
            )
            if (
                resolution.status != "ok"
                or resolution.table_sha256() != record.resolution_table_sha256
            ):
                raise ValueError(
                    "Feature resolution differs from the declared panel digest"
                )
            candidates.append(
                pd.Index([feature.gene_id for feature in resolution.features])
            )
    for candidate in candidates:
        ids = pd.Index(candidate)
        if len(ids) == panel.n_genes and ids.is_unique and set(ids) == wanted:
            positions = ids.get_indexer(panel.ensembl_ids)
            matrix = sparse.csr_matrix(adata.X[:, positions])
            if (
                not np.isfinite(matrix.data).all()
                or (matrix.data < 0).any()
                or (matrix.data != np.floor(matrix.data)).any()
            ):
                raise ValueError(
                    "Hippocampal input requires finite, nonnegative raw integer counts"
                )
            if not adata.obs_names.is_unique:
                raise ValueError("Hippocampal object IDs must be unique")
            return matrix
    raise ValueError("Hippocampal input must retain every full-panel gene exactly once")


def validate_resources(resources: HippocampalResources, output: Path) -> None:
    """Validate local resources and expose deterministic identities for caching."""
    panel = load_annotation_panel(resources.panel)
    identities = {}
    for atlas in ("seaad", "whb"):
        _, policy, qualification, _, _ = bound_reference(
            getattr(resources, atlas), panel
        )
        expected = (
            ("CCN20260630_LEVEL_0", "CCN20260630_LEVEL_1", "CCN20260630_LEVEL_2")
            if atlas == "seaad"
            else ("CCN202210140_SUPC", "CCN202210140_CLUS")
        )
        if policy.binding.levels != expected:
            raise ValueError(f"Unexpected {atlas} hippocampal taxonomy")
        identities[atlas] = {
            "binding": policy.binding.model_dump(mode="json"),
            "policy_sha256": policy.digest(),
            "qualification_sha256": qualification.digest(),
        }
    write_json(
        output,
        {
            "profile": resources.profile,
            "references": identities,
            "panel_sha256": file_sha256(resources.panel),
            "ctm_version": resources.ctm_version,
        },
    )


def map_prepared(resources: HippocampalResources, prepared: Path, output: Path) -> None:
    """Map every expressed object on both references, with all panel columns."""
    panel = load_annotation_panel(resources.panel)
    samples = json.loads((prepared / "manifest.json").read_text())["samples"]
    # Fixed measured engine settings, independent of frontal presets.
    params = mmc.MmcEngineParams(
        n_processors=8,
        bootstrap_iteration=100,
        bootstrap_factor=0.5,
        rng_seed=0,
        chunk_size=1000,
        max_gb=40,
    )
    for sample_id, relative in samples.items():
        if Path(sample_id).name != sample_id:
            raise ValueError("Sample ID must be a filename token")
        data = ad.read_h5ad(prepared / relative)
        matrix = full_panel_counts(data, panel)
        counts = np.asarray(matrix.sum(axis=1, dtype=np.float64)).ravel()
        objects = pd.DataFrame(
            {"cell_id": data.obs_names.astype(str), "n_counts": counts}
        )
        folder = output / sample_id
        folder.mkdir(parents=True, exist_ok=True)
        objects.to_parquet(folder / "objects.parquet", index=False)
        positive = counts > 0
        query = folder / "query.h5ad"
        mmc.write_query_h5ad(
            matrix[positive],
            objects.loc[positive, "cell_id"].tolist(),
            panel.ensembl_ids,
            query,
        )
        for atlas in ("seaad", "whb"):
            bundle, _, _, _, _ = bound_reference(getattr(resources, atlas), panel)
            destination = folder / f"{atlas}_assignments.parquet"
            if positive.any():
                mmc.run_mmc(
                    query,
                    bundle,
                    params,
                    output_parquet=destination,
                    work_dir=folder / f"{atlas}_scratch",
                    expected_ctm_version=resources.ctm_version,
                    log_dir=folder / f"{atlas}_logs",
                )
            else:
                pd.DataFrame(
                    columns=["cell_id", "level", "assignment", "name", "bp"]
                ).to_parquet(destination, index=False)
        write_json(
            folder / "query_identity.json",
            {
                "sha256": file_sha256(query),
                "n_genes": panel.n_genes,
                "n_objects": len(objects),
                "total_counts": int(counts.sum()),
                "engine": params.model_dump(mode="json"),
            },
        )
    write_json(output / "manifest.json", {"samples": samples})


def resolve_prepared(
    resources: HippocampalResources, mapped: Path, output: Path
) -> None:
    """Apply qualified rules with parent fallback; preserve raw and empty calls."""
    from merxen.annotation.hippocampal_profile import resolve_reference_calls

    panel = load_annotation_panel(resources.panel)
    samples = json.loads((mapped / "manifest.json").read_text())["samples"]
    for sample_id in samples:
        source = mapped / sample_id
        objects = pd.read_parquet(source / "objects.parquet")
        folder = output / sample_id
        folder.mkdir(parents=True, exist_ok=True)
        wide = objects.set_index("cell_id")
        provenance = {}
        for atlas, tokens in (
            ("seaad", ("class", "subclass", "supertype")),
            ("whb", ("supercluster", "cluster")),
        ):
            bundle, policy, qualification, parents, unsupported = bound_reference(
                getattr(resources, atlas), panel
            )
            predictions = pd.read_parquet(source / f"{atlas}_assignments.parquet")
            predictions["marker_supported"] = predictions.assignment.notna() & ~(
                predictions.level.astype(str) + "/" + predictions.assignment.astype(str)
            ).isin(unsupported)
            resolved = resolve_reference_calls(
                objects,
                predictions,
                policy=policy,
                binding=policy.binding,
                parent_of=parents,
                qualification=qualification,
            )
            resolved.to_parquet(folder / f"{atlas}_resolved.parquet", index=False)
            resolved.groupby(
                ["level", "resolution_status"], observed=True
            ).size().rename("n_objects").reset_index().to_csv(
                folder / f"{atlas}_resolution_counts.csv", index=False
            )
            for level, token in zip(bundle.levels, tokens, strict=True):
                part = (
                    resolved.loc[resolved.level.eq(level)]
                    .set_index("cell_id")
                    .reindex(wide.index)
                )
                for field in (
                    "assignment",
                    "name",
                    "raw_assignment",
                    "raw_name",
                    "bp",
                    "resolution_status",
                    "accepted",
                    "aggregate_probability",
                    "avg_correlation",
                    "directly_assigned",
                    "n_runners_up",
                    *(
                        f"runner_up_{rank}_{field}"
                        for rank in range(1, 6)
                        for field in (
                            "assignment",
                            "name",
                            "probability",
                            "correlation",
                        )
                    ),
                ):
                    if field in part:
                        wide[f"ct_{atlas}_{token}_{field}"] = part[field]
            provenance[atlas] = {
                "policy": policy.model_dump(mode="json"),
                "qualification": qualification.model_dump(mode="json"),
            }
        # Final labels are the deepest accepted SEA-AD node. WHB remains an
        # explicitly separate taxonomy, rather than a purported consensus.
        wide["ct_final_name"] = "Mixed/Unknown"
        wide["ct_final_level"] = "none"
        wide["ct_final_reference"] = "none"
        for token in ("class", "subclass", "supertype"):
            accepted = wide[f"ct_seaad_{token}_accepted"].fillna(False).astype(bool)
            wide.loc[accepted, "ct_final_name"] = wide.loc[
                accepted, f"ct_seaad_{token}_name"
            ].astype(str)
            wide.loc[accepted, "ct_final_level"] = token
            wide.loc[accepted, "ct_final_reference"] = "seaad_hippocampal"
        wide["ct_branch"] = wide.ct_seaad_class_name.astype("string").fillna(
            "Mixed/Unknown"
        )
        wide["ct_leaf"] = wide.ct_seaad_subclass_name.astype("string").fillna(
            "unresolved"
        )
        wide["ct_mender_state"] = wide.ct_final_name
        wide.reset_index().to_parquet(folder / "celltype_labels.parquet", index=False)
        write_json(
            folder / "annotation_manifest.json",
            {
                "profile": resources.profile,
                "evidence_basis": "existing_atlas_diagnostics_reused",
                "spatial_ground_truth_validated": False,
                "depth_basis": "all_18049_raw_panel_genes",
                "references": provenance,
            },
        )
    write_json(output / "manifest.json", {"samples": samples})


def compute_prepared(
    config_path: Path, prepared: Path, resolved: Path, output: Path
) -> None:
    """Build a hippocampal hierarchy and QC embedding for FINALIZE and MENDER."""
    from merxen.analysis.clustering_squidpy import save_clustered_adata
    from merxen.clustering.representation import (
        EmbeddingParams,
        normalize_table_cells,
        whole_section_qc,
    )
    from merxen.config import ClusteringSquidpyConfig

    config = ClusteringSquidpyConfig.model_validate_json(config_path.read_text())
    samples = json.loads((prepared / "manifest.json").read_text())["samples"]
    for sample in config.samples:
        if sample.platform != "ATERA" or sample.anatomical_region != "hippocampus":
            raise ValueError(
                "This workflow requires an unpaired human ATERA hippocampus sample"
            )
        data = ad.read_h5ad(prepared / samples[sample.sample_id])
        folder = resolved / sample.sample_id
        labels = pd.read_parquet(folder / "celltype_labels.parquet").set_index(
            "cell_id"
        )
        if not labels.index.is_unique or set(labels.index) != set(data.obs_names):
            raise ValueError(
                "Hippocampal labels must cover all prepared objects exactly once"
            )
        if sample.table_key is None or sample.shape_key is None:
            raise ValueError(
                "Hippocampal compute requires explicit source table and shape keys"
            )
        source_metadata = dict(data.uns.get("merxen_clustering_squidpy", {}))
        if source_metadata.get("table_key") not in (None, sample.table_key):
            raise ValueError(
                "Prepared table identity differs from the configured source"
            )
        source_metadata.update(table_key=sample.table_key, shape_key=sample.shape_key)
        data.uns["merxen_clustering_squidpy"] = source_metadata
        counts = np.asarray(data.X.sum(axis=1, dtype=np.float64)).ravel()
        if not np.array_equal(
            counts, labels.reindex(data.obs_names).n_counts.to_numpy()
        ):
            raise ValueError("Prepared counts differ from mapped counts")
        keep = counts >= config.min_counts
        data = data[keep].copy()
        aligned = labels.reindex(data.obs_names)
        params = EmbeddingParams.from_config(config)
        normalize_table_cells(data, params)
        for column in aligned:
            values = aligned[column]
            data.obs[column] = (
                pd.Categorical(values.astype("string").fillna("Mixed/Unknown"))
                if column != "n_counts"
                and (pd.api.types.is_string_dtype(values) or values.dtype == object)
                else values.array
            )
        for column, values in {
            "broad_class": aligned.ct_branch,
            "broad_atlas_label": aligned.ct_branch,
            "subcluster_label": aligned.ct_leaf,
            "subcluster_status": np.where(
                aligned.ct_leaf.eq("unresolved"), "unresolved", "mapped_subclass"
            ),
            "hierarchical_cluster": aligned.ct_branch + ":" + aligned.ct_leaf,
        }.items():
            data.obs[column] = pd.Categorical(values)
        # This QC partition has no say in the reference-derived hierarchy.
        whole_section_qc(
            data,
            resolution=config.qc_leiden_resolution,
            seed=config.random_seed,
            params=params,
        )
        data.uns["merxen_hierarchical_clustering"] = {
            "enabled": True,
            "mode": "map_first",
            "profile": "human_hippocampus_whole_panel_v1",
            "species": "human",
            "branch_level": "seaad_class",
            "leaf_level": "seaad_subclass",
            "leaf_source": "mapped",
            "table_key_suffix": "rcahippocampal",
            "mender_unassigned_state_policy": "exclude_from_features",
            "panel_trust": "provisional",
            "gate_level": "existing_atlas_diagnostics_reused",
        }
        data.uns["merxen_hippocampal_annotation_json"] = (
            folder / "annotation_manifest.json"
        ).read_text()
        save_clustered_adata(
            data, output / "atera" / f"{sample.sample_id}_clustered.h5ad"
        )


def report_resolved(
    resolved: Path,
    output: Path,
    *,
    mender_dirs: list[Path] | None = None,
    expect_mender: bool = False,
    clustered: Path | None = None,
) -> None:
    """Write the complete hippocampal outcome report, including abstentions."""
    import html

    samples = json.loads((resolved / "manifest.json").read_text())["samples"]
    mender_records = []
    for folder in mender_dirs or []:
        paths = list(folder.rglob("mender_manifest.json"))
        for path in paths:
            mender_records.append(json.loads(path.read_text()))
    if expect_mender and not mender_records:
        raise ValueError("MENDER was requested but its completion manifest is missing")
    sections = []
    if mender_records:
        sections.append(
            "<h2>MENDER completion</h2><pre>"
            + html.escape(json.dumps(mender_records, indent=2))
            + "</pre>"
        )
    for sample_id in samples:
        folder = resolved / sample_id
        labels = pd.read_parquet(folder / "celltype_labels.parquet")
        table = (
            labels.groupby(["ct_final_level", "ct_final_name"], observed=True)
            .size()
            .rename("n_objects")
            .reset_index()
        )
        from merxen.annotation.hippocampal_report import (
            annotation_figures,
            diagnostic_table,
            outcome_tables,
        )

        tables = outcome_tables(labels, output / sample_id / "tables")
        manifest = json.loads((folder / "annotation_manifest.json").read_text())
        diagnostics = diagnostic_table(manifest)
        diagnostics.to_csv(
            output / sample_id / "tables/policy_diagnostics.csv", index=False
        )
        sections.append(
            "<h2>Coverage by raw transcript depth</h2>"
            + tables["depth_coverage"].to_html(index=False)
        )
        sections.append(
            "<h2>SEA-AD / WHB calls</h2><p>Distinct taxonomies; this "
            "crosstab is not an accuracy measure.</p>"
            + tables["cross_atlas_calls"].to_html(index=False)
        )
        sections.append(
            "<details><summary>Every frozen class/depth diagnostic</summary>"
            + diagnostics.to_html(index=False)
            + "</details>"
        )
        if clustered is not None:
            paths = list(clustered.rglob(f"{sample_id}_clustered.h5ad"))
            if len(paths) != 1:
                raise ValueError(
                    "Report requires exactly one clustered H5AD per sample"
                )
            figures = annotation_figures(paths[0], output / sample_id / "figures")
            for figure in figures:
                sections.append(
                    '<img style="max-width:100%" '
                    'alt="Accepted SEA-AD subclasses with unresolved cells" '
                    f'src="{figure.relative_to(output)}">'
                )
        sections.append(
            f"<h2>{html.escape(sample_id)}</h2><p>{len(labels):,} objects; "
            f"{int(labels.n_counts.eq(0).sum()):,} without expression.</p>"
            + table.to_html(index=False)
        )
        for atlas in ("seaad", "whb"):
            counts = pd.read_csv(folder / f"{atlas}_resolution_counts.csv")
            sections.append(
                f"<h3>{atlas.upper()} outcomes at every level</h3>"
                + counts.to_html(index=False)
            )
    output.mkdir(parents=True, exist_ok=True)
    page = (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        "<title>Hippocampal RCA report</title><body>"
        "<h1>Whole-panel hippocampal RCA</h1>"
        "<p>SEA-AD primary, Allen WHB comparison; all 18,049 raw panel genes. "
        "Confidence rules use reused atlas diagnostics. These are provisional "
        "spatial annotations, without spatial ground-truth validation. "
        "Unsupported fine calls remain unresolved; final labels fall back to "
        "accepted parents. Raw calls, support, per-class/depth diagnostic "
        "evidence, artifact hashes and all zero-expression objects are "
        "retained in the accompanying annotation outputs.</p>"
        + "".join(sections)
        + "</body></html>"
    )
    (output / "annotation_report.html").write_text(page)
    write_json(
        output / "report_manifest.json",
        {
            "status": "complete",
            "samples": list(samples),
            "evidence_basis": "existing_atlas_diagnostics_reused",
            "spatial_ground_truth_validated": False,
        },
    )


def main() -> None:
    """Run one deterministic workflow stage from explicit local inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("validate", "map", "resolve", "compute", "report")
    )
    parser.add_argument("--resources", type=Path)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--mapped", type=Path)
    parser.add_argument("--resolved", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--clustered", type=Path)
    parser.add_argument("--mender", type=Path, action="append", default=[])
    parser.add_argument("--expect-mender", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    requirements = {
        "validate": ("resources",),
        "map": ("resources", "prepared"),
        "resolve": ("resources", "mapped"),
        "compute": ("config", "prepared", "resolved"),
        "report": ("resolved",),
    }
    for key in requirements[args.stage]:
        if getattr(args, key) is None:
            parser.error(f"{args.stage} requires --{key}")
    resources = load_resources(args.resources) if args.resources is not None else None
    if args.stage == "validate":
        assert resources is not None
        validate_resources(resources, args.output)
        write_json(
            args.output.parent / "resources.json", resources.model_dump(mode="json")
        )
        identity = json.loads(args.output.read_text())
        write_json(
            args.output.parent / "mapping_identity.json",
            {
                "panel_sha256": identity["panel_sha256"],
                "ctm_version": identity["ctm_version"],
                "references": {
                    atlas: data["binding"]
                    for atlas, data in identity["references"].items()
                },
            },
        )
    elif args.stage == "map":
        assert resources is not None
        map_prepared(resources, args.prepared, args.output)
    elif args.stage == "resolve":
        assert resources is not None
        resolve_prepared(resources, args.mapped, args.output)
    elif args.stage == "compute":
        compute_prepared(args.config, args.prepared, args.resolved, args.output)
    else:
        report_resolved(
            args.resolved,
            args.output,
            mender_dirs=args.mender,
            expect_mender=args.expect_mender,
            clustered=args.clustered,
        )


if __name__ == "__main__":
    main()

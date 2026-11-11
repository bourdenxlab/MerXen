"""Inspectable outcome tables and scientific figures for hippocampal RCA."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd


def diagnostic_table(manifest: dict[str, Any]) -> pd.DataFrame:
    """Expose every class/depth diagnostic, including withheld rules."""
    from merxen.annotation.hippocampal_profile import (
        HippocampalPolicy,
        PolicyQualification,
        wilson_lower,
    )

    rows = []
    for atlas, records in manifest["references"].items():
        policy = HippocampalPolicy.model_validate(records["policy"])
        qualification = PolicyQualification.model_validate(records["qualification"])
        allowed = qualification.allowed_rules(policy)
        for check in qualification.checks:
            n = check.n_accepted
            rows.append(
                {
                    "atlas": atlas,
                    **check.model_dump(),
                    "precision": check.n_correct / n if n else None,
                    "wilson_lower": wilson_lower(
                        check.n_correct, n, policy.objective.wilson_z
                    )
                    if n
                    else None,
                    "qualified_in_all_reused_checks": (
                        check.level,
                        check.assignment,
                        check.min_counts,
                    )
                    in allowed,
                }
            )
    return pd.DataFrame(rows)


def outcome_tables(labels: pd.DataFrame, output: Path) -> dict[str, pd.DataFrame]:
    """Write full-denominator depth coverage and an explicit cross-atlas crosstab."""
    data = labels.copy()
    edges = [0, 1, 10, 25, 50, 100, 250, 500, 1000, 5000, np.inf]
    data["depth_interval"] = pd.cut(data.n_counts, edges, right=False).astype(str)
    depth = (
        data.groupby(["depth_interval", "ct_final_level"], observed=True)
        .size()
        .rename("n_objects")
        .reset_index()
    )
    cross_atlas = (
        data.groupby(
            ["ct_seaad_class_name", "ct_whb_supercluster_name"],
            observed=True,
            dropna=False,
        )
        .size()
        .rename("n_objects")
        .reset_index()
    )
    outcomes = (
        data.groupby(["ct_final_level", "ct_final_name"], observed=True)
        .size()
        .rename("n_objects")
        .reset_index()
    )
    tables = {
        "depth_coverage": depth,
        "cross_atlas_calls": cross_atlas,
        "final_labels": outcomes,
    }
    output.mkdir(parents=True, exist_ok=True)
    for token, table in tables.items():
        table.to_csv(output / f"{token}.csv", index=False)
    return tables


def annotation_figures(clustered_path: Path, output: Path) -> list[Path]:
    """Plot accepted subclass calls on QC UMAP and native spatial coordinates."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    data = ad.read_h5ad(clustered_path, backed="r")
    try:
        values = data.obs.ct_seaad_subclass_name.astype(str).to_numpy()
        accepted = data.obs.ct_seaad_subclass_accepted.astype(bool).to_numpy()
        names = sorted(set(values[accepted]))
        palette = {}
        for name in names:
            hue = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
            rgb = colors.hsv_to_rgb((hue, 0.65, 0.72))
            palette[name] = colors.to_hex((float(rgb[0]), float(rgb[1]), float(rgb[2])))
        output.mkdir(parents=True, exist_ok=True)
        paths = []
        for key, title, axes in (
            ("X_umap", "SEA-AD subclasses on QC UMAP", ("UMAP 1", "UMAP 2")),
            (
                "spatial",
                "SEA-AD subclasses in native coordinates",
                ("Native x (µm)", "Native y (µm)"),
            ),
        ):
            if key not in data.obsm:
                raise ValueError(f"Annotation report requires {key} coordinates")
            xy = np.asarray(data.obsm[key])[:, :2]
            if not np.isfinite(xy).all():
                raise ValueError("Annotation figure contains nonfinite coordinates")
            fig, ax = plt.subplots(figsize=(12, 8), layout="constrained")
            point_size = max(0.7, min(12.0, 2000.0 / max(data.n_obs, 1)))
            ax.scatter(
                xy[~accepted, 0],
                xy[~accepted, 1],
                s=point_size * 0.8,
                c="#c8c8c8",
                rasterized=True,
            )
            for name in names:
                mask = accepted & (values == name)
                ax.scatter(
                    xy[mask, 0],
                    xy[mask, 1],
                    s=point_size,
                    c=[palette[name]],
                    rasterized=True,
                )
            handles = [
                Line2D(
                    [],
                    [],
                    color="#c8c8c8",
                    marker="o",
                    linestyle="",
                    label=f"Subclass unresolved ({int((~accepted).sum()):,})",
                )
            ]
            handles.extend(
                Line2D(
                    [],
                    [],
                    color=palette[name],
                    marker="o",
                    linestyle="",
                    label=f"{name} ({int((accepted & (values == name)).sum()):,})",
                )
                for name in names
            )
            ax.legend(
                handles=handles,
                loc="upper left",
                bbox_to_anchor=(1.01, 1),
                fontsize=8,
                markerscale=0.7,
            )
            ax.set(
                title=f"{title}\n{data.n_obs:,} table cells; reused atlas diagnostics",
                xlabel=axes[0],
                ylabel=axes[1],
            )
            if key == "spatial":
                ax.set_aspect("equal")
                ax.invert_yaxis()
            path = output / f"subclass_{key}.png"
            fig.savefig(path, dpi=180)
            plt.close(fig)
            paths.append(path)
        (output / "figure_sources.json").write_text(
            json.dumps(
                {
                    "source_h5ad": str(clustered_path),
                    "n_table_cells": data.n_obs,
                    "n_subclass_accepted": int(accepted.sum()),
                    "n_subclass_unresolved": int((~accepted).sum()),
                    "embeddings_reused_from_compute": True,
                    "spatial_ground_truth_validated": False,
                },
                indent=2,
            )
            + "\n"
        )
        return paths
    finally:
        data.file.close()

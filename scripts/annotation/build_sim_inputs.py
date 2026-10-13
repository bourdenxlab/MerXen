#!/usr/bin/env python
"""Build the M3c simulation-input assets from the phase-1 5K evidence.

Your decision of 2026-09-28 (OD-E1 amended) lets public vendor data supply
simulation inputs, stored with provenance, never as trust evidence (plan
§4.7, §8.3 v7.4-v7.5). This script derives the small tables of
``src/merxen/assets/annotation/sim_inputs/`` from the phase-1 evidence
tables (which were computed from the public 10x files recorded in
``public_validation/MANIFEST.tsv``) and writes, for each, a provenance
sidecar ``<asset_id>.provenance.json`` and the ``NOTICE``:

- ``efficiency__xenium_prime__mouse__wmb10xv3.csv`` (role ``member``): the
  5,006 Prime 5K mouse genes' class-level factor against WMB 10Xv3
  (``log2_factor``: centred on the median informative gene, capped +-3),
  tier, expected and observed counts, top WMB class and that class's share
  of the section's called cells;
- ``depth__xenium_prime__mouse_brain_ff.csv`` (role ``profile``): the 63,147
  vendor cells' called WMB class (coded; legend in the sidecar) and total
  counts in source order, no cell ids; cross-checked against
  ``mouse5k/depth_profile.csv``;
- ``ratio__xenium_prime_vs_v1__human_lung_ffpe.csv`` (role ``stress``): the
  196 shared lung genes' per-area 5K / v1 log2 ratio, centred on its median
  and capped +-2;
- ``depth__xenium_prime__human_lung_ffpe.csv`` (role ``scenario``): the
  histogram of the 275,556 lung 5K cells' totals (>= 10 counts);
- ``panels__xenium_prime.csv`` (role ``panel_list``): the pinned public
  Prime 5K human and mouse gene lists (Ensembl ids), which resolve a
  panel's chemistry.

The ``NOTICE`` also lists, apart from these, the in-house assets other
scripts register in the same directory (their sidecars' ``source_kind`` is
``in_house``; M13 D7: ``build_xplatform_stress.py``), by the source paths
under the evidence root their sidecars record. Re-run this script after
such a script changes a sidecar.

Usage::

    python scripts/annotation/build_sim_inputs.py \\
        --evidence-root /srv/storage/MerXen/annotation_dev/evidence_20260926 \\
        --public-root /srv/storage/MerXen/public_validation

``--check`` writes nothing and exits 1 when a committed file differs from
what the inputs produce. Sidecars record the inputs by their path under the
evidence root (never the local root) and their sha256.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import logging
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from merxen.annotation import sim_inputs as si
from merxen.annotation.public_panels import PUBLIC_PANEL_LISTS

logger = logging.getLogger("build_sim_inputs")

DERIVATION_DATE = "2026-09-28"
SCRIPT_PATH = "scripts/annotation/build_sim_inputs.py"
MOUSE_DATASET = "xenium_prime_5k_mouse_brain_ff"
LUNG_5K_DATASET = "xenium_prime_5k_human_lung_ffpe"
LUNG_V1_DATASET = "xenium_v1_human_lung_ffpe"
# The vendor files phase 1 read (no transcripts, images or outs bundles).
DATASET_FILE_SUFFIXES = (
    "_cell_feature_matrix.h5",
    "_cells.parquet",
    "_gene_panel.json",
    "_experiment.xenium",
)
MOUSE_LABEL = "XOA 3.0 vendor segmentation, public 10x section"
LUNG_LABEL = (
    "XOA 3.0 vendor segmentation, public 10x FFPE lung tumour sections "
    "(cross-tissue approximation)"
)
SEGMENTATION_MOUSE = (
    "vendor XOA 3.0.0.15 multimodal cell segmentation (Xenium Multi-Tissue "
    "Stain); 58% of transcripts assigned, median cell area 76 um2"
)
SEGMENTATION_LUNG = "vendor XOA 3.0.0.15 segmentation of each section (5K and v1)"
REFERENCE_WMB = (
    "WMB-10Xv3 (Allen Brain Cell Atlas, taxonomy CCN20230722; Yao et al. 2023); "
    "expected counts from the reference pseudobulk of the called WMB classes "
    "(RCTD-style; equal to shadow.reference_pseudobulk_log2_factors)"
)
EVIDENCE = {
    "synthesis": "5k_real/SYNTHESIS.txt",
    "sensitivity": "5k_real/sensitivity/REPORT.txt",
    "sim": "5k_real/sim/REPORT.txt",
    "review": "5k_real/review/REVIEW_CHECKS.txt",
    "mouse5k": "5k_real/mouse5k/REPORT.txt",
    "stage_a": "m3c/STAGE_A_DESIGN_NUMBERS.txt",
}


class SimInputBuildError(RuntimeError):
    """An input is missing or inconsistent."""


@dataclass(frozen=True)
class BuiltAsset:
    """One generated asset: its CSV text and sidecar."""

    asset: si.SimInputAsset
    csv_text: str


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_record(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    if not path.is_file():
        raise SimInputBuildError(f"input {path} is missing")
    return {
        "evidence_path": relative,
        "size": path.stat().st_size,
        "sha256": si._sha256_file(path),
    }


def _manifest(public_root: Path) -> pd.DataFrame:
    path = public_root / "MANIFEST.tsv"
    if not path.is_file():
        raise SimInputBuildError(f"{path} is missing")
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def _dataset_sources(manifest: pd.DataFrame, dataset: str) -> list[dict[str, Any]]:
    rows = manifest[manifest["dataset_key"] == dataset]
    records = []
    for row in rows.itertuples(index=False):
        if not str(row.file).endswith(DATASET_FILE_SUFFIXES):
            continue
        records.append(
            {
                "dataset_key": dataset,
                "file": str(row.file),
                "url": str(row.url),
                "size": int(row.size_bytes),
                "sha256": str(row.sha256),
                "downloaded_utc": str(row.download_date_utc),
                "server_last_modified": str(row.server_last_modified),
            }
        )
    if len(records) != len(DATASET_FILE_SUFFIXES):
        raise SimInputBuildError(
            f"MANIFEST.tsv holds {len(records)} of the {len(DATASET_FILE_SUFFIXES)} "
            f"vendor files of {dataset}"
        )
    return records


def _dataset_meta(manifest: pd.DataFrame, dataset: str) -> dict[str, str]:
    row = manifest[manifest["dataset_key"] == dataset].iloc[0]
    return {
        "licence_terms": str(row["licence_terms"]),
        "source_page": str(row["source_page"]),
        "notes": str(row["notes"]),
    }


def _script_sha256(repo_root: Path) -> str:
    return str(si._sha256_file(repo_root / SCRIPT_PATH))


def _csv_text(frame: pd.DataFrame, float_format: Mapping[str, str]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(list(frame.columns))
    formats = dict(float_format)
    for row in frame.itertuples(index=False):
        values = []
        for column, value in zip(frame.columns, row, strict=True):
            if value is None or (isinstance(value, float) and not np.isfinite(value)):
                values.append("")
            elif column in formats:
                values.append(format(float(value), formats[column]))
            else:
                values.append(str(value))
        writer.writerow(values)
    return buffer.getvalue()


def _asset(
    *,
    asset_id: str,
    csv_text: str,
    role: str,
    species: str | None,
    tissue: str,
    reference: str | None,
    label: str,
    description: str,
    columns: Mapping[str, str],
    provenance: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> BuiltAsset:
    payload = csv_text.encode("utf-8")
    asset = si.SimInputAsset(
        schema_version=si.SIM_INPUTS_SCHEMA_VERSION,
        asset_id=asset_id,
        asset_version=1,
        file=f"{asset_id}.csv",
        sha256=_sha256_bytes(payload),
        size=len(payload),
        n_rows=csv_text.count("\n") - 1,
        role=role,  # type: ignore[arg-type]
        species=species,  # type: ignore[arg-type]
        chemistry="xenium_prime",
        tissue=tissue,
        reference=reference,
        label=label,
        description=description,
        columns=dict(columns),
        extra=dict(extra or {}),
        trust_effect="none",
        provenance=dict(provenance),
    )
    return BuiltAsset(asset=asset, csv_text=csv_text)


def _common_provenance(
    *,
    repo_root: Path,
    manifest: pd.DataFrame,
    datasets: Sequence[str],
    derived_from: Sequence[dict[str, Any]],
    xoa: str,
    segmentation: str,
    method: str,
    evidence: Sequence[str],
) -> dict[str, Any]:
    meta = _dataset_meta(manifest, datasets[0])
    sources = [
        item for dataset in datasets for item in _dataset_sources(manifest, dataset)
    ]
    return {
        "source_dataset": "; ".join(
            f"{dataset} (10x Genomics public dataset)" for dataset in datasets
        ),
        "source_page": "; ".join(
            sorted(
                {
                    _dataset_meta(manifest, dataset)["source_page"]
                    for dataset in datasets
                }
            )
        ),
        "source_notes": "; ".join(
            _dataset_meta(manifest, dataset)["notes"] for dataset in datasets
        ),
        "source_files": sources,
        "derived_from": list(derived_from),
        "licence": "CC BY 4.0 (Creative Commons Attribution 4.0 International)",
        "licence_terms_recorded": meta["licence_terms"],
        "licence_note": si.LICENCE_NOTE,
        "attribution": "10x Genomics (public Xenium datasets)",
        "xoa_version": xoa,
        "segmentation": segmentation,
        "deriving_script": SCRIPT_PATH,
        "deriving_script_sha256": _script_sha256(repo_root),
        "method": method,
        "evidence": [f"$A/{item}" for item in evidence],
        "evidence_root": "$A = /srv/storage/MerXen/annotation_dev/evidence_20260926",
        "date": DERIVATION_DATE,
        "use": "simulation input only (OD-E1 amended 2026-09-28); never trust "
        "evidence, never a gate",
    }


def build_efficiency(
    evidence_root: Path, manifest: pd.DataFrame, repo_root: Path
) -> BuiltAsset:
    """Return the 5K mouse member table (class-level factors vs WMB 10Xv3)."""
    factors_rel = "5k_real/sensitivity/mouse_5k_platform_factors.csv"
    specificity_rel = "5k_real/sensitivity/mouse_5k_class_specificity.tsv"
    calls_rel = "5k_real/mouse5k/qc/percell_calls_mouse5k.parquet"
    factors = pd.read_csv(evidence_root / factors_rel)
    specificity = pd.read_csv(evidence_root / specificity_rel, sep="\t")
    calls = pd.read_parquet(evidence_root / calls_rel, columns=["class"])
    if not factors["gene_id"].is_unique or len(factors) != 5006:
        raise SimInputBuildError(f"{factors_rel}: expected 5,006 unique genes")
    if set(specificity["gene_id"]) != set(factors["gene_id"]):
        raise SimInputBuildError(f"{specificity_rel}: genes differ from the factors")
    capped = factors["log2_factor"].to_numpy(np.float64)
    if np.nanmax(np.abs(capped)) > si.R3_CAP_LOG2 + 1e-12:
        raise SimInputBuildError(f"{factors_rel}: log2_factor exceeds the +-3 cap")
    sim = factors["sim_log2_factor"].to_numpy(np.float64)
    measured = factors["tier"].isin([si.TIER_INFORMATIVE, si.TIER_WEAK]).to_numpy()
    if not np.array_equal(sim[measured], capped[measured]):
        raise SimInputBuildError(
            f"{factors_rel}: sim_log2_factor differs from log2_factor on "
            "informative / weak genes"
        )
    share = calls["class"].astype(str).value_counts(normalize=True)
    top = specificity.set_index("gene_id")["top_class"].astype(str)
    table = pd.DataFrame(
        {
            "gene_id": factors["gene_id"].astype(str),
            "tier": factors["tier"].astype(str),
            "log2_factor": capped,
            "expected_counts": factors["expected_counts"].to_numpy(np.float64),
            "observed_counts": factors["observed_counts"].to_numpy(np.float64),
            "top_class": top.reindex(factors["gene_id"]).to_numpy(),
            "top_class_section_share": top.reindex(factors["gene_id"])
            .map(share)
            .fillna(0.0)
            .to_numpy(np.float64),
        }
    ).sort_values("gene_id", kind="mergesort")
    if not np.array_equal(
        table["observed_counts"].to_numpy(), np.round(table["observed_counts"])
    ):
        raise SimInputBuildError(f"{factors_rel}: observed counts are not integers")
    table["observed_counts"] = table["observed_counts"].astype(np.int64)
    text = _csv_text(
        table,
        {
            "log2_factor": ".17g",
            "expected_counts": ".10g",
            "top_class_section_share": ".6g",
        },
    )
    tiers = table["tier"].value_counts().to_dict()
    restricted = si.measured_mask(table.set_index("gene_id"))
    provenance = _common_provenance(
        repo_root=repo_root,
        manifest=manifest,
        datasets=[MOUSE_DATASET],
        derived_from=[
            _file_record(evidence_root, factors_rel),
            _file_record(evidence_root, specificity_rel),
            _file_record(evidence_root, calls_rel),
        ],
        xoa="xenium-3.0.0.15",
        segmentation=SEGMENTATION_MOUSE,
        method=(
            "Per-gene platform factor of the public 5K section against WMB 10Xv3: "
            "log2(observed / expected) over confidently called cells, expected = "
            "cell totals x the reference pseudobulk profile of the cell's called "
            "class, centred on the median informative gene and capped at +-3 log2 "
            "(sensitivity task; class-level, class vs subclass r .991, REVIEW_CHECKS "
            "C1). Tier: informative expected >= 1,000 counts, weak 200-1,000, "
            "uninformative < 200. top_class: the WMB class with the largest share "
            "of the gene's reference expression (mouse_5k_class_specificity.tsv); "
            "top_class_section_share: that class's share of the section's 63,147 "
            "called cells (percell_calls_mouse5k.parquet)."
        ),
        evidence=[
            EVIDENCE["synthesis"],
            EVIDENCE["sensitivity"],
            EVIDENCE["review"],
            EVIDENCE["sim"],
        ],
    )
    provenance["reference_source"] = REFERENCE_WMB
    provenance["summary"] = {
        "n_genes": int(len(table)),
        "tiers": {str(key): int(value) for key, value in sorted(tiers.items())},
        "n_restricted_measured": int(restricted.sum()),
    }
    return _asset(
        asset_id=si.EFFICIENCY_MOUSE_PRIME,
        csv_text=text,
        role="member",
        species="mouse",
        tissue="brain_ff",
        reference="wmb_10xv3",
        label=MOUSE_LABEL,
        description=(
            "Measured per-gene factors of the R3_measured_HO member for Xenium "
            "Prime 5K mouse against WMB 10Xv3 (plan §8.3 v7.4); rule restricted: "
            "informative genes whose top class holds >= 0.5% of the section take "
            "their factor + N(0, 0.20 log2), all others a keyed resample"
        ),
        columns={
            "gene_id": "Ensembl gene id",
            "tier": "informative | weak | uninformative (expected counts)",
            "log2_factor": "class-level log2 factor, centred, capped +-3",
            "expected_counts": "expected counts from the reference",
            "observed_counts": "observed counts in the section",
            "top_class": "WMB class with the largest share of the gene's expression",
            "top_class_section_share": "that class's share of the section's cells",
        },
        provenance=provenance,
    )


def build_mouse_profile(
    evidence_root: Path, manifest: pd.DataFrame, repo_root: Path
) -> BuiltAsset:
    """Return the public 5K mouse per-cell depth profile (class, totals)."""
    from merxen.annotation.vocab import NEURONS, load_asset_table

    calls_rel = "5k_real/mouse5k/qc/percell_calls_mouse5k.parquet"
    profile_rel = "5k_real/mouse5k/depth_profile.csv"
    calls = pd.read_parquet(
        evidence_root / calls_rel, columns=["cell_id", "total_counts", "class"]
    )
    check = pd.read_csv(
        evidence_root / profile_rel, usecols=["cell_id", "total_counts", "class"]
    )
    if not (
        len(calls) == len(check)
        and np.array_equal(calls["cell_id"].astype(str), check["cell_id"].astype(str))
        and np.array_equal(calls["total_counts"], check["total_counts"])
        and np.array_equal(calls["class"].astype(str), check["class"].astype(str))
    ):
        raise SimInputBuildError(f"{calls_rel} and {profile_rel} differ")
    vocab = load_asset_table("wmb_class_vocab.csv")
    order = list(vocab["class"])
    classes = sorted(set(calls["class"].astype(str)), key=order.index)
    code = {name: index for index, name in enumerate(classes)}
    neuronal = dict(zip(vocab["class"], vocab["broad_class"] == NEURONS, strict=True))
    table = pd.DataFrame(
        {
            "class_code": calls["class"].astype(str).map(code).astype(np.int64),
            "total_counts": calls["total_counts"].astype(np.int64),
        }
    )
    text = _csv_text(table, {})
    legend = [
        {
            "code": code[name],
            "class": name,
            "neuronal": bool(neuronal[name]),
            "n_cells": int((calls["class"] == name).sum()),
            "median_counts": float(
                calls.loc[calls["class"] == name, "total_counts"].median()
            ),
        }
        for name in classes
    ]
    provenance = _common_provenance(
        repo_root=repo_root,
        manifest=manifest,
        datasets=[MOUSE_DATASET],
        derived_from=[
            _file_record(evidence_root, calls_rel),
            _file_record(evidence_root, profile_rel),
        ],
        xoa="xenium-3.0.0.15",
        segmentation=SEGMENTATION_MOUSE,
        method=(
            "Total counts (Gene Expression features) and the production-settings "
            "WMB class call (bundle c30f7eab, bf 0.5, 100 iterations, seed 0) of "
            "each of the 63,147 vendor cells with >= 10 counts, in source order, "
            "no cell ids; cross-checked cell by cell against depth_profile.csv "
            "(sha256 ff35db2e...)."
        ),
        evidence=[EVIDENCE["synthesis"], EVIDENCE["mouse5k"], EVIDENCE["sim"]],
    )
    provenance["cross_check"] = {
        "file": profile_rel,
        "sha256": si._sha256_file(evidence_root / profile_rel),
        "equal": True,
    }
    provenance["summary"] = {
        "n_cells": int(len(table)),
        "median_counts": float(calls["total_counts"].median()),
        "q05": float(calls["total_counts"].quantile(0.05)),
        "q95": float(calls["total_counts"].quantile(0.95)),
    }
    return _asset(
        asset_id=si.PROFILE_MOUSE_PRIME_FF,
        csv_text=text,
        role="profile",
        species="mouse",
        tissue="brain_ff",
        reference="wmb_10xv3",
        label=MOUSE_LABEL,
        description=(
            "Per-class depth profile of Xenium Prime 5K fresh-frozen mouse brain "
            "(one coronal hemisphere, ~AP 6.8 mm) for profile mode and the "
            "class-depth predictions (plan §8.3 v7.5); in-sample for the section"
        ),
        columns={
            "class_code": "called WMB class (legend in extra.class_legend)",
            "total_counts": "total counts of the cell",
        },
        extra={"class_legend": legend},
        provenance=provenance,
    )


def build_lung_ratio(
    evidence_root: Path, manifest: pd.DataFrame, repo_root: Path
) -> BuiltAsset:
    """Return the lung 5K / v1 per-area ratio table (stress recipe)."""
    ratio_rel = "5k_real/sensitivity/human_lung_5k_vs_v1.csv"
    lung = pd.read_csv(evidence_root / ratio_rel)
    ratio = lung["ratio_per_tissue_area_um2"].to_numpy(np.float64)
    log2 = np.clip(
        np.log2(ratio / np.median(ratio)), -si.XTISSUE_CAP_LOG2, si.XTISSUE_CAP_LOG2
    )
    table = pd.DataFrame(
        {"gene_id": lung["gene_id"].astype(str), "log2_ratio": log2}
    ).sort_values("gene_id", kind="mergesort")
    text = _csv_text(table, {"log2_ratio": ".17g"})
    provenance = _common_provenance(
        repo_root=repo_root,
        manifest=manifest,
        datasets=[LUNG_5K_DATASET, LUNG_V1_DATASET],
        derived_from=[_file_record(evidence_root, ratio_rel)],
        xoa="xenium-3.0.0.15 (both sections)",
        segmentation=SEGMENTATION_LUNG,
        method=(
            "Per-gene 5K / v1 ratio of counts per um2 of tissue for the 196 genes "
            "of both lung panels (FFPE lung tumour sections, not confirmed "
            "serial), log2 of ratio / median ratio, capped at +-2 log2. A "
            "cross-tissue approximation: 5K-vs-v1 in lung, not 5K-vs-WHB in brain "
            "(SYNTHESIS §4.1, §5.3); stress recipe only."
        ),
        evidence=[EVIDENCE["synthesis"], EVIDENCE["sensitivity"], EVIDENCE["review"]],
    )
    provenance["summary"] = {
        "n_genes": int(len(table)),
        "median_ratio": float(np.median(ratio)),
        "sd_ln": float(np.std(np.log(ratio), ddof=1)),
    }
    return _asset(
        asset_id=si.STRESS_HUMAN_LUNG,
        csv_text=text,
        role="stress",
        species="human",
        tissue="lung_ffpe",
        reference=None,
        label=LUNG_LABEL,
        description=(
            "Lung 5K / v1 per-area log2 ratios of the R1_xtissue_lung_stress "
            "recipe (plan §8.3 v7.4): reported for human Prime families and added "
            "to gate P's NP6, never an emission member"
        ),
        columns={
            "gene_id": "Ensembl gene id",
            "log2_ratio": "log2(5K / v1 per-area ratio / its median), capped +-2",
        },
        provenance=provenance,
    )


def build_lung_scenario(
    evidence_root: Path, manifest: pd.DataFrame, repo_root: Path
) -> BuiltAsset:
    """Return the lung 5K FFPE depth histogram (a human scenario)."""
    totals_rel = "5k_real/sim/tables/lung5k_totals.npy"
    totals = np.load(evidence_root / totals_rel)
    if not np.array_equal(totals, np.round(totals)) or totals.min() < 10:
        raise SimInputBuildError(f"{totals_rel}: expected integer totals >= 10")
    values, counts = np.unique(totals.astype(np.int64), return_counts=True)
    table = pd.DataFrame({"total_counts": values, "n_cells": counts})
    text = _csv_text(table, {})
    provenance = _common_provenance(
        repo_root=repo_root,
        manifest=manifest,
        datasets=[LUNG_5K_DATASET],
        derived_from=[_file_record(evidence_root, totals_rel)],
        xoa="xenium-3.0.0.15",
        segmentation=SEGMENTATION_LUNG,
        method=(
            "Histogram of the Gene Expression totals of the 5K lung section's "
            "cells with >= 10 counts (cell_feature_matrix.h5; simlib.lung5k_totals)."
            " A pooled depth scenario for human Prime (FFPE, median 245): never a "
            "per-class profile, never brain."
        ),
        evidence=[EVIDENCE["synthesis"], EVIDENCE["sim"]],
    )
    provenance["summary"] = {
        "n_cells": int(len(totals)),
        "median_counts": float(np.median(totals)),
        "n_distinct_totals": int(len(values)),
    }
    return _asset(
        asset_id=si.SCENARIO_HUMAN_LUNG,
        csv_text=text,
        role="scenario",
        species="human",
        tissue="lung_ffpe",
        reference=None,
        label=LUNG_LABEL,
        description=(
            "Pooled total-count histogram of the public lung 5K FFPE section: the "
            "lung-FFPE depth scenario of human Prime predictions (plan §8.10)"
        ),
        columns={
            "total_counts": "total counts",
            "n_cells": "cells with that total",
        },
        provenance=provenance,
    )


def build_panel_lists(
    evidence_root: Path, manifest: pd.DataFrame, repo_root: Path
) -> BuiltAsset:
    """Return the pinned public Prime 5K gene lists (chemistry resolution)."""
    rows = []
    derived = []
    sources = []
    for key in ("xenium_prime_5k_human", "xenium_prime_5k_mouse"):
        item = PUBLIC_PANEL_LISTS[key]
        relative = f"m3b/panels/{key}.gene_list.csv"
        derived.append(_file_record(evidence_root, relative))
        table = pd.read_csv(evidence_root / relative, dtype=str)
        genes = sorted(set(table["gene_id"].astype(str)))
        if len(genes) != item.n_genes:
            raise SimInputBuildError(
                f"{relative}: {len(genes)} genes, the pin says {item.n_genes}"
            )
        rows += [
            {"panel_key": key, "species": item.species, "gene_id": g} for g in genes
        ]
        sources.append(
            {
                "dataset_key": key,
                "file": item.file_name,
                "url": item.url,
                "size": item.size,
                "sha256": item.sha256,
            }
        )
    text = _csv_text(pd.DataFrame(rows, columns=list(si.PANEL_LIST_COLUMNS)), {})
    provenance = {
        "source_dataset": "10x Genomics pre-designed Xenium Prime 5K panel "
        "definitions (human and mouse Pan Tissue and Pathways)",
        "source_page": "https://www.10xgenomics.com/products/xenium-panels",
        "source_files": sources,
        "derived_from": derived,
        "licence": "vendor panel definition files (public); gene identifiers only",
        "licence_note": si.LICENCE_NOTE,
        "attribution": "10x Genomics (Xenium panel definitions)",
        "xoa_version": "n/a (panel definition)",
        "segmentation": "n/a (panel definition)",
        "deriving_script": SCRIPT_PATH,
        "deriving_script_sha256": _script_sha256(repo_root),
        "method": (
            "Ensembl gene ids of the pinned vendor panel files "
            "(merxen.annotation.public_panels, URL + sha256 pinned in M3b), from "
            "the normalised lists annotation-panel-fetch wrote."
        ),
        "evidence": ["$A/m3b/panels/RETRIEVAL.txt"],
        "evidence_root": "$A = /srv/storage/MerXen/annotation_dev/evidence_20260926",
        "date": DERIVATION_DATE,
        "use": "chemistry resolution only (plan §3.7 panel_chemistry)",
    }
    asset = _asset(
        asset_id=si.PRIME_PANEL_LISTS,
        csv_text=text,
        role="panel_list",
        species=None,
        tissue="n/a",
        reference=None,
        label="pinned public Xenium Prime 5K panel lists",
        description=(
            "Gene ids of the public Prime 5K panels; a Xenium panel with Jaccard "
            ">= 0.95 to its species' list resolves to chemistry xenium_prime"
        ),
        columns={
            "panel_key": "public panel key (annotation-panel-fetch)",
            "species": "human | mouse",
            "gene_id": "Ensembl gene id",
        },
        provenance=provenance,
    )
    return asset


BUILDERS = (
    build_efficiency,
    build_mouse_profile,
    build_lung_ratio,
    build_lung_scenario,
    build_panel_lists,
)


def in_house_assets(directory: Path) -> list[si.SimInputAsset]:
    """Return the in-house assets registered in ``directory``, by asset id.

    Other scripts build them (``source_kind`` ``in_house``, e.g. M13 D7's
    ``build_xplatform_stress.py``); the NOTICE lists them apart.

    Args:
        directory: The asset directory.

    Returns:
        Their sidecar records.
    """
    assets = [
        si.read_sidecar(path)
        for path in sorted(directory.glob(f"*{si.SIDECAR_SUFFIX}"))
    ]
    return [
        asset
        for asset in assets
        if asset.provenance.get(si.SOURCE_KIND_KEY) == si.SOURCE_KIND_IN_HOUSE
    ]


def _in_house_lines(assets: Sequence[si.SimInputAsset]) -> list[str]:
    """The NOTICE section of the in-house assets (empty without one)."""
    if not assets:
        return []
    lines = [
        "In-house assets",
        "---------------",
        "",
        "Derived from this project's own sections by their own scripts, not",
        "from public data: their sidecars record each source file by its path",
        "under the evidence root and its sha256 (no URL).",
        "",
    ]
    for asset in assets:
        provenance = asset.provenance
        lines.append(f"- {asset.file} ({asset.role}, {asset.species}, {asset.tissue}):")
        lines.append(f"  {asset.label}.")
        lines.append(f"  Source: {provenance.get('source_dataset')}.")
        for item in provenance.get("source_files") or []:
            lines.append(f"    {item['path']}")
            lines.append(f"      sha256 {item['sha256']}")
        lines.append(f"  Deriving script: {provenance.get('deriving_script')}.")
        lines.append(f"  Licence: {provenance.get('licence')}.")
        lines.append(f"  Attribution: {provenance.get('attribution')}.")
        lines.append("")
    return lines


def render_notice(
    assets: Sequence[si.SimInputAsset],
    in_house: Sequence[si.SimInputAsset] = (),
) -> str:
    """Return the ``sim_inputs/NOTICE`` text.

    Args:
        assets: The public-data assets this script builds.
        in_house: The in-house assets other scripts registered
            (``in_house_assets``), listed in their own section.

    Returns:
        The NOTICE text.
    """
    lines = [
        "MerXen simulation-input assets: provenance and licences",
        "=======================================================",
        "",
        f"Generated by {SCRIPT_PATH}; do not edit.",
        "",
        "Simulation inputs derived from public vendor data (plan §4.7; OD-E1",
        'amended 2026-09-28) and, listed under "In-house assets", from this',
        "project's own sections: they parameterise simulated reference cells",
        "only, never raise a trust state and never enter a gate. Each table has",
        "a provenance sidecar (<asset>.provenance.json) with the source URLs (an",
        "in-house asset: the paths under the evidence root) and sha256 of every",
        "source file, the evidence tables it was derived from, the XOA version,",
        "the segmentation, the deriving script's sha256 and the date.",
        "",
    ]
    for asset in assets:
        provenance = asset.provenance
        lines.append(f"- {asset.file} ({asset.role}, {asset.species}, {asset.tissue}):")
        lines.append(f"  {asset.label}.")
        lines.append(f"  Source: {provenance.get('source_dataset')}.")
        for item in provenance.get("source_files") or []:
            lines.append(f"    {item['url']}")
            lines.append(f"      sha256 {item['sha256']}")
        lines.append(f"  Licence: {provenance.get('licence')}.")
        lines.append(f"  Attribution: {provenance.get('attribution')}.")
        if provenance.get("reference_source"):
            lines.append(f"  Reference: {provenance['reference_source']}.")
        lines.append("")
    lines += [
        "Licences",
        "--------",
        "",
        "The 10x Genomics public Xenium datasets are distributed under the",
        "Creative Commons Attribution 4.0 International licence (CC BY 4.0) as",
        "stated on their dataset pages; attribution: 10x Genomics. The licence",
        f"text was {si.LICENCE_NOTE}.",
        "The WMB-10Xv3 reference behind the expected counts is Allen Brain Cell",
        "Atlas data (CC BY-NC 4.0 unless a release states otherwise; see",
        "../NOTICE); attribution: Allen Institute for Brain Science.",
        "",
    ]
    lines += _in_house_lines(in_house)
    return "\n".join(lines)


def build_all(
    evidence_root: Path,
    public_root: Path,
    repo_root: Path,
    asset_dir: Path = si.SIM_INPUTS_DIR,
) -> dict[str, str]:
    """Build every generated file: CSVs, sidecars and the NOTICE.

    Args:
        evidence_root: The evidence root (``$A``).
        public_root: The public-data root (``MANIFEST.tsv``).
        repo_root: The repository root (for this script's sha256).
        asset_dir: The asset directory whose in-house sidecars the NOTICE
            lists.

    Returns:
        File name -> text.
    """
    manifest = _manifest(public_root)
    built = [builder(evidence_root, manifest, repo_root) for builder in BUILDERS]
    files: dict[str, str] = {}
    for item in built:
        files[item.asset.file] = item.csv_text
        files[f"{item.asset.asset_id}{si.SIDECAR_SUFFIX}"] = si.sidecar_json(item.asset)
        if item.asset.size > si.MAX_ASSET_BYTES:
            raise SimInputBuildError(
                f"{item.asset.file}: {item.asset.size} bytes exceed the "
                "repository limit"
            )
    ours = {item.asset.asset_id for item in built}
    in_house = [
        asset for asset in in_house_assets(asset_dir) if asset.asset_id not in ours
    ]
    files[si.NOTICE_FILE] = render_notice([item.asset for item in built], in_house)
    return files


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=si.SIM_INPUTS_DIR)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Write nothing; exit 1 if a committed file is stale.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the generator.

    Args:
        argv: Command-line arguments (``sys.argv[1:]`` when ``None``).

    Returns:
        The process exit code.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv)
    files = build_all(
        args.evidence_root, args.public_root, args.repo_root, args.output_dir
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stale = []
    for name, text in files.items():
        path = args.output_dir / name
        current = path.read_text(encoding="utf-8") if path.is_file() else None
        if current == text:
            logger.info("%s is up to date", name)
            continue
        stale.append(name)
        if not args.check:
            path.write_text(text, encoding="utf-8")
            logger.info("Wrote %s (%d bytes)", path, len(text.encode("utf-8")))
    if args.check and stale:
        logger.error("Stale generated files: %s", ", ".join(stale))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

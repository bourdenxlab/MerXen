#!/usr/bin/env python
"""Build gate P's cross-platform stress asset from the set a offsets (M13 D7).

Gate P's NP6 stresses each R1 member with "per-gene log2 factors drawn from
the measured human cross-platform offsets (capped +-2)" (plan §14 NP6). The
M13 decision D7 (a) + (i) (pre-registration §23.10) fixes the source and the
rule: the registered D-A7 basis ``research/xplat/
log2_xen_over_mer_gene_means.csv`` (the 296 set a genes' log2 Xenium /
MERSCOPE mean counts per cell on the four paired sections P7513, P7113,
P1212 and P5011, proseg_hybrid), centred on the median gene, capped at +-2
log2 and registered as a ``sim_inputs`` ``stress`` asset with provenance
(source path, sha256 and this deriving script); panel genes the table does
not cover take a keyed resample of its values at simulation time
(``sim_inputs.xplatform_stress_efficiency``).

This script writes ``ratio__xenium_v1_vs_merscope__human_brain_ffpe.csv``
(``gene_id``, ``log2_ratio``) and its sidecar into
``src/merxen/assets/annotation/sim_inputs/``:

1. each pair's column is centred on its median gene (the per-pair offsets
   of -0.12 to +1.52 log2 are a platform-wide scale, which the median
   normalisation of every efficiency removes anyway);
2. the gene's offset is the mean of its four centred values;
3. the means are centred on their median gene and capped at +-2 log2;
4. gene symbols take the Ensembl ids of the packaged set a panel
   (``validated_panel_genes.csv``, ``human_set_a_296``); a symbol without
   one is refused.

Neither D7 nor pre-registration §23.10 says how the four pairs are
combined: steps 1 and 2 are this script's choice, put to the user in
pre-registration §23.15 item 8 (on the source, the averaged offsets' SD
before the cap is 1.617 log2 against 1.59-1.77 for single pairs, and 20.6%
of genes are capped against 19-24%).

The data are in-house, so the sidecar records ``source_kind: in_house`` and
each source file by its path under the evidence root and its sha256 (no
URL). It never records a local path outside the evidence root.

Usage::

    python scripts/annotation/build_xplatform_stress.py \\
        --evidence-root /srv/storage/MerXen/annotation_dev/evidence_20260926

``--check`` writes nothing and exits 1 when a committed file differs from
what the inputs produce. ``scripts/annotation/build_sim_inputs.py`` writes
the directory's ``NOTICE``, which lists this asset under "In-house assets";
re-run it after this script changes the sidecar.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from merxen.annotation import sim_inputs as si
from merxen.annotation.vocab import ASSET_DIR

logger = logging.getLogger("build_xplatform_stress")

DERIVATION_DATE = "2026-10-06"
SCRIPT_PATH = "scripts/annotation/build_xplatform_stress.py"
SOURCE_RELATIVE = "research/xplat/log2_xen_over_mer_gene_means.csv"
SOURCE_SCRIPT_RELATIVE = "research/xplat/xplat_genes.py"
SOURCE_REPORT_RELATIVE = "research/xplat/xplat_genes.out"
EVIDENCE_ROOT_TEXT = "$A = /srv/storage/MerXen/annotation_dev/evidence_20260926"
PAIRS = ("P7513", "P7113", "P1212", "P5011")
SET_A_PANEL = "human_set_a_296"
GENES_FILE = "validated_panel_genes.csv"
GENES_REPO_PATH = f"src/merxen/assets/annotation/{GENES_FILE}"
LABEL = (
    "MerXen proseg_hybrid segmentation, in-house paired Xenium / MERSCOPE "
    "brain sections (set a)"
)
SEGMENTATION = (
    "MerXen proseg_hybrid (Cellpose-SAM prior and ProSeg) on both platforms of "
    "each pair"
)
METHOD = (
    "Per gene, log2 of the Xenium / MERSCOPE mean counts per cell over all "
    "cells of each of the four paired sections (P7513, P7113, P1212, P5011; "
    "set a's 296 shared genes; research/xplat/xplat_genes.py); each pair "
    "centred on its median gene, averaged over the pairs, centred on the "
    "median gene and capped at +-2 log2. Gate P's NP6 multiplies an R1 "
    "member's LogNormal draw by 2 ** offset; panel genes the table does not "
    "cover take a keyed resample of these values (M13 decision D7 (a) + (i))."
)


class XplatformBuildError(RuntimeError):
    """An input is missing or inconsistent."""


@dataclass(frozen=True)
class BuiltAsset:
    """The generated asset: its CSV text and sidecar."""

    asset: si.SimInputAsset
    csv_text: str


def _file_record(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    if not path.is_file():
        raise XplatformBuildError(f"input {path} is missing")
    return {
        "evidence_path": relative,
        "size": path.stat().st_size,
        "sha256": si._sha256_file(path),
    }


def set_a_gene_ids(genes_path: Path) -> pd.Series:
    """Return set a's Ensembl id per gene symbol (``human_set_a_296``).

    Raises:
        XplatformBuildError: If the panel is missing or a symbol or id repeats.
    """
    table = pd.read_csv(genes_path, dtype=str, keep_default_na=False)
    panel = table[table["panel_id"] == SET_A_PANEL]
    if panel.empty:
        raise XplatformBuildError(f"{genes_path} has no {SET_A_PANEL} rows")
    if not (panel["gene_symbol"].is_unique and panel["ensembl_id"].is_unique):
        raise XplatformBuildError(f"{SET_A_PANEL}: a symbol or an id repeats")
    return pd.Series(
        panel["ensembl_id"].to_numpy(), index=panel["gene_symbol"].to_numpy()
    ).sort_index()


def gene_map_sha256(gene_ids: pd.Series) -> str:
    """Return the sha256 of the symbol -> id rows used (``symbol,id`` lines).

    Only these rows are hashed, so rows other panels add to the table later
    do not change the record.
    """
    text = "".join(f"{symbol},{gene_id}\n" for symbol, gene_id in gene_ids.items())
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def xplatform_offsets(source: pd.DataFrame, cap_log2: float) -> pd.Series:
    """Return the per-gene offsets (steps 1-3 of the module docstring).

    Args:
        source: Gene symbols x pairs (``PAIRS``), log2 Xenium / MERSCOPE.
        cap_log2: The cap (+-2 log2).

    Returns:
        The capped offsets per symbol, in the source order.

    Raises:
        XplatformBuildError: If a pair column is missing, a symbol repeats or
            a value is not finite.
    """
    missing = [pair for pair in PAIRS if pair not in source.columns]
    if missing:
        raise XplatformBuildError(f"the source lacks the pairs {missing}")
    if not source.index.is_unique:
        raise XplatformBuildError("the source repeats a gene symbol")
    values = source[list(PAIRS)].to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise XplatformBuildError("the source holds values that are not finite")
    centred = values - np.median(values, axis=0, keepdims=True)
    mean = centred.mean(axis=1)
    offsets = np.clip(mean - np.median(mean), -cap_log2, cap_log2)
    return pd.Series(offsets, index=source.index.astype(str), name="log2_ratio")


def _csv_text(table: pd.DataFrame) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(list(si.RATIO_COLUMNS))
    for gene_id, value in zip(table["gene_id"], table["log2_ratio"], strict=True):
        writer.writerow([str(gene_id), format(float(value), ".17g")])
    return buffer.getvalue()


def build_asset(evidence_root: Path, genes_path: Path, repo_root: Path) -> BuiltAsset:
    """Return the cross-platform stress asset (CSV text and sidecar).

    Args:
        evidence_root: The evidence root (``$A``).
        genes_path: The packaged ``validated_panel_genes.csv``.
        repo_root: The repository root (for this script's sha256).

    Returns:
        The asset.

    Raises:
        XplatformBuildError: For a missing input, a symbol without a set a id
            or an inconsistent source.
    """
    source_path = evidence_root / SOURCE_RELATIVE
    if not source_path.is_file():
        raise XplatformBuildError(f"input {source_path} is missing")
    source = pd.read_csv(source_path, index_col=0)
    offsets = xplatform_offsets(source, si.XPLATFORM_CAP_LOG2)
    gene_ids = set_a_gene_ids(genes_path)
    unmapped = sorted(set(offsets.index) - set(gene_ids.index))
    if unmapped:
        raise XplatformBuildError(
            f"the symbols {unmapped[:10]} have no {SET_A_PANEL} Ensembl id"
        )
    used = gene_ids.reindex(sorted(offsets.index))
    table = (
        pd.DataFrame(
            {
                "gene_id": gene_ids.reindex(offsets.index).to_numpy(),
                "log2_ratio": offsets.to_numpy(),
            }
        )
        .sort_values("gene_id", kind="mergesort")
        .reset_index(drop=True)
    )
    text = _csv_text(table)
    payload = text.encode("utf-8")
    pair_medians = source[list(PAIRS)].median(axis=0)
    source_record = _file_record(evidence_root, SOURCE_RELATIVE)
    provenance: dict[str, Any] = {
        si.SOURCE_KIND_KEY: si.SOURCE_KIND_IN_HOUSE,
        "source_dataset": (
            "MerXen set a paired sections P7513, P7113, P1212 and P5011 "
            "(in-house human brain sections, each pair one Xenium and one "
            "adjacent MERSCOPE section on the same custom gene panel)"
        ),
        "source_page": (
            "plan D-A7 (docs/plans/robust-celltype-annotation-plan.md); "
            f"$A/{SOURCE_REPORT_RELATIVE}"
        ),
        "source_files": [
            {
                "file": Path(SOURCE_RELATIVE).name,
                "path": f"$A/{SOURCE_RELATIVE}",
                "size": source_record["size"],
                "sha256": source_record["sha256"],
            }
        ],
        "derived_from": [
            source_record,
            _file_record(evidence_root, SOURCE_SCRIPT_RELATIVE),
            {
                "repo_path": GENES_REPO_PATH,
                "panel_id": SET_A_PANEL,
                "n_rows": int(len(used)),
                "rows": "symbol,ensembl_id lines of the genes used, by symbol",
                "sha256": gene_map_sha256(used),
            },
        ],
        "licence": (
            "in-house data, no public licence; the asset holds derived per-gene "
            "offsets only"
        ),
        "licence_note": "in-house data: no public licence text applies",
        "attribution": "in-house paired sections of this project (set a)",
        "xoa_version": (
            "not applicable: both platforms' sections were re-segmented by MerXen"
        ),
        "segmentation": SEGMENTATION,
        "deriving_script": SCRIPT_PATH,
        "deriving_script_sha256": si._sha256_file(repo_root / SCRIPT_PATH),
        "method": METHOD,
        "evidence": [
            f"$A/{SOURCE_REPORT_RELATIVE}",
            "$A/m13/M13_IMPLEMENTATION_PLAN.md",
            "$A/m13/DECISIONS_RESOLVED.md",
        ],
        "evidence_root": EVIDENCE_ROOT_TEXT,
        "date": DERIVATION_DATE,
        "use": (
            "gate-P stress input only (plan §14 NP6; M13 decision D7); never "
            "trust evidence, never a PREP member"
        ),
        "summary": {
            "n_genes": int(len(table)),
            "pairs": list(PAIRS),
            "pair_median_log2": {pair: float(pair_medians[pair]) for pair in PAIRS},
            "sd_log2_before_cap": float(
                np.std(xplatform_offsets(source, np.inf).to_numpy(np.float64), ddof=1)
            ),
            "n_capped": int(
                (np.abs(table["log2_ratio"]) >= si.XPLATFORM_CAP_LOG2 - 1e-12).sum()
            ),
        },
    }
    asset = si.SimInputAsset(
        schema_version=si.SIM_INPUTS_SCHEMA_VERSION,
        asset_id=si.STRESS_HUMAN_XPLATFORM,
        asset_version=1,
        file=f"{si.STRESS_HUMAN_XPLATFORM}.csv",
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
        n_rows=text.count("\n") - 1,
        role="stress",
        species="human",
        chemistry="xenium_v1",
        tissue="brain_ffpe",
        reference=None,
        label=LABEL,
        description=(
            "Set a's per-gene log2 Xenium / MERSCOPE offsets, gate P's NP6 "
            "cross-platform stress (plan §14 NP6; M13 decision D7): each R1 "
            "member's efficiency times 2 ** offset, uncovered panel genes a "
            "keyed resample; never a PREP member"
        ),
        columns={
            "gene_id": f"Ensembl gene id ({SET_A_PANEL})",
            "log2_ratio": (
                "log2(Xenium / MERSCOPE) mean counts per cell, each pair centred "
                "on its median gene, averaged over the four pairs, centred on "
                "the median gene and capped +-2"
            ),
        },
        extra={
            si.STRESS_KIND_KEY: si.STRESS_KIND_CROSS_PLATFORM,
            "pairs": list(PAIRS),
            "ratio": "log2(Xenium / MERSCOPE)",
        },
        trust_effect="none",
        provenance=provenance,
    )
    if asset.size > si.MAX_ASSET_BYTES:
        raise XplatformBuildError(f"{asset.file}: {asset.size} bytes exceed the limit")
    return BuiltAsset(asset=asset, csv_text=text)


def build_files(
    evidence_root: Path, genes_path: Path, repo_root: Path
) -> dict[str, str]:
    """Return the generated files: the CSV and its sidecar."""
    built = build_asset(evidence_root, genes_path, repo_root)
    return {
        built.asset.file: built.csv_text,
        f"{built.asset.asset_id}{si.SIDECAR_SUFFIX}": si.sidecar_json(built.asset),
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--genes", type=Path, default=ASSET_DIR / GENES_FILE)
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
    files = build_files(args.evidence_root, args.genes, args.repo_root)
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

#!/usr/bin/env python
"""M6b inputs: gene sets, section subsamples and prepared queries (§24).

Pre-registration §24 (``docs/acceptance/annotation-v1-preregistration.md``)
and plan §7.8 item 1. Three steps, each writing under ``--out``:

1. ``gene-sets``: the three WMB panel gene sets, MERFISH 500 (the 500
   genes of the MERFISH-C57BL6J-638850 ``gene.csv``, Blank controls
   excluded), MERFISH ∩ MS1 and MERFISH ∩ MS2 (Ensembl IDs of the resolved
   MS1 / MS2 ``panel_genes.json``), in MERFISH ``gene.csv`` order
   (``gene_sets/<name>.csv``, ``gene_sets.json``).
2. ``subsample``: per in-scope section (CB + OB < 50% of grey cells in
   ``plan_rev2/ap_axis/section_region_share.csv``), the cells of the CCF view
   that are also in the expression h5ad, sorted by ``cell_label``, and
   30,000 of them drawn without replacement by
   ``numpy.random.default_rng([0, NN])`` (NN = the section number), or all
   when fewer (``subsamples/<section>.parquet`` with the Allen labels, the
   CCF parcellation and the section coordinates; ``subsamples.json`` with
   the counts and the sha256 of every cell list).
3. ``prepare``: one ``CLUSTERING_SQUIDPY_PREPARE``-style directory per gene
   set and section (``prepared/<gene_set>/<pair>/clustering_prepare_out``
   with ``manifest.json`` and ``merscope/<sid>_prepared.h5ad``: raw counts of
   the subsample on the gene set in ``X``, ``var`` indexed by symbol with
   the Ensembl ID in ``gene_ids``, ``obsm["spatial"]`` in µm; plus
   ``clustering_squidpy_config.json``), which ``merxen annotate
   --prepared-dir`` maps as a MERSCOPE sample.

Usage::

    mouse_ap_axis_inputs.py gene-sets --merfish-genes gene.csv \
        --ms1-panel ms1/panel_genes.json --ms2-panel ms2/panel_genes.json --out DIR
    mouse_ap_axis_inputs.py subsample --ccf-metadata VIEW.csv --h5ad RAW.h5ad \
        --section-table section_region_share.csv --out DIR
    mouse_ap_axis_inputs.py prepare --h5ad RAW.h5ad --out DIR

The gene sets are keyed by the sections' aliases; the files and records
name them as the data does (``privacy.to_real(form="name")``: on a host with
the alias table the names of the recorded M6b inputs).

Imports: numpy, pandas, h5py, anndata, scipy (the merxen analysis env).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from merxen import privacy

logger = logging.getLogger("mouse_ap_axis_inputs")

SUBSAMPLE_CELLS = 30_000
SUBSAMPLE_SEED = 0
MIN_COUNTS = 10
PLATFORM = "MERSCOPE"
SEGMENTATION = "allen_merfish"
GENE_SET_NAMES: tuple[str, ...] = ("merfish500", "merfish_ms1", "merfish_ms2")
# Plan §7.8: MERFISH 500, MERFISH ∩ MS1 188, MERFISH ∩ MS2 346.
EXPECTED_GENE_SET_SIZES: dict[str, int] = {
    "merfish500": 500,
    "merfish_ms1": 188,
    "merfish_ms2": 346,
}
CCF_COLUMNS: tuple[str, ...] = (
    "cell_label",
    "brain_section_label",
    "cluster_alias",
    "average_correlation_score",
    "class",
    "subclass",
    "supertype",
    "x_section",
    "y_section",
    "x_ccf",
    "parcellation_division",
    "parcellation_structure",
)
SECTION_PREFIX = "C57BL6J-638850."


@dataclass(frozen=True)
class SubsampleRecord:
    """The subsample of one section.

    Attributes:
        section: MERFISH section label.
        section_number: NN of the label (the seed's second entry).
        ap_ccf_mm: Median ``x_ccf`` of the section's view cells.
        n_view_cells: Cells of the section in the CCF view.
        n_h5ad_cells: Of these, cells in the expression h5ad.
        n_drawn: Cells drawn.
        seed: ``numpy.random.default_rng`` seed sequence.
        cells_sha256: sha256 of the drawn ``cell_label`` values joined by
            newlines (sorted).
    """

    section: str
    section_number: int
    ap_ccf_mm: float
    n_view_cells: int
    n_h5ad_cells: int
    n_drawn: int
    seed: list[int]
    cells_sha256: str


def section_number(section: str) -> int:
    """Return the section number NN of ``C57BL6J-638850.NN``.

    Args:
        section: A MERFISH-C57BL6J-638850 section label.

    Returns:
        The number after the last dot.

    Raises:
        ValueError: If the label has no numeric suffix.
    """
    suffix = section.rsplit(".", 1)[-1]
    if not suffix.isdigit() or "." not in section:
        raise ValueError(f"{section!r} is not a C57BL6J-638850.NN section label")
    return int(suffix)


def pair_id_for(section: str) -> str:
    """Return the pair id of a section's prepared query (``MF638850_NN``)."""
    return f"MF638850_{section_number(section):02d}"


def sample_id_for(section: str) -> str:
    """Return the sample id of a section's prepared query."""
    return f"{pair_id_for(section)}_{PLATFORM}"


def cells_sha256(cell_labels: Sequence[str]) -> str:
    """Return the sha256 of cell labels joined by newlines."""
    return hashlib.sha256("\n".join(cell_labels).encode("utf-8")).hexdigest()


def draw_subsample(
    cell_labels: Sequence[str],
    section: str,
    *,
    n_cells: int = SUBSAMPLE_CELLS,
    seed: int = SUBSAMPLE_SEED,
) -> list[str]:
    """Draw the pre-registered subsample of one section (§24).

    Args:
        cell_labels: The section's candidate cells (any order).
        section: Section label (its number seeds the draw).
        n_cells: Cells to draw; all when fewer exist.
        seed: First entry of the seed sequence.

    Returns:
        The drawn cell labels, sorted.

    Raises:
        ValueError: If a label repeats.
    """
    ordered = sorted(str(label) for label in cell_labels)
    if len(set(ordered)) != len(ordered):
        raise ValueError(f"{section}: duplicate cell labels")
    if len(ordered) <= n_cells:
        return ordered
    rng = np.random.default_rng([seed, section_number(section)])
    chosen = np.sort(rng.choice(len(ordered), size=n_cells, replace=False))
    return [ordered[int(position)] for position in chosen]


def in_scope_sections(share_table: pd.DataFrame) -> list[str]:
    """Return the in-scope sections (CB + OB < 50% of grey cells), by AP.

    Args:
        share_table: ``section_region_share.csv`` (index = section; columns
            ``CB``, ``OB``, ``ap``).

    Returns:
        Section labels sorted by AP.
    """
    kept = share_table[(share_table["CB"] + share_table["OB"]) < 0.5]
    return [str(section) for section in kept.sort_values("ap").index]


def build_gene_sets(
    merfish_genes: pd.DataFrame, panels: Mapping[str, Sequence[str]]
) -> dict[str, pd.DataFrame]:
    """Return the three M6b gene sets in MERFISH ``gene.csv`` order.

    Args:
        merfish_genes: MERFISH ``gene.csv`` (``gene_identifier``,
            ``gene_symbol``).
        panels: ``{"ms1": ids, "ms2": ids}``: resolved Ensembl IDs.

    Returns:
        ``{name: DataFrame(gene_ids, gene_symbol)}`` for ``GENE_SET_NAMES``.
    """
    genes = merfish_genes[
        merfish_genes["gene_identifier"].astype(str).str.startswith("ENSMUSG")
    ]
    frame = pd.DataFrame(
        {
            "gene_ids": genes["gene_identifier"].astype(str).to_numpy(),
            "gene_symbol": genes["gene_symbol"].astype(str).to_numpy(),
        }
    )
    ms1 = {str(value) for value in panels["ms1"]}
    ms2 = {str(value) for value in panels["ms2"]}
    return {
        "merfish500": frame.reset_index(drop=True),
        "merfish_ms1": frame[frame["gene_ids"].isin(ms1)].reset_index(drop=True),
        "merfish_ms2": frame[frame["gene_ids"].isin(ms2)].reset_index(drop=True),
    }


def data_name(name: str) -> str:
    """How the files and records name a gene set or panel (R3, ``form="name"``)."""
    return privacy.to_real(name, form="name")


def rows_of(obs_labels: Sequence[str], wanted: Sequence[str]) -> np.ndarray:
    """Return the h5ad row of each wanted cell label.

    Args:
        obs_labels: The h5ad ``obs`` index, in row order.
        wanted: Cell labels to find.

    Returns:
        Row positions in the order of ``wanted``.

    Raises:
        KeyError: If a wanted cell is not in the h5ad.
    """
    index = pd.Index([str(label) for label in obs_labels])
    positions = index.get_indexer([str(label) for label in wanted])
    if (positions < 0).any():
        missing = [str(wanted[i]) for i in np.flatnonzero(positions < 0)[:5]]
        raise KeyError(f"cells missing from the h5ad: {missing}")
    return positions.astype(np.int64)


def csr_rows(
    data: np.ndarray, indices: np.ndarray, indptr: np.ndarray, rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gather rows of a CSR matrix held as its three arrays.

    Args:
        data: CSR data.
        indices: CSR column indices.
        indptr: CSR row pointers.
        rows: Rows to take, in output order.

    Returns:
        ``(data, indices, indptr)`` of the gathered rows.
    """
    starts = indptr[rows]
    ends = indptr[rows + 1]
    lengths = ends - starts
    out_indptr = np.zeros(len(rows) + 1, dtype=np.int64)
    np.cumsum(lengths, out=out_indptr[1:])
    if out_indptr[-1] == 0:
        return (
            np.zeros(0, dtype=data.dtype),
            np.zeros(0, dtype=indices.dtype),
            out_indptr,
        )
    offsets = np.repeat(starts - out_indptr[:-1], lengths)
    take = np.arange(out_indptr[-1], dtype=np.int64) + offsets
    return data[take], indices[take], out_indptr


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _script_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _provenance() -> dict[str, Any]:
    return {"git_commit": _git_commit(), "script_sha256": _script_sha256()}


def _read_h5ad_obs(path: Path) -> tuple[list[str], list[str]]:
    import h5py

    with h5py.File(path, "r") as handle:
        obs = handle["obs"]
        index_key = obs.attrs.get("_index", "_index")
        labels = [value.decode() for value in obs[index_key][:]]
        section = obs["brain_section_label"]
        if isinstance(section, h5py.Group):  # categorical
            categories = [value.decode() for value in section["categories"][:]]
            codes = section["codes"][:]
            sections = [categories[int(code)] for code in codes]
        else:
            sections = [value.decode() for value in section[:]]
    return labels, sections


def run_gene_sets(args: argparse.Namespace) -> None:
    """Write the three gene sets (step 1)."""
    genes = pd.read_csv(args.merfish_genes)
    panels = {
        "ms1": json.loads(Path(args.ms1_panel).read_text())["ensembl_ids"],
        "ms2": json.loads(Path(args.ms2_panel).read_text())["ensembl_ids"],
    }
    sets = build_gene_sets(genes, panels)
    out = Path(args.out) / "gene_sets"
    out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"provenance": _provenance(), "sets": {}}
    for name, frame in sets.items():
        expected = EXPECTED_GENE_SET_SIZES[name]
        if len(frame) != expected:
            raise SystemExit(f"{name}: {len(frame)} genes, expected {expected}")
        frame.to_csv(out / f"{data_name(name)}.csv", index=False)
        summary["sets"][data_name(name)] = {
            "n_genes": len(frame),
            "ids_sha256": cells_sha256(sorted(frame["gene_ids"])),
        }
    summary["inputs"] = {
        "merfish_genes": str(args.merfish_genes),
        data_name("ms1_panel"): str(args.ms1_panel),
        data_name("ms2_panel"): str(args.ms2_panel),
    }
    (Path(args.out) / "gene_sets.json").write_text(json.dumps(summary, indent=2))
    logger.info("gene sets: %s", {k: v["n_genes"] for k, v in summary["sets"].items()})


def run_subsample(args: argparse.Namespace) -> None:
    """Draw and write the per-section subsamples (step 2)."""
    shares = pd.read_csv(args.section_table, index_col=0)
    sections = in_scope_sections(shares)
    if args.expected_sections and len(sections) != args.expected_sections:
        raise SystemExit(
            f"{len(sections)} in-scope sections, expected {args.expected_sections}"
        )
    view = pd.read_csv(args.ccf_metadata, usecols=list(CCF_COLUMNS), engine="pyarrow")
    labels, _ = _read_h5ad_obs(Path(args.h5ad))
    in_h5ad = set(labels)
    out = Path(args.out) / "subsamples"
    out.mkdir(parents=True, exist_ok=True)
    records: list[SubsampleRecord] = []
    for section in sections:
        cells = view[view["brain_section_label"] == section]
        present = cells[cells["cell_label"].astype(str).isin(in_h5ad)]
        drawn = draw_subsample(present["cell_label"].astype(str).tolist(), section)
        table = (
            present.set_index(present["cell_label"].astype(str))
            .loc[drawn]
            .reset_index(drop=True)
        )
        table.to_parquet(out / f"{section}.parquet", index=False)
        records.append(
            SubsampleRecord(
                section=section,
                section_number=section_number(section),
                ap_ccf_mm=float(cells["x_ccf"].median()),
                n_view_cells=len(cells),
                n_h5ad_cells=len(present),
                n_drawn=len(drawn),
                seed=[SUBSAMPLE_SEED, section_number(section)],
                cells_sha256=cells_sha256(drawn),
            )
        )
        logger.info("%s: %d of %d cells drawn", section, len(drawn), len(present))
    summary = {
        "provenance": _provenance(),
        "rule": (
            "cells of the CCF view also in the h5ad, sorted by cell_label; "
            f"{SUBSAMPLE_CELLS} drawn without replacement by "
            "numpy.random.default_rng([0, NN]), all when fewer"
        ),
        "inputs": {
            "ccf_metadata": str(args.ccf_metadata),
            "h5ad": str(args.h5ad),
            "section_table": str(args.section_table),
        },
        "n_h5ad_cells_total": len(labels),
        "n_view_cells_total": len(view),
        "sections": [asdict(record) for record in records],
    }
    (Path(args.out) / "subsamples.json").write_text(json.dumps(summary, indent=2))


def _write_prepared(
    directory: Path,
    *,
    section: str,
    counts: Any,
    cells: pd.DataFrame,
    genes: pd.DataFrame,
) -> None:
    import anndata as ad

    pair_id = pair_id_for(section)
    sample_id = sample_id_for(section)
    prepare_out = directory / "clustering_prepare_out"
    (prepare_out / PLATFORM.lower()).mkdir(parents=True, exist_ok=True)
    var = pd.DataFrame(
        {"gene_ids": genes["gene_ids"].to_numpy()},
        index=pd.Index(genes["gene_symbol"].to_numpy(), name=None),
    )
    obs = pd.DataFrame(
        {"brain_section_label": cells["brain_section_label"].astype(str).to_numpy()},
        index=pd.Index(cells["cell_label"].astype(str).to_numpy(), name=None),
    )
    adata = ad.AnnData(X=counts, obs=obs, var=var)
    adata.obsm["spatial"] = (
        cells[["x_section", "y_section"]].to_numpy(dtype=np.float64) * 1000.0
    )
    relative = f"{PLATFORM.lower()}/{sample_id}_prepared.h5ad"
    adata.write_h5ad(prepare_out / relative)
    (prepare_out / "manifest.json").write_text(
        json.dumps({"samples": {sample_id: relative}}, indent=2)
    )
    config = {
        "pair_id": pair_id,
        "samples": [
            {
                "sample_id": sample_id,
                "platform": PLATFORM,
                "segmentation": SEGMENTATION,
                "merfish_section": section,
            }
        ],
        "min_counts": MIN_COUNTS,
    }
    (directory / "clustering_squidpy_config.json").write_text(
        json.dumps(config, indent=2)
    )


def run_prepare(args: argparse.Namespace) -> None:
    """Write the prepared queries of every section and gene set (step 3)."""
    import h5py
    from scipy import sparse

    root = Path(args.out)
    summary = json.loads((root / "subsamples.json").read_text())
    sets = {
        data_name(name): pd.read_csv(root / "gene_sets" / f"{data_name(name)}.csv")
        for name in GENE_SET_NAMES
    }
    with h5py.File(args.h5ad, "r") as handle:
        var = handle["var"]
        var_ids = [value.decode() for value in var[var.attrs["_index"]][:]]
        matrix = handle["X"]
        indptr = matrix["indptr"][:]
        logger.info("reading X (%d non-zeros)", int(indptr[-1]))
        data = matrix["data"][:]
        indices = matrix["indices"][:]
    labels, _ = _read_h5ad_obs(Path(args.h5ad))
    columns = {
        name: np.array([var_ids.index(gene) for gene in frame["gene_ids"]])
        for name, frame in sets.items()
    }
    written: list[dict[str, Any]] = []
    for record in summary["sections"]:
        section = record["section"]
        cells = pd.read_parquet(root / "subsamples" / f"{section}.parquet")
        rows = rows_of(labels, cells["cell_label"].astype(str).tolist())
        row_data, row_indices, row_indptr = csr_rows(data, indices, indptr, rows)
        full = sparse.csr_matrix(
            (row_data, row_indices, row_indptr), shape=(len(rows), len(var_ids))
        )
        if not np.allclose(full.data, np.round(full.data)):
            raise SystemExit(f"{section}: non-integer raw counts")
        for name, frame in sets.items():
            counts = full[:, columns[name]].astype(np.float32).tocsr()
            directory = root / "prepared" / name / pair_id_for(section)
            _write_prepared(
                directory, section=section, counts=counts, cells=cells, genes=frame
            )
            totals = np.asarray(counts.sum(axis=1)).ravel()
            written.append(
                {
                    "gene_set": name,
                    "section": section,
                    "pair_id": pair_id_for(section),
                    "n_cells": int(counts.shape[0]),
                    "n_table_cells": int((totals >= MIN_COUNTS).sum()),
                    "median_counts": float(np.median(totals)),
                }
            )
        logger.info("%s: prepared on %d gene sets", section, len(sets))
    table = pd.DataFrame(written)
    table.to_csv(root / "prepared_summary.csv", index=False)
    (root / "prepared.json").write_text(
        json.dumps({"provenance": _provenance(), "h5ad": str(args.h5ad)}, indent=2)
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    genes = commands.add_parser("gene-sets")
    genes.add_argument("--merfish-genes", type=Path, required=True)
    genes.add_argument("--ms1-panel", type=Path, required=True)
    genes.add_argument("--ms2-panel", type=Path, required=True)
    genes.add_argument("--out", type=Path, required=True)
    sample = commands.add_parser("subsample")
    sample.add_argument("--ccf-metadata", type=Path, required=True)
    sample.add_argument("--h5ad", type=Path, required=True)
    sample.add_argument("--section-table", type=Path, required=True)
    sample.add_argument("--expected-sections", type=int, default=38)
    sample.add_argument("--out", type=Path, required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--h5ad", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one step.

    Args:
        argv: Command-line arguments (default: ``sys.argv[1:]``).

    Returns:
        Exit code.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = _parser().parse_args(argv)
    {"gene-sets": run_gene_sets, "subsample": run_subsample, "prepare": run_prepare}[
        args.command
    ](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

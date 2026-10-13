#!/usr/bin/env python
"""M8b segmentation comparison, reseg vs proseg_hybrid (pre-registration §19).

Implements the arms and analyses 1-4 of pre-registration §19 (plan §12 M8b)
on the M8 stage B outputs of P7513, P1212, P7113 and P5011 (MERSCOPE and
Xenium; P7113 and P5011 are held-out donors, reported separately: every
table carries ``role``). M8b scores nothing for gate H and fits no
threshold: every number here is evidence for the user's OD-B7' decision.

Arms (§19):

- ``reseg``: the reseg map_first outputs (``table_MOSAIK_proseg``);
- ``hybrid``: the proseg_hybrid map_first outputs;
- ``hybrid_matched``: the proseg_hybrid cells thinned to the reseg cell of
  the same id (``thin_to_targets``: multivariate hypergeometric without
  replacement, seed 0, each cell's generator keyed by its id; a cell whose
  proseg_hybrid total is at most its reseg total keeps its counts), then
  mapped and resolved with the production configuration: stage B1's MAP,
  RESOLVE and ANNOTATION_REPORT tasks re-run in fresh directories with the
  same staged inputs and command, the prepared counts (MAP, RESOLVE) and the
  MAP / RESOLVE outputs (REPORT) replaced, nothing else (``matched-task``).
  Every row of the prepared H5AD is thinned to its reseg total (0 for an id
  reseg lacks), so the table cells (>= 10 counts) are exactly the
  proseg_hybrid table cells whose id is a reseg table cell, and the
  segmented objects stay those of proseg_hybrid;
- ``combined``: the proseg_hybrid table cells carrying the RESOLVE labels of
  the reseg cell with the same id (unlabelled without a reseg table cell),
  with proseg_hybrid counts (analysis 4; analysis 1 records it as
  ``not_available``);
- context: ``proseg_mask`` and ``original_seg`` (analysis 1 only).

Analysis 1 (side-by-side scores). Every ``acceptance_metrics.json`` record
of each arm (the pipeline's reports; hybrid_matched's report from
``matched-task REPORT``), keyed by criterion, name, kind, level, region,
scope, platform and group, wide by arm, with ``reseg - hybrid`` and
``reseg - hybrid_matched`` for numeric values and each arm's own CI beside
it; the broad JSDs (soft, soft_ge30, confident, argmax; whole section and
shared mask) also get the paired CI of those differences
(``shadow.paired_block_bootstrap_jsd_difference``, one 500 µm tile grid over
both arms and platforms, 200 replicates, seed 0). The §14 criterion values
of ``resolve_criteria.py``, ``heldout_genes.py`` (H4) and
``marker_referee.py``, run identically per arm on a symlink farm that puts
the arm where the scripts read proseg_hybrid (``farms``, ``acceptance``),
and H12 from ``report_scoring.score_h12`` on each arm's records. Coverage
and the status shares per depth bin (10, 15, 30, 60, 120, 250) and level,
from each arm's label table. Inputs the pre-registration leaves unchanged for
hybrid_matched (the clustered H5ADs, so the count-based report items 4 and
7, H9's pseudo-labels and H10's marker scores read the proseg_hybrid counts;
the cortical depth, MENDER and the shared mask) are named in every row's
``inputs_note``; hybrid_matched's H4 is ``not_available`` (its held-out
re-map would re-map the unchanged proseg_hybrid counts).

Analysis 2 (transcript-loss audit, per dataset). Transcripts: the rows of
``points/transcripts`` of the published store with qv >= the samplesheet's
``xenium_min_qv`` on Xenium (no filter on MERSCOPE: the segmentation passes
no qv threshold there and ProSeg ran without one), control features removed
(``panel.ControlRegistry``). Assigned under reseg: ``assignment`` set,
``background`` false, a reseg table cell; under proseg_hybrid:
``hybrid_assignment`` set, ``hybrid_background`` false, a proseg_hybrid table
cell. In a nucleus: (x, y) inside a ``shapes/cellpose_nuclei`` polygon
(boundary included). Per gene: ``A_reseg``, ``A_hybrid``, ``N_bg`` (with its
background and non-table-cell parts) and ``L = 1 - A_reseg / A_hybrid``; per
gene within each confident proseg_hybrid broad class (and supercluster where
the gate is ``full``) and within glia / neurons (E2's classes: Astro, Oligo,
OPC, Immune, Vascular incl. fibroblasts vs Exc + Inh), over the transcripts
of the proseg_hybrid cells of that label: an in-nucleus transcript belongs
to the cell that holds its nucleus (``nucleus_holders``: the proseg_hybrid
polygon of largest overlap), any other transcript to its proseg_hybrid table
cell. ``A_reseg_same_cell`` (assigned under reseg to the cell of the same
id) is reported beside. Covariates: expression level (log10 of the mean
proseg_hybrid counts per table cell), the longest annotated transcript
(``--gene-annotation-gtf``; primary chromosomes), the Xenium probe count
(``gene_panel.json`` ``info.gene_coverage``; MERSCOPE: ``not_available``, no
probe counts on the host), membership of ``NUCLEAR_RETAINED`` and
``PROCESS_LOCALISED`` (below, with sources; genes not on the panel dropped).
Tests (one Benjamini-Hochberg family over all of analysis 2): Spearman rho
of L and of N_bg with each continuous covariate (95% CI from 1,000 gene
resamples, seed 0); Mann-Whitney of L and of N_bg by each list; glia vs
neurons: Wilcoxon signed-rank of the per-gene L difference; platforms:
Wilcoxon signed-rank of L, MERSCOPE vs Xenium, over the pair's genes. Effect
on expression: per confident broad class c (cells confident c in both arms,
same ids) and gene g, ``p_arm = (sum g + 1) / (sum genes + n_genes)``, the
log2 ratio reseg / hybrid, the factor ``log2(p_arm / p_WHB)`` against the
primary bundle's supercluster profiles aggregated to broad (cell-weighted),
its spread (q90 - q10) per arm and the arms' Spearman rho; marker contrasts
``log2(p(g, c1) / p(g, c2))`` for the E1 referee markers. The genes of the
expression metrics are the dataset's genes with an Ensembl id in the bundle
profiles.

Analysis 3 (spill-over audit, per dataset). Cells confident at broad level in
both reseg and hybrid (same ids); each arm's own label, its RESOLVE
``contamination_score`` and its counts; the distance from the arm's native
centroid (the store's shape polygon) to the nearest centroid of a confident
cell of another broad class in that arm, binned 0-10, 10-15, 15-20, 20-30,
30-50, > 50 µm. Per class, bin and arm: the mean contamination score and the
mean detection rate of the class's negative genes (bundle
``negative_genes.parquet``); hybrid - reseg per bin with a 95% CI from 200
resamples of 500 µm tiles of the proseg_hybrid centroids (seed 0, one draw
for both arms); per negative gene and class, its detection rate per arm and
hybrid - reseg (with the same tile CI). ``cells = same_label`` repeats the
rows on the cells with the same confident label in both arms.

Analysis 4 (combined mode). On the proseg_hybrid table cells: the share with
a confident reseg label per level and the share without a reseg table cell;
the confident broad composition and the cross-platform confident broad JSD
(whole section, §2's matched-tile CI) beside reseg and hybrid, with paired
CIs of combined - hybrid and combined - reseg; the analysis 2 expression
metrics with each arm's own confident cells (combined: reseg labels on
proseg_hybrid counts); the pipeline change the mode would need (stated, not
implemented).

Post hoc (added after the 2026-10-01 review of the first decision memo; not
pre-registered, information only: they fit, score and select nothing, and
analyses 1-4 are unchanged). ``posthoc.csv`` and ``tables/posthoc_*.csv``:

- ``selection_depth``: each arm's confident share per level and H2's
  implausible share on the cells in both tables (the hybrid_matched table
  cells) beside the whole tables, and ``reseg(R) - hybrid(H)`` split into
  cell selection, depth (``hybrid_matched - hybrid`` on the same cells) and
  the rest (``reseg - hybrid_matched`` on the same cells);
- ``jsd_reading``: each paired JSD difference of analysis 1 with the side
  of 0 its CI lies on;
- ``spillover_summary``: analysis 3's rows counted by the side of 0 of the
  CI, distance bins apart from whole-class rows, and the near (0-10 µm)
  minus far (> 50 µm) difference per class and dataset;
- ``leaking_genes``: analysis 3's per-gene detection-rate differences
  ranked by their median over class x dataset rows;
- ``noise_floor``, ``noise_strata``: the control features (MERSCOPE blank
  barcodes; the Xenium stores' transcripts carry none) tallied like analysis
  2's genes (``controls``), and analysis 2's per-gene loss within strata of
  the gene's decoded count over the median control feature's;
- ``common_bins``, ``common_bins_summary``: analysis 3 on the cells with the
  same confident label in both arms, with one distance per cell for both
  arms (its proseg_hybrid centroid to the nearest proseg_hybrid confident
  cell of another broad class), so a bin holds the same cells in each arm;
- ``matched_count_sources``: which hybrid_matched items read the thinned
  counts.

Subcommands, in order (each writes only under ``--out``; the published
results and the stores are read, never written):

1. ``matched-prepare --pair P``: the thinned prepared H5ADs of P;
2. ``matched-task --pair P --process MAP|RESOLVE|REPORT``: re-run B1's task;
3. ``farms``; 4. ``acceptance --arm A`` (heldout_genes, resolve_criteria,
   marker_referee on the arm's farm);
5. ``transcripts --pair P --platform PLAT``: analysis 2's transcript tallies;
6. ``analyze``: analyses 1-4 -> ``<out>/segmentation_comparison/``
   (``analysis<N>.csv``, one per analysis with a ``table`` column, and
   ``tables/``);
7. post hoc: ``controls --pair P --platform PLAT`` (the control features'
   tallies), then ``posthoc`` (``posthoc.csv``, after ``analyze``);
8. ``render``: figures (PNG + PDF + CSV) and ``index.html``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from scipy import sparse

from merxen.annotation import composition as co
from merxen.annotation import segmentation_compare as sc
from merxen.annotation.panel import ControlRegistry
from merxen.annotation.schema import safe_token
from merxen.annotation.vocab import HUMAN_BROAD_CLASSES

logger = logging.getLogger("segmentation_comparison")

PAIRS: Final[tuple[str, ...]] = ("P7513", "P1212", "P7113", "P5011")
DEVELOPMENT_PAIRS: Final[frozenset[str]] = frozenset({"P7513", "P1212"})
PLATFORMS: Final[tuple[str, ...]] = ("MERSCOPE", "XENIUM")
MAIN_ARMS: Final[tuple[str, ...]] = ("reseg", "hybrid", "hybrid_matched")
CONTEXT_ARMS: Final[tuple[str, ...]] = ("proseg_mask", "original_seg")
COMBINED: Final = "combined"
ARM_SEGMENTATION: Final[dict[str, str]] = {
    "reseg": "reseg",
    "hybrid": "proseg_hybrid",
    "hybrid_matched": "proseg_hybrid",
    "proseg_mask": "proseg_mask",
    "original_seg": "original_seg",
}
# Store shape keys of the two compared segmentations (native frame).
ARM_SHAPES: Final[dict[str, str]] = {
    "reseg": "MOSAIK_proseg",
    "hybrid": "MOSAIK_proseg_hybrid",
}
NUCLEI_SHAPES: Final = "cellpose_nuclei"
FARM_SEGMENTATION: Final = "proseg_hybrid"
LEVELS: Final[tuple[str, ...]] = (
    "lineage",
    "broad",
    "nt",
    "supercluster",
    "seaad_subclass",
)
DEPTH_BINS: Final[tuple[int, ...]] = (10, 15, 30, 60, 120, 250)
STATUSES: Final[tuple[str, ...]] = (
    "confident",
    "low_counts",
    "parent_unresolved",
    "not_resolvable",
    "low_confidence",
    "implausible",
    "below_floor",
    "single_method",
    "method_disagree",
    "not_attempted_gate",
    "not_applicable",
)
CONFIDENT: Final = "confident"
# The table-cell threshold (clustering min_counts; MAP's in_table).
TABLE_MIN_COUNTS: Final = 10
FULL_GATE: Final = "full"
PROFILE_LEVEL: Final = "CCN202210140_SUPC"
PRIMARY_REFERENCE: Final = "whb_frontal_supc_clus"
TILE_UM: Final = 500.0
N_JSD_BOOTSTRAP: Final = 200
JSD_KINDS: Final[tuple[str, ...]] = ("soft", "soft_ge30", "confident", "argmax")
PROCESSES: Final[dict[str, str]] = {
    "PREPARE": "CLUSTERING_SQUIDPY_PREPARE",
    "MAP": "CLUSTERING_SQUIDPY_ANNOTATE_MAP",
    "RESOLVE": "CLUSTERING_SQUIDPY_ANNOTATE_RESOLVE",
    "REPORT": "ANNOTATION_REPORT",
}
PYTHONPATH_LINE: Final = re.compile(r'export PYTHONPATH="([^":]+):\$\{PYTHONPATH:-\}"')
RECORD_KEY: Final[tuple[str, ...]] = (
    "criterion",
    "name",
    "kind",
    "level",
    "region",
    "scope",
    "platform",
    "group",
)
MATCHED_NOTE: Final = (
    "hybrid_matched: MAP and RESOLVE on the thinned counts; every other input "
    "unchanged (pre-registration §19): the proseg_hybrid clustered H5ADs (the "
    "counts of report items 4 and 7, of H9's pseudo-labels and H10's marker "
    "scores), cortical depth, MENDER and the shared mask"
)
H4_MATCHED_REASON: Final = (
    "not_available: heldout_genes.py re-maps the held-out query from the "
    "clustered H5ADs, which §19 leaves unchanged for hybrid_matched, so its "
    "re-map would repeat proseg_hybrid's (no depth-matched H4)"
)
COMBINED_A1_REASON: Final = (
    "not_available: the combined mode is analysis 4 (reseg labels on "
    "proseg_hybrid counts; no RESOLVE or report of its own)"
)
PIPELINE_CHANGE: Final = (
    "Combined mode, not implemented in M8b: MAP and RESOLVE run on reseg only; "
    "a new FINALIZE input (e.g. annotation_label_segmentation = reseg for the "
    "proseg_hybrid branch) joins the reseg label table onto the proseg_hybrid "
    "table by cell id, so table_MOSAIK_proseg_hybrid carries reseg's ct_* "
    "columns and proseg_hybrid's counts; proseg_hybrid table cells without a "
    "reseg table cell get a new status (e.g. no_label_source_cell) and no "
    "label; the provenance, the report (its label and count sources differ) "
    "and the cross-platform JSD (reseg cells' coordinates vs proseg_hybrid's) "
    "record the label source; MENDER and cortical depth read the joined table."
)


@dataclass(frozen=True)
class GeneListEntry:
    """One gene of a literature gene list.

    Attributes:
        symbol: HGNC symbol.
        source: Citation (author, year, journal, DOI).
        evidence: What the source shows for the gene.
    """

    symbol: str
    source: str
    evidence: str


_BOULAY: Final = "Boulay et al. 2017, Cell Discov 3:17005, doi:10.1038/celldisc.2017.5"
_ARONOV: Final = (
    "Aronov et al. 2001, J Neurosci 21:6577-6587, "
    "doi:10.1523/JNEUROSCI.21-17-06577.2001"
)
_HOLZ: Final = (
    "Holz et al. 1996, J Neurosci 16:467-477, doi:10.1523/JNEUROSCI.16-02-00467.1996"
)
_HUTCHINSON: Final = (
    "Hutchinson et al. 2007, BMC Genomics 8:39, doi:10.1186/1471-2164-8-39"
)
_BAHAR_HALPERN: Final = (
    "Bahar Halpern et al. 2015, Cell Rep 13:2653-2662, doi:10.1016/j.celrep.2015.11.036"
)
# Fixed before the script's first real run (pre-registration §19). Genes are
# taken as the sources name them (human symbols of the rodent genes where the
# source is rodent); genes not on a dataset's panel are dropped there.
PROCESS_LOCALISED: Final[tuple[GeneListEntry, ...]] = (
    GeneListEntry("MAPT", _ARONOV, "tau mRNA targeted to the axon by its 3'UTR"),
    GeneListEntry("MAP2", _ARONOV, "MAP2 mRNA dendritic targeting signal"),
    GeneListEntry(
        "MOBP", _HOLZ, "MOBP81-A mRNA localised in oligodendrocyte processes"
    ),
    GeneListEntry("AQP4", _BOULAY, "mRNA in astrocyte perivascular processes"),
    GeneListEntry("GFAP", _BOULAY, "mRNA in astrocyte perivascular processes"),
    GeneListEntry("GJA1", _BOULAY, "endfeet transcriptome and endfeetome (Cx43)"),
    GeneListEntry("KCNJ10", _BOULAY, "endfeet transcriptome (Kir4.1)"),
    GeneListEntry("AGT", _BOULAY, "endfeetome; mRNA in endfeet by FISH"),
    GeneListEntry("PTPRZ1", _BOULAY, "endfeetome; mRNA in endfeet by FISH"),
    GeneListEntry("HEPACAM", _BOULAY, "endfeetome; mRNA in endfeet by FISH"),
    GeneListEntry("GPR37L1", _BOULAY, "endfeetome; mRNA in endfeet by FISH"),
)
NUCLEAR_RETAINED: Final[tuple[GeneListEntry, ...]] = (
    GeneListEntry("XIST", _HUTCHINSON, "nuclear-enriched polyadenylated RNA"),
    GeneListEntry("NEAT1", _HUTCHINSON, "nuclear-enriched polyadenylated RNA"),
    GeneListEntry("MALAT1", _HUTCHINSON, "NEAT2; nuclear speckle RNA"),
    GeneListEntry("MLXIPL", _BAHAR_HALPERN, "ChREBP mRNA nuclear-retained"),
    GeneListEntry("NLRP6", _BAHAR_HALPERN, "mRNA nuclear-retained"),
    GeneListEntry("GCK", _BAHAR_HALPERN, "glucokinase mRNA nuclear-retained"),
    GeneListEntry("GCGR", _BAHAR_HALPERN, "glucagon receptor mRNA nuclear-retained"),
)
GENE_LISTS: Final[dict[str, tuple[GeneListEntry, ...]]] = {
    "nuclear_retained": NUCLEAR_RETAINED,
    "process_localised": PROCESS_LOCALISED,
}
COVARIATES: Final[tuple[str, ...]] = (
    "expression_level",
    "gene_length",
    "probe_count",
)
LOSS_METRICS: Final[tuple[str, ...]] = ("L", "N_bg")
SHARE_COLUMNS: Final[tuple[str, ...]] = (
    "A_reseg",
    "A_hybrid",
    "A_reseg_same_cell",
    "N_bg",
    "N_bg_background",
    "N_bg_nontable",
    "L",
)


# --------------------------------------------------------------------------
# Small utilities


def role(pair: str) -> str:
    """Return ``development`` or ``held_out`` (pre-registration rule 3)."""
    return "development" if pair in DEVELOPMENT_PAIRS else "held_out"


def sample_id(pair: str, platform: str) -> str:
    """Return ``<pair>_<PLATFORM>``."""
    return f"{pair}_{platform.upper()}"


def file_sha256(path: Path) -> str:
    """Return a file's sha256."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(payload: Any, path: Path) -> Path:
    """Write JSON (sorted keys, NaN as null) and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=1, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return path


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_table(frame: pd.DataFrame, path: Path) -> Path:
    """Write a CSV deterministically (no index, 6 significant digits)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.6g", lineterminator="\n")
    return path


def is_number(value: Any) -> bool:
    """Whether a value is a finite int / float (not a bool)."""
    if isinstance(value, bool | np.bool_) or value is None:
        return False
    return isinstance(value, int | float | np.integer | np.floating) and math.isfinite(
        float(value)
    )


def control_mask(names: Sequence[str], platform: str) -> np.ndarray:
    """Return which features are controls (``panel.ControlRegistry``)."""
    registry = ControlRegistry()
    return np.array(
        [
            registry.control_reason(str(name), platform=platform.upper()) is not None
            for name in names
        ],
        dtype=bool,
    )


# --------------------------------------------------------------------------
# Locations


@dataclass(frozen=True)
class Locations:
    """Where the inputs are and where the outputs go.

    Attributes:
        runs_root: M8 stage B ``runs`` (``<pair>/<seg>/annotation_*``).
        results_root: The published results (stores, legacy clustered H5ADs,
            alignment).
        stage_b: The stage B evidence directory (traces, work).
        out: M8b's output root.
    """

    runs_root: Path
    results_root: Path
    stage_b: Path
    out: Path

    def run_dir(self: Locations, pair: str, segmentation: str) -> Path:
        """Return ``<runs>/<pair>/<segmentation>``."""
        return self.runs_root / pair / segmentation

    def matched_dir(self: Locations, pair: str) -> Path:
        """Return hybrid_matched's directory of a pair."""
        return self.out / "arms" / "hybrid_matched" / pair

    def prepared_dir(self: Locations, pair: str) -> Path:
        """Return hybrid_matched's thinned ``clustering_prepare_out``."""
        return self.matched_dir(pair) / "prepared" / "clustering_prepare_out"

    def resolve_dir(self: Locations, arm: str, pair: str) -> Path:
        """Return an arm's ``annotation_resolve_out``."""
        if arm == "hybrid_matched":
            return self.matched_dir(pair) / "resolve" / "annotation_resolve_out"
        return (
            self.run_dir(pair, ARM_SEGMENTATION[arm])
            / "annotation_resolve"
            / "annotation_resolve_out"
        )

    def map_dir(self: Locations, arm: str, pair: str) -> Path:
        """Return an arm's ``annotation_map_out``."""
        if arm == "hybrid_matched":
            return self.matched_dir(pair) / "map" / "annotation_map_out"
        return (
            self.run_dir(pair, ARM_SEGMENTATION[arm])
            / "annotation_map"
            / "annotation_map_out"
        )

    def panel_dir(self: Locations, arm: str, pair: str) -> Path:
        """Return an arm's ``annotation_panel_out`` (hybrid_matched: hybrid's)."""
        return (
            self.run_dir(pair, ARM_SEGMENTATION[arm])
            / "annotation_panel"
            / "annotation_panel_out"
        )

    def metrics_path(self: Locations, arm: str, pair: str) -> Path:
        """Return an arm's ``acceptance_metrics.json``."""
        if arm == "hybrid_matched":
            base = self.matched_dir(pair) / "report"
        else:
            base = self.run_dir(pair, ARM_SEGMENTATION[arm]) / "annotation_report"
        return base / "annotation_report_out" / "acceptance_metrics.json"

    def labels_path(self: Locations, arm: str, pair: str, platform: str) -> Path:
        """Return an arm's label table of one sample."""
        sid = sample_id(pair, platform)
        return (
            self.resolve_dir(arm, pair)
            / platform.lower()
            / f"{sid}_celltype_labels.parquet"
        )

    def summary_path(self: Locations, arm: str, pair: str) -> Path:
        """Return an arm's ``<pair>_resolve_summary.json``."""
        return self.resolve_dir(arm, pair) / f"{pair}_resolve_summary.json"

    def clustered_path(self: Locations, arm: str, pair: str, platform: str) -> Path:
        """Return the map_first clustered H5AD of an arm's segmentation."""
        sid = sample_id(pair, platform)
        return (
            self.run_dir(pair, ARM_SEGMENTATION[arm])
            / "clustering_squidpy_m8"
            / "clustering_squidpy_out"
            / platform.lower()
            / f"{sid}_clustered.h5ad"
        )

    def legacy_clustered_path(
        self: Locations, segmentation: str, pair: str, platform: str
    ) -> Path:
        """Return the published legacy clustered H5AD."""
        sid = sample_id(pair, platform)
        return (
            self.results_root
            / pair
            / segmentation
            / "clustering_squidpy"
            / "clustering_squidpy_out"
            / platform.lower()
            / f"{sid}_clustered.h5ad"
        )

    def store_path(self: Locations, pair: str, platform: str) -> Path:
        """Return the published SpatialData store of a sample."""
        return (
            self.results_root
            / pair
            / platform.lower()
            / "latest"
            / "latest_spatialdata.zarr"
        )

    def alignment_dir(self: Locations, pair: str) -> Path:
        """Return the pair's ``alignment/align_out``."""
        return self.results_root / pair / "alignment" / "align_out"

    def farm(self: Locations, arm: str) -> Path:
        """Return an arm's symlink farm."""
        return self.out / "farms" / arm

    def acceptance_dir(self: Locations, arm: str) -> Path:
        """Return an arm's acceptance-script outputs."""
        return self.out / "acceptance" / arm

    def transcripts_cache(self: Locations, pair: str, platform: str) -> Path:
        """Return analysis 2's tally of one sample."""
        return self.out / "cache" / "transcripts" / f"{sample_id(pair, platform)}"

    def controls_cache(self: Locations, pair: str, platform: str) -> Path:
        """Return the control features' tally of one sample (post hoc)."""
        return self.out / "cache" / "controls" / f"{sample_id(pair, platform)}"

    def result_dir(self: Locations) -> Path:
        """Return ``<out>/segmentation_comparison``."""
        return self.out / "segmentation_comparison"


def locations_from(args: argparse.Namespace) -> Locations:
    """Return the locations of the parsed arguments."""
    return Locations(
        runs_root=Path(args.runs_root),
        results_root=Path(args.results_root),
        stage_b=Path(args.stage_b),
        out=Path(args.out),
    )


def read_labels(path: Path, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Read a label table (``cell_id`` as str)."""
    import pyarrow.parquet as pq

    frame = pq.read_table(path, columns=None if columns is None else list(columns))
    labels = frame.to_pandas()
    labels["cell_id"] = labels["cell_id"].astype(str)
    return labels


def read_json(path: Path) -> Any:
    """Read a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Stage B1 tasks (hybrid_matched)


def b1_traces(stage_b: Path) -> list[Path]:
    """Return B1's trace files, earlier attempts first."""
    attempts = sorted(
        (stage_b / "nextflow").glob("trace_B1.attempt*.tsv"),
        key=lambda path: [int(part) for part in re.findall(r"\d+", path.name)],
    )
    final = stage_b / "nextflow" / "trace_B1.tsv"
    return [*attempts, *([final] if final.is_file() else [])]


def find_task(traces: Sequence[Path], process: str, tag: str) -> Path:
    """Return the work directory of a task's last COMPLETED record.

    Args:
        traces: Nextflow trace TSVs (later files win).
        process: The process's simple name (e.g. ``ANNOTATION_REPORT``).
        tag: The task tag (``<pair>:<segmentation>``).

    Returns:
        The work directory.

    Raises:
        SystemExit: If no completed task with an existing directory is found.
    """
    found: Path | None = None
    for trace in traces:
        with trace.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if (
                    row["process"].rsplit(":", 1)[-1] == process
                    and row["tag"] == tag
                    and row["status"] == "COMPLETED"
                ):
                    found = Path(row["workdir"])
    if found is None or not found.is_dir():
        raise SystemExit(f"no completed {process} ({tag}) task directory in {traces}")
    return found


def stage_task(
    source: Path, out: Path, replace: Mapping[str, Path]
) -> list[dict[str, str]]:
    """Recreate a task directory's staged inputs (symlinks) under ``out``.

    Args:
        source: The Nextflow work directory.
        out: The new task directory (created by the caller).
        replace: Staged path (relative) to its replacement target.

    Returns:
        One record per staged link (path and target).

    Raises:
        SystemExit: If a path to replace is not staged in ``source``.
    """
    staged: list[dict[str, str]] = []
    for dirpath, dirnames, filenames in os.walk(source, followlinks=False):
        for name in sorted(dirnames) + sorted(filenames):
            full = Path(dirpath) / name
            if not full.is_symlink():
                continue
            relative = full.relative_to(source).as_posix()
            target = Path(os.readlink(full))
            if not target.is_absolute():
                target = (full.parent / target).resolve()
            target = Path(replace.get(relative, target))
            link = out / relative
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(target)
            staged.append({"path": relative, "target": str(target)})
        dirnames[:] = [d for d in dirnames if not (Path(dirpath) / d).is_symlink()]
    missing = sorted(set(replace) - {item["path"] for item in staged})
    if missing:
        raise SystemExit(f"staged inputs to replace not found in {source}: {missing}")
    return staged


def rewrite_pythonpath(script: str, export: Path) -> tuple[str, str, str]:
    """Point a task script's ``PYTHONPATH`` export at ``<export>/src``.

    Args:
        script: The task's ``.command.sh``.
        export: The code export (git archive of the committed HEAD).

    Returns:
        ``(new script, old path, new path)``.

    Raises:
        SystemExit: Unless the script has exactly one such export line.
    """
    matches = list(PYTHONPATH_LINE.finditer(script))
    if len(matches) != 1:
        raise SystemExit(f"expected one PYTHONPATH export line, found {len(matches)}")
    match = matches[0]
    new_path = str(export / "src")
    updated = script[: match.start(1)] + new_path + script[match.end(1) :]
    return updated, match.group(1), new_path


def source_tree_differences(first: Path, second: Path) -> list[dict[str, str]]:
    """Return the ``.py`` files that differ between two ``src`` trees.

    Args:
        first: One ``src`` directory (the B1 task's export).
        second: The other (the code export this step runs).

    Returns:
        One record per file only in one tree or with another sha256.
    """

    def files(root: Path) -> dict[str, str]:
        return {
            path.relative_to(root).as_posix(): file_sha256(path)
            for path in sorted(root.rglob("*.py"))
            if "__pycache__" not in path.parts
        }

    a, b = files(first.resolve()), files(second.resolve())
    out = []
    for name in sorted(set(a) | set(b)):
        if a.get(name) != b.get(name):
            state = (
                "only_second"
                if name not in a
                else "only_first"
                if name not in b
                else "differs"
            )
            out.append({"file": name, "state": state})
    return out


def task_replacements(locations: Locations, pair: str, process: str) -> dict[str, Path]:
    """Return the staged inputs ``matched-task`` replaces for one process."""
    prepared = locations.prepared_dir(pair)
    map_out = locations.map_dir("hybrid_matched", pair)
    resolve_out = locations.resolve_dir("hybrid_matched", pair)
    if process == "MAP":
        return {"map_inputs/clustering_prepare_out": prepared}
    if process == "RESOLVE":
        return {
            "resolve_inputs/clustering_prepare_out": prepared,
            "resolve_inputs/annotation_map_out": map_out,
        }
    if process == "REPORT":
        return {
            "report_inputs/annotation_resolve_out": resolve_out,
            "report_inputs/annotation_map_out": map_out,
        }
    raise SystemExit(f"unknown process {process!r}")


# --------------------------------------------------------------------------
# hybrid_matched: thinning the prepared counts


def thin_prepared_h5ad(
    source: Path,
    destination: Path,
    *,
    platform: str,
    reseg_totals: pd.Series,
    hybrid_totals: pd.Series,
    seed: int = sc.THINNING_SEED,
) -> dict[str, Any]:
    """Write a prepared H5AD whose gene counts are thinned to the reseg totals.

    Every row is thinned to its reseg cell's total (gene features, controls
    removed as MAP removes them); an id reseg lacks gets target 0. Control
    features keep their counts; the obs QC columns are recomputed from the
    new ``X``. The gene totals of the source must equal proseg_hybrid's
    label-table ``total_counts`` (else the gene set differs from MAP's).

    Args:
        source: The B1 PREPARE H5AD (proseg_hybrid).
        destination: The new H5AD.
        platform: ``MERSCOPE`` or ``XENIUM``.
        reseg_totals: reseg ``total_counts`` by cell id (every object).
        hybrid_totals: proseg_hybrid ``total_counts`` by cell id.
        seed: The run seed (0).

    Returns:
        The thinning record.

    Raises:
        SystemExit: If a check fails.
    """
    import anndata as ad

    adata = ad.read_h5ad(source)
    ids = pd.Index(adata.obs_names.astype(str))
    if ids.has_duplicates:
        raise SystemExit(f"{source}: duplicate cell ids")
    matrix = sparse.csr_matrix(adata.X)
    controls = control_mask(list(adata.var_names.astype(str)), platform)
    genes = np.flatnonzero(~controls)
    gene_counts = matrix[:, genes].tocsr()
    before = np.asarray(gene_counts.sum(axis=1)).ravel().astype(np.int64)
    expected = hybrid_totals.reindex(ids)
    if expected.isna().any() or not np.array_equal(before, expected.to_numpy(np.int64)):
        raise SystemExit(
            f"{source}: gene totals differ from the proseg_hybrid label table "
            "(the control rule does not match MAP's)"
        )
    targets = reseg_totals.reindex(ids).fillna(0).to_numpy(np.int64)
    result = sc.thin_to_targets(gene_counts, list(ids), targets, seed=seed)
    expected_after = np.minimum(before, targets)
    if not np.array_equal(result.totals_after, expected_after):
        raise SystemExit(f"{source}: thinned totals differ from min(total, target)")
    difference = (gene_counts - result.counts).tocsr()
    if difference.nnz and difference.data.min() < 0:
        raise SystemExit(f"{source}: a thinned count exceeds the original")
    new_x = merge_columns(
        matrix.shape,
        (genes, result.counts),
        (np.flatnonzero(controls), matrix[:, np.flatnonzero(controls)]),
    ).astype(matrix.dtype)
    new_x.eliminate_zeros()
    new_x.sort_indices()
    table_before = before >= TABLE_MIN_COUNTS
    table_reseg = targets >= TABLE_MIN_COUNTS
    table_after = result.totals_after >= TABLE_MIN_COUNTS
    if not np.array_equal(table_after, table_before & table_reseg):
        raise SystemExit(f"{source}: thinned table cells != hybrid table & reseg table")
    adata.X = new_x
    _recompute_qc(adata, controls)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    adata.write_h5ad(partial)
    from merxen.annotation import pipeline

    names, _, written, _ = pipeline.read_h5ad_counts(partial, "prepared")
    if list(names) != list(ids) or (written != new_x).nnz:
        raise SystemExit(f"{partial}: the written counts differ from the thinned ones")
    partial.rename(destination)
    return {
        "source": str(source),
        "source_sha256": file_sha256(source),
        "destination": str(destination),
        "destination_sha256": file_sha256(destination),
        "platform": platform,
        "seed": int(seed),
        "numpy_version": np.__version__,
        "n_objects": int(len(ids)),
        "n_gene_features": int(len(genes)),
        "n_control_features": int(controls.sum()),
        "n_thinned": int(result.thinned.sum()),
        "n_kept_at_or_below_target": int((~result.thinned).sum()),
        "n_without_reseg_object": int((~ids.isin(reseg_totals.index)).sum()),
        "n_target_zero": int((targets == 0).sum()),
        "n_table_hybrid": int(table_before.sum()),
        "n_table_reseg": int(table_reseg.sum()),
        "n_table_matched": int(table_after.sum()),
        "gene_counts_before": int(before.sum()),
        "gene_counts_after": int(result.totals_after.sum()),
        "matched_table_counts_before": int(before[table_after].sum()),
        "matched_table_counts_after": int(result.totals_after[table_after].sum()),
    }


def merge_columns(
    shape: tuple[int, int], *parts: tuple[np.ndarray, sparse.spmatrix]
) -> sparse.csr_matrix:
    """Assemble a matrix from column blocks placed at the given columns.

    Args:
        shape: The result's shape.
        parts: ``(columns, block)``: ``block``'s column ``j`` goes to
            ``columns[j]``; every result column must be covered once.

    Returns:
        The CSR matrix.
    """
    columns = np.concatenate([np.asarray(cols, dtype=np.int64) for cols, _ in parts])
    if sorted(columns.tolist()) != list(range(shape[1])):
        raise ValueError("the blocks must cover every column exactly once")
    stacked = sparse.hstack([sparse.csr_matrix(block) for _, block in parts]).tocsc()
    order = np.argsort(columns)
    return sparse.csr_matrix(stacked[:, order])


def _recompute_qc(adata: Any, controls: np.ndarray) -> None:
    """Recompute the obs QC columns the prepared H5AD carries from ``X``."""
    matrix = sparse.csr_matrix(adata.X)
    total = np.asarray(matrix.sum(axis=1)).ravel()
    n_genes = np.diff(matrix.indptr)
    obs = adata.obs
    if "total_counts" in obs:
        obs["total_counts"] = total.astype(obs["total_counts"].dtype)
    if "log1p_total_counts" in obs:
        obs["log1p_total_counts"] = np.log1p(total)
    if "n_genes_by_counts" in obs:
        obs["n_genes_by_counts"] = n_genes.astype(obs["n_genes_by_counts"].dtype)
    if "log1p_n_genes_by_counts" in obs:
        obs["log1p_n_genes_by_counts"] = np.log1p(n_genes)
    if "pct_control_counts" in obs and "control_counts" in obs:
        control = obs["control_counts"].to_numpy(np.float64)
        obs["pct_control_counts"] = np.divide(
            100.0 * control,
            total,
            out=np.full(len(total), math.nan),
            where=total > 0,
        )


def command_matched_prepare(args: argparse.Namespace) -> int:
    """Write hybrid_matched's thinned prepared counts of one pair."""
    locations = locations_from(args)
    pair = args.pair
    traces = b1_traces(locations.stage_b)
    map_task = find_task(traces, PROCESSES["MAP"], f"{pair}:{FARM_SEGMENTATION}")
    staged = map_task / "map_inputs" / "clustering_prepare_out"
    source_dir = Path(os.readlink(staged)) if staged.is_symlink() else staged
    manifest = read_json(source_dir / "manifest.json")
    out_dir = locations.prepared_dir(pair)
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists (never overwritten)")
    out_dir.mkdir(parents=True)
    shutil.copyfile(source_dir / "manifest.json", out_dir / "manifest.json")
    records = []
    for sid, relative in sorted(manifest["samples"].items()):
        platform = sid.rsplit("_", 1)[-1]
        reseg = read_labels(
            locations.labels_path("reseg", pair, platform), ["cell_id", "total_counts"]
        )
        hybrid = read_labels(
            locations.labels_path("hybrid", pair, platform),
            ["cell_id", "total_counts"],
        )
        record = thin_prepared_h5ad(
            source_dir / relative,
            out_dir / relative,
            platform=platform,
            reseg_totals=reseg.set_index("cell_id")["total_counts"],
            hybrid_totals=hybrid.set_index("cell_id")["total_counts"],
            seed=args.seed,
        )
        records.append({"sample_id": sid, **record})
        logger.info(
            "%s: %s", sid, {k: record[k] for k in ("n_thinned", "n_table_matched")}
        )
    write_json(
        {
            "pair": pair,
            "map_task": str(map_task),
            "prepared_source": str(source_dir),
            "samples": records,
        },
        out_dir.parent / "thinning_record.json",
    )
    return 0


def command_matched_task(args: argparse.Namespace) -> int:
    """Re-run one of B1's tasks of a pair for hybrid_matched."""
    locations = locations_from(args)
    pair, process = args.pair, args.process
    source = find_task(
        b1_traces(locations.stage_b), PROCESSES[process], f"{pair}:{FARM_SEGMENTATION}"
    )
    task_dir = locations.matched_dir(pair) / process.lower()
    if task_dir.exists():
        raise SystemExit(f"{task_dir} exists (never overwritten)")
    replace = task_replacements(locations, pair, process)
    for target in replace.values():
        if not target.exists():
            raise SystemExit(f"{target} does not exist (run the earlier step first)")
    task_dir.mkdir(parents=True)
    staged = stage_task(source, task_dir, replace)
    original = (source / ".command.sh").read_text(encoding="utf-8")
    script, old_path, new_path = rewrite_pythonpath(original, Path(args.export))
    (task_dir / ".command.sh").write_text(script, encoding="utf-8")
    record: dict[str, Any] = {
        "pair": pair,
        "process": PROCESSES[process],
        "source_workdir": str(source),
        "replaced": {key: str(value) for key, value in replace.items()},
        "staged": staged,
        "pythonpath_before": old_path,
        "pythonpath_after": new_path,
        "script_sha256_source": hashlib.sha256(original.encode()).hexdigest(),
        "script_sha256_run": hashlib.sha256(script.encode()).hexdigest(),
        "export": str(args.export),
        "src_differences_vs_task_code": source_tree_differences(
            Path(old_path), Path(new_path)
        ),
    }
    if not args.dry_run:
        runtime = locations.out / "runtime_cache"
        env = {
            **os.environ,
            "PATH": f"{args.env_bin}:{os.environ.get('PATH', '/usr/bin:/bin')}",
            "PYTHONPATH": str(Path(args.export) / "src"),
            "CUDA_VISIBLE_DEVICES": "",
            "MPLCONFIGDIR": str(runtime / "mpl"),
            "NUMBA_CACHE_DIR": str(runtime / "numba"),
        }
        started = time.time()
        with (
            (task_dir / ".command.out").open("w") as out,
            (task_dir / ".command.err").open("w") as err,
        ):
            result = subprocess.run(
                [
                    "/usr/bin/time",
                    "-v",
                    "-o",
                    str(task_dir / ".command.time"),
                    "bash",
                    ".command.sh",
                ],
                cwd=task_dir,
                env=env,
                stdout=out,
                stderr=err,
                check=False,
            )
        record["exit"] = result.returncode
        record["wall_s"] = round(time.time() - started, 3)
        timing = (task_dir / ".command.time").read_text(encoding="utf-8")
        rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", timing)
        record["max_rss_kb"] = int(rss.group(1)) if rss else None
    write_json(record, task_dir / "matched_task.json")
    logger.info(
        "%s %s: exit %s, %s s",
        pair,
        process,
        record.get("exit"),
        record.get("wall_s"),
    )
    return int(record.get("exit", 0))


# --------------------------------------------------------------------------
# Farms and the acceptance scripts


def farm_qc_summary(qc_summary: pd.DataFrame, segmentation: str) -> pd.DataFrame:
    """Return a qc_summary whose proseg_hybrid rows are ``segmentation``'s."""
    keep = qc_summary[qc_summary["seg"] != FARM_SEGMENTATION]
    swapped = qc_summary[qc_summary["seg"] == segmentation].copy()
    swapped["seg"] = FARM_SEGMENTATION
    if segmentation == FARM_SEGMENTATION:
        return qc_summary.copy()
    return pd.concat([keep, swapped], ignore_index=True)


def build_farm(
    locations: Locations,
    arm: str,
    qc_summary: pd.DataFrame,
    pairs: Sequence[str] = PAIRS,
) -> dict[str, Any]:
    """Link an arm's outputs where the acceptance scripts read proseg_hybrid.

    Args:
        locations: Inputs and outputs.
        arm: The arm.
        qc_summary: ``research/lowcount/qc_summary.csv``.
        pairs: Pairs.

    Returns:
        The farm record (every link and its target).

    Raises:
        SystemExit: If the farm exists or an input is missing.
    """
    farm = locations.farm(arm)
    if farm.exists():
        raise SystemExit(f"{farm} exists (never overwritten)")
    segmentation = ARM_SEGMENTATION[arm]
    links: list[dict[str, str]] = []

    def link(path: Path, target: Path) -> None:
        if not target.exists():
            raise SystemExit(f"farm input missing: {target}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(target)
        links.append({"path": str(path.relative_to(farm)), "target": str(target)})

    for pair in pairs:
        link(
            farm / "resolve" / pair / FARM_SEGMENTATION,
            locations.resolve_dir(arm, pair),
        )
        map_dir = locations.map_dir(arm, pair)
        runs = farm / "runs" / pair / FARM_SEGMENTATION
        for item in (
            "map_manifest.json",
            "merscope",
            "xenium",
            "logs",
            "annotation_config.json",
        ):
            if (map_dir / item).exists():
                link(runs / item, map_dir / item)
        link(runs / "panel", locations.panel_dir(arm, pair))
        for platform in PLATFORMS:
            sid = sample_id(pair, platform)
            link(
                farm
                / "results"
                / pair
                / FARM_SEGMENTATION
                / "clustering_squidpy"
                / "clustering_squidpy_out"
                / platform.lower()
                / f"{sid}_clustered.h5ad",
                locations.legacy_clustered_path(segmentation, pair, platform),
            )
    summary = farm_qc_summary(qc_summary, segmentation)
    summary.to_csv(farm / "qc_summary.csv", index=False)
    record = {"arm": arm, "segmentation": segmentation, "links": links}
    write_json(record, farm / "farm.json")
    return record


def _pairs(args: argparse.Namespace) -> list[str]:
    """The pairs a step covers (``--pairs``; all four by default)."""
    pairs = [item for item in str(getattr(args, "pairs", "") or "").split(",") if item]
    unknown = sorted(set(pairs) - set(PAIRS))
    if unknown:
        raise SystemExit(f"unknown pairs: {unknown}")
    return pairs or list(PAIRS)


def command_farms(args: argparse.Namespace) -> int:
    """Build the symlink farm of every arm."""
    locations = locations_from(args)
    qc_summary = pd.read_csv(args.qc_summary)
    for arm in [item for item in args.arms.split(",") if item]:
        record = build_farm(locations, arm, qc_summary, _pairs(args))
        logger.info("farm %s: %d links", arm, len(record["links"]))
    return 0


def acceptance_commands(
    locations: Locations, arm: str, args: argparse.Namespace
) -> list[tuple[str, list[str]]]:
    """Return the acceptance-script commands of one arm (in order)."""
    farm = locations.farm(arm)
    out = locations.acceptance_dir(arm)
    scripts = Path(args.export) / "scripts" / "acceptance"
    python = sys.executable
    common = ["--pairs", ",".join(_pairs(args))]
    commands: list[tuple[str, list[str]]] = []
    if arm != "hybrid_matched":
        commands.append(
            (
                "heldout_genes",
                [
                    python,
                    str(scripts / "heldout_genes.py"),
                    "--runs-root",
                    str(farm / "runs"),
                    "--results-root",
                    str(farm / "results"),
                    "--out",
                    str(out / "heldout"),
                    "--work-dir",
                    str(out / "tmp" / "heldout_work"),
                    "--variants",
                    "set_a",
                    "--n-processors",
                    str(args.n_processors),
                    "--gene-id-fallback-csv",
                    str(args.gene_id_fallback_csv),
                    *common,
                ],
            )
        )
    criteria = [
        python,
        str(scripts / "resolve_criteria.py"),
        "--resolve-root",
        str(farm / "resolve"),
        "--runs-root",
        str(farm / "runs"),
        "--results-root",
        str(farm / "results"),
        "--segmentations",
        FARM_SEGMENTATION,
        "--out",
        str(out / "criteria"),
        "--gene-id-fallback-csv",
        str(args.gene_id_fallback_csv),
        *common,
    ]
    if arm == "hybrid_matched":
        criteria.append("--skip-h4")
    else:
        criteria += [
            "--heldout-root",
            str(out / "heldout"),
            "--store",
            str(args.store),
            "--qc-summary",
            str(farm / "qc_summary.csv"),
        ]
    commands.append(("resolve_criteria", criteria))
    commands.append(
        (
            "marker_referee",
            [
                python,
                str(scripts / "marker_referee.py"),
                "--resolve-root",
                str(farm / "resolve"),
                "--results-root",
                str(farm / "results"),
                "--segmentations",
                FARM_SEGMENTATION,
                "--out",
                str(out / "referee"),
                "--check-against",
                str(out / "criteria" / "referee.csv"),
                *common,
            ],
        )
    )
    return commands


def command_acceptance(args: argparse.Namespace) -> int:
    """Run the acceptance scripts on one arm's farm, one after the other."""
    locations = locations_from(args)
    arm = args.arm
    out = locations.acceptance_dir(arm)
    if out.exists():
        raise SystemExit(f"{out} exists (never overwritten)")
    (out / "logs").mkdir(parents=True)
    (out / "tmp").mkdir()
    env = {
        **os.environ,
        "PYTHONPATH": str(Path(args.export) / "src"),
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMBA_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "TMPDIR": str(out / "tmp"),
        "MPLCONFIGDIR": str(locations.out / "runtime_cache" / "mpl"),
        "NUMBA_CACHE_DIR": str(locations.out / "runtime_cache" / "numba"),
    }
    record: dict[str, Any] = {"arm": arm, "steps": []}
    status = 0
    for name, command in acceptance_commands(locations, arm, args):
        started = time.time()
        with (out / "logs" / f"{name}.log").open("w") as log:
            result = subprocess.run(
                command, env=env, stdout=log, stderr=subprocess.STDOUT, check=False
            )
        step = {
            "name": name,
            "command": command,
            "exit": result.returncode,
            "wall_s": round(time.time() - started, 1),
        }
        record["steps"].append(step)
        logger.info("%s %s: exit %d", arm, name, result.returncode)
        if result.returncode != 0:
            status = result.returncode
            if name != "marker_referee":
                break
    write_json(record, out / "acceptance_run.json")
    return status


# --------------------------------------------------------------------------
# Analysis 2: transcript tallies (one sample)


@dataclass(frozen=True)
class TranscriptInputs:
    """One sample's transcript columns after the decoding filter.

    Attributes:
        x: Native x per transcript.
        y: Native y.
        gene: Gene symbol code per transcript (into ``genes``).
        genes: Gene symbols.
        assignment: reseg cell id per transcript (``valid_reseg`` marks set).
        valid_reseg: Whether ``assignment`` is set.
        background: ProSeg's ``background``.
        hybrid_assignment: proseg_hybrid cell id.
        valid_hybrid: Whether ``hybrid_assignment`` is set.
        hybrid_background: ``hybrid_background``.
        record: The filter counts.
    """

    x: np.ndarray
    y: np.ndarray
    gene: np.ndarray
    genes: list[str]
    assignment: np.ndarray
    valid_reseg: np.ndarray
    background: np.ndarray
    hybrid_assignment: np.ndarray
    valid_hybrid: np.ndarray
    hybrid_background: np.ndarray
    record: dict[str, Any]


TRANSCRIPT_COLUMNS: Final[tuple[str, ...]] = (
    "x",
    "y",
    "gene",
    "qv",
    "assignment",
    "background",
    "hybrid_assignment",
    "hybrid_background",
)


def _id_column(table: Any, name: str) -> tuple[np.ndarray, np.ndarray]:
    import pyarrow.compute as pc

    column = table.column(name)
    valid = pc.is_valid(column).to_numpy(zero_copy_only=False)
    values = pc.fill_null(column, 0).to_numpy(zero_copy_only=False)
    return np.asarray(values, dtype=np.uint64), np.asarray(valid, dtype=bool)


FEATURE_SETS: Final[tuple[str, ...]] = ("genes", "controls")


def load_transcripts(
    store: Path, platform: str, min_qv: float | None, *, features: str = "genes"
) -> TranscriptInputs:
    """Read a store's transcripts and apply the decoding filter.

    Args:
        store: ``latest_spatialdata.zarr``.
        platform: ``MERSCOPE`` or ``XENIUM``.
        min_qv: Minimum qv (Xenium); ``None``: no qv filter.
        features: ``genes`` (analysis 2: control features removed) or
            ``controls`` (post hoc: only the control features kept, e.g.
            MERSCOPE's blank barcodes; ``genes`` then lists those).

    Returns:
        The kept transcripts' columns and the filter record
        (``n_control_removed`` counts the control features either way).
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    if features not in FEATURE_SETS:
        raise ValueError(f"features must be one of {FEATURE_SETS}")
    parts = sorted(
        (store / "points" / "transcripts" / "points.parquet").glob("*.parquet")
    )
    if not parts:
        raise SystemExit(f"{store}: no points/transcripts parquet parts")
    table = pa.concat_tables(
        [pq.read_table(part, columns=list(TRANSCRIPT_COLUMNS)) for part in parts]
    )
    n_total = table.num_rows
    gene_column = pc.cast(table.column("gene"), pa.string())
    names = sorted(set(pc.unique(gene_column).to_pylist()) - {None})
    controls = set(np.asarray(names)[control_mask(names, platform)].tolist())
    keep = pc.invert(
        pc.is_in(gene_column, value_set=pa.array(sorted(controls), pa.string()))
    )
    n_control = n_total - int(pc.sum(keep).as_py() or 0)
    if features == "controls":
        keep = pc.is_in(gene_column, value_set=pa.array(sorted(controls), pa.string()))
    if min_qv is not None:
        passes = pc.greater_equal(
            table.column("qv"), pa.scalar(float(min_qv), pa.float32())
        )
        passes = pc.fill_null(passes, False)
        n_qv = int(pc.sum(pc.and_(keep, pc.invert(passes))).as_py() or 0)
        keep = pc.and_(keep, passes)
    else:
        n_qv = 0
    table = table.filter(keep)
    gene_column = pc.cast(table.column("gene"), pa.string())
    genes = sorted(set(pc.unique(gene_column).to_pylist()) - {None})
    codes = pc.index_in(gene_column, value_set=pa.array(genes, pa.string()))
    gene = np.asarray(codes.to_numpy(zero_copy_only=False), dtype=np.int64)
    assignment, valid_reseg = _id_column(table, "assignment")
    hybrid_assignment, valid_hybrid = _id_column(table, "hybrid_assignment")
    background = np.asarray(
        pc.fill_null(table.column("background"), True).to_numpy(zero_copy_only=False),
        dtype=bool,
    )
    hybrid_background = np.asarray(
        pc.fill_null(table.column("hybrid_background"), True).to_numpy(
            zero_copy_only=False
        ),
        dtype=bool,
    )
    record = {
        "features": features,
        "n_transcripts": int(n_total),
        "n_control_removed": int(n_control),
        "n_below_min_qv_removed": int(n_qv),
        "min_qv": min_qv,
        "n_decoded": int(table.num_rows),
        "n_genes": len(genes),
        "control_features_removed": sorted(controls),
        "parts": [
            {
                "path": str(part),
                "size": part.stat().st_size,
                "sha256": file_sha256(part),
            }
            for part in parts
        ],
    }
    return TranscriptInputs(
        x=np.asarray(table.column("x").to_numpy(), dtype=np.float64),
        y=np.asarray(table.column("y").to_numpy(), dtype=np.float64),
        gene=gene,
        genes=genes,
        assignment=assignment,
        valid_reseg=valid_reseg,
        background=background,
        hybrid_assignment=hybrid_assignment,
        valid_hybrid=valid_hybrid,
        hybrid_background=hybrid_background,
        record=record,
    )


def table_ids_uint64(labels: pd.DataFrame) -> np.ndarray:
    """Return the table cells' ids as sorted ``uint64`` (ids must be integers)."""
    table = labels[labels["in_table"].to_numpy(bool)]
    values = pd.to_numeric(table["cell_id"], errors="raise")
    if (values < 0).any():
        raise SystemExit("cell ids must be non-negative integers")
    return np.sort(values.to_numpy(np.uint64))


def assigned_to(
    ids: np.ndarray, valid: np.ndarray, background: np.ndarray, table: np.ndarray
) -> np.ndarray:
    """Whether each transcript is assigned (set, not background) to a table cell."""
    position = np.searchsorted(table, ids)
    position = np.minimum(position, max(len(table) - 1, 0))
    in_table = (table[position] == ids) if len(table) else np.zeros(len(ids), bool)
    return valid & ~background & in_table


NO_CELL: Final = np.uint64(np.iinfo(np.uint64).max)


def holder_ids(holders: np.ndarray) -> np.ndarray:
    """Return nucleus holders as ``uint64`` (``NO_CELL`` for none)."""
    out = np.full(len(holders), NO_CELL, dtype=np.uint64)
    for index, value in enumerate(holders):
        if value is not None:
            out[index] = np.uint64(int(value))
    return out


def label_codes(
    cells: np.ndarray, label_of: Mapping[int, str], classes: Sequence[str]
) -> np.ndarray:
    """Return the class code (into ``classes``) of each transcript's cell (-1)."""
    index = {name: position for position, name in enumerate(classes)}
    keys = np.fromiter(label_of.keys(), dtype=np.uint64, count=len(label_of))
    values = np.array([index[label_of[int(key)]] for key in keys], dtype=np.int64)
    order = np.argsort(keys)
    keys, values = keys[order], values[order]
    out = np.full(len(cells), -1, dtype=np.int64)
    if not len(keys):
        return out
    position = np.minimum(np.searchsorted(keys, cells), len(keys) - 1)
    hit = keys[position] == cells
    out[hit] = values[position[hit]]
    return out


def confident_labels(labels: pd.DataFrame, level: str) -> dict[int, str]:
    """Return table cell id -> confident name at a level."""
    status = labels[f"ct_{level}_status"].astype(str).to_numpy()
    keep = labels["in_table"].to_numpy(bool) & (status == CONFIDENT)
    names = labels[f"ct_{level}_name"].to_numpy(object)[keep]
    ids = pd.to_numeric(labels["cell_id"].to_numpy()[keep]).astype(np.uint64)
    return {int(i): str(n) for i, n in zip(ids, names, strict=True)}


def glia_or_neuron(broad: str) -> str:
    """Return ``neurons`` or ``glia`` of a broad class (E2's classes)."""
    from merxen.annotation.shadow import E2_CLASS_OF_HUMAN_BROAD

    return "neurons" if E2_CLASS_OF_HUMAN_BROAD.get(broad) == "N" else "glia"


def transcript_tally_frames(
    inputs: TranscriptInputs,
    *,
    nucleus_index: np.ndarray,
    holders: np.ndarray,
    reseg_table: np.ndarray,
    hybrid_table: np.ndarray,
    groupings: Mapping[str, tuple[Mapping[int, str], Sequence[str]]],
) -> pd.DataFrame:
    """Tally one sample's transcripts per gene, overall and per label group.

    Args:
        inputs: The decoded transcripts.
        nucleus_index: Nucleus per transcript (``-1``: none).
        holders: Holder cell id per nucleus (``uint64``, ``NO_CELL``).
        reseg_table: Sorted reseg table cell ids.
        hybrid_table: Sorted proseg_hybrid table cell ids.
        groupings: Level name to (cell id -> label, label order); a
            transcript's label is that of the proseg_hybrid cell it belongs
            to (``segmentation_compare.transcript_cells``).

    Returns:
        Long frame: ``level``, ``group``, ``gene`` and ``TALLY_FIELDS``.
    """
    reseg = assigned_to(
        inputs.assignment, inputs.valid_reseg, inputs.background, reseg_table
    )
    hybrid = assigned_to(
        inputs.hybrid_assignment,
        inputs.valid_hybrid,
        inputs.hybrid_background,
        hybrid_table,
    )
    hybrid_cell = np.where(hybrid, inputs.hybrid_assignment, NO_CELL)
    cell = sc.transcript_cells(nucleus_index, holders, hybrid_cell)
    same = reseg & (cell != NO_CELL) & (inputs.assignment == cell)
    in_nucleus = nucleus_index >= 0
    common = {
        "assigned_reseg": reseg,
        "assigned_hybrid": hybrid,
        "assigned_reseg_same_cell": same,
        "in_nucleus": in_nucleus,
        "background_reseg": inputs.background | ~inputs.valid_reseg,
    }
    n_genes = len(inputs.genes)
    frames = []
    overall = sc.loss_tally(inputs.gene, n_genes, **common)
    frames.append(_tally_frame(overall, "all", ["all"], inputs.genes))
    for level, (label_of, classes) in groupings.items():
        codes = label_codes(cell, label_of, classes)
        tally = sc.loss_tally(
            inputs.gene, n_genes, group_codes=codes, n_groups=len(classes), **common
        )
        frames.append(_tally_frame(tally, level, list(classes), inputs.genes))
    return pd.concat(frames, ignore_index=True)


def _tally_frame(
    tally: np.ndarray, level: str, groups: Sequence[str], genes: Sequence[str]
) -> pd.DataFrame:
    n_groups, n_genes, _ = tally.shape
    frame = pd.DataFrame(
        tally.reshape(n_groups * n_genes, len(sc.TALLY_FIELDS)),
        columns=list(sc.TALLY_FIELDS),
    )
    frame.insert(0, "gene", np.tile(np.asarray(genes, dtype=object), n_groups))
    frame.insert(0, "group", np.repeat(np.asarray(groups, dtype=object), n_genes))
    frame.insert(0, "level", level)
    return frame


def read_shapes(store: Path, key: str) -> Any:
    """Read a store's shapes element (GeoDataFrame, native frame)."""
    import geopandas as gpd

    return gpd.read_parquet(store / "shapes" / key / "shapes.parquet")


def command_transcripts(args: argparse.Namespace) -> int:
    """Analysis 2's transcript tallies of one sample (cached)."""
    locations = locations_from(args)
    pair, platform = args.pair, args.platform.upper()
    cache = locations.transcripts_cache(pair, platform)
    if (cache / "tally.parquet").exists():
        raise SystemExit(f"{cache} exists (never overwritten)")
    cache.mkdir(parents=True, exist_ok=True)
    started = time.time()
    store = locations.store_path(pair, platform)
    min_qv = args.xenium_min_qv if platform == "XENIUM" else None
    inputs = load_transcripts(store, platform, min_qv)
    logger.info(
        "%s %s: %d decoded transcripts", pair, platform, inputs.record["n_decoded"]
    )
    reseg_labels = read_labels(locations.labels_path("reseg", pair, platform))
    hybrid_labels = read_labels(locations.labels_path("hybrid", pair, platform))
    reseg_table = table_ids_uint64(reseg_labels)
    hybrid_table = table_ids_uint64(hybrid_labels)
    nuclei = read_shapes(store, NUCLEI_SHAPES)
    nucleus_index = sc.points_in_polygons(inputs.x, inputs.y, nuclei.geometry.values)
    cells = read_shapes(store, ARM_SHAPES["hybrid"])
    holders = holder_ids(
        sc.nucleus_holders(
            nuclei.geometry.values, cells.geometry.values, [int(i) for i in cells.index]
        )
    )
    summary = read_json(locations.summary_path("hybrid", pair))
    gate = summary["samples"][sample_id(pair, platform)]["resolution"]["gate"]["level"]
    broad = confident_labels(hybrid_labels, "broad")
    groupings: dict[str, tuple[Mapping[int, str], Sequence[str]]] = {
        "broad": (broad, HUMAN_BROAD_CLASSES),
        "cell_class": (
            {key: glia_or_neuron(value) for key, value in broad.items()},
            ("glia", "neurons"),
        ),
    }
    if gate == FULL_GATE:
        supercluster = confident_labels(hybrid_labels, "supercluster")
        groupings["supercluster"] = (supercluster, sorted(set(supercluster.values())))
    tally = transcript_tally_frames(
        inputs,
        nucleus_index=nucleus_index,
        holders=holders,
        reseg_table=reseg_table,
        hybrid_table=hybrid_table,
        groupings=groupings,
    )
    tally.to_parquet(cache / "tally.parquet", index=False)
    checks = _tally_checks(tally, reseg_labels, hybrid_labels)
    write_json(
        {
            "pair": pair,
            "platform": platform,
            "store": str(store),
            "filter": inputs.record,
            "gate_level": gate,
            "n_nuclei": int(len(nuclei)),
            "n_nuclei_with_holder": int((holders != NO_CELL).sum()),
            "n_in_nucleus": int((nucleus_index >= 0).sum()),
            "n_reseg_table": int(len(reseg_table)),
            "n_hybrid_table": int(len(hybrid_table)),
            "checks": checks,
            "wall_s": round(time.time() - started, 1),
        },
        cache / "tally_record.json",
    )
    return 0


def command_controls(args: argparse.Namespace) -> int:
    """The control features' tallies of one sample (post hoc; cached).

    The same decoding filter, assignment and nucleus definitions as analysis
    2's genes (``command_transcripts``), on the control features only (the
    MERSCOPE blank barcodes). A sample whose store transcripts carry no
    control feature gets an empty tally and a record that says so.
    """
    locations = locations_from(args)
    pair, platform = args.pair, args.platform.upper()
    cache = locations.controls_cache(pair, platform)
    if (cache / "tally.parquet").exists():
        raise SystemExit(f"{cache} exists (never overwritten)")
    cache.mkdir(parents=True, exist_ok=True)
    started = time.time()
    store = locations.store_path(pair, platform)
    min_qv = args.xenium_min_qv if platform == "XENIUM" else None
    inputs = load_transcripts(store, platform, min_qv, features="controls")
    logger.info(
        "%s %s: %d control transcripts", pair, platform, inputs.record["n_decoded"]
    )
    reseg_table = table_ids_uint64(
        read_labels(locations.labels_path("reseg", pair, platform))
    )
    hybrid_table = table_ids_uint64(
        read_labels(locations.labels_path("hybrid", pair, platform))
    )
    nuclei = read_shapes(store, NUCLEI_SHAPES)
    nucleus_index = sc.points_in_polygons(inputs.x, inputs.y, nuclei.geometry.values)
    cells = read_shapes(store, ARM_SHAPES["hybrid"])
    holders = holder_ids(
        sc.nucleus_holders(
            nuclei.geometry.values, cells.geometry.values, [int(i) for i in cells.index]
        )
    )
    tally = transcript_tally_frames(
        inputs,
        nucleus_index=nucleus_index,
        holders=holders,
        reseg_table=reseg_table,
        hybrid_table=hybrid_table,
        groupings={},
    )
    tally.to_parquet(cache / "tally.parquet", index=False)
    write_json(
        {
            "pair": pair,
            "platform": platform,
            "store": str(store),
            "filter": inputs.record,
            "n_control_features": len(inputs.genes),
            "n_in_nucleus": int((nucleus_index >= 0).sum()),
            "n_reseg_table": int(len(reseg_table)),
            "n_hybrid_table": int(len(hybrid_table)),
            "note": ""
            if inputs.genes
            else "not_available: the store's transcripts carry no control feature",
            "wall_s": round(time.time() - started, 1),
        },
        cache / "tally_record.json",
    )
    return 0


def _tally_checks(
    tally: pd.DataFrame, reseg: pd.DataFrame, hybrid: pd.DataFrame
) -> dict[str, Any]:
    """Assigned transcripts vs the label tables' table counts (consistency)."""
    overall = tally[tally["level"] == "all"]
    reseg_sum = int(reseg.loc[reseg["in_table"], "total_counts"].sum())
    hybrid_sum = int(hybrid.loc[hybrid["in_table"], "total_counts"].sum())
    return {
        "assigned_reseg": int(overall["n_assigned_reseg"].sum()),
        "reseg_table_total_counts": reseg_sum,
        "assigned_hybrid": int(overall["n_assigned_hybrid"].sum()),
        "hybrid_table_total_counts": hybrid_sum,
        "reseg_equal": int(overall["n_assigned_reseg"].sum()) == reseg_sum,
        "hybrid_equal": int(overall["n_assigned_hybrid"].sum()) == hybrid_sum,
    }


# --------------------------------------------------------------------------
# Datasets: labels, counts and coordinates of the compared arms


@dataclass
class ArmData:
    """One arm of one sample: table cells' labels, counts and coordinates.

    Attributes:
        labels: Table cells' label rows (index ``cell_id``).
        counts: Table cells x ``genes`` counts (CSR; ``None`` if not read).
        genes: Gene symbols of ``counts``.
        gene_ids: Ensembl ids of ``genes`` (``""`` when unresolved).
        xy: Coordinates of the table cells (``obsm['spatial']``).
        aligned: Whether ``xy`` is in the pair's fixed (Xenium) frame.
    """

    labels: pd.DataFrame
    counts: sparse.csr_matrix | None
    genes: list[str]
    gene_ids: list[str]
    xy: np.ndarray | None
    aligned: bool


def read_clustered(
    path: Path,
) -> tuple[pd.Index, list[str], list[str], sparse.csr_matrix, np.ndarray | None, bool]:
    """Read ids, genes, Ensembl ids, counts and coordinates of a clustered H5AD."""
    from merxen.annotation import pipeline

    obs, var, counts, _ = pipeline.read_h5ad_counts(path, "clustered")
    xy, shape_key = pipeline.read_spatial(path)
    genes = [str(name) for name in var.index]
    ids = (
        [str(value) if pd.notna(value) else "" for value in var["ensembl_id"]]
        if "ensembl_id" in var
        else [""] * len(genes)
    )
    platform = "XENIUM" if "_XENIUM_" in path.name else "MERSCOPE"
    return obs, genes, ids, counts, xy, pipeline.in_fixed_frame(platform, shape_key)


def load_arm(
    locations: Locations,
    arm: str,
    pair: str,
    platform: str,
    *,
    counts_arm: str | None = None,
    with_counts: bool = True,
) -> ArmData:
    """Load one arm of a sample (labels from the arm, counts from ``counts_arm``).

    Args:
        locations: Inputs.
        arm: The labels' arm.
        pair: Pair.
        platform: Platform.
        counts_arm: The arm whose map_first clustered H5AD gives counts and
            coordinates (default ``arm``; hybrid_matched reads hybrid's).
        with_counts: Read counts (else coordinates only).

    Returns:
        The arm's table cells.
    """
    labels = read_labels(locations.labels_path(arm, pair, platform))
    labels = labels[labels["in_table"].to_numpy(bool)].set_index("cell_id", drop=False)
    source = counts_arm or ("hybrid" if arm == "hybrid_matched" else arm)
    obs, genes, gene_ids, counts, xy, aligned = read_clustered(
        locations.clustered_path(source, pair, platform)
    )
    position = obs.get_indexer(labels.index)
    if (position < 0).any():
        raise SystemExit(
            f"{pair} {platform} {arm}: {int((position < 0).sum())} table cells "
            f"missing from {source}'s clustered H5AD"
        )
    return ArmData(
        labels=labels,
        counts=counts[position] if with_counts else None,
        genes=genes,
        gene_ids=gene_ids,
        xy=None if xy is None else np.asarray(xy)[position],
        aligned=aligned,
    )


def align_genes(
    counts: sparse.spmatrix, genes: Sequence[str], order: Sequence[str]
) -> sparse.csr_matrix:
    """Return counts with columns in ``order`` (absent genes are zero).

    A gene listed twice in ``genes`` takes its first column.
    """
    position: dict[str, int] = {}
    for index, name in enumerate(genes):
        position.setdefault(str(name), index)
    width = counts.shape[1]
    columns = np.array(
        [position.get(str(name), width) for name in order], dtype=np.int64
    )
    padded = sparse.hstack(
        [
            sparse.csr_matrix(counts, dtype=np.float64),
            sparse.csr_matrix((counts.shape[0], 1)),
        ]
    ).tocsc()
    return sparse.csr_matrix(padded[:, columns])


def is_confident(labels: pd.DataFrame, level: str) -> np.ndarray:
    """Whether each row is confident at a level."""
    return labels[f"ct_{level}_status"].astype(str).to_numpy() == CONFIDENT


def confident_names(labels: pd.DataFrame, level: str) -> np.ndarray:
    """Confident name at a level (``None`` elsewhere)."""
    names = labels[f"ct_{level}_name"].to_numpy(object).copy()
    names[~is_confident(labels, level)] = None
    return names


# --------------------------------------------------------------------------
# Analysis 1


def record_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return the identity of an acceptance_metrics record."""
    return tuple(record.get(field) for field in RECORD_KEY)


def records_wide(
    records: Mapping[str, Sequence[Mapping[str, Any]]],
    pair: str,
) -> pd.DataFrame:
    """Lay each arm's metric records side by side (one row per record key).

    Args:
        records: Arm to its ``acceptance_metrics.json`` records (an arm
            without a report maps to ``[]``).
        pair: The pair.

    Returns:
        ``RECORD_KEY`` columns, then per arm ``value_<arm>``, ``ci_low_<arm>``,
        ``ci_high_<arm>``, ``n_<arm>``, ``status_<arm>``, and the numeric
        differences ``reseg_minus_hybrid`` and ``reseg_minus_hybrid_matched``.
        A key repeated within an arm is numbered (``occurrence``).
    """
    rows: dict[tuple[Any, ...], dict[str, Any]] = {}
    definitions: dict[tuple[Any, ...], str] = {}
    for arm, items in records.items():
        seen: dict[tuple[Any, ...], int] = {}
        for record in items:
            key = record_key(record)
            occurrence = seen.get(key, 0)
            seen[key] = occurrence + 1
            full = (*key, occurrence)
            row = rows.setdefault(
                full,
                {
                    "pair": pair,
                    "role": role(pair),
                    **dict(zip(RECORD_KEY, key, strict=True)),
                    "occurrence": occurrence,
                },
            )
            definitions.setdefault(full, str(record.get("definition") or ""))
            row[f"value_{arm}"] = record.get("value")
            row[f"ci_low_{arm}"] = record.get("ci_low")
            row[f"ci_high_{arm}"] = record.get("ci_high")
            row[f"n_{arm}"] = record.get("n")
            row[f"status_{arm}"] = record.get("status")
    for full, row in rows.items():
        row["definition"] = definitions[full]
        for other in ("hybrid", "hybrid_matched"):
            first, second = row.get("value_reseg"), row.get(f"value_{other}")
            row[f"reseg_minus_{other}"] = (
                float(first) - float(second)
                if is_number(first) and is_number(second)
                else None
            )
    frame = pd.DataFrame(list(rows.values()))
    for arm in records:
        for prefix in ("value", "ci_low", "ci_high", "n", "status"):
            if f"{prefix}_{arm}" not in frame:
                frame[f"{prefix}_{arm}"] = None
    return frame


def depth_bin_coverage(
    labels: pd.DataFrame, *, pair: str, platform: str, arm: str
) -> pd.DataFrame:
    """Coverage and status shares per depth bin and level (table cells).

    Args:
        labels: The arm's table cells' label rows.
        pair: Pair.
        platform: Platform.
        arm: Arm.

    Returns:
        One row per level and depth bin (the bundle grid; ``all`` too):
        ``n_table``, ``n_confident``, ``coverage`` and ``share_<status>``.
    """
    rows = []
    bins = labels["depth_bin"].to_numpy()
    for level in LEVELS:
        column = f"ct_{level}_status"
        if column not in labels:
            continue
        status = labels[column].astype(str).to_numpy()
        for depth in (*DEPTH_BINS, "all"):
            keep = np.ones(len(labels), bool) if depth == "all" else bins == depth
            n = int(keep.sum())
            row: dict[str, Any] = {
                "pair": pair,
                "role": role(pair),
                "platform": platform,
                "arm": arm,
                "level": level,
                "depth_bin": depth,
                "n_table": n,
                "n_confident": int((status[keep] == CONFIDENT).sum()),
            }
            row["coverage"] = row["n_confident"] / n if n else math.nan
            for value in STATUSES:
                row[f"share_{value}"] = (
                    float((status[keep] == value).sum() / n) if n else math.nan
                )
            rows.append(row)
    return pd.DataFrame(rows)


def section_of(arm: ArmData, platform: str, sid: str) -> co.SectionComposition:
    """Return the section composition RESOLVE builds from a label table."""
    labels = arm.labels
    columns = [f"soft_broad_{safe_token(name)}" for name in co.COMPOSITION_COLUMNS]
    broad_of = co.whb_broad_of()
    argmax = [
        broad_of.get(str(name)) if isinstance(name, str) else None
        for name in labels["mmc_whb_supercluster_name"]
    ]
    return co.section_composition(
        sid,
        platform,
        labels[columns].to_numpy(np.float64),
        total_counts=labels["total_counts"].to_numpy(np.float64),
        argmax_broad=argmax,
        confident_broad=labels["ct_broad_name"].to_numpy(object),
        confident=is_confident(labels, "broad"),
        xy=arm.xy,
        aligned_frame=arm.aligned,
    )


def shared_grid(
    sections: Mapping[str, co.SectionComposition],
    keep: Mapping[str, np.ndarray],
    tile_um: float = TILE_UM,
) -> tuple[dict[str, np.ndarray], int] | None:
    """Tile several sections' kept cells on one grid (``None`` without frame).

    Args:
        sections: Name to section (all must be in the fixed frame).
        keep: Name to the kept cells of that section.
        tile_um: Tile edge.

    Returns:
        ``(codes per name, n_tiles)``, or ``None`` when a section has no
        coordinates in the fixed frame.
    """
    if not all(
        section.aligned_frame and section.xy is not None
        for section in sections.values()
    ):
        return None
    names = list(sections)
    stacked = np.vstack([np.asarray(sections[name].xy)[keep[name]] for name in names])
    codes = co.tile_codes(stacked, tile_um)
    n_tiles = int(codes.max()) + 1 if len(codes) and codes.max() >= 0 else 0
    bounds = np.cumsum([0] + [int(np.count_nonzero(keep[name])) for name in names])
    return {
        name: codes[bounds[i] : bounds[i + 1]] for i, name in enumerate(names)
    }, n_tiles


def paired_jsd_rows(
    first: Mapping[str, co.SectionComposition],
    second: Mapping[str, co.SectionComposition],
    *,
    mask: Any,
    names: tuple[str, str],
    pair: str,
    kinds: Sequence[str] = JSD_KINDS,
) -> list[dict[str, Any]]:
    """JSDs of two labellings and the paired CI of their difference.

    Both labellings' sections are tiled on one grid (every kept cell of both
    arms and both platforms), and one bootstrap draw per replicate is
    applied to all four tile tables (``paired_block_bootstrap_jsd_difference``,
    joint resampling, 200 replicates, seed 0).

    Args:
        first: Platform to the first arm's section.
        second: Platform to the second arm's section.
        mask: The pair's shared tissue mask (``None``: whole section only).
        names: The two arms' names.
        pair: Pair.
        kinds: Composition kinds.

    Returns:
        One row per kind and region: both JSDs, ``first - second``, its CI
        and the two JSDs' CIs on the shared grid.
    """
    from merxen.annotation.shadow import paired_block_bootstrap_jsd_difference

    sections = {
        "a_m": first["MERSCOPE"],
        "a_x": first["XENIUM"],
        "b_m": second["MERSCOPE"],
        "b_x": second["XENIUM"],
    }
    regions_a, note = co.shared_mask_regions(sections["a_m"], sections["a_x"], mask)
    regions_b, _ = co.shared_mask_regions(sections["b_m"], sections["b_x"], mask)
    rows = []
    for region, (keep_am, keep_ax) in regions_a.items():
        if region not in regions_b:
            continue
        keep = {
            "a_m": keep_am,
            "a_x": keep_ax,
            "b_m": regions_b[region][0],
            "b_x": regions_b[region][1],
        }
        grid = shared_grid(sections, keep)
        for kind in kinds:
            matrices = {
                name: sections[name].matrices[kind][keep[name]] for name in sections
            }
            jsd_first = float(
                co.jensen_shannon_distance(
                    matrices["a_m"].sum(0), matrices["a_x"].sum(0)
                )
            )
            jsd_second = float(
                co.jensen_shannon_distance(
                    matrices["b_m"].sum(0), matrices["b_x"].sum(0)
                )
            )
            row: dict[str, Any] = {
                "pair": pair,
                "role": role(pair),
                "kind": kind,
                "region": region,
                "first": names[0],
                "second": names[1],
                "jsd_first": jsd_first,
                "jsd_second": jsd_second,
                "difference": jsd_first - jsd_second,
                "mask_note": note,
                "note": "",
            }
            if grid is None:
                row["note"] = "no shared frame: no paired CI"
            else:
                codes, n_tiles = grid
                sums = {
                    name: co.tile_sums(matrices[name], codes[name], n_tiles)
                    for name in sections
                }
                result = paired_block_bootstrap_jsd_difference(
                    sums["a_m"],
                    sums["a_x"],
                    sums["b_m"],
                    sums["b_x"],
                    n_reps=N_JSD_BOOTSTRAP,
                    seed=0,
                )
                row |= {
                    "difference_ci_low": result.ci_low,
                    "difference_ci_high": result.ci_high,
                    "first_ci_low": result.first_ci[0],
                    "first_ci_high": result.first_ci[1],
                    "second_ci_low": result.second_ci[0],
                    "second_ci_high": result.second_ci[1],
                    "share_replicates_positive": result.share_positive,
                    "n_reps": result.n_reps,
                    "resampling": result.resampling,
                }
            rows.append(row)
    return rows


def criteria_wide(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Lay the arms' criteria rows side by side.

    Args:
        frames: Arm to its criteria rows (``criterion``, ``pair``,
            ``dataset``, ``value``, ``comparator``, ``threshold``, ``note``;
            ``passes`` is kept as information only).

    Returns:
        One row per criterion and dataset (repeats numbered), ``value_<arm>``,
        ``note_<arm>`` and the numeric differences.
    """
    rows: dict[tuple[Any, ...], dict[str, Any]] = {}
    for arm, frame in frames.items():
        seen: dict[tuple[Any, ...], int] = {}
        for record in frame.to_dict("records"):
            key = (record["criterion"], record["pair"], record["dataset"])
            occurrence = seen.get(key, 0)
            seen[key] = occurrence + 1
            row = rows.setdefault(
                (*key, occurrence),
                {
                    "criterion": key[0],
                    "pair": key[1],
                    "role": role(str(key[1])),
                    "dataset": key[2],
                    "occurrence": occurrence,
                    "comparator": record.get("comparator"),
                    "threshold_for_reference": record.get("threshold"),
                },
            )
            row[f"value_{arm}"] = record.get("value")
            row[f"meets_threshold_{arm}"] = record.get("passes")
            row[f"note_{arm}"] = record.get("note")
    for row in rows.values():
        for other in ("hybrid", "hybrid_matched"):
            first, second = row.get("value_reseg"), row.get(f"value_{other}")
            row[f"reseg_minus_{other}"] = (
                float(first) - float(second)
                if _numeric(first) and _numeric(second)
                else None
            )
    return pd.DataFrame(list(rows.values()))


def _numeric(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return not isinstance(value, bool) and math.isfinite(number)


def h12_rows(
    records: Mapping[str, Sequence[Mapping[str, Any]]], pair: str
) -> list[dict[str, Any]]:
    """H12 (``report_scoring.score_h12``) of each arm's records."""
    from merxen.annotation import report_scoring

    rows = []
    for arm, items in records.items():
        if not items:
            rows.append(
                {
                    "pair": pair,
                    "role": role(pair),
                    "arm": arm,
                    "verdict": None,
                    "note": "no report",
                }
            )
            continue
        try:
            score = report_scoring.score_h12(items, pair_id=pair)
        except ValueError as error:
            rows.append(
                {
                    "pair": pair,
                    "role": role(pair),
                    "arm": arm,
                    "verdict": None,
                    "note": f"inconsistent report: {error}",
                }
            )
            continue
        payload = score.to_json()
        rows.append(
            {
                "pair": pair,
                "role": role(pair),
                "arm": arm,
                "verdict": payload.get("verdict"),
                "ci_scored": payload.get("ci_scored"),
                "ordering": json.dumps(payload.get("ordering"), sort_keys=True),
                "ordering_verdict": payload.get("ordering_verdict"),
                "wm_gm": json.dumps(_json_safe(payload.get("wm_gm")), sort_keys=True),
                "wm_gm_verdict": payload.get("wm_gm_verdict"),
                "reported_beside_tangential": json.dumps(
                    _json_safe(payload.get("reported_beside")), sort_keys=True
                ),
                "note": "; ".join(payload.get("notes") or []),
            }
        )
    return rows


def _criteria_frame(path: Path, arm: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.copy()
    frame["segmentation"] = arm
    return frame


def analysis1(locations: Locations, pairs: Sequence[str]) -> dict[str, pd.DataFrame]:
    """Analysis 1: side-by-side scores of every arm."""
    arms = (*MAIN_ARMS, *CONTEXT_ARMS)
    record_frames, h12, depth, jsd, availability = [], [], [], [], []
    for pair in pairs:
        records: dict[str, list[dict[str, Any]]] = {}
        for arm in arms:
            path = locations.metrics_path(arm, pair)
            if path.is_file():
                records[arm] = list(read_json(path)["metrics"])
                availability.append(
                    {
                        "pair": pair,
                        "arm": arm,
                        "item": "acceptance_metrics",
                        "status": "measured",
                        "source": str(path),
                    }
                )
            else:
                records[arm] = []
                availability.append(
                    {
                        "pair": pair,
                        "arm": arm,
                        "item": "acceptance_metrics",
                        "status": "not_available",
                        "source": f"missing: {path}",
                    }
                )
        availability.append(
            {
                "pair": pair,
                "arm": COMBINED,
                "item": "all",
                "status": "not_available",
                "source": COMBINED_A1_REASON,
            }
        )
        wide = records_wide(records, pair)
        wide["inputs_note"] = MATCHED_NOTE
        record_frames.append(wide)
        h12 += h12_rows(records, pair)
        sections: dict[str, dict[str, co.SectionComposition]] = {}
        for arm in arms:
            for platform in PLATFORMS:
                path = locations.labels_path(arm, pair, platform)
                if not path.is_file():
                    continue
                if arm in MAIN_ARMS:
                    data = load_arm(locations, arm, pair, platform, with_counts=False)
                    sections.setdefault(arm, {})[platform] = section_of(
                        data, platform, sample_id(pair, platform)
                    )
                    labels = data.labels
                else:
                    labels = read_labels(path)
                    labels = labels[labels["in_table"].to_numpy(bool)]
                depth.append(
                    depth_bin_coverage(labels, pair=pair, platform=platform, arm=arm)
                )
        from merxen.annotation import pipeline

        mask = pipeline.load_pair_mask(locations.alignment_dir(pair))
        for other in ("hybrid", "hybrid_matched"):
            if "reseg" in sections and other in sections and len(sections[other]) == 2:
                jsd += paired_jsd_rows(
                    sections["reseg"],
                    sections[other],
                    mask=mask,
                    names=("reseg", other),
                    pair=pair,
                )
        jsd += _resolve_jsd_check(locations, pair, sections, mask)
    criteria_frames: dict[str, pd.DataFrame] = {}
    for arm in arms:
        base = locations.acceptance_dir(arm)
        parts = []
        for name in ("criteria/criteria_table.csv", "referee/h10_table.csv"):
            path = base / name
            if path.is_file():
                frame = _criteria_frame(path, arm)
                frame = frame[frame["pair"].astype(str).isin(list(pairs))]
                if name.startswith("referee"):
                    frame["criterion"] = "H10/marker_referee.py"
                parts.append(frame)
                availability.append(
                    {
                        "pair": "all",
                        "arm": arm,
                        "item": name,
                        "status": "measured",
                        "source": str(path),
                    }
                )
            else:
                availability.append(
                    {
                        "pair": "all",
                        "arm": arm,
                        "item": name,
                        "status": "not_available",
                        "source": f"missing: {path}",
                    }
                )
        if parts:
            criteria_frames[arm] = pd.concat(parts, ignore_index=True)
    criteria = criteria_wide(criteria_frames) if criteria_frames else pd.DataFrame()
    if len(criteria):
        h4 = criteria["criterion"].astype(str).str.startswith("H4")
        criteria.loc[h4, "note_hybrid_matched"] = H4_MATCHED_REASON
        criteria["inputs_note"] = MATCHED_NOTE
    return {
        "records": pd.concat(record_frames, ignore_index=True),
        "criteria": criteria,
        "h12": pd.DataFrame(h12),
        "depth_bins": pd.concat(depth, ignore_index=True) if depth else pd.DataFrame(),
        "jsd_paired": pd.DataFrame(jsd),
        "availability": pd.DataFrame(availability),
    }


def _resolve_jsd_check(
    locations: Locations,
    pair: str,
    sections: Mapping[str, Mapping[str, co.SectionComposition]],
    mask: Any,
) -> list[dict[str, Any]]:
    """Recompute each arm's pair JSDs and compare with RESOLVE's summary."""
    rows = []
    for arm, platform_sections in sections.items():
        if len(platform_sections) != 2:
            continue
        summary = read_json(locations.summary_path(arm, pair))
        recorded = {
            (item["kind"], item["region"]): item["jsd"]
            for item in summary["pair"]["jsd"]
        }
        mine, _ = co.pair_jsd(
            platform_sections["MERSCOPE"],
            platform_sections["XENIUM"],
            mask=mask,
            n_reps=1,
        )
        for record in mine:
            expected = recorded.get((record.kind, record.region))
            rows.append(
                {
                    "pair": pair,
                    "role": role(pair),
                    "kind": record.kind,
                    "region": record.region,
                    "first": arm,
                    "second": "resolve_summary",
                    "jsd_first": record.jsd,
                    "jsd_second": expected,
                    "difference": (
                        None
                        if expected is None or record.jsd is None
                        else float(record.jsd) - float(expected)
                    ),
                    "note": "check: recomputed vs RESOLVE (6-digit rounding)",
                }
            )
    return rows


# --------------------------------------------------------------------------
# Analysis 2


def gene_list_table(panel_genes: Iterable[str]) -> pd.DataFrame:
    """Every gene of the two lists, its source and whether the panel has it."""
    panel = {str(gene) for gene in panel_genes}
    rows = []
    for name, entries in GENE_LISTS.items():
        for entry in entries:
            rows.append(
                {
                    "list": name,
                    "symbol": entry.symbol,
                    "on_panel": entry.symbol in panel,
                    "evidence": entry.evidence,
                    "source": entry.source,
                }
            )
    return pd.DataFrame(rows)


def gene_lengths(
    gtf: Path | None, gene_ids: Iterable[str], cache: Path
) -> pd.DataFrame:
    """Longest annotated transcript per gene (cached; empty without a GTF)."""
    columns = ["gene_id", "gene_name", "transcript_id", "transcript_length"]
    if gtf is None:
        return pd.DataFrame(columns=columns)
    if cache.is_file():
        return pd.read_csv(cache)
    with gtf.open(encoding="utf-8") as handle:
        frame = sc.longest_transcript_lengths(handle, gene_ids)
    write_table(frame, cache)
    return frame


def gene_covariates(
    genes: pd.DataFrame,
    *,
    lengths: pd.DataFrame,
    probes: pd.DataFrame | None,
    n_hybrid_table: int,
) -> pd.DataFrame:
    """Add the analysis 2 covariates to one sample's gene rows.

    Args:
        genes: One row per gene (``gene``, ``gene_id``, ``n_assigned_hybrid``).
        lengths: ``longest_transcript_lengths`` output.
        probes: ``xenium_probe_counts`` output (``None``: not available).
        n_hybrid_table: proseg_hybrid table cells.

    Returns:
        A copy with ``expression_level``, ``gene_length``, ``probe_count``
        and the list memberships.
    """
    out = genes.copy()
    mean = out["n_assigned_hybrid"].astype(np.float64) / max(1, n_hybrid_table)
    out["expression_level"] = np.where(
        mean > 0, np.log10(mean.where(mean > 0)), math.nan
    )
    length_of = dict(
        zip(lengths["gene_id"].astype(str), lengths["transcript_length"], strict=True)
    )
    out["gene_length"] = [
        float(length_of[g.split(".")[0]])
        if g and g.split(".")[0] in length_of
        else math.nan
        for g in out["gene_id"].astype(str)
    ]
    if probes is None or probes.empty:
        out["probe_count"] = math.nan
    else:
        by_name = dict(
            zip(probes["gene_name"].astype(str), probes["probe_count"], strict=True)
        )
        out["probe_count"] = [float(by_name.get(str(g), math.nan)) for g in out["gene"]]
    for name, entries in GENE_LISTS.items():
        members = {entry.symbol for entry in entries}
        out[name] = out["gene"].astype(str).isin(members)
    return out


def gene_tests(
    genes: pd.DataFrame,
    *,
    pair: str,
    platform: str,
    glia_neurons: pd.DataFrame | None,
    probe_note: str,
) -> list[dict[str, Any]]:
    """Analysis 2's tests of one sample (before the BH correction).

    Args:
        genes: The sample's gene rows with shares and covariates.
        pair: Pair.
        platform: Platform.
        glia_neurons: Per gene ``L_glia`` and ``L_neurons`` (``None``: none).
        probe_note: Why the probe count is missing, if it is.

    Returns:
        One row per test.
    """
    rows = []
    base = {"pair": pair, "role": role(pair), "platform": platform}
    for metric in LOSS_METRICS:
        for covariate in COVARIATES:
            result = sc.spearman_bootstrap(genes[metric], genes[covariate])
            note = result.note
            if covariate == "probe_count" and genes[covariate].isna().all():
                note = probe_note
            if covariate == "gene_length" and genes[covariate].isna().all():
                note = "not_available: no gene annotation table"
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "covariate": covariate,
                    **result.to_row(),
                    "note": note,
                }
            )
        for name in GENE_LISTS:
            member = genes[name].to_numpy(bool)
            result = sc.mann_whitney(
                genes.loc[member, metric], genes.loc[~member, metric]
            )
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "covariate": f"member:{name}",
                    **result.to_row(),
                    "n_on_panel": int(member.sum()),
                }
            )
    if glia_neurons is not None:
        result = sc.wilcoxon_paired(glia_neurons["L_glia"], glia_neurons["L_neurons"])
        rows.append(
            {
                **base,
                "metric": "L",
                "covariate": "glia_minus_neurons",
                **result.to_row(),
            }
        )
    return rows


def platform_tests(genes: pd.DataFrame, pair: str) -> list[dict[str, Any]]:
    """Wilcoxon signed-rank of L, MERSCOPE vs Xenium, over a pair's genes."""
    wide = genes[genes["pair"] == pair].pivot_table(
        index="gene", columns="platform", values="L"
    )
    if not set(PLATFORMS) <= set(wide.columns):
        return []
    result = sc.wilcoxon_paired(wide["MERSCOPE"], wide["XENIUM"])
    return [
        {
            "pair": pair,
            "role": role(pair),
            "platform": "MERSCOPE_minus_XENIUM",
            "metric": "L",
            "covariate": "platform",
            **result.to_row(),
        }
    ]


def expression_frames(
    reseg: ArmData,
    hybrid: ArmData,
    *,
    reference: pd.DataFrame,
    pair: str,
    platform: str,
    arms: Mapping[str, tuple[ArmData, np.ndarray]] | None = None,
) -> dict[str, pd.DataFrame]:
    """The effect on expression (analysis 2) or the arms' own versions (4).

    With ``arms`` unset: per confident broad class c, the cells confident c
    in both reseg and hybrid (same ids), ``p_reseg`` from reseg counts and
    ``p_hybrid`` from hybrid counts. With ``arms``: each arm's own cells
    (arm name -> (counts source, confident broad names per row)).

    Args:
        reseg: reseg's table cells.
        hybrid: proseg_hybrid's table cells.
        reference: ``classes x gene_ids`` WHB shares (the genes used).
        pair: Pair.
        platform: Platform.
        arms: Arms with their own cells (analysis 4).

    Returns:
        ``expression`` (class x gene), ``summary`` (class) and
        ``marker_contrasts`` frames.
    """
    from merxen.annotation.shadow import E1_REFEREE_MARKERS

    gene_ids = [str(gene) for gene in reference.columns]
    symbol_of = dict(zip(hybrid.gene_ids, hybrid.genes, strict=True))
    symbols = [symbol_of.get(gene, gene) for gene in gene_ids]
    classes = list(HUMAN_BROAD_CLASSES)
    if arms is None:
        names_r = pd.Series(
            confident_names(reseg.labels, "broad"), index=reseg.labels.index
        )
        names_h = pd.Series(
            confident_names(hybrid.labels, "broad"), index=hybrid.labels.index
        )
        both = names_r.index.intersection(names_h.index)
        same = both[
            (names_r.loc[both] == names_h.loc[both]).to_numpy(bool)
            & names_r.loc[both].notna().to_numpy()
        ]
        selection = {
            "reseg": (
                reseg,
                names_r.reindex(reseg.labels.index)
                .where(names_r.index.isin(same), None)
                .to_numpy(object),
            ),
            "hybrid": (
                hybrid,
                names_h.reindex(hybrid.labels.index)
                .where(names_h.index.isin(same), None)
                .to_numpy(object),
            ),
        }
    else:
        selection = dict(arms)
    shares: dict[str, np.ndarray] = {}
    cells: dict[str, np.ndarray] = {}
    for arm, (data, names) in selection.items():
        if data.counts is None:
            raise SystemExit(f"{arm}: counts not loaded")
        counts = align_genes(data.counts, data.gene_ids, gene_ids)
        shares[arm], cells[arm] = sc.pseudobulk_shares(counts, names, classes)
    whb = reference.reindex(index=classes).to_numpy(np.float64)
    rows = []
    for c_index, cls in enumerate(classes):
        for g_index, gene_id in enumerate(gene_ids):
            row: dict[str, Any] = {
                "pair": pair,
                "role": role(pair),
                "platform": platform,
                "class": cls,
                "gene": symbols[g_index],
                "gene_id": gene_id,
                "p_whb": whb[c_index, g_index],
            }
            for arm in selection:
                p = shares[arm][c_index, g_index]
                row[f"n_cells_{arm}"] = int(cells[arm][c_index])
                row[f"p_{arm}"] = p if cells[arm][c_index] else math.nan
                ref = whb[c_index, g_index]
                row[f"factor_{arm}"] = (
                    math.log2(p / ref) if cells[arm][c_index] and ref > 0 else math.nan
                )
            rows.append(row)
    expression = pd.DataFrame(rows)
    arm_names = list(selection)
    for first, second in _arm_pairs(arm_names):
        expression[f"log2_{first}_over_{second}"] = np.log2(
            expression[f"p_{first}"] / expression[f"p_{second}"]
        )
    summary_rows = []
    for cls, part in expression.groupby("class", sort=False):
        row = {"pair": pair, "role": role(pair), "platform": platform, "class": cls}
        for arm in arm_names:
            factor = part[f"factor_{arm}"].to_numpy(np.float64)
            finite = factor[np.isfinite(factor)]
            row[f"n_cells_{arm}"] = int(part[f"n_cells_{arm}"].iloc[0])
            row[f"factor_spread_q90_q10_{arm}"] = (
                float(np.quantile(finite, 0.9) - np.quantile(finite, 0.1))
                if len(finite) >= 3
                else math.nan
            )
        for first, second in _arm_pairs(arm_names):
            ratio = part[f"log2_{first}_over_{second}"].to_numpy(np.float64)
            finite = ratio[np.isfinite(ratio)]
            row[f"median_abs_log2_{first}_over_{second}"] = (
                float(np.median(np.abs(finite))) if len(finite) else math.nan
            )
            result = sc.spearman_bootstrap(
                part[f"factor_{first}"],
                part[f"factor_{second}"],
                n_reps=sc.N_GENE_BOOTSTRAP,
            )
            row[f"factor_spearman_{first}_{second}"] = result.effect
            row[f"factor_spearman_{first}_{second}_ci_low"] = result.ci_low
            row[f"factor_spearman_{first}_{second}_ci_high"] = result.ci_high
        summary_rows.append(row)
    contrast_rows = []
    gene_index = {symbol: index for index, symbol in enumerate(symbols)}
    for c1, markers in E1_REFEREE_MARKERS.items():
        if c1 not in classes:
            continue
        i1 = classes.index(c1)
        for marker in markers:
            if marker not in gene_index:
                continue
            g = gene_index[marker]
            for c2 in classes:
                if c2 == c1:
                    continue
                i2 = classes.index(c2)
                row = {
                    "pair": pair,
                    "role": role(pair),
                    "platform": platform,
                    "marker": marker,
                    "marker_class": c1,
                    "other_class": c2,
                }
                for arm in arm_names:
                    ok = cells[arm][i1] > 0 and cells[arm][i2] > 0
                    row[f"contrast_{arm}"] = (
                        math.log2(shares[arm][i1, g] / shares[arm][i2, g])
                        if ok
                        else math.nan
                    )
                for first, second in _arm_pairs(arm_names):
                    row[f"contrast_{first}_minus_{second}"] = (
                        row[f"contrast_{first}"] - row[f"contrast_{second}"]
                    )
                contrast_rows.append(row)
    return {
        "expression": expression,
        "expression_summary": pd.DataFrame(summary_rows),
        "marker_contrasts": pd.DataFrame(contrast_rows),
    }


def _arm_pairs(arms: Sequence[str]) -> list[tuple[str, str]]:
    """The comparisons of a set of arms (reseg vs hybrid; combined vs both)."""
    if COMBINED in arms:
        return [(COMBINED, other) for other in arms if other != COMBINED]
    return [("reseg", "hybrid")] if {"reseg", "hybrid"} <= set(arms) else []


def reference_for(
    locations: Locations, store: Path, pair: str, gene_ids: Sequence[str]
) -> pd.DataFrame:
    """The primary bundle's broad-class shares over a sample's genes."""
    summary = read_json(locations.summary_path("hybrid", pair))
    first = next(iter(summary["samples"].values()))
    build = first["bundles"][PRIMARY_REFERENCE]["resolved_build_hash"]
    bundle = store / PRIMARY_REFERENCE / build
    profiles = pd.read_parquet(bundle / "profiles.parquet")
    present = set(profiles["gene_id"].astype(str))
    genes = list(dict.fromkeys(gene for gene in gene_ids if gene and gene in present))
    return sc.reference_broad_shares(
        profiles, PROFILE_LEVEL, co.whb_broad_of(), genes, list(HUMAN_BROAD_CLASSES)
    )


def analysis2(
    locations: Locations,
    pairs: Sequence[str],
    *,
    store: Path,
    gtf: Path | None,
    xenium_panels: Mapping[str, Path],
) -> dict[str, pd.DataFrame]:
    """Analysis 2: the transcript-loss audit and its effect on expression."""
    gene_rows, type_rows, tests, records = [], [], [], []
    expression, summaries, contrasts = [], [], []
    all_ids: set[str] = set()
    datasets = []
    for pair in pairs:
        for platform in PLATFORMS:
            hybrid = load_arm(locations, "hybrid", pair, platform)
            reseg = load_arm(locations, "reseg", pair, platform)
            datasets.append((pair, platform, reseg, hybrid))
            all_ids |= {gene for gene in hybrid.gene_ids if gene}
    lengths = gene_lengths(gtf, all_ids, locations.out / "cache" / "gene_lengths.csv")
    for pair, platform, reseg, hybrid in datasets:
        cache = locations.transcripts_cache(pair, platform)
        tally = pd.read_parquet(cache / "tally.parquet")
        record = read_json(cache / "tally_record.json")
        records.append({"pair": pair, "platform": platform, **_flatten(record)})
        shares = sc.loss_shares(tally)
        id_of = dict(zip(hybrid.genes, hybrid.gene_ids, strict=True))
        shares["gene_id"] = [id_of.get(str(g), "") for g in shares["gene"]]
        shares.insert(0, "platform", platform)
        shares.insert(0, "role", role(pair))
        shares.insert(0, "pair", pair)
        overall = shares[shares["level"] == "all"].drop(columns=["level", "group"])
        panel_path = xenium_panels.get(pair) if platform == "XENIUM" else None
        probes = sc.xenium_probe_counts(panel_path) if panel_path is not None else None
        probe_note = (
            "not_available: no probe counts in the MERSCOPE outputs on this host"
            if platform == "MERSCOPE"
            else ("" if probes is not None else "not_available: no gene_panel.json")
        )
        overall = gene_covariates(
            overall,
            lengths=lengths,
            probes=probes,
            n_hybrid_table=int(record["n_hybrid_table"]),
        )
        gene_rows.append(overall)
        type_rows.append(shares[shares["level"] != "all"])
        cell_class = shares[shares["level"] == "cell_class"].pivot_table(
            index="gene", columns="group", values="L"
        )
        glia_neurons = (
            cell_class.rename(columns={"glia": "L_glia", "neurons": "L_neurons"})
            if {"glia", "neurons"} <= set(cell_class.columns)
            else None
        )
        tests += gene_tests(
            overall,
            pair=pair,
            platform=platform,
            glia_neurons=glia_neurons,
            probe_note=probe_note,
        )
        reference = reference_for(locations, store, pair, hybrid.gene_ids)
        frames = expression_frames(
            reseg, hybrid, reference=reference, pair=pair, platform=platform
        )
        expression.append(frames["expression"])
        summaries.append(frames["expression_summary"])
        contrasts.append(frames["marker_contrasts"])
    genes = pd.concat(gene_rows, ignore_index=True)
    for pair in pairs:
        tests += platform_tests(genes, pair)
    test_frame = pd.DataFrame(tests)
    test_frame["q_value_bh"] = sc.benjamini_hochberg(
        test_frame["p_value"].to_numpy(np.float64)
    )
    test_frame["bh_family"] = "analysis 2 (all tests, all datasets)"
    panel_genes = set(genes["gene"].astype(str))
    return {
        "genes": genes,
        "genes_by_type": pd.concat(type_rows, ignore_index=True),
        "tests": test_frame,
        "expression": pd.concat(expression, ignore_index=True),
        "expression_summary": pd.concat(summaries, ignore_index=True),
        "marker_contrasts": pd.concat(contrasts, ignore_index=True),
        "gene_lists": gene_list_table(panel_genes),
        "inputs": pd.DataFrame(records),
        "gene_lengths": lengths,
    }


def _flatten(record: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in record.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            out |= _flatten(value, f"{name}.")
        elif isinstance(value, list):
            out[name] = json.dumps(_json_safe(value))[:2000]
        else:
            out[name] = value
    return out


# --------------------------------------------------------------------------
# Analysis 3


def native_centroids(store: Path, key: str, ids: pd.Index) -> np.ndarray:
    """Centroids of a store's shape polygons for ``ids`` (NaN if absent)."""
    shapes = read_shapes(store, key)
    centroid = shapes.geometry.centroid
    frame = pd.DataFrame(
        {"x": centroid.x.to_numpy(), "y": centroid.y.to_numpy()},
        index=pd.Index(shapes.index.astype(str)),
    )
    return frame.reindex(ids).to_numpy(np.float64)


def negative_mask(negatives: pd.DataFrame, gene_ids: Sequence[str]) -> np.ndarray:
    """``classes x genes`` mask of each broad class's negative genes."""
    from merxen.annotation.flags import negative_gene_mask

    return negative_gene_mask(negatives, list(gene_ids), list(HUMAN_BROAD_CLASSES))


def analysis3_sample(
    reseg: ArmData,
    hybrid: ArmData,
    *,
    xy_reseg: np.ndarray,
    xy_hybrid: np.ndarray,
    negatives: pd.DataFrame,
    pair: str,
    platform: str,
) -> dict[str, list[dict[str, Any]]]:
    """Analysis 3 of one sample.

    Args:
        reseg: reseg table cells (labels, counts).
        hybrid: proseg_hybrid table cells.
        xy_reseg: reseg native centroids of ``reseg.labels`` rows.
        xy_hybrid: proseg_hybrid native centroids of ``hybrid.labels`` rows.
        negatives: The primary bundle's ``negative_genes.parquet``.
        pair: Pair.
        platform: Platform.

    Returns:
        ``bins``, ``genes``, ``gene_bins`` and ``summary`` rows.
    """
    classes = list(HUMAN_BROAD_CLASSES)
    names = {
        "reseg": pd.Series(
            confident_names(reseg.labels, "broad"), index=reseg.labels.index
        ),
        "hybrid": pd.Series(
            confident_names(hybrid.labels, "broad"), index=hybrid.labels.index
        ),
    }
    xy = {"reseg": xy_reseg, "hybrid": xy_hybrid}
    data = {"reseg": reseg, "hybrid": hybrid}
    both = names["reseg"].dropna().index.intersection(names["hybrid"].dropna().index)
    same = both[(names["reseg"].loc[both] == names["hybrid"].loc[both]).to_numpy(bool)]
    gene_ids = [gene for gene in hybrid.gene_ids if gene]
    mask = negative_mask(negatives, gene_ids)
    per_arm: dict[str, dict[str, np.ndarray]] = {}
    for arm in ("reseg", "hybrid"):
        labels = data[arm].labels
        distance = sc.nearest_other_class_distance(xy[arm], names[arm].to_numpy(object))
        position = labels.index.get_indexer(both)
        counts = align_genes(data[arm].counts, data[arm].gene_ids, gene_ids)  # type: ignore[arg-type]
        detected = (counts[position] > 0).toarray()
        cls = names[arm].loc[both].to_numpy(object)
        class_index = np.array([classes.index(str(c)) for c in cls], dtype=np.int64)
        per_arm[arm] = {
            "class": class_index,
            "bin": sc.distance_bin_codes(distance[position]),
            "distance": distance[position],
            "contamination": labels["contamination_score"].to_numpy(np.float64)[
                position
            ],
            "detected": detected,
        }
    tiles = co.tile_codes(
        np.where(
            np.isfinite(xy_hybrid[hybrid.labels.index.get_indexer(both)]),
            xy_hybrid[hybrid.labels.index.get_indexer(both)],
            xy_reseg[reseg.labels.index.get_indexer(both)],
        ),
        TILE_UM,
    )
    same_set = both.isin(same)
    out: dict[str, list[dict[str, Any]]] = {
        "bins": [],
        "genes": [],
        "gene_bins": [],
        "summary": [],
    }
    n_bins = len(sc.DISTANCE_BIN_LABELS)
    for cell_set, keep in (
        ("both_confident", np.ones(len(both), bool)),
        ("same_label", same_set),
    ):
        for c_index, cls in enumerate(classes):
            negative_genes = np.flatnonzero(mask[c_index])
            values: dict[str, dict[str, np.ndarray]] = {}
            for arm in ("reseg", "hybrid"):
                arm_data = per_arm[arm]
                member = keep & (arm_data["class"] == c_index)
                contamination = np.where(member, arm_data["contamination"], math.nan)
                if len(negative_genes):
                    rate = arm_data["detected"][:, negative_genes].mean(axis=1)
                else:
                    rate = np.full(len(member), math.nan)
                values[arm] = {
                    "member": member,
                    "contamination_score": contamination,
                    "negative_detection_rate": np.where(member, rate, math.nan),
                }
            for metric in ("contamination_score", "negative_detection_rate"):
                result = sc.tile_bootstrap_ratio(
                    values["reseg"][metric],
                    per_arm["reseg"]["bin"],
                    values["hybrid"][metric],
                    per_arm["hybrid"]["bin"],
                    tiles,
                    n_bins,
                )
                overall = sc.tile_bootstrap_ratio(
                    values["reseg"][metric],
                    np.where(values["reseg"]["member"], 0, -1),
                    values["hybrid"][metric],
                    np.where(values["hybrid"]["member"], 0, -1),
                    tiles,
                    1,
                )
                for b_index, label in enumerate((*sc.DISTANCE_BIN_LABELS, "all")):
                    source, index = (
                        (overall, 0) if label == "all" else (result, b_index)
                    )
                    out["bins"].append(
                        {
                            "pair": pair,
                            "role": role(pair),
                            "platform": platform,
                            "cells": cell_set,
                            "class": cls,
                            "metric": metric,
                            "distance_bin": label,
                            "n_reseg": int(source.n_a[index]),
                            "n_hybrid": int(source.n_b[index]),
                            "mean_reseg": source.mean_a[index],
                            "mean_hybrid": source.mean_b[index],
                            "hybrid_minus_reseg": source.difference[index],
                            "ci_low": source.ci_low[index],
                            "ci_high": source.ci_high[index],
                            "n_tiles": source.n_tiles,
                            "n_negative_genes": int(len(negative_genes)),
                        }
                    )
            for gene in negative_genes:
                gene_values = {
                    arm: np.where(
                        values[arm]["member"],
                        per_arm[arm]["detected"][:, gene],
                        math.nan,
                    )
                    for arm in ("reseg", "hybrid")
                }
                overall = sc.tile_bootstrap_ratio(
                    gene_values["reseg"],
                    np.where(values["reseg"]["member"], 0, -1),
                    gene_values["hybrid"],
                    np.where(values["hybrid"]["member"], 0, -1),
                    tiles,
                    1,
                )
                symbol = hybrid.genes[hybrid.gene_ids.index(gene_ids[gene])]
                out["genes"].append(
                    {
                        "pair": pair,
                        "role": role(pair),
                        "platform": platform,
                        "cells": cell_set,
                        "class": cls,
                        "gene": symbol,
                        "gene_id": gene_ids[gene],
                        "n_reseg": int(overall.n_a[0]),
                        "n_hybrid": int(overall.n_b[0]),
                        "detection_reseg": overall.mean_a[0],
                        "detection_hybrid": overall.mean_b[0],
                        "hybrid_minus_reseg": overall.difference[0],
                        "ci_low": overall.ci_low[0],
                        "ci_high": overall.ci_high[0],
                    }
                )
                if cell_set != "both_confident":
                    continue
                for b_index, label in enumerate(sc.DISTANCE_BIN_LABELS):
                    rates = {}
                    for arm in ("reseg", "hybrid"):
                        in_bin = values[arm]["member"] & (
                            per_arm[arm]["bin"] == b_index
                        )
                        rates[arm] = (
                            float(per_arm[arm]["detected"][in_bin, gene].mean())
                            if in_bin.any()
                            else math.nan,
                            int(in_bin.sum()),
                        )
                    out["gene_bins"].append(
                        {
                            "pair": pair,
                            "role": role(pair),
                            "platform": platform,
                            "class": cls,
                            "gene": symbol,
                            "distance_bin": label,
                            "n_reseg": rates["reseg"][1],
                            "n_hybrid": rates["hybrid"][1],
                            "detection_reseg": rates["reseg"][0],
                            "detection_hybrid": rates["hybrid"][0],
                            "hybrid_minus_reseg": rates["hybrid"][0]
                            - rates["reseg"][0],
                        }
                    )
    out["summary"].append(
        {
            "pair": pair,
            "role": role(pair),
            "platform": platform,
            "n_reseg_confident_broad": int(names["reseg"].notna().sum()),
            "n_hybrid_confident_broad": int(names["hybrid"].notna().sum()),
            "n_both_confident": int(len(both)),
            "n_same_label": int(len(same)),
            "n_both_confident_other_label": int(len(both) - len(same)),
            "n_tiles": int(tiles.max() + 1) if len(tiles) else 0,
            "median_distance_reseg": float(np.nanmedian(per_arm["reseg"]["distance"]))
            if len(both)
            else math.nan,
            "median_distance_hybrid": float(np.nanmedian(per_arm["hybrid"]["distance"]))
            if len(both)
            else math.nan,
        }
    )
    return out


def analysis3(
    locations: Locations, pairs: Sequence[str], *, store: Path
) -> dict[str, pd.DataFrame]:
    """Analysis 3: the spill-over audit."""
    rows: dict[str, list[dict[str, Any]]] = {
        "bins": [],
        "genes": [],
        "gene_bins": [],
        "summary": [],
    }
    for pair in pairs:
        summary = read_json(locations.summary_path("hybrid", pair))
        first = next(iter(summary["samples"].values()))
        build = first["bundles"][PRIMARY_REFERENCE]["resolved_build_hash"]
        negatives = pd.read_parquet(
            store / PRIMARY_REFERENCE / build / "negative_genes.parquet"
        )
        for platform in PLATFORMS:
            reseg = load_arm(locations, "reseg", pair, platform)
            hybrid = load_arm(locations, "hybrid", pair, platform)
            sample_store = locations.store_path(pair, platform)
            result = analysis3_sample(
                reseg,
                hybrid,
                xy_reseg=native_centroids(
                    sample_store, ARM_SHAPES["reseg"], reseg.labels.index
                ),
                xy_hybrid=native_centroids(
                    sample_store, ARM_SHAPES["hybrid"], hybrid.labels.index
                ),
                negatives=negatives,
                pair=pair,
                platform=platform,
            )
            for key, value in result.items():
                rows[key] += value
            logger.info("analysis 3: %s %s done", pair, platform)
    return {key: pd.DataFrame(value) for key, value in rows.items()}


# --------------------------------------------------------------------------
# Analysis 4


def combined_coverage(
    hybrid: ArmData, reseg_labels: pd.DataFrame, *, pair: str, platform: str
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """The combined labels of a sample and their coverage per level.

    Args:
        hybrid: proseg_hybrid table cells.
        reseg_labels: reseg's label table (every object).
        pair: Pair.
        platform: Platform.

    Returns:
        ``(combined, rows)``: the combined label rows (index ``cell_id``,
        hybrid table order) and one coverage row per level.
    """
    columns = [
        f"ct_{level}_{field}" for level in LEVELS for field in ("status", "name")
    ]
    columns = [column for column in columns if column in reseg_labels]
    combined = sc.combined_labels(hybrid.labels.index, reseg_labels, columns)
    n = len(combined)
    with_cell = combined["has_reseg_table_cell"].to_numpy(bool)
    rows = []
    reseg_table = reseg_labels[reseg_labels["in_table"].to_numpy(bool)]
    for level in LEVELS:
        column = f"ct_{level}_status"
        if column not in combined:
            continue
        status = combined[column].astype(object).to_numpy()
        confident = status == CONFIDENT
        rows.append(
            {
                "pair": pair,
                "role": role(pair),
                "platform": platform,
                "level": level,
                "n_hybrid_table": n,
                "n_with_reseg_table_cell": int(with_cell.sum()),
                "share_without_reseg_table_cell": float((~with_cell).mean())
                if n
                else math.nan,
                "n_confident_combined": int(confident.sum()),
                "share_confident_combined": float(confident.mean()) if n else math.nan,
                "share_confident_hybrid": float(
                    is_confident(hybrid.labels, level).mean()
                )
                if n
                else math.nan,
                "share_confident_reseg_of_reseg_table": (
                    float(is_confident(reseg_table, level).mean())
                    if len(reseg_table)
                    else math.nan
                ),
                "n_reseg_table": int(len(reseg_table)),
            }
        )
    return combined, rows


def confident_section(
    names: np.ndarray, xy: np.ndarray | None, aligned: bool, platform: str, sid: str
) -> co.SectionComposition:
    """A section whose only kind is the confident broad one-hot."""
    confident = np.array([value is not None for value in names])
    return co.SectionComposition(
        sample_id=sid,
        platform=platform,
        matrices={"confident": co.one_hot_broad_matrix(names, include=confident)},
        xy=xy,
        aligned_frame=aligned and xy is not None,
    )


def analysis4(
    locations: Locations, pairs: Sequence[str], *, store: Path
) -> dict[str, pd.DataFrame]:
    """Analysis 4: the combined mode."""
    coverage, composition, jsd, expression, summaries, contrasts = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for pair in pairs:
        sections: dict[str, dict[str, co.SectionComposition]] = {}
        for platform in PLATFORMS:
            sid = sample_id(pair, platform)
            hybrid = load_arm(locations, "hybrid", pair, platform)
            reseg = load_arm(locations, "reseg", pair, platform)
            reseg_labels = read_labels(locations.labels_path("reseg", pair, platform))
            combined, rows = combined_coverage(
                hybrid, reseg_labels, pair=pair, platform=platform
            )
            coverage += rows
            combined_names = combined["ct_broad_name"].to_numpy(object).copy()
            combined_names[
                combined["ct_broad_status"].to_numpy(object) != CONFIDENT
            ] = None
            arm_names = {
                "reseg": (reseg, confident_names(reseg.labels, "broad")),
                "hybrid": (hybrid, confident_names(hybrid.labels, "broad")),
                COMBINED: (hybrid, combined_names),
            }
            for arm, (data, names) in arm_names.items():
                sections.setdefault(arm, {})[platform] = confident_section(
                    names, data.xy, data.aligned, platform, sid
                )
                values = pd.Series(names, dtype=object).dropna()
                counts = values.value_counts()
                total = float(counts.sum())
                for cls in HUMAN_BROAD_CLASSES:
                    composition.append(
                        {
                            "pair": pair,
                            "role": role(pair),
                            "platform": platform,
                            "arm": arm,
                            "class": cls,
                            "n_confident": int(counts.get(cls, 0)),
                            "share": float(counts.get(cls, 0) / total)
                            if total
                            else math.nan,
                        }
                    )
            reference = reference_for(locations, store, pair, hybrid.gene_ids)
            frames = expression_frames(
                reseg,
                hybrid,
                reference=reference,
                pair=pair,
                platform=platform,
                arms=arm_names,
            )
            expression.append(frames["expression"])
            summaries.append(frames["expression_summary"])
            contrasts.append(frames["marker_contrasts"])
        for arm, platform_sections in sections.items():
            records, note = co.pair_jsd(
                platform_sections["MERSCOPE"],
                platform_sections["XENIUM"],
                mask=None,
                kinds=("confident",),
            )
            for record in records:
                jsd.append(
                    {
                        "pair": pair,
                        "role": role(pair),
                        "arm": arm,
                        "kind": "confident",
                        "region": record.region,
                        "jsd": record.jsd,
                        "ci_low": record.ci_low,
                        "ci_high": record.ci_high,
                        "resampling": record.resampling,
                        "n_cells_merscope": record.n_cells_a,
                        "n_cells_xenium": record.n_cells_b,
                        "n_tile_locations": record.n_tile_locations,
                    }
                )
        for other in ("hybrid", "reseg"):
            for row in paired_jsd_rows(
                sections[COMBINED],
                sections[other],
                mask=None,
                names=(COMBINED, other),
                pair=pair,
                kinds=("confident",),
            ):
                jsd.append({"arm": f"{COMBINED}_minus_{other}", **row})
    return {
        "coverage": pd.DataFrame(coverage),
        "composition": pd.DataFrame(composition),
        "jsd": pd.DataFrame(jsd),
        "expression": pd.concat(expression, ignore_index=True),
        "expression_summary": pd.concat(summaries, ignore_index=True),
        "marker_contrasts": pd.concat(contrasts, ignore_index=True),
        "pipeline_change": pd.DataFrame(
            [
                {
                    "mode": COMBINED,
                    "change_needed": PIPELINE_CHANGE,
                    "implemented_in_m8b": False,
                }
            ]
        ),
    }


# --------------------------------------------------------------------------
# Post hoc (after the 2026-10-01 review of the first memo; not pre-registered)

POSTHOC_NOTE: Final = (
    "post hoc: added after the 2026-10-01 review of the first decision memo; "
    "not pre-registered; information only (fits, scores and selects nothing)"
)
SELECTION_METRICS: Final[tuple[str, ...]] = (
    "coverage_lineage",
    "coverage_broad",
    "coverage_supercluster",
    "H2_implausible_share",
)
SELECTION_LABEL_COLUMNS: Final[tuple[str, ...]] = (
    "cell_id",
    "in_table",
    "total_counts",
    "flag_implausible",
    "ct_lineage_status",
    "ct_broad_status",
    "ct_supercluster_status",
)
NEAR_BIN: Final = sc.DISTANCE_BIN_LABELS[0]
FAR_BIN: Final = sc.DISTANCE_BIN_LABELS[-1]
# A gene's decoded count over the median control feature's: <= 2x, between,
# >= 10x.
NOISE_STRATA: Final[tuple[str, ...]] = ("<=2x", "2-10x", ">=10x")
NOISE_LOW: Final = 2.0
NOISE_HIGH: Final = 10.0
ALL_GENES: Final = "all genes"
CONTROL_STRATUM: Final = "control features"
NO_CONTROLS: Final = (
    "not_available: the store's transcripts carry no control feature "
    "(analysis 2's tally removed none)"
)
COMMON_NEIGHBOURS: Final = (
    "one distance per cell for both arms: its proseg_hybrid centroid to the "
    "nearest proseg_hybrid confident broad cell of another class; cells with "
    "the same confident broad label in both arms"
)
OWN_NEIGHBOURS: Final = (
    "analysis 3 as pre-registered: each arm's own centroids and its own "
    "confident cells as neighbours"
)
MATCHED_COUNT_SOURCES: Final[tuple[tuple[str, str, str], ...]] = (
    (
        "MAP / RESOLVE labels, statuses and soft compositions",
        "thinned counts",
        "yes",
    ),
    (
        "label-table measures: H1, H2, H3, H7, H8, the cross-platform JSDs, the "
        "depth-bin coverage and the labels H12 orders",
        "thinned counts (through the labels)",
        "yes",
    ),
    (
        "H9 pseudo-labels (resolve_criteria.py on the clustered H5ADs)",
        "unthinned proseg_hybrid counts",
        "no: thinned labels scored against pseudo-labels of unthinned counts",
    ),
    (
        "H10 marker scores (marker_referee.py)",
        "unthinned proseg_hybrid counts",
        "no: thinned labels with marker scores of unthinned counts",
    ),
    (
        "report items 4 (reference expectations) and 7 (cross-platform "
        "concordance, platform factors)",
        "unthinned proseg_hybrid counts",
        "no: thinned labels with unthinned counts",
    ),
    ("H4 (held-out genes)", "not_available", H4_MATCHED_REASON),
    (
        "cortical depth, MENDER, the shared mask",
        "unchanged inputs (not count-based)",
        "not applicable",
    ),
)


def label_metric(labels: pd.DataFrame, metric: str) -> float:
    """A label-table measure over some table cells (``SELECTION_METRICS``).

    Args:
        labels: Label rows of the cells.
        metric: ``coverage_<level>`` (the confident share at the level; H7
            at broad) or ``H2_implausible_share`` (``flag_implausible``, H2).

    Returns:
        The share (NaN without cells or without the level's column).

    Raises:
        ValueError: For an unknown metric.
    """
    if metric == "H2_implausible_share":
        column = "flag_implausible"
    elif metric.startswith("coverage_"):
        column = f"ct_{metric.removeprefix('coverage_')}_status"
    else:
        raise ValueError(f"unknown metric {metric!r}")
    if not len(labels) or column not in labels:
        return math.nan
    if column == "flag_implausible":
        return float(labels[column].to_numpy(bool).mean())
    return float((labels[column].astype(str).to_numpy() == CONFIDENT).mean())


def selection_depth_rows(
    reseg: pd.DataFrame,
    hybrid: pd.DataFrame,
    matched: pd.DataFrame,
    *,
    pair: str,
    platform: str,
) -> list[dict[str, Any]]:
    """Split ``reseg - hybrid`` of the label measures by cell set and depth.

    With R and H the reseg and proseg_hybrid table cells and S = R & H (the
    hybrid_matched table cells), ``reseg(R) - hybrid(H)`` is the sum of the
    cell selection ``(hybrid(S) - hybrid(H)) + (reseg(R) - reseg(S))``, the
    depth ``hybrid_matched(S) - hybrid(S)`` (the same cells thinned) and the
    rest ``reseg(S) - hybrid_matched(S)`` (the same cells at the same depth).

    Args:
        reseg: reseg's table cells' label rows (index ``cell_id``).
        hybrid: proseg_hybrid's.
        matched: hybrid_matched's.
        pair: Pair.
        platform: Platform.

    Returns:
        One row per ``SELECTION_METRICS`` entry.
    """
    shared = reseg.index.intersection(hybrid.index)
    hybrid_only = hybrid.index.difference(shared)
    reseg_only = reseg.index.difference(shared)
    medians = {
        "median_counts_hybrid_only": hybrid.loc[hybrid_only, "total_counts"],
        "median_counts_shared_hybrid": hybrid.loc[shared, "total_counts"],
        "median_counts_shared_reseg": reseg.loc[shared, "total_counts"],
        "median_counts_shared_matched": matched.reindex(shared)["total_counts"],
    }
    base: dict[str, Any] = {
        "pair": pair,
        "role": role(pair),
        "platform": platform,
        "n_reseg_table": len(reseg),
        "n_hybrid_table": len(hybrid),
        "n_shared": len(shared),
        "n_hybrid_only": len(hybrid_only),
        "n_reseg_only": len(reseg_only),
        "n_matched_table": len(matched),
        "matched_table_is_shared": set(matched.index) == set(shared),
        **{
            name: float(values.median()) if len(values) else math.nan
            for name, values in medians.items()
        },
    }
    rows = []
    for metric in SELECTION_METRICS:
        value = {
            "reseg_all": label_metric(reseg, metric),
            "reseg_shared": label_metric(reseg.loc[shared], metric),
            "reseg_only": label_metric(reseg.loc[reseg_only], metric),
            "hybrid_all": label_metric(hybrid, metric),
            "hybrid_shared": label_metric(hybrid.loc[shared], metric),
            "hybrid_only": label_metric(hybrid.loc[hybrid_only], metric),
            "matched_shared": label_metric(
                matched.loc[matched.index.intersection(shared)], metric
            ),
        }
        headline = value["reseg_all"] - value["hybrid_all"]
        parts = {
            "part_cell_selection": (value["hybrid_shared"] - value["hybrid_all"])
            + (value["reseg_all"] - value["reseg_shared"]),
            "part_depth": value["matched_shared"] - value["hybrid_shared"],
            "part_same_cells_same_depth": value["reseg_shared"]
            - value["matched_shared"],
        }
        rows.append(
            {
                **base,
                "metric": metric,
                **value,
                "headline_reseg_minus_hybrid": headline,
                "same_cells_reseg_minus_hybrid": value["reseg_shared"]
                - value["hybrid_shared"],
                **parts,
                **{
                    f"share_of_headline_{name.removeprefix('part_')}": (
                        part / headline if headline else math.nan
                    )
                    for name, part in parts.items()
                },
                "hybrid_matched_closes_share_of_headline": (
                    (value["matched_shared"] - value["hybrid_all"]) / headline
                    if headline
                    else math.nan
                ),
                "note": POSTHOC_NOTE,
            }
        )
    return rows


def ci_side(low: Any, high: Any) -> str:
    """Where a 95% CI lies: ``below 0``, ``above 0``, ``spans 0`` or ``no CI``."""
    if not (is_number(low) and is_number(high)):
        return "no CI"
    if float(high) < 0:
        return "below 0"
    if float(low) > 0:
        return "above 0"
    return "spans 0"


def jsd_reading(jsd: pd.DataFrame) -> pd.DataFrame:
    """Analysis 1's paired JSD differences, wide by comparison, with the CI side.

    Args:
        jsd: ``analysis1_jsd_paired``.

    Returns:
        One row per pair, kind and region: reseg's JSD, each other arm's,
        ``reseg - arm`` with its paired CI and the CI's side of 0 (below 0:
        reseg's two platforms agree more).
    """
    part = jsd[
        (jsd["first"] == "reseg") & jsd["second"].isin(["hybrid", "hybrid_matched"])
    ]
    rows: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in part.to_dict("records"):
        key = (record["pair"], record["kind"], record["region"])
        row = rows.setdefault(
            key,
            {
                "pair": record["pair"],
                "role": role(str(record["pair"])),
                "kind": record["kind"],
                "region": record["region"],
                "jsd_reseg": record["jsd_first"],
            },
        )
        other = record["second"]
        low, high = record.get("difference_ci_low"), record.get("difference_ci_high")
        row[f"jsd_{other}"] = record["jsd_second"]
        row[f"reseg_minus_{other}"] = record["difference"]
        row[f"reseg_minus_{other}_ci_low"] = low
        row[f"reseg_minus_{other}_ci_high"] = high
        row[f"reseg_minus_{other}_ci_side"] = ci_side(low, high)
    frame = pd.DataFrame(list(rows.values()))
    frame["reading"] = (
        "below 0: reseg's platforms agree more; above 0: less; spans 0: no "
        "difference at 95%"
    )
    frame["note"] = POSTHOC_NOTE
    return frame


def spillover_summary(
    bins: pd.DataFrame, *, analysis: str, neighbours: str
) -> pd.DataFrame:
    """Count analysis 3's rows by the side of 0 of their CI.

    Distance-bin rows and whole-class rows (``distance_bin = all``) are
    counted apart. Near minus far is, per class and dataset, the hybrid -
    reseg difference in the 0-10 µm bin minus that in the > 50 µm bin (rows
    with both).

    Args:
        bins: ``analysis3_bins`` (or ``posthoc_common_bins``).
        analysis: Which bins these are (a short name).
        neighbours: How their distances were measured.

    Returns:
        One row per cell set, metric and scope (all datasets, development,
        held-out).
    """
    rows = []
    for (cells, metric), part in bins.groupby(["cells", "metric"], sort=False):
        for scope, subset in (
            ("all datasets", part),
            ("development", part[part["role"] == "development"]),
            ("held_out", part[part["role"] == "held_out"]),
        ):
            in_bins = subset[subset["distance_bin"].astype(str) != "all"]
            whole = subset[subset["distance_bin"].astype(str) == "all"]
            wide = in_bins.pivot_table(
                index=["pair", "platform", "class"],
                columns="distance_bin",
                values="hybrid_minus_reseg",
            )
            near_far = (
                (wide[NEAR_BIN] - wide[FAR_BIN]).dropna()
                if {NEAR_BIN, FAR_BIN} <= set(wide.columns)
                else pd.Series(dtype=np.float64)
            )
            rows.append(
                {
                    "analysis": analysis,
                    "neighbours": neighbours,
                    "cells": cells,
                    "metric": metric,
                    "scope": scope,
                    "n_bin_rows": len(in_bins),
                    "n_bin_ci_above_0": int((in_bins["ci_low"] > 0).sum()),
                    "n_bin_ci_below_0": int((in_bins["ci_high"] < 0).sum()),
                    "n_class_rows": len(whole),
                    "n_class_ci_above_0": int((whole["ci_low"] > 0).sum()),
                    "n_class_ci_below_0": int((whole["ci_high"] < 0).sum()),
                    "n_class_difference_positive": int(
                        (whole["hybrid_minus_reseg"] > 0).sum()
                    ),
                    "median_class_mean_hybrid": float(whole["mean_hybrid"].median()),
                    "median_class_mean_reseg": float(whole["mean_reseg"].median()),
                    "median_class_difference": float(
                        whole["hybrid_minus_reseg"].median()
                    ),
                    "n_near_far_rows": len(near_far),
                    "median_near_minus_far": float(near_far.median())
                    if len(near_far)
                    else math.nan,
                    "share_near_minus_far_positive": float((near_far > 0).mean())
                    if len(near_far)
                    else math.nan,
                    "note": POSTHOC_NOTE,
                }
            )
    return pd.DataFrame(rows)


def leaking_genes(genes: pd.DataFrame) -> pd.DataFrame:
    """Rank analysis 3's negative genes by their median detection-rate gain.

    Args:
        genes: ``analysis3_genes`` (the ``both_confident`` rows are used).

    Returns:
        One row per gene: the median, mean and largest hybrid - reseg
        detection-rate difference over its class x dataset rows, where the
        largest is, and how many rows have their CI above / below 0; ranked
        by the median (largest first).
    """
    part = genes[genes["cells"] == "both_confident"]
    rows = []
    for gene, group in part.groupby("gene", sort=True):
        difference = group["hybrid_minus_reseg"].astype(np.float64)
        top = group.loc[difference.idxmax()] if difference.notna().any() else None
        rows.append(
            {
                "gene": gene,
                "n_rows": len(group),
                "n_datasets": len(group[["pair", "platform"]].drop_duplicates()),
                "classes_where_negative": "; ".join(
                    sorted(set(group["class"].astype(str)))
                ),
                "median_difference": float(difference.median()),
                "mean_difference": float(difference.mean()),
                "max_difference": float(difference.max()),
                "max_row": (
                    f"{top['pair']} {top['platform']} {top['class']}"
                    if top is not None
                    else ""
                ),
                "median_detection_hybrid": float(group["detection_hybrid"].median()),
                "median_detection_reseg": float(group["detection_reseg"].median()),
                "n_ci_above_0": int((group["ci_low"] > 0).sum()),
                "n_ci_below_0": int((group["ci_high"] < 0).sum()),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.sort_values(
        ["median_difference", "gene"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    frame.insert(0, "rank_by_median", np.arange(1, len(frame) + 1))
    frame["note"] = POSTHOC_NOTE
    return frame


def noise_stratum(ratio: float) -> str | None:
    """A gene's stratum by its decoded count over the median control's."""
    if not math.isfinite(ratio):
        return None
    if ratio <= NOISE_LOW:
        return NOISE_STRATA[0]
    if ratio < NOISE_HIGH:
        return NOISE_STRATA[1]
    return NOISE_STRATA[2]


def pooled_shares(frame: pd.DataFrame) -> dict[str, float]:
    """Analysis 2's shares pooled over a set of features (tally rows)."""

    def total(column: str) -> float:
        return float(frame[column].sum())

    decoded = total("n_decoded")
    nucleus = total("n_in_nucleus")
    reseg = total("n_assigned_reseg") / decoded if decoded else math.nan
    hybrid = total("n_assigned_hybrid") / decoded if decoded else math.nan
    return {
        "pooled_A_reseg": reseg,
        "pooled_A_hybrid": hybrid,
        "pooled_L": 1.0 - reseg / hybrid if hybrid else math.nan,
        "pooled_N_bg": total("n_in_nucleus_unassigned_reseg") / nucleus
        if nucleus
        else math.nan,
        "in_nucleus_share": nucleus / decoded if decoded else math.nan,
    }


def _quantiles(values: pd.Series) -> dict[str, float]:
    finite = values.astype(np.float64).dropna()
    if not len(finite):
        return {"median_L": math.nan, "q10_L": math.nan, "q90_L": math.nan}
    return {
        "median_L": float(finite.median()),
        "q10_L": float(finite.quantile(0.1)),
        "q90_L": float(finite.quantile(0.9)),
    }


def noise_floor_tables(
    genes: pd.DataFrame,
    controls: Mapping[tuple[str, str], tuple[pd.DataFrame, Mapping[str, Any]]],
) -> dict[str, pd.DataFrame]:
    """The control features' assignment and analysis 2's loss by noise stratum.

    The control features (MERSCOPE's blank barcodes) carry no transcript of a
    gene, so what a segmentation assigns of them is noise. A gene's stratum
    is its decoded count over the median control feature's (``<= 2x``,
    ``2-10x``, ``>= 10x``); the other platform's L of the same genes is
    paired with it. The noise share of the table counts assumes every gene
    has the mean control feature's assigned count.

    Args:
        genes: ``analysis2_genes`` rows.
        controls: ``(pair, platform)`` to the control tally's ``all`` rows
            with ``segmentation_compare.loss_shares`` and its record; a
            dataset without one, or with no control transcript, is
            ``not_available``.

    Returns:
        ``noise_floor`` (one row per dataset) and ``noise_strata`` (one row
        per dataset with controls and stratum).
    """
    floor_rows, strata_rows = [], []
    for (pair, platform), dataset in genes.groupby(["pair", "platform"], sort=False):
        base = {"pair": pair, "role": role(str(pair)), "platform": platform}
        entry = controls.get((str(pair), str(platform)))
        tally = None if entry is None else entry[0]
        if tally is None or not len(tally) or not tally["n_decoded"].sum():
            note = (
                "not_available: no control tally"
                if entry is None
                else str(entry[1].get("note") or "not_available: no control transcript")
            )
            floor_rows.append({**base, "n_control_features": 0, "note": note})
            continue
        n_controls = len(tally)
        median_control = float(tally["n_decoded"].median())
        control_total = float(tally["n_decoded"].sum())
        gene_total = float(dataset["n_decoded"].sum())
        mean_control = {
            arm: float(tally[f"n_assigned_{arm}"].sum()) / n_controls
            for arm in ("reseg", "hybrid")
        }
        floor_rows.append(
            {
                **base,
                "n_control_features": n_controls,
                "n_control_decoded": int(control_total),
                "median_control_decoded": median_control,
                "control_share_of_decoded": control_total
                / (control_total + gene_total),
                **{
                    f"control_{key}": value
                    for key, value in pooled_shares(tally).items()
                },
                "control_median_L": float(tally["L"].median()),
                **{
                    f"genes_{key}": value
                    for key, value in pooled_shares(dataset).items()
                },
                **{
                    f"noise_share_of_table_counts_{arm}": mean_control[arm]
                    * len(dataset)
                    / float(dataset[f"n_assigned_{arm}"].sum())
                    for arm in ("reseg", "hybrid")
                },
                "note": POSTHOC_NOTE,
            }
        )
        ratio = dataset["n_decoded"].astype(np.float64) / (
            median_control if median_control > 0 else math.nan
        )
        stratum = ratio.map(noise_stratum)
        other = genes[(genes["pair"] == pair) & (genes["platform"] != platform)]
        other_platform = str(other["platform"].iloc[0]) if len(other) else ""
        other_l = other.set_index("gene")["L"]
        for name in (ALL_GENES, *NOISE_STRATA):
            members = dataset if name == ALL_GENES else dataset[stratum == name]
            loss = members["L"].astype(np.float64)
            rho = sc.spearman_bootstrap(loss, members["expression_level"])
            paired = sc.wilcoxon_paired(
                loss.to_numpy(),
                other_l.reindex(members["gene"].astype(str)).to_numpy(np.float64),
            )
            strata_rows.append(
                {
                    **base,
                    "stratum": name,
                    "n_genes": len(members),
                    "share_of_gene_transcripts": float(members["n_decoded"].sum())
                    / gene_total
                    if gene_total
                    else math.nan,
                    **_quantiles(loss),
                    "median_N_bg": float(members["N_bg"].median()),
                    **pooled_shares(members),
                    "rho_L_expression": rho.effect,
                    "rho_ci_low": rho.ci_low,
                    "rho_ci_high": rho.ci_high,
                    "rho_p_value": rho.p_value,
                    "other_platform": other_platform,
                    "median_L_minus_other_platform": paired.median_difference,
                    "n_paired_genes": paired.n,
                    "wilcoxon_p_value": paired.p_value,
                    "note": POSTHOC_NOTE,
                }
            )
        strata_rows.append(
            {
                **base,
                "stratum": CONTROL_STRATUM,
                "n_genes": n_controls,
                **_quantiles(tally["L"]),
                "median_N_bg": float(tally["N_bg"].median()),
                **pooled_shares(tally),
                "note": POSTHOC_NOTE,
            }
        )
    return {
        "noise_floor": pd.DataFrame(floor_rows),
        "noise_strata": pd.DataFrame(strata_rows),
    }


def common_bins_sample(
    reseg: ArmData,
    hybrid: ArmData,
    *,
    xy_reseg: np.ndarray,
    xy_hybrid: np.ndarray,
    negatives: pd.DataFrame,
    pair: str,
    platform: str,
) -> dict[str, list[dict[str, Any]]]:
    """Analysis 3 with one distance per cell for both arms (post hoc).

    The cells with the same confident broad label in both arms; each cell's
    distance is from its proseg_hybrid centroid to the nearest proseg_hybrid
    confident broad cell of another class, and both arms use it, so a bin
    holds the same cells in each arm. The metrics, the tiles (proseg_hybrid
    centroids) and the bootstrap are analysis 3's.

    Args:
        reseg: reseg table cells (labels, counts).
        hybrid: proseg_hybrid table cells.
        xy_reseg: reseg native centroids of ``reseg.labels`` rows.
        xy_hybrid: proseg_hybrid native centroids of ``hybrid.labels`` rows.
        negatives: The primary bundle's ``negative_genes.parquet``.
        pair: Pair.
        platform: Platform.

    Returns:
        ``bins`` rows (analysis 3's columns) and one ``cells`` row.
    """
    classes = list(HUMAN_BROAD_CLASSES)
    names_r = pd.Series(
        confident_names(reseg.labels, "broad"), index=reseg.labels.index
    )
    names_h = pd.Series(
        confident_names(hybrid.labels, "broad"), index=hybrid.labels.index
    )
    both = names_r.dropna().index.intersection(names_h.dropna().index)
    same = both[(names_r.loc[both] == names_h.loc[both]).to_numpy(bool)]
    position_h = hybrid.labels.index.get_indexer(same)
    position_r = reseg.labels.index.get_indexer(same)
    distance = sc.nearest_other_class_distance(xy_hybrid, names_h.to_numpy(object))[
        position_h
    ]
    bins = sc.distance_bin_codes(distance)
    class_index = np.array(
        [classes.index(str(name)) for name in names_h.loc[same]], dtype=np.int64
    )
    gene_ids = [gene for gene in hybrid.gene_ids if gene]
    mask = negative_mask(negatives, gene_ids)
    measured: dict[str, dict[str, np.ndarray]] = {}
    for arm, data, position in (
        ("reseg", reseg, position_r),
        ("hybrid", hybrid, position_h),
    ):
        if data.counts is None:
            raise SystemExit(f"{arm}: counts not loaded")
        counts = align_genes(data.counts, data.gene_ids, gene_ids)
        measured[arm] = {
            "detected": (counts[position] > 0).toarray(),
            "contamination": data.labels["contamination_score"].to_numpy(np.float64)[
                position
            ],
        }
    xy = xy_hybrid[position_h]
    tiles = co.tile_codes(np.where(np.isfinite(xy), xy, xy_reseg[position_r]), TILE_UM)
    n_bins = len(sc.DISTANCE_BIN_LABELS)
    rows = []
    for c_index, cls in enumerate(classes):
        member = class_index == c_index
        negative_genes = np.flatnonzero(mask[c_index])
        whole = np.where(member, 0, -1)
        for metric in ("contamination_score", "negative_detection_rate"):
            values = {}
            for arm in ("reseg", "hybrid"):
                if metric == "contamination_score":
                    value = measured[arm]["contamination"]
                elif len(negative_genes):
                    value = measured[arm]["detected"][:, negative_genes].mean(axis=1)
                else:
                    value = np.full(len(member), math.nan)
                values[arm] = np.where(member, value, math.nan)
            result = sc.tile_bootstrap_ratio(
                values["reseg"], bins, values["hybrid"], bins, tiles, n_bins
            )
            overall = sc.tile_bootstrap_ratio(
                values["reseg"], whole, values["hybrid"], whole, tiles, 1
            )
            for b_index, label in enumerate((*sc.DISTANCE_BIN_LABELS, "all")):
                source, index = (overall, 0) if label == "all" else (result, b_index)
                rows.append(
                    {
                        "pair": pair,
                        "role": role(pair),
                        "platform": platform,
                        "cells": "same_label",
                        "class": cls,
                        "metric": metric,
                        "distance_bin": label,
                        "n_reseg": int(source.n_a[index]),
                        "n_hybrid": int(source.n_b[index]),
                        "mean_reseg": source.mean_a[index],
                        "mean_hybrid": source.mean_b[index],
                        "hybrid_minus_reseg": source.difference[index],
                        "ci_low": source.ci_low[index],
                        "ci_high": source.ci_high[index],
                        "n_tiles": source.n_tiles,
                        "n_negative_genes": int(len(negative_genes)),
                        "neighbours": COMMON_NEIGHBOURS,
                    }
                )
    cells = {
        "pair": pair,
        "role": role(pair),
        "platform": platform,
        "n_same_label": int(len(same)),
        "n_with_distance": int(np.isfinite(distance).sum()),
        "median_distance_common": float(np.nanmedian(distance))
        if np.isfinite(distance).any()
        else math.nan,
        "n_tiles": int(tiles.max() + 1) if len(tiles) else 0,
        "neighbours": COMMON_NEIGHBOURS,
        "note": POSTHOC_NOTE,
    }
    return {"bins": rows, "cells": [cells]}


def matched_count_sources() -> pd.DataFrame:
    """Which hybrid_matched items read the thinned counts (§19 literal)."""
    return pd.DataFrame(
        [
            {
                "item": item,
                "counts_read": counts,
                "depth_matched": matched,
                "note": "pre-registration §19: MAP / RESOLVE on the thinned "
                "counts, every other input unchanged",
            }
            for item, counts, matched in MATCHED_COUNT_SOURCES
        ]
    )


def primary_negatives(locations: Locations, store: Path, pair: str) -> pd.DataFrame:
    """The primary bundle's ``negative_genes.parquet`` of a pair's run."""
    summary = read_json(locations.summary_path("hybrid", pair))
    first = next(iter(summary["samples"].values()))
    build = first["bundles"][PRIMARY_REFERENCE]["resolved_build_hash"]
    return pd.read_parquet(store / PRIMARY_REFERENCE / build / "negative_genes.parquet")


def control_tallies(
    locations: Locations, pairs: Sequence[str]
) -> dict[tuple[str, str], tuple[pd.DataFrame, dict[str, Any]]]:
    """The cached control tallies (``all`` rows with the analysis 2 shares).

    A sample without a control tally whose analysis 2 tally removed no
    control feature gets an empty tally and that reason.
    """
    out: dict[tuple[str, str], tuple[pd.DataFrame, dict[str, Any]]] = {}
    for pair in pairs:
        for platform in PLATFORMS:
            cache = locations.controls_cache(pair, platform)
            if not (cache / "tally.parquet").is_file():
                record = locations.transcripts_cache(pair, platform) / (
                    "tally_record.json"
                )
                if record.is_file() and not read_json(record)["filter"].get(
                    "n_control_removed"
                ):
                    out[(pair, platform)] = (
                        pd.DataFrame(columns=list(sc.TALLY_FIELDS)),
                        {"note": NO_CONTROLS},
                    )
                continue
            tally = pd.read_parquet(cache / "tally.parquet")
            overall = tally[tally["level"] == "all"].drop(columns=["level", "group"])
            out[(pair, platform)] = (
                sc.loss_shares(overall.reset_index(drop=True)),
                read_json(cache / "tally_record.json"),
            )
    return out


def posthoc(
    locations: Locations, pairs: Sequence[str], *, store: Path
) -> dict[str, pd.DataFrame]:
    """The post-hoc tables (after the review; not pre-registered)."""
    directory = locations.result_dir()
    needed = {
        name: _read_table(directory, name)
        for name in (
            "analysis1_jsd_paired",
            "analysis2_genes",
            "analysis3_bins",
            "analysis3_genes",
        )
    }
    missing = [name for name, frame in needed.items() if frame is None]
    if missing:
        raise SystemExit(f"run analyze first: {missing} missing in {directory}")
    selection = []
    for pair in pairs:
        for platform in PLATFORMS:
            frames = {}
            for arm in MAIN_ARMS:
                labels = read_labels(
                    locations.labels_path(arm, pair, platform),
                    SELECTION_LABEL_COLUMNS,
                )
                frames[arm] = labels[labels["in_table"].to_numpy(bool)].set_index(
                    "cell_id", drop=False
                )
            selection += selection_depth_rows(
                frames["reseg"],
                frames["hybrid"],
                frames["hybrid_matched"],
                pair=pair,
                platform=platform,
            )
    inputs = {name: frame for name, frame in needed.items() if frame is not None}
    genes2 = inputs["analysis2_genes"]
    bins = inputs["analysis3_bins"]
    tables: dict[str, pd.DataFrame] = {"selection_depth": pd.DataFrame(selection)}
    tables["jsd_reading"] = jsd_reading(inputs["analysis1_jsd_paired"])
    tables |= noise_floor_tables(
        genes2[genes2["pair"].astype(str).isin(list(pairs))],
        control_tallies(locations, pairs),
    )
    common, cells = [], []
    for pair in pairs:
        negatives = primary_negatives(locations, store, pair)
        for platform in PLATFORMS:
            reseg = load_arm(locations, "reseg", pair, platform)
            hybrid = load_arm(locations, "hybrid", pair, platform)
            sample_store = locations.store_path(pair, platform)
            result = common_bins_sample(
                reseg,
                hybrid,
                xy_reseg=native_centroids(
                    sample_store, ARM_SHAPES["reseg"], reseg.labels.index
                ),
                xy_hybrid=native_centroids(
                    sample_store, ARM_SHAPES["hybrid"], hybrid.labels.index
                ),
                negatives=negatives,
                pair=pair,
                platform=platform,
            )
            common += result["bins"]
            cells += result["cells"]
            logger.info("posthoc common bins: %s %s done", pair, platform)
    common_bins = pd.DataFrame(common)
    tables["spillover_summary"] = pd.concat(
        [
            spillover_summary(bins, analysis="analysis 3", neighbours=OWN_NEIGHBOURS),
            spillover_summary(
                common_bins, analysis="common bins", neighbours=COMMON_NEIGHBOURS
            ),
        ],
        ignore_index=True,
    )
    tables["leaking_genes"] = leaking_genes(inputs["analysis3_genes"])
    tables["common_bins"] = common_bins
    tables["common_bins_cells"] = pd.DataFrame(cells)
    tables["matched_count_sources"] = matched_count_sources()
    return tables


# --------------------------------------------------------------------------
# analyze / render


def write_tables(
    directory: Path, stem: str, tables: Mapping[str, pd.DataFrame]
) -> Path:
    """Write ``<stem>.csv`` (all tables, ``table`` column) and ``tables/``."""
    parts = []
    for name, frame in tables.items():
        write_table(frame, directory / "tables" / f"{stem}_{name}.csv")
        part = frame.copy()
        part.insert(0, "table", name)
        parts.append(part)
    combined = (
        pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()
    )
    return write_table(combined, directory / f"{stem}.csv")


def write_analysis(
    directory: Path, number: int, tables: Mapping[str, pd.DataFrame]
) -> Path:
    """Write ``analysis<N>.csv`` (all tables, ``table`` column) and ``tables/``."""
    return write_tables(directory, f"analysis{number}", tables)


def xenium_gene_panels(samplesheet: Path | None) -> dict[str, Path]:
    """Pair -> its Xenium ``gene_panel.json`` (from the samplesheet's xenium_dir)."""
    if samplesheet is None:
        return {}
    frame = pd.read_csv(samplesheet)
    out = {}
    for row in frame.to_dict("records"):
        directory = row.get("xenium_dir")
        if (
            isinstance(directory, str)
            and (Path(directory) / "gene_panel.json").is_file()
        ):
            out[str(row["pair_id"])] = Path(directory) / "gene_panel.json"
    return out


def command_analyze(args: argparse.Namespace) -> int:
    """Run analyses 1-4 into ``<out>/segmentation_comparison``."""
    locations = locations_from(args)
    directory = locations.result_dir()
    directory.mkdir(parents=True, exist_ok=True)
    pairs = _pairs(args)
    wanted = {int(item) for item in args.analyses.split(",") if item}
    store = Path(args.store)
    run: dict[str, Any] = {"args": _run_args(args), "analyses": {}}
    if 1 in wanted:
        started = time.time()
        write_analysis(directory, 1, analysis1(locations, pairs))
        run["analyses"]["1"] = {"wall_s": round(time.time() - started, 1)}
    if 2 in wanted:
        started = time.time()
        gtf = Path(args.gene_annotation_gtf) if args.gene_annotation_gtf else None
        tables = analysis2(
            locations,
            pairs,
            store=store,
            gtf=gtf,
            xenium_panels=xenium_gene_panels(
                Path(args.samplesheet) if args.samplesheet else None
            ),
        )
        write_analysis(directory, 2, tables)
        run["analyses"]["2"] = {
            "wall_s": round(time.time() - started, 1),
            "gene_annotation_gtf": None if gtf is None else str(gtf),
        }
    if 3 in wanted:
        started = time.time()
        write_analysis(directory, 3, analysis3(locations, pairs, store=store))
        run["analyses"]["3"] = {"wall_s": round(time.time() - started, 1)}
    if 4 in wanted:
        started = time.time()
        write_analysis(directory, 4, analysis4(locations, pairs, store=store))
        run["analyses"]["4"] = {"wall_s": round(time.time() - started, 1)}
    run["code"] = _code_record(Path(args.export) if args.export else None)
    write_json(
        run, directory / f"analyze_run_{'_'.join(str(n) for n in sorted(wanted))}.json"
    )
    return 0


def command_posthoc(args: argparse.Namespace) -> int:
    """The post-hoc tables -> ``posthoc.csv`` and ``tables/posthoc_*.csv``."""
    locations = locations_from(args)
    directory = locations.result_dir()
    started = time.time()
    write_tables(
        directory,
        "posthoc",
        posthoc(locations, _pairs(args), store=Path(args.store)),
    )
    write_json(
        {
            "args": _run_args(args),
            "note": POSTHOC_NOTE,
            "wall_s": round(time.time() - started, 1),
            "code": _code_record(Path(args.export) if args.export else None),
        },
        directory / "posthoc_run.json",
    )
    return 0


def _run_args(args: argparse.Namespace) -> dict[str, Any]:
    """The parsed arguments without the handler (whose repr is an address)."""
    return {key: value for key, value in vars(args).items() if not callable(value)}


def _code_record(export: Path | None) -> dict[str, Any]:
    record: dict[str, Any] = {"script_sha256": file_sha256(Path(__file__))}
    if export is not None and (export / "COMMIT").is_file():
        record["commit"] = (export / "COMMIT").read_text(encoding="utf-8").strip()
    record["helpers_sha256"] = file_sha256(Path(sc.__file__))
    return record


# Rendering: figures (PNG + PDF + CSV) and the static page.

# Arms: the reference palette's slots 1-4 (validated with the dataviz
# validator: adjacent CVD dE >= 9.1, normal-vision >= 22.9 on the light
# surface). Other series (pairs, metrics) take slots 7, 5, 6 and 8 in that
# order, never an arm's hue; their first three pass all checks, the fourth
# is in the CVD warn band (7.2), so a fourth series always has a second
# encoding. Slots 3-4 and 5 are below 3:1 contrast: every figure has its
# table (the CSV beside it and a table on the page). A series keeps its
# colour in every figure; a dataset takes its pair's colour and Xenium bars
# are hatched.
ARM_PALETTE: Final[tuple[str, ...]] = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
OTHER_PALETTE: Final[tuple[str, ...]] = ("#4a3aa7", "#e87ba4", "#008300", "#e34948")
SERIES_COLOURS: Final = ARM_PALETTE
ARM_COLOURS: Final[dict[str, str]] = {
    "hybrid": ARM_PALETTE[0],
    "reseg": ARM_PALETTE[1],
    "hybrid_matched": ARM_PALETTE[2],
    COMBINED: ARM_PALETTE[3],
}
# A paired difference "reseg - <arm>" takes the colour of the other arm.
COMPARISON_COLOURS: Final[dict[str, str]] = {
    f"reseg - {arm}": ARM_COLOURS[arm] for arm in ("hybrid", "hybrid_matched")
}


def series_colours(names: Iterable[str]) -> dict[str, str]:
    """Colour per series.

    An arm (or a ``reseg - <arm>`` comparison) has its own colour; a dataset
    ``<pair> <PLATFORM>`` takes its pair's colour; anything else takes the
    other palette's slots in order of first appearance.
    """
    out: dict[str, str] = {}
    slot: dict[str, int] = {}
    for name in dict.fromkeys(str(value) for value in names):
        if name in ARM_COLOURS:
            out[name] = ARM_COLOURS[name]
        elif name in COMPARISON_COLOURS:
            out[name] = COMPARISON_COLOURS[name]
        else:
            key = name.split(" ")[0] if name.split(" ")[0] in PAIRS else name
            slot.setdefault(key, len(slot))
            out[name] = OTHER_PALETTE[slot[key] % len(OTHER_PALETTE)]
    return out


def bar_figure(
    frame: pd.DataFrame,
    *,
    category: str,
    group: str,
    value: str,
    low: str | None = None,
    high: str | None = None,
    facet: str | None = None,
    ylabel: str = "",
    title: str = "",
    hatched: str | None = None,
    hatched_label: str = "",
) -> Any:
    """``report_figures.grouped_bars`` with the comparison's series colours."""
    from matplotlib.container import BarContainer

    from merxen.annotation.report_figures import grouped_bars

    colours = series_colours(frame[group].astype(str))
    figure = grouped_bars(
        frame,
        category=category,
        group=group,
        value=value,
        low=low,
        high=high,
        facet=facet,
        ylabel=ylabel,
        title=title,
        horizontal_line=0.0,
        hatched=hatched,
        hatched_label=hatched_label,
    )
    for ax in figure.axes:
        for container in ax.containers:
            if isinstance(container, BarContainer) and container.get_label() in colours:
                for patch in container:
                    patch.set_facecolor(colours[container.get_label()])
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.5)
        ax.set_axisbelow(True)
        legend = ax.get_legend()
        if legend is not None:
            for handle, text in zip(
                legend.legend_handles, legend.get_texts(), strict=False
            ):
                if handle is not None and text.get_text() in colours:
                    handle.set_facecolor(colours[text.get_text()])
    if title:
        figure.tight_layout(rect=(0, 0, 1, 1 - 0.45 / figure.get_figheight()))
    return figure


def forest_figure(
    frame: pd.DataFrame,
    *,
    label: str,
    group: str,
    value: str,
    low: str,
    high: str,
    title: str,
    xlabel: str,
) -> Any:
    """Point estimates with CIs per label, one colour per group, a zero line."""
    from merxen.annotation.report_figures import new_figure, placeholder

    if frame.empty:
        return placeholder(f"{title}: no data")
    labels = list(dict.fromkeys(frame[label].astype(str)))
    groups = list(dict.fromkeys(frame[group].astype(str)))
    colours = series_colours(groups)
    figure = new_figure(6.5, 0.3 * len(labels) + 1.4)
    ax = figure.add_subplot(1, 1, 1)
    ax.axvline(0.0, color="#999999", linewidth=0.8, linestyle="--")
    step = 0.6 / max(1, len(groups))
    for index, name in enumerate(groups):
        part = frame[frame[group].astype(str) == name]
        y = (
            np.array([labels.index(str(v)) for v in part[label]])
            - 0.3
            + step * (index + 0.5)
        )
        centre = part[value].astype(float).to_numpy()
        lows = part[low].astype(float).to_numpy()
        highs = part[high].astype(float).to_numpy()
        ax.errorbar(
            centre,
            y,
            xerr=np.vstack([centre - lows, highs - centre]).clip(0),
            fmt="o",
            color=colours[name],
            markersize=4,
            elinewidth=1.0,
            label=name,
        )
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", color="#e5e5e5", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title(title, fontsize=9)
    figure.tight_layout()
    return figure


def _read_table(directory: Path, name: str) -> pd.DataFrame | None:
    path = directory / "tables" / f"{name}.csv"
    return pd.read_csv(path) if path.is_file() else None


def render_figures(directory: Path) -> list[Any]:
    """Draw the page's figures from the analysis tables (PNG + PDF + CSV)."""
    from merxen.annotation.report_figures import save_report_figure

    figures_dir = directory / "figures"
    records = []
    roles = (("development", "development pairs"), ("held_out", "held-out donors"))
    depth = _read_table(directory, "analysis1_depth_bins")
    if depth is not None and len(depth):
        part = depth[
            (depth["level"] == "broad")
            & (depth["depth_bin"].astype(str) != "all")
            & depth["arm"].isin(MAIN_ARMS)
        ].copy()
        part["dataset"] = part["pair"] + " " + part["platform"]
        for value, text in roles:
            sub = part[part["role"] == value]
            if len(sub):
                figure = bar_figure(
                    sub,
                    category="depth_bin",
                    group="arm",
                    value="coverage",
                    facet="dataset",
                    ylabel="confident broad / table cells",
                    title=f"Analysis 1: confident broad coverage by depth bin ({text})",
                )
                records.append(
                    save_report_figure(
                        figure,
                        figures_dir,
                        f"a1_broad_coverage_depth_{value}",
                        sub,
                        "Confident broad coverage of the table cells per depth bin "
                        "(total counts), per arm.",
                    )
                )
    jsd = _read_table(directory, "analysis1_jsd_paired")
    if jsd is not None and len(jsd):
        part = jsd[
            jsd["second"].isin(["hybrid", "hybrid_matched"])
            & (jsd["region"] == "whole_section")
        ].copy()
        if len(part):
            part["comparison"] = "reseg - " + part["second"]
            part["label"] = part["pair"] + " " + part["kind"]
            figure = forest_figure(
                part,
                label="label",
                group="comparison",
                value="difference",
                low="difference_ci_low",
                high="difference_ci_high",
                title="Analysis 1: cross-platform broad JSD, reseg minus hybrid arms",
                xlabel="JSD difference (paired 95% tile CI; < 0: reseg closer)",
            )
            records.append(
                save_report_figure(
                    figure,
                    figures_dir,
                    "a1_jsd_paired",
                    part,
                    "Paired difference of the MERSCOPE-Xenium broad JSD (whole "
                    "section), reseg minus hybrid and minus hybrid_matched, with "
                    "the paired 500 µm tile CI.",
                )
            )
    genes = _read_table(directory, "analysis2_genes")
    if genes is not None and len(genes):
        rows = []
        for (pair, value, platform), part in genes.groupby(
            ["pair", "role", "platform"]
        ):
            for metric in LOSS_METRICS:
                column = part[metric].astype(float)
                rows.append(
                    {
                        "pair": pair,
                        "role": value,
                        "platform": platform,
                        "dataset": f"{pair} {platform}",
                        "metric": metric,
                        "median": column.median(),
                        "q10": column.quantile(0.1),
                        "q90": column.quantile(0.9),
                    }
                )
        long = pd.DataFrame(rows)
        figure = bar_figure(
            long,
            category="dataset",
            group="metric",
            value="median",
            low="q10",
            high="q90",
            ylabel="median over genes (whiskers q10-q90)",
            title="Analysis 2: reseg loss L and in-nucleus unassigned share N_bg",
        )
        records.append(
            save_report_figure(
                figure,
                figures_dir,
                "a2_loss_per_dataset",
                long,
                "Per gene: L = 1 - A_reseg / A_hybrid and N_bg (share of in-nucleus "
                "transcripts reseg leaves unassigned); median over the panel genes, "
                "whiskers q10-q90.",
            )
        )
    by_type = _read_table(directory, "analysis2_genes_by_type")
    if by_type is not None and len(by_type):
        broad = by_type[by_type["level"] == "broad"]
        keys = ["pair", "role", "platform", "group"]
        pooled = (
            broad.groupby(keys, as_index=False)[
                [
                    "n_assigned_reseg",
                    "n_assigned_hybrid",
                    "n_in_nucleus",
                    "n_in_nucleus_unassigned_reseg",
                ]
            ]
            .sum()
            .rename(columns={"group": "class"})
        )
        hybrid = pooled["n_assigned_hybrid"].where(pooled["n_assigned_hybrid"] > 0)
        nucleus = pooled["n_in_nucleus"].where(pooled["n_in_nucleus"] > 0)
        pooled = pd.concat(
            [
                pooled.assign(
                    metric="L", value=1.0 - pooled["n_assigned_reseg"] / hybrid
                ),
                pooled.assign(
                    metric="N_bg",
                    value=pooled["n_in_nucleus_unassigned_reseg"] / nucleus,
                ),
            ],
            ignore_index=True,
        )
        medians = (
            broad.groupby(keys, as_index=False)
            .agg(L=("L", "median"), N_bg=("N_bg", "median"))
            .rename(columns={"group": "class"})
            .melt(id_vars=["pair", "role", "platform", "class"], var_name="metric")
        )
        for kind, frame, ylabel, caption in (
            (
                "pooled",
                pooled,
                "pooled over the panel genes",
                "L and N_bg of all panel transcripts of each confident proseg_hybrid "
                "broad class's cells (pooled over genes).",
            ),
            (
                "median",
                medians,
                "median over the panel genes",
                "Median over the panel genes of the per-gene L and N_bg within each "
                "confident proseg_hybrid broad class (genes a class does not express "
                "count as much as its markers; per gene in "
                "analysis2_genes_by_type.csv).",
            ),
        ):
            frame = frame.copy()
            frame["dataset"] = frame["pair"] + " " + frame["platform"]
            for value, text in roles:
                sub = frame[frame["role"] == value]
                if not len(sub):
                    continue
                figure = bar_figure(
                    sub,
                    category="class",
                    group="metric",
                    value="value",
                    facet="dataset",
                    ylabel=ylabel,
                    title=(
                        f"Analysis 2: loss per confident proseg_hybrid broad class, "
                        f"{kind} ({text})"
                    ),
                )
                records.append(
                    save_report_figure(
                        figure,
                        figures_dir,
                        f"a2_loss_by_class_{kind}_{value}",
                        sub,
                        caption,
                    )
                )
    bins = _read_table(directory, "analysis3_bins")
    if bins is not None and len(bins):
        for metric, text_metric in (
            ("contamination_score", "contamination score"),
            ("negative_detection_rate", "negative-gene detection rate"),
        ):
            part = bins[
                (bins["cells"] == "both_confident")
                & (bins["metric"] == metric)
                & (bins["distance_bin"] != "all")
            ].copy()
            part["dataset"] = part["pair"] + " " + part["platform"]
            part["xenium"] = part["platform"] == "XENIUM"
            for value, text in roles:
                sub = part[part["role"] == value]
                if len(sub):
                    figure = bar_figure(
                        sub,
                        category="distance_bin",
                        group="dataset",
                        hatched="xenium",
                        hatched_label="Xenium (hatched)",
                        value="hybrid_minus_reseg",
                        low="ci_low",
                        high="ci_high",
                        facet="class",
                        ylabel=f"hybrid - reseg {text_metric}",
                        title=(
                            f"Analysis 3: {text_metric} by distance to another "
                            f"class ({text})"
                        ),
                    )
                    records.append(
                        save_report_figure(
                            figure,
                            figures_dir,
                            f"a3_{metric}_{value}",
                            sub,
                            f"hybrid minus reseg mean {text_metric} per distance bin "
                            "(µm, centroid to the nearest confident cell of another "
                            "broad class) and class; 95% CI from 200 resamples of "
                            "500 µm tiles.",
                        )
                    )
    coverage = _read_table(directory, "analysis4_coverage")
    if coverage is not None and len(coverage):
        part = coverage[coverage["level"] == "broad"].copy()
        part["dataset"] = part["pair"] + " " + part["platform"]
        long = pd.concat(
            [
                part.assign(arm="hybrid", value=part["share_confident_hybrid"]),
                part.assign(arm=COMBINED, value=part["share_confident_combined"]),
            ],
            ignore_index=True,
        )
        figure = bar_figure(
            long,
            category="dataset",
            group="arm",
            value="value",
            ylabel="confident broad / proseg_hybrid table cells",
            title="Analysis 4: broad coverage of the proseg_hybrid table cells",
        )
        records.append(
            save_report_figure(
                figure,
                figures_dir,
                "a4_coverage",
                long,
                "Share of the proseg_hybrid table cells with a confident broad label: "
                "proseg_hybrid's own labels vs the combined mode (the reseg cell of "
                "the same id).",
            )
        )
    return records + posthoc_figures(directory)


HYBRID_ONLY_SERIES: Final = "hybrid-only cells (hybrid)"


def legend_above_bars(figure: Any, top: float) -> Any:
    """Raise the y limit to ``top`` and put the legend in one row at the top."""
    for ax in figure.axes:
        ax.set_ylim(0.0, top)
        legend = ax.get_legend()
        if legend is not None:
            ax.legend(
                handles=legend.legend_handles,
                labels=[text.get_text() for text in legend.get_texts()],
                loc="upper left",
                ncol=4,
                fontsize=7,
                frameon=False,
            )
    return figure


def posthoc_figures(directory: Path) -> list[Any]:
    """The post-hoc figures (same-cell coverage, loss by noise stratum)."""
    from merxen.annotation.report_figures import save_report_figure

    figures_dir = directory / "figures"
    records = []
    roles = (("development", "development pairs"), ("held_out", "held-out donors"))
    selection = _read_table(directory, "posthoc_selection_depth")
    if selection is not None and len(selection):
        part = selection[selection["metric"] == "coverage_broad"].copy()
        part["dataset"] = part["pair"] + " " + part["platform"]
        long = pd.concat(
            [
                part.assign(series="reseg", value=part["reseg_shared"]),
                part.assign(series="hybrid", value=part["hybrid_shared"]),
                part.assign(series="hybrid_matched", value=part["matched_shared"]),
                part.assign(series=HYBRID_ONLY_SERIES, value=part["hybrid_only"]),
            ],
            ignore_index=True,
        )
        for value, text in roles:
            sub = long[long["role"] == value]
            if not len(sub):
                continue
            figure = bar_figure(
                sub,
                category="dataset",
                group="series",
                value="value",
                ylabel="confident broad / cells",
                title=f"Post hoc: broad coverage on the cells in both tables ({text})",
            )
            legend_above_bars(figure, 1.1)
            records.append(
                save_report_figure(
                    figure,
                    figures_dir,
                    f"p_same_cell_coverage_{value}",
                    sub,
                    "Post hoc. Confident broad share of the cells in both tables "
                    "(the hybrid_matched table cells) under reseg, proseg_hybrid "
                    "and hybrid_matched, and of the proseg_hybrid table cells "
                    "without a reseg table cell under proseg_hybrid.",
                )
            )
    strata = _read_table(directory, "posthoc_noise_strata")
    if strata is not None and len(strata):
        part = strata[strata["stratum"] != ALL_GENES].copy()
        part["dataset"] = part["pair"] + " " + part["platform"]
        for value, text in roles:
            sub = part[part["role"] == value]
            if not len(sub):
                continue
            figure = bar_figure(
                sub,
                category="dataset",
                group="stratum",
                value="median_L",
                low="q10_L",
                high="q90_L",
                ylabel="median L over features (whiskers q10-q90)",
                title=f"Post hoc: reseg loss L by noise stratum ({text})",
            )
            legend_above_bars(figure, 1.2)
            records.append(
                save_report_figure(
                    figure,
                    figures_dir,
                    f"p_noise_strata_{value}",
                    sub,
                    "Post hoc. Per-feature L = 1 - A_reseg / A_hybrid of the genes "
                    "by their decoded count over the median blank barcode's (<= 2x, "
                    "2-10x, >= 10x) and of the blank barcodes themselves (MERSCOPE; "
                    "the Xenium stores carry no control feature).",
                )
            )
    return records


def _html_table(
    frame: pd.DataFrame | None, columns: Sequence[str] | None = None, max_rows: int = 80
) -> str:
    from merxen.annotation.report_html import frame_html

    if frame is None or frame.empty:
        return '<p class="muted">no rows</p>'
    shown = frame[[c for c in columns if c in frame.columns]] if columns else frame
    return frame_html(shown, max_rows=max_rows)


def _by_role(frame: pd.DataFrame | None, columns: Sequence[str] | None = None) -> str:
    """Tables of the development pairs and of the held-out donors, apart."""
    if frame is None or frame.empty or "role" not in frame:
        return _html_table(frame, columns)
    parts = []
    for value, title in (
        ("development", "Development pairs (P7513, P1212)"),
        ("held_out", "Held-out donors (P7113, P5011; reported separately)"),
    ):
        parts.append(
            f"<h4>{title}</h4>" + _html_table(frame[frame["role"] == value], columns)
        )
    return "".join(parts)


MEMO_NAME: Final = "M8B_DECISION_MEMO.txt"


def posthoc_sections(directory: Path, figure_html: Any) -> list[str]:
    """The page's post-hoc section (after the review; not pre-registered)."""

    def table(name: str) -> pd.DataFrame | None:
        return _read_table(directory, f"posthoc_{name}")

    selection = table("selection_depth")
    if selection is not None and len(selection):
        selection = selection[
            selection["metric"].isin(["coverage_broad", "H2_implausible_share"])
        ]
    leaking = table("leaking_genes")
    return [
        "<h2>Post-hoc additions after the review (not pre-registered)</h2>",
        f'<div class="banner"><b>Post hoc.</b> {html.escape(POSTHOC_NOTE)}. '
        "Analyses 1-4 above are unchanged. The memo's section 9 is the review; "
        "its section 10 says how each review item was handled.</div>",
        "<h3>The cells in both tables: broad coverage and H2 split into cell "
        "selection, depth and the rest</h3>",
        '<p class="muted">R, H: the reseg and proseg_hybrid table cells; S = R ∩ H '
        "(the hybrid_matched table cells). reseg(R) − hybrid(H) = cell selection "
        "[hybrid(S) − hybrid(H) + reseg(R) − reseg(S)] + depth [hybrid_matched(S) − "
        "hybrid(S)] + the rest [reseg(S) − hybrid_matched(S)].</p>",
        figure_html("p_same_cell"),
        _by_role(
            selection,
            [
                "pair",
                "platform",
                "metric",
                "n_shared",
                "n_hybrid_only",
                "median_counts_hybrid_only",
                "reseg_all",
                "hybrid_all",
                "reseg_shared",
                "hybrid_shared",
                "matched_shared",
                "hybrid_only",
                "headline_reseg_minus_hybrid",
                "same_cells_reseg_minus_hybrid",
                "part_cell_selection",
                "part_depth",
                "part_same_cells_same_depth",
                "share_of_headline_cell_selection",
            ],
        ),
        "<h3>Paired JSD differences: the side of 0 of each CI</h3>",
        _by_role(
            table("jsd_reading"),
            [
                "pair",
                "kind",
                "region",
                "jsd_reseg",
                "jsd_hybrid",
                "jsd_hybrid_matched",
                "reseg_minus_hybrid",
                "reseg_minus_hybrid_ci_side",
                "reseg_minus_hybrid_matched",
                "reseg_minus_hybrid_matched_ci_side",
            ],
        ),
        "<h3>MERSCOPE decoding noise: the blank barcodes</h3>",
        '<p class="muted">The blank barcodes encode no gene, so what a segmentation '
        "assigns of them is noise. Genes are put in strata by their decoded count "
        "over the median blank barcode's. The noise share of the table counts "
        "assumes every gene has the mean blank barcode's assigned count. The Xenium "
        "stores' transcripts carry no control feature.</p>",
        figure_html("p_noise"),
        _by_role(
            table("noise_floor"),
            [
                "pair",
                "platform",
                "n_control_features",
                "median_control_decoded",
                "control_pooled_A_reseg",
                "control_pooled_A_hybrid",
                "control_pooled_L",
                "control_pooled_N_bg",
                "control_in_nucleus_share",
                "genes_in_nucleus_share",
                "noise_share_of_table_counts_hybrid",
                "noise_share_of_table_counts_reseg",
                "note",
            ],
        ),
        _by_role(
            table("noise_strata"),
            [
                "pair",
                "platform",
                "stratum",
                "n_genes",
                "share_of_gene_transcripts",
                "median_L",
                "q10_L",
                "q90_L",
                "pooled_N_bg",
                "rho_L_expression",
                "rho_ci_low",
                "rho_ci_high",
                "median_L_minus_other_platform",
                "wilcoxon_p_value",
            ],
        ),
        "<h3>Spill-over: rows by the side of 0 of their CI, and the same cells in "
        "the same bins</h3>",
        _html_table(
            table("spillover_summary"),
            [
                "analysis",
                "cells",
                "metric",
                "scope",
                "n_bin_rows",
                "n_bin_ci_above_0",
                "n_bin_ci_below_0",
                "n_class_rows",
                "n_class_ci_above_0",
                "n_class_ci_below_0",
                "median_class_mean_hybrid",
                "median_class_mean_reseg",
                "n_near_far_rows",
                "median_near_minus_far",
                "share_near_minus_far_positive",
            ],
        ),
        _by_role(table("common_bins_cells")),
        "<h3>Negative genes ranked by their median detection gain under "
        "proseg_hybrid (top 25)</h3>",
        _html_table(None if leaking is None else leaking.head(25)),
        "<h3>hybrid_matched: which items read the thinned counts</h3>",
        _html_table(table("matched_count_sources")),
    ]


def render_page(directory: Path, figures: Sequence[Any]) -> Path:
    """Write ``index.html`` (inline CSS, no scripts, no network assets)."""
    from merxen.annotation.report_html import CSS

    def figure_html(prefix: str) -> str:
        parts = []
        for record in figures:
            if not record.stem.startswith(prefix):
                continue
            caption = html.escape(record.caption)
            parts.append(
                f'<figure><img src="figures/{html.escape(record.png.name)}" '
                f'alt="{caption}"><figcaption>{caption} '
                f'<a href="figures/{html.escape(record.pdf.name)}">PDF</a> · '
                f'<a href="figures/{html.escape(record.csv.name)}">CSV</a>'
                "</figcaption></figure>"
            )
        return "".join(parts)

    def table(name: str) -> pd.DataFrame | None:
        return _read_table(directory, name)

    depth = table("analysis1_depth_bins")
    if depth is not None and len(depth):
        depth = depth[
            (depth["level"] == "broad") & (depth["depth_bin"].astype(str) == "all")
        ]
    bins = table("analysis3_bins")
    if bins is not None and len(bins):
        bins = bins[
            (bins["distance_bin"] == "all") & (bins["cells"] == "both_confident")
        ]
    links = " ".join(
        f'<a href="{stem}.csv">{stem}.csv</a>'
        for stem in ("analysis1", "analysis2", "analysis3", "analysis4", "posthoc")
        if (directory / f"{stem}.csv").is_file()
    )
    memo = (
        f' The decision memo: <a href="{MEMO_NAME}">{MEMO_NAME}</a>.'
        if (directory / MEMO_NAME).is_file()
        else ""
    )
    sections = [
        "<h1>M8b segmentation comparison: reseg vs proseg_hybrid</h1>",
        '<p class="muted">Pre-registration §19. M8b scores nothing for gate H and '
        "fits no threshold; the user decides OD-B7′ (the default human segmentation "
        "for cell typing and for per-gene expression) from this evidence. Every table "
        f"is in <code>tables/</code> and in the per-analysis CSVs: {links}.{memo}</p>",
        '<div class="banner"><b>Arms.</b> reseg: ProSeg\'s own assignment. hybrid: '
        "proseg_hybrid. hybrid_matched: proseg_hybrid cells thinned to the reseg "
        "cell's total (same id), re-mapped and re-resolved with the production "
        "tasks. combined: reseg labels on proseg_hybrid counts (analysis 4). "
        f"{html.escape(MATCHED_NOTE)}.</div>",
        "<h2>Analysis 1: side-by-side scores</h2>",
        figure_html("a1_"),
        "<h3>§14 criterion values per arm</h3>",
        _by_role(
            table("analysis1_criteria"),
            [
                "criterion",
                "dataset",
                "value_reseg",
                "value_hybrid",
                "value_hybrid_matched",
                "reseg_minus_hybrid",
                "reseg_minus_hybrid_matched",
                "value_proseg_mask",
                "value_original_seg",
                "threshold_for_reference",
            ],
        ),
        "<h3>H12 (score_h12 on the square 500 µm tile CI)</h3>",
        _by_role(
            table("analysis1_h12"),
            ["pair", "arm", "verdict", "ordering_verdict", "wm_gm_verdict", "note"],
        ),
        "<h3>Confident broad coverage of the table cells (all depths)</h3>",
        _by_role(
            depth, ["pair", "platform", "arm", "n_table", "n_confident", "coverage"]
        ),
        "<h3>Paired JSD differences</h3>",
        _by_role(
            table("analysis1_jsd_paired"),
            [
                "pair",
                "kind",
                "region",
                "first",
                "second",
                "jsd_first",
                "jsd_second",
                "difference",
                "difference_ci_low",
                "difference_ci_high",
                "note",
            ],
        ),
        "<h2>Analysis 2: transcript-loss audit (reseg)</h2>",
        figure_html("a2_"),
        "<h3>Tests (one Benjamini–Hochberg family)</h3>",
        _by_role(
            table("analysis2_tests"),
            [
                "pair",
                "platform",
                "metric",
                "covariate",
                "test",
                "effect_name",
                "effect",
                "ci_low",
                "ci_high",
                "median_difference",
                "n",
                "n_other",
                "p_value",
                "q_value_bh",
                "note",
            ],
        ),
        "<h3>Effect on expression (cells confident in both arms, same label)</h3>",
        _by_role(table("analysis2_expression_summary")),
        "<h3>Gene lists (fixed before the first run)</h3>",
        _html_table(table("analysis2_gene_lists")),
        "<h2>Analysis 3: spill-over audit (proseg_hybrid)</h2>",
        figure_html("a3_"),
        "<h3>Cells and the median distance to a confident cell of another class "
        "(each arm's own neighbours)</h3>",
        _by_role(table("analysis3_summary")),
        "<h3>Per class, all distances</h3>",
        _by_role(
            bins,
            [
                "pair",
                "platform",
                "class",
                "metric",
                "n_reseg",
                "n_hybrid",
                "mean_reseg",
                "mean_hybrid",
                "hybrid_minus_reseg",
                "ci_low",
                "ci_high",
            ],
        ),
        "<h2>Analysis 4: combined mode</h2>",
        figure_html("a4_"),
        _by_role(
            table("analysis4_coverage"),
            [
                "pair",
                "platform",
                "level",
                "n_hybrid_table",
                "share_without_reseg_table_cell",
                "share_confident_combined",
                "share_confident_hybrid",
                "share_confident_reseg_of_reseg_table",
            ],
        ),
        "<h3>Cross-platform confident broad JSD (whole section, matched-tile CI)</h3>",
        _by_role(
            table("analysis4_jsd"),
            [
                "pair",
                "arm",
                "jsd",
                "ci_low",
                "ci_high",
                "difference",
                "difference_ci_low",
                "difference_ci_high",
            ],
        ),
        "<h3>Effect on expression per arm (each arm's own confident cells)</h3>",
        _by_role(table("analysis4_expression_summary")),
        "<h3>The pipeline change the mode would need (not implemented)</h3>",
        _html_table(table("analysis4_pipeline_change")),
    ]
    if (directory / "posthoc.csv").is_file():
        sections += posthoc_sections(directory, figure_html)
    page = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>Segmentation comparison M8b</title>"
        f"<style>{CSS}</style></head><body><main>{''.join(sections)}</main>"
        "</body></html>\n"
    )
    path = directory / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


def command_render(args: argparse.Namespace) -> int:
    """Draw the figures and write the page."""
    directory = locations_from(args).result_dir()
    figures = render_figures(directory)
    render_page(directory, figures)
    logger.info("rendered %d figures", len(figures))
    return 0


# --------------------------------------------------------------------------
# CLI


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--runs-root",
        required=True,
        help="M8 stage B runs (results/acceptance/<date>/runs)",
    )
    parser.add_argument(
        "--results-root", required=True, help="the published results (read-only)"
    )
    parser.add_argument(
        "--stage-b", required=True, help="the M8 stage B evidence directory"
    )
    parser.add_argument("--out", required=True, help="M8b's output root")
    parser.add_argument(
        "--export", default="", help="the code export the step runs from"
    )


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("matched-prepare")
    _common(prepare)
    prepare.add_argument("--pair", required=True, choices=PAIRS)
    prepare.add_argument("--seed", type=int, default=sc.THINNING_SEED)
    prepare.set_defaults(handler=command_matched_prepare)
    task = commands.add_parser("matched-task")
    _common(task)
    task.add_argument("--pair", required=True, choices=PAIRS)
    task.add_argument("--process", required=True, choices=("MAP", "RESOLVE", "REPORT"))
    task.add_argument("--env-bin", default=str(Path(sys.executable).parent))
    task.add_argument("--dry-run", action="store_true")
    task.set_defaults(handler=command_matched_task)
    farms = commands.add_parser("farms")
    _common(farms)
    farms.add_argument("--qc-summary", required=True)
    farms.add_argument("--arms", default=",".join((*MAIN_ARMS, *CONTEXT_ARMS)))
    farms.add_argument("--pairs", default=",".join(PAIRS))
    farms.set_defaults(handler=command_farms)
    acceptance = commands.add_parser("acceptance")
    _common(acceptance)
    acceptance.add_argument("--arm", required=True, choices=(*MAIN_ARMS, *CONTEXT_ARMS))
    acceptance.add_argument("--store", required=True)
    acceptance.add_argument("--gene-id-fallback-csv", required=True)
    acceptance.add_argument("--n-processors", type=int, default=6)
    acceptance.add_argument("--pairs", default=",".join(PAIRS))
    acceptance.set_defaults(handler=command_acceptance)
    transcripts = commands.add_parser("transcripts")
    _common(transcripts)
    transcripts.add_argument("--pair", required=True, choices=PAIRS)
    transcripts.add_argument(
        "--platform",
        required=True,
        choices=("MERSCOPE", "XENIUM", "merscope", "xenium"),
    )
    transcripts.add_argument("--xenium-min-qv", type=float, default=20.0)
    transcripts.set_defaults(handler=command_transcripts)
    controls = commands.add_parser("controls")
    _common(controls)
    controls.add_argument("--pair", required=True, choices=PAIRS)
    controls.add_argument(
        "--platform",
        required=True,
        choices=("MERSCOPE", "XENIUM", "merscope", "xenium"),
    )
    controls.add_argument("--xenium-min-qv", type=float, default=20.0)
    controls.set_defaults(handler=command_controls)
    analyze = commands.add_parser("analyze")
    _common(analyze)
    analyze.add_argument("--pairs", default=",".join(PAIRS))
    analyze.add_argument("--analyses", default="1,2,3,4")
    analyze.add_argument("--store", required=True)
    analyze.add_argument("--gene-annotation-gtf", default="")
    analyze.add_argument(
        "--samplesheet", default="", help="the stage B samplesheet (xenium_dir)"
    )
    analyze.set_defaults(handler=command_analyze)
    posthoc_parser = commands.add_parser("posthoc")
    _common(posthoc_parser)
    posthoc_parser.add_argument("--pairs", default=",".join(PAIRS))
    posthoc_parser.add_argument("--store", required=True)
    posthoc_parser.set_defaults(handler=command_posthoc)
    render = commands.add_parser("render")
    _common(render)
    render.set_defaults(handler=command_render)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one subcommand."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    sys.exit(main())

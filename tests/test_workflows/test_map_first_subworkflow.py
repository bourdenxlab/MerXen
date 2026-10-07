"""CLUSTERING_MAP_FIRST: MAP -> RESOLVE -> COMPUTE_CPU (plan §3.1-§3.5).

``workflows/subworkflows/clustering_map_first.nf`` runs
``ANNOTATION_PREPARED_REFERENCES`` (ANNOTATE_PANEL, one ANNOTATE_REFERENCE_PREP
per unique bundle, the per pair x segmentation ``groupKey`` collection),
then ``CLUSTERING_SQUIDPY_ANNOTATE_MAP`` on each released pair x
segmentation (``CLUSTERING_ANNOTATE_MAP``),
``CLUSTERING_SQUIDPY_ANNOTATE_RESOLVE`` after each MAP
(``CLUSTERING_ANNOTATE``) and ``CLUSTERING_SQUIDPY_COMPUTE_CPU`` after each
RESOLVE (``CLUSTERING_MAP_FIRST``, main.nf hook H5). When ``nextflow`` is
installed, a harness runs ``CLUSTERING_MAP_FIRST`` with ``-stub-run`` (PREP,
MAP, RESOLVE and COMPUTE_CPU write stub outputs; ANNOTATE_PANEL runs for real
on synthetic prepared H5ADs, or, for the fixture branches, a fake ``merxen
annotation-panel`` writes hand-written ``required_bundles.json`` files:
same-panel set c, ``per_platform``, mouse, refused, a failing and a slow
PREP, a large panel) and checks that:

* MAP starts for a pair x segmentation only after every PREP it needs has
  completed, and never waits for another branch's bundles;
* MAP receives exactly the bundle refs its ``required_bundles.json`` lists
  (none for a refused panel), and a failed PREP drops only its branches;
* RESOLVE runs once per pair x segmentation, after its own MAP, on that
  MAP's output, the same bundle refs, the pair's ALIGN files and the
  branch's registration checks from the QC stage (read with
  ``--registration-qc-dir``; a branch without an entry still resolves);
* COMPUTE_CPU runs once per pair x segmentation, after its own RESOLVE, on
  the RESOLVE output, in the main environment without a GPU lock, and emits
  FINALIZE's input shape;
* MAP's, RESOLVE's and COMPUTE_CPU's resources follow the plan (6 CPUs,
  24 GB / 48 GB above 1,000 genes; 2 CPUs, 16 GB / 32 GB; 8 CPUs, 32 GB);
* a ``-resume`` run re-runs PREP (never cached) but keeps MAP, RESOLVE and
  COMPUTE_CPU cached, because they hash their inputs' content
  (``cache "deep"``);
* a ``-resume`` run after a RESOLVE-only change (``annotation_allow_single_method``)
  re-runs every RESOLVE and no MAP or ANNOTATE_PANEL, and COMPUTE_CPU, whose
  staged labels changed; a RESOLVE re-run with byte-identical outputs leaves
  COMPUTE_CPU cached;
* the end-of-run summary (hook H6) lists the branch whose PREP failed and the
  refused panels.

A second run executes the real MAP and RESOLVE scripts (no ``-stub-run``)
with a fake ``merxen`` that records their command lines and environments;
for the refused panel both run the real commands (slow). ``main.nf`` calls
``CLUSTERING_MAP_FIRST`` only in map_first runs (hook H5).
"""

from __future__ import annotations

import csv
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from merxen.annotation import pipeline as pipeline_module
from merxen.annotation.config import AnnotationConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / "workflows"
SUBWORKFLOW = WORKFLOWS / "subworkflows" / "clustering_map_first.nf"
MODULE = WORKFLOWS / "modules" / "annotation.nf"
ANNOTATION_CONFIG = WORKFLOWS / "conf" / "annotation.config"
LIB_DIR = WORKFLOWS / "lib"
MAIN_NF = WORKFLOWS / "main.nf"
NEXTFLOW = shutil.which("nextflow")
needs_nextflow = pytest.mark.skipif(NEXTFLOW is None, reason="nextflow is unavailable")
MAP = "CLUSTERING_SQUIDPY_ANNOTATE_MAP"
RESOLVE = "CLUSTERING_SQUIDPY_ANNOTATE_RESOLVE"
PREP = "ANNOTATE_REFERENCE_PREP"
PANEL = "ANNOTATE_PANEL"
SLOW_PREP_SECONDS = 4


def _hash(tag: str) -> str:
    return (tag * 64)[:64]


# Hand-written required bundles per fixture branch: (reference, species,
# role, purpose, panel tag or None, n_panel_genes).
FIXTURE_BRANCHES: dict[
    tuple[str, str], list[tuple[str, str, str, str, str | None, int | None]]
] = {
    # Same-panel human pair on proseg_hybrid: set a (WHB, SEA-AD) + set c.
    ("F1", "proseg_hybrid"): [
        ("whb_frontal_supc_clus", "human", "primary", "annotation", "a", 296),
        ("whb_frontal_supc_clus", "human", "primary", "setc_sensitivity", "c", 264),
        ("seaad_mr_panel", "human", "secondary", "annotation", "a", 296),
    ],
    # The same pair on reseg: no set-c run.
    ("F1", "reseg"): [
        ("whb_frontal_supc_clus", "human", "primary", "annotation", "a", 296),
        ("seaad_mr_panel", "human", "secondary", "annotation", "a", 296),
    ],
    # per_platform: both platform panels x {WHB, SEA-AD} + WHB on the
    # intersection. Its Xenium panel's PREP tasks are slow: only F3 waits.
    ("F3", "proseg_hybrid"): [
        ("whb_frontal_supc_clus", "human", "primary", "annotation", "m", 500),
        ("whb_frontal_supc_clus", "human", "primary", "annotation", "x", 480),
        ("seaad_mr_panel", "human", "secondary", "annotation", "m", 500),
        ("seaad_mr_panel", "human", "secondary", "annotation", "x", 480),
        ("whb_frontal_supc_clus", "human", "primary", "intersection_xpanel", "i", 300),
    ],
    # A refused panel: no bundle; MAP runs at once and records the refusal.
    ("F4", "proseg_hybrid"): [],
    # A mouse section: the WMB panel bundle and the panel-independent shares.
    ("F5", "original_seg"): [
        ("wmb_panel", "mouse", "primary", "annotation", "g", 500),
        ("wmb_region_share", "mouse", "region_share", "panel_independent", None, None),
    ],
    # One of its bundles fails in PREP: the branch is dropped, not stalled.
    ("F7", "proseg_hybrid"): [
        ("whb_frontal_supc_clus", "human", "primary", "annotation", "a", 296),
        ("failing_reference", "human", "secondary", "annotation", "b", 296),
    ],
    # A large panel: MAP gets 48 GB.
    ("F8", "proseg_hybrid"): [
        ("whb_frontal_supc_clus", "human", "primary", "annotation", "l", 5000),
    ],
}
SLOW_TAG = "x"
# Real ANNOTATE_PANEL branches on synthetic prepared H5ADs: (pair, genes).
REAL_BRANCHES = {("PSAME", "proseg_hybrid"): 60, ("PFEW", "proseg_hybrid"): 30}


# --------------------------------------------------------------------------
# Synthetic inputs


def _write_prepared_h5ad(path: Path, *, platform: str, n_genes: int) -> None:
    symbols = [f"GENE{i}" for i in range(n_genes)]
    var = pd.DataFrame(index=pd.Index(symbols, dtype=str))
    var["gene"] = symbols
    if platform == "XENIUM":
        var["ensembl_id"] = [f"ENSG{i:011d}" for i in range(n_genes)]
    counts = np.ones((40, n_genes), dtype=np.float32)
    adata = ad.AnnData(
        X=sparse.csr_matrix(counts),
        obs=pd.DataFrame(index=[f"{platform}_{i}" for i in range(counts.shape[0])]),
        var=var,
    )
    adata.obsm["spatial"] = np.full((counts.shape[0], 2), 2.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(path)


def _prepared_branch(
    root: Path, pair_id: str, segmentation: str, n_genes: int
) -> dict[str, str]:
    """Write a prepared directory, clustering config and samples JSON."""
    base = root / f"{pair_id}_{segmentation}"
    prepared = base / "clustering_prepare_out"
    samples = []
    entries = {}
    for platform in ("MERSCOPE", "XENIUM"):
        sample_id = f"{pair_id}_{platform}"
        relative = f"{platform.lower()}/{sample_id}_prepared.h5ad"
        _write_prepared_h5ad(prepared / relative, platform=platform, n_genes=n_genes)
        entries[sample_id] = relative
        samples.append(
            {"sample_id": sample_id, "platform": platform, "segmentation": segmentation}
        )
    (prepared / "manifest.json").write_text(json.dumps({"samples": entries}))
    config = base / "clustering_squidpy_config.json"
    config.write_text(
        json.dumps({"pair_id": pair_id, "min_counts": 10, "samples": samples})
    )
    return {
        "pair_id": pair_id,
        "segmentation": segmentation,
        "samples_json": json.dumps(samples),
        "config": str(config),
        "prepared_dir": str(prepared),
        "alignment_files": [],
    }


# The pair whose ALIGN files the harness passes (every other pair has none).
ALIGNED_PAIR = "F1"
ALIGN_FILES = ("shared_tissue_mask.npy", "registration_summary.json")
# The branch whose QC registration checks the harness passes (M13 C15), the
# branch with no registration entry (it must still resolve, without checks)
# and an entry of a branch that is never mapped (it must be left out).
REGISTERED_BRANCH = ("F1", "proseg_hybrid")
REGISTRATION_FILES = (
    "f1_merscope_registration_qc.json",
    "f1_xenium_registration_qc.json",
)
UNREGISTERED_BRANCH = ("PFEW", "proseg_hybrid")
STRAY_REGISTRATION_BRANCH = ("F9", "proseg_hybrid")


def _write_alignment(root: Path, pair_ids: list[str]) -> list[str]:
    """Write one ALIGN entry per pair: F1's align_out files, [] for the others."""
    align_out = root / "align" / ALIGNED_PAIR / "align_out"
    align_out.mkdir(parents=True)
    files = [str(align_out / name) for name in ALIGN_FILES]
    for path in files:
        Path(path).write_text("{}")
    entries = [
        {"pair_id": pair_id, "files": files if pair_id == ALIGNED_PAIR else []}
        for pair_id in sorted(set(pair_ids))
    ]
    (root / "alignment_inputs.json").write_text(json.dumps(entries))
    return files


def _write_registration(root: Path, branches: list[tuple[str, str]]) -> list[str]:
    """Write one registration entry per branch (M13 C15).

    ``REGISTERED_BRANCH`` gets its two QC checks, ``UNREGISTERED_BRANCH`` no
    entry at all, a branch that is never mapped a stray entry, and every
    other branch ``[]`` (no QC stage).
    """
    qc_out = root / "qc" / "_".join(REGISTERED_BRANCH) / "qc_out"
    qc_out.mkdir(parents=True)
    files = [str(qc_out / name) for name in REGISTRATION_FILES]
    for path in files:
        Path(path).write_text(json.dumps({"status": "PASS", "density_ratio": 3.0}))
    stray = root / "qc" / "stray" / "f9_merscope_registration_qc.json"
    stray.parent.mkdir(parents=True)
    stray.write_text("{}")
    entries = [
        {
            "pair_id": pair_id,
            "segmentation": segmentation,
            "files": files if (pair_id, segmentation) == REGISTERED_BRANCH else [],
        }
        for pair_id, segmentation in sorted(set(branches))
        if (pair_id, segmentation) != UNREGISTERED_BRANCH
    ]
    entries.append(
        {
            "pair_id": STRAY_REGISTRATION_BRANCH[0],
            "segmentation": STRAY_REGISTRATION_BRANCH[1],
            "files": [str(stray)],
        }
    )
    (root / "registration_inputs.json").write_text(json.dumps(entries))
    return files


def _write_fixture_panels(root: Path) -> Path:
    """Write the fixture ANNOTATE_PANEL outputs the fake command copies."""
    fixtures = root / "fixture_panels"
    for (pair_id, segmentation), bundles in FIXTURE_BRANCHES.items():
        panel_dir = fixtures / f"{pair_id}_{segmentation}"
        panel_dir.mkdir(parents=True)
        entries = []
        for reference_id, species, role, purpose, tag, n_genes in bundles:
            panel_file = None
            if tag is not None:
                panel_file = f"panel_genes_{tag}.json"
                (panel_dir / panel_file).write_text(
                    json.dumps({"panel_hash": _hash(tag), "sample_ids": [pair_id]})
                )
            entries.append(
                {
                    "reference_id": reference_id,
                    "role": role,
                    "species": species,
                    "purpose": purpose,
                    "panel_name": tag,
                    "panel_hash": _hash(tag) if tag else None,
                    "panel_file": panel_file,
                    "n_panel_genes": n_genes,
                }
            )
        (panel_dir / "required_bundles.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "pair_id": pair_id,
                    "segmentation": segmentation,
                    "species": bundles[0][1] if bundles else "human",
                    "panel_mode": "per_platform" if pair_id == "F3" else "intersection",
                    "status": "ok" if bundles else "refused",
                    "reasons": []
                    if bundles
                    else ["intersection: 12 resolved genes < 50"],
                    "bundles": entries,
                    "n_required": len(entries),
                }
            )
        )
    return fixtures


FAKE_MERXEN = """#!__PYTHON__
# Test double. annotation-panel: copy the fixture panel of a fixture branch,
# else run the real command. annotate / annotate-resolve (real MAP and
# RESOLVE scripts only): record the command line and environment; run the
# real command for a refused panel (MAP writes the refusal, RESOLVE the
# statuses) and write a minimal output otherwise.
# annotation-reference-prep (real PREP script only): write a bundle ref.
import json
import os
import shutil
import sys
from pathlib import Path

args = sys.argv[1:]
REAL = "__REAL__"
FIXTURES = Path("__FIXTURES__")


def value(flag):
    return args[args.index(flag) + 1]


command = args[0] if args else ""
if command == "annotation-panel":
    fixture = FIXTURES / f"{value('--pair-id')}_{value('--segmentation')}"
    if fixture.is_dir():
        shutil.copytree(fixture, value("--output-dir"))
        (Path(value("--output-dir")) / "fake_panel_argv.json").write_text(
            json.dumps(args)
        )
        sys.exit(0)
elif command == "annotation-reference-prep":
    from merxen.annotation.store import BundleRef

    panel_hash = None
    if "--panel-genes" in args:
        panel_hash = json.loads(Path(value("--panel-genes")).read_text())["panel_hash"]
    BundleRef(
        reference_id=value("--reference-id"),
        species=value("--species"),
        role={"seaad_mr_panel": "secondary", "wmb_region_share": "region_share"}.get(
            value("--reference-id"), "primary"
        ),
        panel_hash=panel_hash,
        build_hash="f" * 64,
        path=str(Path(value("--store")) / value("--reference-id") / ("f" * 64)),
        store_root=value("--store"),
    ).write(value("--output"))
    sys.exit(0)
elif command in ("annotate", "annotate-resolve"):
    record = {
        "argv": args,
        "env": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "NUMBA_NUM_THREADS",
                "CUDA_VISIBLE_DEVICES",
                "PYTHONPATH",
            )
        },
        "refs": [
            json.loads(Path(args[i + 1]).read_text())
            for i, arg in enumerate(args)
            if arg == "--bundle-ref"
        ],
    }
    Path(f"fake_{command.replace('-', '_')}.json").write_text(json.dumps(record))
    required = json.loads(
        (Path(value("--panel-dir")) / "required_bundles.json").read_text()
    )
    if required["status"] != "refused":
        out = Path(value("--out"))
        out.mkdir(parents=True)
        name = "map_manifest.json" if command == "annotate" else "fake_summary.json"
        (out / name).write_text(json.dumps({"fake": True}))
        if "--run-record" in args:
            Path(value("--run-record")).write_text(json.dumps({"fake": True}))
        sys.exit(0)
os.execv(REAL, [REAL, *args])
"""


def _nextflow_env() -> dict[str, str]:
    env = {**os.environ, "NXF_DISABLE_CHECK_LATEST": "true", "NXF_ANSI_LOG": "false"}
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT / "src"), *filter(None, [os.environ.get("PYTHONPATH")])]
    )
    env["PATH"] = os.pathsep.join([str(Path(sys.executable).parent), env["PATH"]])
    env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def _fake_merxen_env(root: Path, fixtures: Path) -> dict[str, str]:
    env = _nextflow_env()
    real = shutil.which("merxen", path=env["PATH"])
    assert real is not None
    bin_dir = root / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "merxen"
    fake.write_text(
        FAKE_MERXEN.replace("__PYTHON__", sys.executable)
        .replace("__REAL__", real)
        .replace("__FIXTURES__", str(fixtures))
    )
    fake.chmod(0o755)
    env["PATH"] = os.pathsep.join([str(bin_dir), env["PATH"]])
    return env


def _read_trace(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


HARNESS_CLASS = """
import groovy.json.JsonOutput
import groovy.json.JsonSlurperClassic

class MapHarness {
    static List readList(Object path) {
        return new JsonSlurperClassic().parse(new File(path.toString())) as List
    }

    static void write(Object path, Object rows) {
        new File(path.toString()).text = JsonOutput.toJson(rows)
    }

    static Map mapRow(List item) {
        def (pairId, segmentation, samplesJson, config) = item[0..3]
        def (preparedDir, panelDir, refs, mapDir) = item[4..7]
        return [
            pair_id: pairId,
            segmentation: segmentation,
            samples_json: samplesJson,
            config: config.toString(),
            prepared_dir: preparedDir.toString(),
            panel_dir: panelDir.toString(),
            refs: (refs as List).collect { ref -> ref.toString() },
            map_dir: mapDir.toString(),
        ]
    }

    static Map labelRow(List item) {
        return mapRow(item[0..7]) + [resolve_dir: item[8].toString()]
    }

    static Map computedRow(List item) {
        def (pairId, segmentation, samplesJson, computedDir) = item[0..3]
        return [
            pair_id: pairId,
            segmentation: segmentation,
            samples_json: samplesJson,
            computed_dir: computedDir.toString(),
        ]
    }

    static void writeText(Object path, String text) {
        new File(path.toString()).text = text
    }
}
"""

# The registration checks per pair x segmentation, as hook H5 builds them.
REGISTRATION_CHANNEL = """registration_ch = channel
        .fromList(MapHarness.readList(params.registration_inputs))
        .map { item ->
            tuple(
                item.pair_id, item.segmentation,
                item.files.collect { path -> file(path) },
            )
        }"""

# CLUSTERING_ANNOTATE takes PREPARE's tuple plus the pair's ALIGN files, and
# the registration checks.
HARNESS_ANNOTATE = """
include { CLUSTERING_ANNOTATE } from '__SUBWORKFLOW__'

workflow {
    prepared_ch = channel
        .fromList(MapHarness.readList(params.prepared_inputs))
        .map { item ->
            tuple(
                item.pair_id, item.segmentation, item.samples_json,
                file(item.config), file(item.prepared_dir),
                item.alignment_files.collect { path -> file(path) },
            )
        }
    __REGISTRATION__
    mapped = CLUSTERING_ANNOTATE(prepared_ch, registration_ch)
    __EMIT__
}
"""

# CLUSTERING_MAP_FIRST, as main.nf's hook H5 calls it: PREPARE's tuple, one
# ALIGN entry per pair, the registration checks per pair x segmentation, and
# the end-of-run summary of hook H6.
HARNESS_MAP_FIRST = """
include { CLUSTERING_MAP_FIRST } from '__SUBWORKFLOW__'

workflow {
    prepared_ch = channel
        .fromList(MapHarness.readList(params.prepared_inputs))
        .map { item ->
            tuple(
                item.pair_id, item.segmentation, item.samples_json,
                file(item.config), file(item.prepared_dir),
            )
        }
    alignment_ch = channel
        .fromList(MapHarness.readList(params.alignment_inputs))
        .map { item -> tuple(item.pair_id, item.files.collect { path -> file(path) }) }
    __REGISTRATION__
    prepared_ch.subscribe { pairId, segmentation, _samplesJson, _config, _preparedDir ->
        AnnotationRunRecord.expect(pairId, segmentation)
    }
    mapped = CLUSTERING_MAP_FIRST(prepared_ch, alignment_ch, registration_ch)
    __EMIT__
    mapped.computed
        .map { item -> MapHarness.computedRow(item) }
        .collect()
        .subscribe { rows -> MapHarness.write(params.computed_out, rows) }
    def harnessParams = params
    def harnessWorkflow = workflow
    harnessWorkflow.onComplete {
        MapHarness.writeText(
            harnessParams.summary_out,
            AnnotationSettings.completionSummary(
                harnessParams,
                AnnotationRunRecord.runInfo() + [success: harnessWorkflow.success],
            ),
        )
    }
}
"""

EMIT_LABELS = """mapped.maps
        .map { item -> MapHarness.mapRow(item) }
        .collect()
        .subscribe { rows -> MapHarness.write(params.maps_out, rows) }
    mapped.labels
        .map { item -> MapHarness.labelRow(item) }
        .collect()
        .subscribe { rows -> MapHarness.write(params.labels_out, rows) }"""


def _write_harness(root: Path, *, entry: str = "CLUSTERING_MAP_FIRST") -> None:
    (root / "lib").mkdir(parents=True)
    for source in LIB_DIR.glob("*.groovy"):
        shutil.copy(source, root / "lib" / source.name)
    (root / "lib" / "MapHarness.groovy").write_text(HARNESS_CLASS)
    template = (
        HARNESS_MAP_FIRST if entry == "CLUSTERING_MAP_FIRST" else HARNESS_ANNOTATE
    )
    (root / "main.nf").write_text(
        template.replace("__SUBWORKFLOW__", str(SUBWORKFLOW))
        .replace("__EMIT__", EMIT_LABELS)
        .replace("__REGISTRATION__", REGISTRATION_CHANNEL)
    )


def _write_inputs(root: Path) -> list[dict[str, str]]:
    items = [
        _prepared_branch(root / "prepared", pair_id, segmentation, 60)
        for pair_id, segmentation in FIXTURE_BRANCHES
    ]
    items += [
        _prepared_branch(root / "prepared", pair_id, segmentation, n_genes)
        for (pair_id, segmentation), n_genes in REAL_BRANCHES.items()
    ]
    (root / "prepared_inputs.json").write_text(json.dumps(items))
    _write_alignment(root, [item["pair_id"] for item in items])
    _write_registration(
        root, [(item["pair_id"], item["segmentation"]) for item in items]
    )
    return items


def _write_config(
    root: Path,
    *,
    failing: bool = True,
    slow: bool = True,
    single_method: bool = False,
) -> Path:
    """Write the harness config (a failing and a slow PREP when asked).

    ``single_method`` sets ``annotation_allow_single_method``, a setting only
    RESOLVE reads.
    """
    fail = '"exit 3"' if failing else '"true"'
    wait = f'"sleep {SLOW_PREP_SECONDS}"' if slow else '"true"'
    before = f"""
    withName: ".*:{PREP}" {{
        beforeScript = {{
            bundle.reference_id == "failing_reference" ? {fail} :
                (bundle.panel_tag == "{_hash(SLOW_TAG)}" ? {wait} : "true")
        }}
    }}"""
    config = root / "nextflow.config"
    config.write_text(
        f"""
includeConfig '{ANNOTATION_CONFIG}'
params {{
    species = "human"
    outdir = "{root / "results"}"
    annotation_reference_store = "{root / "store"}"
    prepared_inputs = "{root / "prepared_inputs.json"}"
    alignment_inputs = "{root / "alignment_inputs.json"}"
    registration_inputs = "{root / "registration_inputs.json"}"
    maps_out = "{root / "maps_out.json"}"
    labels_out = "{root / "labels_out.json"}"
    computed_out = "{root / "computed_out.json"}"
    summary_out = "{root / "summary_out.txt"}"
    annotation_allow_single_method = {str(single_method).lower()}
    clustering_squidpy_mode = "map_first"
}}
executor {{
    cpus = 64
    memory = "512 GB"
}}
process {{
    executor = "local"
    errorStrategy = "ignore"{before}
}}
trace {{
    enabled = true
    overwrite = true
    raw = true
    file = "{root / "trace.tsv"}"
    fields = "task_id,process,tag,status,submit,start,complete,cpus,memory,workdir"
}}
"""
    )
    return config


def _run(
    root: Path, env: dict[str, str], *args: str
) -> subprocess.CompletedProcess[str]:
    assert NEXTFLOW is not None
    return subprocess.run(
        [NEXTFLOW, "-log", str(root / "nextflow.log"), "run", "main.nf", *args],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )


def _published_resolve(root: Path) -> dict[str, dict[str, Any]]:
    """Read every published stub RESOLVE output: summary and config."""
    published: dict[str, dict[str, Any]] = {}
    for summary in (root / "results").glob(
        "*/*/annotation_resolve/annotation_resolve_out/*_resolve_summary.json"
    ):
        output = summary.parent
        published[f"{output.parents[2].name}|{output.parents[1].name}"] = {
            "summary": json.loads(summary.read_text()),
            "config": json.loads((output / "annotation_config.json").read_text()),
        }
    return published


def _collect(root: Path) -> dict[str, Any]:
    """Read one run's trace, emitted rows, published RESOLVE outputs, summary."""
    results = {
        "trace": _read_trace(root / "trace.tsv"),
        "maps": json.loads((root / "maps_out.json").read_text()),
        "labels": json.loads((root / "labels_out.json").read_text()),
        "computed": json.loads((root / "computed_out.json").read_text()),
        "summary": (root / "summary_out.txt").read_text(),
        "resolve": _published_resolve(root),
    }
    for name in ("maps_out.json", "labels_out.json", "computed_out.json"):
        (root / name).unlink()
    (root / "summary_out.txt").unlink()
    return results


def _move_resolve_work_dirs(trace: list[dict[str, str]]) -> None:
    """Move RESOLVE's task directories aside: -resume must re-run RESOLVE."""
    for row in _rows(trace, RESOLVE):
        work = Path(row["workdir"])
        work.rename(work.with_name(work.name + ".moved"))


def _run_harness(root: Path) -> dict[str, Any]:
    """Run the stub harness, then twice more with ``-resume``.

    The first ``-resume`` run changes nothing (the slow PREP only skips its
    wait); the second sets ``annotation_allow_single_method``, a setting only
    RESOLVE reads.
    """
    fixtures = _write_fixture_panels(root)
    env = _fake_merxen_env(root, fixtures)
    _write_harness(root)
    _write_inputs(root)
    _write_config(root)
    first = _run(root, env, "-stub-run")
    assert first.returncode == 0, first.stdout + first.stderr
    runs = {"first": _collect(root)}
    # PREP never caches; the resume run skips the slow PREP's wait only.
    _write_config(root, slow=False)
    second = _run(root, env, "-stub-run", "-resume")
    assert second.returncode == 0, second.stdout + second.stderr
    runs["resume"] = _collect(root)
    _write_config(root, slow=False, single_method=True)
    third = _run(root, env, "-stub-run", "-resume")
    assert third.returncode == 0, third.stdout + third.stderr
    runs["resolve_change"] = _collect(root)
    # RESOLVE re-runs (its task directories are gone) and writes the same
    # bytes: COMPUTE_CPU, which hashes the staged labels, stays cached.
    _move_resolve_work_dirs(runs["resolve_change"]["trace"])
    fourth = _run(root, env, "-stub-run", "-resume")
    assert fourth.returncode == 0, fourth.stdout + fourth.stderr
    runs["resolve_rerun"] = _collect(root)
    return {
        "root": str(root),
        "trace": runs["first"]["trace"],
        "maps": runs["first"]["maps"],
        "labels": runs["first"]["labels"],
        "computed": runs["first"]["computed"],
        "summary": runs["first"]["summary"],
        "resolve": runs["first"]["resolve"],
        "resume_trace": runs["resume"]["trace"],
        "resume_maps": runs["resume"]["maps"],
        "resume_labels": runs["resume"]["labels"],
        "resume_computed": runs["resume"]["computed"],
        "change_trace": runs["resolve_change"]["trace"],
        "change_maps": runs["resolve_change"]["maps"],
        "change_labels": runs["resolve_change"]["labels"],
        "change_computed": runs["resolve_change"]["computed"],
        "change_resolve": runs["resolve_change"]["resolve"],
        "rerun_trace": runs["resolve_rerun"]["trace"],
        "rerun_labels": runs["resolve_rerun"]["labels"],
        "rerun_computed": runs["resolve_rerun"]["computed"],
        "align_files": [
            str(root / "align" / ALIGNED_PAIR / "align_out" / name)
            for name in ALIGN_FILES
        ],
        "outdir": str(root / "results"),
    }


@pytest.fixture(scope="module")
def harness(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Run the harness once per test session, also under pytest-xdist."""
    base = tmp_path_factory.getbasetemp()
    shared = base.parent if os.environ.get("PYTEST_XDIST_WORKER") else base
    with (shared / "map_first_harness.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        results_path = shared / "map_first_harness.json"
        if not results_path.exists():
            results = _run_harness(shared / "map_first_harness")
            results_path.write_text(json.dumps(results))
        results = json.loads(results_path.read_text())
    return {**results, "root": Path(results["root"]), "outdir": Path(results["outdir"])}


def _rows(trace: list[dict[str, str]], process: str) -> list[dict[str, str]]:
    return [row for row in trace if row["process"].split(":")[-1] == process]


def _branch_bundles(
    harness: dict[str, Any], branch: tuple[str, str]
) -> list[tuple[str, str | None]]:
    """The (reference_id, panel_hash) the branch's panel requires."""
    map_row = next(
        row
        for row in harness["maps"]
        if (row["pair_id"], row["segmentation"]) == branch
    )
    required = json.loads(
        (Path(map_row["panel_dir"]) / "required_bundles.json").read_text()
    )
    return [(item["reference_id"], item["panel_hash"]) for item in required["bundles"]]


def _prep_tag(reference_id: str, panel_hash: str | None) -> str:
    return f"{reference_id}:{(panel_hash or 'panel_independent')[:12]}"


def _refs(paths: list[str]) -> list[tuple[str, str | None]]:
    refs = [json.loads(Path(path).read_text()) for path in paths]
    return [(ref["reference_id"], ref["panel_hash"]) for ref in refs]


EXPECTED_BRANCHES = {
    branch for branch in FIXTURE_BRANCHES if branch != ("F7", "proseg_hybrid")
} | set(REAL_BRANCHES)


# --------------------------------------------------------------------------
# Channel logic (stub runs)


@needs_nextflow
def test_map_runs_once_per_released_branch(harness: dict[str, Any]) -> None:
    rows = _rows(harness["trace"], MAP)
    tags = Counter(tuple(row["tag"].split(":")) for row in rows)
    assert set(tags) == EXPECTED_BRANCHES
    assert all(count == 1 for count in tags.values()), tags
    assert all(row["status"] == "COMPLETED" for row in rows)
    released = {(row["pair_id"], row["segmentation"]) for row in harness["maps"]}
    assert released == EXPECTED_BRANCHES


@needs_nextflow
def test_map_waits_for_every_bundle_its_branch_requires(
    harness: dict[str, Any],
) -> None:
    trace = harness["trace"]
    preps = {row["tag"]: row for row in _rows(trace, PREP)}
    for row in _rows(trace, MAP):
        branch = tuple(row["tag"].split(":"))
        needed = [_prep_tag(*bundle) for bundle in _branch_bundles(harness, branch)]
        for tag in needed:
            assert preps[tag]["status"] == "COMPLETED", tag
            assert int(row["submit"]) >= int(preps[tag]["complete"]), (branch, tag)


@needs_nextflow
def test_map_never_waits_for_another_branch_bundles(harness: dict[str, Any]) -> None:
    """F3's slow Xenium-panel PREP holds back F3 only (no global barrier)."""
    trace = harness["trace"]
    slow = [
        row
        for row in _rows(trace, PREP)
        if row["tag"].endswith(":" + _hash(SLOW_TAG)[:12])
    ]
    assert len(slow) == 2
    slow_done = min(int(row["complete"]) for row in slow)
    maps = {tuple(row["tag"].split(":")): row for row in _rows(trace, MAP)}
    assert int(maps[("F3", "proseg_hybrid")]["submit"]) >= slow_done
    # Every other fixture branch (seconds of stub work each) was mapped while
    # F3's slow bundles were still building.
    others = [
        branch
        for branch in maps
        if branch in FIXTURE_BRANCHES and branch != ("F3", "proseg_hybrid")
    ]
    assert len(others) == 5
    late = [branch for branch in others if int(maps[branch]["submit"]) >= slow_done]
    assert late == []


@needs_nextflow
def test_map_receives_exactly_its_bundle_refs(harness: dict[str, Any]) -> None:
    maps = {(row["pair_id"], row["segmentation"]): row for row in harness["maps"]}
    for branch, row in maps.items():
        expected = _branch_bundles(harness, branch)
        assert _refs(row["refs"]) == expected, branch
        # The stub copies the staged refs next to its manifest.
        manifest = json.loads((Path(row["map_dir"]) / "map_manifest.json").read_text())
        assert manifest["stub"] is True
        assert manifest["pair_id"] == branch[0]
        assert manifest["n_required"] == len(expected)
        assert [Path(name).name for name in manifest["bundle_refs"]] == [
            f"bundle_ref_{index + 1}.json" for index in range(len(expected))
        ]
        staged = sorted((Path(row["map_dir"]) / "stub_bundle_refs").glob("*.json"))
        assert sorted(_refs([str(path) for path in staged])) == sorted(expected)
    assert len(maps[("F1", "proseg_hybrid")]["refs"]) == 3
    assert len(maps[("F1", "reseg")]["refs"]) == 2
    assert len(maps[("F3", "proseg_hybrid")]["refs"]) == 5
    assert len(maps[("F5", "original_seg")]["refs"]) == 2
    assert len(maps[("PSAME", "proseg_hybrid")]["refs"]) == 2


@needs_nextflow
def test_a_refused_panel_is_mapped_at_once_without_bundles(
    harness: dict[str, Any],
) -> None:
    maps = {(row["pair_id"], row["segmentation"]): row for row in harness["maps"]}
    for branch in (("F4", "proseg_hybrid"), ("PFEW", "proseg_hybrid")):
        assert maps[branch]["refs"] == []
        manifest = json.loads(
            (Path(maps[branch]["map_dir"]) / "map_manifest.json").read_text()
        )
        assert manifest["panel_status"] == "refused"
        assert manifest["panel_reasons"]
    # The real ANNOTATE_PANEL refused PFEW's 30-gene panel (< min_mapped_genes).
    assert "min_mapped_genes" in " ".join(
        json.loads(
            (
                Path(maps[("PFEW", "proseg_hybrid")]["map_dir"]) / "map_manifest.json"
            ).read_text()
        )["panel_reasons"]
    )


@needs_nextflow
def test_a_failed_prep_drops_only_the_branches_that_need_it(
    harness: dict[str, Any],
) -> None:
    preps = {row["tag"]: row for row in _rows(harness["trace"], PREP)}
    assert preps[_prep_tag("failing_reference", _hash("b"))]["status"] == "FAILED"
    released = {(row["pair_id"], row["segmentation"]) for row in harness["maps"]}
    assert ("F7", "proseg_hybrid") not in released
    # F7 shares set a "a" with F1, which is mapped.
    assert ("F1", "proseg_hybrid") in released


@needs_nextflow
def test_map_resources_follow_the_panel_size(harness: dict[str, Any]) -> None:
    rows = {tuple(row["tag"].split(":")): row for row in _rows(harness["trace"], MAP)}
    gib = 1024**3
    for branch, row in rows.items():
        assert row["cpus"] == "6", branch
        expected = 48 if branch == ("F8", "proseg_hybrid") else 24
        assert int(row["memory"]) == expected * gib, branch


@needs_nextflow
def test_map_publishes_under_the_pair_and_segmentation(harness: dict[str, Any]) -> None:
    for pair_id, segmentation in EXPECTED_BRANCHES:
        published = (
            harness["outdir"]
            / pair_id
            / segmentation
            / "annotation_map"
            / "annotation_map_out"
            / "map_manifest.json"
        )
        assert published.is_file(), published


@needs_nextflow
def test_resume_reruns_prep_but_keeps_map_cached(harness: dict[str, Any]) -> None:
    """PREP never caches; its byte-identical refs keep MAP cached (deep)."""
    trace = harness["resume_trace"]
    prep = Counter(row["status"] for row in _rows(trace, PREP))
    assert prep["CACHED"] == 0
    assert prep["COMPLETED"] >= 10
    maps = _rows(trace, MAP)
    assert {tuple(row["tag"].split(":")) for row in maps} == EXPECTED_BRANCHES
    assert {row["status"] for row in maps} == {"CACHED"}
    assert {row["status"] for row in _rows(trace, PANEL)} == {"CACHED"}
    first = {(r["pair_id"], r["segmentation"]): r["map_dir"] for r in harness["maps"]}
    again = {
        (r["pair_id"], r["segmentation"]): r["map_dir"] for r in harness["resume_maps"]
    }
    assert first == again


# --------------------------------------------------------------------------
# RESOLVE after MAP (stub runs)


def _branch(key: str) -> tuple[str, str]:
    pair_id, segmentation = key.split("|")
    return pair_id, segmentation


@needs_nextflow
def test_resolve_runs_once_per_branch_after_its_own_map(
    harness: dict[str, Any],
) -> None:
    trace = harness["trace"]
    resolves = _rows(trace, RESOLVE)
    tags = Counter(tuple(row["tag"].split(":")) for row in resolves)
    assert set(tags) == EXPECTED_BRANCHES
    assert all(count == 1 for count in tags.values()), tags
    assert all(row["status"] == "COMPLETED" for row in resolves)
    maps = {tuple(row["tag"].split(":")): row for row in _rows(trace, MAP)}
    for row in resolves:
        branch = tuple(row["tag"].split(":"))
        assert int(row["submit"]) >= int(maps[branch]["complete"]), branch
    # F3's slow bundles hold back F3's RESOLVE only: the other fixture
    # branches were resolved before F3 was even mapped.
    f3_map = int(maps[("F3", "proseg_hybrid")]["submit"])
    early = [
        tuple(row["tag"].split(":"))
        for row in resolves
        if tuple(row["tag"].split(":")) in FIXTURE_BRANCHES
        and int(row["submit"]) < f3_map
    ]
    assert len(early) == 5
    released = {(row["pair_id"], row["segmentation"]) for row in harness["labels"]}
    assert released == EXPECTED_BRANCHES


@needs_nextflow
def test_resolve_reads_its_own_map_output_and_bundle_refs(
    harness: dict[str, Any],
) -> None:
    maps = {(row["pair_id"], row["segmentation"]): row for row in harness["maps"]}
    for row in harness["labels"]:
        branch = (row["pair_id"], row["segmentation"])
        # The labels tuple carries MAP's tuple unchanged plus RESOLVE's output.
        assert {key: row[key] for key in maps[branch]} == maps[branch]
        output = Path(row["resolve_dir"])
        summary = json.loads(
            (
                output / f"{branch[0]}{pipeline_module.RESOLVE_SUMMARY_SUFFIX}"
            ).read_text()
        )
        assert summary["stub"] is True
        assert (summary["pair_id"], summary["segmentation"]) == branch
        # The staged MAP output is this branch's own.
        manifest = json.loads((output / "stub_map_manifest.json").read_text())
        assert (manifest["pair_id"], manifest["segmentation"]) == branch
        expected = _branch_bundles(harness, branch)
        assert summary["n_required"] == len(expected)
        assert summary["bundle_refs"] == [
            f"resolve_inputs/bundle_refs/bundle_ref_{index + 1}.json"
            for index in range(len(expected))
        ]
        staged = sorted((output / "stub_bundle_refs").glob("*.json"))
        assert sorted(_refs([str(path) for path in staged])) == sorted(expected)
        # The pair's ALIGN files come from the ALIGN channel (M5).
        assert summary["alignment_files"] == (
            [f"resolve_inputs/align_out/{name}" for name in ALIGN_FILES]
            if branch[0] == ALIGNED_PAIR
            else []
        )
        # The branch's registration checks (M13 C15) are staged and read with
        # --registration-qc-dir; a branch without checks, or without an
        # entry at all, resolves without them.
        registered = branch == REGISTERED_BRANCH
        assert summary["registration_files"] == (
            [f"resolve_inputs/registration_qc/{name}" for name in REGISTRATION_FILES]
            if registered
            else []
        )
        arguments = summary["resolve_arguments"].split(" ")
        assert ("--registration-qc-dir" in arguments) is registered
        if registered:
            option = arguments.index("--registration-qc-dir")
            assert arguments[option + 1] == "resolve_inputs/registration_qc"
        assert summary["panel_status"] == ("refused" if not expected else "ok")
        config = AnnotationConfig.model_validate_json(
            (output / "annotation_config.json").read_text()
        )
        assert config.species == ("mouse" if branch[0] == "F5" else "human")
        assert config.allow_single_method is False


@needs_nextflow
def test_resolve_resources_follow_the_panel_size(harness: dict[str, Any]) -> None:
    rows = {
        tuple(row["tag"].split(":")): row for row in _rows(harness["trace"], RESOLVE)
    }
    gib = 1024**3
    for branch, row in rows.items():
        assert row["cpus"] == "2", branch
        expected = 32 if branch == ("F8", "proseg_hybrid") else 16
        assert int(row["memory"]) == expected * gib, branch


@needs_nextflow
def test_resolve_publishes_under_the_pair_and_segmentation(
    harness: dict[str, Any],
) -> None:
    assert {_branch(key) for key in harness["resolve"]} == EXPECTED_BRANCHES
    for pair_id, segmentation in EXPECTED_BRANCHES:
        output = (
            harness["outdir"]
            / pair_id
            / segmentation
            / "annotation_resolve"
            / "annotation_resolve_out"
        )
        assert (output / f"{pair_id}_resolve_summary.json").is_file()
        assert (output / "annotation_config.json").is_file()


@needs_nextflow
def test_resume_keeps_resolve_cached(harness: dict[str, Any]) -> None:
    trace = harness["resume_trace"]
    resolves = _rows(trace, RESOLVE)
    assert {tuple(row["tag"].split(":")) for row in resolves} == EXPECTED_BRANCHES
    assert {row["status"] for row in resolves} == {"CACHED"}
    # The same MAP and RESOLVE outputs (the refs are PREP's new, identical
    # files: PREP never caches).
    first = {
        (r["pair_id"], r["segmentation"]): (r["map_dir"], r["resolve_dir"])
        for r in harness["labels"]
    }
    again = {
        (r["pair_id"], r["segmentation"]): (r["map_dir"], r["resolve_dir"])
        for r in harness["resume_labels"]
    }
    assert first == again


@needs_nextflow
def test_a_resolve_only_change_reruns_resolve_alone(harness: dict[str, Any]) -> None:
    """A threshold-like RESOLVE setting re-runs RESOLVE, never MAP (plan §3.1)."""
    trace = harness["change_trace"]
    assert {row["status"] for row in _rows(trace, MAP)} == {"CACHED"}
    assert {row["status"] for row in _rows(trace, PANEL)} == {"CACHED"}
    resolves = _rows(trace, RESOLVE)
    assert {tuple(row["tag"].split(":")) for row in resolves} == EXPECTED_BRANCHES
    assert {row["status"] for row in resolves} == {"COMPLETED"}
    # The same MAP outputs, new RESOLVE outputs with the changed config and
    # the same rules fingerprint (no code changed).
    first = {(r["pair_id"], r["segmentation"]): r for r in harness["labels"]}
    changed = {(r["pair_id"], r["segmentation"]): r for r in harness["change_labels"]}
    assert set(first) == set(changed) == EXPECTED_BRANCHES
    for branch, row in changed.items():
        assert row["map_dir"] == first[branch]["map_dir"]
        assert row["resolve_dir"] != first[branch]["resolve_dir"]
    for key, published in harness["change_resolve"].items():
        assert published["config"]["allow_single_method"] is True, key
        assert (
            published["summary"]["rules_fingerprint"]
            == (harness["resolve"][key]["summary"]["rules_fingerprint"])
        )


@needs_nextflow
def test_panel_stages_the_pairs_align_files(harness: dict[str, Any]) -> None:
    """ANNOTATE_PANEL gets the ALIGN files from the channel, never a lookup."""
    maps = {(row["pair_id"], row["segmentation"]): row for row in harness["maps"]}
    for branch, row in maps.items():
        if branch not in FIXTURE_BRANCHES:
            continue
        argv = json.loads((Path(row["panel_dir"]) / "fake_panel_argv.json").read_text())
        if branch[0] == ALIGNED_PAIR:
            assert argv[argv.index("--shared-tissue-mask") + 1] == (
                "panel_inputs/shared_tissue_mask.npy"
            )
            assert argv[argv.index("--registration-summary") + 1] == (
                "panel_inputs/registration_summary.json"
            )
        else:
            assert "--shared-tissue-mask" not in argv, branch
        assert "--require-shared-tissue-mask" in argv


# --------------------------------------------------------------------------
# COMPUTE_CPU after RESOLVE (stub runs)


COMPUTE = "CLUSTERING_SQUIDPY_COMPUTE_CPU"


@needs_nextflow
def test_compute_runs_once_per_branch_after_its_own_resolve(
    harness: dict[str, Any],
) -> None:
    trace = harness["trace"]
    computes = _rows(trace, COMPUTE)
    tags = Counter(tuple(row["tag"].split(":")) for row in computes)
    assert set(tags) == EXPECTED_BRANCHES
    assert all(count == 1 for count in tags.values()), tags
    assert all(row["status"] == "COMPLETED" for row in computes)
    resolves = {tuple(row["tag"].split(":")): row for row in _rows(trace, RESOLVE)}
    for row in computes:
        branch = tuple(row["tag"].split(":"))
        assert int(row["submit"]) >= int(resolves[branch]["complete"]), branch


@needs_nextflow
def test_compute_emits_finalize_input_and_reads_its_own_labels(
    harness: dict[str, Any],
) -> None:
    """tuple(pair, segmentation, samples_json, clustering_compute_out) per branch."""
    labels = {(row["pair_id"], row["segmentation"]): row for row in harness["labels"]}
    computed = {
        (row["pair_id"], row["segmentation"]): row for row in harness["computed"]
    }
    assert set(computed) == EXPECTED_BRANCHES
    for branch, row in computed.items():
        output = Path(row["computed_dir"])
        assert output.name == "clustering_compute_out"
        assert row["samples_json"] == labels[branch]["samples_json"]
        manifest = json.loads((output / "stub_compute_manifest.json").read_text())
        assert (manifest["pair_id"], manifest["segmentation"]) == branch
        assert manifest["mode"] == "map_first"
        # The run's suffix (empty for human since its flip) and MENDER policy
        # travel in each clustered table (FINALIZE and MENDER_PREPARE keep
        # their legacy scripts).
        assert manifest["table_key_suffix"] == ""  # human has flipped (§20)
        assert manifest["mender_unassigned_state_policy"] == "exclude_from_features"
        # The harness has no src/ next to its project directory; the
        # fingerprint of the real sources is tested in test_annotation_module.
        assert manifest["hierarchy_fingerprint"] == "missing"
        # COMPUTE_CPU staged this branch's RESOLVE output.
        listing = (output / "stub_labels_listing.txt").read_text().split()
        assert f"{branch[0]}_resolve_summary.json" in listing


@needs_nextflow
def test_compute_resources_and_no_gpu(harness: dict[str, Any]) -> None:
    gib = 1024**3
    for row in _rows(harness["trace"], COMPUTE):
        assert row["cpus"] == "8"
        assert int(row["memory"]) == 32 * gib
        command = (Path(row["workdir"]) / ".command.run").read_text()
        assert "MERXEN_GPU_LOCK_FILE" not in command
        assert "--gpus" not in command


@needs_nextflow
def test_resume_keeps_compute_cached(harness: dict[str, Any]) -> None:
    computes = _rows(harness["resume_trace"], COMPUTE)
    assert {tuple(row["tag"].split(":")) for row in computes} == EXPECTED_BRANCHES
    assert {row["status"] for row in computes} == {"CACHED"}
    first = {
        (r["pair_id"], r["segmentation"]): r["computed_dir"]
        for r in harness["computed"]
    }
    again = {
        (r["pair_id"], r["segmentation"]): r["computed_dir"]
        for r in harness["resume_computed"]
    }
    assert first == again


@needs_nextflow
def test_compute_reruns_when_its_labels_change(harness: dict[str, Any]) -> None:
    """A RESOLVE-only change rewrote annotation_resolve_out: COMPUTE_CPU re-runs."""
    computes = _rows(harness["change_trace"], COMPUTE)
    assert {tuple(row["tag"].split(":")) for row in computes} == EXPECTED_BRANCHES
    assert {row["status"] for row in computes} == {"COMPLETED"}


def _tree(root: Path) -> dict[str, bytes]:
    """Relative path -> bytes of every file under ``root`` (links followed)."""
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@needs_nextflow
def test_compute_stays_cached_when_resolve_reruns_with_the_same_labels(
    harness: dict[str, Any],
) -> None:
    """The M4 review note: a deep cache on annotation_resolve_out (plan §3.4)."""
    trace = harness["rerun_trace"]
    resolves = _rows(trace, RESOLVE)
    assert {row["status"] for row in resolves} == {"COMPLETED"}
    assert {row["status"] for row in _rows(trace, MAP)} == {"CACHED"}
    computes = _rows(trace, COMPUTE)
    assert {tuple(row["tag"].split(":")) for row in computes} == EXPECTED_BRANCHES
    assert {row["status"] for row in computes} == {"CACHED"}
    before = {(r["pair_id"], r["segmentation"]): r for r in harness["change_labels"]}
    for row in harness["rerun_labels"]:
        branch = (row["pair_id"], row["segmentation"])
        # RESOLVE re-ran (its old task directory was moved aside) and wrote
        # the same bytes: new files with new timestamps, identical content.
        output = Path(row["resolve_dir"])
        old_dir = Path(before[branch]["resolve_dir"])
        moved = old_dir.parent.with_name(old_dir.parent.name + ".moved") / old_dir.name
        assert moved.is_dir()
        assert _tree(output) == _tree(moved)
    assert {
        (r["pair_id"], r["segmentation"]): r["computed_dir"]
        for r in harness["rerun_computed"]
    } == {
        (r["pair_id"], r["segmentation"]): r["computed_dir"]
        for r in harness["change_computed"]
    }


@needs_nextflow
def test_end_of_run_summary_lists_the_failed_pair_and_refused_panels(
    harness: dict[str, Any],
) -> None:
    """Hook H6: F7's PREP failed, so F7 has no label tables and is listed."""
    summary = harness["summary"]
    lines = {
        line.split(":", 1)[0].strip(): line.split(":", 1)[1].strip()
        for line in summary.splitlines()[1:]
        if ":" in line
    }
    assert summary.splitlines()[0].startswith(
        "Annotation summary (human, clustering_squidpy_mode map_first)"
    )
    assert lines["pair x segmentation branches clustered"] == str(
        len(EXPECTED_BRANCHES) + 1
    )
    failed = lines["failed annotations (no label tables; see the failed tasks above)"]
    assert failed == "F7:proseg_hybrid"
    assert lines["failed hierarchies (label tables, no COMPUTE_CPU output)"] == "none"
    refused = lines["refused panels"]
    assert "F4:proseg_hybrid" in refused and "PFEW:proseg_hybrid" in refused


# --------------------------------------------------------------------------
# Wiring (string tests)


def test_map_is_connected_after_the_required_bundle_join() -> None:
    text = SUBWORKFLOW.read_text()
    body = text[text.index("workflow CLUSTERING_ANNOTATE_MAP {") :]
    body = body[: body.index("workflow CLUSTERING_MAP_FIRST {")]
    assert "references = ANNOTATION_PREPARED_REFERENCES(prepared_ch)" in body
    assert "references.map_inputs.map {" in body
    assert "CLUSTERING_SQUIDPY_ANNOTATE_MAP(map_inputs_ch)" in body
    assert body.index("ANNOTATION_PREPARED_REFERENCES(") < body.index(
        "CLUSTERING_SQUIDPY_ANNOTATE_MAP("
    )
    # CLUSTERING_MAP_FIRST = CLUSTERING_ANNOTATE -> COMPUTE_CPU (M5).
    map_first = re.sub(
        r"//[^\n]*", "", text[text.index("workflow CLUSTERING_MAP_FIRST {") :]
    )
    assert "error(" not in map_first
    assert (
        "annotated = CLUSTERING_ANNOTATE(prepared_ch.combine(alignment_ch, by: 0), "
        "registration_ch)" in map_first
    )
    assert "CLUSTERING_SQUIDPY_COMPUTE_CPU(compute_inputs_ch)" in map_first
    assert (
        'AnnotationReferences.computeSpec(params, "${projectDir}/../src")' in map_first
    )
    assert map_first.index("CLUSTERING_ANNOTATE(") < map_first.index(
        "CLUSTERING_SQUIDPY_COMPUTE_CPU("
    )
    assert "computed = computed_ch" in map_first


def test_resolve_is_connected_after_map() -> None:
    text = SUBWORKFLOW.read_text()
    body = text[text.index("workflow CLUSTERING_ANNOTATE {") :]
    body = re.sub(
        r"//[^\n]*", "", body[: body.index("workflow CLUSTERING_MAP_FIRST {")]
    )
    assert "mapped = CLUSTERING_ANNOTATE_MAP(prepared_ch)" in body
    assert "resolve_inputs_ch = mapped.maps\n" in body
    assert ".join(alignment_by_branch_ch)" in body
    # Never an inner join on the registration checks: it would drop a branch.
    assert ".join(registration_by_branch_ch, remainder: true)" in body
    assert "registrationFiles ?: []" in body
    assert (
        'AnnotationReferences.resolveSpec(params, panelDir, "${projectDir}/../src")'
        in body
    )
    assert "CLUSTERING_SQUIDPY_ANNOTATE_RESOLVE(resolve_inputs_ch)" in body
    # Only the deterministic annotation_resolve_out flows downstream.
    assert "resolve_out_ch = CLUSTERING_SQUIDPY_ANNOTATE_RESOLVE.out.resolved" in body
    assert "run_record" not in body
    assert body.index("CLUSTERING_ANNOTATE_MAP(") < body.index(
        "CLUSTERING_SQUIDPY_ANNOTATE_RESOLVE("
    )
    # Only the subworkflow file calls RESOLVE.
    for path in WORKFLOWS.rglob("*.nf"):
        if path != SUBWORKFLOW:
            assert "CLUSTERING_SQUIDPY_ANNOTATE_RESOLVE(" not in path.read_text(), path


def test_main_nf_calls_map_first_only_through_hook_h5() -> None:
    """Legacy runs are unchanged: H5's map_first branch calls CLUSTERING_MAP_FIRST."""
    main_text = MAIN_NF.read_text()
    for name in (
        "CLUSTERING_SQUIDPY_ANNOTATE_MAP",
        "CLUSTERING_SQUIDPY_ANNOTATE_RESOLVE",
        "CLUSTERING_SQUIDPY_COMPUTE_CPU",
        "CLUSTERING_ANNOTATE_MAP",
        "CLUSTERING_ANNOTATE(",
    ):
        assert name not in main_text, name
    include = (
        'include { CLUSTERING_MAP_FIRST } from "./subworkflows/clustering_map_first"'
    )
    assert main_text.count(include) == 1
    assert main_text.count("CLUSTERING_MAP_FIRST(") == 1
    branch = main_text.index("    if (AnnotationSettings.isMapFirstRun(params)) {")
    legacy = main_text.index("\n    } else {\n", branch)
    assert branch < main_text.index("CLUSTERING_MAP_FIRST(") < legacy
    preflight = (LIB_DIR / "AnnotationPreflight.groovy").read_text()
    assert "MAP_FIRST_WIRED" not in preflight
    # Only the subworkflow calls COMPUTE_CPU.
    for path in WORKFLOWS.rglob("*.nf"):
        if path != SUBWORKFLOW:
            assert "CLUSTERING_SQUIDPY_COMPUTE_CPU(" not in path.read_text(), path


REGISTRATION_MAIN = """
workflow {
    def rows = new groovy.json.JsonSlurperClassic().parseText(params.rows)
    sample_rows_ch = channel.fromList(
        rows.collect { row -> tuple(row.pair_id, row, row.settings) }
    )
    // QC's tuple: key, pair, platform, segmentation, zarr, qc_out, table and
    // shape keys.
    def qcRows = new groovy.json.JsonSlurperClassic().parseText(params.qc)
    qc_results_ch = channel.fromList(
        qcRows.collect { item ->
            tuple(
                "key", item.pair_id, item.platform, item.segmentation, "x.zarr",
                file(item.qc_out), "table", "shape",
            )
        }
    )
__REGISTRATION__
    map_first_registration_ch
        .map { pairId, segmentation, files ->
            [pairId, segmentation, files.collect { f -> f.toString() }]
        }
        .collect(flat: false)
        .ifEmpty([])
        .subscribe { items ->
            new File(params.out).text = groovy.json.JsonOutput.toJson(items)
        }
}
"""


def _main_nf_block(main_text: str, start: str, end: str) -> str:
    """Return main.nf from the line starting with ``start`` up to ``end``."""
    begin = main_text.index(start)
    return main_text[begin : main_text.index(end, begin)]


def test_the_registration_branches_complement_the_qc_stage() -> None:
    """Hook H5 gives [] exactly to the branches qc_inputs_ch leaves out.

    A branch given [] although its QC runs could resolve before its checks
    arrive, so the two predicates must stay each other's negation.
    """
    main_text = MAIN_NF.read_text()
    qc_filter = _main_nf_block(
        main_text, "    qc_inputs_ch = analysis_layer_validation_results_ch", ".map {"
    )
    assert (
        "settings.run_qc && settings.analysis_segmentations.contains(_segmentation)"
        in qc_filter
    )
    block = _main_nf_block(
        main_text,
        "        map_first_registration_ch = qc_results_ch",
        "        map_first_alignment_ch = alignment_results_ch",
    )
    assert (
        "!(settings.run_qc && settings.analysis_segmentations.contains(segmentation))"
        in block
    )
    assert "settings.analysis_input_segmentations" in block
    assert "groupKey(" in block and ".groupTuple(remainder: true)" in block
    assert "AnnotationReferences.registrationQcFiles(qcOutDirs)" in block
    branch = main_text.index("    if (AnnotationSettings.isMapFirstRun(params)) {")
    legacy = main_text.index("\n    } else {\n", branch)
    assert branch < main_text.index(block) < legacy
    assert "            map_first_registration_ch,\n        ).computed" in main_text


@needs_nextflow
def test_map_first_registration_gives_one_tuple_per_branch(tmp_path: Path) -> None:
    """Hook H5's registration checks (M13 C15): main.nf's own block.

    Per pair x segmentation: the QC tasks' ``*_registration_qc.json`` (one
    per active platform, released once all have finished), ``[]`` for a
    branch the QC stage skips (``run_qc`` off, or a segmentation that is not
    an analysis segmentation) and for a QC output without a registration
    check. The QC summary CSV is never staged.
    """
    assert NEXTFLOW is not None
    block = _main_nf_block(
        MAIN_NF.read_text(),
        "        map_first_registration_ch = qc_results_ch",
        "        map_first_alignment_ch = alignment_results_ch",
    )
    outputs = {}
    for name, files in (
        (
            "p1_merscope",
            ["p1_merscope_registration_qc.json", "p1_merscope_qc_summary.csv"],
        ),
        ("p1_xenium", ["p1_xenium_registration_qc.json"]),
        ("p3_merscope", ["p3_merscope_qc_summary.csv"]),
    ):
        qc_out = tmp_path / name / "qc_out"
        qc_out.mkdir(parents=True)
        for file_name in files:
            (qc_out / file_name).write_text("{}")
        (qc_out / "nested").mkdir()
        (qc_out / "nested" / f"{name}_registration_qc.json").write_text("{}")
        outputs[name] = qc_out
    both = ["MERSCOPE", "XENIUM"]
    rows = [
        {
            "pair_id": "P1",
            "settings": {
                "active_platforms": both,
                "run_qc": True,
                "analysis_segmentations": ["proseg_hybrid"],
                "analysis_input_segmentations": ["proseg_hybrid", "reseg"],
            },
        },
        {
            "pair_id": "P2",
            "settings": {
                "active_platforms": both,
                "run_qc": False,
                "analysis_segmentations": ["proseg_hybrid"],
                "analysis_input_segmentations": ["proseg_hybrid"],
            },
        },
        {
            "pair_id": "P3",
            "settings": {
                "active_platforms": ["MERSCOPE"],
                "run_qc": True,
                "analysis_segmentations": ["proseg_hybrid"],
                "analysis_input_segmentations": ["proseg_hybrid"],
            },
        },
    ]
    qc = [
        {
            "pair_id": "P1",
            "platform": "MERSCOPE",
            "segmentation": "proseg_hybrid",
            "qc_out": str(outputs["p1_merscope"]),
        },
        {
            "pair_id": "P3",
            "platform": "MERSCOPE",
            "segmentation": "proseg_hybrid",
            "qc_out": str(outputs["p3_merscope"]),
        },
        {
            "pair_id": "P1",
            "platform": "XENIUM",
            "segmentation": "proseg_hybrid",
            "qc_out": str(outputs["p1_xenium"]),
        },
    ]
    (tmp_path / "lib").mkdir()
    for source in LIB_DIR.glob("*.groovy"):
        shutil.copy(source, tmp_path / "lib" / source.name)
    (tmp_path / "main.nf").write_text(
        REGISTRATION_MAIN.replace("__REGISTRATION__", block)
    )
    out = tmp_path / "registration.json"
    (tmp_path / "nextflow.config").write_text(
        f"params.rows = '{json.dumps(rows)}'\n"
        f"params.qc = '{json.dumps(qc)}'\n"
        f"params.out = '{out}'\n"
    )
    completed = subprocess.run(
        [NEXTFLOW, "-log", str(tmp_path / "nextflow.log"), "run", "main.nf"],
        cwd=tmp_path,
        env=_nextflow_env(),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    items = sorted(json.loads(out.read_text()))
    assert items == [
        [
            "P1",
            "proseg_hybrid",
            [
                str(outputs["p1_merscope"] / "p1_merscope_registration_qc.json"),
                str(outputs["p1_xenium"] / "p1_xenium_registration_qc.json"),
            ],
        ],
        ["P1", "reseg", []],
        ["P2", "proseg_hybrid", []],
        ["P3", "proseg_hybrid", []],
    ]


# --------------------------------------------------------------------------
# The real MAP script (fake merxen annotate)


@pytest.mark.slow
@needs_nextflow
def test_map_runs_the_real_script_with_the_pipeline_arguments(tmp_path: Path) -> None:
    """Without -stub-run: ctm check, env, arguments, reuse and a real refusal.

    A fake ``merxen`` writes fixture panels and bundle refs and records what
    ``annotate`` and ``annotate-resolve`` receive; for the refused panel it
    runs the real ``merxen annotate``, which writes the refusal without
    mapping, and the real ``merxen annotate-resolve``, which writes the
    statuses of every object.
    """
    fixtures = _write_fixture_panels(tmp_path)
    env = _fake_merxen_env(tmp_path, fixtures)
    _write_harness(tmp_path, entry="CLUSTERING_ANNOTATE")
    items = [
        _prepared_branch(tmp_path / "prepared", pair_id, segmentation, 60)
        for pair_id, segmentation in (
            ("F1", "proseg_hybrid"),
            ("F4", "proseg_hybrid"),
            ("F5", "original_seg"),
        )
    ]
    (tmp_path / "prepared_inputs.json").write_text(json.dumps(items))
    _write_registration(
        tmp_path, [(item["pair_id"], item["segmentation"]) for item in items]
    )
    _write_config(tmp_path, failing=False, slow=False)
    # F1 has a published MAP output: its task reuses from there.
    published = tmp_path / "results/F1/proseg_hybrid/annotation_map/annotation_map_out"
    published.mkdir(parents=True)
    (published / "map_manifest.json").write_text("{}")

    completed = _run(tmp_path, env)

    assert completed.returncode == 0, completed.stdout + completed.stderr
    maps = {
        (row["pair_id"], row["segmentation"]): row
        for row in json.loads((tmp_path / "maps_out.json").read_text())
    }
    assert set(maps) == {
        ("F1", "proseg_hybrid"),
        ("F4", "proseg_hybrid"),
        ("F5", "original_seg"),
    }
    records = {}
    for path in (tmp_path / "work").glob("*/*/fake_annotate.json"):
        record = json.loads(path.read_text())
        argv = record["argv"]
        records[
            (argv[argv.index("--pair-id") + 1], argv[argv.index("--segmentation") + 1])
        ] = record
    assert set(records) == set(maps)
    for branch, record in records.items():
        argv, environment = record["argv"], record["env"]
        assert argv[argv.index("--n-processors") + 1] == "6"
        assert argv[argv.index("--prepared-dir") + 1] == (
            "map_inputs/clustering_prepare_out"
        )
        assert argv[argv.index("--clustering-config") + 1] == (
            "map_inputs/clustering_squidpy_config.json"
        )
        assert argv[argv.index("--panel-dir") + 1] == "map_inputs/annotation_panel_out"
        assert argv[argv.index("--work-dir") + 1] == "map_scratch"
        assert argv[argv.index("--out") + 1] == "annotation_map_out"
        assert "--require-bundle-refs" in argv and "--allow-refused-panel" in argv
        config = AnnotationConfig.model_validate_json(
            (Path(maps[branch]["map_dir"]) / "annotation_config.json").read_text()
        )
        assert config.species == ("mouse" if branch[0] == "F5" else "human")
        assert config.ctm_version == "1.7.2"
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "NUMBA_NUM_THREADS",
        ):
            assert environment[name] == "1", (branch, name)
        assert environment["CUDA_VISIBLE_DEVICES"] == ""
        # ${projectDir}/../src first: the harness main.nf is the project here.
        first_entry = environment["PYTHONPATH"].split(os.pathsep)[0]
        assert os.path.realpath(first_entry) == os.path.realpath(
            tmp_path.parent / "src"
        )
    assert sorted(
        (ref["reference_id"], ref["panel_hash"])
        for ref in records[("F1", "proseg_hybrid")]["refs"]
    ) == sorted(
        (reference_id, _hash(tag) if tag else None)
        for reference_id, _s, _r, _p, tag, _n in FIXTURE_BRANCHES[
            ("F1", "proseg_hybrid")
        ]
    )
    f1 = records[("F1", "proseg_hybrid")]["argv"]
    assert f1[f1.index("--reuse-from") + 1] == str(published)
    assert "--reuse-from" not in records[("F5", "original_seg")]["argv"]
    assert [ref["reference_id"] for ref in records[("F5", "original_seg")]["refs"]] == [
        "wmb_panel",
        "wmb_region_share",
    ]
    # The refused panel ran the real command: a refusal manifest, no mapping.
    refused = json.loads(
        (
            Path(maps[("F4", "proseg_hybrid")]["map_dir"]) / "map_manifest.json"
        ).read_text()
    )
    assert refused["panel_status"] == "refused"
    assert refused["samples"] == {}
    assert refused["min_counts"] == 10
    assert records[("F4", "proseg_hybrid")]["refs"] == []
    # Task-local scratch is removed after a successful MAP.
    for record_path in (tmp_path / "work").glob("*/*/fake_annotate.json"):
        assert not (record_path.parent / "map_scratch").exists()

    # RESOLVE: after each MAP, on the staged inputs and the same refs.
    labels = {
        (row["pair_id"], row["segmentation"]): row
        for row in json.loads((tmp_path / "labels_out.json").read_text())
    }
    assert set(labels) == set(maps)
    resolve_records = {}
    record_path_dirs = {}
    for path in (tmp_path / "work").glob("*/*/fake_annotate_resolve.json"):
        required = json.loads(
            (
                path.parent
                / "resolve_inputs/annotation_panel_out/required_bundles.json"
            ).read_text()
        )
        key = (required["pair_id"], required["segmentation"])
        resolve_records[key] = json.loads(path.read_text())
        record_path_dirs[key] = path.parent
    assert set(resolve_records) == set(maps)
    for branch, record in resolve_records.items():
        argv, environment = record["argv"], record["env"]
        assert argv[0] == "annotate-resolve"
        for option, staged in (
            ("--annotation-config", "annotation_config.json"),
            ("--map-dir", "resolve_inputs/annotation_map_out"),
            ("--panel-dir", "resolve_inputs/annotation_panel_out"),
            ("--prepared-dir", "resolve_inputs/clustering_prepare_out"),
            ("--clustering-config", "resolve_inputs/clustering_squidpy_config.json"),
            ("--out", "annotation_resolve_out"),
            ("--run-record", "annotation_resolve_run.json"),
        ):
            assert argv[argv.index(option) + 1] == staged, (branch, option)
        assert "--require-bundle-refs" in argv and "--no-alignment-lookup" in argv
        assert "--alignment-dir" not in argv
        # The QC stage's registration checks (G1; M13 C15), when staged.
        if branch == REGISTERED_BRANCH:
            option = argv.index("--registration-qc-dir")
            assert argv[option + 1] == "resolve_inputs/registration_qc"
            staged = record_path_dirs[branch] / "resolve_inputs" / "registration_qc"
            assert sorted(path.name for path in staged.iterdir()) == sorted(
                REGISTRATION_FILES
            )
        else:
            assert "--registration-qc-dir" not in argv, branch
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "NUMBA_NUM_THREADS",
        ):
            assert environment[name] == "1", (branch, name)
        assert environment["CUDA_VISIBLE_DEVICES"] == ""
        first_entry = environment["PYTHONPATH"].split(os.pathsep)[0]
        assert os.path.realpath(first_entry) == os.path.realpath(
            tmp_path.parent / "src"
        )
        # The refs MAP mapped with, in the same order.
        assert record["refs"] == records[branch]["refs"], branch
        config = AnnotationConfig.model_validate_json(
            (Path(labels[branch]["resolve_dir"]) / "annotation_config.json").read_text()
        )
        assert config.species == ("mouse" if branch[0] == "F5" else "human")
    # The refused panel ran the real RESOLVE: statuses only, gate failed.
    refused_out = Path(labels[("F4", "proseg_hybrid")]["resolve_dir"])
    summary = json.loads((refused_out / "F4_resolve_summary.json").read_text())
    assert summary["panel_status"] == "refused"
    assert set(summary["samples"]) == {"F4_MERSCOPE", "F4_XENIUM"}
    for sample_id, sample in summary["samples"].items():
        assert sample["resolution"]["gate"]["level"] == "failed", sample_id
        platform = sample_id.split("_")[-1].lower()
        assert (
            refused_out / platform / f"{sample_id}_celltype_labels.parquet"
        ).is_file()
    published = (
        tmp_path
        / "results/F4/proseg_hybrid/annotation_resolve/annotation_resolve_out"
        / "F4_resolve_summary.json"
    )
    assert published.is_file()

"""CLUSTERING_ANNOTATE_MAP: MAP after the required-bundle join (plan §3.1, §3.3).

``workflows/subworkflows/clustering_map_first.nf`` runs
``ANNOTATION_PREPARED_REFERENCES`` (ANNOTATE_PANEL, one ANNOTATE_REFERENCE_PREP
per unique bundle, the per pair x segmentation ``groupKey`` collection) and
then ``CLUSTERING_SQUIDPY_ANNOTATE_MAP`` on each released pair x
segmentation. When ``nextflow`` is installed, a harness runs it with
``-stub-run`` (PREP and MAP write stub outputs; ANNOTATE_PANEL runs for real
on synthetic prepared H5ADs, or, for the fixture branches, a fake
``merxen annotation-panel`` writes hand-written ``required_bundles.json``
files: same-panel set c, ``per_platform``, mouse, refused, a failing and a
slow PREP, a large panel) and checks that:

* MAP starts for a pair x segmentation only after every PREP it needs has
  completed, and never waits for another branch's bundles;
* MAP receives exactly the bundle refs its ``required_bundles.json`` lists
  (none for a refused panel), and a failed PREP drops only its branches;
* MAP's resources follow the plan (6 CPUs; 24 GB, 48 GB above 1,000 genes);
* a ``-resume`` run re-runs PREP (never cached) but keeps MAP cached, because
  MAP hashes its inputs' content (``cache "deep"``).

A second run executes the real MAP script (no ``-stub-run``) with a fake
``merxen annotate`` that records its command line and environment (slow).
``CLUSTERING_MAP_FIRST`` itself still refuses to run until M5, and
``main.nf`` never calls any of it (legacy runs are unchanged).
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
    }


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
# else run the real command. annotate (real MAP script only): record the
# command line and environment; run the real command for a refused panel
# (it only writes the refusal) and write a minimal manifest otherwise.
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
elif command == "annotate":
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
    Path("fake_annotate.json").write_text(json.dumps(record))
    required = json.loads(
        (Path(value("--panel-dir")) / "required_bundles.json").read_text()
    )
    if required["status"] != "refused":
        out = Path(value("--out"))
        out.mkdir(parents=True)
        (out / "map_manifest.json").write_text(json.dumps({"fake": True}))
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
}
"""

HARNESS_MAIN = """
include { __ENTRY__ } from '__SUBWORKFLOW__'

workflow {
    prepared_ch = channel
        .fromList(MapHarness.readList(params.prepared_inputs))
        .map { item ->
            tuple(
                item.pair_id, item.segmentation, item.samples_json,
                file(item.config), file(item.prepared_dir),
            )
        }
    mapped = __ENTRY__(prepared_ch)
    __EMIT__
}
"""

EMIT_MAPS = """mapped.maps
        .map { item -> MapHarness.mapRow(item) }
        .collect()
        .subscribe { rows -> MapHarness.write(params.maps_out, rows) }"""


def _write_harness(root: Path, *, entry: str = "CLUSTERING_ANNOTATE_MAP") -> None:
    (root / "lib").mkdir(parents=True)
    for source in LIB_DIR.glob("*.groovy"):
        shutil.copy(source, root / "lib" / source.name)
    (root / "lib" / "MapHarness.groovy").write_text(HARNESS_CLASS)
    emit = EMIT_MAPS if entry == "CLUSTERING_ANNOTATE_MAP" else ""
    (root / "main.nf").write_text(
        HARNESS_MAIN.replace("__ENTRY__", entry)
        .replace("__SUBWORKFLOW__", str(SUBWORKFLOW))
        .replace("__EMIT__", emit)
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
    return items


def _write_config(root: Path, *, failing: bool = True, slow: bool = True) -> Path:
    """Write the harness config (a failing and a slow PREP when asked)."""
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
    maps_out = "{root / "maps_out.json"}"
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
    fields = "task_id,process,tag,status,submit,start,complete,cpus,memory"
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


def _run_harness(root: Path) -> dict[str, Any]:
    """Run the stub harness, then the same run again with ``-resume``."""
    fixtures = _write_fixture_panels(root)
    env = _fake_merxen_env(root, fixtures)
    _write_harness(root)
    _write_inputs(root)
    _write_config(root)
    first = _run(root, env, "-stub-run")
    assert first.returncode == 0, first.stdout + first.stderr
    trace = _read_trace(root / "trace.tsv")
    maps = json.loads((root / "maps_out.json").read_text())
    (root / "maps_out.json").unlink()
    # PREP never caches; the resume run skips the slow PREP's wait only.
    _write_config(root, slow=False)
    second = _run(root, env, "-stub-run", "-resume")
    assert second.returncode == 0, second.stdout + second.stderr
    return {
        "root": str(root),
        "trace": trace,
        "maps": maps,
        "resume_trace": _read_trace(root / "trace.tsv"),
        "resume_maps": json.loads((root / "maps_out.json").read_text()),
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


@needs_nextflow
def test_map_first_still_refuses_to_run(tmp_path: Path) -> None:
    """CLUSTERING_MAP_FIRST fails before any task until M5 wires RESOLVE/COMPUTE_CPU."""
    _write_harness(tmp_path, entry="CLUSTERING_MAP_FIRST")
    _write_inputs(tmp_path)
    _write_config(tmp_path, failing=False, slow=False)
    completed = _run(tmp_path, _nextflow_env(), "-stub-run")
    assert completed.returncode != 0
    output = completed.stdout + completed.stderr
    assert "CLUSTERING_MAP_FIRST has no RESOLVE (M4) or COMPUTE_CPU (M5) yet" in output
    work = tmp_path / "work"
    assert not work.exists() or not any(work.iterdir())


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
    # CLUSTERING_MAP_FIRST keeps its guard until M5.
    guard = re.sub(
        r"//[^\n]*", "", text[text.index("workflow CLUSTERING_MAP_FIRST {") :]
    )
    assert "error(" in guard
    assert "CLUSTERING_ANNOTATE_MAP(" not in guard


def test_main_nf_never_calls_the_map_step() -> None:
    """Legacy runs are unchanged: main.nf only includes CLUSTERING_MAP_FIRST (H1)."""
    main_text = MAIN_NF.read_text()
    for name in (
        "CLUSTERING_SQUIDPY_ANNOTATE_MAP",
        "CLUSTERING_ANNOTATE_MAP",
        "CLUSTERING_MAP_FIRST(",
    ):
        assert name not in main_text, name
    include = (
        'include { CLUSTERING_MAP_FIRST } from "./subworkflows/clustering_map_first"'
    )
    assert main_text.count(include) == 1
    preflight = (LIB_DIR / "AnnotationPreflight.groovy").read_text()
    assert re.search(r"static final boolean MAP_FIRST_WIRED = false\b", preflight)


# --------------------------------------------------------------------------
# The real MAP script (fake merxen annotate)


@pytest.mark.slow
@needs_nextflow
def test_map_runs_the_real_script_with_the_pipeline_arguments(tmp_path: Path) -> None:
    """Without -stub-run: ctm check, env, arguments, reuse and a real refusal.

    A fake ``merxen`` writes fixture panels and bundle refs and records what
    ``annotate`` receives; for the refused panel it runs the real
    ``merxen annotate``, which writes the refusal without mapping.
    """
    fixtures = _write_fixture_panels(tmp_path)
    env = _fake_merxen_env(tmp_path, fixtures)
    _write_harness(tmp_path)
    items = [
        _prepared_branch(tmp_path / "prepared", pair_id, segmentation, 60)
        for pair_id, segmentation in (
            ("F1", "proseg_hybrid"),
            ("F4", "proseg_hybrid"),
            ("F5", "original_seg"),
        )
    ]
    (tmp_path / "prepared_inputs.json").write_text(json.dumps(items))
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

"""ANNOTATION_REPORT and ANNOTATION_REPORTING (plan §3.6, §9; M7).

``workflows/subworkflows/annotation_report.nf`` runs ``ANNOTATION_REPORT``
(``workflows/modules/annotation.nf``) once per map_first pair x
segmentation, after FINALIZE, the pair's cortical depth and the branch's
MENDER when those run; main.nf calls it at hook H11, in map_first runs only.
When ``nextflow`` is installed, a harness feeds the subworkflow synthetic
label, FINALIZE, ALIGN and row channels, and fake depth and MENDER processes
(one slow, one failing), and checks that:

* a report runs once per labelled branch FINALIZE finished, after every
  depth output and MENDER output it waits for, and a branch that waits for
  neither starts at once;
* a failed depth task drops only its branch's report, which the end-of-run
  summary lists; a branch without a FINALIZE output gets no report;
* the report stages RESOLVE, MAP, PANEL, FINALIZE, each platform's depth
  and MENDER output (under that platform's sample id) and the ALIGN files,
  and publishes under ``<pair>/<seg>/annotation_report``;
* its resources follow the plan (4 CPUs, 32 GB) and ``-resume`` keeps it
  cached;
* without ``-stub-run`` the real script calls ``merxen annotation-report``
  with the same arguments, CPU only, and a failed report item reaches the
  end-of-run summary.

The Groovy helpers (``AnnotationReport``) are checked on edge cases in the
same run.
"""

from __future__ import annotations

import csv
import fcntl
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / "workflows"
SUBWORKFLOW = WORKFLOWS / "subworkflows" / "annotation_report.nf"
MODULE = WORKFLOWS / "modules" / "annotation.nf"
MAP_FIRST = WORKFLOWS / "subworkflows" / "clustering_map_first.nf"
ANNOTATION_CONFIG = WORKFLOWS / "conf" / "annotation.config"
DWIGHT_ANNOTATION_CONFIG = WORKFLOWS / "conf" / "dwight.annotation.config"
LIB_DIR = WORKFLOWS / "lib"
MAIN_NF = WORKFLOWS / "main.nf"
NEXTFLOW = shutil.which("nextflow")
needs_nextflow = pytest.mark.skipif(NEXTFLOW is None, reason="nextflow is unavailable")
REPORT = "ANNOTATION_REPORT"
SLOW_SECONDS = 4

# (pair, segmentation) -> platforms of its samples.
BRANCHES: dict[tuple[str, str], list[str]] = {
    # Waits for both depth outputs (Xenium's slow) and both MENDER outputs
    # (MERSCOPE's slow); has an alignment.
    ("P1", "proseg_hybrid"): ["MERSCOPE", "XENIUM"],
    # Neither depth nor MENDER runs for reseg: released at once.
    ("P1", "reseg"): ["MERSCOPE", "XENIUM"],
    # Its Xenium depth task fails: no report, listed in the summary.
    ("P2", "proseg_hybrid"): ["MERSCOPE", "XENIUM"],
    # A single-platform section (as a mouse section): one depth, one MENDER.
    ("P3", "original_seg"): ["MERSCOPE"],
    # Labelled but FINALIZE never finished (COMPUTE_CPU failed): no report.
    ("P4", "proseg_hybrid"): ["MERSCOPE", "XENIUM"],
}
FINALIZED = {branch for branch in BRANCHES if branch[0] != "P4"}
ALIGNED_PAIRS = {"P1"}
# Row settings (rowSampleSettings keys AnnotationReport.pairSpec reads).
ROWS: dict[str, dict[str, Any]] = {
    "P1": {
        "run_clustering_squidpy": True,
        "run_compute_cortical_depth": True,
        "run_distance_from_object": False,
        "run_mender": True,
        "active_platforms": ["XENIUM", "MERSCOPE"],
        "analysis_segmentations": ["proseg_hybrid"],
        "distance_from_object_segmentations": ["reseg"],
        "mender_segmentations": ["proseg_hybrid"],
    },
    "P2": {
        "run_clustering_squidpy": True,
        "run_compute_cortical_depth": True,
        "run_distance_from_object": False,
        "run_mender": False,
        "active_platforms": ["MERSCOPE", "XENIUM"],
        "analysis_segmentations": ["proseg_hybrid"],
        "distance_from_object_segmentations": [],
        "mender_segmentations": ["proseg_hybrid"],
    },
    "P3": {
        "run_clustering_squidpy": True,
        "run_compute_cortical_depth": True,
        "run_distance_from_object": False,
        "run_mender": True,
        "active_platforms": ["MERSCOPE"],
        "analysis_segmentations": ["original_seg"],
        "distance_from_object_segmentations": [],
        "mender_segmentations": ["original_seg"],
    },
    "P4": {
        "run_clustering_squidpy": True,
        "run_compute_cortical_depth": False,
        "run_distance_from_object": False,
        "run_mender": False,
        "active_platforms": ["MERSCOPE", "XENIUM"],
        "analysis_segmentations": ["proseg_hybrid"],
        "distance_from_object_segmentations": [],
        "mender_segmentations": [],
    },
}
# Fake depth tasks: (pair, platform) -> (delay seconds, fails).
DEPTH_TASKS: dict[tuple[str, str], tuple[int, bool]] = {
    ("P1", "MERSCOPE"): (0, False),
    ("P1", "XENIUM"): (SLOW_SECONDS, False),
    ("P2", "MERSCOPE"): (0, False),
    ("P2", "XENIUM"): (0, True),
    ("P3", "MERSCOPE"): (0, False),
}
# Fake MENDER tasks: (pair, segmentation, platform) -> delay seconds.
MENDER_TASKS: dict[tuple[str, str, str], int] = {
    ("P1", "proseg_hybrid", "MERSCOPE"): SLOW_SECONDS,
    ("P1", "proseg_hybrid", "XENIUM"): 0,
    ("P3", "original_seg", "MERSCOPE"): 0,
}
# The branch whose (fake) report records a failed item in the real run.
FAILED_ITEM_BRANCH = ("P3", "original_seg")
FAILED_ITEM = "item04_reference_expectation"

HARNESS_CLASS = """
import groovy.json.JsonOutput
import groovy.json.JsonSlurperClassic

class ReportHarness {
    static List readList(Object path) {
        return new JsonSlurperClassic().parse(new File(path.toString())) as List
    }

    static void write(Object path, Object rows) {
        new File(path.toString()).text = JsonOutput.toJson(rows)
    }

    static void writeText(Object path, String text) {
        new File(path.toString()).text = text
    }

    static Map evaluate(Map c, Map params) {
        try {
            switch (c.fn) {
                case "pairSpec":
                    def runParams = c.params ?: params
                    return [value: AnnotationReport.pairSpec(c.settings, runParams)]
                case "branchSpec":
                    return [value: AnnotationReport.branchSpec(
                        c.pair_spec, c.segmentation, c.samples_json,
                    )]
                case "enabled":
                    return [value: AnnotationReport.enabled(c.params)]
                case "sortedByPlatform":
                    def pairs = AnnotationReport.sortedByPlatform(
                        c.platforms, c.outputs,
                    )
                    return [value: pairs]
                case "reportArguments":
                    return [value: AnnotationReport.reportArguments(
                        [species: "human", report_fingerprint: "f"],
                        c.pair_id, c.segmentation, c.samples_json,
                        c.depth_platforms, c.depth_dirs,
                        c.mender_platforms, c.mender_dirs,
                        c.alignment_files,
                    )]
            }
            throw new IllegalStateException("unknown case ${c.fn}")
        } catch (IllegalArgumentException error) {
            return [error: error.message]
        }
    }

    static void cases(Object inPath, Object outPath, Map params) {
        def cases = new JsonSlurperClassic().parse(new File(inPath.toString())) as Map
        def results = [:]
        cases.each { name, c -> results[name] = evaluate(c as Map, params) }
        write(outPath, results)
    }
}
"""

HARNESS_MAIN = """
include { ANNOTATION_REPORTING } from '__SUBWORKFLOW__'

process FAKE_DEPTH {
    tag "${pair_id}:${platform}"

    input:
    tuple val(pair_id), val(platform), val(delay), val(fail)

    output:
    tuple val(pair_id), val(platform), path("compute_cortical_depth_out")

    script:
    def lower = platform.toLowerCase()
    \"\"\"
    sleep ${delay}
    if ${fail}; then exit 3; fi
    mkdir -p compute_cortical_depth_out/segmentation
    echo ${platform} > compute_cortical_depth_out/platform.txt
    \"\"\"
}

process FAKE_MENDER {
    tag "${pair_id}:${platform}:${segmentation}"

    input:
    tuple val(pair_id), val(segmentation), val(platform), val(delay)

    output:
    tuple val(pair_id), val(segmentation), val(platform),
        path("mender_out/${platform.toLowerCase()}")

    script:
    def lower = platform.toLowerCase()
    \"\"\"
    sleep ${delay}
    mkdir -p mender_out/${lower}
    echo '{"platform": "${platform}"}' > mender_out/${lower}/mender_manifest.json
    \"\"\"
}

workflow {
    ReportHarness.cases(params.cases_in, params.cases_out, params)
    labels_ch = channel
        .fromList(ReportHarness.readList(params.labels_inputs))
        .map { item ->
            tuple(
                item.pair_id, item.segmentation, item.samples_json,
                file(item.config), file(item.prepared_dir), file(item.panel_dir),
                item.refs.collect { path -> file(path) },
                file(item.map_dir), file(item.resolve_dir),
            )
        }
    alignment_ch = channel
        .fromList(ReportHarness.readList(params.alignment_inputs))
        .map { item -> tuple(item.pair_id, item.files.collect { path -> file(path) }) }
    clustered_ch = channel
        .fromList(ReportHarness.readList(params.clustered_inputs))
        .map { item ->
            tuple(
                item.pair_id, item.segmentation, item.samples_json,
                file(item.clustering_dir),
            )
        }
    depth_ch = FAKE_DEPTH(
        channel.fromList(ReportHarness.readList(params.depth_tasks))
            .map { item -> tuple(item.pair_id, item.platform, item.delay, item.fail) }
    )
    mender_ch = FAKE_MENDER(
        channel.fromList(ReportHarness.readList(params.mender_tasks))
            .map { item ->
                tuple(item.pair_id, item.segmentation, item.platform, item.delay)
            }
    )
    pair_specs_ch = channel
        .fromList(ReportHarness.readList(params.rows))
        .map { item ->
            tuple(item.pair_id, AnnotationReport.pairSpec(item.settings, params))
        }
    reports = ANNOTATION_REPORTING(
        labels_ch, alignment_ch, clustered_ch, depth_ch, mender_ch, pair_specs_ch,
    )
    reports
        .map { item ->
            [pair_id: item[0], segmentation: item[1], report_dir: item[2].toString()]
        }
        .collect()
        .subscribe { rows -> ReportHarness.write(params.reports_out, rows) }
    def harnessParams = params
    def harnessWorkflow = workflow
    harnessWorkflow.onComplete {
        ReportHarness.writeText(
            harnessParams.summary_out,
            AnnotationSettings.completionSummary(
                harnessParams,
                AnnotationRunRecord.runInfo() + [success: harnessWorkflow.success],
            ),
        )
    }
}
"""

# A fake merxen for the real-script run: records argv and environment, and
# writes the outputs the pipeline reads (report_run.json with item statuses).
FAKE_MERXEN = """#!__PYTHON__
import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
out = Path(args[args.index("--out") + 1])
out.mkdir(parents=True)
pair = args[args.index("--pair") + 1]
segmentation = args[args.index("--segmentation") + 1]
failed = (pair, segmentation) == tuple(__FAILED_BRANCH__)
status = "failed" if failed else "ok"
(out / "report_run.json").write_text(json.dumps({"items": {
    "item01_annotatability": "ok", "__FAILED_ITEM__": status,
}}))
(out / "acceptance_metrics.json").write_text("{}")
(out / "report.html").write_text("<html></html>")
env = {name: os.environ.get(name) for name in (
    "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "MPLBACKEND", "PYTHONPATH",
)}
(out / "fake_call.json").write_text(json.dumps({
    "argv": sys.argv[1:], "env": env, "cwd": os.getcwd(),
}))
"""


def _nextflow_env() -> dict[str, str]:
    env = {**os.environ, "NXF_DISABLE_CHECK_LATEST": "true", "NXF_ANSI_LOG": "false"}
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT / "src"), *filter(None, [os.environ.get("PYTHONPATH")])]
    )
    env["PATH"] = os.pathsep.join([str(Path(sys.executable).parent), env["PATH"]])
    env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def _samples_json(pair_id: str, platforms: list[str]) -> str:
    return json.dumps(
        [
            {
                "sample_id": f"{pair_id}_{platform}",
                "platform": platform,
                "zarr_path": f"/data/{pair_id}_{platform}.zarr",
            }
            for platform in platforms
        ]
    )


def _directory(path: Path, name: str, text: str) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / name).write_text(text)
    return path


# Groovy helper cases: name -> call.
CASES: dict[str, dict[str, Any]] = {
    "pairSpec|depth-and-distance": {
        "fn": "pairSpec",
        "settings": {
            **ROWS["P1"],
            "run_distance_from_object": True,
            "distance_from_object_segmentations": ["reseg", "proseg_hybrid"],
        },
    },
    "pairSpec|no-clustering": {
        "fn": "pairSpec",
        "settings": {**ROWS["P1"], "run_clustering_squidpy": False},
    },
    "pairSpec|disabled": {
        "fn": "pairSpec",
        "settings": ROWS["P1"],
        "params": {"annotation_report_enabled": "false"},
    },
    "enabled|unset": {"fn": "enabled", "params": {}},
    "enabled|false": {"fn": "enabled", "params": {"annotation_report_enabled": False}},
    "branchSpec|mender-from-samples": {
        "fn": "branchSpec",
        "pair_spec": {
            "enabled": True,
            "platforms": ["MERSCOPE", "XENIUM"],
            "depth_segmentations": ["reseg"],
            "mender_segmentations": ["proseg_hybrid"],
        },
        "segmentation": "proseg_hybrid",
        "samples_json": _samples_json("PX", ["XENIUM"]),
    },
    "sortedByPlatform": {
        "fn": "sortedByPlatform",
        "platforms": ["XENIUM", "MERSCOPE"],
        "outputs": ["x_dir", "m_dir"],
    },
    "reportArguments|quoted": {
        "fn": "reportArguments",
        "pair_id": "P 9",
        "segmentation": "proseg_hybrid",
        "samples_json": json.dumps(
            [{"sample_id": "P 9_MERSCOPE", "platform": "MERSCOPE"}]
        ),
        "depth_platforms": [],
        "depth_dirs": [],
        "mender_platforms": [],
        "mender_dirs": [],
        "alignment_files": [],
    },
    "reportArguments|mismatch": {
        "fn": "reportArguments",
        "pair_id": "P1",
        "segmentation": "proseg_hybrid",
        "samples_json": _samples_json("P1", ["MERSCOPE"]),
        "depth_platforms": ["MERSCOPE", "XENIUM"],
        "depth_dirs": ["report_inputs/cortical_depth_1/compute_cortical_depth_out"],
        "mender_platforms": [],
        "mender_dirs": [],
        "alignment_files": [],
    },
}


def _write_inputs(root: Path) -> None:
    labels, clustered, alignment = [], [], []
    for (pair_id, segmentation), platforms in BRANCHES.items():
        base = root / "inputs" / pair_id / segmentation
        samples = _samples_json(pair_id, platforms)
        tag = f"{pair_id}:{segmentation}"
        labels.append(
            {
                "pair_id": pair_id,
                "segmentation": segmentation,
                "samples_json": samples,
                "config": str(
                    _directory(base, "clustering_squidpy_config.json", "{}")
                    / "clustering_squidpy_config.json"
                ),
                "prepared_dir": str(
                    _directory(base / "clustering_prepare_out", "x.txt", tag)
                ),
                "panel_dir": str(
                    _directory(base / "annotation_panel_out", "panel_report.json", tag)
                ),
                "refs": [],
                "map_dir": str(
                    _directory(base / "annotation_map_out", "map_manifest.json", tag)
                ),
                "resolve_dir": str(
                    _directory(
                        base / "annotation_resolve_out",
                        f"{pair_id}_resolve_summary.json",
                        tag,
                    )
                ),
            }
        )
        if (pair_id, segmentation) in FINALIZED:
            clustering = base / "clustering_squidpy_out"
            for platform in platforms:
                _directory(
                    clustering / platform.lower(),
                    f"{pair_id}_{platform}_clustered.h5ad",
                    tag,
                )
            clustered.append(
                {
                    "pair_id": pair_id,
                    "segmentation": segmentation,
                    "samples_json": samples,
                    "clustering_dir": str(clustering),
                }
            )
    for pair_id in ROWS:
        files: list[str] = []
        if pair_id in ALIGNED_PAIRS:
            align = root / "inputs" / pair_id / "align_out"
            align.mkdir(parents=True)
            for name in ("shared_tissue_mask.npy", "registration_summary.json"):
                (align / name).write_text(name)
                files.append(str(align / name))
        alignment.append({"pair_id": pair_id, "files": files})
    payloads = {
        "labels_inputs.json": labels,
        "clustered_inputs.json": clustered,
        "alignment_inputs.json": alignment,
        "rows.json": [
            {"pair_id": pair_id, "settings": settings}
            for pair_id, settings in ROWS.items()
        ],
        "depth_tasks.json": [
            {"pair_id": pair, "platform": platform, "delay": delay, "fail": fail}
            for (pair, platform), (delay, fail) in DEPTH_TASKS.items()
        ],
        "mender_tasks.json": [
            {
                "pair_id": pair,
                "segmentation": segmentation,
                "platform": platform,
                "delay": delay,
            }
            for (pair, segmentation, platform), delay in MENDER_TASKS.items()
        ],
        "cases_in.json": CASES,
    }
    for name, payload in payloads.items():
        (root / name).write_text(json.dumps(payload))


def _write_harness(root: Path) -> None:
    (root / "lib").mkdir(parents=True)
    for source in LIB_DIR.glob("*.groovy"):
        shutil.copy(source, root / "lib" / source.name)
    (root / "lib" / "ReportHarness.groovy").write_text(HARNESS_CLASS)
    (root / "main.nf").write_text(
        HARNESS_MAIN.replace("__SUBWORKFLOW__", str(SUBWORKFLOW))
    )
    (root / "nextflow.config").write_text(
        f"""
includeConfig '{ANNOTATION_CONFIG}'
params {{
    species = "human"
    clustering_squidpy_mode = "map_first"
    outdir = "{root / "results"}"
    labels_inputs = "{root / "labels_inputs.json"}"
    clustered_inputs = "{root / "clustered_inputs.json"}"
    alignment_inputs = "{root / "alignment_inputs.json"}"
    rows = "{root / "rows.json"}"
    depth_tasks = "{root / "depth_tasks.json"}"
    mender_tasks = "{root / "mender_tasks.json"}"
    cases_in = "{root / "cases_in.json"}"
    cases_out = "{root / "cases_out.json"}"
    reports_out = "{root / "reports_out.json"}"
    summary_out = "{root / "summary_out.txt"}"
}}
executor {{
    cpus = 64
    memory = "512 GB"
}}
process {{
    executor = "local"
    errorStrategy = "ignore"
}}
trace {{
    enabled = true
    overwrite = true
    raw = true
    file = "{root / "trace.tsv"}"
    fields = "task_id,process,tag,status,start,complete,cpus,memory,workdir"
}}
"""
    )


def _fake_merxen_env(root: Path) -> dict[str, str]:
    env = _nextflow_env()
    bin_dir = root / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "merxen"
    fake.write_text(
        FAKE_MERXEN.replace("__PYTHON__", sys.executable)
        .replace("__FAILED_BRANCH__", json.dumps(list(FAILED_ITEM_BRANCH)))
        .replace("__FAILED_ITEM__", FAILED_ITEM)
    )
    fake.chmod(0o755)
    env["PATH"] = os.pathsep.join([str(bin_dir), env["PATH"]])
    return env


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


def _read_trace(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _collect(root: Path) -> dict[str, Any]:
    results = {
        "trace": _read_trace(root / "trace.tsv"),
        "reports": json.loads((root / "reports_out.json").read_text()),
        "summary": (root / "summary_out.txt").read_text(),
        "cases": json.loads((root / "cases_out.json").read_text()),
    }
    for name in ("reports_out.json", "summary_out.txt", "cases_out.json"):
        (root / name).unlink()
    return results


def _run_harness(base: Path) -> dict[str, Any]:
    """Run the stub harness, a -resume run, then the real script (fake merxen)."""
    stub_root = base / "stub"
    stub_root.mkdir(parents=True)
    _write_inputs(stub_root)
    _write_harness(stub_root)
    env = _nextflow_env()
    first = _run(stub_root, env, "-stub-run")
    assert first.returncode == 0, first.stdout + first.stderr
    runs: dict[str, Any] = {"stub": _collect(stub_root)}
    second = _run(stub_root, env, "-stub-run", "-resume")
    assert second.returncode == 0, second.stdout + second.stderr
    runs["resume"] = _collect(stub_root)
    real_root = base / "real"
    real_root.mkdir(parents=True)
    _write_inputs(real_root)
    _write_harness(real_root)
    third = _run(real_root, _fake_merxen_env(real_root))
    assert third.returncode == 0, third.stdout + third.stderr
    runs["real"] = _collect(real_root)
    runs["stub_root"] = str(stub_root)
    runs["real_root"] = str(real_root)
    return runs


@pytest.fixture(scope="module")
def harness(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Run the harness once per test session, also under pytest-xdist."""
    base = tmp_path_factory.getbasetemp()
    shared = base.parent if os.environ.get("PYTEST_XDIST_WORKER") else base
    with (shared / "annotation_report_harness.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        results_path = shared / "annotation_report_harness.json"
        if not results_path.exists():
            results = _run_harness(shared / "annotation_report_harness")
            results_path.write_text(json.dumps(results))
        return json.loads(results_path.read_text())


def _rows(trace: list[dict[str, str]], process: str) -> list[dict[str, str]]:
    return [row for row in trace if row["process"].split(":")[-1] == process]


def _report_rows(trace: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    rows = {}
    for row in _rows(trace, REPORT):
        pair_id, segmentation = row["tag"].split(":")
        rows[(pair_id, segmentation)] = row
    return rows


def _arguments(workdir: str, name: str) -> list[str]:
    record = json.loads((Path(workdir) / "annotation_report_out" / name).read_text())
    return shlex.split(record.get("arguments", ""))


def _option_values(arguments: list[str], option: str) -> list[str]:
    return [
        arguments[index + 1] for index, value in enumerate(arguments) if value == option
    ]


# --------------------------------------------------------------------------
# Channel logic (stub runs)


@needs_nextflow
def test_a_report_runs_once_per_finalized_branch(harness: dict[str, Any]) -> None:
    reports = _report_rows(harness["stub"]["trace"])
    assert set(reports) == {
        ("P1", "proseg_hybrid"),
        ("P1", "reseg"),
        ("P3", "original_seg"),
    }
    assert all(row["status"] == "COMPLETED" for row in reports.values())
    assert len(_rows(harness["stub"]["trace"], REPORT)) == 3
    emitted = {
        (row["pair_id"], row["segmentation"]) for row in harness["stub"]["reports"]
    }
    assert emitted == set(reports)


@needs_nextflow
def test_a_report_waits_for_its_depth_and_mender_outputs(
    harness: dict[str, Any],
) -> None:
    trace = harness["stub"]["trace"]
    reports = _report_rows(trace)
    depth = {row["tag"]: row for row in _rows(trace, "FAKE_DEPTH")}
    mender = {row["tag"]: row for row in _rows(trace, "FAKE_MENDER")}
    start = int(reports[("P1", "proseg_hybrid")]["start"])
    for tag in ("P1:MERSCOPE", "P1:XENIUM"):
        assert start >= int(depth[tag]["complete"]), tag
    for tag in ("P1:MERSCOPE:proseg_hybrid", "P1:XENIUM:proseg_hybrid"):
        assert start >= int(mender[tag]["complete"]), tag
    single = int(reports[("P3", "original_seg")]["start"])
    assert single >= int(depth["P3:MERSCOPE"]["complete"])
    assert single >= int(mender["P3:MERSCOPE:original_seg"]["complete"])
    # reseg waits for neither: it starts before P1's slow depth task ends.
    assert int(reports[("P1", "reseg")]["start"]) < int(depth["P1:XENIUM"]["complete"])


@needs_nextflow
def test_a_failed_depth_task_drops_only_its_branch(harness: dict[str, Any]) -> None:
    trace = harness["stub"]["trace"]
    depth = {row["tag"]: row for row in _rows(trace, "FAKE_DEPTH")}
    assert depth["P2:XENIUM"]["status"] == "FAILED"
    assert ("P2", "proseg_hybrid") not in _report_rows(trace)
    lines = {
        line.split(":", 1)[0].strip(): line.split(":", 1)[1].strip()
        for line in harness["stub"]["summary"].splitlines()[1:]
        if ":" in line
    }
    # P4 never reached FINALIZE: its failed hierarchy is listed elsewhere.
    assert (
        lines["annotation reports not built (see the failed tasks above)"]
        == "P2:proseg_hybrid"
    )
    assert lines["annotation report items failed (see the report)"] == "none"


@needs_nextflow
def test_a_report_stages_each_platforms_outputs_under_its_sample(
    harness: dict[str, Any],
) -> None:
    reports = _report_rows(harness["stub"]["trace"])
    workdir = Path(reports[("P1", "proseg_hybrid")]["workdir"])
    arguments = _arguments(str(workdir), "stub_report.json")
    assert arguments[:4] == ["--pair", "P1", "--segmentation", "proseg_hybrid"]
    assert _option_values(arguments, "--resolve-dir") == [
        "report_inputs/annotation_resolve_out"
    ]
    staged = workdir / "report_inputs"
    assert (
        staged / "annotation_resolve_out" / "P1_resolve_summary.json"
    ).read_text() == "P1:proseg_hybrid"
    for option, name in (
        ("--map-dir", "map_manifest.json"),
        ("--panel-dir", "panel_report.json"),
    ):
        (value,) = _option_values(arguments, option)
        assert (workdir / value / name).read_text() == "P1:proseg_hybrid"
    depth = dict(
        value.split("=", 1)
        for value in _option_values(arguments, "--cortical-depth-dir")
    )
    assert set(depth) == {"P1_MERSCOPE", "P1_XENIUM"}
    for sample_id, directory in depth.items():
        platform = sample_id.split("_", 1)[1]
        assert (workdir / directory / "platform.txt").read_text().strip() == platform
    mender = dict(
        value.split("=", 1) for value in _option_values(arguments, "--mender-manifest")
    )
    for sample_id, manifest in mender.items():
        platform = sample_id.split("_", 1)[1]
        assert json.loads((workdir / manifest).read_text()) == {"platform": platform}
    assert set(mender) == {"P1_MERSCOPE", "P1_XENIUM"}
    clustered = dict(
        value.split("=", 1) for value in _option_values(arguments, "--clustered-h5ad")
    )
    assert clustered == {
        "P1_MERSCOPE": "report_inputs/clustering_squidpy_out/merscope/"
        "P1_MERSCOPE_clustered.h5ad",
        "P1_XENIUM": "report_inputs/clustering_squidpy_out/xenium/"
        "P1_XENIUM_clustered.h5ad",
    }
    for path in clustered.values():
        assert (workdir / path).read_text() == "P1:proseg_hybrid"
    assert _option_values(arguments, "--alignment-dir") == ["report_inputs/align_out"]
    assert (staged / "align_out" / "shared_tissue_mask.npy").is_file()
    assert _option_values(arguments, "--out") == ["annotation_report_out"]
    for flag in ("--no-cortical-depth", "--no-mender", "--no-alignment", "--strict"):
        assert flag not in arguments, flag
    listing = (
        (workdir / "annotation_report_out" / "stub_inputs_listing.txt")
        .read_text()
        .split()
    )
    assert "./cortical_depth_1/compute_cortical_depth_out" in listing
    assert "./mender_2/xenium" in listing


@needs_nextflow
def test_a_branch_without_depth_mender_or_alignment_says_so(
    harness: dict[str, Any],
) -> None:
    reports = _report_rows(harness["stub"]["trace"])
    reseg = _arguments(reports[("P1", "reseg")]["workdir"], "stub_report.json")
    for flag in ("--no-cortical-depth", "--no-mender"):
        assert flag in reseg
    assert "--cortical-depth-dir" not in reseg and "--mender-manifest" not in reseg
    # P1 has an alignment: reseg's report gets it too.
    assert _option_values(reseg, "--alignment-dir") == ["report_inputs/align_out"]
    single = _arguments(reports[("P3", "original_seg")]["workdir"], "stub_report.json")
    assert "--no-alignment" in single
    assert _option_values(single, "--cortical-depth-dir") == [
        "P3_MERSCOPE=report_inputs/cortical_depth_1/compute_cortical_depth_out"
    ]
    assert _option_values(single, "--mender-manifest") == [
        "P3_MERSCOPE=report_inputs/mender_1/merscope/mender_manifest.json"
    ]


@needs_nextflow
def test_report_resources_and_publish_dir(harness: dict[str, Any]) -> None:
    reports = _report_rows(harness["stub"]["trace"])
    for row in reports.values():
        assert row["cpus"] == "4"
        assert int(row["memory"]) == 32 * 1024**3
    outdir = Path(harness["stub_root"]) / "results"
    for pair_id, segmentation in reports:
        published = (
            outdir
            / pair_id
            / segmentation
            / "annotation_report"
            / "annotation_report_out"
        )
        for name in ("report.html", "acceptance_metrics.json", "report_run.json"):
            assert (published / name).is_file(), (pair_id, segmentation, name)
        record = json.loads((published / "stub_report.json").read_text())
        assert record["pair_id"] == pair_id and record["species"] == "human"


@needs_nextflow
def test_resume_keeps_the_reports_cached(harness: dict[str, Any]) -> None:
    rows = _rows(harness["resume"]["trace"], REPORT)
    assert len(rows) == 3
    assert {row["status"] for row in rows} == {"CACHED"}


# --------------------------------------------------------------------------
# The real script (fake merxen)


@needs_nextflow
def test_the_real_script_calls_the_report_command_cpu_only(
    harness: dict[str, Any],
) -> None:
    stub = _report_rows(harness["stub"]["trace"])
    real = _report_rows(harness["real"]["trace"])
    assert set(real) == set(stub)
    for branch, row in real.items():
        call = json.loads(
            (
                Path(row["workdir"]) / "annotation_report_out" / "fake_call.json"
            ).read_text()
        )
        expected = _arguments(stub[branch]["workdir"], "stub_report.json")
        assert call["argv"] == ["annotation-report", *expected], branch
        env = call["env"]
        assert env["CUDA_VISIBLE_DEVICES"] == ""
        assert env["MPLBACKEND"] == "Agg"
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMBA_NUM_THREADS",
        ):
            assert env[name] == "4", name
        assert env["PYTHONPATH"].split(os.pathsep)[0].endswith("/../src")


@needs_nextflow
def test_a_failed_report_item_reaches_the_summary(harness: dict[str, Any]) -> None:
    lines = {
        line.split(":", 1)[0].strip(): line.split(":", 1)[1].strip()
        for line in harness["real"]["summary"].splitlines()[1:]
        if ":" in line
    }
    pair_id, segmentation = FAILED_ITEM_BRANCH
    assert lines["annotation report items failed (see the report)"] == (
        f"{pair_id}:{segmentation} ({FAILED_ITEM})"
    )


# --------------------------------------------------------------------------
# Groovy helpers


@needs_nextflow
def test_pair_and_branch_specs_follow_the_row_settings(
    harness: dict[str, Any],
) -> None:
    cases = harness["stub"]["cases"]
    both = cases["pairSpec|depth-and-distance"]["value"]
    assert both == {
        "enabled": True,
        "platforms": ["MERSCOPE", "XENIUM"],
        "depth_segmentations": ["proseg_hybrid", "reseg"],
        "mender_segmentations": ["proseg_hybrid"],
    }
    off = cases["pairSpec|no-clustering"]["value"]
    assert off["enabled"] is False
    assert off["depth_segmentations"] == [] and off["mender_segmentations"] == []
    assert cases["pairSpec|disabled"]["value"]["enabled"] is False
    assert cases["enabled|unset"]["value"] is True
    assert cases["enabled|false"]["value"] is False
    branch = cases["branchSpec|mender-from-samples"]["value"]
    # MENDER runs per sample of the branch; depth does not cover it.
    assert branch == {
        "enabled": True,
        "depth_platforms": [],
        "mender_platforms": ["XENIUM"],
    }
    assert cases["sortedByPlatform"]["value"] == [
        ["MERSCOPE", "XENIUM"],
        ["m_dir", "x_dir"],
    ]


@needs_nextflow
def test_report_arguments_quote_and_check_their_inputs(
    harness: dict[str, Any],
) -> None:
    cases = harness["stub"]["cases"]
    quoted = shlex.split(cases["reportArguments|quoted"]["value"])
    assert quoted[:2] == ["--pair", "P 9"]
    assert _option_values(quoted, "--clustered-h5ad") == [
        "P 9_MERSCOPE=report_inputs/clustering_squidpy_out/merscope/"
        "P 9_MERSCOPE_clustered.h5ad"
    ]
    assert (
        "2 platforms for 1 staged outputs" in cases["reportArguments|mismatch"]["error"]
    )


# --------------------------------------------------------------------------
# Wiring (string tests)


def test_the_process_is_cpu_only_and_publishes_per_branch() -> None:
    text = MODULE.read_text()
    start = text.index("process ANNOTATION_REPORT {")
    body = text[start:]
    assert (
        'publishDir { "${params.outdir}/${pair_id}/${segmentation}/annotation_report" }'
        in body
    )
    assert 'export CUDA_VISIBLE_DEVICES=""' in body
    assert "merxen annotation-report" in body
    assert "AnnotationReport.reportArguments(" in body
    for forbidden in ("conda", "container", "gpu", "queue", "cache false", "--strict"):
        assert forbidden not in body.split("script:")[0].lower(), forbidden
    assert "stub:" in body


def test_resources_follow_the_plan() -> None:
    config = ANNOTATION_CONFIG.read_text()
    block = config[config.index('withName: "ANNOTATION_REPORT" {') :]
    block = block[: block.index("}")]
    assert "cpus = 4" in block and 'memory = "32 GB"' in block
    dwight = DWIGHT_ANNOTATION_CONFIG.read_text()
    block = dwight[dwight.index('withName: "ANNOTATION_REPORT" {') :]
    assert "maxForks = params.annotation_report_max_forks" in block[: block.index("}")]
    assert "annotation_report_max_forks = 2" in dwight


def test_the_subworkflow_waits_for_finalize_depth_and_mender() -> None:
    text = SUBWORKFLOW.read_text()
    main = text[text.index("main:") :]
    assert main.count("groupKey(branchKey, platforms.size())") == 2
    assert ".join(\n            clustered_ch" in main
    assert ".combine(alignment_ch, by: 0)" in main
    assert ".join(depth_inputs_ch)\n        .join(mender_inputs_ch)" in main
    assert main.index(".join(mender_inputs_ch)") < main.index(
        "ANNOTATION_REPORT(report_inputs_ch)"
    )
    assert "AnnotationRunRecord.reportExpected(" in main
    assert "AnnotationRunRecord.reported(" in main


def test_map_first_emits_the_alignment_it_was_given() -> None:
    text = MAP_FIRST.read_text()
    block = text[text.index("workflow CLUSTERING_MAP_FIRST {") :]
    emit = block[block.index("emit:") :]
    assert "alignment = alignment_ch" in emit


def test_main_nf_reports_only_through_hook_h11() -> None:
    text = MAIN_NF.read_text()
    assert text.count("ANNOTATION_REPORTING(") == 1
    hook = text.index("rca-hook:H11")
    call = text.index("ANNOTATION_REPORTING(")
    assert hook < call
    window = text[call : text.index("\n    }\n", call)]
    for expected in (
        "CLUSTERING_MAP_FIRST.out.labels,",
        "CLUSTERING_MAP_FIRST.out.alignment,",
        "clustering_results_ch,",
        "compute_cortical_depth_results_ch.map {",
        "mender_finalized_ch.map {",
        "AnnotationReport.pairSpec(settings, params)",
    ):
        assert expected in window, expected

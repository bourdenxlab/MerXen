"""Mixed frontal/hippocampal workflow routing and restart contracts."""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / "workflows"


@pytest.mark.nextflow
@pytest.mark.skipif(shutil.which("nextflow") is None, reason="Nextflow unavailable")
def test_mixed_profiles_reach_both_reports_and_resume(tmp_path: Path) -> None:
    (tmp_path / "lib").mkdir()
    for source in (WORKFLOWS / "lib").glob("*.groovy"):
        shutil.copy(source, tmp_path / "lib" / source.name)
    items = {}
    for platform, n_genes in (("XENIUM", 500), ("ATERA", 18049)):
        sid = f"SYNTHETIC_{platform}"
        prepared = tmp_path / platform / "prepared"
        target = prepared / platform.lower() / f"{sid}_prepared.h5ad"
        target.parent.mkdir(parents=True)
        genes = [f"ENSG{i:011d}" for i in range(n_genes)]
        data = ad.AnnData(
            sparse.csr_matrix(np.ones((10, n_genes), dtype=np.float32)),
            var=pd.DataFrame(
                {"ensembl_id": genes}, index=[f"G{i}" for i in range(n_genes)]
            ),
        )
        data.obsm["spatial"] = np.ones((10, 2))
        data.write_h5ad(target)
        (prepared / "manifest.json").write_text(
            json.dumps({"samples": {sid: f"{platform.lower()}/{target.name}"}})
        )
        samples = [
            {"sample_id": sid, "platform": platform, "segmentation": "original_seg"}
        ]
        config = tmp_path / platform / "config.json"
        config.write_text(
            json.dumps({"pair_id": platform, "samples": samples, "min_counts": 10})
        )
        items[platform] = dict(
            config=str(config), prepared=str(prepared), samples=json.dumps(samples)
        )
    (tmp_path / "inputs.json").write_text(json.dumps(items))
    (tmp_path / "resources.json").write_text("{}")
    harness = (
        """
include { CLUSTERING_MAP_FIRST } from '__OLD__'
include { HIPPOCAMPAL_ANNOTATION; HIPPOCAMPAL_REPORT } from '__NEW__'
include { ANNOTATION_REPORT } from '__REPORT__'
workflow {
    def input = new groovy.json.JsonSlurperClassic().parseText(file(params.inputs).text)
    def x = input.XENIUM
    def a = input.ATERA
    old = CLUSTERING_MAP_FIRST(
        channel.of(tuple('XENIUM', 'original_seg', x.samples,
            file(x.config), file(x.prepared))),
        channel.of(tuple('XENIUM', [])),
        channel.of(tuple('XENIUM', 'original_seg', [])))
    hpc = HIPPOCAMPAL_ANNOTATION(channel.of(tuple('ATERA', 'original_seg',
        a.samples, file(a.config), file(a.prepared))))
    old_reports = old.labels.map {
        pair, seg, samples, config, prepared, panel, refs, mapped, resolved ->
        tuple(pair, seg, [species:'human', report_fingerprint:'test'], samples,
            resolved, mapped, panel, prepared, [], [], [], [], [])
    }
    ANNOTATION_REPORT(old_reports)
    HIPPOCAMPAL_REPORT(hpc.labels.combine(hpc.computed, by: [0,1]).map {
        pair, seg, resolved, samples, computed ->
        tuple(pair, seg, resolved, computed, [], false, 'test')
    })
}
""".replace("__OLD__", str(WORKFLOWS / "subworkflows/clustering_map_first"))
        .replace("__NEW__", str(WORKFLOWS / "subworkflows/hippocampal_annotation"))
        .replace("__REPORT__", str(WORKFLOWS / "modules/annotation"))
    )
    (tmp_path / "main.nf").write_text(harness)
    (tmp_path / "nextflow.config").write_text(f"""
includeConfig '{WORKFLOWS / "conf/annotation.config"}'
params {{
 species='human'
 outdir='{tmp_path / "results"}'
 inputs='{tmp_path / "inputs.json"}'
 annotation_hippocampal_resources='{tmp_path / "resources.json"}'
 annotation_reference_store='{tmp_path / "store"}'
}}
process {{ executor='local'; errorStrategy='terminate'; maxForks=2 }}
executor {{ cpus=32; memory='512 GB' }}
trace {{ enabled=true; overwrite=true; file='trace.tsv'; fields='process,status' }}
""")
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(ROOT / "src"),
        NXF_DISABLE_CHECK_LATEST="true",
        MERXEN_SITE_CONFIG="",
        MERXEN_DWIGHT_SITE_CONFIG="",
    )
    env["PATH"] = str(Path(os.sys.executable).parent) + os.pathsep + env["PATH"]
    for resume in (False, True):
        result = subprocess.run(
            [
                "nextflow",
                "run",
                "main.nf",
                "-stub-run",
                "-ansi-log",
                "false",
                *(["-resume"] if resume else []),
            ],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
        with (tmp_path / "trace.tsv").open() as handle:
            trace = list(csv.DictReader(handle, delimiter="\t"))
        names = [row["process"].split(":")[-1] for row in trace]
        assert "HIPPOCAMPAL_REPORT" in names and "ANNOTATION_REPORT" in names
        assert "HIPPOCAMPAL_MAP" in names and "CLUSTERING_SQUIDPY_ANNOTATE_MAP" in names
        if resume:
            mapping = [
                row
                for row in trace
                if row["process"].endswith(":HIPPOCAMPAL_MAP")
                or row["process"] == "HIPPOCAMPAL_MAP"
            ]
            assert len(mapping) == 1 and mapping[0]["status"] == "CACHED"


@pytest.mark.nextflow
@pytest.mark.skipif(shutil.which("nextflow") is None, reason="Nextflow unavailable")
def test_policy_identity_change_reruns_resolve_and_keeps_mapping(
    tmp_path: Path,
) -> None:
    (tmp_path / "lib").mkdir()
    for source in (WORKFLOWS / "lib").glob("*.groovy"):
        shutil.copy(source, tmp_path / "lib" / source.name)
    for name, content in {
        "resources.json": "{}",
        "mapping.json": '{"reference":"fixed"}',
        "policy.json": '{"qualification":"first"}',
        "config.json": "{}",
        "manifest.json": '{"samples":{"S":"atera/S_prepared.h5ad"}}',
        "S_prepared.h5ad": "synthetic staged counts",
    }.items():
        (tmp_path / name).write_text(content)
    (tmp_path / "main.nf").write_text(
        """
include { HIPPOCAMPAL_MAP; HIPPOCAMPAL_RESOLVE } from '__MODULE__'
workflow {
    mapped = HIPPOCAMPAL_MAP(channel.of(tuple('PAIR', 'original_seg', '[]',
        file('config.json'), file('manifest.json'), [file('S_prepared.h5ad')],
        file('resources.json'), file('mapping.json'), 'fixed-code')))
    HIPPOCAMPAL_RESOLVE(mapped.map {
        pair, segmentation, samples, config, manifest, counts, mappedDir ->
        tuple(pair, segmentation, samples, config, manifest, counts, mappedDir,
            file('resources.json'), file('policy.json'), 'fixed-code')
    })
}
""".replace("__MODULE__", str(WORKFLOWS / "subworkflows/hippocampal_annotation"))
    )
    (tmp_path / "nextflow.config").write_text("""
params.outdir='results'
executor { cpus=16; memory='256 GB' }
process { executor='local'; errorStrategy='terminate' }
trace { enabled=true; overwrite=true; file='trace.tsv'; fields='process,status' }
""")
    env = os.environ.copy()
    env["NXF_DISABLE_CHECK_LATEST"] = "true"
    for changed in (False, True):
        if changed:
            (tmp_path / "policy.json").write_text('{"qualification":"second"}')
        result = subprocess.run(
            [
                "nextflow",
                "run",
                "main.nf",
                "-stub-run",
                "-ansi-log",
                "false",
                *(["-resume"] if changed else []),
            ],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
        if changed:
            with (tmp_path / "trace.tsv").open() as handle:
                trace = {
                    row["process"]: row["status"]
                    for row in csv.DictReader(handle, delimiter="\t")
                }
            assert trace["HIPPOCAMPAL_MAP"] == "CACHED"
            assert trace["HIPPOCAMPAL_RESOLVE"] == "COMPLETED"

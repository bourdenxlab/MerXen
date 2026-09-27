"""The annotation processes and their wiring (plan §3.1-§3.3, §13.2).

String tests pin the process contract: no ``storeDir`` (the Python
``ReferenceStore`` owns the store), CPU-only resources without a GPU lock,
PREP sized by panel size, one PREP and two MAPs at a time on dwight, MAP's
ctm check, content cache and publish path, and ``main.nf`` calling the
annotation workflows only behind hook H10. The MAP channel logic is tested
in ``test_map_first_subworkflow.py``.

When ``nextflow`` is installed, small runs execute the real workflow code
(``-stub-run``: PREP writes a stub ``bundle_ref.json``; ANNOTATE_PANEL has no
stub and runs its real, seconds-long command):

* a harness runs ``ANNOTATION_BUNDLES`` on hand-written
  ``required_bundles.json`` fixtures (same-panel set c, ``per_platform``,
  mouse, refused, a failing PREP, a large panel) and
  ``ANNOTATION_PREPARED_REFERENCES`` on synthetic prepared H5ADs, and calls
  the ``AnnotationReferences`` / ``AnnotationPreflight`` helpers;
* ``main.nf --annotation_prepare_only`` on a two-row synthetic samplesheet,
  with ``-stub-run`` and, for a mouse section, without it but with a fake
  ``merxen annotation-reference-prep`` that records the rendered command.

They check that PREP runs once per unique (species, reference, panel), that
each pair x segmentation is released with exactly its required bundles
(groupKey size = n_required), and that nothing else runs.
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

from merxen.annotation import panel as panel_module
from merxen.annotation import pipeline as pipeline_module
from merxen.annotation import reference as reference_module
from merxen.annotation.config import (
    DEFAULT_REFERENCE_IDS,
    KNOWN_REFERENCES,
    LARGE_PANEL_GENES,
    AnnotationConfig,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / "workflows"
MODULE = WORKFLOWS / "modules" / "annotation.nf"
SUBWORKFLOW = WORKFLOWS / "subworkflows" / "annotation_references.nf"
MAP_FIRST = WORKFLOWS / "subworkflows" / "clustering_map_first.nf"
MAIN_NF = WORKFLOWS / "main.nf"
ANNOTATION_CONFIG = WORKFLOWS / "conf" / "annotation.config"
DWIGHT_ANNOTATION_CONFIG = WORKFLOWS / "conf" / "dwight.annotation.config"
DWIGHT_CONFIG = WORKFLOWS / "conf" / "dwight.config"
LIB_DIR = WORKFLOWS / "lib"
REFERENCES_SOURCE = (LIB_DIR / "AnnotationReferences.groovy").read_text()
NEXTFLOW = shutil.which("nextflow")
needs_nextflow = pytest.mark.skipif(NEXTFLOW is None, reason="nextflow is unavailable")
PROCESSES = (
    "ANNOTATE_PANEL",
    "ANNOTATE_REFERENCE_PREP",
    "CLUSTERING_SQUIDPY_ANNOTATE_MAP",
)
PREP = "ANNOTATE_REFERENCE_PREP"
MAP = "CLUSTERING_SQUIDPY_ANNOTATE_MAP"


def _process_block(text: str, name: str) -> str:
    start = text.index(f"process {name} {{")
    following = [m.start() for m in re.finditer(r"^process ", text, re.M)]
    end = min((s for s in following if s > start), default=len(text))
    return text[start:end]


def _with_name_block(config_text: str, name: str) -> str:
    start = config_text.index(f'withName: "{name}" {{')
    depth, index = 0, config_text.index("{", start)
    while True:
        depth += {"{": 1, "}": -1}.get(config_text[index], 0)
        index += 1
        if depth == 0:
            return config_text[start:index]


# --------------------------------------------------------------------------
# Process contract (string tests)


def test_annotation_module_defines_panel_prep_and_map() -> None:
    text = MODULE.read_text()
    assert re.findall(r"^process (\w+) \{", text, re.M) == list(PROCESSES)
    assert "merxen annotation-panel" in _process_block(text, "ANNOTATE_PANEL")
    prep = _process_block(text, PREP)
    assert "merxen annotation-reference-prep" in prep
    assert "stub:" in prep
    # ANNOTATE_PANEL is seconds long and task-local: -stub-run runs it for real.
    assert "stub:" not in _process_block(text, "ANNOTATE_PANEL")
    map_block = _process_block(text, MAP)
    assert "merxen annotate \\\\" in map_block
    assert "stub:" in map_block


def test_no_store_dir_anywhere_in_the_annotation_workflows() -> None:
    """The ReferenceStore owns the store; a storeDir would bypass its lock."""
    for path in (
        MODULE,
        SUBWORKFLOW,
        MAP_FIRST,
        ANNOTATION_CONFIG,
        DWIGHT_ANNOTATION_CONFIG,
    ):
        code = re.sub(r"//[^\n]*|/\*.*?\*/", "", path.read_text(), flags=re.S)
        assert "storeDir" not in code, path


def test_prep_caches_deeply_and_publishes_its_bundle_ref() -> None:
    prep = _process_block(MODULE.read_text(), PREP)
    assert 'cache "deep"' in prep
    assert (
        'publishDir { "${params.outdir}/annotation_reference_prep/'
        '${bundle.reference_id}/${bundle.panel_tag}" }'
    ) in prep
    assert 'path("bundle_ref.json")' in prep
    assert 'path(panel_genes, arity: "0..*", stageAs: "panel_genes/*")' in prep
    panel = _process_block(MODULE.read_text(), "ANNOTATE_PANEL")
    assert '"${params.outdir}/${pair_id}/${segmentation}/annotation_panel"' in panel


@pytest.mark.parametrize("name", PROCESSES)
def test_annotation_processes_run_on_the_cpu_with_current_code(name: str) -> None:
    block = _process_block(MODULE.read_text(), name)
    assert 'export CUDA_VISIBLE_DEVICES=""' in block
    assert 'export PYTHONPATH="${projectDir}/../src:\\${PYTHONPATH:-}"' in block
    for variable in ("OMP", "OPENBLAS", "MKL", "NUMBA"):
        assert f"export {variable}_NUM_THREADS=1" in block
    code = re.sub(r"^\s*(//|#)[^\n]*", "", block, flags=re.M)
    for token in ("flock", "MERXEN_GPU_LOCK_FILE", "gpu", "queue", "clusterOptions"):
        assert token not in code.replace("CUDA_VISIBLE_DEVICES", ""), token


@pytest.mark.parametrize("name", [PREP, MAP])
def test_prep_and_map_check_the_cell_type_mapper_version(name: str) -> None:
    block = _process_block(MODULE.read_text(), name)
    script = block[block.index("script:") : block.index("stub:")]
    assert 'importlib.metadata.version("cell_type_mapper")' in script
    assert 'expected = "${params.annotation_ctm_version}"' in script
    # The check runs before the command.
    assert script.index("cell_type_mapper") < script.index("merxen annotat")


def test_map_caches_on_content_and_publishes_under_the_segmentation() -> None:
    """PREP re-writes identical refs in new work dirs: MAP must hash content."""
    block = _process_block(MODULE.read_text(), MAP)
    assert 'cache "deep"' in block
    assert (
        'publishDir { "${params.outdir}/${pair_id}/${segmentation}/annotation_map" }, '
        'mode: "copy", overwrite: true'
    ) in block
    assert 'path("annotation_map_out")' in block
    assert (
        'path(bundle_refs, arity: "0..*", '
        'stageAs: "map_inputs/bundle_refs/bundle_ref_?.json")'
    ) in block
    for name, staged in (
        ("clustering_config", "clustering_squidpy_config.json"),
        ("prepared_dir", "clustering_prepare_out"),
        ("panel_dir", "annotation_panel_out"),
    ):
        assert f'path({name}, stageAs: "map_inputs/{staged}")' in block
    assert "task.cpus as int" in block
    assert "--out annotation_map_out" in block
    assert "rm -rf ${AnnotationReferences.MAP_SCRATCH_DIR}" in block


def test_map_resources_follow_the_plan() -> None:
    """MAP: 6 cpus, 24 GB, 48 GB above 1,000 panel genes (plan §3.3)."""
    block = _with_name_block(ANNOTATION_CONFIG.read_text(), MAP)
    assert re.search(r"cpus = 6\n", block)
    assert (
        f"memory = {{ (map_spec.n_panel_genes ?: 0) > {LARGE_PANEL_GENES} "
        '? "48 GB" : "24 GB" }'
    ) in block
    assert "maxForks" not in block


def test_dwight_runs_two_maps_at_a_time() -> None:
    text = DWIGHT_ANNOTATION_CONFIG.read_text()
    block = _with_name_block(text, MAP)
    assert "maxForks = params.annotation_max_forks" in block
    assert re.search(r"annotation_max_forks = 2\b", text)
    assert re.search(r"annotation_max_forks = 2\b", ANNOTATION_CONFIG.read_text())


def test_no_gpu_lock_or_gpu_queue_on_the_annotation_processes() -> None:
    dwight = DWIGHT_CONFIG.read_text()
    locked = re.findall(
        r'withName: "(\w+)" \{\s*maxForks = [^\n]+\n\s*beforeScript', dwight
    )
    assert set(locked) == {
        "CELLPOSE_SEGMENT",
        "CELLPOSE_NUCLEI_SEGMENT",
        "ALIGN",
        "CLUSTERING_SQUIDPY_COMPUTE",
    }
    for path in (ANNOTATION_CONFIG, DWIGHT_ANNOTATION_CONFIG):
        text = path.read_text()
        for token in (
            "beforeScript",
            "MERXEN_GPU_LOCK_FILE",
            "queue",
            "clusterOptions",
        ):
            assert token not in text, (path.name, token)


def test_resources_follow_the_plan() -> None:
    """PANEL 1 cpu / 4 GB; PREP 8 cpus / 64 GB / 8 h, large panels 24 h (§3.1)."""
    text = ANNOTATION_CONFIG.read_text()
    panel = _with_name_block(text, "ANNOTATE_PANEL")
    assert re.search(r"cpus = 1\n", panel) and 'memory = "4 GB"' in panel
    prep = _with_name_block(text, PREP)
    assert "cpus = 8" in prep
    assert '"64 GB"' in prep and "params.annotation_prep_large_memory" in prep
    assert '"24h" : "8h"' in prep
    assert prep.count(f"(bundle.n_panel_genes ?: 0) > {LARGE_PANEL_GENES}") == 2


def test_dwight_runs_one_prep_at_a_time() -> None:
    text = DWIGHT_ANNOTATION_CONFIG.read_text()
    block = _with_name_block(text, PREP)
    assert "maxForks = params.annotation_prep_max_forks" in block
    assert re.search(r"annotation_prep_max_forks = 1\b", text)
    assert (
        'annotation_reference_store = "/media/mathieubo/SSD1/MerXen/'
        'annotation_references"' in text
    )


def test_main_nf_calls_only_the_prepare_only_entry() -> None:
    main_text = MAIN_NF.read_text()
    assert main_text.count("ANNOTATION_PREPARE_ONLY(sample_rows_raw_ch)") == 1
    for name in (
        "ANNOTATE_PANEL(",
        "ANNOTATE_REFERENCE_PREP(",
        "ANNOTATION_REFERENCES(",
        "ANNOTATION_PREPARED_REFERENCES(",
        "CLUSTERING_SQUIDPY_ANNOTATE_MAP(",
        "CLUSTERING_ANNOTATE_MAP(",
        "CLUSTERING_MAP_FIRST(",
    ):
        assert name not in main_text
    # CLUSTERING_MAP_FIRST still refuses to run until M5.
    assert "error(" in MAP_FIRST.read_text()


def test_bundle_collection_waits_for_each_branch_own_count() -> None:
    """MAP's wait uses groupKey(branch, n_required), never a global group."""
    text = SUBWORKFLOW.read_text()
    assert ".unique { request -> request[0].key }" in text
    assert "groupKey(branchKey, nRequired)" in text
    assert ".combine(bundle_refs_ch, by: 0)" in text
    assert text.count(".groupTuple()") == 1


# --------------------------------------------------------------------------
# Groovy constants in step with Python


def _groovy_map_literal(name: str) -> str:
    match = re.search(rf"static final [\w<>, ]+ {name} = \[", REFERENCES_SOURCE)
    assert match is not None, name
    depth, index = 0, match.end() - 1
    while True:
        depth += {"[": 1, "]": -1}.get(REFERENCES_SOURCE[index], 0)
        index += 1
        if depth == 0:
            return REFERENCES_SOURCE[match.end() - 1 : index]


def _source_params() -> dict[str, dict[str, str]]:
    body = _groovy_map_literal("SOURCE_PARAMS")
    result: dict[str, dict[str, str]] = {}
    for reference_id, inner in re.findall(
        r"(\w+): \[(.*?)\]\.asImmutable\(\)", body, re.S
    ):
        result[reference_id] = dict(re.findall(r'(\w+): "(\w+)"', inner))
    return result


def test_source_params_name_builder_sources_and_real_params(
    workflow_config_params: dict[str, dict[str, str]],
) -> None:
    sources = _source_params()
    builder_names = {
        value
        for name, value in vars(reference_module).items()
        if name.startswith("SOURCE_") and isinstance(value, str)
    }
    annotation_params = set(workflow_config_params["conf/annotation.config"])
    assert set(sources) <= set(KNOWN_REFERENCES)
    for species, defaults in DEFAULT_REFERENCE_IDS.items():
        assert set(defaults) <= set(sources), species
    for reference_id, mapping in sources.items():
        assert set(mapping) <= builder_names, reference_id
        assert set(mapping.values()) <= annotation_params, reference_id


def test_groovy_file_names_match_python() -> None:
    def constant(name: str) -> str:
        match = re.search(rf'static final String {name} = "([^"]+)"', REFERENCES_SOURCE)
        assert match is not None, name
        return match.group(1)

    assert constant("REQUIRED_BUNDLES_FILE") == panel_module.REQUIRED_BUNDLES_FILE
    assert constant("PANEL_INDEPENDENT") == "panel_independent"
    assert constant("GENE_LIST_SOURCE") == "gene_list"
    assert constant("MAP_MANIFEST_FILE") == pipeline_module.MAP_MANIFEST_NAME


# --------------------------------------------------------------------------
# Synthetic inputs


def _human_ids(n: int, offset: int = 0) -> list[str]:
    return [f"ENSG{i:011d}" for i in range(offset, offset + n)]


def _write_prepared_h5ad(
    path: Path,
    *,
    symbols: list[str],
    platform: str,
    ensembl_ids: list[str] | None,
    counts: np.ndarray,
    shape_key: str,
) -> None:
    var = pd.DataFrame(index=pd.Index(symbols, dtype=str))
    var["gene"] = symbols
    if ensembl_ids is not None:
        var["ensembl_id"] = ensembl_ids
    adata = ad.AnnData(
        X=sparse.csr_matrix(counts.astype(np.float32)),
        obs=pd.DataFrame(index=[f"{platform}_{i}" for i in range(counts.shape[0])]),
        var=var,
    )
    adata.obsm["spatial"] = np.full((counts.shape[0], 2), 2.0)
    adata.uns["merxen_clustering_squidpy"] = {
        "platform": platform,
        "shape_key": shape_key,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(path)


def _prepared_pair(
    root: Path,
    pair_id: str,
    segmentation: str,
    *,
    merscope_extra: int = 0,
    xenium_extra: int = 0,
) -> tuple[Path, Path, str]:
    """Write a prepared directory and clustering config of one pair.

    Both platforms declare 60 shared genes; Xenium plants a 10x factor on the
    first one (so set c differs from set a); platform-only genes
    (``merscope_extra``, ``xenium_extra``) make the pair ``per_platform``.
    """
    base = root / f"{pair_id}_{segmentation}"
    prepared = base / "clustering_prepare_out"
    shared_ids = _human_ids(60)
    shared_symbols = [f"GENE{i}" for i in range(60)]
    extra_ids = _human_ids(merscope_extra, offset=5000)
    merscope_symbols = shared_symbols + [f"MEXTRA{i}" for i in range(merscope_extra)]
    merscope_symbols += ["Blank-1", "Blank-2"]
    merscope_ids = (shared_ids + extra_ids + ["", ""]) if merscope_extra else None
    _write_prepared_h5ad(
        prepared / "merscope" / f"{pair_id}_M_prepared.h5ad",
        symbols=merscope_symbols,
        platform="MERSCOPE",
        ensembl_ids=merscope_ids,
        counts=np.ones((80, len(merscope_symbols))),
        shape_key=f"MOSAIK_{segmentation}_aligned_nonrigid",
    )
    xenium_symbols = shared_symbols + [f"XEXTRA{i}" for i in range(xenium_extra)]
    xenium_counts = np.ones((80, len(xenium_symbols)))
    xenium_counts[:, 0] = 10.0
    _write_prepared_h5ad(
        prepared / "xenium" / f"{pair_id}_X_prepared.h5ad",
        symbols=xenium_symbols,
        platform="XENIUM",
        ensembl_ids=shared_ids + _human_ids(xenium_extra, offset=7000),
        counts=xenium_counts,
        shape_key=f"MOSAIK_{segmentation}",
    )
    (prepared / "manifest.json").write_text(
        json.dumps(
            {
                "samples": {
                    f"{pair_id}_M": f"merscope/{pair_id}_M_prepared.h5ad",
                    f"{pair_id}_X": f"xenium/{pair_id}_X_prepared.h5ad",
                }
            }
        )
    )
    config = base / "clustering_squidpy_config.json"
    samples = [
        {"sample_id": f"{pair_id}_{p[0]}", "platform": p, "segmentation": segmentation}
        for p in ("MERSCOPE", "XENIUM")
    ]
    config.write_text(
        json.dumps({"pair_id": pair_id, "min_counts": 10, "samples": samples})
    )
    return prepared, config, json.dumps(samples)


def _publish_shared_tissue_mask(outdir: Path, pair_id: str) -> None:
    """A published align_out mask whose left half (all cells) is tissue.

    ANNOTATION_PREPARED_REFERENCES must ignore it: a published mask looked up
    at channel time races with ALIGN and may be stale (M5 wires the ALIGN
    output channel instead).
    """
    align_out = outdir / pair_id / "alignment" / "align_out"
    align_out.mkdir(parents=True)
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[:, :10] = 1
    np.save(align_out / "shared_tissue_mask.npy", mask)
    (align_out / "registration_summary.json").write_text(
        json.dumps(
            {
                "coordinate_frames": {
                    "fixed_platform": "XENIUM",
                    "fixed_dataset_to_image_matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
                }
            }
        )
    )


def _hash(tag: str) -> str:
    return (tag * 64)[:64]


# Hand-written required bundles per fixture branch: (reference, species,
# role, panel tag or None, n_panel_genes).
FIXTURE_BRANCHES: dict[tuple[str, str], list[tuple[str, str, str, str | None, int]]] = {
    # Same-panel human pair on proseg_hybrid: set a (WHB, SEA-AD) + set c.
    ("F1", "proseg_hybrid"): [
        ("whb_frontal_supc_clus", "human", "primary", "a", 296),
        ("whb_frontal_supc_clus", "human", "primary", "c", 264),
        ("seaad_mr_panel", "human", "secondary", "a", 296),
    ],
    # The same pair on reseg: no set-c run (annotation_xplat_sensitivity_segmentations).
    ("F1", "reseg"): [
        ("whb_frontal_supc_clus", "human", "primary", "a", 296),
        ("seaad_mr_panel", "human", "secondary", "a", 296),
    ],
    # Another pair with the same set a and its own set c.
    ("F2", "proseg_hybrid"): [
        ("whb_frontal_supc_clus", "human", "primary", "a", 296),
        ("whb_frontal_supc_clus", "human", "primary", "d", 270),
        ("seaad_mr_panel", "human", "secondary", "a", 296),
    ],
    # per_platform: both platform panels x {WHB, SEA-AD} + WHB on the intersection.
    ("F3", "proseg_hybrid"): [
        ("whb_frontal_supc_clus", "human", "primary", "m", 500),
        ("whb_frontal_supc_clus", "human", "primary", "x", 480),
        ("seaad_mr_panel", "human", "secondary", "m", 500),
        ("seaad_mr_panel", "human", "secondary", "x", 480),
        ("whb_frontal_supc_clus", "human", "primary", "i", 300),
    ],
    # Refused panels: nothing to build, released at once.
    ("F4", "proseg_hybrid"): [],
    # Mouse sections sharing one panel; region shares need no panel.
    ("F5", "original_seg"): [
        ("wmb_panel", "mouse", "primary", "g", 500),
        ("wmb_region_share", "mouse", "region_share", None, 0),
    ],
    ("F6", "original_seg"): [
        ("wmb_panel", "mouse", "primary", "g", 500),
        ("wmb_region_share", "mouse", "region_share", None, 0),
    ],
    # One of its bundles fails in PREP: the branch is dropped, not stalled.
    ("F7", "proseg_hybrid"): [
        ("whb_frontal_supc_clus", "human", "primary", "a", 296),
        ("failing_reference", "human", "secondary", "b", 296),
    ],
    # A large panel: PREP gets annotation_prep_large_memory and 24 h.
    ("F8", "proseg_hybrid"): [
        ("whb_frontal_supc_clus", "human", "primary", "l", 5000),
    ],
}
LARGE_MEMORY = "96 GB"


def _write_fixture_panels(root: Path) -> list[dict[str, str]]:
    items = []
    for (pair_id, segmentation), bundles in FIXTURE_BRANCHES.items():
        panel_dir = root / "fixture_panels" / f"{pair_id}_{segmentation}"
        panel_dir.mkdir(parents=True)
        entries = []
        for reference_id, species, role, tag, n_genes in bundles:
            panel_file = None
            if tag is not None:
                panel_file = f"panel_genes_{tag}.json"
                # Pairs sharing a panel hash write different panel files
                # (sample ids, per-platform symbols such as H2AX / H2AFX).
                (panel_dir / panel_file).write_text(
                    json.dumps(
                        {
                            "panel_hash": _hash(tag),
                            "sample_ids": [f"{pair_id}_MERSCOPE"],
                            "symbols": [f"SYMBOL_{pair_id}"],
                        }
                    )
                )
            entries.append(
                {
                    "reference_id": reference_id,
                    "role": role,
                    "species": species,
                    "purpose": "annotation" if tag else "panel_independent",
                    "panel_name": tag,
                    "panel_hash": _hash(tag) if tag else None,
                    "panel_file": panel_file,
                    "n_panel_genes": n_genes if tag else None,
                }
            )
        (panel_dir / "required_bundles.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "pair_id": pair_id,
                    "segmentation": segmentation,
                    "species": bundles[0][1] if bundles else "human",
                    "panel_mode": "intersection",
                    "status": "ok" if bundles else "refused",
                    "reasons": [],
                    "bundles": entries,
                    "n_required": len(entries),
                }
            )
        )
        items.append(
            {
                "pair_id": pair_id,
                "segmentation": segmentation,
                "panel_dir": str(panel_dir),
            }
        )
    return items


# --------------------------------------------------------------------------
# Nextflow harness

HARNESS_CLASS = """
import groovy.json.JsonOutput
import groovy.json.JsonSlurperClassic

class AnnotationModuleHarness {
    static void run(Object casesPath, Object outPath) {
        def source = new File(casesPath.toString())
        Map cases = new JsonSlurperClassic().parse(source) as Map
        Map results = [:]
        cases.each { name, testCase -> results[name] = evaluate(testCase as Map) }
        new File(outPath.toString()).text = JsonOutput.toJson(results)
    }

    static Map evaluate(Map testCase) {
        try {
            return [value: call(testCase)]
        } catch (Exception error) {
            return [error: error.message, type: error.class.simpleName]
        }
    }

    static Object call(Map c) {
        switch (c.fn) {
            case "annotationConfig":
                return AnnotationReferences.annotationConfig(c.params, c.species)
            case "panelArguments":
                return AnnotationReferences.panelArguments(
                    c.spec, c.pair_id, c.segmentation
                )
            case "preparedPanelSpec":
                return AnnotationReferences.preparedPanelSpec(
                    c.config, c.prepared, c.masks
                )
            case "prepArguments":
                return AnnotationReferences.prepArguments(
                    c.params, c.bundle, c.panel_genes,
                    c.cpus as int, c.memory_gb as long
                )
            case "referenceStore":
                return AnnotationReferences.referenceStore(c.params)
            case "prepareOnlyErrors":
                return AnnotationPreflight.prepareOnlyErrors(c.params)
            case "orderedBundleRefs":
                return AnnotationReferences.orderedBundleRefs(
                    c.required, c.keys, c.refs
                )
            case "requiredBundles":
                return AnnotationReferences.requiredBundles(c.panel_dir)
            case "prepareOnly":
                return AnnotationReferences.prepareOnly(c.params)
            case "prepRequests":
                def required = AnnotationReferences.requiredBundles(c.panel_dir)
                return AnnotationReferences.prepRequests(c.panel_dir, required)
                    .collect { request ->
                        [bundle: request[0], files: request[1]*.toString()]
                    }
            case "mapSpec":
                return AnnotationReferences.mapSpec(c.panel_dir)
            case "mapArguments":
                return AnnotationReferences.mapArguments(
                    c.pair_id, c.segmentation, c.spec, c.refs, c.cpus as int
                )
            case "mapReuseDir":
                return AnnotationReferences.mapReuseDir(
                    c.params, c.pair_id, c.segmentation
                )
        }
        throw new IllegalStateException("unknown case function ${c.fn}")
    }

    static List readList(Object path) {
        return new JsonSlurperClassic().parse(new File(path.toString())) as List
    }

    static void write(Object path, Object rows) {
        new File(path.toString()).text = JsonOutput.toJson(rows)
    }

    static Map bundleRow(
        Object pairId, Object segmentation, Object panelDir, List refs
    ) {
        return [
            pair_id: pairId,
            segmentation: segmentation,
            panel_dir: panelDir.toString(),
            refs: refs.collect { ref -> ref.toString() },
        ]
    }

    static Map mapInputRow(List item) {
        def (pairId, segmentation, samplesJson) = item[0..2]
        def (config, preparedDir, panelDir, refs) = item[3..6]
        return bundleRow(pairId, segmentation, panelDir, refs as List) + [
            samples_json: samplesJson,
            config: config.toString(),
            prepared_dir: preparedDir.toString(),
        ]
    }
}
"""

HARNESS_MAIN = """
include {
    ANNOTATION_BUNDLES as FIXTURE_BUNDLES;
    ANNOTATION_PREPARED_REFERENCES
} from '__SUBWORKFLOW__'

workflow {
    AnnotationModuleHarness.run(params.cases, params.function_results)

    fixture_ch = channel
        .fromList(AnnotationModuleHarness.readList(params.fixture_panels))
        .map { item -> tuple(item.pair_id, item.segmentation, file(item.panel_dir)) }
    fixture = FIXTURE_BUNDLES(fixture_ch)
    fixture.bundles
        .map { pairId, segmentation, panelDir, refs ->
            AnnotationModuleHarness.bundleRow(pairId, segmentation, panelDir, refs)
        }
        .collect()
        .subscribe { rows -> AnnotationModuleHarness.write(params.fixture_out, rows) }

    prepared_ch = channel
        .fromList(AnnotationModuleHarness.readList(params.prepared_inputs))
        .map { item ->
            tuple(
                item.pair_id, item.segmentation, item.samples_json,
                file(item.config), file(item.prepared_dir),
            )
        }
    mapped = ANNOTATION_PREPARED_REFERENCES(prepared_ch)
    mapped.map_inputs
        .map { item -> AnnotationModuleHarness.mapInputRow(item) }
        .collect()
        .subscribe { rows -> AnnotationModuleHarness.write(params.prepared_out, rows) }
}
"""


def _nextflow_env() -> dict[str, str]:
    env = {**os.environ, "NXF_DISABLE_CHECK_LATEST": "true", "NXF_ANSI_LOG": "false"}
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT / "src"), *filter(None, [os.environ.get("PYTHONPATH")])]
    )
    env["PATH"] = os.pathsep.join([str(Path(sys.executable).parent), env["PATH"]])
    env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def _read_trace(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _resolved_params() -> dict[str, Any]:
    """Base + annotation.config params as Nextflow resolves them (no dwight)."""
    assert NEXTFLOW is not None
    completed = subprocess.run(
        [NEXTFLOW, "config", "-o", "json", "-profile", "conda", str(WORKFLOWS)],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
        env=_nextflow_env(),
    )
    params: dict[str, Any] = json.loads(completed.stdout)["params"]
    return params


def _cases(root: Path, params: dict[str, Any]) -> dict[str, dict[str, Any]]:
    gene_list = root / "genes.csv"
    gene_list.write_text("gene,ensembl_id\nGENE0,ENSG00000000000\n")
    region = root / "region_precompute"
    region.mkdir()
    store = root / "store_parent" / "store"
    (root / "store_parent").mkdir()
    readonly = root / "readonly"
    readonly.mkdir()
    readonly.chmod(0o555)
    human = {**params, "species": "human", "outdir": str(root / "results")}
    ready = {
        **human,
        "annotation_prepare_only": True,
        "annotation_panel_genes_path": str(gene_list),
        "annotation_whb_region_precompute_source": str(region),
        # The held-out self-map (resolvability on by default) reads these.
        "annotation_whb_h5ad_dir": str(region),
        "annotation_whb_metadata_dir": str(region),
        "annotation_reference_store": str(store),
    }
    bundle = {
        "key": "human|whb_frontal_supc_clus|" + _hash("a"),
        "species": "human",
        "reference_id": "whb_frontal_supc_clus",
        "role": "primary",
        "panel_hash": _hash("a"),
        "panel_tag": _hash("a"),
        "n_panel_genes": 296,
    }
    bad_required = root / "bad_required"
    bad_required.mkdir()
    (bad_required / "required_bundles.json").write_text(
        json.dumps({"bundles": [{"reference_id": "x"}], "n_required": 2})
    )
    return {
        "config_human": {"fn": "annotationConfig", "params": human, "species": "human"},
        "config_mouse": {
            "fn": "annotationConfig",
            "params": {**params, "annotation_wmb_max_cells_per_cluster": 20},
            "species": "mouse",
        },
        "config_human_custom": {
            "fn": "annotationConfig",
            "params": {
                **human,
                "annotation_human_references": "whb_frontal_supc_clus",
                "annotation_panel_mode": "per_platform",
                "annotation_gene_id_fallback_csv": str(gene_list),
                "annotation_human_gene_table": str(gene_list),
                "annotation_mouse_gene_table": str(gene_list),
                "annotation_xplat_sensitivity_segmentations": "proseg_hybrid,reseg",
                "annotation_resolvability": False,
            },
            "species": "human",
        },
        "panel_args_gene_list": {
            "fn": "panelArguments",
            "spec": {
                "source": "gene_list",
                "gene_list": "set a.csv",
                "platforms": ["MERSCOPE"],
            },
            "pair_id": "P1",
            "segmentation": "proseg_hybrid",
        },
        "prepared_spec": {
            "fn": "preparedPanelSpec",
            "config": "clustering_squidpy_config.json",
            "prepared": "clustering_prepare_out",
            "masks": ["shared_tissue_mask.npy", "registration_summary.json"],
        },
        "panel_args_prepared_require_mask": {
            "fn": "panelArguments",
            "spec": {
                "source": "prepared",
                "clustering_config": "clustering_squidpy_config.json",
                "prepared_dir": "clustering_prepare_out",
                "require_shared_tissue_mask": True,
            },
            "pair_id": "P1",
            "segmentation": "proseg_hybrid",
        },
        "panel_args_prepared": {
            "fn": "panelArguments",
            "spec": {
                "source": "prepared",
                "clustering_config": "clustering_squidpy_config.json",
                "prepared_dir": "clustering_prepare_out",
                "shared_tissue_mask": "shared_tissue_mask.npy",
                "registration_summary": "registration_summary.json",
            },
            "pair_id": "P1",
            "segmentation": "reseg",
        },
        "panel_args_unknown": {
            "fn": "panelArguments",
            "spec": {"source": "zarr"},
            "pair_id": "P1",
            "segmentation": "reseg",
        },
        "prep_args_panel": {
            "fn": "prepArguments",
            "params": {
                **ready,
                "annotation_seaad_precomputed_stats_path": str(gene_list),
                "annotation_reference_store_large": str(root / "large"),
            },
            "bundle": bundle,
            "panel_genes": "panel_genes/panel_genes.json",
            "cpus": 8,
            "memory_gb": 64,
        },
        "prep_args_region_share": {
            "fn": "prepArguments",
            "params": {**params, "annotation_auto_download": False},
            "bundle": {
                **bundle,
                "species": "mouse",
                "reference_id": "wmb_region_share",
                "role": "region_share",
                "panel_hash": None,
                "panel_tag": "panel_independent",
            },
            "panel_genes": None,
            "cpus": 8,
            "memory_gb": 64,
        },
        "store_default": {"fn": "referenceStore", "params": {"outdir": "rel_results"}},
        "preflight_ready": {"fn": "prepareOnlyErrors", "params": ready},
        "preflight_no_gene_list": {
            "fn": "prepareOnlyErrors",
            "params": {**ready, "annotation_panel_genes_path": None},
        },
        "preflight_missing_paths": {
            "fn": "prepareOnlyErrors",
            "params": {
                **ready,
                "annotation_panel_genes_path": str(root / "missing.csv"),
                "annotation_whb_region_precompute_source": None,
                "annotation_whb_h5ad_dir": None,
                "annotation_whb_metadata_dir": None,
            },
        },
        "preflight_no_pipeline_sources": {
            "fn": "prepareOnlyErrors",
            "params": {
                **ready,
                "annotation_human_references": (
                    "whb_frontal_supc_clus,whb_whole_ctx_panel"
                ),
            },
        },
        "preflight_human_no_selfmap_sources": {
            "fn": "prepareOnlyErrors",
            "params": {
                **ready,
                "annotation_whb_h5ad_dir": None,
                "annotation_whb_metadata_dir": None,
            },
        },
        "preflight_human_seaad_no_selfmap_sources": {
            "fn": "prepareOnlyErrors",
            "params": {
                **ready,
                "annotation_human_references": "seaad_mr_panel",
                "annotation_whb_h5ad_dir": None,
            },
        },
        "preflight_human_resolvability_off": {
            "fn": "prepareOnlyErrors",
            "params": {
                **ready,
                "annotation_whb_h5ad_dir": None,
                "annotation_whb_metadata_dir": None,
                "annotation_resolvability": False,
            },
        },
        "preflight_mouse": {
            "fn": "prepareOnlyErrors",
            "params": {**ready, "species": "mouse"},
        },
        "preflight_mouse_no_test_cells": {
            "fn": "prepareOnlyErrors",
            "params": {
                **ready,
                "species": "mouse",
                "annotation_wmb_h5ad_dir": str(region),
                "annotation_wmb_metadata_dir": str(region),
                "annotation_wmb_mapping_stats_path": str(gene_list),
            },
        },
        "preflight_mouse_resolvability_off": {
            "fn": "prepareOnlyErrors",
            "params": {
                **ready,
                "species": "mouse",
                "annotation_wmb_h5ad_dir": str(region),
                "annotation_wmb_metadata_dir": str(region),
                "annotation_wmb_mapping_stats_path": str(gene_list),
                "annotation_resolvability": False,
            },
        },
        "preflight_readonly_store": {
            "fn": "prepareOnlyErrors",
            "params": {**ready, "annotation_reference_store": str(readonly / "store")},
        },
        "preflight_region": {
            "fn": "prepareOnlyErrors",
            "params": {**ready, "annotation_human_region": "hippocampus"},
        },
        "ordered_missing": {
            "fn": "orderedBundleRefs",
            "required": {
                "bundles": [
                    {"species": "human", "reference_id": "a", "panel_hash": "h"},
                    {"species": "human", "reference_id": "b", "panel_hash": None},
                ]
            },
            "keys": ["human|a|h"],
            "refs": ["a.json"],
        },
        "ordered": {
            "fn": "orderedBundleRefs",
            "required": {
                "bundles": [
                    {"species": "human", "reference_id": "a", "panel_hash": "h"},
                    {"species": "human", "reference_id": "b", "panel_hash": None},
                ]
            },
            "keys": ["human|b|panel_independent", "human|a|h"],
            "refs": ["b.json", "a.json"],
        },
        "required_count_mismatch": {
            "fn": "requiredBundles",
            "panel_dir": str(bad_required),
        },
        "prep_requests": {
            "fn": "prepRequests",
            "panel_dir": str(root / "fixture_panels" / "F5_original_seg"),
        },
        "map_spec_per_platform": {
            "fn": "mapSpec",
            "panel_dir": str(root / "fixture_panels" / "F3_proseg_hybrid"),
        },
        "map_spec_mouse": {
            "fn": "mapSpec",
            "panel_dir": str(root / "fixture_panels" / "F5_original_seg"),
        },
        "map_spec_refused": {
            "fn": "mapSpec",
            "panel_dir": str(root / "fixture_panels" / "F4_proseg_hybrid"),
        },
        "map_arguments": {
            "fn": "mapArguments",
            "pair_id": "P7513",
            "segmentation": "proseg_hybrid",
            "spec": {"species": "human"},
            "refs": [
                "map_inputs/bundle_refs/bundle_ref_1.json",
                "map_inputs/bundle_refs/bundle_ref_2.json",
            ],
            "cpus": 6,
        },
        "map_arguments_refused": {
            "fn": "mapArguments",
            "pair_id": "P1",
            "segmentation": "reseg",
            "spec": {"species": "human"},
            "refs": [],
            "cpus": 6,
        },
        "map_reuse_dir": {
            "fn": "mapReuseDir",
            "params": {"outdir": "rel_results"},
            "pair_id": "P1",
            "segmentation": "reseg",
        },
        "map_reuse_dir_off": {
            "fn": "mapReuseDir",
            "params": {"outdir": "rel_results", "annotation_reuse_published": False},
            "pair_id": "P1",
            "segmentation": "reseg",
        },
        "prepare_only_default": {"fn": "prepareOnly", "params": params},
        "prepare_only_string": {
            "fn": "prepareOnly",
            "params": {"annotation_prepare_only": "true"},
        },
    }


def _run_harness(root: Path) -> dict[str, Any]:
    """Build the harness inputs under ``root`` and run it (``-stub-run``)."""
    assert NEXTFLOW is not None
    resolved_params = _resolved_params()
    (root / "lib").mkdir(parents=True)
    for source in LIB_DIR.glob("*.groovy"):
        shutil.copy(source, root / "lib" / source.name)
    (root / "lib" / "AnnotationModuleHarness.groovy").write_text(HARNESS_CLASS)
    (root / "main.nf").write_text(
        HARNESS_MAIN.replace("__SUBWORKFLOW__", str(SUBWORKFLOW))
    )
    cases = _cases(root, resolved_params)
    (root / "cases.json").write_text(json.dumps(cases))
    (root / "fixture_panels.json").write_text(json.dumps(_write_fixture_panels(root)))

    outdir = root / "results"
    prepared_items = []
    for pair_id, segmentation, merscope_extra, xenium_extra in (
        ("PSAME", "proseg_hybrid", 0, 0),
        ("PSAME", "reseg", 0, 0),
        ("PDIFF", "proseg_hybrid", 40, 5),
    ):
        prepared, config, samples_json = _prepared_pair(
            root / "prepared",
            pair_id,
            segmentation,
            merscope_extra=merscope_extra,
            xenium_extra=xenium_extra,
        )
        prepared_items.append(
            {
                "pair_id": pair_id,
                "segmentation": segmentation,
                "samples_json": samples_json,
                "config": str(config),
                "prepared_dir": str(prepared),
            }
        )
    _publish_shared_tissue_mask(outdir, "PSAME")
    (root / "prepared_inputs.json").write_text(json.dumps(prepared_items))

    (root / "nextflow.config").write_text(
        f"""
includeConfig '{ANNOTATION_CONFIG}'
params {{
    species = "human"
    outdir = "{outdir}"
    annotation_reference_store = "{root / "store"}"
    annotation_prep_large_memory = "{LARGE_MEMORY}"
    cases = "{root / "cases.json"}"
    function_results = "{root / "function_results.json"}"
    fixture_panels = "{root / "fixture_panels.json"}"
    fixture_out = "{root / "fixture_out.json"}"
    prepared_inputs = "{root / "prepared_inputs.json"}"
    prepared_out = "{root / "prepared_out.json"}"
}}
process {{
    executor = "local"
    errorStrategy = "ignore"
    withName: ".*FIXTURE_BUNDLES:ANNOTATE_REFERENCE_PREP" {{
        beforeScript = {{
            bundle.reference_id == "failing_reference" ? "exit 3" : "true"
        }}
    }}
}}
trace {{
    enabled = true
    overwrite = true
    file = "{root / "trace.tsv"}"
    fields = "process,tag,status,memory,time"
}}
"""
    )
    completed = subprocess.run(
        [NEXTFLOW, "-log", str(root / "nextflow.log"), "run", "main.nf", "-stub-run"],
        cwd=root,
        env=_nextflow_env(),
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return {
        "root": str(root),
        "functions": json.loads((root / "function_results.json").read_text()),
        "fixture": json.loads((root / "fixture_out.json").read_text()),
        "prepared": json.loads((root / "prepared_out.json").read_text()),
        "trace": _read_trace(root / "trace.tsv"),
        "outdir": str(outdir),
    }


@pytest.fixture(scope="module")
def harness(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Run the harness once per test session, also under pytest-xdist.

    xdist gives each worker its own module fixtures; the workers share the
    parent of their base temp directories, so the first one runs Nextflow
    under an exclusive lock and the others read its results.
    """
    base = tmp_path_factory.getbasetemp()
    shared = base.parent if os.environ.get("PYTEST_XDIST_WORKER") else base
    with (shared / "annotation_module_harness.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        results_path = shared / "annotation_module_harness.json"
        if not results_path.exists():
            results = _run_harness(shared / "annotation_module_harness")
            results_path.write_text(json.dumps(results))
        results = json.loads(results_path.read_text())
    return {**results, "root": Path(results["root"]), "outdir": Path(results["outdir"])}


def _value(harness: dict[str, Any], name: str) -> Any:
    result = harness["functions"][name]
    assert "error" not in result, result
    return result["value"]


def _error(harness: dict[str, Any], name: str) -> str:
    result = harness["functions"][name]
    assert "error" in result, result
    return str(result["error"])


def _prep_rows(harness: dict[str, Any], prefix: str) -> list[dict[str, str]]:
    return [
        row
        for row in harness["trace"]
        if row["process"].startswith(prefix) and row["process"].endswith(PREP)
    ]


def _refs(paths: list[str]) -> list[tuple[str, str | None]]:
    refs = [json.loads(Path(path).read_text()) for path in paths]
    return [(ref["reference_id"], ref["panel_hash"]) for ref in refs]


# --------------------------------------------------------------------------
# Channel logic


@needs_nextflow
def test_prep_runs_once_per_unique_species_reference_and_panel(
    harness: dict[str, Any],
) -> None:
    rows = _prep_rows(harness, "FIXTURE_BUNDLES")
    expected_keys = {
        (reference_id, (_hash(tag) if tag else "panel_independent")[:12])
        for bundles in FIXTURE_BRANCHES.values()
        for reference_id, _species, _role, tag, _n in bundles
    }
    tags = Counter(tuple(row["tag"].split(":")) for row in rows)
    assert set(tags) == expected_keys
    assert all(count == 1 for count in tags.values()), tags
    # F1, F2 and F7 stage different panel files for set a "a" (other sample
    # ids and symbols): still one PREP task, whose bundle all three share.
    shared = f"whb_frontal_supc_clus:{_hash('a')[:12]}"
    assert tags[tuple(shared.split(":"))] == 1
    # 20 bundle needs across 9 branches, 13 distinct bundles.
    assert sum(len(bundles) for bundles in FIXTURE_BRANCHES.values()) == 20
    assert len(rows) == len(expected_keys) == 13
    failed = [row["tag"] for row in rows if row["status"] == "FAILED"]
    assert failed == [f"failing_reference:{_hash('b')[:12]}"]


@needs_nextflow
def test_each_branch_is_released_with_exactly_its_required_bundles(
    harness: dict[str, Any],
) -> None:
    released = {
        (row["pair_id"], row["segmentation"]): row for row in harness["fixture"]
    }
    expected = {
        branch: [
            (reference_id, _hash(tag) if tag else None)
            for reference_id, _species, _role, tag, _n in bundles
        ]
        for branch, bundles in FIXTURE_BRANCHES.items()
        if branch != ("F7", "proseg_hybrid")
    }
    assert set(released) == set(expected)
    for branch, bundles in expected.items():
        assert _refs(released[branch]["refs"]) == bundles, branch
    # groupKey sizes: set c 3, reseg 2, per_platform 5, mouse 2, refused 0.
    assert len(released[("F1", "proseg_hybrid")]["refs"]) == 3
    # Branches differing only in their panel files' symbols get one ref.
    assert (
        released[("F1", "proseg_hybrid")]["refs"][0]
        == released[("F2", "proseg_hybrid")]["refs"][0]
    )
    assert len(released[("F3", "proseg_hybrid")]["refs"]) == 5
    assert released[("F4", "proseg_hybrid")]["refs"] == []


@needs_nextflow
def test_a_failed_prep_drops_only_the_branches_that_need_it(
    harness: dict[str, Any],
) -> None:
    released = {(row["pair_id"], row["segmentation"]) for row in harness["fixture"]}
    assert ("F7", "proseg_hybrid") not in released
    # F7 shares set a with F1 and F2, which are released.
    assert ("F1", "proseg_hybrid") in released and ("F2", "proseg_hybrid") in released


@needs_nextflow
def test_prep_resources_follow_the_panel_size(harness: dict[str, Any]) -> None:
    rows = {row["tag"]: row for row in _prep_rows(harness, "FIXTURE_BUNDLES")}
    large = rows[f"whb_frontal_supc_clus:{_hash('l')[:12]}"]
    assert large["memory"] == LARGE_MEMORY and large["time"] == "1d"
    normal = rows[f"whb_frontal_supc_clus:{_hash('a')[:12]}"]
    assert normal["memory"] == "64 GB" and normal["time"] == "8h"
    region = rows["wmb_region_share:panel_indepe"]
    assert region["memory"] == "64 GB" and region["time"] == "8h"


@needs_nextflow
def test_prepared_references_emit_map_inputs_after_their_bundles(
    harness: dict[str, Any],
) -> None:
    """Real ANNOTATE_PANEL on prepared H5ADs: same-panel (2) and per_platform (5).

    PSAME's synthetic set a has no curated set-c family, so its label-free
    set c needs the shared tissue mask, which M2 never takes from a published
    file (M5 wires ALIGN): set c is refused although PSAME has a published
    mask. The set-c groupKey path is covered by the fixture panels.
    """
    released = {
        (row["pair_id"], row["segmentation"]): row for row in harness["prepared"]
    }
    assert set(released) == {
        ("PSAME", "proseg_hybrid"),
        ("PSAME", "reseg"),
        ("PDIFF", "proseg_hybrid"),
    }
    purposes = {}
    for branch, row in released.items():
        required = json.loads(
            (Path(row["panel_dir"]) / "required_bundles.json").read_text()
        )
        assert required["n_required"] == len(row["refs"])
        assert _refs(row["refs"]) == [
            (item["reference_id"], item["panel_hash"]) for item in required["bundles"]
        ]
        purposes[branch] = [
            (item["reference_id"], item["purpose"]) for item in required["bundles"]
        ]
        assert Path(row["prepared_dir"]).name == "clustering_prepare_out"
        assert json.loads(row["samples_json"])[0]["segmentation"] == branch[1]
    assert purposes[("PSAME", "proseg_hybrid")] == [
        ("whb_frontal_supc_clus", "annotation"),
        ("seaad_mr_panel", "annotation"),
    ]
    assert purposes[("PSAME", "reseg")] == [
        ("whb_frontal_supc_clus", "annotation"),
        ("seaad_mr_panel", "annotation"),
    ]
    assert len(purposes[("PDIFF", "proseg_hybrid")]) == 5
    report = json.loads(
        (
            Path(released[("PSAME", "proseg_hybrid")]["panel_dir"])
            / "panel_report.json"
        ).read_text()
    )
    assert report["setc"] is None
    assert "needs the shared tissue mask" in report["setc_skipped"]
    # The per_platform intersection is PSAME's set a: one WHB bundle serves both.
    rows = _prep_rows(harness, "ANNOTATION_PREPARED_REFERENCES")
    tags = Counter(row["tag"] for row in rows)
    assert all(count == 1 for count in tags.values())
    assert len(rows) == 6


@needs_nextflow
def test_prep_stub_writes_a_valid_bundle_ref(harness: dict[str, Any]) -> None:
    from merxen.annotation.store import BundleRef

    row = harness["prepared"][0]
    for path in row["refs"]:
        ref = BundleRef.model_validate_json(Path(path).read_text())
        assert ref.store_root == str(harness["root"] / "store")
    published = list(
        (harness["outdir"] / "annotation_reference_prep").glob("*/*/bundle_ref.json")
    )
    assert len(published) >= 7


# --------------------------------------------------------------------------
# Groovy helpers


@needs_nextflow
@pytest.mark.parametrize("species", ["human", "mouse"])
def test_annotation_config_json_validates_in_python(
    harness: dict[str, Any], species: str
) -> None:
    raw = _value(harness, f"config_{species}")
    config = AnnotationConfig.model_validate(raw)
    assert config.enabled is True and config.species == species
    assert [spec.reference_id for spec in config.references] == list(
        DEFAULT_REFERENCE_IDS[species]
    )
    assert config.ctm_version == "1.7.2"
    assert config.reference_store is not None and config.reference_store.is_absolute()
    if species == "mouse":
        assert config.references[0].max_cells_per_cluster == 20
        assert config.references[0].drop_level == "CCN20230722_SUPT"
        assert config.mouse_section_regions == "auto"
    else:
        assert config.anatomical_region == "frontal_cortex"
        assert config.panel.gene_id_fallback_csv is None
        assert config.panel.gene_tables == {}


@needs_nextflow
def test_annotation_config_json_follows_the_params(harness: dict[str, Any]) -> None:
    config = AnnotationConfig.model_validate(_value(harness, "config_human_custom"))
    assert [spec.reference_id for spec in config.references] == [
        "whb_frontal_supc_clus"
    ]
    assert config.panel.panel_mode == "per_platform"
    assert config.panel.gene_id_fallback_csv is not None
    # Both species' gene tables reach the exact-case species test (plan §8.4).
    assert set(config.panel.gene_tables) == {"human", "mouse"}
    assert all(path.is_absolute() for path in config.panel.gene_tables.values())
    assert config.xplat_sensitivity_segmentations == ["proseg_hybrid", "reseg"]
    assert config.resolvability.enabled is False


@needs_nextflow
def test_panel_arguments(harness: dict[str, Any]) -> None:
    assert _value(harness, "panel_args_gene_list") == (
        "--pair-id P1 --segmentation proseg_hybrid "
        "--panel-genes-path 'panel_inputs/set a.csv' --platforms MERSCOPE"
    )
    assert _value(harness, "panel_args_prepared") == (
        "--pair-id P1 --segmentation reseg "
        "--prepared-dir panel_inputs/clustering_prepare_out "
        "--clustering-config panel_inputs/clustering_squidpy_config.json "
        "--shared-tissue-mask panel_inputs/shared_tissue_mask.npy "
        "--registration-summary panel_inputs/registration_summary.json"
    )
    spec = _value(harness, "prepared_spec")
    assert spec["shared_tissue_mask"] == "shared_tissue_mask.npy"
    assert spec["require_shared_tissue_mask"] is True
    assert _value(harness, "panel_args_prepared_require_mask").endswith(
        "--clustering-config panel_inputs/clustering_squidpy_config.json "
        "--require-shared-tissue-mask"
    )
    assert "Unknown annotation panel source" in _error(harness, "panel_args_unknown")


@needs_nextflow
def test_prep_arguments(harness: dict[str, Any]) -> None:
    args = _value(harness, "prep_args_panel").split()
    root = harness["root"]
    assert args[:4] == ["--reference-id", "whb_frontal_supc_clus", "--species", "human"]
    assert args[args.index("--store") + 1] == str(root / "store_parent" / "store")
    assert args[args.index("--store-large") + 1] == str(root / "large")
    assert args[args.index("--n-processors") + 1] == "8"
    assert args[args.index("--max-gb") + 1] == "40"
    assert args[args.index("--panel-genes") + 1] == "panel_genes/panel_genes.json"
    assert args[args.index("--scratch-dir") + 1] == "prep_scratch"
    assert "--auto-download" in args
    sources = [args[i + 1] for i, arg in enumerate(args) if arg == "--source"]
    assert sources == [
        f"region_precompute={root / 'region_precompute'}",
        f"seaad_precomputed_stats={root / 'genes.csv'}",
        f"whb_h5ad_dir={root / 'region_precompute'}",
        f"whb_metadata_dir={root / 'region_precompute'}",
    ]
    region = _value(harness, "prep_args_region_share").split()
    assert "--no-auto-download" in region
    assert "--panel-genes" not in region and "--source" not in region


@needs_nextflow
def test_map_spec_reads_species_status_and_panel_size(harness: dict[str, Any]) -> None:
    assert _value(harness, "map_spec_per_platform") == {
        "species": "human",
        "panel_status": "ok",
        "reasons": [],
        "n_required": 5,
        "n_panel_genes": 500,
    }
    mouse = _value(harness, "map_spec_mouse")
    assert mouse["species"] == "mouse" and mouse["n_required"] == 2
    assert mouse["n_panel_genes"] == 500
    refused = _value(harness, "map_spec_refused")
    assert refused["panel_status"] == "refused"
    assert refused["n_required"] == 0 and refused["n_panel_genes"] == 0


@needs_nextflow
def test_map_arguments(harness: dict[str, Any]) -> None:
    """MAP maps exactly PREP's refs, on the clustering run's table cells."""
    args = _value(harness, "map_arguments").split(" ")
    assert args[: args.index("--n-processors")] == [
        "--species",
        "human",
        "--pair-id",
        "P7513",
        "--segmentation",
        "proseg_hybrid",
        "--prepared-dir",
        "map_inputs/clustering_prepare_out",
        "--clustering-config",
        "map_inputs/clustering_squidpy_config.json",
        "--panel-dir",
        "map_inputs/annotation_panel_out",
    ]
    assert args[args.index("--n-processors") + 1] == "6"
    assert args[args.index("--work-dir") + 1] == "map_scratch"
    assert "--require-bundle-refs" in args and "--allow-refused-panel" in args
    refs = [args[i + 1] for i, arg in enumerate(args) if arg == "--bundle-ref"]
    assert refs == [
        "map_inputs/bundle_refs/bundle_ref_1.json",
        "map_inputs/bundle_refs/bundle_ref_2.json",
    ]
    # The standalone-only options are never passed by the pipeline.
    for option in ("--store", "--bundle ", "--from-clustered-h5ad", "--min-counts"):
        assert option not in _value(harness, "map_arguments") + " "
    refused = _value(harness, "map_arguments_refused")
    assert "--bundle-ref" not in refused and "--allow-refused-panel" in refused


@needs_nextflow
def test_map_reuse_dir_is_the_published_map_output(harness: dict[str, Any]) -> None:
    reuse = Path(_value(harness, "map_reuse_dir"))
    assert reuse.is_absolute()
    assert reuse.parts[-5:] == (
        "rel_results",
        "P1",
        "reseg",
        "annotation_map",
        "annotation_map_out",
    )
    assert _value(harness, "map_reuse_dir_off") is None


@needs_nextflow
def test_reference_store_defaults_under_outdir(harness: dict[str, Any]) -> None:
    store = Path(_value(harness, "store_default"))
    assert store.is_absolute() and store.parts[-2:] == (
        "rel_results",
        "annotation_references",
    )


@needs_nextflow
def test_prepare_only_preflight(harness: dict[str, Any]) -> None:
    assert _value(harness, "preflight_ready") == []
    (no_gene_list,) = _value(harness, "preflight_no_gene_list")
    assert "annotation_panel_genes_path" in no_gene_list and "M5" in no_gene_list
    missing = " | ".join(_value(harness, "preflight_missing_paths"))
    assert "Missing annotation_panel_genes_path" in missing
    assert (
        "whb_frontal_supc_clus needs annotation_whb_region_precompute_source" in missing
    )
    (no_sources,) = _value(harness, "preflight_no_pipeline_sources")
    assert "whb_whole_ctx_panel have no pipeline source params" in no_sources
    # The human self-map needs the WHB held-out donor's sources up front.
    (no_selfmap,) = _value(harness, "preflight_human_no_selfmap_sources")
    assert "whb_frontal_supc_clus, seaad_mr_panel need" in no_selfmap
    assert "annotation_whb_h5ad_dir, annotation_whb_metadata_dir" in no_selfmap
    seaad_only = " | ".join(_value(harness, "preflight_human_seaad_no_selfmap_sources"))
    assert "seaad_mr_panel need annotation_whb_h5ad_dir" in seaad_only
    assert _value(harness, "preflight_human_resolvability_off") == []
    mouse = " | ".join(_value(harness, "preflight_mouse"))
    assert "wmb_panel needs annotation_wmb_h5ad_dir" in mouse
    assert "wmb_region_share" not in mouse  # the MERFISH metadata is downloaded
    (no_test_cells,) = _value(harness, "preflight_mouse_no_test_cells")
    assert "wmb_panel needs annotation_wmb_selfmap_test_cells_path" in no_test_cells
    assert _value(harness, "preflight_mouse_resolvability_off") == []
    (readonly,) = _value(harness, "preflight_readonly_store")
    assert "is not writable" in readonly
    (region,) = _value(harness, "preflight_region")
    assert "hippocampus" in region and "OD-C7" in region


@needs_nextflow
def test_prep_requests_depend_only_on_the_bundle_key(harness: dict[str, Any]) -> None:
    """PREP's val input is fixed by its key, whichever pair asks first."""
    wmb, region = _value(harness, "prep_requests")
    assert wmb["bundle"] == {
        "key": f"mouse|wmb_panel|{_hash('g')}",
        "species": "mouse",
        "reference_id": "wmb_panel",
        "role": "primary",
        "panel_hash": _hash("g"),
        "panel_tag": _hash("g"),
        "n_panel_genes": 500,
    }
    assert [Path(path).name for path in wmb["files"]] == ["panel_genes_g.json"]
    assert region["bundle"]["key"] == "mouse|wmb_region_share|panel_independent"
    assert region["bundle"]["panel_hash"] is None and region["files"] == []


@needs_nextflow
def test_bundle_bookkeeping_helpers(harness: dict[str, Any]) -> None:
    assert _value(harness, "ordered") == ["a.json", "b.json"]
    assert "human|b|panel_independent was not collected" in _error(
        harness, "ordered_missing"
    )
    assert "n_required 2 != 1 bundles" in _error(harness, "required_count_mismatch")
    assert _value(harness, "prepare_only_default") is False
    assert _value(harness, "prepare_only_string") is True


# --------------------------------------------------------------------------
# main.nf --annotation_prepare_only
#
# Each of these launches main.nf (about 10-22 s of Nextflow start-up each), so
# they are marked slow and stay out of the pre-push suite; the session
# harness above covers the channel logic, arguments and preflight fast.


def _host_free_config(tmp_path: Path, **params: str) -> Path:
    """A ``-c`` config that unsets the host source paths of the dwight profile.

    The runs use the default (dwight) profile, whose conda-free execution
    matches the workstation; the overrides keep them independent of the
    host's reference files and stores.
    """
    host_params = [
        "annotation_reference_store_large",
        "annotation_gene_id_fallback_csv",
        "annotation_whb_region_precompute_source",
        "annotation_whb_h5ad_dir",
        "annotation_whb_metadata_dir",
        "annotation_wmb_h5ad_dir",
        "annotation_wmb_metadata_dir",
        "annotation_wmb_mapping_stats_path",
        "annotation_wmb_selfmap_test_cells_path",
    ]
    lines = [f"    {name} = null" for name in host_params if name not in params]
    lines += [f'    {name} = "{value}"' for name, value in params.items()]
    config = tmp_path / "host_free.config"
    config.write_text("params {\n" + "\n".join(lines) + "\n}\n")
    return config


@pytest.mark.slow
@needs_nextflow
def test_prepare_only_stub_run_builds_bundles_and_runs_no_stage(tmp_path: Path) -> None:
    """Two paired rows x two segmentations: 4 panels, 2 bundles, nothing else."""
    assert NEXTFLOW is not None
    genes = tmp_path / "set_a.csv"
    rows = ["gene,ensembl_id"] + [
        f"GENE{i},{gene}" for i, gene in enumerate(_human_ids(80))
    ]
    genes.write_text("\n".join(rows) + "\n")
    samplesheet = tmp_path / "samplesheet.csv"
    samplesheet.write_text(
        "pair_id,analysis_mode,merscope_dir,xenium_dir\n"
        "P0001,paired,/nonexistent/m1,/nonexistent/x1\n"
        "P0002,paired,/nonexistent/m2,/nonexistent/x2\n"
    )
    region = tmp_path / "region_precompute"
    region.mkdir()
    outdir = tmp_path / "results"
    store = tmp_path / "store"
    trace = tmp_path / "trace.tsv"
    config = _host_free_config(
        tmp_path,
        annotation_reference_store=str(store),
        annotation_whb_region_precompute_source=str(region),
    )
    completed = subprocess.run(
        [
            NEXTFLOW,
            "-log",
            str(tmp_path / "nextflow.log"),
            "-c",
            str(config),
            "run",
            str(MAIN_NF),
            "-stub-run",
            "-with-trace",
            str(trace),
            "--samplesheet",
            str(samplesheet),
            "--outdir",
            str(outdir),
            "--species",
            "human",
            "--analysis_segmentation",
            "proseg_hybrid,reseg",
            "--annotation_prepare_only",
            "true",
            "--annotation_panel_genes_path",
            str(genes),
        ],
        cwd=tmp_path,
        env=_nextflow_env(),
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    processes = Counter(row["name"].split(" (")[0] for row in _read_trace(trace))
    assert processes == {
        "ANNOTATION_PREPARE_ONLY:ANNOTATION_REFERENCES:ANNOTATE_PANEL": 4,
        "ANNOTATION_PREPARE_ONLY:ANNOTATION_REFERENCES:ANNOTATION_BUNDLES:"
        "ANNOTATE_REFERENCE_PREP": 2,
    }
    assert "no pipeline stage runs" in completed.stdout
    for pair_id in ("P0001", "P0002"):
        for segmentation in ("proseg_hybrid", "reseg"):
            out = (
                outdir
                / pair_id
                / segmentation
                / "annotation_panel"
                / "annotation_panel_out"
            )
            required = json.loads((out / "required_bundles.json").read_text())
            assert [b["reference_id"] for b in required["bundles"]] == list(
                DEFAULT_REFERENCE_IDS["human"]
            )
            assert {b["n_panel_genes"] for b in required["bundles"]} == {80}
    refs = sorted((outdir / "annotation_reference_prep").glob("*/*/bundle_ref.json"))
    assert [path.parts[-3] for path in refs] == [
        "seaad_mr_panel",
        "whb_frontal_supc_clus",
    ]
    # A stub run writes nothing into the reference store.
    assert not store.exists()


FAKE_MERXEN = """#!__PYTHON__
# Test double: ANNOTATE_PANEL runs the real merxen; annotation-reference-prep
# records its arguments, validates its inputs and writes a bundle ref without
# building anything.
import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
if not args or args[0] != "annotation-reference-prep":
    os.execv("__REAL__", ["__REAL__", *args])

from merxen.annotation.config import AnnotationConfig
from merxen.annotation.panel import load_annotation_panel
from merxen.annotation.store import BundleRef


def value(flag):
    return args[args.index(flag) + 1]


config = AnnotationConfig.model_validate_json(
    Path(value("--annotation-config")).read_text()
)
panel = None
if "--panel-genes" in args:
    panel = load_annotation_panel(value("--panel-genes"))
reference_id = value("--reference-id")
spec = next(s for s in config.references if s.reference_id == reference_id)
Path("fake_prep_argv.json").write_text(json.dumps(args))
BundleRef(
    reference_id=spec.reference_id,
    species=config.species,
    role=spec.role,
    panel_hash=None if panel is None else panel.panel_hash,
    build_hash="f" * 64,
    path=str(Path(value("--store")) / spec.reference_id / ("f" * 64)),
    store_root=value("--store"),
    reused=False,
).write(value("--output"))
"""


@pytest.mark.slow
@needs_nextflow
def test_prepare_only_runs_the_real_prep_script(tmp_path: Path) -> None:
    """Without -stub-run: the PREP script checks ctm and passes its arguments.

    A fake ``merxen`` stands in for ``annotation-reference-prep`` only, so the
    rendered command (store, sources, --n-processors, --max-gb from the task
    memory, --panel-genes) and the annotation config are checked without
    building a bundle.
    """
    assert NEXTFLOW is not None
    env = _nextflow_env()
    real = shutil.which("merxen", path=env["PATH"])
    assert real is not None
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "merxen"
    fake.write_text(
        FAKE_MERXEN.replace("__PYTHON__", sys.executable).replace("__REAL__", real)
    )
    fake.chmod(0o755)
    env["PATH"] = os.pathsep.join([str(bin_dir), env["PATH"]])
    genes = tmp_path / "vzg2.csv"
    mouse_ids = [f"ENSMUSG{i:011d}" for i in range(80)]
    genes.write_text(
        "gene,ensembl_id\n"
        + "".join(f"Gene{i},{gene}\n" for i, gene in enumerate(mouse_ids))
    )
    samplesheet = tmp_path / "samplesheet.csv"
    samplesheet.write_text(
        "pair_id,analysis_mode,merscope_dir\nVZG2,merscope,/nonexistent/vzg2\n"
    )
    wmb_h5ad = tmp_path / "wmb_h5ad"
    wmb_metadata = tmp_path / "wmb_metadata"
    wmb_h5ad.mkdir()
    wmb_metadata.mkdir()
    mapping_stats = tmp_path / "precomputed_stats_ABC_revision_230821.h5"
    mapping_stats.write_bytes(b"stats")
    test_cells = tmp_path / "truth.csv"
    test_cells.write_text("cell_label\nAAAC-1\n")
    outdir = tmp_path / "results"
    store = tmp_path / "store"
    trace = tmp_path / "trace.tsv"
    config = _host_free_config(
        tmp_path,
        annotation_reference_store=str(store),
        annotation_wmb_h5ad_dir=str(wmb_h5ad),
        annotation_wmb_metadata_dir=str(wmb_metadata),
        annotation_wmb_mapping_stats_path=str(mapping_stats),
        annotation_wmb_selfmap_test_cells_path=str(test_cells),
    )
    completed = subprocess.run(
        [
            NEXTFLOW,
            "-log",
            str(tmp_path / "nextflow.log"),
            "-c",
            str(config),
            "run",
            str(MAIN_NF),
            "-with-trace",
            str(trace),
            "--samplesheet",
            str(samplesheet),
            "--outdir",
            str(outdir),
            "--species",
            "mouse",
            "--analysis_segmentation",
            "original_seg",
            "--annotation_prepare_only",
            "true",
            "--annotation_panel_genes_path",
            str(genes),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    processes = Counter(row["name"].split(" (")[0] for row in _read_trace(trace))
    assert processes == {
        "ANNOTATION_PREPARE_ONLY:ANNOTATION_REFERENCES:ANNOTATE_PANEL": 1,
        "ANNOTATION_PREPARE_ONLY:ANNOTATION_REFERENCES:ANNOTATION_BUNDLES:"
        "ANNOTATE_REFERENCE_PREP": 2,
    }
    argvs = {}
    for path in (tmp_path / "work").glob("*/*/fake_prep_argv.json"):
        args = json.loads(path.read_text())
        argvs[args[args.index("--reference-id") + 1]] = args
    assert set(argvs) == {"wmb_panel", "wmb_region_share"}
    wmb = argvs["wmb_panel"]
    assert wmb[wmb.index("--store") + 1] == str(store)
    assert wmb[wmb.index("--n-processors") + 1] == "8"
    assert wmb[wmb.index("--max-gb") + 1] == "40"
    assert wmb[wmb.index("--panel-genes") + 1] == "panel_genes/panel_genes.json"
    assert "--auto-download" in wmb and "--store-large" not in wmb
    assert [wmb[i + 1] for i, arg in enumerate(wmb) if arg == "--source"] == [
        f"wmb_h5ad_dir={wmb_h5ad}",
        f"wmb_mapping_stats={mapping_stats}",
        f"wmb_metadata_dir={wmb_metadata}",
        f"wmb_selfmap_test_cells={test_cells}",
    ]
    region = argvs["wmb_region_share"]
    assert "--panel-genes" not in region and "--source" not in region
    refs = {
        path.parts[-3]: json.loads(path.read_text())
        for path in (outdir / "annotation_reference_prep").glob("*/*/bundle_ref.json")
    }
    assert set(refs) == {"wmb_panel", "wmb_region_share"}
    assert refs["wmb_region_share"]["panel_hash"] is None
    assert not store.exists()


@pytest.mark.slow
@needs_nextflow
def test_prepare_only_refuses_a_run_without_a_gene_list(tmp_path: Path) -> None:
    assert NEXTFLOW is not None
    samplesheet = tmp_path / "samplesheet.csv"
    samplesheet.write_text("pair_id,analysis_mode\nP0001,paired\n")
    config = _host_free_config(
        tmp_path, annotation_reference_store=str(tmp_path / "store")
    )
    completed = subprocess.run(
        [
            NEXTFLOW,
            "-log",
            str(tmp_path / "nextflow.log"),
            "-c",
            str(config),
            "run",
            str(MAIN_NF),
            "-stub-run",
            "--samplesheet",
            str(samplesheet),
            "--outdir",
            str(tmp_path / "results"),
            "--annotation_prepare_only",
            "true",
        ],
        cwd=tmp_path,
        env=_nextflow_env(),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode != 0
    output = completed.stdout + completed.stderr
    assert "Preflight checks failed for --annotation_prepare_only" in output
    assert "needs --annotation_panel_genes_path" in output
    assert not (tmp_path / "work").exists() or not any((tmp_path / "work").iterdir())

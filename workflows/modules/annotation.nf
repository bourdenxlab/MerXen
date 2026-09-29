/*
 * Reference-based cell-type annotation processes (plan §3.1-§3.4, §11.4).
 *
 * ANNOTATE_PANEL resolves one pair x segmentation's declared panels and lists
 * the reference bundles it needs; ANNOTATE_REFERENCE_PREP gets or builds one
 * bundle per unique (species, reference_id, panel_hash);
 * CLUSTERING_SQUIDPY_ANNOTATE_MAP maps each sample of a pair x segmentation
 * onto its bundles with MapMyCells; CLUSTERING_SQUIDPY_ANNOTATE_RESOLVE turns
 * the mapping into one label table per sample (thresholds, floors, gate,
 * consensus, flags, compositions). All run on the CPU in the main
 * environment and take no GPU lock. COMPUTE_CPU (M5) is in
 * clustering_squidpy.nf, ANNOTATION_REPORT arrives in M7.
 * workflows/subworkflows/annotation_references.nf wires PANEL and PREP, only
 * for --annotation_prepare_only runs (hook H10) and map_first runs;
 * workflows/subworkflows/clustering_map_first.nf wires MAP after them
 * (CLUSTERING_ANNOTATE_MAP), RESOLVE after MAP (CLUSTERING_ANNOTATE) and
 * COMPUTE_CPU after RESOLVE (CLUSTERING_MAP_FIRST, called by hook H5 in
 * map_first runs only). Legacy runs never call any of them.
 *
 * Resources are in conf/annotation.config, host concurrency in
 * conf/dwight.annotation.config.
 */

process ANNOTATE_PANEL {
    tag "${pair_id}:${segmentation}"

    publishDir { "${params.outdir}/${pair_id}/${segmentation}/annotation_panel" }, mode: "copy", overwrite: true

    input:
    // panel_spec: AnnotationReferences.geneListPanelSpec or preparedPanelSpec;
    // panel_inputs: the files it names (gene list, or clustering config,
    // prepared directory and the optional shared tissue mask).
    tuple val(pair_id),
        val(segmentation),
        val(panel_spec),
        path(panel_inputs, stageAs: "panel_inputs/*")

    output:
    tuple val(pair_id),
        val(segmentation),
        path("annotation_panel_out")

    script:
    def species = AnnotationDefaults.normalizeSpecies(params.species)
    def annotationConfigJson = AnnotationReferences.annotationConfigJson(params, species)
    def panelArgs = AnnotationReferences.panelArguments(panel_spec, pair_id, segmentation)
    """
    set -euo pipefail
    export PYTHONPATH="${projectDir}/../src:\${PYTHONPATH:-}"
    export CUDA_VISIBLE_DEVICES=""
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    export NUMBA_NUM_THREADS=1

    cat > annotation_config.json <<'JSON'
${annotationConfigJson}
JSON

    merxen annotation-panel \\
        --species ${species} \\
        --annotation-config annotation_config.json \\
        ${panelArgs} \\
        --output-dir annotation_panel_out
    cp annotation_config.json annotation_panel_out/annotation_config.json
    """
}

process ANNOTATE_REFERENCE_PREP {
    tag "${bundle.reference_id}:${bundle.panel_tag.take(12)}"

    // PREP writes bundles only through merxen's ReferenceStore (flock, build
    // in <store>/.tmp-<uuid>, atomic rename, never deletes), never through a
    // storeDir: build_hash is known only inside the task (plan §3.1). It is
    // never cached: the task hash could not see the builder code or the
    // content of the --source files (paths only), so -resume would hand MAP
    // a stale bundle after a builder fix or a changed source. Re-running it
    // costs seconds when the bundle exists (the store checks build_hash),
    // and bundle_ref.json holds only the bundle's identity, so an unchanged
    // bundle gives byte-identical output and MAP (cache "deep" on it, M3)
    // stays cached. Same pattern as MATERIALIZE_ALIGNMENT.
    cache false

    publishDir { "${params.outdir}/annotation_reference_prep/${bundle.reference_id}/${bundle.panel_tag}" }, mode: "copy", overwrite: true

    input:
    // bundle: AnnotationReferences.prepRequests map (key, species,
    // reference_id, role, panel_hash, panel_tag, n_panel_genes), all fixed
    // by the bundle key.
    tuple val(bundle),
        path(panel_genes, arity: "0..*", stageAs: "panel_genes/*")

    output:
    tuple val(bundle),
        path("bundle_ref.json")

    script:
    def annotationConfigJson = AnnotationReferences.annotationConfigJson(params, bundle.species)
    def prepArgs = AnnotationReferences.prepArguments(
        params,
        bundle,
        panel_genes ? panel_genes[0] : null,
        task.cpus as int,
        task.memory.toGiga(),
    )
    """
    set -euo pipefail
    export PYTHONPATH="${projectDir}/../src:\${PYTHONPATH:-}"
    export CUDA_VISIBLE_DEVICES=""
    # cell_type_mapper parallelises over --n-processors worker processes;
    # more BLAS threads per worker would oversubscribe the reserved CPUs.
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    export NUMBA_NUM_THREADS=1

    python - <<'PY'
import importlib.metadata
import sys

expected = "${params.annotation_ctm_version}"
found = importlib.metadata.version("cell_type_mapper")
if found != expected:
    sys.exit(
        f"cell_type_mapper {found} is installed but annotation_ctm_version is "
        f"{expected}; rebuild the environment from requirements/requirements.lock"
    )
PY

    cat > annotation_config.json <<'JSON'
${annotationConfigJson}
JSON

    mkdir -p prep_scratch
    merxen annotation-reference-prep \\
        --annotation-config annotation_config.json \\
        ${prepArgs} \\
        --output bundle_ref.json
    """

    stub:
    def stubBundleRef = AnnotationReferences.stubBundleRefJson(params, bundle)
    """
    cat > bundle_ref.json <<'JSON'
${stubBundleRef}
JSON
    """
}

process CLUSTERING_SQUIDPY_ANNOTATE_MAP {
    tag "${pair_id}:${segmentation}"

    // PREP never caches and re-writes byte-identical bundle refs for an
    // unchanged bundle in a new work directory, so only content hashing
    // keeps an unchanged MAP cached (-resume). It hashes the prepared H5ADs
    // too: seconds per pair, against 3-5 min of mapping (M3 shadow runs).
    cache "deep"

    publishDir { "${params.outdir}/${pair_id}/${segmentation}/annotation_map" }, mode: "copy", overwrite: true

    input:
    // map_spec: AnnotationReferences.mapSpec(annotation_panel_out) (species,
    // panel status, panel size); bundle_refs: the bundle_ref.json of every
    // bundle the pair x segmentation's required_bundles.json lists, released
    // by ANNOTATION_BUNDLES once the last one is ready (none for a refused
    // panel).
    tuple val(pair_id),
        val(segmentation),
        val(map_spec),
        val(samples_json),
        path(clustering_config, stageAs: "map_inputs/clustering_squidpy_config.json"),
        path(prepared_dir, stageAs: "map_inputs/clustering_prepare_out"),
        path(panel_dir, stageAs: "map_inputs/annotation_panel_out"),
        path(bundle_refs, arity: "0..*", stageAs: "map_inputs/bundle_refs/bundle_ref_?.json")

    output:
    // annotation_map_out/<platform>/<sid>_mmc_<run_id>.parquet (tidy, per
    // cell x level), <sid>_ct_provisional.parquet (raw-threshold labels,
    // provisional until RESOLVE), map_manifest.json (query fingerprints,
    // bundle hashes, engine parameters, ctm version, wall time), logs/.
    tuple val(pair_id),
        val(segmentation),
        path("annotation_map_out")

    script:
    def annotationConfigJson = AnnotationReferences.annotationConfigJson(params, map_spec.species)
    def mapArgs = AnnotationReferences.mapArguments(
        pair_id,
        segmentation,
        map_spec,
        bundle_refs as List,
        task.cpus as int,
    )
    def reuseDir = AnnotationReferences.mapReuseDir(params, pair_id, segmentation)
    def reuseDirArg = reuseDir ? AnnotationReferences.shellQuote(reuseDir) : "''"
    """
    set -euo pipefail
    export PYTHONPATH="${projectDir}/../src:\${PYTHONPATH:-}"
    export CUDA_VISIBLE_DEVICES=""
    # cell_type_mapper parallelises over --n-processors worker processes
    # (task.cpus); more BLAS threads per worker would oversubscribe them.
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    export NUMBA_NUM_THREADS=1

    python - <<'PY'
import importlib.metadata
import sys

expected = "${params.annotation_ctm_version}"
found = importlib.metadata.version("cell_type_mapper")
if found != expected:
    sys.exit(
        f"cell_type_mapper {found} is installed but annotation_ctm_version is "
        f"{expected}; rebuild the environment from requirements/requirements.lock"
    )
PY

    cat > annotation_config.json <<'JSON'
${annotationConfigJson}
JSON

    # Published-output reuse (annotation_reuse_published): identical runs of
    # the published manifest are copied instead of re-mapped.
    reuse_dir=${reuseDirArg}
    if [[ -n "\${reuse_dir}" && -f "\${reuse_dir}/${AnnotationReferences.MAP_MANIFEST_FILE}" ]]; then
        set -- --reuse-from "\${reuse_dir}"
    else
        set --
    fi

    mkdir -p ${AnnotationReferences.MAP_SCRATCH_DIR}
    merxen annotate \\
        --annotation-config annotation_config.json \\
        ${mapArgs} \\
        "\$@" \\
        --out annotation_map_out
    rm -rf ${AnnotationReferences.MAP_SCRATCH_DIR}
    cp annotation_config.json annotation_map_out/annotation_config.json
    """

    stub:
    def stubManifest = AnnotationReferences.stubMapManifestJson(
        pair_id,
        segmentation,
        map_spec,
        bundle_refs as List,
    )
    def copyRefs = (bundle_refs as List).collect { ref ->
        "cp ${ref} annotation_map_out/stub_bundle_refs/"
    }.join("\n    ")
    """
    mkdir -p annotation_map_out/stub_bundle_refs
    ${copyRefs}
    cat > annotation_map_out/map_manifest.json <<'JSON'
${stubManifest}
JSON
    """
}

process CLUSTERING_SQUIDPY_ANNOTATE_RESOLVE {
    tag "${pair_id}:${segmentation}"

    // PREP never caches and re-writes byte-identical bundle refs in new work
    // directories, so, as for MAP, only content hashing keeps an unchanged
    // RESOLVE cached (-resume). resolve_spec carries the annotation config
    // and the RESOLVE rules fingerprint (AnnotationReferences.resolveSpec), so
    // a threshold, floor, trust, flag or degraded-mode change re-runs this
    // task alone: MAP is a separate task whose inputs and published-output
    // reuse key hold none of them (plan §3.1).
    cache "deep"

    publishDir { "${params.outdir}/${pair_id}/${segmentation}/annotation_resolve" }, mode: "copy", overwrite: true

    input:
    // resolve_spec: AnnotationReferences.resolveSpec(params, annotation_panel_out,
    // src) (species, panel status and size, annotation config, rules
    // fingerprint); bundle_refs: the refs MAP mapped with; alignment_files:
    // the pair's shared tissue mask and registration summary from ALIGN (M5),
    // or none.
    tuple val(pair_id),
        val(segmentation),
        val(resolve_spec),
        val(samples_json),
        path(clustering_config, stageAs: "resolve_inputs/clustering_squidpy_config.json"),
        path(prepared_dir, stageAs: "resolve_inputs/clustering_prepare_out"),
        path(panel_dir, stageAs: "resolve_inputs/annotation_panel_out"),
        path(bundle_refs, arity: "0..*", stageAs: "resolve_inputs/bundle_refs/bundle_ref_?.json"),
        path(map_dir, stageAs: "resolve_inputs/annotation_map_out"),
        path(alignment_files, arity: "0..*", stageAs: "resolve_inputs/align_out/*")

    output:
    // annotation_resolve_out/<platform>/<sid>_celltype_labels.parquet (plan
    // §4.1), <platform>/<sid>_annotation_manifest.json (§4.6),
    // <pair>_resolve_summary.json (gate levels and warnings, trust, realised
    // flag rates, coverage, resolvable share per level, compositions and the
    // pair JSD) and the annotation_config.json it resolved with: all
    // deterministic for given inputs, so a deep-cached COMPUTE_CPU (M5)
    // staging it re-runs only when a label changes. The run record (clock,
    // wall time, absolute paths) is published beside it and staged nowhere.
    tuple val(pair_id),
        val(segmentation),
        path("annotation_resolve_out"), emit: resolved
    path("annotation_resolve_run.json"), emit: run_record

    script:
    def annotationConfigJson = AnnotationReferences.resolveConfigJson(resolve_spec)
    def resolveArgs = AnnotationReferences.resolveArguments(
        resolve_spec,
        bundle_refs as List,
        alignment_files as List,
    )
    """
    set -euo pipefail
    export PYTHONPATH="${projectDir}/../src:\${PYTHONPATH:-}"
    export CUDA_VISIBLE_DEVICES=""
    # One Python process: numpy, the block bootstrap and the flag nulls run
    # single-threaded; the second CPU covers parquet and HDF5 I/O.
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    export NUMBA_NUM_THREADS=1

    cat > annotation_config.json <<'JSON'
${annotationConfigJson}
JSON

    merxen annotate-resolve \\
        --annotation-config annotation_config.json \\
        ${resolveArgs} \\
        --out annotation_resolve_out \\
        --run-record annotation_resolve_run.json
    cp annotation_config.json annotation_resolve_out/annotation_config.json
    """

    stub:
    def annotationConfigJson = AnnotationReferences.resolveConfigJson(resolve_spec)
    def stubSummary = AnnotationReferences.stubResolveSummaryJson(
        pair_id,
        segmentation,
        resolve_spec,
        bundle_refs as List,
        alignment_files as List,
    )
    def summaryFile = AnnotationReferences.resolveSummaryFile(pair_id)
    def copyRefs = (bundle_refs as List).collect { ref ->
        "cp ${ref} annotation_resolve_out/stub_bundle_refs/"
    }.join("\n    ")
    """
    mkdir -p annotation_resolve_out/stub_bundle_refs
    ${copyRefs}
    cat > annotation_resolve_out/${summaryFile} <<'JSON'
${stubSummary}
JSON
    cat > annotation_resolve_out/annotation_config.json <<'JSON'
${annotationConfigJson}
JSON
    cp resolve_inputs/annotation_map_out/${AnnotationReferences.MAP_MANIFEST_FILE} annotation_resolve_out/stub_map_manifest.json
    echo '{"stub": true}' > annotation_resolve_run.json
    """
}

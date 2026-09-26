/*
 * Reference-based cell-type annotation processes (plan §3.1, §3.2, §11.4).
 *
 * ANNOTATE_PANEL resolves one pair x segmentation's declared panels and lists
 * the reference bundles it needs; ANNOTATE_REFERENCE_PREP gets or builds one
 * bundle per unique (species, reference_id, panel_hash). Both run on the CPU
 * in the main environment and take no GPU lock. ANNOTATION_REPORT arrives in
 * M7. workflows/subworkflows/annotation_references.nf wires them, only for
 * --annotation_prepare_only runs (hook H10) and map_first runs (through
 * CLUSTERING_MAP_FIRST, hook H5 in M5); legacy runs never call them.
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

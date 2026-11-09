/* Explicit whole-panel hippocampal RCA. Existing frontal workflows are untouched. */

process HIPPOCAMPAL_REFERENCE {
    cache false
    cpus 1
    memory '8 GB'
    input:
    path resource_manifest
    val source_identity
    output:
    path 'reference', emit: resources
    script:
    """
    set -euo pipefail
    export PYTHONPATH="${projectDir}/../src:\${PYTHONPATH:-}"
    python -m merxen.annotation.hippocampal_pipeline validate \\
        --resources '${resource_manifest}' --output reference/identity.json
    """
    stub:
    """
    mkdir -p reference
    cp '${resource_manifest}' reference/resources.json
    echo '{}' > reference/identity.json
    echo '{}' > reference/mapping_identity.json
    """
}

process HIPPOCAMPAL_MAP {
    tag "${pair_id}:${segmentation}"
    cache 'deep'
    cpus 8
    memory '128 GB'
    time '7d'
    publishDir { "${params.outdir}/${pair_id}/${segmentation}/annotation_map" }, mode: 'copy'
    input:
    tuple val(pair_id), val(segmentation), val(samples_json), path(config),
        path(prepared_manifest, stageAs: 'prepared/manifest.json'),
        path(prepared_h5ads, stageAs: 'prepared/atera/*'),
        path(resources), path(mapping_identity), val(source_identity)
    output:
    tuple val(pair_id), val(segmentation), val(samples_json), path(config),
        path(prepared_manifest), path(prepared_h5ads), path('mapped')
    script:
    """
    set -euo pipefail
    export PYTHONPATH="${projectDir}/../src:\${PYTHONPATH:-}"
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    python -m merxen.annotation.hippocampal_pipeline map \\
        --resources '${resources}' --prepared prepared --output mapped
    """
    stub:
    """
    mkdir -p mapped
    cp prepared/manifest.json mapped/manifest.json
    """
}

process HIPPOCAMPAL_RESOLVE {
    tag "${pair_id}:${segmentation}"
    cache 'deep'
    cpus 1
    memory '16 GB'
    publishDir { "${params.outdir}/${pair_id}/${segmentation}/annotation_resolve" }, mode: 'copy'
    input:
    tuple val(pair_id), val(segmentation), val(samples_json), path(config),
        path(prepared_manifest), path(prepared_h5ads), path(mapped),
        path(resources), path(resolve_identity), val(source_identity)
    output:
    tuple val(pair_id), val(segmentation), val(samples_json), path(config),
        path(prepared_manifest), path(prepared_h5ads), path('resolved')
    script:
    """
    set -euo pipefail
    export PYTHONPATH="${projectDir}/../src:\${PYTHONPATH:-}"
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    python -m merxen.annotation.hippocampal_pipeline resolve \\
        --resources '${resources}' --mapped '${mapped}' --output resolved
    """
    stub:
    """
    mkdir -p resolved
    cp '${mapped}/manifest.json' resolved/manifest.json
    """
}

process HIPPOCAMPAL_COMPUTE {
    tag "${pair_id}:${segmentation}"
    cache 'deep'
    cpus 8
    memory '128 GB'
    time '7d'
    input:
    tuple val(pair_id), val(segmentation), val(samples_json), path(config),
        path(prepared_manifest, stageAs: 'prepared/manifest.json'),
        path(prepared_h5ads, stageAs: 'prepared/atera/*'), path(resolved), val(source_identity)
    output:
    tuple val(pair_id), val(segmentation), val(samples_json), path('clustering_compute_out')
    script:
    """
    set -euo pipefail
    export PYTHONPATH="${projectDir}/../src:\${PYTHONPATH:-}"
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=8
    python -m merxen.annotation.hippocampal_pipeline compute \\
        --config '${config}' --prepared prepared --resolved '${resolved}' \\
        --output clustering_compute_out
    """
    stub:
    """
    mkdir -p clustering_compute_out/atera
    echo '{}' > clustering_compute_out/stub.json
    """
}

process HIPPOCAMPAL_REPORT {
    tag "${pair_id}:${segmentation}"
    cache 'deep'
    cpus 1
    memory '8 GB'
    publishDir { "${params.outdir}/${pair_id}/${segmentation}/annotation_report" }, mode: 'copy'
    input:
    tuple val(pair_id), val(segmentation), path(resolved), path(clustered),
        path(mender_dirs), val(expect_mender), val(source_identity)
    output:
    tuple val(pair_id), val(segmentation), path('annotation_report_out')
    script:
    def menderArgs = mender_dirs.collect { item -> "--mender '${item}'" }.join(' ')
    def expectArg = expect_mender ? '--expect-mender' : ''
    """
    set -euo pipefail
    export PYTHONPATH="${projectDir}/../src:\${PYTHONPATH:-}"
    python -m merxen.annotation.hippocampal_pipeline report \\
        --resolved '${resolved}' --clustered '${clustered}' --output annotation_report_out ${menderArgs} ${expectArg}
    """
    stub:
    """
    mkdir -p annotation_report_out
    echo '<html><body>Hippocampal RCA workflow stub</body></html>' > annotation_report_out/annotation_report.html
    """
}

workflow HIPPOCAMPAL_ANNOTATION {
    take:
    prepared_ch
    main:
    def codeIdentity = AnnotationReferences.sourcesFingerprint("${projectDir}/../src", [
        'merxen/annotation', 'merxen/clustering', 'merxen/config.py',
        'merxen/table_keys.py', 'merxen/control_features.py',
        'merxen/analysis/clustering_squidpy.py', 'merxen/analysis/mapmycells_entrypoint.py',
        'merxen/analysis/mender.py', 'merxen/mender_compute.py',
    ])
    resources_ch = HIPPOCAMPAL_REFERENCE(
        channel.value(file(params.annotation_hippocampal_resources)), channel.value(codeIdentity)
    ).resources
    inputs_ch = prepared_ch.map { pairId, segmentation, samplesJson, config, prepared ->
        def manifest = prepared.resolve('manifest.json')
        def samplePaths = new groovy.json.JsonSlurperClassic().parseText(manifest.text).samples.values()
        tuple(pairId, segmentation, samplesJson, config, manifest,
            samplePaths.collect { relative -> prepared.resolve(relative.toString()) })
    }
    mapped_ch = HIPPOCAMPAL_MAP(inputs_ch.combine(resources_ch).map {
        pairId, segmentation, samplesJson, config, manifest, h5ads, resources ->
        tuple(pairId, segmentation, samplesJson, config, manifest, h5ads,
            resources.resolve('resources.json'), resources.resolve('mapping_identity.json'), codeIdentity)
    })
    resolved_ch = HIPPOCAMPAL_RESOLVE(mapped_ch.combine(resources_ch).map {
        pairId, segmentation, samplesJson, config, manifest, h5ads, mapped, resources ->
        tuple(pairId, segmentation, samplesJson, config, manifest, h5ads, mapped,
            resources.resolve('resources.json'), resources.resolve('identity.json'), codeIdentity)
    })
    resolved_ch.subscribe { pairId, segmentation, _samples, _config, _manifest, _h5ads, _resolved ->
        AnnotationRunRecord.labelledHippocampal(pairId, segmentation)
    }
    computed_ch = HIPPOCAMPAL_COMPUTE(resolved_ch.map { pairId, segmentation, samplesJson, config, manifest, h5ads, resolved ->
        tuple(pairId, segmentation, samplesJson, config, manifest, h5ads, resolved, codeIdentity)
    })
    computed_ch.subscribe { pairId, segmentation, _samples, _computed ->
        AnnotationRunRecord.computed(pairId, segmentation)
    }
    emit:
    computed = computed_ch
    labels = resolved_ch.map { pairId, segmentation, _samples, _config, _manifest, _h5ads, resolved ->
        tuple(pairId, segmentation, resolved)
    }
}

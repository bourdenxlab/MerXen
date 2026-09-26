/*
 * Annotation reference bundles (plan §3.1, §3.2, §8.5).
 *
 * ANNOTATION_REFERENCES runs ANNOTATE_PANEL once per pair x segmentation,
 * one ANNOTATE_REFERENCE_PREP task per unique (species, reference_id,
 * panel_hash) across all of them, and collects for each pair x segmentation
 * exactly the bundle refs its required_bundles.json lists. The collection
 * uses groupKey with that pair x segmentation's own n_required, so it is
 * released as soon as its last bundle is ready: pairs with different panels,
 * set-c panels (3 bundles for a same-panel human pair on proseg_hybrid) or
 * per_platform pairs (5 bundles) never wait for each other's bundles. A
 * pair x segmentation whose panels are refused (no bundle) is released at
 * once with no refs; one whose PREP failed is dropped, as any failed task
 * drops its branch under errorStrategy "ignore".
 *
 * Entry points:
 * - ANNOTATION_PREPARE_ONLY: --annotation_prepare_only (main.nf hook H10),
 *   one gene-list panel (annotation_panel_genes_path) per row x segmentation.
 * - ANNOTATION_PREPARED_REFERENCES: map_first runs, from the
 *   CLUSTERING_SQUIDPY_PREPARE outputs; CLUSTERING_MAP_FIRST calls it once
 *   MAP exists (M3-M5) and it emits MAP's input tuple (plan §3.3).
 */

include { ANNOTATE_PANEL; ANNOTATE_REFERENCE_PREP } from "../modules/annotation"

workflow ANNOTATION_BUNDLES {
    take:
    // tuple(pair_id, segmentation, annotation_panel_out): ANNOTATE_PANEL output.
    panels_ch

    main:
    required_ch = panels_ch.map { pairId, segmentation, panelDir ->
        tuple(pairId, segmentation, panelDir, AnnotationReferences.requiredBundles(panelDir))
    }

    // One PREP task per unique bundle key, whichever pair asks first; the
    // panel file of an identical panel_hash is identical in every pair.
    prep_inputs_ch = required_ch
        .flatMap { _pairId, _segmentation, panelDir, required ->
            AnnotationReferences.prepRequests(panelDir, required)
        }
        .unique { request -> request[0].key }

    prep_results_ch = ANNOTATE_REFERENCE_PREP(prep_inputs_ch)
    bundle_refs_ch = prep_results_ch.map { bundle, bundleRef -> tuple(bundle.key, bundleRef) }

    collected_ch = required_ch
        .flatMap { pairId, segmentation, _panelDir, required ->
            AnnotationReferences.bundleNeeds(pairId, segmentation, required)
        }
        .combine(bundle_refs_ch, by: 0)
        .map { bundleKey, branchKey, nRequired, bundleRef ->
            tuple(groupKey(branchKey, nRequired), bundleKey, bundleRef)
        }
        .groupTuple()
        .map { branchKey, bundleKeys, bundleRefs ->
            tuple(branchKey.toString(), bundleKeys, bundleRefs)
        }

    // Refused panels need no bundle: release those branches at once.
    without_bundles_ch = required_ch
        .filter { _pairId, _segmentation, _panelDir, required -> !required.bundles }
        .map { pairId, segmentation, _panelDir, _required ->
            tuple(AnnotationReferences.branchKey(pairId, segmentation), [], [])
        }

    bundles_ch = required_ch
        .map { pairId, segmentation, panelDir, required ->
            tuple(
                AnnotationReferences.branchKey(pairId, segmentation),
                pairId,
                segmentation,
                panelDir,
                required,
            )
        }
        .join(collected_ch.mix(without_bundles_ch))
        .map { _branchKey, pairId, segmentation, panelDir, required, bundleKeys, bundleRefs ->
            tuple(
                pairId,
                segmentation,
                panelDir,
                AnnotationReferences.orderedBundleRefs(required, bundleKeys, bundleRefs),
            )
        }

    emit:
    // tuple(bundle key, bundle_ref.json): one per PREP task.
    bundle_refs = bundle_refs_ch
    // tuple(pair_id, segmentation, annotation_panel_out, [bundle_ref.json, ...])
    // in required_bundles.json order: MAP's bundle input (plan §3.3). A PREP
    // re-run rewrites an equivalent bundle_ref.json ("reused" true), so MAP
    // should key its cache on each ref's build_hash and path, not the file.
    bundles = bundles_ch
}

workflow ANNOTATION_REFERENCES {
    take:
    // tuple(pair_id, segmentation, panel_spec, panel_inputs): ANNOTATE_PANEL input.
    panel_inputs_ch

    main:
    panels_ch = ANNOTATE_PANEL(panel_inputs_ch)
    references = ANNOTATION_BUNDLES(panels_ch)

    emit:
    panels = panels_ch
    bundle_refs = references.bundle_refs
    bundles = references.bundles
}

workflow ANNOTATION_PREPARE_ONLY {
    take:
    // tuple(pair_id, row, settings) of every samplesheet row.
    sample_rows_ch

    main:
    def preflightErrors = AnnotationPreflight.prepareOnlyErrors(params)
    if (preflightErrors) {
        error(
            "Preflight checks failed for --annotation_prepare_only:\n" +
            preflightErrors.collect { message -> " - ${message}" }.join("\n")
        )
    }
    log.info(
        "annotation_prepare_only: building the ${params.species} reference bundles " +
        "of ${params.annotation_panel_genes_path} into " +
        "${AnnotationReferences.referenceStore(params)}; no pipeline stage runs"
    )
    gene_list = file(params.annotation_panel_genes_path, checkIfExists: true)

    // The gene list is the declared panel of every row x segmentation, so
    // each branch gets the same panel and PREP builds each bundle once.
    panel_inputs_ch = sample_rows_ch.flatMap { pairId, _row, settings ->
        settings.analysis_segmentations.collect { segmentation ->
            tuple(
                pairId,
                segmentation,
                AnnotationReferences.geneListPanelSpec(gene_list.name, settings.active_platforms),
                [gene_list],
            )
        }
    }
    references = ANNOTATION_REFERENCES(panel_inputs_ch)
    references.bundle_refs.subscribe { bundleKey, bundleRef ->
        log.info("annotation_prepare_only: ${bundleKey} -> ${bundleRef}")
    }

    emit:
    bundle_refs = references.bundle_refs
    bundles = references.bundles
}

workflow ANNOTATION_PREPARED_REFERENCES {
    take:
    // tuple(pair_id, segmentation, samples_json, clustering_squidpy_config.json,
    // clustering_prepare_out): the output of CLUSTERING_SQUIDPY_PREPARE.
    prepared_ch

    main:
    // Set c is computed inside the pair's shared tissue mask when its
    // alignment outputs are published, else over the whole section (plan §3.2).
    panel_inputs_ch = prepared_ch.map { pairId, segmentation, _samplesJson, clusteringConfig, preparedDir ->
        def maskFiles = AnnotationReferences.publishedSharedTissueMask(params, pairId)
            .collect { path -> file(path.toString()) }
        tuple(
            pairId,
            segmentation,
            AnnotationReferences.preparedPanelSpec(
                clusteringConfig.name,
                preparedDir.name,
                maskFiles.collect { path -> path.name },
            ),
            [clusteringConfig, preparedDir] + maskFiles,
        )
    }
    references = ANNOTATION_REFERENCES(panel_inputs_ch)

    map_inputs_ch = prepared_ch
        .map { pairId, segmentation, samplesJson, clusteringConfig, preparedDir ->
            tuple(
                AnnotationReferences.branchKey(pairId, segmentation),
                pairId,
                segmentation,
                samplesJson,
                clusteringConfig,
                preparedDir,
            )
        }
        .join(
            references.bundles.map { pairId, segmentation, panelDir, bundleRefs ->
                tuple(AnnotationReferences.branchKey(pairId, segmentation), panelDir, bundleRefs)
            }
        )
        .map { _branchKey, pairId, segmentation, samplesJson, clusteringConfig, preparedDir, panelDir, bundleRefs ->
            tuple(pairId, segmentation, samplesJson, clusteringConfig, preparedDir, panelDir, bundleRefs)
        }

    emit:
    bundle_refs = references.bundle_refs
    // tuple(pair_id, segmentation, samples_json, clustering_squidpy_config.json,
    // clustering_prepare_out, annotation_panel_out, [bundle_ref.json, ...]).
    map_inputs = map_inputs_ch
}

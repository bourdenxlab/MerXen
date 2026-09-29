/*
 * Map-first clustering of one pair x segmentation (plan §3.1, §3.3-§3.5).
 *
 * CLUSTERING_ANNOTATE_MAP (M3) takes the CLUSTERING_SQUIDPY_PREPARE outputs
 * through ANNOTATION_PREPARED_REFERENCES (subworkflows/annotation_references.nf,
 * M2): ANNOTATE_PANEL, one ANNOTATE_REFERENCE_PREP per unique bundle and the
 * per pair x segmentation collection, whose groupKey has that pair x
 * segmentation's own n_required. CLUSTERING_SQUIDPY_ANNOTATE_MAP therefore
 * starts as soon as the last bundle its required_bundles.json lists is
 * ready: pairs with different panels, set c or per_platform panels never
 * wait for each other, a refused panel is mapped at once with no bundle
 * (MAP records the refusal), and a pair x segmentation whose PREP failed is
 * dropped, as any failed task drops its branch under errorStrategy "ignore".
 *
 * CLUSTERING_ANNOTATE (M4) runs CLUSTERING_SQUIDPY_ANNOTATE_RESOLVE on each
 * pair x segmentation once its own MAP has finished (plan §3.4). RESOLVE is
 * a separate task from MAP: a threshold, floor, trust, flag or degraded-mode
 * change re-runs RESOLVE alone (minutes) under -resume, and MAP, whose
 * inputs and published-output reuse key hold none of them, stays cached.
 *
 * CLUSTERING_MAP_FIRST (M5) = CLUSTERING_ANNOTATE -> CLUSTERING_SQUIDPY_COMPUTE_CPU
 * and emits the input shape of CLUSTERING_SQUIDPY_FINALIZE, so main.nf's
 * hook H5 feeds the same FINALIZE, the same MENDER barrier and the same
 * downstream stages in both clustering modes. COMPUTE_CPU hashes the
 * content of its staged inputs (cache "deep"): it re-runs when a label
 * table, the prepared H5ADs, the clustering config, the run's table-key
 * suffix or MENDER policy, or the hierarchy code changes, and stays cached
 * when RESOLVE re-runs with byte-identical outputs. Its only caller is hook
 * H5, in map_first runs.
 */

include { ANNOTATION_PREPARED_REFERENCES } from "./annotation_references"
include { CLUSTERING_SQUIDPY_ANNOTATE_MAP; CLUSTERING_SQUIDPY_ANNOTATE_RESOLVE } from "../modules/annotation"
include { CLUSTERING_SQUIDPY_COMPUTE_CPU } from "../modules/clustering_squidpy"

workflow CLUSTERING_ANNOTATE_MAP {
    take:
    // tuple(pair_id, segmentation, samples_json, clustering_squidpy_config.json,
    // clustering_prepare_out, alignment_files): the output of
    // CLUSTERING_SQUIDPY_PREPARE and the pair's ALIGN files ([] without).
    prepared_ch

    main:
    references = ANNOTATION_PREPARED_REFERENCES(prepared_ch)

    // The map_inputs tuple is released only after the M2 required-bundle
    // join, i.e. once every bundle of the pair x segmentation is ready.
    map_inputs_ch = references.map_inputs.map { pairId, segmentation, samplesJson, clusteringConfig, preparedDir, panelDir, bundleRefs ->
        tuple(
            pairId,
            segmentation,
            AnnotationReferences.mapSpec(panelDir),
            samplesJson,
            clusteringConfig,
            preparedDir,
            panelDir,
            bundleRefs,
        )
    }
    map_out_ch = CLUSTERING_SQUIDPY_ANNOTATE_MAP(map_inputs_ch)

    // RESOLVE (M4) reads the prepared counts, the panel, the bundles and the
    // MAP outputs: join them back by pair x segmentation.
    mapped_ch = map_inputs_ch
        .map { pairId, segmentation, _mapSpec, samplesJson, clusteringConfig, preparedDir, panelDir, bundleRefs ->
            tuple(
                AnnotationReferences.branchKey(pairId, segmentation),
                pairId,
                segmentation,
                samplesJson,
                clusteringConfig,
                preparedDir,
                panelDir,
                bundleRefs,
            )
        }
        .join(
            map_out_ch.map { pairId, segmentation, mapDir ->
                tuple(AnnotationReferences.branchKey(pairId, segmentation), mapDir)
            }
        )
        .map { _branchKey, pairId, segmentation, samplesJson, clusteringConfig, preparedDir, panelDir, bundleRefs, mapDir ->
            tuple(pairId, segmentation, samplesJson, clusteringConfig, preparedDir, panelDir, bundleRefs, mapDir)
        }

    emit:
    // tuple(bundle key, bundle_ref.json): one per PREP task.
    bundle_refs = references.bundle_refs
    // tuple(pair_id, segmentation, samples_json, clustering_squidpy_config.json,
    // clustering_prepare_out, annotation_panel_out, [bundle_ref.json, ...],
    // annotation_map_out): RESOLVE's inputs (M4).
    maps = mapped_ch
}

workflow CLUSTERING_ANNOTATE {
    take:
    // tuple(pair_id, segmentation, samples_json, clustering_squidpy_config.json,
    // clustering_prepare_out, alignment_files): see CLUSTERING_ANNOTATE_MAP.
    prepared_ch

    main:
    mapped = CLUSTERING_ANNOTATE_MAP(prepared_ch)

    // A maps tuple exists only once its branch's MAP has finished, so each
    // RESOLVE waits for its own MAP and no other. resolveSpec carries the
    // RESOLVE config and the fingerprint of this checkout's RESOLVE rules
    // (task inputs, so -resume sees them). The pair's ALIGN files (shared
    // tissue mask and registration summary) come from the take channel,
    // i.e. from ALIGN's output channel: RESOLVE never looks one up among
    // published files.
    alignment_by_branch_ch = prepared_ch.map { pairId, segmentation, _samplesJson, _clusteringConfig, _preparedDir, alignmentFiles ->
        tuple(AnnotationReferences.branchKey(pairId, segmentation), alignmentFiles)
    }
    resolve_inputs_ch = mapped.maps
        .map { pairId, segmentation, samplesJson, clusteringConfig, preparedDir, panelDir, bundleRefs, mapDir ->
            tuple(
                AnnotationReferences.branchKey(pairId, segmentation),
                pairId,
                segmentation,
                samplesJson,
                clusteringConfig,
                preparedDir,
                panelDir,
                bundleRefs,
                mapDir,
            )
        }
        .join(alignment_by_branch_ch)
        .map { _branchKey, pairId, segmentation, samplesJson, clusteringConfig, preparedDir, panelDir, bundleRefs, mapDir, alignmentFiles ->
            tuple(
                pairId,
                segmentation,
                AnnotationReferences.resolveSpec(params, panelDir, "${projectDir}/../src"),
                samplesJson,
                clusteringConfig,
                preparedDir,
                panelDir,
                bundleRefs,
                mapDir,
                alignmentFiles,
            )
        }
    CLUSTERING_SQUIDPY_ANNOTATE_RESOLVE(resolve_inputs_ch)
    // The deterministic annotation_resolve_out only; the run record stays
    // out of every downstream input.
    resolve_out_ch = CLUSTERING_SQUIDPY_ANNOTATE_RESOLVE.out.resolved

    labels_ch = mapped.maps
        .map { pairId, segmentation, samplesJson, clusteringConfig, preparedDir, panelDir, bundleRefs, mapDir ->
            tuple(
                AnnotationReferences.branchKey(pairId, segmentation),
                pairId,
                segmentation,
                samplesJson,
                clusteringConfig,
                preparedDir,
                panelDir,
                bundleRefs,
                mapDir,
            )
        }
        .join(
            resolve_out_ch.map { pairId, segmentation, resolveDir ->
                tuple(AnnotationReferences.branchKey(pairId, segmentation), resolveDir)
            }
        )
        .map { _branchKey, pairId, segmentation, samplesJson, clusteringConfig, preparedDir, panelDir, bundleRefs, mapDir, resolveDir ->
            tuple(pairId, segmentation, samplesJson, clusteringConfig, preparedDir, panelDir, bundleRefs, mapDir, resolveDir)
        }

    emit:
    // tuple(bundle key, bundle_ref.json): one per PREP task.
    bundle_refs = mapped.bundle_refs
    // CLUSTERING_ANNOTATE_MAP's maps (RESOLVE's inputs).
    maps = mapped.maps
    // tuple(pair_id, segmentation, samples_json, clustering_squidpy_config.json,
    // clustering_prepare_out, annotation_panel_out, [bundle_ref.json, ...],
    // annotation_map_out, annotation_resolve_out): the label tables with what
    // COMPUTE_CPU (M5) and ANNOTATION_REPORT (M7) read.
    labels = labels_ch
}

workflow CLUSTERING_MAP_FIRST {
    take:
    // tuple(pair_id, segmentation, samples_json, clustering_squidpy_config.json,
    // clustering_prepare_out): the output of CLUSTERING_SQUIDPY_PREPARE.
    prepared_ch
    // tuple(pair_id, alignment_files): one per samplesheet pair, the ALIGN
    // files AnnotationReferences.alignmentFiles returns ([] without an
    // alignment).
    alignment_ch

    main:
    annotated = CLUSTERING_ANNOTATE(prepared_ch.combine(alignment_ch, by: 0))

    // COMPUTE_CPU stages RESOLVE's deterministic label tables, manifests and
    // pair summary as files and hashes them by content (cache "deep"), so it
    // re-runs only when a label (or its own inputs, run settings or code)
    // changes (M4 review).
    compute_inputs_ch = annotated.labels.map { pairId, segmentation, samplesJson, clusteringConfig, preparedDir, _panelDir, _bundleRefs, _mapDir, resolveDir ->
        tuple(
            pairId,
            segmentation,
            AnnotationReferences.computeSpec(params, "${projectDir}/../src"),
            samplesJson,
            clusteringConfig,
            preparedDir,
            AnnotationReferences.computeLabelFiles(resolveDir),
        )
    }
    computed_ch = CLUSTERING_SQUIDPY_COMPUTE_CPU(compute_inputs_ch)

    // The end-of-run summary (hook H6): which branches got label tables and
    // a hierarchy, their panel status, trust and cross-platform scope.
    annotated.labels.subscribe { pairId, segmentation, _samplesJson, _clusteringConfig, _preparedDir, panelDir, _bundleRefs, _mapDir, resolveDir ->
        AnnotationRunRecord.labelled(pairId, segmentation, panelDir, resolveDir)
    }
    computed_ch.subscribe { pairId, segmentation, _samplesJson, _computedDir ->
        AnnotationRunRecord.computed(pairId, segmentation)
    }

    emit:
    // tuple(pair_id, segmentation, samples_json, clustering_compute_out):
    // CLUSTERING_SQUIDPY_FINALIZE's input, as CLUSTERING_SQUIDPY_COMPUTE's.
    computed = computed_ch
    // CLUSTERING_ANNOTATE's maps (RESOLVE's inputs) and labels
    // (ANNOTATION_REPORT, M7).
    maps = annotated.maps
    labels = annotated.labels
    // tuple(bundle key, bundle_ref.json): one per PREP task.
    bundle_refs = annotated.bundle_refs
}

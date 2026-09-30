/*
 * The annotation QC report of each map_first pair x segmentation (plan §3.6,
 * §9; M7).
 *
 * ANNOTATION_REPORTING runs ANNOTATION_REPORT (modules/annotation.nf) once
 * per pair x segmentation that CLUSTERING_MAP_FIRST labelled and FINALIZE
 * finished, after the stages whose outputs the report reads:
 *
 * - FINALIZE (the clustered H5ADs: coordinates, counts, the hierarchy);
 * - cortical depth, when it runs after clustering for the segmentation: the
 *   report waits for the pair's depth output of every active platform
 *   (groupKey of the platform count), since depth runs once per platform
 *   over all of the pair's analysis segmentations;
 * - MENDER, when it runs for the segmentation: the report waits for the
 *   MENDER output of every sample of the branch (groupKey), whose
 *   mender_manifest.json it digests (item 12).
 *
 * Branches where neither runs are released as soon as FINALIZE finishes;
 * no branch waits for another. A branch whose depth or MENDER task failed
 * is dropped under errorStrategy "ignore", like its MENDER, and listed by
 * the end-of-run summary (hook H6) as a report not built; so is one whose
 * report task failed. A report item that fails inside the report is
 * recorded there (status failed) and listed too.
 *
 * Its only caller is main.nf's hook H11, in map_first runs with
 * annotation_report_enabled; legacy runs never reach it.
 */

include { ANNOTATION_REPORT } from "../modules/annotation"

workflow ANNOTATION_REPORTING {
    take:
    // CLUSTERING_MAP_FIRST's labels: tuple(pair_id, segmentation, samples_json,
    // clustering_squidpy_config.json, clustering_prepare_out,
    // annotation_panel_out, [bundle_ref.json, ...], annotation_map_out,
    // annotation_resolve_out).
    labels_ch
    // CLUSTERING_MAP_FIRST's alignment: tuple(pair_id, alignment_files), one
    // per pair ([] without an alignment).
    alignment_ch
    // CLUSTERING_SQUIDPY_FINALIZE: tuple(pair_id, segmentation, samples_json,
    // clustering_squidpy_out).
    clustered_ch
    // tuple(pair_id, platform, compute_cortical_depth_out), one per
    // COMPUTE_CORTICAL_DEPTH task.
    depth_ch
    // tuple(pair_id, segmentation, platform, mender_out/<platform>), one per
    // MENDER_FINALIZE task.
    mender_ch
    // tuple(pair_id, AnnotationReport.pairSpec(settings, params)), one per
    // samplesheet row.
    pair_specs_ch

    main:
    // One tuple per labelled branch the run reports, once FINALIZE has
    // finished it: tuple(branch key, pair_id, segmentation, samples_json,
    // annotation_panel_out, annotation_map_out, annotation_resolve_out,
    // branch spec, clustering_squidpy_out).
    branches_ch = labels_ch
        .map { pairId, segmentation, samplesJson, _clusteringConfig, _preparedDir, panelDir, _bundleRefs, mapDir, resolveDir ->
            tuple(pairId, segmentation, samplesJson, panelDir, mapDir, resolveDir)
        }
        .combine(pair_specs_ch, by: 0)
        .map { pairId, segmentation, samplesJson, panelDir, mapDir, resolveDir, pairSpec ->
            tuple(
                AnnotationReferences.branchKey(pairId, segmentation),
                pairId,
                segmentation,
                samplesJson,
                panelDir,
                mapDir,
                resolveDir,
                AnnotationReport.branchSpec(pairSpec, segmentation, samplesJson),
            )
        }
        .filter { _branchKey, _pairId, _segmentation, _samplesJson, _panelDir, _mapDir, _resolveDir, spec ->
            spec.enabled
        }
        .join(
            clustered_ch.map { pairId, segmentation, _samplesJson, clusteringDir ->
                tuple(AnnotationReferences.branchKey(pairId, segmentation), clusteringDir)
            }
        )
    branches_ch.subscribe { _branchKey, pairId, segmentation, _samplesJson, _panelDir, _mapDir, _resolveDir, _spec, _clusteringDir ->
        AnnotationRunRecord.reportExpected(pairId, segmentation)
    }

    // Cortical depth: the pair's output of every depth platform, per branch
    // that waits for it; tuple(branch key, platforms, compute_cortical_depth_out
    // dirs), sorted by platform. Other branches get empty lists at once.
    depth_waits_ch = branches_ch
        .filter { _branchKey, _pairId, _segmentation, _samplesJson, _panelDir, _mapDir, _resolveDir, spec, _clusteringDir ->
            spec.depth_platforms
        }
        .map { branchKey, pairId, _segmentation, _samplesJson, _panelDir, _mapDir, _resolveDir, spec, _clusteringDir ->
            tuple(pairId, branchKey, spec.depth_platforms)
        }
    depth_groups_ch = depth_ch
        .combine(depth_waits_ch, by: 0)
        .filter { _pairId, platform, _depthDir, _branchKey, platforms ->
            platform in platforms
        }
        .map { _pairId, platform, depthDir, branchKey, platforms ->
            tuple(groupKey(branchKey, platforms.size()), platform, depthDir)
        }
        .groupTuple()
        .map { branchKey, platforms, depthDirs ->
            def sorted = AnnotationReport.sortedByPlatform(platforms, depthDirs)
            tuple(branchKey.getGroupTarget(), sorted[0], sorted[1])
        }
    depth_inputs_ch = depth_groups_ch.mix(
        branches_ch
            .filter { _branchKey, _pairId, _segmentation, _samplesJson, _panelDir, _mapDir, _resolveDir, spec, _clusteringDir ->
                !spec.depth_platforms
            }
            .map { branchKey, _pairId, _segmentation, _samplesJson, _panelDir, _mapDir, _resolveDir, _spec, _clusteringDir ->
                tuple(branchKey, [], [])
            }
    )

    // MENDER: the branch's output of every sample, per branch that waits for
    // it; tuple(branch key, platforms, mender_out/<platform> dirs).
    mender_waits_ch = branches_ch
        .filter { _branchKey, _pairId, _segmentation, _samplesJson, _panelDir, _mapDir, _resolveDir, spec, _clusteringDir ->
            spec.mender_platforms
        }
        .map { branchKey, _pairId, _segmentation, _samplesJson, _panelDir, _mapDir, _resolveDir, spec, _clusteringDir ->
            tuple(branchKey, spec.mender_platforms)
        }
    mender_groups_ch = mender_ch
        .map { pairId, segmentation, platform, menderDir ->
            tuple(AnnotationReferences.branchKey(pairId, segmentation), platform, menderDir)
        }
        .combine(mender_waits_ch, by: 0)
        .filter { _branchKey, platform, _menderDir, platforms ->
            platform in platforms
        }
        .map { branchKey, platform, menderDir, platforms ->
            tuple(groupKey(branchKey, platforms.size()), platform, menderDir)
        }
        .groupTuple()
        .map { branchKey, platforms, menderDirs ->
            def sorted = AnnotationReport.sortedByPlatform(platforms, menderDirs)
            tuple(branchKey.getGroupTarget(), sorted[0], sorted[1])
        }
    mender_inputs_ch = mender_groups_ch.mix(
        branches_ch
            .filter { _branchKey, _pairId, _segmentation, _samplesJson, _panelDir, _mapDir, _resolveDir, spec, _clusteringDir ->
                !spec.mender_platforms
            }
            .map { branchKey, _pairId, _segmentation, _samplesJson, _panelDir, _mapDir, _resolveDir, _spec, _clusteringDir ->
                tuple(branchKey, [], [])
            }
    )

    // The pair's ALIGN files (the shared tissue mask), then the depth and
    // MENDER outputs: a branch is released once all three have arrived.
    report_inputs_ch = branches_ch
        .map { branchKey, pairId, segmentation, samplesJson, panelDir, mapDir, resolveDir, _spec, clusteringDir ->
            tuple(pairId, branchKey, segmentation, samplesJson, panelDir, mapDir, resolveDir, clusteringDir)
        }
        .combine(alignment_ch, by: 0)
        .map { pairId, branchKey, segmentation, samplesJson, panelDir, mapDir, resolveDir, clusteringDir, alignmentFiles ->
            tuple(branchKey, pairId, segmentation, samplesJson, panelDir, mapDir, resolveDir, clusteringDir, alignmentFiles)
        }
        .join(depth_inputs_ch)
        .join(mender_inputs_ch)
        .map { _branchKey, pairId, segmentation, samplesJson, panelDir, mapDir, resolveDir, clusteringDir, alignmentFiles, depthPlatforms, depthDirs, menderPlatforms, menderDirs ->
            tuple(
                pairId,
                segmentation,
                AnnotationReport.reportSpec(params, "${projectDir}/../src"),
                samplesJson,
                resolveDir,
                mapDir,
                panelDir,
                clusteringDir,
                depthPlatforms,
                depthDirs,
                menderPlatforms,
                menderDirs,
                alignmentFiles,
            )
        }
    ANNOTATION_REPORT(report_inputs_ch)
    ANNOTATION_REPORT.out.subscribe { pairId, segmentation, reportDir ->
        AnnotationRunRecord.reported(pairId, segmentation, reportDir)
    }

    emit:
    // tuple(pair_id, segmentation, annotation_report_out).
    ANNOTATION_REPORT.out
}

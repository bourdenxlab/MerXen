import groovy.json.JsonOutput
import groovy.json.JsonSlurperClassic

/*
 * The annotation QC report of a map_first pair x segmentation (plan §3.6,
 * §9; M7).
 *
 * ANNOTATION_REPORT (modules/annotation.nf) runs `merxen annotation-report`
 * on one pair x segmentation (a mouse section x segmentation) once its
 * label tables (RESOLVE), clustered H5ADs (FINALIZE), the pair's cortical
 * depth (when it runs for the segmentation) and the branch's MENDER outputs
 * (when MENDER runs for it) exist. ANNOTATION_REPORTING
 * (subworkflows/annotation_report.nf), called by main.nf's hook H11 in
 * map_first runs only, does the waiting; this class holds what it and the
 * process share: which depth and MENDER outputs a branch waits for, the
 * staged-input layout, the command arguments and the code fingerprint that
 * lets -resume re-run a report whose code changed. A legacy run never calls
 * any of it.
 */
class AnnotationReport {

    // ANNOTATION_REPORT: staged-input directory and output directory
    // (published under <outdir>/<pair>/<seg>/annotation_report/, plan §3.6).
    static final String PUBLISH_DIR = "annotation_report"
    static final String INPUT_DIR = "report_inputs"
    static final String OUTPUT_DIR = "annotation_report_out"
    // The report's run record (wall time, version, item statuses; the only
    // non-deterministic file, merxen.annotation.report.RUN_FILENAME).
    static final String RUN_FILE = "report_run.json"
    // Staged names: RESOLVE, MAP and PANEL outputs, FINALIZE's
    // clustering_squidpy_out, the pair's ALIGN files, and one directory per
    // cortical-depth output (compute_cortical_depth_out) and MENDER output
    // (mender_out/<platform>), numbered in the order of their platforms.
    static final String RESOLVE_INPUT = "annotation_resolve_out"
    static final String MAP_INPUT = "annotation_map_out"
    static final String PANEL_INPUT = "annotation_panel_out"
    static final String CLUSTERING_INPUT = "clustering_squidpy_out"
    static final String ALIGN_INPUT = "align_out"
    static final String MENDER_MANIFEST_FILE = "mender_manifest.json"

    // What the report fingerprint covers, relative to this checkout's src/:
    // the report and everything it reads through (the annotation package with
    // its packaged tables, the clustering package, the command). The task
    // hash cannot see Python code, so the fingerprint travels in report_spec
    // and a report change re-runs the report (about a minute) under -resume;
    // no other task stages the report's output.
    static final List<String> REPORT_SOURCES = [
        "merxen/annotation",
        "merxen/assets/annotation",
        "merxen/clustering",
        "merxen/cli/run_annotation_report.py",
    ].asImmutable()

    /**
     * Return whether the run builds annotation reports (annotation_report_enabled).
     *
     * @param params Pipeline params.
     * @return The param, true when unset.
     */
    static boolean enabled(Map params) {
        return AnnotationReferences.isTrue(params?.get("annotation_report_enabled"), true)
    }

    /**
     * Return what a pair's reports wait for, from its row settings.
     *
     * Cortical depth after clustering runs once per active platform over
     * analysis_segmentations (+ distance_from_object_segmentations when that
     * stage runs), as main.nf's compute_cortical_depth_after_clustering_gate_ch
     * builds it; MENDER from this run's clustering runs once per sample of
     * each mender_segmentations branch (mender_current_clustering_gate_ch).
     *
     * @param settings rowSampleSettings of the pair's samplesheet row.
     * @param params Pipeline params.
     * @return [enabled, platforms (sorted), depth_segmentations,
     *     mender_segmentations].
     */
    static Map pairSpec(Map settings, Map params) {
        def clustering = AnnotationReferences.isTrue(settings?.run_clustering_squidpy)
        def depthSegmentations = []
        if (clustering && AnnotationReferences.isTrue(settings?.run_compute_cortical_depth)) {
            depthSegmentations = texts(settings.analysis_segmentations)
            if (AnnotationReferences.isTrue(settings?.run_distance_from_object)) {
                depthSegmentations = (depthSegmentations + texts(settings.distance_from_object_segmentations)).unique()
            }
        }
        def menderSegmentations = (clustering && AnnotationReferences.isTrue(settings?.run_mender))
            ? texts(settings.mender_segmentations)
            : []
        return [
            enabled: enabled(params) && clustering,
            platforms: texts(settings?.active_platforms).sort(),
            depth_segmentations: depthSegmentations,
            mender_segmentations: menderSegmentations,
        ]
    }

    /**
     * Return what one pair x segmentation's report waits for.
     *
     * @param pairSpec pairSpec of the pair.
     * @param segmentation Segmentation.
     * @param samplesJson The branch's samples JSON (MENDER runs per sample).
     * @return [enabled, depth_platforms, mender_platforms]: the platforms
     *     whose cortical-depth and MENDER outputs the report waits for
     *     (sorted; empty when that stage does not run for the branch).
     */
    static Map branchSpec(Map pairSpec, Object segmentation, Object samplesJson) {
        def seg = segmentation.toString()
        def samplePlatforms = samples(samplesJson).collect { sample -> sample.platform.toString() }.unique().sort()
        return [
            enabled: pairSpec?.enabled == true,
            depth_platforms: seg in (pairSpec?.depth_segmentations ?: []) ? (pairSpec.platforms as List) : [],
            mender_platforms: seg in (pairSpec?.mender_segmentations ?: []) ? samplePlatforms : [],
        ]
    }

    /**
     * Return what ANNOTATION_REPORT needs to know before it runs.
     *
     * @param params Pipeline params.
     * @param sourceRoot This checkout's src/ (${projectDir}/../src).
     * @return [species, report_fingerprint]; task inputs, so -resume sees them.
     */
    static Map reportSpec(Map params, Object sourceRoot) {
        return [
            species: AnnotationDefaults.normalizeSpecies(params?.get("species")),
            report_fingerprint: AnnotationReferences.sourcesFingerprint(sourceRoot, REPORT_SOURCES),
        ]
    }

    /**
     * Sort platform-keyed outputs by platform.
     *
     * @param platforms Platforms, in arrival order.
     * @param outputs Their outputs, in the same order.
     * @return [platforms, outputs], both sorted by platform.
     */
    static List sortedByPlatform(List platforms, List outputs) {
        def pairs = [platforms, outputs].transpose().sort { item -> item[0].toString() }
        return [
            pairs.collect { item -> item[0].toString() },
            pairs.collect { item -> item[1] },
        ]
    }

    /**
     * Return the merxen annotation-report arguments of one ANNOTATION_REPORT task.
     *
     * Every input is staged under report_inputs/: RESOLVE, MAP and PANEL
     * outputs, FINALIZE's clustering_squidpy_out (one clustered H5AD per
     * sample), one compute_cortical_depth_out per depth platform (the report
     * reads <dir>/<segmentation>/*_cells_with_cortical_depth.parquet), one
     * mender_out/<platform> per MENDER platform and the pair's ALIGN files.
     * The report writes annotation_report_out/, outside every staged input.
     *
     * @param spec reportSpec result.
     * @param pairId Pair id (mouse: the section id RESOLVE used).
     * @param segmentation Segmentation.
     * @param samplesJson The branch's samples JSON (sample_id, platform).
     * @param depthPlatforms Platforms of the staged depth outputs, in order.
     * @param depthDirs The staged depth outputs.
     * @param menderPlatforms Platforms of the staged MENDER outputs, in order.
     * @param menderDirs The staged MENDER outputs.
     * @param alignmentFiles The staged ALIGN files (none without an alignment).
     * @return The quoted arguments.
     */
    static String reportArguments(
        Map spec,
        Object pairId,
        Object segmentation,
        Object samplesJson,
        List depthPlatforms,
        List depthDirs,
        List menderPlatforms,
        List menderDirs,
        List alignmentFiles
    ) {
        def depthByPlatform = platformMap(depthPlatforms, depthDirs)
        def menderByPlatform = platformMap(menderPlatforms, menderDirs)
        def args = [
            "--pair", pairId.toString(),
            "--segmentation", segmentation.toString(),
            "--species", spec.species.toString(),
            "--resolve-dir", staged(RESOLVE_INPUT),
            "--map-dir", staged(MAP_INPUT),
            "--panel-dir", staged(PANEL_INPUT),
        ]
        samples(samplesJson).each { sample ->
            def sampleId = sample.sample_id.toString()
            def platform = sample.platform.toString()
            args += [
                "--clustered-h5ad",
                "${sampleId}=${staged(CLUSTERING_INPUT)}/${platform.toLowerCase()}/${sampleId}_clustered.h5ad".toString(),
            ]
            if (depthByPlatform.containsKey(platform)) {
                args += ["--cortical-depth-dir", "${sampleId}=${depthByPlatform[platform]}".toString()]
            }
            if (menderByPlatform.containsKey(platform)) {
                args += [
                    "--mender-manifest",
                    "${sampleId}=${menderByPlatform[platform]}/${MENDER_MANIFEST_FILE}".toString(),
                ]
            }
        }
        if (!depthByPlatform) {
            args += ["--no-cortical-depth"]
        }
        if (!menderByPlatform) {
            args += ["--no-mender"]
        }
        args += (alignmentFiles ? ["--alignment-dir", staged(ALIGN_INPUT)] : ["--no-alignment"])
        args += ["--out", OUTPUT_DIR]
        return args.collect { arg -> AnnotationReferences.shellQuote(arg) }.join(" ")
    }

    /**
     * Return the stub record of an ANNOTATION_REPORT task (for -stub-run).
     *
     * @param pairId Pair id.
     * @param segmentation Segmentation.
     * @param spec reportSpec result.
     * @param arguments reportArguments result.
     * @return The JSON text.
     */
    static String stubReportJson(Object pairId, Object segmentation, Map spec, String arguments) {
        return JsonOutput.prettyPrint(JsonOutput.toJson([
            stub: true,
            step: "annotation_report",
            pair_id: pairId.toString(),
            segmentation: segmentation.toString(),
            species: spec.species,
            report_fingerprint: spec.report_fingerprint,
            arguments: arguments,
        ]))
    }

    private static Map platformMap(List platforms, List outputs) {
        def items = (platforms ?: []) as List
        def paths = (outputs ?: []) as List
        if (items.size() != paths.size()) {
            throw new IllegalArgumentException(
                "ANNOTATION_REPORT: ${items.size()} platforms for ${paths.size()} staged outputs"
            )
        }
        def mapped = [:] as LinkedHashMap<String, String>
        [items, paths].transpose().each { item ->
            mapped[item[0].toString()] = item[1].toString()
        }
        return mapped
    }

    private static List<Map> samples(Object samplesJson) {
        def parsed = new JsonSlurperClassic().parseText(samplesJson.toString())
        return (parsed instanceof List ? parsed : []) as List<Map>
    }

    private static List<String> texts(Object values) {
        return ((values ?: []) as List).collect { value -> value.toString() }
    }

    private static String staged(String name) {
        return "${INPUT_DIR}/${name}".toString()
    }
}

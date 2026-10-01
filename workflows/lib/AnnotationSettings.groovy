import groovy.json.JsonOutput
import groovy.json.JsonSlurperClassic

/*
 * Per-row annotation settings for main.nf (hooks H3, H5, H6; plan §2.4, §3.1).
 *
 * rowSampleSettings merges forRow into each row's settings map. A legacy row
 * gets no key at all: VALIDATE_ANALYSIS_LAYER takes the settings map as a
 * `val` input, so any new key would change its task hash and invalidate
 * legacy -resume. Readers therefore use the accessors below, which return
 * the legacy values when a key is absent.
 */
class AnnotationSettings {

    // Keys forRow adds to the settings of a map_first row.
    static final List<String> KEYS = [
        "clustering_squidpy_mode",
        "clustering_squidpy_table_key_suffix",
        "annotation_anatomical_region",
        "annotation_mouse_section_regions",
        "annotation_mode_mapmycells_stage",
        "mender_unassigned_state_policy",
    ].asImmutable()

    static final String LEGACY_MENDER_UNASSIGNED_STATE_POLICY = "state"
    static final String MAP_FIRST_MENDER_UNASSIGNED_STATE_POLICY = "exclude_from_features"

    // Optional samplesheet columns a map_first row carries into samples_json,
    // per species (merxen.annotation.samplesheet_columns; plan §3.3, §3.7):
    // ClusteringSquidpySampleConfig fields, None inheriting the global param.
    static final Map<String, String> ROW_COLUMNS = [
        human: "anatomical_region",
        mouse: "mouse_section_regions",
    ].asImmutable()

    // completionSummary lines: runInfo key -> label.
    static final Map<String, String> SUMMARY_ITEMS = [
        failed_annotations: "failed annotations (no label tables; see the failed tasks above)",
        failed_hierarchies: "failed hierarchies (label tables, no COMPUTE_CPU output)",
        refused_panels: "refused panels",
        broad_only_panels: "broad-only panels",
        provisional_panels: "provisional panels",
        restricted_dataset_gates: "broad-only or failed dataset gates",
        cross_platform_restricted: "restricted cross-platform statistics",
        mender_skipped: "MENDER skipped (no assigned cell state)",
        failed_reports: "annotation reports not built (see the failed tasks above)",
        failed_report_items: "annotation report items failed (see the report)",
    ].asImmutable()

    /**
     * Return the annotation settings of one samplesheet row.
     *
     * The optional samplesheet columns anatomical_region (human) and
     * mouse_section_regions (mouse) override the global params when set.
     *
     * @param row Samplesheet row.
     * @param params Pipeline params.
     * @param species Run species (name or alias).
     * @return An empty map for a legacy row, else the KEYS of a map_first row.
     * @throws IllegalArgumentException On an unknown mode or species, or an
     *     invalid table-key suffix.
     */
    static Map forRow(Map row, Map params, Object species) {
        def speciesName = AnnotationDefaults.normalizeSpecies(species)
        // The row's own clustering_squidpy_mode column wins (§20 D15).
        def rowMode = rowValue(row, AnnotationDefaults.ROW_MODE_COLUMN)
        def mode = AnnotationDefaults.resolveRowMode(rowMode, params, speciesName)
        params = AnnotationDefaults.withRowMode(rowMode, params)
        if (mode == AnnotationDefaults.LEGACY) {
            return [:]
        }
        def human = speciesName == "human"
        def region = human
            ? regionToken(rowValue(row, "anatomical_region") ?: params?.get("annotation_human_region"))
            : null
        def sectionRegions = human
            ? null
            : (
                rowValue(row, "mouse_section_regions") ?:
                stringOrNull(params?.get("annotation_mouse_section_regions"))
            )
        return [
            clustering_squidpy_mode: mode,
            clustering_squidpy_table_key_suffix: AnnotationDefaults.tableKeySuffix(params, speciesName),
            annotation_anatomical_region: region,
            annotation_mouse_section_regions: sectionRegions,
            annotation_mode_mapmycells_stage: (
                stringOrNull(params?.get("annotation_mode_mapmycells_stage"))?.toLowerCase() ?:
                AnnotationDefaults.defaultMapmycellsStage(speciesName)
            ),
            mender_unassigned_state_policy: (
                stringOrNull(params?.get("mender_unassigned_state_policy")) ?:
                MAP_FIRST_MENDER_UNASSIGNED_STATE_POLICY
            ),
        ]
    }

    /**
     * Return the clustering mode recorded in a row's settings.
     *
     * @param settings Row settings from rowSampleSettings.
     * @return "legacy" when forRow added no mode, else the resolved mode.
     */
    static String mode(Map settings) {
        return settings?.get("clustering_squidpy_mode") ?: AnnotationDefaults.LEGACY
    }

    // The run's samplesheet row modes (useRowModes; null until main.nf sets
    // them, so a caller without a samplesheet resolves the params only).
    private static List ROW_MODES = null

    /**
     * Record each samplesheet row's clustering_squidpy_mode value (§20 D15).
     *
     * main.nf reads the samplesheet once before any channel runs, because
     * the clustering wiring (hook H5) is chosen when the DAG is built.
     *
     * @param rowModes The rows' values in samplesheet order (null clears).
     */
    static void useRowModes(Collection rowModes) {
        ROW_MODES = rowModes == null ? null : new ArrayList(rowModes).asImmutable()
    }

    /**
     * Whether the run takes the map_first clustering wiring (hook H5).
     *
     * The mode is one per row (pre-registration §20 D15): the row's
     * clustering_squidpy_mode column, else the run's mode from the run
     * species (params.species) and the clustering_squidpy_mode* params. The
     * run takes the map_first wiring when any row resolves to map_first;
     * without row modes (null) the run's mode decides. An invalid species
     * or mode gives false here; rowSampleSettings then reports it.
     *
     * @param params Pipeline params.
     * @param rowModes Each samplesheet row's clustering_squidpy_mode value
     *     (null or blank: the run's mode), or null to use the params only;
     *     by default the modes main.nf recorded with useRowModes.
     * @return Whether any row resolves to map_first.
     */
    static boolean isMapFirstRun(Map params, Collection rowModes = ROW_MODES) {
        try {
            def species = AnnotationDefaults.normalizeSpecies(params?.get("species"))
            if (rowModes == null) {
                return AnnotationDefaults.resolveMode(params, species) == AnnotationDefaults.MAP_FIRST
            }
            return rowModes.any { rowMode ->
                AnnotationDefaults.resolveRowMode(rowMode, params, species) == AnnotationDefaults.MAP_FIRST
            }
        } catch (IllegalArgumentException ignored) {
            return false
        }
    }

    /**
     * Samplesheet headers that look like clustering_squidpy_mode but are not it.
     *
     * A header that equals the column only after case-folding, trimming or
     * replacing spaces and hyphens by underscores is ignored by the pipeline,
     * so its rows would silently follow the run's mode; main.nf warns.
     *
     * @param headers The samplesheet's column names.
     * @return The near-miss headers (empty when there are none).
     */
    static List<String> nearMissModeColumns(Collection headers) {
        def wanted = AnnotationDefaults.ROW_MODE_COLUMN
        return (headers ?: []).collect { header -> header?.toString() }.findAll { header ->
            header != null && header != wanted &&
                header.trim().toLowerCase().replaceAll(/[\s\-]+/, "_") == wanted
        }
    }

    /**
     * Whether a human row still clusters in the deprecated legacy mode.
     *
     * Human flipped to map_first at M8 (pre-registration §20); a human run
     * whose params or row column ask for legacy gets a deprecation warning.
     *
     * @param params Pipeline params.
     * @param rowModes As isMapFirstRun (default: the recorded row modes).
     * @return Whether the run is human and a row resolves to legacy.
     */
    static boolean usesDeprecatedHumanLegacy(Map params, Collection rowModes = ROW_MODES) {
        try {
            def species = AnnotationDefaults.normalizeSpecies(params?.get("species"))
            if (species != "human") {
                return false
            }
            def modes = rowModes == null ? [null] : rowModes
            return modes.any { rowMode ->
                AnnotationDefaults.resolveRowMode(rowMode, params, species) == AnnotationDefaults.LEGACY
            }
        } catch (IllegalArgumentException ignored) {
            return false
        }
    }

    /**
     * Carry a map_first row's own annotation column into its samples JSON.
     *
     * Each sample gets the row's anatomical_region (human) or
     * mouse_section_regions (mouse) when the samplesheet row sets it; an
     * unset column is left out, so the sample inherits the global param
     * (ClusteringSquidpySampleConfig; plan §3.3). Legacy rows, and map_first
     * rows without the column, get the samples JSON back unchanged, so their
     * PREPARE task hash is the legacy one.
     *
     * @param samplesJson The pair x segmentation's samples JSON.
     * @param row Samplesheet row.
     * @param settings Row settings from rowSampleSettings.
     * @return The samples JSON.
     */
    static String samplesJsonWithRowColumns(Object samplesJson, Map row, Map settings) {
        if (!isMapFirst(settings)) {
            return samplesJson.toString()
        }
        def species = AnnotationDefaults.normalizeSpecies(settings.get("species"))
        def column = ROW_COLUMNS[species]
        def value = rowValue(row, column)
        if (value == null) {
            return samplesJson.toString()
        }
        def samples = new JsonSlurperClassic().parseText(samplesJson.toString()) as List
        samples.each { Map sample -> sample[column] = value }
        return JsonOutput.prettyPrint(JsonOutput.toJson(samples))
    }

    /**
     * Return the clustered-table fields a cortical-depth table config adds.
     *
     * A map_first row reads the clustered table of its own run, under the
     * run's table-key suffix (plan §4.8; CorticalDepthTableConfig
     * clustered_table_key_suffix); a legacy row adds nothing, so its config
     * JSON and task hash are unchanged.
     *
     * @param row Samplesheet row.
     * @param params Pipeline params.
     * @return [clustered_table_key_suffix: suffix] for a suffixed map_first
     *     row, else an empty map.
     */
    static Map clusteredTableFields(Map row, Map params) {
        def suffix = tableKeySuffix(forRow(row, params, params?.get("species")))
        return suffix ? [clustered_table_key_suffix: suffix] : [:]
    }

    /** Whether a row's settings select legacy clustering (the default). */
    static boolean isLegacy(Map settings) {
        return mode(settings) == AnnotationDefaults.LEGACY
    }

    /** Whether a row's settings select map_first clustering. */
    static boolean isMapFirst(Map settings) {
        return mode(settings) == AnnotationDefaults.MAP_FIRST
    }

    /**
     * Return the clustered table-key suffix of a row ("" in legacy).
     *
     * @param settings Row settings from rowSampleSettings.
     * @return The suffix token, "" when none.
     */
    static String tableKeySuffix(Map settings) {
        return settings?.get("clustering_squidpy_table_key_suffix") ?: ""
    }

    /**
     * Apply the map_first MAPMYCELLS rule to the stage selection (plan §3.1).
     *
     * A map_first row runs the legacy MAPMYCELLS stage only when
     * annotation_mode_mapmycells_stage is "legacy" and the user stops at
     * mapmycells (--stop_stage or --only_stage mapmycells). Legacy rows keep
     * the stage selection unchanged.
     *
     * @param stageSelected Whether mapmycells lies in the selected stage range.
     * @param annotationSettings The row's forRow result.
     * @param stopStage The row's normalized stop stage.
     * @return Whether MAPMYCELLS runs for the row.
     */
    static boolean runMapMyCells(
        boolean stageSelected,
        Map annotationSettings,
        Object stopStage
    ) {
        if (!isMapFirst(annotationSettings)) {
            return stageSelected
        }
        return (
            stageSelected &&
            annotationSettings.get("annotation_mode_mapmycells_stage") == AnnotationDefaults.LEGACY &&
            stopStage?.toString() == "mapmycells"
        )
    }

    // Published clustering directory of a legacy table (FINALIZE's publishDir).
    static final String CLUSTERING_PUBLISH_DIR = "clustering_squidpy"

    /**
     * Return the published clustering directory MENDER reads a clustered
     * H5AD from in a MENDER-only restart (hook H2).
     *
     * A legacy row (no suffix) reads clustering_squidpy/. A map_first row
     * with suffix s reads clustering_squidpy_<s>/, where a run into a
     * results directory with legacy outputs publishes FINALIZE (docs/stages/
     * annotation.md), and else clustering_squidpy/, FINALIZE's default
     * publishDir. MENDER_PREPARE and MENDER_IMPORT refuse a clustered H5AD
     * whose mode or suffix does not match the table key they write, so a
     * legacy H5AD found there is never imported into the map_first table.
     *
     * @param outdir Results root.
     * @param pairId Pair id.
     * @param segmentation Segmentation.
     * @param platform Platform.
     * @param sampleId Sample id.
     * @param suffix The row's clustering_squidpy_table_key_suffix (null or
     *     blank for legacy rows).
     * @return The directory name under <outdir>/<pair>/<segmentation>/.
     */
    static String publishedClusteringDir(
        Object outdir, Object pairId, Object segmentation, Object platform, Object sampleId, Object suffix
    ) {
        def token = suffix == null ? "" : suffix.toString().trim()
        if (!token) {
            return CLUSTERING_PUBLISH_DIR
        }
        def suffixed = "${CLUSTERING_PUBLISH_DIR}_${token}".toString()
        def h5ad = java.nio.file.Paths.get(
            outdir.toString(),
            pairId.toString(),
            segmentation.toString(),
            suffixed,
            "clustering_squidpy_out",
            platform.toString().toLowerCase(),
            "${sampleId}_clustered.h5ad".toString(),
        )
        return java.nio.file.Files.isRegularFile(h5ad) ? suffixed : CLUSTERING_PUBLISH_DIR
    }

    /**
     * Summarize a map_first run's failed branches and panels (hook H6).
     *
     * @param params Pipeline params.
     * @param runInfo Optional run facts: success (Boolean), n_expected and
     *     the SUMMARY_ITEMS lists (AnnotationRunRecord.runInfo).
     * @return "" when no row of the run is map_first (nothing to report;
     *     isMapFirstRun, which follows the recorded row modes), else the
     *     summary text, which covers the map_first rows.
     */
    static String completionSummary(Map params, Map runInfo = [:]) {
        def species
        try {
            species = AnnotationDefaults.normalizeSpecies(params?.get("species"))
        } catch (IllegalArgumentException ignored) {
            // rowSampleSettings has already reported the invalid param.
            return ""
        }
        if (!isMapFirstRun(params)) {
            return ""
        }
        def mode = AnnotationDefaults.MAP_FIRST
        def lines = ["Annotation summary (${species}, clustering_squidpy_mode ${mode}):".toString()]
        if (runInfo?.get("n_expected") != null) {
            lines << "  pair x segmentation branches clustered: ${runInfo.n_expected}".toString()
        }
        SUMMARY_ITEMS.each { key, label ->
            def values = (runInfo?.get(key) ?: []) as List
            lines << "  ${label}: ${values ? values.join(', ') : 'none'}".toString()
        }
        return lines.join("\n")
    }

    private static String rowValue(Map row, String column) {
        return stringOrNull(row?.get(column))
    }

    private static String stringOrNull(Object value) {
        if (value == null) {
            return null
        }
        def text = value.toString().trim()
        return text ?: null
    }

    // As merxen.annotation.samplesheet_columns.parse_anatomical_region.
    private static String regionToken(Object value) {
        def text = stringOrNull(value)
        return text == null ? null : text.toLowerCase().replaceAll(/[\s\-]+/, "_")
    }
}

/*
 * Preflight checks of map_first annotation runs (hook H4; plan §3.7).
 *
 * appendClusteringSquidpyPreflightChecks in main.nf calls append for every
 * row before its legacy-only checks. A legacy row gets no check from here.
 */
class AnnotationPreflight {

    // The map_first processes (ANNOTATE_PANEL, ANNOTATE_REFERENCE_PREP, MAP,
    // RESOLVE, COMPUTE_CPU) and hook H5 arrive in M2-M5; until M5 sets this,
    // a map_first run that touches clustered tables is refused.
    static final boolean MAP_FIRST_WIRED = false

    static final Map<String, List<String>> CHOICES = [
        annotation_panel_mode: ["auto", "intersection", "per_platform"],
        annotation_xplat_sensitivity: ["off", "geneset_c", "geneset_c+rescale"],
        annotation_mode_mapmycells_stage: ["legacy", "skip"],
        mender_unassigned_state_policy: ["state", "exclude_from_features"],
    ].asImmutable()

    // Optional local inputs checked for existence when set, by species.
    static final Map<String, List<String>> OPTIONAL_PATH_PARAMS = [
        both: [
            "annotation_gene_alias_table",
            "annotation_gene_id_overrides_csv",
            "annotation_panel_genes_path",
        ],
        human: [
            "annotation_whb_region_precompute_source",
            "annotation_whb_h5ad_dir",
            "annotation_whb_metadata_dir",
            "annotation_seaad_precomputed_stats_path",
            "annotation_seaad_metadata_dir",
        ],
        mouse: [
            "annotation_wmb_h5ad_dir",
            "annotation_wmb_metadata_dir",
            "annotation_wmb_mapping_stats_path",
            "annotation_merfish_ccf_metadata_path",
        ],
    ].asImmutable()

    /**
     * Append the map_first preflight errors of one row.
     *
     * @param errors Error list of runPreflightChecks (appended in place).
     * @param settings Row settings from rowSampleSettings.
     * @param params Pipeline params.
     */
    static void append(List errors, Map settings, Map params) {
        if (!AnnotationSettings.isMapFirst(settings)) {
            return
        }
        def species = AnnotationDefaults.normalizeSpecies(settings.get("species"))
        def label = "${settings.get('pair_id')} (${species}, clustering_squidpy_mode map_first)"
        if (!MAP_FIRST_WIRED && usesClusteredTables(settings)) {
            errors << (
                "clustering_squidpy_mode map_first is not available yet for " +
                "${label}: this version has only the annotation scaffolding " +
                "(milestone M1 of docs/plans/robust-celltype-annotation-plan.md); " +
                "run with the default legacy mode"
            ).toString()
        }
        CHOICES.each { paramName, allowed ->
            def value = params?.get(paramName)
            if (value != null && !(value.toString().trim().toLowerCase() in allowed)) {
                errors << (
                    "Unknown ${paramName} '${value}' for ${label}; " +
                    "expected one of ${allowed.join(', ')}"
                ).toString()
            }
        }
        if (!params?.get("annotation_ctm_version")?.toString()?.trim()) {
            errors << "annotation_ctm_version must name the cell_type_mapper version for ${label}".toString()
        }
        appendReferenceChecks(errors, species, params, label)
        if (species == "human") {
            def region = settings.get("annotation_anatomical_region")
            if (!(region in AnnotationDefaults.VALIDATED_HUMAN_REGIONS)) {
                errors << (
                    "Human anatomical_region '${region}' for ${label} has no validated " +
                    "references; only ${AnnotationDefaults.VALIDATED_HUMAN_REGIONS.join(', ')} " +
                    "is supported (OD-C7; a new region needs the work in plan §8.9)"
                ).toString()
            }
        } else {
            appendSectionRegionChecks(errors, settings.get("annotation_mouse_section_regions"), label)
        }
        (OPTIONAL_PATH_PARAMS.both + OPTIONAL_PATH_PARAMS[species]).each { paramName ->
            appendOptionalPathCheck(errors, params?.get(paramName), paramName, label)
        }
    }

    private static boolean usesClusteredTables(Map settings) {
        return [
            "run_clustering_squidpy",
            "run_mapmycells",
            "run_compute_cortical_depth",
            "run_distance_from_object",
            "run_mender",
        ].any { key -> settings.get(key) }
    }

    private static void appendReferenceChecks(
        List errors,
        String species,
        Map params,
        String label
    ) {
        def paramName = "annotation_${species}_references".toString()
        def raw = params?.get(paramName)
        def references = raw == null
            ? []
            : raw.toString().split(",").collect { part -> part.trim() }.findAll { part -> part }
        if (!references) {
            errors << "${paramName} names no reference for ${label}".toString()
            return
        }
        def unknown = references.findAll { reference ->
            !(reference in AnnotationDefaults.KNOWN_REFERENCES[species])
        }
        if (unknown) {
            errors << (
                "Unknown ${species} references in ${paramName} for ${label}: " +
                "${unknown.join(', ')}; known: " +
                "${AnnotationDefaults.KNOWN_REFERENCES[species].join(', ')}"
            ).toString()
        }
    }

    private static void appendSectionRegionChecks(List errors, Object rawValue, String label) {
        def value = rawValue == null ? "" : rawValue.toString().trim()
        if (!value) {
            errors << "mouse_section_regions must not be blank for ${label}".toString()
            return
        }
        if (value.toLowerCase() in AnnotationDefaults.MOUSE_SECTION_REGION_KEYWORDS) {
            return
        }
        def known = AnnotationDefaults.MOUSE_CCF_REGIONS.collect { region -> region.toLowerCase() }
        def parts = value.split(";").collect { part -> part.trim() }.findAll { part -> part }
        def unknown = parts.findAll { part -> !(part.toLowerCase() in known) }
        if (!parts || unknown) {
            errors << (
                "Unknown mouse_section_regions '${value}' for ${label}; use auto, none " +
                "or ';'-separated divisions from " +
                "${AnnotationDefaults.MOUSE_CCF_REGIONS.join(', ')}"
            ).toString()
        }
    }

    private static void appendOptionalPathCheck(
        List errors,
        Object rawPath,
        String paramName,
        String label
    ) {
        if (rawPath == null) {
            return
        }
        def text = rawPath.toString().trim()
        if (!text || text.toLowerCase() == "null") {
            return
        }
        def path = java.nio.file.Paths.get(text).toAbsolutePath().normalize()
        if (!path.toFile().exists()) {
            errors << "Missing ${paramName} for ${label}: ${path}".toString()
        }
    }
}

/*
 * Preflight checks of map_first annotation runs (hook H4; plan §3.7).
 *
 * appendClusteringSquidpyPreflightChecks in main.nf calls append for every
 * row before its legacy-only checks. A legacy row gets no check from here.
 * ANNOTATION_PREPARE_ONLY (hook H10) calls prepareOnlyErrors once per
 * --annotation_prepare_only run instead: such a run preflights no row.
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
            "annotation_human_gene_table",
            "annotation_mouse_gene_table",
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
            "annotation_wmb_selfmap_test_cells_path",
            "annotation_wmb_marker_gene_universe_path",
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
        def suffix = AnnotationSettings.tableKeySuffix(settings)
        if (!suffix && !(species in AnnotationDefaults.FLIPPED_SPECIES)) {
            errors << "${label}: ${AnnotationDefaults.emptySuffixMessage(species)}".toString()
        }
        appendChoiceChecks(errors, params, label)
        appendCtmVersionCheck(errors, params, label)
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
        if (usesClusteredTables(settings)) {
            // Only rows with a clustered-table stage build reference bundles.
            appendSelfmapTestCellCheck(errors, species, params, label)
        }
    }

    /**
     * Return the errors of a --annotation_prepare_only run (plan §3.2).
     *
     * A prepare-only run builds the reference bundles of its declared panel
     * and runs no pipeline stage, so only the params it reads are checked:
     * the gene list (the declared panel until M5 wires the prepared H5ADs),
     * the species' references and their source params, the reference store
     * and the settings annotation_config.json takes.
     *
     * @param params Pipeline params.
     * @return Error messages (empty when the run can start).
     */
    static List<String> prepareOnlyErrors(Map params) {
        def errors = []
        def species
        try {
            species = AnnotationDefaults.normalizeSpecies(params?.get("species"))
        } catch (IllegalArgumentException error) {
            return [error.message]
        }
        def label = "annotation_prepare_only (${species})".toString()
        def geneList = AnnotationReferences.pathText(params?.get("annotation_panel_genes_path"))
        if (!geneList) {
            errors << (
                "${label} needs --annotation_panel_genes_path (the declared panel). " +
                "Building bundles from a pair's prepared H5ADs arrives with the " +
                "map_first wiring (milestone M5 of docs/plans/robust-celltype-annotation-plan.md)"
            ).toString()
        }
        appendChoiceChecks(errors, params, label)
        appendCtmVersionCheck(errors, params, label)
        appendReferenceChecks(errors, species, params, label)
        def references = AnnotationReferences.referenceIds(params, species)
        def withoutSources = references.findAll { referenceId ->
            referenceId in AnnotationDefaults.KNOWN_REFERENCES[species] &&
                !AnnotationReferences.SOURCE_PARAMS.containsKey(referenceId)
        }
        if (withoutSources) {
            errors << (
                "${label}: ${withoutSources.join(', ')} have no pipeline source params " +
                "yet; build them with merxen annotation-reference-prep --source NAME=PATH, " +
                "or leave them out of annotation_${species}_references"
            ).toString()
        }
        references.findAll { referenceId ->
            AnnotationReferences.REQUIRED_SOURCE_PARAMS.containsKey(referenceId)
        }.each { referenceId ->
            def alternatives = AnnotationReferences.REQUIRED_SOURCE_PARAMS[referenceId]
            def satisfied = alternatives.any { names ->
                names.every { name -> AnnotationReferences.pathText(params?.get(name)) }
            }
            if (!satisfied) {
                errors << (
                    "${label}: ${referenceId} needs " +
                    alternatives.collect { names -> names.join(' + ') }.join(' or ')
                ).toString()
            }
        }
        def sourceParams = references.collectMany { referenceId ->
            (AnnotationReferences.SOURCE_PARAMS[referenceId] ?: [:]).values() as List
        }.unique()
        (sourceParams + OPTIONAL_PATH_PARAMS.both).unique().each { paramName ->
            appendOptionalPathCheck(errors, params?.get(paramName), paramName, label)
        }
        if (species == "human") {
            def region = params?.get("annotation_human_region")
            def token = region == null ? null : region.toString().trim().toLowerCase().replaceAll(/[\s\-]+/, "_")
            if (!(token in AnnotationDefaults.VALIDATED_HUMAN_REGIONS)) {
                errors << (
                    "Human annotation_human_region '${region}' for ${label} has no validated " +
                    "references; only ${AnnotationDefaults.VALIDATED_HUMAN_REGIONS.join(', ')} " +
                    "is supported (OD-C7)"
                ).toString()
            }
        }
        appendSelfmapTestCellCheck(errors, species, params, label)
        appendStoreCheck(errors, AnnotationReferences.referenceStore(params), "annotation_reference_store", label)
        def large = AnnotationReferences.referenceStoreLarge(params)
        if (large) {
            appendStoreCheck(errors, large, "annotation_reference_store_large", label)
        }
        return errors
    }

    // Params the human held-out test set (whb_frontal_supc_clus_ho) reads:
    // the region directory (region_cell_metadata.csv), the raw WHB h5ads and
    // the WHB taxonomy tables (cluster annotation and membership).
    static final List<String> HUMAN_SELFMAP_SOURCE_PARAMS = [
        "annotation_whb_region_precompute_source",
        "annotation_whb_h5ad_dir",
        "annotation_whb_metadata_dir",
    ].asImmutable()

    // Human references whose PREP runs the WHB held-out self-map.
    static final List<String> HUMAN_SELFMAP_REFERENCES = [
        "whb_frontal_supc_clus",
        "seaad_mr_panel",
    ].asImmutable()

    /**
     * Require the self-map sources while resolvability is on. Mouse: the
     * self-map test cells of a wmb_panel build, which the marker training
     * cells must exclude (plan §3.2, §8.3). Human: the WHB held-out donor's
     * sources of the whb_frontal_supc_clus and seaad_mr_panel self-maps,
     * which PREP would otherwise miss only after its marker steps.
     */
    private static void appendSelfmapTestCellCheck(List errors, Object species, Map params, String label) {
        if (!AnnotationReferences.isTrue(params?.get("annotation_resolvability"), true)) {
            return
        }
        if (species == "human") {
            def selfMapped = AnnotationReferences.referenceIds(params, species).findAll { referenceId ->
                referenceId in HUMAN_SELFMAP_REFERENCES
            }
            def missing = HUMAN_SELFMAP_SOURCE_PARAMS.findAll { paramName ->
                !AnnotationReferences.pathText(params?.get(paramName))
            }
            if (selfMapped && missing) {
                errors << (
                    "${label}: ${selfMapped.join(', ')} need ${missing.join(', ')} " +
                    "(the WHB held-out donor's cells for the resolvability self-map) " +
                    "while annotation_resolvability is true"
                ).toString()
            }
            return
        }
        if (species != "mouse" || !("wmb_panel" in AnnotationReferences.referenceIds(params, species))) {
            return
        }
        if (!AnnotationReferences.pathText(params?.get("annotation_wmb_selfmap_test_cells_path"))) {
            errors << (
                "${label}: wmb_panel needs annotation_wmb_selfmap_test_cells_path " +
                "(the WMB self-map test cells, research/selfmap/truth.csv) while " +
                "annotation_resolvability is true: the marker training cells must " +
                "exclude them"
            ).toString()
        }
    }

    private static void appendChoiceChecks(List errors, Map params, String label) {
        CHOICES.each { paramName, allowed ->
            def value = params?.get(paramName)
            if (value != null && !(value.toString().trim().toLowerCase() in allowed)) {
                errors << (
                    "Unknown ${paramName} '${value}' for ${label}; " +
                    "expected one of ${allowed.join(', ')}"
                ).toString()
            }
        }
    }

    private static void appendCtmVersionCheck(List errors, Map params, String label) {
        if (!params?.get("annotation_ctm_version")?.toString()?.trim()) {
            errors << "annotation_ctm_version must name the cell_type_mapper version for ${label}".toString()
        }
    }

    // The store is created on first use: its nearest existing ancestor must
    // be a writable directory.
    private static void appendStoreCheck(List errors, String root, String paramName, String label) {
        def path = java.nio.file.Paths.get(root).toAbsolutePath().normalize()
        while (path != null && !java.nio.file.Files.exists(path)) {
            path = path.parent
        }
        if (path == null || !java.nio.file.Files.isDirectory(path) || !java.nio.file.Files.isWritable(path)) {
            errors << "${paramName} ${root} is not writable for ${label} (nearest existing: ${path})".toString()
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

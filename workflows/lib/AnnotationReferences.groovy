import groovy.json.JsonOutput
import groovy.json.JsonSlurperClassic

import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths

/*
 * Reference bundles of annotation runs (plan §3.1, §3.2, §3.7).
 *
 * ANNOTATE_PANEL writes one pair x segmentation's declared panels and its
 * required_bundles.json; ANNOTATE_REFERENCE_PREP gets or builds one bundle
 * per unique (species, reference_id, panel_hash). This class holds what the
 * processes and workflows/subworkflows/annotation_references.nf share: the
 * annotation_config.json both commands read, the command arguments, the
 * bundle keys and the per pair x segmentation bookkeeping that lets MAP wait
 * for exactly the bundles its panels need.
 *
 * Nothing here runs in a legacy run: main.nf calls ANNOTATION_PREPARE_ONLY
 * only with --annotation_prepare_only, and CLUSTERING_MAP_FIRST (map_first)
 * has no caller before milestone M5.
 */
class AnnotationReferences {

    static final String PANEL_OUTPUT_DIR = "annotation_panel_out"
    static final String PANEL_INPUT_DIR = "panel_inputs"
    static final String REQUIRED_BUNDLES_FILE = "required_bundles.json"
    static final String BUNDLE_REF_FILE = "bundle_ref.json"
    static final String ANNOTATION_CONFIG_FILE = "annotation_config.json"
    static final String PREP_SCRATCH_DIR = "prep_scratch"
    static final String PANEL_INDEPENDENT = "panel_independent"
    static final String DEFAULT_STORE_DIR = "annotation_references"
    static final String PREPARED_SOURCE = "prepared"
    static final String GENE_LIST_SOURCE = "gene_list"

    // Share of the PREP memory given to cell_type_mapper's --max_gb (the
    // reference-marker step; 40 GB of the 64 GB reserve, as validated).
    static final double PREP_MAX_GB_FRACTION = 0.625

    // Pipeline param of each builder source, per reference (plan §3.7). The
    // builders expand directory sources and fill SEA-AD files from the pinned
    // download cache (merxen.annotation.reference.prepare_reference_spec).
    static final Map<String, Map<String, String>> SOURCE_PARAMS = [
        whb_frontal_supc_clus: [
            region_precompute: "annotation_whb_region_precompute_source",
            whb_h5ad_dir: "annotation_whb_h5ad_dir",
            whb_metadata_dir: "annotation_whb_metadata_dir",
            seaad_precomputed_stats: "annotation_seaad_precomputed_stats_path",
        ].asImmutable(),
        seaad_mr_panel: [
            seaad_precomputed_stats: "annotation_seaad_precomputed_stats_path",
            seaad_metadata_dir: "annotation_seaad_metadata_dir",
        ].asImmutable(),
        wmb_panel: [
            wmb_h5ad_dir: "annotation_wmb_h5ad_dir",
            wmb_metadata_dir: "annotation_wmb_metadata_dir",
            wmb_mapping_stats: "annotation_wmb_mapping_stats_path",
            // Self-map test cells kept out of the marker build (plan §3.2).
            wmb_selfmap_test_cells: "annotation_wmb_selfmap_test_cells_path",
        ].asImmutable(),
        wmb_region_share: [
            merfish_ccf_metadata: "annotation_merfish_ccf_metadata_path",
        ].asImmutable(),
    ].asImmutable()

    // Source params that must be set for a reference to build, as
    // alternatives (one alternative fully set suffices). Files with a pinned
    // download (SEA-AD, the MERFISH CCF metadata) are not required: an unset
    // path is fetched into <store>/.downloads when annotation_auto_download is
    // true, or taken from a copy already verified there.
    static final Map<String, List<List<String>>> REQUIRED_SOURCE_PARAMS = [
        whb_frontal_supc_clus: [
            ["annotation_whb_region_precompute_source"],
            ["annotation_whb_h5ad_dir", "annotation_whb_metadata_dir"],
        ],
        wmb_panel: [
            [
                "annotation_wmb_h5ad_dir",
                "annotation_wmb_metadata_dir",
                "annotation_wmb_mapping_stats_path",
            ],
        ],
    ].asImmutable()

    /** Whether the run only builds reference bundles (--annotation_prepare_only). */
    static boolean prepareOnly(Map params) {
        return isTrue(params?.get("annotation_prepare_only"))
    }

    /**
     * Return the reference ids of a species (annotation_<species>_references).
     *
     * @param params Pipeline params.
     * @param species Species name or alias.
     * @return The ids in their param order, without blanks or duplicates.
     */
    static List<String> referenceIds(Map params, Object species) {
        def speciesName = AnnotationDefaults.normalizeSpecies(species)
        def raw = params?.get("annotation_${speciesName}_references".toString())
        if (raw == null) {
            return AnnotationDefaults.DEFAULT_REFERENCES[speciesName]
        }
        return raw.toString().split(",")
            .collect { part -> part.trim() }
            .findAll { part -> part }
            .unique()
    }

    /**
     * Return the absolute reference store root.
     *
     * @param params Pipeline params.
     * @return annotation_reference_store, or <outdir>/annotation_references.
     */
    static String referenceStore(Map params) {
        def explicit = pathText(params?.get("annotation_reference_store"))
        def root = explicit ?: Paths.get(
            (params?.get("outdir") ?: "results").toString(),
            DEFAULT_STORE_DIR,
        ).toString()
        return absolute(root)
    }

    /**
     * Return the absolute large-panel store root, if configured.
     *
     * @param params Pipeline params.
     * @return annotation_reference_store_large, or null.
     */
    static String referenceStoreLarge(Map params) {
        def explicit = pathText(params?.get("annotation_reference_store_large"))
        return explicit ? absolute(explicit) : null
    }

    /**
     * Return the annotation_config.json content ANNOTATE_PANEL and PREP read.
     *
     * References are named by id; merxen.annotation.config expands them to
     * their known specs, so the taxonomy settings live in Python only.
     *
     * @param params Pipeline params.
     * @param species Species name or alias.
     * @return A map for merxen.annotation.config.AnnotationConfig.
     */
    static Map annotationConfig(Map params, Object species) {
        def speciesName = AnnotationDefaults.normalizeSpecies(species)
        def references = referenceIds(params, speciesName).collect { referenceId ->
            def maxCells = params?.get("annotation_wmb_max_cells_per_cluster")
            if (referenceId == "wmb_panel" && maxCells != null) {
                return [reference_id: referenceId, max_cells_per_cluster: maxCells as int]
            }
            return referenceId
        }
        def panel = [
            panel_mode: textOr(params?.get("annotation_panel_mode"), "auto").toLowerCase(),
        ]
        [
            gene_id_fallback_csv: "annotation_gene_id_fallback_csv",
            gene_alias_table: "annotation_gene_alias_table",
            gene_id_overrides_csv: "annotation_gene_id_overrides_csv",
        ].each { field, paramName ->
            def path = pathText(params?.get(paramName))
            if (path) {
                panel[field] = absolute(path)
            }
        }
        def config = [
            enabled: true,
            species: speciesName,
            references: references,
            panel: panel,
            resolvability: [
                enabled: isTrue(params?.get("annotation_resolvability"), true),
                holdout_donor: textOr(params?.get("annotation_calibration_holdout_donor"), "auto"),
            ],
            thresholds: [
                allow_fine_levels: isTrue(params?.get("annotation_allow_fine_levels")),
            ],
            allow_single_method: isTrue(params?.get("annotation_allow_single_method")),
            xplat_sensitivity: textOr(params?.get("annotation_xplat_sensitivity"), "geneset_c").toLowerCase(),
            xplat_sensitivity_segmentations: listOf(
                params?.get("annotation_xplat_sensitivity_segmentations"),
                ["proseg_hybrid"],
            ),
            reuse_published: isTrue(params?.get("annotation_reuse_published"), true),
            keep_extended_json: isTrue(params?.get("annotation_keep_extended_json")),
            ctm_version: textOr(params?.get("annotation_ctm_version"), "1.7.2"),
            reference_store: referenceStore(params),
            reference_store_large: referenceStoreLarge(params),
        ]
        if (speciesName == "human") {
            config.anatomical_region = regionToken(params?.get("annotation_human_region"))
        } else {
            config.mouse_section_regions = textOr(params?.get("annotation_mouse_section_regions"), "auto")
        }
        return config
    }

    /** Return annotationConfig as pretty JSON. */
    static String annotationConfigJson(Map params, Object species) {
        return JsonOutput.prettyPrint(JsonOutput.toJson(annotationConfig(params, species)))
    }

    /**
     * Return the ANNOTATE_PANEL spec of a gene-list panel (prepare-only runs).
     *
     * @param geneListName File name of the staged gene list.
     * @param platforms The row's active platforms.
     * @return The panel spec map.
     */
    static Map geneListPanelSpec(String geneListName, List platforms) {
        return [
            source: GENE_LIST_SOURCE,
            gene_list: geneListName,
            platforms: (platforms ?: []).collect { platform -> platform.toString().toUpperCase() },
        ]
    }

    /**
     * Return the ANNOTATE_PANEL spec of a prepared directory (map_first runs).
     *
     * @param clusteringConfigName Staged clustering_squidpy_config.json name.
     * @param preparedDirName Staged CLUSTERING_SQUIDPY_PREPARE output name.
     * @param maskNames Staged [shared_tissue_mask.npy, registration_summary.json]
     *     names, or an empty list (set c then uses the whole section).
     * @return The panel spec map.
     */
    static Map preparedPanelSpec(
        String clusteringConfigName,
        String preparedDirName,
        List maskNames = []
    ) {
        def spec = [
            source: PREPARED_SOURCE,
            clustering_config: clusteringConfigName,
            prepared_dir: preparedDirName,
        ]
        if (maskNames && maskNames.size() == 2) {
            spec.shared_tissue_mask = maskNames[0].toString()
            spec.registration_summary = maskNames[1].toString()
        }
        return spec
    }

    /**
     * Return the published shared tissue mask of a pair, if both files exist.
     *
     * @param params Pipeline params.
     * @param pairId Pair id.
     * @return [mask, registration summary] paths, or an empty list.
     */
    static List<Path> publishedSharedTissueMask(Map params, Object pairId) {
        def alignOut = Paths.get(
            (params?.get("outdir") ?: "results").toString(),
            pairId.toString(),
            "alignment",
            "align_out",
        ).toAbsolutePath().normalize()
        def files = [
            alignOut.resolve("shared_tissue_mask.npy"),
            alignOut.resolve("registration_summary.json"),
        ]
        return files.every { path -> Files.isRegularFile(path) } ? files : []
    }

    /**
     * Return the merxen annotation-panel arguments of one panel spec.
     *
     * @param spec geneListPanelSpec or preparedPanelSpec result.
     * @param pairId Pair id.
     * @param segmentation Segmentation.
     * @return Shell-quoted arguments (without --species, --annotation-config
     *     and --output-dir).
     */
    static String panelArguments(Map spec, Object pairId, Object segmentation) {
        def args = ["--pair-id", pairId.toString(), "--segmentation", segmentation.toString()]
        if (spec.source == GENE_LIST_SOURCE) {
            args += ["--panel-genes-path", staged(spec.gene_list)]
            if (spec.platforms) {
                args += ["--platforms", spec.platforms.join(",")]
            }
        } else if (spec.source == PREPARED_SOURCE) {
            args += [
                "--prepared-dir", staged(spec.prepared_dir),
                "--clustering-config", staged(spec.clustering_config),
            ]
            if (spec.shared_tissue_mask && spec.registration_summary) {
                args += [
                    "--shared-tissue-mask", staged(spec.shared_tissue_mask),
                    "--registration-summary", staged(spec.registration_summary),
                ]
            }
            (spec.panel_files ?: [:]).each { key, name ->
                args += ["--panel-file", "${key}=${staged(name)}".toString()]
            }
        } else {
            throw new IllegalArgumentException("Unknown annotation panel source '${spec.source}'")
        }
        return args.collect { arg -> shellQuote(arg) }.join(" ")
    }

    /**
     * Return the merxen annotation-reference-prep arguments of one bundle.
     *
     * @param params Pipeline params.
     * @param bundle bundleRequest map.
     * @param panelGenes Staged panel_genes*.json path, or null.
     * @param cpus Task cpus (--n-processors).
     * @param memoryGb Task memory in GB (sets --max-gb).
     * @return Shell-quoted arguments (without --annotation-config and --output).
     */
    static String prepArguments(Map params, Map bundle, Object panelGenes, int cpus, long memoryGb) {
        def args = [
            "--reference-id", bundle.reference_id.toString(),
            "--species", bundle.species.toString(),
            "--store", referenceStore(params),
            "--scratch-dir", PREP_SCRATCH_DIR,
            "--n-processors", cpus.toString(),
            "--max-gb", maxGb(memoryGb).toString(),
            isTrue(params?.get("annotation_auto_download"), true) ? "--auto-download" : "--no-auto-download",
        ]
        def large = referenceStoreLarge(params)
        if (large) {
            args += ["--store-large", large]
        }
        if (panelGenes) {
            args += ["--panel-genes", panelGenes.toString()]
        }
        referenceSources(params, bundle.reference_id).each { name, path ->
            args += ["--source", "${name}=${path}".toString()]
        }
        return args.collect { arg -> shellQuote(arg) }.join(" ")
    }

    /**
     * Return the configured builder sources of a reference.
     *
     * @param params Pipeline params.
     * @param referenceId Reference id.
     * @return Source name to absolute path, for the params that are set.
     */
    static Map<String, String> referenceSources(Map params, Object referenceId) {
        def sources = [:] as TreeMap
        (SOURCE_PARAMS[referenceId?.toString()] ?: [:]).each { name, paramName ->
            def path = pathText(params?.get(paramName))
            if (path) {
                sources[name] = absolute(path)
            }
        }
        return sources
    }

    /** Return cell_type_mapper's --max_gb for a PREP memory reserve. */
    static int maxGb(long memoryGb) {
        return Math.max(1, (int) Math.floor(memoryGb * PREP_MAX_GB_FRACTION))
    }

    /**
     * Return the PREP task key of a bundle.
     *
     * @return "<species>|<reference_id>|<panel_hash or panel_independent>".
     */
    static String bundleKey(Object species, Object referenceId, Object panelHash) {
        return [species, referenceId, panelHash ?: PANEL_INDEPENDENT].join("|")
    }

    /** Return the join key of one pair x segmentation. */
    static String branchKey(Object pairId, Object segmentation) {
        return "${pairId}|${segmentation}".toString()
    }

    /**
     * Read the required_bundles.json of an ANNOTATE_PANEL output directory.
     *
     * @param panelDir ANNOTATE_PANEL output directory.
     * @return The parsed file (merxen.annotation.panel.RequiredBundles).
     */
    static Map requiredBundles(Object panelDir) {
        def path = asPath(panelDir).resolve(REQUIRED_BUNDLES_FILE)
        // JsonSlurperClassic builds plain maps: the lazy maps of JsonSlurper
        // are not safe to read from the operator threads that share them.
        def required = new JsonSlurperClassic().parseText(Files.readString(path)) as Map
        def bundles = (required.bundles ?: []) as List
        if ((required.n_required ?: 0) as int != bundles.size()) {
            throw new IllegalStateException(
                "${path}: n_required ${required.n_required} != ${bundles.size()} bundles"
            )
        }
        return required
    }

    /**
     * Return one PREP request per required bundle of a pair x segmentation.
     *
     * @param panelDir ANNOTATE_PANEL output directory.
     * @param required requiredBundles(panelDir).
     * @return [bundle, panel files] per bundle, where bundle is the map PREP
     *     reads (key, species, reference_id, role, panel_hash, panel_tag,
     *     n_panel_genes; all fixed by the key, so the task hash does not
     *     depend on which pair asked) and panel files holds the bundle's
     *     panel_genes*.json (empty for panel-independent references).
     */
    static List prepRequests(Object panelDir, Map required) {
        def directory = asPath(panelDir)
        return ((required.bundles ?: []) as List).collect { Map item ->
            def panelHash = item.panel_hash?.toString()
            def bundle = [
                key: bundleKey(item.species, item.reference_id, panelHash),
                species: item.species.toString(),
                reference_id: item.reference_id.toString(),
                role: item.role.toString(),
                panel_hash: panelHash,
                panel_tag: panelHash ?: PANEL_INDEPENDENT,
                n_panel_genes: item.n_panel_genes == null ? null : item.n_panel_genes as int,
            ]
            def panelFiles = item.panel_file ? [directory.resolve(item.panel_file.toString())] : []
            [bundle, panelFiles]
        }
    }

    /**
     * Return what one pair x segmentation waits for.
     *
     * @return [bundle key, branch key, n_required] per required bundle.
     */
    static List bundleNeeds(Object pairId, Object segmentation, Map required) {
        def branch = branchKey(pairId, segmentation)
        def bundles = (required.bundles ?: []) as List
        return bundles.collect { Map item ->
            [
                bundleKey(item.species, item.reference_id, item.panel_hash),
                branch,
                bundles.size(),
            ]
        }
    }

    /**
     * Order a pair x segmentation's bundle refs as its required_bundles.json.
     *
     * @param required requiredBundles result.
     * @param keys Bundle keys, as collected.
     * @param refs bundle_ref.json paths, parallel to keys.
     * @return The refs in required order.
     * @throws IllegalStateException If a required bundle is missing.
     */
    static List orderedBundleRefs(Map required, List keys, List refs) {
        def byKey = [:]
        keys.eachWithIndex { key, index -> byKey[key.toString()] = refs[index] }
        return ((required.bundles ?: []) as List).collect { Map item ->
            def key = bundleKey(item.species, item.reference_id, item.panel_hash)
            if (!byKey.containsKey(key)) {
                throw new IllegalStateException("bundle ${key} was not collected")
            }
            byKey[key]
        }
    }

    /** Return the stub bundle_ref.json of a bundle (for -stub-run). */
    static String stubBundleRefJson(Map params, Map bundle) {
        def store = referenceStore(params)
        return JsonOutput.prettyPrint(JsonOutput.toJson([
            reference_id: bundle.reference_id,
            species: bundle.species,
            role: bundle.role,
            panel_hash: bundle.panel_hash,
            build_hash: "0" * 64,
            path: Paths.get(store, bundle.reference_id.toString(), "stub-${bundle.panel_tag}").toString(),
            store_root: store,
            panel_trust: null,
            reused: false,
        ]))
    }

    /** Parse a boolean param ("true"/"false", booleans; null gives the default). */
    static boolean isTrue(Object value, boolean defaultValue = false) {
        if (value == null) {
            return defaultValue
        }
        if (value instanceof Boolean) {
            return value
        }
        def text = value.toString().trim().toLowerCase()
        if (!text) {
            return defaultValue
        }
        return text in ["true", "yes", "1"]
    }

    /** Return a param path as text, or null when unset ("", "null", "false"). */
    static String pathText(Object value) {
        if (value == null) {
            return null
        }
        def text = value.toString().trim()
        return text.toLowerCase() in ["", "null", "false"] ? null : text
    }

    private static String absolute(String path) {
        return Paths.get(path).toAbsolutePath().normalize().toString()
    }

    private static Path asPath(Object value) {
        return value instanceof Path ? (Path) value : Paths.get(value.toString())
    }

    private static String staged(Object name) {
        return "${PANEL_INPUT_DIR}/${name}".toString()
    }

    private static String textOr(Object value, String fallback) {
        def text = value == null ? "" : value.toString().trim()
        return text ?: fallback
    }

    private static List<String> listOf(Object value, List<String> fallback) {
        if (value == null) {
            return fallback
        }
        def items = value instanceof Collection
            ? value.collect { item -> item.toString().trim() }
            : value.toString().split(",").collect { item -> item.trim() }
        return items.findAll { item -> item }
    }

    // As merxen.annotation.samplesheet_columns.parse_anatomical_region.
    private static String regionToken(Object value) {
        def text = value == null ? "" : value.toString().trim()
        return text ? text.toLowerCase().replaceAll(/[\s\-]+/, "_") : null
    }

    static String shellQuote(Object value) {
        def text = value.toString()
        if (text ==~ /[A-Za-z0-9_@%+=:,.\/\-]+/) {
            return text
        }
        return "'" + text.replace("'", "'\\''") + "'"
    }
}

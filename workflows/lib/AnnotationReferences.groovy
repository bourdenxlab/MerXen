import groovy.json.JsonOutput
import groovy.json.JsonSlurperClassic

import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.security.MessageDigest
import java.util.concurrent.ConcurrentHashMap

/*
 * Reference bundles of annotation runs (plan §3.1, §3.2, §3.7).
 *
 * ANNOTATE_PANEL writes one pair x segmentation's declared panels and its
 * required_bundles.json; ANNOTATE_REFERENCE_PREP gets or builds one bundle
 * per unique (species, reference_id, panel_hash); CLUSTERING_SQUIDPY_ANNOTATE_MAP
 * maps the pair x segmentation onto its bundles (plan §3.3) and
 * CLUSTERING_SQUIDPY_ANNOTATE_RESOLVE turns the mapping into label tables
 * (plan §3.4). This class holds what the processes and
 * workflows/subworkflows/annotation_references.nf and clustering_map_first.nf
 * share: the annotation_config.json the commands read, the command
 * arguments, the bundle keys, the per pair x segmentation bookkeeping that
 * lets MAP wait for exactly the bundles its panels need, and the RESOLVE
 * rules fingerprint.
 *
 * Nothing here runs in a legacy run: main.nf calls ANNOTATION_PREPARE_ONLY
 * only with --annotation_prepare_only, and CLUSTERING_MAP_FIRST (hook H5)
 * only in map_first runs.
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

    // CLUSTERING_SQUIDPY_ANNOTATE_MAP: output directory (published under
    // <outdir>/<pair>/<seg>/annotation_map/), staged-input directory,
    // task-local scratch and the manifest published-output reuse reads.
    static final String MAP_PUBLISH_DIR = "annotation_map"
    static final String MAP_OUTPUT_DIR = "annotation_map_out"
    static final String MAP_INPUT_DIR = "map_inputs"
    static final String MAP_SCRATCH_DIR = "map_scratch"
    static final String MAP_MANIFEST_FILE = "map_manifest.json"

    // CLUSTERING_SQUIDPY_ANNOTATE_RESOLVE: staged-input directory and the
    // pair summary's suffix (merxen.annotation.pipeline). The task writes
    // annotation_resolve_out/, published under
    // <outdir>/<pair>/<seg>/annotation_resolve/.
    static final String RESOLVE_INPUT_DIR = "resolve_inputs"
    static final String RESOLVE_OUTPUT_DIR = "annotation_resolve_out"
    static final String RESOLVE_SUMMARY_SUFFIX = "_resolve_summary.json"

    // CLUSTERING_SQUIDPY_COMPUTE_CPU: staged-input directory (the task
    // writes clustering_compute_out/, FINALIZE's input) and the per-sample
    // RESOLVE outputs it stages (merxen.annotation.schema.label_table_filename,
    // merxen.annotation.provenance.annotation_manifest_filename).
    static final String COMPUTE_INPUT_DIR = "compute_inputs"
    static final List<String> LABEL_FILE_SUFFIXES = [
        "_celltype_labels.parquet",
        "_annotation_manifest.json",
    ].asImmutable()

    // The ALIGN outputs ANNOTATE_PANEL (label-free set c) and RESOLVE (the
    // pair JSD's shared-mask restriction) read (merxen.annotation.pipeline
    // .load_pair_mask).
    static final String SHARED_TISSUE_MASK_FILE = "shared_tissue_mask.npy"
    static final String REGISTRATION_SUMMARY_FILE = "registration_summary.json"

    // What the RESOLVE rules fingerprint covers, relative to this checkout's
    // src/: the threshold, floor, trust, consensus, flag and composition code
    // with its packaged tables (floors, vocab, validated panels, state
    // genes), the cell-set rule and the annotate-resolve command. The task
    // hash cannot see Python code or packaged files, so the fingerprint
    // travels in RESOLVE's resolve_spec: a rule change re-runs RESOLVE
    // (minutes) under -resume and never MAP (plan §3.1). Of merxen.clustering
    // RESOLVE runs only the cell-set rule; the map_first hierarchy is
    // COMPUTE_CPU's (HIERARCHY_SOURCES), so its edits do not re-run RESOLVE.
    static final List<String> RESOLVE_RULE_SOURCES = [
        "merxen/annotation",
        "merxen/assets/annotation",
        "merxen/clustering/cellset.py",
        "merxen/cli/run_annotation.py",
    ].asImmutable()

    // Files under RESOLVE_RULE_SOURCES the fingerprint leaves out (relative
    // path prefixes): the annotation report (merxen/annotation/report.py and
    // report_*.py, M7) reads RESOLVE's outputs and RESOLVE never imports it
    // (test-enforced), so a report edit must not re-run RESOLVE; the
    // fingerprint of a checkout is then the one it had before the report.
    static final List<String> RESOLVE_RULE_EXCLUDES = [
        "merxen/annotation/report",
    ].asImmutable()

    // What the map_first hierarchy fingerprint covers (COMPUTE_CPU, plan
    // §3.5): the hierarchy, QC embedding, stability and cross-platform code,
    // the label-table schema, vocabularies and provenance it reads, the
    // table-key suffix rule, the compute command and the clustering module
    // it runs through (control-feature removal and its registry, plots,
    // H5AD writing), and merxen/config.py, whose ClusteringSquidpyConfig
    // defaults set the map_first fields no Nextflow param or PREPARE config
    // writes (qc_leiden_resolution, leaf_source, adaptive_split; M5 review:
    // an edit there re-runs COMPUTE_CPU, also for unrelated fields). Like
    // RESOLVE's, it travels in the task inputs (compute_spec): a hierarchy
    // change re-runs COMPUTE_CPU under -resume, and a RESOLVE re-run with
    // byte-identical labels leaves it cached (cache "deep" on the labels).
    static final List<String> HIERARCHY_SOURCES = [
        "merxen/clustering",
        "merxen/annotation/schema.py",
        "merxen/annotation/vocab.py",
        "merxen/annotation/provenance.py",
        "merxen/annotation/config.py",
        "merxen/assets/annotation/whb_supercluster_vocab.csv",
        "merxen/assets/annotation/seaad_mr_subclass_vocab.csv",
        "merxen/assets/annotation/wmb_class_vocab.csv",
        "merxen/analysis/clustering_squidpy.py",
        "merxen/clustering_squidpy_stages.py",
        "merxen/control_features.py",
        "merxen/config.py",
    ].asImmutable()

    // One fingerprint per (source root, source list) and run: every task of
    // a run reads the same code.
    private static final Map<String, String> SOURCE_FINGERPRINTS = new ConcurrentHashMap<String, String>()

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
            // The resolvability self-map (M3b) maps the WHB held-out donor's
            // cells onto SEA-AD too, so it needs the held-out test set's
            // sources: the region directory (region_cell_metadata.csv), the
            // raw WHB h5ads and the WHB taxonomy tables.
            whb_region_dir: "annotation_whb_region_precompute_source",
            whb_h5ad_dir: "annotation_whb_h5ad_dir",
            whb_metadata_dir: "annotation_whb_metadata_dir",
        ].asImmutable(),
        wmb_panel: [
            wmb_h5ad_dir: "annotation_wmb_h5ad_dir",
            wmb_metadata_dir: "annotation_wmb_metadata_dir",
            wmb_mapping_stats: "annotation_wmb_mapping_stats_path",
            // Self-map test cells kept out of the marker build (plan §3.2);
            // a copy matching the pinned sha256 is seeded into
            // <store>/.downloads and hashed there.
            wmb_selfmap_test_cells: "annotation_wmb_selfmap_test_cells_path",
            // Optional override of the marker universe; by default a panel
            // inside the validated ag7 | VZG2 union uses it (plan §7.1).
            wmb_marker_gene_universe: "annotation_wmb_marker_gene_universe_path",
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
     * Return the annotation_config.json content ANNOTATE_PANEL, PREP and MAP read.
     *
     * References are named by id; merxen.annotation.config expands them to
     * their known specs, so the taxonomy settings live in Python only. The
     * settings only RESOLVE reads are added by resolveConfig, so changing one
     * never changes these tasks' scripts (-resume keeps MAP cached).
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
        def geneTables = [:]
        ["human", "mouse"].each { tableSpecies ->
            def path = pathText(params?.get("annotation_${tableSpecies}_gene_table".toString()))
            if (path) {
                geneTables[tableSpecies] = absolute(path)
            }
        }
        if (geneTables) {
            panel.gene_tables = geneTables
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
     * Return the annotation_config.json content RESOLVE reads.
     *
     * annotationConfig plus the settings no earlier task reads: the human
     * degraded mode annotation_allow_single_method (plan §5.3). Thresholds,
     * floors, gate and flag settings are Python defaults and packaged tables,
     * which resolveRulesFingerprint covers.
     *
     * @param params Pipeline params.
     * @param species Species name or alias.
     * @return A map for merxen.annotation.config.AnnotationConfig.
     */
    static Map resolveConfig(Map params, Object species) {
        def config = annotationConfig(params, species)
        config.allow_single_method = isTrue(params?.get("annotation_allow_single_method"))
        return config
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
     * The shared tissue mask comes only from the pair's ALIGN output channel
     * (wired in M5), never from a published file looked up at channel time:
     * that lookup would race with ALIGN in the same run and could find a
     * stale mask. The seeded set-a family's curated set c needs no mask; a
     * label-free set c (other families) is refused without one when
     * requireMask is set.
     *
     * @param clusteringConfigName Staged clustering_squidpy_config.json name.
     * @param preparedDirName Staged CLUSTERING_SQUIDPY_PREPARE output name.
     * @param maskNames Staged [shared_tissue_mask.npy, registration_summary.json]
     *     names from ALIGN, or an empty list.
     * @param requireMask Refuse a label-free set c without the mask.
     * @return The panel spec map.
     */
    static Map preparedPanelSpec(
        String clusteringConfigName,
        String preparedDirName,
        List maskNames = [],
        boolean requireMask = true
    ) {
        def spec = [
            source: PREPARED_SOURCE,
            clustering_config: clusteringConfigName,
            prepared_dir: preparedDirName,
            require_shared_tissue_mask: requireMask,
        ]
        if (maskNames && maskNames.size() == 2) {
            spec.shared_tissue_mask = maskNames[0].toString()
            spec.registration_summary = maskNames[1].toString()
        }
        return spec
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
            if (spec.require_shared_tissue_mask) {
                args += ["--require-shared-tissue-mask"]
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
     *     n_panel_genes; all fixed by the key) and panel files holds the
     *     bundle's panel_genes*.json (empty for panel-independent
     *     references). Pairs sharing a key stage different copies of the
     *     panel file (sample ids, symbols); the bundle depends only on the
     *     key's IDs, so any copy builds the same bundle.
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

    /**
     * Return what CLUSTERING_SQUIDPY_ANNOTATE_MAP needs to know before it runs.
     *
     * The process reads the species and the panel status in its script and
     * the panel size in its memory directive (48 GB above 1,000 genes, plan
     * §3.3), so they travel as a value next to the staged panel directory.
     *
     * @param panelDir ANNOTATE_PANEL output directory.
     * @return [species, panel_status, reasons, n_required, n_panel_genes
     *     (largest panel of a mapped bundle, 0 without one)].
     */
    static Map mapSpec(Object panelDir) {
        def required = requiredBundles(panelDir)
        def sizes = ((required.bundles ?: []) as List)
            .findAll { Map item -> item.n_panel_genes != null }
            .collect { Map item -> item.n_panel_genes as int }
        return [
            species: AnnotationDefaults.normalizeSpecies(required.species),
            panel_status: (required.status ?: "ok").toString(),
            reasons: ((required.reasons ?: []) as List).collect { reason -> reason.toString() },
            n_required: (required.n_required ?: 0) as int,
            n_panel_genes: sizes ? sizes.max() : 0,
        ]
    }

    /**
     * Return the merxen annotate arguments of one MAP task.
     *
     * The task maps only the bundles PREP resolved (--require-bundle-refs),
     * takes min_counts and the sample platforms from the clustering config,
     * and records a refused panel instead of failing (--allow-refused-panel;
     * RESOLVE then writes statuses only).
     *
     * @param pairId Pair id.
     * @param segmentation Segmentation.
     * @param spec mapSpec result.
     * @param bundleRefs Staged bundle_ref.json paths.
     * @param cpus Task cpus (--n-processors: MapMyCells worker processes).
     * @return Shell-quoted arguments (without --annotation-config,
     *     --reuse-from and --out).
     */
    static String mapArguments(Object pairId, Object segmentation, Map spec, List bundleRefs, int cpus) {
        def args = [
            "--species", spec.species.toString(),
            "--pair-id", pairId.toString(),
            "--segmentation", segmentation.toString(),
            "--prepared-dir", "${MAP_INPUT_DIR}/clustering_prepare_out".toString(),
            "--clustering-config", "${MAP_INPUT_DIR}/clustering_squidpy_config.json".toString(),
            "--panel-dir", "${MAP_INPUT_DIR}/${PANEL_OUTPUT_DIR}".toString(),
            "--n-processors", cpus.toString(),
            "--work-dir", MAP_SCRATCH_DIR,
            "--require-bundle-refs",
            "--allow-refused-panel",
        ]
        (bundleRefs ?: []).each { ref -> args += ["--bundle-ref", ref.toString()] }
        return args.collect { arg -> shellQuote(arg) }.join(" ")
    }

    /**
     * Return what CLUSTERING_SQUIDPY_ANNOTATE_RESOLVE needs to know before it runs.
     *
     * mapSpec (species, panel status, the panel size its memory directive
     * reads) plus the annotation config RESOLVE writes and the fingerprint of
     * the RESOLVE rules. Both are task inputs, so the task hash sees them,
     * with -stub-run too: a RESOLVE-only param or a threshold, floor, trust
     * or flag change re-runs RESOLVE and leaves MAP cached.
     *
     * @param params Pipeline params.
     * @param panelDir ANNOTATE_PANEL output directory.
     * @param sourceRoot This checkout's src/ (${projectDir}/../src).
     * @return mapSpec(panelDir) + [annotation_config, rules_fingerprint].
     */
    static Map resolveSpec(Map params, Object panelDir, Object sourceRoot) {
        def spec = mapSpec(panelDir)
        return spec + [
            annotation_config: resolveConfig(params, spec.species),
            rules_fingerprint: resolveRulesFingerprint(sourceRoot),
        ]
    }

    /** Return a resolveSpec's annotation config as pretty JSON. */
    static String resolveConfigJson(Map spec) {
        return JsonOutput.prettyPrint(JsonOutput.toJson(spec.annotation_config))
    }

    /**
     * Return the merxen annotate-resolve arguments of one RESOLVE task.
     *
     * The task reads the staged MAP output, panel, prepared H5ADs and
     * clustering config, resolves every run with exactly the bundle refs PREP
     * resolved (--require-bundle-refs: the ones MAP mapped with, no store
     * lookup), and takes the shared tissue mask only from staged ALIGN files
     * (M5), never from a published align_out ALIGN may still be writing.
     *
     * @param spec resolveSpec result.
     * @param bundleRefs Staged bundle_ref.json paths.
     * @param alignmentFiles Staged [shared_tissue_mask.npy,
     *     registration_summary.json], or an empty list.
     * @return Shell-quoted arguments (without --annotation-config and --out).
     */
    static String resolveArguments(Map spec, List bundleRefs, List alignmentFiles) {
        def args = [
            "--species", spec.species.toString(),
            "--map-dir", "${RESOLVE_INPUT_DIR}/${MAP_OUTPUT_DIR}".toString(),
            "--panel-dir", "${RESOLVE_INPUT_DIR}/${PANEL_OUTPUT_DIR}".toString(),
            "--prepared-dir", "${RESOLVE_INPUT_DIR}/clustering_prepare_out".toString(),
            "--clustering-config", "${RESOLVE_INPUT_DIR}/clustering_squidpy_config.json".toString(),
            "--require-bundle-refs",
            "--no-alignment-lookup",
        ]
        (bundleRefs ?: []).each { ref -> args += ["--bundle-ref", ref.toString()] }
        if (alignmentFiles) {
            args += ["--alignment-dir", "${RESOLVE_INPUT_DIR}/align_out".toString()]
        }
        return args.collect { arg -> shellQuote(arg) }.join(" ")
    }

    /** Return RESOLVE's <pair>_resolve_summary.json name. */
    static String resolveSummaryFile(Object pairId) {
        return "${pairId}${RESOLVE_SUMMARY_SUFFIX}".toString()
    }

    /**
     * Return the fingerprint of the RESOLVE rules under a source root.
     *
     * The sha256 of each RESOLVE_RULE_SOURCES file's path (relative to the
     * root) and content, in path order; __pycache__ directories and compiled
     * files are skipped. Computed once per root and run.
     *
     * @param sourceRoot This checkout's src/ (${projectDir}/../src).
     * @return The hex digest, or "missing" when the root holds none of the
     *     sources (an installed package without its checkout).
     */
    static String resolveRulesFingerprint(Object sourceRoot) {
        return sourcesFingerprint(sourceRoot, RESOLVE_RULE_SOURCES, RESOLVE_RULE_EXCLUDES)
    }

    /**
     * Return the fingerprint of the map_first hierarchy code (HIERARCHY_SOURCES).
     *
     * @param sourceRoot This checkout's src/ (${projectDir}/../src).
     * @return The hex digest, or "missing" (see resolveRulesFingerprint).
     */
    static String hierarchyFingerprint(Object sourceRoot) {
        return sourcesFingerprint(sourceRoot, HIERARCHY_SOURCES)
    }

    /**
     * Return the fingerprint of source files under a root.
     *
     * @param sourceRoot Root the sources are relative to.
     * @param sources Files or directories (directories are walked).
     * @param excludes Relative path prefixes of walked files to leave out.
     * @return The hex digest of each file's relative path and content, in
     *     path order, or "missing" when none exists.
     */
    static String sourcesFingerprint(Object sourceRoot, List<String> sources, List<String> excludes = []) {
        def root = asPath(sourceRoot).toAbsolutePath().normalize()
        def key = "${root}|${sources.join(',')}|${excludes.join(',')}".toString()
        return SOURCE_FINGERPRINTS.computeIfAbsent(key) { String ignored -> fingerprint(root, sources, excludes) }
    }

    private static String fingerprint(Path root, List<String> sources, List<String> excludes) {
        def files = [:] as TreeMap<String, Path>
        sources.each { String source ->
            def path = root.resolve(source)
            if (Files.isRegularFile(path)) {
                files[source] = path
            } else if (Files.isDirectory(path)) {
                path.toFile().eachFileRecurse(groovy.io.FileType.FILES) { File file ->
                    def relative = root.relativize(file.toPath()).toString().replace(File.separator, "/")
                    def compiled = relative.endsWith(".pyc") || relative.endsWith(".pyo")
                    def excluded = excludes.any { String prefix -> relative.startsWith(prefix) }
                    if (!compiled && !excluded && !relative.split("/").contains("__pycache__")) {
                        files[relative] = file.toPath()
                    }
                }
            }
        }
        if (!files) {
            return "missing"
        }
        def digest = MessageDigest.getInstance("SHA-256")
        files.each { relative, path ->
            digest.update(relative.getBytes(StandardCharsets.UTF_8))
            digest.update((byte) 0)
            digest.update(Files.readAllBytes(path))
            digest.update((byte) 0)
        }
        return digest.digest().encodeHex().toString()
    }

    /**
     * Return what CLUSTERING_SQUIDPY_COMPUTE_CPU needs to know before it runs.
     *
     * The map_first rows' table-key suffix and MENDER unassigned-state policy (both
     * recorded in each clustered table, which FINALIZE and MENDER_PREPARE
     * read: their scripts are the legacy ones) and the fingerprint of the
     * hierarchy code. All are task inputs, so -resume sees them.
     *
     * @param params Pipeline params.
     * @param sourceRoot This checkout's src/ (${projectDir}/../src).
     * @return [table_key_suffix, mender_unassigned_state_policy,
     *     hierarchy_fingerprint].
     */
    static Map computeSpec(Map params, Object sourceRoot) {
        def species = AnnotationDefaults.normalizeSpecies(params?.get("species"))
        // COMPUTE_CPU runs for map_first rows only, so its suffix is that of a
        // map_first row, also in a run whose own mode is legacy and that a
        // row's clustering_squidpy_mode column opts in (§20 D15): forRow
        // gives the row the same suffix.
        def mapFirstParams = AnnotationDefaults.withRowMode(AnnotationDefaults.MAP_FIRST, params)
        return [
            table_key_suffix: AnnotationDefaults.tableKeySuffix(mapFirstParams, species),
            mender_unassigned_state_policy: textOr(
                params?.get("mender_unassigned_state_policy"),
                AnnotationSettings.MAP_FIRST_MENDER_UNASSIGNED_STATE_POLICY,
            ).toLowerCase(),
            hierarchy_fingerprint: hierarchyFingerprint(sourceRoot),
        ]
    }

    /**
     * Return the compute arguments of one COMPUTE_CPU task (after --config,
     * --input-dir and --output-dir).
     *
     * @param spec computeSpec result.
     * @return Shell-quoted arguments.
     */
    static String computeArguments(Map spec) {
        def args = [
            "--mode", AnnotationDefaults.MAP_FIRST,
            "--labels-dir", "${COMPUTE_INPUT_DIR}/${RESOLVE_OUTPUT_DIR}".toString(),
            "--table-key-suffix", spec.table_key_suffix.toString(),
            "--mender-unassigned-state-policy", spec.mender_unassigned_state_policy.toString(),
        ]
        return args.collect { arg -> shellQuote(arg) }.join(" ")
    }

    /**
     * Return the stub compute manifest of a COMPUTE_CPU task (for -stub-run).
     *
     * @param pairId Pair id.
     * @param segmentation Segmentation.
     * @param spec computeSpec result.
     * @param samplesJson The samples JSON the task received.
     * @return The JSON text.
     */
    static String stubComputeManifestJson(Object pairId, Object segmentation, Map spec, Object samplesJson) {
        return JsonOutput.prettyPrint(JsonOutput.toJson([
            stub: true,
            step: "compute_cpu",
            pair_id: pairId.toString(),
            segmentation: segmentation.toString(),
            mode: AnnotationDefaults.MAP_FIRST,
            table_key_suffix: spec.table_key_suffix,
            mender_unassigned_state_policy: spec.mender_unassigned_state_policy,
            hierarchy_fingerprint: spec.hierarchy_fingerprint,
            samples: new JsonSlurperClassic().parseText(samplesJson.toString()),
        ]))
    }

    /**
     * Return the RESOLVE outputs COMPUTE_CPU stages, as files (plan §3.4, §3.5).
     *
     * The pair summary and each sample's label table and annotation manifest,
     * staged flat into compute_inputs/annotation_resolve_out/ (sample ids
     * are unique within a pair; merxen.clustering.map_first reads the flat
     * layout). Files, not the directory: cache "deep" hashes a file's
     * content but a directory only by its metadata, so a RESOLVE re-run
     * with byte-identical outputs would otherwise re-run COMPUTE_CPU. The
     * copied annotation_config.json is left out: a RESOLVE setting reaches
     * COMPUTE_CPU only through the labels, manifests or summary it changes.
     *
     * @param resolveDir RESOLVE's annotation_resolve_out.
     * @return The files, sorted by name.
     */
    static List computeLabelFiles(Object resolveDir) {
        def root = asPath(resolveDir)
        def files = []
        Files.list(root).withCloseable { entries ->
            entries.each { Path entry ->
                def name = entry.fileName.toString()
                if (Files.isRegularFile(entry) && name.endsWith(RESOLVE_SUMMARY_SUFFIX)) {
                    files << entry
                } else if (Files.isDirectory(entry)) {
                    Files.list(entry).withCloseable { children ->
                        children.each { Path child ->
                            def childName = child.fileName.toString()
                            if (Files.isRegularFile(child) && LABEL_FILE_SUFFIXES.any { suffix -> childName.endsWith(suffix) }) {
                                files << child
                            }
                        }
                    }
                }
            }
        }
        return files.sort { Path path -> path.fileName.toString() }
    }

    /**
     * Return the ALIGN files a pair's map_first annotation reads (plan §3.2, §3.4).
     *
     * The pair's alignment comes from alignment_results_ch: the ALIGN task
     * of this run, or the published align_out when ALIGN does not run in
     * it, so the files never race with an ALIGN still writing them.
     *
     * @param alignOut The pair's align_out directory.
     * @return [shared_tissue_mask.npy, registration_summary.json] when both
     *     exist, else an empty list.
     */
    static List alignmentFiles(Object alignOut) {
        if (alignOut == null) {
            return []
        }
        def directory = asPath(alignOut)
        def files = [SHARED_TISSUE_MASK_FILE, REGISTRATION_SUMMARY_FILE].collect { name -> directory.resolve(name) }
        return files.every { path -> Files.isRegularFile(path) } ? files : []
    }

    /**
     * Return the stub <pair>_resolve_summary.json of a RESOLVE task (for -stub-run).
     *
     * @param pairId Pair id.
     * @param segmentation Segmentation.
     * @param spec resolveSpec result.
     * @param bundleRefs Staged bundle_ref.json paths.
     * @param alignmentFiles Staged ALIGN files.
     * @return The JSON text.
     */
    static String stubResolveSummaryJson(
        Object pairId,
        Object segmentation,
        Map spec,
        List bundleRefs,
        List alignmentFiles
    ) {
        return JsonOutput.prettyPrint(JsonOutput.toJson([
            stub: true,
            step: "annotate_resolve",
            pair_id: pairId.toString(),
            segmentation: segmentation.toString(),
            species: spec.species,
            panel_status: spec.panel_status,
            n_required: spec.n_required,
            rules_fingerprint: spec.rules_fingerprint,
            // As the real summary's thresholds and flags config: a RESOLVE
            // setting changes the summary COMPUTE_CPU stages.
            annotation_config: spec.annotation_config,
            bundle_refs: (bundleRefs ?: []).collect { ref -> ref.toString() },
            alignment_files: (alignmentFiles ?: []).collect { item -> item.toString() },
            samples: [:],
        ]))
    }

    /**
     * Return the published MAP directory whose identical runs a task reuses.
     *
     * dwight prunes work directories, so -resume alone cannot skip a MAP
     * whose inputs are unchanged but whose task directory is gone; merxen
     * annotate then copies each run whose query fingerprint, build_hash,
     * engine parameters and ctm version match the published manifest
     * (annotation_reuse_published, plan §3.1).
     *
     * @param params Pipeline params.
     * @param pairId Pair id.
     * @param segmentation Segmentation.
     * @return The absolute published annotation_map_out directory, or null
     *     when annotation_reuse_published is false.
     */
    static String mapReuseDir(Map params, Object pairId, Object segmentation) {
        if (!isTrue(params?.get("annotation_reuse_published"), true)) {
            return null
        }
        return absolute(
            Paths.get(
                (params?.get("outdir") ?: "results").toString(),
                pairId.toString(),
                segmentation.toString(),
                MAP_PUBLISH_DIR,
                MAP_OUTPUT_DIR,
            ).toString()
        )
    }

    /**
     * Return the stub map_manifest.json of a MAP task (for -stub-run).
     *
     * @param pairId Pair id.
     * @param segmentation Segmentation.
     * @param spec mapSpec result.
     * @param bundleRefs Staged bundle_ref.json paths (the stub copies them
     *     next to the manifest, so a stub run shows what MAP received).
     * @return The JSON text.
     */
    static String stubMapManifestJson(Object pairId, Object segmentation, Map spec, List bundleRefs) {
        return JsonOutput.prettyPrint(JsonOutput.toJson([
            stub: true,
            pair_id: pairId.toString(),
            segmentation: segmentation.toString(),
            species: spec.species,
            panel_status: spec.panel_status,
            panel_reasons: spec.reasons,
            n_required: spec.n_required,
            bundle_refs: (bundleRefs ?: []).collect { ref -> ref.toString() },
            samples: [:],
        ]))
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

import groovy.json.JsonOutput
import groovy.json.JsonSlurper

/** Optional managed annotation training and its dependency on segmentation. */
class CellposeHitlSettings {
    static boolean enabled(params) {
        return params.cellpose_hitl_training_config != null &&
            params.cellpose_hitl_training_config.toString().trim()
    }

    static Map specification(params) {
        if (!enabled(params)) return [:]
        def configFile = new File(params.cellpose_hitl_training_config.toString())
        if (!configFile.isFile()) throw new IllegalArgumentException('Missing --cellpose_hitl_training_config JSON file')
        def spec = new LinkedHashMap(new JsonSlurper().parse(configFile))
        def rootValue = spec.hitl_root ?: params.cellpose_hitl_root
        if (!rootValue) throw new IllegalArgumentException('Set MERXEN_HITL_ROOT, --cellpose_hitl_root or hitl_root in the training config')
        def root = new File(rootValue.toString())
        if (!root.isAbsolute()) throw new IllegalArgumentException('HITL root must be absolute')
        root = root.canonicalFile
        spec.model_name = params.cellpose_hitl_model_name ?: spec.model_name
        spec.species = spec.species ?: params.species
        if (!(spec.model_name ==~ /[a-z][a-z0-9_-]{2,63}/) ||
            spec.model_name in ['cpsam', 'cpsam_v2']) {
            throw new IllegalArgumentException('Choose a unique custom model_name in the training config')
        }
        if (!(spec.platform in ['ATERA', 'XENIUM']) || !(spec.species in ['human', 'mouse'])) {
            throw new IllegalArgumentException('Training config requires its platform and species')
        }
        if (!(spec.annotation_package_ids instanceof Collection) || !spec.annotation_package_ids ||
            spec.annotation_package_ids.unique(false).size() != spec.annotation_package_ids.size()) {
            throw new IllegalArgumentException('Select distinct annotation_package_ids in the training config')
        }
        def annotationRoot = new File(root, 'annotations').canonicalFile
        def packagePaths = spec.annotation_package_ids.collect { id ->
            if (!(id ==~ /hitl-[A-Za-z0-9_-]{1,80}/)) throw new IllegalArgumentException('Invalid annotation package ID')
            def directory = new File(annotationRoot, id.toString()).canonicalFile
            if (!directory.toPath().startsWith(annotationRoot.toPath()) || !directory.isDirectory()) {
                throw new IllegalArgumentException("Missing returned annotation package: ${id}")
            }
            directory.toString()
        }
        spec.hitl_root = root.toString()
        spec.package_dirs = spec.annotation_package_ids
        if (!spec.containsKey('gpu')) spec.gpu = params.cellpose_gpu
        if (!spec.containsKey('protected_output_roots')) {
            spec.protected_output_roots = params.cellpose_protected_output_roots ?: []
        }
        return [config: spec, package_paths: packagePaths]
    }

    static void validateRows(Collection entries, Map specification, params) {
        def spec = specification.config
        def protectedPaths = CellposeSettings.protectedPaths(spec.protected_output_roots ?: [])
        if (CellposeSettings.isProtected(new File(params.outdir.toString()), protectedPaths)) {
            throw new IllegalArgumentException('Training/inference needs a separate HITL outdir outside protected baselines')
        }
        def matching = entries.findAll { entry ->
            entry[2].active_platforms.contains(spec.platform) &&
                entry[2].selected_stages.contains('train_cellpose')
        }
        if (!matching) throw new IllegalArgumentException('Training requires a matching platform row selecting train_cellpose')
        matching.each { entry ->
            def settings = entry[2]
            if (settings.species != spec.species || settings.segmentation_source != 'cellpose_proseg') {
                throw new IllegalArgumentException('Training row must match species and use cellpose_proseg')
            }
            def old = CellposeSettings.value(entry[1], params, "${spec.platform.toLowerCase()}_cellpose_model")
            if (old != null && old.toString().trim()) {
                throw new IllegalArgumentException('Select either training_config or an existing named model, not both')
            }
        }
    }

    static Map applyTrainedModel(Map row, Map settings, Map selector, params) {
        if (!settings.active_platforms.contains(selector.platform) || !settings.run_segment ||
            !settings.selected_stages.contains('train_cellpose')) return row
        def prefix = selector.platform.toLowerCase()
        def updated = new LinkedHashMap(row)
        updated["${prefix}_cellpose_model"] = selector.model_name
        updated["${prefix}_cellpose_model_store"] = new File(selector.bundle_path.toString()).parent
        updated["${prefix}_allow_candidate_model"] = true
        CellposeSettings.preflight(updated, settings, params)
        return updated
    }

    static String configJson(Map specification) {
        return JsonOutput.prettyPrint(JsonOutput.toJson(specification.config))
    }

    static Map seedConfig(pairId, String platform, Map row, Map settings, params) {
        if (!settings.need_build_results || settings.run_build) return [:]
        def direct = row["${platform.toLowerCase()}_hitl_source_zarr"] ?: params.cellpose_hitl_source_zarr
        if (params.cellpose_hitl_source_zarr && settings.active_platforms.size() != 1) {
            throw new IllegalArgumentException('For paired runs use per-platform hitl_source_zarr columns or a source outdir')
        }
        def sourceRoot = params.cellpose_hitl_source_outdir
        if (!direct && !sourceRoot) return [:]
        def relative = "${pairId}/${platform.toLowerCase()}"
        def source = direct ? new File(direct.toString()) : new File(sourceRoot.toString(), "${relative}/spatialdata/spatialdata_out/source_spatialdata.zarr")
        if (!source.isDirectory()) throw new IllegalArgumentException('The processed HITL source SpatialData store does not exist')
        def result = [source_zarr: source.canonicalPath,
            destination_zarr: new File(params.outdir.toString(), "${relative}/spatialdata/spatialdata_out/source_spatialdata.zarr").absolutePath,
            protected_output_roots: params.cellpose_protected_output_roots ?: []]
        if (platform != 'ATERA' && !settings.run_segment_nuclei && settings.run_segment) {
            if (!sourceRoot) throw new IllegalArgumentException('Late Xenium/MERSCOPE starts require cellpose_hitl_source_outdir for the completed stock nuclei')
            def nuclei = new File(sourceRoot.toString(), "${relative}/segmentation")
            if (!new File(nuclei, 'cellpose_nuclei_masks_tiled.npy').isFile()) {
                throw new IllegalArgumentException('The source run is missing its stock nuclei mask')
            }
            result.nuclei_source_dir = nuclei.canonicalPath
            result.nuclei_destination_dir = new File(params.outdir.toString(), "${relative}/segmentation").absolutePath
        }
        return result
    }
}

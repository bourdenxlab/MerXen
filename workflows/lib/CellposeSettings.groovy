import groovy.json.JsonSlurper
import groovy.json.JsonOutput
import java.security.MessageDigest
import java.nio.file.Files

/** Opt-in model settings; empty settings never alter legacy segment JSON. */
class CellposeSettings {
    static final String STOCK_SHA256 = '0f1cc3f7ecdd8a037a57c6c48d9d8921391be4cbce3fa9f13c3e3a2e1253c667'

    static Map geometryFilter(params) {
        def enabled = params.cellpose_geometry_filter_enabled
        if (!(enabled.toString().toLowerCase() in ['true', 'false'])) {
            throw new IllegalArgumentException('--cellpose_geometry_filter_enabled must be true or false')
        }
        if (enabled.toString().toLowerCase() == 'false') return null
        return [max_area_um2: params.cellpose_geometry_max_area_um2,
            max_thickness_um: params.cellpose_geometry_max_thickness_um,
            max_solidity: params.cellpose_geometry_max_solidity]
    }

    static Map stockComparisonConfig(Map configured) {
        def config = new JsonSlurper().parseText(JsonOutput.toJson(configured))
        if (config.dataset.platform != 'ATERA') {
            throw new IllegalArgumentException('Stock comparison is an ATERA route')
        }
        config.cellpose.putAll([pretrained_model: 'cpsam_v2', model_sha256: null,
            model_metadata_path: null])
        config.dataset.name = "${config.dataset.name}_STOCK"
        config.dataset.findAll { name, value ->
            name.startsWith('persistent_') && value instanceof String &&
                value.contains('/segmentation/')
        }.each { name, value ->
            config.dataset[name] = value.replace('/segmentation/', '/segmentation/stock_cellpose/')
        }
        config.cellpose_model_policy = (config.cellpose_model_policy ?: [:]) + [experimental_arm: true]
        return config
    }

    static Map selectedBundle(row, String platform, params) {
        def selected = value(row, params, "${platform.toLowerCase()}_cellpose_model")
        if (selected == null || !selected.toString().trim()) return [:]
        def requested = selected.toString().trim()
        def named = false
        File weights
        if (requested ==~ /[0-9a-f]{64}/) {
            if (!params.cellpose_model_store) {
                throw new IllegalArgumentException('Registry hashes require --cellpose_model_store')
            }
            weights = new File(params.cellpose_model_store.toString(), "${requested}/model")
        } else if (requested ==~ /[a-z][a-z0-9_-]{2,63}/) {
            def stores = []
            def rowStore = row["${platform.toLowerCase()}_cellpose_model_store"]
            if (rowStore) {
                stores << new File(rowStore.toString())
            } else {
                if (params.cellpose_model_store) stores << new File(params.cellpose_model_store.toString())
                if (params.cellpose_hitl_root) stores << new File(params.cellpose_hitl_root.toString(), 'models')
            }
            def matches = stores.collect { new File(it, "${requested}/model").canonicalFile }
                .findAll { it.isFile() }.unique { it.path }
            if (matches.size() > 1) throw new IllegalArgumentException('Model name is ambiguous across configured stores; select its full bundle path')
            if (matches) {
                weights = matches[0]
                named = true
            } else {
                weights = new File(requested)
            }
        } else {
            weights = new File(requested)
        }
        if (weights.isDirectory() || weights.name == 'model.json') {
            weights = new File(weights.isDirectory() ? weights : weights.parentFile, 'model')
        }
        weights = new File(weights.absoluteFile.parentFile.canonicalFile, weights.name)
        def metadata = new File(weights.parentFile, 'model.json').canonicalFile
        if (!weights.isFile() || !metadata.isFile()) {
            throw new IllegalArgumentException("Custom model requires weights and model.json: ${requested}")
        }
        return [weights: weights, metadata: metadata, named: named,
                model: new JsonSlurper().parse(metadata)]
    }

    static List channelsForRow(row, platform, params = null) {
        def spec = PlatformSpec.forName(platform)
        def explicit = row["${platform.toLowerCase()}_channels"]
        if (explicit != null && explicit.toString().trim()) {
            return explicit.toString().split(',').collect { it.trim() }.findAll { it }
        }
        if (params != null) {
            def bundle = selectedBundle(row, platform, params)
            if (bundle.named && bundle.model.channels) return bundle.model.channels
        }
        if (platform == 'ATERA' && params != null) {
            return AteraDensitySettings.channels(row, params)
        }
        def raw = row["${platform.toLowerCase()}_channels"]
        return raw != null && raw.toString().trim()
            ? raw.toString().split(',').collect { it.trim() }.findAll { it }
            : spec.defaultChannels
    }
    static Set protectedPaths(Collection roots) {
        def protectedPaths = [] as Set
        roots.each { name ->
            def root = new File(name.toString())
            protectedPaths.add(root.absoluteFile.toPath().normalize())
            protectedPaths.add(root.canonicalFile.toPath())
            def pending = root.isDirectory() ? [root] : []
            while (pending) {
                def directory = pending.remove(pending.size() - 1)
                def entries = directory.listFiles()
                if (entries == null) {
                    throw new IllegalArgumentException("Cannot inspect protected root: ${directory}")
                }
                entries.each { entry ->
                    if (Files.isSymbolicLink(entry.toPath())) {
                        protectedPaths.add(entry.canonicalFile.toPath())
                    } else if (entry.isDirectory() && !entry.name.endsWith('.zarr')) {
                        pending.add(entry)
                    }
                }
            }
        }
        return protectedPaths
    }

    static boolean isProtected(File file, Collection roots) {
        def absolute = file.absoluteFile.toPath().normalize()
        def resolved = file.canonicalFile.toPath()
        def ancestors = []
        def parent = file.parentFile
        while (parent != null) {
            ancestors.add(parent.canonicalFile.toPath())
            parent = parent.parentFile
        }
        return roots.any { root ->
            absolute.startsWith(root) || resolved.startsWith(root) ||
                root.startsWith(resolved) || ancestors.any { it.startsWith(root) }
        }
    }

    static Object value(row, params, String name) {
        def rowValue = row[name]
        return rowValue != null && rowValue.toString().trim()
            ? rowValue.toString().trim() : params[name]
    }

    static boolean matchesFields(Map actual, Map expected) {
        return actual != null && actual.keySet() == expected.keySet() &&
            expected.every { key, value -> actual[key] == value }
    }

    static void requirePublishedIdentity(rowSettings, platform, params, selected, metadata, channels) {
        def directory = new File(params.outdir.toString(),
            "${rowSettings.pair_id}/${platform.toLowerCase()}/segmentation")
        def mask = new File(directory, 'cellpose_masks_tiled.npy')
        def sidecar = new File(directory, 'cellpose_model_identity.json')
        if (!sidecar.isFile() && mask.isFile()) {
            sidecar = new File(mask.canonicalFile.parentFile, 'cellpose_model_identity.json')
        }
        def stats = new File(directory, 'cellpose_stitching_stats.json')
        if (!stats.isFile()) {
            stats = new File(sidecar.parentFile, 'cellpose_stitching_stats.json')
        }
        if (!mask.isFile() || !sidecar.isFile() || !stats.isFile()) {
            throw new IllegalArgumentException('Downstream custom-model run requires published masks and verified provenance')
        }
        def identity = new JsonSlurper().parse(sidecar)
        def statsIdentity = new JsonSlurper().parse(stats).model_identity
        def isStock = metadata == null
        if (isStock) {
            metadata = [weights_sha256: STOCK_SHA256, cellpose_version: '4.2.1.1',
                torch_version: identity.torch_version, channels: channels,
                image_key: PlatformSpec.forName(platform).nativeImageKey]
        }
        def expectedCellpose = [
            pretrained_model: 'cpsam_v2', model_sha256: null, model_metadata_path: null,
            model_type: params.cellpose_model_type, gpu: params.cellpose_gpu,
            diameter: params.cellpose_diameter, flow_threshold: params.cellpose_flow_threshold,
            cellprob_threshold: params.cellpose_cellprob, tile_overlap: params.cellpose_tile_overlap,
            bsize: params.cellpose_bsize, factor_rescale: 1.0,
            use_bfloat16: params.cellpose_use_bfloat16,
            input_normalization: platform == 'ATERA' && AteraDensitySettings.enabled(params)
                ? 'per_channel' : 'joint',
        ] + selected.cellpose
        def recordedCellpose = [input_normalization: 'joint'] + (identity.cellpose_config ?: [:])
        def expectedFilter = [
            max_eccentricity: 0.99, min_area_percentile: 1.0, min_area_px: null,
            final_min_area_um2: params.cellpose_final_min_area_um2,
            final_max_area_um2: params[PlatformSpec.forName(platform).cellposeMaxAreaParam],
            final_filter_chunk_mb: params.cellpose_final_filter_chunk_mb,
            geometry_filter: geometryFilter(params),
            n_jobs: 16, show_progress: true,
        ] + selected.mask_filter
        if (expectedFilter.geometry_filter == null) expectedFilter.remove('geometry_filter')
        def expectedTiling = [
            tile_size_candidates: params.cellpose_tile_size_candidates,
            stitch_overlap_px: params.cellpose_stitch_overlap_px,
            min_tile_size: params.cellpose_min_tile_size,
            status_every_tiles: params.cellpose_stitch_status_every_tiles,
            filter_per_tile: params.cellpose_filter_per_tile,
            duplicate_iou_threshold: params.cellpose_duplicate_iou_threshold,
            duplicate_overlap_fraction: params.cellpose_duplicate_overlap_fraction,
            min_remaining_fraction: params.cellpose_min_remaining_fraction,
            edge_touch_policy: params.cellpose_edge_touch_policy,
            write_stitching_stats: params.cellpose_write_stitching_stats,
        ]
        expectedTiling.putAll(selected.tiling ?: [:])
        def context = identity.inference_context
        def validPixels = context?.pixel_size_um?.size() == 2 && context.pixel_size_um.every {
            it instanceof Number && Double.isFinite(it as double) && it > 0 &&
                (isStock || Math.abs((it as double) - (metadata.pixel_size_um as double)) <=
                    (metadata.pixel_size_um as double) * 0.01 + 1e-12)
        }
        if (identity.status != 'verified' || identity.weights_sha256 != metadata.weights_sha256 ||
            identity.model_name != expectedCellpose.pretrained_model ||
            identity.cellpose_version != metadata.cellpose_version ||
            identity.torch_version != metadata.torch_version ||
            (isStock && !(identity.torch_version instanceof String && identity.torch_version.startsWith('2.14.'))) ||
            !matchesFields(recordedCellpose, expectedCellpose) ||
            context == null || context.platform != platform || context.species != rowSettings.species ||
            context.channels != metadata.channels || context.image_key != metadata.image_key ||
            (platform == 'ATERA' && context.transcript_density?.profile !=
                AteraDensitySettings.selectedProfile(selected, params)) ||
            !validPixels ||
            !matchesFields(context.mask_filter, expectedFilter) ||
            !matchesFields(context.tiling, expectedTiling) || identity != statsIdentity) {
            throw new IllegalArgumentException('Downstream cached model/configuration identity mismatch; use a separate arm')
        }
    }

    static Map settings(row, String platform, params) {
        def prefix = platform.toLowerCase()
        def selected = value(row, params, "${prefix}_cellpose_model")
        def result = [cellpose: [:], mask_filter: [:]]
        [
            cellpose_cellprob: ['cellpose', 'cellprob_threshold'],
            cellpose_flow_threshold: ['cellpose', 'flow_threshold'],
            cellpose_diameter: ['cellpose', 'diameter'],
            cellpose_final_max_area_um2: ['mask_filter', 'final_max_area_um2'],
            mask_max_eccentricity: ['mask_filter', 'max_eccentricity'],
        ].each { suffix, target ->
            def raw = value(row, params, "${prefix}_${suffix}")
            if (raw != null && raw.toString().trim()) {
                def number = Double.parseDouble(raw.toString())
                if (!Double.isFinite(number)) {
                    throw new IllegalArgumentException("${prefix}_${suffix} must be finite")
                }
                result[target[0]][target[1]] = number
            }
        }
        if (selected != null && selected.toString().trim()) {
            def requested = selected.toString().trim()
            def bundle = selectedBundle(row, platform, params)
            File weights = bundle.weights
            // Preserve a model-file symlink inside the bundle so model.json
            // remains adjacent to the requested file, rather than its cache target.
            weights = weights.absoluteFile
            weights = new File(weights.parentFile.canonicalFile, weights.name)
            def metadata = bundle.metadata
            def model = bundle.model
            if (bundle.named) {
                def inherited = (model.evaluated_cellpose ?: [:]).findAll { key, _value ->
                    key in ['diameter', 'flow_threshold', 'cellprob_threshold',
                            'input_normalization', 'use_bfloat16', 'bsize', 'factor_rescale', 'tile_overlap']
                }
                result.cellpose = inherited + result.cellpose
                result.mask_filter = (model.evaluated_mask_filter ?: [:]) + result.mask_filter
                // Older evaluated models retain their recipe; the current
                // Xenium neuron ceiling follows the pipeline unless overridden.
                def areaParam = PlatformSpec.forName(platform).cellposeMaxAreaParam
                if (areaParam != 'cellpose_final_max_area_um2' &&
                    value(row, params, "${prefix}_cellpose_final_max_area_um2") == null) {
                    result.mask_filter.final_max_area_um2 = params[areaParam]
                }
                result.tiling = model.training_tiling ?: [tile_size_candidates: [model.tile_size],
                    min_tile_size: Math.min(params.cellpose_min_tile_size as int, model.tile_size as int)]
                result.transcript_density_profile = model.transcript_density_profile
            }
            if (!(model.weights_sha256 ==~ /[0-9a-f]{64}/)) {
                throw new IllegalArgumentException("Invalid weights_sha256 in ${metadata}")
            }
            result.cellpose.putAll([
                pretrained_model: weights.toString(),
                model_sha256: model.weights_sha256,
                model_metadata_path: metadata.toString(),
            ])
            def roots = params.cellpose_protected_output_roots
            roots = roots instanceof Collection ? roots : (roots ? [roots] : [])
            def candidateKey = "${prefix}_allow_candidate_model"
            def candidate = row.containsKey(candidateKey) ? row[candidateKey] : params.allow_candidate_model
            if (!(candidate.toString().toLowerCase() in ['true', 'false'])) {
                throw new IllegalArgumentException('--allow_candidate_model must be true or false')
            }
            result.cellpose_model_policy = [
                species: params.species.toString().toLowerCase(),
                allow_candidate_model: candidate.toString().toLowerCase() == 'true',
                protected_output_roots: roots.collect { new File(it.toString()).canonicalPath },
            ]
        }
        if (result.cellpose || result.mask_filter) {
            // The current cleanup policy also applies to older named models.
            result.mask_filter.geometry_filter = geometryFilter(params)
            if (!result.cellpose_model_policy) {
                def roots = params.cellpose_protected_output_roots
                roots = roots instanceof Collection ? roots : (roots ? [roots] : [])
                result.cellpose_model_policy = [species: params.species.toString().toLowerCase(),
                    protected_output_roots: roots.collect { new File(it.toString()).canonicalPath }]
            }
            result.cellpose_model_policy.experimental_arm = true
            return result
        }
        return [:]
    }

    static void preflight(row, rowSettings, params) {
        def needsPublishedMasks = rowSettings.run_segment == false &&
            rowSettings.stage_order.indexOf(rowSettings.start_stage) >
                rowSettings.stage_order.indexOf('segment')
        rowSettings.active_platforms.each { platform ->
            def selected = settings(row, platform, params)
            if (platform == 'ATERA' && rowSettings.segmentation_source == 'cellpose_comparison' &&
                !selected.cellpose?.model_metadata_path) {
                throw new IllegalArgumentException('cellpose_comparison requires a retrained --atera_cellpose_model')
            }
            if (!selected.cellpose_model_policy) {
                return
            }
            if (platform == 'ATERA' && rowSettings.segmentation_source != null &&
                !(rowSettings.segmentation_source in ['cellpose_proseg', 'cellpose_comparison'])) {
                throw new IllegalArgumentException(
                    'ATERA Cellpose model/overrides require segmentation_source=cellpose_proseg; ' +
                    'native and native_proseg use vendor masks and do not run Cellpose')
            }
            def policy = selected.cellpose_model_policy
            if (!policy.protected_output_roots) {
                throw new IllegalArgumentException('Custom models require --cellpose_protected_output_roots')
            }
            def outdir = new File(params.outdir.toString()).canonicalFile.toPath()
            def datasetDir = new File(params.outdir.toString(),
                "${rowSettings.pair_id}/${platform.toLowerCase()}").canonicalFile.toPath()
            def sourceOutput = new File(datasetDir.toFile(),
                'spatialdata/spatialdata_out/source_spatialdata.zarr').canonicalFile.toPath()
            def sourceValue = row["${platform.toLowerCase()}_spatialdata_path"] ?:
                row["${platform.toLowerCase()}_zarr_path"]
            // A late-start seed reads the baseline but writes an independent
            // sourceOutput copy, whose destination is checked below.
            if (!rowSettings.run_build && rowSettings.need_build_results &&
                (params.cellpose_hitl_source_outdir || params.cellpose_hitl_source_zarr ||
                 row["${platform.toLowerCase()}_hitl_source_zarr"])) sourceValue = null
            def sourceInput = sourceValue?.toString()?.trim()
                ? new File(sourceValue.toString()).canonicalFile.toPath() : null
            def protectedTargets = protectedPaths(policy.protected_output_roots)
            def existingArmTargets = protectedPaths([params.outdir.toString()])
            if (([outdir, datasetDir, sourceOutput, sourceInput] + existingArmTargets).findAll { it != null }
                .any { isProtected(it.toFile(), protectedTargets) }) {
                throw new IllegalArgumentException('Custom-model arm targets a protected baseline')
            }
            if (!selected.cellpose.model_metadata_path) {
                if (params.cellpose_tile_size_candidates.size() != 1 ||
                    (params.cellpose_tile_size_candidates[0] as int) < (params.cellpose_min_tile_size as int)) {
                    throw new IllegalArgumentException('Stock controls require a single pinned tile size')
                }
                if (needsPublishedMasks) {
                    requirePublishedIdentity(rowSettings, platform, params, selected, null,
                        channelsForRow(row, platform, params))
                }
                return
            }
            def metadata = new JsonSlurper().parse(new File(selected.cellpose.model_metadata_path))
            def nativeImage = PlatformSpec.forName(platform).nativeImageKey
            if (metadata.normalization_route != 'A' || metadata.image_key != nativeImage ||
                !(metadata.pixel_size_um instanceof Number) ||
                !Double.isFinite(metadata.pixel_size_um as double) || metadata.pixel_size_um <= 0) {
                throw new IllegalArgumentException('Model requires native image, positive pixel size and normalization route A')
            }
            if (metadata.platform != platform || metadata.species != rowSettings.species) {
                throw new IllegalArgumentException('Model platform/species does not match the row')
            }
            def channels = channelsForRow(row, platform, params)
            if (metadata.channels != channels) {
                throw new IllegalArgumentException('Model channel names/order do not match the row')
            }
            if (platform == 'ATERA') {
                def profile = AteraDensitySettings.selectedProfile(selected, params)
                def normalization = selected.cellpose.input_normalization ?: (profile != null ? 'per_channel' : 'joint')
                if (metadata.transcript_density_profile != profile ||
                    (metadata.evaluated_cellpose?.input_normalization ?: 'joint') != normalization) {
                    throw new IllegalArgumentException('Model density recipe/input normalization does not match the requested Atera input')
                }
            }
            if (!(metadata.status == 'accepted' ||
                (metadata.status == 'candidate' && policy.allow_candidate_model))) {
                throw new IllegalArgumentException('Model requires accepted status or candidate opt-in')
            }
            def candidates = (selected.tiling?.tile_size_candidates ?: params.cellpose_tile_size_candidates).collect { it as int }
            if (candidates != [metadata.tile_size as int] ||
                (metadata.tile_size as int) < ((selected.tiling?.min_tile_size ?: params.cellpose_min_tile_size) as int)) {
                throw new IllegalArgumentException('Custom models require a single evaluated tile size')
            }
            def digest = MessageDigest.getInstance('SHA-256')
            new File(selected.cellpose.pretrained_model).withInputStream { stream ->
                byte[] buffer = new byte[1024 * 1024]
                int count
                while ((count = stream.read(buffer)) != -1) {
                    digest.update(buffer, 0, count)
                }
            }
            if (digest.digest().encodeHex().toString() != metadata.weights_sha256) {
                throw new IllegalArgumentException('Custom model weights checksum mismatch')
            }
            if (needsPublishedMasks) {
                requirePublishedIdentity(rowSettings, platform, params, selected, metadata, channels)
            }
        }
    }
}

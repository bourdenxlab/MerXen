/** Atera-only pre-segmentation density settings; native routes never call it. */
class AteraDensitySettings {
    static boolean enabled(params) {
        def raw = params.atera_transcript_density_enabled
        if (raw == null) { return true }
        def value = raw.toString().toLowerCase()
        if (!(value in ['true', 'false'])) {
            throw new IllegalArgumentException('atera_transcript_density_enabled must be true or false')
        }
        return value == 'true'
    }

    static List channels(row, params) {
        def raw = row.atera_channels
        def selected = raw != null && raw.toString().trim()
            ? raw.toString().split(',').collect { it.trim() }.findAll { it }
            : PlatformSpec.forName('ATERA').defaultChannels
        if (!enabled(params)) { return selected }
        def stains = PlatformSpec.forName('ATERA').defaultChannels
        def expected = stains + ['transcript_density']
        if (!(selected == stains || selected == expected)) {
            throw new IllegalArgumentException('Atera density requires DAPI,18S,transcript_density; disable density for other channel sets')
        }
        return expected
    }

    static boolean enabledForMeta(meta, params) {
        def selected = meta.cellpose_settings ?: [:]
        return selected.containsKey('transcript_density_profile')
            ? selected.transcript_density_profile != null : enabled(params)
    }

    static Map selectedProfile(selected, params) {
        return selected.containsKey('transcript_density_profile')
            ? selected.transcript_density_profile : (enabled(params) ? profile(params) : null)
    }

    static Map config(source, params, requestedProfile = null) {
        def result = [source_zarr_path: source.toString(), output_dir: 'density_out',
            min_qv: params.atera_transcript_density_min_qv == null ? 20.0 : params.atera_transcript_density_min_qv as double,
            sigma_um: params.atera_transcript_density_sigma_um == null ? 0.15 : params.atera_transcript_density_sigma_um as double,
            bin_size_px: params.atera_transcript_density_bin_size_px == null ? 1 : params.atera_transcript_density_bin_size_px as int]
        if (requestedProfile != null) {
            result.min_qv = requestedProfile.min_qv as double
            result.sigma_um = requestedProfile.sigma_um as double
            result.bin_size_px = requestedProfile.bin_size_px as int
        }
        if (!Double.isFinite(result.sigma_um) || result.sigma_um <= 0 ||
            !Double.isFinite(result.min_qv) || result.min_qv < 0 ||
            result.bin_size_px < 1 || result.bin_size_px > 8) {
            throw new IllegalArgumentException('Invalid Atera transcript-density quality, smoothing or bin width')
        }
        if (requestedProfile != null && requestedProfile != profileFromConfig(result)) {
            throw new IllegalArgumentException('The named model requires an unsupported density recipe')
        }
        return result
    }

    static Map profile(params) {
        return profileFromConfig(config('', params))
    }

    static Map profileFromConfig(values) {
        return [version: 1, min_qv: values.min_qv, bin_size_px: values.bin_size_px,
            sigma_um: values.sigma_um, gene_filter: 'is_gene', assignment_filter: 'none',
            z_projection: 'sum_all', units: 'transcripts_per_um2', boundary_mode: 'reflect',
            gaussian_truncate: 4.0, reconstruction: 'bilinear_pixel_centres']
    }
}

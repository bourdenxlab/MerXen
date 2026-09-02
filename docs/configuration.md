# Configuration

MerXen configuration lives in four places:

1. **Shell environment** — a `.env` file for Python-side defaults.
2. **Portable Nextflow parameters** — scientific and algorithmic defaults in
   [workflows/nextflow.config](../workflows/nextflow.config), overridable per
   run with `--<name>`.
3. **Execution profiles** — host capacity, local software/reference paths,
   hardware choices, concurrency, and GPU locking. The default Dwight settings
   live in
   [workflows/conf/dwight.config](../workflows/conf/dwight.config).
4. **Pydantic models** — [src/merxen/config.py](../src/merxen/config.py) — the
   authoritative schema every CLI command validates its JSON config against.

## Environment variables

Copy the template and fill in values:

```bash
cp .env.example .env
```

Variables, from [.env.example](../.env.example):

| Variable | Default | Used by |
|----------|---------|---------|
| `MERXEN_OUTPUT_ROOT` | `./results` | `PipelineConfig.output_root`. Not consumed directly by the pipeline today — Nextflow's `--outdir` is authoritative — but available to Python code that imports `PipelineConfig()`. |
| `MERXEN_PROSEG_INSTALL_PATH` | `/usr/bin/proseg` | Optional Python-side default for `PipelineConfig.proseg_install_path`. Nextflow uses `proseg_install_path` from `workflows/nextflow.config`. |
| `MERXEN_MAX_RAM_GB` | `600.0` | `PipelineConfig.max_ram_gb`. Mirrored into the Nextflow `max_ram_gb` param. |

The `MERXEN_` prefix is wired up via `model_config = {"env_prefix": "MERXEN_"}`
in [config.py:202](../src/merxen/config.py#L202).

## Nextflow parameters

Portable defaults live in
[workflows/nextflow.config](../workflows/nextflow.config). Dwight-specific
defaults are included automatically through Nextflow's reserved `standard`
profile. Override either kind with `--<name>` on the command line.

### Required

| Param | Description |
|-------|-------------|
| `samplesheet` | Path to the samplesheet CSV. |

### Stage-specific

| Param | Default | Description |
|-------|---------|-------------|
| `proseg_search_paths` | Dwight: `/usr/bin/proseg`, `/usr/local/bin/proseg` | Ordered paths checked by `ENSURE_PROSEG` before segmentation. Entries may be executable paths or directories containing `proseg`; `command -v proseg` is checked after this list. |
| `proseg_install_path` | Dwight: `/usr/local/bin/proseg` | Destination used when ProSeg is missing and automatic install is enabled. If the directory is not writable, the bootstrap step requests `sudo`. |
| `proseg_auto_install` | `true` | Install ProSeg automatically with Cargo when no configured search path contains an executable binary. |
| `proseg_cargo_package` | `proseg` | Cargo package name installed by the bootstrap step. |
| `proseg_version` | `3.2.0` | Required ProSeg version; other discovered binaries are ignored. |
| `proseg_git_url` | `https://github.com/dcjones/proseg.git` | Source repository used for the pinned Cargo build. |
| `proseg_git_rev` | `e7df1eace923ce4c6ec70b2c597c5d126aa3db88` | Source revision whose package version is 3.2.0. |

### General

| Param | Default | Description |
|-------|---------|-------------|
| `outdir` | `./results` | Output root. |
| `species` | `human` | Dataset species for the whole invocation: `human` or `mouse`. Controls reference-backed defaults; mixed-species samplesheets are not supported. |
| `analysis_mode` | `paired` | Fallback row mode: `paired`, `merscope`, or `xenium`. A non-empty samplesheet `analysis_mode` value overrides this per row. |
| `enable_alignment` | `false` | Fallback row alignment switch. A non-empty samplesheet `enable_alignment` value overrides this per row; alignment only applies to paired rows. |
| `analysis_segmentation` | `both` | Fallback downstream analysis branches after enrichment. Valid values: `both`, `all`, `reseg`, `original_seg`, `proseg_mask`/`cellpose`, `proseg_hybrid`; comma-separated combinations are accepted. `both` remains `reseg,original_seg`, while `all` includes all four branches. A non-empty samplesheet value overrides this per row. |
| `mask_image_quantification_enabled` | `true` | Insert the Cellpose-mask image quantification stage between enrichment and QC. A non-empty samplesheet `mask_image_quantification_enabled` value overrides this per row. |
| `mecr_enabled` | species-dependent | Insert species-matched whole-brain mutually exclusive co-expression rate analysis after QC. Defaults to `true` for human and `false` for mouse; mouse may opt in to the complete WMB reference. |
| `spatial_gene_analysis_enabled` | `true` | Insert spatial gene analysis between visualization and clustering. A non-empty samplesheet value overrides this per row; disabled rows proceed directly from visualization to clustering. |
| `spatial_gene_analysis_transcript_analysis_enabled` | `true` | Run the annotation-dependent transcript-pattern component inside spatial gene analysis. A non-empty samplesheet value overrides this per row; `false` retains cell-level autocorrelation without requiring tissue GeoJSON files. |
| `cortical_depth_enabled` | `false` | Insert the cortical-depth stage after clustering. Requires per-sample pial/tissue-edge annotations, with optional gray/white boundaries for depth pieces. A non-empty samplesheet `cortical_depth_enabled` value overrides this per row. |
| `distance_from_object_enabled` | `false` | Insert registered polygon-edge distance analysis after cortical depth/clustering. A non-empty samplesheet value overrides this per row. |
| `distance_from_object_segmentations` | `proseg, original, cellpose` | Cell-table branches for object distance. Legacy `reseg`, `original_seg`, and `proseg_mask` values remain accepted aliases. A samplesheet value may override this per row. |
| `mender_enabled` | `false` | Enable independent per-platform/per-segmentation MENDER spatial-domain analysis. A non-empty samplesheet value may override this per row. |
| `mender_segmentations` | `proseg_hybrid` | MENDER branches: any comma-separated subset of `reseg`, `original_seg`, `proseg_mask`, and `proseg_hybrid`, or `all`. A non-empty samplesheet value may override this per row. |
| `force_spatialdata_build` | `false` | Rebuild SpatialData zarrs even if cached. |
| `force_proseg_rerun` | `false` | Rebuild ProSeg bases from the current Cellpose/transcript inputs instead of reusing a persistent `latest_spatialdata.zarr`. Useful with `-resume` after upstream inputs were rebuilt. |
| `start_stage` | `build_spatialdata` | Fallback first stage. Skipped upstream stages are read from published outputs. A samplesheet `start_stage` value overrides this per row. |
| `stop_stage` | `clustering_squidpy` | Fallback last stage. This includes `spatial_gene_analysis`, which runs between visualization and clustering. MapMyCells is available after clustering but opt-in because its atlas downloads are large. A samplesheet `stop_stage` value overrides this per row. |
| `only_stage` | `null` | Fallback single-stage selector. A row-level `only_stage` overrides row start/stop values; row start/stop values suppress the global `only_stage` fallback for that row. |
| `gpu_process_lock_enabled` | Dwight: `true` | Serialize local GPU-heavy processes so `CELLPOSE_SEGMENT`, GPU `ALIGN`, and GPU `CLUSTERING_SQUIDPY` do not compete for one workstation GPU. ProSeg does not take this lock. |
| `gpu_process_lock_file` | Dwight: `/tmp/merxen-dwight-gpu.lock` | One host-wide lock shared by tasks and concurrent launches on Dwight. |

Stage names accepted by `start_stage`, `stop_stage`, and `only_stage` are:
`build_spatialdata`, `segment_nuclei`, `segment`, `enrich`, `mask_image_quantification`,
`qc`, `mecr`, `align`, `align_qc`, `compare`, `visualize`,
`spatial_gene_analysis`, `clustering_squidpy`, `compute_cortical_depth`,
`distance_from_object`, `mender`, and `mapmycells`.
`mask_image_quantification` is
available only when the effective `mask_image_quantification_enabled` value is
`true`. `compute_cortical_depth` is available only when the effective
`cortical_depth_enabled` value is `true`. `align` and `align_qc` are available
only for rows whose effective `enable_alignment` value is `true`.
`mecr` is available only when the effective `mecr_enabled` value is `true`.
`spatial_gene_analysis` is available only when the effective
`spatial_gene_analysis_enabled` value is `true`.
`distance_from_object` is available only when the effective
`distance_from_object_enabled` value is `true`.
`mender` is available only when the effective `mender_enabled` value is `true`.
`align`, `align_qc`, and `compare` are available only when
`analysis_mode = paired`.

### Cellpose

`segment_nuclei` uses DAPI only, the Cellpose `nuclei` model, and the same
inference/tiling settings as cell Cellpose. Both masks use the default final
area filter of 5–400 µm². On Dwight,
`cellpose_segment_max_forks = 1` and the shared GPU lock serialize both
Cellpose process types.

| Param | Default | Description |
|-------|---------|-------------|
| `cellpose_model_type` | `cyto3` | Cellpose model preset. |
| `cellpose_gpu` | Dwight: `true` | Use GPU for inference. Other execution profiles choose their own hardware policy. |
| `cellpose_diameter` | `null` | Cell diameter (px). `null` → Cellpose auto-estimates. |
| `cellpose_flow_threshold` | `0.7` | Cellpose flow threshold. |
| `cellpose_cellprob` | `-5.0` | Cellpose cell probability threshold. |
| `cellpose_tile_overlap` | `0.15` | Cellpose model's internal fractional tile overlap. |
| `cellpose_bsize` | `256` | Cellpose internal batch block size. |
| `cellpose_tile_size_candidates` | `[6144, 4096, 3072, 2048]` | Candidate halo tile sizes probed from largest to smallest. |
| `cellpose_min_tile_size` | `1024` | Smallest allowed Cellpose halo tile size. |
| `cellpose_stitch_overlap_px` | `256` | Halo overlap, in pixels, used for MerXen object-level stitching. |
| `cellpose_stitch_status_every_tiles` | `10` | Progress/status interval for Cellpose tile stitching. |
| `cellpose_filter_per_tile` | `true` | Apply regionprops filtering to each tile before stitching. |
| `cellpose_duplicate_iou_threshold` | `0.25` | Skip an owned tile object as a duplicate when overlap IoU with an existing label is at least this value. |
| `cellpose_duplicate_overlap_fraction` | `0.5` | Skip an owned tile object as a duplicate when at least this fraction of its pixels overlap one existing label. |
| `cellpose_min_remaining_fraction` | `0.05` | Skip a non-duplicate object if too little of it remains after preserving existing labels. |
| `cellpose_edge_touch_policy` | `keep` | Keep or skip labels touching an artificial tile edge; `keep` records them in stitching stats. |
| `cellpose_write_stitching_stats` | `true` | Write `cellpose_stitching_stats.json` beside the stitched mask. |
| `cellpose_final_min_area_um2` | `5.0` | Drop final Cellpose masks smaller than this area before ProSeg. |
| `cellpose_final_max_area_um2` | `400.0` | Drop final Cellpose masks larger than this area before ProSeg. |
| `cellpose_final_filter_chunk_mb` | `256` | Approximate row-chunk size for streaming the final mask filter. |
| `cellpose_segment_max_forks` | Dwight: `1` | Maximum concurrent GPU Cellpose processes. |

### Mask image quantification

| Param | Default | Description |
|-------|---------|-------------|
| `mask_image_quantification_enabled` | `true` | Run Cellpose-mask image quantification after enrichment by default. |
| `mask_image_quantification_max_forks` | Dwight: `3` | Maximum concurrent quantification processes. |

### Viewer caches

| Param | Default | Description |
|-------|---------|-------------|
| `viewer_cache_enabled` | `true` | Pre-build image pyramids and rasterized labels/outlines for ProSeg, ProSeg hybrid, Cellpose, and original segmentation shapes that exist in the store. |
| `viewer_cache_downsample` | `4` | Downsampling factor for derived image and label pyramids. |
| `viewer_cache_label_chunk_size` | `2048` | Label rasterization chunk edge length in pixels. |
| `viewer_cache_contour_width` | `1` | Cached label-outline width. |
| `viewer_cache_min_size` | `4096` | Minimum image dimension at which another pyramid level is generated. |
| `viewer_cache_build_image_pyramid` | `true` | Also pre-build the viewer image pyramid. |
| `viewer_cache_max_forks` | Dwight: `9` | Maximum concurrent viewer-cache processes. The workstation profile pairs this with a 60 GB process reservation, capping the stage at 540 GB and 72 CPUs. |

### Cortical depth

| Param | Default | Description |
|-------|---------|-------------|
| `cortical_depth_enabled` | `false` | Run `COMPUTE_CORTICAL_DEPTH` as a terminal stage after `CLUSTERING_SQUIDPY`. |
| `cortical_depth_coordinate_unit_um` | `1.0` | Microns per coordinate unit in annotation/cell coordinates. Use the image pixel size if annotations are in pixel coordinates; keep `1.0` when coordinates are already microns. |
| `cortical_depth_raster_resolution_um` | `5.0` | Finite-difference raster spacing. Smaller values improve geometry fidelity and increase memory/time. |
| `cortical_depth_raster_padding_um` | `null` | Optional padding around the ribbon bounds. `null` uses a small automatic padding. |
| `cortical_depth_boundary_band_um` | `null` | Width of the rasterized Dirichlet boundary band. `null` uses about 1.5 raster pixels. |
| `cortical_depth_boundary_smoothing_window` | `0` | Optional moving-average smoothing window over boundary vertices. |
| `cortical_depth_streamline_spacing_um` | `50.0` | Approximate pial arc-length spacing between streamline seeds. |
| `cortical_depth_streamline_step_um` | `null` | Integration step length. `null` uses about half the raster resolution. |
| `cortical_depth_streamline_max_steps` | `4000` | Maximum integration steps per streamline. |
| `cortical_depth_streamline_resample_points` | `101` | Number of points stored per streamline. |
| `cortical_depth_side_boundary_distance_um` | `25.0` | Distance from artificial side boundaries used to flag cells/streamlines. |
| `cortical_depth_contour_levels` | `0.1..0.9` | Depth contours written to GeoJSON/QC overlays. |
| `cortical_depth_write_spatialdata_table` | `true` | Replace selected SpatialData tables with cortical-depth columns added to `obs`. |
| `cortical_depth_max_forks` | Dwight: `3` | Maximum concurrent cortical-depth processes. |

### Distance from object

| Param | Default | Description |
|-------|---------|-------------|
| `distance_from_object_enabled` | `false` | Run polygon-edge annotation and cohort pseudobulk analysis. |
| `distance_from_object_segmentations` | `[proseg, original, cellpose]` | Selected cell-table branches. `proseg_geometry_assignment` and `proseg_hybrid` can be selected explicitly when their tables exist. |
| `distance_from_object_object_types` | `null` | Optional object-type allow-list; `null` analyses every named type together. |
| `distance_from_object_coordinate_unit_um` | `1.0` | Micrometres per registered coordinate unit. |
| `distance_from_object_near_distance_um` | `50.0` | Exclusive upper boundary of `near`; polygon interiors are always near. |
| `distance_from_object_far_distance_um` | `100.0` | Inclusive lower boundary of `far`; the intervening band is `middle`. |
| `distance_from_object_max_distance_um` | `200.0` | Inclusive upper far boundary; larger distances are `beyond_max`. |
| `distance_from_object_min_cells_per_pseudobulk` | `10` | Minimum eligible grey-matter cells in each pair/proximity pseudobulk. |
| `distance_from_object_min_pairs` | `2` | Minimum complete tissue blocks with both near and far samples for PyDESeq2. |
| `distance_from_object_n_cpus` | Dwight: `8` | PyDESeq2 workers and cohort process CPUs. |
| `distance_from_object_write_spatialdata_table` | `true` | Add distance columns to selected tables while preserving existing `obs`. |
| `distance_from_object_max_forks` | Dwight: `3` | Maximum concurrent per-platform annotation processes. |

### ProSeg

Before `SEGMENT` runs, the `ENSURE_PROSEG` process checks
`proseg_search_paths`, then `command -v proseg`. If no executable is found and
`proseg_auto_install=true`, it builds the pinned ProSeg 3.2.0 Git revision into
a temporary root and copies the resulting binary to `proseg_install_path`. System-owned
install paths trigger a `sudo` prompt.

| Param | Default | Description |
|-------|---------|-------------|
| `proseg_samples` | `1200` | MCMC samples. |
| `proseg_segment_max_forks` | Dwight: `2` | Maximum concurrent CPU-only ProSeg processes. |
| `proseg_voxel_size` | `0.5` | Voxel size (µm). |
| `proseg_burnin_voxel_size` | `1.0` | Burn-in voxel size (µm). |
| `proseg_nuclear_reassignment_prob` | `0.20` | Nuclear reassignment probability. |
| `proseg_diffusion_probability` | `0.20` | Diffusion probability. |
| `proseg_cell_compactness` | `0.04` | Cell compactness prior. |
| `proseg_num_threads` | Dwight: `32` | ProSeg thread count. |

### ProSeg hybrid branch

| Param | Default | Description |
|-------|---------|-------------|
| `proseg_hybrid_enabled` | `true` | Generate the separate overlap-aware hybrid shape/table branch. |
| `proseg_hybrid_min_transcripts` | `10` | Minimum ProSeg-foreground transcript count, before and after outlier filtering, required for transcript-driven expansion. |
| `proseg_hybrid_outlier_neighbors` | `2` | Nearest-neighbour rank used for robust bulk-component detection. |
| `proseg_hybrid_outlier_mad_multiplier` | `2.0` | MAD multiplier defining the robust component link distance. Lower values reject small detached groups more harshly. |
| `proseg_hybrid_minimum_external_group` | `3` | Minimum distant external group size allowed to drive an expansion. |
| `proseg_hybrid_chain_radius_scale` | `2.0` | External-chain link radius as a multiple of the robust neighbour scale. A distant group must chain back to Cellpose. |
| `proseg_hybrid_near_surface_radius_fraction` | `0.25` | Distance from the Cellpose surface, as a fraction of equivalent Cellpose radius, where retained external transcripts can be accepted individually. |
| `proseg_hybrid_maximum_expansion_radius_fraction` | `1.0` | Hard expansion cap beyond the Cellpose surface, as a fraction of equivalent Cellpose radius. |
| `proseg_hybrid_attachment_arc_width_scale` | `0.5` | Width multiplier for the local Cellpose boundary arc attached to each supported convex expansion. |
| `proseg_hybrid_rounding_radius_fraction` | `0.15` | Outward rounding of each local convex expansion, as a fraction of equivalent Cellpose radius. |
| `proseg_hybrid_smoothing_radius_um` | `10.0` | Fixed-radius morphological closing applied to the assembled mask. Only the added area is retained, so smoothing cannot shrink the mask. |
| `proseg_hybrid_outward_rounding_um` | `0.2` | Fixed outward offset applied during final smoothing. |
| `proseg_hybrid_smoothing_quad_segs` | `32` | Circular approximation resolution per quadrant for final smoothing and its cap. |
| `proseg_hybrid_containment_tolerance_um` | `1.0e-5` | Numerical tolerance used by strict containment and cap checks. |
| `default_merscope_voxel_layers` | `7` | Fallback when samplesheet column is empty. |
| `default_xenium_voxel_layers` | `2` | Fallback when samplesheet column is empty. |

### Platform-specific

| Param | Default | Description |
|-------|---------|-------------|
| `xenium_min_qv` | `20.0` | Minimum transcript QV to retain. |

### Alignment

Alignment is optional and runs in the isolated `envs/environment.alignment.yml`.
The default `valis` backend registers DAPI morphology only; it does not use
transcripts, expression, cell labels, or RNA-derived images. The former method
is retained as the explicit `legacy_spateo` backend. Non-alignment stages keep
using `envs/environment.yml`.

| Param | Default | Description |
|-------|---------|-------------|
| `enable_alignment` | `false` | Run `ALIGN` and `ALIGN_QC` between QC and comparison by default. A samplesheet `enable_alignment` value can override this per paired row. |
| `alignment_backend` | `valis` | `valis` (DAPI-only default) or `legacy_spateo`. |
| `alignment_fixed_platform` / `alignment_moving_platform` | `XENIUM` / `MERSCOPE` | Reference and transformed dataset. They must differ. |
| `alignment_conda` | `envs/environment.alignment.yml` | Conda env file or existing env path used only for `ALIGN`. |
| `alignment_bootstrap_dependencies` | `true` | Install the exact selected backend package when its dependency check fails. The locked transitive VALIS stack is already part of the conda environment. |
| `alignment_valis_requirement` | `valis-wsi==1.2.0` | Exact package installed with `--no-deps` after the locked NumPy-2-compatible runtime. |
| `alignment_device` | Dwight: `auto` | DISK/LightGlue device; automatically uses CUDA when available. Use `cpu` for an explicit CPU fallback. |
| `alignment_max_forks` | Dwight: `1` | Maximum concurrent `ALIGN` tasks. Raise only with sufficient CPU/GPU memory. |
| `alignment_pytorch_cuda_alloc_conf` | Dwight: `expandable_segments:True,max_split_size_mb:256` | PyTorch allocator setting exported by `ALIGN`. |

#### VALIS DAPI input and preprocessing

VALIS alignment requires both platform-specific combined annotation files in
the samplesheet: `merscope_cortical_depth_annotation_geojson` and
`xenium_cortical_depth_annotation_geojson`. This requirement is independent of
`cortical_depth_enabled`. Direct CLI JSON uses
`merscope_image.tissue_annotation_path` and
`xenium_image.tissue_annotation_path`. The annotation's pial pieces and single
shared tissue edge define the anatomical mask; exclusions are subtracted, all
other roles are ignored, and no mask morphology is performed.

| Param | Default | Description |
|-------|---------|-------------|
| `alignment_merscope_image_key` / `alignment_xenium_image_key` | `MERSCOPE_z_projection` / `morphology_focus` | SpatialData image elements used for registration. |
| `alignment_merscope_dapi_channel` / `alignment_xenium_dapi_channel` | `DAPI` / `DAPI` | Channel selected by name, not position. |
| `alignment_merscope_pixel_size_um` / `alignment_xenium_pixel_size_um` | `null` / `null` | Optional physical-size overrides. Values are checked against inferred SpatialData transforms. |
| `alignment_registration_pixel_size_um` | `null` | Shared isotropic registration pixel size. `null` chooses a resolution bounded by both native scales and the source-size limit. |
| `alignment_registration_source_max_dim_px` | `3200` | Maximum source diagonal used to choose temporary registration resolution. |
| `alignment_background_sigma_um` | `75.0` | Broad background-removal scale. |
| `alignment_background_boundary_mode` | `mirror` | Boundary mode used by support-normalized Gaussian background and smoothing. |
| `alignment_intensity_lower_percentile` / `alignment_intensity_upper_percentile` | `0.5` / `99.5` | Robust intensity clipping limits. |
| `alignment_intensity_compression` | `asinh` | Monotonic `asinh` or `log1p` compression. |
| `alignment_clahe_clip_limit` | `0.01` | Mild local contrast limit. |
| `alignment_smoothing_sigma_um` | `3.0` | Nuclear-density smoothing scale. |
| `alignment_edge_taper_um` | `150.0` | Cosine taper measured inward from the acquired MERSCOPE FOV footprint (or rectangular support on other inputs), applied only to temporary registration images and fields. |
| `alignment_edge_exclusion_um` | `150.0` | Acquired-support margin excluded from feature/intensity/non-rigid scoring. It does not erode the annotation-derived anatomical mask used for overlap and boundaries. |
| `alignment_mask_smoothing_sigma_um` | `20.0` | Deprecated automatic-mask compatibility value; unused by production VALIS registration. |
| `alignment_mask_closing_radius_um` | `30.0` | Deprecated automatic-mask compatibility value; unused by production VALIS registration. |
| `alignment_mask_min_area_um2` / `alignment_mask_hole_area_um2` | `25000.0` / `25000.0` | Deprecated automatic-mask compatibility values; unused by production VALIS registration. |
| `alignment_mask_dilation_um` | `10.0` | Deprecated automatic-mask compatibility value; the annotation-derived mask is not dilated. |

#### Orientation, VALIS, and transform output

| Param | Default | Description |
|-------|---------|-------------|
| `alignment_orientation_max_dim_px` | `512` | Maximum image dimension for coarse orientation search. |
| `alignment_orientation_coarse_step_degrees` / `alignment_orientation_refine_step_degrees` / `alignment_orientation_final_step_degrees` | `10.0` / `2.0` / `0.5` | Full-circle angular-search increments when SIFT/RANSAC support is insufficient. |
| `alignment_allow_reflection` / `alignment_reflection_mode` | `true` / `auto` | Search independent reflected and non-reflected beams; `force` and `forbid` provide per-sample overrides. |
| `alignment_reflection_minimum_score_improvement` | `0.01` | Symmetric handedness margin; closer scores are marked ambiguous while the higher score proceeds provisionally. |
| `alignment_orientation_translation_candidates_per_angle` | `3` | Translation seeds retained per angle during joint search. |
| `alignment_orientation_coarse_translation_radius_px` / `alignment_orientation_refine_translation_radius_px` / `alignment_orientation_final_translation_radius_px` | `64` / `16` / `4` | Translation neighborhoods on the orientation-search canvas. |
| `alignment_orientation_min_fixed_overlap_fraction` / `alignment_orientation_min_moving_overlap_fraction` | `0.45` / `0.45` | Candidate eligibility coverage gates. |
| `alignment_orientation_min_retained_moving_fraction` / `alignment_orientation_min_relative_dice` | `0.6` / `0.7` | Reject clipped or grossly inferior candidates before ranking. |
| `alignment_orientation_initial_angle_degrees` / `alignment_orientation_initial_translation_x_um` / `alignment_orientation_initial_translation_y_um` | `null` | Optional per-sample joint-search seeds. |
| `alignment_orientation_local_fine_search_enabled` | `true` | Refine and assess stability around the selected handedness. |
| `alignment_orientation_local_fine_angle_radius_degrees` / `alignment_orientation_local_fine_translation_radius_um` | `2.5` / `500` | Final local search window in angle and full-scale physical X/Y translation. |
| `alignment_orientation_local_fine_coarse_angle_step_degrees` / `alignment_orientation_local_fine_coarse_translation_step_um` | `0.5` / `100` | Coarse local 3D score-volume increments. |
| `alignment_orientation_local_fine_refine_angle_step_degrees` / `alignment_orientation_local_fine_refine_translation_step_um` | `0.1` / `25` | Fine increments used to confirm persistent interior maxima. |
| `alignment_orientation_local_fine_maxima_to_refine` / `alignment_orientation_local_fine_competing_score_margin` | `4` / `0.002` | Maximum local peaks tested and score margin used to flag a competing stable solution. |
| `alignment_partial_overlap_enabled` | `true` | Run joint residual rotation/X/Y refinement before VALIS. |
| `alignment_partial_overlap_max_dim_px` | `512` | Maximum dimension of the robust rigid-search canvas. |
| `alignment_partial_overlap_angle_radius_degrees` / `alignment_partial_overlap_angle_step_degrees` | `10.0` / `1.0` | Coarse residual-angle search window and increment. |
| `alignment_partial_overlap_max_translation_um` | `1500.0` | Maximum absolute residual translation per axis. |
| `alignment_partial_overlap_retained_boundary_fraction` | `0.7` | Closest fraction retained independently in both directions of the trimmed boundary distance. |
| `alignment_partial_overlap_boundary_distance_scale_um` | `150.0` | Physical scale converting robust boundary distance into an objective score. |
| `alignment_partial_overlap_density_sigma_um` | `75.0` | DAPI-density smoothing for overlap-normalized internal-structure correlation. |
| `alignment_partial_overlap_min_fixed_overlap_fraction` / `alignment_partial_overlap_min_moving_overlap_fraction` | `0.45` / `0.45` | Independent coverage constraints preventing a tiny coincidental match. |
| `alignment_partial_overlap_candidates_to_refine` | `5` | Spatially distinct coarse candidates refined with bounded Powell optimization. |
| `alignment_valis_num_features` | `7500` | DISK features supplied to LightGlue. |
| `alignment_valis_max_processed_image_dim_px` | `1600` | VALIS global feature-registration image limit. |
| `alignment_valis_max_non_rigid_registration_dim_px` | `3200` | VALIS non-rigid registration image limit. |
| `alignment_valis_thumbnail_size` | `1024` | VALIS diagnostic thumbnail size. |
| `alignment_valis_global_transform` | `rigid` | Deprecated compatibility/provenance field. The accepted MerXen partial-overlap transform is locked and VALIS global fitting is disabled. |
| `alignment_seed` | `21` | Shared deterministic seed for VALIS/OpenCV/PyTorch and legacy subsampling. |
| `alignment_valis_non_rigid_enabled` | `true` | Run conservative non-rigid refinement after global QC passes. |
| `alignment_valis_non_rigid_backend` | `optical_flow` | Explicit `optical_flow` or `simple_elastix`; unavailable Elastix raises rather than silently changing algorithms. |
| `alignment_valis_non_rigid_grid_spacing_ratio` | `0.08` | SimpleElastix grid spacing when that backend is selected. |
| `alignment_valis_non_rigid_maximum_iterations` | `500` | SimpleElastix iteration limit. |
| `alignment_valis_non_rigid_smoothing_sigma_ratio` | `0.02` | Optical-flow field smoothing ratio. |
| `alignment_valis_field_sample_spacing_px` | `8` | Spacing for serialized forward/backward displacement fields. |
| `alignment_coordinate_system_name` | `merxen_xenium` | Named fixed-platform SpatialData coordinate system. |
| `alignment_transform_transcripts` / `alignment_transform_centroids` / `alignment_transform_polygons` | `true` / `true` / `true` | Materialize selected registered vectors and table centroid coordinates. |
| `alignment_mark_shared_tissue_domain` | `true` | Annotate transformed points, shapes, and centroids with the shared annotation-derived anatomical tissue domain. |
| `alignment_resume` | `true` | Reuse a complete transform bundle when its stored VALIS parameters, platform roles, and annotation content hashes match. Nextflow `-resume` remains the normal workflow-level cache. |

#### VALIS QC selection

| Param | Default | Description |
|-------|---------|-------------|
| `alignment_qc_minimum_global_dice` | `0.15` | Minimum global tissue-mask Dice. |
| `alignment_qc_minimum_global_mutual_information` | `0.02` | Minimum global DAPI normalized mutual information. |
| `alignment_qc_minimum_global_inliers` | `8` | Minimum registered DAPI feature inliers. |
| `alignment_qc_minimum_inlier_coverage` | `0.05` | Minimum spatial coverage of feature support. |
| `alignment_qc_non_rigid_minimum_nmi_improvement` | `0.0` | Minimum absolute DAPI NMI gain required to select non-rigid output. The default rejects NMI degradation; density, robust-score, Dice, drift, displacement, and Jacobian gates must also pass. |
| `alignment_qc_non_rigid_maximum_p95_displacement_um` | `500.0` | Maximum accepted 95th-percentile displacement. |
| `alignment_qc_non_rigid_maximum_coherent_rotation_degrees` | `0.25` | Maximum global rotation that may be encoded inside the nominally local displacement field. |
| `alignment_qc_non_rigid_maximum_coherent_translation_um` | `25.0` | Maximum global translation that may be encoded inside the nominally local displacement field. |
| `alignment_qc_non_rigid_maximum_density_correlation_degradation` | `0.0` | Maximum allowed loss of smoothed DAPI-density correlation relative to the locked transform. |
| `alignment_qc_non_rigid_maximum_robust_score_degradation` | `0.002` | Maximum allowed loss of the partial-overlap robust objective. |
| `alignment_qc_non_rigid_maximum_tissue_dice_degradation` | `0.01` | Maximum allowed loss of tissue-mask Dice. |

Affine determinant/singular-value/shear gates and non-rigid Jacobian gates are
also configurable in direct `AlignmentConfig` JSON. See
[Section alignment](stages/alignment.md) for the complete transform and QC
contract.

#### Legacy Spateo parameters

The following settings are read only when
`alignment_backend=legacy_spateo`: `alignment_spateo_mode`,
`alignment_dtype`, `alignment_selected_mode`, `alignment_max_iter`,
`alignment_nonrigid_start_iter`, `alignment_beta`, `alignment_lambda_vf`,
`alignment_k`, `alignment_partial_robust_level`, `alignment_allow_flip`,
`alignment_svi_mode`, `alignment_n_sampling`, `alignment_sparse_top_k`,
`alignment_sparse_calculation_mode`, `alignment_use_chunk`,
`alignment_chunk_capacity`, `alignment_use_hvg`, `alignment_n_top_genes`,
`alignment_use_pca`, `alignment_n_pcs`, `alignment_max_alignment_cells`,
`alignment_rbf_neighbors`, `alignment_rbf_smoothing`, and
`alignment_max_nonrigid_anchors`. Its bootstrap retains
`alignment_dynamo_requirement`, `alignment_spateo_requirement`, and
`alignment_anndata_requirement`. `alignment_qc_grid_rows` and
`alignment_qc_grid_cols` are likewise legacy expression-QC settings.

### Squidpy clustering

| Param | Default | Description |
|-------|---------|-------------|
| `clustering_squidpy_drop_control_features` | `true` | Remove blank/negative/control-like features before cell/gene filtering and clustering. |
| `clustering_squidpy_min_counts` | `10` | Minimum counts per cell passed to `scanpy.pp.filter_cells`. |
| `clustering_squidpy_min_cells` | `5` | Minimum cells per gene passed to `scanpy.pp.filter_genes`. |
| `clustering_squidpy_normalize_target_sum` | `null` | Optional target sum for `scanpy.pp.normalize_total`; `null` uses Scanpy's default. |
| `clustering_squidpy_normalize_exclude_highly_expressed` | `false` | Exclude highly expressed genes from Scanpy size-factor calculation. |
| `clustering_squidpy_normalize_max_fraction` | `0.05` | Fraction threshold used when excluding highly expressed genes. |
| `clustering_squidpy_n_pcs` | `60` | Maximum PCs for `scanpy.pp.pca`. |
| `clustering_squidpy_n_neighbors` | `30` | Neighbor count for `scanpy.pp.neighbors`. |
| `clustering_squidpy_leiden_resolution` | `0.5` | Leiden clustering resolution. |
| `clustering_squidpy_umap_min_dist` | `0.3` | Minimum distance parameter for `scanpy.tl.umap`. |
| `clustering_squidpy_umap_spread` | `1.0` | Spread parameter for `scanpy.tl.umap`. |
| `clustering_squidpy_random_seed` | `0` | Seed for PCA/UMAP/Leiden. |
| `clustering_squidpy_spatial_point_size` | `0.5` | Highlight point size for spatial cluster grid plots. |
| `clustering_squidpy_spatial_scatter_point_size` | `2.0` | Point size for regular spatial scatter plots. |
| `clustering_squidpy_figure_dpi` | `180` | DPI for PNG plots. |
| `clustering_squidpy_use_gpu` | Dwight: `true` | Use RAPIDS single-cell acceleration when available. |
| `clustering_squidpy_gpu_conda` | `envs/environment.clustering-gpu.yml` | Dedicated RAPIDS environment used only by `CLUSTERING_SQUIDPY_COMPUTE`. |
| `clustering_squidpy_gpu_container` | Site GPU image path | Dedicated RAPIDS image used only by `CLUSTERING_SQUIDPY_COMPUTE` with Apptainer. Build it from `containers/Dockerfile.clustering-gpu` or override this path. |
| `clustering_squidpy_max_forks` | Dwight: `4` | Maximum concurrent Squidpy clustering tasks. GPU-backed tasks still share the local GPU lock when enabled. |
| `clustering_squidpy_gpu_vram_monitor` | Dwight: `true` | Run a lightweight `nvidia-smi` sampler around each `CLUSTERING_SQUIDPY_COMPUTE` task. |
| `clustering_squidpy_gpu_vram_monitor_interval_seconds` | `2` | Sampling interval for the clustering GPU VRAM monitor. |
| `clustering_squidpy_write_spatialdata_table` | `true` | Add or replace a final clustered AnnData table in each source `latest_spatialdata.zarr`. |
| `clustering_squidpy_hierarchical_enabled` | `true` | Run broad atlas-guided annotation and per-branch subclustering using the species-matched atlas. |
| `clustering_squidpy_broad_leiden_resolution` | `0.2` | Low-resolution Leiden round used for broad atlas annotation. |
| `clustering_squidpy_subcluster_leiden_resolution` | `0.5` | Default Leiden resolution for non-neuron broad-class branches. |
| `clustering_squidpy_subcluster_resolution_overrides` | `[:]` | Optional Nextflow map from broad class or neuron split label to a custom branch Leiden resolution. |
| `clustering_squidpy_neuron_split_leiden_resolution` | `0.15` | Coarse neuron round used before Excitatory/Inhibitory/Other annotation. |
| `clustering_squidpy_neuron_subcluster_leiden_resolution` | `0.5` | Default Leiden resolution for neuron subtype branches. |
| `clustering_squidpy_min_branch_cells` | `50` | Smallest branch/split size that will be reclustered. Smaller groups keep labels but skip PCA/UMAP/Leiden. |
| `clustering_squidpy_broad_reference_atlas` | species-dependent | `whb` for human or `wmb` for mouse. Must agree with `species` when hierarchy is enabled. |
| `clustering_squidpy_broad_marker_lookup_path` | atlas/cache-dependent | WHB or WMB MapMyCells marker lookup used for atlas-guided cluster annotation. |
| `clustering_squidpy_broad_taxonomy_metadata_path` | atlas/cache-dependent | Allen `cluster_annotation_term.csv` used to map marker lookup IDs to atlas labels. |
| `clustering_squidpy_broad_cluster_membership_path` | atlas/cache-dependent | Allen membership metadata used for neuron neurotransmitter split labels. |
| `clustering_squidpy_broad_reference_cache_dir` | `<outdir>/mapmycells_cache` | Cache searched for matching WHB/WMB taxonomy, markers, and reference H5AD gene-symbol metadata. The Dwight profile points this at its shared cache. |
| `clustering_squidpy_broad_auto_download_reference` | `true` | Download missing compact WMB marker, taxonomy, membership, and gene metadata into the reference cache. WHB continues to use configured local inputs. |
| `clustering_squidpy_broad_reference_gene_metadata_paths` | `[]` | Optional reference H5AD files used to bridge panel symbols to species-appropriate Ensembl IDs. |
| `clustering_squidpy_broad_marker_level` | atlas-dependent | `CCN202210140_SUPC` for WHB or `CCN20230722_CLAS` for WMB. |
| `clustering_squidpy_broad_min_marker_overlap` | `3` | Minimum query-panel marker overlap required to score an atlas label. |
| `clustering_squidpy_broad_max_markers_per_label` | `80` | Maximum resolved markers used per atlas label. |
| `clustering_squidpy_broad_score_margin_threshold` | `0.0` | Minimum difference between best and runner-up atlas scores; lower margins become `Mixed/Unknown`. |
| `clustering_squidpy_broad_unknown_label` | `Mixed/Unknown` | Label used when no atlas marker set scores confidently. |

### MENDER spatial domains

| Param | Default | Description |
|-------|---------|-------------|
| `mender_enabled` | `false` | Enable the terminal MENDER stage. |
| `mender_segmentations` | `proseg_hybrid` | One segmentation, a comma-separated subset, or `all`. |
| `mender_cell_state_key` | `hierarchical_cluster` | Required categorical cell-state column; ordinary Leiden is never used as a fallback. |
| `mender_missing_state_policy` | `error` | Reject missing or empty cell states. |
| `mender_nn_mode` | `radius` | MENDER neighbourhood mode. |
| `mender_radius_um` | `20.0` | Radius increment in native micrometres for each scale. |
| `mender_n_scales` | `5` | Number of spatial context scales. |
| `mender_count_rep` | `s` | Per-scale (`s`) rather than accumulated (`a`) state frequencies. |
| `mender_include_self` | `false` | Exclude the central cell so domains are driven by surrounding neighbourhood composition. |
| `mender_clustering_mode` | `resolution` | `resolution` passes a negative Leiden resolution; `target_k` passes a positive domain target. |
| `mender_leiden_resolution` | `0.8` | Fixed layer-focused resolution, passed to MENDER as `-0.8`. |
| `mender_target_k` | `null` | Required integer of at least 2 only in target-K mode. |
| `mender_random_seed` | `666` | MENDER tutorial seed. |
| `mender_run_umap` | `true` | Generate the context embedding. |
| `mender_write_spatialdata_table` | `true` | Import `mender_domain` into the derived clustered SpatialData table. |
| `mender_conda` | `envs/environment.mender.yml` | Old compatible CPU environment used only by compute. |
| `mender_container` | Site MENDER SIF path | CPU-only image built from `containers/Dockerfile.mender`; override for portable Apptainer runs. |
| `mender_compute_max_forks` | Dwight: `2` | At 192 GB per compute task, at most 384 GB is reserved concurrently. |

See [MENDER spatial domains](stages/mender.md) for the data contract, restart
mode, output layout, and CPU container instructions.

### Spatial gene analysis

| Param | Default | Description |
|-------|---------|-------------|
| `spatial_gene_analysis_drop_control_features` | `true` | Remove blank/negative/control-like genes before autocorrelation. |
| `spatial_gene_analysis_min_counts` | `0` | Optional minimum total counts per cell before analysis. |
| `spatial_gene_analysis_min_cells` | `5` | Minimum cells with a gene detected before calculating metrics. |
| `spatial_gene_analysis_normalize_target_sum` | `null` | Optional target sum for `scanpy.pp.normalize_total`; `null` uses Scanpy's default. |
| `spatial_gene_analysis_normalize_exclude_highly_expressed` | `false` | Exclude highly expressed genes from Scanpy size-factor calculation. |
| `spatial_gene_analysis_normalize_max_fraction` | `0.05` | Fraction threshold used when excluding highly expressed genes. |
| `spatial_gene_analysis_n_neighbors` | `6` | Spatial nearest-neighbor count used by Squidpy's generic-coordinate neighbor graph. |
| `spatial_gene_analysis_top_n` | `10` | Number of highest and lowest genes retained for each metric ranking. |
| `spatial_gene_analysis_spatial_point_size` | `2.0` | Point size for individual spatial gene expression plots. |
| `spatial_gene_analysis_figure_dpi` | `180` | PNG output DPI. |
| `spatial_gene_analysis_max_forks` | Dwight: `4` | Maximum concurrent spatial gene analysis tasks. |

### MECR

| Param | Default | Description |
|-------|---------|-------------|
| `mecr_enabled` | human: `true`; mouse: `false` | Run species-matched whole-brain mutually exclusive co-expression rate analysis after QC. Mouse is opt-in because the complete WMB raw reference is about 151 GB. |
| `mecr_reference_h5ad_paths` | `[]` | Optional complete list of raw reference H5AD shards. WMB auto-download populates all 10Xv2, 10Xv3, and 10X Multiome shards. |
| `mecr_neurons_h5ad_path` | Complete WHB-10Xv3 neuron raw H5AD | Legacy WHB neuronal input used when the generic list is empty. |
| `mecr_nonneurons_h5ad_path` | Complete WHB-10Xv3 non-neuron raw H5AD | Legacy WHB non-neuronal input used when the generic list is empty. |
| `mecr_cell_metadata_path` | Atlas-dependent | Maps raw H5AD cell labels to cluster aliases. |
| `mecr_taxonomy_metadata_path` | Atlas-dependent | Resolves taxonomy node labels. |
| `mecr_cluster_membership_path` | Atlas-dependent | Maps cluster aliases to taxonomy nodes. |
| `mecr_reference_cache_dir` | `<outdir>/mapmycells_cache` | Durable Allen reference cache. The Dwight profile uses `/media/mathieubo/SSD1/MerXen/mapmycells`. |
| `mecr_auto_download_reference` | `true` | For mouse, download or reuse the complete WMB raw reference and metadata. Downloads are resumable and use four concurrent transfers. |
| `mecr_taxonomy_level` | atlas-dependent | `CCN202210140_SUPC` for WHB or `CCN20230722_CLAS` for WMB. |
| `mecr_gene_symbol_column` | `gene_symbol` | Reference `var` column containing gene symbols. |
| `mecr_target_broad_classes` | atlas-dependent | Seven WHB core classes or six WMB classes: neurons, Astro-Epen, OPC-Oligo, OEC, vascular, and immune/microglia. |
| `mecr_marker_min_target_fraction` | `0.25` | A marker must be detected in strictly more than this fraction of its target class. |
| `mecr_marker_max_other_fraction` | `0.01` | A marker must be detected in strictly less than this fraction of the remaining retained cells. |
| `mecr_normalize_target_sum` | `10000.0` | Full-library per-cell normalization target before `log1p` and Wilcoxon. |
| `mecr_reference_chunk_rows` | `5000` | Reference rows streamed from each raw H5AD at a time. |
| `mecr_wilcoxon_tie_correct` | `true` | Enable tie correction in Scanpy's Python Wilcoxon implementation. |
| `mecr_figure_dpi` | `180` | Pair-rate distribution plot DPI. |
| `mecr_barnyard_top_n_pairs` | `6` | Maximum canonical/high-MECR/high-detection pairs selected for barnyard count plots. Set to `0` to disable them. |
| `mecr_barnyard_max_points` | `50000` | Deterministic maximum plotted cells per platform and barnyard pair; exact MECR still uses every cell. |
| `mecr_barnyard_random_seed` | `0` | Seed for barnyard display-point downsampling. |
| `mecr_barnyard_log1p` | `false` | Plot barnyard axes in natural count space by default. Set to `true` for `log1p`-transformed display coordinates. |
| `mecr_max_forks` | Dwight: `4` | Maximum concurrent branch-scoring tasks. The shared reference task always has one fork. |

The reference task uses the union of genes in all selected spatial panels,
runs once, and is shared across all samples and segmentation branches. See
[Mutually exclusive co-expression rate](stages/mecr.md) for the method.

### MapMyCells

| Param | Default | Description |
|-------|---------|-------------|
| `mapmycells_reference_mode` | species-dependent | Human: `both`; mouse: `whole_brain`. May be overridden with `whole_brain`, `region`, or `both`. |
| `mapmycells_reference_atlas` | species-dependent | Human: Whole Human Brain (`whb`); mouse: Yao 2023 Whole Mouse Brain (`wmb`). |
| `mapmycells_query_species` | `species` | Query species. Human-to-WMB mapping enables Allen ortholog mapping; mouse queries require WMB. |
| `mapmycells_auto_download_references` | `true` | Download missing published stats, markers, and the WMB gene-mapper DB into the durable cache. |
| `mapmycells_marker_lookup_path` | `null` | Optional explicit marker JSON. When unset, download the atlas-appropriate Allen asset. |
| `mapmycells_precomputed_stats_path` | `null` | Optional explicit stats H5. When unset, download the atlas-appropriate Allen asset. |
| `mapmycells_gene_mapping_db_path` | `null` | Optional `mmc_gene_mapper` SQLite DB. Human-to-WMB runs download it when automatic downloads are enabled. |
| `mapmycells_region_name` | species-dependent | Human: `frontal_a44_a45_a46_a32_acc`; mouse: `region`. Short safe name used in region output directories and annotation prefixes. |
| `mapmycells_region_labels` | species-dependent | Human defaults to the four frontal WHB labels. Mouse defaults empty and requires explicit WMB `region_of_interest_acronym` values for region mode. |
| `mapmycells_region_cache_dir` | `<outdir>/mapmycells_cache` | Durable cache for Allen WHB/WMB downloads, the gene mapper, and generated region reference files. The Dwight profile overrides this with `/media/mathieubo/SSD1/MerXen/mapmycells`. |
| `mapmycells_region_min_cells_per_leaf` | `10` | Drop region taxonomy leaf aliases with fewer cells than this before precomputing stats. |
| `mapmycells_region_force_rebuild` | `false` | Rebuild the generated region reference even if matching cached files exist. |
| `mapmycells_region_query_markers_n_per_utility` | `10` | Marker count target passed to Allen's `QueryMarkerRunner` for the region reference. |
| `mapmycells_drop_level` | `null` | Optional taxonomy level to drop before mapping. |
| `mapmycells_normalization` | `raw` | Query normalization passed to MapMyCells. |
| `mapmycells_bootstrap_factor` | `0.9` | Marker downsampling factor for bootstrapping; default keeps the historical spatial-data setting. |
| `mapmycells_bootstrap_iteration` | `100` | Number of bootstrap iterations. |
| `mapmycells_n_processors` | Dwight: `8` | Number of worker processes passed to MapMyCells. |
| `mapmycells_chunk_size` | `null` | Optional cells-per-worker chunk size. |
| `mapmycells_rng_seed` | `null` | Optional mapper random seed. |
| `mapmycells_max_gb` | `null` | Optional memory budget for H5AD conversion. |
| `mapmycells_tmp_dir` | `null` | Optional fast temporary directory for mapper scratch data. |
| `mapmycells_cloud_safe` | `false` | Passed to MapMyCells `cloud_safe`. |
| `mapmycells_flatten` | `false` | Flatten taxonomy and map directly to leaf nodes. |
| `mapmycells_verbose_csv` | `false` | Include verbose confidence columns when supported by the mapper. |
| `mapmycells_plots_only` | `false` | Reuse existing mapper CSV/extended JSON outputs in published `mapmycells_out/` and regenerate only the annotated H5AD and plots. |
| `mapmycells_query_layer` | `counts` | AnnData layer copied into `X` before mapping. Use `null` to keep current `X`. |
| `mapmycells_gene_id_column` | `ensembl_id` | `var` column used as query gene identifiers. Human `ENSG` and mouse `ENSMUSG` IDs are preserved; missing mouse IDs can be recovered from WMB metadata or the gene-mapper DB. |
| `mapmycells_obs_id_column` | `null` | Optional `obs` column used as cell identifiers for the query H5AD. |

### Resource limits

The reserved `standard` profile includes
[dwight.config](../workflows/conf/dwight.config), so running without
`-profile` selects Dwight automatically. `-profile dwight` is the explicit
equivalent. When combining the workstation settings with another profile, name
both; for example, use `-profile dwight,conda`. Selecting only an explicit
`conda`, `apptainer`, or HPC profile suppresses Nextflow's implicit `standard`
profile and therefore does not inherit Dwight's host settings.

| Param | Default | Description |
|-------|---------|-------------|
| `max_ram_gb` | Dwight: `640` | System memory limit passed to `MemoryConfig`. |
| `warn_ram_gb` | Dwight: `600` | RAM warning threshold. |
| `transcript_chunk_rows` | Dwight: `1_000_000` | Points chunk size when streaming transcripts. |

Dwight advertises 72 CPUs and 640 GB of usable capacity to the local executor:

```groovy
executor {
    cpus = 72
    memory = "640 GB"
}
```

Those values limit aggregate local scheduling; they are not default requests
for every task. Portable per-process CPU/memory requests remain in
[nextflow.config](../workflows/nextflow.config), while the following
`maxForks` guards belong to the Dwight profile:

| Process | CPUs | Memory | Max forks |
|---------|-----:|-------:|-----------|
| `BUILD_SPATIALDATA` | 8 | 80 GB | `build_spatialdata_max_forks` = 3 |
| `CELLPOSE_SEGMENT` | 12 | 212 GB | `cellpose_segment_max_forks` = 1 |
| `PROSEG_SEGMENT` | 32 | 220 GB | `proseg_segment_max_forks` = 2 |
| `ENRICH` | 8 | 300 GB | unbounded |
| `VIEWER_CACHE` | 8 | 60 GB | `viewer_cache_max_forks` = 9 |
| `QC` | 4 | 24 GB | unbounded |
| `ALIGN` | 12 | 100 GB | `alignment_max_forks` = 1 |
| `ALIGN_QC` | 4 | 32 GB | unbounded |
| `COMPARE` | 4 | 32 GB | unbounded |
| `VISUALIZE` | 4 | 32 GB | unbounded |
| `MECR_REFERENCE` | 16 | 240 GB | 1 |
| `MECR` | 4 | 48 GB | `mecr_max_forks` = 4 |
| `CLUSTERING_SQUIDPY` | 8 | 32 GB | `clustering_squidpy_max_forks` = 4 |
| `MAPMYCELLS` | 8 | 160 GB | unbounded |

On Dwight, `CELLPOSE_SEGMENT`, `ALIGN` when `alignment_device != "cpu"`, and
`CLUSTERING_SQUIDPY` when `clustering_squidpy_use_gpu=true` also share the
host-wide `gpu_process_lock_file`. The lock is held for the full process shell,
then released automatically when the task exits.

A future HPC profile should provide its own executor, capacity/concurrency,
software paths, reference paths, worker counts, and GPU policy. The portable
process requests and scientific defaults do not need to be duplicated.

All processes use `errorStrategy = "ignore"` with
`workflow.failOnIgnore = true`. A failed task therefore stops only branches that
depend on its missing outputs, while unrelated samples continue. The overall
Nextflow run still exits non-zero if any task failure was ignored.

## Pydantic config models

Every CLI subcommand takes `--config <path>.json` and validates the JSON
against a Pydantic model. Adding, removing, or renaming fields in these
models is the ground truth for how stages are configured.

| Model | Stage | File |
|-------|-------|------|
| `SpatialDataBuildConfig` | `build-spatialdata` | [config.py:112](../src/merxen/config.py#L112) |
| `SegmentationConfig` | `segment` | [config.py:146](../src/merxen/config.py#L146) |
| `EnrichmentConfig` | `enrich` | [config.py:157](../src/merxen/config.py#L157) |
| `QCConfig` | `qc` | [config.py:169](../src/merxen/config.py#L169) |
| `AlignmentConfig` | `align` | [config.py](../src/merxen/config.py) |
| `AlignmentQCConfig` | `alignment-qc` | [config.py](../src/merxen/config.py) |
| `ComparisonConfig` | `compare` | [config.py](../src/merxen/config.py) |
| `VisualizationConfig` | `visualize` | [config.py](../src/merxen/config.py) |
| `MecrReferenceConfig` | `mecr-reference` | [config.py](../src/merxen/config.py) |
| `MecrConfig` | `mecr` | [config.py](../src/merxen/config.py) |
| `ClusteringSquidpyConfig` | `clustering-squidpy` | [config.py](../src/merxen/config.py) |
| `MapMyCellsConfig` | `mapmycells` | [config.py](../src/merxen/config.py) |

Nested sub-models:

- `CellposeConfig`, `TilingConfig`, `MaskFilterConfig` — Cellpose behaviour.
- `ProsegConfig` — ProSeg parameters, including `binary_path`.
- `MemoryConfig` — memory limits and chunk sizes.
- `DatasetConfig` — one dataset (one half of a pair) within a `SegmentationConfig`.
- `MecrSampleConfig` — one SpatialData cell-count table scored by MECR.
- `MerscopeBuildConfig` / `XeniumBuildConfig` — platform-specific build options
  nested under `SpatialDataBuildConfig`.
- `PipelineConfig(BaseSettings)` — top-level, loaded from `MERXEN_*` env vars.

`load_config_from_json(path, cls)` in
[config.py:205](../src/merxen/config.py#L205) is the helper every CLI entry
point uses to parse and validate.

## Precedence

1. CLI flags passed to `nextflow run` override `nextflow.config` defaults.
2. `nextflow.config` defaults populate the JSON config written into the work
   directory.
3. The Python stage loads that JSON, validated through the Pydantic model.
4. `MERXEN_*` environment variables only affect code that instantiates
   `PipelineConfig()` directly — they do **not** back-propagate into
   `nextflow.config`. Set them explicitly via `--<name>` if you need them in
   Nextflow.

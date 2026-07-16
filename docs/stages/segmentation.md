# Stage 2 — Segmentation

Runs two image-based segmentations: a DAPI-only Cellpose `nuclei` model and a
cell segmentation on DAPI/PolyT (MERSCOPE) or DAPI/18S (Xenium). The cell mask
is then refined using the actual transcript positions with ProSeg. The nuclei
mask remains independent of transcript assignment and is retained for
subcellular transcript analysis.

## What it does

1. Run DAPI-only Cellpose with the `nuclei` model and write an independently
   reusable nuclei mask.
2. Load the SpatialData zarr from stage 1 and pick the cell channels.
3. Tile the image, run Cellpose on each tile, merge masks back into a
   single global-pixel-coordinate mask array.
4. Filter both masks with the configured 5–400 µm² final area bounds.
5. Convert the Cellpose mask from pixel coordinates to microns using the
   platform transform matrix.
6. Export transcripts to a ProSeg-friendly CSV, seeded with the cell id each
   transcript falls inside (from the Cellpose mask).
7. Resolve or bootstrap the external ProSeg binary, then run it against that
   CSV + mask.
8. Convert ProSeg's raw output zarr to "latest" SpatialData format.

## Nextflow subworkflow

The module has three independently scheduled processes per dataset:

| Process | Tool | Default resources | Scheduler route |
|---------|------|-------------------|-----------------|
| `CELLPOSE_NUCLEI_SEGMENT` | DAPI-only Cellpose `nuclei` model | 12 CPUs, 212 GB RAM, `cellpose_segment_max_forks = 1` | Same GPU queue and shared GPU lock as cell Cellpose |
| `CELLPOSE_SEGMENT` | Cellpose-SAM | 12 CPUs, 212 GB RAM, `cellpose_segment_max_forks = 1` | GPU queue and one GPU when `cellpose_gpu=true`; local GPU lock on workstations |
| `PROSEG_SEGMENT` | ProSeg | 32 CPUs, 220 GB RAM, `proseg_segment_max_forks = 2` | CPU/HTC queue; no GPU request, lock, or Apptainer `--nv` |

The two Cellpose process types use the same file lock, so only one GPU job can
run at a time even across samples. `segment_nuclei` is a separate selectable
pipeline stage immediately before `segment`:

```bash
nextflow run workflows/main.nf \
  --samplesheet workflows/samplesheet.csv \
  --only_stage segment_nuclei
```

A routine range run includes both stages. `--only_stage segment` reuses the
published nuclei mask and fails early if it is absent.

- **Input:** `tuple(key, pair_id, platform, seg_config_json)`.
- **CLIs:** `merxen cellpose-nuclei-segment --config segment_config.json`,
  `merxen cellpose-segment --config segment_config.json`, followed by
  `merxen proseg-segment --config segment_config.json ...`. The original
  `merxen segment` command remains available for direct, single-process use.
- **Output:**
  - `segment_out/proseg_base_latest.zarr` — refined segmentation as SpatialData.
  - `segment_out/cellpose_masks_tiled.npy` — global-pixel mask (uint32).
  - `segment_out/cellpose_nuclei_masks_tiled.npy` — DAPI-only nuclei labels.
  - `segment_out/transcripts_for_proseg.csv` — transcripts with seeded cell ids.
  - `segment_out/cellpose_transforms.json` — affine metadata handed from
    Cellpose to ProSeg.
- **publishDir:** `${outdir}/${pair_id}/${platform}/segmentation/` (symlink mode).

The durable latest zarr written by this stage lives at
`${outdir}/${pair_id}/${platform}/latest/latest_spatialdata.zarr`. The
`segment_out/proseg_base_latest.zarr` in the work dir is a staged symlink to
that path.

## Python entry points

| Function | File |
|----------|------|
| CLI `segment_command` | [cli/run_segmentation.py](../../src/merxen/cli/run_segmentation.py) |
| Cellpose stage `run_cellpose_segmentation` | [segmentation/pipeline.py](../../src/merxen/segmentation/pipeline.py) |
| Nuclei stage `run_cellpose_nuclei_segmentation` | [segmentation/pipeline.py](../../src/merxen/segmentation/pipeline.py) |
| ProSeg stage `run_proseg_segmentation` | [segmentation/pipeline.py](../../src/merxen/segmentation/pipeline.py) |
| Compatibility orchestration `run_segmentation_pipeline` | [segmentation/pipeline.py](../../src/merxen/segmentation/pipeline.py) |
| Tiled Cellpose `run_tiled_cellpose` | [segmentation/cellpose.py:255](../../src/merxen/segmentation/cellpose.py#L255) |
| Mask filter `filter_cell_by_regionprops` | [segmentation/mask_filter.py:78](../../src/merxen/segmentation/mask_filter.py#L78) |
| Final area filter `filter_labeled_mask_by_area` | [segmentation/mask_filter.py](../../src/merxen/segmentation/mask_filter.py) |
| Masks → polygons `masks_to_polygons` | [segmentation/mask_geometry.py:84](../../src/merxen/segmentation/mask_geometry.py#L84) |
| ProSeg subprocess `run_proseg_refinement` | [segmentation/proseg.py:99](../../src/merxen/segmentation/proseg.py#L99) |
| Transcript CSV `write_proseg_csv_from_points` | [io/transcript_io.py:140](../../src/merxen/io/transcript_io.py#L140) |

## Config schema

`SegmentationConfig` — [config.py:146](../../src/merxen/config.py#L146).

```
SegmentationConfig
├── dataset: DatasetConfig
│   ├── name, platform, data_path, channels, output_dir
│   ├── persistent_latest_zarr_path, persistent_mask_path, persistent_transcripts_path
│   ├── persistent_cellpose_stitching_stats_path
│   ├── persistent_nuclei_mask_path, persistent_nuclei_stitching_stats_path
│   ├── MERSCOPE: image_prefix, z_range, transform_path
│   ├── Xenium: xenium_spec_path, min_qv
│   └── proseg_overrides: dict      # per-platform voxel_layers
├── cellpose: CellposeConfig         # model_type, gpu, diameter, thresholds
├── nuclei_cellpose: CellposeConfig  # nuclei model; other inference defaults match
├── mask_filter: MaskFilterConfig    # eccentricity, area percentile
├── nuclei_mask_filter: MaskFilterConfig
├── tiling: TilingConfig             # tile sizes, stitch overlap, duplicate policy
├── proseg: ProsegConfig             # binary path, MCMC params
└── memory: MemoryConfig             # RAM cap, chunk sizes
```

See [Configuration → Pydantic config models](../configuration.md#pydantic-config-models)
for all fields and defaults.

## Walkthrough

1. **Load and prepare images.** For MERSCOPE, image z-planes and channels are
   stacked and projected. For Xenium, morphology focus is used directly.
   Image I/O lives in
   [io/image_source.py](../../src/merxen/io/image_source.py).
2. **Cellpose tiling.** `run_tiled_cellpose` picks a tile size from
   `TilingConfig.tile_size_candidates` (`6144 → 1024`) small enough for
   available RAM, iterates over overlapping tiles, runs Cellpose on each,
   filters each tile's masks by regionprops, and stitches whole objects whose
   centroids fall inside each tile core. Duplicate objects from neighboring
   halos are skipped by overlap thresholds, while accepted objects are pasted
   into a global label space. The result is a `(H, W)` uint32 array saved as
   `cellpose_masks_tiled.npy`.
3. **Transform to microns.** `build_cellpose_affine_to_microns` composes the
   platform transform with any rescale factor. This gives `(x_transform,
   y_transform)` 1D affine components used when writing the ProSeg CSV and
   seeding cell IDs.
4. **Final Cellpose area filter.** The saved mask is memory-mapped, label
   areas are converted to square microns, and masks outside
   `cellpose_final_min_area_um2` / `cellpose_final_max_area_um2` are removed in
   row chunks. The cleaned `cellpose_masks_tiled.npy` is the only mask used by
   the transcript seeding and ProSeg steps.
5. **Seed transcripts.** `write_proseg_csv_from_points` streams the
   transcripts points object in chunks of
   `memory.transcript_chunk_rows`, looks each transcript's pixel location
   up in the mask, and writes a row with `x_micron`, `y_micron`, `z_micron`,
   `feature_name`, `cell_id` (0 if outside any cell). Xenium transcripts
   below `dataset.min_qv` are dropped.
6. **Process handoff.** Nextflow stages the mask, transcript CSV, and affine
   metadata from the GPU process into the CPU process work directory.
7. **ProSeg.** The workflow resolves ProSeg via `ENSURE_PROSEG`; if it is not
   found in the configured search paths, the bootstrap step installs it with
   Cargo. `run_proseg_refinement` then spawns the resolved external binary.
   ProSeg
   uses the Cellpose-seeded `cell_id` column as a prior and performs MCMC
   sampling over the transcript field, letting cell boundaries move to
   better match transcript density.
8. **To "latest" zarr.** `convert_to_latest_zarr` rewrites the raw ProSeg
   output so it can be read with the current SpatialData version, then stages
   that durable zarr back into the work dir for Nextflow.

## Outputs

| File | Contents |
|------|----------|
| `latest/latest_spatialdata.zarr` | Durable refined SpatialData zarr. This is the object enrichment mutates in place. |
| `segmentation/proseg_base_latest.zarr` | Staged symlink to the durable latest zarr. |
| `cellpose_masks_tiled.npy` | Cleaned global-pixel Cellpose labels, consumed by ProSeg and enrichment. |
| `cellpose_nuclei_masks_tiled.npy` | Cleaned DAPI-only nuclei labels, consumed by enrichment and transcript spatial analysis. |
| `cellpose_stitching_stats.json` | Tile stitching diagnostics: accepted labels, duplicates, conflicts, edge-touching labels, and thresholds. |
| `cellpose_nuclei_stitching_stats.json` | Equivalent tile-stitching diagnostics for nuclei. |
| `transcripts_for_proseg.csv` | The transcript CSV fed into ProSeg. Retained for debugging. |
| `cellpose_transforms.json` | Pixel-to-micron affine terms used by the ProSeg process. |

`proseg_base_raw.zarr` is treated as a transient intermediate and removed
after the latest-format zarr is written successfully.

## Memory guardrails

The stage functions free memory aggressively:

- `force_release()` is called after the transcript CSV write and after the
  full run, triggering `gc.collect()` and `torch.cuda.empty_cache()`.
- `enforce_memory_limit` in
  [memory.py:47](../../src/merxen/memory.py#L47) is called while streaming
  transcripts and raises when `MemoryConfig.max_system_ram_gb` is exceeded.
- Tile size auto-selection falls back down `tile_size_candidates` until
  memory fits.

## Common failures

- **ProSeg bootstrap failed** — no executable was found in
  `proseg_search_paths`, `cargo install proseg` failed, or the configured
  `proseg_install_path` needs `sudo` and permission was denied.
- **All transcripts filtered out** — the QV filter (`xenium_min_qv`) is too
  strict, or the points columns didn't resolve. `resolve_col` tries
  `x`, `global_x`, `x_location` and `gene`, `feature_name`, `target`.
- **Cellpose GPU OOM** — lower `cellpose_bsize` or drop the largest entry
  from `tile_size_candidates`. Or pass `--cellpose_gpu false` to force CPU.
- **Zarr half-written on crash** — delete the durable
  `latest/latest_spatialdata.zarr` and rerun with `-resume`.

# Stage 2 — Segmentation

Runs image-based segmentation with Cellpose-SAM on DAPI/PolyT (MERSCOPE) or
DAPI/18S (Xenium), then refines those masks using the actual transcript
positions with ProSeg. The Cellpose output is the **prior** that ProSeg's
MCMC sampler starts from.

## What it does

1. Load the SpatialData zarr from stage 1 and pick the right image channels.
2. Tile the image, run Cellpose on each tile, merge masks back into a
   single global-pixel-coordinate mask array.
3. Filter masks by eccentricity / area (reject junk).
4. Convert the Cellpose mask from pixel coordinates to microns using the
   platform transform matrix.
5. Export transcripts to a ProSeg-friendly CSV, seeded with the cell id each
   transcript falls inside (from the Cellpose mask).
6. Run the external ProSeg binary against that CSV + mask.
7. Convert ProSeg's raw output zarr to "latest" SpatialData format.

## Nextflow process

[`SEGMENT`](../../workflows/modules/segmentation.nf) — one instance per
dataset. This is the heaviest stage (default 75 CPUs, 500 GB RAM).

- **Input:** `tuple(key, pair_id, platform, seg_config_json)`.
- **CLI:** `merxen segment --config segment_config.json`.
- **Output:**
  - `segment_out/proseg_base_latest.zarr` — refined segmentation as SpatialData.
  - `segment_out/cellpose_masks_tiled.npy` — global-pixel mask (uint32).
  - `segment_out/transcripts_for_proseg.csv` — transcripts with seeded cell ids.
- **publishDir:** `${outdir}/${pair_id}/${platform}/segmentation/` (symlink mode).

The durable latest zarr written by this stage lives at
`${outdir}/${pair_id}/${platform}/latest/latest_spatialdata.zarr`. The
`segment_out/proseg_base_latest.zarr` in the work dir is a staged symlink to
that path.

## Python entry points

| Function | File |
|----------|------|
| CLI `segment_command` | [cli/run_segmentation.py](../../src/merxen/cli/run_segmentation.py) |
| Orchestration `run_segmentation_pipeline` | [segmentation/pipeline.py:189](../../src/merxen/segmentation/pipeline.py#L189) |
| Tiled Cellpose `run_tiled_cellpose` | [segmentation/cellpose.py:255](../../src/merxen/segmentation/cellpose.py#L255) |
| Mask filter `filter_cell_by_regionprops` | [segmentation/mask_filter.py:78](../../src/merxen/segmentation/mask_filter.py#L78) |
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
│   ├── MERSCOPE: image_prefix, z_range, transform_path
│   ├── Xenium: xenium_spec_path, min_qv
│   └── proseg_overrides: dict      # per-platform voxel_layers
├── cellpose: CellposeConfig         # model_type, gpu, diameter, thresholds
├── mask_filter: MaskFilterConfig    # eccentricity, area percentile
├── tiling: TilingConfig             # tile sizes, overlap
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
   relabels local IDs into a global label space, and filters each tile's
   masks by regionprops. The result is a `(H, W)` uint32 array saved as
   `cellpose_masks_tiled.npy`.
3. **Transform to microns.** `build_cellpose_affine_to_microns` composes the
   platform transform with any rescale factor. This gives `(x_transform,
   y_transform)` 1D affine components used when writing the ProSeg CSV and
   seeding cell IDs.
4. **Seed transcripts.** `write_proseg_csv_from_points` streams the
   transcripts points object in chunks of
   `memory.transcript_chunk_rows`, looks each transcript's pixel location
   up in the mask, and writes a row with `x_micron`, `y_micron`, `z_micron`,
   `feature_name`, `cell_id` (0 if outside any cell). Xenium transcripts
   below `dataset.min_qv` are dropped.
5. **ProSeg.** `run_proseg_refinement` spawns the external binary. ProSeg
   uses the Cellpose-seeded `cell_id` column as a prior and performs MCMC
   sampling over the transcript field, letting cell boundaries move to
   better match transcript density.
6. **To "latest" zarr.** `convert_to_latest_zarr` rewrites the raw ProSeg
   output so it can be read with the current SpatialData version, then stages
   that durable zarr back into the work dir for Nextflow.

## Outputs

| File | Contents |
|------|----------|
| `latest/latest_spatialdata.zarr` | Durable refined SpatialData zarr. This is the object enrichment mutates in place. |
| `segmentation/proseg_base_latest.zarr` | Staged symlink to the durable latest zarr. |
| `cellpose_masks_tiled.npy` | Global-pixel Cellpose labels, consumed by enrichment. |
| `transcripts_for_proseg.csv` | The transcript CSV fed into ProSeg. Retained for debugging. |

`proseg_base_raw.zarr` is treated as a transient intermediate and removed
after the latest-format zarr is written successfully.

## Memory guardrails

`run_segmentation_pipeline` frees memory aggressively:

- `force_release()` is called after the transcript CSV write and after the
  full run, triggering `gc.collect()` and `torch.cuda.empty_cache()`.
- `enforce_memory_limit` in
  [memory.py:47](../../src/merxen/memory.py#L47) is called while streaming
  transcripts and raises when `MemoryConfig.max_system_ram_gb` is exceeded.
- Tile size auto-selection falls back down `tile_size_candidates` until
  memory fits.

## Common failures

- **`Proseg binary '...' not found or not executable`** — wrong
  `--proseg_binary`, or binary missing the execute bit.
- **All transcripts filtered out** — the QV filter (`xenium_min_qv`) is too
  strict, or the points columns didn't resolve. `resolve_col` tries
  `x`, `global_x`, `x_location` and `gene`, `feature_name`, `target`.
- **Cellpose GPU OOM** — lower `cellpose_bsize` or drop the largest entry
  from `tile_size_candidates`. Or pass `--cellpose_gpu false` to force CPU.
- **Zarr half-written on crash** — delete the durable
  `latest/latest_spatialdata.zarr` and rerun with `-resume`.

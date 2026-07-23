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
   single global-pixel-coordinate mask array, and retain the corresponding
   Cellpose cell-probability logits.
4. Filter both masks with the configured 5–400 µm² final area bounds.
5. Convert the Cellpose mask from pixel coordinates to microns using the
   platform transform matrix.
6. Export transcripts to a ProSeg-friendly CSV, seeded with the cell id each
   transcript falls inside (from the Cellpose mask), with a stable transcript
   id and the Xenium QV or MERSCOPE transcript score.
7. Resolve or bootstrap the external ProSeg binary, then run it against that
   CSV + mask.
8. Convert ProSeg's raw output zarr to "latest" SpatialData format, assigning
   positive canonical instance IDs, repairing stable transcript IDs, and
   recording the versioned segmentation registry.
9. By default, build a separate `MOSAIK_proseg_hybrid` polygon branch from
   supported local convex expansions of the Cellpose masks, apply growth-only
   boundary smoothing, and reassign transcripts with overlap-aware rules.

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
  - `segment_out/cellpose_cellprobs_tiled.npy` — aligned Cellpose probability
    logits (float32) supplied to ProSeg.
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
| Tiled Cellpose `run_tiled_cellpose` | [segmentation/cellpose.py](../../src/merxen/segmentation/cellpose.py) |
| Mask filter `filter_cell_by_regionprops` | [segmentation/mask_filter.py:78](../../src/merxen/segmentation/mask_filter.py#L78) |
| Final area filter `filter_labeled_mask_by_area` | [segmentation/mask_filter.py](../../src/merxen/segmentation/mask_filter.py) |
| Masks → polygons `masks_to_polygons` | [segmentation/mask_geometry.py:84](../../src/merxen/segmentation/mask_geometry.py#L84) |
| ProSeg subprocess `run_proseg_refinement` | [segmentation/proseg.py:99](../../src/merxen/segmentation/proseg.py#L99) |
| Hybrid refinement `run_proseg_hybrid_refinement` | [segmentation/proseg_hybrid.py](../../src/merxen/segmentation/proseg_hybrid.py) |
| Transcript CSV `write_proseg_csv_from_points` | [io/transcript_io.py:140](../../src/merxen/io/transcript_io.py#L140) |

## Config schema

`SegmentationConfig` — [config.py:146](../../src/merxen/config.py#L146).

```
SegmentationConfig
├── dataset: DatasetConfig
│   ├── name, platform, data_path, channels, output_dir
│   ├── persistent_latest_zarr_path, persistent_mask_path
│   ├── persistent_cellpose_cellprob_path, persistent_transcripts_path
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
├── proseg_hybrid: ProsegHybridConfig # optional separate hybrid branch
└── memory: MemoryConfig             # RAM cap, chunk sizes
```

See [Configuration → Pydantic config models](../configuration.md#pydantic-config-models)
for all fields and defaults.

## ProSeg 3.2.0 default comparison

The Xenium and MERSCOPE presets primarily select platform column conventions;
their shared inference defaults come from ProSeg's main argument definitions.
MerXen deliberately overrides several of those defaults:

| Parameter | ProSeg 3.2.0 Xenium/MERSCOPE preset | MerXen |
|-----------|---------------------------------------|--------|
| Post-burn-in `samples` | 200 | 1200 |
| Burn-in voxel size | 2.0 µm | 1.0 µm |
| Final voxel size | 1.0 µm | 0.5 µm |
| Voxel layers | 4 | Xenium 2; MERSCOPE 7 |
| Prior-segmentation reassignment probability | 0.5 | 0.2 |
| Cell initialization | off | on |
| Threads | all available cores | 32 |

The following values match upstream: 200 burn-in samples, nuclear
reassignment 0.2, diffusion probability 0.2, cell compactness 0.04,
initial expansion 0, maximum transcript-to-nucleus distance 60 µm, and
Cellpose probability discount 0.85. Xenium QV ≥20 and negative-control
exclusion now match the Xenium preset, while MERSCOPE `transcript_score` is
passed as ProSeg's quality column without adding a score threshold.

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
   `cellpose_masks_tiled.npy`. Cellpose's raw probability logits are stitched
   over exactly the accepted pixels and saved as a matching float32 array.
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
   `feature_name`, `cell_id` (0 if outside any cell), `transcript_id`, and
   `qv`. Xenium transcripts below `dataset.min_qv` and Xenium control features
   matching `^(Deprecated|NegControl|Unassigned|Intergenic)` are dropped.
   MERSCOPE `transcript_score` is passed through ProSeg as `qv` and restored
   under its platform-native name in the latest zarr.
6. **Process handoff.** Nextflow stages the mask, probability logits,
   transcript CSV, and affine
   metadata from the GPU process into the CPU process work directory.
7. **ProSeg.** The workflow requires ProSeg 3.2.0. `ENSURE_PROSEG` ignores a
   different installed version and builds the pinned source revision with
   Cargo. `run_proseg_refinement` then spawns the resolved external binary.
   ProSeg
   uses the Cellpose-seeded `cell_id` column as a prior and performs MCMC
   sampling over the transcript field, letting cell boundaries move to
   better match transcript density. The matching Cellpose probability-logit
   array is supplied through `--cellpose-cellprobs`.
8. **To "latest" zarr.** `convert_to_latest_zarr` rewrites the raw ProSeg
   output so it can be read with the supported SpatialData version. ProSeg's
   zero-based cell ID becomes provenance in `proseg_internal_id`; the public
   `assignment`, shape `instance_id`, and table `instance_id` use the same
   positive `uint64` Cellpose-label namespace. Transcript IDs are regenerated
   as unique positive values rather than filling missing IDs with zero. Existing
   pre-schema latest stores are upgraded once through a validated temporary
   store and atomically swapped into place.
9. **Hybrid branch.** For every Cellpose label, ProSeg-foreground transcript
   coordinates are filtered with robust nearest-neighbour components. Retained
   transcripts just outside the Cellpose surface can support an expansion
   individually; more distant transcripts must occur in a group of at least
   three and form a neighbour chain back to the Cellpose surface. Each accepted
   group creates a rounded local convex wedge attached to a short Cellpose
   boundary arc and clipped to a fixed `1.0R` dilation cap. Cells below the
   10-transcript evidence threshold have no transcript-driven expansion.
   Finally, every mask receives fixed-micron morphological closing plus a small
   outward offset. The pre-smoothed mask is unioned back in, holes are filled,
   and new area is clipped to the same cap, so smoothing can only grow the mask.
   Transcripts inside one hybrid mask are assigned geometrically; in overlaps,
   only a ProSeg assignment to one of the candidate cells is accepted, including
   a transcript ProSeg marked as background. Otherwise the overlap remains
   unassigned.

## Outputs

| File | Contents |
|------|----------|
| `latest/latest_spatialdata.zarr` | Durable refined SpatialData zarr. This is the object enrichment mutates in place. |
| `segmentation/proseg_base_latest.zarr` | Staged symlink to the durable latest zarr. |
| `cellpose_masks_tiled.npy` | Cleaned global-pixel Cellpose labels, consumed by ProSeg and enrichment. |
| `cellpose_cellprobs_tiled.npy` | Cellpose probability logits aligned to the cleaned mask and consumed by ProSeg. |
| `cellpose_nuclei_masks_tiled.npy` | Cleaned DAPI-only nuclei labels, consumed by enrichment and transcript spatial analysis. |
| `cellpose_stitching_stats.json` | Tile stitching diagnostics: accepted labels, duplicates, conflicts, edge-touching labels, and thresholds. |
| `cellpose_nuclei_stitching_stats.json` | Equivalent tile-stitching diagnostics for nuclei. |
| `transcripts_for_proseg.csv` | The transcript CSV fed into ProSeg. Retained for debugging. |
| `cellpose_transforms.json` | Pixel-to-micron affine terms used by the ProSeg process. |

The durable zarr contains `MOSAIK_proseg_hybrid` SpatialData/GeoParquet
polygons and `table_MOSAIK_proseg_hybrid` counts. Its points element retains
the original ProSeg fields plus `hybrid_assignment`, `hybrid_background`,
`hybrid_candidate_count`, and `hybrid_assignment_source`. This is a third
branch; it does not replace `MOSAIK_proseg`. `hybrid_assignment` is nullable
`UInt64` and directly matches the hybrid shape/table `instance_id`; assignment
source is stored as a finite categorical vocabulary. The root
`merxen_schema.segmentations` registry is the authoritative branch pairing.

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

- **ProSeg bootstrap failed** — no matching 3.2.0 executable was found,
  the pinned Git/Cargo build failed, or the configured
  `proseg_install_path` needs `sudo` and permission was denied.
- **All transcripts filtered out** — the QV filter (`xenium_min_qv`) is too
  strict, or the points columns didn't resolve. `resolve_col` tries
  `x`, `global_x`, `x_location` and `gene`, `feature_name`, `target`.
- **Cellpose GPU OOM** — lower `cellpose_bsize` or drop the largest entry
  from `tile_size_candidates`. Or pass `--cellpose_gpu false` to force CPU.
- **Interrupted element replacement** — MerXen retains a recoverable sibling
  backup until the replacement has been written successfully. Rerun with
  `-resume`; inspect any `.merxen-backup-*` sibling before removing it manually.

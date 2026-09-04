# Stage 1 — SpatialData build

Converts platform-specific raw exports (MERSCOPE region folder, Xenium export
folder) into a standard [SpatialData](https://spatialdata.scverse.org/) zarr
archive. All downstream stages consume this zarr.

## What it does

- Locates a reusable SpatialData zarr if one was provided in the samplesheet.
  If it exists **and** `--force_spatialdata_build` is not set, the build is
  skipped and the existing zarr is staged to the work directory.
- Otherwise, reads the raw platform folder with a platform-specific builder
  and writes out a zarr under
  `${outdir}/${pair_id}/<platform>/spatialdata/source_spatialdata.zarr`.

## Nextflow process

[`BUILD_SPATIALDATA`](../../workflows/modules/spatialdata_build.nf) — one
instance per active dataset. Paired mode creates MERSCOPE + Xenium tasks; a
single-platform mode creates one task per samplesheet row.

- **Input:** `tuple(key, pair_id, platform, build_config_json)`.
- **CLI:** `merxen build-spatialdata --config build_config.json [--force-rerun]`.
- **Output:** `tuple(key, pair_id, platform, path("source_spatialdata.zarr"))`.
- **publishDir:** `${outdir}/${pair_id}/${platform}/spatialdata/` (symlink mode).

## Python entry point

- CLI:
  [`build_spatialdata_command`](../../src/merxen/cli/run_build_spatialdata.py)
- Orchestrator:
  [`build_spatialdata_artifact`](../../src/merxen/io/builders/pipeline.py#L14)
- MERSCOPE builder:
  [`write_merscope_spatialdata`](../../src/merxen/io/builders/merscope.py#L61)
- Xenium builder:
  [`write_xenium_spatialdata`](../../src/merxen/io/builders/xenium.py#L15)

## Config schema

`SpatialDataBuildConfig` — [config.py:112](../../src/merxen/config.py#L112).

| Field | Type | Purpose |
|-------|------|---------|
| `dataset_name` | `str` | Identifier like `EXAMPLE01_MERSCOPE`. |
| `platform` | `"MERSCOPE"` \| `"XENIUM"` | Dispatch key. |
| `input_path` | `Path` | Raw folder **or** existing SpatialData zarr. |
| `output_path` | `Path` | Destination inside the Nextflow work dir. |
| `persistent_output_path` | `Path \| None` | Samplesheet-provided reusable zarr. |
| `merscope_transform_path` | `Path \| None` | Override for the micron-to-mosaic transform CSV. |
| `xenium_spec_path` | `Path \| None` | Override for the Xenium experiment / specs JSON. |
| `merscope` | `MerscopeBuildConfig` | `z_layers`, `region_name`, `slide_name`. |
| `xenium` | `XeniumBuildConfig` | Boundary / label / transcripts toggles. |

## Walkthrough

A paired row with `pair_id=EXAMPLE01` fans out to **two**
`BUILD_SPATIALDATA` tasks, one per platform. In `--analysis_mode merscope` or
`--analysis_mode xenium`, it fans out only to the selected platform:

1. Nextflow composes a `build_config.json` with `platform` set and
   `input_path` pointing at the samplesheet value.
2. `merxen build-spatialdata` loads the JSON, validates it through
   `SpatialDataBuildConfig`, and calls `build_spatialdata_artifact`.
3. `_find_reusable_source` checks whether `persistent_output_path` or
   `input_path` already looks like a `.zarr`. If yes, and `force_rerun` is
   false, the existing zarr is staged to `output_path` as a symlink.
4. Otherwise, the platform-specific builder reads the raw export, builds a
   `SpatialData` object (images, shapes, points, metadata tables), and writes
   it to the target path using
   [`write_spatialdata_zarr`](../../src/merxen/io/spatialdata_io.py#L17).
   MERSCOPE images are stored as a single `MERSCOPE_z_projection` max
   projection over the selected z range, not as one image element per z plane.
   Plane `0` is the MERSCOPE fiducial-bead layer, so the selected range should
   normally start at plane `1`. Xenium morphology images are already projected
   by the platform and do not use this range.
5. The resulting path is returned and flows into the next Nextflow channel.

## Caching behaviour

| Samplesheet state | `force_spatialdata_build` | Result |
|-------------------|----------------------------|--------|
| Only `<platform>_dir` set, no cache exists | any | Build from raw. |
| `<platform>_spatialdata_path` set and exists | `false` (default) | Reuse — skip build. |
| `<platform>_spatialdata_path` set and exists | `true` | Rebuild from raw; requires `<platform>_dir` to also be set. |
| Neither `<platform>_dir` nor an existing zarr | any | Fails at samplesheet validation. |

## Common pitfalls

- **MERSCOPE plane 0 included in the projection.** Plane `0` contains fiducial
  beads rather than the biological image stack. Including it in
  `merscope_z_range` can dominate or distort the DAPI/PolyT max projection and
  every image-derived downstream result. Set an explicit range beginning at
  `1`, such as `1-7`. This does not apply to Xenium, whose morphology image is
  already projected.
- **MERSCOPE transform missing.** The builder expects
  `micron_to_mosaic_pixel_transform.csv` either inside the region folder or
  at `merscope_transform_path`. Without it, segmentation will also fail.
- **Xenium spec ambiguity.** If the Xenium export contains a non-standard
  `specs.json` location, set `xenium_spec_path` in the samplesheet.
- **Partial builds.** If a build crashes half-way, delete the target zarr
  before rerunning — SpatialData does not currently distinguish a
  half-written zarr from a valid one.

# Stage 1 — SpatialData build

Converts platform-specific raw exports (MERSCOPE region folder or `.vzg2`
archive, Xenium export folder) into a standard
[SpatialData](https://spatialdata.scverse.org/) zarr archive. All downstream
stages consume this zarr.

## What it does

- Locates a reusable SpatialData zarr if one was provided in the samplesheet.
  If it exists **and** `--force_spatialdata_build` is not set, the build is
  skipped and the existing zarr is staged to the work directory.
- Otherwise, reads the raw platform folder with a platform-specific builder
  and writes out a zarr under
  `${outdir}/${pair_id}/<platform>/spatialdata/source_spatialdata.zarr`.
- For raw-TIFF MERSCOPE data from older, bead-coated coverslips, can
  optionally subtract the fiducial beads from the PolyT and DAPI planes before
  the max projection. This is off by default; see
  [MERSCOPE bead removal](#merscope-bead-removal-optional).

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
| `input_path` | `Path` | Raw folder, direct MERSCOPE `.vzg2`, **or** existing SpatialData zarr. |
| `output_path` | `Path` | Destination inside the Nextflow work dir. |
| `persistent_output_path` | `Path \| None` | Samplesheet-provided reusable zarr. |
| `merscope_transform_path` | `Path \| None` | Override for the micron-to-mosaic transform CSV. |
| `xenium_spec_path` | `Path \| None` | Override for the Xenium experiment / specs JSON. |
| `merscope` | `MerscopeBuildConfig` | `z_layers`, `region_name`, `slide_name`, `allow_transform_override_mismatch`, and the optional `bead_removal` block ([below](#configuration)). |
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
   by the platform and do not use this range. A `.vzg2` normally contains one
   Vizualizer image plane; MerXen stores that plane under the same projection
   key and records its source z-layer in the SpatialData attributes.
5. The resulting path is returned and flows into the next Nextflow channel.

## Caching behaviour

| Samplesheet state | `force_spatialdata_build` | Result |
|-------------------|----------------------------|--------|
| Only `<platform>_dir` set, no cache exists | any | Build from raw. |
| `<platform>_spatialdata_path` set and exists | `false` (default) | Reuse — skip build. |
| `<platform>_spatialdata_path` set and exists | `true` | Rebuild from raw; requires `<platform>_dir` to also be set. |
| Neither `<platform>_dir` nor an existing zarr | any | Fails at samplesheet validation. |

A reused MERSCOPE zarr must also agree with the requested bead-removal setting;
see [Caching and how to rebuild](#caching-and-how-to-rebuild).

## MERSCOPE bead removal (optional)

Older MERSCOPE coverslips carry fluorescent fiducial beads a fraction of a
micron below plane `0`. A bead is a sharp dot in plane `0` and a defocused disc
with a bright rim in planes `1` and above. The rim is about 30 px in radius at
plane `1` and grows by about 23 px per plane. Planes `1`-`6` therefore still
carry a ring and disc for every bead, and the max projection keeps the
brightest of them. In bead-coverslip data this puts Cellpose masks on beads
(about 190 of 504 masks in the validation cores, against about 10 expected by
chance) and adds a bead pedestal to the PolyT and DAPI intensities.

With bead removal on, the build subtracts a model of each bead before the max
projection. Only the PolyT and DAPI planes are corrected. Every other stain
(the antibody channels) is projected exactly as without the step.

### What the step does

1. **Calibrate once per dataset.** Beads are found in plane `0` on a few
   tissue-free tiles (at least 500 px from tissue, 2048 px wide). From them the
   step measures an empirical radial PSF for planes `0`-`7`, the PolyT-to-DAPI
   response, a per-dataset bead classifier, and the noise levels. It decides
   whether the dataset has focused beads at all.
2. **Model each bead.** Chunk by chunk (one task per 4096 px store chunk, with a
   256 px halo), beads are detected in plane `0` and the library PSF is fitted
   to each bead in every plane with one gain per bead and plane.
3. **Subtract.** The modelled light is subtracted from every plane in
   `merscope_z_range` before the max projection. The subtraction can only
   remove light: it never goes below the local lower envelope of the plane and
   never adds signal.

Plane `0` is read for detection only. It is never corrected or projected, so the
range must start at plane `1` or later (for example `1-6`).

### The bead classifier, its limits and the rescue

Which plane-`0` sources are beads is decided per dataset. The calibration takes
the bright, isolated beads of its tiles that are spectrally confirmed (PolyT and
DAPI light in the ratio of a bead) and show a ring at plane `1`, and derives the
classifier thresholds from them, each clamped to a safe range. A source is a
bead when it passes the amplitude, plane-`1` centre, spectral and plane-`0` ring
tests of those thresholds.

**Limit of the baseline classifier.** The plane-`0` ring test is disturbed
where beads sit close together (clusters), where the focus varies strongly
across the mosaic, and in sparse-bead data whose beads are focused differently
from the calibration tiles. Such beads fail the ring test and stay
**uncorrected**. The calibration measures this on held-out tiles and the QC says
so:

- the recall of the isolated reference beads is reported
  (`summary.json["classifier"]["recall"]`) and warns (`classifier_recall`) only
  when it rests on at least 20 beads and is below 0.90;
- the recall that includes beads in clusters
  (`recall_with_clusters`) warns when it rests on at least 50 beads and is below
  0.75. The isolated set cannot see beads that fail in clusters; this number can.

**The rescue.** A candidate that is spectrally and in plane `1` a bead but fails
the plane-`0` ring test on the low side (a ring weaker than the threshold) is
accepted as a bead when all of these hold:

1. its plane-`0` amplitude is at least 1,000 counts;
2. the texture around it is low relative to its amplitude: the plane-`1` null sd
   divided by its amplitude (`sd1_rel`) is at most the dataset's
   `rescue_sd1_max`;
3. it has no secondary focus: its contrast at planes `2` to `6` (those the
   dataset has) stays within the plane-`1` bead tolerance;
4. its raw plane-`1` ring is coherent at all 16 azimuth sectors (t-score at
   least 4, measured at the detection centre).

The texture gate `rescue_sd1_max` is derived per dataset: the 98th percentile of
`sd1_rel` of the isolated reference beads, clipped to 0.02-0.08. Clean-bead
datasets get a low gate and the rescue stays almost silent; textured data get a
higher one. With fewer than 50 texture measurements the rescue stays off. The
rescue is also **off for every calibration written without the gate**: such a
file loads unchanged, behaves as before and keeps its calibration hash. A
calibration whose rescue is on has a different hash, so a cached zarr built
with the other setting is flagged by the cache guard. The CPU and the GPU
backend apply the same rule (the plane-`1` ring scores are bitwise equal).
Planes `2`-`6` are read for the reference stain whenever the rescue (with its z-profile test) is on.

What it buys, on a development dataset with a dense bead coating where many beads
fail the ring test: the bead-centred Cellpose masks over four crops
fell from 58 to 29 (about 17 expected by chance), and from 45 to 16 in the
densest crop. On the datasets whose beads already pass the ring test (the
primary-type validation sets) it rescued no source in the census tiles.

**What it does not fix, and its risks.** Keep these in mind when reading a
corrected dataset:

- **The cluster warning stays.** On the dense development dataset the recall
  with clusters is still 0.57 with the rescue, so `classifier_recall` still
  warns there. The texture gate costs some bead recall in tissue on purpose:
  beads in textured surroundings are not rescued.
- **The rescue can remove a compact tissue object in flat surroundings** (a
  cell-edge punctum) with a bead PSF. It is rare (about 0.08 rescues per
  1,024 px tissue tile on bead-free tissue measured with the dense dataset's
  thresholds; 2 false rescues in 24 such tiles) and the local damage was at most
  about 600 counts within about 30 px. In validation windows with beads injected
  into tissue it also removed two tissue puncta (123 to 264 counts). It is small
  next to what the baseline classifier already removes on the same tissue (about
  56 accepted sources per 1,024 px tile on bead-free tissue).
- **The gate depends on which tiles calibrated.** The same dataset gave gates of
  0.031, 0.036 and 0.047 on three disjoint tile sets; the choice only matters
  where the rescue acts at all.
- **Antibody stains are not used** to veto a rescue (a puncta guard was
  dropped: inert in textured tissue).

`summary.json["rescue"]` reports whether the rescue was active, `rescue_sd1_max`
with the raw percentile it was clipped from, the z-profile depth, the rescued
beads of the calibration tiles (`n_rescued_tiles`, `rescued_fraction_tiles`) and
of the windows (`n_rescued_windows`, `rescued_fraction`, which counts window
beads including their halos like `n_beads`), and the residual risk above. The
per-chunk count is the `n_rescued` column of `chunks.csv`.

### When to use it

- The coverslip is an older, beaded generation and the raw PolyT plane `1`
  shows bead rings. Open a raw `mosaic_PolyT_z1.tif` crop away from tissue:
  rings about 30 px in radius, centred on bright dots in plane `0`, mean the
  step applies. A clean plane `1` means it does not.
- You want Cellpose and the image-derived measurements free of bead artefacts.

New slide generations probably have no beads. The step stays off by default and
refuses a dataset in which it finds none (`BeadCalibrationError`), so switching
it on for a bead-free dataset fails the build rather than silently doing
nothing.

### Enabling it

Set the Nextflow param for every MERSCOPE row, or the samplesheet column for
single rows (the column wins when it is non-empty):

```bash
nextflow run workflows/main.nf --samplesheet samples.csv \
    --merscope_bead_removal true
```

```csv
pair_id,merscope_dir,merscope_z_range,merscope_bead_removal
EXAMPLE01,/path/to/merscope/EXAMPLE01,1-6,true
EXAMPLE02,/path/to/merscope/EXAMPLE02,1-6,false
```

The preflight check refuses a row that cannot use the step, before any task
starts:

| Refused | Why |
|---------|-----|
| `merscope_z_range` starting at `0` | Plane `0` is used for detection only. Use `1-6`. |
| a range ending above `7` | The PSF library covers planes `0`-`7`. A range ending at `7` on a dataset with fewer planes is refused by the build with the planes it is missing. |
| a `.vzg2` archive | An archive holds one already projected plane. Bead removal needs the raw `images/mosaic_<stain>_z<N>.tif` planes. |

The corrected stains must be uint16 mosaics with square store chunks, and both
PolyT and DAPI must be present with planes `0` to the highest requested plane.

### Configuration

`MerscopeBuildConfig.bead_removal` is `None` (off) unless the workflow adds it.
Nextflow only ever sets `enabled`, `fill` when the fill param is on and `device`
when the device param is not `cpu`; the other fields are fixed algorithm
settings for Python callers and tests. Unknown keys are rejected.

| Field | Default | Meaning |
|-------|---------|---------|
| `enabled` | `false` | Turn the correction on. |
| `method` | `psf_subtract` | The only method. |
| `stains` | `["PolyT", "DAPI"]` | Stains that are corrected. |
| `reference_stain` | `PolyT` | Stain the PSF library is measured on; must be in `stains`. |
| `detection_plane` | `0` | Must be `0` in this version. |
| `halo_px` | `256` | Read margin around each 4096 px chunk. At least 256 and a multiple of 16, so chunk results do not depend on the chunk grid. |
| `max_workers` | `5` | Upper bound for the process pool. The pool is also sized from the available memory (80 % of it, less 8 GB for the parent). |
| `calibration_tiles` | `6` | Tissue-free tiles used for calibration. |
| `calibration_tile_px` | `2048` | Width of a calibration tile. |
| `min_offtissue_distance_px` | `500` | Minimum distance of a calibration tile from tissue. |
| `min_ring_prominence` | `0.010` | Gate on the plane-1 ring that decides whether a tile has beads. |
| `fill` | `false` | The experimental fill below. Requires `enabled`. |
| `device` | `cpu` | `cpu`, `cuda` or `auto`: the device that runs the correction ([GPU backend](#gpu-backend-optional)). The fill forces `cpu`. The device is not part of the parameter hash. |

### Experimental fill (`merscope_bead_removal_fill`)

`--merscope_bead_removal_fill true` also fills the small residual of bead light
that the subtraction leaves in the reference stain, with an estimate of the flat
background around it. It is **experimental and off by default.** It is one
pipeline param with no samplesheet column, applies only to samples that have
bead removal on, and is refused without it.

Treat it as flat background only. Where the surroundings are not flat (tissue,
and in practice nearly every tissue window) it does nothing, and it never adds
light to structure. Its benefit is small and limited to bead rings off tissue.
It costs about 4-8 % more CPU per chunk and about 1.7 GB more peak memory per
worker. The fill is never used to judge whether a sample passes. The QC warns
`fill_guard` when more than half of the fill's candidate pixels are skipped as a
densely coated area. `summary.json["fill"]["counters"]` are pixel totals summed
over the chunks; each counts the whole window of its chunk including the 256 px
halo, so they exceed the pixels of the output by about 1.3 times (the
`fill_guard` ratio is not affected). The cache guard treats the fill like the switch itself (see
below).

### Caching and how to rebuild

Every raw-TIFF MERSCOPE build records what was done in the root attribute
`merxen_image_preprocessing` of `source_spatialdata.zarr`, also when bead
removal is off. It holds the z planes, the stains, the bead-removal switch,
method and version, a hash of the parameters, a hash of the calibration, the
fill setting, the device, the calibration thresholds and rho, and a run
summary. A zarr built before this record existed reads as "bead removal off".

A cached zarr is **never** reused for a request that disagrees with its record:

| Cached zarr | Request | Result |
|-------------|---------|--------|
| bead removal on | bead removal off | `ValueError`, no reuse |
| bead removal off or built before the record | bead removal on | `ValueError`, no reuse |
| bead removal on, fill on | fill off (or the reverse) | `ValueError`, no reuse |
| same switch and fill, but another z range, version, parameter hash or device | same | reused with a `WARNING` that names the difference |

The same rules apply when the build stage is skipped (`--start_stage` or
`--only_stage` past `build`): the workflow then reads the published
`source_spatialdata.zarr`, and the preflight compares its record with the
requested `merscope_bead_removal` and `merscope_bead_removal_fill` in Groovy
before any task starts. A mismatch (the zarr has bead removal on and off is
requested, the reverse, or the fills differ) stops the run with the remedy
below; a row is never run silently on the other image.

To change the setting of a sample that has a cached zarr:

1. Set `merscope_dir` in the samplesheet (the build needs the raw folder) and
   run with `--force_spatialdata_build true`.
2. A forced rebuild deletes the old zarr **before** it builds. If the new build
   then fails (for example `BeadCalibrationError`), the cache is gone and must
   be rebuilt from the raw data.
3. Cellpose masks that were computed from the old image are not reused for the
   new one. Segmentation writes a sidecar next to each mask,
   `image_preprocessing.json` for cell masks and
   `nuclei_image_preprocessing.json` for nuclei masks, holding the bead-removal
   identity of the image they came from. A mask is reused only when its sidecar
   equals the source zarr's record. A mismatch, an unreadable sidecar, or masks
   without a sidecar next to a bead-corrected source stops the run with an
   error. Move the segmentation outputs aside (the `segmentation/` and the
   `latest/` folders) and recompute the masks with `merxen cellpose-segment
   --force-rerun` (CLI) or a new `--outdir` (Nextflow has no per-step force
   flag). The persistent ProSeg latest zarr is checked the same way (its root
   holds `segmentation_image_preprocessing.json`): it is not reused for masks of
   another image, and `--force_proseg_rerun true` rebuilds it from the new
   masks ([segmentation.md](segmentation.md)).
4. Enrichment refuses a latest zarr whose record differs from the source's, then
   copies `merxen_image_preprocessing` onto it (also when the store is already
   enriched), and mask image quantification records it in its table (see
   [Downstream](#downstream-and-what-not-to-pool)).

### The QC folder

When the build ends, the step writes `bead_removal_qc/` inside
`source_spatialdata.zarr`. It is published with the zarr and survives the
conversion to the latest zarr. Writing it adds a few minutes after the zarr is
written (an overview read, six window re-fits and the figures). A failure while
writing the report is logged with its traceback and does not fail the build.

| File | What it shows |
|------|---------------|
| `summary.json` | `status` (`pass` or `warn`), the list of warnings, version, parameter and calibration hashes, corrected stains, bead density (A0 >= 400 and >= 1,600 counts per mm2), runtimes, workers, peak memory, the classifier recall numbers, the `rescue` block and the PSF library spread. It also names the backend (device, update order and, on the GPU, the torch, Triton and CUDA versions and the GPU name) and, for a GPU run, the `gpu` block: seconds waited for the GPU lock, seconds the lock was held, GPU seconds in total and per window (median, maximum), reader threads and windows read ahead. |
| `calibration.json` | Per-tile status, ring prominence and bead counts, the presence decision with its margins, focus check, classifier thresholds with the recall measurements, rho of every stain, noise per plane, and the library ring radius and flux against the 7 + 22.8 z law. |
| `psf_library.npz`, `psf_montage.png` | The PSF templates, radial profiles for planes `0`-`7`, and the per-tile templates overlaid. |
| `overview.png` | 1/16 overview with the tissue outline, calibration tiles (green pass, red fail) and the chunk grid. |
| `beads.csv.gz` | One row per modelled bead. Capped at 25 MB by dropping the faintest beads; `summary.json` says when that happened. |
| `bead_density.png` | Modelled beads per mm2 with the chunk grid. |
| `stretch_field.png` | Median PSF stretch per 256 px cell (field-of-view tilt). |
| `chunks.csv` | Per chunk: counts of candidates, beads, rescued beads and pruned beads, time, memory, the fraction of pixels changed by more than 10 and 100 counts per stain and plane, removed light, saturated pixels, and the residual ring scores. A GPU run adds `gpu_s`, the seconds that chunk kept the GPU busy. |
| `residual_profiles.json`, `.png` | Median residual bead profile at planes `1`-`4`, on and off tissue. |
| `before_after.png`, `before_after_dapi.png`, `before_after_crops.npz` | Six 768 px locations: raw projection, corrected projection, their difference, and PolyT plane `1` before and after. |
| `change_overview.png` | 1/16 overview of the light removed from PolyT and DAPI. |

How to read it:

- **Status.** `pass` means no warning. `warn` lists what to look at; each
  warning is also logged at `WARNING`.
- **Bead density** should be about uniform (`bead_density.png`). Chunk-shaped
  holes next to dense chunks mean a chunk was not corrected.
- **`change_overview.png`** should show no square chunk blocks.
- **Calibration tiles** should agree (`psf_montage.png`). The tile-to-tile
  spread of the z1 ring radius and flux is reported in
  `summary.json["library_spread"]` as numbers, not as a warning.
- **Residual rings.** The share of bright, isolated beads whose plane-1 ring is
  still visible after the correction should be near the control level measured
  at random positions of the raw image, which is also reported. The step warns
  at more than 0.05 for rings and 0.20 for discs, once at least 20 beads were
  scored. These limits come from validation data and have not been re-measured
  on every dataset.
- **Antibody rho** should be zero for every non-corrected stain. A non-zero
  value warns (`antibody_rho`) and the stain is still never corrected.
- **Before/after crops** should show bead discs removed and tissue intact.

Warning codes: `low_tissue_tiles`, `forecast`, `classifier_recall`,
`classifier_thresholds`, `antibody_rho`, `focus_check`, `stretch_fallback`,
`saturation`, `calibration_margin`, `fill_guard`, `residual_ring`,
`residual_disc`, `qc_incomplete`, `qc_size`.

### Runtime and memory

The step adds to the build task's wall time. The numbers in this paragraph are for
the default CPU backend; the [GPU backend](#gpu-backend-optional) is several
times faster. For a
2.6 Gpx mosaic (about 150-175 chunks) on a 40-core host, estimated from
per-chunk timings: about 1.4-2 h with 5-8 workers when the host is otherwise
idle, and about 3.5-4 h when three bead builds share the host (a full run of
169 chunks with 16 workers took 1.1 h). Calibration takes a few minutes of
that. A dense, bright coating is much slower: a dataset of about 370 chunks was
estimated at about 10 h with 8 workers. The calibration logs a cost forecast and
the QC warns (`forecast`) when it exceeds 12 h. Size the task's cpus and memory
for the worker pool: each worker needs about 6-10 GB (plus 1.8 GB with the
fill), and the pool shrinks to fit the memory that is available.

### GPU backend (optional)

The window correction can run on the workstation GPU (PyTorch with fused Triton
kernels) instead of the CPU process pool. It is the same method: the numpy
implementation stays the specification, the test oracle and the only path on
hosts without a GPU. The backend is chosen with `merscope_bead_removal_device`
(`MerscopeBeadRemovalConfig.device`):

- `cpu` (default everywhere, CI and the HPC profiles) is the validated process
  pool described above.
- `cuda` requires a CUDA device, a Triton kernel that compiles and gives the
  right numbers, and a C compiler at run time. The build checks this with a
  64 px smoke test at the start of the sample, before the calibration, and fails
  there with the remedy (`set merscope_bead_removal_device to cpu`) if it
  cannot. The smoke test uses the GPU, so it runs under the host GPU lock
  (`MERXEN_GPU_LOCK_FILE`, when it is set) for a few seconds, in a short-lived
  child process: the build process itself creates no CUDA context and holds
  neither the lock nor GPU memory until the window loop. The smoke test compiles
  only the 64 px kernel specialisation: the first production window compiles its
  own and adds about 5 s. The Triton cache is the folder
  `merxen_triton_cache_<uid>` in the temporary directory (or `TRITON_CACHE_DIR`
  when that is set), shared by every process of the user that sees the same
  temporary directory, so a kernel compiles once and later tasks reuse it.
- `auto` uses the GPU only when CUDA works and `MERXEN_GPU_LOCK_FILE` is set,
  otherwise the CPU. A smoke test failure under `auto` logs a warning and falls
  back to the CPU.

The device is resolved once per sample, before any work, and never changes
mid-sample: windows are not mixed between devices and there is no fallback after
the sample has started. The fill has no GPU implementation, so a build with the
fill on runs on the CPU (with a warning when `cuda` was requested). The code
supports torch 2.6 to 2.14 and Triton 3.2 to 3.8 (CUDA 12.4 to 13.0) and adds no
dependency; the library versions and the GPU name are recorded with the run, and
outputs differ between library versions by at most 0.01 counts.

**How a GPU run is scheduled.** The calibration runs first on the CPU pool, as
before. The GPU phase is then one consumer in the build process: it takes the
host-wide GPU lock (`MERXEN_GPU_LOCK_FILE`, the same lock the Cellpose, alignment
and clustering tasks hold) **once, for the whole window loop**, not for the whole
build, and releases it after the last window or when a window fails. Three
reader threads read the next windows (at most two ahead of the one on the GPU,
about 1.2 GB of float32 planes per 4096 px window; the first two are read while
the lock is awaited, so the GPU does not idle during the first read) and one
writer thread writes
the finished chunks, so reading and writing overlap the GPU work. Window order,
store layout, per-chunk statistics and QC records are identical to the CPU path,
and the result does not depend on the number of reader threads. A build that
waits for the lock holds its `BUILD_SPATIALDATA` slot while it waits; the wait is
logged (`Waiting for MerXen GPU process lock`, `Acquired ... (waited N s)`) and
recorded as `lock_wait_s` in the QC `summary.json`. Without a lock file the run
goes ahead unlocked, with a warning, and may collide with other GPU work. A
process peaks at 5 to 6 GB of allocated GPU memory, and up to about 8.5 GB as
`nvidia-smi` shows it on a bright coating (the allocator's reserve and a context
of about 0.25 GB); the cache is emptied before the lock is released.

**Agreement with the CPU backend.** The GPU updates overlapping beads in colour
classes where the CPU updates them one at a time in brightness order. With the
final calibration, on primary-type windows at most 0.5 % of the PolyT pixels of
a window differ from the CPU result by more than 10 counts and the removed light
agrees to within 0.6 %; on a dense, bright coating the difference is larger (a
few to about 8 % of the pixels above 10 counts), which is the size of the CPU
algorithm's own sensitivity to the update order. The GPU output never exceeds the
raw image, leaves the uncorrected stains untouched and is bit-for-bit
reproducible. Do not pool per-cell PolyT or DAPI intensities of GPU-corrected and
CPU-corrected samples without checking the QC. A cached zarr built on the other
device (or with other torch, Triton or CUDA versions) is reused with a warning
(see [Caching](#caching-and-how-to-rebuild)).

**Speed.** About 5-6 s per 4096 px window on a primary-size mosaic and 11-14 s on
a dense coating, so a 2.6 Gpx mosaic takes roughly 15-20 min including the
calibration (about 1.5 h for the densest datasets) against 1.4-2 h on 5-8 CPU
workers. The workstation profile still defaults to `cpu`; it switches to `cuda`
once the GPU backend has been validated there.

### Limits

- **Bright beads that fail the classifier can stay uncorrected.** The classifier
  accepts a bead only if its plane-0 ring ratio passes a per-dataset limit
  (about 0.30). On data whose beads are at a slightly different focus, bright
  beads fall below it. The [rescue](#the-bead-classifier-its-limits-and-the-rescue) accepts most of them, but not
  all: on one such validation dataset the bead-centred mask counts fell with the
  rescue from 58 and 45 to 29 and 16 (against the acceptance bars of 35 and 20),
  while the cluster-inclusive classifier recall stayed at 0.57 (0.88-0.96 on
  every other dataset). The step warns `classifier_recall` when the
  cluster-inclusive recall is below 0.75 with at least 50 reference beads, so
  that warning still fires on such a dataset. The isolated held-out recall is
  reported and warns only when at least 20 beads were measured and it is below
  0.90. Check the QC before using such a sample.
- **The bead-centred mask count is conditional.** In the validation cores the
  number of Cellpose masks on beads fell from 190 (raw) to 19 (median of three
  independent calibrations: 19, 23 and 19; acceptance bar 15; about 10 by
  chance). The masks that remain sit on the brightest beads (median plane-0
  amplitude about 5,500 counts), whose disc is not fully removed. The spread
  between calibrations comes from the PSF library; the full-mosaic production
  run gave 18. Do not expect fully bead-free segmentation near very bright
  beads.
- **Intensity bias near beads.** Pixels inside cells within about 40 px of
  bright beads are pushed slightly below the truth. Per-cell means move by about
  -0.1 % to -0.8 %; the PolyT 10th percentile of cells next to beads is about
  -4 % (measured -4.2 %). Segmentation recall is unaffected.
- **Partial removal under tissue.** Plane 3 and 4 rings under tissue keep
  35-45 % of their contrast (10-13 counts for a median bead, more for a bright
  aggregate). Most of this is hidden by the plane 1-2 maximum and the tissue.
- **Field-of-view centres** are corrected somewhat worse than field-of-view
  edges.
- **Dim beads** (plane-0 amplitude below 300 counts) are not modelled; their
  summed halo remains.
- **Border beads.** Beads within 24 px of the mosaic border or of internal
  zero-valued holes are not modelled and keep their rings.
- **Saturated beads** are not modelled specially; none occurred in the
  validation data, and the QC warns (`saturation`) when saturated reference
  pixels exist.
- **Dense coatings** make the result somewhat sensitive to the fitting order,
  and chunk results depend only weakly on the chunk grid; the Cellpose outcome
  was unaffected in validation.
- **A tissue-free area is required** (three calibration tiles at least 500 px
  from tissue). A slide without one is refused.
- **Plane 0 is never corrected or projected.** Anyone who used plane `0` for
  nuclei must use planes `1`-`6`.
- **No transcript-level validation.** The step changes images only. Changed
  Cellpose seeds near beads should remove bead "cells" from the ProSeg input,
  but this has not been tested.
- **Not applicable to Xenium and VZG2.**

### Downstream and what not to pool

Bead removal changes PolyT and DAPI pixel values, so per-cell intensity
features from a corrected sample are **not comparable** with those of an
uncorrected sample. Do not pool corrected and uncorrected samples (or samples
with the fill on and off) for PolyT or DAPI intensity features. Antibody-stain
features change only through mask shape.

When the source is bead-corrected, the mask image quantification table
(`uns["mask_image_quantification"]`) carries `bead_corrected_stains` (sorted),
`bead_removal_version` and `bead_removal_fill`, and its summary JSON gains a
`bead_removal` entry. A stored table that does not match the store's record is
re-quantified. Tables of uncorrected samples are unchanged.

## Common pitfalls

- **VZG2-only builds have no transcript points.** MerXen can recover the
  Vizualizer image, original LOD0 cell polygons, centroids, volumes, and the
  cell-by-gene/blank matrices directly from `.vzg2`. Vizgen does not publish a
  decoder for the packed transcript-coordinate tiles, and VZG2 also omits the
  canonical detected-transcript table. MerXen therefore does not invent
  coordinates. The build stage and cell-level original data are usable, but
  `reseg`, ProSeg, transcript-density QC, and other point-dependent stages need
  `detected_transcripts.csv` or `detected_transcripts.parquet` from the original
  MERSCOPE export. Use `--only_stage build` when the archive is the only source.
- **VZG2 plus canonical transcripts supports downstream stages.** If a region
  folder contains exactly one `.vzg2` and a sibling `detected_transcripts.csv`
  or `detected_transcripts.parquet`, but no mosaic TIFFs, MerXen combines the
  compressed VZG2 image with those external transcript coordinates. This
  layout supports Cellpose, ProSeg, enrichment, QC, and transcript analyses.
- **VZG2 image z-layer selection.** `merscope_z_range` must include the layer
  listed in the archive manifest. Most VZG2 files contain one mid-stack plane.
  The default `0-6` includes the common layer `3`.
- **MERSCOPE plane 0 included in the projection.** Plane `0` contains fiducial
  beads rather than the biological image stack. Including it in
  `merscope_z_range` can dominate or distort the DAPI/PolyT max projection and
  every image-derived downstream result. Set an explicit range beginning at
  `1`, such as `1-7`. With
  [bead removal](#merscope-bead-removal-optional) on, the range must start at
  `1` or later. This does not apply to Xenium, whose morphology image is
  already projected.
- **MERSCOPE transform missing.** Canonical raw exports need
  `micron_to_mosaic_pixel_transform.csv` either inside the region folder or
  at `merscope_transform_path`. Direct VZG2 ingestion can derive the transform
  from the manifest's mosaic dimensions and micron bounding box; an explicit
  `merscope_transform_path` still takes precedence.
- **VZG2 transform origin.** Mosaic pixel 0 is the manifest's `bbox_microns`
  minimum, which can be negative (the Vizgen MsBrain VZG2 region starts at
  (-21.86, -111.78) µm). A transform without that translation shifts every
  Cellpose / ProSeg cell by minus the minimum; the 2026-09-08 VZG2 run was
  built from such a stale override and its `reseg`, `proseg_mask` and
  `proseg_hybrid` outputs are offset by (+21.9, +111.8) µm (`original_seg` is
  unaffected). The build now fails when `merscope_transform_path` disagrees
  with the manifest by more than 1 µm; for an archive whose `bbox_microns` is
  known to be wrong, `--merscope_allow_transform_override_mismatch true`
  keeps the override with a warning. The build records
  `attrs["merxen_vzg2"]["transform_source"]` (`override` or `manifest`).
  Segmentation refuses a `merscope_transform_path` that differs from the
  `micron_to_mosaic_pixel_transform.csv` stored in the built zarr, so a
  corrected CSV requires a rebuild (`--force_spatialdata_build true`), and it
  refuses to reuse a `transcripts_for_proseg.csv` seeded with another
  transform ([segmentation.md](segmentation.md)). The QC registration check
  ([qc.md](qc.md)) flags any remaining offset.
- **Xenium spec ambiguity.** If the Xenium export contains a non-standard
  `specs.json` location, set `xenium_spec_path` in the samplesheet.
- **Partial builds.** If a build crashes half-way, delete the target zarr
  before rerunning — SpatialData does not currently distinguish a
  half-written zarr from a valid one.

## VZG2 example

Pass either the archive itself or a folder containing exactly one `.vzg2` in
the existing `merscope_dir` column:

```csv
pair_id,analysis_mode,only_stage,merscope_dir,merscope_z_range
MOUSE_VZG2,merscope,build,/srv/storage/mouse_vizgen_reference/sample.vzg2,0-6
```

If a folder also contains a canonical raw `images/` directory, MerXen prefers
the canonical raw export. If a folder contains multiple `.vzg2` files, pass the
intended archive path explicitly.

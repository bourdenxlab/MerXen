# Image registration

Registers externally acquired images of the **same Xenium section** onto the
Xenium mask grid, so their channels are quantified over the same Cellpose
masks as the instrument channels. The typical input is immunofluorescence (IF)
imaged after the Xenium run. Several images per section are supported, each
with its own alignment matrix.

The stage is opt-in per samplesheet row and runs only on the Xenium side of a
row: `ENRICH → REGISTER_IMAGES → VIEWER_CACHE → MASK_IMAGE_QUANTIFICATION`.

## What it does

For each listed image:

1. **Read** the OME-TIFF lazily. Flat and pyramidal files are both accepted; only
   level 0 is resampled, and reduced levels, when present, speed up the
   registration step. Channel names and the pixel size come from the OME-XML.
2. **Load the alignment matrix** — the 3×3 affine exported by Xenium Explorer
   after aligning the image. It maps image level-0 pixels to Xenium
   `morphology_focus` level-0 pixels (the convention `spatialdata-io` uses for
   Xenium aligned images). Non-affine, singular, or malformed matrices are
   rejected.
3. **Check the matrix scale.** An image-pixel → Xenium-pixel affine must scale
   lengths by `image pixel size / 0.2125 µm`. A mismatch beyond 2 % fails the
   stage and names the convention the matrix looks like instead (inverted, or
   expressed in microns). Images without an OME pixel size skip this check.
4. **Refine the affine** (default on). The registration channel (`DAPI`) of the
   image is overlaid on the Xenium DAPI at ~0.85 µm/px using the Explorer
   matrix. Local shifts are measured by phase correlation in ~200 µm windows
   over tissue, and a robust (outlier-trimmed) affine is fitted to them and
   composed with the Explorer matrix. The refinement is applied only within
   safety limits (≤ 50 µm displacement, ≤ 2° rotation, ≤ 2 % scale change)
   and only with at least 20 usable windows; otherwise the Explorer matrix is
   kept and the reason is recorded.
5. **Resample** the image tile by tile (bilinear) onto the `morphology_focus`
   level-0 grid, which is the Cellpose mask grid, and write it as a new image
   element named by `image_key`. Pixels outside the image footprint are 0. A
   viewer pyramid is built alongside it.
6. **Record provenance** in `sdata.attrs["merxen_registered_images"]`. An image
   whose file, matrix, channel names, and settings are unchanged is skipped on
   rerun, also from a fresh Nextflow work directory: its stored summary is
   written out again (marked `"reused": true`) and its QC files are copied when
   the earlier work directory still holds them. Images registered earlier but
   no longer listed are removed, together with their viewer pyramid.

`MASK_IMAGE_QUANTIFICATION` then quantifies every source image, including the
registered ones, and joins the Cellpose-mask measurements onto
`table_MOSAIK_proseg_hybrid` by `instance_id`, exactly as for MERSCOPE image
channels. Features are named `{image_key}__{channel}__{statistic}`, e.g.
`post_xenium_if__p62__mean`. When the set of images changes, an existing
quantification table is recomputed rather than reused.

## Inputs

Set `xenium_registered_images_csv` on a samplesheet row to a CSV with one row
per image:

| Column | Required | Description |
|--------|----------|-------------|
| `image_key` | yes | SpatialData element name and feature prefix. Letters, digits, `_`, `-`, `.`; must not start with `_`; unique per section; cannot be `morphology_focus`. |
| `image_path` | yes | OME-TIFF (flat or pyramidal), 2D multichannel. Z-stacks must be projected first. |
| `alignment_matrix` | yes | 3×3 matrix CSV exported by Xenium Explorer for this image. |
| `channel_names` | no | `;`-separated names replacing the file's channel names, e.g. `DAPI;p62;AT8`. Use this to turn fluorophore names into marker names. |
| `registration_channel` | no | Channel used for refinement and QC, matched case-insensitively against the Xenium `DAPI`. Defaults to `DAPI`; `none` disables refinement and QC for that image. |

Relative paths resolve against the CSV's own directory.

```csv
image_key,image_path,alignment_matrix,channel_names,registration_channel
post_xenium_if,Slide511_s2.ome.tiff,Slide511_s2_matrix.csv,DAPI;AF568;AF647,DAPI
post_xenium_if_r2,Slide511_s2_round2.ome.tiff,Slide511_s2_round2_matrix.csv,DAPI;GFAP;IBA1,
```

Use the **lossless original** image for registration. The 10x conversion script
commonly used to make Xenium Explorer-compatible pyramids writes JPEG 2000 with
`compressionargs={'level': 85}`. In `imagecodecs` that `level` is a target PSNR
in dB against the 16-bit full scale, so the pyramid carries about 3.5 counts
RMS error per pixel (on P7513 only ~11 % of level-0 pixels were bit-identical).
The original and its pyramid share the level-0 grid, so the matrix exported for
the pyramid applies to the original unchanged. If a lossless pyramid is needed,
`compressionargs={'reversible': True}` is lossless.

## Nextflow process

[`REGISTER_IMAGES`](../../workflows/modules/register_images.nf) — one instance
per Xenium dataset whose row sets `xenium_registered_images_csv`.

- **Input:** enriched latest zarr.
- **CLI:** `merxen register-images --config register_images_config.json`.
- **Output:** `tuple(key, pair_id, platform, latest_input.zarr, register_images_out/)`.
- **publishDir:** `${outdir}/${pair_id}/xenium/register_images/` (symlink mode).

Preflight checks fail the launch when the CSV, an image, or a matrix is
missing, a required column is empty, or an `image_key` repeats. The stage is
selectable as `register_images` in `start_stage`/`stop_stage`/`only_stage`; it
exists only for rows that set the CSV.

## Outputs

| File | Contents |
|------|----------|
| `latest/latest_spatialdata.zarr` | Updated in place with one image element per `image_key` (plus its viewer pyramid) and the provenance registry in `attrs`. |
| `register_images_out/<dataset>_registered_images.json` | All images of the section, plus any keys removed as stale. |
| `register_images_out/<dataset>_<image_key>_registration_summary.json` | Matrices (Explorer and final), scale check, refinement status and correction, residuals before/after, output shape. |
| `register_images_out/<dataset>_<image_key>_registration_qc.png` (+ `.pdf`) | DAPI overlay (Xenium green, image magenta), residual-shift field after registration, and before/after residual histograms. |
| `register_images_out/<dataset>_<image_key>_registration_local_shifts.csv` | Per-window residual shifts (µm) and correlation before and after refinement. |

### Reading the QC

`residuals_before` and `residuals_after` summarise the per-window shifts still
needed to superimpose the two DAPI images: median, p90, p99 and max in µm, plus
the log-intensity Pearson correlation over shared tissue. Windows are kept only
when the shift-corrected correlation is ≥ 0.3. The shift estimator is biased by
about 0.1 working pixel (~0.1 µm) towards zero, so sub-0.2 µm residuals are at
the measurement floor.

Residuals that remain after refinement are local, not global. A smooth field
would point to tissue deformation; rectangular blocks with sharp edges point
to tile-stitching differences between the two acquisitions. Neither can be
removed by an affine.

### Worked example: P7513 post-Xenium IF

Zeiss orthogonal projection, 3 channels (DAPI, AF568, AF647), 0.3434 µm/px,
33,453 × 31,347 px; Xenium `morphology_focus` 47,618 × 51,313 px at 0.2125 µm/px.
The Explorer matrix is a −89.4° rotation with scale 1.6154, against 1.6159
expected from the pixel sizes (0.03 % off).

| | median | p90 | p99 | max |
|---|---|---|---|---|
| Explorer matrix | 2.11 µm | 5.78 µm | 10.67 µm | 17.0 µm |
| after affine refinement | 0.26 µm | 3.06 µm | 6.86 µm | 15.9 µm |

Over 5,096 windows; the refinement was a 0.04° rotation, +0.04 % scale and at
most 6.5 µm displacement, and raised the log-DAPI correlation from 0.65 to 0.75.
Most of the tissue is aligned to below half a micron; the remaining tail sits
in rectangular blocks at the left edge of the section, consistent with
stitching differences, and at the tissue margins.

The tail matters for quantification. Per-cell mean DAPI over the real Cellpose
masks, Xenium versus registered IF (log scale), correlates at r = 0.89 in a
well-aligned 870 µm crop (3,555 cells, residual 0.15 µm) but only r = 0.34 in a
crop of the left-edge tail (1,511 cells, residual 7.1 µm), where IF nuclei sit
about one nucleus diameter away from their masks. A non-rigid second pass is
not implemented; cells in tail regions should be treated with caution.

**Capture range.** Perturbing the Explorer matrix about the tissue centre and
refining again (limits relaxed) recovered the unperturbed result to within
0.2 µm for shifts up to 80 µm, 0.5 µm for a 0.5° rotation, 0.4 µm for a 1 %
scale error, and about 1.4 µm for 1° or 2 % (errors of 136–158 µm at the image
corners). The 50 µm limit therefore rejects fits well inside the measurable
range; corrections that large usually mean the matrix belongs to another
image.

The run took 14 min wall time (refinement ~4 min, resampling and viewer
pyramid ~10 min, without a CPU cap) and peaked at 9.7 GB RSS. The registered
image occupies 4.9 GB in the zarr, plus 0.3 GB for its viewer pyramid.

## Config schema

`ImageRegistrationConfig` and `RegisteredImageSpec` —
[config.py](../../src/merxen/config.py).

| Field | Default | Description |
|-------|---------|-------------|
| `images` | — | One `RegisteredImageSpec` per image (`image_key`, `image_path`, `alignment_matrix_path`, `channel_names`, `registration_channel`). |
| `reference_image_key` | `morphology_focus` | Image whose level-0 grid receives the registered images. |
| `reference_channel` | `DAPI` | Reference channel for refinement and QC. |
| `reference_pixel_size_um` | `0.2125` | Xenium morphology pixel size ([10x Genomics](https://kb.10xgenomics.com/hc/en-us/articles/35386990499853-How-can-I-convert-coordinates-between-H-E-image-and-Xenium-data)). |
| `matrix_scale_tolerance` | `0.02` | Accepted relative deviation of the matrix scale. |
| `refine_affine` | `true` | Fit and apply the DAPI affine correction. Nextflow: `--register_images_refine_affine`. |
| `registration_pixel_size_um` | `0.85` | Working resolution for refinement and QC. |
| `refinement_window_um` | `200` | Window size for local shift measurement. |
| `min_refinement_windows` | `20` | Minimum usable windows to attempt refinement. |
| `max_refinement_shift_um` / `max_refinement_rotation_deg` / `max_refinement_scale_change` | `50` / `2` / `0.02` | Safety limits on the correction. |
| `tile_size` / `chunk_size` | `4096` / `4096` | Resampling tile and output chunk size (equal sizes avoid partial-chunk rewrites). |
| `build_viewer_pyramid` | `true` | Build the viewer pyramid for each registered image. Nextflow follows `--viewer_cache_build_image_pyramid`. |

## Python entry points

| Function | File |
|----------|------|
| CLI `register_images_command` | [cli/run_image_registration.py](../../src/merxen/cli/run_image_registration.py) |
| `run_image_registration` | [image_registration.py](../../src/merxen/image_registration.py) |
| `load_alignment_matrix`, `check_matrix_scale` | [image_registration.py](../../src/merxen/image_registration.py) |
| `measure_local_shifts`, `fit_affine_displacement` | [image_registration.py](../../src/merxen/image_registration.py) |
| `read_ome_tiff_info`, `open_ome_tiff_level` | [io/ome_tiff.py](../../src/merxen/io/ome_tiff.py) |

# Manual landmark registration

Registers two images of **adjacent or identical tissue sections** by hand-placed
landmarks, then refines the result automatically. It is meant for pairs the
automatic pipelines cannot align, for example a Zeiss slide scan of a serial
section against a Xenium DAPI image, or a post-Xenium IF image whose Explorer
matrix is unavailable. The registered moving image shares the reference pixel
grid, so its channels can be read against the reference directly.

This is **not a stage of `main.nf`**. It is two standalone Nextflow workflows
(prepare and register) with a human step between them, run from the same
samplesheet:

```
 samplesheet row: id (alias), reference image, moving image, optional per-image settings, presets
        |
 [1] nextflow run workflows/manual_registration_prepare.nf
        MANUAL_REGISTRATION_PREPARE   CZI -> pyramidal OME-TIFF (or validate/normalise an OME-TIFF)
                                      -> image-only SpatialData "prepared store" (atomic, idempotent)
        VIEWER_CACHE (image-only mode) -> ds4 pyramids for both images
        |   <outdir>/<id>/prepared/<id>_prepared.zarr
 [2] viewer: Dataset loader > Manual Registration > "Load image data"
        crops (optional) -> coarse rotate/flip/shift of the moving image -> landmark pairs
        (F flicks, auto-switch, live residuals, preview fit) -> Save draft / Finish labelling (>= 8 pairs)
        |   writes ONE root attribute of the prepared store
 [3] nextflow run workflows/manual_registration_register.nf
        MANUAL_REGISTRATION_REGISTER  per (row x preset): crop -> robust similarity fit from landmarks ->
                                      guarded DAPI affine refinement -> guarded TV-L1 flow -> warp all channels
        MANUAL_REGISTRATION_COMPARE   per id with >= 2 runs: side-by-side metrics
        |   <outdir>/<id>/registered/<run_name>/<run_id>_registered.zarr  + CSVs + QC
 [4] viewer: same button on a result store -> reference + warped moving, landmark overlays, flick to check
```

The two workflows live in `workflows/` so they inherit `nextflow.config`, the
execution profiles, the site configs and `workflows/lib/`. `main.nf`, its stage
order and the metro map do not know about them.

## What it does

**Two presets.** The register workflow runs every row once per preset. The
presets differ in the length scale they trust:

- `same_section`: both images show the same cells (a post-Xenium IF image and
  the Xenium DAPI of the same section). Refinement and the optical flow work at
  nucleus scale, exactly the `REGISTER_IMAGES` settings with the flow switched
  on.
- `serial_section`: adjacent sections have no 1:1 nuclei. The preset aligns
  tissue-scale structure (layers, vessels, white-matter boundaries) at 6 to
  12 µm per pixel with wider windows, stronger smoothing and tighter Jacobian
  limits; nucleus-level QC is computed but informative only.

For a serial pair, run both: `same_section` then shows what nucleus-scale
refinement and flow do on serial sections (its guards may reject them; status
and reason are reported), and the comparison table puts the runs side by side.

**Register, per (row x preset)** (`merxen manual-registration-register`):

1. Validate the prepared store, resolve the settings (preset, then params, then
   row; a `crop_snap_px` below 64 is refused), and skip the run when an identical
   result already exists (see "Re-runs" under Outputs).
2. Crop both images to the effective crops (snapped inward to 64 px).
3. Fit a robust similarity (rotation, scale, optional reflection) to the enabled
   landmark pairs; flag outliers; report the scale against the pixel-size ratio.
4. Refine with the DAPI affine of the image registration core
   (`refinement_*` settings), then optionally estimate the TV-L1 flow
   (`nonrigid_*`); both are guarded and fall back with a recorded reason.
5. Warp every moving channel onto the cropped reference grid, place every
   landmark at each stage (landmark, affine, final) and compute tissue-scale
   Pearson correlation, residuals and optional nucleus-scale QC.
6. Write the result store, the matrices, the landmark table, the summary and the
   QC figure.

## Inputs and samplesheet

One samplesheet for both workflows, one row per pair
(`workflows/samplesheet.manual_registration.example.csv`):

```csv
id,reference_image,moving_image,presets,registration_settings
LCM1,/path/to/slide_reference.czi,/path/to/slide_moving.czi,same_section;serial_section,same_section:matrix_scale_tolerance=0.05
HD1,/path/to/morphology_focus_dapi.ome.tif,/path/to/post_xenium_if.ome.tif,same_section,
```

| Column | Required | Default | Validation |
|---|---|---|---|
| `id` | yes | | `^[A-Za-z0-9][A-Za-z0-9_.-]*$`, unique; an alias by convention |
| `reference_image`, `moving_image` | yes | | must exist; relative paths resolve against the sheet's folder; extension `.czi`, `.ome.tif`, `.ome.tiff`, `.tif` or `.tiff`; the two differ (also through a symlink) |
| `reference_scene`, `moving_scene` | no | | only for `.czi` (error on TIFF); S index or scene name (case-insensitive); resolved in Python |
| `reference_channel_names`, `moving_channel_names` | no | auto | `;`-list, non-empty, unique; count checked in Python; used verbatim |
| `reference_registration_channel`, `moving_registration_channel` | no | `DAPI` | non-blank; resolved against the channel names in Python |
| `reference_pixel_size_um`, `moving_pixel_size_um` | no | metadata | number > 0 |
| `presets` | no | `--manual_registration_presets` (`same_section`) | `;`-list of `same_section`, `serial_section`, no duplicates |
| `registration_settings` | no | | `[preset:]key=value;...` (grammar below); keys are checked before any task starts |
| `run_label` | no | | `^[A-Za-z0-9][A-Za-z0-9_.-]*$`; appended to run names |
| `prepared_store` | no | `<outdir>/<id>/prepared/<id>_prepared.zarr` | register workflow only; must exist |

Blank cells mean absent. Unknown columns are an error. A cell may be quoted
(RFC 4180); a BOM and CRLF line ends are accepted. All errors of a sheet, and of
the `--manual_registration_*` params, are collected into **one**
`Preflight checks failed:` block before any task is created. Both workflows
validate every column. The register workflow still requires the source images
to exist, although it reads only the prepared OME-TIFFs. `prepared_store` in a
prepare run is ignored with a warning (prepare always writes the default
location).

**CZI files.** A `.czi` is read by an **isolated runtime**: `czifile` cannot join
the base environment (squidpy caps `imagecodecs<2026`), so MerXen re-runs
itself under the interpreter named by `manual_registration_czi_python`
(Nextflow; `MERXEN_CZI_PYTHON` for the CLI and tests). Scenes: a multi-scene
file needs `<role>_scene`; the converted OME-TIFF grid is the scene bounding
box, whose origin may be negative (it is recorded as `czi.bbox_xywh`). Channel
names come from the CZI `Name`, then `Fluor`, `DyeName`, `ShortName`, else
`c<i>`; a single nuclear-class channel is renamed `DAPI`, and
`<role>_channel_names` overrides all names verbatim. RGB images are refused.
Z-stacks are projected (the count is recorded). The converter is deterministic: a
forced re-conversion of unchanged inputs keeps the OME-TIFF fingerprint. CZI
rows are **unsupported under the docker/apptainer profiles** in this version:
the container has no CZI runtime and a host `MERXEN_CZI_PYTHON` path does not
exist inside it. Use `-profile conda`, or convert first with
[`merxen convert-czi`](../cli.md#merxen-convert-czi) and pass the OME-TIFF.

**Settings grammar.** `items := item (";" item)*`,
`item := [preset ":"] key "=" value`; the key matches `[a-z][a-z0-9_]*` and is
one of the fields of [`ManualRegistrationSettings`](#config-schema) except the
two forbidden ones (`reference_pixel_size_um`, `reference_channel`: set pixel
sizes and registration channels in the samplesheet and re-run prepare). A single
trailing `;` is allowed; an empty item anywhere else (`a=1;;b=2`) and a key given
twice in one scope are preflight errors. (Python's settings parser, used by
`merxen manual-registration-fit --settings`, skips every empty item, so `a=1;;b=2`
is accepted there and refused by the workflows; a duplicate key is refused by
both.) Precedence, lowest to highest: the preset, the global
`--manual_registration_settings` without prefix, the global `<preset>:` items,
the row's items without prefix, the row's `<preset>:` items. The Groovy
preflight checks the grammar, the keys, and that `landmark_min_pairs` and
`landmark_min_used_pairs` are integers >= 3; other value types are checked by
pydantic in the task.

> **Serial-section pairs run with `same_section`** (to compare against
> `serial_section`) need
> `registration_settings=same_section:matrix_scale_tolerance=0.05`, or the same
> through `--manual_registration_settings`. Adjacent sections commonly differ in
> scale by 1 to 3 %, and the `same_section` default of 0.02 stops the run at the
> landmark fit with "landmark fit scale ... differs from the pixel-size ratio".
> The example sheet's first row shows it.

## Workflows and launch

```bash
# from the usual launch directory, with the merxen environment active
nextflow run workflows/manual_registration_prepare.nf -profile dwight,conda \
    --samplesheet workflows/samplesheet.manual_registration.csv \
    --outdir "$MERXEN_RESULTS_ROOT/manual_registration"
# label: viewer > Dataset loader > Manual Registration > Load image data >
#        $MERXEN_RESULTS_ROOT/manual_registration/<id>/prepared/<id>_prepared.zarr
# optional quick check of the landmarks (seconds):
merxen manual-registration-fit --store "$MERXEN_RESULTS_ROOT/manual_registration/<id>/prepared/<id>_prepared.zarr" --preset serial_section
nextflow run workflows/manual_registration_register.nf -profile dwight,conda \
    --samplesheet workflows/samplesheet.manual_registration.csv \
    --outdir "$MERXEN_RESULTS_ROOT/manual_registration" \
    [--manual_registration_settings "serial_section:nonrigid_smoothing_sigma_um=150"]
# check: Load image data > .../<id>/registered/<preset>/<id>_<preset>_registered.zarr
```

Use an `--outdir` **separate from the main pipeline's**: the Nextflow report,
timeline and trace files under `<outdir>/nextflow/` are overwritten per launch.
Both workflows refuse `--outdir` equal to the main pipeline's default `./results`
(also written `results` or `./results/`) with "pass --outdir with a folder
separate from the main pipeline's results", before any task starts. The report,
timeline and trace observers are configured in `nextflow.config` and run before
the script does, so a launch that forgets `--outdir` still rewrites
`./results/nextflow/{report.html,timeline.html,trace.tsv}`; no task runs and no
per-id folder is written.
`-resume` is optional (all three processes are `cache false`; Python keeps its
own provenance, skips unchanged work in seconds, and the viewer edits the
prepared store in place, which Nextflow's input hashing cannot see). Keep one
launch directory per workflow, or use `-resume <session>`.

**Re-running prepare.** An unchanged pair keeps its store, labelling included. When
the images are rebuilt on the same grid (same shapes and pixel sizes) the labelling
is carried over. A changed grid under an existing labelling is refused, because the
landmarks would no longer fit; `--manual_registration_force_prepare true` then
builds the new store and keeps the old one next to it as
`<id>_prepared.zarr.superseded-<UTC>`. The labelling is carried over into the new
store **only when both images keep their shape and pixel size**; otherwise the
active store has no landmarks, they stay in the `.superseded-<UTC>` store (keep it,
or copy it elsewhere, before forcing) and the new store must be labelled again.

**The CZI runtime.** Create it once from `envs/environment.czi.yml` (see
[Development workflow](../development.md#dependency-management)), then set
`manual_registration_czi_python` in the gitignored `workflows/conf/site.config`
(template `site.config.example`) to its python, or to `in_process` when
`czifile` is installed in the task environment itself. The prepare workflow
exports it to the task as `MERXEN_CZI_PYTHON`. A `.czi` row without it is a
preflight error.

| Param | Default | Meaning |
|---|---|---|
| `manual_registration_presets` | `"same_section"` | Presets of rows with a blank `presets` column |
| `manual_registration_settings` | `""` | Settings overrides for every row (the row's column wins) |
| `manual_registration_force_prepare` | `false` | Rebuild the prepared store; a changed grid with a labelling needs it (see "Re-running prepare") |
| `manual_registration_force_register` | `false` | Re-run registrations Python would skip |
| `manual_registration_czi_python` | `null` | Python of the CZI runtime, or `in_process` (site config) |

Resources: `MANUAL_REGISTRATION_PREPARE` 8 CPUs / 16 GB,
`MANUAL_REGISTRATION_REGISTER` 8 CPUs / 48 GB (a full 80k x 61k px Zeiss scan;
refinement working images are capped at 32767 px per side),
`MANUAL_REGISTRATION_COMPARE` 1 CPU / 4 GB. The Dwight profile limits the first
two to two concurrent tasks each (`manual_registration_prepare_max_forks`,
`manual_registration_register_max_forks`).

**The preflight.** The prepare workflow checks the sheet and the params. The
register workflow additionally reads the root attributes of every prepared store
(zarr v2 or v3, without Zarr) and stops, before any task, on:

| Message starts | What to do |
|---|---|
| `<id>: prepared store not found` | run the prepare workflow with the same samplesheet and `--outdir` |
| `<id>: cannot read the root attributes` | the store is damaged (or not a Zarr group): re-run prepare with `--manual_registration_force_prepare true`; the old store is kept as `<id>_prepared.zarr.superseded-<UTC>` and its landmarks are not carried over (see "Re-running prepare") |
| `<id>: ... belongs to dataset <name>` | `prepared_store` points at another id's store: set it to this id's `<id>_prepared.zarr` |
| `... was written by a newer MerXen/viewer` | update MerXen (or the viewer) |
| `... is a result store, not a prepared store` | set `prepared_store` to the `<id>_prepared.zarr` store |
| `no landmarks saved in ...` | open the store in the viewer, place the pairs, press Finish labelling |
| `the landmarks in ... are a draft` | press Finish labelling |
| `<id> (preset <p>): <n> enabled landmark pairs ..., at least <m> are needed` | enable or place more pairs, or lower `landmark_min_pairs` and `landmark_min_used_pairs` together (both >= 3; `landmark_min_used_pairs` is 6 by default and must not exceed `landmark_min_pairs`) |
| `<id> (preset <p>): landmark_min_pairs (<n>) is below landmark_min_used_pairs (<m>)` | lower `landmark_min_used_pairs` to at most `landmark_min_pairs`, or raise `landmark_min_pairs` (a row's or the params' override of either one counts) |
| `run id '<run_id>' is used twice` | set `run_label` to tell the runs apart |

A register task that fails drops only its own branch (`errorStrategy ignore`);
the run still exits non-zero (`failOnIgnore`), and an id left with a single
successful run gets no comparison.

## Viewer workflow

Documented in detail by the viewer's own guide (`docs/manual_registration_guide.md`
of the napari viewer repository); the outline:

1. **Load** the prepared store: Dataset loader > Manual Registration > Load image
   data.
2. **Crops** (optional, per image, in the original pixel frame). They are snapped
   inward to multiples of 64 px (`crop_snap_px`, which must be at least 64: a
   smaller value is refused at the start of the run, because the pyramid levels
   it reads are cropped at factors that divide 64), and an effective side must
   be at least 1024 px.
3. **Coarse alignment** of the moving image: rotate (large angles, a 180 degree
   button), flip, shift. The state is stored as rotation, mirror and offset.
4. **Landmark pairs**: place a point in the reference and its partner in the
   moving image. `F` flicks between the images, the view switches automatically
   after each point, and a residual table shows the live fit. The outlier floor
   has two buttons, "serial (30 µm)" and "same (10 µm)", matching the presets;
   the viewer's floor is saved informatively and register prints it next to the
   floor it used.
5. **Save draft / Finish labelling** (at least 8 enabled pairs). A disabled pair
   is held out of the fit and reported as a held-out residual.
6. **Reopen a result store** with the same button to see the reference and the
   warped moving image, the landmark overlays at each stage, and flick to check.

"Preview fit" in the viewer equals the workflow fit only at the preset's outlier
floor: `merxen manual-registration-fit --store ... --preset serial_section` is
the authoritative check.

## Outputs

Everything below is durable and written by Python to the absolute paths computed
in Groovy (`<outdir>` is the launch's `--outdir`; the processes publish nothing
through Nextflow):

```
<id>/prepared/<id>_prepared.zarr                         # the store the viewer edits; written atomically
<id>/prepared/<id>_reference.ome.tif                     # only when converted (CZI) or normalised
<id>/prepared/<id>_moving.ome.tif                        # same
<id>/prepared/<id>_reference.prepare.json                # per-image provenance (conversion skip key)
<id>/prepared/<id>_moving.prepare.json
<id>/prepared/<id>_prepare_summary.json
<id>/prepared/viewer_cache/                              # VIEWER_CACHE publishDir (symlink to the work dir)
<id>/registered/<run_name>/<run_id>_registered.zarr      # run_name = <preset>[_<run_label>]
<id>/registered/<run_name>/<run_id>_landmark_matrix.csv
<id>/registered/<run_name>/<run_id>_final_matrix.csv
<id>/registered/<run_name>/<run_id>_landmarks.csv
<id>/registered/<run_name>/<run_id>_manual_registration_summary.json
<id>/registered/<run_name>/<run_id>_tissue_qc.png (+ .pdf)
<id>/registered/<run_name>/qc/<run_id lower>_moving_registered_{registration_qc.png,registration_qc.pdf,registration_local_shifts.csv,nonrigid_flow.npz,nonrigid_qc.png,nonrigid_qc.pdf}   # only the files the run wrote, see "Core QC files"
<id>/registered/<id>_run_comparison.{json,csv,png}
<outdir>/nextflow/{report,timeline,trace}*
```

`run_id = <id>_<run_name>`: the preset is in both the folder and the file names,
so runs never collide. `VIEWER_CACHE` publishes `viewer_cache_out/` and its
`latest_input.zarr` symlink into `<id>/prepared/viewer_cache/`; the symlink is
harmless.

**Core QC files.** The registration core writes the `qc/` files, and a run lists
in the summary (`outputs.core_qc`) only the ones it actually wrote:

| File (suffix after `<run_id lower>_moving_registered_`) | Written when |
|---|---|
| `registration_qc.png` / `.pdf`, `registration_local_shifts.csv` | the refinement measured its windows |
| `nonrigid_flow.npz`, `nonrigid_qc.png` / `.pdf` | a flow was computed (applied or rejected); not with `nonrigid_enabled=false` |

A run deletes this run id's core QC files from earlier runs before the core
starts, so a file under `qc/` belongs to the latest attempt, never to an earlier
run with other settings or crops. When that attempt fails in the core, the files
the previous result listed are gone, and the next launch registers again.

**Re-runs.** An unchanged re-run is skipped in seconds, but only when the
result is provably complete: the result store and the durable summary
(published last) both carry the same `provenance_digest` as the new run, the
summary says `complete`, every durable output and every file `outputs.core_qc`
lists exists, and none of those core QC files is newer than the durable summary
(a newer one belongs to a later run with other inputs that failed after the
core). Anything else registers again; `--manual_registration_force_register true`
(CLI `--force-rerun`) registers again regardless. A result store swap keeps the
previous store as `<run_id>_registered.zarr.superseded-<UTC>` only for the
instant of the swap; it is deleted after a successful swap, and any such
leftover from a run that died between the two renames is removed by the next
successful run.

**Matrix CSVs** (`<run_id>_landmark_matrix.csv` = robust similarity,
`<run_id>_final_matrix.csv` = refined affine **without** the flow): 3 lines of 3
comma-separated values, no header, `%.17g`, written by `write_alignment_matrix`
and re-read with `load_alignment_matrix`. Both are in the **full, uncropped**
level-0 pixel frames of both images and map moving to reference.

**Landmarks CSV** (`<run_id>_landmarks.csv`), one row per pair:
`id, colour, enabled, used_in_fit, outlier, outside_crop, reference_x_px,
reference_y_px, moving_x_px, moving_y_px, landmark_x_px, landmark_y_px,
affine_x_px, affine_y_px, final_x_px, final_y_px, residual_landmark_um,
residual_loo_um, residual_affine_um, residual_final_um, final_includes_flow,
push_converged`. The stage positions are in full reference px; floats `%.6g`.

**Comparison CSV** (`<id>_run_comparison.csv`), one row per run:
`run_id, preset, run_label, fit_n_enabled, fit_n_used, fit_rms_um,
fit_scale_ratio, fit_rotation_deg, refinement_status, refinement_max_shift_um,
refinement_rotation_deg, refinement_scale_change, nonrigid_status,
nonrigid_confident_fraction, nonrigid_pearson_before, nonrigid_pearson_after,
jacobian_min, jacobian_max, tissue_pearson_coarse, tissue_pearson_landmark,
tissue_pearson_affine, tissue_pearson_final, used_median_landmark_um,
used_median_affine_um, used_median_final_um, held_out_median_landmark_um,
held_out_median_affine_um, held_out_median_final_um, nucleus_qc_median_um,
nucleus_qc_pearson, wall_time_s, peak_rss_gb`.

**Result store placement.** The result store's images are on the **cropped
reference level-0 grid**; their `global` transformation is an `Affine` with the
reference pixel size and the crop origin, so `global` is micrometres of the full
reference frame and landmarks stored in full reference pixels line up. The store
is written to a temporary sibling and renamed into place only when complete.

**Feeding the main pipeline.** The final matrix can serve as the
`alignment_matrix` of an `xenium_registered_images_csv` row (see
[Samplesheet format](../samplesheet.md#registered-image-csv)): `image_path` is the
moving OME-TIFF recorded in `merxen_manual_registration_store.images.moving.ome_tiff.path`,
`alignment_matrix` is `<run_id>_final_matrix.csv`. This is valid **only when the
prepared reference image is the Xenium `morphology_focus` level-0 grid**, because
the main pipeline's matrices map into that grid. The flow is not part of the
matrix; the main pipeline applies its own refinement.

## Reading the QC

- **Fit block** (`fit` in the summary): scale ratio against the pixel-size ratio,
  the reflection check (the chosen class's median residual against the other
  class's), the spread ratio of the used pairs, the number of outliers, and the
  outside-crop policy (`landmarks_outside_crop`: `use`, `exclude` or `fail`).
  Outliers are marked, excluded from the fit and still reported.
- **Refinement and non-rigid status**: `applied`, `disabled`, `skipped_<reason>`
  or `rejected_<reason>`, with a reason. A rejected guard keeps the previous mapping; the
  run is still `complete`. The flow reports confident fraction, Pearson before and
  after, displacement and Jacobian statistics.
- **Tissue Pearson per stage** (`tissue_qc.pearson`: coarse, landmark, affine,
  final): correlation of band-passed tissue-scale intensity. The working image
  is the coarsest pyramid level that is not coarser than `tissue_qc_pixel_size_um`
  (10 µm by default), block-averaged when that level is at least twice as fine,
  so its size is at most the setting and often finer (5.5 µm for a 0.345 µm
  scan); the size actually used is `tissue_qc.pixel_size_um`. It should not fall
  from one stage to the next.
- **Held-out residuals**: disabled pairs are never used in the fit; their median
  residual at each stage (`residual_summary_um.held_out`) is the honest measure
  of the registration error, in reference micrometres.
- **Nucleus QC** (`nucleus_qc`): informative only, at 1.7 µm per pixel in
  200 µm windows after the final affine; on for `serial_section`, where it only
  tells how far nuclei agree.
- **Comparison**: `<id>_run_comparison.csv/png` put the presets (and labelled
  re-runs) side by side.

## Coordinate conventions

- A point is `(x, y)` in **level-0 pixels of the full, uncropped image** it
  belongs to; an integer value is a pixel centre. Lengths are µm unless the name
  ends in `_px`; residuals are reference µm.
- Every matrix maps **moving to reference**, 3 x 3 with last row `[0, 0, 1]`
  (the Xenium Explorer convention).
- Positive rotation is counter-clockwise on screen (y points down); angles are
  wrapped to (-180, 180].
- The coarse state is rotation `theta`, `mirror_x` and `offset_um`:
  `M_coarse = T(c_r + offset/p_r) R(theta) F(mirror_x) S(p_m/p_r) T(-c_m)`, with
  `c = ((W-1)/2, (H-1)/2)` the image centre, `p_r` and `p_m` the pixel sizes.
- A crop is half-open (`arr[y0:y1, x0:x1]`) in the full level-0 frame. A point
  is inside when `x0 - 0.5 <= x < x1 - 0.5` (likewise y). With effective crop
  origins `o_r` and `o_m`: `M_full = T(o_r) M_crop T(-o_m)`.
- The pipeline resamples with a pull mapping; a moving point is pushed through
  the final affine plus flow by a fixed-point inversion (converged flag recorded
  per landmark).

## Config schema

The Nextflow side emits one JSON config per task from
`workflows/lib/ManualRegistrationSheet.groovy`; the Python side validates it with
pydantic (a test keeps the key sets equal). The models live in
`src/merxen/manual_registration/config.py` (and `ViewerCacheConfig` in
`src/merxen/config.py`).

| Model | CLI | Fields |
|---|---|---|
| `ManualRegistrationPrepareConfig` | `manual-registration-prepare` | `dataset_name`, `reference`, `moving` (`ManualRegistrationImageInput`), `prepared_dir`, `output_dir`, `force`, `chunk_size` (4096), `tile_size` (1024), `band_height` (2048), `min_level_size` (1024), `compression` (`zlib`), `compression_level` (6), `workers`, `tmp_dir` |
| `ManualRegistrationImageInput` | | `path`, `scene`, `channel_names`, `registration_channel` (`DAPI`), `pixel_size_um` |
| `ManualRegistrationRunConfig` | `manual-registration-register` | `dataset_name`, `preset`, `run_label`, `prepared_store_path`, `result_dir`, `output_dir`, `global_settings`, `row_settings`, `force` |
| `ManualRegistrationCompareConfig` | `manual-registration-compare` | `dataset_name`, `summary_paths` (at least 2), `output_dir`, `task_output_dir` |
| `ViewerCacheConfig` (image-only) | `build-viewer-caches` | `dataset_name`, `latest_zarr_path`, `output_dir`, `image_only: true`, `image_keys`, `downsample`, `min_size`, `build_image_pyramid` |

`ManualRegistrationSettings` extends the registration core's `RegistrationSettings`
and is the set of keys `registration_settings` can override. Defaults are the
`same_section` preset.

| Field | Default | Meaning |
|---|---|---|
| `matrix_scale_tolerance` | 0.02 | Allowed relative difference between the fitted scale and the pixel-size ratio |
| `refine_affine` | true | Run the DAPI affine refinement |
| `registration_pixel_size_um` | 0.85 | Working pixel size of the refinement |
| `refinement_window_um` | 200 | Phase-correlation window |
| `min_refinement_windows` | 20 | Usable windows needed to apply the refinement |
| `max_refinement_shift_um` | 50 | Largest accepted refinement displacement |
| `max_refinement_rotation_deg` | 2 | Largest accepted rotation |
| `max_refinement_scale_change` | 0.02 | Largest accepted scale change |
| `refinement_min_window_quality` | 0.3 | Window quality threshold |
| `refinement_require_moving_tissue` | false | Also require tissue in the moving image (holes, laser-microdissection) |
| `nonrigid_enabled` | true | Estimate the TV-L1 flow |
| `nonrigid_pixel_size_um` | 3.4 | Flow grid pixel size |
| `nonrigid_attachment`, `nonrigid_tightness`, `nonrigid_num_warp`, `nonrigid_num_iter` | 10, 0.3, 5, 10 | TV-L1 parameters |
| `nonrigid_smoothing_sigma_um` | 10 | Confidence-weighted smoothing scale |
| `nonrigid_max_displacement_um`, `nonrigid_max_cycle_error_um` | 20, 5 | Displacement and forward/backward consistency limits |
| `nonrigid_min_local_correlation` | 0.2 | Local DAPI correlation floor |
| `nonrigid_min_jacobian`, `nonrigid_max_jacobian` | 0.5, 2.0 | Jacobian bounds |
| `nonrigid_min_confident_fraction` | 0.1 | Smallest confident fraction of the tissue |
| `nonrigid_highpass_sigma_um`, `nonrigid_local_correlation_sigma_um` | 25, 15 | Length scales of the flow's pre-filter and local correlation |
| `nonrigid_local_residual_tiles`, `nonrigid_local_residual_tile_um` | true, 220 | Local residual passes and their tile size |
| `nonrigid_residual_passes`, `nonrigid_min_residual_correlation_improvement`, `nonrigid_min_residual_initial_confidence` | 1, 0.02, 0.75 | Residual-pass guards |
| `landmark_min_pairs` | 8 | Enabled pairs needed (also checked by the preflight; at least 3) |
| `landmark_min_used_pairs` | 6 | Pairs left after outlier removal |
| `landmark_outlier_floor_um`, `landmark_outlier_k` | 10, 3.0 | A pair is an outlier when its leave-one-out residual exceeds both the floor and `k` times the median residual |
| `landmark_max_outlier_fraction` | 0.25 | Largest outlier share |
| `landmark_outlier_policy` | `exclude` | `exclude`, `keep` or `fail` |
| `landmarks_outside_crop` | `use` | `use`, `exclude` or `fail` for pairs outside the crops |
| `landmark_min_spread_ratio`, `landmark_reflection_check_ratio` | 0.05, 0.5 | Spread and reflection checks |
| `crop_snap_px`, `crop_min_size_px` | 64, 1024 | Crop snapping (at least 64, a power of two; smaller values are refused) and minimum size |
| `tissue_qc_pixel_size_um`, `tissue_qc_sigma_um`, `tissue_qc_highpass_sigma_um` | 10, 20, 200 | Tissue-scale QC (the pixel size is a ceiling: the level used is recorded as `tissue_qc.pixel_size_um`) |
| `nucleus_qc`, `nucleus_qc_pixel_size_um`, `nucleus_qc_window_um` | false, 1.7, 200 | Nucleus-scale QC |
| `tile_size`, `chunk_size`, `build_viewer_pyramid` | 4096, 4096, true | Writing the result store |

**Presets.** `same_section` is the defaults above. `serial_section` overrides
(v1 starting values; changing a preset bumps its version, recorded in every
summary):

| Field | `same_section` | `serial_section` |
|---|---|---|
| `matrix_scale_tolerance` | 0.02 | 0.05 |
| `registration_pixel_size_um` | 0.85 | 6.0 |
| `refinement_window_um` / `min_refinement_windows` | 200 / 20 | 1200 / 12 |
| `max_refinement_shift_um` / `_rotation_deg` / `_scale_change` | 50 / 2 / 0.02 | 200 / 5 / 0.05 |
| `refinement_min_window_quality` / `refinement_require_moving_tissue` | 0.3 / false | 0.2 / true |
| `nonrigid_enabled` / `nonrigid_pixel_size_um` | true / 3.4 | true / 12.0 |
| `nonrigid_smoothing_sigma_um` | 10 | 100 |
| `nonrigid_max_displacement_um` / `_max_cycle_error_um` | 20 / 5 | 200 / 30 |
| `nonrigid_min_local_correlation` | 0.2 | 0.15 |
| `nonrigid_min_jacobian` / `_max_jacobian` | 0.5 / 2.0 | 0.67 / 1.5 |
| `nonrigid_highpass_sigma_um` / `_local_correlation_sigma_um` | 25 / 15 | 120 / 80 |
| `nonrigid_local_residual_tiles` (`_tile_um`) | true (220) | false (1500) |
| `landmark_outlier_floor_um` | 10 | 30 |
| `nucleus_qc` | false | true |

## Python entry points

| Function | Module | Purpose |
|---|---|---|
| `run_manual_registration_prepare(config)` | `merxen.manual_registration.prepare` | Convert/normalise both images and write the prepared store |
| `run_manual_registration(config)` | `merxen.manual_registration.register` | One (pair, preset) run |
| `fit_landmarks(...)` | `merxen.manual_registration.register` | Landmark fit of a prepared store's labelling (also `manual-registration-fit`); it calls `robust_similarity(...)` of `merxen.manual_registration.landmark_fit` |
| `compare_runs(config)` | `merxen.manual_registration.compare` | Comparison CSV, JSON and figure |
| `make_synthetic_pair`, `simulate_landmark_pairs`, `inject_labelling` | `merxen.manual_registration.simulate` | Synthetic pairs and viewer-compatible labelling for tests |
| `run_read_czi_info`, `run_convert_czi_scene` | `merxen.io.czi` | CZI runtime bridge |
| `write_pyramidal_ome_tiff(...)` | `merxen.io.ome_tiff_writer` | Streamed pyramidal OME-TIFF writer |

CLI commands: `convert-czi`, `manual-registration-prepare`,
`manual-registration-register`, `manual-registration-fit`,
`manual-registration-compare` ([CLI reference](../cli.md)).

Scripts for the real-data validation (never imported by the package):
`scripts/manual_registration/derive_test_landmarks.py` (landmarks without a
human, for pairs without ground truth), `export_reference_image.py`
(a SpatialData image channel to a pyramidal OME-TIFF) and
`compare_czi_readers.py` (one-off comparison of two CZI readers).
`derive_test_landmarks.py --inject` writes its labelling into the store as the
viewer's save does, but never replaces a labelling that was not written by
`merxen.manual_registration.simulate` (landmarks placed in the viewer): it stops
with a message, and `--inject --force` saves the old labelling as
`<out stem>.previous_labelling.<UTC>.json` next to `--out` before replacing it.
The slow CZI-pair test skips instead when the store holds such a labelling.

## Validation results

The validation runs on real data (host only, `pytest -m slow`; never in CI). The
tables below are filled from `$MERXEN_EVIDENCE_ROOT/manual_registration/runs/`;
aliases only. **Status: not yet measured** for this revision (the implementation
shipped the tests and scripts; the runs are a later step).

**HD1 ground truth** (a post-Xenium IF image registered with the Explorer matrix
and the pipeline's own refinement and flow; the test simulates landmarks from the
Explorer matrix, plus 1 µm noise and one 40 µm outlier). Test:
`tests/test_manual_registration/test_hd1_ground_truth.py::test_hd1_ground_truth`.

| # | Check | Gate | Measured | Result |
|---|---|---|---|---|
| 1 | Outlier handling: `n_outliers`, pair 3 flagged, `n_used` | 1, pair 3, 11 | | |
| 2 | Landmark matrix vs Explorer matrix at the 4 tissue corners | <= 3 µm | | |
| 3 | Refinement status | `applied` | | |
| 4 | Final affine vs HD1's final affine over the tissue grid | mean <= 0.5, max <= 1.0 µm | | |
| 5 | Non-rigid: status, confident fraction, Pearson after | `applied`; within 0.05 and 0.01 of HD1's | | |
| 6 | Held-out pairs 13 to 16: median final error; affine error | <= 1.0 µm; mean <= 0.5, max <= 1.0 µm | | |
| 7 | Registered DAPI vs HD1's registered DAPI (x16 block mean, log; tissue = Otsu of the downsampled Xenium `morphology_focus` DAPI) | Pearson >= 0.98 | | |
| 8 | Both matrix CSVs load and pass the scale check | no exception | | |
| 9 | Crop to the inner 80 % vs full | mean <= 0.5, max <= 1.0 µm | | |
| 10 | Peak RSS of both runs | <= 48 GB | | |

Check 6 compares each stage against its own truth: `final` against HD1's final
mapping (its affine plus its stored flow, pushed through), `affine` against HD1's
final affine. The simulated points come from the Explorer matrix, which carries
its own error against the image truth, so a comparison against them would
penalise every micron the flow correctly moves; the residuals against the
simulated points are recorded only.

**Zeiss CZI pair, LCM1** (a serial pair; no ground truth; both presets, run
through Nextflow, with the documented `same_section:matrix_scale_tolerance=0.05`
override). Landmarks from `derive_test_landmarks.py` (24 pairs, 8 held out). Test:
`tests/test_manual_registration/test_czi_pair.py::test_czi_pair_both_presets`.

| Quantity | `same_section` | `serial_section` |
|---|---|---|
| Fit: pairs used / enabled, scale ratio, rotation | | |
| Refinement status / reason | | |
| Non-rigid status / reason, confident fraction, Jacobian range | | |
| Tissue Pearson: coarse / landmark / affine / final | | |
| Held-out median residual: landmark / affine / final (µm) | | |
| Nucleus QC | | |
| Wall time, peak RSS | | |

Gates for `serial_section` (provisional, v1 values): `n_used >= 0.75 n_enabled`,
`|scale_ratio - 1| <= 0.05`, the fitted reflection equals the searched mirror and
the other class's median is at least twice the chosen class's, `spread_ratio >=
0.2`; tissue Pearson non-decreasing (within 0.01); held-out final median at most
the landmark median plus 5 µm; an applied flow has Jacobian in [0.67, 1.5], a
skipped one a reason; both result stores, the comparison CSV (2 rows) and PNG
exist; peak RSS <= 48 GB. `same_section` must reach `complete`; its statuses and
metrics are recorded, not gated.

**CZI reader comparison** (one-off, `scripts/manual_registration/compare_czi_readers.py`):
per file, compression, ROIs compared, maximum absolute difference, mismatching
pixel fraction and milliseconds per ROI for `czifile`, `pylibCZIrw` and the
converted OME-TIFF. The two readers are compared on plane Z=0; the OME-TIFF, which
the converter max-projects over Z, against the `pylibCZIrw` maximum over all Z
planes.

| File | Compression | ROIs | Max abs diff | Mismatch fraction | ms/ROI czifile | ms/ROI pylibCZIrw | ms/ROI OME-TIFF |
|---|---|---|---|---|---|---|---|
| LCM1 reference | | | | | | | |
| LCM1 moving | | | | | | | |
| LCM2 reference | | | | | | | |
| LCM2 moving | | | | | | | |

Run date and MerXen commit: to be recorded with the results. Notes on tile-seam
artifacts and crops go here as well.

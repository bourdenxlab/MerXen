# Manual landmark registration

The standalone workflows below retain their full image-evidence/check suite.
For DAPI-only Xenium/MERSCOPE landmarks used as an optional `main.nf` alignment
backend, see [paired manual registration](paired-manual-registration.md).

Registers two images of **adjacent or identical tissue sections** by hand-placed
landmarks, then refines the result automatically. It is meant for pairs the
automatic pipelines cannot align, for example a Zeiss slide scan of a serial
section against a Xenium DAPI image, or a post-Xenium IF image whose Explorer
matrix is unavailable. The registered moving image shares the reference pixel
grid, so its channels can be read against the reference directly.

This is **not a stage of `main.nf`**. It is two standalone Nextflow workflows
(prepare and register) with a human step between them, run from the same
samplesheet, and a third, the landmark check, that tells the labeller how good the
landmarks are while they are being placed (see
[Checking landmarks labelled on another machine](#checking-landmarks-labelled-on-another-machine)):

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

The three workflows live in `workflows/` so they inherit `nextflow.config`, the
execution profiles, the site configs and `workflows/lib/`. `main.nf`, its stage
order and the metro map do not know about them.

## What it does

**Three presets.** The register workflow runs every row once per preset. The
presets differ in the length scale they trust (`cortex_serial` is for adjacent
cortex sections stained for AT8, MAP2 and DAPI; its landmarks are checked by
`merxen manual-registration-check` and registered through the robust landmark
field, see "Registration through the landmark field" below):

- `same_section`: both images show the same cells (a post-Xenium IF image and
  the Xenium DAPI of the same section). Refinement and the optical flow work at
  nucleus scale, the `REGISTER_IMAGES` settings with the flow switched on and
  one extra guard: the refinement is applied only when it does not lower the
  registration-channel correlation (`refinement_min_correlation_gain`).
- `serial_section`: adjacent sections have no 1:1 nuclei. The preset aligns
  tissue-scale structure (layers, vessels, white-matter boundaries) at 6 to
  12 µm per pixel with wider windows, stronger smoothing and tighter Jacobian
  limits; nucleus-level QC is computed but informative only. It inherits the
  `same_section` correlation guard.
- `cortex_serial` (`cortex_serial_v2`): adjacent cortex sections whose nuclei do
  not match. The similarity fit and the DAPI refinement are replaced by a smooth
  **robust landmark field** (a Student-t Gaussian process over the landmark
  displacements) that is checked against the AT8, MAP2 and DAPI images; the DAPI
  affine refinement and the TV-L1 flow are off (`refine_affine=false`,
  `nonrigid_enabled=false`). It keeps the default similarity trimming (10 µm floor,
  `k` 3, at most 25 % trimmed), which is the viewer's live rigid fit, so the
  check's rigid residuals match the flags the viewer shows in its rigid mode. It
  needs the channel roles (`channel_roles` column).

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
   (`nonrigid_*`); both are guarded and fall back with a recorded reason. The
   refinement is applied only when it stays inside the shift, rotation and scale
   limits **and** does not lower the registration-channel correlation of the
   overlay it was fitted on: a correction whose correlation gain is below
   `refinement_min_correlation_gain` is rejected as
   `rejected_no_correlation_gain`, the landmark matrix is kept and the summary
   records `pearson_before` and `pearson_candidate`.
5. Warp every moving channel onto the cropped reference grid, place every
   landmark at each stage (landmark, affine, final) and compute tissue-scale
   Pearson correlation, residuals and optional nucleus-scale QC.
6. Write the result store, the matrices, the landmark table, the summary and the
   QC figure.

### Registration through the landmark field

This path serves `cortex_serial`, or any run with
`field_model=robust_gp`. Steps 1 to 3 are the same (the similarity fit is still
reported, as the `landmark` stage and the `fit` block), then:

1. Fit the robust field to the enabled landmarks with the image check and, when
   `field_window_observations` is on, tissue windows (the same fit as the landmark
   check, so a `landmarks-only` run reproduces the check's field to rounding). The
   classes of every pair (`ok`, `kept_confirmed`, `check`, `likely_misclick`,
   `excluded`, `disabled`) are recorded; nothing is disabled or deleted.
2. Take the least-squares affine of the field mean over the reference crop's tissue
   (64 x 64 lattice) as the **base affine**; `<run_id>_final_matrix.csv` holds it,
   without the field's non-affine part.
3. Sample the field as a pull flow on a grid of about `field_grid_um` (11 µm: 32
   reference pixels at 0.345 µm): node 0 sits one node before the crop, so every
   crop pixel lies inside the grid and the flow is never replaced by the bare
   affine. The flow is the format the registration core's pull and push functions
   read, so the landmarks are pushed through it exactly like the TV-L1 flow.
4. Resample the moving image through the flow. `materialize` decides what is
   written: `full` (the default, `cortex_serial` included) the level-0 grid as for
   the other presets, `preview` the reference and the registered moving image at
   the coarsest pyramid level not coarser than `preview_pixel_size_um` (1 %
   tolerance; 5.5 µm: level factor 16 on a 0.345 µm scan), `none` no result store
   at all (the summary JSON is the record, and an older result store of the run is
   removed). The viewer pyramids of both elements are built from the stored
   elements once the moving image is written.
5. Compute the tissue QC on the registration channel and on each channel role,
   write the field figure, the model and flow files and the summary.

The preview store is a quick look at the registration, too coarse to match single
cells (5.5 µm pixels); `full` is the level-0 grid that cell matching and analysis
need (the milestone's validation table records the time and memory of each).

## Inputs and samplesheet

One samplesheet for both workflows, one row per pair
(`workflows/samplesheet.manual_registration.example.csv`):

```csv
id,reference_image,moving_image,presets,registration_settings,channel_roles
LCM1,/path/to/slide_reference.czi,/path/to/slide_moving.czi,same_section;serial_section,same_section:matrix_scale_tolerance=0.05,
HD1,/path/to/morphology_focus_dapi.ome.tif,/path/to/post_xenium_if.ome.tif,same_section,,
LCM2,/path/to/cortex_reference.czi,/path/to/cortex_moving.czi,cortex_serial,,at8=AF568;map2=AF647;dapi=DAPI
```

| Column | Required | Default | Validation |
|---|---|---|---|
| `id` | yes | | `^[A-Za-z0-9][A-Za-z0-9_.-]*$`, unique; an alias by convention |
| `reference_image`, `moving_image` | yes | | must exist; relative paths resolve against the sheet's folder; extension `.czi`, `.ome.tif`, `.ome.tiff`, `.tif` or `.tiff`; the two differ (also through a symlink) |
| `reference_scene`, `moving_scene` | no | | only for `.czi` (error on TIFF); S index or scene name (case-insensitive); resolved in Python |
| `reference_channel_names`, `moving_channel_names` | no | auto | `;`-list, non-empty, unique; count checked in Python; used verbatim |
| `reference_registration_channel`, `moving_registration_channel` | no | `DAPI` | non-blank; resolved against the channel names in Python |
| `reference_pixel_size_um`, `moving_pixel_size_um` | no | metadata | number > 0 |
| `presets` | no | `--manual_registration_presets` (`same_section`) | `;`-list of `same_section`, `serial_section`, `cortex_serial`, no duplicates |
| `registration_settings` | no | | `[preset:]key=value;...` (grammar below); keys are checked before any task starts |
| `channel_roles` | for `cortex_serial` | | which channel plays each role, `[reference:\|moving:]role=channel;...` (grammar below); syntax and role names are checked before any task starts, the channel names against the prepared store in the task |
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

**Channel roles.** The image evidence of the `cortex_serial` preset reads three
stains: `at8`, `map2` and `dapi`. `channel_roles` names the channel that plays each
role: `roles := item (";" item)*`, `item := [scope ":"] role "=" channel`, with
`scope` `reference` or `moving` (an unscoped item applies to both images and a
scoped one overrides it for that image), `role` one of `at8`, `map2`, `dapi`, and a
channel name that contains no `;`, `=` or `:`. Empty items are skipped; a role
given twice in one scope is an error; a channel must be one of that image's
canonical channel names in the prepared store (exact, case-sensitive; the error
lists the names). For the LCM pairs it is `at8=AF568;map2=AF647;dapi=DAPI`. **Quote
it on a command line**: `--channel-roles 'at8=AF568;map2=AF647;dapi=DAPI'`, because
the shell splits an unquoted value at each `;`. A row whose presets include
`cortex_serial` needs all three roles.

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

The landmark check of a cortex pair runs on the host, from the samplesheet or from
the command line (see
[Checking landmarks labelled on another machine](#checking-landmarks-labelled-on-another-machine)):

```bash
nextflow run workflows/manual_registration_check.nf -profile dwight,conda \
    --samplesheet workflows/samplesheet.manual_registration.csv \
    --outdir "$MERXEN_RESULTS_ROOT/manual_registration"
merxen manual-registration-check --store <prepared store> \
    --channel-roles 'at8=AF568;map2=AF647;dapi=DAPI' --out-dir <dir>
```

`workflows/manual_registration_check.nf` runs `MANUAL_REGISTRATION_CHECK` once per
samplesheet row, with the settings of `--manual_registration_check_preset`. It checks
the store's own labelling (a draft too), writes `<outdir>/<id>/check/` and the check
attribute of the store.

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
| `manual_registration_check_preset` | `"cortex_serial"` | Preset of the check workflow |
| `manual_registration_check_write_attr` | `true` | Write the check attribute into the prepared store |
| `manual_registration_check_suggestions` | `20` | Suggested next clicks (0 to 200) |

Resources: `MANUAL_REGISTRATION_PREPARE` 8 CPUs / 16 GB (the conversion streams
one channel band at a time and maps no file; Nextflow trace of the LCM2 pair
(71849 x 61012 and 79953 x 48136 px, 3 channels): peak RSS 3.8 GB and 358.5 % cpu,
about 3.6 cores busy),
`MANUAL_REGISTRATION_REGISTER` 8 CPUs / 48 GB (a full 80k x 61k px Zeiss scan;
refinement working images are capped at 32767 px per side),
`MANUAL_REGISTRATION_CHECK` 8 CPUs / 16 GB (provisional: to be measured on LCM2),
`MANUAL_REGISTRATION_COMPARE` 1 CPU / 4 GB. The Dwight profile limits prepare,
register and check to two concurrent tasks each
(`manual_registration_prepare_max_forks`, `manual_registration_register_max_forks`,
`manual_registration_check_max_forks`).

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

## Checking landmarks labelled on another machine

The labelling is made in the viewer, often on another machine than the host that
holds the images, and the labeller needs a verdict on every pair, and suggestions
for the next clicks, while clicking. `merxen manual-registration-check` gives them.
It fits the robust landmark field to the clicks, checks every click against the
AT8, MAP2 and DAPI images, classifies every pair and suggests where to click next.
It runs on the store's own labelling or, with `--labelling`, on a file uploaded from
the other machine. The loop:

1. **Save** in the viewer (a draft is fine).
2. **Upload** the store's `zarr.json` (it holds the labelling) to the host.
3. **On the host**, check it:
   `merxen manual-registration-check --store <prepared store> --labelling <uploaded zarr.json> --channel-roles 'at8=AF568;map2=AF647;dapi=DAPI' --out-dir <dir>`
   (the store is the host's prepared store of the same pair; examples use
   placeholders and `$MERXEN_*` variables, never host paths).
4. **Download** `<dataset>_cortex_serial_check.json` from `<dir>` and press
   **Load check file…** in the viewer's Manual registration tab.

The check matches while the labelling in the viewer is unchanged: it records the
digest of the labelling and of the store it was made on, and any edit makes it
stale. **Never copy the host's `zarr.json` back over the local one**: it would
replace the local labelling. A labelling file that was made on another preparation
of the store (its store content digest differs) is refused with "download the store
again".

**What the check does.**

1. Fit the clicks alone (one Matérn 5/2 component; Student-t noise), with the
   leave-one-out residual and p-value of every click.
2. Warp the moving images by that field, check each click against the images
   (the verdict) and find tissue windows (displacements of tissue-scale structure,
   calibrated against an empirical null). Elongated vessel profiles are masked out
   of both (see **Vessel profiles** below).
3. Refit with the evidence: windows as extra observations (two components), a ridge
   click (the image only fixes it along one direction) with an elongated noise
   shape, a click on an elongated vessel profile with extra noise along the vessel,
   and, with `field_trust_confirmed_clicks` (on for `cortex_serial`), clicks the
   image strongly confirms kept at full weight (never a click on an elongated vessel
   profile).
4. Classify every pair from the **lower p and weight of the two fields** (the clicks
   alone, each with its own noise, and the final field), so the windows cannot hide
   a misclick the clicks alone expose, and suggest the next clicks where the field is most uncertain
   (grey matter weighs more than white matter).

**Verdicts** (per click): `confirmed` (the image puts the structure where the click
is), `contradicted` (the image puts it elsewhere; the JSON has the snap position),
`ridge` (the image fixes it only along a line), `inconclusive` and `not_checked`.
A click is matched only where its 300 and 600 µm windows are at least 20 % tissue and
60 % covered by the other section (the tissue windows need 50 % and 90 %): clicks on
a tissue edge or near the pia carry most of the signal two serial sections share, and
the empirical null of the click check is drawn with the same gates. A click with no
usable window stays `inconclusive` (reason `no_tissue` or `outside_coverage`).

**Vessel profiles.** An obliquely cut vessel shows as an elongated profile (lumen,
wall and perivascular space) that moves along its own axis from one serial section
to the next, while the tissue around it follows the field. A click on the end of such
a profile is therefore precise across the vessel (about 6 µm on LCM1) but not along
it: on the LCM1 revision-3 labelling the residuals of clicks on elongated profiles
had an RMS of 45 µm along the vessel against 9 µm across it, and the tissue 100 to
400 µm around twelve flagged vessel-end clicks followed the field, not the click.
Matching the vessel itself would confirm such a click. So, with `field_vessel_mask`
(on by default), the image evidence leaves these profiles out:

- *Non-neuropil* pixels are MAP2-poor (below half-way from the glass level to the
  local MAP2 level, or below 80 % of the way and AT8-dark), or bright in both AT8 and
  DAPI (erythrocytes and vessel walls autofluoresce; tangles are not DAPI-bright).
- Their components (holes of the tissue mask included) are vessel profiles when
  they cover 150 to 30 000 µm², are at least twice as long as wide and at least
  40 µm long, and do not run along the tissue edge. Larger components are gaps and
  tears and stay in. The profiles are dilated by 10 µm (wall and perivascular space).
- The masked pixels never feed the band-pass and orientation filters (normalised
  convolution), and every correlation (click check, tissue windows and their nulls)
  is a masked normalised cross-correlation over the pixels valid in both sections.

Each click's check also records the profile under it (within 15 µm; its axis from
the profile's shape within 200 µm of the click). A click on an elongated, masked
profile gets a fixed noise covariance along the vessel's axis
(`field_vessel_click_along_sd_um`, 45 µm by default, so its SD along the vessel is
about 48 µm and across it the usual click noise) and is never kept at full weight.
The clicks-only field that the classes compare against carries the same
along-vessel noise, so a vessel-end click is not flagged for an offset along its
vessel; an offset across the vessel, or one far beyond the vessel noise (more than
about 140 µm along it), still is. Such a click is often `contradicted`, with a snap
position on the tissue: a stable landmark near it is worth more than the vessel end,
but the field no longer needs it moved. Round vessel profiles (vessels cut across) are stable landmarks and are not
masked.

**Classes and what to do.** p\* and w\* are the lower p-value and weight of the two
fields. A pair is *gross* when p\* < 1e-6 or w\* < 0.05, and *flagged* when
p\* < `field_outlier_p` (0.01). The first row that matches wins.

| Class | Viewer text | When | What to do |
|---|---|---|---|
| `disabled` | `off` | the pair is disabled | nothing |
| `excluded` | `excluded` | enabled but outside a crop (`landmarks_outside_crop=exclude`) | nothing |
| `check` | `check` | gross but the image confirms; flagged with a ridge, inconclusive or unchecked verdict; not flagged but the image contradicts | look at the pair |
| `likely_misclick` | `likely misclick` | gross and not confirmed; flagged and contradicted | re-click or disable it; the JSON has the snap position |
| `kept_confirmed` | `kept: image confirms` | flagged by the field, but the image confirms the click | leave it |
| `ok` | | none of the above | nothing |

Nothing is disabled or deleted automatically.

**Outputs.** In `--out-dir`: `<dataset>_<preset>_check.json` (the check report of
the `merxen.manual_registration.check` schema, plus a `details` block with the
per-click image estimates and vessel profiles, the window summary, the fitted field
model, the timings and the warnings; each pair's `vessel_profile` says whether its
click sits on an elongated vessel profile) and `<dataset>_<preset>_check.png` (with a PDF twin: the clicks by
class with their leave-one-out vectors, the field's uncertainty with the windows and
the suggestions, the residual histograms, and a table of the pairs that are not
`ok`). On the store's own labelling the check is also written as the prepared store's
root attribute `merxen_manual_registration_check`, under the store's write lock and
only if the labelling and the store are still the ones the check read (otherwise the
command exits with code 3 and writes no attribute). With `--labelling` the attribute
is never written. Prepare does not carry the attribute over a rebuilt store: after a
forced rebuild, run the check again. The check is the attribute's only writer;
register never writes the prepared store.

A landmarks-only check (`--no-image-check --no-windows`) needs no channel roles.

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
<id>/prepared/<id>_reference.prepare.json                # per-image provenance (conversion skip key); only when converted (CZI) or normalised
<id>/prepared/<id>_moving.prepare.json                   # same
<id>/prepared/<id>_prepare_summary.json
<id>/prepared/viewer_cache/                              # VIEWER_CACHE publishDir (symlink to the work dir)
<id>/registered/<run_name>/<run_id>_registered.zarr      # run_name = <preset>[_<run_label>]
<id>/registered/<run_name>/<run_id>_landmark_matrix.csv
<id>/registered/<run_name>/<run_id>_final_matrix.csv
<id>/registered/<run_name>/<run_id>_landmarks.csv
<id>/registered/<run_name>/<run_id>_manual_registration_summary.json
<id>/registered/<run_name>/<run_id>_tissue_qc.png (+ .pdf)
<id>/registered/<run_name>/<run_id>_landmark_field.json             # robust-field runs: the fitted model (contract C1)
<id>/registered/<run_name>/<run_id>_field_flow.npz                  # robust-field runs: the pull flow and the uncertainty grid
<id>/registered/<run_name>/<run_id>_field_qc.png (+ .pdf)           # robust-field runs: the field figure
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
`<run_id>_final_matrix.csv` = refined affine **without** the flow; for a robust-field
run the base affine, the least-squares affine of the field, without the field's
non-affine part): 3 lines of 3
comma-separated values, no header, `%.17g`, written by `write_alignment_matrix`
and re-read with `load_alignment_matrix`. Both are in the **full, uncropped**
level-0 pixel frames of both images and map moving to reference.

**Landmarks CSV** (`<run_id>_landmarks.csv`), one row per pair:
`id, colour, enabled, used_in_fit, outlier, outside_crop, reference_x_px,
reference_y_px, moving_x_px, moving_y_px, landmark_x_px, landmark_y_px,
affine_x_px, affine_y_px, final_x_px, final_y_px, residual_landmark_um,
residual_loo_um, residual_affine_um, residual_final_um, final_includes_flow,
push_converged, field_weight, field_p_value, field_verdict, field_class`. The stage
positions are in full reference px; floats `%.6g`. The four `field_*` columns are
empty for similarity runs (and for results written before they existed). For a
robust-field run `used_in_fit` means the pair is an observation of the field,
`outlier` that its class is `likely_misclick`, `residual_loo_um` is the field's
leave-one-out residual, `affine` uses the base affine and `final` is the pair
pushed through the flow; `field_weight` is the final field's Student-t weight,
`field_p_value` its leave-one-out p-value at unit weight, `field_verdict` the image
check's verdict and `field_class` the class.

**Comparison CSV** (`<id>_run_comparison.csv`), one row per run:
`run_id, preset, run_label, fit_n_enabled, fit_n_used, fit_rms_um,
fit_scale_ratio, fit_rotation_deg, refinement_status, refinement_max_shift_um,
refinement_rotation_deg, refinement_scale_change, nonrigid_status,
nonrigid_confident_fraction, nonrigid_pearson_before, nonrigid_pearson_after,
jacobian_min, jacobian_max, tissue_pearson_coarse, tissue_pearson_landmark,
tissue_pearson_affine, tissue_pearson_final, used_median_landmark_um,
used_median_affine_um, used_median_final_um, held_out_median_landmark_um,
held_out_median_affine_um, held_out_median_final_um, nucleus_qc_median_um,
nucleus_qc_pearson, wall_time_s, peak_rss_gb, field_model, field_loo_median_um,
field_loo_p90_um, field_n_check, field_n_likely_misclick, field_n_windows,
materialize_mode`. The `field_*` columns come from the summary's `field` block and
are empty for similarity runs; `materialize_mode` is `full` for them and for results
written before the field existed, and `full`, `preview` or `none` for a robust-field
run.

**Robust-field files** (`<run_id>_landmark_field.json`, `<run_id>_field_flow.npz`).
The JSON is the fitted model (`merxen.manual_registration.landmark_field`, version 1:
trend, kernel, noise, observations with their classes and weights, the digest);
`LandmarkField.from_dict` rebuilds it and must reproduce the digest. The NPZ holds
`flow_yx` (float32, `(2, h, w)`, pull displacement in working pixels),
`working_to_reference` (working pixel to reference crop level-0 pixel),
`base_matrix_crop` and `base_matrix_full` (moving to reference), the crop origins
(`reference_origin_xy`, `moving_origin_xy`), `sd_um` (the field's posterior SD per
axis on a grid of `field_sd_grid_um`, 88 µm by default) with its own
`sd_working_to_reference`, `algorithm` and `model_digest`. The pull of a reference
crop point `r` is
`pull_points_through_registration(r, final_matrix=base_matrix_crop, flow_yx=flow_yx, working_to_reference=working_to_reference)`,
in moving crop pixels. The summary's `field` block records the hyperparameters, the
leave-one-out statistics (final, clicks only, rigid), the class and verdict counts,
the flow (`grid_factor`, `shape_yx`, `max_displacement_um`) and `materialize`.

**Result store placement.** The result store's images are on the **cropped
reference level-0 grid**; their `global` transformation is an `Affine` with the
reference pixel size and the crop origin, so `global` is micrometres of the full
reference frame and landmarks stored in full reference pixels line up. The store
is written to a temporary sibling and renamed into place only when complete.

**Preview stores.** With `materialize=preview` the two elements are coarser: the
reference is the pyramid level of the prepared OME-TIFF with factor `f` (the
largest kept factor not above `preview_pixel_size_um` over the reference pixel
size, with a 1 % tolerance) cropped to the effective crop, and the moving image is resampled onto it. The
result store's attribute records this as `images.<role>.materialization`
(`{"mode": "preview", "level_factor": f, "pixel_size_um": f p}`; a full store has
`{"mode": "full", "level_factor": 1, ...}`, and no block means full). The image
record keeps the reference level-0 pixel size `p` and the crop in full-frame level-0
pixels; `shape_yx` is the element's own shape. An element pixel (row `i`, column
`j`) has its centre at crop-frame level-0 pixel `(j f + (f-1)/2, i f + (f-1)/2)`,
and its SpatialData transform is `[[f p, 0, p x0], [0, f p, p y0], [0, 0, 1]]`
(SpatialData's corner convention). **A viewer must read `materialization` to place
a preview store: the released viewers up to 0.5.2 ignore it and misplace the
elements.** Use `materialize=full` for those viewers.

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
  run is still `complete`. The refinement is `rejected_exceeds_limits` (shift,
  rotation or scale beyond the limits) or `rejected_no_correlation_gain` (the
  correction did not raise the registration-channel correlation: the reason starts
  with the channel's name and reads like
  `DAPI correlation 0.4876 -> 0.4863 (gain -0.0013 < 0.0)`); in both cases the
  residual statistics and the QC figure show the unrefined state. With the check on,
  `refinement.pearson_before` and `refinement.pearson_candidate` give the
  correlation without and with the correction (null when it cannot be measured,
  which also rejects). They are present only when the correction passed the
  limits (status `applied` or `rejected_no_correlation_gain`); otherwise the
  correlation before is `refinement_qc.before.pearson_log_registration_channel`.
  The flow reports confident fraction, Pearson before and
  after, displacement and Jacobian statistics.
- **Field figure** (`<run_id>_field_qc.png`, robust-field runs): the reference
  tissue with every pair coloured by class and its leave-one-out vector (x5), the
  field's posterior SD, the non-affine displacement (the field mean minus the base
  affine, in moving µm, over the reference crop; where it is large the flow does
  work a rigid or affine registration cannot), histograms of the leave-one-out
  residuals (final field, clicks only, rigid), the pairs that are not `ok` with
  their p and weight, and the tissue Pearson of each channel role at each stage.
- **Tissue QC per channel role** (`tissue_qc.by_role`, robust-field runs with
  channel roles): the same band-passed correlation as below for each of AT8, MAP2
  and DAPI (`{role: {channel, moving_channel, pearson, mask_fraction}}`). The
  registration-channel values stay in `tissue_qc.pearson`. The final value should be
  at least the landmark value for every role; a role that falls says the field moved
  that stain's structure away.
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
| `ManualRegistrationRunConfig` | `manual-registration-register` | `dataset_name`, `preset`, `run_label`, `prepared_store_path`, `result_dir`, `output_dir`, `global_settings`, `row_settings`, `channel_roles`, `force` |
| `ManualRegistrationCheckConfig` | `manual-registration-check` | `dataset_name`, `preset` (`cortex_serial`), `prepared_store_path`, `labelling_path` (an uploaded file, or none for the store's own), `output_dir`, `task_output_dir`, `global_settings`, `row_settings`, `channel_roles`, `image_check`, `windows` (none: the settings'), `write_attr` (ignored with `labelling_path`), `n_suggestions` (20) |
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
| `refinement_min_correlation_gain` | 0.0 | Smallest rise of the registration-channel (DAPI) log Pearson correlation a refinement inside the limits must give to be applied; a number in [-2, 2], and `-2` turns the check off (the `REGISTER_IMAGES` behaviour; the override grammar cannot express `None`) |
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
| `field_model` | `similarity` | `similarity` or `robust_gp` (the robust landmark field; the check always uses it) |
| `field_kernel` | `auto` | Matérn components: `auto` (one for clicks alone, two with windows), `single`, `two_scale` |
| `field_student_t_nu` | 4.0 | Degrees of freedom of the Student-t noise (1 to 1000) |
| `field_outlier_p` | 0.01 | A pair is flagged when its p-value falls below this |
| `field_image_check` | true | Check each click against the images (needs channel roles) |
| `field_window_observations` | false (true in `cortex_serial`) | Add tissue-window observations (needs channel roles) |
| `field_window_step_um`, `field_window_search_um`, `field_window_min_spacing_um` | 200, 300, 400 | Window lattice step, search radius and minimum spacing of accepted windows |
| `field_window_noise_floor_um` | 15 | Lower bound of the window noise (per axis) |
| `field_trust_confirmed_clicks` | false (true in `cortex_serial`) | Keep strongly image-confirmed clicks at full weight (never a click on an elongated vessel profile) |
| `field_vessel_mask` | true | Leave elongated vessel profiles out of the image evidence (normalised convolution and masked NCC) |
| `field_vessel_click_along_sd_um` | 45 | Extra noise SD along the vessel of a click on an elongated vessel profile, in µm (0: none) |
| `field_grid_um` | 11 | Node spacing of the pull flow of a field registration, in µm (rounded to whole reference pixels: 32 at 0.345 µm) |
| `field_sd_grid_um` | 88 | Node spacing of the uncertainty grid in the flow file |
| `materialize` | `full` | What a field registration writes: `full` (level-0 grid), `preview` (a coarse pyramid level) or `none` (no result store) |
| `preview_pixel_size_um` | 5.5 | Target pixel size of a preview store; the coarsest kept pyramid level not coarser than this, 1 % tolerance (16 x 0.345 µm on a 0.345 µm scan) |

**Presets.** `same_section` is the defaults above. It differs from the
`REGISTER_IMAGES` defaults in the non-rigid stage being on and in the correlation
guard (`refinement_min_correlation_gain`, 0.0). `serial_section` overrides the
values below and keeps the `same_section` value of every key not listed. They are
the v1 values, kept unchanged after one tuning round on a real serial pair (see
"serial_section tuning round" under Validation results); the preset is
`serial_section_v2` only because it inherits the correlation guard. Changing a
preset bumps its version, recorded in every summary.

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

`cortex_serial` (`cortex_serial_v2`) sets `matrix_scale_tolerance` 0.05,
`refine_affine` false, `nonrigid_enabled` false, `nucleus_qc` false, `field_model`
`robust_gp`, `field_window_observations` true and `field_trust_confirmed_clicks`
true, and keeps `same_section` for every other key (`cortex_serial_v1` also set
`materialize` `preview`).

A setting rule (checked before any work): `materialize` other than `full` is only for
`robust_gp` runs, and a `robust_gp` register run needs `refine_affine=false` and
`nonrigid_enabled=false`. The older presets' provenance digests leave the `field_*`
settings out at their defaults, so their results stay valid after the upgrade.

## Python entry points

| Function | Module | Purpose |
|---|---|---|
| `run_manual_registration_prepare(config)` | `merxen.manual_registration.prepare` | Convert/normalise both images and write the prepared store |
| `run_manual_registration(config, *, force_rerun=False, labelling_payload=None)` | `merxen.manual_registration.register` | One (pair, preset) run; `labelling_payload` (a raw finished labelling) is for validation only, the CLI and Nextflow never set it |
| `fit_landmarks(...)` | `merxen.manual_registration.register` | Landmark fit of a prepared store's labelling (also `manual-registration-fit`); it calls `robust_similarity(...)` of `merxen.manual_registration.landmark_fit` |
| `compare_runs(config)` | `merxen.manual_registration.compare` | Comparison CSV, JSON and figure |
| `run_check(config, *, labelling_payload=None)` | `merxen.manual_registration.check` | The landmark check: check JSON, figure and attribute (`manual-registration-check`); `read_labelling_file`, `write_check_attr` |
| `build_landmark_field(...)`, `classify_pairs(...)`, `check_payload(...)`, `validate_field_settings(...)` | `merxen.manual_registration.field_fit` | The robust field of a pair with image evidence, the C7 classes and the check report |
| `best_affine(...)`, `field_flow(...)`, `field_sd_grid(...)` | `merxen.manual_registration.field_fit` | The field's base affine, its pull flow and its uncertainty grid (register) |
| `fit_landmark_field(...)`, `suggest_landmarks(...)` | `merxen.manual_registration.landmark_field` | The robust Student-t field and the next-click suggestions |
| `check_clicks(...)`, `click_profile(...)`, `tissue_window_observations(...)`, `masked_ncc_map(...)`, `resolve_channel_roles(...)` | `merxen.manual_registration.image_evidence` | The image check, vessel profiles, tissue windows, masked correlation and channel roles |
| `make_synthetic_pair`, `simulate_landmark_pairs`, `inject_labelling` | `merxen.manual_registration.simulate` | Synthetic pairs and viewer-compatible labelling for tests |
| `run_read_czi_info`, `run_convert_czi_scene` | `merxen.io.czi` | CZI runtime bridge |
| `write_pyramidal_ome_tiff(...)` | `merxen.io.ome_tiff_writer` | Streamed pyramidal OME-TIFF writer |

CLI commands: `convert-czi`, `manual-registration-prepare`,
`manual-registration-register`, `manual-registration-fit`,
`manual-registration-check`, `manual-registration-compare`
([CLI reference](../cli.md)).

Scripts for the real-data validation (never imported by the package):
`scripts/manual_registration/derive_test_landmarks.py` (landmarks without a
human, for pairs without ground truth: SIFT for the same section, tissue-scale
window matching for serial sections, `--method auto|sift|window_ncc`),
`export_reference_image.py`
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
tables below are filled from `$MERXEN_EVIDENCE_ROOT/manual_registration/runs/`
(the reader comparison from `$MERXEN_EVIDENCE_ROOT/manual_registration/`);
aliases only. **Run date: 2026-10-28.** HD1 and LCM1 ran at the
integration-branch commit `e77c005` (presets `same_section_v2` and
`serial_section_v2`). The reader comparison and the prepare of the two CZI pairs
ran at the earlier commit `9054957` (later prepare launches found the LCM1 store
unchanged); the code they exercise (the CZI reader and converter, prepare, the
comparison script) did not change between the two commits.

**HD1 ground truth** (a post-Xenium IF image registered with the Explorer matrix
and the pipeline's own refinement and flow; the test simulates landmarks from the
Explorer matrix, plus 1 µm noise and one 40 µm outlier, and compares the result
with a control run of the current code from the Explorer matrix). Test:
`tests/test_manual_registration/test_hd1_ground_truth.py::test_hd1_ground_truth`.

| # | Check | Gate | Measured | Result |
|---|---|---|---|---|
| 1 | Outlier handling: `n_outliers`, pair 3 flagged, `n_used` | 1, pair 3, 11 | `n_outliers` 1, pair 3 flagged and unused, `n_used` 11 | pass |
| 2 | Landmark matrix vs Explorer matrix at the 4 tissue corners | <= 3 µm | 0.64 µm | pass |
| 3 | Refinement status | `applied` | `applied` | pass |
| 4 | Final affine vs a current-code control run from the Explorer matrix, over the tissue grid | mean <= 0.5, max <= 1.0 µm | mean 0.013, max 0.029 µm | pass |
| 5 | Non-rigid: status, confident fraction, Pearson after, vs a current-code control run from the Explorer matrix | `applied`; within 0.05 and 0.01 of the control's | `applied`; confident fraction 0.94497 vs 0.94497 (difference 2e-6); Pearson after 0.8316 vs 0.8315 (difference 8e-5) | pass |
| 6 | Held-out pairs 13 to 16 vs a current-code control run from the Explorer matrix: flow inversion converged at those pairs in both runs; median final error; affine error | converged; <= 1.0 µm; mean <= 0.5, max <= 1.0 µm | converged at pairs 13 to 16 in both runs; median final error 0.014 µm; affine error mean 0.015, max 0.021 µm | pass |
| 7 | Registered DAPI vs the registered DAPI of a current-code control run from the Explorer matrix (x16 block mean, log; tissue = Otsu of the downsampled Xenium `morphology_focus` DAPI) | Pearson >= 0.98 | Pearson 0.99995 | pass |
| 8 | Both matrix CSVs load and pass the scale check | no exception | 4 CSVs load, scale check passes | pass |
| 9 | Crop to the inner 80 % vs full | mean <= 0.5, max <= 1.0 µm | mean 0.36, max 0.58 µm | pass |
| 10 | Peak RSS of the full and crop runs (the control run's wall time and peak RSS are recorded) | <= 48 GB | full 1497 s, 8.8 GB; crop80 946 s, 9.6 GB; control (recorded) 1451 s, 11.5 GB | pass |

Why a control run. HD1's stored registration was written by older flow code: its
recorded settings have no `nonrigid_highpass_sigma_um`,
`nonrigid_local_correlation_sigma_um` or `nonrigid_local_residual_tiles`, which
the current code has. Compared with it, the manual path would be measured
against the difference between two code versions as well as against its own
accuracy. The control is the current pipeline's own registration from the
Explorer matrix: a third run, labelled `explorer`, whose 16 landmark pairs are
the same tissue points as the simulated ones, mapped through the Explorer matrix
exactly (no noise, no outlier, all enabled), with the same coarse, no crops and
HD1's recorded row settings, and with the correlation-gain check off
(`refinement_min_correlation_gain=-2`, as `REGISTER_IMAGES` has none; the `full`
and crop runs keep the preset's 0.0), through the shared core as `REGISTER_IMAGES`
runs it. Checks 4 to 7 compare the `full` run (noisy landmarks and the outlier) with
it, so they measure what the manual path adds to the pipeline's own result.
The comparisons with HD1's stored registration are kept in the validation JSON as
recorded values, never gates: `4_vs_stored`, `5_vs_stored`, `6_vs_stored`,
`7_vs_stored`, `control_vs_stored` (the control's final affine against HD1's over
the grid, and the x16 Pearson of the two registered DAPI images) and
`7_vs_xenium_dapi` (Pearson of the `full`, control and stored registered DAPI
against the Xenium `morphology_focus` DAPI, over the same tissue mask, on the
x16 grid and on 2 and 4 times coarser blocks). `stored_registration_note` states
that HD1's stored registration predates the current non-rigid settings when the
key `nonrigid_highpass_sigma_um` is absent from its recorded settings, and is
null otherwise.

Recorded values (final run, `e77c005`; none of these is a gate). All three runs
(`full`, `crop80`, control) were registered afresh, none reused. The control has
the correlation-gain check off (`-2`, as `REGISTER_IMAGES`); the check passed on
`full` and `crop80` (registration-channel Pearson before and after the
refinement 0.683 to 0.738 and 0.723 to 0.789; window shift median 1.68 to 0.48 µm
on `full`), and every gate value of checks 1 to 9 is identical to an earlier run
at `761ffba`, before the check existed (check 10's times and memory differ from
run to run). Against HD1's stored registration: final
affine over the tissue grid mean 0.32, max 0.65 µm (`4_vs_stored`); confident
fraction 0.9450 vs 0.9467 and Pearson after 0.8316 vs 0.8302 (`5_vs_stored`);
held-out median final error 0.66 µm and affine error mean 0.504, max 0.686 µm
(`6_vs_stored`); for the control, final affine mean 0.32, max 0.67 µm and an x16
Pearson of 0.886 between the two registered DAPI images (`control_vs_stored`;
`7_vs_stored` is also 0.886). Pearson of the registered DAPI against the Xenium
`morphology_focus` DAPI over the tissue mask (`7_vs_xenium_dapi`; x16 is blocks of
16 pixels of the 0.2125 µm grid, 3.4 µm, and x64 is 13.6 µm):

| Block size | `full` | control | stored |
|---|---|---|---|
| x16 | 0.342 | 0.342 | 0.427 |
| x32 | 0.385 | 0.385 | 0.408 |
| x64 | 0.373 | 0.373 | 0.378 |

The manual path reproduces the current pipeline: `full` and the control differ by
at most 0.03 µm in the final affine (check 4), and their Pearson values agree to
the digits shown. The stored registration, written by older code, has a higher
Pearson against the Xenium DAPI at the finest scale (0.427 against 0.342 at 3.4 µm
blocks) and nearly the same at 13.6 µm (0.378 against 0.373). Its final affine also
differs from the control's (mean 0.32 µm). The gap therefore lies between two
versions of the shared registration code (refinement and flow), not in the manual
path; following it up is outside this feature.

Check 6 compares each stage with the control run's: `final` against the control's
final mapping (its affine plus its flow, pushed through), `affine` against the
control's final affine. The simulated points come from the Explorer matrix, which
carries its own error against the image truth, so a comparison against them would
penalise every micron the flow correctly moves; the residuals against the
simulated points, and the landmark matrix against HD1's stored final affine, are
recorded only. Only the held-out pairs count, convergence of the flow inversion
included: the validation JSON lists, per mapping (`full`, `control`), the held-out
ids it did not converge at (`not_converged_ids`).

The validation JSON is written even when a check raises: a gate whose computation
raises is recorded as failed with `{"error": ...}` as its value, a recorded-only
value as `{"error": ...}`, and the later checks still run. Its `runs` entry says,
per run label, whether the run was `reused` (its inputs were unchanged since an
earlier run, so it was not registered again, as its task summary records).

**Zeiss CZI pair, LCM1** (a serial pair; no ground truth; both presets, run
through Nextflow, with the documented `same_section:matrix_scale_tolerance=0.05`
override; final run 2026-10-28 at `e77c005`). Test:
`tests/test_manual_registration/test_czi_pair.py::test_czi_pair_both_presets`.

Landmark derivation (`derive_test_landmarks.py`; 24 pairs, 16 enabled, 8 held
out). SIFT fails on serial sections (the nuclei of adjacent sections do not
correspond), and it did here: retried at 5 µm it found 6957 and 15699 keypoints and
48 ratio matches, none within the gate of the coarse prediction, so no inliers. The
script fell back to tissue-scale window matching (`window_ncc`; the method is
recorded in the validation JSON). The coarse search of the tissue masks chose a
rotation of 180 deg without mirroring (mask IoU 0.875, 0.819 mirrored). Window
matching at 11 µm ran two passes: search ±600 µm, 510 windows, 72 confident, 46
inliers, scale 1.0127, rotation 178.49 deg; search ±200 µm, 577 windows, 83
confident, 59 inliers, scale 1.0127, rotation 177.75 deg, inlier residual median
24.6 µm and p90 44.8 µm. The similarity fit of the 16 enabled pairs (all used) has
scale 1.0132 and rotation 177.76 deg. The moving section is mounted rotated by about
180 deg relative to the reference; the coarse search and the landmark fit handle it
(no mirror).

| Quantity | `same_section` | `serial_section` |
|---|---|---|
| Fit: pairs used / enabled, scale ratio, rotation | 16 / 16, 1.0132, 177.76 deg (rms 33.2 µm, median 28.5 µm; spread ratio 0.54; no reflection, the other class's median is 3226 µm) | same fit |
| Refinement status / reason | `rejected_no_correlation_gain`: DAPI correlation 0.4876 -> 0.4863 (gain -0.0013 < 0.0); the candidate moved the image by up to 40 µm | `rejected_exceeds_limits`: shift 250 µm > 200 µm (scale change -1.3 %) |
| Non-rigid status / reason, confident fraction, Jacobian range | `rejected`: confident fraction 0.093 < 0.100; Jacobian range - (raw flow: median 26 µm, p90 40 µm, Jacobian p01 to p99 0.26 to 2.18) | `rejected`: confident fraction 0.011 < 0.100; Jacobian range - (raw flow: median 137 µm, p90 239 µm, Jacobian p01 to p99 0.02 to 1.80) |
| Tissue Pearson: coarse / landmark / affine / final | 0.174 / 0.194 / 0.194 / 0.194 | 0.174 / 0.194 / 0.194 / 0.194 |
| Held-out median residual: landmark / affine / final (µm) | 19.2 / 19.2 / 19.2 | 19.2 / 19.2 / 19.2 |
| Nucleus QC | disabled (preset default) | informative only: 325 windows of 200 µm, median 67 µm, p90 100 µm, Pearson 0.53 |
| Wall time, peak RSS | 50.6 min, 13.6 GB | 18.0 min, 10.6 GB |

Both presets therefore end at the landmark similarity fit on this serial pair. Its
held-out residuals (final = landmark fit, 8 pairs): median 19.2, p90 34.6, max
51.2 µm.

An earlier run at `632df90`, before the correlation-gain check, shows what the
check is for: `same_section` applied the refinement (DAPI correlation 0.4876 ->
0.4863, up to 40 µm), the held-out median rose from 19.2 to 23.9 µm, the tissue
Pearson went from 0.194 to 0.195, and the flow was rejected (confident fraction
0.068).

Gates for `serial_section` (the preset runs with its v1 values, kept after one
tuning round, see below; preset version `serial_section_v2` because it now
inherits the correlation-gain check, no value changed): `n_used >= 0.75 n_enabled`,
`|scale_ratio - 1| <= 0.05`, the fitted reflection equals the searched mirror and
the other class's median is at least twice the chosen class's, `spread_ratio >=
0.2`; tissue Pearson non-decreasing (within 0.01); held-out final median at most
the landmark median plus 5 µm; an applied flow has Jacobian in [0.67, 1.5], a
skipped one a reason; both result stores, the comparison CSV (2 rows) and PNG
exist; peak RSS <= 48 GB. `same_section` must reach `complete`; its statuses and
metrics are recorded, not gated. Every gate passed on the final run
(`LCM1_validation.json`), and both presets reached `complete`.

**serial_section tuning round** (LCM1, commit `632df90`, before the
correlation-gain check; `serial_section` only, one run per variant). The v1 values
rejected both the refinement and the flow on the serial pair, so four variants
were tried:

- refinement limit: `max_refinement_shift_um=400`;
- lenient flow: `nonrigid_min_local_correlation=0.05;
  nonrigid_min_confident_fraction=0.05`;
- coarse flow: `nonrigid_pixel_size_um=24; nonrigid_smoothing_sigma_um=200;
  nonrigid_highpass_sigma_um=300; nonrigid_local_correlation_sigma_um=200;
  nonrigid_min_local_correlation=0.08; nonrigid_min_confident_fraction=0.05;
  nonrigid_max_displacement_um=300`;
- both: the refinement limit and the coarse flow.

| Variant | Refinement | Flow confident fraction | Held-out median landmark / final (µm) | Tissue Pearson final |
|---|---|---|---|---|
| preset v1 | rejected (shift 250 > 200 µm) | 0.011 (rejected) | 19.2 / 19.2 | 0.194 |
| refinement limit 400 µm | applied (scale change -1.3 %) | 0.004 (rejected) | 19.2 / 44.6 | 0.193 |
| lenient flow | rejected | 0.030 (rejected) | 19.2 / 19.2 | 0.194 |
| coarse flow | rejected | 0.005 (rejected) | 19.2 / 19.2 | 0.194 |
| both | applied | 0.002 (rejected) | 19.2 / 44.6 | 0.193 |

The raw TV-L1 flow of the lenient variant has a median displacement of 137 µm and a
p90 of 239 µm, against a held-out landmark residual of 19 µm: on this pair the flow
does not follow real structure. The local DAPI correlation is high only at tissue
borders, sulci and vessels, and close to 0 in the grey-matter interior. Allowing
the larger refinement made the held-out median worse (19.2 to 44.6 µm).
Conclusion: keep the v1 values. The correlation-gain check (preset version v2,
values unchanged) was added because of the `same_section` run above; the tuning
runs predate it. On this serial pair both presets end at the landmark similarity
fit: held-out median 19 µm, p90 35 µm, max 51 µm (8 pairs). The held-out landmarks
come from window matching, whose own inlier residual median is 25 µm, so these
figures include landmark error. Other serial pairs are not measured; the guards
decide per run.

**CZI reader comparison** (one-off, `scripts/manual_registration/compare_czi_readers.py`):
per file, compression, ROIs compared, maximum absolute difference, mismatching
pixel fraction and milliseconds per ROI for `czifile`, `pylibCZIrw` and the
converted OME-TIFF. The two readers are compared on plane Z=0; the OME-TIFF, which
the converter max-projects over Z, against the `pylibCZIrw` maximum over all Z
planes. The comparison used 20 ROIs per channel and 3 channels per file; the
milliseconds are means over the channels.

| File | Compression | ROIs | Max abs diff | Mismatch fraction | ms/ROI czifile | ms/ROI pylibCZIrw | ms/ROI OME-TIFF |
|---|---|---|---|---|---|---|---|
| LCM1 reference | uncompressed | 60 | 0 | 0 | 2.9 | 7.2 | 22.5 |
| LCM1 moving | uncompressed | 60 | 0 | 0 | 2.8 | 7.4 | 25.4 |
| LCM2 reference | uncompressed | 60 | 0 | 0 | 2.6 | 6.0 | 22.7 |
| LCM2 moving | JPEG-XR | 60 | 0 | 0 | 192.6 | 172.7 | 25.3 |

Versions: `czifile` 2026.10.14 and `pylibCZIrw` 6.1.0 (run once through `uv`, not a
project dependency; commit `9054957`). All four files are Gray16 with one Z plane
and 3 channels (DAPI, AF568, AF647, names from the CZI `Name` attribute), so the
Z projection is not exercised. The readers and the converted OME-TIFF agree on
every pixel of the 60 ROIs per file. Reading JPEG-XR tiles is the expensive case: 173 to 193 ms per ROI
with either reader, against 2.6 to 2.9 ms (`czifile`) and 6.0 to 7.4 ms
(`pylibCZIrw`) on uncompressed files; the converted OME-TIFF reads in 22 to 25 ms
per ROI whatever the CZI compression.

**Prepare** (both pairs, on the isolated CZI runtime). LCM1: 3 channels kept on
both sides (DAPI, AF568, AF647), 0.3451 µm per pixel, reference 77261 x 53651 px
and moving 77257 x 48135 px, both uncompressed CZI (26 to 28 GB each). LCM2: the
same channels; reference 71849 x 61012 px, uncompressed; moving 79953 x 48136 px,
JPEG-XR (6.3 GB). LCM2's moving image has a negative CZI origin (its scene bounding
box starts at x = -131315) and JPEG-XR tiles; both were handled, and the prepared
grid is the bounding box. A prepare took about 17 min per pair (the
`MANUAL_REGISTRATION_PREPARE` task about 16 min, of which the two CZI conversions
took about 12 min; the viewer-cache task about 1 min). At commit `9054957` the
prepare peaked at 38.3 GB (LCM1) and 40.5 GB (LCM2) resident memory in the Nextflow
task trace, well above the 16 GB the process declares (see Resources under Workflows
and launch; the local executor does not enforce the directive). A per-process sampler
on LCM2 (resident anonymous and file-backed pages) showed that almost all of it was
file pages kept mapped by the converter. The reference conversion (uncompressed CZI,
29.6 GB) peaked at 41.9 GB, of which 2.8 GB was anonymous and 39.2 GB file-backed
(the memory map of the CZI, 30.1 GB: each reader thread mapped the whole file and
every touched page stayed resident until the handles closed; and the memory maps of
the reduced pyramid levels, 8.4 GB). The moving conversion (JPEG-XR) peaked at
21.3 GB, 2.5 GB anonymous and 18.8 GB file-backed. The anonymous memory, which the
banding bounds, was 2 to 3 GB. Since the conversion no longer maps any file
(`memmap=False` for the CZI, plain scratch files for the reduced levels), the LCM2
prepare run through `workflows/manual_registration_prepare.nf` (commit `4062912`)
peaked at 3.8 GB in the Nextflow trace, against 40.5 GB at `9054957`:
`MANUAL_REGISTRATION_PREPARE` took 16 min 47 s at 358.5 % cpu (about 3.6 cores, within
the 8 it declares) and the viewer-cache task 1 min 9 s at 2.8 GB. Both converted
OME-TIFFs (reference and moving) are byte-identical (`cmp`) to those of the `9054957`
prepare. The `%cpu` column of a task's `.command.trace` is in tenths of a percent;
Nextflow's own trace report divides it by 10, and 358.5 % is the report's figure.
Re-running prepare on the unchanged LCM1 pair took 12 s (status `unchanged`).

**Tile seams and crops.** The scanner's tile grid is faintly visible in both LCM1
images, and strips of edge artefact (duplicated or bright tiles) run along the top
(moving image) and the bottom (reference) of the overlay in the reference frame
(see the non-rigid QC overlay). The validation's crops are automatic:
`derive_test_landmarks.py` proposes each image's tissue bounding box inset by 1 %,
snapped inward to 64 px. They trim the empty frame but leave those strips inside; a
crop drawn by hand in the viewer, as intended for real use, can exclude them (not
measured here). The LCM1 results above therefore include the artefacts.

### M1: registration through the landmark field (`cortex_serial`)

To be filled from `$MERXEN_EVIDENCE_ROOT/manual_registration/cortex_serial/`
after the milestone's validation runs. Aliases only.

| Run | Settings | Wall time | Peak memory | Store size | Field LOO median / p90 (µm) | Tissue Pearson final (AT8 / MAP2 / DAPI) |
|---|---|---|---|---|---|---|
| | | | | | | |

### Vessel-end clicks (`cortex_serial`, LCM1 revision 3)

The check of the LCM1 revision-3 labelling (121 pairs), run from its snapshot
before and after the vessel mask and the along-vessel click noise (image evidence
`image_evidence_bp20_v2` → `_v3`, landmark field `robust_gp_matern52_v1` → `_v2`).
The evidence behind the change is under
`$MERXEN_EVIDENCE_ROOT/manual_registration/cortex_serial/lcm1_vessel_end_test_rev3/`.

| | Before | After |
|---|---|---|
| Classes ok / kept_confirmed / check / likely_misclick | 105 / 1 / 13 / 2 | 111 / 0 / 7 / 3 |
| Verdicts confirmed / contradicted / ridge / inconclusive | 74 / 0 / 14 / 33 | 42 / 1 / 11 / 67 |
| Clicks kept at full weight (exempt) | 59 | 23 |
| Clicks on an elongated vessel profile | | 41 |
| LOO median / p90 / max, final field (µm) | 18.3 / 54.4 / 116.7 | 17.9 / 56.1 / 118.4 |
| LOO median / p90 / max, clicks only (µm) | 18.3 / 54.0 / 120.2 | 17.8 / 56.5 / 121.9 |
| Click noise σ (µm) | 15.8 | 10.4 |
| Check time (image check, windows, total; s; one run each on a shared host) | 10, 45, 320 | 24, 116, 574 |

- The vessel-end clicks the image check confirmed before (pairs 14, 53 and 81,
  strong and exempt) are no longer confirmed or exempt; their final-field p-values
  rose from 0.0006, 0.0008 and 8e-8 to 0.17, 0.20 and 0.035, so they no longer bend
  the field, and all three are `ok`: their offsets lie along their vessels. So are
  the other vessel-end pairs the flagged review found (31, 51, 58, 61, 102, 109,
  112).
- 34 confirmations were lost: 22 on clicks that sit on elongated vessel profiles, 12
  on clicks next to masked vessels (most on round or oval vessels), whose windows
  drew most of their match from the vessels; the confirmations that stayed lost
  0.01 of their peak correlation.
- With fewer exempt clicks and the vessel-end residuals explained by the
  along-vessel noise, the click noise fits tighter (σ 15.8 to 10.4 µm). Five pairs
  that were `ok` are now `check` (18, 39, 40, 42 and 96), and two non-vessel pairs
  that were `check` are now `likely_misclick`: 12 (torn tissue, where the tissue
  200 to 400 µm around follows the field) and 5 (a pial site).
- Pair 91 (a vessel in a perivascular channel, where the tissue 100 to 400 µm around
  follows the field) is now `contradicted` and `likely_misclick`, with a snap
  position on the tissue; pair 51 (an obliquely cut vessel end) is now `ok`: the
  along-vessel noise explains its residual.
- Two of the twelve flagged vessel-end pairs (30 and 80) are not found as vessel
  profiles at 2.76 µm (their vessels merge with neighbouring dark channels); they
  keep isotropic noise.
- Pair 50 (a round vessel in a gap where the tissue follows the click) is now `ok`
  (`inconclusive`, as before): the vessel-end clicks near it no longer pull the
  field, and its leave-one-out residual fell from 31.5 to 19.4 µm.
- On LCM2 (revision 2) the confirmations that remain are tissue matches: the median
  offset of confirmed clicks rose from 8.4 to 10.2 µm, so the M1 bar moved from 10 to
  12 µm (plan D11). Held-out errors fell (12 held-out windows: median 46.1 to 37.3 µm;
  block hold-out: 60.9 to 55.1 µm), and the strong confirmations fell from 20 to 7, so
  keeping them at full weight no longer changes the field there.

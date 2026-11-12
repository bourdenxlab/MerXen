# Cellpose HITL model selection (HITL-1)

HITL-1 adds optional cell-model selection and safe reuse checks. It also shares
the production halo-tile input, inference and filtering operation with acceptance
checks. Managed local annotation, numeric-mask import and named-model training
are described in [the annotation/training workflow](../hitl/managed-annotation-training.md).
Held-out model evaluation remains separate work.
Dwight acceptance uses Conda; Apptainer validation remains deferred.

With no optional model/override parameters, Nextflow emits the same segmentation
configuration JSON and process scripts as before. Stock `cpsam_v2` remains the
default. Legacy stock masks retain `unverified` provenance and remain reusable.

## Select a model

Set `xenium_cellpose_model`, `merscope_cellpose_model` or `atera_cellpose_model` to a bundle directory,
its weights file, a unique trained model name, or a 64-character registry hash. A directory contains `model`
and `model.json`. Hashes resolve under `cellpose_model_store`, which may also be
set with the exported `MERXEN_CELLPOSE_MODEL_STORE` variable. Dwight supplies
`$MERXEN_STORAGE_ROOT/cellpose_models` as its site default.

The same-named samplesheet columns override the run parameter for that platform;
blank cells inherit it. Cell-model selection and the overrides below leave the
independent stock DAPI nuclei model and its settings unchanged.

For Atera, select `segmentation_source=cellpose_proseg` explicitly. Its default
`native` route and `native_proseg` use vendor masks, so model/threshold options
are refused there rather than silently ignored. Atera keeps the vendor nuclei;
it does not run the independent Cellpose nuclei process. Its Cellpose stage
writes no transcript CSV; tiled ProSeg reads the store's parquet.

| Parameter | Default | Effect |
|---|---|---|
| `<platform>_cellpose_model` | unset | Select a bundle/path/hash for that platform |
| `<platform>_cellpose_cellprob` | unset | Override the cell-probability threshold |
| `<platform>_cellpose_flow_threshold` | unset | Override the flow threshold |
| `<platform>_cellpose_diameter` | unset | Override inference diameter |
| `<platform>_cellpose_final_max_area_um2` | unset | Override the final cell area ceiling |
| `<platform>_mask_max_eccentricity` | unset | Override the per-tile eccentricity limit; 1.0 disables exclusion by eccentricity |
| `allow_candidate_model` | false | Explicitly permit a candidate bundle |
| `cellpose_protected_output_roots` | Dwight: published results root | Roots protected from experimental arms; required on other hosts |

Here `<platform>` is `xenium`, `merscope` or `atera`. Numeric overrides also accept
same-named samplesheet columns. Explicit zero is preserved. Named models restore
their saved inference/filter defaults, tile grid, channels and density recipe;
explicit platform overrides take precedence. Existing full-path/hash selectors
retain their prior behavior: evaluation settings are not applied automatically.

Use a Nextflow JSON/YAML params file to supply a singleton tile-size list, for
example `"cellpose_tile_size_candidates": [6144]`. A custom model must use its
evaluated tile size; memory pressure causes refusal rather than silently choosing
a differently normalized tile grid.
Stock controls using the new per-platform overrides also require a single pinned
tile size and a separate protected arm. Unknown historical stock masks remain
reusable for ordinary runs; an explicit control refuses unknown or incomplete
provenance so the requested change cannot silently reuse an older mask.

## Bundle metadata and preflight

`model.json` follows `CellposeModelMetadata` in
`src/merxen/segmentation/hitl/registry.py`:

| Field | Contract |
|---|---|
| `schema_version` | 1 |
| `weights_sha256` | Full lowercase SHA-256 of `model` |
| `platform`, `species` | `XENIUM`/`MERSCOPE`/`ATERA`, `human`/`mouse`; match the run |
| `channels` | One to three unique channel names in exact input order |
| `image_key` | `morphology_focus` or `MERSCOPE_z_projection`, matching the platform |
| `pixel_size_um` | Positive native pixel size; both axes of the full affine must agree within 1% |
| `tile_size` | Evaluated halo-tile size, matching the singleton candidate list |
| `normalization_route` | `A` (the plan's production tile normalization contract) |
| `annotation_style`, `annotation_guide_version`, `license_note` | Nonempty provenance strings |
| `status` | `accepted`, `candidate` with opt-in, or `rejected` (always refused) |
| `cellpose_version`, `torch_version` | Exact runtime versions, including torch's build suffix |
| `evaluated_cellpose`, `evaluated_mask_filter` | Effective evaluation settings using the corresponding Python config schemas |
| `evaluation_summary` | Optional evaluation-summary path |

Preflight checks actual checkpoint bytes and native-image metadata before model
inference. It refuses missing channels, wrong channel order, aligned-image
substitutions, resolution/runtime mismatches and unpinned tile sizes. Checksum
validation also surrounds checkpoint loading to catch fallback or changed bytes.

## Output and reuse safety

Each custom arm needs its own output directory. Both Nextflow preflight and the
Python entry points protect the configured baseline roots, following symlinks.
Published links also protect their targets in old Nextflow work directories.
Custom arms must also copy reusable source zarrs outside the baseline: a linked
baseline source could be mutated by build or later stages. The same checks apply
when forced reruns are requested.

A verified custom mask records weights, exact runtime, effective Cellpose
settings, platform/species/channels/native image, affine-derived pixel sizes,
filters and tiling. Reuse requires that identity and the statistics identity to
match. Missing/unverified provenance, a changed configuration, or incomplete
custom outputs cause refusal without replacing the existing artifacts. Select
a fresh arm directory to test a different configuration.

Runs that start after segmentation still verify the published mask, sidecar and
statistics against any supplied model selector and the effective settings.
They refuse a mismatch before scheduling downstream work. Omitting a selector
preserves the usual downstream workflow and its stored provenance.

The shared per-tile function is `run_cellpose_tile`. It returns the whole halo
tile's uint8 model input, raw labels, filtered labels and optional probability
logits. Production keeps the original stitching, final area filtering and
transcript assignment. Regression hashes captured before extraction pin the
synthetic masks, logits and stitching statistics.

## Start an annotation pilot

`merxen cellpose-hitl-export --config pilot.json` exports a CPU-only correction
package from a native image and existing Cellpose labels. It reads images and
labels only, avoiding whole-transcriptome tables and transcript partitions.
The baseline stitching statistics pin the actual global tile grid. Training
inputs are normalized on each complete enclosing halo tile before cropping;
the manifest records percentile bounds, their numeric dtype and sampling stride.
The guide is bundled with the installed package, so export also works outside a
source checkout.

The default is six 768 px crops spread across an image-derived tissue mask,
with at least 90% tissue coverage and the plan's 656 px tile-core margin.
Explicit native pixel boxes are also supported. These are unstratified pilot
training crops, never a held-out evaluation split. The portable archive contains
TIFFs, editable `_seg.npy` pre-fills, raw/normalized arrays, previews, a manifest,
checksums, `review.tsv` and the [annotation guide](../hitl/annotation-guide.md).
Existing export directories containing edits are refused without replacement.
Import, training and automatic adoption are not launched by this command.

For user-selected regions, `cellpose-hitl-prepare-rois` creates portable native
context. The lightweight `add_roi_picker` helper adds a crop-selection dock to
the existing napari viewer. Its GeoJSON is passed as `roi_geojson_path`; source,
frame and tile-grid identity are verified before any crops are written. See the
[automatic and viewer-selected workflows](../hitl/pilot-workflows.md).

See [the Atera pilot instructions](../hitl/testing-guide.md#5-atera-correction-pilot)
and [compatibility audit](../hitl/atera-compatibility.md).

See the [testing guide](../hitl/testing-guide.md) for a stock-checkpoint fixture,
fast validation tests and the locked HD1 tile check. A passing fixture confirms
plumbing; it does not establish that a fine-tune improves segmentation.


## Whole-mask geometry cleanup

Every new Nextflow Cellpose run (Xenium, MERSCOPE and Atera, including stock
weights) enables a physical geometry filter after whole-image mask stitching
and before transcript seeding and ProSeg. A mask is removed only when **all**
three conditions hold:

- area below 100 µm²;
- maximum local thickness below 5 µm;
- solidity (area divided by convex-hull area) below 0.75.

This rejects thin concave crescents while protecting compact small cells.
Retained cell masks keep every original pixel, including attached processes.
Objects clipped by the outside of the full image are retained because their
complete geometry is unknown. The maximum thickness uses calibrated native
row/column pixel spacing. No erosion or per-crop cleanup is performed.

The shared controls are `cellpose_geometry_filter_enabled` (default `true`),
`cellpose_geometry_max_area_um2`, `cellpose_geometry_max_thickness_um` and
`cellpose_geometry_max_solidity`. Set the enabled flag to `false` to reproduce
an older area-only recipe. These controls do not change Cellpose flow or
cell-probability settings. Final minimum/maximum area bounds still apply;
use an appropriate `atera_cellpose_final_max_area_um2` for larger neurons.

Geometry settings form part of the saved mask/model identity. Old identities
without this field retain their original meaning; enabling cleanup requires
fresh masks in a separate output directory, rather than relabeling old outputs
as cleaned. Removed-mask probabilities are cleared before transcript assignment.

## Four-way Atera comparison

Select `segmentation_source=cellpose_comparison` with a named retrained model to
retain independent analysis branches:

| Analysis selector | SpatialData cell table | Segmentation |
| --- | --- | --- |
| `original_seg` | `table_original` | Vendor masks, no ProSeg |
| `stock_cellpose` | `table_MOSAIK_stock_cellpose` | Stock weights, no ProSeg |
| `proseg_mask` | `table_MOSAIK_cellpose` | Retrained weights, no ProSeg |
| `reseg` | `table_MOSAIK_proseg` | Retrained weights followed by ProSeg |

Both Cellpose jobs use identical channels, native tiles, normalization and mask
filtering. Stock outputs are published under `segmentation/stock_cellpose/`;
retrained outputs retain `segmentation/`. ProSeg receives only the retrained
masks. During its final transcript pass, the stitch also looks up stock labels
without refining them. Each branch has separate assignment columns, masks,
shapes, expression/control counts and model provenance. Viewer caches and image
quantification include the stock branch. Select all four branches explicitly
in analysis/MENDER settings to apply the configured downstream analyses to each.
The `all` analysis selector includes all four on comparison rows.

Use a fresh output directory to preserve earlier runs. This route does not
retrain weights or overwrite a published model bundle.

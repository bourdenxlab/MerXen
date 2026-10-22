# Cellpose HITL model selection (HITL-1)

HITL-1 adds optional cell-model selection and safe reuse checks. It also shares
the production halo-tile input, inference and filtering operation with acceptance
checks. Crop annotation, training and model evaluation arrive in later milestones.
Dwight acceptance uses Conda; Apptainer validation remains deferred.

With no optional model/override parameters, Nextflow emits the same segmentation
configuration JSON and process scripts as before. Stock `cpsam_v2` remains the
default. Legacy stock masks retain `unverified` provenance and remain reusable.

## Select a model

Set `xenium_cellpose_model` or `merscope_cellpose_model` to a bundle directory,
its weights file, or a 64-character registry hash. A directory contains `model`
and `model.json`. Hashes resolve under `cellpose_model_store`, which may also be
set with the exported `MERXEN_CELLPOSE_MODEL_STORE` variable. Dwight supplies
`$MERXEN_STORAGE_ROOT/cellpose_models` as its site default.

The same-named samplesheet columns override the run parameter for that platform;
blank cells inherit it. Cell-model selection and the overrides below leave the
independent stock DAPI nuclei model and its settings unchanged.

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

Here `<platform>` is `xenium` or `merscope`. Numeric overrides also accept
same-named samplesheet columns. Explicit zero is preserved. Values from a model's
evaluation are **not** applied automatically; differences produce warnings.

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
| `platform`, `species` | `XENIUM`/`MERSCOPE`, `human`/`mouse`; match the run |
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

See the [testing guide](../hitl/testing-guide.md) for a stock-checkpoint fixture,
fast validation tests and the locked HD1 tile check. A passing fixture confirms
plumbing; it does not establish that a fine-tune improves segmentation.

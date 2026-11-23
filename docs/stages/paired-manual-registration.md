# Manual registration of paired Xenium / MERSCOPE sections

Select `alignment_backend = "manual_landmarks"` in `main.nf` to replace an
unsatisfactory automatic serial-section alignment. Xenium is the reference;
MERSCOPE is moving. This implements the existing `align` stage through the
robust cortex landmark field. `MATERIALIZE_ALIGNMENT`, `align_qc` and downstream
contracts are shared with automatic alignment. VALIS remains the default.

These packages contain DAPI only. This route explicitly fits **clicks alone**,
with the AT8/MAP2 image checks and tissue-window observations off. The standalone
[manual registration](manual-registration.md) prepare / check / register
workflows retain their complete channel-role, vessel, image-evidence and
comparison suite for other tissue. No standalone presets are changed.

## Prepare on the processing host

Use the existing **native** images in the enriched `latest_spatialdata.zarr`
stores, rather than previously aligned images. Create an `AlignmentConfig` JSON
per pair, for example `HD1.prepare.json`. These illustrative relative paths
should be replaced by expanded local-configuration values:

```json
{
  "pair_id": "HD1",
  "merscope_zarr_path": "results/HD1/merscope/latest/latest_spatialdata.zarr",
  "xenium_zarr_path": "results/HD1/xenium/latest/latest_spatialdata.zarr",
  "output_dir": "paired_landmarks/HD1/qc",
  "merscope_image": {"image_key": "MERSCOPE_z_projection", "dapi_channel": "DAPI"},
  "xenium_image": {"image_key": "morphology_focus", "dapi_channel": "DAPI"}
}
```

Run with the MerXen runtime from the chosen feature revision:

```bash
merxen prepare-paired-landmarks --config HD1.prepare.json \
  --output-store "$MERXEN_EVIDENCE_ROOT/paired_landmarks/HD1/HD1_prepared.zarr" \
  --target-pixel-size-um 0.5
```

The package contains `reference` and `moving`, both single-channel DAPI, plus
viewer image pyramids. It contains no transcripts, masks or count tables.
Integer block averaging keeps the nominal display pixel size at or below the
target. Original pixel-centre offsets and complete calibrated dataset-to-image
affines, including anisotropy, are retained separately. Native shape, channel
identity and small image-content fingerprints are recorded. The nominal
display scale does not replace the native affine.

Creation is atomic. An unchanged invocation preserves the package and its
landmarks. Different inputs under an existing package are refused: use a new
destination. Native stores and tissue GeoJSON annotations are read only here.
Original segmentation metadata helps resolve image frames; explicit image
`pixel_size_um` / `dataset_to_image_matrix` overrides remain available when an
older store lacks reliable calibration.

## Download and label each pair

Copy each `<id>/` directory to the laptop with rsync. In the viewer:

1. Dataset loader → **Manual Registration** → **Load image data** →
   `<id>_prepared.zarr`. Use the manual-registration loader.
2. Reference is Xenium; moving is MERSCOPE. Coarse rotation / flip / shift and
   optional crops work as in the standalone tool. Crops affect the fit; the
   pipeline still materializes onto the full Xenium image grid.
3. Click corresponding **tissue structures**, such as vessel bifurcations and
   grey/white-matter boundaries. Adjacent sections need not share nuclei.
   Distribute clicks over the shared tissue, including perimeter and interior.
   Start with about 20–30 well-distributed pairs and add points where the preview
   remains poor. At least eight enabled pairs are required by default.
4. **Finish labelling**; a draft is insufficient. Repeat for each pair.

The viewer saves landmarks in the prepared Zarr's root metadata. Upload only
its **`zarr.json`** as `<landmark-root>/<id>/landmarks.json` on the processing
host. Do not overwrite the host package's root metadata with the laptop's
metadata. The backend reads the uploaded JSON separately, validates it against
the server package, and leaves the package unchanged.

## Use the alternative alignment stage

Set the samplesheet columns `alignment_manual_store` (server prepared Zarr) and
`alignment_manual_landmarks` (uploaded JSON) per pair, or use
`--alignment_manual_root <landmark-root>` with this layout:

```text
<landmark-root>/HD1/HD1_prepared.zarr/
<landmark-root>/HD1/landmarks.json
```

First run a bounded alignment phase using a **new pinned code export**, a new
launch/work directory, the existing results root and a sheet containing only
the pairs being fixed. Keep the original run's other settings:

```bash
export PYTHONPATH="$MERXEN_CODE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
nextflow run "$MERXEN_CODE_ROOT/workflows/main.nf" -profile conda \
  -params-file "$MERXEN_EVIDENCE_ROOT/paired_landmarks/runtime.params.json" \
  --samplesheet "$MERXEN_EVIDENCE_ROOT/paired_landmarks/pairs.csv" \
  --outdir "$MERXEN_RESULTS_ROOT" \
  --alignment_backend manual_landmarks \
  --alignment_manual_root "$MERXEN_EVIDENCE_ROOT/paired_landmarks" \
  --start_stage align --stop_stage align_qc \
  -work-dir "$MERXEN_EVIDENCE_ROOT/paired_landmarks/work"
```

Use the site's usual profile/site config in place of `conda` where appropriate.
The source export must head `PYTHONPATH` for every task, including the existing
materializer, even when reusing a runtime installed from another checkout.
Remove row-level stage overrides, which take precedence over the command.
Stop an older run that could write these same stores **before applying** the
new transform. Retain its pinned source and work cache.

To review the fit **before any analysis-store write**, use `merxen align
--config <config.json>` with `backend=manual_landmarks`, `manual.prepared_store`,
`manual.labelling_path`, and `write_aligned_zarrs=false`. Inspect
`manual_dapi_overlay.png`, `landmark_diagnostics.csv` and
`registration_summary.json`. The bounded pipeline phase initializes and applies
the validated transform; it is not a review-only command.

The field must converge, satisfy Jacobian limits and pass a sampled push/pull
round trip. Draft, wrong-pair, stale-grid and changed-native-image inputs are
refused. Landmarks, settings and package digests are recorded. A tissue-scale
DAPI shared-coverage mask and its affine support paired annotation/reports;
this does not enable AT8/MAP2 evidence or nucleus-scale refinement.

`alignment_manual_settings` accepts the cortex settings-override grammar; this
route always keeps image checks and tissue windows off.
`alignment_manual_grid_spacing_um` defaults to 11 µm. The CLI's `manual` config
also exposes inverse convergence and Jacobian limits.

## Reuse completed work and refresh affected consumers

The bounded phase above runs `MANUAL_ALIGN`, `MATERIALIZE_ALIGNMENT` and
`ALIGN_QC`, plus input-layer validation. It does not launch gene clustering or
the rest of the downstream range.

| Result | Action |
|---|---|
| Native images, masks, ProSeg assignments, counts, image quantification, native QC/MECR | Reuse |
| Native tissue boundary GeoJSON files | Preserve without changing their coordinates |
| Aligned images, masks, transcripts, registered centroids and viewer pyramids | Rebuild through `MATERIALIZE_ALIGNMENT` |
| Alignment QC, paired spatial overlays and aligned density plots | Refresh |
| `COMPARE` gene-total statistics | Reuse completed results; these depend on counts |
| Gene PCA/UMAP and expression reference mapping | Reuse completed expression results |
| Neighbourhood statistics or reports using aligned coordinates/shared tissue | Refresh the affected consumers |
| Cortical depth/object distances | Reuse native-frame results; refresh consumers configured in the aligned frame |
| Tasks never completed in the earlier run | Run once after registration is corrected |

Use the old trace and actual task settings to select subsequent bounded or
`only_stage` launches. Clustering also computes shape/coordinate features, so
do not blindly reuse a whole clustering task because its UMAP is based on
expression. Conversely, starting the complete `align`-to-end range does not
automatically skip independent stages. A combined task may still repeat an
expression calculation when its spatial features need updating; those operations
are not all separate cache units in the current pipeline.

Aligned variants keep the existing `*_aligned_nonrigid` names and pair manifest
in the **existing** `latest_spatialdata.zarr` stores. Native elements and count
matrices stay present. Replacement updates the transform fingerprint and its
aligned variants. Archive earlier alignment outputs separately when comparing
automatic and manual results; never replace the native annotation files.

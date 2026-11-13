# Prepare Xenium datasets for HITL annotation

Xenium models use **DAPI + 18S RNA**. The optional transcript-density image is
for viewing while choosing crops; its presence never adds a model input channel.
It is stored permanently as `images/xenium_transcript_density`, with native-grid
coordinates, provenance and viewer pyramids. Atera's existing three-channel
route and density provenance remain compatible.

The Xenium Cellpose cell-area ceiling is now **2,000 µm²**, controlled by
`xenium_cellpose_default_max_area_um2`. The explicit row/parameter override
`xenium_cellpose_final_max_area_um2` takes precedence. The new ceiling also
applies when using named Xenium models whose old evaluation metadata recorded
400 µm², unless explicitly overridden. Nuclear and MERSCOPE ceilings retain their
existing settings. Geometry filtering still removes small thin concave objects.
Changing the ceiling does not restore cells already removed from saved masks;
new segmentation after training uses the updated ceiling.

## Prepare existing review stores

Keep each existing review store and its original native SpatialData build. The
native build must retain `experiment.xenium` and detected micron transcript
coordinates, `is_gene` and `qv`; processed ProSeg point coordinates must not be
mistaken for raw micron positions. The preparation checks matching image grids,
rasterizes original gene detections at QV ≥20 across all z planes (including
unassigned detections), and uses Gaussian σ = 0.15 µm by default.

Create a private JSON manifest, with one entry per dataset:

```json
[
  {
    "dataset_alias": "HD1",
    "platform": "XENIUM",
    "source_zarr": "review.zarr",
    "native_source_zarr": "native.zarr",
    "baseline_stats_path": "cellpose_stitching_stats.json",
    "output_dir": "hitl-preparation/HD1"
  }
]
```

Resolve paths before launching, and use absolute paths in this private manifest:
tasks run in their own work directories. Run:

```bash
nextflow run "$MERXEN_CODE_ROOT/workflows/prepare_hitl_viewer.nf" \
  -profile dwight -c "$MERXEN_RUNTIME_CONFIG" \
  --hitl_preparation_manifest "$MERXEN_HITL_PREPARATION_MANIFEST" \
  -work-dir "$MERXEN_HITL_PREPARATION_WORK" \
  -with-trace "$MERXEN_HITL_PREPARATION_TRACE"
```

The launch environment must provide the configured MerXen runtime, as for the
main workflow. This preparation runs four CPU tasks at once by default; set
`--hitl_preparation_max_forks` to reduce concurrency. It runs no segmentation,
ProSeg, clustering or other downstream analysis.

For each review store the task:

1. Backs up root metadata/files and all annotation geometry in
   `output_dir/annotation_backup`.
2. Retains its microscope metadata and installs the density image and pyramids.
3. Reuses existing label/outline caches and prepares image caches.
4. Adds `merxen_hitl_roi_context.json` and `hitl_export_settings.json` to the
   review store, so downloading the whole store also transfers the crop setup.
5. Writes `output_dir/preparation.json`, proving that the shapes, labels,
   transcript data and tables were unchanged. No existing crop setup is replaced
   by a different setup. Other platform stores and external annotation packages
   are untouched.

The existing Cellpose masks remain the annotation prefills. Old stock masks need
no new model-identity sidecar to select crops; the saved original tile statistics
still define the crop grid and normalization context.

## Select crops and annotate locally

1. Download the complete updated review Zarr, retaining all its files and
   subdirectories. Keep existing annotation packages beside it.
2. Open the dataset **standalone as Xenium** in the viewer. In Images, use the
   native morphology DAPI and 18S channels; the red Transcript density image is
   available as a viewing guide.
3. Open **HITL seg crops**. The crop setup loads automatically from the store.
4. Add rectangles inside the cyan guides. Include large neuronal somata and
   processes as well as small cells and difficult/background areas, spreading
   crops across donors and tissue regions. Aim for 800–1,200 cells initially.
5. Choose **Save areas and prepare for annotation**. Select each crop and choose
   **Annotate cell masks**, save in Cellpose, then close that GUI. The crop is
   marked annotated after its output passes validation.
6. Transfer each completed package to `annotations/<package-id>/` under the
   configured server HITL root. Preserve old packages; select the new Xenium
   packages together when training a uniquely named Xenium model.

The exported model arrays contain DAPI + 18S and zero padding in the third
Cellpose slot. Density remains a viewer guide. Start training from stock
Cellpose-SAM rather than the three-channel Atera model. See
[cellpose-hitl.md](cellpose-hitl.md) for training, named model reuse and resuming
segmentation and downstream analysis.

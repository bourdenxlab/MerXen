# Choose and correct HITL pilot crops

There are two ways to choose crops: let MerXen spread a small set across tissue,
or choose locations yourself while inspecting the processed sample in the
viewer. Both produce the same pre-filled correction package for the **separate
Cellpose GUI**. The MerXen viewer selects crop locations; Cellpose edits cell
outlines. Import, training and adoption follow in subsequent milestones.

```mermaid
flowchart LR
    A[Build masks and viewer caches] --> B{Choose crop locations}
    B --> C[Automatic tissue-spanning pilot]
    B --> D[Select regions in the viewer]
    C --> E[Export pre-filled crops]
    D --> E
    E --> F[Correct cells in Cellpose GUI]
    F --> G[Save edits and review sheet]
```

## 1. Prepare the sample once

Use the current Atera integration branch and your usual site/Conda setup. For
a new Atera run, prepare Cellpose masks, enriched stores and viewer caches:

```bash
nextflow run workflows/main.nf -profile dwight,conda \
  --samplesheet samples.csv \
  --analysis_mode atera \
  --segmentation_source cellpose_proseg \
  --viewer_cache_enabled true \
  --stop_stage build_viewer_caches \
  --outdir "$MERXEN_HITL_RUN_OUT"
```

Set `MERXEN_HITL_RUN_OUT` to a fresh private output root. This is the current
Cellpose-to-tiled-ProSeg route, so it includes ProSeg. If the sample has already
been processed, reuse its completed latest/review store and stitching statistics
instead of recomputing it. Atera's default `native` route provides vendor masks,
not Cellpose pre-fills.

Create a private `pilot.json` with the config shown in
[the Atera pilot instructions](testing-guide.md#5-atera-correction-pilot).
Point `source_zarr` at the completed latest/review store, `mask_key` at
`cellpose_cell_labels`, and `baseline_stats_path` at the original Cellpose
`cellpose_stitching_stats.json`. Keep the source and published output in
`protected_roots`. Use a fresh `output_dir` for every export.

The initial crop exporter supports native Atera and Xenium images. Neither
path loads transcript partitions or whole-transcriptome expression tables.

## 2A. Automatically selected crops

Leave `boxes` and `roi_geojson_path` unset, then run in the pipeline environment:

```bash
merxen cellpose-hitl-export --config pilot.json
```

Defaults select six 768 px crops with at least 90% coverage by an image-derived
tissue mask and spread them spatially. Inspect `context/overview.png` and the
per-crop previews before correcting anything. This is an unstratified training
pilot, not an anatomically balanced or held-out test set. Selection does not
rank windows by segmentation errors.

Continue at step 3. The archive also contains `roi_context.json`, which can be
used later to choose a different set of locations in the viewer.

## 2B. Regions selected in the viewer

### Prepare the small viewer context file

On the pipeline machine, before opening the sample for ROI selection:

```bash
merxen cellpose-hitl-prepare-rois --config pilot.json --output roi_context.json
```

This reads image metadata and the original tile grid, and writes a small JSON
file. It creates no crops or masks. Copy it to the machine running the viewer;
the source data need not be accessible to the ROI helper itself.

### Select locations

Start the existing viewer in its usual environment, making the current MerXen
checkout available to its Python console:

```bash
PYTHONPATH="<current-MerXen-checkout>/src" \
  napari-compare-xenium-merscope --atera-zarr "<completed-store.zarr>" \
  --skip-transcripts
```

Open the sample **standalone in its native frame**. Load the native morphology
stains and display the Cellpose segmentation (`MOSAIK_cellpose` /
`cellpose_cell_labels`); the viewer initially shows vendor masks for Atera.
Transcript loading is unnecessary for crop selection.

Open napari's Python console and run, replacing the context path:

```python
from merxen.segmentation.hitl.viewer import add_roi_picker

picker = add_roi_picker(viewer, "roi_context.json")
```

The **HITL crop selection** dock adds independent crop and guide layers:

1. Choose the native `morphology_focus` image in the dock's image selector.
2. Pan/zoom to a location you want to correct and click **Add crop at view centre**.
   It places a fixed-size square at the nearest valid tile position.
3. Drag the square to refine its position. Keep each crop completely inside
   one cyan guide and keep crops separate. The guides preserve the full-tile
   normalization context; crossing their boundaries is refused on export.
4. Repeat for other regions. Choose representative tissue as well as difficult
   areas; selected crops remain pilot training material.
5. Click **Save crop regions…** and save a new `.geojson` file. The helper checks
   the native micron frame, square geometry, overlap and tile margins. Existing
   ROI files are preserved; choose a new filename for a revised selection.

The general cortical-depth/object annotation buttons are not used here; those
tools serve other analyses and are disabled for Atera. This dedicated helper
requires no viewer-repository changes and does not modify the SpatialData store.
Aligned, reflected, shifted and native-pixel-unit image views are refused.

To reopen a previous set in a fresh viewer session:

```python
picker = add_roi_picker(viewer, "roi_context.json", regions_path="hitl-regions.geojson")
```

### Export the selected regions as pre-filled crops

Copy the GeoJSON back to the pipeline machine. Make `selected-pilot.json`
from the same `pilot.json`, change `output_dir` to a new destination and add:

```json
"roi_geojson_path": "<saved-hitl-regions.geojson>"
```

Leave `boxes` unset. Keep the source, dataset alias, crop size, margin and
baseline-grid settings the same as when the context was prepared. Then run:

```bash
merxen cellpose-hitl-export --config selected-pilot.json
```

The file binds selections to the source and tile-grid checksums. Coordinates
are saved as GeoJSON `(x, y)` in native level-zero pixels; the helper/exporter
handle the viewer's `(y, x)` micron display without manual coordinate edits.
The resulting manifest records `viewer_rois`, stable ROI IDs, exact crop bounds
and the selection-file hash. `selected_regions.geojson` is preserved with it.

## 3. Correct cells in the separate Cellpose GUI

Prepare the drawing environment once on the annotation machine, from the
current MerXen checkout. The generated GUI lock is separate from the pipeline
lock and includes Cellpose's Qt extras:

```bash
uv venv --python 3.12 .hitl-gui-venv
uv pip install --python .hitl-gui-venv/bin/python --no-deps \
  -r requirements/requirements.hitl-gui.lock
.hitl-gui-venv/bin/python -m cellpose
```

Keep the pipeline and viewer in their existing environments. The drawing GUI
works locally on CPU; no GPU, server connection or model inference is needed
to correct the exported masks.

For either selection method:

1. Copy `annotation-pilot.tar.gz` to the annotation laptop and extract it into a
   fresh folder. Keep an untouched archive as the audit copy.
2. Open a root-level TIFF in the **Cellpose 4.2.1.1 GUI**. Its matching
   `_seg.npy` supplies the pre-filled masks. This is where cell outlines are edited.
3. Follow the package's versioned `annotation-guide.md`. Correct every visible
   cell consistently, including missing cells and false masks.
4. Press **Ctrl+S** for each crop, even if no edit was needed. Mark it `reviewed`
   in `review.tsv` only after saving; record minutes and uncertainties. Use
   `excluded` with a reason for unusable tissue.
5. Return corrected `_seg.npy` files, `review.tsv` and the original manifest.
   Keep image, raw and model-input files unchanged. Do not train in the GUI.

The exported model inputs were normalized on the complete original halo tile
before cropping; future server training must use `normalize=False`. Changing
display contrast in either GUI does not replace those arrays. Both workflows
preserve the original masks, source store and earlier annotation exports.

# Annotate crops and train a reusable model

The viewer manages local crop preparation and Cellpose annotation. MerXen
validates the returned masks, fine-tunes the pinned `cpsam_v2` checkpoint and
publishes a uniquely named **candidate** model. Training can be the first step
of a new segmentation run. The original source and stock outputs are preserved.

## 1. Prepare data and choose crops

Follow [the pilot workflow](pilot-workflows.md) to process the sample and either
choose your own rectangles or export an automatic tissue-spanning set. For
viewer-selected crops, copy these items to the annotation computer:

- The unchanged native SpatialData store containing morphology and Cellpose labels.
- `roi_context.json` **and** its companion `hitl_export_settings.json`, generated
  by the current `merxen cellpose-hitl-prepare-rois` command. Keep them together.
- Current Atera outputs already contain `images/atera_transcript_density`, its
  provenance and viewer pyramids. Copy the complete SpatialData Zarr; no second
  density store is needed. Older outputs can still use a matching separate raster.

The viewer accepts an existing pilot package's `manifest.json` as the export
settings when it sits beside `roi_context.json`. Copying a completed store does
not require copying the raw vendor directory. Avoid modifying its root metadata
between context preparation and crop selection.

In the viewer's **HITL seg crops** tab, load the setup, add rectangles and aim for
**800–1,200 estimated cells** across representative tissue and difficult areas.
That is a practical first pilot, not a guaranteed sufficient training set. Fully
correct every cell in each selected rectangle. Keep some separate regions for
later evaluation; the current pilot packages are all training material.

Click **Save areas and prepare annotation**. This creates a fresh
`hitl-<timestamp>-<id>/` folder **beside the SpatialData Zarr**. It contains all
crop TIFFs, prefilled Cellpose files, original/pre-normalized arrays, provenance,
an annotation guide and `training-template.json`. Preparation runs in the
background and locks the exported selection. Use **Start new crop selection**
for another package. **Save regions only…** remains available for remote export.

Atera uses **DAPI → 18S RNA → transcript density**, at native resolution. The
viewer automatically loads the embedded density image and HITL reads that same
image without a file picker. Its provenance, generation recipe and grid must
match the setup. For an older store without embedded density, the separate-raster
discovery/file-picker fallback remains available. Neither path regenerates
transcript density locally. The standard density recipe
uses Gaussian σ **0.15 µm**. Two-stain packages retain an empty third model slot.

To upgrade an existing server store without recounting transcripts, run
`merxen embed-transcript-density --spatialdata <review.zarr> --density
<verified-generated-density.zarr>`. This adds the image and pyramids, preserving
the morphology, masks and transcript data. Prepare fresh crop context/settings
after this metadata change, then sync the updated Zarr to the annotation computer.

Automatic or previously exported crops can enter the same managed workflow:
extract the archive and click **Open annotation package…**. The viewer copies
an older package into a new managed folder, preserving existing mask edits and
the original package. Open and save each crop once to confirm its completion.
No full SpatialData store is needed to resume annotation of an exported package.

## 2. Correct masks from the viewer

Install the separate drawing environment once from MerXen's generated GUI lock:

```bash
uv venv --python 3.12 .hitl-gui-venv
uv pip install --python .hitl-gui-venv/bin/python --no-deps \
  -r requirements/requirements.hitl-gui.lock
```

Leave the viewer in its normal environment. In **HITL seg crops**, use
**Choose Cellpose environment…** to select the drawing environment's Python
executable (`bin/python`, or `Scripts/python.exe` on Windows). This choice is
remembered. No GPU or server connection is needed for drawing.

1. Select a crop in the list and press **Annotate cell masks**.
2. Cellpose 4.2.1.1 opens that crop with its current masks. Follow the package's
   `annotation-guide.md`, correcting missing cells, merges, splits and false masks.
3. Press **Ctrl+S**, even if the prefill needed no changes, then close Cellpose.
4. Return to the viewer. A successfully saved crop turns **green**. The next
   unfinished crop is selected and the viewer prompts you to continue.
5. Repeat until the completion message appears. Record uncertainties in
   `review.tsv`'s notes column; the viewer preserves those notes.

Completion requires a successful save from the launched GUI, valid native-size
integer masks and one connected component per cell. Closing without saving or
a failed disk write leaves the crop pending. Launch through the viewer so the
save also produces the numeric masks and receipt needed by MerXen. The server
never deserializes Cellpose's pickle-based `_seg.npy` files.

The package can be reopened with **Open annotation package…**, including after
restarting the viewer and without loading the full sample. Keep one Cellpose
editing window per crop and finish/close it before reopening the same crop.
Display contrast does not change the stored training arrays.

## 3. Transfer the complete folder

The viewer's completion message shows both the local folder and its server
relative destination. Python commands read `MERXEN_HITL_ROOT` from the server's
private `.env` or exported environment. For Nextflow, export that variable or
pass `--cellpose_hitl_root` (or `hitl_root` in the training JSON):

```text
<configured HITL root>/
  annotations/
    hitl-<timestamp>-<id>/   # complete viewer folder, unchanged name
  models/
    <unique-model-name>/    # created by training
```

Transfer the **whole** package to `annotations/<package-id>/`, including
`manifest.json`, `review.tsv`, `masks/`, `receipts/`, images and model inputs.
Do not transfer only the `_seg.npy` files or rename the package. The folder's
relative destination is also recorded in its manifest. Validate it before GPU use:

```bash
merxen cellpose-hitl-import --package hitl-<timestamp>-<id>
```

This prints reviewed crop/cell counts and a training-set fingerprint. Unfinished
crops are refused. To omit an unusable crop deliberately, set its review status
to `excluded` and provide a `reason`; all other crops must be `reviewed` with
verified saves. Packages combined in one run must share their platform, channels,
pixel scale, normalization, tile grid, density recipe and annotation guide.

## 4. Train before segmentation

Use the returned `training-template.json`. Supply species and a new model name
through the run parameters (or fill the corresponding JSON fields). Model names
are 3–64 lowercase letters/digits/underscores/hyphens, starting with a letter,
for example `neuronal-pilot-v1`. A name cannot be reused for a different build.

Create a private Nextflow `hitl-run.json`; replace placeholder paths and IDs:

```json
{
  "analysis_mode": "atera",
  "species": "human",
  "segmentation_source": "cellpose_proseg",
  "start_stage": "train_cellpose",
  "stop_stage": "build_viewer_caches",
  "viewer_cache_enabled": true,
  "cellpose_hitl_root": "<absolute configured HITL root>",
  "cellpose_hitl_training_config": "<returned package>/training-template.json",
  "cellpose_hitl_model_name": "neuronal-pilot-v1",
  "cellpose_hitl_source_zarr": "<original completed source or review store.zarr>",
  "cellpose_protected_output_roots": ["<original published run root>"],
  "outdir": "<fresh HITL run root>"
}
```

```bash
nextflow run workflows/main.nf -profile dwight,conda \
  --samplesheet samples.csv -params-file hitl-run.json
```

Use the original samplesheet/pair ID. The source option copies completed inputs
into the fresh output layout, using copy-on-write reflinks where supported and
ordinary copies elsewhere. It never hard-links mutable stores. This can need
substantial space on filesystems without reflinks. Existing unrelated output
stores are refused. Atera keeps vendor nuclei; late Xenium starts instead use
`cellpose_hitl_source_outdir` pointing to the completed run, so stock nuclei
artifacts can also be copied. Paired runs may use per-platform
`<platform>_hitl_source_zarr` columns. Omit source-copy options when building a
fresh source from raw data and begin at `build_spatialdata` instead.

Only rows of the training platform that select `train_cellpose` and segmentation
use the new model. The optional stage is inactive without a training config.
Training starts from the pinned stock checkpoint, with the package's prepared
arrays and `normalize=False`. Defaults are 100 epochs, learning rate `1e-5`,
weight decay `0.1`, batch size 1 and seed 0; edit the template to change them.
Empty reviewed crops are recorded but omitted by the current positive-mask
training path. All-empty training sets are refused.

For training alone, set `only_stage` to `train_cellpose`. Alternatively:

```bash
merxen cellpose-hitl-train --config training-template.json \
  --model-name neuronal-pilot-v1 --species human --output training-report
```

The immutable bundle contains `model`, `model.json` and `train.json`, recording
weights, source corpus, recipe, runtime and training losses. Identical reruns
reuse the completed build; a different recipe/corpus under the same name fails.
The current train→segment run explicitly permits its newly trained candidate.
Held-out evaluation and model acceptance remain separate work.

## 5. Reuse the model without training

Remove the training config/name parameters, start at `segment`, keep a fresh
outdir/source-copy/protection setup and select:

```json
{
  "atera_cellpose_model": "neuronal-pilot-v1",
  "allow_candidate_model": true,
  "cellpose_hitl_root": "<absolute configured HITL root>"
}
```

Names resolve under `models/` in the configured HITL root, or the explicit
`cellpose_model_store`. Named models restore their saved channels, tile grid,
input normalization, inference/filter defaults and Atera density recipe.
Explicit platform threshold overrides still take precedence. Runtime, native
pixel scale, file checksums and protected-output checks still apply. A later
run starting after segmentation verifies the identity of the saved masks.
Use `xenium_cellpose_model` for a Xenium model. Existing full paths/hash selectors
keep their prior behavior. Training does not establish improved segmentation
quality; inspect new masks and evaluate on independently held-out tissue before
accepting a model.

> **Status:** draft (rev0), 2026-09-30. This is a plan only: nothing is implemented, run or trained. The work is an optional, later improvement with its own workflow. Current segmentation and analysis continue unchanged, and the pipeline default (stock Cellpose-SAM for both platforms) stays the default unless you change it after the acceptance tests in §9 pass. The decisions that need your input are collected in the final section; each has a recommended default.

# MerXen plan: optional human-in-the-loop (HITL) Cellpose fine-tuning for Xenium segmentation

- **Effort name:** `cellpose-hitl` ("hitl"). **Date and base:** 2026-09-30; `origin/main` `6daa255`.
- **Why:** MERSCOPE Cellpose masks are excellent, but Xenium masks are not ideal for neurons and some other cells, including cells in white matter. The Xenium 18S RNA stain varies more between cell types than MERSCOPE's PolyT. It labels somata and some processes well, but it also carries neuropil-like background and does not always give smooth boundaries. The Cellpose masks are the ProSeg prior, so these errors carry through to the final segmentation and transcript assignment for every cell (§2.1).
- **Evidence:** `$H` = `/srv/storage/MerXen/annotation_dev/hitl_plan/`. **[CP §n]** = `$H/RESEARCH_CELLPOSE.txt` section n (Cellpose 4.2.1.1 source, docs, issues and papers, with URLs). **[CODE §n / Fn]** = `$H/RESEARCH_CODE.txt` section n or headline finding Fn (MerXen code and published data, with file:line). Code references are to `6daa255` unless marked [rca] (the `feature/robust-celltype-annotation` branch).
- **Confidence tags:** [H] verified in code, logs or data; [M] documented upstream or measured but indirect; [L] design judgement or estimate, to be checked in the named milestone.
- **Abbreviations:** cpsam_v2 = Cellpose-SAM v2, the default model of Cellpose 4.2 (June 2026). GT = ground-truth (human-annotated) masks. AP@t = average precision at IoU threshold t, defined as TP / (TP + FP + FN) as in `cellpose.metrics.average_precision`. ROI = one annotated cell mask. GM / WM = grey / white matter. X / M = Xenium / MERSCOPE.

---

## 0. Summary

1. **HITL training works in current Cellpose, with caveats** [H, CP §0, §2]. Cellpose 4.2.1.1 (the latest release) can train from a folder of images and `<image>_seg.npy` files, through the GUI (Models > "Train new model with image+masks in folder", Ctrl+T), the command line or `train.train_seg`. The GUI can load pre-made `_seg.npy` masks for correction and autosaves each edit. Two caveats rule out GUI training for this work: the GUI trains only from a built-in checkpoint, and training this ViT-L model on a laptop CPU is impractical. **Plan:** the pipeline cuts crops and pre-fills them with the current masks; you correct them in the Cellpose GUI on a laptop; the pipeline then trains on the server and segments as usual. Cellpose's own FAQ describes this as "HITL without the GUI" [M, CP §5.3].
2. **Prerequisite: production and CI use different Cellpose versions and models, and outputs do not record which one ran** [H, CP §1, CODE F1–F2]. Every published mask (both platforms, cells and nuclei) was made by cpsam_v2 on cellpose 4.2.1.1, because the conda env resolves `cellpose>=3.0` (`pyproject.toml:23`). The lock pins 4.1.1 (`requirements/requirements.lock:92`), whose only model is the original cpsam. MerXen's `model_type="cyto3"` has no effect in Cellpose 4 (`src/merxen/segmentation/cellpose.py:49-53`, `docs/configuration.md:117`). Nothing in the outputs records the weights. **HITL-0** fixes this on `main` as a data-integrity change that also covers MERSCOPE, before any fine-tuning, so that the baseline in any comparison is known.
3. **Cheap baselines come first** [H, CODE F4–F6]. (B1) The Xenium images already carry an unused boundary stain, ATP1A1/CD45/E-Cadherin, that can be added as Cellpose's third channel with no code change. (B2) `cellpose_cellprob = -5.0` (`workflows/nextflow.config:19`; Cellpose's default is 0) may itself cause masks to spread into neuropil. (B3) The final area filter drops masks over 400 µm² (`config.py:47-48`), which removed 1,185 masks on P7513 Xenium, and the per-tile eccentricity > 0.99 filter penalises masks that include processes. These fixes are evaluated on the same annotated test crops as the fine-tune, so a fine-tune is adopted only if it beats them.
4. **Six stages** in a separate entry workflow (`workflows/cellpose_hitl.nf`): S1 `CROP_EXPORT` (region-stratified crops at native resolution with exactly the production model input, pre-filled masks, a manifest and a download package); S2 an annotation guide; S3 `IMPORT` (validation and a versioned, content-hashed training set); S4 `TRAIN` (fine-tune cpsam_v2, holding out whole samples, under the pipeline's GPU lock); S5 `EVALUATE` (AP per region and size class against the stock model, then downstream effects after ProSeg); S6 a content-hashed model registry and optional per-platform model parameters. When those parameters are unset, legacy config JSON and `-resume` hashes stay byte-identical.
5. **Annotation effort** [L]: about 20–23 h of your time for the pilot and two rounds, and up to about 30 h with an optional third round and the MERSCOPE fairness crops. This covers a pilot, about 25 pre-filled training and validation crops, a sealed set of 10 held-out test crops annotated from blank, and a double-annotated subset that measures the human ceiling (§4.2, §6). Cellpose 2.0 needed 100–200 ROIs per category with HITL [M, CP §4.3]. At P5011's density, a 768 px Xenium crop holds about 60 cells.
6. **Compute** [M/L]: one fine-tune takes about 8.5 GB of GPU memory at batch size 1 and minutes to tens of minutes on the A5000. Re-segmenting a Xenium section takes about 82 min of Cellpose and 2.6 h of ProSeg (P7513), plus the downstream stages. Each model is about 1.2 GB (§6).
7. **Scope** [decided by your brief]: Xenium first, human first. MERSCOPE keeps its current model unless the evaluation shows a gain there (§3.4). Mouse is covered by the design (cortex, hippocampus and midbrain strata) but waits for mouse Xenium data or your request (HITL-n). Developer effort for HITL-0 to HITL-7 is about 18–26 days [L], plus annotation time and compute.

---

## 1. Goals and non-goals

**Goals.**
- G1. Xenium cell masks that match the true soma (plus an agreed proximal-process rule) for neurons, glia in grey and white matter, and vascular cells, measured on held-out annotated crops from held-out samples.
- G2. Better ProSeg priors, measured downstream: fewer missed cells and fewer false neuropil masks, without more contamination (§4.5 S5b, §9).
- G3. A fair comparison between platforms. Any model difference is recorded in provenance and in the comparison outputs, and MERSCOPE gets the same scrutiny (§3.4).
- G4. A repeatable, versioned loop: crops, annotations, training sets and models are content-hashed and every run records which model produced which mask.
- G5. Generality. Nothing hard-codes a panel, a channel name, a species or a region list, because Xenium 5K, custom panels and mouse AP-axis sections are coming.

**Non-goals (v1).**
- No change to the default segmentation. A custom model is used only when a run sets it explicitly (§4.6).
- No fine-tuning of the nuclei model; `CELLPOSE_NUCLEI_SEGMENT` is cpsam_v2 on DAPI alone [H, CODE F1] (OD-17).
- No 3D, no multi-z Xenium input and no change to ProSeg's own algorithm. ProSeg parameters are re-checked only where the new prior changes their meaning (§4.5 S5b).
- No GUI on the server, no remote GUI mode (none exists: Cellpose issue #1424 [M, CP §5.3]) and no training in the GUI.
- No redistribution of fine-tuned weights outside the lab. They derive from CC-BY-NC training data [M, CP §4.5] (OD-19).

---

## 2. What is known

### 2.1 The problem

- 10x describes 18S as "a pan-cell type marker" and uses it only to expand outward from the nucleus. In P5011 its own segmentation used the boundary stain for about 0.3% of cells and the 18S interior method for about 85% [H, CP §6.1; CG000750].
- In FFPE human cortex, 18S "was the strongest in the gray matter". An 18S-threshold mask captured cells that were 76.7% neuronal, and microglia (small somata, highly ramified) were the class most affected by the segmentation method [M, CP §6.2; doi 10.1038/s42003-025-08518-6]. So white-matter glia need their own annotated crops, where the only evidence may be DAPI plus a thin perinuclear 18S rim.
- ProSeg does not add cells. Cells that Cellpose misses are lost, and false neuropil masks stay as cells. ProSeg regularises shape with a compactness prior (MerXen: 0.04 at 0.5 µm voxels, `workflows/nextflow.config:49-54`), and each mask pixel's vote is weighted by the Cellpose cell probability [M, CP §6.3; github.com/dcjones/proseg, `src/sampler/voxelcheckerboard.rs`]. A fine-tune therefore changes both the labels and the probability map that ProSeg uses.
- Some unassigned transcripts near neurons and glia are real RNA in processes [M, CP §6.3; bioRxiv 10.64898/2025.12.07.692889]. Soma-centred masks leave that RNA unassigned, which is a known and accepted trade-off.
- Nothing published says whether ProSeg priors should be drawn as soma only or soma plus processes. Cellpose 2.0 treats the choice as a style that must be applied consistently [M, CP §6.3; doi 10.1038/s41592-022-01663-4]. §4.2 proposes a rule and a downstream comparison against a soma-only variant derived from the same masks.

### 2.2 What MerXen runs today (facts the design depends on)

| # | Fact | Source | Consequence |
|---|---|---|---|
| K1 | Production loads cpsam_v2 on cellpose 4.2.1.1. The lock pins 4.1.1 (cpsam), and the container installs the lock. Two weight files are cached: cpsam (sha256 `e1440429…`) and cpsam_v2 (`0f1cc3f7…`) | `.command.log:87-90` of `work/58/e6e470…` (P5011 X); `pyproject.toml:23`; `requirements.lock:92`; CODE F2 | HITL-0; fine-tune from cpsam_v2 on the same version |
| K2 | `model_type` is ignored in Cellpose 4, and there is no `pretrained_model` field | `segmentation/cellpose.py:49-53`; `config.py:19`; CODE F1 | The one inference change: `CellposeModel(pretrained_model=…)` |
| K3 | A missing `pretrained_model` path falls back silently to cpsam_v2 (warning only), and weights load with `strict=False` | cellpose `models.py` L141-152, `vit.py` L31; CP §4.5 | The preflight checks the file and its sha256; a fixed-crop load check |
| K4 | The Python reuse rule returns the published mask whenever the mask, cellprobs and CSV exist, whatever model is configured. A re-run deletes and rewrites the mask in place. ProSeg reuses an existing latest zarr | `segmentation/pipeline.py:570-611`, `:996-999`; `cellpose.py:510-521`; CODE F3 | Custom-model runs go to a separate outdir and carry a model-identity sidecar check (§4.6) |
| K5 | Thresholds are the same for both platforms: flow 0.7, cellprob −5.0, diameter None, bf16 off, bsize 256. Only the channels differ per platform | `nextflow.config:16-38`; `main.nf:334-366`, `:2091-2113` | Per-platform overrides are optional and inert when unset |
| K6 | Normalisation: one joint p2/p98 clip over all channels of a 6144 px tile, conversion to uint8, zero-padding to 3 channels; then Cellpose's per-channel [1, 99] over the whole tile | `io/image_source.py:327-339`; `cellpose.py:73-80`; CODE §1.2 | Crops must carry the tile's statistics (§4.0) |
| K7 | A missing requested channel is dropped silently | `io/image_source.py:375-381` | Crop export and the preflight fail on a missing channel |
| K8 | Filters: final area 5–400 µm²; per tile, eccentricity > 0.99 and the smallest 1% of areas are removed | `config.py:44-48`; `mask_filter.py:79-167`; CODE F5 | Evaluate both raw and filtered output; bounds per model (OD-14) |
| K9 | Xenium `morphology_focus` has DAPI, ATP1A1/CD45/E-Cadherin, 18S and AlphaSMA/Vimentin; uint16; 0.2125 µm/px; identity transform. MERSCOPE `MERSCOPE_z_projection` has [PolyT, DAPI] at 0.1080 µm/px. VZG2 (mouse) has non-zero offsets | CODE §2; CP §6.1 | Crops use the full affine and name the image element; never `*_aligned_nonrigid` |
| K10 | The cortical-depth GeoJSONs hold pia, grey/white and tissue-edge lines in native microns for all 10 published human files. Every edge line is open, so no cell is ever labelled `white_matter`; in practice the white matter is `outside_brain` | CODE §3.1, F7; `cortical_depth/pipeline.py:540-588` | WM strata = full tissue polygon (`cortical_depth/tissue.py:15`) minus the ribbon (`ribbon.py:103`) |
| K11 | Mouse data are MERSCOPE only, with no manual region annotations. [rca] M6 region inference labels 150 µm tiles (`Isocortex`, `HPF`, `MB`, …) but has not been run yet | CODE §3.2 | Mouse strata are pluggable; the mouse work waits (HITL-n) |
| K12 | GPU: one RTX A5000 (24 GB). Dwight takes an flock in `beforeScript` for each GPU process; a GUI or manual run outside Nextflow does not | `workflows/conf/dwight.config:65-69`, `:86-125`; CODE §1.4 | Training and evaluation run only as Nextflow processes that take the lock |
| K13 | `source_spatialdata.zarr` in results is a symlink into `work/`, which dangles once `work/` is cleaned; ENRICH copies the native images into the durable latest zarr | CODE §2; `enrichment/enrich.py:509-523` | Crop export reads the native element from the durable zarr, by name |

### 2.3 What Cellpose supports

- **GUI files** [H, CP §3]: the GUI opens one image and cycles through its folder with A/D. An `<image>_seg.npy` beside it loads automatically; it must contain `outlines` and `masks`, and `flows` is optional. Every manual add, delete or merge rewrites the `_seg.npy`; running a model does not (use Ctrl+S). Editing is limited to single-stroke outlines, delete, merge and rectangle bulk-delete; there is no pixel paint, so fixing a boundary means delete and redraw. If the stored `filename` path is missing, the GUI uses the image with the same basename in the `_seg.npy` folder, which is what lets the crops move to a laptop.
- **Training** [H, CP §4.1–4.2]: `train.train_seg` always uses AdamW with warm-up and decay and updates all weights. The tile size must be 256 for SAM models. At most 3 channels are used. The loss treats every unlabelled pixel as background, so **annotations must be dense**. The command line drops crops with fewer than 5 masks unless `--min_train_masks 0` is set. The paper's few-shot protocol is lr 1e-5, weight decay 0.1, batch 1, 100 epochs and at least 8 images per epoch.
- **Normalisation warning** [M, CP §4.2; train.rst "Re-training a model"]: crops normalised on their own look different from full images. The GUI also stores the annotator's display settings in `normalize_params`, which training must ignore.
- **Inference** [H, CP §4.5]: `CellposeModel(pretrained_model=<path>)` goes through the same eval and tiling path, with eval bsize 256 (issue #1335).
- **Evaluation** [H, CP §5.1]: `metrics.average_precision`, `aggregated_jaccard_index` and `boundary_scores`. Loss values cannot be compared across versions (issue #1493).
- **Alternatives** [M, CP §5.4]: cellpose-napari does not work with Cellpose 4. The QuPath BIOP extension (v0.12.1) supports Cellpose-SAM, trains from densely annotated rectangles on the whole slide and supports global normalisation. It is the fallback editor (OD-3).

---

## 3. The HITL loop

### 3.1 Design

```
 server (Nextflow, cellpose_hitl.nf)                laptop (cellpose[gui], same version)
 ───────────────────────────────────                ─────────────────────────────────────
 S1 CROP_EXPORT  ── package round r ──────────────▶  open crop, correct pre-filled masks
   (strata, tile-exact input, pre-fill,               (or draw test crops from blank),
    manifest, guide, review.tsv)                      Ctrl+S on every crop, fill review.tsv
                                                              │
 S3 IMPORT  ◀────────── return package ───────────────────────┘
   (validate, versioned training set)
 S4 TRAIN   (cpsam_v2 + ALL training crops so far; GPU lock; seeds)
 S5 EVALUATE (val each round; sealed test once, on the final candidate)
 S6 REGISTER (content-hashed model) ── round r+1 pre-filled with this model
 then: main.nf run with --xenium_cellpose_model <hash> in a separate outdir → downstream checks (S5b)
```

### 3.2 GUI training or pipeline training: pipeline training (OD-4)

The GUI is the editor only. Training runs in the pipeline, from the GUI-edited masks, because:
1. The GUI trains only from a built-in checkpoint (`gui/guiparts.py` L391-411; issue #1206). This is not a real limit: the maintainers recommend retraining the built-in model on all annotations every round, and the pipeline does the same [M, CP §2].
2. GUI training on a laptop CPU is impractical for ViT-L [M, CP §5.3]. Run on the server, the GUI would bypass the GPU lock (K12) and has no display there.
3. GUI training takes normalisation from the annotator's display settings (`gui.py` L1773-1798) and uses `nimg_per_epoch = max(2, n)` rather than the paper's minimum of 8. Neither can be recorded or reproduced [H, CP §4.2].
4. Pipeline training records the base-weight sha256, the training-set hash, the split, the seed and the parameters (§4.6).

The annotator should not run models in the GUI on the pre-filled crops, because a model run replaces the masks and does not autosave (§4.2).

### 3.3 Rounds (the HITL part)

- **Round 0 (pilot, HITL-2):** 3 crops (deep GM, the GM/WM boundary and WM), fully annotated in the recommended style (§4.2), plus a short responsiveness test: open one crop at each of 512, 768 and 1024 px on your laptop and make a few edits. Output: the crop size (OD-10), measured minutes per crop, and whether the style rules are workable (OD-7).
- **Round 1:** training and validation crops pre-filled with the published baseline mask, plus the sealed test set annotated from blank (§4.1). Train, then validate.
- **Round 2:** new training crops pre-filled with the round-1 model, weighted towards the strata with the lowest validation AP and the most correction effort (`manual_changes`, §4.3). Retrain from cpsam_v2 on all training crops so far.
- **Round 3 (optional):** as round 2.
- **Stop rule** (pre-registered with §9 in HITL-3): stop when validation AP@0.5 improves by less than 0.02 over the previous round, after round 3, or when your annotation budget is spent. The test set is scored once, on the final candidate. Round and hyperparameter choices use validation only.

### 3.4 Platform scope and fairness (OD-12)

- v1 fine-tunes **Xenium only**. MERSCOPE keeps stock cpsam_v2.
- **MERSCOPE fairness check** (recommended, about 4 h of annotation): 5 MERSCOPE test crops from the paired sections, annotated from blank with the same guide and scored with the stock model. If stock MERSCOPE AP is at least as good as the fine-tuned Xenium AP, the pair comparison is fair in the sense that both platforms are near their achievable segmentation. If it is clearly lower (below the Xenium fine-tune by more than 0.05 AP@0.5, the A1 margin), a matched MERSCOPE fine-tune becomes HITL-n. MERSCOPE is re-segmented only if its own evaluation shows a gain.
- Every comparison output records each platform's model identity (§4.6). A Xenium-only lab fine-tune is a pre-processing difference that reports must state.

---

## 4. Stage design

### 4.0 Shared contracts (every stage)

- **Frame.** Everything is in the native frame of the platform being segmented: the s0 pixel grid of the named image element (`morphology_focus` or `MERSCOPE_z_projection`), which matches `cellpose_masks_tiled.npy`. Conversion between pixels and microns always uses the full affine from `cellpose_transforms.json` (`build_cellpose_affine_to_microns`, `cellpose.py:698-727`), never µm divided by pixel size (VZG2 offsets, K9). Aligned elements are refused [H, CODE §2].
- **Model input.** The crop is the exact array the production model saw at those pixels. The crop lies wholly inside one tile core of the deterministic grid (`iter_core_tiles`, `cellpose.py:141-209`), using the tile size recorded in `cellpose_stitching_stats.json` (6144 on P7513, 90 tiles) and a 256 px overlap. Its values are computed from the enclosing halo tile: the joint p2/p98 clip and uint8 step of `prepare_cellpose_input` (`image_source.py:327-339`), then Cellpose's per-channel [1, 99] rescale using that whole tile's percentiles. Cellpose 4.2.1.1 normalises the whole image passed to `eval` before its internal 256 px tiling (`cellpose/models.py` L271-293, read in the pipeline's installed copy) [H]. The result is stored as float32. Training uses `normalize=False`, so training input equals inference input (route A, OD-5) [M, CP §4.2, CODE §6.3]. HITL-2 tests this equality (§8).
- **Raw copy.** The raw uint16 crop of *all* channels and the tile statistics are stored too, so a different channel set or normalisation can be derived later without re-annotating. Masks do not depend on channels.
- **Channels.** Selected by name in the seg-config order. A missing channel is an error (K7). The default training set is `[DAPI, 18S, 0]`; variant B1 is `[DAPI, 18S, ATP1A1/CD45/E-Cadherin]` (OD-6).
- **Scale.** Native resolution only (`factor_rescale` must be 1.0, `config.py:29-38`). Large human neuron somata are about 70–140 px at 0.2125 µm/px, at or above cpsam's 7.5–120 px training range; fine-tuning at native scale teaches the model that scale [M, CP §5.2] (OD-13).
- **Tile-size stability.** `choose_working_tile_size` keeps the largest candidate that does not run out of GPU memory on a probe tile (`cellpose.py:212-286`) [H]. A run on a busier GPU could pick a smaller tile and change the normalisation statistics. HITL runs therefore pin `cellpose_tile_size_candidates = [<recorded size>]` and refuse to run if that size does not fit (R6).

### 4.1 S1 `CROP_EXPORT` (CPU; `merxen cellpose-hitl-export`)

**Inputs:** the row's seg config (the same `_load_dataset_sdata` call, channels and transform as `CELLPOSE_SEGMENT`); the durable latest zarr's native image element (K13); the persistent mask and cellprobs (memory-mapped); the nuclei mask; the stratum source; optionally the cells parquet (cortical depth and broad class), used only to weight sampling, with its hash recorded.

**Strata** (a pluggable `StratumSource` built from manual annotations; no hard-coded region list, G5):

| Species | Stratum | Geometry (native µm) | Source |
|---|---|---|---|
| Human | L1 / pia band | ribbon, `laplace_depth` < 0.10 | cortical-depth GeoJSON → ribbon (`ribbon.py:103`) and depth field |
| Human | Upper GM | depth 0.10–0.45 | same |
| Human | Deep GM | depth 0.45–1.0 | same (P7513 GM depth quartiles .11 / .34 / .63, CODE §3.1) |
| Human | GM/WM boundary | within 150 µm of the grey/white line, either side | GeoJSON grey/white line |
| Human | WM | full tissue polygon minus ribbon, > 150 µm from the grey/white line | `build_full_tissue_polygon` (`tissue.py:15-69`, already used by VALIS) minus the ribbon (K10) |
| Mouse | Cortex, hippocampus, midbrain | `Isocortex`, `HPF`, `MB` | manual region polygons (a new GeoJSON role `region` with a `name`, which needs a loader alias, `cortical_depth/boundaries.py:22-38`) or [rca] M6 `tile_region` (OD-15) |

Strata are disjoint, and the boundary band takes precedence over deep GM and WM. The depth cut-points are [L] and are chosen so that each band holds recognisable laminar morphology. They are recorded in the manifest.

**Candidate windows and sampling.**
1. Candidate windows lie on a grid inside each stratum. A window is kept only if it lies wholly within one tile core, is at least 90% inside the stratum and the tissue polygon, and does not overlap an earlier crop. Test crops keep a buffer of one crop width from training and validation crops.
2. Label-free window features: nuclei count (from the nuclei mask, which does not depend on the cell model); baseline cell count; **nuclei without a cell** (nuclei with less than 50% overlap with any cell mask, a missed-cell signal); **anucleate cell area share** (a false-neuropil signal); an 18S neuropil index (18S signal outside cell masks); and the share of saturated or empty pixels (artefact screen).
3. **Density-aware:** within each stratum, one crop per density tercile (from nuclei count) before any repeats.
4. **Cell-type-aware:** where published labels exist, a greedy diversity step makes the round's crops in each stratum contain at least 30 cells of every broad class present there, including oligodendrocytes and microglia in WM and large pyramidal neurons in the upper and deep GM bands. Labels only weight the choice; they never enter the GT.
5. **Hard examples:** each round adds one crop per stratum from the top decile of "nuclei without a cell" or the neuropil index, aimed at the failure modes you described.
6. The RNG seed, candidate counts and the features of every chosen window are written to the manifest.

**Crop size** [L; the pilot confirms, OD-10]: set in microns, converted per platform, at least 256 px (the training bsize) and at most 1,024 px for GUI responsiveness. Every edit rebuilds a full-image overlay and rewrites the `_seg.npy` (`gui.py` L1699-1730) [H, CP §5.2]. Defaults: Xenium 768 px (163 µm, about 60 cells at P5011's mean of about 2,250 cells/mm²); MERSCOPE 1,024 px (111 µm).

**Pre-fill** (OD-9):
- Training and validation crops in round 1 get the published final mask crop (`cellpose_masks_tiled.npy[y0:y1, x0:x1]`, relabelled 1..n), which is exactly the ProSeg prior. In later rounds they get the latest model's prediction, made in the production tile context (§4.5).
- **Test crops are annotated from blank**, so the GT is not anchored to either model (R2).
- The `_seg.npy` is minimal: `masks`, `outlines` (`masks * utils.masks_to_outlines(masks)`), `filename` (basename only) and `ismanual = zeros(n)`. It has no flows, which keeps autosave fast [H, CP §3.3].
- Optionally, when the export is given a raw baseline prediction for the tile (made before the area filter, on the GPU in HITL-5), the masks that the 400 µm² filter removed are shown as an overlay PNG for reference. They are not pre-filled.

**Files per crop** (`<crop_id>` = `<pair>_<platform>_<stratum>_r<round>_<nn>`):
- `<crop_id>.tif`: the GUI view image, (3, Y, X) float32. The default view is [DAPI, 18S, ATP1A1/CD45/E-Cadherin], so the annotator sees the boundary stain; the GUI shows up to 3 channels.
- `<crop_id>_seg.npy`: the pre-fill (or empty masks for test crops).
- `<crop_id>_context.png`: a downsampled view of the section (pyramid level 4) with the crop box, for orientation.
- Server side only: `raw/<crop_id>.npy` (uint16, all channels) and `model_input/<crop_id>.<channelset>.npy` (float32, route A).

**Manifest** (`manifest.tsv` + `manifest.json`), one row per crop: crop_id; round; split (train / val / test, assigned at export); pair, sample, platform, species, panel ID; stratum and its source file sha256; bbox in native pixels (y0, y1, x0, x1) and in µm (from the affine); pixel size and the full affine; image element name; channel map (names → source indices, view order, training order); tile ID, tile bbox, joint p2/p98 and per-channel p1/p99 (the normalisation contract); source mask path and crop sha256; source model name and weights sha256; cellpose and torch versions; sampler seed and window features; image and raw-crop sha256; annotation-guide version and style.

**Package:** `hitl_<platform>_r<round>.tar` holding crops, `_seg.npy`, context PNGs, `manifest.tsv`, `review.tsv` (crop_id, status reviewed / excluded, minutes, notes), the annotation guide, `SHA256SUMS` and the laptop env file (§7). About 40 crops at roughly 8 MB each make about 0.3 GB [L]. Exports are written under `<hitl_root>/rounds/<round>/export/`, outside every Nextflow `publishDir` (modules publish with `overwrite: true`, `segmentation.nf:4`).

### 4.2 S2 Annotation guide (`docs/hitl/annotation-guide.md`, versioned; the manifest records its version)

**Rules.**
1. **Dense.** Every cell whose visible soma or nucleus lies in the crop is drawn, including cells cut by the crop edge (draw the visible part). An undrawn cell is learned as background [H, CP §4.2].
2. **What counts as a cell.** A DAPI nucleus with or without an 18S rim. An **anucleate soma** (sectioned above or below its nucleus) is drawn only when its 18S outline is compact, clearly bounded and at least the size of a small nucleus (about 5 µm across); otherwise it is left undrawn (OD-8) [L].
3. **Neurons: soma vs processes** (OD-7; recommended default, confirmed downstream in HITL-6): draw the soma plus the continuous proximal dendrite trunk, only where the 18S signal is unbroken from the soma, up to about one soma diameter. Do not draw distal or isolated processes or neuropil 18S. Reasons: 2D masks must be single strokes; processes cross other cells' territory and would seed ProSeg with voxels from other cells; ProSeg's compactness prior pulls thin appendages back anyway; ProSeg can still grow a soma prior where the transcripts support it [M/L, CP §6.3]. **No second annotation pass is needed to compare styles** [L]: a soma-only variant is derived from the same masks by morphological opening (a disk of about 2 µm radius, which removes thin trunks and keeps somata). One model is trained per variant; each is scored against GT derived the same way, so image AP does not by itself pick a style. The choice is made on the downstream metrics of §4.5 S5b for a training sample (P1212), never on the test set, and the test set is scored only after it. The reverse derivation is impossible, which is why the drawn style is the richer one.
4. **White-matter and other small cells:** outline the DAPI nucleus plus any 18S rim contiguous with it; do not invent cytoplasm. Draw each nucleus of oligodendrocyte rows separately. Draw microglia as the nucleus plus the visible perinuclear soma, without branches. Draw endothelial cells and pericytes as each elongated nucleus plus its rim, never the vessel lumen or wall.
5. **Not cells:** lipofuscin or autofluorescent granules outside a soma (inside a soma they stay part of it), red blood cells, debris, edge-of-tissue smears, and out-of-focus blur with no nucleus.
6. **Ambiguous cases:** split touching nuclei along the DAPI boundary. If you cannot decide within about 10 s, use the "draw only if clear" rule and note the crop in `review.tsv`. Mark a crop `excluded` if more than about 20% of it is a fold, tear, bubble or saturated area.
7. **GUI mechanics:** one stroke per mask; no overlaps (the GUI crops new masks); to fix a boundary, delete and redraw; press **Ctrl+S on every crop**, even with no edits, so that the saved keys show it was reviewed (§4.3); do not run models in the GUI; fill in `review.tsv` (status, minutes).

**Time and targets** [L; the pilot measures them]:
- A pre-filled 768 px crop takes about 15–25 min to correct.
- A blank test crop takes about 40–60 min.
- Targets for round 1: at least 600 training ROIs (at least 100 per stratum) plus about 5 validation crops, and at least 600 test ROIs over two held-out samples (at least 100 per stratum).
- Later rounds add about 600 ROIs each.

### 4.3 S3 `IMPORT` and validation (CPU; `merxen cellpose-hitl-import`)

**Checks** (each failure is listed per crop; nothing is fixed silently):
1. Every file's crop_id and image sha256 match the manifest; a changed image is refused.
2. The `_seg.npy` is loaded with `allow_pickle=True`, which is acceptable only for packages you return yourself (R13). It must hold `masks` of the crop shape and an integer dtype.
3. Each label is one 8-connected component: a split label goes back to the annotator, and a component smaller than 20 px is flagged.
4. Masks and `outlines` agree.
5. **Reviewed:** the GUI-save keys (`manual_changes`, `model_path`, `normalize_params`) are present, or `review.tsv` says `reviewed`. Unreviewed crops are refused.
6. Crops marked `excluded` are dropped and logged.
7. Test crops arrive in a separate package, and their masks are not already present in any training set.

**Diagnostics:** per-crop correction effort (`manual_changes` count, the share of ROIs that are `ismanual`, the minutes from `review.tsv`); agreement between GT and pre-fill (AP@0.5), which is an anchoring diagnostic (R2); ROI counts per stratum and per size class.

**Versioned training set:** `<hitl_root>/trainsets/<hash>/` holds `masks/<crop_id>.tif` (uint16 or uint32), `model_input/…` for each channel set, `split.tsv`, `manifest.json` and the returned `_seg.npy` files archived unchanged. The hash is sha256 over canonical JSON of the per-crop image and mask sha256s, the splits, the guide version and style, and the normalisation contract. The set is built in `.tmp-<uuid>`, renamed into place under flock, made read-only and never deleted, following the [rca] `annotation/store.py` pattern [H, CODE §4.3]. GUI `normalize_params` are ignored. Test masks sit in a sibling `testsets/<hash>/`, which the training code refuses to read.

### 4.4 S4 `TRAIN` (GPU; `CELLPOSE_HITL_TRAIN`; `merxen cellpose-hitl-train`)

**Split** (OD-11):
- By sample: whole samples are held out. Recommended: train on P7513 and P1212; test on **P7113 and P5011**, the same held-out donors as the rca plan, so later annotation acceptance stays non-circular.
- By crop: validation crops come from the training samples and keep a one-crop-width spatial buffer from training crops.
- Test crops are never used for any choice.

**Base and version:** cpsam_v2 from the pinned cellpose (HITL-0). The base weights sha256 must equal `0f1cc3f7…` (K1), or training refuses to start.

**Parameters** (the paper protocol [M, CP §4.2]; explicit, recorded):

| Parameter | Value | Why |
|---|---|---|
| `learning_rate`, `weight_decay`, optimiser | 1e-5, 0.1, AdamW (always) | paper, issue #1346 |
| `n_epochs` | 100 (grid {100, 300} on validation only) | paper; longer schedules decay differently (train.py L412-421) |
| `batch_size` | 1 | paper; memory |
| `nimg_per_epoch` | max(8, n_train) | paper minimum of 8, rather than the GUI's max(2, n) |
| `min_train_masks` | 0 | keeps sparse WM and neuropil crops (the command-line default of 5 drops them) |
| `bsize` | 256 | required for SAM (train.py L360-363) |
| `normalize` | `{"normalize": False}` (route A), passed as a dict, never a bool | the input is already tile-normalised; a bool mutates the module-level default dict (train.py L378-379) |
| `rescale`, `scale_range` | False, 0.5 | native scale; augmentation of ±25% |
| channels | set by the model's channel config; `channel_axis` explicit | at most 3 channels (train.py L74) |
| precision | float32; inference as in production (`use_bfloat16 = false`) | `nextflow.config:38` |
| seeds | 0, 1, 2 (torch, numpy, python); cuDNN non-determinism recorded | seed spread is an acceptance item (§9) |

**Model choice:** the median-validation-AP seed of the best validation configuration. The grid is at most {100, 300} epochs × {1e-5, 5e-6} learning rate, so there are few ways to overfit the validation set. The grid runs only for the default channel set and drawn style; the variants (channel set B1, derived soma-only style) reuse the chosen settings with 3 seeds each, in the final round only.

**GPU scheduling:**
- `CELLPOSE_HITL_TRAIN` and `CELLPOSE_HITL_EVAL` carry the same Dwight flock `beforeScript` block as `CELLPOSE_SEGMENT` (`dwight.config:86-105`), with `maxForks 1`, so they queue behind pipeline GPU tasks and never share the A5000.
- Batch-1 training needs about 8.5 GB [M, CP §4.4]. The process checks free GPU memory after taking the lock and fails early rather than running out of memory.
- No training outside Nextflow (K12).

**Outputs:** the model file (about 1.2 GB), train/validation loss arrays (never compared across cellpose versions, issue #1493), a `train.json` with every parameter, the timings and the peak GPU memory.

### 4.5 S5 `EVALUATE`

**S5a: image level** (`CELLPOSE_HITL_EVAL`, GPU).
- **Production context.** For each test or validation crop, the candidate model runs through `run_tiled_cellpose` on the crop's enclosing tile, with the production thresholds, per-tile filters, stitching and final area filter. The result is then cropped. This reproduces exactly what ProSeg would receive; one tile takes about 55 s (82 min for 90 tiles on P7513).
- **Baseline.** The stock baseline is the published mask crop. HITL-5 first re-runs stock cpsam_v2 on those tiles and requires IoU-matched agreement of at least 0.99 with the published crop, which confirms the version and normalisation are reproduced (R11).
- **Raw scores.** Raw model output, before the area and eccentricity filters, is scored as well, so filter losses are visible (K8).
- **Quick path.** During rounds, a quicker crop-only evaluation on `model_input` arrays scores the validation set.

**Metrics:**
- AP@0.5 and AP@0.75 (AP@0.9 reported), plus precision, recall and F1 at 0.5, pooled (TP/FP/FN summed over crops) and per crop.
- Aggregated Jaccard index and boundary scores [H, CP §5.1].
- **Per region:** each stratum.
- **Per cell class:** (i) label-free GT size classes (< 40, 40–150 and > 150 µm² [L]) as a neuron/glia proxy; (ii) diagnostically, the broad class of the matched published cell (IoU ≥ 0.5; else "unmatched"). Class (ii) inherits the current segmentation's labels, so it is reported but never gated.
- **Failure-mode counts:** FP per mm² in WM and neuropil-index-high crops; recall of GT cells > 150 µm²; missed GT cells that have a nucleus.
- **Uncertainty:** 95% CIs from a paired bootstrap over crops for every model difference.
- **Human ceiling:** 3 test crops re-annotated from blank by you at least 2 weeks later (or by a second annotator, OD-18). AP between the two annotations is the ceiling and is reported beside each model [M, CP §5.1; Cellpose FAQ].
- **Comparators** on the same crops: stock cpsam_v2 (baseline); B1 (third channel ATP1A1/CD45/E-Cadherin, stock model); B2 (cellprob −5 vs 0 vs −2, stock model); B3 (area bound 400 vs 800 µm², eccentricity 0.99 vs off); the fine-tune with each channel set; and the fine-tune with retuned thresholds (flow and cellprob tuned on validation only).

**S5b: downstream after ProSeg** (a normal `main.nf` run with the candidate model in a separate outdir, §4.6; held-out pairs P7113 and P5011, plus P7513 for information). Each metric is compared with the baseline outdir:

| Metric | Direction | Source |
|---|---|---|
| Share of transcripts assigned to cells after ProSeg; Cellpose seeding rate (P7513 X baseline 62.80%) | report; a move of more than 5 points triggers review | ProSeg output; `pipeline.py:686-755` seeding |
| Share of **nuclear transcripts** assigned (transcripts inside the unchanged DAPI nuclei mask that end up in a cell) | should rise | nuclei mask + ProSeg assignment |
| Share of cells overlapping ≥ 2 nuclei (doublet proxy) and cells with 0 nuclei (neuropil / fragment proxy) | should fall | nuclei mask |
| MECR and mutually exclusive marker co-expression | must not worsen | existing MECR stage (`mecr_enabled`) [M, CP §5.1; arXiv 2606.09675] |
| Cells per mm² per stratum; area distribution; masks lost to the area filter | report | segmentation stats |
| Annotation coverage and marker purity: confident broad coverage, canonical-marker plausibility, WHB–SEA agreement ([rca] H7, H9, H3 methods, or legacy equivalents if rca is not on `main`) | must not worsen | annotation pipeline |
| Cross-platform concordance with MERSCOPE: soft broad-composition JSD ([rca] H1 estimator), per-stratum density ratio, class depth profiles ([rca] H12) | must not worsen; improvement reported | COMPARE, cortical depth, rca |
| WM oligodendrocyte density | must not fall by more than 5% | annotation + strata |

VALIS registration does not change when only Xenium (the fixed section) is re-segmented [H, CODE §5.2].

### 4.6 S6 Model registry and pipeline parameters

**Registry** (`cellpose_model_store`, default `/srv/storage/MerXen/cellpose_models/`, OD-16):
- Layout: `<store>/<build_hash>/{model, model.json, train.json, eval/}`, built in `.tmp-<uuid>`, renamed under flock, read-only and never deleted (the [rca] store pattern; reuse `merxen.annotation.store` if it is on `main` by then, otherwise a minimal copy to unify later).
- `build_hash` = sha256 of canonical JSON over the base-weights sha256, the cellpose and torch versions, the training-set hash, the split, the training parameters, the seed, the normalisation contract and the channel list [H pattern, CODE §4.3].
- `model.json` records: platform, species, channel names in order, pixel size, tile size, normalisation route, annotation style and guide version, the thresholds and filters used in its evaluation, the model file sha256, and the status (`candidate` / `accepted` / `rejected`, with the evaluation summary path).

**Pipeline parameters** (inert when unset, so legacy config JSON stays byte-identical and `-resume` keeps working, following the [rca] `AnnotationSettings.groovy:4-11` pattern):
- `CellposeConfig.pretrained_model: str | None` (Python default `cpsam_v2` from HITL-0) and `model_sha256: str | None`, passed through `build_cellpose_model` (`cellpose.py:34-53`). The Nextflow config JSON carries these keys only when a run sets them.
- `params.xenium_cellpose_model` and `params.merscope_cellpose_model` take a registry hash or path, with optional samplesheet columns of the same names for per-row overrides. They are merged into `baseConfig.cellpose` (`main.nf:2091-2101`) only when set.
- Optional per-platform overrides, also inert when unset: `<platform>_cellpose_cellprob`, `<platform>_cellpose_flow_threshold`, `<platform>_cellpose_final_max_area_um2`, `<platform>_mask_max_eccentricity`. They are explicit parameters, never read automatically from `model.json`; the preflight warns when they differ from the values the model was evaluated with.

**Preflight** (refuses to run): the model file exists; its sha256 matches `model.json`; the row's channels, pixel size (±1%), platform and species match `model.json`; the model status is `accepted`, or `candidate` with `--allow_candidate_model`; the tile-size candidates are pinned (§4.0). This closes K3's silent fallback.

**Reuse guard and baseline safety** (K4):
- A `cellpose_model_identity.json` sidecar (model name, sha256, effective CellposeConfig) is written beside the persistent mask. The reuse rule (`pipeline.py:570-611`) compares it the way the transforms sidecar is compared (`pipeline.py:249-358`). On a mismatch it refuses to run; it never overwrites.
- v1 requires a **separate outdir** for any custom-model run (route 5.3a [H, CODE §5.3]). The MERSCOPE side is *copied* into the new outdir (persistent segmentation and latest zarr, never symlinked), so MERSCOPE is not re-segmented, no downstream stage mutates the baseline, and the paired analyses have both platforms.

**Provenance** (HITL-0 for the stock model, HITL-1 for custom models): the cellpose and torch versions, base-weights sha256, custom `build_hash` and effective CellposeConfig go into the sidecar and into the segmentation stats JSON. The cellpose and torch versions are also added to `merxen_schema` `writer_versions` (`io/spatialdata_schema.py:100`) [H, CODE §4.4].

---

## 5. Pipeline integration

- **New files:** `workflows/cellpose_hitl.nf` (entry workflow, run with `--hitl_step export | import | train | evaluate`); `workflows/modules/cellpose_hitl.nf` (`HITL_CROP_EXPORT`, `HITL_IMPORT`, `CELLPOSE_HITL_TRAIN`, `CELLPOSE_HITL_EVAL`, `HITL_REPORT`); `src/merxen/segmentation/hitl/{strata,crops,export,seg_npy,ingest,trainset,train,evaluate,registry,report}.py`; `src/merxen/cli/run_cellpose_hitl.py` (`merxen cellpose-hitl-*`); `docs/stages/cellpose-hitl.md`; `docs/hitl/annotation-guide.md`; `envs/environment.cellpose-gui.yml` (laptop only, §7).
- **Shared files are changed only at marked hook points** (`// hitl-hook:C<n>` / `# hitl-hook:C<n>`, each asserted to occur exactly once, as in [rca] `test_rca_hooks.py`): C1 `CellposeConfig` fields (`config.py:16-27`); C2 `build_cellpose_model` (`cellpose.py:49-53`); C3 the reuse guard (`pipeline.py:570-611`); C4 `baseConfig.cellpose` and the per-platform overrides (`main.nf:2091-2113`, `:334-366`); C5 the preflight; C6 the Dwight lock blocks for the two new GPU processes (`dwight.config`). The rca integration branch also edits `main.nf`; C4 and C5 sit away from its H2–H5 hooks, and `main` is merged into rca as usual.
- **Pins:** a test pins the sha256 of the resolved `CELLPOSE_SEGMENT`, `CELLPOSE_NUCLEI_SEGMENT` and `PROSEG_SEGMENT` script text, and another asserts that a legacy row's segment config JSON is byte-identical before and after each milestone.
- **Metro map:** the new processes live in a separate optional entry workflow, so they go into `OMITTED_PROCESSES` in `tests/test_workflows/test_metro_map.py` with that reason, unless you would rather have a side line on the map.
- **Mouse and new panels:** strata come from a `StratumSource` interface (cortical-depth GeoJSON, region-polygon GeoJSON, [rca] tile regions). Channel names and species come from the row and `model.json`. A new stain set or panel whose channel names differ is refused until it has its own evaluation (G5).

---

## 6. Compute, storage and time

| Item | Resource | Estimate | Basis |
|---|---|---|---|
| Crop export, per sample (~20 crops) | CPU; reads ~20 halo tiles (6144² × 4 ch uint16, ~300 MB each) | minutes to under 1 h | [L] |
| Pilot (round 0) | you | 2–3 h | [L] |
| Round 1 annotation | you | train and val ~6 h; blank test ~8 h; double-annotation ~2 h; optional MERSCOPE fairness ~4 h | [L] |
| Rounds 2–3 | you | ~3–4 h each | [L] |
| One training run | A5000, ~8.5 GB, batch 1 | ~2,000 iterations: roughly 10–20 min | docs benchmark, 200 iterations in 27 s (A100) to 364 s (RTX 4060) [M, CP §4.4]; A5000 not benchmarked [L] |
| Training per round | 3 seeds × ≤ 4 grid points; final round adds ~3 variants × 3 seeds | ~2–4 h GPU per round, plus ~1.5–3 h in the final round | [L] |
| Evaluation in production context | ~55 s per 6144 tile | ~20 min per model for ~20 test and validation tiles | P7513 trace 82 min / 90 tiles [H] |
| Downstream, per pair | Xenium Cellpose ~82 min GPU; ProSeg ~2.6 h (32 threads); then enrichment and downstream stages | ~0.5–1 day wall per pair; HITL-6 needs 4–5 runs (P1212 × 2 styles, P7113, P5011, optionally P7513) | CODE §5.4 [H] + [L] |
| MERSCOPE re-segmentation (only if HITL-n) | ~6.7 h Cellpose per section | – | CODE §5.4 [H] |
| Storage | model ~1.2 GB each; crops ~10 MB each; a copied-pair outdir up to a few hundred GB (MERSCOPE mask alone 50 GB) | measured in HITL-6 before the first run; `/srv/storage` had 3.2 TB free on 2026-09-30 | CODE §2 [H]; df |

**Developer effort** [L, about 30% contingency]: HITL-0 1–1.5 days; HITL-1 2–3; HITL-2 4–5; HITL-3 2–3; HITL-4 3–4; HITL-5 3–4; HITL-6 2–3 plus compute; HITL-7 1–2. Total about 18–26 days.

---

## 7. Laptop setup (for the annotator)

- `envs/environment.cellpose-gui.yml` pins `cellpose[gui]` to the **same version** as the lock (4.2.1.1 after HITL-0) and the same numpy major version, because `_seg.npy` is a pickled dict and must round-trip; this is a precaution, not a documented issue [L, CP §5.3]. A test (like `test_env_lock_sync.py`) asserts that the pin equals the lock's cellpose version.
- This deviates from the pyproject-only rule (Agents.md) on purpose: the laptop env installs only the GUI, not MerXen. The alternative is a `[hitl-gui]` extra, which would pull all of MerXen's dependencies onto the laptop (OD-20).
- The GUI runs on CPU; no GPU is needed for editing. Unpack the package, open any crop and cycle with A/D.

---

## 8. Milestones (PR-sized; each merges with defaults unchanged)

Branches: short-lived `feature/hitl-<n>-<slug>` from `main`, one PR each into `main` (OD-21). No integration branch is needed, because every milestone is inert until a run sets a model. Commit prefixes follow Agents.md. Before each commit run `ruff check . --fix`, `ruff format .` and `mypy src/`; before each push run `pytest` (the pre-push hook). Tests that need real data or a GPU are marked `@pytest.mark.slow`.

### HITL-0 `[bugfix]`: pin Cellpose, name the model, record it (on `main`; recommended now, independent of HITL; 1–1.5 days)
- **Scope:** `pyproject.toml` `cellpose>=4.2.1.1,<4.3` (OD-2), lock regenerated to 4.2.1.1, and the env lock-hash header refreshed (`scripts/update_env_lock_hash.py`). Add `CellposeConfig.pretrained_model = "cpsam_v2"`; `model_type` is kept, deprecated and warned about. The nuclei config names its model the same way. Record the model name, the weights sha256 and the cellpose and torch versions in the stitching stats and in `writer_versions`. Fix `docs/configuration.md:117`. Side fix: the `gpu` profile lacks `--nv` for `CELLPOSE_NUCLEI_SEGMENT` (`nextflow.config:629-646`).
- **Cost:** a new conda env hash invalidates `-resume` under the conda profile, so schedule it between runs. Published masks do not change, because production already runs 4.2.1.1 with cpsam_v2 (K1).
- **Tests:** the config accepts `pretrained_model`, and `model_type` alone warns; `build_cellpose_model` passes `pretrained_model` (with a monkeypatched `CellposeModel`); the provenance fields are written; env/lock sync.
- **Exit:** a fresh env reports cellpose 4.2.1.1 and loads cpsam_v2 with sha256 `0f1cc3f7…`; CI uses the same version; a re-run on one P7513 Xenium tile matches the published mask crop (IoU-matched ≥ 0.99).

### HITL-1 `[feature]`: inert model plumbing (2–3 days)
- **Scope:** hooks C1–C5; the model and override parameters (§4.6); the preflight; the model-identity sidecar and reuse guard; the separate-outdir rule; the script and config pins; docs.
- **Tests:** legacy config JSON byte-identical; a missing model file, wrong sha256, wrong channels or wrong pixel size each refuse; a sidecar mismatch refuses without deleting anything; each override is applied only when set; hook markers occur once.
- **Exit:** full `pytest` green; the P7513 dry-run config is unchanged.

### HITL-2 `[feature]`: S1 crop export, S2 guide, pilot package (4–5 days + your pilot time)
- **Scope:** `StratumSource` for cortical-depth GeoJSON (human) and region polygons; window features; sampling; the normalisation contract; the `_seg.npy` writer; the manifest; the package; `HITL_CROP_EXPORT`; the annotation guide; the laptop env.
- **Tests:** synthetic image + GeoJSON → strata polygons and windows inside one tile core; **the exported model input equals the array production feeds the model** for the same pixels (float tolerance), on a synthetic tile; non-zero-offset affine (VZG2-like) round trip; a missing channel raises an error; `_seg.npy` keys, and a load through cellpose's own `io` path (`--mask_filter _seg.npy` reads `masks`); frame check on real data (slow): the mask labels under each transcript in the crop equal that transcript's `cell_id` in `transcripts_for_proseg.csv`.
- **Pilot:** 3 crops on P7513 Xenium plus the 512 / 768 / 1024 px responsiveness test (§3.3). Also a test that the soma-only derivation (opening) removes a synthetic trunk and keeps the soma.
- **Exit:** you have annotated the pilot; the style rules (OD-7), crop size (OD-10) and minutes per crop are recorded in the guide.

### HITL-3 `[feature]`: S3 import and versioned training sets; pre-registration (2–3 days)
- **Scope:** `HITL_IMPORT`; checks and diagnostics (§4.3); training and test set stores; `docs/acceptance/cellpose-hitl-preregistration.md`, which copies §9 with your approved thresholds **before any training**.
- **Tests:** each validation failure (changed sha256, split label, wrong shape, unreviewed crop, excluded crop, test crop in a training set); hash stability and sensitivity; a crash leaves no partial set; concurrent builds build once.
- **Exit:** round-1 package exported, annotated and imported; the pre-registration merged.

### HITL-4 `[feature]`: S4 training and S6 registry (3–4 days + GPU)
- **Scope:** `CELLPOSE_HITL_TRAIN` with the Dwight lock block (hook C6); `merxen cellpose-hitl-train`; the registry; `model.json`.
- **Tests:** a monkeypatched `train_seg` receives the §4.4 parameters (explicit normalize dict, `min_train_masks=0`, bsize 256); the base-sha256 refusal; registry hash, atomic build and read-only mode; a workflow string test that the lock block is present and `maxForks` is 1; slow GPU smoke test: 2 crops, 2 epochs.
- **Exit:** a round-1 candidate is registered with its training record; peak GPU memory and wall time are measured and added to §6.

### HITL-5 `[feature]`: S5a image-level evaluation and report (3–4 days + GPU)
- **Scope:** `CELLPOSE_HITL_EVAL` (production-context and quick paths); comparators B1–B3; metrics, bootstrap CIs, size classes, the human ceiling; `HITL_REPORT` (HTML with per-stratum overlays of GT, baseline and candidate).
- **Tests:** AP on hand-made mask pairs (known TP/FP/FN); the paired bootstrap on a fixture; a baseline reproduction check that fails on a mismatch.
- **Exit:** rounds 1–2 (and 3 if the stop rule allows) are done; one finalist per style variant is chosen on validation; the test set stays sealed until HITL-6 has chosen the style.

### HITL-6 `[feature]`: S5b downstream evaluation, style choice, test scoring (2–3 days + ~1 day compute per pair)
- **Scope:** the separate-outdir run recipe (the MERSCOPE side copied), downstream metric scripts (`scripts/acceptance/hitl_downstream.py`), and the report section. Order: (1) both style finalists run downstream on P1212 (a training sample) and the style is chosen from the S5b metrics; (2) the chosen finalist is scored once on the sealed test set (A-criteria); (3) it runs downstream on P7113 and P5011 (D-criteria).
- **Tests:** the metrics on a synthetic small SpatialData; the outdir preflight refuses the baseline outdir.
- **Exit:** the A- and D-criteria of §9 are evaluated; the baseline outdirs are unchanged (sha256 of the persistent masks and latest zarr metadata before and after).

### HITL-7 `[docs]` / `[minor]`: acceptance decision and opt-in use (1–2 days)
- **Scope:** mark the model `accepted` or `rejected` in the registry; document the opt-in (`--xenium_cellpose_model <hash>`) in `docs/stages/segmentation.md` and `docs/configuration.md`; record the platform-model difference in COMPARE outputs.
- **Default:** stays stock unless you decide otherwise in a separate `[feature]` PR.
- **Exit:** you have signed off the acceptance summary.

### HITL-n (later, each its own PR set)
Mouse strata and a mouse Xenium model when mouse Xenium data arrive (cortex, hippocampus, midbrain; manual region polygons or [rca] M6); a matched MERSCOPE fine-tune if the fairness check (§3.4) calls for it; Xenium 5K / custom-panel evaluation; the QuPath route (OD-3); a second annotator; a nuclei fine-tune.

---

## 9. Acceptance criteria (pre-registered in HITL-3, before any training)

**Rules.**
- Thresholds are fixed before training and copied into `docs/acceptance/cellpose-hitl-preregistration.md`. They may be tightened, but loosening needs your written approval in the acceptance PR (the rca rule).
- Held out: whole samples P7113 and P5011 (test crops annotated from blank). Validation is for choices; test is scored once.
- All thresholds below are [L] first-run targets for you to confirm (OD-22).

| ID | Criterion (held-out test crops unless stated) | Threshold |
|---|---|---|
| A1 | Pooled AP@0.5, candidate minus stock baseline (production context, filtered output) | ≥ +0.05, and the paired-bootstrap 95% CI lower bound > 0 |
| A2 | Pooled AP@0.75, candidate minus baseline | ≥ +0.03 |
| A3 | No stratum worse: per-stratum AP@0.5 | ≥ baseline − 0.03 in every stratum (L1, upper, deep, boundary, WM) |
| A4 | Recall at IoU 0.5 of GT cells > 150 µm² (large-neuron proxy) | ≥ baseline + 0.10 |
| A5 | FP per mm² in the WM test crops and in test crops in the top neuropil-index tercile | ≤ baseline |
| A6 | Generalisation: AP@0.5 gain on held-out samples vs on validation (training samples) | ≥ 50% of the validation gain |
| A7 | Seed stability: AP@0.5 spread over the 3 seeds | ≤ 0.02, and A1 holds for the worst seed |
| A8 | Beats the cheap baselines: candidate AP@0.5 vs the best of B1–B3 | ≥ best + 0.02; otherwise adopt the cheap fix instead |
| A9 | Human ceiling (double annotation) | reported; a candidate above the ceiling triggers an overfitting review |
| D1 | Nuclear transcripts assigned after ProSeg (P7113, P5011 Xenium) | ≥ baseline |
| D2 | Share of cells overlapping ≥ 2 nuclei | ≤ baseline |
| D3 | MECR | ≤ baseline × 1.05 |
| D4 | Confident broad annotation coverage; marker plausibility | ≥ baseline − 0.02; ≥ baseline − 0.01 |
| D5 | Soft broad-composition JSD, Xenium vs MERSCOPE | ≤ baseline + 0.01 |
| D6 | WM oligodendrocyte density | ≥ baseline × 0.95 |
| D7 | Total transcripts assigned; masks lost to the area filter | reported; a move of more than 5 points, or a rise in filter losses, needs a written explanation |

**Style rule** (applied in HITL-6 on P1212, before the test set is opened): choose the variant with the lower MECR, unless the other variant assigns at least 1 point more nuclear transcripts with MECR within 5%; on a tie, keep soma + proximal trunk.

**Accept rule:** A1–A5, A7 and A8 on the pooled test set (with only one crop per stratum per sample, per-sample strata are too noisy to gate). Per held-out sample: an AP@0.5 point gain of at least +0.03, and A6. D1–D6 on each held-out pair. A failure means another round (up to 3), a cheap-baseline adoption (A8), or stopping; any exception needs your written approval.

**MERSCOPE:** stays on stock unless a MERSCOPE model passes the same table on MERSCOPE test crops.

---

## 10. Risk register

| ID | Risk | L / I | Mitigation | Where |
|---|---|---|---|---|
| R1 | Overfitting to few crops or one annotator's style | H / M | Whole-sample hold-out; paper hyperparameters with a ≤ 4-point grid; seed spread; A6, A7, A9 | S4, §9 |
| R2 | Annotator bias: anchoring to the pre-fill, drift between rounds, one annotator | H / M | Blank test crops; a pre-fill agreement diagnostic; a versioned guide; double annotation; optional second annotator (OD-18) | S2, S3 |
| R3 | Domain shift across donors, staining batches, XOA versions and future panels | M / H | Held-out samples; per-sample reporting; `model.json` channel and pixel checks; a new panel or stain set needs its own evaluation | S5, S6 |
| R4 | GPU contention on the single A5000 (another process used 6.4 GB on 2026-09-30) | M / M | Nextflow-only training and evaluation with the Dwight flock, `maxForks 1`, a free-memory check; no GUI or ad hoc training on the server | S4 |
| R5 | Frame mistakes: aligned vs native, VZG2 offsets, `list(sdata.images.keys())[0]` (`pipeline.py:445`) picking the wrong element | M / H | Named native element; full affine; refuse aligned elements; transcript-label frame test | S1, HITL-2 |
| R6 | Normalisation mismatch between crops and production tiles, including tile-size changes under GPU load | H / H | Route A tile-exact input with an equality test; pinned tile size; raw crops kept | §4.0 |
| R7 | Silent model fallback or loose weight loading (K3) | M / H | Preflight file and sha256 checks; a fixed-crop load check | S6 |
| R8 | Filters delete the improved cells (400 µm² bound, eccentricity) | H / M | Raw vs filtered evaluation; B3; per-model bounds as explicit parameters (OD-14) | S5 |
| R9 | A different cellprob distribution changes ProSeg's prior weighting and the meaning of the −5 threshold | M / M | Validation-tuned thresholds; the downstream D-criteria; cellprob histograms in the report | S5 |
| R10 | Baseline destroyed by an in-place rewrite, or a new model silently reused as the old mask (K4) | M / H | Separate outdir; identity sidecar; refuse on mismatch; sha256 before and after in HITL-6 | S6 |
| R11 | Version drift (lock 4.1.1 vs env 4.2.1.1), or a future cellpose release | H / H | HITL-0; the baseline-reproduction check; versions in the build hash | HITL-0, S5 |
| R12 | Unfair platform comparison (Xenium fine-tuned, MERSCOPE stock) | M / M | MERSCOPE fairness crops; provenance in COMPARE; a matched fine-tune if needed | §3.4 |
| R13 | `_seg.npy` is a pickle (`allow_pickle=True`) | L / M | Import only packages you return; checksum against the manifest; never from third parties | S3 |
| R14 | Licence: weights derive from CC-BY-NC training data | L / M | Weights kept in the lab store, not in the repo; licence recorded in `model.json` | S6 |
| R15 | Evaluation circularity: class labels and sampling weights come from the current segmentation | M / M | Label-free size classes and strata gate; class-matched metrics are diagnostic only | S5 |
| R16 | Dangling `source_spatialdata.zarr` links (K13) | M / L | Read from the durable latest zarr's native element | S1 |
| R17 | Annotation time overruns | M / M | Pilot timing; per-round budget; the stop rule | §3.3 |
| R18 | Conflicts with the rca integration branch in `main.nf` / `config.py` | M / L | Marked hooks away from rca's; merge `main` into rca as usual | §5 |
| R19 | Neurons with process-including masks are split by the per-tile connectivity filter or trimmed by ProSeg compactness | M / M | Single-stroke rule; the derived soma-only variant scored downstream against the drawn style | S2, S5 |

---

## 11. Sources

- Cellpose 4.2.1.1 docs: https://cellpose.readthedocs.io/en/latest/ (gui.html, train.html, models.html, outputs.html, settings.html, faq.html, benchmark.html); source https://github.com/MouseLand/cellpose/tree/v4.2.1.1/cellpose (models.py, train.py, io.py, cli.py, metrics.py, vit.py, gui/{gui,io,guiparts,menus}.py); releases https://github.com/MouseLand/cellpose/releases/tag/v4.2.1.1; issues #1206, #1291, #1335, #1346, #1353, #1424, #1474, #1476, #1493 (https://github.com/MouseLand/cellpose/issues/<n>); paper script https://github.com/MouseLand/cellpose/blob/main/paper/cpsam/train_subsets.py. Line-level references: [CP §2–§5].
- Pachitariu, Rariden & Stringer 2025, Cellpose-SAM, bioRxiv 10.1101/2025.04.28.651001. Pachitariu & Stringer 2022, Cellpose 2.0, Nat Methods, doi 10.1038/s41592-022-01663-4.
- Jones et al. 2025, ProSeg, Nat Methods, doi 10.1038/s41592-025-02697-0; https://github.com/dcjones/proseg (README limitations and Cellpose initialisation; `src/sampler/voxelcheckerboard.rs`).
- 10x CG000750 segmentation technical note: https://cdn.10xgenomics.com/image/upload/v1710785020/CG000750_XeniumInSitu_CellSegmentation_TechNote_RevA.pdf
- Communications Biology 2025, doi 10.1038/s42003-025-08518-6 (18S in human cortex). Marco Salas et al. 2025, bioRxiv 10.64898/2025.12.07.692889 (extrasomatic RNA). arXiv 2606.09675 (segmentation evaluation in spatial transcriptomics; MECR).
- Tools: https://github.com/BIOP/qupath-extension-cellpose; https://github.com/MouseLand/cellpose-napari (issue #61).
- MerXen code and data: [CODE §0–§6], with file:line references throughout this plan; task log `/srv/storage/MerXen/work/58/e6e47088f09f52cb75b0f3744d8687/.command.log:87-90`; `/srv/storage/MerXen/results/P7513/xenium/segmentation/cellpose_stitching_stats.json`.

---

## Open decisions for the user

Each decision has a recommended default, which the plan uses unless you choose otherwise. "Blocking" means the decision is needed before the named milestone.

| ID | Question | Options | Recommended | Blocking |
|---|---|---|---|---|
| OD-1 | Do HITL-0 (pin cellpose, name cpsam_v2, record weights) now, as a data-integrity fix on `main`? It affects MERSCOPE too | now / with HITL / never | **Now**, separately from HITL | – |
| OD-2 | Base model and pin | cpsam_v2 on `>=4.2.1.1,<4.3` / exact `==4.2.1.1` in pyproject / cpsam on 4.1.1 | **cpsam_v2, `>=4.2.1.1,<4.3` + lock 4.2.1.1 + sha256 check** | HITL-0 |
| OD-3 | Annotation editor | Cellpose GUI / QuPath BIOP extension (whole-slide context, global normalisation, but it recommends continuing from custom models) | **Cellpose GUI**; QuPath as fallback | HITL-2 |
| OD-4 | Where training runs | pipeline (Nextflow, GPU lock) / GUI | **Pipeline**; the GUI edits only (§3.2) | HITL-4 |
| OD-5 | Normalisation | A: tile-exact crops, `normalize=False` / B: per-crop training + `tile_norm_blocksize` at inference (changes production for both platforms) | **A** | HITL-2 |
| OD-6 | Channels | training `[DAPI, 18S, 0]` with B1 `[DAPI, 18S, ATP1A1/CD45/E-Cad]` as a variant; GUI view shows ATP1A1 / view DAPI+18S only / AlphaSMA-Vimentin variant | **Train both channel sets on the same annotations; view with ATP1A1** | HITL-2 |
| OD-7 | Neuron annotation style | soma + proximal trunk (≤ ~1 soma diameter) / soma only / soma + all visible processes | **Draw soma + proximal trunk**; derive soma-only by opening; choose by the §9 style rule on a training sample (P1212) downstream (§4.2) | HITL-2 (drawn style); HITL-6 (final choice) |
| OD-8 | Anucleate somata | draw if compact and ≥ ~5 µm / never / always | **Draw if compact and ≥ ~5 µm** | HITL-2 |
| OD-9 | Pre-fill | train/val pre-filled, test blank / all pre-filled / all blank | **Train/val pre-filled; test blank** | HITL-2 |
| OD-10 | Crop size | Xenium 512 / 768 / 1024 px | **768 px**, confirmed by the pilot | HITL-2 pilot |
| OD-11 | Split | train P7513 + P1212, test P7113 + P5011 / leave-one-sample-out / add other Xenium samples if any (P7417, P5822, P4815 appear in logs; platform not checked) | **Train P7513 + P1212; test P7113 + P5011** | HITL-3 |
| OD-12 | Platform scope | Xenium only + MERSCOPE fairness crops / matched MERSCOPE fine-tune now / one joint model | **Xenium only + 5 MERSCOPE fairness crops** | HITL-3 |
| OD-13 | Scale | native / downsampled via diameter | **Native** | HITL-2 |
| OD-14 | Filters and thresholds for custom models | explicit per-platform overrides tuned on validation / keep global / auto from `model.json` | **Explicit overrides, tuned on validation, recorded in `model.json`** | HITL-5 |
| OD-15 | Mouse strata source and timing | manual region polygons (new `region` role) / [rca] M6 tile regions; when mouse Xenium arrives / MERSCOPE mouse now | **Manual polygons; wait for mouse Xenium** | HITL-n |
| OD-16 | Model store location | `/srv/storage/MerXen/cellpose_models/` / SSD / elsewhere | **`/srv/storage/MerXen/cellpose_models/`** | HITL-4 |
| OD-17 | Fine-tune the nuclei model as well? | no / later | **No** (the nuclei mask is the independent downstream check) | – |
| OD-18 | Annotators | you only, with 3 crops re-annotated for the ceiling / add a second annotator | **You only + self re-annotation**; second annotator if available | HITL-3 |
| OD-19 | Sharing weights | lab only / publish (CC-BY-NC derived) | **Lab only** | – |
| OD-20 | Laptop env | standalone `envs/environment.cellpose-gui.yml` (a stated deviation) / `[hitl-gui]` pyproject extra | **Standalone env file with a lock-sync test** | HITL-2 |
| OD-21 | Branching | per-milestone PRs to `main` / an integration branch | **Per-milestone PRs to `main`** (every milestone is inert) | HITL-1 |
| OD-22 | Acceptance thresholds A1–A9, D1–D7 | as proposed / tighter / looser | **As proposed**; confirm before HITL-3 merges | HITL-3 |
| OD-23 | Output strategy for custom-model runs | separate outdir with the MERSCOPE side copied / model-keyed paths in the same outdir (much larger change) | **Separate outdir** | HITL-6 |

> **Status:** draft (rev1), 2026-09-30, revised after review (§12 lists each review finding and what changed). This is a plan only: nothing is implemented, run or trained. The work is an optional, later improvement with its own workflow. Current segmentation and analysis continue unchanged, and the pipeline default (stock Cellpose-SAM for both platforms) stays the default unless you change it after the acceptance tests in §9 pass. The decisions that need your input are collected in the final section; each has a recommended default.

# MerXen plan: optional human-in-the-loop (HITL) Cellpose fine-tuning for Xenium segmentation

- **Effort name:** `cellpose-hitl` ("hitl"). **Date and base:** 2026-09-30; `origin/main` `6daa255`.
- **Why:** MERSCOPE Cellpose masks are excellent, but Xenium masks are not ideal for neurons and some other cells, including cells in white matter. The Xenium 18S RNA stain varies more between cell types than MERSCOPE's PolyT. It labels somata and some processes well, but it also carries neuropil-like background and does not always give smooth boundaries. The Cellpose masks are the ProSeg prior, so these errors carry through to the final segmentation and transcript assignment for every cell (§2.1).
- **Evidence:** `$H` = `/srv/storage/MerXen/annotation_dev/hitl_plan/`. **[CP §n]** = `$H/RESEARCH_CELLPOSE.txt` section n (Cellpose 4.2.1.1 source, docs, issues and papers, with URLs). **[CODE §n / Fn]** = `$H/RESEARCH_CODE.txt` section n or headline finding Fn (MerXen code and published data, with file:line). `$H/PROGRESS_plan_reviser.txt` records the checks made for rev1. Code references are to `6daa255` unless marked [rca] (the `feature/robust-celltype-annotation` branch). Cellpose references (`cellpose/…`) are to the installed 4.2.1.1 copy, which matches https://github.com/MouseLand/cellpose/tree/v4.2.1.1/cellpose.
- **Confidence tags:** [H] verified in code, logs or data; [M] documented upstream or measured but indirect; [L] design judgement or estimate, to be checked in the named milestone.
- **Abbreviations:** cpsam_v2 = Cellpose-SAM v2, the default model of Cellpose 4.2 (June 2026). GT = ground-truth (human-annotated) masks. AP@t = average precision at IoU threshold t, defined as TP / (TP + FP + FN) as in `cellpose.metrics.average_precision`. ROI = one annotated cell mask. GM / WM = grey / white matter. X / M = Xenium / MERSCOPE. Arm = one full `main.nf` run in its own outdir (§4.5 S5b).

---

## 0. Summary

1. **HITL training works in current Cellpose** [H, CP §0, §2]. Cellpose 4.2.1.1 (the latest release, 2026-06-14) can fine-tune cpsam_v2 from a folder of images and `<image>_seg.npy` files, through the GUI (Models > "Train new model with image+masks in folder", Ctrl+T, `cellpose/gui/menus.py:118-119`), the command line or `train.train_seg`. The GUI loads pre-made `_seg.npy` masks for correction and autosaves each edit. GUI training always starts from a built-in model (`gui/guiparts.py:411`; issue #1206), but that matches the maintainers' advice to retrain the base model on all annotations each round, so it is not a blocker. Training runs in the pipeline instead because of the GPU lock, provenance, reproducible normalisation and settings, and because a laptop CPU is too slow for this ViT-L model (§3.2). **Plan:** the pipeline cuts crops and pre-fills them with the current masks; you correct them in the Cellpose GUI on a laptop; the pipeline then trains on the server and segments as usual. Cellpose's FAQ describes this as "HITL without the GUI" [M, CP §5.3].
2. **Prerequisite: production and CI use different Cellpose versions and models, and outputs do not record which one ran** [H, CP §1, CODE F1–F2]. Every published mask (both platforms, cells and nuclei) was made by cpsam_v2 on cellpose 4.2.1.1, because the conda env resolves `cellpose>=3.0` (`pyproject.toml:23`) on the day it is built. The lock pins 4.1.1 (`requirements/requirements.lock:92`), whose only model is the original cpsam. torch is not pinned either: the published masks ran on torch 2.13.0, the current env has 2.14.0 and the lock has 2.10.0 (K14). MerXen's `model_type="cyto3"` has no effect in Cellpose 4 (`src/merxen/segmentation/cellpose.py:49-53`, `docs/configuration.md:117`). Nothing in the outputs records the weights. **HITL-0** fixes this on `main` as a data-integrity change that also covers MERSCOPE, before any fine-tuning.
3. **Cheap baselines come first** [H, CODE F4–F6]. (B1) The Xenium images already carry an unused boundary stain, ATP1A1/CD45/E-Cadherin, that can be added as Cellpose's third channel with no code change (B1′ combines it with 18S in one channel, as the Cellpose docs suggest). (B2) `cellpose_cellprob = -5.0` (`workflows/nextflow.config:19`; Cellpose's default is 0) may itself cause masks to spread into neuropil; the 5 µm² floor of the final area filter removed 13,351 masks on P7513 Xenium and 35,529 (15%) on P5011, a sign of many spurious small masks. (B3) The final area filter drops masks over 400 µm² (`src/merxen/config.py:47-48`; 1,185 on P7513, 2,641 on P5011), and the per-tile eccentricity > 0.99 filter penalises masks that include processes. (B4) Cellpose's `diameter` setting, which the docs recommend for large cells. These are scored on the same test crops, and a fine-tune is adopted only if it beats the best combination of them (B*), chosen on validation (§9 A8).
4. **Every comparison is same-code** [H, K14–K15]. The published outputs were made with older code and another torch, so downstream effects are measured between arms run at one commit that differ only in the Xenium model. ProSeg has no seed option, so a replicate arm measures its run-to-run noise and sets the downstream margins (§4.5 S5b).
5. **Six stages** in a separate entry workflow (`workflows/cellpose_hitl.nf`): S1 `CROP_EXPORT` (region-stratified crops at native resolution with exactly the production model input, pre-filled masks, a manifest and a download package); S2 an annotation guide; S3 `IMPORT` (validation and a versioned, content-hashed training set); S4 `TRAIN` (fine-tune cpsam_v2, holding out whole samples, under the pipeline's GPU lock); S5 `EVALUATE` (AP per region and size class against the stock model and the cheap baselines, then downstream effects after ProSeg); S6 a content-hashed model registry and optional per-platform model parameters. When those parameters are unset, legacy config JSON and `-resume` hashes stay byte-identical.
6. **Test design** [L]. Test crops come from two held-out donors, are drawn at random within strata (never selected from baseline errors), are annotated from blank and are sized by a precision simulation before pre-registration (default 24 crops of 512 px). The test set is opened once, after a freeze record has fixed every choice made on validation (§9).
7. **Annotation effort** [L]: about 20–33 h of your time to the end of round 2 (pilot, about 25 pre-filled training and validation crops, 24 blank test crops, the human-ceiling and anchoring re-annotations, one correction round), and up to about 45 h with round 3 and the MERSCOPE fairness crops (§6). Cellpose 2.0 needed 100–200 ROIs per category with HITL [M, CP §4.3].
8. **Compute** [M/L]: one fine-tune takes about 8.5 GB of GPU memory at batch size 1 and 10–50 min on the A5000 in round 1; iterations grow with the training set, so training totals about 7–30 GPU-h over the effort. Each downstream arm takes about 82 min of Xenium Cellpose, 2.6 h of ProSeg (P7513) and the downstream stages; HITL-6 needs 7 arms of about 240 GB each, run in sequence (§6). Each model is about 1.2 GB.
9. **Scope** [decided by your brief]: Xenium first, human first. MERSCOPE keeps its current model unless the evaluation shows a gain there (§3.4). Mouse is covered by the design (cortex, hippocampus and midbrain strata) but waits for mouse Xenium data or your request (HITL-n). Developer effort for HITL-0 to HITL-7 is about 22–30 days [L], plus annotation time and compute.

---

## 1. Goals and non-goals

**Goals.**
- G1. Xenium cell masks that match the true soma (plus an agreed short proximal-process rule) for neurons, glia in grey and white matter, and vascular cells, measured on held-out annotated crops from held-out samples.
- G2. Better ProSeg priors, measured downstream against a same-commit stock arm: fewer missed cells and fewer false neuropil masks, without more contamination (§4.5 S5b, §9).
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
- ProSeg does not add cells. Cells that Cellpose misses are lost, and false neuropil masks stay as cells. ProSeg regularises shape with a compactness prior (MerXen: 0.04 at 0.5 µm voxels, `workflows/nextflow.config:49-54`), and each mask pixel's vote is weighted by `logistic(cellprob) · 0.5 · discount + 0.5` [M, CP §6.3; github.com/dcjones/proseg at MerXen's pinned rev e7df1ea, `src/sampler/voxelcheckerboard.rs`]. A fine-tune therefore changes both the labels and the probability map that ProSeg uses.
- Some unassigned transcripts near neurons and glia are real RNA in processes [M, CP §6.3; bioRxiv 10.64898/2025.12.07.692889]. Soma-centred masks leave that RNA unassigned, which is a known and accepted trade-off.
- Nothing published says whether ProSeg priors should be drawn as soma only or soma plus processes. Cellpose 2.0 treats the choice as a style that must be applied consistently [M, CP §6.3; doi 10.1038/s41592-022-01663-4]. §4.2 proposes a rule and a downstream comparison against a soma-only variant derived from the same masks.

### 2.2 What MerXen runs today (facts the design depends on)

| # | Fact | Source | Consequence |
|---|---|---|---|
| K1 | Production loads cpsam_v2 on cellpose 4.2.1.1. The lock pins 4.1.1 (cpsam), and the container installs the lock. Two weight files are cached: cpsam (sha256 `e1440429…`) and cpsam_v2 (`0f1cc3f7…`) | `.command.log:87-90` of `work/58/e6e470…` (P5011 X); `pyproject.toml:23`; `requirements.lock:92`; CODE F2 | HITL-0; fine-tune from cpsam_v2 on the same version |
| K2 | `model_type` is ignored in Cellpose 4, and there is no `pretrained_model` field | `segmentation/cellpose.py:49-53`; `config.py:19`; CODE F1 | The one inference change: `CellposeModel(pretrained_model=…)` |
| K3 | A missing `pretrained_model` path falls back silently to cpsam_v2 (warning only), and weights load with `strict=False` | `cellpose/models.py:141-152`, `cellpose/vit.py:31`; CP §4.5 | The preflight checks the file and its sha256; a fixed-crop load check |
| K4 | The Python reuse rule returns the published mask whenever the mask, cellprobs and CSV all exist, whatever model is configured. If any is missing, Cellpose re-runs, and `run_tiled_cellpose` first deletes the output paths it is given. ProSeg reuses an existing latest zarr; the nuclei mask is reused whenever it exists | `segmentation/pipeline.py:570-611`, `:825-827`, `:996-999`; `cellpose.py:510-521`; CODE F3 | Custom-model runs go to a separate outdir and carry a model-identity sidecar check (§4.6); a copied mask without its CSV is not reusable (§4.5 S5b) |
| K5 | Thresholds are the same for both platforms: flow 0.7, cellprob −5.0, diameter None, bf16 off, bsize 256. Only the channels differ per platform (Xenium default `[DAPI, 18S]`) | `nextflow.config:16-38`; `main.nf:334-366`, `:2091-2113` | Per-platform overrides are optional and inert when unset |
| K6 | Normalisation: one joint p2/p98 clip over all selected channels of the halo tile, conversion to uint8 and zero-padding to 3 channels. Then, inside `CellposeModel.eval` and before its internal 256 px tiling, per-channel [1, 99] percentiles computed on a strided subsample of the whole tile: stride = side // 224 when the tile has more than 224³ pixels (27 on a 6144 tile, 26 on a 5888 edge tile); smaller tiles use every pixel | `io/image_source.py:327-339`; `cellpose.py:73-80`; `cellpose/models.py:262-293`; `cellpose/transforms.py:180-185`, `:633`; CODE §1.2 | Crops are cut from the model input computed on the exact halo tile (§4.0) |
| K7 | A missing requested channel is dropped silently | `io/image_source.py:375-381` | Crop export and the preflight fail on a missing channel |
| K8 | Filters: final area 5–400 µm² (on P7513 X it removed 13,351 small and 1,185 large masks; on P5011 X 35,529 small and 2,641 large of 238,225); per tile, eccentricity > 0.99 and the smallest 1% of areas are removed | `config.py:44-48`; `mask_filter.py:79-167`; `pipeline.py:654-682`; `.command.log` of `work/d2/2bb636…` and `work/58/e6e470…`; CODE F5 | Evaluate both raw and filtered output; B2, B3; bounds per model (OD-14) |
| K9 | Xenium `morphology_focus` has DAPI, ATP1A1/CD45/E-Cadherin, 18S and AlphaSMA/Vimentin; uint16; 0.2125 µm/px; identity transform. MERSCOPE `MERSCOPE_z_projection` has [PolyT, DAPI] at 0.1080 µm/px. VZG2 (mouse) has non-zero offsets | CODE §2; CP §6.1 | Crops use the full affine and name the image element; never `*_aligned_nonrigid` |
| K10 | The cortical-depth GeoJSONs hold pia, grey/white and side lines in native microns for all 10 published human files. Every side line is open, so no cell is ever labelled `white_matter`; in practice the white matter is `outside_brain`. P1212 M and P5011 M each have a piece with a pia line but no grey/white line (`piece_mode: mask_qc_only`) | CODE §3.1, F7; `cortical_depth/pipeline.py:540-588`; `cortical_depth/ribbon.py:462`; the GeoJSON files | WM strata use a positive rule and depth-mode pieces only (§4.1) |
| K11 | Mouse data are MERSCOPE only, with no manual region annotations. [rca] M6 region inference labels 150 µm tiles (`Isocortex`, `HPF`, `MB`, …) but has not been run yet | CODE §3.2 | Mouse strata are pluggable; the mouse work waits (HITL-n) |
| K12 | GPU: one RTX A5000 (24 GB). Dwight takes an flock in `beforeScript` for `CELLPOSE_SEGMENT` and `CELLPOSE_NUCLEI_SEGMENT` only when `params.cellpose_gpu == "true"`; a GUI or manual run outside Nextflow never does. The `gpu` profile adds `--nv` only to `CELLPOSE_SEGMENT`, `ALIGN` and clustering | `workflows/conf/dwight.config:65-69`, `:86-125` (condition at `:89`); `nextflow.config:629-646`; CODE §1.4 | HITL GPU processes take the lock unconditionally and get `--nv` (§4.4) |
| K13 | `source_spatialdata.zarr` in results is a symlink into `work/`, which dangles once `work/` is cleaned; ENRICH copies the native images into the durable latest zarr | CODE §2; `enrichment/enrich.py:509-523` | Crop export reads the native element from the durable zarr, by name, and a test checks it equals the image Cellpose read |
| K14 | The published P7513 Xenium segmentation is dated 2026-07-31. Since then the ProSeg input changed (2579e9d, 2026-09-26: all control features dropped; the human Xenium sections lose the same transcripts as before, MERSCOPE now drops its Blank codewords), the reuse guards changed (fd539cd, b208800, 49ebb40) and many downstream stages changed. The published masks ran on torch 2.13.0 (env-95b80925; the `/srv` copy is gone, a same-hash copy survives on `/media/mathieubo/SSD2`), the current env has torch 2.14.0 and the lock 2.10.0; torch is not pinned in `pyproject.toml` | file dates in `results/P7513/xenium/segmentation/`; `git log`; `torch-*.dist-info` in each env | Downstream comparisons use same-commit arms, never the published outdir (§4.5 S5b); HITL-0 bounds torch |
| K15 | ProSeg at e7df1ea has no seed option (only `--nthreads`), so two runs on the same input differ | proseg `src/main.rs` at e7df1ea | A replicate arm measures run-to-run noise and sets the downstream margins |
| K16 | `train_seg` re-seeds numpy with the epoch index every epoch, and all order and augmentation draws (rotation, flip, scale, crop) use numpy; only layer-drop (rdrop 0.4) uses torch | `cellpose/train.py:441-448`; `cellpose/transforms.py:870-880`; `cellpose/vit.py:98`, `:139-142` | A seed must also permute the training list (§4.4) |
| K17 | `build_cellpose_model` passes `use_bfloat16` only when `model_type` is set. The Python default and Cellpose's own default are True; production passes False from Nextflow. A model built with the default loads its float32 weights as bf16, and `train_seg` then converts the rounded weights back to float32. The GUI's own training does this too | `cellpose.py:49-52`; `config.py:25`; `cellpose/models.py:105`, `:155`; `cellpose/vit.py:50-52`; `cellpose/train.py:365-370`; `cellpose/gui/gui.py:1967` | HITL-0 always passes `use_bfloat16`; training builds its model with `use_bfloat16=False` |

### 2.3 What Cellpose supports

- **GUI files** [H, CP §3]: the GUI opens one image and cycles through its folder with A/D. It lists every `.png`, `.tif` and similar file in that folder except names ending `_cp_output`, `_flows`, `_cellprob` and `_masks`, and it does not look into subfolders (`cellpose/io.py:420-437`; `gui/gui.py:903-906`). An `<image>_seg.npy` beside it loads automatically; it must contain `outlines` and `masks`, and `flows`, `colors` and `ismanual` are optional (`gui/io.py:218-320`). Every manual add, delete or merge rewrites the `_seg.npy` (autosave, `gui/gui.py:1228`, `:1358`, `:1376`, `:1558`); running a model does not (use Ctrl+S). Autosave writes `manual_changes`, `model_path`, `normalize_params` and the laptop's full `filename` path after the first edit (`gui/io.py:540-621`), so these keys show that a crop was touched, not that it was finished. Editing is limited to single-stroke outlines, delete, merge and rectangle bulk-delete; there is no pixel paint, so fixing a boundary means delete and redraw. If the stored `filename` path is missing, the GUI uses the image with the same basename in the `_seg.npy` folder, which is what lets the crops move to a laptop.
- **Training** [H, CP §4.1–4.2]: `train.train_seg` always uses AdamW with warm-up and decay and updates all weights. The tile size must be 256 for SAM models (`train.py:360-363`). At most 3 channels are used. The loss treats every unlabelled pixel as background, so **annotations must be dense**. Both `train_seg` and the command line drop crops with fewer than 5 masks unless `min_train_masks=0` is set (`train.py:317`). Each iteration trains on one randomly scaled, rotated and cropped 256 px patch per image (`transforms.py:866-880`), so a 768 px crop contributes about 1/9 of its area per epoch. The paper's few-shot protocol is lr 1e-5, weight decay 0.1, batch 1, 100 epochs and at least 8 images per epoch.
- **Normalisation warning** [M, CP §4.2; train.rst "Re-training a model"]: crops normalised on their own look different from full images. The GUI also stores the annotator's display settings in `normalize_params`, which training must ignore.
- **Channels and size** [M, https://cellpose.readthedocs.io/en/latest/settings.html]: Cellpose-SAM uses the first 3 channels; for a third channel the docs say to "omit it from the input, or combine it with the cytoplasm channel, or train a new model with all three inputs". For very big cells they suggest `diameter`, which rescales the image by 30 / diameter inside `eval`; with the default `resample=True` the flows are resized back, so masks come out at native resolution (`models.py:267-269`, `:353-366`) and MerXen's `factor_rescale = 1.0` rule is not affected.
- **Inference** [H, CP §4.5]: `CellposeModel(pretrained_model=<path>)` goes through the same eval and tiling path, with eval bsize 256 (issue #1335).
- **Evaluation** [H, CP §5.1]: `metrics.average_precision`, `aggregated_jaccard_index` and `boundary_scores`. Loss values cannot be compared across versions (issue #1493, still open).
- **Alternatives** [M, CP §5.4]: cellpose-napari does not work with Cellpose 4. The QuPath BIOP extension (v0.12.1) supports Cellpose-SAM, trains from densely annotated rectangles on the whole slide and supports global normalisation. It is the fallback editor (OD-3). Training on Apple Silicon through MPS is supported (4.0.7 release notes, PR #1278), but pipeline training stays the plan (§3.2).

---

## 3. The HITL loop

### 3.1 Design

```
 server (Nextflow, cellpose_hitl.nf)                laptop (cellpose[gui], same version)
 ───────────────────────────────────                ─────────────────────────────────────
 S1 CROP_EXPORT  ── package round r ──────────────▶  open crop, correct pre-filled masks
   (strata, tile-exact input, pre-fill,               (or draw test crops from blank),
    manifest, guide, review.tsv)                      Ctrl+S on every crop, mark it reviewed
                                                              │
 S3 IMPORT  ◀────────── return package ───────────────────────┘
   (validate, versioned training set)
 S4 TRAIN   (cpsam_v2 + ALL training crops so far; GPU lock; seeds permute the data)
 S5 EVALUATE (validation each round; sealed test opened once, after the freeze record)
 S6 REGISTER (content-hashed model) ── round r+1 pre-filled with this model
 then: same-commit main.nf arms (stock, candidate, replicate) in separate outdirs → S5b
```

### 3.2 GUI training or pipeline training: pipeline training (OD-4)

The GUI is the editor only. Training runs in the pipeline, from the GUI-edited masks, because:
1. GUI training on a laptop CPU is impractical for ViT-L [M, CP §5.3]. Run on the server, the GUI would bypass the GPU lock (K12) and has no display there.
2. GUI training takes normalisation from the annotator's display settings (`gui/gui.py:1773-1798`, `:1959-1979`), uses `nimg_per_epoch = max(2, n)` rather than the paper's minimum of 8, and builds its model with the bf16 default (K17). None of this is recorded or reproducible [H, CP §4.2].
3. Pipeline training records the base-weight sha256, the training-set hash, the split, the seed and the parameters (§4.6).

The GUI's other limit, that it trains only from a built-in checkpoint (`gui/guiparts.py:391-411`; issue #1206), does not matter here: the maintainers recommend retraining the built-in model on all annotations every round, and the pipeline does the same [M, CP §2].

The annotator should not run models in the GUI on the pre-filled crops, because a model run replaces the masks and does not autosave (§4.2).

### 3.3 Rounds (the HITL part)

- **Round 0 (pilot, HITL-2):** 3 crops (deep GM, the GM/WM boundary and WM), fully annotated in the recommended style (§4.2), plus a short responsiveness test: open one crop at each of 512, 768 and 1024 px on your laptop and make a few edits. You also check the strata overlay of each sample (§4.1). Output: the crop sizes (OD-10), measured minutes per crop, and whether the style rules are workable (OD-7).
- **Round 1a:** about 25 training and validation crops (768 px by default, about 7 of them validation) pre-filled with the published baseline mask. Two validation crops are also annotated from blank at least 2 weeks later, to measure how much the pre-fill anchors the GT (R2).
- **Precision simulation and pre-registration (HITL-3):** round-0 and round-1a GT, scored against the stock model and against a stock-model variant (cellprob 0), give the per-crop spread of AP differences. Pre-filled GT is anchored to the stock mask, so the pilot crops and the two blank-annotated validation crops are weighted as the least biased inputs [L]. From it the simulation sets the test-set size and allocation and decides which per-stratum criteria are gates and which are report-only (§9). The pre-registration is merged before any training.
- **Round 1b: the sealed test set**, annotated from blank before you see any trained model's output (default 24 crops of 512 px, §4.1). Then round-1 training and validation.
- **Round 2:** new training crops pre-filled with the round-1 model, weighted towards the strata with the lowest validation AP and the most correction effort (`manual_changes`, §4.3). Retrain from cpsam_v2 on all training crops so far. The validation set stays fixed after round 1a, so rounds are compared on the same crops.
- **Round 3 (optional):** as round 2.
- **Stop rule** (pre-registered): stop when validation AP@0.5 improves by less than 0.02 over the previous round, after round 3, or when your annotation budget is spent. Round, hyperparameter, channel-set, threshold and style choices use validation (and, for style, the downstream metrics of a training sample) only.
- **Test opening:** once, after the freeze record (§9). A test failure never leads to another round on the same test set (§9 rules).

### 3.4 Platform scope and fairness (OD-12)

- v1 fine-tunes **Xenium only**. MERSCOPE keeps stock cpsam_v2.
- **MERSCOPE fairness check** (recommended, about 5–7 h of annotation): 5 MERSCOPE test crops from the paired held-out sections, annotated from blank with the same guide, 2 of them re-annotated for a MERSCOPE human ceiling. Raw AP is not compared across platforms, because the stains, resolutions and crop contents differ. Instead each platform is compared with its own ceiling: gap = ceiling AP@0.5 − model AP@0.5 on that platform's crops. If the stock MERSCOPE gap is larger than the fine-tuned Xenium gap by more than 0.05, a matched MERSCOPE fine-tune becomes HITL-n. With 5 crops this is a report with confidence intervals, not a gate. MERSCOPE is re-segmented only if its own evaluation shows a gain.
- Cross-platform downstream results (D5) are reported under both Xenium settings, the stock arm and the candidate arm.
- Every comparison output records each platform's model identity (§4.6). A Xenium-only lab fine-tune is a pre-processing difference that reports must state.

---

## 4. Stage design

### 4.0 Shared contracts (every stage)

- **Frame.** Everything is in the native frame of the platform being segmented: the s0 pixel grid of the named image element (`morphology_focus` or `MERSCOPE_z_projection`), which matches `cellpose_masks_tiled.npy`. Conversion between pixels and microns always uses the full affine from `cellpose_transforms.json` (`build_cellpose_affine_to_microns`, `cellpose.py:698-727`), never µm divided by pixel size (VZG2 offsets, K9). Aligned elements are refused [H, CODE §2].
- **Tile grid.** A crop lies inside one tile core of the full-image grid (`iter_core_tiles`, `cellpose.py:141-209`) with the tile size recorded in `cellpose_stitching_stats.json` (6144 on P7513, 90 tiles) and the 256 px stitch overlap. It keeps a margin of at least 256 px plus the largest expected cell (about 400 px on Xenium) from every core edge. Objects from a neighbouring tile reach at most 256 px into this core, so with that margin stitching cannot change what lies inside the crop.
- **Model input.** The crop is the exact array the production model saw at those pixels. HITL-1 factors the per-tile body of `run_tiled_cellpose` (`cellpose.py:610-675`) into one shared function (hook C7) that fetches a global tile, runs `prepare_cellpose_input` and returns the model input, the raw and per-tile-filtered masks and the cellprobs; production calls it unchanged. The export builds the model input by calling `prepare_cellpose_input` on the exact halo tile (joint p2/p98 over the selected channels, uint8, zero-pad) and then `cellpose.transforms.normalize_img` exactly as `CellposeModel.eval` calls it (batch axis added, `normalize_default`, so the strided [1, 99] percentiles of K6), and only then crops it. The per-channel low and high values are recomputed with the same subsample and recorded, and a test asserts that `normalize_img` with those values as `lowhigh` gives the same array. The result is stored as float32. Training uses `normalize=False`, so training input equals inference input (route A, OD-5) [M, CP §4.2, CODE §6.3]. HITL-2 tests this equality on 6144 px, 5888 px edge and small tiles (§8).
- **Raw copy.** The raw uint16 crop of *all* channels is stored for inspection. It cannot rebuild the model input for another channel set, because the joint p2/p98 and Cellpose's percentiles are whole-tile statistics of the selected channels. A new channel set is built by re-reading the halo tile (its bbox is in the manifest) from the durable zarr. Masks do not depend on channels.
- **Channels.** Selected by name in the seg-config order. A missing channel is an error (K7). The default training set is `[DAPI, 18S, 0]`; variant B1 is `[DAPI, 18S, ATP1A1/CD45/E-Cadherin]` (OD-6).
- **Scale.** Native resolution (`factor_rescale` must be 1.0, `config.py:29-38`). Large human neuron somata are about 70–140 px at 0.2125 µm/px, at or above cpsam's 7.5–120 px training range; fine-tuning at native scale teaches the model that scale [M, CP §5.2]. A soma plus trunk must fit the 256 px training patch, which is why the trunk is capped (§4.2). Downsampling via `diameter` is evaluated as comparator B4, and a matching downsampled fine-tune is trained only if B4 beats native stock on validation (OD-13).
- **Tile-size stability.** `choose_working_tile_size` keeps the largest candidate that does not run out of GPU memory on a probe tile (`cellpose.py:212-286`) [H]. A run on a busier GPU could pick a smaller tile and change the normalisation statistics. HITL runs therefore pin `cellpose_tile_size_candidates = [<recorded size>]` and refuse to run if that size does not fit (R6).

### 4.1 S1 `CROP_EXPORT` (CPU; `merxen cellpose-hitl-export`)

**Inputs:** the row's seg config (the same `_load_dataset_sdata` call, channels and transform as `CELLPOSE_SEGMENT`); the durable latest zarr's native image element (K13); the persistent mask and cellprobs (memory-mapped); the nuclei mask; the stratum source; optionally the cells parquet (cortical depth and broad class), used only to weight training and validation sampling, with its hash recorded.

**Strata** (a pluggable `StratumSource` built from manual annotations; no hard-coded region list, G5). Only pieces with `piece_mode: depth` (a pia line and a grey/white line) are used; `mask_qc_only` pieces and tissue beyond the end points of a grey/white line belong to no stratum and are never sampled.

| Species | Stratum | Geometry (native µm) | Source |
|---|---|---|---|
| Human | L1 / pia band | ribbon, `laplace_depth` < 0.10 | cortical-depth GeoJSON → ribbon (`ribbon.py:103`) and depth field |
| Human | Upper GM | depth 0.10–0.45 | same |
| Human | Deep GM | depth 0.45–1.0 | same (P7513 GM depth quartiles .11 / .34 / .63, CODE §3.1) |
| Human | GM/WM boundary | within 150 µm of the piece's grey/white line, either side, where the nearest point of the line is not one of its end points | GeoJSON grey/white line |
| Human | WM (positive rule) | inside the tissue polygon, outside the ribbon, on the far side of the piece's grey/white line from its pia line, with the nearest point of the line not an end point, and more than 150 µm from the line | `build_full_tissue_polygon` (`tissue.py:15-69`, already used by VALIS), the ribbon and the grey/white line (K10) |
| Mouse | Cortex, hippocampus, midbrain | `Isocortex`, `HPF`, `MB` | manual region polygons (a new GeoJSON role `region` with a `name`, which needs a loader alias, `cortical_depth/boundaries.py:22-38`) or [rca] M6 `tile_region` (OD-15) |

Strata are disjoint, and the boundary band takes precedence over deep GM and WM. The depth cut-points are [L] and are chosen so that each band holds recognisable laminar morphology. They are recorded in the manifest. The export writes one strata-overlay PNG per sample (pyramid level 4, with every candidate and chosen crop box); you check it in the pilot, and you can veto any crop (`review.tsv` status `excluded`, reason `stratum`).

**Label-free window features** (computed for every candidate window):
- nuclei count, from the nuclei mask, which does not depend on the cell model;
- the 18S neuropil index: 18S signal outside the nuclei mask dilated by 3 µm, also independent of the cell model;
- the share of saturated or empty pixels (artefact screen).

**Baseline-derived features** (training and validation only): baseline cell count; **nuclei without a cell** (nuclei with less than 50% overlap with any cell mask, a missed-cell signal); **anucleate cell area share** (a false-neuropil signal).

**Candidate windows.** Candidate windows lie on a grid inside each stratum. A window is kept only if it lies inside one tile core with the margin of §4.0, is at least 90% inside the stratum and the tissue polygon, and does not overlap an earlier crop. Test crops keep a buffer of one crop width from training and validation crops.

**Test sampling** (no baseline-derived feature is used):
1. Allocation per sample and stratum as set by the precision simulation (default: 2 crops per stratum per held-out sample, plus 2 extra WM crops per sample, 24 in all).
2. Within each stratum, windows are drawn at random within nuclei-density terciles (equal allocation across terciles), after the artefact screen.
3. The inclusion probability of every chosen window is recorded, so an area-weighted estimate can be reported beside the pre-registered unweighted one.

**Training and validation sampling:**
1. **Density-aware:** within each stratum, one crop per density tercile before any repeats.
2. **Cell-type-aware:** where published labels exist, a greedy diversity step makes each round's crops contain at least 30 cells of every broad class present, pooled across strata (a per-stratum target is not reachable with 3–4 crops per stratum, e.g. microglia are about 5% of cells). Labels only weight the choice; they never enter the GT.
3. **Hard examples:** each round adds one training crop per stratum from the top decile of "nuclei without a cell" or the neuropil index, aimed at the failure modes you described. Hard examples never go to validation or test.
4. The RNG seed, candidate counts and the features of every chosen window are written to the manifest.

**Crop size** [L; the pilot confirms, OD-10]: set in microns, converted per platform, at least 256 px (the training bsize) and at most 1,024 px for GUI responsiveness. Every edit rebuilds a full-image overlay and rewrites the `_seg.npy` (`gui/gui.py:1699-1730`) [H, CP §5.2]. Defaults: Xenium training and validation 768 px (163 µm, about 60 cells at P5011's mean of about 2,250 cells/mm²); Xenium test 512 px (109 µm, about 27 cells); MERSCOPE 1,024 px (111 µm).

**Pre-fill** (OD-9):
- Training and validation crops in round 1 get the published final mask crop (`cellpose_masks_tiled.npy[y0:y1, x0:x1]`, relabelled 1..n), which is exactly the ProSeg prior. In later rounds they get the latest model's prediction, made in the production tile context (§4.5).
- A label that the crop edge cuts into several pieces is split: each 8-connected piece gets its own label, so the GUI and the import check see one component per label.
- **Test crops are annotated from blank**, so the GT is not anchored to either model (R2).
- The `_seg.npy` is minimal: `masks`, `outlines` (`masks * utils.masks_to_outlines(masks)`), `filename` (basename only) and `ismanual = zeros(n)`. It has no flows, which keeps autosave fast [H, CP §3.3].
- Optionally, when the export is given a raw baseline prediction for the tile (made before the area filter, on the GPU in HITL-5), the masks that the area filter removed are shown in an overlay PNG for reference. They are not pre-filled.

**Files per crop** (`<crop_id>` = `<pair>_<platform>_<stratum>_r<round>_<nn>`):
- `<crop_id>.tif`: the GUI view image, (3, Y, X) float32. The default view is [DAPI, 18S, ATP1A1/CD45/E-Cadherin], so the annotator sees the boundary stain; the GUI shows up to 3 channels.
- `<crop_id>_seg.npy`: the pre-fill (or empty masks for test crops).
- `context/<crop_id>_context.png` (a downsampled view of the section with the crop box, for orientation) and `context/<crop_id>_filtered.png` (the optional overlay). They sit in a subfolder because the GUI would otherwise cycle into them with A/D and could save stray `_seg.npy` files next to them (§2.3).
- Server side only: `raw/<crop_id>.npy` (uint16, all channels) and `model_input/<crop_id>.<channelset>.npy` (float32, route A).

**Manifest** (`manifest.tsv` + `manifest.json`), one row per crop: crop_id; round; split (train / val / test, assigned at export); pair, sample, platform, species, panel ID; donor fields when available (RNA quality such as DV200, pathology), for per-donor reporting; stratum and its source file sha256; bbox in native pixels (y0, y1, x0, x1) and in µm (from the affine); pixel size and the full affine; image element name; channel map (names → source indices, view order, training order); tile ID, tile and halo bbox, the joint p2/p98, the Cellpose per-channel low/high values and the subsample stride (the normalisation contract); source mask path and crop sha256; source model name and weights sha256; cellpose and torch versions; sampler seed, window features and, for test crops, the inclusion probability; image and raw-crop sha256; annotation-guide version and style.

**Package:** `hitl_<platform>_r<round>[_test].tar` holding crops, `_seg.npy`, `context/`, `manifest.tsv`, `review.tsv` (crop_id, status `reviewed` / `excluded`, reason, minutes, notes), the annotation guide, `SHA256SUMS` and the laptop env file (§7). About 40 crops at roughly 8 MB each make about 0.3 GB [L]. Exports are written under `<hitl_root>/rounds/<round>/export/`, outside every Nextflow `publishDir` (modules publish with `overwrite: true`, `segmentation.nf:4`). Test crops go in their own package.

### 4.2 S2 Annotation guide (`docs/hitl/annotation-guide.md`, versioned; the manifest records its version)

**Rules.**
1. **Dense.** Every cell whose visible soma or nucleus lies in the crop is drawn, including cells cut by the crop edge (draw the visible part). An undrawn cell is learned as background [H, CP §4.2].
2. **What counts as a cell.** A DAPI nucleus with or without an 18S rim. An **anucleate soma** (sectioned above or below its nucleus) is drawn only when its 18S outline is compact, clearly bounded and at least the size of a small nucleus (about 5 µm across); otherwise it is left undrawn (OD-8) [L].
3. **Neurons: soma vs processes** (OD-7; recommended default, confirmed downstream in HITL-6): draw the soma plus the continuous proximal dendrite trunk, only where the 18S signal is unbroken from the soma, up to half a soma diameter and at most 10 µm. The cap keeps a large soma and its trunk inside the 256 px training patch and eval tile: a 30 µm soma is about 141 px, and with a trunk of one soma diameter it would be about 280 px. Do not draw distal or isolated processes or neuropil 18S. Reasons: 2D masks must be single strokes; processes cross other cells' territory and would seed ProSeg with voxels from other cells; ProSeg's compactness prior pulls thin appendages back anyway; ProSeg can still grow a soma prior where the transcripts support it [M/L, CP §6.3].
   **No second annotation pass is needed to compare styles** [L]. A soma-only variant is derived from the same masks, label by label: only masks larger than 150 µm² are opened (a disk of about 2 µm radius, which removes thin trunks and keeps somata); the component holding the most nucleus pixels is kept; and if the opening removes more than half of a mask, the original mask is kept. Small cells (endothelial and pericyte nuclei are often narrower than 4 µm; microglia and small glia) are therefore never eroded, so the style comparison changes only large cells. One model is trained per variant; each is scored against GT derived the same way, so image AP does not by itself pick a style. The choice is made by the §9 style rule on downstream metrics for a training sample (P1212), never on the test set, and the test set is opened only after it. The reverse derivation is impossible, which is why the drawn style is the richer one.
4. **White-matter and other small cells:** outline the DAPI nucleus plus any 18S rim contiguous with it; do not invent cytoplasm. Draw each nucleus of oligodendrocyte rows separately. Draw microglia as the nucleus plus the visible perinuclear soma, without branches. Draw endothelial cells and pericytes as each elongated nucleus plus its rim, never the vessel lumen or wall.
5. **Not cells:** lipofuscin or autofluorescent granules outside a soma (inside a soma they stay part of it), red blood cells, debris, edge-of-tissue smears, and out-of-focus blur with no nucleus.
6. **Ambiguous cases:** split touching nuclei along the DAPI boundary. If you cannot decide within about 10 s, use the "draw only if clear" rule and note the crop in `review.tsv`. Mark a crop `excluded` if more than about 20% of it is a fold, tear, bubble or saturated area, or if its stratum looks wrong (reason `stratum`).
7. **GUI mechanics:** one stroke per mask; no overlaps (the GUI crops new masks); to fix a boundary, delete and redraw; do not run models in the GUI. When a crop is finished, press **Ctrl+S** (even with no edits) and only then set its `review.tsv` status to `reviewed` and fill in the minutes. The import accepts a crop only when both are present (§4.3).

**Time and targets** [L; the pilot measures them]:
- A pre-filled 768 px crop takes about 15–25 min to correct.
- A blank 512 px test crop takes about 18–27 min (a blank 768 px crop about 40–60 min).
- Targets for round 1: at least 600 training ROIs plus about 7 validation crops; the test set's size and per-stratum ROI counts come from the precision simulation (L1 is sparse and may fall short, in which case its criteria are report-only).
- Later rounds add about 600 ROIs each.

### 4.3 S3 `IMPORT` and validation (CPU; `merxen cellpose-hitl-import`)

**Checks** (each failure is listed per crop; nothing is fixed silently):
1. Every file's crop_id and image sha256 match the manifest; a changed image is refused.
2. The `_seg.npy` is read with a restricted unpickler that allows only numpy arrays and dtypes and built-in containers and scalars; `np.load(allow_pickle=True)` with the default unpickler is never used (R13). It must hold `masks` of the crop shape and an integer dtype. The stored `filename` (a laptop path after any GUI save) is ignored.
3. Each label is one 8-connected component: a split label goes back to the annotator, and a component smaller than 20 px is flagged.
4. Masks and `outlines` agree.
5. **Reviewed:** `review.tsv` says `reviewed` **and** the GUI-save keys (`manual_changes`, `model_path`, `normalize_params`) are present. If they disagree the crop is refused: keys without `reviewed` means a partly corrected crop (autosave writes the keys after the first edit), and `reviewed` without keys means the crop was never saved.
6. Crops marked `excluded` are dropped and logged, with the reason.
7. Test crops arrive in a separate package, and their masks are not already present in any training set.

**Diagnostics:** per-crop correction effort (`manual_changes` count, the share of ROIs that are `ismanual`, the minutes from `review.tsv`); agreement between GT and pre-fill (AP@0.5), which is an anchoring diagnostic (R2); on the two doubly annotated validation crops, the AP between the corrected and the blank annotation; ROI counts per stratum and per size class.

**Versioned training set:** `<hitl_root>/trainsets/<hash>/` holds `masks/<crop_id>.tif` (uint16 or uint32), `model_input/…` for each channel set, `split.tsv`, `manifest.json` and the returned `_seg.npy` files archived unchanged. The hash is sha256 over canonical JSON of the per-crop image and mask sha256s, the splits, the guide version and style, and the normalisation contract. The set is built in `.tmp-<uuid>`, renamed into place under flock, made read-only and never deleted, following the [rca] `annotation/store.py` pattern [H, CODE §4.3]. GUI `normalize_params` are ignored. Test masks sit in a sibling `testsets/<hash>/`, which the training code refuses to read.

### 4.4 S4 `TRAIN` (GPU; `CELLPOSE_HITL_TRAIN`; `merxen cellpose-hitl-train`)

**Split** (OD-11, now fixed by the data):
- Only four human Xenium sections exist (P7417, P5822 and P4815 appear only as MERSCOPE in the work logs). By sample: train on P7513 and P1212; test on **P7113 and P5011**, the same held-out donors as the rca plan, so later annotation acceptance stays non-circular.
- By crop: validation crops come from the training samples and keep a one-crop-width spatial buffer from training crops.
- Test crops are never used for any choice.
- Once, in round 1, the chosen setting is also trained on one training donor and validated on the other (both ways). This leave-one-donor-out estimate is reported as the closest proxy for A6 before the test set is opened; it does not change the choice.

**Base and version:** cpsam_v2 from the pinned cellpose (HITL-0). The base weights sha256 must equal `0f1cc3f7…` (K1), or training refuses to start. The model is built with `CellposeModel(pretrained_model=<cpsam_v2 path>, use_bfloat16=False)`, so the base weights load in float32 (K17).

**Parameters** (the paper protocol [M, CP §4.2]; explicit, recorded):

| Parameter | Value | Why |
|---|---|---|
| `learning_rate`, `weight_decay`, optimiser | 1e-5, 0.1, AdamW (always) | paper, issue #1346 |
| `n_epochs` | 100 (grid {100, 300} on validation only) | paper. Both values use the same schedule: 10 warm-up epochs, flat, then 10 halvings over the last 50 epochs; only `n_epochs` > 300 changes it (`train.py:411-421`). 300 epochs triples the iterations |
| `batch_size` | 1 | paper; memory |
| `nimg_per_epoch` | max(8, n_train) | paper minimum of 8, rather than the GUI's max(2, n). Each iteration sees one 256 px patch (§2.3). An area-based count (e.g. n_train × crop area / 256²) with `train_probs` for stratum weights is an optional round-2 grid point if 300 epochs beats 100; `train_probs` is used only when `nimg_per_epoch` ≠ n_train (`train.py:442-448`) |
| `min_train_masks` | 0 | keeps sparse WM and neuropil crops (the default of 5 drops them) |
| `bsize` | 256 | required for SAM (`train.py:360-363`) |
| `normalize` | `{"normalize": False}` (route A), passed as a dict, never a bool | the input is already tile-normalised; a bool mutates the module-level default dict (`train.py:373-379`) |
| `rescale`, `scale_range` | False, 0.5 | native scale; augmentation of ±25% |
| channels | set by the model's channel config; `channel_axis` explicit | at most 3 channels (`train.py:74`) |
| precision | training float32 from float32 weights (`use_bfloat16=False`); inference as in production (`use_bfloat16 = false`) | K17; `nextflow.config:38` |
| seeds | 0, 1, 2: each seeds torch and python **and permutes the training list** with its own RNG; cuDNN non-determinism recorded | `train_seg` re-seeds numpy with the epoch index (K16), so a numpy seed changes nothing. Permuting the list changes which crop receives each position's draws of order, rotation, flip, scale and crop; torch changes layer-drop |

**Model choice** [L]:
- Round 1: the grid (at most {100, 300} epochs × {1e-5, 5e-6} learning rate) runs with seed 0; the best point on validation is then trained with seeds 1 and 2. Later rounds train the chosen point only (3 seeds). The candidate is the median-validation-AP seed.
- The grid runs only for the default channel set and the drawn style. The variants (channel set B1, the derived soma-only style, and a downsampled model if B4 wins on validation) reuse the chosen settings with 3 seeds each, in the final round only.
- **Channel-set rule** (pre-registered): channel set B1 replaces the default only if its pooled validation AP@0.5 is at least 0.02 higher with the paired-bootstrap 95% CI lower bound above 0; otherwise the default (fewer channels) stays.

**GPU scheduling:**
- `CELLPOSE_HITL_TRAIN` and `CELLPOSE_HITL_EVAL` carry a Dwight flock `beforeScript` block like `CELLPOSE_SEGMENT`'s (`dwight.config:86-105`), but take the lock whenever `gpu_process_lock_enabled` is true, not only when `params.cellpose_gpu == "true"` (`:89`), and fail at once if CUDA is not available. Both get `maxForks 1` and, in the `gpu` profile, `--nv` (K12). They queue behind pipeline GPU tasks and never share the A5000.
- Batch-1 training needs about 8.5 GB [M, CP §4.4]. The process checks free GPU memory after taking the lock and fails early rather than running out of memory.
- No training outside Nextflow (K12).

**Outputs:** the model file (about 1.2 GB), train/validation loss arrays (never compared across cellpose versions, issue #1493), a `train.json` with every parameter, the seed and the list permutation, the timings and the peak GPU memory.

### 4.5 S5 `EVALUATE`

**S5a: image level** (`CELLPOSE_HITL_EVAL`, GPU).
- **Production context.** For each validation or test crop, the model runs through the shared per-tile function (C7) on the crop's enclosing global tile of the full-image grid (recorded tile size; a tile allow-list), with the production thresholds and per-tile filters. The final area filter is applied explicitly with the production bounds (`filter_labeled_mask_by_area`, as in `pipeline.py:654-682`), and the result is cropped. All outputs go to task scratch only. Calling `run_tiled_cellpose` on the tile itself is not used: it would re-grid the tile (a 6144 px image becomes a 5888 px tile plus a second tile, `cellpose.py:141-209`), change the normalisation windows, lack the neighbour state its stitching expects, and delete the output paths it is given (`cellpose.py:510-521`). One tile takes about 55 s (82 min for 90 tiles on P7513).
- **Baseline.** The stock baseline is stock cpsam_v2 run the same way, at the same commit, in the same env. The published mask crop is the round-1 pre-fill and a drift diagnostic: the stock re-run should match it (IoU-matched agreement of at least 0.99, R11). A shortfall is investigated (torch 2.13.0 then vs 2.14.0 now, cuDNN) but does not invalidate the comparison, because both arms share the env.
- **Raw scores.** Raw model output, before the area and eccentricity filters, is scored as well, so filter losses are visible (K8).
- **Quick path.** During rounds, a quicker crop-only evaluation on `model_input` arrays scores the validation set. It lacks the halo context, so it is used only to compare models with each other.

**Metrics:**
- AP@0.5 and AP@0.75 (AP@0.9 reported), plus precision, recall and F1 at 0.5, pooled (TP/FP/FN summed over crops) and per crop.
- **Border rule** [L]: matching uses every GT and predicted label in the crop, but TP, FP and FN are counted only for objects whose centroid lies at least 5 µm inside the crop edge, so cells cut by the edge do not dominate small crops.
- Aggregated Jaccard index and boundary scores [H, CP §5.1].
- **Per region:** each stratum.
- **Per cell class:** (i) label-free GT size classes (< 40, 40–150 and > 150 µm² [L]) as a neuron/glia proxy; (ii) diagnostically, the broad class of the matched published cell (IoU ≥ 0.5; else "unmatched"). Class (ii) inherits the current segmentation's labels, so it is reported but never gated.
- **Failure-mode counts:** FP per mm² in WM crops and in crops in the top tercile of the label-free neuropil index; recall of GT cells > 150 µm²; missed GT cells that have a nucleus.
- **Uncertainty:** 95% CIs from a paired cluster bootstrap over crops, resampling crops within each sample × stratum cell, for every model difference.
- **Human ceiling:** 4 test crops (2 per held-out sample) re-annotated from blank by you at least 2 weeks later (or by a second annotator, OD-18). AP between the two annotations is the ceiling, reported beside each model [M, CP §5.1; Cellpose FAQ]. The MERSCOPE fairness crops get their own ceiling (§3.4).
- **Comparators** on the same crops. Their settings are chosen on validation and fixed in the freeze record before the test set is opened (§9):
  - stock cpsam_v2 (baseline);
  - B0: the vendor XOA segmentation (`original_seg` in the latest zarr), rasterised in the crop; report only;
  - B1: stock model with `[DAPI, 18S, ATP1A1/CD45/E-Cadherin]`, which the docs call off-distribution for the stock model;
  - B1′: stock model with 18S and ATP1A1/CD45/E-Cadherin combined into one channel (evaluation harness only; production would need a new option);
  - B2: cellprob −5 vs 0 vs −2;
  - B3: area bound 400 vs 800 µm², eccentricity 0.99 vs off;
  - B4: stock model with `diameter` (e.g. 60 px, a 2× downsampling inside Cellpose);
  - B*: the best combination of B2 × B3 (× B4 if it helps) on validation, which A8 must beat;
  - the fine-tune with each channel set and style, and with retuned thresholds (flow and cellprob tuned on validation only).

**S5b: downstream after ProSeg, in same-commit arms** (HITL-6).
- **Arms.** Every arm is a normal `main.nf` run at one commit and one env, in its own outdir, with the same samplesheet row and params. Arms differ only in `--xenium_cellpose_model`. The published outdir is never the comparator, because it was made with older code and another torch (K14).
- **Stock arm:** stock cpsam_v2 re-runs Xenium Cellpose (about 82 min GPU). Copying the published mask instead does not work: the reuse rule needs the mask, cellprobs and CSV together, and without the CSV Cellpose re-runs and first deletes the copied mask (K4). Copying all three would keep the old torch's mask. The stock arm's mask doubles as the full-section drift check against the published mask.
- **Shared inputs**, copied byte-identical into every arm (sha256 recorded before and after): the Xenium nuclei mask (reused, K4, so D1 and D2 use the same reference); the pair-level alignment outputs (otherwise VALIS runs again; registration does not depend on the Xenium cell mask [H, CODE §5.2]); and one MERSCOPE side (OD-24). The HITL-6 dry run checks that the copies pass the current reuse guards (49ebb40 seeding-affine check, b208800 enrichment source-image check, the transforms sidecar) and measures the smallest MERSCOPE subset that must be copied.
- **ProSeg replicate:** ProSeg has no seed option (K15). On P1212, a replicate arm copies the drawn-style arm's persistent Xenium segmentation (mask, cellprobs and CSV, all written at this commit), so Cellpose is reused and ProSeg and the downstream stages run again. For each D-metric the margin is m = max(the preset floor in §9, 2 × |arm − replicate|). The rule is pre-registered; the values go into the freeze record.
- **Arm list:** P1212: drawn-style finalist, its replicate, soma-only finalist; P7113 and P5011: stock and the chosen candidate; optionally P7513 for information. That is 7 arms (9 with P7513), plus one per held-out pair for B* if it is to be adopted.

| Metric | Direction | Source |
|---|---|---|
| Share of transcripts assigned to cells after ProSeg; Cellpose seeding rate (P7513 X published 62.80%) | report; a move of more than 5 points triggers review | ProSeg output; `pipeline.py:686-755` seeding |
| Share of **nuclear transcripts** assigned (transcripts inside the unchanged DAPI nuclei mask that end up in a cell) | should rise | nuclei mask + ProSeg assignment |
| Share of cells overlapping ≥ 2 nuclei (doublet proxy) and cells with 0 nuclei (neuropil / fragment proxy) | should fall | nuclei mask |
| **Depth-matched MECR**: each cell's transcripts are downsampled to a common depth before binarising (cells below it are excluded); raw MECR is reported too. MerXen's MECR is binary co-detection (`analysis/mecr.py` docstring), so raw MECR rises with transcripts per cell and would favour smaller masks | must not worsen beyond the margin | existing MECR stage (`mecr_enabled`) plus a depth-matching step in `scripts/acceptance/hitl_downstream.py` [M, CP §5.1; arXiv 2606.09675] |
| **Per-class negative-marker purity**: for cells of each broad class, the share of their transcripts that belong to other classes' markers (the MECR reference markers) | must not worsen beyond the margin | `hitl_downstream.py` |
| Cells per mm² per stratum; area distribution; masks lost to the area filter | report | segmentation stats |
| Annotation coverage and marker plausibility: confident broad coverage, canonical-marker plausibility, WHB–SEA agreement ([rca] criteria H7, H9, H3, or legacy equivalents if rca is not on `main`) | must not worsen beyond the margin | annotation pipeline |
| Cross-platform concordance with MERSCOPE: soft broad-composition JSD ([rca] H1 estimator), per-stratum density ratio, class depth profiles ([rca] H12) | must not worsen beyond the margin; improvement reported | COMPARE, cortical depth, rca |
| WM oligodendrocyte density | must not fall by more than 5% (or the margin, if larger) | annotation + strata |

### 4.6 S6 Model registry and pipeline parameters

**Registry** (`cellpose_model_store`, default `/srv/storage/MerXen/cellpose_models/`, OD-16):
- Layout: `<store>/<build_hash>/{model, model.json, train.json, eval/}`, built in `.tmp-<uuid>`, renamed under flock, read-only and never deleted (the [rca] store pattern; reuse `merxen.annotation.store` if it is on `main` by then, otherwise a minimal copy to unify later).
- `build_hash` = sha256 of canonical JSON over the base-weights sha256, the cellpose and torch versions, the training-set hash, the split, the training parameters, the seed and its list permutation, the normalisation contract and the channel list [H pattern, CODE §4.3].
- `model.json` records: platform, species, channel names in order, pixel size, tile size, normalisation route, annotation style and guide version, the thresholds and filters used in its evaluation, the model file sha256, the licence note (R14), and the status (`candidate` / `accepted` / `rejected`, with the evaluation summary path).

**Pipeline parameters** (inert when unset, so legacy config JSON stays byte-identical and `-resume` keeps working, following the [rca] `AnnotationSettings.groovy:4-11` pattern):
- `CellposeConfig.pretrained_model: str | None` (Python default `cpsam_v2` from HITL-0) and `model_sha256: str | None`, passed through `build_cellpose_model` (`cellpose.py:34-53`), which from HITL-0 always passes `use_bfloat16` (K17). The Nextflow config JSON carries the new keys only when a run sets them.
- `params.xenium_cellpose_model` and `params.merscope_cellpose_model` take a registry hash or path, with optional samplesheet columns of the same names for per-row overrides. They are merged into `baseConfig.cellpose` (`main.nf:2091-2101`) only when set.
- Optional per-platform overrides, also inert when unset: `<platform>_cellpose_cellprob`, `<platform>_cellpose_flow_threshold`, `<platform>_cellpose_diameter`, `<platform>_cellpose_final_max_area_um2`, `<platform>_mask_max_eccentricity`. They are explicit parameters, never read automatically from `model.json`; the preflight warns when they differ from the values the model was evaluated with.

**Preflight** (refuses to run): the model file exists; its sha256 matches `model.json`; the row's channels, pixel size (±1%), platform and species match `model.json`; the model status is `accepted`, or `candidate` with `--allow_candidate_model`; the tile-size candidates are pinned (§4.0). This closes K3's silent fallback.

**Reuse guard and baseline safety** (K4):
- A `cellpose_model_identity.json` sidecar (model name, sha256, effective CellposeConfig, cellpose and torch versions) is written beside the persistent mask. The reuse rule (`pipeline.py:570-611`) compares it the way the transforms sidecar is compared (`pipeline.py:249-358`). On a mismatch it refuses to run; it never overwrites.
- v1 requires a **separate outdir** for every arm (route 5.3a [H, CODE §5.3]; OD-23), with the shared inputs of §4.5 S5b copied, never symlinked, so no downstream stage mutates the baseline.

**Provenance** (HITL-0 for the stock model, HITL-1 for custom models):
- The model identity is written **only when Cellpose actually runs**. A reused mask keeps its sidecar. A legacy mask without one gets a sidecar with status `unverified`, whose evidence field points at the task log that shows the model (e.g. `.command.log:87-90` of `work/58/e6e470…` for P5011 X); it is never stamped with the current env's versions. The same applies to the transforms sidecar pattern, which today writes the current config when the file is missing (`pipeline.py:588-590`).
- The segmentation stats JSON and the segmentation registry entry in `merxen_schema` carry the identity copied from the sidecar. `writer_versions` (`io/spatialdata_schema.py:100`) keeps its meaning: the environment that wrote the zarr, not the model.
- HITL-0 records the torch version at run time and HITL runs assert it equals the version in `model.json` (R11).

---

## 5. Pipeline integration

- **New files:** `workflows/cellpose_hitl.nf` (entry workflow, run with `--hitl_step export | import | train | evaluate`); `workflows/modules/cellpose_hitl.nf` (`HITL_CROP_EXPORT`, `HITL_IMPORT`, `CELLPOSE_HITL_TRAIN`, `CELLPOSE_HITL_EVAL`, `HITL_REPORT`); `src/merxen/segmentation/hitl/{strata,crops,export,seg_npy,ingest,trainset,train,evaluate,registry,report}.py`; `src/merxen/cli/run_cellpose_hitl.py` (`merxen cellpose-hitl-*`); `scripts/acceptance/hitl_simulation.py` and `hitl_downstream.py`; `docs/stages/cellpose-hitl.md`; `docs/hitl/annotation-guide.md`; `envs/environment.cellpose-gui.yml` (laptop only, §7).
- **Shared files are changed only at marked hook points** (`// hitl-hook:C<n>` / `# hitl-hook:C<n>`, each asserted to occur exactly once, as in [rca] `test_rca_hooks.py`): C1 `CellposeConfig` fields (`config.py:16-27`); C2 `build_cellpose_model` (`cellpose.py:49-53`); C3 the reuse guard (`pipeline.py:570-611`); C4 `baseConfig.cellpose` and the per-platform overrides (`main.nf:2091-2113`, `:334-366`); C5 the preflight; C6 the Dwight lock blocks and `gpu` profile entries for the two new GPU processes (`dwight.config`, `nextflow.config`); C7 the shared per-tile function in `run_tiled_cellpose` (`cellpose.py:610-675`).
- **rca branch:** the rca integration branch has hook points H1–H6, H10 and H11 in `main.nf` (markers `rca-hook:H<n>`). On that branch `baseConfig.cellpose` (C4) sits at about `main.nf:2133`, between H6 (about `:2003`) and H5 (about `:3405`), so the hooks do not overlap; `main` is merged into rca as usual. HITL-0 changes the conda env hash, which also restarts `-resume` for rca runs, so it is scheduled between runs of both efforts.
- **Pins:** a test pins the sha256 of the resolved `CELLPOSE_SEGMENT`, `CELLPOSE_NUCLEI_SEGMENT` and `PROSEG_SEGMENT` script text; another asserts that a legacy row's segment config JSON is byte-identical before and after each milestone; a third runs `run_tiled_cellpose` on a synthetic image (monkeypatched model) before and after C7 and asserts identical masks, cellprobs and stitching stats.
- **Metro map:** the new processes live in a separate optional entry workflow, so they go into `OMITTED_PROCESSES` in `tests/test_workflows/test_metro_map.py` with that reason, unless you would rather have a side line on the map.
- **Mouse and new panels:** strata come from a `StratumSource` interface (cortical-depth GeoJSON, region-polygon GeoJSON, [rca] tile regions). Channel names and species come from the row and `model.json`. A new stain set or panel whose channel names differ is refused until it has its own evaluation (G5).

---

## 6. Compute, storage and time

| Item | Resource | Estimate | Basis |
|---|---|---|---|
| Crop export, per sample (~20 crops) | CPU; reads ~20 halo tiles (6144² × 4 ch uint16, ~300 MB each) | minutes to under 1 h | [L] |
| Pilot (round 0) | you | 2–3 h | [L] |
| Round 1a: training and validation crops | you | ~25 pre-filled crops × 15–25 min = 6.3–10.4 h | [L] |
| Round 1b: test crops | you | 24 × 512 px from blank at ~18–27 min = 7–11 h | [L] |
| Ceiling and anchoring re-annotation | you | 4 test crops from blank again (~1.2–1.8 h); 2 validation crops from blank (~1.3–2 h) | [L] |
| Rounds 2–3 | you | ~10 crops × 15–25 min = 2.5–4.2 h each | [L] |
| MERSCOPE fairness (optional) | you | 5 × 1,024 px from blank plus 2 re-annotated: ~5–7 h | [L] |
| **Your total** | you | ~20–33 h to the end of round 2; up to ~45 h with round 3 and MERSCOPE | sum of the rows above |
| One training run | A5000, ~8.5 GB, batch 1 | iterations = `n_epochs` × max(8, n_train). Round 1 (n_train ≈ 20): 2,000 or 6,000 iterations, ~10–17 or ~30–50 min at an assumed 0.3–0.5 s per iteration | docs benchmark: 200 iterations in 27 s (A100) to 364 s (RTX 4060) [M, CP §4.4]; A5000 not benchmarked [L] |
| Training per round | round 1: 4 grid points with seed 0, then 2 more seeds of the chosen point; later rounds the chosen point × 3 seeds; final-round variants × 3 seeds each | round 1 ~2.5–4 GPU-h; rounds 2–3 ~1–5 GPU-h each (grows with n_train); final-round variants ~2–15 GPU-h; ~7–30 GPU-h in all | [L]; running the full grid × 3 seeds every round would cost ~6 GPU-h in round 1 and ~12–15 GPU-h by round 3 |
| Evaluation in production context | ~55 s per 6144 tile | ~20–30 min per model or comparator for ~20–30 tiles; ~4–8 GPU-h for the final evaluation (3 seeds × variants, B0–B4, B*) | P7513 trace 82 min / 90 tiles [H] |
| Downstream arms | per arm: Xenium Cellpose ~82 min GPU (not in the replicate arm), ProSeg ~2.6 h (32 threads), then enrichment and downstream | ~0.5–1 day wall per arm; 7 arms (§4.5 S5b), ~4–7 days wall in sequence | CODE §5.4 [H] + [L] |
| MERSCOPE re-segmentation (only if OD-24 option b, or HITL-n) | ~6.7 h Cellpose per section, plus ProSeg | – | CODE §5.4 [H] |
| Storage | model ~1.2 GB each; crops ~10 MB each; one arm ~240 GB at P7513's size (MERSCOPE side 189 GB: segmentation 142 GB and latest zarr 47 GB; Xenium ~42 GB; alignment 5 GB) | 7 arms ~1.7 TB if all are kept; `/srv/storage` had 3.2 TB free on 2026-09-30 and is ext4 (no copy-on-write copies). Arms run in sequence; once an arm's metrics JSON is written you can archive or remove its outdir (the pipeline never deletes it), so the peak is about two arms (~0.5 TB). The HITL-6 dry run measures the smallest MERSCOPE subset | `du` and `df` on 2026-09-30 [H] |

**Developer effort** [L, about 30% contingency]: HITL-0 1.5–2 days; HITL-1 3–4; HITL-2 4–5; HITL-3 3–4; HITL-4 3–4; HITL-5 4–5; HITL-6 3–4 plus compute; HITL-7 1–2. Total about 22–30 days.

---

## 7. Laptop setup (for the annotator)

- `envs/environment.cellpose-gui.yml` pins `cellpose[gui]` to the **same version** as the lock (4.2.1.1 after HITL-0) and the same numpy major version, because `_seg.npy` is a pickled dict and must round-trip; this is a precaution, not a documented issue [L, CP §5.3]. A test (like `test_env_lock_sync.py`) asserts that the pin equals the lock's cellpose version.
- This deviates from the pyproject-only rule (Agents.md) on purpose: the laptop env installs only the GUI, not MerXen. The alternative is a `[hitl-gui]` extra, which would pull all of MerXen's dependencies onto the laptop (OD-20).
- The GUI runs on CPU; no GPU is needed for editing. Unpack the package, open any crop and cycle with A/D; the `context/` folder holds the orientation images. Which laptop you use (OS, Apple Silicon or not, memory) sets the env file and the crop-size test (OD-26). An Apple Silicon laptop could train through MPS, but training still runs in the pipeline (§3.2).

---

## 8. Milestones (PR-sized; each merges with defaults unchanged)

Branches: short-lived `feature/hitl-<n>-<slug>` from `main`, one PR each into `main` (OD-21). No integration branch is needed, because every milestone is inert until a run sets a model. Commit prefixes follow Agents.md. Before each commit run `ruff check . --fix`, `ruff format .` and `mypy src/`; before each push run `pytest` (the pre-push hook). Tests that need real data or a GPU are marked `@pytest.mark.slow`.

### HITL-0 `[bugfix]`: pin Cellpose and torch, name the model, record it (on `main`; recommended now, independent of HITL; 1.5–2 days)
- **Scope:** `pyproject.toml` pins cellpose (OD-2: recommended `==4.2.1.1`) and bounds torch to one minor release that the lock, the container (CUDA 12.6) and the conda env all resolve; lock regenerated; the env lock-hash header refreshed (`scripts/update_env_lock_hash.py`). Add `CellposeConfig.pretrained_model = "cpsam_v2"`; `model_type` is kept, deprecated and warned about. `build_cellpose_model` always passes `use_bfloat16` (K17); decide whether the Python default should match production (False). The nuclei config names its model the same way. Record the model name, the weights sha256 and the cellpose and torch versions in the identity sidecar and the stitching stats, only when Cellpose runs; legacy masks get `unverified` sidecars (§4.6). Fix `docs/configuration.md:117`. Side fix: the `gpu` profile lacks `--nv` for `CELLPOSE_NUCLEI_SEGMENT` (`nextflow.config:629-646`).
- **Cost:** a new conda env hash restarts `-resume` under the conda profile, for rca runs too, so schedule it between runs. Published masks do not change, because production already runs 4.2.1.1 with cpsam_v2 (K1).
- **Tests:** the config accepts `pretrained_model`, and `model_type` alone warns; `build_cellpose_model` passes `pretrained_model` and `use_bfloat16=False` when `model_type` is None (monkeypatched `CellposeModel`); the identity is written on a run and not on a reuse; a legacy mask gets `unverified`; env/lock sync.
- **Exit:** a fresh env reports cellpose 4.2.1.1 and the pinned torch and loads cpsam_v2 with sha256 `0f1cc3f7…`; CI uses the same versions; a re-run on one P7513 Xenium tile is compared with the published mask crop (IoU-matched ≥ 0.99 expected; a shortfall is investigated and recorded, since the published masks used torch 2.13.0).

### HITL-1 `[feature]`: inert model plumbing and the shared per-tile function (3–4 days)
- **Scope:** hooks C1–C5 and C7; the model and override parameters (§4.6); the preflight; the model-identity sidecar and reuse guard; the separate-outdir rule; the script and config pins; docs.
- **Tests:** legacy config JSON byte-identical; `run_tiled_cellpose` output identical before and after C7 on a synthetic image; a missing model file, wrong sha256, wrong channels or wrong pixel size each refuse; a sidecar mismatch refuses without deleting anything; each override is applied only when set; hook markers occur once.
- **Exit:** full `pytest` green; the P7513 dry-run config is unchanged.

### HITL-2 `[feature]`: S1 crop export, S2 guide, pilot package (4–5 days + your pilot time)
- **Scope:** `StratumSource` for cortical-depth GeoJSON (human, depth-mode pieces, the positive WM rule) and region polygons; the strata overlay; label-free and baseline-derived window features; test and training sampling; the normalisation contract; the `_seg.npy` writer with edge-label splitting; the manifest; the package with `context/`; `HITL_CROP_EXPORT`; the annotation guide; the laptop env.
- **Tests:**
  - synthetic image + GeoJSON → strata polygons (a `mask_qc_only` piece and tissue beyond a line's end are in no stratum) and windows inside one tile core with the margin;
  - **the exported model input equals the array production feeds the model** for the same pixels (float tolerance), on synthetic 6144 px, 5888 px and small tiles, including the recorded low/high values;
  - non-zero-offset affine (VZG2-like) round trip; a missing channel raises an error;
  - `_seg.npy` keys, a load through cellpose's own `io` path (`--mask_filter _seg.npy` reads `masks`), and a C-shaped label cut by the edge becomes two labels;
  - the soma-only derivation removes a synthetic trunk, keeps the soma, and leaves a synthetic 3 × 15 µm endothelial nucleus and a small glial mask unchanged;
  - real data (slow): the mask labels under each transcript in the crop equal that transcript's `cell_id` in `transcripts_for_proseg.csv`; the durable latest-zarr image equals the image Cellpose read (the source zarr, while it still resolves) on a few tiles.
- **Pilot:** 3 crops on P7513 Xenium, the 512 / 768 / 1024 px responsiveness test (§3.3) and the strata overlays of all four Xenium samples.
- **Exit:** you have annotated the pilot; the style rules (OD-7), crop sizes (OD-10) and minutes per crop are recorded in the guide.

### HITL-3 `[feature]`: S3 import, versioned training sets, precision simulation, pre-registration (3–4 days)
- **Scope:** `HITL_IMPORT`; checks and diagnostics (§4.3), with the restricted unpickler; training and test set stores; `scripts/acceptance/hitl_simulation.py` (§3.3); `docs/acceptance/cellpose-hitl-preregistration.md`, which copies §9 with your approved thresholds, the test design and the margin rule **before any training**; the round-1b test export after it.
- **Tests:** each validation failure (changed sha256, split label, wrong shape, `reviewed` without GUI keys, GUI keys without `reviewed`, excluded crop, test crop in a training set, a pickle that loads a non-allowed class); hash stability and sensitivity; a crash leaves no partial set; concurrent builds build once; the simulation on a fixture.
- **Exit:** round-1a package exported, annotated and imported; the simulation run; the pre-registration merged; the test package exported and annotated.

### HITL-4 `[feature]`: S4 training and S6 registry (3–4 days + GPU)
- **Scope:** `CELLPOSE_HITL_TRAIN` with the unconditional lock block and `--nv` (hook C6); `merxen cellpose-hitl-train`; the registry; `model.json`.
- **Tests:** a monkeypatched `train_seg` receives the §4.4 parameters (explicit normalize dict, `min_train_masks=0`, bsize 256) and a network built with `use_bfloat16=False`; different seeds give different training-list permutations; the base-sha256 refusal; registry hash, atomic build and read-only mode; a workflow string test that the lock block is present without the `cellpose_gpu` condition and `maxForks` is 1; slow GPU smoke test: 2 crops, 2 epochs.
- **Exit:** a round-1 candidate is registered with its training record; peak GPU memory and seconds per iteration are measured and added to §6.

### HITL-5 `[feature]`: S5a image-level evaluation, report and freeze record (4–5 days + GPU)
- **Scope:** `CELLPOSE_HITL_EVAL` (production-context and quick paths); comparators B0–B4 and B*; metrics with the border rule, the stratified cluster bootstrap, size classes, the human ceiling; `HITL_REPORT` (HTML with per-stratum overlays of GT, baseline and candidate).
- **Tests:** AP on hand-made mask pairs (known TP/FP/FN) and the border rule; the stratified bootstrap on a fixture; a drift check that reports a mismatch; the evaluation code refuses to read `testsets/` unless a committed freeze record names the candidate.
- **Exit:** rounds 1–2 (and 3 if the stop rule allows) are done; one finalist per style variant and B* are chosen on validation; the test set stays sealed.

### HITL-6 `[feature]`: S5b downstream evaluation, style choice, test scoring (3–4 days + ~4–7 days compute)
- **Scope:** the arm recipe (§4.5 S5b) and its dry run (reuse guards, storage, the MERSCOPE subset); `scripts/acceptance/hitl_downstream.py` (depth-matched MECR, negative-marker purity, margins) and the report section.
- **Order:** (1) on P1212, the two style finalists and the replicate run downstream; the margins are computed and the style is chosen by the §9 rule; (2) `docs/acceptance/cellpose-hitl-freeze.md` is committed (§9); (3) the test set is opened once and every arm is scored (A-criteria); (4) the stock and candidate arms run downstream on P7113 and P5011 (D-criteria).
- **Tests:** the metrics on a synthetic small SpatialData; depth matching on a fixture where raw MECR favours the smaller masks; the outdir preflight refuses the baseline outdir.
- **Exit:** the A- and D-criteria of §9 are evaluated; the published outdirs are unchanged (sha256 of the persistent masks and latest zarr metadata before and after).

### HITL-7 `[docs]` / `[minor]`: acceptance decision and opt-in use (1–2 days)
- **Scope:** mark the model `accepted` or `rejected` in the registry; document the opt-in (`--xenium_cellpose_model <hash>`) in `docs/stages/segmentation.md` and `docs/configuration.md`; record the platform-model difference in COMPARE outputs.
- **Default:** stays stock unless you decide otherwise in a separate `[feature]` PR.
- **Exit:** you have signed off the acceptance summary.

### HITL-n (later, each its own PR set)
Mouse strata and a mouse Xenium model when mouse Xenium data arrive (cortex, hippocampus, midbrain; manual region polygons or [rca] M6); a matched MERSCOPE fine-tune if the fairness check (§3.4) calls for it; Xenium 5K / custom-panel evaluation; the QuPath route (OD-3); a second annotator; a nuclei fine-tune; a production option for a combined channel (B1′) if it wins.

---

## 9. Acceptance criteria (pre-registered in HITL-3, before any training)

**Rules.**
- The thresholds, the test design (crop count, size and allocation), the bootstrap, the margin rule, the tuning grids, the channel-set rule and the style rule are fixed in `docs/acceptance/cellpose-hitl-preregistration.md` after the precision simulation and before any training. They may be tightened, but loosening needs your written approval in the acceptance PR (the rca rule). If the simulation shows that the pooled 95% CI half-width of ΔAP@0.5 would exceed 0.03 at the default test size, the test set grows before pre-registration; the thresholds do not loosen.
- **Freeze record** (`docs/acceptance/cellpose-hitl-freeze.md`, committed before the test set is opened): the candidate (model hash, seed set, style, channel set), its thresholds and filter bounds, B* and its settings, every other comparator's settings, and the D-margins from the replicate. Everything in it is chosen on validation or on P1212.
- **One look.** The test set is opened once and scores every arm at the same time: stock, B0 (report), B1, B1′, B4, B* and the candidate (all 3 seeds). Nothing is scored on test before that, not even the stock model.
- **After a failure:** stop and report. With your written approval, at most one fresh test set may be annotated from blank in the held-out samples (buffered from the old crops); the old test set then becomes validation, and a new freeze record precedes the new look. The cheap fix is adopted only through the pre-registered B* arm on the same look, never chosen from test results.
- Held out: whole samples P7113 and P5011 (test crops annotated from blank). Validation is for choices.
- All thresholds below are [L] first-run targets for you to confirm (OD-22).

| ID | Criterion (held-out test crops, pooled over both samples, unless stated) | Threshold |
|---|---|---|
| A1 | Pooled AP@0.5, candidate minus stock (production context, filtered output) | ≥ +0.05, and the stratified cluster-bootstrap 95% CI lower bound > 0 |
| A2 | Pooled AP@0.75, candidate minus stock | ≥ +0.03 |
| A3 | Per-stratum non-inferiority of AP@0.5, pooled over both samples | 95% CI lower bound of the difference ≥ −0.05 in every stratum the simulation marks as gated; report-only for the others |
| A4 | Recall at IoU 0.5 of GT cells > 150 µm² (large-neuron proxy) | ≥ stock + 0.10 |
| A5 | FP per mm² in the WM test crops and in test crops in the top tercile of the label-free neuropil index | upper 95% CI bound of candidate / stock ≤ 1.10, if gated by the simulation; else report-only |
| A6 | Generalisation: AP@0.5 gain on held-out samples vs on validation (training samples) | ≥ 50% of the validation gain |
| A7 | Seed stability (seeds vary the data order, §4.4): AP@0.5 spread over the 3 seeds | ≤ 0.02, and A1 holds for the worst seed |
| A8 | Beats the cheap baselines: candidate AP@0.5 vs B* (frozen on validation) | ≥ B* + 0.02 |
| A9 | Human ceiling (double annotation) and each model's gap to it | report only: a model can agree with one annotator's first pass more than that annotator's second pass does without being overfitted |
| D1 | Nuclear transcripts assigned after ProSeg (P7113, P5011 Xenium) | ≥ stock arm − m |
| D2 | Share of cells overlapping ≥ 2 nuclei | ≤ stock arm + m |
| D3 | Depth-matched MECR; per-class negative-marker purity | MECR ≤ stock arm + max(5% of stock, m); purity ≥ stock arm − m |
| D4 | Confident broad annotation coverage; marker plausibility | ≥ stock arm − max(0.02, m); ≥ stock arm − max(0.01, m) |
| D5 | Soft broad-composition JSD, Xenium vs MERSCOPE (shared MERSCOPE side) | ≤ stock arm + max(0.01, m) |
| D6 | WM oligodendrocyte density | ≥ stock arm × 0.95, or stock arm − m if that is lower |
| D7 | Total transcripts assigned; masks lost to the area filter | reported; a move of more than 5 points, or a rise in filter losses, needs a written explanation |

m is the metric's margin from the replicate (§4.5 S5b): max(the floor shown, 2 × |arm − replicate| on P1212).

**Style rule** (applied in HITL-6 on P1212, before the test set is opened): choose the variant with the lower depth-matched MECR if the two differ by more than its margin m; otherwise the variant with the higher negative-marker purity if they differ by more than its margin; otherwise keep soma + proximal trunk.

**Accept rule:** A1, A2, A3 (gated strata), A4, A5 (if gated), A7 and A8 on the pooled test set. Per held-out sample: an AP@0.5 point gain of at least +0.03, and A6. D1–D6 on each held-out pair against its same-commit stock arm. If the candidate fails and B* passes A1–A5 against stock on the same look, B* may be adopted (it then needs its own D-criteria arms). Any exception needs your written approval.

**MERSCOPE:** stays on stock unless a MERSCOPE model passes the same table on MERSCOPE test crops.

---

## 10. Risk register

| ID | Risk | L / I | Mitigation | Where |
|---|---|---|---|---|
| R1 | Overfitting to few crops or one annotator's style | H / M | Whole-sample hold-out; paper hyperparameters with a ≤ 4-point grid; seeds that vary the data; leave-one-donor-out report; A6, A7 | S4, §9 |
| R2 | Annotator bias: anchoring to the pre-fill, drift between rounds, one annotator | H / M | Blank test crops annotated before any model output is seen; two validation crops annotated both ways; a pre-fill agreement diagnostic; a versioned guide; double annotation; optional second annotator (OD-18) | S2, S3 |
| R3 | Domain shift across donors, staining batches, XOA versions and future panels | M / H | Held-out samples; per-sample and per-donor reporting (RNA quality, pathology); `model.json` channel and pixel checks; a new panel or stain set needs its own evaluation | S5, S6 |
| R4 | GPU contention on the single A5000 (another process used 6.4 GB on 2026-09-30) | M / M | Nextflow-only training and evaluation with an unconditional Dwight flock, `maxForks 1`, a free-memory check; no GUI or ad hoc training on the server | S4 |
| R5 | Frame mistakes: aligned vs native, VZG2 offsets, `list(sdata.images.keys())[0]` (`pipeline.py:445`) picking the wrong element | M / H | Named native element; full affine; refuse aligned elements; transcript-label frame test | S1, HITL-2 |
| R6 | Normalisation mismatch between crops and production tiles, including the strided percentiles and tile-size changes under GPU load | H / H | Model input built by the production functions on the exact halo tile, with an equality test on three tile sizes; pinned tile size; a new channel set re-reads the tile | §4.0 |
| R7 | Silent model fallback or loose weight loading (K3) | M / H | Preflight file and sha256 checks; a fixed-crop load check | S6 |
| R8 | Filters delete the improved cells (400 µm² bound, eccentricity) | H / M | Raw vs filtered evaluation; B3; per-model bounds as explicit parameters (OD-14) | S5 |
| R9 | A different cellprob distribution changes ProSeg's prior weighting and the meaning of the −5 threshold | M / M | Validation-tuned thresholds; the downstream D-criteria; cellprob histograms in the report | S5 |
| R10 | Baseline destroyed by an in-place rewrite, or a new model silently reused as the old mask (K4) | M / H | Separate outdirs; identity sidecar; refuse on mismatch; sha256 before and after in HITL-6 | S6 |
| R11 | Version drift (lock 4.1.1 vs env 4.2.1.1; torch 2.10 / 2.13 / 2.14), or a future release | H / H | HITL-0 pins; the drift check; versions in the build hash; runtime torch assert; same-commit arms | HITL-0, S5 |
| R12 | Unfair platform comparison (Xenium fine-tuned, MERSCOPE stock) | M / M | MERSCOPE fairness crops against their own ceiling; provenance in COMPARE; D5 under both Xenium settings; a matched fine-tune if needed | §3.4 |
| R13 | `_seg.npy` is a pickle | L / M | Restricted unpickler; checksum against the manifest; only packages you return | S3 |
| R14 | Licence: weights derive from CC-BY-NC training data | L / M | Weights kept in the lab store, not in the repo; licence recorded in `model.json` | S6 |
| R15 | Evaluation circularity: class labels and sampling weights come from the current segmentation | M / M | Test crops sampled only on label-free features; label-free size classes and strata gate; class-matched metrics are diagnostic only | S1, S5 |
| R16 | Dangling `source_spatialdata.zarr` links (K13) | M / L | Read from the durable latest zarr's native element; an equality test | S1 |
| R17 | Annotation time overruns | M / M | Pilot timing; per-round budget; the stop rule | §3.3 |
| R18 | Conflicts with the rca integration branch in `main.nf` / `config.py`, and the shared env hash | M / L | Marked hooks away from rca's; merge `main` into rca as usual; HITL-0 scheduled between runs | §5 |
| R19 | Neurons with process-including masks are split by the per-tile connectivity filter or trimmed by ProSeg compactness | M / M | Single-stroke rule; trunk cap; the derived soma-only variant scored downstream against the drawn style | S2, S5 |
| R20 | Code-version confound: comparing a current-code run with outputs made by older code and torch | H / H | Same-commit arms; shared, byte-identical nuclei mask, alignment and MERSCOPE side | S5b |
| R21 | Too few test crops for the gates, or a second look at the test set | H / H | Precision simulation before pre-registration; stratified cluster bootstrap; non-inferiority or report-only per stratum; freeze record; one look | §3.3, §9 |
| R22 | ProSeg run-to-run noise decides a downstream criterion | M / M | Replicate arm; margins from it | S5b |
| R23 | Storage exhaustion from arm outdirs | M / M | Arms in sequence; you archive or remove each after its metrics; the dry run measures the MERSCOPE subset | §6 |

---

## 11. Sources

- Cellpose 4.2.1.1 docs: https://cellpose.readthedocs.io/en/latest/ (gui.html, train.html, models.html, outputs.html, settings.html, faq.html, benchmark.html); source https://github.com/MouseLand/cellpose/tree/v4.2.1.1/cellpose (models.py, train.py, transforms.py, io.py, cli.py, metrics.py, vit.py, gui/{gui,io,guiparts,menus}.py); releases https://github.com/MouseLand/cellpose/releases/tag/v4.2.1.1 and https://github.com/MouseLand/cellpose/releases/tag/v4.0.7 (MPS training fix, PR #1278); issues #1206, #1291, #1335, #1346, #1353, #1424, #1474, #1476, #1493 (https://github.com/MouseLand/cellpose/issues/<n>); paper script https://github.com/MouseLand/cellpose/blob/main/paper/cpsam/train_subsets.py. Line-level references: [CP §2–§5] and inline `cellpose/…` references.
- Pachitariu, Rariden & Stringer 2025, Cellpose-SAM, bioRxiv 10.1101/2025.04.28.651001. Pachitariu & Stringer 2022, Cellpose 2.0, Nat Methods, doi 10.1038/s41592-022-01663-4.
- Jones et al. 2025, ProSeg, Nat Methods, doi 10.1038/s41592-025-02697-0; https://github.com/dcjones/proseg at e7df1eace923ce4c6ec70b2c597c5d126aa3db88 (README limitations and Cellpose initialisation; `src/main.rs` options; `src/sampler/voxelcheckerboard.rs`).
- 10x CG000750 segmentation technical note: https://cdn.10xgenomics.com/image/upload/v1710785020/CG000750_XeniumInSitu_CellSegmentation_TechNote_RevA.pdf
- Communications Biology 2025, doi 10.1038/s42003-025-08518-6 (18S in human cortex). Marco Salas et al. 2025, bioRxiv 10.64898/2025.12.07.692889 (extrasomatic RNA). arXiv 2606.09675 (segmentation evaluation in spatial transcriptomics; MECR).
- Tools: https://github.com/BIOP/qupath-extension-cellpose; https://github.com/MouseLand/cellpose-napari (issue #61).
- MerXen code and data: [CODE §0–§6], with file:line references throughout this plan; task logs `/srv/storage/MerXen/work/58/e6e47088f09f52cb75b0f3744d8687/.command.log` (P5011 X, model load at :87-90, area filter) and `/srv/storage/MerXen/work/d2/2bb63695bfa15cdb697b604cb6f624/.command.log` (P7513 X area filter); `/srv/storage/MerXen/results/P7513/xenium/segmentation/cellpose_stitching_stats.json`; commits 2579e9d, 49ebb40, fd539cd, b208800.

---

## 12. Review log (rev1)

The review of rev0 (2026-09-30) was checked against the installed Cellpose 4.2.1.1 source, MerXen at `6daa255`, the task logs and the published data ($H/PROGRESS_plan_reviser.txt). Every finding was confirmed; two fixes were adjusted.

| # | Finding | Change |
|---|---|---|
| 1 | Downstream baseline made with older code | Same-commit arms (§4.5 S5b, K14). Adjusted: the stock arm re-runs Cellpose, because a copied mask is not reusable without its CSV (K4), and copying the CSV would keep the old torch. Nuance: for the human Xenium sections 2579e9d drops the same transcripts as before; MERSCOPE's ProSeg input and many downstream stages did change |
| 2 | "Reviewed" accepted partly corrected crops | `reviewed` AND GUI keys; refuse on disagreement (§4.3) |
| 3 | Seeds did not vary data order or augmentation | Seeds permute the training list (K16, §4.4) |
| 4 | Test set could be looked at twice | Freeze record, one look, fresh test set after a failure (§9) |
| 5 | Test set too small for per-stratum gates | Precision simulation; 24 × 512 px default; stratified cluster bootstrap; non-inferiority or report-only; pooled class target (§3.3, §4.1, §9) |
| 6 | Test sampling used baseline-derived features | Random within stratum × density tercile; label-free neuropil index (§4.1) |
| 7 | S5a did not reproduce production | Shared per-tile function C7, tile allow-list, explicit area filter, scratch outputs, core margin (§4.0, §4.5) |
| 8 | Normalisation contract imprecise | Strided percentiles stated; production functions on the halo tile; re-read the tile for a new channel set (K6, §4.0) |
| 9 | Two bf16 traps | K17; explicit `use_bfloat16`; training model in float32; tests (HITL-0, HITL-4) |
| 10 | Provenance on reused masks; torch unpinned | Identity only when Cellpose runs; `unverified` legacy sidecars; torch bound and assert (§4.6, HITL-0). The rev0 research note's "deleted" env survives on SSD2 |
| 11 | Soma-only opening damaged small cells | Label-by-label, > 150 µm² only, keep-original rule, endothelial test (§4.2) |
| 12 | Style rule and D3 favoured smaller masks | Depth-matched MECR and negative-marker purity (§4.5, §9) |
| 13 | Zero margins without a noise model | ProSeg replicate arm and margins (K15, §4.5, §9) |
| 14 | Cheap baselines incomplete | B0, B1′, B4, B*; B2 small-mask evidence (§0, §4.5) |
| 15 | Native scale vs 256 px patch | Trunk cap; B4 and a conditional downsampled variant (§4.0, §4.2, OD-13) |
| 16 | WM stratum picked up grey matter | Depth-mode pieces, positive WM rule, overlay review and veto (§4.1) |
| 17 | GUI cycled into context PNGs | `context/` subfolder (§4.1) |
| 18 | Separate-outdir recipe gaps | Nuclei mask, alignment and MERSCOPE copies; storage budget (§4.5, §6) |
| 19 | GPU lock conditional | Unconditional lock, CUDA check, `--nv` (§4.4) |
| 20 | Compute underestimated; schedule claim wrong | §6 recomputed; schedule corrected; patch-exposure note (§4.4) |
| 21 | Annotation time optimistic | §6 recomputed |
| 22 | MERSCOPE fairness compared raw APs | Each platform against its own ceiling (§3.4) |
| 23 | Train and test GT differ in style | Anchoring measured. Adjusted: on two validation crops rather than a test crop, so no test crop is shown a pre-fill (§3.3) |
| 24 | A9 premise wrong | A9 report only (§9) |
| 25 | Smaller fixes | §0 / §3.2 made consistent; rca hooks listed (§5); restricted unpickler; edge-label splitting; env hash and rca `-resume`; durable-zarr test; donor fields; leave-one-donor-out report; channel-set rule |

---

## Open decisions for the user

Each decision has a recommended default, which the plan uses unless you choose otherwise. "Blocking" means the decision is needed before the named milestone.

| ID | Question | Options | Recommended | Blocking |
|---|---|---|---|---|
| OD-1 | Do HITL-0 (pin cellpose and torch, name cpsam_v2, record weights) now, as a data-integrity fix on `main`? It affects MERSCOPE too | now / with HITL / never | **Now**, separately from HITL, between runs | – |
| OD-2 | Base model and pins | cellpose `==4.2.1.1` / `>=4.2.1.1,<4.3` / cpsam on 4.1.1; torch: a one-minor bound in pyproject plus a runtime record and assert / record only | **cpsam_v2, cellpose `==4.2.1.1` (a stated exception to loose pins, because the conda env resolves pyproject on build day), torch bounded to one minor that the container supports, runtime assert, sha256 check** | HITL-0 |
| OD-3 | Annotation editor | Cellpose GUI / QuPath BIOP extension (whole-slide context, global normalisation, but it recommends continuing from custom models) | **Cellpose GUI**; QuPath as fallback | HITL-2 |
| OD-4 | Where training runs | pipeline (Nextflow, GPU lock) / GUI | **Pipeline**; the GUI edits only (§3.2) | HITL-4 |
| OD-5 | Normalisation | A: tile-exact crops, `normalize=False` / B: per-crop training + `tile_norm_blocksize` at inference (changes production for both platforms) | **A** | HITL-2 |
| OD-6 | Channels | training `[DAPI, 18S, 0]` with B1 `[DAPI, 18S, ATP1A1/CD45/E-Cad]` as a variant chosen by the channel-set rule; GUI view shows ATP1A1 / view DAPI+18S only / AlphaSMA-Vimentin variant | **Train both channel sets on the same annotations; view with ATP1A1** | HITL-2 |
| OD-7 | Neuron annotation style | soma + proximal trunk (≤ half a soma diameter, ≤ 10 µm) / soma only / soma + all visible processes | **Draw soma + capped proximal trunk**; derive soma-only label by label; choose by the §9 style rule on P1212 downstream | HITL-2 (drawn style); HITL-6 (final choice) |
| OD-8 | Anucleate somata | draw if compact and ≥ ~5 µm / never / always | **Draw if compact and ≥ ~5 µm** | HITL-2 |
| OD-9 | Pre-fill | train/val pre-filled, test blank / all pre-filled / all blank | **Train/val pre-filled; test blank** | HITL-2 |
| OD-10 | Crop sizes | Xenium train/val 512 / 768 / 1024 px; test 512 / 768 px | **768 px train/val, 512 px test**, confirmed by the pilot | HITL-2 pilot |
| OD-11 | Split | train P7513 + P1212, test P7113 + P5011 / the reverse | **Train P7513 + P1212; test P7113 + P5011** (only four human Xenium sections exist; P7417, P5822 and P4815 are MERSCOPE only) | HITL-3 |
| OD-12 | Platform scope | Xenium only + MERSCOPE fairness crops / matched MERSCOPE fine-tune now / one joint model | **Xenium only + 5 MERSCOPE fairness crops against their own ceiling** | HITL-3 |
| OD-13 | Scale | native / downsampled via `diameter` | **Native**, with B4 as a comparator and a downsampled fine-tune only if B4 beats native stock on validation | HITL-5 |
| OD-14 | Filters and thresholds for custom models | explicit per-platform overrides tuned on validation / keep global / auto from `model.json` | **Explicit overrides, tuned on validation, recorded in `model.json`** | HITL-5 |
| OD-15 | Mouse strata source and timing | manual region polygons (new `region` role) / [rca] M6 tile regions; when mouse Xenium arrives / MERSCOPE mouse now | **Manual polygons; wait for mouse Xenium** | HITL-n |
| OD-16 | Model store location | `/srv/storage/MerXen/cellpose_models/` / SSD / elsewhere | **`/srv/storage/MerXen/cellpose_models/`** | HITL-4 |
| OD-17 | Fine-tune the nuclei model as well? | no / later | **No** (the nuclei mask is the independent downstream check) | – |
| OD-18 | Annotators | you only, with 4 test crops re-annotated for the ceiling / add a second annotator | **You only + self re-annotation**; second annotator if available | HITL-3 |
| OD-19 | Sharing weights | lab only / publish (CC-BY-NC derived) | **Lab only** | – |
| OD-20 | Laptop env | standalone `envs/environment.cellpose-gui.yml` (a stated deviation) / `[hitl-gui]` pyproject extra | **Standalone env file with a lock-sync test** | HITL-2 |
| OD-21 | Branching | per-milestone PRs to `main` / an integration branch | **Per-milestone PRs to `main`** (every milestone is inert) | HITL-1 |
| OD-22 | Acceptance thresholds A1–A9, D1–D7 and margin floors | as proposed / tighter / looser | **As proposed**; confirm before HITL-3 merges | HITL-3 |
| OD-23 | Output strategy for custom-model runs | separate outdir per arm / model-keyed paths in the same outdir (much larger change) | **Separate outdir per arm** | HITL-6 |
| OD-24 | MERSCOPE side shared by the arms | (a) copy the published MERSCOPE side into every arm (old code: its tables still carry the Blank codewords that 2579e9d now drops, but it is identical in every arm) / (b) re-run MERSCOPE once per pair at the HITL-6 commit (~6.7 h Cellpose plus ProSeg per section) and copy that into every arm / (c) re-run it in every arm (adds ProSeg noise) | **(a)**; switch to (b) if the dry run shows the copy fails the current reuse guards, or if you want current-code absolute cross-platform numbers | HITL-6 |
| OD-25 | Test-set design | 24 × 512 px (2 per stratum per sample + 2 extra WM per sample) / 10 × 768 px (rev0) / larger, as the simulation requires | **24 × 512 px, adjusted by the precision simulation before pre-registration** | HITL-3 |
| OD-26 | Annotation laptop | which machine (OS, Apple Silicon or not, memory) | tell us before HITL-2; it sets the laptop env file and the crop-size test | HITL-2 |

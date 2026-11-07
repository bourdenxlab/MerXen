# 10x Atera support: build stage and downstream plan

> **Status:** revision 4, 2026-11-04. This is a plan only; no code has changed. Decisions marked
> *Recommended* in §3 and §10 are the defaults the plan adopts until the user overrides them. Milestone 1
> (§1) covers the build stage. Everything after it in §8 is sequenced but has not started. Revision 4
> applies the fidelity, scale, docs-conformance and completeness reviews. Each accepted or rejected point
> is listed in the Review log (Appendix A). **User decision 2026-11-05 (metadata retention):** the store
> keeps all vendor metadata verbatim; privacy rules apply only to what is committed to git and to
> human-facing output (§3.11 D13, Appendix A). **AT-4 (2026-11-05):** the builder and BUILD plumbing are
> implemented; the deviations from this plan are listed in Appendix A ("AT-4 implementation notes"), and
> the AT-4 review decisions in Appendix A ("AT-4 review"). **AT-5a (2026-11-05):** the public dataset
> builds through Nextflow in 8.5 minutes at 11.7 GB peak anonymous memory and passes every §1.1
> criterion (status after §1.1 item 7; measured numbers in §6.1; decisions in Appendix A, "AT-5 pilot").
> Milestone 1 is met. **AT-5b (2026-11-05):** the ProSeg benchmark ran on the crop and on the densest
> 2 mm tile; AT-9 goes ahead as an opt-in tiled route, and a single whole-section run is ruled out
> (§8 AT-5 and AT-9, Appendix A "AT-5b ProSeg benchmark"). **AT-6 (2026-11-06):** the native
> (vendor-segmentation) route runs end to end through Nextflow on the crop and on the full public section,
> from the AT-5 source stores to clustering and MENDER, and passes every exit check (§1.4, §6.2, §8 AT-6,
> Appendix A "AT-6 exit runs"). **AT-9 (2026-11-06):** the opt-in ProSeg route
> (`segmentation_source=native_proseg`: vendor masks as the prior, 2 mm label-owned tiles, a stitched
> `reseg` layer) runs end to end on the crop and on the full public section through MENDER on both
> layers (full section: 30 tiles, 91,431 cells, 8 h 41 min on a busy host, every check passed), and
> the short ProSeg schedule is confirmed on the densest tile (§6.2, §8 AT-9, Appendix A "AT-9
> implementation and crop run", "AT-9 full-section run" and "AT-9 final review").

*Names and paths (2026-11-04).* The only dataset used here is the public 10x Atera whole-transcriptome
dataset of a human hippocampus section (atera-1.1.1, fresh frozen); its name is not written here, and the
slow tests find it as the one extracted bundle under `$MERXEN_ATERA_RAW_ROOT`. Data locations
are written as `.env` variables. `$MERXEN_ATERA_RAW_ROOT` is new (AT-0); `$MERXEN_EVIDENCE_ROOT` already
exists. Earlier MerXen runs used as scale baselines are anonymised as Xenium A/B and MERSCOPE C–G.

Two kinds of Atera text identify samples. They must never appear in committed text (code, test fixtures,
docs, commit messages, PR titles and bodies), and logs and plot titles print them only through
`merxen.privacy.to_alias`. The data products are different: the built store keeps them verbatim (D13):

- Run directory names: `<INST_SN>__<SLIDE>__<ROI>__<date>__<time>`, plus the parent
  `<USER_ID>/<date>__<time>__<RUN_NAME>`.
- Manifest fields: `run_name`, `slide_name`, `slide_id`, `region_name`, `instrument_sn`, the UUIDs and
  `panel_config.customer`.

- **Effort:** Atera support (code `AT`).
- **Date and base:** `origin/main` `c5c1659`, worktree branch `feature/atera-support`.
- **Why:** Atera is a 10x whole-transcriptome panel (18,049 genes) that produces about 1.64 billion
  transcripts per section. MerXen handles XENIUM and MERSCOPE today. Every downstream stage reads the
  `source_spatialdata.zarr` that `BUILD_SPATIALDATA` writes.
- **Evidence:**
  - Format probes and audits under `$MERXEN_ATERA_RAW_ROOT/<dataset>/inspection/`.
  - The user-supplied 10x docs (`OFFICIAL_10X_DOCS.md`), which are authoritative for format semantics.
  - Full-section streaming checks over both masks, the CFM and the cell summary, run during review.
  - Repo audits were done at `c5c1659`.
- **Confidence tags:**
  - `[H]` verified on the real dataset or in code at `c5c1659`
  - `[M]` measured, but indirect or extrapolated
  - `[L]` judgement or estimate
  - `UNVERIFIED` means nobody has checked it yet.
- **Abbreviations:**
  - QV: calibrated Phred Q-score
  - CFM: `cell_feature_matrix.zarr.zip`
  - WTA: whole-transcriptome assay
  - MIQ: mask image quantification
  - DFO: distance-from-object
  - PR426: scverse/spatialdata-io#426
  - ps: pixel size, 0.21249222 µm.

---

## 1. Summary and scope

### 1.1 What milestone 1 delivers

`merxen build-spatialdata` with `platform: ATERA` turns one extracted Atera ROI bundle into
`source_spatialdata.zarr`. The store follows the same element contract as a Xenium build (§4.1).
Segmentation, enrichment and the later stages can open it with only the consumer changes listed in §5.

**Milestone 1 is done when every item below holds** (AT-0 to AT-5 in §8):

1. **The public dataset builds.**
   - The Nextflow `BUILD_SPATIALDATA` task builds the public dataset with `analysis_mode=atera` and a
     per-row `stop_stage=build_spatialdata`.
   - It stays within the resource envelope in §6, enforced by the build's own RSS watchdog.
   - Peak RSS, wall time and per-element output size are recorded under `$MERXEN_EVIDENCE_ROOT`.
   - Killing the task part-way and retrying it produces a full rebuild, never reuse of the half-written
     store.
2. **The store validates.**
   - `validate_merxen_schema(deep=False)` passes.
   - A new streaming deep check passes:
     - `transcript_id` is unique per tile;
     - per-tile row counts reconcile with `grid_number_objects` after the QV filter;
     - `original_assignment` ⊆ shape IDs;
     - table IDs = shape IDs.
3. **Points reproduce the vendor matrix exactly.**
   - Count points per cell × feature, keeping rows with `qv ≥ 20` and an assigned cell.
   - The result must equal the vendor CFM `X` over every column whose vendor all-QV `gene_counts` is
     above 0: 638,849,843 counts, of which 638,693,512 are gene counts [H, target totals].
   - The CFM excess is asserted to be exactly the 84 counts in the four `DeprecatedCodeword_*` columns that
     have no transcripts at all (`codeword_counts` = 0, grid `gene_counts` = 0; AT-3 review). It is a
     vendor matrix quirk, recorded as a vendor difference.
   - Every sum is taken in int64 (`X.sum(dtype=np.int64)` or the integer CSR data); a float32 sum is off
     by tens of counts at this size.
   - The count runs as a streaming COO accumulation over all features, so it also checks the control
     mapping.
4. **Labels match the vendor cell summary exactly.** The full-section streaming pass takes about 85–103 s
   at under 3 GB RSS [H].
   - `cell_labels` pixel count × ps² equals `cell_summary.cell_area` for every cell (largest difference
     ≤ 1e-4 µm²).
   - Per-label centroids at **(i + 0.5)·ps** equal `cell_summary` centroids within 0.005 µm.
   - Every `nucleus_labels` pixel lies on its owning cell's `cell_labels` label: 0 violations, mapped via
     `polygon_sets/0/cell_index[label−1]`.
   - The largest labels are 91,431 (cells) and 90,238 (nuclei), and no label is empty.
   - Mask-derived shape centroids, simplified at the default 0.25 px, are within 0.01 µm of
     `cell_summary`. This is the half-pixel check (§3.5).
5. **Downstream code can find the store's metadata.**
   - `sd.read_zarr(store)` succeeds.
   - `platforms.find_um_per_px(store, "ATERA") == 0.21249222`.
   - The enrichment and viewer-cache transform lookups gain their ATERA arm with the native route in
     AT-6, not in milestone 1: they map Cellpose mask pixels to microns, and the ATERA pixel convention
     of those masks is decided there (AT-4 review).
   - The segmentation loader arm (`DatasetConfig` Literal) is not in milestone 1; it lands in AT-6.
6. **The store keeps the vendor metadata complete** (user decision 2026-11-05, D13). A fast test and the
   acceptance script assert that:
   - every vendor metadata and report file is a byte-identical sidecar under `atera/`, with a stable name,
     and its original file name, size and SHA-256 are recorded in attrs;
   - `attrs["merxen_atera"]` carries the full `experiment.spatial` (run, slide, region, instrument, UUIDs,
     versions, panels, counts), every metrics summary column and the `panel_config.json` header;
   - the metadata inside the vendor zarr stores (every `.zattrs`/`.zgroup`/`.zarray` member) and the
     OME-XML of every morphology TIFF are kept verbatim under `atera/`, and the vendor table data the
     table has room for is kept (nucleus centroids, `de_*` statistics, category colours);
   - nothing committed (fixture, tests, evidence names) holds a real sample, run, slide or instrument
     name, and logs name the sample by its alias.
7. **Existing platforms are unaffected.**
   - Rendered XENIUM and MERSCOPE build JSON is byte-identical to `main`.
   - Every existing process `script:` text matches its pinned copy, so task hashes are unchanged.

**Status (AT-5a, 2026-11-05): all seven items hold on the public dataset.** The Nextflow task built the
store, then `scripts/acceptance/atera_build.py` checked it as re-read from disk. Its report is
`$MERXEN_EVIDENCE_ROOT/atera_at5/full/acceptance_full.json`, with the trace, report, timeline and logs
beside it. After the AT-5 review the acceptance script gained checks (Appendix A, "AT-5 review"); the
store passes all eleven criteria (`atera_at5/rerun/acceptance_full_v2.json`), and so does a rebuild at
the reviewed revision with the new resource defaults (`atera_at5/full_r2/`).

| Item | Result |
|---|---|
| 1. Build | `BUILD_SPATIALDATA`, one ATERA row with `stop_stage=build_spatialdata`, 16 CPUs: 8 min 24 s; watchdog peak 11.7 GB anonymous RSS (cap then 120 GB). Sizes in §6.1. A crop build killed with SIGKILL after it had written its points, images and labels exited 137. Nextflow retried it once (256 GB reservation): the retry removed the partial store and its mask folder, built from scratch and passed acceptance. Rebuilt after the review with the 64 GB reservation and 48 GB cap: 9 min 22 s, 11.65 GB peak anonymous RSS, the same build fingerprint |
| 2. Schema | Shallow schema passes. Streaming deep check: 1,409,601,228 points rows = raw QV ≥ 20 rows; IDs unique per tile; every per-tile count reconciles over 34 non-empty tiles and 409 files |
| 3. Matrix | Points total 638,849,843 = vendor total on the columns with transcripts; 638,693,512 gene counts; 0 mismatched entries. The vendor total is 638,849,927; the excess is exactly 84 counts, all in four `DeprecatedCodeword_*` columns without transcripts (24 + 21 + 14 + 25) |
| 4. Labels | 91,431 cells and 90,238 nuclei, no empty label; largest area difference 7.2e-5 µm²; largest label centroid difference 0.0014 µm; 0 containment violations; every label pixel equals the vendor `masks/1` / `masks/0`; one shape per cell, indexed by its label; shape centroids within 0.0046 µm (median 0.00035 µm) |
| 5. Metadata lookups | `sd.read_zarr` opens every element; `find_um_per_px(store, "ATERA") == 0.21249222` |
| 6. Vendor metadata | 10 byte-identical sidecars, 5,167 vendor zarr metadata members, 5 OME-XML headers, 49 `experiment.spatial` fields and 103 metrics columns in attrs; the fingerprint recomputed from the bundle matches the marker. The table has `instance_id` = vendor row + 1, X equal to the vendor gene columns (638,693,512 counts), every vendor `obs` column, the `cell_summary` values (nucleus centroids included), the vendor PCA/UMAP, 270 `de_*` columns and 4 colour lists |
| 7. Existing platforms | The XENIUM and MERSCOPE goldens and process script pins are unchanged; only the ATERA build JSON goldens follow the AT-5 cap change (§6.1) |

### 1.2 Out of scope for milestone 1

Each item is sequenced in §8:

- Running SEGMENT, ENRICH, QC, VISUALIZE or the analysis stages on Atera. The "native segmentation"
  route that lets them run is AT-6, the next milestone (done; §1.4). §5 lists every downstream change.
- ProSeg on Atera. The AT-5b benchmark has run; ProSeg comes in AT-9 as an opt-in tiled route.
- H&E registration.
- Pairing Atera with XENIUM or MERSCOPE.
- Ingesting the 3D DAPI stack.
- Atera "preview" bundles in Xenium-v4 layout. Milestone 1 only detects and refuses them (§3.10).

### 1.3 Headline facts

| # | Fact | Tag |
|---|---|---|
| 1 | **No released reader opens an Atera bundle.**<br>• spatialdata-io 0.6.0 `xenium()` fails on the first file it opens (`experiment.xenium`).<br>• PR426 is open, unreviewed and unreleased.<br>• MerXen pins `spatialdata-io>=0.6,<0.7` (`pyproject.toml:24`). | [H] |
| 2 | **Scale.**<br>• 1,640,508,930 transcripts in 35 tiles of 2000 µm.<br>• 1,409,401,472 are gene transcripts with QV ≥ 20: 86.0% of gene transcripts, 85.9% of all transcripts.<br>• About 150× a typical Xenium section. | [H] |
| 3 | **Same geometry as Xenium.**<br>• Pixel size 0.21249222 µm.<br>• `masks/homogeneous_transform` = diag(1/ps) to float32 precision, with no offset.<br>• Mask grid = morphology grid (42,548 × 61,369).<br>• DAPI vs nucleus mask: best integer shift (0, 0).<br>• **Pixel-corner convention:** pixel i covers [i·ps, (i+1)·ps). Transcripts map to pixels by `floor`, and vendor centroids equal (mean index + 0.5)·ps. | [H] |
| 4 | **Vendor matrix rule.**<br>• The CFM counts only features with QV ≥ 20 that are assigned to a cell (official docs).<br>• No transcript with QV < 20 is assigned among 1.40 B rows checked.<br>• A QV ≥ 20 filter therefore reproduces the matrix. | [H] |
| 5 | **The masks are the true segmentation; the polygons are display simplifications** (official docs).<br>• Cell mask label = cell index + 1 (official docs; exhaustively verified).<br>• Nucleus mask label = nucleus-polygon row + 1 (empirical, exhaustively verified on 90,238 nuclei). The docs state the label rule for cell masks only.<br>• Nucleus pixels are 100% inside their owning cell. | [H] |
| 6 | **Near-whole-transcriptome panel.**<br>• 18,049 genes and 6,628 control features.<br>• Several downstream stages densify or loop per gene and break or run for many hours at this size (§5). | [H]/[M] |
| 7 | **Sample-identifying metadata.** `experiment.spatial`, the `metrics_summary_*.csv` header and file name, `analysis_summary_*.html` and `panel_config.customer` all carry run, slide, ROI or serial identifiers, or customer PII. The store keeps them verbatim; committed text and human-facing output never show them (D13). | [H] |

### 1.4 AT-6 status: the native route end to end

**Status (2026-11-06): the AT-6 exit runs pass on the crop and on the full public section.** Each ran
as one Nextflow row (`analysis_mode=atera`, `segmentation_source=native`, `mender_enabled=true`,
`stop_stage=mender`) from the AT-5 source store through `NATIVE_LATEST`, ENRICH, VIEWER_CACHE, MIQ, QC,
VISUALIZE, spatial gene analysis, clustering and MENDER. pciSeq, GASTON, standalone MapMyCells, MECR,
transcript spatial patterns, cortical depth and distance from object are off or refused for ATERA rows
(user decision). The runs found three bugs, each fixed with a test (Appendix A, "AT-6 exit runs").
Evidence: `$MERXEN_EVIDENCE_ROOT/atera_at6/` (run scripts, samplesheets, traces, reports, timelines,
`verify_{crop,full}.json`, `stage_resources.json`, `disk_{crop,full}.json`, the check scripts). Stage
times, memory and disk are in §6.2.

| Check | Crop (3,013 cells, 40.6 M transcripts) | Full section (91,431 cells, 1.41 B transcripts) |
|---|---|---|
| Source store unchanged (inode, size, mtime of every file; SHA-256 of every metadata file and 200 random payloads) | 927 files, 1.48 GB unchanged; 143 payloads made read-only (0664 → 0444) | 3,352 files, 38.8 GB unchanged; 1,543 payloads made read-only |
| Latest store shares no metadata inode with the source | 185 payload files hardlinked, 0 metadata | 1,905 payload files (38.2 GB) hardlinked, 0 metadata |
| Viewer reads every transcript (`points/transcripts` rows of the latest store) | 40,642,374 = source = `n_points_written` | 1,409,601,228 = source = `n_points_written` |
| Native label check (corner convention, float32) | passed, 0 mismatches | passed, 0 mismatches; 638,849,843 assigned rows; `overlaps_nucleus` agreement 1.0 |
| ENRICH gene counts per cell = vendor matrix row sums (int64) and vendor `transcript_counts` | 0 cells differ; 18,894,687 | 0 cells differ; 638,693,512 |
| ENRICH all-feature counts = vendor genes + controls | short by 3 counts in 3 cells, all `DeprecatedCodeword` | short by 84 counts in 84 cells, all `DeprecatedCodeword` (the AT-5 vendor quirk) |
| QC: transcript total, assigned total, registration | 40,642,374; 18,894,687 (after the float32 fix); pass, density ratio 7.2 | 1,409,601,228; 638,693,512; pass, density ratio 4.8 |
| Density figures bin every transcript | section pass bins 40,642,374 of 40,642,374 | section pass bins 1,409,601,228 of 1,409,601,228 (409 of 409 files) |
| Clustering | 3,012 cells × 15,479 genes, 2,000 HVGs (`cell_ranger`), PCA `mask_var` | 91,381 cells × 18,049 genes, 2,000 HVGs, GPU |
| Nucleus metrics via `cell_instance_id` | 2 two-nucleus cells; `nucleus_count` equals the join for every cell | 332 multi-nucleus cells (up to 5 nuclei); `nucleus_count` equals the join for every cell; 10 spot-checked cells (2–4 nuclei): count equal, largest and summed nucleus area within 0.5% of the nucleus polygons |
| MENDER | 15 outputs | 15 outputs (after the env pin) |

---

## 2. Atera bundle format and differences from Xenium

### 2.1 Bundle layout

There is one output folder per ROI, named `<INST_SN>__<SLIDE>__<ROI>__<date>__<time>`. It sits under
`output/customer/<USER_ID>/<date>__<time>__<RUN_NAME>/` (official docs). Each ROI has its own µm frame,
so two ROIs from one slide are unrelated datasets.

The public download ships an `*_outs.zip` (48.86 GB; every member STORED) plus separately prefixed
companions: `<dataset>_transcripts.zarr.zip`, `_cells.zarr.zip`, `_cell_feature_matrix.zarr.zip`,
`_he_image.ome.tif`, `_he_imagealignment.csv`, `_keypoints.csv` and `_analysis_summary.html`. The nested and
standalone `transcripts.zarr.zip` are the same size (15,627,165,982 B) [H].

| File | Content | Tag |
|---|---|---|
| `experiment.spatial` | JSON manifest, the analogue of `experiment.xenium`.<br>Useful fields:<br>• `pixel_size` 0.21249222, `z_step_size` 3.0<br>• `analysis_sw_version` `atera-1.1.1.0`<br>• `panel_wta.{panel_design_id, panel_num_targets, panel_organism}`<br>• `segmentation_stain`<br>• `num_transcripts` 1,638,575,672 (genes only), `num_transcripts_high_quality` 1,409,401,472<br>• `images.*` / `explorer_files.*` paths<br>**Identifying fields:** `run_name`, `run_start_time`, `slide_name`, `slide_id`, `region_name`, `instrument_sn`, and the `*_uuid` fields (analysis, experiment, cassette, roi, well, calibration) | [H] |
| `transcripts.zarr.zip` | zarr v2, `RnaDataset` major 6. See §2.2 | [H] |
| `cells.zarr.zip` | `CellSegmentationDataset` major 6, minor 3. See §2.3 | [H] |
| `cell_feature_matrix.zarr.zip` | Zipped AnnData zarr. See §2.4 | [H] |
| `csc_cell_feature_matrix.zarr.zip` | The same matrix in CSC layout, for Explorer, with different var names (`feature_ids`, `feature_types`). **Not needed**: CSR→CSC of the 3 GB table is trivial | [H] docs |
| `binned_transcripts.zarr.zip` | Explorer density bins as a zip of nested per-gene zips (`gene/{ix}.zip/grid/{X},{Y}/{data,indices,indptr}`).<br>Root attrs: `origin`, `grid_size`, `gene_counts`, `gene_names`, `grid_keys`, `grid_key_names`, `name=BinnedTranscripts`, versions.<br>Bin size and QV filtering are undocumented. `ZipStore` cannot open the nested zips in place. **Not ingested; not recommended for now** | [H] docs; not inspected |
| `morphology_2d/chNNNN_<name>.ome.tif` | Multi-file OME-TIFF, one channel per file, 16-bit, JPEG-2000, 1024² tiles, 8 pyramid levels at 2× steps.<br>**The channel set varies by run:**<br>• default: DAPI, 18S (2 files);<br>• with the Cell Segmentation Staining Reagents: DAPI, ATP1A1/CD45/E-Cadherin, 18S, alphaSMA/Vimentin (4 files; this dataset).<br>File-to-channel mapping is in each file's OME-XML `TiffData` (FileName/UUID → FirstC). `experiment.spatial` names only `ch0000_dapi.ome.tif` | [H] |
| `morphology_3d/ch0000_dapi_3d.ome.tif` | (15, 42548, 61369) ZYX uint16, PhysicalSizeZ 3.0 µm, 21.2 GB. No `PositionZ`; the mapping from plane to transcript z is UNVERIFIED | [H] |
| `overview_scan.png`, `overview_scan_fov_locations.json` | Overview image; FOV locations in overview pixels | [H] docs |
| `morphology_fov_locations.json` | 92 FOVs in µm (same frame as transcripts and cells).<br>Measured here: FOV width 1,029.28 µm (4,843.8 px), pitch about 1,010.7 µm (about 88 px overlap).<br>The docs give 4,819 px with 113 px overlap (§2.6) | [H] data / docs conflict |
| `panel_config.json` | `spec_version` 1.1.1, `payloads` (a list; Xenium's `gene_panel.json` has a `payload` dict).<br>Payload: `chemistry`, `customer{email,name}` (**PII for custom panels**), `designer`, `panel{design_id hWTA_v1, num_gene_targets 18049}`, `targets[18196]` with `codewords` and `info.gene_coverage` | [H] |
| `metrics_summary_<SN>_<SLIDE>_<ROI>….csv`, `analysis_summary_….html`, `qc_images.html` | QC metrics and vendor reports. **The file names and the metrics header** (`run_name`, `instrument_sn`, `slide_name`, `region_name`, `slide_id`) **identify the sample** | [H] |
| `diffexp_genes/` | `gene_expression_graphclust.csv`; `gene_expression_azimuth.csv` (human WTA only) | [H] docs |
| H&E companions (separate download) | `he_image.ome.tif`: (3, 45694, 34152) uint8 RGB, 0.27378 µm, 9 levels, uncompressed, 6.5 GB.<br>`he_imagealignment.csv`: 3×3 affine, **H&E px → morphology px**.<br>`keypoints.csv`: the 8 keypoints are reproduced to within 0.2–3.9 px | [H] |

### 2.2 `transcripts.zarr.zip`

**Root attributes and arrays:**
- `coordinate_space=refined-final_global_micron`, `spatial_units=micron`, `number_rnas=1,640,508,930`,
  `number_genes=24,677`, `codeword_count=53,385`.
- `gene_names`, `gene_index_map`, `codeword_gene_mapping`, `codeword_gene_names`, and `kit_info` (one entry
  per kit; this dataset has one).
- Root arrays `gene_category[24677, 9]` and `codeword_category[53385, 9]`, both bool, with column names in
  each array's `.zattrs` `column_names`. **Select columns by name, never by position.**
- `codeword_category == gene_category[codeword_gene_mapping]` holds for all 53,385 codewords. Each gene has
  exactly `is_gene` and `is_predesigned_gene` set; each control has one flag. [H]

**`grid` group:**
- One resolution level; `grid_size` is 2000 µm; `grid_key_names` = `[grid_x_loc, grid_y_loc]`.
- **35 tile keys** (x −1…6, y −1…4). Tile `5,-1` is empty; the largest, `2,1`, has 101,948,261 rows.
- `grid_number_objects` is index-aligned with the **`grid_keys` attribute order**. That order happens to be
  lexicographic here. The docs' example key has three parts (`"0,0,0"`), so keys must be parsed via
  `grid_key_names`.
- Tile `i,j` holds x ∈ [2000i, 2000(i+1)). Coordinates can be slightly negative (down to about −2.5 µm).
- Group attrs `gene_counts` and `codeword_counts` are **all-QV** totals. [H]

**Per-tile arrays:**

| Array | dtype / chunks | Semantics |
|---|---|---|
| `location` | f4 [N,3], **Fortran order**, chunks 87,382 | x, y, z in µm, in multiples of 1/64 µm. z is absolute (about 6.9–31 µm) and continuous |
| `codeword_identity` | u4 [N,1], chunks 262,144 | gene = `codeword_gene_mapping[codeword]` |
| `gene_offset` | u4 [24677,2] | Rows **sorted by gene**, then by codeword. Each gene has an `[start, end)` range, end-exclusive (the docs write `[start, end]`). Per-row gene = `searchsorted(gene_offset[:,1], row, side="right")`; on probe tiles this equals the codeword mapping for 100% of rows |
| `cell_id` | u4 [N,2], F order, chunks 131,072 | (prefix, suffix). Unassigned = `(0xFFFFFFFF, 0xFFFFFFFF)` |
| `overlaps_nucleus` | u1 [N,1], chunks 1,048,576 | Meaningful only where a cell is assigned (docs). 1 iff assigned and on a `masks/0` label. On unassigned rows 0 means "not applicable" |
| `quality_score` | **f2** [N,1], chunks 524,288 | QV 8.0–40.0 in steps of 0.25; 47% of rows sit at the cap of 40 |

- Every array uses blosc-zstd clevel 1 with shuffle.
- 15.63 GB on disk, about 9.53 B per transcript; native arrays take 27 B per transcript in memory (about
  44 GB in total).
- There is no `id` array and no per-row gene string. [H]

**Assignment rule** (probe, 8.16 M-transcript crop; QV checked on 1.40 B rows):
- A transcript is assigned when `qv ≥ 20` and its floor-pixel lies on a `masks/1` label. This holds for
  99.998% of such rows on the crop (71 exceptions).
- 0 transcripts with QV < 20 are assigned among the 1.40 B rows checked.
- Assignment is purely 2D. [H]

**Pixel convention:**
- pixel = `floor(µm / ps)`, so pixel i covers [i·ps, (i+1)·ps).
- On the crop, 100.00% of assigned transcripts land on their own cell's label with floor, against 98.65%
  with `rint`. [H]

### 2.3 `cells.zarr.zip`

- **Cell IDs.**
  - `cell_id` u4 [91431, 2]: prefixes strictly increasing, suffix always 1.
  - The Xenium-style string (hex digits mapped to a–p, then `-1`) equals the CFM `obs_names`, in the same
    row order.
  - The packed int64 `(suffix << 32) | prefix` equals CFM `obs.cell_id`. [H]
- **`cell_summary`** f8 [n, 8]:
  - Columns `cell_centroid_x/y`, `cell_area`, `nucleus_centroid_x/y`, `nucleus_area`, `z_level`,
    `nucleus_count`.
  - `cell_area` = mask pixel count × ps² for all 91,431 cells (largest difference 7e-5 µm²) [H].
  - Centroids = (mean pixel index + 0.5)·ps (median residual 0.0002 µm, largest 0.0014 µm) [H].
  - **`nucleus_area` and `nucleus_centroid` describe the largest nucleus** for all 332 multinucleate cells.
    They are not the sum, and not the first or last nucleus. The docs say only "the nucleus" [H].
  - Cells with no nucleus have NaN nucleus fields.
  - `z_level` holds integers 0–7 and 9 (no 8); the docs say "in µm" [H].
- **`bboxes`:** f4 flat [4n], laid out as (xmin, ymin, xmax, ymax) in µm. [H]
- **`polygon_sets/{0 nucleus, 1 cell}`**, with arrays `cell_index`, `method`, `num_vertices` and
  `vertices [n,50]` (25 x,y pairs in µm):
  - Rings are closed: `num_vertices` includes the repeated first vertex, and padding repeats the first
    vertex.
  - `num_vertices == 0` means "no polygon" (docs), though it occurs 0 times here. The minimum is 10
    (nuclei) and 12 (cells).
  - Stored in increasing label-ID order.
  - Set 1 has 91,431 rows (identity `cell_index`).
  - Set 0 has **90,238 nuclei over 89,840 cells**: 1,591 cells have no nucleus and 332 have more than one.
    `nucleus_count == bincount(cell_index)` for 100% of cells.
  - `polygon_sets/0/cell_index` is non-decreasing over the whole array.
  - Vertices sit on integer pixel index × ps, i.e. a pixel-centre contour shifted −0.5 px against the
    corner convention. Polygon area / mask area has a median of 0.954. [H]
- **`masks/0`, `masks/1`:**
  - u4 (42548, 61369), chunks (665, 1918), 10.4 GB each uncompressed, 143 MB for the whole zip.
  - `masks/1` label = cell row + 1; `masks/0` label = nucleus-polygon row + 1. Both are exhaustively
    verified and contiguous (1…91,431 and 1…90,238).
  - Nucleus label → owning cell: `cell_index[label−1] + 1`. 100% of nucleus pixels (71,564,681) lie on
    that cell's `masks/1` label. [H]
- **`masks/homogeneous_transform`:** a 4×4 float32 matrix, **µm → stitched-image pixels**, equal to
  diag(4.7060547, 4.7060547, 1, 1). It is 1/ps to about 1e-8 relative, so compare with a tolerance. [H]
- **`segmentation_methods` codes:**
  - Cells: 0 boundary stain (6,174), 1 interior 18S (83,333), 2 nucleus expansion 5 µm (1,924).
  - Nuclei: 3 DAPI multi-focus (88,037), 4 DAPI z-stack (2,201).
  - These match `metrics_summary` exactly. [H]
- **`gridded_polygon_sets`:** an Explorer level-of-detail structure; not ingested.

### 2.4 `cell_feature_matrix.zarr.zip`

- **X:** CSR (int32 data), 91,431 × 24,677, 258,265,955 nnz. Total 638,849,927, of which genes are
  638,693,512. [H]
- **var:**
  - Index `feature_id` (ENSG for genes, the name itself for controls).
  - `feature_name`: 18,049 unique symbols. **17 symbols match `^ENSG\d+`.**
  - `feature_type`, `genome`, `highly_variable` (2,000 True), about 270 `de_*` columns.
  - `filtered`: **True means kept**. 18,033 genes are True; 16 genes and all 6,628 controls are False. The
    docs' wording is ambiguous.
  - Feature order is identical to the transcripts zarr `gene_names`. [H]
- **obs** (this dataset):
  - `barcode` (index), `cell_id` (int64 packed), `filtered`, `leiden_res_1.0`, `centroid_column/row` (µm),
    `cell_area`, `nucleus_area` (largest nucleus, equal to `cell_summary`).
  - `transcript_counts` (genes, QV ≥ 20), `control_probe_counts`, `control_codeword_counts`,
    `total_counts`, `azimuth_{broad,coarse,fine,full_hierarchy,score}`.
  - **`is_sketch` is listed in the docs but absent here.**
  - The docs' `total_counts` formula includes `genomic_control_counts` and `unassigned_codeword_counts`, but
    neither exists as an obs column. They exist only as X columns: Genomic Control 5,676 (in 5,000 cells),
    Unassigned Codeword 2,032, Deprecated Codeword 575.
  - In the data, `total_counts` = the full X row sum **including deprecated codewords**. [H]
  - `filtered=False` (50 cells) holds exactly for cells with 0–9 counts. Every kept cell has ≥ 10 counts,
    by either `transcript_counts` or `total_counts`. [H]
  - The 1,591 nucleus-less cells **do** have Leiden, Azimuth and `X_pca` values. Only the 50
    `filtered=False` cells are NaN. [H]
  - Azimuth (pan-human model): 2 "Unannotated", 4,732 "Unassigned" (broad), plus "Coarse/Fine Not
    Confidently Annotated"; kidney, retina and RBC classes appear. [H]
- **obsm:** `X_pca` (10), `X_umap`. [H]
- **Feature types:**

  | `feature_type` | Features | Transcripts (all QV) | Name prefix |
  |---|---|---|---|
  | Gene Expression | 18,049 | 1,638,575,672 | gene symbol |
  | Negative Control Codeword | 6,390 | 1,862,535 | `NegControlCodeword_` |
  | Unassigned Codeword | 91 | 33,341 | `UnassignedCodeword_` |
  | Negative Control Probe | 81 | 16,930 | `NegControlProbe_` |
  | Genomic Control | 61 | 19,443 | `Intergenic_Region_` |
  | Deprecated Codeword | 5 | 1,009 | `DeprecatedCodeword_` |

### 2.5 Differences from Xenium and what each means for MerXen

| Atera file | Xenium analogue | Consequence for MerXen |
|---|---|---|
| `experiment.spatial` | `experiment.xenium` | • The first line of spatialdata-io `xenium()` fails.<br>• Three duplicate pixel-size parsers know only `experiment.xenium` / `specs.json`: `segmentation/pipeline.py:183-223`, `enrichment/enrich.py:317-349`, `visualization/sanity_plots.py:1418-1446`.<br>• Copy `experiment.spatial` **verbatim** into the store's `atera/` sidecar folder (§3.11) and add a shared lookup. |
| `transcripts.zarr.zip` (35 gene-sorted tiles) | `transcripts.parquet` | • A new streaming reader is needed.<br>• Rows are not spatially ordered, so spatial queries scan whole tiles.<br>• 1.6e9 rows rule out any pandas materialisation. |
| `quality_score` (f2) | `qv` | • MerXen resolves only `qv \| quality \| quality_value` (`segmentation/pipeline.py:875-879`; `enrichment/assignment.py:66-69`).<br>• The builder must emit `qv`. |
| Unassigned `(0xFFFFFFFF, 0xFFFFFFFF)` | `"UNASSIGNED"` | • Not in MerXen's unassigned tokens (`io/spatialdata_io.py:620-623`).<br>• The builder maps it to NA. |
| `polygon_sets` (simplified, −0.5 px) | `*_boundaries.parquet` | • Display-only per the docs.<br>• Shapes come from the masks (§3.5).<br>• Nuclei are per nucleus, so their IDs are in a different namespace from cells (§3.6). |
| `cells.zarr.zip` masks | Same arrays | • Same label rule for cells.<br>• Nucleus labels need the `cell_index` remap.<br>• **Lookups must use floor, not `rint`** (§3.8). |
| CFM (AnnData zarr, ENSG index, controls included) | `cell_feature_matrix.h5` (`gex_only=True`) + `cells.parquet` | • Read with `anndata` from a `ZipStore`.<br>• Derive per-cell control counts from the control columns, then drop controls from X.<br>• Symbol `var_names`, `gene_ids` = ENSG, vendor annotations prefixed. |
| `morphology_2d/` | `morphology_focus/` (`chNNNN_<name>.ome.tif`) | • 2 or 4 channels.<br>• Opening one file of the series **zero-fills** siblings it cannot find.<br>• Map file → channel from OME `TiffData`.<br>• Store as `morphology_focus`. |
| `morphology_3d/` | `morphology.ome.tif` | 21 GB; z mapping UNVERIFIED. Not ingested by default. |
| No parquet or CSV anywhere | Xenium has both | No CSV fallbacks apply (`enrich.py:110-217`). |
| `panel_config.json` (`payloads` list) | `gene_panel.json` (`payload` dict) | Verbatim sidecar (customer and designer included); `gene_coverage` joined into `var`. |
| H&E + 3×3 alignment | Xenium Explorer `*_imagealignment.csv` | Same convention (H&E px → morphology px), verified with keypoints; fits `REGISTER_IMAGES` (§5). |
| 18,049 genes | 300–5,000 genes | Several downstream stages break or slow down (§5). |

### 2.6 Where the official docs and the data disagree with earlier notes

| Topic | Earlier source | Official docs / verified | Resolution |
|---|---|---|---|
| Number of transcript tiles | Format notes: 37 | 35 grid keys, one empty | **35** [H] |
| Boundaries | PR426 / early design: shapes from `polygon_sets` | Polygons are display simplifications; masks drive assignment | Shapes from masks (§3.5) |
| `morphology_2d` channels | Reports assumed 4, with 18S = `ch0002` | Default runs have 2 (18S = `ch0001`); order is in the OME-XML | Map file → channel from `TiffData`; select by name |
| `obs.filtered` | "Meaning UNVERIFIED" | Docs: True = kept; "< 5 transcripts" | Direction resolved. **Data: the cut is < 10 counts** (gene or total). "< 5" is the Azimuth "Unannotated" rule. Not applied at build; kept as `vendor_filtered` |
| `var.filtered` | Docs: "whether the gene was filtered" | Data: True = kept (controls all False) | Document the polarity on `vendor_filtered` |
| `obs.is_sketch` | Docs list it | Absent in atera-1.1.1 (dataset under 1 M cells) | Optional column |
| `total_counts` | Docs formula: genes + probes + codewords + genomic + unassigned | Data: also includes 575 deprecated-codeword counts | Recorded; derive the per-type counts ourselves |
| Vendor clustering | Docs: only cells with a DAPI nucleus | Data: all nucleus-less cells have Leiden, Azimuth and PCA | Recorded; NaN only for the 50 filtered cells |
| `cell_summary.nucleus_area` | Docs: "the nucleus" | Largest nucleus for multinucleate cells | Document; add `nucleus_area_total` |
| `cell_summary.z_level` | Probe: "plane index?" | Docs: "in µm"; data: integers 0–7, 9 | Pass through as `vendor_z_level`; no interpretation (UNVERIFIED) |
| FOV geometry | Docs: 4,819 px, 113 px overlap | Data: 4,843.8 px, about 88 px overlap | Data value used; docs value tagged docs-only |
| Nucleus label rule | Earlier draft: "(official docs)" | Docs state it for cell masks only | Empirical [H] |
| `gene_offset` bounds | Docs `[start, end]` | End-exclusive | `[start, end)` [H] |
| Codeword → gene | PR426 comment: only via `panel_config.json` | Root attr `codeword_gene_mapping` | Root attr, or `gene_offset` [H] |
| CFM quality filter | Downstream report: UNVERIFIED | Q ≥ 20 only; transcripts zarr holds all | Resolved [H] |
| `homogeneous_transform` direction | Legacy Xenium doc: "into microns" | µm → pixels | µm → px [H] |
| `polygon_sets/0/cell_index` order | Reports: "monotonic" / "not monotonic" | Non-decreasing (verified); docs silent | Map by explicit indexing anyway |
| Closing vertex | Seurat: closing y "corrupted to 0" | 4 of 91,431 cell polygons | Moot with mask shapes; drop the closing vertex in the fallback |
| Polygon vertex position | — | Pixel-centre contour (−0.5 px against the corner convention) | The fallback adds +0.5·ps |
| Environment versions | Format notes: zarr 3.1.6, anndata 0.12.11 | Lock: zarr 3.1.6, anndata 0.12.10. Nextflow envs: zarr 3.4.1, anndata 0.12.19 | Test both (§7) |

---

## 3. Design decisions

| # | Decision | Recommended | Alternatives | Status |
|---|---|---|---|---|
| D1 | Reader | MerXen-native reader that ports PR426's verified logic, with BSD-3 attribution | Pin the PR426 fork; wait for a release | Recommended |
| D2 | Platform identity | New `ATERA` enum value plus a `PlatformSpec` registry | Xenium flavour | Recommended |
| D3 | Transcript policy | QV ≥ 20 at build; controls kept and flagged; vendor-ordinal `transcript_id`; pre-filter QV histograms recorded | Keep everything; also drop controls/unassigned | Recommended (Q4) |
| D4 | Transcript storage | Lean numeric/categorical columns; partitions never cross a tile; **`row_range` default**; `tile_blocks` opt-in only with a decode cap | One partition per tile; full spatial re-sort | Recommended; AT-5 may promote `tile_blocks` |
| D5 | Images | `morphology_focus` = all `morphology_2d` channels in OME order, stored pyramid; no 3D, no H&E | Single-channel elements; 3D included | Recommended |
| D6 | Segmentation → elements | Labels from masks; shapes polygonised from masks with the **+0.5 px** correction and 0.25 px topology-preserving simplification | Vendor polygons | Recommended (Q5) |
| D7 | Cell IDs | `instance_id` = cell row + 1 = mask label. Nuclei get their own IDs (nucleus row + 1) **plus `cell_instance_id`; nucleus consumers join via `cell_instance_id`** | Nuclei keyed by cell ID | Recommended |
| D8 | Genes | Symbol `var_names`, `gene_ids` = ENSG; gene-only X; per-type control counts in obs; points keep controls with `is_gene` | ENSG `var_names`; controls in X | Recommended |
| D9 | Coordinates | `global` = level-0 morphology pixels; µm elements carry `Scale(1/ps)`; vendor transform asserted (rtol 1e-6); **pixel-corner convention, floor lookups** | Use `homogeneous_transform` verbatim | Recommended |
| D10 | H&E | Not in the build; later via `REGISTER_IMAGES` | Build-time element | Recommended |
| D11 | Original-segmentation names | `atera_cell_boundaries` / `atera_nucleus` through the schema registry and `PlatformSpec` | Reuse `xenium_*` | Recommended (Q2) |
| D12 | First downstream route | `segmentation_source=native`, then `cellpose_only`, then ProSeg if the benchmark allows | Cellpose + ProSeg now | Recommended (Q6) |
| D13 | Vendor metadata and privacy | **Store keeps everything:** verbatim sidecars under stable names, full header in attrs. Privacy applies to committed text and human-facing output only (`to_alias` for anything printed) | Sanitised sidecars and an attrs allow-list (revision 4; superseded) | **User decision 2026-11-05** |
| D14 | Build atomicity and cache | Write to a temp sibling, then atomic rename with a completion marker; reuse only marked stores whose `build_fingerprint` matches | Current reuse-if-exists | Recommended (Q19) |

### 3.1 D1: reader strategy

**Options:**

- **(a) Depend on PR426.**
  - It reads the format correctly on this dataset [H].
  - But it is open, has 0 reviews, needs Python ≥ 3.12 and spatialdata-io main, and breaks MerXen's `<0.7`
    pin. That pin change alters XENIUM reader output (instance-key/label linking, shapes index name,
    multinucleate handling), so it needs its own regression pass.
  - Its output also misses MerXen's contract:
    - the image key is `morphology`; MerXen hard-codes `morphology_focus` at
      `segmentation/pipeline.py:494-500, 560-582`, `config.py:569` and `CellposeSettings.groovy:87, 253`;
    - the QV column is `quality_score`;
    - the table includes the controls;
    - there is no QV filter and no `transcript_id`;
    - it ignores `homogeneous_transform`;
    - it takes plane i from file i of a sorted glob;
    - it uses polygon shapes.
- **(b) Write a reader from scratch.** This wastes PR426's verified logic.
- **(c) Recommended: a MerXen-native builder.**
  - Built on zarr, anndata, tifffile, dask and psutil, all already declared. imagecodecs with JPEG-2000
    support is available in both environments [H].
  - It ports PR426's I/O patterns: per-task `ZipStore`, tile row-range partitions, `gene_offset`
    decoding, stored-pyramid reads.
  - No dependency or lock change, so the conda env hash and `-resume` are unaffected.
  - Ported code is attributed in `THIRD_PARTY_NOTICES.md`, with `licenses/spatialdata-io-BSD-3-Clause.txt`
    (`MANIFEST.in` ships `licenses/`).
  - A slow parity test against `spatialdata_io.atera()` runs once a release ships it (Q3).

**Notes for the implementation:**

- Open zips with `zarr.storage.ZipStore(path, mode="r")`. A plain string path fails with
  `GroupNotFoundError` [H].
- Open one store **per dask task**. `ZipStore` serialises reads behind an `RLock`
  (`zarr/storage/_zip.py:66,95`). With 8 threads each holding its own store, a read took 1.2–1.3 s
  against 0.5 s single-threaded, measured on a page-cached file [M]. Parallel speedup is therefore not
  assumed; AT-5 measures it.
- Set `zarr.config` `threading.max_workers` to `task.cpus`. zarr's async decode pool does not follow
  `task.cpus`.
- Do **not** port PR426's vlen monkey-patch. `read_elem` and `read_zarr` work without it under zarr 3.1.6
  and 3.4.1 [H]. Add a test with a ragged vlen chunk that fails with a clear message if it ever appears.

### 3.2 D2: platform enum value

**Recommended:** `ATERA` as a platform in its own right, plus a registry, so that downstream code does not
grow a third arm in every ternary.

- **Why not a Xenium flavour.** About a dozen `else:` arms silently mean "Xenium":
  - `enrich.py:544-570, 608-628`
  - `workflows/modules/qc.nf:65-67`
  - `main.nf:514-516, 1912-1922, 3474-3483, 3626-3628`
  - `viewer_cache/transform.py:47-51`
  - `CellposeSettings.groovy:13, 87, 253`, where ATERA would actually fall into the *MERSCOPE* arm.

  A flavour would hide real differences (per-nucleus IDs, gene-sorted points, no
  `cell_feature_matrix.h5`, variable channels) and make failures silent.
- **Registry:** `src/merxen/platforms.py` (§4.2) and `workflows/lib/PlatformSpec.groovy`.
  - `== "XENIUM"` tests that really mean "10x geometry" become `spec.family == "10X"`.
  - Shape, image and sidecar names become registry lookups.
  - An unknown platform raises.
- **Literal widening is staged.**
  - `platform` Literals: 17 sites in `config.py` (`:403, 417, 472, 492, 509, 565, 654, 723, 769, 1223,
    1282, 1382, 1427, 1494, 1578, 1675, 1926`), plus the paired-only `AlignmentConfig` fixed/moving
    Literals at `:1152-1153`, plus `segmentation/model_identity.py:25` and
    `segmentation/hitl/registry.py:36`. That is 21 in total.
  - Each is widened to the shared `Platform` alias **only when its stage handles ATERA**. Milestone 1
    widens only `config.py:403`. `:1152-1153` stay unchanged until pairing.
- **Preflight.** A check in `runPreflightChecks` (`main.nf:2140`, not the alignment-gated block at
  `:1437-1451`) rejects any ATERA row whose effective `stop_stage` is later than the latest implemented
  stage (`build_spatialdata` until AT-6).
  - Preflight throws inside the `sample_rows` map (`main.nf:3003-3006`), so a violation aborts the whole
    run. With the global default `stop_stage = "clustering_squidpy"` (`nextflow.config:40`), ATERA rows
    must set a per-row `stop_stage` (supported, `main.nf:2539-2559`).
  - The message names the fix. Clamping instead is Q18.

### 3.3 D3/D4: transcripts at 1.6 billion rows

**QV filter at build.**
- Recommended: `min_qv = 20` by default. That keeps 1,409,401,472 gene rows plus about 0.2 M control rows
  (controls are mostly low-QV).
- Rationale:
  - It reproduces the vendor matrix rule.
  - It matches `xenium_min_qv = 20` (`workflows/nextflow.config:195-196`).
  - It drops 14% of rows, and no QV < 20 transcript is ever assigned.
- `min_qv = null` keeps all 1.64e9 rows. The raw `transcripts.zarr.zip` remains the archive for QV QC.
- **Spatial QV QC survives the filter.** In the same pass as the write, the build records, before
  filtering:
  - per-tile and per-FOV QV histograms (161 quarter-QV bins from 0 to 40, stable whatever `min_qv` is)
    and mean QV;
  - Q20 per-gene totals (`gene_counts_q20`, recorded whatever `min_qv` is);
  - bounds;
  - a coarse density grid (100 µm).

  All go into `attrs["merxen_atera"]`, under unambiguous names: `*_all_qv` for the vendor `gene_counts`,
  `*_q20` for ours. QC and COMPARE never mix the two populations.
- QV ≥ 20 alone is **not** a substitute for subsampling in the heavy downstream stages (§5).
- The statistics are recorded by the partition reads themselves, in this process, so the points are
  computed with dask's threaded (or synchronous) scheduler. The partition function refuses to be pickled,
  so the `processes` scheduler or a distributed cluster fails loudly instead of losing the statistics.

**Columns of the `transcripts` points element** (about 38 B per row in memory):

| Column | dtype | Source |
|---|---|---|
| `x`, `y`, `z` | float32, µm | `location` (negative edge values kept) |
| `feature_name` | `CategoricalDtype(gene_names)`, **known**, 24,677 categories | Codes from `gene_offset` (`searchsorted` over end offsets; no `codeword_identity` decode). The codeword path is a cross-check in the fast tier and on one real tile. The build asserts `gene_names == cfm.var.feature_name` |
| `is_gene` | bool | `gene_category[:, column_names.index("is_gene")]` indexed by gene code; boosted/custom genes count as genes |
| `qv` | float32 | `quality_score` (f2) |
| `overlaps_nucleus` | bool | Documented as "not applicable" where unassigned |
| `original_assignment` | `UInt64` nullable | Cell row + 1, computed by `order = argsort(keys)`, `pos = searchsorted(keys[order], q)`, row = `order[pos]`, with an assert `keys[order][pos] == q` before any row is assigned. An unknown ID fails the build; the sentinel maps to NA |
| `transcript_id` | uint64 | **Vendor global ordinal + 1**: offsets = cumulative `grid_number_objects` in **`grid_keys` attribute order** (never re-sorted), plus the row within the tile. Assigned before the QV filter. The build asserts each count == `location.shape[0]` (metadata only) |
| `source_tile` | categorical (35) | Tile key |
| `codeword_index` | uint32, optional (`emit_codeword_index=False`) | `codeword_identity` |

- **No per-row strings.** MerXen's current string normalisation would cost about 229 GB at this scale
  (probe, schema C) [M].
- **Why vendor-ordinal IDs:**
  - unique, positive and traceable;
  - stable when `min_qv` changes;
  - no extra pass. `with_stable_transcript_ids` calls `map_partitions(len).compute()`
    (`io/spatialdata_schema.py:292-298`), which would decode the whole graph a second time.
  - Gaps are allowed: the schema checks only uniqueness and a minimum of 1
    (`spatialdata_schema.py:492-504`).

**Partitioning (D4).**
- `dd.from_map` over `(tile, row_start, row_stop)`. Partitions never cross a tile, and the empty tile is
  skipped.
- `meta` is declared exactly, with the full `CategoricalDtype`, so spatialdata 0.8.0 `PointsModel` does
  not compute categories (`models.py` ~921-936).
- **`PointsModel.parse(..., sort=True)` must always be passed explicitly.** Without `sort`, spatialdata
  0.8.0 computes `data.index.is_monotonic_increasing` over every partition (`models.py:826-833`, verified),
  which is a full decode pass. For dask input `sort=True` only skips that check. The MERSCOPE builder
  already passes it (`io/builders/merscope.py:1108`).
- **Default `spatial_order="row_range"`.**
  - Row ranges are multiples of 1,048,576; default `transcript_partition_rows = 4,194,304`, about 400
    partitions.
  - Each `location` chunk boundary costs at most one extra 87,382-row decode.
  - Tiny tiles give one small partition each.
- **`tile_blocks` (opt-in; AT-5 decides whether to promote it).** It gives spatially coherent partitions
  (500 µm blocks, up to about 8.2 M rows each at the crop's 32.7 tx/µm²). The draft's memory claim had no
  mechanism behind it:
  - The module exports `DASK_NUM_WORKERS=${task.cpus}` (`spatialdata_build.nf:25`). Dask reads it as
    config `num_workers` (verified: 16 → 16), so 16 tasks run at once.
  - Siblings of one tile are scheduled together, so up to 16 blocks would each decode the whole 2.75 GB
    tile (about 60–100 GB in total) [L].

  If enabled, it needs one of:
  - **(i)** a process-wide `threading.BoundedSemaphore(max_tiles_in_flight)` around tile decodes in
    `read_transcript_partition` (AT-3). The semaphore covers the block selection too: each whole-tile
    column is cut to its block before the next one is read, so at most `max_tiles_in_flight` whole-tile
    columns are alive. Each block still reads its whole tile, so a tile is read 16 times; or
  - **(ii)** preferred: a per-tile pre-pass that reads only `location` and caches per-block row indices
    as `.npy` (about 0.4 GB per large tile, uint32). Each block task then reads orthogonal selections.

  Blocks under about 1 M rows merge into a tile-level partition.

  **AT-5 decision: `row_range` stays the default; `tile_blocks` stays opt-in.** The measurements
  (`scripts/acceptance/atera_partition_bench.py`, pyarrow reads of `x`, `y`, `feature_name` and
  `original_assignment`, warm page cache; reports under `$MERXEN_EVIDENCE_ROOT/atera_at5/bench/`):
  - **Full store, `row_range`:** a 1 mm² window (8 windows, median 23 M rows) reads in a median 0.71 s
    (0.08–1.56 s). Index pruning keeps a median 44 of 409 files (154 M rows). With no pruning, the same
    window reads in 0.95 s, because pyarrow skips row groups by their `x`/`y` statistics. A per-gene read
    over all 409 files takes 5.5–5.9 s, from 5,643 to 5.7 M rows; a `row_range` file is gene-sorted, so
    the `feature_name` statistics skip most row groups.
  - **Crop, both orders, the same windows and genes:** `tile_blocks` reads fewer rows per window (16 M
    against 28 M) but is no faster: 0.17 s against 0.13 s warm per window, and 0.22 s against 0.17 s per
    gene. It writes 5% less (645 against 679 MB).
  - **Build cost:** the crop build took 69 s with `tile_blocks` against 48 s with `row_range`, although
    only one of its tiles is split. Each split tile is decoded once per block, about 20× the decode time
    (AT-4). On the full section that adds tens of minutes to an 8.5-minute build.
  - **Bug found:** a `tile_blocks` build that split any tile failed at the end, because the points index
    held infinite block edges (fixed in AT-5; open edges are now `null`).

  Sub-second window reads already serve QC, plots and the viewer cache, so nothing downstream needs
  `tile_blocks`. Revisit it only if a consumer needs many small windows per second.
- **Concurrency is explicit.** `write_atera_spatialdata` calls
  `dask.config.set(scheduler="threads", num_workers=int(os.environ.get("DASK_NUM_WORKERS", cpus)))`, and
  zarr `threading.max_workers` gets the same value. Both are asserted in the fast tier.
- **Partition index.**
  - Written at the **zarr root** as `merxen_atera_points_index.json`, so it survives rewrites
    (`_copy_non_spatialdata_sidecars` skips element roots, `spatialdata_io.py:1630-1634`).
  - Keyed by points element, **parquet file name** and row-group statistics (tile, bbox, row count,
    `transcript_id` range), with a fingerprint (file count, total rows).
  - Consumers prune by reading only the selected files, or with pyarrow `filters`, and fall back to a
    full scan when the fingerprint mismatches.
  - Partition numbers are never trusted. `dd.read_parquet` may split files adaptively under
    `split_row_groups="infer"` (UNVERIFIED for these sizes).
- **Write.**
  - Through `write_spatialdata_zarr` (`io/spatialdata_io.py:146-156`). spatialdata writes with dask's
    default snappy (`spatialdata/_io/io_points.py:88`), so expect 17–20 GB (probe 11.7–12.1 B/row snappy,
    plus `is_gene` and `source_tile`) [M].
  - Direct zstd and delta parquet (8–10 B/row) would bypass spatialdata's writer; it is a later
    optimisation only.
- **Known categories are lost on read.** A dask parquet round trip returns `feature_name` with *unknown*
  categories (verified, dask 2026.8.0). spatialdata's points writer then calls `cat.as_known()` (a
  full-column pass, `io_points.py:75-85`) whenever points are rewritten.
  - The build stores the category list in attrs.
  - The native `latest` (§5) hardlinks points rather than rewriting them.
  - Consumers use `cat.set_categories(known)`, which is lazy.
- **Per-gene reads.** `read_atera_transcripts_for_genes(bundle, genes)` reads only `gene_offset` slices.
  It is for QC and debugging, not the build.

**Shared-code fast paths** (needed so the contract does not run 3 or more full passes):
- `prepare_source_spatialdata_contract(..., points_canonical=True)` (`spatialdata_io.py:642-889`):
  - skips `_canonicalize_source_points_partition` (`:610-639`), the transcript-ID regeneration
    (`:837-840`), `_set_known_feature_categories` and the re-parse;
  - verifies columns and dtypes from `_meta` only.
  - **The category skip is keyed on this flag, not on "categories already known".** MERSCOPE points
    arrive known from `PointsModel.parse` and are re-sorted today (`spatialdata_io.py:544-561`), so a
    generic early return would change MERSCOPE codes.
- `normalize_points_for_latest_write` (`spatialdata_io.py:470-476, 507-511`): keep known categoricals and
  numeric assignments behind the same opt-in. MERSCOPE and XENIUM equivalence tests guard it.
- **Streaming deep validation**, replacing the materialising check (`spatialdata_schema.py:492-525`, about
  60–100 GB at this scale):
  - per tile, IDs unique (102 M × 8 B), and block row counts summed equal the QV-filtered tile count;
  - assignment uniques gathered per partition.
- **The ProSeg and `convert_to_latest` paths** (`_canonicalize_proseg_points`, `spatialdata_io.py:1022-1083`,
  and `upgrade_spatialdata_contract`, `segmentation/pipeline.py:1170`) have the same passes. They get the
  same opt-in fast paths before any ATERA route reaches them (AT-7 at the latest).

### 3.4 D5: images

- **`images["morphology_focus"]`:**
  - (c, y, x) uint16. The `c` coordinates are OME channel names in OME order, whatever the count (2 or 4).
    Identity transform to `global`.
  - **File → channel mapping** comes from the OME-XML `TiffData` entries (`FileName`/UUID → `FirstC`), not
    from a glob or file-name order. The build asserts that every C index is covered exactly once and that
    the channel names match.
  - Optional cross-check: `experiment.spatial.segmentation_stain` tells whether the reagents (and so 4
    channels) were used.
  - Built from each file's **stored** pyramid levels 0–4 via `TiffFile(f).series[0].aszarr(level=k)`,
    taking only that file's own plane. A missing sibling is an error, never a zero-fill [H gotcha].
  - TIFF handles are closed; PR426 leaks them.
  - **Level transforms come from the actual shape ratios**, not exactly 2^k, because odd sizes round. The
    build asserts that image level shapes equal label level shapes. If spatialdata's label coarsening
    rounds differently from the TIFF pyramid, the build builds the label pyramid to the TIFF level shapes
    (UNVERIFIED; tested on the crop).
  - Write chunks (1, 4096, 4096).
  - `morphology_focus` must be the **first** image key: segmentation takes `list(images)[0]`
    (`segmentation/pipeline.py:494-498`).
- **`morphology_3d`:** not written by default.
  - Enrichment copies every image (`enrich.py:520-533`).
  - MIQ and the viewer cache iterate over non-derived images (`mask_image_quantification.py:516-525`,
    `viewer_cache/build.py:220-254`).
  - `image_to_cyx` cannot handle `z`.
  - Its z origin is UNVERIFIED.
- **Channel names:** `alphaSMA/Vimentin` (lowercase "alpha") and `ATP1A1/CD45/E-Cadherin`. Cellpose
  defaults `["DAPI", "18S"]` work for both layouts. A partial mismatch must be an error for ATERA:
  `_select_channels` silently drops missing names today (`io/image_source.py:378-394`).

### 3.5 D6: vendor segmentation → labels, shapes and table

**Labels:**
- `labels["cell_labels"]` from `masks/1`; `labels["nucleus_labels"]` from `masks/0`.
- Lazy `da.from_zarr` per task, rechunked to aligned (4096, 4096) before `Labels2DModel.parse`. The native
  (665, 1918) chunks do not halve cleanly; memory is measured in AT-5.
- Assert `homogeneous_transform` ≈ diag(1/ps, 1/ps, 1, 1) with `rtol=1e-6`, zero off-diagonal and
  translation terms, and last row [0, 0, 0, 1].
- Assert mask shape == morphology level-0 shape.

**Shapes, recommended `boundary_source="masks"`:**
- Polygonise `masks/1` and `masks/0` with `masks_to_labeled_polygons` (`segmentation/mask_geometry.py:123`).
- **Pixel → µm conversion is `(c + 0.5)·ps`.**
  - `find_contours(level=0.5)` returns coordinates with pixel centres at integers (`mask_geometry.py:35`),
    while the vendor and spatialdata conventions put the centre of pixel i at i + 0.5.
  - Measured on 167–420 cells: plain `c·ps` shifts centroids by (−0.106, −0.106) µm; with +0.5 the shift
    is (0.0003, 0.0002) µm. Area ratio 0.9997 [H].
  - Equivalently, apply the inverse of `homogeneous_transform` after the +0.5 shift.
  - The existing Cellpose branch (`enrich.py:240-266`) has the same −0.5 px bias. That is noted for a
    separate fix, not changed here (cache and output invariance).
- **Simplification.** Contours are full-resolution (about 100–200 vertices per cell; UNVERIFIED, measured
  in AT-5). They are simplified with topology-preserving `shapely.simplify` at 0.25 px. Below 0.5 px the
  simplification removes only collinear staircase vertices: on 379 interior cells of a dense crop
  centroids and areas equal the unsimplified ones (centroid max 0.0011 µm, median 71 vertices against
  215), while 0.5 px moves 21% of centroids by more than 0.01 µm (max 0.033 µm; AT-3 review). Vertex
  statistics are recorded in attrs.
- **Process safety and cost.**
  - The mask is first written as a memmapped `.npy` in the task work directory (10.4 GB, deleted after).
  - Polygonisation runs in a **subprocess pool started before any dask or zarr thread pool**
    (`forkserver`), with `n_jobs = task.cpus`. Today the default is `os.cpu_count()` = 80 fork workers
    (`mask_geometry.py:148-156`). `atera_shapes_from_masks` refuses `fork`: reading the masks has already
    started zarr threads.
  - Every worker maps the whole `.npy`, so the pages it touches count as file-backed RSS: 112.7 GB summed
    over 16 workers on `masks/1`, of which 1.6 GB is anonymous memory (AT-3 review). The watchdog
    therefore measures anonymous memory (`RssAnon`, or USS/PSS), never `rss`.
  - A new `serial_fallback=False` argument makes a pool failure an error rather than a silent serial loop
    over 91k labels (`:171-178`). The default stays `True` for existing callers.
  - The cell mask is released (`del` + `force_release`) before `masks/0` is loaded.
- **Multi-component labels** become a MultiPolygon via `symmetric_difference` (`mask_geometry.py:44-47`).
  - Policy: keep MultiPolygons (geopandas and spatialdata accept them) and record their count.
  - The build fails if any table cell lacks a shape, because the validator requires equal ID sets.
- **`boundary_source="vendor_polygons"`** is the fast fallback. It drops the closing vertex, skips
  `num_vertices == 0`, adds **+0.5·ps** to the vertices, and builds with `shapely.from_ragged_array`.

**`shapes["cell_boundaries"]`:**
- `instance_id` (uint64) is **both a column and the index**, equal to the `cell_labels` value, as in the
  Xenium canonical shapes (`spatialdata_io.py:743-786`).
- `canonical_shape_instance_series` falls back to `source_cell_id` without the column
  (`spatialdata_schema.py:200-236`); that would raise on duplicates.
- Columns `source_cell_id` and `segmentation_method` (decoded text). Transform `Scale(1/ps)`.

**`shapes["nucleus_boundaries"]`:**
- One row per nucleus. `instance_id` column and index = nucleus row + 1 = the `nucleus_labels` value.
- Columns `cell_instance_id` (owning cell row + 1), `source_cell_id` of the owning cell, and
  `segmentation_method`.

### 3.6 D7: cell ID mapping and the nucleus namespace

| Representation | Where | Value |
|---|---|---|
| `instance_id` (canonical) | cell shapes, `obs.instance_id`, `original_assignment` | cell row + 1 = `masks/1` label |
| `source_cell_id` | shapes, `obs` | Xenium-style string, e.g. `aaaaokel-1` (CFM `obs_names`) |
| `vendor_cell_id` | `obs` only | int64 `(suffix<<32)\|prefix` (CFM `obs.cell_id`) |
| Nucleus `instance_id` | nucleus shapes, `nucleus_labels` | nucleus row + 1; **same integer range as cells but a different namespace** |
| Nucleus `cell_instance_id` | nucleus shapes | owning cell row + 1 |

- **Nucleus-ID hazard.** Nucleus IDs (1…90,238) overlap cell IDs (1…91,431).
  - Clustering builds shape metrics keyed by `instance_id` first (`clustering_squidpy.py:373-376,
    3899-3912`).
  - `_apply_area_metric` (`:4005-4020`) then joins them to table cell IDs and fills `obs.nucleus_area`
    wherever it is missing or all NaN.
  - So adding `atera_nucleus` to `_choose_nucleus_shape_key` (`:3745-3757`) would silently write nucleus
    k's area onto cell k.

  **Rule:**
  - Every nucleus consumer joins via `cell_instance_id` and aggregates per cell (sum, max, count).
  - A cell-keyed consumer that needs `masks/0` remaps labels with `cell_index[label−1] + 1`. This is safe:
    100% containment, 0 violations [H].
  - A grep test lists every consumer of `nucleus_boundaries` / `original_nucleus` / `*_nucleus`. A fixture
    test with a two-nucleus cell checks `nucleus_area` after clustering ingest.
- **Same `source_cell_id` representation on shapes and table.** A mixed representation makes
  `clone_table_for_region` (`assignment.py:182-192`) and `canonical_shape_instance_series` disagree, and the
  `original` layer fails `analysis_layers.py:83-87`.
- **Join rule.** CFM `obs` row order equals `cells.zarr` row order. The build asserts packed-ID equality
  elementwise.

### 3.7 D8: gene IDs, symbols and controls

- **Table X:** `feature_type == "Gene Expression"` only (18,049 columns), matching Xenium `gex_only=True`.
  Before slicing, the build derives per-cell `vendor_genomic_control_counts`,
  `vendor_unassigned_codeword_counts` and `vendor_deprecated_codeword_counts` from the control X columns.
  This keeps the docs' gDNA rate computable.
- **var:**
  - `var_names` = `feature_name` (symbols; uniqueness asserted, and the build fails if a future add-on
    kit duplicates one).
  - `var["gene_ids"]` = ENSG, the only source of IDs. The 17 ENSG-shaped symbols must never be read as
    IDs: test `gene_ids.is_ensembl_gene_id` consumers (`clustering_squidpy.py:3831`,
    `mapmycells.py:351-360`) against the real `var`.
  - `var["feature_types"]`, `var["genome"]`.
  - `var["vendor_filtered"]` (True = kept).
  - `var["vendor_gene_coverage"]`, joined on ENSG from `panel_config.targets[].info.gene_coverage`.
  - **`var["vendor_highly_variable"]`, renamed.** scanpy's `sc.pp.pca` silently uses `highly_variable` when
    it exists (scanpy 1.12.4 `_pca/__init__.py:431-436`).
  - The `de_*` columns are dropped; `diffexp_genes/*.csv` are copied as sidecars.
- **obs:**
  - Vendor annotations take a `vendor_` prefix, and **every vendor column is optional** (renamed only if
    present):
    - `vendor_leiden_res_1.0`, `vendor_azimuth_*`, `vendor_filtered`, `vendor_is_sketch`;
    - `vendor_total_counts`, `vendor_transcript_counts`, `vendor_control_probe_counts`,
      `vendor_control_codeword_counts`. Clustering's `add_qc_metrics` recomputes an unprefixed
      `total_counts`.
  - Geometry, kept unprefixed for Xenium parity:
    - `cell_area`;
    - `nucleus_area` (**largest nucleus**, documented);
    - `nucleus_area_total`, from a streaming `masks/0` bincount remapped to cells (part of the
      acceptance pass);
    - `nucleus_count`;
    - `vendor_z_level` (vendor units unclear).
  - `obsm["spatial"]` = cell centroids in µm. `obsm["X_pca_vendor"]`, `obsm["X_umap_vendor"]`.
  - The NaN pattern is documented: vendor annotations are NaN only for the 50 `vendor_filtered=False`
    cells. MapMyCells cross-checks treat "Unannotated", "Unassigned" and "… Not Confidently Annotated" as
    missing.
- **Controls in points:**
  - Kept and flagged with `is_gene`. `control_features.classify_control_transcripts` takes the
    feature-type path first (`control_features.py:206-213`).
  - Every Atera control name matches `XENIUM_CONTROL_NAME_PATTERN` (`:65-67`), and no gene symbol matches
    [H].
  - The name-only `CONTROL_TOKENS` / `control_token_mask` (`:46-53, 117-134`) **misses all 61
    `Intergenic_Region_*`**. That affects `transcript_spatial_patterns.py:189`. Fixed in AT-2.
  - `qc/gene_comparison.apply_dataset_filter` (`:101-127`) is routed through `control_features` in AT-2.
  - `"ATERA": (XENIUM_CONTROL_NAME_PATTERN,)` is added to `_CONTROL_NAME_PATTERNS` (`:71-78`).
  - If a `codeword_category` string column is ever emitted, add `boosted_gene` to
    `XENIUM_GENE_CODEWORD_CATEGORIES` (`:57-59`).

### 3.8 D9: coordinate systems, transforms and pixel convention

- **`global`** = level-0 morphology pixels.
  - Images and labels: `Identity()`.
  - Points and shapes in µm with `Scale([1/ps, 1/ps], axes=("x","y"))`; z stays in µm. This is the
    spatialdata-io Xenium convention (`readers/xenium.py:449, 469, 558, 622`).
  - Consumers: `analysis/pciseq.py:228, 245-246`, `alignment/frames.py:399`,
    `viewer_cache/build.py:201, 246`.
- **Pixel size.** `pixel_size` comes from `experiment.spatial`, with OME `PhysicalSizeX` as fallback; a
  mismatch above 1e-6 relative fails the build. `homogeneous_transform` is asserted to be a pure scale of
  1/ps within `rtol=1e-6`.
- **Pixel convention (ATERA vendor masks): corner/floor.** Any µm → mask-pixel lookup against vendor masks
  uses `floor(µm/ps)` (100.00% vendor agreement) [H].
  - The vendor computes it in float32 as `floor(float32(µm) * float32(1/ps))`. Float64 `µm/ps` puts
    about 1.4e-5 of transcripts, all within 0.0012 px of a pixel edge, on the neighbouring pixel; the
    float32 rule has 0 mismatches over the whole public section (AT-6). The AT-6 native check uses
    `merxen.enrichment.label_lookup.PixelFrame` (default `precision="float32"`); the later vendor-mask
    lookups (AT-7 `assign_labels_from_masks(pixel_convention="corner")`, the AT-9 ProSeg prior) must use
    the same `PixelFrame`, or they will disagree with the vendor at pixel edges.
  - `assign_labels_from_masks` uses `np.rint` (`segmentation/cellpose.py:896-897`), which disagrees on
    about 1.35% of transcripts.
  - Add a `pixel_convention: Literal["center", "corner"] = "center"` argument. The default keeps Xenium
    and MERSCOPE outputs unchanged; vendor-mask lookups (native route, enrichment, MIQ, ProSeg prior)
    pass `"corner"`.
  - Under spatialdata's raster convention, `"corner"` is also correct for Cellpose masks. Switching them
    is a separate, output-changing decision.
- **Negative coordinates** (84 at x < 0, 10,582 at y < 0) are kept. The helper bounds-checks
  (`cellpose.py:899-903`), so negative coordinates get label 0 under floor. Under `rint`, x in
  [−0.5·ps, 0) would wrongly land on column 0. Resolved.

### 3.9 D10: H&E

- Not part of the build. A raw H&E in `sdata.images` is off-grid and would break MIQ
  (`mask_image_quantification.py:563-567`) and the viewer-cache base grid.
- It fits `REGISTER_IMAGES` as is:
  - `read_ome_tiff_info` gives 3 channels at 0.27378 µm.
  - Matrix scale 1.28904 against 1.28840 passes the 2% `check_matrix_scale`.
  - The row needs `registration_channel=none`.
- Gating changes are in §5 (AT-10).

### 3.10 Input layout, preview bundles, species and multi-ROI

- **Builder input is one ROI directory:**
  - extracted outs members (`experiment.spatial`, `morphology_2d/`, `panel_config.json`,
    `metrics_summary*.csv`, the JSONs, `diffexp_genes/`);
  - plus the zarr zips;
  - plus optional H&E companions.
- **Resolution.** `resolve_atera_bundle` resolves paths from `experiment.spatial` (`images.*`,
  `explorer_files.*`) relative to `input_path`. This is the main case for in-house instrument directories,
  which hold un-prefixed names. It then falls back to globbing the prefixed public companions.
- **`atera_spec_path`** overrides **metadata only**. Paths always resolve against `input_path`.
- **Staging** happens outside the pipeline and is documented in `docs/stages/spatialdata-build.md`:
  - Extract the small files and `morphology_2d/` (about 9 GB).
  - Skip `morphology_3d/` (21 GB), `binned_transcripts` (2.7 GB) and the CSC matrix.
  - The nested zips inside `outs.zip` cannot be used in place, because `ZipStore` needs a file path. Use
    the standalone downloads.
  - `morphology_2d` siblings must sit together under their original names.
  - Under apptainer, `$MERXEN_ATERA_RAW_ROOT` must be inside `apptainer_bind_paths`
    (`nextflow.config:589, 848`).
- **Preview bundles.** Atera "preview" datasets in Xenium-v4 layout (`experiment.xenium`, parquet, h5)
  would go through `analysis_mode=xenium` (`io/builders/xenium.py:44-56`) with none of the scale guards.
  The XENIUM builder refuses a bundle whose `analysis_sw_version` starts with `atera-`, or that has more
  than 5,000 gene features, with a message pointing at Q16. This changes no hash: Python is not part of
  the task hash, and it is a no-op for real Xenium data.
- **Species.** The build asserts `experiment.spatial.panel_wta.panel_organism` matches `--species`
  (`main.nf:131`).
- **Multi-ROI.** Each ROI is its own samplesheet row with an unrelated µm frame. Any `dataset_name` or
  `pair_id` derived from ROI or slide fields must be an alias (D13).

### 3.11 D13: vendor metadata (complete) and privacy (committed text only)

**User decision 2026-11-05.** The metadata stored with each run must be as complete as possible. Privacy
rules apply only to what is committed to git (code, tests, fixtures, docs, commit messages, PR text) and to
human-facing output, never to the data products written to storage. This replaces revision 4's sanitised
sidecars and attrs allow-list, which would have removed run, slide, instrument, reagent and customer fields
from the store.

- **Sidecars are byte-identical copies** under the store's `atera/` folder (`copy_atera_sidecars`), with
  stable names so code can find them whatever the run (`stable_sidecar_name`):
  - `experiment.spatial`; with `atera_spec_path`, the override is kept too, as
    `experiment_override.spatial`;
  - `metrics_summary.csv` (from `metrics_summary_<…>.csv`);
  - `analysis_summary.html` (from `analysis_summary_<…>.html`) and `qc_images.html`;
  - `panel_config.json`, including `customer` and `designer`;
  - `morphology_fov_locations.json`, `overview_scan_fov_locations.json`, `overview_scan.png`;
  - `diffexp_genes/*.csv`;
  - `he_imagealignment.csv` and `keypoints.csv` when present.

  Two bundle files with the same stable name fail the build loudly instead of dropping one. The `atera/`
  folder sits at the zarr root, so store rewrites keep it (`_copy_non_spatialdata_sidecars`).
- **Metadata inside the vendor files is kept verbatim too** (`write_atera_embedded_metadata`, AT-4
  review):
  - `atera/zarr_metadata/<store>.json` for `transcripts`, `cells`, `cell_feature_matrix` and, when
    present, `binned_transcripts` and `csc_cell_feature_matrix`: the exact text of every `.zattrs`,
    `.zgroup`, `.zarray`, `.zmetadata` and `zarr.json` member (dataset UUID, kit, codeword maps,
    segmentation methods, polygon set descriptions, cell annotations, clusterings, matrix properties,
    column descriptions);
  - `atera/ome_xml/<relative path>.xml`: the OME-XML header of every `morphology_2d` file and of the 3D
    stack (instrument, channels, map annotations with observed and suggested intensity ranges and the
    pixel value offset).
- **The table keeps the vendor table data it has room for:** the `cell_summary` columns without a named
  place (`vendor_nucleus_centroid_x/y`, the centroid of the largest nucleus; `vendor_z_level`), the CFM
  `centroid_column/row`, the `de_*` statistics of the kept features (`varm["vendor_de"]`) and the vendor
  category colours (`uns["vendor_<column>_colors"]`, `#rrggbbaa`).
- **`attrs["merxen_atera"]` carries the full header** (`atera_metadata_attrs`), as JSON-serialisable
  structures with no field dropped:
  - `experiment`: the whole `experiment.spatial` (run name, slide, region, instrument serial, UUIDs,
    versions, panels, counts and metrics);
  - `metrics_summary`: every column of every row, numbers parsed, empty cells `null`;
  - `panel_config`: the whole `panel_config.json` except the per-target list (18,196 targets, 11 MB),
    which is replaced by `num_targets` and stays in the sidecar;
  - `bundle_files`: the file names of the zips and morphology files;
  - `sidecars` and `embedded_metadata`: per sidecar, its store path, **original file name**, size and
    SHA-256;
  - `experiment_source`: `bundle`, or `override` when `atera_spec_path` replaced the manifest (the
    override's sidecar then comes first in the pixel-size lookup);
  - `vendor_store_attrs`: the root attributes of every vendor store; a value over 32 KiB of JSON (the
    codeword and gene maps, 2.4 MB in all) is named under `attrs_in_sidecar_only` and kept in its
    sidecar, so the store's root attrs stay small enough to read on every open;
  - `morphology_2d_annotations`: the OME map annotations, one per channel.
- **Committed text** never holds a real sample, run, slide or instrument name or a host path. The
  synthetic fixture uses obviously fake values; real-data tests are `slow`, read `MERXEN_ATERA_RAW_ROOT`
  and never print manifest values.
- **Human-facing output** (log lines, error messages, plot titles, report file names) names the sample
  by its `dataset_name` (already an alias). Any manifest field it prints passes through
  `merxen.privacy.to_alias`, as the Xenium and MERSCOPE code does. Evidence files written by the
  acceptance script use aliases in their names and text.
- **Aliases.** In-house Atera samples need an alias family (Q15). CLAUDE.md lists only the HD, HM and MS
  families, and `privacy.py` detects only host paths, `P\d{4}` and alias-table literals (`:61-80`).

### 3.12 D14: build atomicity, cache reuse and fingerprint

- **The hazard today:**
  - `_find_reusable_source` returns any existing `persistent_output_path`
    (`io/builders/pipeline.py:160-175`), and that reuse happens before `remove_path(target_path)`
    (`:56-63` vs `:87-89`).
  - `buildConfigForPlatform` writes straight to `persistent_output_path` (`main.nf:419, 437`).
  - So an OOM retry would reuse attempt 1's half-written 30–45 GB store, and changed ATERA settings would
    be silently ignored.
- **ATERA behaviour:**
  - Write to `<target>.partial-<uuid>`, write `merxen_build_complete.json` (with `build_fingerprint`), then
    `os.replace` onto the target.
  - Reuse only a store that has the marker *and* a matching fingerprint: a hash of `AteraBuildConfig`,
    input file sizes and mtimes, and the builder version.
  - On mismatch, refuse unless `--force-rerun` (Q19).
  - The same guard shape as `_check_cached_bead_removal` (`pipeline.py:121-157`), ATERA-only, so
    XENIUM/MERSCOPE behaviour is unchanged.

---

## 4. Build-stage changes, file by file

### 4.1 Mapping Atera outputs to SpatialData elements

| Atera source | Element (source zarr) | Transform | Notes |
|---|---|---|---|
| `morphology_2d/ch*.ome.tif` (mapped via OME `TiffData`) | `images/morphology_focus` (c,y,x) u16, c = OME names | Identity | First image key; 5 stored levels; level scales from shape ratios |
| `cells.zarr.zip/masks/1` | `labels/cell_labels` u32 | Identity | Label = cell row + 1 |
| `cells.zarr.zip/masks/0` | `labels/nucleus_labels` u32 | Identity | Label = nucleus row + 1 |
| `masks/1` polygonised (+0.5 px, simplified), `polygon_sets/1/method` | `shapes/cell_boundaries` (`instance_id` column = index; `source_cell_id`, `segmentation_method`) | `Scale(1/ps)` | Canonical when written |
| `masks/0` polygonised, `polygon_sets/0/{cell_index,method}` | `shapes/nucleus_boundaries` (`instance_id` column = index per nucleus; `cell_instance_id`, `source_cell_id`) | `Scale(1/ps)` | Registered later as `original_nucleus` (`masks.py:43-46, 63-92`) |
| `transcripts.zarr.zip/grid/*` + root attrs | `points/transcripts` (§3.3 columns); instance key `original_assignment`; feature key `feature_name` | `Scale(1/ps)` | Primary points |
| CFM + `cells.zarr.zip/cell_summary`, `cell_id`, `masks/0` bincount | `tables/table`: region `cell_boundaries`, `region_key="region"`, `instance_key="instance_id"` | — | Gene-only X; per-type control counts; `vendor_*` |
| `experiment.spatial`, `metrics_summary_*.csv`, `analysis_summary_*.html`, `qc_images.html`, `panel_config.json`, `morphology_fov_locations.json`, `overview_scan_fov_locations.json`, `overview_scan.png`, `diffexp_genes/*.csv`, H&E alignment + keypoints (if present) | Byte-identical sidecars in the zarr-root `atera/` folder, under stable names; original names, sizes and SHA-256 in attrs | — | Survive rewrites (`spatialdata_io.py:1626-1639`); nothing redacted (D13) |
| Zarr metadata members of every vendor zip; OME-XML of every morphology TIFF | `atera/zarr_metadata/<store>.json`, `atera/ome_xml/<path>.xml` (verbatim text); root attrs and map annotations parsed into attrs | — | D13 (AT-4 review) |
| Computed in the points pass | `merxen_atera_points_index.json` (root) | — | Keyed by parquet file + fingerprint |
| Build state | `merxen_build_complete.json` (root) | — | Completion marker + `build_fingerprint` |
| Header facts | `attrs["merxen_atera"]`, complete (D13) | — | Keys:<br>• `experiment` (all of `experiment.spatial`), `metrics_summary` (every column), `panel_config` (all but the target list), `bundle_files`, `sidecars`<br>• `pixel_size_um`, `chemistry`, `panel_design_ids[]`, `num_gene_targets`<br>• `number_rnas`, `min_qv`, `n_points_written`<br>• `spatial_order`, `boundary_source`, polygon vertex stats<br>• `gene_counts_all_qv`, `gene_counts_q20`, QV histograms per tile and FOV, bounds, density grid<br>• `feature_categories`, `build_fingerprint` |
| `attrs["merxen_schema"]` | `platform="ATERA"`, `primary_points="transcripts"`, `segmentations.original` = {points `transcripts`, assignment `original_assignment`, shape `cell_boundaries`, table `table`, id_namespace `source_segmentation_v1`} | — | Via `stamp_merxen_schema` / `register_segmentation_branch` (`spatialdata_schema.py:70-154`) |
| Not ingested as elements | `morphology_3d`, `binned_transcripts`, `csc_cell_feature_matrix`, `gridded_polygon_sets` | — | 3D opt-in only; the overview scan and `qc_images.html` are sidecars |

### 4.2 New files

**`src/merxen/platforms.py`** (AT-1 with MERSCOPE and XENIUM only; ATERA added in AT-4):

```python
Platform = Literal["MERSCOPE", "XENIUM", "ATERA"]   # ATERA appended in AT-4

@dataclass(frozen=True)
class PlatformSpec:
    name: str
    family: Literal["10X", "MERSCOPE"]
    native_image_key: str                  # "morphology_focus" | "MERSCOPE_z_projection"
    original_cell_shape_key: str           # "xenium_cell_boundaries" | "atera_cell_boundaries" | ...
    original_nucleus_shape_key: str | None # "xenium_nucleus" | "atera_nucleus"
    nucleus_ids_are_cell_ids: bool         # XENIUM True; ATERA False (join via cell_instance_id)
    spec_sidecar_names: tuple[str, ...]    # ("experiment.xenium","specs.json","specs/specs.json") | ("experiment.spatial",)
    quality_column_alias: str | None
    default_channels: tuple[str, ...]      # ATERA: ("DAPI", "18S")
    mask_pixel_convention: Literal["center", "corner"]  # vendor masks: ATERA "corner"
    fallback_um_per_px: float              # 0.108 | 0.2125 | 0.21249222

def platform_spec(platform: str) -> PlatformSpec            # raises on unknown
def find_um_per_px(root: Path, platform: str) -> float      # replaces the 3 duplicate parsers
def original_cell_shape_key(sdata_or_attrs, platform: str) -> str
```

`workflows/lib/PlatformSpec.groovy` is its twin. The `buildConfigForPlatform` harness
(`tests/test_workflows/test_spatialdata_build_config.py:351-376`) concatenates `main.nf` function text into
a temp script, so it must copy `workflows/lib/` next to that script.

**`src/merxen/io/atera_reader.py`** holds the pure readers:

```python
@dataclass(frozen=True)
class AteraBundle:
    root: Path; experiment: dict; pixel_size_um: float; analysis_sw_version: str
    transcripts_zip: Path; cells_zip: Path; cell_feature_matrix_zip: Path
    morphology_2d_files: tuple[Path, ...]   # ordered by OME TiffData FirstC
    morphology_channel_names: tuple[str, ...]
    morphology_3d_file: Path | None
    sidecars: tuple[AteraSidecar, ...]      # source, stored_path (stable), original_name

def resolve_atera_bundle(input_path: Path, *, spec_override: Path | None = None,
                         transcripts_path: Path | None = None, cells_path: Path | None = None,
                         cfm_path: Path | None = None) -> AteraBundle
    # analysis_sw_version startswith "atera-"; RnaDataset major 6, coordinate_space, spatial_units;
    # CellSegmentationDataset major 6; species check; fails loudly on unknown majors
def stable_sidecar_name(name: str) -> str                         # metrics_summary_<…>.csv -> metrics_summary.csv
def copy_atera_sidecars(bundle, store_root: Path) -> list[dict]   # verbatim into atera/ (D13)
def write_atera_embedded_metadata(bundle, store_root: Path) -> list[dict]
    # zarr metadata members and OME-XML headers, verbatim into atera/ (D13)
def atera_metadata_attrs(bundle, sidecars, embedded=()) -> dict   # full header for attrs (D13)
def read_atera_tile_index(bundle) -> list[AteraTile]   # key (parsed via grid_key_names), n_rows, global_offset in attr order
def plan_transcript_partitions(tiles, *, partition_rows: int, spatial_order: str,
                               block_um: float, min_block_rows: int,
                               tile_allowlist: Sequence[str] | None) -> list[PartitionSpec]
def read_transcript_partition(bundle, spec: PartitionSpec, *, gene_dtype: CategoricalDtype,
                              is_gene: np.ndarray, cell_lookup: CellKeyLookup,
                              min_qv: float | None) -> pd.DataFrame
def read_atera_transcripts(bundle, *, min_qv, partition_rows, spatial_order, block_um,
                           max_tiles_in_flight, emit_codeword_index=False
                           ) -> tuple[dd.DataFrame, list[PartitionSpec], TranscriptAggregates]
def read_atera_transcripts_for_genes(bundle, genes: Sequence[str], *, min_qv=None) -> pd.DataFrame
def read_atera_cells(bundle) -> AteraCells
def read_atera_labels(bundle) -> dict[str, da.Array]
def atera_shapes_from_masks(bundle, cells: AteraCells, *, n_jobs: int, simplify_px: float = 0.25,
                            workdir: Path) -> dict[str, gpd.GeoDataFrame]   # subprocess pool, +0.5 px
def atera_shapes_from_vendor_polygons(cells: AteraCells, pixel_size_um: float) -> dict[str, gpd.GeoDataFrame]
def read_atera_table(bundle, cells: AteraCells, *, gex_only=True, keep_vendor_annotations=True,
                     nucleus_area_total: np.ndarray | None = None) -> AnnData
def read_atera_morphology_2d(bundle, *, n_levels: int = 5) -> xr.DataTree
```

**`src/merxen/io/builders/atera.py`:**

```python
def write_atera_spatialdata(*, input_path: Path, output_path: Path,
                            build_config: AteraBuildConfig, atera_spec_path: Path | None = None,
                            n_workers: int | None = None) -> Path
def _check_cached_atera_build(path: Path, fingerprint: str, *, force_rerun: bool) -> bool
def _assemble_atera_sdata(bundle, build_config) -> SpatialData
# sidecars: copy_atera_sidecars + atera_metadata_attrs (verbatim, stable names, original names recorded)
def _write_points_index(output_path, specs) -> None
class RssWatchdog: ...   # process-tree anonymous RSS (RssAnon/USS, not rss: §3.5); exit 137 above the limit
```

**Build sequence:**
1. Start the watchdog.
2. Configure dask and zarr concurrency.
3. `resolve_atera_bundle`.
4. Polygonise the masks (subprocess pool, before any thread pools).
5. Run the readers.
6. Call `PointsModel.parse(sort=True)`, `ShapesModel`, `Labels2DModel`, `Image2DModel` and `TableModel`.
7. `prepare_source_spatialdata_contract(sdata, platform="ATERA", points_canonical=True)`.
8. `write_spatialdata_zarr` to the `.partial-*` path. The aggregates are computed in the same dask graph,
   in one `dask.compute`.
9. Write sidecars, the points index, and the completion marker.
10. Atomic rename.
11. `force_release`.

It reuses `merxen.io.ome_tiff` (`ome_tiff.py:66-151`) and a vendored
`cell_id_str_from_prefix_suffix_uint32`, with BSD-3 attribution.

**Tests and scripts:**
- `tests/test_io/atera_synth.py`, `tests/test_io/test_atera_reader.py`, `tests/test_io/test_atera_build.py`
  (§7).
- `scripts/acceptance/atera_build.py`: the streaming acceptance checks (§1.1). Results go under
  `$MERXEN_EVIDENCE_ROOT`.
- `THIRD_PARTY_NOTICES.md` entry and `licenses/spatialdata-io-BSD-3-Clause.txt`.

### 4.3 Modified Python files

| File:line | Change |
|---|---|
| `src/merxen/config.py:384-397` | Add `AteraBuildConfig` (fields below) |
| `config.py:403-410` | `SpatialDataBuildConfig.platform: Platform`; add `atera_spec_path` and `atera: AteraBuildConfig`. Keep rendered XENIUM/MERSCOPE configs byte-identical |
| `src/merxen/io/builders/pipeline.py:56-63, 93-113` | `elif platform == "ATERA": write_atera_spatialdata(...)`. For ATERA, the reuse branch first calls `_check_cached_atera_build` (marker + fingerprint) |
| `pipeline.py:178-192` | No change for directory input; a zip input is rejected with an "extract outs" message |
| `src/merxen/io/builders/xenium.py:44` | Refuse Atera preview bundles (§3.10) |
| `src/merxen/io/spatialdata_io.py:642-889` | `prepare_source_spatialdata_contract(..., points_canonical: bool = False)`; the category skip is gated on this flag only |
| `spatialdata_io.py:470-476, 507-511` | `normalize_points_for_latest_write`: opt-in to keep known categoricals and numeric assignments |
| `spatialdata_io.py:1022-1083` | `_canonicalize_proseg_points`: the same opt-in fast paths (before AT-7) |
| `src/merxen/io/spatialdata_schema.py:281-350` | `with_stable_transcript_ids(..., partition_lengths=None)` |
| `spatialdata_schema.py:492-525` | Add `validate_merxen_schema_streaming(sdata)` (per-tile checks) |
| `src/merxen/segmentation/mask_geometry.py:123-178` | Add `serial_fallback: bool = True` and `mp_context: str = "fork"` arguments; defaults unchanged |
| `src/merxen/segmentation/cellpose.py:870-903` | `assign_labels_from_masks(..., pixel_convention="center")` |
| `src/merxen/control_features.py:46-53, 71-78` | `intergenic` token; `"ATERA"` pattern entry; docstrings (AT-2) |
| `segmentation/pipeline.py:183-236`, `enrichment/enrich.py:317-349, 389-418`, `visualization/sanity_plots.py:1386-1446`, `viewer_cache/transform.py:24-51` | Route pixel size through `platforms.find_um_per_px`. The ATERA dispatch arms land in AT-6 |
| `src/merxen/io/samplesheet.py` | Unused by the workflow; delete it (Q14) or add `atera_*` fields |
| `src/merxen/cli/run_build_spatialdata.py:28` | Docstring |
| `src/merxen/local_config.py:49-108` + `.env.example` | `MERXEN_ATERA_RAW_ROOT` in the same position (`tests/test_privacy/test_templates.py:25-27`) |
| `scripts/cleanup_assignment_tables_and_outputs.py:104-111` | Add the `atera` mode, or import the shared map |

**`AteraBuildConfig` fields:**

```python
transcripts: bool = True
min_qv: float | None = 20.0
transcript_partition_rows: int = 4_194_304          # multiple of 1_048_576
spatial_order: Literal["row_range", "tile_blocks"] = "row_range"   # AT-5 may promote tile_blocks
spatial_block_um: float = 500.0
min_block_rows: int = 1_048_576                     # smaller blocks merge into a tile partition
max_tiles_in_flight: int = 4                        # enforced by a semaphore (tile_blocks only)
emit_codeword_index: bool = False
tile_allowlist: list[str] | None = None             # pilot/debug only
cells_labels: bool = True
nucleus_labels: bool = True
cells_boundaries: bool = True
nucleus_boundaries: bool = True
boundary_source: Literal["masks", "vendor_polygons"] = "masks"
boundary_simplify_px: float = 0.25
polygon_n_jobs: int | None = None                   # None -> DASK_NUM_WORKERS / task.cpus
cells_table: bool = True
table_gex_only: bool = True
keep_vendor_annotations: bool = True
morphology_2d: bool = True
morphology_3d: bool = False
max_rss_gb: float | None = None                     # watchdog; set from task memory
transcripts_path: Path | None = None
cells_path: Path | None = None
cell_feature_matrix_path: Path | None = None
```

### 4.4 Nextflow and samplesheet

| File:line | Change |
|---|---|
| `workflows/main.nf:92-117` `normalizeAnalysisMode` | Add `atera` (alias `a`); update the error text at `:113` |
| `main.nf:156-167` `activePlatformsForMode` | `atera → ["ATERA"]`. `paired` stays MERSCOPE + XENIUM; alignment and comparison drop out (`main.nf:967-972, 2262-2267`) |
| `main.nf:345-374` `requirePlatformInput` | ATERA: `atera_dir` or `atera_spatialdata_path` |
| `main.nf:376-446` `buildConfigForPlatform` | ATERA arm: `[dataset_name: "${pairId}_ATERA", platform: "ATERA", input_path, output_path, persistent_output_path, atera_spec_path, merscope: [:], xenium: [:], atera: [max_rss_gb: …]]`. Add `atera:` keys **only** to the ATERA map (`:383-384`) |
| `main.nf:2140` `runPreflightChecks` | ATERA `stop_stage` check (§3.2); a harness test covers single and mixed samplesheets |
| `main.nf:2164-2166`, `modules/spatialdata_build.nf:4` | No change; publishes under `<pair>/atera/spatialdata/` |
| `workflows/nextflow.config:612-643` `BUILD_SPATIALDATA` | Platform-aware closures (directives are not hashed):<br>`cpus = { platform == 'ATERA' ? 16 : 8 }`<br>`memory = { platform == 'ATERA' ? (params.atera_build_memory_gb as int).GB * task.attempt : 80.GB }`<br>`errorStrategy = { platform == 'ATERA' && task.exitStatus in [137,140] && task.attempt < 2 ? 'retry' : 'ignore' }` (safe only with D14 atomicity)<br>`time` for ATERA under profiles that set `process.time` (`:915-919`) |
| `nextflow.config:612-643, 888-900` | GPU `queue` / `clusterOptions` / `--nv` closures require `platform == 'MERSCOPE'`. Check `containerOptions` hashing with `tests/test_workflows/test_dwight_profile.py:160-196, 250-340` (UNVERIFIED) |
| `nextflow.config` params (AT-4) | `atera_build_memory_gb = 128`, `atera_min_qv = 20.0`, `default_atera_voxel_layers = 2`, plus the **six** Cellpose names `CellposeSettings.settings` builds (`CellposeSettings.groovy:137-150`): `atera_cellpose_model`, `atera_cellpose_cellprob`, `atera_cellpose_flow_threshold`, `atera_cellpose_diameter`, `atera_cellpose_final_max_area_um2`, `atera_mask_max_eccentricity`, all `null`. `CellposeSettings.preflight` evaluates them for every active platform on every row (`:213-218`) |
| `workflows/conf/dwight.config:60, 76-87` | Keep `build_spatialdata_max_forks = 3`. **The local executor only schedules memory; it does not cap it** (`process.executor = "local"`, `executor { memory = "640 GB" }`). The 128 GB is a reservation; the in-task RSS watchdog is the cap |
| `workflows/samplesheet.example.csv` | Add the `atera_*` columns |
| `assets/metro_map.mmd` + `scripts/render_metro_map.py` + `src/merxen/palette.py:8-57` | ATERA line colour; re-render (`tests/test_workflows/test_metro_map.py:157-184`) |

**Samplesheet columns:**

| Column | When | Notes |
|---|---|---|
| `atera_dir` | AT-4 | Staged ROI directory |
| `atera_spatialdata_path` | AT-4 | Persistent build target / reuse (guarded by D14) |
| `atera_spec_path` | AT-4 | Metadata-only override |
| `stop_stage` (per row) | AT-4 | Required as `build_spatialdata` for ATERA until AT-6 |
| `atera_channels` | AT-7 | Default `DAPI;18S` |
| `atera_min_qv` | AT-7 | Default `params.atera_min_qv` |
| `atera_voxel_layers` | AT-7 | ProSeg only; 2 is plausible (Q20 thickness 4.65 µm), UNVERIFIED |
| `atera_cellpose_model` | AT-7 | Custom Cellpose model |
| `atera_registered_images_csv` | AT-10 | H&E registration |

The prefix helpers that already build names from `platform.toLowerCase()` (`main.nf:1297-1395`,
`CellposeSettings.groovy:9-10, 138`) pick up `atera_*` unchanged.

### 4.5 Docs (AT-4)

- `docs/stages/spatialdata-build.md`, ATERA section:
  - required files and staging;
  - apptainer binds;
  - element mapping (§4.1);
  - QV policy;
  - pixel convention;
  - privacy handling;
  - what is skipped;
  - resources;
  - plus fixes to the stale line anchors.
- `docs/samplesheet.md` (`:4-11, 33-35, 78-83, 151-157`): `analysis_mode=atera`, the `atera_*` columns,
  per-row `stop_stage`, and one row per ROI. The example uses `$MERXEN_ATERA_RAW_ROOT`.
- `docs/running-the-pipeline.md:43, 155-192`, `docs/getting-started.md:87-89`, `docs/cli.md`,
  `docs/python-api.md:54, 64, 74`, `docs/local-configuration.md` (new KNOWN variable), `README.md:5, 23,
  31, 81`.
- `docs/configuration.md`, `docs/outputs.md`, `docs/pipeline.md`, and `docs/index.md` (link to this plan).
- `docs/development.md:13-21`: fix the false "plain `pytest` skips slow tests / `--run-slow`" text.

---

## 5. Downstream changes

**Legend:**
- **REQ:** the stage fails or gives wrong output without the change.
- **REC:** recommended for scale or quality.
- Effort is developer-days without contingency.
- "Native" means the AT-6 route: vendor masks as the original segmentation, with no Cellpose or ProSeg.

### 5.1 Cross-cutting

| Change | Kind | Where | Effort |
|---|---|---|---|
| Widen `platform` Literals stage by stage | REQ | `config.py` (19 sites incl. paired `:1152-1153`; §3.2) + 2 | 0.5 |
| `PlatformSpec` lookups replacing binary ternaries: `analysisLayerKeys`, `pciseqSpatialdataPath`, enrich restart path, viewer-cache transform, QC reference shape | REQ | `main.nf:514-516, 1912-1922, 3474-3483, 3626-3628`; `qc.nf:65-67` (**edited in AT-6; one-time QC re-run for existing rows**, Q17); `CellposeSettings.groovy:13, 87, 253` | 2 |
| Original-shape names through the registry and spec | REQ | `enrich.py:63-65`; `assignment.py:369-375, 407`; `masks.py:45`; `sanity_plots.py:39-44, 779-787, 922-923`; `clustering_squidpy.py:3726-3757`; `pciseq.py:41-42`; `viewer_cache/build.py:69-86`; `alignment/frames.py:365-367`; `legacy_features.py:92-97`; `config.py:514`; grep test | 2 |
| **Nucleus consumers join via `cell_instance_id`** and aggregate per cell; vendor `masks/0` remapped to cell IDs for any cell-keyed `.npy` | REQ | `clustering_squidpy.py:373-376, 3899-3912, 4005-4020`; MIQ, ProSeg prior and enrichment nucleus attribution; grep test plus a two-nucleus fixture | 1 |
| **Vendor-mask lookups use `pixel_convention="corner"`** | REQ | `cellpose.py:870-903` and its native-route callers | 0.5 |
| `segmentation_source ∈ {native, cellpose_only, cellpose_proseg}` per row; "all" expands per source; unavailable layers fail at config time; `nuclei_shape_key` from the spec | REQ (AT-6) | `main.nf:176, 720`; `nextflow.config:38, 180, 461, 481`; `config.py:1499`; restart channels `main.nf:3305-3342, 3350-3404, 3711-3733`; `:3078` | 3 |
| Use the build-time aggregates (bounds, n_transcripts, Q20 per-gene totals, QV maps, density grid) | REC | QC / VISUALIZE readers | 1.5 |
| Partition-index pruning (file-keyed) in crop helpers | REC | `sanity_plots.py:683-692, 1137-1152`; `density_overview.py:445-487` | 1.5 |
| `enforce_memory_limit` uses process-tree RSS, not system-wide used RAM | REQ for ATERA BUILD (as watchdog); REC elsewhere | `memory.py:62`; `transcript_io.py:391-396` | 0.5 |
| Lazy `set_categories` on read; never rewrite points needlessly | REC | Consumers that rewrite points | 0.5 |

### 5.2 Per stage

| Stage | REQUIRED to run | RECOMMENDED Atera-specific path | Why | Effort |
|---|---|---|---|---|
| **Segmentation, native (new `NATIVE_LATEST` process)** | • New CPU process. It builds `latest_spatialdata.zarr` by **hardlinking only data payloads** (chunk files, `points.parquet/*`) and **copying every `zarr.json`, `.zattrs`, `.zgroup`, `.zmetadata`**. ENRICH's `_prune_zarr_consolidated_metadata` writes those in place (`enrich.py:669, 682-692`), which would otherwise corrupt the source through the shared inode.<br>• Registers `original` (shape `atera_cell_boundaries`, table `table_original`, assignment `original_assignment`) and `original_nucleus` (`atera_nucleus`).<br>• Never calls `convert_to_latest` or `upgrade_spatialdata_contract`.<br>• Explicit `NO_FILE` placeholders in the SEGMENT output tuple (`segmentation.nf:70`).<br>• Test: source inodes and bytes unchanged after ENRICH and its partial cleanup | Export vendor `masks/1` once to `.npy` (+10.4 GB; +10.4 GB if a cell-keyed nucleus mask is also exported) for MIQ and as a future ProSeg prior | ENRICH today requires Cellpose masks and turns the first shape into `MOSAIK_proseg` (`enrich.py:930-994`) | 4.5 |
| **Segmentation, Cellpose** (`cellpose_only`) | • ATERA in `DatasetConfig` and `hitl/registry.py:36`.<br>• Transform via the spec (`pipeline.py:225-236`).<br>• Zarr-only loader, never `xenium_reader` (`:474-522`).<br>• Explicit `image_key=morphology_focus` (`main.nf:3246-3267`); native key via the spec (`:561-582, 1047-1048`).<br>• Channels `["DAPI","18S"]`.<br>• Partial channel mismatch is an error (`image_source.py:388-394`) | • Evaluate `["DAPI","ATP1A1/CD45/E-Cadherin","18S"]` (vendor used 18S for 91% of cells).<br>• Per-channel normalisation behind a flag | Image 2.61 Gpx, about 1.5 h per Cellpose task [M] | 2 |
| **ProSeg** | • QV branch takes ATERA → `qv`, `min_qv` 20 (`pipeline.py:863-886`).<br>• ATERA-specific `PROSEG_SEGMENT` memory and time.<br>• `convert_to_latest` fast paths (§3.3) | • Benchmarked in AT-5b (§8).<br>• Parquet input in Xenium layout (`proseg_src/src/sampler/transcripts.rs:459-573`).<br>• Controls excluded.<br>• Vendor masks as prior (floor lookups).<br>• 2 mm tiles with a ≥ 60 µm halo and label-owned stitching; explicit `--prior-weight`.<br>• Optional gene subset plus back-assignment.<br>• CSV export out of the GPU lock; delete the CSV after success | • Measured in AT-5b [H]: one whole-section run would need about 300 GB and 32–37 h at the default schedule (no-go); a 2 mm tile peaks at 21.7 GB.<br>• proseg#162: OOM at 256 GB with 1.04e9 transcripts.<br>• CSV about 73 GB (52 B/row) against about 12 GB of parquet.<br>• Neuropil: 47% of QV ≥ 20 gene transcripts are outside vendor cells on the dense tile; ProSeg takes in 15–18% of them under the vendor prior | 8–12 |
| **ProSeg hybrid** | Off for ATERA | Vectorise (`proseg_hybrid.py:983-1031`), `n_jobs > 1` (`:798`), COO (`:1246-1256`) | 10–26 h single-threaded [M] | 3 |
| **Enrichment** | • Native mode with optional masks; no `MOSAIK_proseg` alias; no fake Cellpose provenance (`enrich.py:816-836`).<br>• ATERA arm in the copy step (`:999-1102`), `_is_already_enriched` (`:536-571`) and partial artifacts (`:608-635`).<br>• Original shape → `table_original` clone (`assignment.py:369-375, 407-419`).<br>• Nucleus shapes excluded from per-shape assignment (`:510-516`).<br>• Image copy by hardlink or allow-list (`:520-533`) | • Label-mask lookup (corner convention) instead of `gpd.sjoin` (`assignment.py:205-367`), with COO accumulation and parallel partitions.<br>• Nucleus attribution via remapped `masks/0`.<br>• `max_ram_gb` from task memory (`assignment.py:450-455`) | • sjoin about 1.3–1.5 h per shape × 3 [M benchmark, 25-vertex polygons].<br>• Mask-derived polygons have more vertices, so simplification (§3.5) or the mask lookup is needed | 4 |
| **QC** | • `apply_dataset_filter` via `control_features` (`qc/gene_comparison.py:101-127`).<br>• No `to_pandas` fallback for large dask points (`qc/metrics.py:107-110`) | • Cached aggregates.<br>• Fewer or subsampled registration windows (`qc/registration.py:165-222`).<br>• `control_features` instead of the local regex (`:42`).<br>• Controls dropped from `original` table metrics, or both reported.<br>• Spatial mean-QV map from build attrs | The 6,628 controls enter gene summaries today | 1.5 |
| **Analysis-layer validation** | None in code; depends on consistent IDs (§3.6) | — | `analysis_layers.py:83-87` | 0 |
| **Viewer cache** | • ATERA transform via the spec (`viewer_cache/transform.py:39-51`).<br>• Explicit base image key (`build.py:220-254`) | External napari viewer may need an ATERA branch (outside this repo). `binned_transcripts` not used (schema undocumented) | Same cost class as Xenium; rasterising more-vertex polygons, measure in AT-6 | 1 |
| **MIQ** | • Native: vendor cell `.npy` + `shape_key` from the spec (`config.py:513-514`).<br>• Only on-grid 2D images (`mask_image_quantification.py:516-525, 563-567`) | Streaming per-label stats (`:570-591, 637-654`); tiled `bincount` (`:673-682`) | Fits 160 GB today | 1 + 2 |
| **Visualization** | • ATERA in `_resolve_dataset_mask_affine` (`sanity_plots.py:1386-1404`).<br>• Scale bar (`:461`).<br>• Percentile heatmap vmax (`density_overview.py:28`; saturates about 11×) | • Fused min/max/len (`density_overview.py:415-420`) or attrs.<br>• Histogram once per dataset.<br>• Scatter cap about 200k (`sanity_plots.py:29`).<br>• File-keyed bbox pushdown.<br>• ATERA colours | About 12 full x/y passes per branch [L] | 1 + 2 |
| **Clustering (squidpy)** | • Explicit `mask_var` / HVG (`clustering_squidpy.py:597, 861`).<br>• Sparse cluster means (`:3543-3563, 3642-3645`).<br>• Bounded dotplots (`:2274-2287, 1410-1413, 1487-1514`).<br>• **Nucleus metrics joined via `cell_instance_id`** (§3.6) | • `EXPECTED_PANEL_GENE_COUNT` from config (`:51`).<br>• `CLUSTERING_SQUIDPY_COMPUTE` ≥ 64 GB.<br>• Coordinate with RCA | Dense path about 20 GB; dotplot render raises at about 357k px | 3.5 |
| **MapMyCells** | Hippocampal region labels (`config.py:1700-1707` defaults to frontal cortex); WHB cache has `Human CA1 … DG-CA4` [H] | • Lean query h5ad (`mapmycells.py:220-281`).<br>• Revisit `bootstrap_factor`.<br>• Contingency vs `vendor_azimuth_*`, with all "unannotated" variants treated as missing.<br>• Test ENSG-shaped symbols vs `mapmycells.py:351-360` | Full n10 marker coverage | 1 |
| **pciSeq** (opt-in) | • Refuse above about 500 genes unless given a list; wire `PciSeqReferenceConfig.gene_names` (`config.py:1838`).<br>• `pixel_size_um=2` or `max_pixels ≥ 150M` | • Parquet spots (`pciseq.py:742`; `pciseq_compute.py:30`).<br>• Per-partition ID reads in the viewer (`pciseq_viewer.py:466-516`) | Spots > 100 GB; model arrays possibly cells × genes × classes (UNVERIFIED) | 2 |
| **MECR** | Off for ATERA until marker discovery streams (`mecr.py:425-470, 527-529`) | Two-pass discovery; panel intersection when paired (`:302-318`); generalised plot | Reference build likely > 240 GB (UNVERIFIED) | 3 |
| **MENDER** | ATERA Literal | `backed="r"` obs reads (`mender.py:179, 454`) | — | 0.25 |
| **GASTON** (opt-in) | Ranked gene selection (deviance or HVG, 2–5k) instead of var-order truncation (`gaston.py:1030`) | Record the method | 10k × 91k float64 = 7.3 GB; hours [L] | 1 |
| **Spatial gene analysis** | ATERA Literal | Optional HVG ∪ markers | Sparse numba; minutes | 0.25 |
| **Transcript spatial patterns** | Off for ATERA rows (`main.nf:2313-2314`) until the cap exists | • Per-gene cap by hashing `transcript_id` before the geometry queries (`transcript_spatial_patterns.py:196-201`).<br>• `is_gene` instead of the token mask (`:189`).<br>• `paircorr_n_jobs > 1` | About 70–80 GB, about 15–20 h [M/L] | 2 |
| **Cortical depth** | Off for non-cortical sections (hippocampus) | — | `docs/stages/cortical-depth.md:3` | 0 |
| **Distance-from-object** | Non-cortical `tissue_annotation_column` / `included_tissue_annotations`, or skip (`config.py:733-736`) | — | Defaults depend on `grey_matter` | 0.25 |
| **Register images** (H&E) | • Gate on `${platform}_registered_images_csv` for any 10x platform (`main.nf:2286-2291, 3521-3533, 3594, 1029, 1678`).<br>• ATERA Literal (`config.py:565`).<br>• `registration_channel=none` | `reference_pixel_size_um` from the sidecar (`config.py:571`); exclude RGB from MIQ | Matrix and reader verified [H] | 1.5 |
| **Alignment / pairing** | Not needed in single mode (`main.nf:967-972`) | Role-based rewrite (`main.nf:3982-4307`, `alignment.nf`, `comparison.nf`, `config.py:1144-1216`, `materialize.py:90-118`); **ATERA always fixed** (`materialize.py:832-873` materialises > 100 GB) | Out of scope until a paired section exists | 10+ |
| **Comparison** | — (paired only) | Shared-gene renormalisation (`gene_comparison.py:186-194`); symbol join; platform-named columns; Q20 totals only | Panel-size dilution | in pairing |

---

## 6. Performance and resource estimates

**Workstation:**
- 739 GB RAM, of which about 264 GB is used by other work and 474 GB is available at the time of writing.
  The swap is full.
- 80 CPUs; 1× RTX A5000 (24 GB).
- Project storage has about 3.9 TB free. The system disk has about 330 GB, so work and temp directories
  stay on project storage.
- The Dwight local executor advertises 72 CPUs and 640 GB **for scheduling only. It does not enforce
  memory.** A runaway task meets the kernel OOM killer, which may kill another session's process. Hence
  the in-task watchdog (REQ).

### 6.1 Build stage (milestone 1)

| Item | Resource | Estimate | Basis |
|---|---|---|---|
| Transcript decode | CPU / IO | 4 columns over 1.40 B rows ≈ 45 s warm, single process; tile `2,1` in 5.1 s. Per-task ZipStores gave **no** parallel speedup in one page-cached test. Cold disk UNVERIFIED | Probe [M] |
| `tile_blocks` overhead (opt-in) | CPU / RAM | If reads stay serial: about 16 × 52 s ≈ 14 min warm, plus cold-disk cost. With the block-index pre-pass: one decode per tile, plus about 0.4 GB `.npy` per large tile | [M/L] |
| Points output | Disk | **17–20 GB** at 1.41e9 rows (snappy via spatialdata's writer) | Probe 11.7–12.1 B/row [M] |
| Points in memory per row | RAM | About 38 B/row (12 + 2 + 1 + 4 + 1 + 9 + 8 + 1) | [M] |
| Points write | Wall | pyarrow 4–6 M rows/s/thread; 15–45 min end to end | [M/L] |
| `morphology_focus` | Disk / CPU | 20.9 GB raw level 0 (4 ch); about 10–15 GB on disk; JPEG-2000 decode about 0.5–1.5 h (UNVERIFIED) | [L] |
| Labels | Disk / RAM | 2 × 10.4 GB raw, about 1–3 GB on disk; aligned 4096² chunks; pyramid memory UNVERIFIED | [M] |
| Mask polygonisation | RAM / CPU / disk | 10.4 GB memmapped `.npy` per mask (temporary) plus a `task.cpus` subprocess pool; minutes to tens of minutes (UNVERIFIED) | [M] |
| Acceptance streaming checks | RAM / time | About 85–103 s, < 3 GB RSS for both masks | Review run [H] |
| Table | RAM | About 7.8 GB peak process RSS, 12–16 s (AT-3 review; the CFM decode dominates) | [H] |
| Peak RSS | RAM | About 40–80 GB expected with `row_range` at 16 workers (about 1–1.5 GB per task × 16, plus polygonisation, image, table). **128 GB reservation**, watchdog cap at 120 GB, one retry at ×2 (with atomic writes) | [L] |
| Staged input | Disk | About 9 GB extracted plus 15.6 + 0.14 + about 0.5 GB of zips | [M] |
| Source zarr | Disk | About 35–50 GB (plus a transient `.partial` during the write) | [L] |
| BUILD wall time | Time | About 1.5–3 h | [L]; the pilot measures it |

**Measured on the public dataset (AT-5a, 2026-11-05).** These numbers replace the estimates above.
Nextflow `BUILD_SPATIALDATA`, 16 CPUs, shared host at load 15–30. The page-cache state of the bundle is
unknown: the crop had been cut from it earlier that day.

| Item | Measured |
|---|---|
| BUILD wall time | **8 min 24 s** task (8 min 36 s for the whole `nextflow run`); 852% mean CPU |
| Phases (from file times) | Masks, polygonisation and table: 2 min 27 s. Image write: 2 min 14 s. Labels (both pyramids): 31 s. Points: 2 min 43 s (1.41e9 rows, about 8.6 M rows/s). Shapes, table, sidecars, index and marker: 17 s |
| Peak memory | Watchdog: **11.7 GB anonymous RSS**, during the points write. The trace's `peak_rss` of 103.6 GB counts the two memory-mapped 10.4 GB masks once per polygonisation worker; it is page cache, not process memory. The largest anonymous level in the polygonisation phase was 11.4 GB |
| Label pyramids alone | 65 s and 4.75 GB anonymous RSS for both masks (5 levels, 16 threads); level shapes equal the morphology TIFF pyramid (42,548 × 61,369 down to 2,659 × 3,835) |
| Polygons (0.25 px) | Cells: 13 / 61 / 278 vertices (min / median / max), 3 MultiPolygons. Nuclei: 10 / 47 / 332, none |
| Store size | **38.8 GB** in all. Points 24.1 GB (17.1 B/row; the dask index column `__null_dask_index__` and `transcript_id` take about a quarter each). Image 14.0 GB. Table 493 MB. Shapes 56 + 41 MB. Labels 38 + 28 MB. Vendor sidecars and metadata 67 MB |
| Reads and writes | 23.7 GB read, 36.2 GB written (`rchar` / `wchar`) |
| Acceptance script | 6 min 26 s, 21.1 GB peak RSS (the int64 matrix comparison over 258 M nonzeros; the streaming schema check takes 4 min 46 s at 5.3 GB). With the checks added after the review: 7 min 37 s, 20.9 GB (vendor-mask pixel comparison 31 s, table check 39 s) |

**Resource defaults after AT-5:** `atera_build_memory_gb = 64` (128 GB on the one retry) and
`atera_build_max_rss_gb = 48`. The cap is about 4× the measured anonymous peak. Together with the
memory-mapped masks, the build's physical footprint is about 33 GB, so the reservation is about 2× it.
The full section was rebuilt with these defaults after the review: 9 min 22 s task time, 11.65 GB peak
anonymous RSS, 724% mean CPU (host load 6–10). `atera_build_cpus` stays 16. A section several
times larger scales the table and the masks; raise both values together if the watchdog stops the build
(exit 99).

### 6.2 Per section, downstream

| Item | Estimate | Basis |
|---|---|---|
| Native route: `latest` (hardlinked payloads) + viewer cache + plots | About 2 GB + 5–15 GB | [L] |
| Vendor-mask `.npy` export (native MIQ / prior) | +10.4 GB (cell), +10.4 GB (cell-keyed nucleus, optional) | [H] sizes |
| Total disk, native route (download + staging + source + latest + caches + `.npy`) | **About 130–160 GB**; worst case about 250–270 GB if `latest` is rewritten | [L] |
| Cellpose cells / nuclei | About 1.5 h each on the GPU; three 10.4 GB `.npy` | Xenium A at 2.46 Gpx took 1 h 22 min [M] |
| ProSeg, whole section | 1.5–3 days, 170–250 GB peak; CSV 77 GB; disk peak about 300–350 GB | [M/L] |
| ProSeg, tiled | About 8–12 h, about 10–25 GB per tile | [L] UNVERIFIED |
| ENRICH sjoin (today) | About 4–5 h single-threaded for 3 shapes with simple polygons (more with full-resolution contours); minutes with mask lookup | [M] |
| Clustering | About 20 GB dense transient today; < 100 MB with sparse means | [M] |
| Transcript spatial patterns | About 70–80 GB, about 15–20 h; about 1–1.5 h with the cap | [M/L] |
| VISUALIZE | About 12 x/y passes per branch, 15–40 min per branch; one pass less with attrs | [L] |

**Measured on the public section (AT-6 exit runs, 2026-11-06).** Nextflow, `-profile dwight,conda`, host
load 8–30, about 280–510 GB free. Time and peak RSS are the trace's `realtime` and `peak_rss` of the run
that did the work (a `-resume` re-runs ENRICH and the stages that read the latest store, as skips; their
numbers are in `stage_resources.json`). The native route uses no `.npy` export: MIQ reads the vendor
`cell_labels`.

| Process | Reserved | Crop: time / peak RSS | Full: time / peak RSS |
|---|---|---|---|
| BUILD_SPATIALDATA (reuse check only) | 16 CPU, 64 GB | 8 s / 1.1 GB | 9 s / 1.1 GB |
| NATIVE_LATEST | 8 CPU, 32 GB | 28 s / 1.3 GB | 35 s / 3.8 GB |
| ENRICH (native label check and counts) | 8 CPU, 64 GB | 38 s / 5.5 GB | 5 min 4 s / 11.7 GB |
| VIEWER_CACHE | 8 CPU, 60 GB | 26 s / 2.5 GB | 2 min 22 s / 9.6 GB |
| MASK_IMAGE_QUANTIFICATION | 4 CPU, 160 GB | 23 s / 2.0 GB | 3 min 9 s / 9.6 GB |
| QC | 4 CPU, 48 GB | 28 s / 2.4 GB | 1 min 15 s / 4.9 GB |
| VISUALIZE | 4 CPU, 64 GB | 40 s / 2.1 GB | 3 min 21 s / 4.5 GB |
| SPATIAL_GENE_ANALYSIS | 12 CPU, 120 GB | 1 min 1 s / 2.1 GB | 13 min 52 s / 22.8 GB |
| CLUSTERING_SQUIDPY (prepare + compute + finalize) | 64 / 96 / 64 GB | 2 min / 3.2 GB | 6 min 32 s / 19.4 GB |
| MENDER (prepare + compute + finalize + import) | 64 / 192 / 64 / 48 GB | 1 min 43 s / 1.0 GB | 6 min 30 s / 17.7 GB |

**ATERA ProSeg route (AT-9, `native_proseg`), measured 2026-11-06.** Both rows ran both layers
through MENDER. The crop ran at host load 10–55. The full section ran in one Nextflow run from
11:11 to 19:52 (8 h 41 min wall, 60 tasks, no failure, no retry), at host load 30–345 (other
sessions' test suites and, for the first 3 h 49 min, the 200 + 600 schedule check of `c2_r1` at
16 threads beside the 4 tiles); the lowest available host memory was 404 GB.

| Process | Reserved | Crop: time / peak RSS | Full section: time / peak RSS |
|---|---|---|---|
| ATERA_PROSEG_PLAN | 2 CPU, 16 GB | 11 s / 1.1 GB | 12 s / 1.5 GB (30 of 35 tiles run) |
| ATERA_PROSEG_TILE | 16 CPU, 32 GB (×2 on retry), 40 GB cap | 10 min 40 s / 5.5 GB (13.9 M rows); 17 min 50 s / 7.2 GB (29.2 M rows) | 30 tiles, 4 at once: 3 min 15 s (0.1 M rows) to 2 h 1 min (`c2_r1`, 102.6 M rows); peak 24.1 GB (trace; 22.7 GB anonymous). 1.69 B exported rows, 26.2 h of task time, tile phase 7 h 26 min wall. Earlier single-tile checks of `c2_r1` at a quieter time: 53 min 51 s / 24.2 GB at 16 CPUs, 40 min 15 s / 25.0 GB at 32 |
| ATERA_PROSEG_STITCH | 16 CPU, 64 GB | 30 s / 7.8 GB; a re-run on `-resume` is a 10 s skip | 6 min 29 s / 46.7 GB (a 42,548 × 61,369 raster, 409 points files, 1.41 B rows rewritten) |
| ENRICH (with the layer check) | 8 CPU, 64 GB | 46 s | 6 min 33 s / 13.6 GB |
| VIEWER_CACHE (both layers' pyramids) | 8 CPU, 60 GB | — | 3 min 17 s / 11.4 GB |
| MASK_IMAGE_QUANTIFICATION (both layers) | 4 CPU, 160 GB | 41 s | 6 min 40 s / 11.5 GB |
| VALIDATE_ANALYSIS_LAYER, per layer | 1 CPU | — | 21 s / 4.9 GB |
| QC, per layer | 4 CPU, 48 GB | as native | 2 min 14 s / 7.1 GB |
| VISUALIZE, per layer | 4 CPU, 64 GB | as native | 4 min 49 s – 5 min 22 s / 6.5 GB |
| SPATIAL_GENE_ANALYSIS, per layer | 12 CPU, 120 GB | as native | 14 min 18 s – 14 min 56 s / 22.8 GB |
| CLUSTERING_SQUIDPY (prepare + compute + finalize), per layer | 64 / 96 / 64 GB | as native | about 9.5–10 min / 19.4 GB |
| MENDER (prepare + compute + finalize + import), per layer | 64 / 192 / 64 / 48 GB | as native | about 10 min / 24.0 GB |

Stitch to MENDER on the full section takes 1 h 15 min wall for both layers. Per-tile numbers
(time, RSS, rows, owned and empty cells, prior weight, disk) are in
`$MERXEN_EVIDENCE_ROOT/atera_at9/tiles_full.tsv`; the stage table in `stage_table_full.md`.
Nextflow does not start the tiles in plan or size order: the largest tile still pending at
17:18 (`c2_r2`, 102.8 M rows) set the end of the tile phase. Starting the largest tiles first
would shorten it, but the order in which Nextflow releases tasks held by `maxForks` is not
controlled by the emission order; left as a possible improvement.

| Disk (full section, ProSeg route) | Measured |
|---|---|
| Latest store, bytes not shared with the source | **29.5 GB**: rewritten points 25.4 GB, clustering tables 1.07 GB (vendor) and 0.93 GB (ProSeg), image pyramid 0.74 GB, `table_original` 0.49 GB, `table_MOSAIK_proseg` 0.41 GB, ProSeg labels and their viewer pyramids 0.14 GB, shapes 0.10 GB, MIQ tables 15 MB |
| Tile results (30 tiles, inputs dropped) | 2.9 GB |
| Work directories | Clustering about 12 GB per layer (h5ad outputs), MENDER about 5 GB per layer, the rest under 1 GB |

The crop's rewritten points take 0.68 GB (no longer hardlinked to the source); the ProSeg labels,
table and shapes add 19 MB; each tile result keeps 40–181 MB (cells, claims, polygons, ProSeg's
cell metadata and table).

| Disk (full section) | Measured |
|---|---|
| Latest store, bytes not shared with the source | **2.56 GB**: clustering table 1.07 GB, `table_original` 0.49 GB, image pyramid 0.74 GB, label pyramids and outlines 0.08 GB, shapes 0.10 GB, metadata and sidecar copies 0.07 GB, MIQ table 7 MB. The 38.2 GB of payloads are hardlinks |
| Work directories | Clustering about 13 GB (h5ad outputs), MENDER about 6 GB, the rest under 0.3 GB |
| Total for the native route on top of the 38.8 GB source store | About 22 GB; source plus route about 61 GB (the 130–160 GB estimate above included the raw download and staging) |

The whole full-section route takes about 45 min of task time (40 min wall for the first run up to the
MENDER failure, then 14 min for the resumed run). MIQ, ENRICH, VIEWER_CACHE and QC use under 15 GB, far
under their reservations; the reservations are left as they are (directives are not hashed, so they can
be lowered for ATERA later without a re-run).

---

## 7. Testing plan

### 7.1 Synthetic mini Atera fixture

`tests/test_io/atera_synth.py`, generated into `tmp_path`, with no committed binaries. The prototype writes
about 270 KB in about 0.3 s under zarr 3.1.6 and 3.4.1 [H].

```python
write_atera_bundle(root, *, tiles={"0,0": 40, "-1,1": 3, "1,0": 0, "0,1": 25},
                   grid_keys_order=("0,1", "-1,1", "0,0", "1,0"),   # not lexicographic
                   n_cells=5, unassigned_fraction=0.25, low_qv_fraction=0.2,
                   n_channels=2, file_order_differs=False, with_is_sketch=False,
                   multifile_ome=False, prefixed_companions=False, seed=0) -> AteraSynth
```

**Invariants reproduced:**
- `experiment.spatial` with `pixel_size`, `analysis_sw_version="atera-1.1.1.0"`, `panel_organism`,
  `segmentation_stain`, file paths, and **obviously fake** `run_name`, `slide_*`, `region_name`,
  `instrument_sn` and UUIDs. Never `P` + four digits.
- **Transcripts:**
  - Zarr v2 zips (STORED); root attrs `RnaDataset` major 6.
  - `gene_category` / `codeword_category` [n, 9] bool, F order, with `column_names` in a **non-default
    column order** variant.
  - Several codewords per gene; every control class.
  - Rows sorted by gene; end-exclusive `gene_offset`; F-order `location` / `cell_id`; chunks of about 16
    rows.
  - A negative tile key; an empty tile; `grid_keys` attribute order ≠ sorted order.
  - Unassigned = `(0xFFFFFFFF, 0xFFFFFFFF)` (the prototype wrote zeros; fixed).
  - QV < 20 rows never assigned; transcripts at negative x and y.
- **Cells:**
  - Realistic prefixes with suffix 1.
  - `cell_summary` with NaN nucleus fields for a nucleus-less cell.
  - A two-nucleus cell, where `nucleus_area` = the largest nucleus.
  - Masks consistent with `polygon_sets`, using the pixel-corner convention: centroids = (mean index +
    0.5)·ps, and vendor polygons at the pixel-centre contour.
  - A float32 `homogeneous_transform`.
  - One polygon with `num_vertices == 0`.
  - One cell whose label has two connected components.
- **CFM:**
  - `anndata.io.write_elem` into a zarr v2 group in a ZipStore.
  - `barcode` index; packed `obs.cell_id`; ENSG `var` index with `feature_name` / `feature_type` /
    `filtered` / `highly_variable` / one `de_*`.
  - One ENSG-shaped symbol; Azimuth and Leiden columns; `X_pca` / `X_umap`.
  - `is_sketch` absent by default; control columns in X (including deprecated); `total_counts` including
    deprecated.
- **`panel_config.json`:** `customer{email,name}` and `designer.designed_by` stubs (fake values) and
  `gene_coverage`.
- **Reports and companions:** `analysis_summary_<fake>.html`, `qc_images.html`, both FOV JSONs, a fake
  `overview_scan.png`, `diffexp_genes/`, and prefixed H&E alignment and keypoints CSVs.
- **Morphology:**
  - 2 single-channel pyramidal JPEG-2000 OME-TIFFs, with odd level shapes.
  - Variants: `n_channels=4`; `file_order_differs=True`, where OME `TiffData` maps files to channels in a
    different order from file names.
  - A true multi-file OME header is optional (`multifile_ome=True`); how tifffile writes cross-file
    `TiffData` is UNVERIFIED.

### 7.2 Fast tier (unmarked; pre-push and CI)

- **Bundle resolution:**
  - From `experiment.spatial` (un-prefixed, the in-house case) and from globbed prefixed companions.
  - The spec override is metadata-only.
  - Unknown major fails; species mismatch fails.
  - An Atera preview bundle is refused by the XENIUM builder.
- **Transcripts:**
  - `gene_offset`-derived gene == codeword-derived gene.
  - Known categories in `meta` and every partition.
  - `is_gene` selected by column name.
  - Sentinel → NA. The `original_assignment` lookup is correct with shuffled cell keys (argsort path) and
    fails on an unknown ID.
  - QV filter; vendor-ordinal IDs follow the **attribute** key order, are unique, stay stable under
    `min_qv`, and need no extra pass.
  - Negative coordinates kept; empty tile → no partition; partitions never cross tiles.
  - `row_range` and `tile_blocks` give the same multiset of rows; tiny blocks merge.
  - Aggregates (Q20 per-gene totals, QV histograms) match a direct count.
- **Pass and concurrency budget:**
  - A graph-execution counter asserts **one** read per tile slice for the whole build write, including
    `PointsModel.parse` (fails if `sort` is omitted).
  - A zarr-array wrapper asserts no full-tile `location` read in `row_range` mode.
  - Dask `num_workers` and zarr `max_workers` are set.
  - The `tile_blocks` semaphore bounds concurrent tile decodes.
- **Cells and shapes:**
  - Cell label k ↔ row k−1; nucleus label k ↔ nucleus row k−1, with correct `cell_instance_id`.
  - **Mask-derived shape centroids match `cell_summary` within 0.01 µm** (half-pixel test); the
    vendor-polygon fallback too.
  - `instance_id` is present as both column and index on both shape elements.
  - A multi-component label yields a MultiPolygon.
  - A polygonisation pool failure raises when `serial_fallback=False`.
  - `homogeneous_transform` passes at float32 precision and a non-scale matrix fails.
- **Pixel convention:** `assign_labels_from_masks(pixel_convention="corner")` reproduces the vendor
  assignment on the fixture; the `"center"` default is unchanged for existing callers.
- **Table:**
  - Gene-only X; per-type control counts derived (deprecated included).
  - Symbol `var_names`, `gene_ids` = ENSG; `vendor_*` renames, each optional.
  - `highly_variable` absent; `vendor_gene_coverage` joined.
  - `nucleus_area` = largest; `nucleus_area_total` = sum.
  - Same `source_cell_id` on shapes and table.
- **Nucleus join:** after clustering ingest of the native store, the two-nucleus cell gets its own
  aggregated nucleus area, never nucleus k's area. A grep test lists nucleus-shape consumers.
- **Images:**
  - Channel names and order come from OME `TiffData` (the file-order variant).
  - Stored levels are used; level transforms come from shape ratios; image and label level shapes are
    equal.
  - A missing sibling is an error.
- **Metadata (D13):** every sidecar is a byte-identical copy under its stable name, with its original
  name, size and SHA-256 recorded; `attrs["merxen_atera"]` carries the full `experiment.spatial`, every
  metrics column and the `panel_config.json` header (customer and designer included), all checked against
  the fixture's fake values; two files with one stable name fail. The zarr metadata sidecars hold the
  zips' members byte for byte, the OME-XML sidecars equal the TIFF headers, and the vendor store attrs,
  map annotations, `de_*` statistics and colours reach the store.
- **Atomicity:** kill the writer mid-write (monkeypatched exception), rerun, and assert a rebuild. A
  fingerprint mismatch refuses without `--force-rerun`.
- **Contract and config:**
  - `prepare_source_spatialdata_contract(platform="ATERA", points_canonical=True)`, then
    `validate_merxen_schema(deep=False)` and the streaming deep check.
  - `SpatialDataBuildConfig` accepts ATERA; dispatch through a fake module
    (`tests/test_io/test_spatialdata_build.py:76-110`).
  - **XENIUM and MERSCOPE contract outputs unchanged** for the opt-in fast paths, including MERSCOPE
    category order.
- **Controls:** `control_features` ATERA cases, including `Intergenic_Region_*` and `is_boosted_gene`;
  `apply_dataset_filter` uses the registry.
- **Text tests on `main.nf` and the config:**
  - XENIUM/MERSCOPE `buildConfigForPlatform` JSON is byte-identical
    (`tests/test_workflows/test_spatialdata_build_config.py:22-68`), with `workflows/lib/` on the harness
    classpath.
  - **Pins on every existing process `script:` text.**
  - ATERA arm present; new params declared, including the six Cellpose names.
  - Metro-map colour.

### 7.3 `integration` and `nextflow` tiers (CI full tier)

- **`integration`:** build an on-disk store from the fixture with `write_atera_spatialdata`, re-read it with
  `sd.read_zarr`, and check:
  - elements, transforms and attrs;
  - sidecars (byte-identical, stable names) and the full header in attrs;
  - the points index matches the parquet files;
  - the pixel size resolves via the shared helper;
  - the lost-known-categories behaviour on re-read is handled by consumers.

  From AT-6, the native route plus a test that ENRICH leaves the source store's inodes and bytes unchanged,
  and the viewer cache on that store (`tests/test_viewer_cache/test_build.py:43-89`).
- **`nextflow`:**
  - A harness run of `BUILD_SPATIALDATA` with an ATERA row.
  - A harness for `runPreflightChecks`: an ATERA row with per-row `stop_stage=build_spatialdata` passes
    `CellposeSettings.preflight`; one without it fails with the documented message, including in a mixed
    samplesheet.
  - **The cache gate is the script-text pins plus golden rendered JSON, not stub `-resume`.** Stub runs
    hash the `stub:` block (Nextflow behaviour, UNVERIFIED here), and only `pciseq.nf` has stubs today.

### 7.4 Real-data checks (`slow` and acceptance scripts; never in CI)

- **`slow` pytest:**
  - `real_path("MERXEN_ATERA_RAW_ROOT")` (`tests/conftest.py:102-128`); skips when unset.
  - Opens the real zips and reads tile `5,4` (5,756 rows) and `obs` / `var`.
  - Checks gene order, packed IDs, sentinel, `gene_offset` vs codeword gene, OME `TiffData` mapping, and
    ENSG-shaped symbols vs `gene_ids` helpers.
- **`scripts/acceptance/atera_build.py`** runs on the built store re-read from disk and records results under
  `$MERXEN_EVIDENCE_ROOT` (aliases only):
  1. Streaming per-cell × feature counts (QV ≥ 20, assigned, **all features**) vs CFM X: exact on every
     column with transcripts (638,849,843 counts). The vendor total, 638,849,927, also holds 84 counts in
     four `DeprecatedCodeword_*` columns without any transcript (§1.1 item 3); `--expected-vendor-excess`
     pins that number and is required.
  2. Exhaustive mask checks: area equality, centroids at (i + 0.5)·ps within 0.005 µm, nucleus containment
     0 violations, max labels, no empty label.
  3. Shape centroids vs `cell_summary` within 0.01 µm; vertex statistics.
  4. Points row count = number of zarr transcripts with QV ≥ 20; per-tile reconciliation.
  5. Metadata completeness: sidecars byte-identical to the bundle, attrs header equal to
     `experiment.spatial`, metrics equal to the CSV text, zarr metadata members listed from the zips;
     evidence written under aliases only.
  - Added after the AT-5 review: the recorded build options must be the §1.1 ones (`min_qv` 20,
    0.25 px, no allow-list); every label pixel equals the vendor `masks/1` / `masks/0`; one shape per
    cell indexed by its label; the table keeps `instance_id` = vendor row + 1, the vendor gene counts,
    every vendor `obs` column, the `cell_summary` values, the vendor PCA/UMAP, `varm["vendor_de"]` and
    the category colours; no `.partial-*` folder is left; the report records the git revision.
  6. Peak RSS, wall time and per-element size from `trace.tsv` and the watchdog log.
  7. `row_range` vs `tile_blocks` timing and memory on the crop bundle.
- **Crop bundle (AT-5a):** a 2–3-tile real-data bundle under `$MERXEN_ATERA_RAW_ROOT`, cropped from tiles,
  cells, masks and the image window. It is the end-to-end acceptance input for later milestones.
- **Optional parity:** the public 10x `tiny_atera_dataset` (atera 1.0.0, about 3.5 MB; major-6
  compatibility UNVERIFIED), and `spatialdata_io.atera()` once released. Slow tier only (Q21).

### 7.5 Environment matrix

- Fast tier against the lock (CI: zarr 3.1.6, anndata 0.12.10).
- Integration and slow tiers in a fresh Nextflow conda env (zarr 3.4.x, anndata 0.12.19).
- No lockfile change is planned. If `src` ever imports `numcodecs`, declare it (it is locked at 0.16.5) and
  follow the regeneration steps, including the undocumented git-pin override file.

---

## 8. Milestones and PR breakdown

**Branching:**
- AT-0 to AT-2 are behaviour-neutral or bugfix PRs straight to `main`.
- Atera work (AT-3 onward) uses the integration branch `feature/atera-support` with work-package branches
  `feature/atera-<wp>-<slug>`. `main` is merged in regularly; the branch is never rebased.
- This needs a CLAUDE.md exception line like the RCA one (Q13).
- Acceptance-gate PRs merge into `main` after AT-5 and AT-6.
- **RCA merged in (2026-11-06, user decision):** `feature/robust-celltype-annotation` was merged into
  `feature/atera-support` (never the other way) and is frozen; it never merges into `main` by itself.
  RCA work continues on this branch (branches `feature/rca-m<N>-<slug>`, CLAUDE.md exception line). The
  next Atera acceptance-gate PR into `main` therefore also carries RCA: the human and mouse `map_first`
  default for XENIUM and MERSCOPE rows (RCA gates H and M) and version 0.2.0. It needs RCA's gate H/M
  sign-off as well as the Atera exit checks; RCA plan §2.2 left that flip to merge into `main` "only
  when you say so" (U10).

**Conventions:** commit prefixes follow Agents.md; PR text goes through
`scripts/privacy/check_private_text.py --stdin`.

**Coordination:**
- `feature/manual-registration` and `feature/robust-celltype-annotation` both edit the `config.py`
  Literals and `main.nf`. Land AT-1 after manual-registration.
- Do AT-8 clustering on top of RCA (RCA is on this branch since the merge above).
- ATERA rows are held off RCA's `map_first` annotation until AT-11 (below): a row without its own
  `clustering_squidpy_mode` clusters in `legacy` mode, `map_first` on the row is refused, and
  `--annotation_prepare_only` builds no reference bundle for ATERA rows.

**Cache invariant (every milestone):**
- Existing XENIUM and MERSCOPE configs and process `script:` text stay byte-identical.
- ATERA behaviour lives in ATERA-only JSON keys, directive closures, Python behind opt-in flags, or new
  processes.
- **The one announced exception** is the `qc.nf` reference-shape edit in AT-6 (Q17).

### AT-0 `[docs]`: this plan (0.5 d)

- **Scope:**
  - `docs/plans/atera-support-plan.md`, linked from `docs/index.md`.
  - `MERXEN_ATERA_RAW_ROOT` in `local_config.KNOWN` and `.env.example`.
- **Tests:** privacy hook; template-order test.
- **Exit:** merged to `main`.

### AT-1 `[refactor]`: platform registry, MERSCOPE and XENIUM only (5 d)

- **Scope:**
  - Guard tests first: golden rendered JSON for BUILD / SEGMENT / ENRICH / VIEWER_CACHE / QC through the
    harness, with `workflows/lib/` on its classpath; pins on every process `script:` section; parity across
    the three `analysis_mode` maps.
  - Then `platforms.py` and `PlatformSpec.groovy`, and the ternaries outside `script:` blocks moved to
    lookups. `qc.nf` stays untouched until AT-6.
  - The grep tests for shape-name literals and nucleus consumers.
- **Exit:** merged to `main`; no behaviour or hash change.

### AT-2 `[bugfix]` / `[perf]`: shared-code fixes on `main` (4.5 d)

- **Scope:**
  - Control registry (`intergenic`, ATERA pattern, `apply_dataset_filter`).
  - Opt-in contract fast paths (`points_canonical`, `normalize_points_for_latest_write`,
    `_canonicalize_proseg_points`).
  - `with_stable_transcript_ids(partition_lengths=)`; streaming deep validation.
  - `assign_labels_from_masks(pixel_convention=)`.
  - `masks_to_labeled_polygons(serial_fallback=, mp_context=)`.
  - Enrichment COO accumulation (byte-identical on fixtures).
- **Tests:** Xenium and MERSCOPE equivalence tests (including MERSCOPE category order).
- **Exit:** each PR states which published outputs become stale. The Intergenic/Deprecated fix changes
  Xenium gene summaries.

### AT-3 `[feature]`: fixture and Atera readers (5 d)

- **Scope:** `atera_synth.py`, `io/atera_reader.py` (including OME `TiffData` mapping, mask polygonisation
  with +0.5 px, and the verbatim metadata sidecars and full header of D13).
- **Tests:** §7.2 reader, shape, pixel-convention, metadata and pass-budget tests.
- **Exit:** fast tier green on the lock env and a fresh conda env.

### AT-4 `[feature]`: builder and BUILD plumbing (6.5 d)

- **Scope:**
  - `io/builders/atera.py` and `AteraBuildConfig`.
  - Atomic write, fingerprint and reuse guard; RSS watchdog.
  - Dispatch, sidecars, points index, aggregates.
  - `find_um_per_px` consumers (the ATERA arms of the enrichment and viewer-cache lookups are AT-6);
    preview-bundle refusal; licence notices.
  - `main.nf` mode, input, build arms and preflight.
  - Directive closures; params, including the six Cellpose names.
  - Samplesheet example, metro map, docs.
- **Tests:** §7.2 contract, config, atomicity and metadata tests; §7.3 integration and nextflow.
- **Exit:** a fixture build validates; existing configs and scripts are byte-identical.

### AT-5 `[test]` / ops: real-data pilot and milestone-1 acceptance (4 d)

- **Scope:**
  - (a) Crop bundle, then the full public-dataset build with `stop_stage=build_spatialdata`. Run the
    acceptance script; tune resources from `trace.tsv` and the watchdog; decide whether to promote
    `tile_blocks`; measure label-pyramid memory and polygon vertex counts.
  - (b) **Experiment, not a merge:** ProSeg 3.2.0 benchmark on the crop with the vendor-mask prior (floor
    lookups) and QV ≥ 20 parquet. It measures peak RSS, time per iteration at 18k genes, and neuropil
    absorption. The result decides AT-9.
- **Exit:** all §1.1 criteria met and recorded. **This is milestone 1.**
- **Status (2026-11-05):** (a) is done: crop pilot, full build, acceptance, atomicity check, resource
  tuning and the `tile_blocks` decision (§1.1 status, §3.3, §6.1). (b) is done; evidence in
  `$MERXEN_EVIDENCE_ROOT/atera_at5/proseg_benchmark/`.
- **AT-5(b) result.** ProSeg 3.2.0 (the pinned binary), vendor cell mask as the prior, MerXen SEGMENT
  settings, 32 threads, one run at a time. Besides the crop (a sparse corner: 3,013 cells in 8 mm²) it
  ran on the densest 2 mm tile of the full section with a 100 µm halo, because the crop is a poor basis
  for extrapolation.

  | Run | Transcripts | Schedule (burn-in + samples + final) | Wall | Burn-in / sampling s per iteration | Peak RSS |
  |---|---|---|---|---|---|
  | Crop | 40.5 M | 100 + 200 + 20 | 15 m 29 s | 0.91 / 2.53 | 8.8 GB |
  | Crop, repeat | 40.5 M | 100 + 200 + 20 | 15 m 50 s | 0.90 / 2.62 | 8.8 GB |
  | Crop, MerXen default | 40.5 M | 200 + 1200 + 50 | 1 h 04 m 47 s | 0.90 / 2.73 | 8.9 GB |
  | Crop, weak prior (effective weight 1) | 40.5 M | 100 + 200 + 20 | 15 m 54 s | 0.96 / 2.57 | 8.8 GB |
  | Dense tile | 102.4 M | 100 + 200 + 20 | 37 m 42 s | 1.76 / 5.83 | 21.7 GB |

  - Time and memory are linear in transcripts (about 227 B per transcript). Calibrated on the dense
    tile (about 15% optimistic against the crop), the full section costs:

    | Scenario | MerXen default schedule | Short schedule (100 + 200) |
    |---|---|---|
    | 32 tiles of 2 mm, one at a time | about 39–45 h | about 10–12 h |
    | 3–4 tiles at once (UNVERIFIED) | about 12–15 h | about 3–4 h |
    | Whole section in one run | about 32–37 h, about 300 GB | about 9 h, about 300 GB |

  - Neuropil absorption stays small under the vendor prior (R15). At the automatic prior weight
    (134–146 here) ProSeg takes 15–18% of the vendor-unassigned transcripts into cells, half of them
    within 1.7–2.4 µm of a vendor cell, and releases 20–25% of the vendor-assigned ones: median area
    ratio 0.80–0.86 against the vendor masks, total cell area −2.5% to +1.8%. Only a prior weakened to
    weight 1 grows cells (median 1.22×, total area +36%). On the dense tile 74 of 6,877 cells end with
    no transcripts.
  - The 1200-sample default bought nothing measurable on the crop: it agrees with a 200-sample run on
    79.2% of per-transcript assignments, as two 200-sample runs agree with each other (79.6%). About 20%
    of boundary assignments flip between runs whatever the schedule.
  - Whether ProSeg improves on the vendor segmentation is untested: there is no ground truth here.
- **AT-9 decision (from AT-5b): go, with tiling, as an opt-in route** (AT-9 below).

### AT-6 `[feature]`: native segmentation route end to end (11 d)

- **Scope:**
  - `segmentation_source`; `NATIVE_LATEST` (payload hardlinks, copied metadata).
  - SEGMENT tuple placeholders and the restart channels.
  - ENRICH native mode; MIQ with the vendor mask.
  - Nucleus-join REQ in clustering.
  - QC (including the `qc.nf` edit with the announced re-run), viewer-cache and visualization REQ items.
  - Clustering dotplot guard and sparse means.
  - Transcript patterns, MECR and cortical depth off for ATERA.
  - Literals widened for these stages.
- **Tests:** registry contents; source-inode test; `-preview` with `start_stage=enrich`; harness runs;
  script pins.
- **Exit:** the crop, then the full section, run in native mode through VISUALIZE and clustering.
- **Status (2026-11-06): exit met.** Both runs pass through clustering and MENDER (§1.4, §6.2). User
  decisions applied in AT-6: pciSeq off and untested for ATERA; GASTON and standalone MapMyCells refused;
  MECR off until it streams (with RCA); transcript spatial patterns off; cortical depth and distance from
  object off for non-cortical sections; no transcript caps or subsampling (figures are density rasters
  over every transcript the build kept, i.e. every QV >= 20 row, and the viewer reads every one of them).
  One figure bounds genes rather than transcripts: a branch dotplot of a panel wider than 400 genes draws
  each group's top 5 genes (the CSV tables keep every gene; approved 2026-11-06, together with extending
  the top-5 selection to every panel size). The `qc.nf`
  reference-shape edit changes the QC task hash of existing XENIUM and MERSCOPE rows once (Q17); the
  MENDER env pin changes the MENDER_COMPUTE hash of every platform once (Appendix A, "AT-6 exit runs";
  awaiting the user's OK). The broad atlas labels of ATERA rows are provisional (Appendix A, "AT-6
  reviews").

### AT-7 `[feature]`: `cellpose_only` for Atera (5 d)

- **Scope:** segmentation REQ items; CSV export out of the GPU lock or skipped; enrichment mask lookup;
  `convert_to_latest` fast paths verified on ATERA.
- **Exit:** a `reseg` layer on the crop and on the full section.

### AT-8 `[feature]`: whole-transcriptome analysis paths (8 d)

- **Scope:** clustering HVG gate; GASTON ranking; MapMyCells region config and lean query; pciSeq
  guardrails; capped transcript patterns; MECR streaming reference; shared "analysis gene set". Guards key
  on gene count, not only the platform (Q22).
- **Exit:** the default analysis stages run on the full section within their resources.
- **Status (2026-11-07): in progress, in five work packages** (gene-count guards; analysis gene set; MECR
  streaming reference; raster transcript patterns; exit run). User decisions of 2026-11-07 in Appendix A
  ("AT-8 user decisions"). **WP1 (gene-count guards) done:** `merxen.gene_sets` holds the shared rule
  (more than 5,000 non-control genes is whole-transcriptome, so Xenium Prime 5K counts). Legacy clustering
  runs the PCA on 2,000 `cell_ranger` highly variable genes for such a panel when the sample sets no
  `hvg_n_top_genes`, and records why in `uns`; GASTON raises instead of keeping the first
  `gaston_max_genes` genes; pciSeq refuses an atlas reference without an explicit `gene_names` list on such
  a panel. The rules live in Python: no sample JSON or process script changes, so the clustering, GASTON
  and pciSeq task hashes of every row are unchanged and the 300-gene XENIUM/MERSCOPE outputs are unchanged.
  A cached 5K clustering task is not re-run by `-resume`. As with any edit of
  `analysis/clustering_squidpy.py` or `config.py`, the map_first hierarchy fingerprint
  (`HIERARCHY_SOURCES`) changes, so the annotation `COMPUTE_CPU` task of every row re-runs once (same
  labels expected: the hierarchy does not run the legacy HVG rule). The map_first QC embedding's own HVG rule (`seurat`, above 1,000 genes) is
  unchanged.

### AT-9 `[feature]`: ProSeg for Atera, conditional on AT-5b (8–12 d)

- **Decision (AT-5b, 2026-11-05): go, as an opt-in route.** The native route stays the default. No
  single whole-section run (about 300 GB, 32–37 h, no checkpoint).
- **Requirements from the benchmark:**
  - Tiles of 2 mm with a halo (100 µm benchmarked); cells at tile borders go to the tile that owns
    their label.
  - Input as parquet in **Xenium layout** (plain-string `feature_name` and `cell_id`, uint64
    `transcript_id`, uint8 `overlaps_nucleus`, string `fov_name`, float32 x/y/z/qv). A plain parquet
    makes ProSeg 3.2.0 stop with "No parquet file format specified"; CSV is 52 B/row (about 73 GB and
    about 2 h to write for the section) against 8.8 B/row. The Xenium layout also turns on ProSeg's
    nucleus-compartment model, which the CSV route does not use.
  - An explicit `--prior-weight`, not the automatic value.
  - Mask transforms `[ps, 0, col0·ps]` and `[0, ps, row0·ps]`: ProSeg samples pixel centres, which
    matches the floor convention of the vendor masks.
  - The short schedule (100 burn-in + 200 samples) by default, after checking it on one dense tile.
  - Report the number of cells left with no transcripts.
- **Status (2026-11-06): implemented; the crop and the full section run end to end.** Opt-in
  `segmentation_source=native_proseg` (aliases `vendor_proseg`, `instrument_proseg`); `native` stays
  the default. The segment stage is `NATIVE_LATEST`, `ATERA_PROSEG_PLAN`, one `ATERA_PROSEG_TILE`
  task per 2 mm tile (100 µm halo, label-owned: a vendor label belongs to the tile whose core holds
  its vendor centroid) and `ATERA_PROSEG_STITCH`, which writes the layer `reseg` into the latest
  store (shapes `MOSAIK_proseg`, table `table_MOSAIK_proseg`, labels `proseg_cell_labels`, points
  columns `assignment` and `background`; registry branch `proseg`, alias `reseg`, with `label` and
  `prior`). ProSeg instance ids are the vendor labels. The effective prior weight is fixed at 140
  per tile (the multiplier undoes ProSeg's own scaling). ENRICH checks the layer; MIQ, the viewer
  cache, QC, VISUALIZE, spatial gene analysis, clustering and MENDER run on both layers with
  `analysis_segmentation=all`. Docs: `docs/stages/segmentation.md` (ATERA ProSeg route). Evidence:
  `$MERXEN_EVIDENCE_ROOT/atera_at9/`.
  - **Crop (`ATERA_PUB_CROP`, 2 tiles, one Nextflow row through MENDER):** 32 tasks, no failure.
    Tiles: 13.9 M and 29.2 M rows, 10 min 40 s and 17 min 50 s at 16 threads (sampling 1.9 and
    3.0 s per iteration on a host at load 20–55), 5.5 and 7.2 GB peak RSS; reported prior weight
    139.57 and 139.59. Stitch 30 s, 7.8 GB: 3,013 cells (every vendor label), 4 with no
    transcripts, 18,138,163 assigned rows, 292 seam conflicts (0.0016%). The verification passes:
    source store unchanged (927 files), points files and row counts unchanged and the build's points
    index still matching, table and shape ids equal, every assignment a table id, int64 row sums
    equal to the foreground counts, ENRICH check recorded, both layers through MIQ, the viewer cache,
    QC, VISUALIZE, spatial gene analysis, clustering and MENDER. A `-resume` re-runs no tile; the
    stitch task re-runs as a 10 s skip (fingerprint), ENRICH and its check skip.
  - **ProSeg against the vendor layer (crop):** ProSeg assigns 44.6% of gene transcripts (vendor
    46.5%), absorbs 15.4% of the vendor-unassigned ones and releases 21.7% of the vendor-assigned
    ones (70.2% stay in the same cell, 8.1% move); median area ratio 0.83 (p05–p95 0.49–1.83),
    total area −2.2%. These match the CSV benchmark (`run_s200`: 45.4%, 16.6%, 21.5%, 0.85), so the
    nuclear compartment model the Xenium parquet layout turns on does not change the result on the
    crop (Q3 of the AT-9 design: compartment model kept on).
  - **Dense-tile check (`ATERA_PUB` tile `c2_r1`, the benchmark's [2,1]; `stop_stage=segment`,
    `atera_proseg_tile_subset=c2_r1`, 16 threads):** 102.6 M rows exported, 102.4 M kept, 7,008
    labels in the window, 5,716 owned. 53 min 51 s task time (ProSeg 48.8 min: burn-in 2.5 s and
    sampling 8.8 s per iteration, against 1.76 and 5.83 s at 32 threads in the benchmark), 22.7 GB
    peak anonymous RSS (trace 24.2 GB) against the 32 GB reservation and the 40 GB cap. Reported
    prior weight 139.77 (benchmark auto weight 146.17). The recount equals ProSeg's count matrix;
    39 owned cells (0.7%) end with no transcripts; no owned cell reaches an interior window edge.
    Against the CSV benchmark run on the same owned cells: median area ratio 0.841 (benchmark
    0.844), 41.7 M against 41.1 M assigned rows, and 57.7% of the rows either run assigns to these
    cells are assigned to the same cell by both (two CSV runs on the crop: 62.1%; ProSeg's chain
    noise, AT-5(b)). The compartment model therefore stays on. Re-run at 32 threads
    (`atera_proseg_tile_cpus=32`): 40 min 15 s task time, burn-in 1.68 s and sampling 5.82 s per
    iteration (benchmark 1.76 and 5.83 s; total 1.07× the benchmark's 37 min 42 s), 23.5 GB peak
    anonymous RSS (trace 25.0 GB), 73 owned cells with no transcripts (benchmark: 74). Halving the
    threads costs 1.34× the time, so the default stays 16 threads with 4 tiles at once.
  - **Schedule check on the dense tile (2026-11-06): the short default stays.** The same tile
    `c2_r1` at 200 burn-in + 600 samples (+ 20 hillclimb, 100 recorded; one run, 16 threads, at
    nice 15 beside the full-section run, host load 30–175) against three short-schedule runs (the
    16- and 32-thread checks and the full-section tile), on the 5,716 owned cells
    (`dense_schedule_compare.json`):

    | | Short runs (3) | Long 200 + 600 |
    |---|---|---|
    | Foreground rows on owned cells | 41.7–43.3 M | 42.2 M |
    | Owned cells with no transcripts | 39–73 | 50 |
    | Median area ratio to vendor (p05–p95) | 0.841–0.852 (0.47–0.49 to 1.56–1.64) | 0.811 (0.45–1.78) |
    | Total area change | −1.6% to −0.9% | −0.6% |
    | Same cell among rows either run assigns, short vs short / short vs long | 58.5–59.2% | 57.1–57.8% |
    | Same cell when both assign | 84.7–85.0% | 83.7–84.0% |
    | Per-cell count Pearson, short vs short / short vs long | 0.991 | 0.984–0.985 |
    | Per-cell area Pearson | 0.977–0.978 | 0.961–0.964 |

    The long chain differs from the short ones a little more than the short ones differ from each
    other (about 1.3 points of agreement), and it widens the area distribution (the same direction
    as the crop's 1,200-sample run in AT-5(b): median 0.80 against 0.85, p95 2.11 against 1.94).
    Totals, assigned rows and empty cells stay inside the short runs' spread. The long run took
    3 h 49 min wall (ProSeg sampling 16.8 s per iteration under contention), 3.4 times the short
    schedule's ProSeg time, for a shift smaller than the chain noise between two short runs, so the
    default stays 100 + 200 + 20. Longer schedules remain available through the
    `atera_proseg_*` schedule params.
  - **Full section (`ATERA_PUB`, one Nextflow row through MENDER on both layers, 2026-11-06):** 60
    tasks, no failure, no retry, no prior-weight restart; 8 h 41 min wall on a host at load 30–345
    (§6.2). 30 tiles, 1.69 B exported rows (halos included), reported prior weights 138.76–139.99.
    The stitch keeps **91,431 ProSeg cells, every vendor cell once** (no vendor label without a
    ProSeg cell, no owned cell without a polygon), **806 (0.88%) with no transcripts**, 572.8 M
    assigned rows, 160,659 seam conflicts (0.028% of the assigned rows; 1,592 cells lose some),
    no owned cell near an interior window edge. The verification
    (`verify_full.json`) passes: source store unchanged (3,352 files: inode, size, mtime), points
    files and rows unchanged (409 files, 1,409,601,228 rows, every transcript, with
    `original_assignment`, `assignment` and `background` for the viewer), points index fingerprint
    still matching, table and shape ids equal and unique, all 91,431 polygons valid and non-empty,
    every assignment a table id, table row sums equal to the per-cell foreground counts of the
    points (572,765,066), registry entry `proseg` (label `proseg_cell_labels`, prior
    `vendor_cell_labels`, ProSeg 3.2.0 identity), ENRICH layer check recorded, viewer pyramids of
    both layers, MIQ and clustering tables of both layers, and VISUALIZE, spatial gene analysis,
    clustering, MENDER and QC outputs for both layers (QC as `atera/<layer>/qc/qc_out`, 5 files each; the
    first version of the check looked in a folder that does not exist and is replaced by the
    final review's re-run).
  - **ProSeg against the vendor layer (full section, `layer_compare_full.json`):** ProSeg assigns
    **40.6% of gene transcripts (vendor 45.3%)**; it absorbs 12.9% of the vendor-unassigned ones and
    releases 25.9% of the vendor-assigned ones (66.0% stay in the same cell, 8.1% move to another);
    **median area ratio 0.81** (p05–p95 0.48–1.71), total cell area −3.2%. Denser than the crop
    (44.6% / 46.5%, area ratio 0.83), as the AT-5(b) dense tile predicted (46.5% / 53.1%, 0.83).
  - **Clustering of the two layers (`cluster_compare_full.json`):** clustering QC keeps 88,960
    ProSeg cells against 91,381 vendor cells (median 4,323 against 4,812 counts and 2,291 against
    2,687 genes per cell); both give 12 Leiden clusters and 19 MENDER domains. On the 88,958 shared
    cells (ProSeg ids are the vendor labels) Leiden ARI 0.88 (NMI 0.85), broad class ARI 0.87 with
    93.5% the same label, neuron split 96.7% the same, MENDER domains ARI 0.65. Both layers agree as
    well with the vendor Azimuth broad labels (ARI 0.659 ProSeg, 0.664 vendor). The class that
    moves is the vendor layer's mixed "Fibroblasts" cluster (7,244 cells, against 4,748 on ProSeg):
    in ProSeg 752 of its cells go to Vascular cells and 611 to Microglia; and 1,730 vendor
    "Oligodendrocyte precursors" (mature oligodendrocytes, AT-6 note on the broad labels) become
    Neurons on ProSeg (not checked; a likely cause is neuronal transcripts entering small glial
    cells next to neurons). The broad labels stay provisional for ATERA (AT-6).
  - **Cells without label pixels:** 9 ProSeg cells (0.25–4 µm² polygons, covered by later cells
    or smaller than a pixel centre) have no pixel in `proseg_cell_labels`, so the ProSeg MIQ table
    has 91,422 rows; the crop has 1. Found by this run and now reported by the stitch
    (`cells_without_label_pixels`, with a warning); the full-section store was stitched before
    the change and skips a re-stitch (same fingerprint), so its count comes from
    `label_pixels_full.json`. After the final review, MIQ also lists the shapes without a row
    in its table's `uns` and summary (`shapes_without_label_pixels`, with a warning) and adds
    the list to an older table without quantifying again; the full-section store got it that
    way (the 9 ids, checked by `verify_full.json`).

### AT-11 `[feature]`: RCA on Atera (map_first annotation for ATERA rows)

- **Why:** since the RCA merge, `map_first` is the human and mouse default, but its panel, reference and
  gate steps (ANNOTATE_PANEL, ANNOTATE_REFERENCE_PREP, MAP, RESOLVE, COMPUTE_CPU, ANNOTATION_REPORT) are
  built for the targeted MERSCOPE and XENIUM panels and have never run on a whole-transcriptome ATERA
  section. ATERA rows are gated off until this milestone (`main.nf` `annotationModeRow`, the ATERA
  preflight, `ateraPrepareOnlyRow`; `docs/configuration.md#atera-rows`).
- **Scope (to be planned with RCA):** an ATERA panel mode for the 18,049-gene panel (§1.3 fact 6; gene
  set shared with AT-8's "analysis gene set"); reference prep, MAP and RESOLVE resources at that panel
  size; pairing rules for ATERA rows; then lift the gate and the deprecation exception for ATERA
  rows.
- **Depends on:** AT-8 (analysis gene set, lean MapMyCells query).
- **Exit:** a mixed samplesheet stub run (XENIUM `map_first` + ATERA) reaches ANNOTATION_REPORT for both
  rows, then the crop and the full section run `map_first` through MENDER within their resources.
- **Notes from AT-8 for AT-11 (2026-11-07):**
  - **Gene set.** AT-8 built the analysis gene set: `merxen analysis-gene-set`
    (`merxen.analysis_gene_set`, `docs/cli.md`). ATERA defaults to the recipe `all_panel_genes`,
    every one of the 18,049 panel genes (user decision of 2026-11-07: maximum accuracy, compute
    cost accepted); `intersect_public_panel:<key>` and `explicit:<path>` exist but are not the
    default. The set depends only on the panel (both layers of a store, and the crop, give the same
    `gene_set_hash`), so one reference preparation serves every ATERA section of the panel. Its
    `analysis_gene_set.csv` is the `annotation-panel --panel-genes-path` layout, and
    `gene_set_hash` equals the `panel_hash` that command gives it. Wire the command beside
    ANNOTATE_PANEL for ATERA rows. The Atera ∩ Xenium Prime 5K human overlap (4,874 of 5,001) is
    recorded for information only. The user's rule of 2026-11-07 (RCA on Atera is a new rule set:
    no existing RCA rule, such as family trust inheritance, the large-panel marker prefilter,
    marker caps, gates or thresholds, is kept where it would cost accuracy, and each departure is
    shown by a comparison) applies to everything built on this set.
  - **Query rule 1: the MapMyCells query is the full gene set, never trimmed to marker genes.**
    `cell_type_mapper` 1.7.2 turns a raw query into log2 CPM using the row sums of the genes in
    the query file (`cell_by_gene/cell_by_gene.py`), so trimming the query to the bundle's marker
    genes changes every cell's CPM and therefore the mapping. The query columns must equal the
    bundle panel, which is the analysis gene set; RCA's `build_sample_query` and
    `write_query_h5ad` already do this, and AT-8 tests it on an ATERA-shaped fixture
    (`tests/test_analysis_gene_set.py`).
  - **Query rule 2: decide which depth RESOLVE uses.** `SampleQuery.total_counts` counts every
    non-control feature of the sample (`annotation/pipeline.py`, `load_samples`), not the panel's
    genes. On the 300-gene panels the two are the same; on ATERA with `all_panel_genes` they are
    also the same, but with any smaller gene set they differ (the whole transcriptome against the
    set). AT-11 must decide which depth the RESOLVE depth bins use, so that it matches the depth the
    bundle's simulations assume, and record the choice.

### AT-10 optional, on request

- H&E via `REGISTER_IMAGES` (1.5 d).
- Role-based pairing with Atera fixed (10+ d).
- spatialdata-io upgrade and adoption of the upstream reader.

**Developer effort:**
- Milestone 1 (AT-0 to AT-5): about 25.5 d, about **33 d** with 30% contingency.
- AT-6 to AT-8: about 24 d, about 31 d with contingency.

---

## 9. Risks and unknowns

| ID | Risk | L / I | Mitigation | Where |
|---|---|---|---|---|
| R1 | Hidden full passes or materialisation over 1.6e9 rows (contract code, `PointsModel.parse` without `sort`, `as_known()` on rewrite) | H / H | Opt-in fast paths; `sort=True`; pass-budget test; no `deep=True` on real data; hardlinked `latest` | AT-2–AT-4 |
| R2 | Build memory runs past the reservation; Dwight does not cap it, so the OOM killer may hit other sessions (including downloads) | M / H | RSS watchdog; explicit dask/zarr concurrency; `row_range` default; tile semaphore | AT-4 |
| R3 | Partial store reused after a crash or retry, or stale settings silently reused | M / H | Atomic write, completion marker, fingerprint | AT-4 |
| R4 | Mask polygonisation is slow, deadlocks after fork, or drops labels | M / M | Subprocess pool before thread pools; `n_jobs = task.cpus`; no silent serial fallback; `vendor_polygons` fallback; build fails on missing shapes | AT-3, AT-4 |
| R5 | Half-pixel or rounding misregistration (shapes, lookups, pyramid levels) | M / M | +0.5 px; floor lookups; shape-ratio level transforms; centroid acceptance checks | AT-3–AT-5 |
| R6 | Nucleus/cell ID namespace confusion corrupts per-cell metrics silently | M / H | `cell_instance_id` joins; grep test; two-nucleus fixture | AT-6 |
| R7 | Sample-identifying metadata leaks into committed text, logs, plot titles or evidence (the store keeps it on purpose, D13) | M / H | Fake fixture values; slow real-data tests that print no manifest value; `to_alias` for printed fields; privacy hooks on commits and PR text | AT-3, AT-4 |
| R8 | ZipStore lock serialises reads; parallel throughput not better than serial | M / M | Budget assumes serial; benchmark; unzip the transcripts zip (+15.6 GB) if needed | AT-5 |
| R9 | ProSeg infeasible at this scale (proseg#162) | H / M | Native and `cellpose_only` routes do not depend on it. AT-5b confirmed one whole-section run is infeasible (about 300 GB) and 2 mm tiles fit in 22 GB; AT-9 tiles | AT-5b, AT-9 |
| R10 | Format churn across Atera versions; preview bundles in Xenium layout | M / M | Version checks; preview refusal in the XENIUM builder; parity test | AT-3, AT-4 |
| R11 | Variable channel sets and file-order ≠ channel-order | M / M | OME `TiffData` mapping; partial-mismatch error | AT-3, AT-7 |
| R12 | Cache-invalidating edit to existing processes | M / H | Script-text pins + golden JSON (stub runs do not prove hashes); announced QC re-run only | AT-1 onward |
| R13 | Merge conflicts with manual-registration and RCA | H / M | Sequencing in §8 | AT-1, AT-8 |
| R14 | Silent Xenium fall-through (`else` arms) | H / H | Registry raising on unknown platforms; grep test | AT-1 |
| R15 | Neuropil absorption by ProSeg or Cellpose (55% unassigned by the vendor) | M / M | Absorption report vs vendor masks. AT-5b: under the vendor prior ProSeg absorbs 15–18% of the unassigned transcripts, mostly within a few µm of a cell; a weak prior grows cell area by 36% | AT-5b, AT-7 |
| R16 | Disk: worst case about 270 GB per section plus ProSeg intermediates | L / M | Hardlinked `latest`; partial extraction; CSV deletion | AT-6 |
| R17 | Misleading vendor Azimuth labels (pan-human) | M / L | `vendor_` prefix; cross-check only | AT-4 |
| R18 | Partition index stale or misaligned on read | M / L | File-keyed index with fingerprint and full-scan fallback | AT-4 |
| R19 | Transcript z and 3D stack mapping unknown | L / L | 3D off; z passed through | — |
| R20 | External napari viewer needs an ATERA branch | M / L | Track outside the repo | AT-6 |

**Unverified items to resolve first:**
- ZipStore parallel throughput, cold disk. The warm-or-unknown full build decodes 1.41e9 rows in under
  3 minutes, so it no longer limits the build.
- `dd.read_parquet` adaptive splitting at these file sizes.
- Whether stub `-resume` hashes `stub:` text.
- `containerOptions` hashing after the GPU-closure change.
- tifffile cross-file OME fixture writing.
- `z_level` semantics.
- ProSeg memory with 18k genes; pciSeq model arrays; MECR WHB nnz.

*Resolved since revision 3:* `obs.filtered` threshold (< 10 observed); negative coordinates in label
lookups (bounds-checked; use floor); multi-component labels (MultiPolygon); `cell_index` order
(non-decreasing).

*Resolved in AT-5 (§6.1):* BUILD peak RSS and wall time; polygonisation time and vertex counts; `tile_blocks`
cost (§3.3); label-pyramid memory and level shapes (equal to the TIFF pyramid); points parquet size with
spatialdata's writer.

---

## 10. Open questions for the user

**Answered 2026-11-04:**

- Q4: QV ≥ 20 at build.
- Q6: the native vendor-mask route first, then `cellpose_only`, then ProSeg only if AT-5b passes.
- Q13: the `feature/atera-support` integration branch.
- Pairing: Atera runs single-sample for now. The role-based pairing rewrite stays optional (AT-10).

**Decided 2026-11-05:**

- Metadata retention: the store keeps all vendor metadata unredacted (verbatim sidecars, full header in
  attrs); privacy rules cover only committed text and human-facing output (D13).

The remaining questions use their recommended defaults until the user says otherwise.

1. **Platform identity.** (a) New `ATERA` value plus `PlatformSpec`; (b) a Xenium flavour.
   *Recommended: (a).*
2. **Original-segmentation names.** (a) `atera_cell_boundaries` / `atera_nucleus` via the registry;
   (b) reuse `xenium_*`. *Recommended: (a).*
3. **Reader.** (a) MerXen-native, porting PR426 with attribution; (b) pin the PR426 fork; (c) wait for a
   release. *Recommended: (a)*, plus a slow parity test once upstream ships.
4. **Transcript QV policy at build.** (a) QV ≥ 20 (1.41e9 rows; reproduces the vendor matrix; pre-filter QV
   maps kept in attrs); (b) keep all (1.64e9); (c) also drop controls and unassigned (about 0.64e9).
   *Recommended: (a).*
5. **Shape source.** (a) Polygonise the vendor masks (+0.5 px, 0.25 px simplification); (b) vendor polygons
   (+0.5 px). *Recommended: (a)*, with (b) as a fallback.
6. **First segmentation route.** (a) `native`, then `cellpose_only`, then ProSeg only if AT-5b passes;
   (b) Cellpose + ProSeg now; (c) ProSeg with the vendor prior only. *Recommended: (a).* Is a multi-day or
   tiled ProSeg acceptable at all?
7. **Spatial order of points.** (a) `row_range` (gene-sorted, tile-pruned); (b) `tile_blocks` with a
   decode cap. *Recommended: (a)* until AT-5 measures (b).
8. **Transcript IDs.** (a) Vendor global ordinal + 1 (gaps after the filter); (b) contiguous 1..N (extra
   pre-pass). *Recommended: (a).*
9. **Table controls and vendor extras.** (a) Gene-only X, per-type control counts in obs, all vendor
   columns `vendor_*`, `de_*` dropped, diffexp CSVs as sidecars; (b) everything as shipped; (c) drop vendor
   annotations. *Recommended: (a).*
10. **Data location.** (a) New `MERXEN_ATERA_RAW_ROOT`; (b) under `MERXEN_PUBLIC_VALIDATION_ROOT`.
    *Recommended: (a).* May the staged ROI directory be created next to the downloads once they finish?
11. **3D DAPI stack.** (a) Not ingested; (b) opt-in. *Recommended: (a).*
12. **H&E.** (a) `REGISTER_IMAGES` later; (b) a build-time view-only element; (c) both.
    *Recommended: (a).*
13. **Branching.** (a) Integration branch with a CLAUDE.md exception plus neutral PRs to `main`;
    (b) short-lived PRs to `main` gated by preflight. *Recommended: (a).*
14. **Unused `io/samplesheet.py`.** (a) Keep in sync; (b) delete in a `[chore]` PR.
    *Recommended: (b).*
15. **Alias family for in-house Atera.** (a) A new family (e.g. `HA<n>` with ROI suffixes, one alias per
    ROI); (b) reuse HD/HM. *Recommended: (a)*; you choose the prefix.
16. **Atera preview bundles (Xenium-v4 layout).** (a) Refuse in the XENIUM builder; (b) route to ATERA;
    (c) allow as Xenium. *Recommended: (a)* for now.
17. **One-time QC re-run.** Is a single re-run of existing XENIUM/MERSCOPE QC tasks acceptable when AT-6
    edits `qc.nf`? (a) Yes, announced; (b) no, keep `qc.nf` frozen and add a new QC process for ATERA.
    *Recommended: (a).*
18. **Mixed samplesheets.** An ATERA row whose `stop_stage` is too late: (a) fail the run with a clear
    message; (b) clamp to `build_spatialdata` with a warning. *Recommended: (a)* until AT-6.
19. **Cache path and rebuild policy.** (a) Allow `atera_spatialdata_path` as a persistent target, guarded by
    the completion marker; refuse on fingerprint mismatch unless `--force_spatialdata_build`;
    (b) auto-rebuild on mismatch; (c) published-output reuse only. *Recommended: (a).*
20. **Compute.** (a) BUILD 128 GB reservation at 16 CPUs with a watchdog cap and one retry; heavy ATERA
    downstream tasks up to about 250 GB with `maxForks=1`; (b) existing limits plus subsampling.
    *Recommended: (a)*, tightened from traces.
21. **Network in tests.** (a) Synthetic only in CI; `tiny_atera_dataset` only in the slow tier; (b) allow
    CI to download it. *Recommended: (a).*
22. **Gating rule for WTA guards.** (a) Build-stage guards key on `platform == ATERA`; downstream
    whole-transcriptome guards key on gene count (> 5,000), so Xenium Prime 5K and later large panels also
    benefit; (b) platform only. *Recommended: (a).*
23. **Stages off by default for ATERA until adapted:** transcript spatial patterns, MECR, the ProSeg hybrid,
    pciSeq and GASTON (already opt-in), cortical depth and DFO for non-cortical tissue.
    *Recommended: yes, all.* Should any be prioritised?

---

## Appendix A. Review log (revision 3 → 4)

Each critique was checked against the repo at `c5c1659`, the installed environment, or the review
scripts' full-section results.

**Fidelity review: all accepted.**

| Point | Disposition |
|---|---|
| Nucleus/cell ID collision in the clustering join | Accepted; `clustering_squidpy.py:373-376, 3899-3912, 4005-4020` confirmed (§3.6, §5) |
| Half-pixel offset of mask shapes | Accepted; `find_contours(level=0.5)` and the `enrich.py:240-266` scaling confirmed (§3.5) |
| Acceptance criterion 4 circular or loose | Accepted (§1.1) |
| `nucleus_area` = largest nucleus | Accepted (§2.3, §3.7) |
| `rint` vs floor | Accepted; `cellpose.py:896-897` confirmed. Plan adds `pixel_convention` with an unchanged default, to keep Xenium outputs and hashes stable |
| `TiffData` channel mapping | Accepted |
| `instance_id` column requirement | Accepted; `spatialdata_schema.py:200-236` falls back to `source_cell_id` and raises on duplicates |
| `searchsorted` ordering | Accepted |
| Tile-order offsets | Accepted |
| Missing §2.6 conflicts | Accepted |
| Over-stated evidence | Accepted |
| `homogeneous_transform` tolerance | Accepted |
| Pyramid level transforms | Accepted, with the spatialdata rounding behaviour marked UNVERIFIED |
| ENSG-shaped symbols | Accepted |
| Category column by name | Accepted |

**Scale review.**

| Point | Disposition |
|---|---|
| Dwight does not enforce memory | Accepted; `process.executor = "local"` with only `executor { memory }` confirmed |
| `tile_blocks` memory claim had no mechanism | Accepted. Default switched to `row_range`. `DASK_NUM_WORKERS` is read by dask as `num_workers`: verified in the env, 16 → 16 |
| `PointsModel.parse` monotonic check | Accepted; `models.py:826-833` confirmed |
| Polygonisation pool size and fork | Accepted; `os.cpu_count()` default and silent serial fallback confirmed |
| Mask contour vertex counts | Accepted (simplification) |
| Hardlinked `latest` corrupted by in-place write | Accepted; `Path.write_text` at `enrich.py:669` confirmed |
| Partition index lost on rewrite | Accepted; `spatialdata_io.py:1630-1634` confirmed. The adaptive-split claim is kept as UNVERIFIED |
| ProSeg / `convert_to_latest` passes | Accepted |
| Deep check under `tile_blocks` | Accepted |
| Size figures | Accepted (snappy, 38 B/row) |
| `.npy` disk | Accepted |
| Parallel-read speedup | Accepted |
| Label chunking | Accepted, to measure in AT-5 |
| Tiny partitions | Accepted |
| **Added during review** | Parquet round trip returns *unknown* categories (verified, dask 2026.8.0), and spatialdata's points writer then runs `as_known()`; added to R1 and §3.3 |

**Docs-conformance review: all accepted.**

| Point | Disposition |
|---|---|
| Privacy of copied metadata | Accepted; `experiment.spatial`, the metrics header and `panel_config.customer` fields confirmed (D13) |
| Half-pixel offset; floor | Accepted (duplicates of the fidelity points) |
| Nucleus namespace and area semantics | Accepted. The nucleus label rule is re-tagged empirical: the docs state it for cell masks only |
| `is_sketch` absent | Accepted |
| Per-type control counts and deprecated in `total_counts` | Accepted; criterion 3 extended to all 24,677 columns |
| `binned_transcripts` inconsistency | Accepted; the recommendation is dropped and CSC is marked derivable |
| §2.6 rows | Accepted, including vendor clustering, `var.filtered` polarity, Azimuth labels and FOV size |
| Mixed QV populations | Accepted (`*_all_qv` / `*_q20`) |
| Gene from `gene_offset` | Accepted |
| Glob vs `TiffData` | Accepted |
| Multi-kit | Accepted |
| `gene_coverage` | Accepted |
| QV spatial QC | Accepted |
| `overlaps_nucleus` | Accepted |
| Multi-ROI | Accepted |
| `vendor_z_level` | Accepted |
| Line numbers | The review cites `OFFICIAL_10X_DOCS.md` lines (e.g. DOCS:227) that do not match the 243-line file in the inspection directory. Content was checked by section instead, and every cited statement was found there |

**Completeness review.**

| Point | Disposition |
|---|---|
| Partial-store reuse and settings changes ignored | Accepted; `pipeline.py:56-63` vs `:87-89` confirmed (D14) |
| Concurrency unbounded | Accepted in substance. **Rejected in part:** "nothing on the ATERA path would read `DASK_NUM_WORKERS`" is wrong, because dask maps the env var to its `num_workers` config (verified). `CPU_COUNT == 80` does not govern the threaded scheduler when `num_workers` is set. The plan still sets concurrency explicitly |
| `PointsModel.parse` | Accepted; `sort=True`, as MERSCOPE (`merscope.py:1108`) |
| Partition index | Accepted |
| `_set_known_feature_categories` early return changes MERSCOPE | Accepted; MERSCOPE points arrive known and are re-sorted today. The skip is gated on the opt-in flag only |
| Cache-invariant strategy | Accepted. Only `pciseq.nf` has `stub:` blocks today. Stub hashing is marked UNVERIFIED and the gate is now script pins plus golden JSON. QC hash change deferred to AT-6 and announced |
| Cellpose param names | Accepted; the suffixes are `cellpose_cellprob` etc. plus `mask_max_eccentricity` (`CellposeSettings.groovy:137-150`). Undeclared params read as `null`, so this is not a hard failure, but they are declared in AT-4 anyway |
| Criterion 5 contradiction | Accepted |
| Polygonisation details | Accepted |
| Privacy gaps | Accepted |
| Preflight placement and run-wide abort | Accepted; `:1437-1451` is gated on `run_align`; `runPreflightChecks` is at `:2140`; `stop_stage` default confirmed |
| Preview bundles | Accepted |
| `transcript_id` order | Accepted |
| Per-cell control counts | Accepted |
| Docs and licence notices | Accepted; `THIRD_PARTY_NOTICES.md` and `licenses/` exist |
| Harness classpath | Accepted |
| Literal count | Accepted: 21 including `:1152-1153` |
| Spec override | Accepted |
| Species check | Accepted |
| Staging packaging | Accepted |
| Vendor annotation gaps | Accepted, corrected by the data: only the 50 filtered cells are NaN |
| ENSG-shaped symbols | Accepted |
| Bind paths | Accepted |
| Missing open questions | Added as Q15–Q22 (the cache-path and rebuild questions merged into Q19) |

**User decisions after revision 4.**

| Date | Decision | Effect |
|---|---|---|
| 2026-11-05 | Metadata retention: the metadata stored with each run must be as complete as possible; only commits to the repo (and human-facing output) must never show sample names | D13 rewritten (§3.11). The AT-3 sanitisers (`sanitize_experiment_spatial`, `sanitize_panel_config`, `sanitize_metrics_summary`) and their absence tests are removed; `copy_atera_sidecars` and `atera_metadata_attrs` keep everything, with original file names recorded. §1.1 item 6, §2.5, §4.1, §4.2, §7, R7 and Q-list updated |

**AT-4 implementation notes (2026-11-05).** Where the code departs from the text above, and why:

| Plan text | Implementation | Reason |
|---|---|---|
| §4.2 `RssWatchdog`: "exit 137 above the limit"; §6.1 "watchdog cap at 120 GB, one retry at ×2" | The watchdog exits **99**; the task retries only exit 137/140 (external kills), with twice the memory reservation | The build JSON is part of the task hash and cannot carry `task.attempt`, so a retry would run with the same cap and stop at the same point. An external kill is retried; a watchdog stop asks the user to raise `atera_build_max_rss_gb` and `atera_build_memory_gb` |
| §4.4 params | Adds `atera_build_cpus` (16) and `atera_build_max_rss_gb` (120) beside `atera_build_memory_gb` | Tests and small hosts can run the real BUILD task; the cap is explicit |
| §3.10 preview refusal: "more than 5,000 gene features" | `analysis_sw_version` starting `atera-`, or more than **10,000** panel targets in `experiment.xenium` | Xenium Prime 5K panels have about 5,000 genes plus add-ons and must keep building as XENIUM (Q22 counts them as large Xenium panels) |
| §4.4 `normalizeAnalysisMode`: update the error text | `atera` and `a` are accepted; the error text is unchanged | The text is in the `modes` golden; the goldens stay byte-identical (new ATERA cases only) |
| §4.3 `samplesheet.py`: delete (Q14) or add `atera_*` fields | `atera` mode and `atera_dir` / `atera_spatialdata_path` / `atera_spec_path` added | Deleting the module is Q14's separate `[chore]` PR; the mode-parity test needs all three maps to agree now |
| §4.3 `AteraBuildConfig.morphology_3d` | Not added; `morphology_levels` (default 5) added | 3D ingestion is out of milestone 1, so the flag would only raise; the level count lets the fixture build stay small |
| §4.3 `find_um_per_px` consumers | The enrichment and sanity-plot Xenium parsers read spec files through `spec_file_pixel_size` / `find_um_per_px`, with unchanged results; the segmentation parser (reads `pixel_size` only) is unchanged | Its behaviour differs from the others (pinned in `tests/test_xenium_pixel_size_parsers.py`); merging it and adding the ATERA arms of all three is AT-6 |
| §4.4 BUILD GPU closures "require `platform == 'MERSCOPE'`" | They require `platform != 'ATERA'` | Keeps the XENIUM queue exactly as it was |
| §4.2 build sequence step 8: "aggregates computed in one `dask.compute`" | The aggregates are recorded by the partition reads of spatialdata's points write (one read per partition) and summarised after it | Same single pass, without a second graph |
| §3.12 "refuse unless `--force-rerun`" | Also: a store without the marker is rebuilt (not refused); with no raw folder a complete store is reused without a fingerprint and an incomplete one fails | A half-written store is never reused, and a store given alone (no `atera_dir`) cannot be fingerprinted |

AT-4 also moves the tests that used `ATERA` as an example of an unregistered platform to `COSMX`.

**AT-4 review (2026-11-05).** Three reviews (fidelity, scale, tests) of the AT-4 branch. Fixed:

| Finding | Change |
|---|---|
| Vendor metadata inside the zarr zips and the OME-TIFF headers never reached the store | `write_atera_embedded_metadata` keeps every zarr metadata member and every OME-XML header verbatim; the root attrs and map annotations are parsed into attrs (D13) |
| `nucleus_centroid_x/y`, the `de_*` statistics and the category colours were dropped | Kept as `vendor_*` obs columns, `varm["vendor_de"]` and `uns["vendor_<column>_colors"]` |
| A killed build left the dot-prefixed mask scratch folder (about 21 GB) | Scratch is `<target>.partial-<uuid>.work`; the next build removes it with the partial store; the watchdog removes both before it exits |
| Labels read in square chunks decoded each vendor chunk about twice | Reads follow the vendor chunk grid and are rechunked once |
| `masks/0` was decoded a third time for the total nucleus area | Counted while it is exported for polygonisation (`MaskPixelCounter`) |
| A failed watchdog reading ended the watchdog thread | The thread logs once and keeps polling |
| The pipeline resolved the bundle twice | The resolved bundle is passed to the builder |
| Metric text such as `00123` became a number | Kept as text |
| Reader errors printed input paths; the stale-build log printed a folder name | Path-free messages; the log counts the folders |
| The spec override was not marked and lost the pixel-size lookup to the original | `experiment_source` in attrs; `atera/experiment_override.spatial` first in `spec_sidecar_names` |
| The public dataset's name was written in tests and here | The slow tests take the one extracted bundle under `$MERXEN_ATERA_RAW_ROOT` |
| Untested paths (fingerprint inputs, pass budget, watchdog contract, guards through the build, points-index order, XENIUM preview wiring, `keep_vendor_annotations`, spec override, empty-stage preflight, input paths out of the fingerprint) | Tests added; a slow test checks one tile's interior cells against the vendor values |

Not changed: `emit_codeword_index` stays off (it is per-row data, not metadata; about 6.5 GB uncompressed
on the public dataset and in every downstream points load; AT-5 measures it); `tile_blocks` stays
opt-in with its cost documented (AT-5 decides); the zarr event-loop ceiling and the remaining repeated
zip opens are seconds per build (AT-5 measures); the watchdog cap is the same on the retry on purpose
(the retry is for external memory pressure); the ATERA arms of the enrichment and viewer-cache lookups
move to AT-6 (§1.1 item 5).

**AT-5 pilot (2026-11-05).** Crop pilot, full public build and milestone-1 acceptance (§1.1 status).

| Finding | Change or decision |
|---|---|
| The crop's label check peaked at 6.7 GB | The acceptance label statistics are counted in row strips; the crop run peaks at 4.2 GB and the full run spends 37 s on both masks |
| A `tile_blocks` build that split a tile failed while writing the points index (infinite block edges with `allow_nan=False`) | Open edges are written as `null`; a unit test and an end-to-end `tile_blocks` build test cover it |
| Peak memory is 11.7 GB anonymous, not the 40–80 GB estimate; the build takes 8.5 minutes, not 1.5–3 h | `atera_build_memory_gb` 128 → 64 and `atera_build_max_rss_gb` 120 → 48 (§6.1). Only the ATERA build JSON goldens change |
| `row_range` or `tile_blocks` | `row_range` stays the default (§3.3) |
| Points take 17.1 B/row against the 11.7–12.1 B/row probe; the dask index column `__null_dask_index__` is about a quarter of the bytes | Not changed here: dropping it means writing parquet outside spatialdata's writer, the later optimisation §3.3 already lists |
| The trace's `peak_rss` (103.6 GB) far exceeds the watchdog's peak (11.7 GB) | Expected: the trace counts the shared memory-mapped masks once per worker. The watchdog's anonymous measure is the one to size against (§6.1) |

Not measured in AT-5: `emit_codeword_index` (left off), and the zarr event-loop ceiling and repeated zip
opens on their own. The whole build is 8.5 minutes, so neither is worth separate work now.

**AT-5 review (2026-11-05).** Two reviews of the AT-5 branch re-checked the full store with their own
code (raw zips, `zipfile` and `tifffile`, not the Atera reader) and found no failing criterion. Their
findings and what was done:

| Finding | Change or decision |
|---|---|
| The acceptance script did not check the vendor table data of §1.1 item 6 (nucleus centroids, `de_*` statistics, colours) | New `table_vs_vendor` criterion, read straight from the zips: `instance_id` = vendor row + 1, X against the vendor gene columns, every vendor `obs` column, the `cell_summary` values, the vendor PCA/UMAP, `varm["vendor_de"]` (270 columns) and the colours |
| Labels were matched to rows by assumption, never compared with the vendor masks | Every label pixel is compared with `masks/1` and `masks/0` (0 differing pixels on the full section, 31 s more) |
| The build options came from the store's own header | New `build_options` criterion: `min_qv` 20, 0.25 px and no allow-list, unless overridden on the command line |
| `--expected-vendor-excess` was optional | Required for `points_vs_matrix` unless `--allow-unpinned-excess` |
| One shape per cell was checked by unique positions only; leftover `.partial-*` folders were reported, not failed | Shape count must equal the cell count, shapes are indexed by their label, nucleus shapes by nucleus row + 1 with the right owner; a leftover partial fails `build_marker` |
| The metadata check reused the reader's helpers for zip members, metrics and panel header | Members are listed from the zip name list, metrics are compared with the CSV text and the panel header is rebuilt in the script |
| The module docstring claimed bounded memory | It now says the matrix and table checks grow with the nonzeros (about 21 GB) |
| Missing negative tests | Tests for a per-tile row mismatch, an off-tolerance shape centroid, a duplicate shape, other build options, an unpinned excess, a leftover partial, table edits, a metrics edit and a pixel swap that keeps every area |
| §7.4 item 1 still said "exact, 638,849,927" | States the 84-count deprecated-codeword excess |
| "4× headroom on both" | The cap is about 4× the anonymous peak; the reservation is about 2× the physical footprint |
| The new 64 / 48 GB defaults had not run on the full section; no code revision in the evidence | The full section was rebuilt at the reviewed revision with the defaults (§6.1) and both stores pass the new acceptance; the report records the git revision; the task's `.command.run` / `.command.sh` and the scratch scripts are copied into the evidence |
| The points index box order was undocumented | Documented as `[xmin, xmax, ymin, ymax]` µm in the builder and `docs/stages/spatialdata-build.md`; `block_bounds_um` edges can be `null` |
| The plan said AT-5b had not run | §8 AT-5 and AT-9 record the benchmark and the decision |
| Commit messages carry `Claude-Session` lines | Not rewritten here: rewriting both stacked branches is the maintainer's call before any push |

**AT-5b ProSeg benchmark (2026-11-05).** Experiment only; no code change. The numbers and the AT-9
decision are in §8 (AT-5, AT-9); R9 and R15 are updated. The crop is a sparse corner, so the densest
2 mm tile of the full section was run as well and the extrapolation is calibrated on it.

**AT-6 exit runs (2026-11-06).** Crop, then full public section, in native mode from the AT-5 source
stores through MENDER (§1.4, §6.2). Findings and changes:

| Finding | Change or decision |
|---|---|
| An explicit `stop_stage=mender` on an ATERA row selected the standalone `mapmycells` stage between clustering and MENDER, so preflight refused the very stop its own message suggests | ATERA rows pass over `mapmycells` unless the row names it as its start, stop or only stage (which preflight still refuses). Harness cases for `stop_stage=mender`, `only_stage=mapmycells` and `start_stage=mapmycells` |
| QC summed the float32 cell table in float32: the crop's assigned total was 18,894,688, one above the vendor matrix | Stores with build aggregates (ATERA) sum per cell in float64 and total in int64; XENIUM and MERSCOPE keep their historical sums. Full section: 638,693,512, exact. Since the user decision of 2026-11-06 every platform takes the exact sum |
| `MENDER_COMPUTE` failed on the full section in `sc.pp.neighbors`: the MENDER env, created fresh during the run, resolved `pynndescent` 0.7.0, whose parallel subtree builder numba 0.55.2 cannot compile. Under 4,096 cells (the crop) Scanpy computes exact neighbours and never reaches that path; MENDER runs of 100–164k cells completed in September, before this env was re-created | `envs/environment.mender.yml` pins `pynndescent==0.5.13`, `numba==0.55.2`, `llvmlite==0.38.1` (tested). This changes the MENDER_COMPUTE env and so its task hash for every platform, once |
| ENRICH's all-feature counts fall short of the vendor gene + control obs columns by 3 (crop) and 84 (full) counts | Not a MerXen difference: every short count is in a `DeprecatedCodeword` column with no transcript, the vendor matrix quirk recorded in AT-5 |
| A `-resume` re-runs ENRICH and every stage that takes the latest store as input: ENRICH rewrites the store's root `zarr.json`, which changes the directory's modification time that Nextflow hashes | Not changed in AT-6: the reruns are skips (ENRICH 1.5 min on the full section) and the analysis stages downstream of VISUALIZE stay cached. Recorded for the AT-7 restart work |
| The first hardlink run made the source stores' linked payloads read-only (143 crop files, 1,543 full-section files) | By design (`readonly_payloads`); bytes, inodes and modification times are unchanged. Writing to those files now fails instead of corrupting the source |
| MIQ on the full section was unmeasured (WP-X) | 3 min 9 s, 9.6 GB peak, against a 160 GB reservation |

**AT-9 implementation and crop run (2026-11-06).** The ProSeg route for ATERA
(`segmentation_source=native_proseg`; §8 AT-9). Findings and changes:

| Finding | Change or decision |
|---|---|
| ProSeg 3.2.0 refuses an `--output-path` that does not exist (found on the first real crop tile) | The tile task creates it; the fake ProSeg of the tests now refuses a missing one too |
| With two ATERA sources, ENRICH's native-store fallback (`segmentation_sources == ("native",)`) turned false, and per-shape assignment ran its Cellpose path (spatial joins of every shape, including the 1.4 B-row points against `MOSAIK_proseg`) for any source other than `native` | The fallback keys on the default source; per-shape assignment treats `native_proseg` as native. Tests for both |
| A failed ENRICH's partial cleanup would delete `table_MOSAIK_proseg`, which ENRICH cannot rebuild | Tables of registry branches with a `prior` are never derived tables |
| ATERA refused to draw any mask that is not a vendor label element | A registry branch with a `label` element is backed by it (`MOSAIK_proseg` -> `proseg_cell_labels`) |
| `sanity_plots._crop_points` and `qc.gene_comparison` use a points `background` column whenever one exists, whatever the assignment column | Not reached on ATERA (overlays use the rasterised path with the registry column; `assigned_only` has no caller); a grep test lists every generic `assignment` pick with its reason |
| Changed test text | The `native` golden changes the error text of two cases (the list of sources): the ATERA-only `atera_cellpose_only_error` and the platform-independent `unknown_error`, so an unknown `segmentation_source` on a XENIUM or MERSCOPE row now also lists `native_proseg` (message only; no task input or hash changes; corrected by the final review, which found the second case is not ATERA-only; for the user's OK). The ENSURE_PROSEG trigger pin of `test_analysis_segmentation_branches.py` follows the new trigger (same value). No process script or pin changed |
| The Xenium parquet layout turns on ProSeg's nuclear compartment model, untested at the full schedule | Crop and dense tile agree with the CSV benchmark (above); kept on, `atera_proseg_nuclear_compartment=false` is the fallback |
| `-resume` | No tile re-runs (their inputs are values); the stitch task re-runs and skips by fingerprint; ENRICH and its layer check skip |

**AT-9 reviews (2026-11-06).** A code and data review and a scale and mutation review of the AT-9
branch. Both rebuilt the crop stitch independently and found no defect in the stitching, the pixel
convention, the parquet layout, the filtering or the layer registration. Findings and changes:

| Finding | Change or decision |
|---|---|
| A re-stitch (another prior weight or schedule) rewrote the ProSeg layer in place but kept the tables computed from the old cells; MIQ's skip check (table, sidecars, image keys, bead removal) would keep intensities of the old polygons under the same vendor ids | A stitch that writes deletes every table of region `MOSAIK_proseg` other than `table_MOSAIK_proseg` (image quantification, clustering) and reports them; test with a changed tile manifest (also covers the stitch skip ignoring the fingerprint) |
| One failed tile dropped the row after `segment` on both layers, with no message naming the tile (`failOnIgnore` exits non-zero) | The log names the failed tiles of each row and says `-resume` re-runs only those. A tile task is retried once unless it exits 99 (watchdog) or 3 (new: a refusal a retry would repeat); a failed ProSeg process exits 1 and is retried. Routing the row's native layer on alone was rejected: the row asks for both layers, and finishing it with one would be a quieter partial result than a named failure that `-resume` repairs cheaply. Wiring test with a failing tile (exit 3: one attempt; exit 1: two) |
| The printed prior weight is lower than requested by the share of rows ProSeg's cell-free filter drops (crop 0.31%; at most 0.68% per tile on the full section from a 2% sample); a section with large cell-free areas could pass 2% and fail its tiles | A tile whose printed weight is off restarts once with the multiplier rescaled by requested / printed (ProSeg's factor is fixed by the data), and fails only if it is still off; restarts are in the manifest |
| `atera_proseg_allow_untested_version=true` passed the preflight, but every tile refused any version but 3.2.0 | The flag reaches the tile config (only when true, so default task hashes are unchanged) and the tile runs that version with a warning |
| The stitch report summed only the vendor labels without a ProSeg cell; owned cells without a polygon reached only the tile manifests | Both are listed and counted in the report, with a warning when either is non-zero (their transcripts become background) |
| `seam_overlap_pixels` mostly counted overlaps of two cells of one tile (crop: 567 of 568 pairs) | Split into `overlap_pixels_within_tile` and `overlap_pixels_across_tiles` |
| A tile trusted the plan's source build fingerprint | The tile compares it with the source store's build marker before exporting |
| ProSeg writes `membership_debug.txt` and `rebalanced_membership_debug.txt` to its working directory (the repo root when tests ran from there) | ProSeg runs in its output folder (`work/proseg`); the command's paths are absolute |
| The stitch's label raster (10.4 GB on the full section) was removed only on success | Removed whether or not the stitch succeeds |
| The `export_tile` docstring said a row below `min_qv` raises; the code warns and drops it | Docstring fixed; a test now covers the QV filter |
| Surviving mutants: claims naming points files the store lacks, two tiles owning one label, two ProSeg cells of one label, the export QV filter, the float-rounding correction of the core index, the prior-weight multiplier (killed only by the real-ProSeg test) | One test each (the multiplier on a synthetic store with more than 100 transcripts per cell); all seven mutants are now killed |
| The `nextflow.config` comment gave the dense tile's peak as 21.7 GB | 24.2 GB (16 threads) and 25 GB (32 threads) against the 32 GB reservation; the watchdog cap (40 GB) sits above the reservation, so an executor that enforces memory kills first and the retry gets 64 GB |
| A failed tile keeps its inputs in the work folder (1–2 GB of parquet, up to 430 MB of prior) | Kept: they are what a failure is diagnosed from; finished tiles drop them unless `atera_proseg_keep_tile_inputs` |
| A halo tile's claim on a transcript beats the core tile's background; the core rule settles only two claims | As designed: in effect the tile that owns the cell decides, which is the requirement "transcripts assigned in halos resolved by the owning tile". For the user to confirm, together with the seam polygons of cells that lose a conflict (AT-9 final review) |
| The effective prior weight is 140 on every tile; ProSeg's automatic weight follows local density (about 113 and 152 on the two crop tiles) | Open question for the user with Q1–Q8 of the design. Crop and dense tile agree with the benchmark; sparse tiles get a stronger prior than the benchmark's automatic one |
| A `-resume` re-runs 8 tasks (the stitch, ENRICH, VIEWER_CACHE, MIQ, both layer validations, both QC); QC recomputes in full | Recorded with AT-6's ENRICH `-resume` finding for the AT-7 restart work; the full-section stitch (30 tiles, a 42,548 x 61,369 raster) has not run yet |
| Changed test text missing from the implementation entry above | The golden harness span signatures gained `rowSource = null`; the process-pin count went from 44 to 47 (three new pins, none changed); the rule that a default keeping no layer resolves to every layer also applies to `cellpose_only`, which no platform allows today. All need the user's OK |

**AT-9 full-section run (2026-11-06).** `ATERA_PUB` through the ProSeg route and both layers to
MENDER (§6.2, §8 AT-9; evidence `$MERXEN_EVIDENCE_ROOT/atera_at9/`: `run_full.sh`, trace, report,
timeline, `verify_full.json`, `tiles_full.tsv`, `layer_compare_full.json`,
`cluster_compare_full.json`, `dense_schedule_compare.json`, `disk_full.json`). Findings and
changes:

| Finding | Change or decision |
|---|---|
| The run passes every check on the first attempt: 30 tiles, no retry, no prior-weight restart, every vendor cell once, 806 empty cells reported, table counts equal to the points assignment, source store unchanged | None |
| 9 ProSeg cells (0.01%) have no pixel in `proseg_cell_labels` (tiny or covered polygons) and are silently absent from the ProSeg MIQ table; their control counts are zero | The rasteriser counts each cell's final pixels; the stitch report lists them (`cells_without_label_pixels`) and warns. Tests on covered, sub-pixel and off-grid polygons. Mask-based stages still leave them out (they have no mask) |
| Short schedule against 200 + 600 on the densest tile | Short default kept (§8 AT-9: the long chain differs by about 1.3 points of agreement, less than the chain noise between short runs, at 3.4× the ProSeg time). The longer chain widens the area distribution (median ratio 0.81 against 0.84–0.85), as on the crop in AT-5(b) |
| Tile phase 7 h 26 min on a host at load up to 345, with 4 tiles at once and the schedule check beside them; the largest pending tile started last | Concurrency default kept (4). Largest-first ordering is a possible improvement; Nextflow's release order under `maxForks` is not the emission order |
| The ProSeg route adds 29.5 GB to the latest store (25.4 GB of rewritten points) against 2.6 GB for the native route | As designed (the points carry the second assignment column); recorded in §6.2 |
| ProSeg cells carry fewer counts (median 4,323 against 4,812); clustering QC drops 2,471 ProSeg cells against 50 vendor cells | Recorded; follows from the 25.9% release and the 806 empty cells. No threshold change |

**AT-9 final review (2026-11-06).** An independent check of the full-section outputs against
the report (source store, trace, tiles, table, shapes, points, labels, downstream, clustering
numbers, AT-1 guards) found no blocking problem. Findings and changes:

| Finding | Change or decision |
|---|---|
| The label-pixel fix does not reach the existing store (the stitch format version and fingerprint are unchanged), and MIQ of the ProSeg layer left out the 9 cells without a label pixel without saying so | MIQ of an extra layer lists the shapes it has no row for in its table's `uns` and summary (`shapes_without_label_pixels`, `n_shapes_without_label_pixels`) and warns; a table without the field gets it on the next run without quantifying again. Tests with a blanked label and an older table. The full-section store was updated that way; the docs say stores stitched before the report field lack it there |
| A cell that loses a seam conflict keeps its full polygon: 174 cross-tile polygon pairs overlap by more than 1 µm² (48 by more than 10 µm², at most 91.5 µm²); the label raster settles their pixels by ascending id, not by the transcripts' core rule, so MIQ and control counts can give a pixel to another cell than its transcripts | Added to the pending "halo tile claim vs owning tile" decision (AT-9 reviews). Options: clip the losing polygon, or rasterise cross-tile overlap pixels by the core rule. Scale: the 160,659 conflicts are 0.028% of the assigned rows (1,592 cells). Documented in the segmentation stage doc |
| `verify_full.json` never checked QC (it looked in `atera/qc`), but §8 cites it for QC outputs of both layers | The check looks in `atera/<layer>/qc/qc_out` (5 files per layer) and also checks the MIQ field above; re-run (`verify_full.json`, previous result kept as `verify_full_v2.json`) |
| The evidence `manifests/full_tile_plan.json` was the dense-tile check's plan (`tile_subset: ["c2_r1"]`) | Renamed `dense_tile_plan.json`; `full_tile_plan.json` is now the full run's published plan |
| The `native` golden's `unknown_error` case is not ATERA-only, so XENIUM/MERSCOPE rows see the new error text | Docstring and the AT-9 entry above corrected; listed for the user (message only, no hash change) |
| The tiles (2.7 GB) and the stitch report were published as links into the work directory; cleaning it would lose what a re-stitch needs | Tiles and the report are published by copy (publishDir only: no script pin or task hash changes); the store link and placeholders stay links. The full run's published tiles and report were materialised the same way |
| Tiles do not start largest-first (`c2_r2` started last and set the end of the tile phase) | Already recorded (AT-9 full-section run); not a defect |

**AT-6 reviews (2026-11-06).** An end-to-end data review and a code review of the AT-6 branch.
Findings and changes:

| Finding | Change or decision |
|---|---|
| A `[docs]` commit reworded a comment in `envs/environment.mender.yml` after the exit runs; Nextflow names a conda env after the file's full text, so HEAD would build an untested env and recompute MENDER_COMPUTE once more | The file is back to the exact text the exit runs used (their env). A new guard pins the SHA-256 of every `envs/*.yml` (`test_env_file_pins.py`), so an env edit is a reviewed decision; env edits stay out of `[docs]` commits |
| The MENDER env pin changes the MENDER_COMPUTE hash of every platform | Needs the user's OK as a second announced re-run. Without it any freshly built MENDER env fails above 4,096 cells on every platform |
| The broad atlas labels misname the human hippocampus: 27,805 mature oligodendrocytes (MOBP, PLP1, MOG high, PDGFRA 0.2 per 10k) are "Oligodendrocyte precursor", 8,345 PROX1/CALB1/SLC17A7-positive excitatory neurons are "Upper rhombic lip", 3,512 SLC17A7-high cells are "Cerebellar inhibitory"; the names carry into `broad_class`, the subcluster names and the MENDER cell states | Documented as provisional for ATERA (clustering doc). Follow-up for the robust annotation work: restrict the WHB superclusters by region (no cerebellar classes for hippocampus), add a marker sanity gate (an OPC label needs PDGFRA), and check whether human XENIUM rows show the same pattern |
| An ATERA row whose `only_stage`/`stop_stage` names `pciseq`, `gaston` or `mecr` was told to pass `--<stage>_enabled true`, which leads into the ATERA refusal | ATERA rows are told which stages they run instead |
| Global switches of off stages were handled two ways: a global `pciseq_enabled`, `gaston_enabled`, `cortical_depth_enabled` or `distance_from_object_enabled` aborted every run with an ATERA row (mixed samplesheets too), while a global `mecr_enabled` or transcript-pattern switch was ignored with a note | One policy: a global switch is ignored on ATERA rows with a setting note (so mixed samplesheets keep it for XENIUM/MERSCOPE rows); a value set on the ATERA row is refused, and the message names the row setting to remove |
| QC without a table key raised on every platform when the dask aggregation failed, where XENIUM/MERSCOPE used to warn and retry in pandas | Only stores with build aggregates (ATERA) raise; the others keep the warning and the fallback |
| The float32 regression test did not show that its fixture drifts on the legacy sum | It now asserts the drift (240,333,536 against 240,333,534) |
| XENIUM and MERSCOPE keep the float32 QC assigned total, which is inexact past 2^24 counts (a real Xenium section exceeds it) | Unchanged (byte-identity). Candidate accepted change for the user to decide with the QC re-run. Accepted 2026-11-06: every platform now takes the exact sum; the QC re-run of the published datasets follows |
| The all-transcript figures pruned parquet files by the boxes stored in the points index, which the fingerprint (names, row counts) does not validate | Files are chosen by their own footer statistics, as QC's registration windows are; section bounds come from the footers too, so the overview crop no longer centres on the build's recorded `bounds_um` (0.7 um wider than the rows on the full section) |
| The wide-panel paths (sparse means above 1,000 genes, dotplot gene selection above 400) depend on panel width, not platform | Documented as panel-width rules: the 300-gene XENIUM/MERSCOPE panels never reach them; a XENIUM 5K panel would (its dotplot exceeded the render limit before). Since the user decision of 2026-11-06 the dotplot selection (top 5 genes per group) applies to every panel size |
| "All transcripts" means all QV >= 20 transcripts (1,409,601,228 of 1,640,508,930 vendor rows on the full section; D3, `atera_min_qv`) | Stated in the visualization doc and in AT-6's status |
| The resumed viewer-cache summary records the mask, label and image pyramids as skipped (built in the first attempt) | Noted; the pyramids were built at 03:26–03:29 in the first attempt. Part of the AT-7 restart work with the `-resume` rerun of ENRICH |
| The source stores' hardlinked payloads are now read-only (143 crop, 1,543 full-section files) | Announced to whoever shares the AT-5 stores (exit report) |

**AT-8 user decisions (2026-11-07).** Taken on the AT-8 design report (remaining scope after AT-6 and AT-9):

| Question | Decision |
|---|---|
| Transcript pair correlation on ATERA | Off on the ATERA (label-raster) transcript-pattern engine in AT-8, and recorded in its summary. An all-transcript method (raster FFT plus an analytic null) is a follow-up; it may later replace the 5,000-transcripts-per-gene cap of the XENIUM/MERSCOPE engine |
| Q22, panels above 5,000 genes | The whole-transcriptome rules apply automatically to any panel with more than 5,000 genes, whatever the platform: HVG-only PCA in legacy clustering, the streaming MECR reference (`reference_mode` auto), the pciSeq refusal without an explicit gene list, and GASTON raising instead of truncating. Not warn-only. Outputs of 5K-panel Xenium rows change (accepted); task hashes of the ~300-gene XENIUM/MERSCOPE rows stay unchanged |
| Analysis gene set | RCA on ATERA uses all 18,049 panel genes (recipe `all_panel_genes`, the identity), not a subset, for the best achievable accuracy even at a much higher compute cost. AT-8 builds the analysis-gene-set machinery with ATERA defaulting to `all_panel_genes`. The overlap with the Xenium Prime 5K human panel (4,874 of 5,001 genes) is information only. (This supersedes an earlier decision the same day to leave the choice to AT-11.) |
| Tissue support for non-cortical sections | A mask from the all-transcript density (10 µm raster, threshold, morphological close, hole fill), with the method recorded; hand-drawn tissue annotations take precedence when present |

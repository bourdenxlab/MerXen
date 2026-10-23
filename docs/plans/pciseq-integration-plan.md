# Optional pciSeq cell typing

Implement on `feature/pciSeq`, following three worker investigations of the
upstream API, Nextflow orchestration, and MerXen's spatial schema.

The subsequent [reference-preparation layer](pciseq-reference-preparation-plan.md)
extends the initial explicit-matrix contract with species-matched Allen atlases,
regional selection, donor-balanced profiles, and custom reference preparation.

## Contracts established before implementation

- Add `pciseq` immediately after `clustering_squidpy`, disabled by default.
  Run per active platform and analysis segmentation, with published-input
  support for `--only_stage pciseq`. Enabling it extends the historical default
  stop without implicitly enabling MapMyCells.
- Split preparation, computation, and finalization. Preparation and
  finalization use MerXen's SpatialData environment; computation uses an
  isolated, pinned pciSeq environment. Later clustering consumers wait for
  pciSeq completion when enabled.
- Use the public `pciSeq.fit` API from `acycliq/pciSeq_3d`, commit
  `5e2c0fd1841f8a4022a2fa341ece965687e343dd`. Require an explicit reference
  expression matrix with genes as rows and cell types as columns. Pin the
  compute runtime to pandas 2.2 or later and below 3, matching upstream API use.
- Read native segmentation shapes and native transcript points in a common
  coordinate frame. Clustering's aligned geometry must not replace native
  seed geometry. Include segmentation cells and transcripts independently of
  clustering QC filters; annotate the clustered cell subset afterward.
- Default to an explicitly recorded `project_2d` mode, rasterizing native cell
  polygons and projecting transcript XY coordinates. Genuine 3D input requires
  an explicit label volume, measured transcript Z coordinates, physical origin,
  and voxel dimensions. Never invent a 3D volume from polygon outlines.
- Map canonical uint64 cell IDs reversibly to dense positive mask labels.
  Preserve transcript IDs through filtering and upstream output. Validate
  geometry, coordinate transforms, reference overlap, empty inputs, and raster
  resource limits before calling the model.
- Run headlessly. Disable upstream viewer/export side effects and retain raw
  cell/spot result records plus ID maps. Returned class probabilities are
  rounded and thresholded upstream, so describe them as reported summaries.
- Join predictions by explicit canonical IDs, retaining clustered counts and
  existing annotations. Write a new H5AD and optionally a derived SpatialData
  table using MerXen's store locking/writer conventions.

## Work allocation

1. Workflow worker: Nextflow module, stage selection, completion gates,
   samplesheet settings, and behavioral workflow checks.
2. Spatial-data worker: input adaptation, serialization, output import, and
   focused data-integrity tests.
3. Upstream worker: isolated compute entry point, API-contract tests, and a
   real synthetic fit using the pinned upstream implementation.
4. Main worker: validated configuration, CLI, dependency declarations and lock,
   user documentation, integrated review, and repository checks.

## Acceptance checks

- Disabled runs retain their stage selection; enabled and standalone runs
  select the new stage and enforce completion before downstream consumers.
- Known tiny inputs preserve coordinates and IDs, reject unusable references,
  and retain unassigned transcripts that have supported genes.
- A synthetic real upstream fit completes headlessly. Adapter/finalization
  tests verify derived annotations against explicit ID mappings.
- Targeted and existing workflow tests, lint, formatting, and type checks pass.
  Record any environment-dependent checks that could not execute.

## Serialized process boundary

Preparation writes one platform/segmentation task per directory:
`manifest.json`, `spots.csv`, `reference.csv`, `masks/plane_0000.npz` (one COO
plane per Z index), `cell_id_map.csv`, `spot_id_map.csv`, and `source.h5ad`.
The manifest records `options`, `voxel_size` in XYZ order, `cell_count`, and
`geometry_mode`. Spots use `gene_name`, `x`, `y`, and optional `z_plane`;
their row numbers are the upstream spot IDs. All filtering happens before
these row numbers and the transcript map are written. Mask labels are dense
positive integers, with zero reserved for background.

The isolated `python -m merxen.pciseq_compute` entry point reads only this
boundary and imports pciSeq lazily. It writes `cell_data.json` and
`gene_data.json` as JSON records, plus `compute_manifest.json`. It verifies
that every prepared cell and transcript survives the call with unchanged
identifiers and transcript gene identity. The wrapper forces headless
operation, disables upstream exports and single-plane cell removal, and
caps the requested neighbor count to the available cells.

The public upstream `fit` API returns results even when it reaches its
iteration limit and does not expose the fitted model or convergence flag.
Record convergence as unavailable, rather than treating a successful return
as proof of convergence. The returned class lists retain probabilities above
0.001 and truncate them to three decimal places; they can be empty for very
diffuse posteriors and must not be described as a complete posterior matrix.

## Verified upstream compatibility

The isolated, generated lock was exercised through MerXen's compute entry
point against the pinned upstream source. Synthetic 2D inputs passed with
one, two, and sixteen cells; the sixteen-cell run retained all 900 spot IDs.
A genuine three-plane input also retained sixteen cells and 900 spots with
anisotropic Z spacing. These intentionally short two-iteration smoke runs
validate execution and identifier contracts, rather than convergence or
biological accuracy.

The smoke runs exposed two dependency constraints: pandas must remain below
3 because upstream uses `include_groups=True`; the runtime also pins NumPy
below 2.4. The initial NumPy 2.5.4 resolution returned complex eigenvectors
for the real background covariance and broke upstream's 3D Euler-angle
export; the verified resolution is NumPy 2.3.5. Square XY voxels are required
because upstream scales Y before indexing the original raster; anisotropic
Z spacing is supported.

A full synthetic SpatialData round trip also passed preparation, a separate
locked compute process, and finalization with derived-table writeback.
Sixteen native model cells and 900 transcripts produced annotations for the
eight-cell clustered subset. Canonical cell IDs above the exact-integer
range of float64 survived unchanged, the clustered counts and Leiden labels
were preserved, and the source table remained unchanged when the derived
pciSeq table was added.

## Hybrid segmentation refinement

Implementation review confirmed that native ProSeg hybrid polygons permit
positive-area overlaps by design. The projection adapter therefore defaults
to an explicit `background` overlap policy: ambiguous raster seed pixels
become background, allowing pciSeq to estimate their assignment. Native
polygons and transcript assignments are preserved. The alternative `error`
policy retains strict rejection, boundary-only touches keep a deterministic
tie break, and a cell losing all seed pixels fails preparation. The manifest
records the policy and affected pixels.

## Validation results

- Full non-slow regression run: 1,777 passed and 35 skipped. The skips cover
  optional environments/data; no tests failed.
- After final overlap and provenance hardening, the affected data, compute,
  configuration, CLI, workflow, and samplesheet checks passed: 88 tests.
- Repository-wide Ruff lint, formatting, and mypy checks passed. Lock/cache
  checksum, whitespace, and added-text privacy checks passed.
- A paired/all-segmentation Nextflow stub run completed 24 process tasks and
  published eight per-platform/segmentation outputs. A separate completion-gate
  check verified that a partial pair stays blocked without delaying other pairs.
- A real overlapping-hybrid fit retained sixteen cells and 900 transcript IDs;
  one overlap pair contributed 70 background seed pixels. Preparation and
  computation manifests were linked by SHA-256.

The real fits use synthetic inputs. Biological assignment quality on a user's
data still depends on its reference and model settings. Container build and
full production-dataset execution were not part of these checks.

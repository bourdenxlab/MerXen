# Section alignment

> **Status: optional stage; VALIS is the default backend.** Enable it globally
> with `--analysis_mode paired --enable_alignment true`, or per paired
> samplesheet row with `enable_alignment=true`.

## Intent

Adjacent MERSCOPE and Xenium sections can differ by arbitrary rotation,
translation, modest scale, partial tissue, and local section deformation.
`ALIGN` maps the moving dataset into the fixed dataset's physical coordinate
system. Xenium is fixed and MERSCOPE is moving by default; both are
configurable.

The default backend uses only DAPI morphology for registration and QC. It does
not use transcripts, expression, cell types, RNA images, or cell
correspondences.

## Default VALIS method

The stage:

1. Resolves each selected DAPI image, its pixel size, and its SpatialData
   dataset-physical ↔ image-pixel affine. It fails rather than guessing when
   these are ambiguous. Platform-specific matrix and pixel-size overrides are
   available for external images.
2. Selects DAPI by channel name and derives an acquired-support mask. Vizgen
   MERSCOPE's exact-zero, FOV-shaped padding defines its irregular support;
   Xenium and uncertain sparse inputs conservatively retain rectangular
   support. Broad background and final smoothing use support-normalized
   convolution, `G(image × support) / G(support)`, so unacquired zeros cannot
   create a positive high-pass rim. Robust clipping, compression, and mild
   CLAHE are followed by a 150 µm cosine taper measured inward from the actual
   support boundary.
3. Creates a downsampled tissue-density mask with automatic thresholding,
   closing, hole filling, physical-area filtering, and edge dilation. Multiple
   substantial fragments are retained. The registration validity mask erodes
   acquired support by 150 µm, and tissue outside that safe domain is excluded
   from features, rigid scoring, and QC.
4. Resamples both images to a shared isotropic physical pixel size and places
   them on a padded registration-only canvas. Source images and native
   SpatialData elements are not resampled or overwritten.
5. Estimates a moving-to-fixed rigid `T_pre` after metadata-based physical
   resampling. SIFT/RANSAC is accepted only with adequate inliers, spatial
   coverage, and tissue overlap, then its matches are refit with scale fixed
   to one. Otherwise a configurable 0–360° search uses mask distance
   correlation, Dice, and DAPI normalized mutual information. Reflections are
   searched by default for paired MERSCOPE/Xenium sections, but the reflected
   candidate must beat the non-reflected candidate by a configured margin.
6. Runs a final local orientation search on the selected handedness over
   ±2.5° and ±500 µm in X/Y. A dense coarse score volume locates nearby 3D
   maxima, and a finer grid tests whether each maximum persists away from the
   search boundaries. The best local transform is retained. QC records whether
   that coordinate is stable, whether another stable maximum exists nearby,
   and whether another maximum lies within the configured competing-score
   margin.
7. Jointly refines rotation and X/Y translation with a partial-overlap
   objective. A trimmed symmetric tissue-boundary distance ignores the worst
   unmatched boundary fraction, while overlap-normalized DAPI-density
   correlation selects internal anatomical agreement. Independent fixed and
   moving coverage gates prevent a tiny coincidental match. Scale and shear
   remain disabled. Its objective diagnostic distinguishes zero, centroid, and
   phase-correlation seeds, plots refined candidates in physical units, and
   includes a true local X/Y score slice around the selected solution.
8. Pre-warps only the temporary moving registration image with the refined
   `T_pre`. This accepted MerXen transform is the complete authoritative global
   alignment: VALIS receives the already-aligned image with `do_rigid=false`
   and an identity-only transformer. A dense postcondition verifies that
   VALIS's rigid point mapping remains identity before any field is accepted.
9. Runs VALIS only as a local deformation engine. Both temporary inputs are
   weighted by the feathered intersection of fixed/moving tissue and their
   footprint-eroded validity masks. The reproducible default is VALIS
   `OpticalFlowWarper`; `simple_elastix` remains an explicit alternative and
   errors when Elastix is unavailable. DISK/LightGlue may still be used for
   VALIS bookkeeping and QC, but cannot change the locked global frame.
10. Samples forward and backward transformations through VALIS's point-warp
   APIs, preserving its level, crop, and direction conventions. Displacements
   taper smoothly to zero outside shared valid tissue. Raw internal VALIS field
   arrays are never serialized directly.
11. Selects non-rigid output only when NMI does not degrade and the independent
    DAPI-density, tissue-Dice, and authoritative partial-overlap score gates
    pass. P95 displacement, Jacobian, coherent Euclidean drift (default at most
    0.25° and 25 µm), and topology gates must also pass. Otherwise the locked
    partial-overlap transform is retained.

Micro-rigid and micro-non-rigid registration are not run. For two slides,
`compose_non_rigid=false`.

## Coordinate convention

All image points are `(x, y)` with a top-left origin; array shapes remain
`(row, column)`. Matrices are forward 3×3 homogeneous xy matrices. The applied
chain is:

```text
moving dataset µm
  → moving original DAPI pixels
  → moving registration pixels
  → T_pre / pre-oriented pixels
  → VALIS global + optional non-rigid warp
  → fixed registration pixels
  → fixed original DAPI pixels
  → fixed dataset µm
```

`T_pre` is applied exactly once. VALIS point conversion uses explicit source
and destination image shapes. The transform bundle stores the individual
matrices plus sampled forward and backward non-rigid fields, and can be
reloaded without the JVM.

Native SpatialData elements remain untouched. For compatibility with existing
downstream branch selection, materialized selected VALIS vectors use the
existing `*_aligned_nonrigid` suffix even when QC selected the global fallback.
Their `merxen_alignment` metadata records the actual selected mode and backend.
The native elements also receive the global affine in the configured named
coordinate system (default `merxen_xenium`). Table centroids are preserved in
`obsm["spatial"]` and added in `obsm["spatial_merxen_xenium"]`.

Transformed points and shape centroids receive
`in_shared_tissue_domain`. This marks the intersection of fixed tissue and
registered moving tissue without discarding platform-only regions.

## Nextflow and configuration

`ALIGN` runs after per-platform `QC` and before paired downstream stages.
`ALIGN_QC` collates the already-computed DAPI QC report and selected overlay;
it does not recompute expression metrics. When alignment is disabled,
downstream stages receive the enriched native zarrs directly.

Important defaults in `workflows/nextflow.config` include:

| Parameter | Default | Meaning |
|---|---:|---|
| `alignment_backend` | `valis` | `valis` or explicit `legacy_spateo`. |
| `alignment_fixed_platform` / `alignment_moving_platform` | `XENIUM` / `MERSCOPE` | Reference and transformed platforms. |
| `alignment_*_image_key` | platform default | SpatialData DAPI image element. |
| `alignment_*_pixel_size_um` | `null` | Optional validated physical-size override. |
| `alignment_registration_source_max_dim_px` | `3200` | Bounds temporary padded registration images. |
| `alignment_background_sigma_um` | `75` | Broad DAPI background scale. |
| `alignment_background_boundary_mode` | `mirror` | Boundary mode used inside support-normalized Gaussian filtering. |
| `alignment_edge_taper_um` / `alignment_edge_exclusion_um` | `150` / `150` | Taper registration intensities and exclude scoring inward from the actual acquired-support boundary. |
| `alignment_smoothing_sigma_um` | `3` | Nuclear-density smoothing scale. |
| `alignment_orientation_*_step_degrees` | `10`, `2`, `0.5` | Full-circle coarse and refinement increments. |
| `alignment_allow_reflection` / `alignment_reflection_mode` | `true` / `auto` | Search independent handedness branches, or explicitly `force`/`forbid` reflection for a sample. |
| `alignment_reflection_minimum_score_improvement` | `0.01` | Symmetric handedness confidence margin. Near ties proceed provisionally with an ambiguity flag. |
| `alignment_orientation_translation_candidates_per_angle` | `3` | Translation initial conditions retained for every angle before joint refinement. |
| `alignment_orientation_*_translation_radius_px` | `64`, `16`, `4` | Coarse, refined, and final translation neighborhoods on the low-resolution search canvas. |
| `alignment_orientation_initial_angle_degrees` / `alignment_orientation_initial_translation_*_um` | `null` | Optional per-sample angle and physical-translation seeds. |
| `alignment_orientation_local_fine_search_enabled` | `true` | Run the final local stability and competing-maximum search on the selected handedness. |
| `alignment_orientation_local_fine_angle_radius_degrees` | `2.5` | Local angular radius around the selected orientation. |
| `alignment_orientation_local_fine_translation_radius_um` | `500` | Local X/Y translation radius in full-scale physical coordinates. |
| `alignment_orientation_local_fine_coarse_angle_step_degrees` / `alignment_orientation_local_fine_coarse_translation_step_um` | `0.5` / `100` | Initial deterministic 3D score-grid increments. |
| `alignment_orientation_local_fine_refine_angle_step_degrees` / `alignment_orientation_local_fine_refine_translation_step_um` | `0.1` / `25` | Fine increments used to test persistence of local maxima. |
| `alignment_orientation_local_fine_competing_score_margin` | `0.002` | Score distance within which another persistent maximum is flagged as competing. |
| `alignment_partial_overlap_enabled` | `true` | Refine rigid rotation/translation using trimmed boundaries and DAPI density. |
| `alignment_partial_overlap_angle_radius_degrees` / `alignment_partial_overlap_angle_step_degrees` | `10` / `1` | Residual rotation search around the coarse pre-orientation. |
| `alignment_partial_overlap_max_translation_um` | `1500` | Bounded X/Y refinement range in physical units. |
| `alignment_partial_overlap_retained_boundary_fraction` | `0.7` | Closest boundary fraction retained by the robust distance. |
| `alignment_valis_num_features` | `7500` | DISK feature count. |
| `alignment_valis_max_processed_image_dim_px` | `1600` | VALIS global feature image limit. |
| `alignment_valis_max_non_rigid_registration_dim_px` | `3200` | VALIS non-rigid image limit. |
| `alignment_valis_global_transform` | `rigid` | Deprecated configuration/provenance field retained for compatibility. VALIS global fitting is disabled; `T_pre` is locked. |
| `alignment_seed` | `21` | Deterministic Python/NumPy/OpenCV/PyTorch seed. |
| `alignment_valis_non_rigid_backend` | `optical_flow` | Explicit non-rigid backend. |
| `alignment_coordinate_system_name` | `merxen_xenium` | Registered SpatialData coordinate system. |
| `alignment_resume` | `true` | Reload a complete parameter-compatible transform bundle for direct reruns. |

The Pydantic `ValisAlignmentConfig` exposes the full preprocessing, masking,
orientation, feature, transform, non-rigid, output, resume, and QC thresholds.
Direct CLI JSON additionally supports external DAPI paths and explicit 3×3
`dataset_to_image_matrix` values.

The orientation QC directory records both handedness branches in
`orientation_candidates.json`, side-by-side candidate overlays, and an
angle/translation landscape. The final local pass additionally writes
`orientation_local_fine_search.json`, a before/after overlay, and projected
angle/X/Y score landscapes with persistent and boundary maxima marked.
Candidate metadata uses explicit search-angle, matrix-angle, reflection-axis,
and equivalent-flip fields.

## Environment

VALIS 1.2 declares NumPy `<2`, while SpatialData 0.8 requires NumPy 2. The
dedicated alignment environment therefore installs the generated
`requirements/requirements.alignment.lock` (including a NumPy-2-compatible OpenCV,
PyTorch/Kornia, SimpleITK, Java, and libvips stack), then installs the exact
`valis-wsi==1.2.0` package without re-resolving its old dependency metadata.
MerXen applies narrow compatibility aliases for the two removed NumPy scalar
names still referenced by VALIS. `merxen check-alignment-deps --backend valis`
validates the resulting stack. The exact imported versions are recorded in
every registration summary.

The JVM is shut down in a `finally` block after every VALIS attempt.

## Legacy Spateo backend

The former expression/cell-centroid implementation remains available only via:

```bash
nextflow run workflows/main.nf --enable_alignment true \
  --alignment_backend legacy_spateo
```

Its code lives in `alignment/legacy_spateo.py`,
`alignment/legacy_features.py`, and `alignment/legacy_qc.py`. The old Spateo
parameters remain under `legacy_spateo` in CLI JSON and retain their existing
Nextflow names. Compatibility wrappers preserve older Python imports, but this
backend is not the default and its expression-based QC is never used for a
VALIS run.

## Outputs

`${outdir}/<pair_id>/alignment/align_out/` contains:

| Artifact | Contents |
|---|---|
| `alignment_transform.json` | Pipeline contract: selected transform, metadata, parameters, QC, and dependency versions. |
| `transform_chain.json` | Explicit physical/pixel/registration frame chain and matrices. |
| `forward_displacement_field.npz` / `backward_displacement_field.npz` | Sampled fields when non-rigid registration ran. |
| `registration_summary.json` / `.csv` | Status (`non_rigid_pass`, `global_only`, or failure diagnostics) and stage-wise DAPI QC. |
| `resume_manifest.json` | Exact input paths, platform roles, and VALIS parameters required before a completed bundle can be reused. |
| `shared_tissue_mask.npy` / `.tif` | Valid cross-platform comparison domain on the original fixed DAPI pixel grid. |
| `shared_tissue_mask_registration.npy` / `.tif` | The same domain on the padded registration grid used for point annotation. |
| `registration_inputs/` | Processed DAPI images, tissue masks, outlines, edge-validity masks, and edge-artifact metrics. |
| `registration_images/` | Stable fixed and pre-oriented TIFFs supplied to VALIS. |
| `valis/` | VALIS registrar, summary, thumbnails, matches, overlaps, and deformation artifacts. |
| `qc/partial_overlap/` | Before/after overlays, candidate contact sheet, robust-objective profile, and candidate metrics. |
| `qc/` | Original, pre-oriented, global, non-rigid, checkerboard, mask, feature, displacement, Jacobian, and deformation-grid views. |
| `alignment_coords/` | Legacy coordinate diagnostics; retained as an empty contract directory for VALIS. |

The separate `${outdir}/<pair_id>/alignment_qc/` directory contains a compact
JSON/CSV copy of the selected DAPI QC and the downstream overlay PNG/PDF.

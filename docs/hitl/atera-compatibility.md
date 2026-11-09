# HITL compatibility after Atera integration

Audit date: 2026-11-09. Base: `feature/atera-support` at `f7954247`.
HITL-1 merged in PR #75 (`43cf265d`). Atera's current segmentation and
downstream implementation is on the integration branch, ahead of `main`;
this update targets that branch, following the Atera plan's PR workflow.

## What changed since HITL-1

| Change | HITL consequence |
|---|---|
| Platform registry and Atera loader | Model metadata and inference identities already accept Atera. The loader uses its native morphology, precise pixel size and image-only reads. These changes are retained. |
| Atera's separate Cellpose/tiled-ProSeg route | Atera writes no transcript CSV and keeps vendor nuclei. Existing custom-cache preflight already honors the CSV exception; a regression now checks complete custom-cache reuse and mismatch refusal. |
| Vendor-mask routes | Atera defaults to `native`. A custom model or threshold override on `native`/`native_proseg` could previously validate and then be ignored. Preflight now requires `cellpose_proseg` when these options are supplied. |
| MERSCOPE image preprocessing and bead-mask veto | The tile diagnostic already checked source-image identity, but not the new veto. It now checks the veto identity and reproduces it in halo-local coordinates after area filtering. |
| Memory-adaptive tile selection | The fixture used a fixed Xenium resolution and tile assumption. It now reads platform/species/native geometry and can pin the actual grid from baseline statistics. The diagnostic refuses a different tile size or overlap. |
| Review-store exports | Pilot export explicitly selects the native image and chosen label layer, reads no tables/points, checks geometry and writes only to a fresh independent folder. |

The platform/channel/model checks do not make an old Xenium model an Atera
model: metadata must name Atera, the exact stain order, native pixel size,
evaluated tile size and runtime. The stock-weight candidate fixture validates
loading/provenance; it does not demonstrate better masks.

## Initial crop export and annotation

HITL-0 and HITL-1 provided runtime/provenance and model plumbing. They did not
implement annotation or training. This update begins HITL-2 with
`merxen cellpose-hitl-export`: a small tissue-spanning training pilot, explicit
native ROI support, GUI-compatible pre-fills, previews and a versioned guide.
The dedicated napari ROI helper lets users inspect native images and masks,
select locations, and return a source/grid-bound GeoJSON to the exporter.
[Both workflows](pilot-workflows.md) begin with masks and viewer caches and
finish with editing cells in the separate Cellpose GUI.

Model inputs follow route A: fetch the complete original halo tile, apply the
production joint p2/p98 uint8 conversion, add the batch dimension, apply
Cellpose's default per-channel normalization, and only then crop. Percentile
calls, numeric dtype and strided sampling are recorded and reconstructed
exactly. Constant/padded channels stay zero. Disconnected pieces of a cropped
label become separate eight-connected objects; touching different labels stay
separate. The manifest binds the arrays, image/mask hashes and tile context.

Tissue sampling uses image signal and spatial spread. Pilot crops are marked
`pilot_train` and `evaluation_eligible=false`; they are not an independently
sampled test set or a hippocampal region-balanced design. Full anatomical
stratification, import validation, fine-tuning, evaluation and adoption remain
subsequent work. The [annotation guide](annotation-guide.md) records the pilot
style and save/review procedure.

See [the testing guide](testing-guide.md#5-atera-correction-pilot) for the config
and commands. Runtime configs, reports and crop data belong outside git under
the site's configured evidence/output locations. Existing experimental and
published outputs are preserved; default stock segmentation is unchanged.

## Validation

A production-context Atera GPU tile check reproduced all 2,172 reference cells:
AP at IoU 0.5, mean matched IoU and foreground IoU were all 1.0. The baseline
mask and statistics checksums stayed unchanged. The check used Cellpose 4.2.1.1
and PyTorch 2.14.1+cu130, with peak allocated GPU memory of about 6.33 GB.

Six automatic tissue crops and one viewer-selected crop were exported from a
completed Atera review store. File checksums and Cellpose image reading passed.
The Qt add/save round trip was exercised with real napari image/shape models;
an interactive OpenGL viewer session remains a user acceptance check. These
results establish compatibility and export integrity, not improved segmentation
accuracy: that requires human corrections and subsequent training/evaluation.

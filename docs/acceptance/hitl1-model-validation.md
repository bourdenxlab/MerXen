# HITL-1 model-plumbing validation — 2026-10-19

Source: `bd50f06`, on `codex/hitl-1-model-plumbing`, based on HITL-0's merge
into `main`. This validates optional model selection and the shared production
tile function. The candidate fixture contains stock weights; it does not
evaluate a trained model or biological segmentation accuracy.

*Names and paths (2026-10-21).* Under the privacy rule of 2026-10-14 (CLAUDE.md), this record names the sample by its alias (`docs/local-configuration.md`) and host locations by `.env` variables (`$MERXEN_...`); a path that contains the alias names a directory that carries the original name. No number, checksum or verdict changed.

## Production-context GPU check: passed

The documented fixture helper wrapped Dwight's verified stock `cpsam_v2`
checkpoint as a candidate bundle with explicit opt-in. Only the weights were
linked. Its configuration cleared all persistent baseline output paths.
The acceptance workflow acquired `/tmp/merxen-dwight-gpu.lock`, read the
published source and baseline, and wrote all predictions outside published
results. The production Conda environment was reused without modification.

| Setting | Recorded value |
|---|---|
| Sample and native image | HD1 Xenium, `morphology_focus` |
| Input channels | DAPI, then 18S |
| Native pixel size | 0.2125 µm on both axes |
| Model selection | Explicit candidate bundle path, verified stock weights |
| Cellpose / torch | 4.2.1.1 / 2.14.0+cu130 |
| Runtime / hardware | Existing Conda environment / RTX A5000 |
| Nextflow | Dwight's installed 26.04.4 |
| Tile / halo size / overlap | Index 44 / 6144 px / 256 px |
| Halo bbox, y0/y1/x0/x1 | 22272 / 28416 / 22272 / 28416 px |
| Scoring bbox, y0/y1/x0/x1 | 23184 / 27504 / 23184 / 27504 px |
| Scoring rule | 656 px core margin; exclude cut border labels |
| Precision / flow / cellprob | float32 / 0.7 / -5.0 |
| Final area filter | 5–400 µm² |
| Reference / predicted cells scored | 3745 / 3745 |
| TP / FP / FN at IoU 0.5 | 3745 / 0 / 0 |
| AP at IoU 0.5 | 1.0 |
| Mean matched IoU | 0.9999953134062893 |
| Foreground IoU | 0.9999950558905552 |
| Acceptance threshold | AP and mean matched IoU both ≥ 0.99 |
| Peak torch GPU allocation | 6.374 GiB |
| Total task time, including input checksums | 149.4 seconds |

The report's model identity is `verified` and includes the selected bundle path,
checkpoint checksum, exact runtime, native image, ordered channels, both pixel
sizes, effective filters and tiling settings. The assertion in the testing guide
printed `HITL1_STOCK_FIXTURE_OK`.

Checkpoint SHA-256:
`0f1cc3f7ecdd8a037a57c6c48d9d8921391be4cbce3fa9f13c3e3a2e1253c667`.

Published mask SHA-256 was identical before and after:
`0a3e8b79ed2e8da1df6731c8fcd4be10a56e0cce241c89ae47f42cd193497b41`.
Published stitching-statistics SHA-256 was also unchanged:
`bac2a39ebdbd2e374a0db5026f9c1ac51b50c6d27e129cf9475f79f1545666b4`.
These also match the earlier HITL-0 acceptance record.

The full report, prediction, identity sidecar, fixture metadata/configuration and
task logs are preserved on Dwight under:

```text
$MERXEN_ANNOTATION_DEV_DIR/hitl_acceptance/2026-10-19/hitl1-HD1-t44.BO4r5m/
```

The fixture's metadata labels it for plumbing tests only. A tile report is not a
complete reusable full-section segmentation cache.

## Workflow and code checks

[Linux CI](https://github.com/bourdenxlab/MerXen/actions/runs/39304841490) passed
820 tests with 3 slow tests deselected and no skips. Ruff lint/format and mypy
passed. CI installed the pinned runtime and Nextflow 26.10.2; executable checks
cover default config equivalence, optional overrides, bundle validation,
protected paths, complete/interrupted caches and downstream saved-mask identity
verification.

A separate main-workflow preview used the real HD1 source on Dwight with
`-profile dwight,conda`, no model/override options, and `--stop_stage segment`.
It exited successfully without creating source zarrs or prediction arrays in
the isolated arm. The preview log and samplesheet are preserved beside the
GPU report under `default-preview/`.

Reproduction commands are in the [testing guide](../hitl/testing-guide.md),
steps 1 and 4. Container validation remains deferred; this check used Conda.

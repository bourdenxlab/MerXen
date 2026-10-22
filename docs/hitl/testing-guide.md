# Testing the Cellpose HITL changes

Start with HITL-0's runtime and provenance checks. HITL-1 follows after HITL-0's
acceptance and merge. The published baseline is `$MERXEN_RESULTS_ROOT/`;
all acceptance predictions and reports must go outside that directory.

Commands below name samples by their aliases (HD1; see
[sample aliases](../local-configuration.md#sample-aliases)) and host locations by
`.env` variables (`$MERXEN_...`). Load the variables into the shell first
(`. "$MERXEN_ENV_FILE"`, or `. .env` in the main checkout), and give the real pair
id (`.aliases.tsv`) where a path or samplesheet shows an alias: result directories
carry the real pair ids.

## 1. Check the code without running segmentation

On Dwight, check out the implementation branch in a separate test checkout.
Use `origin/main` instead once HITL-1 has merged:

```bash
cd /path/to/MerXen
git fetch origin
export MERXEN_TEST_CHECKOUT=$(mktemp -d $MERXEN_TASK_TMP_DIR/hitl-checkout.XXXXXX)
git worktree add --detach "$MERXEN_TEST_CHECKOUT" origin/codex/hitl-1-model-plumbing
cd "$MERXEN_TEST_CHECKOUT"
export PYTHONPATH="$MERXEN_TEST_CHECKOUT/src"
$MERXEN_STORAGE_ROOT/ci-venv/bin/python -m pytest \
  tests/test_segmentation/test_model_identity.py \
  tests/test_segmentation/test_output_paths.py \
  tests/test_scripts/test_hitl0_tile_check.py \
  tests/test_workflows/test_segmentation_subworkflow.py \
  tests/test_workflows/test_env_lock_sync.py
```

Expected: every selected test passes. These use small synthetic inputs and
mocked model loading, so this step does not run Cellpose on your images or
change the production environment. It checks explicit model selection,
checksums, legacy reuse, provenance through staged symlinks and workflow wiring.

## 2. Check one real HD1 tile

Use the production environment checked on 2026-10-19, rather than rebuilding the
environment used by active pipeline runs. If that cached environment has since
been removed, select another environment satisfying the new pins first.

From the test checkout above:

```bash
export MERXEN_TEST_PYTHON=$MERXEN_STORAGE_ROOT/work/conda/env-0e319c40ef50b2834c65ae2f7f53bbcd/bin/python
export MERXEN_CHECK_ROOT=$(mktemp -d $MERXEN_TASK_TMP_DIR/hitl0-check.XXXXXX)
"$MERXEN_TEST_PYTHON" -c \
  'from merxen.segmentation.model_identity import require_cellpose_runtime; print(require_cellpose_runtime())'

nextflow run workflows/acceptance/hitl0_tile_check.nf \
  -ansi-log false -work-dir "$MERXEN_CHECK_ROOT/work" \
  --source_root "$MERXEN_TEST_CHECKOUT" \
  --python "$MERXEN_TEST_PYTHON" \
  --config $MERXEN_STORAGE_ROOT/work/d2/2bb63695bfa15cdb697b604cb6f624/segment_config.json \
  --source_zarr $MERXEN_STORAGE_ROOT/work/c9/800bf12cfc879ab1368ba767fff599/spatialdata_out/source_spatialdata.zarr \
  --baseline_mask $MERXEN_RESULTS_ROOT/HD1/xenium/segmentation/cellpose_masks_tiled.npy \
  --baseline_stats $MERXEN_RESULTS_ROOT/HD1/xenium/segmentation/cellpose_stitching_stats.json \
  --protected_root $MERXEN_RESULTS_ROOT \
  --gpu_lock_file /tmp/merxen-dwight-gpu.lock \
  --image_key morphology_focus --tile_index 44 \
  --outdir "$MERXEN_CHECK_ROOT/report"

export MERXEN_TILE_REPORT="$MERXEN_CHECK_ROOT/report/tile_check_out/tile_check.json"
"$MERXEN_TEST_PYTHON" - <<'PY'
import json
import os
from pathlib import Path

report = json.loads(Path(os.environ["MERXEN_TILE_REPORT"]).read_text())
print(json.dumps(report, indent=2))
assert report["baseline_unchanged"], "Published baseline checksum changed"
assert report["passed"], "Tile agreement requires investigation"
PY
```

The workflow waits for the shared GPU lock. It checks for at least 16 GiB of free
GPU memory, uses the recorded 6144 px tile grid and 256 px overlap, and preserves
the original channels, normalization and filters. It does not run ProSeg.
The scoring window excludes 656 px from every tile-core edge and drops labels
cut by the scoring-window border. Label numbers are remapped before comparison.

Expected report fields:

| Field | Expected |
|---|---|
| `model_identity.status` | `verified` |
| `model_identity.model_name` | `cpsam_v2` |
| `model_identity.cellpose_version` | `4.2.1.1` |
| `model_identity.torch_version` | `2.14.x`, including the actual CUDA build suffix |
| `model_identity.weights_sha256` | `0f1cc3f7ecdd8a037a57c6c48d9d8921391be4cbce3fa9f13c3e3a2e1253c667` |
| `agreement.ap_at_05` | At least `0.99` |
| `agreement.mean_matched_iou` | At least `0.99` |
| `baseline_unchanged` / `passed` | Both `true` |

The two overlap requirements make the plan's expected 0.99 agreement explicit:
AP penalizes missing and extra cells, while mean matched IoU checks their shapes.
This is a drift diagnostic against a published stock mask, not a claim that the
published mask is biological ground truth. The report also contains counts,
foreground IoU, tile coordinates, timing and peak GPU allocation.

Nextflow publishes completed diagnostic reports even when agreement is below
the threshold; its successful exit alone does not mean scientific acceptance.
The final Python assertion above checks that acceptance explicitly. A lower
score needs investigation before training. Preserve the report and the task's
`.command.log`; avoid changing thresholds just to make the check pass.

The source path above is the surviving source used by the published run. If it
has been cleaned up, stop and validate an equivalent native image in the durable
latest zarr before substituting it. Each repeat needs a fresh output directory.

## 3. Container validation: deferred

The agreed acceptance path for this milestone is Dwight's existing Conda
environment, using steps 1 and 2 above. Container validation is deferred and is
not an acceptance gate for HITL-0 or a prerequisite for starting HITL-1.

MerXen's container execution profile uses **Apptainer with SIF images**, and the
`gpu` profile adds `--nv` to GPU processes. Dwight's local workflow uses
`-profile dwight,conda`. See [Running on a cluster](../running-the-pipeline.md#running-on-a-cluster)
and [the execution profiles](../../workflows/nextflow.config).

The repository's Dockerfile is an existing image-build recipe; it does not make
Docker a requirement for running the pipeline. The earlier instructions to run
`sudo docker` were incorrect for this deployment and have been withdrawn.
Apptainer can consume OCI registry images directly without a local Docker daemon;
see [Apptainer's OCI support](https://apptainer.org/docs/user/main/docker_and_oci.html).

When container work resumes, choose an Apptainer host and an unprivileged build
route, build or obtain a refreshed SIF, and run the checkpoint/runtime and GPU
smoke tests with `apptainer exec --nv`. Neither Apptainer nor Singularity is on
Dwight's current PATH. The existing site profile still names an older CUDA 12.6
SIF; that artifact has not been rebuilt or validated against the new pins.
Do not use it for this milestone's fresh-inference checks. The CUDA 13.0
Dockerfile change remains unvalidated until the container work resumes.

## 4. Test HITL-1 without training a model

From the test checkout, first run the new fast checks:

```bash
$MERXEN_STORAGE_ROOT/ci-venv/bin/python -m pytest \
  tests/test_segmentation/test_hitl_registry.py \
  tests/test_segmentation/test_cellpose_tile.py \
  tests/test_segmentation/test_model_identity.py \
  tests/test_scripts/test_prepare_hitl1_fixture.py \
  tests/test_workflows/test_hitl_model_settings.py \
  tests/test_workflows/test_segmentation_subworkflow.py
```

Expected: all tests pass, including the actual Nextflow checks (no skips on
Dwight). They check byte-identical default configuration, each platform override,
candidate opt-in, model/image/runtime mismatches, cache reuse and refusal, and
protected paths through symlinks, including forced reruns. Synthetic output hashes
were captured before the shared tile function was extracted.

Prepare a candidate fixture using the already verified stock weights:

```bash
export MERXEN_TEST_PYTHON=$MERXEN_STORAGE_ROOT/work/conda/env-0e319c40ef50b2834c65ae2f7f53bbcd/bin/python
export MERXEN_HITL1_FIXTURE=$(mktemp -d $MERXEN_TASK_TMP_DIR/hitl1-fixture.XXXXXX)
"$MERXEN_TEST_PYTHON" scripts/acceptance/prepare_hitl1_fixture.py \
  --config $MERXEN_STORAGE_ROOT/work/d2/2bb63695bfa15cdb697b604cb6f624/segment_config.json \
  --source-zarr $MERXEN_STORAGE_ROOT/work/c9/800bf12cfc879ab1368ba767fff599/spatialdata_out/source_spatialdata.zarr \
  --weights "$HOME/.cellpose/models/cpsam_v2" \
  --protected-root $MERXEN_RESULTS_ROOT \
  --output-dir "$MERXEN_HITL1_FIXTURE"
```

Inspect `model.json`, `segment_config.json` and `nextflow-params.json` in that
directory. The fixture is marked **candidate**, with explicit opt-in in its
config. It links only the stock weights and clears every persistent baseline
output path. It creates no masks, training data or annotations. The fixture is
for testing plumbing; it is not an accepted fine-tune.

Run the same locked, production-context tile through the shared tile function:

```bash
export MERXEN_HITL1_CHECK_ROOT=$(mktemp -d $MERXEN_TASK_TMP_DIR/hitl1-check.XXXXXX)
nextflow run workflows/acceptance/hitl0_tile_check.nf \
  -ansi-log false -work-dir "$MERXEN_HITL1_CHECK_ROOT/work" \
  --source_root "$MERXEN_TEST_CHECKOUT" \
  --python "$MERXEN_TEST_PYTHON" \
  --config "$MERXEN_HITL1_FIXTURE/segment_config.json" \
  --source_zarr $MERXEN_STORAGE_ROOT/work/c9/800bf12cfc879ab1368ba767fff599/spatialdata_out/source_spatialdata.zarr \
  --baseline_mask $MERXEN_RESULTS_ROOT/HD1/xenium/segmentation/cellpose_masks_tiled.npy \
  --baseline_stats $MERXEN_RESULTS_ROOT/HD1/xenium/segmentation/cellpose_stitching_stats.json \
  --protected_root $MERXEN_RESULTS_ROOT \
  --gpu_lock_file /tmp/merxen-dwight-gpu.lock \
  --image_key morphology_focus --tile_index 44 \
  --outdir "$MERXEN_HITL1_CHECK_ROOT/report"

export MERXEN_TILE_REPORT="$MERXEN_HITL1_CHECK_ROOT/report/tile_check_out/tile_check.json"
"$MERXEN_TEST_PYTHON" - <<'PY'
import json
import os
from pathlib import Path

report = json.loads(Path(os.environ["MERXEN_TILE_REPORT"]).read_text())
assert report["baseline_unchanged"] and report["passed"], report
assert report["model_identity"]["status"] == "verified"
assert report["model_identity"]["inference_context"]["channels"] == ["DAPI", "18S"]
print(json.dumps(report["agreement"], indent=2))
print("HITL1_STOCK_FIXTURE_OK")
PY
```

Expected: `HITL1_STOCK_FIXTURE_OK`, AP and mean matched IoU at least 0.99, and
unchanged published checksums. The report's model name identifies the fixture
path; the weights SHA remains the stock `cpsam_v2` checksum. This check runs one
Cellpose tile and no ProSeg or full-section arm.

The [completed Dwight validation](../acceptance/hitl1-model-validation.md)
records the passing fixture run, runtime, agreement scores and preserved report
location. Use fresh fixture and report directories for each repeat.

To check the main-workflow wiring without scheduling segmentation, use the
fixture's `nextflow-params.json` with a HD1 samplesheet and `-preview`:

```bash
nextflow run workflows/main.nf -profile dwight,conda -preview \
  -params-file "$MERXEN_HITL1_FIXTURE/nextflow-params.json" \
  --samplesheet /absolute/path/to/HD1-samplesheet.csv \
  --analysis_mode xenium --stop_stage segment \
  --outdir "$MERXEN_HITL1_FIXTURE/preview-arm"
```

Use a samplesheet whose active platform is Xenium and whose source paths exist;
row settings override run defaults. Custom arms must copy any shared source
zarr outside the protected baseline rather than link it. Preview still runs
preflight but schedules no processes. The fast executable workflow tests above
exercise every override and refusal using small fixtures.

Cache-reuse checks use synthetic complete artifacts in the fast tests; a tile
report is not a reusable full-section segmentation. Full-section custom arms,
annotation and adoption decisions follow later milestones. Preserve failed
reports and task logs rather than relaxing the agreement threshold.

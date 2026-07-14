# MerXen

Pre-processing, segmentation, and comparative analysis of MERSCOPE and Xenium spatial transcriptomics datasets, either as paired sections or as single-platform runs.

## What it does

MerXen takes spatial transcriptomics datasets and runs a standardised pipeline. By default, each samplesheet row is treated as a paired experiment (one MERSCOPE, one Xenium per tissue section pair), but the same workflow can run MERSCOPE-only or Xenium-only analyses with `--analysis_mode merscope` / `--analysis_mode xenium` or row-level `analysis_mode` values in the samplesheet.

1. **SpatialData build** — Builds platform-specific SpatialData zarrs from raw MERSCOPE and Xenium output folders
2. **Cell segmentation** — A `SEGMENT` subworkflow chains GPU Cellpose-SAM image segmentation and CPU-only ProSeg transcript refinement as independently scheduled processes
3. **Mask image quantification** — Quantifies all SpatialData image channels over final Cellpose masks
4. **Section alignment** — Optionally registers paired adjacent sections to a Xenium reference coordinate system with Spateo
5. **Analysis and visualisation** — QC metrics, paired gene-level comparison when both platforms are present, single-platform or paired visualisation, per-gene spatial autocorrelation, first-pass Scanpy/Squidpy clustering, and optional local MapMyCells cell type assignment. By default, downstream analysis runs for both ProSeg-resegmented cells and original instrument segmentations; use `--analysis_segmentation reseg` or `--analysis_segmentation original_seg` to restrict it.
6. **Distance from object** — Optionally annotates cells by nearest registered polygon edge and runs grey-matter-only, paired near-vs-far PyDESeq2 for resegmented, original, and mask-derived cells.

The workflow is orchestrated by Nextflow to process multiple sample pairs with logging and reproducibility.

## Documentation

Full documentation lives in [docs/](docs/). Start with [docs/index.md](docs/index.md).

- Usage: [Getting started](docs/getting-started.md) · [Samplesheet format](docs/samplesheet.md) · [Running the pipeline](docs/running-the-pipeline.md) · [Configuration](docs/configuration.md) · [Outputs](docs/outputs.md)
- Pipeline stages: [SpatialData build](docs/stages/spatialdata-build.md) · [Segmentation](docs/stages/segmentation.md) · [Enrichment](docs/stages/enrichment.md) · [Mask image quantification](docs/stages/mask-image-quantification.md) · [Cortical depth](docs/stages/cortical-depth.md) · [QC](docs/stages/qc.md) · [Alignment](docs/stages/alignment.md) · [Comparison](docs/stages/comparison.md) · [Visualization](docs/stages/visualization.md) · [Spatial gene analysis](docs/stages/spatial-gene-analysis.md) · [Squidpy clustering](docs/stages/clustering-squidpy.md) · [Distance from object](docs/stages/distance-from-object.md) · [MapMyCells](docs/stages/mapmycells.md)
- Developer reference: [Pipeline architecture](docs/pipeline.md) · [Python API](docs/python-api.md) · [CLI reference](docs/cli.md) · [Development workflow](docs/development.md)

## Repository layout

```
MerXen/
├── workflows/                  # Nextflow pipeline
│   ├── main.nf                 # DSL2 entry point
│   ├── nextflow.config         # Parameters, executor, per-process resources
│   ├── samplesheet.example.csv # Template samplesheet
│   └── modules/                # One .nf module per pipeline stage
├── src/merxen/                 # Installable Python package
│   ├── config.py               # Pydantic configs (pipeline contract)
│   ├── cli/                    # Click entry points (one per stage)
│   ├── io/                     # Samplesheet, SpatialData builders, image/transcript I/O
│   ├── segmentation/           # Cellpose tiling + ProSeg subprocess
│   ├── enrichment/             # Shape layers + per-shape gene tables
│   ├── qc/                     # Per-dataset and cross-platform metrics
│   ├── visualization/          # Plotting
│   ├── analysis/               # Scanpy/Squidpy downstream analyses
│   ├── distance_from_object/   # Polygon-edge distance + paired pseudobulk DE
│   └── alignment/              # Optional Spateo cross-section registration
├── tests/                      # pytest suite, mirrors src/merxen/
├── docs/                       # Project documentation (start at docs/index.md)
├── notebooks/                  # Exploratory notebooks only
├── pyproject.toml              # Dependencies, merxen entry point, tool config
├── environment.yml             # Conda env (Python 3.12 + pip)
├── environment.alignment.yml   # Nextflow ALIGN env with Spateo bootstrap
├── environment.clustering-gpu.yml # Isolated RAPIDS clustering environment
├── Dockerfile.clustering-gpu   # Isolated RAPIDS clustering image
├── requirements.lock           # Pinned dependency tree
├── .env.example                # Required environment variables template
├── Agents.md                   # Project standards (must-read for contributors)
└── CLAUDE.md                   # Short overview + pointer to Agents.md
```

## Setup

```bash
# Create conda environment
conda env create -f environment.yml
conda activate merxen

# Install pre-commit hooks
pre-commit install
pre-commit install --hook-type pre-push
```

Nextflow uses `environment.alignment.yml` for the optional `ALIGN` process.
That process bootstraps Spateo in its own conda env and restores modern
AnnData, so regular pipeline stages are not affected by Spateo's older
dependency metadata.

## Required environment variables

Copy `.env.example` to `.env` and fill in values:

```bash
cp .env.example .env
```

See [.env.example](.env.example) for optional machine defaults and [docs/configuration.md](docs/configuration.md) for the full environment + Nextflow parameter reference. ProSeg is resolved automatically from the configured search paths and installed with Cargo if needed.

## Running the pipeline

```bash
# Run via Nextflow with a samplesheet
nextflow run workflows/main.nf --samplesheet samples.csv --outdir ./results

# Run a single-platform workflow
nextflow run workflows/main.nf --samplesheet samples.csv --analysis_mode xenium --outdir ./results
```

A template samplesheet is provided at [workflows/samplesheet.example.csv](workflows/samplesheet.example.csv). Copy and edit this file with your dataset-specific paths before running the workflow.

```bash
cp workflows/samplesheet.example.csv workflows/samplesheet.csv
```

The samplesheet points at raw platform folders with optional reusable SpatialData cache paths (`merscope_dir`, `merscope_spatialdata_path`, `xenium_dir`, `xenium_spatialdata_path`, plus per-platform channel, z-range, and voxel-layer settings). Optional row-level columns (`analysis_mode`, `enable_alignment`, `analysis_segmentation`, `cortical_depth_enabled`, `distance_from_object_enabled`, `distance_from_object_segmentations`, `start_stage`, `stop_stage`, `only_stage`) can override the run defaults per sample. Object-distance runs also provide registered object GeoJSON paths per platform. In single-platform rows, only the selected platform's source/cache columns are required. The full schema, validation rules, and worked examples are documented in [docs/samplesheet.md](docs/samplesheet.md). For Nextflow invocation options — analysis mode, resuming, stage-range runs, force rebuild, parameter overrides, cluster execution — see [docs/running-the-pipeline.md](docs/running-the-pipeline.md).

## Running tests

```bash
# All tests (excluding slow)
pytest

# Including integration tests
pytest -m "not slow"

# Full suite
pytest --run-slow
```

## Development

```bash
# Lint and format
ruff check . --fix
ruff format .

# Type check
mypy src/

# Run the same lockfile-backed checks as GitHub Actions
scripts/run_ci_checks.sh

# Bump package version
uv run bump-my-version bump patch  # small fixes, e.g. 0.1.0 -> 0.1.1
uv run bump-my-version bump minor  # new features, e.g. 0.1.0 -> 0.2.0
uv run bump-my-version bump major  # breaking changes, e.g. 0.1.0 -> 1.0.0

# Regenerate lockfile after changing dependencies
uv pip compile pyproject.toml --extra dev -o requirements.lock
```

Project standards (layout, dependencies, naming, type hints, docstrings, git workflow, commit message prefixes) are defined in [Agents.md](Agents.md). Day-to-day development mechanics — testing, pre-commit hooks, CI, debugging, adding a new pipeline stage — are in [docs/development.md](docs/development.md).

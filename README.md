# MerXen

Pre-processing, segmentation, and comparative analysis of paired MERSCOPE and Xenium spatial transcriptomics datasets.

## What it does

MerXen takes paired spatial transcriptomics datasets (one MERSCOPE, one Xenium per tissue section pair) and runs a standardised pipeline:

1. **SpatialData build** — Builds platform-specific SpatialData zarrs from raw MERSCOPE and Xenium output folders
2. **Cell segmentation** — Cellpose-SAM image-based segmentation followed by ProSeg transcript-based refinement
3. **Section alignment** — Optionally registers paired adjacent sections to a Xenium reference coordinate system with Spateo
4. **Comparative analysis** — QC metrics, gene-level comparison, visualisation, first-pass Scanpy/Squidpy clustering, and optional local MapMyCells cell type assignment across platforms

The workflow is orchestrated by Nextflow to process multiple sample pairs with logging and reproducibility.

## Documentation

Full documentation lives in [docs/](docs/). Start with [docs/index.md](docs/index.md).

- Usage: [Getting started](docs/getting-started.md) · [Samplesheet format](docs/samplesheet.md) · [Running the pipeline](docs/running-the-pipeline.md) · [Configuration](docs/configuration.md) · [Outputs](docs/outputs.md)
- Pipeline stages: [SpatialData build](docs/stages/spatialdata-build.md) · [Segmentation](docs/stages/segmentation.md) · [Enrichment](docs/stages/enrichment.md) · [QC](docs/stages/qc.md) · [Alignment](docs/stages/alignment.md) · [Comparison](docs/stages/comparison.md) · [Visualization](docs/stages/visualization.md) · [Squidpy clustering](docs/stages/clustering-squidpy.md) · [MapMyCells](docs/stages/mapmycells.md)
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
│   └── alignment/              # Optional Spateo cross-section registration
├── tests/                      # pytest suite, mirrors src/merxen/
├── docs/                       # Project documentation (start at docs/index.md)
├── notebooks/                  # Exploratory notebooks only
├── pyproject.toml              # Dependencies, merxen entry point, tool config
├── environment.yml             # Conda env (Python 3.12 + pip)
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

# Optional: enable Spateo-based section alignment
pip install spateo-release==1.1.1
pip install "anndata>=0.12.10"

# Install pre-commit hooks
pre-commit install
pre-commit install --hook-type pre-push
```

## Required environment variables

Copy `.env.example` to `.env` and fill in values:

```bash
cp .env.example .env
```

See [.env.example](.env.example) for required variables (ProSeg binary path, output root, etc.) and [docs/configuration.md](docs/configuration.md) for the full environment + Nextflow parameter reference.

## Running the pipeline

```bash
# Run via Nextflow with a samplesheet
nextflow run workflows/main.nf --samplesheet samples.csv --outdir ./results --proseg_binary /path/to/proseg
```

A template samplesheet is provided at [workflows/samplesheet.example.csv](workflows/samplesheet.example.csv). Copy and edit this file with your dataset-specific paths before running the workflow.

```bash
cp workflows/samplesheet.example.csv workflows/samplesheet.csv
```

The samplesheet points at raw platform folders with optional reusable SpatialData cache paths (`merscope_dir`, `merscope_spatialdata_path`, `xenium_dir`, `xenium_spatialdata_path`, plus per-platform channel, z-range, and voxel-layer settings). The full schema, validation rules, and worked examples are documented in [docs/samplesheet.md](docs/samplesheet.md). For Nextflow invocation options — resuming, stage-range runs, force rebuild, parameter overrides, cluster execution — see [docs/running-the-pipeline.md](docs/running-the-pipeline.md).

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

# Regenerate lockfile after changing dependencies
uv pip compile pyproject.toml --extra dev -o requirements.lock
```

Project standards (layout, dependencies, naming, type hints, docstrings, git workflow, commit message prefixes) are defined in [Agents.md](Agents.md). Day-to-day development mechanics — testing, pre-commit hooks, CI, debugging, adding a new pipeline stage — are in [docs/development.md](docs/development.md).

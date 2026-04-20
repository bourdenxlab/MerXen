# MerXen

Pre-processing, segmentation, and comparative analysis of paired MERSCOPE and Xenium spatial transcriptomics datasets.

## What it does

MerXen takes paired spatial transcriptomics datasets (one MERSCOPE, one Xenium per tissue section pair) and runs a standardised pipeline:

1. **SpatialData build** — Builds platform-specific SpatialData zarrs from raw MERSCOPE and Xenium output folders
2. **Cell segmentation** — Cellpose-SAM image-based segmentation followed by ProSeg transcript-based refinement
3. **Section alignment** — Registers paired adjacent sections to a common coordinate system *(planned)*
4. **Comparative analysis** — QC metrics, gene-level comparison, and visualisation across platforms

The workflow is orchestrated by Nextflow to process multiple sample pairs with logging and reproducibility.

## Setup

```bash
# Create conda environment
conda env create -f environment.yml
conda activate merxen

# Install pre-commit hooks
pre-commit install
pre-commit install --hook-type pre-push
```

## Required environment variables

Copy `.env.example` to `.env` and fill in values:

```bash
cp .env.example .env
```

See `.env.example` for required variables (ProSeg binary path, output root, etc.).

## Running the pipeline

```bash
# Run via Nextflow with a samplesheet
nextflow run workflows/main.nf --samplesheet samples.csv --outdir ./results --proseg_binary /path/to/proseg
```

A template samplesheet is provided at `workflows/samplesheet.example.csv`. Copy and
edit this file with your dataset-specific paths before running the workflow.

```bash
cp workflows/samplesheet.example.csv workflows/samplesheet.csv
```

The samplesheet now points at raw platform folders with optional reusable
SpatialData cache paths:

- `merscope_dir`: Raw MERSCOPE folder containing transcripts, boundaries, counts,
  metadata, and `images/`.
- `merscope_spatialdata_path`: Optional existing or desired reusable MERSCOPE
  SpatialData zarr. If it already exists, the build step is skipped unless
  `--force_spatialdata_build true` is supplied to Nextflow.
- `xenium_dir`: Raw Xenium export folder.
- `xenium_spatialdata_path`: Optional existing or desired reusable Xenium
  SpatialData zarr.

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

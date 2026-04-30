# MerXen Documentation

MerXen pre-processes, segments, and comparatively analyses paired MERSCOPE and
Xenium spatial transcriptomics datasets. For every pair of adjacent human
brain tissue sections — one run on Vizgen MERSCOPE and one on 10x Xenium with
the same 300-gene custom panel — MerXen coerces both platforms through an
identical downstream pipeline so they can be compared directly.

The pipeline is orchestrated by [Nextflow](https://www.nextflow.io/), with all
scientific logic living in the installable Python package `merxen`.

## Who these docs are for

- **Users** running the pipeline on their own data — start at
  [Getting started](getting-started.md), then [Samplesheet format](samplesheet.md)
  and [Running the pipeline](running-the-pipeline.md).
- **Developers** extending or debugging the Python package — start at
  [Pipeline architecture](pipeline.md), then [Python API](python-api.md) and
  [Development workflow](development.md).

## Table of contents

### Usage
- [Getting started](getting-started.md) — install, first run, smoke test.
- [Samplesheet format](samplesheet.md) — CSV schema and examples.
- [Running the pipeline](running-the-pipeline.md) — Nextflow invocation,
  resuming, caching.
- [Configuration](configuration.md) — environment variables, `nextflow.config`
  parameters, and Pydantic config models.
- [Outputs](outputs.md) — directory layout and artifacts produced by each stage.

### Pipeline stages
- [SpatialData build](stages/spatialdata-build.md)
- [Segmentation](stages/segmentation.md) (Cellpose-SAM + ProSeg)
- [Enrichment](stages/enrichment.md)
- [QC](stages/qc.md)
- [Section alignment](stages/alignment.md)
- [Comparison](stages/comparison.md)
- [Visualization](stages/visualization.md)

### Developer reference
- [Pipeline architecture](pipeline.md) — stage graph and data flow.
- [Python API overview](python-api.md) — subpackage map with key functions.
- [CLI reference](cli.md) — every `merxen` subcommand.
- [Development workflow](development.md) — testing, linting, contributing.

## Project files at a glance

| Path | Purpose |
|------|---------|
| [workflows/main.nf](../workflows/main.nf) | Nextflow DSL2 entry point |
| [workflows/nextflow.config](../workflows/nextflow.config) | Nextflow parameters and resource limits |
| [workflows/modules/](../workflows/modules/) | One `.nf` process per pipeline stage |
| [src/merxen/](../src/merxen/) | Installable Python package |
| [src/merxen/config.py](../src/merxen/config.py) | Pydantic config schemas |
| [src/merxen/cli/](../src/merxen/cli/) | Click CLI entry points |
| [tests/](../tests/) | pytest test suite, mirrors `src/merxen/` |
| [environment.yml](../environment.yml) | Conda env (Python 3.12 + pip) |
| [pyproject.toml](../pyproject.toml) | Dependencies, `merxen` entry point, tool config |
| [.env.example](../.env.example) | Required environment variables |

## Project standards

Contribution rules and code standards live in [Agents.md](../Agents.md). All
contributors (human and AI) are expected to follow them. [CLAUDE.md](../CLAUDE.md)
provides a short project overview and a pointer into `Agents.md`.

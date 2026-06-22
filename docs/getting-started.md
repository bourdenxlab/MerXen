# Getting started

This guide takes you from a fresh clone to a running pipeline.

## Prerequisites

- Linux host (the pipeline has been developed and tested on Linux; it will not
  run on macOS or Windows as-is).
- [Conda](https://docs.conda.io/) or [Miniforge](https://github.com/conda-forge/miniforge).
- [Nextflow](https://www.nextflow.io/docs/latest/getstarted.html) `>=23.04` on
  your `PATH`.
- Rust/Cargo on your `PATH`, or an existing
  [ProSeg](https://github.com/dcjones/proseg) binary in one of the configured
  search paths. MerXen searches `/usr/bin/proseg` and
  `/usr/local/bin/proseg` by default, then falls back to `command -v proseg`;
  if none exists, it installs ProSeg with Cargo before segmentation starts.
- Ample RAM. Defaults target a **75-CPU / 600 GB** machine; segmentation alone
  reserves 500 GB by default. See [Configuration](configuration.md#resource-limits)
  to dial this down.
- GPU strongly recommended for the Cellpose-SAM step (CPU fallback works but is
  very slow on full sections).

## 1. Install the Python environment

```bash
git clone <repo-url> MerXen
cd MerXen

conda env create -f environment.yml
conda activate merxen
```

The environment installs Python 3.12 and then `pip install -e ".[dev]"`, which
pulls every runtime and dev dependency from [pyproject.toml](../pyproject.toml)
and registers the `merxen` CLI entry point.

## 2. Install the pre-commit hooks

```bash
pre-commit install
pre-commit install --hook-type pre-push
```

The pre-commit hook runs `ruff` on every commit; the pre-push hook runs the
test suite. See [Development workflow](development.md).

## 3. Set environment variables

```bash
cp .env.example .env
```

Typically fill in:

| Variable | Description |
|----------|-------------|
| `MERXEN_OUTPUT_ROOT` | Directory to write pipeline outputs into. |
| `MERXEN_MAX_RAM_GB` | System RAM in GB the pipeline is allowed to use (default 600). |
| `MERXEN_PROSEG_INSTALL_PATH` | Optional Python-side default. Nextflow uses `proseg_install_path` from `workflows/nextflow.config`. |

`.env` is git-ignored. See [Configuration](configuration.md) for the full list
and how these are consumed.

## 4. Sanity check: run the tests

```bash
pytest                          # fast tests
pytest -m "not slow"            # alias for the same thing
pytest --run-slow               # include integration tests
```

If `pytest` passes, your Python install is healthy.

## 5. Create a samplesheet

Copy the template and fill it in with your own dataset paths:

```bash
cp workflows/samplesheet.example.csv workflows/samplesheet.csv
```

By default, each row pairs one MERSCOPE folder with one Xenium folder. For
single-platform runs, provide only the selected platform's source/cache columns
and pass `--analysis_mode merscope` or `--analysis_mode xenium`. See the full
schema in [Samplesheet format](samplesheet.md).

## 6. Run the pipeline

```bash
nextflow run workflows/main.nf \
    --samplesheet workflows/samplesheet.csv \
    --outdir ./results
```

Outputs land in `./results/<pair_id>/...`. Nextflow also writes an HTML
report, execution timeline, and trace TSV under `./results/nextflow/`.

More on invocation options (resume, caching, force rebuild) in
[Running the pipeline](running-the-pipeline.md). For an explanation of every
directory and file produced, see [Outputs](outputs.md).

## Troubleshooting

**`merxen: command not found`** — activate the conda env:
`conda activate merxen`. The `merxen` CLI is registered by
[pyproject.toml:44](../pyproject.toml#L44).

**ProSeg bootstrap failed** — the workflow could not find ProSeg in
`proseg_search_paths` and could not install it with `cargo install proseg`.
Install Cargo or edit `proseg_install_path` in
[workflows/nextflow.config](../workflows/nextflow.config). If the install path
is system-owned, the bootstrap step asks for `sudo` permission.

**`Missing required parameter: --samplesheet`** — you invoked `nextflow run`
without `--samplesheet`.

**Out of memory** — lower the per-process memory requests in
[workflows/nextflow.config](../workflows/nextflow.config) and set
`MERXEN_MAX_RAM_GB` accordingly.

**Cellpose GPU errors** — set `--cellpose_gpu false` to force CPU mode.

# CLAUDE.md

Read and follow all standards in `Agents.md` before doing any work in this repo.

## Project overview

This project is used for pre-processing and then analysing spatial transcriptomic datasets. The datasets have been generated in pairs to enable direct comparison between two spatial transcriptomic platforms: 10x Genomic's Xenium and Vizgen's MERSCOPE. For each pair of adjacent tissue sections from human brain sections, one was processed with Xenium and the other with MERSCOPE. The exact same custom panel of 300 genes was used for each technology. Several pairs of tissue sections have been processed. This project takes the data output from each platform and performs the following, matching all steps of analysis after the initial pre-processing steps to co-erce the data into the same formats between platforms:

- Performs cell segmentation from scratch using Cellpose-SAM image based segmentation, followed by ProSeg transcript based segmentation with the cellpose assignment as a prior.
- Aligns each pair of adjacent tissue sections so that they share a common coordinate system
- Peforms comparative analysis of each paired dataset

This workflow is achieved by using Nextflow to run different modules of pre-processing, segmentation, analysis etc. in a way that can be monitored per sample and scaled to many sample pairs with logging etc.

## Commands

```bash
# Run tests
pytest

# Run tests excluding slow markers
pytest -m "not slow"

# Lint (auto-fix) and format
ruff check . --fix
ruff format .

# Type check
mypy src/

# Regenerate lockfile after changing dependencies
uv pip compile pyproject.toml -o requirements.lock
```

## Workflow

- Run `ruff check` and `ruff format` before every commit.
- Run `pytest` before pushing. The pre-push hook enforces this.
- Do not install packages with `pip install <package>`. Add to `pyproject.toml` and regenerate the lockfile.
- Do not commit to `main` directly. Use a feature branch and open a PR.

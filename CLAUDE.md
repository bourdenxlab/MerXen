# CLAUDE.md

Read and follow all standards in `Agents.md` before doing any work in this repo.

## Project overview

This project is used for pre-processing and then analysing human or mouse spatial transcriptomic datasets. The datasets can be generated in pairs to enable direct comparison between two spatial transcriptomic platforms: 10x Genomics' Xenium and Vizgen's MERSCOPE. For each pair of adjacent brain sections, one is processed with Xenium and the other with MERSCOPE using the same custom gene panel. This project takes the data output from each platform and performs the following, matching all steps of analysis after the initial pre-processing steps to coerce the data into the same formats between platforms:

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
uv pip compile pyproject.toml --extra dev -o requirements/requirements.lock
# ...then refresh the lock checksum header in envs/environment.yml
python scripts/update_env_lock_hash.py
```

## Workflow

- Run `ruff check` and `ruff format` before every commit.
- Run `pytest` before pushing. The pre-push hook enforces this.
- Do not install packages with `pip install <package>`. Add to `pyproject.toml` and regenerate the lockfile.
- Do not commit to `main` directly. Use a feature branch and open a PR.
- Exception: the robust cell-type annotation effort (`docs/plans/robust-celltype-annotation-plan.md`) uses the long-lived integration branch `feature/robust-celltype-annotation`. Milestone branches `feature/rca-m<N>-<slug>` are cut from it and merged into it by PR. Keep it current by merging `main` into it; never rebase it. Only acceptance-gate PRs merge it into `main`. Data-integrity bugfixes still go to `main` first and are then merged into it.

## Privacy: no host paths, no sample names

- Nothing committed may contain an absolute path of a machine (storage roots, mounts, home
  directories, another host's directories) or the name of a sample: donor, section, slide, run or
  instrument ids taken from a samplesheet, and colleagues' user names. This covers code, configs,
  tests, docs, notebooks, file names, commit messages, branch names, PR titles and PR descriptions.
- Locations live in gitignored files: `.env` (Python, scripts, tests, notebooks; template
  `.env.example`), and the Nextflow site configs `workflows/conf/site.config` and
  `workflows/conf/dwight.site.config` (templates `*.example`). Python reads `.env` through
  `merxen.local_config`; a missing value is an error, never a built-in default.
- Samples are named by aliases (HD1-HD4 paired human donors, HD1_IF / HD3_IF IF samples, HM1-HM4
  new-panel human MERSCOPE donors, MS1 / MS2 mouse sections; docs/local-configuration.md). The
  gitignored `.aliases.tsv` maps them to the real names. Code that reads a real name from data
  translates it with `merxen.privacy.to_alias`; paths to existing data are built with
  `merxen.privacy.to_real`.
- Docs write locations as `.env` variables (`$MERXEN_EVIDENCE_ROOT/...`). Inside a verbatim
  quotation a name or path is replaced by its alias or variable in square brackets.
- The hooks enforce this (`pre-commit install`). Before `gh pr create` or `gh pr edit`, pipe the
  title and body through `python3 scripts/privacy/check_private_text.py --stdin`.
- Not sample names: public references (WHB, WMB, SEA-AD, MERFISH-638850 and its sections, the ABC
  atlas, public 10x and Vizgen datasets) and the `.vzg2` file format.

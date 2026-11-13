# Versioning and releases

MerXen releases follow [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).
A release is numbered `MAJOR.MINOR.PATCH`, tagged `vMAJOR.MINOR.PATCH` on `main`
and published as a [GitHub release](https://github.com/bourdenxlab/MerXen/releases)
whose notes are its section of [CHANGELOG.md](../CHANGELOG.md).

- **MAJOR** goes up when a run that worked on the previous release needs the
  user to change something: a parameter, samplesheet column, output or command
  was removed or renamed, or a changed default alters results in a way no
  setting restores.
- **MINOR** goes up for new features and for changed behaviour that a setting
  can switch back.
- **PATCH** goes up for bug fixes, even when they change outputs (the entry
  says which), and for changes users do not see.

## Which version did I run?

| Where | What it shows |
| --- | --- |
| Launch log | `MerXen 1.2.0 (checkout v1.2.0)`: the release, then the checkout it ran from. |
| `<outdir>/nextflow/merxen_versions.jsonl` | One JSON line per launch into that results folder: release, `git_describe`, Nextflow version, run name, session, `resume`, profile and start time. A folder resumed across releases lists every one of them. `-preview` writes nothing. |
| `<outdir>/nextflow/report.html` | Nextflow's execution report, which names the manifest version. |
| `merxen --version` | The release of the installed Python package. |
| `merxen_version` inside outputs | Annotation provenance and `report_run.json`, the manual-registration manifests and the OME-TIFF `Creator` tag stamp the package release. |

Every entry script records the version: `main.nf` and the standalone
workflows (`manual_registration_prepare.nf`, `manual_registration_register.nf`,
`pciseq_resume.nf`, `prepare_hitl_viewer.nf`), each into its own `--outdir`.
The record also has `revision` and `commit_id` fields. They stay null, because
the pipeline runs from a checkout: `nextflow run bourdenxlab/MerXen -r v1.2.0`
does not work yet, since the entry scripts live in `workflows/`.

`git_describe` tells a release apart from the code around it:

- `v1.2.0` is the release itself.
- `v1.2.0-3-gabc1234` is three commits after it (commit `abc1234`), so not a
  release.
- A `-dirty` suffix means the checkout had uncommitted edits.
- A bare commit hash means the checkout has no release tag (fetch the tags
  with `git fetch --tags`).

## Running a release

Check the release out and run it as usual ([running the pipeline](running-the-pipeline.md)):

```bash
git fetch --tags
git checkout v1.2.0
```

Use a separate worktree to keep a release next to your working checkout:

```bash
git worktree add ../MerXen-v1.2.0 v1.2.0
```

Rebuild the conda environments or images when the release's lockfile differs
from the one you ran before. The environment hash already follows the lockfile
([development](development.md#dependency-management)).

## What the version covers

The version applies to what users run or read:

- **Nextflow parameters**: their names, accepted values, defaults and meaning
  ([configuration](configuration.md)), and the profiles.
- **Samplesheet columns**: their names and accepted values ([samplesheet](samplesheet.md)).
- **Outputs**: folder layout, file names and formats, SpatialData element and
  table keys, and the documented columns ([outputs](outputs.md)).
- **The `merxen` command line**: commands and their options ([CLI](cli.md)).
- **The documented Python API** ([Python API](python-api.md)).
- **Local configuration**: the `.env` variables and site-config params a launch
  needs ([local configuration](local-configuration.md)).
- **Process names**, where the docs tell users to select them with `withName`.

Undocumented internals are not covered: module-private functions, scripts under
`scripts/annotation_dev/`, `scripts/acceptance/` and `scripts/ci/`, notebooks,
tests and process names nobody is told to target. They can change in any
release. Their changes still get a changelog entry when users would notice.

## Choosing the version

Choose the version from the [Unreleased] entries: the largest change decides.

| Change | Version |
| --- | --- |
| Removing or renaming a parameter, samplesheet column, profile, command, option, output file or folder, table key or documented column | major |
| Changing a parameter's meaning or accepted values so that an old setting fails or does something else | major |
| Changing an output format so that existing readers break | major |
| A new required input, `.env` variable or site-config value | major |
| Dropping a platform, species, Nextflow version or Python version | major |
| A default that changes results with no setting to get the old results back | major |
| A new stage, parameter, column, output, platform or species that existing runs do not have to use | minor |
| A default that changes results when a setting restores the old behaviour. The entry names the setting. | minor |
| Deprecating something: it still works, warns, and is removed in the next major | minor |
| A bug fix. The entry says which outputs change. | patch |
| Faster or lighter execution with the same outputs, and documentation | patch |

Two cases come up often:

- **Changed results.** A run on the same inputs and settings that gives
  different numbers always gets an entry, whatever the version. The entry says
  which outputs change and how to keep the old ones.
- **Breaking changes.** Prefer deprecating first: keep the old name working with
  a warning for one minor release, then remove it in the next major.

Before 1.0.0 the version was not tied to releases: `0.1.0` and `0.2.0` were
never tagged, and runs before 1.0.0 have no `merxen_versions.jsonl`. Outputs
stamped `merxen_version` 0.1.0 or 0.2.0 do not identify a commit.

## Changelog entries

[CHANGELOG.md](../CHANGELOG.md) follows [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/).

- **Every PR adds an entry.** A pull request that changes what users run adds
  its entry under `## [Unreleased]` in the same PR. The `Changelog` CI job
  fails a PR that changes `src/`, `workflows/`, `envs/`, `containers/`,
  `assets/` or `requirements/` without touching `CHANGELOG.md`. Label the PR
  `skip-changelog` when users would notice nothing, such as a refactor with
  identical outputs or a test-only change.
- **Change types.** File the entry under one of these `###` headings: `Added`,
  `Changed`, `Deprecated`, `Removed`, `Fixed` or `Security`. A release can also
  have `Highlights` (a short summary at the top), `Upgrading` (what users must
  do) and `Acceptance` (validation records). A `####` heading can group the
  bullets of a larger change.
- **Write for users.** Say what changed, who it affects and what to do about it.
  Give the setting that restores the old behaviour, and link the docs page.

The `Changelog` job also checks that the release version agrees across
`pyproject.toml`, `src/merxen/__init__.py` and the Nextflow manifest, and that
`CHANGELOG.md` is well formed. Run the same checks locally:

```bash
python3 scripts/ci/changelog.py versions
python3 scripts/ci/changelog.py validate
python3 scripts/ci/changelog.py check-pr --base origin/main
```

## Making a release

1. **Branch from `main`.**

   ```bash
   git switch main && git pull
   git switch -c release/v1.3.0
   ```

2. **Edit the [Unreleased] entries.** Merge entries that duplicate each other.
   Add `### Highlights` for a minor or major release, and `### Upgrading` when
   users must change anything. Choose the version as described in
   [Choosing the version](#choosing-the-version).

3. **Set the version.** `bump-my-version` writes it into `pyproject.toml`,
   `src/merxen/__init__.py` and the Nextflow manifest. It renames
   `## [Unreleased]` to `## [1.3.0] - <today>` above a new empty
   `## [Unreleased]`, and updates the compare links at the bottom of
   `CHANGELOG.md`. Run it from the project's dev environment, where the dev
   extra installs it (or `uvx bump-my-version@1.3.0`). Preview it first with
   `--dry-run -v`.

   ```bash
   bump-my-version bump minor   # or major / patch
   python3 scripts/ci/changelog.py versions
   python3 scripts/ci/changelog.py validate
   python3 scripts/ci/changelog.py notes 1.3.0   # the GitHub release notes
   ```

4. **Open the release PR.** Commit as `[release]: v1.3.0` and open a PR with
   that title and the release notes as its body. Once its CI passes, squash-merge
   it. If another PR reached `main` first, merge `main` into the release branch
   and move that PR's entries into the release section. If the merge happens on a
   later day, fix the date in the heading first.

5. **Check the release.** The [release workflow](../.github/workflows/release.yml)
   runs when the merge reaches `main`. It checks that the versions agree, that
   the version is newer than every earlier `v*` tag and that `CHANGELOG.md` has
   its section. It then tags the merge commit `v1.3.0` and publishes the GitHub
   release with the section as its notes:

   ```bash
   gh release view v1.3.0
   ```

   The workflow does nothing for a push whose version is already tagged. If the
   publish failed, re-run it from the Actions tab (`workflow_dispatch`). It tags
   the commit that set the version, not the newest `main`.

**Merge method.** Squash-merge release PRs as usual. The release workflow tags
the first-parent commit on `main` that set the version, so a merge commit also
works, but a rebase merge would tag the bump commit only.

**Pre-releases** (`1.3.0-rc.1`) are not used yet: the tooling accepts only
`X.Y.Z`. To test unreleased work, run a branch or commit; its version record
shows `git_describe`.

**Patch releases of an older major** have no automation yet. Branch from the
old tag, bump, tag the branch head and run `gh release create` by hand.

## Repository setup

These GitHub settings are needed once, by a repository admin:

- **The `skip-changelog` label**, which exempts a PR from the changelog check:
  `gh label create skip-changelog --description "PR needs no CHANGELOG entry"`.
- **Workflow permissions** that let the release workflow create tags and
  releases. It asks for `contents: write` itself; an organisation policy that
  caps the token at read-only blocks it.
- **Optional:** branch protection on `main` that requires the `lint`, `test`
  and `Changelog` checks, so that the checks block a merge instead of only
  warning.

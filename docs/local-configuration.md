# Local configuration and privacy

Nothing committed to this repository may contain an absolute path of a
machine or the name of a sample (CLAUDE.md, [Agents.md](../Agents.md)). This
page describes where locations and real names live instead, how code reads
them, and the hooks that keep them out of commits.

## The private files

All four kinds are gitignored and have a tracked template holding
placeholders only.

| Private file | Template | Format | Read by |
|---|---|---|---|
| `.env` (repository root) | [`.env.example`](../.env.example) | `KEY=VALUE` (below) | `merxen.local_config`, shell scripts (`. "$MERXEN_ENV_FILE"`), the hooks (exact values) |
| `.aliases.tsv` (repository root) | [`.aliases.example.tsv`](../.aliases.example.tsv) | TSV, 7 columns | `merxen.privacy`, the hooks, `scripts/privacy/rewrite_text.py` |
| `workflows/conf/site.config` | `workflows/conf/site.config.example` | Nextflow config (params) | `workflows/nextflow.config` (every profile) |
| `workflows/conf/dwight.site.config` | `workflows/conf/dwight.site.config.example` | Nextflow config (params, `env`) | `workflows/conf/dwight.config` (standard, dwight, local) |

A future host adds `workflows/conf/<host>.site.config` beside its own
profile file. The project keeps one canonical alias table, group-readable
and outside every repository, so that every developer and every pipeline task
sees the same aliases; `.aliases.tsv` in a checkout is a link to it, or
`MERXEN_ALIAS_FILE` names it.

### `.env` format

The same file is read by Python, by bash and by the hooks, so the format is
strict and has no interpolation anywhere:

- `KEY=VALUE`, one per line; a line starting with `#` is a comment (free
  text); blank lines are ignored.
- A value with spaces is double-quoted. No `$`, backticks, backslashes,
  single quotes, `~` or `export`. `merxen.local_config` raises
  `LocalConfigError` on a line that breaks this (the message names the line
  and key, never the value).
- Every location variable starts with `MERXEN_` and is listed in the location
  section of `.env.example` (`merxen.local_config.KNOWN`; a test keeps the two
  equal). Runtime knobs (`MERXEN_TMPDIR`, `MERXEN_ANNOTATION_*_N_PROCESSORS`,
  `MERXEN_CI_*`, ...) are not locations: they are read from the process
  environment only and never go into `.env`.
- A variable that is commented out is unset on purpose: code that needs it
  stops with an error naming it, and tests that need it skip.
- Shell scripts source the file without `set -a`; nothing exports it
  wholesale.

### Discovery (checkouts, worktrees and code exports)

A linked git worktree and a `git archive` code export contain no gitignored
file. Python, the hooks and the shell scripts look for each private file in
the same order:

1. An explicit variable: `MERXEN_ENV_FILE`, `MERXEN_ALIAS_FILE`,
   `MERXEN_SITE_CONFIG`, `MERXEN_DWIGHT_SITE_CONFIG`. The value `none` (or the
   null device) means "absent on purpose". An empty value means unset, and
   discovery goes on. A value naming a file that does not exist is an error.
2. The checkout's own file.
3. Python and the hooks only: in a linked worktree, the main checkout's file
   (read from the `.git` link file; no git call).
4. Otherwise absent.

`MERXEN_ALIAS_FILE` is read from the environment, then from `.env`; without
it the table beside the `.env` in use is tried before the checkout. `.env`
holds absolute pointers to the other three private files, so one variable,
`MERXEN_ENV_FILE`, bootstraps a code export. Exporting
`MERXEN_ENV_FILE=<main checkout>/.env` in the shell profile makes every export
and worktree find the private files.

## Reading locations in Python

```python
from merxen import local_config

evidence = local_config.require_path("MERXEN_EVIDENCE_ROOT", "reads recorded runs")
```

- `get(name)`: process environment (non-empty) first, then `.env`, else None.
- `path(name)` / `require_path(name, purpose)`: a `Path`;
  `require_path` raises `LocalConfigError` ("MERXEN_EVIDENCE_ROOT is not set:
  add it to .env (template .env.example) or export it") and never falls back
  to a built-in default. Neither checks that the path exists.
- `add_location_argument(parser, "--evidence-root", "MERXEN_EVIDENCE_ROOT",
  help)`: an argparse option whose default is the variable, required when it
  is unset (the help says `default: $MERXEN_EVIDENCE_ROOT`).
- `MERXEN_CZI_PYTHON` (the python of the isolated CZI runtime used by
  `merxen.io.czi`, `merxen convert-czi` and the CZI tests) and
  `MERXEN_MANUAL_REGISTRATION_TEST_DIR` (the folder of the Zeiss test pairs) are
  locations like the others: read them through `merxen.local_config` (the
  environment first, then `.env`). Nextflow runs take the CZI runtime from the
  `manual_registration_czi_python` param of the site config instead
  ([manual landmark registration](stages/manual-registration.md)).
- Resolve locations at the point of use, never at import, so `--help`, test
  imports and CI work without a `.env`. Package CLIs under `src/merxen/cli`
  take every path from Nextflow params and do not use the loader.

## Sample aliases

Tracked code, data, tests and docs name samples by aliases only:

| Alias | Kind |
|---|---|
| HD1, HD2 | human paired donors (Xenium and MERSCOPE), development pairs |
| HD3, HD4 | human paired donors, held-out pairs |
| HD1_IF, HD3_IF | human IF samples (MERSCOPE only), derived from the donor alias |
| HM1, HM2, HM3, HM4 | new-panel human MERSCOPE donors |
| MS1, MS2 | mouse sections |
| LCM1, LCM2 | Zeiss Axioscan CZI pairs (a Xenium section as reference and a serial LCM section as moving image), manual-registration test data |

Aliases are stable: never renumbered, a new sample takes the next free number,
and a new sample of a donor that already has an alias gets a derived alias
(`HD1_<suffix>`, as `HD1_IF`). New rows are appended to the canonical table
only. The table also holds deny-only rows (instrument, slide, run stamp, user
and host directory names) that have no alias and are only refused by the
hooks.

Each row holds one real form of an alias: `pair_id` (the samplesheet pair
id), `name` (a display nickname), `number` (a bare donor number),
`bulk_column`, `merscope_run`, `xenium_output`, or a `variant` spelling. A
form is unique per alias, except variants and deny rows.

Rules for code that meets real names in data (`merxen.privacy`):

- A value read from outside the repository (samplesheet pair ids, directory
  names, keys of files written before the cleanup, `KEY=PATH` arguments)
  passes through `to_alias` before it is compared with a tracked constant or
  used as a key.
- A path to existing data built from an alias uses `to_real(alias, form=...)`
  with the form that path family was named by: `pair_id` for
  samplesheet-keyed results, `name` for directories named by a nickname.
- An identifier written to an output that used to be a hard-coded real name
  goes through `display()` (the `name` form, else `pair_id`), so outputs on a
  host with the table stay as they were.
- Case follows the text: inside a lower-case identifier the alias is lower
  case (`mouse_ms1_500`); a standalone id maps to the canonical alias.
- Without a table `to_alias` and `to_real` raise, unless the table is disabled
  on purpose with `MERXEN_ALIAS_FILE=none` (the tests do this, so they see
  aliases only); `display()` then returns its input with one warning. Bare
  donor numbers and deny rows are never applied to runtime data.

## Docs: paths become variables

Docs write locations as `.env` variables, for example
`$MERXEN_EVIDENCE_ROOT/exp/...`, and samples by alias. The shorthand "`$A` =
`$MERXEN_EVIDENCE_ROOT`" of the annotation plan stays. The one-time rewrite is
deterministic:

```bash
python3 scripts/privacy/rewrite_text.py --paths --names \
    --report "$PRIVATE_DIR/rewrite_report.tsv" docs/some-page.md
```

It replaces every location value of the resolved `.env` (longest first,
followed by a path boundary; ties go to the earlier variable of `KNOWN`, and
the launch guard's refused prefix is never used) by `$NAME`, then every real
name by its alias or placeholder. Its report holds the real text, so it is
refused inside any git working tree.

Verbatim quotations keep their words: a name or path inside a quotation is
replaced by its alias or variable in square brackets, `[MS1]`,
`[$MERXEN_MS1_OWNER_ROOT]`. A name inside an existing editorial bracket is
replaced without new brackets. The rewrite brackets every substitution on a
blockquote line and flags inline quotations and a created `](` for the manual
pass. A document with such a substitution says once, at its top: "Inside
quotations, names and paths in square brackets are aliases or `.env`
variables standing for the originals (privacy rule, CLAUDE.md); the originals
are kept in the private records."

## The hooks

`scripts/privacy/check_private_text.py` (stdlib only, run by the system
`python3`) applies four rules:

1. Generic host paths: an absolute path under one of the host roots (the
   words srv, media, home, mnt, Users, nfsdata, scratch, Volumes), data
   followed by a separator, the coding assistant's session scratch
   directories under tmp, and a Windows user directory. Always applied,
   CI included.
2. Exact private values: every absolute value of `.env` and every quoted
   absolute string of the site configs (when found), except values under
   `/path/to/`, `/usr/`, `/proc/` and `/dev/`.
3. Generic donor ids: `P` followed by four digits, except the synthetic
   fixture ids `P0001`, `P0002`, `P9999`. Always applied.
4. Real names: every row of the alias table (when found; literal rows in any
   case, although `to_alias` translates data only on an exact match). Without
   a table the hook prints `privacy: no alias table (.aliases.tsv); real-name check
   skipped` and still applies rules 1-3.

Base64 payload lines and notebook image output lines are skipped. Binary files
are searched for rules 2 and 4 in their printable runs. File and branch names
are checked like content, and the private files are refused by name whatever
their content.

| Mode | Stage | Checks |
|---|---|---|
| `--staged` | pre-commit, pre-merge-commit | added lines and new file names of the index (only added lines, so a partial cleanup of a file can be committed) |
| `--commit-msg FILE` | commit-msg | the message, without git's comment lines |
| `--pre-push` | pre-push | branch names, then the message and added lines of every pushed commit that no remote-tracking ref holds (a merge against its first parent); catches `--no-verify` commits |
| `--stdin` | by hand, CI | PR title and description |
| `--tree [ROOT]` | tests, progress reports | every tracked file (or every file of an export) |

Output lines name the place, the rule and the alias, placeholder, root word
or variable, never the matched text, because hook output gets pasted into
commits, PRs and public CI logs. `--show-match` adds the matched text for
interactive local use only. `--report FILE` writes the lines to a file outside
every git working tree. The exit status is 0 when clean, 1 on hits, 2 on a
configuration error (a malformed `.env`, a variable naming a missing file).

Install the hooks once per clone; the hook types (pre-commit,
pre-merge-commit, commit-msg, pre-push) come from
`default_install_hook_types` in `.pre-commit-config.yaml`:

```bash
pre-commit install
```

Hooks live in the shared git directory, so every worktree gets them; each
branch's own `.pre-commit-config.yaml` decides which hooks run. An existing
local `commit-msg` hook keeps running first (pre-commit's migration mode).

Before `gh pr create` or `gh pr edit`, check the title and body:

```bash
printf '%s\n\n%s\n' "$TITLE" "$BODY" | python3 scripts/privacy/check_private_text.py --stdin
```

The Privacy workflow on GitHub checks the PR branch name, title, description
and commit messages on every change, edits included (rules 1 and 3: CI has no
table). The branch name is checked there too because pre-commit runs the push
hook only for the first pushed ref that brings new commits: a new branch on
commits already pushed, or a second ref of one push, is not checked locally.

### What may stay

`python3 scripts/privacy/check_private_text.py --list-exceptions` prints the
same list as `merxen.privacy.EXCEPTIONS`. None needs a code path that lets a
hit through; each lies outside the rules by construction.

- OS tool and system paths: `/usr/bin/proseg`, `/usr/local/bin/proseg`,
  `/usr/bin/time`, `/proc`, `/dev/null`, `/opt/...` inside container images.
- The generic GPU lock file `/tmp/merxen-dwight-gpu.lock`.
- The host profile name `dwight` and its file names.
- The disk nicknames SSD1, SSD2 and SS3 in prose (their paths are `.env`
  variables).
- The copyright holder of LICENSE and README, and the project's GitHub
  organisation URL and CI badge.
- Placeholder paths under `/path/to/`.
- The public `.vzg2` archive format name, and the format files listed in the
  MS2 row's `skip_paths`.
- The synthetic donor ids `P0001`, `P0002`, `P9999`.

Not sample names either: public references (WHB, WMB, SEA-AD, MERFISH-638850
and its sections, the ABC atlas, public 10x and Vizgen datasets).

## Tests

`tests/conftest.py` makes every test hermetic: it captures the real locations
and the alias table once, then hides them (`MERXEN_ENV_FILE=none`,
`MERXEN_ALIAS_FILE=none`, both site-config variables set to the null device,
every `KNOWN` variable removed), for the whole session and again for each
test. Tests therefore see what GitHub CI sees, and so do the subprocesses they
start. A test that reads real data asks for a fixture:

- `real_path("MERXEN_EVIDENCE_ROOT")`: the path, also set for the test; skips
  when the variable is unset, when its value lies under
  `MERXEN_REFUSED_PATH_PREFIX` (no test reads the failing disk), or when the
  path does not exist.
- `real_aliases`: re-enables the real alias table for one test; skips without
  one.
- `real_env_file`: the real `.env`; skips without one.

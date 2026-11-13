#!/usr/bin/env python3
"""Release versions and CHANGELOG.md checks (docs/versioning.md).

``versions``  checks that the release version agrees in pyproject.toml, the
              bump-my-version config, src/merxen/__init__.py and the Nextflow
              manifest, is a plain ``X.Y.Z`` and, once a release exists, has its
              CHANGELOG section; prints the version. ``--newer-than-tags``
              also requires it to be newer than every given ``vX.Y.Z`` tag
              (the release workflow passes the repository's tags).
``validate``  checks the layout of CHANGELOG.md: an [Unreleased] section first,
              release sections ``## [X.Y.Z] - YYYY-MM-DD`` newest first, known
              ``###`` change types and a link for every section.
``notes``     prints one release's section body (the GitHub release notes).
``check-pr``  fails when a pull request changes what users run (code, workflows,
              environments, containers, assets, the dependency lock) without touching
              CHANGELOG.md.

Stdlib only: the CI jobs run it without installing the package.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

SEMVER = re.compile(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)")
RELEASE_HEADING = re.compile(
    r"## \[(?P<version>[^\]]+)\] - (?P<date>\d{4}-\d{2}-\d{2})"
)
UNRELEASED_HEADING = "## [Unreleased]"
LINK_DEFINITION = re.compile(r"\[(?P<label>[^\]]+)\]: (?P<url>\S+)")

# Keep a Changelog's change types, plus the release summary, upgrade notes and
# the acceptance record this project keeps.
CHANGE_TYPES = (
    "Highlights",
    "Upgrading",
    "Added",
    "Changed",
    "Deprecated",
    "Removed",
    "Fixed",
    "Security",
    "Acceptance",
)

# Paths whose change reaches users: a PR touching one needs a changelog entry.
USER_FACING_PREFIXES = (
    "src/",
    "workflows/",
    "envs/",
    "containers/",
    "assets/",
    "requirements/",
)


@dataclass(frozen=True)
class Release:
    """One ``## [X.Y.Z] - date`` section of the changelog."""

    version: str
    date: str
    line: int
    body: str


def parse_version(text: str) -> tuple[int, int, int]:
    """``(major, minor, patch)`` of a plain ``X.Y.Z`` version."""
    match = SEMVER.fullmatch(text)
    if match is None:
        raise ValueError(f"not a MAJOR.MINOR.PATCH version: {text!r}")
    major, minor, patch = (int(part) for part in match.groups())
    return major, minor, patch


def _structural_lines(lines: Sequence[str]) -> list[bool]:
    """Per line: True outside fenced code blocks (where headings count)."""
    result = []
    fence: str | None = None
    for line in lines:
        stripped = line.lstrip()
        marker = stripped[:3] if stripped[:3] in ("```", "~~~") else None
        if fence is None and marker is not None:
            fence = marker
            result.append(False)
        elif fence is not None:
            if marker == fence:
                fence = None
            result.append(False)
        else:
            result.append(True)
    return result


def _trailing_links_start(lines: Sequence[str]) -> int:
    """Index where the closing block of link definitions starts."""
    start = len(lines)
    for index in range(len(lines) - 1, -1, -1):
        line = lines[index].rstrip()
        if not line:
            continue
        if LINK_DEFINITION.fullmatch(line):
            start = index
        else:
            break
    return start


def releases(changelog: str) -> list[Release]:
    """The release sections of a changelog, in file order.

    A section runs to the next ``## `` heading outside a code fence, or to the
    link definitions that close the file. Link definitions inside a section are
    part of it.
    """
    lines = changelog.splitlines()
    structural = _structural_lines(lines)
    links_start = _trailing_links_start(lines)
    found: list[tuple[str, str, int]] = []
    for index, line in enumerate(lines[:links_start]):
        match = RELEASE_HEADING.fullmatch(line.rstrip())
        if match and structural[index]:
            found.append((match["version"], match["date"], index))
    result = []
    for version, date, start in found:
        end = links_start
        for index in range(start + 1, links_start):
            if structural[index] and lines[index].startswith("## "):
                end = index
                break
        body = "\n".join(lines[start + 1 : end]).strip("\n")
        result.append(Release(version, date, start + 1, body))
    return result


def validate(changelog: str) -> list[str]:
    """Layout problems of CHANGELOG.md (empty when it is well formed)."""
    problems: list[str] = []
    lines = changelog.splitlines()
    structural = _structural_lines(lines)
    level_two = [
        (i, line.rstrip())
        for i, line in enumerate(lines)
        if structural[i] and line.startswith("## ")
    ]
    if not level_two or level_two[0][1] != UNRELEASED_HEADING:
        problems.append(f"the first section must be '{UNRELEASED_HEADING}'")
    if sum(1 for _, line in level_two if line == UNRELEASED_HEADING) > 1:
        problems.append(f"'{UNRELEASED_HEADING}' appears more than once")
    for index, line in level_two:
        if line != UNRELEASED_HEADING and not RELEASE_HEADING.fullmatch(line):
            problems.append(
                f"line {index + 1}: '{line}' is neither '{UNRELEASED_HEADING}' "
                "nor '## [X.Y.Z] - YYYY-MM-DD'"
            )
    previous: tuple[int, int, int] | None = None
    seen: set[str] = set()
    for release in releases(changelog):
        try:
            current = parse_version(release.version)
        except ValueError as error:
            problems.append(f"line {release.line}: {error}")
            continue
        if release.version in seen:
            problems.append(f"line {release.line}: {release.version} appears twice")
        seen.add(release.version)
        if previous is not None and current >= previous:
            problems.append(
                f"line {release.line}: {release.version} is not older than the "
                "release above it (newest first)"
            )
        previous = current
        if not release.body.strip():
            problems.append(f"line {release.line}: release {release.version} is empty")
    for index, line in enumerate(lines):
        if structural[index] and line.startswith("### "):
            kind = line[4:].strip()
            if kind not in CHANGE_TYPES:
                problems.append(
                    f"line {index + 1}: change type '{kind}' is not one of "
                    f"{', '.join(CHANGE_TYPES)}"
                )
    links = {
        match["label"]
        for index, line in enumerate(lines)
        if structural[index]
        and (match := LINK_DEFINITION.fullmatch(line.rstrip())) is not None
    }
    if "Unreleased" not in links:
        problems.append("no link definition for [Unreleased]")
    for release in releases(changelog):
        if release.version not in links:
            problems.append(f"no link definition for [{release.version}]")
    return problems


# A markdown link target that is relative to the repository: not a URL, not an
# in-page anchor and not root-relative.
RELATIVE_LINK = re.compile(r"\]\((?![a-zA-Z][a-zA-Z0-9+.-]*:|#|/)([^)\s]+)\)")


def absolute_links(text: str, base_url: str) -> str:
    """Point repository-relative link targets at ``base_url`` (e.g. a tag tree).

    A GitHub release body is not rendered against the repository, so the
    relative ``docs/...`` links of CHANGELOG.md would break there.
    """
    base = base_url.rstrip("/") + "/"
    return RELATIVE_LINK.sub(lambda match: f"]({base}{match[1]})", text)


def notes(changelog: str, version: str, link_base: str | None = None) -> str:
    """The body of one release's section, links made absolute with ``link_base``."""
    for release in releases(changelog):
        if release.version == version:
            body = release.body
            return absolute_links(body, link_base) if link_base else body
    raise KeyError(f"CHANGELOG.md has no section '## [{version}] - <date>'")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def declared_versions(root: Path = REPO_ROOT) -> dict[str, str | None]:
    """The release version as each file declares it (None when not found)."""
    pyproject = tomllib.loads(_read(root / "pyproject.toml"))
    init = re.search(
        r'^__version__ = "([^"]+)"$', _read(root / "src/merxen/__init__.py"), re.M
    )
    nextflow = _read(root / "workflows/nextflow.config")
    manifest = re.search(r"^manifest \{(?P<body>.*?)^\}", nextflow, re.M | re.S)
    manifest_version = (
        re.search(r"^\s*version = '([^']+)'", manifest["body"], re.M)
        if manifest
        else None
    )
    return {
        "pyproject.toml [project] version": pyproject.get("project", {}).get("version"),
        "pyproject.toml [tool.bumpversion] current_version": pyproject.get("tool", {})
        .get("bumpversion", {})
        .get("current_version"),
        "src/merxen/__init__.py __version__": init.group(1) if init else None,
        "workflows/nextflow.config manifest.version": (
            manifest_version.group(1) if manifest_version else None
        ),
    }


def check_versions(
    root: Path = REPO_ROOT, tags: Sequence[str] = ()
) -> tuple[str | None, list[str]]:
    """The agreed release version and the problems found with it."""
    declared = declared_versions(root)
    problems = [f"{where}: not found" for where, value in declared.items() if not value]
    values = {value for value in declared.values() if value}
    if len(values) > 1:
        listing = "; ".join(f"{where} = {value}" for where, value in declared.items())
        problems.append(f"the release version differs between files: {listing}")
        return None, problems
    if not values:
        return None, problems
    version = values.pop()
    try:
        current = parse_version(version)
    except ValueError as error:
        return None, [*problems, str(error)]
    changelog = _read(root / "CHANGELOG.md")
    released = [release.version for release in releases(changelog)]
    if released and version != released[0]:
        problems.append(
            f"version {version} is not the newest CHANGELOG.md release ({released[0]})"
        )
    for tag in tags:
        if not tag.startswith("v") or SEMVER.fullmatch(tag[1:]) is None:
            continue
        if parse_version(tag[1:]) >= current and tag != f"v{version}":
            problems.append(f"version {version} is not newer than the tag {tag}")
    return version, problems


def changed_files(base: str, head: str = "HEAD") -> list[str]:
    """Files a branch changed since it left ``base`` (``git diff base...head``)."""
    result = subprocess.run(
        # --no-renames: a file moved out of a user-facing folder lists its old
        # path too.
        ["git", "diff", "--name-only", "--no-renames", f"{base}...{head}"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def needs_entry(paths: Sequence[str]) -> list[str]:
    """The changed paths that call for a changelog entry."""
    return [path for path in paths if path.startswith(USER_FACING_PREFIXES)]


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    versions_parser = commands.add_parser(
        "versions", help="check and print the version"
    )
    versions_parser.add_argument(
        "--newer-than-tags",
        nargs="*",
        default=[],
        metavar="TAG",
        help="also require the version to be newer than these vX.Y.Z tags",
    )
    commands.add_parser("validate", help="check the layout of CHANGELOG.md")
    notes_parser = commands.add_parser("notes", help="print one release's notes")
    notes_parser.add_argument("version")
    notes_parser.add_argument(
        "--link-base",
        help="make relative links absolute under this URL "
        "(the release workflow passes the tag's tree)",
    )
    pr_parser = commands.add_parser("check-pr", help="require a changelog entry")
    pr_parser.add_argument("--base", required=True, help="base ref, e.g. origin/main")
    pr_parser.add_argument("--head", default="HEAD")
    args = parser.parse_args(argv)

    if args.command == "versions":
        version, problems = check_versions(tags=args.newer_than_tags)
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        if problems or version is None:
            return 1
        print(version)
        return 0
    if args.command == "validate":
        problems = validate(_read(REPO_ROOT / "CHANGELOG.md"))
        for problem in problems:
            print(f"error: CHANGELOG.md {problem}", file=sys.stderr)
        return 1 if problems else 0
    if args.command == "notes":
        try:
            print(
                notes(_read(REPO_ROOT / "CHANGELOG.md"), args.version, args.link_base)
            )
        except KeyError as error:
            print(f"error: {error.args[0]}", file=sys.stderr)
            return 1
        return 0
    changed = changed_files(args.base, args.head)
    user_facing = needs_entry(changed)
    if user_facing and "CHANGELOG.md" not in changed:
        listing = "\n".join(f"  {path}" for path in user_facing[:20])
        more = len(user_facing) - 20
        print(
            "error: this branch changes what users run but adds no CHANGELOG.md "
            "entry. Add one under '## [Unreleased]' (docs/versioning.md), or label "
            "the PR 'skip-changelog' if users would notice nothing.\n"
            f"{listing}" + (f"\n  ... and {more} more" if more > 0 else ""),
            file=sys.stderr,
        )
        return 1
    print(
        "changelog entry present"
        if user_facing
        else "no user-facing change; no changelog entry needed"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

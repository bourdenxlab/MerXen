"""Tests for the release-version and changelog checks (scripts/ci/changelog.py)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

from ._ci_scripts import load_ci_script

changelog: ModuleType = load_ci_script("changelog")

REPO_ROOT = Path(__file__).resolve().parents[2]

GOOD = """# Changelog

Intro text.

## [Unreleased]

### Added

- Something new.

## [1.1.0] - 2026-12-01

### Highlights

- Big things.

### Fixed

#### A grouped fix

- A fix.

## [1.0.0] - 2026-11-13

### Added

- First release.

[Unreleased]: https://github.com/o/r/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/o/r/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/o/r/releases/tag/v1.0.0
"""


def test_releases_are_parsed_in_file_order() -> None:
    found = changelog.releases(GOOD)
    assert [(r.version, r.date) for r in found] == [
        ("1.1.0", "2026-12-01"),
        ("1.0.0", "2026-11-13"),
    ]


def test_notes_stop_at_the_next_section_and_the_links() -> None:
    assert changelog.notes(GOOD, "1.1.0") == (
        "### Highlights\n\n- Big things.\n\n### Fixed\n\n#### A grouped fix\n\n- A fix."
    )
    assert changelog.notes(GOOD, "1.0.0") == "### Added\n\n- First release."
    with pytest.raises(KeyError):
        changelog.notes(GOOD, "2.0.0")


def test_a_good_changelog_has_no_problems() -> None:
    assert changelog.validate(GOOD) == []


@pytest.mark.parametrize(
    ("edit", "expected"),
    [
        (lambda t: t.replace("## [Unreleased]\n", ""), "first section"),
        (lambda t: t.replace("## [1.0.0] - 2026-11-13", "## 1.0.0"), "neither"),
        (
            lambda t: t.replace(
                "## [1.0.0] - 2026-11-13", "## [1.2.0] - 2026-11-13"
            ).replace("[1.0.0]: ", "[1.2.0]: "),
            "not older",
        ),
        (lambda t: t.replace("### Highlights", "### Improvements"), "change type"),
        (lambda t: t.replace("[1.0.0]: https", "[x]: https"), "[1.0.0]"),
        (lambda t: t.replace("[Unreleased]: https", "[y]: https"), "[Unreleased]"),
        (
            lambda t: t.replace("### Added\n\n- First release.\n", ""),
            "1.0.0 is empty",
        ),
        (lambda t: t.replace("## [1.0.0]", "## [1.0]"), "MAJOR.MINOR.PATCH"),
    ],
)
def test_layout_problems_are_reported(edit, expected: str) -> None:  # noqa: ANN001
    problems = changelog.validate(edit(GOOD))
    assert any(expected in problem for problem in problems), problems


def test_parse_version_accepts_plain_semver_only() -> None:
    assert changelog.parse_version("10.2.0") == (10, 2, 0)
    for bad in ("1.2", "01.2.3", "1.2.3-rc.1", "v1.2.3"):
        with pytest.raises(ValueError):
            changelog.parse_version(bad)


def test_user_facing_paths_need_an_entry() -> None:
    paths = [
        "src/merxen/qc/metrics.py",
        "workflows/main.nf",
        "requirements/requirements.lock",
        "docs/outputs.md",
        "tests/test_qc/test_metrics.py",
        "scripts/ci/select_tests.py",
        ".github/workflows/ci.yml",
    ]
    assert changelog.needs_entry(paths) == paths[:3]


def _fake_repo(tmp_path: Path, versions: dict[str, str], changelog_text: str) -> Path:
    (tmp_path / "src/merxen").mkdir(parents=True)
    (tmp_path / "workflows").mkdir()
    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nversion = "{versions["project"]}"\n\n'
        f'[tool.bumpversion]\ncurrent_version = "{versions["bump"]}"\n'
    )
    (tmp_path / "src/merxen/__init__.py").write_text(
        f'"""Doc."""\n\n__version__ = "{versions["init"]}"\n'
    )
    (tmp_path / "workflows/nextflow.config").write_text(
        "manifest {\n    name = 'o/r'\n"
        f"    version = '{versions['manifest']}'\n}}\n\n"
        "params {\n    proseg_version = '3.2.0'\n}\n"
    )
    (tmp_path / "CHANGELOG.md").write_text(changelog_text)
    return tmp_path


def _same(version: str) -> dict[str, str]:
    return {key: version for key in ("project", "bump", "init", "manifest")}


def test_versions_agree(tmp_path: Path) -> None:
    root = _fake_repo(tmp_path, _same("1.1.0"), GOOD)
    assert changelog.check_versions(root) == ("1.1.0", [])


def test_a_differing_file_is_named(tmp_path: Path) -> None:
    versions = _same("1.1.0") | {"manifest": "1.0.0"}
    root = _fake_repo(tmp_path, versions, GOOD)
    version, problems = changelog.check_versions(root)
    assert version is None
    assert any("manifest.version = 1.0.0" in problem for problem in problems)


def test_the_version_must_be_the_newest_release(tmp_path: Path) -> None:
    root = _fake_repo(tmp_path, _same("1.0.0"), GOOD)
    _, problems = changelog.check_versions(root)
    assert any("not the newest" in problem for problem in problems)


def test_before_the_first_release_no_section_is_needed(tmp_path: Path) -> None:
    unreleased_only = "# Changelog\n\n## [Unreleased]\n\n- x\n\n[Unreleased]: u\n"
    root = _fake_repo(tmp_path, _same("0.2.0"), unreleased_only)
    assert changelog.check_versions(root) == ("0.2.0", [])


def test_the_version_must_be_newer_than_every_tag(tmp_path: Path) -> None:
    root = _fake_repo(tmp_path, _same("1.1.0"), GOOD)
    assert changelog.check_versions(root, ["v1.0.0", "v1.1.0", "archive/x"])[1] == []
    _, problems = changelog.check_versions(root, ["v1.0.0", "v1.2.0"])
    assert problems == ["version 1.1.0 is not newer than the tag v1.2.0"]


def test_check_pr_requires_an_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        changelog, "changed_files", lambda base, head: ["workflows/main.nf"]
    )
    assert changelog.main(["check-pr", "--base", "origin/main"]) == 1
    assert "skip-changelog" in capsys.readouterr().err
    monkeypatch.setattr(
        changelog,
        "changed_files",
        lambda base, head: ["workflows/main.nf", "CHANGELOG.md"],
    )
    assert changelog.main(["check-pr", "--base", "origin/main"]) == 0
    monkeypatch.setattr(changelog, "changed_files", lambda base, head: ["docs/x.md"])
    assert changelog.main(["check-pr", "--base", "origin/main"]) == 0


def test_this_repository_is_consistent() -> None:
    """The release version agrees everywhere and CHANGELOG.md is well formed."""
    version, problems = changelog.check_versions(REPO_ROOT)
    assert problems == [] and version is not None
    assert changelog.validate((REPO_ROOT / "CHANGELOG.md").read_text()) == []


def test_bump_config_finds_every_version_string(tmp_path: Path) -> None:
    """A dry-run bump finds the search strings in every configured file."""
    bump = shutil.which("bump-my-version")
    if bump is None:
        pytest.skip("bump-my-version is not installed")
    result = subprocess.run(
        [bump, "bump", "minor", "--dry-run", "--allow-dirty", "-v"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]

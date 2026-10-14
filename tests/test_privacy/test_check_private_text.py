"""Tests of the privacy hook script on scratch git repositories.

The script is called directly (no pre-commit dependency), with a synthetic
alias table and ``.env``. Every git call sets an identity and disables hooks
and signing, so the tests run on CI runners without a git identity.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from .synthetic import (
    DONOR,
    HOST_ROOT,
    MOUSE,
    STORE_VALUE,
    USER,
    write_env,
    write_table,
)

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/privacy/check_private_text.py"
GIT_CONFIG = (
    "-c",
    "core.hooksPath=/dev/null",
    "-c",
    "commit.gpgsign=false",
    "-c",
    "init.defaultBranch=main",
)
NOTICE = "privacy: no alias table (.aliases.tsv); real-name check skipped"


class Repo:
    """A scratch repository with the synthetic private files."""

    def __init__(self, tmp_path: Path) -> None:
        self.path = tmp_path / "repo"
        self.path.mkdir()
        private = tmp_path / "private"
        private.mkdir()
        self.env = os.environ.copy()
        self.env.update(
            {
                "GIT_AUTHOR_NAME": "Test",
                "GIT_AUTHOR_EMAIL": "test@example.org",
                "GIT_COMMITTER_NAME": "Test",
                "GIT_COMMITTER_EMAIL": "test@example.org",
                "GIT_CONFIG_NOSYSTEM": "1",
                "MERXEN_ENV_FILE": str(write_env(private / "env")),
                "MERXEN_ALIAS_FILE": str(write_table(private / "aliases.tsv")),
            }
        )
        for name in [n for n in self.env if n.startswith("PRE_COMMIT")]:
            del self.env[name]
        self.git("init", "-q")
        self.write("README.md", "clean start\n")
        self.commit("[docs]: start")

    def git(self, *args: str) -> str:
        """Run git in the repository."""
        return subprocess.run(
            ["git", *GIT_CONFIG, *args],
            cwd=self.path,
            env=self.env,
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    def write(self, name: str, text: str | bytes) -> None:
        """Write and stage a file."""
        target = self.path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(text, bytes):
            target.write_bytes(text)
        else:
            target.write_text(text, encoding="utf-8")
        self.git("add", "-f", name)

    def commit(self, message: str) -> str:
        """Commit the index (hooks off) and return the sha."""
        self.git("commit", "-q", "-m", message)
        return self.git("rev-parse", "HEAD").strip()

    def check(
        self, *args: str, stdin: str | None = None, **env: str
    ) -> subprocess.CompletedProcess[str]:
        """Run the hook script."""
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            cwd=self.path,
            env={**self.env, **env},
            input=stdin,
            capture_output=True,
            text=True,
        )


@pytest.fixture
def repo(tmp_path: Path) -> Repo:
    """A scratch repository with one clean commit."""
    return Repo(tmp_path)


def test_staged_accepts_clean_lines(repo: Repo) -> None:
    repo.write("docs/page.md", "HD1 and MS1 under $MERXEN_STORAGE_ROOT/x\n")
    result = repo.check("--staged")
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout == ""


def test_staged_refuses_a_name_without_printing_it(repo: Repo) -> None:
    repo.write("docs/page.md", f"line one\nthe {DONOR} sample\n")
    result = repo.check("--staged")
    assert result.returncode == 1
    assert result.stdout.strip() == "docs/page.md:2:5: name: HD1"
    assert DONOR not in result.stdout + result.stderr


def test_staged_refuses_a_host_path_and_an_exact_value(repo: Repo) -> None:
    repo.write("src/a.py", f'ROOT = "{HOST_ROOT}/x"\nSTORE = "{STORE_VALUE}/y"\n')
    result = repo.check("--staged")
    assert result.returncode == 1
    assert "src/a.py:1:9: host-path: root=srv" in result.stdout
    assert "src/a.py:2:10: env-value: $MERXEN_STORAGE_ROOT" in result.stdout
    assert STORE_VALUE not in result.stdout + result.stderr


def test_staged_refuses_a_name_in_a_new_file_name(repo: Repo) -> None:
    repo.write(f"notebooks/{DONOR}_qc.ipynb", "{}\n")
    result = repo.check("--staged")
    assert result.returncode == 1
    assert "notebooks/[HD1]_qc.ipynb (path name): name: HD1" in result.stdout
    assert DONOR not in result.stdout


def test_staged_refuses_private_files_by_name(repo: Repo) -> None:
    repo.write(".env", "MERXEN_X=1\n")
    result = repo.check("--staged")
    assert result.returncode == 1
    assert ".env: private-file" in result.stdout


def test_staged_searches_binary_files(repo: Repo) -> None:
    repo.write("assets/blob.bin", b"\x00\x01\x02" + f"by {USER}!".encode() + b"\xff")
    result = repo.check("--staged")
    assert result.returncode == 1
    assert "assets/blob.bin:byte" in result.stdout
    assert "name: <user>" in result.stdout


def test_staged_checks_only_added_lines(repo: Repo) -> None:
    repo.write("docs/old.md", f"old {DONOR} line\nkeep\n")
    repo.commit("[docs]: old page")
    repo.write("docs/old.md", f"old {DONOR} line\nkeep\nnew clean line\n")
    assert repo.check("--staged").returncode == 0


def test_staged_honours_skip_paths(repo: Repo) -> None:
    repo.write("fmt/reader.py", "# reads the qqz5 archive\n")
    assert repo.check("--staged").returncode == 0
    repo.write("docs/data.md", "the qqz5 section\n")
    result = repo.check("--staged")
    assert result.returncode == 1
    assert "docs/data.md:1:5: name: MS2" in result.stdout


def test_show_match_prints_the_matched_text(repo: Repo) -> None:
    repo.write("docs/page.md", f"the {DONOR} sample\n")
    result = repo.check("--staged", "--show-match")
    assert f"[{DONOR}]" in result.stdout


def test_disabled_table_skips_names_with_the_notice(repo: Repo) -> None:
    repo.write("docs/page.md", f"the {MOUSE} section\n")
    result = repo.check("--staged", MERXEN_ALIAS_FILE="none")
    assert result.returncode == 0
    assert NOTICE in result.stderr
    repo.write("docs/page.md", f"the {MOUSE} section in {HOST_ROOT}/x and {DONOR}\n")
    result = repo.check("--staged", MERXEN_ALIAS_FILE="none")
    assert result.returncode == 1
    assert "host-path: root=srv" in result.stdout
    assert "donor-id: P####" in result.stdout


def test_malformed_env_file_fails_closed(repo: Repo, tmp_path: Path) -> None:
    bad = tmp_path / "bad.env"
    bad.write_text("export MERXEN_X=1\n", encoding="utf-8")
    result = repo.check("--staged", MERXEN_ENV_FILE=str(bad))
    assert result.returncode == 2
    assert "no 'export'" in result.stderr


def test_commit_msg_refuses_a_name_and_ignores_comments(
    repo: Repo, tmp_path: Path
) -> None:
    message = tmp_path / "COMMIT_EDITMSG"
    message.write_text(
        "[docs]: describe HD1\n\n# renamed: notebooks/"
        f"{DONOR}_x.ipynb\n# ------------------------ >8 ------------------------\n"
        f"+{DONOR}\n",
        encoding="utf-8",
    )
    assert repo.check("--commit-msg", str(message)).returncode == 0
    message.write_text(f"[docs]: describe {DONOR}\n", encoding="utf-8")
    result = repo.check("--commit-msg", str(message))
    assert result.returncode == 1
    assert result.stdout.strip() == "commit message:1:18: name: HD1"


def test_stdin_checks_pr_text(repo: Repo) -> None:
    assert repo.check("--stdin", stdin="[chore]: tidy\n\nHD1 only\n").returncode == 0
    result = repo.check("--stdin", stdin=f"title\n\nbody with {USER}\n")
    assert result.returncode == 1
    assert "<stdin>:3:11: name: <user>" in result.stdout


def _pushed(repo: Repo) -> dict[str, str]:
    """Pre-push variables for the current branch, the remote at main."""
    return {
        "PRE_COMMIT_FROM_REF": repo.git("rev-parse", "main").strip(),
        "PRE_COMMIT_TO_REF": repo.git("rev-parse", "HEAD").strip(),
        "PRE_COMMIT_LOCAL_BRANCH": "refs/heads/feature/work",
        "PRE_COMMIT_REMOTE_BRANCH": "refs/heads/feature/work",
    }


def test_pre_push_accepts_clean_commits(repo: Repo) -> None:
    repo.git("update-ref", "refs/remotes/origin/main", "HEAD")
    repo.git("checkout", "-q", "-b", "feature/work")
    repo.write("docs/page.md", "HD1 only\n")
    repo.commit("[docs]: HD1 page")
    result = repo.check("--pre-push", **_pushed(repo))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "scanned 1 pushed commit(s)" in result.stderr


def test_pre_push_refuses_a_branch_name(repo: Repo) -> None:
    repo.git("update-ref", "refs/remotes/origin/main", "HEAD")
    variables = {**_pushed(repo), "PRE_COMMIT_LOCAL_BRANCH": f"refs/heads/fix-{MOUSE}"}
    result = repo.check("--pre-push", **variables)
    assert result.returncode == 1
    assert "fix-[MS1] (local branch name): name: MS1" in result.stdout
    assert MOUSE not in result.stdout


def test_pre_push_refuses_names_in_messages_and_bypassed_commits(repo: Repo) -> None:
    repo.git("update-ref", "refs/remotes/origin/main", "HEAD")
    repo.git("checkout", "-q", "-b", "feature/work")
    repo.write("docs/a.md", "clean\n")
    first = repo.commit(f"[docs]: {DONOR} notes")
    repo.write("docs/b.md", f"{HOST_ROOT}/x\n")
    second = repo.commit("[docs]: clean message")
    result = repo.check("--pre-push", **_pushed(repo))
    assert result.returncode == 1
    assert f"{first[:10]} message:1:9: name: HD1" in result.stdout
    assert f"{second[:10]} docs/b.md:1:1: host-path: root=srv" in result.stdout
    assert "scanned 2 pushed commit(s)" in result.stderr


def test_pre_push_skips_commits_already_on_a_remote(repo: Repo) -> None:
    repo.write("docs/legacy.md", f"{DONOR}\n")
    repo.commit(f"[docs]: legacy {DONOR}")
    repo.git("update-ref", "refs/remotes/origin/main", "HEAD")
    repo.git("checkout", "-q", "-b", "feature/work")
    repo.write("docs/new.md", "clean\n")
    repo.commit("[docs]: new page")
    result = repo.check("--pre-push", **_pushed(repo))
    assert result.returncode == 0, result.stdout
    assert "scanned 1 pushed commit(s)" in result.stderr


def test_pre_push_checks_a_merge_against_its_first_parent(repo: Repo) -> None:
    repo.git("update-ref", "refs/remotes/origin/main", "HEAD")
    repo.git("checkout", "-q", "-b", "side")
    repo.write("docs/side.md", f"{DONOR}\n")
    repo.commit("[docs]: side page")
    repo.git("checkout", "-q", "-b", "feature/work", "main")
    repo.write("docs/own.md", "clean\n")
    repo.commit("[docs]: own page")
    repo.git("merge", "-q", "--no-ff", "--no-edit", "side")
    result = repo.check("--pre-push", **_pushed(repo))
    assert result.returncode == 1
    merge = repo.git("rev-parse", "HEAD").strip()[:10]
    assert f"{merge} docs/side.md:1:1: name: HD1" in result.stdout


def test_pre_push_skips_a_branch_deletion(repo: Repo) -> None:
    result = repo.check("--pre-push", PRE_COMMIT_TO_REF="0" * 40)
    assert result.returncode == 0


def test_tree_scans_tracked_files(repo: Repo) -> None:
    repo.write("docs/a.md", f"clean\n{DONOR}\n")
    repo.commit("[docs]: a")
    (repo.path / "untracked.md").write_text(f"{DONOR}\n", encoding="utf-8")
    result = repo.check("--tree")
    assert result.returncode == 1
    assert result.stdout.strip() == "docs/a.md:2:1: name: HD1"
    assert "scanned 2 file(s)" in result.stderr


def test_tree_walks_an_export_without_git(repo: Repo, tmp_path: Path) -> None:
    export = tmp_path / "export"
    (export / "docs").mkdir(parents=True)
    (export / "docs/a.md").write_text(f"{HOST_ROOT}/x\n", encoding="utf-8")
    result = repo.check("--tree", str(export))
    assert result.returncode == 1
    assert "docs/a.md:1:1: host-path: root=srv" in result.stdout


def test_tree_report_is_refused_inside_a_work_tree(repo: Repo, tmp_path: Path) -> None:
    inside = repo.check("--tree", "--report", str(repo.path / "report.txt"))
    assert inside.returncode == 2
    assert not (repo.path / "report.txt").exists()
    repo.write("docs/a.md", f"{DONOR}\n")
    repo.commit("[docs]: a")
    report = tmp_path / "private" / "report.txt"
    outside = repo.check("--tree", "--report", str(report))
    assert outside.returncode == 1
    assert report.read_text(encoding="utf-8") == "docs/a.md:1:1: name: HD1\n"
    assert outside.stdout == ""


def test_list_exceptions(repo: Repo) -> None:
    result = repo.check("--list-exceptions")
    assert result.returncode == 0
    assert "/tmp/merxen-dwight-gpu.lock" in result.stdout

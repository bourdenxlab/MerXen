#!/usr/bin/env python3
"""Refuse host paths and sample names in commits, messages, branches and PR text.

The privacy hook of the repository (CLAUDE.md, docs/local-configuration.md).
Stdlib only: it runs under the system ``python3`` and imports
``merxen.privacy`` from this checkout's ``src``.

Rules (``merxen.privacy``):

1. generic host paths (host roots, a session scratch dir, Windows user dirs);
2. exact private values of ``.env`` and the Nextflow site configs (when found);
3. generic donor ids (``P`` + 4 digits, synthetic fixture ids excepted);
4. real names of the alias table (when found; otherwise a notice says the
   check is skipped).

Modes:

``--staged``          added lines and new file names of the index (pre-commit,
                      pre-merge-commit); private files are refused by name.
``--commit-msg FILE`` the commit message (commit-msg).
``--pre-push``        branch names, messages and added lines of every pushed
                      commit not on a remote (pre-push; run by pre-commit with
                      ``PRE_COMMIT_*`` variables, or by hand on ``HEAD``).
``--stdin``           text on standard input (PR title and body).
``--tree [ROOT]``     every tracked file (``git ls-files``, or a walk of an
                      export).

Output lines are ``where: rule: label``; the label is the alias, placeholder,
root word or variable, never the matched text (``--show-match`` adds it, for
interactive local use only). Exit status: 0 clean, 1 hits, 2 error.
"""

from __future__ import annotations

import argparse
import codecs
import os
import subprocess
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from merxen import privacy  # noqa: E402
from merxen.local_config import LocalConfigError  # noqa: E402

ZERO_SHA = "0" * 40
EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
NOTICE = "privacy: no alias table (.aliases.tsv); real-name check skipped"
HINT = (
    "privacy: write aliases (HD1, MS1, ...) instead of sample names and .env "
    "variables ($MERXEN_...) instead of host paths; see "
    "docs/local-configuration.md. --show-match prints the matched text (local "
    "use only; never paste it into a commit, PR or report)."
)


@dataclass(frozen=True)
class Finding:
    """One hit with the place it was found."""

    where: str
    hit: privacy.Hit


def git(*args: str, cwd: Path | None = None) -> bytes:
    """Run git and return its standard output (raises on failure)."""
    command = ["git", "-c", "core.quotepath=off", *args]
    return subprocess.run(command, cwd=cwd, check=True, capture_output=True).stdout


def git_text(*args: str, cwd: Path | None = None) -> str:
    """Run git and return its output as text."""
    return git(*args, cwd=cwd).decode("utf-8", errors="replace")


def unquote_path(raw: str) -> str:
    """Undo git's C-style quoting of a path in diff headers."""
    if len(raw) >= 2 and raw[0] == raw[-1] == '"':
        escaped = raw[1:-1].encode("latin-1", errors="backslashreplace")
        return codecs.escape_decode(escaped)[0].decode("utf-8", errors="replace")
    return raw


def added_lines(diff: str) -> Iterator[tuple[str, int, str]]:
    """Yield (path, new line number, text) of every added line of a -U0 diff."""
    path = ""
    number = 0
    in_header = False
    for line in diff.split("\n"):
        if line.startswith("diff --git "):
            in_header, path = True, ""
        elif in_header and line.startswith("+++ "):
            target = unquote_path(line[4:].rstrip("\t"))
            path = "" if target == "/dev/null" else target.removeprefix("b/")
        elif line.startswith("@@"):
            in_header = False
            header = line.split("+", 1)[1].split(" ", 1)[0]
            number = int(header.split(",", 1)[0])
        elif not in_header and line.startswith("+") and path:
            yield path, number, line[1:]
            number += 1


def binary_paths(numstat: bytes) -> list[str]:
    """Paths that ``git diff --numstat -z`` reports as binary."""
    paths: list[str] = []
    fields = numstat.split(b"\0")
    index = 0
    while index < len(fields):
        entry = fields[index]
        index += 1
        if not entry:
            continue
        added, deleted, rest = entry.split(b"\t", 2)
        if rest == b"":
            # Rename or copy: the next two fields are the old and new paths.
            rest = fields[index + 1]
            index += 2
        if added == b"-" and deleted == b"-":
            paths.append(rest.decode("utf-8", errors="replace"))
    return paths


def name_status(raw: bytes) -> list[tuple[str, str]]:
    """(status letter, new path) pairs of ``--name-status -z`` output."""
    fields = [field.decode("utf-8", errors="replace") for field in raw.split(b"\0")]
    pairs: list[tuple[str, str]] = []
    index = 0
    while index < len(fields) and fields[index]:
        status = fields[index]
        if status[0] in "RC":
            pairs.append((status[0], fields[index + 2]))
            index += 3
        else:
            pairs.append((status[0], fields[index + 1]))
            index += 2
    return pairs


class Scanner:
    """Collects findings for one run of the hook."""

    def __init__(self, show_match: bool) -> None:
        self.show_match = show_match
        self.values = privacy.private_values()
        self.findings: list[Finding] = []

    def text(self, where: str, text: str, path: str | None = None) -> None:
        """Scan free text (a message, stdin)."""
        for hit in privacy.scan_text(text, path=path, values=self.values):
            self.findings.append(Finding(f"{where}:{hit.line}:{hit.column}", hit))

    def line(self, where: str, number: int, text: str, path: str) -> None:
        """Scan one added line of ``path``."""
        safe = privacy.safe_text(path, path)
        for hit in privacy.scan_line(
            text, path=path, values=self.values, number=number
        ):
            self.findings.append(Finding(f"{where}{safe}:{number}:{hit.column}", hit))

    def blob(self, where: str, data: bytes, path: str) -> None:
        """Scan the content of a binary file."""
        safe = privacy.safe_text(path, path)
        for hit in privacy.scan_bytes(data, path=path, values=self.values):
            self.findings.append(Finding(f"{where}{safe}:byte {hit.column}", hit))

    def name(self, where: str, name: str, kind: str = "path") -> None:
        """Scan a file or branch name; refuse private files by name."""
        safe = privacy.safe_text(name, name)
        if kind == "path" and privacy.is_private_file(name):
            hit = privacy.Hit(0, 1, "private-file", "gitignored private file", name)
            self.findings.append(Finding(f"{where}{safe}", hit))
        for hit in privacy.scan_line(name, path=name, values=self.values, number=0):
            self.findings.append(Finding(f"{where}{safe} ({kind} name)", hit))

    def report(self, out: list[str]) -> None:
        """Append the report lines (never the matched text unless asked)."""
        for finding in self.findings:
            hit = finding.hit
            line = f"{finding.where}: {hit.rule}: {hit.label}"
            if self.show_match:
                line += f"  [{hit.text}]"
            out.append(line)


def scan_diff(scanner: Scanner, where: str, diff_args: Sequence[str]) -> None:
    """Scan added lines, binaries and new names of one diff."""
    # T: a type change (a file replaced by a symlink) carries new content.
    common = ["--no-color", "--no-ext-diff", "--no-textconv", "--diff-filter=ACMRT"]
    status = git("diff", *diff_args, "--name-status", "-z", *common)
    for letter, path in name_status(status):
        if letter in "ACR" or privacy.is_private_file(path):
            scanner.name(where, path)
    binaries = set(binary_paths(git("diff", *diff_args, "--numstat", "-z", *common)))
    diff = git_text(
        "diff", *diff_args, "-U0", "--src-prefix=a/", "--dst-prefix=b/", *common
    )
    for path, number, text in added_lines(diff):
        if path not in binaries:
            scanner.line(where, number, text, path)
    for path in sorted(binaries):
        spec = f":{path}" if "--cached" in diff_args else f"{diff_args[-1]}:{path}"
        scanner.blob(where, git("show", spec), path)


def mode_staged(scanner: Scanner) -> None:
    """Pre-commit and pre-merge-commit: the index against HEAD."""
    scan_diff(scanner, "", ["--cached"])


def strip_message(text: str) -> str:
    """Drop git comment lines and everything below the scissors line."""
    try:
        char = git_text("config", "core.commentChar").strip() or "#"
    except subprocess.CalledProcessError:
        char = "#"
    if char == "auto":
        char = "#"
    kept: list[str] = []
    for line in text.split("\n"):
        if line.startswith(f"{char} ") and ">8" in line and "---" in line:
            break
        kept.append("" if line.startswith(char) else line)
    return "\n".join(kept)


def mode_commit_msg(scanner: Scanner, message_file: Path) -> None:
    """Commit-msg: the message, without git's comment lines."""
    text = message_file.read_text(encoding="utf-8", errors="replace")
    scanner.text("commit message", strip_message(text))


def pushed_commits(from_ref: str, to_ref: str) -> list[str]:
    """Commits being pushed that no remote-tracking ref already holds."""
    exclude = [from_ref] if from_ref and from_ref != ZERO_SHA else []
    out = git_text("rev-list", "--reverse", to_ref, "--not", *exclude, "--remotes")
    return out.split()


def mode_pre_push(scanner: Scanner) -> int:
    """Pre-push: branch names, then every pushed commit's message and diff."""
    to_ref = os.environ.get("PRE_COMMIT_TO_REF") or "HEAD"
    from_ref = os.environ.get("PRE_COMMIT_FROM_REF", "")
    if to_ref == ZERO_SHA:
        return 0
    local = os.environ.get("PRE_COMMIT_LOCAL_BRANCH")
    if local is None:
        local = git_text("rev-parse", "--abbrev-ref", "HEAD").strip()
    remote = os.environ.get("PRE_COMMIT_REMOTE_BRANCH", "")
    for kind, ref in (("local branch", local), ("remote branch", remote)):
        if ref:
            scanner.name("", ref.removeprefix("refs/heads/"), kind=kind)
    commits = pushed_commits(from_ref, to_ref)
    for commit in commits:
        short = commit[:10]
        message = git_text("log", "-1", "--format=%B", commit)
        scanner.text(f"{short} message", message)
        parents = git_text("rev-list", "--parents", "-n", "1", commit).split()[1:]
        base = parents[0] if parents else EMPTY_TREE
        scan_diff(scanner, f"{short} ", [base, commit])
    return len(commits)


def tracked_files(root: Path) -> list[str]:
    """Tracked files of a checkout, or every file of an export."""
    try:
        raw = git("ls-files", "-z", cwd=root)
        return [name.decode("utf-8") for name in raw.split(b"\0") if name]
    except (subprocess.CalledProcessError, FileNotFoundError):
        files: list[str] = []
        for directory, subdirs, names in os.walk(root):
            subdirs[:] = [name for name in subdirs if name != ".git"]
            for name in names:
                files.append(str((Path(directory) / name).relative_to(root)))
        return sorted(files)


def mode_tree(scanner: Scanner, root: Path) -> int:
    """Every tracked file: names and content (binaries for rules 2 and 4)."""
    files = tracked_files(root)
    for name in files:
        target = root / name
        scanner.name("", name)
        if target.is_symlink():
            data = os.readlink(target).encode("utf-8")
        elif target.is_file():
            data = target.read_bytes()
        else:
            continue
        if b"\0" in data[:8000]:
            scanner.blob("", data, name)
            continue
        text = data.decode("utf-8", errors="replace")
        safe = privacy.safe_text(name, name)
        hits = privacy.scan_text(text, path=name, values=scanner.values)
        for hit in hits:
            where = f"{safe}:{hit.line}:{hit.column}"
            scanner.findings.append(Finding(where, hit))
    return len(files)


def inside_work_tree(path: Path) -> bool:
    """Whether ``path`` lies inside any git working tree."""
    return any((parent / ".git").exists() for parent in [path, *path.parents])


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--staged", action="store_true", help="index vs HEAD")
    mode.add_argument("--commit-msg", type=Path, metavar="FILE", help="message")
    mode.add_argument("--pre-push", action="store_true", help="pushed commits")
    mode.add_argument("--stdin", action="store_true", help="standard input")
    mode.add_argument("--tree", type=Path, nargs="?", const=Path("."), metavar="ROOT")
    mode.add_argument(
        "--list-exceptions", action="store_true", help="print what may stay"
    )
    parser.add_argument(
        "--show-match",
        action="store_true",
        help="also print the matched text (interactive local use only)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="write the hit lines here (refused inside a git working tree)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one mode and print the findings."""
    args = parse_args(argv)
    if args.list_exceptions:
        for item in privacy.EXCEPTIONS:
            print(f"- {item.what} ({item.where}): {item.why}")
        return 0
    if args.report is not None:
        target = args.report.resolve()
        if inside_work_tree(target.parent):
            print(
                "privacy: refusing a report inside a git working tree "
                "(it may hold matched text); write it to a private location",
                file=sys.stderr,
            )
            return 2
    try:
        if privacy.table() is None:
            print(NOTICE, file=sys.stderr)
        scanner = Scanner(show_match=args.show_match)
        summary = ""
        if args.staged:
            mode_staged(scanner)
        elif args.commit_msg is not None:
            mode_commit_msg(scanner, args.commit_msg)
        elif args.pre_push:
            count = mode_pre_push(scanner)
            summary = f"privacy: scanned {count} pushed commit(s)"
        elif args.stdin:
            scanner.text("<stdin>", sys.stdin.read())
        else:
            count = mode_tree(scanner, args.tree.resolve())
            summary = f"privacy: scanned {count} file(s)"
    except LocalConfigError as error:
        print(f"privacy: {error}", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as error:
        stderr = error.stderr.decode("utf-8", errors="replace").strip()
        print(f"privacy: git failed: {stderr}", file=sys.stderr)
        return 2
    if summary:
        print(summary, file=sys.stderr)
    lines: list[str] = []
    scanner.report(lines)
    if args.report is not None:
        args.report.write_text("\n".join(lines) + ("\n" if lines else ""))
        os.chmod(args.report, 0o600)
        print(f"privacy: {len(lines)} hit(s) written to the report", file=sys.stderr)
    else:
        for line in lines:
            print(line)
    if lines:
        print(f"privacy: {len(lines)} hit(s)", file=sys.stderr)
        print(HINT, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Pick the tests a change can break, or decide that the full suite must run.

CI's ``plan`` job and the local ``changed`` test tier (scripts/run_ci_checks.sh)
call this script. It reads the changed files (``git diff base...head``, plus
the working tree with ``--worktree``) and prints either FULL or the test files
to run.

A test file is selected when a changed file is in its static dependency
closure: the modules it imports (directly, through helpers and through every
``conftest.py`` above it, transitively), the scripts it loads, the repository
files it names by path or base name, and, for tests that launch Nextflow, the
``merxen`` commands and ``python -m merxen.*`` modules of the workflow files
they run. The guard tests below always run. Anything the rules cannot map, and
every change to the shared fixtures, dependencies or CI itself, runs the full
suite. Over-selection is the intended failure mode; see docs/development.md
("Test selection") for what static analysis cannot see.

The script uses only the standard library so that the CI plan job can run it
before any dependency is installed. Do not rename it to ``select.py``: that
would shadow the stdlib ``select`` module, which ``subprocess`` imports.
"""

from __future__ import annotations

import argparse
import ast
import functools
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

# ---------------------------------------------------------------------------
# Policy. Every rule that decides between FULL and a selection lives here.
# ---------------------------------------------------------------------------

#: Long-lived integration branches (CLAUDE.md, Workflow). A PR that merges one
#: of them into main always runs the full suite.
INTEGRATION_BRANCHES = (
    "feature/robust-celltype-annotation",
    "feature/atera-support",
    "feature/manual-registration",
)
#: A PR carrying this label runs the full suite.
FULL_LABEL = "full-ci"
#: Events that always run the full suite (pushes to main, the nightly run and
#: manual runs).
FULL_EVENTS = ("push", "schedule", "workflow_dispatch")
#: Cheap repository-wide guards that run on every selection.
GUARDS = (
    "tests/test_privacy",
    "tests/test_workflows/test_process_script_pins.py",
    "tests/test_workflows/test_platform_config_golden.py",
    "tests/test_platform_sites.py",
    "tests/test_workflows/test_metro_map.py",
    "tests/test_workflows/test_analysis_mode_parity.py",
    "tests/test_workflows/test_env_lock_sync.py",
)
#: A change to any of these runs the full suite: shared fixtures, dependencies,
#: CI and this selector, and the modules nearly every test imports.
FULL_PATTERNS = (
    "**/conftest.py",
    "pyproject.toml",
    "requirements/**",
    "envs/**",
    ".github/**",
    ".pre-commit-config.yaml",
    "scripts/run_ci_checks.sh",
    "scripts/ci/**",
    "src/merxen/__init__.py",
    "src/merxen/config.py",
    "src/merxen/io/spatialdata_io.py",
    "src/merxen/io/spatialdata_schema.py",
)
#: Files no test imports. They select only the tests that name them (and the
#: guards). Every other unmapped path runs the full suite.
REFERENCE_ONLY_PATTERNS = (
    "docs/**",
    "*.md",
    "LICENSE",
    "MANIFEST.in",
    ".gitignore",
    ".gitattributes",
    ".env.example",
    ".aliases.example.tsv",
    "notebooks/**",
    "licenses/**",
    "assets/**",
    "containers/**",
)
#: Shards of a full run, and the most a selected run gets.
FULL_SHARDS = 5
#: Estimated worker-seconds one shard of a selected run takes on (4 workers x
#: about 150 s).
SELECTED_SHARD_SECONDS = 600.0
#: A selection estimated above this fraction of the suite runs FULL instead.
CAP = 0.60
#: Estimate for a test file without recorded durations.
UNKNOWN_FILE_SECONDS = 30.0

# Base names too common to count as a reference to one particular file.
_GENERIC_BASENAMES = frozenset(
    {"__init__.py", "__main__.py", "conftest.py", "config.py", "README.md"}
)
_MIN_NAME_LENGTH = 5
# Top-level directories a repository path can start with.
_REPO_DIRS = (
    "workflows",
    "docs",
    "scripts",
    "assets",
    "envs",
    "containers",
    "requirements",
    "notebooks",
    "licenses",
    "src",
    "tests",
    ".github",
)
_PATH_IN_TEXT = re.compile(
    r"(?<![\w.-])((?:" + "|".join(re.escape(d) for d in _REPO_DIRS) + r")/[\w./*-]*)"
)
_MODULE_IN_TEXT = re.compile(r"\bmerxen(?:\.[A-Za-z_]\w*)+")
_NF_INCLUDE = re.compile(r"\binclude\s*\{[^}]*\}\s*from\s*['\"]([^'\"]+)['\"]")
_NF_PYTHON_MODULE = re.compile(r"\bpython3?\s+(?:-\w+\s+)*-m\s+(merxen[\w.]*)")
_NF_COMMAND = re.compile(r"\bmerxen\s+([a-z][a-z0-9-]+)")
_NF_SCRIPT = re.compile(r"(?<![\w-])(scripts/[\w./-]+\.py)")
_CLICK_NAME = re.compile(
    r"@\w+\.(?:command|group)\(\s*(?:name\s*=\s*)?['\"]([^'\"]+)['\"]"
)
_ROOT_DIRECTORIES = frozenset({"src", "src/merxen", "tests"})
_INDEXED_SUFFIXES = (".py", ".nf", ".groovy", ".config")
_SKIP_DIRS = frozenset({"__pycache__", ".git", ".ci-venv", ".venv", "node_modules"})


# ---------------------------------------------------------------------------
# Glob matching with ``**`` (fnmatch's ``*`` crosses ``/``).
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=4096)
def _glob_regex(pattern: str) -> re.Pattern[str]:
    out = []
    i = 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("/**", i) and i + 3 == len(pattern):
            out.append("(?:/.*)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("".join(out) + r"\Z")


def glob_match(path: str, pattern: str) -> bool:
    """Whether a repository path matches a glob (``**`` spans directories)."""
    return _glob_regex(pattern).match(path) is not None


def reference_matches(path: str, reference: str) -> bool:
    """Whether ``path`` is the referenced file, below the referenced directory,
    or matched by the referenced glob."""
    if any(c in reference for c in "*?["):
        return glob_match(path, reference)
    if reference in ("", "."):
        return True
    return path == reference or path.startswith(reference.rstrip("/") + "/")


def _normalise(path: str) -> str | None:
    """Collapse ``.`` and ``..`` in a relative POSIX path (None above the root)."""
    parts: list[str] = []
    for part in path.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if not parts:
                return None
            parts.pop()
        else:
            parts.append(part)
    return "/".join(parts)


def _is_test_file(path: str) -> bool:
    name = PurePosixPath(path).name
    return (
        path.startswith("tests/")
        and name.endswith(".py")
        and (name.startswith("test_") or name.endswith("_test.py"))
    )


# ---------------------------------------------------------------------------
# Static analysis of one file.
# ---------------------------------------------------------------------------


@dataclass
class FileFacts:
    """What one indexed file depends on."""

    #: Repository files this file needs, with whether their own dependencies
    #: count too (``deep``) or only the file itself.
    edges: dict[str, bool] = field(default_factory=dict)
    #: Repository paths, directories and globs the file names.
    references: set[str] = field(default_factory=set)
    #: Single-token string literals (for base-name and script-stem matches).
    tokens: set[str] = field(default_factory=set)
    #: The references and tokens outside ``.read_text()``/``open()`` calls:
    #: the files the code may run rather than only read.
    run_references: set[str] = field(default_factory=set)
    run_tokens: set[str] = field(default_factory=set)
    #: Whether the file starts Nextflow.
    launches_nextflow: bool = False

    def add_edge(self, target: str, *, deep: bool = True) -> None:
        self.edges[target] = self.edges.get(target, False) or deep


class _PathResolver:
    """Resolve ``Path(__file__)``-based expressions to repository paths."""

    def __init__(self, relpath: str, tree: ast.AST) -> None:
        self.relpath = relpath
        self.bindings: dict[str, list[ast.expr]] = {}
        for node in ast.walk(tree):
            targets: list[ast.expr] = []
            value: ast.expr | None = None
            if isinstance(node, ast.Assign):
                targets, value = list(node.targets), node.value
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                targets, value = [node.target], node.value
            for target in targets:
                if isinstance(target, ast.Name) and value is not None:
                    self.bindings.setdefault(target.id, []).append(value)
        self._active: set[str] = set()

    def resolve(self, node: ast.expr) -> list[tuple[str, bool]]:
        """Paths an expression can evaluate to, as (path, is_prefix_only).

        ``is_prefix_only`` is set when the tail is computed at run time (the
        result names a directory whose contents the code picks)."""
        if isinstance(node, ast.Name):
            if node.id == "__file__":
                return [(self.relpath, False)]
            if node.id in self._active or node.id not in self.bindings:
                return []
            self._active.add(node.id)
            try:
                out: list[tuple[str, bool]] = []
                for value in self.bindings[node.id]:
                    out.extend(self.resolve(value))
                return out
            finally:
                self._active.discard(node.id)
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in (
                "Path",
                "PurePath",
                "PosixPath",
            ):
                return self._join_args(node.args)
            if isinstance(func, ast.Attribute):
                if (
                    func.attr in ("Path", "PurePath", "PosixPath")
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "pathlib"
                ):
                    return self._join_args(node.args)
                if func.attr in ("resolve", "absolute", "expanduser"):
                    return self.resolve(func.value)
                if (
                    func.attr in ("dirname", "abspath", "realpath", "normpath")
                    and node.args
                ):
                    inner = self.resolve(node.args[0])
                    if func.attr == "dirname":
                        return [(_parent(p), pre) for p, pre in inner if p is not None]
                    return inner
                if func.attr == "join" and node.args and _is_os_path(func.value):
                    return self._join_args(node.args)
                if func.attr == "joinpath":
                    return self._join(self.resolve(func.value), node.args)
            return []
        if isinstance(node, ast.Attribute) and node.attr == "parent":
            return [(_parent(p), pre) for p, pre in self.resolve(node.value) if not pre]
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute):
            if node.value.attr == "parents" and isinstance(node.slice, ast.Constant):
                level = node.slice.value
                if isinstance(level, int):
                    out = []
                    for path, pre in self.resolve(node.value.value):
                        if pre:
                            continue
                        for _ in range(level + 1):
                            path = _parent(path)
                        out.append((path, False))
                    return out
            return []
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            return self._join(self.resolve(node.left), [node.right])
        return []

    def _join_args(self, args: Sequence[ast.expr]) -> list[tuple[str, bool]]:
        if not args:
            return []
        return self._join(self.resolve(args[0]), args[1:])

    def _join(
        self, bases: list[tuple[str, bool]], parts: Sequence[ast.expr]
    ) -> list[tuple[str, bool]]:
        out = []
        for base, prefix_only in bases:
            path = base
            for part in parts:
                if prefix_only:
                    break
                if isinstance(part, ast.Constant) and isinstance(part.value, str):
                    path = f"{path}/{part.value}" if path else part.value
                elif isinstance(part, ast.JoinedStr):
                    head = _fstring_head(part)
                    if "/" in head:
                        path = f"{path}/{head.rsplit('/', 1)[0]}" if path else head
                    prefix_only = True
                else:
                    prefix_only = True
            out.append((path, prefix_only))
        return out


class _Context:
    """One parsed file and what the analysis of any part of it needs."""

    def __init__(self, path: str, tree: ast.AST, *, is_test_side: bool) -> None:
        self.path = path
        self.tree = tree
        self.is_test_side = is_test_side
        self.read = _read_only_nodes(tree) if is_test_side else set()
        self.resolver = _PathResolver(path, tree)
        # Only whole ``a / "b" / "c"`` chains count: their left parts name
        # directories the code does not read as a whole.
        self.inner = {
            id(node.left)
            for node in ast.walk(tree)
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)
        }


def _fixture_info(node: ast.AST) -> tuple[str | None, bool]:
    """(fixture name, autouse) of a ``@pytest.fixture`` function, else (None, False)."""
    if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        return None, False
    for decorator in node.decorator_list:
        call = decorator if isinstance(decorator, ast.Call) else None
        target = call.func if call is not None else decorator
        name = (
            target.attr
            if isinstance(target, ast.Attribute)
            else getattr(target, "id", "")
        )
        if name != "fixture":
            continue
        fixture_name, autouse = node.name, False
        for keyword in call.keywords if call is not None else []:
            if keyword.arg == "autouse":
                # Anything but a literal False counts as autouse.
                autouse = not (
                    isinstance(keyword.value, ast.Constant)
                    and keyword.value.value is False
                )
            elif keyword.arg == "name" and isinstance(keyword.value, ast.Constant):
                fixture_name = str(keyword.value.value)
        return fixture_name, autouse
    return None, False


def _names(node: ast.AST) -> set[str]:
    """Identifiers used under ``node``: names, attribute roots, parameters."""
    out: set[str] = set()
    for inner in ast.walk(node):
        if isinstance(inner, ast.Name):
            out.add(inner.id)
        elif isinstance(inner, ast.arg):
            out.add(inner.arg)
        elif (
            isinstance(inner, ast.Constant)
            and isinstance(inner.value, str)
            and inner.value.isidentifier()
        ):
            # request.getfixturevalue("x"), usefixtures("x")
            out.add(inner.value)
    return out


_READ_METHODS = frozenset({"read_text", "read_bytes", "open"})


def _read_only_nodes(tree: ast.AST) -> set[int]:
    """Ids of the nodes that name a file only to read it (``p.read_text()``,
    ``open(p)``): text the test inspects, not a workflow it runs."""
    out: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target: ast.AST | None = None
        if isinstance(node.func, ast.Attribute) and node.func.attr in _READ_METHODS:
            target = node.func.value
        elif isinstance(node.func, ast.Name) and node.func.id == "open" and node.args:
            target = node.args[0]
        if target is not None:
            out.update(id(inner) for inner in ast.walk(target))
    return out


def _launches_nextflow(tree: ast.AST) -> bool:
    """Whether code starts Nextflow: ``shutil.which("nextflow")`` or a command
    list that starts with ``"nextflow"``."""
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "which"
            and any(
                isinstance(a, ast.Constant) and a.value == "nextflow" for a in node.args
            )
        ):
            return True
        if (
            isinstance(node, ast.List | ast.Tuple)
            and node.elts
            and isinstance(node.elts[0], ast.Constant)
            and node.elts[0].value == "nextflow"
        ):
            return True
    return False


def _is_os_path(node: ast.expr) -> bool:
    if isinstance(node, ast.Attribute):
        return node.attr == "path"
    return isinstance(node, ast.Name) and node.id in ("path", "osp")


def _parent(path: str) -> str:
    return path.rsplit("/", 1)[0] if "/" in path else ""


def _fstring_head(node: ast.JoinedStr) -> str:
    head = ""
    for value in node.values:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            head += value.value
        else:
            break
    return head


def _path_from_literal(text: str, top_level: frozenset[str]) -> str | None:
    """A repository path spelled as a whole string literal, if it is one."""
    if not text or len(text) > 300 or any(c.isspace() for c in text):
        return None
    candidate = text[2:] if text.startswith("./") else text
    first = candidate.split("/", 1)[0]
    if first in top_level:
        return _normalise(candidate)
    return None


class RepoIndex:
    """Static dependency graph of a repository checkout."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.files = self._walk()
        self.file_set = frozenset(self.files)
        self.top_level = frozenset(
            p.name for p in root.iterdir() if p.name not in _SKIP_DIRS
        )
        self.module_files = {
            self._module_name(p): p
            for p in self.files
            if p.startswith("src/merxen/") and p.endswith(".py")
        }
        self.script_files = [
            p for p in self.files if p.startswith("scripts/") and p.endswith(".py")
        ]
        self.nextflow_files = [p for p in self.files if p.endswith(".nf")]
        self.command_modules = self._command_modules()
        self.facts: dict[str, FileFacts] = {}
        self.fixtures: dict[str, dict[str, str]] = {}
        for path in self.files:
            self.facts[path] = self._analyse(path)
        self._link_fixtures()
        self.test_files = sorted(p for p in self.files if _is_test_file(p))
        self._closures: dict[str, frozenset[str]] = {}
        self._link_references()

    # -- discovery ----------------------------------------------------------

    def _walk(self) -> list[str]:
        out = []
        for top in ("src", "tests", "scripts", "workflows"):
            base = self.root / top
            if not base.is_dir():
                continue
            for dirpath, dirnames, filenames in os.walk(base):
                dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
                for name in sorted(filenames):
                    if name.endswith(_INDEXED_SUFFIXES):
                        full = Path(dirpath) / name
                        out.append(full.relative_to(self.root).as_posix())
        return out

    @staticmethod
    def _module_name(path: str) -> str:
        parts = list(PurePosixPath(path).with_suffix("").parts[1:])
        if parts[-1] == "__init__":
            parts.pop()
        return ".".join(parts)

    def _resolve_module(self, dotted: str) -> str | None:
        parts = dotted.split(".")
        for end in range(len(parts), 0, -1):
            name = ".".join(parts[:end])
            if name in self.module_files:
                return name
        return None

    def _module_edges(self, facts: FileFacts, dotted: str) -> None:
        """Edges for importing ``dotted``: the module and its parent packages.

        A parent package is imported implicitly; its own imports count too,
        except for ``merxen.cli``, whose ``__init__`` only registers every
        command (a test reaches a command's behaviour through ``main`` only by
        importing ``merxen.cli`` itself)."""
        name = self._resolve_module(dotted)
        if name is None:
            return
        facts.add_edge(self.module_files[name])
        parts = name.split(".")
        for end in range(1, len(parts)):
            parent = ".".join(parts[:end])
            if parent in self.module_files:
                facts.add_edge(self.module_files[parent], deep=parent != "merxen.cli")

    def _package_edges(self, facts: FileFacts, package: str) -> None:
        """Edges to a package and every module below it (computed imports)."""
        name = self._resolve_module(package)
        if name is None:
            return
        for module, path in self.module_files.items():
            if module == name or module.startswith(name + "."):
                facts.add_edge(path)

    def _command_modules(self) -> dict[str, str]:
        out = {}
        for path in self.files:
            if path.startswith("src/merxen/cli/") and path.endswith(".py"):
                for name in _CLICK_NAME.findall(self._read(path)):
                    out[name] = path
        return out

    def _read(self, path: str) -> str:
        try:
            return (self.root / path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    # -- per-file analysis --------------------------------------------------

    def _analyse(self, path: str) -> FileFacts:
        if path.endswith(".py"):
            return self._analyse_python(path)
        if path.endswith(".nf"):
            return self._analyse_nextflow(path)
        return FileFacts()

    def _analyse_python(self, path: str) -> FileFacts:
        text = self._read(path)
        try:
            tree = ast.parse(text, filename=path)
        except SyntaxError:
            # Unparseable: depend on the whole package so any change selects it.
            facts = FileFacts()
            facts.references.add("src/merxen")
            return facts
        context = _Context(path, tree, is_test_side=not path.startswith("src/"))
        if PurePosixPath(path).name == "conftest.py" and path.startswith("tests/"):
            facts = self._analyse_conftest(context)
        else:
            facts = FileFacts()
            facts.launches_nextflow = context.is_test_side and _launches_nextflow(tree)
            self._collect(facts, context, [tree])
        if path.startswith("tests/"):
            for conftest in self._ancestor_conftests(path):
                facts.add_edge(conftest)
        return facts

    def _ancestor_conftests(self, path: str) -> list[str]:
        """The ``conftest.py`` files whose fixtures ``path`` can use (not itself)."""
        out = []
        directory = PurePosixPath(path).parent
        while True:
            conftest = f"{directory.as_posix()}/conftest.py"
            if conftest != path and conftest in self.file_set:
                out.append(conftest)
            if directory.as_posix() in ("tests", ".", ""):
                break
            directory = directory.parent
        return out

    def _analyse_conftest(self, context: _Context) -> FileFacts:
        """Split a conftest into what every test below it needs and what only
        the tests requesting a fixture need.

        Module-level code, hooks, autouse fixtures and decorators apply to
        every test in the directory. A fixture's body (with the helpers and
        fixtures it names) applies only to the files that name the fixture;
        its node is ``<conftest>::<fixture>``."""
        defs: dict[str, ast.AST] = {}
        fixtures: dict[str, str] = {}
        shared: list[ast.AST] = []
        tree = context.tree
        assert isinstance(tree, ast.Module)
        for statement in tree.body:
            if not isinstance(
                statement, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
            ):
                shared.append(statement)
                continue
            defs[statement.name] = statement
            shared.extend(statement.decorator_list)
            name, autouse = _fixture_info(statement)
            if name is not None and not autouse:
                fixtures[name] = statement.name
            elif name is not None or statement.name.startswith("pytest_"):
                shared.append(statement)

        def expand(roots: list[ast.AST]) -> list[ast.AST]:
            nodes = list(roots)
            seen: set[str] = set()
            stack = list(roots)
            while stack:
                for name in _names(stack.pop()):
                    target = fixtures.get(name, name)
                    if target in defs and target not in seen:
                        seen.add(target)
                        nodes.append(defs[target])
                        stack.append(defs[target])
            return nodes

        facts = FileFacts()
        shared_nodes = expand(shared)
        self._collect(facts, context, shared_nodes)
        facts.launches_nextflow = any(_launches_nextflow(n) for n in shared_nodes)
        registered = self.fixtures.setdefault(context.path, {})
        for fixture, function in fixtures.items():
            node = f"{context.path}::{fixture}"
            fixture_facts = FileFacts()
            nodes = expand([defs[function]])
            self._collect(fixture_facts, context, nodes)
            fixture_facts.launches_nextflow = any(_launches_nextflow(n) for n in nodes)
            fixture_facts.add_edge(context.path)
            self.facts[node] = fixture_facts
            registered[fixture] = node
        return facts

    def _collect(
        self, facts: FileFacts, context: _Context, roots: Sequence[ast.AST]
    ) -> None:
        """Add the imports, strings and path expressions under ``roots``."""
        path = context.path
        seen: set[int] = set()
        for root in roots:
            for node in ast.walk(root):
                if id(node) in seen:
                    continue
                seen.add(id(node))
                in_read = id(node) in context.read
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self._absolute_import(facts, path, alias.name)
                elif isinstance(node, ast.ImportFrom):
                    self._from_import(facts, path, node)
                elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                    self._string(
                        facts, node.value, context.is_test_side, in_read=in_read
                    )
                elif isinstance(node, ast.JoinedStr):
                    head = _fstring_head(node)
                    match = re.fullmatch(r"(merxen(?:\.\w+)*)\.?", head)
                    if match:
                        self._package_edges(facts, match.group(1))
                    elif context.is_test_side:
                        for found in _PATH_IN_TEXT.findall(head):
                            directory = (
                                found.rsplit("/", 1)[0] if "/" in found else found
                            )
                            self._reference(facts, directory, in_read=in_read)
                if (
                    context.is_test_side
                    and isinstance(node, ast.BinOp | ast.Call)
                    and id(node) not in context.inner
                ):
                    self._path_expression(
                        facts, context.resolver, node, in_read=in_read
                    )

    def _link_fixtures(self) -> None:
        """Edges from each test-side file to the conftest fixtures it names."""
        for path in self.files:
            if not (path.startswith("tests/") and path.endswith(".py")):
                continue
            ancestors = self._ancestor_conftests(path)
            if not ancestors:
                continue
            text = self._read(path)
            words = set(re.findall(r"[A-Za-z_]\w*", text))
            for conftest in ancestors:
                for fixture, node in self.fixtures.get(conftest, {}).items():
                    if fixture in words:
                        self.facts[path].add_edge(node)

    @staticmethod
    def _reference(facts: FileFacts, reference: str | None, *, in_read: bool) -> None:
        # The repository root and the package and test roots are not
        # references: tests use them as cwd or PYTHONPATH, and the import
        # graph covers what they load from there.
        if not reference or reference in _ROOT_DIRECTORIES:
            return
        facts.references.add(reference)
        if not in_read:
            facts.run_references.add(reference)

    def _string(
        self, facts: FileFacts, value: str, is_test_side: bool, *, in_read: bool
    ) -> None:
        for found in _MODULE_IN_TEXT.findall(value):
            self._module_edges(facts, found)
        if not is_test_side:
            return
        self._reference(
            facts, _path_from_literal(value, self.top_level), in_read=in_read
        )
        for found in _PATH_IN_TEXT.findall(value):
            self._reference(facts, _normalise(found), in_read=in_read)
        if value and len(value) <= 200 and not any(c.isspace() for c in value):
            facts.tokens.add(value)
            if not in_read:
                facts.run_tokens.add(value)

    def _path_expression(
        self, facts: FileFacts, resolver: _PathResolver, node: ast.AST, *, in_read: bool
    ) -> None:
        """References from ``base / "a" / "b"``, ``joinpath`` and ``glob`` calls."""
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("glob", "rglob")
            and node.args
        ):
            pattern = node.args[0]
            if isinstance(pattern, ast.Constant) and isinstance(pattern.value, str):
                for base, prefix_only in resolver.resolve(node.func.value):
                    if prefix_only:
                        self._reference(facts, base, in_read=in_read)
                        continue
                    glob = (
                        pattern.value
                        if node.func.attr == "glob"
                        else f"**/{pattern.value}"
                    )
                    self._reference(
                        facts, f"{base}/{glob}" if base else glob, in_read=in_read
                    )
            return
        if not isinstance(node, ast.BinOp) and not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "joinpath"
        ):
            return
        for resolved, _prefix_only in resolver.resolve(node):  # type: ignore[arg-type]
            self._reference(facts, _normalise(resolved), in_read=in_read)

    def _absolute_import(self, facts: FileFacts, path: str, name: str) -> None:
        if name == "merxen" or name.startswith("merxen."):
            self._module_edges(facts, name)
            return
        self._local_import(facts, path, name, None)

    def _from_import(self, facts: FileFacts, path: str, node: ast.ImportFrom) -> None:
        module = node.module or ""
        if node.level == 0 and (module == "merxen" or module.startswith("merxen.")):
            for alias in node.names:
                if (
                    self._resolve_module(f"{module}.{alias.name}")
                    == f"{module}.{alias.name}"
                ):
                    self._module_edges(facts, f"{module}.{alias.name}")
                else:
                    self._module_edges(facts, module)
            return
        if node.level == 0:
            for alias in node.names:
                self._local_import(facts, path, module, alias.name)
            return
        base = PurePosixPath(path).parent
        for _ in range(node.level - 1):
            base = base.parent
        target = base.joinpath(*module.split(".")) if module else base
        found = False
        for alias in node.names:
            for candidate in (
                f"{target}/{alias.name}.py",
                f"{target}/{alias.name}/__init__.py",
            ):
                if candidate in self.file_set:
                    self._file_edges(facts, candidate)
                    found = True
        for candidate in (f"{target}.py", f"{target}/__init__.py"):
            if candidate in self.file_set:
                self._file_edges(facts, candidate)
                found = True
        if not found and path.startswith("src/"):
            # A name the analysis cannot place: depend on the whole package.
            facts.references.add(target.as_posix())

    def _local_import(
        self, facts: FileFacts, path: str, module: str, name: str | None
    ) -> None:
        """Imports of repository files outside the package (``tests.x``, siblings)."""
        dotted = f"{module}.{name}" if name else module
        parts = dotted.split(".")
        parent = PurePosixPath(path).parent.as_posix()
        for base in ("", parent):
            for end in range(len(parts), 0, -1):
                stem = "/".join(parts[:end])
                rel = f"{base}/{stem}" if base else stem
                for candidate in (f"{rel}.py", f"{rel}/__init__.py"):
                    if candidate in self.file_set and candidate != path:
                        self._file_edges(facts, candidate)
                        break
                else:
                    continue
                break

    def _file_edges(self, facts: FileFacts, target: str) -> None:
        if target.startswith("src/merxen/"):
            self._module_edges(facts, self._module_name(target))
        else:
            facts.add_edge(target)

    def _analyse_nextflow(self, path: str) -> FileFacts:
        facts = FileFacts()
        text = self._read(path)
        directory = PurePosixPath(path).parent.as_posix()
        for include in _NF_INCLUDE.findall(text):
            target = _normalise(f"{directory}/{include}")
            if target is None:
                continue
            for candidate in (target, f"{target}.nf"):
                if candidate in self.file_set:
                    facts.add_edge(candidate)
        for module in _NF_PYTHON_MODULE.findall(text):
            self._module_edges(facts, module)
        for command in _NF_COMMAND.findall(text):
            if command in self.command_modules:
                self._module_edges(
                    facts, self._module_name(self.command_modules[command])
                )
        for script in _NF_SCRIPT.findall(text):
            if script in self.file_set:
                facts.add_edge(script)
        return facts

    # -- reference edges (computed once all files are analysed) -------------

    def _link_references(self) -> None:
        """Turn references to scripts and workflow files into graph edges.

        A referenced Python script runs (or is imported) by the test, so its
        own imports count. A referenced workflow file counts with its own
        dependencies only for a file that launches Nextflow; for the others it
        is text the test reads, matched by :meth:`select`."""
        by_basename: dict[str, list[str]] = {}
        by_stem: dict[str, list[str]] = {}
        for path in [*self.script_files, *self.nextflow_files]:
            pure = PurePosixPath(path)
            by_basename.setdefault(pure.name, []).append(path)
            if path.endswith(".py"):
                by_stem.setdefault(pure.stem, []).append(path)
        for path, facts in self.facts.items():
            if path.startswith("src/") or not path.endswith(".py"):
                continue
            for token in facts.tokens:
                if token in self.command_modules:
                    self._module_edges(
                        facts, self._module_name(self.command_modules[token])
                    )
            scripts = self._named_files(
                facts.references, facts.tokens, by_basename, by_stem
            )
            for target in scripts:
                if (
                    target.startswith("scripts/")
                    and target.endswith(".py")
                    and target != path
                ):
                    facts.add_edge(target)
            if facts.launches_nextflow:
                runs = self._named_files(
                    facts.run_references, facts.run_tokens, by_basename, by_stem
                )
                for target in runs:
                    if target.endswith(".nf"):
                        facts.add_edge(target)

    def _named_files(
        self,
        references: set[str],
        tokens: set[str],
        by_basename: dict[str, list[str]],
        by_stem: dict[str, list[str]],
    ) -> set[str]:
        """Scripts and workflow files named by path, base name, stem or path suffix."""
        targets = {
            r for r in references if r in self.file_set and r.endswith((".py", ".nf"))
        }
        for token in tokens:
            if len(token) < _MIN_NAME_LENGTH or token in _GENERIC_BASENAMES:
                continue
            targets.update(by_basename.get(token, ()))
            targets.update(by_stem.get(token, ()))
            if "/" in token and not token.startswith("/"):
                suffix = "/" + token.lstrip("./")
                targets.update(p for p in self.nextflow_files if p.endswith(suffix))
                targets.update(p for p in self.script_files if p.endswith(suffix))
        return targets

    # -- closures -----------------------------------------------------------

    def closure(self, path: str) -> frozenset[str]:
        """Every file ``path`` depends on, itself included."""
        cached = self._closures.get(path)
        if cached is not None:
            return cached
        seen = {path}
        stack = [path]
        while stack:
            current = stack.pop()
            for target, deep in self.facts.get(current, FileFacts()).edges.items():
                if target in seen:
                    continue
                seen.add(target)
                if deep:
                    stack.append(target)
        result = frozenset(seen)
        self._closures[path] = result
        return result

    def test_profile(
        self, test: str
    ) -> tuple[frozenset[str], set[str], set[str], bool]:
        """(closure, references, tokens, launches Nextflow) of one test file."""
        closure = self.closure(test)
        references: set[str] = set()
        tokens: set[str] = set()
        launches = False
        for path in closure:
            if path.startswith("src/"):
                continue
            facts = self.facts.get(path)
            if facts is None:
                continue
            references |= facts.references
            tokens |= facts.tokens
            launches = launches or facts.launches_nextflow
        return closure, references, tokens, launches


# ---------------------------------------------------------------------------
# Selection.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Change:
    """One changed path and its git status letter (A, M, D, ...)."""

    path: str
    status: str = "M"


@dataclass
class Plan:
    """The selector's answer."""

    mode: str  # "full" or "selected"
    reason: str
    tests: list[str] = field(default_factory=list)
    shards: int = FULL_SHARDS
    est_seconds: float = 0.0
    total_seconds: float = 0.0
    changed: list[str] = field(default_factory=list)
    why: dict[str, list[str]] = field(default_factory=dict)

    def as_json(self) -> dict[str, object]:
        """The plan as plain JSON data."""
        return {
            "mode": self.mode,
            "reason": self.reason,
            "shards": self.shards,
            "est_seconds": round(self.est_seconds, 1),
            "total_seconds": round(self.total_seconds, 1),
            "changed": self.changed,
            "tests": self.tests,
            "why": self.why,
        }


def forced_full_reason(
    *,
    event: str | None,
    labels: Sequence[str],
    base_ref: str | None,
    head_ref: str | None,
    head_repo: str | None,
    repo: str | None,
) -> str | None:
    """Why the event alone requires the full suite, or None."""
    if event in FULL_EVENTS:
        return f"event {event} runs the full suite"
    if FULL_LABEL in labels:
        return f"label {FULL_LABEL}"
    if (
        base_ref == "main"
        and head_ref in INTEGRATION_BRANCHES
        and head_repo
        and head_repo == repo
    ):
        return f"integration branch {head_ref} into main"
    return None


def file_key(path: str) -> str:
    """The digest that stands for a test file in a durations file (the part of
    a key before ``:``; scripts/ci/pytest_shard.py writes the keys)."""
    return hashlib.sha256(path.encode()).hexdigest()[:12]


def file_seconds(durations: dict[str, float]) -> dict[str, float]:
    """Summed recorded seconds per test file digest (see :func:`file_key`)."""
    out: dict[str, float] = {}
    for key, seconds in durations.items():
        digest = key.split(":", 1)[0]
        out[digest] = out.get(digest, 0.0) + float(seconds)
    return out


def _expand(selection: Iterable[str], test_files: Sequence[str]) -> set[str]:
    out: set[str] = set()
    for entry in selection:
        out.update(t for t in test_files if reference_matches(t, entry))
    return out


def _depends_on(
    path: str,
    closure: frozenset[str],
    references: set[str],
    tokens: set[str],
    name_hit: bool,
    stem_hit: bool,
) -> bool:
    """Whether a test with this profile depends on the repository file ``path``."""
    pure = PurePosixPath(path)
    if path in closure:
        return True
    if any(reference_matches(path, r) for r in references):
        return True
    if (name_hit and pure.name in tokens) or (stem_hit and pure.stem in tokens):
        return True
    return any(
        "/" in t and not t.startswith("/") and path.endswith("/" + t.lstrip("./"))
        for t in tokens
    )


def select(
    index: RepoIndex,
    changes: Sequence[Change],
    durations: dict[str, float],
    *,
    guards: Sequence[str] = GUARDS,
) -> Plan:
    """Map changed files to the test files that can see them."""
    changed = sorted({c.path for c in changes})
    per_file = file_seconds(durations)
    test_files = index.test_files

    def cost(files: Iterable[str]) -> float:
        return sum(per_file.get(file_key(f), UNKNOWN_FILE_SECONDS) for f in files)

    total = cost(test_files)

    def full(reason: str) -> Plan:
        return Plan(
            "full",
            reason,
            shards=FULL_SHARDS,
            est_seconds=total,
            total_seconds=total,
            changed=changed,
        )

    for change in changes:
        path = change.path
        if any(glob_match(path, p) for p in FULL_PATTERNS):
            return full(f"{path} affects every test")
        if change.status.startswith("D") and (
            path.startswith("src/")
            or (
                path.startswith("tests/")
                and path.endswith(".py")
                and not _is_test_file(path)
            )
        ):
            return full(f"{path} was deleted")
        if path.startswith("src/") and not (
            path.startswith("src/merxen/") and path.endswith(".py")
        ):
            return full(f"{path} is not a Python module of the package")

    profiles = {t: index.test_profile(t) for t in test_files}
    launchers = {t for t, profile in profiles.items() if profile[3]}
    why: dict[str, set[str]] = {}

    def hit(test: str, path: str) -> None:
        why.setdefault(test, set()).add(path)

    for change in changes:
        path = change.path
        pure = PurePosixPath(path)
        mapped = False
        if path.startswith("src/merxen/"):
            mapped = True
        elif path.startswith("tests/"):
            if _is_test_file(path):
                mapped = True
                if path in profiles:
                    hit(path, path)
            elif path.endswith(".py"):
                mapped = True
            elif len(pure.parts) > 2:
                mapped = True
                prefix = "/".join(pure.parts[:2]) + "/"
                for test in test_files:
                    if test.startswith(prefix):
                        hit(test, path)
        elif path.startswith("workflows/") or pure.name == "nextflow.config":
            mapped = True
            for test in test_files:
                if test.startswith("tests/test_workflows/") or test in launchers:
                    hit(test, path)
        elif path.startswith("scripts/"):
            mapped = True
            for test in test_files:
                if test.startswith("tests/test_scripts/"):
                    hit(test, path)
        elif any(glob_match(path, p) for p in REFERENCE_ONLY_PATTERNS):
            mapped = True
        if not mapped:
            return full(f"{path} has no mapping to tests")

        name_hit = (
            len(pure.name) >= _MIN_NAME_LENGTH and pure.name not in _GENERIC_BASENAMES
        )
        stem_hit = path.startswith("scripts/") and len(pure.stem) >= _MIN_NAME_LENGTH
        for test, (closure, references, tokens, _launches) in profiles.items():
            if _depends_on(path, closure, references, tokens, name_hit, stem_hit):
                hit(test, path)

    guard_files = _expand(guards, test_files)
    selected = guard_files | set(why)
    missing = [g for g in guards if not (index.root / g).exists()]
    for entry in missing:
        print(f"select_tests: guard {entry} does not exist", file=sys.stderr)
    estimate = cost(selected)
    if total and estimate > CAP * total:
        plan = full(f"selection is {estimate / total:.0%} of the suite (cap {CAP:.0%})")
        plan.why = {t: sorted(v) for t, v in sorted(why.items())}
        return plan
    shards = max(1, min(FULL_SHARDS, math.ceil(estimate / SELECTED_SHARD_SECONDS)))
    reason = f"{len(changed)} changed file(s) select {len(selected)} test file(s)"
    return Plan(
        "selected",
        reason,
        tests=sorted(selected),
        shards=shards,
        est_seconds=estimate,
        total_seconds=total,
        changed=changed,
        why={t: sorted(v) for t, v in sorted(why.items())},
    )


# ---------------------------------------------------------------------------
# Git and command line.
# ---------------------------------------------------------------------------


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout


def git_changes(root: Path, base: str, head: str, *, worktree: bool) -> list[Change]:
    """Changed paths of ``base...head`` (and the working tree with ``worktree``)."""
    changes: dict[str, str] = {}
    out = _git(
        root,
        "-c",
        "core.quotepath=off",
        "diff",
        "--name-status",
        "--no-renames",
        f"{base}...{head}",
    )
    for line in out.splitlines():
        status, _, path = line.partition("\t")
        if path:
            changes[path] = status
    if worktree:
        out = _git(
            root,
            "-c",
            "core.quotepath=off",
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--no-renames",
        )
        for entry in out.split("\0"):
            if len(entry) < 4:
                continue
            status, path = entry[:2], entry[3:]
            letter = "D" if "D" in status else ("A" if "?" in status else "M")
            # A file deleted in the working tree but changed in the branch
            # is deleted; anything else keeps the branch status.
            if letter == "D" or path not in changes:
                changes[path] = letter
    return [Change(p, s) for p, s in sorted(changes.items())]


def load_durations(path: Path | None) -> dict[str, float]:
    """Recorded seconds per test (``file::digest`` keys; empty when absent)."""
    if path is None or not path.is_file():
        return {}
    data = json.loads(path.read_text())
    return {str(k): float(v) for k, v in data.items()}


def _write_github(plan: Plan) -> None:
    output = os.environ.get("GITHUB_OUTPUT")
    lines = [
        f"mode={plan.mode}",
        f"reason={plan.reason}",
        f"count={plan.shards}",
        f"shards={json.dumps(list(range(1, plan.shards + 1)))}",
        f"est_seconds={round(plan.est_seconds)}",
    ]
    if output:
        with open(output, "a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
    else:
        print("\n".join(lines))


def _summary(plan: Plan) -> str:
    rows = [
        "### Test plan",
        "",
        f"- Mode: **{plan.mode}** ({plan.reason})",
        f"- Shards: {plan.shards}",
        f"- Estimated worker time: {plan.est_seconds:.0f} s"
        f" of {plan.total_seconds:.0f} s",
    ]
    if plan.changed:
        rows.append(f"- Changed files: {len(plan.changed)}")
    if plan.mode == "selected":
        rows += ["", "<details><summary>Selected test files</summary>", "", "```"]
        rows += plan.tests
        rows += ["```", "", "</details>"]
    return "\n".join(rows) + "\n"


def _labels(value: str | None) -> list[str]:
    if not value:
        return []
    try:
        data = json.loads(value)
    except json.JSONDecodeError:
        return [v.strip() for v in value.split(",") if v.strip()]
    if not isinstance(data, list):
        return []
    return [str(v) for v in data]


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--base", help="base commit (the diff is base...head)")
    parser.add_argument("--head", default="HEAD")
    parser.add_argument(
        "--worktree",
        action="store_true",
        help="also count uncommitted and untracked files",
    )
    parser.add_argument(
        "--changed",
        action="append",
        default=[],
        help="a changed path (instead of git); repeatable",
    )
    parser.add_argument("--event")
    parser.add_argument("--base-ref")
    parser.add_argument("--head-ref")
    parser.add_argument("--head-repo")
    parser.add_argument("--repo")
    parser.add_argument("--labels", help="JSON list (or comma-separated) PR labels")
    parser.add_argument(
        "--durations", type=Path, help="durations file (scripts/ci/test_durations.json)"
    )
    parser.add_argument("--format", choices=("json", "paths", "github"), default="json")
    parser.add_argument(
        "--selection-file",
        type=Path,
        help="write the selected paths here, one per line",
    )
    parser.add_argument(
        "--summary", type=Path, help="append a Markdown summary to this file"
    )
    args = parser.parse_args(argv)

    root = args.root.resolve()
    durations = load_durations(
        args.durations or root / "scripts" / "ci" / "test_durations.json"
    )
    reason = forced_full_reason(
        event=args.event,
        labels=_labels(args.labels),
        base_ref=args.base_ref,
        head_ref=args.head_ref,
        head_repo=args.head_repo,
        repo=args.repo,
    )
    plan: Plan
    if reason is not None:
        total = sum(file_seconds(durations).values())
        plan = Plan("full", reason, est_seconds=total, total_seconds=total)
    else:
        changes: list[Change] | None = None
        if args.changed:
            changes = [Change(p) for p in args.changed]
        elif args.base:
            try:
                changes = git_changes(
                    root, args.base, args.head, worktree=args.worktree
                )
            except subprocess.CalledProcessError as error:
                reason = f"git diff failed: {error.stderr.strip() or error}"
        else:
            reason = "no --base to diff against"
        if changes is None:
            total = sum(file_seconds(durations).values())
            plan = Plan(
                "full",
                reason or "no changes known",
                est_seconds=total,
                total_seconds=total,
            )
        else:
            plan = select(RepoIndex(root), changes, durations)

    if plan.mode == "selected":
        kept = [t for t in plan.tests if (root / t).exists()]
        for dropped in sorted(set(plan.tests) - set(kept)):
            print(f"select_tests: {dropped} does not exist; dropped", file=sys.stderr)
        plan.tests = kept or [g for g in GUARDS if (root / g).exists()]

    if args.selection_file is not None:
        args.selection_file.parent.mkdir(parents=True, exist_ok=True)
        args.selection_file.write_text("".join(f"{t}\n" for t in plan.tests))
    if args.summary is not None:
        with open(args.summary, "a", encoding="utf-8") as handle:
            handle.write(_summary(plan))

    print(f"select_tests: {plan.mode}: {plan.reason}", file=sys.stderr)
    if args.format == "json":
        print(json.dumps(plan.as_json(), indent=1))
    elif args.format == "paths":
        print("FULL" if plan.mode == "full" else "\n".join(plan.tests))
    else:
        _write_github(plan)
    return 0


if __name__ == "__main__":
    sys.exit(main())

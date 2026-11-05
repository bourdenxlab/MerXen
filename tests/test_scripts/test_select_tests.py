"""Tests for the change-aware test selector (scripts/ci/select_tests.py)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

from ._ci_scripts import CI_DIR, load_ci_script

sel: ModuleType = load_ci_script("select_tests")
check: ModuleType = load_ci_script("check_shards")


def _keyed(durations: dict[str, float]) -> dict[str, float]:
    """Durations by node id, keyed as in a durations file."""
    return {check.duration_key(n): v for n, v in durations.items()}


# A small repository: src/merxen with a CLI, tests with helpers and a
# conftest, scripts, workflows and docs.
TREE: dict[str, str] = {
    "pyproject.toml": "",
    "README.md": "",
    "docs/guide.md": "",
    "docs/images/map.png": "",
    "assets/map.mmd": "",
    "src/merxen/__init__.py": "",
    "src/merxen/config.py": "",
    "src/merxen/core.py": "VALUE = 1\n",
    "src/merxen/leaf.py": "from merxen.core import VALUE\n",
    "src/merxen/mid.py": "from merxen import leaf\n",
    "src/merxen/lonely.py": "",
    "src/merxen/typed.py": (
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n    from merxen.lonely import X\n"
    ),
    "src/merxen/pkg/__init__.py": "from merxen.pkg import registry\n",
    "src/merxen/pkg/registry.py": "",
    "src/merxen/pkg/user.py": "from .helper import f\n",
    "src/merxen/pkg/helper.py": "",
    "src/merxen/dyn/__init__.py": "",
    "src/merxen/dyn/a.py": "",
    "src/merxen/dyn/b.py": "",
    "src/merxen/stringly.py": "",
    "src/merxen/gstage.py": "",
    "src/merxen/cli/__init__.py": (
        "from merxen.cli.run_one import one\nfrom merxen.cli.run_two import two\n"
    ),
    "src/merxen/cli/run_one.py": '@click.command(name="do-one")\ndef one(): ...\n',
    "src/merxen/cli/run_two.py": '@click.command(name="do-two")\ndef two(): ...\n',
    "src/merxen/cli/run_three.py": (
        '@click.command(name="do-three")\ndef three(): ...\n'
    ),
    "scripts/tools/my_tool.py": "import merxen.lonely\n",
    "scripts/other.py": "",
    "workflows/main.nf": 'include { ONE } from "./modules/one"\n',
    "workflows/modules/one.nf": (
        "process ONE {\n  script:\n  '''\n  merxen do-one --x 1\n"
        "  python -m merxen.gstage\n  '''\n}\n"
    ),
    "workflows/lib/Spec.groovy": "",
    "workflows/nextflow.config": "",
    "tests/conftest.py": "",
    "tests/test_core.py": "from merxen.core import VALUE\n",
    "tests/test_mid.py": "import merxen.mid\n",
    "tests/test_typed.py": "import merxen.typed\n",
    "tests/test_pkg.py": "from merxen.pkg.user import f\n",
    "tests/test_dyn.py": (
        "import importlib\n"
        "def test(name):\n    importlib.import_module(f'merxen.dyn.{name}')\n"
    ),
    "tests/test_stringly.py": (
        "def test(monkeypatch):\n    monkeypatch.setattr('merxen.stringly.thing', 1)\n"
    ),
    "tests/test_cli_one.py": "from merxen.cli.run_one import one\n",
    "tests/test_cli_main.py": "from merxen.cli import main\n",
    "tests/test_cli_subprocess.py": (
        "import subprocess\ndef test():\n    subprocess.run(['merxen', 'do-three'])\n"
    ),
    "tests/test_unrelated.py": "",
    "tests/test_guard.py": "",
    "tests/test_area/__init__.py": "",
    "tests/test_area/conftest.py": "import merxen.lonely\n",
    "tests/test_area/helpers.py": "from merxen.leaf import VALUE\n",
    "tests/test_area/test_uses_helper.py": "from .helpers import VALUE\n",
    "tests/test_area/test_plain.py": "",
    "tests/test_area/golden/out.json": "",
    "tests/test_fx/conftest.py": (
        "import pytest\n"
        "def _helper():\n    from merxen.leaf import VALUE\n    return VALUE\n"
        "@pytest.fixture\ndef widget():\n    return _helper()\n"
        "@pytest.fixture\ndef other(widget):\n    return widget\n"
        "@pytest.fixture(autouse=True)\ndef _always():\n    import merxen.typed\n"
    ),
    "tests/test_fx/test_uses_widget.py": "def test(widget): pass\n",
    "tests/test_fx/test_uses_other.py": "def test(other): pass\n",
    "tests/test_fx/test_plain.py": "def test(): pass\n",
    "tests/test_refs/test_chain.py": (
        "from pathlib import Path\n"
        "REPO_ROOT = Path(__file__).resolve().parents[2]\n"
        "DOCS = REPO_ROOT / 'docs'\n"
        "def test(name):\n"
        "    (DOCS / 'guide.md').read_text()\n"
        "    (REPO_ROOT / 'docs' / 'images' / name).read_bytes()\n"
    ),
    "tests/test_refs/test_message.py": (
        "def test():\n    raise AssertionError('regenerate with docs/guide.md first')\n"
    ),
    "tests/test_refs/test_glob.py": (
        "from pathlib import Path\n"
        "ROOT = Path(__file__).parents[2]\n"
        "def test():\n    list(ROOT.glob('assets/*.mmd'))\n"
    ),
    "tests/test_refs/test_basename.py": "NAME = 'guide.md'\n",
    "tests/test_refs/test_stem.py": (
        "from tests.test_refs._loader import load\nTOOL = load('my_tool')\n"
    ),
    "tests/test_refs/_loader.py": (
        "from pathlib import Path\n"
        "TOOLS = Path(__file__).resolve().parents[2] / 'scripts' / 'tools'\n"
        "def load(name):\n    return TOOLS / f'{name}.py'\n"
    ),
    "tests/test_workflows/test_text_only.py": (
        "from pathlib import Path\n"
        "WF = Path(__file__).resolve().parents[2] / 'workflows'\n"
        "def test():\n    (WF / 'main.nf').read_text()\n"
    ),
    "tests/test_nf/test_launch.py": (
        "import shutil, subprocess\nfrom pathlib import Path\n"
        "NEXTFLOW = shutil.which('nextflow')\n"
        "WF = Path(__file__).resolve().parents[2] / 'workflows'\n"
        "def test():\n    subprocess.run([NEXTFLOW, 'run', str(WF / 'main.nf')])\n"
    ),
    "tests/test_nf/test_launch_reads.py": (
        "import subprocess\nfrom pathlib import Path\n"
        "WF = Path(__file__).resolve().parents[2] / 'workflows'\n"
        "def test():\n"
        "    assert 'ONE' in (WF / 'main.nf').read_text()\n"
        "    subprocess.run(['nextflow', 'run', 'harness.nf'])\n"
    ),
    "tests/test_nf/test_reads_main.py": (
        "from pathlib import Path\n"
        "def test():\n    Path('workflows/main.nf').read_text()\n"
    ),
    "tests/test_scripts/test_other.py": "",
    # Test packages that import each other (pytest puts tests/ on sys.path).
    "tests/test_pkga/__init__.py": "",
    "tests/test_pkga/_shared.py": "def helper(): ...\n",
    "tests/test_pkga/dynhelp.py": "",
    "tests/test_pkga/test_inner.py": "from ._shared import helper\n",
    "tests/test_pkgb/__init__.py": "",
    "tests/test_pkgb/test_cross.py": (
        "from test_pkga._shared import helper\nfrom test_pkga.test_inner import x\n"
    ),
    "tests/test_pkgb/test_dyn_helper.py": (
        "import importlib\n"
        "def test(name):\n"
        "    importlib.import_module('test_pkga.' + name)  # dynhelp\n"
    ),
    # A test scanning a package's source tree through the module's __file__.
    "src/merxen/scan/__init__.py": "",
    "src/merxen/scan/a.py": "",
    "tests/test_scan.py": (
        "from pathlib import Path\n"
        "import merxen.scan as scan_pkg\n"
        "PKG = Path(scan_pkg.__file__).resolve().parent\n"
        "def test():\n    list(PKG.rglob('*.py'))\n"
    ),
    # Imports inside code a test runs with python -c, after sys.path.insert.
    "src/merxen/cs/__init__.py": "",
    "src/merxen/cs/inner.py": "",
    "tests/cs_synth.py": "",
    "tests/test_codestr.py": (
        "import subprocess, sys\nfrom pathlib import Path\n"
        "HERE = Path(__file__).resolve().parent\n"
        "def test():\n"
        '    code = f"""\nimport sys\nsys.path.insert(0, {str(HERE)!r})\n'
        "from merxen.cs import inner\nimport cs_synth\nassert {1} == 1\n"
        '"""\n'
        "    subprocess.run([sys.executable, '-c', code])\n"
    ),
    # A directory added to sys.path by a path expression.
    "tests/helpers_dir/sp_helper.py": "",
    "tests/test_syspath/test_sp.py": (
        "import sys\nfrom pathlib import Path\n"
        "sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'helpers_dir'))\n"
        "import sp_helper\n"
    ),
    # A fixture launches Nextflow on the workflow the test names.
    "tests/test_nfx/conftest.py": (
        "import pytest, subprocess\n"
        "@pytest.fixture\ndef run_nf():\n"
        "    def run(wf):\n        subprocess.run(['nextflow', 'run', wf])\n"
        "    return run\n"
    ),
    "tests/test_nfx/test_via_fixture.py": (
        "def test(run_nf):\n    run_nf('workflows/main.nf')\n"
    ),
}
GUARDS = ("tests/test_guard.py",)
ALWAYS = set(GUARDS)


@pytest.fixture(scope="module")
def repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("repo")
    for path, text in TREE.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(text)
    return root


def _select(
    repo: Path,
    *paths: str,
    status: str = "M",
    durations: dict[str, float] | None = None,
) -> object:
    changes = [sel.Change(p, status) for p in paths]
    return sel.select(sel.RepoIndex(repo), changes, durations or {}, guards=GUARDS)


def _tests(repo: Path, *paths: str, status: str = "M") -> set[str]:
    plan = _select(repo, *paths, status=status)
    assert plan.mode == "selected", plan.reason
    return set(plan.tests) - ALWAYS


def test_transitive_imports_select_every_importer(repo: Path) -> None:
    assert _tests(repo, "src/merxen/core.py") == {
        "tests/test_core.py",
        "tests/test_mid.py",  # mid -> leaf -> core
        "tests/test_area/test_uses_helper.py",  # helper -> leaf -> core
        "tests/test_fx/test_uses_widget.py",  # fixture -> _helper -> leaf -> core
        "tests/test_fx/test_uses_other.py",  # fixture -> fixture -> ...
    }


def test_type_checking_imports_count(repo: Path) -> None:
    assert "tests/test_typed.py" in _tests(repo, "src/merxen/lonely.py")


def test_parent_package_and_its_imports_count(repo: Path) -> None:
    assert _tests(repo, "src/merxen/pkg/__init__.py") == {"tests/test_pkg.py"}
    # merxen.pkg.user implies merxen.pkg, whose __init__ imports registry.
    assert _tests(repo, "src/merxen/pkg/registry.py") == {"tests/test_pkg.py"}


def test_relative_imports_in_the_package(repo: Path) -> None:
    assert _tests(repo, "src/merxen/pkg/helper.py") == {"tests/test_pkg.py"}


def test_cli_package_is_shallow_for_command_modules(repo: Path) -> None:
    # Importing one command does not depend on the other commands...
    assert "tests/test_cli_one.py" not in _tests(repo, "src/merxen/cli/run_two.py")
    # ...but importing the command group does.
    assert "tests/test_cli_main.py" in _tests(repo, "src/merxen/cli/run_two.py")
    assert {"tests/test_cli_one.py", "tests/test_cli_main.py"} <= _tests(
        repo, "src/merxen/cli/__init__.py"
    )


def test_command_names_in_tests_select_the_command_module(repo: Path) -> None:
    assert "tests/test_cli_subprocess.py" in _tests(repo, "src/merxen/cli/run_three.py")


def test_computed_and_string_imports(repo: Path) -> None:
    assert "tests/test_dyn.py" in _tests(repo, "src/merxen/dyn/b.py")
    assert "tests/test_stringly.py" in _tests(repo, "src/merxen/stringly.py")


def test_conftest_imports_apply_to_its_directory(repo: Path) -> None:
    selected = _tests(repo, "src/merxen/lonely.py")
    assert {
        "tests/test_area/test_plain.py",
        "tests/test_area/test_uses_helper.py",
    } <= selected
    assert "tests/test_core.py" not in selected


def test_conftest_fixture_imports_apply_to_the_tests_that_request_it(
    repo: Path,
) -> None:
    selected = _tests(repo, "src/merxen/leaf.py")
    assert {
        "tests/test_fx/test_uses_widget.py",
        "tests/test_fx/test_uses_other.py",
    } <= selected
    assert "tests/test_fx/test_plain.py" not in selected
    # An autouse fixture applies to every test below its conftest.
    assert {
        "tests/test_fx/test_uses_widget.py",
        "tests/test_fx/test_plain.py",
    } <= _tests(repo, "src/merxen/typed.py")


def test_helper_change_selects_its_importers(repo: Path) -> None:
    assert _tests(repo, "tests/test_area/helpers.py") == {
        "tests/test_area/test_uses_helper.py"
    }


def test_changed_and_new_test_files_select_themselves(repo: Path) -> None:
    assert _tests(repo, "tests/test_unrelated.py") == {"tests/test_unrelated.py"}
    (repo / "tests" / "test_brand_new.py").write_text("")
    try:
        assert _tests(repo, "tests/test_brand_new.py", status="A") == {
            "tests/test_brand_new.py"
        }
    finally:
        (repo / "tests" / "test_brand_new.py").unlink()


def test_imports_from_another_test_package(repo: Path) -> None:
    index = sel.RepoIndex(repo)
    closure = index.closure("tests/test_pkgb/test_cross.py")
    assert {
        "tests/test_pkga/_shared.py",
        "tests/test_pkga/test_inner.py",
    } <= closure
    assert "tests/test_pkgb/test_cross.py" in _tests(repo, "tests/test_pkga/_shared.py")
    assert "tests/test_pkgb/test_cross.py" in _tests(
        repo, "tests/test_pkga/test_inner.py"
    )


def test_a_helper_named_but_not_imported_statically_is_selected(repo: Path) -> None:
    # importlib with a computed name: only the safety net (a file naming the
    # helper's module stem) links them.
    index = sel.RepoIndex(repo)
    assert "tests/test_pkga/dynhelp.py" not in index.closure(
        "tests/test_pkgb/test_dyn_helper.py"
    )
    assert "tests/test_pkgb/test_dyn_helper.py" in _tests(
        repo, "tests/test_pkga/dynhelp.py"
    )


def test_a_test_package_init_selects_its_tests(repo: Path) -> None:
    selected = _tests(repo, "tests/test_pkga/__init__.py")
    assert "tests/test_pkga/test_inner.py" in selected
    # test_cross imports test_pkga._shared, which imports the package first.
    assert "tests/test_pkgb/test_cross.py" in selected
    assert "tests/test_core.py" not in selected


def test_a_module_scanned_through_its_package_file(repo: Path) -> None:
    (repo / "src/merxen/scan/extra.py").write_text("import torch\n")
    try:
        assert "tests/test_scan.py" in _tests(
            repo, "src/merxen/scan/extra.py", status="A"
        )
    finally:
        (repo / "src/merxen/scan/extra.py").unlink()
    assert "tests/test_scan.py" in _tests(repo, "src/merxen/scan/a.py")


def test_imports_in_code_strings(repo: Path) -> None:
    assert "tests/test_codestr.py" in _tests(repo, "src/merxen/cs/inner.py")
    assert "tests/test_codestr.py" in _tests(repo, "tests/cs_synth.py")
    assert "tests/helpers_dir/sp_helper.py" in sel.RepoIndex(repo).closure(
        "tests/test_syspath/test_sp.py"
    )


def test_code_imports_parses_each_import_line() -> None:
    code = (
        "import numpy as np\n"
        "x = {broken\n"
        "from merxen.io import (\n    a,\n    b,\n)\n"
        "import the package first\n"
    )
    found = sel._code_imports(code)
    assert [type(n).__name__ for n in found] == ["Import", "ImportFrom"]
    assert [a.name for a in found[1].names] == ["a", "b"]
    assert sel._code_imports("no imports here") == []


def test_a_fixture_that_launches_nextflow(repo: Path) -> None:
    assert "tests/test_nfx/test_via_fixture.py" in _tests(
        repo, "src/merxen/cli/run_one.py"
    )


def test_test_data_selects_its_directory(repo: Path) -> None:
    assert _tests(repo, "tests/test_area/golden/out.json") == {
        "tests/test_area/test_plain.py",
        "tests/test_area/test_uses_helper.py",
    }


def test_path_references(repo: Path) -> None:
    selected = _tests(repo, "docs/guide.md")
    assert {
        "tests/test_refs/test_chain.py",  # DOCS / "guide.md"
        "tests/test_refs/test_message.py",  # substring of a message
        "tests/test_refs/test_basename.py",  # base name literal
    } <= selected
    assert "tests/test_refs/test_glob.py" not in selected
    # ROOT / "docs" / "images" / name names the whole directory.
    assert "tests/test_refs/test_chain.py" in _tests(repo, "docs/images/other.png")
    assert _tests(repo, "assets/map.mmd") == {"tests/test_refs/test_glob.py"}


def test_scripts_reach_tests_through_stems_and_imports(repo: Path) -> None:
    assert "tests/test_refs/test_stem.py" in _tests(repo, "scripts/tools/my_tool.py")
    # The script imports merxen.lonely, so the test that loads it depends on it.
    assert "tests/test_refs/test_stem.py" in _tests(repo, "src/merxen/lonely.py")
    assert "tests/test_scripts/test_other.py" in _tests(repo, "scripts/other.py")


def test_nextflow_bridge(repo: Path) -> None:
    # The launcher runs main.nf, which includes a module calling `merxen
    # do-one` and `python -m merxen.gstage`.
    for module in ("src/merxen/cli/run_one.py", "src/merxen/gstage.py"):
        selected = _tests(repo, module)
        assert "tests/test_nf/test_launch.py" in selected
        # Reading the workflow text is not running it, even in a test that
        # runs another workflow.
        assert "tests/test_nf/test_reads_main.py" not in selected
        assert "tests/test_nf/test_launch_reads.py" not in selected
        assert "tests/test_workflows/test_text_only.py" not in selected


def test_workflow_changes_select_workflow_tests_and_launchers(repo: Path) -> None:
    for path in ("workflows/lib/Spec.groovy", "workflows/nextflow.config"):
        selected = _tests(repo, path)
        assert {
            "tests/test_workflows/test_text_only.py",
            "tests/test_nf/test_launch.py",
        } <= selected
    assert "tests/test_nf/test_reads_main.py" in _tests(repo, "workflows/main.nf")


def test_docs_only_change_runs_the_guards(repo: Path) -> None:
    plan = _select(repo, "README.md")
    assert plan.mode == "selected"
    assert set(plan.tests) == ALWAYS
    assert plan.shards == 1


@pytest.mark.parametrize(
    "path",
    [
        "tests/conftest.py",
        "tests/test_area/conftest.py",
        "pyproject.toml",
        "requirements/requirements.lock",
        "envs/environment.yml",
        ".github/workflows/ci.yml",
        "scripts/run_ci_checks.sh",
        "scripts/ci/select_tests.py",
        "src/merxen/config.py",
        "src/merxen/__init__.py",
        "src/merxen/io/spatialdata_io.py",
        "src/merxen/data.json",
        "something/unknown.txt",
        "Makefile",
    ],
)
def test_full_triggers(repo: Path, path: str) -> None:
    assert _select(repo, path).mode == "full"


def test_deleted_package_module_or_helper_runs_everything(repo: Path) -> None:
    assert _select(repo, "src/merxen/lonely.py", status="D").mode == "full"
    assert _select(repo, "tests/test_area/helpers.py", status="D").mode == "full"
    # A deleted test file has nothing left to run.
    assert _select(repo, "tests/test_gone.py", status="D").mode == "selected"


def test_large_selection_runs_everything(repo: Path) -> None:
    durations = {f"{t}::test": 1.0 for t in sel.RepoIndex(repo).test_files}
    durations["tests/test_core.py::test"] = 1000.0
    plan = _select(repo, "src/merxen/core.py", durations=_keyed(durations))
    assert plan.mode == "full"
    assert "cap" in plan.reason


def test_shard_count_grows_with_the_estimate(repo: Path) -> None:
    durations = {
        "tests/test_core.py::test": 1300.0,
        "tests/test_unrelated.py::test": 9000.0,
    }
    plan = _select(repo, "tests/test_core.py", durations=_keyed(durations))
    assert plan.mode == "selected"
    assert plan.shards == 3  # ceil((1300 + 30) / 600)


@pytest.mark.parametrize(
    ("kwargs", "full"),
    [
        ({"event": "push"}, True),
        ({"event": "schedule"}, True),
        ({"event": "workflow_dispatch"}, True),
        ({"event": "pull_request"}, False),
        ({"event": "pull_request", "labels": ["full-ci"]}, True),
        ({"event": "pull_request", "labels": ["docs"]}, False),
        (
            {
                "event": "pull_request",
                "base_ref": "main",
                "head_ref": "feature/atera-support",
                "head_repo": "org/repo",
                "repo": "org/repo",
            },
            True,
        ),
        (  # a fork's branch of the same name
            {
                "event": "pull_request",
                "base_ref": "main",
                "head_ref": "feature/atera-support",
                "head_repo": "someone/repo",
                "repo": "org/repo",
            },
            False,
        ),
        (  # into the integration branch, not into main
            {
                "event": "pull_request",
                "base_ref": "feature/atera-support",
                "head_ref": "feature/atera-x",
                "head_repo": "org/repo",
                "repo": "org/repo",
            },
            False,
        ),
    ],
)
def test_forced_full_reason(kwargs: dict, full: bool) -> None:
    arguments = {
        "event": None,
        "labels": [],
        "base_ref": None,
        "head_ref": None,
        "head_repo": None,
        "repo": None,
        **kwargs,
    }
    assert (sel.forced_full_reason(**arguments) is not None) is full


def test_every_integration_branch_forces_full() -> None:
    for branch in sel.INTEGRATION_BRANCHES:
        assert sel.forced_full_reason(
            event="pull_request",
            labels=[],
            base_ref="main",
            head_ref=branch,
            head_repo="o/r",
            repo="o/r",
        )


@pytest.mark.parametrize(
    ("path", "pattern", "matches"),
    [
        ("tests/conftest.py", "**/conftest.py", True),
        ("conftest.py", "**/conftest.py", True),
        ("README.md", "*.md", True),
        ("docs/a.md", "*.md", False),
        ("docs/a/b.md", "docs/**", True),
        ("workflows/modules/x.nf", "workflows/**/*.nf", True),
        ("workflows/x.nf", "workflows/**/*.nf", True),
    ],
)
def test_glob_match(path: str, pattern: str, matches: bool) -> None:
    assert sel.glob_match(path, pattern) is matches


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    root = tmp_path / "git"
    root.mkdir()
    for path, text in TREE.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(text)
    _git(root, "init", "-q", "-b", "main")
    _git(root, "-c", "user.name=t", "-c", "user.email=t@example.org", "add", ".")
    _git(
        root,
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@example.org",
        "commit",
        "-qm",
        "base",
    )
    return root


def test_git_changes_and_worktree(git_repo: Path) -> None:
    base = _git(git_repo, "rev-parse", "HEAD")
    _git(git_repo, "checkout", "-qb", "topic")
    (git_repo / "src/merxen/leaf.py").write_text(
        "from merxen.core import VALUE\nX = 1\n"
    )
    _git(git_repo, "rm", "-q", "tests/test_unrelated.py")
    _git(
        git_repo,
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@example.org",
        "commit",
        "-qam",
        "x",
    )
    (git_repo / "docs/new.md").write_text("")
    (git_repo / "README.md").write_text("changed")
    committed = sel.git_changes(git_repo, base, "HEAD", worktree=False)
    assert {(c.path, c.status) for c in committed} == {
        ("src/merxen/leaf.py", "M"),
        ("tests/test_unrelated.py", "D"),
    }
    everything = {
        c.path: c.status for c in sel.git_changes(git_repo, base, "HEAD", worktree=True)
    }
    assert everything["docs/new.md"] == "A"
    assert everything["README.md"] == "M"


def test_main_formats(
    git_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    base = _git(git_repo, "rev-parse", "HEAD")
    (git_repo / "docs/guide.md").write_text("changed")
    output = tmp_path / "github_output"
    selection = tmp_path / "plan" / "selection.txt"
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    common = [
        "--root",
        str(git_repo),
        "--base",
        base,
        "--worktree",
        "--durations",
        str(tmp_path / "none.json"),
    ]
    assert (
        sel.main(
            [
                *common,
                "--format",
                "github",
                "--selection-file",
                str(selection),
                "--summary",
                str(summary),
            ]
        )
        == 0
    )
    values = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert values["mode"] == "selected"
    assert json.loads(values["shards"]) == [1]
    assert values["count"] == "1"
    listed = selection.read_text().splitlines()
    assert "tests/test_refs/test_chain.py" in listed
    assert "Test plan" in summary.read_text()
    capsys.readouterr()

    assert sel.main([*common, "--format", "paths"]) == 0
    assert "tests/test_refs/test_chain.py" in capsys.readouterr().out.splitlines()
    assert sel.main([*common, "--format", "paths", "--event", "push"]) == 0
    assert capsys.readouterr().out.strip() == "FULL"
    assert sel.main([*common, "--format", "json", "--labels", '["full-ci"]']) == 0
    assert json.loads(capsys.readouterr().out)["mode"] == "full"


def test_unknown_base_runs_everything(
    git_repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        sel.main(
            ["--root", str(git_repo), "--base", "no-such-ref", "--format", "paths"]
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == "FULL"


# The real repository.

REPO_ROOT = CI_DIR.parents[1]


def test_guards_exist() -> None:
    for guard in sel.GUARDS:
        assert (REPO_ROOT / guard).exists(), guard


def test_committed_durations_name_collected_files() -> None:
    durations = sel.load_durations(CI_DIR / "test_durations.json")
    assert durations
    recorded = set(sel.file_seconds(durations))
    test_files = {sel.file_key(t) for t in sel.RepoIndex(REPO_ROOT).test_files}
    # A few stale files are harmless (they only weigh the estimate); many mean
    # the file needs a refresh (docs/development.md).
    stale = recorded - test_files
    assert len(stale) <= max(5, len(recorded) // 20)
    assert (
        sel.file_key("tests/test_a.py") == check.duration_key("tests/test_a.py::t")[:12]
    )


def test_real_repository_selection() -> None:
    index = sel.RepoIndex(REPO_ROOT)
    durations = sel.load_durations(CI_DIR / "test_durations.json")
    docs = sel.select(index, [sel.Change("docs/development.md")], durations)
    assert docs.mode == "selected"
    assert sel._expand(sel.GUARDS, index.test_files) <= set(docs.tests)
    assert (
        sel.select(index, [sel.Change("src/merxen/config.py")], durations).mode
        == "full"
    )
    this_file = "tests/test_scripts/test_select_tests.py"
    assert this_file in sel.select(index, [sel.Change(this_file)], durations).tests


def test_github_output_keeps_the_reason_on_one_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    sel._write_github(sel.Plan("full", "docs/a\nmode=selected has no mapping"))
    lines = output.read_text().splitlines()
    assert [line.split("=", 1)[0] for line in lines] == [
        "mode",
        "reason",
        "count",
        "shards",
        "est_seconds",
    ]

"""The launch template reads its locations from the private ``.env``.

``scripts/annotation_dev/run_template.sh`` is copied into a run directory and
filled in. It takes every location from the ``.env`` named by
``MERXEN_ENV_FILE`` (a code export has none), lets a variable the operator
exported win over the file, passes the alias and site-config variables on to
the guard and to Nextflow, and stops on a site variable that names a missing
file. These tests run it with a stub guard, a stub ``nextflow`` and a stub
``df`` in a temporary directory.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = REPO_ROOT / "scripts" / "annotation_dev" / "run_template.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("ionice") is None or shutil.which("nice") is None,
    reason="the template needs ionice and nice",
)


def _executable(path: Path, text: str) -> Path:
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


@pytest.fixture
def sandbox(tmp_path: Path) -> Path:
    """A run area: stub ``nextflow`` and ``df``, a stub guard, a filled template."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _executable(
        bin_dir / "nextflow", '#!/bin/bash\necho "nextflow $*" >> "$STUB_LOG"\n'
    )
    _executable(bin_dir / "df", "#!/bin/bash\nprintf 'Avail\\n900G\\n'\n")
    _executable(
        tmp_path / "guard.sh",
        '#!/bin/bash\necho "guard ALIAS=${MERXEN_ALIAS_FILE-unset} '
        "SITE=${MERXEN_SITE_CONFIG-unset} DWIGHT=${MERXEN_DWIGHT_SITE_CONFIG-unset} "
        'TMPDIR=$TMPDIR ENV=${MERXEN_ENV_FILE-unset}" >> "$STUB_LOG"\n',
    )
    (tmp_path / "dev").mkdir()
    (tmp_path / "tmp").mkdir()
    (tmp_path / "private.env").write_text(
        f"MERXEN_ANNOTATION_DEV_DIR={tmp_path / 'dev'}\n"
        f"MERXEN_STORAGE_ROOT={tmp_path}\n"
        f"MERXEN_TASK_TMP_DIR={tmp_path / 'tmp'}\n"
        "MERXEN_DWIGHT_SITE_CONFIG=/dev/null\n"
    )
    filled = TEMPLATE.read_text()
    filled = filled.replace("CHANGE_ME_run_dir", "run1")
    filled = filled.replace("CHANGE_ME_published_results_root", "results")
    _executable(tmp_path / "run.sh", filled)
    return tmp_path


def _run(
    sandbox: Path, script: Path | None = None, **variables: str | None
) -> subprocess.CompletedProcess[str]:
    """Run the filled template; ``None`` removes a variable from the environment."""
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("MERXEN_")
    }
    environment.pop("LAUNCH_GUARD", None)
    environment.pop("TMPDIR", None)
    environment.update(
        PATH=f"{sandbox / 'bin'}:{environment['PATH']}",
        STUB_LOG=str(sandbox / "log"),
        LAUNCH_GUARD=str(sandbox / "guard.sh"),
        MERXEN_ENV_FILE=str(sandbox / "private.env"),
    )
    for name, value in variables.items():
        if value is None:
            environment.pop(name, None)
        else:
            environment[name] = value
    return subprocess.run(
        ["bash", str(script or sandbox / "run.sh")],
        env=environment,
        cwd=sandbox,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def _log(sandbox: Path) -> list[str]:
    path = sandbox / "log"
    return path.read_text().splitlines() if path.exists() else []


def test_template_needs_the_env_file_variable(sandbox: Path) -> None:
    """A code export has no .env: the variable must name it, and the error says so."""
    result = _run(sandbox, MERXEN_ENV_FILE=None)

    assert result.returncode != 0
    assert "MERXEN_ENV_FILE" in result.stderr
    assert not _log(sandbox)


def test_template_takes_its_locations_from_the_env_file(sandbox: Path) -> None:
    """The run directory, the temporary directory and the site variables."""
    result = _run(sandbox)

    assert result.returncode == 0, result.stdout + result.stderr
    guard, nextflow = _log(sandbox)
    assert f"TMPDIR={sandbox / 'tmp'}" in guard
    assert "DWIGHT=/dev/null" in guard
    assert f"ENV={sandbox / 'private.env'}" in guard
    assert f"{sandbox / 'dev' / 'run1'}/code/workflows/main.nf" in nextflow
    assert "--outdir results" in nextflow


def test_template_lets_an_exported_variable_win_over_the_env_file(
    sandbox: Path,
) -> None:
    """MERXEN_DWIGHT_SITE_CONFIG=none survives the ``.env`` that sets another value."""
    result = _run(sandbox, MERXEN_DWIGHT_SITE_CONFIG="none")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "DWIGHT=none" in _log(sandbox)[0]


def test_template_treats_an_empty_exported_variable_as_unset(sandbox: Path) -> None:
    """An empty value (an exported unset variable) does not switch the file off."""
    result = _run(sandbox, MERXEN_DWIGHT_SITE_CONFIG="")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "DWIGHT=/dev/null" in _log(sandbox)[0]


@pytest.mark.parametrize(
    "variable",
    ["MERXEN_ALIAS_FILE", "MERXEN_SITE_CONFIG", "MERXEN_DWIGHT_SITE_CONFIG"],
)
def test_template_refuses_a_variable_naming_a_missing_file(
    sandbox: Path, variable: str
) -> None:
    """Before the guard and Nextflow, naming the variable."""
    result = _run(sandbox, **{variable: str(sandbox / "missing")})

    assert result.returncode == 1
    assert f"{variable} names a file that does not exist" in result.stdout
    assert not _log(sandbox)


def test_template_refuses_a_run_directory_that_was_not_filled_in(
    sandbox: Path,
) -> None:
    """The CHANGE_ME placeholders stop the script before it creates anything."""
    result = _run(sandbox, script=TEMPLATE)

    assert result.returncode == 2
    assert "CHANGE_ME" in result.stderr
    assert not (sandbox / "dev" / "CHANGE_ME_run_dir").exists()


def test_template_keeps_an_exported_temporary_directory(sandbox: Path) -> None:
    """TMPDIR defaults to the task temporary directory of .env, not over an export."""
    result = _run(sandbox, TMPDIR=str(sandbox / "other"))

    assert result.returncode == 0, result.stdout + result.stderr
    assert f"TMPDIR={sandbox / 'other'}" in _log(sandbox)[0]

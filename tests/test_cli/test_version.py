"""``merxen --version`` prints the release (docs/versioning.md)."""

from __future__ import annotations

from click.testing import CliRunner

from merxen import __version__
from merxen.cli import main


def test_version_option_prints_the_release() -> None:
    result = CliRunner().invoke(main, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"merxen, version {__version__}"

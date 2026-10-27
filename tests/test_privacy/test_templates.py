"""The tracked templates hold placeholders only and match the loader."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from merxen import local_config, privacy
from merxen.local_config import LocalConfigError

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_EXAMPLE = REPO_ROOT / ".env.example"
ALIASES_EXAMPLE = REPO_ROOT / ".aliases.example.tsv"
LOCATIONS_HEADER = "# ===== Locations"
OTHER_HEADER = "# ===== Other settings"


def _location_section() -> str:
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    return text.split(LOCATIONS_HEADER, 1)[1].split(OTHER_HEADER, 1)[0]


def test_known_equals_the_location_keys_of_the_template() -> None:
    keys = re.findall(r"^# (MERXEN_[A-Z0-9_]+)=", _location_section(), re.M)
    assert tuple(keys) == local_config.KNOWN


def test_template_locations_are_commented_placeholders() -> None:
    for line in _location_section().splitlines()[1:]:
        if re.match(r"^#\s?MERXEN_\w+=", line):
            value = line.split("=", 1)[1].strip('"')
            assert value.startswith("/path/to/"), line
        else:
            assert not line or line.startswith("#"), line


def test_template_other_settings_parse_and_are_not_locations(tmp_path: Path) -> None:
    other = ENV_EXAMPLE.read_text(encoding="utf-8").split(OTHER_HEADER, 1)[1]
    copy = tmp_path / "env"
    copy.write_text(ENV_EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    parsed = local_config.read_env_file(copy)
    assert set(parsed) == set(re.findall(r"^(MERXEN_\w+)=", other, re.M))
    assert not set(parsed) & set(local_config.KNOWN)


@pytest.mark.parametrize("template", [ENV_EXAMPLE, ALIASES_EXAMPLE])
def test_templates_hold_no_host_path_or_donor_id(template: Path) -> None:
    hits = privacy.scan_text(template.read_text(encoding="utf-8"))
    assert hits == []


def test_alias_template_lists_the_aliases_and_is_refused_unedited() -> None:
    aliases = set()
    for line in ALIASES_EXAMPLE.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            aliases.add(line.split("\t")[0])
    expected = {"HD1", "HD2", "HD3", "HD4", "HD1_IF", "HD3_IF", "MS1", "MS2", "-"}
    assert aliases == expected | {"HM1", "HM2", "HM3", "HM4", "LCM1", "LCM2"}
    with pytest.raises(LocalConfigError, match="template placeholder"):
        privacy.load_table(ALIASES_EXAMPLE)


def test_private_env_keys_are_known(real_env_file: Path) -> None:
    """Host only: every location key of the real .env is in KNOWN."""
    keys = local_config.read_env_file(real_env_file)
    other = set(local_config.read_env_file(ENV_EXAMPLE)) | {"PROSEG_BINARY"}
    unknown = [
        key
        for key in keys
        if key.startswith("MERXEN_") and key not in set(local_config.KNOWN) | other
    ]
    assert unknown == []

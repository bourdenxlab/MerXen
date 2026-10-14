"""Tests of the scanner rules of merxen.privacy (host paths, values, ids)."""

from __future__ import annotations

from pathlib import Path

import pytest

from merxen import local_config, privacy

from .synthetic import DONOR, HOST_ROOT, LEGACY_VALUE, MOUSE, STORE_VALUE, USER


@pytest.mark.parametrize(
    "text",
    [
        HOST_ROOT + "/storage/x",
        "see " + HOST_ROOT,
        "cd " + "/" + "media/someone/disk",
        "root=" + "/" + "home/someone",
        "${X:-" + "/" + "mnt/data}",
        "file://" + "/" + "nfsdata/images/a.sif",
        "-B " + "/" + "data,/x",
        "/" + "data/raw",
        "/" + "Users/someone/Library",
        "/" + "Volumes/disk",
        "/" + "scratch/job",
        "/" + "tmp/" + "claude-1001/x",
        "C:" + "\\" + "Users" + "\\" + "someone",
    ],
)
def test_host_path_rule_matches_host_roots(text: str) -> None:
    assert privacy.find_host_paths(text)


@pytest.mark.parametrize(
    "text",
    [
        "/usr/bin/proseg and /usr/local/bin/proseg and /usr/bin/time",
        "/proc/meminfo, /dev/null and /opt/conda/bin",
        "/tmp/merxen-dwight-gpu.lock",
        "/path/to/storage/MerXen",
        "tests/data/x.csv and src/home/page.py",
        "./media/logo.svg and ~/media/x",
        "/srvx /homepage /database",
        "https://example.org/home/page",
    ],
)
def test_host_path_rule_ignores_os_paths_and_relative_paths(text: str) -> None:
    assert privacy.find_host_paths(text) == []


def test_generic_donor_ids_skip_the_synthetic_ones() -> None:
    text = f"{DONOR} P0001 P0002 P9999 X{DONOR} {DONOR}0 {DONOR}_IF"
    found = [match.group(0) for match in privacy.find_generic_donor_ids(text)]
    assert found == [DONOR, DONOR]


def test_scan_line_labels_never_hold_the_real_text(
    fake_table: Path, fake_env: Path
) -> None:
    values = privacy.private_values()
    line = f"{DONOR} in {STORE_VALUE}/x by {USER} under {HOST_ROOT}/y"
    hits = privacy.scan_line(line, values=values)
    assert [(hit.rule, hit.label) for hit in hits] == [
        ("name", "HD1"),
        ("env-value", "$MERXEN_STORAGE_ROOT"),
        ("name", "<user>"),
        ("host-path", "root=srv"),
    ]
    for hit in hits:
        assert hit.text not in hit.label
        assert line[hit.column - 1 :].startswith(hit.text)


def test_scan_line_reports_a_donor_id_inside_a_name_once(fake_table: Path) -> None:
    hits = privacy.scan_line(f"{DONOR}-2 and {DONOR}")
    assert [(hit.rule, hit.label) for hit in hits] == [("name", "HD1"), ("name", "HD1")]


def test_scan_line_without_table_still_applies_the_generic_rules() -> None:
    hits = privacy.scan_line(f"{DONOR} and {MOUSE} under {HOST_ROOT}/x")
    assert [(hit.rule, hit.label) for hit in hits] == [
        ("donor-id", "P####"),
        ("host-path", "root=srv"),
    ]


def test_scan_text_skips_base64_and_notebook_image_lines(fake_table: Path) -> None:
    payload = ("QUJD" * 30) + DONOR
    text = f'"{payload}",\n  "image/png": "{DONOR}"\n{DONOR}\n'
    hits = privacy.scan_text(text)
    assert [(hit.line, hit.rule) for hit in hits] == [(3, "name")]


def test_private_values_skip_public_and_short_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = tmp_path / "env"
    env.write_text(
        f"MERXEN_STORAGE_ROOT={STORE_VALUE}/\n"
        "MERXEN_OUTPUT_ROOT=/path/to/output\n"
        "MERXEN_PROSEG_INSTALL_PATH=/usr/bin/proseg\n"
        "MERXEN_TASK_TMP_DIR=/labshare\n"
        "MERXEN_MAX_RAM_GB=600\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(local_config.ENV_FILE_VAR, str(env))
    local_config.clear_cache()
    assert privacy.private_values() == [(STORE_VALUE, "$MERXEN_STORAGE_ROOT")]


def test_private_values_include_quoted_site_config_strings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = tmp_path / "site.config"
    site.write_text(
        "params {\n"
        '    apptainer_container = "file:///labshare/images/merxen.sif"\n'
        '    task_tmp_dir = "/labshare/proj/tmp"\n'
        '    proseg = "/usr/bin/proseg"\n'
        "}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MERXEN_SITE_CONFIG", str(site))
    privacy.clear_cache()
    values = privacy.private_values()
    assert ("/labshare/images/merxen.sif", "site.config:apptainer_container") in values
    assert ("/labshare/proj/tmp", "site.config:task_tmp_dir") in values
    assert not [value for value, _ in values if value.startswith("/usr/")]


def test_exact_values_need_a_path_boundary(fake_env: Path) -> None:
    values = privacy.private_values()
    assert privacy.scan_line(f"{STORE_VALUE}_old", values=values) == []
    hits = privacy.scan_line(f"x={LEGACY_VALUE}/a", values=values)
    assert [(hit.rule, hit.label) for hit in hits] == [
        ("env-value", "$MERXEN_SSD1_ROOT")
    ]


def test_scan_bytes_searches_printable_runs(fake_table: Path, fake_env: Path) -> None:
    data = b"\x00\x01PK\x03" + DONOR.encode() + b"\x00" + STORE_VALUE.encode() + b"\xff"
    hits = privacy.scan_bytes(data, values=privacy.private_values())
    assert [(hit.rule, hit.label) for hit in hits] == [
        ("name", "HD1"),
        ("env-value", "$MERXEN_STORAGE_ROOT"),
    ]


@pytest.mark.parametrize(
    ("path", "private"),
    [
        (".env", True),
        ("sub/.env", True),
        (".aliases.tsv", True),
        ("workflows/conf/site.config", True),
        ("workflows/conf/dwight.site.config", True),
        (".env.example", False),
        (".aliases.example.tsv", False),
        ("workflows/conf/site.config.example", False),
        ("workflows/conf/dwight.site.config.example", False),
        ("workflows/conf/dwight.config", False),
    ],
)
def test_private_files_are_recognised_by_name(path: str, private: bool) -> None:
    assert privacy.is_private_file(path) is private


def test_safe_text_masks_names_ids_and_roots(fake_table: Path) -> None:
    assert privacy.safe_text(f"notebooks/{DONOR}_x.ipynb") == "notebooks/[HD1]_x.ipynb"


def test_exceptions_name_exactly_what_may_stay() -> None:
    allowed = " ".join(item.what for item in privacy.EXCEPTIONS)
    for expected in [
        "/usr/bin/proseg",
        "/usr/local/bin/proseg",
        "/usr/bin/time",
        "/proc",
        "/tmp/merxen-dwight-gpu.lock",
        "dwight",
        "SSD1",
        "SSD2",
        "SS3",
        "copyright holder",
        "GitHub organisation URL and CI badge",
        "/path/to/",
        ".vzg2",
        "P0001, P0002, P9999",
    ]:
        assert expected in allowed
    assert len(privacy.EXCEPTIONS) == 8


def test_exceptions_pass_every_rule(fake_table: Path, fake_env: Path) -> None:
    text = (
        "/usr/bin/proseg /usr/local/bin/proseg /usr/bin/time /proc/meminfo "
        "/dev/null /opt/conda /tmp/merxen-dwight-gpu.lock conf/dwight.config "
        "SSD1 SSD2 SS3 /path/to/storage a.qqz5 P0001 P0002 P9999"
    )
    assert privacy.scan_line(text, values=privacy.private_values()) == []

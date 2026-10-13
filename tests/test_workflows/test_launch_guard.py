"""Tests of the mouse reference defaults and the SSD1 launch guard.

SSD1 (``/media/mathieubo/SSD1``) is failing, so no mouse default may point
there and a dev launch must refuse a configuration that reads from it. These
tests read config text and run ``scripts/annotation_dev/launch_guard.sh``
against a stub ``nextflow`` (the guard's logic) and, where Nextflow is
installed, against the real configuration. Nothing here touches SSD1.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / "workflows"
GUARD = REPO_ROOT / "scripts" / "annotation_dev" / "launch_guard.sh"
TEMPLATE = REPO_ROOT / "scripts" / "annotation_dev" / "run_template.sh"
DWIGHT_ANNOTATION = "conf/dwight.annotation.config"
SSD1 = "/media/mathieubo/SSD1"
HUMAN_GENE_TABLE = (
    f"{SSD1}/MerXen/mapmycells/abc_whb/expression_matrices/WHB-10Xv3/20240330/"
    "WHB-10Xv3-Nonneurons-raw.h5ad"
)
ABC = "/srv/storage/MerXen/references/abc_atlas"

# The strings of the VZG2 map_first run that built the wmb_panel bundle
# af3fab09: build_hash covers the resolved source paths, so only these paths
# keep that bundle reusable.
MOUSE_DEFAULTS = {
    "annotation_wmb_h5ad_dir": f"{ABC}/expression_matrices/WMB-10Xv3/20230630",
    "annotation_wmb_metadata_dir": f"{ABC}/metadata",
    "annotation_merfish_ccf_metadata_path": (
        f"{ABC}/metadata/MERFISH-C57BL6J-638850-CCF/20231215/views/"
        "cell_metadata_with_parcellation_annotation.csv"
    ),
    "annotation_mouse_gene_table": f"{ABC}/metadata/WMB-10X/20241115/gene.csv",
    "annotation_wmb_selfmap_test_cells_path": (
        "/srv/storage/MerXen/references/wmb_selfmap/truth_20260925.csv"
    ),
}

# Annotation params the Dwight file still points at SSD1: the store (waits on
# the user's decision U6) and the human sources (a mouse launch reads only the
# human gene table's var index).
SSD1_ANNOTATION_PARAMS_LEFT = {
    "annotation_reference_store",
    "annotation_whb_region_precompute_source",
    "annotation_whb_h5ad_dir",
    "annotation_whb_metadata_dir",
    "annotation_human_gene_table",
}

MOUSE_PARAM_PATTERN = re.compile(r"wmb|mouse|merfish")


def _unquoted(raw: str) -> str:
    """Return a raw Groovy string literal without its quotes and comment."""
    match = re.match(r'"([^"]*)"', raw)
    assert match, f"not a string literal: {raw}"
    return match.group(1)


def test_dwight_mouse_defaults_are_the_verified_srv_paths(
    workflow_config_params: dict[str, dict[str, str]],
) -> None:
    """The mouse sources default to the verified downloads on /srv/storage."""
    params = workflow_config_params[DWIGHT_ANNOTATION]

    resolved = {name: _unquoted(params[name]) for name in MOUSE_DEFAULTS}

    assert resolved == MOUSE_DEFAULTS


def test_no_mouse_param_default_points_at_ssd1(
    workflow_config_params: dict[str, dict[str, str]],
) -> None:
    """No mouse-named param of any workflow config defaults to SSD1."""
    on_ssd1 = {
        f"{relative}:{name}"
        for relative, params in workflow_config_params.items()
        for name, raw in params.items()
        if MOUSE_PARAM_PATTERN.search(name) and SSD1 in raw
    }

    assert not on_ssd1


def test_annotation_params_left_on_ssd1_are_the_known_store_and_human_ones(
    workflow_config_params: dict[str, dict[str, str]],
) -> None:
    """A new SSD1 default in the Dwight annotation file needs a decision."""
    on_ssd1 = {
        name
        for name, raw in workflow_config_params[DWIGHT_ANNOTATION].items()
        if SSD1 in raw
    }

    assert on_ssd1 == SSD1_ANNOTATION_PARAMS_LEFT


def test_run_template_calls_the_guard_before_nextflow_run() -> None:
    """The template guards the very arguments it launches with."""
    text = TEMPLATE.read_text()

    assert text.index('"$GUARD" "${args[@]}"') < text.index('nextflow -log "$R/logs')
    assert 'run "${args[@]}"' in text
    for param in (
        "--annotation_reference_store ",
        "--mecr_reference_cache_dir ",
        "--mapmycells_region_cache_dir ",
        "--clustering_squidpy_broad_reference_cache_dir ",
    ):
        assert param in text
    assert subprocess.run(["bash", "-n", str(TEMPLATE)], check=False).returncode == 0


# --- the guard against a stub nextflow ---------------------------------------

BASE_CONFIG = f"""\
params.species = 'human'
params.outdir = 'results'
params.annotation_reference_store = '{SSD1}/MerXen/annotation_references'
params.annotation_human_gene_table = '{HUMAN_GENE_TABLE}'
params.annotation_gene_id_fallback_csv = '{HUMAN_GENE_TABLE}'
params.annotation_whb_h5ad_dir = '{SSD1}/MerXen/mapmycells/abc_whb/expression_matrices'
params.annotation_wmb_h5ad_dir = '{MOUSE_DEFAULTS["annotation_wmb_h5ad_dir"]}'
params.annotation_mouse_gene_table = '{MOUSE_DEFAULTS["annotation_mouse_gene_table"]}'
params.annotation_max_forks = 2
params.annotation_prepare_only = false
process.executor = 'local'
"""


@pytest.fixture
def stub_nextflow(tmp_path: Path) -> Path:
    """Return a ``nextflow`` stub that records its arguments and prints a config."""
    config = tmp_path / "flat.config"
    config.write_text(BASE_CONFIG)
    stub = tmp_path / "nextflow"
    stub.write_text(
        '#!/bin/bash\necho "$@" > "${STUB_ARGS_FILE}"\n'
        'if [[ -n "${STUB_FAIL:-}" ]]; then echo boom >&2; exit 3; fi\n'
        'cat "${STUB_CONFIG_FILE}"\n'
    )
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    return stub


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    """Return a directory that looks like a Nextflow project."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "main.nf").write_text("// stub\n")
    return project


def _run_guard(
    stub: Path,
    project: Path,
    args: list[str],
    **env: str,
) -> subprocess.CompletedProcess[str]:
    """Run the guard with the stub as nextflow; ``env`` adds environment."""
    tmp = stub.parent
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("LAUNCH_GUARD_")
    }
    environment.update(
        NEXTFLOW=str(stub),
        STUB_CONFIG_FILE=str(tmp / "flat.config"),
        STUB_ARGS_FILE=str(tmp / "args.txt"),
        **env,
    )
    return subprocess.run(
        ["bash", str(GUARD), str(project / "main.nf"), *args],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_guard_allows_the_human_gene_table_alone(
    stub_nextflow: Path, project_dir: Path
) -> None:
    """The human gene table on SSD1 is the one default exception."""
    result = _run_guard(
        stub_nextflow,
        project_dir,
        [
            "--species",
            "mouse",
            "--annotation_reference_store",
            "/srv/storage/MerXen/annotation_references",
            "--annotation_whb_h5ad_dir",
            "/srv/storage/whb",
        ],
    )

    assert result.returncode == 0, result.stderr
    assert "allowed (allow-list): params.annotation_human_gene_table" in result.stderr
    assert "ok (species 'mouse'" in result.stderr


def test_guard_refuses_a_default_still_on_ssd1(
    stub_nextflow: Path, project_dir: Path
) -> None:
    """The store default is on SSD1 until the launch overrides it."""
    result = _run_guard(stub_nextflow, project_dir, ["--species", "mouse"])

    assert result.returncode == 1
    assert "REFUSED" in result.stderr
    assert (
        f"params.annotation_reference_store = {SSD1}/MerXen/annotation_references"
        in result.stderr
    )


@pytest.mark.parametrize(
    "override",
    [
        ["--annotation_reference_store", "/srv/storage/store"],
        ["--annotation_reference_store=/srv/storage/store"],
    ],
)
def test_guard_applies_param_overrides_of_both_spellings(
    stub_nextflow: Path, project_dir: Path, override: list[str]
) -> None:
    """A --param on the command line replaces the config default it names."""
    result = _run_guard(
        stub_nextflow,
        project_dir,
        ["--species", "mouse", "--annotation_whb_h5ad_dir=/srv/x", *override],
    )

    assert result.returncode == 0, result.stderr


def test_guard_refuses_an_override_that_points_at_ssd1(
    stub_nextflow: Path, project_dir: Path
) -> None:
    """An override can introduce an SSD1 path as well as remove one."""
    result = _run_guard(
        stub_nextflow,
        project_dir,
        [
            "--species",
            "mouse",
            "--annotation_reference_store",
            "/srv/storage/store",
            "--annotation_mouse_gene_table",
            f"{SSD1}/MerXen/mapmycells/abc_atlas/metadata/WMB-10X/20241115/gene.csv",
            "--annotation_new_param",
            f"{SSD1}/other",
        ],
    )

    assert result.returncode == 1
    assert "params.annotation_mouse_gene_table = " in result.stderr
    assert "params.annotation_new_param = " in result.stderr
    assert "annotation_reference_store" not in result.stderr.split("REFUSED")[1]


def test_guard_skips_human_sources_for_mouse_only(
    stub_nextflow: Path, project_dir: Path
) -> None:
    """A mouse launch does not read the WHB sources; a human launch does."""
    store = ["--annotation_reference_store", "/srv/storage/store"]

    mouse = _run_guard(stub_nextflow, project_dir, ["--species", "mouse", *store])
    human = _run_guard(stub_nextflow, project_dir, ["--species", "human", *store])
    config_default = _run_guard(stub_nextflow, project_dir, store)

    assert mouse.returncode == 0, mouse.stderr
    assert "skipped 1 human WHB / SEA-AD source value(s)" in mouse.stderr
    assert human.returncode == 1
    assert "params.annotation_whb_h5ad_dir" in human.stderr
    assert config_default.returncode == 1


def test_guard_allow_keys_name_params_a_launch_does_not_read(
    stub_nextflow: Path, project_dir: Path
) -> None:
    """LAUNCH_GUARD_ALLOW_KEYS lets named params through, and reports them."""
    result = _run_guard(
        stub_nextflow,
        project_dir,
        ["--species", "human"],
        LAUNCH_GUARD_ALLOW_KEYS="annotation_reference_store,annotation_whb_h5ad_dir",
    )

    assert result.returncode == 0, result.stderr
    assert "allowed (allow-list): params.annotation_reference_store" in result.stderr


def test_guard_allow_list_takes_exact_values(
    stub_nextflow: Path, project_dir: Path
) -> None:
    """LAUNCH_GUARD_ALLOW lets exact values through and nothing near them."""
    store = f"{SSD1}/MerXen/annotation_references"
    args = ["--species", "mouse", "--annotation_whb_h5ad_dir", "/srv/x"]

    allowed = _run_guard(stub_nextflow, project_dir, args, LAUNCH_GUARD_ALLOW=store)
    near_miss = _run_guard(
        stub_nextflow, project_dir, args, LAUNCH_GUARD_ALLOW=f"{store}/"
    )

    assert allowed.returncode == 0, allowed.stderr
    assert near_miss.returncode == 1


def test_guard_refuses_when_the_configuration_cannot_be_read(
    stub_nextflow: Path, project_dir: Path
) -> None:
    """A launch whose configuration cannot be resolved is not allowed."""
    result = _run_guard(stub_nextflow, project_dir, [], STUB_FAIL="1")

    assert result.returncode == 2
    assert "'nextflow config' failed" in result.stderr


def test_guard_does_not_understand_a_params_file(
    stub_nextflow: Path, project_dir: Path
) -> None:
    """A -params-file hides values from the guard, so it is refused."""
    result = _run_guard(stub_nextflow, project_dir, ["-params-file", "p.json"])

    assert result.returncode == 2
    assert "-params-file" in result.stderr


def test_guard_resolves_the_configuration_with_the_launch_profile_and_configs(
    stub_nextflow: Path, project_dir: Path, tmp_path: Path
) -> None:
    """The profile and the -c files reach `nextflow config`; other options do not."""
    extra = tmp_path / "launch.config"
    extra.write_text("")

    result = _run_guard(
        stub_nextflow,
        project_dir,
        [
            "-profile",
            "dwight,conda",
            "-c",
            str(extra),
            "-w",
            "work",
            "-resume",
            "--species",
            "mouse",
            "--annotation_reference_store",
            "/srv/storage/store",
        ],
    )

    assert result.returncode == 0, result.stderr
    recorded = (tmp_path / "args.txt").read_text().split()
    assert recorded[:3] == ["-c", str(extra), "config"]
    assert "-profile" in recorded
    assert recorded[recorded.index("-profile") + 1] == "dwight,conda"
    assert recorded[-3:] == ["-o", "flat", str(project_dir)]
    assert "work" not in recorded
    assert "-resume" not in recorded


def test_guard_prefix_can_be_changed(stub_nextflow: Path, project_dir: Path) -> None:
    """LAUNCH_GUARD_PREFIX points the guard at another mount."""
    result = _run_guard(
        stub_nextflow,
        project_dir,
        ["--species", "mouse"],
        LAUNCH_GUARD_PREFIX="/mnt/nowhere",
    )

    assert result.returncode == 0, result.stderr


# --- the guard against the real configuration --------------------------------

# The params a mouse launch must still set because their config defaults are on
# SSD1: the store (U6) and the legacy MapMyCells caches (not read while every
# row has mecr_enabled false, which the run template points off SSD1).
MOUSE_LAUNCH_OVERRIDES = [
    "--species",
    "mouse",
    "--annotation_reference_store",
    "/srv/storage/store",
    "--mecr_reference_cache_dir",
    "/srv/storage/cache",
    "--mapmycells_region_cache_dir",
    "/srv/storage/cache",
    "--clustering_squidpy_broad_reference_cache_dir",
    "/srv/storage/cache",
]


@pytest.mark.skipif(shutil.which("nextflow") is None, reason="nextflow is unavailable")
def test_guard_passes_a_mouse_launch_on_the_dwight_profile() -> None:
    """The mouse sources are off SSD1 in the real Dwight profile."""
    environment = {**os.environ, "NXF_DISABLE_CHECK_LATEST": "true"}

    result = subprocess.run(
        [
            "bash",
            str(GUARD),
            str(WORKFLOWS / "main.nf"),
            "-profile",
            "dwight",
            *MOUSE_LAUNCH_OVERRIDES,
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(shutil.which("nextflow") is None, reason="nextflow is unavailable")
def test_guard_names_only_the_known_ssd1_defaults_of_dwight() -> None:
    """With no overrides, exactly the store and the legacy caches are refused."""
    environment = {**os.environ, "NXF_DISABLE_CHECK_LATEST": "true"}

    result = subprocess.run(
        ["bash", str(GUARD), str(WORKFLOWS / "main.nf"), "-profile", "dwight"]
        + ["--species", "mouse"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    refused = set(re.findall(r"^launch_guard:   params\.(\w+) = ", result.stderr, re.M))
    assert result.returncode == 1
    assert refused == {
        "annotation_reference_store",
        "mecr_reference_cache_dir",
        "mapmycells_region_cache_dir",
        "clustering_squidpy_broad_reference_cache_dir",
    }

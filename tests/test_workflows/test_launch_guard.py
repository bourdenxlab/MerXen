"""Tests of the mouse reference defaults and the SSD1 launch guard.

SSD1 is failing, so no mouse default and no reference-store default may point
there, and a dev launch must refuse a configuration that reads from it. The
defaults are host locations of the Dwight site config, so these tests read its
tracked template (``dwight.site.config.example``: placeholders that mirror the
real layout, ``/path/to/legacy-disk`` for SSD1 and ``/path/to/storage`` for the
storage RAID). They run ``scripts/annotation_dev/launch_guard.sh`` against a stub
``nextflow`` (the guard's logic) and, where Nextflow is installed, against the
real configuration resolved with the template. The guard takes its refused
prefix and its one allow-listed file from ``.env`` (or ``LAUNCH_GUARD_*``);
the tests give them as ``LAUNCH_GUARD_*``. Nothing here touches SSD1.
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
# The Dwight site config template: the placeholder locations of the host.
DWIGHT_SITE = "conf/dwight.site.config.example"
SSD1 = "/path/to/legacy-disk"
STORAGE = "/path/to/storage/MerXen"
HUMAN_GENE_TABLE = (
    f"{SSD1}/MerXen/mapmycells/abc_whb/expression_matrices/WHB-10Xv3/20240330/"
    "WHB-10Xv3-Nonneurons-raw.h5ad"
)
ABC = f"{STORAGE}/references/abc_atlas"

# The strings of the MS2 map_first run that built a wmb_panel bundle:
# build_hash covers the resolved source paths, so only these paths keep that
# bundle reusable.
MOUSE_DEFAULTS = {
    "annotation_wmb_h5ad_dir": f"{ABC}/expression_matrices/WMB-10Xv3/20230630",
    "annotation_wmb_metadata_dir": f"{ABC}/metadata",
    "annotation_merfish_ccf_metadata_path": (
        f"{ABC}/metadata/MERFISH-C57BL6J-638850-CCF/20231215/views/"
        "cell_metadata_with_parcellation_annotation.csv"
    ),
    "annotation_mouse_gene_table": f"{ABC}/metadata/WMB-10X/20241115/gene.csv",
    "annotation_wmb_selfmap_test_cells_path": (
        f"{STORAGE}/references/wmb_selfmap/truth_20260925.csv"
    ),
}

# The Dwight reference stores (U6 (a), 2026-10-14): both on the storage RAID.
STORE_DEFAULTS = {
    "annotation_reference_store": f"{STORAGE}/annotation_references",
    "annotation_reference_store_large": f"{STORAGE}/annotation_references_large",
}

# Annotation params the Dwight site config still points at SSD1: only the human
# sources, until they are re-downloaded (U6 (b)); a mouse launch reads only the
# human gene table's var index.
SSD1_ANNOTATION_PARAMS_LEFT = {
    "annotation_whb_region_precompute_source",
    "annotation_whb_h5ad_dir",
    "annotation_whb_metadata_dir",
    "annotation_human_gene_table",
    # The same table, as the M0e symbol fallback (the allow-list covers both).
    "annotation_gene_id_fallback_csv",
}

MOUSE_PARAM_PATTERN = re.compile(r"wmb|mouse|merfish")


def _unquoted(raw: str) -> str:
    """Return a raw Groovy string literal without its quotes and comment."""
    match = re.match(r'"([^"]*)"', raw)
    assert match, f"not a string literal: {raw}"
    return match.group(1)


def test_dwight_mouse_defaults_are_the_verified_storage_paths(
    workflow_config_params: dict[str, dict[str, str]],
) -> None:
    """The mouse sources default to the verified downloads on the storage RAID."""
    params = workflow_config_params[DWIGHT_SITE]

    resolved = {name: _unquoted(params[name]) for name in MOUSE_DEFAULTS}

    assert resolved == MOUSE_DEFAULTS


def test_dwight_reference_stores_default_to_the_storage_raid(
    workflow_config_params: dict[str, dict[str, str]],
) -> None:
    """Both stores default under the storage RAID; none under SSD1 (U6 (a))."""
    params = workflow_config_params[DWIGHT_SITE]

    resolved = {name: _unquoted(params[name]) for name in STORE_DEFAULTS}

    assert resolved == STORE_DEFAULTS
    assert len(set(resolved.values())) == len(resolved)


def test_no_reference_store_param_default_points_at_ssd1(
    workflow_config_params: dict[str, dict[str, str]],
) -> None:
    """No store param of any workflow config defaults to SSD1."""
    on_ssd1 = {
        f"{relative}:{name}"
        for relative, params in workflow_config_params.items()
        for name, raw in params.items()
        if "reference_store" in name and SSD1 in raw
    }

    assert not on_ssd1


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


def test_annotation_params_left_on_ssd1_are_the_known_human_ones(
    workflow_config_params: dict[str, dict[str, str]],
) -> None:
    """A new annotation default on SSD1 in the Dwight site config needs a decision."""
    on_ssd1 = {
        name
        for name, raw in workflow_config_params[DWIGHT_SITE].items()
        if name.startswith("annotation_") and SSD1 in raw
    }

    assert on_ssd1 == SSD1_ANNOTATION_PARAMS_LEFT


def test_run_template_calls_the_guard_before_nextflow_run() -> None:
    """The template guards the very arguments it launches with."""
    text = TEMPLATE.read_text()

    assert text.index('"$GUARD" "${args[@]}"') < text.index('nextflow -log "$R/logs')
    assert 'run "${args[@]}"' in text
    for param in (
        "--mecr_reference_cache_dir ",
        "--mapmycells_region_cache_dir ",
        "--clustering_squidpy_broad_reference_cache_dir ",
    ):
        assert param in text
    assert subprocess.run(["bash", "-n", str(TEMPLATE)], check=False).returncode == 0


def test_run_template_leaves_the_reference_stores_to_the_config_defaults() -> None:
    """The stores default to the storage RAID, so the template overrides neither."""
    code = [
        line
        for line in TEMPLATE.read_text().splitlines()
        if not line.lstrip().startswith("#")
    ]

    assert not [line for line in code if "annotation_reference_store" in line]
    assert not [line for line in code if "STORE" in line]


# --- the guard against a stub nextflow ---------------------------------------

BASE_CONFIG = f"""\
params.species = 'human'
params.outdir = 'results'
params.annotation_reference_store = '{STORE_DEFAULTS["annotation_reference_store"]}'
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
def ssd1_store_config(stub_nextflow: Path) -> Path:
    """Make the stub's configuration default the store to SSD1, as Dwight once did."""
    flat = stub_nextflow.parent / "flat.config"
    text = flat.read_text()
    storage_store = STORE_DEFAULTS["annotation_reference_store"]
    assert storage_store in text
    flat.write_text(text.replace(storage_store, f"{SSD1}/MerXen/annotation_references"))
    return stub_nextflow


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
    # The refused prefix and the allow-listed file come from .env in real
    # use; here, from the variables that override it (an empty value = unset).
    environment.update(
        LAUNCH_GUARD_PREFIX=SSD1,
        LAUNCH_GUARD_HUMAN_GENE_TABLE=HUMAN_GENE_TABLE,
    )
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
        ["--species", "mouse", "--annotation_whb_h5ad_dir", f"{STORAGE}/whb"],
    )

    assert result.returncode == 0, result.stderr
    assert "allowed (allow-list): params.annotation_human_gene_table" in result.stderr
    assert "ok (species 'mouse'" in result.stderr


def test_guard_passes_a_store_default_on_the_storage_raid(
    stub_nextflow: Path, project_dir: Path
) -> None:
    """With the store default on the storage RAID, a mouse launch passes."""
    result = _run_guard(stub_nextflow, project_dir, ["--species", "mouse"])

    assert result.returncode == 0, result.stderr
    assert "annotation_reference_store" not in result.stderr


def test_guard_refuses_a_default_still_on_ssd1(
    ssd1_store_config: Path, project_dir: Path
) -> None:
    """A store default on SSD1 (an old checkout) is refused until overridden."""
    result = _run_guard(ssd1_store_config, project_dir, ["--species", "mouse"])

    assert result.returncode == 1
    assert "REFUSED" in result.stderr
    assert (
        f"params.annotation_reference_store = {SSD1}/MerXen/annotation_references"
        in result.stderr
    )


@pytest.mark.parametrize(
    "override",
    [
        ["--annotation_reference_store", f"{STORAGE}/store"],
        [f"--annotation_reference_store={STORAGE}/store"],
    ],
)
def test_guard_applies_param_overrides_of_both_spellings(
    ssd1_store_config: Path, project_dir: Path, override: list[str]
) -> None:
    """A --param on the command line replaces the config default it names."""
    result = _run_guard(
        ssd1_store_config,
        project_dir,
        ["--species", "mouse", f"--annotation_whb_h5ad_dir={STORAGE}/x", *override],
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
            f"{SSD1}/MerXen/annotation_references",
            "--annotation_mouse_gene_table",
            f"{SSD1}/MerXen/mapmycells/abc_atlas/metadata/WMB-10X/20241115/gene.csv",
            "--annotation_new_param",
            f"{SSD1}/other",
        ],
    )

    assert result.returncode == 1
    assert "params.annotation_reference_store = " in result.stderr
    assert "params.annotation_mouse_gene_table = " in result.stderr
    assert "params.annotation_new_param = " in result.stderr


def test_guard_skips_human_sources_for_mouse_only(
    stub_nextflow: Path, project_dir: Path
) -> None:
    """A mouse launch does not read the WHB sources; a human launch does."""
    mouse = _run_guard(stub_nextflow, project_dir, ["--species", "mouse"])
    human = _run_guard(stub_nextflow, project_dir, ["--species", "human"])
    config_default = _run_guard(stub_nextflow, project_dir, [])

    assert mouse.returncode == 0, mouse.stderr
    assert "skipped 1 human WHB / SEA-AD source value(s)" in mouse.stderr
    assert human.returncode == 1
    assert "params.annotation_whb_h5ad_dir" in human.stderr
    assert config_default.returncode == 1


def test_guard_allows_the_human_gene_table_value_only_under_its_own_params(
    stub_nextflow: Path, project_dir: Path
) -> None:
    """The default exception is the gene-table read, not every use of the file.

    The legacy MapMyCells source ``mecr_whb_nonneurons_h5ad_path`` holds the
    same SSD1 file but reads the whole matrix, so a human launch that leaves it
    on SSD1 is refused even though its value equals the allowed gene table.
    """
    with (stub_nextflow.parent / "flat.config").open("a") as handle:
        handle.write(f"params.mecr_whb_nonneurons_h5ad_path = '{HUMAN_GENE_TABLE}'\n")

    result = _run_guard(
        stub_nextflow,
        project_dir,
        ["--species", "human", "--annotation_whb_h5ad_dir", f"{STORAGE}/whb"],
    )

    assert result.returncode == 1, result.stderr
    refused = set(re.findall(r"^launch_guard:   params\.(\w+) = ", result.stderr, re.M))
    assert refused == {"mecr_whb_nonneurons_h5ad_path"}
    assert "allowed (allow-list): params.annotation_human_gene_table" in result.stderr
    assert (
        "allowed (allow-list): params.annotation_gene_id_fallback_csv" in result.stderr
    )


def test_guard_allow_keys_name_params_a_launch_does_not_read(
    stub_nextflow: Path, project_dir: Path
) -> None:
    """LAUNCH_GUARD_ALLOW_KEYS lets named params through, and reports them."""
    result = _run_guard(
        stub_nextflow,
        project_dir,
        ["--species", "human"],
        LAUNCH_GUARD_ALLOW_KEYS="annotation_whb_h5ad_dir",
    )

    assert result.returncode == 0, result.stderr
    assert "allowed (allow-list): params.annotation_whb_h5ad_dir" in result.stderr


def test_guard_allow_list_takes_exact_values(
    stub_nextflow: Path, project_dir: Path
) -> None:
    """LAUNCH_GUARD_ALLOW lets exact values through and nothing near them."""
    whb_dir = f"{SSD1}/MerXen/mapmycells/abc_whb/expression_matrices"
    args = ["--species", "human"]

    allowed = _run_guard(stub_nextflow, project_dir, args, LAUNCH_GUARD_ALLOW=whb_dir)
    near_miss = _run_guard(
        stub_nextflow, project_dir, args, LAUNCH_GUARD_ALLOW=f"{whb_dir}/"
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
        LAUNCH_GUARD_PREFIX="/path/to/nowhere",
    )

    assert result.returncode == 0, result.stderr


# --- the guard against the real configuration --------------------------------

# The params a mouse launch must still set because their config defaults are on
# SSD1: the legacy MapMyCells caches (not read while every row has
# mecr_enabled false, which the run template points off SSD1). The reference
# stores default to the storage RAID (U6 (a)), so a launch passes none.
MOUSE_LAUNCH_OVERRIDES = [
    "--species",
    "mouse",
    "--mecr_reference_cache_dir",
    f"{STORAGE}/cache",
    "--mapmycells_region_cache_dir",
    f"{STORAGE}/cache",
    "--clustering_squidpy_broad_reference_cache_dir",
    f"{STORAGE}/cache",
]


def _real_environment() -> dict[str, str]:
    """Environment of a guard run against the real configuration.

    The site configs are the templates (``example_site_env``) and the refused
    prefix and allow-listed file are those of the template's layout.
    """
    return {
        **os.environ,
        "NXF_DISABLE_CHECK_LATEST": "true",
        "LAUNCH_GUARD_PREFIX": SSD1,
        "LAUNCH_GUARD_HUMAN_GENE_TABLE": HUMAN_GENE_TABLE,
    }


@pytest.mark.skipif(shutil.which("nextflow") is None, reason="nextflow is unavailable")
def test_guard_passes_a_mouse_launch_on_the_dwight_profile(
    example_site_env: dict[str, str],
) -> None:
    """The mouse sources are off SSD1 in the Dwight profile with its site config."""
    environment = _real_environment()

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
def test_guard_names_only_the_known_ssd1_defaults_of_dwight(
    example_site_env: dict[str, str],
) -> None:
    """With no overrides, exactly the legacy MapMyCells caches are refused."""
    environment = _real_environment()

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
        "mecr_reference_cache_dir",
        "mapmycells_region_cache_dir",
        "clustering_squidpy_broad_reference_cache_dir",
    }


@pytest.mark.skipif(shutil.which("nextflow") is None, reason="nextflow is unavailable")
def test_guard_still_refuses_the_ssd1_human_sources_of_dwight(
    example_site_env: dict[str, str],
) -> None:
    """A human launch is refused for its SSD1 sources, never for the stores.

    The store has no human bundles yet (U6 (b)), so a human run would
    rebuild from the human sources, which stay on SSD1 and stay refused.
    """
    environment = _real_environment()

    result = subprocess.run(
        ["bash", str(GUARD), str(WORKFLOWS / "main.nf"), "-profile", "dwight"]
        + ["--species", "human"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    refused = set(re.findall(r"^launch_guard:   params\.(\w+) = ", result.stderr, re.M))
    assert result.returncode == 1
    assert {
        "annotation_whb_region_precompute_source",
        "annotation_whb_h5ad_dir",
        "annotation_whb_metadata_dir",
    } <= refused
    assert not {name for name in refused if "reference_store" in name}
    assert "allowed (allow-list): params.annotation_human_gene_table" in result.stderr


# --- where the guard finds its locations -------------------------------------
#
# The refused prefix and the one allow-listed file are not in the script: they
# come from the private .env (MERXEN_ENV_FILE, else the checkout's own), the
# process environment winning over the file, or from LAUNCH_GUARD_*. The site
# configs it resolves with are reported, and a variable naming a missing file
# stops it with the variable's name.

NO_LAUNCH_GUARD_VARIABLES = {
    "LAUNCH_GUARD_PREFIX": "",
    "LAUNCH_GUARD_HUMAN_GENE_TABLE": "",
}


def _env_file(tmp_path: Path, **values: str) -> Path:
    """Write a ``.env`` with the given variables and return its path."""
    path = tmp_path / "private.env"
    path.write_text("".join(f"{name}={value}\n" for name, value in values.items()))
    return path


def test_guard_reads_the_prefix_and_gene_table_from_the_env_file(
    stub_nextflow: Path, project_dir: Path, tmp_path: Path
) -> None:
    """With no LAUNCH_GUARD_*, the file gives both (as for a real launch)."""
    env_file = _env_file(
        tmp_path,
        MERXEN_REFUSED_PATH_PREFIX=SSD1,
        MERXEN_HUMAN_GENE_TABLE=HUMAN_GENE_TABLE,
    )

    result = _run_guard(
        stub_nextflow,
        project_dir,
        ["--species", "mouse"],
        MERXEN_ENV_FILE=str(env_file),
        **NO_LAUNCH_GUARD_VARIABLES,
    )

    assert result.returncode == 0, result.stderr
    assert "allowed (allow-list): params.annotation_human_gene_table" in result.stderr
    assert f"none on {SSD1} outside the allow-list" in result.stderr


def test_guard_without_a_configured_gene_table_allows_none(
    stub_nextflow: Path, project_dir: Path, tmp_path: Path
) -> None:
    """An unset MERXEN_HUMAN_GENE_TABLE leaves no allow-listed file."""
    env_file = _env_file(tmp_path, MERXEN_REFUSED_PATH_PREFIX=SSD1)

    result = _run_guard(
        stub_nextflow,
        project_dir,
        ["--species", "human"],
        MERXEN_ENV_FILE=str(env_file),
        **NO_LAUNCH_GUARD_VARIABLES,
    )

    refused = set(re.findall(r"^launch_guard:   params\.(\w+) = ", result.stderr, re.M))
    assert result.returncode == 1
    assert "annotation_human_gene_table" in refused
    assert "allowed (allow-list)" not in result.stderr


def test_guard_environment_beats_the_env_file(
    stub_nextflow: Path, project_dir: Path, tmp_path: Path
) -> None:
    """A variable the caller exported wins over .env (as in Python; RC13)."""
    env_file = _env_file(tmp_path, MERXEN_REFUSED_PATH_PREFIX="/path/to/elsewhere")
    arguments = ["--species", "human"]

    from_file = _run_guard(
        stub_nextflow,
        project_dir,
        arguments,
        MERXEN_ENV_FILE=str(env_file),
        **NO_LAUNCH_GUARD_VARIABLES,
    )
    from_environment = _run_guard(
        stub_nextflow,
        project_dir,
        arguments,
        MERXEN_ENV_FILE=str(env_file),
        MERXEN_REFUSED_PATH_PREFIX=SSD1,
        **NO_LAUNCH_GUARD_VARIABLES,
    )

    assert from_file.returncode == 0, from_file.stderr
    assert from_environment.returncode == 1


def test_guard_stops_without_a_refused_prefix(
    stub_nextflow: Path, project_dir: Path
) -> None:
    """No LAUNCH_GUARD_PREFIX, no .env: the guard never guesses a prefix."""
    result = _run_guard(
        stub_nextflow,
        project_dir,
        ["--species", "mouse"],
        MERXEN_ENV_FILE="none",
        **NO_LAUNCH_GUARD_VARIABLES,
    )

    assert result.returncode == 2
    assert "MERXEN_REFUSED_PATH_PREFIX" in result.stderr
    assert not (stub_nextflow.parent / "args.txt").exists()


def test_guard_can_be_switched_off_on_purpose(
    stub_nextflow: Path, project_dir: Path
) -> None:
    """LAUNCH_GUARD_PREFIX=none says so and runs nothing."""
    result = _run_guard(stub_nextflow, project_dir, [], LAUNCH_GUARD_PREFIX="none")

    assert result.returncode == 0
    assert "disabled (LAUNCH_GUARD_PREFIX=none)" in result.stderr
    assert not (stub_nextflow.parent / "args.txt").exists()


def test_guard_stops_on_a_named_env_file_that_is_missing(
    stub_nextflow: Path, project_dir: Path, tmp_path: Path
) -> None:
    """A MERXEN_ENV_FILE that names nothing is an error naming the variable."""
    result = _run_guard(
        stub_nextflow,
        project_dir,
        [],
        MERXEN_ENV_FILE=str(tmp_path / "missing.env"),
        **NO_LAUNCH_GUARD_VARIABLES,
    )

    assert result.returncode == 2
    assert "MERXEN_ENV_FILE" in result.stderr


def test_guard_finds_the_checkouts_own_env_file_when_the_variable_is_empty(
    stub_nextflow: Path, project_dir: Path, tmp_path: Path
) -> None:
    """An empty MERXEN_ENV_FILE is unset: the .env two directories above the script."""
    checkout = tmp_path / "checkout"
    script_dir = checkout / "scripts" / "annotation_dev"
    script_dir.mkdir(parents=True)
    copy = script_dir / "launch_guard.sh"
    shutil.copy(GUARD, copy)
    (checkout / ".env").write_text(f"MERXEN_REFUSED_PATH_PREFIX={SSD1}\n")
    environment = {
        **os.environ,
        "MERXEN_ENV_FILE": "",
        "NEXTFLOW": str(stub_nextflow),
        "STUB_CONFIG_FILE": str(stub_nextflow.parent / "flat.config"),
        "STUB_ARGS_FILE": str(stub_nextflow.parent / "args.txt"),
    }
    environment.pop("LAUNCH_GUARD_PREFIX", None)

    result = subprocess.run(
        ["bash", str(copy), str(project_dir / "main.nf"), "--species", "human"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 1, result.stderr
    assert f"point at {SSD1}" in result.stderr


def test_guard_reports_the_site_configs_it_resolves_with(
    stub_nextflow: Path, project_dir: Path, tmp_path: Path
) -> None:
    """The explicit file, "none on purpose", a file beside the project, or none."""
    site = tmp_path / "site.config"
    site.write_text("params {}\n")
    (project_dir / "conf").mkdir()
    (project_dir / "conf" / "dwight.site.config").write_text("params {}\n")

    explicit = _run_guard(
        stub_nextflow,
        project_dir,
        ["--species", "mouse"],
        MERXEN_SITE_CONFIG=str(site),
        MERXEN_DWIGHT_SITE_CONFIG="none",
    )
    beside = _run_guard(
        stub_nextflow,
        project_dir,
        ["--species", "mouse"],
        MERXEN_SITE_CONFIG="",
        MERXEN_DWIGHT_SITE_CONFIG="",
    )

    assert explicit.returncode == 0, explicit.stderr
    assert f"site config included: {site}" in explicit.stderr
    assert "MERXEN_DWIGHT_SITE_CONFIG=none (none on purpose)" in explicit.stderr
    assert beside.returncode == 0, beside.stderr
    assert "site config included: none found" in beside.stderr
    dwight_line = [
        line for line in beside.stderr.splitlines() if "Dwight site config" in line
    ]
    assert dwight_line[0].endswith(f"{project_dir}/conf/dwight.site.config")


@pytest.mark.parametrize(
    "variable", ["MERXEN_SITE_CONFIG", "MERXEN_DWIGHT_SITE_CONFIG"]
)
def test_guard_stops_on_a_site_variable_naming_a_missing_file(
    stub_nextflow: Path, project_dir: Path, tmp_path: Path, variable: str
) -> None:
    """Nextflow would only say "Unable to parse config file": the guard says which."""
    result = _run_guard(
        stub_nextflow,
        project_dir,
        ["--species", "mouse"],
        **{variable: str(tmp_path / "missing.config")},
    )

    assert result.returncode == 2
    assert f"{variable} names a file that does not exist" in result.stderr
    assert not (stub_nextflow.parent / "args.txt").exists()


@pytest.mark.skipif(shutil.which("nextflow") is None, reason="nextflow is unavailable")
def test_guard_checks_what_the_launch_resolves_with_its_site_configs(
    example_site_env: dict[str, str],
) -> None:
    """The guard hands the caller's environment to ``nextflow config``.

    With the Dwight site config switched off the human sources are not set, so
    nothing resolves to SSD1 and a human launch passes; with the template it is
    refused for them (``test_guard_still_refuses_the_ssd1_human_sources``).
    """
    environment = _real_environment()
    environment["MERXEN_DWIGHT_SITE_CONFIG"] = "none"

    result = subprocess.run(
        ["bash", str(GUARD), str(WORKFLOWS / "main.nf"), "-profile", "dwight"]
        + ["--species", "human"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "MERXEN_DWIGHT_SITE_CONFIG=none (none on purpose)" in result.stderr

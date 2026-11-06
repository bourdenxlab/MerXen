"""The private Nextflow site configs and the start-up check.

No tracked file holds a host path. ``workflows/nextflow.config`` includes
``conf/site.config`` and ``conf/dwight.config`` includes
``conf/dwight.site.config`` (both gitignored; tracked ``*.example`` templates
with placeholders). Each include is found through ``MERXEN_SITE_CONFIG`` /
``MERXEN_DWIGHT_SITE_CONFIG`` (``none`` or ``/dev/null`` = none on purpose, an
empty value = unset, a named missing file = a parse error), else beside the
config. A missing site config would not fail a run by itself (null params fall
back silently), so ``main.nf`` stops a Dwight launch without a Dwight site
config and an apptainer launch without an image
(``workflows/lib/SiteConfigCheck.groovy``).

The tests here resolve the configs with the templates, run the check on a
matrix of cases in one Nextflow run, and run the real ``main.nf`` (it stops at
the first error, before any process, so nothing executes). All these Nextflow
launches run once per session, side by side, in whichever pytest-xdist worker
needs one first. Tests that need the real private files skip when this host has
none.
"""

from __future__ import annotations

import fcntl
import functools
import itertools
import json
import os
import re
import shutil
import subprocess
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, TypedDict

import pytest

from merxen import local_config, privacy

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / "workflows"
NEXTFLOW = shutil.which("nextflow")
needs_nextflow = pytest.mark.skipif(NEXTFLOW is None, reason="nextflow is unavailable")

SITE = "conf/site.config.example"
DWIGHT_SITE = "conf/dwight.site.config.example"
SITE_VARIABLE = "MERXEN_SITE_CONFIG"
DWIGHT_VARIABLE = "MERXEN_DWIGHT_SITE_CONFIG"

# The values of the templates (placeholders mirroring the real layout).
IMAGE = "file:///path/to/apptainer/merxen_gpu.sif"
BIND_PATHS = "/path/to/data,/path/to/shared"
TASK_TMP_DIR = "/path/to/storage/MerXen/tmp"
ALIAS_FILE = "/path/to/private_config/aliases.tsv"
CACHE_DIR = "/path/to/legacy-disk/MerXen/mapmycells"
MODEL_STORE = "/path/to/storage/MerXen/cellpose_models"
STORE = "/path/to/storage/MerXen/annotation_references"

# --- the templates ------------------------------------------------------------


def test_templates_hold_placeholders_only() -> None:
    """A template is tracked: no host path, no donor id."""
    for relative in (SITE, DWIGHT_SITE):
        text = (WORKFLOWS / relative).read_text()
        assert not privacy.find_host_paths(text), relative
        assert not privacy.find_generic_donor_ids(text), relative


def test_the_dwight_template_exports_the_task_environment() -> None:
    """TMPDIR and the alias table reach every task (the wmb_panel hash needs it)."""
    text = (WORKFLOWS / DWIGHT_SITE).read_text()

    assert "TMPDIR = params.task_tmp_dir" in text
    assert f'MERXEN_ALIAS_FILE = "{ALIAS_FILE}"' in text


# --- the real private files (host only) ---------------------------------------

# A site-config value that also lives in .env: param -> variable. The two
# private files must agree, so they cannot drift apart (RC9).
ENV_TWINS = {
    "task_tmp_dir": "MERXEN_TASK_TMP_DIR",
    "annotation_reference_store": "MERXEN_REFERENCE_STORE",
    "annotation_reference_store_large": "MERXEN_REFERENCE_STORE_LARGE",
    "annotation_whb_region_precompute_source": "MERXEN_WHB_REGION_PRECOMPUTE_SOURCE",
    "annotation_whb_h5ad_dir": "MERXEN_WHB_EXPRESSION_DIR",
    "annotation_whb_metadata_dir": "MERXEN_WHB_METADATA_DIR",
    "annotation_wmb_mapping_stats_path": "MERXEN_WMB_MAPPING_STATS",
    "annotation_human_gene_table": "MERXEN_HUMAN_GENE_TABLE",
    "annotation_gene_id_fallback_csv": "MERXEN_HUMAN_GENE_TABLE",
    "annotation_mouse_gene_table": "MERXEN_MOUSE_GENE_TABLE",
    "annotation_wmb_selfmap_test_cells_path": "MERXEN_WMB_SELFMAP_TEST_CELLS",
    "mecr_reference_cache_dir": "MERXEN_MAPMYCELLS_CACHE_DIR",
    "mapmycells_region_cache_dir": "MERXEN_MAPMYCELLS_CACHE_DIR",
    "clustering_squidpy_broad_reference_cache_dir": "MERXEN_MAPMYCELLS_CACHE_DIR",
}
# A list-valued site param whose one entry has a .env twin.
LIST_ENV_TWINS = {"cellpose_protected_output_roots": "MERXEN_RESULTS_ROOT"}


def _real_site_file(real_env_file: Path, variable: str) -> Path:
    """The private site config the real ``.env`` names (skip when there is none)."""
    value = local_config.read_env_file(real_env_file).get(variable)
    if not value or value in local_config.ABSENT:
        pytest.skip(f"{variable} is not set in .env: no private site config")
    path = Path(value)
    if not path.is_file():
        pytest.skip(f"{variable} names no file on this host")
    return path


def _declared_params(workflow_config_params: dict[str, dict[str, str]]) -> set[str]:
    """Every param a tracked config or site template of this tree sets."""
    return {name for params in workflow_config_params.values() for name in params}


@pytest.mark.parametrize(
    ("variable", "template"), [(SITE_VARIABLE, SITE), (DWIGHT_VARIABLE, DWIGHT_SITE)]
)
def test_a_real_site_config_defines_the_params_of_its_template(
    real_env_file: Path,
    workflow_config_params: dict[str, dict[str, str]],
    parse_params: Callable[[str], dict[str, str]],
    variable: str,
    template: str,
) -> None:
    """A private site config sets every param its template lists.

    One private file serves every checkout of this host, including branches
    whose templates list more params; Nextflow keeps a param no code reads
    without effect. Such an extra param must therefore be one this tree does
    not declare at all: a declared one would override a tracked default.
    """
    real = set(parse_params(_real_site_file(real_env_file, variable).read_text()))
    listed = set(workflow_config_params[template])

    assert listed <= real, sorted(listed - real)
    assert not (real - listed) & _declared_params(workflow_config_params)


def test_the_real_dwight_site_config_agrees_with_the_env_file(
    real_env_file: Path,
) -> None:
    """Every site value that has a ``.env`` twin equals it."""
    real = _real_site_file(real_env_file, DWIGHT_VARIABLE)
    env = local_config.read_env_file(real_env_file)
    text = real.read_text()
    values = dict(re.findall(r'(?m)^\s*(\w+)\s*=\s*"([^"]*)"', text))
    lists = dict(re.findall(r"(?m)^\s*(\w+)\s*=\s*\[\s*['\"]([^'\"]*)['\"]\s*\]", text))

    checked = 0
    for param, variable in ENV_TWINS.items():
        if variable in env:
            assert values.get(param) == env[variable], f"{param} != {variable}"
            checked += 1
    for param, variable in LIST_ENV_TWINS.items():
        if variable in env:
            assert lists.get(param) == env[variable], f"{param} != [{variable}]"
            checked += 1
    assert checked, "no site value has a .env twin on this host"
    # The task environment points at the same alias table as .env.
    assert values.get("MERXEN_ALIAS_FILE") == env.get("MERXEN_ALIAS_FILE")


# --- the templates against the tracked configs -----------------------------------

SITE_TEMPLATES = (SITE, DWIGHT_SITE)
# The tracked configs that declare params (with their defaults), and those of
# the Dwight profile; each annotation file is included right after its base.
DECLARING_CONFIGS = ("nextflow.config", "conf/annotation.config")
DWIGHT_TRACKED_CONFIGS = ("conf/dwight.config", "conf/dwight.annotation.config")


def _names(
    workflow_config_params: dict[str, dict[str, str]], *relatives: str
) -> set[str]:
    """The params the given workflow config files set, together."""
    return {name for relative in relatives for name in workflow_config_params[relative]}


def test_site_templates_set_only_params_a_tracked_config_declares(
    workflow_config_params: dict[str, dict[str, str]],
) -> None:
    """A typo in a site config would set a param nothing reads; the template
    cannot carry one."""
    declared = _names(workflow_config_params, *DECLARING_CONFIGS)

    for relative in SITE_TEMPLATES:
        assert not set(workflow_config_params[relative]) - declared, relative


def test_the_two_site_templates_share_no_param(
    workflow_config_params: dict[str, dict[str, str]],
) -> None:
    """Each param has one site file: the later include would shadow the other."""
    base_site, dwight_site = (workflow_config_params[name] for name in SITE_TEMPLATES)

    assert not set(base_site) & set(dwight_site)


def test_site_templates_do_not_repeat_a_tracked_profile_param(
    workflow_config_params: dict[str, dict[str, str]],
) -> None:
    """A param is either a tracked default or a site location, never both."""
    dwight_tracked = _names(workflow_config_params, *DWIGHT_TRACKED_CONFIGS)

    for relative in SITE_TEMPLATES:
        assert not set(workflow_config_params[relative]) & dwight_tracked, relative


def test_site_configs_are_included_after_the_params_they_override() -> None:
    """Each include is one expression with the same none / empty / discovery rules.

    The Nextflow config text is the only place the rule can live (Nextflow 26.04
    parses and lints this form; the tests below resolve it). nextflow.config
    includes its site config once, after its params block and the annotation
    defaults (conf/annotation.config) and before the profiles; dwight.config
    includes its own once, after its params block and the Dwight annotation
    file, and before its process block.
    """
    base_text = (WORKFLOWS / "nextflow.config").read_text()
    dwight_text = (WORKFLOWS / "conf" / "dwight.config").read_text()
    for text, variable, file_name in (
        (base_text, SITE_VARIABLE, "site.config"),
        (dwight_text, DWIGHT_VARIABLE, "dwight.site.config"),
    ):
        expression = (
            f"includeConfig (System.getenv('{variable}') in ['none', '/dev/null'] ? "
            f"'/dev/null' : (System.getenv('{variable}') ?: "
            f'(new File("${{projectDir}}/conf/{file_name}").exists() ? '
            f"\"${{projectDir}}/conf/{file_name}\" : '/dev/null')))"
        )
        assert text.count(expression) == 1, variable
        include = text.index(f"includeConfig (System.getenv('{variable}')")
        params_start = re.search(r"(?m)^params\s*\{", text)
        assert params_start is not None
        # The params block closes at the first unindented brace after it.
        assert text.index("\n}\n", params_start.end()) < include
    base_include = base_text.index("includeConfig (System.getenv('MERXEN_SITE_CONFIG')")
    assert base_text.index("includeConfig 'conf/annotation.config'") < base_include
    assert base_include < base_text.index("\nprofiles {")
    dwight_include = dwight_text.index(
        "includeConfig (System.getenv('MERXEN_DWIGHT_SITE_CONFIG')"
    )
    assert dwight_text.index("includeConfig 'dwight.annotation.config'") < (
        dwight_include
    )
    assert dwight_include < dwight_text.index("\nprocess {")


# --- what Nextflow resolves ----------------------------------------------------

# The private site configs (gitignored, .gitignore) that a checkout may hold
# beside the templates; never copied into a test project.
PRIVATE_SITE_CONFIGS = ("site.config", "*.site.config")


def _copy_project(source: Path, target: Path) -> Path:
    """Copy a ``workflows/`` tree without run state or private site configs."""
    shutil.copytree(
        source,
        target,
        ignore=shutil.ignore_patterns(
            ".nextflow*", "work", "__pycache__", *PRIVATE_SITE_CONFIGS
        ),
    )
    return target


def test_the_project_copy_leaves_out_private_site_configs(tmp_path: Path) -> None:
    """A checkout with installed site configs must not leak them into the copy.

    The main checkout holds the real (gitignored) site configs beside the
    templates; a copy that kept them would let discovery include them, so the
    "no site config" tests would resolve real host locations and fail there.
    """
    source = tmp_path / "workflows"
    (source / "conf").mkdir(parents=True)
    for name in (
        "site.config",
        "dwight.site.config",
        "other.site.config",
        "site.config.example",
        "dwight.site.config.example",
        "dwight.config",
    ):
        (source / "conf" / name).write_text("params {}\n")

    target = _copy_project(source, tmp_path / "copy")

    assert sorted(path.name for path in (target / "conf").iterdir()) == [
        "dwight.config",
        "dwight.site.config.example",
        "site.config.example",
    ]


# Each launch starts a JVM that loads thousands of classes and, for ``run``,
# compiles main.nf and every module. HotSpot's C2 compiler and a JVM sized for
# every core of a large host cost more than such a short run gains back
# (``nextflow config`` already runs on C1 alone). The projects are local and use
# no plugin, so offline mode only skips the remote checks. Together these cut a
# main.nf launch on a large host from about 72 to 11 CPU seconds and 8 to 6.5 s.
JVM_OPTIONS = (
    f"-XX:TieredStopAtLevel=1 -XX:ActiveProcessorCount={min(4, os.cpu_count() or 1)}"
)


def _environment(**overrides: str | None) -> dict[str, str]:
    """The test environment with the given variables set (``None``: removed)."""
    environment = {**os.environ, "NXF_DISABLE_CHECK_LATEST": "true"}
    environment["NXF_ANSI_LOG"] = "false"
    environment["NXF_OFFLINE"] = "true"
    environment["NXF_OPTS"] = f"{environment.get('NXF_OPTS', '')} {JVM_OPTIONS}".strip()
    # An exported model store would win over the site configs (runtime knob).
    environment.pop("MERXEN_CELLPOSE_MODEL_STORE", None)
    for name, value in overrides.items():
        if value is None:
            environment.pop(name, None)
        else:
            environment[name] = value
    return environment


class _Launch(TypedDict):
    """One Nextflow launch: its exit status (``None``: timed out) and output."""

    returncode: int | None
    stdout: str
    stderr: str


def _launch(
    arguments: list[str],
    launch_dir: Path,
    environment: dict[str, str],
    timeout: float,
) -> _Launch:
    """Run Nextflow in ``launch_dir``, where it writes .nextflow/, results/, work/."""
    assert NEXTFLOW is not None
    launch_dir.mkdir(parents=True, exist_ok=True)
    command = [NEXTFLOW, "-log", str(launch_dir / "nextflow.log"), *arguments]
    try:
        completed = subprocess.run(
            command,
            cwd=launch_dir,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        # Recorded, not raised: only the tests of this launch fail.
        return {"returncode": None, "stdout": "", "stderr": f"timed out ({timeout} s)"}
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _once_per_session(
    tmp_path_factory: pytest.TempPathFactory,
    name: str,
    compute: Callable[[Path], object],
) -> Any:
    """Compute a JSON result once per test session, for every xdist worker.

    The pytest-xdist workers of a session share the parent of their base
    temporary directories. The first worker to ask runs ``compute`` in
    ``<name>/`` while it holds ``<name>.lock``; the others wait for the lock and
    read the result instead of repeating the same Nextflow launches.
    """
    root = tmp_path_factory.getbasetemp()
    if os.environ.get("PYTEST_XDIST_WORKER"):
        root = root.parent
    directory = root / name
    result_file = root / f"{name}.json"
    with (root / f"{name}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not result_file.is_file():
            # A worker that failed here may have left a partial directory.
            shutil.rmtree(directory, ignore_errors=True)
            directory.mkdir()
            result_file.write_text(json.dumps(compute(directory)))
    return json.loads(result_file.read_text())


def _config(
    project: Path, profile: str, environment: dict[str, str], launch_dir: Path
) -> _Launch:
    """Run ``nextflow config -o json`` on a project (cwd: outside it)."""
    arguments = ["config", "-o", "json", "-profile", profile, str(project)]
    return _launch(arguments, launch_dir, environment, timeout=120)


# The site-config variables as ``example_site_env`` sets them: the templates.
TEMPLATES: dict[str, str | None] = {
    SITE_VARIABLE: str(WORKFLOWS / SITE),
    DWIGHT_VARIABLE: str(WORKFLOWS / DWIGHT_SITE),
}
# (variable, profile): the variable names a missing file.
MISSING_FILE_CASES = [(SITE_VARIABLE, "conda"), (DWIGHT_VARIABLE, "dwight")]

# label -> (profile, site-config variables), as "<context>|<profile>".
_ConfigCases = dict[str, tuple[str, dict[str, str | None]]]


def _config_cases(missing_file: Path) -> tuple[_ConfigCases, _ConfigCases]:
    """Every config the tests read: without, then with files beside the configs.

    Without: the hermetic environment, both variables naming the templates, and
    one variable naming ``missing_file``. With: both variables unset, empty or
    turned off while both templates lie beside the configs.
    """
    without: _ConfigCases = {}
    for profile in ("standard", "dwight", "conda", "apptainer", "apptainer,gpu"):
        without[f"hermetic|{profile}"] = (profile, {})
    for profile in ("dwight", "conda", "apptainer,gpu"):
        without[f"templates|{profile}"] = (profile, TEMPLATES)
    for variable, profile in MISSING_FILE_CASES:
        without[f"missing-{variable}|{profile}"] = (
            profile,
            {variable: str(missing_file)},
        )
    beside: _ConfigCases = {}
    for state, value in (("unset", None), ("empty", "")):
        for profile in ("dwight", "conda"):
            variables = {SITE_VARIABLE: value, DWIGHT_VARIABLE: value}
            beside[f"beside-{state}|{profile}"] = (profile, variables)
    for value in ("none", "/dev/null"):
        variables = {SITE_VARIABLE: value, DWIGHT_VARIABLE: value}
        beside[f"beside-{value}|dwight"] = ("dwight", variables)
    return without, beside


def _config_launches(
    project: Path, cases: _ConfigCases
) -> dict[str, Callable[[Path], _Launch]]:
    return {
        label: functools.partial(_config, project, profile, _environment(**variables))
        for label, (profile, variables) in cases.items()
    }


@pytest.fixture(scope="module")
def configs(launches: dict[str, Any]) -> dict[str, _Launch]:
    """``nextflow config -o json`` of every case of :func:`_config_cases`."""
    resolved: dict[str, _Launch] = launches["configs"]
    return resolved


def _resolved(configs: dict[str, _Launch], label: str) -> dict[str, Any]:
    launch = configs[label]
    assert launch["returncode"] == 0, launch["stderr"]
    resolved: dict[str, Any] = json.loads(launch["stdout"])
    return resolved


def _site_params(config: dict[str, Any]) -> dict[str, Any]:
    names = (
        "task_tmp_dir",
        "apptainer_container",
        "apptainer_bind_paths",
        "clustering_squidpy_gpu_container",
        "mender_container",
        "mecr_reference_cache_dir",
        "annotation_reference_store",
    )
    return {name: config["params"][name] for name in names}


@needs_nextflow
@pytest.mark.nextflow
@pytest.mark.parametrize("profile", ["standard", "dwight", "conda", "apptainer"])
def test_without_a_site_config_the_site_params_stay_null(
    configs: dict[str, _Launch], profile: str
) -> None:
    """The hermetic environment has both variables off: nothing is included."""
    resolved = _resolved(configs, f"hermetic|{profile}")

    assert set(_site_params(resolved).values()) == {None}
    assert "TMPDIR" not in resolved.get("env", {})
    assert "MERXEN_ALIAS_FILE" not in resolved.get("env", {})


@needs_nextflow
@pytest.mark.nextflow
def test_the_apptainer_profile_mounts_nothing_without_bind_paths(
    configs: dict[str, _Launch],
) -> None:
    """No bind paths: Apptainer's own isolation flags only, and no image."""
    resolved = _resolved(configs, "hermetic|apptainer,gpu")

    assert resolved["process"]["containerOptions"] == "-C --no-home --home $PWD"
    assert resolved["process"].get("container") is None
    for name in (
        "CELLPOSE_SEGMENT",
        "CELLPOSE_NUCLEI_SEGMENT",
        "ALIGN",
        "CLUSTERING_SQUIDPY",
        "CLUSTERING_SQUIDPY_COMPUTE",
    ):
        options = resolved["process"][f"withName:{name}"]["containerOptions"]
        assert options == "-C --no-home --home $PWD --nv", name


@needs_nextflow
@pytest.mark.nextflow
def test_the_templates_fill_the_site_params(configs: dict[str, _Launch]) -> None:
    """The base site config reaches every profile, the Dwight one only Dwight's."""
    dwight = _resolved(configs, "templates|dwight")
    conda = _resolved(configs, "templates|conda")

    assert dwight["params"]["task_tmp_dir"] == TASK_TMP_DIR
    assert dwight["env"]["TMPDIR"] == TASK_TMP_DIR
    assert dwight["env"]["MERXEN_ALIAS_FILE"] == ALIAS_FILE
    assert dwight["params"]["mecr_reference_cache_dir"] == CACHE_DIR
    assert dwight["params"]["cellpose_model_store"] == MODEL_STORE
    assert dwight["params"]["annotation_reference_store"] == STORE
    assert dwight["params"]["apptainer_container"] == IMAGE
    assert conda["params"]["apptainer_container"] == IMAGE
    assert conda["params"]["apptainer_bind_paths"] == BIND_PATHS
    assert conda["params"]["mender_container"] == (
        "file:///path/to/apptainer/merxen_mender.sif"
    )
    assert conda["params"]["task_tmp_dir"] is None
    assert conda["params"]["mecr_reference_cache_dir"] is None
    assert conda["params"]["cellpose_model_store"] is None
    assert conda["params"]["annotation_reference_store"] is None
    assert "TMPDIR" not in conda.get("env", {})


@needs_nextflow
@pytest.mark.nextflow
def test_the_apptainer_profile_reads_the_image_and_bind_paths_from_params(
    configs: dict[str, _Launch],
) -> None:
    """The image and mounts that were literals in nextflow.config are params now."""
    resolved = _resolved(configs, "templates|apptainer,gpu")

    assert resolved["process"]["container"] == IMAGE
    assert resolved["process"]["containerOptions"] == (
        f"-B {BIND_PATHS} -C --no-home --home $PWD"
    )
    for name in (
        "CELLPOSE_SEGMENT",
        "CELLPOSE_NUCLEI_SEGMENT",
        "ALIGN",
        "CLUSTERING_SQUIDPY",
        "CLUSTERING_SQUIDPY_COMPUTE",
    ):
        options = resolved["process"][f"withName:{name}"]["containerOptions"]
        assert options == f"-B {BIND_PATHS} -C --no-home --home $PWD --nv", name


@needs_nextflow
@pytest.mark.nextflow
@pytest.mark.parametrize("state", ["unset", "empty"])
def test_site_configs_beside_the_configs_are_found_when_the_variables_are_off(
    configs: dict[str, _Launch], state: str
) -> None:
    """Unset and empty both mean "look beside the config" (RC2)."""
    names = ("dwight", "conda")
    explicit = {name: _resolved(configs, f"templates|{name}") for name in names}

    for name, expected in explicit.items():
        assert _resolved(configs, f"beside-{state}|{name}") == expected


@needs_nextflow
@pytest.mark.nextflow
@pytest.mark.parametrize("value", ["none", "/dev/null"])
def test_none_beats_a_site_config_beside_the_configs(
    configs: dict[str, _Launch], value: str
) -> None:
    """A deliberate "no site config" holds even with a file beside the config."""
    resolved = _resolved(configs, f"beside-{value}|dwight")

    assert set(_site_params(resolved).values()) == {None}


@needs_nextflow
@pytest.mark.nextflow
@pytest.mark.parametrize(("variable", "profile"), MISSING_FILE_CASES)
def test_a_variable_naming_a_missing_file_is_a_parse_error(
    configs: dict[str, _Launch], variable: str, profile: str
) -> None:
    """A named file that does not exist stops Nextflow (it never skips quietly)."""
    completed = configs[f"missing-{variable}|{profile}"]

    assert completed["returncode"] != 0
    assert "Unable to parse config file" in completed["stdout"] + completed["stderr"]


# --- the start-up check, on a matrix of cases (one Nextflow run) ------------------

HARNESS_CLASS = """
import groovy.json.JsonOutput
import groovy.json.JsonSlurper

class SiteHarness {
    static void run(Object casesPath, Object outPath) {
        Map cases = new JsonSlurper().parse(new File(casesPath.toString())) as Map
        Map results = [:]
        cases.each { name, testCase ->
            Map c = testCase as Map
            results[name] = SiteConfigCheck.problems(
                c.profile, c.engine, c.params as Map, c.env as Map
            )
        }
        new File(outPath.toString()).text = JsonOutput.toJson(results)
    }
}
"""
HARNESS_MAIN = "workflow {\n    SiteHarness.run(params.cases, params.out)\n}\n"

_NO_SITE = {"task_tmp_dir": None, "apptainer_container": None}
# label -> (profile, container engine, environment, the variable the error names).
NAMED_REFUSED: dict[str, tuple[str | None, str | None, dict[str, str], str]] = {
    "standard": ("standard", None, {}, DWIGHT_VARIABLE),
    "dwight-conda-empty-variable": (
        "dwight,conda",
        None,
        {DWIGHT_VARIABLE: ""},
        DWIGHT_VARIABLE,
    ),
    "local-other-variable-none": (
        "local",
        None,
        {SITE_VARIABLE: "none"},
        DWIGHT_VARIABLE,
    ),
    "apptainer": ("apptainer", "apptainer", {}, SITE_VARIABLE),
    "apptainer-gpu-empty-variable": (
        "apptainer,gpu",
        "apptainer",
        {SITE_VARIABLE: ""},
        SITE_VARIABLE,
    ),
}
# label -> (profile, container engine, params, environment).
NAMED_ACCEPTED: dict[
    str, tuple[str | None, str | None, dict[str, Any], dict[str, str]]
] = {
    "standard-none": ("standard", None, _NO_SITE, {DWIGHT_VARIABLE: "none"}),
    "standard-dev-null": ("standard", None, _NO_SITE, {DWIGHT_VARIABLE: "/dev/null"}),
    "dwight-conda-none": ("dwight,conda", None, _NO_SITE, {DWIGHT_VARIABLE: "none"}),
    "standard-with-a-tmp-dir": (
        "standard",
        None,
        {"task_tmp_dir": TASK_TMP_DIR},
        {},
    ),
    "apptainer-none": ("apptainer", "apptainer", _NO_SITE, {SITE_VARIABLE: "none"}),
    "apptainer-with-an-image": (
        "apptainer",
        "apptainer",
        {"apptainer_container": IMAGE},
        {},
    ),
    "conda-needs-neither": ("conda", None, _NO_SITE, {}),
    "hpc-profile-needs-neither": ("azure_slurm_hpc", None, _NO_SITE, {}),
    "docker-needs-no-image": ("conda", "docker", _NO_SITE, {}),
}

# None = the variable is not in the environment at all.
PROFILES = [
    None,
    "",
    "standard",
    "dwight",
    "local",
    "dwight,conda",
    " dwight , conda ",
    "conda",
    "apptainer",
    "apptainer,gpu",
    "gpu,azure_slurm_hpc",
    "dwight,apptainer",
]
SITE_VALUES = [None, "", "none", "/dev/null", "/path/to/some.site.config"]


def _case(
    profile: str | None,
    engine: str | None,
    params: dict[str, Any],
    **variables: str | None,
) -> dict[str, Any]:
    env = {name: value for name, value in variables.items() if value is not None}
    return {"profile": profile, "engine": engine, "params": params, "env": env}


def _matrix() -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    for profile, tmp, dwight in itertools.product(
        PROFILES, [None, "", "  ", TASK_TMP_DIR], SITE_VALUES
    ):
        key = f"dwight|{profile!r}|{tmp!r}|{dwight!r}"
        cases[key] = _case(
            profile,
            None,
            {"task_tmp_dir": tmp, "apptainer_container": IMAGE},
            MERXEN_DWIGHT_SITE_CONFIG=dwight,
        )
    for profile, tmp, dwight, engine, image, site in itertools.product(
        ["standard", "conda", "dwight,apptainer"],
        [None, TASK_TMP_DIR],
        [None, "none", ""],
        [None, "apptainer", "docker"],
        [None, "", IMAGE],
        SITE_VALUES,
    ):
        key = f"both|{profile!r}|{tmp!r}|{dwight!r}|{engine!r}|{image!r}|{site!r}"
        cases[key] = _case(
            profile,
            engine,
            {"task_tmp_dir": tmp, "apptainer_container": image},
            MERXEN_DWIGHT_SITE_CONFIG=dwight,
            MERXEN_SITE_CONFIG=site,
        )
    for label, (profile, engine, env, _variable) in NAMED_REFUSED.items():
        cases[f"refused|{label}"] = _case(profile, engine, _NO_SITE, **env)
    for label, (profile, engine, params, env) in NAMED_ACCEPTED.items():
        cases[f"accepted|{label}"] = _case(profile, engine, params, **env)
    return cases


def _expected_variables(case: dict[str, Any]) -> list[str]:
    """The variables a case's errors name (the rule, stated independently)."""
    profiles = {name.strip() for name in (case["profile"] or "").split(",")}
    params, env = case["params"], case["env"]
    off = {"none", "/dev/null"}
    variables = []
    if (
        profiles & {"standard", "dwight", "local"}
        and not (params.get("task_tmp_dir") or "").strip()
        and env.get(DWIGHT_VARIABLE) not in off
    ):
        variables.append(DWIGHT_VARIABLE)
    if (
        case["engine"] == "apptainer"
        and not (params.get("apptainer_container") or "").strip()
        and env.get(SITE_VARIABLE) not in off
    ):
        variables.append(SITE_VARIABLE)
    return variables


def _check_launch(root: Path) -> dict[str, Any]:
    """Run ``SiteConfigCheck.problems`` on the matrix in one Nextflow run."""
    (root / "lib").mkdir(parents=True)
    shutil.copy(WORKFLOWS / "lib" / "SiteConfigCheck.groovy", root / "lib")
    (root / "lib" / "SiteHarness.groovy").write_text(HARNESS_CLASS)
    (root / "main.nf").write_text(HARNESS_MAIN)
    (root / "cases.json").write_text(json.dumps(_matrix()))
    arguments = ["run", "main.nf", "--cases", str(root / "cases.json")]
    arguments += ["--out", str(root / "results.json")]
    launch = _launch(arguments, root, _environment(), timeout=300)
    results_file = root / "results.json"
    results = json.loads(results_file.read_text()) if results_file.is_file() else None
    return {"launch": launch, "results": results}


@pytest.fixture(scope="module")
def check_results(
    launches: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    """``SiteConfigCheck.problems`` on the matrix, from one Nextflow run."""
    cases = _matrix()
    launch = launches["check"]["launch"]
    assert launch["returncode"] == 0, launch["stdout"] + launch["stderr"]
    results: dict[str, list[str]] = launches["check"]["results"]
    assert set(results) == set(cases)
    return cases, results


@needs_nextflow
@pytest.mark.nextflow
def test_the_check_matches_the_rule_on_every_case(
    check_results: tuple[dict[str, dict[str, Any]], dict[str, list[str]]],
) -> None:
    """Each error names its variable; the cases a rule allows give none."""
    cases, results = check_results
    wrong = {}
    for key, errors in results.items():
        expected = _expected_variables(cases[key])
        named = [
            variable
            for variable in (DWIGHT_VARIABLE, SITE_VARIABLE)
            if any(variable in error for error in errors)
        ]
        if named != expected or len(errors) != len(expected):
            wrong[key] = (expected, errors)

    assert not wrong, list(wrong.items())[:3]
    assert any(results.values()) and not all(results.values())


@needs_nextflow
@pytest.mark.nextflow
@pytest.mark.parametrize("label", sorted(NAMED_REFUSED))
def test_the_check_refuses_a_missing_site_config(
    label: str,
    check_results: tuple[dict[str, dict[str, Any]], dict[str, list[str]]],
) -> None:
    """Each listed launch stops with one error that names its variable."""
    _, results = check_results
    _, _, _, variable = NAMED_REFUSED[label]

    errors = results[f"refused|{label}"]

    assert len(errors) == 1, errors
    assert variable in errors[0]
    assert "docs/local-configuration.md" in errors[0]
    assert f"{variable}=none" in errors[0]


@needs_nextflow
@pytest.mark.nextflow
@pytest.mark.parametrize("label", sorted(NAMED_ACCEPTED))
def test_the_check_accepts_a_launch_that_turned_its_site_config_off(
    label: str,
    check_results: tuple[dict[str, dict[str, Any]], dict[str, list[str]]],
) -> None:
    """Explicit ``none`` / ``/dev/null``, a set param or another profile: no error."""
    _, results = check_results

    assert results[f"accepted|{label}"] == []


# --- the real main.nf ----------------------------------------------------------

DWIGHT_ERROR = "no Dwight site config"
SITE_ERROR = "no site config for the apptainer profile"
SAMPLESHEET_ERROR = "Missing required parameter: --samplesheet"


def _run_main(
    project: Path,
    profile: str | None,
    environment: dict[str, str],
    launch_dir: Path,
) -> _Launch:
    """Run the real main.nf with no samplesheet."""
    arguments = ["run", str(project / "main.nf")]
    if profile is not None:
        arguments += ["-profile", profile]
    return _launch(arguments, launch_dir, environment, timeout=300)


# label -> (profile, site-config variables, the error expected). A variable a
# case does not set is empty (= unset).
MAIN_NF_CASES: dict[str, tuple[str | None, dict[str, str], str]] = {
    # Refused: the Dwight profiles without a Dwight site config (the variables
    # are empty = unset, and no file lies beside the configs).
    "standard-refused": (None, {}, DWIGHT_ERROR),
    "dwight-conda-refused": ("dwight,conda", {}, DWIGHT_ERROR),
    # Refused: an apptainer launch without an image.
    "apptainer-refused": ("apptainer", {}, SITE_ERROR),
    # Accepted, with the explicit "none" (the next check is the samplesheet).
    "standard-none": (None, {DWIGHT_VARIABLE: "none"}, SAMPLESHEET_ERROR),
    "apptainer-none": ("apptainer", {SITE_VARIABLE: "none"}, SAMPLESHEET_ERROR),
    "conda-needs-no-site-config": ("conda", {}, SAMPLESHEET_ERROR),
}


def _main_nf_launches(project: Path) -> dict[str, Callable[[Path], _Launch]]:
    """Every case of ``MAIN_NF_CASES``, and a launch with both templates."""
    unset = {SITE_VARIABLE: "", DWIGHT_VARIABLE: ""}
    launches = {
        label: functools.partial(
            _run_main, project, profile, _environment(**(unset | variables))
        )
        for label, (profile, variables, _) in MAIN_NF_CASES.items()
    }
    launches["templates"] = functools.partial(
        _run_main, project, None, _environment(**TEMPLATES)
    )
    return launches


@pytest.fixture(scope="module")
def main_nf_runs(launches: dict[str, Any]) -> dict[str, _Launch]:
    """The real main.nf launches of :func:`_main_nf_launches`."""
    runs: dict[str, _Launch] = launches["main_nf"]
    return runs


def _stopped_output(launch: _Launch) -> str:
    """The combined output of a main.nf launch, which must have stopped."""
    assert launch["returncode"] != 0
    return launch["stdout"] + launch["stderr"]


@needs_nextflow
@pytest.mark.nextflow
@pytest.mark.parametrize("label", list(MAIN_NF_CASES))
def test_main_nf_stops_a_launch_without_its_site_config(
    main_nf_runs: dict[str, _Launch], label: str
) -> None:
    """RC1: the start-up errors, and what the explicit ``none`` still allows."""
    _, _, expected = MAIN_NF_CASES[label]

    output = _stopped_output(main_nf_runs[label])

    assert expected in output
    if expected == DWIGHT_ERROR:
        assert DWIGHT_VARIABLE in output
        assert SITE_ERROR not in output
        assert SAMPLESHEET_ERROR not in output
    elif expected == SITE_ERROR:
        assert SITE_VARIABLE in output
        assert DWIGHT_ERROR not in output
        assert SAMPLESHEET_ERROR not in output
    else:
        assert DWIGHT_ERROR not in output
        assert SITE_ERROR not in output


@needs_nextflow
@pytest.mark.nextflow
def test_main_nf_accepts_a_launch_with_the_site_configs(
    main_nf_runs: dict[str, _Launch],
) -> None:
    """With both templates a Dwight launch goes on to the samplesheet check."""
    output = _stopped_output(main_nf_runs["templates"])

    assert SAMPLESHEET_ERROR in output
    assert DWIGHT_ERROR not in output


# --- every launch, once per session ----------------------------------------------


def _submit_all(
    pool: ThreadPoolExecutor,
    launch_root: Path,
    launches: dict[str, Callable[[Path], _Launch]],
) -> dict[str, Future[_Launch]]:
    return {
        label: pool.submit(launch, launch_root / str(index))
        for index, (label, launch) in enumerate(launches.items())
    }


def _results(futures: dict[str, Future[_Launch]]) -> dict[str, _Launch]:
    return {label: future.result() for label, future in futures.items()}


def _launch_everything(directory: Path) -> dict[str, Any]:
    """Every Nextflow launch of this module, side by side (up to one per core).

    A launch keeps one or two cores busy, mostly starting its JVM. Launches
    share a project, which they only read, and each writes its run state in its
    own launch directory. The projects are copies of ``workflows/`` that hold
    the tracked files only: a checkout's installed site configs stay out, so the
    tests see the same tree on every host.

    The config cases resolve in one copy, because the resolved config holds its
    path: those without files beside the configs first, then those with both
    templates copied where the private files would lie. main.nf runs in a copy
    with none.
    """
    config_project = _copy_project(WORKFLOWS, directory / "config" / "project")
    main_project = _copy_project(WORKFLOWS, directory / "main" / "project")
    without, beside = _config_cases(directory / "missing.config")
    with ThreadPoolExecutor(os.cpu_count() or 1) as pool:
        # The main.nf runs take longest, so they start first.
        main_nf = _submit_all(pool, directory / "main", _main_nf_launches(main_project))
        check = pool.submit(_check_launch, directory / "check")
        configs = _results(
            _submit_all(
                pool,
                directory / "config" / "without",
                _config_launches(config_project, without),
            )
        )
        conf = config_project / "conf"
        shutil.copy(config_project / SITE, conf / "site.config")
        shutil.copy(config_project / DWIGHT_SITE, conf / "dwight.site.config")
        configs |= _results(
            _submit_all(
                pool,
                directory / "config" / "beside",
                _config_launches(config_project, beside),
            )
        )
        return {
            "configs": configs,
            "main_nf": _results(main_nf),
            "check": check.result(),
        }


@pytest.fixture(scope="module")
def launches(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Every Nextflow launch of this module, run once per session.

    One batch, so a pytest-xdist worker that needs any of them waits at most
    once.
    """
    if NEXTFLOW is None:
        pytest.skip("nextflow is unavailable")
    return _once_per_session(
        tmp_path_factory, "site_config_launches", _launch_everything
    )

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
the first error, before any process, so nothing executes). Tests that need the
real private files skip when this host has none.
"""

from __future__ import annotations

import itertools
import json
import os
import re
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

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

# --- the templates ------------------------------------------------------------


def test_templates_hold_placeholders_only() -> None:
    """A template is tracked: no host path, no donor id."""
    for relative in (SITE, DWIGHT_SITE):
        text = (WORKFLOWS / relative).read_text()
        assert not privacy.find_host_paths(text), relative
        assert not privacy.find_generic_donor_ids(text), relative


def test_the_dwight_template_exports_the_task_environment() -> None:
    """TMPDIR and the alias table reach every task."""
    text = (WORKFLOWS / DWIGHT_SITE).read_text()

    assert "TMPDIR = params.task_tmp_dir" in text
    assert f'MERXEN_ALIAS_FILE = "{ALIAS_FILE}"' in text


# --- the real private files (host only) ---------------------------------------

# A site-config value that also lives in .env: param -> variable. The two
# private files must agree, so they cannot drift apart (RC9).
ENV_TWINS = {
    "task_tmp_dir": "MERXEN_TASK_TMP_DIR",
    "annotation_gene_id_fallback_csv": "MERXEN_HUMAN_GENE_TABLE",
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


def test_site_templates_set_only_params_a_tracked_config_declares(
    workflow_config_params: dict[str, dict[str, str]],
) -> None:
    """A typo in a site config would set a param nothing reads; the template
    cannot carry one."""
    declared = set(workflow_config_params["nextflow.config"])

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
    dwight_tracked = set(workflow_config_params["conf/dwight.config"])

    for relative in SITE_TEMPLATES:
        assert not set(workflow_config_params[relative]) & dwight_tracked, relative


def test_site_configs_are_included_after_the_params_they_override() -> None:
    """Each include is one expression with the same none / empty / discovery rules.

    The Nextflow config text is the only place the rule can live (Nextflow 26.04
    parses and lints this form; the tests below resolve it). nextflow.config
    includes its site config once, after its params block and before the
    profiles; dwight.config includes its own once, after its params block.
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
    assert base_text.index("includeConfig (System.getenv('MERXEN_SITE_CONFIG')") < (
        base_text.index("\nprofiles {")
    )
    assert dwight_text.index(
        "includeConfig (System.getenv('MERXEN_DWIGHT_SITE_CONFIG')"
    ) < dwight_text.index("\nprocess {")


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


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A copy of ``workflows/``, so discovery beside the configs can be tested.

    The copy holds the tracked files only: a checkout's installed site configs
    stay out, so the tests see the same tree on every host.
    """
    return _copy_project(WORKFLOWS, tmp_path / "project")


def _environment(**overrides: str | None) -> dict[str, str]:
    """The test environment with the given variables set (``None``: removed)."""
    environment = {**os.environ, "NXF_DISABLE_CHECK_LATEST": "true"}
    environment["NXF_ANSI_LOG"] = "false"
    # An exported model store would win over the site configs (runtime knob).
    environment.pop("MERXEN_CELLPOSE_MODEL_STORE", None)
    for name, value in overrides.items():
        if value is None:
            environment.pop(name, None)
        else:
            environment[name] = value
    return environment


def _config(
    project: Path,
    profile: str | None,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run ``nextflow config -o json`` on a project (cwd: beside it, not in it)."""
    assert NEXTFLOW is not None
    command = [NEXTFLOW, "-log", str(project.parent / "nextflow.log"), "config"]
    command += ["-o", "json"]
    if profile is not None:
        command += ["-profile", profile]
    return subprocess.run(
        [*command, str(project)],
        cwd=project.parent,
        env=environment or _environment(),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _resolved(
    project: Path, profile: str | None, environment: dict[str, str] | None = None
) -> dict[str, Any]:
    completed = _config(project, profile, environment)
    assert completed.returncode == 0, completed.stderr
    resolved: dict[str, Any] = json.loads(completed.stdout)
    return resolved


def _site_params(config: dict[str, Any]) -> dict[str, Any]:
    names = (
        "task_tmp_dir",
        "apptainer_container",
        "apptainer_bind_paths",
        "clustering_squidpy_gpu_container",
        "mender_container",
        "mecr_reference_cache_dir",
    )
    return {name: config["params"][name] for name in names}


@needs_nextflow
@pytest.mark.parametrize("profile", ["standard", "dwight", "conda", "apptainer"])
def test_without_a_site_config_the_site_params_stay_null(
    project: Path, profile: str
) -> None:
    """The hermetic environment has both variables off: nothing is included."""
    resolved = _resolved(project, profile)

    assert set(_site_params(resolved).values()) == {None}
    assert "TMPDIR" not in resolved.get("env", {})
    assert "MERXEN_ALIAS_FILE" not in resolved.get("env", {})


@needs_nextflow
def test_the_apptainer_profile_mounts_nothing_without_bind_paths(
    project: Path,
) -> None:
    """No bind paths: Apptainer's own isolation flags only, and no image."""
    resolved = _resolved(project, "apptainer,gpu")

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
def test_the_templates_fill_the_site_params(
    project: Path, example_site_env: dict[str, str]
) -> None:
    """The base site config reaches every profile, the Dwight one only Dwight's."""
    dwight = _resolved(project, "dwight")
    conda = _resolved(project, "conda")

    assert dwight["params"]["task_tmp_dir"] == TASK_TMP_DIR
    assert dwight["env"]["TMPDIR"] == TASK_TMP_DIR
    assert dwight["env"]["MERXEN_ALIAS_FILE"] == ALIAS_FILE
    assert dwight["params"]["mecr_reference_cache_dir"] == CACHE_DIR
    assert dwight["params"]["cellpose_model_store"] == MODEL_STORE
    assert dwight["params"]["apptainer_container"] == IMAGE
    assert conda["params"]["apptainer_container"] == IMAGE
    assert conda["params"]["apptainer_bind_paths"] == BIND_PATHS
    assert conda["params"]["mender_container"] == (
        "file:///path/to/apptainer/merxen_mender.sif"
    )
    assert conda["params"]["task_tmp_dir"] is None
    assert conda["params"]["mecr_reference_cache_dir"] is None
    assert conda["params"]["cellpose_model_store"] is None
    assert "TMPDIR" not in conda.get("env", {})


@needs_nextflow
def test_the_apptainer_profile_reads_the_image_and_bind_paths_from_params(
    project: Path, example_site_env: dict[str, str]
) -> None:
    """The image and mounts that were literals in nextflow.config are params now."""
    resolved = _resolved(project, "apptainer,gpu")

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
@pytest.mark.parametrize("state", ["unset", "empty"])
def test_site_configs_beside_the_configs_are_found_when_the_variables_are_off(
    project: Path, example_site_env: dict[str, str], state: str
) -> None:
    """Unset and empty both mean "look beside the config" (RC2)."""
    explicit = {name: _resolved(project, name) for name in ("dwight", "conda")}
    shutil.copy(project / SITE, project / "conf" / "site.config")
    shutil.copy(project / DWIGHT_SITE, project / "conf" / "dwight.site.config")
    value = None if state == "unset" else ""
    environment = _environment(**{SITE_VARIABLE: value, DWIGHT_VARIABLE: value})

    for name, expected in explicit.items():
        assert _resolved(project, name, environment) == expected


@needs_nextflow
@pytest.mark.parametrize("value", ["none", "/dev/null"])
def test_none_beats_a_site_config_beside_the_configs(project: Path, value: str) -> None:
    """A deliberate "no site config" holds even with a file beside the config."""
    shutil.copy(project / SITE, project / "conf" / "site.config")
    shutil.copy(project / DWIGHT_SITE, project / "conf" / "dwight.site.config")
    environment = _environment(**{SITE_VARIABLE: value, DWIGHT_VARIABLE: value})

    resolved = _resolved(project, "dwight", environment)

    assert set(_site_params(resolved).values()) == {None}


@needs_nextflow
@pytest.mark.parametrize(
    ("variable", "profile"),
    [(SITE_VARIABLE, "conda"), (DWIGHT_VARIABLE, "dwight")],
)
def test_a_variable_naming_a_missing_file_is_a_parse_error(
    project: Path, tmp_path: Path, variable: str, profile: str
) -> None:
    """A named file that does not exist stops Nextflow (it never skips quietly)."""
    environment = _environment(**{variable: str(tmp_path / "missing.config")})

    completed = _config(project, profile, environment)

    assert completed.returncode != 0
    assert "Unable to parse config file" in completed.stdout + completed.stderr


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


@pytest.fixture(scope="module")
def check_results(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    """Run ``SiteConfigCheck.problems`` on the matrix in one Nextflow run."""
    if NEXTFLOW is None:
        pytest.skip("nextflow is unavailable")
    root = tmp_path_factory.mktemp("site_config_groovy")
    (root / "lib").mkdir()
    shutil.copy(WORKFLOWS / "lib" / "SiteConfigCheck.groovy", root / "lib")
    (root / "lib" / "SiteHarness.groovy").write_text(HARNESS_CLASS)
    (root / "main.nf").write_text(HARNESS_MAIN)
    cases = _matrix()
    (root / "cases.json").write_text(json.dumps(cases))
    completed = subprocess.run(
        [
            NEXTFLOW,
            "-log",
            str(root / "nextflow.log"),
            "run",
            "main.nf",
            "--cases",
            str(root / "cases.json"),
            "--out",
            str(root / "results.json"),
        ],
        cwd=root,
        env=_environment(),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    results: dict[str, list[str]] = json.loads((root / "results.json").read_text())
    assert set(results) == set(cases)
    return cases, results


@needs_nextflow
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
) -> str:
    """Run the real main.nf with no samplesheet; return its combined output."""
    assert NEXTFLOW is not None
    command = [NEXTFLOW, "-log", str(project.parent / "nextflow.log"), "run"]
    command.append(str(project / "main.nf"))
    if profile is not None:
        command += ["-profile", profile]
    completed = subprocess.run(
        command,
        cwd=project.parent,
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode != 0
    return completed.stdout + completed.stderr


@needs_nextflow
@pytest.mark.parametrize(
    ("profile", "variables", "expected"),
    [
        # Refused: the Dwight profiles without a Dwight site config (the
        # variables are empty = unset, and no file lies beside the configs).
        (None, {}, DWIGHT_ERROR),
        ("dwight,conda", {}, DWIGHT_ERROR),
        # Refused: an apptainer launch without an image.
        ("apptainer", {}, SITE_ERROR),
        # Accepted, with the explicit "none" (the next check is the samplesheet).
        (None, {DWIGHT_VARIABLE: "none"}, SAMPLESHEET_ERROR),
        ("apptainer", {SITE_VARIABLE: "none"}, SAMPLESHEET_ERROR),
        ("conda", {}, SAMPLESHEET_ERROR),
    ],
    ids=[
        "standard-refused",
        "dwight-conda-refused",
        "apptainer-refused",
        "standard-none",
        "apptainer-none",
        "conda-needs-no-site-config",
    ],
)
def test_main_nf_stops_a_launch_without_its_site_config(
    project: Path, profile: str | None, variables: dict[str, str], expected: str
) -> None:
    """RC1: the start-up errors, and what the explicit ``none`` still allows."""
    environment = _environment(**{SITE_VARIABLE: "", DWIGHT_VARIABLE: ""})
    environment.update(variables)

    output = _run_main(project, profile, environment)

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
def test_main_nf_accepts_a_launch_with_the_site_configs(
    project: Path, example_site_env: dict[str, str]
) -> None:
    """With both templates a Dwight launch goes on to the samplesheet check."""
    output = _run_main(project, None, _environment())

    assert SAMPLESHEET_ERROR in output
    assert DWIGHT_ERROR not in output

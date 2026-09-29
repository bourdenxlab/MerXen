"""Annotation params stay in their own config files (plan §2.4 H7, §3.7).

``workflows/conf/annotation.config`` (included by ``nextflow.config``) and
``workflows/conf/dwight.annotation.config`` (included by the Dwight profile)
hold only annotation params. A param defined there and in ``nextflow.config``
or ``conf/dwight.config`` would let the include silently override a base or
profile value, so no param may be defined in both.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / "workflows"
BASE_FILES = ("nextflow.config", "conf/dwight.config")
ANNOTATION = "conf/annotation.config"
DWIGHT_ANNOTATION = "conf/dwight.annotation.config"

# The annotation-only params of plan §3.7, created in M1, plus the wmb_panel
# self-map test-cell source (M2; plan §3.2 excludes those cells) and the two
# species' gene tables of the exact-case species test (M3b; plan §8.4).
PLANNED_ANNOTATION_PARAMS = frozenset(
    {
        "clustering_squidpy_mode_human",
        "clustering_squidpy_mode_mouse",
        "clustering_squidpy_mode",
        "clustering_squidpy_table_key_suffix",
        "annotation_reference_store",
        "annotation_reference_store_large",
        "annotation_human_references",
        "annotation_mouse_references",
        "annotation_human_region",
        "annotation_mouse_section_regions",
        "annotation_whb_region_precompute_source",
        "annotation_whb_h5ad_dir",
        "annotation_whb_metadata_dir",
        "annotation_gene_alias_table",
        "annotation_gene_id_overrides_csv",
        "annotation_human_gene_table",
        "annotation_mouse_gene_table",
        "annotation_panel_mode",
        "annotation_resolvability",
        "annotation_prep_large_memory",
        "annotation_allow_fine_levels",
        "annotation_seaad_precomputed_stats_path",
        "annotation_seaad_metadata_dir",
        "annotation_wmb_h5ad_dir",
        "annotation_wmb_metadata_dir",
        "annotation_wmb_mapping_stats_path",
        "annotation_wmb_selfmap_test_cells_path",
        "annotation_wmb_marker_gene_universe_path",
        "annotation_wmb_max_cells_per_cluster",
        "annotation_merfish_ccf_metadata_path",
        "annotation_auto_download",
        "annotation_panel_genes_path",
        "annotation_prepare_only",
        "annotation_max_forks",
        "annotation_ctm_version",
        "annotation_keep_extended_json",
        "annotation_reuse_published",
        "annotation_xplat_sensitivity",
        "annotation_xplat_sensitivity_segmentations",
        "annotation_human_microglia_flag",
        "annotation_allow_single_method",
        "annotation_mouse_lkloc",
        "annotation_calibration_holdout_donor",
        "annotation_report_enabled",
        "annotation_mode_mapmycells_stage",
        "mender_unassigned_state_policy",
    }
)
# Params that the M0 fixes put on main, in the base configs (plan §3.7).
M0_PARAMS = (
    "mapmycells_n_processors",
    "qc_registration_strict",
    "annotation_gene_id_fallback_csv",
)
# Dwight concurrency with no default elsewhere, as for clustering_squidpy_max_forks.
HOST_ONLY_PATTERN = re.compile(r"^annotation_\w+_max_forks$")


def _names(params: dict[str, dict[str, str]], *files: str) -> set[str]:
    return set().union(*(params[relative] for relative in files))


def test_annotation_config_defines_the_planned_params(
    workflow_config_params: dict[str, dict[str, str]],
) -> None:
    """annotation.config holds exactly the annotation params of plan §3.7."""
    assert set(workflow_config_params[ANNOTATION]) == PLANNED_ANNOTATION_PARAMS


@pytest.mark.parametrize("annotation_file", [ANNOTATION, DWIGHT_ANNOTATION])
def test_no_param_is_defined_in_both(
    workflow_config_params: dict[str, dict[str, str]], annotation_file: str
) -> None:
    """No annotation config param is also a base or Dwight profile param."""
    duplicates = _names(workflow_config_params, annotation_file) & _names(
        workflow_config_params, *BASE_FILES
    )

    assert not duplicates, (
        f"{sorted(duplicates)} are defined in {annotation_file} and in "
        f"{' or '.join(BASE_FILES)}; keep each param in one place (plan §2.4 H7)"
    )


def test_dwight_annotation_config_only_overrides_annotation_params(
    workflow_config_params: dict[str, dict[str, str]],
) -> None:
    """The Dwight file overrides annotation.config or adds host concurrency."""
    unknown = {
        name
        for name in workflow_config_params[DWIGHT_ANNOTATION]
        if name not in workflow_config_params[ANNOTATION]
        and not HOST_ONLY_PATTERN.fullmatch(name)
    }

    assert not unknown


def test_m0_params_stay_in_the_base_configs(
    workflow_config_params: dict[str, dict[str, str]],
) -> None:
    """Params from the M0 fixes are defined on main, not in the annotation files."""
    base = _names(workflow_config_params, *BASE_FILES)
    annotation = _names(workflow_config_params, ANNOTATION, DWIGHT_ANNOTATION)

    for name in M0_PARAMS:
        assert name in base
        assert name not in annotation


def test_annotation_configs_are_included_once_at_hook_h7() -> None:
    """nextflow.config includes annotation.config before the profiles block."""
    base_text = (WORKFLOWS / "nextflow.config").read_text()
    dwight_text = (WORKFLOWS / "conf" / "dwight.config").read_text()
    include = "includeConfig 'conf/annotation.config'"

    assert base_text.count(include) == 1
    assert base_text.index(include) < base_text.index("\nprofiles {")
    assert re.search(r"(?m)^includeConfig 'conf/annotation\.config'$", base_text)
    assert dwight_text.count("includeConfig 'dwight.annotation.config'") == 1
    assert "includeConfig" not in (WORKFLOWS / ANNOTATION).read_text()
    assert "includeConfig" not in (WORKFLOWS / DWIGHT_ANNOTATION).read_text()


# Processes whose resources the annotation configs may set (M2, MAP in M3,
# RESOLVE in M4, COMPUTE_CPU in M5; REPORT joins when it arrives), with the
# module that defines each.
ANNOTATION_PROCESSES = {
    "ANNOTATE_PANEL": "annotation.nf",
    "ANNOTATE_REFERENCE_PREP": "annotation.nf",
    "CLUSTERING_SQUIDPY_ANNOTATE_MAP": "annotation.nf",
    "CLUSTERING_SQUIDPY_ANNOTATE_RESOLVE": "annotation.nf",
    "CLUSTERING_SQUIDPY_COMPUTE_CPU": "clustering_squidpy.nf",
}


@pytest.mark.parametrize("annotation_file", [ANNOTATION, DWIGHT_ANNOTATION])
def test_annotation_configs_set_only_annotation_process_resources(
    annotation_file: str,
) -> None:
    """The process blocks of the annotation configs select annotation processes.

    A selector for a legacy process there would change legacy resources, and
    one for a process that does not exist would warn in every run.
    """
    text = (WORKFLOWS / annotation_file).read_text()
    selectors = re.findall(r'withName:\s*"([^"]+)"', text)

    assert text.count("process {") == 1
    assert selectors
    assert set(selectors) <= set(ANNOTATION_PROCESSES)
    assert not re.search(r"^\s*process\.", text, re.MULTILINE)
    for name in selectors:
        module = (WORKFLOWS / "modules" / ANNOTATION_PROCESSES[name]).read_text()
        assert f"process {name} {{" in module


def _resolved_params(profile: str | None) -> dict[str, Any]:
    nextflow = shutil.which("nextflow")
    assert nextflow is not None
    command = [nextflow, "config", "-o", "json"]
    if profile is not None:
        command.extend(["-profile", profile])
    command.append(str(WORKFLOWS))
    completed = subprocess.run(
        command, check=True, capture_output=True, text=True, timeout=120
    )
    return json.loads(completed.stdout)["params"]


@pytest.mark.skipif(shutil.which("nextflow") is None, reason="nextflow is unavailable")
def test_resolved_params_follow_the_include_order(
    workflow_config_params: dict[str, dict[str, str]],
) -> None:
    """Nextflow sees every parsed param; the Dwight profile wins over defaults."""
    conda = _resolved_params("conda")
    dwight = _resolved_params("dwight")
    base_and_annotation = _names(workflow_config_params, "nextflow.config", ANNOTATION)

    assert set(conda) == base_and_annotation
    assert set(dwight) == base_and_annotation | _names(
        workflow_config_params, "conf/dwight.config", DWIGHT_ANNOTATION
    )
    assert conda["annotation_reference_store"] is None
    assert conda["clustering_squidpy_mode_human"] == "legacy"
    assert dwight["annotation_reference_store"] == (
        "/media/mathieubo/SSD1/MerXen/annotation_references"
    )
    assert dwight["annotation_prep_max_forks"] == 1

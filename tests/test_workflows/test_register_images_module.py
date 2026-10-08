"""Workflow checks for registering external images onto the Xenium mask grid."""

from __future__ import annotations

import re
from pathlib import Path

from merxen.config import ImageRegistrationConfig, RegisteredImageSpec

REPO_ROOT = Path(__file__).resolve().parents[2]
MAIN_TEXT = (REPO_ROOT / "workflows" / "main.nf").read_text()


def _groovy_map_keys(text: str, start_marker: str) -> set[str]:
    """Return the keys of the first Groovy map literal after ``start_marker``."""
    start = text.index(start_marker)
    open_index = text.index("[", start)
    depth = 0
    for index in range(open_index, len(text)):
        depth += {"[": 1, "]": -1}.get(text[index], 0)
        if depth == 0:
            body = text[open_index + 1 : index]
            break
    return set(re.findall(r"^\s{4,}([a-z_]+):", body, re.MULTILINE))


def test_register_images_module_runs_the_cli_and_symlinks_outputs() -> None:
    """The process must not copy the durable zarr when publishing."""
    module_text = (
        REPO_ROOT / "workflows" / "modules" / "register_images.nf"
    ).read_text()

    for expected in [
        "process REGISTER_IMAGES",
        '/register_images" }, mode: "symlink"',
        "merxen register-images --config register_images_config.json",
        'path("register_images_out")',
    ]:
        assert expected in module_text


def test_register_images_runs_between_enrich_and_viewer_cache() -> None:
    """Viewer caches and quantification must consume the registered zarr."""
    stage_order = MAIN_TEXT[MAIN_TEXT.index("def activeStageOrder(") :]
    enrich = stage_order.index('"enrich"]')
    register = stage_order.index('stages += ["register_images"]')
    viewer = stage_order.index('stages += ["build_viewer_caches"]')
    assert enrich < register < viewer

    for expected in [
        'include { REGISTER_IMAGES } from "./modules/register_images"',
        '"image_registration": "register_images"',
        "REGISTER_IMAGES(register_images_inputs_ch)",
        "viewer_cache_inputs_ch = post_registration_zarrs_ch",
        "viewer_cache_passthrough_ch = post_registration_zarrs_ch",
        'tuple("${pairId}|XENIUM", settings.registered_images_csv)',
        "runRegisterImages ||",
        "appendRegisterImagesPreflightChecks(errors, settings)",
        'chooseField(row, ["xenium_registered_images_csv"])',
    ]:
        assert expected in MAIN_TEXT


def test_emitted_config_matches_the_pydantic_contract() -> None:
    """Every key Nextflow writes must be a field the CLI validates."""
    config_keys = _groovy_map_keys(MAIN_TEXT, "def registerImagesConfig = ")
    spec_keys = _groovy_map_keys(MAIN_TEXT, "specs << ")

    assert config_keys <= set(ImageRegistrationConfig.model_fields)
    assert spec_keys == set(RegisteredImageSpec.model_fields)
    assert {"latest_zarr_path", "output_dir", "images"} <= config_keys


def test_register_images_defaults_and_resources() -> None:
    config_text = (REPO_ROOT / "workflows" / "nextflow.config").read_text()
    dwight_text = (REPO_ROOT / "workflows" / "conf" / "dwight.config").read_text()

    assert "register_images_refine_affine = true" in config_text
    assert 'withName: "REGISTER_IMAGES"' in config_text
    assert "maxForks = params.register_images_max_forks" in dwight_text

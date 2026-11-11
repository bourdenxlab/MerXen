"""Exercise named model reuse and optional training/seed planning in Groovy."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from test_hitl_model_settings import ROOT, _defaults, _run_cases

from merxen.config import CellposeConfig, MaskFilterConfig, TilingConfig


@pytest.mark.nextflow
@pytest.mark.skipif(shutil.which("nextflow") is None, reason="nextflow unavailable")
def test_named_atera_model_reuses_channels_grid_and_density_recipe(
    tmp_path: Path,
) -> None:
    root = tmp_path / "hitl"
    bundle = root / "models/pilot-one"
    bundle.mkdir(parents=True)
    (bundle / "model").write_bytes(b"synthetic model")
    profile = {
        "version": 1,
        "min_qv": 20.0,
        "bin_size_px": 1,
        "sigma_um": 0.15,
        "gene_filter": "is_gene",
        "assignment_filter": "none",
        "z_projection": "sum_all",
        "units": "transcripts_per_um2",
        "boundary_mode": "reflect",
        "gaussian_truncate": 4.0,
        "reconstruction": "bilinear_pixel_centres",
    }
    metadata = {
        "weights_sha256": hashlib.sha256((bundle / "model").read_bytes()).hexdigest(),
        "platform": "ATERA",
        "species": "human",
        "channels": ["DAPI", "18S", "transcript_density"],
        "tile_size": 2048,
        "status": "candidate",
        "cellpose_version": "4.2.1.1",
        "torch_version": "2.14.1+cu130",
        "image_key": "morphology_focus",
        "pixel_size_um": 0.2125,
        "normalization_route": "A",
        "evaluated_cellpose": CellposeConfig(
            input_normalization="per_channel", flow_threshold=0.6
        ).model_dump(mode="json"),
        "evaluated_mask_filter": MaskFilterConfig().model_dump(mode="json"),
        "training_tiling": TilingConfig(
            tile_size_candidates=[2048], tile_overlap=128
        ).model_dump(mode="json"),
        "transcript_density_profile": profile,
    }
    (bundle / "model.json").write_text(json.dumps(metadata))
    options = {
        **_defaults(),
        "cellpose_hitl_root": str(root),
        "atera_cellpose_model": "pilot-one",
        "allow_candidate_model": True,
        "cellpose_protected_output_roots": [str(tmp_path / "published")],
        "outdir": str(tmp_path / "trial"),
    }
    cases = [
        {"platform": "ATERA", "options": options, "row": {}, "check_preflight": True},
        {
            "platform": "ATERA",
            "options": {**options, "allow_candidate_model": False},
            "row": {},
            "check_preflight": True,
        },
    ]
    results = _run_cases(tmp_path, cases)
    assert "error" not in results[0], results[0]
    selected = results[0]["selected"]
    assert selected["tiling"]["tile_size_candidates"] == [2048]
    assert selected["cellpose"]["flow_threshold"] == 0.6
    assert selected["transcript_density_profile"] == profile
    assert "candidate" in results[1]["error"]


@pytest.mark.nextflow
@pytest.mark.skipif(shutil.which("nextflow") is None, reason="nextflow unavailable")
def test_training_selection_and_fresh_source_seed_are_explicit(tmp_path: Path) -> None:
    root = tmp_path / "hitl"
    package = root / "annotations/hitl-pilot"
    package.mkdir(parents=True)
    source = tmp_path / "published/source.zarr"
    source.mkdir(parents=True)
    specification = tmp_path / "train.json"
    specification.write_text(
        json.dumps(
            {
                "hitl_root": None,
                "annotation_package_ids": [package.name],
                "model_name": None,
                "platform": "ATERA",
                "species": None,
            }
        )
    )
    (tmp_path / "lib").symlink_to(ROOT / "workflows/lib", target_is_directory=True)
    (tmp_path / "options.json").write_text(
        json.dumps(
            {
                **_defaults(),
                "cellpose_hitl_training_config": str(specification),
                "cellpose_hitl_root": str(root),
                "cellpose_hitl_model_name": "pilot-one",
                "cellpose_hitl_source_zarr": str(source),
                "cellpose_protected_output_roots": [str(source.parent)],
                "outdir": str(tmp_path / "trial"),
            }
        )
    )
    (tmp_path / "check.nf").write_text("""
nextflow.enable.dsl = 2
workflow {
    def opts = new groovy.json.JsonSlurper().parse(new File('options.json'))
    def spec = CellposeHitlSettings.specification(opts)
    def settings = [pair_id: 'P1', active_platforms: ['ATERA'],
        selected_stages: ['train_cellpose', 'segment'], species: 'human',
        segmentation_source: 'cellpose_proseg', run_build: false,
        need_build_results: true, run_segment: true]
    CellposeHitlSettings.validateRows([['P1', [:], settings]], spec, opts)
    def seed = CellposeHitlSettings.seedConfig('P1', 'ATERA', [:], settings, opts)
    def recipe = AteraDensitySettings.selectedProfile(
        [transcript_density_profile: AteraDensitySettings.profile(opts)], opts)
    def density = AteraDensitySettings.config('source.zarr', opts, recipe)
    assert spec.config.model_name == 'pilot-one'
    assert spec.config.species == 'human'
    assert spec.config.package_dirs == ['hitl-pilot']
    assert seed.source_zarr == opts.cellpose_hitl_source_zarr
    assert seed.destination_zarr.startsWith(opts.outdir)
    assert density.sigma_um == 0.15
    assert !AteraDensitySettings.enabledForMeta(
        [cellpose_settings: [transcript_density_profile: null]], opts)
    def trainOnly = new LinkedHashMap(settings) + [run_segment: false]
    assert CellposeHitlSettings.applyTrainedModel([:], trainOnly,
        [platform: 'ATERA'], opts) == [:]
    new File('passed.txt').text = 'ok'
}
""")
    result = subprocess.run(
        ["nextflow", "run", "check.nf", "-ansi-log", "false"],
        cwd=tmp_path,
        env={**os.environ, "NXF_OFFLINE": "true"},
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "passed.txt").read_text() == "ok"

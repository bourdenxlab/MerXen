"""End-to-end tests of the annotation report (``merxen.annotation.report``; M7).

The inputs are the synthetic RESOLVE outputs of ``test_pipeline_resolve``
(real WHB supercluster and SEA-AD subclass names, the fake bundles): the
report reads them exactly as it reads a published pair. Cortical depth and
the shared tissue mask are written beside them when a test needs them.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from merxen.annotation import pipeline as pl
from merxen.annotation.pipeline import (
    MAP_MANIFEST_NAME,
    annotate_resolve,
    write_tidy_parquet,
)
from merxen.annotation.report import (
    ReportOutputError,
    build_annotation_report,
    check_output_dir,
)
from merxen.annotation.report_depth import SQUARE_TILE_CI
from merxen.annotation.report_inputs import ReportSources, discover_sources
from merxen.annotation.report_model import AcceptanceMetrics, ReportOptions
from merxen.annotation.vocab import HUMAN_BROAD_CLASSES
from merxen.cli import main as cli_main

from .conftest import FakeMmc
from .test_pipeline_resolve import (
    Setup,
    _add_xpanel_runs,
    _resolve,
    _run_record,
    _set_panel_mode,
    _setup,
    _whb_tidy,
)

MakeTrust = Callable[..., Any]
OPTIONS = ReportOptions(n_bootstrap=20)


def _resolved(
    root: Path, fake_mmc: FakeMmc, make_trust: MakeTrust, state: str = "validated_real"
) -> Setup:
    setup = _setup(root, fake_mmc, source="clustered")
    _resolve(setup, make_trust, state=state)
    return setup


def _sources(setup: Setup, **changes: Any) -> ReportSources:
    root = setup.root
    base = ReportSources(
        species="human",
        pair_id="PX",
        segmentation="proseg_hybrid",
        resolve_dir=root / "resolve_out",
        map_dir=setup.map_dir,
        panel_dir=setup.panel_dir,
        clustered_h5ad={sample.sample_id: sample.h5ad_path for sample in setup.samples},
    )
    return ReportSources(**{**base.__dict__, **changes})


def _depth_parquet(setup: Setup, sample_index: int, path: Path) -> Path:
    """A cortical-depth cell table: oligodendrocytes enriched in white matter."""
    labels = pd.read_parquet(
        setup.root
        / "resolve_out"
        / setup.samples[sample_index].platform.lower()
        / f"{setup.samples[sample_index].sample_id}_celltype_labels.parquet",
        columns=["cell_id", "ct_broad_name"],
    )
    # Not the seed the synthetic calls were drawn with (it would correlate).
    rng = np.random.default_rng(1000 + sample_index)
    oligo = labels["ct_broad_name"].astype(str).to_numpy() == "Oligodendrocytes"
    white = np.where(
        oligo,
        rng.uniform(size=len(labels)) < 0.9,
        rng.uniform(size=len(labels)) < 0.35,
    )
    depth = np.where(white, np.nan, rng.uniform(size=len(labels)))
    frame = pd.DataFrame(
        {
            "cell_id": labels["cell_id"].astype(str),
            "x": rng.uniform(0, 3000, len(labels)),
            "y": rng.uniform(0, 3000, len(labels)),
            "inside_cortical_ribbon": ~white,
            "cortical_depth_annotation": np.where(white, "white_matter", "grey_matter"),
            "equivolumetric_depth": depth,
            "laplace_depth": depth,
            "cortical_depth_qc_flag": np.where(white, "outside_ribbon", "assigned"),
            "tangential_position_um": np.where(
                white, np.nan, rng.uniform(0, 3000, len(labels))
            ),
            "column_id": np.where(white, np.nan, rng.integers(0, 12, len(labels))),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    return path


def _alignment(root: Path) -> Path:
    """A shared tissue mask covering the left two thirds (10 µm pixels)."""
    align = root / "align_out"
    align.mkdir(parents=True)
    mask = np.zeros((300, 300), dtype=np.uint8)
    mask[:, :200] = 1
    np.save(align / "shared_tissue_mask.npy", mask)
    (align / "registration_summary.json").write_text(
        json.dumps(
            {
                "coordinate_frames": {
                    "fixed_platform": "XENIUM",
                    "fixed_dataset_to_image_matrix": [
                        [0.1, 0, 0],
                        [0, 0.1, 0],
                        [0, 0, 1],
                    ],
                }
            }
        )
    )
    return align


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_report_builds_every_item_with_one_csv_per_figure(
    tmp_path: Path, fake_mmc: FakeMmc, make_trust: MakeTrust
) -> None:
    setup = _resolved(tmp_path / "run", fake_mmc, make_trust)
    result = build_annotation_report(
        _sources(setup), tmp_path / "report", options=OPTIONS, strict=True
    )
    statuses = {item.number: item.status for item in result.items}
    assert statuses[8] == "not_available"  # no held-out acceptance output
    assert statuses[9] == "not_available"  # no cortical depth
    assert statuses[10] == "not_applicable"  # human
    assert all(
        statuses[number] in ("ok", "partial")
        for number in (1, 2, 3, 4, 5, 6, 7, 11, 12)
    )
    figures = [record for item in result.items for record in item.figures]
    assert len(figures) >= 15
    for record in figures:
        assert record.png.is_file() and record.pdf.is_file() and record.csv.is_file()
    stems = [record.stem for record in figures]
    assert len(stems) == len(set(stems))
    html = result.html.read_text()
    assert "<script" not in html and "http://" not in html and "https://" not in html
    for record in figures:
        assert f'src="figures/{record.png.name}"' in html
    document = AcceptanceMetrics.model_validate_json(result.metrics_path.read_text())
    covered = {row.criterion for row in document.criteria}
    assert {"H1", "H2", "H3", "H7", "H8", "H12", "H16", "H18"} <= covered
    labels = pd.read_parquet(
        setup.root / "resolve_out" / "merscope" / "PX_MERSCOPE_celltype_labels.parquet",
        columns=["in_table", "total_counts"],
    )
    table = labels[labels["in_table"]]
    (h3,) = document.find("H3", "whb_sea_agreement_ge20", sample_id="PX_MERSCOPE")
    assert h3.n == int((table["total_counts"] >= 20).sum()) < len(table)
    assert 0.5 < h3.value <= 1.0
    assert document.provenance["samples"]["PX_MERSCOPE"]["labels_sha256"]
    assert (tmp_path / "report" / "PX_platform_gene_factors.csv").is_file()
    # The symbol-only MERSCOPE H5AD is joined to the profiles through PANEL's ids.
    dotplot = pd.read_csv(
        tmp_path
        / "report"
        / "figures"
        / "item04_reference_expectation_dotplot_merscope.csv"
    )
    assert len(dotplot) and dotplot["gene_id"].str.startswith("ENSG").all()


def test_report_h1_reuses_resolves_pair_jsd_and_recomputes_it(
    tmp_path: Path, fake_mmc: FakeMmc, make_trust: MakeTrust
) -> None:
    setup = _resolved(tmp_path / "run", fake_mmc, make_trust)
    summary = json.loads(
        (setup.root / "resolve_out" / "PX_resolve_summary.json").read_text()
    )
    recorded = next(
        row
        for row in summary["pair"]["jsd"]
        if row["kind"] == "soft" and row["region"] == "whole_section"
    )
    options = ReportOptions(n_bootstrap=int(summary["pair"]["n_bootstrap"]))
    result = build_annotation_report(
        _sources(setup),
        tmp_path / "report",
        options=options,
        strict=True,
        make_figures=False,
    )
    (h1,) = result.metrics.find("H1", "broad_jsd", region="whole_section")
    assert h1.value == pytest.approx(recorded["jsd"])
    assert h1.ci_low == pytest.approx(recorded["ci_low"])
    assert h1.ci_high == pytest.approx(recorded["ci_high"])
    assert h1.source == "resolve_summary"
    # The report's own estimate (same estimator, same seed) agrees.
    assert f"report recomputation {recorded['jsd']:.6g}" in h1.note
    supercluster = result.metrics.find(
        "report", "supercluster_jsd", region="whole_section"
    )
    assert supercluster and supercluster[0].value is not None
    assert supercluster[0].ci_low <= supercluster[0].value <= supercluster[0].ci_high


def test_report_is_deterministic(
    tmp_path: Path, fake_mmc: FakeMmc, make_trust: MakeTrust
) -> None:
    setup = _resolved(tmp_path / "run", fake_mmc, make_trust)
    first = build_annotation_report(
        _sources(setup),
        tmp_path / "a",
        options=OPTIONS,
        strict=True,
        make_figures=False,
    )
    second = build_annotation_report(
        _sources(setup),
        tmp_path / "b",
        options=OPTIONS,
        strict=True,
        make_figures=False,
    )
    assert first.metrics_path.read_bytes() == second.metrics_path.read_bytes()
    assert first.html.read_bytes() == second.html.read_bytes()
    names = sorted(
        path.relative_to(first.out_dir)
        for path in first.out_dir.rglob("*")
        if path.is_file()
    )
    names = [name for name in names if name.name != "report_run.json"]
    for name in names:
        assert _hash(first.out_dir / name) == _hash(second.out_dir / name), name
    # Drawn figures (PNG and PDF) are byte-identical too.
    drawn = [
        build_annotation_report(
            _sources(setup), tmp_path / f"fig_{key}", options=OPTIONS, items=[2]
        )
        for key in ("a", "b")
    ]
    pngs = sorted((drawn[0].out_dir / "figures").glob("*.p[dn][gf]"))
    assert pngs
    for path in pngs:
        other = drawn[1].out_dir / "figures" / path.name
        assert _hash(path) == _hash(other), path.name


def test_report_with_cortical_depth_and_the_shared_mask(
    tmp_path: Path, fake_mmc: FakeMmc, make_trust: MakeTrust
) -> None:
    setup = _resolved(tmp_path / "run", fake_mmc, make_trust)
    depth = {
        sample.sample_id: _depth_parquet(
            setup,
            index,
            tmp_path
            / "depth"
            / sample.platform.lower()
            / f"{sample.sample_id}_cells_with_cortical_depth.parquet",
        )
        for index, sample in enumerate(setup.samples)
    }
    sources = _sources(setup, cortical_depth=depth, alignment_dir=_alignment(tmp_path))
    result = build_annotation_report(
        sources, tmp_path / "report", options=OPTIONS, strict=True, make_figures=False
    )
    statuses = {item.number: item.status for item in result.items}
    assert statuses[9] == "ok"
    for sample_id in depth:
        (contrast,) = result.metrics.find(
            "H12", "oligodendrocyte_wm_minus_gm_share", sample_id=sample_id
        )
        assert contrast.kind == "white_matter_label"
        assert contrast.value is not None and contrast.value > 0.2
        assert contrast.ci_low is not None and contrast.ci_low > 0
        (ordering,) = result.metrics.find(
            "H12", "depth_ordering_passes", sample_id=sample_id, kind=None
        )
        # The synthetic pair has no deep-layer superclusters: the ordering fails.
        assert ordering.value is False and "missing" in ordering.note
        assert result.metrics.find(
            "H12", "depth_ordering_passes", sample_id=sample_id, kind=SQUARE_TILE_CI
        )
        (valid,) = result.metrics.find("H12", "depth_input_valid", sample_id=sample_id)
        # Too few cells for the label-free depth check: not checked, and the
        # H12 metrics stay measured.
        assert valid.value is None and valid.note.startswith("not checked")
    assert result.metrics.find("H12", "depth_ordering_replicated")
    composition = pd.read_csv(
        tmp_path / "report" / "tables" / "item02_composition__composition.csv"
    )
    assert set(composition["region"]) == {"whole_section", "shared_mask"}
    (masked,) = result.metrics.find("H1", "broad_jsd", region="shared_mask")
    assert masked.source == "report_metrics" and masked.value is not None
    assert masked.ci_low is not None


def test_a_broad_only_gate_makes_both_h12_orderings_not_available(
    tmp_path: Path, fake_mmc: FakeMmc, make_trust: MakeTrust
) -> None:
    """No supercluster labels: neither CI method records a failed ordering."""
    setup = _resolved(tmp_path / "run", fake_mmc, make_trust)
    path = setup.root / "resolve_out" / "PX_resolve_summary.json"
    summary = json.loads(path.read_text())
    gated = setup.samples[0].sample_id
    summary["samples"][gated]["resolution"]["gate"]["level"] = "broad_only"
    path.write_text(json.dumps(summary))
    suffix = "_cells_with_cortical_depth.parquet"
    depth = {
        sample.sample_id: _depth_parquet(
            setup, index, tmp_path / "depth" / f"{sample.sample_id}{suffix}"
        )
        for index, sample in enumerate(setup.samples)
    }
    result = build_annotation_report(
        _sources(setup, cortical_depth=depth),
        tmp_path / "report",
        options=OPTIONS,
        strict=True,
        make_figures=False,
        items=[9],
    )
    for sample in setup.samples:
        records = result.metrics.find(
            "H12", "depth_ordering_passes", sample_id=sample.sample_id
        )
        assert {record.kind for record in records} == {None, SQUARE_TILE_CI}
        for record in records:
            if sample.sample_id == gated:
                assert record.status == "not_available" and record.value is None
                assert record.note.startswith("dataset gate broad_only")
            else:
                # The synthetic pair lacks deep-layer superclusters: a failure.
                assert record.status == "measured" and record.value is False
    # Replication needs both platforms' orderings.
    (replicated,) = result.metrics.find("H12", "depth_ordering_replicated")
    assert replicated.status == "not_available" and replicated.value is None
    assert replicated.note.startswith("no ordering on")


def test_report_without_depth_or_mask_marks_them_missing(
    tmp_path: Path, fake_mmc: FakeMmc, make_trust: MakeTrust
) -> None:
    setup = _resolved(tmp_path / "run", fake_mmc, make_trust)
    result = build_annotation_report(
        _sources(setup),
        tmp_path / "report",
        options=OPTIONS,
        strict=True,
        make_figures=False,
    )
    depth_item = next(item for item in result.items if item.number == 9)
    assert depth_item.status == "not_available"
    assert not result.metrics.find("H12")
    assert not result.metrics.find("H1", "broad_jsd", region="shared_mask")
    composition = next(item for item in result.items if item.number == 2)
    assert any("mask not applied" in note for note in composition.notes)


def test_a_provisional_panel_gets_its_banner(
    tmp_path: Path, fake_mmc: FakeMmc, make_trust: MakeTrust
) -> None:
    setup = _resolved(tmp_path / "run", fake_mmc, make_trust, state="provisional")
    result = build_annotation_report(
        _sources(setup),
        tmp_path / "report",
        options=OPTIONS,
        strict=True,
        make_figures=False,
    )
    first = next(item for item in result.items if item.number == 1)
    codes = {banner.code for banner in first.banners}
    assert "panel_provisional" in codes and "gate_warning" in codes
    assert "PROVISIONAL panel" in result.html.read_text()
    assert {row["trust_state"] for row in result.metrics.datasets.values()} == {
        "provisional"
    }


def test_a_refused_panel_report_states_it(tmp_path: Path, fake_mmc: FakeMmc) -> None:
    setup = _setup(tmp_path / "run", fake_mmc, source="clustered")
    manifest = pl.load_map_manifest(setup.map_dir / MAP_MANIFEST_NAME)
    manifest.model_copy(
        update={
            "samples": {},
            "panel_status": "refused",
            "panel_reasons": ["intersection: species_mismatch"],
        }
    ).write(setup.map_dir / MAP_MANIFEST_NAME)
    annotate_resolve(
        setup.map_dir,
        setup.config,
        output_dir=setup.root / "resolve_out",
        samples=setup.samples,
        n_bootstrap=10,
    )
    result = build_annotation_report(
        _sources(setup),
        tmp_path / "report",
        options=OPTIONS,
        strict=True,
        make_figures=False,
    )
    first = next(item for item in result.items if item.number == 1)
    codes = {banner.code for banner in first.banners}
    assert "gate_failed" in codes
    assert "panel_refused" in codes
    assert "REFUSED panel" in result.html.read_text()
    for record in result.metrics.find("H7", "confident_coverage_table"):
        assert record.value == 0.0
    assert {row["gate_level"] for row in result.metrics.datasets.values()} == {"failed"}


def test_output_inside_the_inputs_or_non_empty_is_refused(
    tmp_path: Path, fake_mmc: FakeMmc, make_trust: MakeTrust
) -> None:
    setup = _resolved(tmp_path / "run", fake_mmc, make_trust)
    sources = _sources(setup)
    with pytest.raises(ReportOutputError, match="inside the input"):
        check_output_dir(setup.root / "resolve_out" / "annotation_report", sources)
    check_output_dir(
        setup.root / "resolve_out" / "annotation_report",
        sources,
        allow_results_output=True,
    )
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "file.txt").write_text("x")
    with pytest.raises(ReportOutputError, match="not empty"):
        build_annotation_report(sources, busy, options=OPTIONS, make_figures=False)
    check_output_dir(busy, sources, overwrite=True)


def _results_tree(setup: Setup, root: Path, *, legacy: bool = False) -> Path:
    """Copy the synthetic outputs into the published results layout.

    ``legacy`` also writes the legacy ``clustering_squidpy``,
    ``compute_cortical_depth`` and ``mender`` outputs beside the map_first
    ones (``_mapfirst``), as a results root that ran both modes holds them.
    """
    base = root / "PX" / "proseg_hybrid"
    shutil.copytree(
        setup.root / "resolve_out",
        base / "annotation_resolve" / "annotation_resolve_out",
    )
    shutil.copytree(setup.map_dir, base / "annotation_map" / "annotation_map_out")
    suffixes = ("_mapfirst", "") if legacy else ("_mapfirst",)
    for index, sample in enumerate(setup.samples):
        platform = sample.platform.lower()
        for suffix in suffixes:
            target = (
                base
                / f"clustering_squidpy{suffix}"
                / "clustering_squidpy_out"
                / platform
                / f"{sample.sample_id}_clustered.h5ad"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sample.h5ad_path, target)
            if not legacy:
                continue
            depth = (
                root
                / "PX"
                / platform
                / f"compute_cortical_depth{suffix}"
                / "compute_cortical_depth_out"
                / "proseg_hybrid"
                / f"px_{platform}_proseg_hybrid_cells_with_cortical_depth.parquet"
            )
            _depth_parquet(setup, index, depth)
            mender = base / f"mender{suffix}" / "mender_out" / platform
            mender.mkdir(parents=True, exist_ok=True)
            (mender / "mender_manifest.json").write_text(
                json.dumps({"n_cells": 1, "suffix": suffix})
            )
    return root


def test_discover_sources_finds_the_published_layout(
    tmp_path: Path, fake_mmc: FakeMmc, make_trust: MakeTrust
) -> None:
    setup = _resolved(tmp_path / "run", fake_mmc, make_trust)
    results = _results_tree(setup, tmp_path / "results")
    sources = discover_sources(results, "PX", "proseg_hybrid")
    assert sources.resolve_dir.is_dir() and sources.map_dir is not None
    assert set(sources.clustered_h5ad) == {"PX_MERSCOPE", "PX_XENIUM"}
    assert all(
        "clustering_squidpy_mapfirst" in str(path)
        for path in sources.clustered_h5ad.values()
    )
    assert sources.cortical_depth == {} and sources.alignment_dir is None


def test_discover_sources_prefers_the_map_first_outputs_over_legacy_ones(
    tmp_path: Path, fake_mmc: FakeMmc, make_trust: MakeTrust
) -> None:
    """M7 review: legacy outputs beside the map_first ones must not be read.

    On P7513 the legacy clustered H5ADs' MERSCOPE coordinates differ from the
    current pipeline's by about 80-90 µm (H1's shared mask, the tiles).
    """
    setup = _resolved(tmp_path / "run", fake_mmc, make_trust)
    results = _results_tree(setup, tmp_path / "results", legacy=True)
    sources = discover_sources(results, "PX", "proseg_hybrid")
    for mapping, step in (
        (sources.clustered_h5ad, "clustering_squidpy_mapfirst"),
        (sources.cortical_depth, "compute_cortical_depth_mapfirst"),
        (sources.mender, "mender_mapfirst"),
    ):
        assert set(mapping) == {"PX_MERSCOPE", "PX_XENIUM"}, step
        for path in mapping.values():
            assert step in path.parts, (step, path)
    result = build_annotation_report(
        sources, tmp_path / "report", options=OPTIONS, strict=True, make_figures=False
    )
    for sample_id, record in result.metrics.provenance["samples"].items():
        assert "compute_cortical_depth_mapfirst" in record["cortical_depth"], sample_id


def test_cli_builds_a_report_from_a_results_tree(
    tmp_path: Path, fake_mmc: FakeMmc, make_trust: MakeTrust
) -> None:
    setup = _resolved(tmp_path / "run", fake_mmc, make_trust)
    results = _results_tree(setup, tmp_path / "results")
    out = tmp_path / "out"
    base = [
        "annotation-report",
        "--pair",
        "PX",
        "--segmentation",
        "proseg_hybrid",
        "--results-root",
        str(results),
    ]
    runner = CliRunner()
    result = runner.invoke(
        cli_main,
        [*base, "--out", str(out), "--n-bootstrap", "10", "--strict", "--no-figures"],
    )
    assert result.exit_code == 0, result.output
    assert "annotation report:" in result.output
    assert (out / "report.html").is_file() and (
        out / "acceptance_metrics.json"
    ).is_file()
    inside = runner.invoke(
        cli_main,
        [
            *base,
            "--out",
            str(results / "PX" / "proseg_hybrid" / "annotation_report"),
            "--no-figures",
        ],
    )
    assert inside.exit_code != 0 and "inside the input" in inside.output
    missing = runner.invoke(
        cli_main,
        [
            "annotation-report",
            "--pair",
            "PX",
            "--segmentation",
            "s",
            "--out",
            str(tmp_path / "x"),
        ],
    )
    assert (
        missing.exit_code != 0 and "--results-root or --resolve-dir" in missing.output
    )
    subset = runner.invoke(
        cli_main,
        [
            "annotation-report",
            "--pair",
            "PX",
            "--segmentation",
            "proseg_hybrid",
            "--resolve-dir",
            str(setup.root / "resolve_out"),
            "--out",
            str(tmp_path / "subset"),
            "--items",
            "1,2",
            "--no-figures",
            "--no-expression",
        ],
    )
    assert subset.exit_code == 0, subset.output
    items = json.loads((tmp_path / "subset" / "acceptance_metrics.json").read_text())[
        "items"
    ]
    assert set(items) == {
        "item01_annotatability",
        "item02_composition",
        "item12_provenance",
    }


def test_cli_reads_the_pipelines_staged_layout(
    tmp_path: Path, fake_mmc: FakeMmc, make_trust: MakeTrust
) -> None:
    """ANNOTATION_REPORT's arguments: explicit dirs, depth dirs per sample.

    The process stages each platform's compute_cortical_depth_out and passes
    it with --cortical-depth-dir SAMPLE_ID=DIR; the report finds the
    segmentation's cell table inside it (M7 stage B).
    """
    setup = _resolved(tmp_path / "run", fake_mmc, make_trust)
    arguments = [
        "annotation-report",
        "--pair",
        "PX",
        "--segmentation",
        "proseg_hybrid",
        "--resolve-dir",
        str(setup.root / "resolve_out"),
        "--map-dir",
        str(setup.map_dir),
        "--panel-dir",
        str(setup.panel_dir),
        "--no-mender",
        "--no-alignment",
        "--out",
        str(tmp_path / "report"),
        "--n-bootstrap",
        "10",
        "--strict",
        "--no-figures",
    ]
    for index, sample in enumerate(setup.samples):
        out = tmp_path / "staged" / f"cortical_depth_{index + 1}"
        depth_out = out / "compute_cortical_depth_out"
        name = f"{sample.sample_id.lower()}_proseg_hybrid_cells_with_cortical_depth"
        _depth_parquet(setup, index, depth_out / "proseg_hybrid" / f"{name}.parquet")
        arguments += [
            "--clustered-h5ad",
            f"{sample.sample_id}={sample.h5ad_path}",
            "--cortical-depth-dir",
            f"{sample.sample_id}={depth_out}",
        ]
    result = CliRunner().invoke(cli_main, arguments)
    assert result.exit_code == 0, result.output
    document = json.loads((tmp_path / "report" / "acceptance_metrics.json").read_text())
    assert document["items"]["item09_cortical_depth"]["status"] == "ok"
    for sample in setup.samples:
        assert document["datasets"][sample.sample_id]["cortical_depth"] is True
    # A depth directory without the segmentation's table gives no depth.
    empty = tmp_path / "empty" / "compute_cortical_depth_out"
    (empty / "reseg").mkdir(parents=True)
    arguments = [
        value if value != str(tmp_path / "report") else str(tmp_path / "report2")
        for value in arguments
    ]
    arguments = [
        f"{value.split('=', 1)[0]}={empty}"
        if "compute_cortical_depth_out" in value
        else value
        for value in arguments
    ]
    result = CliRunner().invoke(cli_main, arguments)
    assert result.exit_code == 0, result.output
    document = json.loads(
        (tmp_path / "report2" / "acceptance_metrics.json").read_text()
    )
    assert document["items"]["item09_cortical_depth"]["status"] != "ok"
    for sample in setup.samples:
        assert document["datasets"][sample.sample_id]["cortical_depth"] is False


def test_a_broad_only_scope_withholds_supercluster_statistics(
    tmp_path: Path, fake_mmc: FakeMmc, make_trust: MakeTrust
) -> None:
    setup = _resolved(tmp_path / "run", fake_mmc, make_trust)
    path = setup.root / "resolve_out" / "PX_resolve_summary.json"
    summary = json.loads(path.read_text())
    summary["pair"]["cross_platform"]["statistics_level"] = "broad_only"
    summary["pair"]["cross_platform"]["flag"] = True
    summary["pair"]["cross_platform"]["reasons"] = ["intersection_genes:80<100"]
    path.write_text(json.dumps(summary))
    result = build_annotation_report(
        _sources(setup),
        tmp_path / "report",
        options=OPTIONS,
        strict=True,
        make_figures=False,
    )
    (withheld,) = result.metrics.find(
        "report", "supercluster_jsd", region="whole_section"
    )
    assert withheld.status == "withheld" and withheld.value is None
    (h1,) = result.metrics.find("H1", "broad_jsd", region="whole_section")
    assert h1.status == "measured" and h1.value is not None
    item = next(item for item in result.items if item.number == 7)
    assert any("broad_only" in note for note in item.notes)


def _add_setc_runs(setup: Setup) -> None:
    """Add a WHB set c run per sample (the primary calls on the set c panel)."""
    manifest = pl.load_map_manifest(setup.map_dir / MAP_MANIFEST_NAME)
    bundle = setup.bundles["whb_frontal_supc_clus"]
    samples = {}
    for sample_id, record in manifest.samples.items():
        primary = record.runs["whb_frontal_supc_clus"]
        table_ids = set(
            pd.read_parquet(setup.map_dir / primary.parquet)["cell_id"].astype(str)
        )
        frame = setup.calls[record.platform]
        cells = frame[frame["cell_id"].isin(table_ids)].reset_index(drop=True)
        run_id = "whb_frontal_supc_clus_setc"
        parquet = (
            setup.map_dir
            / record.platform.lower()
            / f"{sample_id}_mmc_{run_id}.parquet"
        )
        write_tidy_parquet(_whb_tidy(cells), parquet, {"synthetic": True})
        run = _run_record(
            run_id, bundle, setup.panel, parquet, setup.map_dir, len(cells)
        ).model_copy(update={"purposes": ["setc_sensitivity"]})
        samples[sample_id] = record.model_copy(
            update={"runs": {**record.runs, run_id: run}}
        )
    manifest.model_copy(update={"samples": samples}).write(
        setup.map_dir / MAP_MANIFEST_NAME
    )


def test_set_c_sits_beside_set_a_for_every_composition_statement(
    tmp_path: Path, fake_mmc: FakeMmc, make_trust: MakeTrust
) -> None:
    """Plan §9 item 7: set c beside set a (soft, soft >= 30, argmax, supercluster)."""
    setup = _setup(tmp_path / "run", fake_mmc, source="clustered")
    _add_setc_runs(setup)
    _resolve(setup, make_trust)
    labels = pd.read_parquet(
        setup.root / "resolve_out" / "merscope" / "PX_MERSCOPE_celltype_labels.parquet"
    )
    assert "mmc_whb_setc_supercluster_runner_up_1_bp" in labels.columns
    result = build_annotation_report(
        _sources(setup),
        tmp_path / "report",
        options=OPTIONS,
        strict=True,
        make_figures=False,
        items=[7],
    )
    jsd = pd.read_csv(tmp_path / "report" / "tables" / "item07_cross_platform__jsd.csv")
    setc = jsd[jsd["variant"] == "set_c"]
    assert set(zip(setc["level"], setc["kind"], strict=True)) == {
        ("broad", "soft"),
        ("broad", "soft_ge30"),
        ("broad", "argmax"),
        ("supercluster", "soft"),
    }
    # The set c calls equal set a's here: the same soft JSD.
    seta = jsd[(jsd["variant"] == "set_a") & (jsd["level"] == "supercluster")]
    assert setc[setc["level"] == "supercluster"]["jsd"].iloc[0] == pytest.approx(
        seta["jsd"].iloc[0]
    )
    (record,) = result.metrics.find(
        "report", "supercluster_jsd", kind="set_c:soft", region="whole_section"
    )
    assert record.status == "measured" and record.value is not None
    (h1,) = result.metrics.find("H1", "broad_jsd", region="whole_section")
    assert h1.kind == "set_a:soft"


def test_a_per_platform_pair_takes_every_cross_platform_statement_from_xpanel(
    tmp_path: Path, fake_mmc: FakeMmc, make_trust: MakeTrust
) -> None:
    """Plan §8.5: own-panel results of a per_platform pair are never compared.

    The intersection runs call every cell an astrocyte on both platforms
    (JSD 0), while the own-panel runs differ.
    """
    setup = _setup(tmp_path / "run", fake_mmc, source="clustered")
    _add_xpanel_runs(setup)
    _set_panel_mode(setup, "per_platform")
    _resolve(setup, make_trust)
    summary = json.loads(
        (setup.root / "resolve_out" / "PX_resolve_summary.json").read_text()
    )
    assert summary["pair"]["cross_platform"]["jsd_purpose"] == "intersection_xpanel"
    recorded = next(
        row
        for row in summary["pair"]["jsd"]
        if row["kind"] == "soft" and row["region"] == "whole_section"
    )
    result = build_annotation_report(
        _sources(setup),
        tmp_path / "report",
        options=ReportOptions(n_bootstrap=int(summary["pair"]["n_bootstrap"])),
        strict=True,
        make_figures=False,
    )
    (h1,) = result.metrics.find("H1", "broad_jsd", region="whole_section")
    assert h1.kind == "intersection_run:soft" and h1.source == "resolve_summary"
    assert h1.value == pytest.approx(recorded["jsd"], abs=1e-9)
    assert h1.value == pytest.approx(0.0, abs=1e-6)
    # The note's recomputation is the same (intersection) run, not set a.
    note_value = float(h1.note.split("report recomputation ")[1].split(" ")[0])
    assert note_value == pytest.approx(0.0, abs=1e-6)
    assert "(intersection_run)" in h1.note
    own = result.metrics.find(
        "report", "broad_jsd", kind="set_a:soft", region="whole_section"
    )
    assert len(own) == 1 and own[0].status == "withheld" and own[0].value is None
    assert own[0].note.startswith("own_panel_not_comparable")
    (confident,) = result.metrics.find(
        "report", "broad_jsd", kind="intersection_run:confident"
    )
    assert confident.status == "withheld"
    # The figure plots the intersection run only (the own panel is withheld).
    figure = pd.read_csv(
        tmp_path / "report" / "figures" / "item07_cross_platform_jsd.csv"
    )
    assert set(figure["variant"]) == {"intersection_run"}
    soft = figure[(figure["level"] == "broad") & (figure["kind"] == "soft")]
    assert soft["jsd"].to_numpy() == pytest.approx(0.0, abs=1e-6)
    # Density, gene factors and pseudobulk r: the intersection run's argmax.
    density = result.metrics.find("report", "density_correlation")
    kinds = {record.kind: record for record in density}
    assert {"argmax", "soft", "confident"} <= set(kinds)
    assert kinds["confident"].status == "withheld"
    assert "mmc_whb_xpanel" in kinds["argmax"].note
    pseudo = [
        record
        for record in result.metrics.find("report", "label_pseudobulk_r")
        if record.status == "measured"
    ]
    assert pseudo and {record.kind for record in pseudo} == {"argmax"}
    assert {record.group for record in pseudo} == {"Astrocytes"}
    factors = pd.read_csv(tmp_path / "report" / "PX_platform_gene_factors.csv")
    assert set(factors["label"]) == {"Astrocytes"}
    assert set(factors["label_basis"]) == {"argmax"}
    withheld = [
        record
        for record in result.metrics.find("report", "label_pseudobulk_r")
        if record.status == "withheld"
    ]
    assert len(withheld) == 1 and withheld[0].kind == "confident"


def test_a_broad_only_dataset_gets_a_broad_level_dotplot(
    tmp_path: Path, fake_mmc: FakeMmc, make_trust: MakeTrust
) -> None:
    """Item 4 falls back to confident broad labels when the leaf is withheld."""
    setup = _resolved(tmp_path / "run", fake_mmc, make_trust)
    path = setup.root / "resolve_out" / "PX_resolve_summary.json"
    summary = json.loads(path.read_text())
    summary["samples"]["PX_MERSCOPE"]["resolution"]["gate"]["level"] = "broad_only"
    path.write_text(json.dumps(summary))
    result = build_annotation_report(
        _sources(setup),
        tmp_path / "report",
        options=OPTIONS,
        strict=True,
        make_figures=False,
        items=[4],
    )
    figures = tmp_path / "report" / "figures"
    merscope = pd.read_csv(
        figures / "item04_reference_expectation_dotplot_merscope.csv"
    )
    xenium = pd.read_csv(figures / "item04_reference_expectation_dotplot_xenium.csv")
    assert set(merscope["level"]) == {"broad"}
    assert set(merscope["label"]) <= set(HUMAN_BROAD_CLASSES)
    assert merscope["ref_fraction"].notna().all()
    assert set(xenium["level"]) == {"supercluster"}
    records = result.metrics.find("report", "pseudobulk_centroid_r")
    levels = {record.sample_id: record.level for record in records}
    assert levels == {"PX_MERSCOPE": "broad", "PX_XENIUM": "supercluster"}
    item = next(item for item in result.items if item.number == 4)
    assert any("PX_MERSCOPE: the supercluster level" in note for note in item.notes)


def test_h2_and_h13_come_from_the_label_table_and_the_clustered_ids(
    tmp_path: Path, fake_mmc: FakeMmc, make_trust: MakeTrust
) -> None:
    import anndata as ad

    setup = _resolved(tmp_path / "run", fake_mmc, make_trust)
    merscope = setup.samples[0]
    clustered = ad.read_h5ad(merscope.h5ad_path)
    short = tmp_path / "short" / f"{merscope.sample_id}_clustered.h5ad"
    short.parent.mkdir()
    clustered[1:].copy().write_h5ad(short)
    sources = _sources(
        setup,
        clustered_h5ad={
            **{sample.sample_id: sample.h5ad_path for sample in setup.samples},
            merscope.sample_id: short,
        },
    )
    result = build_annotation_report(
        sources,
        tmp_path / "report",
        options=OPTIONS,
        strict=True,
        make_figures=False,
        items=[2],
    )
    (dropped,) = result.metrics.find(
        "H13", "table_ids_equal_clustered", sample_id=merscope.sample_id
    )
    assert dropped.value is False
    (kept,) = result.metrics.find(
        "H13", "table_ids_equal_clustered", sample_id=setup.samples[1].sample_id
    )
    assert kept.value is True
    for sample in setup.samples:
        labels = pd.read_parquet(
            setup.root
            / "resolve_out"
            / sample.platform.lower()
            / f"{sample.sample_id}_celltype_labels.parquet",
            columns=["in_table", "flag_implausible"],
        )
        table = labels[labels["in_table"]]
        expected = float(table["flag_implausible"].astype(bool).mean())
        assert expected < 0.5
        (h2,) = result.metrics.find(
            "H2", "flag_implausible_share", sample_id=sample.sample_id
        )
        assert h2.value == pytest.approx(expected, rel=1e-5) and h2.n == len(table)


def test_the_digest_takes_the_trust_state_from_resolve_first() -> None:
    from merxen.annotation.report import dataset_digest
    from merxen.annotation.report_inputs import ReportInputs, SampleData

    sample = SampleData(
        sample_id="S_MERSCOPE",
        platform="MERSCOPE",
        labels=pd.DataFrame({"in_table": [True]}),
        summary={
            "trust": {"state": "provisional", "validation_basis": "simulation"},
            "resolution": {"gate": {"level": "broad_only", "warning": True}},
        },
        manifest={"panel": {"panel_trust": "validated", "panel_family": "fam"}},
        labels_path=Path("labels.parquet"),
    )
    inputs = ReportInputs(
        sources=ReportSources(
            species="human", pair_id="S", segmentation="seg", resolve_dir=Path(".")
        ),
        summary={},
        samples={"S_MERSCOPE": sample},
    )
    digest = dataset_digest(inputs)["S_MERSCOPE"]
    assert digest["trust_state"] == "provisional"
    assert digest["validation_basis"] == "simulation"
    assert digest["gate_level"] == "broad_only" and digest["gate_warning"] is True
    assert digest["panel_family"] == "fam"


def test_an_explicit_depth_file_wins_over_a_depth_directory(
    tmp_path: Path, fake_mmc: FakeMmc, make_trust: MakeTrust
) -> None:
    setup = _resolved(tmp_path / "run", fake_mmc, make_trust)
    sample = setup.samples[0]
    staged = tmp_path / "staged" / "compute_cortical_depth_out"
    name = f"{sample.sample_id.lower()}_proseg_hybrid_cells_with_cortical_depth"
    _depth_parquet(setup, 0, staged / "proseg_hybrid" / f"{name}.parquet")
    explicit = _depth_parquet(setup, 0, tmp_path / "explicit" / f"{name}.parquet")
    result = CliRunner().invoke(
        cli_main,
        [
            "annotation-report",
            "--pair",
            "PX",
            "--segmentation",
            "proseg_hybrid",
            "--resolve-dir",
            str(setup.root / "resolve_out"),
            "--cortical-depth-dir",
            f"{sample.sample_id}={staged}",
            "--cortical-depth",
            f"{sample.sample_id}={explicit}",
            "--out",
            str(tmp_path / "report"),
            "--items",
            "9",
            "--no-figures",
            "--no-expression",
        ],
    )
    assert result.exit_code == 0, result.output
    document = json.loads((tmp_path / "report" / "acceptance_metrics.json").read_text())
    recorded = document["provenance"]["samples"][sample.sample_id]["cortical_depth"]
    assert recorded == str(explicit)

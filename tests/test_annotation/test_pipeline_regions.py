"""Tests for the mouse region step of MAP and its RESOLVE view (plan §7.2; M6)."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
from scipy import sparse

from merxen.annotation.config import MOUSE_CCF_REGIONS, AnnotationConfig
from merxen.annotation.mapmycells_engine import MmcBundle, MmcEngineError, level_frame
from merxen.annotation.mouse_regions import RegionShareBundle
from merxen.annotation.panel import (
    PANEL_GENES_FILE,
    REQUIRED_BUNDLES_FILE,
    AnnotationPanel,
    RequiredBundle,
    RequiredBundles,
    compute_panel_hash,
)
from merxen.annotation.pipeline import (
    MAP_MANIFEST_NAME,
    MapError,
    MapSample,
    ResolveError,
    annotate_map,
    load_map_manifest,
    load_resolve_runs,
    map_bundles,
)
from merxen.annotation.samplesheet_columns import (
    effective_section_regions,
    section_regions_by_sample,
)
from merxen.annotation.store import BundleRef
from merxen.cli import main as cli_main
from merxen.config import ClusteringSquidpySampleConfig
from merxen.io.samplesheet import parse_samplesheet

from .conftest import FakeMmc, FakeNode

LEVELS = [
    "CCN20230722_CLAS",
    "CCN20230722_SUBC",
    "CCN20230722_SUPT",
    "CCN20230722_CLUS",
]
CLAS, SUBC = LEVELS[0], LEVELS[1]
IDS = [f"ENSMUSG{index:011d}" for index in range(1, 6)]
SYMBOLS = ["Slc17a7", "Slc17a6", "Hoxb5", "Aqp4", "Other"]
NODES = [
    FakeNode("CL_01", "01 IT-ET Glut", IDS[0], {"broad_class": "Neurons"}),
    FakeNode("CL_19", "19 MB Glut", IDS[1], {"broad_class": "Neurons"}),
    FakeNode("CL_24", "24 MY Glut", IDS[2], {"broad_class": "Neurons"}),
    FakeNode("CL_30", "30 Astro-Epen", IDS[3], {"broad_class": "Astrocytes/Ependymal"}),
]
# MERFISH grey cells per division of each class; the fake tree gives every
# class one subclass, "<class> 1".
MERFISH: dict[str, dict[str, int]] = {
    "01 IT-ET Glut": {"Isocortex": 1000},
    "19 MB Glut": {"MB": 900, "MY": 100},
    "24 MY Glut": {"MY": 1000},
    "30 Astro-Epen": {"MY": 500, "Isocortex": 500},
}
SID = "AG_MERSCOPE"
TILE = 150.0


def _panel() -> AnnotationPanel:
    return AnnotationPanel(
        name="sample",
        kind="single_sample",
        species="mouse",
        platforms=["MERSCOPE"],
        sample_ids=[SID],
        panel_mode="single_sample",
        panel_hash=compute_panel_hash(IDS),
        n_genes=len(IDS),
        ensembl_ids=IDS,
        symbols=SYMBOLS,
    )


def _required(panel: AnnotationPanel) -> RequiredBundles:
    return RequiredBundles(
        pair_id="AG",
        segmentation="proseg_hybrid",
        species="mouse",
        panel_mode="single_sample",
        status="ok",
        bundles=[
            RequiredBundle(
                reference_id="wmb_panel",
                role="primary",
                species="mouse",
                purpose="annotation",
                panel_name="sample",
                panel_hash=panel.panel_hash,
                panel_file=PANEL_GENES_FILE,
                n_panel_genes=panel.n_genes,
            ),
            RequiredBundle(
                reference_id="wmb_region_share",
                role="region_share",
                species="mouse",
                purpose="panel_independent",
            ),
        ],
        n_required=2,
    )


def _region_share_bundle(root: Path) -> Path:
    share_rows, home_rows = [], []
    for class_name, counts in MERFISH.items():
        total = sum(counts.values())
        for level, name in (("class", class_name), ("subclass", f"{class_name} 1")):
            for region in MOUSE_CCF_REGIONS:
                share_rows.append(
                    {
                        "level": level,
                        "node_name": name,
                        "region": region,
                        "n": counts.get(region, 0),
                        "share": counts.get(region, 0) / total,
                        "n_grey": total,
                        "n_all": total,
                    }
                )
        home = max(counts, key=lambda region: counts[region])
        home_rows.append(
            {
                "level": "subclass",
                "node_name": f"{class_name} 1",
                "home": home,
                "home_share": counts[home] / total,
                "n_grey": total,
            }
        )
    bundle_dir = root / "wmb_region_share" / ("e" * 64)
    bundle_dir.mkdir(parents=True)
    pd.DataFrame(share_rows).to_parquet(bundle_dir / "region_share.parquet")
    pd.DataFrame(home_rows).to_parquet(bundle_dir / "region_home.parquet")
    files = [
        {"path": path.name, "size": path.stat().st_size}
        for path in sorted(bundle_dir.iterdir())
    ]
    (bundle_dir / "bundle.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "status": "complete",
                "build_hash": "e" * 64,
                "reference_id": "wmb_region_share",
                "species": "mouse",
                "role": "region_share",
                "builder_version": 3,
                "panel": None,
                "files": files,
            }
        )
    )
    return bundle_dir


def _section() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """200 tiles: Isocortex (01) left, MB (19) right, 5 MY sink calls in MB.

    The MY cells carry some MB marker counts, so without class 24 they map
    to 19 MB Glut; their own bootstrap probability (.71) is too low for them
    to vote. Two astrocytes (not neurons) never vote.
    """
    counts, xy, kinds = [], [], []
    for i in range(20):
        for j in range(10):
            marker = 0 if i < 10 else 1
            for k in range(3):
                row = [0] * 6
                row[marker] = 30
                counts.append(row)
                xy.append(((i + 0.2 + 0.3 * k) * TILE, (j + 0.5) * TILE))
                kinds.append("ctx" if marker == 0 else "mb")
    for index in range(5):
        counts.append([0, 8, 20, 0, 0, 0])
        xy.append(((12 + index + 0.5) * TILE, 3.5 * TILE))
        kinds.append("my")
    for index in range(2):
        counts.append([0, 0, 0, 30, 0, 0])
        xy.append(((2 + index + 0.5) * TILE, 2.5 * TILE))
        kinds.append("astro")
    counts.append([1, 0, 0, 0, 0, 0])  # below min_counts: not mapped
    xy.append((0.5 * TILE, 0.5 * TILE))
    kinds.append("low")
    return np.array(counts), np.array(xy, dtype=np.float64), kinds


def _write_h5ad(path: Path, *, spatial: bool = True) -> tuple[Path, list[str]]:
    counts, xy, kinds = _section()
    var_names = [*SYMBOLS, "Blank-0001"]
    var = pd.DataFrame(index=pd.Index(var_names, dtype=str))
    var["gene"] = var_names
    var["ensembl_id"] = [*IDS, "Blank-0001"]
    obs = pd.DataFrame(index=[f"M{index}" for index in range(len(counts))])
    adata = ad.AnnData(X=sparse.csr_matrix(counts.astype(np.int64)), obs=obs, var=var)
    if spatial:
        adata.obsm["spatial"] = xy
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(path)
    return path, kinds


@pytest.fixture
def mouse_setup(tmp_path: Path, fake_mmc: FakeMmc) -> dict[str, Any]:
    panel = _panel()
    panel_dir = tmp_path / "panel"
    panel.write(panel_dir / PANEL_GENES_FILE)
    required = _required(panel)
    (panel_dir / REQUIRED_BUNDLES_FILE).write_text(
        json.dumps(required.model_dump(mode="json"))
    )
    bundle_dir = fake_mmc.bundle(
        "wmb_panel",
        role="primary",
        species="mouse",
        panel_hash=panel.panel_hash,
        n_genes=panel.n_genes,
        levels=LEVELS,
        nodes=NODES,
        drop_level="CCN20230722_SUPT",
    )
    bundle = MmcBundle.from_dir(bundle_dir)
    config = AnnotationConfig(species="mouse").coupled_to_clustering(10)
    runs = map_bundles(
        required, panel_dir, {("wmb_panel", panel.panel_hash): bundle}, config
    )
    h5ad, kinds = _write_h5ad(tmp_path / "prepared" / "merscope" / f"{SID}.h5ad")
    region_dir = _region_share_bundle(tmp_path / "store")
    return {
        "panel": panel,
        "panel_dir": panel_dir,
        "bundle": bundle,
        "config": config,
        "runs": runs,
        "sample": MapSample(SID, "MERSCOPE", h5ad, "prepared"),
        "kinds": kinds,
        "region_dir": region_dir,
        "regions": RegionShareBundle.from_dir(region_dir),
    }


def _map(setup: dict[str, Any], output: Path, **kwargs: Any) -> Any:
    return annotate_map(
        [kwargs.pop("sample", setup["sample"])],
        setup["runs"],
        kwargs.pop("config", setup["config"]),
        output_dir=output,
        pair_id="AG",
        segmentation="proseg_hybrid",
        region_shares=kwargs.pop("region_shares", setup["regions"]),
        **kwargs,
    )


def test_region_step_prunes_and_remaps_only_the_dropped_cells(
    tmp_path: Path, fake_mmc: FakeMmc, mouse_setup: dict[str, Any]
) -> None:
    output = tmp_path / "out"
    manifest = _map(mouse_setup, output)

    record = manifest.samples[SID]
    regions = record.mouse_regions
    assert regions is not None
    assert (regions.status, regions.source, regions.requested) == (
        "pruned",
        "auto",
        "auto",
    )
    assert regions.n_assigned_tiles == 200
    assert regions.tile_counts == {"Isocortex": 100, "MB": 100}
    assert regions.inferred_regions == ["Isocortex", "MB"]
    assert regions.present_regions == ["Isocortex", "MB"]
    assert [(node.level, node.node, node.kind) for node in regions.nodes_to_drop] == [
        (CLAS, "CL_24", "class")
    ]
    assert regions.region_share is not None
    assert regions.region_share["build_hash"] == "e" * 64
    assert regions.n_table_cells == 607
    assert regions.n_cells_region_dropped == 5
    assert regions.n_cells_region_dropped_class == 5
    assert regions.n_cells_class_changed == 5
    assert regions.params["tile_um"] == 150.0

    # Two mapper calls: the unpruned run, then the 5 MY cells without CL_24,
    # with the lookup filtered to the pruned tree.
    assert len(fake_mmc.calls) == 2
    assert "--nodes_to_drop" not in fake_mmc.calls[0]["command"]
    remap_call = fake_mmc.calls[1]
    assert remap_call["nodes_to_drop"] == [[CLAS, "CL_24"]]
    assert not [key for key in remap_call["lookup"] if "CL_24" in key]
    assert "CCN20230722_CLAS/CL_19" in remap_call["lookup"]
    assert regions.remap is not None
    assert regions.remap.n_cells == 5
    assert regions.remap.run_id == "wmb_panel_pruned"
    assert regions.remap.nodes_to_drop == [[CLAS, "CL_24"]]
    remap_parquet = output / regions.remap.parquet
    assert remap_parquet.name == f"{SID}_mmc_wmb_panel_pruned.parquet"
    assert (output / regions.cells_parquet).name == f"{SID}_mouse_regions.parquet"
    # The run's own parquet stays unpruned.
    assert set(record.runs) == {"wmb_panel"}

    labels = pd.read_parquet(output / str(record.provisional_labels)).set_index(
        "cell_id"
    )
    kinds = pd.Series(mouse_setup["kinds"], index=labels.index)
    my_cells = kinds.index[kinds == "my"]
    assert set(labels.loc[my_cells, "ct_class_name"]) == {"19 MB Glut"}
    assert set(labels.loc[my_cells, "mmc_wmb_class_name"]) == {"19 MB Glut"}
    assert set(labels.loc[my_cells, "mmc_wmb_unpruned_class_name"]) == {"24 MY Glut"}
    assert labels.loc[my_cells, "region_pruned_changed"].all()
    assert set(labels.loc[my_cells, "inferred_region"].astype(str)) == {"MB"}
    others = kinds.index[kinds.isin(["ctx", "mb", "astro"])]
    assert not labels.loc[others, "region_pruned_changed"].any()
    assert (
        labels.loc[others, "mmc_wmb_class_name"].astype(str)
        == labels.loc[others, "mmc_wmb_unpruned_class_name"].astype(str)
    ).all()
    assert labels.loc["M607", "ct_class_status"] == "low_counts"
    assert labels.loc["M607", "region_pruned_changed"] == np.False_
    assert pd.isna(labels.loc["M607", "mmc_wmb_unpruned_class_name"])


def test_resolve_reads_the_pruned_calls_and_checks_their_digests(
    tmp_path: Path, mouse_setup: dict[str, Any]
) -> None:
    output = tmp_path / "out"
    manifest = _map(mouse_setup, output)
    record = manifest.samples[SID]

    runs = load_resolve_runs(output, record)

    run = runs["wmb_panel"]
    assert run.regions is not None and run.unpruned_tidy is not None
    kinds = pd.Series(mouse_setup["kinds"], index=[f"M{i}" for i in range(608)])
    my_cells = kinds.index[kinds == "my"]
    pruned = level_frame(run.tidy, CLAS).reindex(my_cells)
    unpruned = level_frame(run.unpruned_tidy, CLAS).reindex(my_cells)
    assert set(pruned["assignment"].astype(str)) == {"CL_19"}
    assert set(unpruned["assignment"].astype(str)) == {"CL_24"}
    assert run.regions.dropped_level(pd.Index(my_cells)).tolist() == ["class"] * 5
    assert len(run.tidy) == len(run.unpruned_tidy)

    assert record.mouse_regions is not None and record.mouse_regions.remap is not None
    remap = output / record.mouse_regions.remap.parquet
    remap.write_bytes(remap.read_bytes() + b"tampered")
    with pytest.raises(ResolveError, match="differs from the sha256"):
        load_resolve_runs(output, record)


def test_region_step_reuses_an_identical_published_remap(
    tmp_path: Path, fake_mmc: FakeMmc, mouse_setup: dict[str, Any]
) -> None:
    output = tmp_path / "out"
    first = _map(mouse_setup, output)
    assert len(fake_mmc.calls) == 2

    second = _map(mouse_setup, tmp_path / "again", reuse_from=output)

    assert len(fake_mmc.calls) == 2  # nothing mapped again
    remap = second.samples[SID].mouse_regions.remap
    assert remap is not None and remap.reused
    assert remap.parquet_sha256 == first.samples[SID].mouse_regions.remap.parquet_sha256


def test_region_step_does_not_reuse_a_remap_with_another_drop_list(
    tmp_path: Path, fake_mmc: FakeMmc, mouse_setup: dict[str, Any]
) -> None:
    output = tmp_path / "out"
    _map(mouse_setup, output)
    path = output / MAP_MANIFEST_NAME
    manifest = json.loads(path.read_text())
    remap = manifest["samples"][SID]["mouse_regions"]["remap"]
    remap["nodes_to_drop"] = [[CLAS, "CL_19"]]
    path.write_text(json.dumps(manifest))

    second = _map(mouse_setup, tmp_path / "again", reuse_from=output)

    assert len(fake_mmc.calls) == 3  # the unpruned run reused, the re-map not
    assert second.samples[SID].runs["wmb_panel"].reused
    assert not second.samples[SID].mouse_regions.remap.reused


def test_region_remap_refuses_other_recorded_nodes(
    tmp_path: Path, fake_mmc: FakeMmc, mouse_setup: dict[str, Any]
) -> None:
    fake_mmc.record_nodes_to_drop = [[CLAS, "CL_19"]]
    with pytest.raises(MmcEngineError, match="nodes_to_drop"):
        _map(mouse_setup, tmp_path / "out")


def test_override_and_none_follow_the_section_request(
    tmp_path: Path, fake_mmc: FakeMmc, mouse_setup: dict[str, Any]
) -> None:
    listed = _map(
        mouse_setup,
        tmp_path / "listed",
        section_regions={SID: "isocortex;mb;my"},
    )
    regions = listed.samples[SID].mouse_regions
    assert regions is not None
    assert (regions.source, regions.status) == ("override", "no_nodes_dropped")
    assert regions.present_regions == ["Isocortex", "MB", "MY"]
    assert regions.inferred_regions == ["Isocortex", "MB"]  # QC, not used
    assert "differs from the inferred divisions" in regions.reasons[0]
    assert regions.remap is None and len(fake_mmc.calls) == 1

    only_ctx = _map(mouse_setup, tmp_path / "ctx", section_regions={SID: "Isocortex"})
    names = [node.name for node in only_ctx.samples[SID].mouse_regions.nodes_to_drop]
    assert names == ["19 MB Glut", "24 MY Glut"]  # 30 Astro-Epen is never dropped

    config = AnnotationConfig(
        species="mouse", mouse_section_regions="none"
    ).coupled_to_clustering(10)
    disabled = _map(mouse_setup, tmp_path / "none", config=config, region_shares=None)
    regions = disabled.samples[SID].mouse_regions
    assert (regions.source, regions.status) == ("none", "disabled")
    assert regions.region_share is None and regions.nodes_to_drop == []


def test_auto_without_coordinates_skips_pruning(
    tmp_path: Path, mouse_setup: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    h5ad, _ = _write_h5ad(tmp_path / "flat" / f"{SID}.h5ad", spatial=False)
    with caplog.at_level(logging.WARNING, logger="merxen.annotation.pipeline"):
        manifest = _map(
            mouse_setup,
            tmp_path / "out",
            sample=MapSample(SID, "MERSCOPE", h5ad, "prepared"),
        )
    regions = manifest.samples[SID].mouse_regions
    assert regions is not None and regions.status == "skipped_no_coordinates"
    assert regions.remap is None
    assert "no cell coordinates" in caplog.text


def test_map_refuses_region_settings_it_cannot_apply(
    tmp_path: Path, fake_mmc: FakeMmc, mouse_setup: dict[str, Any]
) -> None:
    with pytest.raises(MapError, match="needs the wmb_region_share bundle"):
        _map(mouse_setup, tmp_path / "a", region_shares=None)
    coupled = AnnotationConfig.model_validate(
        {
            "species": "mouse",
            "mouse_regions": {"coupled_regions": {"OB": ["OLF"]}},
        }
    ).coupled_to_clustering(10)
    with pytest.raises(MapError, match="coupled_regions"):
        _map(mouse_setup, tmp_path / "b", config=coupled)
    with pytest.raises(MapError, match="unknown mouse section region"):
        _map(mouse_setup, tmp_path / "c", section_regions={SID: "Striatum"})
    assert not fake_mmc.calls  # all refused before mapping


def test_human_map_refuses_mouse_section_regions(
    tmp_path: Path, fake_mmc: FakeMmc
) -> None:
    config = AnnotationConfig(species="human").coupled_to_clustering(10)
    with pytest.raises(MapError, match="mouse runs only"):
        annotate_map(
            [],
            [],
            config,
            output_dir=tmp_path / "out",
            pair_id="P",
            segmentation="s",
            section_regions={"P_MERSCOPE": "auto"},
        )


# --------------------------------------------------------------------------
# Samplesheet column -> samples_json -> MAP (the pipeline's arguments)


def test_samplesheet_column_reaches_the_sample_config(tmp_path: Path) -> None:
    sheet = tmp_path / "samplesheet.csv"
    with sheet.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pair_id", "mouse_section_regions"])
        writer.writeheader()
        writer.writerow({"pair_id": "AG", "mouse_section_regions": "isocortex;hpf"})
        writer.writerow({"pair_id": "VZ", "mouse_section_regions": " "})
        writer.writerow({"pair_id": "NO", "mouse_section_regions": "None"})
    pairs = {pair.pair_id: pair for pair in parse_samplesheet(sheet)}
    assert pairs["AG"].mouse_section_regions == "Isocortex;HPF"
    assert pairs["VZ"].mouse_section_regions is None
    assert pairs["NO"].mouse_section_regions == "none"

    samples = [
        ClusteringSquidpySampleConfig(
            sample_id=f"{pair_id}_MERSCOPE",
            platform="MERSCOPE",
            zarr_path=tmp_path / "x.zarr",
            mouse_section_regions=pair.mouse_section_regions,
        ).model_dump(mode="json")
        for pair_id, pair in pairs.items()
    ]
    by_sample = section_regions_by_sample(samples)
    assert by_sample == {"AG_MERSCOPE": "Isocortex;HPF", "NO_MERSCOPE": "none"}
    assert effective_section_regions(by_sample.get("VZ_MERSCOPE"), "auto") == "auto"
    assert effective_section_regions("th", "none") == "TH"
    with pytest.raises(ValueError, match="unknown mouse section region"):
        section_regions_by_sample([{"sample_id": "S", "mouse_section_regions": "X"}])
    with pytest.raises(ValueError, match="no sample_id"):
        section_regions_by_sample([{"mouse_section_regions": "auto"}])
    with pytest.raises(ValueError):
        ClusteringSquidpySampleConfig(
            sample_id="S",
            platform="MERSCOPE",
            zarr_path=tmp_path / "x.zarr",
            mouse_section_regions="Striatum",
        )


def _pipeline_inputs(
    root: Path, setup: dict[str, Any], section_regions: str | None
) -> list[str]:
    prepared = root / "clustering_prepare_out"
    (prepared / "merscope").mkdir(parents=True)
    target = prepared / "merscope" / f"{SID}_prepared.h5ad"
    target.write_bytes(setup["sample"].h5ad_path.read_bytes())
    (prepared / "manifest.json").write_text(
        json.dumps({"samples": {SID: f"merscope/{SID}_prepared.h5ad"}})
    )
    sample: dict[str, Any] = {"sample_id": SID, "platform": "MERSCOPE"}
    if section_regions is not None:
        sample["mouse_section_regions"] = section_regions
    config = root / "clustering_squidpy_config.json"
    config.write_text(
        json.dumps({"pair_id": "AG", "min_counts": 10, "samples": [sample]})
    )
    refs = []
    for index, (reference_id, role, panel_hash, build_hash, path) in enumerate(
        [
            (
                "wmb_panel",
                "primary",
                setup["panel"].panel_hash,
                setup["bundle"].build_hash,
                setup["bundle"].path,
            ),
            ("wmb_region_share", "region_share", None, "e" * 64, setup["region_dir"]),
        ]
    ):
        refs.append(
            BundleRef(
                reference_id=reference_id,
                species="mouse",
                role=role,
                panel_hash=panel_hash,
                build_hash=build_hash,
                path=str(path),
                store_root=str(Path(path).parents[1]),
            ).write(root / "refs" / f"bundle_ref_{index + 1}.json")
        )
    return [
        "annotate",
        "--species",
        "mouse",
        "--prepared-dir",
        str(prepared),
        "--clustering-config",
        str(config),
        "--segmentation",
        "proseg_hybrid",
        "--panel-dir",
        str(setup["panel_dir"]),
        *(item for ref in refs for item in ("--bundle-ref", str(ref))),
        "--require-bundle-refs",
        "--allow-refused-panel",
        "--n-processors",
        "2",
    ]


@pytest.mark.parametrize(
    ("sample_value", "extra", "expected"),
    [
        (None, [], ("auto", "pruned")),
        ("Isocortex;MB;MY", [], ("Isocortex;MB;MY", "no_nodes_dropped")),
        ("none", [], ("none", "disabled")),
        ("Isocortex;MB;MY", ["--mouse-section-regions", "auto"], ("auto", "pruned")),
        (
            None,
            ["--mouse-section-regions", f"{SID}=none"],
            ("none", "disabled"),
        ),
    ],
)
def test_cli_annotate_takes_the_sample_section_regions(
    tmp_path: Path,
    mouse_setup: dict[str, Any],
    sample_value: str | None,
    extra: list[str],
    expected: tuple[str, str],
) -> None:
    arguments = _pipeline_inputs(tmp_path / "inputs", mouse_setup, sample_value)
    output = tmp_path / "annotation_map_out"

    result = CliRunner().invoke(cli_main, [*arguments, *extra, "--out", str(output)])

    assert result.exit_code == 0, result.output
    regions = load_map_manifest(output / MAP_MANIFEST_NAME).samples[SID].mouse_regions
    assert regions is not None
    assert (regions.requested, regions.status) == expected
    assert f"{SID} regions" in result.output


def test_cli_annotate_needs_the_staged_region_share_ref(
    tmp_path: Path, mouse_setup: dict[str, Any]
) -> None:
    arguments = _pipeline_inputs(tmp_path / "inputs", mouse_setup, None)
    refs = [index for index, item in enumerate(arguments) if item == "--bundle-ref"]
    del arguments[refs[1] : refs[1] + 2]  # drop the wmb_region_share ref

    result = CliRunner().invoke(cli_main, [*arguments, "--out", str(tmp_path / "out")])
    assert result.exit_code != 0
    assert "no --bundle-ref for wmb_region_share" in result.output

    none = CliRunner().invoke(
        cli_main,
        [
            *arguments,
            "--mouse-section-regions",
            "none",
            "--out",
            str(tmp_path / "out_none"),
        ],
    )
    assert none.exit_code == 0, none.output

    human = CliRunner().invoke(
        cli_main,
        [
            *[item if item != "mouse" else "human" for item in arguments],
            "--mouse-section-regions",
            "auto",
            "--out",
            str(tmp_path / "out_human"),
        ],
    )
    assert human.exit_code != 0
    assert "mouse runs only" in human.output

"""Tests for the mouse region step of MAP and its RESOLVE view (plan §7.2; M6)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from merxen.annotation.config import AnnotationConfig
from merxen.annotation.mapmycells_engine import MmcEngineError, level_frame
from merxen.annotation.pipeline import (
    MAP_MANIFEST_NAME,
    MapError,
    MapSample,
    ResolveError,
    annotate_map,
    load_map_manifest,
    load_resolve_runs,
)
from merxen.annotation.store import BundleRef
from merxen.cli import main as cli_main

from .conftest import MOUSE_CLAS as CLAS
from .conftest import MOUSE_SID as SID
from .conftest import FakeMmc, map_mouse, write_mouse_h5ad


def test_region_step_prunes_and_remaps_only_the_dropped_cells(
    tmp_path: Path, fake_mmc: FakeMmc, mouse_setup: dict[str, Any]
) -> None:
    output = tmp_path / "out"
    manifest = map_mouse(mouse_setup, output)

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
    assert regions.n_table_cells == 608
    assert regions.n_cells_region_dropped == 6
    assert regions.n_cells_region_dropped_class == 6
    assert regions.n_cells_class_changed == 6
    assert regions.params["tile_um"] == 150.0

    # Two mapper calls: the unpruned run, then the 6 MY cells without CL_24,
    # with the lookup filtered to the pruned tree.
    assert len(fake_mmc.calls) == 2
    assert "--nodes_to_drop" not in fake_mmc.calls[0]["command"]
    remap_call = fake_mmc.calls[1]
    assert remap_call["nodes_to_drop"] == [[CLAS, "CL_24"]]
    assert not [key for key in remap_call["lookup"] if "CL_24" in key]
    assert "CCN20230722_CLAS/CL_19" in remap_call["lookup"]
    assert regions.remap is not None
    assert regions.remap.n_cells == 6
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
    assert labels.loc["M0", "ct_class_status"] == "low_counts"
    assert labels.loc["M0", "region_pruned_changed"] == np.False_
    assert pd.isna(labels.loc["M0", "mmc_wmb_unpruned_class_name"])
    lost = kinds.index[kinds == "my_lost"]
    assert labels.loc[lost, "region_pruned_changed"].all()
    assert set(labels.loc[lost, "mmc_wmb_unpruned_class_name"]) == {"24 MY Glut"}
    assert set(labels.loc[lost, "inferred_region"].astype(str)) == {"MB"}
    # Coordinates follow the cell ids (the off-table object comes first).
    ctx = kinds.index[kinds == "ctx"]
    assert set(labels.loc[ctx, "inferred_region"].astype(str)) == {"Isocortex"}


def test_resolve_reads_the_pruned_calls_and_checks_their_digests(
    tmp_path: Path, mouse_setup: dict[str, Any]
) -> None:
    output = tmp_path / "out"
    manifest = map_mouse(mouse_setup, output)
    record = manifest.samples[SID]

    runs = load_resolve_runs(output, record)

    run = runs["wmb_panel"]
    assert run.regions is not None and run.unpruned_tidy is not None
    kinds = pd.Series(mouse_setup["kinds"], index=[f"M{i}" for i in range(609)])
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
    first = map_mouse(mouse_setup, output)
    assert len(fake_mmc.calls) == 2

    second = map_mouse(mouse_setup, tmp_path / "again", reuse_from=output)

    assert len(fake_mmc.calls) == 2  # nothing mapped again
    remap = second.samples[SID].mouse_regions.remap
    assert remap is not None and remap.reused
    assert remap.parquet_sha256 == first.samples[SID].mouse_regions.remap.parquet_sha256


def test_region_step_does_not_reuse_a_remap_with_another_drop_list(
    tmp_path: Path, fake_mmc: FakeMmc, mouse_setup: dict[str, Any]
) -> None:
    output = tmp_path / "out"
    map_mouse(mouse_setup, output)
    path = output / MAP_MANIFEST_NAME
    manifest = json.loads(path.read_text())
    remap = manifest["samples"][SID]["mouse_regions"]["remap"]
    remap["nodes_to_drop"] = [[CLAS, "CL_19"]]
    path.write_text(json.dumps(manifest))

    second = map_mouse(mouse_setup, tmp_path / "again", reuse_from=output)

    assert len(fake_mmc.calls) == 3  # the unpruned run reused, the re-map not
    assert second.samples[SID].runs["wmb_panel"].reused
    assert not second.samples[SID].mouse_regions.remap.reused


def test_region_remap_refuses_other_recorded_nodes(
    tmp_path: Path, fake_mmc: FakeMmc, mouse_setup: dict[str, Any]
) -> None:
    fake_mmc.record_nodes_to_drop = [[CLAS, "CL_19"]]
    with pytest.raises(MmcEngineError, match="nodes_to_drop"):
        map_mouse(mouse_setup, tmp_path / "out")


def test_override_and_none_follow_the_section_request(
    tmp_path: Path, fake_mmc: FakeMmc, mouse_setup: dict[str, Any]
) -> None:
    listed = map_mouse(
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

    only_ctx = map_mouse(
        mouse_setup, tmp_path / "ctx", section_regions={SID: "Isocortex"}
    )
    names = [node.name for node in only_ctx.samples[SID].mouse_regions.nodes_to_drop]
    assert names == ["19 MB Glut", "24 MY Glut"]  # 30 Astro-Epen is never dropped

    config = AnnotationConfig(
        species="mouse", mouse_section_regions="none"
    ).coupled_to_clustering(10)
    disabled = map_mouse(
        mouse_setup, tmp_path / "none", config=config, region_shares=None
    )
    regions = disabled.samples[SID].mouse_regions
    assert (regions.source, regions.status) == ("none", "disabled")
    assert regions.region_share is None and regions.nodes_to_drop == []


def test_auto_without_coordinates_skips_pruning(
    tmp_path: Path, mouse_setup: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    h5ad, _ = write_mouse_h5ad(tmp_path / "flat" / f"{SID}.h5ad", spatial=False)
    with caplog.at_level(logging.WARNING, logger="merxen.annotation.pipeline"):
        manifest = map_mouse(
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
        map_mouse(mouse_setup, tmp_path / "a", region_shares=None)
    coupled = AnnotationConfig.model_validate(
        {
            "species": "mouse",
            "mouse_regions": {"coupled_regions": {"OB": ["OLF"]}},
        }
    ).coupled_to_clustering(10)
    with pytest.raises(MapError, match="coupled_regions"):
        map_mouse(mouse_setup, tmp_path / "b", config=coupled)
    with pytest.raises(MapError, match="unknown mouse section region"):
        map_mouse(mouse_setup, tmp_path / "c", section_regions={SID: "Striatum"})
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
# The sample's section regions -> MAP (the pipeline's arguments)


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
        (
            None,
            [
                "--mouse-section-regions",
                f"{SID}=none",
                "--mouse-section-regions",
                "Isocortex;MB;MY",
            ],
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

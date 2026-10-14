"""Tests for gate P's mouse path (M13b C20b; plan §14, pre-registration §23.9).

End to end on synthetic WMB inputs (``test_reference_draw2``'s nine clusters
with ABC donors, a fake ``cell_type_mapper`` and a marker-count mapper
standing in for MapMyCells): PREP builds the family's ``wmb_panel`` and its
self-map test set (draw 1), the second disjoint draw is built as its own
bundle in another store, and gate P maps both draws onto PREP's own bundle,
scores NP1-NP9 with the WMB composition, NP7 without the vocab, NP6's
offsets resampled for every mouse gene, and reports the per-donor spread.
No MapMyCells, no real reference.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from merxen.annotation import gate_p as gp
from merxen.annotation import gate_p_run as run
from merxen.annotation import reference
from merxen.annotation import resolvability as res
from merxen.annotation.config import AnnotationConfig, AnnotationReferenceSpec
from merxen.annotation.reference import builder_for, prepare_reference_spec
from merxen.annotation.simulate import (
    GatePPlan,
    GatePRequest,
    GatePUnavailableError,
    ReferenceBuild,
)
from merxen.annotation.store import BUNDLE_MANIFEST_NAME, ReferenceStore

from .test_reference import (
    CLAS,
    CLUS_W,
    REAL_CLASSES,
    REAL_SUBCLASS_CLUSTERS,
    SUBC_W,
    SUPT,
    FakeCtm,
    make_panel,
    marker_mapper,
    subtree,
    write_precompute,
)
from .test_reference_draw2 import write_wmb_draw_sources

LEVEL_TARGETS = ("lineage", "broad", "nt", "supercluster", "class", "subclass")


@pytest.fixture
def small_resources() -> Any:
    """PREP resources of 2 processes and 3 GB (as test_reference's fixture)."""
    previous = reference._PREP_RESOURCES
    reference.set_prep_resources(n_processors=2, max_gb=3)
    yield
    reference._PREP_RESOURCES = previous


def mouse_gate_config(**resolvability: Any) -> AnnotationConfig:
    """A version-6 mouse config whose gate-P minimums fit the fixture."""
    settings: dict[str, Any] = {
        "version": 6,
        "min_cells_per_bin": 5,
        "min_confident_n": 5,
        "gate_p_min_confident_n": 5,
        "gate_p_replicate_min_confident_n": 3,
        "gate_p_class_min_test_cells": 40,
        "gate_p_topup_max_cluster_frac": 0.3,
        "mouse_nonneuronal_extra_per_supertype": 2,
    }
    settings.update(resolvability)
    targets = {f"target_{level}": 0.5 for level in LEVEL_TARGETS}
    return AnnotationConfig(species="mouse", resolvability=settings, thresholds=targets)


def with_draw(config: AnnotationConfig, draw: int) -> AnnotationConfig:
    return config.model_copy(
        update={
            "resolvability": config.resolvability.model_copy(
                update={"gate_p_wmb_draw": draw}
            )
        }
    )


class MouseFamily:
    """A mouse family's PREP and second draw on the synthetic WMB."""

    def __init__(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        config: AnnotationConfig,
        *,
        draw2_config: AnnotationConfig | None = None,
        build: bool = True,
    ) -> None:
        sources = write_wmb_draw_sources(tmp_path, n_per_cluster=80, n_listed=6)
        genes = json.loads(str(sources["genes"]))
        tree: dict[str, Any] = {"hierarchy": [CLAS, SUBC_W, SUPT, CLUS_W]}
        tree[CLAS] = {label: subs for label, (_, subs) in REAL_CLASSES.items()}
        tree[SUBC_W] = {
            subclass: [subclass.replace("SUBC", "SUPT")]
            for subclass in REAL_SUBCLASS_CLUSTERS
        }
        tree[SUPT] = {
            subclass.replace("SUBC", "SUPT"): [
                f"CS20230722_CLUS_{alias:04d}" for alias in aliases
            ]
            for subclass, aliases in REAL_SUBCLASS_CLUSTERS.items()
        }
        tree[CLUS_W] = {f"CS20230722_CLUS_{alias:04d}": [] for alias in range(1, 10)}
        tree["name_mapper"] = {
            CLAS: {label: {"name": name} for label, (name, _) in REAL_CLASSES.items()}
        }
        mapping = write_precompute(
            tmp_path / "allen" / "precomputed_stats_ABC_revision_230821.h5",
            tree,
            genes,
            n_cells={leaf: 80 for leaf in tree[CLUS_W]},
            with_detection=False,
        )

        def marker_precompute(ctm_config: dict[str, Any]) -> None:
            import anndata as ad

            cells = pd.read_csv(ctm_config["cell_metadata_path"])
            h5ad_genes = list(
                ad.read_h5ad(ctm_config["h5ad_path_list"][0], backed="r").var_names
            )
            counts = (
                cells["cluster_alias"]
                .map(lambda alias: f"CS20230722_CLUS_{alias:04d}")
                .value_counts()
            )
            write_precompute(
                Path(ctm_config["output_path"]),
                subtree(tree, set(counts.index)),
                h5ad_genes,
                n_cells=counts.to_dict(),
            )

        def lookup(stub: list[str]) -> dict[str, Any]:
            return {"None": stub[:4], f"{CLAS}/CS20230722_CLAS_01": stub[4:6]}

        fake = FakeCtm(lookup).install(monkeypatch)
        fake.precompute = marker_precompute
        markers = {"CS20230722_CLAS_01": genes[20], "CS20230722_CLAS_30": genes[21]}
        for index, subclass in enumerate(REAL_SUBCLASS_CLUSTERS):
            markers[subclass] = genes[10 + index]
        mapper = marker_mapper(markers)
        self.prep_calls: list[dict[str, Any]] = []

        def prep_map_function(context: Any, engine: Any, runs: list[Any]) -> Any:
            def map_query(query: Any, tag: str, seed: int) -> pd.DataFrame:
                self.prep_calls.append({"engine_path": str(engine.path), "tag": tag})
                runs.append(
                    {
                        "tag": tag,
                        "engine": engine.reference_id,
                        "n_processors": reference.prep_resources().n_processors,
                    }
                )
                return mapper(engine, query)

            return map_query

        monkeypatch.setattr(reference, "_mmc_map_function", prep_map_function)
        self.gate_calls: list[dict[str, Any]] = []

        def map_function(engine: Any, *, runs: list[Any], **_: Any) -> Any:
            def map_query(query: Any, tag: str, seed: int) -> pd.DataFrame:
                self.gate_calls.append(
                    {
                        "engine": str(engine.path),
                        "tag": tag,
                        "seed": seed,
                        "cells": set(query.obs["cell_id"].astype(str)),
                    }
                )
                runs.append({"tag": tag, "engine": engine.reference_id})
                return mapper(engine, query)

            return map_query

        monkeypatch.setattr(reference, "mmc_map_function", map_function)
        self.sources = sources
        self.spec = prepare_reference_spec(
            AnnotationReferenceSpec(
                reference_id="wmb_panel",
                species="mouse",
                role="primary",
                drop_level=SUPT,
                max_cells_per_cluster=20,
                sources={
                    "wmb_h5ad_dir": sources["h5ad_dir"],
                    "wmb_metadata_dir": sources["metadata"],
                    "wmb_mapping_stats": mapping,
                    "wmb_selfmap_test_cells": sources["truth"],
                },
            )
        )
        self.config = config
        self.panel = make_panel(genes[:30], species="mouse")
        (tmp_path / "scratch").mkdir(exist_ok=True)
        self.store = ReferenceStore(
            tmp_path / "gate_p_store", scratch_root=tmp_path / "scratch"
        )
        self.builder = builder_for(self.spec, config)
        self.tmp_path = tmp_path
        if not build:
            return
        bundle = self.store.get_or_build(
            self.spec, self.panel, builder=self.builder, config=config
        )
        self.bundle_dir = Path(bundle.path)
        # The second draw, built from the same inputs in another store (the
        # production store holds it; scripts/annotation/build_wmb_draw2.py).
        draw_config = with_draw(draw2_config or config, 2)
        test_spec = reference.wmb_test_set_spec(self.spec, draw2_config or config)
        draw2 = ReferenceStore(tmp_path / "production_store").get_or_build(
            test_spec,
            self.panel,
            builder=builder_for(test_spec, draw_config),
            config=draw_config,
        )
        self.draw2_dir = Path(draw2.path)
        summary_file = self.bundle_dir / res.RESOLVABILITY_SUMMARY_FILE
        summary = json.loads(summary_file.read_text())
        self.draw1_dir = Path(summary["test_set_bundle"]["path"])

    def request(
        self, out: str = "out", *, report: Mapping[str, Any] | None = None
    ) -> GatePRequest:
        scratch = self.tmp_path / f"{out}_scratch"
        scratch.mkdir(exist_ok=True)
        return GatePRequest(
            panel=self.panel,
            config=self.config,
            store=self.store,
            bundles={"wmb_panel": self.bundle_dir},
            out_dir=self.tmp_path / out,
            report=dict(
                report
                or {
                    "platform": "MERSCOPE",
                    "settings": {"expected_depth": 60},
                    "panel": {"gene_ids": {}},
                    "references": {},
                }
            ),
            builds=(ReferenceBuild(spec=self.spec, builder=self.builder),),
            scratch_dir=scratch,
        )

    def options(self, **changes: Any) -> run.GatePOptions:
        settings: dict[str, Any] = {
            "species": "mouse",
            "wmb_draw2_bundle": self.draw2_dir,
            "dry_run": True,
            "run_with_unaccepted_parents": True,
        }
        settings.update(changes)
        return run.GatePOptions(**settings)


def read_run(result: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(Path(result["run"]).read_text())


def test_mouse_gate_p_scores_both_draws_on_preps_own_wmb_panel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_resources: Any
) -> None:
    family = MouseFamily(tmp_path, monkeypatch, mouse_gate_config())
    prep_calls = len(family.prep_calls)
    result = run.run_gate_p(family.request(), family.options())
    assert result["status"] == run.STATUS_SCORED, result
    record = read_run(result)
    out = tmp_path / "out" / run.GATE_P_RUN_DIR
    # The groups are the two draws; draw 2 is the given bundle, checked.
    assert record["default_draw"] == "draw1" and record["extra_draws"] == ["draw2"]
    assert "default_donor" not in record and "leave_one_donor_out" not in record
    assert record["draw2"]["build_hash"] == family.draw2_dir.name
    assert all(record["draw2"]["checks"].values())
    assert record["default_test_set"]["build_hash"] == family.draw1_dir.name
    assert record["options"]["wmb_draw2_bundle"] == str(family.draw2_dir)
    assert record["options"]["dry_run_exemptions"] == []
    assert set(run.GATE_P_RUN_READINGS_MOUSE) <= set(record["readings"])
    # Nothing was built in the gate-P store beyond PREP (no leave-one-out).
    built = {
        path.parent.parent.name
        for path in (tmp_path / "gate_p_store").rglob("bundle.json")
    }
    assert built == {"wmb_panel", reference.WMB_TESTSET_REFERENCE_ID}
    # Both draws were mapped onto PREP's own wmb_panel, nothing else.
    assert {call["engine"] for call in family.gate_calls} == {str(family.bundle_dir)}
    replicates = pd.DataFrame(record["replicates"])
    base = replicates[replicates["member"] == "R1_contam_HO@0"]
    assert sorted(zip(base["group"], base["seed"], strict=True)) == sorted(
        (group, seed) for group in run.MOUSE_DRAW_GROUPS for seed in (0, 1)
    )
    prep_rows = replicates[replicates["source"] == "prep_bundle"]
    assert set(prep_rows["group"]) == {"draw1"} and set(prep_rows["seed"]) == {0}
    stresses = replicates[replicates["role"] == "stress"]
    assert set(stresses["member"]) == {
        "R1_stress_spill@0",
        "R1_stress_lognormal@0",
        "R1_stress_xplatform@0",
    }
    assert sorted(stresses["group"]) == sorted([*run.MOUSE_DRAW_GROUPS] * 3)
    groups: dict[str, set[str]] = {}
    for _, row in base[base["seed"] == 0].iterrows():
        cells = pd.read_parquet(out / row["file"], columns=["cell_id"])
        groups[row["group"]] = set(cells["cell_id"].astype(str))
    assert groups["draw1"] and groups["draw2"]
    assert not groups["draw1"] & groups["draw2"]
    draw2_cells = {
        str(cell) for cell in res.load_test_cells(family.draw2_dir).obs.index
    }
    assert groups["draw2"] <= draw2_cells
    # PREP re-ran only for NP9's identity re-run, in a scratch store.
    rerun = family.prep_calls[prep_calls:]
    assert [call["tag"] for call in rerun] == ["R1_contam_HO", "clean"]
    assert record["prep_identity"]["identical"] is True
    assert record["replicate_identity"] == {"R1_contam_HO@0": True}
    # Draw 2's availability replaces the human pool sizes.
    availability = pd.read_csv(out / run.DRAW2_AVAILABILITY_CSV)
    assert set(availability.columns) == set(run.DRAW2_AVAILABILITY_COLUMNS)
    assert set(availability["class"]) == set(REAL_CLASSES)
    assert not (out / run.POOL_SIZES_CSV).exists()
    assert record["draw2_build_seconds"] >= 0
    assert record["reference_composition"]["n_cells"] == 9 * 80
    # The report: NP7 without the sink part, the per-donor spread reported.
    report = json.loads((out / gp.GATE_P_REPORT_JSON).read_text())
    assert report["species"] == "mouse"
    assert not list(out.glob("np7_excluded__*.csv"))
    spread = pd.read_csv(out / "np4_donor_spread__r1_contam_ho_seed0.csv")
    stats = pd.read_csv(out / "np4_donor_stats__r1_contam_ho_seed0.csv")
    assert set(stats["donor"]) == {"D1", "D2", "D3"}
    assert set(spread.columns) == set(gp.NP4_DONOR_SPREAD_COLUMNS)
    tables = report["tables"]
    assert tables["np4_donor_spread__r1_contam_ho_seed0"]["reported_only"] is True
    assert "n_failed" not in tables["np4_donor_spread__r1_contam_ho_seed0"]
    records = pd.read_csv(out / gp.GATE_P_RECORDS_CSV)
    assert set(records["level"]) <= set(gp.gate_p_levels("mouse"))
    assert result["validated_max_level"] in (None, *gp.gate_p_levels("mouse"))
    # The dry run has no exemption; it expects class for every C_P class.
    dry = record["dry_run"]
    assert dry["exempted"] == [] and dry["min_level"] == "class"
    assert {row["class"] for row in dry["expected"] if row["level"] == "class"} == {
        "01 IT-ET Glut",
        "30 Astro-Epen",
    }


def test_mouse_gate_p_refuses_a_draw_2_that_is_not_preps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_resources: Any
) -> None:
    # Draw 2 built under another draw rule: its build_hash is not the one
    # PREP's test-set spec gives.
    family = MouseFamily(
        tmp_path,
        monkeypatch,
        mouse_gate_config(),
        draw2_config=mouse_gate_config(gate_p_class_min_test_cells=45),
    )
    with pytest.raises(run.GatePRunError, match="other inputs than this PREP"):
        run.run_gate_p(family.request(), family.options())
    assert family.gate_calls == []

    def writable(target: str) -> Path:
        copy = tmp_path / target
        shutil.copytree(family.draw2_dir, copy)
        for path in [copy, *copy.rglob("*")]:
            path.chmod(0o755 if path.is_dir() else 0o644)
        return copy

    # A file that no longer matches bundle.json.
    edited = writable("edited")
    (edited / res.TEST_CELLS_FILE).write_bytes(b"x")
    with pytest.raises(run.GatePRunError, match="files differ"):
        run.run_gate_p(family.request("o2"), family.options(wmb_draw2_bundle=edited))
    # Another panel's draw 2.
    other = writable("other_panel")
    manifest = json.loads((other / BUNDLE_MANIFEST_NAME).read_text())
    manifest["panel"]["panel_hash"] = "0" * 64
    (other / BUNDLE_MANIFEST_NAME).write_text(json.dumps(manifest))
    with pytest.raises(run.GatePRunError, match="not of this family's panel"):
        run.run_gate_p(family.request("o3"), family.options(wmb_draw2_bundle=other))
    # Not a draw-2 bundle at all: PREP's own draw 1.
    with pytest.raises(GatePUnavailableError, match="second WMB test draw"):
        family.options(wmb_draw2_bundle=family.draw1_dir).check_supported()
    assert family.gate_calls == []


def test_mouse_gate_p_checks_the_draw_1_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_resources: Any
) -> None:
    family = MouseFamily(tmp_path, monkeypatch, mouse_gate_config())
    copy = tmp_path / "forged"
    shutil.copytree(family.draw2_dir, copy)
    for path in [copy, *copy.rglob("*")]:
        path.chmod(0o755 if path.is_dir() else 0o644)
    manifest = json.loads((copy / BUNDLE_MANIFEST_NAME).read_text())
    test_set = manifest["builder_output"]["test_set"]
    test_set["gate_p_draw"]["first_draw"]["cell_digest"] = "f" * 64
    (copy / BUNDLE_MANIFEST_NAME).write_text(json.dumps(manifest))
    with pytest.raises(run.GatePRunError, match="another draw 1"):
        run.run_gate_p(family.request(), family.options(wmb_draw2_bundle=copy))
    assert family.gate_calls == []


def test_mouse_options_and_precheck_refuse_before_any_compute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    family = MouseFamily(tmp_path, monkeypatch, mouse_gate_config(), build=False)
    # The path is lifted only when a draw-2 bundle exists.
    with pytest.raises(GatePUnavailableError, match="C20"):
        run.gate_p_hook(run.GatePOptions(species="mouse"))
    with pytest.raises(GatePUnavailableError, match="not a complete bundle"):
        run.GatePOptions(species="mouse", wmb_draw2_bundle=tmp_path).check_supported()
    # No mouse exemption, no human D2 option, no forced version 7, and no
    # draw 2 for a human family.
    assert run.GatePOptions(species="mouse").dry_run_exemptions == ()
    assert run.GatePOptions(species="human").dry_run_exemptions == (
        run.HUMAN_DRY_RUN_EXEMPTIONS
    )
    with pytest.raises(run.GatePRunError, match="no dry-run exemption"):
        run.GatePOptions(species="mouse", dry_run_exemptions=(("subclass", "x"),))
    with pytest.raises(run.GatePRunError, match="human leave-one-donor-out"):
        run.GatePOptions(species="mouse", other_region="shared")
    with pytest.raises(run.GatePRunError, match="human leave-one-donor-out"):
        run.GatePOptions(species="mouse", accept_small_pools=True)
    with pytest.raises(run.GatePRunError, match="v7.6 top-up"):
        run.GatePOptions(species="mouse", resolvability_version=7)
    with pytest.raises(run.GatePRunError, match="mouse path"):
        run.GatePOptions(species="human", wmb_draw2_bundle=tmp_path)
    # The human record keeps its keys (no mouse-only option).
    assert "wmb_draw2_bundle" not in run.GatePOptions(species="human").to_json()

    # The precheck: the draws and the production draw 1.
    store = ReferenceStore(tmp_path / "gate_p_store")
    draw2 = tmp_path / "draw2"
    draw2.mkdir()
    (draw2 / BUNDLE_MANIFEST_NAME).write_text(
        json.dumps(
            {
                "status": "complete",
                "reference_id": reference.WMB_TESTSET_REFERENCE_ID,
                "builder_output": {"test_set": {"draw": 2, "gate_p_draw": {}}},
            }
        )
    )
    options = run.GatePOptions(species="mouse", wmb_draw2_bundle=draw2)

    def plan(config: AnnotationConfig) -> GatePPlan:
        return GatePPlan(
            species="mouse",
            config=config,
            store=store,
            out_dir=tmp_path / "plan_out",
            scratch_dir=tmp_path / "plan_scratch",
            expected_depth=60,
        )

    run.precheck_gate_p(plan(mouse_gate_config()), options)
    with pytest.raises(run.GatePRunError, match="gate_p_mouse_test_draws"):
        run.precheck_gate_p(plan(mouse_gate_config(gate_p_mouse_test_draws=3)), options)
    with pytest.raises(run.GatePRunError, match="production self-map on draw 1"):
        run.precheck_gate_p(plan(with_draw(mouse_gate_config(), 2)), options)
    with pytest.raises(run.GatePRunError, match="gate_p_seeds"):
        run.precheck_gate_p(plan(mouse_gate_config(gate_p_seeds=[0])), options)
    # The human donors are not read for mouse (they name WHB donors).
    run.precheck_gate_p(plan(mouse_gate_config(gate_p_human_donors=["x"])), options)
    assert family.gate_calls == []


def test_the_draw_2_availability_table_lists_every_class() -> None:
    record = {
        "per_class": {
            "B": {"n_draw2": 3, "n_pooled": 9, "reaches_target": False},
            "A": {"n_draw2": 5, "n_pooled": 700, "reaches_target": True},
        }
    }
    table = run.draw2_availability(record)
    assert list(table["class"]) == ["A", "B"]
    assert list(table.columns) == list(run.DRAW2_AVAILABILITY_COLUMNS)
    assert list(table["reaches_target"]) == [True, False]


def test_options_round_trip_keeps_the_mouse_draw(tmp_path: Path) -> None:
    draw2 = tmp_path / "draw2"
    options = run.GatePOptions(species="mouse", wmb_draw2_bundle=draw2)
    copy = dataclasses.replace(options, dry_run=True)
    assert copy.wmb_draw2_bundle == draw2 and copy.dry_run_exemptions == ()
    assert copy.to_json()["wmb_draw2_bundle"] == str(draw2)


def test_the_command_lifts_the_mouse_path_only_with_a_draw_2_bundle(
    tmp_path: Path,
) -> None:
    from click.testing import CliRunner

    from merxen.cli import main as cli_main

    gene_list = tmp_path / "genes.csv"
    gene_list.write_text("gene_symbol,gene_id\nA,ENSMUSG00000000001\n")
    not_a_bundle = tmp_path / "not_a_bundle"
    not_a_bundle.mkdir()

    def invoke(*extra: str) -> Any:
        return CliRunner().invoke(
            cli_main,
            [
                "annotation-panel-simulate",
                "--gene-list",
                str(gene_list),
                "--species",
                "mouse",
                "--store",
                str(tmp_path / "gate_p_store"),
                "--scratch-dir",
                str(tmp_path / "scratch"),
                "--out-dir",
                str(tmp_path / "out"),
                "--expected-depth",
                "60",
                "--gate-p",
                *extra,
            ],
        )

    result = invoke()
    assert result.exit_code != 0 and "C20" in result.output
    result = invoke("--gate-p-wmb-draw2", str(not_a_bundle))
    assert result.exit_code != 0 and "not a complete bundle" in result.output
    result = invoke(
        "--gate-p-wmb-draw2", str(not_a_bundle), "--gate-p-other-region", "shared"
    )
    assert result.exit_code != 0 and "human leave-one-donor-out" in result.output
    # Refused before any compute: nothing was built or written.
    assert not (tmp_path / "gate_p_store").exists()
    assert not (tmp_path / "out").exists()

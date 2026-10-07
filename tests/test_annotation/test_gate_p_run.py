"""Tests for the gate-P driver (M13 chunk C8; plan §14, pre-registration §23).

End to end on the synthetic WHB fixtures of ``test_reference`` (three frontal
donors, a fake ``cell_type_mapper`` and a marker-count mapper standing in for
MapMyCells): PREP builds the family's bundle, gate P builds each other
donor's held-out bundle in the request's store, simulates and maps the
replicates, scores NP1-NP9 and writes the report. No MapMyCells, no real
reference.
"""

from __future__ import annotations

import json
import zlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from merxen.annotation import gate_p as gp
from merxen.annotation import gate_p_run as run
from merxen.annotation import reference
from merxen.annotation import resolvability as res
from merxen.annotation.config import AnnotationConfig
from merxen.annotation.reference import builder_for, prepare_reference_spec
from merxen.annotation.simulate import (
    GatePRequest,
    GatePUnavailableError,
    ReferenceBuild,
)
from merxen.annotation.store import ReferenceStore

from . import test_reference as tr
from .test_reference import (
    GENES,
    make_panel,
    whb_resolvability_setup,
    whb_spec,
)

DONORS = ["H_small", "H_big", "H_mid"]
DEFAULT_DONOR = "H_small"


def fixture_truths(metadata_dir: Path) -> dict[str, str]:
    """Each fixture cell's truth supercluster (``write_ho_sources``)."""
    cells = pd.read_csv(metadata_dir / "cell_metadata.csv")
    return {
        str(label): tr.CLUS_TO_SUPC[tr.SUBC_TO_CLUS[f"s{int(alias)}"]]
        for label, alias in zip(
            cells["cell_label"], cells["cluster_alias"], strict=True
        )
    }


def oracle_mapper(truths: Mapping[str, str], *, error_every: int = 50) -> Any:
    """A mapper that calls each simulated cell's truth, with planted errors.

    Below 15 counts every call is unconfident (bp 0.6 < the WHB raw 0.73);
    from 15 counts it has bp 0.95 and is right, except one simulated cell in
    ``error_every`` (keyed by cell and depth), called another supercluster.
    MapMyCells' seed changes nothing, so seeds 0 and 1 agree.
    """
    from merxen.annotation.mapmycells_engine import N_RUNNERS_UP, runner_up_column

    def map_onto(engine: Any, query: Any) -> pd.DataFrame:
        tree = engine.tree()
        first = tree.hierarchy[0]
        roots = sorted(tree.nodes(first))
        records = []
        obs = query.obs
        for sim_id, cell_id, depth in zip(
            obs.index.astype(str),
            obs["cell_id"].astype(str),
            obs["depth"].astype(int),
            strict=True,
        ):
            truth = truths.get(cell_id, roots[0])
            node = truth if truth in roots else roots[0]
            key = zlib.crc32(f"{cell_id}:{depth}".encode())
            if key % error_every == 0:
                others = [item for item in roots if item != node]
                node = others[(key // error_every) % len(others)]
            bp = 0.6 if depth < 15 else 0.95
            parent = node
            for position, level in enumerate(tree.hierarchy):
                if position > 0:
                    above = tree.hierarchy[position - 1]
                    children = sorted(tree.children_of(above, parent))
                    parent = children[0]
                assigned = parent
                others = (
                    [item for item in roots if item != node] if position == 0 else []
                )
                record: dict[str, Any] = {
                    "cell_id": sim_id,
                    "level": level,
                    "level_name": level.lower(),
                    "assignment": assigned,
                    "name": tree.name(level, assigned),
                    "bp": bp,
                    "aggregate_probability": bp,
                    "avg_correlation": 0.5,
                    "directly_assigned": True,
                    "n_runners_up": len(others),
                }
                for rank in range(1, N_RUNNERS_UP + 1):
                    other = others[rank - 1] if rank <= len(others) else None
                    record[runner_up_column(rank, "assignment")] = other
                    record[runner_up_column(rank, "name")] = (
                        None if other is None else tree.name(level, other)
                    )
                    record[runner_up_column(rank, "probability")] = (
                        np.nan if other is None else round((1 - bp) / len(others), 3)
                    )
                    record[runner_up_column(rank, "correlation")] = (
                        np.nan if other is None else 0.1
                    )
                records.append(record)
        return pd.DataFrame.from_records(records)

    return map_onto


@pytest.fixture
def small_resources() -> Any:
    """PREP resources of 2 processes and 3 GB (as test_reference's fixture)."""
    previous = reference._PREP_RESOURCES
    reference.set_prep_resources(n_processors=2, max_gb=3)
    yield
    reference._PREP_RESOURCES = previous


def gate_config(**resolvability: Any) -> AnnotationConfig:
    """A config whose gate-P minimums fit the fixture's few test cells."""
    settings: dict[str, Any] = {
        "version": 6,
        "min_cells_per_bin": 5,
        "min_confident_n": 5,
        "gate_p_min_confident_n": 5,
        "gate_p_replicate_min_confident_n": 3,
        "gate_p_class_min_test_cells": 5,
        "gate_p_human_donors": list(DONORS),
        "topup_min_class_test_cells": 2,
    }
    settings.update(resolvability)
    # Targets the fixture's few calls can reach (Wilson bounds on ~10 calls):
    # gate P's rules are unchanged, only the constants are small.
    targets = {
        f"target_{level}": 0.5
        for level in ("lineage", "broad", "nt", "supercluster", "class", "subclass")
    }
    return AnnotationConfig(species="human", resolvability=settings, thresholds=targets)


class Family:
    """A family's PREP on the synthetic WHB, ready for gate P."""

    def __init__(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        config: AnnotationConfig,
        *,
        error_every: int = 50,
        build: bool = True,
    ) -> None:
        # Four times the fixture's donors, so each class has tested sets.
        monkeypatch.setattr(
            tr, "HO_DONORS", {donor: 4 * n for donor, n in tr.HO_DONORS.items()}
        )
        sources, ho, self.fake = whb_resolvability_setup(tmp_path, monkeypatch)
        mapper = oracle_mapper(fixture_truths(ho["metadata"]), error_every=error_every)
        metadata = pd.read_csv(ho["metadata"] / "cell_metadata.csv")
        self.donor_of = dict(
            zip(
                metadata["cell_label"].astype(str),
                metadata["donor_label"].astype(str),
                strict=True,
            )
        )
        self.prep_calls: list[dict[str, Any]] = []

        def prep_map_function(context: Any, engine: Any, runs: list[Any]) -> Any:
            def map_query(query: Any, tag: str, seed: int) -> pd.DataFrame:
                self.prep_calls.append(
                    {"engine_path": str(engine.path), "tag": tag, "seed": seed}
                )
                runs.append(
                    {
                        "tag": tag,
                        "engine": engine.reference_id,
                        "n_processors": reference.prep_resources().n_processors,
                    }
                )
                return mapper(engine, query)

            return map_query

        # PREP's self-map maps through reference._mmc_map_function.
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

        # Gate P maps through reference.mmc_map_function (MapMyCells).
        monkeypatch.setattr(reference, "mmc_map_function", map_function)
        self.spec = prepare_reference_spec(
            whb_spec(
                region_precompute=sources["region_dir"],
                seaad_precomputed_stats=sources["seaad"],
                whb_h5ad_dir=ho["h5ad_dir"],
                whb_metadata_dir=ho["metadata"],
                whb_region_cell_metadata=ho["region_dir"]
                / reference.REGION_CELL_METADATA_FILE,
            )
        )
        self.config = config
        self.panel = make_panel(GENES)
        (tmp_path / "scratch").mkdir(exist_ok=True)
        self.store = ReferenceStore(
            tmp_path / "gate_p_store", scratch_root=tmp_path / "scratch"
        )
        self.builder = builder_for(self.spec, config)
        self.tmp_path = tmp_path
        if build:
            bundle = self.store.get_or_build(
                self.spec, self.panel, builder=self.builder, config=config
            )
            self.bundle_dir = Path(bundle.path)

    def request(
        self, out: str = "out", *, report: Mapping[str, Any] | None = None
    ) -> GatePRequest:
        scratch = self.tmp_path / f"{out}_scratch"
        scratch.mkdir(exist_ok=True)
        return GatePRequest(
            panel=self.panel,
            config=self.config,
            store=self.store,
            bundles={run.PRIMARY_REFERENCE: self.bundle_dir},
            out_dir=self.tmp_path / out,
            report=dict(
                report
                or {
                    "platform": "MERSCOPE",
                    "settings": {"expected_depth": 30},
                    "panel": {"gene_ids": {}},
                    "references": {},
                }
            ),
            builds=(ReferenceBuild(spec=self.spec, builder=self.builder),),
            scratch_dir=scratch,
        )


def read_run(result: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(Path(result["run"]).read_text())


def test_gate_p_runs_end_to_end_on_disjoint_leave_one_donor_out_sets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_resources: Any
) -> None:
    family = Family(tmp_path, monkeypatch, gate_config())
    prep_calls = len(family.prep_calls)
    result = run.run_gate_p(
        family.request(),
        run.GatePOptions(species="human", accept_small_pools=True, dry_run=True),
    )
    assert result["status"] == run.STATUS_SCORED, result
    record = read_run(result)
    out = tmp_path / "out" / run.GATE_P_RUN_DIR
    # The default donor is the frozen run's; the others got their own bundles.
    assert record["default_donor"] == DEFAULT_DONOR
    assert record["extra_donors"] == ["H_big", "H_mid"]
    loo = {item["donor"]: item for item in record["leave_one_donor_out"]}
    default_hash = record["default_test_set"]["build_hash"]
    hashes = {item["build_hash"] for item in loo.values()}
    assert len(hashes) == 2 and default_hash not in hashes
    for donor, item in loo.items():
        bundle_dir = Path(item["path"])
        assert bundle_dir.is_relative_to(tmp_path / "gate_p_store")
        manifest = json.loads((bundle_dir / "bundle.json").read_text())
        test_set = manifest["builder_output"]["test_set"]
        assert test_set["holdout_donor"] == donor
        # D2 (d): the donor's own other-region cells only, none of the
        # default held-out test set; the build hash records the rule.
        rule = test_set["gate_p_leave_one_donor_out"]
        assert rule["holdout_donor"] == donor and rule["disjoint_from_excluded"]
        payload = manifest["build_hash_payload"]["builder_params"]
        assert payload["other_region"]["donor_only"] is True
        assert (
            reference.SOURCE_GATE_P_EXCLUDED_TEST_CELLS
            in (manifest["build_hash_payload"]["sources"])
        )
        # M8 D1 (extended to every human gate-P set by M13 D1 (a)): every
        # leave-one-donor-out set carries the self-map exclusion.
        exclusion = item["self_map_exclusion"]
        assert exclusion["revision"] == reference.HO_SELF_MAP_TEST_SET_REVISION
        assert exclusion["excluded_superclusters"] == list(
            reference.HO_SELF_MAP_EXCLUDED_SUPERCLUSTERS
        )
    assert record["default_test_set"]["self_map_exclusion"]["revision"] == 1
    # Each replicate was mapped against its own donor's held-out bundle; PREP
    # did not run again (the default donor's seed-0 rows are PREP's own).
    engines = {Path(call["engine"]).parent.name for call in family.gate_calls}
    assert engines == {reference.HO_REFERENCE_ID}
    by_engine: dict[str, set[str]] = {}
    for call in family.gate_calls:
        by_engine.setdefault(Path(call["engine"]).name, set()).update(call["cells"])
    donor_of = family.donor_of
    for donor, item in loo.items():
        cells = by_engine[item["build_hash"]]
        # D2 (d): a true donor hold-out, its other-region top-up included.
        assert cells and {donor_of[cell] for cell in cells} == {donor}
    # PREP mapped again only for the identity re-run, in a scratch store.
    rerun = family.prep_calls[prep_calls:]
    assert [call["tag"] for call in rerun] == ["R1_contam_HO", "clean"]
    assert all(
        Path(call["engine_path"]).is_relative_to(tmp_path / "out_scratch")
        for call in rerun
    )
    replicates = pd.DataFrame(record["replicates"])
    prep_rows = replicates[replicates["source"] == "prep_bundle"]
    assert set(prep_rows["group"]) == {DEFAULT_DONOR}
    assert set(prep_rows["seed"]) == {0}
    # Seeds 0 and 1 for the base member in every group; the stresses and the
    # clean bound at seed 0.
    base = replicates[replicates["member"] == "R1_contam_HO@0"]
    assert sorted(zip(base["group"], base["seed"], strict=True)) == sorted(
        (donor, seed) for donor in DONORS for seed in (0, 1)
    )
    stresses = replicates[replicates["role"] == "stress"]
    assert set(stresses["member"]) == {
        "R1_stress_spill@0",
        "R1_stress_lognormal@0",
        "R1_stress_xplatform@0",
    }
    assert set(stresses["seed"]) == {0}
    assert sorted(stresses["group"]) == sorted(DONORS * 3)
    for path in replicates["file"]:
        assert (out / path).is_file()
    # Disjoint replicates: no test cell is in two groups.
    groups: dict[str, set[str]] = {}
    for _, row in base[base["seed"] == 0].iterrows():
        cells = pd.read_parquet(out / row["file"], columns=["cell_id"])
        groups[row["group"]] = set(cells["cell_id"].astype(str))
    assert not groups["H_big"] & groups["H_mid"]
    assert not (groups["H_big"] | groups["H_mid"]) & groups[DEFAULT_DONOR]
    # The pool sizes were reported before the builds.
    pools = pd.read_csv(out / run.POOL_SIZES_CSV)
    assert set(pools["donor"]) == {"H_big", "H_mid"}
    assert set(pools.columns) == set(reference.GATE_P_POOL_COLUMNS)
    # NP9 identity: PREP rebuilt identically, one seed-0 replicate per member.
    assert record["replicate_identity"] == {"R1_contam_HO@0": True}
    assert record["prep_identity"]["identical"] is True
    assert record["prep_identity"]["differences"] == []
    # The NP1-NP9 report and its criterion tables.
    report = json.loads((out / gp.GATE_P_REPORT_JSON).read_text())
    assert report["members"] == ["R1_contam_HO@0"]
    assert report["resolvability_version"] == 6
    assert set(report["family_checks"]) == set(gp.GATE_P_FAMILY_CHECKS)
    assert report["family_checks"]["NP9"]["parts"]["identity"] == gp.CHECK_PASSED
    assert report["family_checks"]["NP8"]["status"] == gp.CHECK_NOT_APPLICABLE
    for name in (
        "np3_verdicts",
        "np4_sets",
        "np4_seed",
        "np5_agreement",
        "np6_verdicts",
        "np7_wrong_node",
        "np7_excluded",
    ):
        assert (out / f"{name}__r1_contam_ho_seed0.csv").is_file(), name
    assert (out / gp.GATE_P_REPORT_TXT).is_file()
    records = pd.read_csv(out / gp.GATE_P_RECORDS_CSV)
    assert set(records["level"]) <= set(gp.gate_p_levels("human"))
    # The planted errors all land on other classes, so the class-balanced
    # weighting fails broad Immune at NP3 (its wrong calls' types weigh as
    # much as its own); the other classes of the level stay validated.
    status = records.set_index(["level", "class"])["status"]
    assert status[("broad", "Immune")] == "failed:NP3"
    assert status[("broad", "Astro")] == gp.RECORD_VALIDATED
    assert status[("broad", "Exc")] == gp.RECORD_VALIDATED
    assert result["validated_max_level"] is None and not result["passes"]
    tested = pd.read_csv(out / "tested_sets__r1_contam_ho_seed0.csv")
    assert (tested["n_confident"].dropna() >= 5).all()
    # The dry run (D28) fails on broad Immune. At supercluster it expects only
    # the classes with >= 50 of the default donor's test cells (H18's set).
    dry = record["dry_run"]
    assert not dry["passes"]
    assert [(row["level"], row["class"]) for row in dry["failing"]] == [
        ("broad", "Immune")
    ]
    leaf = {row["class"] for row in dry["expected"] if row["level"] == "supercluster"}
    assert leaf == {"Exc"}
    assert result["simulated_cells"] == record["simulated_cells"] > 0
    # The simulation report holds the run's record (the hook's return value).
    assert result["report"] == str(out / gp.GATE_P_REPORT_TXT)


def passing_report() -> dict[str, Any]:
    """A base-simulation report whose declared panel passes NP1."""
    return {
        "platform": "MERSCOPE",
        "settings": {"expected_depth": 30},
        "panel": {
            "gene_ids": {
                "status": "ok",
                "n_features_in": len(GENES),
                "n_non_control": len(GENES),
                "n_genes": len(GENES),
                "gene_id_resolution": {"native": len(GENES)},
                "gene_id_resolution_share": 1.0,
                "unresolved": {},
                "species_check": {"status": "pass"},
            }
        },
        "references": {},
    }


def test_a_family_whose_classes_all_validate_passes_gate_p(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_resources: Any
) -> None:
    config = gate_config().model_copy(
        update={"panel": gate_config().panel.model_copy(update={"min_root_markers": 3})}
    )
    family = Family(tmp_path, monkeypatch, config, error_every=10**9)
    # The fake ctm steps record no memory; NP9's RSS part needs one.
    monkeypatch.setattr(
        run, "_prep_measurements", lambda *_: (60.0, {"primary:markers": 1.0})
    )
    # The user accepts the weak and collapsed parents in the gate-P PR (NP2).
    inputs = run.np2_inputs(family.bundle_dir)
    accepted = {*inputs["weak_parents"], *inputs["collapsed_parents"]}
    assert accepted
    result = run.run_gate_p(
        family.request(report=passing_report()),
        run.GatePOptions(
            species="human",
            accept_small_pools=True,
            dry_run=True,
            time_reference_seconds=1e6,
            time_reference_basis="test",
            accepted_parents=tuple(sorted(accepted)),
        ),
    )
    assert result["passes"], result["reasons"]
    assert result["validated_max_level"] == "supercluster"
    assert result["dry_run"]["passes"]
    record = read_run(result)
    assert record["family_id"] == result["family_id"]
    report = json.loads(
        (tmp_path / "out" / "gate_p" / gp.GATE_P_REPORT_JSON).read_text()
    )
    for name in ("NP1", "NP2", "NP9"):
        assert report["family_checks"][name]["status"] == gp.CHECK_PASSED, name
    # NP9: PREP's time plus this run's builds and replicates, within 1.5x.
    np9 = report["family_checks"]["NP9"]["detail"]
    assert np9["wall_seconds"] == pytest.approx(record["measured_seconds"])
    assert record["measured_seconds"] >= 60.0
    assert np9["members"] == ["R1_contam_HO@0"]


def test_gate_p_stops_before_any_build_when_a_donor_pool_is_too_small(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_resources: Any
) -> None:
    # Pre-registration §23.10 open item 2: the pool sizes come first; a donor
    # whose own pool cannot meet the per-class top-up rule stops gate P
    # before any leave-one-donor-out set is built or any replicate mapped.
    family = Family(tmp_path, monkeypatch, gate_config(topup_min_class_test_cells=1000))
    held_out = sorted((tmp_path / "gate_p_store" / reference.HO_REFERENCE_ID).iterdir())
    result = run.run_gate_p(family.request(), run.GatePOptions(species="human"))
    assert result["status"] == run.STATUS_STOPPED
    assert "per-class top-up rule" in result["reason"]
    record = read_run(result)
    assert record["status"] == run.STATUS_STOPPED
    short = pd.DataFrame(record["pool_sizes"]["short"])
    assert set(short["donor"]) == {"H_big", "H_mid"}
    assert (short["n_available"] < 1000).all() and short["judged"].all()
    pools = pd.read_csv(Path(result["pool_sizes"]))
    assert len(pools) == len(short)
    # Nothing was built or mapped.
    assert sorted(
        (tmp_path / "gate_p_store" / reference.HO_REFERENCE_ID).iterdir()
    ) == (held_out)
    assert family.gate_calls == []
    assert not (tmp_path / "out" / run.GATE_P_RUN_DIR / run.REPLICATES_DIR).exists()
    # The user's choice is run in the same place: a stopped run blocks nothing.
    facts = run._check_request(
        family.request(), run.GatePOptions(species="human", accept_small_pools=True)
    )
    assert facts["out"] == tmp_path / "out" / run.GATE_P_RUN_DIR


def test_the_d2_c_fallback_shares_the_pool_and_every_set_keeps_the_d1_drop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_resources: Any
) -> None:
    # M8 D1 with Astrocyte in place of COP (the fixture's other-region cells
    # are Astrocytes and Microglia), so the drop has cells to leave out.
    monkeypatch.setattr(reference, "HO_SELF_MAP_EXCLUDED_SUPERCLUSTERS", ("Astrocyte",))
    family = Family(tmp_path, monkeypatch, gate_config(topup_min_class_test_cells=1000))
    result = run.run_gate_p(
        family.request(),
        run.GatePOptions(
            species="human", other_region=run.OTHER_REGION_SHARED, prep_identity=False
        ),
    )
    # (c) never stops on the pool sizes: the user chose it with them in view.
    assert result["status"] == run.STATUS_SCORED, result
    record = read_run(result)
    out = tmp_path / "out" / run.GATE_P_RUN_DIR
    other_region = {
        cell
        for cell, donor in family.donor_of.items()
        if not cell.startswith(tuple(f"{name}-" for name in DONORS))
        or donor not in DONORS
    }
    astro_other = {cell for cell in other_region if cell.startswith("mtg_astro")}
    for item in record["leave_one_donor_out"]:
        assert item["other_region"] == run.OTHER_REGION_SHARED
        manifest = json.loads((Path(item["path"]) / "bundle.json").read_text())
        payload = manifest["build_hash_payload"]
        assert "donor_only" not in payload["builder_params"]["other_region"]
        assert reference.SOURCE_GATE_P_EXCLUDED_TEST_CELLS not in payload["sources"]
        test = res.load_test_cells(Path(item["path"]))
        held = set(test.obs.index.astype(str))
        # The production draw (every donor's other-region cells) ...
        assert held & astro_other
        # ... and every set leaves the D1 cells out of what it simulates.
        exclusion = item["self_map_exclusion"]
        assert exclusion["n_excluded"] == len(held & astro_other) > 0
    replicates = pd.DataFrame(record["replicates"])
    simulated = set()
    for path in replicates["file"]:
        simulated |= set(pd.read_parquet(out / path, columns=["cell_id"])["cell_id"])
    assert not simulated & astro_other
    # C_P counts each shared cell once (D2 (c), disjoint=False).
    report = json.loads((out / gp.GATE_P_REPORT_JSON).read_text())
    assert report["class_sets"]["levels"]
    assert record["prep_identity"]["identical"] is None
    assert report["family_checks"]["NP9"]["parts"]["identity"] == (
        gp.CHECK_NOT_EVALUABLE
    )


def test_gate_p_refuses_requests_before_any_compute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_resources: Any
) -> None:
    family = Family(tmp_path, monkeypatch, gate_config())
    store_before = sorted(path.name for path in (tmp_path / "gate_p_store").rglob("*"))
    options = run.GatePOptions(species="human", accept_small_pools=True)

    def refused(
        request: GatePRequest,
        match: str,
        *,
        error: type[Exception] = run.GatePRunError,
        **changes: Any,
    ) -> None:
        with pytest.raises(error, match=match):
            run.run_gate_p(request, run.GatePOptions(**{**options.__dict__, **changes}))

    def with_config(**resolvability: Any) -> GatePRequest:
        request = family.request()
        config = gate_config(**resolvability)
        return GatePRequest(**{**request.__dict__, "config": config})

    # M13 D4: the species is chosen when the check starts; mouse waits for C20.
    refused(
        family.request(),
        "second disjoint WMB",
        error=GatePUnavailableError,
        species="mouse",
    )
    with pytest.raises(GatePUnavailableError, match="C20"):
        run.gate_p_hook(run.GatePOptions(species="mouse"))
    request = family.request()
    mouse = GatePRequest(
        **{**request.__dict__, "config": AnnotationConfig(species="mouse")}
    )
    refused(mouse, "started for human")
    # The donors: each one once (never a duplicated replicate), each a frontal
    # donor of the held-out bundle, the default among them, and another one.
    refused(with_config(gate_p_human_donors=["H_small", "H_big", "H_big"]), "twice")
    refused(with_config(gate_p_human_donors=["H_small", "H_big", "H_none"]), "H_none")
    refused(
        with_config(gate_p_human_donors=["H_big", "H_mid"]), "default held-out donor"
    )
    refused(with_config(gate_p_human_donors=["H_small"]), "at least one other donor")
    # Seeds 0 (the frozen run) and another mapping seed (D6).
    refused(with_config(gate_p_seeds=[1, 2]), "gate_p_seeds")
    # The leave-one-donor-out sets differ from PREP's only by the donor: a
    # config that changes the held-out build itself is refused.
    refused(with_config(n_test_cells=40), "differ from PREP's")
    # NP5 needs the family's expected depth; a per-class CSV is never read.
    no_depth = family.request(report={"settings": {}, "panel": {}, "references": {}})
    refused(no_depth, "expected depth")
    # The primary bundle, its self-map, the worker count, the output place.
    request = family.request()
    refused(GatePRequest(**{**request.__dict__, "bundles": {}}), "primary reference")
    inside = GatePRequest(
        **{**request.__dict__, "out_dir": tmp_path / "gate_p_store" / "x"}
    )
    refused(inside, "inside a reference store")
    reference.set_prep_resources(n_processors=3, max_gb=3)
    refused(family.request(), "MapMyCells workers")
    reference.set_prep_resources(n_processors=2, max_gb=3)
    done = tmp_path / "done" / run.GATE_P_RUN_DIR
    done.mkdir(parents=True)
    (done / run.GATE_P_RUN_JSON).write_text("{}")
    refused(family.request("done"), "already holds a gate-P run")
    (done / run.GATE_P_RUN_JSON).unlink()
    (done / run.REPLICATES_DIR / "H_small").mkdir(parents=True)
    (done / run.REPLICATES_DIR / "H_small" / "x.parquet").write_bytes(b"")
    refused(family.request("done"), "replicates of an earlier gate-P run")
    # Nothing was built, simulated or mapped.
    assert family.gate_calls == []
    assert sorted(path.name for path in (tmp_path / "gate_p_store").rglob("*")) == (
        store_before
    )
    # A replicate is written once (a duplicated replicate is refused).
    runner = run._Runner(
        family.request("runner"),
        out=tmp_path / "runner_out",
        species="human",
        version7=False,
        grid=[10],
    )
    member = res.EnsembleMember(
        res.member_recipe(res.DECISION_RECIPE, 0, family.config.resolvability),
        "emission",
    )
    rows = res.load_resolvability(family.bundle_dir).cells
    runner.write("H_small", member, 0, rows)
    with pytest.raises(run.GatePRunError, match="duplicated replicate"):
        runner.write("H_small", member, 0, rows)
    # The options themselves.
    with pytest.raises(run.GatePRunError, match="other_region"):
        run.GatePOptions(species="human", other_region="halves")
    with pytest.raises(run.GatePRunError, match="both"):
        run.GatePOptions(species="human", dry_run_seconds=100.0)
    with pytest.raises(run.GatePRunError, match="species"):
        run.GatePOptions(species="rat")  # type: ignore[arg-type]


def test_a_version_7_family_is_scored_in_every_emission_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_resources: Any
) -> None:
    # Two emission members keep the test small; the assembly is told so.
    monkeypatch.setattr(res, "V7_EMISSION_MEMBERS", 2)
    family = Family(
        tmp_path,
        monkeypatch,
        gate_config(version="auto", ensemble_r1_seeds=[0, 6]),
    )
    summary = json.loads(
        (family.bundle_dir / res.RESOLVABILITY_SUMMARY_FILE).read_text()
    )
    assert summary["resolvability_version"] == 7
    result = run.run_gate_p(
        family.request(), run.GatePOptions(species="human", accept_small_pools=True)
    )
    assert result["status"] == run.STATUS_SCORED, result
    record = read_run(result)
    members = ["R1_contam_HO@0", "R1_contam_HO@6"]
    assert record["members"] == members and record["resolvability_version"] == 7
    assert not record["forced_version_7"]
    out = tmp_path / "out" / run.GATE_P_RUN_DIR
    report = json.loads((out / gp.GATE_P_REPORT_JSON).read_text())
    assert report["members"] == members
    assert report["family_checks"]["NP9"]["detail"]["members"] == members
    # Each member's own replicates over donors and seeds, and its own three
    # stress members; the default donor's seed-0 rows are PREP's own.
    replicates = pd.DataFrame(record["replicates"])
    for member in members:
        rows = replicates[replicates["member"] == member]
        assert len(rows) == len(DONORS) * 2
        prep = rows[rows["source"] == "prep_bundle"]
        assert list(zip(prep["group"], prep["seed"], strict=True)) == [
            (DEFAULT_DONOR, 0)
        ]
        seed = member.rsplit("@", 1)[1]
        stresses = replicates[replicates["member"].str.endswith(f"@{seed}")]
        assert set(stresses[stresses["role"] == "stress"]["member"]) == {
            f"R1_stress_spill@{seed}",
            f"R1_stress_lognormal@{seed}",
            f"R1_stress_xplatform@{seed}",
        }
        for name in ("np3_verdicts", "np4_sets", "np5_agreement", "np6_verdicts"):
            tag = f"r1_contam_ho_seed{seed}"
            assert (out / f"{name}__{tag}.csv").is_file(), (name, tag)
    assert record["replicate_identity"] == dict.fromkeys(members, True)
    assert record["prep_identity"]["identical"] is True
    # The ensemble's own NP5 re-derivation is reported beside the members'.
    ensemble = pd.read_csv(out / "np5_ensemble_agreement.csv")
    assert set(ensemble["group"]) == set(DONORS)
    # Every leave-one-donor-out set is a version-7 test set (topped up).
    for item in record["leave_one_donor_out"]:
        assert item["class_top_up"] is not None


def test_version_7_forced_on_a_version_6_family_scores_its_ensemble(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_resources: Any
) -> None:
    # M13 D5 (c) and CHECK K11: the §21 (vii) diagnostic path, never written
    # to a store.
    monkeypatch.setattr(res, "V7_EMISSION_MEMBERS", 2)
    family = Family(tmp_path, monkeypatch, gate_config(ensemble_r1_seeds=[0, 6]))
    summary = json.loads(
        (family.bundle_dir / res.RESOLVABILITY_SUMMARY_FILE).read_text()
    )
    assert summary["resolvability_version"] == 6
    store_before = sorted(path.name for path in (tmp_path / "gate_p_store").rglob("*"))
    result = run.run_gate_p(
        family.request(),
        run.GatePOptions(
            species="human",
            accept_small_pools=True,
            resolvability_version=7,
            prep_identity=False,
        ),
    )
    assert result["status"] == run.STATUS_SCORED, result
    record = read_run(result)
    assert record["forced_version_7"] is True and record["resolvability_version"] == 7
    assert record["bundle_resolvability_version"] == 6
    assert record["members"] == ["R1_contam_HO@0", "R1_contam_HO@6"]
    replicates = pd.DataFrame(record["replicates"])
    # No row comes from PREP: the default donor's version-7 members are
    # simulated on the bundle's own test set and engine.
    assert set(replicates["source"]) == {"simulated"}
    default = replicates[
        (replicates["group"] == DEFAULT_DONOR) & (replicates["seed"] == 0)
    ]
    assert {"R1_contam_HO@0", "R1_contam_HO@6", "clean@0"} <= set(default["member"])
    report = json.loads(
        (tmp_path / "out" / run.GATE_P_RUN_DIR / gp.GATE_P_REPORT_JSON).read_text()
    )
    assert report["resolvability_version"] == 7
    # The store gained only the leave-one-donor-out bundles: no primary
    # bundle holds the version-7 decisions (never written to a store).
    primary = tmp_path / "gate_p_store" / run.PRIMARY_REFERENCE
    assert sorted(path.name for path in primary.iterdir() if path.is_dir()) == [
        family.bundle_dir.name
    ]
    held_out = tmp_path / "gate_p_store" / reference.HO_REFERENCE_ID
    built = {Path(item["path"]).name for item in record["leave_one_donor_out"]}
    added = {
        path.name
        for path in held_out.iterdir()
        if path.is_dir() and path.name not in store_before
    }
    assert added == built

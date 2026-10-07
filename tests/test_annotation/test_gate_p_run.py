"""Tests for the gate-P driver (M13 chunk C8; plan §14, pre-registration §23).

End to end on the synthetic WHB fixtures of ``test_reference`` (three frontal
donors, a fake ``cell_type_mapper`` and a marker-count mapper standing in for
MapMyCells): PREP builds the family's bundle, gate P builds each other
donor's held-out bundle in the request's store, simulates and maps the
replicates, scores NP1-NP9 and writes the report. No MapMyCells, no real
reference.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import zlib
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from merxen.annotation import diagnostics as diag
from merxen.annotation import gate_p as gp
from merxen.annotation import gate_p_run as run
from merxen.annotation import reference
from merxen.annotation import resolvability as res
from merxen.annotation.config import AnnotationConfig
from merxen.annotation.panel import panel_family
from merxen.annotation.reference import builder_for, prepare_reference_spec
from merxen.annotation.simulate import (
    GatePPlan,
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
# The commit an exported tree's COMMIT file names.
EXPORTED = "0123456789abcdef0123456789abcdef01234567"


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
        perturb: Callable[[str, str], bool] | None = None,
    ) -> None:
        # Four times the fixture's donors, so each class has tested sets.
        monkeypatch.setattr(
            tr, "HO_DONORS", {donor: 4 * n for donor, n in tr.HO_DONORS.items()}
        )
        sources, ho, self.fake = whb_resolvability_setup(tmp_path, monkeypatch)
        oracle = oracle_mapper(fixture_truths(ho["metadata"]), error_every=error_every)

        def mapper(engine: Any, query: Any, tag: str = "") -> pd.DataFrame:
            # ``perturb(engine path, tag)`` makes one mapping differ from an
            # identical re-run (a non-reproducible realisation, NP9).
            frame = oracle(engine, query)
            if perturb is not None and perturb(str(engine.path), tag):
                frame["avg_correlation"] = 0.25
            return frame

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
                return mapper(engine, query, tag)

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
                return mapper(engine, query, tag)

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
    # The planted errors all land on other classes. Per call set (scored
    # before pre-registration §23.21) their few truth types take the trim
    # cap, and the class-balanced weighting failed broad Immune at NP3; on
    # the test cells of the scope (R1) a wrong call weighs what its type's
    # test cells weigh, as the unweighted set, and every class validates.
    status = records.set_index(["level", "class"])["status"]
    assert set(status) == {gp.RECORD_VALIDATED}
    np3 = pd.read_csv(out / "np3_verdicts__r1_contam_ho_seed0.csv", dtype={"set": str})
    immune = np3[(np3["level"] == "broad") & (np3["class"] == "Immune")]
    immune = immune.set_index(["set", "scheme"])
    per_set = immune.loc[("250", gp.NP3_CLASS_BALANCED)]
    on_cells = immune.loc[("250", gp.NP3_CLASS_BALANCED_TEST_CELLS)]
    assert not per_set["scored"] and not per_set["passed"]
    assert on_cells["scored"] and on_cells["passed"]
    assert on_cells["precision"] == pytest.approx(
        immune.loc[("250", gp.NP3_UNWEIGHTED), "precision"]
    )
    # One depth walk per member over NP3-NP7 (R2) gives the records' depths.
    walk = pd.read_csv(out / "depth_walk__r1_contam_ho_seed0.csv")
    walked = walk.set_index(["level", "class"])["validated_min_depth"]
    depths = records.set_index(["level", "class"])["validated_min_depth"]
    assert all(depths[key] == walked[key] for key in depths.index)
    # The family checks fail (no gene-ID entry, unaccepted parents, no time
    # reference); every C_P class is validated up to supercluster.
    assert result["validated_max_level"] == "supercluster" and not result["passes"]
    tested = pd.read_csv(out / "tested_sets__r1_contam_ho_seed0.csv")
    assert (tested["n_confident"].dropna() >= 5).all()
    # The dry run (D28) passes. At supercluster it expects only the classes
    # with >= 50 of the default donor's test cells (H18's set).
    dry = record["dry_run"]
    assert dry["passes"] and dry["failing"] == []
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


def run_passing_family(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    accepted_parents: Callable[[dict[str, Any]], tuple[str, ...]] | None = None,
    dry_run: bool = True,
) -> tuple[Family, dict[str, Any], list[gp.GatePResult]]:
    """Gate P on a family whose classes all validate.

    Args:
        tmp_path: The test's directory.
        monkeypatch: The test's monkeypatch.
        accepted_parents: The user's NP2 entries, from PREP's ``np2_inputs``
            (default: every weak and collapsed parent's lookup key).
        dry_run: Run as the seeded family's dry run.

    Returns:
        The family, the driver's run record and the ``GatePResult`` it
        assembled (captured from ``gate_p.assemble_gate_p``).
    """
    config = gate_config().model_copy(
        update={"panel": gate_config().panel.model_copy(update={"min_root_markers": 3})}
    )
    family = Family(tmp_path, monkeypatch, config, error_every=10**9)
    # The fake ctm steps record no memory; NP9's RSS part needs one.
    monkeypatch.setattr(
        run, "_prep_measurements", lambda *_: (60.0, {"primary:markers": 1.0})
    )
    assembled: list[gp.GatePResult] = []
    assemble = gp.assemble_gate_p

    def capture(**kwargs: Any) -> gp.GatePResult:
        assembled.append(assemble(**kwargs))
        return assembled[-1]

    monkeypatch.setattr(gp, "assemble_gate_p", capture)
    # The user accepts the weak and collapsed parents before the run (NP2).
    inputs = run.np2_inputs(family.bundle_dir)
    keys = {*inputs["weak_parents"], *inputs["collapsed_parents"]}
    assert keys
    entries = (
        tuple(sorted(keys)) if accepted_parents is None else accepted_parents(inputs)
    )
    result = run.run_gate_p(
        family.request(report=passing_report()),
        run.GatePOptions(
            species="human",
            accept_small_pools=True,
            dry_run=dry_run,
            time_reference_seconds=1e6,
            time_reference_basis="test",
            accepted_parents=entries,
        ),
    )
    return family, result, assembled


def test_a_family_whose_classes_all_validate_passes_gate_p(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_resources: Any
) -> None:
    _, result, _ = run_passing_family(tmp_path, monkeypatch)
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


def test_np2_parents_accepted_by_name_pass_np2_in_a_full_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_resources: Any
) -> None:
    """B2 (b): the user's names reach NP2 as the lookup keys PREP records.

    The user accepted "Bergmann glia" and "Upper rhombic lip" by name, not
    by lookup key. A family run (not a dry run) whose PREP lists weak and
    collapsed parents, accepted by node name, node label or (for the root,
    which has neither) lookup key, does not stop, scores NP2's
    ``weak_and_collapsed`` part passed and passes; an entry naming no listed
    parent is reported (``accepted_not_listed``) and accepts nothing.
    """
    seen: dict[str, Any] = {}

    def by_name(inputs: dict[str, Any]) -> tuple[str, ...]:
        names = inputs["parent_names"]
        entries = []
        for position, key in enumerate(
            sorted({*inputs["weak_parents"], *inputs["collapsed_parents"]})
        ):
            if key == reference.ROOT_KEY:
                entries.append(key)
            elif position % 2 == 0 and names.get(key):
                entries.append(str(names[key]))
            else:
                entries.append(reference.parse_lookup_key(key)[1])
        seen["keys"] = sorted({*inputs["weak_parents"], *inputs["collapsed_parents"]})
        seen["entries"] = entries
        return (*entries, "Lower rhombic lip")

    _, result, _ = run_passing_family(
        tmp_path, monkeypatch, accepted_parents=by_name, dry_run=False
    )
    # The fixture's parents are accepted by name and node label, not by key
    # (only the root has neither).
    named = [entry for entry in seen["entries"] if entry not in seen["keys"]]
    assert len(named) >= 2
    assert result["status"] == run.STATUS_SCORED, result
    record = read_run(result)
    assert record["np2_parents"]["unaccepted"] == []
    assert record["np2_parents"]["unmatched"] == ["Lower rhombic lip"]
    report = json.loads(
        (tmp_path / "out" / "gate_p" / gp.GATE_P_REPORT_JSON).read_text()
    )
    np2 = report["family_checks"]["NP2"]
    assert np2["parts"]["weak_and_collapsed"] == gp.CHECK_PASSED
    assert np2["status"] == gp.CHECK_PASSED
    assert np2["detail"]["accepted_parents"] == seen["keys"]
    assert np2["detail"]["unaccepted_parents"] == []
    assert np2["detail"]["accepted_not_listed"] == ["Lower rhombic lip"]
    assert result["passes"], result["reasons"]


def _copy_packaged_tables(directory: Path) -> Path:
    """A copy of the packaged validated tables (the gate-P PR appends to them)."""
    tables = directory / "tables"
    tables.mkdir(parents=True)
    packaged = diag.asset_path(diag.VALIDATED_PANELS_FILE).parent
    for name in (
        diag.VALIDATED_PANELS_FILE,
        diag.VALIDATED_PANEL_LEVELS_FILE,
        diag.VALIDATED_PANEL_GENES_FILE,
    ):
        shutil.copy(packaged / name, tables / name)
    return tables


def _writer_inputs(family: Family, result: gp.GatePResult) -> tuple[Any, ...]:
    """The gate-P PR's rows of a driver result, its gene list and self-map."""
    panel = family.panel
    record, levels = gp.validated_table_rows(
        result,
        panel_id="human_merscope_xenium_gate_p_test",
        panel_role="sample_panel",
        platforms=panel.platforms,
        n_genes=panel.n_genes,
        evidence="m13/gate_p/test",
        date="2026-10-07",
        approving_pr="#0",
        root_marker_source="test",
    )
    manifest = json.loads((family.bundle_dir / "bundle.json").read_text())
    self_map = diag.ResolvabilityTrust.from_bundle_manifest(manifest)
    genes = diag.PanelGeneList(
        panel_id=record.panel_id,
        ensembl_ids=tuple(panel.ensembl_ids),
        symbols=dict(zip(panel.ensembl_ids, panel.symbols, strict=True)),
        root_markers=frozenset(panel.ensembl_ids[:3]),
    )
    return record, levels, genes, self_map


def run_family_with_unvalidated_classes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Family, dict[str, Any], list[gp.GatePResult]]:
    """Gate P on the passing family with one class failed and one not evaluable.

    The fixture's classes all validate, so the driver's depth walk (which
    gives every criterion's verdict, pre-registration §23.21 R2) is wrapped:
    NP7 fails supercluster Immune and NP6 cannot evaluate supercluster
    Astro. The driver assembles, records and writes them as it would real
    verdicts; broad and NT stay validated, so the family passes at NT.
    """
    walk = gp.gate_p_depth_walk

    def unvalidated(
        *args: Any, **kwargs: Any
    ) -> tuple[dict[str, dict[tuple[str, str], Any]], pd.DataFrame]:
        verdicts, table = walk(*args, **kwargs)
        changed = {criterion: dict(values) for criterion, values in verdicts.items()}
        changed["NP7"][("supercluster", "Immune")] = False
        changed["NP6"][("supercluster", "Astro")] = None
        return changed, table

    monkeypatch.setattr(gp, "gate_p_depth_walk", unvalidated)
    return run_passing_family(tmp_path, monkeypatch)


def test_a_driver_result_with_unvalidated_classes_and_key_mix_ups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_resources: Any
) -> None:
    """Non-validated records are written as such; key mix-ups are refused.

    The M13 tidy review's open low findings: the driver-to-writer test fed
    only validated records and caught no class key of another level. Here
    the driver's result holds a ``failed:NP7`` and a ``not_evaluable``
    record at supercluster and the family passes at NT. The gate-P PR's two
    steps keep both statuses without depths, and the trust state built on
    the written tables validates neither at any depth while it validates
    the classes' broad records. Then each lineage / broad / NT /
    supercluster key mix-up put into the driver's records is refused by the
    writer, with nothing written.
    """
    family, outcome, assembled = run_family_with_unvalidated_classes(
        tmp_path, monkeypatch
    )
    (result,) = assembled
    status = result.records.set_index(["level", "class"])["status"]
    assert status[("supercluster", "Immune")] == "failed:NP7"
    assert status[("supercluster", "Astro")] == gp.RECORD_NOT_EVALUABLE
    assert status[("supercluster", "Exc")] == gp.RECORD_VALIDATED
    assert result.passes and outcome["passes"]
    assert result.validated_max_level == "nt" == outcome["validated_max_level"]
    tables = _copy_packaged_tables(tmp_path)
    record, levels, genes, self_map = _writer_inputs(family, result)
    assert record.validated_max_level == "nt"
    by_key = {(item.level, item.class_name): item for item in levels}
    for key in (("supercluster", "Immune"), ("supercluster", "Astro")):
        assert by_key[key].validated_min_depth is None
        assert not by_key[key].is_validated
    diag.write_simulation_family(tables, record, levels, genes, self_map=self_map)
    reread = diag.load_validated_panels(tables)
    stored = {
        (item.level, item.class_name): item
        for item in reread.level_records(result.family_id)
    }
    assert stored[("supercluster", "Immune")].status == "failed:NP7"
    assert stored[("supercluster", "Astro")].status == gp.RECORD_NOT_EVALUABLE
    assert stored[("supercluster", "Immune")].validated_min_depth is None
    manifest = json.loads((family.bundle_dir / "bundle.json").read_text())
    panel = family.panel
    decision = diag.trust_state(
        reference_id=run.PRIMARY_REFERENCE,
        role="primary",
        species="human",
        panel_hash=panel.panel_hash,
        n_panel_genes=panel.n_genes,
        family=panel_family(
            panel.ensembl_ids,
            species="human",
            platforms=panel.platforms,
            known_families=reread.known_families(),
        ),
        validated=reread,
        rules=diag.TrustRules(min_mapped_genes=5, min_root_markers=3),
        coverage=diag.CoverageDiagnostics.from_bundle_manifest(manifest),
        resolvability=self_map,
    )
    assert decision.state == "validated"
    for cls in ("Immune", "Astro"):
        assert not decision.is_validated("supercluster", cls, 10**6)
        depth = int(stored[("broad", cls)].validated_min_depth or 0)
        assert decision.is_validated("broad", cls, depth)
    assert decision.is_validated(
        "supercluster", "Exc", int(stored[("supercluster", "Exc")].validated_min_depth)
    )
    # A lineage / broad / NT / supercluster key mix-up in the driver's records
    # is refused by the writer, and nothing is written (D14 (a), D16).
    mix_ups = (
        # A supercluster-only key at broad and lineage (COP is OPC there).
        ("broad", "Astro", "COP"),
        ("lineage", "Exc", "COP"),
        # A glial class at NT, where NT does not apply.
        ("nt", "Exc", "Astro"),
        # A label table's broad name in place of the floor-class key.
        ("broad", "Astro", "Astrocytes"),
        # A lineage name in place of the floor-class key.
        ("lineage", "Exc", "Neurons"),
        # A supercluster's name in place of the floor-class key.
        ("supercluster", "Exc", "Upper-layer intratelencephalic"),
    )
    for level, cls, wrong in mix_ups:
        records = result.records.copy()
        row = (records["level"] == level) & (records["class"] == cls)
        assert int(row.sum()) == 1, (level, cls)
        records.loc[row, "class"] = wrong
        mixed = dataclasses.replace(result, records=records)
        fresh = _copy_packaged_tables(tmp_path / f"{level}_{wrong}")
        before = {path.name: path.read_bytes() for path in sorted(fresh.iterdir())}
        record, levels, genes, self_map = _writer_inputs(family, mixed)
        with pytest.raises(diag.ValidatedPanelsError, match="consensus class key"):
            diag.write_simulation_family(
                fresh, record, levels, genes, self_map=self_map
            )
        after = {path.name: path.read_bytes() for path in sorted(fresh.iterdir())}
        assert after == before, (level, wrong)


def test_a_driver_result_on_whb_class_keys_goes_through_the_promotion_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_resources: Any
) -> None:
    """The gate-P PR's two steps on the driver's own result (M13 final review).

    The writer's other tests feed hand-made results with toy class keys.
    Here the driver scores the synthetic WHB family, whose cells carry real
    WHB supercluster ids, so its records name the classes as the driver
    derives them. ``validated_table_rows`` and ``write_simulation_family``
    must take them as consensus class keys of their levels, appended to a
    copy of the packaged tables, and RESOLVE's trust state must read them
    back per (level, class) at their validated depths.
    """
    family, outcome, assembled = run_passing_family(tmp_path, monkeypatch)
    (result,) = assembled
    assert result.passes and outcome["passes"]
    assert result.family_id == outcome["family_id"]
    records = result.records
    for level, frame in records.groupby("level"):
        keys = diag.level_class_keys("human", str(level))
        assert keys and set(frame["class"].astype(str)) <= keys, level
    validated = records[records["status"] == gp.RECORD_VALIDATED]
    assert {"broad", "supercluster"} <= set(validated["level"])
    # The gate-P PR appends to the packaged tables (a copy of them here).
    tables = _copy_packaged_tables(tmp_path)
    before = diag.load_validated_panels(tables)
    panel = family.panel
    record, levels, genes, self_map = _writer_inputs(family, result)
    assert self_map is not None
    manifest = json.loads((family.bundle_dir / "bundle.json").read_text())
    diag.write_simulation_family(tables, record, levels, genes, self_map=self_map)
    reread = diag.load_validated_panels(tables)
    assert reread.family_ids() == sorted({*before.family_ids(), result.family_id})
    for family_id in before.family_ids():
        assert reread.family_records(family_id) == before.family_records(family_id)
    stored = {
        (item.level, item.class_name): item
        for item in reread.level_records(result.family_id)
    }
    assert set(stored) == set(
        zip(records["level"].astype(str), records["class"].astype(str), strict=True)
    )
    decision = diag.trust_state(
        reference_id=run.PRIMARY_REFERENCE,
        role="primary",
        species="human",
        panel_hash=panel.panel_hash,
        n_panel_genes=panel.n_genes,
        family=panel_family(
            panel.ensembl_ids,
            species="human",
            platforms=panel.platforms,
            known_families=reread.known_families(),
        ),
        validated=reread,
        rules=diag.TrustRules(min_mapped_genes=5, min_root_markers=3),
        coverage=diag.CoverageDiagnostics.from_bundle_manifest(manifest),
        resolvability=self_map,
    )
    assert decision.state == "validated", decision
    assert decision.validation_basis == "simulation"
    for _, row in validated.iterrows():
        level, cls = str(row["level"]), str(row["class"])
        depth = int(row["validated_min_depth"])
        assert stored[(level, cls)].validated_min_depth == depth
        assert decision.is_validated(level, cls, depth), (level, cls)
        assert not decision.is_validated(level, cls, depth - 1), (level, cls)
    for _, row in records[records["status"] != gp.RECORD_VALIDATED].iterrows():
        assert not decision.is_validated(str(row["level"]), str(row["class"]), 10**6)


def test_gate_p_stops_before_any_build_when_a_donor_pool_is_too_small(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_resources: Any
) -> None:
    # Pre-registration §23.10 open item 2: the pool sizes come first; a donor
    # whose own pool cannot meet the per-class top-up rule stops gate P
    # before any leave-one-donor-out set is built or any replicate mapped.
    family = Family(tmp_path, monkeypatch, gate_config(topup_min_class_test_cells=1000))
    held_out = sorted((tmp_path / "gate_p_store" / reference.HO_REFERENCE_ID).iterdir())
    report = {
        "platform": "MERSCOPE",
        "settings": {"expected_depth": 30},
        "panel": {"gene_ids": {}},
        "references": {},
        "provenance": {"code_commit": EXPORTED, "code_commit_source": "export"},
    }
    result = run.run_gate_p(
        family.request(report=report), run.GatePOptions(species="human")
    )
    assert result["status"] == run.STATUS_STOPPED
    assert "per-class top-up rule" in result["reason"]
    record = read_run(result)
    assert record["status"] == run.STATUS_STOPPED
    # The stopped record names the code that ran, as the simulation did.
    assert (record["code_commit"], record["code_commit_source"]) == (
        EXPORTED,
        "export",
    )
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


def test_np2_acceptance_matches_a_parent_by_key_node_or_name() -> None:
    """B2 (pre-registration §23.19): the user accepted parents by name."""
    np2 = {
        "weak_parents": ["CCN202210140_SUPC/CS202210140_493"],
        "collapsed_parents": ["CCN202210140_SUPC/CS202210140_487"],
        "parent_names": {
            "CCN202210140_SUPC/CS202210140_493": "Bergmann glia",
            "CCN202210140_SUPC/CS202210140_487": "Upper rhombic lip",
        },
    }
    named = run.np2_acceptance(np2, ("Bergmann glia", "Upper rhombic lip"))
    assert named.unaccepted == ()
    assert named.accepted == tuple(sorted(np2["parent_names"]))
    assert named.matched["Bergmann glia"] == ["CCN202210140_SUPC/CS202210140_493"]
    by_key = run.np2_acceptance(np2, ("CCN202210140_SUPC/CS202210140_487",))
    assert by_key.unaccepted == ("CCN202210140_SUPC/CS202210140_493",)
    by_node = run.np2_acceptance(np2, ("CS202210140_493", "Lower rhombic lip"))
    assert by_node.accepted == ("CCN202210140_SUPC/CS202210140_493",)
    # An entry naming no listed parent accepts nothing and is reported.
    assert by_node.unmatched == ("Lower rhombic lip",)
    record = by_node.to_json()
    assert record["unaccepted"] == [
        {"key": "CCN202210140_SUPC/CS202210140_487", "name": "Upper rhombic lip"}
    ]
    twice = {
        "weak_parents": ["L1/a", "L2/b"],
        "collapsed_parents": [],
        "parent_names": {"L1/a": "Same", "L2/b": "Same"},
    }
    with pytest.raises(run.GatePRunError, match="more than one"):
        run.np2_acceptance(twice, ("Same",))


def _weak_parent(monkeypatch: pytest.MonkeyPatch) -> str:
    """Make PREP's bundle list one weak parent the user has not accepted."""
    real = run.np2_inputs

    def with_weak(bundle_dir: Path) -> dict[str, Any]:
        inputs = real(bundle_dir)
        key = "CCN202210140_SUPC/CS202210140_493"
        return {
            **inputs,
            "weak_parents": [*inputs["weak_parents"], key],
            "parent_names": {**inputs["parent_names"], key: "Bergmann glia"},
        }

    monkeypatch.setattr(run, "np2_inputs", with_weak)
    return "CCN202210140_SUPC/CS202210140_493"


def test_gate_p_checks_a_family_profiles_bundle_after_prep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_resources: Any
) -> None:
    """A family's NP5 profile of another primary bundle stops gate P after PREP.

    The family's asset records the bundle its map_first tables were resolved
    with; gate P's own PREP must have built that bundle, before any
    leave-one-donor-out build or mapping. With the same bundle the run goes
    on (here it then stops on an unaccepted weak parent, before any build).
    """
    family = Family(tmp_path, monkeypatch, gate_config())
    _weak_parent(monkeypatch)
    held_out = sorted((tmp_path / "gate_p_store" / reference.HO_REFERENCE_ID).iterdir())
    real = run._np5_depths
    recorded: dict[str, Any] = {}

    def family_profile(request: Any, species: str) -> Any:
        per_class, default, record, profile = real(request, species)
        record = {
            **record,
            "source": run.FAMILY_PROFILE_SOURCE,
            "depth_profile_asset": "np5_depth__test_family",
            "primary_build_hash": recorded["hash"],
        }
        return per_class, default, record, profile

    monkeypatch.setattr(run, "_np5_depths", family_profile)
    built = json.loads((family.bundle_dir / "bundle.json").read_text())["build_hash"]
    options = run.GatePOptions(species="human", accept_small_pools=True)
    recorded["hash"] = "f" * 64
    with pytest.raises(run.GatePRunError, match="np5_depth__test_family was built"):
        run.run_gate_p(family.request(), options)
    assert family.gate_calls == []
    assert (
        sorted((tmp_path / "gate_p_store" / reference.HO_REFERENCE_ID).iterdir())
        == held_out
    )
    recorded["hash"] = built
    result = run.run_gate_p(family.request(), options)
    assert result["status"] == run.STATUS_STOPPED
    assert result["stop_reasons"] == [run.STOP_NP2_PARENTS]
    assert read_run(result)["np5_depth"]["primary_build_hash"] == built
    assert family.gate_calls == []


def test_gate_p_stops_before_any_build_on_an_unaccepted_weak_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_resources: Any
) -> None:
    """B2: any weak parent the user has not accepted comes back before gate P runs."""
    family = Family(tmp_path, monkeypatch, gate_config())
    key = _weak_parent(monkeypatch)
    held_out = sorted((tmp_path / "gate_p_store" / reference.HO_REFERENCE_ID).iterdir())
    inputs = run.np2_inputs(family.bundle_dir)
    others = sorted({*inputs["weak_parents"], *inputs["collapsed_parents"]} - {key})
    result = run.run_gate_p(
        family.request(),
        run.GatePOptions(
            species="human", accept_small_pools=True, accepted_parents=tuple(others)
        ),
    )
    assert result["status"] == run.STATUS_STOPPED
    assert result["stop_reasons"] == [run.STOP_NP2_PARENTS]
    assert "Bergmann glia" in result["reason"]
    record = read_run(result)
    assert record["np2_parents"]["unaccepted"] == [
        {"key": key, "name": "Bergmann glia"}
    ]
    # Nothing was built or mapped, and the user's answer runs in place.
    assert (
        sorted((tmp_path / "gate_p_store" / reference.HO_REFERENCE_ID).iterdir())
        == held_out
    )
    assert family.gate_calls == []
    facts = run._check_request(
        family.request(),
        run.GatePOptions(
            species="human",
            accept_small_pools=True,
            accepted_parents=(*others, "Bergmann glia"),
        ),
    )
    assert facts["out"] == tmp_path / "out" / run.GATE_P_RUN_DIR
    # A dry run does not stop on NP2 (not part of the dry-run rule), and the
    # user can run with the parent unaccepted, NP2 then pending.
    for options in (
        run.GatePOptions(species="human", accept_small_pools=True, dry_run=True),
        run.GatePOptions(
            species="human", accept_small_pools=True, run_with_unaccepted_parents=True
        ),
    ):
        assert not run.np2_stop_applies(
            run.np2_acceptance(run.np2_inputs(family.bundle_dir), ()), options
        )


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
            species="human",
            other_region=run.OTHER_REGION_SHARED,
            prep_identity=False,
            # NP2's fixture parents stay unaccepted (pending), as before B2.
            run_with_unaccepted_parents=True,
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


def test_the_command_registers_gate_p_for_the_species_it_is_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_resources: Any
) -> None:
    from click.testing import CliRunner

    from merxen.annotation import simulate
    from merxen.cli import main as cli_main
    from merxen.cli import run_annotation_panels

    family = Family(tmp_path, monkeypatch, gate_config(), build=False)
    monkeypatch.setattr(
        run_annotation_panels, "_reference_spec", lambda *_: family.spec
    )
    gene_list = tmp_path / "genes.csv"
    gene_list.write_text(
        "gene_symbol,gene_id\n"
        + "\n".join(f"G{index},{gene}" for index, gene in enumerate(GENES))
        + "\n"
    )
    config_file = tmp_path / "annotation_config.json"
    # The fixture's ten genes are a panel (as test_simulate's fixture sets).
    config = family.config.model_copy(
        update={"panel": family.config.panel.model_copy(update={"min_mapped_genes": 5})}
    )
    config_file.write_text(config.model_dump_json())

    def invoke(
        *extra: str,
        out: str = "cli_out",
        depth: bool = True,
        config_path: Path = config_file,
    ) -> Any:
        return CliRunner().invoke(
            cli_main,
            [
                "annotation-panel-simulate",
                "--gene-list",
                str(gene_list),
                "--name",
                "family",
                "--references",
                run.PRIMARY_REFERENCE,
                "--store",
                str(tmp_path / "gate_p_store"),
                "--annotation-config",
                str(config_path),
                "--scratch-dir",
                str(tmp_path / f"{out}_scratch"),
                "--out-dir",
                str(tmp_path / out),
                "--n-processors",
                "2",
                "--max-gb",
                "3",
                *(["--expected-depth", "30"] if depth else []),
                "--platform",
                "MERSCOPE",
                *extra,
            ],
        )

    # M13 D4: --gate-p needs the species; mouse waits for C20. Both are
    # refused before any compute.
    result = invoke("--gate-p", out="no_species")
    assert result.exit_code != 0 and "--species" in result.output
    result = invoke("--gate-p", "--species", "mouse", out="mouse")
    assert result.exit_code != 0 and "C20" in result.output
    assert not (tmp_path / "gate_p_store").exists()
    assert not (tmp_path / "no_species").exists() and not (tmp_path / "mouse").exists()
    # M13 C8 review: what needs no bundle is refused before PREP (it was
    # refused only after both bundles were built and the self-map mapped).
    result = invoke("--gate-p", "--species", "human", out="no_depth", depth=False)
    assert result.exit_code != 0 and "expected depth" in result.output
    # D11 (b): the production store (the config's) is refused as gate P's store.
    production = tmp_path / "production_config.json"
    production.write_text(
        config.model_copy(
            update={"reference_store": tmp_path / "gate_p_store"}
        ).model_dump_json()
    )
    result = invoke(
        "--gate-p", "--species", "human", out="production", config_path=production
    )
    assert result.exit_code != 0 and "production reference store" in result.output
    assert family.prep_calls == [] and family.gate_calls == []
    assert not (tmp_path / "gate_p_store").exists()
    assert not (tmp_path / "no_depth").exists()
    assert not (tmp_path / "production").exists()
    # Run as from a `git archive` export: its COMMIT file names the code that
    # runs, and both reports record it (the dry-run script writes the file).
    export = tmp_path / "export"
    export.mkdir()
    (export / run_annotation_panels.CODE_COMMIT_FILE).write_text(EXPORTED + "\n")
    monkeypatch.setattr(run_annotation_panels, "_SOURCE_ROOT", export)
    result = invoke(
        "--gate-p",
        "--species",
        "human",
        "--gate-p-accept-small-pools",
        "--gate-p-skip-prep-identity",
        "--gate-p-run-with-unaccepted-parents",
    )
    assert result.exit_code == 0, result.output
    assert "gate P scored" in result.output
    report = json.loads((tmp_path / "cli_out" / simulate.REPORT_JSON).read_text())
    assert report["gate_p"]["status"] == run.STATUS_SCORED
    record = json.loads(Path(report["gate_p"]["run"]).read_text())
    assert record["options"]["species"] == "human"
    assert record["options"]["prep_identity"] is False
    assert record["options"]["run_with_unaccepted_parents"] is True
    assert report["provenance"]["code_commit"] == EXPORTED
    assert report["provenance"]["code_commit_source"] == "export"
    assert (record["code_commit"], record["code_commit_source"]) == (
        EXPORTED,
        "export",
    )
    assert (tmp_path / "cli_out" / run.GATE_P_RUN_DIR / gp.GATE_P_REPORT_TXT).is_file()
    # The programme is registered for the command's run only.
    assert simulate.gate_p_hook() is None


def test_new_panel_script_requires_the_species_and_runs_the_command() -> None:
    import importlib.util

    script = Path(__file__).resolve().parents[2] / "scripts/acceptance/new_panel.py"
    spec = importlib.util.spec_from_file_location("new_panel", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.command_args("human", ["--out-dir", "x", "--gate-p"]) == [
        "--species",
        "human",
        "--gate-p",
        "--out-dir",
        "x",
    ]
    with pytest.raises(SystemExit):
        module.command_args("human", ["--species", "mouse"])
    with pytest.raises(SystemExit):
        module.main(["--out-dir", "x"])


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
        family.request(),
        run.GatePOptions(
            species="human",
            accept_small_pools=True,
            # A version-6 reference: never used for a version-7 family.
            time_reference_seconds=1e6,
            time_reference_basis="stated",
            # NP2's fixture parents stay unaccepted (pending), as before B2.
            run_with_unaccepted_parents=True,
        ),
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
    # NP9 (M13 D10 (a)): without the set a dry run's values the time part is
    # not evaluable; the stated version-6 reference is not used.
    np9 = report["family_checks"]["NP9"]
    assert np9["parts"]["time"] == gp.CHECK_NOT_EVALUABLE
    assert np9["detail"]["reference_seconds"] is None
    assert record["time_reference_seconds"] is None
    assert "not used" in record["time_reference_basis"]
    # D10 (a)'s basis counts every scored replicate's simulated cells, PREP's
    # default-donor seed-0 rows included (as the forced dry run counts them).
    prep_rows = replicates[replicates["source"] == "prep_bundle"]
    assert set(prep_rows["member"]) == {*members, "clean@0"}
    assert record["simulated_cells_prep_rows"] == int(prep_rows["n_simulated"].sum())
    assert record["simulated_cells_prep_rows"] == sum(
        summary["n_simulated_cells"][name] for name in [*members, "clean@0"]
    )
    assert record["simulated_cells"] == int(replicates["n_simulated"].sum())
    assert record["simulated_cells"] == (
        record["simulated_cells_this_run"] + record["simulated_cells_prep_rows"]
    )
    # The ensemble's own NP5 re-derivation is reported beside the members'.
    ensemble = pd.read_csv(out / "np5_ensemble_agreement.csv")
    assert set(ensemble["group"]) == set(DONORS)
    # Every leave-one-donor-out set is a version-7 test set (topped up).
    for item in record["leave_one_donor_out"]:
        assert item["class_top_up"] is not None


def test_version_7_np5_scores_the_ensemble_re_derived_per_replicate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_resources: Any
) -> None:
    """Pre-registration §23.21 R7: in version 7, NP5's agreement compares the
    ensemble's emission re-derived per replicate with the frozen ensemble
    decisions, R6 and R2 applying to it; each member's own re-derivation is
    reported only. R3 (c) reads the ensemble's t* re-fitted per replicate.

    Every seed-1 replicate's re-derived ensemble withholds every bin, so the
    ensemble flips throughout the D_P group of every tested class: NP5 fails
    in every member, on the ensemble's seed-1 rows. Before §23.21 the
    ensemble's re-derivation was reported only and could fail nothing.
    """
    monkeypatch.setattr(res, "V7_EMISSION_MEMBERS", 2)
    config = gate_config(version="auto", ensemble_r1_seeds=[0, 6])
    config = config.model_copy(
        update={"panel": config.panel.model_copy(update={"min_root_markers": 3})}
    )
    family = Family(tmp_path, monkeypatch, config, error_every=10**9)
    rederive = gp.np5_rederive_ensemble

    def withheld(cells: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        decisions = rederive(cells, *args, **kwargs)
        if int(pd.unique(cells["seed"])[0]) != 1:
            return decisions
        decisions = decisions.copy()
        emitted = (decisions["status"] == res.STATUS_EMITTED).to_numpy()
        decisions.loc[emitted, "status"] = res.STATUS_NOT_RESOLVABLE
        return decisions

    monkeypatch.setattr(gp, "np5_rederive_ensemble", withheld)
    result = run.run_gate_p(
        family.request(report=passing_report()),
        run.GatePOptions(
            species="human",
            accept_small_pools=True,
            dry_run=True,
            run_with_unaccepted_parents=True,
        ),
    )
    assert result["status"] == run.STATUS_SCORED, result
    out = tmp_path / "out" / run.GATE_P_RUN_DIR
    ensemble = pd.read_csv(out / "np5_ensemble_agreement.csv")
    assert ensemble[ensemble["seed"] == 0]["passed"].all()
    assert not ensemble[ensemble["seed"] == 1]["passed"].all()
    records = pd.read_csv(out / gp.GATE_P_RECORDS_CSV)
    tested = records[records["status"] != gp.RECORD_NOT_EVALUABLE]
    assert len(tested) > 0
    for criteria in tested["failed_criteria"].fillna(""):
        assert "NP5" in str(criteria).split(";"), criteria
    for seed in (0, 6):
        tag = f"r1_contam_ho_seed{seed}"
        walk = pd.read_csv(out / f"depth_walk__{tag}.csv")
        failures = walk[walk["passed"].astype(str) == "False"]["deep_failures"]
        assert len(failures) > 0
        for text in failures.astype(str):
            np5 = [item for item in text.split(";") if item.startswith("NP5:")]
            assert len(np5) == 1, text
            agreement = [
                entry
                for entry in np5[0].removeprefix("NP5:").split(",")
                if entry.startswith("agreement ")
            ]
            assert agreement and all("/1@" in entry for entry in agreement), text
        # Each member's own re-derivation is still written (reported only).
        assert (out / f"np5_agreement__{tag}.csv").is_file()
        consequence = pd.read_csv(out / f"np5_tstar_consequence__{tag}.csv")
        assert set(consequence["threshold_from"]) == {gp.NP5_TSTAR_FROM_ENSEMBLE}
        thresholds = pd.read_csv(out / f"np5_ensemble_thresholds__{tag}.csv")
        assert set(thresholds["group"]) == set(DONORS)
    report = json.loads((out / gp.GATE_P_REPORT_JSON).read_text())
    assert report["scored_readings"] == list(gp.GATE_P_SCORED_READINGS)
    assert "open_readings" not in read_run(result)


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
            # NP2's fixture parents stay unaccepted (pending), as before B2.
            run_with_unaccepted_parents=True,
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
    # D10 (a)'s basis: every scored replicate's simulated cells, here all
    # simulated by the run itself (the default donor's seed-0 rows included).
    assert record["simulated_cells_prep_rows"] == 0
    assert record["simulated_cells"] == record["simulated_cells_this_run"]
    assert record["simulated_cells"] == int(replicates["n_simulated"].sum())
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


# --------------------------------------------------------------------------
# M13 C8 review: the precheck, NP9's identity and time reference


def precheck_plan(
    tmp_path: Path,
    config: AnnotationConfig | None = None,
    *,
    store: ReferenceStore | None = None,
    out: str = "out",
    **depth: Any,
) -> GatePPlan:
    """What ``run_panel_simulation`` hands the precheck (D11 (b) store)."""
    (tmp_path / "scratch").mkdir(exist_ok=True)
    return GatePPlan(
        species="human",
        config=config or gate_config(),
        store=store
        or ReferenceStore(tmp_path / "gate_p_store", scratch_root=tmp_path / "scratch"),
        out_dir=tmp_path / out,
        scratch_dir=tmp_path / "sim_scratch",
        **{"expected_depth": 30, **depth},
    )


def test_the_precheck_refuses_what_needs_no_bundle_before_any_compute(
    tmp_path: Path,
) -> None:
    options = run.GatePOptions(species="human")
    # The separate gate-P store, a fresh output and NP5's depth: it runs.
    run.gate_p_precheck(options)(precheck_plan(tmp_path))

    def refused(
        plan: GatePPlan,
        match: str,
        *,
        error: type[Exception] = run.GatePRunError,
        checked: run.GatePOptions = options,
    ) -> None:
        with pytest.raises(error, match=match):
            run.precheck_gate_p(plan, checked)

    # M13 D4: the species selected when the check started.
    refused(
        precheck_plan(tmp_path),
        "second disjoint WMB",
        error=GatePUnavailableError,
        checked=run.GatePOptions(species="mouse"),
    )
    plan = precheck_plan(tmp_path)
    refused(GatePPlan(**{**plan.__dict__, "species": "mouse"}), "started for human")
    # M13 D11 (b): no store root is, or lies in, a production store.
    config = gate_config()
    refused(
        precheck_plan(
            tmp_path, config.model_copy(update={"reference_store": tmp_path})
        ),
        "production reference store",
    )
    large = ReferenceStore(
        tmp_path / "gate_p_store",
        large_root=tmp_path / "production_large",
        scratch_root=tmp_path / "scratch",
    )
    refused(
        precheck_plan(
            tmp_path,
            config.model_copy(
                update={"reference_store_large": tmp_path / "production_large"}
            ),
            store=large,
        ),
        "--store-large",
    )
    separate = config.model_copy(
        update={
            "reference_store": tmp_path / "production",
            "reference_store_large": tmp_path / "production_large",
        }
    )
    run.precheck_gate_p(precheck_plan(tmp_path, separate), options)
    # The output: outside every store, without an earlier run or replicates
    # (a run stopped on the pool sizes is rerun in place).
    refused(precheck_plan(tmp_path, out="gate_p_store/x"), "inside a reference store")
    earlier = tmp_path / "earlier" / run.GATE_P_RUN_DIR
    earlier.mkdir(parents=True)
    (earlier / run.GATE_P_RUN_JSON).write_text(json.dumps({"status": "scored"}))
    refused(precheck_plan(tmp_path, out="earlier"), "already holds a gate-P run")
    run.precheck_gate_p(
        precheck_plan(tmp_path, out="earlier"),
        run.GatePOptions(species="human", overwrite=True),
    )
    (earlier / run.GATE_P_RUN_JSON).write_text(
        json.dumps({"status": run.STATUS_STOPPED})
    )
    run.precheck_gate_p(precheck_plan(tmp_path, out="earlier"), options)
    (earlier / run.REPLICATES_DIR / "H_big").mkdir(parents=True)
    (earlier / run.REPLICATES_DIR / "H_big" / "x.parquet").write_bytes(b"")
    refused(precheck_plan(tmp_path, out="earlier"), "replicates of an earlier")
    # The configured donors and mapping seeds (D6).
    refused(
        precheck_plan(
            tmp_path, gate_config(gate_p_human_donors=["H_small", "H_big", "H_big"])
        ),
        "twice",
    )
    refused(
        precheck_plan(tmp_path, gate_config(gate_p_human_donors=["H_small"])),
        "at least one other donor",
    )
    refused(
        precheck_plan(tmp_path, gate_config(holdout_donor="H_none")),
        "default held-out donor",
    )
    refused(precheck_plan(tmp_path, gate_config(gate_p_seeds=[1, 2])), "gate_p_seeds")
    # NP5's expected depth (D8, CHECK K7): a per-class CSV is never read.
    refused(precheck_plan(tmp_path, expected_depth=None), "expected depth")
    per_class = tmp_path / "per_class.csv"
    pd.DataFrame({"class": ["Astro", "Oligo"], "total_counts": [40, 60]}).to_csv(
        per_class, index=False
    )
    refused(
        precheck_plan(tmp_path, expected_depth=None, depth_profile=per_class),
        "never read for NP5",
    )
    pooled = tmp_path / "pooled.csv"
    pd.DataFrame({"total_counts": [40, 60, 80]}).to_csv(pooled, index=False)
    run.precheck_gate_p(
        precheck_plan(tmp_path, expected_depth=None, depth_profile=pooled), options
    )
    _, default, record, _ = run.np5_depth_source(
        expected_depth=None,
        depth_profile=pooled,
        depth_profile_asset=None,
        species="human",
    )
    assert default == 60.0 and record["source"] == "pooled_profile_median"
    # Nothing was built or written by any of them.
    assert not (tmp_path / "gate_p_store").exists()
    assert not (tmp_path / "out").exists()


def test_same_rows_tells_a_perturbed_replicate_from_an_identical_one() -> None:
    rows = pd.DataFrame(
        {
            "recipe": ["R1_contam_HO"] * 4,
            "seed": [0] * 4,
            "level": ["broad", "broad", "supercluster", "supercluster"],
            "sim_id": ["a", "b", "a", "b"],
            "call": ["Astro", "Exc", "Astrocyte", "Upper-layer IT"],
            "bp": [0.95, 0.9, 0.8, 0.7],
        }
    )
    # The same rows in another order are the same replicate.
    assert run._same_rows(rows, rows.iloc[[3, 1, 0, 2]])
    perturbed = rows.copy()
    perturbed.loc[2, "bp"] = 0.8000001
    assert not run._same_rows(rows, perturbed)
    flipped = rows.copy()
    flipped.loc[1, "call"] = "Astro"
    assert not run._same_rows(rows, flipped)
    assert not run._same_rows(rows, rows.iloc[:3])
    assert not run._same_rows(rows, rows.drop(columns=["bp"]))
    assert not run._same_rows(rows, rows.assign(extra=1))


def writable_copy(source: Path, target: Path) -> Path:
    """A writable copy of a (read-only) bundle directory."""
    shutil.copytree(source, target, copy_function=shutil.copyfile)
    for directory, _, _ in os.walk(target):
        os.chmod(directory, 0o755)
    return target


def test_bundle_differences_lists_every_edited_item(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_resources: Any
) -> None:
    import anndata as ad

    family = Family(tmp_path, monkeypatch, gate_config())
    first = family.bundle_dir
    copies = tmp_path / "copies"
    assert run.bundle_differences(first, writable_copy(first, copies / "same")) == []
    # A re-run's MapMyCells run record (scratch paths, worker count, times) in
    # the marker lookup is not content (the set a dry run's NP9 identity failed
    # on it alone).
    record = writable_copy(first, copies / "record")
    path = record / reference.QUERY_MARKERS_FILTERED_FILE
    markers = json.loads(path.read_text())
    markers["metadata"] = {
        "config": {"query_path": "/scratch/merxen-bundle-other/query.h5ad"},
        "n_processors": 3,
        "timestamp": "another time",
    }
    path.write_text(json.dumps(markers))
    assert run.bundle_differences(first, record) == []
    # An edited marker lookup.
    lookup = writable_copy(first, copies / "lookup")
    path = lookup / reference.QUERY_MARKERS_FILTERED_FILE
    markers = json.loads(path.read_text())
    markers["edited_parent"] = [GENES[0]]
    path.write_text(json.dumps(markers))
    assert run.bundle_differences(first, lookup) == [
        reference.QUERY_MARKERS_FILTERED_FILE
    ]
    # An edited decisions row (the stored table) ...
    decisions = writable_copy(first, copies / "decisions")
    path = decisions / res.RESOLVABILITY_FILE
    table = pd.read_parquet(path)
    numeric = next(
        column
        for column in table.columns
        if pd.api.types.is_float_dtype(table[column]) and table[column].notna().any()
    )
    row = int(np.flatnonzero(table[numeric].notna().to_numpy())[0])
    table.loc[row, numeric] = float(table.loc[row, numeric]) + 0.5
    table.to_parquet(path, index=False)
    assert res.RESOLVABILITY_FILE in run.bundle_differences(first, decisions)
    # ... and an edited simulated cell, from which the decisions are re-derived.
    cells = writable_copy(first, copies / "cells")
    path = cells / res.RESOLVABILITY_CELLS_FILE
    table = pd.read_parquet(path)
    table["bp"] = 0.99
    table.to_parquet(path, index=False)
    differences = run.bundle_differences(first, cells)
    assert res.RESOLVABILITY_CELLS_FILE in differences
    assert "decisions" in differences
    # The held-out test set: one test cell fewer, or one count changed.
    summary = json.loads((first / res.RESOLVABILITY_SUMMARY_FILE).read_text())
    test_dir = Path(summary["test_set_bundle"]["path"])
    fewer = writable_copy(test_dir, copies / "fewer")
    adata = ad.read_h5ad(fewer / res.TEST_CELLS_FILE)
    adata[1:].copy().write_h5ad(fewer / res.TEST_CELLS_FILE)
    assert run.bundle_differences(test_dir, fewer) == [res.TEST_CELLS_FILE]
    counted = writable_copy(test_dir, copies / "counted")
    adata = ad.read_h5ad(counted / res.TEST_CELLS_FILE)
    matrix = adata.X.toarray()
    matrix[0, 0] += 1
    adata.X = matrix
    adata.write_h5ad(counted / res.TEST_CELLS_FILE)
    assert run.bundle_differences(test_dir, counted) == [res.TEST_CELLS_FILE]
    assert run.bundle_differences(test_dir, writable_copy(test_dir, copies / "t")) == []


def test_np9_identity_fails_when_an_identical_re_run_differs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_resources: Any
) -> None:
    member = res.EnsembleMember(
        res.member_recipe(res.DECISION_RECIPE, 0, gate_config().resolvability),
        "emission",
    )
    # The default donor's seed-0 base rows are PREP's, so gate P maps this
    # tag only to re-run the replicate for NP9; PREP's rebuild maps onto the
    # scratch store under the request's scratch directory.
    rerun = f"gate_p_{DEFAULT_DONOR}_seed0_{res.member_tag(member)}"
    scratch = tmp_path / "out_scratch"

    def perturb(engine: str, tag: str) -> bool:
        return tag == rerun or Path(engine).is_relative_to(scratch)

    family = Family(tmp_path, monkeypatch, gate_config(), perturb=perturb)
    result = run.run_gate_p(
        family.request(),
        run.GatePOptions(
            species="human",
            accept_small_pools=True,
            # NP2's fixture parents stay unaccepted (pending), as before B2.
            run_with_unaccepted_parents=True,
        ),
    )
    assert result["status"] == run.STATUS_SCORED, result
    record = read_run(result)
    assert record["replicate_identity"] == {member.name: False}
    prep = record["prep_identity"]
    assert prep["identical"] is False and prep["same_build_hash"] is True
    assert res.RESOLVABILITY_CELLS_FILE in prep["differences"]
    report = json.loads(
        (tmp_path / "out" / run.GATE_P_RUN_DIR / gp.GATE_P_REPORT_JSON).read_text()
    )
    np9 = report["family_checks"]["NP9"]
    assert np9["parts"]["identity"] == gp.CHECK_FAILED
    assert np9["status"] == gp.CHECK_FAILED
    assert np9["detail"]["replicate_identical"] == {member.name: False}
    assert np9["detail"]["bundle_identical"] is False


def test_np9_time_reference_of_a_version_7_family_is_d10s_only() -> None:
    stated = run.GatePOptions(
        species="human", time_reference_seconds=1e6, time_reference_basis="v6"
    )
    # A version-6 family: the reference the caller states.
    assert run.np9_time_reference(
        stated, version7=False, forced_v7=False, simulated_cells=100
    ) == (1e6, "v6")
    # A version-7 family without the dry run's values: the stated version-6
    # reference is not used (D10 (a) fixes it), so the time part is not
    # evaluable.
    seconds, basis = run.np9_time_reference(
        stated, version7=True, forced_v7=False, simulated_cells=100
    )
    assert seconds is None
    assert "D10 (a)" in basis and "not used" in basis
    # The forced version-7 dry run of a version-6 family keeps the stated one.
    assert run.np9_time_reference(
        stated, version7=True, forced_v7=True, simulated_cells=100
    ) == (1e6, "v6")
    # With the dry run's values, the reference scales per simulated cell.
    d10 = run.GatePOptions(
        species="human",
        time_reference_seconds=1e6,
        dry_run_seconds=600.0,
        dry_run_simulated_cells=1200,
    )
    for forced in (False, True):
        seconds, basis = run.np9_time_reference(
            d10, version7=True, forced_v7=forced, simulated_cells=300
        )
        assert seconds == pytest.approx(150.0) and "D10 (a)" in basis
    # A version-6 family never uses the version-7 scaling.
    assert run.np9_time_reference(
        d10, version7=False, forced_v7=False, simulated_cells=300
    ) == (1e6, "")

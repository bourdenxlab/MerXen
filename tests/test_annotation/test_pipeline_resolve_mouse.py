"""Tests for mouse RESOLVE (``mouse_resolve``, ``merxen annotate-resolve``; M6)."""

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

from merxen.annotation.config import AnnotationConfig, MouseGateConfig
from merxen.annotation.mouse_gate import RegistrationSignal
from merxen.annotation.mouse_regions import RegionShareBundle
from merxen.annotation.pipeline import (
    MAP_MANIFEST_NAME,
    ResolveError,
    annotate_resolve,
    load_map_manifest,
    load_resolve_runs,
    read_label_table,
)
from merxen.annotation.schema import validate_label_table
from merxen.annotation.store import BundleRef
from merxen.cli import main as cli_main

from .conftest import (
    MOUSE_CLAS,
    MOUSE_SUBC,
    PROMOTION_CONSTRAINTS,
    FakeMmc,
    map_mouse,
    mouse_region_share_bundle,
)
from .conftest import MOUSE_IDS as IDS
from .conftest import MOUSE_SID as SID

MakeTrust = Callable[..., Any]
PASSING = RegistrationSignal(density_ratio=2.4, shift_um=0.0, source="qc.json")


def _add_profiles(bundle_dir: Path) -> None:
    """Class and subclass profiles for the fake WMB tree (flags, G2)."""
    classes = {
        "01 IT-ET Glut": [0.6, 0.05, 0.01, 0.0001, 0.3399],
        "19 MB Glut": [0.05, 0.6, 0.05, 0.0001, 0.2999],
        "24 MY Glut": [0.01, 0.2, 0.5, 0.0001, 0.2899],
        "30 Astro-Epen": [0.0001, 0.0001, 0.0001, 0.6, 0.3997],
    }
    rows = []
    for name, values in classes.items():
        for level, node in (
            ("CCN20230722_CLAS", name),
            ("CCN20230722_SUBC", f"{name} 1"),
        ):
            rows += [
                {
                    "level": level,
                    "node": node,
                    "node_name": node,
                    "n_cells": 50,
                    "gene_id": gene,
                    "detection_fraction": 0.5,
                    "mean_log2cpm": 1.0,
                    "mean_cpm": value * 1e6,
                    "expected_fraction": value,
                }
                for gene, value in zip(IDS, values, strict=True)
            ]
    pd.DataFrame(rows).to_parquet(bundle_dir / "profiles.parquet")


def _add_panel_coverage(bundle_dir: Path) -> None:
    """The coverage records PREP writes (trust diagnostics without overrides)."""
    path = bundle_dir / "bundle.json"
    manifest = json.loads(path.read_text())
    manifest["builder_output"]["panel_coverage"] = {
        "n_panel_genes": 5,
        "n_query_genes_used": 50,
        "root_markers": 20,
        "root_children": 4,
        "root_children_separated": 4,
    }
    path.write_text(json.dumps(manifest))


def _config(**gate: Any) -> AnnotationConfig:
    config = AnnotationConfig(
        species="mouse",
        mouse_gate=MouseGateConfig(
            g2_min_group_markers=1, g2_min_pseudo_confident=10, **gate
        ),
    )
    return config.coupled_to_clustering(10)


def _resolve(
    setup: dict[str, Any],
    map_dir: Path,
    output: Path,
    make_trust: MakeTrust,
    **kwargs: Any,
) -> Any:
    return annotate_resolve(
        map_dir,
        kwargs.pop("config", _config()),
        output_dir=output,
        panel_dir=setup["panel_dir"],
        trust_overrides={"wmb_panel": make_trust("validated_real", species="mouse")},
        n_bootstrap=5,
        **kwargs,
    )


def test_mouse_resolve_writes_valid_labels_gate_and_region_columns(
    tmp_path: Path, mouse_setup: dict[str, Any], make_trust: MakeTrust
) -> None:
    _add_profiles(mouse_setup["bundle"].path)
    map_dir = tmp_path / "map"
    map_mouse(mouse_setup, map_dir)

    result = _resolve(
        mouse_setup,
        map_dir,
        tmp_path / "resolve",
        make_trust,
        registration={SID: PASSING},
    )

    sample = result.samples[SID]
    labels, _ = read_label_table(sample.labels_path)
    validate_label_table(labels, "mouse")
    labels = labels.set_index("cell_id")
    kinds = pd.Series(mouse_setup["kinds"], index=labels.index)
    my = kinds.index[kinds == "my"]
    ctx = kinds.index[kinds == "ctx"]
    # The five MY sink calls were re-mapped confidently to MB Glut (their
    # MB marker counts): region-dropped with a confident re-map, so kept.
    assert set(labels.loc[my, "ct_class_status"].astype(str)) == {"confident"}
    assert set(labels.loc[my, "ct_class_name"].astype(str)) == {"19 MB Glut"}
    assert not labels.loc[my, "flag_implausible"].any()
    assert set(labels.loc[my, "mmc_wmb_unpruned_class_name"].astype(str)) == {
        "24 MY Glut"
    }
    assert labels.loc[my, "region_pruned_changed"].all()
    assert set(labels.loc[ctx, "ct_class_status"].astype(str)) == {"confident"}
    assert set(labels.loc[ctx, "ct_branch"].astype(str)) == {"01 IT-ET Glut"}
    assert set(labels.loc[ctx, "ct_mender_state"].astype(str)) == {"01 IT-ET Glut"}
    assert labels.loc["M0", "ct_class_status"] == "low_counts"
    # Region-dropped without a confident re-map (an even IT-ET / MB split):
    # implausible at class, kept at broad (plan §4.2, §7.3).
    lost = kinds.index[kinds == "my_lost"]
    assert set(labels.loc[lost, "ct_class_status"].astype(str)) == {"implausible"}
    assert labels.loc[lost, "flag_implausible"].all()
    assert set(labels.loc[lost, "ct_broad_status"].astype(str)) == {"confident"}
    assert set(labels.loc[lost, "ct_final_level"].astype(str)) == {"broad"}
    assert not labels.loc[lost, "exclude_hard"].any()
    levels = sample.summary["resolution"]["levels"]
    assert levels["class"]["status_counts"]["implausible"] == 1
    soft = [column for column in labels.columns if column.startswith("soft_class_")]
    assert len(soft) == 34
    table = labels["in_table"].to_numpy(bool)
    assert np.allclose(labels.loc[table, soft].sum(axis=1), 1.0, atol=1e-5)
    # F1 has coordinates; the fake tree has no Immune class: spill-over null.
    assert labels.loc[table, "region_coherence"].notna().all()
    # Coordinates are matched by cell id: each astrocyte has the other among
    # its 30 nearest neighbours.
    astro = kinds.index[kinds == "astro"]
    assert labels.loc[astro, "region_coherence"].tolist() == pytest.approx(
        [1 / 30, 1 / 30]
    )
    assert labels["flag_microglial_spillover"].isna().all()

    gate = sample.summary["mouse_gate"]
    assert gate["level"] == "full"
    assert gate["signal_status"]["g1"] == "pass"
    assert gate["signal_status"]["g5"] == "not_evaluated"
    assert gate["signals"]["g3_implausible_share"] == pytest.approx(6 / 608, abs=1e-6)
    assert not [r for r in gate["warning_reasons"] if r.startswith("region_")]
    assert gate["signal_status"]["g4"] == "not_evaluated"  # no window sections
    assert gate["referee"]["markers"]["Astrocytes/Ependymal"] == ["Aqp4"]
    provenance = sample.provenance
    assert provenance.mouse_gate is not None
    assert provenance.mouse_gate.section_regions == ["Isocortex", "MB"]
    assert provenance.mouse_gate.region_source == "auto"
    assert provenance.mouse_gate.n_remapped == 6
    # The drop list, the region step and the region model are in the uns.
    pairs = json.dumps([[MOUSE_CLAS, "CL_24"]], sort_keys=True).encode("utf-8")
    assert provenance.mouse_gate.drop_list_size == 1
    assert provenance.mouse_gate.drop_list_sha256 == hashlib.sha256(pairs).hexdigest()
    assert (
        "region_step: pruned (auto; 1 node(s) dropped, 6 cell(s) re-mapped)"
        in provenance.mouse_gate.reasons
    )
    share_ref = provenance.references["wmb_region_share"]
    assert (share_ref.role, share_ref.build_hash) == ("region_share", "e" * 64)
    assert share_ref.bundle_path == str(mouse_setup["region_dir"])
    assert provenance.gate is None
    assert provenance.flags.null_reasons["microglial_spillover"] == (
        "insufficient_panel_genes"
    )
    summary = json.loads(result.summary_path.read_text())
    assert summary["species"] == "mouse"
    step = summary["samples"][SID]["region_step"]
    assert step["status"] == "pruned"
    assert step["drop_list_sha256"] == provenance.mouse_gate.drop_list_sha256
    assert step["region_share_build_hash"] == "e" * 64
    assert step["region_share_problem"] is None
    # Plan §7.2 step 5: class coherence next to the MERFISH coherence (E7).
    per_class = summary["samples"][SID]["region_coherence"]["per_class"]
    astro_class = per_class["30 Astro-Epen"]
    assert astro_class["n_cells"] == 2
    assert astro_class["median_coherence"] == pytest.approx(1 / 30, abs=1e-6)
    assert astro_class["merfish_home_median"] == pytest.approx(0.1333)
    assert astro_class["ratio"] == pytest.approx((1 / 30) / 0.1333, abs=1e-5)
    assert per_class["01 IT-ET Glut"]["n_cells"] == 301  # with the my_lost call
    assert summary["samples"][SID]["class_corr_floor"]["value"] is None


def test_mouse_resolve_fails_the_gate_on_a_misregistered_section(
    tmp_path: Path, mouse_setup: dict[str, Any], make_trust: MakeTrust
) -> None:
    _add_profiles(mouse_setup["bundle"].path)
    map_dir = tmp_path / "map"
    map_mouse(mouse_setup, map_dir)
    bad = RegistrationSignal(density_ratio=1.19, shift_um=114.1)

    result = _resolve(
        mouse_setup, map_dir, tmp_path / "resolve", make_trust, registration={SID: bad}
    )

    labels = result.samples[SID].labels
    table = labels["in_table"].to_numpy(bool)
    for level in ("broad", "class", "nt", "subclass"):
        assert set(labels.loc[table, f"ct_{level}_status"].astype(str)) == {
            "not_attempted_gate"
        }
    assert labels["exclude_hard"].all()
    gate = result.samples[SID].summary["mouse_gate"]
    assert gate["level"] == "failed"
    assert gate["level_reasons"][0].startswith("g1_registration")


def test_mouse_resolve_applies_the_corr_floor_to_validated_families_only(
    tmp_path: Path, mouse_setup: dict[str, Any], make_trust: MakeTrust
) -> None:
    """The class correlation floor applies to real-data-validated panels (§16)."""
    map_dir = tmp_path / "map"
    map_mouse(mouse_setup, map_dir)
    config = _config()
    config = config.model_copy(
        update={
            "thresholds": config.thresholds.model_copy(
                update={"wmb_class_min_corr": 0.99}
            )
        }
    )
    ctx = [
        f"M{index}" for index, kind in enumerate(mouse_setup["kinds"]) if kind == "ctx"
    ]

    real = _resolve(
        mouse_setup,
        map_dir,
        tmp_path / "real",
        make_trust,
        config=config,
        registration={SID: PASSING},
    )
    simulated = annotate_resolve(
        map_dir,
        config,
        output_dir=tmp_path / "simulated",
        panel_dir=mouse_setup["panel_dir"],
        trust_overrides={
            "wmb_panel": make_trust("validated_simulation", species="mouse")
        },
        n_bootstrap=5,
        registration={SID: PASSING},
    )

    # The fake mapper's avg_correlation is 0.5: every class call is below 0.99.
    floor = real.samples[SID].summary["class_corr_floor"]
    assert floor["value"] == 0.99 and floor["note"].startswith("applied")
    labels = real.samples[SID].labels.set_index("cell_id")
    assert set(labels.loc[ctx, "ct_class_status"].astype(str)) == {"low_confidence"}
    floor = simulated.samples[SID].summary["class_corr_floor"]
    assert floor["value"] is None and floor["note"].startswith("not applied")
    labels = simulated.samples[SID].labels.set_index("cell_id")
    assert "low_confidence" not in set(labels.loc[ctx, "ct_class_status"].astype(str))


def test_mouse_resolve_without_a_registration_check_warns(
    tmp_path: Path, mouse_setup: dict[str, Any], make_trust: MakeTrust
) -> None:
    map_dir = tmp_path / "map"
    map_mouse(mouse_setup, map_dir)  # no profiles: flags and G2 not evaluable

    result = _resolve(mouse_setup, map_dir, tmp_path / "resolve", make_trust)

    gate = result.samples[SID].summary["mouse_gate"]
    assert gate["level"] == "full" and gate["warning"]
    assert gate["signal_status"]["g1"] == "not_evaluated"
    assert gate["signal_status"]["g2"] == "not_evaluated"


def test_annotate_resolve_cli_reads_the_registration_check(
    tmp_path: Path, mouse_setup: dict[str, Any], fake_mmc: FakeMmc
) -> None:
    _add_profiles(mouse_setup["bundle"].path)
    _add_panel_coverage(mouse_setup["bundle"].path)
    map_dir = tmp_path / "map"
    map_mouse(mouse_setup, map_dir)
    qc = tmp_path / "qc" / "s_registration_qc.json"
    qc.parent.mkdir()
    qc.write_text(json.dumps({"status": "warn", "density_ratio": 1.3, "shift_um": 0.0}))
    base = [
        "annotate-resolve",
        "--map-dir",
        str(map_dir),
        "--panel-dir",
        str(mouse_setup["panel_dir"]),
        "--no-alignment-lookup",
    ]

    result = CliRunner().invoke(
        cli_main,
        [*base, "--out", str(tmp_path / "out"), "--registration-qc", f"{SID}={qc}"],
    )

    assert result.exit_code == 0, result.output
    assert "gate failed" in result.output
    summary = json.loads(
        next((tmp_path / "out").glob("*_resolve_summary.json")).read_text()
    )
    assert summary["samples"][SID]["mouse_gate"]["signals"]["g1_density_ratio"] == 1.3

    unknown = CliRunner().invoke(
        cli_main,
        [*base, "--out", str(tmp_path / "out2"), "--registration-qc", f"OTHER={qc}"],
    )
    assert unknown.exit_code != 0 and "unknown sample" in unknown.output
    bare = CliRunner().invoke(
        cli_main,
        [
            *base,
            "--out",
            str(tmp_path / "out3"),
            "--registration-qc",
            str(qc),
            "--mouse-g4-sections",
            "C57BL6J-638850.31",
        ],
    )
    assert bare.exit_code == 0, bare.output
    gate = json.loads(
        next((tmp_path / "out3").glob("*_resolve_summary.json")).read_text()
    )
    notes = gate["samples"][SID]["mouse_gate"]["notes"]
    # The fake region-share bundle has no section composition: G4 is a note.
    assert any("section_composition" in note for note in notes)


def test_mouse_resolve_refuses_a_primary_run_of_another_reference(
    tmp_path: Path, mouse_setup: dict[str, Any], make_trust: MakeTrust
) -> None:
    from dataclasses import replace

    from merxen.annotation.mouse_resolve import resolve_mouse_sample
    from merxen.annotation.panel import load_annotation_panel
    from merxen.annotation.pipeline import (
        MAP_MANIFEST_NAME,
        ResolveError,
        load_map_manifest,
        load_resolve_runs,
        load_samples,
    )

    map_dir = tmp_path / "map"
    map_mouse(mouse_setup, map_dir)
    manifest = load_map_manifest(map_dir / MAP_MANIFEST_NAME)
    record = manifest.samples[SID]
    runs = load_resolve_runs(map_dir, record)
    run = runs["wmb_panel"]
    runs = {
        "wmb_panel": replace(
            run, record=run.record.model_copy(update={"reference_id": "other_ref"})
        )
    }
    config = _config()
    (loaded,) = load_samples([mouse_setup["sample"]], config, min_counts=10)
    panel = load_annotation_panel(mouse_setup["panel_dir"] / "panel_genes.json")
    with pytest.raises(ResolveError, match="not the mouse primary wmb_panel"):
        resolve_mouse_sample(
            loaded,
            record,
            runs,
            config,
            manifest=manifest,
            panels={panel.panel_hash: panel},
        )


def _add_marker_unsupported(bundle_dir: Path, level: str, node: str, name: str) -> None:
    """List one node as marker-unsupported, as the WMB builder records it."""
    path = bundle_dir / "bundle.json"
    manifest = json.loads(path.read_text())
    details = [
        {
            "level": level,
            "node": node,
            "name": name,
            "reason": "no_marker_training_cell",
        }
    ]
    manifest["builder_output"]["marker_unsupported_nodes"] = {
        "policy": "report_unresolved",
        "n_nodes": 1,
        "nodes": [f"{level}/{node}"],
        "details": details,
    }
    path.write_text(json.dumps(manifest))


def test_marker_unsupported_subclass_calls_are_not_resolvable(
    tmp_path: Path, mouse_setup: dict[str, Any], make_trust: MakeTrust
) -> None:
    """A subclass without marker-training cells is never emitted (§7.8, R15)."""
    from merxen.annotation.mouse_resolve import marker_unsupported_keys

    map_dir = tmp_path / "map"
    map_mouse(mouse_setup, map_dir)
    plain = _resolve(
        mouse_setup, map_dir, tmp_path / "plain", make_trust, registration={}
    )
    _add_marker_unsupported(
        mouse_setup["bundle"].path, MOUSE_SUBC, "CL_30_1", "30 Astro-Epen 1"
    )
    manifest = load_map_manifest(map_dir / MAP_MANIFEST_NAME)
    runs = load_resolve_runs(map_dir, manifest.samples[SID])
    assert marker_unsupported_keys(runs["wmb_panel"]) == {(MOUSE_SUBC, "CL_30_1")}

    result = _resolve(
        mouse_setup, map_dir, tmp_path / "resolve", make_trust, registration={}
    )

    before = plain.samples[SID].labels.set_index("cell_id")
    labels = result.samples[SID].labels.set_index("cell_id")
    kinds = pd.Series(mouse_setup["kinds"], index=labels.index)
    astro = kinds.index[kinds == "astro"]
    assert set(before.loc[astro, "ct_subclass_status"].astype(str)) == {"confident"}
    assert set(labels.loc[astro, "ct_class_status"].astype(str)) == {"confident"}
    assert set(labels.loc[astro, "ct_subclass_status"].astype(str)) == {
        "not_resolvable"
    }
    assert set(labels.loc[astro, "ct_final_level"].astype(str)) == {"class"}
    # Other cells keep their subclass status.
    others = kinds.index[kinds.isin(["ctx", "mb"])]
    assert (
        labels.loc[others, "ct_subclass_status"].astype(str)
        == before.loc[others, "ct_subclass_status"].astype(str)
    ).all()
    calls = result.samples[SID].summary["marker_unsupported_calls"]
    assert calls == {"class": 0, "subclass": len(astro)}
    assert plain.samples[SID].summary["marker_unsupported_calls"]["subclass"] == 0


def test_mouse_resolve_refuses_a_map_output_without_the_region_step(
    tmp_path: Path, mouse_setup: dict[str, Any], make_trust: MakeTrust
) -> None:
    """A pre-M6 MAP output would resolve unpruned calls behind a clean gate."""
    map_dir = tmp_path / "map"
    map_mouse(mouse_setup, map_dir)
    path = map_dir / MAP_MANIFEST_NAME
    manifest = load_map_manifest(path)
    sample = manifest.samples[SID].model_copy(update={"mouse_regions": None})
    manifest.model_copy(update={"samples": {SID: sample}}).write(path)
    with pytest.raises(ResolveError, match="predates the mouse region step"):
        _resolve(mouse_setup, map_dir, tmp_path / "a", make_trust)

    stale = manifest.model_copy(
        update={"mouse_region_step": "not_run: the region step lands in M6"}
    )
    stale.write(path)
    with pytest.raises(ResolveError, match="predates the mouse region step"):
        _resolve(mouse_setup, map_dir, tmp_path / "b", make_trust)


def test_skipped_pruning_and_a_differing_override_warn(
    tmp_path: Path, mouse_setup: dict[str, Any], make_trust: MakeTrust
) -> None:
    """Plan §7.2: a skipped region step warns; so does an override != inference."""
    from merxen.annotation.pipeline import MapSample

    from .conftest import write_mouse_h5ad

    h5ad, _ = write_mouse_h5ad(tmp_path / "flat" / f"{SID}.h5ad", spatial=False)
    flat = MapSample(SID, "MERSCOPE", h5ad, "prepared")
    map_mouse(mouse_setup, tmp_path / "map_flat", sample=flat)
    skipped = annotate_resolve(
        tmp_path / "map_flat",
        _config(),
        output_dir=tmp_path / "resolve_flat",
        panel_dir=mouse_setup["panel_dir"],
        samples=[flat],
        trust_overrides={"wmb_panel": make_trust("validated_real", species="mouse")},
        n_bootstrap=5,
        registration={SID: PASSING},
    )
    gate = skipped.samples[SID].summary["mouse_gate"]
    assert gate["warning"]
    assert any(
        reason.startswith("region_step: skipped_no_coordinates")
        for reason in gate["warning_reasons"]
    )
    assert gate["signal_status"]["g3"] == "not_evaluated"
    reasons = skipped.samples[SID].provenance.mouse_gate.reasons
    assert any(
        reason.startswith("region_step: skipped_no_coordinates") for reason in reasons
    )

    map_mouse(
        mouse_setup,
        tmp_path / "map_override",
        section_regions={SID: "Isocortex;MB;MY"},
    )
    override = _resolve(
        mouse_setup,
        tmp_path / "map_override",
        tmp_path / "resolve_override",
        make_trust,
        registration={SID: PASSING},
    )
    gate = override.samples[SID].summary["mouse_gate"]
    assert any(
        reason.startswith("region_step: the override differs from the inferred")
        for reason in gate["warning_reasons"]
    )
    # A no_nodes_dropped section is evaluated by G3 (nothing is T2 there).
    assert gate["signal_status"]["g3"] == "pass"
    assert gate["signals"]["g3_implausible_share"] == 0.0


def test_resolve_checks_the_region_share_bundle_it_reads(
    tmp_path: Path, mouse_setup: dict[str, Any], make_trust: MakeTrust
) -> None:
    """RESOLVE reads the region-share build MAP used, or warns without it."""
    map_dir = tmp_path / "map"
    map_mouse(mouse_setup, map_dir)
    other = mouse_region_share_bundle(tmp_path / "other", build_hash="f" * 64)
    with pytest.raises(ResolveError, match="build_hash"):
        _resolve(
            mouse_setup,
            map_dir,
            tmp_path / "a",
            make_trust,
            bundle_overrides={"wmb_region_share": other},
        )
    with pytest.raises(ResolveError, match="build_hash"):
        _resolve(
            mouse_setup, map_dir, tmp_path / "b", make_trust, region_share_dir=other
        )

    # A staged copy of the same build is read from where it was staged.
    staged = tmp_path / "staged" / ("e" * 64)
    shutil.copytree(mouse_setup["region_dir"], staged)
    result = _resolve(
        mouse_setup,
        map_dir,
        tmp_path / "c",
        make_trust,
        region_share_dir=staged,
        registration={SID: PASSING},
    )
    sample = result.samples[SID]
    assert sample.provenance.references["wmb_region_share"].bundle_path == str(staged)
    assert sample.summary["mouse_gate"]["signal_status"]["g3"] == "pass"

    # The recorded bundle is gone and none is given: a gate warning, not a note.
    path = map_dir / MAP_MANIFEST_NAME
    manifest = json.loads(path.read_text())
    manifest["samples"][SID]["mouse_regions"]["region_share"]["path"] = str(
        tmp_path / "missing"
    )
    path.write_text(json.dumps(manifest))
    missing = _resolve(
        mouse_setup, map_dir, tmp_path / "d", make_trust, registration={SID: PASSING}
    )
    gate = missing.samples[SID].summary["mouse_gate"]
    assert gate["warning"]
    assert any(
        reason.startswith("region_share: ") for reason in gate["warning_reasons"]
    )
    assert gate["signal_status"]["g3"] == "not_evaluated"
    assert "wmb_region_share" not in missing.samples[SID].provenance.references


def test_mouse_g4_offsets_compare_astro_epen_and_immune_to_the_window(
    tmp_path: Path, mouse_setup: dict[str, Any], make_trust: MakeTrust
) -> None:
    composition = pd.DataFrame(
        {
            "section": ["S1", "S1"],
            "ap_ccf_mm": [5.0, 5.0],
            "n_cells": [1000, 1000],
            "level": ["class", "class"],
            "node_name": ["30 Astro-Epen", "34 Immune"],
            "n": [100, 30],
            "freq": [0.1, 0.03],
        }
    )
    region_dir = mouse_region_share_bundle(
        tmp_path / "store_g4", build_hash="d" * 64, composition=composition
    )
    map_dir = tmp_path / "map"
    map_mouse(
        mouse_setup, map_dir, region_shares=RegionShareBundle.from_dir(region_dir)
    )

    result = _resolve(
        mouse_setup,
        map_dir,
        tmp_path / "resolve",
        make_trust,
        config=_config(g4_window_sections=["S1"]),
        registration={SID: PASSING},
    )

    gate = result.samples[SID].summary["mouse_gate"]
    # Two astrocytes of 608 table cells vs 10% Astro-Epen; no Immune vs 3%.
    assert gate["signals"]["g4_astro_epen_points"] == pytest.approx(
        100 * (2 / 608 - 0.1), abs=1e-3
    )
    assert gate["signals"]["g4_immune_points"] == pytest.approx(-3.0, abs=1e-6)
    assert gate["signal_status"]["g4"] == "warn"


def test_class_shares_renormalise_the_soft_mass() -> None:
    """Soft shares are over the allocated mass; the rest is ``unallocated``."""
    from merxen.annotation.mouse_resolve import class_shares

    soft = np.array([[1.0, 0.0], [0.25, 0.25]])
    shares = class_shares(soft, ["A", "B"], argmax=["A", "A"], confident=["A", None])
    assert shares["soft"]["A"] == pytest.approx(1.25 / 1.5)
    assert shares["soft"]["B"] == pytest.approx(0.25 / 1.5)
    assert shares["soft"]["unallocated"] == pytest.approx(0.25)
    assert shares["confident"] == {"A": 1.0, "B": 0.0, "n_cells": 2.0}
    assert shares["argmax"]["A"] == 1.0


def _bundle_refs(root: Path, setup: dict[str, Any]) -> list[str]:
    """Stage PREP's bundle refs of the fake mouse run (a pipeline task)."""
    refs = [
        BundleRef(
            reference_id="wmb_panel",
            species="mouse",
            role="primary",
            panel_hash=setup["panel"].panel_hash,
            build_hash=setup["bundle"].build_hash,
            path=str(setup["bundle"].path),
            store_root=str(Path(setup["bundle"].path).parents[1]),
        ).write(root / "bundle_ref_1.json"),
        BundleRef(
            reference_id="wmb_region_share",
            species="mouse",
            role="region_share",
            panel_hash=None,
            build_hash="e" * 64,
            path=str(setup["region_dir"]),
            store_root=str(Path(setup["region_dir"]).parents[1]),
        ).write(root / "bundle_ref_2.json"),
    ]
    return [item for ref in refs for item in ("--bundle-ref", str(ref))]


def test_pipeline_mouse_resolve_needs_the_registration_check(
    tmp_path: Path, mouse_setup: dict[str, Any], fake_mmc: FakeMmc
) -> None:
    """G1 is required in a pipeline task (§7.5/§7.6), found in the QC outputs."""
    _add_profiles(mouse_setup["bundle"].path)
    _add_panel_coverage(mouse_setup["bundle"].path)
    map_dir = tmp_path / "map"
    map_mouse(mouse_setup, map_dir)
    base = [
        "annotate-resolve",
        "--map-dir",
        str(map_dir),
        "--panel-dir",
        str(mouse_setup["panel_dir"]),
        "--no-alignment-lookup",
        *_bundle_refs(tmp_path / "refs", mouse_setup),
        "--require-bundle-refs",
    ]

    missing = CliRunner().invoke(cli_main, [*base, "--out", str(tmp_path / "a")])
    assert missing.exit_code != 0
    assert "needs the M0a registration check" in missing.output

    qc = tmp_path / "qc_out" / "ag"
    qc.mkdir(parents=True)
    (qc / f"{SID.lower()}_registration_qc.json").write_text(
        json.dumps({"status": "pass", "density_ratio": 2.4, "shift_um": 0.0})
    )
    found = CliRunner().invoke(
        cli_main,
        [
            *base,
            "--registration-qc-dir",
            str(tmp_path / "qc_out"),
            "--out",
            str(tmp_path / "b"),
        ],
    )
    assert found.exit_code == 0, found.output
    summary = json.loads(
        next((tmp_path / "b").glob("*_resolve_summary.json")).read_text()
    )
    gate = summary["samples"][SID]["mouse_gate"]
    assert gate["signal_status"]["g1"] == "pass"
    assert gate["registration_source"].endswith("_registration_qc.json")
    staged = summary["samples"][SID]["region_step"]["region_share_build_hash"]
    assert staged == "e" * 64

    opted_out = CliRunner().invoke(
        cli_main, [*base, "--no-registration-qc", "--out", str(tmp_path / "c")]
    )
    assert opted_out.exit_code == 0, opted_out.output
    summary = json.loads(
        next((tmp_path / "c").glob("*_resolve_summary.json")).read_text()
    )
    assert (
        summary["samples"][SID]["mouse_gate"]["signal_status"]["g1"] == "not_evaluated"
    )
    assert summary["samples"][SID]["mouse_gate"]["warning"]

    both = CliRunner().invoke(
        cli_main,
        [
            *base,
            "--no-registration-qc",
            "--registration-qc-dir",
            str(tmp_path / "qc_out"),
            "--out",
            str(tmp_path / "d"),
        ],
    )
    assert both.exit_code != 0 and "--no-registration-qc goes without" in both.output
    with pytest.raises(ResolveError, match="needs the M0a registration check"):
        annotate_resolve(
            map_dir,
            _config(),
            output_dir=tmp_path / "e",
            panel_dir=mouse_setup["panel_dir"],
            require_registration=True,
        )


@pytest.mark.parametrize("constraint", PROMOTION_CONSTRAINTS)
def test_promotion_by_simulation_never_changes_the_mouse_label_table(
    tmp_path: Path,
    mouse_setup: dict[str, Any],
    promotion_trust: Callable[..., tuple[Any, Any]],
    constraint: str,
) -> None:
    """Mouse RESOLVE end to end: a gate-P promotion changes no emitted column.

    The trust decisions come from ``trust_state`` on the section's panel,
    before and after a gate-P PR lists its family (plan §8.2, §14). The
    label tables agree on every column except ``ct_<L>_validated``, and the
    gate level, its reasons and the class correlation floor agree.
    """
    _add_profiles(mouse_setup["bundle"].path)
    map_dir = tmp_path / "map"
    map_mouse(mouse_setup, map_dir)
    config = _config()
    config = config.model_copy(
        update={
            "thresholds": config.thresholds.model_copy(
                update={"wmb_class_min_corr": 0.99}
            )
        }
    )
    trusts = promotion_trust(
        constraint, species="mouse", gene_ids=mouse_setup["panel"].ensembl_ids
    )
    samples = []
    for name, trust in zip(("before", "after"), trusts, strict=True):
        result = annotate_resolve(
            map_dir,
            config,
            output_dir=tmp_path / name,
            panel_dir=mouse_setup["panel_dir"],
            trust_overrides={"wmb_panel": trust},
            n_bootstrap=5,
            registration={SID: PASSING},
        )
        samples.append(result.samples[SID])
    before, after = samples
    validated = [
        column for column in before.labels.columns if column.endswith("_validated")
    ]
    assert validated
    pd.testing.assert_frame_equal(
        before.labels.drop(columns=validated), after.labels.drop(columns=validated)
    )
    for key in ("level", "level_reasons"):
        assert before.summary["mouse_gate"][key] == after.summary["mouse_gate"][key]
    assert before.summary["class_corr_floor"] == after.summary["class_corr_floor"]
    assert before.summary["class_corr_floor"]["value"] is None
    before_trust, after_trust = trusts
    panels = (before.provenance.panel, after.provenance.panel)
    assert panels[0] is not None and panels[1] is not None
    assert [panel.panel_trust for panel in panels] == [
        before_trust.state,
        after_trust.state,
    ]
    if constraint == "resolvable":
        assert (before_trust.state, after_trust.state) == ("provisional", "validated")
        assert panels[0].banner and not panels[1].banner
    else:
        assert before_trust.state == after_trust.state
        assert not after.labels[validated].to_numpy(bool).any()
        assert panels[0].banner and panels[1].banner


PROMOTED_DIGESTS: dict[str, str] = {
    "validated_panels.csv": "1" * 64,
    "validated_panel_levels.csv": "2" * 64,
}


def _promoted_samples(
    tmp_path: Path,
    setup: dict[str, Any],
    promotion_trust: Callable[..., tuple[Any, Any]],
    *,
    panel_dir: Path | None = None,
    coverage: bool = False,
    config: AnnotationConfig | None = None,
) -> tuple[Any, Any, tuple[Any, Any]]:
    """Resolve the section before and after a gate-P promotion of its panel."""
    _add_profiles(setup["bundle"].path)
    if coverage:
        _add_panel_coverage(setup["bundle"].path)
    map_dir = tmp_path / "map"
    map_mouse(setup, map_dir)
    from merxen.annotation.diagnostics import (
        VALIDATED_PANEL_LEVELS_FILE,
        VALIDATED_PANELS_FILE,
    )

    before_trust, after_trust = promotion_trust(
        "resolvable", species="mouse", gene_ids=setup["panel"].ensembl_ids
    )
    # The digests of the tables the gate-P PR changed.
    trusts = (
        before_trust,
        after_trust.model_copy(
            update={"tables_sha256": dict(PROMOTED_DIGESTS)}, deep=True
        ),
    )
    assert set(PROMOTED_DIGESTS) == {VALIDATED_PANELS_FILE, VALIDATED_PANEL_LEVELS_FILE}
    samples = []
    for name, trust in zip(("before", "after"), trusts, strict=True):
        result = annotate_resolve(
            map_dir,
            config or _config(),
            output_dir=tmp_path / name,
            panel_dir=panel_dir or setup["panel_dir"],
            trust_overrides={"wmb_panel": trust},
            n_bootstrap=5,
            registration={SID: PASSING},
        )
        samples.append(result.samples[SID])
    return samples[0], samples[1], trusts


def _unvalidated(reasons: Any) -> list[str]:
    return [str(item) for item in reasons if str(item).startswith("unvalidated_share")]


def test_mouse_resolve_warns_when_simulation_labels_leave_the_validated_region(
    tmp_path: Path,
    mouse_setup: dict[str, Any],
    promotion_trust: Callable[..., tuple[Any, Any]],
) -> None:
    """§8.2's 10% rule after ``resolve_mouse`` (M13 C11, D15 (a)).

    Over broad, class, nt and subclass, a level whose confident labels lie
    outside the simulation family's validated region more than 10% of the
    time adds an ``unvalidated_share:<L>`` warning to the mouse gate; the
    level and its reasons stay those of the provisional panel, and the
    summary, the resolution and the provenance carry the same verdict.
    """
    from merxen.annotation.consensus import MOUSE_CHAIN
    from merxen.annotation.schema import Columns

    before, after, _ = _promoted_samples(tmp_path, mouse_setup, promotion_trust)
    labels = after.labels
    expected = []
    for level in MOUSE_CHAIN:
        status = labels[Columns.level(level, "status")].astype(str).to_numpy()
        confident = status == "confident"
        if not confident.any():
            continue
        validated = labels[Columns.level(level, "validated")].to_numpy(bool)
        if 1.0 - validated[confident].mean() > 0.10 + 1e-12:
            expected.append(
                f"unvalidated_share:{level}: > 0.1 of confident labels outside "
                "the validated region"
            )
    assert len(expected) >= 2  # the fixture has labels outside the region
    gate = after.summary["mouse_gate"]
    assert _unvalidated(gate["warning_reasons"]) == expected
    assert gate["warning"] is True
    assert _unvalidated(before.summary["mouse_gate"]["warning_reasons"]) == []
    for key in ("level", "level_reasons", "signal_status", "notes"):
        assert gate[key] == before.summary["mouse_gate"][key], key
    assert after.summary["resolution"]["gate"] == gate
    assert after.provenance.mouse_gate is not None
    assert _unvalidated(after.provenance.mouse_gate.reasons) == expected
    _, stored = read_label_table(after.labels_path)
    assert stored is not None and stored.mouse_gate is not None
    assert _unvalidated(stored.mouse_gate.reasons) == expected


def test_mouse_panel_provenance_comes_from_the_panel_diagnostics(
    tmp_path: Path,
    mouse_setup: dict[str, Any],
    promotion_trust: Callable[..., tuple[Any, Any]],
) -> None:
    """Mouse ``PanelProvenance`` is ``diagnostics.panel_provenance`` (M13 C11).

    As for human: the family, its basis, the validated level and table
    digests, the panel mode, the gene-ID diagnostics and the validated
    share per level of the resolve summary.
    """
    from merxen.annotation.diagnostics import panel_diagnostics, panel_provenance

    before, after, trusts = _promoted_samples(
        tmp_path, mouse_setup, promotion_trust, coverage=True
    )
    panel = mouse_setup["panel"]
    bundle = json.loads((mouse_setup["bundle"].path / "bundle.json").read_text())
    for sample, trust in zip((before, after), trusts, strict=True):
        record = sample.provenance.panel
        assert record is not None
        shares = {
            level: float(value["validated_share"])
            for level, value in sample.summary["resolution"]["levels"].items()
            if value["validated_share"] is not None
        }
        expected = panel_provenance(
            trust,
            panel_diagnostics(panel, bundles=[bundle]),
            panel_mode="single_sample",
            validated_share=shares,
            n_missing_panel_genes=0,
        )
        assert record == expected
        assert record.panel_hash == panel.panel_hash
        assert record.panel_mode == "single_sample"
        assert record.n_declared_genes == len(IDS)
        assert record.panel_trust == trust.state
        _, stored = read_label_table(sample.labels_path)
        assert stored is not None and stored.panel == record
    promoted = after.provenance.panel
    assert promoted is not None
    assert (promoted.panel_family, promoted.family_basis) == (
        "mouse_sim_family",
        "listed",
    )
    assert promoted.validation_basis == "simulation"
    assert promoted.validated_max_level == "class"
    assert promoted.validated_panels_sha256 == "1" * 64
    assert promoted.validated_panel_levels_sha256 == "2" * 64
    assert promoted.validated_share and min(promoted.validated_share.values()) < 0.9
    provisional = before.provenance.panel
    assert provisional is not None and provisional.family_basis == "own"
    assert provisional.validation_basis is None and provisional.banner is True


def test_mouse_panel_provenance_without_the_panel_file_keeps_the_trust_fields(
    tmp_path: Path,
    mouse_setup: dict[str, Any],
    promotion_trust: Callable[..., tuple[Any, Any]],
) -> None:
    """Without the panel file the record keeps the trust decision's fields."""
    from merxen.annotation.panel import REQUIRED_BUNDLES_FILE

    bare = tmp_path / "bare_panel"
    bare.mkdir()
    shutil.copy(
        mouse_setup["panel_dir"] / REQUIRED_BUNDLES_FILE, bare / REQUIRED_BUNDLES_FILE
    )
    _, after, (_, trust) = _promoted_samples(
        tmp_path, mouse_setup, promotion_trust, panel_dir=bare
    )
    record = after.provenance.panel
    assert record is not None
    assert record.panel_hash == mouse_setup["panel"].panel_hash
    assert (record.panel_family, record.family_basis) == (trust.family_id, "listed")
    assert record.validation_basis == "simulation"
    assert record.validated_max_level == "class"
    assert record.panel_mode == "single_sample"
    assert record.validated_panels_sha256 == "1" * 64
    assert record.validated_panel_levels_sha256 == "2" * 64
    assert record.validated_share
    assert record.n_declared_genes is None and record.gene_id_resolution == {}
    assert _unvalidated(after.summary["mouse_gate"]["warning_reasons"])


def _unvalidated_levels(labels: pd.DataFrame, limit: float) -> list[str]:
    """Chain levels whose confident labels leave the validated region > limit."""
    from merxen.annotation.consensus import MOUSE_CHAIN
    from merxen.annotation.schema import Columns

    levels = []
    for level in MOUSE_CHAIN:
        status = labels[Columns.level(level, "status")].astype(str).to_numpy()
        confident = status == "confident"
        if not confident.any():
            continue
        validated = labels[Columns.level(level, "validated")].to_numpy(bool)
        if 1.0 - validated[confident].mean() > limit + 1e-12:
            levels.append(level)
    return sorted(levels)


def test_mouse_resolve_applies_the_configured_unvalidated_share_limit(
    tmp_path: Path,
    mouse_setup: dict[str, Any],
    promotion_trust: Callable[..., tuple[Any, Any]],
) -> None:
    """``gate.warn_unvalidated_share`` is the mouse limit too (M13 C11, D15 (a)).

    The fixture's broad and class labels lie outside the validated region
    about half the time and its subclass labels always; a configured limit of
    0.6 keeps only the subclass warning, and its text carries that limit. The
    default (0.10) would also warn at broad and class, so a RESOLVE that
    ignored the configured value fails here.
    """
    from merxen.annotation.config import AnnotationGate

    limit = 0.6
    config = _config()
    config = config.model_copy(
        update={"gate": AnnotationGate(warn_unvalidated_share=limit)}
    )
    assert config.gate.warn_unvalidated_share == limit
    before, after, _ = _promoted_samples(
        tmp_path, mouse_setup, promotion_trust, config=config
    )
    default_limit = AnnotationGate().warn_unvalidated_share
    levels = _unvalidated_levels(after.labels, limit)
    assert levels == ["subclass"]
    assert set(_unvalidated_levels(after.labels, default_limit)) > set(levels)
    expected = [
        f"unvalidated_share:{level}: > {limit} of confident labels outside "
        "the validated region"
        for level in levels
    ]
    gate = after.summary["mouse_gate"]
    assert _unvalidated(gate["warning_reasons"]) == expected
    assert gate["warning"] is True
    assert _unvalidated(before.summary["mouse_gate"]["warning_reasons"]) == []
    assert after.summary["resolution"]["gate"] == gate
    assert after.provenance.mouse_gate is not None
    assert _unvalidated(after.provenance.mouse_gate.reasons) == expected
    _, stored = read_label_table(after.labels_path)
    assert stored is not None and stored.mouse_gate is not None
    assert _unvalidated(stored.mouse_gate.reasons) == expected


@pytest.mark.parametrize(
    "update",
    [
        {"level": "broad_only"},
        {"level_reasons": ("moved: a level reason the first verdict lacks",)},
    ],
    ids=["level", "level_reasons"],
)
def test_mouse_resolve_refuses_a_gate_level_that_moves_with_the_validated_share(
    tmp_path: Path,
    mouse_setup: dict[str, Any],
    promotion_trust: Callable[..., tuple[Any, Any]],
    monkeypatch: pytest.MonkeyPatch,
    update: dict[str, Any],
) -> None:
    """The second mouse gate evaluation may only add warnings (M13 C11).

    RESOLVE evaluates the gate again after ``resolve_mouse``, with the
    validated shares; the statuses were decided at the first verdict's level,
    so a second verdict with another level or other level reasons is refused
    rather than recorded beside labels it did not gate.
    """
    import dataclasses

    import merxen.annotation.mouse_gate as mouse_gate

    original = mouse_gate.evaluate_mouse_gate
    calls: list[bool] = []

    def moving(*args: Any, **kwargs: Any) -> Any:
        verdict = original(*args, **kwargs)
        second = kwargs.get("validated_share") is not None
        calls.append(second)
        return dataclasses.replace(verdict, **update) if second else verdict

    monkeypatch.setattr(mouse_gate, "evaluate_mouse_gate", moving)
    with pytest.raises(AssertionError, match="cannot depend on the validated share"):
        _promoted_samples(tmp_path, mouse_setup, promotion_trust)
    assert calls == [False, True]

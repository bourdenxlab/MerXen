"""Tests for mouse RESOLVE (``mouse_resolve``, ``merxen annotate-resolve``; M6)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from merxen.annotation.config import AnnotationConfig, MouseGateConfig
from merxen.annotation.mouse_gate import RegistrationSignal
from merxen.annotation.pipeline import annotate_resolve, read_label_table
from merxen.annotation.schema import validate_label_table
from merxen.cli import main as cli_main

from .conftest import FakeMmc
from .test_pipeline_regions import IDS, SID, _map, mouse_setup

__all__ = ["mouse_setup"]

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
    _map(mouse_setup, map_dir)

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
    assert labels.loc["M607", "ct_class_status"] == "low_counts"
    soft = [column for column in labels.columns if column.startswith("soft_class_")]
    assert len(soft) == 34
    table = labels["in_table"].to_numpy(bool)
    assert np.allclose(labels.loc[table, soft].sum(axis=1), 1.0, atol=1e-5)
    # F1 has coordinates; the fake tree has no Immune class: spill-over null.
    assert labels.loc[table, "region_coherence"].notna().all()
    assert labels["flag_microglial_spillover"].isna().all()

    gate = sample.summary["mouse_gate"]
    assert gate["level"] == "full"
    assert gate["signal_status"]["g1"] == "pass"
    assert gate["signal_status"]["g5"] == "not_evaluated"
    assert gate["signals"]["g3_implausible_share"] == pytest.approx(5 / 607, abs=1e-6)
    assert gate["signal_status"]["g4"] == "not_evaluated"  # no window sections
    assert gate["referee"]["markers"]["Astrocytes/Ependymal"] == ["Aqp4"]
    provenance = sample.provenance
    assert provenance.mouse_gate is not None
    assert provenance.mouse_gate.section_regions == ["Isocortex", "MB"]
    assert provenance.mouse_gate.region_source == "auto"
    assert provenance.mouse_gate.n_remapped == 5
    assert provenance.gate is None
    assert provenance.flags.null_reasons["microglial_spillover"] == (
        "insufficient_panel_genes"
    )
    summary = json.loads(result.summary_path.read_text())
    assert summary["species"] == "mouse"
    assert summary["samples"][SID]["region_step"]["status"] == "pruned"
    assert summary["samples"][SID]["class_corr_floor"]["value"] is None


def test_mouse_resolve_fails_the_gate_on_a_misregistered_section(
    tmp_path: Path, mouse_setup: dict[str, Any], make_trust: MakeTrust
) -> None:
    _add_profiles(mouse_setup["bundle"].path)
    map_dir = tmp_path / "map"
    _map(mouse_setup, map_dir)
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
    from merxen.annotation.mouse_resolve import class_corr_floor

    config = _config()
    config = config.model_copy(
        update={
            "thresholds": config.thresholds.model_copy(
                update={"wmb_class_min_corr": 0.4}
            )
        }
    )
    value, note = class_corr_floor(
        config, make_trust("validated_real", species="mouse")
    )
    assert value == 0.4 and note.startswith("applied")
    value, note = class_corr_floor(config, make_trust("provisional", species="mouse"))
    assert value is None and note.startswith("not applied")
    value, _ = class_corr_floor(
        _config(), make_trust("validated_real", species="mouse")
    )
    assert value is None


def test_mouse_resolve_without_a_registration_check_warns(
    tmp_path: Path, mouse_setup: dict[str, Any], make_trust: MakeTrust
) -> None:
    map_dir = tmp_path / "map"
    _map(mouse_setup, map_dir)  # no profiles: flags and G2 not evaluable

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
    _map(mouse_setup, map_dir)
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
    _map(mouse_setup, map_dir)
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

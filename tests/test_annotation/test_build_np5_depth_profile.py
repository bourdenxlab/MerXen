"""Tests for the NP5 depth-profile builder (M13 D8 with CHECK K7).

``scripts/annotation/build_np5_depth_profile.py`` reads a new family's
``map_first`` RESOLVE label tables and writes its ``gate_p_profile`` asset;
gate P's NP5 reads it through ``sim_inputs.FamilyDepthProfile`` and
``gate_p_run.np5_depth_source`` (pre-registration §23.20). Synthetic label
tables only; no family output is read.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from merxen.annotation import gate_p as gp
from merxen.annotation import gate_p_run as run
from merxen.annotation import sim_inputs as si
from merxen.annotation.diagnostics import own_family_id
from merxen.annotation.pipeline import write_label_table
from merxen.annotation.provenance import (
    AnnotationProvenance,
    DatasetGateProvenance,
    PanelProvenance,
    ReferenceProvenance,
)
from merxen.annotation.simulate import SimulationError, load_simulation_profile

REPO_ROOT = Path(__file__).resolve().parents[2]
PANEL_HASH = "aa25d5a241d0" + "0" * 52
FAMILY = own_family_id("human", ["MERSCOPE"], PANEL_HASH)
BUILD_HASH = "b" * 64


def _builder() -> Any:
    path = REPO_ROOT / "scripts" / "annotation" / "build_np5_depth_profile.py"
    spec = importlib.util.spec_from_file_location("build_np5_depth_profile", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def builder() -> Any:
    return _builder()


def _provenance(**change: Any) -> AnnotationProvenance:
    values: dict[str, Any] = {
        "mode": "map_first",
        "species": "human",
        "anatomical_region": "frontal_cortex",
        "panel": PanelProvenance(
            panel_hash=PANEL_HASH, panel_trust="provisional", banner=True
        ),
        "references": {
            "whb_frontal_supc_clus": ReferenceProvenance(
                reference_id="whb_frontal_supc_clus",
                role="primary",
                build_hash=BUILD_HASH,
            )
        },
        "gate": DatasetGateProvenance(level="full"),
    }
    values.update(change)
    return AnnotationProvenance(**values)


# Per section: (broad name, NT name, confident, total counts) per cell.
CELLS: list[tuple[str | None, str | None, bool, int]] = (
    [("Neurons", "Excitatory", True, 200 + index) for index in range(80)]
    + [("Neurons", "Inhibitory", True, 150 + index) for index in range(30)]
    + [("Astrocytes", None, True, 90 + index) for index in range(60)]
    + [("Microglia", None, True, 60 + index) for index in range(10)]
    + [("Oligodendrocytes", None, False, 40 + index) for index in range(20)]
    + [(None, None, False, 12) for _ in range(10)]
)


def _labels(
    sample_id: str,
    cells: list[tuple[str | None, str | None, bool, int]] = CELLS,
    *,
    segmentation: str = "proseg_hybrid",
    panel_hash: str = PANEL_HASH,
    low_counts: int = 5,
) -> pd.DataFrame:
    """A label table of ``cells`` (table cells) and ``low_counts`` others."""
    rows = [
        {
            "cell_id": f"{sample_id}-{index}",
            "sample_id": sample_id,
            "platform": "MERSCOPE",
            "segmentation": segmentation,
            "species": "human",
            "panel_hash": panel_hash,
            "total_counts": total,
            "in_table": True,
            "ct_broad_name": broad,
            "ct_broad_status": "confident" if confident else "low_confidence",
            "ct_nt_name": nt,
        }
        for index, (broad, nt, confident, total) in enumerate(cells)
    ]
    rows += [
        {
            "cell_id": f"{sample_id}-low{index}",
            "sample_id": sample_id,
            "platform": "MERSCOPE",
            "segmentation": segmentation,
            "species": "human",
            "panel_hash": panel_hash,
            "total_counts": 3,
            "in_table": False,
            "ct_broad_name": None,
            "ct_broad_status": "low_counts",
            "ct_nt_name": None,
        }
        for index in range(low_counts)
    ]
    frame = pd.DataFrame(rows)
    frame["total_counts"] = frame["total_counts"].astype("int32")
    return frame


def _write(
    directory: Path,
    sample_id: str,
    provenance: AnnotationProvenance | None = None,
    **labels: Any,
) -> Path:
    path = directory / f"{sample_id}_celltype_labels.parquet"
    write_label_table(
        _labels(sample_id, **labels),
        path,
        (provenance or _provenance()).to_uns_json(),
    )
    return path


def _argv(tmp_path: Path, tables: list[Path], **change: Any) -> list[str]:
    evidence = tmp_path / "evidence"
    evidence.mkdir(exist_ok=True)
    manifest = change.get("manifest", evidence / "m13" / "np5" / "manifest.json")
    return [
        *(item for path in tables for item in ("--labels", str(path))),
        "--family-id",
        str(change.get("family_id", FAMILY)),
        "--panel-hash",
        str(change.get("panel_hash", PANEL_HASH[:12])),
        "--tissue",
        "brain_ff",
        "--date",
        "2026-10-07",
        "--evidence-root",
        str(evidence),
        "--manifest",
        str(manifest),
        "--output-dir",
        str(tmp_path / "sim_inputs"),
        "--repo-root",
        str(REPO_ROOT),
    ]


def _four_sections(tmp_path: Path) -> list[Path]:
    tables = tmp_path / "tables"
    tables.mkdir(exist_ok=True)
    return [_write(tables, sample) for sample in ("P5822", "P4815", "P3518", "P7417")]


def test_the_builder_writes_the_family_profile_asset(
    builder: Any, tmp_path: Path
) -> None:
    """Reading (i): proseg_hybrid, the four sections pooled, each cell once."""
    tables = _four_sections(tmp_path)
    argv = _argv(tmp_path, tables)
    assert builder.main(argv) == 0
    directory = tmp_path / "sim_inputs"
    registry = si.load_registry(directory)
    asset = registry[f"np5_depth__{FAMILY}"]
    assert asset.role == si.GATE_P_PROFILE_ROLE and asset.trust_effect == "none"
    assert asset.species == "human" and asset.chemistry == "merscope"
    assert asset.reference == "whb_frontal_supc_clus"
    family = si.family_profile_from_asset(asset, directory)
    # Every table cell once (the low-count objects are not table cells).
    assert family.table_totals.size == 4 * len(CELLS)
    confident = [cell for cell in CELLS if cell[2]]
    assert family.confident.n_cells == 4 * len(confident)
    # The confident calls take the class key gate P decides by (E2 floors).
    counts = {name: len(values) for name, values in family.confident.by_class.items()}
    assert counts == {"Exc": 320, "Inh": 120, "Astro": 240, "Immune": 40}
    # Reading (ii): Immune's 40 pooled calls are below the 100 minimum.
    depths = family.class_depths()
    assert set(depths) == {"Exc", "Inh", "Astro"}
    assert family.below_min_cells() == {"Immune": 40}
    astro = [float(t) for b, _, ok, t in CELLS if ok and b == "Astrocytes"] * 4
    assert sorted(depths["Astro"]) == sorted(astro)
    # Reading (iv): the overall median is the confident calls'.
    totals = [total for *_, ok, total in CELLS if ok] * 4
    assert family.overall_median() == pytest.approx(float(np.median(totals)))
    assert family.table_median() == pytest.approx(
        float(np.median([total for *_, total in CELLS] * 4))
    )
    # The sidecar names the tables by file and sha256, the manifest by path.
    provenance = asset.provenance
    assert provenance[si.SOURCE_KIND_KEY] == si.SOURCE_KIND_IN_HOUSE
    (source,) = provenance["source_files"]
    assert source["path"] == "$A/m13/np5/manifest.json"
    manifest_path = tmp_path / "evidence" / "m13" / "np5" / "manifest.json"
    assert source["sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    names = {item["file"] for item in provenance["derived_from"]}
    assert names == {path.name for path in tables}
    # No local path outside the evidence root reaches the repository.
    assert str(tmp_path / "tables") not in json.dumps(asset.model_dump(mode="json"))
    manifest = json.loads(manifest_path.read_text())
    assert {item["path"] for item in manifest["label_tables"]} == {
        str(path.resolve()) for path in tables
    }
    summary = provenance["summary"]["pooled"]
    assert summary["classes_own_depths"] == ["Astro", "Exc", "Inh"]
    assert summary["classes_overall_median"] == ["Immune"]
    assert [item["sample_id"] for item in provenance["summary"]["sections"]] == [
        "P5822",
        "P4815",
        "P3518",
        "P7417",
    ]
    # --check: the written files are up to date; a changed table is stale.
    assert builder.main([*argv, "--check"]) == 0
    _write(tmp_path / "tables", "P7417", cells=CELLS[:-1])
    assert builder.main([*argv, "--check"]) == 1


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"segmentation": "reseg"}, "not the scored proseg_hybrid"),
        ({"duplicate": True}, "given twice"),
        ({"provenance": {"mode": "legacy"}}, "not a map_first RESOLVE output"),
        ({"trust": "validated"}, "not 'provisional'"),
        ({"build_hash": "c" * 64}, "primary build_hash"),
        ({"panel_hash": "dd" + PANEL_HASH[2:]}, "differ in their panel_hash"),
        ({"flag": "--panel-hash", "value": "dcddfbd18fb8"}, "not the frozen"),
        ({"flag": "--family-id", "value": "human_merscope_dcddfbd18fb8"}, "D16"),
        ({"cells": [("Mixed/Unknown", None, True, 100)]}, "without a floor class"),
        ({"no_provenance": True}, "carries no annotation provenance"),
    ],
)
def test_the_builder_refuses_tables_of_another_run(
    builder: Any, tmp_path: Path, change: dict[str, Any], match: str
) -> None:
    tables = _four_sections(tmp_path)
    directory = tmp_path / "tables"
    target = directory / "P7417_celltype_labels.parquet"
    provenance = _provenance(**change.get("provenance", {}))
    if "trust" in change:
        provenance = _provenance(
            panel=PanelProvenance(
                panel_hash=PANEL_HASH, panel_trust=change["trust"], banner=False
            )
        )
    if "build_hash" in change:
        provenance = _provenance(
            references={
                "whb_frontal_supc_clus": ReferenceProvenance(
                    reference_id="whb_frontal_supc_clus",
                    role="primary",
                    build_hash=change["build_hash"],
                )
            }
        )
    labels: dict[str, Any] = {}
    for key in ("segmentation", "panel_hash", "cells"):
        if key in change:
            labels[key] = change[key] if key != "cells" else [*CELLS, *change[key]]
    if change.get("no_provenance"):
        _labels("P7417").to_parquet(target, index=False)
    else:
        _write(directory, "P7417", provenance, **labels)
    if change.get("segmentation"):
        # Every section of another segmentation: refused as not the scored one.
        tables = [
            _write(directory, sample, segmentation=change["segmentation"])
            for sample in ("P5822", "P4815", "P3518", "P7417")
        ]
    if change.get("duplicate"):
        tables = [*tables, tables[0]]
    argv = _argv(tmp_path, tables)
    if "flag" in change:
        position = argv.index(change["flag"])
        argv[position + 1] = change["value"]
    with pytest.raises(builder.ProfileBuildError, match=match):
        builder.build_files(builder._parse_args(argv))
    assert builder.main(argv) == 1
    assert not (tmp_path / "sim_inputs").exists()


def test_the_manifest_must_lie_under_the_evidence_root(
    builder: Any, tmp_path: Path
) -> None:
    tables = _four_sections(tmp_path)
    argv = _argv(tmp_path, tables, manifest=tmp_path / "elsewhere" / "m.json")
    with pytest.raises(builder.ProfileBuildError, match="not under the evidence root"):
        builder.build_files(builder._parse_args(argv))


def _registered(
    builder: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> si.SimInputAsset:
    """Build the asset and make its directory the packaged one."""
    assert builder.main(_argv(tmp_path, _four_sections(tmp_path))) == 0
    directory = tmp_path / "sim_inputs"
    monkeypatch.setattr(si, "SIM_INPUTS_DIR", directory)
    monkeypatch.setattr(
        si,
        "load_registry",
        lambda path=None: si._load_registry(Path(path or directory)),
    )
    return si.get_asset(f"np5_depth__{FAMILY}")


def test_gate_p_reads_the_family_profile_for_np5(
    builder: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§23.20: own depths from 100 calls, the confident calls' median otherwise."""
    asset = _registered(builder, tmp_path, monkeypatch)
    per_class, default, record, histogram = run.np5_depth_source(
        expected_depth=None,
        depth_profile=None,
        depth_profile_asset=asset.asset_id,
        species="human",
    )
    assert set(per_class) == {"Exc", "Inh", "Astro"}
    assert len(per_class["Exc"]) == 320
    totals = [total for *_, ok, total in CELLS if ok] * 4
    assert default == pytest.approx(float(np.median(totals)))
    assert record["source"] == "family_profile_asset"
    assert record["classes_below_min_cells"] == {"Immune": 40}
    assert record["min_class_cells"] == si.PROFILE_MIN_CLASS_CELLS
    assert record["table_cells_median"] == pytest.approx(
        float(np.median([total for *_, total in CELLS] * 4))
    )
    # NP3's report-only depth histogram is label-free: every table cell.
    assert histogram.pooled and histogram.n_cells == 4 * len(CELLS)
    # A class without calls (supercluster COP) and Immune take the default.
    decisions = pd.DataFrame(
        [
            {
                "level": level,
                "class": cls,
                "depth": depth,
                "regime": "provisional",
                "status": "emitted",
                "extrapolated": depth == 250,
            }
            for level, cls in (
                ("broad", "Exc"),
                ("broad", "Immune"),
                ("supercluster", "Exc"),
                ("supercluster", "COP"),
            )
            for depth in (10, 30, 60, 120, 250)
        ]
    )
    shares = gp.np5_extrapolated_share(
        decisions,
        per_class,
        [10, 30, 60, 120, 250],
        gp.Np5Settings(min_test_cells=50),
        default_depth=default,
    ).set_index(["level", "class"])
    assert shares.loc[("broad", "Exc"), "depth_source"] == "class"
    assert shares.loc[("supercluster", "Exc"), "depth_source"] == "class"
    assert shares.loc[("supercluster", "COP"), "depth_source"] == "default"
    assert shares.loc[("broad", "Immune"), "depth_source"] == "default"
    # The base simulation reads the confident calls as a per-class profile.
    profile = load_simulation_profile(None, asset.asset_id, species="human")
    assert not profile.pooled and profile.n_cells == family_confident_cells()
    # No depth profile crosses species (plan §8.3 v7.5).
    with pytest.raises(SimulationError, match="crosses species"):
        run.np5_depth_source(
            expected_depth=None,
            depth_profile=None,
            depth_profile_asset=asset.asset_id,
            species="mouse",
        )


def family_confident_cells() -> int:
    return 4 * sum(1 for cell in CELLS if cell[2])


def test_no_prep_selects_a_family_profile(
    builder: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PREP's resolvability plan takes ``profile`` assets only (no build_hash)."""
    asset = _registered(builder, tmp_path, monkeypatch)
    assert si.find_assets(role="profile", species="human", chemistry="merscope") == []
    assert si.find_assets(
        role=si.GATE_P_PROFILE_ROLE, species="human", chemistry="merscope"
    ) == [asset]


def test_a_family_profile_with_another_class_minimum_is_refused(
    builder: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asset = _registered(builder, tmp_path, monkeypatch)
    changed = asset.model_copy(update={"extra": {**asset.extra, "min_class_cells": 20}})
    with pytest.raises(si.SimInputError, match="min_class_cells 20"):
        si.family_profile_from_asset(changed)
    unknown = asset.model_copy(
        update={
            "extra": {
                **asset.extra,
                "class_legend": [
                    {**item, "kind": "confident_nt"}
                    for item in asset.extra["class_legend"]
                ],
            }
        }
    )
    with pytest.raises(si.SimInputError, match="unknown legend kinds"):
        si.family_profile_from_asset(unknown)
    with pytest.raises(si.SimInputError, match="not a gate_p_profile"):
        si.family_profile_from_asset(si.get_asset(si.STRESS_HUMAN_XPLATFORM, _real()))


def _real() -> dict[str, si.SimInputAsset]:
    return si._load_registry(
        REPO_ROOT / "src" / "merxen" / "assets" / "annotation" / "sim_inputs"
    )

"""Tests of the M13 C17 real-data QC regression script (set a, M8 pairs)."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest

from merxen import local_config, privacy
from merxen.annotation import real_qc as qc
from merxen.annotation.config import AnnotationConfig
from merxen.annotation.mouse_gate import RegistrationSignal

from .test_real_qc_checks import CLASSES, clean_strata, stratum
from .test_real_qc_outcomes import human_gate

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "acceptance" / "m13_real_qc_regression.py"
MakeTrust = Callable[..., Any]
# HD4's paired concordance records (as test_real_qc_checks' fixture).
HD4_JSD = [
    {"kind": "soft", "region": "whole_section", "jsd": 0.226814, "ci_low": 0.21227},
    {"kind": "soft", "region": "shared_mask", "jsd": 0.234905, "ci_high": 0.25},
    {"kind": "confident", "region": "shared_mask", "jsd": 0.334206},
]
WHB_HASH = "f" * 64
SETC_HASH = "c" * 64
SEAAD_HASH = "d" * 64


@pytest.fixture(scope="module")
def script() -> ModuleType:
    name = "m13_real_qc_regression"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# Fixtures: an M8 RESOLVE run and the re-runs' summaries


def _bundles(*, setc: bool = True) -> dict[str, dict[str, Any]]:
    bundles = {
        "whb_frontal_supc_clus": {
            "reference_id": "whb_frontal_supc_clus",
            "resolved_build_hash": WHB_HASH,
            "mapped_build_hash": "1" * 64,
        },
        "seaad_mr_panel": {
            "reference_id": "seaad_mr_panel",
            "resolved_build_hash": SEAAD_HASH,
            "mapped_build_hash": "2" * 64,
        },
    }
    if setc:
        bundles["whb_frontal_supc_clus_setc"] = {
            "reference_id": "whb_frontal_supc_clus",
            "resolved_build_hash": SETC_HASH,
            "mapped_build_hash": "3" * 64,
        }
    return bundles


def _labels(sample_id: str, statuses: Sequence[str]) -> pd.DataFrame:
    n = len(statuses)
    return pd.DataFrame(
        {
            "cell_id": [f"{sample_id}-{index}" for index in range(n)],
            "in_table": np.ones(n, dtype=bool),
            "ct_broad_status": list(statuses),
            "ct_broad_name": [
                "Neurons" if status == "confident" else None for status in statuses
            ],
        }
    )


def _write_m8_run(
    root: Path,
    pair: str,
    segmentation: str,
    *,
    samples: Mapping[str, str] | None = None,
    bundles: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
    jsd: Sequence[Mapping[str, Any]] = HD4_JSD,
) -> Path:
    """Write an M8 RESOLVE run (run record, summary, label tables)."""
    samples = samples or {f"{pair}_MERSCOPE": "MERSCOPE", f"{pair}_XENIUM": "XENIUM"}
    directory = root / pair / segmentation
    directory.mkdir(parents=True)
    records: dict[str, Any] = {}
    for index, (sample_id, platform) in enumerate(sorted(samples.items())):
        labels = f"{platform.lower()}/{sample_id}_celltype_labels.parquet"
        manifest = f"{platform.lower()}/{sample_id}_annotation_manifest.json"
        (directory / platform.lower()).mkdir()
        _labels(sample_id, ["confident", "low_confidence", "confident"]).to_parquet(
            directory / labels
        )
        records[sample_id] = {
            "platform": platform,
            "n_segmented": 1000 + index,
            "bundles": dict((bundles or {}).get(sample_id, _bundles())),
            "labels": labels,
            "annotation_manifest": manifest,
        }
    summary = {
        "pair_id": pair,
        "segmentation": segmentation,
        "species": "human",
        "samples": records,
        "pair": {
            "alignment_dir": f"/results/{pair}/alignment/align_out",
            "jsd": list(jsd),
        },
    }
    (directory / f"{pair}_resolve_summary.json").write_text(json.dumps(summary))
    (directory / f"{pair}_resolve_run.json").write_text(
        json.dumps(
            {
                "map_manifest": f"/runs/{pair}/{segmentation}/map_manifest.json",
                "summary_sha256": "e" * 64,
            }
        )
    )
    return directory


# The settings C17 ran with (the script's qc variant, C17_REAL_QC): the
# defaults until the user's rulings of 2026-10-07 changed them.
C17_CONFIG = AnnotationConfig.model_validate(
    {
        "species": "human",
        "real_qc": {
            "marker_referee_comparator": "node",
            "marker_referee_comparator_by_family": {},
            "registration_g1_effect": "warning",
            "gene_complexity_matching": "lower_edge",
        },
    }
)


def _qc_result(
    make_trust: MakeTrust,
    *,
    config: AnnotationConfig | None = None,
    **change: Any,
) -> qc.RealQcResult:
    """A seeded set a section's QC (warn-only while gate H is pending)."""
    values: dict[str, Any] = {
        "resolvability_version": 6,
        "paired": True,
        "prefilter_applied": False,
        "has_r3_member": False,
        "pair_jsd": HD4_JSD,
        "flag_strata": clean_strata(),
        "registration": RegistrationSignal(1.98, 0.0, status="pass", source="x"),
        "gate": human_gate(),
    }
    values.update(change)
    return qc.real_data_qc(
        qc.RealQcSignals(**values),
        make_trust("validated_real"),
        config or C17_CONFIG,
    )


def _qc_summary(
    results: Mapping[str, qc.RealQcResult | None],
    *,
    pair: str = "HD4",
    jsd: Sequence[Mapping[str, Any]] = HD4_JSD,
) -> dict[str, Any]:
    """A resolve summary whose samples carry the QC blocks RESOLVE writes.

    The gate is ``human_gate()`` (full) after the sample's QC, as RESOLVE
    records it (``RealQcResult.apply_to_gate``).
    """
    from merxen.annotation.pipeline import real_qc_record

    samples: dict[str, Any] = {}
    for sample_id, result in results.items():
        platform = sample_id.rsplit("_", 1)[1]
        gate = human_gate() if result is None else result.apply_to_gate(human_gate())
        samples[sample_id] = {
            "platform": platform,
            "labels": f"{platform.lower()}/{sample_id}_celltype_labels.parquet",
            "annotation_manifest": (
                f"{platform.lower()}/{sample_id}_annotation_manifest.json"
            ),
            "trust": {"state": "validated"},
            "resolution": {"gate": gate.to_json()},
            "real_qc": real_qc_record(result, enabled=result is not None),
        }
    return {
        "pair_id": pair,
        "species": "human",
        "samples": samples,
        "pair": {"jsd": list(jsd), "cross_platform": {"statistics_level": "full"}},
    }


def _referee_signal(value: float | None, groups: int) -> qc.MarkerConsistencySignal:
    markers = {
        "Microglia": ["CSF1R", "C1QB", "P2RY12"],
        "Astrocytes": ["AQP4", "GJA1", "SLC1A3"],
    }
    return qc.MarkerConsistencySignal(
        value,
        groups,
        900 if value is not None else 0,
        reason=None if value is not None else f"{groups} marker group(s)",
        details={
            "n_scored": 900 if value is not None else 0,
            "n_marker_groups": groups,
            "fingerprint": "ab" * 32,
            "marker_sets": {
                "markers": dict(list(markers.items())[:groups]),
                "left_out": ["Neurons"],
            },
        },
    )


# --------------------------------------------------------------------------
# Inputs


def test_layer_keys_follow_the_pipeline(script: ModuleType) -> None:
    """Shape keys of main.nf analysisLayerKeys; references of qc.nf."""
    assert script.layer_keys("MERSCOPE", "proseg_hybrid") == (
        "MOSAIK_proseg_hybrid",
        "merscope_cell_boundaries",
    )
    assert script.layer_keys("xenium", "reseg") == (
        "MOSAIK_proseg",
        "xenium_cell_boundaries",
    )
    main_nf = (REPO_ROOT / "workflows" / "main.nf").read_text()
    qc_nf = (REPO_ROOT / "workflows" / "modules" / "qc.nf").read_text()
    assert 'shape_key: "MOSAIK_proseg_hybrid"' in main_nf
    assert 'shape_key: "MOSAIK_proseg"' in main_nf
    assert '"merscope_cell_boundaries"' in qc_nf
    assert '"xenium_cell_boundaries"' in qc_nf
    with pytest.raises(script.RegressionError, match="platform"):
        script.layer_keys("COSMX", "reseg")
    with pytest.raises(script.RegressionError, match="segmentation"):
        script.layer_keys("XENIUM", "baysor")


def test_read_m8_run_takes_the_resolved_bundles(
    script: ModuleType, tmp_path: Path
) -> None:
    _write_m8_run(tmp_path, "HD4", "proseg_hybrid")
    run = script.read_m8_run(tmp_path, "HD4", "proseg_hybrid")
    assert run.map_dir == Path("/runs/HD4/proseg_hybrid")
    assert run.samples == {"HD4_MERSCOPE": "MERSCOPE", "HD4_XENIUM": "XENIUM"}
    assert run.n_segmented == {"HD4_MERSCOPE": 1000, "HD4_XENIUM": 1001}
    assert run.bundles == {
        "seaad_mr_panel": ("seaad_mr_panel", SEAAD_HASH),
        "whb_frontal_supc_clus": ("whb_frontal_supc_clus", WHB_HASH),
        "whb_frontal_supc_clus_setc": ("whb_frontal_supc_clus", SETC_HASH),
    }
    assert run.alignment_dir == Path("/results/HD4/alignment/align_out")
    assert run.summary_sha256 == "e" * 64


def test_read_m8_run_refuses_inconsistent_or_foreign_runs(
    script: ModuleType, tmp_path: Path
) -> None:
    other = _bundles()
    other["whb_frontal_supc_clus"] = {
        **other["whb_frontal_supc_clus"],
        "resolved_build_hash": "0" * 64,
    }
    _write_m8_run(
        tmp_path,
        "HD1",
        "reseg",
        bundles={"HD1_MERSCOPE": _bundles(), "HD1_XENIUM": other},
    )
    with pytest.raises(script.RegressionError, match="different samples"):
        script.read_m8_run(tmp_path, "HD1", "reseg")
    directory = _write_m8_run(tmp_path, "HD2", "reseg")
    summary = json.loads((directory / "HD2_resolve_summary.json").read_text())
    (directory / "HD2_resolve_summary.json").write_text(
        json.dumps({**summary, "segmentation": "proseg_hybrid"})
    )
    with pytest.raises(script.RegressionError, match="not HD2 / reseg"):
        script.read_m8_run(tmp_path, "HD2", "reseg")
    (directory / "HD2_resolve_summary.json").write_text(
        json.dumps({**summary, "species": "mouse"})
    )
    with pytest.raises(script.RegressionError, match="not a human"):
        script.read_m8_run(tmp_path, "HD2", "reseg")
    with pytest.raises(script.RegressionError, match="does not exist"):
        script.read_m8_run(tmp_path, "HD3", "reseg")


def test_the_resolve_command_pins_m8_bundles_and_adds_registration(
    script: ModuleType, tmp_path: Path
) -> None:
    _write_m8_run(tmp_path / "m8", "HD4", "proseg_hybrid")
    run = script.read_m8_run(tmp_path / "m8", "HD4", "proseg_hybrid")
    store = Path("/store")
    registration = {
        "HD4_MERSCOPE": Path("/reg/hd4_merscope_registration_qc.json"),
        "HD4_XENIUM": Path("/reg/hd4_xenium_registration_qc.json"),
    }
    command = script.resolve_command(
        run,
        out_dir=Path("/out/qc/HD4/proseg_hybrid"),
        store=store,
        registration=registration,
        config_path=None,
        protected_roots=[Path("/m8"), Path("/results")],
        fallback=Path("/fallback.h5ad"),
        python="python",
    )
    text = " ".join(command)
    assert command[:4] == ["python", "-c", script.CLI_SNIPPET, "annotate-resolve"]
    assert "--current-bundles" not in command
    assert (
        f"--bundle whb_frontal_supc_clus=/store/whb_frontal_supc_clus/{WHB_HASH}"
        in text
    )
    assert (
        f"--bundle whb_frontal_supc_clus_setc=/store/whb_frontal_supc_clus/{SETC_HASH}"
        in text
    )
    assert f"--bundle seaad_mr_panel=/store/seaad_mr_panel/{SEAAD_HASH}" in text
    assert "--map-dir /runs/HD4/proseg_hybrid" in text
    assert "--n-segmented HD4_MERSCOPE=1000" in text
    assert "--n-segmented HD4_XENIUM=1001" in text
    assert "--alignment-dir /results/HD4/alignment/align_out" in text
    assert (
        "--registration-qc HD4_MERSCOPE=/reg/hd4_merscope_registration_qc.json" in text
    )
    assert "--gene-id-fallback-csv /fallback.h5ad" in text
    assert "--results-root /m8 --results-root /results" in text
    assert "--annotation-config" not in command
    with_config = script.resolve_command(
        run,
        out_dir=Path("/out"),
        store=store,
        registration={},
        config_path=Path("/cfg/qc_free.json"),
        protected_roots=[],
        fallback=None,
    )
    assert with_config[-2:] == ["--annotation-config", "/cfg/qc_free.json"]
    assert "--gene-id-fallback-csv" not in with_config
    assert "--registration-qc" not in with_config


def _changed(
    script: ModuleType, variant: str, base: str | None = "qc"
) -> dict[str, Any]:
    """The fields of a variant that differ from ``base`` (``None``: defaults)."""
    config = AnnotationConfig.model_validate(script.variant_config(variant))
    reference = (
        AnnotationConfig(species="human")
        if base is None
        else AnnotationConfig.model_validate(script.variant_config(base))
    )
    assert config.model_dump(exclude={"real_qc"}) == reference.model_dump(
        exclude={"real_qc"}
    )
    default = reference.real_qc.model_dump()
    return {
        key: value
        for key, value in config.real_qc.model_dump().items()
        if value != default[key]
    }


def test_variant_configs_are_annotation_configs(script: ModuleType) -> None:
    """Each variant differs from C17's configuration (qc) only as named."""
    # qc pins the settings C17 ran with; the rulings of 2026-10-07 moved the
    # defaults away from them (pre-registration §23.19).
    assert script.variant_config("qc") == {
        "species": "human",
        "real_qc": script.C17_REAL_QC,
    }
    # C2 moved G1's default, C3 (b) NR7's matching, and C1 (b) gave the
    # new-panel family the class comparator; C17 ran with none of them (the
    # node default is unchanged).
    assert _changed(script, "qc", base=None) == {
        "marker_referee_comparator_by_family": {},
        "registration_g1_effect": "warning",
        "gene_complexity_matching": "lower_edge",
    }
    assert AnnotationConfig.model_validate(script.variant_config("qc")) == C17_CONFIG
    assert _changed(script, "qc_free") == {"enabled": False}
    assert _changed(script, "class_comparator") == {
        "marker_referee_comparator": "class"
    }
    # Reported only: the human gate counted as merged (the mouse one stays
    # pending) and G1 at §8.8's effect; no threshold changes.
    assert _changed(script, "applied_after_gate") == {
        "seeded_families_warn_only_until_gate": {"human": "merged", "mouse": "pending"},
        "registration_g1_effect": "gate_failed",
    }
    applied = AnnotationConfig.model_validate(
        script.variant_config("applied_after_gate")
    ).real_qc
    assert applied.seeded_warn_only("human") is False
    assert applied.seeded_warn_only("mouse") is True
    assert set(script.VARIANTS) == {
        "qc",
        "qc_free",
        "class_comparator",
        "applied_after_gate",
    }
    with pytest.raises(script.RegressionError, match="unknown variant"):
        script.variant_config("strict")


def test_the_output_must_stay_out_of_the_read_only_trees(
    script: ModuleType, tmp_path: Path
) -> None:
    m8 = tmp_path / "m8" / "resolve"
    m8.mkdir(parents=True)
    script.check_out_dir(tmp_path / "m13" / "regression", [m8])
    with pytest.raises(script.RegressionError, match="overlaps"):
        script.check_out_dir(m8 / "HD4", [m8])
    with pytest.raises(script.RegressionError, match="overlaps"):
        script.check_out_dir(tmp_path / "m8", [m8])
    with pytest.raises(script.RegressionError, match="overlaps"):
        script.main(["table", "--out", str(m8 / "x"), "--m8-resolve", str(m8)])


def test_run_resolve_needs_the_registration_checks_and_keeps_finished_runs(
    script: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    m8 = tmp_path / "m8"
    _write_m8_run(m8, "HD4", "reseg")
    out = tmp_path / "out"

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("no RESOLVE may run")

    monkeypatch.setattr(script.subprocess, "run", refuse)
    kwargs: dict[str, Any] = {
        "pairs": ["HD4"],
        "segmentations": ["reseg"],
        "m8_root": m8,
        "store": tmp_path / "store",
        "protected_roots": [m8],
    }
    commands = script.run_resolve(
        out, variants=list(script.VARIANTS), dry_run=True, **kwargs
    )
    assert len(commands) == 4
    assert not out.exists()
    with pytest.raises(script.RegressionError, match="registration checks missing"):
        script.run_resolve(out, variants=["qc"], **kwargs)
    for sample_id in ("HD4_MERSCOPE", "HD4_XENIUM"):
        path = script.registration_path(out, "reseg", sample_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    for variant in script.VARIANTS:
        done = script.variant_dir(out, variant, "HD4", "reseg")
        done.mkdir(parents=True)
        (done / "HD4_resolve_summary.json").write_text("{}")
    # Finished re-runs are kept and left out of the commands to run.
    assert script.run_resolve(out, variants=list(script.VARIANTS), **kwargs) == []
    assert (
        script.run_resolve(out, variants=list(script.VARIANTS), dry_run=True, **kwargs)
        == []
    )
    written = json.loads((out / "configs" / "qc_free.json").read_text())
    assert written == script.variant_config("qc_free")
    applied = json.loads((out / "configs" / "applied_after_gate.json").read_text())
    assert applied == script.variant_config("applied_after_gate")
    # qc pins C17's settings, so it has a config file too.
    qc = json.loads((out / "configs" / "qc.json").read_text())
    assert qc == script.variant_config("qc")


def test_each_rerun_records_its_command_and_code(
    script: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A re-run's record names the code it ran with, and its exit status."""
    import subprocess

    m8 = tmp_path / "m8"
    _write_m8_run(m8, "HD4", "reseg")
    out = tmp_path / "out"
    for sample_id in ("HD4_MERSCOPE", "HD4_XENIUM"):
        path = script.registration_path(out, "reseg", sample_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    real_run = subprocess.run
    exit_codes = {"qc": 0, "qc_free": 3}

    def fake_run(command: Sequence[str], *args: Any, **kwargs: Any) -> Any:
        if command[0] == "git":
            return real_run(command, *args, **kwargs)
        target = Path(command[command.index("--out") + 1])
        variant = target.parent.parent.name
        return subprocess.CompletedProcess(command, exit_codes[variant])

    monkeypatch.setattr(script.subprocess, "run", fake_run)
    kwargs: dict[str, Any] = {
        "pairs": ["HD4"],
        "segmentations": ["reseg"],
        "m8_root": m8,
        "store": tmp_path / "store",
        "protected_roots": [m8],
    }
    (command,) = script.run_resolve(out, variants=["qc"], **kwargs)
    target = script.variant_dir(out, "qc", "HD4", "reseg")
    record = json.loads((target / "regression_rerun.json").read_text())
    assert record["command"] == command
    assert (record["variant"], record["returncode"]) == ("qc", 0)
    assert record["config"] == script.variant_config("qc")
    position = command.index("--annotation-config")
    assert command[position + 1] == str(out / "configs" / "qc.json")
    code = record["code"]
    assert (
        code["commit"]
        == real_run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    assert code["script"] == "scripts/acceptance/m13_real_qc_regression.py"
    assert isinstance(code["script_matches_head"], bool)
    assert isinstance(code["dirty"], bool)
    with pytest.raises(script.RegressionError, match="exit 3"):
        script.run_resolve(out, variants=["qc_free"], **kwargs)
    failed = json.loads(
        (
            script.variant_dir(out, "qc_free", "HD4", "reseg") / "regression_rerun.json"
        ).read_text()
    )
    assert failed["returncode"] == 3
    assert failed["config"] == script.variant_config("qc_free")


def test_dry_run_reads_no_store_runs_nothing_and_writes_nothing(
    script: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--dry-run`` covers every step: registration too, and the table."""
    m8 = tmp_path / "m8"
    _write_m8_run(m8, "HD4", "reseg")
    out = tmp_path / "out"

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("nothing may run under --dry-run")

    monkeypatch.setattr(script, "registration_record", refuse)
    monkeypatch.setattr(script.subprocess, "run", refuse)
    monkeypatch.setattr(script, "build_tables", refuse)
    common = [
        "--out",
        str(out),
        "--m8-resolve",
        str(m8),
        "--results",
        str(tmp_path / "results"),
        "--store",
        str(tmp_path / "store"),
        "--pairs",
        "HD4",
        "--segmentations",
        "reseg",
        "--dry-run",
    ]
    for step in ("registration", "resolve", "table", "all"):
        assert script.main([step, *common]) == 0
        assert not out.exists(), step
    printed = [line for line in capsys.readouterr().out.splitlines() if line]
    # The resolve commands, printed by "resolve" and "all" (one per variant).
    assert len(printed) == 2 * len(script.VARIANTS)
    assert all("annotate-resolve" in line for line in printed)
    planned = script.run_registration(
        out,
        pairs=["HD4"],
        segmentations=["reseg"],
        m8_root=m8,
        results=tmp_path / "results",
        dry_run=True,
    )
    assert planned == [
        script.registration_path(out, "reseg", "HD4_MERSCOPE"),
        script.registration_path(out, "reseg", "HD4_XENIUM"),
    ]
    assert not out.exists()


def test_registration_record_runs_the_qc_stage_check_on_the_store(
    script: ModuleType, tmp_path: Path
) -> None:
    """The QC stage's check on the store's shapes and primary points."""
    import geopandas as gpd
    from shapely.geometry import box

    rng = np.random.default_rng(0)
    centres = np.array(
        [(x, y) for x in range(10, 1200, 20) for y in range(10, 1200, 20)], float
    )
    zarr = tmp_path / "latest_spatialdata.zarr"
    (zarr / "points" / "transcripts").mkdir(parents=True)
    (zarr / "points" / "decoy").mkdir(parents=True)
    attrs = {"merxen_schema": {"primary_points": "transcripts"}}
    (zarr / "zarr.json").write_text(json.dumps({"attributes": attrs}))
    cells = gpd.GeoDataFrame(
        geometry=[box(x - 4, y - 4, x + 4, y + 4) for x, y in centres]
    )
    for key in ("MOSAIK_proseg_hybrid", "merscope_cell_boundaries"):
        (zarr / "shapes" / key).mkdir(parents=True)
        cells.to_parquet(zarr / "shapes" / key / "shapes.parquet")
    near = np.repeat(centres, 20, axis=0) + rng.normal(0, 1.5, (len(centres) * 20, 2))
    background = rng.uniform(0, 1200, (20000, 2))
    xy = np.vstack([near, background])
    pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1]}).to_parquet(
        zarr / "points" / "transcripts" / "points.parquet"
    )
    # A decoy element far away: only the primary points may be read.
    pd.DataFrame({"x": [5000.0], "y": [5000.0]}).to_parquet(
        zarr / "points" / "decoy" / "points.parquet"
    )
    record = script.registration_record(
        zarr,
        shape_key="MOSAIK_proseg_hybrid",
        reference_shape_key="merscope_cell_boundaries",
    )
    assert record["status"] == "pass"
    assert record["density_ratio"] > 2.0
    assert record["shift_um"] == pytest.approx(0.0)
    assert record["inputs"]["points_key"] == "transcripts"
    assert record["inputs"]["reference_shape_key"] == "merscope_cell_boundaries"
    signal = RegistrationSignal.from_file(
        _dump(tmp_path / "p_merscope_registration_qc.json", record)
    )
    assert signal.density_ratio == pytest.approx(record["density_ratio"])
    without_reference = script.registration_record(
        zarr, shape_key="MOSAIK_proseg_hybrid", reference_shape_key="missing"
    )
    assert without_reference["inputs"]["reference_shape_key"] is None
    assert without_reference["shift_um"] is None
    with pytest.raises(script.RegressionError, match="no shapes"):
        script.registration_record(
            zarr, shape_key="MOSAIK_proseg", reference_shape_key="missing"
        )


def _dump(path: Path, record: Mapping[str, Any]) -> Path:
    path.write_text(json.dumps(record, default=float))
    return path


# --------------------------------------------------------------------------
# Tables


def test_outcome_rows_record_the_effect_withheld_until_the_gate_merges(
    script: ModuleType, make_trust: MakeTrust
) -> None:
    """D20 (b): HD4's pair statistic warns now, withholds after gate H."""
    summary = _qc_summary({"HD4_MERSCOPE": _qc_result(make_trust)})
    rows = {
        row["check"]: row
        for row in script.outcome_rows(summary, pair="HD4", segmentation="reseg")
    }
    assert set(rows) >= {"marker_consistency", "registration_g1", "flag_rates"}
    paired = rows["paired_concordance"]
    assert (paired["outcome"], paired["effect"]) == ("warn", "warning")
    assert paired["effect_after_gate"] == "withhold_pair_stats"
    assert paired["warn_only"] is True
    assert paired["value"] == pytest.approx(0.234905)
    g1 = rows["registration_g1"]
    assert (g1["outcome"], g1["value"]) == ("warn", pytest.approx(1.98))
    assert g1["effect_after_gate"] == "warning"
    assert rows["marker_consistency"]["state"] == "not_evaluable"
    assert rows["marker_consistency"]["effect_after_gate"] == "none"
    assert rows["flag_rates"]["value"] == 0.0
    assert (rows["flag_rates"]["outcome"], rows["flag_rates"]["effect_after_gate"]) == (
        "pass",
        "none",
    )
    disabled = _qc_summary({"HD4_MERSCOPE": None})
    with pytest.raises(script.RegressionError, match="QC not enabled"):
        script.outcome_rows(disabled, pair="HD4", segmentation="reseg")


def test_gate_cap_after_gate_is_set_only_by_a_fired_outcome(
    script: ModuleType, make_trust: MakeTrust
) -> None:
    """An outcome that did not fire caps nothing, whatever cap it records.

    The referee records ``gate_cap: broad_only`` also when it passes or is
    not evaluable (set a: not evaluable on all 16 sections); the column must
    not read as a cap after gate H merges.
    """

    def marker_row(value: float | None, groups: int, **change: Any) -> Any:
        summary = _qc_summary(
            {
                "HD4_MERSCOPE": _qc_result(
                    make_trust,
                    marker_consistency=_referee_signal(value, groups),
                    **change,
                )
            }
        )
        rows = script.outcome_rows(summary, pair="HD4", segmentation="reseg")
        return next(row for row in rows if row["check"] == "marker_consistency")

    not_evaluable = marker_row(None, 1)
    assert not_evaluable["state"] == "not_evaluable"
    assert not_evaluable["effect_after_gate"] == "none"
    assert not_evaluable["gate_cap_after_gate"] is None
    passing = marker_row(0.9, 2)
    assert (passing["outcome"], passing["fired"]) == ("pass", False)
    assert passing["gate_cap_after_gate"] is None
    warned = marker_row(0.72, 2)
    assert (warned["outcome"], warned["effect_after_gate"]) == ("warn", "warning")
    assert warned["gate_cap_after_gate"] is None
    # Below 0.70 on the seeded family: a warning now, broad_only after gate H.
    withheld = marker_row(0.6, 2)
    assert (withheld["outcome"], withheld["effect"]) == ("warn", "warning")
    assert withheld["effect_after_gate"] == "gate_cap"
    assert withheld["gate_cap_after_gate"] == "broad_only"
    # With the gate merged the cap is applied and read from the outcome.
    merged = AnnotationConfig.model_validate(
        script.variant_config("applied_after_gate")
    )
    applied = marker_row(0.6, 2, config=merged)
    assert (applied["outcome"], applied["effect"]) == ("fail", "gate_cap")
    assert applied["gate_cap_after_gate"] == "broad_only"
    # A pair statistic withheld after the gate carries no gate cap.
    summary = _qc_summary({"HD4_MERSCOPE": _qc_result(make_trust)})
    paired = next(
        row
        for row in script.outcome_rows(summary, pair="HD4", segmentation="reseg")
        if row["check"] == "paired_concordance"
    )
    assert paired["effect_after_gate"] == "withhold_pair_stats"
    assert paired["gate_cap_after_gate"] is None


def test_marker_rows_and_the_d18_reading(
    script: ModuleType, make_trust: MakeTrust
) -> None:
    """Only the registered node comparator decides; class is reported."""
    node = _qc_summary(
        {
            "HD4_MERSCOPE": _qc_result(
                make_trust, marker_consistency=_referee_signal(0.74, 2)
            ),
            "HD4_XENIUM": _qc_result(
                make_trust, marker_consistency=_referee_signal(None, 1)
            ),
        }
    )
    alternative = _qc_summary(
        {
            "HD4_MERSCOPE": _qc_result(
                make_trust, marker_consistency=_referee_signal(0.91, 2)
            ),
            "HD4_XENIUM": _qc_result(
                make_trust, marker_consistency=_referee_signal(0.88, 2)
            ),
        }
    )
    rows = script.marker_rows(node, alternative, pair="HD4", segmentation="reseg")
    first, second = rows
    assert first["node_consistency"] == pytest.approx(0.74)
    assert first["node_outcome"] == "warn"
    assert first["node_groups"] == "Microglia:3;Astrocytes:3"
    assert first["node_markers"].startswith("Microglia: CSF1R,C1QB,P2RY12")
    assert first["node_left_out"] == "Neurons"
    assert first["class_consistency"] == pytest.approx(0.91)
    assert second["node_consistency"] is None
    assert second["node_outcome"] == "not_evaluable"
    assert second["node_reason"] == "1 marker group(s)"
    verdict = script.d18_verdict(rows, warn=0.75)
    assert verdict["verdict"] == "below_warn"
    assert verdict["thresholds_back_to_user"] is True
    assert verdict["class_comparator_reported"]["verdict"] == "at_or_above_warn"

    def reading(node_values: Sequence[float | None], **kwargs: Any) -> Any:
        rows = [{"node_consistency": value, **kwargs} for value in node_values]
        return script.d18_verdict(rows, warn=0.75)

    assert reading([None, None])["verdict"] == "not_evaluable"
    assert reading([None, None])["user_decision_needed"] is True
    assert reading([None, None])["thresholds_back_to_user"] is False
    assert reading([0.8, None])["verdict"] == "partly_not_evaluable"
    done = reading([0.8, 0.75], class_consistency=0.5)
    assert done["verdict"] == "at_or_above_warn"
    assert done["user_decision_needed"] is False
    assert done["class_comparator_reported"]["verdict"] == "below_warn"
    assert reading([])["verdict"] == "not_evaluable"


def test_paired_rows_check_the_d21_expectation(
    script: ModuleType, make_trust: MakeTrust
) -> None:
    def rows(
        pair: str,
        segmentation: str,
        jsd: Sequence[Mapping[str, Any]],
        config: AnnotationConfig | None = None,
    ) -> Any:
        summary = _qc_summary(
            {
                f"{pair}_MERSCOPE": _qc_result(make_trust, pair_jsd=jsd, config=config),
                f"{pair}_XENIUM": _qc_result(make_trust, pair_jsd=jsd, config=config),
            },
            pair=pair,
            jsd=jsd,
        )
        (row,) = script.paired_rows(summary, pair=pair, segmentation=segmentation)
        return row

    hd4 = rows("HD4", "proseg_hybrid", HD4_JSD)
    assert hd4["shared_mask_jsd"] == pytest.approx(0.234905)
    assert hd4["whole_section_jsd"] == pytest.approx(0.226814)
    assert (hd4["outcome"], hd4["effect_after_gate"]) == (
        "warn",
        "withhold_pair_stats",
    )
    assert hd4["expectation_met"] is True
    # HD4's own values with a pass token (the outcome changed while the
    # values did not, e.g. under another warning threshold): not met.
    lenient = AnnotationConfig(species="human", real_qc={"paired_broad_jsd_warn": 0.3})
    hd4_pass = rows("HD4", "proseg_hybrid", HD4_JSD, lenient)
    assert hd4_pass["shared_mask_jsd"] == pytest.approx(0.234905)
    assert hd4_pass["whole_section_jsd"] == pytest.approx(0.226814)
    assert hd4_pass["outcome"] == "pass"
    assert hd4_pass["expectation_met"] is False
    low = [
        {"kind": "soft", "region": "shared_mask", "jsd": 0.1277},
        {"kind": "soft", "region": "whole_section", "jsd": 0.1291},
    ]
    passing = rows("HD1", "proseg_hybrid", low)
    assert (passing["outcome"], passing["effect_after_gate"]) == ("pass", "none")
    assert passing["expectation_met"] is True
    # HD4 passing, or another pair above .154 (still under .20), breaks it.
    assert rows("HD4", "proseg_hybrid", low)["expectation_met"] is False
    above = [{"kind": "soft", "region": "shared_mask", "jsd": 0.17}]
    assert rows("HD2", "proseg_hybrid", above)["outcome"] == "pass"
    assert rows("HD2", "proseg_hybrid", above)["expectation_met"] is False
    # The whole section is reported, never scored: no shared mask, no verdict.
    whole_only = [{"kind": "soft", "region": "whole_section", "jsd": 0.30}]
    row = rows("HD3", "proseg_hybrid", whole_only)
    assert row["outcome"] == "not_evaluable"
    assert row["expectation_met"] is False
    reseg = rows("HD4", "reseg", HD4_JSD)
    assert reseg["expected"] is None
    assert reseg["expectation_met"] is None


def test_flag_rate_rows_report_both_d22_readings(
    script: ModuleType, make_trust: MakeTrust
) -> None:
    """The literal reading warns where option (a) does not (and is scored)."""
    noisy = [
        stratum(
            "contaminated",
            cls,
            0.2 if index < 4 else 0.01,
            informative=True,
        )
        for index, cls in enumerate(CLASSES)
    ]
    strata = noisy + [
        stratum(flag, cls, 0.01)
        for flag in ("diffuse_profile", "ood")
        for cls in CLASSES
    ]
    summary = _qc_summary(
        {
            "HD2_MERSCOPE": _qc_result(make_trust, flag_strata=strata),
            "HD2_XENIUM": _qc_result(make_trust),
        },
        pair="HD2",
    )
    rows = script.flag_rate_rows(summary, pair="HD2", segmentation="reseg")
    merscope = rows[0]
    assert merscope["literal_n_strata"] == 7
    assert merscope["literal_n_uninformative"] == 4
    assert merscope["literal_warn"] is True
    assert merscope["option_a_n_strata"] == 21
    assert merscope["option_a_n_uninformative"] == 0
    assert merscope["option_a_would_warn"] is False
    assert set(merscope["literal_uninformative_strata"].split(";")) == {
        f"{cls}/MERSCOPE" for cls in CLASSES[:4]
    }
    totals = script.d22_summary(rows)
    assert (totals["literal_warnings"], totals["option_a_warnings"]) == (1, 0)
    assert totals["literal_max_share"] == pytest.approx(4 / 7)
    outcome = next(
        row
        for row in script.outcome_rows(summary, pair="HD2", segmentation="reseg")
        if row["check"] == "flag_rates" and row["sample_id"] == "HD2_MERSCOPE"
    )
    assert outcome["outcome"] == "warn"


def test_g1_rows_and_the_d23_reading(script: ModuleType, make_trust: MakeTrust) -> None:
    """A fail rule on set a is a false failure; a missing check decides nothing."""
    summary = _qc_summary(
        {
            "HD3_MERSCOPE": _qc_result(
                make_trust, registration=RegistrationSignal(1.4, 0.0)
            ),
            "HD3_XENIUM": _qc_result(
                make_trust, registration=RegistrationSignal(2.4, 6.0)
            ),
        },
        pair="HD3",
    )
    rows = script.g1_rows(summary, pair="HD3", segmentation="proseg_hybrid")
    assert [row["rule"] for row in rows] == ["fail", "fail"]
    assert rows[0]["density_ratio"] == pytest.approx(1.4)
    assert rows[1]["shift_um"] == pytest.approx(6.0)
    assert rows[0]["effect_setting"] == "warning"
    assert rows[0]["outcome"] == "warn"
    verdict = script.d23_verdict(rows)
    assert verdict["verdict"] == "false_failure"
    assert verdict["fail_rule"] == [
        "HD3_MERSCOPE/proseg_hybrid",
        "HD3_XENIUM/proseg_hybrid",
    ]
    good = _qc_summary(
        {
            "HD3_MERSCOPE": _qc_result(make_trust),
            "HD3_XENIUM": _qc_result(
                make_trust, registration=RegistrationSignal(2.5, 0.0)
            ),
        },
        pair="HD3",
    )
    good_rows = script.g1_rows(good, pair="HD3", segmentation="proseg_hybrid")
    assert [row["rule"] for row in good_rows] == ["warn", None]
    assert script.d23_verdict(good_rows)["verdict"] == "no_false_failure"
    assert script.d23_verdict(good_rows)["warn_rule"] == ["HD3_MERSCOPE/proseg_hybrid"]
    missing = _qc_summary(
        {"HD3_MERSCOPE": _qc_result(make_trust, registration=None)}, pair="HD3"
    )
    missing_rows = script.g1_rows(missing, pair="HD3", segmentation="reseg")
    assert missing_rows[0]["state"] == "not_evaluable"
    assert script.d23_verdict(good_rows + missing_rows)["verdict"] == "incomplete"
    assert script.d23_verdict([])["verdict"] == "incomplete"


def test_label_identity_allows_only_the_null_version_7_column(
    script: ModuleType,
) -> None:
    old = _labels("S", ["confident", "low_confidence", "confident"])
    old["ct_broad_margin"] = [0.5, np.nan, 0.2]
    new = old.copy()
    new["flag_nonneuronal_high_depth"] = pd.array([None] * 3, dtype="boolean")
    assert script.label_identity(new, old) == []
    flagged = new.copy()
    flagged["flag_nonneuronal_high_depth"] = pd.array(
        [None, True, None], dtype="boolean"
    )
    assert script.label_identity(flagged, old) == [
        "new column flag_nonneuronal_high_depth is not null throughout"
    ]
    changed = new.copy()
    changed.loc[2, "ct_broad_margin"] = 0.3
    assert script.label_identity(changed, old) == [
        "column ct_broad_margin differs (1 rows)"
    ]
    extra = new.assign(other=1).drop(columns=["ct_broad_name"])
    assert script.label_identity(extra, old) == [
        "columns missing: ['ct_broad_name']",
        "new column other",
    ]
    assert script.label_identity(new.iloc[:2], old) == ["2 rows, M8 3"]


def _write_rerun(
    directory: Path,
    summary: Mapping[str, Any],
    statuses: Mapping[str, Sequence[str]],
    *,
    extra_null_column: bool = True,
) -> None:
    directory.mkdir(parents=True)
    pair = str(summary["pair_id"])
    for sample_id, item in summary["samples"].items():
        labels = _labels(sample_id, statuses[sample_id])
        if extra_null_column:
            labels["flag_nonneuronal_high_depth"] = pd.array(
                [None] * len(labels), dtype="boolean"
            )
        path = directory / item["labels"]
        path.parent.mkdir(parents=True, exist_ok=True)
        labels.to_parquet(path)
        (directory / item["annotation_manifest"]).write_text(
            json.dumps(
                {
                    "panel": {"panel_trust": "validated"},
                    "resolvability": {},
                    "thresholds": {"floors_sha256": None},
                    "gate": {"level": "full", "reasons": []},
                }
            )
        )
    (directory / f"{pair}_resolve_summary.json").write_text(
        json.dumps(summary, default=str)
    )


def test_build_tables_writes_the_outcome_table_and_the_readings(
    script: ModuleType, tmp_path: Path, make_trust: MakeTrust
) -> None:
    m8 = tmp_path / "m8"
    out = tmp_path / "out"
    _write_m8_run(m8, "HD4", "proseg_hybrid")
    statuses = {
        "HD4_MERSCOPE": ["confident", "low_confidence", "confident"],
        "HD4_XENIUM": ["confident", "low_confidence", "confident"],
    }
    results = {
        "HD4_MERSCOPE": _qc_result(
            make_trust, marker_consistency=_referee_signal(None, 1)
        ),
        "HD4_XENIUM": _qc_result(
            make_trust,
            marker_consistency=_referee_signal(None, 1),
            registration=RegistrationSignal(2.6, 0.0),
        ),
    }
    with pytest.raises(script.RegressionError, match="no qc re-run"):
        script.build_tables(
            out, pairs=["HD4"], segmentations=["proseg_hybrid"], m8_root=m8
        )
    qc_summary = _qc_summary(results)
    _write_rerun(
        script.variant_dir(out, "qc", "HD4", "proseg_hybrid"), qc_summary, statuses
    )
    free_summary = _qc_summary({sample: None for sample in results})
    _write_rerun(
        script.variant_dir(out, "qc_free", "HD4", "proseg_hybrid"),
        free_summary,
        statuses,
    )
    alternative = _qc_summary(
        {
            sample: _qc_result(make_trust, marker_consistency=_referee_signal(0.9, 2))
            for sample in results
        }
    )
    _write_rerun(
        script.variant_dir(out, "class_comparator", "HD4", "proseg_hybrid"),
        alternative,
        statuses,
    )
    result = script.build_tables(
        out, pairs=["HD4"], segmentations=["proseg_hybrid"], m8_root=m8
    )
    for name in (
        "outcomes",
        "samples",
        "marker_consistency",
        "paired_concordance",
        "flag_rates",
        "registration_g1",
        "nr1",
    ):
        assert (out / f"{name}.csv").exists(), name
    outcomes = pd.read_csv(out / "outcomes.csv")
    assert set(outcomes["sample_id"]) == {"HD4_MERSCOPE", "HD4_XENIUM"}
    assert set(outcomes["check"]) >= {"marker_consistency", "paired_concordance"}
    d18 = result["d18_marker_consistency"]["scored_segmentation"]
    assert d18["verdict"] == "not_evaluable"
    assert d18["user_decision_needed"] is True
    assert d18["class_comparator_reported"]["verdict"] == "at_or_above_warn"
    assert result["d21_paired_concordance"]["expectation_met"] is True
    assert result["d22_flag_rates"]["literal_warnings"] == 0
    assert result["d23_registration_g1"]["verdict"] == "no_false_failure"
    assert result["d23_registration_g1"]["warn_rule"] == ["HD4_MERSCOPE/proseg_hybrid"]
    assert result["d20_warn_only"]["warn_only_samples"] == 2
    assert result["d20_warn_only"]["withheld_effects"] == [
        "HD4_MERSCOPE/proseg_hybrid:paired_concordance:withhold_pair_stats",
        "HD4_XENIUM/proseg_hybrid:paired_concordance:withhold_pair_stats",
    ]
    # Warn-only: the qc variant lowers nothing, so NR1 holds trivially (and
    # the report says so); the applied variant was not run here.
    assert result["nr1"] == {
        "violations": [],
        "lowered_samples": [],
        "trivially_satisfied": True,
        "applied_after_gate": {
            "run": False,
            "violations": [],
            "lowered_samples": [],
            "trivially_satisfied": True,
        },
        "qc_identical_to_m8": True,
        "qc_free_identical_to_m8": True,
    }
    markers = outcomes[outcomes["check"] == "marker_consistency"]
    assert set(markers["state"]) == {"not_evaluable"}
    assert markers["gate_cap_after_gate"].isna().all()
    samples = pd.read_csv(out / "samples.csv")
    assert list(samples["gate_level_qc_free"]) == ["full", "full"]
    assert "check:paired_concordance" in samples.columns
    summary = json.loads((out / "regression_summary.json").read_text())
    assert summary["inputs"][0]["variants_run"] == [
        "class_comparator",
        "qc",
        "qc_free",
    ]
    report = (out / "REPORT.txt").read_text()
    for heading in ("D18", "D21", "D22", "D23", "NR1", "Not covered"):
        assert heading in report
    assert "NR1 violations: 0 (trivially satisfied" in report
    assert "applied_after_gate (reported only" in report
    assert "not run on every sample" in report


def test_build_tables_reports_an_nr1_violation_and_a_drift_from_m8(
    script: ModuleType, tmp_path: Path, make_trust: MakeTrust
) -> None:
    """A cell confident only with QC breaks NR1; a changed label breaks M8 identity."""
    m8 = tmp_path / "m8"
    out = tmp_path / "out"
    _write_m8_run(m8, "HD1", "reseg", jsd=[])
    base = ["confident", "low_confidence", "confident"]
    raised = ["confident", "confident", "confident"]
    results = {
        sample: _qc_result(make_trust, pair_jsd=[])
        for sample in ("HD1_MERSCOPE", "HD1_XENIUM")
    }
    _write_rerun(
        script.variant_dir(out, "qc", "HD1", "reseg"),
        _qc_summary(results, pair="HD1", jsd=[]),
        {"HD1_MERSCOPE": raised, "HD1_XENIUM": base},
    )
    _write_rerun(
        script.variant_dir(out, "qc_free", "HD1", "reseg"),
        _qc_summary(dict.fromkeys(results), pair="HD1", jsd=[]),
        {"HD1_MERSCOPE": base, "HD1_XENIUM": base},
    )
    result = script.build_tables(
        out, pairs=["HD1"], segmentations=["reseg"], m8_root=m8
    )
    nr1 = pd.read_csv(out / "nr1.csv").set_index("sample_id")
    assert "confident only with QC" in nr1.loc["HD1_MERSCOPE", "nr1_violations"]
    assert nr1.loc["HD1_XENIUM", "nr1_violations"] == "none"
    assert nr1.loc["HD1_MERSCOPE", "qc_vs_m8"].startswith("column ct_broad_status")
    assert nr1.loc["HD1_MERSCOPE", "qc_free_vs_m8"] == "identical"
    assert result["nr1"]["qc_identical_to_m8"] is False
    assert len(result["nr1"]["violations"]) == 1
    # Without the class-comparator re-run the alternative is simply absent.
    markers = pd.read_csv(out / "marker_consistency.csv")
    assert markers["class_outcome"].isna().all()
    paired = pd.read_csv(out / "paired_concordance.csv")
    assert paired.loc[0, "expected"] is np.nan or pd.isna(paired.loc[0, "expected"])


def test_build_tables_scores_d18_on_the_scored_segmentation_only(
    script: ModuleType, tmp_path: Path, make_trust: MakeTrust
) -> None:
    """A reseg value below 0.75 is reported, never read by the scored verdict."""
    m8 = tmp_path / "m8"
    out = tmp_path / "out"
    statuses = {
        sample: ["confident", "low_confidence", "confident"]
        for sample in ("HD4_MERSCOPE", "HD4_XENIUM")
    }
    for segmentation, value, groups in (
        ("proseg_hybrid", None, 1),
        ("reseg", 0.6, 2),
    ):
        _write_m8_run(m8, "HD4", segmentation)
        _write_rerun(
            script.variant_dir(out, "qc", "HD4", segmentation),
            _qc_summary(
                {
                    sample: _qc_result(
                        make_trust,
                        marker_consistency=_referee_signal(value, groups),
                    )
                    for sample in statuses
                }
            ),
            statuses,
        )
    result = script.build_tables(
        out,
        pairs=["HD4"],
        segmentations=["proseg_hybrid", "reseg"],
        m8_root=m8,
    )
    readings = result["d18_marker_consistency"]
    assert readings["scored_segmentation"]["verdict"] == "not_evaluable"
    assert readings["scored_segmentation"]["n_sections"] == 2
    assert readings["scored_segmentation"]["thresholds_back_to_user"] is False
    assert readings["all_segmentations"]["verdict"] == "below_warn"
    assert readings["all_segmentations"]["n_below_warn"] == 2
    outcomes = pd.read_csv(out / "outcomes.csv")
    reseg = outcomes[
        (outcomes["check"] == "marker_consistency")
        & (outcomes["segmentation"] == "reseg")
    ]
    assert set(reseg["effect_after_gate"]) == {"gate_cap"}
    assert set(reseg["gate_cap_after_gate"]) == {"broad_only"}


def test_build_tables_checks_nr1_where_the_applied_variant_lowers(
    script: ModuleType, tmp_path: Path, make_trust: MakeTrust
) -> None:
    """applied_after_gate applies the withheld effects; NR1 is checked there."""
    m8 = tmp_path / "m8"
    out = tmp_path / "out"
    _write_m8_run(m8, "HD4", "proseg_hybrid")
    base = ["confident", "low_confidence", "confident"]
    raised = ["confident", "confident", "confident"]
    samples = ("HD4_MERSCOPE", "HD4_XENIUM")
    statuses = dict.fromkeys(samples, base)
    merged = AnnotationConfig.model_validate(
        script.variant_config("applied_after_gate")
    )
    _write_rerun(
        script.variant_dir(out, "qc", "HD4", "proseg_hybrid"),
        _qc_summary({sample: _qc_result(make_trust) for sample in samples}),
        statuses,
    )
    _write_rerun(
        script.variant_dir(out, "qc_free", "HD4", "proseg_hybrid"),
        _qc_summary(dict.fromkeys(samples)),
        statuses,
    )
    # The MERSCOPE section's marker referee is below 0.70: broad_only applied.
    applied = {
        "HD4_MERSCOPE": _qc_result(
            make_trust,
            config=merged,
            marker_consistency=_referee_signal(0.6, 2),
        ),
        "HD4_XENIUM": _qc_result(make_trust, config=merged),
    }
    _write_rerun(
        script.variant_dir(out, "applied_after_gate", "HD4", "proseg_hybrid"),
        _qc_summary(applied),
        {"HD4_MERSCOPE": base, "HD4_XENIUM": raised},
    )
    result = script.build_tables(
        out, pairs=["HD4"], segmentations=["proseg_hybrid"], m8_root=m8
    )
    nr1 = result["nr1"]
    assert nr1["trivially_satisfied"] is True
    assert nr1["violations"] == []
    reading = nr1["applied_after_gate"]
    assert reading["run"] is True
    assert reading["trivially_satisfied"] is False
    assert reading["lowered_samples"] == [
        "HD4_MERSCOPE/proseg_hybrid: broad_only:marker_consistency; "
        "withhold_pair_stats:paired_concordance",
        "HD4_XENIUM/proseg_hybrid: withhold_pair_stats:paired_concordance",
    ]
    # The cell confident only with the QC applied breaks NR1 there.
    (violation,) = reading["violations"]
    assert violation.startswith("HD4_XENIUM/proseg_hybrid: ")
    assert "confident only with QC" in violation
    table = pd.read_csv(out / "nr1.csv").set_index("sample_id")
    assert table.loc["HD4_MERSCOPE", "nr1_applied_after_gate_violations"] == "none"
    assert table.loc["HD4_MERSCOPE", "qc_lowerings"] == "none"
    samples_table = pd.read_csv(out / "samples.csv").set_index("sample_id")
    assert samples_table.loc["HD4_MERSCOPE", "gate_level_qc"] == "full"
    assert (
        samples_table.loc["HD4_MERSCOPE", "gate_level_applied_after_gate"]
        == "broad_only"
    )
    report = (out / "REPORT.txt").read_text()
    assert "NR1 violations 1 over 2 lowered samples" in report


# --------------------------------------------------------------------------
# Mouse rows (M13b C22): pending until GM-RUNS, read from its RESOLVE outputs


def _mouse_summary(
    make_trust: MakeTrust,
    *,
    segmentation: str = "proseg_hybrid",
    pair: str = "ms1",
    gate: Any = None,
    enabled: bool = True,
) -> dict[str, Any]:
    """A mouse resolve summary with the QC block mouse RESOLVE writes."""
    from merxen.annotation.pipeline import real_qc_record

    from .test_real_qc_checks import mouse_signals
    from .test_real_qc_mouse import g4_gate

    verdict = g4_gate() if gate is None else gate
    result = qc.mouse_real_data_qc(
        mouse_signals(gate=verdict),
        make_trust("validated_real", species="mouse"),
        AnnotationConfig(species="mouse"),
    )
    sample_id = f"{pair}_MERSCOPE"
    return {
        "pair_id": pair,
        "species": "mouse",
        "segmentation": segmentation,
        "samples": {
            sample_id: {
                "sample_id": sample_id,
                "platform": "MERSCOPE",
                "trust": {"state": "validated"},
                "resolution": {"gate": verdict.to_json()},
                "mouse_gate": verdict.to_json(),
                "real_qc": real_qc_record(result if enabled else None, enabled=enabled),
            }
        },
        "pair": {"jsd": [], "mask_note": None},
    }


def _write_mouse_run(directory: Path, summary: Mapping[str, Any]) -> Path:
    directory.mkdir(parents=True)
    path = directory / f"{summary['pair_id']}_resolve_summary.json"
    path.write_text(json.dumps(summary, default=str))
    return directory


def test_the_mouse_rows_stay_pending_in_the_human_table(script: ModuleType) -> None:
    assert script.MOUSE_EXPECTED == (
        ("MS1", "original_seg"),
        ("MS1", "proseg_hybrid"),
        ("MS1", "proseg_mask"),
        ("MS1", "reseg"),
        ("MS2", "original_seg"),
    )
    assert any("pending" in item and "mouse" in item for item in script.NOT_COVERED)


def test_parse_mouse_input_takes_dataset_segmentation_and_directory(
    script: ModuleType, tmp_path: Path
) -> None:
    item = script.parse_mouse_input(f"MS1/proseg_hybrid={tmp_path}")
    assert (item.dataset, item.segmentation, item.resolve_dir) == (
        "MS1",
        "proseg_hybrid",
        tmp_path,
    )
    for bad in ("MS1=/x", "MS1/proseg_hybrid", "/proseg_hybrid=/x", "MS1/=/x"):
        with pytest.raises(script.RegressionError, match="DATASET/SEGMENTATION"):
            script.parse_mouse_input(bad)


def test_pairs_and_mouse_labels_follow_the_alias_table(
    script: ModuleType, tmp_path: Path, fake_aliases: Path
) -> None:
    """Aliases in, the names the data uses out (R2 then R3)."""
    from test_privacy.synthetic import DONOR, MOUSE, MOUSE2_NAME

    assert script.pair_name("HD1") == DONOR
    assert script.pair_name(DONOR) == DONOR
    assert script.mouse_label("MS1") == MOUSE
    assert script.mouse_label(MOUSE) == MOUSE
    assert script.mouse_label("ms2") == MOUSE2_NAME
    assert script.mouse_label("other") == "other"
    item = script.parse_mouse_input(f"MS2/original_seg={tmp_path}")
    assert item.label == f"{MOUSE2_NAME}/original_seg"
    # D21's expected warning is keyed by alias and found by the real pair id.
    assert privacy.to_alias(DONOR) == "HD1"


def test_pairs_need_the_alias_table(script: ModuleType, no_alias_table: None) -> None:
    from merxen.local_config import LocalConfigError

    with pytest.raises(LocalConfigError, match="no alias table"):
        script.pair_name("HD1")


def test_the_locations_come_from_the_environment(
    script: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A step resolves the locations it uses; unset ones it needs refuse."""
    with pytest.raises(SystemExit):
        script.main(["table", "--dry-run"])
    evidence = tmp_path / "evidence"
    monkeypatch.setenv("MERXEN_EVIDENCE_ROOT", str(evidence))
    monkeypatch.setenv("MERXEN_RESULTS_ROOT", str(tmp_path / "results"))
    local_config.clear_cache()
    assert script.main(["table", "--dry-run"]) == 0
    # resolve needs the store (the legacy one C17 ran with).
    with pytest.raises(SystemExit):
        script.main(["resolve", "--dry-run"])


def test_mouse_rows_read_a_mouse_resolve_output(
    script: ModuleType, tmp_path: Path, make_trust: MakeTrust
) -> None:
    run = _write_mouse_run(tmp_path / "ms1_ph", _mouse_summary(make_trust))
    item = script.MouseInput("MS1", "proseg_hybrid", run)
    summary = script.read_mouse_summary(item)
    samples = script.mouse_sample_rows(
        summary, dataset="MS1", segmentation="proseg_hybrid"
    )
    (row,) = samples
    assert row["dataset"] == "MS1" and row["sample_id"] == "ms1_MERSCOPE"
    assert row["warn_only"] is True and row["effects_withheld"] is True
    assert row["no_effect_problems"] == "none"
    assert row["check:g4_composition"] == "warn"
    assert row["check:paired_concordance"] == "not_applicable"
    assert row["ap_estimate_section"] == "C57BL6J-638850.33"
    assert row["ap_estimate_mm"] == 8.0
    assert row["g4_window_source"] == "ap_estimate"
    assert row["g4_window_sections"] == (
        "C57BL6J-638850.32;C57BL6J-638850.33;C57BL6J-638850.34"
    )
    signals = script.mouse_signal_rows(
        summary, dataset="MS1", segmentation="proseg_hybrid"
    )
    assert [item["signal"] for item in signals] == ["g1", "g2", "g3", "g4", "g5"]
    g4 = signals[3]
    assert (g4["check"], g4["status"], g4["outcome"]) == (
        "g4_composition",
        "warn",
        "warn",
    )
    assert json.loads(g4["values"]) == {
        "g4_astro_epen_points": -6.5,
        "g4_immune_points": 0.4,
    }
    outcomes = script.mouse_outcome_rows(
        summary, dataset="MS1", segmentation="proseg_hybrid"
    )
    assert {row["dataset"] for row in outcomes} == {"MS1"}
    assert {row["check"] for row in outcomes} >= {"flag_rates", "factor_remeasure"}


def test_mouse_rows_refuse_outputs_without_the_mouse_qc(
    script: ModuleType, tmp_path: Path, make_trust: MakeTrust
) -> None:
    old = _mouse_summary(make_trust)
    for item in old["samples"].values():
        item.pop("real_qc")
    cases = {
        "predates M13b C22": old,
        "QC not enabled": _mouse_summary(make_trust, enabled=False),
        "not a mouse": {**_mouse_summary(make_trust), "species": "human"},
        "segmentation reseg": _mouse_summary(make_trust, segmentation="reseg"),
    }
    for index, (match, summary) in enumerate(cases.items()):
        run = _write_mouse_run(tmp_path / f"run{index}", summary)
        with pytest.raises(script.RegressionError, match=match):
            script.read_mouse_summary(script.MouseInput("MS1", "proseg_hybrid", run))
    with pytest.raises(script.RegressionError, match="no .*_resolve_summary.json"):
        empty = tmp_path / "empty"
        empty.mkdir()
        script.read_mouse_summary(script.MouseInput("MS1", "reseg", empty))


def test_a_mouse_row_with_a_qc_effect_is_reported(
    script: ModuleType, tmp_path: Path, make_trust: MakeTrust
) -> None:
    """Warn-only is checked: an applied effect or a real_qc_ gate reason."""
    summary = _mouse_summary(make_trust)
    item = summary["samples"]["ms1_MERSCOPE"]
    item["real_qc"]["downgrades"] = ["not_resolvable_class:prefilter_spotcheck"]
    item["mouse_gate"]["warning_reasons"].append("real_qc_flag_rates: x")
    (row,) = script.mouse_sample_rows(
        summary, dataset="MS1", segmentation="proseg_hybrid"
    )
    assert "downgrades" in row["no_effect_problems"]
    assert "real_qc_flag_rates" in row["no_effect_problems"]


def test_build_mouse_tables_lists_the_pending_rows(
    script: ModuleType, tmp_path: Path, make_trust: MakeTrust
) -> None:
    out = tmp_path / "out"
    with pytest.raises(script.RegressionError, match="pending"):
        script.build_mouse_tables(out, [])
    assert not out.exists()
    inputs = [
        script.MouseInput(
            "MS1",
            "proseg_hybrid",
            _write_mouse_run(tmp_path / "ms1_ph", _mouse_summary(make_trust)),
        ),
        script.MouseInput(
            "MS2",
            "proseg_hybrid",
            _write_mouse_run(
                tmp_path / "ms2_ph",
                _mouse_summary(make_trust, pair="ms2"),
            ),
        ),
    ]
    with pytest.raises(script.RegressionError, match="twice"):
        script.build_mouse_tables(out, [inputs[0], inputs[0]])
    result = script.build_mouse_tables(out, inputs)
    folder = out / script.MOUSE_DIR
    for name in ("outcomes", "samples", "gate_signals"):
        assert (folder / f"{name}.csv").exists(), name
    assert result["rows"] == ["MS1/proseg_hybrid", "MS2/proseg_hybrid"]
    assert result["pending"] == [
        "MS1/original_seg",
        "MS1/proseg_mask",
        "MS1/reseg",
        "MS2/original_seg",
    ]
    assert result["not_expected"] == ["MS2/proseg_hybrid"]
    assert result["no_effect_violations"] == []
    assert result["warn_only_samples"] == 2
    saved = json.loads((folder / "mouse_summary.json").read_text())
    assert saved["pending"] == result["pending"]
    report = (folder / "REPORT.txt").read_text()
    assert "pending" in report and "MS1/original_seg" in report
    assert "AP" in report and "C57BL6J-638850.33" in report
    samples = pd.read_csv(folder / "samples.csv")
    assert list(samples["dataset"]) == ["MS1", "MS2"]


def test_the_mouse_step_reads_its_inputs_and_honours_dry_run(
    script: ModuleType,
    tmp_path: Path,
    make_trust: MakeTrust,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _write_mouse_run(tmp_path / "ms1_ph", _mouse_summary(make_trust))
    out = tmp_path / "out"
    # The mouse step reads no store: the legacy store's path (on the damaged
    # SSD1 disk) is never resolved.
    checked: list[Path] = []
    check = script.check_out_dir

    def recorded(out_dir: Path, protected: Any) -> None:
        roots = [Path(root) for root in protected]
        checked.extend(roots)
        check(out_dir, roots)

    monkeypatch.setattr(script, "check_out_dir", recorded)
    common = [
        "--out",
        str(out),
        "--mouse-resolve",
        f"MS1/proseg_hybrid={run}",
        "--m8-resolve",
        str(tmp_path / "m8"),
        "--results",
        str(tmp_path / "results"),
    ]
    assert script.main(["mouse", *common, "--dry-run"]) == 0
    assert not out.exists()
    printed = capsys.readouterr().out
    assert "MS1/proseg_hybrid" in printed and "pending" in printed
    assert script.main(["mouse", *common]) == 0
    assert (out / script.MOUSE_DIR / "mouse_summary.json").exists()
    # No store among the kept-out trees (the mouse step resolves none).
    assert set(checked) == {tmp_path / "m8", tmp_path / "results", run}
    with pytest.raises(SystemExit):
        script.main(["mouse", "--out", str(tmp_path / "other")])
    # The output must stay out of the RESOLVE outputs it reads.
    with pytest.raises(script.RegressionError, match="overlaps"):
        script.main(
            ["mouse", "--out", str(run / "x"), "--mouse-resolve", f"MS1/reseg={run}"]
        )

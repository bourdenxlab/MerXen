"""Tests for the annotation reference store (plan §3.2, §12 M2, §13.1)."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from merxen.annotation import store as store_module
from merxen.annotation.config import (
    AnnotationConfig,
    AnnotationReferenceSpec,
    AnnotationResolvabilityConfig,
)
from merxen.annotation.panel import AnnotationPanel, compute_panel_hash
from merxen.annotation.store import (
    BUNDLE_MANIFEST_NAME,
    FAILED_DIR_PREFIX,
    TMP_DIR_PREFIX,
    BuildContext,
    BundleBuilder,
    BundleIntegrityError,
    PruneRefusedError,
    ReferenceStore,
    UnknownBuilderError,
    build_hash_payload,
    compute_build_hash,
    file_sha256,
    head_tail_sha256,
    register_builder,
    resolve_builder,
    source_record,
)

REPO_SRC = Path(__file__).resolve().parents[2] / "src"
FAKE_CTM = {"version": "1.7.2", "commit": "824caef975618afdadf31172fe2d57e61e657b92"}


def make_panel(
    n_genes: int = 60, *, offset: int = 0, species: str = "human"
) -> AnnotationPanel:
    """Return a synthetic declared panel of ``n_genes`` IDs."""
    prefix = "ENSG" if species == "human" else "ENSMUSG"
    ids = sorted(f"{prefix}{index:011d}" for index in range(offset, offset + n_genes))
    return AnnotationPanel(
        name="intersection",
        kind="intersection",
        species=species,  # type: ignore[arg-type]
        platforms=["MERSCOPE", "XENIUM"],
        sample_ids=["S_M", "S_X"],
        panel_mode="intersection",
        panel_hash=compute_panel_hash(ids),
        n_genes=len(ids),
        ensembl_ids=ids,
        symbols=[f"GENE{index}" for index in range(len(ids))],
    )


@pytest.fixture
def source_file(tmp_path: Path) -> Path:
    """A small source 'precompute' outside the store."""
    path = tmp_path / "sources" / "precomputed_stats.h5"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"precompute-" * 1000)
    return path


@pytest.fixture
def spec(source_file: Path) -> AnnotationReferenceSpec:
    return AnnotationReferenceSpec(
        reference_id="whb_frontal_supc_clus",
        species="human",
        role="primary",
        sources={"precompute": source_file},
        hierarchy=["CCN202210140_SUPC", "CCN202210140_CLUS"],
    )


def copying_builder(
    counter: Path | None = None, *, delay_s: float = 0.0, name: str = "test_builder"
) -> BundleBuilder:
    """A builder that copies the precompute and writes a lookup."""

    def build(context: BuildContext) -> Mapping[str, Any]:
        if counter is not None:
            with counter.open("a") as handle:
                handle.write(f"{os.getpid()}\n")
        context.copy_source("precompute", "precompute/precomputed_stats.h5")
        (context.work_dir / "query_markers.json").write_text('{"None": {}}')
        (context.scratch_dir / "reference_markers.h5").write_bytes(b"scratch")
        time.sleep(delay_s)
        return {"levels": ["CCN202210140_SUPC"], "panel_trust": "provisional"}

    return BundleBuilder(name=name, build=build, taxonomy_id="CCN202210140")


def payload_for(
    spec: AnnotationReferenceSpec,
    panel: AnnotationPanel | None,
    *,
    builder: BundleBuilder | None = None,
    config: AnnotationConfig | None = None,
    ctm: Mapping[str, str | None] = FAKE_CTM,
) -> dict[str, Any]:
    sources = {name: source_record(name, path) for name, path in spec.sources.items()}
    return build_hash_payload(
        spec,
        panel,
        builder=builder or copying_builder(),
        sources=sources,
        config=config or AnnotationConfig(species=spec.species),
        ctm=ctm,
    )


def snapshot(root: Path) -> dict[str, tuple[int, int]]:
    """Every file under ``root`` with its size and mtime."""
    return {
        str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }


# --------------------------------------------------------------------------
# build_hash


def test_build_hash_is_stable(spec: AnnotationReferenceSpec) -> None:
    panel = make_panel()
    first = compute_build_hash(payload_for(spec, panel))
    second = compute_build_hash(payload_for(spec, panel))
    assert first == second
    assert len(first) == 64


def test_build_hash_is_stable_across_processes(
    spec: AnnotationReferenceSpec, tmp_path: Path
) -> None:
    panel = make_panel()
    panel_path = tmp_path / "panel.json"
    panel.write(panel_path)
    script = (
        "import json,sys\n"
        "from merxen.annotation.config import AnnotationConfig, "
        "AnnotationReferenceSpec\n"
        "from merxen.annotation.panel import load_annotation_panel\n"
        "from merxen.annotation.store import build_hash_payload, "
        "compute_build_hash, source_record, BundleBuilder\n"
        "spec = AnnotationReferenceSpec.model_validate_json(sys.argv[1])\n"
        "panel = load_annotation_panel(sys.argv[2])\n"
        "sources = {n: source_record(n, p) for n, p in spec.sources.items()}\n"
        "builder = BundleBuilder(name='test_builder', build=lambda c: {}, "
        "taxonomy_id='CCN202210140')\n"
        "payload = build_hash_payload(spec, panel, builder=builder, "
        "sources=sources, config=AnnotationConfig(species='human'), "
        "ctm=json.loads(sys.argv[3]))\n"
        "print(compute_build_hash(payload))\n"
    )
    env = {**os.environ, "PYTHONPATH": str(REPO_SRC), "PYTHONHASHSEED": "12345"}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            spec.model_dump_json(),
            str(panel_path),
            json.dumps(FAKE_CTM),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert result.stdout.strip() == compute_build_hash(payload_for(spec, panel))


def test_build_hash_ignores_the_order_of_nodes_to_drop(
    spec: AnnotationReferenceSpec,
) -> None:
    panel = make_panel()
    first = spec.model_copy(update={"nodes_to_drop": ["b", "a"]})
    second = spec.model_copy(update={"nodes_to_drop": ["a", "b"]})
    assert compute_build_hash(payload_for(first, panel)) == compute_build_hash(
        payload_for(second, panel)
    )


def test_build_hash_depends_on_the_builder_version(
    spec: AnnotationReferenceSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    panel = make_panel()
    before = compute_build_hash(payload_for(spec, panel))
    monkeypatch.setattr(
        store_module,
        "ANNOTATION_BUILDER_VERSION",
        store_module.ANNOTATION_BUILDER_VERSION + 1,
    )
    after = compute_build_hash(payload_for(spec, panel))
    assert before != after


@pytest.mark.parametrize(
    "change",
    [
        "panel",
        "panel_symbols",
        "hierarchy",
        "nodes_to_drop",
        "drop_level",
        "n_per_utility",
        "max_cells_per_cluster",
        "bootstrap_factor",
        "rng_seed",
        "depth_grid",
        "ctm_version",
        "ctm_commit",
        "taxonomy",
        "builder_name",
        "builder_params",
        "resolvability",
        "source_content",
        "source_mtime",
        "source_path",
    ],
)
def test_build_hash_changes_with_each_input(
    change: str, spec: AnnotationReferenceSpec, source_file: Path, tmp_path: Path
) -> None:
    panel = make_panel()
    base = compute_build_hash(payload_for(spec, panel))
    kwargs: dict[str, Any] = {}
    changed_spec, changed_panel = spec, panel
    if change == "panel":
        changed_panel = make_panel(offset=1)
    elif change == "panel_symbols":
        changed_panel = panel.model_copy(
            update={"symbols": [f"OTHER{i}" for i in range(panel.n_genes)]}
        )
    elif change == "hierarchy":
        changed_spec = spec.model_copy(update={"hierarchy": ["CCN202210140_SUPC"]})
    elif change == "nodes_to_drop":
        changed_spec = spec.model_copy(update={"nodes_to_drop": ["CS202210140_1"]})
    elif change == "drop_level":
        changed_spec = spec.model_copy(update={"drop_level": "CCN202210140_CLUS"})
    elif change == "n_per_utility":
        changed_spec = spec.model_copy(update={"n_per_utility": 10})
    elif change == "max_cells_per_cluster":
        changed_spec = spec.model_copy(update={"max_cells_per_cluster": 100})
    elif change == "bootstrap_factor":
        changed_spec = spec.model_copy(update={"bootstrap_factor": 0.9})
    elif change == "rng_seed":
        changed_spec = spec.model_copy(update={"rng_seed": 1})
    elif change == "depth_grid":
        changed_spec = spec.model_copy(update={"depth_grid": [10, 20]})
    elif change == "ctm_version":
        kwargs["ctm"] = {**FAKE_CTM, "version": "1.5.5"}
    elif change == "ctm_commit":
        kwargs["ctm"] = {**FAKE_CTM, "commit": "0" * 40}
    elif change == "taxonomy":
        kwargs["builder"] = BundleBuilder(
            name="test_builder", build=lambda c: {}, taxonomy_id="CCN20230505"
        )
    elif change == "builder_name":
        kwargs["builder"] = copying_builder(name="other_builder")
    elif change == "builder_params":
        kwargs["builder"] = BundleBuilder(
            name="test_builder",
            build=lambda c: {},
            taxonomy_id="CCN202210140",
            params={"min_cells_per_leaf": 20},
        )
    elif change == "resolvability":
        config = AnnotationConfig(
            species="human",
            resolvability=AnnotationResolvabilityConfig(spill_fraction=0.35),
        )
        kwargs["config"] = config
    elif change == "source_content":
        # Same size, different bytes.
        data = bytearray(source_file.read_bytes())
        data[0] ^= 0xFF
        stat = source_file.stat()
        source_file.write_bytes(bytes(data))
        os.utime(source_file, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    elif change == "source_mtime":
        stat = source_file.stat()
        os.utime(source_file, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    elif change == "source_path":
        moved = tmp_path / "elsewhere.h5"
        moved.write_bytes(source_file.read_bytes())
        changed_spec = spec.model_copy(update={"sources": {"precompute": moved}})
    assert (
        compute_build_hash(payload_for(changed_spec, changed_panel, **kwargs)) != base
    )


def test_build_hash_covers_the_prefilter_of_large_panels_only(
    spec: AnnotationReferenceSpec,
) -> None:
    small, large = make_panel(60), make_panel(1500)
    config = AnnotationConfig(species="human")
    no_filter = config.model_copy(
        update={
            "panel": config.panel.model_copy(
                update={"large_panel_marker_prefilter": "none"}
            )
        }
    )
    assert payload_for(spec, small, config=config)["large_panel_prefilter"] is None
    assert payload_for(spec, large, config=config)["large_panel_prefilter"] == {
        "method": "per_parent_topk_union",
        "cap": 2000,
    }
    assert compute_build_hash(
        payload_for(spec, large, config=config)
    ) != compute_build_hash(payload_for(spec, large, config=no_filter))
    # Large panels use the wide depth grid.
    assert payload_for(spec, large)["depth_grid"][-1] == 2000


def test_build_hash_payload_rejects_missing_or_foreign_panels(
    spec: AnnotationReferenceSpec,
) -> None:
    with pytest.raises(ValueError, match="needs a panel"):
        payload_for(spec, None)
    with pytest.raises(ValueError, match="species"):
        payload_for(spec, make_panel(species="mouse"))


def test_head_tail_digest_equals_full_digest_for_small_files(
    source_file: Path,
) -> None:
    assert head_tail_sha256(source_file) == file_sha256(source_file)


def test_head_tail_digest_reads_only_both_ends(tmp_path: Path) -> None:
    path = tmp_path / "big.bin"
    path.write_bytes(b"a" * 100 + b"b" * 100 + b"c" * 100)
    before = head_tail_sha256(path, block_bytes=100)
    path.write_bytes(b"a" * 100 + b"X" * 100 + b"c" * 100)
    assert head_tail_sha256(path, block_bytes=100) == before
    path.write_bytes(b"Y" * 100 + b"b" * 100 + b"c" * 100)
    assert head_tail_sha256(path, block_bytes=100) != before


def test_source_record_of_a_directory_lists_its_files(tmp_path: Path) -> None:
    directory = tmp_path / "wmb"
    (directory / "sub").mkdir(parents=True)
    (directory / "b.h5ad").write_bytes(b"b")
    (directory / "sub" / "a.h5ad").write_bytes(b"a")
    (directory / ".hidden").write_bytes(b"h")
    record = source_record("wmb_h5ad_dir", directory)
    assert record.kind == "directory"
    assert [Path(f.path).name for f in record.files] == ["b.h5ad", "a.h5ad"]
    with pytest.raises(FileNotFoundError):
        source_record("missing", tmp_path / "missing")


# --------------------------------------------------------------------------
# get_or_build


def test_get_or_build_builds_once_then_reuses(
    spec: AnnotationReferenceSpec, tmp_path: Path, source_file: Path
) -> None:
    store = ReferenceStore(tmp_path / "store", scratch_root=tmp_path / "scratch")
    (tmp_path / "scratch").mkdir()
    counter = tmp_path / "count.txt"
    panel = make_panel()
    first = store.get_or_build(spec, panel, builder=copying_builder(counter))
    second = store.get_or_build(spec, panel, builder=copying_builder(counter))
    assert counter.read_text().count("\n") == 1
    assert not first.reused and second.reused
    assert first.build_hash == second.build_hash
    bundle_dir = Path(first.path)
    assert bundle_dir == tmp_path / "store" / spec.reference_id / first.build_hash
    manifest = json.loads((bundle_dir / BUNDLE_MANIFEST_NAME).read_text())
    assert manifest["status"] == "complete"
    assert manifest["build_hash"] == first.build_hash
    assert compute_build_hash(manifest["build_hash_payload"]) == first.build_hash
    assert manifest["panel_trust"] == first.panel_trust == "provisional"
    source = manifest["sources"]["precompute"]["files"][0]
    assert source["sha256"] == file_sha256(source_file)
    assert {record["path"] for record in manifest["files"]} == {
        "precompute/precomputed_stats.h5",
        "query_markers.json",
    }
    # The scratch directory lives outside the store and is gone afterwards.
    assert list((tmp_path / "scratch").iterdir()) == []
    assert not any(p.name.startswith(TMP_DIR_PREFIX) for p in store.root.iterdir())
    assert (store.root / spec.reference_id / ".lock").exists()
    assert store.find(spec, panel, builder=copying_builder()) == second


def test_get_or_build_uses_the_default_config_when_none_is_given(
    spec: AnnotationReferenceSpec, tmp_path: Path
) -> None:
    store = ReferenceStore(tmp_path / "store")
    panel = make_panel()
    built = store.get_or_build(spec, panel, builder=copying_builder())
    again = store.get_or_build(
        spec,
        panel,
        builder=copying_builder(),
        config=AnnotationConfig(species="human"),
    )
    assert again.reused and again.build_hash == built.build_hash


def test_sources_are_copied_never_symlinked(
    spec: AnnotationReferenceSpec, tmp_path: Path, source_file: Path
) -> None:
    store = ReferenceStore(tmp_path / "store")
    bundle = store.get_or_build(spec, make_panel(), builder=copying_builder())
    copy = Path(bundle.path) / "precompute" / "precomputed_stats.h5"
    assert copy.is_file() and not copy.is_symlink()
    assert copy.stat().st_ino != source_file.stat().st_ino
    assert file_sha256(copy) == file_sha256(source_file)
    source_file.unlink()
    assert copy.is_file()


def test_a_builder_that_symlinks_is_refused(
    spec: AnnotationReferenceSpec, tmp_path: Path, source_file: Path
) -> None:
    def build(context: BuildContext) -> Mapping[str, Any]:
        (context.work_dir / "precompute.h5").symlink_to(source_file)
        return {}

    store = ReferenceStore(tmp_path / "store")
    with pytest.raises(BundleIntegrityError, match="symlink"):
        store.get_or_build(
            spec, make_panel(), builder=BundleBuilder(name="bad", build=build)
        )
    assert not (store.root / spec.reference_id).exists() or not any(
        not p.name.startswith(".") for p in (store.root / spec.reference_id).iterdir()
    )
    assert source_file.is_file()


def test_a_failed_build_leaves_no_partial_bundle(
    spec: AnnotationReferenceSpec, tmp_path: Path
) -> None:
    def build(context: BuildContext) -> Mapping[str, Any]:
        context.copy_source("precompute", "precompute/precomputed_stats.h5")
        raise RuntimeError("marker discovery failed")

    store = ReferenceStore(tmp_path / "store")
    panel = make_panel()
    with pytest.raises(RuntimeError, match="marker discovery failed"):
        store.get_or_build(spec, panel, builder=BundleBuilder(name="b", build=build))
    reference_dir = store.root / spec.reference_id
    assert [p.name for p in reference_dir.iterdir()] == [".lock"]
    failed = [p for p in store.root.iterdir() if p.name.startswith(FAILED_DIR_PREFIX)]
    assert len(failed) == 1
    assert "marker discovery failed" in (failed[0] / "build_error.txt").read_text()
    # The failed files are kept, and a later build succeeds beside them.
    assert (failed[0] / "precompute" / "precomputed_stats.h5").is_file()
    bundle = store.get_or_build(spec, panel, builder=copying_builder())
    assert Path(bundle.path).is_dir()
    assert failed[0].is_dir()
    kinds = sorted(entry.kind for entry in store.list())
    assert kinds == ["bundle", "failed"]


# A separate interpreter per builder, as for two Nextflow tasks or launches.
BUILDER_SCRIPT = """
import os, sys, time
from pathlib import Path
from merxen.annotation.config import AnnotationReferenceSpec
from merxen.annotation.panel import load_annotation_panel
from merxen.annotation.store import BundleBuilder, ReferenceStore

spec = AnnotationReferenceSpec.model_validate_json(Path(sys.argv[1]).read_text())
panel = load_annotation_panel(sys.argv[2])
store_root, counter, go, mode = sys.argv[3:7]


def build(context):
    with open(counter, "a") as handle:
        handle.write(f"{os.getpid()}\\n")
    context.copy_source("precompute", "precompute/precomputed_stats.h5")
    (context.work_dir / "query_markers.json").write_text("{}")
    if mode == "hang":
        Path(go + ".building").write_text("building")
        time.sleep(120)
    time.sleep(1.0)
    return {"levels": ["CCN202210140_SUPC"]}


while not Path(go).exists():
    time.sleep(0.01)
builder = BundleBuilder(name="test_builder", build=build, taxonomy_id="CCN202210140")
print(ReferenceStore(Path(store_root)).get_or_build(spec, panel, builder=builder)
      .model_dump_json())
"""


def start_builders(
    tmp_path: Path,
    spec: AnnotationReferenceSpec,
    panel: AnnotationPanel,
    *,
    n_processes: int,
    mode: str = "normal",
) -> tuple[list[subprocess.Popen[str]], Path, Path]:
    """Start builder interpreters that wait for a go file."""
    script = tmp_path / "builder.py"
    script.write_text(BUILDER_SCRIPT)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(spec.model_dump_json())
    panel_path = panel.write(tmp_path / "panel.json")
    counter = tmp_path / "count.txt"
    go = tmp_path / "go"
    env = {**os.environ, "PYTHONPATH": str(REPO_SRC), "CUDA_VISIBLE_DEVICES": ""}
    processes = [
        subprocess.Popen(
            [
                sys.executable,
                str(script),
                str(spec_path),
                str(panel_path),
                str(tmp_path / "store"),
                str(counter),
                str(go),
                mode,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        for _ in range(n_processes)
    ]
    return processes, counter, go


def test_two_concurrent_processes_build_once(
    spec: AnnotationReferenceSpec, tmp_path: Path
) -> None:
    panel = make_panel()
    processes, counter, go = start_builders(tmp_path, spec, panel, n_processes=2)
    go.write_text("go")
    outcomes = []
    for process in processes:
        stdout, stderr = process.communicate(timeout=120)
        assert process.returncode == 0, stderr
        outcomes.append(json.loads(stdout.strip().splitlines()[-1]))
    assert counter.read_text().count("\n") == 1
    assert len({outcome["build_hash"] for outcome in outcomes}) == 1
    assert sorted(outcome["reused"] for outcome in outcomes) == [False, True]
    assert [entry.kind for entry in ReferenceStore(tmp_path / "store").list()] == [
        "bundle"
    ]


def test_a_killed_builder_leaves_no_bundle_and_releases_the_lock(
    spec: AnnotationReferenceSpec, tmp_path: Path
) -> None:
    panel = make_panel()
    (process,), _counter, go = start_builders(
        tmp_path, spec, panel, n_processes=1, mode="hang"
    )
    go.write_text("go")
    building = Path(f"{go}.building")
    deadline = time.monotonic() + 60
    while not building.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert building.exists(), process.stderr.read() if process.stderr else ""
    process.send_signal(signal.SIGKILL)
    process.communicate(timeout=30)
    store = ReferenceStore(tmp_path / "store")
    entries = store.list()
    assert [entry.kind for entry in entries] == ["tmp"]
    assert entries[0].builder_alive is False
    assert entries[0].reference_id == spec.reference_id
    reference_dir = tmp_path / "store" / spec.reference_id
    assert [path.name for path in reference_dir.iterdir()] == [".lock"]
    bundle = store.get_or_build(spec, panel, builder=copying_builder())
    assert Path(bundle.path).is_dir()
    # The dead build's directory is still there: nothing is deleted.
    assert entries[0].path.is_dir()
    report = store.prune(unreferenced_by=tmp_path, dry_run=True)
    assert entries[0].path in {entry.path for entry in report.candidates}


def test_existing_bundles_are_never_deleted_or_modified(
    spec: AnnotationReferenceSpec, tmp_path: Path
) -> None:
    store = ReferenceStore(tmp_path / "store")
    first = store.get_or_build(spec, make_panel(), builder=copying_builder())
    before = snapshot(Path(first.path))
    second = store.get_or_build(spec, make_panel(offset=5), builder=copying_builder())
    third = store.get_or_build(
        spec.model_copy(update={"n_per_utility": 10}),
        make_panel(),
        builder=copying_builder(),
    )
    assert len({first.path, second.path, third.path}) == 3
    assert snapshot(Path(first.path)) == before
    results = tmp_path / "results"
    results.mkdir()
    store.list()
    store.prune(unreferenced_by=results)
    assert snapshot(Path(first.path)) == before
    assert Path(second.path).is_dir() and Path(third.path).is_dir()


def test_an_invalid_bundle_directory_is_reported_and_left_alone(
    spec: AnnotationReferenceSpec, tmp_path: Path
) -> None:
    store = ReferenceStore(tmp_path / "store")
    panel = make_panel()
    built = store.get_or_build(spec, panel, builder=copying_builder())
    bundle_dir = Path(built.path)
    (bundle_dir / "query_markers.json").write_text("edited by hand, longer")
    before = snapshot(bundle_dir)
    with pytest.raises(BundleIntegrityError, match="left untouched"):
        store.get_or_build(spec, panel, builder=copying_builder())
    assert snapshot(bundle_dir) == before


def test_a_source_changing_during_the_build_fails_it(
    spec: AnnotationReferenceSpec, tmp_path: Path, source_file: Path
) -> None:
    def build(context: BuildContext) -> Mapping[str, Any]:
        source_file.write_bytes(b"rewritten while building")
        return {}

    store = ReferenceStore(tmp_path / "store")
    with pytest.raises(BundleIntegrityError, match="changed during the build"):
        store.get_or_build(spec, make_panel(), builder=BundleBuilder("b", build))
    assert [e.kind for e in store.list()] == ["failed"]


def test_large_panels_go_to_the_large_store(
    spec: AnnotationReferenceSpec, tmp_path: Path
) -> None:
    store = ReferenceStore(
        tmp_path / "ssd", large_root=tmp_path / "large", large_panel_genes=100
    )
    small = store.get_or_build(spec, make_panel(60), builder=copying_builder())
    large = store.get_or_build(spec, make_panel(150), builder=copying_builder())
    assert Path(small.path).is_relative_to(tmp_path / "ssd")
    assert Path(large.path).is_relative_to(tmp_path / "large")
    assert large.store_root == str(tmp_path / "large")
    assert {entry.store_root for entry in store.list()} == {
        tmp_path / "ssd",
        tmp_path / "large",
    }


def test_panel_independent_references_ignore_the_panel(tmp_path: Path) -> None:
    metadata = tmp_path / "cell_metadata.csv"
    metadata.write_text("cell_label,x_ccf\n1,5.0\n")
    spec = AnnotationReferenceSpec(
        reference_id="wmb_region_share",
        species="mouse",
        role="region_share",
        sources={"merfish_ccf_metadata": metadata},
    )

    def build(context: BuildContext) -> Mapping[str, Any]:
        assert context.panel is None
        (context.work_dir / "region_share.parquet").write_bytes(b"table")
        return {}

    builder = BundleBuilder(name="region_share", build=build, uses_panel=False)
    store = ReferenceStore(tmp_path / "store")
    first = store.get_or_build(spec, None, builder=builder)
    second = store.get_or_build(spec, make_panel(species="mouse"), builder=builder)
    assert second.reused and first.build_hash == second.build_hash
    assert first.panel_hash is None


def test_scratch_inside_the_store_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="inside the reference store"):
        ReferenceStore(tmp_path / "store", scratch_root=tmp_path / "store" / "tmp")
    with pytest.raises(ValueError, match="inside the reference store"):
        ReferenceStore(
            tmp_path / "store",
            large_root=tmp_path / "large",
            scratch_root=tmp_path / "large",
        )


def test_resolve_builder_reports_unknown_references(
    spec: AnnotationReferenceSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(store_module, "_BUILDER_FACTORIES", {})
    monkeypatch.setattr(store_module, "BUILDERS_MODULE", "merxen.annotation._absent_")
    with pytest.raises(UnknownBuilderError, match="whb_frontal_supc_clus"):
        resolve_builder(spec)
    register_builder(spec.reference_id, lambda s, c: copying_builder())
    assert resolve_builder(spec).name == "test_builder"


# --------------------------------------------------------------------------
# list and prune


def test_prune_dry_run_only_lists(
    spec: AnnotationReferenceSpec, tmp_path: Path
) -> None:
    store = ReferenceStore(tmp_path / "store")
    kept = store.get_or_build(spec, make_panel(), builder=copying_builder())
    unused = store.get_or_build(spec, make_panel(offset=3), builder=copying_builder())
    results = tmp_path / "results" / "P7513" / "proseg_hybrid" / "annotation"
    results.mkdir(parents=True)
    kept.write(results / "bundle_ref.json")
    # Records inside zarr stores are never read.
    zarr_record = tmp_path / "results" / "x.zarr" / "bundle_ref.json"
    zarr_record.parent.mkdir()
    unused.write(zarr_record)
    before = snapshot(store.root)
    report = store.prune(unreferenced_by=tmp_path / "results", dry_run=True)
    assert [entry.build_hash for entry in report.referenced] == [kept.build_hash]
    assert [entry.build_hash for entry in report.candidates] == [unused.build_hash]
    assert report.n_record_files == 1
    assert report.candidate_bytes > 0
    assert snapshot(store.root) == before
    with pytest.raises(PruneRefusedError):
        store.prune(unreferenced_by=tmp_path / "results", dry_run=False)
    assert snapshot(store.root) == before


def test_list_reports_bundle_details(
    spec: AnnotationReferenceSpec, tmp_path: Path
) -> None:
    store = ReferenceStore(tmp_path / "store")
    panel = make_panel()
    bundle = store.get_or_build(spec, panel, builder=copying_builder())
    (entry,) = store.list()
    assert entry.kind == "bundle"
    assert entry.reference_id == spec.reference_id
    assert entry.build_hash == bundle.build_hash
    assert entry.panel_hash == panel.panel_hash
    assert entry.n_panel_genes == 60
    assert entry.size_bytes > 0
    assert json.dumps(entry.to_json())


# --------------------------------------------------------------------------
# Imports


def test_store_and_panel_import_without_heavy_dependencies() -> None:
    # ctm, anndata and h5py load only when a bundle is built or data is read.
    script = (
        "import sys\n"
        "BLOCKED = {'anndata', 'cell_type_mapper', 'h5py', 'scanpy', 'scipy', "
        "'spatialdata', 'matplotlib', 'torch'}\n"
        "class Blocker:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in BLOCKED:\n"
        "            raise ImportError(name)\n"
        "sys.meta_path.insert(0, Blocker())\n"
        "import merxen.annotation.store, merxen.annotation.panel\n"
        "print('ok')\n"
    )
    env = {**os.environ, "PYTHONPATH": str(REPO_SRC)}
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr

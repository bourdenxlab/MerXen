"""Pin the bundle-builder code to ``ANNOTATION_BUILDER_VERSION`` (plan §3.2, E-M7).

``build_hash`` covers ``ANNOTATION_BUILDER_VERSION``, not the builder code: a
builder change that alters bundle content without a version bump would let
every existing bundle be reused with outdated content. This test fingerprints
the code that decides bundle content, without docstrings, comments or
formatting, and fails whenever it changes. Then either

* bump ``ANNOTATION_BUILDER_VERSION`` in ``merxen/annotation/store.py`` (the
  change alters what a bundle contains) and pin the new fingerprint to the
  new version, or
* keep the version and update the fingerprint of the current version, adding
  a line to ``PIN_HISTORY`` that says why no bundle content changes.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

from merxen.annotation import store

SRC = Path(__file__).resolve().parents[2] / "src" / "merxen" / "annotation"
# Everything in reference.py decides bundle content; in store.py only the
# build-hash and bundle-writing helpers do.
STORE_NAMES = (
    "canonical_json",
    "head_tail_sha256",
    "FileIdentity",
    "file_identity",
    "SourceRecord",
    "source_record",
    "BuildContext",
    "BundleBuilder",
    "build_hash_payload",
    "compute_build_hash",
    "_bundle_files",
    "_make_read_only",
)
PINNED_FINGERPRINTS = {
    2: "f2790c373d24f879672ee2524fa3eac885960b69771e5bc89888a9de9d89c9a8",
    3: "3dfd14c7fea3c007d4688f7ad6401d5b975f09054e48dbe112e7c8e7d84212d5",
}
# Why a fingerprint changed without a version bump (newest last).
PIN_HISTORY = (
    "2: first pin (set c from the curated family list, ID-only bundle tables, "
    "validated WMB universe, self-map test-cell source, read-only bundles)",
    "3: the resolvability self-map (held-out WHB and WMB test-set builders, "
    "self-map tables in primary and secondary bundles); resolvability.py pinned",
    "3: level_emission and composition docs in resolvability.py (RESOLVE-side "
    "helpers; the bundle files are unchanged)",
    "3: RESOLVABILITY_VERSION 2 (M3b review: validated regime on both halves, "
    "per-depth trimmed composition weights, the WHB COP rule on the cells "
    "table, per-platform packaged floors); it enters build_hash through the "
    "resolvability builder params, so every self-map bundle gets a new hash "
    "without a builder bump",
    "3: large-panel builds refused and the prefilter kept out of the payload "
    "until a builder applies it (no payload of a buildable panel changes); "
    "the held-out test-set sources checked before the marker steps (same "
    "bundles, earlier failure)",
    "3: RESOLVABILITY_VERSION 3 (H18 follow-up, user decision 2026-09-27: "
    "pooled deep bins with the point precision in the rule); the version "
    "enters every self-map bundle's build_hash through the resolvability "
    "builder params, so those bundles get new hashes without a builder bump",
    "3: the human held-out test set topped up with other-region non-neuronal "
    "WHB cells (user decision 2026-09-27); the new content enters the "
    "whb_frontal_supc_clus_ho build_hash through its hashed other_region "
    "params and the WHB cell metadata source, and every self-map bundle's "
    "through the test-set params and RESOLVABILITY_VERSION 3; the mouse "
    "test-set and all other bundles are unchanged",
    "3: pooled deep sets reweighted as one set in RESOLVE "
    "(ResolvabilityTables.decisions, pooled_composition_weights, "
    "DatasetComposition.at_least); PREP's decisions are unweighted, so the "
    "bundle files are unchanged",
    "3: large-panel support (M3b stage D): the per-parent marker prefilter "
    "(prefilter.py, pinned from here on) runs only above large_panel_genes, "
    "where builds were refused before, and its method, version and settings "
    "enter build_hash; the builder refusals move to BundleBuilder.refuse; "
    "reference markers of panels up to 1,000 genes are deleted right after "
    "the query markers instead of with the build scratch (same files); ctm "
    "steps also record process-tree peaks and bundle.json the candidate-gene "
    "count (metrics only); the self-map mapper and marker steps are shared "
    "with annotation-panel-simulate. No bundle of a panel up to 1,000 genes "
    "changes content or build_hash",
    "3: the WMB query-marker memory model of the no-prefilter refusal is the "
    "measured M3b 5K envelope (refusals only; no bundle content changes)",
    "3: RESOLVABILITY_VERSION 4 (M3b review 2: weights trimmed per judged set, "
    "positive-weight calls only, judged bins at or above D_P keep their own "
    "verdict, would_raise only with a fit, PREP decides on the stored float32 "
    "cells, gate-P sets of one replicate); the version enters every self-map "
    "bundle's build_hash through the resolvability builder params, so those "
    "bundles get new hashes without a builder bump",
    "3: the human held-out test set leaves out truth superclusters no call "
    "can name (sinks, no floor class, region-implausible; M3b review 2); the "
    "exclusion's version and region enter the whb_frontal_supc_clus_ho "
    "build_hash through its hashed params, and every human self-map bundle's "
    "through its test-set params",
    "3: the no-prefilter refusal keeps the OD-E8 +30% margin on the predicted "
    "WMB query-marker peak and names the pipeline's option (refusals only; "
    "no bundle content changes)",
    "3: the human held-out test set's other-region top-up draws only from "
    "clusters of the held-out training reference (user decision 2026-09-27); "
    "HO_OTHER_REGION_VERSION 2 enters the whb_frontal_supc_clus_ho build_hash "
    "through its hashed other_region params, and every human self-map "
    "bundle's through its test-set params; the mouse bundles are unchanged",
    "3: RESOLVABILITY_VERSION 5 (M3b final follow-up: every simulation draw "
    "keyed by the seed, recipe name and version, cell id and depth, the "
    "gene efficiency by the gene id, the spill partner by rendezvous "
    "hashing); the version enters every self-map bundle's build_hash through "
    "the resolvability builder params, so those bundles get new hashes "
    "without a builder bump; the test-set bundles are unchanged",
    "3: RESOLVABILITY_VERSION 6 (M3b review 3: the per-gene efficiency is "
    "again the pre-registered version-4 draw over the panel's gene order; "
    "the per-cell keys stay); the version enters every self-map bundle's "
    "build_hash through the resolvability builder params, so those bundles "
    "get new hashes without a builder bump; the test-set bundles are "
    "unchanged",
    "3: the human self-maps leave the held-out test set's other-region COP "
    "cells out (user decision 2026-09-30, M8 D1; "
    "HO_SELF_MAP_TEST_SET_REVISION 1): the revision and its rule enter every "
    "human self-map bundle's build_hash through its test-set params, so "
    "those bundles get new hashes without a builder bump; the held-out "
    "test-set bundle, RESOLVABILITY_VERSION and the mouse bundles are "
    "unchanged",
    "3: thin_and_contaminate takes an optional efficiency_seed for the M8 "
    "draw-spread grid (user decision 2026-09-30, M8 D2); every builder "
    "leaves it unset, so the efficiency stays the recipe seed's and no "
    "bundle content or build_hash changes",
)


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    for node in ast.walk(tree):
        if isinstance(
            node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
        ):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                node.body = body[1:] or [ast.Pass()]
    return tree


def _dump(node: ast.AST) -> str:
    return ast.dump(_strip_docstrings(node), include_attributes=False)


def builder_code_fingerprint() -> str:
    """Return the sha256 of the builder-relevant code, docstrings stripped.

    ``resolvability.py`` writes the self-map tables into primary and
    secondary bundles (M3b), so it is covered too; a change there that
    alters those tables also bumps ``RESOLVABILITY_VERSION`` (hashed).
    ``prefilter.py`` chooses the marker candidates of large panels (M3b
    stage D); its version and settings are hashed.
    """
    parts = [
        _dump(ast.parse((SRC / "reference.py").read_text())),
        _dump(ast.parse((SRC / "resolvability.py").read_text())),
        _dump(ast.parse((SRC / "prefilter.py").read_text())),
    ]
    store_tree = ast.parse((SRC / "store.py").read_text())
    by_name = {
        node.name: node
        for node in store_tree.body
        if isinstance(node, ast.FunctionDef | ast.ClassDef)
    }
    missing = [name for name in STORE_NAMES if name not in by_name]
    assert not missing, f"store.py no longer defines {missing}; update STORE_NAMES"
    parts += [_dump(by_name[name]) for name in STORE_NAMES]
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def test_builder_code_is_pinned_to_the_builder_version() -> None:
    version = store.ANNOTATION_BUILDER_VERSION
    assert version in PINNED_FINGERPRINTS, (
        f"ANNOTATION_BUILDER_VERSION {version} has no pinned fingerprint; add "
        f"{version}: {builder_code_fingerprint()!r} to PINNED_FINGERPRINTS"
    )
    current = builder_code_fingerprint()
    assert current == PINNED_FINGERPRINTS[version], (
        "The bundle-builder code changed. If bundle content changes, bump "
        "ANNOTATION_BUILDER_VERSION in merxen/annotation/store.py and pin "
        f"{current!r} to the new version; otherwise update the pin of version "
        f"{version} to {current!r} and say why in PIN_HISTORY."
    )


def test_the_fingerprint_ignores_docstrings_and_comments() -> None:
    source = 'def f(x):\n    """Doc."""\n    # comment\n    return x + 1\n'
    other = "def f(x):\n    return x + 1\n"
    assert _dump(ast.parse(source)) == _dump(ast.parse(other))
    assert _dump(ast.parse(other)) != _dump(ast.parse("def f(x):\n    return x\n"))

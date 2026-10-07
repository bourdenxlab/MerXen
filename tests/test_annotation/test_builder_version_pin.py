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
    3: "2d4a46d01cc56edf450c1e01f1a4aa688a7485e217d29644c1a22f439bddaf28",
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
    "3: resolvability version 7 simulation added (M3c stage B: exact-total "
    "thinning, the totals grid, ensemble members, the R3 and lung-stress "
    "efficiencies from sim_inputs.py (pinned from here on), version "
    "selection); additive: no builder calls it yet, the version-6 path and "
    "every version-6 build_hash are byte-identical "
    "(test_resolvability_v6_golden), so no bundle changes",
    "3: resolvability version 7 wired into PREP per family (M3c stage C: the "
    "ensemble decisions, saturated-bp rule, monotone fill, class-depth table, "
    "test-set class top-up, BundleBuilder.panel_params); a version-7 family's "
    "build_hash gains its version-7 inputs through panel_params (so its "
    "bundles are new build directories), a version-6 family's payload and "
    "every version-6 output are byte-identical (test_resolvability_v6_golden), "
    "so no existing bundle changes",
    "3: the version-7 trust constraint guarded against simulation-input assets "
    "(M3c stage D, pre-registration §21 (v): the more severe of the full "
    "ensemble's and the asset-free members' constraint; the summary records "
    "trust_asset_guard); it changes the trust of version-7 bundles with an "
    "asset member only, none of which existed in any store (the 5K mouse "
    "build was stopped before its self-map for this change); other version-7 "
    "bundles gain only that summary record; the version-6 path is "
    "byte-identical (test_resolvability_v6_golden), so no existing bundle "
    "changes",
    "3: the version-7 ensemble amended (M3c amendment of 2026-09-29, "
    "pre-registration §22: eight emission members per family and a "
    "one-standard-error margin on the spread route); ensemble_rule_version 2 "
    "and the new members enter every version-7 build_hash (panel_params), so "
    "the new decisions go only into new build directories and the stage-D "
    "version-7 bundles are never reused by the amended code; those bundles "
    "re-derive their decisions as built (no recorded margin: 0); the "
    "version-6 path and payload are byte-identical "
    "(test_resolvability_v6_golden), so no existing bundle changes",
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
    "3: M3c merged after M8 (2026-10-01): the builder carries both the "
    "version-7 ensemble (M3c) and the human test-set revision and "
    "efficiency_seed of M8 D1 / D2; each change keeps its own build_hash rule "
    "above, so no bundle built by either branch changes",
    "3: the version-7 self-map records M8 D1's test_set_exclusion in its "
    "summary and bundle output, as the version-6 path does (its test cells "
    "already followed D1); provenance only. The version-7 human bundles "
    "built before the merge (two WHB, two SEA-AD) lack D1's revision in their "
    "hashed test-set params, so the merged code never reuses them and builds "
    "new ones; no existing bundle changes",
    "3: the version-7 summary's per-regime and per-level bin counts moved "
    "into ensemble_bin_counts, which RESOLVE reuses (refactor; the summary "
    "is unchanged), so no bundle content changes",
    "3: load_resolvability and ResolvabilityTables refuse a resolvability "
    "version the code does not know (checked_resolvability_version; reading "
    "only, the RESOLVE follow-up of M3c); no bundle content changes",
    "3: gate_p_tested_sets takes its confident calls from the new public "
    "frozen_confident_mask (M13 gate P, which scores at frozen thresholds "
    "and never runs in a builder); the same rule, moved, so no bundle "
    "content or build_hash changes",
    "3: the M13 branch merged with the RESOLVE version-7 follow-up "
    "(2026-10-06): resolvability.py carries both frozen_confident_mask "
    "and the version guard and ensemble_bin_counts above; each change "
    "keeps its own rule, so no bundle content or build_hash changes",
    "3: sim_inputs.py gains gate P's cross-platform stress efficiency and "
    "the provenance rule of in-house assets (a source path in place of a "
    "URL; M13 C4); no builder reads either, the packaged assets a bundle "
    "hashes are unchanged, so no bundle content or build_hash changes",
    "3: gate P's NP6 stress recipes in resolvability.py (M13 C4: "
    "R1_stress_spill, R1_stress_lognormal, R1_stress_xplatform, "
    "R3_stress_spill, gate_p_stress_members; thin_and_contaminate reads a "
    "table recipe's efficiency); no builder simulates them, the R1, R3, clean "
    "and lung recipe records are unchanged (factor_cap_log2 only when set) and "
    "every lognormal recipe's draw is the same, so no bundle content or "
    "build_hash changes (test_resolvability_v6_golden)",
    "3: the held-out test set's region metadata, other-region metadata and "
    "test-set sources are read from source paths (ho_region_metadata, "
    "other_region_metadata, _test_set_source_paths, _spec_of_test_set; M13 "
    "C8 refactor): the same files are read and the same specs built, so no "
    "bundle content or build_hash changes (test_resolvability_v6_golden)",
    "3: gate P's leave-one-donor-out held-out test sets (M13 C8, D2 (d)): "
    "build_whb_frontal_ho draws the other-region top-up from the held-out "
    "donor only and never a cell of the default test set when "
    "resolvability.holdout_other_region_donor_only and the "
    "gate_p_excluded_test_cells source are both set; the flag enters the "
    "held-out params only when set and the source only when given, so every "
    "production payload, bundle and build_hash is unchanged "
    "(test_resolvability_v6_golden); the pool-size and composition reports "
    "and held_out_test_set_spec are added (no builder output changes)",
    "3: the D1 drop's supercluster labels move to ho_self_map_excluded_labels "
    "(the same labels, read by self_map_test_cells as before), and gate P's "
    "pool-size report (ho_donor_pool_sizes, which no builder calls) leaves "
    "the self-map's dropped other-region cells out of its counts (M13 C8 "
    "review); no bundle content or build_hash changes "
    "(test_resolvability_v6_golden)",
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
    stage D); its version and settings are hashed. ``sim_inputs.py`` (M3c)
    turns the simulation-input assets into version-7 member efficiencies and
    depth profiles; the assets' sha256 enter a version-7 ``build_hash``, the
    code is covered here.
    """
    parts = [
        _dump(ast.parse((SRC / "reference.py").read_text())),
        _dump(ast.parse((SRC / "resolvability.py").read_text())),
        _dump(ast.parse((SRC / "prefilter.py").read_text())),
        _dump(ast.parse((SRC / "sim_inputs.py").read_text())),
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

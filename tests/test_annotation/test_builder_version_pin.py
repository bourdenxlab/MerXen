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
    3: "aa28ead8457b28bf6e268f0184c234dd3a619f265b69ee3f7d6a8395fb7ed10f",
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
    """
    parts = [
        _dump(ast.parse((SRC / "reference.py").read_text())),
        _dump(ast.parse((SRC / "resolvability.py").read_text())),
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

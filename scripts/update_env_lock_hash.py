#!/usr/bin/env python
"""Refresh the ``requirements.lock`` checksum header in ``envs/environment.yml``.

Nextflow names each cached conda environment (``work/conda/env-<hash>``) after
a hash of the environment file's text. ``envs/environment.yml`` installs MerXen
with a thin ``-e "../[dev]"`` line, so its text, and therefore the cached env,
stays the same when only ``requirements/requirements.lock`` changes. Stale envs
then keep running old dependency versions. The header comment records the
lockfile's sha256, so every lockfile change also changes the environment file
and makes Nextflow build a fresh env.

Run this after regenerating the lockfile::

    python scripts/update_env_lock_hash.py

``--check`` leaves the file untouched and exits non-zero when the header is
missing or stale. ``tests/test_workflows/test_env_lock_sync.py`` enforces the
same check in CI.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = REPO_ROOT / "requirements" / "requirements.lock"
ENV_PATH = REPO_ROOT / "envs" / "environment.yml"

HEADER_PREFIX = "# requirements.lock sha256: "
HEADER_PATTERN = re.compile(
    r"^" + re.escape(HEADER_PREFIX) + r"(?P<digest>[0-9a-f]{64})$", re.MULTILINE
)

logger = logging.getLogger(__name__)


def lock_sha256(lock_path: Path) -> str:
    """Return the hex sha256 of a lockfile's raw bytes.

    Args:
        lock_path: Path to the lockfile.

    Returns:
        The lowercase hex digest.
    """
    return hashlib.sha256(lock_path.read_bytes()).hexdigest()


def read_header_digest(env_text: str) -> str | None:
    """Return the lockfile digest recorded in an environment file's header.

    Args:
        env_text: Text of the conda environment file.

    Returns:
        The recorded hex digest, or ``None`` when the file has no header.
    """
    match = HEADER_PATTERN.search(env_text)
    return match.group("digest") if match else None


def update_env_text(env_text: str, digest: str) -> str:
    """Return environment file text whose header records ``digest``.

    An existing header line is replaced in place. A file without one gets the
    header as its first line.

    Args:
        env_text: Text of the conda environment file.
        digest: Hex sha256 of the lockfile.

    Returns:
        The updated environment file text.
    """
    header = f"{HEADER_PREFIX}{digest}"
    if HEADER_PATTERN.search(env_text):
        return HEADER_PATTERN.sub(header, env_text, count=1)
    return f"{header}\n{env_text}"


def main(argv: list[str] | None = None) -> int:
    """Rewrite or check the lockfile checksum header.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 when the header is current (or was rewritten),
        1 when ``--check`` finds it missing or stale.
    """
    parser = argparse.ArgumentParser(
        description="Refresh the requirements.lock checksum header in an env file."
    )
    parser.add_argument("--lock", type=Path, default=LOCK_PATH)
    parser.add_argument("--env", type=Path, default=ENV_PATH)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 instead of rewriting when the header is missing or stale.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    digest = lock_sha256(args.lock)
    env_text = args.env.read_text()
    if read_header_digest(env_text) == digest:
        logger.info("%s already records %s sha256 %s", args.env, args.lock, digest)
        return 0
    if args.check:
        logger.error(
            "%s does not record %s sha256 %s; run %s",
            args.env,
            args.lock,
            digest,
            Path(__file__).name,
        )
        return 1
    args.env.write_text(update_env_text(env_text, digest))
    logger.info("Updated %s to %s sha256 %s", args.env, args.lock, digest)
    return 0


if __name__ == "__main__":
    sys.exit(main())

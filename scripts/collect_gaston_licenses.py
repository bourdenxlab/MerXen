#!/usr/bin/env python3
"""Preserve installed Python dependency notices in the GASTON image."""

from __future__ import annotations

import argparse
import json
import re
from importlib import metadata
from pathlib import Path

REQUIRED = {"gaston-spatial", "glmpca", "kneed"}


def collect_licenses(output: Path) -> None:
    """Copy installed notice files and metadata; require GASTON's core notices.

    Args:
        output: Destination for package subdirectories and inventory.json.

    Raises:
        RuntimeError: A required package or its license notice is missing.
    """
    output.mkdir(parents=True, exist_ok=True)
    inventory = []
    verified = set()
    for distribution in metadata.distributions():
        name = distribution.metadata["Name"]
        normalized = re.sub(r"[-_.]+", "-", name).lower()
        target = output / f"{normalized}-{distribution.version}"
        target.mkdir(exist_ok=True)
        target.joinpath("METADATA").write_text(
            distribution.read_text("METADATA") or str(distribution.metadata),
            encoding="utf-8",
        )
        notices = []
        for entry in distribution.files or []:
            if not (
                any(part.lower() == "licenses" for part in entry.parts[:-1])
                or re.match(
                    r"^(licen[cs]es?|copying|notice|copyright|authors)([._-]|$)",
                    entry.name,
                    re.IGNORECASE,
                )
            ):
                continue
            source = Path(distribution.locate_file(entry))
            if not source.is_file():
                raise RuntimeError(f"Missing installed notice for {name}: {entry}")
            # Distribution records can contain ../ paths; never use them as
            # destination paths. Numbering also preserves duplicate basenames.
            filename = f"{len(notices):03d}-{entry.name}"
            target.joinpath(filename).write_bytes(source.read_bytes())
            notices.append(filename)
        if notices:
            verified.add(normalized)
        inventory.append(
            {
                "name": name,
                "version": distribution.version,
                "license": distribution.metadata.get("License"),
                "license_expression": distribution.metadata.get("License-Expression"),
                "classifiers": distribution.metadata.get_all("Classifier") or [],
                "notices": notices,
            }
        )
    missing = REQUIRED - verified
    if missing:
        raise RuntimeError(
            f"Required package license notices missing: {sorted(missing)}"
        )
    output.joinpath("inventory.json").write_text(
        json.dumps(sorted(inventory, key=lambda item: item["name"]), indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Collect the environment's notices into the requested directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    collect_licenses(parser.parse_args().output)


if __name__ == "__main__":
    main()

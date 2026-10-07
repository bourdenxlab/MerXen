#!/usr/bin/env python
"""Gate P for a new panel family (M13; plan §12 M13, §14 gate P).

A thin entry point over ``merxen annotation-panel-simulate --gate-p``: the
command builds the family's PREP bundles, registers the gate-P programme
(``merxen.annotation.gate_p_run.gate_p_hook``, through
``merxen.annotation.simulate.register_gate_p_hook``; the CLI cannot import
``scripts/``) and runs it after the simulation, writing the NP1-NP9 report
under ``<out-dir>/gate_p``.

``--species`` is required (M13 D4, the user's rule of 2026-10-02: "The species
must be selected when starting a new panel check"); a human family is dry-run
on set a only. Every other argument goes to ``annotation-panel-simulate``
unchanged (``merxen annotation-panel-simulate --help`` lists them, the
``--gate-p-*`` options included). Point ``--store`` at the separate gate-P
store (M13 D11 (b)); the production store is never written by gate P.

Run it from a frozen ``git archive`` export of the code with a ``COMMIT``
file at the export's root holding the full commit id (the M13 dry-run
script's export step writes it): an export has no ``.git``, and the command
records that commit in ``simulate_report.json`` (``provenance.code_commit``)
and ``gate_p/gate_p_run.json`` (``code_commit``). From a worktree the commit
comes from ``git rev-parse HEAD``.

Exit status: 0 when gate P was scored, 3 when it stopped on the per-donor pool
sizes (pre-registration §23.10 open item 2: the user chooses D2 (d) as it
stands, ``--gate-p-accept-small-pools``, or the fallback (c),
``--gate-p-other-region shared``), the command's own status otherwise.

Usage (set a dry run, version 6)::

    python scripts/acceptance/new_panel.py --species human \\
        --gene-list <set a gene list> --name set_a_dry_run \\
        --references whb_frontal_supc_clus --store $A/m13/gate_p_store \\
        --scratch-dir <scratch> --out-dir $A/m13/dryrun/v6 \\
        --param annotation_whb_h5ad_dir=<...> --n-processors 8 \\
        --expected-depth <median> --gate-p-dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

EXIT_STOPPED = 3


def command_args(species: str, rest: Sequence[str]) -> list[str]:
    """Return the ``annotation-panel-simulate`` arguments of a gate-P run.

    Args:
        species: The species selected for the check (M13 D4).
        rest: The other arguments, passed on unchanged.

    Returns:
        The command's arguments, ``--gate-p`` and ``--species`` included.

    Raises:
        SystemExit: If ``rest`` names the species again (it is given once).
    """
    if any(item == "--species" or item.startswith("--species=") for item in rest):
        raise SystemExit("give --species once (it selects the gate-P dry run)")
    return [
        "--species",
        species,
        "--gate-p",
        *[item for item in rest if item != "--gate-p"],
    ]


def _out_dir(rest: Sequence[str]) -> Path | None:
    for position, item in enumerate(rest):
        if item == "--out-dir" and position + 1 < len(rest):
            return Path(rest[position + 1])
        if item.startswith("--out-dir="):
            return Path(item.split("=", 1)[1])
    return None


def main(argv: Sequence[str] | None = None) -> int:
    """Run gate P on a new panel family through ``annotation-panel-simulate``.

    Args:
        argv: Arguments (default: ``sys.argv[1:]``).

    Returns:
        The exit status.
    """
    import click

    from merxen.annotation.gate_p_run import GATE_P_RUN_DIR, GATE_P_RUN_JSON
    from merxen.cli.run_annotation_panels import annotation_panel_simulate_command

    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog="Every other argument is passed to annotation-panel-simulate.",
    )
    parser.add_argument(
        "--species",
        required=True,
        choices=("human", "mouse"),
        help="The species of the new panel family (M13 D4).",
    )
    args, rest = parser.parse_known_args(argv)
    try:
        annotation_panel_simulate_command.main(
            args=command_args(args.species, rest),
            prog_name="new_panel.py",
            standalone_mode=False,
        )
    except click.ClickException as error:
        error.show()
        return int(error.exit_code)
    out_dir = _out_dir(rest)
    if out_dir is None:
        return 0
    record_path = out_dir / GATE_P_RUN_DIR / GATE_P_RUN_JSON
    if not record_path.is_file():
        return 0
    record = json.loads(record_path.read_text(encoding="utf-8"))
    print(
        f"new_panel.py: gate P {record.get('status')} for {record.get('family_id')} "
        f"-> {record_path}"
    )
    return EXIT_STOPPED if record.get("status") == "stopped" else 0


if __name__ == "__main__":
    sys.exit(main())

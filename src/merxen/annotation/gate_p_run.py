"""The gate-P driver: build, simulate, map and score one family (M13 chunk C8).

Plan §14 (new panel family, gate P), §8.8; pre-registration §23.9 and the
decisions of 2026-10-06 (§23.10). ``run_gate_p`` is the programme that
``annotation-panel-simulate --gate-p`` and ``scripts/acceptance/new_panel.py``
register with ``simulate.register_gate_p_hook`` (``gate_p_hook``). It gets the
base simulation (``simulate.GatePRequest``: the panel, the production PREP
bundle of the primary reference ``whb_frontal_supc_clus``, its held-out test
set, the store and the prepared specs) and:

1. **Checks the request**, in two stages. *Before any compute*
   (``precheck_gate_p``, which ``run_panel_simulation`` calls before the
   panel is computed or PREP builds anything; registered with
   ``gate_p_precheck``): the species is the one selected when the check
   started (M13 D4: a required argument; the mouse path waits for the
   second WMB draw, M13 C20, after M6b) and the config's; the store is not
   a production store (D11 (b): no root is or lies in the config's
   ``reference_store`` or ``reference_store_large``); the output lies
   outside every store and holds no earlier run or its replicates (a run
   stopped on the pool sizes is rerun in place); the configured gate-P
   donors are distinct, at least two, a fixed default donor among them;
   the mapping seeds hold 0 and another one (D6); and NP5's expected depth
   is given (D8; a family's own profile asset records the primary bundle
   its tables were resolved with). *After PREP, before any gate-P build or
   mapping*
   (``_check_request``, which repeats the checks above): the panel is of
   the species; the bundle has a self-map onto its held-out bundle; the
   gate-P donors are frontal donors of that held-out bundle, its held-out
   donor among them; the replicates are mapped with the self-map's
   recorded MapMyCells worker count (pre-registration §18 item 2: another
   count is another realisation; a bundle PREP builds in the run is mapped
   with the run's count, so only a reused bundle can differ); and the
   held-out test set's spec derived here reproduces the bundle's held-out
   ``build_hash`` (so every leave-one-donor-out build differs from PREP's
   only by the donor); and a family's own profile asset was built from
   tables resolved with PREP's primary bundle (``check_profile_bundle``:
   the same ``build_hash``). A refusal at that stage is recorded in the
   simulation report, which is written before gate P runs.
2. **Reports the per-donor pool sizes** (pre-registration §23.10, open item
   2; ``reference.ho_donor_pool_sizes``, from the reference metadata only)
   and NP2's weak and collapsed parents (PREP's ``bundle.json``, reference
   data only; ``np2_acceptance``) before any leave-one-donor-out build.
   Gate P stops here (``status`` ``stopped``), before any replicate is
   mapped, when under D2 (d) a donor's own pool cannot meet the per-class
   top-up rule, unless the user chose (d) as it stands
   (``accept_small_pools``) or the fallback (c) (``other_region="shared"``);
   and, outside a dry run, when PREP lists a weak or collapsed parent that
   the user has not accepted (``accepted_parents``), unless
   ``run_with_unaccepted_parents`` (the user's ruling B2 (b) of 2026-10-07,
   pre-registration §23.19: any parent other than those accepted comes back
   to the user before gate P runs; an NP2 that stays pending cannot pass,
   and a gate-P output is never re-scored in place).
3. **Builds each extra donor's held-out bundle** in the request's store
   (M13 D11 (b): the separate gate-P store the command is pointed at)
   through a config override (``resolvability.holdout_donor``; a new
   ``build_hash`` per donor). Under D2 (d) the override also sets
   ``holdout_other_region_donor_only`` and the ``gate_p_excluded_test_cells``
   source (the default held-out test set's cells), so each donor draws its
   own other-region top-up and the replicates are disjoint. Every
   self-map test set leaves out the other-region COP cells (M8 D1, extended
   to every human gate-P set by M13 D1 (a); ``reference.self_map_test_cells``).
4. **Fixes C_P from the test cells before any cell is mapped**
   (``gate_p.gate_p_class_sets`` on each group's test-cell truths; the
   default donor's check half).
5. **Simulates and maps** every emission member at mapping seeds 0 and 1
   (D6: seed 1 re-maps seed 0's simulated cells), each member's NP6 stress
   members and the clean upper bound at seed 0, against each donor's own
   held-out bundle with the WHB cells rules (the COP rule); the default
   donor's seed-0 base and clean rows are PREP's own (the frozen run). Each
   replicate's cells table is written to parquet as soon as it is mapped
   and dropped from memory (``replicates/<donor>/seed<k>/<member>.parquet``);
   scoring reads one member at a time.
6. **Re-runs for identity** (NP9; CHECK K13): one seed-0 replicate per
   emission member (the default donor's, compared with PREP's stored rows)
   and PREP's bundle (rebuilt in a scratch store, its tables and test set
   compared).
7. **Scores** NP3-NP7 per emission member (``gate_p`` C1-C6) under the
   criteria revision the user approved on 2026-10-07 (pre-registration
   §23.21: R1, R2, R3 (c), R5, R6, R7; ``gate_p.GATE_P_SCORED_READINGS``),
   with one depth walk per member over NP3-NP7 (``gate_p.gate_p_depth_walk``),
   NP1, NP2, NP8 and NP9, assembles the family (``gate_p.assemble_gate_p``)
   and writes the NP1-NP9 report under ``<out_dir>/gate_p`` with every
   criterion table (``gate_p.write_gate_p_report``) and ``gate_p_run.json``
   (the run's record). With ``dry_run`` the seeded family's dry-run rule is
   scored too (``gate_p.dry_run_verdict``; M13 D28; R5).

Resolvability version 7 forced on a version-6 family (M13 D5 (c) and CHECK
K11, pre-registration §21 (vii)'s diagnostic path) simulates the version-7
emission members on the bundle's own test set and engine, as
``simulate.run_v7_diagnostic`` does, freezes their ensemble decisions and
scores the family as a version-7 family; nothing is written to a store.

Readings this driver takes where §14 and the decisions are not explicit
(each listed in the run record's ``readings``; put to the user in
pre-registration §23.17 and ruled on 2026-10-07, §23.19, the version-7 NP5
reading revised the same day by §23.21 R7):

- C_P is counted on every test cell of each group's test set (after the D1
  drop; the default donor's check half), the "test-cell table" of §14,
  including the few cells whose native counts reach no grid depth.
- NP5 in a version-7 family (revision R7, replacing the C8 reading): the
  ensemble's emission is re-derived in each replicate
  (``gate_p.np5_rederive_ensemble``, with the bundle's lineage) and compared
  with the frozen ensemble decisions, the same table for every member; R6
  and R2 apply to it. R3 (c)'s consequence check reads the ensemble's t*
  re-fitted per replicate at every bin of a tested set
  (``gate_p.np5_ensemble_set_thresholds``, pre-registration §23.22). Each
  member's own re-derivation against its own base decisions (the default
  donor at seed 0) is reported only (``np5_agreement__<member>``).
- NP9's measured time is PREP (the primary and its held-out test set, from
  their ``bundle.json`` timings) plus this run's leave-one-donor-out builds
  and replicates (not the identity re-runs); the simulated cells that scale
  the version-7 time reference (M13 D10 (a)) are every scored replicate's,
  the default donor's seed-0 rows included whether PREP or this run
  simulated them (``np9_time_reference``). A version-7 family's reference
  is D10 (a)'s only: a stated version-6 reference is not used.
- The per-donor pool rule (``reference.ho_donor_pool_sizes``): a judged
  class meets it when the donor's own pools hold
  ``topup_min_class_test_cells`` (200) cells.
- NP2's accepted parents (``np2_acceptance``): an accepted entry names a
  weak or collapsed parent by its lookup key (``<level>/<node>``), its node
  label or its node's name in the bundle's vocab snapshot, so the user's
  names match the keys PREP records; an entry matching two parents is
  refused, one matching none is reported (``unmatched``).
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, cast

import numpy as np
import pandas as pd

from merxen.annotation import gate_p as gp
from merxen.annotation import resolvability as res
from merxen.annotation.simulate import (
    GatePPlan,
    GatePRequest,
    GatePUnavailableError,
    SimulationError,
)
from merxen.annotation.vocab import Species

if TYPE_CHECKING:
    from merxen.annotation.config import AnnotationConfig, AnnotationReferenceSpec
    from merxen.annotation.mapmycells_engine import MmcBundle
    from merxen.annotation.simulate import GatePHook, GatePPrecheck
    from merxen.annotation.store import ReferenceStore

logger = logging.getLogger(__name__)

GATE_P_RUN_SCHEMA_VERSION: Final = 1
GATE_P_RUN_DIR: Final = "gate_p"
GATE_P_RUN_JSON: Final = "gate_p_run.json"
POOL_SIZES_CSV: Final = "gate_p_pool_sizes.csv"
REPLICATES_DIR: Final = "replicates"
PRIMARY_REFERENCE: Final = "whb_frontal_supc_clus"
STATUS_SCORED: Final = "scored"
# NP5's depth source when a family's own gate_p_profile asset gives it.
FAMILY_PROFILE_SOURCE: Final = "family_profile_asset"
STATUS_STOPPED: Final = "stopped"
# Why a run stopped before any leave-one-donor-out build (``stop_reasons``).
STOP_POOL_SIZES: Final = "pool_sizes"
STOP_NP2_PARENTS: Final = "np2_unaccepted_parents"
# Pre-registration §23.2 D2: (d) each extra donor draws its own other-region
# cells (the default); (c) the fallback, the production pool shared by every
# replicate, each cell counted once in pooled sets.
OTHER_REGION_DONOR_OWN: Final = "donor_own"
OTHER_REGION_SHARED: Final = "shared"
OTHER_REGION_MODES: Final[tuple[str, ...]] = (
    OTHER_REGION_DONOR_OWN,
    OTHER_REGION_SHARED,
)
# The regime gate P freezes (§14: the provisional emission rule and margins).
GATE_P_REGIME: Final[res.Regime] = "provisional"
# The readings this driver takes (module docstring), ruled on 2026-10-07
# (pre-registration §23.19; NP5's version-7 reading revised by §23.21 R7);
# listed in every run record.
GATE_P_RUN_READINGS: Final[tuple[str, ...]] = (
    "C8 C_P: every test cell of each group's test set (after the D1 drop; the "
    "default donor's check half), cells reaching no grid depth included",
    "NP5 (version 7, §23.21 R7): the ensemble re-derived per replicate against "
    "the frozen ensemble decisions, the same for every member; each member's "
    "own re-derivation is reported only",
    "C8 NP9: PREP from its bundle.json timings plus the leave-one-donor-out "
    "builds and replicates (identity re-runs left out); the version-7 time "
    "reference scales by every scored replicate's simulated cells, the default "
    "donor's seed-0 rows included whoever simulated them (D10 (a))",
    "C8 D2 pool sizes: a judged class meets the per-class top-up rule when "
    "the donor's own pools hold topup_min_class_test_cells (200) cells",
    "B2 NP2 parents: an accepted entry matches a weak or collapsed parent by "
    "its lookup key, node label or node name; outside a dry run an unaccepted "
    "parent stops gate P before any build (ruling B2 (b), 2026-10-07)",
)


class GatePRunError(SimulationError):
    """Gate P cannot run on this request (refused before any gate-P compute)."""


@dataclass(frozen=True)
class GatePOptions:
    """How a gate-P run is set up (M13; pre-registration §23.9, §23.10).

    Attributes:
        species: The species selected when the check started (M13 D4: a
            required argument; a human family is dry-run on set a only).
        other_region: ``donor_own`` (D2 (d), the default) or ``shared``
            (the fallback (c), the user's choice after the pool sizes).
        accept_small_pools: Run under D2 (d) although a donor's own pool
            cannot meet the per-class top-up rule (the user chose (d) as it
            stands with the sizes in view); otherwise gate P stops there.
        resolvability_version: ``auto`` (the bundle's), or ``7`` to score a
            version-6 family's version-7 ensemble (M13 D5 (c), the §21 (vii)
            diagnostic path; never written to a store).
        dry_run: Score the seeded family's dry-run rule (M13 D28): broad for
            every class of C_P and supercluster for the classes H18 expects.
        dry_run_exemptions: The (level, class) pairs the user already removed
            from H18's expectation (pre-registration §18 C1: supercluster
            COP); any further narrowing is a loosening.
        time_reference_seconds: NP9's time reference of a version-6 family
            (§8.7 / §10), stated by the caller; never used for a version-7
            family, whose reference is D10 (a)'s (``np9_time_reference``).
        time_reference_basis: Where it comes from (reported).
        dry_run_seconds: The set a version-7 dry run's measured time (the
            ``measured_seconds`` of its ``gate_p_run.json``), for NP9's
            version-7 reference (M13 D10 (a)).
        dry_run_simulated_cells: That dry run's ``simulated_cells`` (every
            scored replicate's simulated cells, ``np9_time_reference``).
        unresolved_reviewed: The user reviewed NP1's unresolved list in the
            gate-P PR.
        accepted_parents: NP2's weak or collapsed parents the user accepted
            (none by default): each by its lookup key, node label or node
            name (``np2_acceptance``). They are passed before the run: a
            gate-P output is never re-scored in place.
        run_with_unaccepted_parents: Run although PREP lists a weak or
            collapsed parent the user has not accepted (NP2 then stays
            pending, so the family cannot pass); otherwise a run that is not
            a dry run stops before any leave-one-donor-out build (ruling B2
            (b) of 2026-10-07).
        partners: NP8's intended partner panels and whether each one's
            intersection passed NP3 at broad (none: not applicable, D26).
        prep_identity: Rebuild PREP's bundle in a scratch store and compare
            it (NP9 identity, CHECK K13); without it NP9 is not evaluable.
        x1_factors: The X1 factor table (M13 D7 (b): reported beside NP6's
            offsets; optional).
        overwrite: Replace an earlier gate-P run's report in the output
            directory (refused by default).
    """

    species: Species
    other_region: str = OTHER_REGION_DONOR_OWN
    accept_small_pools: bool = False
    resolvability_version: Literal["auto", 7] = "auto"
    dry_run: bool = False
    dry_run_exemptions: tuple[tuple[str, str], ...] = (("supercluster", "COP"),)
    time_reference_seconds: float | None = None
    time_reference_basis: str = ""
    dry_run_seconds: float | None = None
    dry_run_simulated_cells: int | None = None
    unresolved_reviewed: bool = False
    accepted_parents: tuple[str, ...] = ()
    run_with_unaccepted_parents: bool = False
    partners: Mapping[str, bool | None] | None = None
    prep_identity: bool = True
    x1_factors: Path | None = None
    overwrite: bool = False

    def __post_init__(self) -> None:
        """Validate the options.

        Raises:
            GatePRunError: For an unknown species, other-region mode or
                version, a non-positive time, or a dry-run time without its
                simulated cells (or the reverse).
        """
        if self.species not in ("human", "mouse"):
            raise GatePRunError(
                f"gate P needs the species (human or mouse), got {self.species!r}"
            )
        if self.other_region not in OTHER_REGION_MODES:
            raise GatePRunError(
                f"other_region must be one of {OTHER_REGION_MODES}, got "
                f"{self.other_region!r}"
            )
        if self.resolvability_version not in ("auto", 7):
            raise GatePRunError(
                "resolvability_version must be 'auto' or 7, got "
                f"{self.resolvability_version!r}"
            )
        for name in ("time_reference_seconds", "dry_run_seconds"):
            value = getattr(self, name)
            if value is not None and not value > 0:
                raise GatePRunError(f"{name} must be > 0, got {value!r}")
        if (self.dry_run_seconds is None) != (self.dry_run_simulated_cells is None):
            raise GatePRunError(
                "the version-7 time reference needs both the dry run's seconds "
                "and its simulated cells (M13 D10 (a))"
            )
        if (
            self.dry_run_simulated_cells is not None
            and self.dry_run_simulated_cells < 1
        ):
            raise GatePRunError("dry_run_simulated_cells must be >= 1")

    def check_supported(self) -> None:
        """Refuse a species whose gate-P path is not built yet (before compute).

        Raises:
            GatePUnavailableError: For mouse: the second disjoint WMB test
                draw and the ag7 / VZG2 dry run come with M13 C20-C21, after
                M6b.
        """
        if self.species == "mouse":
            raise GatePUnavailableError(
                "gate P for a mouse family needs the second disjoint WMB test draw "
                "and the ag7 / VZG2 dry run (M13 C20-C21, after M6b); only the "
                "human path is available"
            )

    def to_json(self) -> dict[str, Any]:
        """Return the options as JSON-native values."""
        return {
            "species": self.species,
            "other_region": self.other_region,
            "accept_small_pools": self.accept_small_pools,
            "resolvability_version": self.resolvability_version,
            "dry_run": self.dry_run,
            "dry_run_exemptions": [list(item) for item in self.dry_run_exemptions],
            "time_reference_seconds": self.time_reference_seconds,
            "time_reference_basis": self.time_reference_basis,
            "dry_run_seconds": self.dry_run_seconds,
            "dry_run_simulated_cells": self.dry_run_simulated_cells,
            "unresolved_reviewed": self.unresolved_reviewed,
            "accepted_parents": list(self.accepted_parents),
            "run_with_unaccepted_parents": self.run_with_unaccepted_parents,
            "partners": None if self.partners is None else dict(self.partners),
            "prep_identity": self.prep_identity,
            "x1_factors": None if self.x1_factors is None else str(self.x1_factors),
            "overwrite": self.overwrite,
        }


def gate_p_hook(options: GatePOptions) -> GatePHook:
    """Return the gate-P programme to register with ``simulate.register_gate_p_hook``.

    Register it with ``precheck=gate_p_precheck(options)``, so that what
    needs no bundle is refused before any compute.

    Args:
        options: The run's options (the species is required, M13 D4).

    Returns:
        ``request -> run record`` (``run_gate_p`` with these options).

    Raises:
        GatePUnavailableError: For a species without a gate-P path yet.
    """
    options.check_supported()

    def hook(request: GatePRequest) -> dict[str, Any]:
        return run_gate_p(request, options)

    return hook


# --------------------------------------------------------------------------
# The base run and the groups


@dataclass
class _Group:
    """One held-out donor: its bundle, test cells, engine and mapper."""

    name: str
    bundle_dir: Path
    build_hash: str
    test: res.HeldOutCells
    exclusion: dict[str, Any] | None
    engine: MmcBundle
    specs: list[res.LevelSpec]
    map_fn: res.MapFunction
    runs: list[dict[str, Any]]
    manifest: dict[str, Any]


@dataclass
class _Base:
    """What the frozen base run fixes (§14: thresholds from the default donor)."""

    bundle_dir: Path
    version: int
    forced_v7: bool
    tables: res.ResolvabilityTables
    levels: list[res.LevelMeta]
    grid: list[int]
    settings: res.RuleSettings
    ensemble: res.EnsembleSettings | None
    emission: list[res.EnsembleMember]
    clean: res.EnsembleMember
    neuronal: dict[str, bool | None]
    decisions: pd.DataFrame = field(repr=False)

    @property
    def is_v7(self) -> bool:
        """Whether the frozen decisions are a version-7 ensemble's."""
        return self.version == res.RESOLVABILITY_VERSION_V7

    def selector(self, member: res.EnsembleMember) -> tuple[str | None, str | None]:
        """Return ``(recipe, member)`` that select a member's rows."""
        if self.is_v7:
            return None, member.name
        return member.recipe.name, None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _manifest(bundle_dir: Path) -> dict[str, Any]:
    from merxen.annotation.store import BUNDLE_MANIFEST_NAME

    return _read_json(bundle_dir / BUNDLE_MANIFEST_NAME)


def _inside(path: Path, roots: Iterable[Path]) -> bool:
    resolved = path.resolve()
    for root in roots:
        base = Path(root).resolve()
        if resolved == base or base in resolved.parents:
            return True
    return False


def _gate_levels(species: str) -> tuple[str, ...]:
    return gp.gate_p_levels(species)


def _primary_build(request: GatePRequest) -> Any:
    for item in request.builds:
        if item.spec.reference_id == PRIMARY_REFERENCE:
            return item
    raise GatePRunError(
        f"gate P needs the prepared {PRIMARY_REFERENCE} spec of the base "
        "simulation (simulate.GatePRequest.builds)"
    )


def _mapping_workers(summary: Mapping[str, Any]) -> set[int]:
    runs = summary.get("mapping_runs") or (summary.get("provenance") or {}).get(
        "mapping_runs"
    )
    workers: set[int] = set()
    for run in runs or []:
        value = run.get("n_processors") if isinstance(run, Mapping) else None
        if value is not None:
            workers.add(int(value))
    return workers


def _check_species(species: str, config_species: str, options: GatePOptions) -> None:
    """Refuse a run of another species than the one selected (M13 D4)."""
    options.check_supported()
    if species != options.species or config_species != options.species:
        raise GatePRunError(
            f"gate P was started for {options.species} (M13 D4: the species is "
            f"selected when the check starts), but the panel is {species} and "
            f"the config {config_species}"
        )


def _check_store(store: ReferenceStore, config: AnnotationConfig) -> None:
    """Refuse a store that lies in a production store (M13 D11 (b)).

    Gate P builds its leave-one-donor-out bundles in the store it is given,
    which must be the separate gate-P store: no root of it may be, or lie
    inside, the config's ``reference_store`` or ``reference_store_large``
    (the production stores, written only by the user-started runs).

    Raises:
        GatePRunError: If a store root is or lies inside a production store.
    """
    production = [
        Path(root)
        for root in (config.reference_store, config.reference_store_large)
        if root is not None
    ]
    for root in store.roots:
        if production and _inside(Path(root), production):
            raise GatePRunError(
                f"gate P would build its bundles in {root}, a production reference "
                f"store ({', '.join(str(path) for path in production)}); point "
                "--store (and --store-large) at the separate gate-P store (M13 D11 "
                "(b))"
            )


def _check_out_dir(out: Path, store: ReferenceStore, options: GatePOptions) -> None:
    """Refuse an output inside a store or holding an earlier run.

    Raises:
        GatePRunError: If ``out`` lies inside a store root, holds an earlier
            gate-P run that did not stop on the pool sizes (unless
            ``overwrite``), or holds an earlier run's replicates.
    """
    if _inside(out, store.roots):
        raise GatePRunError(
            f"gate P writes to {out}, inside a reference store; reports never go "
            "into a store"
        )
    # A run stopped on the pool sizes wrote no replicate: the user's choice
    # runs in the same place.
    earlier = _read_json(out / GATE_P_RUN_JSON).get("status")
    if (
        (out / GATE_P_RUN_JSON).exists()
        and earlier != STATUS_STOPPED
        and not options.overwrite
    ):
        raise GatePRunError(
            f"{out} already holds a gate-P run ({GATE_P_RUN_JSON}); give another "
            "output directory or overwrite it explicitly"
        )
    replicates = out / REPLICATES_DIR
    if replicates.is_dir() and any(replicates.rglob("*.parquet")):
        raise GatePRunError(
            f"{replicates} holds the replicates of an earlier gate-P run; give "
            "another output directory (a replicate is never written twice)"
        )


def _check_donors_and_seeds(config: AnnotationConfig) -> tuple[list[str], list[int]]:
    """Check the configured gate-P donors and mapping seeds (no bundle needed).

    Returns:
        ``(donors, seeds)``: the donors in config order, the seeds sorted.

    Raises:
        GatePRunError: If a donor is named twice, fewer than two donors are
            named, a fixed ``holdout_donor`` is not among them, or the seeds
            do not hold 0 and another seed (D6).
    """
    resolvability = config.resolvability
    donors = [str(donor) for donor in resolvability.gate_p_human_donors]
    if len(set(donors)) != len(donors):
        raise GatePRunError(
            f"gate_p_human_donors names a donor twice ({donors}): each held-out "
            "donor is one group of replicates, never a duplicated replicate"
        )
    if len(donors) < 2:
        raise GatePRunError(
            f"gate P needs at least one other donor than the default (NP4 compares "
            f">= 2 groups); gate_p_human_donors is {donors}"
        )
    default = str(resolvability.holdout_donor)
    if default != "auto" and default not in donors:
        raise GatePRunError(
            f"the default held-out donor {default!r} (the frozen thresholds' donor) "
            f"is not among the gate-P donors {donors}"
        )
    seeds = sorted({int(seed) for seed in resolvability.gate_p_seeds})
    if 0 not in seeds or len(seeds) < 2:
        raise GatePRunError(
            f"gate_p_seeds must hold 0 (the frozen run) and another mapping seed "
            f"(D6), got {seeds}"
        )
    return donors, seeds


def precheck_gate_p(plan: GatePPlan, options: GatePOptions) -> None:
    """Refuse, before any compute, a gate-P run that needs no bundle to refuse.

    ``run_panel_simulation`` calls it (``gate_p_precheck``) before the panel
    is computed or PREP builds anything: the species (M13 D4), the store
    (D11 (b): not a production store), the output (outside every store,
    without an earlier run or its replicates), the configured donors and
    mapping seeds (D6), and NP5's expected-depth source (D8, CHECK K7; a
    family profile must record its primary bundle). What needs the built
    bundle (its self-map, held-out donor and frontal donors, the recorded
    worker count, the held-out ``build_hash``, a family profile's primary
    ``build_hash``) is checked by ``run_gate_p`` after PREP and before any
    gate-P build or mapping.

    Args:
        plan: What the simulation was started with.
        options: The run's options.

    Raises:
        GatePRunError: For each refusal above.
        GatePUnavailableError: For a species without a gate-P path yet.
    """
    _check_species(plan.species, plan.config.species, options)
    _check_store(plan.store, plan.config)
    _check_out_dir(Path(plan.out_dir) / GATE_P_RUN_DIR, plan.store, options)
    _check_donors_and_seeds(plan.config)
    _, _, depth_record, _ = np5_depth_source(
        expected_depth=plan.expected_depth,
        depth_profile=plan.depth_profile,
        depth_profile_asset=plan.depth_profile_asset,
        species=options.species,
    )
    check_profile_bundle(depth_record, None)


def gate_p_precheck(options: GatePOptions) -> GatePPrecheck:
    """Return the precheck to register beside ``gate_p_hook(options)``.

    Args:
        options: The run's options.

    Returns:
        ``plan -> None`` (``precheck_gate_p`` with these options).
    """

    def precheck(plan: GatePPlan) -> None:
        precheck_gate_p(plan, options)

    return precheck


def _check_request(request: GatePRequest, options: GatePOptions) -> dict[str, Any]:
    """Check the request before any gate-P compute; return the run's facts.

    The checks of ``precheck_gate_p`` run again here, then those that need
    the built bundle.

    Raises:
        GatePRunError: For every refusal of the module docstring's step 1.
    """
    from merxen.annotation import reference as ref

    _check_species(request.panel.species, request.config.species, options)
    _check_store(request.store, request.config)
    if request.scratch_dir is None:
        raise GatePRunError("gate P needs the simulation's scratch directory")
    out = Path(request.out_dir) / GATE_P_RUN_DIR
    _check_out_dir(out, request.store, options)
    donors, seeds = _check_donors_and_seeds(request.config)
    bundle = request.bundles.get(PRIMARY_REFERENCE)
    if bundle is None:
        raise GatePRunError(
            f"gate P scores the primary reference {PRIMARY_REFERENCE}, which the "
            f"base simulation did not build (built: {sorted(request.bundles)}; "
            f"simulation status {request.report.get('status')!r})"
        )
    bundle_dir = Path(bundle)
    tables = res.load_resolvability(bundle_dir, allow_version_7=True)
    if tables is None:
        raise GatePRunError(
            f"{bundle_dir} has no resolvability self-map: gate P needs the frozen "
            "thresholds of the base run (M13 D13 (a))"
        )
    summary = tables.summary
    engine = summary.get("engine") or {}
    if engine.get("self", False):
        raise GatePRunError(
            f"{bundle_dir}: the self-map mapped onto the bundle itself, not onto "
            "its held-out bundle; gate P maps each donor onto its own held-out "
            "bundle"
        )
    test_dir = Path(str((summary.get("test_set_bundle") or {}).get("path") or ""))
    if not test_dir.is_dir():
        raise GatePRunError(f"{bundle_dir}: the held-out test-set bundle is missing")
    test_output = _manifest(test_dir).get("builder_output") or {}
    test_set = test_output.get("test_set") or {}
    default_donor = test_set.get("holdout_donor")
    donor_cells = {
        str(key): int(value)
        for key, value in (test_set.get("donor_cells") or {}).items()
    }
    if not default_donor:
        raise GatePRunError(
            f"{test_dir}: the held-out bundle records no held-out donor"
        )
    missing = sorted(set(donors) - set(donor_cells))
    if missing:
        raise GatePRunError(
            f"the gate-P donors {missing} have no frontal WHB cell (frontal donors "
            f"of {test_dir.name[:12]}: {sorted(donor_cells)})"
        )
    if default_donor not in donors:
        raise GatePRunError(
            f"the default held-out donor {default_donor!r} (the frozen thresholds' "
            f"donor) is not among the gate-P donors {donors}"
        )
    extras = [donor for donor in donors if donor != default_donor]
    workers = _mapping_workers(summary)
    current = ref.prep_resources().n_processors
    if workers and workers != {current}:
        raise GatePRunError(
            f"the self-map was mapped with {sorted(workers)} MapMyCells workers and "
            f"this run would use {current}: another worker count is another "
            "realisation (pre-registration §18 item 2); set --n-processors to the "
            "recorded count"
        )
    if (
        options.resolvability_version == 7
        and tables.version == res.RESOLVABILITY_VERSION_V7
    ):
        logger.info("gate P: the bundle is version 7 already; scoring it as built")
    return {
        "bundle_dir": bundle_dir,
        "tables": tables,
        "test_dir": test_dir,
        "test_build_hash": str(
            (summary.get("test_set_bundle") or {}).get("build_hash")
        ),
        "default_donor": str(default_donor),
        "donor_cells": donor_cells,
        "extras": extras,
        "out": out,
        "workers": sorted(workers),
        "seeds": seeds,
    }


def check_profile_bundle(
    depth_record: Mapping[str, Any], bundle_dir: Path | None
) -> None:
    """Refuse a family's NP5 profile resolved with another primary bundle.

    A ``gate_p_profile`` asset (``scripts/annotation/build_np5_depth_profile.py``)
    records the primary bundle the family's ``map_first`` tables were
    resolved with. Gate P scores the decisions of its own PREP bundle, which
    must be the production bundle that run used (the family's gate-P
    preconditions), so the two ``build_hash`` values must be equal.
    ``precheck_gate_p`` calls it before any compute (``bundle_dir`` ``None``:
    the asset must record a bundle) and ``run_gate_p`` after PREP and before
    any gate-P build or mapping. Other depth sources pass.

    Args:
        depth_record: ``np5_depth_source``'s record.
        bundle_dir: PREP's primary bundle (``None``: not built yet).

    Raises:
        GatePRunError: When the asset records no primary bundle, or another.
    """
    if depth_record.get("source") != FAMILY_PROFILE_SOURCE:
        return
    recorded = depth_record.get("primary_build_hash")
    if not recorded:
        raise GatePRunError(
            f"{depth_record.get('depth_profile_asset')} records no primary "
            "build_hash: gate P cannot check that the family's profile and its "
            "own PREP come from one bundle"
        )
    if bundle_dir is None:
        return
    built = str(_manifest(bundle_dir).get("build_hash") or bundle_dir.name)
    if str(recorded) != built:
        raise GatePRunError(
            f"{depth_record.get('depth_profile_asset')} was built from tables "
            f"resolved with the primary bundle {str(recorded)[:16]}, but gate P's "
            f"PREP built {built[:16]}: the gate-P store's PREP must be the "
            "production bundle the family's map_first run used, so rebuild the "
            "store or the asset"
        )


def _np5_depths(
    request: GatePRequest, species: str
) -> tuple[dict[str, Any], float | list[float] | None, dict[str, Any], Any]:
    """NP5's expected depth from the base simulation's settings.

    Returns:
        See ``np5_depth_source``.

    Raises:
        GatePRunError: Without any expected depth.
    """
    settings = request.report.get("settings") or {}
    path = settings.get("depth_profile")
    asset_id = settings.get("depth_profile_asset")
    return np5_depth_source(
        expected_depth=settings.get("expected_depth"),
        depth_profile=None if path is None else Path(str(path)),
        depth_profile_asset=None if asset_id is None else str(asset_id),
        species=species,
    )


def np5_depth_source(
    *,
    expected_depth: float | None,
    depth_profile: Path | None,
    depth_profile_asset: str | None,
    species: str,
) -> tuple[dict[str, Any], float | list[float] | None, dict[str, Any], Any]:
    """NP5's expected depth per class and its source (D8; CHECK K7).

    A registered per-class ``profile`` asset gives each class's depths and
    the overall median for the others; otherwise the label-free pooled median
    (``--expected-depth``, or a pooled profile's median). A per-class CSV is
    never read for NP5: only a registered, frozen asset may carry real
    labels into gate P (pre-registration §23.9 item 6). Reading the profile
    needs no compute, so ``precheck_gate_p`` calls this before PREP.

    A family's own ``gate_p_profile`` asset (M13 D8 with CHECK K7,
    ``sim_inputs.FamilyDepthProfile``; built by
    ``scripts/annotation/build_np5_depth_profile.py``) is read with the
    readings of pre-registration §23.20: a class with at least
    ``sim_inputs.PROFILE_MIN_CLASS_CELLS`` (100) confident broad calls takes
    their depths, and every other class (fewer calls, none, or supercluster
    COP, which has no broad key) the median of the confident broad calls of
    every class; the profile returned for NP3's report-only depth histogram
    is every table cell of the family, label-free.

    Args:
        expected_depth: ``--expected-depth``.
        depth_profile: ``--depth-profile`` (a CSV).
        depth_profile_asset: ``--depth-profile-asset`` (a registered asset).
        species: The family's species.

    Returns:
        ``(per class, default, record, profile)``; the profile (any kind) also
        gives the report-only depth histogram.

    Raises:
        GatePRunError: Without any expected depth.
    """
    from merxen.annotation import sim_inputs as si
    from merxen.annotation import simulate as sim

    asset_id = depth_profile_asset
    path = None if depth_profile is None else str(depth_profile)
    expected = expected_depth
    profile = sim.load_simulation_profile(depth_profile, asset_id, species=species)
    record: dict[str, Any] = {
        "depth_profile_asset": asset_id,
        "depth_profile": path,
        "expected_depth": expected,
    }
    asset = None if asset_id is None else si.get_asset(asset_id)
    if asset is not None and asset.role == si.GATE_P_PROFILE_ROLE:
        family = si.family_profile_from_asset(asset)
        per_class = family.class_depths()
        default = family.overall_median()
        if default is None:
            raise GatePRunError(
                f"{asset_id} holds no confident broad call: NP5 has no overall "
                "median (pre-registration §23.20)"
            )
        run_facts = asset.provenance.get("run") or {}
        record.update(
            {
                "source": FAMILY_PROFILE_SOURCE,
                "sha256": asset.sha256,
                # The primary bundle the family's tables were resolved with;
                # gate P's own PREP must be that bundle (check_profile_bundle).
                "primary_reference": run_facts.get("primary_reference"),
                "primary_build_hash": run_facts.get("primary_build_hash"),
                "min_class_cells": family.min_cells,
                "classes_own_depths": sorted(per_class),
                "classes_below_min_cells": family.below_min_cells(),
                "overall_median": default,
                "table_cells_median": family.table_median(),
                "n_table_cells": int(family.table_totals.size),
                "n_confident_broad": int(family.confident.n_cells),
            }
        )
        return per_class, default, record, family.table_profile()
    if profile is not None and asset_id is not None and not profile.pooled:
        per_class = {
            str(cls): [float(value) for value in values]
            for cls, values in profile.by_class.items()
        }
        default = float(np.median(profile.totals)) if profile.n_cells else None
        record["source"] = "profile_asset_per_class"
        return per_class, default, record, profile
    if expected is not None:
        record["source"] = "expected_depth"
        return {}, float(expected), record, profile
    if profile is not None and profile.pooled and profile.n_cells:
        record["source"] = "pooled_profile_median"
        return {}, float(np.median(profile.totals)), record, profile
    raise GatePRunError(
        "gate P's NP5 needs the family's expected depth: --depth-profile-asset (a "
        "registered per-class profile, M13 D8 / CHECK K7) or --expected-depth (the "
        "label-free pooled median); a per-class CSV is never read for NP5"
    )


def _depth_histogram(profile: Any, grid: Sequence[int]) -> dict[int, float] | None:
    """A depth profile's cell mass per grid bin (NP3, report-only)."""
    if profile is None or not profile.n_cells:
        return None
    bins = res.depth_bin(np.asarray(profile.totals, dtype=np.float64), grid)
    values = pd.Series(bins).dropna().astype(int).value_counts()
    return {int(depth): float(count) for depth, count in values.items()}


def _base(
    request: GatePRequest,
    options: GatePOptions,
    facts: Mapping[str, Any],
    default: _Group,
    runner: _Runner,
) -> _Base:
    """The frozen base run: its version, members, grid, rules and decisions."""
    from merxen.annotation import reference as ref
    from merxen.annotation import sim_inputs as si

    config = request.config
    tables: res.ResolvabilityTables = facts["tables"]
    species = options.species
    levels_all = list(tables.levels)
    gate_levels = set(_gate_levels(species))
    clean = res.EnsembleMember(
        res.member_recipe(res.CLEAN_RECIPE, 0, config.resolvability), "reported"
    )
    chemistry = runner.chemistry
    lung = si.get_asset(si.STRESS_HUMAN_LUNG) if chemistry == "xenium_prime" else None
    if tables.version == res.RESOLVABILITY_VERSION_V7:
        names = [str(name) for name in tables.summary.get("emission_members") or []]
        emission = res.members_from_names(
            names, config.resolvability, stress_table=lung, role="emission"
        )
        ensemble = res.EnsembleSettings.from_json(
            tables.summary.get("ensemble_settings")
        )
        decisions = tables.decisions()
        return _Base(
            bundle_dir=facts["bundle_dir"],
            version=res.RESOLVABILITY_VERSION_V7,
            forced_v7=False,
            tables=tables,
            levels=[meta for meta in levels_all if meta.level in gate_levels],
            grid=list(tables.depth_grid),
            settings=tables.settings,
            ensemble=ensemble,
            emission=emission,
            clean=clean,
            neuronal=dict(tables.summary.get("neuronal_classes") or {}),
            decisions=_gate_rows(decisions, gate_levels),
        )
    if options.resolvability_version != 7:
        base = res.EnsembleMember(
            res.member_recipe(res.DECISION_RECIPE, 0, config.resolvability), "emission"
        )
        return _Base(
            bundle_dir=facts["bundle_dir"],
            version=res.RESOLVABILITY_VERSION_V6,
            forced_v7=False,
            tables=tables,
            levels=[meta for meta in levels_all if meta.level in gate_levels],
            grid=list(tables.depth_grid),
            settings=tables.settings,
            ensemble=None,
            emission=[base],
            clean=clean,
            neuronal={},
            decisions=_gate_rows(tables.decisions(), gate_levels),
        )
    # M13 D5 (c): version 7 forced on a version-6 family (the §21 (vii)
    # diagnostic path): the version-7 emission members on the bundle's own
    # test set and engine, frozen as the ensemble decides them.
    spec = runner.primary_spec
    grid = res.v7_depth_grid(species, request.panel.n_genes, spec.depth_grid)
    runner.grid = list(grid)
    r1_seeds = config.resolvability.ensemble_r1_seeds
    members = [
        member
        for member in res.ensemble_members(
            config.resolvability,
            species=species,
            chemistry=chemistry,
            member_table=None,
            r1_seeds=None if r1_seeds is None else tuple(r1_seeds),
        )
        if member.role == "emission"
    ]
    settings = ref.self_map_rule_settings(config)
    ensemble = res.EnsembleSettings.from_config(config.resolvability)
    frames = [
        runner.load(default, member, 0, version7=True) for member in [*members, clean]
    ]
    cells = pd.concat(frames, ignore_index=True)
    levels = [spec_item.meta for spec_item in default.specs]
    lineage = res.neuronal_classes(cells, species)
    decided = res.decide_v7(
        cells,
        levels,
        list(grid),
        settings,
        ensemble,
        members=[*members, clean],
        neuronal=lineage,
    )
    return _Base(
        bundle_dir=facts["bundle_dir"],
        version=res.RESOLVABILITY_VERSION_V7,
        forced_v7=True,
        tables=tables,
        levels=[meta for meta in levels if meta.level in gate_levels],
        grid=list(grid),
        settings=settings,
        ensemble=ensemble,
        emission=members,
        clean=clean,
        neuronal=dict(lineage),
        decisions=_gate_rows(decided.decisions, gate_levels),
    )


def _gate_rows(frame: pd.DataFrame, levels: Iterable[str]) -> pd.DataFrame:
    """Keep the rows of gate P's levels (the fine levels are report-only)."""
    keep = frame["level"].astype(str).isin(list(levels)).to_numpy(bool)
    return frame[keep].reset_index(drop=True)


# --------------------------------------------------------------------------
# Simulating and mapping the replicates


class _Runner:
    """Simulate, map and stream every replicate of the run to parquet."""

    def __init__(
        self,
        request: GatePRequest,
        *,
        out: Path,
        species: str,
        version7: bool,
        grid: Sequence[int],
    ) -> None:
        from merxen.annotation import reference as ref
        from merxen.annotation import sim_inputs as si

        self.request = request
        self.out = out
        self.species = species
        self.version7 = version7
        self.grid = list(grid)
        self.primary_spec: AnnotationReferenceSpec = _primary_build(request).spec
        self.rules = ref.cells_rules_for(PRIMARY_REFERENCE, request.config)
        self.levels = set(_gate_levels(species))
        self.chemistry = si.resolve_chemistry(
            request.panel.ensembl_ids,
            species=species,
            platform=request.report.get("platform"),
            declared=str(getattr(request.config.panel, "panel_chemistry", "auto")),
        ).chemistry
        self.records: list[dict[str, Any]] = []
        self.wall_s = 0.0
        # Cells this run simulated, and those of the replicates taken from
        # PREP's bundle: their sum is NP9's per-cell basis (D10 (a)), the
        # same for a version-7 family (whose default-donor seed-0 rows are
        # PREP's) and for the forced version-7 dry run (which simulates them).
        self.simulated_cells = 0
        self.prep_simulated_cells = 0

    def path(self, group: str, member: res.EnsembleMember, seed: int) -> Path:
        """Where a replicate's cells table is written."""
        return (
            self.out
            / REPLICATES_DIR
            / group
            / f"seed{int(seed)}"
            / f"{res.member_tag(member)}.parquet"
        )

    def map_function(
        self, engine: MmcBundle, group: str, runs: list[dict[str, Any]]
    ) -> res.MapFunction:
        """The production mapper onto one donor's held-out bundle."""
        from merxen.annotation import reference as ref

        scratch = cast(Path, self.request.scratch_dir)
        return cast(
            res.MapFunction,
            ref.mmc_map_function(
                engine,
                spec=self.primary_spec,
                config=self.request.config,
                scratch_dir=scratch / "gate_p_mapping" / group,
                log_dir=self.out / "logs" / group,
                runs=runs,
            ),
        )

    def simulate(
        self,
        group: _Group,
        member: res.EnsembleMember,
        seed: int,
        *,
        version7: bool | None = None,
    ) -> tuple[pd.DataFrame, int]:
        """Simulate one member on a group's test cells and map it at a seed."""
        use_v7 = self.version7 if version7 is None else version7
        tag = f"gate_p_{group.name}_seed{int(seed)}_"
        if use_v7:
            simulation = res.simulate_members(
                group.test,
                specs=group.specs,
                depths=self.grid,
                members=[member],
                map_fn=group.map_fn,
                cells_rules=self.rules,
                mapping_seed=int(seed),
                tag_prefix=tag,
            )
            return simulation.cells, int(simulation.n_simulated[member.name])
        query = res.thin_and_contaminate(group.test, self.grid, member.recipe)
        tidy = group.map_fn(query, f"{tag}{res.member_tag(member)}", int(seed))
        frame = res.level_cells(tidy, query, group.test, group.specs, seed=int(seed))
        for rule in self.rules:
            frame = rule(frame)
        return res.as_stored(frame), int(len(query.obs))

    def write(
        self, group: str, member: res.EnsembleMember, seed: int, frame: pd.DataFrame
    ) -> Path:
        """Write a replicate's gate-P rows (refusing a duplicated replicate)."""
        path = self.path(group, member, seed)
        if path.exists():
            raise GatePRunError(
                f"{path} exists: the replicate {group}/{member.name}/seed {seed} is "
                "written twice (a duplicated replicate)"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = _gate_rows(frame, self.levels)
        res.coerce_cells(rows).to_parquet(path, index=False)
        return path

    def run(
        self,
        group: _Group,
        member: res.EnsembleMember,
        seed: int,
        *,
        version7: bool | None = None,
    ) -> Path:
        """Simulate, map and write one replicate; record its time and cells."""
        started = time.monotonic()
        frame, n_simulated = self.simulate(group, member, seed, version7=version7)
        path = self.write(group.name, member, seed, frame)
        wall = time.monotonic() - started
        self.wall_s += wall
        self.simulated_cells += n_simulated
        self.records.append(
            {
                "group": group.name,
                "member": member.name,
                "role": member.role,
                "seed": int(seed),
                "n_simulated": n_simulated,
                "n_rows": int(len(frame)),
                "wall_s": round(wall, 3),
                "file": str(path.relative_to(self.out)),
                "source": "simulated",
            }
        )
        return path

    def store_rows(
        self,
        group: str,
        member: res.EnsembleMember,
        seed: int,
        frame: pd.DataFrame,
        *,
        n_simulated: int,
    ) -> Path:
        """Write rows taken from PREP's bundle (the frozen base run).

        Their simulated cells (PREP simulated them, in PREP's time) count in
        ``prep_simulated_cells``.
        """
        path = self.write(group, member, seed, frame)
        self.prep_simulated_cells += int(n_simulated)
        self.records.append(
            {
                "group": group,
                "member": member.name,
                "role": member.role,
                "seed": int(seed),
                "n_simulated": int(n_simulated),
                "n_rows": int(len(frame)),
                "wall_s": 0.0,
                "file": str(path.relative_to(self.out)),
                "source": "prep_bundle",
            }
        )
        return path

    @property
    def scaling_cells(self) -> int:
        """Every scored replicate's simulated cells (NP9's D10 (a) basis)."""
        return self.simulated_cells + self.prep_simulated_cells

    def load(
        self,
        group: _Group | str,
        member: res.EnsembleMember,
        seed: int,
        *,
        version7: bool | None = None,
    ) -> pd.DataFrame:
        """Read a replicate's rows (simulating it first when it is missing)."""
        name = group if isinstance(group, str) else group.name
        path = self.path(name, member, seed)
        if not path.exists():
            if isinstance(group, str):
                raise GatePRunError(f"the replicate {path} was never written")
            self.run(group, member, seed, version7=version7)
        return res.restore_labels(pd.read_parquet(path))


def _prep_simulated(
    summary: Mapping[str, Any],
    member: res.EnsembleMember,
    rows: pd.DataFrame,
    *,
    version7: bool,
) -> int:
    """How many cells PREP simulated for a member's seed-0 rows.

    The self-map summary's ``n_simulated_cells`` (keyed by member for
    version 7, by recipe for version 6); without it, the rows' distinct
    simulated cells.
    """
    key = member.name if version7 else member.recipe.name
    recorded = (summary.get("n_simulated_cells") or {}).get(key)
    if recorded is not None:
        return int(recorded)
    return int(rows["sim_id"].astype(str).nunique())


def _member_rows(
    cells: pd.DataFrame, member: res.EnsembleMember, *, version7: bool
) -> pd.DataFrame:
    """A member's seed-0 rows of PREP's stored cells table."""
    seed = cells["seed"].to_numpy() == 0
    if version7:
        rows = cells[
            seed & (cells[res.MEMBER_COLUMN].astype(str) == member.name).to_numpy()
        ]
    else:
        rows = cells[
            seed & (cells["recipe"].astype(str) == member.recipe.name).to_numpy()
        ]
    if rows.empty:
        raise GatePRunError(f"PREP's cells table holds no seed-0 rows of {member.name}")
    return rows.reset_index(drop=True)


def _group(
    runner: _Runner,
    name: str,
    bundle_dir: Path,
    *,
    config: AnnotationConfig,
) -> _Group:
    """Load one donor's held-out bundle: test cells (D1 drop), engine, mapper."""
    from merxen.annotation import reference as ref
    from merxen.annotation.mapmycells_engine import MmcBundle

    test, exclusion = ref.self_map_test_cells(
        res.load_test_cells(bundle_dir), ref.HO_REFERENCE_ID
    )
    engine = MmcBundle.from_dir(bundle_dir)
    runs: list[dict[str, Any]] = []
    return _Group(
        name=name,
        bundle_dir=bundle_dir,
        build_hash=engine.build_hash,
        test=test,
        exclusion=exclusion,
        engine=engine,
        specs=ref.level_specs_for(PRIMARY_REFERENCE, engine, config),
        map_fn=runner.map_function(engine, name, runs),
        runs=runs,
        manifest=_manifest(bundle_dir),
    )


# --------------------------------------------------------------------------
# Leave-one-donor-out builds (pre-registration §23.2 D2)


def _loo_config(
    config: AnnotationConfig, donor: str, *, donor_own: bool
) -> AnnotationConfig:
    """The config of a donor's leave-one-donor-out build (only the override)."""
    resolvability = config.resolvability.model_copy(
        update={
            "holdout_donor": donor,
            "holdout_other_region_donor_only": bool(donor_own),
        }
    )
    return config.model_copy(update={"resolvability": resolvability})


def _loo_spec(
    test_spec: AnnotationReferenceSpec, excluded: Path | None
) -> AnnotationReferenceSpec:
    """The held-out spec of a leave-one-donor-out build (D2 (d): + exclusions)."""
    from merxen.annotation import reference as ref

    if excluded is None:
        return test_spec
    sources = {
        **dict(test_spec.sources),
        ref.SOURCE_GATE_P_EXCLUDED_TEST_CELLS: excluded,
    }
    return test_spec.model_copy(update={"sources": dict(sorted(sources.items()))})


def _build_loo(
    request: GatePRequest,
    test_spec: AnnotationReferenceSpec,
    donor: str,
    *,
    excluded: Path | None,
    default_cells: set[str],
) -> tuple[Path, dict[str, Any]]:
    """Get or build a donor's held-out bundle in the request's store.

    Raises:
        GatePRunError: If the bundle holds out another donor, shares the
            default held-out bundle's build_hash, or (D2 (d)) holds a cell of
            the default held-out test set.
    """
    from merxen.annotation import reference as ref

    donor_own = excluded is not None
    config = _loo_config(request.config, donor, donor_own=donor_own)
    spec = _loo_spec(test_spec, excluded)
    started = time.monotonic()
    bundle = request.store.get_or_build(
        spec, request.panel, builder=ref.builder_for(spec, config), config=config
    )
    wall = time.monotonic() - started
    bundle_dir = Path(bundle.path)
    test_set = (_manifest(bundle_dir).get("builder_output") or {}).get("test_set") or {}
    if str(test_set.get("holdout_donor")) != donor:
        raise GatePRunError(
            f"{bundle_dir} holds out {test_set.get('holdout_donor')!r}, not {donor!r}"
        )
    if donor_own:
        cells = {str(cell) for cell in res.load_test_cells(bundle_dir).obs.index}
        shared = sorted(cells & default_cells)
        if shared:
            raise GatePRunError(
                f"the leave-one-donor-out set of {donor} holds {len(shared)} cells of "
                f"the default held-out test set (e.g. {shared[:3]}): the replicates "
                "must be disjoint (D2 (d))"
            )
    record = {
        "donor": donor,
        "build_hash": bundle.build_hash,
        "path": str(bundle_dir),
        "reused": bool(bundle.reused),
        "wall_s": round(wall, 1),
        "other_region": OTHER_REGION_DONOR_OWN if donor_own else OTHER_REGION_SHARED,
        "n_test_cells": test_set.get("n_cells"),
        "per_source": test_set.get("per_source"),
        "gate_p_leave_one_donor_out": test_set.get("gate_p_leave_one_donor_out"),
        "class_top_up": test_set.get("class_top_up"),
    }
    return bundle_dir, record


# --------------------------------------------------------------------------
# C_P before mapping (§14 class set)


def _test_cell_table(groups: Sequence[_Group], levels: Iterable[str]) -> pd.DataFrame:
    """Each group's test-cell truths per gate-P level (no mapping result)."""
    kept = set(levels)
    frames: list[pd.DataFrame] = []
    for group in groups:
        cell_ids = [str(cell) for cell in group.test.obs.index]
        halves = res.cell_split_half(cell_ids)
        for spec in group.specs:
            if spec.meta.level not in kept:
                continue
            truth = spec.truth(group.test).reindex(cell_ids)
            frames.append(
                pd.DataFrame(
                    {
                        "level": spec.meta.level,
                        "cell_id": cell_ids,
                        "truth": truth["truth"].astype(object).to_numpy(),
                        "truth_parent": truth["truth_parent"].astype(object).to_numpy(),
                        res.TRUTH_LEAF_COLUMN: group.test.obs[res.TRUTH_LEAF_COLUMN]
                        .astype(str)
                        .to_numpy(),
                        "group": group.name,
                        "half": halves,
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def _reference_shares(
    test_cells: pd.DataFrame, composition: Mapping[str, float]
) -> dict[str, dict[str, float]]:
    """Each class's share of the reference composition, per level (reported)."""
    shares: dict[str, dict[str, float]] = {}
    pairs = test_cells[["level", res.TRUTH_LEAF_COLUMN, "truth_parent"]].dropna()
    pairs = pairs.drop_duplicates()
    for level, rows in pairs.groupby("level", sort=True):
        per_class: dict[str, float] = {}
        for leaf, cls in zip(
            rows[res.TRUTH_LEAF_COLUMN].astype(str),
            rows["truth_parent"].astype(str),
            strict=True,
        ):
            per_class[cls] = per_class.get(cls, 0.0) + float(composition.get(leaf, 0.0))
        shares[str(level)] = per_class
    return shares


# --------------------------------------------------------------------------
# Scoring one emission member (gate_p C1-C6)


def _views(
    tables: Mapping[gp.ReplicateKey, pd.DataFrame],
    *,
    default_group: str,
    shared: bool,
    fit_cells: set[str],
) -> tuple[dict[gp.ReplicateKey, pd.DataFrame], dict[gp.ReplicateKey, pd.DataFrame]]:
    """The per-replicate and the pooled views of one simulation's tables.

    D2 (d): both are the tables themselves (disjoint; a fit-half leak raises
    in ``held_out_replicates``). D2 (c): the default donor's fit-half cells
    leave every other group; in the pooled view a cell already held by an
    earlier group (the default donor first) is left out, so each shared cell
    is counted once.
    """
    if not shared:
        return dict(tables), dict(tables)
    per_replicate: dict[gp.ReplicateKey, pd.DataFrame] = {}
    for key, table in tables.items():
        if str(key[0]) == default_group:
            per_replicate[key] = table
            continue
        keep = ~table["cell_id"].astype(str).isin(fit_cells).to_numpy(bool)
        per_replicate[key] = table[keep].reset_index(drop=True)
    pooled: dict[gp.ReplicateKey, pd.DataFrame] = {}
    order = sorted(per_replicate, key=lambda key: (str(key[0]) != default_group, key))
    seen: dict[int, set[str]] = {}
    for key in order:
        table = per_replicate[key]
        held = seen.setdefault(int(key[1]), set())
        cells = table["cell_id"].astype(str)
        if str(key[0]) == default_group:
            check = table["half"].to_numpy() == 1
            held.update(cells.to_numpy()[check])
            pooled[key] = table
            continue
        keep = ~cells.isin(held).to_numpy(bool)
        pooled[key] = table[keep].reset_index(drop=True)
        held.update(cells.to_numpy()[keep])
    return per_replicate, {key: pooled[key] for key in tables}


@dataclass
class _MemberScore:
    """One emission member's per-criterion verdicts and tables."""

    verdicts: dict[str, dict[tuple[str, str], bool | None]]
    depths: pd.DataFrame
    tables: dict[str, pd.DataFrame]


def _score_member(
    *,
    base: _Base,
    member: res.EnsembleMember,
    replicates: Mapping[gp.ReplicateKey, pd.DataFrame],
    stresses: Mapping[str, Mapping[gp.ReplicateKey, pd.DataFrame]],
    clean: Mapping[gp.ReplicateKey, pd.DataFrame],
    stress_members: Sequence[res.EnsembleMember],
    default_group: str,
    shared: bool,
    composition: Mapping[str, float],
    expected_depth: Mapping[str, Any],
    default_depth: float | list[float] | None,
    depth_histogram: Mapping[int, float] | None,
    vocab: pd.DataFrame,
    config: AnnotationConfig,
    species: Species,
    ensemble: _EnsembleNp5 | None = None,
) -> _MemberScore:
    """Score NP3-NP7 of one emission member (§14; every member for version 7).

    Under the criteria revision of pre-registration §23.21: NP3 and NP6 read
    the test-cell weightings (R1); NP5's agreement compares bins where both
    runs hold the minimum (R6) and, in version 7, is the ensemble's
    (``ensemble``, R7), and its t* part is the consequence check (R3 (c);
    version 7 on the ensemble's re-fitted t*); the verdicts and
    ``validated_min_depth`` come from one depth walk over NP3-NP7 (R2). The
    readings they replace are written beside them (NP3's own walk
    ``np3_depths``, the t* spread, each member's own NP5 agreement).
    """
    resolvability = config.resolvability
    recipe, name = base.selector(member)
    default_table = replicates.get((default_group, 0))
    if default_table is None:
        raise GatePRunError(
            f"{member.name}: the default donor's seed-0 replicate is missing"
        )
    fit_cells = set(
        default_table["cell_id"]
        .astype(str)
        .to_numpy()[default_table["half"].to_numpy() == 0]
    )
    per_replicate, pooled = _views(
        replicates, default_group=default_group, shared=shared, fit_cells=fit_cells
    )
    pooled_cells = gp.pooled_held_out_cells(pooled, default_group=default_group, seed=0)
    tested = res.gate_p_tested_sets(
        pooled_cells,
        base.decisions,
        min_confident_n=resolvability.gate_p_min_confident_n,
        regime=GATE_P_REGIME,
        recipe=recipe,
        seed=0,
        member=name,
    )
    decisions = base.decisions
    # NP3.
    np3_settings = gp.Np3Settings.from_config(resolvability)
    np3_stats = gp.np3_set_stats(
        pooled,
        decisions,
        tested,
        default_group=default_group,
        composition=composition,
        settings=np3_settings,
        depth_histogram=depth_histogram,
        regime=GATE_P_REGIME,
        recipe=recipe,
        seed=0,
        member=name,
    )
    np3_verdicts = gp.np3_verdicts(np3_stats, config.thresholds, np3_settings)
    depths = gp.validated_min_depth(np3_verdicts, tested, base.grid)
    # NP4.
    np4_settings = gp.Np4Settings.from_config(resolvability)
    targets = gp.level_targets(config.thresholds, _gate_levels(species))
    np4_stats = gp.replicate_set_stats(
        per_replicate,
        decisions,
        tested,
        default_group=default_group,
        regime=GATE_P_REGIME,
        recipe=recipe,
        member=name,
    )
    np4_sets = gp.np4_set_verdicts(np4_stats, targets, np4_settings)
    np4_seed = gp.np4_seed_stability(
        per_replicate,
        decisions,
        default_group=default_group,
        settings=np4_settings,
        regime=GATE_P_REGIME,
        recipe=recipe,
        member=name,
    )
    # NP5.
    np5_settings = gp.Np5Settings.from_config(resolvability)
    saturated = base.ensemble.saturated_bp_share if base.ensemble is not None else None
    rederived = {
        key: gp.np5_rederive(
            table,
            base.levels,
            base.grid,
            base.settings,
            recipe=recipe,
            member=name,
            saturated_bp_share=saturated,
        )
        for key, table in per_replicate.items()
    }
    # Each member's own re-derivation against its own base decisions: scored
    # in version 6; in version 7 reported only (§23.21 R7).
    np5_member_agreement = gp.np5_decision_agreement(
        rederived[(default_group, 0)], rederived, np5_settings, regime=GATE_P_REGIME
    )
    np5_thresholds = gp.np5_set_thresholds(
        per_replicate,
        tested,
        base.levels,
        base.settings,
        regime=GATE_P_REGIME,
        recipe=recipe,
        member=name,
        saturated_bp_share=saturated,
    )
    # The t* spread is reported only (§23.21 R3 (c)).
    np5_spread = gp.np5_tstar_spread(np5_thresholds, np5_settings)
    if ensemble is not None:
        np5_agreement = ensemble.agreement
        consequence_thresholds = gp.np5_ensemble_set_thresholds(
            ensemble.rederived, tested, base.settings, regime=GATE_P_REGIME
        )
        threshold_from = gp.NP5_TSTAR_FROM_ENSEMBLE
    else:
        np5_agreement = np5_member_agreement
        consequence_thresholds = np5_thresholds
        threshold_from = gp.NP5_TSTAR_FROM_MEMBER
    np5_consequence = gp.np5_tstar_consequence(
        pooled,
        tested,
        consequence_thresholds,
        targets,
        default_group=default_group,
        settings=np5_settings,
        recipe=recipe,
        seed=0,
        member=name,
        threshold_from=threshold_from,
    )
    np5_extrapolated = gp.np5_extrapolated_share(
        decisions,
        expected_depth,
        base.grid,
        np5_settings,
        regime=GATE_P_REGIME,
        default_depth=default_depth,
    )
    # NP6.
    np6_settings = gp.Np6Settings.from_config(resolvability)
    seed0 = {key: table for key, table in pooled.items() if int(key[1]) == 0}
    base_rows = gp.SimulationRows(seed0, recipe=recipe, member=name)
    clean_recipe, clean_name = base.selector(base.clean)

    def pooled_view(
        tables: Mapping[gp.ReplicateKey, pd.DataFrame],
    ) -> dict[gp.ReplicateKey, pd.DataFrame]:
        return _views(
            tables, default_group=default_group, shared=shared, fit_cells=fit_cells
        )[1]

    clean_rows = gp.SimulationRows(
        pooled_view(clean), recipe=clean_recipe, member=clean_name
    )
    np6_frames: list[pd.DataFrame] = []
    stress_names: list[str] = []
    for stress in stress_members:
        stress_recipe, stress_name = base.selector(stress)
        rows = gp.SimulationRows(
            pooled_view(stresses[stress.name]),
            recipe=stress_recipe,
            member=stress_name,
        )
        stress_names.append(rows.name)
        np6_frames.append(
            gp.np6_set_stats(
                base_rows,
                rows,
                decisions,
                tested,
                default_group=default_group,
                settings=np6_settings,
                composition=composition,
                clean=clean_rows,
                regime=GATE_P_REGIME,
                seed=0,
            )
        )
    np6_stats = pd.concat(np6_frames, ignore_index=True)
    np6_verdicts = gp.np6_verdicts(np6_stats, config.thresholds, np6_settings)
    # NP7.
    np7_settings = gp.Np7Settings.from_config(resolvability)
    np7 = gp.np7_error_structure(
        pooled,
        decisions,
        tested,
        default_group=default_group,
        species=species,
        settings=np7_settings,
        vocab=vocab,
        regime=GATE_P_REGIME,
        recipe=recipe,
        seed=0,
        member=name,
    )
    # One depth rule for NP3-NP7 (§23.21 R2).
    verdicts, walk = gp.gate_p_depth_walk(
        gp.CriterionTables(
            np3_verdicts=np3_verdicts,
            np4_sets=np4_sets,
            np4_seed=np4_seed,
            np5_agreement=np5_agreement,
            np5_tstar=np5_consequence,
            np5_extrapolated=np5_extrapolated,
            np6_verdicts=np6_verdicts,
            np6_stresses=tuple(stress_names),
            np7_wrong_node=np7.wrong_node,
            np7_excluded=np7.excluded,
        ),
        tested,
        base.grid,
        np6_settings=np6_settings,
    )
    tables = {
        "depth_walk": walk,
        "np3_verdicts": np3_verdicts,
        # NP3's own walk (the floor before §23.21 R2), reported only.
        "np3_depths": depths,
        "np4_stats": np4_stats,
        "np4_sets": np4_sets,
        "np4_seed": np4_seed,
        # The member's own re-derivation (scored in version 6 only, R7).
        "np5_agreement": np5_member_agreement,
        "np5_thresholds": np5_thresholds,
        "np5_spread": np5_spread,
        "np5_tstar_consequence": np5_consequence,
        "np5_extrapolated": np5_extrapolated,
        "np5_class": gp.np5_class_table(
            np5_agreement, np5_consequence, np5_extrapolated, tested
        ),
        "np6_verdicts": np6_verdicts,
        "np7_wrong_node": np7.wrong_node,
        "tested_sets": _tested_table(tested),
    }
    if ensemble is not None:
        tables["np5_ensemble_thresholds"] = consequence_thresholds
    if np7.excluded is not None:
        tables["np7_excluded"] = np7.excluded
    return _MemberScore(verdicts=verdicts, depths=walk, tables=tables)


def _tested_table(
    tested: Mapping[tuple[str, str], Sequence[res.GatePTestedSet] | None],
) -> pd.DataFrame:
    """The tested sets per (level, class) as a table (reported)."""
    rows: list[dict[str, Any]] = []
    for (level, cls), items in sorted(tested.items()):
        if items is None:
            rows.append(
                {
                    "level": level,
                    "class": cls,
                    "set": None,
                    "pooled": None,
                    "depths": None,
                    "n_confident": 0,
                    "precision": math.nan,
                    "wilson_lb": math.nan,
                }
            )
            continue
        for item in items:
            rows.append(
                {
                    "level": level,
                    "class": cls,
                    "set": gp.tested_set_label(item),
                    "pooled": item.pooled,
                    "depths": ";".join(str(depth) for depth in item.depths),
                    "n_confident": item.n_confident,
                    "precision": item.precision,
                    "wilson_lb": item.wilson_lb,
                }
            )
    return pd.DataFrame.from_records(rows)


@dataclass(frozen=True)
class _EnsembleNp5:
    """The version-7 ensemble's NP5 re-derivation (pre-registration §23.21 R7).

    Attributes:
        rederived: Per (group, seed), the replicate's re-derived ensemble
            decisions (``gate_p.np5_rederive_ensemble``).
        agreement: Their agreement with the frozen ensemble decisions
            (``gate_p.np5_decision_agreement``, R6), NP5's scored agreement
            in every member.
    """

    rederived: dict[gp.ReplicateKey, pd.DataFrame]
    agreement: pd.DataFrame


def _np5_ensemble(
    *,
    base: _Base,
    runner: _Runner,
    groups: Sequence[_Group],
    seeds: Sequence[int],
    shared: bool,
    default_group: str,
) -> _EnsembleNp5 | None:
    """The ensemble's NP5 re-derivation per replicate (version 7; R7, scored)."""
    if not base.is_v7 or base.ensemble is None:
        return None
    names = [member.name for member in base.emission]
    rederived: dict[gp.ReplicateKey, pd.DataFrame] = {}
    default_cells: set[str] = set()
    for group in groups:
        for seed in seeds:
            cells = pd.concat(
                [runner.load(group.name, member, seed) for member in base.emission],
                ignore_index=True,
            )
            if shared and group.name != default_group:
                cells = cells[
                    ~cells["cell_id"].astype(str).isin(default_cells).to_numpy(bool)
                ]
            if group.name == default_group and seed == 0:
                default_cells = set(
                    cells["cell_id"]
                    .astype(str)
                    .to_numpy()[cells["half"].to_numpy() == 0]
                )
            rederived[(group.name, int(seed))] = gp.np5_rederive_ensemble(
                cells,
                base.levels,
                base.grid,
                base.settings,
                base.ensemble,
                members=names,
                neuronal=base.neuronal,
            )
    agreement = gp.np5_decision_agreement(
        base.decisions,
        rederived,
        gp.Np5Settings.from_config(runner.request.config.resolvability),
        regime=GATE_P_REGIME,
    )
    return _EnsembleNp5(rederived=rederived, agreement=agreement)


# --------------------------------------------------------------------------
# Family checks (NP1, NP2, NP8, NP9)


def _np1(request: GatePRequest, options: GatePOptions) -> gp.FamilyCheck:
    from merxen.annotation import diagnostics as diag

    entry = (request.report.get("panel") or {}).get("gene_ids") or {}
    gene_ids = (
        [diag.GeneIdDiagnostics.from_report_entry("gene_list", entry)] if entry else []
    )
    return gp.np1_gene_ids(
        gene_ids,
        list(request.panel.symbols),
        settings=gp.Np1Settings.from_config(request.config.panel),
        unresolved_reviewed=options.unresolved_reviewed,
    )


def np2_inputs(bundle_dir: Path) -> dict[str, Any]:
    """NP2's inputs from the production primary bundle (CHECK K14).

    The root's markers and, per root child, its cells in the bundle's
    reference (the mapping precompute's cells of the child's leaves) and its
    markers in the bundle's lookup (a child without children of its own is
    separated by the root's markers, as ``reference.root_children_with_markers``
    counts it); the weak and auto-collapsed parents ``bundle.json`` records.

    Args:
        bundle_dir: The primary bundle.

    Returns:
        ``root_markers``, ``root_children`` (``gate_p.NP2_ROOT_CHILD_COLUMNS``),
        ``weak_parents``, ``collapsed_parents`` and ``parent_names`` (each
        weak or collapsed parent's node name in the bundle's vocab
        snapshot, ``None`` where it has none).
    """
    from merxen.annotation import reference as ref
    from merxen.annotation.mapmycells_engine import MmcBundle

    engine = MmcBundle.from_dir(bundle_dir)
    tree = engine.tree()
    lookup = ref.read_lookup(engine.lookup)
    first = tree.hierarchy[0]
    root_markers = len(lookup.get(ref.ROOT_KEY, []) or [])
    stats = ref.read_precomputed_stats(engine.mapping_precompute, genes=[])
    n_by_leaf = {
        str(leaf): float(count)
        for leaf, count in zip(stats.leaves, stats.n_cells, strict=True)
    }
    rows: list[dict[str, Any]] = []
    for child in tree.children_of(None, None):
        leaves = tree.leaves_under(first, child)
        markers = (
            root_markers
            if len(tree.children_of(first, child)) <= 1
            else len(lookup.get(ref.lookup_key(first, child), []) or [])
        )
        rows.append(
            {
                "child": str(child),
                "n": int(sum(n_by_leaf.get(str(leaf), 0.0) for leaf in leaves)),
                "n_markers": int(markers),
            }
        )
    markers_output = (_manifest(bundle_dir).get("builder_output") or {}).get(
        "markers"
    ) or {}
    result: dict[str, Any] = {
        "root_markers": root_markers,
        "root_children": pd.DataFrame.from_records(
            rows, columns=list(gp.NP2_ROOT_CHILD_COLUMNS)
        ),
        "weak_parents": [
            str(item) for item in markers_output.get("weak_parents") or []
        ],
        "collapsed_parents": [
            str(item) for item in markers_output.get("collapsed_parents") or []
        ],
    }
    parents = {*result["weak_parents"], *result["collapsed_parents"]}
    result["parent_names"] = _parent_names(engine, sorted(parents))
    return result


def _parent_names(engine: MmcBundle, keys: Sequence[str]) -> dict[str, str | None]:
    """Each lookup key's node name in the bundle's vocab snapshot."""
    from merxen.annotation import reference as ref

    vocab = engine.vocab()
    names: dict[str, str | None] = {}
    for key in keys:
        name: str | None = None
        if key != ref.ROOT_KEY and vocab is not None and "key_name" in vocab:
            level, node = ref.parse_lookup_key(key)
            if (level, node) in vocab.index:
                value = str(vocab.loc[(level, node), "key_name"])
                name = value or None
        names[key] = name
    return names


@dataclass(frozen=True)
class Np2Acceptance:
    """NP2's weak and collapsed parents against the user's accepted entries.

    Attributes:
        parents: The weak or collapsed parents' lookup keys, sorted.
        names: Each parent's node name (``None`` where the vocab has none).
        accepted: The parents an entry accepts.
        unaccepted: The parents no entry accepts.
        matched: Per entry, the parents it names.
        unmatched: The entries that name no listed parent.
    """

    parents: tuple[str, ...]
    names: Mapping[str, str | None]
    accepted: tuple[str, ...]
    unaccepted: tuple[str, ...]
    matched: Mapping[str, list[str]]
    unmatched: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        """Return the record ``gate_p_run.json`` keeps (``np2_parents``)."""

        def named(keys: Iterable[str]) -> list[dict[str, Any]]:
            return [{"key": key, "name": self.names.get(key)} for key in keys]

        return {
            "parents": named(self.parents),
            "accepted": named(self.accepted),
            "unaccepted": named(self.unaccepted),
            "matched": {entry: list(keys) for entry, keys in self.matched.items()},
            "unmatched": list(self.unmatched),
        }


def np2_acceptance(
    np2: Mapping[str, Any], accepted_parents: Sequence[str]
) -> Np2Acceptance:
    """Match the user's accepted entries to NP2's weak and collapsed parents.

    An entry accepts a parent it names by the parent's lookup key
    (``<level>/<node>``), its node label or its node's name (``np2_inputs``
    ``parent_names``), so the user's names match the keys PREP records (the
    user's ruling B2 (b) of 2026-10-07 names Bergmann glia and Upper rhombic
    lip).

    Args:
        np2: ``np2_inputs`` of the primary bundle.
        accepted_parents: The user's entries.

    Returns:
        The acceptance.

    Raises:
        GatePRunError: If an entry names more than one parent.
    """
    from merxen.annotation import reference as ref

    parents = tuple(
        sorted({*np2.get("weak_parents", ()), *np2.get("collapsed_parents", ())})
    )
    names = {key: (np2.get("parent_names") or {}).get(key) for key in parents}

    def labels(key: str) -> set[str]:
        found = {key}
        node = None if key == ref.ROOT_KEY else ref.parse_lookup_key(key)[1]
        for label in (node, names.get(key)):
            if label:
                found.add(str(label))
        return found

    matched: dict[str, list[str]] = {}
    for entry in dict.fromkeys(str(item) for item in accepted_parents):
        hits = [key for key in parents if entry in labels(key)]
        if len(hits) > 1:
            raise GatePRunError(
                f"the accepted parent {entry!r} names more than one weak or "
                f"collapsed parent ({hits}); give their lookup keys"
            )
        matched[entry] = hits
    accepted = tuple(sorted({key for hits in matched.values() for key in hits}))
    return Np2Acceptance(
        parents=parents,
        names=names,
        accepted=accepted,
        unaccepted=tuple(key for key in parents if key not in accepted),
        matched=matched,
        unmatched=tuple(entry for entry, hits in matched.items() if not hits),
    )


def np2_stop_applies(acceptance: Np2Acceptance, options: GatePOptions) -> bool:
    """Whether NP2's parents stop the run before any build (ruling B2 (b)).

    A run that is not a dry run stops when PREP lists a parent the user has
    not accepted, unless ``run_with_unaccepted_parents``. A dry run never
    stops on NP2, which is not part of its rule (M13 D28).
    """
    return bool(
        acceptance.unaccepted
        and not options.dry_run
        and not options.run_with_unaccepted_parents
    )


def np9_time_reference(
    options: GatePOptions, *, version7: bool, forced_v7: bool, simulated_cells: int
) -> tuple[float | None, str]:
    """NP9's time reference and its basis (§14 NP9; M13 D10 (a)).

    - A version-7 family: only D10 (a)'s reference, the set a version-7 dry
      run's time scaled per simulated cell (``dry_run_seconds`` and
      ``dry_run_simulated_cells``). A stated version-6 reference
      (``time_reference_seconds``) is not used, since D10 (a) fixes the
      version-7 reference and any other value would be an unregistered one;
      without the dry run's values the time part is not evaluable.
    - Version 7 forced on a version-6 family (the §21 (vii) diagnostic path,
      the set a dry run itself): the dry run's values when given, else the
      stated version-6 reference.
    - A version-6 family: the stated reference (§8.7 / §10), else none.

    Both counts of the D10 (a) scaling are every scored replicate's simulated
    cells (``_Runner.scaling_cells``), the default donor's seed-0 rows
    included whether PREP (a version-7 family) or the run itself (the forced
    dry run) simulated them, so the dry run and the family share one basis.

    Args:
        options: The run's options.
        version7: Whether the frozen decisions are a version-7 ensemble's.
        forced_v7: Whether version 7 was forced on a version-6 family.
        simulated_cells: This run's scaling basis.

    Returns:
        ``(reference seconds or None, basis)``.
    """
    dry = options.dry_run_seconds
    dry_cells = options.dry_run_simulated_cells
    if version7 and dry is not None and dry_cells is not None and simulated_cells > 0:
        reference = gp.np9_time_reference_v7(
            dry_run_seconds=dry,
            dry_run_simulated_cells=dry_cells,
            family_simulated_cells=simulated_cells,
        )
        return reference, (
            f"set a version-7 dry run: {dry:.0f} s for {dry_cells} simulated cells, "
            f"scaled to {simulated_cells} (M13 D10 (a))"
        )
    if version7 and not forced_v7:
        stated = options.time_reference_seconds
        ignored = (
            ""
            if stated is None
            else f"; the stated version-6 reference ({stated:.0f} s) is not used"
        )
        return None, (
            "a version-7 family's reference is the set a version-7 dry run's time "
            "scaled per simulated cell (M13 D10 (a)); without the dry run's seconds "
            f"and simulated cells the time part is not evaluable{ignored}"
        )
    return options.time_reference_seconds, options.time_reference_basis


def _prep_measurements(
    bundle_dir: Path, test_dir: Path
) -> tuple[float | None, dict[str, float]]:
    """PREP's time and each PREP step's peak RSS from the bundles' records."""
    seconds = 0.0
    found = False
    peaks: dict[str, float] = {}
    for label, directory in (("primary", bundle_dir), ("held_out", test_dir)):
        output = _manifest(directory).get("builder_output") or {}
        timings = output.get("timings_s") or {}
        for value in timings.values():
            if value is not None:
                seconds += float(value)
                found = True
        for step, metrics in (output.get("ctm_steps") or {}).items():
            peak = (metrics or {}).get("peak_rss_gb")
            if peak is not None:
                peaks[f"{label}:{step}"] = float(peak)
        resolvability = output.get("resolvability") or {}
        for run in resolvability.get("mapping_runs") or []:
            peak = (run or {}).get("peak_rss_gb")
            if peak is not None:
                peaks[f"{label}:map_{run.get('tag')}"] = float(peak)
    return (seconds if found else None), peaks


# --------------------------------------------------------------------------
# Identity re-runs (NP9; CHECK K13)


def _same_rows(first: pd.DataFrame, second: pd.DataFrame) -> bool:
    """Whether two cells tables hold the same rows (any order)."""
    if set(first.columns) != set(second.columns) or len(first) != len(second):
        return False
    columns = sorted(first.columns)
    keys = [
        column
        for column in ("recipe", res.MEMBER_COLUMN, "seed", "level", "sim_id")
        if column in columns
    ]

    def ordered(frame: pd.DataFrame) -> pd.DataFrame:
        table = frame[columns].copy()
        for column in keys:
            table[column] = table[column].astype(str)
        return table.sort_values(keys, kind="mergesort").reset_index(drop=True)

    try:
        pd.testing.assert_frame_equal(
            ordered(first),
            ordered(second),
            check_dtype=False,
            check_categorical=False,
            check_exact=True,
        )
    except AssertionError:
        return False
    return True


def _same_frames(first: pd.DataFrame, second: pd.DataFrame) -> bool:
    try:
        pd.testing.assert_frame_equal(
            first.reset_index(drop=True),
            second.reset_index(drop=True),
            check_dtype=False,
            check_categorical=False,
            check_exact=True,
        )
    except AssertionError:
        return False
    return True


# The marker lookup's MapMyCells run record (config with scratch paths,
# timestamp, duration, version): not bundle content (NP9 identity).
_LOOKUP_RUN_RECORD: Final = "metadata"


def bundle_differences(first: Path, second: Path) -> list[str]:
    """Return what differs between two builds of one bundle (NP9 identity).

    §14 NP9: "identical re-run -> identical bundle and table content". The
    compared content is the self-map's tables (``resolvability_cells``,
    ``resolvability``, the class-depth table and the decisions re-derived
    from them), the marker lookup, the mapping tree and the vocab snapshot;
    for a held-out test-set bundle, its test cells (obs, genes and counts).
    Records that hold times, paths or memory (``bundle.json``, the summary's
    runtime and mapping runs, the marker lookup's ``metadata`` block) are not
    content.

    Args:
        first: One bundle directory.
        second: The other.

    Returns:
        The names of the differing items (empty when identical).
    """
    from merxen.annotation import reference as ref

    differences: list[str] = []
    one = res.load_resolvability(first, allow_version_7=True)
    two = res.load_resolvability(second, allow_version_7=True)
    if (one is None) != (two is None):
        differences.append("resolvability")
    elif one is not None and two is not None:
        if not _same_rows(one.cells, two.cells):
            differences.append(res.RESOLVABILITY_CELLS_FILE)
        if not _same_frames(one.decisions(), two.decisions()):
            differences.append("decisions")
        for name in (res.RESOLVABILITY_FILE, res.CLASS_DEPTH_FILE):
            paths = (first / name, second / name)
            if (
                paths[0].is_file() != paths[1].is_file()
                or paths[0].is_file()
                and not _same_frames(
                    pd.read_parquet(paths[0]), pd.read_parquet(paths[1])
                )
            ):
                differences.append(name)
    for name in (ref.QUERY_MARKERS_FILTERED_FILE, ref.MAPPING_TREE_FILE):
        values = [_read_json(directory / name) for directory in (first, second)]
        if name == ref.QUERY_MARKERS_FILTERED_FILE:
            # MapMyCells' run record: the build's scratch paths, worker count,
            # timestamp and duration differ on every re-run.
            values = [
                {key: item for key, item in value.items() if key != _LOOKUP_RUN_RECORD}
                for value in values
            ]
        if values[0] != values[1]:
            differences.append(name)
    vocab = [first / ref.VOCAB_SNAPSHOT_FILE, second / ref.VOCAB_SNAPSHOT_FILE]
    if vocab[0].is_file() != vocab[1].is_file() or (
        vocab[0].is_file()
        and not _same_frames(pd.read_csv(vocab[0]), pd.read_csv(vocab[1]))
    ):
        differences.append(ref.VOCAB_SNAPSHOT_FILE)
    tests = [directory / res.TEST_CELLS_FILE for directory in (first, second)]
    if tests[0].is_file() != tests[1].is_file():
        differences.append(res.TEST_CELLS_FILE)
    elif tests[0].is_file():
        cells = [res.load_test_cells(directory) for directory in (first, second)]
        if (
            cells[0].genes != cells[1].genes
            or not _same_frames(cells[0].obs, cells[1].obs)
            or cells[0].counts.shape != cells[1].counts.shape
            or (cells[0].counts != cells[1].counts).nnz
        ):
            differences.append(res.TEST_CELLS_FILE)
    return differences


def _prep_identity(request: GatePRequest, facts: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild PREP's primary bundle in a scratch store and compare it (K13)."""
    from merxen.annotation.store import ReferenceStore

    build = _primary_build(request)
    scratch = cast(Path, request.scratch_dir)
    root = scratch / "gate_p_identity_store"
    builds = scratch / "gate_p_identity_scratch"
    builds.mkdir(parents=True, exist_ok=True)
    store = ReferenceStore(
        root,
        large_panel_genes=request.store.large_panel_genes,
        scratch_root=builds,
    )
    started = time.monotonic()
    rebuilt = store.get_or_build(
        build.spec, request.panel, builder=build.builder, config=request.config
    )
    wall = time.monotonic() - started
    first = Path(facts["bundle_dir"])
    second = Path(rebuilt.path)
    differences = bundle_differences(first, second)
    summary = _read_json(second / res.RESOLVABILITY_SUMMARY_FILE)
    rebuilt_test = Path(str((summary.get("test_set_bundle") or {}).get("path") or ""))
    if rebuilt_test.is_dir():
        differences += [
            f"held_out:{item}"
            for item in bundle_differences(Path(facts["test_dir"]), rebuilt_test)
        ]
    else:
        differences.append("held_out:missing")
    return {
        "identical": not differences and rebuilt.build_hash == first.name,
        "build_hash": rebuilt.build_hash,
        "same_build_hash": rebuilt.build_hash == first.name,
        "path": str(second),
        "differences": differences,
        "wall_s": round(wall, 1),
    }


# --------------------------------------------------------------------------
# The vocab of the mapped bundles (NP7)


def _union_vocab(groups: Sequence[_Group]) -> pd.DataFrame:
    """The vocab snapshots of every group's engine, each node once.

    Raises:
        GatePRunError: If a node's flags differ between the bundles.
    """
    frames = []
    for group in groups:
        vocab = group.engine.vocab()
        if vocab is None:
            raise GatePRunError(f"{group.bundle_dir} has no vocab snapshot")
        frames.append(vocab)
    union = pd.concat(frames, ignore_index=True).drop_duplicates()
    key = union[["level", "node"]].astype(str)
    if bool(key.duplicated().any()):
        clash = key[key.duplicated()].iloc[0]
        raise GatePRunError(
            f"the held-out bundles' vocab snapshots disagree on {clash['level']}/"
            f"{clash['node']}"
        )
    return union.reset_index(drop=True)


# --------------------------------------------------------------------------
# The run


def run_gate_p(request: GatePRequest, options: GatePOptions) -> dict[str, Any]:
    """Run gate P on one family (M13 chunk C8; plan §14; the module docstring).

    Args:
        request: The base simulation (``simulate.GatePRequest``).
        options: The run's options (the species is required, M13 D4).

    Returns:
        The run record stored under ``gate_p`` in the simulation report:
        ``status`` (``scored``, or ``stopped`` by the pool sizes, D2, or by
        NP2's unaccepted parents, B2; ``stop_reasons``), the
        family's verdict, ``validated_max_level``, the dry-run verdict, the
        files written and the NP9 measurements.

    Raises:
        GatePRunError: For a request gate P refuses (before any gate-P build
            or mapping; ``precheck_gate_p`` refuses what needs no bundle
            before any compute).
        GatePUnavailableError: For a species without a gate-P path yet.
    """
    from merxen.annotation import diagnostics as diag
    from merxen.annotation import reference as ref
    from merxen.annotation import sim_inputs as si
    from merxen.annotation.memory import ProcessTreeSampler

    facts = _check_request(request, options)
    expected_depth, default_depth, depth_record, profile = _np5_depths(
        request, options.species
    )
    check_profile_bundle(depth_record, Path(facts["bundle_dir"]))
    config = request.config
    out: Path = facts["out"]
    out.mkdir(parents=True, exist_ok=True)
    tables: res.ResolvabilityTables = facts["tables"]
    version7 = tables.version == res.RESOLVABILITY_VERSION_V7 or (
        options.resolvability_version == 7
    )
    runner = _Runner(
        request,
        out=out,
        species=options.species,
        version7=version7,
        grid=tables.depth_grid,
    )
    primary = _primary_build(request)
    test_spec = ref.held_out_test_set_spec(primary.spec, config)
    expected_hash = facts["test_build_hash"]
    derived = request.store.prepare_request(
        test_spec,
        request.panel,
        builder=ref.builder_for(test_spec, config),
        config=config,
    ).build_hash
    if derived != expected_hash:
        raise GatePRunError(
            f"the held-out spec derived from {PRIMARY_REFERENCE} has build_hash "
            f"{derived[:16]}, not the bundle's {expected_hash[:16]}: the "
            "leave-one-donor-out builds would differ from PREP's in more than the "
            "donor"
        )
    started = time.monotonic()
    default = _group(runner, facts["default_donor"], facts["test_dir"], config=config)
    default_all = {
        str(cell) for cell in res.load_test_cells(facts["test_dir"]).obs.index
    }
    # The code that ran, as the base simulation recorded it (the command
    # reads an exported tree's COMMIT file, else git); a stopped run keeps it.
    provenance = request.report.get("provenance") or {}
    record: dict[str, Any] = {
        "schema_version": GATE_P_RUN_SCHEMA_VERSION,
        "code_commit": provenance.get("code_commit"),
        "code_commit_source": provenance.get("code_commit_source"),
        "options": options.to_json(),
        "family_id": None
        if request.panel.panel_family is None
        else request.panel.panel_family.family_id,
        "panel_hash": request.panel.panel_hash,
        "bundle": str(facts["bundle_dir"]),
        "bundle_resolvability_version": tables.version,
        "default_donor": facts["default_donor"],
        "extra_donors": list(facts["extras"]),
        "default_test_set": {
            "path": str(facts["test_dir"]),
            "build_hash": expected_hash,
            "self_map_exclusion": default.exclusion,
        },
        "store_roots": [str(root) for root in request.store.roots],
        "np5_depth": depth_record,
        "mapping_workers": facts["workers"],
        "readings": list(GATE_P_RUN_READINGS),
        "readings_ruled": list(gp.GATE_P_READINGS_RULED),
    }
    # B2 (b) of 2026-10-07: NP2's weak and collapsed parents (reference data
    # only), before any leave-one-donor-out build.
    np2 = np2_inputs(Path(facts["bundle_dir"]))
    acceptance = np2_acceptance(np2, options.accepted_parents)
    record["np2_parents"] = acceptance.to_json()
    # D2: the per-donor pool sizes, before any leave-one-donor-out build.
    donor_own = options.other_region == OTHER_REGION_DONOR_OWN
    pools = ref.ho_donor_pool_sizes(
        test_spec.sources,
        donors=facts["extras"],
        excluded_cells=default_all,
        default_test_superclusters=default.test.obs[res.TRUTH_LEAF_COLUMN].astype(str),
        region=config.anatomical_region or ref.WHB_WHOLE_CORTEX_REGION,
        target=int(config.resolvability.topup_min_class_test_cells),
        scratch_dir=cast(Path, request.scratch_dir),
    )
    pools.to_csv(out / POOL_SIZES_CSV, index=False)
    short = pools[pools["judged"].astype(bool) & ~pools["meets"].astype(bool)]
    record["pool_sizes"] = {
        "file": POOL_SIZES_CSV,
        "target": int(config.resolvability.topup_min_class_test_cells),
        "short": _jsonable(short.to_dict(orient="records")),
    }
    stops: dict[str, str] = {}
    if donor_own and not short.empty and not options.accept_small_pools:
        stops[STOP_POOL_SIZES] = (
            f"{len(short)} (donor, class) pools cannot meet the per-class top-up "
            "rule under D2 (d); gate P stops before any replicate is mapped "
            "(pre-registration §23.10 open item 2): the user chooses (d) as it "
            "stands (accept_small_pools) or the fallback (c) (other_region "
            "'shared'; the user's ruling A48 (b) of 2026-10-07)"
        )
    if np2_stop_applies(acceptance, options):
        listed = ", ".join(
            f"{key} ({acceptance.names.get(key) or 'no name'})"
            for key in acceptance.unaccepted
        )
        stops[STOP_NP2_PARENTS] = (
            f"PREP lists weak or collapsed parents the user has not accepted: "
            f"{listed}; NP2 would stay pending, so gate P stops before any "
            "replicate is mapped (the user's ruling B2 (b) of 2026-10-07): the "
            "user accepts them (accepted_parents) or runs with them unaccepted "
            "(run_with_unaccepted_parents)"
        )
    if stops:
        record["status"] = STATUS_STOPPED
        record["stop_reasons"] = list(stops)
        record["reason"] = "; ".join(stops.values())
        _write_json(out / GATE_P_RUN_JSON, record)
        logger.warning("gate P stopped: %s", record["reason"])
        return {
            "status": STATUS_STOPPED,
            "stop_reasons": list(stops),
            "reason": record["reason"],
            "run": str(out / GATE_P_RUN_JSON),
            "pool_sizes": str(out / POOL_SIZES_CSV),
        }
    with ProcessTreeSampler() as sampler:
        # Leave-one-donor-out bundles (D2 (d) or (c)), in the request's store.
        excluded_source = (
            Path(facts["test_dir"]) / res.TEST_CELLS_OBS_FILE if donor_own else None
        )
        loo_records = []
        groups = [default]
        loo_wall = 0.0
        for donor in facts["extras"]:
            loo_started = time.monotonic()
            bundle_dir, loo = _build_loo(
                request,
                test_spec,
                donor,
                excluded=excluded_source,
                default_cells=default_all,
            )
            loo_wall += time.monotonic() - loo_started
            group = _group(runner, donor, bundle_dir, config=config)
            loo["self_map_exclusion"] = group.exclusion
            loo_records.append(loo)
            groups.append(group)
        record["leave_one_donor_out"] = loo_records
        # C_P from the test cells, before any cell is mapped.
        test_cells = _test_cell_table(groups, _gate_levels(options.species))
        composition = ref.whb_frontal_composition(
            test_spec.sources, cast(Path, request.scratch_dir)
        )
        class_sets = gp.gate_p_class_sets(
            test_cells,
            species=options.species,
            default_group=default.name,
            settings=gp.ClassSetSettings.from_config(config.resolvability),
            reference_shares=_reference_shares(test_cells, composition),
            disjoint=donor_own,
        )
        base = _base(request, options, facts, default, runner)
        record["resolvability_version"] = base.version
        record["forced_version_7"] = base.forced_v7
        record["members"] = [member.name for member in base.emission]
        stress_by_member = res.gate_p_stress_members(
            base.emission,
            config.resolvability,
            species=options.species,
            chemistry=runner.chemistry,
            xplatform_table=si.get_asset(si.STRESS_HUMAN_XPLATFORM),
            lung_table=si.get_asset(si.STRESS_HUMAN_LUNG)
            if runner.chemistry == "xenium_prime"
            else None,
        )
        seeds: list[int] = facts["seeds"]
        # The default donor's seed-0 base and clean rows are PREP's own.
        if not base.forced_v7:
            for member in [*base.emission, base.clean]:
                rows = _member_rows(tables.cells, member, version7=base.is_v7)
                runner.store_rows(
                    default.name,
                    member,
                    0,
                    rows,
                    n_simulated=_prep_simulated(
                        tables.summary, member, rows, version7=base.is_v7
                    ),
                )
        for group in groups:
            for member in base.emission:
                for seed in seeds:
                    if not runner.path(group.name, member, seed).exists():
                        runner.run(group, member, seed)
                for stress in stress_by_member[member.name]:
                    runner.run(group, stress, 0)
            if not runner.path(group.name, base.clean, 0).exists():
                runner.run(group, base.clean, 0)
        gate_p_seconds = time.monotonic() - started
        # NP9 identity: one seed-0 replicate per emission member (K13).
        replicate_identical: dict[str, bool] = {}
        for member in base.emission:
            first = runner.load(default.name, member, 0)
            frame, _ = runner.simulate(default, member, 0)
            second = _gate_rows(frame, runner.levels)
            second = res.restore_labels(res.coerce_cells(second))
            replicate_identical[member.name] = _same_rows(first, second)
        prep: dict[str, Any] = (
            _prep_identity(request, facts)
            if options.prep_identity
            else {"identical": None, "reason": "not run (prep_identity off)"}
        )
    peak = sampler.peak()
    record["replicates"] = runner.records
    record["replicate_identity"] = replicate_identical
    record["prep_identity"] = prep
    # Score every emission member (one at a time from the parquet files).
    vocab = _union_vocab(groups)
    histogram = _depth_histogram(profile, base.grid)
    clean_tables = {
        (group.name, 0): runner.load(group.name, base.clean, 0) for group in groups
    }
    verdicts: dict[str, dict[str, Any]] = {
        name: {} for name in gp.GATE_P_CLASS_CRITERIA
    }
    depths: dict[str, pd.DataFrame] = {}
    report_tables: dict[str, pd.DataFrame] = {}
    # Version 7: the ensemble's NP5 re-derivation, scored in every member
    # (pre-registration §23.21 R7), before the members are scored.
    ensemble_np5 = _np5_ensemble(
        base=base,
        runner=runner,
        groups=groups,
        seeds=seeds,
        shared=not donor_own,
        default_group=default.name,
    )
    if ensemble_np5 is not None:
        report_tables["np5_ensemble_agreement"] = ensemble_np5.agreement
    for member in base.emission:
        replicates = {
            (group.name, int(seed)): runner.load(group.name, member, seed)
            for group in groups
            for seed in seeds
        }
        stresses = {
            stress.name: {
                (group.name, 0): runner.load(group.name, stress, 0) for group in groups
            }
            for stress in stress_by_member[member.name]
        }
        score = _score_member(
            base=base,
            member=member,
            replicates=replicates,
            stresses=stresses,
            clean=clean_tables,
            stress_members=stress_by_member[member.name],
            default_group=default.name,
            shared=not donor_own,
            composition=composition,
            expected_depth=expected_depth,
            default_depth=default_depth,
            depth_histogram=histogram,
            vocab=vocab,
            config=config,
            species=options.species,
            ensemble=ensemble_np5,
        )
        for criterion, values in score.verdicts.items():
            verdicts[criterion][member.name] = values
        depths[member.name] = score.depths
        tag = res.member_tag(member).lower()
        for name, frame in score.tables.items():
            report_tables[f"{name}__{tag}"] = frame
        del replicates, stresses
    report_tables["pool_sizes"] = pools
    if options.x1_factors is not None:
        x1 = pd.read_csv(options.x1_factors)
        factors = si.xplatform_factors(si.get_asset(si.STRESS_HUMAN_XPLATFORM))
        for member in base.emission:
            per_gene, factor_summary = gp.np6_factor_report(
                request.panel.ensembl_ids, factors, x1, seed=int(member.recipe.seed)
            )
            report_tables[f"np6_factors__{res.member_tag(member).lower()}"] = per_gene
            record.setdefault("np6_factor_report", {})[member.name] = factor_summary
    # Family checks.
    prep_seconds, prep_peaks = _prep_measurements(
        Path(facts["bundle_dir"]), Path(facts["test_dir"])
    )
    measured = (
        None if prep_seconds is None else float(prep_seconds) + float(gate_p_seconds)
    )
    reference_seconds, basis = np9_time_reference(
        options,
        version7=base.is_v7,
        forced_v7=base.forced_v7,
        simulated_cells=runner.scaling_cells,
    )
    engine_markers = (default.manifest.get("builder_output") or {}).get("markers") or {}
    prefilter = (
        (request.report.get("references") or {}).get(PRIMARY_REFERENCE) or {}
    ).get("prefilter_comparison") or {}
    np9 = gp.np9_resources(
        gp.Np9Inputs(
            n_genes=int(request.panel.n_genes),
            wall_seconds=measured,
            reference_seconds=reference_seconds,
            reference_basis=basis,
            peak_rss_gb=prep_peaks,
            rss_reserve_gb=ref.prep_resources().max_gb / ref.PREP_MAX_GB_FRACTION,
            bundle_identical=None
            if prep.get("identical") is None
            else bool(prep["identical"]),
            replicate_identical=replicate_identical,
            prefiltered=bool(engine_markers.get("prefilter")),
            prefilter_verdict=prefilter if prefilter.get("status") == "run" else None,
        ),
        members=[member.name for member in base.emission],
        settings=gp.Np9Settings.from_config(config.panel),
    )
    family_checks = {
        "NP1": _np1(request, options),
        "NP2": gp.np2_panel_coverage(
            root_markers=np2["root_markers"],
            root_children=np2["root_children"],
            weak_parents=np2["weak_parents"],
            collapsed_parents=np2["collapsed_parents"],
            settings=gp.Np2Settings.from_config(config.panel),
            # The parents the user's entries name, by their lookup keys (an
            # entry may name one by its node label or name), and the entries
            # that name none, which NP2 reports as accepted_not_listed.
            accepted_parents=(*acceptance.accepted, *acceptance.unmatched),
        ),
        "NP8": gp.np8_cross_panel(options.partners),
        "NP9": np9,
    }
    family = request.panel.panel_family
    family_id = (
        family.family_id
        if family is not None
        else diag.own_family_id(
            options.species, request.panel.platforms, request.panel.panel_hash
        )
    )
    result = gp.assemble_gate_p(
        family_id=family_id,
        panel_hash=request.panel.panel_hash,
        species=options.species,
        resolvability_version=base.version,
        verdicts=verdicts,
        depths=depths,
        class_sets=class_sets,
        family_checks=family_checks,
    )
    dry_run = None
    if options.dry_run:
        dry_run = gp.dry_run_verdict(
            result,
            h18_classes=gp.h18_expected_classes(
                test_cells[test_cells["group"].astype(str) == default.name],
                species=options.species,
            ),
            exemptions=options.dry_run_exemptions,
        )
    written = gp.write_gate_p_report(
        result, out, tables=report_tables, overwrite=options.overwrite
    )
    record.update(
        {
            "status": STATUS_SCORED,
            "family_id": family_id,
            "passes": result.passes,
            "reasons": list(result.reasons),
            "validated_max_level": result.validated_max_level,
            "dry_run": dry_run,
            "measured_seconds": measured,
            "prep_seconds": prep_seconds,
            "gate_p_seconds": round(gate_p_seconds, 1),
            "leave_one_donor_out_seconds": round(loo_wall, 1),
            "replicate_seconds": round(runner.wall_s, 1),
            "simulated_cells": runner.scaling_cells,
            "simulated_cells_this_run": runner.simulated_cells,
            "simulated_cells_prep_rows": runner.prep_simulated_cells,
            "time_reference_seconds": reference_seconds,
            "time_reference_basis": basis,
            "process_tree_peak": peak.to_json(),
            "files": {name: str(path) for name, path in written.items()},
        }
    )
    _write_json(out / GATE_P_RUN_JSON, record)
    logger.info(
        "gate P %s: %s (validated_max_level %s)",
        family_id,
        "passes" if result.passes else "does not pass",
        result.validated_max_level,
    )
    return {
        "status": STATUS_SCORED,
        "family_id": family_id,
        "passes": result.passes,
        "reasons": list(result.reasons),
        "validated_max_level": result.validated_max_level,
        "dry_run": dry_run,
        "run": str(out / GATE_P_RUN_JSON),
        "report": str(out / gp.GATE_P_REPORT_TXT),
        "measured_seconds": measured,
        "simulated_cells": runner.scaling_cells,
    }


def _jsonable(value: Any) -> Any:
    """A value with paths, numpy scalars and non-finite floats made JSON-safe."""
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = sorted(value, key=str) if isinstance(value, (set, frozenset)) else value
        return [_jsonable(item) for item in items]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_jsonable(dict(payload)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "FAMILY_PROFILE_SOURCE",
    "GATE_P_RUN_DIR",
    "GATE_P_RUN_JSON",
    "OTHER_REGION_DONOR_OWN",
    "OTHER_REGION_SHARED",
    "POOL_SIZES_CSV",
    "STOP_NP2_PARENTS",
    "STOP_POOL_SIZES",
    "GatePOptions",
    "GatePRunError",
    "Np2Acceptance",
    "bundle_differences",
    "check_profile_bundle",
    "gate_p_hook",
    "gate_p_precheck",
    "np2_acceptance",
    "np2_inputs",
    "np2_stop_applies",
    "np5_depth_source",
    "np9_time_reference",
    "precheck_gate_p",
    "run_gate_p",
]

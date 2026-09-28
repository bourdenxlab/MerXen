"""Simulation-input assets derived from public vendor data (plan §4.7, §8.3; M3c).

Your decision of 2026-09-28 (OD-E1 amended, D-G9) lets public vendor data
supply **simulation inputs** — per-gene factor tables and per-class depth
profiles — stored with provenance, never as trust evidence: they never
promote a family or enter a gate, and real data stay downgrade-only (§8.8).
This module is the registry of those inputs and the recipes that read them
(resolvability version 7, §8.3 v7.4-v7.5):

* **Assets** (``assets/annotation/sim_inputs/``): small derived CSVs (each
  < 1 MB), each with a provenance sidecar ``<asset_id>.provenance.json``
  (source dataset and URLs, sha256 of every source file, licence note, XOA
  version and segmentation, deriving script and its sha256, evidence, date,
  ``role``, ``trust_effect: none``). ``load_registry`` verifies every
  asset's sha256 against its sidecar and refuses an incomplete provenance.
  ``asset_hashes`` gives the sha256 values a version-7 bundle's
  ``build_hash`` holds; version-6 bundles use none. Roles: ``member`` (a
  per-gene factor table of an ensemble member), ``profile`` (per-cell total
  counts and called class of a public section), ``stress`` (a cross-tissue
  ratio of a stress recipe), ``scenario`` (a pooled depth histogram, reported
  only) and ``panel_list`` (the pinned public Prime gene lists that resolve a
  panel's chemistry).
* **R3_measured_HO** (``r3_efficiency``; table rules ``restricted``, the
  production rule, and ``all_measured``, phase 1's D3 vector for the
  regression of pre-registration §14 (ii)). Every draw is keyed by the gene
  id (``draw_key``), so a gene's value does not depend on the other panel
  genes (up to the common median normalisation).
* **R1_xtissue_lung_stress** (``xtissue_stress_efficiency``): the R1@0
  efficiency times the lung 5K / v1 per-area ratio, a cross-tissue
  approximation reported only (§8.3 v7.4, §8.10).
* **Depth profiles** (``DepthProfile``): per-class totals with the pooled
  neuronal and non-neuronal fallbacks of phase 1's D-recipes
  (``5k_real/sim/scripts/simlib.py``); no depth prior crosses species (no
  mouse profile or kappa for human, SYNTHESIS §5.3).
* **Chemistry** (``resolve_chemistry``): MERSCOPE → ``merscope``; a Xenium
  panel with Jaccard ≥ 0.95 to a pinned public Prime list → ``xenium_prime``;
  a declared value; else ``unknown`` (no measured table).

Needs numpy, pandas and pydantic only (GPU-env safe).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from merxen.annotation.vocab import ASSET_DIR, Species

logger = logging.getLogger(__name__)

SIM_INPUTS_SCHEMA_VERSION: Final = 1
SIM_INPUTS_DIR: Final[Path] = ASSET_DIR / "sim_inputs"
SIDECAR_SUFFIX: Final = ".provenance.json"
NOTICE_FILE: Final = "NOTICE"
# Repository assets stay small derived tables (Agents.md); anything larger is
# a store asset with its sha256 in build_hash.
MAX_ASSET_BYTES: Final = 1_000_000
TRUST_EFFECT: Final = "none"

AssetRole = Literal["member", "profile", "stress", "scenario", "panel_list"]
Chemistry = Literal["xenium_prime", "xenium_v1", "merscope", "unknown"]
DeclaredChemistry = Literal["auto", "xenium_prime", "xenium_v1", "merscope"]
TableRule = Literal["restricted", "all_measured"]
ASSET_ROLES: Final[tuple[str, ...]] = (
    "member",
    "profile",
    "stress",
    "scenario",
    "panel_list",
)
CHEMISTRIES: Final[tuple[str, ...]] = (
    "xenium_prime",
    "xenium_v1",
    "merscope",
    "unknown",
)
TABLE_RULES: Final[tuple[str, ...]] = ("restricted", "all_measured")

# Provenance every sidecar must carry (OD-E1 amended, §4.7).
REQUIRED_PROVENANCE: Final[tuple[str, ...]] = (
    "source_dataset",
    "source_page",
    "source_files",
    "derived_from",
    "licence",
    "licence_note",
    "attribution",
    "xoa_version",
    "segmentation",
    "deriving_script",
    "deriving_script_sha256",
    "method",
    "evidence",
    "date",
)
LICENCE_NOTE: Final = (
    "licence text taken from search-indexed pages; re-confirm in a browser "
    "before redistributing"
)

# The M3c assets (plan §4.7).
EFFICIENCY_MOUSE_PRIME: Final = "efficiency__xenium_prime__mouse__wmb10xv3"
PROFILE_MOUSE_PRIME_FF: Final = "depth__xenium_prime__mouse_brain_ff"
STRESS_HUMAN_LUNG: Final = "ratio__xenium_prime_vs_v1__human_lung_ffpe"
SCENARIO_HUMAN_LUNG: Final = "depth__xenium_prime__human_lung_ffpe"
PRIME_PANEL_LISTS: Final = "panels__xenium_prime"

# R3_measured_HO (pre-registered, plan §8.3 v7.4 and pre-registration §14.3;
# only tightenable).
R3_RESIDUAL_SD_LOG2: Final = 0.20
R3_CAP_LOG2: Final = 3.0
R3_MIN_EXPECTED_COUNTS: Final = 1000.0
R3_MIN_TOP_CLASS_SHARE: Final = 0.005
# The D3 regression vector (table rule all_measured): the measured 5K factor
# SD of the informative genes, for genes without information.
D3_UNINFORMATIVE_SD_LOG2: Final = 1.365
STREAM_MEASURED_RESIDUAL: Final = "measured_residual"
STREAM_MEASURED_RESAMPLE: Final = "measured_resample"
TIER_INFORMATIVE: Final = "informative"
TIER_WEAK: Final = "weak"
# R1_xtissue_lung_stress (plan §8.3 v7.4): the lung 5K / v1 per-area ratio,
# centred and capped at +-2 log2; unmeasured genes a keyed LogNormal(0, 0.702),
# the measured lung spread (5k_real/review/REVIEW_CHECKS.txt C2).
XTISSUE_CAP_LOG2: Final = 2.0
XTISSUE_SD_LN: Final = 0.702
STREAM_XTISSUE: Final = "xtissue_lung_ratio"
# Depth profiles (phase 1's D-recipes): a class with fewer profile cells takes
# the pooled neuronal or non-neuronal profile.
PROFILE_MIN_CLASS_CELLS: Final = 100
POOL_NEURONAL: Final = "__neurons__"
POOL_NON_NEURONAL: Final = "__nonneurons__"
POOL_ALL: Final = "__all__"
# Chemistry resolution (plan §3.7 panel_chemistry).
CHEMISTRY_MIN_JACCARD: Final = 0.95
# Human broad classes of the resolvability levels that are neuronal
# (``vocab.HUMAN_FLOOR_CLASSES``).
HUMAN_NEURONAL_CLASSES: Final[frozenset[str]] = frozenset(
    {"Exc", "Inh", "OtherNeuron", "Neurons"}
)

EFFICIENCY_COLUMNS: Final[tuple[str, ...]] = (
    "gene_id",
    "tier",
    "log2_factor",
    "expected_counts",
    "observed_counts",
    "top_class",
    "top_class_section_share",
)
PROFILE_COLUMNS: Final[tuple[str, ...]] = ("class_code", "total_counts")
RATIO_COLUMNS: Final[tuple[str, ...]] = ("gene_id", "log2_ratio")
SCENARIO_COLUMNS: Final[tuple[str, ...]] = ("total_counts", "n_cells")
PANEL_LIST_COLUMNS: Final[tuple[str, ...]] = ("panel_key", "species", "gene_id")
ROLE_COLUMNS: Final[dict[str, tuple[str, ...]]] = {
    "member": EFFICIENCY_COLUMNS,
    "profile": PROFILE_COLUMNS,
    "stress": RATIO_COLUMNS,
    "scenario": SCENARIO_COLUMNS,
    "panel_list": PANEL_LIST_COLUMNS,
}


class SimInputError(RuntimeError):
    """A simulation-input asset is missing, altered or incomplete."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class SimInputAsset(BaseModel):
    """One registered simulation-input asset (its provenance sidecar).

    Attributes:
        schema_version: Sidecar schema version.
        asset_id: Registry id (the file stem).
        asset_version: Content version; a re-measure (e.g. phase 1b) is a new
            version with a new sha256.
        file: CSV file name next to the sidecar.
        sha256: sha256 of the CSV.
        size: Bytes of the CSV.
        n_rows: Data rows of the CSV.
        role: ``member``, ``profile``, ``stress``, ``scenario`` or
            ``panel_list``.
        species: ``human`` or ``mouse`` (``None``: several, per row, as the
            panel lists).
        chemistry: The platform chemistry measured (``xenium_prime``).
        tissue: Tissue and preservation (``brain_ff``, ``lung_ffpe``).
        reference: The reference the factors are measured against
            (``wmb_10xv3``), when any.
        label: The label reports print (e.g. "XOA 3.0 vendor segmentation,
            public 10x section").
        description: What the table holds and how a recipe reads it.
        columns: Column descriptions.
        extra: Role-specific metadata (the profile's class legend).
        trust_effect: Always ``none``: an asset never raises a trust state.
        provenance: ``REQUIRED_PROVENANCE`` fields and more.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int
    asset_id: str
    asset_version: int = Field(ge=1)
    file: str
    sha256: str
    size: int = Field(ge=0)
    n_rows: int = Field(ge=0)
    role: AssetRole
    species: Species | None
    chemistry: Chemistry
    tissue: str
    reference: str | None = None
    label: str
    description: str
    columns: dict[str, str]
    extra: dict[str, Any] = Field(default_factory=dict)
    trust_effect: Literal["none"]
    provenance: dict[str, Any]

    def path(self, directory: Path | str | None = None) -> Path:
        """Return the CSV path (in ``directory`` or the packaged directory)."""
        return Path(directory or SIM_INPUTS_DIR) / self.file

    def hash_payload(self) -> dict[str, Any]:
        """Return what a version-7 ``build_hash`` records of the asset."""
        return {
            "asset_id": self.asset_id,
            "asset_version": self.asset_version,
            "role": self.role,
            "sha256": self.sha256,
        }


def validate_asset(asset: SimInputAsset, directory: Path | str | None = None) -> None:
    """Check one asset against its file and the provenance rules.

    Args:
        asset: The sidecar record.
        directory: The asset directory (default: the packaged one).

    Raises:
        SimInputError: If the file is missing, larger than ``MAX_ASSET_BYTES``
            or altered (sha256, size), its columns differ from the role's, or
            the provenance lacks a required field.
    """
    path = asset.path(directory)
    if not path.is_file():
        raise SimInputError(f"sim input {asset.asset_id}: {path} is missing")
    size = path.stat().st_size
    if size > MAX_ASSET_BYTES:
        raise SimInputError(
            f"sim input {asset.asset_id}: {size} bytes exceed the repository "
            f"limit of {MAX_ASSET_BYTES} (ship it as a store asset instead)"
        )
    digest = _sha256_file(path)
    if digest != asset.sha256 or size != asset.size:
        raise SimInputError(
            f"sim input {asset.asset_id}: {path.name} has sha256 {digest[:16]} and "
            f"{size} bytes, the sidecar records {asset.sha256[:16]} and "
            f"{asset.size} (re-run scripts/annotation/build_sim_inputs.py)"
        )
    missing = [key for key in REQUIRED_PROVENANCE if not asset.provenance.get(key)]
    if missing:
        raise SimInputError(
            f"sim input {asset.asset_id}: provenance lacks {', '.join(missing)}"
        )
    for item in asset.provenance.get("source_files") or []:
        if not (isinstance(item, Mapping) and item.get("sha256") and item.get("url")):
            raise SimInputError(
                f"sim input {asset.asset_id}: every source file needs its url "
                "and sha256"
            )
    header = path.open(encoding="utf-8").readline().strip().split(",")
    expected = list(ROLE_COLUMNS[asset.role])
    if header != expected:
        raise SimInputError(
            f"sim input {asset.asset_id}: columns {header} differ from the "
            f"{asset.role} columns {expected}"
        )


def read_sidecar(path: Path | str) -> SimInputAsset:
    """Read one provenance sidecar."""
    return SimInputAsset.model_validate_json(Path(path).read_text(encoding="utf-8"))


def load_registry(directory: Path | str | None = None) -> dict[str, SimInputAsset]:
    """Return every registered asset of a directory, verified.

    Args:
        directory: The asset directory (default: the packaged
            ``assets/annotation/sim_inputs``; cached).

    Returns:
        Asset id to asset, sorted by id.

    Raises:
        SimInputError: If a sidecar's id differs from its file stem or an
            asset fails ``validate_asset``.
    """
    if directory is None:
        return dict(_packaged_registry())
    return _load_registry(Path(directory))


@cache
def _packaged_registry() -> tuple[tuple[str, SimInputAsset], ...]:
    return tuple(_load_registry(SIM_INPUTS_DIR).items())


def _load_registry(directory: Path) -> dict[str, SimInputAsset]:
    registry: dict[str, SimInputAsset] = {}
    for sidecar in sorted(directory.glob(f"*{SIDECAR_SUFFIX}")):
        asset = read_sidecar(sidecar)
        stem = sidecar.name[: -len(SIDECAR_SUFFIX)]
        if asset.asset_id != stem or Path(asset.file).stem != stem:
            raise SimInputError(
                f"{sidecar.name}: asset id {asset.asset_id!r} and file "
                f"{asset.file!r} must match the sidecar name"
            )
        validate_asset(asset, directory)
        registry[asset.asset_id] = asset
    return registry


def get_asset(
    asset_id: str, registry: Mapping[str, SimInputAsset] | None = None
) -> SimInputAsset:
    """Return a registered asset by id.

    Raises:
        SimInputError: If no such asset is registered.
    """
    assets = load_registry() if registry is None else registry
    if asset_id not in assets:
        raise SimInputError(
            f"no simulation-input asset {asset_id!r} (registered: {sorted(assets)})"
        )
    return assets[asset_id]


def find_assets(
    *,
    role: AssetRole,
    species: str,
    chemistry: str | None = None,
    reference: str | None = None,
    registry: Mapping[str, SimInputAsset] | None = None,
) -> list[SimInputAsset]:
    """Return the assets of a role for a species (and chemistry, reference).

    Args:
        role: The asset role.
        species: ``human`` or ``mouse``; assets never cross species.
        chemistry: Chemistry to match, or ``None`` for any.
        reference: Reference to match, or ``None`` for any.
        registry: The registry (default: the packaged one).

    Returns:
        Matching assets, sorted by id.
    """
    assets = load_registry() if registry is None else registry
    return [
        asset
        for asset in assets.values()
        if asset.role == role
        and asset.species == species
        and (chemistry is None or asset.chemistry == chemistry)
        and (reference is None or asset.reference == reference)
    ]


def member_table_for(
    species: str,
    chemistry: str,
    reference: str,
    registry: Mapping[str, SimInputAsset] | None = None,
) -> SimInputAsset | None:
    """Return the measured factor table of a species x chemistry x reference.

    Human has none (no factor table against WHB exists; mouse factors are
    never used for human, SYNTHESIS §5.3); ``unknown`` chemistry has none.

    Returns:
        The single ``member`` asset, or ``None``.

    Raises:
        SimInputError: If more than one asset matches.
    """
    if chemistry == "unknown":
        return None
    matches = find_assets(
        role="member",
        species=species,
        chemistry=chemistry,
        reference=reference,
        registry=registry,
    )
    if len(matches) > 1:
        raise SimInputError(
            f"{len(matches)} member tables for {species} x {chemistry} x "
            f"{reference}: {[item.asset_id for item in matches]}"
        )
    return matches[0] if matches else None


def asset_hashes(assets: Iterable[SimInputAsset]) -> dict[str, dict[str, Any]]:
    """Return the ``build_hash`` record of the assets a version-7 bundle uses.

    Args:
        assets: The assets (members, profile, stress, scenario, panel lists).

    Returns:
        Asset id to ``SimInputAsset.hash_payload()``, sorted by id.
    """
    return {
        asset.asset_id: asset.hash_payload()
        for asset in sorted(assets, key=lambda item: item.asset_id)
    }


def read_asset_table(
    asset: SimInputAsset, directory: Path | str | None = None
) -> pd.DataFrame:
    """Read an asset's CSV (verified against its sidecar first).

    Args:
        asset: The asset.
        directory: The asset directory (default: the packaged one).

    Returns:
        The table.
    """
    validate_asset(asset, directory)
    return pd.read_csv(
        asset.path(directory),
        dtype={"gene_id": str, "tier": str, "top_class": str, "panel_key": str},
        keep_default_na=False,
        na_values={"log2_factor": [""], "top_class_section_share": [""]},
    )


# --------------------------------------------------------------------------
# Keyed draws


def keyed_normal(seed: int, stream: str, key: str) -> float:
    """Return one standard normal keyed by (seed, stream, key).

    The generator is seeded by ``resolvability.draw_key(seed, stream, key)``,
    as phase 1's ``simlib.keyed_normals``, so a gene's draw depends on its id
    only.
    """
    from merxen.annotation.resolvability import draw_key

    return float(
        np.random.default_rng(draw_key(int(seed), stream, str(key))).standard_normal()
    )


def keyed_uniform(seed: int, stream: str, key: str) -> float:
    """Return one uniform on [0, 1) keyed by (seed, stream, key)."""
    from merxen.annotation.resolvability import draw_key

    return float(np.random.default_rng(draw_key(int(seed), stream, str(key))).random())


def keyed_normals(genes: Sequence[str], stream: str, seed: int) -> np.ndarray:
    """Return ``keyed_normal`` for each gene (float64, gene order)."""
    return np.array(
        [keyed_normal(seed, stream, str(gene)) for gene in genes], dtype=np.float64
    )


def _median_normalised(log2_values: np.ndarray) -> np.ndarray:
    """Return ``2 ** log2_values`` divided by its median (phase 1's order)."""
    values = np.power(2.0, log2_values)
    return np.asarray(values / np.median(values), dtype=np.float64)


# --------------------------------------------------------------------------
# R3_measured_HO


@dataclass(frozen=True)
class R3Efficiency:
    """The measured per-gene efficiency of one R3 member.

    Attributes:
        genes: Panel genes (the test cells' columns).
        efficiency: Efficiency per gene, median 1 over the panel.
        log2_raw: ``log2`` efficiency before the median normalisation (what a
            gene's own draw gives; independent of the other panel genes).
        source: Per gene: ``measured`` (its factor plus the residual),
            ``resampled`` (from the empirical distribution; rule
            ``restricted``), or ``informative``, ``weak`` and
            ``uninformative`` (rule ``all_measured``).
        rule: The table rule.
        seed: The member seed.
        asset: The factor table's asset id.
        n_pool: Measured genes of the table the resample draws from.
    """

    genes: tuple[str, ...]
    efficiency: np.ndarray
    log2_raw: np.ndarray
    source: tuple[str, ...]
    rule: str
    seed: int
    asset: str
    n_pool: int

    @property
    def log2_efficiency(self) -> np.ndarray:
        """``log2`` of the normalised efficiency."""
        return np.asarray(np.log2(self.efficiency), dtype=np.float64)

    def summary(self) -> dict[str, Any]:
        """Return the counts and spread reports record."""
        counts = pd.Series(self.source).value_counts().sort_index()
        ln = np.log(self.efficiency) if len(self.efficiency) else np.array([])
        return {
            "asset": self.asset,
            "rule": self.rule,
            "seed": self.seed,
            "n_genes": len(self.genes),
            "n_pool": self.n_pool,
            "sources": {str(key): int(value) for key, value in counts.items()},
            "sd_ln": float(np.std(ln, ddof=1)) if len(ln) > 1 else None,
            "frac_below_0.25": float(np.mean(self.efficiency < 0.25))
            if len(ln)
            else None,
            "frac_below_0.125": float(np.mean(self.efficiency < 0.125))
            if len(ln)
            else None,
        }

    def frame(self) -> pd.DataFrame:
        """Return the per-gene table (gene, source, log2 raw and normalised)."""
        return pd.DataFrame(
            {
                "gene_id": list(self.genes),
                "source": list(self.source),
                "log2_raw": self.log2_raw,
                "log2_efficiency": self.log2_efficiency,
            }
        )


def measured_mask(
    table: pd.DataFrame,
    *,
    min_expected: float = R3_MIN_EXPECTED_COUNTS,
    min_top_class_share: float = R3_MIN_TOP_CLASS_SHARE,
) -> pd.Series:
    """Return which table genes take their measured factor (rule ``restricted``).

    An informative gene (expected >= 1,000 counts) whose top WMB class holds
    >= 0.5% of the measuring section's cells, with a finite factor. Weak-tier
    genes (Pcp2 at -3: composition, not chemistry), uninformative genes and
    genes whose top class is rare in the section (45% of informative genes,
    measured on off-target or ambient expression) do not (SYNTHESIS §4.2).

    Args:
        table: The ``member`` table indexed by gene id.
        min_expected: Expected-count minimum.
        min_top_class_share: Section share the gene's top class needs.

    Returns:
        Boolean per table gene.
    """
    factor = pd.to_numeric(table["log2_factor"], errors="coerce")
    expected = pd.to_numeric(table["expected_counts"], errors="coerce")
    share = pd.to_numeric(table["top_class_section_share"], errors="coerce")
    return (
        (table["tier"].astype(str) == TIER_INFORMATIVE)
        & (expected >= min_expected)
        & (share >= min_top_class_share)
        & np.isfinite(factor)
    )


def efficiency_table(
    asset: SimInputAsset, directory: Path | str | None = None
) -> pd.DataFrame:
    """Return a ``member`` asset's factor table, indexed by gene id.

    Raises:
        SimInputError: If the asset is not a member table or holds a gene twice.
    """
    if asset.role != "member":
        raise SimInputError(f"{asset.asset_id} is a {asset.role} asset, not a member")
    table = read_asset_table(asset, directory)
    if not table["gene_id"].is_unique:
        raise SimInputError(f"{asset.asset_id}: duplicate gene ids")
    return table.set_index("gene_id")


def r3_efficiency(
    genes: Sequence[str],
    table: pd.DataFrame,
    *,
    rule: str = "restricted",
    seed: int = 0,
    asset_id: str = EFFICIENCY_MOUSE_PRIME,
    residual_sd_log2: float = R3_RESIDUAL_SD_LOG2,
    cap_log2: float = R3_CAP_LOG2,
    min_expected: float = R3_MIN_EXPECTED_COUNTS,
    min_top_class_share: float = R3_MIN_TOP_CLASS_SHARE,
) -> R3Efficiency:
    """Return the R3_measured_HO per-gene efficiency of a panel (§8.3 v7.4).

    Rule ``restricted`` (production): a gene of ``measured_mask`` takes
    ``log2 e = clip(f, +-cap) + residual_sd_log2 * z`` with ``z`` the keyed
    normal of ``(seed, "measured_residual", gene)``; every other panel gene
    (weak, uninformative, top class rare in the section, or absent from the
    table) takes ``sorted(clip(f, +-cap))[floor(u * n)]`` over the table's
    ``n`` measured genes, ``u`` the keyed uniform of ``(seed,
    "measured_resample", gene)``. Rule ``all_measured`` (phase 1's D3 vector,
    the regression of pre-registration §14 (ii) only): informative and weak
    genes take ``f`` plus the residual (weak: ``sqrt(0.20^2 + (1 / ln 2)^2 /
    (observed + 1))``), uninformative and missing genes ``0 + 1.365 z``, with
    the same residual key. The efficiency is ``2 ** log2 e`` divided by its
    median over the panel.

    Args:
        genes: Panel genes in the test cells' column order.
        table: ``efficiency_table`` output (indexed by gene id).
        rule: ``restricted`` or ``all_measured``.
        seed: The member seed (R3@0 is the emission member).
        asset_id: The table's asset id (recorded).
        residual_sd_log2: Residual SD of measured genes (the split-half
            noise, 0.20 log2).
        cap_log2: Cap of the measured factors (+-3 log2).
        min_expected: ``measured_mask`` expected-count minimum.
        min_top_class_share: ``measured_mask`` section-share minimum.

    Returns:
        The efficiency.

    Raises:
        SimInputError: For an unknown rule or a table without measured genes.
    """
    if rule not in TABLE_RULES:
        raise SimInputError(f"unknown R3 table rule {rule!r} (one of {TABLE_RULES})")
    names = [str(gene) for gene in genes]
    z = keyed_normals(names, STREAM_MEASURED_RESIDUAL, seed)
    tier = table["tier"].reindex(names).fillna("missing").astype(str).to_numpy()
    factor = (
        pd.to_numeric(table["log2_factor"], errors="coerce").reindex(names).to_numpy()
    )
    if rule == "all_measured":
        observed = (
            pd.to_numeric(table["observed_counts"], errors="coerce")
            .reindex(names)
            .fillna(0.0)
            .to_numpy(np.float64)
        )
        informative = tier == TIER_INFORMATIVE
        weak = tier == TIER_WEAK
        none = ~(informative | weak)
        measured_factor = np.where(none, 0.0, np.nan_to_num(factor, nan=0.0))
        sd = np.where(
            informative,
            residual_sd_log2,
            np.where(
                weak,
                np.sqrt(
                    residual_sd_log2**2 + (1.0 / math.log(2.0)) ** 2 / (observed + 1.0)
                ),
                D3_UNINFORMATIVE_SD_LOG2,
            ),
        )
        log2_raw = measured_factor + sd * z
        source = np.where(
            informative, TIER_INFORMATIVE, np.where(weak, TIER_WEAK, "uninformative")
        )
        n_pool = int(informative.sum() + weak.sum())
    else:
        mask = measured_mask(
            table, min_expected=min_expected, min_top_class_share=min_top_class_share
        )
        pool = np.sort(
            np.clip(
                pd.to_numeric(table.loc[mask, "log2_factor"]).to_numpy(np.float64),
                -cap_log2,
                cap_log2,
            )
        )
        if len(pool) == 0:
            raise SimInputError(
                f"{asset_id}: no measured gene to resample from (rule restricted)"
            )
        own = np.array([bool(mask.get(name, False)) for name in names], dtype=bool)
        log2_raw = np.empty(len(names), dtype=np.float64)
        capped = np.clip(np.nan_to_num(factor, nan=0.0), -cap_log2, cap_log2)
        log2_raw[own] = capped[own] + residual_sd_log2 * z[own]
        for position in np.flatnonzero(~own):
            u = keyed_uniform(seed, STREAM_MEASURED_RESAMPLE, names[position])
            log2_raw[position] = pool[
                min(int(math.floor(u * len(pool))), len(pool) - 1)
            ]
        source = np.where(own, "measured", "resampled")
        n_pool = len(pool)
    efficiency = (
        _median_normalised(log2_raw) if len(names) else np.ones(0, dtype=np.float64)
    )
    return R3Efficiency(
        genes=tuple(names),
        efficiency=efficiency,
        log2_raw=np.asarray(log2_raw, dtype=np.float64),
        source=tuple(str(value) for value in source),
        rule=rule,
        seed=int(seed),
        asset=asset_id,
        n_pool=n_pool,
    )


# --------------------------------------------------------------------------
# R1_xtissue_lung_stress


def ratio_table(asset: SimInputAsset, directory: Path | str | None = None) -> pd.Series:
    """Return a ``stress`` asset's log2 ratio per gene id.

    Raises:
        SimInputError: If the asset is not a stress table.
    """
    if asset.role != "stress":
        raise SimInputError(f"{asset.asset_id} is a {asset.role} asset, not stress")
    table = read_asset_table(asset, directory)
    return pd.Series(
        pd.to_numeric(table["log2_ratio"]).to_numpy(np.float64),
        index=table["gene_id"].astype(str),
        name="log2_ratio",
    )


def xtissue_stress_efficiency(
    genes: Sequence[str],
    base_efficiency: np.ndarray,
    ratio: pd.Series,
    *,
    seed: int = 0,
    cap_log2: float = XTISSUE_CAP_LOG2,
    sd_ln: float = XTISSUE_SD_LN,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the R1_xtissue_lung_stress efficiency (a cross-tissue approximation).

    ``ln e = ln e_R1 + d``: ``d`` is the lung 5K / v1 per-area ratio (log2,
    centred, capped at +-2) in ln units for the genes measured there, else a
    keyed ``N(0, 0.702)`` (stream ``xtissue_lung_ratio``); ``e`` is divided
    by its median. The ratios are 5K-vs-v1 in FFPE lung, not 5K-vs-WHB in
    brain, so the recipe is reported as a stress member only (never
    emission; plan §8.3 v7.4, §8.10).

    Args:
        genes: Panel genes (test-cell column order).
        base_efficiency: The R1@0 efficiency (``gene_efficiency``).
        ratio: ``ratio_table`` output.
        seed: The member seed.
        cap_log2: Cap of the ratio.
        sd_ln: SD of the keyed draw for unmeasured genes.

    Returns:
        The efficiency and a boolean mask of the genes measured in lung.
    """
    names = [str(gene) for gene in genes]
    base = np.log(np.asarray(base_efficiency, dtype=np.float64))
    measured_values = ratio.reindex(names)
    measured = measured_values.notna().to_numpy(bool)
    d = np.clip(measured_values.fillna(0.0).to_numpy(np.float64), -cap_log2, cap_log2)
    d = d * math.log(2.0)
    z = keyed_normals(names, STREAM_XTISSUE, seed)
    d = np.where(measured, d, sd_ln * z)
    values = np.exp(base + d)
    efficiency = values / np.median(values) if len(values) else values
    return np.asarray(efficiency, dtype=np.float64), measured


# --------------------------------------------------------------------------
# Depth profiles


def is_neuronal_class(name: str, species: str) -> bool | None:
    """Return whether a resolvability class is neuronal (``None``: unknown).

    Mouse: the WMB class vocabulary's broad class (``Neurons``); human: the
    broad classes of the resolvability levels (``Exc``, ``Inh``,
    ``OtherNeuron``).
    """
    if species == "mouse":
        from merxen.annotation.vocab import NEURONS, load_asset_table

        vocab = load_asset_table("wmb_class_vocab.csv")
        match = vocab[vocab["class"] == str(name)]
        if match.empty:
            return None
        return bool(match["broad_class"].iloc[0] == NEURONS)
    if str(name) in HUMAN_NEURONAL_CLASSES:
        return True
    from merxen.annotation.vocab import HUMAN_FLOOR_CLASSES

    return False if str(name) in HUMAN_FLOOR_CLASSES else None


class DepthProfile:
    """Per-class empirical distribution of real TOTAL counts (§8.3 v7.5).

    Phase 1's ``simlib.DepthProfile``: a class with at least ``min_cells``
    profile cells samples its own totals; any other class samples the pooled
    neuronal or non-neuronal profile (all cells when its lineage is unknown
    or the profile is pooled). Values keep the cells' source order, so a
    keyed sample reproduces phase 1's draws.

    Attributes:
        classes: Called class per cell (source order).
        totals: Total counts per cell (source order).
        by_class: Totals per class, in source order.
        neuronal: Per class, whether it is neuronal (``None``: unknown).
        label: Where the profile comes from (asset label or file).
        species: The species the profile describes.
        pooled: Every class samples all cells (a pooled profile or scenario).
        min_cells: Cells a class needs to sample its own totals.
        asset: The asset id, when the profile is a registered asset.
        sha256: sha256 of the source (asset or file).
        used: Class -> the pool it sampled (filled by ``sample``).
    """

    def __init__(
        self,
        classes: Sequence[str] | np.ndarray,
        totals: Sequence[float] | np.ndarray,
        *,
        species: str,
        label: str,
        neuronal: Callable[[str], bool | None]
        | Mapping[str, bool | None]
        | None = None,
        pooled: bool = False,
        min_cells: int = PROFILE_MIN_CLASS_CELLS,
        asset: str | None = None,
        sha256: str | None = None,
    ) -> None:
        """Build a profile from per-cell classes and totals (source order).

        Args:
            classes: Called class per cell.
            totals: Total counts per cell.
            species: The species.
            label: Provenance label.
            neuronal: Per class, whether it is neuronal (mapping or callable;
                default ``is_neuronal_class``).
            pooled: Sample every class from all cells.
            min_cells: Cells a class needs to sample its own totals.
            asset: The asset id, if any.
            sha256: The source sha256, if known.

        Raises:
            SimInputError: If classes and totals differ in length.
        """
        self.classes = np.asarray([str(value) for value in classes], dtype=object)
        self.totals = np.asarray(totals, dtype=np.float64)
        if len(self.classes) != len(self.totals):
            raise SimInputError("depth profile: classes and totals differ in length")
        unique = sorted(set(self.classes.tolist()))
        if neuronal is None:
            flags = {name: is_neuronal_class(name, species) for name in unique}
        elif isinstance(neuronal, Mapping):
            flags = {name: neuronal.get(name) for name in unique}
        else:
            flags = {name: neuronal(name) for name in unique}
        self.neuronal: dict[str, bool | None] = flags
        self.by_class: dict[str, np.ndarray] = {
            name: self.totals[self.classes == name] for name in unique
        }
        cell_flags = np.array([flags.get(name) for name in self.classes], dtype=object)
        self._pools: dict[str, np.ndarray] = {
            POOL_NEURONAL: self.totals[cell_flags == True],  # noqa: E712
            POOL_NON_NEURONAL: self.totals[cell_flags == False],  # noqa: E712
            POOL_ALL: self.totals,
        }
        self.label = label
        self.species = species
        self.pooled = pooled
        self.min_cells = int(min_cells)
        self.asset = asset
        self.sha256 = sha256
        self.used: dict[str, str] = {}

    @property
    def n_cells(self) -> int:
        """Cells of the profile."""
        return len(self.totals)

    def pool(self, name: str) -> np.ndarray:
        """Return a pool: ``POOL_NEURONAL``, ``POOL_NON_NEURONAL`` or ``POOL_ALL``."""
        return self._pools[name]

    def source(self, cls_name: str) -> str:
        """Return the class or pool a truth class samples its totals from."""
        name = str(cls_name)
        if not self.pooled:
            values = self.by_class.get(name)
            if values is not None and len(values) >= self.min_cells:
                return name
            flag = self.neuronal.get(name)
            if flag is None:
                flag = is_neuronal_class(name, self.species)
            if flag is True and len(self._pools[POOL_NEURONAL]):
                return POOL_NEURONAL
            if flag is False and len(self._pools[POOL_NON_NEURONAL]):
                return POOL_NON_NEURONAL
        return POOL_ALL

    def values(self, source: str) -> np.ndarray:
        """Return the totals of a class or pool."""
        if source in self._pools:
            return self._pools[source]
        return self.by_class[source]

    def sample(self, cls_name: str, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` totals for a truth class (``rng.integers`` over the values).

        Raises:
            SimInputError: If the class's source holds no cells.
        """
        source = self.source(cls_name)
        self.used[str(cls_name)] = source
        values = self.values(source)
        if len(values) == 0:
            raise SimInputError(f"depth profile {self.label}: no cells for {cls_name}")
        return values[rng.integers(0, len(values), size=int(n))]

    def median(self, cls_name: str) -> float | None:
        """Return the median total of the source a class samples from."""
        values = self.values(self.source(cls_name))
        return float(np.median(values)) if len(values) else None

    def bin_shares(self, grid: Sequence[int]) -> dict[str, dict[int, float]]:
        """Return each class's share of cells per depth bin (``0``: below the grid).

        Classes are the profile's own (their totals, never a pool), plus
        ``POOL_ALL``.
        """
        from merxen.annotation.resolvability import depth_bin

        result: dict[str, dict[int, float]] = {}
        classes = dict(self.by_class)
        classes[POOL_ALL] = self.totals
        for name, values in classes.items():
            if len(values) == 0:
                continue
            bins = depth_bin(values, list(grid))
            shares = {0: float(np.mean(np.isnan(bins)))}
            for depth in grid:
                shares[int(depth)] = float(np.mean(bins == float(depth)))
            result[name] = shares
        return result

    def to_json(self) -> dict[str, Any]:
        """Return the provenance record reports print."""
        return {
            "label": self.label,
            "species": self.species,
            "asset": self.asset,
            "sha256": self.sha256,
            "pooled": self.pooled,
            "min_cells": self.min_cells,
            "n_cells": self.n_cells,
            "classes": {
                name: {
                    "n_cells": int(len(values)),
                    "median": float(np.median(values)) if len(values) else None,
                    "neuronal": self.neuronal.get(name),
                }
                for name, values in self.by_class.items()
            },
            "sampled_from": dict(self.used),
        }


def profile_from_asset(
    asset: SimInputAsset, directory: Path | str | None = None
) -> DepthProfile:
    """Return the depth profile of a ``profile`` or ``scenario`` asset.

    A ``profile`` asset holds per-cell class codes (legend in the sidecar's
    ``extra.class_legend``: code, class, neuronal) and totals in source
    order; a ``scenario`` asset holds a pooled histogram (total, cells),
    expanded in ascending total order and sampled pooled.

    Raises:
        SimInputError: For another role or a code missing from the legend.
    """
    table = read_asset_table(asset, directory)
    species = str(asset.species or "")
    if asset.role == "profile":
        legend = asset.extra.get("class_legend") or []
        names = {int(item["code"]): str(item["class"]) for item in legend}
        flags = {str(item["class"]): item.get("neuronal") for item in legend}
        codes = table["class_code"].astype(int).to_numpy()
        missing = sorted(set(codes.tolist()) - set(names))
        if missing:
            raise SimInputError(
                f"{asset.asset_id}: class codes {missing} lack a legend"
            )
        classes = np.array([names[int(code)] for code in codes], dtype=object)
        return DepthProfile(
            classes,
            table["total_counts"].to_numpy(np.float64),
            species=species,
            label=asset.label,
            neuronal=flags,
            asset=asset.asset_id,
            sha256=asset.sha256,
        )
    if asset.role == "scenario":
        totals = np.repeat(
            table["total_counts"].to_numpy(np.float64),
            table["n_cells"].to_numpy(np.int64),
        )
        return DepthProfile(
            np.full(len(totals), POOL_ALL, dtype=object),
            totals,
            species=species,
            label=asset.label,
            neuronal={POOL_ALL: None},
            pooled=True,
            asset=asset.asset_id,
            sha256=asset.sha256,
        )
    raise SimInputError(
        f"{asset.asset_id} is a {asset.role} asset, not a depth profile"
    )


def read_depth_profile_table(path: Path | str, *, species: str) -> DepthProfile:
    """Read a depth-profile CSV: per-class or pooled (§8.3 v7.5).

    One row per cell with a ``total_counts`` (or ``depth``, ``counts``)
    column; a ``class`` (or ``truth_class``, ``cls_call``) column makes it
    per-class (the ``5k_real/mouse5k/depth_profile.csv`` format), otherwise
    it is pooled; an optional ``weight`` column repeats a row that many times
    (rounded).

    Args:
        path: The CSV.
        species: The species of the panel (a profile never crosses species).

    Returns:
        The profile.

    Raises:
        SimInputError: If no count column exists.
    """
    table = pd.read_csv(path)
    column = next(
        (name for name in ("total_counts", "depth", "counts") if name in table.columns),
        None,
    )
    if column is None:
        raise SimInputError(f"{path} needs a total_counts, depth or counts column")
    class_column = next(
        (
            name
            for name in ("class", "truth_class", "cls_call")
            if name in table.columns
        ),
        None,
    )
    totals = table[column].astype(float).to_numpy()
    classes = (
        table[class_column].astype(str).to_numpy(dtype=object)
        if class_column is not None
        else np.full(len(table), POOL_ALL, dtype=object)
    )
    if "weight" in table.columns:
        repeats = np.maximum(np.round(table["weight"].astype(float)).astype(int), 0)
        totals = np.repeat(totals, repeats.to_numpy())
        classes = np.repeat(classes, repeats.to_numpy())
    digest = _sha256_file(Path(path))
    return DepthProfile(
        classes,
        totals,
        species=species,
        label=f"file {Path(path).name}",
        neuronal=None if class_column is not None else {POOL_ALL: None},
        pooled=class_column is None,
        sha256=digest,
    )


# --------------------------------------------------------------------------
# Chemistry


@dataclass(frozen=True)
class ChemistryResolution:
    """A panel's chemistry and why (plan §3.7 ``panel_chemistry``).

    Attributes:
        chemistry: ``xenium_prime``, ``xenium_v1``, ``merscope`` or
            ``unknown``.
        reason: How it was resolved.
        jaccard: Best Jaccard to a pinned public Prime list (Xenium panels).
        panel_key: That list's key.
    """

    chemistry: Chemistry
    reason: str
    jaccard: float | None = None
    panel_key: str | None = None

    def to_json(self) -> dict[str, Any]:
        """Return the record as JSON-native values."""
        return {
            "chemistry": self.chemistry,
            "reason": self.reason,
            "jaccard": None if self.jaccard is None else round(self.jaccard, 6),
            "panel_key": self.panel_key,
        }


def prime_panel_lists(
    registry: Mapping[str, SimInputAsset] | None = None,
) -> dict[str, tuple[str, frozenset[str]]]:
    """Return the pinned public Prime gene lists: key -> (species, gene ids)."""
    asset = get_asset(PRIME_PANEL_LISTS, registry)
    table = read_asset_table(asset)
    return {
        str(key): (
            str(group["species"].iloc[0]),
            frozenset(group["gene_id"].astype(str)),
        )
        for key, group in table.groupby("panel_key", sort=True)
    }


def resolve_chemistry(
    genes: Iterable[str],
    *,
    species: str,
    platform: str | None,
    declared: str = "auto",
    registry: Mapping[str, SimInputAsset] | None = None,
    min_jaccard: float = CHEMISTRY_MIN_JACCARD,
) -> ChemistryResolution:
    """Return a panel's chemistry (plan §3.7, §8.3 v7.4).

    A declared value wins; ``auto``: MERSCOPE → ``merscope``; a Xenium panel
    (or one without a platform) with Jaccard >= ``min_jaccard`` to a pinned
    public Prime list of its species → ``xenium_prime``; otherwise
    ``unknown`` (no measured factor table applies).

    Args:
        genes: The panel's Ensembl gene ids.
        species: The panel species.
        platform: ``MERSCOPE``, ``XENIUM`` or ``None``.
        declared: ``AnnotationPanelConfig.panel_chemistry``.
        registry: The asset registry (default: the packaged one).
        min_jaccard: Jaccard a Prime list needs.

    Returns:
        The resolution.
    """
    if declared != "auto":
        return ChemistryResolution(chemistry=declared, reason="declared")  # type: ignore[arg-type]
    platform_name = (platform or "").upper()
    if platform_name == "MERSCOPE":
        return ChemistryResolution(chemistry="merscope", reason="platform MERSCOPE")
    panel = frozenset(str(gene) for gene in genes)
    best: tuple[float, str | None] = (0.0, None)
    for key, (list_species, list_genes) in prime_panel_lists(registry).items():
        if list_species != species or not panel:
            continue
        jaccard = len(panel & list_genes) / len(panel | list_genes)
        if jaccard > best[0]:
            best = (jaccard, key)
    if (
        best[1] is not None
        and best[0] >= min_jaccard
        and platform_name in ("", "XENIUM")
    ):
        return ChemistryResolution(
            chemistry="xenium_prime",
            reason=f"Jaccard {best[0]:.4f} >= {min_jaccard} to {best[1]}",
            jaccard=best[0],
            panel_key=best[1],
        )
    return ChemistryResolution(
        chemistry="unknown",
        reason=(
            f"no pinned public Prime list within Jaccard {min_jaccard} "
            f"(best {best[0]:.4f}); declare panel_chemistry to use a table"
        ),
        jaccard=best[0] if best[1] is not None else None,
        panel_key=best[1],
    )


def sidecar_json(asset: SimInputAsset) -> str:
    """Return a sidecar's canonical JSON text (what the builder writes)."""
    return json.dumps(asset.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"

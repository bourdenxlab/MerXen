"""Explicit, reference-bound confidence policies for whole-panel hippocampus.

These policies use existing atlas diagnostics. They do not inherit frontal
family trust and do not claim new independent or spatial ground-truth validation.
Raw calls survive every abstention; a child can be emitted only with its parent.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Annotated, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class ConfidenceObjective(_FrozenModel):
    """Declared error/support objective; all operating limits are explicit."""

    min_precision: float = Field(gt=0, le=1)
    min_wilson_lower: float = Field(gt=0, lt=1)
    wilson_z: float = Field(gt=0)
    bootstrap_cuts: tuple[float, ...]

    @model_validator(mode="after")
    def _validate_limits(self) -> ConfidenceObjective:
        if self.min_wilson_lower > self.min_precision:
            raise ValueError("Wilson lower target exceeds point precision target")
        cuts = self.bootstrap_cuts
        if not cuts or tuple(sorted(set(cuts))) != cuts:
            raise ValueError("Bootstrap cuts must be nonempty, increasing and unique")
        if any(not math.isfinite(x) or not 0 <= x <= 1 for x in cuts):
            raise ValueError("Bootstrap cuts must be finite probabilities")
        return self


class ReferenceBinding(_FrozenModel):
    """Exact artifacts and full-query panel to which a policy applies."""

    profile_version: Literal[1] = 1
    species: Literal["human"] = "human"
    anatomical_region: Literal["hippocampus"] = "hippocampus"
    reference_id: str = Field(min_length=1)
    taxonomy_id: str = Field(min_length=1)
    regions: tuple[str, ...]
    levels: tuple[str, ...]
    panel_hash: Sha256
    query_genes_sha256: Sha256
    n_query_genes: Literal[18049] = 18049
    reference_sha256: Sha256
    lookup_sha256: Sha256
    tree_sha256: Sha256

    @model_validator(mode="after")
    def _validate_identity(self) -> ReferenceBinding:
        if not self.regions or len(set(self.regions)) != len(self.regions):
            raise ValueError("Reference regions must be nonempty and unique")
        if not self.levels or len(set(self.levels)) != len(self.levels):
            raise ValueError("Taxonomy levels must be nonempty and unique")
        return self

    def check_query(self, panel_hash: str, genes: Sequence[str]) -> None:
        """Refuse a subset, reordered panel, duplicates or a different panel hash."""
        if (
            panel_hash != self.panel_hash
            or len(genes) != self.n_query_genes
            or len(set(genes)) != len(genes)
            or ordered_genes_sha256(genes) != self.query_genes_sha256
        ):
            raise ValueError("Hippocampal query must retain the exact full panel")


class DepthRule(_FrozenModel):
    """One class/depth decision, with distinct-cell evidence counts."""

    level: str
    assignment: str
    min_counts: int = Field(ge=1)
    max_counts: int | None = Field(default=None, ge=1)
    min_bootstrap: float | None = Field(default=None, ge=0, le=1)
    n_evaluated: int = Field(ge=0)
    n_accepted: int = Field(ge=0)
    n_correct: int = Field(ge=0)
    precision: float | None = Field(default=None, ge=0, le=1)
    wilson_lower: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def _validate_evidence(self) -> DepthRule:
        if self.max_counts is not None and self.max_counts <= self.min_counts:
            raise ValueError("Depth interval must be nonempty")
        if not 0 <= self.n_correct <= self.n_accepted <= self.n_evaluated:
            raise ValueError("Inconsistent calibration evidence counts")
        if self.min_bootstrap is None:
            if (
                self.n_accepted
                or self.precision is not None
                or self.wilson_lower is not None
            ):
                raise ValueError("An unavailable rule cannot claim accepted evidence")
        elif not self.n_accepted or self.precision is None or self.wilson_lower is None:
            raise ValueError("An emitting rule needs observed support and precision")
        elif not math.isclose(self.precision, self.n_correct / self.n_accepted):
            raise ValueError("Rule precision differs from its evidence counts")
        return self


class HippocampalPolicy(_FrozenModel):
    """Versioned policy with explicit provenance and no inherited trust flag."""

    binding: ReferenceBinding
    objective: ConfidenceObjective
    calibration_sha256: Sha256
    evidence_basis: Literal["existing_atlas_diagnostics_reused"]
    depth_edges: tuple[int, ...]
    rules: tuple[DepthRule, ...]

    @model_validator(mode="after")
    def _validate_rules(self) -> HippocampalPolicy:
        edges = self.depth_edges
        if not edges or edges[0] < 1 or tuple(sorted(set(edges))) != edges:
            raise ValueError("Depth edges must be positive, increasing and unique")
        keys = set()
        for rule in self.rules:
            key = (rule.level, rule.assignment, rule.min_counts)
            if key in keys or rule.level not in self.binding.levels:
                raise ValueError("Duplicate rule or unknown taxonomy level")
            keys.add(key)
            if rule.min_counts not in edges:
                raise ValueError("Rule does not start at a declared depth edge")
            i = edges.index(rule.min_counts)
            end = edges[i + 1] if i + 1 < len(edges) else None
            if rule.max_counts != end:
                raise ValueError("Rule does not match a declared depth interval")
            if rule.min_bootstrap is not None:
                assert rule.precision is not None and rule.wilson_lower is not None
                if (
                    rule.precision < self.objective.min_precision
                    or rule.wilson_lower < self.objective.min_wilson_lower
                    or rule.min_bootstrap not in self.objective.bootstrap_cuts
                ):
                    raise ValueError("Emitting rule fails the declared objective")
                lower = wilson_lower(
                    rule.n_correct, rule.n_accepted, self.objective.wilson_z
                )
                if not math.isclose(rule.wilson_lower, lower):
                    raise ValueError("Wilson bound differs from its evidence counts")
        return self

    def digest(self) -> str:
        """Hash semantic policy content, excluding filesystem paths and clocks."""
        payload = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode()).hexdigest()


def ordered_genes_sha256(genes: Sequence[str]) -> str:
    """Hash all query columns in their declared order."""
    return hashlib.sha256("\n".join(genes).encode()).hexdigest()


def wilson_lower(n_correct: int, n: int, z: float) -> float:
    """Return a binomial Wilson lower bound at the explicitly supplied z score."""
    if not 0 <= n_correct <= n or n <= 0 or not math.isfinite(z) or z <= 0:
        raise ValueError("Wilson evidence requires valid counts and positive z")
    p = n_correct / n
    return (
        p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    ) / (1 + z * z / n)


def _check_counts(values: pd.Series) -> np.ndarray:
    counts = np.asarray(pd.to_numeric(values, errors="raise"), dtype=float)
    if (
        not np.isfinite(counts).all()
        or (counts < 0).any()
        or (counts != np.floor(counts)).any()
    ):
        raise ValueError("Transcript depths must be finite nonnegative integer counts")
    return counts


def calibrate_rules(
    observations: pd.DataFrame,
    *,
    levels: Sequence[str],
    depth_edges: Sequence[int],
    objective: ConfidenceObjective,
) -> tuple[DepthRule, ...]:
    """Maximize observed coverage subject to a predeclared class/depth objective.

    Multiple thinnings of a cell count once in each level/depth interval. Its
    lowest-depth realization is selected by counts then query ID, before seeing
    correctness. Cuts tie-break by precision, then higher bootstrap support.
    This is development fitting; its Wilson bounds do not confer independent
    confirmation or model spatial platform noise.
    """
    required = {
        "cell_id",
        "query_id",
        "level",
        "truth",
        "assignment",
        "bp",
        "n_counts",
        "marker_supported",
    }
    if not required <= set(observations) or observations.empty:
        raise ValueError("Missing or empty calibration observations")
    data = observations.copy()
    for key in ["cell_id", "query_id", "level", "truth"]:
        if data[key].isna().any():
            raise ValueError("Calibration identity and truth cannot be missing")
        data[key] = data[key].astype(str)
    if data.duplicated(["query_id", "cell_id", "level"]).any():
        raise ValueError("Duplicate calibration query/cell/level")
    if not set(data.level) <= set(levels):
        raise ValueError("Calibration contains unknown taxonomy levels")
    if (data.groupby(["cell_id", "level"]).truth.nunique() != 1).any():
        raise ValueError("Repeated cell realizations have inconsistent truth")
    counts = _check_counts(data.n_counts)
    data["n_counts"] = counts
    bp = pd.to_numeric(data.bp, errors="raise").to_numpy(dtype=float)
    if np.isinf(bp).any() or ((bp < 0) | (bp > 1)).any():
        raise ValueError("Bootstrap support must be a probability or missing")
    data["bp"] = bp
    if not data.marker_supported.isin([True, False]).all():
        raise ValueError("Marker support must be explicitly boolean")
    edges = tuple(depth_edges)
    if not edges or edges[0] < 1 or tuple(sorted(set(edges))) != edges:
        raise ValueError("Invalid calibration depth intervals")
    data["depth_bin"] = np.searchsorted(edges, counts, side="right") - 1
    data = data.sort_values(["n_counts", "query_id"], kind="stable").drop_duplicates(
        ["cell_id", "level", "depth_bin"]
    )
    rules = []
    for i, start in enumerate(edges):
        interval = data.loc[data.depth_bin.eq(i)]
        for level in levels:
            part = interval.loc[interval.level.eq(level)]
            for label in sorted(part.assignment.dropna().astype(str).unique()):
                rows = part.loc[part.assignment.eq(label).fillna(False)]
                eligible = rows.loc[rows.marker_supported & rows.bp.notna()]
                options = []
                for cut in objective.bootstrap_cuts:
                    selected = eligible.loc[eligible.bp.ge(cut - 1e-6)]
                    if selected.empty:
                        continue
                    correct = int(selected.truth.eq(label).sum())
                    precision = correct / len(selected)
                    lower = wilson_lower(correct, len(selected), objective.wilson_z)
                    if (
                        precision >= objective.min_precision
                        and lower >= objective.min_wilson_lower
                    ):
                        options.append((len(selected), precision, cut, correct, lower))
                best = max(options) if options else None
                rules.append(
                    DepthRule(
                        level=level,
                        assignment=label,
                        min_counts=start,
                        max_counts=edges[i + 1] if i + 1 < len(edges) else None,
                        min_bootstrap=best[2] if best else None,
                        n_evaluated=len(rows),
                        n_accepted=best[0] if best else 0,
                        n_correct=best[3] if best else 0,
                        precision=best[1] if best else None,
                        wilson_lower=best[4] if best else None,
                    )
                )
    return tuple(rules)


def resolve_reference_calls(
    objects: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    policy: HippocampalPolicy,
    binding: ReferenceBinding,
    parent_of: Mapping[tuple[str, str], str],
) -> pd.DataFrame:
    """Apply a bound policy, retaining all objects, raw calls and abstention causes.

    ``objects`` contains unique cell_id and full-query n_counts. Predictions
    contain exactly every positive-count object at every bound taxonomy level,
    plus explicit marker_supported. Parent keys are (child level, child label).
    Artifact bindings must already have been derived from the actual runtime
    panel, precompute, lookup and taxonomy files, not copied from the policy.
    """
    if binding != policy.binding:
        raise ValueError("Policy reference, taxonomy or full-panel binding changed")
    if (
        not {"cell_id", "n_counts"} <= set(objects)
        or objects.cell_id.isna().any()
        or not objects.cell_id.is_unique
    ):
        raise ValueError("Objects require unique, non-null IDs and transcript depths")
    counts = _check_counts(objects.n_counts)
    required = {"cell_id", "level", "assignment", "name", "bp", "marker_supported"}
    if not required <= set(predictions):
        raise ValueError("Missing raw prediction fields")
    positive_ids = set(objects.loc[counts > 0, "cell_id"])
    if not set(predictions.level) <= set(binding.levels):
        raise ValueError("Predictions contain an unexpected taxonomy level")
    out = []
    previous: pd.DataFrame | None = None
    rule_of = {(r.level, r.assignment, r.min_counts): r for r in policy.rules}
    for level in binding.levels:
        part = predictions.loc[predictions.level.eq(level)].copy()
        if (
            part.cell_id.isna().any()
            or not part.cell_id.is_unique
            or set(part.cell_id) != positive_ids
        ):
            raise ValueError(
                "Predictions must cover every expressed object exactly once per level"
            )
        bp = pd.to_numeric(part.bp, errors="raise").to_numpy(dtype=float)
        if np.isinf(bp).any() or ((bp < 0) | (bp > 1)).any():
            raise ValueError("Bootstrap support must be a probability or missing")
        if not part.marker_supported.isin([True, False]).all():
            raise ValueError("Marker support must be explicitly boolean")
        part = objects[["cell_id", "n_counts"]].merge(
            part, on="cell_id", how="left", validate="one_to_one"
        )
        part["level"] = level
        part["raw_assignment"] = part.assignment
        part["raw_name"] = part["name"]
        accepted = np.zeros(len(part), dtype=bool)
        status = np.full(len(part), "no_expression", dtype=object)
        depth_bins = np.searchsorted(policy.depth_edges, counts, side="right") - 1
        for i, row in enumerate(part.itertuples()):
            if counts[i] == 0:
                continue
            status[i] = "not_calibrated"
            if depth_bins[i] < 0:
                status[i] = "outside_calibrated_depth"
                continue
            if pd.isna(row.assignment):
                status[i] = "unassigned"
                continue
            if not row.marker_supported:
                status[i] = "marker_unsupported"
                continue
            rule = rule_of.get(
                (level, str(row.assignment), policy.depth_edges[depth_bins[i]])
            )
            if rule is None or rule.min_bootstrap is None:
                continue
            if pd.isna(row.bp) or row.bp < rule.min_bootstrap - 1e-6:
                status[i] = "low_support"
                continue
            if previous is not None:
                parent = parent_of.get((level, str(row.assignment)))
                if parent is None or parent != previous.raw_assignment.iloc[i]:
                    raise ValueError("Mapped child does not match its taxonomy parent")
                if not previous.accepted.iloc[i]:
                    status[i] = "parent_unresolved"
                    continue
            accepted[i] = True
            status[i] = "accepted_existing_atlas_policy"
        part["accepted"] = accepted
        part["resolution_status"] = status
        part["assignment"] = part.assignment.where(accepted)
        part["name"] = part["name"].where(accepted)
        part["policy_sha256"] = policy.digest()
        part["evidence_basis"] = policy.evidence_basis
        out.append(part)
        previous = part
    return pd.concat(out, ignore_index=True)

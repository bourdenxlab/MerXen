#!/usr/bin/env python
"""Marker pseudo-labels of the H9 criterion (plan §14 H9; ports of the dataset reports).

H9 scores confident ``ct_broad`` calls against crude canonical-marker
pseudo-labels, with the methods of the dataset characterisations the plan
names ("marker pseudo-labels, ``data/P7513`` §C, ``data/P1212`` §4"). Both
methods are ported verbatim here so that H9 can be scored on segmentations
the reports never labelled (reseg, H17) and on the held-out donors (for
information):

- ``p7513_pseudo_labels`` (``data/P7513/characterise_dataset.py``
  ``marker_scores`` / ``pseudo_labels``): top-2 scaled marker scores on the
  log-normalised matrix; seed cells per class; a multinomial class profile
  over all panel genes from the seeds' raw counts; label at posterior >= 0.99,
  else ``Unresolved``; ``Mixed`` when a 50/50 two-class mixture beats the
  best class by >= 10 nats. Broad collapse ``PSEUDO_TO_BROAD``; resolved cells
  are those not ``Unresolved`` / ``Mixed``.
- ``p1212_pseudo_labels`` (``data/P1212/scripts/common.py``
  ``pseudo_labels``): each marker's counts over its mean positive count;
  class score = the sum; label when the top class has >= 1.5 units and
  >= 60% of the summed scores, else ``Unassigned``; the 6-class collapse
  ``PSEUDO6`` (Exc + Inh -> Neuron, Vasc + Fibro -> Vascular).

``h9_agreement`` scores a broad labelling against either pseudo-labelling on
the 6-class scheme both reports used (Neuron, Astro, Oligo, OPC, Micro,
Vascular; our Vascular cells and Fibroblasts both collapse to Vascular):
the share of confident cells with a resolved pseudo-label whose collapsed
class equals the pseudo-label's (P7513's ``Immune_other`` pseudo-label is
resolved and never matches, as in the report's MapMyCells comparison).

The module is imported by ``resolve_criteria.py``; running it prints the
marker sets.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

# data/P7513/characterise_dataset.py PSEUDO_HUMAN (verbatim).
P7513_PSEUDO_MARKERS: dict[str, list[str]] = {
    "Exc": ["SLC17A7", "SLC17A6", "SATB2", "CUX2", "RORB", "THEMIS", "FEZF2", "TBR1"],
    "Inh": ["GAD1", "GAD2", "SLC32A1", "LHX6", "PVALB", "SST", "VIP", "ADARB2"],
    "Astro": ["AQP4", "GJA1", "SOX9", "FGFR3", "SLC1A2", "SLC1A3", "GFAP", "ALDH1L1"],
    "Oligo": ["MOG", "MOBP", "OPALIN", "MAG", "CLDN11", "ERMN", "PLP1", "MBP"],
    "OPC": ["PDGFRA", "VCAN", "CSPG4"],
    "Micro": [
        "CX3CR1",
        "P2RY12",
        "P2RY13",
        "GPR34",
        "AIF1",
        "SPI1",
        "CSF1R",
        "TMEM119",
        "C1QA",
    ],
    "Endo": ["FLT1", "PECAM1", "CAV1", "CLDN5"],
    "Mural/VLMC": ["DCN", "FBLN1", "ABCC9", "PDGFRB", "RGS5", "COL1A2", "LUM"],
    "Lymph": ["CD3G", "CD2", "TRAC", "NKG7", "GZMA"],
}
# data/P7513/characterise_dataset.py PSEUDO_TO_BROAD (verbatim).
P7513_PSEUDO_TO_BROAD: dict[str, str] = {
    "Exc": "Neuron",
    "Inh": "Neuron",
    "Astro": "Astro",
    "Oligo": "Oligo",
    "OPC": "OPC",
    "Micro": "Micro",
    "Endo": "Vascular",
    "Mural/VLMC": "Vascular",
    "Lymph": "Immune_other",
    "Unresolved": "Unresolved",
    "Mixed": "Mixed",
}
P7513_UNRESOLVED: tuple[str, ...] = ("Unresolved", "Mixed")

# data/P1212/scripts/common.py PSEUDO_MARKERS and PSEUDO6 (verbatim).
P1212_PSEUDO_MARKERS: dict[str, list[str]] = {
    "Exc": ["SLC17A7", "SLC17A6", "CUX2", "RORB", "THEMIS"],
    "Inh": ["GAD1", "GAD2", "PVALB", "SST", "VIP", "LHX6"],
    "Astro": ["AQP4", "GJA1", "SOX9", "FGFR3"],
    "Oligo": ["MOG", "MOBP", "OPALIN", "MAG", "CLDN11", "ERMN"],
    "OPC": ["PDGFRA", "VCAN", "CSPG4", "PTPRZ1"],
    "Micro": ["CX3CR1", "P2RY12", "P2RY13", "GPR34", "AIF1", "SPI1", "PTPRC"],
    "Vasc": ["FLT1", "PECAM1", "CALCRL", "ABCC9"],
    "Fibro": ["DCN", "FBLN1"],
}
P1212_PSEUDO6: dict[str, str] = {
    "Exc": "Neuron",
    "Inh": "Neuron",
    "Astro": "Astro",
    "Oligo": "Oligo",
    "OPC": "OPC",
    "Micro": "Micro",
    "Vasc": "Vascular",
    "Fibro": "Vascular",
}
P1212_UNRESOLVED: tuple[str, ...] = ("Unassigned",)

# Our broad classes collapsed to the reports' 6-class scheme (both reports
# map the pipeline's "Fibroblasts" to "Vascular").
BROAD_TO_SIX: dict[str, str] = {
    "Neurons": "Neuron",
    "Astrocytes": "Astro",
    "Oligodendrocytes": "Oligo",
    "Oligodendrocyte precursors": "OPC",
    "Microglia": "Micro",
    "Vascular cells": "Vascular",
    "Fibroblasts": "Vascular",
}
METHODS: tuple[str, ...] = ("p7513_c", "p1212_4")


def _present(genes: Sequence[str], var_names: Sequence[str]) -> list[str]:
    names = set(map(str, var_names))
    return [gene for gene in genes if gene in names]


def p7513_marker_scores(
    lognorm: sparse.spmatrix,
    var_names: Sequence[str],
    markers: Mapping[str, Sequence[str]] = P7513_PSEUDO_MARKERS,
    top_k: int = 2,
) -> pd.DataFrame:
    """Per-cell class scores: the mean of the top-k scaled markers (P7513 §C step 1).

    Each marker's log-normalised expression is divided by its 90th percentile
    among expressing cells and clipped at 1.

    Args:
        lognorm: Cells x genes ``log1p(normalize_total)`` matrix (the clustered
            H5AD's ``X``).
        var_names: Gene symbol per column.
        markers: Class to marker symbols.
        top_k: Markers averaged per class.

    Returns:
        Cells x classes scores (classes without a panel marker left out).
    """
    sets = {key: _present(value, var_names) for key, value in markers.items()}
    sets = {key: value for key, value in sets.items() if value}
    genes = sorted({gene for value in sets.values() for gene in value})
    column = {str(name): index for index, name in enumerate(var_names)}
    matrix = sparse.csc_matrix(lognorm[:, [column[gene] for gene in genes]]).astype(
        np.float32
    )
    scale = np.ones(len(genes), dtype=np.float32)
    for j in range(len(genes)):
        values = matrix.data[matrix.indptr[j] : matrix.indptr[j + 1]]
        scale[j] = np.quantile(values, 0.9) if values.size else 1.0
    scaled = matrix.multiply(1.0 / scale[None, :]).tocsr()
    scaled.data = np.minimum(scaled.data, 1.0)
    position = {gene: index for index, gene in enumerate(genes)}
    scores = pd.DataFrame(index=pd.RangeIndex(lognorm.shape[0]))
    for key, value in sets.items():
        dense = scaled[:, [position[gene] for gene in value]].toarray()
        k = min(top_k, dense.shape[1])
        scores[key] = np.sort(dense, axis=1)[:, -k:].mean(1)
    return scores


def p7513_pseudo_labels(
    counts: sparse.spmatrix,
    lognorm: sparse.spmatrix,
    var_names: Sequence[str],
    *,
    markers: Mapping[str, Sequence[str]] = P7513_PSEUDO_MARKERS,
    seed_min: float = 0.5,
    seed_other_ratio: float = 0.5,
    min_seeds: int = 30,
    post_min: float = 0.99,
    mixed_delta: float = 10.0,
    alpha: float = 0.5,
) -> pd.DataFrame:
    """Marker-seeded multinomial pseudo-labels (``data/P7513`` §C).

    Args:
        counts: Cells x genes raw counts (``layers["counts"]``).
        lognorm: The log-normalised matrix (``X``).
        var_names: Gene symbol per column.
        markers: Class to marker symbols.
        seed_min: Minimum top-2 score of a seed cell.
        seed_other_ratio: A seed's best other-group score is at most this
            fraction of its own.
        min_seeds: Classes with fewer seeds get no profile.
        post_min: Posterior needed for a label.
        mixed_delta: Nats by which a 50/50 mixture must beat the best class.
        alpha: Pseudo-count of the class profiles.

    Returns:
        Per cell: ``pseudo`` (fine class, ``Unresolved`` or ``Mixed``),
        ``pseudo_broad`` (``PSEUDO_TO_BROAD``) and ``nb_posterior``.
    """
    scores = p7513_marker_scores(lognorm, var_names, markers)
    classes = list(scores.columns)
    broad_of = np.array([P7513_PSEUDO_TO_BROAD[c] for c in classes])
    values = scores.to_numpy()
    seeds = {}
    for j, cls in enumerate(classes):
        other = values[:, broad_of != broad_of[j]].max(1)
        same_other = values[
            :, (broad_of == broad_of[j]) & (np.arange(len(classes)) != j)
        ]
        condition = (values[:, j] >= seed_min) & (
            other <= seed_other_ratio * values[:, j]
        )
        if same_other.shape[1]:
            condition &= same_other.max(1) <= seed_other_ratio * values[:, j]
        seeds[cls] = np.where(condition)[0]
    keep = [cls for cls in classes if len(seeds[cls]) >= min_seeds]
    raw = sparse.csr_matrix(counts).astype(np.float64)
    profiles = []
    for cls in keep:
        total = np.asarray(raw[seeds[cls]].sum(0)).ravel() + alpha
        profiles.append(total / total.sum())
    profile = np.vstack(profiles)
    loglik = np.asarray(raw @ np.log(profile).T)
    shifted = loglik - loglik.max(1, keepdims=True)
    posterior = np.exp(shifted)
    posterior /= posterior.sum(1, keepdims=True)
    best = np.argmax(loglik, 1)
    label = np.array(keep, dtype=object)[best]
    best_posterior = posterior[np.arange(len(posterior)), best]
    label = np.where(best_posterior >= post_min, label, "Unresolved")
    keep_broad = np.array([P7513_PSEUDO_TO_BROAD[c] for c in keep])
    best_mix = np.full(len(label), -np.inf)
    for i in range(len(keep)):
        for j in range(i + 1, len(keep)):
            if keep_broad[i] == keep_broad[j]:
                continue
            mixture = np.asarray(
                raw @ np.log(0.5 * profile[i] + 0.5 * profile[j])
            ).ravel()
            best_mix = np.maximum(best_mix, mixture)
    mixed = (best_mix - loglik.max(1)) >= mixed_delta
    label = np.where(mixed, "Mixed", label)
    return pd.DataFrame(
        {
            "pseudo": label,
            "pseudo_broad": [P7513_PSEUDO_TO_BROAD[str(x)] for x in label],
            "nb_posterior": best_posterior,
        }
    )


def p1212_pseudo_labels(
    counts: sparse.spmatrix,
    var_names: Sequence[str],
    *,
    markers: Mapping[str, Sequence[str]] = P1212_PSEUDO_MARKERS,
    min_score: float = 1.5,
    min_frac: float = 0.6,
) -> pd.DataFrame:
    """Marker-count pseudo-labels (``data/P1212`` §4).

    Args:
        counts: Cells x genes raw counts.
        var_names: Gene symbol per column.
        markers: Class to marker symbols.
        min_score: Units the top class needs.
        min_frac: Share of the summed class scores the top class needs.

    Returns:
        Per cell: ``pseudo8`` (class or ``Unassigned``), ``pseudo6``.
    """
    column = {str(name): index for index, name in enumerate(var_names)}
    classes = list(markers)
    matrix = sparse.csr_matrix(counts)
    scores = np.zeros((matrix.shape[0], len(classes)))
    for j, cls in enumerate(classes):
        index = [column[gene] for gene in markers[cls] if gene in column]
        sub = matrix[:, index].toarray().astype(float)
        positive_mean = np.array(
            [
                sub[:, k][sub[:, k] > 0].mean() if (sub[:, k] > 0).any() else 1.0
                for k in range(sub.shape[1])
            ]
        )
        scores[:, j] = (sub / positive_mean).sum(axis=1)
    total = scores.sum(axis=1)
    top = scores.argmax(axis=1)
    top_score = scores.max(axis=1)
    fraction = np.divide(
        top_score, total, out=np.zeros_like(top_score), where=total > 0
    )
    label = np.array(classes, dtype=object)[top]
    ok = (top_score >= min_score) & (fraction >= min_frac)
    label = np.where(ok, label, "Unassigned")
    return pd.DataFrame(
        {
            "pseudo8": label,
            "pseudo6": [P1212_PSEUDO6.get(str(x), "Unassigned") for x in label],
        }
    )


@dataclass(frozen=True)
class H9Result:
    """H9 agreement of one labelling with one pseudo-labelling.

    Attributes:
        method: ``p7513_c`` or ``p1212_4``.
        n_confident: Confident cells (any pseudo-label).
        n_scored: Confident cells with a resolved pseudo-label.
        agreement: Share of the scored cells whose 6-class label matches.
        by_class: Agreement and n per collapsed class of the labelling.
    """

    method: str
    n_confident: int
    n_scored: int
    agreement: float
    by_class: dict[str, dict[str, float]]

    def to_row(self) -> dict[str, Any]:
        """Return a flat CSV row."""
        row: dict[str, Any] = {
            "method": self.method,
            "n_confident": self.n_confident,
            "n_scored": self.n_scored,
            "agreement": self.agreement,
        }
        for cls, values in sorted(self.by_class.items()):
            row[f"agree_{cls}"] = values["agreement"]
            row[f"n_{cls}"] = values["n"]
        return row


def resolved_six(pseudo: pd.DataFrame, method: str) -> np.ndarray:
    """Return each cell's resolved 6-class pseudo-label (``None`` if unresolved).

    Args:
        pseudo: ``p7513_pseudo_labels`` or ``p1212_pseudo_labels`` output (or
            the reports' stored tables with the same columns).
        method: ``p7513_c`` or ``p1212_4``.

    Returns:
        Object array.
    """
    if method == "p7513_c":
        values = pseudo["pseudo_broad"].astype(object).to_numpy()
        unresolved = P7513_UNRESOLVED
    elif method == "p1212_4":
        values = pseudo["pseudo6"].astype(object).to_numpy()
        unresolved = P1212_UNRESOLVED
    else:
        raise ValueError(f"unknown pseudo-label method {method!r}")
    return np.array(
        [None if value in unresolved else value for value in values], dtype=object
    )


def h9_agreement(
    broad_names: Sequence[object] | np.ndarray,
    confident: np.ndarray,
    pseudo_six: np.ndarray,
    method: str,
) -> H9Result:
    """Score confident broad calls against resolved pseudo-labels (H9).

    Args:
        broad_names: Broad class per cell (our vocabulary).
        confident: Whether the cell's broad call is confident.
        pseudo_six: ``resolved_six`` output aligned with the cells.
        method: The pseudo-label method (recorded).

    Returns:
        The agreement.
    """
    names = np.asarray(broad_names, dtype=object)
    six = np.array([BROAD_TO_SIX.get(str(name)) for name in names], dtype=object)
    confident = np.asarray(confident, dtype=bool)
    resolved = np.array([value is not None for value in pseudo_six], dtype=bool)
    scored = confident & resolved
    same = np.array(
        [a is not None and a == b for a, b in zip(six, pseudo_six, strict=True)],
        dtype=bool,
    )
    by_class: dict[str, dict[str, float]] = {}
    for cls in sorted({value for value in six[scored] if value is not None}):
        mask = scored & (six == cls)
        by_class[str(cls)] = {
            "agreement": float(same[mask].mean()),
            "n": float(mask.sum()),
        }
    return H9Result(
        method=method,
        n_confident=int(confident.sum()),
        n_scored=int(scored.sum()),
        agreement=float(same[scored].mean()) if scored.any() else float("nan"),
        by_class=by_class,
    )


def main() -> int:
    """Print the marker sets of both methods."""
    for name, markers in (
        ("p7513_c", P7513_PSEUDO_MARKERS),
        ("p1212_4", P1212_PSEUDO_MARKERS),
    ):
        print(name)
        for cls, genes in markers.items():
            print(f"  {cls}: {', '.join(genes)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

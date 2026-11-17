"""Robust smooth landmark field: a Student-t Gaussian process over displacements.

The module fits a displacement field, reference micrometres to moving
micrometres, to landmark clicks and other point observations. It is pure
numpy/scipy (standard library, numpy and scipy only; no merxen import and no
pydantic) so that the viewer can mirror it later.

Model (all in micrometres; reference full-frame input, moving full-frame output):

* The displacement ``d = y - x`` is an affine trend (flat prior, universal
  kriging) plus a zero-mean Gaussian process with a sum of one or two
  Matern 5/2 components, shared by both axes.
* Each observation ``i`` has a class ``g`` (``click``, ``window``, ...), a unit
  determinant shape ``C_i`` and a fixed covariance ``F_i``; its noise is
  ``(sigma_g^2 C_i + F_i) / w_i`` with a Student-t EM weight ``w_i``.
* Hyperparameters maximise the REML likelihood (analytic gradient, L-BFGS-B from a
  fixed start grid). The weights come from a robust affine initialisation and EM.
* Closed-form leave-one-out residuals (Dubrule) give chi-square(2) p-values
  against the unit-weight noise, so a down-weighted misclick is still exposed.

Residuals are observed minus predicted, in moving micrometres.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)

FIELD_SCHEMA = "merxen.manual_registration.landmark_field"
FIELD_SCHEMA_VERSION = 1
FIELD_ALGORITHM = "robust_gp_matern52_v1"
TREND_SCALE_UM = 10000.0
JITTER_UM2 = 1e-6
MAX_SHAPE_ANISOTROPY = 16.0
DEFAULT_NOISE_BOUNDS_UM: dict[str, tuple[float, float]] = {
    "click": (3.0, 300.0),
    "window": (15.0, 100.0),
}
RIDGE_SHAPE_RATIO = 60.0 / 27.0

_SQRT5 = math.sqrt(5.0)
_CHI2_MEDIAN_SD = 1.1774100225154747  # sqrt(2 ln 2), the median of a chi(2) variable
_FAILED_OBJECTIVE = 1e15
_ROBUST_INIT_ITERATIONS = 20
_ROBUST_INIT_TOLERANCE = 1e-3
_DEFAULT_NOISE_GUESS_UM = {"click": 30.0, "window": 15.0}
_CLICK_LIKE_CLASSES = ("click", "confirmed")


# --------------------------------------------------------------------------- basics
def matern52(r: np.ndarray, length: float) -> np.ndarray:
    """Matern 5/2 correlation ``(1 + b + b^2 / 3) exp(-b)`` with ``b = sqrt(5) r / l``.

    Args:
        r: Distances (any shape), micrometres.
        length: Length scale, micrometres.

    Returns:
        Correlations in ``[0, 1]`` with the shape of ``r``.
    """
    b = _SQRT5 * np.asarray(r, dtype=float) / length
    return np.asarray((1.0 + b + b * b / 3.0) * np.exp(-b), dtype=float)


def isotropic_shapes(n: int) -> np.ndarray:
    """Return ``n`` identity shapes, ``(n, 2, 2)``."""
    return np.broadcast_to(np.eye(2), (n, 2, 2)).copy()


def ridge_shape(
    direction_xy: Sequence[float] | np.ndarray, ratio: float = RIDGE_SHAPE_RATIO
) -> np.ndarray:
    """Return the det-1 shape of a ridge click, ``R diag(ratio, 1 / ratio) R'``.

    The SD is ``sqrt(ratio)`` times sigma along the ridge and ``1 / sqrt(ratio)``
    across it.

    Args:
        direction_xy: Ridge direction in the moving frame (need not be unit).
        ratio: Eigenvalue of the along direction (its inverse is across).

    Returns:
        A ``(2, 2)`` symmetric positive definite matrix of determinant 1.

    Raises:
        ValueError: If the direction is zero or not finite, or the ratio is not
            positive.
    """
    direction = np.asarray(direction_xy, dtype=float).reshape(-1)
    if direction.shape != (2,) or not np.all(np.isfinite(direction)):
        raise ValueError("ridge direction must be two finite numbers")
    length = float(np.hypot(direction[0], direction[1]))
    if length == 0.0:
        raise ValueError("ridge direction must not be zero")
    if not ratio > 0.0:
        raise ValueError("ridge ratio must be positive")
    unit = direction / length
    outer = np.outer(unit, unit)
    return ratio * outer + (np.eye(2) - outer) / ratio


def normalise_shape(cov: np.ndarray) -> np.ndarray:
    """Scale covariance matrices ``(..., 2, 2)`` to determinant 1.

    Raises:
        ValueError: If a determinant is not positive.
    """
    array = np.asarray(cov, dtype=float)
    det = array[..., 0, 0] * array[..., 1, 1] - array[..., 0, 1] * array[..., 1, 0]
    if not np.all(det > 0.0):
        raise ValueError("shape covariance must have a positive determinant")
    return np.asarray(array / np.sqrt(det)[..., None, None], dtype=float)


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    """Digest recipe of ``merxen.manual_registration.schema.canonical_sha256``."""
    text = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sym_batch(matrices: np.ndarray) -> np.ndarray:
    return np.asarray(0.5 * (matrices + np.swapaxes(matrices, -1, -2)), dtype=float)


# ------------------------------------------------------------------ data containers
@dataclass(frozen=True)
class FieldObservations:
    """Point observations of the displacement field.

    Attributes:
        reference_xy_um: ``(N, 2)`` reference full-frame positions.
        moving_xy_um: ``(N, 2)`` observed moving full-frame positions.
        shape_cov: ``(N, 2, 2)`` det-1 noise shapes in the moving frame.
        fixed_cov_um2: ``(N, 2, 2)`` fixed noise covariances (zero for clicks).
        obs_class: ``(N,)`` class names.
        obs_id: ``(N,)`` pair ids for clicks, running indices for other classes.
        em_exempt: ``(N,)`` True keeps the Student-t weight at 1.
    """

    reference_xy_um: np.ndarray
    moving_xy_um: np.ndarray
    shape_cov: np.ndarray
    fixed_cov_um2: np.ndarray
    obs_class: tuple[str, ...]
    obs_id: np.ndarray
    em_exempt: np.ndarray | None = None

    def __post_init__(self) -> None:
        ref = np.array(self.reference_xy_um, dtype=float)
        mov = np.array(self.moving_xy_um, dtype=float)
        shape = np.array(self.shape_cov, dtype=float)
        fixed = np.array(self.fixed_cov_um2, dtype=float)
        ids = np.array(self.obs_id, dtype=np.int64)
        classes = tuple(str(name) for name in self.obs_class)
        count = len(classes)
        exempt = (
            np.zeros(count, dtype=bool)
            if self.em_exempt is None
            else np.array(self.em_exempt, dtype=bool)
        )
        expected = {
            "reference_xy_um": (ref, (count, 2)),
            "moving_xy_um": (mov, (count, 2)),
            "shape_cov": (shape, (count, 2, 2)),
            "fixed_cov_um2": (fixed, (count, 2, 2)),
            "obs_id": (ids, (count,)),
            "em_exempt": (exempt, (count,)),
        }
        for name, (array, wanted) in expected.items():
            if array.shape != wanted:
                raise ValueError(f"{name} has shape {array.shape}, expected {wanted}")
        for array in (ref, mov, shape, fixed, ids, exempt):
            array.setflags(write=False)
        object.__setattr__(self, "reference_xy_um", ref)
        object.__setattr__(self, "moving_xy_um", mov)
        object.__setattr__(self, "shape_cov", shape)
        object.__setattr__(self, "fixed_cov_um2", fixed)
        object.__setattr__(self, "obs_class", classes)
        object.__setattr__(self, "obs_id", ids)
        object.__setattr__(self, "em_exempt", exempt)

    @property
    def n(self) -> int:
        """Number of observations."""
        return len(self.obs_class)

    @classmethod
    def clicks(
        cls,
        reference_xy_um: np.ndarray,
        moving_xy_um: np.ndarray,
        ids: Sequence[int] | np.ndarray,
        shape_cov: np.ndarray | None = None,
        em_exempt: np.ndarray | None = None,
    ) -> FieldObservations:
        """Build click observations (class ``click``, zero fixed covariance).

        Args:
            reference_xy_um: ``(N, 2)`` reference positions.
            moving_xy_um: ``(N, 2)`` moving positions.
            ids: Pair ids.
            shape_cov: Optional ``(N, 2, 2)`` det-1 shapes (identity by default).
            em_exempt: Optional ``(N,)`` flags that keep the weight at 1.

        Returns:
            The observations.
        """
        count = len(np.asarray(reference_xy_um))
        return cls(
            reference_xy_um=np.asarray(reference_xy_um, dtype=float),
            moving_xy_um=np.asarray(moving_xy_um, dtype=float),
            shape_cov=isotropic_shapes(count) if shape_cov is None else shape_cov,
            fixed_cov_um2=np.zeros((count, 2, 2)),
            obs_class=("click",) * count,
            obs_id=np.asarray(ids, dtype=np.int64),
            em_exempt=em_exempt,
        )

    @classmethod
    def windows(
        cls,
        reference_xy_um: np.ndarray,
        moving_xy_um: np.ndarray,
        fixed_cov_um2: np.ndarray,
        *,
        first_id: int = 0,
    ) -> FieldObservations:
        """Build window observations (class ``window``, identity shapes).

        Args:
            reference_xy_um: ``(K, 2)`` reference positions.
            moving_xy_um: ``(K, 2)`` moving positions.
            fixed_cov_um2: ``(K, 2, 2)`` peak-region covariances (moving frame).
            first_id: Id of the first window; the others count up from it.

        Returns:
            The observations.
        """
        count = len(np.asarray(reference_xy_um))
        return cls(
            reference_xy_um=np.asarray(reference_xy_um, dtype=float),
            moving_xy_um=np.asarray(moving_xy_um, dtype=float),
            shape_cov=isotropic_shapes(count),
            fixed_cov_um2=np.asarray(fixed_cov_um2, dtype=float),
            obs_class=("window",) * count,
            obs_id=first_id + np.arange(count, dtype=np.int64),
        )

    def concat(self, other: FieldObservations) -> FieldObservations:
        """Return the observations of ``self`` followed by those of ``other``."""
        return FieldObservations(
            reference_xy_um=np.concatenate(
                [self.reference_xy_um, other.reference_xy_um]
            ),
            moving_xy_um=np.concatenate([self.moving_xy_um, other.moving_xy_um]),
            shape_cov=np.concatenate([self.shape_cov, other.shape_cov]),
            fixed_cov_um2=np.concatenate([self.fixed_cov_um2, other.fixed_cov_um2]),
            obs_class=self.obs_class + other.obs_class,
            obs_id=np.concatenate([self.obs_id, other.obs_id]),
            em_exempt=np.concatenate([self.em_exempt, other.em_exempt]),  # type: ignore[type-var]
        )

    def subset(self, mask: np.ndarray) -> FieldObservations:
        """Return the observations selected by a boolean mask or an index array."""
        selector = np.asarray(mask)
        if selector.dtype == bool:
            if selector.shape != (self.n,):
                raise ValueError("boolean mask must have one entry per observation")
            selector = np.flatnonzero(selector)
        return FieldObservations(
            reference_xy_um=self.reference_xy_um[selector],
            moving_xy_um=self.moving_xy_um[selector],
            shape_cov=self.shape_cov[selector],
            fixed_cov_um2=self.fixed_cov_um2[selector],
            obs_class=tuple(self.obs_class[int(i)] for i in selector),
            obs_id=self.obs_id[selector],
            em_exempt=self.em_exempt[selector],  # type: ignore[index]
        )


@dataclass(frozen=True)
class FieldBounds:
    """Hyperparameter bounds, micrometres (lower, upper)."""

    long_length_um: tuple[float, float] = (3000.0, 100000.0)
    long_signal_um: tuple[float, float] = (1.0, 5000.0)
    short_length_um: tuple[float, float] = (300.0, 3000.0)
    short_signal_um: tuple[float, float] = (0.5, 300.0)
    single_length_um: tuple[float, float] = (500.0, 100000.0)
    single_signal_um: tuple[float, float] = (1.0, 5000.0)


@dataclass(frozen=True)
class FieldOptions:
    """Options of a field fit.

    Attributes:
        components: Number of Matern components, or None for automatic (1 when every
            observation is a click, else 2).
        nu: Student-t degrees of freedom; ``math.inf`` is Gaussian (no EM).
        max_em_iterations: EM iteration cap.
        em_tolerance: EM stops when the largest weight change is below this.
        robust_init: Start from robust affine weights.
        bounds: Kernel bounds.
        noise_bounds_um: Per-class bounds of the noise sigma.
        max_optimizer_iterations: L-BFGS-B iteration cap per start.
    """

    components: int | None = None
    nu: float = 4.0
    max_em_iterations: int = 30
    em_tolerance: float = 3e-3
    robust_init: bool = True
    bounds: FieldBounds = field(default_factory=FieldBounds)
    noise_bounds_um: Mapping[str, tuple[float, float]] = field(
        default_factory=lambda: dict(DEFAULT_NOISE_BOUNDS_UM)
    )
    max_optimizer_iterations: int = 200


@dataclass(frozen=True)
class FieldHyperparameters:
    """Fitted (or chosen) kernel and noise parameters, micrometres."""

    long_length_um: float
    long_signal_um: float
    short_length_um: float | None
    short_signal_um: float | None
    noise_um: dict[str, float]


@dataclass(frozen=True)
class PointResidual:
    """Residuals of points against the field, with predictive covariances.

    Attributes:
        residual_um: ``(M, 2)`` observed minus predicted, moving micrometres.
        predicted_um: ``(M, 2)`` predictions.
        covariance_um2: ``(M, 2, 2)`` predictive covariance at unit weight.
        latent_covariance_um2: ``(M, 2, 2)`` latent posterior covariance.
        d2: ``(M,)`` squared Mahalanobis distance.
        p_value: ``(M,)`` ``exp(-d2 / 2)``, the chi-square(2) tail.
        residual_reference_um: ``(M, 2)`` residuals with the inverse trend Jacobian
            applied (reference-frame magnitudes).
    """

    residual_um: np.ndarray
    predicted_um: np.ndarray
    covariance_um2: np.ndarray
    latent_covariance_um2: np.ndarray
    d2: np.ndarray
    p_value: np.ndarray
    residual_reference_um: np.ndarray

    @property
    def magnitude_reference_um(self) -> np.ndarray:
        """Length of each reference-frame residual, micrometres."""
        return np.asarray(
            np.linalg.norm(self.residual_reference_um, axis=1), dtype=float
        )


@dataclass(frozen=True)
class LooResult(PointResidual):
    """Leave-one-out residuals of the observations of a fit (same fields)."""


@dataclass(frozen=True)
class Suggestion:
    """One suggested next click."""

    rank: int
    reference_xy_um: np.ndarray
    predicted_moving_xy_um: np.ndarray
    sd_um: float
    covariance_um2: np.ndarray
    weight: float
    score: float


@dataclass(frozen=True)
class FieldGrid:
    """The field on a regular grid of reference positions.

    Attributes:
        origin_um: Reference position of node ``(0, 0)``.
        step_um: Node spacing.
        mean_um: ``(H, W, 2)`` moving positions, NaN outside the mask.
        sd_um: ``(H, W)`` float32 per-axis SD, NaN outside the mask, or None.
        mask: ``(H, W)`` bool mask of the evaluated nodes, or None.
    """

    origin_um: tuple[float, float]
    step_um: float
    mean_um: np.ndarray
    sd_um: np.ndarray | None
    mask: np.ndarray | None


# ----------------------------------------------------------------------- validation
def _validate_observations(
    observations: FieldObservations, noise_bounds: Mapping[str, tuple[float, float]]
) -> None:
    obs = observations
    for name in ("reference_xy_um", "moving_xy_um", "shape_cov", "fixed_cov_um2"):
        bad = ~np.all(np.isfinite(getattr(obs, name)).reshape(obs.n, -1), axis=1)
        if bad.any():
            index = int(np.flatnonzero(bad)[0])
            raise ValueError(f"observation {index}: {name} is not finite")
    shape = obs.shape_cov
    asymmetry = np.abs(shape[:, 0, 1] - shape[:, 1, 0])
    for index in np.flatnonzero(asymmetry >= 1e-9):
        raise ValueError(f"observation {int(index)}: shape_cov is not symmetric")
    det = shape[:, 0, 0] * shape[:, 1, 1] - shape[:, 0, 1] ** 2
    eigenvalues = np.linalg.eigvalsh(_sym_batch(shape))
    for index in range(obs.n):
        if not eigenvalues[index, 0] > 0.0:
            raise ValueError(f"observation {index}: shape_cov is not positive definite")
        if abs(det[index] - 1.0) >= 1e-6:
            raise ValueError(
                f"observation {index}: shape_cov determinant is {det[index]:.6g}, not 1"
            )
        ratio = eigenvalues[index, 1] / eigenvalues[index, 0]
        if ratio > MAX_SHAPE_ANISOTROPY * (1.0 + 1e-9):
            raise ValueError(
                f"observation {index}: shape_cov anisotropy {ratio:.4g} exceeds "
                f"{MAX_SHAPE_ANISOTROPY:g}"
            )
    fixed = obs.fixed_cov_um2
    trace = np.trace(fixed, axis1=1, axis2=2)
    scale = np.maximum(1.0, trace)
    fixed_asymmetry = np.abs(fixed[:, 0, 1] - fixed[:, 1, 0])
    fixed_eigenvalues = np.linalg.eigvalsh(_sym_batch(fixed))
    for index in range(obs.n):
        if fixed_asymmetry[index] >= 1e-9 * scale[index]:
            raise ValueError(f"observation {index}: fixed_cov_um2 is not symmetric")
        if fixed_eigenvalues[index, 0] < -1e-9 * scale[index]:
            raise ValueError(
                f"observation {index}: fixed_cov_um2 is not positive semi-definite"
            )
    for index, name in enumerate(obs.obs_class):
        if name not in noise_bounds:
            raise ValueError(f"observation {index}: class {name!r} has no noise bounds")
    if obs.n < 4:
        raise ValueError(f"need at least 4 observations, got {obs.n}")
    centred = obs.reference_xy_um - obs.reference_xy_um.mean(axis=0)
    spread = np.linalg.eigvalsh(centred.T @ centred / obs.n)
    if not spread[1] > 0.0 or math.sqrt(max(spread[0], 0.0) / spread[1]) <= 1e-3:
        raise ValueError("reference positions are collinear")


def _validate_weights(weights: np.ndarray, count: int) -> np.ndarray:
    array = np.array(weights, dtype=float)
    if array.shape != (count,):
        raise ValueError(f"weights have shape {array.shape}, expected ({count},)")
    bad = ~(np.isfinite(array) & (array > 0.0))
    if bad.any():
        raise ValueError(
            f"observation {int(np.flatnonzero(bad)[0])}: weight must be > 0"
        )
    return array


def _resolve_components(observations: FieldObservations, options: FieldOptions) -> int:
    if options.components is None:
        return 1 if all(name == "click" for name in observations.obs_class) else 2
    if options.components not in (1, 2):
        raise ValueError(f"components must be None, 1 or 2, got {options.components}")
    return options.components


def _trend_centre(observations: FieldObservations) -> np.ndarray:
    clicks = np.array([name == "click" for name in observations.obs_class])
    positions = observations.reference_xy_um
    return np.asarray(positions[clicks if clicks.any() else slice(None)].mean(axis=0))


def _trend_basis(positions: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """``(M, 3)`` rows ``[1, zx, zy]`` of the scaled, centred positions."""
    scaled = (np.asarray(positions, dtype=float) - centre) / TREND_SCALE_UM
    return np.column_stack([np.ones(len(scaled)), scaled])


# ------------------------------------------------------------------- REML objective
@dataclass
class _Context:
    """Fixed inputs of the objective for one set of observations."""

    count: int
    distance: np.ndarray
    design: np.ndarray  # (2N, 6)
    target: np.ndarray  # (2N,)
    shape_flat: np.ndarray  # (N, 3) xx, xy, yy
    fixed_flat: np.ndarray  # (N, 3)
    class_names: tuple[str, ...]
    class_index: np.ndarray
    components: int
    log_bounds: list[tuple[float, float]]


def _flat_cov(matrices: np.ndarray) -> np.ndarray:
    return np.column_stack([matrices[:, 0, 0], matrices[:, 0, 1], matrices[:, 1, 1]])


def _make_context(
    observations: FieldObservations,
    centre: np.ndarray,
    components: int,
    options: FieldOptions,
) -> _Context:
    count = observations.n
    basis = _trend_basis(observations.reference_xy_um, centre)
    design = np.zeros((2 * count, 6))
    design[:count, :3] = basis
    design[count:, 3:] = basis
    displacement = observations.moving_xy_um - observations.reference_xy_um
    class_names = tuple(sorted(set(observations.obs_class)))
    lookup = {name: i for i, name in enumerate(class_names)}
    bounds = options.bounds
    if components == 2:
        kernel_bounds = [
            bounds.long_length_um,
            bounds.long_signal_um,
            bounds.short_length_um,
            bounds.short_signal_um,
        ]
    else:
        kernel_bounds = [bounds.single_length_um, bounds.single_signal_um]
    noise_bounds = [tuple(options.noise_bounds_um[name]) for name in class_names]
    log_bounds = [
        (math.log(lo), math.log(hi)) for lo, hi in kernel_bounds + noise_bounds
    ]
    return _Context(
        count=count,
        distance=cdist(observations.reference_xy_um, observations.reference_xy_um),
        design=design,
        target=np.concatenate([displacement[:, 0], displacement[:, 1]]),
        shape_flat=_flat_cov(observations.shape_cov),
        fixed_flat=_flat_cov(observations.fixed_cov_um2),
        class_names=class_names,
        class_index=np.array([lookup[name] for name in observations.obs_class]),
        components=components,
        log_bounds=log_bounds,
    )


def _noise_flat(
    context: _Context, sigma_by_class: np.ndarray, weights: np.ndarray | None
) -> np.ndarray:
    """``(N, 3)`` noise (xx, xy, yy) ``(sigma^2 C + F) / w`` of every observation."""
    sigma = sigma_by_class[context.class_index]
    flat = sigma[:, None] ** 2 * context.shape_flat + context.fixed_flat
    result = flat if weights is None else flat / weights[:, None]
    return np.asarray(result, dtype=float)


def _assemble(
    kernel_matrix: np.ndarray, noise_flat: np.ndarray, jitter: float
) -> np.ndarray:
    count = kernel_matrix.shape[0]
    full = np.zeros((2 * count, 2 * count))
    full[:count, :count] = kernel_matrix
    full[count:, count:] = kernel_matrix
    index = np.arange(count)
    full[index, index] += noise_flat[:, 0]
    full[count + index, count + index] += noise_flat[:, 2]
    full[index, count + index] = noise_flat[:, 1]
    full[count + index, index] = noise_flat[:, 1]
    full[np.arange(2 * count), np.arange(2 * count)] += jitter
    return full


def _kernel_parts(
    distance: np.ndarray, lengths: Sequence[float], signals: Sequence[float]
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Kernel matrix and its derivatives with respect to log length, log signal."""
    total = np.zeros_like(distance)
    derivatives: list[np.ndarray] = []
    for length, signal in zip(lengths, signals, strict=True):
        b = _SQRT5 * distance / length
        decay = np.exp(-b)
        correlation = (1.0 + b + b * b / 3.0) * decay
        total += signal**2 * correlation
        derivatives.append(signal**2 * decay * b * b * (1.0 + b) / 3.0)
        derivatives.append(2.0 * signal**2 * correlation)
    return total, derivatives


def _split_theta(
    theta: np.ndarray, context: _Context
) -> tuple[list[float], list[float], np.ndarray]:
    params = np.exp(theta)
    count = 2 * context.components
    lengths = [float(params[2 * c]) for c in range(context.components)]
    signals = [float(params[2 * c + 1]) for c in range(context.components)]
    return lengths, signals, params[count:]


def _reml(
    theta: np.ndarray, context: _Context, weights: np.ndarray
) -> tuple[float, np.ndarray]:
    """REML objective and its gradient with respect to ``theta = log(params)``."""
    count = context.count
    lengths, signals, sigma_by_class = _split_theta(theta, context)
    kernel_matrix, derivatives = _kernel_parts(context.distance, lengths, signals)
    noise = _noise_flat(context, sigma_by_class, weights)
    gradient_size = len(theta)
    covariance = _assemble(kernel_matrix, noise, JITTER_UM2)
    try:
        factor = cho_factor(covariance, lower=True)
    except (LinAlgError, ValueError):
        try:
            factor = cho_factor(
                _assemble(kernel_matrix, noise, 100.0 * JITTER_UM2), lower=True
            )
        except (LinAlgError, ValueError):
            return _FAILED_OBJECTIVE, np.zeros(gradient_size)
    design = context.design
    target = context.target
    inv_design = cho_solve(factor, design)
    inv_target = cho_solve(factor, target)
    info = design.T @ inv_design
    info = 0.5 * (info + info.T)
    try:
        info_factor = cho_factor(info, lower=True)
    except (LinAlgError, ValueError):
        return _FAILED_OBJECTIVE, np.zeros(gradient_size)
    beta = cho_solve(info_factor, design.T @ inv_target)
    projected = inv_target - inv_design @ beta  # P t
    log_det_k = 2.0 * float(np.sum(np.log(np.diag(factor[0]))))
    log_det_a = 2.0 * float(np.sum(np.log(np.diag(info_factor[0]))))
    objective = 0.5 * float(target @ projected) + 0.5 * log_det_k + 0.5 * log_det_a
    if not math.isfinite(objective):
        return _FAILED_OBJECTIVE, np.zeros(gradient_size)

    inverse = cho_solve(factor, np.eye(2 * count))
    projector = inverse - inv_design @ cho_solve(info_factor, inv_design.T)
    p_sum = projector[:count, :count] + projector[count:, count:]
    a_x = projected[:count]
    a_y = projected[count:]
    gradient = np.zeros(gradient_size)
    for j, derivative in enumerate(derivatives):
        quadratic = float(a_x @ derivative @ a_x + a_y @ derivative @ a_y)
        gradient[j] = -0.5 * quadratic + 0.5 * float(np.sum(p_sum * derivative))
    index = np.arange(count)
    p_xx = projector[index, index]
    p_xy = projector[index, count + index]
    p_yy = projector[count + index, count + index]
    # d noise / d log sigma_g = 2 sigma_g^2 C_i / w_i on the class-g observations.
    sigma_obs = sigma_by_class[context.class_index]
    scale = 2.0 * sigma_obs**2 / weights
    d_xx = scale * context.shape_flat[:, 0]
    d_xy = scale * context.shape_flat[:, 1]
    d_yy = scale * context.shape_flat[:, 2]
    quadratic_obs = a_x**2 * d_xx + 2.0 * a_x * a_y * d_xy + a_y**2 * d_yy
    trace_obs = p_xx * d_xx + 2.0 * p_xy * d_xy + p_yy * d_yy
    offset = len(derivatives)
    for class_number in range(len(context.class_names)):
        selected = context.class_index == class_number
        gradient[offset + class_number] = -0.5 * float(
            quadratic_obs[selected].sum()
        ) + 0.5 * float(trace_obs[selected].sum())
    if not np.all(np.isfinite(gradient)):
        return _FAILED_OBJECTIVE, np.zeros(gradient_size)
    return objective, gradient


def _theta_from_hyperparameters(
    hyper: FieldHyperparameters, context: _Context
) -> np.ndarray:
    values: list[float] = [hyper.long_length_um, hyper.long_signal_um]
    if context.components == 2:
        assert hyper.short_length_um is not None and hyper.short_signal_um is not None
        values += [hyper.short_length_um, hyper.short_signal_um]
    values += [hyper.noise_um[name] for name in context.class_names]
    return np.asarray(np.log(np.asarray(values, dtype=float)), dtype=float)


def _hyperparameters_from_theta(
    theta: np.ndarray, context: _Context
) -> FieldHyperparameters:
    lengths, signals, sigma_by_class = _split_theta(theta, context)
    return FieldHyperparameters(
        long_length_um=lengths[0],
        long_signal_um=signals[0],
        short_length_um=lengths[1] if context.components == 2 else None,
        short_signal_um=signals[1] if context.components == 2 else None,
        noise_um={
            name: float(sigma_by_class[i]) for i, name in enumerate(context.class_names)
        },
    )


# --------------------------------------------------------------- robust initialisation
def _noise_guesses(
    observations: FieldObservations, noise_bounds: Mapping[str, tuple[float, float]]
) -> dict[str, float]:
    guesses: dict[str, float] = {}
    for name in sorted(set(observations.obs_class)):
        low, high = noise_bounds[name]
        guess = _DEFAULT_NOISE_GUESS_UM.get(name, math.sqrt(low * high))
        guesses[name] = float(np.clip(guess, low, high))
    return guesses


def _robust_init(
    observations: FieldObservations,
    centre: np.ndarray,
    *,
    nu: float,
    noise_bounds: Mapping[str, tuple[float, float]],
    noise_guess_um: Mapping[str, float],
) -> tuple[np.ndarray, dict[str, float]]:
    """Robust affine weights and noise guesses (section 6.2 of the WP1 spec)."""
    count = observations.n
    basis = _trend_basis(observations.reference_xy_um, centre)
    displacement = observations.moving_xy_um - observations.reference_xy_um
    classes = np.array(observations.obs_class)
    exempt = observations.em_exempt
    assert exempt is not None
    guesses = {name: float(value) for name, value in noise_guess_um.items()}
    weights = np.ones(count)
    outer = basis[:, :, None] * basis[:, None, :]  # (N, 3, 3)
    for _ in range(_ROBUST_INIT_ITERATIONS):
        sigma = np.array([guesses[name] for name in observations.obs_class])
        noise = (
            sigma[:, None, None] ** 2 * observations.shape_cov
            + observations.fixed_cov_um2
        )
        inverse = np.linalg.inv(noise)
        weighted = weights[:, None, None] * inverse
        normal = np.zeros((6, 6))
        rhs = np.zeros(6)
        for a in range(2):
            for b in range(2):
                normal[3 * a : 3 * a + 3, 3 * b : 3 * b + 3] = np.einsum(
                    "i,ijk->jk", weighted[:, a, b], outer
                )
            rhs[3 * a : 3 * a + 3] = np.einsum(
                "ij,ib->j", basis, weighted[:, a, :] * displacement
            )
        beta = np.linalg.solve(normal, rhs).reshape(2, 3)
        residual = displacement - basis @ beta.T
        d2 = np.einsum("ia,iab,ib->i", residual, inverse, residual)
        distance = np.sqrt(np.maximum(d2, 0.0))
        for name in guesses:
            low, high = noise_bounds[name]
            median = float(np.median(distance[classes == name]))
            guesses[name] = float(
                np.clip(guesses[name] * median / _CHI2_MEDIAN_SD, low, high)
            )
        new_weights = (nu + 2.0) / (nu + d2)
        new_weights[exempt] = 1.0
        change = float(np.max(np.abs(new_weights - weights)))
        weights = new_weights
        if change < _ROBUST_INIT_TOLERANCE:
            break
    return weights, guesses


def robust_affine_weights(
    observations: FieldObservations,
    *,
    nu: float,
    noise_guess_um: Mapping[str, float],
) -> np.ndarray:
    """Student-t weights from a robust affine fit (the fit's initial weights).

    Args:
        observations: The observations.
        nu: Student-t degrees of freedom (finite).
        noise_guess_um: Initial noise sigma per class.

    Returns:
        ``(N,)`` weights; ``em_exempt`` observations stay at 1.
    """
    bounds = {name: (0.0, math.inf) for name in set(observations.obs_class)}
    for name in bounds:
        if name not in noise_guess_um:
            raise ValueError(f"no noise guess for class {name!r}")
    bounds = {name: (1e-6, 1e6) for name in bounds}
    weights, _ = _robust_init(
        observations,
        _trend_centre(observations),
        nu=nu,
        noise_bounds=bounds,
        noise_guess_um=noise_guess_um,
    )
    return weights


# ------------------------------------------------------------------------ posterior
class _Posterior:
    """Factorised model at fixed hyperparameters and weights."""

    def __init__(
        self,
        observations: FieldObservations,
        hyper: FieldHyperparameters,
        weights: np.ndarray,
        centre: np.ndarray,
        components: int,
    ) -> None:
        self.observations = observations
        self.centre = centre
        self.count = observations.n
        self.lengths = [hyper.long_length_um]
        self.signals = [hyper.long_signal_um]
        if components == 2:
            assert (
                hyper.short_length_um is not None and hyper.short_signal_um is not None
            )
            self.lengths.append(hyper.short_length_um)
            self.signals.append(hyper.short_signal_um)
        self.signal_variance = float(sum(s * s for s in self.signals))
        options = FieldOptions(components=components)
        context = _make_context(observations, centre, components, options)
        self.context = context
        sigma_by_class = np.array(
            [hyper.noise_um[name] for name in context.class_names]
        )
        self.sigma_by_class = sigma_by_class
        self.weights = weights
        self.noise_flat = _noise_flat(context, sigma_by_class, weights)
        self.unit_noise_flat = _noise_flat(context, sigma_by_class, None)
        kernel_matrix, _ = _kernel_parts(context.distance, self.lengths, self.signals)
        covariance = _assemble(kernel_matrix, self.noise_flat, JITTER_UM2)
        try:
            self.factor = cho_factor(covariance, lower=True)
        except (LinAlgError, ValueError) as exc:
            raise ValueError(
                "the landmark field covariance is not positive definite"
            ) from exc
        self.lower = np.tril(self.factor[0])
        self.design = context.design
        self.target = context.target
        self.inv_design = cho_solve(self.factor, self.design)  # K^-1 H
        info = self.design.T @ self.inv_design
        self.info = 0.5 * (info + info.T)
        self.info_factor = cho_factor(self.info, lower=True)
        self.info_inverse = cho_solve(self.info_factor, np.eye(6))
        inv_target = cho_solve(self.factor, self.target)
        self.beta = cho_solve(self.info_factor, self.design.T @ inv_target)
        self.projected = (
            inv_target - self.inv_design @ self.beta
        )  # P t = K^-1 (t - H beta)
        self.alpha = self.projected
        self.whitened_design = solve_triangular(self.lower, self.design, lower=True)
        self._projector: np.ndarray | None = None

    # --- evaluation
    def _cross_kernel(self, positions: np.ndarray) -> np.ndarray:
        distance = cdist(positions, self.observations.reference_xy_um)
        total = np.zeros_like(distance)
        for length, signal in zip(self.lengths, self.signals, strict=True):
            total += signal**2 * matern52(distance, length)
        return np.asarray(total, dtype=float)

    def trend_coefficients(self) -> np.ndarray:
        return np.asarray(self.beta).reshape(2, 3)

    def mean(self, positions: np.ndarray, chunk: int) -> np.ndarray:
        coefficients = self.trend_coefficients()
        out = np.empty((len(positions), 2))
        for start in range(0, len(positions), chunk):
            block = positions[start : start + chunk]
            basis = _trend_basis(block, self.centre)
            cross = self._cross_kernel(block)
            out[start : start + chunk, 0] = (
                block[:, 0] + basis @ coefficients[0] + cross @ self.alpha[: self.count]
            )
            out[start : start + chunk, 1] = (
                block[:, 1] + basis @ coefficients[1] + cross @ self.alpha[self.count :]
            )
        return out

    def covariance(self, positions: np.ndarray, chunk: int) -> np.ndarray:
        count = self.count
        out = np.empty((len(positions), 2, 2))
        for start in range(0, len(positions), chunk):
            block = positions[start : start + chunk]
            size = len(block)
            cross = self._cross_kernel(block).T  # (N, m)
            rhs_x = np.zeros((2 * count, size))
            rhs_x[:count] = cross
            rhs_y = np.zeros((2 * count, size))
            rhs_y[count:] = cross
            white_x = solve_triangular(self.lower, rhs_x, lower=True)
            white_y = solve_triangular(self.lower, rhs_y, lower=True)
            basis = _trend_basis(block, self.centre).T  # (3, m)
            r_x = np.zeros((6, size))
            r_x[:3] = basis
            r_x -= self.whitened_design.T @ white_x
            r_y = np.zeros((6, size))
            r_y[3:] = basis
            r_y -= self.whitened_design.T @ white_y
            a_rx = self.info_inverse @ r_x
            a_ry = self.info_inverse @ r_y
            out[start : start + chunk, 0, 0] = (
                self.signal_variance
                - np.sum(white_x * white_x, axis=0)
                + np.sum(r_x * a_rx, axis=0)
            )
            out[start : start + chunk, 1, 1] = (
                self.signal_variance
                - np.sum(white_y * white_y, axis=0)
                + np.sum(r_y * a_ry, axis=0)
            )
            cross_xy = -np.sum(white_x * white_y, axis=0) + np.sum(r_x * a_ry, axis=0)
            out[start : start + chunk, 0, 1] = cross_xy
            out[start : start + chunk, 1, 0] = cross_xy
        return out

    def projector(self) -> np.ndarray:
        """``P = K^-1 - K^-1 H A^-1 H' K^-1`` (bordered-inverse block)."""
        if self._projector is None:
            inverse = cho_solve(self.factor, np.eye(2 * self.count))
            self._projector = inverse - self.inv_design @ cho_solve(
                self.info_factor, self.inv_design.T
            )
        return self._projector

    def reference_jacobian_inverse(self) -> np.ndarray:
        """Inverse of the trend Jacobian ``J_T = I + B / S`` (reference <- moving)."""
        coefficients = self.trend_coefficients()
        jacobian = np.eye(2) + coefficients[:, 1:3] / TREND_SCALE_UM
        return np.linalg.inv(jacobian)

    def noise_unit_matrices(self) -> np.ndarray:
        flat = self.unit_noise_flat
        out = np.empty((self.count, 2, 2))
        out[:, 0, 0] = flat[:, 0]
        out[:, 0, 1] = out[:, 1, 0] = flat[:, 1]
        out[:, 1, 1] = flat[:, 2]
        return out

    def loo(self) -> LooResult:
        count = self.count
        projector = self.projector()
        index = np.arange(count)
        gram = np.empty((count, 2, 2))
        gram[:, 0, 0] = projector[index, index]
        gram[:, 0, 1] = projector[index, count + index]
        gram[:, 1, 0] = projector[count + index, index]
        gram[:, 1, 1] = projector[count + index, count + index]
        gram = _sym_batch(gram)
        pair = np.stack([self.projected[:count], self.projected[count:]], axis=1)
        predictive = _sym_batch(np.linalg.inv(gram))  # S_i
        residual = np.einsum("iab,ib->ia", predictive, pair)  # G_i^-1 u_i
        flat = self.noise_flat
        own = np.empty((count, 2, 2))
        own[:, 0, 0] = flat[:, 0]
        own[:, 0, 1] = own[:, 1, 0] = flat[:, 1]
        own[:, 1, 1] = flat[:, 2]
        latent = _sym_batch(predictive - own)
        unit = _sym_batch(latent + self.noise_unit_matrices())
        d2 = np.einsum("ia,iab,ib->i", residual, np.linalg.inv(unit), residual)
        d2 = np.maximum(d2, 0.0)
        observed = self.observations.moving_xy_um
        return LooResult(
            residual_um=residual,
            predicted_um=observed - residual,
            covariance_um2=unit,
            latent_covariance_um2=latent,
            d2=d2,
            p_value=np.exp(-0.5 * d2),
            residual_reference_um=residual @ self.reference_jacobian_inverse().T,
        )


# ----------------------------------------------------------------------------- field
class LandmarkField:
    """A fitted robust landmark field (use :func:`fit_landmark_field`).

    Attributes:
        observations: The observations the field was fitted to.
        hyperparameters: Kernel and noise parameters.
        weights: ``(N,)`` Student-t weights (1 for Gaussian fits and exempt points).
        options: The fit options.
        components: Resolved number of Matern components, 1 or 2.
        log_marginal_likelihood: Minus the REML objective at the optimum.
        em_iterations: EM iterations performed.
        converged: Whether EM reached its tolerance.
        n_starts: Number of hyperparameter starts of the first fit.
        centre_um: Trend centre, ``(2,)``.
        trend_coefficients: ``(2, 3)`` rows x, y over ``[1, zx, zy]`` (displacement).
    """

    def __init__(
        self,
        observations: FieldObservations,
        hyperparameters: FieldHyperparameters,
        weights: np.ndarray,
        options: FieldOptions,
        components: int,
        *,
        log_marginal_likelihood: float | None = None,
        em_iterations: int = 0,
        converged: bool = True,
        n_starts: int = 0,
    ) -> None:
        self.observations = observations
        self.hyperparameters = hyperparameters
        self.weights = np.array(weights, dtype=float)
        self.weights.setflags(write=False)
        self.options = options
        self.components = components
        self.em_iterations = em_iterations
        self.converged = converged
        self.n_starts = n_starts
        self.centre_um = _trend_centre(observations)
        self._posterior = _Posterior(
            observations, hyperparameters, self.weights, self.centre_um, components
        )
        self.trend_coefficients = self._posterior.trend_coefficients().copy()
        self._loo: LooResult | None = None
        self.log_marginal_likelihood = (
            -self.objective()
            if log_marginal_likelihood is None
            else log_marginal_likelihood
        )

    # --- prediction
    def mean(self, reference_xy_um: np.ndarray, *, chunk: int = 65536) -> np.ndarray:
        """Predicted moving positions, ``(M, 2)`` micrometres."""
        positions = np.atleast_2d(np.asarray(reference_xy_um, dtype=float))
        return self._posterior.mean(positions, chunk)

    def covariance(
        self, reference_xy_um: np.ndarray, *, chunk: int = 8192
    ) -> np.ndarray:
        """Latent posterior covariance (trend included), ``(M, 2, 2)`` um^2."""
        positions = np.atleast_2d(np.asarray(reference_xy_um, dtype=float))
        return self._posterior.covariance(positions, chunk)

    def sd_um(self, reference_xy_um: np.ndarray) -> np.ndarray:
        """Per-axis latent SD ``sqrt(trace / 2)``, ``(M,)`` micrometres."""
        covariance = self.covariance(reference_xy_um)
        return np.asarray(
            np.sqrt(np.maximum(np.trace(covariance, axis1=1, axis2=2) / 2.0, 0.0)),
            dtype=float,
        )

    def trend_matrix(self) -> np.ndarray:
        """The affine trend as a ``3 x 3`` matrix, reference um -> moving um."""
        coefficients = self.trend_coefficients
        linear = np.eye(2) + coefficients[:, 1:3] / TREND_SCALE_UM
        offset = (
            coefficients[:, 0] - coefficients[:, 1:3] @ self.centre_um / TREND_SCALE_UM
        )
        matrix = np.eye(3)
        matrix[:2, :2] = linear
        matrix[:2, 2] = offset
        return matrix

    def jacobian(
        self, reference_xy_um: np.ndarray, *, step_um: float = 5.0
    ) -> np.ndarray:
        """Jacobian of the mean by central differences, ``(M, 2, 2)`` (rows: output)."""
        positions = np.atleast_2d(np.asarray(reference_xy_um, dtype=float))
        out = np.empty((len(positions), 2, 2))
        for axis in range(2):
            shift = np.zeros(2)
            shift[axis] = step_um
            out[:, :, axis] = (
                self.mean(positions + shift) - self.mean(positions - shift)
            ) / (2.0 * step_um)
        return out

    def loo(self) -> LooResult:
        """Closed-form leave-one-out residuals of the observations (cached)."""
        if self._loo is None:
            self._loo = self._posterior.loo()
        return self._loo

    def predict_residual(
        self,
        reference_xy_um: np.ndarray,
        moving_xy_um: np.ndarray,
        *,
        obs_class: str = "click",
        shape_cov: np.ndarray | None = None,
        fixed_cov_um2: np.ndarray | None = None,
    ) -> PointResidual:
        """Residual of points that are not in the fit.

        Args:
            reference_xy_um: ``(M, 2)`` reference positions.
            moving_xy_um: ``(M, 2)`` observed moving positions.
            obs_class: Noise class of the points (must be a class of the fit).
            shape_cov: Optional det-1 shape, ``(2, 2)`` or ``(M, 2, 2)`` (identity).
            fixed_cov_um2: Optional fixed covariance, ``(2, 2)`` or ``(M, 2, 2)``.

        Returns:
            Residuals, unit-weight predictive covariances, ``d2`` and p-values.

        Raises:
            ValueError: If the class is unknown or shapes are inconsistent.
        """
        reference = np.atleast_2d(np.asarray(reference_xy_um, dtype=float))
        moving = np.atleast_2d(np.asarray(moving_xy_um, dtype=float))
        if reference.shape != moving.shape or reference.shape[1] != 2:
            raise ValueError("reference and moving positions must both be (M, 2)")
        if obs_class not in self.hyperparameters.noise_um:
            raise ValueError(f"class {obs_class!r} is not a class of this field")
        count = len(reference)
        shape = (
            np.broadcast_to(np.eye(2), (count, 2, 2))
            if shape_cov is None
            else np.broadcast_to(np.asarray(shape_cov, dtype=float), (count, 2, 2))
        )
        fixed = (
            np.zeros((count, 2, 2))
            if fixed_cov_um2 is None
            else np.broadcast_to(np.asarray(fixed_cov_um2, dtype=float), (count, 2, 2))
        )
        sigma = self.hyperparameters.noise_um[obs_class]
        predicted = self.mean(reference)
        residual = moving - predicted
        latent = self.covariance(reference)
        total = _sym_batch(latent + sigma**2 * shape + fixed)
        d2 = np.maximum(
            np.einsum("ia,iab,ib->i", residual, np.linalg.inv(total), residual), 0.0
        )
        return PointResidual(
            residual_um=residual,
            predicted_um=predicted,
            covariance_um2=total,
            latent_covariance_um2=latent,
            d2=d2,
            p_value=np.exp(-0.5 * d2),
            residual_reference_um=residual
            @ self._posterior.reference_jacobian_inverse().T,
        )

    def objective(self) -> float:
        """REML objective at the stored hyperparameters and weights."""
        context = self._posterior.context
        theta = _theta_from_hyperparameters(self.hyperparameters, context)
        return float(_reml(theta, context, self.weights)[0])

    # --- serialisation
    @property
    def digest(self) -> str:
        """The ``sha256:`` digest of the serialised model."""
        return str(self.to_dict()["digest"])

    def to_dict(self) -> dict[str, Any]:
        """Serialise to the version-1 JSON model of contract C1 (with its digest)."""
        obs = self.observations
        hyper = self.hyperparameters
        components = [
            {
                "name": "long",
                "length_um": float(hyper.long_length_um),
                "signal_um": float(hyper.long_signal_um),
            }
        ]
        if self.components == 2:
            components.append(
                {
                    "name": "short",
                    "length_um": float(hyper.short_length_um),  # type: ignore[arg-type]
                    "signal_um": float(hyper.short_signal_um),  # type: ignore[arg-type]
                }
            )
        options = self.options
        bounds = options.bounds
        payload: dict[str, Any] = {
            "schema": FIELD_SCHEMA,
            "schema_version": FIELD_SCHEMA_VERSION,
            "algorithm": FIELD_ALGORITHM,
            "frame": {"input": "reference_full_um", "output": "moving_full_um"},
            "trend": {
                "centre_um": [float(v) for v in self.centre_um],
                "scale_um": TREND_SCALE_UM,
                "coefficients": [
                    [float(v) for v in row] for row in self.trend_coefficients
                ],
            },
            "kernel": {"family": "matern52", "components": components},
            "noise_um": {name: float(v) for name, v in sorted(hyper.noise_um.items())},
            "student_t_nu": None if math.isinf(options.nu) else float(options.nu),
            "jitter_um2": JITTER_UM2,
            "observations": {
                "reference_xy_um": obs.reference_xy_um.tolist(),
                "moving_xy_um": obs.moving_xy_um.tolist(),
                "shape_cov": obs.shape_cov.tolist(),
                "fixed_cov_um2": obs.fixed_cov_um2.tolist(),
                "class": list(obs.obs_class),
                "id": [int(v) for v in obs.obs_id],
                "weight": [float(v) for v in self.weights],
                "em_exempt": [bool(v) for v in obs.em_exempt],  # type: ignore[union-attr]
            },
            "fit": {
                "log_marginal_likelihood": float(self.log_marginal_likelihood),
                "em_iterations": int(self.em_iterations),
                "converged": bool(self.converged),
                "n_starts": int(self.n_starts),
                "options": {
                    "components": options.components,
                    "nu": None if math.isinf(options.nu) else float(options.nu),
                    "robust_init": bool(options.robust_init),
                    "max_em_iterations": int(options.max_em_iterations),
                    "em_tolerance": float(options.em_tolerance),
                    "max_optimizer_iterations": int(options.max_optimizer_iterations),
                    "bounds": {
                        name: [float(low), float(high)]
                        for name, (low, high) in vars(bounds).items()
                    },
                    "noise_bounds_um": {
                        name: [float(low), float(high)]
                        for name, (low, high) in sorted(options.noise_bounds_um.items())
                    },
                },
            },
        }
        payload["digest"] = _canonical_sha256(payload)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> LandmarkField:
        """Rebuild a field from :meth:`to_dict` output and verify it.

        The factorisation is recomputed from the observations, hyperparameters and
        weights; it must reproduce the stored trend coefficients.

        Raises:
            ValueError: On a wrong schema or version, a digest mismatch, a malformed
                payload, or trend coefficients that differ from the stored ones.
        """
        try:
            return cls._from_dict(payload)
        except (KeyError, TypeError, IndexError) as exc:
            raise ValueError(f"malformed landmark field payload: {exc!r}") from exc

    @classmethod
    def _from_dict(cls, payload: Mapping[str, Any]) -> LandmarkField:
        if payload.get("schema") != FIELD_SCHEMA:
            raise ValueError(
                f"not a landmark field payload (schema {payload.get('schema')!r})"
            )
        if payload.get("schema_version") != FIELD_SCHEMA_VERSION:
            raise ValueError(
                "unsupported landmark field schema_version "
                f"{payload.get('schema_version')!r}"
            )
        body = {key: value for key, value in payload.items() if key != "digest"}
        try:
            expected = _canonical_sha256(body)
        except ValueError as exc:
            raise ValueError(f"payload is not plain JSON: {exc}") from exc
        if payload.get("digest") != expected:
            raise ValueError("landmark field digest does not match its content")
        stored = payload["observations"]
        observations = FieldObservations(
            reference_xy_um=np.array(stored["reference_xy_um"], dtype=float),
            moving_xy_um=np.array(stored["moving_xy_um"], dtype=float),
            shape_cov=np.array(stored["shape_cov"], dtype=float),
            fixed_cov_um2=np.array(stored["fixed_cov_um2"], dtype=float),
            obs_class=tuple(stored["class"]),
            obs_id=np.array(stored["id"], dtype=np.int64),
            em_exempt=np.array(
                stored.get("em_exempt", [False] * len(stored["class"])), dtype=bool
            ),
        )
        parts = payload["kernel"]["components"]
        hyper = FieldHyperparameters(
            long_length_um=float(parts[0]["length_um"]),
            long_signal_um=float(parts[0]["signal_um"]),
            short_length_um=float(parts[1]["length_um"]) if len(parts) == 2 else None,
            short_signal_um=float(parts[1]["signal_um"]) if len(parts) == 2 else None,
            noise_um={name: float(v) for name, v in payload["noise_um"].items()},
        )
        stored_options = payload["fit"]["options"]
        nu_value = stored_options["nu"]
        options = FieldOptions(
            components=stored_options["components"],
            nu=math.inf if nu_value is None else float(nu_value),
            max_em_iterations=int(stored_options["max_em_iterations"]),
            em_tolerance=float(stored_options["em_tolerance"]),
            robust_init=bool(stored_options["robust_init"]),
            bounds=FieldBounds(
                **{
                    name: (float(low), float(high))
                    for name, (low, high) in stored_options["bounds"].items()
                }
            ),
            noise_bounds_um={
                name: (float(low), float(high))
                for name, (low, high) in stored_options["noise_bounds_um"].items()
            },
            max_optimizer_iterations=int(
                stored_options.get("max_optimizer_iterations", 200)
            ),
        )
        weights = np.array(stored["weight"], dtype=float)
        rebuilt = landmark_field_from_hyperparameters(
            observations, hyper, weights, options
        )
        centre = np.array(payload["trend"]["centre_um"], dtype=float)
        if not np.allclose(centre, rebuilt.centre_um, rtol=1e-12, atol=1e-9):
            raise ValueError("trend centre does not match the observations")
        coefficients = np.array(payload["trend"]["coefficients"], dtype=float)
        if not np.allclose(
            rebuilt.trend_coefficients, coefficients, rtol=1e-8, atol=1e-6
        ):
            raise ValueError("trend coefficients do not match the recomputed fit")
        fit = payload["fit"]
        rebuilt.log_marginal_likelihood = float(fit["log_marginal_likelihood"])
        rebuilt.em_iterations = int(fit["em_iterations"])
        rebuilt.converged = bool(fit["converged"])
        rebuilt.n_starts = int(fit["n_starts"])
        rebuilt.trend_coefficients = coefficients
        return rebuilt


# ---------------------------------------------------------------------------- fitting
def landmark_field_from_hyperparameters(
    observations: FieldObservations,
    hyperparameters: FieldHyperparameters,
    weights: np.ndarray,
    options: FieldOptions | None = None,
) -> LandmarkField:
    """Build a field at fixed hyperparameters and weights (no optimisation, no EM).

    Args:
        observations: The observations.
        hyperparameters: Kernel and noise parameters; ``short_*`` None means one
            component.
        weights: ``(N,)`` positive weights.
        options: Options recorded with the field (defaults otherwise).

    Returns:
        The field.

    Raises:
        ValueError: If the inputs are invalid.
    """
    opts = options if options is not None else FieldOptions()
    _validate_observations(observations, opts.noise_bounds_um)
    components = 1 if hyperparameters.short_length_um is None else 2
    if opts.components is not None and opts.components != components:
        raise ValueError(
            f"options.components={opts.components} conflicts with the hyperparameters"
        )
    if hyperparameters.short_signal_um is None and components == 2:
        raise ValueError("short_signal_um is missing")
    for name in set(observations.obs_class):
        if name not in hyperparameters.noise_um:
            raise ValueError(f"hyperparameters have no noise for class {name!r}")
    checked = _validate_weights(weights, observations.n)
    return LandmarkField(observations, hyperparameters, checked, opts, components)


def _optimise(
    context: _Context,
    weights: np.ndarray,
    starts: Sequence[np.ndarray],
    max_iterations: int,
) -> tuple[np.ndarray, float]:
    lower = np.array([bound[0] for bound in context.log_bounds])
    upper = np.array([bound[1] for bound in context.log_bounds])
    best_theta: np.ndarray | None = None
    best_value = math.inf
    for start in starts:
        initial = np.clip(start, lower, upper)
        result = minimize(
            _reml,
            initial,
            args=(context, weights),
            jac=True,
            method="L-BFGS-B",
            bounds=context.log_bounds,
            options={"maxiter": max_iterations},
        )
        value = float(result.fun)
        if value < best_value:
            best_value = value
            best_theta = np.clip(np.asarray(result.x, dtype=float), lower, upper)
    if best_theta is None or best_value >= _FAILED_OBJECTIVE:
        raise ValueError("the landmark field hyperparameter fit failed")
    return best_theta, best_value


def _start_grid(context: _Context, noise_start: Sequence[float]) -> list[np.ndarray]:
    noise = np.log(np.asarray(noise_start, dtype=float))
    starts: list[np.ndarray] = []
    if context.components == 2:
        for long_length in (5000.0, 30000.0):
            for long_signal in (50.0, 500.0):
                for short_length in (800.0, 2000.0):
                    for short_signal in (10.0, 50.0):
                        kernel = np.log(
                            [long_length, long_signal, short_length, short_signal]
                        )
                        starts.append(np.concatenate([kernel, noise]))
    else:
        for length in (2000.0, 10000.0, 50000.0):
            for signal in (50.0, 500.0):
                starts.append(np.concatenate([np.log([length, signal]), noise]))
    return starts


def fit_landmark_field(
    observations: FieldObservations, options: FieldOptions | None = None
) -> LandmarkField:
    """Fit a robust landmark field by REML with Student-t EM weights.

    Args:
        observations: The observations (clicks, windows, ...).
        options: Fit options (defaults otherwise).

    Returns:
        The fitted field.

    Raises:
        ValueError: If the observations or options are invalid, or the
            optimisation fails for every start.
    """
    opts = options if options is not None else FieldOptions()
    _validate_observations(observations, opts.noise_bounds_um)
    if not (opts.nu > 0.0):
        raise ValueError("nu must be positive")
    components = _resolve_components(observations, opts)
    centre = _trend_centre(observations)
    context = _make_context(observations, centre, components, opts)
    is_robust = math.isfinite(opts.nu)
    exempt = observations.em_exempt
    assert exempt is not None
    guesses = _noise_guesses(observations, opts.noise_bounds_um)
    weights = np.ones(observations.n)
    if opts.robust_init and is_robust:
        weights, guesses = _robust_init(
            observations,
            centre,
            nu=opts.nu,
            noise_bounds=opts.noise_bounds_um,
            noise_guess_um=guesses,
        )
    noise_start = [guesses[name] for name in context.class_names]
    starts = _start_grid(context, noise_start)
    theta, value = _optimise(context, weights, starts, opts.max_optimizer_iterations)
    iterations = 0
    converged = True
    if is_robust:
        converged = False
        for step in range(1, opts.max_em_iterations + 1):
            iterations = step
            hyper = _hyperparameters_from_theta(theta, context)
            posterior = _Posterior(observations, hyper, weights, centre, components)
            latent = posterior.covariance(observations.reference_xy_um, 8192)
            fitted = posterior.mean(observations.reference_xy_um, 65536)
            residual = observations.moving_xy_um - fitted
            unit = posterior.noise_unit_matrices()
            inverse = np.linalg.inv(unit)
            outer = residual[:, :, None] * residual[:, None, :] + latent
            d2_em = np.einsum("iab,iba->i", inverse, outer)
            new_weights = (opts.nu + 2.0) / (opts.nu + np.maximum(d2_em, 0.0))
            new_weights[exempt] = 1.0
            change = float(np.max(np.abs(new_weights - weights)))
            weights = new_weights
            theta, value = _optimise(
                context, weights, [theta], opts.max_optimizer_iterations
            )
            if change < opts.em_tolerance:
                converged = True
                break
        if not converged:
            logger.warning(
                "landmark field EM did not converge in %d iterations",
                opts.max_em_iterations,
            )
    hyper = _hyperparameters_from_theta(theta, context)
    return LandmarkField(
        observations,
        hyper,
        weights,
        opts,
        components,
        log_marginal_likelihood=-value,
        em_iterations=iterations,
        converged=converged,
        n_starts=len(starts),
    )


# --------------------------------------------------------------------------- grids
def evaluate_on_grid(
    field: LandmarkField,
    *,
    origin_um: tuple[float, float],
    step_um: float,
    shape_yx: tuple[int, int],
    mask: np.ndarray | None = None,
    with_sd: bool = True,
) -> FieldGrid:
    """Evaluate the mean (and SD) on a regular grid of reference positions.

    Node ``(i, j)`` sits at ``origin + step * (j, i)``.

    Args:
        field: The fitted field.
        origin_um: Reference position of node ``(0, 0)``.
        step_um: Node spacing.
        shape_yx: Grid size ``(H, W)``.
        mask: Optional ``(H, W)`` bool; nodes outside it get NaN.
        with_sd: Also compute the per-axis SD.

    Returns:
        The grid.
    """
    height, width = shape_yx
    if mask is not None and np.asarray(mask).shape != (height, width):
        raise ValueError("mask must have the grid shape")
    selected = (
        np.ones((height, width), dtype=bool) if mask is None else np.asarray(mask, bool)
    )
    rows, columns = np.nonzero(selected)
    positions = np.column_stack(
        [origin_um[0] + step_um * columns, origin_um[1] + step_um * rows]
    ).astype(float)
    mean = np.full((height, width, 2), np.nan)
    sd = np.full((height, width), np.nan, dtype=np.float32) if with_sd else None
    if len(positions):
        mean[rows, columns] = field.mean(positions)
        if sd is not None:
            sd[rows, columns] = field.sd_um(positions).astype(np.float32)
    return FieldGrid(
        origin_um=(float(origin_um[0]), float(origin_um[1])),
        step_um=float(step_um),
        mean_um=mean,
        sd_um=sd,
        mask=None if mask is None else selected,
    )


# ------------------------------------------------------------------------ suggestions
def suggest_landmarks(
    field: LandmarkField,
    candidates_reference_um: np.ndarray,
    candidate_weights: np.ndarray,
    *,
    n: int = 20,
    click_noise_um: float | None = None,
    min_spacing_um: float = 300.0,
    exclusion_radius_um: float = 300.0,
) -> list[Suggestion]:
    """Greedily pick the next clicks that most reduce the weighted field SD.

    Each pick maximises ``weight * SD``; the posterior is then conditioned on a
    hypothetical click there (noise ``click_noise_um`` per axis), so later picks
    spread out. Candidates within ``exclusion_radius_um`` of an existing click and
    candidates with non-positive weight are dropped first; each pick also drops the
    candidates within ``min_spacing_um`` of it.

    Args:
        field: The fitted field.
        candidates_reference_um: ``(M, 2)`` candidate reference positions.
        candidate_weights: ``(M,)`` weights (e.g. 1 in grey matter).
        n: Maximum number of suggestions.
        click_noise_um: Noise sigma of the hypothetical clicks (default: the
            fitted click sigma).
        min_spacing_um: Minimum spacing between suggestions.
        exclusion_radius_um: Exclusion radius around existing clicks.

    Returns:
        Up to ``n`` suggestions, rank 1 first.

    Raises:
        ValueError: If the inputs are inconsistent or no click noise is known.
    """
    candidates = np.atleast_2d(np.asarray(candidates_reference_um, dtype=float))
    weights = np.asarray(candidate_weights, dtype=float)
    if (
        candidates.ndim != 2
        or candidates.shape[1] != 2
        or weights.shape != (len(candidates),)
    ):
        raise ValueError("candidates must be (M, 2) with one weight each")
    noise = field.hyperparameters.noise_um
    if click_noise_um is None:
        known = [name for name in _CLICK_LIKE_CLASSES if name in noise]
        if not known:
            raise ValueError(
                "click_noise_um is required when the fit has no click class"
            )
        click_noise_um = noise[known[0]]
    posterior = field._posterior
    count = len(candidates)
    active = weights > 0.0
    clicks = np.array(
        [name in _CLICK_LIKE_CLASSES for name in field.observations.obs_class]
    )
    if clicks.any() and count:
        nearest = cdist(candidates, field.observations.reference_xy_um[clicks]).min(
            axis=1
        )
        active &= nearest > exclusion_radius_um
    if not active.any():
        return []
    diag = field.covariance(candidates)
    noise_matrix = click_noise_um**2 * np.eye(2)
    inv_design = posterior.inv_design
    big_n = posterior.count
    basis_all = _trend_basis(candidates, posterior.centre)  # (M, 3)
    picks: list[int] = []
    cross_cache: list[np.ndarray] = []  # E_m(a) = Sigma_{m-1}(a, s_m), (M, 2, 2)
    inverse_cache: list[np.ndarray] = []  # T_m
    suggestions: list[Suggestion] = []
    for rank in range(1, n + 1):
        if not active.any():
            break
        trace = np.trace(diag, axis1=1, axis2=2)
        sd = np.sqrt(np.maximum(trace / 2.0, 0.0))
        score = np.where(active, weights * sd, -np.inf)
        pick = int(np.argmax(score))
        pick_covariance = diag[pick].copy()
        predicted = field.mean(candidates[[pick]])[0]
        suggestions.append(
            Suggestion(
                rank=rank,
                reference_xy_um=candidates[pick].copy(),
                predicted_moving_xy_um=predicted,
                sd_um=float(sd[pick]),
                covariance_um2=pick_covariance,
                weight=float(weights[pick]),
                score=float(score[pick]),
            )
        )
        picks.append(pick)
        if rank == n:
            break
        # Base posterior cross-covariance Sigma_0(a, s) of all candidates with the pick.
        pick_position = candidates[pick][None, :]
        cross_pick = posterior._cross_kernel(pick_position)[0]  # (N,)
        rhs = np.zeros((2 * big_n, 2))
        rhs[:big_n, 0] = cross_pick
        rhs[big_n:, 1] = cross_pick
        solved = cho_solve(posterior.factor, rhs)  # K^-1 K*_s, (2N, 2)
        pick_basis = basis_all[pick]
        h_pick = np.zeros((6, 2))
        h_pick[:3, 0] = pick_basis
        h_pick[3:, 1] = pick_basis
        r_pick = h_pick - inv_design.T @ rhs  # (6, 2)
        a_r_pick = posterior.info_inverse @ r_pick
        base = np.empty((count, 2, 2))
        distance_pick = np.linalg.norm(candidates - candidates[pick], axis=1)
        kernel_pick = np.zeros(count)
        for length, signal in zip(posterior.lengths, posterior.signals, strict=True):
            kernel_pick += signal**2 * matern52(distance_pick, length)
        for start in range(0, count, 8192):
            block = candidates[start : start + 8192]
            cross = posterior._cross_kernel(block)  # (m, N)
            part = np.empty((len(block), 2, 2))
            part[:, 0, :] = cross @ solved[:big_n]
            part[:, 1, :] = cross @ solved[big_n:]
            r_x = np.zeros((6, len(block)))
            r_x[:3] = basis_all[start : start + 8192].T
            r_x -= inv_design[:big_n].T @ cross.T
            r_y = np.zeros((6, len(block)))
            r_y[3:] = basis_all[start : start + 8192].T
            r_y -= inv_design[big_n:].T @ cross.T
            correction = np.empty((len(block), 2, 2))
            correction[:, 0, :] = r_x.T @ a_r_pick
            correction[:, 1, :] = r_y.T @ a_r_pick
            base[start : start + 8192] = -part + correction
        base[:, 0, 0] += kernel_pick
        base[:, 1, 1] += kernel_pick
        # Condition on the earlier picks: Sigma_{k-1}(a, s_k).
        current = base
        for cross_m, inverse_m in zip(cross_cache, inverse_cache, strict=True):
            # Sigma_{m-1}(s_m, s_k) is the transpose of E_m evaluated at the pick.
            current = current - np.einsum(
                "iab,bc,dc->iad", cross_m, inverse_m, cross_m[pick]
            )
        total = pick_covariance + noise_matrix
        inverse = np.linalg.inv(total)
        diag = diag - np.einsum("iab,bc,idc->iad", current, inverse, current)
        diag = _sym_batch(diag)
        cross_cache.append(current)
        inverse_cache.append(inverse)
        active[pick] = False
        active &= np.linalg.norm(candidates - candidates[pick], axis=1) > min_spacing_um
    return suggestions

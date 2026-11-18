"""Robust landmark field of a pair: settings rules, fit, classes and the check payload.

The field itself lives in ``landmark_field`` and the image evidence in
``image_evidence``; this module joins them with the saved landmarks. It is used by
``merxen manual-registration-check`` and by the register path, which also turns the
field into a pull flow (``field_flow``) and a base affine (``best_affine``).

Frames (contract C2): reference and moving positions are full-frame micrometres
(level-0 pixels times the pixel size); the field maps reference to moving. Per-pair
magnitudes (``loo_reference_um``) are in reference micrometres; residual vectors
are in moving micrometres.
"""

from __future__ import annotations

import logging
import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

import merxen
from merxen.manual_registration.config import ManualRegistrationSettings
from merxen.manual_registration.image_evidence import (
    IMAGE_EVIDENCE_ALGORITHM,
    ChannelRoles,
    ClickCheck,
    Representation,
    RoleImages,
    WindowObservations,
    WindowParams,
    check_clicks,
    coarse_role_images,
    compute_representation,
    load_role_images,
    resolve_channel_roles,
    target_weight_map,
    tissue_window_observations,
    warp_role_images,
)
from merxen.manual_registration.landmark_field import (
    FIELD_ALGORITHM,
    FieldObservations,
    FieldOptions,
    LandmarkField,
    PointResidual,
    Suggestion,
    fit_landmark_field,
    ridge_shape,
)
from merxen.manual_registration.presets import PRESET_VERSIONS
from merxen.manual_registration.register import LandmarkFitReport
from merxen.manual_registration.schema import (
    CHECK_CLASSES,
    CHECK_SCHEMA,
    CHECK_VERDICTS,
    SCHEMA_VERSION,
    CropRect,
    LabellingAttrs,
    StoreAttrs,
    canonical_json,
    canonical_sha256,
    parse_check_attrs,
    utc_now,
)

logger = logging.getLogger(__name__)

WINDOW_SIZES_UM: tuple[float, ...] = (600.0, 800.0)  # tests monkeypatch smaller sizes
GROSS_P = 1e-6
GROSS_WEIGHT = 0.05
# A3: strong image confirmations get their own noise class from this many on.
MIN_CONFIRMED_FOR_CLASS = 5
_SIMILARITY_OUTLIER_WARNING = re.compile(r"flagged as outlier and (excluded|kept):")

ROLES_EXAMPLE = "at8=AF568;map2=AF647;dapi=DAPI"
COARSE_PIXEL_SIZE_UM = 22.0


def resolve_store_roles(text: str, store: StoreAttrs) -> ChannelRoles | None:
    """Resolve a channel-roles text against the channels of a prepared store.

    Args:
        text: The roles text (C6 grammar); blank gives no roles.
        store: Attrs of the prepared store.

    Returns:
        The resolved roles, or None when the text is blank.

    Raises:
        ValueError: On a roles error; the message starts with ``channel roles: ``.
    """
    if not text.strip():
        return None

    def names(key: str) -> list[str]:
        record = store.images[key]
        return list(record.canonical_channel_names or record.channel_names)

    return resolve_channel_roles(
        text, reference_channels=names("reference"), moving_channels=names("moving")
    )


def validate_field_settings(
    settings: ManualRegistrationSettings,
    *,
    preset: str,
    roles: ChannelRoles | None,
    run_id: str,
    purpose: Literal["register", "check"],
) -> None:
    """Check the cross-field rules of the robust-field settings.

    Args:
        settings: The resolved settings.
        preset: The preset name.
        roles: The resolved channel roles, or None when none were given.
        run_id: Prefix of every message.
        purpose: ``register`` for a registration run, ``check`` for the landmark
            check (which always fits the robust field and never registers, so the
            refinement, flow and materialize settings do not matter).

    Raises:
        ValueError: With the ``[run_id]`` prefix when a rule is broken.
    """
    pfx = f"[{run_id}]"
    if purpose == "check":
        if (settings.field_image_check or settings.field_window_observations) and (
            roles is None
        ):
            raise ValueError(
                f"{pfx} the image check and tissue windows need channel roles: pass "
                f"--channel-roles '{ROLES_EXAMPLE}', or --no-image-check --no-windows "
                "for a landmarks-only check"
            )
        return
    is_field = settings.field_model == "robust_gp"
    if settings.materialize != "full" and not is_field:
        raise ValueError(
            f"{pfx} materialize={settings.materialize} (preview and none) is only "
            "for robust-field runs (field_model=robust_gp)"
        )
    if is_field and (settings.refine_affine or settings.nonrigid_enabled):
        raise ValueError(
            f"{pfx} cortex field runs skip the DAPI refinement and TV-L1 (plan D4); "
            "set refine_affine=false and nonrigid_enabled=false"
        )
    if preset == "cortex_serial" and roles is None:
        raise ValueError(
            f"{pfx} preset cortex_serial needs channel roles: set the samplesheet "
            f"column channel_roles, e.g. {ROLES_EXAMPLE}"
        )
    if (
        is_field
        and (settings.field_image_check or settings.field_window_observations)
        and roles is None
    ):
        raise ValueError(
            f"{pfx} the robust field with the image check or tissue windows needs "
            f"channel roles: set the samplesheet column channel_roles, e.g. "
            f"{ROLES_EXAMPLE}"
        )


# ---------------------------------------------------------------------------
# Options and the fit
# ---------------------------------------------------------------------------


@dataclass
class ImageContext:
    """Image evidence inputs of a field fit, kept for the suggestions and the figure.

    Attributes:
        roles: The resolved channel roles.
        reference: Representation of the reference image.
        moving_warped: Representation of the moving image warped by ``pull_um``.
        reference_images: The reference role images (working grid).
        pull_um: The clicks-only field mean (the warp prior): reference full um to
            moving full um.
    """

    roles: ChannelRoles
    reference: Representation
    moving_warped: Representation
    reference_images: RoleImages
    pull_um: Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True, eq=False)
class FieldFitReport:
    """The robust field of a pair with every per-pair diagnostic.

    All ``(P,)`` arrays follow the labelling order and cover every pair (disabled
    and excluded ones too). Magnitudes are reference micrometres.

    Attributes:
        similarity: The robust similarity fit (``register.fit_landmarks``).
        field: The final field.
        clicks_only: The pass-1 field (the same object as ``field`` without a second
            pass).
        has_second_pass: Whether image evidence changed the field.
        pair_ids: Pair ids.
        observed: Pair is a click observation of the field.
        loo_reference_um: Final field: leave-one-out (observed pairs) or residual
            against the field (the others).
        loo_vector_moving_um: ``(P, 2)`` the same residual vector, moving um.
        p_value: Final field p-value at unit weight.
        weight: Final field weight, NaN when not observed.
        clicks_only_loo_reference_um: As ``loo_reference_um`` for pass 1.
        clicks_only_p_value: As ``p_value`` for pass 1.
        clicks_only_weight: As ``weight`` for pass 1.
        verdict: Image verdict per pair (``not_checked`` without an image check).
        pair_class: Class per pair (contract C7).
        checks: Image checks by pair id.
        windows: The window observations, or None.
        em_exempt: Observation exempted from the Student-t weight (A3).
        confirm_strength: ``strong``/``weak`` for a confirmed verdict, else None.
        warnings: Warnings of the fit.
        timings_s: Seconds per step.
    """

    similarity: LandmarkFitReport
    field: LandmarkField
    clicks_only: LandmarkField
    has_second_pass: bool
    pair_ids: np.ndarray
    observed: np.ndarray
    loo_reference_um: np.ndarray
    loo_vector_moving_um: np.ndarray
    p_value: np.ndarray
    weight: np.ndarray
    clicks_only_loo_reference_um: np.ndarray
    clicks_only_p_value: np.ndarray
    clicks_only_weight: np.ndarray
    verdict: tuple[str, ...]
    pair_class: tuple[str, ...]
    checks: dict[int, ClickCheck]
    windows: WindowObservations | None
    em_exempt: np.ndarray
    confirm_strength: tuple[str | None, ...]
    warnings: tuple[str, ...]
    timings_s: dict[str, float]

    @property
    def n_windows(self) -> int:
        """Number of window observations in the final field."""
        return 0 if self.windows is None else len(self.windows.reference_xy_um)


def field_options(settings: ManualRegistrationSettings) -> FieldOptions:
    """Return the field fit options of the settings.

    ``field_kernel`` ``auto`` leaves the component count to the fit (one for clicks
    alone, two with windows); ``single`` and ``two_scale`` force 1 and 2.
    """
    components = {"auto": None, "single": 1, "two_scale": 2}[settings.field_kernel]
    return FieldOptions(
        components=components,
        nu=settings.field_student_t_nu,
        noise_bounds_um={
            "click": (3.0, 300.0),
            "confirmed": (3.0, 100.0),
            "window": (settings.field_window_noise_floor_um, 100.0),
        },
    )


def window_params(settings: ManualRegistrationSettings) -> WindowParams:
    """Return the tissue-window parameters of the settings (other values default)."""
    return WindowParams(
        sizes_um=WINDOW_SIZES_UM,
        step_um=settings.field_window_step_um,
        search_um=settings.field_window_search_um,
        thinning_um=settings.field_window_min_spacing_um,
    )


def _elapsed(timings: dict[str, float], name: str, started: float) -> None:
    timings[name] = round(time.perf_counter() - started, 3)
    logger.info("landmark field: %s took %.1f s", name, timings[name])


def _per_pair(
    field: LandmarkField,
    *,
    reference_um: np.ndarray,
    moving_um: np.ndarray,
    observation_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Residual magnitude, vector, p-value and weight of every pair for one field.

    Args:
        field: The field.
        reference_um: ``(P, 2)`` reference positions of all pairs.
        moving_um: ``(P, 2)`` moving positions of all pairs.
        observation_index: ``(P,)`` index of the pair among the field's click
            observations (they come first), or -1 for a pair that is not one.

    Returns:
        ``(magnitude_reference_um, vector_moving_um, p_value, weight)``; the weight
        is NaN for a pair that is not an observation.
    """
    count = len(reference_um)
    magnitude = np.empty(count)
    vector = np.empty((count, 2))
    p_value = np.empty(count)
    weight = np.full(count, np.nan)
    observed = observation_index >= 0
    if observed.any():
        loo = field.loo()
        rows = observation_index[observed]
        magnitude[observed] = loo.magnitude_reference_um[rows]
        vector[observed] = loo.residual_um[rows]
        p_value[observed] = loo.p_value[rows]
        weight[observed] = field.weights[rows]
    if (~observed).any():
        names = field.hyperparameters.noise_um
        name = "click" if "click" in names else "confirmed"
        other: PointResidual = field.predict_residual(
            reference_um[~observed], moving_um[~observed], obs_class=name
        )
        magnitude[~observed] = other.magnitude_reference_um
        vector[~observed] = other.residual_um
        p_value[~observed] = other.p_value
    return magnitude, vector, p_value, weight


def build_landmark_field(
    *,
    similarity: LandmarkFitReport,
    store: StoreAttrs,
    settings: ManualRegistrationSettings,
    roles: ChannelRoles | None,
    crops: Mapping[str, CropRect],
    image_check: bool,
    windows: bool,
    run_id: str,
) -> tuple[FieldFitReport, ImageContext | None]:
    """Fit the robust landmark field with image evidence and classify every pair.

    Pass 1 fits the clicks alone. When roles are given and the image check or the
    windows are on, the moving images are warped by the pass-1 mean, every click is
    checked, and tissue windows are matched. Pass 2 then refits with ridge shapes,
    the windows and (setting ``field_trust_confirmed_clicks``) EM-exempt strong
    confirmations; without any such evidence pass 2 is pass 1.

    Args:
        similarity: The robust similarity fit of the same labelling.
        store: Attrs of the prepared store.
        settings: The resolved settings.
        roles: Channel roles; None means no image evidence.
        crops: Effective (snapped) crops by ``reference`` and ``moving``.
        image_check: Check every click against the images.
        windows: Add tissue-window observations.
        run_id: Prefix of messages.

    Returns:
        The report and the image context (None without image evidence).

    Raises:
        ValueError: If image evidence is asked for without roles, or the field
            cannot be fitted.
    """
    pfx = f"[{run_id}]"
    timings: dict[str, float] = {}
    want_images = image_check or windows
    if want_images and roles is None:
        raise ValueError(f"{pfx} image evidence needs channel roles")
    pairs = similarity.pairs
    ids = np.array([pair.id for pair in pairs], dtype=np.int64)
    pixel_ref = store.images["reference"].pixel_size_um
    pixel_mov = store.images["moving"].pixel_size_um
    reference_um = (
        np.array([pair.reference_xy_px for pair in pairs], dtype=float).reshape(-1, 2)
        * pixel_ref
    )
    moving_um = (
        np.array([pair.moving_xy_px for pair in pairs], dtype=float).reshape(-1, 2)
        * pixel_mov
    )
    fit_index = np.asarray(similarity.fit_index, dtype=int)
    n_clicks = len(fit_index)
    observed = np.zeros(len(pairs), dtype=bool)
    observed[fit_index] = True
    observation_index = np.full(len(pairs), -1, dtype=int)
    observation_index[fit_index] = np.arange(n_clicks)
    options = field_options(settings)

    started = time.perf_counter()
    pass1 = fit_landmark_field(
        FieldObservations.clicks(
            reference_um[fit_index], moving_um[fit_index], ids[fit_index]
        ),
        options,
    )
    _elapsed(timings, "clicks_only_fit", started)

    context: ImageContext | None = None
    checks: dict[int, ClickCheck] = {}
    window_set: WindowObservations | None = None
    if want_images:
        assert roles is not None
        started = time.perf_counter()
        images = load_role_images(store, roles, crops=crops)
        reference_rep = compute_representation(images["reference"])
        _elapsed(timings, "reference_images", started)
        started = time.perf_counter()
        warped = warp_role_images(
            images["moving"], reference=images["reference"], pull_um=pass1.mean
        )
        moving_rep = compute_representation(warped)
        _elapsed(timings, "moving_warp", started)
        context = ImageContext(
            roles=roles,
            reference=reference_rep,
            moving_warped=moving_rep,
            reference_images=images["reference"],
            pull_um=pass1.mean,
        )
        if image_check:
            started = time.perf_counter()
            predicted = moving_um[fit_index] - pass1.loo().residual_um[:n_clicks]
            results = check_clicks(
                reference_rep,
                moving_rep,
                pair_ids=[int(v) for v in ids[fit_index]],
                reference_xy_um=reference_um[fit_index],
                moving_xy_um=moving_um[fit_index],
                predicted_moving_xy_um=predicted,
                pull_um=pass1.mean,
            )
            checks = {check.pair_id: check for check in results}
            _elapsed(timings, "image_check", started)
        if windows:
            started = time.perf_counter()
            window_set = tissue_window_observations(
                reference_rep,
                moving_rep,
                pull_um=pass1.mean,
                params=window_params(settings),
            )
            _elapsed(timings, "tissue_windows", started)

    # Pass 2: ridge shapes, windows and trusted confirmations.
    verdict = tuple(
        checks[int(i)].verdict if int(i) in checks else "not_checked" for i in ids
    )
    strength = tuple(
        checks[int(i)].strength
        if int(i) in checks and checks[int(i)].verdict == "confirmed"
        else None
        for i in ids
    )
    click_ids = ids[fit_index]
    shape = np.broadcast_to(np.eye(2), (n_clicks, 2, 2)).copy()
    exempt_clicks = np.zeros(n_clicks, dtype=bool)
    has_ridge = False
    for row, pair_id in enumerate(click_ids):
        check = checks.get(int(pair_id))
        if check is None:
            continue
        if check.verdict == "ridge" and check.ridge_direction_moving_xy is not None:
            shape[row] = ridge_shape(check.ridge_direction_moving_xy)
            has_ridge = True
        elif (
            settings.field_trust_confirmed_clicks
            and check.verdict == "confirmed"
            and check.strength == "strong"
        ):
            exempt_clicks[row] = True
    n_windows = 0 if window_set is None else len(window_set.reference_xy_um)
    has_second_pass = has_ridge or n_windows > 0 or bool(exempt_clicks.any())
    if has_second_pass:
        classes = (
            "confirmed" if exempt_clicks.sum() >= MIN_CONFIRMED_FOR_CLASS else "click"
        )
        click_observations = FieldObservations(
            reference_xy_um=reference_um[fit_index],
            moving_xy_um=moving_um[fit_index],
            shape_cov=shape,
            fixed_cov_um2=np.zeros((n_clicks, 2, 2)),
            obs_class=tuple(classes if exempt else "click" for exempt in exempt_clicks),
            obs_id=click_ids,
            em_exempt=exempt_clicks,
        )
        observations = click_observations
        if window_set is not None and n_windows > 0:
            observations = observations.concat(
                FieldObservations.windows(
                    window_set.reference_xy_um,
                    window_set.moving_xy_um,
                    window_set.peak_cov_um2,
                    # Window ids continue after the largest pair id so that the
                    # C1 observation ids stay unique across classes.
                    first_id=int(ids.max()) + 1,
                )
            )
        started = time.perf_counter()
        final = fit_landmark_field(observations, options)
        _elapsed(timings, "final_fit", started)
    else:
        final = pass1

    magnitude1, _vector1, p_value1, weight1 = _per_pair(
        pass1,
        reference_um=reference_um,
        moving_um=moving_um,
        observation_index=observation_index,
    )
    if has_second_pass:
        magnitude, vector, p_value, weight = _per_pair(
            final,
            reference_um=reference_um,
            moving_um=moving_um,
            observation_index=observation_index,
        )
    else:
        magnitude, vector, p_value, weight = magnitude1, _vector1, p_value1, weight1
    enabled = np.asarray(similarity.enabled, dtype=bool)
    pair_class = classify_pairs(
        enabled=enabled,
        observed=observed,
        p_value=p_value,
        weight=weight,
        clicks_only_p_value=p_value1,
        clicks_only_weight=weight1,
        verdict=verdict,
        outlier_p=settings.field_outlier_p,
    )

    warnings = [
        text
        for text in similarity.warnings
        if not _SIMILARITY_OUTLIER_WARNING.search(text)
    ]
    rigid_outliers = np.flatnonzero(similarity.outlier)
    if len(rigid_outliers):
        listed = ", ".join(str(pairs[i].id) for i in rigid_outliers)
        warnings.append(
            f"rigid (similarity) fit: {len(rigid_outliers)} pair(s) exceed the "
            f"similarity outlier rule (informative; the robust field decides): {listed}"
        )
    exempt_pairs = np.zeros(len(pairs), dtype=bool)
    exempt_pairs[fit_index] = exempt_clicks
    report = FieldFitReport(
        similarity=similarity,
        field=final,
        clicks_only=pass1,
        has_second_pass=has_second_pass,
        pair_ids=ids,
        observed=observed,
        loo_reference_um=magnitude,
        loo_vector_moving_um=vector,
        p_value=p_value,
        weight=weight,
        clicks_only_loo_reference_um=magnitude1,
        clicks_only_p_value=p_value1,
        clicks_only_weight=weight1,
        verdict=verdict,
        pair_class=pair_class,
        checks=checks,
        windows=window_set,
        em_exempt=exempt_pairs,
        confirm_strength=strength,
        warnings=tuple(warnings),
        timings_s=timings,
    )
    return report, context


def classify_pairs(
    *,
    enabled: np.ndarray,
    observed: np.ndarray,
    p_value: np.ndarray,
    weight: np.ndarray,
    clicks_only_p_value: np.ndarray,
    clicks_only_weight: np.ndarray,
    verdict: Sequence[str],
    outlier_p: float,
) -> tuple[str, ...]:
    """Classify every pair (contract C7).

    p* and w* are the lower of the final and clicks-only p-value and weight (NaN
    weights ignored), so tissue windows cannot hide a misclick the clicks alone
    expose. Gross means p* below ``GROSS_P`` or w* below ``GROSS_WEIGHT``; flagged
    means p* below ``outlier_p``. The first matching row wins:

    1. disabled; 2. enabled but not an observation (``excluded``);
    3. gross and confirmed (``check``); 4. gross and not confirmed
    (``likely_misclick``); 5. flagged and contradicted (``likely_misclick``);
    6. flagged and confirmed (``kept_confirmed``); 7. flagged otherwise
    (``check``); 8. not flagged and contradicted (``check``); 9. ``ok``.

    Args:
        enabled: ``(P,)`` pair is enabled.
        observed: ``(P,)`` pair is an observation of the field.
        p_value: Final field p-values.
        weight: Final field weights (NaN when not observed).
        clicks_only_p_value: Clicks-only field p-values (the final ones without a
            second pass).
        clicks_only_weight: Clicks-only field weights.
        verdict: Image verdict per pair.
        outlier_p: The flagging threshold (``field_outlier_p``).

    Returns:
        One class name of ``CHECK_CLASSES`` per pair.
    """
    p_star = np.minimum(
        np.asarray(p_value, float), np.asarray(clicks_only_p_value, float)
    )
    w_star = np.fmin(np.asarray(weight, float), np.asarray(clicks_only_weight, float))
    classes: list[str] = []
    for i, name in enumerate(verdict):
        gross = bool(p_star[i] < GROSS_P or w_star[i] < GROSS_WEIGHT)
        flagged = bool(p_star[i] < outlier_p)
        if not enabled[i]:
            classes.append("disabled")
        elif not observed[i]:
            classes.append("excluded")
        elif gross:
            classes.append("check" if name == "confirmed" else "likely_misclick")
        elif flagged:
            if name == "contradicted":
                classes.append("likely_misclick")
            elif name == "confirmed":
                classes.append("kept_confirmed")
            else:
                classes.append("check")
        elif name == "contradicted":
            classes.append("check")
        else:
            classes.append("ok")
    return tuple(classes)


# ---------------------------------------------------------------------------
# The check payload (contract C3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class SuggestionRecord:
    """A suggested next click with the region of its candidate position."""

    suggestion: Suggestion
    region: str


def _stats(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"median": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "median": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90)),
        "max": float(np.max(finite)),
    }


def _ellipse(report_field: LandmarkField, suggestion: Suggestion) -> dict[str, float]:
    """Posterior ellipse of a suggestion in the reference frame (y down)."""
    jacobian = report_field.jacobian(suggestion.reference_xy_um[None, :])[0]
    inverse = np.linalg.inv(jacobian)
    covariance = inverse @ suggestion.covariance_um2 @ inverse.T
    values, vectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
    major = vectors[:, 1]
    return {
        "major": float(np.sqrt(max(values[1], 0.0))),
        "minor": float(np.sqrt(max(values[0], 0.0))),
        "angle_deg": float(np.degrees(np.arctan2(major[1], major[0]))),
    }


def _hyperparameters(field: LandmarkField) -> dict[str, Any]:
    hyper = field.hyperparameters
    return {
        "long_length_um": float(hyper.long_length_um),
        "long_signal_um": float(hyper.long_signal_um),
        "short_length_um": (
            None if hyper.short_length_um is None else float(hyper.short_length_um)
        ),
        "short_signal_um": (
            None if hyper.short_signal_um is None else float(hyper.short_signal_um)
        ),
        "noise_um": {name: float(v) for name, v in sorted(hyper.noise_um.items())},
    }


def field_summary(
    report: FieldFitReport,
    *,
    outlier_p: float,
    sd_summary: Mapping[str, float] | None,
) -> dict[str, Any]:
    """Return the ``field`` block shared by the check payload and the result attrs.

    Args:
        report: The field fit.
        outlier_p: The flagging threshold (``field_outlier_p``).
        sd_summary: Median, p90 and max of the field SD over the candidate centres,
            or None when there are none.

    Returns:
        ``n_clicks``, ``n_windows``, ``hyperparameters``, the leave-one-out
        statistics (final, clicks-only or None, rigid), ``sd_um``, ``outlier_p`` and
        the class and verdict counts (every class and verdict, zeros included).
    """
    similarity = report.similarity
    fit_index = np.asarray(similarity.fit_index, dtype=int)
    return {
        "n_clicks": int(len(fit_index)),
        "n_windows": int(report.n_windows),
        "hyperparameters": _hyperparameters(report.field),
        "loo_um": _stats(report.loo_reference_um[fit_index]),
        "loo_um_clicks_only": (
            _stats(report.clicks_only_loo_reference_um[fit_index])
            if report.has_second_pass
            else None
        ),
        "rigid_loo_um": _stats(similarity.loo_um[fit_index]),
        "sd_um": None if sd_summary is None else dict(sd_summary),
        "outlier_p": float(outlier_p),
        "class_counts": {name: report.pair_class.count(name) for name in CHECK_CLASSES},
        "verdict_counts": {name: report.verdict.count(name) for name in CHECK_VERDICTS},
    }


def check_payload(
    report: FieldFitReport,
    *,
    store: StoreAttrs,
    labelling: LabellingAttrs,
    labelling_sha: str,
    labelling_source: Literal["store", "file"],
    settings: ManualRegistrationSettings,
    preset: str,
    roles: ChannelRoles | None,
    suggestions: Sequence[SuggestionRecord],
    sd_summary: Mapping[str, float] | None,
    outputs: Mapping[str, str],
) -> dict[str, Any]:
    """Build the check report of contract C3 (without the ``details`` block).

    Args:
        report: The field fit.
        store: Attrs of the prepared store.
        labelling: The checked labelling.
        labelling_sha: ``labelling_content_sha256`` of its raw payload.
        labelling_source: ``store`` or ``file``.
        settings: The resolved settings the check ran with.
        preset: Preset name.
        roles: Channel roles, or None.
        suggestions: Suggested next clicks.
        sd_summary: Median, p90 and max of the field SD over the candidate
            centres, or None when there are none.
        outputs: File names of the JSON and the PNG (no directories).

    Returns:
        A validated, canonical-JSON-clean payload.

    Raises:
        ValueError: If the payload is invalid or holds a non-finite number.
    """
    pixel_ref = store.images["reference"].pixel_size_um
    pixel_mov = store.images["moving"].pixel_size_um
    similarity = report.similarity
    pairs = similarity.pairs
    pair_records: dict[str, Any] = {}
    for i, pair in enumerate(pairs):
        rigid = float(similarity.loo_um[i])
        check = report.checks.get(pair.id)
        snap = None if check is None else check.snap_moving_xy_um
        offset = None if check is None else check.image_offset_um
        clicks_only: dict[str, Any] | None = None
        if report.has_second_pass:
            weight1 = float(report.clicks_only_weight[i])
            clicks_only = {
                "weight": weight1 if np.isfinite(weight1) else None,
                "loo_um": float(report.clicks_only_loo_reference_um[i]),
                "p_value": float(report.clicks_only_p_value[i]),
            }
        weight = float(report.weight[i])
        pair_records[str(pair.id)] = {
            "enabled": bool(report.similarity.enabled[i]),
            "used": bool(report.observed[i]),
            "weight": weight if np.isfinite(weight) else None,
            "loo_um": float(report.loo_reference_um[i]),
            "loo_vector_moving_um": [float(v) for v in report.loo_vector_moving_um[i]],
            "p_value": float(report.p_value[i]),
            "clicks_only": clicks_only,
            "rigid_loo_um": rigid if np.isfinite(rigid) else None,
            "rigid_outlier": bool(similarity.outlier[i]),
            "verdict": report.verdict[i],
            "image_offset_um": None if offset is None else float(offset),
            "snap_moving_xy_px": (
                None if snap is None else [float(v) / pixel_mov for v in snap]
            ),
            "class": report.pair_class[i],
            "em_exempt": bool(report.em_exempt[i]),
            "confirm_strength": report.confirm_strength[i],
        }
    image_evidence = bool(report.checks) or report.windows is not None
    payload: dict[str, Any] = {
        "schema": CHECK_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "written_at": utc_now(),
        "written_by": {"software": "merxen", "version": merxen.__version__},
        "dataset_name": store.dataset_name,
        "labelling_source": labelling_source,
        "labelling_content_sha256": labelling_sha,
        "labelling_revision": labelling.revision,
        "labelling_status": labelling.status,
        "store_content_digest": store.content_digest,
        "preset": preset,
        "preset_version": PRESET_VERSIONS[preset],
        "settings_digest": canonical_sha256(settings.model_dump(mode="json")),
        "channel_roles": None if roles is None else roles.to_dict(),
        "algorithm": {
            "landmark_field": FIELD_ALGORITHM,
            "image_evidence": IMAGE_EVIDENCE_ALGORITHM if image_evidence else None,
        },
        "field": field_summary(
            report, outlier_p=settings.field_outlier_p, sd_summary=sd_summary
        ),
        "pairs": pair_records,
        "suggestions": [
            {
                "rank": int(record.suggestion.rank),
                "reference_xy_px": [
                    float(v) / pixel_ref for v in record.suggestion.reference_xy_um
                ],
                "predicted_moving_xy_px": [
                    float(v) / pixel_mov
                    for v in record.suggestion.predicted_moving_xy_um
                ],
                "sd_um": float(record.suggestion.sd_um),
                "ellipse_um": _ellipse(report.field, record.suggestion),
                "weight": float(record.suggestion.weight),
                "region": record.region,
            }
            for record in suggestions
        ],
        "outputs": dict(outputs),
    }
    canonical_json(payload)  # rejects NaN and infinities
    parse_check_attrs(payload)
    return payload


def details_payload(report: FieldFitReport) -> dict[str, Any]:
    """Return the ``details`` block of the check JSON (not part of the attribute).

    Holds the per-click image estimates, the window summary and candidate counts,
    the C1 model of the final field, the pass-1 hyperparameters, the timings and
    the warnings.
    """
    estimates: dict[str, Any] = {}
    for pair_id, check in sorted(report.checks.items()):
        estimates[str(pair_id)] = {
            "reason": check.reason,
            "elongation": None if check.elongation is None else float(check.elongation),
            "prediction_offset_um": (
                None
                if check.prediction_offset_um is None
                else float(check.prediction_offset_um)
            ),
            "image_moving_xy_um": (
                None
                if check.image_moving_xy_um is None
                else [float(v) for v in check.image_moving_xy_um]
            ),
            "snap_moving_xy_um": (
                None
                if check.snap_moving_xy_um is None
                else [float(v) for v in check.snap_moving_xy_um]
            ),
            "estimates": [
                {
                    "representation": estimate.representation,
                    "window_um": float(estimate.window_um),
                    "peak": float(estimate.peak),
                    "distinct": float(estimate.distinct),
                    "on_border": bool(estimate.on_border),
                    "confident": bool(estimate.confident),
                    "elongation": float(estimate.elongation),
                    "image_moving_xy_um": [
                        float(v) for v in estimate.image_moving_xy_um
                    ],
                }
                for estimate in check.estimates
            ],
        }
    windows: dict[str, Any] | None = None
    if report.windows is not None:
        candidates = report.windows.candidates
        windows = {
            "summary": report.windows.summary,
            "n_accepted_windows": report.n_windows,
            "n_candidates": int(len(candidates)),
            "n_confident": int(candidates["confident"].sum()),
            "n_accepted": int(candidates["accepted"].sum()),
        }
    return {
        "click_estimates": estimates,
        "windows": windows,
        "field_model": report.field.to_dict(),
        "clicks_only_hyperparameters": _hyperparameters(report.clicks_only),
        "timings_s": dict(report.timings_s),
        "warnings": list(report.warnings),
    }


# ---------------------------------------------------------------------------
# The field as a pull flow (contracts C2, C5)
# ---------------------------------------------------------------------------

_AFFINE_LATTICE = 64
_MIN_AFFINE_POINTS = 16


@dataclass(frozen=True)
class FieldFlow:
    """The field as a pull flow on a coarse working grid of the reference crop.

    Attributes:
        flow_yx: float32 ``(2, h, w)`` pull displacement in working px (row, column).
        working_to_reference: ``(3, 3)`` working px to reference crop level-0 px.
        factor: Reference crop level-0 px per working px.
        max_displacement_um: Largest node displacement, in reference um.
    """

    flow_yx: np.ndarray
    working_to_reference: np.ndarray
    factor: int
    max_displacement_um: float


def _scale_matrix(pixel_size_um: float) -> np.ndarray:
    return np.diag([pixel_size_um, pixel_size_um, 1.0])


def best_affine(
    field: LandmarkField,
    *,
    crop_reference: CropRect,
    reference_pixel_size_um: float,
    moving_pixel_size_um: float,
    mask_xy_um: Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """Fit the least-squares affine of the field mean over the reference crop.

    The mean is evaluated on a 64 x 64 lattice of the crop (pixel centres from the
    first to the last pixel); ``mask_xy_um`` restricts the lattice to tissue. With
    fewer than 16 lattice points inside the mask the whole lattice is used.

    Args:
        field: The field (reference full um to moving full um).
        crop_reference: The effective reference crop, full-frame level-0 px.
        reference_pixel_size_um: Reference level-0 pixel size.
        moving_pixel_size_um: Moving level-0 pixel size.
        mask_xy_um: Maps ``(M, 2)`` reference full um to a boolean ``(M,)`` (True
            inside the tissue), or None for no mask.

    Returns:
        The ``(3, 3)`` moving full px to reference full px matrix
        ``S_r^-1 T^-1 S_m``, with ``T`` the fitted reference-to-moving um map.
    """
    xs = np.linspace(crop_reference.x0, crop_reference.x1 - 1, _AFFINE_LATTICE)
    ys = np.linspace(crop_reference.y0, crop_reference.y1 - 1, _AFFINE_LATTICE)
    grid_x, grid_y = np.meshgrid(xs, ys)
    reference_um = (
        np.column_stack([grid_x.ravel(), grid_y.ravel()]) * reference_pixel_size_um
    )
    if mask_xy_um is not None:
        inside = np.asarray(mask_xy_um(reference_um), dtype=bool)
        if int(inside.sum()) >= _MIN_AFFINE_POINTS:
            reference_um = reference_um[inside]
        else:
            logger.warning(
                "best affine: only %d lattice points in the tissue mask; using the "
                "whole crop",
                int(inside.sum()),
            )
    moving_um = np.asarray(field.mean(reference_um), dtype=float)
    # Centred before the solve: the positions are tens of millimetres.
    centre_reference = reference_um.mean(axis=0)
    centre_moving = moving_um.mean(axis=0)
    design = np.column_stack(
        [reference_um - centre_reference, np.ones(len(reference_um))]
    )
    coefficients, *_ = np.linalg.lstsq(design, moving_um - centre_moving, rcond=None)
    linear = coefficients[:2].T
    offset = centre_moving - linear @ centre_reference
    reference_to_moving = np.eye(3)
    reference_to_moving[:2, :2] = linear
    reference_to_moving[:2, 2] = offset
    return np.asarray(
        np.linalg.inv(_scale_matrix(reference_pixel_size_um))
        @ np.linalg.inv(reference_to_moving)
        @ _scale_matrix(moving_pixel_size_um)
    )


def _grid_to_reference(factor: int) -> np.ndarray:
    """Working px to reference crop level-0 px: node 0 sits one node before the crop."""
    shift = (factor - 1) / 2.0 - factor
    return np.array(
        [[factor, 0.0, shift], [0.0, factor, shift], [0.0, 0.0, 1.0]], dtype=float
    )


def field_flow(
    field: LandmarkField,
    *,
    crop_reference: CropRect,
    crop_moving: CropRect,
    reference_pixel_size_um: float,
    moving_pixel_size_um: float,
    base_matrix_crop: np.ndarray,
    grid_um: float,
) -> FieldFlow:
    """Sample the field as a pull flow for ``pull_points_through_registration``.

    The grid has ``f = max(1, round(grid_um / p_r))`` reference px per node and
    ``ceil(H / f) + 2`` by ``ceil(W / f) + 2`` nodes, node 0 one node before the
    crop, so every crop pixel lies strictly inside the grid (the pull samples the
    flow with zero outside it). At node ``w`` the flow holds
    ``d = W^-1 M m - w`` with ``m`` the field's moving crop px of the node, so the
    pull of the node returns ``m`` exactly.

    Args:
        field: The field (reference full um to moving full um).
        crop_reference: Effective reference crop (full-frame level-0 px).
        crop_moving: Effective moving crop (full-frame level-0 px).
        reference_pixel_size_um: Reference level-0 pixel size.
        moving_pixel_size_um: Moving level-0 pixel size.
        base_matrix_crop: ``(3, 3)`` moving crop px to reference crop px base affine.
        grid_um: Wanted node spacing in um.

    Returns:
        The flow, its grid matrix, the factor and the largest displacement.
    """
    factor = max(1, round(grid_um / reference_pixel_size_um))
    height, width = crop_reference.height(), crop_reference.width()
    shape = (math.ceil(height / factor) + 2, math.ceil(width / factor) + 2)
    working_to_reference = _grid_to_reference(factor)
    rows, columns = np.indices(shape, dtype=float)
    nodes = np.column_stack([columns.ravel(), rows.ravel()])
    reference_crop = (
        nodes @ working_to_reference[:2, :2].T + working_to_reference[:2, 2]
    )
    origin_reference = np.array([crop_reference.x0, crop_reference.y0], dtype=float)
    origin_moving = np.array([crop_moving.x0, crop_moving.y0], dtype=float)
    reference_um = _reference_um(
        reference_crop, origin_reference, reference_pixel_size_um
    )
    moving_crop = (
        np.asarray(field.mean(reference_um), dtype=float) / moving_pixel_size_um
        - origin_moving
    )
    base = np.asarray(base_matrix_crop, dtype=float)
    target = np.linalg.inv(working_to_reference) @ base
    q = moving_crop @ target[:2, :2].T + target[:2, 2]
    displacement = q - nodes
    flow = np.stack(
        [displacement[:, 1].reshape(shape), displacement[:, 0].reshape(shape)]
    ).astype(np.float32)
    max_um = float(
        np.max(np.linalg.norm(displacement, axis=1)) * factor * reference_pixel_size_um
    )
    return FieldFlow(
        flow_yx=flow,
        working_to_reference=working_to_reference,
        factor=int(factor),
        max_displacement_um=max_um,
    )


def _reference_um(
    crop_xy: np.ndarray, origin_xy: np.ndarray, pixel_size_um: float
) -> np.ndarray:
    """Reference full-frame um of crop-frame level-0 px: ``p (r + o)``."""
    return np.asarray(pixel_size_um * (np.asarray(crop_xy, dtype=float) + origin_xy))


def field_sd_grid(
    field: LandmarkField,
    *,
    crop_reference: CropRect,
    reference_pixel_size_um: float,
    grid_um: float,
) -> tuple[np.ndarray, np.ndarray]:
    """The posterior SD per axis on a coarse grid of the reference crop.

    The grid follows the flow grid convention (node 0 one node before the crop,
    ``ceil(H / g) + 2`` by ``ceil(W / g) + 2`` nodes with ``g = max(1,
    round(grid_um / p_r))``).

    Args:
        field: The field.
        crop_reference: Effective reference crop.
        reference_pixel_size_um: Reference level-0 pixel size.
        grid_um: Wanted node spacing in um.

    Returns:
        ``(sd_um, working_to_reference)``: float32 ``(h, w)`` SD per axis and the
        ``(3, 3)`` grid matrix to reference crop level-0 px.
    """
    factor = max(1, round(grid_um / reference_pixel_size_um))
    shape = (
        math.ceil(crop_reference.height() / factor) + 2,
        math.ceil(crop_reference.width() / factor) + 2,
    )
    matrix = _grid_to_reference(factor)
    rows, columns = np.indices(shape, dtype=float)
    nodes = np.column_stack([columns.ravel(), rows.ravel()])
    reference_crop = nodes @ matrix[:2, :2].T + matrix[:2, 2]
    origin = np.array([crop_reference.x0, crop_reference.y0], dtype=float)
    positions = _reference_um(reference_crop, origin, reference_pixel_size_um)
    sd = np.asarray(field.sd_um(positions), dtype=np.float32).reshape(shape)
    return sd, matrix


def tissue_mask_function(
    representation: Representation,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a tissue lookup of a representation, for ``best_affine``.

    Args:
        representation: The representation of the reference image.

    Returns:
        A function from ``(M, 2)`` reference full um to a boolean ``(M,)``: the
        nearest working pixel is tissue (False outside the image).
    """
    tissue = representation.tissue
    height, width = tissue.shape

    def inside(xy_um: np.ndarray) -> np.ndarray:
        nearest = np.rint(representation.um_to_px(np.asarray(xy_um, dtype=float)))
        column = nearest[:, 0].astype(int)
        row = nearest[:, 1].astype(int)
        valid = (column >= 0) & (column < width) & (row >= 0) & (row < height)
        result = np.zeros(len(nearest), dtype=bool)
        result[valid] = tissue[row[valid], column[valid]]
        return result

    return inside


def candidate_sd_summary(
    field: LandmarkField,
    *,
    store: StoreAttrs,
    roles: ChannelRoles | None,
    crops: Mapping[str, CropRect],
) -> dict[str, float] | None:
    """Median, p90 and max of the field SD over the suggestion candidate centres.

    The centres are the shared tissue of the coarse images on a 100 um lattice
    (``target_weight_map``), as in the landmark check.

    Args:
        field: The field.
        store: Attrs of the prepared store.
        roles: Channel roles, or None to use the registration channels.
        crops: Effective crops by ``reference`` and ``moving``.

    Returns:
        The statistics, or None when there are no candidates.
    """
    coarse = coarse_role_images(
        store, roles, crops=crops, pixel_size_um=COARSE_PIXEL_SIZE_UM
    )
    targets = target_weight_map(
        coarse["reference"], coarse["moving"], pull_um=field.mean
    )
    centres = np.asarray(targets.centres_um, dtype=float)
    if not len(centres):
        return None
    sd = field.sd_um(centres)
    return {
        "median": float(np.median(sd)),
        "p90": float(np.percentile(sd, 90)),
        "max": float(np.max(sd)),
    }

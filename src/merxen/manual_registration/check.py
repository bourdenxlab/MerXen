"""Landmark check of a cortex pair: the verdicts and suggestions the labeller needs.

``run_check`` fits the robust landmark field to the saved landmarks (the store's own
labelling or an uploaded labelling file), checks every click against the images,
classifies every pair (contract C7) and suggests the next clicks. It always writes
the check JSON (contract C3 plus a ``details`` block) and a figure; on the store's
own labelling it also writes the C3 payload as the prepared store's root attribute
``merxen_manual_registration_check``, the only writer of that attribute. The viewer
shows the check only while the labelling and the store are the ones it was made on.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from merxen.io.spatialdata_io import spatialdata_write_lock
from merxen.manual_registration.config import ManualRegistrationCheckConfig
from merxen.manual_registration.field_fit import (
    FieldFitReport,
    SuggestionRecord,
    build_landmark_field,
    check_payload,
    details_payload,
    resolve_store_roles,
    validate_field_settings,
)
from merxen.manual_registration.field_report import write_field_figure
from merxen.manual_registration.geometry import snap_crop
from merxen.manual_registration.image_evidence import (
    ChannelRoles,
    RoleImages,
    coarse_role_images,
    target_weight_map,
)
from merxen.manual_registration.landmark_field import suggest_landmarks
from merxen.manual_registration.presets import resolve_settings
from merxen.manual_registration.register import (
    _plain_json,
    _read_prepared_store,
    fit_landmarks,
)
from merxen.manual_registration.schema import (
    CHECK_ATTR,
    LABELLING_ATTR,
    LABELLING_SCHEMA,
    STORE_ATTR,
    CropRect,
    LabellingAttrs,
    StoreAttrs,
    labelling_content_sha256,
    parse_check_attrs,
    read_root_attrs,
    write_root_attr,
)

logger = logging.getLogger(__name__)

COARSE_PIXEL_SIZE_UM = 22.0


class StaleLabellingError(ValueError):
    """The labelling or store changed during the check; the attribute was not written.

    Attributes:
        run: The finished check (its JSON and figure are written), when known.
    """

    run: CheckRun | None = None


@dataclass
class CheckRun:
    """Everything a finished check produced.

    Attributes:
        paths: ``json`` and ``png`` (and ``task_json`` when a task folder is set).
        payload: The C3 payload (without ``details``).
        report: The field fit.
        suggestions: The suggested next clicks.
        attr_written: Whether the check attribute was written to the store.
        warnings: Warnings of the store validation and the fit.
    """

    paths: dict[str, Path]
    payload: dict[str, Any]
    report: FieldFitReport
    suggestions: list[SuggestionRecord]
    attr_written: bool
    warnings: list[str] = field(default_factory=list)


def read_labelling_file(path: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Read a labelling from an uploaded file.

    Accepts a prepared store's root ``zarr.json`` (``{"attributes": {...}}``, as the
    viewer saves it), a JSON object holding the attributes at top level, or a bare
    labelling payload (``schema`` equal to the labelling schema).

    Args:
        path: The file.

    Returns:
        ``(labelling payload, store attr or None)``; the raw payloads, as read.

    Raises:
        ValueError: Naming the file, when it is not JSON or holds no labelling.
    """
    path = Path(path)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"{path}: cannot read the labelling file: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{path}: the labelling file is not valid JSON: {exc}"
        ) from exc
    if isinstance(document, dict):
        attributes = document.get("attributes")
        for holder in (attributes, document):
            if isinstance(holder, dict) and isinstance(
                holder.get(LABELLING_ATTR), dict
            ):
                store_attr = holder.get(STORE_ATTR)
                return dict(holder[LABELLING_ATTR]), (
                    dict(store_attr) if isinstance(store_attr, dict) else None
                )
        if document.get("schema") == LABELLING_SCHEMA:
            return dict(document), None
    raise ValueError(
        f"{path}: no labelling found (expected a prepared store's zarr.json, a JSON "
        f"object with {LABELLING_ATTR!r}, or a labelling payload)"
    )


def write_check_attr(store: Path, payload: Mapping[str, Any]) -> None:
    """Write the check payload as the store's check attribute, if still current.

    Under the store's write lock, re-reads the root attributes and recomputes the
    labelling content digest; the store content digest must also be unchanged.

    Args:
        store: The prepared store.
        payload: The C3 payload (without ``details``).

    Raises:
        StaleLabellingError: If the labelling or the store changed since the check
            read them. Nothing is written.
        ValueError: If the payload is not a valid check.
    """
    store = Path(store)
    with spatialdata_write_lock(store):
        attrs = read_root_attrs(store)
        labelling = attrs.get(LABELLING_ATTR)
        store_attr = attrs.get(STORE_ATTR) or {}
        if not isinstance(labelling, dict):
            raise StaleLabellingError(
                f"the labelling of {store} is gone; the check was not written"
            )
        if labelling_content_sha256(labelling) != payload["labelling_content_sha256"]:
            raise StaleLabellingError(
                f"the labelling of {store} changed while the check ran; the check "
                "was not written"
            )
        if store_attr.get("content_digest") != payload["store_content_digest"]:
            raise StaleLabellingError(
                f"{store} was prepared again while the check ran; the check was not "
                "written"
            )
        parse_check_attrs(payload)
        write_root_attr(store, CHECK_ATTR, payload)


def _snapped_crops(
    labelling: LabellingAttrs,
    store: StoreAttrs,
    *,
    snap_px: int,
    min_size_px: int,
    pfx: str,
) -> dict[str, CropRect]:
    crops: dict[str, CropRect] = {}
    for role in ("reference", "moving"):
        try:
            crops[role] = snap_crop(
                getattr(labelling.crops, role),
                store.images[role].shape_yx,
                multiple=snap_px,
                min_size_px=min_size_px,
            )
        except ValueError as exc:
            raise ValueError(f"{pfx} {role} crop: {exc}") from exc
    return crops


def _suggest(
    report: FieldFitReport,
    store: StoreAttrs,
    roles: ChannelRoles | None,
    crops: Mapping[str, CropRect],
    n_suggestions: int,
) -> tuple[list[SuggestionRecord], dict[str, float] | None, RoleImages]:
    coarse = coarse_role_images(
        store, roles, crops=crops, pixel_size_um=COARSE_PIXEL_SIZE_UM
    )
    targets = target_weight_map(
        coarse["reference"], coarse["moving"], pull_um=report.field.mean
    )
    centres = np.asarray(targets.centres_um, dtype=float)
    if not len(centres):
        return [], None, coarse["reference"]
    sd = report.field.sd_um(centres)
    sd_summary = {
        "median": float(np.median(sd)),
        "p90": float(np.percentile(sd, 90)),
        "max": float(np.max(sd)),
    }
    records: list[SuggestionRecord] = []
    if n_suggestions > 0:
        picked = suggest_landmarks(
            report.field, centres, np.asarray(targets.weights), n=n_suggestions
        )
        for suggestion in picked:
            row = int(
                np.argmin(np.linalg.norm(centres - suggestion.reference_xy_um, axis=1))
            )
            records.append(SuggestionRecord(suggestion, targets.region[row]))
    return records, sd_summary, coarse["reference"]


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    """Write JSON atomically (temporary sibling, then ``os.replace``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial-{uuid.uuid4().hex}")
    try:
        partial.write_text(
            json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        os.replace(partial, path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def execute_check(
    config: ManualRegistrationCheckConfig,
    *,
    labelling_payload: Mapping[str, Any] | None = None,
) -> CheckRun:
    """Run the landmark check and return everything it produced.

    See :func:`run_check` for the steps and the errors.
    """
    started = time.perf_counter()
    run_id = f"{config.dataset_name}_{config.preset}"
    pfx = f"[{run_id}]"
    store_path = Path(config.prepared_store_path).resolve()

    source: Literal["store", "file"] = "store"
    file_store_attr: dict[str, Any] | None = None
    payload_in: Mapping[str, Any] | None = None
    if labelling_payload is not None:
        source, payload_in = "file", labelling_payload
    elif config.labelling_path is not None:
        source = "file"
        payload_in, file_store_attr = read_labelling_file(config.labelling_path)
    write_attr = config.write_attr and source == "store"

    def read(check_fingerprints: bool) -> Any:
        return _read_prepared_store(
            store_path,
            run_id=run_id,
            dataset_name=config.dataset_name,
            check_fingerprints=check_fingerprints,
            require_finished=False,
            labelling_payload=payload_in,
        )

    # The suggestions and the sd_um summary always load the OME-TIFFs, so the
    # fingerprints are always checked.
    store, labelling, raw_labelling, warnings = read(True)
    if (
        file_store_attr is not None
        and file_store_attr.get("content_digest") != store.content_digest
    ):
        raise ValueError(
            f"{pfx} the labelling file was made on a different preparation of this "
            f"store (content digest {file_store_attr.get('content_digest')}, the "
            f"store has {store.content_digest}); download the store again"
        )
    reference = store.images["reference"]
    settings = resolve_settings(
        config.preset,
        reference_pixel_size_um=reference.pixel_size_um,
        reference_channel=reference.registration_channel,
        global_settings=config.global_settings,
        row_settings=config.row_settings,
    )
    image_check = (
        settings.field_image_check if config.image_check is None else config.image_check
    )
    windows = (
        settings.field_window_observations if config.windows is None else config.windows
    )
    # The check always fits the robust field and never registers.
    settings = settings.model_copy(
        update={
            "field_model": "robust_gp",
            "field_image_check": image_check,
            "field_window_observations": windows,
        }
    )
    roles = resolve_store_roles(config.channel_roles, store)
    validate_field_settings(
        settings, preset=config.preset, roles=roles, run_id=run_id, purpose="check"
    )
    labelling_sha = labelling_content_sha256(raw_labelling)
    crops = _snapped_crops(
        labelling,
        store,
        snap_px=settings.crop_snap_px,
        min_size_px=settings.crop_min_size_px,
        pfx=pfx,
    )

    similarity = fit_landmarks(labelling, store, settings, run_id=run_id)
    report, _context = build_landmark_field(
        similarity=similarity,
        store=store,
        settings=settings,
        roles=roles,
        crops=crops,
        image_check=image_check,
        windows=windows,
        run_id=run_id,
    )
    started_suggestions = time.perf_counter()
    suggestions, sd_summary, overview = _suggest(
        report, store, roles, crops, config.n_suggestions
    )
    report.timings_s["suggestions"] = round(
        time.perf_counter() - started_suggestions, 3
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_name = f"{config.dataset_name}_{config.preset}_check.json"
    png_name = f"{config.dataset_name}_{config.preset}_check.png"
    payload = check_payload(
        report,
        store=store,
        labelling=labelling,
        labelling_sha=labelling_sha,
        labelling_source=source,
        settings=settings,
        preset=config.preset,
        roles=roles,
        suggestions=suggestions,
        sd_summary=sd_summary,
        outputs={"json": json_name, "png": png_name},
    )
    report.timings_s["total"] = round(time.perf_counter() - started, 3)
    paths = {"json": output_dir / json_name, "png": output_dir / png_name}
    write_field_figure(
        paths["png"],
        report=report,
        store=store,
        crops=crops,
        suggestions=suggestions,
        overview=overview,
    )
    document = {**payload, "details": _plain_json(details_payload(report))}
    _write_json(paths["json"], document)
    if config.task_output_dir is not None:
        task_dir = Path(config.task_output_dir)
        task_dir.mkdir(parents=True, exist_ok=True)
        paths["task_json"] = task_dir / json_name
        shutil.copy2(paths["json"], paths["task_json"])

    run = CheckRun(
        paths=paths,
        payload=payload,
        report=report,
        suggestions=suggestions,
        attr_written=False,
        warnings=[*warnings, *report.warnings],
    )
    if write_attr:
        try:
            write_check_attr(store_path, payload)
        except StaleLabellingError as exc:
            exc.run = run
            raise
        run.attr_written = True
    logger.info(
        "%s landmark check finished in %.1f s (attribute written: %s)",
        pfx,
        report.timings_s["total"],
        run.attr_written,
    )
    return run


def run_check(
    config: ManualRegistrationCheckConfig,
    *,
    labelling_payload: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Check the landmarks of a pair and write the check JSON, figure and attribute.

    Steps: read the labelling (the store's own, ``config.labelling_path`` or
    ``labelling_payload``) and the prepared store; resolve the settings (the check
    always fits the robust field); resolve the channel roles; snap the crops; fit
    the similarity (at least ``landmark_min_pairs`` enabled pairs); fit the robust
    field with image evidence; make suggestions; write the outputs; and, for the
    store's own labelling with ``write_attr``, the check attribute.

    Args:
        config: The check configuration.
        labelling_payload: A raw labelling payload to check instead of the store's
            own (tests and validation); the attribute is never written for it.

    Returns:
        ``json`` and ``png`` (and ``task_json`` when ``task_output_dir`` is set).

    Raises:
        FileNotFoundError: If the store or an image is missing.
        LandmarkFitError: If the landmarks are unusable.
        StaleLabellingError: If the labelling or store changed while the check ran
            (the JSON and figure are written, the attribute is not).
        ValueError: On any other invalid input, such as a labelling file made on
            another preparation of the store.
    """
    return execute_check(config, labelling_payload=labelling_payload).paths

"""CLI commands of the register step of manual landmark registration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

from merxen.config import load_config_from_json
from merxen.manual_registration.compare import compare_runs
from merxen.manual_registration.config import (
    ManualRegistrationCompareConfig,
    ManualRegistrationRunConfig,
)
from merxen.manual_registration.presets import PRESETS, resolve_settings
from merxen.manual_registration.register import (
    LandmarkFitError,
    LandmarkFitReport,
    _validate_prepared_store,
    fit_landmarks,
    fit_summary,
    run_manual_registration,
)
from merxen.manual_registration.schema import STORE_ATTR, read_root_attrs


@click.command(name="manual-registration-register")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Path to JSON config validated against ManualRegistrationRunConfig.",
)
@click.option(
    "--force-rerun",
    is_flag=True,
    default=False,
    help="Re-register even when an identical result exists.",
)
def manual_registration_register_command(config_path: Path, force_rerun: bool) -> None:
    """Register the moving image onto the reference from the saved landmarks."""
    cfg = load_config_from_json(config_path, ManualRegistrationRunConfig)
    assert isinstance(cfg, ManualRegistrationRunConfig)
    outputs = run_manual_registration(cfg, force_rerun=force_rerun)
    click.echo("Manual registration complete:")
    for key, value in outputs.items():
        click.echo(f"- {key}: {value}")


def _fit_table(report: LandmarkFitReport) -> str:
    lines = [
        f"{'id':>4}  {'enabled':<7}  {'used':<5}  {'outlier':<7}  "
        f"{'outside_crop':<12}  {'residual_um':>11}  {'loo_um':>9}"
    ]
    for i, pair in enumerate(report.pairs):
        lines.append(
            f"{pair.id:>4}  {str(bool(report.enabled[i])):<7}  "
            f"{str(bool(report.used[i])):<5}  {str(bool(report.outlier[i])):<7}  "
            f"{str(bool(report.outside_crop[i])):<12}  "
            f"{report.residuals_um[i]:>11.2f}  {report.loo_um[i]:>9.2f}"
        )
    return "\n".join(lines)


@click.command(name="manual-registration-fit")
@click.option(
    "--store",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    required=True,
    help="Prepared store holding the labelling.",
)
@click.option(
    "--preset",
    type=click.Choice(sorted(PRESETS)),
    required=True,
    help="Preset whose landmark settings apply.",
)
@click.option(
    "--settings",
    "settings_text",
    default="",
    help='Overrides "k=v;preset:k=v".',
)
@click.option(
    "--json",
    "json_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Also write the fit as JSON.",
)
def manual_registration_fit_command(
    store: Path, preset: str, settings_text: str, json_path: Path | None
) -> None:
    """Fit the saved landmarks and print what register would flag (reads only)."""
    try:
        raw = read_root_attrs(store).get(STORE_ATTR) or {}
    except (FileNotFoundError, ValueError) as exc:
        _fail(exc, None, json_path)
        return
    dataset_name = str(raw.get("dataset_name", ""))
    run_id = f"{dataset_name}_{preset}"
    try:
        store_attrs, labelling, validation_warnings = _validate_prepared_store(
            store,
            run_id=run_id,
            dataset_name=dataset_name,
            check_fingerprints=False,
            require_finished=False,
        )
        settings = resolve_settings(
            preset,
            reference_pixel_size_um=store_attrs.images["reference"].pixel_size_um,
            reference_channel=store_attrs.images["reference"].registration_channel,
            row_settings=settings_text,
        )
    except (FileNotFoundError, ValueError) as exc:
        _fail(exc, None, json_path)
        return
    click.echo(f"labelling status: {labelling.status}")
    try:
        report = fit_landmarks(labelling, store_attrs, settings, run_id=run_id)
    except LandmarkFitError as exc:
        _fail(exc, exc.report, json_path)
        return
    click.echo(_fit_table(report))
    summary = fit_summary(
        report, store=store_attrs, settings=settings, labelling=labelling
    )
    click.echo(
        f"fit: n_used {summary['n_used']} of {summary['n_enabled']} enabled, "
        f"rms {summary['rms_um']:.2f} um, median {summary['median_um']:.2f} um, "
        f"scale ratio {summary['scale_ratio']:.5f}, "
        f"rotation {summary['rotation_deg']:.3f} deg, "
        f"reflected {summary['reflected']}, spread {summary['spread_ratio']:.3f}"
    )
    viewer = labelling.outlier_floor_um
    click.echo(
        f"outlier floor: {settings.landmark_outlier_floor_um:g} µm "
        f"(preset {preset}); viewer: "
        f"{'not saved' if viewer is None else format(viewer, 'g') + ' µm'}"
    )
    warnings = [*validation_warnings, *report.warnings]
    for warning in warnings:
        click.echo(f"warning: {warning}")
    if json_path is not None:
        payload: dict[str, Any] = {
            "run_id": run_id,
            "preset": preset,
            "status": labelling.status,
            "fit": summary,
            "warnings": warnings,
            "landmarks": [
                {
                    "id": pair.id,
                    "enabled": bool(report.enabled[i]),
                    "used_in_fit": bool(report.used[i]),
                    "outlier": bool(report.outlier[i]),
                    "outside_crop": bool(report.outside_crop[i]),
                    "residual_um": float(report.residuals_um[i]),
                    "loo_um": float(report.loo_um[i]),
                }
                for i, pair in enumerate(report.pairs)
            ],
        }
        json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _fail(
    exc: Exception, report: LandmarkFitReport | None, json_path: Path | None
) -> None:
    if report is not None:
        click.echo(_fit_table(report))
    click.echo(f"error: {exc}", err=True)
    if json_path is not None:
        json_path.write_text(json.dumps({"error": str(exc)}) + "\n", encoding="utf-8")
    raise SystemExit(1)


@click.command(name="manual-registration-compare")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Path to JSON config validated against ManualRegistrationCompareConfig.",
)
def manual_registration_compare_command(config_path: Path) -> None:
    """Compare several manual-registration runs of one pair."""
    cfg = load_config_from_json(config_path, ManualRegistrationCompareConfig)
    assert isinstance(cfg, ManualRegistrationCompareConfig)
    outputs = compare_runs(cfg)
    click.echo("Manual registration comparison complete:")
    for key, value in outputs.items():
        click.echo(f"- {key}: {value}")

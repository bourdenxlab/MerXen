"""CLI commands of the register step of manual landmark registration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click
import numpy as np

from merxen.config import load_config_from_json
from merxen.manual_registration.check import (
    CheckRun,
    StaleLabellingError,
    execute_check,
)
from merxen.manual_registration.compare import compare_runs
from merxen.manual_registration.config import (
    ManualRegistrationCheckConfig,
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

EXIT_ERROR = 1
EXIT_STALE = 3


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


def _check_table(run: CheckRun) -> str:
    report = run.report
    p_star = np.minimum(report.p_value, report.clicks_only_p_value)
    w_star = np.fmin(report.weight, report.clicks_only_weight)
    lines = [
        f"{'id':>4}  {'enabled':<7}  {'class':<15}  {'verdict':<12}  "
        f"{'LOO_um':>8}  {'p':>9}  {'weight':>6}  {'rigid_LOO_um':>12}"
    ]
    for i, pair in enumerate(report.similarity.pairs):
        rigid = float(report.similarity.loo_um[i])
        weight = "-" if not np.isfinite(w_star[i]) else f"{w_star[i]:.2f}"
        lines.append(
            f"{pair.id:>4}  {str(bool(report.similarity.enabled[i])):<7}  "
            f"{report.pair_class[i]:<15}  {report.verdict[i]:<12}  "
            f"{report.loo_reference_um[i]:>8.1f}  {p_star[i]:>9.2e}  "
            f"{weight:>6}  {rigid if np.isfinite(rigid) else float('nan'):>12.1f}"
        )
    return "\n".join(lines)


def _stat_text(stats: dict[str, float] | None) -> str:
    if stats is None:
        return "-"
    return f"{stats['median']:.1f}/{stats['p90']:.1f}"


def _echo_check(run: CheckRun) -> None:
    payload = run.payload
    field = payload["field"]
    click_class_counts = {k: v for k, v in field["class_counts"].items() if v}
    click.echo(_check_table(run))
    click.echo(
        f"field: {field['n_clicks']} clicks, {field['n_windows']} windows; "
        f"LOO median/p90 um: {_stat_text(field['loo_um'])} (clicks only "
        f"{_stat_text(field['loo_um_clicks_only'])}, rigid "
        f"{_stat_text(field['rigid_loo_um'])}); classes {click_class_counts}; "
        f"sd_um {_stat_text(field['sd_um'])}"
    )
    for warning in run.warnings:
        click.echo(f"warning: {warning}")
    if run.suggestions:
        click.echo("suggested next clicks (reference px, predicted moving px, SD um):")
        for item in payload["suggestions"]:
            ref = item["reference_xy_px"]
            mov = item["predicted_moving_xy_px"]
            click.echo(
                f"  {item['rank']:>2}  ({ref[0]:.0f}, {ref[1]:.0f}) -> "
                f"({mov[0]:.0f}, {mov[1]:.0f})  {item['sd_um']:.1f}"
            )
    for key, value in run.paths.items():
        click.echo(f"- {key}: {value}")
    click.echo(f"check attribute written: {'yes' if run.attr_written else 'no'}")


@click.command(name="manual-registration-check")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
    help="JSON config validated against ManualRegistrationCheckConfig; excludes "
    "every other option.",
)
@click.option(
    "--store",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=None,
    help="Prepared store (required without --config).",
)
@click.option(
    "--labelling",
    "labelling_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
    help="An uploaded zarr.json or labelling JSON to check instead of the store's "
    "own labelling; no attribute is written.",
)
@click.option(
    "--preset",
    type=click.Choice(sorted(PRESETS)),
    default=None,
    help="Preset whose settings apply (default cortex_serial).",
)
@click.option(
    "--settings", "settings_text", default=None, help='Overrides "k=v;preset:k=v".'
)
@click.option(
    "--channel-roles",
    default=None,
    help="Channel roles, e.g. 'at8=AF568;map2=AF647;dapi=DAPI' (quote it: the shell "
    "splits at ';').",
)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Folder for the check JSON and figure (required without --config).",
)
@click.option(
    "--image-check/--no-image-check",
    default=None,
    help="Check every click against the images (default: the settings').",
)
@click.option(
    "--windows/--no-windows",
    default=None,
    help="Add tissue-window observations (default: the settings').",
)
@click.option(
    "--no-write-attr",
    is_flag=True,
    default=False,
    help="Do not write the check attribute into the store.",
)
@click.option(
    "--suggestions",
    "n_suggestions",
    type=click.IntRange(0, 200),
    default=None,
    help="Number of suggested next clicks (default 20).",
)
def manual_registration_check_command(
    config_path: Path | None,
    store: Path | None,
    labelling_path: Path | None,
    preset: str | None,
    settings_text: str | None,
    channel_roles: str | None,
    out_dir: Path | None,
    image_check: bool | None,
    windows: bool | None,
    no_write_attr: bool,
    n_suggestions: int | None,
) -> None:
    """Check the landmarks of a pair: verdicts per pair and the next clicks to place.

    Exit codes: 0 ok, 1 error, 3 stale (the labelling changed during the check; the
    JSON and figure are written, the attribute is not).
    """
    direct = {
        "--store": store,
        "--labelling": labelling_path,
        "--preset": preset,
        "--settings": settings_text,
        "--channel-roles": channel_roles,
        "--out-dir": out_dir,
        "--image-check/--no-image-check": image_check,
        "--windows/--no-windows": windows,
        "--no-write-attr": no_write_attr or None,
        "--suggestions": n_suggestions,
    }
    given = [name for name, value in direct.items() if value is not None]
    if config_path is not None and given:
        raise click.UsageError(f"--config excludes {', '.join(given)}")
    try:
        if config_path is not None:
            loaded = load_config_from_json(config_path, ManualRegistrationCheckConfig)
            assert isinstance(loaded, ManualRegistrationCheckConfig)
            config = loaded
        else:
            if store is None or out_dir is None:
                raise click.UsageError(
                    "--store and --out-dir are required without --config"
                )
            raw = read_root_attrs(store).get(STORE_ATTR) or {}
            config = ManualRegistrationCheckConfig(
                dataset_name=str(raw.get("dataset_name", "")),
                preset=preset or "cortex_serial",  # type: ignore[arg-type]
                prepared_store_path=store,
                labelling_path=labelling_path,
                output_dir=out_dir,
                row_settings=settings_text or "",
                channel_roles=channel_roles or "",
                image_check=image_check,
                windows=windows,
                write_attr=not no_write_attr,
                n_suggestions=20 if n_suggestions is None else n_suggestions,
            )
        run = execute_check(config)
    except StaleLabellingError as exc:
        if exc.run is not None:
            _echo_check(exc.run)
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(EXIT_STALE) from exc
    except LandmarkFitError as exc:
        if exc.report is not None:
            click.echo(_fit_table(exc.report))
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(EXIT_ERROR) from exc
    except (FileNotFoundError, ValueError) as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(EXIT_ERROR) from exc
    _echo_check(run)

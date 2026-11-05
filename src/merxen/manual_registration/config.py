"""Run configuration models of the manual-registration steps.

Every model forbids unknown keys, so a misspelled samplesheet or params key fails
loudly. ``ManualRegistrationSettings`` holds the per-run algorithm settings; its
defaults are the ``same_section`` preset (the presets themselves live with the
register step).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from merxen.config import RegistrationSettings
from merxen.manual_registration.schema import (
    CROP_MIN_SIZE_PX,
    CROP_SNAP_PX,
    DATASET_ID_PATTERN,
    DEFAULT_OUTLIER_FLOOR_UM,
    MAX_OUTLIER_FRACTION,
    MIN_LANDMARK_PAIRS,
    MIN_USED_PAIRS,
    OUTLIER_K,
)

MANUAL_REGISTRATION_ID_PATTERN = DATASET_ID_PATTERN
PresetName = Literal["same_section", "serial_section"]
PRESET_NAMES: tuple[str, ...] = ("same_section", "serial_section")


class ManualRegistrationImageInput(BaseModel):
    """One input image (reference or moving) of a manual-registration pair."""

    model_config = ConfigDict(extra="forbid")

    path: Path
    scene: str | None = None  # CZI only: S index or scene name
    channel_names: list[str] | None = None  # verbatim override of detected names
    registration_channel: str = "DAPI"
    pixel_size_um: float | None = Field(default=None, gt=0.0)

    @field_validator("scene")
    @classmethod
    def _validate_scene(
        cls: type[ManualRegistrationImageInput], value: str | None
    ) -> str | None:
        if value is None:
            return None
        scene = value.strip()
        if not scene:
            raise ValueError("scene must not be blank")
        return scene

    @field_validator("channel_names")
    @classmethod
    def _validate_channel_names(
        cls: type[ManualRegistrationImageInput], value: list[str] | None
    ) -> list[str] | None:
        if value is None:
            return None
        names = [str(name).strip() for name in value]
        if not names or any(not name for name in names):
            raise ValueError("channel_names must be non-empty strings")
        if len(set(names)) != len(names):
            raise ValueError(f"channel_names must be unique, got {names}")
        return names

    @field_validator("registration_channel")
    @classmethod
    def _validate_registration_channel(
        cls: type[ManualRegistrationImageInput], value: str
    ) -> str:
        channel = value.strip()
        if not channel:
            raise ValueError("registration_channel must not be blank")
        return channel


class ManualRegistrationPrepareConfig(BaseModel):
    """Configuration of the prepare step (images to a prepared image-only store)."""

    model_config = ConfigDict(extra="forbid")

    dataset_name: str = Field(pattern=MANUAL_REGISTRATION_ID_PATTERN)
    reference: ManualRegistrationImageInput
    moving: ManualRegistrationImageInput
    prepared_dir: Path  # durable: ${outdir}/<id>/prepared
    output_dir: Path  # task-local, e.g. "prepare_out"
    force: bool = False
    chunk_size: int = Field(default=4096, gt=0)
    tile_size: int = Field(default=1024, gt=0)  # OME-TIFF tile
    band_height: int = Field(default=2048, gt=0)
    min_level_size: int = Field(default=1024, gt=0)
    compression: Literal["zlib", "zstd", "none"] = "zlib"
    compression_level: int = Field(default=6, ge=0, le=22)
    workers: int | None = Field(default=None, ge=1)  # None: resolved by the step
    tmp_dir: Path | None = None  # None: a scratch directory inside prepared_dir

    @field_validator("tile_size")
    @classmethod
    def _validate_tile_size(
        cls: type[ManualRegistrationPrepareConfig], value: int
    ) -> int:
        # TIFF tiles must be a multiple of 16 (tifffile rejects others).
        if value % 16 != 0:
            raise ValueError(f"tile_size must be a multiple of 16, got {value}")
        return value

    @model_validator(mode="after")
    def _validate_band_height(
        self: ManualRegistrationPrepareConfig,
    ) -> ManualRegistrationPrepareConfig:
        if self.band_height % self.tile_size != 0:
            raise ValueError(
                f"band_height ({self.band_height}) must be a multiple of "
                f"tile_size ({self.tile_size})"
            )
        return self

    @property
    def store_path(self) -> Path:
        """Path of the prepared store."""
        return self.prepared_dir / f"{self.dataset_name}_prepared.zarr"


class ManualRegistrationSettings(RegistrationSettings):
    """Settings of one manual-registration run (defaults: the ``same_section`` preset).

    Every field is overridable per samplesheet row and via params. The defaults
    equal the REGISTER_IMAGES defaults except ``nonrigid_enabled = True`` and
    ``refinement_min_correlation_gain = 0.0`` (``REGISTER_IMAGES`` applies every
    refinement inside the limits). The gain is a number in [-2, 2], the smallest
    rise of the registration-channel correlation a refinement must give; the
    override grammar cannot express ``None``, so ``-2`` (below any possible gain)
    turns the check off.
    """

    model_config = ConfigDict(extra="forbid")

    nonrigid_enabled: bool = True  # overrides the base default (False)
    refinement_min_window_quality: float = Field(default=0.3, ge=-1.0, le=1.0)
    refinement_require_moving_tissue: bool = False
    refinement_min_correlation_gain: float = Field(default=0.0, ge=-2.0, le=2.0)
    nonrigid_highpass_sigma_um: float = Field(default=25.0, gt=0.0)
    nonrigid_local_correlation_sigma_um: float = Field(default=15.0, gt=0.0)
    nonrigid_local_residual_tiles: bool = True
    nonrigid_local_residual_tile_um: float = Field(default=220.0, gt=0.0)
    landmark_min_pairs: int = Field(default=MIN_LANDMARK_PAIRS, ge=3)
    landmark_min_used_pairs: int = Field(default=MIN_USED_PAIRS, ge=3)
    landmark_outlier_floor_um: float = Field(default=DEFAULT_OUTLIER_FLOOR_UM, gt=0.0)
    landmark_outlier_k: float = Field(default=OUTLIER_K, gt=0.0)
    landmark_max_outlier_fraction: float = Field(
        default=MAX_OUTLIER_FRACTION, ge=0.0, lt=0.5
    )
    landmark_outlier_policy: Literal["exclude", "keep", "fail"] = "exclude"
    landmarks_outside_crop: Literal["use", "exclude", "fail"] = "use"
    landmark_min_spread_ratio: float = Field(default=0.05, ge=0.0, le=1.0)
    landmark_reflection_check_ratio: float = Field(default=0.5, gt=0.0, le=1.0)
    crop_snap_px: int = Field(default=CROP_SNAP_PX, ge=1)
    crop_min_size_px: int = Field(default=CROP_MIN_SIZE_PX, ge=64)
    tissue_qc_pixel_size_um: float = Field(default=10.0, gt=0.0)
    tissue_qc_sigma_um: float = Field(default=20.0, gt=0.0)
    tissue_qc_highpass_sigma_um: float = Field(default=200.0, gt=0.0)
    nucleus_qc: bool = False
    nucleus_qc_pixel_size_um: float = Field(default=1.7, gt=0.0)
    nucleus_qc_window_um: float = Field(default=200.0, gt=0.0)

    @field_validator("crop_snap_px")
    @classmethod
    def _validate_crop_snap(cls: type[ManualRegistrationSettings], value: int) -> int:
        if value & (value - 1) != 0:
            raise ValueError(f"crop_snap_px must be a power of two, got {value}")
        return value

    @model_validator(mode="after")
    def _validate_pair_counts(
        self: ManualRegistrationSettings,
    ) -> ManualRegistrationSettings:
        if self.landmark_min_used_pairs > self.landmark_min_pairs:
            raise ValueError(
                "landmark_min_used_pairs must not exceed landmark_min_pairs"
            )
        return self


class ManualRegistrationRunConfig(BaseModel):
    """Configuration of the register step for one (pair, preset) run."""

    model_config = ConfigDict(extra="forbid")

    dataset_name: str = Field(pattern=MANUAL_REGISTRATION_ID_PATTERN)
    preset: PresetName
    run_label: str | None = Field(default=None, pattern=MANUAL_REGISTRATION_ID_PATTERN)
    prepared_store_path: Path
    result_dir: Path  # durable: ${outdir}/<id>/registered/<run_name>
    output_dir: Path  # task-local, e.g. "register_out"
    global_settings: str = ""  # params.manual_registration_settings
    row_settings: str = ""  # samplesheet registration_settings
    force: bool = False

    @property
    def run_name(self) -> str:
        """Preset name, plus the label when one is given."""
        if self.run_label is None:
            return self.preset
        return f"{self.preset}_{self.run_label}"

    @property
    def run_id(self) -> str:
        """Dataset name and run name; unique per (pair, preset, label)."""
        return f"{self.dataset_name}_{self.run_name}"

    @property
    def result_store_path(self) -> Path:
        """Path of the result store."""
        return self.result_dir / f"{self.run_id}_registered.zarr"


class ManualRegistrationCompareConfig(BaseModel):
    """Configuration of the comparison of several runs of one pair."""

    model_config = ConfigDict(extra="forbid")

    dataset_name: str = Field(pattern=MANUAL_REGISTRATION_ID_PATTERN)
    summary_paths: list[Path] = Field(min_length=2)
    output_dir: Path  # durable: ${outdir}/<id>/registered
    task_output_dir: Path  # task-local, e.g. "compare_out"

"""Normalized training configuration models for the training entrypoint."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


def _mapping(value: object) -> Mapping[str, Any]:
    """Return a mapping-like value or an empty mapping for missing config blocks."""

    if isinstance(value, Mapping):
        return value
    return {}


@dataclass(frozen=True)
class LoggingConfig:
    """Logging settings accepted by the training entrypoint."""

    level: str = "INFO"

    @classmethod
    def from_mapping(cls, value: object) -> "LoggingConfig":
        config = _mapping(value)
        return cls(level=str(config.get("level", cls.level)))


@dataclass(frozen=True)
class DataloaderConfig:
    """Dataloader settings consumed by the training bootstrap."""

    num_workers: int = 4
    persistent_workers: bool = True
    pin_memory: bool = True
    cache_size: int = 0

    @classmethod
    def from_mapping(cls, value: object) -> "DataloaderConfig":
        config = _mapping(value)
        return cls(
            num_workers=int(config.get("num_workers", cls.num_workers)),
            persistent_workers=bool(config.get("persistent_workers", cls.persistent_workers)),
            pin_memory=bool(config.get("pin_memory", cls.pin_memory)),
            cache_size=int(config.get("cache_size", cls.cache_size)),
        )


@dataclass(frozen=True)
class AugmentationConfig:
    """Augmentation settings supported by the training transform builder."""

    rotate_prob: float = 1.0
    rotate_limit: int = 0
    rotate_border_mode: str = "constant"
    rotate_border_value: int = 0
    rotate_mask_border_value: int = 0
    flip_prob: float = 0.5
    brightness_contrast_prob: float = 0.0
    brightness_limit: float | list[float] = field(default_factory=lambda: [-0.2, 0.2])
    contrast_limit: float | list[float] = field(default_factory=lambda: [-0.2, 0.2])
    brightness_by_max: bool = True
    gamma_prob: float = 0.0
    gamma_limit: list[int] = field(default_factory=lambda: [80, 120])
    blur_prob: float = 0.0
    blur_limit: list[int] = field(default_factory=lambda: [3, 7])
    gaussian_noise_prob: float = 0.0
    gaussian_noise_limit: list[float] = field(default_factory=lambda: [0.005, 0.02])

    @classmethod
    def from_mapping(cls, value: object) -> "AugmentationConfig":
        config = _mapping(value)
        return cls(
            rotate_prob=float(config.get("rotate_prob", cls.rotate_prob)),
            rotate_limit=int(config.get("rotate_limit", cls.rotate_limit)),
            rotate_border_mode=str(config.get("rotate_border_mode", cls.rotate_border_mode)),
            rotate_border_value=int(config.get("rotate_border_value", cls.rotate_border_value)),
            rotate_mask_border_value=int(
                config.get("rotate_mask_border_value", cls.rotate_mask_border_value)
            ),
            flip_prob=float(config.get("flip_prob", cls.flip_prob)),
            brightness_contrast_prob=float(
                config.get("brightness_contrast_prob", cls.brightness_contrast_prob)
            ),
            brightness_limit=config.get("brightness_limit", [-0.2, 0.2]),
            contrast_limit=config.get("contrast_limit", [-0.2, 0.2]),
            brightness_by_max=bool(config.get("brightness_by_max", cls.brightness_by_max)),
            gamma_prob=float(config.get("gamma_prob", cls.gamma_prob)),
            gamma_limit=list(config.get("gamma_limit", [80, 120])),
            blur_prob=float(config.get("blur_prob", cls.blur_prob)),
            blur_limit=list(config.get("blur_limit", [3, 7])),
            gaussian_noise_prob=float(
                config.get("gaussian_noise_prob", cls.gaussian_noise_prob)
            ),
            gaussian_noise_limit=list(config.get("gaussian_noise_limit", [0.005, 0.02])),
        )


@dataclass(frozen=True)
class LossConfig:
    """Loss configuration block used by the training entrypoint."""

    type: str = "crossentropy_dice"
    params: Mapping[str, float] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: object) -> "LossConfig":
        config = _mapping(value)
        return cls(
            type=str(config.get("type", cls.type)),
            params=dict(_mapping(config.get("params"))),
        )


@dataclass(frozen=True)
class OptimizerConfig:
    """Optimizer configuration block used by the training entrypoint."""

    type: str = "AdamW"
    params: Mapping[str, float] = field(default_factory=lambda: {"lr": 0.001})

    @classmethod
    def from_mapping(cls, value: object) -> "OptimizerConfig":
        config = _mapping(value)
        return cls(
            type=str(config.get("type", cls.type)),
            params=dict(_mapping(config.get("params"))) or {"lr": 0.001},
        )


@dataclass(frozen=True)
class SchedulerConfig:
    """Scheduler configuration block used by the training entrypoint."""

    type: str = "OneCycleLR"
    params: Mapping[str, float | int] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: object) -> "SchedulerConfig":
        config = _mapping(value)
        return cls(
            type=str(config.get("type", cls.type)),
            params=dict(_mapping(config.get("params"))),
        )


@dataclass(frozen=True)
class RegularizationConfig:
    """Regularization settings read by the training entrypoint."""

    gradient_clip_val: float = 1.0

    @classmethod
    def from_mapping(cls, value: object) -> "RegularizationConfig":
        config = _mapping(value)
        return cls(
            gradient_clip_val=float(config.get("gradient_clip_val", cls.gradient_clip_val))
        )


@dataclass(frozen=True)
class PredictionParams:
    """Cleanup configuration used when converting logits into labels."""

    enable_cleanup: bool = True
    cleanup_kernel_size: int = 3
    cleanup_threshold: float = 5.0
    min_hole_size_factor: int = 64

    @classmethod
    def from_mapping(cls, value: object) -> "PredictionParams":
        config = _mapping(value)
        return cls(
            enable_cleanup=bool(config.get("enable_cleanup", cls.enable_cleanup)),
            cleanup_kernel_size=int(
                config.get("cleanup_kernel_size", cls.cleanup_kernel_size)
            ),
            cleanup_threshold=float(config.get("cleanup_threshold", cls.cleanup_threshold)),
            min_hole_size_factor=int(
                config.get("min_hole_size_factor", cls.min_hole_size_factor)
            ),
        )


@dataclass(frozen=True)
class VisualizationConfig:
    """Visualization callback settings used by the trainer."""

    num_samples: int = 4
    min_coverage: float = 0.05
    dpi: int = 300
    style: str = "seaborn"
    cmap: str = "viridis"

    @classmethod
    def from_mapping(cls, value: object) -> "VisualizationConfig":
        config = _mapping(value)
        return cls(
            num_samples=int(config.get("num_samples", cls.num_samples)),
            min_coverage=float(config.get("min_coverage", cls.min_coverage)),
            dpi=int(config.get("dpi", cls.dpi)),
            style=str(config.get("style", cls.style)),
            cmap=str(config.get("cmap", cls.cmap)),
        )


@dataclass(frozen=True)
class ModelConfigSnapshot:
    """Model-defining settings persisted with checkpoints."""

    encoder_name: str = "nvidia/mit-b0"
    encoder_revision: str | None = None
    num_classes: int = 1
    pretrained: bool = True

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        num_classes: int,
    ) -> "ModelConfigSnapshot":
        config = _mapping(value)
        return cls(
            encoder_name=str(config.get("encoder_name", cls.encoder_name)),
            encoder_revision=(
                str(config["encoder_revision"]) if config.get("encoder_revision") else None
            ),
            num_classes=int(config.get("num_classes", num_classes)),
            pretrained=bool(config.get("pretrained", cls.pretrained)),
        )


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset and augmentation settings consumed by dataloader creation."""

    processed_data_dir: str
    batch_size: int
    data_fraction: float
    dataloader: DataloaderConfig
    augmentation: AugmentationConfig


@dataclass(frozen=True)
class TrainerModuleConfig:
    """Settings needed to build the Lightning module and its dependencies."""

    num_epochs: int
    log_dir: str
    test_mode: bool
    num_classes: int
    model: ModelConfigSnapshot
    loss: LossConfig
    prediction: PredictionParams
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    visualization: VisualizationConfig


@dataclass(frozen=True)
class TrainerRuntimeConfig:
    """Settings used when constructing the PyTorch Lightning trainer."""

    num_epochs: int
    gradient_accumulation_steps: int
    gradient_clip_val: float


@dataclass(frozen=True)
class TrainingAppConfig:
    """Normalized application config for the training entrypoint."""

    logging: LoggingConfig
    dataset: DatasetConfig
    trainer_module: TrainerModuleConfig
    trainer_runtime: TrainerRuntimeConfig


def normalize_training_config(raw_config: Mapping[str, Any]) -> TrainingAppConfig:
    """Normalize raw YAML into explicit config objects for each training boundary."""

    config = dict(raw_config)
    if "thresholding" in config:
        raise ValueError(
            "The 'thresholding' block is no longer supported. Use the 'prediction' block instead."
        )

    test_mode = bool(config.get("test_mode", False))
    num_epochs = int(config.get("num_epochs", 100))
    batch_size = int(config.get("batch_size", 16))
    data_fraction = float(config.get("data_fraction", 1.0))

    if test_mode:
        test_settings = _mapping(config.get("test_mode_settings"))
        num_epochs = int(test_settings.get("num_epochs", num_epochs))
        batch_size = int(test_settings.get("batch_size", batch_size))
        data_fraction = float(test_settings.get("data_fraction", data_fraction))

    num_classes = int(config.get("num_classes", 1))
    model = ModelConfigSnapshot.from_mapping(config, num_classes=num_classes)
    prediction = PredictionParams.from_mapping(config.get("prediction"))
    visualization = VisualizationConfig.from_mapping(config.get("visualization"))

    dataset_config = DatasetConfig(
        processed_data_dir=str(config.get("processed_data_dir", "")),
        batch_size=batch_size,
        data_fraction=data_fraction,
        dataloader=DataloaderConfig.from_mapping(config.get("dataloader")),
        augmentation=AugmentationConfig.from_mapping(config.get("augmentation")),
    )
    trainer_module_config = TrainerModuleConfig(
        num_epochs=num_epochs,
        log_dir=str(config.get("log_dir", "logs")),
        test_mode=test_mode,
        num_classes=num_classes,
        model=model,
        loss=LossConfig.from_mapping(config.get("loss")),
        prediction=prediction,
        optimizer=OptimizerConfig.from_mapping(config.get("optimizer")),
        scheduler=SchedulerConfig.from_mapping(config.get("scheduler")),
        visualization=visualization,
    )
    trainer_runtime_config = TrainerRuntimeConfig(
        num_epochs=num_epochs,
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 1)),
        gradient_clip_val=RegularizationConfig.from_mapping(
            config.get("regularization")
        ).gradient_clip_val,
    )

    return TrainingAppConfig(
        logging=LoggingConfig.from_mapping(config.get("logging")),
        dataset=dataset_config,
        trainer_module=trainer_module_config,
        trainer_runtime=trainer_runtime_config,
    )

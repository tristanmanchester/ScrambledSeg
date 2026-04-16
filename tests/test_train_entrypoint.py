"""Direct smoke tests for the training entrypoint."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import scrambledSeg.training.train as train_module
import yaml
from scrambledSeg.training.config import AugmentationConfig, VisualizationConfig
from scrambledSeg.training.rich_progress import RichProgressCallback


class _DummyLoader:
    def __init__(self, size: int) -> None:
        self.dataset = list(range(size))
        self._size = size

    def __len__(self) -> int:
        return self._size


def test_training_main_builds_trainer_with_progress_callback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The public training entrypoint should wire datasets, module, and callbacks together."""

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "processed_data_dir": str(tmp_path / "processed"),
                "num_classes": 2,
                "num_epochs": 3,
                "batch_size": 2,
                "optimizer": {"params": {"lr": 0.001}},
                "logging": {"level": "INFO"},
                "regularization": {"gradient_clip_val": 1.0},
            }
        )
    )

    train_loader = _DummyLoader(size=4)
    val_loader = _DummyLoader(size=2)
    trainer_module = SimpleNamespace(vis_callback=SimpleNamespace(metrics_file=tmp_path / "metrics.csv"))
    captured: dict[str, object] = {}

    class FakeTrainer:
        def __init__(self, **kwargs) -> None:
            captured["trainer_kwargs"] = kwargs

        def fit(self, module) -> None:
            captured["fit_module"] = module

    monkeypatch.setattr(
        train_module,
        "create_dataloaders",
        lambda config: {"train": train_loader, "val": val_loader},
    )
    monkeypatch.setattr(
        train_module,
        "create_trainer_module",
        lambda config, train_dataloader, val_dataloader: trainer_module,
    )
    monkeypatch.setattr(train_module, "display_config_info", lambda config, config_path: None)
    monkeypatch.setattr(train_module, "setup_logging", lambda config: None)
    monkeypatch.setattr(train_module, "create_lightning_trainer", lambda config: FakeTrainer(**config))

    train_module.main(str(config_path))

    trainer_kwargs = captured["trainer_kwargs"]
    assert isinstance(trainer_kwargs, dict)
    assert trainer_kwargs["max_epochs"] == 3
    assert any(isinstance(callback, RichProgressCallback) for callback in trainer_kwargs["callbacks"])
    assert captured["fit_module"] is trainer_module


def test_training_main_applies_test_mode_overrides_before_setup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Test mode should rewrite the config before dataset and trainer setup."""

    config_path = tmp_path / "test_mode.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "processed_data_dir": str(tmp_path / "processed"),
                "num_classes": 2,
                "num_epochs": 20,
                "batch_size": 8,
                "data_fraction": 1.0,
                "optimizer": {"params": {"lr": 0.001}},
                "logging": {"level": "INFO"},
                "regularization": {"gradient_clip_val": 1.0},
                "test_mode": True,
                "test_mode_settings": {
                    "num_epochs": 2,
                    "batch_size": 1,
                    "data_fraction": 0.25,
                },
            }
        )
    )

    captured: dict[str, object] = {}
    trainer_module = SimpleNamespace(vis_callback=SimpleNamespace(metrics_file=tmp_path / "metrics.csv"))

    class FakeTrainer:
        def __init__(self, **kwargs) -> None:
            captured["trainer_kwargs"] = kwargs

        def fit(self, module) -> None:
            captured["fit_module"] = module

    def _capture_datasets(config):
        captured["dataset_config"] = config
        return {
            "train": _DummyLoader(size=3),
            "val": _DummyLoader(size=1),
        }

    def _capture_trainer_module(config, train_dataloader, val_dataloader):
        captured["trainer_config"] = config
        return trainer_module

    monkeypatch.setattr(train_module, "create_dataloaders", _capture_datasets)
    monkeypatch.setattr(train_module, "create_trainer_module", _capture_trainer_module)
    monkeypatch.setattr(train_module, "display_config_info", lambda config, config_path: None)
    monkeypatch.setattr(train_module, "setup_logging", lambda config: None)
    monkeypatch.setattr(train_module, "create_lightning_trainer", lambda config: FakeTrainer(**config))

    train_module.main(str(config_path))

    dataset_config = captured["dataset_config"]
    trainer_config = captured["trainer_config"]
    assert dataset_config.batch_size == 1
    assert dataset_config.data_fraction == 0.25
    assert dataset_config.augmentation == AugmentationConfig()
    assert trainer_config.test_mode is True
    assert trainer_config.num_epochs == 2
    assert trainer_config.visualization == VisualizationConfig()


def test_training_main_reraises_fit_errors_after_rendering_failure_panel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The training entrypoint should print one failure panel and re-raise the error."""

    config_path = tmp_path / "failure.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "processed_data_dir": str(tmp_path / "processed"),
                "num_classes": 2,
                "num_epochs": 3,
                "batch_size": 2,
                "optimizer": {"params": {"lr": 0.001}},
                "logging": {"level": "INFO"},
                "regularization": {"gradient_clip_val": 1.0},
            }
        )
    )

    rendered: list[object] = []
    trainer_module = SimpleNamespace(vis_callback=SimpleNamespace(metrics_file=tmp_path / "metrics.csv"))

    class FakeTrainer:
        def __init__(self, **kwargs) -> None:
            pass

        def fit(self, module) -> None:
            raise RuntimeError("boom")

    monkeypatch.setattr(
        train_module,
        "create_dataloaders",
        lambda config: {"train": _DummyLoader(size=4), "val": _DummyLoader(size=2)},
    )
    monkeypatch.setattr(
        train_module,
        "create_trainer_module",
        lambda config, train_dataloader, val_dataloader: trainer_module,
    )
    monkeypatch.setattr(train_module, "display_config_info", lambda config, config_path: None)
    monkeypatch.setattr(train_module, "setup_logging", lambda config: None)
    monkeypatch.setattr(train_module, "create_lightning_trainer", lambda config: FakeTrainer(**config))
    monkeypatch.setattr(
        train_module.Panel,
        "fit",
        staticmethod(lambda message, *args, **kwargs: f"PANEL:{message}"),
    )
    monkeypatch.setattr(train_module.console, "print", lambda *args, **kwargs: rendered.extend(args))

    with pytest.raises(RuntimeError, match="boom"):
        train_module.main(str(config_path))

    assert any("Training failed: boom" in str(item) for item in rendered)

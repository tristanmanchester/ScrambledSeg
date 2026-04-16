"""Direct tests for visualization helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from scrambledSeg.visualization.core import SegmentationVisualizer


def test_visualizer_mask_coverage_handles_multiclass_masks() -> None:
    """Coverage should treat any non-zero label as foreground."""

    visualizer = SegmentationVisualizer()
    mask = np.array([[0, 1], [0, 2]], dtype=np.uint8)

    assert visualizer._mask_coverage(mask) == pytest.approx(0.5)


def test_visualizer_finds_interesting_slices_by_coverage() -> None:
    """Slice ranking should prefer masks with the highest non-background coverage."""

    visualizer = SegmentationVisualizer(min_coverage=0.25)
    masks = np.array(
        [
            [[0, 0], [0, 0]],
            [[1, 1], [0, 0]],
            [[1, 1], [1, 0]],
            [[1, 1], [1, 1]],
        ],
        dtype=np.uint8,
    )

    assert visualizer.find_interesting_slices(masks, k=2) == [3, 2]


def test_plot_training_efficiency_returns_none_without_metrics_file() -> None:
    """Efficiency plotting should honor the optional metrics file contract."""

    visualizer = SegmentationVisualizer(metrics_file=None)

    assert visualizer.plot_training_efficiency() is None


def test_plot_training_efficiency_uses_recorded_progress_metrics(tmp_path: Path) -> None:
    """Efficiency plotting should render recorded runtime metrics when available."""

    metrics_file = tmp_path / "metrics.csv"
    pd.DataFrame(
        [
            {
                "step": 0,
                "epoch": 0,
                "train_learning_rate": 0.1,
                "train_gradient_norm": 1.2,
                "train_avg_gradient_norm": 0.12,
                "train_samples_per_second": 32.0,
                "train_batch_time": 0.5,
                "train_gpu_memory_used_gb": 2.0,
                "train_gpu_memory_total_gb": 3.0,
                "train_cpu_percent": 45.0,
            },
            {
                "step": 1,
                "epoch": 0,
                "train_learning_rate": 0.05,
                "train_gradient_norm": 1.0,
                "train_avg_gradient_norm": 0.1,
                "train_samples_per_second": 36.0,
                "train_batch_time": 0.45,
                "train_gpu_memory_used_gb": 2.2,
                "train_gpu_memory_total_gb": 3.0,
                "train_cpu_percent": 50.0,
            },
        ]
    ).to_csv(metrics_file, index=False)

    visualizer = SegmentationVisualizer(metrics_file=str(metrics_file))
    figure = visualizer.plot_training_efficiency()

    assert figure is not None
    axes = figure.axes
    assert len(axes) == 4
    assert axes[0].lines
    assert axes[1].lines
    assert axes[2].lines
    assert axes[3].lines

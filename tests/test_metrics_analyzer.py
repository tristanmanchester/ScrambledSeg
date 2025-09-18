"""Tests for the metrics analysis utilities."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("matplotlib")
pytest.importorskip("seaborn")
pd = pytest.importorskip("pandas")

from scrambledSeg.analysis.metrics_analyzer import MetricsAnalyzer


def _create_metrics_csv(tmp_path: Path) -> Path:
    data = {
        "step": [0, 1, 2],
        "epoch": [1, 2, 3],
        "train_loss": [1.0, 0.8, 0.6],
        "val_loss": [1.2, 0.85, 0.65],
        "train_iou": [0.40, 0.55, 0.70],
        "val_iou": [0.35, 0.50, 0.68],
        "train_precision": [0.50, 0.60, 0.70],
        "val_precision": [0.45, 0.58, 0.69],
        "train_recall": [0.48, 0.58, 0.65],
        "val_recall": [0.40, 0.57, 0.63],
        "train_f1": [0.49, 0.59, 0.68],
        "val_f1": [0.42, 0.56, 0.66],
        "train_dice": [0.50, 0.60, 0.70],
        "val_dice": [0.45, 0.57, 0.69],
        "train_specificity": [0.90, 0.92, 0.94],
        "val_specificity": [0.88, 0.91, 0.93],
        "epoch_time_minutes": [10.0, 9.0, 8.0],
    }

    df = pd.DataFrame(data)
    csv_path = tmp_path / "metrics.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_metrics_analyzer_generates_outputs(tmp_path) -> None:
    """The metrics analyzer should produce report artifacts without error."""

    csv_path = _create_metrics_csv(tmp_path)
    analyzer = MetricsAnalyzer(str(csv_path), experiment_name="demo")

    gaps = analyzer.analyze_generalization_gap()
    assert "iou" in gaps
    assert gaps["iou"] == pytest.approx(0.02, abs=1e-6)

    analysis = analyzer.run_full_analysis()
    assert analysis.total_epochs == 3
    assert analysis.generalization_gap["iou"] == pytest.approx(0.02, abs=1e-6)

    export_paths = analyzer.export_for_paper(tmp_path, format_for_latex=False)
    assert export_paths["metrics_table"].exists()
    table_df = pd.read_csv(export_paths["metrics_table"])
    assert "IoU" in table_df["Metric"].values

    assert export_paths["key_statistics"].exists()


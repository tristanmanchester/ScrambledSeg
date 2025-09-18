"""Post-training metrics analysis and reporting tools."""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

DEFAULT_CLASS_NAMES = ["Class_0", "Class_1", "Class_2", "Class_3"]

@dataclass
class MetricSummary:
    """Summary statistics for a single metric."""
    metric_name: str
    mean: float
    std: float
    min: float
    max: float
    median: float
    q25: float
    q75: float
    final_value: float
    improvement: float  # final - initial
    improvement_percent: float
    convergence_epoch: Optional[int]
    best_value: float
    best_epoch: int

@dataclass
class TrainingAnalysis:
    """Complete training analysis results."""
    experiment_name: str
    total_epochs: int
    total_steps: int
    training_time_minutes: Optional[float]
    convergence_analysis: Dict[str, Any]
    metric_summaries: Dict[str, MetricSummary]
    per_class_analysis: Dict[str, Dict[str, MetricSummary]]
    generalization_gap: Dict[str, float]
    statistical_tests: Dict[str, Any]

class MetricsAnalyzer:
    """Comprehensive metrics analysis and reporting."""

    def __init__(
        self,
        metrics_file: str,
        experiment_name: str = "experiment",
        class_names: Optional[List[str]] = None,
    ):
        """Initialize analyzer with metrics CSV file."""

        self.metrics_file = Path(metrics_file)
        self.experiment_name = experiment_name
        self.df = None
        self.set_class_names(class_names or DEFAULT_CLASS_NAMES)

        self._load_data()

    @staticmethod
    def _to_column_suffix(name: str) -> str:
        """Normalize class names for column lookup."""

        return name.strip().lower().replace(" ", "_")

    def set_class_names(self, class_names: List[str]) -> None:
        """Update tracked class names for per-class summaries."""

        if not class_names:
            raise ValueError("class_names must contain at least one entry")

        self.class_names = list(class_names)
        self._class_column_suffix = {
            name: self._to_column_suffix(name) for name in class_names
        }
    
    def _load_data(self):
        """Load and preprocess metrics data."""
        if not self.metrics_file.exists():
            raise FileNotFoundError(f"Metrics file not found: {self.metrics_file}")
        
        self.df = pd.read_csv(self.metrics_file)
        
        # Convert columns to numeric
        for col in self.df.columns:
            if col in ['step', 'epoch']:
                continue
            self.df[col] = pd.to_numeric(self.df[col].replace('', np.nan), errors='coerce')
        
        logger.info(f"Loaded metrics data: {len(self.df)} rows, {len(self.df.columns)} columns")
    
    def _calculate_metric_summary(self, series: pd.Series, metric_name: str) -> MetricSummary:
        """Calculate summary statistics for a metric series."""
        # Remove NaN values
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            logger.warning(f"No valid data for metric: {metric_name}")
            return MetricSummary(
                metric_name=metric_name,
                mean=np.nan, std=np.nan, min=np.nan, max=np.nan,
                median=np.nan, q25=np.nan, q75=np.nan,
                final_value=np.nan, improvement=np.nan, improvement_percent=np.nan,
                convergence_epoch=None, best_value=np.nan, best_epoch=-1
            )
        
        # Basic statistics
        mean_val = clean_series.mean()
        std_val = clean_series.std()
        min_val = clean_series.min()
        max_val = clean_series.max()
        median_val = clean_series.median()
        q25 = clean_series.quantile(0.25)
        q75 = clean_series.quantile(0.75)
        
        # Initial and final values
        initial_value = clean_series.iloc[0]
        final_value = clean_series.iloc[-1]
        improvement = final_value - initial_value
        improvement_percent = (improvement / abs(initial_value)) * 100 if initial_value != 0 else 0
        
        # Best value and epoch
        if 'loss' in metric_name.lower():
            best_value = min_val
            best_idx = clean_series.idxmin()
        else:
            best_value = max_val
            best_idx = clean_series.idxmax()
        
        best_epoch = self.df.loc[best_idx, 'epoch'] if best_idx in self.df.index else -1
        
        # Convergence analysis (simplified)
        convergence_epoch = self._estimate_convergence_epoch(clean_series, metric_name)
        
        return MetricSummary(
            metric_name=metric_name,
            mean=mean_val,
            std=std_val,
            min=min_val,
            max=max_val,
            median=median_val,
            q25=q25,
            q75=q75,
            final_value=final_value,
            improvement=improvement,
            improvement_percent=improvement_percent,
            convergence_epoch=convergence_epoch,
            best_value=best_value,
            best_epoch=best_epoch
        )
    
    def _estimate_convergence_epoch(self, series: pd.Series, metric_name: str, window_size: int = 10) -> Optional[int]:
        """Estimate when a metric converged using moving average stability."""
        if len(series) < window_size * 2:
            return None
        
        # Calculate moving average and its standard deviation
        moving_avg = series.rolling(window=window_size).mean()
        moving_std = series.rolling(window=window_size).std()
        
        # Define convergence as when the moving std is below a threshold
        # relative to the overall range of the metric
        threshold = (series.max() - series.min()) * 0.01  # 1% of range
        
        converged_indices = moving_std < threshold
        
        if converged_indices.any():
            # Find first sustained convergence (at least window_size consecutive steps)
            for i in range(window_size, len(converged_indices) - window_size):
                if converged_indices.iloc[i:i+window_size].all():
                    return self.df.loc[series.index[i], 'epoch']
        
        return None
    
    def analyze_generalization_gap(self) -> Dict[str, float]:
        """Calculate generalization gap between training and validation metrics."""
        gaps = {}
        
        # Common metrics to analyze
        base_metrics = ['loss', 'iou', 'precision', 'recall', 'f1', 'dice', 'specificity']
        
        for metric in base_metrics:
            train_col = f'train_{metric}'
            val_col = f'val_{metric}'
            
            if train_col in self.df.columns and val_col in self.df.columns:
                # Use final values for gap calculation
                train_final = self.df[train_col].dropna().iloc[-1] if not self.df[train_col].dropna().empty else np.nan
                val_final = self.df[val_col].dropna().iloc[-1] if not self.df[val_col].dropna().empty else np.nan
                
                if not (np.isnan(train_final) or np.isnan(val_final)):
                    if 'loss' in metric:
                        # For loss, gap is val_loss - train_loss (higher is worse)
                        gap = val_final - train_final
                    else:
                        # For performance metrics, gap is train - val (higher train is overfitting)
                        gap = train_final - val_final
                    
                    gaps[metric] = gap
        
        return gaps
    
    def perform_statistical_tests(self) -> Dict[str, Any]:
        """Perform statistical tests on metrics."""
        tests = {}
        
        # Test for normality of residuals (difference between train and val)
        base_metrics = ['iou', 'precision', 'recall', 'f1']
        
        for metric in base_metrics:
            train_col = f'train_{metric}'
            val_col = f'val_{metric}'
            
            if train_col in self.df.columns and val_col in self.df.columns:
                train_data = self.df[train_col].dropna()
                val_data = self.df[val_col].dropna()
                
                if len(train_data) > 0 and len(val_data) > 0:
                    # Paired t-test for significant difference
                    common_indices = train_data.index.intersection(val_data.index)
                    if len(common_indices) > 1:
                        train_paired = train_data.loc[common_indices]
                        val_paired = val_data.loc[common_indices]
                        
                        t_stat, p_value = stats.ttest_rel(train_paired, val_paired)
                        
                        tests[f'{metric}_ttest'] = {
                            't_statistic': t_stat,
                            'p_value': p_value,
                            'significant': p_value < 0.05,
                            'interpretation': 'Significant difference between train and val' if p_value < 0.05 else 'No significant difference'
                        }
        
        return tests
    
    def analyze_per_class_performance(self) -> Dict[str, Dict[str, MetricSummary]]:
        """Analyze per-class metric performance."""
        per_class_analysis = {}
        
        metric_types = ['iou', 'precision', 'recall', 'f1']
        
        for class_name in self.class_names:
            class_analysis = {}
            class_suffix = self._class_column_suffix[class_name]

            for metric_type in metric_types:
                # Analyze both training and validation
                for split in ['train', 'val']:
                    col_name = f'{split}_{metric_type}_{class_suffix}'

                    if col_name in self.df.columns:
                        summary = self._calculate_metric_summary(
                            self.df[col_name], 
                            f'{class_name}_{split}_{metric_type}'
                        )
                        class_analysis[f'{split}_{metric_type}'] = summary
            
            per_class_analysis[class_name] = class_analysis
        
        return per_class_analysis
    
    def run_full_analysis(self) -> TrainingAnalysis:
        """Run complete training analysis."""
        logger.info("Running comprehensive training analysis...")
        
        # Basic experiment info
        total_epochs = int(self.df['epoch'].max()) if 'epoch' in self.df.columns else 0
        total_steps = int(self.df['step'].max()) if 'step' in self.df.columns else len(self.df)
        
        # Estimate training time if epoch timing data is available
        training_time_minutes = None
        if 'epoch_time_minutes' in self.df.columns:
            training_time_minutes = self.df['epoch_time_minutes'].sum()
        
        # Analyze main metrics
        main_metrics = ['train_loss', 'val_loss', 'train_iou', 'val_iou', 
                       'train_precision', 'val_precision', 'train_recall', 'val_recall',
                       'train_f1', 'val_f1', 'train_dice', 'val_dice', 'train_specificity', 'val_specificity']
        
        metric_summaries = {}
        for metric in main_metrics:
            if metric in self.df.columns:
                summary = self._calculate_metric_summary(self.df[metric], metric)
                metric_summaries[metric] = summary
        
        # Per-class analysis
        per_class_analysis = self.analyze_per_class_performance()
        
        # Generalization gap
        generalization_gap = self.analyze_generalization_gap()
        
        # Statistical tests
        statistical_tests = self.perform_statistical_tests()
        
        # Convergence analysis
        convergence_info = {}
        for metric_name, summary in metric_summaries.items():
            if summary.convergence_epoch is not None:
                convergence_info[metric_name] = summary.convergence_epoch
        
        return TrainingAnalysis(
            experiment_name=self.experiment_name,
            total_epochs=total_epochs,
            total_steps=total_steps,
            training_time_minutes=training_time_minutes,
            convergence_analysis=convergence_info,
            metric_summaries=metric_summaries,
            per_class_analysis=per_class_analysis,
            generalization_gap=generalization_gap,
            statistical_tests=statistical_tests
        )
    
    def generate_report(self, output_dir: str, analysis: Optional[TrainingAnalysis] = None) -> Path:
        """Generate comprehensive analysis report."""
        if analysis is None:
            analysis = self.run_full_analysis()
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate different report formats
        self._generate_json_report(analysis, output_dir / "analysis_report.json")
        self._generate_markdown_report(analysis, output_dir / "analysis_report.md")
        self._generate_csv_summary(analysis, output_dir / "metrics_summary.csv")
        
        logger.info(f"Analysis reports generated in {output_dir}")
        return output_dir
    
    def _generate_json_report(self, analysis: TrainingAnalysis, filepath: Path):
        """Generate JSON report with all analysis results."""
        # Convert dataclasses to dictionaries for JSON serialization
        report_data = asdict(analysis)
        
        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
    
    def _generate_markdown_report(self, analysis: TrainingAnalysis, filepath: Path):
        """Generate human-readable markdown report."""
        with open(filepath, 'w') as f:
            f.write(f"# Training Analysis Report: {analysis.experiment_name}\\n\\n")
            
            # Experiment overview
            f.write("## Experiment Overview\\n")
            f.write(f"- **Total Epochs**: {analysis.total_epochs}\\n")
            f.write(f"- **Total Steps**: {analysis.total_steps}\\n")
            if analysis.training_time_minutes:
                f.write(f"- **Training Time**: {analysis.training_time_minutes:.1f} minutes ({analysis.training_time_minutes/60:.1f} hours)\\n")
            f.write("\\n")
            
            # Key metrics summary
            f.write("## Key Metrics Summary\\n")
            f.write("| Metric | Final Value | Best Value | Best Epoch | Improvement | Convergence Epoch |\\n")
            f.write("|--------|-------------|------------|------------|-------------|-------------------|\\n")
            
            for metric_name, summary in analysis.metric_summaries.items():
                if not np.isnan(summary.final_value):
                    f.write(f"| {metric_name} | {summary.final_value:.4f} | {summary.best_value:.4f} | {summary.best_epoch} | {summary.improvement:+.4f} | {summary.convergence_epoch or 'N/A'} |\\n")
            f.write("\\n")
            
            # Generalization gap
            f.write("## Generalization Analysis\\n")
            f.write("| Metric | Generalization Gap |\\n")
            f.write("|--------|-------------------|\\n")
            for metric, gap in analysis.generalization_gap.items():
                f.write(f"| {metric} | {gap:+.4f} |\\n")
            f.write("\\n")
            
            # Statistical tests
            if analysis.statistical_tests:
                f.write("## Statistical Tests\\n")
                for test_name, test_result in analysis.statistical_tests.items():
                    f.write(f"### {test_name}\\n")
                    f.write(f"- **T-statistic**: {test_result.get('t_statistic', 'N/A'):.4f}\\n")
                    f.write(f"- **P-value**: {test_result.get('p_value', 'N/A'):.4f}\\n")
                    f.write(f"- **Significant**: {test_result.get('significant', 'N/A')}\\n")
                    f.write(f"- **Interpretation**: {test_result.get('interpretation', 'N/A')}\\n\\n")
            
            # Per-class performance
            f.write("## Per-Class Performance Summary\\n")
            for class_name, class_metrics in analysis.per_class_analysis.items():
                f.write(f"### {class_name}\\n")
                f.write("| Metric | Final Value | Best Value | Improvement |\\n")
                f.write("|--------|-------------|------------|-------------|\\n")
                for metric_name, summary in class_metrics.items():
                    if not np.isnan(summary.final_value):
                        f.write(f"| {metric_name} | {summary.final_value:.4f} | {summary.best_value:.4f} | {summary.improvement:+.4f} |\\n")
                f.write("\\n")
    
    def _generate_csv_summary(self, analysis: TrainingAnalysis, filepath: Path):
        """Generate CSV summary for easy import into other tools."""
        rows = []
        
        # Main metrics
        for metric_name, summary in analysis.metric_summaries.items():
            rows.append({
                'experiment': analysis.experiment_name,
                'metric_category': 'main',
                'metric_name': metric_name,
                'final_value': summary.final_value,
                'best_value': summary.best_value,
                'best_epoch': summary.best_epoch,
                'mean': summary.mean,
                'std': summary.std,
                'improvement': summary.improvement,
                'improvement_percent': summary.improvement_percent,
                'convergence_epoch': summary.convergence_epoch
            })
        
        # Per-class metrics
        for class_name, class_metrics in analysis.per_class_analysis.items():
            for metric_name, summary in class_metrics.items():
                rows.append({
                    'experiment': analysis.experiment_name,
                    'metric_category': f'per_class_{class_name}',
                    'metric_name': metric_name,
                    'final_value': summary.final_value,
                    'best_value': summary.best_value,
                    'best_epoch': summary.best_epoch,
                    'mean': summary.mean,
                    'std': summary.std,
                    'improvement': summary.improvement,
                    'improvement_percent': summary.improvement_percent,
                    'convergence_epoch': summary.convergence_epoch
                })
        
        df_summary = pd.DataFrame(rows)
        df_summary.to_csv(filepath, index=False)
    
    def export_for_paper(self, output_dir: str, format_for_latex: bool = True) -> Dict[str, Path]:
        """Export analysis results formatted for academic paper inclusion."""
        analysis = self.run_full_analysis()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        exported_files = {}
        
        # Create paper-ready summary table
        paper_table = self._create_paper_table(analysis, format_for_latex)
        table_file = output_dir / ("paper_metrics_table.tex" if format_for_latex else "paper_metrics_table.csv")
        
        if format_for_latex:
            with open(table_file, 'w') as f:
                f.write(paper_table)
        else:
            paper_table.to_csv(table_file, index=False)
        
        exported_files['metrics_table'] = table_file
        
        # Export key statistics for text
        key_stats = self._extract_key_statistics(analysis)
        stats_file = output_dir / "key_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(key_stats, f, indent=2)
        
        exported_files['key_statistics'] = stats_file
        
        return exported_files
    
    def _create_paper_table(self, analysis: TrainingAnalysis, latex_format: bool = True) -> str:
        """Create a formatted metrics table suitable for reports."""
        # Select key metrics for paper
        key_metrics = {
            'val_iou': 'IoU',
            'val_precision': 'Precision', 
            'val_recall': 'Recall',
            'val_f1': 'F1-Score',
            'val_dice': 'Dice'
        }
        
        if latex_format:
            table_lines = [
                "\\\\begin{table}[h]",
                "\\\\centering",
                "\\\\caption{Model Performance Metrics}",
                "\\\\label{tab:performance_metrics}",
                "\\\\begin{tabular}{lccc}",
                "\\\\toprule",
                "Metric & Final Value & Best Value & Improvement \\\\\\\\",
                "\\\\midrule"
            ]
            
            for metric_key, metric_display in key_metrics.items():
                if metric_key in analysis.metric_summaries:
                    summary = analysis.metric_summaries[metric_key]
                    if not np.isnan(summary.final_value):
                        table_lines.append(
                            f"{metric_display} & {summary.final_value:.3f} & {summary.best_value:.3f} & {summary.improvement:+.3f} \\\\\\\\"
                        )
            
            table_lines.extend([
                "\\\\bottomrule",
                "\\\\end{tabular}",
                "\\\\end{table}"
            ])
            
            return "\\n".join(table_lines)
        else:
            # Return pandas DataFrame for CSV export
            rows = []
            for metric_key, metric_display in key_metrics.items():
                if metric_key in analysis.metric_summaries:
                    summary = analysis.metric_summaries[metric_key]
                    rows.append({
                        'Metric': metric_display,
                        'Final Value': f"{summary.final_value:.3f}",
                        'Best Value': f"{summary.best_value:.3f}",
                        'Improvement': f"{summary.improvement:+.3f}"
                    })
            
            return pd.DataFrame(rows)
    
    def _extract_key_statistics(self, analysis: TrainingAnalysis) -> Dict[str, Any]:
        """Extract key statistics for paper text."""
        stats = {
            'total_epochs': analysis.total_epochs,
            'training_time_hours': analysis.training_time_minutes / 60 if analysis.training_time_minutes else None,
            'final_val_iou': None,
            'final_val_precision': None,
            'final_val_recall': None,
            'final_val_f1': None,
            'convergence_epoch_iou': None,
            'generalization_gap_iou': analysis.generalization_gap.get('iou'),
            'best_val_iou': None,
            'best_val_iou_epoch': None
        }
        
        # Extract specific metric values
        if 'val_iou' in analysis.metric_summaries:
            iou_summary = analysis.metric_summaries['val_iou']
            stats['final_val_iou'] = iou_summary.final_value
            stats['convergence_epoch_iou'] = iou_summary.convergence_epoch
            stats['best_val_iou'] = iou_summary.best_value
            stats['best_val_iou_epoch'] = iou_summary.best_epoch
        
        if 'val_precision' in analysis.metric_summaries:
            stats['final_val_precision'] = analysis.metric_summaries['val_precision'].final_value
        
        if 'val_recall' in analysis.metric_summaries:
            stats['final_val_recall'] = analysis.metric_summaries['val_recall'].final_value
        
        if 'val_f1' in analysis.metric_summaries:
            stats['final_val_f1'] = analysis.metric_summaries['val_f1'].final_value
        
        # Remove None values
        return {k: v for k, v in stats.items() if v is not None}

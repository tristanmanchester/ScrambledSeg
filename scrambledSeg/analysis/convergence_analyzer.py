"""Real-time convergence analysis and plateau detection for training optimization."""

import numpy as np
import torch
import logging
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from dataclasses import dataclass
from enum import Enum
import pytorch_lightning as pl

logger = logging.getLogger(__name__)

class ConvergenceStatus(Enum):
    """Status of convergence analysis."""
    NOT_STARTED = "not_started"
    IMPROVING = "improving"
    PLATEAUED = "plateaued"
    CONVERGED = "converged"
    DIVERGING = "diverging"

@dataclass
class ConvergenceState:
    """Current state of convergence analysis."""
    status: ConvergenceStatus
    plateau_steps: int
    best_value: float
    best_step: int
    steps_since_improvement: int
    improvement_threshold: float
    stability_metric: float
    trend_direction: str  # 'improving', 'stable', 'degrading'

class ConvergenceDetector:
    """Detects convergence and plateau in training metrics."""
    
    def __init__(
        self,
        metric_name: str,
        patience: int = 10,
        min_delta: float = 1e-4,
        window_size: int = 5,
        stability_threshold: float = 1e-3,
        mode: str = 'min'  # 'min' for loss, 'max' for accuracy metrics
    ):
        """Initialize convergence detector.
        
        Args:
            metric_name: Name of the metric to monitor
            patience: Number of steps to wait for improvement before declaring plateau
            min_delta: Minimum change to qualify as improvement
            window_size: Window size for computing moving statistics
            stability_threshold: Threshold for declaring stability
            mode: 'min' for metrics where lower is better, 'max' for higher is better
        """
        self.metric_name = metric_name
        self.patience = patience
        self.min_delta = min_delta
        self.window_size = window_size
        self.stability_threshold = stability_threshold
        self.mode = mode
        
        # State tracking
        self.values = deque(maxlen=100)  # Keep last 100 values
        self.steps = deque(maxlen=100)
        self.moving_avg = deque(maxlen=50)
        self.moving_std = deque(maxlen=50)
        
        # Convergence state
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.best_step = 0
        self.steps_since_improvement = 0
        self.plateau_steps = 0
        self.converged = False
        self.current_status = ConvergenceStatus.NOT_STARTED
        
    def _is_improvement(self, new_value: float) -> bool:
        """Check if new value represents an improvement."""
        if self.mode == 'min':
            return new_value < (self.best_value - self.min_delta)
        else:
            return new_value > (self.best_value + self.min_delta)
    
    def _calculate_stability(self) -> float:
        """Calculate stability metric based on recent variance."""
        if len(self.moving_std) < 3:
            return float('inf')
        
        recent_std = np.mean(list(self.moving_std)[-3:])
        value_range = max(self.values) - min(self.values) if len(self.values) > 1 else 1.0
        
        # Normalize std by the range of values
        normalized_std = recent_std / max(value_range, 1e-8)
        return normalized_std
    
    def _determine_trend(self) -> str:
        """Determine the trend direction of recent values."""
        if len(self.moving_avg) < 3:
            return 'unknown'
        
        recent_values = list(self.moving_avg)[-3:]
        
        if self.mode == 'min':
            # For loss: improving means decreasing
            if recent_values[-1] < recent_values[0] - self.min_delta:
                return 'improving'
            elif recent_values[-1] > recent_values[0] + self.min_delta:
                return 'degrading'
        else:
            # For accuracy metrics: improving means increasing
            if recent_values[-1] > recent_values[0] + self.min_delta:
                return 'improving'
            elif recent_values[-1] < recent_values[0] - self.min_delta:
                return 'degrading'
        
        return 'stable'
    
    def update(self, value: float, step: int) -> ConvergenceState:
        """Update detector with new metric value and return current state."""
        self.values.append(value)
        self.steps.append(step)
        
        # Update moving statistics
        if len(self.values) >= self.window_size:
            recent_values = list(self.values)[-self.window_size:]
            self.moving_avg.append(np.mean(recent_values))
            self.moving_std.append(np.std(recent_values))
        
        # Check for improvement
        if self._is_improvement(value):
            self.best_value = value
            self.best_step = step
            self.steps_since_improvement = 0
            self.plateau_steps = 0
            self.current_status = ConvergenceStatus.IMPROVING
        else:
            self.steps_since_improvement += 1
        
        # Check for plateau
        if self.steps_since_improvement >= self.patience:
            self.plateau_steps = self.steps_since_improvement
            
            # Calculate stability
            stability = self._calculate_stability()
            
            if stability < self.stability_threshold:
                self.current_status = ConvergenceStatus.CONVERGED
                self.converged = True
            else:
                self.current_status = ConvergenceStatus.PLATEAUED
        
        # Determine trend
        trend = self._determine_trend()
        
        return ConvergenceState(
            status=self.current_status,
            plateau_steps=self.plateau_steps,
            best_value=self.best_value,
            best_step=self.best_step,
            steps_since_improvement=self.steps_since_improvement,
            improvement_threshold=self.min_delta,
            stability_metric=self._calculate_stability(),
            trend_direction=trend
        )
    
    def reset(self):
        """Reset detector state."""
        self.values.clear()
        self.steps.clear()
        self.moving_avg.clear()
        self.moving_std.clear()
        
        self.best_value = float('inf') if self.mode == 'min' else float('-inf')
        self.best_step = 0
        self.steps_since_improvement = 0
        self.plateau_steps = 0
        self.converged = False
        self.current_status = ConvergenceStatus.NOT_STARTED

class MultiMetricConvergenceAnalyzer:
    """Analyzes convergence across multiple metrics simultaneously."""
    
    def __init__(self, convergence_configs: Dict[str, Dict[str, Any]]):
        """Initialize analyzer with multiple metrics.
        
        Args:
            convergence_configs: Dict mapping metric names to detector configurations
        """
        self.detectors = {}
        
        for metric_name, config in convergence_configs.items():
            self.detectors[metric_name] = ConvergenceDetector(
                metric_name=metric_name,
                **config
            )
        
        self.global_convergence_state = {}
    
    def update(self, metrics: Dict[str, float], step: int) -> Dict[str, ConvergenceState]:
        """Update all detectors with new metric values."""
        states = {}
        
        for metric_name, value in metrics.items():
            if metric_name in self.detectors:
                state = self.detectors[metric_name].update(value, step)
                states[metric_name] = state
        
        # Update global convergence state
        self._update_global_state(states, step)
        
        return states
    
    def _update_global_state(self, states: Dict[str, ConvergenceState], step: int):
        """Update global convergence analysis."""
        converged_metrics = sum(1 for state in states.values() if state.status == ConvergenceStatus.CONVERGED)
        plateaued_metrics = sum(1 for state in states.values() if state.status == ConvergenceStatus.PLATEAUED)
        improving_metrics = sum(1 for state in states.values() if state.status == ConvergenceStatus.IMPROVING)
        
        self.global_convergence_state = {
            'step': step,
            'total_metrics': len(states),
            'converged_metrics': converged_metrics,
            'plateaued_metrics': plateaued_metrics,
            'improving_metrics': improving_metrics,
            'convergence_ratio': converged_metrics / max(len(states), 1),
            'plateau_ratio': plateaued_metrics / max(len(states), 1),
            'overall_status': self._determine_overall_status(states)
        }
    
    def _determine_overall_status(self, states: Dict[str, ConvergenceState]) -> str:
        """Determine overall training status."""
        if not states:
            return 'unknown'
        
        converged_count = sum(1 for state in states.values() if state.status == ConvergenceStatus.CONVERGED)
        plateaued_count = sum(1 for state in states.values() if state.status == ConvergenceStatus.PLATEAUED)
        improving_count = sum(1 for state in states.values() if state.status == ConvergenceStatus.IMPROVING)
        
        total_count = len(states)
        
        if converged_count >= total_count * 0.7:  # 70% of metrics converged
            return 'converged'
        elif plateaued_count >= total_count * 0.7:  # 70% of metrics plateaued
            return 'plateaued'
        elif improving_count >= total_count * 0.5:  # 50% of metrics improving
            return 'improving'
        else:
            return 'mixed'
    
    def should_stop_training(self, 
                           min_epochs: int = 10,
                           convergence_threshold: float = 0.8,
                           plateau_threshold: float = 0.6) -> bool:
        """Determine if training should be stopped based on convergence analysis."""
        if not self.global_convergence_state:
            return False
        
        current_epoch = self.global_convergence_state.get('step', 0)
        
        # Don't stop before minimum epochs
        if current_epoch < min_epochs:
            return False
        
        # Stop if enough metrics have converged
        if self.global_convergence_state.get('convergence_ratio', 0) >= convergence_threshold:
            logger.info(f"Training convergence detected: {self.global_convergence_state['convergence_ratio']:.2%} of metrics converged")
            return True
        
        # Stop if too many metrics are plateaued without improvement
        if self.global_convergence_state.get('plateau_ratio', 0) >= plateau_threshold:
            logger.info(f"Training plateau detected: {self.global_convergence_state['plateau_ratio']:.2%} of metrics plateaued")
            return True
        
        return False
    
    def get_convergence_summary(self) -> Dict[str, Any]:
        """Get comprehensive convergence summary."""
        summary = {
            'global_state': self.global_convergence_state.copy(),
            'metric_details': {}
        }
        
        for metric_name, detector in self.detectors.items():
            if detector.values:
                summary['metric_details'][metric_name] = {
                    'current_value': detector.values[-1] if detector.values else None,
                    'best_value': detector.best_value,
                    'best_step': detector.best_step,
                    'steps_since_improvement': detector.steps_since_improvement,
                    'status': detector.current_status.value,
                    'converged': detector.converged,
                    'stability': detector._calculate_stability(),
                    'trend': detector._determine_trend()
                }
        
        return summary

class ConvergenceCallback(pl.Callback):
    """PyTorch Lightning callback for convergence analysis."""
    
    def __init__(
        self,
        convergence_configs: Dict[str, Dict[str, Any]],
        log_interval: int = 10,
        enable_early_stopping: bool = False,
        early_stop_patience: int = 20,
        min_epochs: int = 10
    ):
        """Initialize convergence callback.
        
        Args:
            convergence_configs: Configuration for each metric to monitor
            log_interval: How often to log convergence analysis (in steps)
            enable_early_stopping: Whether to enable early stopping based on convergence
            early_stop_patience: Additional patience for early stopping
            min_epochs: Minimum epochs before early stopping can trigger
        """
        super().__init__()
        
        self.analyzer = MultiMetricConvergenceAnalyzer(convergence_configs)
        self.log_interval = log_interval
        self.enable_early_stopping = enable_early_stopping
        self.early_stop_patience = early_stop_patience
        self.min_epochs = min_epochs
        
        self.last_log_step = 0
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Update convergence analysis after each training batch."""
        if outputs is None:
            return
        
        # Extract relevant metrics from outputs
        metrics = {}
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                metrics[key] = value.item()
            elif isinstance(value, (int, float)):
                metrics[key] = float(value)
        
        # Update convergence analysis
        states = self.analyzer.update(metrics, trainer.global_step)
        
        # Log convergence analysis periodically
        if trainer.global_step - self.last_log_step >= self.log_interval:
            self._log_convergence_state(trainer, states)
            self.last_log_step = trainer.global_step
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Check for early stopping after validation."""
        if not self.enable_early_stopping:
            return
        
        # Check if training should stop
        should_stop = self.analyzer.should_stop_training(
            min_epochs=self.min_epochs,
            convergence_threshold=0.8,
            plateau_threshold=0.6
        )
        
        if should_stop:
            logger.info("Early stopping triggered by convergence analysis")
            trainer.should_stop = True
    
    def _log_convergence_state(self, trainer, states: Dict[str, ConvergenceState]):
        """Log convergence state information."""
        # Log individual metric convergence states
        for metric_name, state in states.items():
            prefix = f"convergence/{metric_name}"
            trainer.logger.log_metrics({
                f"{prefix}/status": state.status.value,
                f"{prefix}/steps_since_improvement": state.steps_since_improvement,
                f"{prefix}/stability": state.stability_metric,
                f"{prefix}/plateau_steps": state.plateau_steps
            }, step=trainer.global_step)
        
        # Log global convergence state
        global_state = self.analyzer.global_convergence_state
        if global_state:
            trainer.logger.log_metrics({
                "convergence/global_status": global_state.get('overall_status', 'unknown'),
                "convergence/convergence_ratio": global_state.get('convergence_ratio', 0),
                "convergence/plateau_ratio": global_state.get('plateau_ratio', 0),
                "convergence/converged_metrics": global_state.get('converged_metrics', 0),
                "convergence/plateaued_metrics": global_state.get('plateaued_metrics', 0)
            }, step=trainer.global_step)
    
    def get_convergence_report(self) -> Dict[str, Any]:
        """Get final convergence analysis report."""
        return self.analyzer.get_convergence_summary()
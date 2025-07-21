"""Hardware profiling utilities for monitoring GPU, CPU, and memory usage during training."""

import time
import threading
import logging
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import psutil
import torch

logger = logging.getLogger(__name__)

@dataclass
class HardwareSnapshot:
    """Single snapshot of hardware metrics."""
    timestamp: float
    cpu_percent: float
    cpu_count: int
    memory_used_gb: float
    memory_total_gb: float
    memory_percent: float
    gpu_memory_used_gb: float
    gpu_memory_total_gb: float
    gpu_utilization_percent: float
    gpu_temperature: Optional[float]
    gpu_power_draw: Optional[float]

class HardwareProfiler:
    """Hardware profiler for monitoring system resources during training."""
    
    def __init__(
        self,
        sample_interval: float = 1.0,
        enable_gpu_monitoring: bool = True,
        log_file: Optional[str] = None,
        callback: Optional[Callable[[HardwareSnapshot], None]] = None
    ):
        """Initialize hardware profiler.
        
        Args:
            sample_interval: Time between samples in seconds
            enable_gpu_monitoring: Whether to monitor GPU metrics
            log_file: Optional file to save hardware logs
            callback: Optional callback function for each hardware snapshot
        """
        self.sample_interval = sample_interval
        self.enable_gpu_monitoring = enable_gpu_monitoring and torch.cuda.is_available()
        self.log_file = Path(log_file) if log_file else None
        self.callback = callback
        
        self._monitoring = False
        self._monitor_thread = None
        self._snapshots: List[HardwareSnapshot] = []
        
        # Initialize GPU monitoring if available
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available and self.enable_gpu_monitoring:
            try:
                import pynvml
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.pynvml = pynvml
                logger.info("GPU monitoring enabled with NVML")
            except (ImportError, Exception) as e:
                logger.warning(f"Could not initialize GPU monitoring: {e}")
                self.enable_gpu_monitoring = False
                self.pynvml = None
        else:
            self.pynvml = None
            
        logger.info(f"Hardware profiler initialized - GPU monitoring: {self.enable_gpu_monitoring}")
    
    def _get_gpu_metrics(self) -> Dict[str, float]:
        """Get GPU metrics using PyTorch and optionally NVML."""
        metrics = {
            'gpu_memory_used_gb': 0.0,
            'gpu_memory_total_gb': 0.0,
            'gpu_utilization_percent': 0.0,
            'gpu_temperature': None,
            'gpu_power_draw': None
        }
        
        if not self.gpu_available:
            return metrics
            
        try:
            # PyTorch GPU memory
            metrics['gpu_memory_used_gb'] = torch.cuda.memory_allocated() / 1024**3
            metrics['gpu_memory_total_gb'] = torch.cuda.memory_reserved() / 1024**3
            
            # NVML metrics if available
            if self.pynvml and self.enable_gpu_monitoring:
                try:
                    # GPU utilization
                    util = self.pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                    metrics['gpu_utilization_percent'] = util.gpu
                    
                    # Temperature
                    temp = self.pynvml.nvmlDeviceGetTemperature(self.gpu_handle, self.pynvml.NVML_TEMPERATURE_GPU)
                    metrics['gpu_temperature'] = temp
                    
                    # Power draw
                    power = self.pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000.0  # Convert mW to W
                    metrics['gpu_power_draw'] = power
                    
                except Exception as e:
                    logger.debug(f"Error getting NVML metrics: {e}")
                    
        except Exception as e:
            logger.debug(f"Error getting GPU metrics: {e}")
            
        return metrics
    
    def _get_cpu_memory_metrics(self) -> Dict[str, float]:
        """Get CPU and memory metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_used_gb = memory.used / 1024**3
            memory_total_gb = memory.total / 1024**3
            memory_percent = memory.percent
            
            return {
                'cpu_percent': cpu_percent,
                'cpu_count': cpu_count,
                'memory_used_gb': memory_used_gb,
                'memory_total_gb': memory_total_gb,
                'memory_percent': memory_percent
            }
        except Exception as e:
            logger.debug(f"Error getting CPU/memory metrics: {e}")
            return {
                'cpu_percent': 0.0,
                'cpu_count': 1,
                'memory_used_gb': 0.0,
                'memory_total_gb': 0.0,
                'memory_percent': 0.0
            }
    
    def _capture_snapshot(self) -> HardwareSnapshot:
        """Capture a single hardware snapshot."""
        timestamp = time.time()
        
        # Get CPU and memory metrics
        cpu_memory_metrics = self._get_cpu_memory_metrics()
        
        # Get GPU metrics
        gpu_metrics = self._get_gpu_metrics()
        
        # Combine all metrics
        snapshot = HardwareSnapshot(
            timestamp=timestamp,
            **cpu_memory_metrics,
            **gpu_metrics
        )
        
        return snapshot
    
    def _monitoring_loop(self):
        """Main monitoring loop run in background thread."""
        while self._monitoring:
            try:
                snapshot = self._capture_snapshot()
                self._snapshots.append(snapshot)
                
                # Call callback if provided
                if self.callback:
                    self.callback(snapshot)
                
                # Log to file if specified
                if self.log_file:
                    self._write_snapshot_to_file(snapshot)
                    
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                
            time.sleep(self.sample_interval)
    
    def _write_snapshot_to_file(self, snapshot: HardwareSnapshot):
        """Write a single snapshot to log file."""
        try:
            # Ensure log file directory exists
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Write as JSON lines
            with open(self.log_file, 'a') as f:
                json.dump(asdict(snapshot), f)
                f.write('\n')
        except Exception as e:
            logger.error(f"Error writing to log file: {e}")
    
    def start(self):
        """Start hardware monitoring."""
        if self._monitoring:
            logger.warning("Hardware monitoring already running")
            return
            
        self._monitoring = True
        self._snapshots.clear()
        
        # Create and start monitoring thread
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
        
        logger.info("Hardware monitoring started")
    
    def stop(self):
        """Stop hardware monitoring."""
        if not self._monitoring:
            return
            
        self._monitoring = False
        
        # Wait for thread to finish
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)
            
        logger.info(f"Hardware monitoring stopped. Captured {len(self._snapshots)} snapshots")
    
    def get_snapshots(self) -> List[HardwareSnapshot]:
        """Get all captured snapshots."""
        return self._snapshots.copy()
    
    def get_summary_stats(self) -> Dict[str, float]:
        """Get summary statistics from captured snapshots."""
        if not self._snapshots:
            return {}
        
        # Extract metrics from snapshots
        cpu_percents = [s.cpu_percent for s in self._snapshots]
        memory_percents = [s.memory_percent for s in self._snapshots]
        gpu_memory_used = [s.gpu_memory_used_gb for s in self._snapshots]
        gpu_utilizations = [s.gpu_utilization_percent for s in self._snapshots if s.gpu_utilization_percent > 0]
        gpu_temperatures = [s.gpu_temperature for s in self._snapshots if s.gpu_temperature is not None]
        
        stats = {
            'num_samples': len(self._snapshots),
            'duration_minutes': (self._snapshots[-1].timestamp - self._snapshots[0].timestamp) / 60.0,
            'cpu_percent_avg': sum(cpu_percents) / len(cpu_percents),
            'cpu_percent_max': max(cpu_percents),
            'memory_percent_avg': sum(memory_percents) / len(memory_percents),
            'memory_percent_max': max(memory_percents),
            'gpu_memory_max_gb': max(gpu_memory_used),
            'gpu_memory_avg_gb': sum(gpu_memory_used) / len(gpu_memory_used),
        }
        
        if gpu_utilizations:
            stats['gpu_utilization_avg'] = sum(gpu_utilizations) / len(gpu_utilizations)
            stats['gpu_utilization_max'] = max(gpu_utilizations)
            
        if gpu_temperatures:
            stats['gpu_temperature_avg'] = sum(gpu_temperatures) / len(gpu_temperatures)
            stats['gpu_temperature_max'] = max(gpu_temperatures)
        
        return stats
    
    def save_summary(self, filepath: str):
        """Save summary statistics to JSON file."""
        stats = self.get_summary_stats()
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)
            
        logger.info(f"Hardware profiling summary saved to {filepath}")
    
    def clear_snapshots(self):
        """Clear all captured snapshots."""
        self._snapshots.clear()

class TrainingHardwareCallback:
    """Callback for integrating hardware profiling with PyTorch Lightning training."""
    
    def __init__(
        self,
        sample_interval: float = 2.0,
        log_dir: str = "logs/hardware",
        enable_gpu_monitoring: bool = True
    ):
        """Initialize training hardware callback.
        
        Args:
            sample_interval: Time between hardware samples in seconds
            log_dir: Directory to save hardware logs
            enable_gpu_monitoring: Whether to monitor GPU metrics
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create profiler
        log_file = self.log_dir / "hardware_profile.jsonl"
        self.profiler = HardwareProfiler(
            sample_interval=sample_interval,
            enable_gpu_monitoring=enable_gpu_monitoring,
            log_file=str(log_file)
        )
        
    def on_train_start(self):
        """Start hardware monitoring when training begins."""
        self.profiler.start()
        
    def on_train_end(self):
        """Stop hardware monitoring and save summary when training ends."""
        self.profiler.stop()
        
        # Save summary statistics
        summary_file = self.log_dir / "hardware_summary.json"
        self.profiler.save_summary(summary_file)
        
        # Log summary stats
        stats = self.profiler.get_summary_stats()
        logger.info("Hardware Profiling Summary:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
    
    def get_current_stats(self) -> Dict[str, float]:
        """Get current hardware statistics for logging."""
        try:
            snapshot = self.profiler._capture_snapshot()
            return asdict(snapshot)
        except Exception as e:
            logger.debug(f"Error getting current hardware stats: {e}")
            return {}
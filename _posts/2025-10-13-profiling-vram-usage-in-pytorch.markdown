---
layout: post
title:  "Profiling VRAM Usage in PyTorch"
date:   2025-10-13 12:00:00 -0400
categories: Tools
---

When training deep learning models using PyTorch, we often come across OOM errors. In this post, I will discuss some methods to profile VRAM usage in PyTorch to identify memory bottlenecks.

# PyTorch Built-in Profiler

PyTorch offers a built-in profiler that profiles net allocated VRAM during operations.

```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    profile_memory=True,
    record_shapes=True,
    with_stack=True # Capture stack traces
) as prof:
    output = self.model.forward()

# Save profiling results to file
with open(profile_txt_path, 'w') as f:
    f.write(prof.key_averages(group_by_stack_n=5).table(
        sort_by="cuda_memory_usage",
        row_limit=30,
        max_name_column_width=60
    ))
```

This will create a txt VRAM profile table that looks like this.

![]({{ '/assets/2025-10-13/PyTorchProfiler.png' | relative_url }})

You can also save a Chrome trace file and visualize it in Chrome's trace viewer.

```python
prof.export_chrome_trace(trace_path)
```

You can upload the saved json file in [Perfetto Trace Viewer](https://ui.perfetto.dev/#!/viewer) to visualize the trace. Zoom in and it will look like this.

![]({{ '/assets/2025-10-13/ChromeTrace.png' | relative_url }})

There are 2 python processes in the Chrome trace. The `python 3201404` is CPU / Runtime operations process. It contains PyTorch dispatchers (e.g. `aten::_efficient_attention_forward`)and CUDA runtime API calls (e.g. `cudaLaunchKernel`).  The `python 0` is the CUDA kernel execution process. It contains the actual GPU kernels launched by those runtime calls.

GPU kernels are scheduled into **CUDA streams**, which are like ordered command queues. By default, PyTorch uses the default stream. Therefore, the GPU kernels are executed in sequential order. The different rows in the same stream represents the call stack.

# Custom Profiler

I wrote a custom profiler to record the peak total VRAM between two checkpoints.

```python
import torch
import functools
from contextlib import contextmanager
from typing import Optional, Dict, List
import time

class VRAMProfiler:
    """Utility for profiling VRAM usage during inference."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.checkpoints: List[Dict] = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.peak_memory = 0

    def reset(self):
        """Reset profiling data."""
        self.checkpoints = []
        if self.enabled and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.peak_memory = 0

    def checkpoint(self, name: str, tensor: Optional[torch.Tensor] = None,
                   tensor_info: Optional[str] = None):
        """Record a memory checkpoint."""
        if not self.enabled or not torch.cuda.is_available():
            return

        current_memory = torch.cuda.memory_allocated()
        peak_since_last = torch.cuda.max_memory_allocated()

        self.peak_memory = max(self.peak_memory, peak_since_last)

        checkpoint_data = {
            'name': name,
            'current_gb': current_memory / 1e9,
            'peak_gb': self.peak_memory / 1e9,
            'peak_since_last_gb': peak_since_last / 1e9,
            'timestamp': time.time()
        }

        if tensor is not None:
            tensor_memory = tensor.element_size() * tensor.nelement() / 1e9
            checkpoint_data['tensor_gb'] = tensor_memory
            checkpoint_data['tensor_shape'] = str(tuple(tensor.shape))
            checkpoint_data['tensor_dtype'] = str(tensor.dtype)

        if tensor_info is not None:
            checkpoint_data['info'] = tensor_info

        self.checkpoints.append(checkpoint_data)

        torch.cuda.reset_peak_memory_stats()

    def print_summary(self):
        """Print formatted summary of memory usage."""
        if not self.enabled or not self.checkpoints:
            return

        print("\n" + "="*80)
        print("VRAM PROFILING SUMMARY")
        print("="*80)

        # Print header
        print(f"{'Stage':<40} {'Current (GB)':<15} {'Interval Peak (GB)':<20} {'Overall Peak (GB)':<20}")
        print("-"*100)

        # Print each checkpoint
        for cp in self.checkpoints:
            name = cp['name']
            current = f"{cp['current_gb']:.3f}"
            interval_peak = f"{cp['peak_since_last_gb']:.3f}"
            peak = f"{cp['peak_gb']:.3f}"
            print(f"{name:<40} {current:<15} {interval_peak:<20} {peak:<20}")

            if 'tensor_gb' in cp:
                tensor_info = f"  └─ Tensor: {cp['tensor_shape']} ({cp['tensor_dtype']}) = {cp['tensor_gb']:.3f} GB"
                print(tensor_info)

        print("="*80)

        # Print top consumers
        if len(self.checkpoints) > 1:
            print("\nTOP MEMORY CONSUMERS:")
            print("-"*80)

            # Calculate deltas between consecutive checkpoints
            consumers = []
            for i in range(1, len(self.checkpoints)):
                delta = self.checkpoints[i]['current_gb'] - self.checkpoints[i-1]['current_gb']
                if delta > 0:
                    consumers.append((self.checkpoints[i]['name'], delta))

            consumers.sort(key=lambda x: x[1], reverse=True)
            for name, delta in consumers[:5]:
                print(f"  {name:<40} +{delta:.3f} GB")
            print("="*80 + "\n")

    def get_report(self) -> str:
        """Generate a detailed text report."""
        if not self.enabled or not self.checkpoints:
            return "No profiling data available."

        report = []
        report.append("="*80)
        report.append("VRAM PROFILING DETAILED REPORT")
        report.append("="*80)
        report.append("")

        # Overall statistics
        max_peak = max(cp['peak_gb'] for cp in self.checkpoints)
        report.append(f"Maximum Peak Memory: {max_peak:.3f} GB")
        report.append("")

        # Detailed checkpoint information
        report.append("DETAILED CHECKPOINTS:")
        report.append("-"*80)
        for i, cp in enumerate(self.checkpoints):
            report.append(f"\n[{i+1}] {cp['name']}")
            report.append(f"    Current:        {cp['current_gb']:.3f} GB")
            report.append(f"    Interval Peak:  {cp['peak_since_last_gb']:.3f} GB")
            report.append(f"    Overall Peak:   {cp['peak_gb']:.3f} GB")
            if 'tensor_gb' in cp:
                report.append(f"    Tensor:         {cp['tensor_shape']} ({cp['tensor_dtype']}) = {cp['tensor_gb']:.3f} GB")
            if 'info' in cp:
                report.append(f"    Info:           {cp['info']}")

        report.append("\n" + "="*80)
        return "\n".join(report)

    def save_report(self, filepath: str):
        """Save profiling report to file."""
        if not self.enabled:
            return

        with open(filepath, 'w') as f:
            f.write(self.get_report())
        print(f"Profiling report saved to: {filepath}")


@contextmanager
def profile_memory(profiler: VRAMProfiler, name: str, tensor: Optional[torch.Tensor] = None):
    """Context manager for profiling a code block."""
    if profiler.enabled:
        profiler.checkpoint(f"{name} - START")

    try:
        yield
    finally:
        if profiler.enabled:
            if tensor is not None:
                profiler.checkpoint(f"{name} - END", tensor=tensor)
            else:
                profiler.checkpoint(f"{name} - END")


def profile_function(profiler: VRAMProfiler, name: Optional[str] = None):
    """Decorator for profiling a function."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = name or f"{func.__module__}.{func.__name__}"

            if profiler.enabled:
                profiler.checkpoint(f"{func_name} - START")

            result = func(*args, **kwargs)

            if profiler.enabled:
                if isinstance(result, torch.Tensor):
                    profiler.checkpoint(f"{func_name} - END", tensor=result)
                else:
                    profiler.checkpoint(f"{func_name} - END")

            return result
        return wrapper
    return decorator


# Global profiler instance
_global_profiler: Optional[VRAMProfiler] = None

def get_profiler() -> VRAMProfiler:
    """Get the global profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = VRAMProfiler(enabled=False)
    return _global_profiler

def enable_profiling():
    """Enable global profiling."""
    global _global_profiler
    _global_profiler = VRAMProfiler(enabled=True)
    _global_profiler.reset()
    return _global_profiler

def disable_profiling():
    """Disable global profiling."""
    global _global_profiler
    if _global_profiler is not None:
        _global_profiler.enabled = False
```

To use this profiler, first enable the profiler.

```python
from FastAvatar.utils.vram_profiler import enable_profiling
profiler = enable_profiling()
```

To create a checkpoint, first import and get the profiler, then call `checkpoint`.

```python
from FastAvatar.utils.vram_profiler import get_profiler
profiler = get_profiler()

profiler.checkpoint("Query Points Generated", tensor=query_points, tensor_info=f"Query Points")
```

In the end, to print and write the profile to a txt file.

```python
profiler.print_summary()
profiler.save_report(report_path)
```

This will create a txt VRAM profile report that looks like this.
![]( {{ '/assets/2025-10-13/CustomProfiler.png' | relative_url }})

# Model Size Profiler
Sometimes we need to calculate the number of parameters a model has and how much VRAM it takes. Therefore, I wrote this script to profile model size.

```python
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from collections import defaultdict

class ModelSizeProfiler:
    """Analyze model parameter sizes and memory usage."""

    def __init__(self):
        self.module_stats: Dict[str, Dict] = {}

    def profile_module(self, module: nn.Module, name: str = "Model") -> Dict:
        """
        Profile a PyTorch module to get parameter counts and memory usage.

        Args:
            module: PyTorch module to profile
            name: Name for the module

        Returns:
            Dictionary with statistics
        """
        stats = {
            'name': name,
            'total_params': 0,
            'trainable_params': 0,
            'frozen_params': 0,
            'param_memory_mb': 0.0,
            'buffer_memory_mb': 0.0,
            'total_memory_mb': 0.0,
            'submodules': {}
        }

        # Count parameters
        for param in module.parameters():
            num_params = param.numel()
            stats['total_params'] += num_params

            if param.requires_grad:
                stats['trainable_params'] += num_params
            else:
                stats['frozen_params'] += num_params

            # Memory in MB (assuming float32 = 4 bytes)
            param_size_mb = (num_params * param.element_size()) / (1024 ** 2)
            stats['param_memory_mb'] += param_size_mb

        # Count buffers
        for buffer in module.buffers():
            buffer_size_mb = (buffer.numel() * buffer.element_size()) / (1024 ** 2)
            stats['buffer_memory_mb'] += buffer_size_mb

        stats['total_memory_mb'] = stats['param_memory_mb'] + stats['buffer_memory_mb']

        return stats

    def print_model_summary(self, model: nn.Module, detailed: bool = False):
        """Print a formatted summary of model size."""
        print("\n" + "="*80)
        print("MODEL SIZE PROFILING SUMMARY")
        print("="*80)

        # Overall stats
        overall = self.profile_module(model, "Overall Model")
        print(f"\nOverall Statistics:")
        print(f"  Total Parameters:     {overall['total_params']:,}")
        print(f"  Trainable Parameters: {overall['trainable_params']:,}")
        print(f"  Frozen Parameters:    {overall['frozen_params']:,}")
        print(f"  Parameter Memory:     {overall['param_memory_mb']:.2f} MB")
        print(f"  Buffer Memory:        {overall['buffer_memory_mb']:.2f} MB")
        print(f"  Total Memory:         {overall['total_memory_mb']:.2f} MB")

        if detailed:
            # Detailed breakdown by top-level modules
            print("\n" + "-"*80)
            print("Top-Level Module Breakdown:")
            print(f"{'Module':<40} {'Params':>15} {'Memory (MB)':>15}")
            print("-"*80)

            module_stats = []
            for name, submodule in model.named_children():
                stats = self.profile_module(submodule, name)
                module_stats.append((name, stats))

            # Sort by memory usage
            module_stats.sort(key=lambda x: x[1]['total_memory_mb'], reverse=True)

            for name, stats in module_stats:
                print(f"{name:<40} {stats['total_params']:>15,} {stats['total_memory_mb']:>15.2f}")

        print("="*80 + "\n")

    def print_detailed_breakdown(self, model: nn.Module, top_n: int = 10):
        """Print detailed breakdown of largest modules."""
        print("\n" + "="*80)
        print(f"TOP {top_n} LARGEST MODULES")
        print("="*80)
        print(f"{'Module Path':<50} {'Params':>15} {'Memory (MB)':>15}")
        print("-"*80)

        # Get all submodules
        all_modules = []
        for name, submodule in model.named_modules():
            if name and len(list(submodule.children())) == 0:  # Only leaf modules
                stats = self.profile_module(submodule, name)
                all_modules.append((name, stats))

        # Sort by memory
        all_modules.sort(key=lambda x: x[1]['total_memory_mb'], reverse=True)

        # Print top N
        for name, stats in all_modules[:top_n]:
            print(f"{name:<50} {stats['total_params']:>15,} {stats['total_memory_mb']:>15.2f}")

        print("="*80 + "\n")


def profile_model_size(model: nn.Module, detailed: bool = True):
    """
    Quick helper to profile a model's size.

    Args:
        model: PyTorch model
        detailed: Whether to show detailed breakdown
    """
    profiler = ModelSizeProfiler()
    profiler.print_model_summary(model, detailed=detailed)

    if detailed:
        profiler.print_detailed_breakdown(model, top_n=15)

    return profiler
```

To use.

```python
from FastAvatar.utils.model_size_profiler import profile_model_size
profile_model_size(self.model, detailed=False)
```
 

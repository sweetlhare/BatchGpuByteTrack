"""
Profiler utility for tracking operation timing.

Supports both CPU and GPU operations with proper CUDA synchronization.
"""

import time
import torch
from contextlib import contextmanager
from collections import defaultdict
from typing import Dict, List, Optional
import numpy as np


class Profiler:
    """
    Simple profiler for tracking operation timing.

    Usage:
        profiler = Profiler(device='cuda')

        with profiler.profile("operation_name"):
            # ... code to profile
            pass

        # Get timing statistics
        stats = profiler.get_stats()
        profiler.print_summary()
    """

    def __init__(self, device: str = 'cuda', enabled: bool = True):
        """
        Args:
            device: 'cuda' or 'cpu' - determines synchronization behavior
            enabled: If False, profiling becomes no-op (zero overhead)
        """
        self.device = str(device)
        self.enabled = enabled
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self._sync_cuda = self.device.startswith('cuda') and torch.cuda.is_available()

    @contextmanager
    def profile(self, name: str):
        """
        Context manager for profiling a code block.

        Args:
            name: Name of the operation to profile
        """
        if not self.enabled:
            yield
            return

        # Synchronize GPU before timing (if using CUDA)
        if self._sync_cuda:
            torch.cuda.synchronize()

        start = time.perf_counter()
        try:
            yield
        finally:
            # Synchronize GPU after operation (if using CUDA)
            if self._sync_cuda:
                torch.cuda.synchronize()

            elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
            self.timings[name].append(elapsed)

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get timing statistics for all operations.

        Returns:
            Dict mapping operation name to statistics:
            - 'mean': Average time in ms
            - 'std': Standard deviation in ms
            - 'min': Minimum time in ms
            - 'max': Maximum time in ms
            - 'total': Total cumulative time in ms
            - 'count': Number of calls
        """
        stats = {}
        for name, times in self.timings.items():
            if len(times) > 0:
                times_arr = np.array(times)
                stats[name] = {
                    'mean': float(np.mean(times_arr)),
                    'std': float(np.std(times_arr)),
                    'min': float(np.min(times_arr)),
                    'max': float(np.max(times_arr)),
                    'total': float(np.sum(times_arr)),
                    'count': len(times)
                }
        return stats

    def print_summary(self, title: Optional[str] = None):
        """
        Print a formatted summary of all profiled operations.

        Args:
            title: Optional title for the summary
        """
        if title:
            print(f"\n{'=' * 70}")
            print(f"{title}")
            print(f"{'=' * 70}")
        else:
            print(f"\n{'=' * 70}")
            print(f"Profiler Summary")
            print(f"{'=' * 70}")

        stats = self.get_stats()

        if not stats:
            print("No operations profiled")
            return

        # Sort by mean time (descending)
        sorted_ops = sorted(stats.items(), key=lambda x: x[1]['mean'], reverse=True)

        # Print header
        print(f"{'Operation':<30} {'Mean (ms)':>10} {'Std (ms)':>10} {'Count':>8} {'Total (ms)':>12} {'%':>7}")
        print("-" * 70)

        # Print each operation
        total_time = sum(s['total'] for s in stats.values())
        for name, s in sorted_ops:
            pct = (s['total'] / total_time * 100) if total_time > 0 else 0
            print(f"{name:<30} {s['mean']:>10.2f} {s['std']:>10.2f} {s['count']:>8} {s['total']:>12.2f} ({pct:>5.1f}%)")

        print("-" * 70)
        print(f"{'TOTAL':<30} {'':<10} {'':<10} {'':<8} {total_time:>12.2f} (100.0%)")
        print()

    def reset(self):
        """Clear all timing data."""
        self.timings.clear()

    def get_mean_time(self, name: str) -> float:
        """
        Get mean time for a specific operation.

        Args:
            name: Operation name

        Returns:
            Mean time in ms, or 0.0 if operation not found
        """
        if name in self.timings and len(self.timings[name]) > 0:
            return float(np.mean(self.timings[name]))
        return 0.0

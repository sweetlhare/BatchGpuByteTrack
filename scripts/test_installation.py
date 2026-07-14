#!/usr/bin/env python3
"""
Test BatchGpuByteTrack installation and dependencies.

Verifies:
- Python version
- PyTorch (+ CUDA if present)
- Dependencies (numpy, cv2, scipy, lap)
- Package imports (bytetrack)
- Tracker initialization (CPU)
- OSNet model construction

CUDA/GPU and optional dependencies (ultralytics, gdown) are reported
informationally and do not fail the check — the package fully supports
CPU-only setups.

Usage:
    python scripts/test_installation.py
"""

import os
import sys

# Allow running from a source checkout without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def check_python():
    """Check Python version."""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    ok = version >= (3, 8)
    return ("Python", version_str, ok)


def check_pytorch():
    """Check PyTorch installation."""
    try:
        import torch
        return ("PyTorch", torch.__version__, True)
    except ImportError:
        return ("PyTorch", "NOT INSTALLED", False)


def info_cuda():
    """Report CUDA availability (informational)."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            return ("CUDA", f"{torch.version.cuda} ({gpu_name})", True)
        return ("CUDA", "not available (CPU mode)", True)
    except ImportError:
        return ("CUDA", "PyTorch not installed", True)


def check_dependency(package_name, import_name=None, required=True):
    """Check if a dependency is installed."""
    if import_name is None:
        import_name = package_name

    try:
        mod = __import__(import_name)
        ver = getattr(mod, '__version__', 'unknown')
        return (package_name, ver, True)
    except ImportError:
        label = "NOT INSTALLED" if required else "not installed (optional)"
        return (package_name, label, not required)


def check_bytetrack():
    """Check bytetrack package imports."""
    try:
        from bytetrack import BYTETracker, BatchGPUTracker, TrackerConfig  # noqa: F401
        return ("bytetrack", "OK", True)
    except ImportError as e:
        return ("bytetrack", str(e), False)


def check_tracker_update():
    """Test tracker initialization and one update step (CPU)."""
    try:
        import numpy as np
        from bytetrack import BYTETracker, BatchGPUTracker, TrackerConfig

        class Args:
            track_thresh = 0.5
            track_buffer = 30
            match_thresh = 0.8
            mot20 = False

        cpu_tracker = BYTETracker(Args(), frame_rate=30)
        cpu_tracker.update(
            np.array([[100, 100, 200, 200, 0.9]], dtype=np.float32),
            (720, 1280), (720, 1280)
        )

        gpu_tracker = BatchGPUTracker(num_streams=2, config=TrackerConfig(device='cpu'))
        gpu_tracker.update([
            np.array([[100, 100, 200, 200, 0.9]], dtype=np.float32),
            None
        ])

        return ("Tracker update", "BYTETracker + BatchGPUTracker OK", True)
    except Exception as e:
        return ("Tracker update", str(e), False)


def check_osnet():
    """Check OSNet model construction (no weight download)."""
    try:
        import torch
        from bytetrack.osnet import build_osnet

        model = build_osnet('osnet_x0_25', pretrained=False, device='cpu')
        with torch.no_grad():
            out = model(torch.randn(1, 3, 256, 128))
        assert out.shape == (1, 512)
        return ("OSNet", "osnet_x0_25 built, forward OK", True)
    except Exception as e:
        return ("OSNet", str(e), False)


def main():
    """Run all installation checks."""
    print("Testing BatchGpuByteTrack Installation")
    print("=" * 60)

    checks = [
        check_python(),
        check_pytorch(),
        info_cuda(),
        check_dependency('numpy'),
        check_dependency('opencv-python', 'cv2'),
        check_dependency('scipy'),
        check_dependency('lap'),
        check_dependency('ultralytics', required=False),
        check_dependency('gdown', required=False),
        check_bytetrack(),
        check_tracker_update(),
        check_osnet(),
    ]

    for name, value, ok in checks:
        status = "✓" if ok else "✗"
        print(f"{status} {name:<20} {value}")

    print("=" * 60)

    if all(c[2] for c in checks):
        print("\nAll checks passed! ✓")
        sys.exit(0)
    else:
        print("\nSome checks failed! ✗")
        print("\nTo install missing dependencies:")
        print("  pip install -r requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    main()

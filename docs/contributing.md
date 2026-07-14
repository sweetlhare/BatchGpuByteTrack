# Contributing Guide

Thank you for your interest in contributing to ByteTrack GPU! This guide will help you get started.

## About This Project

This project extends the original [ByteTrack](https://github.com/FoundationVision/ByteTrack) implementation with GPU acceleration and integrates [OSNet ReID models](https://github.com/KaiyangZhou/deep-person-reid) for robust tracking. We focus on production-ready multi-stream tracking with stability and performance optimizations.

## Development Setup

### 1. Fork and Clone

```bash
# Fork on GitHub first, then clone
git clone https://github.com/YOUR_USERNAME/BatchGpuByteTrack.git
cd BatchGpuByteTrack

# Add upstream remote
git remote add upstream https://github.com/sweetlhare/BatchGpuByteTrack.git
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install in Development Mode

```bash
pip install -e .
pip install -r requirements-dev.txt  # If exists
```

### 4. Verify Installation

```bash
python -c "from bytetrack import BatchGPUTracker; print('OK')"
python scripts/test_installation.py
```

## Development Workflow

### 1. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
```

Branch naming conventions:
- `feature/add-multi-gpu-support` - New features
- `fix/nan-in-kalman-filter` - Bug fixes
- `docs/improve-readme` - Documentation
- `perf/optimize-matching` - Performance improvements

### 2. Make Changes

Follow the [Code Style](#code-style) guidelines below.

### 3. Test Your Changes

```bash
# Run unit tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_tracker.py

# Run benchmarks
python tools/benchmark_endtoend.py --video test_videos/6387-191695740_medium.mp4 --streams 8
```

### 4. Commit Changes

Use clear, descriptive commit messages:

```bash
git add .
git commit -m "Add multi-GPU support for large-scale tracking

- Implement device assignment per stream
- Add GPU load balancing
- Update documentation with multi-GPU examples

Closes #123"
```

Commit message format:
- First line: Brief summary (50 chars or less)
- Blank line
- Detailed description (wrap at 72 chars)
- Reference issues: `Closes #123`, `Fixes #456`

### 5. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## Code Style

### Python Style

Follow [PEP 8](https://pep8.org/) with these specifics:

**Formatting**:
- 4 spaces for indentation (no tabs)
- 88 character line limit (Black formatter)
- Double quotes for strings
- Trailing commas in multi-line structures

**Use Black formatter**:
```bash
pip install black
black bytetrack/ tools/ examples/
```

**Use isort for imports**:
```bash
pip install isort
isort bytetrack/ tools/ examples/
```

### Type Hints

Add type hints to all functions:

```python
from typing import Any, Dict, List, Optional
import numpy as np

def update(
    detections: List[np.ndarray],
    frames: Optional[List[np.ndarray]] = None
) -> List[List[Dict[str, Any]]]:
    """Update tracker with new detections.

    Args:
        detections: List of [N, 5] detection arrays per stream
        frames: Optional list of BGR frames for ReID

    Returns:
        List of track dicts per stream
    """
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def compute_distance(tracks, detections, threshold=0.8):
    """Compute distance matrix between tracks and detections.

    Uses IoU distance for bounding box similarity.

    Args:
        tracks (torch.Tensor): Track bounding boxes (M, 4)
        detections (torch.Tensor): Detection bounding boxes (N, 4)
        threshold (float): IoU threshold for matching

    Returns:
        torch.Tensor: Distance matrix (M, N)

    Raises:
        ValueError: If input tensors have invalid shape

    Example:
        >>> tracks = torch.tensor([[100, 100, 200, 200]])
        >>> detections = torch.tensor([[105, 105, 205, 205]])
        >>> dist = compute_distance(tracks, detections)
        >>> print(dist.shape)
        torch.Size([1, 1])
    """
    ...
```

### Comments

**Good comments** explain *why*, not *what*:

```python
# ✅ Good: Explains reasoning
# Use lower threshold for occluded objects to reduce ID switches
if track.is_occluded:
    threshold = 0.7

# ❌ Bad: Obvious from code
# Set threshold to 0.7
threshold = 0.7
```

**Non-obvious constraints need comments**:

```python
# Covariance regularization: prevents overflow on long-lost tracks,
# which would otherwise propagate NaN through the cost matrix
diag.clamp_(min=MIN_EIGENVALUE, max=MAX_COVARIANCE)
```

## Testing

### Unit Tests

Create tests in `tests/` directory:

```python
# tests/test_tracker.py
import numpy as np
import pytest
from bytetrack import BatchGPUTracker, TrackerConfig


def test_basic_tracking():
    """Test basic tracking functionality."""
    config = TrackerConfig(device='cpu')
    tracker = BatchGPUTracker(num_streams=2, config=config)

    # Create dummy detections
    dets = np.array([[100, 100, 200, 200, 0.9]], dtype=np.float32)
    all_tracks = tracker.update([dets, dets])

    assert len(all_tracks) == 2
    assert len(all_tracks[0]) > 0  # frame 1: immediate activation


def test_nan_handling():
    """NaN detections must not crash the tracker."""
    config = TrackerConfig(device='cpu')
    tracker = BatchGPUTracker(num_streams=1, config=config)

    dets = np.array([[100, 100, 200, 200, np.nan]], dtype=np.float32)
    all_tracks = tracker.update([dets])
    assert len(all_tracks) == 1


@pytest.mark.parametrize("num_streams", [1, 4, 8, 16, 32])
def test_scaling(num_streams):
    """Tracker must scale to different stream counts."""
    config = TrackerConfig(device='cpu')
    tracker = BatchGPUTracker(num_streams=num_streams, config=config)

    dets = np.array([[100, 100, 200, 200, 0.9]], dtype=np.float32)
    all_tracks = tracker.update([dets for _ in range(num_streams)])
    assert len(all_tracks) == num_streams
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=bytetrack --cov-report=html

# Run specific test
pytest "tests/test_tracker.py::TestTrackLifecycle::test_id_persistence"

# Run in verbose mode
pytest tests/ -v
```

### Benchmarks

Add benchmarks for performance-critical changes:

```python
# tools/benchmark_my_feature.py
import time
import numpy as np
from bytetrack import BatchGPUTracker, TrackerConfig


def benchmark_my_feature():
    config = TrackerConfig(device='cuda')
    tracker = BatchGPUTracker(num_streams=8, config=config)

    # Prepare test data
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 1000, size=(30, 2)).astype(np.float32)
    dets = np.hstack([xy, xy + 60, rng.uniform(0.3, 1, size=(30, 1)).astype(np.float32)])
    all_dets = [dets for _ in range(8)]

    # Warm up
    for _ in range(10):
        tracker.update(all_dets)

    # Benchmark
    times = []
    for _ in range(100):
        start = time.time()
        tracker.update(all_dets)
        times.append(time.time() - start)

    avg_time = sum(times) / len(times) * 1000  # ms
    print(f"Average time: {avg_time:.2f}ms")
    assert avg_time < 20.0, f"Too slow: {avg_time:.2f}ms"


if __name__ == '__main__':
    benchmark_my_feature()
```

## Documentation

### Code Documentation

- Add docstrings to all public functions/classes
- Update docstrings when changing function signatures
- Include examples in docstrings

### User Documentation

When adding features, update relevant docs:

- `README.md` - Main documentation
- `docs/quick_start.md` - Basic usage examples
- `docs/api_reference.md` - API documentation
- `docs/benchmarks.md` - Performance data

### Examples

Add examples to `examples/` for significant features:

```python
# examples/my_feature_example.py
"""
Example: Using My New Feature

This example demonstrates how to use my new feature.
"""

from bytetrack import BatchGPUTracker, TrackerConfig

# Setup
config = TrackerConfig(device='cuda')
tracker = BatchGPUTracker(num_streams=4, config=config)

# Your example code here...
```

## Pull Request Guidelines

### Before Submitting

- ✅ Code follows style guidelines (Black + isort)
- ✅ All tests pass
- ✅ New tests added for new functionality
- ✅ Documentation updated
- ✅ Commit messages are clear
- ✅ Branch is up to date with main

### PR Description Template

```markdown
## Description

Brief description of changes.

## Motivation

Why is this change needed? What problem does it solve?

## Changes

- List of specific changes
- Another change
- Yet another change

## Testing

How was this tested?

- Unit tests added: `test_my_feature.py`
- Benchmark results: 15% faster
- Tested on: RTX 3090, CUDA 11.3, PyTorch 1.12

## Screenshots (if applicable)

Add screenshots for UI changes.

## Checklist

- [ ] Code follows style guidelines
- [ ] Tests added and passing
- [ ] Documentation updated
- [ ] Benchmarks show improvement (for performance changes)
- [ ] No breaking changes (or documented)

## Related Issues

Closes #123
Related to #456
```

### Review Process

1. Maintainers will review your PR
2. Address feedback and push updates
3. Once approved, PR will be merged

## Areas for Contribution

### High Priority

- [ ] Multi-GPU support for large-scale tracking
- [ ] TensorRT INT8 quantization for ReID models
- [ ] Fused CUDA kernels for distance computation
- [ ] Sparse Hungarian algorithm for faster matching
- [ ] Online learning for ReID embeddings

### Medium Priority

- [ ] Support for additional ReID models
- [ ] 3D tracking (depth estimation)
- [ ] Trajectory prediction
- [ ] Track smoothing and interpolation
- [ ] Distributed tracking across multiple machines

### Documentation

- [ ] More examples (custom detectors, different scenarios)
- [ ] Video tutorials
- [ ] Performance optimization guide expansions
- [ ] Comparison with other trackers (MOT benchmarks)

### Testing

- [ ] Integration tests with real videos
- [ ] Edge case testing (empty frames, single object, etc.)
- [ ] Stress testing (1000+ streams)
- [ ] Memory leak detection

## Code Review Standards

When reviewing PRs, check:

1. **Correctness**: Does it work as intended?
2. **Testing**: Adequate test coverage?
3. **Performance**: No regressions?
4. **Readability**: Clear and maintainable?
5. **Documentation**: Well documented?
6. **Style**: Follows guidelines?

## Release Process

We use semantic versioning: `MAJOR.MINOR.PATCH`

- **MAJOR**: Breaking changes
- **MINOR**: New features (backwards compatible)
- **PATCH**: Bug fixes

## Communication

- **GitHub Issues**: Bug reports, feature requests
- **Pull Requests**: Code contributions
- **Discussions**: General questions, ideas

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors will be:
- Listed in the release notes
- Acknowledged in release notes
- Credited in relevant documentation

## Questions?

Feel free to:
- Open an issue for questions
- Start a discussion
- Reach out to maintainers

Thank you for contributing to ByteTrack GPU! 🚀

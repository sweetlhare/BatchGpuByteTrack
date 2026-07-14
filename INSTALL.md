# Installation Guide

Quick guide to get ByteTrack GPU up and running.

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/sweetlhare/BatchGpuByteTrack.git
cd BatchGpuByteTrack
```

### 2. Create Virtual Environment

**Using uv (recommended)** - [docs](https://docs.astral.sh/uv/)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and activate
uv venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows
```

**Using Python venv:**

```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows
```

**Using conda:**

```bash
conda create -n bytetrack python=3.9
conda activate bytetrack
```

### 3. Install Dependencies

**With uv (faster):**

```bash
uv pip install -r requirements.txt
```

**With pip:**

```bash
pip install -r requirements.txt
```

**Dependencies** (`requirements.txt`):
- `torch>=1.13` - PyTorch (install the CUDA build for GPU tracking)
- `torchvision` - Vision utilities (fast IoU)
- `numpy` - Numerical operations
- `opencv-python` - Image processing
- `scipy` - Fallback linear assignment solver
- `lap` - Linear assignment problem solver
- `ultralytics` - YOLO detector (for examples/benchmarks only)
- `gdown` - Fallback source for OSNet weights (optional)
- `pytest` - Tests

### 4. Install PyTorch with CUDA (if needed)

See [PyTorch website](https://pytorch.org/get-started/locally/) for your CUDA version.

**CUDA 11.8:**
```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**CUDA 12.1:**
```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 5. Verify Installation

```bash
python scripts/test_installation.py
```

Expected output (versions will differ):
```
✓ Python               3.12.3
✓ PyTorch              2.x.x
✓ CUDA                 12.x (NVIDIA GeForce RTX 4090)
✓ numpy                ...
✓ opencv-python        ...
✓ scipy                ...
✓ lap                  ...
✓ bytetrack            OK
✓ Tracker update       BYTETracker + BatchGPUTracker OK
✓ OSNet                osnet_x0_25 built, forward OK

All checks passed! ✓
```

The check passes on CPU-only machines too — CUDA is reported informationally.

## Running Examples

All examples can run without package installation (they add the path automatically):

```bash
# CPU tracking (uses the bundled test video by default)
python examples/basic_cpu_tracking.py

# GPU single stream
python examples/gpu_single_stream.py --video your_video.mp4

# GPU multi-stream (8 streams)
python examples/gpu_multistream.py --video your_video.mp4 --streams 8

# GPU with ReID
python examples/gpu_with_reid.py --video your_video.mp4 --streams 4
```

## Troubleshooting

### ModuleNotFoundError: No module named 'bytetrack'

**Solution 1**: Install package in development mode:
```bash
pip install -e .
```

**Solution 2**: Examples already add the path automatically (run from project root):
```bash
cd /path/to/BatchGpuByteTrack
python examples/basic_cpu_tracking.py --video video.mp4
```

### ModuleNotFoundError: No module named 'lap'

**Solution**: Install lap package:
```bash
pip install lap
```

If build fails, install build dependencies:
```bash
# Ubuntu/Debian
sudo apt-get install python3-dev

# macOS
brew install python

# Then try again
pip install lap
```

### CUDA not available

**Check PyTorch CUDA**:
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

**If False**, reinstall PyTorch with CUDA:
```bash
# Check your CUDA version first
nvcc --version

# Install matching PyTorch version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### ImportError: cannot import name 'bbox_overlaps'

This is normal - we replaced `cython_bbox` with pure Python IoU computation. If you see this error, make sure you're using the latest version from GitHub.

## Development Setup

For contributing or development:

```bash
# Install with dev dependencies (pytest)
pip install -e ".[dev]"

# Run tests
pytest tests/
```

## System Requirements

**Minimum**:
- Python 3.8+
- 4GB RAM
- NVIDIA GPU with CUDA support (optional — CPU mode is fully supported)

**Recommended**:
- Python 3.10+
- 16GB+ RAM
- NVIDIA GPU with 8GB+ VRAM (for multi-stream tracking)
- CUDA 11.8+

## Next Steps

- Read [Quick Start Guide](docs/quick_start.md)
- Try [Examples](examples/)
- Check [API Reference](docs/api_reference.md)
- Run [Benchmarks](scripts/run_benchmarks.sh)

## Getting Help

If you encounter issues:

1. Check [Troubleshooting Guide](docs/troubleshooting.md)
2. Search [GitHub Issues](https://github.com/sweetlhare/BatchGpuByteTrack/issues)
3. Open a new issue with:
   - Python version (`python --version`)
   - PyTorch version (`python -c "import torch; print(torch.__version__)"`)
   - CUDA version (`nvcc --version`)
   - Full error traceback
   - Steps to reproduce

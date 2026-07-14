# Quick Setup for Server (Ubuntu + CUDA)

**For**: Ubuntu server with NVIDIA GPU (CUDA 11.8 / 12.x)

## One-Line Setup

```bash
git clone https://github.com/sweetlhare/BatchGpuByteTrack.git && \
cd BatchGpuByteTrack && \
curl -LsSf https://astral.sh/uv/install.sh | sh && \
source ~/.bashrc && \
uv venv && \
source .venv/bin/activate && \
uv pip install -r requirements.txt && \
python -c "import torch; print(f'✓ PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

The default PyPI `torch` wheels for Linux already ship with CUDA support.
If you need a specific CUDA build, install it explicitly afterwards:

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Step-by-Step

```bash
# 1. Clone and enter the project
git clone https://github.com/sweetlhare/BatchGpuByteTrack.git
cd BatchGpuByteTrack

# 2. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # or ~/.zshrc

# 3. Create virtual environment
uv venv

# 4. Activate
source .venv/bin/activate

# 5. Install dependencies
uv pip install -r requirements.txt

# 6. Verify installation
python scripts/test_installation.py
python -m pytest tests/ -q
```

## Quick Test

```bash
# Run the ReID benchmark on the bundled test video
python tools/benchmark_reid.py --video test_videos/6387-191695740_medium.mp4 --num-frames 200 --streams 8 --device cuda
```

## Expected Output

```
Using GPU: <your GPU name>
...
[BatchGPU x8 - NO ReID]
  Total time: ..., XX.XXms/frame, XX.X FPS

  BatchGPU Profiling - NO ReID (8 streams)
  8_state_update_seq      ...
  6_hungarian_parallel    ...
  4_compute_iou_cost      ...
```

Reference numbers for an RTX 4090 are in [docs/benchmarks.md](docs/benchmarks.md).

## Troubleshooting

### uv command not found

```bash
# Add to PATH manually
export PATH="$HOME/.local/bin:$PATH"
source ~/.bashrc
```

### CUDA version mismatch

```bash
# Check CUDA version
nvidia-smi

# For CUDA 11.8
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1+
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Old venv exists

```bash
# Remove old venv
rm -rf .venv

# Create fresh one
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Next Steps

After setup:
1. Run the benchmarks to get reference numbers for your GPU
2. Test examples: `python examples/gpu_multistream.py --video test_videos/6387-191695740_medium.mp4 --streams 8 --no-display`
3. Check GPU usage: `watch -n 1 nvidia-smi`

## Full Documentation

See [INSTALL.md](INSTALL.md) for complete installation guide.

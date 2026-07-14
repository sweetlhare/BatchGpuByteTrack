#!/bin/bash
# Pre-download model weights for ByteTrack GPU tracker.
#
# OSNet weights go to $TORCH_HOME/checkpoints (~/.cache/torch/checkpoints by
# default) — the exact location bytetrack.osnet loads them from, so the first
# ReID run won't need network access.

set -eo pipefail

RELEASE_URL="https://github.com/sweetlhare/BatchGpuByteTrack/releases/download/weights-v1.0"
CACHE_DIR="${TORCH_HOME:-$HOME/.cache/torch}/checkpoints"

echo "======================================"
echo "Downloading Model Weights"
echo "======================================"
echo "Cache dir: $CACHE_DIR"
echo ""

mkdir -p "$CACHE_DIR"

MODELS=(osnet_x1_0 osnet_x0_75 osnet_x0_5 osnet_x0_25)
i=1
for model in "${MODELS[@]}"; do
    echo "[$i/5] Downloading ${model}..."
    wget -nc -q --show-progress -P "$CACHE_DIR" \
        "$RELEASE_URL/${model}_imagenet.pth" \
        || echo "Warning: failed to download ${model}, skipping..."
    i=$((i + 1))
done

# Download YOLO model (optional, for examples/benchmarks)
echo "[5/5] Downloading YOLO26s (optional, needs ultralytics)..."
python -c "from ultralytics import YOLO; YOLO('yolo26s.pt')" \
    || echo "Warning: failed to download yolo26s (pip install ultralytics), skipping..."

echo ""
echo "======================================"
echo "Download Complete!"
echo "======================================"
echo ""
echo "OSNet weights:"
ls -lh "$CACHE_DIR"/osnet_*.pth 2>/dev/null || echo "  No .pth files found"
ls -lh yolo26s.pt 2>/dev/null || echo "  yolo26s.pt not found (optional)"
echo ""
echo "You can now run the examples:"
echo "  python examples/gpu_with_reid.py --video test_videos/6387-191695740_medium.mp4"
echo ""

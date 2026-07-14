#!/bin/bash
# Run all benchmarks for ByteTrack GPU tracker

# pipefail: a failing benchmark must fail the script even through `| tee`
set -eo pipefail

echo "======================================"
echo "ByteTrack GPU Benchmarks"
echo "======================================"
echo ""

# Check if video file exists
VIDEO="${1:-video.mp4}"

if [ ! -f "$VIDEO" ]; then
    echo "Error: Video file not found: $VIDEO"
    echo ""
    echo "Usage: $0 [video_file]"
    echo "  video_file: Path to test video (default: video.mp4)"
    echo ""
    echo "Example:"
    echo "  $0 my_video.mp4"
    exit 1
fi

echo "Using video: $VIDEO"
echo ""

# Create output directory
OUTPUT_DIR="benchmark_results"
mkdir -p $OUTPUT_DIR

# Create full log file with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FULL_LOG="$OUTPUT_DIR/benchmark_full_${TIMESTAMP}.log"

# Start logging
echo "Full benchmark log: $FULL_LOG"
echo ""

# Function to log with timestamp
log_and_run() {
    echo "[$( date +"%H:%M:%S" )] $1" | tee -a "$FULL_LOG"
    shift
    "$@" 2>&1 | tee -a "$FULL_LOG"
}

# Benchmark 1: CPU vs GPU comparison
{
echo "======================================"
echo "[1/3] CPU vs GPU Comparison"
echo "======================================"
echo ""
echo "Testing 1, 2, 4, 8, 16, 32, and 64 streams..."
} | tee -a "$FULL_LOG"

python tools/benchmark_comparison.py \
    --video "$VIDEO" \
    --num-frames 200 \
    2>&1 | tee -a "$FULL_LOG" | tee $OUTPUT_DIR/cpu_vs_gpu.txt

{
echo ""
echo "Results saved to: $OUTPUT_DIR/cpu_vs_gpu.txt"
echo ""
} | tee -a "$FULL_LOG"

# Benchmark 2: ReID performance
{
echo "======================================"
echo "[2/3] ReID Performance"
echo "======================================"
echo ""
echo "Testing ReID performance on 1, 2, 4, 8, 16, 32, and 64 streams..."
} | tee -a "$FULL_LOG"

python tools/benchmark_reid.py \
    --video "$VIDEO" \
    --num-frames 200 \
    --device cuda \
    2>&1 | tee -a "$FULL_LOG" | tee $OUTPUT_DIR/reid_performance.txt

{
echo ""
echo "Results saved to: $OUTPUT_DIR/reid_performance.txt"
echo ""
} | tee -a "$FULL_LOG"

# Benchmark 3: End-to-end comparison
{
echo "======================================"
echo "[3/3] End-to-End Comparison"
echo "======================================"
echo ""
echo "Testing end-to-end on 1, 2, 4, 8, 16, 32, and 64 streams..."
} | tee -a "$FULL_LOG"

python tools/benchmark_endtoend.py \
    --video "$VIDEO" \
    --num-frames 200 \
    2>&1 | tee -a "$FULL_LOG" | tee $OUTPUT_DIR/endtoend.txt

{
echo ""
echo "Results saved to: $OUTPUT_DIR/endtoend.txt"
echo ""
} | tee -a "$FULL_LOG"

# Summary
{
echo "======================================"
echo "Benchmarks Complete!"
echo "======================================"
echo ""
echo "Results directory: $OUTPUT_DIR/"
ls -lh $OUTPUT_DIR/
echo ""
echo "Results files:"
echo "  - Full log: $FULL_LOG"
echo "  - CPU vs GPU: $OUTPUT_DIR/cpu_vs_gpu.txt"
echo "  - ReID overhead: $OUTPUT_DIR/reid_performance.txt"
echo "  - End-to-end: $OUTPUT_DIR/endtoend.txt"
echo ""
echo "Key findings:"
echo "1. Check CPU vs GPU speedup: $OUTPUT_DIR/cpu_vs_gpu.txt"
echo "2. Check ReID overhead: $OUTPUT_DIR/reid_performance.txt"
echo "3. Check end-to-end performance: $OUTPUT_DIR/endtoend.txt"
echo "4. Full log with all details: $FULL_LOG"
echo ""
} | tee -a "$FULL_LOG"

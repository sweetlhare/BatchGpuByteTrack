#!/bin/bash
# Run scripts without installing package

# Set PYTHONPATH to current directory
export PYTHONPATH="${PYTHONPATH}:$(dirname "$0")"

# Check if command provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <python_script> [arguments...]"
    echo ""
    echo "Examples:"
    echo "  $0 scripts/test_installation.py"
    echo "  $0 tools/benchmark_endtoend.py --video video.mp4 --num-frames 200 --num-streams 8"
    echo "  $0 tools/benchmark_reid.py --video video.mp4 --num-frames 200 --streams 8"
    exit 1
fi

# Run python script with arguments
python "$@"

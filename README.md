# BatchGpuByteTrack: GPU-Accelerated Multi-Stream Object Tracking

**High-performance multi-object tracking with GPU batching and Re-Identification**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.13+](https://img.shields.io/badge/pytorch-1.13+-red.svg)](https://pytorch.org/)

> Based on [ByteTrack](https://github.com/ifzhang/ByteTrack) with GPU acceleration, batched Kalman filtering, and production stability fixes.

---

## 🎯 Why This Project?

**Simple, modular, production-ready** tracking solution:

- ✅ **Detector Agnostic** - Works with YOLO, Faster R-CNN, or any detector
- ✅ **Flexible ReID** - Appearance features fused into association, swappable OSNet models
- ✅ **GPU Optimized** - **6x+ faster** than CPU tracking on multi-stream workloads
- ✅ **Production Stable** - Tested on 64 concurrent streams with numerical safeguards
- ✅ **Faithful ByteTrack** - Full three-stage association (incl. low-confidence BYTE recovery)
- ✅ **Tested** - Pytest suite covering matching, Kalman math, and track lifecycle

---

## ⚡ Performance

**Test Setup**: NVIDIA RTX 4090, YOLO26s detector, 1280x720 pedestrian video, 200 frames

### End-to-End (YOLO + Tracking, 8 streams)

| Configuration    | YOLO (ms) | Tracking (ms) | Total (ms) | FPS  |
|------------------|-----------|---------------|------------|------|
| CPU, no ReID     | 5.8       | 31.8          | 37.6       | 26.6 |
| **GPU, no ReID** | 5.8       | **5.1**       | **11.0**   | **91.3** |
| CPU + ReID x0.5  | 5.8       | 59.8          | 65.6       | 15.2 |
| GPU + ReID x0.5  | 5.8       | 33.1          | 38.9       | 25.7 |

GPU tracking is **6.2x faster** at 8 streams (5.6x at 16); with ReID
enabled — **1.8x faster** (6.3x at 16 streams).

### GPU vs CPU Scaling (tracking only, dense synthetic load: 30 dets/frame)

| Streams | CPU (ms/frame) | GPU (ms/frame) | GPU Speedup |
|---------|----------------|----------------|-------------|
| 1       | 1.2            | 2.6            | 0.5x ❌     |
| 4       | 9.6            | 5.1            | **1.9x** ✓  |
| 8       | 47.2           | 8.2            | **5.8x** ✓✓ |
| 16      | 104.1          | 14.5           | **7.2x** ✓✓ |
| 32      | 216.2          | 26.3           | **8.2x** ✓✓✓|
| 64      | 441.7          | 51.4           | **8.6x** ✓✓✓|

**Why does the speedup grow with load?** The GPU cost per frame is dominated
by a fixed number of batched operations whose size grows gently with the
total track count, while the CPU tracker pays per-stream, per-track Python
and numpy costs that grow super-linearly on dense scenes (random boxes churn
tracks, inflating the lost-track pools and cost matrices every frame).
Single-stream tracking stays faster on CPU — the GPU's per-frame kernel
launch and transfer overhead only pays off once several streams share it.
Note: multi-threaded CPU baselines are sensitive to host load; reproduce on
your hardware with `tools/benchmark_comparison.py` (numbers above were
reproduced within ~5% across repeated runs).

**Recommendation**: CPU for 1-2 streams, GPU for 4+ streams. With ReID the
GPU wins from ~8 streams (at 4 sparse streams CPU and GPU ReID are
comparable); batch OSNet inference runs in FP16 by default.

### Full Benchmark Results

See [docs/benchmarks.md](docs/benchmarks.md) for methodology, ReID overhead
analysis, profiling breakdowns and memory usage.

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/sweetlhare/BatchGpuByteTrack.git
cd BatchGpuByteTrack

# Install uv (fast package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv
source .venv/bin/activate  # Linux/Mac

# Install dependencies (Linux PyPI torch wheels already include CUDA)
uv pip install -r requirements.txt

# Optional: pin a specific CUDA build of PyTorch
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Alternative**: See [INSTALL.md](INSTALL.md) for pip/conda installation.

### Verify

```bash
python scripts/test_installation.py   # environment check
python -m pytest tests/ -q            # test suite (GPU tests auto-skip without CUDA)
```

### Basic Usage

**CPU Tracking** (single stream):
```python
from bytetrack import BYTETracker
import numpy as np

class Args:
    track_thresh = 0.6
    track_buffer = 30
    match_thresh = 0.8
    mot20 = False

tracker = BYTETracker(Args(), frame_rate=30)

# detections: numpy array [N, 5] - [x1, y1, x2, y2, score]
detections = np.array([[100, 100, 200, 200, 0.9]])
img_info = (720, 1280)  # height, width
img_size = (720, 1280)

tracks = tracker.update(detections, img_info, img_size)

for track in tracks:
    print(f"Track {track.track_id}: bbox={track.tlbr}, score={track.score}")
```

**GPU Batch Tracking** (multi-stream):
```python
from bytetrack import BatchGPUTracker, TrackerConfig

config = TrackerConfig(
    track_thresh=0.6,
    track_buffer=30,
    match_thresh=0.8,
    device='cuda'
)

tracker = BatchGPUTracker(num_streams=8, config=config)

# Batch detections: list of numpy arrays
batch_detections = [
    np.array([[100, 100, 200, 200, 0.9]]),  # Stream 0
    np.array([[150, 150, 250, 250, 0.8]]),  # Stream 1
    # ... 6 more streams
]

all_tracks = tracker.update(batch_detections)

for stream_id, tracks in enumerate(all_tracks):
    for track in tracks:
        print(f"Stream {stream_id}, Track {track['track_id']}: {track['tlbr']}")
```

**GPU + ReID** (robust cross-camera tracking):
```python
config = TrackerConfig(
    track_thresh=0.6,
    track_buffer=30,
    match_thresh=0.8,
    enable_reid=True,
    reid_model_path='osnet_x0_5',  # x0.5, x0.75, x1.0
    reid_threshold=0.5,
    lambda_emb=0.3,  # ReID weight (0-1)
    device='cuda'
)

tracker = BatchGPUTracker(num_streams=4, config=config)
# ... same usage as above
```

---

## 📦 Examples

All examples work without package installation:

```bash
# CPU tracking (single stream)
python examples/basic_cpu_tracking.py --video video.mp4

# GPU single stream
python examples/gpu_single_stream.py --video video.mp4

# GPU multi-stream (8 streams)
python examples/gpu_multistream.py --video video.mp4 --streams 8

# GPU with ReID (cross-camera tracking)
python examples/gpu_with_reid.py --video video.mp4 --streams 4
```

---

## 🔧 Configuration

### TrackerConfig Parameters

```python
TrackerConfig(
    # Core tracking
    track_thresh=0.5,      # Detection confidence threshold (default 0.5)
    track_buffer=30,       # Frames to keep lost tracks
    match_thresh=0.8,      # IoU threshold for matching

    # ReID (optional)
    enable_reid=False,     # Enable appearance features
    reid_model_path='osnet_x0_5',  # Model: x0.25, x0.5, x0.75, x1.0, ibn_x1.0
    reid_checkpoint=None,  # Optional local .pth file (offline setups)
    reid_threshold=0.5,    # Max appearance distance for fusion
    lambda_emb=0.3,        # ReID weight in fused cost (0..1)
    reid_fp16=True,        # FP16 OSNet inference on CUDA (~2x faster)
    reid_interval=1,       # Extract features every K-th frame (1 = every frame)

    # Performance
    device='cuda',         # 'cuda' or 'cpu'
    num_threads=8,         # CPU threads for parallel Hungarian
)
```

### ReID Models Comparison

| Model | Params | Feature Dim | Relative speed | Accuracy |
|-------|--------|-------------|----------------|----------|
| x0.25 | 0.2M   | 512         | fastest        | Lower    |
| **x0.5** | **0.6M** | **512** | **~2x faster than x1.0** | **Good** ✓ |
| x0.75 | 1.3M   | 512         | moderate       | High     |
| x1.0  | 2.2M   | 512         | baseline       | Highest  |

**Recommended**: OSNet **x0.5** (best speed/accuracy balance). Weights download
automatically on first use (sha256-verified); for offline setups pre-download
with `scripts/download_weights.sh` or pass `reid_checkpoint=`.

---

## 📊 Benchmarking

Run benchmarks yourself:

```bash
# End-to-end (YOLO + Tracking)
python tools/benchmark_endtoend.py --video video.mp4 --streams 1,4,8,16

# ReID performance
python tools/benchmark_reid.py --video video.mp4 --streams 8

# CPU vs GPU comparison
python tools/benchmark_comparison.py --video video.mp4 --streams 8,16,32
```

---

## 📖 Documentation

- [Installation Guide](INSTALL.md) - Detailed setup instructions
- [Quick Setup](QUICK_SETUP.md) - Server one-line setup
- [API Reference](docs/api_reference.md) - Complete API documentation
- [Benchmarks](docs/benchmarks.md) - Performance analysis
- [ReID Guide](docs/reid_guide.md) - Re-identification setup
- [Troubleshooting](docs/troubleshooting.md) - Common issues

---

## 🏗️ Architecture

**Key Components**:

1. **BYTETracker** (CPU) - Single-stream tracker with IoU + Kalman filtering
2. **BatchGPUTracker** (GPU) - Multi-stream batched tracker
3. **GPUKalmanFilter** - Batched Kalman predict/update on GPU
4. **OSNet** - Appearance-based re-identification
5. **Hungarian Algorithm** - Optimal assignment (lap.lapjv)

### Association logic (same in both trackers)

Both trackers run the original three-stage ByteTrack association per stream:

```
detections --+-- high-conf (score > track_thresh) --> Stage 1: match vs tracked + lost
             |                                        tracks (IoU*score, + ReID if enabled)
             +-- low-conf (0.1 < score <= thresh) --> Stage 2 "BYTE": match remaining
             |                                        tracked tracks on raw IoU
             +-- leftover high-conf ----------------> Stage 3: confirm unconfirmed tracks,
                                                      init new ones (score >= thresh+0.1)
```

### Original ByteTrack: one serial CPU pipeline per stream

```
Stream 1 --> KF predict -> IoU cost -> Hungarian -> state update --> tracks 1
Stream 2 --> KF predict -> IoU cost -> Hungarian -> state update --> tracks 2
   ...        (every step is per-stream numpy on CPU; N streams =
Stream N       N independent pipelines competing for CPU cores)  --> tracks N
```

### BatchGpuByteTrack: the math of all streams collapses into GPU batches

```
Streams 1..N: detections (numpy [N_i, 5])
      |
      | gather Kalman states of ALL tracks from ALL streams
      v
+------------------- GPU: one batch for all streams --------------------+
| batched Kalman predict [sum(M), 8] --> per-stream IoU cost matrices   |
| (optional: OSNet embeddings for all detection crops of all streams)   |
+-----------------------------------------------------------------------+
      |
      v  one bulk GPU->CPU download (all cost matrices)
+--------------- CPU thread pool: streams in parallel ------------------+
| 3-stage association per stream (Hungarian via lap.lapjv,              |
| score/ReID fusion on the small per-stream matrices)                   |
+-----------------------------------------------------------------------+
      |
      v  one bulk CPU->GPU upload (matched pairs of ALL streams)
+------------- GPU: batched Kalman update for matched pairs ------------+
+-----------------------------------------------------------------------+
      |
      v  one bulk download of updated states
per-stream track state update (CPU) --> tracks 1..N (list of dicts)
```

The speedup comes from replacing N×(many tiny numpy ops) with a few large
GPU batches. Association and track bookkeeping intentionally stay on the CPU:
Hungarian solvers on tiny per-stream matrices (tens×tens) are faster there
than any GPU alternative, and all CPU↔GPU traffic is a **fixed number of bulk
copies per frame** — batched across streams, never per-stream or per-track —
so transfer overhead does not grow with stream count.

---

## 🤝 Contributing

Contributions welcome! See [docs/contributing.md](docs/contributing.md).

Key areas:
- Support for new detectors (Faster R-CNN, DETR, etc.)
- Alternative ReID models (ResNet, ViT)
- Optimization improvements
- Additional benchmarks

---

## 📄 License

MIT License - see [LICENSE](LICENSE).

---

## 🙏 Credits

- **ByteTrack**: [ifzhang/ByteTrack](https://github.com/ifzhang/ByteTrack) - Original tracking algorithm
- **Deep Person ReID**: [KaiyangZhou/deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid) - OSNet models
- **LAP**: [gatagat/lap](https://github.com/gatagat/lap) - Linear assignment solver

See [CREDITS.md](CREDITS.md) for full acknowledgments.

---

## 📬 Contact

Issues: [GitHub Issues](https://github.com/sweetlhare/BatchGpuByteTrack/issues)

For questions about:
- Installation → [INSTALL.md](INSTALL.md)
- Usage → [docs/quick_start.md](docs/quick_start.md)
- Performance → [docs/benchmarks.md](docs/benchmarks.md)
- Bugs → [Open an issue](https://github.com/sweetlhare/BatchGpuByteTrack/issues/new)

---

**Star ⭐ this repo if you find it useful!**

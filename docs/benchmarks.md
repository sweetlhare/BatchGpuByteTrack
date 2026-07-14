# Performance Benchmarks

**Last Updated**: July 14, 2026 (after the batching optimization pass:
cross-stream batched Kalman update, padded batched IoU, `roi_align` crop
extraction with FP16 OSNet — GPU frame time at 16 streams dropped from
31.6 to 14.5 ms, ReID extraction from 153 to 66 ms)

Performance analysis of BatchGpuByteTrack. All numbers below were measured on
this exact codebase with the bundled scripts — you can reproduce every table
(see [Running Benchmarks Yourself](#running-benchmarks-yourself)).

---

## Test Setup

**Hardware**:
- GPU: NVIDIA GeForce RTX 4090 (24GB)
- CUDA: 13.0, PyTorch 2.13

**Workloads**:
- *Synthetic*: 30 random detections per stream per frame
  (`tools/benchmark_comparison.py`, `tools/benchmark_reid.py`) — a dense,
  worst-case association load.
- *Real*: YOLO26s detections on the bundled 1280x720 pedestrian video
  (`tools/benchmark_endtoend.py`).

**Frames tested**: 200 per configuration, after 10 warmup frames.

---

## 1. GPU vs CPU: Tracking Only

Synthetic workload, 30 detections/stream/frame, no ReID
(`tools/benchmark_comparison.py`):

| Streams | CPU (ms/frame) | GPU (ms/frame) | GPU Speedup | Recommendation |
|---------|----------------|----------------|-------------|----------------|
| 1       | 1.2            | 2.6            | 0.46x ❌    | Use CPU        |
| 4       | 9.6            | 5.1            | **1.9x** ✓  | Use GPU        |
| 8       | 47.2           | 8.2            | **5.8x** ✓✓ | Use GPU        |
| 16      | 104.1          | 14.5           | **7.2x** ✓✓ | Use GPU        |
| 32      | 216.2          | 26.3           | **8.2x** ✓✓✓| Use GPU        |
| 64      | 441.7          | 51.4           | **8.6x** ✓✓✓| Use GPU        |

**Key insights**:
- The GPU tracker has a fixed per-frame overhead (kernel launches, transfers),
  so the CPU tracker wins on 1–2 streams.
- **Breakeven point**: ~3-4 streams; from 8 streams the GPU wins by 4x+.
- 64 concurrent streams run stably.
- The speedup *grows* with load: GPU per-frame cost is a fixed number of
  batched operations whose size grows gently with total track count, while
  the CPU tracker pays per-stream/per-track Python+numpy costs that grow
  super-linearly on dense scenes (track churn inflates lost pools and cost
  matrices). Multi-threaded CPU baselines are host-sensitive — numbers above
  were reproduced within ~5% across repeated runs on an otherwise idle box.

**GPU cost vs detection density** (GPU ms/frame; CPU shown for contrast):

| Streams | GPU @10 dets | GPU @30 dets | GPU @50 dets | CPU @50 dets |
|---------|--------------|--------------|--------------|--------------|
| 8       | 3.3          | 8.2          | 10.2         | 69.8         |
| 16      | 5.0          | 14.5         | 18.2         | 155.9        |
| 32      | 8.6          | 26.3         | 34.5         | 327.9        |

GPU cost grows sub-linearly with density (batched kernels amortize), CPU cost
grows super-linearly — the denser the scene, the stronger the GPU case.

## 2. End-to-End: YOLO26s + Tracking

Real detections on the bundled pedestrian video
(`tools/benchmark_endtoend.py`). Times are per batch of N streams:

**8 streams**:

| Configuration    | YOLO (ms) | Tracking (ms) | Total (ms) | FPS  |
|------------------|-----------|---------------|------------|------|
| CPU, no ReID     | 5.8       | 31.8          | 37.6       | 26.6 |
| **GPU, no ReID** | 5.8       | **5.1**       | **11.0**   | **91.3** |
| CPU + ReID x0.5  | 5.8       | 59.8          | 65.6       | 15.2 |
| GPU + ReID x0.5  | 5.8       | 33.1          | 38.9       | 25.7 |

GPU tracking is **6.2x faster** than CPU at 8 streams; with ReID enabled the
GPU is **1.8x faster** (6.3x at 16 streams).

**GPU vs CPU tracking speedup by stream count** (same script):

| Streams | No ReID | With ReID (x0.5) |
|---------|---------|------------------|
| 4       | 2.60x   | 0.93x            |
| 8       | 6.19x   | 1.81x            |
| 16      | 5.59x   | 6.28x            |

With ReID at only 4 sparse streams the CPU and GPU are on par — the fixed
cost of frame uploads and batched extraction needs more crops to amortize;
from 8 streams the GPU pulls ahead.

## 3. ReID Overhead

OSNet inference dominates the cost of ReID. On the dense synthetic workload
(30 crops per stream per frame, OSNet x0.5, `tools/benchmark_reid.py`):

| Streams | GPU no ReID (ms) | GPU + ReID (ms) | ReID share |
|---------|------------------|-----------------|------------|
| 1       | 2.4              | 11.4            | ~75%       |
| 4       | 4.9              | 21.2            | ~70%       |
| 8       | 8.2              | 39.9            | ~76%       |
| 16      | 14.0             | 83.3            | ~79%       |

Extraction uses a single batched `roi_align` per frame (frames upload once
through pinned staging buffers) and FP16 OSNet inference; the remaining cost
is dominated by the OSNet forward itself — see the roadmap for TensorRT and
`reid_interval` for a config-level lever.

**Takeaways**:
- ReID cost scales with the *number of detection crops*, not streams per se.
  30 objects/frame is a dense scene; typical pedestrian scenes (~10–15
  objects) cost proportionally less (see the end-to-end table above).
- Feature extraction is ~70-80% of the ReID cost; the remaining lever is the
  OSNet forward itself — see the roadmap in
  [optimization_guide.md](optimization_guide.md#future-optimizations-roadmap)
  (TensorRT) and the `reid_interval` config knob.
- Use the smallest OSNet variant that holds your accuracy target, and enable
  ReID only when you actually need occlusion robustness or cross-camera IDs.
- Measure your own workload with `bytetrack.Profiler` — the
  `0_reid_extraction` key isolates ReID time exactly.

## 4. Profiling Breakdown

GPU tracker, 16 streams, synthetic workload, no ReID (percentages of tracking
time):

| Operation                 | Mean (ms) | % of total |
|---------------------------|-----------|------------|
| `8_state_update_seq`      | 4.7       | 34.6%      |
| `6_hungarian_parallel`    | 3.3       | 24.4%      |
| `3_kalman_predict`        | 2.3       | 16.7%      |
| `7_kalman_update_matched` | 1.6       | 11.7%      |
| `4_compute_iou_cost`      | 0.8       | 5.6%       |
| `2_gather_states`         | 0.5       | 3.6%       |
| `1_convert_detections`    | 0.4       | 2.8%       |
| `5_gpu_to_cpu_transfer`   | 0.1       | 0.6%       |

Two optimization passes shaped this profile: the cross-stream batched Kalman
update cut `7_kalman_update_matched` from 11.8 ms (37.8% of the frame) to
1.6 ms, and the padded batched IoU with vectorized box preparation cut
`4_compute_iou_cost` from 5.6 ms to 0.8 ms. The remaining top items are
CPU-side Python bookkeeping — see
[optimization_guide.md](optimization_guide.md) for remaining candidates.

See [optimization_guide.md](optimization_guide.md) for how to interpret and
act on these numbers.

## 5. Memory Usage

GPU memory at 16 streams with ReID (x0.5): **~580 MB allocated**
(~5.5 GB reserved by the PyTorch caching allocator — normal, reused between
frames).

**Recommendation**: 8GB+ VRAM comfortably covers 16+ streams with ReID.

## 6. Recommendations

✅ **Use GPU when**:
- 4+ video streams (5.8x faster from 8 streams, 8.6x at 64)
- ReID is enabled — batch OSNet inference is much faster on GPU
- High detection density (20+ objects/frame)

❌ **Use CPU when**:
- 1–2 video streams without ReID
- No CUDA device available (`device='cpu'` is fully supported)

**Balanced production config**:

```python
config = TrackerConfig(
    track_thresh=0.5,
    match_thresh=0.8,
    track_buffer=30,
    enable_reid=True,
    reid_model_path='osnet_x0_5',   # best speed/accuracy balance
    reid_threshold=0.5,
    lambda_emb=0.3,
    device='cuda',
)
```

## Running Benchmarks Yourself

```bash
# CPU vs GPU comparison (tracking only, synthetic detections)
python tools/benchmark_comparison.py --video test_videos/6387-191695740_medium.mp4 \
    --num-frames 200 --streams 1,4,8,16,32,64 --device cuda

# ReID overhead (synthetic detections)
python tools/benchmark_reid.py --video test_videos/6387-191695740_medium.mp4 \
    --num-frames 200 --streams 1,4,8,16 --device cuda

# End-to-end with a real YOLO detector (needs ultralytics)
python tools/benchmark_endtoend.py --video test_videos/6387-191695740_medium.mp4 \
    --num-frames 200 --streams 4,8,16 --device cuda

# Or run everything at once
bash scripts/run_benchmarks.sh test_videos/6387-191695740_medium.mp4
```

Each script prints per-operation profiling, FPS metrics and GPU memory usage.

---

**Note on comparing versions**: these numbers were measured after the tracker
gained the full three-stage ByteTrack association (low-confidence BYTE
recovery and a track confirmation stage) and in-association ReID fusion.
They are not directly comparable to numbers published for earlier commits,
which tracked less state per frame.

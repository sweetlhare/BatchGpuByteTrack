# Optimization Guide

How `BatchGPUTracker` spends its time, how to measure it, and which knobs actually matter.

## Architecture: What Runs Where

One `update()` call processes one frame from every stream. The pipeline stages below are exactly the keys the built-in profiler reports:

| Stage | Where | What it does |
|-------|-------|--------------|
| `0_reid_extraction` | GPU | Batched `roi_align` crop extraction + OSNet inference for all streams (only with ReID + `frames`) |
| `1_convert_detections` | CPU | Vectorized detection validation, track objects |
| `2_gather_states` | CPU | Collect Kalman states of all tracks from all streams |
| `3_kalman_predict` | GPU | **One** batched Kalman predict over every track of every stream |
| `4_compute_iou_cost` | GPU | **One** padded batched IoU kernel over all streams |
| `5_gpu_to_cpu_transfer` | — | One bulk download of all cost matrices |
| `5b_embedding_cost` | CPU | Cosine-distance matrices (only with ReID embeddings) |
| `6_hungarian_parallel` | CPU | Three-stage ByteTrack association, one stream per thread (`ThreadPoolExecutor(num_threads)`) |
| `7_kalman_update_matched` | GPU | Batched Kalman update for matched pairs, batched init for new tracks |
| `8_state_update_seq` | CPU | Per-stream track lifecycle updates, sequential |

Design rationale:

- **Batching across streams** is the whole point: Kalman predict/update and IoU costs for N streams cost roughly the same number of kernel launches as for one stream.
- **Hungarian assignment stays on CPU** (`lap.lapjv` per stream) — the matrices are small and the solver is sequential, so the win comes from running streams in parallel threads, not from the GPU.
- **State updates are sequential** on purpose: they are pure-Python object manipulation, and threading them is slower under the GIL.

## When GPU Wins (and When It Doesn't)

Every `update()` pays a fixed cost for tensor transfers and kernel launches. With one stream and a dozen boxes there is almost nothing to batch, so plain CPU tracking is faster. As stream count grows, the fixed cost is amortized and the GPU pulls ahead — on the reference RTX 4090 setup the crossover is around 4 streams, and the gap widens from there. **Do not rely on our numbers**: see [benchmarks.md](benchmarks.md) for the measured tables, and re-run the benchmarks on your hardware:

```bash
# Tracking only, CPU (BYTETracker) vs GPU (BatchGPUTracker), synthetic detections
python tools/benchmark_comparison.py --video test_videos/6387-191695740_medium.mp4 \
    --num-frames 200 --streams 1,4,8,16 --device cuda

# End-to-end with a real YOLO detector
python tools/benchmark_endtoend.py --video test_videos/6387-191695740_medium.mp4 \
    --num-frames 200 --streams 1,4,8,16 --yolo-model yolo26s.pt --device cuda

# ReID overhead (with vs without appearance features)
python tools/benchmark_reid.py --video test_videos/6387-191695740_medium.mp4 \
    --num-frames 200 --streams 1,2,4,8 --device cuda
```

Rules of thumb:

- **1–3 streams, no ReID** → CPU (`BYTETracker` or `ParallelCPUTracker`).
- **4+ streams** → `BatchGPUTracker` on GPU.
- **ReID enabled** → GPU regardless of stream count; batched OSNet inference is where the GPU helps most.
- End-to-end, detector inference usually dominates the frame budget — profile the whole pipeline (`benchmark_endtoend.py`) before optimizing the tracker.

## Profiling

Pass a `Profiler` to the tracker and every stage above is timed (with proper CUDA synchronization):

```python
from bytetrack import BatchGPUTracker, Profiler, TrackerConfig

profiler = Profiler(device='cuda')
config = TrackerConfig(device='cuda')
tracker = BatchGPUTracker(num_streams=8, config=config, profiler=profiler)

# ... tracker.update(...) in your loop ...

profiler.print_summary()                      # per-stage mean/std/total table
stats = profiler.get_stats()                  # dict of the same numbers
reid_ms = profiler.get_mean_time('0_reid_extraction')
```

Reading the summary:

- **`0_reid_extraction` dominates** → use a smaller OSNet variant, or disable ReID where you don't need it (see the [ReID guide](reid_guide.md)).
- **`6_hungarian_parallel` dominates** → raise `num_threads` (up to your stream count), or reduce the number of boxes per frame (higher detector confidence, higher `track_thresh`).
- **`3_kalman_predict` / `4_compute_iou_cost` / `5_gpu_to_cpu_transfer` dominate at low load** → GPU overhead is not being amortized; you likely have too few streams for the GPU path, try `device='cpu'`.
- **`8_state_update_seq` dominates** → you have very many tracks; lower `track_buffer` so lost tracks expire sooner.

## Tuning Parameters

```python
config = TrackerConfig(
    track_thresh=0.6,
    track_buffer=30,
    match_thresh=0.8,
    num_threads=8,
    device='cuda',
)
```

- **`num_threads`** — size of the CPU thread pool for Hungarian assignment. More threads than streams gives nothing; on small machines 4–8 is plenty.
- **`track_thresh`** — splits detections into high/low confidence for the two BYTE association stages; new tracks are only created above `track_thresh + 0.1`. Higher values mean fewer tracks and smaller cost matrices (faster), at the risk of missing weak objects.
- **`match_thresh`** — first-stage association gate on the (score-fused) IoU cost. Mostly a quality knob, negligible speed impact.
- **`track_buffer`** — how many frames a lost track is kept. Lost tracks stay in the association pool, so a larger buffer means more Kalman states and bigger cost matrices every frame. Use 30 as a baseline; raise it only if you need long occlusion recovery.
- **ReID on/off** — by far the largest single cost when enabled; see the [ReID guide](reid_guide.md) for variant selection and the external-embeddings option (compute features once in your own pipeline and pass them via `update(..., embeddings=...)`).

## Numerical Stability (Built In)

Long multi-stream runs stress the Kalman filter in ways short demos don't. The tracker ships with safeguards — nothing to configure, listed here so the behavior isn't surprising:

- **Covariance clamping** — after batched predict/update, covariance diagonals are clamped to a sane range (small positive floor, `1e5` ceiling), and the state's aspect ratio and height are clamped to physical bounds (`gpu_kalman_filter.py`).
- **Corrupted-track isolation** — after each predict, any track whose state contains NaN/inf (or an exploded covariance) is marked lost/removed instead of poisoning the shared cost matrix; a warning is logged.
- **Detection validation** — detections with NaN/inf coordinates or non-positive size are replaced by dummy zero-score boxes (keeping indices aligned) and warned about once.

The overhead of these checks is negligible relative to the pipeline stages above.

## Future Optimizations (Roadmap)

**Recently implemented** (all measured in [benchmarks.md](benchmarks.md)):

- Cross-stream batched Kalman update — one upload/kernel/download per frame
  instead of per-stream loops (was 11.8 ms → 1.6 ms at 16 streams).
- Batched crop extraction via `torchvision.ops.roi_align` — frames upload
  once, all crops of all streams resize in a single kernel (previously each
  crop paid its own transfer + resize).
- FP16 OSNet inference (`reid_fp16=True`, default on CUDA).
- Every-K-frames feature extraction (`reid_interval`).
- Padded batched IoU — track/detection boxes of all streams packed on CPU,
  two bulk uploads, one batched kernel, one bulk download.
- Vectorized detection validation and track-box preparation (per-array numpy
  passes instead of per-object Python).
- Adaptive Hungarian dispatch — tiny frames solve inline instead of paying
  thread-pool overhead.
- CPU `BYTETracker`: bounded `removed_stracks` (the original implementation
  iterated the full removal history every frame — long runs got slower and
  leaked memory).

**Remaining candidates**, ordered by expected impact:

1. **GPU-resident Kalman state** — keep means/covariances in persistent GPU
   tensors (track objects hold row indices) so predict runs in place and the
   per-frame gather/upload/download/scatter disappears. Also makes shapes
   static, which unlocks CUDA graphs / `torch.compile(mode="reduce-overhead")`
   for the whole GPU section.
2. **TensorRT / `torch.compile` for OSNet** — the approach used by
   [FastMOT](https://github.com/GeekAlexis/FastMOT); typically another 2-4x
   on the model inference itself.
3. **Ambiguity-driven selective ReID** — extract features only when the IoU
   association is ambiguous, rather than on a fixed interval (see
   [When to Extract ReID Features, arXiv:2409.06617](https://arxiv.org/pdf/2409.06617)).
4. **State-update micro-optimizations** (`8_state_update_seq`) — per-track
   Python bookkeeping; modest headroom (caching box conversions in the output
   path, cheaper duplicate removal).

**Considered and rejected:**

- **Hungarian on GPU** — GPU LAP solvers ([Date & Nagi 2016](https://www.sciencedirect.com/science/article/abs/pii/S016781911630045X))
  pay off on problems with millions of variables; per-stream tracking
  matrices are tens×tens, where `lap.lapjv` on CPU finishes in microseconds.
- **`torch.compile` kernel fusion on dynamic shapes** — tried in this
  project's history; recompilation and CUDA-graph invalidation on
  shape changes ate the gains. Revisit only after (1) makes shapes static.

## Related Documentation

- [Benchmarks](benchmarks.md) — measured GPU vs CPU and ReID numbers
- [ReID Guide](reid_guide.md) — appearance features, OSNet variants, weights
- [API Reference](api_reference.md) — complete API docs
- [Troubleshooting](troubleshooting.md) — common issues

# API Reference

Complete API documentation for BatchGpuByteTrack. Every code sample on this page runs as-is against the package (CPU samples shown with `device='cpu'`; switch to `'cuda'` on a GPU machine).

## Contents

- [Package exports](#package-exports)
- [TrackerConfig](#trackerconfig)
- [BatchGPUTracker](#batchgputracker)
- [Track lifecycle and behavior notes](#track-lifecycle-and-behavior-notes)
- [Re-Identification](#re-identification)
- [BYTETracker (CPU reference)](#bytetracker-cpu-reference)
- [ParallelCPUTracker](#parallelcputracker)
- [Track objects: GPUSTrack and STrack](#track-objects-gpustrack-and-strack)
- [TrackState](#trackstate)
- [build_osnet](#build_osnet)
- [Profiler](#profiler)
- [Matching utilities (bytetrack.gpu_matching)](#matching-utilities-bytetrackgpu_matching)

---

## Package exports

```python
from bytetrack import (
    BatchGPUTracker,     # batched multi-stream tracker (GPU or CPU)
    ParallelCPUTracker,  # thread-parallel fallback built on BYTETracker
    TrackerConfig,       # configuration for BatchGPUTracker
    BYTETracker,         # reference single-stream CPU tracker
    STrack,              # track object returned by BYTETracker
    GPUSTrack,           # track object used by BatchGPUTracker
    StreamState,         # per-stream state container
    TrackState,          # track lifecycle enum
    build_osnet,         # OSNet Re-ID model factory
    Profiler,            # stage timing profiler
)
```

---

## TrackerConfig

Configuration dataclass for `BatchGPUTracker`.

```python
from bytetrack import TrackerConfig

config = TrackerConfig()  # all fields have defaults
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `track_thresh` | `float` | `0.5` | High-confidence threshold. Detections with `score > track_thresh` enter the first association. New tracks are only started for detections with `score >= track_thresh + 0.1`. |
| `track_buffer` | `int` | `30` | Max frames a lost track is kept before it is removed. |
| `match_thresh` | `float` | `0.8` | Max cost accepted in the first association stage. |
| `enable_reid` | `bool` | `False` | Enable appearance (Re-ID) fusion in the first association. |
| `reid_model_path` | `Optional[str]` | `None` | OSNet variant **name** (not a file path): `'osnet_x1_0'`, `'osnet_x0_75'`, `'osnet_x0_5'`, `'osnet_x0_25'`, `'osnet_ibn_x1_0'`. When set together with `enable_reid=True`, the tracker loads OSNet and extracts embeddings from `frames=` itself. |
| `reid_checkpoint` | `Optional[str]` | `None` | Path to a local `.pth` state dict for the Re-ID model. Use for fully offline setups (skips the weight download). |
| `reid_embedding_dim` | `int` | `512` | Embedding dimensionality (OSNet produces 512). |
| `reid_threshold` | `float` | `0.5` | Max embedding distance for a track/detection pair to use the fused appearance cost. Pairs above it fall back to pure IoU. |
| `lambda_emb` | `float` | `0.3` | Weight of the appearance cost in the fused cost (0..1). `0` disables appearance fusion entirely. |
| `reid_fp16` | `bool` | `True` | Run OSNet inference in FP16 (CUDA only, via autocast) — roughly 2x faster feature extraction with negligible embedding drift. Ignored on CPU. |
| `reid_interval` | `int` | `1` | Extract appearance features only on every K-th frame. In between, association falls back to IoU and tracks keep their smoothed embeddings. Large real-world savings when ReID dominates. |
| `enable_cross_camera` | `bool` | `False` | Assign global IDs across streams via an embedding gallery (requires Re-ID embeddings). |
| `cross_camera_threshold` | `float` | `0.7` | Min cosine similarity to match a track to an existing gallery identity. |
| `device` | `str` | `'cuda'` | `'cuda'` or `'cpu'`. |
| `num_threads` | `int` | `8` | CPU thread pool size for the parallel Hungarian assignment. |

---

## BatchGPUTracker

Batched multi-stream tracker. Kalman prediction/update and IoU cost matrices are computed in one batch across all streams (on GPU when `device='cuda'`); Hungarian assignment runs on parallel CPU threads.

### Constructor

```python
BatchGPUTracker(num_streams, config=None, profiler=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_streams` | `int` | required | Number of video streams processed per `update()` call. |
| `config` | `Optional[TrackerConfig]` | `None` | Tracker configuration; `None` means `TrackerConfig()` defaults (note: default device is `'cuda'`). |
| `profiler` | `Optional[Profiler]` | `None` | Optional [`Profiler`](#profiler) for per-stage timing. |

Raises `RuntimeError` at construction if `config.device='cuda'` but CUDA is not available.

### `update(detections, frames=None, embeddings=None)`

Process one frame from each stream.

| Parameter | Type | Description |
|-----------|------|-------------|
| `detections` | `List[np.ndarray]` | One entry per stream (length must equal `num_streams`, otherwise `ValueError`). Each entry is a `[N, 5]` **numpy** array with rows `[x1, y1, x2, y2, score]` — tlbr format in **pixel** coordinates. Not torch tensors. An entry may be `None` or empty for streams without detections. Invalid rows (NaN/inf or non-positive width/height) are replaced with dummy score-0 boxes and a warning is logged once. |
| `frames` | `Optional[List[np.ndarray]]` | One `[H, W, 3]` **BGR uint8** numpy array per stream (exactly what `cv2.VideoCapture`/`cv2.imread` return). Only used to extract Re-ID crops when `enable_reid=True` and `embeddings` is not given. |
| `embeddings` | `Optional[List[np.ndarray]]` | Pre-computed embeddings, one `[N_i, D]` array per stream (row-aligned with `detections`). When provided, the built-in extraction is skipped. |

**Returns** — `List[List[dict]]`: one list of track dicts per stream. Only confirmed tracks are reported. Each dict has exactly these keys:

| Key | Type | Description |
|-----|------|-------------|
| `'track_id'` | `int` | Track ID, unique within the stream (per-stream counters start at 1). |
| `'tlbr'` | `np.ndarray [4]` | Bounding box `[x1, y1, x2, y2]` in pixels (Kalman-filtered). |
| `'score'` | `float` | Score of the last matched detection. |
| `'stream_id'` | `int` | Index of the stream this track belongs to. |
| `'global_id'` | `int` or `None` | Cross-camera ID; `None` unless `enable_cross_camera=True`. |

### `reset()`

Clears all tracks, frame counters and per-stream track-ID counters (and the cross-camera gallery, if enabled).

### Example

```python
import numpy as np
from bytetrack import BatchGPUTracker, TrackerConfig

config = TrackerConfig(
    track_thresh=0.5,
    track_buffer=30,
    match_thresh=0.8,
    device='cpu',  # use 'cuda' on a GPU machine
)
tracker = BatchGPUTracker(num_streams=2, config=config)

for f in range(3):
    detections = [
        # stream 0: one object, rows are [x1, y1, x2, y2, score] in pixels
        np.array([[100 + 2 * f, 100, 150 + 2 * f, 220, 0.9]], dtype=np.float32),
        # stream 1: no detections this frame
        None,
    ]
    tracks = tracker.update(detections)

for stream_id, stream_tracks in enumerate(tracks):
    for t in stream_tracks:
        print(stream_id, t['track_id'], t['tlbr'], round(t['score'], 2), t['global_id'])

tracker.reset()  # clears all tracks and per-stream frame/ID counters
```

### Profiling

Pass a `Profiler` to time each pipeline stage (`0_reid_extraction`, `1_convert_detections`, `2_gather_states`, `3_kalman_predict`, `4_compute_iou_cost`, `5_gpu_to_cpu_transfer`, `5b_embedding_cost`, `6_hungarian_parallel`, `7_kalman_update_matched`, `8_state_update_seq`):

```python
import numpy as np
from bytetrack import BatchGPUTracker, TrackerConfig, Profiler

profiler = Profiler(device='cpu')  # match the tracker device
tracker = BatchGPUTracker(
    num_streams=1,
    config=TrackerConfig(device='cpu'),
    profiler=profiler,
)

for f in range(20):
    tracker.update([np.array([[100 + f, 100, 150 + f, 220, 0.9]], dtype=np.float32)])

profiler.print_summary("BatchGPUTracker timing")

stats = profiler.get_stats()  # dict: stage name -> {mean, std, min, max, total, count}
print(stats['6_hungarian_parallel']['mean'])
print(profiler.get_mean_time('3_kalman_predict'))  # ms; 0.0 if never recorded
profiler.reset()
```

---

## Track lifecycle and behavior notes

The tracker follows the original ByteTrack logic:

- **Detection score bands.** `score > track_thresh`: used in the first association. `0.1 < score <= track_thresh`: used only in the second ("BYTE") association, which matches remaining *tracked* tracks against low-confidence detections on raw IoU (threshold 0.5) — this is what keeps tracks alive through occlusion and motion blur. `score <= 0.1`: ignored entirely.
- **New tracks** are only started from unmatched detections with `score >= track_thresh + 0.1`.
- **Confirmation stage.** On frame 1, new tracks are confirmed and reported immediately. On any later frame, a new track is *unconfirmed* for one frame: it is not reported until it is matched again on the next frame, and it is removed if it is not. This suppresses one-frame false positives:

```python
import numpy as np
from bytetrack import BatchGPUTracker, TrackerConfig

tracker = BatchGPUTracker(num_streams=1, config=TrackerConfig(device='cpu'))
det = np.array([[100, 100, 150, 220, 0.9]], dtype=np.float32)

tracker.update([None])        # frame 1: nothing
out = tracker.update([det])   # frame 2: new detection -> unconfirmed, not reported
assert out == [[]]
out = tracker.update([det])   # frame 3: second detection -> confirmed, reported
assert len(out[0]) == 1
```

- **Lost tracks** are kept for `track_buffer` frames and keep participating in the first association; a re-detected object gets its old `track_id` back. After the buffer expires, the track is removed.
- **Association thresholds** per stage: first — `match_thresh` (default 0.8) on score-fused IoU cost; second — fixed 0.5 on raw IoU; third (unconfirmed tracks vs remaining high-confidence detections) — fixed 0.7 on score-fused cost.
- **Streams are independent.** Each stream has its own frame counter and track-ID counter starting at 1.

---

## Re-Identification

With `enable_reid=True`, appearance embeddings are fused into the **first association stage**. For each track/detection pair where both sides have an embedding and the embedding distance is `<= reid_threshold`, the IoU cost is blended with an IoU-weighted appearance cost using weight `lambda_emb` (`cost = lambda_emb * appearance_cost + (1 - lambda_emb) * iou_cost`). All other pairs use pure IoU. Track embeddings are smoothed over time with an exponential moving average.

There are two ways to supply embeddings:

### 1. External embeddings (bring your own model)

```python
import numpy as np
from bytetrack import BatchGPUTracker, TrackerConfig

config = TrackerConfig(
    enable_reid=True,
    lambda_emb=0.3,      # weight of the appearance cost in the fused cost
    reid_threshold=0.5,  # max appearance distance for fusion
    device='cpu',
)
tracker = BatchGPUTracker(num_streams=1, config=config)

rng = np.random.default_rng(0)
emb = rng.normal(size=(1, 512)).astype(np.float32)  # one [N, D] array per stream
emb /= np.linalg.norm(emb, axis=1, keepdims=True)

for f in range(3):
    dets = np.array([[100 + 2 * f, 100, 150 + 2 * f, 220, 0.9]], dtype=np.float32)
    tracks = tracker.update([dets], embeddings=[emb])

print(tracks[0][0]['track_id'])
```

### 2. Built-in OSNet extraction (pass frames)

Set `reid_model_path` to an OSNet variant name and pass `frames=`. Crops are cut from the frames, converted BGR to RGB, resized to 256x128, normalized with ImageNet statistics, and run through OSNet in one batch across all streams:

```python
import numpy as np
from bytetrack import BatchGPUTracker, TrackerConfig

config = TrackerConfig(
    enable_reid=True,
    reid_model_path='osnet_x0_5',  # OSNet variant name, not a file path
    device='cpu',
)
tracker = BatchGPUTracker(num_streams=1, config=config)

frame = np.zeros((720, 1280, 3), dtype=np.uint8)  # HWC BGR, as returned by cv2
dets = np.array([[100, 100, 150, 220, 0.9]], dtype=np.float32)

tracks = tracker.update([dets], frames=[frame])
print(tracks)
```

Weights are downloaded automatically on first use (see [`build_osnet`](#build_osnet)). For offline machines set `reid_checkpoint='path/to/osnet_x0_5_imagenet.pth'` in the config.

If `enable_reid=True` but neither `frames` nor `embeddings` are given, the tracker silently falls back to pure IoU matching.

### Cross-camera global IDs

With `enable_cross_camera=True` (in addition to Re-ID embeddings), the tracker maintains a global embedding gallery. Each reported track is matched against the gallery by cosine similarity (`cross_camera_threshold`, default 0.7) and gets a `'global_id'` in the output dict that is consistent across streams.

---

## BYTETracker (CPU reference)

The original single-stream ByteTrack implementation. `BatchGPUTracker` is behavior-matched against it; use it for single-stream workloads (fastest for 1-3 streams) or as a reference.

### Constructor

```python
BYTETracker(args, frame_rate=30, enable_reid=False, reid_model=None,
            reid_model_path=None, reid_threshold=0.5, lambda_emb=0.3,
            device='cpu', profiler=None)
```

`args` is any object (namespace, dataclass, plain class) with these attributes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `track_thresh` | `float` | High-confidence threshold (new tracks require `score >= track_thresh + 0.1`). |
| `track_buffer` | `int` | Lost-track buffer length; the effective buffer is `int(frame_rate / 30 * track_buffer)`. |
| `match_thresh` | `float` | Max cost for the first association. |
| `mot20` | `bool` | If `True`, skips detection-score fusion in the cost matrix (MOT20 evaluation convention). Use `False` normally. |

Remaining parameters: `frame_rate` scales the lost-track buffer; `enable_reid`/`reid_model_path`/`reid_threshold`/`lambda_emb`/`device` enable OSNet-based appearance fusion (pass `frame=` to `update()`); `reid_model` lets several trackers share one already-built model; `profiler` accepts a [`Profiler`](#profiler).

### `update(output_results, img_info, img_size, frame=None)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `output_results` | `np.ndarray` | `[N, 5]` numpy array, rows `[x1, y1, x2, y2, score]` (tlbr, pixels). A raw YOLOX-style output with 6+ columns is also accepted, but that path expects a **torch tensor** and computes `score = obj_conf * cls_conf`. |
| `img_info` | `(int, int)` | Original image `(height, width)`. |
| `img_size` | `(int, int)` | Model input `(height, width)`. Boxes are divided by `min(img_size[0]/img_h, img_size[1]/img_w)` to map them back to original-image coordinates. **If your detections are already in original-image pixels, pass the same tuple for both.** |
| `frame` | `Optional[np.ndarray]` | `[H, W, 3]` BGR uint8 frame; only used for Re-ID crop extraction when `enable_reid=True`. |

**Returns** — `list[STrack]`: currently confirmed tracks. Same lifecycle rules as `BatchGPUTracker` (confirmation stage, BYTE second association, `track_buffer`).

Note: the rescaling modifies the input array in place. Pass a copy if you reuse the detections array afterwards.

### Example

```python
import numpy as np
from bytetrack import BYTETracker

class Args:
    track_thresh = 0.5
    track_buffer = 30
    match_thresh = 0.8
    mot20 = False

tracker = BYTETracker(Args(), frame_rate=30)

for f in range(3):
    detections = np.array([[100 + 2 * f, 100, 150 + 2 * f, 220, 0.9]], dtype=np.float32)
    # detections already in original-image pixels -> pass img_info == img_size
    tracks = tracker.update(detections, (720, 1280), (720, 1280))

for t in tracks:
    print(t.track_id, t.tlbr, t.score)
```

---

## ParallelCPUTracker

CPU-only fallback for multi-stream workloads: one `BYTETracker` per stream, updated in parallel via a thread pool. Useful when no GPU is available or GPU overhead is not worth it (1-3 streams).

```python
ParallelCPUTracker(num_streams, args)   # args: same object as for BYTETracker
```

- **`update(detections, img_infos, img_sizes)`** — `detections` is a list of `[N, 5]` numpy arrays (one per stream); `img_infos` and `img_sizes` are lists of `(height, width)` tuples (see `BYTETracker.update` for their meaning). Returns a list of `list[STrack]` per stream.
- **`reset()`** — recreates all per-stream trackers.

Note: `STrack` IDs come from a process-global counter shared by all `BYTETracker` instances, so track IDs are unique **across** streams here (unlike `BatchGPUTracker`, where each stream counts from 1).

```python
import numpy as np
from bytetrack import ParallelCPUTracker

class Args:
    track_thresh = 0.5
    track_buffer = 30
    match_thresh = 0.8
    mot20 = False

tracker = ParallelCPUTracker(num_streams=2, args=Args())

for f in range(3):
    detections = [
        np.array([[100 + 2 * f, 100, 150 + 2 * f, 220, 0.9]], dtype=np.float32),
        np.array([[500, 300, 560, 420, 0.85]], dtype=np.float32),
    ]
    tracks = tracker.update(
        detections,
        img_infos=[(720, 1280), (720, 1280)],
        img_sizes=[(720, 1280), (720, 1280)],
    )

for stream_id, stream_tracks in enumerate(tracks):
    for t in stream_tracks:
        print(stream_id, t.track_id, t.tlbr)
```

---

## Track objects: GPUSTrack and STrack

### GPUSTrack

Used internally by `BatchGPUTracker` (its `update()` returns plain dicts, see above). You can inspect the full track objects through `tracker.stream_states`:

```python
import numpy as np
from bytetrack import BatchGPUTracker, TrackerConfig

tracker = BatchGPUTracker(num_streams=1, config=TrackerConfig(device='cpu'))
for f in range(3):
    tracker.update([np.array([[100 + f, 100, 150 + f, 220, 0.9]], dtype=np.float32)])

state = tracker.stream_states[0]     # StreamState for stream 0
for track in state.tracked_stracks:  # list of GPUSTrack
    print(track.track_id, track.tlwh, track.tlbr, track.xyah)
    print(track.score, track.state, track.is_activated, track.global_id)
```

| Attribute / property | Type | Description |
|----------------------|------|-------------|
| `track_id` | `int` | Track ID within the stream. |
| `stream_id` | `int` | Owning stream index. |
| `tlwh` | `np.ndarray [4]` (property) | Current box `[x, y, w, h]` from the Kalman state (falls back to the last detection box before activation or if the state is invalid). |
| `tlbr` | `np.ndarray [4]` (property) | Current box `[x1, y1, x2, y2]`. |
| `xyah` | `np.ndarray [4]` (property) | Current state `[center_x, center_y, aspect_ratio, height]`. |
| `score` | `float` | Score of the last matched detection. |
| `state` | `TrackState` | `New` / `Tracked` / `Lost` / `Removed`. |
| `is_activated` | `bool` | Whether the track is confirmed (reported in output). |
| `mean`, `covariance` | `np.ndarray` | `[8]` / `[8, 8]` Kalman state. |
| `frame_id`, `start_frame`, `end_frame` | `int` | Last update frame / first frame / alias of `frame_id`. |
| `tracklet_len` | `int` | Consecutive matched frames. |
| `embedding` | `np.ndarray` or `None` | Last Re-ID embedding. |
| `smooth_embedding` | `np.ndarray` or `None` | EMA-smoothed embedding (alpha 0.9). |
| `global_id` | `int` or `None` | Cross-camera ID. |

`StreamState` (one per stream) exposes `tracked_stracks`, `lost_stracks`, `removed_stracks`, `frame_id`, and `reset()`.

### STrack

Returned by `BYTETracker.update()` and `ParallelCPUTracker.update()`. Key attributes: `track_id` (`int`, globally unique in the process), `tlwh` / `tlbr` (`np.ndarray [4]` properties), `score` (`float`), `state`, `is_activated`, `frame_id`, `start_frame`, `end_frame`, `tracklet_len`, `curr_feat` / `smooth_feat` (Re-ID features or `None`). Static helpers: `STrack.tlbr_to_tlwh(box)`, `STrack.tlwh_to_tlbr(box)`, `STrack.tlwh_to_xyah(box)` (numpy).

---

## TrackState

```python
from bytetrack import TrackState

TrackState.New      # 0 - created, not yet activated
TrackState.Tracked  # 1 - actively tracked
TrackState.Lost     # 2 - lost, kept in buffer
TrackState.Removed  # 3 - removed
```

`bytetrack.TrackState` is an `IntEnum` used by `GPUSTrack`. The CPU tracker's `STrack` uses `bytetrack.BaseTrackState` (a plain class with the same names and values).

---

## build_osnet

Factory for OSNet re-identification models.

```python
build_osnet(model_name='osnet_x0_5', pretrained=True, device='cuda', checkpoint=None, **kwargs)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | `str` | `'osnet_x0_5'` | One of `'osnet_x1_0'`, `'osnet_x0_75'`, `'osnet_x0_5'`, `'osnet_x0_25'`, `'osnet_ibn_x1_0'`. Anything else raises `ValueError`. |
| `pretrained` | `bool` | `True` | Download and load ImageNet-pretrained weights (ignored when `checkpoint` is given). |
| `device` | `str` | `'cuda'` | Device to move the model to; pass `None` to keep it on CPU for manual placement. |
| `checkpoint` | `Optional[str]` | `None` | Path to a local `.pth` state dict. Use for offline setups — no download is attempted. |

**Returns** — `torch.nn.Module` in eval mode, with a `feature_dim` attribute (512 for all variants). Forward pass takes `[N, 3, 256, 128]` RGB crops normalized with ImageNet statistics and returns `[N, 512]` features.

Weights are cached under `$TORCH_HOME/checkpoints` (default `~/.cache/torch/checkpoints`). If the primary download fails, a Google Drive fallback is tried (requires the optional `gdown` package); as a last resort the error message tells you which file to download manually.

```python
import torch
from bytetrack import build_osnet

# Downloads ImageNet-pretrained weights on first use
# (cached under ~/.cache/torch/checkpoints)
model = build_osnet('osnet_x0_5', pretrained=True, device='cpu')

# OSNet expects [N, 3, 256, 128] RGB crops normalized with ImageNet stats
batch = torch.randn(4, 3, 256, 128)
with torch.no_grad():
    features = model(batch)

print(features.shape)     # torch.Size([4, 512])
print(model.feature_dim)  # 512
```

Offline (no network access):

```python
model = build_osnet('osnet_x0_5', device='cpu',
                    checkpoint='weights/osnet_x0_5_imagenet.pth')
```

---

## Profiler

Stage timer with proper CUDA synchronization. Accepted by both `BatchGPUTracker` and `BYTETracker` via the `profiler=` constructor argument (see the [profiling example](#profiling)), and usable standalone:

```python
Profiler(device='cuda', enabled=True)
```

| Method | Description |
|--------|-------------|
| `profile(name)` | Context manager; times the enclosed block in ms. Calls `torch.cuda.synchronize()` around the block when `device` is CUDA. |
| `get_stats()` | `dict` mapping name to `{'mean', 'std', 'min', 'max', 'total', 'count'}` (times in ms). |
| `print_summary(title=None)` | Prints a formatted table sorted by mean time. |
| `get_mean_time(name)` | Mean time in ms for one operation (`0.0` if never recorded). |
| `reset()` | Clears all recorded timings. |

With `enabled=False` the profiler becomes a no-op with zero overhead.

---

## Matching utilities (bytetrack.gpu_matching)

Low-level building blocks used by `BatchGPUTracker`, importable for custom pipelines. All functions except `linear_assignment` operate on **torch tensors** (any device); `linear_assignment` takes a **numpy** cost matrix.

| Function | Signature | Description |
|----------|-----------|-------------|
| `compute_box_iou` | `(boxes1 [N,4], boxes2 [M,4]) -> [N,M]` | Pairwise IoU for tlbr boxes (uses `torchvision.ops.box_iou`, pure-torch fallback if torchvision is missing). |
| `iou_distance` | `(track_boxes [N,4], det_boxes [M,4]) -> [N,M]` | IoU cost matrix, `1 - IoU`. |
| `embedding_distance` | `(track_embeddings [N,D], det_embeddings [M,D], metric='cosine') -> [N,M]` | Embedding distance matrix; `metric` is `'cosine'` (1 - cosine similarity, inputs are re-normalized) or `'euclidean'`. |
| `fuse_score` | `(cost_matrix [N,M], det_scores [M]) -> [N,M]` | ByteTrack score fusion: `1 - IoU * score`. |
| `linear_assignment` | `(cost_matrix: np.ndarray [N,M], thresh: float) -> (matches, unmatched_tracks, unmatched_dets)` | Hungarian assignment with a cost cutoff. `matches` is a list of `(track_idx, det_idx)` tuples; the other two are lists of indices. Uses `lap.lapjv`, with a `scipy` fallback if `lap` is not installed. |
| `tlwh_to_tlbr` | `([N,4]) -> [N,4]` | `[x, y, w, h]` to `[x1, y1, x2, y2]`. |
| `tlwh_to_xyah` | `([N,4]) -> [N,4]` | `[x, y, w, h]` to `[cx, cy, aspect, h]`. Aspect ratio is clamped to `[0.1, 10.0]` and height to at least `1e-2` for numerical safety. |
| `xyah_to_tlwh` | `([N,4]) -> [N,4]` | Inverse of the above; width capped at `1e5`. |
| `xyah_to_tlbr` | `([N,4]) -> [N,4]` | `[cx, cy, aspect, h]` to `[x1, y1, x2, y2]`. |

```python
import torch
from bytetrack.gpu_matching import (
    iou_distance, embedding_distance, fuse_score, linear_assignment,
    tlwh_to_tlbr, tlwh_to_xyah, xyah_to_tlwh, xyah_to_tlbr,
)

track_boxes = torch.tensor([[100., 100., 200., 200.],
                            [300., 300., 400., 400.]])
det_boxes = torch.tensor([[110., 110., 210., 210.],
                          [600., 500., 660., 620.]])

cost = iou_distance(track_boxes, det_boxes)  # [2, 2] tensor, values = 1 - IoU

det_scores = torch.tensor([0.9, 0.6])
fused = fuse_score(cost, det_scores)         # 1 - IoU * score

matches, u_tracks, u_dets = linear_assignment(fused.numpy(), thresh=0.8)
print(matches)   # [(0, 0)]
print(u_tracks)  # [1]
print(u_dets)    # [1]

emb_tracks = torch.nn.functional.normalize(torch.randn(3, 512), dim=1)
emb_dets = torch.nn.functional.normalize(torch.randn(2, 512), dim=1)
emb_cost = embedding_distance(emb_tracks, emb_dets, metric='cosine')  # [3, 2]

tlwh = torch.tensor([[100., 100., 50., 120.]])
print(tlwh_to_tlbr(tlwh))  # tensor([[100., 100., 150., 220.]])
print(tlwh_to_xyah(tlwh))  # tensor([[125.0000, 160.0000, 0.4167, 120.0000]])
```

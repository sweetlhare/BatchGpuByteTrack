# Quick Start

From clone to tracked boxes in a few minutes. Every Python snippet below is self-contained: save it as a file in the repository root and run `python3 <file>.py` (with the package installed via `pip install -e .` it runs from anywhere).

## Installation

```bash
git clone https://github.com/sweetlhare/BatchGpuByteTrack.git
cd BatchGpuByteTrack

# Option A: uv (fast)
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Option B: pip
pip install -r requirements.txt

# Option C: editable package install
pip install -e .
```

For `device='cuda'` you need a CUDA build of PyTorch (adjust for your CUDA version):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

See [INSTALL.md](../INSTALL.md) for conda and platform details.

## Detections in, tracks out

Both trackers are detector-agnostic. Per frame you pass one `[N, 5]` **numpy** array — rows `[x1, y1, x2, y2, score]` in **pixel** coordinates (not normalized, not torch tensors — see [Troubleshooting](troubleshooting.md)). Frames are only needed when ReID is enabled, and are plain **BGR** arrays exactly as `cv2` reads them — the tracker handles color conversion internally.

## 1. CPU tracking (single stream)

Best for 1–3 streams — lowest overhead. `BYTETracker` takes a plain args object:

```python
import numpy as np
from bytetrack import BYTETracker


class Args:
    track_thresh = 0.6   # high-confidence detection threshold
    track_buffer = 30    # frames to keep lost tracks alive
    match_thresh = 0.8   # IoU threshold for matching
    mot20 = False

tracker = BYTETracker(Args(), frame_rate=30)

img_info = (720, 1280)   # original frame (height, width)
img_size = (720, 1280)   # detector input (height, width); equal -> no rescaling

for frame_idx in range(5):
    # One moving object: [x1, y1, x2, y2, score] in pixels
    x = 100 + 3 * frame_idx
    detections = np.array([[x, 100, x + 80, 300, 0.9]], dtype=np.float32)

    tracks = tracker.update(detections, img_info, img_size)

    for t in tracks:
        print(f"frame {frame_idx}: id={t.track_id} "
              f"tlbr={t.tlbr.astype(int)} score={t.score:.2f}")
```

Output — one object, one stable ID:

```
frame 0: id=1 tlbr=[100 100 180 300] score=0.90
frame 1: id=1 tlbr=[102 100 182 300] score=0.90
...
```

`update()` returns `STrack` objects with `.track_id`, `.tlbr`, `.score`. If your detector runs on resized input, pass its input size as `img_size` and the tracker rescales boxes back to `img_info` coordinates.

## 2. GPU batch tracking (multi-stream)

`BatchGPUTracker` processes one frame from *every* stream per `update()` call — Kalman prediction/update and IoU run batched on the GPU. Use it for 4+ streams (see [benchmarks](benchmarks.md)).

```python
import numpy as np
import torch
from bytetrack import BatchGPUTracker, TrackerConfig

# Use 'cuda' in production (recommended for 4+ streams); 'cpu' also works
device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = TrackerConfig(
    track_thresh=0.6,
    track_buffer=30,
    match_thresh=0.8,
    device=device,
)
tracker = BatchGPUTracker(num_streams=4, config=config)

for frame_idx in range(5):
    x = 100 + 3 * frame_idx
    # One [N, 5] numpy array per stream: [x1, y1, x2, y2, score] in pixels.
    # Streams without detections this frame may pass None.
    batch_detections = [
        np.array([[x, 100, x + 80, 300, 0.9]], dtype=np.float32),   # stream 0
        np.array([[500, 200, 580, 400, 0.8]], dtype=np.float32),    # stream 1
        np.array([[x, 50, x + 60, 200, 0.85],
                  [300, 300, 380, 500, 0.9]], dtype=np.float32),    # stream 2
        None,                                                       # stream 3
    ]

    all_tracks = tracker.update(batch_detections)

    for stream_id, tracks in enumerate(all_tracks):
        for t in tracks:
            print(f"frame {frame_idx} stream {stream_id}: "
                  f"id={t['track_id']} tlbr={t['tlbr'].astype(int)} "
                  f"score={t['score']:.2f}")
```

Notes:

- Each track is a dict with keys `'track_id'`, `'tlbr'`, `'score'`, `'stream_id'`, `'global_id'`.
- Track IDs are per-stream (each stream counts from 1).
- `TrackerConfig` defaults to `device='cuda'` and raises a `RuntimeError` on machines without CUDA — pass `device='cpu'` there.

## 3. GPU + ReID (appearance features)

ReID fuses OSNet appearance embeddings into the matching cost, which helps keep IDs through occlusions and crossings. Pass the raw BGR frames alongside the detections — crops, color conversion, resize, and batched inference happen inside the tracker. Pretrained OSNet weights are downloaded automatically on first use and cached in `~/.cache/torch/checkpoints`.

```python
import cv2
import numpy as np
import torch
from bytetrack import BatchGPUTracker, TrackerConfig

# Use 'cuda' in production: ReID inference is much faster on GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = TrackerConfig(
    track_thresh=0.6,
    track_buffer=30,
    match_thresh=0.8,
    enable_reid=True,
    reid_model_path='osnet_x0_5',  # osnet_x0_25 / x0_5 / x0_75 / x1_0
    reid_threshold=0.5,            # max appearance distance for fusion
    lambda_emb=0.3,                # appearance weight (0 = IoU only)
    device=device,
)
tracker = BatchGPUTracker(num_streams=2, config=config)

caps = [cv2.VideoCapture('test_videos/6387-191695740_medium.mp4')
        for _ in range(2)]

for frame_idx in range(5):
    # cv2.read() returns BGR frames -- pass them as-is,
    # the tracker converts crops to RGB internally
    frames = [cap.read()[1] for cap in caps]

    # Replace with your detector's output ([N, 5] per stream, pixels)
    detections = [
        np.array([[100, 100, 300, 500, 0.9]], dtype=np.float32),
        np.array([[400, 150, 600, 550, 0.8]], dtype=np.float32),
    ]

    all_tracks = tracker.update(detections, frames=frames)

    for stream_id, tracks in enumerate(all_tracks):
        for t in tracks:
            print(f"frame {frame_idx} stream {stream_id}: "
                  f"id={t['track_id']} tlbr={t['tlbr'].astype(int)}")

for cap in caps:
    cap.release()
```

If you already compute embeddings yourself (e.g. from your own ReID model), pass them directly instead of frames: `tracker.update(detections, embeddings=per_stream_embeddings)` where each entry is an `[N, D]` array. For offline machines, see "OSNet weight download fails" in [Troubleshooting](troubleshooting.md).

## Run the bundled examples

The repository ships a short test video (`test_videos/6387-191695740_medium.mp4`). The examples use YOLO from `ultralytics` as the detector (installed via `requirements.txt`; `yolo26s.pt` is downloaded automatically on first run):

```bash
# CPU, single stream (works on any machine)
python examples/basic_cpu_tracking.py --video test_videos/6387-191695740_medium.mp4 --no-display --max-frames 100

# GPU examples (require a CUDA device)
python examples/gpu_single_stream.py --video test_videos/6387-191695740_medium.mp4 --no-display
python examples/gpu_multistream.py   --video test_videos/6387-191695740_medium.mp4 --streams 8 --no-display
python examples/gpu_with_reid.py     --video test_videos/6387-191695740_medium.mp4 --streams 4 --no-display
```

Drop `--no-display` to get a live visualization window (press `q` to quit). Each example prints a timing summary at the end. There is also `examples/custom_detector.py` showing how to plug in a non-YOLO detector.

## Next steps

- [API Reference](api_reference.md) — all classes, parameters, and return types
- [Benchmarks](benchmarks.md) — CPU vs GPU, ReID cost, stream scaling
- [ReID Guide](reid_guide.md) — choosing an OSNet variant and tuning fusion
- [Troubleshooting](troubleshooting.md) — common errors and fixes

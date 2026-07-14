# Troubleshooting

Common errors and unexpected behaviors, with fixes. All error messages below are the actual messages produced by the current code.

## RuntimeError: CUDA is not available

```
RuntimeError: TrackerConfig(device='cuda') requested but CUDA is not available.
Pass TrackerConfig(device='cpu') or use ParallelCPUTracker instead.
```

`TrackerConfig` defaults to `device='cuda'`, so `BatchGPUTracker` raises this at construction on CPU-only machines even if you never wrote `'cuda'` yourself.

**Fixes**:

- Run on CPU: `TrackerConfig(device='cpu')` — the full `BatchGPUTracker` API works on CPU (and is the faster choice for 1–3 streams anyway, see [benchmarks](benchmarks.md)).
- Or install a CUDA build of PyTorch and verify it:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available())"
```

If `nvidia-smi` works but `torch.cuda.is_available()` is `False`, your PyTorch wheel was built without CUDA (the default `pip install torch` on some platforms) — reinstall from the CUDA index URL matching your driver.

## ModuleNotFoundError: No module named 'lap'

```
File ".../bytetrack/matching.py", line 3, in <module>
    import lap
ModuleNotFoundError: No module named 'lap'
```

`import bytetrack` itself fails without `lap` — the Hungarian solver is a hard dependency of the CPU tracker. It is listed in `requirements.txt`.

**Fix**:

```bash
pip install lap
# or, if lap has no prebuilt wheel for your Python version:
pip install lapx   # maintained fork, installs the same `lap` module
```

## OSNet weight download fails

With `enable_reid=True` the tracker downloads ImageNet-pretrained OSNet weights on first use from this repository's GitHub releases into `$TORCH_HOME/checkpoints` (`~/.cache/torch/checkpoints` by default). If that fails, it falls back to the original Google Drive links, which require the optional `gdown` package. With neither source reachable you get:

```
RuntimeError: Failed to download pretrained weights for "osnet_x0_5" from
https://github.com/sweetlhare/BatchGpuByteTrack/releases/download/weights-v1.0/osnet_x0_5_imagenet.pth
and the fallback requires the optional "gdown" package (pip install gdown). ...
```

**Fixes** (pick one):

1. **Pre-download** on a machine with internet access — this fills the exact cache location the tracker reads from:

   ```bash
   bash scripts/download_weights.sh
   ```

2. **Enable the Google Drive fallback**: `pip install gdown` (already in `requirements.txt`).

3. **Fully offline**: place the `.pth` file anywhere and point the tracker at it:

   ```python
   config = TrackerConfig(
       enable_reid=True,
       reid_model_path='osnet_x0_5',
       reid_checkpoint='/path/to/osnet_x0_5_imagenet.pth',  # local file, no network
       device='cpu',
   )
   ```

   Or when building the model directly: `build_osnet('osnet_x0_5', checkpoint='/path/to/osnet_x0_5_imagenet.pth')`.

4. **Manual placement**: download the file yourself and put it at `~/.cache/torch/checkpoints/osnet_x0_5_imagenet.pth` (same pattern for the other variants).

## Tracks don't appear on the first frames

This is by design, inherited from ByteTrack: a brand-new track is *unconfirmed* until it is matched to a detection on a second frame, so single-frame false positives never get reported. The only exception is frame 1, where new tracks activate immediately.

```python
import numpy as np
from bytetrack import BatchGPUTracker, TrackerConfig

tracker = BatchGPUTracker(num_streams=1, config=TrackerConfig(device='cpu'))

det = np.array([[100, 100, 200, 300, 0.9]], dtype=np.float32)

print(tracker.update([None]))    # frame 1: nothing        -> [[]]
print(tracker.update([det]))     # frame 2: new detection  -> [[]] (unconfirmed)
print(tracker.update([det]))     # frame 3: matched again  -> track reported
```

An unconfirmed track that is not re-detected on the very next frame is discarded.

Also note: **new tracks are only created from detections scoring at least `track_thresh + 0.1`**. With `track_thresh=0.6`, a 0.65-score detection can keep an *existing* track alive (via the low-confidence association stage) but will never start a new one. If objects are never picked up at all, lower `track_thresh` or raise your detector's recall.

## Low FPS with ReID enabled

ReID feature extraction usually dominates the frame budget. Measure before tuning — pass a `Profiler` to the tracker:

```python
from bytetrack import BatchGPUTracker, Profiler, TrackerConfig

profiler = Profiler(device=config.device)
tracker = BatchGPUTracker(num_streams=4, config=config, profiler=profiler)

# ... run your tracking loop ...

profiler.print_summary()
print(profiler.get_mean_time('0_reid_extraction'), "ms per batch on ReID")
```

Typical summary — here ReID is >90% of tracking time:

```
Operation                       Mean (ms)   Std (ms)    Count   Total (ms)       %
----------------------------------------------------------------------
0_reid_extraction                   17.35       0.87       10       173.49 ( 90.6%)
6_hungarian_parallel                 0.69       1.41       10         6.86 (  3.6%)
...
```

**Fixes**:

- Use a smaller OSNet variant: `reid_model_path='osnet_x0_25'` or `'osnet_x0_5'` instead of `'osnet_x1_0'` (see the comparison table in the [README](../README.md)).
- Run on GPU (`device='cuda'`) — batched ReID inference is 2–5x faster than CPU ([benchmarks](benchmarks.md)).
- If your detector pipeline already computes embeddings, pass them via `tracker.update(dets, embeddings=...)` and skip the built-in extraction.
- If appearance matching isn't actually needed, set `enable_reid=False`.

## TypeError inside update() when passing torch tensors

Detections must be **numpy arrays**, not torch tensors. Passing a tensor fails inside `update()`:

```
File ".../bytetrack/batch_gpu_tracker.py", line 326, in update
    is_valid = np.all(np.isfinite(tlbr))
TypeError: all() received an invalid combination of arguments ...
```

(CUDA tensors fail even earlier, at the numpy conversion step.)

**Fix** — convert on the way out of your detector, e.g. for ultralytics YOLO:

```python
boxes = result.boxes.xyxy.cpu().numpy()                    # (N, 4)
scores = result.boxes.conf.cpu().numpy().reshape(-1, 1)    # (N, 1)
detections = np.concatenate([boxes, scores], axis=1)       # (N, 5)
```

## IDs switch too often / tracks die during occlusion

Knobs, in the order worth trying:

- **`track_buffer`** (default 30): how many frames a lost track is kept and still eligible for re-matching. Raise to 60–90 for long occlusions — a track re-matched within the buffer keeps its original ID. (The CPU `BYTETracker` scales the buffer by `frame_rate / 30`.)
- **`match_thresh`** (default 0.8): maximum matching cost accepted in the first association. Raise it (e.g. 0.9) to accept lower-overlap matches when motion is fast; lower it if different objects get merged into one track.
- **`track_thresh`** (default 0.5 in `TrackerConfig`): lower it so more detections count as high-confidence; detections scoring in `(0.1, track_thresh]` are still used by the second (BYTE) association stage to bridge blurry/occluded frames.
- **Enable ReID**: appearance features are what actually resolves crossings of similar-motion objects:

  ```python
  config = TrackerConfig(
      enable_reid=True,
      reid_model_path='osnet_x0_5',
      reid_threshold=0.5,   # max appearance distance to trust
      lambda_emb=0.3,       # try 0.3-0.5; 0 disables appearance in the cost
      device='cuda',
  )
  ```

## Tracks vanish when using normalized coordinates

Coordinates must be in **pixels**. The GPU Kalman filter clamps box heights to at least 1 pixel as a numerical safeguard, so boxes in normalized `[0, 1]` coordinates get destroyed by the first Kalman update — the observable symptom is a track that appears once and is never reported again, with no error message.

**Fix** — scale before tracking:

```python
dets[:, [0, 2]] *= frame_width
dets[:, [1, 3]] *= frame_height
```

## Getting help

Still stuck? [Open an issue](https://github.com/sweetlhare/BatchGpuByteTrack/issues/new) with your Python/PyTorch/CUDA versions, GPU model, a minimal script that reproduces the problem, and the full traceback.

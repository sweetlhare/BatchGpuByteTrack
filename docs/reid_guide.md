# Re-Identification (ReID) Guide

How to use appearance features with `BatchGPUTracker`.

By default the tracker matches detections to tracks by motion only (Kalman prediction + IoU). Enabling ReID adds appearance embeddings on top of that:

- **Fewer ID switches** when objects cross paths or reappear after occlusion — two overlapping boxes with different appearance are no longer interchangeable.
- **Cross-camera global IDs** (optional) — the same object seen by different streams can share one `global_id` via an embedding gallery.

For a runnable end-to-end demo see [examples/gpu_with_reid.py](../examples/gpu_with_reid.py).

## How It Works

1. **Feature extraction** — frames are uploaded to the GPU once per stream, and all detection crops of all streams are cut and resized to 256×128 in a **single batched `roi_align` call** (BGR → RGB conversion and ImageNet normalization run on the GPU). Embeddings come from **one batched OSNet inference across all streams** — in FP16 by default on CUDA (`reid_fp16=True`, ~2x faster with negligible embedding drift). With `reid_interval=K` extraction runs only on every K-th frame; in between, association falls back to IoU and tracks keep their smoothed embeddings. The resulting embeddings are L2-normalized.
2. **Track memory** — each track keeps an exponentially smoothed embedding: `smooth = 0.9 * smooth + 0.1 * new` on every matched detection. This is what tracks are compared against.
3. **Cost fusion (first association stage only)** — ByteTrack matches in three stages. Appearance is used only in the **first stage** (confirmed + lost tracks vs high-confidence detections). For each track/detection pair where **both sides have an embedding** and the cosine distance is within `reid_threshold`, the IoU cost is blended with an appearance cost:

   ```
   iou_sim        = 1 - iou_cost
   reid_sim       = 1 - cosine_distance
   fused_emb_cost = 1 - reid_sim * (1 + iou_sim) / 2
   cost           = lambda_emb * fused_emb_cost + (1 - lambda_emb) * iou_cost
   ```

   Pairs that fail the gate (missing embedding, or appearance too dissimilar) keep the plain IoU cost. The second (BYTE) stage — remaining tracks vs low-confidence detections — always uses raw IoU, as does the third stage for unconfirmed tracks.

## OSNet Model Variants

The built-in extractor is OSNet (Zhou et al., ICCV 2019). All variants produce **512-dim** embeddings; parameter counts are for inference (classifier head excluded).

| Variant | Params | Notes |
|---------|--------|-------|
| `osnet_x1_0` | 2.2M | Highest accuracy, slowest |
| `osnet_x0_75` | 1.3M | |
| `osnet_x0_5` | 0.6M | **Recommended** — best speed/accuracy balance |
| `osnet_x0_25` | 0.2M | Fastest, weakest features |
| `osnet_ibn_x1_0` | 2.2M | x1.0 + instance-batch normalization, better cross-domain generalization |

Per-variant timings depend on GPU and detection count — see [benchmarks.md](benchmarks.md) or run `tools/benchmark_reid.py` on your own hardware.

## Enabling ReID

```python
from bytetrack import BatchGPUTracker, TrackerConfig

config = TrackerConfig(
    track_thresh=0.6,
    track_buffer=30,
    match_thresh=0.8,
    enable_reid=True,
    reid_model_path='osnet_x0_5',  # OSNet variant name
    reid_threshold=0.5,            # appearance gate (cosine distance)
    lambda_emb=0.3,                # appearance weight in fused cost
    device='cuda',
)
tracker = BatchGPUTracker(num_streams=2, config=config)

# frames: one BGR uint8 [H, W, 3] array per stream, exactly as returned by
# cv2.VideoCapture / cv2.imread. Do NOT convert to RGB or crop/resize —
# the tracker does all preprocessing internally.
# detections: one [N_i, 5] float array per stream: [x1, y1, x2, y2, score]
tracks = tracker.update(detections, frames=frames)

for stream_tracks in tracks:
    for t in stream_tracks:
        print(t['track_id'], t['tlbr'], t['score'])
```

If `frames` is omitted on some calls, those frames are simply tracked by IoU only — ReID is per-call opt-in.

### Tuning

- **`reid_threshold`** — cosine *distance* gate (0 = identical appearance). Only pairs with distance ≤ threshold get appearance fusion. Lower it (e.g. 0.3) to fuse only near-certain appearance matches; raise it (e.g. 0.7) to let appearance influence more pairs.
- **`lambda_emb`** — weight of the appearance term, 0..1. `0` disables fusion entirely; `0.3` (default) keeps motion dominant; `0.5+` lets appearance override IoU during crossings.
- **`track_buffer`** — raise (e.g. 60) for long occlusions so lost tracks survive until the object reappears; ReID then helps re-attach them in the first stage.

## Model Weights

Weights are ImageNet-pretrained OSNet checkpoints from [deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid), re-hosted on this repository's GitHub releases.

- **Automatic**: on first use, weights are downloaded from the GitHub release (sha256-verified) into `$TORCH_HOME/checkpoints` (`~/.cache/torch/checkpoints` by default). If that fails, a Google Drive fallback is tried (requires the optional `gdown` package). Checkpoints are loaded with `torch.load(..., weights_only=True)`.
- **Pre-download** (e.g. when preparing a server):

  ```bash
  bash scripts/download_weights.sh
  ```

- **Fully offline**: point the tracker at a local `.pth` state dict:

  ```python
  config = TrackerConfig(
      enable_reid=True,
      reid_model_path='osnet_x0_5',
      reid_checkpoint='/models/osnet_x0_5_imagenet.pth',  # no network access
      device='cuda',
  )
  ```

  Or build the model directly:

  ```python
  from bytetrack import build_osnet

  model = build_osnet('osnet_x0_5', device='cuda',
                      checkpoint='/models/osnet_x0_5_imagenet.pth')
  ```

## External Embeddings (Custom ReID Models)

You can skip the built-in OSNet entirely and pass precomputed embeddings — useful when you already run a ReID model elsewhere in your pipeline, or want a different architecture:

```python
config = TrackerConfig(
    enable_reid=True,
    reid_embedding_dim=256,   # must match your model's output dim
    reid_threshold=0.5,
    lambda_emb=0.3,
    device='cuda',
)
tracker = BatchGPUTracker(num_streams=2, config=config)

# embeddings: one [N_i, D] float array per stream, row j corresponds to
# detections[i][j]
embeddings = [my_reid_model(frame, dets) for frame, dets in zip(frames, detections)]
tracks = tracker.update(detections, embeddings=embeddings)
```

Notes:

- Leave `reid_model_path=None` — no OSNet is loaded and no weights are downloaded.
- Set `reid_embedding_dim` to your model's output dimension (default 512); it is used for empty per-stream arrays and the cross-camera gallery.
- If both `frames` and `embeddings` are passed, `embeddings` wins — internal extraction is skipped.
- Embeddings are compared by cosine distance, so any L2-normalizable feature vector works.

## Cross-Camera Tracking

With `enable_cross_camera=True`, confirmed tracks are additionally matched against a global embedding gallery shared by all streams. Matching tracks get the same `global_id` in the output dicts:

```python
config = TrackerConfig(
    enable_reid=True,
    reid_model_path='osnet_x0_5',
    enable_cross_camera=True,
    cross_camera_threshold=0.7,   # cosine similarity: higher = stricter
    device='cuda',
)
tracker = BatchGPUTracker(num_streams=4, config=config)

tracks = tracker.update(detections, frames=frames)
for stream_tracks in tracks:
    for t in stream_tracks:
        print(t['stream_id'], t['track_id'], '->', t['global_id'])
```

- `cross_camera_threshold` is a cosine **similarity** threshold (note: opposite direction from `reid_threshold`): a track joins an existing global ID only if similarity exceeds it, otherwise a new global ID is created. Raise it to avoid merging distinct objects; lower it to tolerate viewpoint changes between cameras.
- Gallery entries are updated with an exponential moving average and expire after long inactivity.
- `global_id` is only assigned to tracks that have embeddings (from OSNet or external); without cross-camera mode the key is present but `None`.

## Performance

ReID cost is dominated by OSNet inference — it typically dwarfs all tracking math combined. Measure it with the built-in profiler (`'0_reid_extraction'` covers crop preprocessing + inference):

```python
from bytetrack import BatchGPUTracker, Profiler, TrackerConfig

profiler = Profiler(device='cuda')
tracker = BatchGPUTracker(num_streams=4, config=config, profiler=profiler)

# ... tracker.update(detections, frames=frames) in your loop ...

profiler.print_summary()
print(profiler.get_mean_time('0_reid_extraction'), 'ms per update')
```

Practical levers, in order of impact:

1. **Run ReID on GPU.** Extraction is batched across all streams, so GPU batch inference scales far better than CPU — see [benchmarks.md](benchmarks.md).
2. **Pick a smaller variant** (`osnet_x0_5` or `osnet_x0_25`) if extraction dominates your frame budget.
3. **Pass `frames` only when needed** — e.g. skip ReID on streams or frames where IoU tracking is sufficient.

Benchmark on your own hardware:

```bash
python tools/benchmark_reid.py --video test_videos/6387-191695740_medium.mp4 \
    --num-frames 200 --streams 1,2,4,8 --device cuda
```

## Related Documentation

- [Optimization Guide](optimization_guide.md) — profiling and tuning
- [Benchmarks](benchmarks.md) — measured performance
- [API Reference](api_reference.md) — complete API docs

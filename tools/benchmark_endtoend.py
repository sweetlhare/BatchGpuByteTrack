#!/usr/bin/env python3
"""
End-to-End Benchmark: YOLO Detection + Tracking (CPU vs GPU, with/without ReID)

Compares 4 configurations:
1) YOLO GPU + BYTETracker CPU + No ReID
2) YOLO GPU + BYTETracker CPU + ReID
3) YOLO GPU + BatchGPUTracker GPU + No ReID
4) YOLO GPU + BatchGPUTracker GPU + ReID

Usage:
    python tools/benchmark_endtoend.py --video path/to/video.mp4 --num-frames 200 --streams 1,2,4,8
"""

import argparse
import time
import sys
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import torch
import cv2

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bytetrack import BYTETracker
from bytetrack import BatchGPUTracker, TrackerConfig
from bytetrack.profiler import Profiler


class Args:
    """Arguments for BYTETracker (CPU version)"""
    def __init__(self):
        self.track_thresh = 0.5
        self.track_buffer = 30
        self.match_thresh = 0.8
        self.mot20 = False


def load_video_frames(video_path: str, num_frames: int) -> List[np.ndarray]:
    """Load frames from video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    print(f"Loaded {len(frames)} frames from {video_path}")
    return frames


def load_yolo_model(model_name: str = 'yolo26s.pt', device: str = 'cuda'):
    """
    Load YOLOv8 model using ultralytics.

    Args:
        model_name: Model name (yolov8n.pt, yolov8s.pt, yolov8m.pt, etc.)
        device: Device to run on ('cuda' or 'cpu')

    Returns:
        YOLO model
    """
    try:
        from ultralytics import YOLO
        model = YOLO(model_name)
        model.to(device)
        print(f"Loaded {model_name} on {device}")
        return model
    except ImportError:
        print("ERROR: ultralytics not installed. Install with: pip install ultralytics")
        sys.exit(1)


def run_yolo_detection(
    model,
    frames: List[np.ndarray],
    conf_threshold: float = 0.5,
    device: str = 'cuda',
    profiler: Profiler = None
) -> List[np.ndarray]:
    """
    Run YOLO detection on frames.

    Returns:
        List of detections per frame: [N, 5] arrays with [x1, y1, x2, y2, conf]
    """
    all_detections = []

    for frame in frames:
        if profiler:
            with profiler.profile("yolo_inference"):
                results = model(frame, conf=conf_threshold, verbose=False, device=device)
        else:
            results = model(frame, conf=conf_threshold, verbose=False, device=device)

        # Extract detections
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            # Format: [x1, y1, x2, y2, conf, class]
            dets = torch.cat([
                boxes.xyxy,  # x1, y1, x2, y2
                boxes.conf.unsqueeze(1),  # confidence
                boxes.cls.unsqueeze(1)  # class
            ], dim=1).cpu().numpy()

            # Filter for person class (class 0 in COCO)
            person_mask = dets[:, 5] == 0
            dets = dets[person_mask]

            # Keep only [x1, y1, x2, y2, conf] for tracker
            dets = dets[:, :5]
        else:
            dets = np.empty((0, 5), dtype=np.float32)

        all_detections.append(dets)

    return all_detections


def benchmark_cpu_tracker(
    detections: List[np.ndarray],
    frames: List[np.ndarray],
    num_streams: int,
    enable_reid: bool,
    device: str = 'cuda'
) -> Tuple[float, int, Profiler]:
    """
    Benchmark CPU-based BYTETracker (original).

    For multi-stream, we run separate tracker instances in parallel threads.
    """
    from concurrent.futures import ThreadPoolExecutor

    profiler = Profiler('cpu')
    num_frames = len(detections)

    # Create ONE shared ReID model if needed (critical optimization!)
    shared_reid_model = None
    if enable_reid:
        from bytetrack.osnet import build_osnet
        print(f"Loading shared ReID model: osnet_x0_5 on {device}")
        shared_reid_model = build_osnet('osnet_x0_5', pretrained=True, device=None)
        # Move to device manually to handle CUDA errors gracefully
        try:
            shared_reid_model = shared_reid_model.to(device)
        except (RuntimeError, AssertionError) as e:
            if 'cuda' in str(e).lower():
                print(f"Warning: CUDA not available ({e}), using CPU for ReID")
                shared_reid_model = shared_reid_model.to('cpu')
                device = 'cpu'  # Update device for consistency
            else:
                raise
        shared_reid_model.eval()

    # Create tracker instances (all share the same ReID model)
    args = Args()
    trackers = [BYTETracker(args, frame_rate=30, enable_reid=enable_reid,
                           reid_model=shared_reid_model,  # Share the model!
                           reid_threshold=0.5, lambda_emb=0.3, device=device,
                           profiler=profiler)  # Add profiler
                for _ in range(num_streams)]

    img_size = (frames[0].shape[0], frames[0].shape[1])  # H, W
    test_size = (640, 640)

    # Warmup (use .copy() to prevent in-place modification)
    for i in range(min(10, num_frames)):
        for tracker in trackers:
            _ = tracker.update(detections[i].copy(), img_size, test_size,
                             frame=frames[i] if enable_reid else None)

    # Reset trackers (still sharing the same ReID model)
    trackers = [BYTETracker(args, frame_rate=30, enable_reid=enable_reid,
                           reid_model=shared_reid_model,  # Share the model!
                           reid_threshold=0.5, lambda_emb=0.3, device=device,
                           profiler=profiler)  # Add profiler
                for _ in range(num_streams)]
    executor = ThreadPoolExecutor(max_workers=num_streams)

    def update_tracker(tracker, dets, frame):
        tracks = tracker.update(dets, img_size, test_size, frame=frame)
        return len(tracks)

    # Benchmark
    total_tracks = 0
    start_time = time.perf_counter()

    for frame_idx in range(num_frames):
        with profiler.profile("tracking_update"):
            # Run all trackers in parallel (realistic post-YOLO scenario)
            # Use .copy() to prevent in-place modification by BYTETracker
            futures = [executor.submit(update_tracker, tracker, detections[frame_idx].copy(),
                                      frames[frame_idx] if enable_reid else None)
                      for tracker in trackers]
            for f in futures:
                total_tracks += f.result()

    tracking_time = time.perf_counter() - start_time
    executor.shutdown(wait=False)

    return tracking_time, total_tracks, profiler


def benchmark_gpu_tracker(
    detections: List[np.ndarray],
    frames: List[np.ndarray],
    num_streams: int,
    enable_reid: bool,
    device: str = 'cuda'
) -> Tuple[float, int, Profiler]:
    """
    Benchmark GPU-based BatchGPUTracker.
    """
    profiler = Profiler(device)
    num_frames = len(detections)

    # Setup tracker
    config = TrackerConfig(
        track_thresh=0.5,
        match_thresh=0.8,
        track_buffer=30,
        enable_reid=enable_reid,
        reid_model_path='osnet_x0_5' if enable_reid else None,  # 2x faster than osnet_x1_0
        reid_embedding_dim=512,
        reid_threshold=0.5,
        lambda_emb=0.3,
        device=device
    )

    tracker = BatchGPUTracker(num_streams, config, profiler=profiler)

    # Warmup
    for i in range(min(10, num_frames)):
        batch_dets = [detections[i] for _ in range(num_streams)]
        batch_frames = [frames[i] for _ in range(num_streams)] if enable_reid else None
        _ = tracker.update(batch_dets, frames=batch_frames)

    # Reset tracker
    tracker.reset()

    # Benchmark
    if device == 'cuda':
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    total_tracks = 0
    for frame_idx in range(num_frames):
        batch_dets = [detections[frame_idx] for _ in range(num_streams)]
        batch_frames = [frames[frame_idx] for _ in range(num_streams)] if enable_reid else None

        with profiler.profile("tracking_update"):
            if enable_reid:
                with profiler.profile("reid_extraction"):
                    embeddings = tracker._extract_reid_features(batch_frames, batch_dets)
                results = tracker.update(batch_dets, frames=None, embeddings=embeddings)
            else:
                results = tracker.update(batch_dets)

        total_tracks += sum(len(r) for r in results)

    if device == 'cuda':
        torch.cuda.synchronize()
    tracking_time = time.perf_counter() - start_time

    return tracking_time, total_tracks, profiler


def benchmark_endtoend(
    video_path: str,
    num_frames: int,
    num_streams: int,
    yolo_model: str = 'yolo26s.pt',
    device: str = 'cuda'
):
    """
    Run end-to-end benchmark with all 4 configurations.
    """
    # Load frames
    frames = load_video_frames(video_path, num_frames)
    if len(frames) < num_frames:
        num_frames = len(frames)

    # Load YOLO model
    print(f"\nLoading YOLO model: {yolo_model}")
    yolo = load_yolo_model(yolo_model, device)

    # Warmup YOLO
    print("Warming up YOLO...")
    for i in range(min(10, num_frames)):
        _ = yolo(frames[i], conf=0.5, verbose=False, device=device)

    # Run YOLO detection ONCE (shared across all configs)
    print(f"Running YOLO detection on {num_frames} frames...")
    profiler = Profiler(device)

    if device == 'cuda':
        torch.cuda.synchronize()
    yolo_start = time.perf_counter()

    detections = run_yolo_detection(yolo, frames, conf_threshold=0.5, device=device, profiler=profiler)

    if device == 'cuda':
        torch.cuda.synchronize()
    yolo_time = time.perf_counter() - yolo_start

    avg_dets = np.mean([len(d) for d in detections])
    print(f"YOLO detection done. Avg detections/frame: {avg_dets:.1f}")
    print(f"YOLO time: {yolo_time:.3f}s ({yolo_time/num_frames*1000:.2f}ms/frame)")

    results = []

    # Configuration 1: CPU Tracker + No ReID
    print(f"\n{'='*80}")
    print("1) YOLO GPU + BYTETracker CPU + No ReID")
    print(f"{'='*80}")
    track_time, tracks, prof = benchmark_cpu_tracker(detections, frames, num_streams, enable_reid=False, device=device)
    total_time = yolo_time + track_time
    ms_per_frame = (total_time / num_frames) * 1000
    fps = num_frames / total_time

    print(f"\nResults:")
    print(f"  YOLO time: {yolo_time:.3f}s ({yolo_time/num_frames*1000:.2f}ms/frame)")
    print(f"  Tracking time: {track_time:.3f}s ({track_time/num_frames*1000:.2f}ms/frame)")
    print(f"  Total time: {total_time:.3f}s ({ms_per_frame:.2f}ms/frame)")
    print(f"  FPS: {fps:.1f}")
    print(f"  Total tracks: {tracks}")

    results.append({
        'config': 'CPU + No ReID',
        'yolo_time': yolo_time,
        'track_time': track_time,
        'total_time': total_time,
        'ms_per_frame': ms_per_frame,
        'fps': fps,
        'tracks': tracks
    })

    # Configuration 2: CPU Tracker + ReID
    print(f"\n{'='*80}")
    print("2) YOLO GPU + BYTETracker CPU + ReID")
    print(f"{'='*80}")
    track_time, tracks, prof = benchmark_cpu_tracker(detections, frames, num_streams, enable_reid=True, device=device)
    total_time = yolo_time + track_time
    ms_per_frame = (total_time / num_frames) * 1000
    fps = num_frames / total_time

    print(f"\nResults:")
    print(f"  YOLO time: {yolo_time:.3f}s ({yolo_time/num_frames*1000:.2f}ms/frame)")
    print(f"  Tracking time: {track_time:.3f}s ({track_time/num_frames*1000:.2f}ms/frame)")
    print(f"  Total time: {total_time:.3f}s ({ms_per_frame:.2f}ms/frame)")
    print(f"  FPS: {fps:.1f}")
    print(f"  Total tracks: {tracks}")

    prof.print_summary("Profiling Breakdown")

    results.append({
        'config': 'CPU + ReID',
        'yolo_time': yolo_time,
        'track_time': track_time,
        'total_time': total_time,
        'ms_per_frame': ms_per_frame,
        'fps': fps,
        'tracks': tracks
    })

    # Configuration 3: GPU Tracker + No ReID
    print(f"\n{'='*80}")
    print("3) YOLO GPU + BatchGPUTracker GPU + No ReID")
    print(f"{'='*80}")
    track_time, tracks, prof = benchmark_gpu_tracker(detections, frames, num_streams, enable_reid=False, device=device)
    total_time = yolo_time + track_time
    ms_per_frame = (total_time / num_frames) * 1000
    fps = num_frames / total_time

    print(f"\nResults:")
    print(f"  YOLO time: {yolo_time:.3f}s ({yolo_time/num_frames*1000:.2f}ms/frame)")
    print(f"  Tracking time: {track_time:.3f}s ({track_time/num_frames*1000:.2f}ms/frame)")
    print(f"  Total time: {total_time:.3f}s ({ms_per_frame:.2f}ms/frame)")
    print(f"  FPS: {fps:.1f}")
    print(f"  Total tracks: {tracks}")

    prof.print_summary("Profiling Breakdown")

    results.append({
        'config': 'GPU + No ReID',
        'yolo_time': yolo_time,
        'track_time': track_time,
        'total_time': total_time,
        'ms_per_frame': ms_per_frame,
        'fps': fps,
        'tracks': tracks
    })

    # Configuration 4: GPU Tracker + ReID
    print(f"\n{'='*80}")
    print("4) YOLO GPU + BatchGPUTracker GPU + ReID")
    print(f"{'='*80}")
    track_time, tracks, prof = benchmark_gpu_tracker(detections, frames, num_streams, enable_reid=True, device=device)
    total_time = yolo_time + track_time
    ms_per_frame = (total_time / num_frames) * 1000
    fps = num_frames / total_time

    print(f"\nResults:")
    print(f"  YOLO time: {yolo_time:.3f}s ({yolo_time/num_frames*1000:.2f}ms/frame)")
    print(f"  Tracking time: {track_time:.3f}s ({track_time/num_frames*1000:.2f}ms/frame)")
    print(f"  Total time: {total_time:.3f}s ({ms_per_frame:.2f}ms/frame)")
    print(f"  FPS: {fps:.1f}")
    print(f"  Total tracks: {tracks}")

    prof.print_summary("Profiling Breakdown")

    results.append({
        'config': 'GPU + ReID',
        'yolo_time': yolo_time,
        'track_time': track_time,
        'total_time': total_time,
        'ms_per_frame': ms_per_frame,
        'fps': fps,
        'tracks': tracks
    })

    return results


def main():
    parser = argparse.ArgumentParser(description="End-to-End YOLO + Tracking Benchmark")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--num-frames", type=int, default=200, help="Number of frames")
    parser.add_argument("--streams", type=str, default="1,2,4,8,16,32,64",
                        help="Comma-separated list of stream counts")
    parser.add_argument("--yolo-model", type=str, default="yolo26s.pt",
                        help="YOLO model (yolo26s.pt, yolov8n.pt, yolov8s.pt, etc.)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    args = parser.parse_args()

    stream_counts = [int(x) for x in args.streams.split(",")]

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    if args.device == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        torch.cuda.empty_cache()

    print("=" * 80)
    print("End-to-End Benchmark: YOLO Detection + Tracking")
    print("=" * 80)
    print(f"Video: {args.video}")
    print(f"Frames: {args.num_frames}")
    print(f"YOLO Model: {args.yolo_model}")
    print(f"Device: {args.device}")
    print(f"Stream counts: {stream_counts}")
    print()

    all_results = {}

    for num_streams in stream_counts:
        print("\n" + "=" * 80)
        print(f"Testing with {num_streams} streams")
        print("=" * 80)

        results = benchmark_endtoend(
            args.video, args.num_frames, num_streams,
            yolo_model=args.yolo_model, device=args.device
        )

        all_results[num_streams] = results

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY - End-to-End Performance")
    print("=" * 80)

    for num_streams in stream_counts:
        print(f"\n>>> {num_streams} Streams <<<")
        print(f"{'Config':<25} {'YOLO (ms)':<12} {'Track (ms)':<12} {'Total (ms)':<12} {'FPS':<10} {'Speedup':<10}")
        print("-" * 95)

        # Get baseline (CPU + No ReID) tracking time
        baseline_track_time = None
        for r in all_results[num_streams]:
            if r['config'] == 'CPU + No ReID':
                baseline_track_time = r['track_time'] / args.num_frames * 1000
                break

        for r in all_results[num_streams]:
            yolo_ms = r['yolo_time'] / args.num_frames * 1000
            track_ms = r['track_time'] / args.num_frames * 1000

            # Calculate speedup vs baseline (CPU + No ReID)
            if baseline_track_time and baseline_track_time > 0:
                speedup = baseline_track_time / track_ms
                speedup_str = f"{speedup:.2f}x"
            else:
                speedup_str = "1.00x"

            print(f"{r['config']:<25} {yolo_ms:<12.2f} {track_ms:<12.2f} "
                  f"{r['ms_per_frame']:<12.2f} {r['fps']:<10.1f} {speedup_str:<10}")

        # Print GPU vs CPU comparison
        print("\nGPU vs CPU Tracking Speedup:")
        cpu_no_reid = next((r for r in all_results[num_streams] if r['config'] == 'CPU + No ReID'), None)
        gpu_no_reid = next((r for r in all_results[num_streams] if r['config'] == 'GPU + No ReID'), None)
        cpu_reid = next((r for r in all_results[num_streams] if r['config'] == 'CPU + ReID'), None)
        gpu_reid = next((r for r in all_results[num_streams] if r['config'] == 'GPU + ReID'), None)

        if cpu_no_reid and gpu_no_reid:
            cpu_track_ms = cpu_no_reid['track_time'] / args.num_frames * 1000
            gpu_track_ms = gpu_no_reid['track_time'] / args.num_frames * 1000
            speedup = cpu_track_ms / gpu_track_ms
            print(f"  No ReID:  GPU is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than CPU ({gpu_track_ms:.2f}ms vs {cpu_track_ms:.2f}ms)")

        if cpu_reid and gpu_reid:
            cpu_track_ms = cpu_reid['track_time'] / args.num_frames * 1000
            gpu_track_ms = gpu_reid['track_time'] / args.num_frames * 1000
            speedup = cpu_track_ms / gpu_track_ms
            print(f"  With ReID: GPU is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than CPU ({gpu_track_ms:.2f}ms vs {cpu_track_ms:.2f}ms)")

    # GPU memory
    if args.device == 'cuda' and torch.cuda.is_available():
        print(f"\nGPU Memory:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print(f"  Cached: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Benchmark comparison: Original BYTETracker vs BatchGPUTracker
With detailed profiling of all operations.

Usage:
    python tools/benchmark_comparison.py --video path/to/video.mp4 --num-frames 200
"""

import argparse
import time
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import cv2

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bytetrack import BatchGPUTracker, BYTETracker, Profiler, TrackerConfig


class Args:
    """Arguments for BYTETracker"""
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


def generate_fake_detections(frame: np.ndarray, num_dets: int = 30, seed: int = None) -> np.ndarray:
    """
    Generate fake detections for a frame.
    In real usage, you would run YOLOX here.

    Returns: [N, 5] array with [x1, y1, x2, y2, score]
    """
    if seed is not None:
        np.random.seed(seed)

    h, w = frame.shape[:2]

    # Random box centers
    cx = np.random.uniform(50, w - 50, num_dets)
    cy = np.random.uniform(50, h - 50, num_dets)

    # Random box sizes
    bw = np.random.uniform(30, 150, num_dets)
    bh = np.random.uniform(50, 200, num_dets)

    # Convert to tlbr
    x1 = np.clip(cx - bw / 2, 0, w)
    y1 = np.clip(cy - bh / 2, 0, h)
    x2 = np.clip(cx + bw / 2, 0, w)
    y2 = np.clip(cy + bh / 2, 0, h)

    # Random scores
    scores = np.random.uniform(0.3, 1.0, num_dets)

    return np.stack([x1, y1, x2, y2, scores], axis=1).astype(np.float32)


def benchmark_original_bytetrack_profiled(
    detections_per_frame: List[np.ndarray],
    num_streams: int,
    img_size: Tuple[int, int],
    device: str = 'cuda'
) -> Tuple[float, int, Profiler]:
    """
    Benchmark original BYTETracker with parallel threads (realistic scenario).
    Simulates: YOLOX batch detection → parallel tracker updates.
    """
    from concurrent.futures import ThreadPoolExecutor

    profiler = Profiler(device)
    args = Args()

    num_frames = len(detections_per_frame)
    img_info = img_size

    # Warmup
    trackers = [BYTETracker(args) for _ in range(num_streams)]
    for _ in range(min(10, num_frames)):
        for tracker in trackers:
            tracker.update(detections_per_frame[0], img_info, img_size)

    # Reset
    trackers = [BYTETracker(args) for _ in range(num_streams)]
    total_tracks = [0]  # Use list for mutable in closure
    executor = ThreadPoolExecutor(max_workers=num_streams)

    def update_tracker(tracker, dets):
        tracks = tracker.update(dets, img_info, img_size)
        return len(tracks)

    # Benchmark with profiling
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    for frame_idx in range(num_frames):
        dets = detections_per_frame[frame_idx]

        with profiler.profile("total_frame"):
            # Run all trackers in parallel (simulating post-YOLOX batch scenario)
            with profiler.profile("parallel_tracker_updates"):
                futures = [executor.submit(update_tracker, tracker, dets) for tracker in trackers]
                for f in futures:
                    total_tracks[0] += f.result()

    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.perf_counter() - t_start

    executor.shutdown(wait=False)
    return total_time, total_tracks[0], profiler


def benchmark_batch_gpu_tracker_detailed(
    detections_per_frame: List[np.ndarray],
    num_streams: int,
    device: str = 'cuda'
) -> Tuple[float, int, Profiler]:
    """
    Benchmark BatchGPUTracker with detailed internal profiling.
    """
    profiler = Profiler(device)

    config = TrackerConfig(
        track_thresh=0.5,
        match_thresh=0.8,
        track_buffer=30,
        device=device
    )
    tracker = BatchGPUTracker(num_streams, config, profiler=profiler)

    num_frames = len(detections_per_frame)
    total_tracks = 0

    # Warmup
    for _ in range(min(10, num_frames)):
        batch_dets = [detections_per_frame[0] for _ in range(num_streams)]
        tracker.update(batch_dets)

    # Reset tracker for benchmark
    tracker.reset()
    profiler.reset()

    # Benchmark
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    for frame_idx in range(num_frames):
        dets = detections_per_frame[frame_idx]
        batch_dets = [dets.copy() for _ in range(num_streams)]
        results = tracker.update(batch_dets)
        total_tracks += sum(len(r) for r in results)

    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.perf_counter() - t_start

    return total_time, total_tracks, profiler


def run_comparison(
    video_path: str,
    num_frames: int,
    stream_counts: List[int],
    device: str = 'cuda',
    num_dets: int = 30
):
    """Run full comparison benchmark with profiling."""

    print("=" * 80)
    print("ByteTrack vs BatchGPUTracker Comparison Benchmark (with Profiling)")
    print("=" * 80)

    # Load video
    frames = load_video_frames(video_path, num_frames)
    if len(frames) < num_frames:
        print(f"Warning: Only {len(frames)} frames available")
        num_frames = len(frames)

    img_size = (frames[0].shape[0], frames[0].shape[1])
    print(f"Video size: {img_size[1]}x{img_size[0]}")
    print(f"Frames: {num_frames}")
    print(f"Device: {device}")

    # Generate detections
    print("\nGenerating detections...")
    detections = []
    for i in range(num_frames):
        dets = generate_fake_detections(frames[i], num_dets=num_dets, seed=i * 1000)
        detections.append(dets)

    avg_dets = np.mean([len(d) for d in detections])
    print(f"Avg detections/frame: {avg_dets:.1f}")

    results = []

    print("\n" + "=" * 80)
    print(f"{'Streams':<10} {'Tracker':<20} {'Time (s)':<12} {'ms/frame':<12} {'FPS':<10}")
    print("=" * 80)

    for num_streams in stream_counts:
        print(f"\n>>> Testing with {num_streams} streams <<<")

        # Benchmark Original BYTETracker
        print(f"\n[BYTETracker x{num_streams}]")
        try:
            time_orig, tracks_orig, profiler_orig = benchmark_original_bytetrack_profiled(
                detections, num_streams, img_size, device
            )
            ms_per_frame_orig = (time_orig / num_frames) * 1000
            fps_orig = num_frames / time_orig

            total_dets_orig = num_frames * num_streams * avg_dets
            print(f"  Total time: {time_orig:.3f}s, {ms_per_frame_orig:.2f}ms/frame, {fps_orig:.1f} FPS")
            print(f"  Data: {num_frames} frames × {num_streams} streams × {avg_dets:.0f} dets = {total_dets_orig:.0f} det updates")
            print(f"  Output tracks: {tracks_orig}")
            profiler_orig.print_summary(f"  BYTETracker Profiling ({num_streams} streams)")

            results.append({
                'streams': num_streams,
                'tracker': 'BYTETracker',
                'time': time_orig,
                'ms_per_frame': ms_per_frame_orig,
                'fps': fps_orig,
                'tracks': tracks_orig
            })
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

        # Benchmark BatchGPUTracker (detailed)
        print(f"\n[BatchGPUTracker x{num_streams}]")
        try:
            time_batch, tracks_batch, profiler_batch = benchmark_batch_gpu_tracker_detailed(
                detections, num_streams, device
            )
            ms_per_frame_batch = (time_batch / num_frames) * 1000
            fps_batch = num_frames / time_batch

            total_dets_batch = num_frames * num_streams * avg_dets
            print(f"  Total time: {time_batch:.3f}s, {ms_per_frame_batch:.2f}ms/frame, {fps_batch:.1f} FPS")
            print(f"  Data: {num_frames} frames × {num_streams} streams × {avg_dets:.0f} dets = {total_dets_batch:.0f} det updates")
            print(f"  Output tracks: {tracks_batch}")
            profiler_batch.print_summary(f"  BatchGPUTracker Profiling ({num_streams} streams)")

            results.append({
                'streams': num_streams,
                'tracker': 'BatchGPUTracker',
                'time': time_batch,
                'ms_per_frame': ms_per_frame_batch,
                'fps': fps_batch,
                'tracks': tracks_batch
            })

            # Speedup (if we have orig results)
            orig_res = [r for r in results if r['streams'] == num_streams and r['tracker'] == 'BYTETracker']
            if orig_res:
                speedup = orig_res[0]['time'] / time_batch
                print(f"\n  >>> SPEEDUP: {speedup:.2f}x <<<")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

        print("-" * 80)

    # Final Summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print(f"\n{'Streams':<10} {'BYTETracker':<20} {'BatchGPU':<20} {'Speedup':<12} {'Status':<6}")
    print(f"{'':10} {'(ms/frame)':<20} {'(ms/frame)':<20}")
    print("-" * 70)

    for num_streams in stream_counts:
        orig_results = [r for r in results if r['streams'] == num_streams and r['tracker'] == 'BYTETracker']
        batch_results = [r for r in results if r['streams'] == num_streams and r['tracker'] == 'BatchGPUTracker']

        if orig_results and batch_results:
            orig_ms = orig_results[0]['ms_per_frame']
            batch_ms = batch_results[0]['ms_per_frame']
            speedup = orig_ms / batch_ms if batch_ms > 0 else 0
            status = "WIN" if speedup > 1.0 else "LOSE"
            print(f"{num_streams:<10} {orig_ms:<20.2f} {batch_ms:<20.2f} {speedup:<12.2f}x {status:<6}")

    # GPU memory
    if device == 'cuda' and torch.cuda.is_available():
        print(f"\nGPU Memory:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print(f"  Cached: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")

    return results


def main():
    parser = argparse.ArgumentParser(description="ByteTrack Comparison Benchmark")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--num-frames", type=int, default=200, help="Number of frames")
    parser.add_argument("--streams", type=str, default="1,2,4,8,16,32,64",
                        help="Comma-separated list of stream counts")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num-dets", type=int, default=30,
                        help="Synthetic detections per stream per frame")

    args = parser.parse_args()

    stream_counts = [int(x) for x in args.streams.split(",")]

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    if args.device == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        torch.cuda.empty_cache()

    run_comparison(args.video, args.num_frames, stream_counts, args.device, args.num_dets)


if __name__ == "__main__":
    main()

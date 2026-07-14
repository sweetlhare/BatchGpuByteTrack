#!/usr/bin/env python3
"""
Benchmark ReID overhead: BatchGPUTracker with and without Re-identification.

Measures how much OSNet feature extraction and appearance fusion add on top
of pure IoU tracking, per stream count.

Usage:
    python tools/benchmark_reid.py --video path/to/video.mp4 --num-frames 200 --streams 1,2,4,8
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

from bytetrack import BatchGPUTracker, Profiler, TrackerConfig


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


def benchmark_batch_gpu_tracker_reid(
    detections_per_frame: List[np.ndarray],
    frames_list: List[np.ndarray],
    num_streams: int,
    enable_reid: bool = True,
    device: str = 'cuda'
) -> Tuple[float, int, Profiler]:
    """
    Benchmark BatchGPUTracker with or without ReID.
    """
    profiler = Profiler(device)

    config = TrackerConfig(
        track_thresh=0.5,
        match_thresh=0.8,
        track_buffer=30,
        enable_reid=enable_reid,
        reid_model_path='osnet_x0_5' if enable_reid else None,  # 2x faster than osnet_x1_0
        reid_embedding_dim=512,
        reid_threshold=0.5,
        lambda_emb=0.3,
        device=device,
    )
    tracker = BatchGPUTracker(num_streams, config, profiler=profiler)

    num_frames = len(detections_per_frame)
    total_tracks = 0

    # Warmup
    for i in range(min(10, num_frames)):
        batch_dets = [detections_per_frame[i] for _ in range(num_streams)]
        batch_frames = [frames_list[i] for _ in range(num_streams)] if enable_reid else None
        tracker.update(batch_dets, frames=batch_frames)

    # Reset tracker for benchmark
    tracker.reset()
    profiler.reset()

    # Benchmark
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    for frame_idx in range(num_frames):
        dets = detections_per_frame[frame_idx]
        frame = frames_list[frame_idx]

        batch_dets = [dets.copy() for _ in range(num_streams)]
        batch_frames = [frame for _ in range(num_streams)] if enable_reid else None

        results = tracker.update(batch_dets, frames=batch_frames)
        total_tracks += sum(len(r) for r in results)

    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.perf_counter() - t_start

    return total_time, total_tracks, profiler


def run_reid_comparison(
    video_path: str,
    num_frames: int,
    stream_counts: List[int],
    device: str = 'cuda'
):
    """Run ReID-enabled benchmark comparison."""

    print("=" * 80)
    print("BatchGPUTracker ReID Overhead Benchmark")
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
        dets = generate_fake_detections(frames[i], num_dets=30, seed=i * 1000)
        detections.append(dets)

    avg_dets = np.mean([len(d) for d in detections])
    print(f"Avg detections/frame: {avg_dets:.1f}")
    print(f"ReID model: OSNet x0.5 (auto-download pretrained weights)")
    print(f"ReID embedding dimension: 512")

    results = []

    print("\n" + "=" * 80)
    print(f"{'Streams':<10} {'Tracker':<25} {'Time (s)':<12} {'ms/frame':<12} {'FPS':<10}")
    print("=" * 80)

    for num_streams in stream_counts:
        print(f"\n>>> Testing with {num_streams} streams <<<")

        # BatchGPUTracker WITHOUT ReID
        print(f"\n[BatchGPU x{num_streams} - NO ReID]")
        try:
            time_no_reid, tracks_no_reid, profiler_no_reid = benchmark_batch_gpu_tracker_reid(
                detections, frames, num_streams, enable_reid=False, device=device
            )
            ms_per_frame = (time_no_reid / num_frames) * 1000
            fps = num_frames / time_no_reid

            print(f"  Total time: {time_no_reid:.3f}s, {ms_per_frame:.2f}ms/frame, {fps:.1f} FPS")
            print(f"  Output tracks: {tracks_no_reid}")
            profiler_no_reid.print_summary(f"  BatchGPU Profiling - NO ReID ({num_streams} streams)")

            results.append({
                'streams': num_streams,
                'tracker': 'BatchGPU (No ReID)',
                'time': time_no_reid,
                'ms_per_frame': ms_per_frame,
                'fps': fps,
                'tracks': tracks_no_reid
            })
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

        # BatchGPUTracker WITH ReID
        print(f"\n[BatchGPU x{num_streams} - WITH ReID]")
        try:
            time_reid, tracks_reid, profiler_reid = benchmark_batch_gpu_tracker_reid(
                detections, frames, num_streams, enable_reid=True, device=device
            )
            ms_per_frame = (time_reid / num_frames) * 1000
            fps = num_frames / time_reid

            print(f"  Total time: {time_reid:.3f}s, {ms_per_frame:.2f}ms/frame, {fps:.1f} FPS")
            print(f"  Output tracks: {tracks_reid}")
            profiler_reid.print_summary(f"  BatchGPU Profiling - WITH ReID ({num_streams} streams)")

            results.append({
                'streams': num_streams,
                'tracker': 'BatchGPU (ReID)',
                'time': time_reid,
                'ms_per_frame': ms_per_frame,
                'fps': fps,
                'tracks': tracks_reid
            })

            # Calculate ReID overhead
            no_reid_res = [r for r in results if r['streams'] == num_streams and r['tracker'] == 'BatchGPU (No ReID)']
            if no_reid_res:
                overhead = ((time_reid - time_no_reid) / time_no_reid) * 100
                print(f"\n  >>> ReID Overhead: {overhead:.1f}% ({time_reid - time_no_reid:.3f}s) <<<")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

        print("-" * 80)

    # Final Summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY - ReID Impact")
    print("=" * 80)

    print(f"\n{'Streams':<10} {'No ReID (ms)':<15} {'ReID (ms)':<15} {'Overhead':<12} {'Status':<10}")
    print("-" * 70)

    for num_streams in stream_counts:
        no_reid_results = [r for r in results if r['streams'] == num_streams and r['tracker'] == 'BatchGPU (No ReID)']
        reid_results = [r for r in results if r['streams'] == num_streams and r['tracker'] == 'BatchGPU (ReID)']

        if no_reid_results and reid_results:
            no_reid_ms = no_reid_results[0]['ms_per_frame']
            reid_ms = reid_results[0]['ms_per_frame']
            overhead = ((reid_ms - no_reid_ms) / no_reid_ms) * 100
            status = "ACCEPTABLE" if overhead < 20 else "HIGH"
            print(f"{num_streams:<10} {no_reid_ms:<15.2f} {reid_ms:<15.2f} {overhead:<11.1f}% {status:<10}")

    # GPU memory
    if device == 'cuda' and torch.cuda.is_available():
        print(f"\nGPU Memory:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print(f"  Cached: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")

    return results


def main():
    parser = argparse.ArgumentParser(description="ByteTrack ReID Benchmark")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--num-frames", type=int, default=200, help="Number of frames")
    parser.add_argument("--streams", type=str, default="1,2,4,8,16",
                        help="Comma-separated list of stream counts")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    args = parser.parse_args()

    stream_counts = [int(x) for x in args.streams.split(",")]

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    if args.device == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        torch.cuda.empty_cache()

    run_reid_comparison(args.video, args.num_frames, stream_counts, args.device)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
GPU Tracking with ReID Example - Real YOLO Detection

Demonstrates GPU-accelerated tracking with Re-Identification features.
Uses real YOLO detection and OSNet for appearance features.

Requirements:
    pip install -e .  # Install package in development mode
    # OR
    pip install ultralytics lap scipy torch

Usage:
    python examples/gpu_with_reid.py --video video.mp4 --streams 4 --conf 0.5
"""

import sys
from pathlib import Path

# Add project root to path (allows running without installation)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import torch
import numpy as np
import time
import argparse
from bytetrack import BatchGPUTracker, Profiler, TrackerConfig

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics not installed")
    print("Install with: pip install ultralytics")
    exit(1)


def main():
    parser = argparse.ArgumentParser(description='GPU Tracking with ReID')
    parser.add_argument('--video', type=str, default='test_videos/6387-191695740_medium.mp4',
                       help='Path to input video')
    parser.add_argument('--streams', type=int, default=4,
                       help='Number of streams (4 recommended with ReID)')
    parser.add_argument('--model', type=str, default='yolo26s.pt',
                       help='YOLO model')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Detection confidence threshold')
    parser.add_argument('--reid-model', type=str, default='osnet_x0_5',
                       choices=['osnet_x1_0', 'osnet_x0_75', 'osnet_x0_5', 'osnet_x0_25'],
                       help='OSNet ReID model')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='YOLO batch size')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable display (for headless/SSH mode)')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum frames to process (for testing)')
    args = parser.parse_args()

    # Check CUDA
    if not torch.cuda.is_available():
        print("Error: CUDA not available")
        return

    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # Load YOLO detector
    print(f"Loading YOLO model: {args.model}")
    yolo = YOLO(args.model)
    yolo.to('cuda')
    yolo.fuse()

    # Configuration with ReID enabled
    num_streams = args.streams
    config = TrackerConfig(
        track_thresh=0.6,
        track_buffer=30,
        match_thresh=0.8,
        enable_reid=True,              # Enable ReID
        reid_model_path=args.reid_model, # Balanced model
        reid_threshold=0.6,            # Embedding similarity threshold
        lambda_emb=0.3,            # Equal weight for IoU and ReID
        device='cuda'
    )

    # Initialize tracker (profiler gives the real ReID/tracking time split)
    print(f"Loading OSNet ReID model: {args.reid_model}...")
    profiler = Profiler(device='cuda')
    tracker = BatchGPUTracker(num_streams, config, profiler=profiler)
    print("Tracker ready!")

    # Open video streams
    caps = [cv2.VideoCapture(args.video) for _ in range(num_streams)]

    if not all(cap.isOpened() for cap in caps):
        print("Error: Cannot open video streams")
        return

    width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Processing {num_streams} streams with ReID...")
    print("ReID helps maintain track IDs across occlusions")
    print("Press 'q' to quit")

    frame_id = 0
    total_detection_time = 0
    total_tracking_time = 0

    # Display grid
    grid_rows = 2
    grid_cols = num_streams // grid_rows

    while True:
        frames = []

        # Read from all streams (keep BGR: the tracker converts crops for ReID itself)
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()

            frames.append(frame.copy())

        frame_id += 1

        # Stop if max_frames reached
        if args.max_frames and frame_id > args.max_frames:
            break

        # Batch YOLO detection
        det_start = time.time()

        all_detections = []
        for i in range(0, num_streams, args.batch_size):
            batch_frames = frames[i:i+args.batch_size]
            results = yolo(batch_frames, conf=args.conf, verbose=False)

            for result in results:
                if len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    scores = result.boxes.conf.cpu().numpy().reshape(-1, 1)
                    detections = np.concatenate([boxes, scores], axis=1)
                else:
                    detections = np.empty((0, 5), dtype=np.float32)
                all_detections.append(detections)

        det_time = time.time() - det_start
        total_detection_time += det_time

        # Track with ReID (includes feature extraction)
        track_start = time.time()
        all_tracks = tracker.update(all_detections, frames=frames)
        track_time = time.time() - track_start
        total_tracking_time += track_time

        # Measured ReID/tracking split from the profiler
        reid_time = profiler.get_mean_time('0_reid_extraction') / 1000
        pure_track_time = max(track_time - reid_time, 0)

        # Visualize
        display_frames = []
        for stream_idx, (frame, tracks, dets) in enumerate(zip(frames, all_tracks, all_detections)):
            # Draw detections in blue
            for det in dets:
                x1, y1, x2, y2, score = det
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                             (255, 0, 0), 1)

            # Draw tracks with colors based on global ID
            for track in tracks:
                track_id = track['track_id']
                bbox = track['tlbr']
                x1, y1, x2, y2 = map(int, bbox)

                # Color based on track ID (for visualization)
                np.random.seed(track_id)
                color = tuple(map(int, np.random.randint(0, 255, 3)))

                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Draw ID with background
                label = f'ID:{track_id}'
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (x1, y1 - label_h - 5), (x1 + label_w, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Stream info
            info = f'Stream {stream_idx} | D:{len(dets)} T:{len(tracks)}'
            cv2.putText(frame, info, (5, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Resize for display
            display_frame = cv2.resize(frame, (640, 360))
            display_frames.append(display_frame)

        # Create grid
        rows = []
        for i in range(grid_rows):
            row_frames = display_frames[i*grid_cols:(i+1)*grid_cols]
            if len(row_frames) == grid_cols:
                row = np.hstack(row_frames)
                rows.append(row)

        if len(rows) == grid_rows:
            grid = np.vstack(rows)

            # Overall info
            total_time = det_time + track_time
            fps = 1 / total_time if total_time > 0 else 0
            avg_fps = frame_id / (total_detection_time + total_tracking_time)

            total_tracks_count = sum(len(t) for t in all_tracks)

            info_text = f'Frame: {frame_id} | Det: {det_time*1000:.1f}ms + Track: {pure_track_time*1000:.1f}ms + ReID: {reid_time*1000:.1f}ms = {total_time*1000:.1f}ms | FPS: {fps:.1f}'
            cv2.putText(grid, info_text, (10, grid.shape[0] - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            reid_info = f'ReID: {args.reid_model} | Tracks: {total_tracks_count} | Global IDs | Avg FPS: {avg_fps:.1f}'
            cv2.putText(grid, reid_info, (10, grid.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if not args.no_display:
                cv2.imshow('Multi-Stream Tracking with ReID', grid)

        # Headless progress
        if args.no_display and frame_id % 10 == 0:
            print(f"Frame {frame_id}: {sum(len(d) for d in all_detections)} dets, {sum(len(t) for t in all_tracks)} tracks")

        if not args.no_display and cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()

    # Summary
    avg_det = (total_detection_time / frame_id * 1000) if frame_id > 0 else 0
    avg_track = (total_tracking_time / frame_id * 1000) if frame_id > 0 else 0
    avg_reid = profiler.get_mean_time('0_reid_extraction')
    avg_pure_track = max(avg_track - avg_reid, 0)
    avg_total = avg_det + avg_track
    avg_fps = frame_id / (total_detection_time + total_tracking_time)

    print(f"\n{'='*60}")
    print(f"Performance Summary")
    print(f"{'='*60}")
    print(f"Streams: {num_streams}")
    print(f"Frames: {frame_id}")
    print(f"ReID Model: {args.reid_model}")
    print(f"\nTiming breakdown per batch:")
    print(f"  Detection:    {avg_det:.2f}ms ({avg_det/avg_total*100:.1f}%)")
    print(f"  Tracking:     {avg_pure_track:.2f}ms ({avg_pure_track/avg_total*100:.1f}%)")
    print(f"  ReID Extract: {avg_reid:.2f}ms ({avg_reid/avg_total*100:.1f}%)")
    print(f"  Total:        {avg_total:.2f}ms")
    print(f"  Batch FPS:    {avg_fps:.1f}")
    print(f"  Per-stream:   {avg_fps/num_streams:.1f} FPS")
    print(f"\nReID Overhead:")
    print(f"  ReID adds ~{avg_reid:.1f}ms per batch")
    print(f"  Still {'real-time' if avg_fps/num_streams > 30 else 'acceptable'} for {num_streams} streams")


if __name__ == '__main__':
    main()

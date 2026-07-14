#!/usr/bin/env python3
"""
GPU Multi-Stream Tracking Example with Real YOLO Detector

Demonstrates GPU-accelerated tracking for multiple video streams simultaneously.
Shows efficient batch processing with real YOLO detection.

Requirements:
    pip install -e .  # Install package in development mode
    # OR
    pip install ultralytics lap scipy torch

Usage:
    python examples/gpu_multistream.py --video video.mp4 --streams 8 --conf 0.5
"""

import sys
from pathlib import Path

# Add project root to path (allows running without installation)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import torch
import time
import numpy as np
import argparse
from bytetrack import BatchGPUTracker, TrackerConfig

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics not installed")
    print("Install with: pip install ultralytics")
    exit(1)


def main():
    parser = argparse.ArgumentParser(description='GPU Multi-Stream Tracking with YOLO')
    parser.add_argument('--video', type=str, default='test_videos/6387-191695740_medium.mp4',
                       help='Path to input video (will be duplicated for multiple streams)')
    parser.add_argument('--streams', type=int, default=8,
                       help='Number of streams to process')
    parser.add_argument('--model', type=str, default='yolo26s.pt',
                       help='YOLO model')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Detection confidence threshold')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='YOLO batch size for detection')
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

    # Load YOLO detector on GPU
    print(f"Loading YOLO model: {args.model}")
    yolo = YOLO(args.model)
    yolo.to('cuda')
    yolo.fuse()

    # Configuration
    num_streams = args.streams
    config = TrackerConfig(
        track_thresh=0.6,
        track_buffer=30,
        match_thresh=0.8,
        enable_reid=False,  # Disable for speed
        device='cuda'
    )

    # Initialize tracker
    tracker = BatchGPUTracker(num_streams, config)

    # Open multiple video captures
    # In practice, these would be different video files or camera streams
    caps = [cv2.VideoCapture(args.video) for _ in range(num_streams)]

    # Check all opened
    if not all(cap.isOpened() for cap in caps):
        print("Error: Cannot open all video streams")
        return

    # Get video properties
    width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Processing {num_streams} streams simultaneously...")
    print("Press 'q' to quit")

    frame_id = 0
    total_detection_time = 0
    total_tracking_time = 0

    # Create display grid
    grid_rows = 2
    grid_cols = num_streams // grid_rows

    while True:
        frames = []

        # Read from all streams
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                # End of video, reset
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            frames.append(frame)

        frame_id += 1

        # Stop if max_frames reached
        if args.max_frames and frame_id > args.max_frames:
            break

        # Batch YOLO detection on all frames
        det_start = time.time()

        # Process frames in batches for efficiency
        all_detections = []
        for i in range(0, num_streams, args.batch_size):
            batch_frames = frames[i:i+args.batch_size]

            # YOLO batch inference
            results = yolo(batch_frames, conf=args.conf, verbose=False)

            # Convert each result to numpy array
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

        # Track all streams in one batch
        track_start = time.time()
        all_tracks = tracker.update(all_detections)
        track_time = time.time() - track_start
        total_tracking_time += track_time

        # Visualize each stream
        display_frames = []
        for stream_idx, (frame, tracks, dets) in enumerate(zip(frames, all_tracks, all_detections)):
            # Draw detections in blue
            for det in dets:
                x1, y1, x2, y2, score = det
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                             (255, 0, 0), 1)

            # Draw tracks in green
            for track in tracks:
                track_id = track['track_id']
                x1, y1, x2, y2 = map(int, track['tlbr'])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'ID:{track_id}', (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            # Add stream info
            info = f'S{stream_idx} | D:{len(dets)} T:{len(tracks)}'
            cv2.putText(frame, info, (5, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Resize for display
            display_frame = cv2.resize(frame, (480, 270))
            display_frames.append(display_frame)

        # Create grid display
        rows = []
        for i in range(grid_rows):
            row_frames = display_frames[i*grid_cols:(i+1)*grid_cols]
            if len(row_frames) == grid_cols:
                row = np.hstack(row_frames)
                rows.append(row)

        if len(rows) == grid_rows:
            grid = np.vstack(rows)

            # Add overall info
            total_time = det_time + track_time
            fps = 1 / total_time if total_time > 0 else 0
            avg_fps = frame_id / (total_detection_time + total_tracking_time)

            total_dets = sum(len(d) for d in all_detections)
            total_tracks_count = sum(len(t) for t in all_tracks)

            info_text = f'Frame: {frame_id} | Det: {det_time*1000:.1f}ms | Track: {track_time*1000:.1f}ms | Total: {total_time*1000:.1f}ms | FPS: {fps:.1f}'
            cv2.putText(grid, info_text, (10, grid.shape[0] - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            info2 = f'Streams: {num_streams} | Detections: {total_dets} | Tracks: {total_tracks_count} | Avg FPS: {avg_fps:.1f}'
            cv2.putText(grid, info2, (10, grid.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Show grid
            if not args.no_display:
                cv2.imshow('Multi-Stream GPU Tracking with YOLO', grid)

        # Headless progress
        if args.no_display and frame_id % 10 == 0:
            total_dets = sum(len(d) for d in all_detections)
            total_tracks = sum(len(t) for t in all_tracks)
            print(f"Frame {frame_id}: {total_dets} dets, {total_tracks} tracks")
        if not args.no_display and cv2.waitKey(1) & 0xFF == ord('q'):

            break
    # Cleanup
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()

    # Summary
    avg_det = (total_detection_time / frame_id * 1000) if frame_id > 0 else 0
    avg_track = (total_tracking_time / frame_id * 1000) if frame_id > 0 else 0
    avg_total = avg_det + avg_track
    avg_fps = frame_id / (total_detection_time + total_tracking_time)

    print(f"\n{'='*60}")
    print(f"Performance Summary")
    print(f"{'='*60}")
    print(f"Streams: {num_streams}")
    print(f"Frames: {frame_id}")
    print(f"\nTiming per batch:")
    print(f"  Detection:  {avg_det:.2f}ms")
    print(f"  Tracking:   {avg_track:.2f}ms")
    print(f"  Total:      {avg_total:.2f}ms")
    print(f"  Batch FPS:  {avg_fps:.1f}")
    print(f"  Per-stream: {avg_fps/num_streams:.1f} FPS")
    print(f"\nGPU Utilization:")
    print(f"  Detection uses batching for efficiency")
    print(f"  Tracking processes all streams in parallel")


if __name__ == '__main__':
    main()

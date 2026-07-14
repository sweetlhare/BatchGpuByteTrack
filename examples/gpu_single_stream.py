#!/usr/bin/env python3
"""
GPU Single-Stream Tracking Example with Real YOLO Detector

Demonstrates GPU-accelerated tracking for a single video stream.
Uses real YOLO detection and shows speedup compared to CPU tracking.

Requirements:
    pip install -e .  # Install package in development mode
    # OR
    pip install ultralytics lap scipy torch

Usage:
    python examples/gpu_single_stream.py --video video.mp4 --conf 0.5
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
from bytetrack import BatchGPUTracker, TrackerConfig

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics not installed")
    print("Install with: pip install ultralytics")
    exit(1)


def main():
    parser = argparse.ArgumentParser(description='GPU Single-Stream Tracking with YOLO')
    parser.add_argument('--video', type=str, default='test_videos/6387-191695740_medium.mp4',
                       help='Path to input video')
    parser.add_argument('--model', type=str, default='yolo26s.pt',
                       help='YOLO model (yolo26s.pt, yolov8n.pt, yolov8s.pt, etc.)')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Detection confidence threshold')
    parser.add_argument('--track-thresh', type=float, default=0.6,
                       help='High-confidence track threshold')
    parser.add_argument('--output', type=str, default=None,
                       help='Output video path (optional)')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable display (for headless/SSH mode)')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum frames to process (for testing)')
    args = parser.parse_args()

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("Error: CUDA not available. Please install PyTorch with CUDA support.")
        return

    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # Load YOLO detector on GPU
    print(f"Loading YOLO model: {args.model}")
    yolo = YOLO(args.model)
    yolo.to('cuda')
    yolo.fuse()

    # Configure GPU tracker
    config = TrackerConfig(
        track_thresh=args.track_thresh,
        track_buffer=30,
        match_thresh=0.8,
        enable_reid=False,  # Disable ReID for speed
        device='cuda'
    )

    # Initialize tracker for 1 stream
    tracker = BatchGPUTracker(1, config)

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Cannot open video {args.video}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Setup output video writer if needed
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"Saving output to: {args.output}")

    print(f"Video: {width}x{height} @ {fps} FPS")
    print("Starting GPU tracking...")
    print("Press 'q' to quit")

    frame_id = 0
    total_detection_time = 0
    total_tracking_time = 0
    total_detections = 0
    total_tracks = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        # Stop if max_frames reached
        if args.max_frames and frame_id > args.max_frames:
            break

        # YOLO detection on GPU
        det_start = time.time()
        results = yolo(frame, conf=args.conf, verbose=False)[0]
        det_time = time.time() - det_start
        total_detection_time += det_time


        # Convert YOLO results to numpy arrays
        if len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy().reshape(-1, 1)
            detections = np.concatenate([boxes, scores], axis=1)
        else:
            detections = np.empty((0, 5), dtype=np.float32)

        total_detections += len(detections)

        # Update tracker (GPU)
        track_start = time.time()
        all_tracks = tracker.update([detections])  # Pass as list
        track_time = time.time() - track_start
        total_tracking_time += track_time

        tracks = all_tracks[0]  # Get tracks for stream 0
        total_tracks += len(tracks)

        # Visualize
        # Draw detections in blue
        for det in detections:
            x1, y1, x2, y2, score = det
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                         (255, 0, 0), 1)

        # Draw tracks in green
        for track in tracks:
            track_id = track['track_id']
            x1, y1, x2, y2 = map(int, track['tlbr'])

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw track ID
            label = f'ID: {track_id}'
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display performance info
        total_time = det_time + track_time
        current_fps = 1 / total_time if total_time > 0 else 0
        avg_fps = frame_id / (total_detection_time + total_tracking_time) if frame_id > 0 else 0

        info1 = f'Frame: {frame_id} | Dets: {len(detections)} | Tracks: {len(tracks)}'
        info2 = f'Det: {det_time*1000:.1f}ms | Track: {track_time*1000:.1f}ms | FPS: {current_fps:.1f}'
        info3 = f'Avg FPS: {avg_fps:.1f}'

        cv2.putText(frame, info1, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, info2, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, info3, (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Legend
        cv2.putText(frame, 'Blue: Detections | Green: Tracks', (10, height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Write frame if output is enabled
        if writer is not None:
            writer.write(frame)

        # Show frame
        # Display
        if not args.no_display:
            cv2.imshow('GPU Tracking with YOLO', frame)
        else:
            # Print progress in headless mode
            if frame_id % 30 == 0:
                print(f"Frame {frame_id}: {len(detections)} dets, {len(tracks)} tracks")

        if not args.no_display and cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    # Print summary
    avg_det_time = (total_detection_time / frame_id * 1000) if frame_id > 0 else 0
    avg_track_time = (total_tracking_time / frame_id * 1000) if frame_id > 0 else 0
    avg_total_time = avg_det_time + avg_track_time
    avg_fps = frame_id / (total_detection_time + total_tracking_time) if frame_id > 0 else 0

    print(f"\n{'='*60}")
    print(f"Performance Summary")
    print(f"{'='*60}")
    print(f"Processed frames: {frame_id}")
    print(f"Total detections: {total_detections} (avg {total_detections/frame_id:.1f}/frame)")
    print(f"Total tracks: {total_tracks} (avg {total_tracks/frame_id:.1f}/frame)")
    print(f"\nTiming:")
    print(f"  Detection:  {avg_det_time:.2f}ms/frame")
    print(f"  Tracking:   {avg_track_time:.2f}ms/frame")
    print(f"  Total:      {avg_total_time:.2f}ms/frame")
    print(f"  Average FPS: {avg_fps:.1f}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Basic CPU Tracking Example with Real YOLO Detector

Demonstrates simple CPU-based single-stream tracking with YOLO detector.
Uses real object detection (not mock data).

Requirements:
    pip install -e .  # Install package in development mode
    # OR
    pip install ultralytics lap scipy

Usage:
    python examples/basic_cpu_tracking.py --video video.mp4 --conf 0.5
"""

import sys
from pathlib import Path

# Add project root to path (allows running without installation)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import argparse
from bytetrack import BYTETracker

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics not installed")
    print("Install with: pip install ultralytics")
    exit(1)


class Args:
    """Arguments for BYTETracker"""
    def __init__(self, track_thresh=0.6, track_buffer=30, match_thresh=0.8):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.mot20 = False


def main():
    parser = argparse.ArgumentParser(description='CPU Tracking with YOLO')
    parser.add_argument('--video', type=str, default='test_videos/6387-191695740_medium.mp4',
                       help='Path to input video')
    parser.add_argument('--model', type=str, default='yolo26s.pt',
                       help='YOLO model (yolo26s.pt, yolov8n.pt, yolov8s.pt, etc.)')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Detection confidence threshold')
    parser.add_argument('--track-thresh', type=float, default=0.6,
                       help='High-confidence track threshold')
    parser.add_argument('--track-buffer', type=int, default=30,
                       help='Track buffer size (frames)')
    parser.add_argument('--match-thresh', type=float, default=0.8,
                       help='IoU matching threshold')
    parser.add_argument('--output', type=str, default=None,
                       help='Output video path (optional)')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable display (for headless/SSH mode)')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum frames to process (for testing)')
    args = parser.parse_args()

    # Load YOLO detector
    print(f"Loading YOLO model: {args.model}")
    yolo = YOLO(args.model)
    yolo.fuse()  # Fuse Conv2d + BatchNorm2d layers for faster inference

    # Initialize ByteTrack CPU tracker
    tracker_args = Args(
        track_thresh=args.track_thresh,
        track_buffer=args.track_buffer,
        match_thresh=args.match_thresh
    )
    tracker = BYTETracker(tracker_args, frame_rate=30)

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
    print("Starting CPU tracking...")
    if not args.no_display:
        print("Press 'q' to quit")
    else:
        print("Running in headless mode (no display)")

    frame_id = 0
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

        # YOLO detection
        results = yolo(frame, conf=args.conf, verbose=False)[0]

        # Convert YOLO results to ByteTrack format
        # Format: np.array([[x1, y1, x2, y2, score], ...])
        if len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()  # (N, 4)
            scores = results.boxes.conf.cpu().numpy().reshape(-1, 1)  # (N, 1)
            detections = np.concatenate([boxes, scores], axis=1)  # (N, 5)
        else:
            detections = np.empty((0, 5))

        total_detections += len(detections)

        # Update tracker
        tracks = tracker.update(detections, (height, width), (height, width))
        total_tracks += len(tracks)

        # Visualize only if display is enabled
        if not args.no_display or writer is not None:
            # Draw detections in blue
            for det in detections:
                x1, y1, x2, y2, score = det
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                             (255, 0, 0), 1)

            # Draw tracks in green
            for track in tracks:
                if not track.is_activated:
                    continue

                x1, y1, x2, y2 = map(int, track.tlbr)
                track_id = track.track_id

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw track ID
                label = f'ID: {track_id}'
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Display info
            info_text = f'Frame: {frame_id} | Dets: {len(detections)} | Tracks: {len(tracks)}'
            cv2.putText(frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Legend
            cv2.putText(frame, 'Blue: Detections | Green: Tracks', (10, height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Write frame if output is enabled
        if writer is not None:
            writer.write(frame)

        # Show frame only if display is enabled
        if not args.no_display:
            cv2.imshow('CPU Tracking with YOLO', frame)
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            # Print progress in headless mode
            if frame_id % 30 == 0:  # Every 30 frames
                print(f"Frame {frame_id}: {len(detections)} dets, {len(tracks)} tracks")

    # Cleanup
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    # Print summary
    print(f"\nProcessed {frame_id} frames")
    print(f"Total detections: {total_detections} (avg {total_detections/frame_id:.1f}/frame)")
    print(f"Total track instances: {total_tracks} (avg {total_tracks/frame_id:.1f}/frame)")


if __name__ == '__main__':
    main()

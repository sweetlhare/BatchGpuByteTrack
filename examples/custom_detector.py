#!/usr/bin/env python3
"""
Custom Detector Integration Example

Demonstrates how to integrate ByteTrack with any object detector.
Shows the detector-agnostic design of ByteTrack.

Requirements:
    pip install -e .  # Install package in development mode
    # OR
    pip install lap scipy torch
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
from bytetrack import BatchGPUTracker, TrackerConfig


class CustomDetector:
    """
    Example custom detector class.

    Replace this with your actual detector (Faster R-CNN, RetinaNet, etc.)
    The only requirement is to return detections as a NumPy array (N, 5)
    with format [x1, y1, x2, y2, confidence_score]. If your model returns
    torch tensors, convert them with `.cpu().numpy()` before tracking.
    """

    def __init__(self, device='cuda'):
        self.device = device
        print(f"Initializing custom detector on {device}")

        # Load your model here
        # self.model = load_model(...)
        # self.model.to(device)
        # self.model.eval()

    def detect(self, frame):
        """
        Run detection on frame.

        Args:
            frame: Input frame (BGR, numpy array)

        Returns:
            np.ndarray: Detections (N, 5) with format [x1, y1, x2, y2, score]
        """
        # Preprocess
        # input_tensor = self.preprocess(frame)

        # Inference
        # with torch.no_grad():
        #     output = self.model(input_tensor)

        # Postprocess (don't forget .cpu().numpy() for torch models)
        # detections = self.postprocess(output)

        # For demonstration, return dummy detections
        detections = np.array([
            [100, 100, 200, 200, 0.95],
            [300, 150, 400, 250, 0.88],
            [500, 200, 600, 300, 0.82],
        ], dtype=np.float32)

        return detections

    def preprocess(self, frame):
        """Preprocess frame for your detector."""
        # Example: Resize, normalize, convert to tensor
        pass

    def postprocess(self, output):
        """Postprocess detector output to [x1, y1, x2, y2, score] format."""
        # Example: Apply NMS, filter by confidence, format boxes
        pass


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Custom detector + ByteTrack example')
    parser.add_argument('--video', default='test_videos/6387-191695740_medium.mp4',
                        help='Path to input video')
    parser.add_argument('--no-display', action='store_true',
                        help='Disable display (for headless/SSH mode)')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Maximum frames to process (for testing)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Initialize custom detector
    detector = CustomDetector(device=device)

    # Configure ByteTrack tracker
    config = TrackerConfig(
        track_thresh=0.6,      # Adjust based on your detector
        track_buffer=30,
        match_thresh=0.8,
        enable_reid=False,
        device=device
    )

    tracker = BatchGPUTracker(1, config)

    # Open video
    cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        print(f"Error: Cannot open video {args.video}")
        return

    print("Processing video with custom detector + ByteTrack...")
    print("Press 'q' to quit")

    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        if args.max_frames and frame_id > args.max_frames:
            break

        # Step 1: Run your custom detector
        detections = detector.detect(frame)

        # Step 2: Update ByteTrack tracker (expects numpy arrays)
        # Note: Pass detections as list for batch processing
        all_tracks = tracker.update([detections])
        tracks = all_tracks[0]  # Get tracks for stream 0

        # Step 3: Visualize results
        # Draw detections in blue
        for det in detections:
            x1, y1, x2, y2, score = det
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                         (255, 0, 0), 1)
            cv2.putText(frame, f'{score:.2f}', (int(x1), int(y1) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        # Draw tracks in green
        for track in tracks:
            track_id = track['track_id']
            x1, y1, x2, y2 = map(int, track['tlbr'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display info
        info = f'Frame: {frame_id} | Detections: {len(detections)} | Tracks: {len(tracks)}'
        cv2.putText(frame, info, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(frame, 'Blue: Detections | Green: Tracks', (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if not args.no_display:
            cv2.imshow('Custom Detector + ByteTrack', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        elif frame_id % 10 == 0:
            print(f"Frame {frame_id}: {len(detections)} dets, {len(tracks)} tracks")

    cap.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    print(f"Processed {frame_id} frames")


# Example: Integrating with different detectors
class YOLODetector:
    """YOLO detector wrapper."""

    def __init__(self):
        from ultralytics import YOLO
        self.model = YOLO('yolo26s.pt')

    def detect(self, frame):
        results = self.model(frame, verbose=False)[0]
        boxes = results.boxes.xyxy
        scores = results.boxes.conf.unsqueeze(1)
        detections = torch.cat([boxes, scores], dim=1)
        return detections.cpu().numpy()


class FasterRCNNDetector:
    """Faster R-CNN detector wrapper."""

    def __init__(self):
        import torchvision
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        self.model.cuda()

    def detect(self, frame):
        # Convert frame to tensor
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        frame_tensor = frame_tensor.unsqueeze(0).cuda()

        # Inference
        with torch.no_grad():
            output = self.model(frame_tensor)[0]

        # Format output
        boxes = output['boxes']
        scores = output['scores'].unsqueeze(1)
        detections = torch.cat([boxes, scores], dim=1)

        # Filter by confidence
        detections = detections[scores.squeeze() > 0.5]

        return detections.cpu().numpy()


class MMDetectionDetector:
    """MMDetection detector wrapper."""

    def __init__(self, config_file, checkpoint_file):
        from mmdet.apis import init_detector
        self.model = init_detector(config_file, checkpoint_file, device='cuda:0')

    def detect(self, frame):
        from mmdet.apis import inference_detector
        result = inference_detector(self.model, frame)

        # Convert result to a numpy array [x1, y1, x2, y2, score];
        # this depends on your MMDetection model output format
        raise NotImplementedError("Adapt to your MMDetection model output")


if __name__ == '__main__':
    main()

    # To use with different detectors:
    # detector = YOLODetector()
    # detector = FasterRCNNDetector()
    # detector = MMDetectionDetector('config.py', 'checkpoint.pth')

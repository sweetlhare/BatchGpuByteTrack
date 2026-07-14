"""
Batched GPU Tracker for multi-stream real-time tracking.

This module provides a high-performance tracker that processes multiple video
streams in batches, minimizing GPU-CPU transfers and maximizing parallelism.
"""

import logging
import torch
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from contextlib import nullcontext

from .gpu_kalman_filter import GPUKalmanFilter
from .gpu_matching import linear_assignment
from .batch_track_state import GPUSTrack, StreamState, GlobalGallery, TrackState

logger = logging.getLogger(__name__)


@dataclass
class TrackerConfig:
    """Configuration for BatchGPUTracker."""
    # Detection thresholds
    track_thresh: float = 0.5       # High-confidence threshold
    track_buffer: int = 30          # Max frames to keep lost tracks
    match_thresh: float = 0.8       # IoU matching threshold

    # Re-ID settings
    enable_reid: bool = False
    reid_model_path: Optional[str] = None   # OSNet variant name, e.g. 'osnet_x0_5'
    reid_checkpoint: Optional[str] = None   # Optional local .pth file (offline setups)
    reid_embedding_dim: int = 512
    reid_threshold: float = 0.5     # Max embedding distance for appearance fusion
    lambda_emb: float = 0.3         # Weight of appearance cost in fused cost (0..1)
    reid_fp16: bool = True          # FP16 OSNet inference on CUDA (~2x faster)
    reid_interval: int = 1          # Extract features every K-th frame (1 = every frame)

    # Cross-camera settings
    enable_cross_camera: bool = False
    cross_camera_threshold: float = 0.7

    # Hardware settings
    device: str = 'cuda'
    num_threads: int = 8            # CPU threads for Hungarian


def _fuse_score_np(cost_matrix: np.ndarray, det_scores: np.ndarray) -> np.ndarray:
    """Fuse IoU cost with detection scores (same as original ByteTrack)."""
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1.0 - cost_matrix
    return 1.0 - iou_sim * det_scores[None, :]


class BatchGPUTracker:
    """
    Batched GPU tracker for processing multiple video streams.

    Architecture:
    - GPU: Kalman prediction/update, IoU computation, Re-ID embedding
    - CPU (parallel): Hungarian assignment, track state updates

    Usage:
        tracker = BatchGPUTracker(num_streams=8, config=TrackerConfig())

        for frames in video_streams:
            # frames: list of [H, W, 3] BGR numpy arrays (only needed for ReID)
            # detections: list of [N_i, 5] arrays (x1,y1,x2,y2,score)
            tracks = tracker.update(detections)
            # tracks: list of 8 lists of track dicts
    """

    def __init__(
        self,
        num_streams: int,
        config: Optional[TrackerConfig] = None,
        profiler: Optional['Profiler'] = None
    ):
        self.num_streams = num_streams
        self.config = config or TrackerConfig()
        self.device = torch.device(self.config.device)
        self.profiler = profiler

        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError(
                "TrackerConfig(device='cuda') requested but CUDA is not available. "
                "Pass TrackerConfig(device='cpu') or use ParallelCPUTracker instead."
            )

        # New tracks are only initialized above this score (same as original ByteTrack)
        self.det_thresh = self.config.track_thresh + 0.1

        # GPU components
        self.kalman = GPUKalmanFilter(self.config.device)

        # Per-stream state (CPU)
        self.stream_states = [StreamState(i) for i in range(num_streams)]

        # Re-ID model (optional)
        self.reid_model = None
        if self.config.enable_reid and self.config.reid_model_path:
            self.reid_model = self._load_reid_model(self.config.reid_model_path)

        # Cross-camera gallery (optional)
        self.global_gallery = None
        if self.config.enable_cross_camera:
            self.global_gallery = GlobalGallery(
                similarity_threshold=self.config.cross_camera_threshold,
                embedding_dim=self.config.reid_embedding_dim
            )

        # Thread pool for parallel Hungarian
        self.executor = ThreadPoolExecutor(max_workers=self.config.num_threads)

        # Frame counter for logging
        self.frame_id = 0
        self._warned_invalid_dets = False

        # Persistent pinned staging buffers for ReID frame uploads,
        # keyed by (num_frames, H, W). Pageable-memory uploads run at a
        # fraction of PCIe bandwidth; pinned staging makes them ~10x faster.
        self._frame_buffers: Dict[Tuple[int, int, int], torch.Tensor] = {}

    def _profile(self, name: str):
        """Get profiling context manager (no-op if profiler is None)."""
        if self.profiler is not None:
            return self.profiler.profile(name)
        return nullcontext()

    def _load_reid_model(self, model_name: str):
        """
        Load Re-ID model (OSNet).

        Args:
            model_name: Model variant ('osnet_x1_0', 'osnet_x0_75', 'osnet_x0_5',
                        'osnet_x0_25', 'osnet_ibn_x1_0'). Weights are downloaded
                        automatically unless config.reid_checkpoint points to a
                        local .pth file.

        Raises:
            ValueError: unknown model name
            RuntimeError: weight download/loading failed
        """
        from .osnet import build_osnet

        model = build_osnet(
            model_name,
            pretrained=True,
            device=self.config.device,
            checkpoint=self.config.reid_checkpoint
        )
        logger.info("Loaded %s ReID model (embedding_dim=%d)", model_name, model.feature_dim)

        # Warm up the inference buckets: cuDNN pays a one-time kernel-selection
        # cost per input shape (~300 ms each under FP16). Doing it here keeps
        # per-frame latency flat from the first frame.
        if self.device.type == 'cuda':
            use_fp16 = self.config.reid_fp16
            with torch.no_grad():
                for n in (32, 64, 128, 256, 512):
                    dummy = torch.zeros(n, 3, 256, 128, device=self.device)
                    with torch.autocast(device_type='cuda', enabled=use_fp16):
                        model(dummy)
            torch.cuda.synchronize()
            logger.info("ReID inference warmed up")

        return model

    def _extract_reid_features(
        self,
        frames: List[np.ndarray],
        bboxes: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Extract ReID embeddings from detection crops.

        Frames are uploaded once per stream and ALL crops are extracted and
        resized in a single batched `roi_align` call — no per-crop transfers
        or kernels. Inference runs in one batched (optionally FP16) forward.

        Args:
            frames: List of [H, W, 3] numpy arrays (BGR format, as returned by cv2)
            bboxes: List of [N_i, ...] detection arrays, first 4 columns (x1, y1, x2, y2)

        Returns:
            embeddings: List of [N_i, D] embedding arrays
        """
        if self.reid_model is None:
            return [None] * len(frames)

        import torch.nn.functional as F
        from torchvision.ops import roi_align

        # Sanitize boxes per stream (vectorized): clip to frame, replace
        # NaN/inf/degenerate boxes with a dummy so indices stay aligned
        stream_boxes = []
        stream_counts = []
        for frame, boxes in zip(frames, bboxes):
            n = len(boxes)
            stream_counts.append(n)
            if n == 0:
                stream_boxes.append(np.empty((0, 4), dtype=np.float32))
                continue
            h, w = frame.shape[:2]
            b = np.array(boxes[:, :4], dtype=np.float32, copy=True)
            bad = ~np.isfinite(b).all(axis=1)
            b[bad] = 0.0
            b[:, 0::2] = np.clip(b[:, 0::2], 0, w)
            b[:, 1::2] = np.clip(b[:, 1::2], 0, h)
            bad |= (b[:, 2] <= b[:, 0]) | (b[:, 3] <= b[:, 1])
            b[bad] = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
            stream_boxes.append(b)

        total_crops = sum(stream_counts)
        if total_crops == 0:
            return [np.empty((0, self.config.reid_embedding_dim), dtype=np.float32) for _ in frames]

        use_fp16 = self.config.reid_fp16 and self.device.type == 'cuda'

        with torch.no_grad():
            # Extract all crops with batched roi_align, grouping frames by
            # resolution (usually a single group). Crops come out in
            # group order; scatter back by original position.
            crops = torch.empty(
                total_crops, 3, 256, 128, device=self.device, dtype=torch.float32
            )
            offsets = np.concatenate([[0], np.cumsum(stream_counts)])

            groups = {}
            for sid, frame in enumerate(frames):
                if stream_counts[sid] > 0:
                    groups.setdefault(frame.shape[:2], []).append(sid)

            for (fh, fw), sids in groups.items():
                # One bulk upload per resolution group through a persistent
                # pinned staging buffer: [F, H, W, 3] uint8
                key = (len(sids), fh, fw)
                buf = self._frame_buffers.get(key)
                if buf is None:
                    buf = torch.empty(
                        len(sids), fh, fw, 3, dtype=torch.uint8,
                        pin_memory=self.device.type == 'cuda'
                    )
                    self._frame_buffers[key] = buf
                for i, s in enumerate(sids):
                    buf[i].copy_(torch.from_numpy(np.ascontiguousarray(frames[s])))

                # Keep frames uint8->float only; BGR->RGB flip and value
                # normalization run on the much smaller crops below
                imgs = buf.to(self.device, non_blocking=True)
                imgs = imgs.permute(0, 3, 1, 2).float()  # [F, 3, H, W], 0..255

                boxes_t = [torch.from_numpy(stream_boxes[s]).to(self.device) for s in sids]
                group_crops = roi_align(
                    imgs, boxes_t, output_size=(256, 128), aligned=True
                )  # [sum(n_s), 3, 256, 128]

                start = 0
                for s in sids:
                    n = stream_counts[s]
                    crops[offsets[s]:offsets[s] + n] = group_crops[start:start + n]
                    start += n

            # BGR -> RGB, [0, 1] scaling and ImageNet normalization on crops
            crops = crops.flip(1).div_(255.0)
            mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
            crops = (crops - mean) / std

            # Pad the batch to a multiple of 32: cuDNN selects kernels per
            # input shape (a ~300 ms one-time cost per NEW batch size under
            # FP16), and detection counts change every frame — bucketing
            # keeps the set of distinct shapes tiny
            bucket = 32
            padded_n = -(-total_crops // bucket) * bucket
            if padded_n != total_crops:
                pad = torch.zeros(
                    padded_n - total_crops, 3, 256, 128,
                    device=self.device, dtype=crops.dtype
                )
                crops = torch.cat([crops, pad], dim=0)

            # Batched inference (chunked; FP16 via autocast when enabled)
            max_batch_size = 512
            all_features = []
            for i in range(0, crops.shape[0], max_batch_size):
                chunk = crops[i:i + max_batch_size]
                with torch.autocast(device_type='cuda', enabled=use_fp16):
                    features = self.reid_model(chunk)
                features = F.normalize(features.float(), p=2, dim=1)
                all_features.append(features)

            all_features = torch.cat(all_features, dim=0)[:total_crops].cpu().numpy()

        # Split embeddings back to per-stream arrays
        return [
            all_features[offsets[s]:offsets[s] + stream_counts[s]]
            if stream_counts[s] > 0
            else np.empty((0, self.config.reid_embedding_dim), dtype=np.float32)
            for s in range(len(frames))
        ]

    def update(
        self,
        detections: List[np.ndarray],
        frames: Optional[List[np.ndarray]] = None,
        embeddings: Optional[List[np.ndarray]] = None
    ) -> List[List[Dict[str, Any]]]:
        """
        Process one frame from each stream.

        Args:
            detections: List of [N_i, 5] numpy arrays per stream.
                       Each row: [x1, y1, x2, y2, score] (tlbr format, pixel units).
                       An entry may be None or empty for streams without detections.
            frames: Optional list of [H, W, 3] BGR uint8 numpy arrays per stream
                       (as returned by cv2). Only used for Re-ID crop extraction.
            embeddings: Optional list of [N_i, D] embedding arrays per stream
                       (if you computed embeddings externally)

        Returns:
            tracks: List of track lists per stream.
                   Each track is a dict with keys:
                   - 'track_id': int
                   - 'tlbr': [4] bounding box
                   - 'score': float
                   - 'stream_id': int
                   - 'global_id': int or None (set if cross-camera enabled)
        """
        if len(detections) != self.num_streams:
            raise ValueError(
                f"Expected {self.num_streams} detection arrays "
                f"(one per stream), got {len(detections)}"
            )

        # Normalize detections: None / empty -> [0, 5] arrays
        detections = [
            dets if dets is not None and len(dets) > 0 else np.empty((0, 5), dtype=np.float32)
            for dets in detections
        ]

        # Extract ReID embeddings if frames provided but embeddings not
        # (with reid_interval > 1 only on every K-th frame; in between,
        # association falls back to IoU and tracks keep their smoothed
        # embeddings)
        if (
            self.config.enable_reid and embeddings is None and frames is not None
            and (self.config.reid_interval <= 1
                 or self.frame_id % self.config.reid_interval == 0)
        ):
            with self._profile("0_reid_extraction"):
                embeddings = self._extract_reid_features(frames, detections)

        # Validate detections (vectorized) and create GPUSTrack objects.
        # Invalid rows (NaN/inf or non-positive size) are replaced by dummy
        # score=0 boxes so that detection indices stay aligned.
        with self._profile("1_convert_detections"):
            all_det_tracks = []
            total_dets = 0
            invalid_dets = 0

            for stream_id, dets in enumerate(detections):
                n = len(dets)
                total_dets += n
                if n == 0:
                    all_det_tracks.append([])
                    continue

                w = dets[:, 2] - dets[:, 0]
                h = dets[:, 3] - dets[:, 1]
                valid = np.isfinite(dets[:, :4]).all(axis=1) & (w > 0) & (h > 0)
                invalid_dets += int(n - valid.sum())

                tlwhs = np.stack([dets[:, 0], dets[:, 1], w, h], axis=1).astype(np.float32)
                scores = dets[:, 4].astype(np.float32, copy=True)
                if not valid.all():
                    tlwhs[~valid] = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
                    scores[~valid] = 0.0

                stream_embs = embeddings[stream_id] if embeddings is not None else None
                stream_tracks = [
                    GPUSTrack(
                        tlwhs[i], float(scores[i]), stream_id,
                        stream_embs[i] if stream_embs is not None else None
                    )
                    for i in range(n)
                ]
                all_det_tracks.append(stream_tracks)

        if invalid_dets > 0 and not self._warned_invalid_dets:
            logger.warning(
                "%d/%d detections invalid (NaN/inf or non-positive size); "
                "replaced with dummy boxes. Further warnings suppressed.",
                invalid_dets, total_dets
            )
            self._warned_invalid_dets = True

        # Build per-stream association pools with a fixed ordering:
        #   [confirmed tracked | lost | unconfirmed tracked]
        # Stages 1-2 associate over the first two segments, stage 3 over the last.
        pools = []
        pool_splits = []  # (n_tracked_confirmed, n_assoc = confirmed + lost)
        for state in self.stream_states:
            confirmed = [t for t in state.tracked_stracks if t.is_activated]
            unconfirmed = [t for t in state.tracked_stracks if not t.is_activated]
            pool = confirmed + state.lost_stracks + unconfirmed
            pools.append(pool)
            pool_splits.append((len(confirmed), len(confirmed) + len(state.lost_stracks)))

        # ═══════════════════ GPU ZONE ═══════════════════

        # 1. Gather track states for batch prediction
        with self._profile("2_gather_states"):
            all_means, all_covs, track_info = self._gather_track_states(pools)

        # 2. Batched Kalman predict (GPU)
        if all_means.shape[0] > 0:
            with self._profile("3_kalman_predict"):
                all_means_t = torch.from_numpy(all_means).float().to(self.device)
                all_covs_t = torch.from_numpy(all_covs).float().to(self.device)
                pred_means, pred_covs = self.kalman.batch_predict(all_means_t, all_covs_t)
                pred_means_np = pred_means.cpu().numpy()
                pred_covs_np = pred_covs.cpu().numpy()

                # Update track states with predictions
                self._scatter_predictions(pred_means_np, pred_covs_np, track_info)

                # Detect and isolate corrupted tracks (NaN/inf) to prevent
                # cascade failures through the cost matrix
                num_corrupted = self._detect_and_fix_corrupted_tracks()
                if num_corrupted > 0:
                    logger.warning(
                        "Detected and reset %d corrupted tracks (NaN/inf values)",
                        num_corrupted
                    )

        # 3. Compute raw IoU cost matrices for ALL streams as one padded
        # batched GPU kernel (two bulk uploads, one kernel).
        # Score/appearance fusion happens per association stage on CPU.
        with self._profile("4_compute_iou_cost"):
            padded_cost, pool_sizes, det_sizes = self._compute_cost_matrices(detections, pools)

        # ═══════════════════ GPU → CPU TRANSFER ═══════════════════

        with self._profile("5_gpu_to_cpu_transfer"):
            if padded_cost is not None:
                padded_cost_np = padded_cost.cpu().numpy()
                cost_matrices_cpu = [
                    padded_cost_np[s, :pool_sizes[s], :det_sizes[s]]
                    for s in range(self.num_streams)
                ]
            else:
                cost_matrices_cpu = [
                    np.empty((pool_sizes[s], det_sizes[s]), dtype=np.float32)
                    for s in range(self.num_streams)
                ]

            all_det_scores = [
                dets[:, 4].astype(np.float32) if len(dets) > 0 else np.array([])
                for dets in detections
            ]

        # Appearance cost matrices (CPU, small): only when ReID data is present
        emb_dists = [None] * self.num_streams
        if self.config.enable_reid and embeddings is not None:
            with self._profile("5b_embedding_cost"):
                emb_dists = self._compute_embedding_costs(pools, all_det_tracks)

        # ═══════════════════ CPU ZONE: Hungarian (Parallel) ═══════════════════

        # 4. Hungarian assignment. For small workloads the thread-pool
        # submit overhead rivals the solve itself, so tiny frames run inline.
        with self._profile("6_hungarian_parallel"):
            total_elems = sum(cm.size for cm in cost_matrices_cpu)
            if self.num_streams == 1 or total_elems < 4096:
                all_matches_info = [
                    self._hungarian_assignment(
                        cost_matrices_cpu[s],
                        all_det_scores[s],
                        emb_dists[s],
                        pool_splits[s][0],
                        pool_splits[s][1],
                        len(pools[s])
                    )
                    for s in range(self.num_streams)
                ]
            else:
                futures = []
                for stream_id in range(self.num_streams):
                    n_tracked, n_assoc = pool_splits[stream_id]
                    future = self.executor.submit(
                        self._hungarian_assignment,
                        cost_matrices_cpu[stream_id],
                        all_det_scores[stream_id],
                        emb_dists[stream_id],
                        n_tracked,
                        n_assoc,
                        len(pools[stream_id])
                    )
                    futures.append(future)

                # Collect matches from all streams
                all_matches_info = [f.result() for f in futures]

        # ═══════════════════ GPU ZONE: Kalman updates for matched only ═══════════════════

        # 5. Batch Kalman updates for matched pairs of ALL streams in a single
        # GPU call (one upload, one kernel, one download — per-stream loops
        # here would pay N transfer/launch overheads per frame)
        with self._profile("7_kalman_update_matched"):
            all_kalman_updates = [{} for _ in range(self.num_streams)]
            all_new_track_states = [{} for _ in range(self.num_streams)]

            matched_entries = []  # (stream_id, (track_idx, det_idx))
            matched_means = []
            matched_covs = []
            matched_measurements = []
            new_entries = []      # (stream_id, det_idx)
            new_xyahs = []

            for stream_id in range(self.num_streams):
                matches, u_tracks, u_dets_high = all_matches_info[stream_id]
                det_tracks = all_det_tracks[stream_id]
                strack_pool = pools[stream_id]

                for track_idx, det_idx in matches:
                    track = strack_pool[track_idx]
                    if track.mean is not None:
                        matched_entries.append((stream_id, (track_idx, det_idx)))
                        matched_means.append(track.mean)
                        matched_covs.append(track.covariance)
                        matched_measurements.append(det_tracks[det_idx].xyah)

                for det_idx in u_dets_high:
                    det = det_tracks[det_idx]
                    if det.score >= self.det_thresh:
                        new_entries.append((stream_id, det_idx))
                        new_xyahs.append(det.xyah)

            if matched_entries:
                means_t = torch.from_numpy(np.stack(matched_means)).float().to(self.device)
                covs_t = torch.from_numpy(np.stack(matched_covs)).float().to(self.device)
                meas_t = torch.from_numpy(np.stack(matched_measurements)).float().to(self.device)

                upd_means, upd_covs = self.kalman.batch_update(means_t, covs_t, meas_t)
                upd_means_np = upd_means.cpu().numpy()
                upd_covs_np = upd_covs.cpu().numpy()

                for i, (stream_id, key) in enumerate(matched_entries):
                    all_kalman_updates[stream_id][key] = (upd_means_np[i], upd_covs_np[i])

            if new_entries:
                xyahs_t = torch.from_numpy(np.stack(new_xyahs)).float().to(self.device)
                init_means, init_covs = self.kalman.batch_initiate(xyahs_t)
                init_means_np = init_means.cpu().numpy()
                init_covs_np = init_covs.cpu().numpy()

                for i, (stream_id, det_idx) in enumerate(new_entries):
                    all_new_track_states[stream_id][det_idx] = (init_means_np[i], init_covs_np[i])

        # ═══════════════════ CPU ZONE: State updates (Sequential) ═══════════════════
        # Note: Sequential is faster than ThreadPoolExecutor due to Python GIL

        # 6. Sequential state updates
        with self._profile("8_state_update_seq"):
            results = []
            for stream_id in range(self.num_streams):
                matches, u_tracks, u_dets_high = all_matches_info[stream_id]
                result = self._update_stream_state(
                    stream_id,
                    pools[stream_id],
                    matches,
                    u_tracks,
                    u_dets_high,
                    all_det_tracks[stream_id],
                    all_kalman_updates[stream_id],
                    all_new_track_states[stream_id]
                )
                results.append(result)

        # 7. Cross-camera Re-ID (optional)
        if self.config.enable_cross_camera and self.global_gallery is not None:
            self._cross_camera_association(results)

        # Increment frame counter
        self.frame_id += 1

        return results

    def _gather_track_states(
        self,
        pools: List[List[GPUSTrack]]
    ) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, GPUSTrack]]]:
        """
        Gather Kalman states from all streams for batch processing.

        Returns:
            all_means: [N_total, 8] concatenated means
            all_covs: [N_total, 8, 8] concatenated covariances
            track_info: List of (stream_id, track) for each state
        """
        all_means = []
        all_covs = []
        track_info = []

        for stream_id, pool in enumerate(pools):
            for track in pool:
                if track.mean is not None:
                    mean = track.mean
                    cov = track.covariance
                    if isinstance(mean, torch.Tensor):
                        mean = mean.cpu().numpy()
                        cov = cov.cpu().numpy()
                    if track.state != TrackState.Tracked:
                        # Zero out height velocity for lost tracks
                        # (same as original ByteTrack multi_predict)
                        mean = mean.copy()
                        mean[7] = 0.0
                    all_means.append(mean)
                    all_covs.append(cov)
                    track_info.append((stream_id, track))

        if not all_means:
            return np.empty((0, 8)), np.empty((0, 8, 8)), []

        return np.stack(all_means), np.stack(all_covs), track_info

    def _scatter_predictions(
        self,
        pred_means: np.ndarray,
        pred_covs: np.ndarray,
        track_info: List[Tuple[int, GPUSTrack]]
    ):
        """Update track states with predicted values."""
        for i, (stream_id, track) in enumerate(track_info):
            track.mean = pred_means[i]
            track.covariance = pred_covs[i]

    def _detect_and_fix_corrupted_tracks(self) -> int:
        """
        Detect tracks with NaN/inf values in mean or covariance and mark them
        as Lost/Removed. This prevents cascade failures where one bad track
        corrupts the entire cost matrix.

        Returns:
            num_corrupted: Number of corrupted tracks detected and fixed
        """
        num_corrupted = 0

        for stream_id in range(self.num_streams):
            state = self.stream_states[stream_id]

            # Check tracked tracks
            for track in state.tracked_stracks:
                if track.mean is not None:
                    # Check for NaN/inf in mean (position/velocity)
                    if not np.all(np.isfinite(track.mean[:4])):  # Check position components
                        track.mark_lost()
                        num_corrupted += 1
                        continue

                    # Check for NaN/inf or overflow in covariance
                    if track.covariance is not None:
                        diag = np.diagonal(track.covariance)
                        if not np.all(np.isfinite(diag)) or np.any(diag > 1e6):
                            track.mark_lost()
                            num_corrupted += 1

            # Check lost tracks (to prevent re-activation of corrupted tracks)
            for track in state.lost_stracks:
                if track.mean is not None:
                    if not np.all(np.isfinite(track.mean[:4])):
                        # Mark for removal
                        track.mark_removed()
                        num_corrupted += 1

        return num_corrupted

    def _track_boxes_tlbr(self, pool: List[GPUSTrack]) -> np.ndarray:
        """
        Build [M, 4] tlbr boxes for an association pool (vectorized).

        Uses the (predicted) Kalman xyah state when available and finite,
        clamped to sane ranges; falls back to the track's cached box otherwise.
        """
        m = len(pool)
        xyah = np.full((m, 4), np.nan, dtype=np.float32)
        for i, track in enumerate(pool):
            mean = track.mean
            if mean is not None:
                xyah[i] = mean[:4]

        bad = ~np.isfinite(xyah).all(axis=1)

        # Clamp aspect ratio / width to prevent extreme values
        with np.errstate(invalid='ignore'):
            a = np.clip(xyah[:, 2], 0.1, 10.0)
            h = xyah[:, 3]
            w = np.clip(a * h, 0, 1e5)
            boxes = np.stack([
                xyah[:, 0] - w / 2,
                xyah[:, 1] - h / 2,
                xyah[:, 0] + w / 2,
                xyah[:, 1] + h / 2
            ], axis=1)
            np.clip(boxes, 0, 1e5, out=boxes)

        if bad.any():
            # Rare path: no state yet or corrupted — use the cached box
            for i in np.where(bad)[0]:
                tlbr = pool[i].tlbr
                boxes[i] = tlbr if np.all(np.isfinite(tlbr)) else (0, 0, 1, 1)

        return boxes.astype(np.float32)

    def _compute_cost_matrices(
        self,
        detections: List[np.ndarray],
        pools: List[List[GPUSTrack]]
    ) -> Tuple[Optional[torch.Tensor], List[int], List[int]]:
        """
        Compute raw IoU cost matrices (1 - IoU) for ALL streams at once.

        Track and detection boxes are packed into zero-padded [S, M_max, 4] /
        [S, N_max, 4] arrays on the CPU, uploaded in two bulk transfers, and
        IoU is computed as a single batched kernel. Padded entries are never
        read back (callers slice [:M_s, :N_s]).

        Returns:
            padded_cost: [S, M_max, N_max] tensor, or None if any dimension is 0
            pool_sizes: per-stream M_s
            det_sizes: per-stream N_s
        """
        pool_sizes = [len(p) for p in pools]
        det_sizes = [len(d) for d in detections]
        m_max = max(pool_sizes)
        n_max = max(det_sizes)

        if m_max == 0 or n_max == 0:
            return None, pool_sizes, det_sizes

        s = self.num_streams
        track_padded = np.zeros((s, m_max, 4), dtype=np.float32)
        det_padded = np.zeros((s, n_max, 4), dtype=np.float32)

        for sid in range(s):
            if pool_sizes[sid] > 0:
                track_padded[sid, :pool_sizes[sid]] = self._track_boxes_tlbr(pools[sid])
            if det_sizes[sid] > 0:
                det_padded[sid, :det_sizes[sid]] = detections[sid][:, :4]

        # Invalid (NaN/inf) detection boxes become degenerate zero boxes:
        # IoU 0 against everything, so they can never be matched
        np.nan_to_num(det_padded, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        tb = torch.from_numpy(track_padded).to(self.device)  # [S, M, 4]
        db = torch.from_numpy(det_padded).to(self.device)    # [S, N, 4]

        area_t = (tb[..., 2] - tb[..., 0]) * (tb[..., 3] - tb[..., 1])  # [S, M]
        area_d = (db[..., 2] - db[..., 0]) * (db[..., 3] - db[..., 1])  # [S, N]

        lt = torch.maximum(tb[:, :, None, :2], db[:, None, :, :2])  # [S, M, N, 2]
        rb = torch.minimum(tb[:, :, None, 2:], db[:, None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]                              # [S, M, N]
        union = area_t[:, :, None] + area_d[:, None, :] - inter

        return 1.0 - inter / (union + 1e-7), pool_sizes, det_sizes

    def _compute_embedding_costs(
        self,
        pools: List[List[GPUSTrack]],
        all_det_tracks: List[List[GPUSTrack]]
    ) -> List[Optional[np.ndarray]]:
        """
        Compute per-stream appearance cost matrices (cosine distance).

        Entries where the track or detection has no embedding are NaN;
        the association stage falls back to pure IoU cost for those pairs.

        Returns:
            List of [M_i, N_i] float32 arrays or None per stream.
        """
        emb_dists = []

        for pool, dets in zip(pools, all_det_tracks):
            if len(pool) == 0 or len(dets) == 0:
                emb_dists.append(None)
                continue

            track_feats = [
                t.smooth_embedding if t.smooth_embedding is not None else t.embedding
                for t in pool
            ]
            det_feats = [d.embedding for d in dets]

            t_idx = [i for i, f in enumerate(track_feats) if f is not None]
            d_idx = [j for j, f in enumerate(det_feats) if f is not None]
            if not t_idx or not d_idx:
                emb_dists.append(None)
                continue

            t_mat = np.stack([track_feats[i] for i in t_idx]).astype(np.float32)
            d_mat = np.stack([det_feats[j] for j in d_idx]).astype(np.float32)
            t_mat /= (np.linalg.norm(t_mat, axis=1, keepdims=True) + 1e-7)
            d_mat /= (np.linalg.norm(d_mat, axis=1, keepdims=True) + 1e-7)

            dist = np.full((len(pool), len(dets)), np.nan, dtype=np.float32)
            dist[np.ix_(t_idx, d_idx)] = np.maximum(0.0, 1.0 - t_mat @ d_mat.T)
            emb_dists.append(dist)

        return emb_dists

    def _hungarian_assignment(
        self,
        cost_matrix: np.ndarray,
        det_scores: np.ndarray,
        emb_dist: Optional[np.ndarray],
        n_tracked: int,
        n_assoc: int,
        n_pool: int
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Run the three ByteTrack association stages for one stream (CPU).

        The cost matrix rows follow the pool ordering
        [confirmed tracked | lost | unconfirmed tracked]:
        - rows [0, n_tracked): confirmed tracked
        - rows [n_tracked, n_assoc): lost
        - rows [n_assoc, n_pool): unconfirmed

        Args:
            cost_matrix: [n_pool, N] raw IoU cost matrix (1 - IoU)
            det_scores: [N] detection scores
            emb_dist: optional [n_pool, N] appearance cost (NaN where unavailable)
            n_tracked / n_assoc / n_pool: pool segment boundaries

        Returns:
            matches: List of (track_idx, det_idx) tuples (pool indexing)
            u_tracks: List of unmatched track indices (pool indexing)
            u_dets: List of unmatched high-confidence detection indices
        """
        # Split detections by confidence (same thresholds as original ByteTrack)
        if len(det_scores) > 0:
            high_mask = det_scores > self.config.track_thresh
            low_mask = (det_scores > 0.1) & ~high_mask
            high_conf_indices = np.where(high_mask)[0].tolist()
            low_conf_indices = np.where(low_mask)[0].tolist()
        else:
            high_conf_indices = []
            low_conf_indices = []

        # ---- First association: confirmed + lost vs high-confidence detections
        if n_assoc > 0 and len(high_conf_indices) > 0:
            iou_sub = cost_matrix[np.ix_(range(n_assoc), high_conf_indices)]

            # Optionally fuse appearance cost (only where both sides have
            # embeddings and appearance distance is within reid_threshold)
            if emb_dist is not None and self.config.lambda_emb > 0:
                emb_sub = emb_dist[np.ix_(range(n_assoc), high_conf_indices)]
                valid = np.isfinite(emb_sub) & (emb_sub <= self.config.reid_threshold)
                if np.any(valid):
                    # Same fusion as the CPU BYTETracker:
                    # fuse_iou weights appearance similarity by IoU overlap,
                    # then blends with the IoU cost via lambda_emb
                    iou_sim = 1.0 - iou_sub
                    reid_sim = 1.0 - emb_sub
                    fused_emb_cost = 1.0 - reid_sim * (1.0 + iou_sim) / 2.0
                    lam = self.config.lambda_emb
                    iou_sub = np.where(
                        valid,
                        lam * fused_emb_cost + (1.0 - lam) * iou_sub,
                        iou_sub
                    )

            cost_sub = _fuse_score_np(iou_sub, det_scores[high_conf_indices])
            matches, u_tracks, u_dets = linear_assignment(cost_sub, self.config.match_thresh)
            matches = [(t, high_conf_indices[d]) for t, d in matches]
            u_dets = [high_conf_indices[d] for d in u_dets]
        else:
            matches = []
            u_tracks = list(range(n_assoc))
            u_dets = high_conf_indices

        # ---- Second association: remaining *tracked* tracks vs low-confidence
        # detections, on raw IoU (no score fusion), thresh 0.5 — this is the
        # "BYTE" step that recovers occluded/blurred objects
        r_track_indices = [i for i in u_tracks if i < n_tracked]

        if len(r_track_indices) > 0 and len(low_conf_indices) > 0:
            cost_sub = cost_matrix[np.ix_(r_track_indices, low_conf_indices)]
            matches2, _, _ = linear_assignment(cost_sub, 0.5)

            matched_rows = set()
            for i, d in matches2:
                track_idx = r_track_indices[i]
                matches.append((track_idx, low_conf_indices[d]))
                matched_rows.add(track_idx)
            if matched_rows:
                u_tracks = [i for i in u_tracks if i not in matched_rows]

        # ---- Third association: unconfirmed tracks vs remaining high-confidence
        # detections (score-fused cost, thresh 0.7)
        unconfirmed_rows = list(range(n_assoc, n_pool))
        if len(unconfirmed_rows) > 0 and len(u_dets) > 0:
            iou_sub = cost_matrix[np.ix_(unconfirmed_rows, u_dets)]
            cost_sub = _fuse_score_np(iou_sub, det_scores[u_dets])
            matches3, u_unconfirmed, u_dets_rel = linear_assignment(cost_sub, 0.7)

            for i, d in matches3:
                matches.append((unconfirmed_rows[i], u_dets[d]))
            u_tracks.extend(unconfirmed_rows[i] for i in u_unconfirmed)
            u_dets = [u_dets[d] for d in u_dets_rel]
        else:
            u_tracks.extend(unconfirmed_rows)

        return matches, u_tracks, u_dets

    def _update_stream_state(
        self,
        stream_id: int,
        strack_pool: List[GPUSTrack],
        matches: List[Tuple[int, int]],
        u_tracks: List[int],
        u_dets_high: List[int],
        det_tracks: List[GPUSTrack],
        kalman_updates: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]],
        new_track_states: Dict[int, Tuple[np.ndarray, np.ndarray]]
    ) -> List[Dict[str, Any]]:
        """
        Update track states after matching (CPU only).
        """
        state = self.stream_states[stream_id]
        state.frame_id += 1

        activated_tracks = []
        refind_tracks = []
        lost_tracks = []
        removed_tracks = []

        # Process matches
        for track_idx, det_idx in matches:
            track = strack_pool[track_idx]
            det = det_tracks[det_idx]

            key = (track_idx, det_idx)
            if key in kalman_updates:
                new_mean_np, new_cov_np = kalman_updates[key]
            else:
                new_mean_np = np.zeros(8, dtype=np.float32)
                new_mean_np[:4] = det.xyah
                new_cov_np = np.eye(8, dtype=np.float32) * 100

            if track.state == TrackState.Tracked:
                track.update(det, state.frame_id, new_mean_np, new_cov_np)
                activated_tracks.append(track)
            else:
                track.re_activate(det, state.frame_id, new_mean_np, new_cov_np)
                refind_tracks.append(track)

        # Handle unmatched tracks: unconfirmed ones are removed (they never
        # got a second detection), tracked ones become lost
        for track_idx in u_tracks:
            track = strack_pool[track_idx]
            if not track.is_activated:
                track.mark_removed()
                removed_tracks.append(track)
            elif track.state != TrackState.Lost:
                track.mark_lost()
                lost_tracks.append(track)

        # Initialize new tracks (unconfirmed until re-detected on a later
        # frame, except on the very first frame)
        for det_idx in u_dets_high:
            det = det_tracks[det_idx]
            if det.score >= self.det_thresh:
                if det_idx in new_track_states:
                    mean_np, cov_np = new_track_states[det_idx]
                else:
                    mean_np = np.zeros(8, dtype=np.float32)
                    mean_np[:4] = det.xyah
                    cov_np = np.eye(8, dtype=np.float32) * 100

                track_id = state.next_track_id()
                det.activate(state.frame_id, track_id, mean_np, cov_np)
                activated_tracks.append(det)

        # Remove old lost tracks
        for track in state.lost_stracks:
            if state.frame_id - track.end_frame > self.config.track_buffer:
                track.mark_removed()
                removed_tracks.append(track)

        # Update state lists
        state.tracked_stracks = [t for t in state.tracked_stracks if t.state == TrackState.Tracked]
        state.tracked_stracks = self._joint_stracks(state.tracked_stracks, activated_tracks)
        state.tracked_stracks = self._joint_stracks(state.tracked_stracks, refind_tracks)

        state.lost_stracks = self._sub_stracks(state.lost_stracks, state.tracked_stracks)
        state.lost_stracks.extend(lost_tracks)
        state.lost_stracks = self._sub_stracks(state.lost_stracks, removed_tracks)

        state.tracked_stracks, state.lost_stracks = self._remove_duplicate_stracks(
            state.tracked_stracks, state.lost_stracks
        )

        # Build output
        output_tracks = []
        for track in state.tracked_stracks:
            if track.is_activated:
                output_tracks.append({
                    'track_id': track.track_id,
                    'tlbr': track.tlbr,
                    'score': track.score,
                    'stream_id': stream_id,
                    'global_id': track.global_id
                })

        return output_tracks

    def _cross_camera_association(self, all_results: List[List[Dict[str, Any]]]):
        """
        Perform cross-camera Re-ID association.

        Updates global_id for each track based on gallery matching.
        """
        if self.global_gallery is None:
            return

        for stream_id, tracks in enumerate(all_results):
            state = self.stream_states[stream_id]

            for track_dict in tracks:
                track_id = track_dict['track_id']
                # Find corresponding GPUSTrack
                track = None
                for t in state.tracked_stracks:
                    if t.track_id == track_id:
                        track = t
                        break

                if track is None or track.smooth_embedding is None:
                    continue

                # Match or create global ID
                global_id = self.global_gallery.match_or_create(
                    track.smooth_embedding,
                    stream_id,
                    track_id,
                    state.frame_id
                )
                track.global_id = global_id
                track_dict['global_id'] = global_id

    @staticmethod
    def _joint_stracks(
        list_a: List[GPUSTrack],
        list_b: List[GPUSTrack]
    ) -> List[GPUSTrack]:
        """Merge two track lists, avoiding duplicates by track_id."""
        exists = {t.track_id for t in list_a}
        result = list_a.copy()
        for t in list_b:
            if t.track_id not in exists:
                exists.add(t.track_id)
                result.append(t)
        return result

    @staticmethod
    def _sub_stracks(
        list_a: List[GPUSTrack],
        list_b: List[GPUSTrack]
    ) -> List[GPUSTrack]:
        """Remove tracks in list_b from list_a."""
        ids_b = {t.track_id for t in list_b}
        return [t for t in list_a if t.track_id not in ids_b]

    @staticmethod
    def _remove_duplicate_stracks(
        list_a: List[GPUSTrack],
        list_b: List[GPUSTrack],
        iou_thresh: float = 0.15
    ) -> Tuple[List[GPUSTrack], List[GPUSTrack]]:
        """
        Remove duplicate tracks based on IoU.
        Keeps the track with longer tracklet.
        Uses vectorized numpy for fast IoU computation.
        """
        if len(list_a) == 0 or len(list_b) == 0:
            return list_a, list_b

        # Compute pairwise IoU (vectorized)
        boxes_a = np.array([t.tlbr for t in list_a])  # [N, 4]
        boxes_b = np.array([t.tlbr for t in list_b])  # [M, 4]

        # Vectorized IoU computation
        # boxes_a[:, None, :] -> [N, 1, 4], boxes_b[None, :, :] -> [1, M, 4]
        x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])  # [N, M]
        y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
        x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
        y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)  # [N, M]

        area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])  # [N]
        area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])  # [M]

        union = area_a[:, None] + area_b[None, :] - inter  # [N, M]
        ious = inter / (union + 1e-7)  # [N, M]

        # Find duplicates (vectorized)
        tracklet_lens_a = np.array([t.tracklet_len for t in list_a])  # [N]
        tracklet_lens_b = np.array([t.tracklet_len for t in list_b])  # [M]

        # Where IoU > threshold
        dup_mask = ious > iou_thresh  # [N, M]

        # For each duplicate pair, mark the shorter tracklet
        dup_a = set()
        dup_b = set()

        dup_indices = np.where(dup_mask)
        for i, j in zip(dup_indices[0], dup_indices[1]):
            if tracklet_lens_a[i] > tracklet_lens_b[j]:
                dup_b.add(j)
            else:
                dup_a.add(i)

        result_a = [t for i, t in enumerate(list_a) if i not in dup_a]
        result_b = [t for j, t in enumerate(list_b) if j not in dup_b]

        return result_a, result_b

    def reset(self):
        """Reset all stream states."""
        for state in self.stream_states:
            state.reset()
        if self.global_gallery is not None:
            self.global_gallery.reset()

    def __del__(self):
        """Cleanup thread pool."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


class ParallelCPUTracker:
    """
    Fallback tracker using parallel CPU processing only.

    Useful when GPU is not available or overhead is too high.
    Uses original BYTETracker with ThreadPoolExecutor for parallelism.
    """

    def __init__(self, num_streams: int, args):
        """
        Args:
            num_streams: Number of video streams
            args: Configuration namespace with track_thresh, track_buffer, etc.
        """
        self.num_streams = num_streams
        self.args = args

        # Import original BYTETracker
        from .byte_tracker import BYTETracker

        # Create one tracker per stream
        self.trackers = [BYTETracker(args) for _ in range(num_streams)]
        self.executor = ThreadPoolExecutor(max_workers=num_streams)

    def update(
        self,
        detections: List[np.ndarray],
        img_infos: List[Tuple[int, int]],
        img_sizes: List[Tuple[int, int]]
    ) -> List[List]:
        """
        Update all trackers in parallel.

        Args:
            detections: List of detection arrays per stream
            img_infos: List of (height, width) tuples per stream
            img_sizes: List of (height, width) tuples per stream (model input size)

        Returns:
            tracks: List of track lists per stream (original STrack format)
        """
        futures = []
        for i, (tracker, dets, info, size) in enumerate(
            zip(self.trackers, detections, img_infos, img_sizes)
        ):
            future = self.executor.submit(
                tracker.update, dets, info, size
            )
            futures.append(future)

        return [f.result() for f in futures]

    def reset(self):
        """Reset all trackers."""
        from .byte_tracker import BYTETracker
        self.trackers = [BYTETracker(self.args) for _ in range(self.num_streams)]

    def __del__(self):
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)

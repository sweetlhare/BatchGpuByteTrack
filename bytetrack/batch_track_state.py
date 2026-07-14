"""
Per-stream track state management for batched multi-stream tracking.

This module provides state containers for individual streams without global counters,
enabling independent track ID management across multiple video streams.
"""

import numpy as np
import torch
from typing import List, Dict, Optional, Tuple, Any
from enum import IntEnum


class TrackState(IntEnum):
    """Track lifecycle states."""
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class GPUSTrack:
    """
    Single track representation with GPU-compatible state storage.

    Attributes:
        stream_id: ID of the stream this track belongs to
        track_id: Unique ID within the stream
        mean: [8] Kalman state (GPU tensor or numpy)
        covariance: [8, 8] Kalman covariance (GPU tensor or numpy)
        tlwh: [4] Current bounding box [x, y, w, h]
        score: Detection confidence
        state: Current TrackState
        is_activated: Whether track is confirmed
        frame_id: Last update frame
        start_frame: First frame of track
        tracklet_len: Number of consecutive frames tracked
        embedding: Optional Re-ID feature vector
        global_id: Optional cross-camera global ID
    """

    def __init__(
        self,
        tlwh: np.ndarray,
        score: float,
        stream_id: int,
        embedding: Optional[np.ndarray] = None
    ):
        self.stream_id = stream_id
        self.track_id = 0  # Will be assigned by StreamState

        # Bounding box (numpy for compatibility)
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.score = score

        # Kalman state (will be initialized on activation)
        self.mean = None
        self.covariance = None

        # Track lifecycle
        self.state = TrackState.New
        self.is_activated = False
        self.frame_id = 0
        self.start_frame = 0
        self.tracklet_len = 0

        # Re-ID features
        self.embedding = embedding
        self.smooth_embedding = None
        self.global_id = None  # Cross-camera ID

    @property
    def tlwh(self) -> np.ndarray:
        """Get current bounding box in [x, y, w, h] format."""
        if self.mean is not None:
            # Convert from Kalman state [cx, cy, a, h] to [x, y, w, h]
            mean = self.mean
            if isinstance(mean, torch.Tensor):
                mean = mean.cpu().numpy()

            # ===== CRITICAL FIX: Defense against NaN/inf from GPU =====
            # If mean is corrupted, fall back to cached tlwh to prevent
            # NaN propagation into cost matrix
            if not np.all(np.isfinite(mean[:4])):
                return self._tlwh.copy()

            # Safe conversion with clamped values
            aspect_ratio = np.clip(mean[2], 0.1, 10.0)
            height = np.clip(mean[3], 1.0, 1e4)
            w = aspect_ratio * height

            return np.array([
                mean[0] - w / 2,
                mean[1] - height / 2,
                w,
                height
            ], dtype=np.float32)
        return self._tlwh.copy()

    @property
    def tlbr(self) -> np.ndarray:
        """Get current bounding box in [x1, y1, x2, y2] format."""
        tlwh = self.tlwh
        return np.array([
            tlwh[0],
            tlwh[1],
            tlwh[0] + tlwh[2],
            tlwh[1] + tlwh[3]
        ], dtype=np.float32)

    @property
    def xyah(self) -> np.ndarray:
        """Get current state in [cx, cy, a, h] format."""
        if self.mean is not None:
            mean = self.mean
            if isinstance(mean, torch.Tensor):
                mean = mean.cpu().numpy()
            return mean[:4].copy()
        tlwh = self._tlwh
        return np.array([
            tlwh[0] + tlwh[2] / 2,
            tlwh[1] + tlwh[3] / 2,
            tlwh[2] / (tlwh[3] + 1e-7),
            tlwh[3]
        ], dtype=np.float32)

    def activate(self, frame_id: int, track_id: int, mean: np.ndarray, covariance: np.ndarray):
        """
        Start a new tracklet with Kalman state.

        The track stays unconfirmed (is_activated=False) until it is matched
        again on a later frame, except on the very first frame — same behavior
        as the original ByteTrack. Unconfirmed tracks are not reported in the
        tracker output and are removed if not re-detected.

        Args:
            frame_id: Current frame number
            track_id: Assigned unique track ID
            mean: [8] Initial Kalman state
            covariance: [8, 8] Initial covariance
        """
        self.track_id = track_id
        self.mean = mean
        self.covariance = covariance
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = frame_id == 1
        if self.embedding is not None and self.smooth_embedding is None:
            self.smooth_embedding = self.embedding.copy()

    def re_activate(
        self,
        new_track: 'GPUSTrack',
        frame_id: int,
        new_mean: np.ndarray,
        new_covariance: np.ndarray,
        new_id: bool = False,
        new_track_id: Optional[int] = None
    ):
        """
        Re-activate a lost track with new detection.

        Args:
            new_track: Detection to use for update
            frame_id: Current frame number
            new_mean: Updated Kalman state
            new_covariance: Updated covariance
            new_id: Whether to assign new track ID
            new_track_id: New ID if new_id is True
        """
        self.mean = new_mean
        self.covariance = new_covariance
        self.frame_id = frame_id
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score

        if new_id and new_track_id is not None:
            self.track_id = new_track_id

        if new_track.embedding is not None:
            self.update_embedding(new_track.embedding)

    def update(
        self,
        new_track: 'GPUSTrack',
        frame_id: int,
        new_mean: np.ndarray,
        new_covariance: np.ndarray
    ):
        """
        Update track with new detection.

        Args:
            new_track: Detection to use for update
            frame_id: Current frame number
            new_mean: Updated Kalman state
            new_covariance: Updated covariance
        """
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.mean = new_mean
        self.covariance = new_covariance
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score

        if new_track.embedding is not None:
            self.update_embedding(new_track.embedding)

    def update_embedding(self, embedding: np.ndarray, alpha: float = 0.9):
        """
        Update smooth embedding with exponential moving average.

        Args:
            embedding: New embedding vector
            alpha: Smoothing factor (higher = more weight to old embedding)
        """
        self.embedding = embedding
        if self.smooth_embedding is None:
            self.smooth_embedding = embedding.copy()
        else:
            self.smooth_embedding = alpha * self.smooth_embedding + (1 - alpha) * embedding

    def mark_lost(self):
        """Mark track as lost."""
        self.state = TrackState.Lost

    def mark_removed(self):
        """Mark track as removed."""
        self.state = TrackState.Removed

    @property
    def end_frame(self) -> int:
        """Get last frame where track was seen."""
        return self.frame_id

    def __repr__(self) -> str:
        return f"GPUSTrack(stream={self.stream_id}, id={self.track_id}, state={self.state.name})"


class StreamState:
    """
    State container for a single video stream.

    Manages tracks, track ID counter, and frame counter for one stream.
    """

    def __init__(self, stream_id: int):
        self.stream_id = stream_id

        # Track lists
        self.tracked_stracks: List[GPUSTrack] = []
        self.lost_stracks: List[GPUSTrack] = []
        self.removed_stracks: List[GPUSTrack] = []

        # Counters
        self.frame_id = 0
        self._track_id_counter = 0

    def next_track_id(self) -> int:
        """Get next unique track ID for this stream."""
        self._track_id_counter += 1
        return self._track_id_counter

    def reset(self):
        """Reset all state."""
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        self._track_id_counter = 0

    def get_all_tracks(self) -> List[GPUSTrack]:
        """Get all active tracks (tracked + lost)."""
        return self.tracked_stracks + self.lost_stracks

    def get_track_states(self) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Get Kalman states for all tracked tracks.

        Returns:
            means: [N, 8] Kalman means
            covariances: [N, 8, 8] Kalman covariances
            track_indices: list of indices mapping to tracked_stracks
        """
        if not self.tracked_stracks:
            return np.empty((0, 8)), np.empty((0, 8, 8)), []

        means = []
        covs = []
        indices = []

        for i, track in enumerate(self.tracked_stracks):
            if track.mean is not None:
                mean = track.mean
                cov = track.covariance
                if isinstance(mean, torch.Tensor):
                    mean = mean.cpu().numpy()
                    cov = cov.cpu().numpy()
                means.append(mean)
                covs.append(cov)
                indices.append(i)

        if not means:
            return np.empty((0, 8)), np.empty((0, 8, 8)), []

        return np.stack(means), np.stack(covs), indices

    def get_track_boxes_tlbr(self) -> np.ndarray:
        """Get bounding boxes for tracked tracks in tlbr format."""
        if not self.tracked_stracks:
            return np.empty((0, 4))
        return np.array([t.tlbr for t in self.tracked_stracks])

    def get_track_embeddings(self) -> Optional[np.ndarray]:
        """Get embeddings for tracked tracks."""
        if not self.tracked_stracks:
            return None
        embs = [t.smooth_embedding for t in self.tracked_stracks if t.smooth_embedding is not None]
        if not embs:
            return None
        return np.stack(embs)

    def __repr__(self) -> str:
        return (f"StreamState(id={self.stream_id}, frame={self.frame_id}, "
                f"tracked={len(self.tracked_stracks)}, lost={len(self.lost_stracks)})")


class GlobalGallery:
    """
    Cross-camera Re-ID gallery for global track association.

    Maintains a gallery of embeddings for global ID assignment across streams.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.7,
        max_gallery_size: int = 10000,
        embedding_dim: int = 512
    ):
        self.similarity_threshold = similarity_threshold
        self.max_gallery_size = max_gallery_size
        self.embedding_dim = embedding_dim

        # Gallery storage
        self.gallery: Dict[int, Dict[str, Any]] = {}
        self._next_global_id = 0

    def match_or_create(
        self,
        embedding: np.ndarray,
        stream_id: int,
        local_track_id: int,
        frame_id: int
    ) -> int:
        """
        Match embedding to gallery or create new global ID.

        Args:
            embedding: [D] feature vector
            stream_id: Source stream ID
            local_track_id: Local track ID within stream
            frame_id: Current frame number

        Returns:
            global_id: Matched or newly created global ID
        """
        if len(self.gallery) == 0:
            return self._create_new(embedding, stream_id, local_track_id, frame_id)

        # Compute cosine similarities
        gallery_embs = np.stack([g['embedding'] for g in self.gallery.values()])
        embedding_norm = embedding / (np.linalg.norm(embedding) + 1e-7)
        gallery_norms = gallery_embs / (np.linalg.norm(gallery_embs, axis=1, keepdims=True) + 1e-7)

        similarities = embedding_norm @ gallery_norms.T

        best_idx = np.argmax(similarities)
        best_sim = similarities[best_idx]

        if best_sim > self.similarity_threshold:
            global_id = list(self.gallery.keys())[best_idx]
            # Update embedding with EMA
            self.gallery[global_id]['embedding'] = (
                0.9 * self.gallery[global_id]['embedding'] + 0.1 * embedding
            )
            self.gallery[global_id]['last_seen'] = frame_id
            self.gallery[global_id]['stream_id'] = stream_id
            self.gallery[global_id]['local_track_id'] = local_track_id
            return global_id
        else:
            return self._create_new(embedding, stream_id, local_track_id, frame_id)

    def _create_new(
        self,
        embedding: np.ndarray,
        stream_id: int,
        local_track_id: int,
        frame_id: int
    ) -> int:
        """Create new global ID entry."""
        global_id = self._next_global_id
        self._next_global_id += 1

        self.gallery[global_id] = {
            'embedding': embedding.copy(),
            'stream_id': stream_id,
            'local_track_id': local_track_id,
            'last_seen': frame_id
        }

        # Cleanup if gallery too large
        if len(self.gallery) > self.max_gallery_size:
            self._cleanup_old_entries(frame_id)

        return global_id

    def _cleanup_old_entries(self, current_frame: int, max_age: int = 1000):
        """Remove old gallery entries."""
        old_ids = [
            gid for gid, entry in self.gallery.items()
            if current_frame - entry['last_seen'] > max_age
        ]
        for gid in old_ids:
            del self.gallery[gid]

    def get_gallery_embeddings(self) -> Tuple[np.ndarray, List[int]]:
        """Get all gallery embeddings and their global IDs."""
        if not self.gallery:
            return np.empty((0, self.embedding_dim)), []
        global_ids = list(self.gallery.keys())
        embeddings = np.stack([self.gallery[gid]['embedding'] for gid in global_ids])
        return embeddings, global_ids

    def reset(self):
        """Clear gallery."""
        self.gallery = {}
        self._next_global_id = 0

    def __len__(self) -> int:
        return len(self.gallery)

    def __repr__(self) -> str:
        return f"GlobalGallery(size={len(self.gallery)}, next_id={self._next_global_id})"

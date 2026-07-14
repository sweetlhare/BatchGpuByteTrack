"""
GPU-accelerated matching functions for batched multi-stream tracking.

Uses torchvision for efficient IoU computation and PyTorch for distance matrices.
"""

import logging
import torch
import numpy as np
from typing import List, Tuple

logger = logging.getLogger(__name__)

try:
    from torchvision.ops import box_iou
except ImportError:
    box_iou = None
    logger.warning("torchvision not found, falling back to custom IoU implementation")


def _custom_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Custom IoU implementation when torchvision is not available.

    Args:
        boxes1: [N, 4] boxes in tlbr format
        boxes2: [M, 4] boxes in tlbr format

    Returns:
        ious: [N, M] IoU matrix
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2[None, :] - inter
    ious = inter / (union + 1e-7)

    return ious


def compute_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two sets of boxes on GPU.

    Args:
        boxes1: [N, 4] boxes in tlbr format (top-left, bottom-right)
        boxes2: [M, 4] boxes in tlbr format

    Returns:
        ious: [N, M] IoU matrix
    """
    if box_iou is not None:
        return box_iou(boxes1, boxes2)
    else:
        return _custom_box_iou(boxes1, boxes2)


def iou_distance(
    track_boxes: torch.Tensor,
    det_boxes: torch.Tensor
) -> torch.Tensor:
    """
    Compute IoU-based cost matrix (1 - IoU).

    Args:
        track_boxes: [N, 4] track bboxes in tlbr format
        det_boxes: [M, 4] detection bboxes in tlbr format

    Returns:
        cost_matrix: [N, M] cost matrix where cost = 1 - IoU
    """
    if track_boxes.shape[0] == 0 or det_boxes.shape[0] == 0:
        return torch.empty(
            track_boxes.shape[0], det_boxes.shape[0],
            device=track_boxes.device, dtype=torch.float32
        )

    ious = compute_box_iou(track_boxes, det_boxes)
    cost_matrix = 1.0 - ious
    return cost_matrix


def embedding_distance(
    track_embeddings: torch.Tensor,
    det_embeddings: torch.Tensor,
    metric: str = 'cosine'
) -> torch.Tensor:
    """
    Compute embedding-based distance matrix.

    Args:
        track_embeddings: [N, D] track feature vectors
        det_embeddings: [M, D] detection feature vectors
        metric: 'cosine' or 'euclidean'

    Returns:
        cost_matrix: [N, M] distance matrix
    """
    if track_embeddings.shape[0] == 0 or det_embeddings.shape[0] == 0:
        return torch.empty(
            track_embeddings.shape[0], det_embeddings.shape[0],
            device=track_embeddings.device, dtype=torch.float32
        )

    if metric == 'cosine':
        # Normalize embeddings
        track_norm = track_embeddings / (track_embeddings.norm(dim=1, keepdim=True) + 1e-7)
        det_norm = det_embeddings / (det_embeddings.norm(dim=1, keepdim=True) + 1e-7)

        # Cosine similarity -> distance
        similarity = track_norm @ det_norm.T
        cost_matrix = 1.0 - similarity
    elif metric == 'euclidean':
        cost_matrix = torch.cdist(track_embeddings, det_embeddings, p=2)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return cost_matrix


def fuse_score(
    cost_matrix: torch.Tensor,
    det_scores: torch.Tensor
) -> torch.Tensor:
    """
    Fuse cost matrix with detection scores (same as original ByteTrack).

    Args:
        cost_matrix: [N, M] IoU cost matrix (1 - IoU)
        det_scores: [M] detection confidence scores

    Returns:
        fused_cost: [N, M] score-fused cost matrix
    """
    if cost_matrix.numel() == 0:
        return cost_matrix

    iou_sim = 1.0 - cost_matrix  # Convert back to IoU
    fuse_cost = 1.0 - iou_sim * det_scores.unsqueeze(0)
    return fuse_cost


def tlwh_to_tlbr(tlwh: torch.Tensor) -> torch.Tensor:
    """
    Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2].

    Args:
        tlwh: [N, 4] boxes in top-left width-height format

    Returns:
        tlbr: [N, 4] boxes in top-left bottom-right format
    """
    tlbr = tlwh.clone()
    tlbr[:, 2] = tlwh[:, 0] + tlwh[:, 2]  # x2 = x + w
    tlbr[:, 3] = tlwh[:, 1] + tlwh[:, 3]  # y2 = y + h
    return tlbr


def tlwh_to_xyah(tlwh: torch.Tensor) -> torch.Tensor:
    """
    Convert bounding box format from [x, y, w, h] to [cx, cy, a, h].

    Args:
        tlwh: [N, 4] boxes in top-left width-height format

    Returns:
        xyah: [N, 4] boxes in center-x, center-y, aspect-ratio, height format
    """
    xyah = torch.empty_like(tlwh)
    xyah[:, 0] = tlwh[:, 0] + tlwh[:, 2] / 2  # cx = x + w/2
    xyah[:, 1] = tlwh[:, 1] + tlwh[:, 3] / 2  # cy = y + h/2

    # Prevent division by zero and extreme aspect ratios
    MIN_HEIGHT = 1e-2  # Minimum height to prevent division by near-zero
    MIN_ASPECT = 0.1   # Minimum aspect ratio (very tall boxes)
    MAX_ASPECT = 10.0  # Maximum aspect ratio (very wide boxes)

    h_safe = torch.clamp(tlwh[:, 3], min=MIN_HEIGHT)
    aspect_ratio = tlwh[:, 2] / h_safe
    xyah[:, 2] = torch.clamp(aspect_ratio, min=MIN_ASPECT, max=MAX_ASPECT)
    xyah[:, 3] = tlwh[:, 3]  # h
    return xyah


def xyah_to_tlwh(xyah: torch.Tensor) -> torch.Tensor:
    """
    Convert bounding box format from [cx, cy, a, h] to [x, y, w, h].

    Args:
        xyah: [N, 4] boxes in center-x, center-y, aspect-ratio, height format

    Returns:
        tlwh: [N, 4] boxes in top-left width-height format
    """
    tlwh = torch.empty_like(xyah)

    # Clamp width to prevent overflow in w = a * h
    MAX_WIDTH = 1e5  # Maximum width (prevents overflow)

    w = xyah[:, 2] * xyah[:, 3]  # w = a * h
    w = torch.clamp(w, max=MAX_WIDTH)  # Prevent overflow

    tlwh[:, 0] = xyah[:, 0] - w / 2  # x = cx - w/2
    tlwh[:, 1] = xyah[:, 1] - xyah[:, 3] / 2  # y = cy - h/2
    tlwh[:, 2] = w
    tlwh[:, 3] = xyah[:, 3]
    return tlwh


def xyah_to_tlbr(xyah: torch.Tensor) -> torch.Tensor:
    """
    Convert bounding box format from [cx, cy, a, h] to [x1, y1, x2, y2].

    Args:
        xyah: [N, 4] boxes in center-x, center-y, aspect-ratio, height format

    Returns:
        tlbr: [N, 4] boxes in top-left bottom-right format
    """
    tlwh = xyah_to_tlwh(xyah)
    return tlwh_to_tlbr(tlwh)


# Check library availability
try:
    import lap
    _LAP_AVAILABLE = True
except ImportError:
    _LAP_AVAILABLE = False
    try:
        from scipy.optimize import linear_sum_assignment
        _SCIPY_AVAILABLE = True
    except ImportError:
        _SCIPY_AVAILABLE = False


# Linear assignment using lap library
if _LAP_AVAILABLE:
    def linear_assignment(
        cost_matrix: np.ndarray,
        thresh: float
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Solve linear assignment problem using lap library.

        Args:
            cost_matrix: [N, M] cost matrix (numpy array on CPU)
            thresh: maximum cost threshold

        Returns:
            matches: list of (track_idx, det_idx) tuples
            unmatched_tracks: list of track indices
            unmatched_dets: list of detection indices
        """
        if cost_matrix.size == 0:
            return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

        # lap.lapjv expects float64
        cost_matrix = cost_matrix.astype(np.float64)

        # Extend cost matrix to square if needed
        cost_limit = thresh + 1e-5
        cost_matrix = np.where(cost_matrix > thresh, cost_limit, cost_matrix)

        _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=cost_limit)

        matches = []
        unmatched_tracks = []
        unmatched_dets = []

        for track_idx, det_idx in enumerate(x):
            if det_idx >= 0 and cost_matrix[track_idx, det_idx] < cost_limit:
                matches.append((track_idx, det_idx))
            else:
                unmatched_tracks.append(track_idx)

        matched_dets = {m[1] for m in matches}
        for det_idx, track_idx in enumerate(y):
            if track_idx < 0 or cost_matrix[track_idx, det_idx] >= cost_limit:
                if det_idx not in matched_dets:
                    unmatched_dets.append(det_idx)

        return matches, unmatched_tracks, unmatched_dets

elif _SCIPY_AVAILABLE:
    from scipy.optimize import linear_sum_assignment

    def linear_assignment(
        cost_matrix: np.ndarray,
        thresh: float
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Fallback using scipy when lap is not available.
        """
        if cost_matrix.size == 0:
            return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matches = []
        unmatched_tracks = list(range(cost_matrix.shape[0]))
        unmatched_dets = list(range(cost_matrix.shape[1]))

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] <= thresh:
                matches.append((r, c))
                unmatched_tracks.remove(r)
                unmatched_dets.remove(c)

        return matches, unmatched_tracks, unmatched_dets
else:
    raise ImportError("Neither lap nor scipy is available for linear assignment")

import numpy as np
import pytest
import torch

from bytetrack.matching import bbox_ious, fuse_score as fuse_score_np_cpu, linear_assignment as la_cpu
from bytetrack.gpu_matching import (
    compute_box_iou, _custom_box_iou, iou_distance, embedding_distance,
    fuse_score, linear_assignment, tlwh_to_tlbr, tlwh_to_xyah, xyah_to_tlbr
)


class TestIoU:
    def test_identical_boxes(self):
        a = np.array([[0, 0, 10, 10]], dtype=np.float64)
        assert bbox_ious(a, a)[0, 0] == pytest.approx(1.0, abs=1e-5)

    def test_disjoint_boxes(self):
        a = np.array([[0, 0, 10, 10]], dtype=np.float64)
        b = np.array([[20, 20, 30, 30]], dtype=np.float64)
        assert bbox_ious(a, b)[0, 0] == 0.0

    def test_half_overlap(self):
        a = np.array([[0, 0, 10, 10]], dtype=np.float64)
        b = np.array([[5, 0, 15, 10]], dtype=np.float64)
        # intersection 50, union 150
        assert bbox_ious(a, b)[0, 0] == pytest.approx(1 / 3, abs=1e-5)

    def test_gpu_matches_numpy(self):
        rng = np.random.default_rng(42)
        xy = rng.uniform(0, 500, size=(20, 2))
        wh = rng.uniform(10, 100, size=(20, 2))
        boxes_a = np.hstack([xy, xy + wh])
        xy = rng.uniform(0, 500, size=(15, 2))
        wh = rng.uniform(10, 100, size=(15, 2))
        boxes_b = np.hstack([xy, xy + wh])

        ious_np = bbox_ious(boxes_a, boxes_b)
        ious_t = compute_box_iou(
            torch.from_numpy(boxes_a).float(), torch.from_numpy(boxes_b).float()
        ).numpy()
        np.testing.assert_allclose(ious_np, ious_t, atol=1e-4)

    def test_custom_iou_matches_torchvision(self):
        a = torch.tensor([[0., 0., 10., 10.], [5., 5., 20., 20.]])
        b = torch.tensor([[0., 0., 10., 10.], [100., 100., 110., 110.]])
        np.testing.assert_allclose(
            _custom_box_iou(a, b).numpy(), compute_box_iou(a, b).numpy(), atol=1e-6
        )

    def test_iou_distance_empty(self):
        empty = torch.empty(0, 4)
        boxes = torch.tensor([[0., 0., 10., 10.]])
        assert iou_distance(empty, boxes).shape == (0, 1)
        assert iou_distance(boxes, empty).shape == (1, 0)


class TestEmbeddingDistance:
    def test_cosine(self):
        a = torch.tensor([[1., 0.], [0., 1.]])
        b = torch.tensor([[1., 0.]])
        d = embedding_distance(a, b, metric='cosine')
        assert d[0, 0] == pytest.approx(0.0, abs=1e-5)
        assert d[1, 0] == pytest.approx(1.0, abs=1e-5)

    def test_unknown_metric(self):
        a = torch.ones(1, 4)
        with pytest.raises(ValueError):
            embedding_distance(a, a, metric='hamming')


class TestFuseScore:
    def test_torch_fuse_score(self):
        cost = torch.tensor([[0.2, 0.5]])
        scores = torch.tensor([1.0, 0.5])
        fused = fuse_score(cost, scores)
        np.testing.assert_allclose(fused.numpy(), [[0.2, 0.75]], atol=1e-6)


class TestLinearAssignment:
    def test_simple_assignment(self):
        cost = np.array([[0.1, 0.9], [0.9, 0.2]])
        matches, u_tracks, u_dets = linear_assignment(cost, thresh=0.5)
        assert sorted(matches) == [(0, 0), (1, 1)]
        assert u_tracks == [] and u_dets == []

    def test_threshold_rejects(self):
        cost = np.array([[0.9]])
        matches, u_tracks, u_dets = linear_assignment(cost, thresh=0.5)
        assert matches == []
        assert u_tracks == [0] and u_dets == [0]

    def test_empty(self):
        matches, u_tracks, u_dets = linear_assignment(np.empty((0, 3)), 0.5)
        assert matches == [] and u_tracks == [] and u_dets == [0, 1, 2]

    def test_rectangular(self):
        cost = np.array([[0.1, 0.2, 0.3]])
        matches, u_tracks, u_dets = linear_assignment(cost, thresh=0.5)
        assert matches == [(0, 0)]
        assert sorted(u_dets) == [1, 2]


class TestConversions:
    def test_roundtrip(self):
        tlwh = torch.tensor([[10., 20., 30., 60.]])
        tlbr = tlwh_to_tlbr(tlwh)
        np.testing.assert_allclose(tlbr.numpy(), [[10., 20., 40., 80.]])
        xyah = tlwh_to_xyah(tlwh)
        np.testing.assert_allclose(xyah.numpy(), [[25., 50., 0.5, 60.]])
        back = xyah_to_tlbr(xyah)
        np.testing.assert_allclose(back.numpy(), tlbr.numpy(), atol=1e-4)

    def test_degenerate_height(self):
        tlwh = torch.tensor([[0., 0., 10., 0.]])
        xyah = tlwh_to_xyah(tlwh)
        assert torch.isfinite(xyah).all()

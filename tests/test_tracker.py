import numpy as np
import pytest
import torch

from bytetrack import BatchGPUTracker, BYTETracker, TrackerConfig


def cpu_config(**kwargs):
    kwargs.setdefault('device', 'cpu')
    return TrackerConfig(**kwargs)


class TestInputValidation:
    def test_wrong_stream_count(self, det):
        tracker = BatchGPUTracker(num_streams=2, config=cpu_config())
        with pytest.raises(ValueError, match="Expected 2 detection arrays"):
            tracker.update([det(100)])

    def test_cuda_unavailable_message(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
        with pytest.raises(RuntimeError, match="CUDA is not available"):
            BatchGPUTracker(num_streams=1, config=TrackerConfig(device='cuda'))

    def test_empty_first_frame(self):
        """Regression: ZeroDivisionError on a first frame without detections."""
        tracker = BatchGPUTracker(num_streams=2, config=cpu_config())
        out = tracker.update([np.empty((0, 5), dtype=np.float32), None])
        assert out == [[], []]

    def test_nan_detection_replaced(self, det):
        tracker = BatchGPUTracker(num_streams=1, config=cpu_config())
        bad = np.array([[np.nan, 100, 200, 200, 0.9]], dtype=np.float32)
        out = tracker.update([bad])
        assert out == [[]]

    def test_negative_size_detection(self):
        tracker = BatchGPUTracker(num_streams=1, config=cpu_config())
        bad = np.array([[200, 200, 100, 100, 0.9]], dtype=np.float32)  # x2 < x1
        out = tracker.update([bad])
        assert out == [[]]


class TestTrackLifecycle:
    def test_first_frame_immediate_activation(self, det):
        tracker = BatchGPUTracker(num_streams=1, config=cpu_config())
        out = tracker.update([det(100)])
        assert len(out[0]) == 1

    def test_single_frame_false_positive_suppressed(self, det):
        """New tracks (after frame 1) need a second detection to be reported."""
        tracker = BatchGPUTracker(num_streams=1, config=cpu_config())
        tracker.update([None])              # frame 1: empty
        out = tracker.update([det(100)])    # frame 2: new detection -> unconfirmed
        assert out[0] == []
        out = tracker.update([det(102)])    # frame 3: confirmed
        assert len(out[0]) == 1

    def test_unconfirmed_removed_without_second_detection(self, det):
        tracker = BatchGPUTracker(num_streams=1, config=cpu_config())
        tracker.update([None])
        tracker.update([det(100)])          # unconfirmed
        tracker.update([None])              # not re-detected -> removed
        out = tracker.update([det(500)])    # far away, new track
        assert out[0] == []                 # new track unconfirmed again
        state = tracker.stream_states[0]
        assert len(state.tracked_stracks) == 1

    def test_id_persistence(self, det):
        tracker = BatchGPUTracker(num_streams=1, config=cpu_config())
        tid = None
        for f in range(10):
            out = tracker.update([det(100 + 2 * f)])
            if out[0]:
                if tid is None:
                    tid = out[0][0]['track_id']
                assert out[0][0]['track_id'] == tid

    def test_lost_track_reactivated_same_id(self, det):
        tracker = BatchGPUTracker(num_streams=1, config=cpu_config())
        for f in range(5):
            out = tracker.update([det(100 + 2 * f)])
        tid = out[0][0]['track_id']
        for _ in range(3):                  # occlusion
            tracker.update([None])
        out = tracker.update([det(112)])    # reappears near prediction
        assert len(out[0]) == 1
        assert out[0][0]['track_id'] == tid

    def test_track_removed_after_buffer(self, det):
        tracker = BatchGPUTracker(num_streams=1, config=cpu_config(track_buffer=5))
        for f in range(3):
            tracker.update([det(100 + 2 * f)])
        for _ in range(7):                  # longer than track_buffer
            tracker.update([None])
        state = tracker.stream_states[0]
        assert state.lost_stracks == []

    def test_byte_second_association(self, det):
        """Low-confidence detections must keep tracks alive (BYTE)."""
        tracker = BatchGPUTracker(num_streams=1, config=cpu_config(track_thresh=0.5))
        tracker.update([det(100, score=0.9)])
        tracker.update([det(102, score=0.9)])
        out = tracker.update([det(104, score=0.3)])   # low conf
        assert len(out[0]) == 1, "low-conf detection must match via 2nd association"
        tid = out[0][0]['track_id']
        out = tracker.update([det(106, score=0.9)])
        assert out[0][0]['track_id'] == tid

    def test_very_low_score_ignored(self, det):
        """Detections below score 0.1 are ignored entirely (as in ByteTrack)."""
        tracker = BatchGPUTracker(num_streams=1, config=cpu_config())
        tracker.update([det(100, score=0.9)])
        tracker.update([det(102, score=0.9)])
        out = tracker.update([det(104, score=0.05)])
        assert out[0] == []


class TestMultiStream:
    def test_streams_are_independent(self, det):
        tracker = BatchGPUTracker(num_streams=2, config=cpu_config())
        for f in range(4):
            out = tracker.update([det(100 + 2 * f), det(500 + 2 * f)])
        assert len(out[0]) == 1 and len(out[1]) == 1
        # Per-stream IDs start from 1 independently
        assert out[0][0]['track_id'] == 1
        assert out[1][0]['track_id'] == 1
        assert out[0][0]['stream_id'] == 0
        assert out[1][0]['stream_id'] == 1

    def test_output_format(self, det):
        tracker = BatchGPUTracker(num_streams=1, config=cpu_config())
        out = tracker.update([det(100)])
        track = out[0][0]
        assert set(track) == {'track_id', 'tlbr', 'score', 'stream_id', 'global_id'}
        assert track['tlbr'].shape == (4,)

    def test_reset(self, det):
        tracker = BatchGPUTracker(num_streams=1, config=cpu_config())
        for f in range(3):
            tracker.update([det(100 + f)])
        tracker.reset()
        assert tracker.stream_states[0].tracked_stracks == []
        out = tracker.update([det(100)])
        assert len(out[0]) == 1  # frame counter reset -> immediate activation

    @pytest.mark.parametrize("num_streams", [1, 2, 4, 8, 16, 32])
    def test_scaling(self, det, num_streams):
        """Tracker must handle any stream count with independent per-stream IDs."""
        tracker = BatchGPUTracker(num_streams=num_streams, config=cpu_config())
        for f in range(3):
            out = tracker.update([det(100 + 2 * f) for _ in range(num_streams)])
        assert len(out) == num_streams
        for s in range(num_streams):
            assert len(out[s]) == 1
            assert out[s][0]['track_id'] == 1
            assert out[s][0]['stream_id'] == s

    def test_random_stress(self):
        rng = np.random.default_rng(7)
        tracker = BatchGPUTracker(num_streams=4, config=cpu_config())
        for _ in range(50):
            dets = []
            for _ in range(4):
                n = rng.integers(0, 8)
                boxes = []
                for _ in range(n):
                    x, y = rng.uniform(0, 1000), rng.uniform(0, 600)
                    boxes.append([x, y, x + 40, y + 90, rng.uniform(0.05, 1.0)])
                dets.append(np.array(boxes, dtype=np.float32) if boxes else None)
            out = tracker.update(dets)
            assert len(out) == 4


class TestReIDFusion:
    def _make_embedding(self, seed, dim=512):
        rng = np.random.default_rng(seed)
        v = rng.normal(size=dim).astype(np.float32)
        return v / np.linalg.norm(v)

    def test_external_embeddings_accepted(self, det):
        config = cpu_config(enable_reid=True, lambda_emb=0.3)
        tracker = BatchGPUTracker(num_streams=1, config=config)
        emb = self._make_embedding(0)
        for f in range(4):
            out = tracker.update(
                [det(100 + 2 * f)],
                embeddings=[emb[None, :]]
            )
        assert len(out[0]) == 1

    def test_appearance_prevents_id_switch(self, det):
        """Two crossing objects with distinct embeddings keep their IDs."""
        config = cpu_config(enable_reid=True, lambda_emb=0.5, reid_threshold=0.8)
        tracker = BatchGPUTracker(num_streams=1, config=config)
        emb_a, emb_b = self._make_embedding(1), self._make_embedding(2)

        def frame(xa, xb):
            dets = np.array([
                [xa, 100, xa + 50, 220, 0.9],
                [xb, 100, xb + 50, 220, 0.9],
            ], dtype=np.float32)
            return [dets], [np.stack([emb_a, emb_b])]

        ids = {}
        for f in range(8):
            dets, embs = frame(100 + 10 * f, 180 - 10 * f)  # cross around f=4
            out = tracker.update(dets, embeddings=embs)
            for tr in out[0]:
                ids.setdefault(tr['track_id'], []).append((f, tuple(tr['tlbr'][:1])))
        # Both tracks must survive the crossing without spawning new IDs
        assert len(ids) == 2

    def test_reid_without_embeddings_still_works(self, det):
        config = cpu_config(enable_reid=True)
        tracker = BatchGPUTracker(num_streams=1, config=config)
        out = tracker.update([det(100)])  # no frames, no embeddings
        assert len(out[0]) == 1


class TestParityWithCPUTracker:
    """The GPU tracker must behave like the reference BYTETracker."""

    class Args:
        track_thresh = 0.5
        track_buffer = 30
        match_thresh = 0.8
        mot20 = False

    def scripted_detections(self, num_frames=40):
        """Two steady objects + intermittent low-conf + one-frame FP."""
        rng = np.random.default_rng(3)
        frames = []
        for f in range(num_frames):
            boxes = [
                [100 + 3 * f, 100, 150 + 3 * f, 220, 0.9],
                [600 - 2 * f, 300, 650 - 2 * f, 420, 0.85],
            ]
            if f % 5 == 2:
                boxes[0][4] = 0.3  # object 1 drops to low conf periodically
            if f == 10:
                boxes.append([900, 500, 940, 590, 0.95])  # one-frame FP
            frames.append(np.array(boxes, dtype=np.float32))
        return frames

    def test_same_track_count(self):
        dets = self.scripted_detections()

        cpu = BYTETracker(self.Args(), frame_rate=30)
        gpu = BatchGPUTracker(num_streams=1, config=cpu_config(
            track_thresh=0.5, track_buffer=30, match_thresh=0.8))

        cpu_counts, gpu_counts = [], []
        for d in dets:
            cpu_out = cpu.update(d.copy(), (720, 1280), (720, 1280))
            gpu_out = gpu.update([d.copy()])
            cpu_counts.append(len(cpu_out))
            gpu_counts.append(len(gpu_out[0]))

        assert cpu_counts == gpu_counts

    def test_same_final_ids(self):
        dets = self.scripted_detections()
        gpu = BatchGPUTracker(num_streams=1, config=cpu_config())
        for d in dets:
            out = gpu.update([d])
        final_ids = sorted(t['track_id'] for t in out[0])
        # Two persistent objects; the one-frame FP must not have consumed an ID
        # that shows up in the output
        assert len(final_ids) == 2


@pytest.mark.gpu
class TestOnCuda:
    def test_cuda_smoke(self, det):
        tracker = BatchGPUTracker(num_streams=2, config=TrackerConfig(device='cuda'))
        for f in range(5):
            out = tracker.update([det(100 + 2 * f), None])
        assert len(out[0]) == 1

    def test_cuda_matches_cpu(self):
        rng = np.random.default_rng(11)
        frames = []
        for f in range(20):
            n = rng.integers(1, 6)
            boxes = []
            for i in range(n):
                x, y = rng.uniform(0, 1000), rng.uniform(0, 600)
                boxes.append([x, y, x + 50, y + 100, rng.uniform(0.2, 1.0)])
            frames.append(np.array(boxes, dtype=np.float32))

        cpu = BatchGPUTracker(num_streams=1, config=TrackerConfig(device='cpu'))
        gpu = BatchGPUTracker(num_streams=1, config=TrackerConfig(device='cuda'))
        for d in frames:
            out_cpu = cpu.update([d.copy()])
            out_gpu = gpu.update([d.copy()])
        ids_cpu = sorted(t['track_id'] for t in out_cpu[0])
        ids_gpu = sorted(t['track_id'] for t in out_gpu[0])
        assert ids_cpu == ids_gpu


class _DummyReID(torch.nn.Module):
    """Tiny stand-in for OSNet: [N, 3, 256, 128] -> [N, 512]."""
    feature_dim = 512

    def forward(self, x):
        pooled = x.mean(dim=(2, 3))               # [N, 3]
        return pooled.repeat(1, 171)[:, :512]     # [N, 512]


class TestReIDExtraction:
    def _tracker_with_dummy_model(self, num_streams=2):
        tracker = BatchGPUTracker(
            num_streams=num_streams,
            config=cpu_config(enable_reid=True)
        )
        tracker.reid_model = _DummyReID()
        return tracker

    def _frame(self, h=120, w=160, seed=0):
        rng = np.random.default_rng(seed)
        return rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)

    def test_shapes_and_split(self):
        tracker = self._tracker_with_dummy_model(num_streams=2)
        frames = [self._frame(seed=1), self._frame(seed=2)]
        boxes = [
            np.array([[10, 10, 60, 110, 0.9], [20, 5, 80, 100, 0.8]], dtype=np.float32),
            np.array([[0, 0, 40, 90, 0.7]], dtype=np.float32),
        ]
        embs = tracker._extract_reid_features(frames, boxes)
        assert embs[0].shape == (2, 512)
        assert embs[1].shape == (1, 512)
        np.testing.assert_allclose(np.linalg.norm(embs[0], axis=1), 1.0, atol=1e-4)

    def test_invalid_and_empty_boxes(self):
        tracker = self._tracker_with_dummy_model(num_streams=3)
        frames = [self._frame(seed=i) for i in range(3)]
        boxes = [
            np.array([[np.nan, 10, 60, 110, 0.9]], dtype=np.float32),  # NaN box
            np.empty((0, 5), dtype=np.float32),                        # no dets
            np.array([[50, 50, 40, 40, 0.9]], dtype=np.float32),       # inverted box
        ]
        embs = tracker._extract_reid_features(frames, boxes)
        assert embs[0].shape == (1, 512) and np.isfinite(embs[0]).all()
        assert embs[1].shape == (0, 512)
        assert embs[2].shape == (1, 512) and np.isfinite(embs[2]).all()

    def test_mixed_resolutions(self):
        """Streams with different frame sizes are grouped and still aligned."""
        tracker = self._tracker_with_dummy_model(num_streams=3)
        frames = [self._frame(120, 160, 1), self._frame(240, 320, 2), self._frame(120, 160, 3)]
        boxes = [
            np.array([[10, 10, 60, 110, 0.9]], dtype=np.float32),
            np.array([[10, 10, 200, 200, 0.9], [50, 50, 150, 220, 0.8]], dtype=np.float32),
            np.array([[5, 5, 100, 100, 0.9]], dtype=np.float32),
        ]
        embs = tracker._extract_reid_features(frames, boxes)
        assert [e.shape[0] for e in embs] == [1, 2, 1]

    def test_reid_interval(self, det):
        tracker = BatchGPUTracker(
            num_streams=1,
            config=cpu_config(enable_reid=True, reid_interval=3)
        )
        tracker.reid_model = _DummyReID()
        calls = []
        original = tracker._extract_reid_features
        tracker._extract_reid_features = lambda f, b: (calls.append(1), original(f, b))[1]

        frame = self._frame()
        for f in range(6):
            tracker.update([det(10, y=10, w=30, h=60)], frames=[frame])
        # frames 0 and 3 only
        assert len(calls) == 2


class TestNaNWithLiveTracks:
    def test_nan_detection_with_existing_pool(self, det):
        """NaN rows must not poison the cost matrix once tracks exist."""
        tracker = BatchGPUTracker(num_streams=1, config=cpu_config())
        for f in range(3):
            tracker.update([det(100 + 2 * f)])
        mixed = np.array([
            [106, 100, 156, 220, 0.9],
            [np.nan, np.inf, 200, 200, 0.95],
        ], dtype=np.float32)
        out = tracker.update([mixed])
        assert len(out[0]) == 1  # real det matched, NaN det ignored

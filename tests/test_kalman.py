import numpy as np
import pytest
import torch

from bytetrack.kalman_filter import KalmanFilter
from bytetrack.gpu_kalman_filter import GPUKalmanFilter


@pytest.fixture
def gpu_kf():
    return GPUKalmanFilter(device='cpu')


@pytest.fixture
def cpu_kf():
    return KalmanFilter()


MEASUREMENT = np.array([320.0, 240.0, 0.5, 120.0])  # cx, cy, a, h


class TestAgainstReference:
    """GPU Kalman filter must match the original numpy implementation."""

    def test_initiate(self, gpu_kf, cpu_kf):
        mean_ref, cov_ref = cpu_kf.initiate(MEASUREMENT)
        mean_t, cov_t = gpu_kf.initiate(torch.from_numpy(MEASUREMENT).float())
        np.testing.assert_allclose(mean_t.numpy(), mean_ref, atol=1e-4)
        np.testing.assert_allclose(cov_t.numpy(), cov_ref, atol=1e-4)

    def test_predict(self, gpu_kf, cpu_kf):
        mean_ref, cov_ref = cpu_kf.initiate(MEASUREMENT)
        mean_ref, cov_ref = cpu_kf.predict(mean_ref, cov_ref)

        mean_t, cov_t = gpu_kf.initiate(torch.from_numpy(MEASUREMENT).float())
        mean_t, cov_t = gpu_kf.predict(mean_t, cov_t)

        np.testing.assert_allclose(mean_t.numpy(), mean_ref, atol=1e-3)
        np.testing.assert_allclose(cov_t.numpy(), cov_ref, atol=1e-3)

    def test_update(self, gpu_kf, cpu_kf):
        obs = MEASUREMENT + np.array([5.0, -3.0, 0.01, 2.0])

        mean_ref, cov_ref = cpu_kf.initiate(MEASUREMENT)
        mean_ref, cov_ref = cpu_kf.predict(mean_ref, cov_ref)
        mean_ref, cov_ref = cpu_kf.update(mean_ref, cov_ref, obs)

        mean_t, cov_t = gpu_kf.initiate(torch.from_numpy(MEASUREMENT).float())
        mean_t, cov_t = gpu_kf.predict(mean_t, cov_t)
        mean_t, cov_t = gpu_kf.update(mean_t, cov_t, torch.from_numpy(obs).float())

        np.testing.assert_allclose(mean_t.numpy(), mean_ref, atol=1e-2)
        np.testing.assert_allclose(cov_t.numpy(), cov_ref, atol=1e-2)

    def test_multi_frame_tracking(self, gpu_kf, cpu_kf):
        """Track a constant-velocity target for 20 frames; states must agree."""
        mean_ref, cov_ref = cpu_kf.initiate(MEASUREMENT)
        mean_t, cov_t = gpu_kf.initiate(torch.from_numpy(MEASUREMENT).float())

        for frame in range(1, 21):
            obs = MEASUREMENT + np.array([3.0 * frame, 1.5 * frame, 0.0, 0.0])
            mean_ref, cov_ref = cpu_kf.predict(mean_ref, cov_ref)
            mean_ref, cov_ref = cpu_kf.update(mean_ref, cov_ref, obs)
            mean_t, cov_t = gpu_kf.predict(mean_t, cov_t)
            mean_t, cov_t = gpu_kf.update(mean_t, cov_t, torch.from_numpy(obs).float())

        np.testing.assert_allclose(mean_t.numpy(), mean_ref, rtol=1e-3, atol=1e-2)


class TestBatchConsistency:
    """Batched ops must match single-track ops."""

    def test_batch_initiate(self, gpu_kf):
        meas = torch.tensor([
            [320.0, 240.0, 0.5, 120.0],
            [100.0, 50.0, 0.8, 60.0],
        ])
        means_b, covs_b = gpu_kf.batch_initiate(meas)
        for i in range(2):
            mean_s, cov_s = gpu_kf.initiate(meas[i])
            np.testing.assert_allclose(means_b[i].numpy(), mean_s.numpy(), atol=1e-5)
            np.testing.assert_allclose(covs_b[i].numpy(), cov_s.numpy(), atol=1e-5)

    def test_batch_predict(self, gpu_kf):
        meas = torch.tensor([
            [320.0, 240.0, 0.5, 120.0],
            [100.0, 50.0, 0.8, 60.0],
        ])
        means, covs = gpu_kf.batch_initiate(meas)
        means_b, covs_b = gpu_kf.batch_predict(means.clone(), covs.clone())
        for i in range(2):
            mean_s, cov_s = gpu_kf.predict(means[i], covs[i])
            np.testing.assert_allclose(means_b[i].numpy(), mean_s.numpy(), atol=1e-4)
            np.testing.assert_allclose(covs_b[i].numpy(), cov_s.numpy(), atol=1e-4)

    def test_batch_update(self, gpu_kf):
        meas = torch.tensor([
            [320.0, 240.0, 0.5, 120.0],
            [100.0, 50.0, 0.8, 60.0],
        ])
        means, covs = gpu_kf.batch_initiate(meas)
        means, covs = gpu_kf.batch_predict(means, covs)
        obs = meas + torch.tensor([[2.0, 1.0, 0.0, 0.5], [-1.0, 3.0, 0.0, -0.5]])
        means_b, covs_b = gpu_kf.batch_update(means.clone(), covs.clone(), obs)
        for i in range(2):
            mean_s, cov_s = gpu_kf.update(means[i], covs[i], obs[i])
            np.testing.assert_allclose(means_b[i].numpy(), mean_s.numpy(), atol=1e-3)
            np.testing.assert_allclose(covs_b[i].numpy(), cov_s.numpy(), atol=1e-3)

    def test_empty_batch(self, gpu_kf):
        means, covs = gpu_kf.batch_initiate(torch.empty(0, 4))
        assert means.shape == (0, 8)
        means, covs = gpu_kf.batch_predict(means, covs)
        assert means.shape == (0, 8)


class TestNumericalStability:
    def test_long_lost_track_no_overflow(self, gpu_kf):
        """Predicting many frames without updates must not overflow."""
        means, covs = gpu_kf.batch_initiate(torch.tensor([[320.0, 240.0, 0.5, 120.0]]))
        for _ in range(500):
            means, covs = gpu_kf.batch_predict(means, covs)
        assert torch.isfinite(means).all()
        assert torch.isfinite(covs).all()

    def test_gating_distance(self, gpu_kf, cpu_kf):
        mean_ref, cov_ref = cpu_kf.initiate(MEASUREMENT)
        obs = np.stack([MEASUREMENT, MEASUREMENT + [50, 50, 0, 0]])
        d_ref = cpu_kf.gating_distance(mean_ref, cov_ref, obs, metric='maha')

        mean_t, cov_t = gpu_kf.initiate(torch.from_numpy(MEASUREMENT).float())
        d_t = gpu_kf.gating_distance(mean_t, cov_t, torch.from_numpy(obs).float())
        np.testing.assert_allclose(d_t.numpy(), d_ref, rtol=1e-3, atol=1e-3)
